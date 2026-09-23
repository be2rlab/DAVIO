import numpy as np
from . import sim3
from .graph import Graph, PointFactor


def pose_information(edge, nodes, sigma, sigma_radial, sigma_lateral):
    a, b = edge['a'], edge['b']
    graph = Graph()
    graph.add_node(a, nodes[a]['pose'], dimensions=5)
    graph.add_node(b, nodes[b]['pose'], dimensions=5)
    factor = PointFactor(a, b, edge['points_a'], edge['points_b'], sigma=sigma,
                         sigma_radial=sigma_radial, sigma_lateral=sigma_lateral)
    _cost, _residual, blocks = factor.linearize(graph.nodes)
    jb = blocks[b] @ graph.chart(b, graph.nodes[b])
    ja = blocks[a] @ graph.chart(a, graph.nodes[a])
    jp, js = jb[:, :4], np.column_stack((ja[:, 4], jb[:, 4]))
    hpp, hps, hss = jp.T @ jp, jp.T @ js, js.T @ js
    schur = hpp - hps @ np.linalg.pinv(hss, rcond=1e-6) @ hps.T
    # With both depth scales free, no point group can see the global scale gauge:
    # enlarging both submaps while moving b out along the a->b baseline leaves every
    # residual unchanged. That direction is pinned by the metric VIO baselines, not by
    # geometry, so it is removed analytically rather than credited to a prior.
    baseline = sim3.pose(nodes[b]['pose'])[:3, 3] - sim3.pose(nodes[a]['pose'])[:3, 3]
    gauge = np.r_[baseline / max(np.linalg.norm(baseline), 1e-12), 0.]
    projector = np.eye(4) - np.outer(gauge, gauge)
    eig = np.sort(np.linalg.eigvalsh(projector @ schur @ projector))[1:]   # drop the gauge
    lam = float(max(eig[0], 0.))
    return dict(lambda_min=lam, condition=float(eig[-1] / lam) if lam > 0 else float('inf'),
                gauge_direction='translation along the a->b baseline, removed')


def held_out_residual(edge, nodes, sigma, sigma_radial, sigma_lateral):
    pair = edge.get('held_out')
    if pair is None or not len(pair[0]):
        return None
    pa, pb = np.asarray(pair[0], float), np.asarray(pair[1], float)
    ta = nodes[edge['a']]['pose']
    tb = np.asarray(sim3.pose(ta) @ edge['measurement'], float)
    tb[:3, :3] *= sim3.scale(nodes[edge['b']]['pose'])
    factor = PointFactor(edge['a'], edge['b'], pa, pb, sigma=sigma, sigma_radial=sigma_radial,
                         sigma_lateral=sigma_lateral)
    covariance = factor._covariance(pa, ta) + factor._covariance(pb, tb)
    whitener = np.linalg.inv(np.linalg.cholesky(covariance))
    difference = pa @ ta[:3, :3].T + ta[:3, 3] - pb @ tb[:3, :3].T - tb[:3, 3]
    return float(np.median(np.linalg.norm(np.einsum('kij,kj->ki', whitener, difference), axis=1)))


def drift_compatible(edge, nodes, cfg):
    """Innovation against the declared VIO drift model alone (no loop-sigma inflation)."""
    k = float(cfg['admission_drift_sigma'])
    arc = abs(nodes[edge['b']]['path_m'] - nodes[edge['a']]['path_m'])
    info = edge['info']
    limit_m = k * max(cfg['odometry_sigma_floor_m'], cfg['odometry_drift_m_per_m'] * arc)
    limit_deg = k * max(cfg['odometry_sigma_floor_deg'], cfg['odometry_drift_deg_per_m'] * arc)
    return bool(info['innovation_m'] <= limit_m and info['innovation_deg'] <= limit_deg), \
        dict(arc_m=float(arc), limit_m=float(limit_m), limit_deg=float(limit_deg))


def implied_correction(edge, nodes):
    """The odom->map transform this group's rigid hypothesis implies for node b."""
    return sim3.pose(nodes[edge['a']]['pose']) @ edge['measurement'] @ np.linalg.inv(nodes[edge['b']]['odom'])


def confirmed_by(edge, groups, nodes, tol_m, tol_deg):
    keys = list(nodes)
    ia = keys.index(edge['a'])
    neighbours = set(keys[max(0, ia - 1):ia + 2])
    own = implied_correction(edge, nodes)
    for other in groups:
        if other is edge or other['b'] <= edge['b'] or other['a'] not in neighbours:
            continue
        delta = np.linalg.inv(own) @ implied_correction(other, nodes)
        angle = np.degrees(np.linalg.norm(sim3.log(delta)[3:6]))
        if np.linalg.norm(delta[:3, 3]) <= tol_m and angle <= tol_deg:
            return other['b']
    return None


def decide(edge, groups, nodes, cfg):
    """One decision for a currently deferred loop group; returns (state, reason, checks)."""
    sig = (cfg.get('sparse_sigma_m', .10), cfg.get('sparse_sigma_radial', 0.), cfg.get('sparse_sigma_lateral', 0.))
    ok, drift = drift_compatible(edge, nodes, cfg)
    checks = dict(drift=drift)
    if not ok:
        return 'rejected', 'drift model', checks
    held = held_out_residual(edge, nodes, *sig)
    checks['held_out_median_whitened'] = held
    if held is not None and held > float(cfg['admission_heldout_max']):
        return 'rejected', 'held-out residual', checks
    info = pose_information(edge, nodes, *sig)
    checks.update(info)
    if info['lambda_min'] < float(cfg['admission_min_information']) or \
            info['condition'] > float(cfg['admission_max_condition']):
        return 'deferred', 'conditioning', checks
    witness = confirmed_by(edge, groups, nodes, float(cfg['admission_confirm_m']),
                           float(cfg['admission_confirm_deg']))
    checks['confirmed_by'] = witness
    if witness is None:
        return 'deferred', 'unconfirmed', checks
    return 'admitted', f'confirmed by {witness}', checks
