from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from . import sim3

UP = np.array([0., 0., 1.])     # the vertical of the filter's world frame


def robust(residual, blocks, sigmas, delta):
    """Whitened residual under a Huber loss: its cost, and the residual and Jacobians reweighted."""
    whitening = 1. / np.broadcast_to(np.asarray(sigmas, float), residual.shape)
    residual = whitening * residual
    length = np.linalg.norm(residual)
    if not np.isfinite(length):
        raise ValueError('Non-finite factor residual')
    cost = .5 * length ** 2 if length <= delta else delta * (length - .5 * delta)
    weight = np.sqrt(min(1., delta / max(length, 1e-30)))
    return cost, weight * residual, {k: weight * whitening[:, None] * j for k, j in blocks.items()}


@dataclass
class PoseFactor:
    """The pose of node b seen from node a, their depth scales set aside."""
    a: str
    b: str
    measurement: np.ndarray
    sigmas: np.ndarray              # translation (3), rotation (3)
    huber_delta: float = 2.
    _inverse: tuple = field(default=None, init=False, repr=False)

    def linearize(self, nodes, jacobians=True):
        a, b = nodes[self.a], nodes[self.b]
        if self._inverse is None:
            z_inv = sim3.inverse(self.measurement)
            self._inverse = z_inv, sim3.adjoint(z_inv)
        z_inv, adjoint = self._inverse
        xi = sim3.log(z_inv @ sim3.inverse(sim3.pose(a)) @ sim3.pose(b))
        blocks = {}
        if jacobians:
            left, right = sim3.log_jacobians(xi)
            blocks = {self.a: (-left @ adjoint @ sim3.projection_jacobian(a))[:6],
                      self.b: (right @ sim3.projection_jacobian(b))[:6]}
        return robust(xi[:6], blocks, self.sigmas, self.huber_delta)


@dataclass
class ScaleFactor:
    """log scale(b) - log scale(a) = log_ratio."""
    a: str
    b: str
    log_ratio: float
    sigma: float
    huber_delta: float = 2.

    def linearize(self, nodes, jacobians=True):
        residual = np.array([np.log(sim3.scale(nodes[self.b])) - np.log(sim3.scale(nodes[self.a])) - self.log_ratio])
        row = np.zeros((1, 7))
        row[0, 6] = 1.
        return robust(residual, {self.a: -row, self.b: row} if jacobians else {}, np.array([self.sigma]), self.huber_delta)


@dataclass
class PointFactor:
    a: str
    b: str
    points_a: np.ndarray
    points_b: np.ndarray
    sigma: float
    sigma_radial: float
    sigma_lateral: float
    huber_delta: float = 2.
    _whitener: np.ndarray = field(default=None, init=False, repr=False)

    def covariance(self, points, transform):
        depth = np.maximum(np.linalg.norm(points, axis=1), 1e-9)
        radial = np.hypot(self.sigma, self.sigma_radial * depth) ** 2
        lateral = np.hypot(self.sigma, self.sigma_lateral * depth) ** 2
        rays = points / depth[:, None]
        local = (lateral[:, None, None] * np.eye(3)
                 + (radial - lateral)[:, None, None] * rays[:, :, None] * rays[:, None, :])
        return np.einsum('ij,kjl,ml->kim', transform[:3, :3], local, transform[:3, :3])

    def linearize(self, nodes, jacobians=True):
        pa, pb = np.asarray(self.points_a), np.asarray(self.points_b)
        if pa.shape != pb.shape or pa.ndim != 2 or pa.shape[1] != 3 or not len(pa):
            raise ValueError('Correspondences must be nonempty paired Nx3 points')
        if not np.isfinite(pa).all() or not np.isfinite(pb).all():
            raise ValueError('Non-finite correspondences')
        a, b = nodes[self.a], nodes[self.b]
        difference = pa @ a[:3, :3].T + a[:3, 3] - pb @ b[:3, :3].T - b[:3, 3]
        if self._whitener is None:      # the covariance is taken once, where the solve starts
            self._whitener = np.linalg.inv(np.linalg.cholesky(self.covariance(pa, a) + self.covariance(pb, b)))
        residual = np.einsum('kij,kj->ki', self._whitener, difference)
        lengths = np.linalg.norm(residual, axis=1)
        d = self.huber_delta
        costs = np.where(lengths <= d, .5 * lengths ** 2, d * (lengths - .5 * d))
        weights = np.sqrt(np.minimum(1., d / np.maximum(lengths, 1e-30)) / len(pa))
        blocks = {}
        if jacobians:
            def block(points, transform):
                x, y, z = points.T
                o = np.zeros(len(points))
                skews = np.stack((np.stack((o, -z, y), 1), np.stack((z, o, -x), 1), np.stack((-y, x, o), 1)), 1)
                out = np.zeros((len(points), 3, 7))
                out[:, :, :3] = transform[:3, :3]
                out[:, :, 3:6] = -transform[:3, :3] @ skews
                out[:, :, 6] = points @ transform[:3, :3].T
                out = np.einsum('kij,kjl->kil', self._whitener, out)
                return (out * weights[:, None, None]).reshape(-1, 7)
            blocks = {self.a: block(pa, a), self.b: -block(pb, b)}
        return float(costs.mean()), (residual * weights[:, None]).ravel(), blocks


class Graph:
    def __init__(self):
        self.nodes, self.free, self.factors = {}, {}, []

    def add_node(self, key, transform, free=True):
        if key in self.nodes:
            raise ValueError('Duplicate node')
        self.nodes[key] = sim3.validate(transform)
        self.free[key] = free

    def add(self, factor):
        if factor.a == factor.b or factor.a not in self.nodes or factor.b not in self.nodes:
            raise ValueError('A factor needs two distinct existing nodes')
        self.factors.append(factor)

    @staticmethod
    def chart(transform):
        """7 x 5: a step (translation, yaw, log scale) as an increment of the similarity on its right."""
        s = sim3.scale(transform)
        r = transform[:3, :3] / s
        m = np.zeros((7, 5))
        m[:3, :3] = r.T / s
        m[3:6, 3] = r.T @ UP
        m[6, 4] = 1.
        return m

    @staticmethod
    def retract(transform, step):
        out = transform.copy()
        yaw = sim3.exp(np.r_[0., 0., 0., UP * step[3], 0.])[:3, :3]
        out[:3, :3] = np.exp(step[4]) * yaw @ transform[:3, :3]
        out[:3, 3] = transform[:3, 3] + step[:3]
        return out

    def check_connected(self):
        """Every node must reach the fixed one through factors that constrain poses."""
        fixed = [k for k, free in self.free.items() if not free]
        if not fixed:
            raise ValueError('Fix a reference node to remove the global gauge')
        neighbours = {k: set() for k in self.nodes}
        for f in self.factors:
            if not isinstance(f, ScaleFactor):
                neighbours[f.a].add(f.b)
                neighbours[f.b].add(f.a)
        reached, stack = set(fixed), list(fixed)
        while stack:
            for other in neighbours[stack.pop()] - reached:
                reached.add(other)
                stack.append(other)
        if reached != set(self.nodes):
            raise ValueError(f'Nodes disconnected from the gauge: {sorted(set(self.nodes) - reached)}')

    def system(self, nodes, jacobians):
        offsets, n = {}, 0
        for key, free in self.free.items():
            if free:
                offsets[key], n = n, n + 5
        rows, columns, data, errors, charts = [], [], [], [], {}
        cost, row = 0., 0
        for factor in self.factors:
            value, residual, blocks = factor.linearize(nodes, jacobians)
            cost += value
            errors.append(residual)
            for key, block in blocks.items():
                if self.free[key]:
                    if key not in charts:
                        charts[key] = self.chart(nodes[key])
                    rr, cc = np.indices((len(residual), 5))
                    rows.append((rr + row).ravel())
                    columns.append((cc + offsets[key]).ravel())
                    data.append((block @ charts[key]).ravel())
            row += len(residual)
        if not errors:
            return cost, np.empty(0), coo_matrix((0, n)).tocsr() if jacobians else None, offsets
        j = None
        if jacobians:
            j = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))), shape=(row, n)).tocsr()
        return cost, np.concatenate(errors), j, offsets

    def optimize(self, max_iterations=50, gradient_tolerance=1e-7, relative_tolerance=1e-8):
        self.check_connected()
        damping = 1e-3
        for _ in range(max_iterations):
            cost, residual, j, offsets = self.system(self.nodes, True)
            h, g = (j.T @ j).tocsc(), np.asarray(j.T @ residual)
            if len(g) == 0 or np.linalg.norm(g, ord=np.inf) < gradient_tolerance:
                break
            diagonal = np.maximum(h.diagonal(), 1e-8)
            accepted = False
            for _ in range(12):
                with warnings.catch_warnings():
                    warnings.simplefilter('error', MatrixRankWarning)
                    delta = spsolve(h + diags(damping * diagonal), -g)
                try:
                    trial = {key: self.retract(value, delta[offsets[key]:offsets[key] + 5]) if self.free[key] else value
                             for key, value in self.nodes.items()}
                    trial_cost = self.system(trial, False)[0]
                except (ValueError, np.linalg.LinAlgError, OverflowError):
                    trial_cost = np.inf
                predicted = -.5 * float(delta @ (g - damping * diagonal * delta))
                gain = (cost - trial_cost) / max(predicted, 1e-30)
                if np.isfinite(trial_cost) and trial_cost < cost and gain > 0:
                    self.nodes = trial
                    damping = max(damping * max(1. / 3., 1. - (2. * min(gain, 1.) - 1.) ** 3), 1e-12)
                    accepted = True
                    break
                damping *= 10.
            if not accepted or cost - trial_cost < relative_tolerance * max(1., cost):
                break
