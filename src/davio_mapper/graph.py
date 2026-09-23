from dataclasses import dataclass, field
import time
import warnings
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from . import sim3

@dataclass
class Factor:
    a: str
    b: str
    kind: str
    measurement: np.ndarray = field(default_factory=lambda: np.eye(4))
    local_a: np.ndarray = field(default_factory=lambda: np.eye(4))
    local_b: np.ndarray = field(default_factory=lambda: np.eye(4))
    projected: bool = False
    log_ratio: float = 0.
    sigmas: np.ndarray = field(default_factory=lambda: np.ones(7))
    weight: float = 1.
    huber_delta: float = 1.

    def raw(self, nodes, jacobians=True):
        a, b = nodes[self.a], nodes[self.b]
        if self.kind in ("anchor", "submap_scale"):
            residual = np.array([np.log(sim3.scale(b)) - np.log(sim3.scale(a)) - self.log_ratio])
            row = np.zeros((1, 7))
            row[0, 6] = 1.
            return residual, {self.a: -row, self.b: row} if jacobians else {}
        u, v = a @ self.local_a, b @ self.local_b
        z_inv = sim3.inverse(self.measurement)
        relative = z_inv @ sim3.inverse(sim3.pose(u) if self.projected else u) @ (sim3.pose(v) if self.projected else v)
        xi = sim3.log(relative)
        n = 6 if self.projected else 7
        if not jacobians:
            return xi[:n], {}
        left, right = sim3.log_jacobians(xi)
        du = sim3.projection_jacobian(u) if self.projected else np.eye(7)
        dv = sim3.projection_jacobian(v) if self.projected else np.eye(7)
        ja = -left @ sim3.adjoint(z_inv) @ du @ sim3.adjoint(sim3.inverse(self.local_a))
        jb = right @ dv @ sim3.adjoint(sim3.inverse(self.local_b))
        return xi[:n], {self.a: ja[:n], self.b: jb[:n]}

    def linearize(self, nodes, jacobians=True):
        residual, blocks = self.raw(nodes, jacobians)
        sigmas = np.broadcast_to(np.asarray(self.sigmas, dtype=float), residual.shape)
        if (np.any(sigmas <= 0) or not np.isfinite(sigmas).all()
                or not np.isfinite(self.weight) or self.weight <= 0
                or self.huber_delta <= 0 or not np.isfinite(self.huber_delta)):
            raise ValueError("Factor sigmas, weight and Huber delta must be finite and positive")
        whitening = np.sqrt(self.weight) / sigmas
        residual = whitening * residual
        length = np.linalg.norm(residual)
        if not np.isfinite(length):
            raise ValueError("Non-finite factor residual")
        d = self.huber_delta
        cost = .5 * length ** 2 if length <= d else d * (length - .5 * d)
        robust = np.sqrt(min(1., d / max(length, 1e-30)))
        return cost, robust * residual, {k: robust * whitening[:, None] * j for k, j in blocks.items()}


class Graph:
    DIMENSIONS = (0, 1, 5, 6, 7)
    GRAVITY = np.array([0., 0., 1.])   # world z of the (gravity-aligned) odometry frame

    def __init__(self):
        self.nodes, self.dimensions, self.factors = {}, {}, []

    def add_node(self, key, transform, dimensions=7):
        if key in self.nodes or dimensions not in self.DIMENSIONS:
            raise ValueError("Duplicate node or unsupported node dimension")
        transform = sim3.validate(transform)
        if dimensions == 6 and not np.isclose(sim3.scale(transform), 1.):
            raise ValueError("Physical SE(3) camera nodes must have unit scale")
        self.nodes[key] = transform
        self.dimensions[key] = dimensions

    def chart(self, key, transform):
        """7 x d matrix: chart increment -> right Sim(3) increment, at ``transform``."""
        d = self.dimensions[key]
        if d in (6, 7):
            return np.eye(7)[:, :d]
        if d == 1:
            return np.eye(7)[:, 6:7]
        s = sim3.scale(transform)
        r = transform[:3, :3] / s
        m = np.zeros((7, 5))
        m[:3, :3] = r.T / s              # world translation v -> body v/s
        m[3:6, 3] = r.T @ self.GRAVITY   # yaw about world z -> body rotation vector
        m[6, 4] = 1.
        return m

    def retract(self, key, transform, step):
        d = self.dimensions[key]
        step = np.asarray(step, float)
        if d in (6, 7):
            full = np.zeros(7)
            full[:d] = step
            return transform @ sim3.exp(full)
        out = transform.copy()
        if d == 1:
            out[:3, :3] *= np.exp(step[0])
            return out
        v, yaw, sigma = step[:3], step[3], step[4]
        rz = sim3.exp(np.r_[0., 0., 0., self.GRAVITY * yaw, 0.])[:3, :3]
        out[:3, :3] = np.exp(sigma) * rz @ transform[:3, :3]
        out[:3, 3] = transform[:3, 3] + v
        return out

    @staticmethod
    def _factor_keys(factor):
        if hasattr(factor, "keys"):
            keys = tuple(factor.keys)
        else:
            keys = (factor.a, factor.b)

        if len(keys) < 2 or len(set(keys)) != len(keys):
            raise ValueError(
                "A factor needs distinct existing nodes"
            )

        return keys

    def add(self, factor):
        keys = self._factor_keys(factor)

        missing = [
            key
            for key in keys
            if key not in self.nodes
        ]

        if missing:
            raise ValueError(
                f"Factor references missing nodes: {missing}"
            )

        self.factors.append(factor)

    def validate_connectivity(self):
        fixed = [k for k, d in self.dimensions.items() if d == 0]
        if not fixed:
            raise ValueError("Fix a reference node to remove the global gauge")
        neighbors = {k: set() for k in self.nodes}
        for f in self.factors:
            if f.kind in ("anchor", "submap_scale"):
                continue

            keys = self._factor_keys(f)

            # A multi-camera factor forms one connected hyperedge.
            anchor = keys[0]

            for other in keys[1:]:
                neighbors[anchor].add(other)
                neighbors[other].add(anchor)

        reached, stack = set(fixed), list(fixed)
        while stack:
            for other in neighbors[stack.pop()] - reached:
                reached.add(other)
                stack.append(other)
        if reached != set(self.nodes):
            raise ValueError(f"Nodes disconnected from the gauge: {sorted(set(self.nodes) - reached)}")

    def _system(self, nodes, jacobians):
        offsets, n = {}, 0
        for key, d in self.dimensions.items():
            if d:
                offsets[key], n = n, n + d
        row_indices, column_indices, data, errors = [], [], [], []
        cost, row = 0., 0
        for factor in self.factors:
            value, residual, blocks = factor.linearize(nodes, jacobians)
            cost += value
            errors.extend(residual)
            if jacobians:
                for key, block in blocks.items():
                    d = self.dimensions[key]
                    if not d:
                        continue
                    rr, cc = np.indices((len(residual), d))
                    row_indices.extend((rr + row).ravel())
                    column_indices.extend((cc + offsets[key]).ravel())
                    data.extend((block @ self.chart(key, nodes[key])).ravel())
            row += len(residual)
        j = coo_matrix((data, (row_indices, column_indices)), shape=(row, n)).tocsr() if jacobians else None
        return cost, np.asarray(errors), j, offsets

    def optimize(self, max_iterations=50, gradient_tolerance=1e-7, relative_tolerance=1e-8):
        self.validate_connectivity()
        start = time.perf_counter()
        damping, trace = 1e-3, []
        initial = self._system(self.nodes, False)[0]
        reason, converged = "iteration_limit", False
        for iteration in range(max_iterations):
            cost, residual, j, offsets = self._system(self.nodes, True)
            h, g = (j.T @ j).tocsc(), np.asarray(j.T @ residual)
            if len(g) == 0 or np.linalg.norm(g, ord=np.inf) < gradient_tolerance:
                reason, converged = "gradient", True
                break
            diagonal = np.maximum(h.diagonal(), 1e-8)
            accepted = False
            for _ in range(12):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    delta = spsolve(h + diags(damping * diagonal), -g)
                trial = {}
                try:
                    for key, value in self.nodes.items():
                        d = self.dimensions[key]
                        trial[key] = (self.retract(key, value, delta[offsets[key]:offsets[key] + d])
                                      if d else value)
                    trial_cost = self._system(trial, False)[0]
                except (ValueError, np.linalg.LinAlgError, OverflowError):
                    trial_cost = np.inf
                predicted = -.5 * float(delta @ (g - damping * diagonal * delta))
                gain = (cost - trial_cost) / max(predicted, 1e-30)
                if np.isfinite(trial_cost) and trial_cost < cost and gain > 0:
                    self.nodes = trial
                    damping *= max(1. / 3., 1. - (2. * min(gain, 1.) - 1.) ** 3)
                    damping = max(damping, 1e-12)
                    accepted = True
                    break
                damping *= 10.
            trace.append({"iteration": iteration, "cost": cost, "trial_cost": float(trial_cost),
                          "accepted": accepted, "damping": damping})
            if not accepted:
                reason = "no_descent_step"
                break
            if cost - trial_cost < relative_tolerance * max(1., cost):
                reason, converged = "relative_cost", True
                break
        return {"converged": converged, "termination": reason, "iterations": len(trace),
                "initial_cost": initial, "final_cost": self._system(self.nodes, False)[0],
                "optimization_s": time.perf_counter() - start, "trace": trace}



@dataclass
class PointFactor:
    a: str
    b: str
    points_a: np.ndarray
    points_b: np.ndarray
    sigma: float = .10
    sigma_radial: float = 0.
    sigma_lateral: float = 0.
    huber_delta: float = 2.
    kind: str = 'sparse'
    _reference_whitener: np.ndarray = field(default=None, init=False, repr=False)

    def _covariance(self, points, transform):
        """Map one camera's ray-aligned depth uncertainty into the map frame."""
        depth = np.linalg.norm(points, axis=1)
        safe = np.maximum(depth, 1e-9)
        radial = np.hypot(self.sigma, self.sigma_radial * safe) ** 2
        lateral = np.hypot(self.sigma, self.sigma_lateral * safe) ** 2
        rays = points / safe[:, None]
        local = (lateral[:, None, None] * np.eye(3)
                 + (radial - lateral)[:, None, None] * rays[:, :, None] * rays[:, None, :])
        # transform[:3, :3] is s*R, so this carries the submap's own scale with it.
        return np.einsum('ij,kjl,ml->kim', transform[:3, :3], local, transform[:3, :3])

    def linearize(self, nodes, jacobians=True):
        pa, pb = np.asarray(self.points_a), np.asarray(self.points_b)
        if pa.shape != pb.shape or pa.ndim != 2 or pa.shape[1] != 3 or not len(pa):
            raise ValueError('Correspondences must be nonempty paired Nx3 points')
        if not np.isfinite(pa).all() or not np.isfinite(pb).all() or self.sigma <= 0:
            raise ValueError('Invalid sparse geometry or sigma')
        if self.sigma_radial < 0 or self.sigma_lateral < 0:
            raise ValueError('Sparse depth uncertainty fractions must be nonnegative')
        a, b = nodes[self.a], nodes[self.b]
        difference = pa @ a[:3, :3].T + a[:3, 3] - pb @ b[:3, :3].T - b[:3, 3]
        # Freeze at the factor's first evaluation. OnlineMapper constructs fresh
        # factors for each solve, so all LM steps and trial costs use one reference
        # covariance. This is a weighted least-squares surrogate, not a likelihood
        # with state-dependent covariance (whose derivatives would also be needed).
        if self._reference_whitener is None:
            self._reference_whitener = np.linalg.inv(np.linalg.cholesky(
                self._covariance(pa, a) + self._covariance(pb, b)))
        whitener = self._reference_whitener
        residual = np.einsum('kij,kj->ki', whitener, difference)
        lengths = np.linalg.norm(residual, axis=1)
        d = self.huber_delta
        costs = np.where(lengths <= d, .5 * lengths**2, d * (lengths - .5*d))
        weights = np.sqrt(np.minimum(1., d / np.maximum(lengths, 1e-30)) / len(pa))
        blocks = {}
        if jacobians:
            def block(points, transform):
                out = np.zeros((len(points), 3, 7))
                out[:, :, :3] = transform[:3, :3]
                for i, point in enumerate(points):
                    x, y, z = point
                    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                    out[i, :, 3:6] = -transform[:3, :3] @ skew
                out[:, :, 6] = points @ transform[:3, :3].T
                out = np.einsum('kij,kjl->kil', whitener, out)
                return (out * weights[:, None, None]).reshape(-1, 7)
            blocks = {self.a: block(pa, a), self.b: -block(pb, b)}
        return float(costs.mean()), (residual * weights[:, None]).ravel(), blocks
