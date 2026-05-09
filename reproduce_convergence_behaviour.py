#!/usr/bin/env python3
"""Reproduce the convergence-behaviour section for the layered magnitude report.

This is a self-contained script.  It does not import the other manuscript
runners.  It contains the two benchmark maps, the nondomination-layer logic,
the 2-D and 3-D dominated-set magnitude calculations, sweep-based indicator
subgradients, projected ascent, and all export routines used for the section
"Convergence behaviour".

Experiments
-----------
1. Curved 2-D Pareto front with 10 decision points.  The initial set is chosen
   away from the final indicator optimum.  The run uses 50 projected ascent
   iterations.
2. Supersphere / bulged three-peaks benchmark with gamma=1 and a Das--Dennis
   H=4 point set, hence mu=15 points.  The run uses 70 projected ascent
   iterations.
3. A layered-start 3-D box experiment with gamma=1 and mu=15 points sampled
   mildly outside the Pareto-front range, in the box [-0.25,1.25]^3 and
   optimized in [-0.4,1.4]^3. The run uses 45 projected ascent iterations;
   its layer profile remains multi-layered for the first sampled iterations
   and then reaches a single nondominated layer.
4. A 500-episode variant of the layered-start 3-D box experiment.  If the
   layered magnitude grows by less than 5e-3 over a 10-episode window, the
   algorithm tries one backtracking perturbation step: several points are perturbed, a temporary
   indicator drop is allowed, and gradient ascent then resumes.
5. The same stochastic-recovery strategy with an H=5 point budget, i.e.
   mu=(H+1)(H+2)/2=21 points, initialized mildly outside the natural front
   range in [-0.25,1.25]^3.

The output samples convergence curves and layer tables every 20 iterations/episodes and also writes final 3-D point sets for the 500-episode runs, so that the
LaTeX report can remain self-contained.  If Numba is installed, the dense
benchmark evaluations and Jacobians are JIT compiled; otherwise the script
falls back to pure NumPy.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

Array = np.ndarray


@njit(cache=False)
def _curved2d_objective_jit(X: Array) -> Array:
    n = X.shape[0]
    Y = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        x1 = X[i, 0]
        x2 = X[i, 1]
        Y[i, 0] = 1.0 - 0.5 * ((x1 - 1.0) * (x1 - 1.0) + (x2 - 1.0) * (x2 - 1.0))
        Y[i, 1] = 1.0 - 0.5 * (x1 * x1 + x2 * x2)
    return Y


@njit(cache=False)
def _curved2d_jacobian_jit(X: Array) -> Array:
    n = X.shape[0]
    J = np.zeros((n, 2, 2), dtype=np.float64)
    for i in range(n):
        x1 = X[i, 0]
        x2 = X[i, 1]
        J[i, 0, 0] = 1.0 - x1
        J[i, 0, 1] = 1.0 - x2
        J[i, 1, 0] = -x1
        J[i, 1, 1] = -x2
    return J


@njit(cache=False)
def _supersphere_gamma1_objective_jit(X: Array, clip_to_unit: bool) -> Array:
    n = X.shape[0]
    Y = np.empty((n, 3), dtype=np.float64)
    for r in range(n):
        for i in range(3):
            s = 0.0
            for k in range(3):
                e = 1.0 if k == i else 0.0
                d = X[r, k] - e
                s += d * d
            val = 1.0 - 0.5 * s
            if clip_to_unit:
                if val < 0.0:
                    val = 0.0
                elif val > 1.0:
                    val = 1.0
            Y[r, i] = val
    return Y


@njit(cache=False)
def _supersphere_gamma1_jacobian_jit(X: Array) -> Array:
    n = X.shape[0]
    J = np.empty((n, 3, 3), dtype=np.float64)
    for r in range(n):
        for i in range(3):
            for k in range(3):
                e = 1.0 if k == i else 0.0
                J[r, i, k] = -(X[r, k] - e)
    return J


@njit(cache=False)
def _clip_box_jit(X: Array, lo: float, hi: float) -> Array:
    Y = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            v = X[i, j]
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            Y[i, j] = v
    return Y

# ---------------------------------------------------------------------------
# Common nondomination, hypervolume/magnitude values, and subgradients
# ---------------------------------------------------------------------------

def nondomination_layers(Y: Array) -> list[list[int]]:
    Y = np.asarray(Y, dtype=float)
    remaining = list(range(len(Y)))
    layers: list[list[int]] = []
    while remaining:
        front: list[int] = []
        for i in remaining:
            yi = Y[i]
            dominated = False
            for j in remaining:
                if j != i and np.all(Y[j] >= yi) and np.any(Y[j] > yi):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        layers.append(front)
        front_set = set(front)
        remaining = [i for i in remaining if i not in front_set]
    return layers


def layer_sizes(layers: list[list[int]]) -> list[int]:
    return [len(layer) for layer in layers]


def layer_string(layers: list[list[int]]) -> str:
    return "+".join(str(len(layer)) for layer in layers)


def nondominated_2d(points: Array) -> Array:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.empty((0, 2), dtype=float)
    order = np.argsort(pts[:, 0], kind="mergesort")
    pts = pts[order]
    out = []
    max_y = -np.inf
    for p in pts[::-1]:
        if p[1] > max_y + 1e-15:
            out.append(p)
            max_y = p[1]
    return np.array(out[::-1], dtype=float)


def hv2_value(points: Array) -> float:
    pts = nondominated_2d(np.maximum(np.asarray(points, dtype=float), 0.0))
    area = 0.0
    x_prev = 0.0
    for x, y in pts:
        area += (float(x) - x_prev) * float(y)
        x_prev = float(x)
    return float(area)


def hv3_value(points: Array) -> float:
    pts = np.maximum(np.asarray(points, dtype=float), 0.0)
    if len(pts) == 0:
        return 0.0
    xs = np.unique(pts[:, 0])
    xs.sort()
    total = 0.0
    x_prev = 0.0
    for x in xs:
        active = pts[pts[:, 0] >= x - 1e-15][:, [1, 2]]
        total += (float(x) - x_prev) * hv2_value(active)
        x_prev = float(x)
    return float(total)


def exclusive_length_1d(t: float, active: list[float]) -> float:
    covered = max(active) if active else 0.0
    return max(0.0, float(t) - float(covered))


def exclusive_area_2d(y: float, z: float, active_rects: list[tuple[float, float]]) -> float:
    if not active_rects:
        return float(y) * float(z)
    clipped = np.array([(min(a, y), min(b, z)) for a, b in active_rects], dtype=float)
    clipped = clipped[(clipped[:, 0] > 0.0) & (clipped[:, 1] > 0.0)]
    covered = hv2_value(clipped) if len(clipped) else 0.0
    return max(0.0, float(y) * float(z) - covered)


def axis_gradient_forward(q: Array) -> Array:
    n, d = q.shape
    g = np.zeros((n, d), dtype=float)
    for k in range(d):
        order = sorted(range(n), key=lambda i: (q[i, k], i))
        if order:
            g[order[-1], k] = 1.0
    return g


def hv2_gradient_forward(points2d: Array) -> Array:
    pts = np.asarray(points2d, dtype=float)
    n = len(pts)
    g = np.zeros((n, 2), dtype=float)
    active_y: list[float] = []
    for i in sorted(range(n), key=lambda i: (pts[i, 0], i), reverse=True):
        g[i, 0] = exclusive_length_1d(pts[i, 1], active_y)
        active_y.append(float(pts[i, 1]))
    active_x: list[float] = []
    for i in sorted(range(n), key=lambda i: (pts[i, 1], i), reverse=True):
        g[i, 1] = exclusive_length_1d(pts[i, 0], active_x)
        active_x.append(float(pts[i, 0]))
    return g


def hv3_gradient_forward(points3d: Array) -> Array:
    pts = np.asarray(points3d, dtype=float)
    n = len(pts)
    g = np.zeros((n, 3), dtype=float)
    active_yz: list[tuple[float, float]] = []
    for i in sorted(range(n), key=lambda i: (pts[i, 0], i), reverse=True):
        g[i, 0] = exclusive_area_2d(pts[i, 1], pts[i, 2], active_yz)
        active_yz.append((float(pts[i, 1]), float(pts[i, 2])))
    active_xz: list[tuple[float, float]] = []
    for i in sorted(range(n), key=lambda i: (pts[i, 1], i), reverse=True):
        g[i, 1] = exclusive_area_2d(pts[i, 0], pts[i, 2], active_xz)
        active_xz.append((float(pts[i, 0]), float(pts[i, 2])))
    active_xy: list[tuple[float, float]] = []
    for i in sorted(range(n), key=lambda i: (pts[i, 2], i), reverse=True):
        g[i, 2] = exclusive_area_2d(pts[i, 0], pts[i, 1], active_xy)
        active_xy.append((float(pts[i, 0]), float(pts[i, 1])))
    return g


def magnitude_value(points: Array) -> float:
    q = np.maximum(np.asarray(points, dtype=float), 0.0)
    if len(q) == 0:
        return 0.0
    if q.shape[1] == 2:
        return float(1.0 + 0.5 * (np.max(q[:, 0]) + np.max(q[:, 1])) + 0.25 * hv2_value(q))
    lx, ly, lz = np.max(q, axis=0)
    area_xy = hv2_value(q[:, [0, 1]])
    area_xz = hv2_value(q[:, [0, 2]])
    area_yz = hv2_value(q[:, [1, 2]])
    return float(1.0 + 0.5 * (lx + ly + lz) + 0.25 * (area_xy + area_xz + area_yz) + 0.125 * hv3_value(q))


def magnitude_gradient(points: Array) -> Array:
    q = np.maximum(np.asarray(points, dtype=float), 0.0)
    if q.shape[1] == 2:
        return 0.5 * axis_gradient_forward(q) + 0.25 * hv2_gradient_forward(q)
    axis_g = axis_gradient_forward(q)
    hv3_g = hv3_gradient_forward(q)
    g_xy = hv2_gradient_forward(q[:, [0, 1]])
    g_xz = hv2_gradient_forward(q[:, [0, 2]])
    g_yz = hv2_gradient_forward(q[:, [1, 2]])
    proj_g = np.zeros_like(q)
    proj_g[:, 0] += g_xy[:, 0]
    proj_g[:, 1] += g_xy[:, 1]
    proj_g[:, 0] += g_xz[:, 0]
    proj_g[:, 2] += g_xz[:, 1]
    proj_g[:, 1] += g_yz[:, 0]
    proj_g[:, 2] += g_yz[:, 1]
    return 0.5 * axis_g + 0.25 * proj_g + 0.125 * hv3_g




def hypervolume_value(points: Array) -> float:
    q = np.maximum(np.asarray(points, dtype=float), 0.0)
    if len(q) == 0:
        return 0.0
    if q.shape[1] == 2:
        return hv2_value(q)
    return hv3_value(q)


def hypervolume_gradient(points: Array) -> Array:
    q = np.maximum(np.asarray(points, dtype=float), 0.0)
    if q.shape[1] == 2:
        return hv2_gradient_forward(q)
    return hv3_gradient_forward(q)


def front_indicator_value(points: Array, indicator: str = "magnitude") -> float:
    if indicator == "magnitude":
        return magnitude_value(points)
    if indicator == "hypervolume":
        return hypervolume_value(points)
    raise ValueError(f"unknown indicator: {indicator}")


def front_indicator_gradient(points: Array, indicator: str = "magnitude") -> Array:
    if indicator == "magnitude":
        return magnitude_gradient(points)
    if indicator == "hypervolume":
        return hypervolume_gradient(points)
    raise ValueError(f"unknown indicator: {indicator}")

def repulsion_value(Y: Array, sigma: float) -> float:
    val = 0.0
    for i in range(len(Y)):
        diff = Y[i + 1 :] - Y[i]
        if len(diff):
            val += float(np.sum(np.exp(-np.sum(diff * diff, axis=1) / (sigma * sigma))))
    return val


def repulsion_gradient(Y: Array, sigma: float) -> Array:
    G = np.zeros_like(Y)
    inv = 1.0 / (sigma * sigma)
    for i in range(len(Y)):
        for j in range(i + 1, len(Y)):
            diff = Y[i] - Y[j]
            e = math.exp(-inv * float(np.dot(diff, diff)))
            c = 2.0 * inv * e * diff
            G[i] += c
            G[j] -= c
    return G




def layered_value_only(Y: Array, eps_layer: float, tau: float, sigma: float, indicator: str = "magnitude") -> tuple[float, list[list[int]]]:
    """Layered indicator value without computing indicator gradients.

    Used inside stochastic recovery trials. The indicator can be either
    ``magnitude`` or ``hypervolume``. Layering and repulsion are unchanged.
    """
    Y = np.asarray(Y, dtype=float)
    layers = nondomination_layers(Y)
    value = 0.0
    for ell, front in enumerate(layers):
        value += (eps_layer ** ell) * front_indicator_value(Y[front], indicator=indicator)
    if tau > 0:
        value -= tau * repulsion_value(Y, sigma)
    return float(value), layers


def layered_value_and_gradient(Y: Array, eps_layer: float, tau: float, sigma: float, indicator: str = "magnitude") -> tuple[float, Array, list[list[int]]]:
    Y = np.asarray(Y, dtype=float)
    layers = nondomination_layers(Y)
    value = 0.0
    G = np.zeros_like(Y)
    for ell, front in enumerate(layers):
        w = eps_layer ** ell
        pts = Y[front]
        value += w * front_indicator_value(pts, indicator=indicator)
        gfront = front_indicator_gradient(pts, indicator=indicator)
        for local_idx, global_idx in enumerate(front):
            G[global_idx] += w * gfront[local_idx]
    if tau > 0:
        value -= tau * repulsion_value(Y, sigma)
        G -= tau * repulsion_gradient(Y, sigma)
    return float(value), G, layers

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def curved2d_objective(X: Array) -> Array:
    if NUMBA_AVAILABLE:
        return _curved2d_objective_jit(np.asarray(X, dtype=np.float64))
    x1 = X[:, 0]
    x2 = X[:, 1]
    f1 = 1.0 - 0.5 * ((x1 - 1.0) ** 2 + (x2 - 1.0) ** 2)
    f2 = 1.0 - 0.5 * (x1 * x1 + x2 * x2)
    return np.column_stack([f1, f2])


def curved2d_jacobian(X: Array) -> Array:
    if NUMBA_AVAILABLE:
        return _curved2d_jacobian_jit(np.asarray(X, dtype=np.float64))
    x1 = X[:, 0]
    x2 = X[:, 1]
    J = np.zeros((len(X), 2, 2), dtype=float)
    J[:, 0, 0] = 1.0 - x1
    J[:, 0, 1] = 1.0 - x2
    J[:, 1, 0] = -x1
    J[:, 1, 1] = -x2
    return J


def curved2d_initial_points() -> Array:
    return np.array([
        [-0.22, 0.18], [-0.15, 0.44], [0.03, 0.12], [0.10, 0.48], [0.24, 0.02],
        [0.31, 0.36], [0.47, 0.08], [0.55, 0.30], [0.68, 0.00], [0.80, 0.16],
    ], dtype=float)


def project_2d_box(X: Array) -> Array:
    if NUMBA_AVAILABLE:
        return _clip_box_jit(np.asarray(X, dtype=np.float64), -0.35, 1.25)
    return np.clip(X, -0.35, 1.25)


def project_simplex_rows(V: Array) -> Array:
    V = np.asarray(V, dtype=float)
    U = -np.sort(-V, axis=1)
    cssv = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, V.shape[1] + 1)
    cond = U - cssv / ind > 0
    rho = cond.sum(axis=1) - 1
    theta = cssv[np.arange(V.shape[0]), rho] / (rho + 1)
    return np.maximum(V - theta[:, None], 0.0)


def das_dennis_simplex_grid_3d(H: int) -> Array:
    pts = []
    for a in range(H + 1):
        for b in range(H + 1 - a):
            c = H - a - b
            pts.append((a / H, b / H, c / H))
    return np.array(pts, dtype=float)


def supersphere_initial_points(H: int = 4, seed: int = 8, sigma: float = 0.01) -> Array:
    rng = np.random.default_rng(seed)
    X = das_dennis_simplex_grid_3d(H)
    return project_simplex_rows(X + rng.normal(0.0, sigma, size=X.shape))


def supersphere_gamma1_objective(X: Array) -> Array:
    if NUMBA_AVAILABLE:
        return _supersphere_gamma1_objective_jit(np.asarray(X, dtype=np.float64), True)
    E = np.eye(3)
    Y = np.zeros((len(X), 3), dtype=float)
    for i, e in enumerate(E):
        Y[:, i] = 1.0 - 0.5 * np.sum((X - e) ** 2, axis=1)
    return np.clip(Y, 0.0, 1.0)


def supersphere_gamma1_jacobian(X: Array) -> Array:
    if NUMBA_AVAILABLE:
        return _supersphere_gamma1_jacobian_jit(np.asarray(X, dtype=np.float64))
    E = np.eye(3)
    J = np.zeros((len(X), 3, 3), dtype=float)
    for i, e in enumerate(E):
        J[:, i, :] = -(X - e)
    return J


def supersphere_box_layered_initial_points(seed: int = 0, n_points: int = 15) -> Array:
    """Mildly outside-front 3-D start in a box.

    The points are sampled from [-0.25,1.25]^3. This is deliberately only a
    little outside the natural supersphere range: it creates dominated layers
    at the beginning, but the set does not collapse to one layer immediately.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.25, 1.25, size=(n_points, 3))


def supersphere_box_h_budget_initial_points(H: int = 5, seed: int = 0) -> Array:
    """Mildly outside-front 3-D start with a Das--Dennis-equivalent point budget.

    For three objectives, H=5 corresponds to mu=(H+1)(H+2)/2=21 points.  The
    points are sampled from the same mild box as the layered-start experiment;
    H is used here to choose the comparable point-set budget.
    """
    n_points = (H + 1) * (H + 2) // 2
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.25, 1.25, size=(n_points, 3))


def project_3d_layered_box(X: Array) -> Array:
    if NUMBA_AVAILABLE:
        return _clip_box_jit(np.asarray(X, dtype=np.float64), -0.4, 1.4)
    return np.clip(X, -0.4, 1.4)


def supersphere_gamma1_box_objective(X: Array) -> Array:
    if NUMBA_AVAILABLE:
        return _supersphere_gamma1_objective_jit(np.asarray(X, dtype=np.float64), False)
    E = np.eye(3)
    Y = np.zeros((len(X), 3), dtype=float)
    for i, e in enumerate(E):
        Y[:, i] = 1.0 - 0.5 * np.sum((X - e) ** 2, axis=1)
    return Y

# ---------------------------------------------------------------------------
# Projected ascent and exports
# ---------------------------------------------------------------------------

def normalize_rows(G: Array) -> Array:
    H = np.array(G, dtype=float, copy=True)
    norms = np.linalg.norm(H, axis=1)
    mask = norms > 1.0e-12
    H[mask] = H[mask] / norms[mask][:, None]
    return H


def run_projected_ascent(
    X0: Array,
    objective_fn,
    jacobian_fn,
    projector,
    *,
    max_iter: int,
    alpha0: float,
    eps_layer: float = 1e-3,
    tau: float = 2e-4,
    sigma: float = 0.04,
    shrink: float = 0.70,
    max_retries: int = 12,
    alpha_boost: float = 1.01,
    indicator: str = "magnitude",
) -> dict:
    X = projector(np.asarray(X0, dtype=float).copy())
    Y = objective_fn(X)
    value, GY, layers = layered_value_and_gradient(Y, eps_layer, tau, sigma, indicator=indicator)
    hist_X = [X.copy()]
    hist_Y = [Y.copy()]
    values = [value]
    layer_history = [layers]
    alphas = [alpha0]
    alpha = float(alpha0)
    accepted_steps = 0
    for _it in range(1, max_iter + 1):
        J = jacobian_fn(X)
        GX = np.einsum("nij,ni->nj", J, GY)
        GX = normalize_rows(GX)
        trial_alpha = alpha
        accepted = False
        for _retry in range(max_retries + 1):
            Xt = projector(X + trial_alpha * GX)
            Yt = objective_fn(Xt)
            vt, GYt, layers_t = layered_value_and_gradient(Yt, eps_layer, tau, sigma, indicator=indicator)
            if vt >= value - 1e-12:
                X, Y, value, GY, layers = Xt, Yt, vt, GYt, layers_t
                alpha = trial_alpha * alpha_boost
                accepted = True
                accepted_steps += 1
                break
            trial_alpha *= shrink
        if not accepted:
            alpha = max(1.0e-5, 0.5 * alpha)
        hist_X.append(X.copy())
        hist_Y.append(Y.copy())
        values.append(value)
        layer_history.append(layers)
        alphas.append(alpha)
    return {
        "X": X,
        "Y": Y,
        "hist_X": hist_X,
        "hist_Y": hist_Y,
        "values": values,
        "layers": layer_history,
        "alphas": alphas,
        "accepted_steps": accepted_steps,
        "indicator": indicator,
    }





def run_projected_ascent_with_stochastic_recovery(
    X0: Array,
    objective_fn,
    jacobian_fn,
    projector,
    *,
    max_iter: int,
    alpha0: float,
    eps_layer: float = 1e-3,
    tau: float = 2e-4,
    sigma: float = 0.04,
    shrink: float = 0.65,
    max_retries: int = 6,
    alpha_boost: float = 1.00,
    stagnation_window: int = 10,
    stagnation_tol: float = 5.0e-3,
    stochastic_step: float = 0.16,
    stochastic_trials: int = 1,
    stochastic_seed: int = 31,
    perturb_fraction: float = 0.45,
    perturb_cooldown: int = 10,
    max_backtrack_drop: float = 0.12,
    final_unperturbed_iters: int = 10,
    indicator: str = "magnitude",
) -> dict:
    """Projected ascent with larger backtracking perturbations.

    A deterministic projected ascent step is attempted first. If the layered
    magnitude has stagnated over ``stagnation_window`` episodes and we are not
    inside the last ``final_unperturbed_iters`` episodes, the algorithm applies
    a larger perturbation to several decision points. The perturbation is
    allowed to decrease the layered magnitude temporarily (backtracking), but
    candidates whose immediate drop exceeds ``max_backtrack_drop`` are rejected.
    After a perturbation, the method resumes ordinary gradient ascent until the
    next stagnation. The final ten recorded episodes are always unperturbed.
    """
    rng = np.random.default_rng(stochastic_seed)
    X = projector(np.asarray(X0, dtype=float).copy())
    Y = objective_fn(X)
    value, GY, layers = layered_value_and_gradient(Y, eps_layer, tau, sigma, indicator=indicator)
    hist_X = [X.copy()]
    hist_Y = [Y.copy()]
    values = [float(value)]
    layer_history = [layers]
    alphas = [alpha0]
    recovery_flags = [0]
    alpha = float(alpha0)
    accepted_steps = 0
    stochastic_attempts = 0
    stochastic_accepted = 0
    cooldown_left = 0

    for _it in range(1, max_iter + 1):
        J = jacobian_fn(X)
        GX = np.einsum("nij,ni->nj", J, GY)
        GX = normalize_rows(GX)
        trial_alpha = alpha
        accepted = False
        for _retry in range(max_retries + 1):
            Xt = projector(X + trial_alpha * GX)
            Yt = objective_fn(Xt)
            vt, GYt, layers_t = layered_value_and_gradient(Yt, eps_layer, tau, sigma, indicator=indicator)
            if vt >= value - 1e-12:
                X, Y, value, GY, layers = Xt, Yt, vt, GYt, layers_t
                alpha = trial_alpha * alpha_boost
                accepted = True
                accepted_steps += 1
                break
            trial_alpha *= shrink
        if not accepted:
            alpha = max(1.0e-5, 0.5 * alpha)

        recovered = False
        if cooldown_left > 0:
            cooldown_left -= 1

        allow_perturbation = _it <= max_iter - final_unperturbed_iters
        stagnated = len(values) >= stagnation_window and value - values[-stagnation_window] < stagnation_tol
        if allow_perturbation and cooldown_left == 0 and stagnated:
            stochastic_attempts += 1
            old_value = float(value)
            n_perturb = max(1, int(math.ceil(perturb_fraction * len(X))))
            best = None
            for _trial in range(max(1, stochastic_trials)):
                Xt = np.array(X, copy=True)
                indices = rng.choice(len(X), size=n_perturb, replace=False)
                noise = rng.normal(size=(n_perturb, X.shape[1]))
                norms = np.linalg.norm(noise, axis=1)
                norms[norms <= 1.0e-14] = 1.0
                noise = noise / norms[:, None]
                lengths = stochastic_step * rng.uniform(0.35, 1.00, size=n_perturb)
                Xt[indices] = Xt[indices] + lengths[:, None] * noise
                Xt = projector(Xt)
                Yt = objective_fn(Xt)
                vt, layers_t = layered_value_only(Yt, eps_layer, tau, sigma, indicator=indicator)
                # Backtracking filter: accept temporary drops, but not very large ones.
                if vt >= old_value - max_backtrack_drop and (best is None or vt > best[0]):
                    best = (vt, Xt, Yt, layers_t)
            if best is not None:
                value, X, Y, layers = best
                value, GY, layers = layered_value_and_gradient(Y, eps_layer, tau, sigma, indicator=indicator)
                alpha = max(alpha, alpha0)
                stochastic_accepted += 1
                recovered = True
                cooldown_left = perturb_cooldown

        hist_X.append(X.copy())
        hist_Y.append(Y.copy())
        values.append(float(value))
        layer_history.append(layers)
        alphas.append(alpha)
        recovery_flags.append(1 if recovered else 0)

    return {
        "X": X,
        "Y": Y,
        "hist_X": hist_X,
        "hist_Y": hist_Y,
        "values": values,
        "layers": layer_history,
        "alphas": alphas,
        "accepted_steps": accepted_steps,
        "stochastic_attempts": stochastic_attempts,
        "stochastic_accepted": stochastic_accepted,
        "recovery_flags": recovery_flags,
        "indicator": indicator,
    }

def sample_indices(n_items: int, step: int = 20) -> list[int]:
    """Return every ``step``-th recorded item, always including the final item."""
    if n_items <= 0:
        return []
    idx = list(range(0, n_items, max(1, int(step))))
    if idx[-1] != n_items - 1:
        idx.append(n_items - 1)
    return idx


def write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        writer.writerows(rows)


def coords_2d(points: Array) -> str:
    return " ".join(f"({p[0]:.5f},{p[1]:.5f})" for p in points)


def coords_3d(points: Array) -> str:
    return " ".join(f"({p[0]:.5f},{p[1]:.5f},{p[2]:.5f})" for p in points)



def write_tex_snippets(outdir: Path, res2: dict, res3: dict, res3_far: dict,
                       res3_recovery: dict, res3_h5: dict,
                       res3_recovery_hv: dict, res3_h5_hv: dict) -> None:
    idx2 = sample_indices(len(res2["values"]), 20)
    idx3 = sample_indices(len(res3["values"]), 20)
    idx3_far = sample_indices(len(res3_far["values"]), 20)
    idx3_recovery = sample_indices(len(res3_recovery["values"]), 20)
    idx3_h5 = sample_indices(len(res3_h5["values"]), 20)
    idx3_recovery_hv = sample_indices(len(res3_recovery_hv["values"]), 20)
    idx3_h5_hv = sample_indices(len(res3_h5_hv["values"]), 20)
    lines = []
    lines.append("% Automatically generated by reproduce_convergence_behaviour_v17_2026-05-09.py")
    lines.append("% 2-D convergence coordinates: iteration, layered magnitude")
    lines.append("\\def\\ConvTwoDCoords{" + " ".join(f"({i},{res2['values'][i]:.8f})" for i in idx2) + "}")
    lines.append("% 3-D simplex convergence coordinates: iteration, layered magnitude")
    lines.append("\\def\\ConvThreeDCoords{" + " ".join(f"({i},{res3['values'][i]:.8f})" for i in idx3) + "}")
    lines.append("% 3-D layered-start convergence coordinates: iteration, layered magnitude")
    lines.append("\\def\\ConvThreeDFarCoords{" + " ".join(f"({i},{res3_far['values'][i]:.8f})" for i in idx3_far) + "}")
    lines.append("% 3-D stochastic-recovery convergence coordinates: iteration, layered magnitude")
    lines.append("\\def\\ConvThreeDRecoveryCoords{" + " ".join(f"({i},{res3_recovery['values'][i]:.8f})" for i in idx3_recovery) + "}")
    lines.append("% 3-D H=5 stochastic-recovery convergence coordinates: iteration, layered magnitude")
    lines.append("\\def\\ConvThreeDHFiveRecoveryCoords{" + " ".join(f"({i},{res3_h5['values'][i]:.8f})" for i in idx3_h5) + "}")
    lines.append("% 3-D stochastic-recovery convergence coordinates: iteration, layered hypervolume")
    lines.append("\\def\\ConvThreeDRecoveryHVCoords{" + " ".join(f"({i},{res3_recovery_hv['values'][i]:.8f})" for i in idx3_recovery_hv) + "}")
    lines.append("% 3-D H=5 stochastic-recovery convergence coordinates: iteration, layered hypervolume")
    lines.append("\\def\\ConvThreeDHFiveRecoveryHVCoords{" + " ".join(f"({i},{res3_h5_hv['values'][i]:.8f})" for i in idx3_h5_hv) + "}")
    lines.append("% 2-D initial and final objective-space sets")
    lines.append("\\def\\ConvTwoDInitial{" + coords_2d(res2["hist_Y"][0]) + "}")
    lines.append("\\def\\ConvTwoDFinal{" + coords_2d(res2["hist_Y"][-1]) + "}")
    lines.append("% 2-D sampled paths, each with at most every twentieth point")
    path_names = "ABCDEFGHIJ"
    for point_idx in range(res2["hist_Y"][0].shape[0]):
        path_pts = np.array([res2["hist_Y"][i][point_idx] for i in idx2], dtype=float)
        lines.append(f"\\def\\ConvTwoDPath{path_names[point_idx]}{{" + coords_2d(path_pts) + "}")
    lines.append("% 3-D simplex initial and final objective-space sets")
    lines.append("\\def\\ConvThreeDInitial{" + coords_3d(res3["hist_Y"][0]) + "}")
    lines.append("\\def\\ConvThreeDFinal{" + coords_3d(res3["hist_Y"][-1]) + "}")
    lines.append("% 3-D layered-start initial and final objective-space sets")
    lines.append("\\def\\ConvThreeDFarInitial{" + coords_3d(res3_far["hist_Y"][0]) + "}")
    lines.append("\\def\\ConvThreeDFarFinal{" + coords_3d(res3_far["hist_Y"][-1]) + "}")
    lines.append("% 3-D stochastic-recovery initial and final objective-space sets")
    lines.append("\\def\\ConvThreeDRecoveryInitial{" + coords_3d(res3_recovery["hist_Y"][0]) + "}")
    lines.append("\\def\\ConvThreeDRecoveryFinal{" + coords_3d(res3_recovery["hist_Y"][-1]) + "}")
    lines.append("% 3-D H=5 stochastic-recovery initial and final objective-space sets")
    lines.append("\\def\\ConvThreeDHFiveRecoveryInitial{" + coords_3d(res3_h5["hist_Y"][0]) + "}")
    lines.append("\\def\\ConvThreeDHFiveRecoveryFinal{" + coords_3d(res3_h5["hist_Y"][-1]) + "}")
    lines.append("% 3-D hypervolume stochastic-recovery final objective-space sets")
    lines.append("\\def\\ConvThreeDRecoveryHVFinal{" + coords_3d(res3_recovery_hv["hist_Y"][-1]) + "}")
    lines.append("\\def\\ConvThreeDHFiveRecoveryHVFinal{" + coords_3d(res3_h5_hv["hist_Y"][-1]) + "}")
    (outdir / "convergence_behaviour_coordinates.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_case(name: str, res: dict, n_points: int) -> dict:
    initial = float(res["values"][0])
    final = float(res["values"][-1])
    return {
        "case": name,
        "indicator": res.get("indicator", "magnitude"),
        "points": n_points,
        "iterations": len(res["values"]) - 1,
        "accepted_steps": int(res["accepted_steps"]),
        "cpu_seconds": float(res.get("cpu_seconds", 0.0)),
        "stochastic_attempts": int(res.get("stochastic_attempts", 0)),
        "stochastic_accepted": int(res.get("stochastic_accepted", 0)),
        "indicator_initial": initial,
        "indicator_final": final,
        "absolute_growth": final - initial,
        "relative_growth_percent": 100.0 * (final - initial) / abs(initial) if abs(initial) > 0 else 0.0,
        "initial_layers": layer_string(res["layers"][0]),
        "final_layers": layer_string(res["layers"][-1]),
    }


def write_outputs(outdir: Path, res2: dict, res3: dict, res3_far: dict,
                  res3_recovery: dict, res3_h5: dict,
                  res3_recovery_hv: dict, res3_h5_hv: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    summaries = [
        summarize_case("curved_2d", res2, 10),
        summarize_case("supersphere_3d_gamma1_H4", res3, 15),
        summarize_case("supersphere_3d_layered_box_gamma1", res3_far, 15),
        summarize_case("supersphere_3d_recovery_box_gamma1", res3_recovery, 15),
        summarize_case("supersphere_3d_H5_recovery_box_gamma1", res3_h5, 21),
        summarize_case("supersphere_3d_HV_recovery_box_gamma1", res3_recovery_hv, 15),
        summarize_case("supersphere_3d_HV_H5_recovery_box_gamma1", res3_h5_hv, 21),
    ]
    write_csv(outdir / "convergence_behaviour_summary.csv", summaries[0].keys(), [s.values() for s in summaries])
    # Sampled convergence curves
    sampled_specs = [
        ("convergence_2d_sampled.csv", res2, "layered_magnitude"),
        ("convergence_3d_sampled.csv", res3, "layered_magnitude"),
        ("convergence_3d_layered_sampled.csv", res3_far, "layered_magnitude"),
        ("convergence_3d_recovery_sampled.csv", res3_recovery, "layered_magnitude"),
        ("convergence_3d_h5_recovery_sampled.csv", res3_h5, "layered_magnitude"),
        ("convergence_3d_hv_recovery_sampled.csv", res3_recovery_hv, "layered_hypervolume"),
        ("convergence_3d_h5_hv_recovery_sampled.csv", res3_h5_hv, "layered_hypervolume"),
    ]
    for filename, res, value_name in sampled_specs:
        write_csv(outdir / filename, ["iteration", value_name, "layer_sizes", "stochastic_recovery_accepted"],
                  [(i, f"{res['values'][i]:.10f}", layer_string(res["layers"][i]), res.get("recovery_flags", [0]*len(res["values"]))[i]) for i in sample_indices(len(res["values"]), 20)])
    write_csv(outdir / "convergence_3d_recovery_events.csv", ["case", "indicator", "iteration", "indicator_value", "layer_sizes"],
              [("H4 recovery", "magnitude", i, f"{res3_recovery['values'][i]:.10f}", layer_string(res3_recovery["layers"][i])) for i, flag in enumerate(res3_recovery["recovery_flags"]) if flag] +
              [("H5 recovery", "magnitude", i, f"{res3_h5['values'][i]:.10f}", layer_string(res3_h5["layers"][i])) for i, flag in enumerate(res3_h5["recovery_flags"]) if flag] +
              [("H4 recovery", "hypervolume", i, f"{res3_recovery_hv['values'][i]:.10f}", layer_string(res3_recovery_hv["layers"][i])) for i, flag in enumerate(res3_recovery_hv["recovery_flags"]) if flag] +
              [("H5 recovery", "hypervolume", i, f"{res3_h5_hv['values'][i]:.10f}", layer_string(res3_h5_hv["layers"][i])) for i, flag in enumerate(res3_h5_hv["recovery_flags"]) if flag])
    # Initial/final sets
    initial_final_specs = [
        ("convergence_2d", res2, ["f1", "f2"]),
        ("convergence_3d", res3, ["f1", "f2", "f3"]),
        ("convergence_3d_layered", res3_far, ["f1", "f2", "f3"]),
        ("convergence_3d_recovery", res3_recovery, ["f1", "f2", "f3"]),
        ("convergence_3d_h5_recovery", res3_h5, ["f1", "f2", "f3"]),
        ("convergence_3d_hv_recovery", res3_recovery_hv, ["f1", "f2", "f3"]),
        ("convergence_3d_h5_hv_recovery", res3_h5_hv, ["f1", "f2", "f3"]),
    ]
    for prefix, res, hdr in initial_final_specs:
        write_csv(outdir / f"{prefix}_initial_objectives.csv", hdr, res["hist_Y"][0])
        write_csv(outdir / f"{prefix}_final_objectives.csv", hdr, res["hist_Y"][-1])
    # Layer table for manuscript
    table_rows = []
    for label, res, mu, criterion in [
        ("2-D curved front", res2, 10, "Mag"),
        ("3-D supersphere", res3, 15, "Mag"),
        ("3-D layered box", res3_far, 15, "Mag"),
        ("3-D recovery box", res3_recovery, 15, "Mag"),
        ("3-D H=5 recovery", res3_h5, 21, "Mag"),
        ("3-D HV recovery box", res3_recovery_hv, 15, "HV"),
        ("3-D HV H=5 recovery", res3_h5_hv, 21, "HV"),
    ]:
        for i in sample_indices(len(res["values"]), 20):
            table_rows.append((label, criterion, mu, i, f"{res['values'][i]:.6f}", layer_string(res["layers"][i])))
    write_csv(outdir / "convergence_layer_table.csv", ["case", "criterion", "mu", "iteration", "indicator_value", "layer_sizes"], table_rows)
    tex_rows = [
        r"\begin{tabular}{llrrrl}",
        r"\hline",
        r"case & crit. & $\mu$ & iter. & value & layer sizes \\",
        r"\hline",
    ]
    for label, criterion, mu, it, val, layers in table_rows:
        tex_rows.append(f"{label} & {criterion} & {mu} & {it} & {val} & ${layers}$ " + r"\\")
    tex_rows.extend([r"\hline", r"\end{tabular}"])
    (outdir / "convergence_layer_table.tex").write_text("\n".join(tex_rows) + "\n", encoding="utf-8")
    write_tex_snippets(outdir, res2, res3, res3_far, res3_recovery, res3_h5, res3_recovery_hv, res3_h5_hv)


def _timed(label: str, fn, *args, **kwargs) -> dict:
    start = time.process_time()
    result = fn(*args, **kwargs)
    result["cpu_seconds"] = time.process_time() - start
    result["case_label"] = label
    return result


def run_experiments() -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    res2 = _timed(
        "curved_2d",
        run_projected_ascent,
        curved2d_initial_points(), curved2d_objective, curved2d_jacobian, project_2d_box,
        max_iter=50, alpha0=0.035, eps_layer=1e-3, tau=3e-4, sigma=0.045,
    )
    res3 = _timed(
        "supersphere_3d_gamma1_H4",
        run_projected_ascent,
        supersphere_initial_points(H=4, seed=8, sigma=0.01),
        supersphere_gamma1_objective, supersphere_gamma1_jacobian, project_simplex_rows,
        max_iter=70, alpha0=0.040, eps_layer=1e-3, tau=5e-4, sigma=0.040,
    )
    res3_far = _timed(
        "supersphere_3d_layered_box_gamma1",
        run_projected_ascent,
        supersphere_box_layered_initial_points(seed=0, n_points=15),
        supersphere_gamma1_box_objective, supersphere_gamma1_jacobian, project_3d_layered_box,
        max_iter=45, alpha0=0.040, eps_layer=1e-3, tau=5e-4, sigma=0.060,
        shrink=0.65, max_retries=8, alpha_boost=1.00,
    )
    recovery_kwargs = dict(
        max_iter=500, alpha0=0.020, eps_layer=1e-3, tau=5e-4, sigma=0.060,
        shrink=0.65, max_retries=6, alpha_boost=1.00,
        stagnation_window=10, stagnation_tol=5.0e-3,
        stochastic_step=0.160, stochastic_trials=3, stochastic_seed=31,
        perturb_fraction=0.45, perturb_cooldown=10, max_backtrack_drop=0.12, final_unperturbed_iters=10,
    )
    res3_recovery = _timed(
        "supersphere_3d_recovery_box_gamma1",
        run_projected_ascent_with_stochastic_recovery,
        supersphere_box_layered_initial_points(seed=0, n_points=15),
        supersphere_gamma1_box_objective, supersphere_gamma1_jacobian, project_3d_layered_box,
        indicator="magnitude", **recovery_kwargs,
    )
    res3_h5 = _timed(
        "supersphere_3d_H5_recovery_box_gamma1",
        run_projected_ascent_with_stochastic_recovery,
        supersphere_box_h_budget_initial_points(H=5, seed=0),
        supersphere_gamma1_box_objective, supersphere_gamma1_jacobian, project_3d_layered_box,
        indicator="magnitude", **recovery_kwargs,
    )
    res3_recovery_hv = _timed(
        "supersphere_3d_HV_recovery_box_gamma1",
        run_projected_ascent_with_stochastic_recovery,
        supersphere_box_layered_initial_points(seed=0, n_points=15),
        supersphere_gamma1_box_objective, supersphere_gamma1_jacobian, project_3d_layered_box,
        indicator="hypervolume", **recovery_kwargs,
    )
    res3_h5_hv = _timed(
        "supersphere_3d_HV_H5_recovery_box_gamma1",
        run_projected_ascent_with_stochastic_recovery,
        supersphere_box_h_budget_initial_points(H=5, seed=0),
        supersphere_gamma1_box_objective, supersphere_gamma1_jacobian, project_3d_layered_box,
        indicator="hypervolume", **recovery_kwargs,
    )
    return res2, res3, res3_far, res3_recovery, res3_h5, res3_recovery_hv, res3_h5_hv


def write_readme(path: Path) -> None:
    text = """# Convergence behaviour reproduction

This directory reproduces the section **Convergence behaviour** added to the
layered magnitude report.

## Files

- `reproduce_convergence_behaviour_v17_2026-05-09.py`: self-contained Python
  script with the 2-D and 3-D benchmarks, layered magnitude and hypervolume
  computations, nondomination layering, projected ascent method, and export
  routines.
- `convergence_behaviour_summary.csv`: initial/final indicator growth summary.
- `convergence_*_sampled.csv`: sampled coordinates used for the convergence
  curves in the report.
- `convergence_layer_table.csv/.tex`: layer-size snapshots used in the text.
- `convergence_behaviour_coordinates.tex`: PGFPlots coordinate macros used in
  the manuscript patch.

## How to run

```bash
python reproduce_convergence_behaviour_v17_2026-05-09.py --outdir convergence_outputs
```

The script depends on NumPy and optionally uses Numba acceleration when available.  The recovery runs use 500 episodes.  The plotted curves and sampled tables use every 20th iteration/episode and include the endpoint.  The magnitude-recovery and hypervolume-recovery runs use the same mild starting clouds and the same multi-point backtracking perturbation rule; perturbations are disabled in the final 10 episodes.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("convergence_behaviour_outputs"))
    args = parser.parse_args()
    res2, res3, res3_far, res3_recovery, res3_h5, res3_recovery_hv, res3_h5_hv = run_experiments()
    write_outputs(args.outdir, res2, res3, res3_far, res3_recovery, res3_h5, res3_recovery_hv, res3_h5_hv)
    write_readme(args.outdir / "README.md")
    for summary in [
        summarize_case("curved_2d", res2, 10),
        summarize_case("supersphere_3d_gamma1_H4", res3, 15),
        summarize_case("supersphere_3d_layered_box_gamma1", res3_far, 15),
        summarize_case("supersphere_3d_recovery_box_gamma1", res3_recovery, 15),
        summarize_case("supersphere_3d_H5_recovery_box_gamma1", res3_h5, 21),
        summarize_case("supersphere_3d_HV_recovery_box_gamma1", res3_recovery_hv, 15),
        summarize_case("supersphere_3d_HV_H5_recovery_box_gamma1", res3_h5_hv, 21),
    ]:
        print(summary)
    print(f"Wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
