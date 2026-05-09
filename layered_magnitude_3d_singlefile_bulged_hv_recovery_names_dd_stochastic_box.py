#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

Array = np.ndarray

# ---------------------------- exact small-front reference ----------------------------

def _as_array(points, dim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2:
        raise ValueError("points must be 2D array-like")
    if dim is not None and arr.shape[1] != dim:
        raise ValueError(f"expected dimension {dim}, got {arr.shape[1]}")
    return arr


def _translate_and_validate(points, anchor, dim: Optional[int] = None) -> np.ndarray:
    pts = _as_array(points, dim=dim)
    a = np.asarray(anchor, dtype=float)
    if a.ndim != 1 or a.shape[0] != pts.shape[1]:
        raise ValueError("anchor dimension mismatch")
    q = pts - a
    if np.any(q < -1e-12):
        raise ValueError("all points must weakly dominate the anchor")
    q[q < 0.0] = 0.0
    return q


def _prod_except(mins: np.ndarray, k: int) -> float:
    prod = 1.0
    for idx, val in enumerate(mins):
        if idx != k:
            prod *= float(val)
    return prod


def exact_hypervolume_max(points, anchor) -> float:
    q = _translate_and_validate(points, anchor)
    n, d = q.shape
    total = 0.0
    for r in range(1, n + 1):
        sign = 1.0 if (r % 2 == 1) else -1.0
        for subset in combinations(range(n), r):
            mins = np.min(q[list(subset), :], axis=0)
            total += sign * float(np.prod(mins))
    return total


def exact_hypervolume_gradient_max(points, anchor) -> np.ndarray:
    q = _translate_and_validate(points, anchor)
    n, d = q.shape
    grad = np.zeros((n, d), dtype=float)
    for r in range(1, n + 1):
        sign = 1.0 if (r % 2 == 1) else -1.0
        for subset in combinations(range(n), r):
            mins = np.min(q[list(subset), :], axis=0)
            for k in range(d):
                min_val = mins[k]
                tied = [idx for idx in subset if abs(q[idx, k] - min_val) <= 1e-12]
                if not tied:
                    continue
                coeff = sign * _prod_except(mins, k) / float(len(tied))
                for idx in tied:
                    grad[idx, k] += coeff
    return grad


def _axis_max_gradient_ties_shared(q: np.ndarray) -> np.ndarray:
    n, d = q.shape
    g = np.zeros((n, d), dtype=float)
    for k in range(d):
        m = float(np.max(q[:, k]))
        tied = np.where(np.abs(q[:, k] - m) <= 1e-12)[0]
        if tied.size:
            g[tied, k] += 1.0 / tied.size
    return g


def magnitude_3d_max_exact(points, anchor=(0.0, 0.0, 0.0)) -> float:
    q = _translate_and_validate(points, anchor, dim=3)
    lx = float(np.max(q[:, 0])) if len(q) else 0.0
    ly = float(np.max(q[:, 1])) if len(q) else 0.0
    lz = float(np.max(q[:, 2])) if len(q) else 0.0
    area_xy = exact_hypervolume_max(q[:, [0, 1]], (0.0, 0.0))
    area_xz = exact_hypervolume_max(q[:, [0, 2]], (0.0, 0.0))
    area_yz = exact_hypervolume_max(q[:, [1, 2]], (0.0, 0.0))
    hv3 = exact_hypervolume_max(q, (0.0, 0.0, 0.0))
    return 1.0 + 0.5 * (lx + ly + lz) + 0.25 * (area_xy + area_xz + area_yz) + 0.125 * hv3


def magnitude_gradient_3d_max_exact(points, anchor=(0.0, 0.0, 0.0)) -> np.ndarray:
    q = _translate_and_validate(points, anchor, dim=3)
    n = q.shape[0]
    axis_grad = _axis_max_gradient_ties_shared(q)
    hv3_grad = exact_hypervolume_gradient_max(q, (0.0, 0.0, 0.0))
    hv_xy = exact_hypervolume_gradient_max(q[:, [0, 1]], (0.0, 0.0))
    hv_xz = exact_hypervolume_gradient_max(q[:, [0, 2]], (0.0, 0.0))
    hv_yz = exact_hypervolume_gradient_max(q[:, [1, 2]], (0.0, 0.0))
    proj_grad = np.zeros((n, 3), dtype=float)
    proj_grad[:, 0] += hv_xy[:, 0]
    proj_grad[:, 1] += hv_xy[:, 1]
    proj_grad[:, 0] += hv_xz[:, 0]
    proj_grad[:, 2] += hv_xz[:, 1]
    proj_grad[:, 1] += hv_yz[:, 0]
    proj_grad[:, 2] += hv_yz[:, 1]
    return 0.5 * axis_grad + 0.25 * proj_grad + 0.125 * hv3_grad


# ---------------------------- sweep-based value and forward gradient ----------------------------

def _nondominated_2d(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    pts = np.asarray(points, dtype=float)
    order = np.argsort(pts[:, 0], kind='mergesort')
    pts = pts[order]
    out = []
    max_y = -np.inf
    # scan descending x to keep maxima in y, then reverse
    for p in pts[::-1]:
        y = p[1]
        if y > max_y + 1e-15:
            out.append(p)
            max_y = y
    out = np.array(out[::-1], dtype=float)
    return out


def hv2_value_sweep(points2d: np.ndarray) -> float:
    pts = _nondominated_2d(points2d)
    if len(pts) == 0:
        return 0.0
    area = 0.0
    x_prev = 0.0
    for x, y in pts:
        area += (float(x) - x_prev) * float(y)
        x_prev = float(x)
    return area


def hv3_value_sweep(points3d: np.ndarray) -> float:
    pts = np.asarray(points3d, dtype=float)
    if len(pts) == 0:
        return 0.0
    xs = np.unique(pts[:, 0])
    xs.sort()
    total = 0.0
    x_prev = 0.0
    for x in xs:
        active = pts[pts[:, 0] >= x - 1e-15][:, [1, 2]]
        area = hv2_value_sweep(active)
        total += (float(x) - x_prev) * area
        x_prev = float(x)
    return total


def _exclusive_length_1d(t: float, active_ts: List[float]) -> float:
    covered = max(active_ts) if active_ts else 0.0
    return max(0.0, float(t) - float(covered))


def _exclusive_area_2d(y: float, z: float, active_rects: List[Tuple[float, float]]) -> float:
    if not active_rects:
        return float(y) * float(z)
    clipped = np.array([(min(float(a), float(y)), min(float(b), float(z))) for a, b in active_rects], dtype=float)
    clipped = clipped[(clipped[:, 0] > 0.0) & (clipped[:, 1] > 0.0)]
    covered = hv2_value_sweep(clipped) if len(clipped) else 0.0
    return max(0.0, float(y) * float(z) - covered)


def axis_gradient_forward(q: np.ndarray) -> np.ndarray:
    n, d = q.shape
    g = np.zeros((n, d), dtype=float)
    for k in range(d):
        order = sorted(range(n), key=lambda i: (q[i, k], i))
        g[order[-1], k] = 1.0
    return g


def hv2_gradient_forward(points2d: np.ndarray) -> np.ndarray:
    pts = np.asarray(points2d, dtype=float)
    n = len(pts)
    g = np.zeros((n, 2), dtype=float)
    # x-derivatives: sweep descending in x, ties by index
    active_y: List[float] = []
    for i in sorted(range(n), key=lambda i: (pts[i, 0], i), reverse=True):
        g[i, 0] = _exclusive_length_1d(pts[i, 1], active_y)
        active_y.append(float(pts[i, 1]))
    # y-derivatives: sweep descending in y, ties by index
    active_x: List[float] = []
    for i in sorted(range(n), key=lambda i: (pts[i, 1], i), reverse=True):
        g[i, 1] = _exclusive_length_1d(pts[i, 0], active_x)
        active_x.append(float(pts[i, 0]))
    return g


def hv3_gradient_forward(points3d: np.ndarray) -> np.ndarray:
    pts = np.asarray(points3d, dtype=float)
    n = len(pts)
    g = np.zeros((n, 3), dtype=float)
    # x-derivatives: sweep descending x, active yz-rectangles from larger x (with symbolic tie break by index)
    active_yz: List[Tuple[float, float]] = []
    for i in sorted(range(n), key=lambda i: (pts[i, 0], i), reverse=True):
        g[i, 0] = _exclusive_area_2d(pts[i, 1], pts[i, 2], active_yz)
        active_yz.append((float(pts[i, 1]), float(pts[i, 2])))
    # y-derivatives
    active_xz: List[Tuple[float, float]] = []
    for i in sorted(range(n), key=lambda i: (pts[i, 1], i), reverse=True):
        g[i, 1] = _exclusive_area_2d(pts[i, 0], pts[i, 2], active_xz)
        active_xz.append((float(pts[i, 0]), float(pts[i, 2])))
    # z-derivatives
    active_xy: List[Tuple[float, float]] = []
    for i in sorted(range(n), key=lambda i: (pts[i, 2], i), reverse=True):
        g[i, 2] = _exclusive_area_2d(pts[i, 0], pts[i, 1], active_xy)
        active_xy.append((float(pts[i, 0]), float(pts[i, 1])))
    return g


def magnitude_3d_max_sweep_forward(points, anchor=(0.0, 0.0, 0.0)) -> float:
    q = _translate_and_validate(points, anchor, dim=3)
    lx, ly, lz = map(float, np.max(q, axis=0)) if len(q) else (0.0, 0.0, 0.0)
    area_xy = hv2_value_sweep(q[:, [0, 1]])
    area_xz = hv2_value_sweep(q[:, [0, 2]])
    area_yz = hv2_value_sweep(q[:, [1, 2]])
    hv3 = hv3_value_sweep(q)
    return 1.0 + 0.5 * (lx + ly + lz) + 0.25 * (area_xy + area_xz + area_yz) + 0.125 * hv3


def magnitude_gradient_3d_max_sweep_forward(points, anchor=(0.0, 0.0, 0.0)) -> np.ndarray:
    q = _translate_and_validate(points, anchor, dim=3)
    n = len(q)
    if n == 0:
        return np.zeros((0, 3), dtype=float)
    axis_g = axis_gradient_forward(q)
    hv3_g = hv3_gradient_forward(q)
    g_xy = hv2_gradient_forward(q[:, [0, 1]])
    g_xz = hv2_gradient_forward(q[:, [0, 2]])
    g_yz = hv2_gradient_forward(q[:, [1, 2]])
    proj_g = np.zeros((n, 3), dtype=float)
    proj_g[:, 0] += g_xy[:, 0]
    proj_g[:, 1] += g_xy[:, 1]
    proj_g[:, 0] += g_xz[:, 0]
    proj_g[:, 2] += g_xz[:, 1]
    proj_g[:, 1] += g_yz[:, 0]
    proj_g[:, 2] += g_yz[:, 1]
    return 0.5 * axis_g + 0.25 * proj_g + 0.125 * hv3_g


def exact_gradient_with_index_perturbation(points, anchor=(0.0, 0.0, 0.0), delta: float = 1e-9) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    n = len(pts)
    idx = np.arange(n, dtype=float)[:, None]
    pts = pts + delta * idx
    return magnitude_gradient_3d_max_exact(pts, anchor)


# ---------------------------- optimization code ----------------------------

def nondomination_layers(Y: Array) -> List[List[int]]:
    remaining = list(range(len(Y)))
    layers: List[List[int]] = []
    while remaining:
        front: List[int] = []
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


def nondominated_subset(Y: Array) -> Array:
    return Y[nondomination_layers(Y)[0]].copy() if len(Y) else Y.copy()


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
            e = np.exp(-inv * float(np.dot(diff, diff)))
            c = 2.0 * inv * e * diff
            G[i] += c
            G[j] -= c
    return G


def normalize_rows_local(G: Array) -> Array:
    H = np.array(G, dtype=float, copy=True)
    norms = np.linalg.norm(H, axis=1)
    mask = norms > 1.0e-14
    H[mask] = H[mask] / norms[mask][:, None]
    return H


def hypervolume_3d_max_exact(points, anchor=(0.0, 0.0, 0.0)) -> float:
    return exact_hypervolume_max(points, anchor)


def hypervolume_gradient_3d_max_exact(points, anchor=(0.0, 0.0, 0.0)) -> np.ndarray:
    return exact_hypervolume_gradient_max(points, anchor)


def hypervolume_3d_max_sweep_forward(points, anchor=(0.0, 0.0, 0.0)) -> float:
    q = _translate_and_validate(points, anchor, dim=3)
    return hv3_value_sweep(q)


def hypervolume_gradient_3d_max_sweep_forward(points, anchor=(0.0, 0.0, 0.0)) -> np.ndarray:
    q = _translate_and_validate(points, anchor, dim=3)
    return hv3_gradient_forward(q)


def indicator_value_and_gradient_front(points, anchor, indicator='magnitude', exact_front_threshold: int = 10):
    if indicator == 'magnitude':
        if len(points) <= exact_front_threshold:
            return magnitude_3d_max_exact(points, anchor=anchor), magnitude_gradient_3d_max_exact(points, anchor=anchor), 'exact-mag'
        else:
            return magnitude_3d_max_sweep_forward(points, anchor=anchor), magnitude_gradient_3d_max_sweep_forward(points, anchor=anchor), 'sweep-mag'
    elif indicator == 'hypervolume':
        if len(points) <= exact_front_threshold:
            return hypervolume_3d_max_exact(points, anchor=anchor), hypervolume_gradient_3d_max_exact(points, anchor=anchor), 'exact-hv'
        else:
            return hypervolume_3d_max_sweep_forward(points, anchor=anchor), hypervolume_gradient_3d_max_sweep_forward(points, anchor=anchor), 'sweep-hv'
    else:
        raise ValueError(f'unknown indicator: {indicator}')


def indicator_value_front(points, anchor, indicator='magnitude', exact_front_threshold: int = 10):
    if indicator == 'magnitude':
        if len(points) <= exact_front_threshold:
            return magnitude_3d_max_exact(points, anchor=anchor), 'exact-mag'
        else:
            return magnitude_3d_max_sweep_forward(points, anchor=anchor), 'sweep-mag'
    elif indicator == 'hypervolume':
        if len(points) <= exact_front_threshold:
            return hypervolume_3d_max_exact(points, anchor=anchor), 'exact-hv'
        else:
            return hypervolume_3d_max_sweep_forward(points, anchor=anchor), 'sweep-hv'
    else:
        raise ValueError(f'unknown indicator: {indicator}')


def layered_value_obj(Y: Array, anchor: Sequence[float], eps_layer: float, tau: float, sigma: float, exact_front_threshold: int = 10, indicator: str = 'magnitude'):
    value = 0.0
    layers = nondomination_layers(Y)
    modes = []
    for ell, front in enumerate(layers):
        w = eps_layer ** ell
        pts = Y[front]
        vfront, mode = indicator_value_front(pts, anchor, indicator=indicator, exact_front_threshold=exact_front_threshold)
        value += w * vfront
        modes.append(mode)
    if tau > 0:
        value -= tau * repulsion_value(Y, sigma)
    return float(value), layers, modes


def layered_value_and_gradient_obj(Y: Array, anchor: Sequence[float], eps_layer: float, tau: float, sigma: float, exact_front_threshold: int = 10, indicator: str = 'magnitude'):
    value = 0.0
    G = np.zeros_like(Y)
    layers = nondomination_layers(Y)
    modes = []
    for ell, front in enumerate(layers):
        w = eps_layer**ell
        pts = Y[front]
        vfront, gfront, mode = indicator_value_and_gradient_front(pts, anchor, indicator=indicator, exact_front_threshold=exact_front_threshold)
        value += w * vfront
        modes.append(mode)
        for loc, idx in enumerate(front):
            G[idx] += w * gfront[loc]
    if tau > 0:
        value -= tau * repulsion_value(Y, sigma)
        G -= tau * repulsion_gradient(Y, sigma)
    return float(value), G, layers, modes


def igd(approx: Array, reference: Array) -> float:
    if len(approx) == 0 or len(reference) == 0:
        return float('inf')
    return float(np.mean([np.sqrt(np.min(np.sum((approx - r) ** 2, axis=1))) for r in reference]))


def save_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[float]]):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(list(header))
        w.writerows(list(rows))


def plot_objective_space(path: Path, Y0: Array, Yf: Array, ref: Array, title: str, elev: float = 28, azim: float = 45):
    fig = plt.figure(figsize=(10, 4.6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    for ax, Y, ttl in ((ax1, Y0, 'Initial'), (ax2, Yf, 'Final')):
        if len(ref):
            ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], s=4, alpha=0.14, depthshade=False)
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=34, depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('obj 1')
        ax.set_ylabel('obj 2')
        ax.set_zlabel('obj 3')
        ax.set_title(ttl)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_three_peaks_decision_space(path: Path, X0: Array, Xf: Array, title: str, elev: float = 28, azim: float = 45):
    fig = plt.figure(figsize=(10, 4.6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    E = np.eye(3)
    for ax, X, ttl in ((ax1, X0, 'Initial'), (ax2, Xf, 'Final')):
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=34, depthshade=False)
        ax.scatter(E[:, 0], E[:, 1], E[:, 2], s=46, marker='^', depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_zlim(-2.0, 2.0)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        ax.set_title(ttl)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_convergence(path: Path, values, alphas, title, stride=2):
    idx = np.arange(len(values))[:: max(1, stride)]
    v = np.asarray(values)[idx]
    a = np.asarray(alphas)[idx]
    fig = plt.figure(figsize=(7, 4.2))
    ax1 = fig.add_subplot(111)
    ax1.plot(idx, v, marker='o', markersize=2.2, linewidth=1.2)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('layered value')
    ax1.set_title(title)
    ax2 = ax1.twinx()
    ax2.plot(idx, a, '--', linewidth=1.0)
    ax2.set_ylabel('step size')
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


@dataclass
class RunResult:
    X0: Array
    Y0: Array
    Xf: Array
    Yf: Array
    values: list
    alphas: list
    accepted_steps: int
    completed_iterations: int


def run_projected_ascent(
    objective_fn,
    jacobian_fn,
    projector,
    X0,
    anchor,
    eps_layer=1e-3,
    tau=5e-4,
    sigma=0.03,
    alpha0=0.05,
    max_iter=200,
    normalize_per_point=True,
    shrink=0.99,
    max_retries=8,
    alpha_floor=1e-4,
    stall_limit=30,
    exact_front_threshold=10,
    progress_every=10,
    quiet=False,
    indicator='magnitude',
    recovery_patience=8,
    recovery_boost=2.5,
    recovery_cap=0.02,
):
    X = projector(np.asarray(X0, float).copy())
    Y = objective_fn(X)
    val, GY, layers, modes = layered_value_and_gradient_obj(Y, anchor, eps_layer, tau, sigma, exact_front_threshold=exact_front_threshold, indicator=indicator)
    values = [val]
    alphas = [alpha0]
    alpha = float(alpha0)
    accepted_steps = 0
    consecutive_stalls = 0
    Xinit = X.copy()
    Yinit = Y.copy()

    if not quiet:
        print(f"iter=0/{max_iter} value={val:.10f} alpha={alpha:.6g} nd={len(layers[0]) if layers else 0} layer_sizes={[len(fr) for fr in layers[:5]]} modes={'+'.join(modes[:5])}")

    completed = 0
    for it in range(1, max_iter + 1):
        J = jacobian_fn(X)
        GX = np.einsum('nij,ni->nj', J, GY)
        if normalize_per_point:
            GX = normalize_rows_local(GX)
        trial_alpha = alpha
        accepted = False
        retries_used = 0
        for _k in range(max_retries + 1):
            trial_alpha = max(trial_alpha, alpha_floor)
            Xt = projector(X + trial_alpha * GX)
            Yt = objective_fn(Xt)
            vt, GYt, layers_t, modes_t = layered_value_and_gradient_obj(Yt, anchor, eps_layer, tau, sigma, exact_front_threshold=exact_front_threshold, indicator=indicator)
            if vt >= val - 1e-12:
                X, Y, val, GY, layers, modes = Xt, Yt, vt, GYt, layers_t, modes_t
                alpha = trial_alpha
                accepted_steps += 1
                accepted = True
                consecutive_stalls = 0
                break
            if trial_alpha <= alpha_floor + 1e-15:
                break
            trial_alpha *= shrink
            retries_used += 1
        recovered = False
        if not accepted:
            alpha = max(alpha_floor, trial_alpha)
            consecutive_stalls += 1
            if alpha <= alpha_floor + 1e-15 and consecutive_stalls >= recovery_patience:
                new_alpha = min(recovery_cap, max(alpha_floor, recovery_boost * alpha))
                if new_alpha > alpha + 1e-15:
                    alpha = new_alpha
                    consecutive_stalls = 0
                    recovered = True
        values.append(val)
        alphas.append(alpha)
        completed = it
        if (not quiet) and (it % progress_every == 0 or it == max_iter or not accepted):
            status = 'accepted' if accepted else ('recovered' if recovered else 'stalled')
            print(f"iter={it}/{max_iter} value={val:.10f} alpha={alpha:.6g} nd={len(layers[0]) if layers else 0} layer_sizes={[len(fr) for fr in layers[:5]]} modes={'+'.join(modes[:5])} retries={retries_used} {status} accepted_steps={accepted_steps}")
        if consecutive_stalls >= stall_limit:
            if not quiet:
                print(f"Stopping early after {consecutive_stalls} consecutive stalls at alpha floor {alpha_floor}.")
            break
    return RunResult(Xinit, Yinit, X, Y, values, alphas, accepted_steps, completed)


def run_stochastic_hillclimb(
    objective_fn,
    projector,
    X0,
    anchor,
    rng,
    eps_layer=1e-3,
    tau=5e-4,
    sigma=0.03,
    alpha0=0.05,
    max_iter=200,
    shrink=0.99,
    max_retries=8,
    alpha_floor=1e-4,
    stall_limit=30,
    exact_front_threshold=10,
    progress_every=10,
    quiet=False,
    indicator='magnitude',
    recovery_patience=8,
    recovery_boost=2.5,
    recovery_cap=0.02,
):
    X = projector(np.asarray(X0, float).copy())
    Y = objective_fn(X)
    val, layers, modes = layered_value_obj(Y, anchor, eps_layer, tau, sigma, exact_front_threshold=exact_front_threshold, indicator=indicator)
    values = [val]
    alphas = [alpha0]
    alpha = float(alpha0)
    accepted_steps = 0
    consecutive_stalls = 0
    Xinit = X.copy()
    Yinit = Y.copy()
    if not quiet:
        print(f"iter=0/{max_iter} value={val:.10f} alpha={alpha:.6g} nd={len(layers[0]) if layers else 0} layer_sizes={[len(fr) for fr in layers[:5]]} modes={'+'.join(modes[:5])} move=stochastic")
    completed = 0
    for it in range(1, max_iter + 1):
        trial_alpha = alpha
        accepted = False
        retries_used = 0
        for _k in range(max_retries + 1):
            trial_alpha = max(trial_alpha, alpha_floor)
            idx = int(rng.integers(0, len(X)))
            direction = rng.normal(size=X.shape[1])
            nrm = float(np.linalg.norm(direction))
            if nrm <= 1e-15:
                direction[0] = 1.0
                nrm = 1.0
            direction /= nrm
            Xt = np.array(X, copy=True)
            Xt[idx] = Xt[idx] + trial_alpha * direction
            Xt = projector(Xt)
            Yt = objective_fn(Xt)
            vt, layers_t, modes_t = layered_value_obj(Yt, anchor, eps_layer, tau, sigma, exact_front_threshold=exact_front_threshold, indicator=indicator)
            if vt >= val - 1e-12:
                X, Y, val, layers, modes = Xt, Yt, vt, layers_t, modes_t
                alpha = trial_alpha
                accepted_steps += 1
                accepted = True
                consecutive_stalls = 0
                break
            if trial_alpha <= alpha_floor + 1e-15:
                break
            trial_alpha *= shrink
            retries_used += 1
        recovered = False
        if not accepted:
            alpha = max(alpha_floor, trial_alpha)
            consecutive_stalls += 1
            if alpha <= alpha_floor + 1e-15 and consecutive_stalls >= recovery_patience:
                new_alpha = min(recovery_cap, max(alpha_floor, recovery_boost * alpha))
                if new_alpha > alpha + 1e-15:
                    alpha = new_alpha
                    consecutive_stalls = 0
                    recovered = True
        values.append(val)
        alphas.append(alpha)
        completed = it
        if (not quiet) and (it % progress_every == 0 or it == max_iter or not accepted):
            status = 'accepted' if accepted else ('recovered' if recovered else 'stalled')
            print(f"iter={it}/{max_iter} value={val:.10f} alpha={alpha:.6g} nd={len(layers[0]) if layers else 0} layer_sizes={[len(fr) for fr in layers[:5]]} modes={'+'.join(modes[:5])} retries={retries_used} {status} accepted_steps={accepted_steps} move=stochastic")
        if consecutive_stalls >= stall_limit:
            if not quiet:
                print(f"Stopping early after {consecutive_stalls} consecutive stalls at alpha floor {alpha_floor}.")
            break
    return RunResult(Xinit, Yinit, X, Y, values, alphas, accepted_steps, completed)


# ---------------------------- benchmarks ----------------------------

def three_peaks_objective(X: Array) -> Array:
    E = np.eye(3)
    Y = np.zeros((len(X), 3))
    for i, e in enumerate(E):
        Y[:, i] = 1.0 - np.linalg.norm(X - e, axis=1)
    return Y


def three_peaks_jacobian(X: Array) -> Array:
    E = np.eye(3)
    J = np.zeros((len(X), 3, 3))
    for i, e in enumerate(E):
        diff = X - e
        norms = np.linalg.norm(diff, axis=1)
        safe = norms > 1e-14
        J[safe, i, :] = -diff[safe] / norms[safe][:, None]
    return J


def project_simplex_rows(V: Array) -> Array:
    V = np.asarray(V, dtype=float)
    if V.ndim != 2:
        raise ValueError("V must be a 2D array")
    U = -np.sort(-V, axis=1)
    cssv = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, V.shape[1] + 1)
    cond = U - cssv / ind > 0
    rho = cond.sum(axis=1) - 1
    theta = cssv[np.arange(V.shape[0]), rho] / (rho + 1)
    return np.maximum(V - theta[:, None], 0.0)


def sample_simplex(rng: np.random.Generator, n: int, dim: int = 3) -> Array:
    return rng.dirichlet(np.ones(dim), size=n)


def das_dennis_simplex_grid_3d(H: int) -> Array:
    if H < 1:
        raise ValueError('dd_h must be >= 1')
    pts = []
    for a in range(H + 1):
        for b in range(H + 1 - a):
            c = H - a - b
            pts.append((a / H, b / H, c / H))
    return np.array(pts, dtype=float)


def perturb_and_reproject_simplex(X: Array, rng: np.random.Generator, sigma: float) -> Array:
    if sigma <= 0:
        return np.array(X, dtype=float, copy=True)
    Y = np.asarray(X, dtype=float) + rng.normal(0.0, sigma, size=np.asarray(X).shape)
    return project_simplex_rows(Y)


def make_simplex_initial_points(rng: np.random.Generator, init_mode: str = 'random', n_points: int = 15, dd_h: int = 3, dd_sigma: float = 0.01) -> Array:
    if init_mode == 'dasdenis':
        X0 = das_dennis_simplex_grid_3d(dd_h)
        return perturb_and_reproject_simplex(X0, rng, dd_sigma)
    return sample_simplex(rng, n_points, 3)


def bulged_three_peaks_objective(X: Array, gamma: float = 0.5) -> Array:
    E = np.eye(3)
    Y = np.zeros((len(X), 3), dtype=float)
    for i, e in enumerate(E):
        d2 = np.sum((X - e) ** 2, axis=1)
        t = np.clip(d2 / 2.0, 0.0, 1.0)
        Y[:, i] = 1.0 - np.power(t, gamma)
    return Y


def bulged_three_peaks_jacobian(X: Array, gamma: float = 0.5) -> Array:
    E = np.eye(3)
    J = np.zeros((len(X), 3, 3), dtype=float)
    for i, e in enumerate(E):
        diff = X - e
        d2 = np.sum(diff ** 2, axis=1)
        t = np.clip(d2 / 2.0, 0.0, 1.0)
        coeff = np.zeros(len(X), dtype=float)
        mask = t > 1e-12
        coeff[mask] = -gamma * np.power(t[mask], gamma - 1.0)
        J[:, i, :] = coeff[:, None] * diff
    return J


def approx_reference_simplex(objective_fn, samples: int, rng: np.random.Generator) -> Array:
    X = sample_simplex(rng, samples, 3)
    return nondominated_subset(objective_fn(X))


DEFAULT_RUN_SETTINGS = {
    'seed': 8,
    'n_points': 15,
    'three_peaks_iters': 200,
    'crash_iters': 96,
    'exact_front_threshold': 10,
    'progress_every': 10,
    'bulge_gamma': 0.5,
    'indicator': 'magnitude',
    'initialization': 'random',
    'dd_h': 3,
    'dd_sigma': 0.01,
    'move': 'gradient',
}

OPTION_TAGS = {
    'seed': 'se',
    'n_points': 'np',
    'three_peaks_iters': 'tp',
    'crash_iters': 'cr',
    'exact_front_threshold': 'ex',
    'bulge_gamma': 'bu',
    'indicator': 'in',
    'initialization': 'ii',
    'dd_h': 'dh',
    'dd_sigma': 'ds',
    'move': 'mv',
}


def _fmt_tag_value(val):
    if isinstance(val, bool):
        return '1' if val else '0'
    if isinstance(val, float):
        s = f"{val:g}".replace('-', 'm').replace('.', 'p')
        return s
    return str(val).replace('-', 'm').replace('.', 'p')


def make_setting_suffix(problem: str, **settings) -> str:
    # only include settings that differ from defaults; keep filenames short and reproducible
    parts = []
    for key in ['seed', 'n_points', 'three_peaks_iters', 'crash_iters', 'exact_front_threshold', 'bulge_gamma', 'indicator', 'initialization', 'dd_h', 'dd_sigma', 'move']:
        if key not in settings:
            continue
        default = DEFAULT_RUN_SETTINGS.get(key, None)
        val = settings[key]
        if val != default:
            parts.append(f"{OPTION_TAGS[key]}{_fmt_tag_value(val)}")
    return ("_" + "_".join(parts)) if parts else ""


def attach_suffix(prefix: str, suffix: str) -> str:
    return prefix + suffix if suffix else prefix


def run_bulged_three_peaks(prefix='bulged_three_peaks', outdir='.', seed=8, n_points=15, max_iter=200, gamma=0.5, exact_front_threshold=10, progress_every=10, quiet=False, indicator='magnitude', initialization='random', dd_h=3, dd_sigma=0.01, move='gradient'):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    effective_n_points = int((dd_h + 1) * (dd_h + 2) / 2) if initialization == 'dasdenis' else n_points
    suffix = make_setting_suffix('bulged_three_peaks', seed=seed, n_points=effective_n_points, three_peaks_iters=max_iter, exact_front_threshold=exact_front_threshold, bulge_gamma=gamma, indicator=indicator, initialization=initialization, dd_h=dd_h, dd_sigma=dd_sigma, move=move)
    prefix = attach_suffix(prefix, suffix)
    rng = np.random.default_rng(seed)
    X0 = make_simplex_initial_points(rng, init_mode=initialization, n_points=n_points, dd_h=dd_h, dd_sigma=dd_sigma)
    projector = project_simplex_rows
    anchor = (0.0, 0.0, 0.0)
    obj = lambda X: bulged_three_peaks_objective(X, gamma=gamma)
    jac = lambda X: bulged_three_peaks_jacobian(X, gamma=gamma)
    ref = approx_reference_simplex(obj, 800, np.random.default_rng(seed + 100))
    if move == 'gradient':
        res = run_projected_ascent(obj, jac, projector, X0, anchor, alpha0=0.05, max_iter=max_iter, sigma=0.04, tau=5e-4, shrink=0.99, max_retries=8, alpha_floor=1e-4, stall_limit=30, exact_front_threshold=exact_front_threshold, progress_every=progress_every, quiet=quiet, indicator=indicator)
    elif move == 'stochastic':
        res = run_stochastic_hillclimb(obj, projector, X0, anchor, rng=np.random.default_rng(seed + 1000), alpha0=0.05, max_iter=max_iter, sigma=0.04, tau=5e-4, shrink=0.99, max_retries=8, alpha_floor=1e-4, stall_limit=30, exact_front_threshold=exact_front_threshold, progress_every=progress_every, quiet=quiet, indicator=indicator)
    else:
        raise ValueError(f'unknown move: {move}')
    save_csv(out / f'{prefix}_initial_decisions.csv', ['x1', 'x2', 'x3'], res.X0)
    save_csv(out / f'{prefix}_final_decisions.csv', ['x1', 'x2', 'x3'], res.Xf)
    save_csv(out / f'{prefix}_initial_objectives.csv', ['f1', 'f2', 'f3'], res.Y0)
    save_csv(out / f'{prefix}_final_objectives.csv', ['f1', 'f2', 'f3'], res.Yf)
    save_csv(out / f'{prefix}_reference_archive.csv', ['f1', 'f2', 'f3'], ref)
    save_csv(out / f'{prefix}_history.csv', ['iteration', 'layered_value', 'alpha'], zip(range(len(res.values)), res.values, res.alphas))
    plot_objective_space(out / f'{prefix}_objective_space.png', res.Y0, res.Yf, ref, f'Bulged three-peaks benchmark (objective space, gamma={gamma:g})')
    plot_three_peaks_decision_space(out / f'{prefix}_decision_space.png', res.X0, res.Xf, f'Bulged three-peaks benchmark (decision space, gamma={gamma:g})')
    plot_convergence(out / f'{prefix}_convergence.png', res.values, res.alphas, f'Bulged three-peaks convergence (gamma={gamma:g})', stride=4)
    summary = summarize_result('bulged_three_peaks', res, ref)
    summary['gamma'] = float(gamma)
    summary['indicator'] = indicator
    summary['initialization'] = initialization
    summary['move'] = move
    if initialization == 'dasdenis':
        summary['dd_h'] = int(dd_h)
        summary['dd_sigma'] = float(dd_sigma)
        summary['n_points_effective'] = int((dd_h + 1) * (dd_h + 2) / 2)
    return summary




def bulged_three_peaks_box_objective(X: Array, gamma: float = 0.5) -> Array:
    E = np.eye(3)
    Y = np.zeros((len(X), 3), dtype=float)
    for i, e in enumerate(E):
        d2 = np.sum((X - e) ** 2, axis=1)
        t = np.clip(d2 / 2.0, 0.0, 1.0)
        Y[:, i] = 1.0 - np.power(t, gamma)
    return Y


def bulged_three_peaks_box_jacobian(X: Array, gamma: float = 0.5) -> Array:
    E = np.eye(3)
    J = np.zeros((len(X), 3, 3), dtype=float)
    for i, e in enumerate(E):
        diff = X - e
        d2 = np.sum(diff ** 2, axis=1)
        t = d2 / 2.0
        coeff = np.zeros(len(X), dtype=float)
        mask = (t > 1e-12) & (t < 1.0)
        coeff[mask] = -gamma * np.power(t[mask], gamma - 1.0)
        J[:, i, :] = coeff[:, None] * diff
    return J


def run_bulged_three_peaks_box(prefix='bulged_three_peaks_box', outdir='.', seed=8, n_points=15, max_iter=200, gamma=0.5, exact_front_threshold=10, progress_every=10, quiet=False, indicator='magnitude', initialization='random', dd_h=3, dd_sigma=0.01, move='gradient'):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if initialization == 'dasdenis':
        X0 = project_box(das_denis_points_3obj(dd_h) + rng.normal(0.0, dd_sigma, size=das_denis_points_3obj(dd_h).shape), -2.0, 2.0)
        effective_n_points = len(X0)
    else:
        X0 = np.clip(rng.normal(0.0, 0.06, size=(n_points, 3)), -2.0, 2.0)
        effective_n_points = n_points
    suffix = make_setting_suffix('bulged_three_peaks_box', seed=seed, n_points=effective_n_points, three_peaks_iters=max_iter, exact_front_threshold=exact_front_threshold, bulge_gamma=gamma, indicator=indicator, initialization=initialization, dd_h=dd_h, dd_sigma=dd_sigma, move=move)
    base = prefix + suffix
    projector = lambda X: project_box(X, -2.0, 2.0)
    anchor = (0.0, 0.0, 0.0)
    obj = lambda X: bulged_three_peaks_box_objective(X, gamma=gamma)
    jac = lambda X: bulged_three_peaks_box_jacobian(X, gamma=gamma)
    ref = approx_reference(obj, projector, 3, -2.0, 2.0, 1200, np.random.default_rng(seed + 100))
    if move == 'gradient':
        res = run_projected_ascent(obj, jac, projector, X0, anchor, alpha0=0.05, max_iter=max_iter, sigma=0.08, tau=5e-4, shrink=0.99, max_retries=8, alpha_floor=1e-4, stall_limit=30, exact_front_threshold=exact_front_threshold, progress_every=progress_every, quiet=quiet, indicator=indicator)
    else:
        res = run_stochastic_hillclimb(obj, projector, X0, anchor, rng=np.random.default_rng(seed + 1000), alpha0=0.05, max_iter=max_iter, sigma=0.08, tau=5e-4, shrink=0.99, max_retries=8, alpha_floor=1e-4, stall_limit=30, exact_front_threshold=exact_front_threshold, progress_every=progress_every, quiet=quiet, indicator=indicator)
    save_csv(out / f'{base}_initial_decisions.csv', ['x1', 'x2', 'x3'], res.X0)
    save_csv(out / f'{base}_final_decisions.csv', ['x1', 'x2', 'x3'], res.Xf)
    save_csv(out / f'{base}_initial_objectives.csv', ['f1', 'f2', 'f3'], res.Y0)
    save_csv(out / f'{base}_final_objectives.csv', ['f1', 'f2', 'f3'], res.Yf)
    save_csv(out / f'{base}_reference_archive.csv', ['f1', 'f2', 'f3'], ref)
    save_csv(out / f'{base}_history.csv', ['iteration', 'layered_value', 'alpha'], zip(range(len(res.values)), res.values, res.alphas))
    plot_objective_space(out / f'{base}_objective_space.png', res.Y0, res.Yf, ref, f'Bulged three-peaks box benchmark (objective space, gamma={gamma:g})')
    plot_three_peaks_decision_space(out / f'{base}_decision_space.png', res.X0, res.Xf, f'Bulged three-peaks box benchmark (decision space, gamma={gamma:g})')
    plot_convergence(out / f'{base}_convergence.png', res.values, res.alphas, f'Bulged three-peaks box convergence (gamma={gamma:g})', stride=4)
    summary = summarize_result('bulged_three_peaks_box', res, ref)
    summary['gamma'] = float(gamma)
    return summary

def crashworthiness_raw_objective(X: Array) -> Array:
    x1, x2, x3, x4, x5 = [X[:, i] for i in range(5)]
    f1 = 1640.2823 + 2.3573285 * x1 + 2.3220035 * x2 + 4.5688768 * x3 + 7.7213633 * x4 + 4.4559504 * x5
    f2 = 6.5856 + 1.15 * x1 - 1.0427 * x2 + 0.9738 * x3 + 0.8364 * x4 - 0.3695 * x1 * x4 + 0.0861 * x1 * x5 + 0.3628 * x2 * x4 - 0.1106 * x1**2 - 0.3437 * x3**2 + 0.1764 * x4**2
    f3 = -0.0551 + 0.0181 * x1 + 0.1024 * x2 + 0.0421 * x3 - 0.0073 * x1 * x2 + 0.0240 * x2 * x3 - 0.0118 * x2 * x4 - 0.0204 * x3 * x4 - 0.0080 * x3 * x5 - 0.0241 * x2**2 + 0.0109 * x4**2
    return np.column_stack([f1, f2, f3])


def crashworthiness_raw_jacobian(X: Array) -> Array:
    n = len(X)
    J = np.zeros((n, 3, 5))
    x1, x2, x3, x4, x5 = [X[:, i] for i in range(5)]
    J[:, 0, :] = np.array([2.3573285, 2.3220035, 4.5688768, 7.7213633, 4.4559504])
    J[:, 1, 0] = 1.15 - 0.3695 * x4 + 0.0861 * x5 - 0.2212 * x1
    J[:, 1, 1] = -1.0427 + 0.3628 * x4
    J[:, 1, 2] = 0.9738 - 0.6874 * x3
    J[:, 1, 3] = 0.8364 - 0.3695 * x1 + 0.3628 * x2 + 0.3528 * x4
    J[:, 1, 4] = 0.0861 * x1
    J[:, 2, 0] = 0.0181 - 0.0073 * x2
    J[:, 2, 1] = 0.1024 - 0.0073 * x1 + 0.0240 * x3 - 0.0118 * x4 - 0.0482 * x2
    J[:, 2, 2] = 0.0421 + 0.0240 * x2 - 0.0204 * x4 - 0.0080 * x5
    J[:, 2, 3] = -0.0118 * x2 - 0.0204 * x3 + 0.0218 * x4
    J[:, 2, 4] = -0.0080 * x3
    return J


def make_crash_transform(seed=0, calibration_samples=800):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 3.0, size=(calibration_samples, 5))
    F = crashworthiness_raw_objective(X)
    ideal = np.min(F, axis=0)
    nadir = np.max(F, axis=0)
    scale = np.maximum(nadir - ideal, 1e-12)
    def obj(X):
        return (nadir - crashworthiness_raw_objective(X)) / scale
    def jac(X):
        return -crashworthiness_raw_jacobian(X) / scale[None, :, None]
    return obj, jac, {'ideal': ideal.tolist(), 'nadir': nadir.tolist(), 'scale': scale.tolist()}


def project_box(X: Array, lo: float, hi: float) -> Array:
    return np.clip(X, lo, hi)


def approx_reference(objective_fn, projector, n_dec, lo, hi, samples, rng):
    X = rng.uniform(lo, hi, size=(samples, n_dec))
    return nondominated_subset(objective_fn(projector(X)))


def summarize_result(name: str, res: RunResult, ref: Array) -> dict:
    nd0 = nondominated_subset(res.Y0)
    ndf = nondominated_subset(res.Yf)
    return {
        'problem': name,
        'iterations_completed': int(res.completed_iterations),
        'layered_value_initial': float(res.values[0]),
        'layered_value_final': float(res.values[-1]),
        'igd_initial': igd(nd0, ref),
        'igd_final': igd(ndf, ref),
        'nd_count_initial': int(len(nd0)),
        'nd_count_final': int(len(ndf)),
        'accepted_steps': int(res.accepted_steps),
        'final_alpha': float(res.alphas[-1]),
    }


def run_three_peaks(prefix='three_peaks', outdir='.', seed=8, n_points=15, max_iter=200, exact_front_threshold=10, progress_every=10, quiet=False, indicator='magnitude', move='gradient'):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = make_setting_suffix('three_peaks', seed=seed, n_points=n_points, three_peaks_iters=max_iter, exact_front_threshold=exact_front_threshold, indicator=indicator, move=move)
    prefix = attach_suffix(prefix, suffix)
    rng = np.random.default_rng(seed)
    X0 = np.clip(rng.normal(0.0, 0.06, size=(n_points, 3)), -2.0, 2.0)
    projector = lambda X: project_box(X, -2.0, 2.0)
    anchor = (-4.0, -4.0, -4.0)
    ref = approx_reference(three_peaks_objective, projector, 3, -2.0, 2.0, 600, np.random.default_rng(seed + 100))
    if move == 'gradient':
        res = run_projected_ascent(three_peaks_objective, three_peaks_jacobian, projector, X0, anchor, alpha0=0.05, max_iter=max_iter, sigma=0.08, tau=5e-4, shrink=0.99, max_retries=8, alpha_floor=1e-4, stall_limit=30, exact_front_threshold=exact_front_threshold, progress_every=progress_every, quiet=quiet, indicator=indicator)
    elif move == 'stochastic':
        res = run_stochastic_hillclimb(three_peaks_objective, projector, X0, anchor, rng=np.random.default_rng(seed + 1000), alpha0=0.05, max_iter=max_iter, sigma=0.08, tau=5e-4, shrink=0.99, max_retries=8, alpha_floor=1e-4, stall_limit=30, exact_front_threshold=exact_front_threshold, progress_every=progress_every, quiet=quiet, indicator=indicator)
    else:
        raise ValueError(f'unknown move: {move}')
    save_csv(out / f'{prefix}_initial_decisions.csv', ['x1', 'x2', 'x3'], res.X0)
    save_csv(out / f'{prefix}_final_decisions.csv', ['x1', 'x2', 'x3'], res.Xf)
    save_csv(out / f'{prefix}_initial_objectives.csv', ['f1', 'f2', 'f3'], res.Y0)
    save_csv(out / f'{prefix}_final_objectives.csv', ['f1', 'f2', 'f3'], res.Yf)
    save_csv(out / f'{prefix}_reference_archive.csv', ['f1', 'f2', 'f3'], ref)
    save_csv(out / f'{prefix}_history.csv', ['iteration', 'layered_value', 'alpha'], zip(range(len(res.values)), res.values, res.alphas))
    plot_objective_space(out / f'{prefix}_objective_space.png', res.Y0, res.Yf, ref, 'Three-peaks benchmark (objective space)')
    plot_three_peaks_decision_space(out / f'{prefix}_decision_space.png', res.X0, res.Xf, 'Three-peaks benchmark (decision space)')
    plot_convergence(out / f'{prefix}_convergence.png', res.values, res.alphas, 'Three-peaks convergence', stride=4)
    summary = summarize_result('three_peaks', res, ref)
    summary['indicator'] = indicator
    summary['initialization'] = initialization
    summary['move'] = move
    if initialization == 'dasdenis':
        summary['dd_h'] = int(dd_h)
        summary['dd_sigma'] = float(dd_sigma)
        summary['n_points_effective'] = int((dd_h + 1) * (dd_h + 2) / 2)
    return summary


def run_crashworthiness(prefix='vehicle_crashworthiness', outdir='.', seed=9, n_points=15, max_iter=96, exact_front_threshold=10, progress_every=10, quiet=False, indicator='magnitude', move='gradient'):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = make_setting_suffix('vehicle_crashworthiness', seed=seed, n_points=n_points, crash_iters=max_iter, exact_front_threshold=exact_front_threshold, indicator=indicator, move=move)
    prefix = attach_suffix(prefix, suffix)
    rng = np.random.default_rng(seed)
    obj, jac, meta = make_crash_transform(seed + 50, 500)
    projector = lambda X: project_box(X, 1.0, 3.0)
    X0 = rng.uniform(1.0, 3.0, size=(n_points, 5))
    ref = approx_reference(obj, projector, 5, 1.0, 3.0, 700, np.random.default_rng(seed + 200))
    if move == 'gradient':
        res = run_projected_ascent(obj, jac, projector, X0, (-0.2, -0.2, -0.2), alpha0=0.02, max_iter=max_iter, sigma=0.06, tau=5e-4, shrink=0.99, max_retries=8, alpha_floor=1e-4, stall_limit=30, exact_front_threshold=exact_front_threshold, progress_every=progress_every, quiet=quiet, indicator=indicator)
    elif move == 'stochastic':
        res = run_stochastic_hillclimb(obj, projector, X0, (-0.2, -0.2, -0.2), rng=np.random.default_rng(seed + 1000), alpha0=0.02, max_iter=max_iter, sigma=0.06, tau=5e-4, shrink=0.99, max_retries=8, alpha_floor=1e-4, stall_limit=30, exact_front_threshold=exact_front_threshold, progress_every=progress_every, quiet=quiet, indicator=indicator)
    else:
        raise ValueError(f'unknown move: {move}')
    save_csv(out / f'{prefix}_initial_decisions.csv', [f'x{i}' for i in range(1, 6)], res.X0)
    save_csv(out / f'{prefix}_final_decisions.csv', [f'x{i}' for i in range(1, 6)], res.Xf)
    save_csv(out / f'{prefix}_initial_objectives.csv', ['g1', 'g2', 'g3'], res.Y0)
    save_csv(out / f'{prefix}_final_objectives.csv', ['g1', 'g2', 'g3'], res.Yf)
    save_csv(out / f'{prefix}_initial_objectives_raw.csv', ['f1', 'f2', 'f3'], crashworthiness_raw_objective(res.X0))
    save_csv(out / f'{prefix}_final_objectives_raw.csv', ['f1', 'f2', 'f3'], crashworthiness_raw_objective(res.Xf))
    save_csv(out / f'{prefix}_reference_archive.csv', ['g1', 'g2', 'g3'], ref)
    save_csv(out / f'{prefix}_history.csv', ['iteration', 'layered_value', 'alpha'], zip(range(len(res.values)), res.values, res.alphas))
    plot_objective_space(out / f'{prefix}_objective_space.png', res.Y0, res.Yf, ref, 'Vehicle crashworthiness benchmark (objective space)')
    plot_convergence(out / f'{prefix}_convergence.png', res.values, res.alphas, 'Vehicle crashworthiness convergence', stride=4)
    summary = summarize_result('vehicle_crashworthiness', res, ref)
    summary['normalization'] = meta
    summary['indicator'] = indicator
    summary['initialization'] = initialization
    summary['move'] = move
    if initialization == 'dasdenis':
        summary['dd_h'] = int(dd_h)
        summary['dd_sigma'] = float(dd_sigma)
        summary['n_points_effective'] = int((dd_h + 1) * (dd_h + 2) / 2)
    return summary


# ---------------------------- self test ----------------------------

def run_self_test(n_cases: int = 8, seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)
    max_abs_err_tiefree = 0.0
    max_abs_err_tied = 0.0
    rows = []
    for case in range(n_cases):
        n = 5 + (case % 2)
        # tie-free random case
        pts = rng.uniform(0.1, 3.0, size=(n, 3))
        g_exact = exact_gradient_with_index_perturbation(pts, (0.0, 0.0, 0.0), delta=1e-12)
        g_sweep = magnitude_gradient_3d_max_sweep_forward(pts, (0.0, 0.0, 0.0))
        err = float(np.max(np.abs(g_exact - g_sweep)))
        max_abs_err_tiefree = max(max_abs_err_tiefree, err)
        rows.append((case, 'tiefree', n, err))
        # tied / integer-like case with point-index perturbation in exact reference
        pts_tied = rng.integers(1, 5, size=(n, 3)).astype(float)
        g_exact_t = exact_gradient_with_index_perturbation(pts_tied, (0.0, 0.0, 0.0), delta=1e-9)
        g_sweep_t = magnitude_gradient_3d_max_sweep_forward(pts_tied, (0.0, 0.0, 0.0))
        err_t = float(np.max(np.abs(g_exact_t - g_sweep_t)))
        max_abs_err_tied = max(max_abs_err_tied, err_t)
        rows.append((case, 'tied', n, err_t))
    return {
        'cases': n_cases,
        'max_abs_err_tiefree': max_abs_err_tiefree,
        'max_abs_err_tied_vs_index_perturbed_exact': max_abs_err_tied,
        'rows': rows,
    }


# ---------------------------- CLI ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', choices=['three_peaks', 'bulged_three_peaks', 'bulged_three_peaks_box', 'vehicle_crashworthiness', 'both'], default='bulged_three_peaks')
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--seed', type=int, default=8)
    ap.add_argument('--n-points', type=int, default=15)
    ap.add_argument('--three-peaks-iters', type=int, default=200)
    ap.add_argument('--crash-iters', type=int, default=96)
    ap.add_argument('--exact-front-threshold', type=int, default=10)
    ap.add_argument('--progress-every', type=int, default=10)
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--bulge-gamma', type=float, default=0.5)
    ap.add_argument('--indicator', choices=['magnitude', 'hypervolume'], default='magnitude')
    ap.add_argument('--initialization', choices=['random', 'dasdenis'], default='random')
    ap.add_argument('--dd-h', type=int, default=3)
    ap.add_argument('--dd-sigma', type=float, default=0.01)
    ap.add_argument('--move', choices=['gradient', 'stochastic'], default='gradient')
    args = ap.parse_args()

    if args.self_test:
        test = run_self_test()
        print(json.dumps({k: v for k, v in test.items() if k != 'rows'}, indent=2))
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        save_csv(out / 'self_test_rows.csv', ['case', 'kind', 'n', 'max_abs_err'], test['rows'])
        with open(out / 'self_test_summary.json', 'w') as f:
            json.dump({k: v for k, v in test.items() if k != 'rows'}, f, indent=2)
        return

    summaries = {}
    if args.problem in ('three_peaks', 'both'):
        summaries['three_peaks'] = run_three_peaks(outdir=args.outdir, seed=args.seed, n_points=args.n_points, max_iter=args.three_peaks_iters, exact_front_threshold=args.exact_front_threshold, progress_every=args.progress_every, quiet=args.quiet, indicator=args.indicator, move=args.move)
    if args.problem in ('bulged_three_peaks', 'both'):
        summaries['bulged_three_peaks'] = run_bulged_three_peaks(outdir=args.outdir, seed=args.seed, n_points=args.n_points, max_iter=args.three_peaks_iters, gamma=args.bulge_gamma, exact_front_threshold=args.exact_front_threshold, progress_every=args.progress_every, quiet=args.quiet, indicator=args.indicator, initialization=args.initialization, dd_h=args.dd_h, dd_sigma=args.dd_sigma, move=args.move)
    if args.problem in ('bulged_three_peaks_box', 'both'):
        summaries['bulged_three_peaks_box'] = run_bulged_three_peaks_box(outdir=args.outdir, seed=args.seed, n_points=args.n_points, max_iter=args.three_peaks_iters, gamma=args.bulge_gamma, exact_front_threshold=args.exact_front_threshold, progress_every=args.progress_every, quiet=args.quiet, indicator=args.indicator, initialization=args.initialization, dd_h=args.dd_h, dd_sigma=args.dd_sigma, move=args.move)
    if args.problem in ('vehicle_crashworthiness', 'both'):
        summaries['vehicle_crashworthiness'] = run_crashworthiness(outdir=args.outdir, seed=args.seed + 1, n_points=args.n_points, max_iter=args.crash_iters, exact_front_threshold=args.exact_front_threshold, progress_every=args.progress_every, quiet=args.quiet, indicator=args.indicator, move=args.move)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary_suffix = make_setting_suffix(args.problem, seed=args.seed, n_points=args.n_points, three_peaks_iters=args.three_peaks_iters, crash_iters=args.crash_iters, exact_front_threshold=args.exact_front_threshold, bulge_gamma=args.bulge_gamma, indicator=args.indicator, initialization=args.initialization, dd_h=args.dd_h, dd_sigma=args.dd_sigma, move=args.move)
    with open(out / 'benchmark_summary.json', 'w') as f:
        json.dump(summaries, f, indent=2)
    if summary_suffix:
        with open(out / f'benchmark_summary{summary_suffix}.json', 'w') as f:
            json.dump(summaries, f, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
