#!/usr/bin/env python3
"""Projected Clarke gradient diffusion for a layered magnitude indicator.

This script reproduces the perturbed 10-point summed quadratic example from the
note. The optimization is carried out in decision space; objective values are
computed by the smooth map

    F1(x) = 1/2 (1-(x1-1)^2) + 1/2 (1-(x2-1)^2)
    F2(x) = 1/2 (1-x1^2)     + 1/2 (1-x2^2)

The layered indicator is evaluated in objective space. A short-range repulsion
term prevents exact duplicates. The update uses a finite-difference Clarke-like
gradient, normalized pointwise, followed by small projected diffusion steps.

The script exports CSV files for both decision-space and objective-space paths.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
import numpy as np

Array = np.ndarray


def F(x: Array) -> Array:
    x1, x2 = float(x[0]), float(x[1])
    f1 = 0.5 * (1.0 - (x1 - 1.0) ** 2) + 0.5 * (1.0 - (x2 - 1.0) ** 2)
    f2 = 0.5 * (1.0 - x1 ** 2) + 0.5 * (1.0 - x2 ** 2)
    return np.array([f1, f2], dtype=float)


def dominates(a: Array, b: Array) -> bool:
    return bool(np.all(a >= b) and np.any(a > b))


def nondomination_layers(A: Array):
    remaining = list(range(len(A)))
    layers = []
    while remaining:
        front = []
        for i in remaining:
            if not any(j != i and dominates(A[j], A[i]) for j in remaining):
                front.append(i)
        layers.append(front)
        remaining = [i for i in remaining if i not in front]
    return layers


def nondominated_points(A: Array) -> Array:
    if len(A) == 0:
        return np.empty((0, 2), dtype=float)
    return np.array(A, dtype=float)[nondomination_layers(np.array(A, dtype=float))[0]]


def hv2(points: Array) -> float:
    pts = nondominated_points(points)
    if len(pts) == 0:
        return 0.0
    xs = sorted(set([0.0] + [float(p[0]) for p in pts]))
    area = 0.0
    for a, b in zip(xs[:-1], xs[1:]):
        mid = 0.5 * (a + b)
        ys = [float(y) for x, y in pts if x >= mid]
        area += (b - a) * (max(ys) if ys else 0.0)
    return float(area)


def magnitude_dom(points: Array) -> float:
    pts = nondominated_points(points)
    if len(pts) == 0:
        return 0.0
    x = float(np.max(pts[:, 0]))
    y = float(np.max(pts[:, 1]))
    return 1.0 + 0.5 * (x + y) + 0.25 * hv2(pts)


def repulsion(points: Array, sigma: float) -> float:
    val = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d2 = float(np.sum((points[i] - points[j]) ** 2))
            val += math.exp(-d2 / (sigma * sigma))
    return val


def layered_value_obj(Y: Array, eps: float, tau: float, sigma: float) -> float:
    layers = nondomination_layers(Y)
    val = 0.0
    for ell, front in enumerate(layers):
        val += (eps ** ell) * magnitude_dom(Y[front])
    val -= tau * repulsion(Y, sigma)
    return float(val)


def project_box(X: Array, lo: float = -2.0, hi: float = 2.0) -> Array:
    return np.clip(X, lo, hi)


def finite_difference_gradient(
    X: Array,
    eps: float = 1e-3,
    tau: float = 2e-4,
    sigma: float = 0.03,
    h: float = 1e-5,
) -> Array:
    G = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(2):
            E = np.zeros_like(X)
            E[i, j] = h
            fp = layered_value_obj(np.array([F(x) for x in project_box(X + E)]), eps, tau, sigma)
            fm = layered_value_obj(np.array([F(x) for x in project_box(X - E)]), eps, tau, sigma)
            G[i, j] = (fp - fm) / (2.0 * h)
    return G


def normalize_per_point(G: Array) -> Array:
    H = G.copy()
    for i in range(len(H)):
        nrm = float(np.linalg.norm(H[i]))
        if nrm > 1e-12:
            H[i] /= nrm
    return H


def clarke_gradient_diffusion(
    X0: Array,
    alpha: float = 0.015,
    max_iter: int = 140,
    eps: float = 1e-3,
    tau: float = 2e-4,
    sigma: float = 0.03,
    h: float = 1e-5,
):
    X = project_box(np.array(X0, dtype=float))
    history = [X.copy()]
    values = [layered_value_obj(np.array([F(x) for x in X]), eps, tau, sigma)]
    for _ in range(max_iter):
        G = finite_difference_gradient(X, eps=eps, tau=tau, sigma=sigma, h=h)
        G = normalize_per_point(G)
        X = project_box(X + alpha * G)
        history.append(X.copy())
        values.append(layered_value_obj(np.array([F(x) for x in X]), eps, tau, sigma))
    return X, history, values


def write_csv(path: Path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def export_front(outdir: Path, m: int = 400):
    rows = []
    for k in range(m + 1):
        t = k / m
        rows.append((2.0 * t - t * t, 1.0 - t * t))
    write_csv(outdir / 'quadratic10_front.csv', ['F1', 'F2'], rows)


def export_effset(outdir: Path, m: int = 400):
    rows = []
    for k in range(m + 1):
        t = k / m
        rows.append((t, t))
    write_csv(outdir / 'quadratic10_effset.csv', ['x1', 'x2'], rows)


def export_paths(outdir: Path, history, X0, Xf):
    Y0 = np.array([F(x) for x in X0], dtype=float)
    Yf = np.array([F(x) for x in Xf], dtype=float)
    write_csv(outdir / 'quadratic10_initial.csv', ['F1', 'F2'], Y0.tolist())
    write_csv(outdir / 'quadratic10_final.csv', ['F1', 'F2'], Yf.tolist())
    write_csv(outdir / 'quadratic10_initial_decision.csv', ['x1', 'x2'], np.array(X0).tolist())
    write_csv(outdir / 'quadratic10_final_decision.csv', ['x1', 'x2'], np.array(Xf).tolist())
    for i in range(len(X0)):
        rows_dec = []
        rows_obj = []
        for H in history:
            x = H[i]
            y = F(x)
            rows_dec.append((float(x[0]), float(x[1])))
            rows_obj.append((float(y[0]), float(y[1])))
        write_csv(outdir / f'quadratic10_path_decision_{i+1:02d}.csv', ['x1', 'x2'], rows_dec)
        write_csv(outdir / f'quadratic10_path_objective_{i+1:02d}.csv', ['F1', 'F2'], rows_obj)


def main():
    outdir = Path('.')
    X0 = np.array([
        [0.10, 0.74],
        [0.18, 0.49],
        [0.12, 0.61],
        [0.33, 0.58],
        [0.46, 0.28],
        [0.41, 0.45],
        [0.63, 0.12],
        [0.57, 0.26],
        [0.71, 0.33],
        [0.82, 0.08],
    ], dtype=float)

    Xf, history, values = clarke_gradient_diffusion(X0)
    export_front(outdir)
    export_effset(outdir)
    export_paths(outdir, history, X0, Xf)

    print('Initial objective value:', values[0])
    print('Final objective value  :', values[-1])
    print('Final decision points and objective vectors:')
    for x in Xf:
        print(np.round(x, 6), '->', np.round(F(x), 6))


if __name__ == '__main__':
    main()
