#!/usr/bin/env python3
"""Reproduce the longer-run Figure 4 replacement.

This script reproduces the projected finite-difference diffusion used for
Figure 4 of the manuscript:

  - ten-point summed quadratic biobjective example,
  - active short-range repulsion term,
  - projection to the natural box [0,1]^2 after each decision-space step,
  - alpha = 0.004,
  - max_iter = 540,
  - eps = 1e-3,
  - tau = 2e-4,
  - sigma = 0.03.

Outputs written in the current directory:

  figure4_decision_paths.csv
  figure4_objective_paths.csv
  figure4_initial_final_decision.csv
  figure4_initial_final_objective.csv
  figure4_reproduced.png

The CSV path files contain all iterations, not only the sparse TikZ samples.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray


def F(x: Array) -> Array:
    """Summed quadratic biobjective map, maximization convention."""
    x1, x2 = float(x[0]), float(x[1])
    f1 = 0.5 * (1.0 - (x1 - 1.0) ** 2) + 0.5 * (1.0 - (x2 - 1.0) ** 2)
    f2 = 0.5 * (1.0 - x1**2) + 0.5 * (1.0 - x2**2)
    return np.array([f1, f2], dtype=float)


def dominates(a: Array, b: Array) -> bool:
    return bool(np.all(a >= b) and np.any(a > b))


def nondomination_layers(A: Array) -> list[list[int]]:
    remaining = list(range(len(A)))
    layers: list[list[int]] = []
    while remaining:
        front: list[int] = []
        for i in remaining:
            if not any(j != i and dominates(A[j], A[i]) for j in remaining):
                front.append(i)
        layers.append(front)
        remaining = [i for i in remaining if i not in front]
    return layers


def nondominated_points(A: Array) -> Array:
    if len(A) == 0:
        return np.empty((0, 2), dtype=float)
    arr = np.array(A, dtype=float)
    return arr[nondomination_layers(arr)[0]]


def hv2(points: Array) -> float:
    """Anchored two-dimensional hypervolume with reference point (0,0)."""
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
    """Magnitude of a dominated biobjective set via extent + area terms."""
    pts = nondominated_points(points)
    if len(pts) == 0:
        return 0.0
    x_extent = float(np.max(pts[:, 0]))
    y_extent = float(np.max(pts[:, 1]))
    return 1.0 + 0.5 * (x_extent + y_extent) + 0.25 * hv2(pts)


def repulsion(points: Array, sigma: float) -> float:
    """Short-range Gaussian repulsion, active when tau > 0."""
    val = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d2 = float(np.sum((points[i] - points[j]) ** 2))
            val += math.exp(-d2 / (sigma * sigma))
    return val


def layered_value_obj(Y: Array, eps: float, tau: float, sigma: float) -> float:
    """Layered magnitude indicator minus repulsion penalty."""
    value = 0.0
    for ell, front in enumerate(nondomination_layers(Y)):
        value += (eps**ell) * magnitude_dom(Y[front])
    value -= tau * repulsion(Y, sigma)
    return float(value)


def project_box(X: Array, lo: float = 0.0, hi: float = 1.0) -> Array:
    """Projection used in Figure 4: keep decision points in [0,1]^2."""
    return np.clip(np.array(X, dtype=float), lo, hi)


def finite_difference_gradient(
    X: Array,
    eps: float,
    tau: float,
    sigma: float,
    h: float,
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


def run_diffusion(
    X0: Array,
    alpha: float = 0.004,
    max_iter: int = 540,
    eps: float = 1e-3,
    tau: float = 2e-4,
    sigma: float = 0.03,
    h: float = 1e-5,
) -> tuple[Array, list[Array], list[float]]:
    X = project_box(X0)
    history = [X.copy()]
    values = [layered_value_obj(np.array([F(x) for x in X]), eps, tau, sigma)]
    for _ in range(max_iter):
        G = finite_difference_gradient(X, eps=eps, tau=tau, sigma=sigma, h=h)
        G = normalize_per_point(G)
        X = project_box(X + alpha * G)
        history.append(X.copy())
        values.append(layered_value_obj(np.array([F(x) for x in X]), eps, tau, sigma))
    return X, history, values


def write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        writer.writerows(rows)


def export_paths(history: list[Array], outdir: Path) -> None:
    decision_rows = []
    objective_rows = []
    for iteration, X in enumerate(history):
        for point_index, x in enumerate(X, start=1):
            y = F(x)
            decision_rows.append((iteration, point_index, float(x[0]), float(x[1])))
            objective_rows.append((iteration, point_index, float(y[0]), float(y[1])))
    write_csv(outdir / "figure4_decision_paths.csv", ["iteration", "point", "x1", "x2"], decision_rows)
    write_csv(outdir / "figure4_objective_paths.csv", ["iteration", "point", "F1", "F2"], objective_rows)

    X0 = history[0]
    Xf = history[-1]
    write_csv(
        outdir / "figure4_initial_final_decision.csv",
        ["status", "point", "x1", "x2"],
        [("initial", i + 1, *X0[i]) for i in range(len(X0))]
        + [("final", i + 1, *Xf[i]) for i in range(len(Xf))],
    )
    Y0 = np.array([F(x) for x in X0])
    Yf = np.array([F(x) for x in Xf])
    write_csv(
        outdir / "figure4_initial_final_objective.csv",
        ["status", "point", "F1", "F2"],
        [("initial", i + 1, *Y0[i]) for i in range(len(Y0))]
        + [("final", i + 1, *Yf[i]) for i in range(len(Yf))],
    )


def plot_figure4(history: list[Array], outpath: Path, max_iter: int) -> None:
    samples = np.unique(np.linspace(0, max_iter, 9, dtype=int))
    colors = ["0.55", "0.45", "tab:blue", "tab:cyan", "tab:green", "tab:olive", "tab:orange", "tab:red"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), constrained_layout=True)

    # Decision-space panel
    ax = axes[0]
    ax.plot([0, 1], [0, 1], linewidth=2.0)
    for i in range(len(history[0])):
        pts = np.array([history[k][i] for k in samples])
        for j in range(len(pts) - 1):
            ax.annotate(
                "",
                xy=pts[j + 1],
                xytext=pts[j],
                arrowprops=dict(arrowstyle="->", lw=1.2, color=colors[j], shrinkA=0, shrinkB=0),
            )
    ax.scatter(history[0][:, 0], history[0][:, 1], s=24, color="0.5", zorder=3)
    ax.scatter(history[-1][:, 0], history[-1][:, 1], s=34, color="tab:red", zorder=4)
    ax.set_title("decision space")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")

    # Objective-space panel
    ax = axes[1]
    t = np.linspace(0, 1, 300)
    ax.plot(2 * t - t**2, 1 - t**2, linewidth=2.0)
    for i in range(len(history[0])):
        pts = np.array([F(history[k][i]) for k in samples])
        for j in range(len(pts) - 1):
            ax.annotate(
                "",
                xy=pts[j + 1],
                xytext=pts[j],
                arrowprops=dict(arrowstyle="->", lw=1.2, color=colors[j], shrinkA=0, shrinkB=0),
            )
    Y0 = np.array([F(x) for x in history[0]])
    Yf = np.array([F(x) for x in history[-1]])
    ax.scatter(Y0[:, 0], Y0[:, 1], s=24, color="0.5", zorder=3)
    ax.scatter(Yf[:, 0], Yf[:, 1], s=34, color="tab:red", zorder=4)
    ax.set_title("objective space")
    ax.set_xlabel("F1")
    ax.set_ylabel("F2")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Figure 4 reproduction: projected diffusion with active repulsion")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    outdir = Path(".")
    X0 = np.array(
        [
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
        ],
        dtype=float,
    )

    alpha = 0.004
    max_iter = 540
    eps = 1e-3
    tau = 2e-4
    sigma = 0.03
    h = 1e-5

    Xf, history, values = run_diffusion(X0, alpha=alpha, max_iter=max_iter, eps=eps, tau=tau, sigma=sigma, h=h)
    export_paths(history, outdir)
    plot_figure4(history, outdir / "figure4_reproduced.png", max_iter=max_iter)

    print("Figure 4 reproduction complete.")
    print(f"alpha={alpha}, max_iter={max_iter}, eps={eps}, tau={tau}, sigma={sigma}, h={h}")
    print(f"Initial objective value: {values[0]:.8f}")
    print(f"Final objective value:   {values[-1]:.8f}")
    print("Final decision points:")
    print(np.round(Xf, 6))
    print("Final objective vectors:")
    print(np.round(np.array([F(x) for x in Xf]), 6))


if __name__ == "__main__":
    main()
