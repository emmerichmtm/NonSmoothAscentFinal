#!/usr/bin/env python3
"""Recompute the rank-colored Figure 3 support data and a PNG preview.

The figure compares two objective-space triangle runs with mu=10:
(a) a nondominated line start on F1+F2=0.7, and
(b) a dominated triangular 4+3+2+1 start in the lower-left subtriangle.

This script exports CSV files containing the sampled paths and dominance ranks,
and creates figure3_rank_colored_preview.png. It uses the same layer color
palette as Figure 1 in the paper.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Same point colors as Figure 1 in the article.
LAYER_COLORS = {
    1: "#0b1bcd",  # darkblue
    2: "#0b1b05",  # darkgreen
    3: "#fd120b",  # darkorange/red-orange, used for layer 3+
}
ITER_COLORS = ["0.70", "0.62", "#7f7fff", "#45b8c6", "#009688", "#58a65c", "#d98c00", "#b22222"]

LINE_START_PATHS = [
    [(0.020000, 0.680000), (0.024463, 0.839925), (0.016250, 0.983750), (0.000000, 1.000000), (0.000000, 1.000000), (0.000000, 1.000000), (0.000000, 1.000000), (0.000000, 1.000000), (0.000000, 1.000000)],
    [(0.093000, 0.607000), (0.158463, 0.752040), (0.182388, 0.817612), (0.173289, 0.826711), (0.171543, 0.828457), (0.171196, 0.828804), (0.171126, 0.828874), (0.171112, 0.828888), (0.171109, 0.828891)],
    [(0.167000, 0.533000), (0.254203, 0.666658), (0.282726, 0.717274), (0.278344, 0.721656), (0.277131, 0.722869), (0.276885, 0.723115), (0.276836, 0.723164), (0.276826, 0.723174), (0.276824, 0.723176)],
    [(0.240000, 0.460000), (0.340431, 0.584296), (0.372252, 0.627748), (0.369999, 0.630001), (0.369281, 0.630719), (0.369134, 0.630866), (0.369105, 0.630895), (0.369099, 0.630901), (0.369098, 0.630902)],
    [(0.313000, 0.387000), (0.422661, 0.503412), (0.457795, 0.542205), (0.457089, 0.542911), (0.456850, 0.543150), (0.456802, 0.543198), (0.456792, 0.543208), (0.456790, 0.543210), (0.456790, 0.543210)],
    [(0.387000, 0.313000), (0.503412, 0.422661), (0.542205, 0.457795), (0.542911, 0.457089), (0.543150, 0.456850), (0.543198, 0.456802), (0.543208, 0.456792), (0.543210, 0.456790), (0.543210, 0.456790)],
    [(0.460000, 0.240000), (0.584296, 0.340431), (0.627748, 0.372252), (0.630001, 0.369999), (0.630719, 0.369281), (0.630866, 0.369134), (0.630895, 0.369105), (0.630901, 0.369099), (0.630902, 0.369098)],
    [(0.533000, 0.167000), (0.666658, 0.254203), (0.717274, 0.282726), (0.721656, 0.278344), (0.722869, 0.277131), (0.723115, 0.276885), (0.723164, 0.276836), (0.723174, 0.276826), (0.723176, 0.276824)],
    [(0.607000, 0.093000), (0.752040, 0.158463), (0.817612, 0.182388), (0.826711, 0.173289), (0.828457, 0.171543), (0.828804, 0.171196), (0.828874, 0.171126), (0.828888, 0.171112), (0.828891, 0.171109)],
    [(0.680000, 0.020000), (0.839925, 0.024463), (0.983750, 0.016250), (1.000000, 0.000000), (1.000000, 0.000000), (1.000000, 0.000000), (1.000000, 0.000000), (1.000000, 0.000000), (1.000000, 0.000000)],
]

TRIANGLE_START_PATHS = [
    [(0.030000, 0.270000), (0.038465, 0.509847), (0.050646, 0.749537), (0.031179, 0.968821), (0.000000, 1.000000), (0.000000, 1.000000), (0.000000, 1.000000), (0.000000, 1.000000), (0.000000, 1.000000)],
    [(0.080000, 0.220000), (0.215447, 0.417680), (0.349529, 0.616663), (0.332920, 0.667080), (0.315281, 0.684719), (0.312894, 0.687106), (0.314881, 0.685119), (0.317406, 0.682594), (0.319813, 0.680187)],
    [(0.130000, 0.170000), (0.327746, 0.305364), (0.532736, 0.429781), (0.629322, 0.370678), (0.621759, 0.378241), (0.637987, 0.362013), (0.644501, 0.355499), (0.648827, 0.351173), (0.652116, 0.347884)],
    [(0.180000, 0.120000), (0.419857, 0.128209), (0.659538, 0.140517), (0.876355, 0.123645), (1.000000, 0.000000), (1.000000, 0.000000), (1.000000, 0.000000), (1.000000, 0.000000), (1.000000, 0.000000)],
    [(0.050000, 0.190000), (0.039812, 0.408609), (0.103498, 0.636942), (0.199356, 0.800644), (0.209140, 0.790860), (0.208834, 0.791166), (0.209345, 0.790655), (0.210784, 0.789216), (0.212384, 0.787616)],
    [(0.100000, 0.140000), (0.229054, 0.272609), (0.382290, 0.449439), (0.470083, 0.529917), (0.513735, 0.486265), (0.525212, 0.474788), (0.531577, 0.468423), (0.535968, 0.464032), (0.539398, 0.460602)],
    [(0.150000, 0.090000), (0.367729, 0.069070), (0.607137, 0.082437), (0.838567, 0.075408), (0.895553, 0.104447), (0.878490, 0.121510), (0.879743, 0.120257), (0.881530, 0.118470), (0.882957, 0.117043)],
    [(0.070000, 0.110000), (0.032698, 0.262538), (0.043600, 0.502108), (0.072325, 0.740372), (0.105750, 0.894250), (0.104562, 0.895438), (0.104545, 0.895455), (0.105158, 0.894842), (0.105944, 0.894056)],
    [(0.120000, 0.060000), (0.288432, 0.000000), (0.513376, 0.030636), (0.606975, 0.245327), (0.747077, 0.252923), (0.756691, 0.243309), (0.760816, 0.239184), (0.764175, 0.235825), (0.766767, 0.233233)],
    [(0.090000, 0.030000), (0.114666, 0.086943), (0.200665, 0.292125), (0.370251, 0.460340), (0.415664, 0.584336), (0.417667, 0.582333), (0.421962, 0.578038), (0.425614, 0.574386), (0.428694, 0.571306)],
]


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a >= b) and np.any(a > b))


def nondomination_ranks(points: np.ndarray) -> np.ndarray:
    remaining = list(range(len(points)))
    ranks = np.zeros(len(points), dtype=int)
    rank = 1
    while remaining:
        front = []
        for i in remaining:
            if not any(j != i and dominates(points[j], points[i]) for j in remaining):
                front.append(i)
        for i in front:
            ranks[i] = rank
        remaining = [i for i in remaining if i not in front]
        rank += 1
    return ranks


def capped_color(rank: int) -> str:
    return LAYER_COLORS[rank if rank <= 2 else 3]


def export_csv(paths: list[list[tuple[float, float]]], prefix: str, outdir: Path) -> None:
    rows = []
    arr = np.array(paths, dtype=float)  # point, sample, coordinate
    for sample in range(arr.shape[1]):
        ranks = nondomination_ranks(arr[:, sample, :])
        for point in range(arr.shape[0]):
            rows.append((sample, point + 1, arr[point, sample, 0], arr[point, sample, 1], int(ranks[point])))
    with (outdir / f"{prefix}_ranked_samples.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "point", "F1", "F2", "dominance_rank"])
        writer.writerows(rows)


def plot_panel(ax, paths: list[list[tuple[float, float]]], title: str) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linewidth=0.4, alpha=0.45)
    ax.fill([0, 1, 0], [0, 0, 1], alpha=0.06)
    ax.plot([1, 0], [0, 1], linewidth=2.2)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel(r"$F_1$")
    ax.set_ylabel(r"$F_2$")
    ax.set_title(title)
    arr = np.array(paths, dtype=float)
    for point_path in arr:
        for k in range(len(point_path) - 1):
            ax.annotate(
                "",
                xy=point_path[k + 1],
                xytext=point_path[k],
                arrowprops=dict(arrowstyle="->", lw=0.75, color=ITER_COLORS[k], shrinkA=0, shrinkB=0),
            )
    for sample in range(arr.shape[1]):
        ranks = nondomination_ranks(arr[:, sample, :])
        for p in range(arr.shape[0]):
            ax.scatter(arr[p, sample, 0], arr[p, sample, 1], s=16, color=capped_color(int(ranks[p])), edgecolor="white", linewidth=0.25, zorder=4)
    # final iterates stay red, as in the article
    ax.scatter(arr[:, -1, 0], arr[:, -1, 1], s=18, color="#b22222", zorder=5)


def main() -> None:
    outdir = Path(".")
    export_csv(LINE_START_PATHS, "figure3_line_start", outdir)
    export_csv(TRIANGLE_START_PATHS, "figure3_dominated_triangle_start", outdir)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.7), constrained_layout=True)
    plot_panel(axes[0], LINE_START_PATHS, "line start")
    plot_panel(axes[1], TRIANGLE_START_PATHS, "triangular 4+3+2+1 start")
    fig.savefig(outdir / "figure3_rank_colored_preview.png", dpi=220)
    print("Wrote figure3_rank_colored_preview.png and ranked CSV files.")


if __name__ == "__main__":
    main()
