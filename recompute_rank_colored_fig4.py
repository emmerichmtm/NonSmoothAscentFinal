#!/usr/bin/env python3
"""Regenerate Figure 4 rank-colored support data and a PNG preview.

This fast script reads the embedded TikZ/PGF coordinates from
`article_rank_colored_initials.tex`, reconstructs the sampled decision-space
and objective-space paths, computes objective-space nondomination ranks at each
sample, and writes CSV files plus a preview PNG.

The full numerical rerun that produced the paths is retained in
`reproduce_figure4_result.py`; this script is intentionally focused on
reconstructing the plotted, rank-colored figure from the article source.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

LAYER_COLORS = {
    1: "#0b1bcd",  # darkblue
    2: "#0b1b05",  # darkgreen
    3: "#fd120b",  # darkorange/red-orange, used for layer 3+
}
ITER_COLORS = ["0.70", "0.62", "#7f7fff", "#45b8c6", "#009688", "#58a65c", "#d98c00", "#b22222"]


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


def extract_figure_block(tex: str, label: str) -> str:
    label_pos = tex.index(label)
    start = tex.rfind(r"\begin{figure}", 0, label_pos)
    end = tex.find(r"\end{figure}", label_pos) + len(r"\end{figure}")
    return tex[start:end]


def extract_tikzpictures(fig: str) -> list[str]:
    return re.findall(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", fig, flags=re.S)


def extract_paths_from_tikz(tikz: str, expected_points: int = 10, segments_per_path: int = 8) -> np.ndarray:
    coord_pattern = re.compile(
        r"\(([0-9.\-]+),([0-9.\-]+)\)\s*--\s*\(([0-9.\-]+),([0-9.\-]+)\)"
    )
    segments = []
    for line in tikz.splitlines():
        if r"\draw[" in line and "line width=0.56pt" in line:
            match = coord_pattern.search(line)
            if match:
                segments.append(tuple(map(float, match.groups())))
    needed = expected_points * segments_per_path
    if len(segments) < needed:
        raise RuntimeError(f"Expected at least {needed} path segments, found {len(segments)}")
    segments = segments[:needed]
    paths = []
    for p in range(expected_points):
        segs = segments[p * segments_per_path : (p + 1) * segments_per_path]
        coords = [(segs[0][0], segs[0][1])] + [(s[2], s[3]) for s in segs]
        paths.append(coords)
    return np.array(paths, dtype=float)  # shape: point, sample, coordinate


def export_ranked_samples(decision_paths: np.ndarray, objective_paths: np.ndarray, outdir: Path) -> None:
    dec_rows = []
    obj_rows = []
    for sample in range(objective_paths.shape[1]):
        ranks = nondomination_ranks(objective_paths[:, sample, :])
        for point in range(objective_paths.shape[0]):
            dec_rows.append((sample, point + 1, decision_paths[point, sample, 0], decision_paths[point, sample, 1], int(ranks[point])))
            obj_rows.append((sample, point + 1, objective_paths[point, sample, 0], objective_paths[point, sample, 1], int(ranks[point])))
    for filename, header, rows in [
        ("figure4_decision_ranked_samples.csv", ["sample", "point", "x1", "x2", "objective_space_rank"], dec_rows),
        ("figure4_objective_ranked_samples.csv", ["sample", "point", "F1", "F2", "objective_space_rank"], obj_rows),
    ]:
        with (outdir / filename).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)


def add_rank_markers(ax, coords: np.ndarray, ranks: np.ndarray, final: bool = False) -> None:
    if final:
        ax.scatter(coords[:, 0], coords[:, 1], s=24, color="#b22222", zorder=5)
    else:
        for i, p in enumerate(coords):
            ax.scatter(p[0], p[1], s=18, color=capped_color(int(ranks[i])), edgecolor="white", linewidth=0.25, zorder=4)


def plot_paths(ax, paths: np.ndarray) -> None:
    for point_path in paths:
        for k in range(point_path.shape[0] - 1):
            ax.annotate("", xy=point_path[k + 1], xytext=point_path[k], arrowprops=dict(arrowstyle="->", lw=0.9, color=ITER_COLORS[k], shrinkA=0, shrinkB=0))


def main() -> None:
    outdir = Path(".")
    tex_path = Path("article_rank_colored_initials.tex")
    if not tex_path.exists():
        raise FileNotFoundError("article_rank_colored_initials.tex must be in the same directory as this script")
    tex = tex_path.read_text(encoding="utf-8")
    fig = extract_figure_block(tex, r"\label{fig:quadratic-ten-point-both}")
    tikz = extract_tikzpictures(fig)
    if len(tikz) < 2:
        raise RuntimeError("Could not find both Figure 4 TikZ panels")
    decision_paths = extract_paths_from_tikz(tikz[0])
    objective_paths = extract_paths_from_tikz(tikz[1])
    export_ranked_samples(decision_paths, objective_paths, outdir)

    fig_obj, axes = plt.subplots(1, 2, figsize=(9.7, 4.7), constrained_layout=True)
    axes[0].set_title("decision space")
    axes[0].plot([0, 1], [0, 1], linewidth=2.0)
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    axes[1].set_title("objective space")
    t = np.linspace(0, 1, 300)
    axes[1].plot(2 * t - t * t, 1 - t * t, linewidth=2.0)
    axes[1].set_xlabel(r"$F_1$")
    axes[1].set_ylabel(r"$F_2$")
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.grid(True, linewidth=0.4, alpha=0.45)
    plot_paths(axes[0], decision_paths)
    plot_paths(axes[1], objective_paths)
    for sample in range(objective_paths.shape[1] - 1):
        ranks = nondomination_ranks(objective_paths[:, sample, :])
        add_rank_markers(axes[0], decision_paths[:, sample, :], ranks, final=False)
        add_rank_markers(axes[1], objective_paths[:, sample, :], ranks, final=False)
    add_rank_markers(axes[0], decision_paths[:, -1, :], np.ones(decision_paths.shape[0], dtype=int), final=True)
    add_rank_markers(axes[1], objective_paths[:, -1, :], np.ones(objective_paths.shape[0], dtype=int), final=True)
    fig_obj.savefig(outdir / "figure4_rank_colored_preview.png", dpi=220)
    print("Wrote figure4_rank_colored_preview.png and ranked CSV files.")


if __name__ == "__main__":
    main()
