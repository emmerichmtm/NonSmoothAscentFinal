#!/usr/bin/env python3
"""Reproduce the 3-D supersphere results used in the report.

This script imports the attached single-file implementation
`layered_magnitude_3d_singlefile_bulged_hv_recovery_names_dd_stochastic_box.py`
and runs the simplex-constrained supersphere benchmark.

Manuscript defaults:
  1. Main comparison table: gamma = 0.25, 0.5, 1.0 with Das--Dennis H=3
     (mu=10), for layered magnitude and layered hypervolume.
  2. Figure-5-style point-set comparison for gamma = 0.25 with H=2,3,4.
  3. Additional 3-D point-set figures for gamma = 0.5 and gamma = 1.0,
     again with H=2,3,4 and both indicators.

Numerical defaults:
  seed = 8, Das--Dennis perturbation sigma = 0.01, iterations = 80,
  exact-front threshold = 6, move = gradient.

Outputs in --outdir:
  - supersphere_3d_result_table.csv/.tex
  - supersphere_figure5_pointset_table.csv/.tex  (gamma = 0.25 only)
  - supersphere_pointset_sensitivity_all_gamma.csv/.tex
  - supersphere_figure5_coordinates_gamma025.tex
  - supersphere_figure5_coordinates_gamma05.tex
  - supersphere_figure5_coordinates_gamma10.tex
  - the underlying CSV exports written by the runner

Plotting is disabled by default for speed. Add --make-plots to also generate
PNG plots from the underlying runner.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


def load_runner(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Runner script not found: {path}")
    spec = importlib.util.spec_from_file_location("supersphere_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def effective_mu(dd_h: int) -> int:
    return (dd_h + 1) * (dd_h + 2) // 2


def base_name(runner, *, seed: int, dd_h: int, dd_sigma: float, iters: int,
              exact_front_threshold: int, gamma: float, indicator: str,
              move: str = "gradient") -> str:
    suffix = runner.make_setting_suffix(
        "bulged_three_peaks",
        seed=seed,
        n_points=effective_mu(dd_h),
        three_peaks_iters=iters,
        exact_front_threshold=exact_front_threshold,
        bulge_gamma=gamma,
        indicator=indicator,
        initialization="dasdenis",
        dd_h=dd_h,
        dd_sigma=dd_sigma,
        move=move,
    )
    return "bulged_three_peaks" + suffix


def load_csv_array(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def run_case(runner, outdir: Path, *, gamma: float, indicator: str, dd_h: int, args) -> dict:
    return runner.run_bulged_three_peaks(
        prefix="bulged_three_peaks",
        outdir=str(outdir),
        seed=args.seed,
        n_points=args.n_points,
        max_iter=args.iters,
        gamma=gamma,
        exact_front_threshold=args.exact_front_threshold,
        progress_every=args.progress_every,
        quiet=args.quiet,
        indicator=indicator,
        initialization="dasdenis",
        dd_h=dd_h,
        dd_sigma=args.dd_sigma,
        move="gradient",
    )


def collect_metrics(runner, outdir: Path, *, gamma: float, indicator: str, dd_h: int,
                    args, summary: dict | None = None) -> dict:
    base = base_name(
        runner,
        seed=args.seed,
        dd_h=dd_h,
        dd_sigma=args.dd_sigma,
        iters=args.iters,
        exact_front_threshold=args.exact_front_threshold,
        gamma=gamma,
        indicator=indicator,
    )
    Y0 = load_csv_array(outdir / f"{base}_initial_objectives.csv")
    Yf = load_csv_array(outdir / f"{base}_final_objectives.csv")
    ref = load_csv_array(outdir / f"{base}_reference_archive.csv")
    hist = load_csv_array(outdir / f"{base}_history.csv")
    iterations_completed = int(summary["iterations_completed"]) if summary else int(hist[-1, 0])
    accepted_steps = int(summary["accepted_steps"]) if summary else -1
    return {
        "gamma": gamma,
        "dd_h": dd_h,
        "mu": effective_mu(dd_h),
        "indicator": indicator,
        "iterations": iterations_completed,
        "accepted": accepted_steps,
        "mag_initial": float(runner.magnitude_3d_max_sweep_forward(Y0, (0.0, 0.0, 0.0))),
        "mag_final": float(runner.magnitude_3d_max_sweep_forward(Yf, (0.0, 0.0, 0.0))),
        "hv_initial": float(runner.hypervolume_3d_max_sweep_forward(Y0, (0.0, 0.0, 0.0))),
        "hv_final": float(runner.hypervolume_3d_max_sweep_forward(Yf, (0.0, 0.0, 0.0))),
        "igd_initial": float(runner.igd(runner.nondominated_subset(Y0), ref)),
        "igd_final": float(runner.igd(runner.nondominated_subset(Yf), ref)),
        "base": base,
    }


def write_csv(path: Path, rows: Iterable[dict], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = [
            "gamma", "dd_h", "mu", "indicator", "iterations", "accepted",
            "mag_initial", "mag_final", "hv_initial", "hv_final",
            "igd_initial", "igd_final", "base",
        ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_main_latex_table(path: Path, rows: Iterable[dict]) -> None:
    lines = [
        r"\begin{tabular}{clrrrrrrr}",
        r"\hline",
        r"$\gamma$ & Optimized & iters & acc. & $\Mag_3^0$ & $\Mag_3^f$ & $\HV_3^0$ & $\HV_3^f$ & IGD$^f$ \\",
        r"\hline",
    ]
    for row in rows:
        indicator = "Magnitude" if row["indicator"] == "magnitude" else "Hypervolume"
        acc = row["accepted"] if row["accepted"] >= 0 else "--"
        lines.append(
            f"{row['gamma']:g} & {indicator} & {row['iterations']} & {acc} & "
            f"{row['mag_initial']:.6f} & {row['mag_final']:.6f} & "
            f"{row['hv_initial']:.6f} & {row['hv_final']:.6f} & {row['igd_final']:.6f} "
            + r"\\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pointset_latex_table(path: Path, rows: Iterable[dict], include_gamma: bool = False) -> None:
    if include_gamma:
        lines = [
            r"\begin{tabular}{rrrclrrr}",
            r"\hline",
            r"$\gamma$ & $H$ & $\mu$ & Optimized & iters & $\Mag_3^f$ & $\HV_3^f$ & IGD$^f$ \\",
            r"\hline",
        ]
    else:
        lines = [
            r"\begin{tabular}{rrclrrr}",
            r"\hline",
            r"$H$ & $\mu$ & Optimized & iters & $\Mag_3^f$ & $\HV_3^f$ & IGD$^f$ \\",
            r"\hline",
        ]
    for row in rows:
        indicator = "Magnitude" if row["indicator"] == "magnitude" else "Hypervolume"
        if include_gamma:
            lines.append(
                f"{row['gamma']:g} & {row['dd_h']} & {row['mu']} & {indicator} & {row['iterations']} & "
                f"{row['mag_final']:.6f} & {row['hv_final']:.6f} & {row['igd_final']:.6f} "
                + r"\\"
            )
        else:
            lines.append(
                f"{row['dd_h']} & {row['mu']} & {indicator} & {row['iterations']} & "
                f"{row['mag_final']:.6f} & {row['hv_final']:.6f} & {row['igd_final']:.6f} "
                + r"\\"
            )
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def coordinate_block(name: str, points: np.ndarray) -> str:
    rows = [f"({p[0]:.4f},{p[1]:.4f},{p[2]:.4f})" for p in points]
    return f"% {name}\n" + " ".join(rows) + "\n"


def write_coordinates(path: Path, outdir: Path, rows: Iterable[dict]) -> None:
    blocks = []
    for row in rows:
        Yf = load_csv_array(outdir / f"{row['base']}_final_objectives.csv")
        blocks.append(coordinate_block(
            f"gamma={row['gamma']:g}, H={row['dd_h']}, mu={row['mu']}, indicator={row['indicator']}", Yf
        ))
    path.write_text("\n".join(blocks), encoding="utf-8")


def select_rows(all_rows: list[dict], gamma: float) -> list[dict]:
    return [row for row in all_rows if abs(float(row["gamma"]) - float(gamma)) < 1e-12]


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=here / "layered_magnitude_3d_singlefile_bulged_hv_recovery_names_dd_stochastic_box.py")
    parser.add_argument("--outdir", type=Path, default=here / "supersphere_3d_reproduction")
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--table-h", type=int, default=3)
    parser.add_argument("--figure-gamma", type=float, default=0.25)
    parser.add_argument("--figure-h-values", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--n-points", type=int, default=15)
    parser.add_argument("--dd-sigma", type=float, default=0.01)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--exact-front-threshold", type=int, default=6)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    args = parser.parse_args()

    runner = load_runner(args.runner)
    if not args.make_plots:
        runner.plot_objective_space = lambda *a, **k: None
        runner.plot_three_peaks_decision_space = lambda *a, **k: None
        runner.plot_convergence = lambda *a, **k: None
    args.outdir.mkdir(parents=True, exist_ok=True)

    cases = {(gamma, args.table_h, indicator) for gamma in args.gammas for indicator in ("magnitude", "hypervolume")}
    cases |= {(gamma, h, indicator) for gamma in args.gammas for h in args.figure_h_values for indicator in ("magnitude", "hypervolume")}

    summaries: dict[tuple[float, int, str], dict | None] = {}
    for gamma, dd_h, indicator in sorted(cases):
        base = base_name(
            runner,
            seed=args.seed,
            dd_h=dd_h,
            dd_sigma=args.dd_sigma,
            iters=args.iters,
            exact_front_threshold=args.exact_front_threshold,
            gamma=gamma,
            indicator=indicator,
        )
        expected = args.outdir / f"{base}_final_objectives.csv"
        if args.reuse_existing and expected.exists():
            summaries[(gamma, dd_h, indicator)] = None
        else:
            summaries[(gamma, dd_h, indicator)] = run_case(
                runner, args.outdir, gamma=gamma, indicator=indicator, dd_h=dd_h, args=args
            )

    main_rows = [
        collect_metrics(runner, args.outdir, gamma=gamma, indicator=indicator, dd_h=args.table_h,
                        args=args, summary=summaries.get((gamma, args.table_h, indicator)))
        for gamma in args.gammas for indicator in ("magnitude", "hypervolume")
    ]
    fig_rows = [
        collect_metrics(runner, args.outdir, gamma=args.figure_gamma, indicator=indicator, dd_h=dd_h,
                        args=args, summary=summaries.get((args.figure_gamma, dd_h, indicator)))
        for dd_h in args.figure_h_values for indicator in ("magnitude", "hypervolume")
    ]
    all_point_rows = [
        collect_metrics(runner, args.outdir, gamma=gamma, indicator=indicator, dd_h=dd_h,
                        args=args, summary=summaries.get((gamma, dd_h, indicator)))
        for gamma in args.gammas for dd_h in args.figure_h_values for indicator in ("magnitude", "hypervolume")
    ]

    write_csv(args.outdir / "supersphere_3d_result_table.csv", main_rows)
    write_main_latex_table(args.outdir / "supersphere_3d_result_table.tex", main_rows)
    write_csv(args.outdir / "supersphere_figure5_pointset_table.csv", fig_rows)
    write_pointset_latex_table(args.outdir / "supersphere_figure5_pointset_table.tex", fig_rows)
    write_coordinates(args.outdir / "supersphere_figure5_coordinates_gamma025.tex", args.outdir, select_rows(all_point_rows, 0.25))
    write_coordinates(args.outdir / "supersphere_figure5_coordinates_gamma05.tex", args.outdir, select_rows(all_point_rows, 0.5))
    write_coordinates(args.outdir / "supersphere_figure5_coordinates_gamma10.tex", args.outdir, select_rows(all_point_rows, 1.0))
    write_csv(args.outdir / "supersphere_pointset_sensitivity_all_gamma.csv", all_point_rows)
    write_pointset_latex_table(args.outdir / "supersphere_pointset_sensitivity_all_gamma.tex", all_point_rows, include_gamma=True)

    print("Main all-gamma table written to", args.outdir / "supersphere_3d_result_table.csv")
    print("Figure 5 point-set table written to", args.outdir / "supersphere_figure5_pointset_table.csv")
    print("Extended point-set grid written to", args.outdir / "supersphere_pointset_sensitivity_all_gamma.csv")


if __name__ == "__main__":
    main()
