# Magnitude Ascent reproducibility package

This repository contains the reproducibility material for the manuscript

> **Magnitude Ascent for Multiobjective Optimization: A Nonsmooth Geometric Indicator Approach**

The code reproduces the two-dimensional curved-front examples, the three-dimensional supersphere benchmark, and the convergence-behaviour experiments comparing layered magnitude and hypervolume indicators.

## Repository layout

```text
paper/
  main.tex                 # manuscript source
  main.pdf                 # compiled check PDF
  primeARXIV.sty           # minimal compatibility style for compilation
src/
  reproduce_convergence_behaviour.py
  reproduce_supersphere3d_results.py
  layered_magnitude_3d_singlefile_bulged_hv_recovery_names_dd_stochastic_box.py
  layered_clarke_ascent_2d.py
results/
  convergence/             # CSV/TEX outputs used by the convergence section
  supersphere/             # CSV/TEX outputs used by the 3-D supersphere section
docs/
  README_convergence_behaviour.md
  README_supersphere_3d_runner.md
archive/
  versioned packages used to assemble this release
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

`numba` is used when available by the convergence script; the script keeps a NumPy fallback for portability.

## Reproduce the convergence-behaviour section

```bash
python src/reproduce_convergence_behaviour.py --outdir results/convergence
```

This self-contained script runs the curved 2-D experiment and the 3-D supersphere-box convergence experiments. It writes the CSV/TEX snippets used by the manuscript, including sampled convergence curves, final objective sets, layer-size tables, CPU timings, and perturbation/backtracking event logs.

The 500-episode convergence curves are sampled every 20 episodes in the exported TikZ/TEX data to keep the manuscript self-contained.

## Reproduce the 3-D supersphere comparison tables and coordinates

```bash
python src/reproduce_supersphere3d_results.py \
  --runner src/layered_magnitude_3d_singlefile_bulged_hv_recovery_names_dd_stochastic_box.py \
  --outdir results/supersphere \
  --quiet
```

This script calls the single-file 3-D runner and reproduces the all-gamma and point-set sensitivity tables and coordinate snippets.

## Compile the manuscript

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

A minimal `primeARXIV.sty` file is included only to make the archived source compile. Replace it with the official target style file if needed.

## Quick checks

```bash
make check
```

## Notes on randomness and CPU time

The scripts use fixed random seeds for reproducibility. CPU times are machine-dependent and are reported by the convergence script using Python's process-time clock.

## License

A license has not yet been chosen. See `LICENSE.md` before making the repository public.
