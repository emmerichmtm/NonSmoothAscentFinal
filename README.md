# Reproducibility package: layered magnitude set gradients

This flat archive contains the manuscript source, compiled manuscript PDF, and reproduction scripts for the experiments and figures in the article.

## Files

- `layered_magnitude_set_gradient_article.tex` — main LaTeX source of the article.
- `layered_magnitude_set_gradient_article.pdf` — compiled PDF corresponding to the source in this package.
- `PRIMEarxiv.sty` — LaTeX style file used by the article.
- `layered_magnitude_core_2d.py` — core projected finite-difference ascent code for the two-objective examples.
- `reproduce_2d_path_figures.py` — exports sparse CSV samples for the two-dimensional path figures.
- `reproduce_convergence_behaviour.py` — reproduces the convergence-behaviour experiments and tables for the supersphere examples.
- `reproduce_h5_iteration30_vectorfield.py` — reproduces the H=5, iteration-30 magnitude vector-field snapshot and exports PNG/TikZ output.
- `figure_h5_iteration30_vectorfield_standalone.tex` — standalone LaTeX/TikZ source for the vector-field illustration.
- `figure_h5_iteration30_vectorfield_standalone.pdf` — compiled standalone vector-field figure.
- `figure_h5_iteration30_vectorfield_preview.png` — PNG preview of the vector-field illustration.
- `requirements.txt` — minimal Python requirements.
- `compile_article.sh` — convenience script to compile the article with LuaLaTeX.
- `run_reproduction_scripts.sh` — convenience script to run the reproduction scripts.

## Python environment

The scripts require Python 3.10 or newer and the packages listed in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts are written to avoid optional acceleration dependencies. They use NumPy for numerical work and Matplotlib for the vector-field preview.

## Compile the article

A TeX installation with LuaLaTeX and PGFPlots/TikZ is required. From the directory containing this README:

```bash
./compile_article.sh
```

This runs LuaLaTeX twice on `layered_magnitude_set_gradient_article.tex`.

## Reproduce numerical outputs

To run all reproduction scripts:

```bash
./run_reproduction_scripts.sh
```

The scripts write CSV, PNG, and TikZ/PDF-related output files into the current directory. Runtime depends on the machine and whether the full convergence experiment script is run with default settings.

## Notes

- This package is intentionally flat: there are no subfolders inside the archive.
- The manuscript itself embeds the TikZ/PGF figure code needed for compilation.
- The vector-field standalone files are included as convenience artifacts for checking the H=5 iteration-30 illustration independently.
