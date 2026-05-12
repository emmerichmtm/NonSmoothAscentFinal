#!/usr/bin/env bash
set -euo pipefail
python3 reproduce_2d_path_figures.py
python3 reproduce_convergence_behaviour.py
python3 reproduce_h5_iteration30_vectorfield.py
