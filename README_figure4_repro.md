# Figure 4 reproducibility files

This package reproduces the longer-run replacement for Figure 4.

## Files

- `reproduce_figure4_result.py` — self-contained Python script that runs the projected finite-difference diffusion and exports CSV and PNG output.
- `layered_clarke_ascent_2d.py` — supporting core script from the manuscript package.
- `layered_magnitude_core_2d.py` — supporting core script from the manuscript package.
- `requirements_figure4.txt` — minimal Python dependencies.

## Parameters used for the reproduced Figure 4

- Projection box: `[0,1]^2`
- Step size: `alpha = 0.004`
- Iterations: `max_iter = 540`
- Layer weight: `eps = 1e-3`
- Repulsion strength: `tau = 2e-4`
- Repulsion length scale: `sigma = 0.03`
- Finite-difference step: `h = 1e-5`

The repulsion term is active in the objective-space layered value:

```python
value -= tau * repulsion(Y, sigma)
```

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_figure4.txt
python reproduce_figure4_result.py
```

The script writes:

- `figure4_decision_paths.csv`
- `figure4_objective_paths.csv`
- `figure4_initial_final_decision.csv`
- `figure4_initial_final_objective.csv`
- `figure4_reproduced.png`
