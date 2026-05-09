# Reproducibility notes

## Main scripts

- `src/reproduce_convergence_behaviour.py` is self-contained and contains the ascent methods, indicators, benchmark definitions, perturbation/backtracking recovery, and export routines for the convergence-behaviour section.
- `src/reproduce_supersphere3d_results.py` calls `src/layered_magnitude_3d_singlefile_bulged_hv_recovery_names_dd_stochastic_box.py` to reproduce the supersphere comparison tables and coordinates.
- `src/layered_clarke_ascent_2d.py` is the original compact two-dimensional projected Clarke-style ascent script retained for provenance and comparison.

## Expected outputs

The `results/` directory already contains the exported CSV/TEX files used in the current manuscript. Running the scripts again will overwrite these files with freshly generated results using the same seeds.

## Important parameters

Convergence behaviour:

- 2-D curved front: 10 points.
- 3-D simplex supersphere: gamma = 1, H = 4, mu = 15.
- 3-D box-start recovery runs: gamma = 1, H = 4 and H = 5, 500 episodes.
- 500-episode curves are exported every 20 episodes.
- The final 10 episodes are unperturbed gradient steps.

Supersphere comparison:

- gamma values: 0.25, 0.5, 1.0.
- point-set budgets: H = 2, 3, 4 where applicable.
- indicators: layered magnitude and hypervolume.
