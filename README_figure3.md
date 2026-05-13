# Figure 3 recomputation files

This bundle documents the two `mu=10` objective-space runs shown side by side in the updated Figure 3.

## Included scenarios

1. **Old Figure 3 / left panel**: nondominated line start on `F1 + F2 = 0.7`.
2. **New Figure 3 / right panel**: dominated triangular `4+3+2+1` start in the lower-left subtriangle with vertices `(0,0)`, `(0,0.5)`, and `(0.5,0)`.

## Python file

`recompute_figure3_two_initializations.py` exports the sampled path coordinates used in the paper.
For each scenario it writes one CSV per path and one summary CSV with the initial and final points.

## How to use

Run

```bash
python recompute_figure3_two_initializations.py
```

in the directory where you want the CSV files to be written.

## What this reproduces

The script reproduces the **sampled path coordinates used in the TikZ figure** inside the article source. It is intended for transparent verification and figure regeneration.
