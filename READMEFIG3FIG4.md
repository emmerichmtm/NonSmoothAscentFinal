# Rank-colored Figure 3 and Figure 4 reproducibility package

This flat package contains the current article files and Python scripts for regenerating the rank-colored previews and CSV data for Figures 3 and 4.

## Contents

- `article_rank_colored_initials.pdf` - compiled article version with rank-colored initial/intermediate markers.
- `article_rank_colored_initials.tex` - matching LaTeX source.
- `article_rank_colored_initials.patch.diff` - patch relative to the previous working source.
- `recompute_rank_colored_fig3.py` - regenerates the sampled, rank-colored Figure 3 data and a PNG preview.
- `recompute_rank_colored_fig4.py` - reconstructs the sampled Figure 4 paths from the embedded article source, assigns objective-space dominance ranks, and creates a PNG preview.
- `reproduce_figure4_result.py` - numerical support code used by the Figure 4 recomputation script.
- `requirements.txt` - minimal Python dependencies.

## Layer color convention

The scripts use the same point-color palette as Figure 1 in the article:

- Layer 1: dark blue (`#0b1bcd`)
- Layer 2: dark green (`#0b1b05`)
- Layer 3 and deeper: dark orange/red-orange (`#fd120b`)

Final iterates are shown in red so the end points remain easy to identify.

## Recomputing Figure 3 data and preview

```bash
python recompute_rank_colored_fig3.py
```

This writes:

- `figure3_rank_colored_preview.png`
- `figure3_line_start_ranked_samples.csv`
- `figure3_dominated_triangle_start_ranked_samples.csv`

## Recomputing Figure 4 data and preview

```bash
python recompute_rank_colored_fig4.py
```

This writes:

- `figure4_rank_colored_preview.png`
- `figure4_decision_ranked_samples.csv`
- `figure4_objective_ranked_samples.csv`

The Figure 4 coordinates were produced by the projected finite-difference run with the active repulsion term, using the article parameters:

- `alpha = 0.004`
- `max_iter = 540`
- `eps = 1e-3`
- `tau = 2e-4`
- `sigma = 0.03`
- `h = 1e-5`

## Notes

The LaTeX article itself embeds TikZ/PGF coordinates, so no external image files are needed to compile the paper. The Figure 4 recomputation script reads these embedded coordinates to regenerate the plotted data quickly. The original full-run support script is included as `reproduce_figure4_result.py` for reference.
