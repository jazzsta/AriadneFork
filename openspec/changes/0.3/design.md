## Context

Ariadne **0.2.0** is the published baseline (tag `v0.2.0`). It provides the trace GUI, the analyze workflow, 2D Pareto analysis, 3D path-tortuosity Pareto analysis, tradeoff fields, alpha interpolation, and the scientific/output documentation.

Ariadne **0.3** is the next development release. Its release-level scientific changes are concentrated in the 2D Pareto analysis: the front generator, the random baseline, and the optional region classification. It also adds interactive reports and a reproducible research workflow. The trace GUI, the analyze pipeline, the 3D path-tortuosity analysis, scaling, tradeoff fields, and existing CSV field names are retained.

This document describes the differences between the two releases. It does not describe the internal development history of 0.3.

## Goals

- Replace the v0.2.0 fixed-spacing 2D front with a continuous analytical, tip-preserving front.
- Use the same biological constraints for the 0.3 random baseline, so optimal and null architectures are directly comparable.
- Add optional region classification for optimization level and local trade-off direction.
- Add interactive HTML reports that show the RSA behind each Pareto or sampling point.
- Ship a reproducible research workflow (`research.py` and `Ariadne0.3_Exploration.ipynb`) for the algorithms and figures behind 0.3.
- Keep the trace GUI, analyze workflow, 3D analysis, and existing CSV field names compatible with 0.2.0.
- Document the new algorithms, outputs, and release differences.

## Non-Goals

- Changing the tracing GUI or the interactive tracing workflow.
- Replacing or removing the 3D path-tortuosity analysis introduced in 0.2.0.
- Removing the 2D tradeoff fields, alpha interpolation, or scaling behavior.
- Preserving numerical equality with v0.2.0 2D Pareto values; the 2D generator and actual reference intentionally change.
- Publishing 0.3 to the public package index as part of this change.

## Baseline: Ariadne 0.2.0

| Area | 0.2.0 behavior |
|---|---|
| 2D front | Fixed-spacing greedy Steiner construction over 101 `alpha` values. The `alpha = 0` endpoint uses `satellite_tree()`. No explicit tip-preservation or binary-branching constraint. |
| Front cleanup | Every evaluated `alpha` is recorded; dominated points are not removed. |
| Actual reference | Costs of the traced graph passed to `pareto_front()`. |
| 2D random baseline | Random spanning trees over the base and tips, following Conn et al. (2017). 1000 unconstrained random trees per analysis. |
| Region classification | Not available. |
| Reports | Pareto plots in PNG (2D and 3D). |
| CSV | Base traits, 2D Pareto fields, tradeoff fields, and optional 3D fields. No classification fields. |
| Research workflow | Algorithms and comparison figures exist in the development code but are not shipped as a single research module and notebook. |
| Runtime dependency | No Plotly dependency. |

## Release Differences

### 1. Analytical tip-preserving 2D Pareto front

**0.2.0**

`pareto_front()` calls `pareto_steiner_fast()` for each `alpha`. That generator places new attachment points on a fixed discrete grid using `STEINER_MIDPOINTS`, and it uses `satellite_tree()` for the `alpha = 0` endpoint. It does not enforce tip preservation or binary branching as explicit constraints.

**0.3**

The 2D front is built with the continuous analytical generator:

- `pareto_steiner_fast_analytical_preserve_tips_semi_random()` places each new attachment point anywhere along an existing edge.
- `pareto_steiner_fast()` runs that generator in fully greedy mode by using enough optimization rounds to make every decision greedy.
- Every original tip remains a terminal node; tips are never consumed as internal branch points.
- Each new internal node has two children, so branching is binary.
- The minimum-length / nude-zone constraint is enforced at both ends of every split edge and around every attached tip.
- Constructions that cannot connect every original tip are rejected, so incomplete trees cannot enter the front.
- `pareto_front()` evaluates the same `DEFAULT_ALPHAS` range as 0.2.0, but records only non-dominated front points through `clean_pareto_front()`.
- The actual reference is now the straightened RSA (`straighten_tree()`), not the raw traced graph.
- The `alpha = 0` endpoint is the binary, minimum-length-constrained analytical approximation of the satellite architecture, not the non-binary `satellite_tree()`.
- `pareto_front()` can optionally collect compact tree payloads for the interactive HTML report.

**Consequence:** 2D Pareto values, `alpha`, scaling distance, and tradeoff values can differ from 0.2.0 even for the same input file. The CSV field names do not change.

### 2. Randomized analytical 2D baseline

**0.2.0**

`pareto_calcs()` calls `random_tree()`, which generates 1000 unconstrained random spanning trees over the base and root tips. The reported random values are the means of those trees.

**0.3**

The 2D CSV baseline is `random_steiner_wrapper_analytical()`:

- `alpha` is drawn uniformly from `[0, 1]` for each sampled tree.
- `optimization_rounds = 0`, so every connection is chosen randomly among valid analytical attachments.
- Tip preservation, binary branching, the minimum-length constraint, and tip completeness are enforced exactly as in the optimal front.
- The reported random costs are the means over the generated trees.
- `analyze()` defaults to 100 random trees; the GUI uses 500.
- The baseline is reproducible when NumPy's global RNG is seeded.

The v0.2.0 spanning-tree model remains available in the research workflow (`random_tree()`, `random_tree_normalized()`), and the 3D analysis still uses its spanning-tree random baseline. It is no longer the 2D CSV baseline.

### 3. Optional region classification

**0.2.0**

No classification workflow or CSV fields.

**0.3**

`estimate_rounds_for_region()` samples semi-random analytical architectures and measures how much greedy optimization and which trade-off direction best describe an actual root:

- The target region is an ellipse centered on the actual straightened tree.
- Ellipse radii scale with the actual tree's distance from the front and the `region_factor` setting.
- Each sample draws `alpha ~ Uniform(0, 1)` and an optimization-round budget between 0 and the number of tips.
- Accepted trees receive a weight of `1 - d_norm`, so trees near the actual root count more than trees near the ellipse boundary.
- Weighted means and 95% confidence intervals are computed for the normalized optimization level and `alpha`, using Kish's effective sample size.
- `classify_region()` is the GUI-facing wrapper and can save the static sampling plot and the interactive sampling HTML report.
- The GUI adds a "Run region classification" checkbox, a sample-count setting (default 5000), and a region-factor setting (default 0.5).
- The analysis-settings dialog no longer numbers its options. The two options that add computation, 3D Pareto and region classification, are marked with a leading `->`.
- The six classification fields are `mean_rounds_normalized`, `ci_low_normalized`, `ci_high_normalized`, `local_mean_alpha`, `local_alpha_ci_low`, and `local_alpha_ci_high`.
- Classification is optional. When disabled, all six columns are omitted from the 2D CSV and no sampling HTML is produced.

### 4. Interactive HTML reports

**0.2.0**

Pareto plots are written as PNG files.

**0.3**

- Each analyzed root produces a Pareto plot in PNG, SVG, and interactive HTML.
- The Pareto HTML report can show the front point, actual tree, or random tree under the cursor.
- When region classification is enabled, a second interactive HTML report shows the sampled architectures and the target ellipse.
- Tree geometries are serialized as compact node/edge payloads rather than embedded Matplotlib figures.
- The reports share a single `plotly.min.js` file in the output folder.
- 3D Pareto output remains PNG and SVG only.

### 5. Research workflow and notebook

**0.2.0**

The research algorithms and comparison figures are not exposed as a single import-safe workflow.

**0.3**

- `research.py` provides an import-safe, IPython-friendly implementation of the research story: straightening and tip preservation, fixed-spacing versus analytical comparisons, random null models, optimization-round landscapes, front comparisons, and region sampling.
- `Ariadne0.3_Exploration.ipynb` is the primary research entry point.
- Graph loading for the notebook is provided by the headless `quantify.get_graph_from_json()` helper, so the workflow does not require Tkinter.
- The notebook ships with example graphs under `tests/data`.
- `research.py` has no `demo()` function and no `if __name__ == "__main__":` block.

### 6. Documentation, tests, and packaging

**0.2.0 → 0.3**

- `README.md`, `docs/scientific-methods.md`, and `docs/output-fields.md` describe the analytical front, the randomized analytical baseline, optional region classification, and the notebook.
- The documentation distinguishes 0.3 behavior from the 0.2.0 algorithms and states that 2D numerical values can differ.
- `CHANGELOG.md` records the release-level changes and links 0.3 against `v0.2.0`.
- `pyproject.toml` identifies the package as `0.3`, adds `plotly` as a runtime dependency, and adds Jupyter tooling to the development group.
- New tests cover region classification, the analytical baseline, reproducibility with a fixed seed, the research module boundary, and notebook presence.

## What Remains Compatible

- The trace GUI and its callbacks are unchanged.
- The analyze entry point, file loading, output folder selection, scaling, CSV writing, and progress reporting are retained.
- Existing 2D and 3D CSV field names are retained.
- The 2D tradeoff fields (`Tradeoff`, `Steiner_length`, `Steiner_distance`, `Satellite_length`, `Satellite_distance`, `Actual_ratio`, `Optimal_ratio`) are retained.
- Alpha interpolation and the multiplicative epsilon indicator are retained.
- The 3D path-tortuosity analysis and its spanning-tree random baseline are unchanged.
- 3D output remains a separate `report_3d_*.csv`.

The intentional incompatibility is scientific, not structural: 2D front points, the actual reference, random-baseline values, and derived `alpha`/scaling/tradeoff values can differ from 0.2.0 because the underlying 2D algorithms changed.

## Decisions

- **Continuous analytical attachment over fixed spacing.** It removes the grid-resolution artifact and makes the biological constraints explicit.
- **One generator family for the front and the 2D null model.** Optimal and random architectures then differ only in how much greedy optimization is applied, not in their constraints.
- **Straightened tree as the actual reference.** The front is computed on a straightened graph, so the observed tree is compared on the same representation.
- **Optional classification.** Users can keep the 0.2.0 2D column set when classification is not needed, and the extra sampling cost is opt-in at analysis time.
- **Notebook as the research entry point.** It makes the workflow reproducible and keeps import-time side effects out of the package.
- **Keep 3D analysis unchanged.** The 0.3 release focuses on the 2D front and classification; changing 3D would add unnecessary scientific and compatibility risk.

## Alternatives Considered

- **Keep the 0.2.0 fixed-spacing front.** Rejected because the fixed grid and missing explicit tip/binary constraints are the problems 0.3 is intended to solve.
- **Keep the spanning-tree random baseline for 2D.** Rejected because it does not share the front's biological constraints, which makes the optimal-versus-random comparison less controlled.
- **Run classification unconditionally.** Rejected because sampling is expensive and users may want the original 0.2.0 column set.
- **Expose a `demo()` function instead of a notebook.** Rejected because a demo hides the intermediate computations and encourages import-time execution.

## Risks / Trade-offs

- **2D values change between releases.** Documented in the README, scientific methods, output fields, and changelog; users should not compare 0.2.0 and 0.3 2D values directly.
- **Analytical generation is slower than the fixed-grid greedy path.** The GUI uses background work and progress reporting; the default random baseline is 500 trees in the GUI and 100 in `analyze()`.
- **Region classification adds a large sampling cost.** It is optional, defaults to 5000 samples, and its columns are omitted when disabled.
- **Classification can find no samples inside a very small region.** Empty results are written as `None` rather than fabricated values.
- **HTML reports can be large.** Tree payloads are compact, and Plotly is loaded from a shared local file.
- **The notebook is publication-scale by default.** Sample counts are declared at the top so users can reduce them for exploratory runs.

## Migration / Release Plan

1. Use Ariadne 0.3 from the repository; the published 0.2.0 package remains the prior release.
2. Re-run existing analyses if 2D front-derived values are part of a comparison. Keep 0.2.0 results separate from 0.3 results.
3. Disable region classification when the six new columns are not needed.
4. Use the 3D report as before; its algorithm and field set are unchanged.
5. Consult the updated scientific methods and output-field documentation for the 0.3 definitions.
6. Transfer the 0.3 release files only after the version comparison and test results have been reviewed.

## Verification

- Full test suite: **265 passed, 1 skipped**.
- New coverage includes the analytical front, analytical random-baseline reproducibility, region-classification field mapping, research import boundaries, and notebook presence.
- The 0.2.0 CSV field set is retained apart from the six optional classification columns.
- Manual review covers the GUI controls, HTML reports, and notebook setup path.

## Development History

During 0.3 development the scientific code was temporarily reorganized into separate research and legacy modules so the new workflow could be reviewed in isolation. That intermediate layout is not part of the 0.2.0 → 0.3 user-facing contract and is intentionally omitted from this comparison.

## Open Questions

None. The release-level behavior, compatibility boundary, and documentation scope are defined above.
