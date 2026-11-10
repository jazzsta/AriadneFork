# Ariadne 0.3 Release Inventory

## Baseline

The comparison baseline is the published Ariadne **0.2.0** tag. It provides the trace GUI, the analyze workflow, 2D Pareto analysis, 3D path-tortuosity analysis, tradeoff fields, alpha interpolation, scaling, and the scientific/output documentation.

## Added in 0.3

### Analytical 2D Pareto front

- Continuous analytical tip-preserving generator with attachment points anywhere along an existing edge.
- Fully greedy `pareto_steiner_fast()` wrapper used by the 2D front.
- Explicit constraints: every original tip remains terminal, branching is binary, the minimum-length / nude-zone rule is enforced, and incomplete trees are rejected.
- Dominated-point cleanup through `clean_pareto_front()`.
- Straightened RSA as the actual reference point.
- Optional compact tree payloads for the interactive HTML report.

### Randomized analytical 2D baseline

- `random_steiner_wrapper_analytical()` with random `alpha` and `optimization_rounds = 0`.
- The same tip, branching, minimum-length, and completeness constraints as the optimal front.
- 100 trees by default in `analyze()`; 500 in the GUI.
- Reproducible when NumPy's global RNG is seeded.

### Region classification

- `estimate_rounds_for_region()` sampling around the actual straightened tree.
- Elliptical target region scaled by the actual tree's distance from the front and `region_factor`.
- Weighted means and 95% confidence intervals for optimization rounds and `alpha`, using Kish's effective sample size.
- GUI-facing `classify_region()` wrapper.
- Optional GUI checkbox, sample-count setting (default 5000), and region-factor setting (default 0.5).
- Six optional CSV fields: `mean_rounds_normalized`, `ci_low_normalized`, `ci_high_normalized`, `local_mean_alpha`, `local_alpha_ci_low`, and `local_alpha_ci_high`.
- Static sampling plot and interactive sampling HTML report.

### Interactive reports

- Pareto HTML report with front, actual, and random tree hover payloads.
- Region-sampling HTML report with sampled-tree payloads.
- Shared `plotly.min.js` in the output folder.
- PNG and SVG Pareto output retained alongside HTML.

### Research workflow

- Import-safe `src/ariadne_roots/research.py`.
- `Ariadne0.3_Exploration.ipynb`, following the 0.3 reference narrative.
- Headless graph loading through `quantify.get_graph_from_json()`.
- Fixed-spacing versus analytical comparisons, spanning-tree and normalized spanning-tree models, top-down and bottom-up generators, randomized analytical trees, optimization landscapes, front comparisons, and region sampling.

### Tests and documentation

- Region-classification tests.
- Analytical random-baseline reproducibility test.
- Research-module and import-boundary tests.
- Notebook-presence test.
- Updated README, scientific-methods, output-fields, changelog, and version metadata.

## Changed in 0.3

| Area | 0.2.0 | 0.3 |
|---|---|---|
| 2D front generator | Fixed-spacing greedy Steiner construction | Continuous analytical tip-preserving generator in fully greedy mode |
| 2D `alpha = 0` endpoint | `satellite_tree()` | Binary, minimum-length-constrained analytical approximation of the satellite architecture |
| 2D actual reference | Raw traced graph | Straightened RSA |
| 2D front cleanup | All evaluated `alpha` points recorded | Dominated and duplicate points removed |
| 2D random baseline | Unconstrained random spanning trees (1000) | Randomized analytical trees (100 default, 500 GUI) with the front's constraints |
| Region classification | Absent | Optional six-field classification plus sampling report |
| Analysis-settings labels | Numbered options | Numbering removed; the two compute-adding options are marked with `->` |
| Pareto reports | PNG | PNG, SVG, and interactive HTML |
| Research entry point | Development-only code and figures | `research.py` plus `Ariadne0.3_Exploration.ipynb` |
| Version | 0.2.0 | 0.3 |
| Runtime dependencies | No Plotly | Plotly for interactive reports |
| Development dependencies | No Jupyter tooling | JupyterLab, ipykernel, nbconvert |

## Retained from 0.2.0

- Trace GUI and its callbacks.
- Analyze entry point, file loading, scaling, CSV writing, and progress reporting.
- Existing 2D and 3D CSV field names.
- 2D tradeoff fields: `Tradeoff`, `Steiner_length`, `Steiner_distance`, `Satellite_length`, `Satellite_distance`, `Actual_ratio`, `Optimal_ratio`.
- Alpha interpolation and the multiplicative epsilon indicator.
- 3D path-tortuosity analysis and its spanning-tree random baseline.
- Separate `report_3d_*.csv` output.
- PNG and SVG plotting support.

## Output compatibility

- No 0.2.0 CSV column is removed.
- Six optional classification columns are added when classification is enabled.
- When classification is disabled, the six columns and the sampling report are omitted.
- 2D Pareto values, `alpha`, scaling distance, and tradeoff values can differ from 0.2.0 because the front generator, actual reference, and random baseline change.
- 3D fields and 3D analysis values are produced by the retained 0.2.0 algorithms.

## Release files

- `src/ariadne_roots/pareto_functions.py`
- `src/ariadne_roots/quantify.py`
- `src/ariadne_roots/main.py`
- `src/ariadne_roots/research.py`
- `src/ariadne_roots/scaling.py`
- `src/ariadne_roots/config.py`
- `Ariadne0.3_Exploration.ipynb`
- `README.md`
- `CHANGELOG.md`
- `docs/README.md`
- `docs/scientific-methods.md`
- `docs/output-fields.md`
- `pyproject.toml`
- `tests/test_region_classification.py`
- `tests/test_research.py`
- Updated existing tests under `tests/`
- Example architectures under `tests/data/`

## Verification

- Full test suite: **265 passed, 1 skipped**.
- Region-classification tests cover field mapping, optional behavior, empty regions, front reuse, progress reporting, and plot output.
- The analytical random baseline is reproducible under a fixed NumPy seed.
- The research module imports without opening figures or running simulations.
- The notebook exists and loads its default example graph from `tests/data`.

## Example data

- Notebook default graph: `tests/data/_set1_day1_20240429-125122_012_plantE_day1.json`.
- Additional shipped architecture: `tests/data/_set1_day1_20230320-154926_001_EtOH_EXP2_plantB_day11.json`.

## Development History

An intermediate development layout separated research and legacy code into temporary modules. That layout is not part of the 0.2.0 → 0.3 release contract and is not included in this inventory.
