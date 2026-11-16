## Why

Ariadne 0.3.1 is a corrective patch for output/reporting and scientific bookkeeping. The 0.3 report mixed digitized RSA measurements with straightened Pareto analysis, treated all numeric fields as lengths during unit scaling, and retained several inconsistent definitions around tradeoff fields, zone boundaries, and primary-root tip selection.

The patch makes the report structure explicit and brings the emitted values into line with the intended digitized-versus-straightened distinction.

## What Changes

- **Separate CSV reports.** `report_rsa_*.csv` carries digitized RSA measurements, `report_pareto_*.csv` carries the straightened-RSA 2D Pareto analysis, and `report_3d_*.csv` is restored as the separate optional 3D report.
- **Digitized versus straightened costs.** `report_rsa_*.csv` reports digitized `Total root length` and `Travel distance`; the Pareto report uses `Total root length (straightened)` and `Travel distance (straightened)`.
- **Corrected measurement definitions.** Zone boundaries assign insertion segments to the preceding zone, and primary-root tip lookup uses `LR_index is None` and works with directed or undirected graphs.
- **Corrected tradeoff fields.** `calculate_tradeoff()` uses `Length_ratio` and `Distance_ratio`, with `Tradeoff = Length_ratio / Distance_ratio`.
- **Dimension-aware scaling.** Lengths use the scale factor, areas use its square, inverse-length densities use its inverse, and dimensionless 3D corner tortuosity fields remain unchanged.
- **Corrected Pareto cleaning.** Weakly dominated front points are removed by sorting on both wiring and travel.
- **Documentation and tests.** README, scientific-methods, output-fields, changelog, version metadata, and tests are updated for 0.3.1.

## Impact

- Affected specs: `csv-output`, `analysis-output`, `documentation`, `testing`, `gui`
- Affected code: `src/ariadne_roots/quantify.py`, `src/ariadne_roots/scaling.py`, `src/ariadne_roots/main.py`, `src/ariadne_roots/pareto_functions.py`
- Affected outputs: CSV report names and columns, scaling behavior, tradeoff fields, zone fields, primary-root minimal length
- **Compatibility:** the single 0.3 2D CSV is replaced by two 2D reports; `report_3d_*.csv` remains separate. The old tradeoff component names `Actual_ratio` and `Optimal_ratio` are replaced by `Length_ratio` and `Distance_ratio`.
