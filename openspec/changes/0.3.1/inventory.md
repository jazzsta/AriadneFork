# Ariadne 0.3.1 Release Inventory

## Baseline

The comparison baseline is Ariadne **0.3**. It writes a single 2D report mixing digitized measurements and straightened Pareto analysis, plus a separate 3D report in the GUI workflow.

## Added in 0.3.1

### Separate CSV reports

- `report_rsa_<timestamp>.csv` for digitized RSA measurements.
- `report_pareto_<timestamp>.csv` for the straightened-RSA 2D Pareto analysis.
- `report_3d_<timestamp>.csv` restored as the separate optional 3D report.

### Digitized cost fields

- `Total root length` and `Travel distance` in the RSA report are computed from the original digitized graph.
- The Pareto report uses `Total root length (straightened)` and `Travel distance (straightened)`.

### Verification

- Split-result test for digitized versus straightened cost mapping.
- Area and inverse-length scaling tests.
- Full test suite: **262 passed, 1 skipped**.

## Changed in 0.3.1

| Area | 0.3 | 0.3.1 |
|---|---|---|
| 2D CSV outputs | One mixed 2D report | Separate RSA and Pareto reports |
| RSA total/travel costs | Not reported separately | Digitized `Total root length` and `Travel distance` |
| Pareto cost names | `Total root length`, `Travel distance` | `Total root length (straightened)`, `Travel distance (straightened)` |
| Tradeoff components | `Actual_ratio`, `Optimal_ratio` | `Length_ratio`, `Distance_ratio` |
| Area scaling | Length factor | Squared length factor |
| Density scaling | Unchanged | Inverse length factor |
| 3D corner tortuosity scaling | Length-scaled by generic path | Preserved |
| 3D CSV | Separate file in the GUI workflow | Separate `report_3d_*.csv` retained |
| Zone boundaries | Insertion segment assigned to following zone | Insertion segment assigned to preceding zone |
| Primary-root tip | Deepest descendant | `LR_index is None` tip, directed or undirected |
| Weak Pareto dominance | Equal-wiring dominated points could remain | Removed by sorting on wiring and travel |

## Fixed in 0.3.1

- Basal, branched, and apical zone boundary assignment.
- Primary-root minimal length tip selection.
- Tradeoff ratio formulation and CSV field names.
- Dimension-aware scaling for areas, densities, and 3D corner tortuosity.
- Weak-dominance removal in `clean_pareto_front()`.

## Release files

- `src/ariadne_roots/quantify.py`
- `src/ariadne_roots/scaling.py`
- `src/ariadne_roots/main.py`
- `src/ariadne_roots/pareto_functions.py`
- `README.md`
- `CHANGELOG.md`
- `docs/README.md`
- `docs/scientific-methods.md`
- `docs/output-fields.md`
- `pyproject.toml`
- `uv.lock`
- `tests/test_quantify.py`
- `tests/test_scaling.py`
- `tests/test_main.py`

## Unchanged

- Trace GUI and tracing workflow.
- 3D Pareto algorithm and its comparison with the raw digitized graph.
- 3D random spanning-tree baseline.
- 2D analytical tip-preserving front and randomized analytical baseline.
- Region-classification procedure.
- Interactive Pareto and sampling HTML reports.
