## 1. Report separation

- [x] 1.1 Add `report_rsa_*.csv` for digitized RSA measurements.
- [x] 1.2 Add `report_pareto_*.csv` for straightened-RSA Pareto analysis.
- [x] 1.3 Restore `report_3d_*.csv` as the separate optional 3D report.
- [x] 1.4 Split `quantify.analyze()` results into digitized-RSA and Pareto dictionaries.
- [x] 1.5 Add digitized `Total root length` and `Travel distance` to the RSA report.
- [x] 1.6 Rename the Pareto cost fields to `Total root length (straightened)` and `Travel distance (straightened)`.

## 2. Measurement corrections

- [x] 2.1 Fix basal, branched, and apical zone boundaries so insertion segments belong to the preceding zone.
- [x] 2.2 Fix primary-root tip lookup by identifying `LR_index is None` and supporting directed and undirected graphs.
- [x] 2.3 Replace `Actual_ratio`/`Optimal_ratio` with `Length_ratio`/`Distance_ratio` in `calculate_tradeoff()`.
- [x] 2.4 Remove weak Pareto dominance by sorting on both wiring and travel.

## 3. Scaling corrections

- [x] 3.1 Scale lengths by the length factor.
- [x] 3.2 Scale `Convex Hull Area` by the squared length factor.
- [x] 3.3 Scale `LR density` and `Branched Zone density` by the inverse length factor.
- [x] 3.4 Preserve `Steiner_tortuosity_3d`, `Satellite_tortuosity_3d`, and `Coverage_tortuosity_3d` during scaling.
- [x] 3.5 Preserve the new tradeoff ratios during scaling.

## 4. Documentation and packaging

- [x] 4.1 Update `README.md` for the three CSV reports and the digitized/straightened distinction.
- [x] 4.2 Update `docs/output-fields.md` for the three-report layout, corrected definitions, and scaling rules.
- [x] 4.3 Update `docs/scientific-methods.md` for the report split, tradeoff formula, and 3D report location.
- [x] 4.4 Update `CHANGELOG.md` with the 0.3.1 release entry.
- [x] 4.5 Set the package version to `0.3.1` in `pyproject.toml` and `uv.lock`.
- [x] 4.6 Update documentation and asset links to `v0.3.1`.

## 5. Verification

- [x] 5.1 Update tradeoff, zone, scaling, and analyze tests for 0.3.1 behavior.
- [x] 5.2 Add a split-result test covering digitized and straightened cost mapping.
- [x] 5.3 Add area and inverse-length scaling tests.
- [x] 5.4 Remove redundant scaling and zone tests.
- [x] 5.5 Run the full test suite: **262 passed, 1 skipped**.
- [x] 5.6 Create and push the `v0.3.1` tag.

## 6. Review gate

- [x] 6.1 Present the report split, field-name changes, scaling corrections, and test results for review.
- [x] 6.2 Confirm that no 3D pipeline change is included in 0.3.1.
