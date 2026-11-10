## 1. Baseline and release scope

- [x] 1.1 Review the published Ariadne 0.2.0 source and documentation as the comparison baseline.
- [x] 1.2 Separate release-level 0.3 changes from internal development history.
- [x] 1.3 Confirm the retained 0.2.0 surface: trace GUI, analyze workflow, 3D path-tortuosity analysis, tradeoff fields, alpha interpolation, scaling, and existing CSV field names.
- [x] 1.4 Record the intentional 2D scientific incompatibility: front values, actual reference, random-baseline values, and derived `alpha`/scaling/tradeoff values can change.

## 2. Analytical tip-preserving 2D front

- [x] 2.1 Implement the continuous analytical generator with attachment points anywhere along an existing edge.
- [x] 2.2 Enforce tip preservation, binary branching, the minimum-length / nude-zone constraint, and tip completeness.
- [x] 2.3 Add the fully greedy `pareto_steiner_fast()` wrapper used by the 2D front.
- [x] 2.4 Use the straightened RSA as the actual reference in `pareto_front()`.
- [x] 2.5 Remove dominated and duplicate front points with `clean_pareto_front()`.
- [x] 2.6 Preserve the `alpha = 0` / `alpha = 1` endpoint semantics in the 0.3 documentation.
- [x] 2.7 Support optional compact tree payloads for the interactive HTML report.
- [x] 2.8 Keep the 3D Pareto front and its fixed-spacing 3D generator unchanged.

## 3. Randomized analytical 2D baseline

- [x] 3.1 Implement `random_steiner_wrapper_analytical()` with random `alpha` and `optimization_rounds = 0`.
- [x] 3.2 Apply the same tip, branching, minimum-length, and completeness constraints as the optimal front.
- [x] 3.3 Use the analytical baseline in the 2D CSV report, with 100 trees by default and 500 in the GUI.
- [x] 3.4 Keep the spanning-tree random models available in the research workflow.
- [x] 3.5 Retain the spanning-tree baseline for 3D analysis.
- [x] 3.6 Add a reproducibility test using a fixed NumPy seed.

## 4. Region classification

- [x] 4.1 Implement sampling around the actual straightened tree with an elliptical target region.
- [x] 4.2 Implement weighted means and confidence intervals for optimization rounds and `alpha`.
- [x] 4.3 Add the six CSV fields: `mean_rounds_normalized`, `ci_low_normalized`, `ci_high_normalized`, `local_mean_alpha`, `local_alpha_ci_low`, and `local_alpha_ci_high`.
- [x] 4.4 Add the GUI classification checkbox, sample-count setting, and region-factor setting.
- [x] 4.5 Omit the six fields and the sampling report when classification is disabled.
- [x] 4.6 Preserve empty-region results as `None` rather than fabricating values.
- [x] 4.7 Add region-classification tests for field mapping, optional behavior, progress reporting, and plot output.
- [x] 4.8 Remove incoherent numbering from the analysis-settings options and mark the two compute-adding choices with `->`.

## 5. Interactive reports

- [x] 5.1 Add the interactive Pareto HTML report with front, actual, and random tree hover payloads.
- [x] 5.2 Add the interactive region-sampling HTML report with sampled-tree payloads.
- [x] 5.3 Retain PNG and SVG Pareto output alongside HTML.
- [x] 5.4 Share one `plotly.min.js` file across reports in the output folder.
- [x] 5.5 Keep 3D Pareto output in PNG and SVG only.

## 6. Research workflow and notebook

- [x] 6.1 Create import-safe `src/ariadne_roots/research.py` for the 0.3 research workflow.
- [x] 6.2 Include straightening, fixed-spacing versus analytical comparisons, random null models, optimization landscapes, front comparisons, and region sampling.
- [x] 6.3 Keep computation functions before the plots that visualize them.
- [x] 6.4 Move headless graph loading to `quantify.get_graph_from_json()` so the notebook does not require Tkinter.
- [x] 6.5 Add `Ariadne0.3_Exploration.ipynb` with the reference-document order and example data under `tests/data`.
- [x] 6.6 Confirm there is no `demo()` function or `if __name__ == "__main__":` block in `research.py`.
- [x] 6.7 Add research-module, import-boundary, and notebook-presence tests.

## 7. Documentation, packaging, and verification

- [x] 7.1 Update `README.md` with the 0.2.0 → 0.3 changes, compatibility note, GUI controls, and notebook workflow.
- [x] 7.2 Update `docs/scientific-methods.md` with the analytical front, randomized analytical baseline, and region-classification procedure.
- [x] 7.3 Update `docs/output-fields.md` with the optional classification fields and the 0.3 random-baseline semantics.
- [x] 7.4 Update `docs/README.md` to describe the current module layout and remove obsolete temporary-module notes.
- [x] 7.5 Update `CHANGELOG.md` and the version link for the 0.3 release.
- [x] 7.6 Set the package version to `0.3`, add `plotly`, and add Jupyter development dependencies.
- [x] 7.7 Run the full test suite and record the result: **265 passed, 1 skipped**.
- [x] 7.8 Run the GUI, HTML-report, and notebook setup smoke checks.
- [x] 7.9 Update README, output-field notes, and GUI-label tests for the `->` analysis-settings labels.

## 8. Review gate

- [x] 8.1 Present the 0.2.0 → 0.3 behavior difference, compatibility boundary, and affected files for review.
- [x] 8.2 Confirm that no 0.2.0 CSV column is removed and that the six classification columns are optional.
- [x] 8.3 Confirm that 3D path-tortuosity analysis, tradeoff fields, alpha interpolation, and scaling are retained.
- [x] 8.4 Confirm that the change documents describe release differences rather than internal development steps.
