## Why

Ariadne 0.2.0 uses a fixed-spacing greedy 2D Pareto generator and an unconstrained random-spanning-tree baseline. The two are not built from the same biological constraints, the front is recorded without dominated-point cleanup, and the observed tree is compared against the raw traced graph. Ariadne 0.3 replaces that 2D analysis with a continuous analytical, tip-preserving generator and a matching randomized analytical baseline, and adds optional region classification, interactive reports, and a reproducible research workflow.

This change records the 0.2.0 → 0.3 release difference. The trace GUI, analyze workflow, 3D path-tortuosity analysis, tradeoff fields, alpha interpolation, scaling, and existing CSV field names remain in place.

## What Changes

- **2D Pareto front:** replace the fixed-spacing greedy front with the continuous analytical tip-preserving generator in fully greedy mode. Enforce tip preservation, binary branching, the minimum-length / nude-zone constraint, and tip completeness. Compare against the straightened RSA and remove dominated front points.
- **2D random baseline:** replace the random-spanning-tree baseline with the randomized analytical generator (`optimization_rounds = 0`, random `alpha`) using the same constraints as the front. Keep the spanning-tree model in the research workflow and retain it for 3D analysis.
- **Region classification:** add optional sampling around the actual tree and six CSV fields describing the weighted optimization level and local trade-off direction. Add GUI controls for enabling classification, sample count, and region size.
- **Interactive reports:** add Pareto and region-sampling HTML reports with tree hover panels, while retaining PNG and SVG Pareto output.
- **Research workflow:** add `research.py` and `Ariadne0.3_Exploration.ipynb`, with headless graph loading through `quantify.get_graph_from_json()`.
- **Documentation and packaging:** update the README, scientific-methods, output-fields, changelog, version metadata, and dependencies for 0.3.
- **Compatibility:** retain existing 2D/3D CSV field names, tradeoff fields, alpha interpolation, scaling, and 3D path-tortuosity analysis. No CSV column is removed; six optional classification columns are added.

## Impact

- Affected specs: `research-workflow` (new capability), `gui` (analysis-settings option labels)
- Affected code: `src/ariadne_roots/pareto_functions.py`, `src/ariadne_roots/quantify.py`, `src/ariadne_roots/main.py`, new `src/ariadne_roots/research.py`, `Ariadne0.3_Exploration.ipynb`, documentation, packaging metadata, and tests
- **Scientific behavior:** 2D Pareto values, `alpha`, scaling distance, and tradeoff values can differ from 0.2.0 because the front generator, actual reference, and random baseline change. CSV field names are unchanged.
- **Runtime:** `plotly` is added for interactive reports; Jupyter tooling is added to the development dependencies.
- **Unchanged:** trace GUI, analyze workflow, 3D path-tortuosity analysis, tradeoff fields, alpha interpolation, scaling, and existing CSV columns.
