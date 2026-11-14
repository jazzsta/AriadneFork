# Output Fields Reference

This document describes the CSV fields produced by **Ariadne 0.3.1**.

Ariadne writes two CSV reports per analysis run, plus an optional third 3D report. All reports contain one row per analyzed root and include a `filename` column.

| Report | Contents | Data source |
|---|---|---|
| `report_rsa_<timestamp>.csv` | Basic RSA measurements: root lengths, travel distance, lateral-root counts and angles, zones, tortuosity, barycenter, convex-hull area. | Original digitized graph. No straightening and no Pareto/random analysis. |
| `report_pareto_<timestamp>.csv` | 2D Pareto analysis: front costs, optimality distance, random baseline, tradeoff fields, and optional region classification. | Straightened RSA and generated Pareto/random architectures. |
| `report_3d_<timestamp>.csv` | Optional 3D Pareto analysis with path coverage as a third objective. | Original digitized graph. |

## Changes from Ariadne 0.2.0

Ariadne 0.3.1 splits the previous single 2D report into the `report_rsa_*.csv` and `report_pareto_*.csv` reports described above, and keeps the optional 3D analysis in its own `report_3d_*.csv`. The digitized-RSA report keeps the measurements that depend only on the traced root system; the Pareto report keeps the straightened-tree and optimality analysis.

- 2D Pareto values, `alpha`, scaling distance, and tradeoff values can differ from 0.2.0 because the 2D front now uses the analytical tip-preserving generator and compares against the straightened RSA.
- The tradeoff component fields are now `Length_ratio` (`actual_length / steiner_length`) and `Distance_ratio` (`actual_distance / satellite_distance`); they replace the earlier `Actual_ratio` and `Optimal_ratio` names.
- Zone lengths now assign each primary-root segment to the zone in which it ends, so segments ending at the first and last lateral-root insertions are counted in the basal and branched zones respectively.
- `Total root length (random)`, `Travel distance (random)`, `alpha (random)`, and `scaling (random)` now describe the randomized analytical baseline rather than the 0.2.0 random-spanning-tree model.
- Scaling is now dimension-aware: lengths use the scale factor, areas use its square, and inverse-length densities use its inverse.
- The 3D report remains separate as `report_3d_<timestamp>.csv` and uses the original digitized graph. The old spanning-tree 3D baseline is unchanged from 0.2.0.
- When region classification is disabled, the six classification columns and the sampling report are omitted.

## Units and Scaling

Length measurements are reported in the units selected during analysis:

- if scaling was configured (for example `mm`), lengths use those units;
- otherwise, lengths are in pixels.

Scaling is applied according to each field's dimension:

| Dimension | Scaling rule | Examples |
|---|---|---|
| Length | multiply by `s` | `PR length`, `Travel distance`, `Steiner_length`, `Barycenter x displacement` |
| Area | multiply by `s²` | `Convex Hull Area` |
| Inverse length | divide by `s` | `LR density`, `Branched Zone density` |
| Dimensionless | unchanged | `alpha`, `Tradeoff`, `Tortuosity`, angles, counts, 3D ratios |

Here `s` is the configured length scale factor (real units per pixel). Ratios, angles, and counts are dimensionless.

---

## RSA Measurements (`report_rsa_<timestamp>.csv`)

These fields come directly from the digitized graph. They do not use the straightened RSA, the Pareto front, or random architectures.

### Basic Digitized Measurements

| Field | Description | Units | Range |
|---|---|---|---|
| `Total root length` | Sum of all edge lengths in the digitized root network | length | > 0 |
| `Travel distance` | Sum of path lengths from the base to every root tip along the digitized network | length | > 0 |

The digitized total root length is also equal to `PR length + Σ LR lengths` when all root axes are included.

### Primary Root Metrics

| Field | Description | Units | Range |
|---|---|---|---|
| `PR length` | Length of the primary root along its digitized axis | length | > 0 |
| `PR_minimal_length` | Straight-line distance from the hypocotyl to the primary-root tip (the unique tip with `LR_index is None`) | length | > 0 |

### Lateral Root Metrics

| Field | Description | Units | Range |
|---|---|---|---|
| `LR count` | Number of lateral roots | count | ≥ 0 |
| `LR density` | Lateral-root count divided by primary-root length | 1/length | ≥ 0 |
| `Mean LR lengths` | Mean lateral-root length | length | > 0 |
| `Median LR lengths` | Median lateral-root length | length | > 0 |
| `Mean LR minimal lengths` | Mean straight-line distance from each LR tip to its insertion on its parent root | length | ≥ 0 |
| `Median LR minimal lengths` | Median straight-line LR distance to the insertion on its parent root | length | ≥ 0 |
| `sum LR minimal lengths` | Sum of straight-line LR distances to the insertion on its parent root | length | ≥ 0 |
| `Mean LR angles` | Mean lateral-root set-point angle relative to gravity | degrees | 0–180 |
| `Median LR angles` | Median lateral-root set-point angle relative to gravity | degrees | 0–180 |
| `LR lengths` | Array of individual lateral-root lengths | length[] | > 0 |
| `LR angles` | Array of individual lateral-root angles relative to gravity | degrees[] | 0–180 |
| `LR minimal lengths` | Array of individual straight-line distances from each LR tip to its insertion on its parent root | length[] | ≥ 0 |

`LR count` and the lateral-root arrays include all root axes with an `LR_index`, including first-order and higher-order laterals.

### Zone Measurements

| Field | Description | Units | Range |
|---|---|---|---|
| `Basal Zone length` | Length along the primary root from the hypocotyl to the first lateral-root insertion, including the segment ending at that insertion node | length | ≥ 0 |
| `Branched Zone length` | Length along the primary root from the first insertion to the last insertion, including the segment ending at the last insertion node | length | ≥ 0 |
| `Apical Zone length` | Length along the primary root after the last insertion to the primary-root tip | length | ≥ 0 |
| `Branched Zone density` | LR count divided by branched-zone length | 1/length | ≥ 0 |

### Other Digitized Measurements

| Field | Description | Units | Range |
|---|---|---|---|
| `Barycenter x displacement` | Horizontal displacement from the uppermost node to the mean of the convex-hull vertices | length | any |
| `Barycenter y displacement` | Vertical displacement from the uppermost node to the mean of the convex-hull vertices | length | any |
| `Total minimal Distance` | Primary-root base-to-tip straight-line distance plus the sum of lateral-root insertion-to-tip straight-line distances | length | > 0 |
| `Tortuosity` | Summed digitized root-axis lengths (`PR length + Σ LR lengths`) divided by `Total minimal Distance`; it measures axis straightness, not the straightened `Total root length` divided by `Total minimal Distance` | ratio | ≥ 1 |
| `Convex Hull Area` | Area of the convex hull enclosing all root nodes | area | > 0 |

---

## Pareto Analysis (`report_pareto_<timestamp>.csv`)

These fields use the straightened RSA and the 2D Pareto/random analysis. The optional 3D analysis is written to a separate `report_3d_<timestamp>.csv`.

### 2D Pareto Fields

The 2D analysis evaluates the trade-off between total root length (material cost) and travel distance (transport cost) on the straightened RSA.

| Field | Description | Units | Range |
|---|---|---|---|
| `Total root length (straightened)` | Sum of all edge lengths in the straightened root network | length | > 0 |
| `Travel distance (straightened)` | Sum of path lengths from the base to every root tip in the straightened network | length | > 0 |
| `alpha` | Interpolated trade-off parameter α on the 2D Pareto front | ratio | [0, 1] |
| `scaling distance to front` | Multiplicative ε-indicator measuring distance from the 2D front | ratio | ≥ 1 |
| `Total root length (random)` | Mean total root length of the analytical random baseline | length | > 0 |
| `Travel distance (random)` | Mean travel distance of the analytical random baseline | length | > 0 |
| `alpha (random)` | Interpolated α for the mean random baseline | ratio | [0, 1] |
| `scaling (random)` | ε-indicator for the mean random baseline | ratio | ≥ 1 |

#### Interpretation

- `alpha = 0`: minimum-travel (satellite-like) endpoint.
- `alpha = 1`: minimum-length (Steiner-like) endpoint.
- `scaling distance to front ≈ 1`: root lies close to the Pareto front.
- `scaling distance to front >> 1`: root is far from optimal.

The 0.3.1 report uses the **analytical randomized** null model: α is random, every connection is random among valid analytical attachments, and tips/minimum-length constraints are enforced. The default is 100 random trees; the GUI uses 500.

### Tradeoff Fields

These fields compare the actual root with the length-optimal and transport-optimal points of the 2D front.

| Field | Description | Units | Range |
|---|---|---|---|
| `Tradeoff` | `Length_ratio / Distance_ratio` | ratio | > 0 |
| `Steiner_length` | Total root length at the minimum-length front point | length | > 0 |
| `Steiner_distance` | Travel distance at the minimum-length front point | length | > 0 |
| `Satellite_length` | Total root length at the minimum-travel front point | length | > 0 |
| `Satellite_distance` | Travel distance at the minimum-travel front point | length | > 0 |
| `Length_ratio` | `actual_length / steiner_length` | ratio | > 0 |
| `Distance_ratio` | `actual_distance / satellite_distance` | ratio | > 0 |

A `Tradeoff` above 1 indicates that the actual straightened root is relatively closer to the minimum-travel (Satellite) reference than to the minimum-length (Steiner) reference; a value below 1 indicates the opposite. `Length_ratio`, `Distance_ratio`, and `Tradeoff` are dimensionless and are not rescaled when length units are configured.

### Region Classification Fields

These six fields are present only when **“-> Run region classification (adds report columns)”** is enabled. They are computed from semi-random analytical architectures sampled inside an elliptical region around the actual straightened tree.

| Field | Description | Units | Range |
|---|---|---|---|
| `mean_rounds_normalized` | Weighted mean optimization level inside the target region, normalized by the maximum number of rounds | ratio | [0, 1] |
| `ci_low_normalized` | Lower bound of the confidence interval for `mean_rounds_normalized` | ratio | [0, 1] |
| `ci_high_normalized` | Upper bound of the confidence interval for `mean_rounds_normalized` | ratio | [0, 1] |
| `local_mean_alpha` | Weighted mean trade-off direction α inside the target region | ratio | [0, 1] |
| `local_alpha_ci_low` | Lower bound of the confidence interval for `local_mean_alpha` | ratio | [0, 1] |
| `local_alpha_ci_high` | Upper bound of the confidence interval for `local_mean_alpha` | ratio | [0, 1] |

When classification is disabled, all six columns are omitted from the Pareto CSV and no sampling HTML is generated.

## 3D Pareto Analysis (`report_3d_<timestamp>.csv`, Optional)

These fields are written to the separate `report_3d_<timestamp>.csv` when **“-> Add path tortuosity to Pareto (3D, slower)”** is enabled. The 3D analysis uses the original digitized graph, not the straightened RSA. `Total root length` and `Travel distance` here are the same digitized quantities as the fields with those names in `report_rsa_*.csv`.

#### Actual Tree Metrics

| Field | Description | Units | Range |
|---|---|---|---|
| `Total root length` | Total edge length of the original graph | length | > 0 |
| `Travel distance` | Sum of path lengths from the base to every root tip in the original graph | length | > 0 |
| `Path tortuosity` | Sum of tortuosity values over all root paths | ratio | ≥ number of paths |
| `alpha_3d` | Interpolated α weight for material cost | ratio | [0, 1] |
| `beta_3d` | Interpolated β weight for transport cost | ratio | [0, 1] |
| `gamma_3d` | Coverage weight, `1 - alpha_3d - beta_3d` | ratio | [0, 1] |
| `epsilon_3d` | Multiplicative ε-indicator for the 3D front | ratio | ≥ 1 |
| `epsilon_3d_material` | `actual_length / optimal_length` | ratio | > 0 |
| `epsilon_3d_transport` | `actual_distance / optimal_distance` | ratio | > 0 |
| `epsilon_3d_coverage` | `actual_coverage / optimal_coverage` | ratio | > 0 |

#### Random Baseline (3D)

| Field | Description | Units | Range |
|---|---|---|---|
| `Total root length (random)` | Mean random-baseline material cost | length | > 0 |
| `Travel distance (random)` | Mean random-baseline transport cost | length | > 0 |
| `Path tortuosity (random)` | Mean random-baseline coverage | ratio | ≥ 1 |
| `alpha_3d (random)` | α for the mean random baseline | ratio | [0, 1] |
| `beta_3d (random)` | β for the mean random baseline | ratio | [0, 1] |
| `gamma_3d (random)` | γ for the mean random baseline | ratio | [0, 1] |
| `epsilon_3d (random)` | ε-indicator for the mean random baseline | ratio | ≥ 1 |
| `epsilon_3d_material (random)` | Material ratio for the random baseline | ratio | > 0 |
| `epsilon_3d_transport (random)` | Transport ratio for the random baseline | ratio | > 0 |
| `epsilon_3d_coverage (random)` | Coverage ratio for the random baseline | ratio | > 0 |

#### Corner Reference Costs

| Field | Description | Weight configuration |
|---|---|---|
| `Steiner_length_3d` | Root length of the Steiner corner | α = 1, β = 0, γ = 0 |
| `Steiner_distance_3d` | Travel distance of the Steiner corner | α = 1, β = 0, γ = 0 |
| `Steiner_tortuosity_3d` | Path coverage of the Steiner corner | α = 1, β = 0, γ = 0 |
| `Satellite_length_3d` | Root length of the Satellite corner | α = 0, β = 1, γ = 0 |
| `Satellite_distance_3d` | Travel distance of the Satellite corner | α = 0, β = 1, γ = 0 |
| `Satellite_tortuosity_3d` | Path coverage of the Satellite corner | α = 0, β = 1, γ = 0 |
| `Coverage_length_3d` | Root length of the Coverage corner | α = 0, β = 0, γ = 1 |
| `Coverage_distance_3d` | Travel distance of the Coverage corner | α = 0, β = 0, γ = 1 |
| `Coverage_tortuosity_3d` | Path coverage of the Coverage corner | α = 0, β = 0, γ = 1 |

---

## See Also

- [Scientific Methods](scientific-methods.md) — formulas, algorithms, and references
- [README](../README.md) — installation, GUI usage, and notebook instructions
