# Output Fields Reference

This document describes the CSV fields produced by **Ariadne 0.3**.

The two-dimensional report contains one row per analyzed root. A separate 3D report is written when 3D analysis is enabled. Region-classification fields are optional and are omitted when classification is disabled.

## Changes from Ariadne 0.2.0

Ariadne 0.3 retains every 0.2.0 CSV column and adds six optional region-classification columns.

- 2D Pareto values, `alpha`, scaling distance, and tradeoff values can differ from 0.2.0 because the 2D front now uses the analytical tip-preserving generator and compares against the straightened RSA.
- `Total root length (random)`, `Travel distance (random)`, `alpha (random)`, and `scaling (random)` now describe the randomized analytical baseline rather than the 0.2.0 random-spanning-tree model.
- The 3D report and its spanning-tree random baseline are unchanged from 0.2.0.
- When region classification is disabled, the six classification columns and the sampling report are omitted.

## Table of Contents

1. [Changes from Ariadne 0.2.0](#changes-from-ariadne-020)
2. [Units](#units)
3. [Basic Root Measurements](#basic-root-measurements)
4. [Primary Root Metrics](#primary-root-metrics)
5. [Lateral Root Metrics](#lateral-root-metrics)
6. [Zone Measurements](#zone-measurements)
7. [2D Pareto Fields](#2d-pareto-fields)
8. [Tradeoff Fields](#tradeoff-fields)
9. [Region Classification Fields](#region-classification-fields)
10. [3D Pareto Fields](#3d-pareto-fields)
11. [Other Metrics](#other-metrics)

---

## Units

Length measurements are reported in the units selected during analysis:

- if scaling was configured (for example `mm`), lengths use those units;
- otherwise, lengths are in pixels.

Ratios, angles, counts, and densities are dimensionless or use the indicated units.

---

## Basic Root Measurements

| Field | Description | Units | Range |
|---|---|---|---|
| `Total root length` | Sum of all edge lengths in the straightened root network | length | > 0 |
| `Travel distance` | Sum of path lengths from the base to every root tip | length | > 0 |
| `Total minimal Distance` | Sum of straight-line distances from the base to every tip | length | > 0 |
| `Tortuosity` | Total root length divided by total minimal distance | ratio | ≥ 1 |
| `filename` | Input file name without the `.json` extension | text | — |

---

## Primary Root Metrics

| Field | Description | Units | Range |
|---|---|---|---|
| `PR length` | Length of the primary root | length | > 0 |
| `PR_minimal_length` | Straight-line distance from the hypocotyl to the primary-root tip | length | > 0 |

---

## Lateral Root Metrics

| Field | Description | Units | Range |
|---|---|---|---|
| `LR count` | Number of lateral roots | count | ≥ 0 |
| `LR density` | Lateral-root count divided by primary-root length | 1/length | ≥ 0 |
| `Mean LR lengths` | Mean lateral-root length | length | > 0 |
| `Median LR lengths` | Median lateral-root length | length | > 0 |
| `Mean LR minimal lengths` | Mean straight-line distance from each LR tip to its insertion point | length | ≥ 0 |
| `Median LR minimal lengths` | Median straight-line LR distance | length | ≥ 0 |
| `sum LR minimal lengths` | Sum of straight-line LR distances | length | ≥ 0 |
| `Mean LR angles` | Mean lateral-root set-point angle | degrees | 0–180 |
| `Median LR angles` | Median lateral-root set-point angle | degrees | 0–180 |
| `LR lengths` | Array of individual lateral-root lengths | length[] | > 0 |
| `LR angles` | Array of individual lateral-root angles | degrees[] | 0–180 |
| `LR minimal lengths` | Array of individual straight-line LR distances | length[] | ≥ 0 |

---

## Zone Measurements

| Field | Description | Units | Range |
|---|---|---|---|
| `Basal Zone length` | Length from the hypocotyl to the first lateral-root insertion | length | ≥ 0 |
| `Branched Zone length` | Length between the first and last lateral-root insertions | length | ≥ 0 |
| `Apical Zone length` | Length from the last lateral-root insertion to the primary-root tip | length | ≥ 0 |
| `Branched Zone density` | LR count divided by branched-zone length | 1/length | ≥ 0 |

---

## 2D Pareto Fields

The 2D analysis evaluates the trade-off between total root length (material cost) and travel distance (transport cost).

| Field | Description | Units | Range |
|---|---|---|---|
| `alpha` | Interpolated trade-off parameter α on the 2D Pareto front | ratio | [0, 1] |
| `scaling distance to front` | Multiplicative ε-indicator measuring distance from the 2D front | ratio | ≥ 1 |
| `Total root length (random)` | Mean total root length of the analytical random baseline | length | > 0 |
| `Travel distance (random)` | Mean travel distance of the analytical random baseline | length | > 0 |
| `alpha (random)` | Interpolated α for the mean random baseline | ratio | [0, 1] |
| `scaling (random)` | ε-indicator for the mean random baseline | ratio | ≥ 1 |

### Interpretation

- `alpha = 0`: minimum-travel (satellite-like) endpoint.
- `alpha = 1`: minimum-length (Steiner-like) endpoint.
- `scaling distance to front ≈ 1`: root lies close to the Pareto front.
- `scaling distance to front >> 1`: root is far from optimal.

The 0.3 report uses the **analytical randomized** null model: α is random, every connection is random among valid analytical attachments, and tips/minimum-length constraints are enforced. The default is 100 random trees; the GUI uses 500.

---

## Tradeoff Fields

These fields compare the actual root with the length-optimal and transport-optimal points of the 2D front.

| Field | Description | Units | Range |
|---|---|---|---|
| `Tradeoff` | `Actual_ratio / Optimal_ratio` | ratio | > 0 |
| `Steiner_length` | Total root length at the minimum-length front point | length | > 0 |
| `Steiner_distance` | Travel distance at the minimum-length front point | length | > 0 |
| `Satellite_length` | Total root length at the minimum-travel front point | length | > 0 |
| `Satellite_distance` | Travel distance at the minimum-travel front point | length | > 0 |
| `Actual_ratio` | `actual_length / actual_distance` | ratio | > 0 |
| `Optimal_ratio` | `steiner_length / satellite_distance` | ratio | > 0 |

---

## Region Classification Fields

These six fields are present only when **“-> Run region classification (adds report columns)”** is enabled. They are computed from semi-random analytical architectures sampled inside an elliptical region around the actual straightened tree.

| Field | Description | Units | Range |
|---|---|---|---|
| `mean_rounds_normalized` | Weighted mean optimization level inside the target region, normalized by the maximum number of rounds | ratio | [0, 1] |
| `ci_low_normalized` | Lower bound of the confidence interval for `mean_rounds_normalized` | ratio | [0, 1] |
| `ci_high_normalized` | Upper bound of the confidence interval for `mean_rounds_normalized` | ratio | [0, 1] |
| `local_mean_alpha` | Weighted mean trade-off direction α inside the target region | ratio | [0, 1] |
| `local_alpha_ci_low` | Lower bound of the confidence interval for `local_mean_alpha` | ratio | [0, 1] |
| `local_alpha_ci_high` | Upper bound of the confidence interval for `local_mean_alpha` | ratio | [0, 1] |

When classification is disabled, all six columns are omitted. This restores the Ariadne 0.2.0 2D column set.

---

## 3D Pareto Fields

These fields are written to `report_3d_*.csv` when **“-> Add path tortuosity to Pareto (3D, slower)”** is enabled.

### Actual Tree Metrics

| Field | Description | Units | Range |
|---|---|---|---|
| `Path tortuosity` | Sum of tortuosity values over all root paths | ratio | ≥ number of paths |
| `alpha_3d` | Interpolated α weight for material cost | ratio | [0, 1] |
| `beta_3d` | Interpolated β weight for transport cost | ratio | [0, 1] |
| `gamma_3d` | Coverage weight, `1 - alpha_3d - beta_3d` | ratio | [0, 1] |
| `epsilon_3d` | Multiplicative ε-indicator for the 3D front | ratio | ≥ 1 |
| `epsilon_3d_material` | `actual_length / optimal_length` | ratio | > 0 |
| `epsilon_3d_transport` | `actual_distance / optimal_distance` | ratio | > 0 |
| `epsilon_3d_coverage` | `actual_coverage / optimal_coverage` | ratio | > 0 |

### Random Baseline (3D)

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

### Corner Reference Costs

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

## Other Metrics

| Field | Description | Units | Range |
|---|---|---|---|
| `Barycenter x displacement` | Horizontal displacement from the hypocotyl to the convex-hull barycenter | length | any |
| `Barycenter y displacement` | Vertical displacement from the hypocotyl to the convex-hull barycenter | length | any |
| `Convex Hull Area` | Area of the convex hull enclosing all root nodes | area | > 0 |

---

## See Also

- [Scientific Methods](scientific-methods.md) — formulas, algorithms, and references
- [README](../README.md) — installation, GUI usage, and notebook instructions
