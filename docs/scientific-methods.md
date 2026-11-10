# Scientific Methods

This document describes the scientific calculations implemented in **Ariadne 0.3**. It covers the Pareto framework, the analytical tip-preserving tree generators, the random null models, the 3D path-tortuosity extension, and the region-classification procedure.

## Changes from Ariadne 0.2.0

Ariadne 0.3 changes the **2D Pareto analysis** and adds **region classification**; it does not change the trace GUI or the 3D path-tortuosity analysis.

- The 2D front now uses the continuous analytical tip-preserving generator in fully greedy mode. The 0.2.0 fixed-spacing greedy front is replaced.
- The 2D actual reference is now the straightened RSA, and dominated front points are removed.
- The 2D CSV random baseline is now the randomized analytical generator, which uses the same biological constraints as the front. The 0.2.0 random-spanning-tree model remains part of the research workflow and is still used for the 3D baseline.
- Region classification adds an optional sampling procedure and six optional CSV fields.
- Existing 2D and 3D CSV column names are retained. 2D numerical values, `alpha`, scaling distance, and tradeoff values can differ from 0.2.0 because the underlying 2D algorithms changed.

## Table of Contents

1. [Changes from Ariadne 0.2.0](#changes-from-ariadne-020)
2. [Pareto Optimality in Root Systems](#pareto-optimality-in-root-systems)
3. [Graph Straightening and Tip Preservation](#graph-straightening-and-tip-preservation)
4. [2D Pareto Analysis](#2d-pareto-analysis)
5. [Distance to the 2D Front](#distance-to-the-2d-front)
6. [Tradeoff Metric](#tradeoff-metric)
7. [3D Pareto Analysis](#3d-pareto-analysis)
8. [Random Null Models](#random-null-models)
9. [Region Classification](#region-classification)
10. [References](#references)

---

## Pareto Optimality in Root Systems

Root systems face competing costs:

- **Material cost / wiring cost:** total root length, interpreted as the metabolic investment in building the network.
- **Transport cost / conduction delay:** sum of path lengths from the root base to every root tip.
- **Path coverage / tortuosity:** sum of path-length-to-straight-line ratios over all root paths. This is used only in the 3D analysis.

A root architecture is **Pareto optimal** if no other architecture can improve one objective without worsening another. Ariadne computes a normative Pareto front and compares each observed root against that front.

The framework follows Chandrasekhar & Navlakha (2019) and Conn et al. (2017, 2019).

---

## Graph Straightening and Tip Preservation

The digitized RSA contains many intermediate points that reflect the tracing resolution rather than biological structure. Ariadne therefore works with a **straightened graph** that keeps:

- the root base (node `0`),
- all root tips (degree-1 nodes),
- all branch points (degree ≥ 3).

Intermediate degree-2 points are removed, and retained nodes are connected by straight edges weighted by Euclidean distance.

Ariadne 0.3 uses a **tip-preserving analytical generator**. Every original tip remains a terminal node; tips are never consumed as internal branch points. Constructions also enforce:

- binary branching (each new internal node has two children; the base has at most one),
- a minimum length / nude-zone constraint,
- a basal-tip fallback when the closest tip is too close to the base.

The analytical generator places each new attachment point anywhere along an existing edge, rather than restricting it to a fixed discrete grid.

---

## 2D Pareto Analysis

### Cost Function

For a tree with total root length `L`, travel distance `D`, and weight `α`:

```text
Cost(α) = α · L + (1 - α) · D
```

- `α = 0`: minimize transport cost only (minimum-travel endpoint).
- `α = 1`: minimize material cost only (minimum-length endpoint).

Implementation: `pareto_functions.py`, function `pareto_cost()`.

### Front Construction

The 2D front is computed by:

1. Evaluating `α` over `DEFAULT_ALPHAS` (from `0.00` to `1.00` in steps of `0.01`).
2. For each `α`, running the fully greedy analytical tip-preserving generator (`pareto_steiner_fast()`), which calls the semi-random analytical generator with enough optimization rounds to make every decision greedy.
3. Recording `(total root length, travel distance)` for each generated tree.
4. Removing dominated and duplicate points with `clean_pareto_front()`.

The `α = 0` endpoint is the binary, minimum-length-constrained **analytical approximation** of the satellite architecture. It is not the non-binary satellite tree. The `α = 1` endpoint is the analytical Steiner-like endpoint.

Implementation: `pareto_functions.py`, functions `pareto_front()`, `pareto_steiner_fast()`, and `clean_pareto_front()`.

---

## Distance to the 2D Front

The distance from an actual tree to the front uses the multiplicative ε-indicator:

```text
ε = min over front points r of max(actual_length / r_length,
                                  actual_distance / r_distance)
```

`ε ≥ 1`, with `ε = 1` meaning the tree lies on the front.

For reporting, Ariadne takes the two front points with the smallest ε and interpolates their α values:

```text
α_interp = (ε₂ · α₁ + ε₁ · α₂) / (ε₁ + ε₂)
```

with the closer point receiving the larger weight.

Implementation: `quantify.py`, function `distance_from_front()`.

---

## Tradeoff Metric

The tradeoff metric compares the actual root’s length-per-distance ratio with the corresponding ratio between the length-optimal and transport-optimal front endpoints:

```text
Actual_ratio  = actual_length / actual_distance
Optimal_ratio = steiner_length / satellite_distance
Tradeoff      = Actual_ratio / Optimal_ratio
```

The Steiner point is the front point with minimum total length; the Satellite point is the front point with minimum travel distance.

Implementation: `quantify.py`, function `calculate_tradeoff()`.

---

## 3D Pareto Analysis

### Cost Function

The 3D analysis adds path coverage as a third objective:

```text
Cost(α, β) = α · L + β · D - γ · C
γ = 1 - α - β
```

with `α, β, γ ≥ 0` and `α + β + γ = 1`.

Corner cases:

| Corner | Weights | Optimizes |
|---|---|---|
| Steiner | α = 1, β = 0, γ = 0 | Minimum material cost |
| Satellite | α = 0, β = 1, γ = 0 | Minimum transport cost |
| Coverage | α = 0, β = 0, γ = 1 | Maximum path coverage |

The 3D front is computed over valid `(α, β)` pairs with `α + β ≤ 1`.

Implementation: `pareto_functions.py`, functions `pareto_cost_3d_path_tortuosity()`, `pareto_front_3d_path_tortuosity()`, and `pareto_steiner_fast_3d_path_tortuosity()`.

### Path Coverage (Tortuosity)

For each path from the base to a tip:

```text
tortuosity_i = path_length_i / straight_line_distance_i
PathCoverage = Σ tortuosity_i
```

A perfectly straight path has tortuosity `1.0`.

### Distance to the 3D Front

The 3D distance uses the same multiplicative ε-indicator over three objectives:

```text
ε = min over front points r of max(actual_length / r_length,
                                  actual_distance / r_distance,
                                  actual_coverage / r_coverage)
```

Ariadne then interpolates `(α, β)` from the three nearest front points using inverse-distance weighting, clamps the result to the valid simplex, and computes `γ = 1 - α - β`.

Implementation: `quantify.py`, function `distance_from_front_3d()`.

---

## Random Null Models

Ariadne uses constrained random architectures as null models—networks generated without optimization pressure, against which the actual root and the Pareto front can be compared.

### 2D report baseline

The baseline used in the CSV report is the **randomized analytical** generator:

- α is drawn uniformly from `[0, 1]`;
- `optimization_rounds = 0`, so every connection is chosen randomly among valid analytical attachments;
- minimum-length, nude-zone, binary-branching, and tip-completeness constraints are enforced.

The reported random costs are the means over the generated trees. The UI uses 500 random trees; `analyze()` defaults to 100 unless overridden.

Implementation: `pareto_functions.py`, function `random_steiner_wrapper_analytical()`.

### Research null models

The research workflow also includes:

- the corrected random spanning-tree model, with travel distance normalized by the ratio of original critical nodes to generated tips;
- a top-down generator;
- a bottom-up generator;
- the randomized analytical generator;
- the optimization-rounds landscape, interpolating between fully greedy (`k ≥ number of tips`) and fully random (`k = 0`) analytical constructions.

These are demonstrated in `Ariadne0.3_Exploration.ipynb` and implemented in `research.py`.

---

## Region Classification

Region classification estimates how much greedy optimization and which trade-off direction best describe an actual root.

### Target region

Let `s` be the actual tree’s scaling distance to the front. The scaling excess is:

```text
excess = max(0, (s - 1) / s)
```

An ellipse is centered at the actual tree with semi-axes:

```text
radius_length  = region_factor · excess · actual_length
radius_travel  = region_factor · excess · actual_travel
```

If the actual tree lies on the front, a tiny non-zero radius is used so the normalized distance remains defined.

### Sampling

For each of `n_samples` iterations:

- draw `α ~ Uniform(0, 1)`;
- draw `rounds ~ Uniform{0, 1, ..., n_tips}`;
- generate a semi-random analytical tree;
- retain only complete trees containing every original tip;
- compute the normalized elliptical distance `d_norm` and accept the tree if `d_norm ≤ 1`.

Accepted trees receive weight:

```text
weight = 1 - d_norm
```

so trees near the actual root count more than trees near the ellipse boundary.

### Weighted statistics

For optimization rounds and α, Ariadne computes weighted means and 95% confidence intervals. The effective sample size uses Kish’s formula:

```text
n_eff = (Σ w_i)² / Σ w_i²
```

The result is a probabilistic classification:

- **mean optimization level:** how much greedy construction is needed to produce architectures with costs comparable to the real root;
- **local mean α:** the trade-off direction that best describes the root’s position in the Pareto plane.

Implementation: `quantify.py`, function `estimate_rounds_for_region()`; UI wrapper `classify_region()`.

---

## References

1. Chandrasekhar, A., & Navlakha, S. (2019). Neural arbors are Pareto optimal. *Proceedings of the Royal Society B*, 286(1902), 20182727. https://doi.org/10.1098/rspb.2018.2727
2. Conn, A., Pedmale, U. V., Chory, J., & Navlakha, S. (2017). High-resolution laser scanning reveals plant architectures that reflect universal network design principles. *Cell Systems*, 5(1), 53–62. https://doi.org/10.1016/j.cels.2017.06.017
3. Conn, A., Chandrasekhar, A., van Rongen, M., Leyser, O., Chory, J., & Navlakha, S. (2019). Network trade-offs and homeostasis in Arabidopsis shoot architectures. *PLOS Computational Biology*, 15(9), e1007325. https://doi.org/10.1371/journal.pcbi.1007325
4. Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M., & da Fonseca, V. G. (2003). Performance assessment of multiobjective optimizers: an analysis and review. *IEEE Transactions on Evolutionary Computation*, 7(2), 117–132. https://doi.org/10.1109/TEVC.2003.810758
5. Kish, L. (1965). *Survey Sampling*. Wiley.

---

## Code References

| Calculation | File | Function |
|---|---|---|
| Graph loading | `quantify.py` | `get_graph_from_json()` |
| 2D Pareto cost | `pareto_functions.py` | `pareto_cost()` |
| 2D front | `pareto_functions.py` | `pareto_front()` |
| Greedy analytical generator | `pareto_functions.py` | `pareto_steiner_fast()` |
| Semi-random analytical generator | `pareto_functions.py` | `pareto_steiner_fast_analytical_preserve_tips_semi_random()` |
| 2D distance to front | `quantify.py` | `distance_from_front()` |
| 2D tradeoff | `quantify.py` | `calculate_tradeoff()` |
| 3D Pareto cost | `pareto_functions.py` | `pareto_cost_3d_path_tortuosity()` |
| 3D distance to front | `quantify.py` | `distance_from_front_3d()` |
| 2D random baseline | `pareto_functions.py` | `random_steiner_wrapper_analytical()` |
| 3D random baseline | `pareto_functions.py` | `random_tree_3d_path_tortuosity()` |
| Region classification | `quantify.py` | `estimate_rounds_for_region()` |
