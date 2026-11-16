## ADDED Requirements

### Requirement: CSV Report Split Test Coverage

The test suite SHALL verify that digitized RSA measurements and straightened Pareto measurements are assigned to the correct report dictionaries.

#### Scenario: Digitized and straightened costs are separated
- **WHEN** `split_analysis_results()` is tested
- **THEN** the RSA dictionary SHALL contain `Total root length` and `Travel distance` from the digitized values
- **AND** the Pareto dictionary SHALL contain `Total root length (straightened)` and `Travel distance (straightened)`
- **AND** neither report SHALL contain the other report's cost keys

#### Scenario: Analysis exposes digitized costs
- **WHEN** `quantify.analyze()` is tested on a real graph
- **THEN** the results SHALL contain `Total root length (digitized)` and `Travel distance (digitized)`

### Requirement: Dimension-Aware Scaling Test Coverage

The test suite SHALL verify length, area, inverse-length, and dimensionless scaling separately.

#### Scenario: Area scales by the squared factor
- **WHEN** `Convex Hull Area` is scaled by factor `s`
- **THEN** the result SHALL be the original value multiplied by `s²`

#### Scenario: Densities scale by the inverse factor
- **WHEN** `LR density` or `Branched Zone density` is scaled by factor `s`
- **THEN** the result SHALL be the original value divided by `s`

#### Scenario: 3D corner tortuosity is preserved
- **WHEN** `Steiner_tortuosity_3d`, `Satellite_tortuosity_3d`, or `Coverage_tortuosity_3d` is scaled
- **THEN** the value SHALL remain unchanged

### Requirement: Corrected Measurement Test Coverage

The test suite SHALL cover the corrected zone boundaries, primary-root tip selection, tradeoff fields, and weak-dominance cleaning.

#### Scenario: Zone boundary values are exact
- **GIVEN** a primary root with a known insertion point
- **WHEN** `calc_zones()` is tested
- **THEN** basal, branched, and apical lengths SHALL be asserted exactly rather than only checked for non-negativity

#### Scenario: Tradeoff tests use current field names
- **WHEN** `calculate_tradeoff()` is tested
- **THEN** the tests SHALL use `Length_ratio` and `Distance_ratio`
- **AND** they SHALL NOT expect `Actual_ratio` or `Optimal_ratio`
