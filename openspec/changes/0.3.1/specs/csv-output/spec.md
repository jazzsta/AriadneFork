## ADDED Requirements

### Requirement: Separate RSA, Pareto, and 3D CSV Reports

The analysis workflow SHALL write digitized-RSA measurements and straightened-RSA Pareto analysis to separate CSV reports, and SHALL keep the optional 3D analysis in its own report.

The reports SHALL be named `report_rsa_<timestamp>.csv`, `report_pareto_<timestamp>.csv`, and `report_3d_<timestamp>.csv`.

#### Scenario: RSA report contains digitized measurements
- **WHEN** an analysis run completes
- **THEN** `report_rsa_<timestamp>.csv` SHALL exist
- **AND** it SHALL contain digitized `Total root length`, `Travel distance`, primary/lateral root metrics, zone metrics, `Total minimal Distance`, `Tortuosity`, barycenter, and `Convex Hull Area`
- **AND** it SHALL NOT contain Pareto alpha, random-baseline, tradeoff, or region-classification fields

#### Scenario: Pareto report contains straightened analysis
- **WHEN** an analysis run completes
- **THEN** `report_pareto_<timestamp>.csv` SHALL exist
- **AND** it SHALL contain `Total root length (straightened)`, `Travel distance (straightened)`, alpha, scaling distance, random-baseline fields, tradeoff fields, and optional region-classification fields
- **AND** it SHALL NOT contain the digitized RSA measurement fields

#### Scenario: 3D report is separate and optional
- **WHEN** 3D path-tortuosity analysis is disabled
- **THEN** no `report_3d_<timestamp>.csv` SHALL be written
- **WHEN** 3D path-tortuosity analysis is enabled
- **THEN** `report_3d_<timestamp>.csv` SHALL be written
- **AND** it SHALL keep the 3D field names `Total root length`, `Travel distance`, `Path tortuosity`, and the `_3d` Pareto fields

### Requirement: Digitized and Straightened Cost Field Naming

The digitized RSA report SHALL use `Total root length` and `Travel distance` for the original digitized graph.

The Pareto report SHALL use `Total root length (straightened)` and `Travel distance (straightened)` for the straightened graph.

#### Scenario: Digitized costs use unsuffixed names
- **WHEN** the digitized RSA report is written
- **THEN** `Total root length` SHALL equal the sum of all digitized edge lengths
- **AND** `Travel distance` SHALL equal the sum of base-to-tip path lengths along the digitized graph

#### Scenario: Straightened costs use suffixed names
- **WHEN** the Pareto report is written
- **THEN** `Total root length (straightened)` SHALL equal the sum of all straightened edge lengths
- **AND** `Travel distance (straightened)` SHALL equal the sum of base-to-tip path lengths along the straightened graph

#### Scenario: 3D costs match digitized costs
- **WHEN** the 3D report is written
- **THEN** its `Total root length` and `Travel distance` SHALL equal the corresponding digitized values in `report_rsa_<timestamp>.csv`

### Requirement: Tradeoff CSV Field Naming

The Pareto report SHALL expose the tradeoff ratios as `Length_ratio` and `Distance_ratio`, with `Tradeoff = Length_ratio / Distance_ratio`.

#### Scenario: New tradeoff names are emitted
- **WHEN** the Pareto report is written
- **THEN** it SHALL contain `Length_ratio`, `Distance_ratio`, `Tradeoff`, `Steiner_length`, `Steiner_distance`, `Satellite_length`, and `Satellite_distance`
- **AND** it SHALL NOT contain `Actual_ratio` or `Optimal_ratio`

#### Scenario: Tradeoff ratios are not scaled
- **WHEN** a length scale factor is configured
- **THEN** `Length_ratio`, `Distance_ratio`, and `Tradeoff` SHALL retain their original values

### Requirement: Dimension-Aware CSV Scaling

The CSV scaling transformation SHALL apply the dimension appropriate to each field.

Lengths SHALL be multiplied by the length scale factor, areas by its square, and inverse-length densities by its inverse. Dimensionless fields SHALL remain unchanged.

#### Scenario: Length and area fields use different exponents
- **GIVEN** a length scale factor `s`
- **WHEN** scaling is applied
- **THEN** length fields SHALL be multiplied by `s`
- **AND** `Convex Hull Area` SHALL be multiplied by `s²`

#### Scenario: Inverse-length densities use the reciprocal factor
- **GIVEN** a length scale factor `s`
- **WHEN** scaling is applied
- **THEN** `LR density` and `Branched Zone density` SHALL be divided by `s`

#### Scenario: 3D corner tortuosity is dimensionless
- **WHEN** `Steiner_tortuosity_3d`, `Satellite_tortuosity_3d`, or `Coverage_tortuosity_3d` is present
- **THEN** the value SHALL remain unchanged during scaling
