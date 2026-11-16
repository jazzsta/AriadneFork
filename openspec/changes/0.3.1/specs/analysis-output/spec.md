## ADDED Requirements

### Requirement: Digitized and Straightened Cost Separation

`quantify.analyze()` SHALL compute digitized total root length and travel distance from the original graph and SHALL retain them separately from the straightened Pareto costs.

The digitized values SHALL be exposed through `Total root length (digitized)` and `Travel distance (digitized)` for report splitting, while the existing Pareto keys SHALL remain the straightened costs.

#### Scenario: Digitized costs are computed from the original graph
- **WHEN** `quantify.analyze()` is called
- **THEN** `Total root length (digitized)` SHALL equal `graph_costs()` on the original graph
- **AND** `Travel distance (digitized)` SHALL equal the sum of base-to-tip path lengths on the original graph

#### Scenario: Report split maps digitized and straightened keys
- **WHEN** `split_analysis_results()` is called
- **THEN** the RSA dictionary SHALL contain `Total root length` and `Travel distance` from the digitized values
- **AND** the Pareto dictionary SHALL contain `Total root length (straightened)` and `Travel distance (straightened)` from the Pareto values

### Requirement: Primary Root Tip Identification

`find_lowermost_node_of_primary_root()` SHALL identify primary-root nodes by `LR_index is None` and SHALL support both directed and undirected graphs.

The function SHALL select the terminal primary-root tip rather than the deepest node belonging to any root.

#### Scenario: Directed graph
- **GIVEN** a directed root graph with one primary-root tip
- **WHEN** `find_lowermost_node_of_primary_root()` is called
- **THEN** it SHALL return the position of the `LR_index is None` tip

#### Scenario: Undirected graph
- **GIVEN** an undirected root graph with one primary-root tip
- **WHEN** `find_lowermost_node_of_primary_root()` is called
- **THEN** it SHALL return the position of the `LR_index is None` tip
- **AND** it SHALL NOT raise an error from `nx.descendants()`

### Requirement: Corrected Zone Boundary Assignment

`calc_zones()` SHALL assign each primary-root segment to the zone in which it ends.

Segments ending at or before the first lateral-root insertion SHALL be basal, segments ending after the first insertion and at or before the last insertion SHALL be branched, and remaining segments SHALL be apical.

#### Scenario: Segment into first insertion is basal
- **GIVEN** a primary root with a lateral insertion at node `p2`
- **WHEN** `calc_zones()` is called
- **THEN** the segment ending at `p2` SHALL contribute to `basal_zone_length`

#### Scenario: Segment into last insertion is branched
- **GIVEN** a primary root with lateral insertions at `p2` and `p4`
- **WHEN** `calc_zones()` is called
- **THEN** the segment ending at `p4` SHALL contribute to `branched_zone_length`
- **AND** segments after `p4` SHALL contribute to `apical_zone_length`

#### Scenario: Zone lengths partition the primary root
- **WHEN** `calc_zones()` returns non-zero zones
- **THEN** `basal_zone_length + branched_zone_length + apical_zone_length` SHALL equal the primary-root length

### Requirement: Tradeoff Ratio Formulation

`calculate_tradeoff()` SHALL compute `Length_ratio = actual_length / steiner_length` and `Distance_ratio = actual_distance / satellite_distance`, and `Tradeoff = Length_ratio / Distance_ratio`.

The function SHALL return `None` for a ratio when its denominator is zero and SHALL return `None` for `Tradeoff` when either ratio is invalid or zero.

#### Scenario: Valid ratios
- **GIVEN** a front and an actual tree with positive costs
- **WHEN** `calculate_tradeoff()` is called
- **THEN** it SHALL return `Length_ratio`, `Distance_ratio`, and `Tradeoff` using the ratio formulation

#### Scenario: Zero Steiner length
- **GIVEN** a front whose minimum-length point has zero length
- **WHEN** `calculate_tradeoff()` is called
- **THEN** `Length_ratio` SHALL be `None`
- **AND** `Distance_ratio` SHALL still be computed when possible
- **AND** `Tradeoff` SHALL be `None`

### Requirement: Weak Pareto Dominance Removal

`clean_pareto_front()` SHALL remove points that are weakly dominated in both wiring and travel distance, including equal-wiring or equal-travel duplicates.

#### Scenario: Equal-wiring point is removed
- **GIVEN** two front points with equal wiring and different travel distances
- **WHEN** `clean_pareto_front()` is called
- **THEN** the point with higher travel distance SHALL be removed
- **AND** the point with lower travel distance SHALL remain

#### Scenario: Duplicate points are removed
- **GIVEN** two front points with identical wiring and travel distance
- **WHEN** `clean_pareto_front()` is called
- **THEN** only one of them SHALL remain
