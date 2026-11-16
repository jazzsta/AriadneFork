## ADDED Requirements

### Requirement: Three-Report Output Documentation

The README and `docs/output-fields.md` SHALL describe the three CSV reports produced by Ariadne 0.3.1 and SHALL assign each field to the correct report.

The documentation SHALL distinguish digitized RSA measurements from straightened Pareto measurements and SHALL identify the optional 3D report.

#### Scenario: README lists the three reports
- **WHEN** a user reads the results section of `README.md`
- **THEN** it SHALL list `report_rsa_<timestamp>.csv`, `report_pareto_<timestamp>.csv`, and the optional `report_3d_<timestamp>.csv`
- **AND** it SHALL explain that the RSA report uses the digitized graph while the Pareto report uses the straightened graph

#### Scenario: Output fields are assigned to the correct report
- **WHEN** a user reads `docs/output-fields.md`
- **THEN** primary/lateral root measurements, zones, `Total minimal Distance`, `Tortuosity`, barycenter, and `Convex Hull Area` SHALL be documented under the RSA report
- **AND** `Total root length (straightened)`, `Travel distance (straightened)`, alpha, scaling distance, random-baseline, tradeoff, and region-classification fields SHALL be documented under the Pareto report
- **AND** 3D fields SHALL be documented under the separate 3D report

### Requirement: 0.3.1 Version References

The documentation and package metadata SHALL identify the current release as `0.3.1`.

#### Scenario: Version metadata is consistent
- **WHEN** the package version is read from `pyproject.toml`
- **THEN** it SHALL be `0.3.1`
- **AND** the README current-development-version label SHALL be `0.3.1`

#### Scenario: Release documentation points to v0.3.1
- **WHEN** README documentation and asset links are followed
- **THEN** they SHALL point to the `v0.3.1` tag
