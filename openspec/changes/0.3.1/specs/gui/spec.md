## ADDED Requirements

### Requirement: Analysis Report Naming and Completion Summary

The Analyzer GUI SHALL write the separated CSV reports and SHALL report their names in the completion dialog.

The completion summary SHALL always list the RSA and Pareto reports and SHALL list the 3D report only when 3D analysis was enabled.

#### Scenario: Completion dialog lists the 2D reports
- **WHEN** an analysis run completes without 3D analysis
- **THEN** the completion dialog SHALL list `report_rsa_<timestamp>.csv`
- **AND** it SHALL list `report_pareto_<timestamp>.csv`
- **AND** it SHALL NOT list a 3D CSV report

#### Scenario: Completion dialog lists the 3D report when enabled
- **WHEN** an analysis run completes with 3D analysis enabled
- **THEN** the completion dialog SHALL also list `report_3d_<timestamp>.csv`

#### Scenario: Region classification affects only the Pareto report
- **WHEN** region classification is enabled
- **THEN** the six classification columns SHALL be written to `report_pareto_<timestamp>.csv`
- **AND** they SHALL NOT be written to `report_rsa_<timestamp>.csv`
