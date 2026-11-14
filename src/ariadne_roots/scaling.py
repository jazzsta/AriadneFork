"""Scaling utilities for converting pixel measurements to real-world units."""

AREA_FIELDS = {"Convex Hull Area"}
INVERSE_LENGTH_FIELDS = {"LR density", "Branched Zone density"}


def apply_scaling_transformation(results, scale_factor, excluded_fields=None):
    """Apply dimension-aware scaling to a results dictionary.

    Lengths are multiplied by ``scale_factor``, areas by its square, and
    inverse-length measures (densities) are divided by it. Dimensionless
    metrics and non-numeric values are preserved.

    Uses substring matching for excluded_fields to catch related metrics:
    - "angles" excludes "Mean LR angles", "Median LR angles", "LR angles"

    Args:
        results: Dictionary of analysis results
        scale_factor: Length scale factor (e.g., 2.5 for 1 px = 2.5 mm)
        excluded_fields: Set of substring patterns to exclude from scaling.
                        Uses substring matching (not exact matching).
                        Defaults to the following dimensionless patterns:
                        - "alpha" (shape parameter)
                        - "Mean LR angles", "Median LR angles", "LR angles" (angles)
                        - "LR count" (count)
                        - "scaling distance to front", "scaling (random)" (normalized)
                        - "Tortuosity" (ratio)
                        - "Tradeoff", "Length_ratio", "Distance_ratio"
                          (dimensionless ratios; legacy "Actual_ratio" and
                          "Optimal_ratio" are also excluded)
                        - "tortuosity_3d" (3D corner coverage ratios)
                        - "Path tortuosity", "alpha_beta", "beta", "gamma",
                          "epsilon" (3D Pareto ratios and parameters)

    Returns:
        Dictionary with scaled numeric values, unchanged excluded/non-numeric values

    Examples:
        >>> results = {
        ...     "Total root length": 100,
        ...     "LR density": 0.5,
        ...     "Convex Hull Area": 30,
        ...     "Mean LR angles": 45.0,
        ...     "filename": "test.json"
        ... }
        >>> scaled = apply_scaling_transformation(results, 2.0)
        >>> scaled["Total root length"]  # Length
        200.0
        >>> scaled["LR density"]  # Inverse length
        0.25
        >>> scaled["Convex Hull Area"]  # Area
        120.0
        >>> scaled["Mean LR angles"]  # Excluded (contains "angles" pattern)
        45.0
        >>> scaled["filename"]  # Non-numeric preserved
        'test.json'

        Custom exclusion patterns (substring matching):
        >>> excluded = {"ratio", "count"}
        >>> results = {"Length ratio": 1.5, "Root count": 10, "Length": 100}
        >>> scaled = apply_scaling_transformation(results, 2.0, excluded)
        >>> scaled["Length ratio"]  # Excluded (contains "ratio")
        1.5
        >>> scaled["Root count"]  # Excluded (contains "count")
        10
        >>> scaled["Length"]  # Scaled
        200.0
    """
    if excluded_fields is None:
        excluded_fields = {
            "alpha",
            "Mean LR angles",
            "Median LR angles",
            "LR angles",  # Array of angles (dimensionless) - explicit exclusion
            "LR count",
            "scaling distance to front",
            "Tortuosity",
            "scaling (random)",
            "normalized",  # mean_rounds_normalized, ci_low/high_normalized
            # Tradeoff metrics (dimensionless ratios)
            "Tradeoff",
            "Length_ratio",
            "Distance_ratio",
            "Actual_ratio",  # legacy name
            "Optimal_ratio",  # legacy name
            # 3D Pareto analysis fields (ratios and parameter tuples)
            "Path tortuosity",  # Ratio: sum of (travel_dist / straight_dist)
            "tortuosity_3d",  # Corner coverage ratios
            "alpha_beta",  # Parameter tuple, not a length
            # 3D weight and epsilon fields (dimensionless ratios)
            "beta",  # Catches beta_3d, beta_3d (random)
            "gamma",  # Catches gamma_3d, gamma_3d (random)
            "epsilon",  # Catches epsilon_3d, epsilon_3d_*, and (random) variants
        }

    scaled_results = {}
    for key, value in results.items():
        if any(excl in key for excl in excluded_fields):
            # Do not scale excluded fields
            scaled_results[key] = value
        elif key in INVERSE_LENGTH_FIELDS:
            # Densities have units 1/length, so they scale by 1/s.
            try:
                scaled_results[key] = float(value) / scale_factor
            except (ValueError, TypeError, ZeroDivisionError):
                scaled_results[key] = value
        elif key in AREA_FIELDS:
            # Areas have units length^2, so they scale by s^2.
            try:
                scaled_results[key] = float(value) * scale_factor**2
            except (ValueError, TypeError):
                scaled_results[key] = value
        elif key in ["LR lengths", "LR minimal lengths"]:
            # Special handling for array fields - scale each element
            # Currently only LR lengths and LR minimal lengths are array-valued metrics
            # If new array fields are added to quantify.py, update this list
            try:
                scaled_results[key] = [float(v) * scale_factor for v in value]
            except (ValueError, TypeError):
                scaled_results[key] = value
        else:
            # Scale all other numeric fields as lengths
            try:
                scaled_results[key] = float(value) * scale_factor
            except (ValueError, TypeError):
                # Leave non-numeric values as-is
                scaled_results[key] = value

    return scaled_results
