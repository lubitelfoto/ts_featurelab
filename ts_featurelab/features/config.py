from typing import Any

from ts_featurelab.features.featurespec import FeatureSpec, parse_feature_specs
from ts_featurelab.features.target import TargetSpec, parse_target_spec


def parse_feature_config(config: dict[str, Any]) -> tuple[str, list[FeatureSpec]]:
    """Parse top-level feature configuration.

    Args:
        config: Configuration dictionary containing optional ``time_col`` and
            required ``features`` entries.

    Returns:
        Tuple with the time column name and parsed feature specifications.
    """
    time_col = config.get("time_col", "date")
    specs = parse_feature_specs(config)
    return time_col, specs


def parse_supervised_config(
    config: dict[str, Any],
) -> tuple[str, list[FeatureSpec], TargetSpec | None]:
    """Parse feature and optional target configuration.

    Args:
        config: Configuration dictionary containing feature entries and an
            optional ``target`` entry.

    Returns:
        Tuple with the time column name, parsed feature specs, and an optional
        target spec.
    """
    time_col, specs = parse_feature_config(config)
    target_spec = parse_target_spec(config)
    return time_col, specs, target_spec
