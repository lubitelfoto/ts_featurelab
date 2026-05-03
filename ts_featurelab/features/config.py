from typing import Any

from ts_featurelab.features.featurespec import FeatureSpec, parse_feature_specs


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
