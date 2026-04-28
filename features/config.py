from typing import Any

from features.featurespec import FeatureSpec, parse_feature_specs


def parse_feature_config(config: dict[str, Any]) -> tuple[str, list[FeatureSpec]]:
    time_col = config.get("time_col", "date")
    specs = parse_feature_specs(config)
    return time_col, specs
