from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeatureSpec:
    name: str
    column: str | None = None
    alias: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


def parse_feature_specs(config: dict[str, Any]) -> list[FeatureSpec]:
    if "features" not in config:
        raise ValueError("Feature config must contain 'features'")

    raw_features = config["features"]

    if not isinstance(raw_features, list):
        raise ValueError("'features' must be a list")

    specs: list[FeatureSpec] = []
    for idx, item in enumerate(raw_features):
        if not isinstance(item, dict):
            raise ValueError(f"Feature at index {idx} must be a dict")

        name = item.get("name")
        if not name:
            raise ValueError(f"Feature at index {idx} is missing 'name'")

        params = item.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(f"Feature '{name}' params must be a dict")

        specs.append(
            FeatureSpec(
                name=name,
                column=item.get("column"),
                alias=item.get("alias"),
                params=params,
            )
        )

    return specs
