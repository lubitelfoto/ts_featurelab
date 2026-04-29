from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeatureSpec:
    name: str
    alias: str
    column: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    internal: bool = False


def _default_alias(name: str, column: str | None) -> str:
    if name == "dynamic_pressure":
        return "dynamic_pressure"
    if name == "wavelet":
        return column or "wavelet"
    if name == "value_at_lag":
        return f"{column}_lag" if column else "value_at_lag"
    if column:
        return f"{column}_{name}"
    return name


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
        if not isinstance(name, str):
            raise ValueError(f"Feature at index {idx} has non-string 'name'")

        params = item.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(f"Feature '{name}' params must be a dict")

        column = item.get("column")
        if column is not None and not isinstance(column, str):
            raise ValueError(f"Feature '{name}' column must be a string")

        alias = item.get("alias") or _default_alias(name, column)
        if not isinstance(alias, str):
            raise ValueError(f"Feature '{name}' alias must be a string")

        internal = item.get("internal", False)
        if not isinstance(internal, bool):
            raise ValueError(f"Feature '{name}' internal must be a bool")

        specs.append(
            FeatureSpec(
                name=name,
                alias=alias,
                column=column,
                params=params,
                internal=internal,
            )
        )

    return specs
