from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from ts_featurelab.features.featurespec import FeatureSpec

FeatureOutputKind = Literal["series", "scalar"]


@dataclass
class FeatureResult:
    scalars: dict[str, object] = field(default_factory=dict)
    series: dict[str, pl.Series] = field(default_factory=dict)

    @classmethod
    def from_scalar(cls, alias: str, value: object) -> "FeatureResult":
        return cls(scalars={alias: value})

    @classmethod
    def from_scalars(cls, values: dict[str, object]) -> "FeatureResult":
        return cls(scalars=dict(values))

    @classmethod
    def from_series(cls, alias: str, values: pl.Series) -> "FeatureResult":
        normalized = values if values.name == alias else pl.Series(alias, values)
        return cls(series={alias: normalized})


class FeatureExtractor(ABC):
    name: str
    output_kind: FeatureOutputKind = "scalar"

    @abstractmethod
    def extract(self, context, spec: FeatureSpec) -> FeatureResult:
        raise NotImplementedError

    def get_dependencies(self, spec: FeatureSpec) -> set[str]:
        return {spec.column} if spec.column else set()

    def get_output_kind(self, spec: FeatureSpec) -> FeatureOutputKind:
        return self.output_kind

    def validate_spec(self, spec: FeatureSpec) -> None:
        return None

    def get_output_columns(
        self,
        spec: FeatureSpec,
        result: FeatureResult,
    ) -> list[str]:
        if self.get_output_kind(spec) == "series":
            return list(result.series.keys()) or [spec.alias]
        return list(result.scalars.keys()) or [spec.alias]


class SingleColumnFeatureExtractor(FeatureExtractor):
    def require_column(self, spec: FeatureSpec) -> str:
        if not spec.column:
            raise ValueError(f"Feature '{spec.name}' requires 'column'")
        return spec.column
