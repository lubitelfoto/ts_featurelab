from abc import ABC, abstractmethod
from typing import Literal

import polars as pl

from ts_featurelab.features.featurespec import FeatureSpec

FeatureOutputKind = Literal["series", "scalar"]


class FeatureExtractor(ABC):
    name: str
    output_kind: FeatureOutputKind = "scalar"

    @abstractmethod
    def extract(self, context, spec: FeatureSpec) -> pl.Series | dict[str, object]:
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
        result: object,
    ) -> list[str]:
        if self.get_output_kind(spec) == "series":
            return [spec.alias]
        if isinstance(result, dict):
            return list(result.keys())
        return [spec.alias]


class SingleColumnFeatureExtractor(FeatureExtractor):
    def require_column(self, spec: FeatureSpec) -> str:
        if not spec.column:
            raise ValueError(f"Feature '{spec.name}' requires 'column'")
        return spec.column
