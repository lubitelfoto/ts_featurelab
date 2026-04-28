from abc import ABC, abstractmethod

from ts_featurelab.features.featurespec import FeatureSpec


class FeatureExtractor(ABC):
    name: str

    @abstractmethod
    def extract(self, context, spec: FeatureSpec) -> dict[str, object]:
        raise NotImplementedError


class SingleColumnFeatureExtractor(FeatureExtractor):
    def require_column(self, spec: FeatureSpec) -> str:
        if not spec.column:
            raise ValueError(f"Feature '{spec.name}' requires 'column'")
        return spec.column
