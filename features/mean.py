from features.base import SingleColumnFeatureExtractor
from features.featurespec import FeatureSpec
from features.context import FeatureContext


class MeanFeature(SingleColumnFeatureExtractor):
    name = "mean"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> dict:
        col = self.require_column(spec)
        alias = spec.alias or f"{col}_mean"
        series = context.raw()[col]
        value = series.mean()
        return {alias: None if value is None else float(value)}
