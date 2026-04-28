from ts_featurelab.features.base import SingleColumnFeatureExtractor
from ts_featurelab.features.context import FeatureContext
from ts_featurelab.features.featurespec import FeatureSpec


class TrendFeature(SingleColumnFeatureExtractor):
    name = "trend"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> dict:
        col = self.require_column(spec)
        alias = spec.alias or f"{col}_trend"
        series = context.raw()[col].drop_nulls()

        if series.is_empty():
            return {alias: None}

        value = series[-1] - series[0]
        return {alias: float(value)}
