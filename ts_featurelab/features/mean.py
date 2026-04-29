from ts_featurelab.features.base import FeatureResult, SingleColumnFeatureExtractor
from ts_featurelab.features.featurespec import FeatureSpec
from ts_featurelab.features.context import FeatureContext


class MeanFeature(SingleColumnFeatureExtractor):
    name = "mean"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> FeatureResult:
        col = self.require_column(spec)
        alias = spec.alias
        series = context.get_aggregated_series(
            col,
            every=spec.params.get("resample"),
            agg=spec.params.get("agg", "mean"),
        )
        value = series.mean()
        return FeatureResult.from_scalar(
            alias,
            None if value is None else float(value),
        )
