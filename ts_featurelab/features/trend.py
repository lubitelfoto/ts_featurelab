from ts_featurelab.features.base import FeatureResult, SingleColumnFeatureExtractor
from ts_featurelab.features.context import FeatureContext
from ts_featurelab.features.featurespec import FeatureSpec


class TrendFeature(SingleColumnFeatureExtractor):
    """Compute the difference between the latest and earliest non-null value.

    Params:
        resample: Optional Polars duration string used before extraction.
        agg: Aggregation applied during resampling. Defaults to ``"mean"``.
    """

    name = "trend"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> FeatureResult:
        """Extract a trend scalar for the configured column.

        Args:
            context: Window-scoped feature context.
            spec: Feature specification with ``column`` and optional params.

        Returns:
            Scalar result under ``spec.alias`` or ``None`` for empty input.
        """
        col = self.require_column(spec)
        alias = spec.alias
        series = context.get_aggregated_series(
            col,
            every=spec.params.get("resample"),
            agg=spec.params.get("agg", "mean"),
        ).drop_nulls()

        if series.is_empty():
            return FeatureResult.from_scalar(alias, None)

        value = series[-1] - series[0]
        return FeatureResult.from_scalar(alias, float(value))
