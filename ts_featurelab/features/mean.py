from ts_featurelab.features.base import FeatureResult, SingleColumnFeatureExtractor
from ts_featurelab.features.featurespec import FeatureSpec
from ts_featurelab.features.context import FeatureContext


class MeanFeature(SingleColumnFeatureExtractor):
    """Compute the arithmetic mean of a single input series.

    Params:
        resample: Optional Polars duration string used before extraction.
        agg: Aggregation applied during resampling. Defaults to ``"mean"``.
    """

    name = "mean"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> FeatureResult:
        """Extract a mean scalar for the configured column.

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
        )
        value = series.mean()
        return FeatureResult.from_scalar(
            alias,
            None if value is None else float(value),
        )
