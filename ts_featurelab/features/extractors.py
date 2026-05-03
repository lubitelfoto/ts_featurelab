import polars as pl

from ts_featurelab.features.base import FeatureResult, SingleColumnFeatureExtractor
from ts_featurelab.features.context import FeatureContext
from ts_featurelab.features.featurespec import FeatureSpec
from ts_featurelab.features.mean import MeanFeature
from ts_featurelab.features.trend import TrendFeature
from ts_featurelab.features.wavelet import WaveletFeature


class StdFeature(SingleColumnFeatureExtractor):
    """Compute the standard deviation of a single input series.

    Params:
        resample: Optional Polars duration string used before extraction.
        agg: Aggregation applied during resampling. Defaults to ``"mean"``.
    """

    name = "std"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> FeatureResult:
        """Extract a standard-deviation scalar for the configured column.

        Args:
            context: Window-scoped feature context.
            spec: Feature specification with ``column`` and optional params.

        Returns:
            Scalar result under ``spec.alias`` or ``None`` for empty input.
        """
        col = self.require_column(spec)
        alias = spec.alias
        value = context.get_aggregated_series(
            col,
            every=spec.params.get("resample"),
            agg=spec.params.get("agg", "mean"),
        ).std()
        return FeatureResult.from_scalar(alias, None if value is None else float(value))


class MinFeature(SingleColumnFeatureExtractor):
    """Compute the minimum value of a single input series.

    Params:
        resample: Optional Polars duration string used before extraction.
        agg: Aggregation applied during resampling. Defaults to ``"mean"``.
    """

    name = "min"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> FeatureResult:
        """Extract a minimum scalar for the configured column.

        Args:
            context: Window-scoped feature context.
            spec: Feature specification with ``column`` and optional params.

        Returns:
            Scalar result under ``spec.alias`` or ``None`` for empty input.
        """
        col = self.require_column(spec)
        alias = spec.alias
        value = context.get_aggregated_series(
            col,
            every=spec.params.get("resample"),
            agg=spec.params.get("agg", "mean"),
        ).min()
        return FeatureResult.from_scalar(alias, None if value is None else float(value))


class MaxFeature(SingleColumnFeatureExtractor):
    """Compute the maximum value of a single input series.

    Params:
        resample: Optional Polars duration string used before extraction.
        agg: Aggregation applied during resampling. Defaults to ``"mean"``.
    """

    name = "max"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> FeatureResult:
        """Extract a maximum scalar for the configured column.

        Args:
            context: Window-scoped feature context.
            spec: Feature specification with ``column`` and optional params.

        Returns:
            Scalar result under ``spec.alias`` or ``None`` for empty input.
        """
        col = self.require_column(spec)
        alias = spec.alias
        value = context.get_aggregated_series(
            col,
            every=spec.params.get("resample"),
            agg=spec.params.get("agg", "mean"),
        ).max()
        return FeatureResult.from_scalar(alias, None if value is None else float(value))


class DiffFeature(SingleColumnFeatureExtractor):
    """Create a derived series with discrete differences.

    Params:
        periods: Number of periods passed to ``Series.diff``. Defaults to ``1``.
        resample: Optional Polars duration string used before differencing.
        agg: Aggregation applied during resampling. Defaults to ``"mean"``.
    """

    name = "diff"
    output_kind = "series"

    def validate_spec(self, spec: FeatureSpec) -> None:
        """Validate the diff period.

        Args:
            spec: Feature specification with optional ``periods`` param.

        Raises:
            ValueError: If ``periods`` is negative.
        """
        periods = int(spec.params.get("periods", 1))
        if periods < 0:
            raise ValueError("diff does not allow negative periods")

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> FeatureResult:
        """Extract a derived difference series.

        Args:
            context: Window-scoped feature context.
            spec: Feature specification with ``column`` and optional params.

        Returns:
            Series result named ``spec.alias``.
        """
        col = self.require_column(spec)
        alias = spec.alias
        periods = int(spec.params.get("periods", 1))
        every = spec.params.get("resample")
        agg = spec.params.get("agg", "mean")
        series = context.get_aggregated_series(col, every=every, agg=agg)
        return FeatureResult.from_series(alias, series.diff(n=periods).rename(alias))


class ValueAtLagFeature(SingleColumnFeatureExtractor):
    """Read one historical value from a single input series.

    Params:
        lag: Zero-based distance from the latest non-null value. ``1`` returns
            the value before the latest value.
        resample: Optional Polars duration string used before lag lookup.
        agg: Aggregation applied during resampling. Defaults to ``"mean"``.
    """

    name = "value_at_lag"

    def validate_spec(self, spec: FeatureSpec) -> None:
        """Validate the lag value.

        Args:
            spec: Feature specification with optional ``lag`` param.

        Raises:
            ValueError: If ``lag`` is negative.
        """
        lag = int(spec.params.get("lag", 1))
        if lag < 0:
            raise ValueError("value_at_lag does not allow negative lag")

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> FeatureResult:
        """Extract one lagged scalar value from the configured series.

        Args:
            context: Window-scoped feature context.
            spec: Feature specification with ``column`` and optional params.

        Returns:
            Scalar result under ``spec.alias`` or ``None`` when not enough
            history is available.
        """
        col = self.require_column(spec)
        alias = spec.alias
        lag = int(spec.params.get("lag", 1))
        every = spec.params.get("resample")
        agg = spec.params.get("agg", "mean")
        series = context.get_aggregated_series(col, every=every, agg=agg).drop_nulls()

        if len(series) <= lag:
            return FeatureResult.from_scalar(alias, None)

        return FeatureResult.from_scalar(alias, float(series[-(lag + 1)]))


def build_default_registry():
    """Create a registry populated with built-in feature extractors.

    Returns:
        ``FeatureRegistry`` containing mean, std, min, max, trend, diff,
        value-at-lag, and wavelet extractors.
    """
    from ts_featurelab.features.registry import FeatureRegistry

    registry = FeatureRegistry()
    for extractor in (
        MeanFeature(),
        StdFeature(),
        MinFeature(),
        MaxFeature(),
        TrendFeature(),
        DiffFeature(),
        ValueAtLagFeature(),
        WaveletFeature(),
    ):
        registry.register(extractor)
    return registry
