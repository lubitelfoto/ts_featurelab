from features.base import SingleColumnFeatureExtractor
from features.context import FeatureContext
from features.featurespec import FeatureSpec
from features.mean import MeanFeature
from features.trend import TrendFeature
from features.wavelet import WaveletFeature


class StdFeature(SingleColumnFeatureExtractor):
    name = "std"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> dict:
        col = self.require_column(spec)
        alias = spec.alias or f"{col}_std"
        value = context.raw()[col].std()
        return {alias: None if value is None else float(value)}


class MinFeature(SingleColumnFeatureExtractor):
    name = "min"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> dict:
        col = self.require_column(spec)
        alias = spec.alias or f"{col}_min"
        value = context.raw()[col].min()
        return {alias: None if value is None else float(value)}


class MaxFeature(SingleColumnFeatureExtractor):
    name = "max"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> dict:
        col = self.require_column(spec)
        alias = spec.alias or f"{col}_max"
        value = context.raw()[col].max()
        return {alias: None if value is None else float(value)}


class DiffFeature(SingleColumnFeatureExtractor):
    name = "diff"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> dict:
        col = self.require_column(spec)
        alias = spec.alias or f"{col}_diff"
        periods = int(spec.params.get("periods", 1))
        every = spec.params.get("resample")
        agg = spec.params.get("agg", "mean")
        series = context.get_aggregated_series(col, every=every, agg=agg).drop_nulls()

        if len(series) <= periods:
            return {alias: None}

        value = series[-1] - series[-(periods + 1)]
        return {alias: float(value)}


class ValueAtLagFeature(SingleColumnFeatureExtractor):
    name = "value_at_lag"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> dict:
        col = self.require_column(spec)
        alias = spec.alias or f"{col}_lag"
        lag = int(spec.params.get("lag", 1))
        every = spec.params.get("resample")
        agg = spec.params.get("agg", "mean")
        series = context.get_aggregated_series(col, every=every, agg=agg).drop_nulls()

        if len(series) <= lag:
            return {alias: None}

        return {alias: float(series[-(lag + 1)])}


class DynamicPressureFeature(SingleColumnFeatureExtractor):
    name = "dynamic_pressure"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> dict:
        alias = spec.alias or "dynamic_pressure"
        density_col = spec.params.get("density_column")
        speed_col = spec.params.get("speed_column")
        agg = spec.params.get("agg", "mean")
        every = spec.params.get("resample")

        if not density_col or not speed_col:
            raise ValueError(
                "dynamic_pressure requires 'density_column' and 'speed_column'"
            )

        density_series = context.get_aggregated_series(density_col, every=every, agg=agg)
        speed_series = context.get_aggregated_series(speed_col, every=every, agg=agg)
        density_value = density_series.mean()
        speed_value = speed_series.mean()

        if density_value is None or speed_value is None:
            return {alias: None}

        return {alias: float(density_value * (speed_value ** 2))}


def build_default_registry():
    from features.registry import FeatureRegistry

    registry = FeatureRegistry()
    for extractor in (
        MeanFeature(),
        StdFeature(),
        MinFeature(),
        MaxFeature(),
        TrendFeature(),
        DiffFeature(),
        ValueAtLagFeature(),
        DynamicPressureFeature(),
        WaveletFeature(),
    ):
        registry.register(extractor)
    return registry
