import polars as pl

from ts_featurelab.features.base import SingleColumnFeatureExtractor
from ts_featurelab.features.context import FeatureContext
from ts_featurelab.features.featurespec import FeatureSpec
from ts_featurelab.features.mean import MeanFeature
from ts_featurelab.features.trend import TrendFeature
from ts_featurelab.features.wavelet import WaveletFeature


class StdFeature(SingleColumnFeatureExtractor):
    name = "std"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> dict:
        col = self.require_column(spec)
        alias = spec.alias
        value = context.get_aggregated_series(
            col,
            every=spec.params.get("resample"),
            agg=spec.params.get("agg", "mean"),
        ).std()
        return {alias: None if value is None else float(value)}


class MinFeature(SingleColumnFeatureExtractor):
    name = "min"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> dict:
        col = self.require_column(spec)
        alias = spec.alias
        value = context.get_aggregated_series(
            col,
            every=spec.params.get("resample"),
            agg=spec.params.get("agg", "mean"),
        ).min()
        return {alias: None if value is None else float(value)}


class MaxFeature(SingleColumnFeatureExtractor):
    name = "max"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> dict:
        col = self.require_column(spec)
        alias = spec.alias
        value = context.get_aggregated_series(
            col,
            every=spec.params.get("resample"),
            agg=spec.params.get("agg", "mean"),
        ).max()
        return {alias: None if value is None else float(value)}


class DiffFeature(SingleColumnFeatureExtractor):
    name = "diff"
    output_kind = "series"

    def validate_spec(self, spec: FeatureSpec) -> None:
        periods = int(spec.params.get("periods", 1))
        if periods < 0:
            raise ValueError("diff does not allow negative periods")

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> pl.Series:
        col = self.require_column(spec)
        alias = spec.alias
        periods = int(spec.params.get("periods", 1))
        every = spec.params.get("resample")
        agg = spec.params.get("agg", "mean")
        series = context.get_aggregated_series(col, every=every, agg=agg)
        return series.diff(n=periods).rename(alias)


class ValueAtLagFeature(SingleColumnFeatureExtractor):
    name = "value_at_lag"

    def validate_spec(self, spec: FeatureSpec) -> None:
        lag = int(spec.params.get("lag", 1))
        if lag < 0:
            raise ValueError("value_at_lag does not allow negative lag")

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> dict:
        col = self.require_column(spec)
        alias = spec.alias
        lag = int(spec.params.get("lag", 1))
        every = spec.params.get("resample")
        agg = spec.params.get("agg", "mean")
        series = context.get_aggregated_series(col, every=every, agg=agg).drop_nulls()

        if len(series) <= lag:
            return {alias: None}

        return {alias: float(series[-(lag + 1)])}


def build_default_registry():
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
