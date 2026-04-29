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


class DynamicPressureFeature(SingleColumnFeatureExtractor):
    name = "dynamic_pressure"
    output_kind = "series"

    def validate_spec(self, spec: FeatureSpec) -> None:
        density_col = spec.params.get("density_column")
        speed_col = spec.params.get("speed_column")
        if not density_col or not speed_col:
            raise ValueError(
                "dynamic_pressure requires 'density_column' and 'speed_column'"
            )

    def get_dependencies(self, spec: FeatureSpec) -> set[str]:
        density_col = spec.params.get("density_column")
        speed_col = spec.params.get("speed_column")
        deps = {value for value in (density_col, speed_col) if value}
        return deps

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> pl.Series:
        alias = spec.alias
        density_col = spec.params.get("density_column")
        speed_col = spec.params.get("speed_column")
        agg = spec.params.get("agg", "mean")
        every = spec.params.get("resample")

        density_series = context.get_aggregated_series(density_col, every=every, agg=agg)
        speed_series = context.get_aggregated_series(speed_col, every=every, agg=agg)
        if len(density_series) != len(speed_series):
            raise ValueError(
                f"dynamic_pressure inputs '{density_col}' and '{speed_col}' have incompatible lengths"
            )

        values = [
            None if density is None or speed is None else float(density * (speed ** 2))
            for density, speed in zip(density_series.to_list(), speed_series.to_list())
        ]
        return pl.Series(alias, values)


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
        DynamicPressureFeature(),
        WaveletFeature(),
    ):
        registry.register(extractor)
    return registry
