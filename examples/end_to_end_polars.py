from datetime import datetime, timedelta

import polars as pl

from ts_featurelab.features.base import FeatureResult, SingleColumnFeatureExtractor
from ts_featurelab.features.context import FeatureContext
from ts_featurelab.features.featurespec import FeatureSpec
from ts_featurelab.features import (
    FeatureEngine,
    TargetBuilder,
    TargetSpec,
    WindowBuilder,
    build_default_registry,
    parse_feature_specs,
    parse_target_spec,
)
from ts_featurelab.features.wavelet import pywt


# Example of a domain-specific feature.
# Not part of the core library.
class DynamicPressureFeature(SingleColumnFeatureExtractor):
    """Example domain feature that computes dynamic pressure as a series.

    Params:
        density_column: Raw or derived density series name.
        speed_column: Raw or derived speed series name.
        resample: Optional Polars duration string used before extraction.
        agg: Aggregation applied during resampling. Defaults to ``"mean"``.
    """

    name = "dynamic_pressure"
    output_kind = "series"

    def validate_spec(self, spec: FeatureSpec) -> None:
        """Validate that density and speed inputs are configured.

        Args:
            spec: Feature specification for dynamic pressure.

        Raises:
            ValueError: If ``density_column`` or ``speed_column`` is missing.
        """
        density_col = spec.params.get("density_column")
        speed_col = spec.params.get("speed_column")
        if not density_col or not speed_col:
            raise ValueError(
                "dynamic_pressure requires 'density_column' and 'speed_column'"
            )

    def get_dependencies(self, spec: FeatureSpec) -> set[str]:
        """Return density and speed dependencies.

        Args:
            spec: Feature specification for dynamic pressure.

        Returns:
            Set of configured dependency names.
        """
        density_col = spec.params.get("density_column")
        speed_col = spec.params.get("speed_column")
        return {value for value in (density_col, speed_col) if value}

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> FeatureResult:
        """Compute dynamic pressure values for the current window.

        Args:
            context: Window-scoped feature context.
            spec: Feature specification with density and speed params.

        Returns:
            Series result named ``spec.alias``.

        Raises:
            ValueError: If input series lengths differ after aggregation.
        """
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
        return FeatureResult.from_series(spec.alias, pl.Series(spec.alias, values))


def build_minute_dataframe() -> pl.DataFrame:
    """Build synthetic minute-level input data for the example.

    Returns:
        Polars dataframe with timestamp, speed, and density columns.
    """
    start = datetime(2024, 1, 1, 0, 0, 0)
    minutes = 12 * 60

    rows: list[dict[str, object]] = []
    for idx in range(minutes):
        ts = start + timedelta(minutes=idx)

        # Two raw minute-level signals with different behavior.
        speed = 400.0 + (idx % 180) * 0.8 + ((idx // 60) % 4) * 6.0
        density = 5.0 + (idx % 45) * 0.05 + ((idx // 30) % 3) * 0.2

        rows.append(
            {
                "date": ts,
                "oe_f1m_proton_speed": speed,
                "oe_f1m_proton_density": density,
            }
        )

    return pl.DataFrame(rows)


def build_feature_config() -> dict:
    """Build an example feature configuration dictionary.

    Returns:
        Configuration with raw scalar features, internal derived features, and
        optional wavelet features when PyWavelets is importable.
    """
    features = [
        {
            "name": "mean",
            "column": "oe_f1m_proton_speed",
            "alias": "speed_mean_raw_window",
        },
        {
            "name": "std",
            "column": "oe_f1m_proton_speed",
            "alias": "speed_std_raw_window",
        },
        {
            "name": "trend",
            "column": "oe_f1m_proton_speed",
            "alias": "speed_trend_raw_window",
        },
        {
            "name": "value_at_lag",
            "column": "oe_f1m_proton_speed",
            "alias": "speed_hourly_mean_lag_2",
            "params": {
                "lag": 2,
                "resample": "1h",
                "agg": "mean",
            },
        },
        {
            "name": "dynamic_pressure",
            "alias": "p_dyn",
            "params": {
                "density_column": "oe_f1m_proton_density",
                "speed_column": "oe_f1m_proton_speed",
                "resample": "1h",
                "agg": "mean",
            },
            "internal": True,
        },
        {
            "name": "diff",
            "column": "p_dyn",
            "alias": "p_dyn_diff",
            "params": {
                "periods": 1,
            },
            "internal": True,
        },
        {
            "name": "value_at_lag",
            "column": "p_dyn_diff",
            "alias": "p_dyn_diff_lag2",
            "params": {
                "lag": 2,
            },
        },
    ]

    if pywt is not None:
        features.append(
            {
                "name": "wavelet",
                "column": "oe_f1m_proton_speed",
                "alias": "speed",
                "params": {
                    "wavelet": "db4",
                    "level": 4,
                    "min_points": 128,
                },
            }
        )

    return {
        "time_col": "date",
        "features": features,
        "target": {
            "column": "oe_f1m_proton_speed",
            "alias": "speed_next_hour",
            "task": "regression",
            "horizon": "1h",
            "agg": "last",
        },
    }


def main() -> None:
    """Run the end-to-end Polars feature extraction example."""
    df = build_minute_dataframe()
    config = build_feature_config()

    specs = parse_feature_specs(config)
    registry = build_default_registry()
    registry.register(DynamicPressureFeature())
    engine = FeatureEngine(registry, time_col=config["time_col"])

    # Each sample contains only past data up to prediction_time.
    window_builder = WindowBuilder(
        time_col="date",
        window_size="4h",
        step="2h",
        min_history="4h",
    )
    samples = window_builder.transform(df)

    print("Execution plan:")
    print(engine.explain(specs, raw_columns=set(df.columns)))
    print()

    features_df = engine.transform_many(samples, specs)
    target_builder = TargetBuilder(time_col=config["time_col"])

    regression_target = parse_target_spec(config)
    targets_df = target_builder.transform(df, samples, regression_target)
    regression_dataset = target_builder.attach(
        features_df,
        targets_df,
        target_alias=regression_target.alias,
    )

    classification_target = TargetSpec(
        column="oe_f1m_proton_speed",
        alias="speed_direction_class",
        task="classification",
        horizon="1h",
        agg="last",
        thresholds=[features_df["speed_mean_raw_window"].mean()],
        labels=["low_or_equal", "high"],
    )
    class_targets_df = target_builder.transform(df, samples, classification_target)
    classification_dataset = target_builder.attach(
        features_df,
        class_targets_df,
        target_alias=classification_target.alias,
    )

    print("Raw minute-level input:")
    print(df.head(5))
    print()

    print("Number of windows:", len(samples))
    print("First window metadata:", samples[0].metadata if samples else {})
    print()

    print("Feature rows:")
    print(features_df)
    print()

    print("Regression dataset:")
    print(regression_dataset)
    print()

    print("Classification dataset:")
    print(classification_dataset)


if __name__ == "__main__":
    main()
