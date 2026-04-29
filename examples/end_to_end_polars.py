from datetime import datetime, timedelta

import polars as pl

from ts_featurelab.features import FeatureEngine, WindowBuilder, build_default_registry, parse_feature_specs
from ts_featurelab.features.wavelet import pywt


def build_minute_dataframe() -> pl.DataFrame:
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
            "alias": "_p_dyn",
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
            "column": "_p_dyn",
            "alias": "_p_dyn_diff",
            "params": {
                "periods": 1,
            },
            "internal": True,
        },
        {
            "name": "value_at_lag",
            "column": "_p_dyn_diff",
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
    }


def main() -> None:
    df = build_minute_dataframe()
    config = build_feature_config()

    specs = parse_feature_specs(config)
    registry = build_default_registry()
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

    print("Raw minute-level input:")
    print(df.head(5))
    print()

    print("Number of windows:", len(samples))
    print("First window metadata:", samples[0].metadata if samples else {})
    print()

    print("Feature rows:")
    print(features_df)


if __name__ == "__main__":
    main()
