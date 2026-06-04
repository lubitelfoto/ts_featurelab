# ts_featurelab

`ts_featurelab` builds rolling windows, feature rows, and optional supervised
targets from ordered Polars dataframes.

## Time-Based Windows

Datetime-indexed datasets keep the original duration-based behavior:

```python
from ts_featurelab.features import WindowBuilder

window_builder = WindowBuilder(
    time_col="date",
    window_size="24h",
    step="1h",
    min_history="24h",
)
samples = window_builder.transform(df)
```

## Row-Based Ordered Windows

For drilling, well-log, and other non-time-series datasets, set
`index_mode="row"` and use integer row counts. The dataframe is sorted by
`time_col`, but the selected column can be any sortable ordered axis such as
depth, measured depth, sample number, or frame index.

```python
from ts_featurelab.features import (
    FeatureEngine,
    TargetBuilder,
    TargetSpec,
    WindowBuilder,
    build_default_registry,
    parse_feature_specs,
)

config = {
    "time_col": "depth",
    "index_mode": "row",
    "features": [
        {
            "name": "mean",
            "column": "gamma_ray",
            "alias": "gamma_ray_mean_128",
        },
        {
            "name": "std",
            "column": "resistivity",
            "alias": "resistivity_std_128",
        },
        {
            "name": "value_at_lag",
            "column": "gamma_ray",
            "alias": "gamma_ray_lag_16",
            "params": {"lag": 16},
        },
    ],
}

samples = WindowBuilder(
    time_col=config["time_col"],
    index_mode=config["index_mode"],
    window_size=128,
    step=8,
    min_history=128,
).transform(df)

engine = FeatureEngine(
    build_default_registry(),
    time_col=config["time_col"],
    index_mode=config["index_mode"],
)
features_df = engine.transform_many(samples, parse_feature_specs(config))

target_df = TargetBuilder(time_col=config["time_col"]).transform(
    df,
    samples,
    TargetSpec(
        column="formation",
        alias="target_formation",
        task="classification",
        gap=0,
        horizon=16,
        agg="last",
    ),
)
```

In row mode, `resample` feature parameters are rejected because Polars dynamic
resampling only applies to time-like indices.

## Feature Extraction From One Existing Window

If windows are built outside of `WindowBuilder`, pass a ready Polars dataframe
window directly to `FeatureEngine.transform_window`. The method uses the
standard `features` section of the config and returns a one-row dataframe with
`prediction_time` plus the configured visible scalar features.

```python
import polars as pl

from ts_featurelab.features import (
    FeatureEngine,
    build_default_registry,
    parse_feature_specs,
)

config = {
    "time_col": "depth",
    "index_mode": "row",
    "features": [
        {
            "name": "mean",
            "column": "gamma_ray",
            "alias": "gamma_ray_mean",
        },
        {
            "name": "value_at_lag",
            "column": "gamma_ray",
            "alias": "gamma_ray_lag_1",
            "params": {"lag": 1},
        },
    ],
}

df_window = df.filter(pl.col("depth").is_between(1200.0, 1208.0))
engine = FeatureEngine(
    build_default_registry(),
    time_col=config["time_col"],
    index_mode=config["index_mode"],
)

feature_window = engine.transform_window(
    df_window,
    parse_feature_specs(config),
)
```

When `prediction_time` is not passed, the engine uses the latest value of
`time_col` inside `df_window`. Pass `prediction_time=...` explicitly if the
window has an external prediction/index value.
