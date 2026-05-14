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
