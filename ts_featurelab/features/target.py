from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

import polars as pl

from ts_featurelab.features.window import WindowSample
from ts_featurelab.features.window_builder import parse_duration_to_timedelta

TargetTask = Literal["regression", "classification"]


@dataclass
class TargetSpec:
    """Configuration for building supervised targets from future values.

    Attributes:
        column: Source column used as the future target value.
        alias: Output target column name.
        task: ``"regression"`` returns numeric values, while
            ``"classification"`` returns classes.
        horizon: Duration after ``prediction_time`` used to select future rows.
            Rows are selected from ``(prediction_time, prediction_time + horizon]``.
        agg: Aggregation applied to the selected future values.
        thresholds: Optional ascending cut points for numeric classification.
            With thresholds ``[0.0]``, values ``<= 0`` get the first class and
            values ``> 0`` get the second class.
        labels: Optional class labels. Must contain ``len(thresholds) + 1``
            values when thresholds are provided.
        drop_nulls: Whether rows without a target should be removed.
    """

    column: str
    alias: str = "target"
    task: TargetTask = "regression"
    horizon: str = "1h"
    agg: str = "last"
    thresholds: list[float] | None = None
    labels: list[Any] | None = None
    drop_nulls: bool = True


class TargetBuilder:
    """Build regression or classification targets for rolling windows."""

    def __init__(self, time_col: str = "date"):
        """Initialize target builder.

        Args:
            time_col: Timestamp column used to align future values.
        """
        self.time_col = time_col

    def transform(
        self,
        df: pl.DataFrame,
        samples: list[WindowSample],
        spec: TargetSpec,
    ) -> pl.DataFrame:
        """Build one target row per window sample.

        Args:
            df: Full time-ordered dataframe containing future observations.
            samples: Window samples produced by ``WindowBuilder``.
            spec: Target configuration.

        Returns:
            Dataframe with ``prediction_time`` and the configured target alias.

        Raises:
            ValueError: If required columns are missing or the target spec is
                inconsistent.
        """
        self._validate_inputs(df, spec)
        if not samples:
            return pl.DataFrame({"prediction_time": [], spec.alias: []})

        df = df.sort(self.time_col)
        horizon_td = parse_duration_to_timedelta(spec.horizon)
        rows: list[dict[str, object]] = []

        for sample in samples:
            value = self._future_value(
                df=df,
                prediction_time=sample.prediction_time,
                horizon_td=horizon_td,
                spec=spec,
            )
            if spec.task == "classification" and value is not None:
                value = self._classify(value, spec)

            if value is None and spec.drop_nulls:
                continue

            rows.append(
                {
                    "prediction_time": sample.prediction_time,
                    spec.alias: value,
                }
            )

        if not rows:
            return pl.DataFrame({"prediction_time": [], spec.alias: []})
        return pl.DataFrame(rows)

    def attach(
        self,
        features_df: pl.DataFrame,
        target_df: pl.DataFrame,
        target_alias: str = "target",
        drop_nulls: bool = True,
    ) -> pl.DataFrame:
        """Join targets to already computed feature rows.

        Args:
            features_df: Dataframe returned by ``FeatureEngine.transform_many``.
            target_df: Dataframe returned by ``TargetBuilder.transform``.
            target_alias: Name of the target column to check for nulls.
            drop_nulls: Whether rows with missing targets should be removed.

        Returns:
            Feature dataframe with the target column joined by prediction time.
        """
        dataset = features_df.join(target_df, on="prediction_time", how="left")
        if drop_nulls and target_alias in dataset.columns:
            dataset = dataset.drop_nulls(target_alias)
        return dataset

    def _validate_inputs(self, df: pl.DataFrame, spec: TargetSpec) -> None:
        """Validate dataframe columns and target settings."""
        if self.time_col not in df.columns:
            raise ValueError(f"Missing time column '{self.time_col}'")
        if spec.column not in df.columns:
            raise ValueError(f"Missing target column '{spec.column}'")
        if spec.task not in ("regression", "classification"):
            raise ValueError("target task must be 'regression' or 'classification'")
        if spec.thresholds is not None:
            thresholds = list(spec.thresholds)
            if thresholds != sorted(thresholds):
                raise ValueError("classification thresholds must be sorted ascending")
            if spec.labels is not None and len(spec.labels) != len(thresholds) + 1:
                raise ValueError(
                    "classification labels must have len(thresholds) + 1 values"
                )

    def _future_value(
        self,
        df: pl.DataFrame,
        prediction_time: object,
        horizon_td: timedelta,
        spec: TargetSpec,
    ) -> object:
        """Aggregate future values for one prediction time."""
        future_end = prediction_time + horizon_td
        if horizon_td.total_seconds() == 0:
            future_df = df.filter(pl.col(self.time_col) == prediction_time)
        else:
            future_df = df.filter(
                (pl.col(self.time_col) > prediction_time)
                & (pl.col(self.time_col) <= future_end)
            )

        if future_df.is_empty():
            return None

        series = future_df[spec.column].drop_nulls()
        if series.is_empty():
            return None

        return self._aggregate(series, spec.agg)

    def _aggregate(self, series: pl.Series, agg: str) -> object:
        """Aggregate a non-empty Polars series into a scalar target value."""
        if agg == "first":
            return series[0]
        if agg == "last":
            return series[-1]
        if agg == "mean":
            value = series.mean()
            return None if value is None else float(value)
        if agg == "sum":
            value = series.sum()
            return None if value is None else float(value)
        if agg == "min":
            return series.min()
        if agg == "max":
            return series.max()
        if agg == "std":
            value = series.std()
            return None if value is None else float(value)

        raise ValueError(f"Unsupported target aggregation '{agg}'")

    def _classify(self, value: object, spec: TargetSpec) -> object:
        """Convert a future value into a class label when thresholds are set."""
        if spec.thresholds is None:
            return value

        numeric_value = float(value)
        class_idx = 0
        for threshold in spec.thresholds:
            if numeric_value <= threshold:
                break
            class_idx += 1

        if spec.labels is not None:
            return spec.labels[class_idx]
        return class_idx


def parse_target_spec(config: dict[str, Any]) -> TargetSpec | None:
    """Parse optional top-level target configuration.

    Args:
        config: Configuration dictionary with an optional ``target`` entry.

    Returns:
        Parsed ``TargetSpec`` or ``None`` when no target is configured.

    Raises:
        ValueError: If the target configuration is missing required fields.
    """
    raw_target = config.get("target")
    if raw_target is None:
        return None
    if not isinstance(raw_target, dict):
        raise ValueError("'target' must be a dict")

    column = raw_target.get("column")
    if not column or not isinstance(column, str):
        raise ValueError("target is missing string 'column'")

    alias = raw_target.get("alias", "target")
    if not isinstance(alias, str):
        raise ValueError("target alias must be a string")

    task = raw_target.get("task", "regression")
    if task not in ("regression", "classification"):
        raise ValueError("target task must be 'regression' or 'classification'")

    thresholds = raw_target.get("thresholds")
    if thresholds is not None:
        if not isinstance(thresholds, list):
            raise ValueError("target thresholds must be a list")
        thresholds = [float(value) for value in thresholds]

    labels = raw_target.get("labels")
    if labels is not None and not isinstance(labels, list):
        raise ValueError("target labels must be a list")

    horizon = raw_target.get("horizon", "1h")
    if not isinstance(horizon, str):
        raise ValueError("target horizon must be a string")

    agg = raw_target.get("agg", "last")
    if not isinstance(agg, str):
        raise ValueError("target agg must be a string")

    drop_nulls = raw_target.get("drop_nulls", True)
    if not isinstance(drop_nulls, bool):
        raise ValueError("target drop_nulls must be a bool")

    return TargetSpec(
        column=column,
        alias=alias,
        task=task,
        horizon=horizon,
        agg=agg,
        thresholds=thresholds,
        labels=labels,
        drop_nulls=drop_nulls,
    )
