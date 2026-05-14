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
        horizon: Duration after ``prediction_time`` in time mode, or row count
            after the prediction row in row mode.
        gap: Optional gap before the target horizon. Use a duration in time
            mode or a row count in row mode.
        agg: Aggregation applied to the selected future values.
        thresholds: Optional ascending cut points for numeric classification.
            With thresholds ``[0.0]``, values ``<= 0`` get the first class and
            values ``> 0`` get the second class.
        labels: Optional class labels. Must contain ``len(thresholds) + 1``
            values when thresholds are provided.
        drop_nulls: Whether rows without a target should be removed.
        index_mode: Optional target mode. Defaults to the window sample mode.
    """

    column: str
    alias: str = "target"
    task: TargetTask = "regression"
    horizon: str | int = "1h"
    gap: str | int = 0
    agg: str = "last"
    thresholds: list[float] | None = None
    labels: list[Any] | None = None
    drop_nulls: bool = True
    index_mode: Literal["time", "row"] | None = None


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
        index_mode = self._resolve_index_mode(samples, spec)
        self._validate_inputs(df, spec, index_mode=index_mode)
        if not samples:
            return pl.DataFrame({"prediction_time": [], spec.alias: []})

        df = df.sort(self.time_col)
        if index_mode == "time":
            rows = self._transform_time(df, samples, spec)
        elif index_mode == "row":
            rows = self._transform_row(df, samples, spec)
        else:
            raise ValueError("index_mode must be 'time' or 'row'")

        if not rows:
            return pl.DataFrame({"prediction_time": [], spec.alias: []})
        return pl.DataFrame(rows)

    def _transform_time(
        self,
        df: pl.DataFrame,
        samples: list[WindowSample],
        spec: TargetSpec,
    ) -> list[dict[str, object]]:
        """Build targets by timestamp horizon and optional timestamp gap."""
        horizon_td = parse_duration_to_timedelta(str(spec.horizon))
        gap_td = self._parse_time_gap(spec.gap)
        rows: list[dict[str, object]] = []

        for sample in samples:
            value = self._future_value(
                df=df,
                prediction_time=sample.prediction_time,
                horizon_td=horizon_td,
                gap_td=gap_td,
                spec=spec,
            )
            self._append_target_row(rows, sample.prediction_time, value, spec)

        return rows

    def _transform_row(
        self,
        df: pl.DataFrame,
        samples: list[WindowSample],
        spec: TargetSpec,
    ) -> list[dict[str, object]]:
        """Build targets by row horizon and optional row gap."""
        horizon = int(spec.horizon)
        gap = int(spec.gap)
        rows: list[dict[str, object]] = []

        for sample in samples:
            if "prediction_row_idx" not in sample.metadata:
                raise ValueError(
                    "row target mode requires sample metadata 'prediction_row_idx'"
                )

            value = self._future_row_value(
                df=df,
                prediction_row_idx=int(sample.metadata["prediction_row_idx"]),
                horizon=horizon,
                gap=gap,
                spec=spec,
            )
            self._append_target_row(rows, sample.prediction_time, value, spec)

        return rows

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

    def _validate_inputs(
        self,
        df: pl.DataFrame,
        spec: TargetSpec,
        index_mode: Literal["time", "row"],
    ) -> None:
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
        if index_mode == "time":
            if not isinstance(spec.horizon, str):
                raise ValueError(
                    "target horizon must be a duration string when index_mode='time'"
                )
            self._parse_time_gap(spec.gap)
        elif index_mode == "row":
            if type(spec.horizon) is not int or spec.horizon <= 0:
                raise ValueError(
                    "target horizon must be a positive integer when index_mode='row'"
                )
            if type(spec.gap) is not int or spec.gap < 0:
                raise ValueError(
                    "target gap must be a non-negative integer when index_mode='row'"
                )
        else:
            raise ValueError("index_mode must be 'time' or 'row'")

    def _resolve_index_mode(
        self,
        samples: list[WindowSample],
        spec: TargetSpec,
    ) -> Literal["time", "row"]:
        """Resolve explicit target mode or inherit it from window metadata."""
        mode = spec.index_mode
        if mode is None and samples:
            mode = samples[0].metadata.get("index_mode", "time")
        if mode is None:
            mode = "time"
        if mode not in ("time", "row"):
            raise ValueError("target index_mode must be 'time', 'row', or None")
        return mode

    def _future_value(
        self,
        df: pl.DataFrame,
        prediction_time: object,
        horizon_td: timedelta,
        gap_td: timedelta,
        spec: TargetSpec,
    ) -> object:
        """Aggregate future values for one prediction time."""
        future_start = prediction_time + gap_td
        future_end = future_start + horizon_td
        if horizon_td.total_seconds() == 0:
            future_df = df.filter(pl.col(self.time_col) == future_start)
        else:
            future_df = df.filter(
                (pl.col(self.time_col) > future_start)
                & (pl.col(self.time_col) <= future_end)
            )

        return self._aggregate_future_df(future_df, spec)

    def _future_row_value(
        self,
        df: pl.DataFrame,
        prediction_row_idx: int,
        horizon: int,
        gap: int,
        spec: TargetSpec,
    ) -> object:
        """Aggregate future values for one prediction row index."""
        start_idx = prediction_row_idx + gap + 1
        if start_idx >= df.height:
            return None

        future_df = df.slice(start_idx, horizon)
        return self._aggregate_future_df(future_df, spec)

    def _aggregate_future_df(
        self,
        future_df: pl.DataFrame,
        spec: TargetSpec,
    ) -> object:
        """Aggregate the target column from a future dataframe slice."""
        if future_df.is_empty():
            return None

        series = future_df[spec.column].drop_nulls()
        if series.is_empty():
            return None

        return self._aggregate(series, spec.agg)

    def _append_target_row(
        self,
        rows: list[dict[str, object]],
        prediction_time: object,
        value: object,
        spec: TargetSpec,
    ) -> None:
        """Append a target row after classification/null handling."""
        if spec.task == "classification" and value is not None:
            value = self._classify(value, spec)

        if value is None and spec.drop_nulls:
            return

        rows.append(
            {
                "prediction_time": prediction_time,
                spec.alias: value,
            }
        )

    def _parse_time_gap(self, gap: str | int) -> timedelta:
        """Parse a time-mode gap while preserving legacy default ``0``."""
        if gap == 0:
            return timedelta(0)
        if not isinstance(gap, str):
            raise ValueError(
                "target gap must be a duration string when index_mode='time'"
            )
        return parse_duration_to_timedelta(gap)

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

    index_mode = raw_target.get("index_mode", config.get("index_mode"))
    if index_mode is not None and index_mode not in ("time", "row"):
        raise ValueError("target index_mode must be 'time' or 'row'")

    horizon = raw_target.get("horizon", "1h")
    if index_mode == "row":
        if type(horizon) is not int or horizon <= 0:
            raise ValueError(
                "target horizon must be an integer when index_mode='row'"
            )
    elif not isinstance(horizon, str):
        raise ValueError("target horizon must be a string")

    gap = raw_target.get("gap", 0)
    if index_mode == "row":
        if type(gap) is not int or gap < 0:
            raise ValueError("target gap must be an integer when index_mode='row'")
    elif not (isinstance(gap, str) or gap == 0):
        raise ValueError("target gap must be a string")

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
        gap=gap,
        agg=agg,
        thresholds=thresholds,
        labels=labels,
        drop_nulls=drop_nulls,
        index_mode=index_mode,
    )
