from datetime import timedelta
from typing import Literal

import polars as pl

from ts_featurelab.features.window import WindowSample


class WindowBuilder:
    """Build rolling historical windows from a time-ordered dataframe."""

    def __init__(
        self,
        time_col: str = "date",
        window_size: str | int = "24h",
        step: str | int = "1h",
        min_history: str | int | None = None,
        index_mode: Literal["time", "row"] = "time",
    ):
        """Initialize rolling-window parameters.

        Args:
            time_col: Ordered index column used to sort and slice the dataframe.
            window_size: Duration for time mode, or row count for row mode.
            step: Distance between consecutive prediction points.
            min_history: Minimum history before the first prediction time.
                Defaults to ``window_size``.
            index_mode: ``"time"`` for datetime duration windows, or ``"row"``
                for row-count windows after sorting by ``time_col``.
        """
        self.time_col = time_col
        self.window_size = window_size
        self.step = step
        self.min_history = min_history or window_size
        self.index_mode = index_mode

    def transform(self, df: pl.DataFrame) -> list[WindowSample]:
        """Convert a dataframe into rolling ``WindowSample`` objects.

        Args:
            df: Input dataframe containing ``time_col``.

        Returns:
            List of non-empty windows with prediction time and metadata.

        Raises:
            ValueError: If ``time_col`` is missing.
        """
        if self.time_col not in df.columns:
            raise ValueError(f"Missing time column '{self.time_col}'")

        if self.index_mode == "time":
            return self._transform_time(df)
        if self.index_mode == "row":
            return self._transform_row(df)

        raise ValueError("index_mode must be 'time' or 'row'")

    def _transform_time(self, df: pl.DataFrame) -> list[WindowSample]:
        """Build duration-based rolling windows."""
        self._validate_time_params()
        df = df.sort(self.time_col)
        times = df[self.time_col].to_list()
        if not times:
            return []

        step_td = parse_duration_to_timedelta(self.step)
        window_td = parse_duration_to_timedelta(self.window_size)
        min_history_td = parse_duration_to_timedelta(self.min_history)

        start_time = times[0] + min_history_td
        last_time = times[-1]

        prediction_times: list[object] = []
        current = start_time
        while current <= last_time:
            prediction_times.append(current)
            current += step_td

        samples: list[WindowSample] = []
        for prediction_time in prediction_times:
            window_start = prediction_time - window_td
            df_window = df.filter(
                (pl.col(self.time_col) > window_start)
                & (pl.col(self.time_col) <= prediction_time)
            )
            if df_window.is_empty():
                continue

            samples.append(
                WindowSample(
                    prediction_time=prediction_time,
                    df=df_window,
                    metadata={
                        "index_mode": "time",
                        "window_start": window_start,
                        "window_end": prediction_time,
                        "window_size": self.window_size,
                        "step": self.step,
                    },
                )
            )

        return samples

    def _transform_row(self, df: pl.DataFrame) -> list[WindowSample]:
        """Build row-count windows after sorting by the index column."""
        self._validate_row_params()
        df = df.sort(self.time_col)
        if df.is_empty():
            return []

        window_size = int(self.window_size)
        step = int(self.step)
        min_history = int(self.min_history)

        samples: list[WindowSample] = []
        for end_idx in range(min_history - 1, df.height, step):
            window_start_idx = end_idx - window_size + 1
            if window_start_idx < 0:
                continue

            df_window = df.slice(window_start_idx, window_size)
            prediction_value = df[self.time_col][end_idx]
            window_start = df[self.time_col][window_start_idx]

            samples.append(
                WindowSample(
                    prediction_time=prediction_value,
                    df=df_window,
                    metadata={
                        "index_mode": "row",
                        "prediction_row_idx": end_idx,
                        "prediction_index_value": prediction_value,
                        "window_start_idx": window_start_idx,
                        "window_end_idx": end_idx,
                        "window_start": window_start,
                        "window_end": prediction_value,
                        "window_size": window_size,
                        "step": step,
                    },
                )
            )

        return samples

    def _validate_time_params(self) -> None:
        """Validate duration parameters for time mode."""
        for name, value in (
            ("window_size", self.window_size),
            ("step", self.step),
            ("min_history", self.min_history),
        ):
            if not isinstance(value, str):
                raise ValueError(
                    f"{name} must be a duration string when index_mode='time'"
                )

    def _validate_row_params(self) -> None:
        """Validate row-count parameters for row mode."""
        for name, value in (
            ("window_size", self.window_size),
            ("step", self.step),
            ("min_history", self.min_history),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"{name} must be a positive integer when index_mode='row'"
                )


def parse_duration_to_timedelta(value: str) -> timedelta:
    """Parse a compact duration string into ``datetime.timedelta``.

    Supported units are minutes (``m``), hours (``h``), and days (``d``).

    Args:
        value: Duration string such as ``"30m"``, ``"4h"``, or ``"1d"``.

    Returns:
        Equivalent ``timedelta``.

    Raises:
        ValueError: If the duration format or unit is unsupported.
    """
    unit_map = {
        "m": "minutes",
        "h": "hours",
        "d": "days",
    }
    if len(value) < 2:
        raise ValueError(f"Unsupported duration '{value}'")

    unit = value[-1]
    amount = int(value[:-1])
    if unit not in unit_map:
        raise ValueError(f"Unsupported duration unit '{unit}'")

    return timedelta(**{unit_map[unit]: amount})


_parse_duration_to_timedelta = parse_duration_to_timedelta
