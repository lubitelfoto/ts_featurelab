from datetime import timedelta

import polars as pl

from features.window import WindowSample


class WindowBuilder:
    def __init__(
        self,
        time_col: str = "date",
        window_size: str = "24h",
        step: str = "1h",
        min_history: str | None = None,
    ):
        self.time_col = time_col
        self.window_size = window_size
        self.step = step
        self.min_history = min_history or window_size

    def transform(self, df: pl.DataFrame) -> list[WindowSample]:
        if self.time_col not in df.columns:
            raise ValueError(f"Missing time column '{self.time_col}'")

        df = df.sort(self.time_col)
        times = df[self.time_col].to_list()
        if not times:
            return []

        step_td = _parse_duration_to_timedelta(self.step)
        window_td = _parse_duration_to_timedelta(self.window_size)
        min_history_td = _parse_duration_to_timedelta(self.min_history)

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
                        "window_start": window_start,
                        "window_end": prediction_time,
                        "window_size": self.window_size,
                    },
                )
            )

        return samples


def _parse_duration_to_timedelta(value: str) -> timedelta:
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
