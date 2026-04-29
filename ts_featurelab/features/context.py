from collections.abc import Iterable

import polars as pl


class FeatureContext:
    def __init__(self, df_window: pl.DataFrame, time_col: str = "date"):
        if time_col not in df_window.columns:
            raise ValueError(f"Missing time column '{time_col}' in window")

        self.df_window = df_window.sort(time_col)
        self.time_col = time_col
        self.cache: dict[tuple[object, ...], pl.DataFrame] = {}
        self.derived_series: dict[str, pl.Series] = {}

    def raw(self) -> pl.DataFrame:
        return self.df_window

    def raw_columns(self) -> set[str]:
        return set(self.df_window.columns)

    def has_series(self, column: str) -> bool:
        return column in self.raw_columns() or column in self.derived_series

    def add_series(self, alias: str, series: pl.Series) -> None:
        normalized = series if series.name == alias else pl.Series(alias, series)
        self.derived_series[alias] = normalized

    def get_resampled(
        self,
        every: str,
        agg_map: dict[str, str | Iterable[str]],
    ) -> pl.DataFrame:
        normalized = tuple(
            sorted(
                (column, tuple(sorted(self._normalize_aggs(aggs))))
                for column, aggs in agg_map.items()
            )
        )
        key = ("resampled", self.time_col, every, normalized)

        if key not in self.cache:
            exprs: list[pl.Expr] = []
            for column, aggs in normalized:
                for agg in aggs:
                    exprs.append(self._build_agg_expr(column, agg))

            self.cache[key] = (
                self.df_window.group_by_dynamic(
                    self.time_col,
                    every=every,
                    period=every,
                    closed="right",
                    label="right",
                )
                .agg(exprs)
                .sort(self.time_col)
            )

        return self.cache[key]

    def _normalize_aggs(self, aggs: str | Iterable[str]) -> list[str]:
        if isinstance(aggs, str):
            return [aggs]
        return list(aggs)

    def _build_agg_expr(self, column: str, agg: str) -> pl.Expr:
        expr = pl.col(column)
        alias = f"{column}__{agg}"

        if agg == "mean":
            return expr.mean().alias(alias)
        if agg == "sum":
            return expr.sum().alias(alias)
        if agg == "min":
            return expr.min().alias(alias)
        if agg == "max":
            return expr.max().alias(alias)
        if agg == "std":
            return expr.std().alias(alias)
        if agg == "last":
            return expr.last().alias(alias)
        if agg == "first":
            return expr.first().alias(alias)

        raise ValueError(f"Unsupported aggregation '{agg}'")

    def get_aggregated_series(
        self,
        column: str,
        every: str | None = None,
        agg: str = "mean",
    ) -> pl.Series:
        if column in self.derived_series:
            if every is not None:
                raise ValueError(
                    f"Cannot resample derived series '{column}'; resample it while creating it"
                )
            return self.derived_series[column]

        if column not in self.df_window.columns:
            raise ValueError(f"Unknown series column '{column}'")

        if every is None:
            return self.raw()[column]

        df = self.get_resampled(every=every, agg_map={column: agg})
        return df[f"{column}__{agg}"]
