from collections.abc import Iterable
from typing import Literal

import polars as pl


class FeatureContext:
    """Window-scoped access layer for raw, resampled, and derived series.

    The context is created for one prediction window and caches expensive
    resampling operations so several features can share the same aggregation.
    """

    def __init__(
        self,
        df_window: pl.DataFrame,
        time_col: str = "date",
        index_mode: Literal["time", "row"] = "time",
    ):
        """Initialize a feature context.

        Args:
            df_window: Polars dataframe containing one historical window.
            time_col: Name of the timestamp column used for sorting/resampling.
            index_mode: ``"time"`` for duration-aware windows, or ``"row"``
                for row-count windows.

        Raises:
            ValueError: If ``time_col`` is missing from ``df_window``.
        """
        if time_col not in df_window.columns:
            raise ValueError(f"Missing time column '{time_col}' in window")

        self.df_window = df_window.sort(time_col)
        self.time_col = time_col
        self.index_mode = index_mode
        self.cache: dict[tuple[object, ...], pl.DataFrame] = {}
        self.derived_series: dict[str, pl.Series] = {}

    def raw(self) -> pl.DataFrame:
        """Return the sorted raw dataframe for this window.

        Returns:
            Polars dataframe passed to the context, sorted by ``time_col``.
        """
        return self.df_window

    def raw_columns(self) -> set[str]:
        """Return raw dataframe column names.

        Returns:
            Set of column names available in the raw window dataframe.
        """
        return set(self.df_window.columns)

    def has_series(self, column: str) -> bool:
        """Check whether a raw or derived series is available.

        Args:
            column: Raw column name or derived series alias.

        Returns:
            ``True`` when the series can be consumed by a feature.
        """
        return column in self.raw_columns() or column in self.derived_series

    def add_series(self, alias: str, series: pl.Series) -> None:
        """Store a derived series for downstream features.

        Args:
            alias: Name under which downstream specs can reference the series.
            series: Series produced by an upstream feature.
        """
        normalized = series if series.name == alias else pl.Series(alias, series)
        self.derived_series[alias] = normalized

    def get_resampled(
        self,
        every: str,
        agg_map: dict[str, str | Iterable[str]],
    ) -> pl.DataFrame:
        """Return a cached dynamically resampled dataframe.

        Args:
            every: Polars duration string such as ``"1h"``.
            agg_map: Mapping from source column to aggregation name or names.

        Returns:
            Dataframe grouped by dynamic windows and sorted by ``time_col``.
        """
        if self.index_mode != "time":
            raise ValueError(
                "Feature parameter 'resample' is only supported in index_mode='time'. "
                "For row-based datasets, use raw window features or row-based lags."
            )

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
        """Normalize one or many aggregation names to a list.

        Args:
            aggs: Aggregation name or iterable of names.

        Returns:
            List of aggregation names.
        """
        if isinstance(aggs, str):
            return [aggs]
        return list(aggs)

    def _build_agg_expr(self, column: str, agg: str) -> pl.Expr:
        """Build a Polars aggregation expression with a stable alias.

        Args:
            column: Source column to aggregate.
            agg: Aggregation name.

        Returns:
            Polars expression named ``"{column}__{agg}"``.

        Raises:
            ValueError: If ``agg`` is not supported.
        """
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
        """Return a raw, derived, or resampled series for feature extraction.

        Args:
            column: Raw column name or derived series alias.
            every: Optional resampling interval. Derived series cannot be
                resampled here.
            agg: Aggregation used when ``every`` is provided.

        Returns:
            Polars series containing values for the requested input.

        Raises:
            ValueError: If the series is unknown or a derived series is
                requested with resampling.
        """
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
