from typing import Literal

import polars as pl

from ts_featurelab.features.base import FeatureResult
from ts_featurelab.features.context import FeatureContext
from ts_featurelab.features.featurespec import FeatureSpec
from ts_featurelab.features.planner import ExecutionPlan, PlannedFeature, build_execution_plan
from ts_featurelab.features.registry import FeatureRegistry
from ts_featurelab.features.window import WindowSample


class FeatureEngine:
    """Execute feature specifications over one or many time windows."""

    def __init__(
        self,
        registry: FeatureRegistry,
        time_col: str = "date",
        index_mode: Literal["time", "row"] = "time",
    ):
        """Initialize the feature engine.

        Args:
            registry: Registry containing available feature extractors.
            time_col: Timestamp column used by each ``FeatureContext``.
            index_mode: ``"time"`` for duration-aware contexts, or ``"row"``
                for row-count contexts.
        """
        self.registry = registry
        self.time_col = time_col
        self.index_mode = index_mode

    def build_plan(
        self,
        specs: list[FeatureSpec],
        raw_columns: set[str],
    ) -> ExecutionPlan:
        """Build a dependency-aware execution plan for feature specs.

        Args:
            specs: Feature specifications to execute.
            raw_columns: Columns available in the input dataframe.

        Returns:
            Execution plan grouped into dependency stages.
        """
        return build_execution_plan(specs, raw_columns=raw_columns, registry=self.registry)

    def explain(
        self,
        specs: list[FeatureSpec],
        raw_columns: set[str],
    ) -> str:
        """Describe the execution order and dependencies as text.

        Args:
            specs: Feature specifications to execute.
            raw_columns: Columns available in the input dataframe.

        Returns:
            Human-readable multiline explanation.
        """
        plan = self.build_plan(specs, raw_columns=raw_columns)
        lines: list[str] = []

        for stage_idx, stage in enumerate(plan.stages, start=1):
            aliases = ", ".join(feature.spec.alias for feature in stage)
            lines.append(f"stage {stage_idx}: {aliases}")

        lines.append("")
        for stage in plan.stages:
            for feature in stage:
                dependencies = ", ".join(feature.dependencies) or "raw inputs only"
                lines.append(
                    f"{feature.spec.alias}: kind={feature.output_kind}, depends on {dependencies}"
                )

        return "\n".join(lines)

    def transform_one(
        self,
        sample: WindowSample,
        specs: list[FeatureSpec],
    ) -> dict[str, object]:
        """Compute scalar feature outputs for a single window sample.

        Args:
            sample: Window sample ending at one prediction time.
            specs: Feature specifications to execute.

        Returns:
            Row dictionary containing ``prediction_time`` and visible scalars.
        """
        plan = self.build_plan(specs, raw_columns=set(sample.df.columns))
        row, visible_columns, _ = self._transform_one_with_plan(sample, plan)
        keep = ["prediction_time"] + [
            column for column in visible_columns if column in row
        ]
        return {column: row[column] for column in keep}

    def transform_window(
        self,
        df_window: pl.DataFrame,
        specs: list[FeatureSpec],
        prediction_time: object | None = None,
    ) -> pl.DataFrame:
        """Compute one feature row directly from a raw dataframe window.

        This is a convenience API for callers that already build windows
        outside of ``WindowBuilder`` and want to pass each window straight into
        the feature extractors.

        Args:
            df_window: Raw dataframe slice available to the feature extractors.
            specs: Feature specifications to execute.
            prediction_time: Optional prediction/index value for the output
                row. Defaults to the latest value in ``time_col`` after sorting
                the window.

        Returns:
            One-row dataframe with ``prediction_time`` and visible scalar
            feature columns.

        Raises:
            ValueError: If ``df_window`` is empty and ``prediction_time`` is not
                provided, or if ``time_col`` is missing.
        """
        if prediction_time is None:
            if self.time_col not in df_window.columns:
                raise ValueError(f"Missing time column '{self.time_col}' in window")
            if df_window.is_empty():
                raise ValueError(
                    "prediction_time must be provided for an empty window"
                )
            prediction_time = df_window.sort(self.time_col)[self.time_col][-1]

        sample = WindowSample(prediction_time=prediction_time, df=df_window)
        return self.transform_many([sample], specs)

    def _transform_one_with_plan(
        self,
        sample: WindowSample,
        plan: ExecutionPlan,
    ) -> tuple[dict[str, object], list[str], list[str]]:
        """Compute one row using an already built execution plan.

        Args:
            sample: Window sample ending at one prediction time.
            plan: Precomputed execution plan.

        Returns:
            Tuple of row values, visible scalar columns, and internal columns.
        """
        context = FeatureContext(
            sample.df,
            time_col=self.time_col,
            index_mode=self.index_mode,
        )
        row: dict[str, object] = {"prediction_time": sample.prediction_time}
        visible_columns: list[str] = []
        internal_columns: list[str] = []

        for stage in plan.stages:
            for planned_feature in stage:
                self._apply_feature(
                    context=context,
                    row=row,
                    planned_feature=planned_feature,
                    visible_columns=visible_columns,
                    internal_columns=internal_columns,
                )

        return row, visible_columns, internal_columns

    def _apply_feature(
        self,
        context: FeatureContext,
        row: dict[str, object],
        planned_feature: PlannedFeature,
        visible_columns: list[str],
        internal_columns: list[str],
    ) -> None:
        """Execute one planned feature and merge it into context or output row.

        Args:
            context: Window-scoped context shared by all feature stages.
            row: Mutable scalar output row.
            planned_feature: Feature and dependency metadata to execute.
            visible_columns: Mutable list of scalar columns visible in output.
            internal_columns: Mutable list of scalar columns hidden from output.

        Raises:
            ValueError: If a scalar feature produces a duplicate output name.
        """
        spec = planned_feature.spec
        extractor = self.registry.get(spec.name)
        raw_result = extractor.extract(context, spec)
        result = self._normalize_result(spec, planned_feature, raw_result)

        if planned_feature.output_kind == "series":
            for alias, series in result.series.items():
                context.add_series(alias, series)
            return

        overlap = set(row).intersection(result.scalars)
        if overlap:
            repeated = ", ".join(sorted(overlap))
            raise ValueError(f"Duplicate feature names: {repeated}")

        row.update(result.scalars)

        output_columns = extractor.get_output_columns(spec, result)
        target = internal_columns if spec.internal else visible_columns
        for column in output_columns:
            if column not in target:
                target.append(column)

    def _normalize_result(
        self,
        spec: FeatureSpec,
        planned_feature: PlannedFeature,
        raw_result: object,
    ) -> FeatureResult:
        """Convert extractor return values to ``FeatureResult`` and validate shape.

        Args:
            spec: Feature specification being executed.
            planned_feature: Planned feature metadata with expected output kind.
            raw_result: Value returned by an extractor.

        Returns:
            Normalized feature result.

        Raises:
            TypeError: If the result does not match the planned output kind.
        """
        if isinstance(raw_result, FeatureResult):
            result = raw_result
        elif isinstance(raw_result, pl.Series):
            result = FeatureResult.from_series(spec.alias, raw_result)
        elif isinstance(raw_result, dict):
            result = FeatureResult.from_scalars(raw_result)
        else:
            result = FeatureResult.from_scalar(spec.alias, raw_result)

        if planned_feature.output_kind == "series":
            if not result.series:
                raise TypeError(
                    f"Feature '{spec.alias}' must return a series result"
                )
            if result.scalars:
                raise TypeError(
                    f"Feature '{spec.alias}' cannot mix scalar and series outputs"
                )
        else:
            if result.series:
                raise TypeError(
                    f"Feature '{spec.alias}' must return scalar outputs"
                )

        return result

    def transform_many(
        self,
        samples: list[WindowSample],
        specs: list[FeatureSpec],
    ) -> pl.DataFrame:
        """Compute feature rows for many window samples.

        Args:
            samples: Window samples to transform.
            specs: Feature specifications to execute for each sample.

        Returns:
            Dataframe with one row per sample and visible scalar feature columns.
        """
        if not samples:
            return pl.DataFrame({"prediction_time": []})

        plan = self.build_plan(specs, raw_columns=set(samples[0].df.columns))
        rows: list[dict[str, object]] = []
        visible_columns: list[str] = []
        internal_columns: list[str] = []

        for sample in samples:
            row, row_visible_columns, row_internal_columns = self._transform_one_with_plan(
                sample,
                plan,
            )
            rows.append(row)
            for column in row_visible_columns:
                if column not in visible_columns:
                    visible_columns.append(column)
            for column in row_internal_columns:
                if column not in internal_columns:
                    internal_columns.append(column)

        df = pl.DataFrame(rows)
        keep = ["prediction_time"] + [
            column for column in visible_columns if column in df.columns and column not in internal_columns
        ]
        return df.select(keep)
