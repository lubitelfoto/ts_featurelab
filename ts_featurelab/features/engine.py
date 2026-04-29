import polars as pl

from ts_featurelab.features.context import FeatureContext
from ts_featurelab.features.featurespec import FeatureSpec
from ts_featurelab.features.planner import ExecutionPlan, PlannedFeature, build_execution_plan
from ts_featurelab.features.registry import FeatureRegistry
from ts_featurelab.features.window import WindowSample


class FeatureEngine:
    def __init__(self, registry: FeatureRegistry, time_col: str = "date"):
        self.registry = registry
        self.time_col = time_col

    def build_plan(
        self,
        specs: list[FeatureSpec],
        raw_columns: set[str],
    ) -> ExecutionPlan:
        return build_execution_plan(specs, raw_columns=raw_columns, registry=self.registry)

    def explain(
        self,
        specs: list[FeatureSpec],
        raw_columns: set[str],
    ) -> str:
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
        plan = self.build_plan(specs, raw_columns=set(sample.df.columns))
        row, visible_columns, _ = self._transform_one_with_plan(sample, plan)
        keep = ["prediction_time"] + [
            column for column in visible_columns if column in row
        ]
        return {column: row[column] for column in keep}

    def _transform_one_with_plan(
        self,
        sample: WindowSample,
        plan: ExecutionPlan,
    ) -> tuple[dict[str, object], list[str], list[str]]:
        context = FeatureContext(sample.df, time_col=self.time_col)
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
        spec = planned_feature.spec
        extractor = self.registry.get(spec.name)
        result = extractor.extract(context, spec)

        if planned_feature.output_kind == "series":
            if not isinstance(result, pl.Series):
                raise TypeError(
                    f"Feature '{spec.alias}' must return a polars Series, got {type(result).__name__}"
                )
            context.add_series(spec.alias, result)
            return

        if not isinstance(result, dict):
            result = {spec.alias: result}

        overlap = set(row).intersection(result)
        if overlap:
            repeated = ", ".join(sorted(overlap))
            raise ValueError(f"Duplicate feature names: {repeated}")

        row.update(result)

        output_columns = extractor.get_output_columns(spec, result)
        target = internal_columns if spec.internal else visible_columns
        for column in output_columns:
            if column not in target:
                target.append(column)

    def transform_many(
        self,
        samples: list[WindowSample],
        specs: list[FeatureSpec],
    ) -> pl.DataFrame:
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
