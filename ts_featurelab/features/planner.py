from collections import deque
from dataclasses import dataclass

from ts_featurelab.features.base import FeatureOutputKind
from ts_featurelab.features.featurespec import FeatureSpec
from ts_featurelab.features.registry import FeatureRegistry


@dataclass(frozen=True)
class PlannedFeature:
    spec: FeatureSpec
    dependencies: tuple[str, ...]
    output_kind: FeatureOutputKind


@dataclass(frozen=True)
class ExecutionPlan:
    stages: tuple[tuple[PlannedFeature, ...], ...]

    @property
    def output_kinds(self) -> dict[str, FeatureOutputKind]:
        return {
            feature.spec.alias: feature.output_kind
            for stage in self.stages
            for feature in stage
        }

    @property
    def dependencies(self) -> dict[str, tuple[str, ...]]:
        return {
            feature.spec.alias: feature.dependencies
            for stage in self.stages
            for feature in stage
        }


def build_execution_plan(
    specs: list[FeatureSpec],
    raw_columns: set[str],
    registry: FeatureRegistry,
) -> ExecutionPlan:
    alias_to_spec: dict[str, FeatureSpec] = {}
    alias_order: dict[str, int] = {}
    for idx, spec in enumerate(specs):
        if spec.alias in raw_columns:
            raise ValueError(
                f"Feature alias '{spec.alias}' collides with an existing raw column"
            )
        if spec.alias in alias_to_spec:
            raise ValueError(f"Duplicate feature alias '{spec.alias}'")
        alias_to_spec[spec.alias] = spec
        alias_order[spec.alias] = idx

    output_kinds: dict[str, FeatureOutputKind] = {
        spec.alias: registry.get(spec.name).get_output_kind(spec)
        for spec in specs
    }

    dependency_map: dict[str, set[str]] = {}
    dependents: dict[str, set[str]] = {spec.alias: set() for spec in specs}
    indegree: dict[str, int] = {}

    for spec in specs:
        extractor = registry.get(spec.name)
        extractor.validate_spec(spec)
        dependencies = extractor.get_dependencies(spec)
        feature_dependencies: set[str] = set()

        for dependency in dependencies:
            if dependency in raw_columns:
                continue
            if dependency not in alias_to_spec:
                raise ValueError(
                    f"Feature '{spec.alias}' references unknown dependency '{dependency}'"
                )
            if output_kinds[dependency] != "series":
                raise ValueError(
                    f"Feature '{spec.alias}' cannot use scalar dependency '{dependency}' as a series input"
                )
            feature_dependencies.add(dependency)

        if feature_dependencies and spec.params.get("resample") is not None:
            raise ValueError(
                f"Feature '{spec.alias}' cannot resample derived dependencies; resample them while creating the upstream series"
            )

        dependency_map[spec.alias] = feature_dependencies
        indegree[spec.alias] = len(feature_dependencies)
        for dependency in feature_dependencies:
            dependents[dependency].add(spec.alias)

    ready = deque(
        sorted(
            (alias for alias, degree in indegree.items() if degree == 0),
            key=lambda alias: alias_order[alias],
        )
    )
    built = 0
    stages: list[tuple[PlannedFeature, ...]] = []

    while ready:
        stage_aliases = list(ready)
        ready.clear()
        stage_aliases.sort(key=lambda alias: alias_order[alias])

        stage: list[PlannedFeature] = []
        for alias in stage_aliases:
            spec = alias_to_spec[alias]
            stage.append(
                PlannedFeature(
                    spec=spec,
                    dependencies=tuple(sorted(dependency_map[alias])),
                    output_kind=output_kinds[alias],
                )
            )
            built += 1

        stages.append(tuple(stage))

        next_ready: set[str] = set()
        for alias in stage_aliases:
            for dependent in dependents[alias]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    next_ready.add(dependent)

        for alias in sorted(next_ready, key=lambda item: alias_order[item]):
            ready.append(alias)

    if built != len(specs):
        unresolved = sorted(alias for alias, degree in indegree.items() if degree > 0)
        joined = ", ".join(unresolved)
        raise ValueError(f"Cyclic feature dependencies detected: {joined}")

    return ExecutionPlan(stages=tuple(stages))
