from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from ts_featurelab.features.featurespec import FeatureSpec

FeatureOutputKind = Literal["series", "scalar"]


@dataclass
class FeatureResult:
    """Container for scalar and series outputs produced by a feature extractor.

    Attributes:
        scalars: Mapping from output column name to scalar feature value.
        series: Mapping from output alias to a Polars series for derived inputs.
    """

    scalars: dict[str, object] = field(default_factory=dict)
    series: dict[str, pl.Series] = field(default_factory=dict)

    @classmethod
    def from_scalar(cls, alias: str, value: object) -> "FeatureResult":
        """Build a result with one scalar output.

        Args:
            alias: Output column name.
            value: Scalar value to store under ``alias``.

        Returns:
            A ``FeatureResult`` containing one scalar.
        """
        return cls(scalars={alias: value})

    @classmethod
    def from_scalars(cls, values: dict[str, object]) -> "FeatureResult":
        """Build a result with several scalar outputs.

        Args:
            values: Mapping from output column names to scalar values.

        Returns:
            A ``FeatureResult`` containing a shallow copy of ``values``.
        """
        return cls(scalars=dict(values))

    @classmethod
    def from_series(cls, alias: str, values: pl.Series) -> "FeatureResult":
        """Build a result with one series output.

        Args:
            alias: Name that should be assigned to the returned series.
            values: Series values produced by an extractor.

        Returns:
            A ``FeatureResult`` containing one series named ``alias``.
        """
        normalized = values if values.name == alias else pl.Series(alias, values)
        return cls(series={alias: normalized})


class FeatureExtractor(ABC):
    """Abstract base class for all feature extractors.

    Subclasses implement ``extract`` and may override dependency,
    validation, and output-shape hooks.

    Attributes:
        name: Registry name used in feature configuration.
        output_kind: Declares whether the extractor returns ``"scalar"`` or
            ``"series"`` outputs.
    """

    name: str
    output_kind: FeatureOutputKind = "scalar"

    @abstractmethod
    def extract(self, context, spec: FeatureSpec) -> FeatureResult:
        """Compute feature values for a single window.

        Args:
            context: Feature context with raw and derived window data.
            spec: Parsed feature specification.

        Returns:
            Feature values produced by this extractor.
        """
        raise NotImplementedError

    def get_dependencies(self, spec: FeatureSpec) -> set[str]:
        """Return raw or derived series names required by this feature.

        Args:
            spec: Parsed feature specification.

        Returns:
            Set containing ``spec.column`` when configured, otherwise empty.
        """
        return {spec.column} if spec.column else set()

    def get_output_kind(self, spec: FeatureSpec) -> FeatureOutputKind:
        """Return the declared output kind for a feature spec.

        Args:
            spec: Parsed feature specification.

        Returns:
            ``"scalar"`` or ``"series"``.
        """
        return self.output_kind

    def validate_spec(self, spec: FeatureSpec) -> None:
        """Validate extractor-specific options before execution.

        Args:
            spec: Parsed feature specification.

        Raises:
            ValueError: If the specification is invalid for the extractor.
        """
        return None

    def get_output_columns(
        self,
        spec: FeatureSpec,
        result: FeatureResult,
    ) -> list[str]:
        """Return visible output columns produced by an extraction result.

        Args:
            spec: Parsed feature specification.
            result: Normalized result returned by the extractor.

        Returns:
            List of output column names, falling back to ``spec.alias``.
        """
        if self.get_output_kind(spec) == "series":
            return list(result.series.keys()) or [spec.alias]
        return list(result.scalars.keys()) or [spec.alias]


class SingleColumnFeatureExtractor(FeatureExtractor):
    """Base class for extractors that require ``FeatureSpec.column``."""

    def require_column(self, spec: FeatureSpec) -> str:
        """Return the configured input column or raise a clear error.

        Args:
            spec: Parsed feature specification.

        Returns:
            Required input column name.

        Raises:
            ValueError: If ``spec.column`` is missing.
        """
        if not spec.column:
            raise ValueError(f"Feature '{spec.name}' requires 'column'")
        return spec.column
