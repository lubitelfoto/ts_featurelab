import polars as pl

from features.context import FeatureContext
from features.featurespec import FeatureSpec
from features.registry import FeatureRegistry
from features.window import WindowSample


class FeatureEngine:
    def __init__(self, registry: FeatureRegistry, time_col: str = "date"):
        self.registry = registry
        self.time_col = time_col

    def transform_one(
        self,
        sample: WindowSample,
        specs: list[FeatureSpec],
    ) -> dict[str, object]:
        context = FeatureContext(sample.df, time_col=self.time_col)
        row: dict[str, object] = {"prediction_time": sample.prediction_time}

        for spec in specs:
            extractor = self.registry.get(spec.name)
            features = extractor.extract(context, spec)
            overlap = set(row).intersection(features)
            if overlap:
                repeated = ", ".join(sorted(overlap))
                raise ValueError(f"Duplicate feature names: {repeated}")
            row.update(features)

        return row

    def transform_many(
        self,
        samples: list[WindowSample],
        specs: list[FeatureSpec],
    ) -> pl.DataFrame:
        rows = [self.transform_one(sample, specs) for sample in samples]
        return pl.DataFrame(rows) if rows else pl.DataFrame({"prediction_time": []})
