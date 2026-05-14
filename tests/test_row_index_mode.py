from datetime import datetime, timedelta
import unittest

import polars as pl

from ts_featurelab.features import (
    FeatureEngine,
    FeatureSpec,
    TargetBuilder,
    TargetSpec,
    WindowBuilder,
    build_default_registry,
)


class RowIndexModeTests(unittest.TestCase):
    def test_time_mode_backward_compatible(self) -> None:
        start = datetime(2024, 1, 1)
        df = pl.DataFrame(
            {
                "date": [start + timedelta(hours=idx) for idx in range(5)],
                "value": [float(idx) for idx in range(5)],
            }
        )

        samples = WindowBuilder(
            time_col="date",
            window_size="2h",
            step="1h",
            min_history="2h",
        ).transform(df)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].metadata["index_mode"], "time")
        self.assertEqual(samples[0].df["value"].to_list(), [1.0, 2.0])

    def test_row_mode_builds_expected_number_of_windows(self) -> None:
        df = self._depth_df(12)

        samples = WindowBuilder(
            time_col="depth",
            index_mode="row",
            window_size=4,
            step=3,
            min_history=4,
        ).transform(df)

        self.assertEqual(
            [sample.metadata["prediction_row_idx"] for sample in samples],
            [3, 6, 9],
        )

    def test_row_mode_window_contains_previous_n_rows(self) -> None:
        df = self._depth_df(8)

        sample = WindowBuilder(
            time_col="depth",
            index_mode="row",
            window_size=4,
            step=2,
            min_history=4,
        ).transform(df)[1]

        self.assertEqual(sample.metadata["window_start_idx"], 2)
        self.assertEqual(sample.metadata["window_end_idx"], 5)
        self.assertEqual(sample.df["sample"].to_list(), [2, 3, 4, 5])

    def test_row_mode_prediction_time_equals_index_value(self) -> None:
        df = self._depth_df(6)

        sample = WindowBuilder(
            time_col="depth",
            index_mode="row",
            window_size=3,
            step=1,
        ).transform(df)[0]

        self.assertEqual(sample.prediction_time, 1001.0)
        self.assertEqual(sample.metadata["prediction_index_value"], 1001.0)

    def test_depth_column_can_be_float(self) -> None:
        df = pl.DataFrame(
            {
                "depth": [1502.5, 1500.0, 1501.25, 1503.75],
                "gamma_ray": [3.0, 0.0, 1.0, 4.0],
            }
        )

        sample = WindowBuilder(
            time_col="depth",
            index_mode="row",
            window_size=2,
            step=1,
        ).transform(df)[0]

        self.assertEqual(sample.df["depth"].to_list(), [1500.0, 1501.25])
        self.assertEqual(sample.prediction_time, 1501.25)

    def test_row_mode_rejects_duration_window_size(self) -> None:
        df = self._depth_df(5)

        with self.assertRaisesRegex(ValueError, "window_size.*positive integer"):
            WindowBuilder(
                time_col="depth",
                index_mode="row",
                window_size="24h",
                step=1,
            ).transform(df)

    def test_row_mode_rejects_resample(self) -> None:
        df = self._depth_df(5)
        samples = WindowBuilder(
            time_col="depth",
            index_mode="row",
            window_size=3,
            step=1,
        ).transform(df)
        engine = FeatureEngine(
            build_default_registry(),
            time_col="depth",
            index_mode="row",
        )

        with self.assertRaisesRegex(ValueError, "resample.*index_mode='time'"):
            engine.transform_many(
                samples,
                [
                    FeatureSpec(
                        name="mean",
                        column="gamma_ray",
                        alias="gamma_ray_mean",
                        params={"resample": "1h"},
                    )
                ],
            )

    def test_row_mode_extracts_raw_window_features(self) -> None:
        df = self._depth_df(5)
        samples = WindowBuilder(
            time_col="depth",
            index_mode="row",
            window_size=3,
            step=1,
        ).transform(df)
        engine = FeatureEngine(
            build_default_registry(),
            time_col="depth",
            index_mode="row",
        )

        features_df = engine.transform_many(
            samples[:1],
            [
                FeatureSpec(name="mean", column="gamma_ray", alias="gamma_mean"),
                FeatureSpec(name="std", column="gamma_ray", alias="gamma_std"),
                FeatureSpec(name="trend", column="gamma_ray", alias="gamma_trend"),
                FeatureSpec(
                    name="value_at_lag",
                    column="gamma_ray",
                    alias="gamma_lag_1",
                    params={"lag": 1},
                ),
            ],
        )

        self.assertEqual(features_df["prediction_time"].to_list(), [1001.0])
        self.assertEqual(features_df["gamma_mean"].to_list(), [2.0])
        self.assertEqual(features_df["gamma_std"].to_list(), [2.0])
        self.assertEqual(features_df["gamma_trend"].to_list(), [4.0])
        self.assertEqual(features_df["gamma_lag_1"].to_list(), [2.0])

    def test_row_mode_target_horizon_by_rows(self) -> None:
        df = self._depth_df(8).with_columns(
            formation=pl.Series(["a", "b", "c", "d", "e", "f", "g", "h"])
        )
        samples = WindowBuilder(
            time_col="depth",
            index_mode="row",
            window_size=4,
            step=3,
            min_history=4,
        ).transform(df)

        target_df = TargetBuilder(time_col="depth").transform(
            df,
            samples,
            TargetSpec(
                column="formation",
                alias="target_formation",
                task="classification",
                gap=1,
                horizon=2,
                agg="last",
            ),
        )

        self.assertEqual(target_df["prediction_time"].to_list(), [1001.5])
        self.assertEqual(target_df["target_formation"].to_list(), ["g"])

    @staticmethod
    def _depth_df(rows: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "depth": [1000.0 + idx * 0.5 for idx in range(rows)],
                "sample": list(range(rows)),
                "gamma_ray": [float(idx * 2) for idx in range(rows)],
            }
        )


if __name__ == "__main__":
    unittest.main()
