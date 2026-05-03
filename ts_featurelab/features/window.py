from dataclasses import dataclass, field
from typing import Any

import polars as pl


@dataclass
class WindowSample:
    """Historical data window used to compute one feature row.

    Attributes:
        prediction_time: Timestamp at which the feature row is predicted.
        df: Raw dataframe slice available up to ``prediction_time``.
        metadata: Additional window details such as start/end timestamps.
    """

    prediction_time: Any
    df: pl.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
