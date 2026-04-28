from dataclasses import dataclass, field
from typing import Any

import polars as pl


@dataclass
class WindowSample:
    prediction_time: Any
    df: pl.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
