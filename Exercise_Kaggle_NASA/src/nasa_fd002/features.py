from __future__ import annotations

import pandas as pd
from .data import OPERATING_SETTING_COLUMNS, SENSOR_COLUMNS


BASE_FEATURE_COLUMNS = [
    "time_cycles",
    *OPERATING_SETTING_COLUMNS,
    *SENSOR_COLUMNS,
]

def add_temporal_features(frame: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    result = frame.sort_values(["unit_number", "time_cycles"]).copy()

    for sensor in SENSOR_COLUMNS:
        grouped = result.groupby("unit_number")[sensor]
        result[f"{sensor}_delta"] = grouped.diff().fillna(0.0)
        result[f"{sensor}_rolling_mean_{rolling_window}"] = grouped.transform(
            lambda values: values.rolling(rolling_window, min_periods=1).mean()
        )
        result[f"{sensor}_rolling_std_{rolling_window}"] = grouped.transform(
            lambda values: values.rolling(rolling_window, min_periods=1).std()
        ).fillna(0.0)

    return result


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"unit_number", "rul", "is_critical"}
    return [column for column in frame.columns if column not in excluded]

