from __future__ import annotations
from pathlib import Path
import pandas as pd


OPERATING_SETTING_COLUMNS = [f"operational_setting_{idx}" for idx in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_measurement_{idx}" for idx in range(1, 22)]
CMAPSS_COLUMNS = [
    "unit_number",
    "time_cycles",
    *OPERATING_SETTING_COLUMNS,
    *SENSOR_COLUMNS,
]


def read_cmapss_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)


def add_remaining_useful_life(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    max_cycle = result.groupby("unit_number")["time_cycles"].transform("max")
    result["rul"] = max_cycle - result["time_cycles"]
    return result


def add_binary_target(frame: pd.DataFrame, critical_rul_threshold: int) -> pd.DataFrame:
    result = frame.copy()
    result["is_critical"] = (result["rul"] <= critical_rul_threshold).astype(int)
    return result


def load_training_data(raw_data_dir: Path, dataset_id: str, critical_rul_threshold: int) -> pd.DataFrame:
    path = raw_data_dir / f"train_{dataset_id}.txt"
    frame = read_cmapss_file(path)
    frame = add_remaining_useful_life(frame)
    return add_binary_target(frame, critical_rul_threshold)


def load_test_data(raw_data_dir: Path, dataset_id: str) -> pd.DataFrame:
    path = raw_data_dir / f"test_{dataset_id}.txt"
    return read_cmapss_file(path)


def load_test_rul(raw_data_dir: Path, dataset_id: str) -> pd.DataFrame:
    path = raw_data_dir / f"RUL_{dataset_id}.txt"
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=["final_rul"])
    frame["unit_number"] = range(1, len(frame) + 1)
    return frame[["unit_number", "final_rul"]]


def add_test_remaining_useful_life(test_frame: pd.DataFrame, test_rul: pd.DataFrame) -> pd.DataFrame:
    result = test_frame.copy()
    observed_last_cycle = (
        result.groupby("unit_number", as_index=False)["time_cycles"]
        .max()
        .rename(columns={"time_cycles": "observed_last_cycle"})
    )
    result = result.merge(observed_last_cycle, on="unit_number", how="left")
    result = result.merge(test_rul, on="unit_number", how="left")
    result["rul"] = result["final_rul"] + result["observed_last_cycle"] - result["time_cycles"]
    return result.drop(columns=["observed_last_cycle", "final_rul"])


def summarize_dataset(frame: pd.DataFrame) -> dict[str, int | float]:
    positive_rate = float(frame["is_critical"].mean()) if "is_critical" in frame else 0.0
    return {
        "rows": int(len(frame)),
        "engines": int(frame["unit_number"].nunique()),
        "positive_rows": int(frame["is_critical"].sum()) if "is_critical" in frame else 0,
        "positive_rate": positive_rate,
    }

