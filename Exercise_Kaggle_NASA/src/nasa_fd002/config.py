from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    reports_dir: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    dataset_id: str = "FD002"
    critical_rul_threshold: int = 30
    validation_size: float = 0.20
    random_state: int = 42
    rolling_window: int = 5
    false_negative_cost: float = 10.0
    false_positive_cost: float = 1.0
    cv_splits: int = 3


def get_project_paths(root: Path | None = None) -> ProjectPaths:
    project_root = root or Path(__file__).resolve().parents[2]
    return ProjectPaths(
        root=project_root,
        raw_data_dir=project_root / "data" / "raw",
        processed_data_dir=project_root / "data" / "processed",
        reports_dir=project_root / "reports" / "generated",
        artifacts_dir=project_root / "artifacts",
    )

