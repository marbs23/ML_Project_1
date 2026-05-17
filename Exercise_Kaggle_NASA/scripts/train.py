from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nasa_fd002.config import TrainingConfig, get_project_paths
from nasa_fd002.pipeline import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--critical-rul-threshold", type=int, default=30)
    parser.add_argument("--false-negative-cost", type=float, default=10.0)
    parser.add_argument("--false-positive-cost", type=float, default=1.0)
    parser.add_argument("--rolling-window", type=int, default=5)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--cv-splits", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    config = TrainingConfig(
        critical_rul_threshold=args.critical_rul_threshold,
        false_negative_cost=args.false_negative_cost,
        false_positive_cost=args.false_positive_cost,
        rolling_window=args.rolling_window,
        validation_size=args.validation_size,
        cv_splits=args.cv_splits,
    )
    run_training(get_project_paths(PROJECT_ROOT), config)


if __name__ == "__main__":
    main()
