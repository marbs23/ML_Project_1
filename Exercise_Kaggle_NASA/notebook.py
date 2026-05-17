from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nasa_fd002.config import TrainingConfig, get_project_paths
from nasa_fd002.pipeline import run_training


if __name__ == "__main__":
    run_training(get_project_paths(PROJECT_ROOT), TrainingConfig())
