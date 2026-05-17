from __future__ import annotations
import argparse
import sys
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nasa_fd002.data import read_cmapss_file
from nasa_fd002.features import add_temporal_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-bundle",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "model_bundle.joblib",
    )
    parser.add_argument(
        "--final-cycle-only",
        action="store_true",
        help="conserva solo la ultima observacion de cada motor",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model_bundle)
    frame = read_cmapss_file(args.input)
    engineered = add_temporal_features(frame, bundle["rolling_window"])
    probability = bundle["model"].predict_proba(engineered[bundle["feature_columns"]])[:, 1]

    output = engineered[["unit_number", "time_cycles"]].copy()
    output["critical_probability"] = probability
    output["prediction"] = (probability >= bundle["threshold"]).astype(int)

    if args.final_cycle_only:
        output = (
            output.sort_values(["unit_number", "time_cycles"])
            .groupby("unit_number", as_index=False)
            .tail(1)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
