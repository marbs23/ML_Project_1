from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import ProjectPaths, TrainingConfig
from .data import (
    add_binary_target,
    add_test_remaining_useful_life,
    load_test_data,
    load_test_rul,
    load_training_data,
    summarize_dataset,
)
from .evaluation import (
    build_threshold_table,
    metrics_at_threshold,
    save_class_balance_plot,
    save_confusion_matrix_plot,
    save_rul_distribution_plot,
    save_threshold_curve_plot,
    select_best_threshold,
)
from .features import add_temporal_features, get_feature_columns
from .modeling import build_candidate_models, fit_candidate_model, split_by_engine


def _ensure_directories(paths: ProjectPaths) -> None:
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _last_cycle_per_engine(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values(["unit_number", "time_cycles"])
        .groupby("unit_number", as_index=False)
        .tail(1)
        .copy()
    )


def run_training(paths: ProjectPaths, config: TrainingConfig) -> dict[str, object]:
    _ensure_directories(paths)

    training_frame = load_training_data(
        raw_data_dir=paths.raw_data_dir,
        dataset_id=config.dataset_id,
        critical_rul_threshold=config.critical_rul_threshold,
    )
    engineered_frame = add_temporal_features(training_frame, config.rolling_window)
    feature_columns = get_feature_columns(engineered_frame)

    train_index, validation_index = split_by_engine(
        engineered_frame,
        validation_size=config.validation_size,
        random_state=config.random_state,
    )
    train_frame = engineered_frame.iloc[train_index].copy()
    validation_frame = engineered_frame.iloc[validation_index].copy()

    x_train = train_frame[feature_columns]
    y_train = train_frame["is_critical"]
    x_validation = validation_frame[feature_columns]
    y_validation = validation_frame["is_critical"]

    model_rows: list[dict[str, object]] = []
    fitted_models: dict[str, object] = {}

    for candidate in build_candidate_models(config.random_state):
        search = fit_candidate_model(
            candidate=candidate,
            x_train=x_train,
            y_train=y_train,
            groups=train_frame["unit_number"],
            cv_splits=config.cv_splits,
        )
        fitted_models[candidate.name] = search.best_estimator_
        validation_probability = search.best_estimator_.predict_proba(x_validation)[:, 1]
        threshold_table = build_threshold_table(
            y_true=y_validation,
            y_probability=validation_probability,
            false_negative_cost=config.false_negative_cost,
            false_positive_cost=config.false_positive_cost,
        )
        best_threshold = select_best_threshold(threshold_table)
        metrics = metrics_at_threshold(
            y_true=y_validation,
            y_probability=validation_probability,
            threshold=float(best_threshold["threshold"]),
            false_negative_cost=config.false_negative_cost,
            false_positive_cost=config.false_positive_cost,
        )
        model_rows.append(
            {
                "model_name": candidate.name,
                "cv_best_f1": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                **metrics,
            }
        )

    comparison = pd.DataFrame(model_rows).sort_values(
        ["total_cost", "f1"],
        ascending=[True, False],
    )
    comparison.to_csv(paths.reports_dir / "model_comparison.csv", index=False)
    best_model_row = comparison.iloc[0]
    best_model_name = str(best_model_row["model_name"])
    best_model = fitted_models[best_model_name]

    validation_probability = best_model.predict_proba(x_validation)[:, 1]
    threshold_table = build_threshold_table(
        y_true=y_validation,
        y_probability=validation_probability,
        false_negative_cost=config.false_negative_cost,
        false_positive_cost=config.false_positive_cost,
    )
    threshold_table.to_csv(paths.reports_dir / "threshold_curve.csv", index=False)
    best_threshold = float(select_best_threshold(threshold_table)["threshold"])
    validation_metrics = metrics_at_threshold(
        y_true=y_validation,
        y_probability=validation_probability,
        threshold=best_threshold,
        false_negative_cost=config.false_negative_cost,
        false_positive_cost=config.false_positive_cost,
    )
    validation_predictions = validation_frame[
        ["unit_number", "time_cycles", "rul", "is_critical"]
    ].copy()
    validation_predictions["critical_probability"] = validation_probability
    validation_predictions["prediction"] = (
        validation_probability >= best_threshold
    ).astype(int)
    validation_predictions.to_csv(
        paths.reports_dir / "validation_predictions.csv",
        index=False,
    )
    validation_last_cycle = _last_cycle_per_engine(validation_predictions)
    validation_last_cycle.to_csv(
        paths.reports_dir / "validation_final_cycle_predictions.csv",
        index=False,
    )

    save_class_balance_plot(training_frame, paths.reports_dir / "class_balance.png")
    save_rul_distribution_plot(
        training_frame,
        paths.reports_dir / "rul_distribution.png",
        config.critical_rul_threshold,
    )
    save_threshold_curve_plot(threshold_table, paths.reports_dir / "threshold_cost_curve.png")
    save_confusion_matrix_plot(
        validation_metrics,
        paths.reports_dir / "validation_confusion_matrix.png",
        "Matriz de confusion - validacion",
    )

    final_model = best_model.fit(engineered_frame[feature_columns], engineered_frame["is_critical"])
    model_bundle = {
        "model": final_model,
        "feature_columns": feature_columns,
        "rolling_window": config.rolling_window,
        "threshold": best_threshold,
        "config": config,
    }
    joblib.dump(model_bundle, paths.artifacts_dir / "model_bundle.joblib")

    output: dict[str, object] = {
        "dataset_summary": summarize_dataset(training_frame),
        "best_model_name": best_model_name,
        "validation_metrics": validation_metrics,
        "feature_count": len(feature_columns),
    }
    _write_json(paths.reports_dir / "validation_metrics.json", output)

    test_path = paths.raw_data_dir / f"test_{config.dataset_id}.txt"
    test_rul_path = paths.raw_data_dir / f"RUL_{config.dataset_id}.txt"
    if test_path.exists():
        test_frame = load_test_data(paths.raw_data_dir, config.dataset_id)
        engineered_test = add_temporal_features(test_frame, config.rolling_window)
        test_probability = final_model.predict_proba(engineered_test[feature_columns])[:, 1]
        test_predictions = engineered_test[["unit_number", "time_cycles"]].copy()
        test_predictions["critical_probability"] = test_probability
        test_predictions["prediction"] = (test_probability >= best_threshold).astype(int)

        if test_rul_path.exists():
            test_with_rul = add_test_remaining_useful_life(
                test_frame,
                load_test_rul(paths.raw_data_dir, config.dataset_id),
            )
            test_with_target = add_binary_target(
                test_with_rul,
                config.critical_rul_threshold,
            )
            test_predictions["rul"] = test_with_target["rul"]
            test_predictions["is_critical"] = test_with_target["is_critical"]
            test_metrics = metrics_at_threshold(
                y_true=test_with_target["is_critical"],
                y_probability=test_probability,
                threshold=best_threshold,
                false_negative_cost=config.false_negative_cost,
                false_positive_cost=config.false_positive_cost,
            )
            output["test_metrics"] = test_metrics
            _write_json(paths.reports_dir / "test_metrics.json", test_metrics)
            save_confusion_matrix_plot(
                test_metrics,
                paths.reports_dir / "test_confusion_matrix.png",
                "Matriz de confusion - test",
            )

        test_predictions.to_csv(paths.reports_dir / "test_predictions.csv", index=False)
        test_last_cycle = _last_cycle_per_engine(test_predictions)
        test_last_cycle.to_csv(
            paths.reports_dir / "test_final_cycle_predictions.csv",
            index=False,
        )
        if {"is_critical", "critical_probability"}.issubset(test_last_cycle.columns):
            test_final_cycle_metrics = metrics_at_threshold(
                y_true=test_last_cycle["is_critical"],
                y_probability=test_last_cycle["critical_probability"].to_numpy(),
                threshold=best_threshold,
                false_negative_cost=config.false_negative_cost,
                false_positive_cost=config.false_positive_cost,
            )
            output["test_final_cycle_metrics"] = test_final_cycle_metrics
            _write_json(
                paths.reports_dir / "test_final_cycle_metrics.json",
                test_final_cycle_metrics,
            )
            save_confusion_matrix_plot(
                test_final_cycle_metrics,
                paths.reports_dir / "test_final_cycle_confusion_matrix.png",
                "Matriz de confusion - ultimo ciclo",
            )

    return output
