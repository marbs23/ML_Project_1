from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def build_threshold_table(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    false_negative_cost: float,
    false_positive_cost: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for threshold in np.linspace(0.05, 0.95, 91):
        prediction = (y_probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true, prediction)),
                "precision": float(precision_score(y_true, prediction, zero_division=0)),
                "recall": float(recall_score(y_true, prediction, zero_division=0)),
                "f1": float(f1_score(y_true, prediction, zero_division=0)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "total_cost": float(fp * false_positive_cost + fn * false_negative_cost),
            }
        )
    return pd.DataFrame(rows)


def select_best_threshold(threshold_table: pd.DataFrame) -> pd.Series:
    ordered = threshold_table.sort_values(["total_cost", "f1"], ascending=[True, False])
    return ordered.iloc[0]


def metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
    false_negative_cost: float,
    false_positive_cost: float,
) -> dict[str, float | int]:
    prediction = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_cost": float(fp * false_positive_cost + fn * false_negative_cost),
    }


def save_class_balance_plot(frame: pd.DataFrame, path: Path) -> None:
    counts = frame["is_critical"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["sano", "critico"], counts.values, color=["#4472c4", "#c0504d"])
    ax.set_title("Balance de clases")
    ax.set_ylabel("Numero de ciclos")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_rul_distribution_plot(frame: pd.DataFrame, path: Path, critical_rul_threshold: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(frame["rul"], bins=40, color="#4472c4", alpha=0.85)
    ax.axvline(critical_rul_threshold, color="#c0504d", linestyle="--", linewidth=2)
    ax.set_title("Distribucion de RUL en entrenamiento")
    ax.set_xlabel("Ciclos restantes")
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_threshold_curve_plot(threshold_table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(threshold_table["threshold"], threshold_table["total_cost"], color="#8064a2")
    ax.set_title("Costo total esperado por umbral")
    ax.set_xlabel("Umbral")
    ax.set_ylabel("Costo total")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confusion_matrix_plot(metrics: dict[str, float | int], path: Path, title: str) -> None:
    matrix = np.array(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1], labels=["sano", "critico"])
    ax.set_yticks([0, 1], labels=["sano", "critico"])

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

