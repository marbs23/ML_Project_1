from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



@dataclass(frozen=True)
class CandidateModel:
    name: str
    estimator: object
    param_grid: dict[str, list[object]]

def split_by_engine(
    frame: pd.DataFrame,
    validation_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=random_state,
    )
    train_index, validation_index = next(
        splitter.split(frame, groups=frame["unit_number"])
    )
    return train_index, validation_index


def build_candidate_models(random_state: int) -> list[CandidateModel]:
    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )
    random_forest = RandomForestClassifier(
        class_weight="balanced_subsample",
        n_estimators=250,
        n_jobs=-1,
        random_state=random_state,
    )
    return [
        CandidateModel(
            name="logistic_regression",
            estimator=logistic_regression,
            param_grid={"model__C": [0.1, 1.0, 10.0]},
        ),
        CandidateModel(
            name="random_forest",
            estimator=random_forest,
            param_grid={
                "max_depth": [None, 12],
                "min_samples_leaf": [1, 5],
            },
        ),
    ]


def fit_candidate_model(
    candidate: CandidateModel,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    cv_splits: int,
) -> GridSearchCV:
    search = GridSearchCV(
        estimator=candidate.estimator,
        param_grid=candidate.param_grid,
        scoring="f1",
        cv=GroupKFold(n_splits=cv_splits),
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train, groups=groups)
    return search

