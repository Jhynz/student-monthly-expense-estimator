"""Shared data and model utilities for the student-expense project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Monthly_expenses_$"
NUMERIC_FEATURES = ["Age", "Study_year"]
FEATURES = [
    "Gender",
    "Age",
    "Study_year",
    "Living",
    "Scholarship",
    "Part_time_job",
    "Transporting",
    "Smoking",
    "Drinks",
    "Games_&_Hobbies",
    "Cosmetics_&_Self-care",
    "Monthly_Subscription",
]
CATEGORICAL_FEATURES = [feature for feature in FEATURES if feature not in NUMERIC_FEATURES]
MODEL_NAMES = ("Linear Regression", "Ridge Regression", "Random Forest")


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and minimally normalize the supplied CSV without changing its source file."""
    data = pd.read_csv(path)
    missing_columns = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    data = data.loc[:, FEATURES + [TARGET]].copy()
    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].str.strip().replace("", pd.NA)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data["Age"] = pd.to_numeric(data["Age"], errors="coerce")
    data["Study_year"] = pd.to_numeric(data["Study_year"], errors="coerce")
    return data


def make_preprocessor() -> ColumnTransformer:
    """Create fold-local imputers and encoding to prevent preprocessing leakage."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def make_pipeline(model: str, *, random_state: int = 42) -> Pipeline:
    """Build one of the three preregistered traditional regression pipelines."""
    estimators = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(
            random_state=random_state,
            n_jobs=1,
        ),
    }
    if model not in estimators:
        raise ValueError(f"Unknown model {model!r}; choose from {', '.join(MODEL_NAMES)}.")
    return Pipeline(
        [
            ("preprocessor", make_preprocessor()),
            ("regressor", estimators[model]),
        ]
    )


def model_parameter_grid(model: str) -> dict[str, list[object]]:
    """Return the fixed search space used only within grouped training folds."""
    if model == "Linear Regression":
        return {}
    if model == "Ridge Regression":
        return {"regressor__alpha": [0.1, 1.0, 10.0, 100.0]}
    if model == "Random Forest":
        return {
            "regressor__n_estimators": [200, 400],
            "regressor__max_depth": [None, 4, 8],
            "regressor__min_samples_leaf": [2, 4],
        }
    raise ValueError(f"Unknown model {model!r}; choose from {', '.join(MODEL_NAMES)}.")
