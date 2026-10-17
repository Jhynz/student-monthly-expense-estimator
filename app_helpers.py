"""Validated input construction for the Streamlit estimator."""

from __future__ import annotations

import pandas as pd

from expenses import FEATURES

AGE_MIN = 17
AGE_MAX = 25
INPUT_CATEGORIES = {
    "Gender": ("Female", "Male"),
    "Living": ("Home", "Hostel"),
    "Scholarship": ("No", "Yes"),
    "Part_time_job": ("No", "Yes"),
    "Transporting": ("No", "Car", "Motorcycle"),
    "Smoking": ("No", "Yes"),
    "Drinks": ("No", "Yes"),
    "Games_&_Hobbies": ("No", "Yes"),
    "Cosmetics_&_Self-care": ("No", "Yes"),
    "Monthly_Subscription": ("No", "Yes"),
}


def build_input_frame(values: dict[str, object]) -> pd.DataFrame:
    """Validate one form response and return features in the model's expected order."""
    missing = [feature for feature in FEATURES if feature not in values]
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}.")

    age = values["Age"]
    if isinstance(age, bool) or not isinstance(age, (int, float)) or int(age) != age:
        raise ValueError("Age must be a whole number.")
    if not AGE_MIN <= age <= AGE_MAX:
        raise ValueError(f"Age must be between {AGE_MIN} and {AGE_MAX} years.")

    study_year = values["Study_year"]
    if isinstance(study_year, bool) or study_year not in (1, 2, 3, 4):
        raise ValueError("Year of study must be 1, 2, 3, or 4.")

    invalid_categories = [
        field
        for field, allowed in INPUT_CATEGORIES.items()
        if values[field] not in allowed
    ]
    if invalid_categories:
        raise ValueError(
            "Invalid selection for: " + ", ".join(invalid_categories) + "."
        )

    return pd.DataFrame([{field: values[field] for field in FEATURES}], columns=FEATURES)
