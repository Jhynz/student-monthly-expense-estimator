"""Streamlit interface for the trained monthly-expense estimation model."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from joblib import load

from app_helpers import AGE_MAX, AGE_MIN, INPUT_CATEGORIES, build_input_frame

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "selected_model.joblib"
SELECTION_PATH = ROOT / "model_selection.json"

st.set_page_config(page_title="Student Expense Estimator", page_icon="📊")
st.title("Student Monthly Expense Estimator")
st.write(
    "Estimate a reported total from student survey responses."
)
st.warning(
    "Experimental only: the dataset has 99 target values, and the target's currency "
    "and scale are undocumented. The available fields cannot rank expense categories, "
    "assess budget adequacy, or support student monitoring."
)
st.caption(
    "Profile categories and yes/no answers are not itemized spending amounts."
)

if not MODEL_PATH.exists() or not SELECTION_PATH.exists():
    st.error(
        "The trained model artifacts are missing. Run the analysis module "
        "from the project folder first."
    )
    st.stop()

model = load(MODEL_PATH)
selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
validation_mae = selection["outer_cv_selected_model_mae_mean"]

with st.form("expense_inputs"):
    left, right = st.columns(2)
    with left:
        gender = st.selectbox("Gender category", INPUT_CATEGORIES["Gender"])
        age = st.number_input(
            "Age (years)", min_value=AGE_MIN, max_value=AGE_MAX, value=19, step=1
        )
        study_year = st.selectbox("Year of study", [1, 2, 3, 4], index=1)
        living = st.selectbox("Living arrangement", INPUT_CATEGORIES["Living"])
        scholarship = st.selectbox("Scholarship", INPUT_CATEGORIES["Scholarship"])
        part_time_job = st.selectbox("Part-time job", INPUT_CATEGORIES["Part_time_job"])
    with right:
        transporting = st.selectbox(
            "Transportation response", INPUT_CATEGORIES["Transporting"]
        )
        smoking = st.selectbox("Smoking response", INPUT_CATEGORIES["Smoking"])
        drinks = st.selectbox("Drinks response", INPUT_CATEGORIES["Drinks"])
        games = st.selectbox(
            "Games and hobbies response", INPUT_CATEGORIES["Games_&_Hobbies"]
        )
        cosmetics = st.selectbox(
            "Cosmetics and self-care response",
            INPUT_CATEGORIES["Cosmetics_&_Self-care"],
        )
        subscription = st.selectbox(
            "Monthly subscription", INPUT_CATEGORIES["Monthly_Subscription"]
        )
    submitted = st.form_submit_button("Estimate monthly expenses")

if submitted:
    try:
        features = build_input_frame(
            {
                "Gender": gender,
                "Age": age,
                "Study_year": study_year,
                "Living": living,
                "Scholarship": scholarship,
                "Part_time_job": part_time_job,
                "Transporting": transporting,
                "Smoking": smoking,
                "Drinks": drinks,
                "Games_&_Hobbies": games,
                "Cosmetics_&_Self-care": cosmetics,
                "Monthly_Subscription": subscription,
            }
        )
        estimate = float(model.predict(features)[0])
    except ValueError as error:
        st.error(str(error))
    else:
        st.metric("Estimated reported monthly expenses", f"{estimate:.1f} dataset units")
        st.caption(
            f"Selected model: {selection['selected_model']}. "
            f"Mean outer cross-validation MAE: {validation_mae:.1f} dataset units "
            "(mean validation error, not an individual prediction interval)."
        )
        st.info(
            "Not financial advice. Test-set R² was negative, and performance for new "
            "students has not been established."
        )

st.divider()
st.caption("Form inputs are used for prediction and are not stored.")
