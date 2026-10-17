# Student Monthly Expense Estimator

An educational Streamlit app that estimates the total monthly expenses reported
by students with similar survey responses. It uses a saved Ridge Regression
model trained for the course project.

## Important limitations

- The Kaggle listing describes the data as an open-source survey, but does not
  establish the original publisher, collection location or date, sampling
  method, or currency/scale.
- The dataset has only 99 usable target records and does not contain itemized
  expenses, student identifiers, income, or budget information.
- The model cannot identify a student's largest expense, determine whether a
  budget is adequate, or support monitoring of an individual student.
- On the held-out test set, the selected model's MAE was slightly lower than
  the baseline, while its RMSE and R-squared were worse. Treat its output as a
  classroom demonstration, not financial advice.
- The app does not store submitted form inputs or identify respondents.

Dataset listing: <https://www.kaggle.com/datasets/shariful07/nice-work-thanks-for-share>

## Run locally

Use Python 3.14, install dependencies, and run Streamlit from this directory:

```text
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Keep `selected_model.joblib` and `model_selection.json` next to `app.py` in
the repository root.

## Deploy with Streamlit Community Cloud

1. Put the contents of this folder in a dedicated GitHub repository. The app
   source and model artifact will be public if the repository is public.
2. At <https://share.streamlit.io>, select **Create app** and connect the
   repository, its default branch, and `app.py`.
3. In **Advanced settings**, select Python 3.14 to match the environment used
   to build and validate the saved model.
4. Deploy and test a sample submission. Share the generated `streamlit.app`
   URL only after confirming the app loads and clearly displays its limitations.

Do not add the paper, its authors' contact details, the raw dataset, account
credentials, or private tokens to this deployment repository.
