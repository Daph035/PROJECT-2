# Loan Default Prediction - Streamlit App

This repo contains a notebook (`loan.ipynb`) used to train a Random Forest model for predicting loan default, and a Streamlit application (`app.py`) to run predictions.

Quick steps to run locally

1. (Optional) Create a virtual environment and activate it.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the training notebook `loan.ipynb` in Jupyter (or run the training cells) to generate these artifacts in the project folder:
- `final_random_forest_model.pkl`
- `scaler.pkl`
- `encoders.pkl`
- `feature_columns.pkl`

4. Run the Streamlit app:

```powershell
streamlit run app.py
```

Notes and troubleshooting

- If opening `loan.ipynb` is slow, clear heavy outputs and re-run only the necessary cells. You can clear outputs using:

```powershell
jupyter nbconvert --clear-output --inplace "c:\\Users\\DELL\\Desktop\\project 2\\loan.ipynb"
```

- Reduce expensive operations during development (example: comment out `GridSearchCV` or run it on a smaller sample).
- Ensure the CSV dataset path in the notebook is valid: `c:\\Users\\DELL\\Downloads\\archive (2)\\loan.csv`.

Deployment

- Push this repository to GitHub and deploy via Streamlit Cloud (set the app file to `app.py`). Streamlit Cloud will install packages from `requirements.txt`.
- Alternatively, create a `Dockerfile` or `Procfile` for other hosting providers.
