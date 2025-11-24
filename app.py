import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Loan Default Prediction", layout="centered")
st.title("Loan Default Prediction System")
st.write("Predict customer loan default risk using behavioural data from your training pipeline.")

# Try to load artifacts
model = None
scaler = None
encoders = None
feature_cols = None

if os.path.exists('final_random_forest_model.pkl'):
    try:
        model = joblib.load('final_random_forest_model.pkl')
    except Exception as e:
        st.warning(f"Could not load model: {e}")
else:
    st.warning("Model file `final_random_forest_model.pkl` not found. Run the training notebook first to generate it.")

if os.path.exists('scaler.pkl') and os.path.exists('encoders.pkl') and os.path.exists('feature_columns.pkl'):
    try:
        scaler = joblib.load('scaler.pkl')
        encoders = joblib.load('encoders.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
    except Exception as e:
        st.warning(f"Could not load preprocessing artifacts: {e}")
else:
    st.info('Preprocessing artifacts `scaler.pkl`, `encoders.pkl`, or `feature_columns.pkl` not found. The app will still try a best-effort prediction.')

# --- Inputs (kept concise to match notebook) ---
borrower_id = st.text_input("Borrower ID (optional)")

col1, col2 = st.columns(2)
with col1:
    transaction_freq_open = st.number_input("Number of Open Credit Lines", min_value=0, step=1, value=1)
    total_acc = st.number_input("Total Number of Credit Accounts", min_value=0, step=1, value=5)
    delinq_2yrs = st.number_input("Delinquencies in Last 2 Years", min_value=0, step=1, value=0)
    mths_since_last_delinq = st.number_input("Months Since Last Delinquency (0 if none)", min_value=0, step=1, value=0)
    inq_last_6mths = st.number_input("Credit Inquiries in Last 6 Months", min_value=0, step=1, value=0)
with col2:
    revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, step=100.0, value=0.0)
    revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, step=0.1, value=0.0)
    annual_inc = st.number_input("Annual Income ($)", min_value=0.0, step=1000.0, value=30000.0)
    emp_length = st.number_input("Employment Length (years)", min_value=0, step=1, value=1)

loan_amnt = st.number_input("Loan Amount ($)", min_value=100.0, step=100.0, value=1000.0)

purpose = st.selectbox("Loan Purpose", ("debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "car", "medical", "vacation", "moving", "other"))

# Simple mapping fallback (keeps compatibility with notebook's purpose mapping)
purpose_mapping = {
    "debt_consolidation": 0, "credit_card": 1, "home_improvement": 2,
    "major_purchase": 3, "small_business": 4, "car": 5,
    "medical": 6, "vacation": 7, "moving": 8, "other": 9,
}

if st.button("Predict Default Risk"):
    # Build input vector
    if feature_cols is not None:
        row = {c: 0 for c in feature_cols}
        # map known input names to same-named columns if present
        mapping = {
            'transaction_freq_open': transaction_freq_open,
            'total_acc': total_acc,
            'delinq_2yrs': delinq_2yrs,
            'mths_since_last_delinq': mths_since_last_delinq,
            'revol_bal': revol_bal,
            'revol_util': revol_util,
            'inq_last_6mths': inq_last_6mths,
            'annual_inc': annual_inc,
            'emp_length': emp_length,
            'loan_amnt': loan_amnt,
            'purpose': purpose,
        }
        for k, v in mapping.items():
            if k in row:
                if k == 'purpose':
                    # encode purpose
                    if encoders and 'purpose' in encoders:
                        try:
                            enc_val = encoders['purpose'].transform([v])[0]
                        except Exception:
                            enc_val = purpose_mapping.get(v, 0)
                        row[k] = enc_val
                    else:
                        row[k] = purpose_mapping.get(v, 0)
                else:
                    row[k] = v
        df_input = pd.DataFrame([row], columns=feature_cols)
        if scaler is not None:
            X_input = scaler.transform(df_input)
        else:
            X_input = df_input.values
    else:
        # Fallback to the original simple ordering used in the notebook
        try:
            p_encoded = purpose_mapping.get(purpose, 0)
            X_input = np.array([
                transaction_freq_open,
                total_acc,
                delinq_2yrs,
                mths_since_last_delinq,
                revol_bal,
                revol_util,
                inq_last_6mths,
                annual_inc,
                emp_length,
                loan_amnt,
                p_encoded,
            ]).reshape(1, -1)
        except Exception as e:
            st.error(f"Could not build input vector: {e}")
            X_input = None

    if X_input is None:
        st.error("Unable to prepare input for prediction.")
    elif model is None:
        st.error("Model not loaded. Run training notebook to save `final_random_forest_model.pkl`.")
    else:
        try:
            pred = model.predict(X_input)[0]
            prob = None
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_input)[0]
            risk_mapping = {0: "Not Defaulted", 1: "Defaulted"}
            st.success(f"Prediction for Borrower {borrower_id if borrower_id else '(no ID)'}: {risk_mapping.get(pred, pred)}")
            if prob is not None:
                st.info(f"Prediction probabilities: {prob}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.write("Notes:")
st.write("- If the app warns about missing artifacts, open and run the training cells in `loan.ipynb` that save `final_random_forest_model.pkl`, `scaler.pkl`, `encoders.pkl`, and `feature_columns.pkl`.")
st.write("- For development, run `pip install -r requirements.txt` then `streamlit run app.py` in this folder.")
