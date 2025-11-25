"""Generate small, safe placeholder artifacts so the Streamlit app can run.

This script creates:
- `feature_columns.pkl` : ordered list of feature names
- `encoders.pkl` : dict with a LabelEncoder for 'purpose'
- `scaler.pkl` : StandardScaler fitted on dummy data
- `final_random_forest_model.pkl` : RandomForestClassifier trained on a tiny synthetic dataset

Run:
    python generate_artifacts.py

Note: These are placeholder artifacts for app testing. For production use, run `train.py` on the real dataset to produce proper artifacts.
"""

import joblib
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

OUT_DIR = '.'

feature_cols = [
    'transaction_freq_open',
    'total_acc',
    'delinq_2yrs',
    'mths_since_last_delinq',
    'revol_bal',
    'revol_util',
    'inq_last_6mths',
    'annual_inc',
    'emp_length',
    'loan_amnt',
    'purpose'
]

# Create and fit a simple LabelEncoder for 'purpose'
purpose_categories = ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "car", "medical", "vacation", "moving", "other"]
le = LabelEncoder()
le.fit(purpose_categories)
encoders = {'purpose': le}

# Create a simple scaler fitted on zeros (or random samples)
# We'll generate 10 synthetic rows to fit scaler and model
rng = np.random.RandomState(42)
X_dummy = rng.normal(loc=0.0, scale=1.0, size=(50, len(feature_cols)))
# ensure 'purpose' column is integer encoded
X_dummy[:, -1] = rng.randint(0, len(purpose_categories), size=(50,))

scaler = StandardScaler()
scaler.fit(X_dummy)

# Create a simple RandomForest trained on synthetic labels
y_dummy = rng.randint(0, 2, size=(50,))
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(scaler.transform(X_dummy), y_dummy)

# Save artifacts
os.makedirs(OUT_DIR, exist_ok=True)
joblib.dump(feature_cols, os.path.join(OUT_DIR, 'feature_columns.pkl'))
joblib.dump(encoders, os.path.join(OUT_DIR, 'encoders.pkl'))
joblib.dump(scaler, os.path.join(OUT_DIR, 'scaler.pkl'))
joblib.dump(model, os.path.join(OUT_DIR, 'final_random_forest_model.pkl'))

print("Placeholder artifacts created:")
print(" - feature_columns.pkl")
print(" - encoders.pkl")
print(" - scaler.pkl")
print(" - final_random_forest_model.pkl")
print("Run `python generate_artifacts.py` in the project folder to (re)create them.")
