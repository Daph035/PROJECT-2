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

import argparse


def generate(out_dir='.', seed=42, n_samples=50):
    os.makedirs(out_dir, exist_ok=True)

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
    purpose_categories = [
        "debt_consolidation", "credit_card", "home_improvement",
        "major_purchase", "small_business", "car", "medical",
        "vacation", "moving", "other"
    ]
    le = LabelEncoder()
    le.fit(purpose_categories)
    encoders = {'purpose': le}

    # Generate synthetic data to fit scaler and a small model
    rng = np.random.RandomState(seed)
    X_dummy = rng.normal(loc=0.0, scale=1.0, size=(n_samples, len(feature_cols)))
    # ensure 'purpose' column is integer encoded
    X_dummy[:, -1] = rng.randint(0, len(purpose_categories), size=(n_samples,))

    scaler = StandardScaler()
    scaler.fit(X_dummy)

    # Create a simple RandomForest trained on synthetic labels
    y_dummy = rng.randint(0, 2, size=(n_samples,))
    model = RandomForestClassifier(n_estimators=50, random_state=seed)
    model.fit(scaler.transform(X_dummy), y_dummy)

    # Save artifacts (use compression to reduce size)
    joblib.dump(feature_cols, os.path.join(out_dir, 'feature_columns.pkl'), compress=3)
    joblib.dump(encoders, os.path.join(out_dir, 'encoders.pkl'), compress=3)
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.pkl'), compress=3)
    joblib.dump(model, os.path.join(out_dir, 'final_random_forest_model.pkl'), compress=3)

    print("Placeholder artifacts created in:", os.path.abspath(out_dir))
    for name in ['feature_columns.pkl', 'encoders.pkl', 'scaler.pkl', 'final_random_forest_model.pkl']:
        print(f" - {name}")


def parse_args():
    p = argparse.ArgumentParser(description='Generate placeholder artifacts for Streamlit app')
    p.add_argument('--out', '-o', default='.', help='Output directory')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--samples', type=int, default=50, help='Number of synthetic samples')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        generate(out_dir=args.out, seed=args.seed, n_samples=args.samples)
    except Exception as e:
        print('Failed to generate artifacts:', e)
        raise
