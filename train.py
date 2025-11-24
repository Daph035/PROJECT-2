import argparse
import logging
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_data(path):
    logging.info(f"Loading data from: {path}")
    df = pd.read_csv(path)
    logging.info(f"Data shape: {df.shape}")
    return df


def preprocess(df):
    logging.info("Starting preprocessing")
    # Drop columns with too many missing values
    df = df.dropna(thresh=len(df) * 0.5, axis=1)

    # Fill numeric missing with median
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill categorical missing with mode per column
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            try:
                df[col] = df[col].fillna(df[col].mode()[0])
            except Exception:
                df[col] = df[col].fillna("")

    # Remove irrelevant columns if present
    drop_cols = [
        'id', 'member_id', 'url', 'title', 'zip_code', 'emp_title',
        'policy_code', 'desc', 'issue_d'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Label encode categorical columns and save encoders
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Handle outliers using IQR (drop rows with outliers)
    try:
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
        df = df[mask]
    except Exception:
        logging.info("Skipping outlier removal due to error in numeric calculation")

    # Feature columns (exclude target)
    if 'loan_status' not in df.columns:
        raise KeyError("Target column 'loan_status' not found in dataframe")

    feature_cols = df.drop('loan_status', axis=1).columns.tolist()

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    y = df['loan_status'].values

    logging.info(f"Preprocessing completed. Features: {len(feature_cols)}")
    return X, y, scaler, encoders, feature_cols


def train_model(X_train, y_train, use_grid=False):
    if use_grid:
        logging.info("Running GridSearchCV (this may take a while)")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid = GridSearchCV(rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid.fit(X_train, y_train)
        logging.info(f"Grid search best params: {grid.best_params_}")
        return grid.best_estimator_, grid
    else:
        logging.info("Training RandomForestClassifier with default params")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        return rf, None


def save_artifacts(model, scaler, encoders, feature_cols, out_dir='.'):
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'final_random_forest_model.pkl')
    scaler_path = os.path.join(out_dir, 'scaler.pkl')
    enc_path = os.path.join(out_dir, 'encoders.pkl')
    feat_path = os.path.join(out_dir, 'feature_columns.pkl')

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoders, enc_path)
    joblib.dump(feature_cols, feat_path)

    logging.info(f"Saved model to {model_path}")
    logging.info(f"Saved scaler to {scaler_path}")
    logging.info(f"Saved encoders to {enc_path}")
    logging.info(f"Saved feature columns to {feat_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train RF model and save artifacts for Streamlit app')
    parser.add_argument('--data', type=str, default=r'c:\\Users\\DELL\\Downloads\\archive (2)\\loan.csv',
                        help='Path to loan CSV data')
    parser.add_argument('--out', type=str, default='.', help='Output directory for artifacts')
    parser.add_argument('--grid', action='store_true', help='Run GridSearchCV for hyperparameter tuning')
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    df = load_data(args.data)
    X, y, scaler, encoders, feature_cols = preprocess(df)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model, grid_obj = train_model(X_train, y_train, use_grid=args.grid)

    # evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {acc * 100:.2f}%")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

    # save
    save_artifacts(model, scaler, encoders, feature_cols, out_dir=args.out)


if __name__ == '__main__':
    main()
