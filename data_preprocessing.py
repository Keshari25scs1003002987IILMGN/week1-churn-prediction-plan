"""
Stage 3: Data Preprocessing
----------------------------
Cleaning, feature engineering, encoding, and scaling — implemented as a
scikit-learn ColumnTransformer so the exact same fitted pipeline can be
reused at inference/deployment time (Section 8 of the plan).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges",
    "support_tickets_last_3m", "satisfaction_score",
    "avg_monthly_spend",  # engineered
]
CATEGORICAL_FEATURES = ["contract_type", "payment_method", "internet_service"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features described in Section 3.3 of the plan."""
    df = df.copy()
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["avg_monthly_spend"] = df["total_charges"] / df["tenure_months"].replace(0, 1)
    df["avg_monthly_spend"] = df["avg_monthly_spend"].fillna(df["monthly_charges"])
    return df


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build the cleaning + scaling + encoding pipeline (Section 3.1-3.3):
      - numeric: median imputation -> standard scaling
      - categorical: most-frequent imputation -> one-hot encoding
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def prepare_dataset(df: pd.DataFrame):
    """Return (X, y) ready to be split, with engineered features added."""
    df = engineer_features(df)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["churn"]
    return X, y
