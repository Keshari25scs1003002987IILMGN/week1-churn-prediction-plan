"""
Stage 1-2: Problem Definition & Data Collection
-------------------------------------------------
Generates a synthetic, business-flavoured customer churn dataset so that the
full pipeline can run end-to-end without needing an external data source.

In a real deployment, this module would be replaced by a data-loading step
(database query / CSV export from the CRM / data warehouse extract).
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

RANDOM_STATE = 42


def generate_churn_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """Create a synthetic customer churn dataset with realistic-looking columns."""
    rng = np.random.default_rng(RANDOM_STATE)

    # Underlying signal (informative features) via make_classification,
    # then reshape into human-readable business columns.
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.75, 0.25],          # realistic class imbalance (~25% churn)
        flip_y=0.02,
        class_sep=1.1,
        random_state=RANDOM_STATE,
    )

    tenure_months = np.clip((X[:, 0] * 12 + 24), 0, 72).round().astype(int)
    monthly_charges = np.clip(X[:, 1] * 20 + 70, 18, 150).round(2)
    total_charges = (monthly_charges * np.maximum(tenure_months, 1) *
                      rng.uniform(0.9, 1.1, n_samples)).round(2)
    support_tickets = np.clip((X[:, 2] * 1.5 + 2), 0, 12).round().astype(int)
    satisfaction_score = np.clip((X[:, 3] * 1.2 + 6), 1, 10).round(1)

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples, p=[0.55, 0.25, 0.20]
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_samples
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n_samples, p=[0.35, 0.45, 0.20]
    )

    # Introduce a small amount of realistic missingness for the preprocessing
    # stage to handle.
    missing_mask = rng.random(n_samples) < 0.03
    total_charges = total_charges.astype(object)
    total_charges[missing_mask] = np.nan

    df = pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "support_tickets_last_3m": support_tickets,
        "satisfaction_score": satisfaction_score,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "churn": y,
    })

    return df


if __name__ == "__main__":
    import os
    df = generate_churn_dataset()
    os.makedirs("data", exist_ok=True)
    out_path = "data/churn_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["churn"].value_counts(normalize=True).rename("proportion"))
