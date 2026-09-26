"""
End-to-end orchestration of the churn prediction pipeline described in
Week3_ML_Model_Development_Evaluation_Plan.docx.

Run with:  python main.py
"""

import os
import joblib
from sklearn.model_selection import train_test_split

from src.generate_data import generate_churn_dataset
from src.data_preprocessing import build_preprocessing_pipeline, prepare_dataset
from src.model_training import get_candidate_models, build_model_pipeline, tune_random_forest
from src.evaluate import evaluate_model, print_metrics, cross_validated_f1

RANDOM_STATE = 42


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ---- 1-2. Problem definition & data collection ----
    print("Step 1-2: Generating dataset...")
    df = generate_churn_dataset()
    df.to_csv("data/churn_data.csv", index=False)

    # ---- 3. Preprocessing ----
    print("Step 3: Preparing features...")
    X, y = prepare_dataset(df)
    preprocessor = build_preprocessing_pipeline()

    # ---- 4. Train / test split (stratified) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ---- 5. Baseline model comparison ----
    print("Step 5: Training baseline candidate models...")
    for name, estimator in get_candidate_models().items():
        pipeline = build_model_pipeline(preprocessor, estimator)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        print_metrics(metrics)

    # ---- 6. Hyperparameter tuning of the chosen model ----
    print("\nStep 6: Tuning Random Forest with GridSearchCV (5-fold CV, scoring=F1)...")
    search = tune_random_forest(preprocessor, X_train, y_train)
    best_model = search.best_estimator_
    print(f"  Best params: {search.best_params_}")

    # ---- 7. Final evaluation on held-out test set ----
    print("\nStep 7: Final evaluation of tuned Random Forest on the test set...")
    final_metrics = evaluate_model("Tuned Random Forest", best_model, X_test, y_test)
    print_metrics(final_metrics)

    cv_mean, cv_std = cross_validated_f1(best_model, X_train, y_train, cv=5)
    print(f"\n  Cross-validated F1 (train set, 5-fold): {cv_mean:.3f} +/- {cv_std:.3f}")

    # ---- Save artifacts for deployment (Section 8) ----
    joblib.dump(best_model, "models/churn_model.joblib")
    print("\nSaved tuned pipeline (preprocessing + model) -> models/churn_model.joblib")


if __name__ == "__main__":
    main()
