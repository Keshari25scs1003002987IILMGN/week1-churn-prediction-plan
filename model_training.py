"""
Stage 5-6: Model Selection, Training & Hyperparameter Tuning
--------------------------------------------------------------
Trains the candidate models named in the plan (Logistic Regression,
Decision Tree, Random Forest), then tunes the chosen model (Random Forest)
with GridSearchCV using 5-fold cross-validation, optimizing for F1-score
because the target classes are imbalanced (Section 6 / Section 7).
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def get_candidate_models():
    """Baseline candidate models described in Section 5.1 of the plan."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


def build_model_pipeline(preprocessor, estimator) -> Pipeline:
    """Chain preprocessing + estimator into a single reusable pipeline."""
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def tune_random_forest(preprocessor, X_train, y_train) -> GridSearchCV:
    """
    Hyperparameter tuning for the chosen model (Random Forest), as described
    in Section 6: GridSearchCV + 5-fold cross-validation, scored on F1.
    """
    pipeline = build_model_pipeline(
        preprocessor,
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    )

    param_grid = {
        "model__n_estimators": [150, 250, 350],
        "model__max_depth": [None, 8, 12],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search
