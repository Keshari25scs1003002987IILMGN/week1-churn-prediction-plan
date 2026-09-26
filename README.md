# Customer Churn Prediction — ML Development & Evaluation Plan

**Week 3 Task — Virtual Data Science Explorer Internship (YuvaIntern)**

This repository accompanies the submitted plan document
(`Week3_ML_Model_Development_Evaluation_Plan.docx`). It implements a working,
end-to-end reference pipeline for the plan's example problem: predicting
customer churn for a subscription-based telecom company.

## Problem

Binary classification — predict whether a customer will churn (`1`) or stay
(`0`), based on account, usage, and demographic features.

## Pipeline stages (mirrors the plan document)

| Stage | Script |
|---|---|
| 1–2. Problem definition & synthetic data generation | `src/generate_data.py` |
| 3. Data preprocessing (cleaning, scaling, encoding, feature engineering) | `src/data_preprocessing.py` |
| 5–6. Model selection, training & hyperparameter tuning | `src/model_training.py` |
| 7. Evaluation (accuracy, precision, recall, F1, ROC-AUC, cross-validation) | `src/evaluate.py` |
| End-to-end orchestration | `main.py` |

## Project structure

```
churn-prediction-ml-plan/
├── README.md
├── requirements.txt
├── main.py
├── data/                  # generated dataset lands here
├── models/                # trained model + scaler saved here
├── notebooks/             # optional exploration notebook
└── src/
    ├── __init__.py
    ├── generate_data.py
    ├── data_preprocessing.py
    ├── model_training.py
    └── evaluate.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python main.py
```

This will:
1. Generate a synthetic but realistic churn dataset (`data/churn_data.csv`).
2. Clean, scale, and encode the features.
3. Train Logistic Regression, Decision Tree, and Random Forest models.
4. Tune the Random Forest with `GridSearchCV` + 5-fold cross-validation.
5. Print accuracy, precision, recall, F1-score, and ROC-AUC for every model,
   plus the cross-validated F1 (mean ± std) for the tuned model.
6. Save the best model and scaler to `models/`.

## Week 4: Report Visualizations

`src/make_report_visuals.py` regenerates the five charts used in
`Week4_Data_Science_Report_and_Insights_Presentation_Plan.docx`
(churn by contract type, churn by tenure, feature importance, ROC curve,
confusion matrix) from the trained model, saving them to `report_assets/`:

```bash
python main.py                       # generate data + train + save model
python -m src.make_report_visuals    # generate report_assets/*.png
```

## Notes

- The dataset is synthetically generated (`sklearn.datasets.make_classification`
  plus hand-crafted business-style columns) so the project runs anywhere with
  no external data dependency, while still exercising every step described
  in the plan document.
- See the plan document for the full rationale behind each design choice
  (metric selection, cross-validation, handling class imbalance, deployment
  strategy, etc.).
