# Tourism MLOps - Wellness Tourism Package Predictor

Predicts whether a customer will purchase the **Wellness Tourism Package**, so that the marketing team of "Visit with Us" can prioritise outreach. GitHub Actions automates data validation, data preparation, model training (XGBoost with MLflow experiment tracking) and the delivery of the trained model to a Streamlit app.

## Repository structure

```
tourism_project/
  data/tourism.csv                raw dataset
  model_building/
    data_register.py              validates the dataset columns and prints a summary
    prep.py                       cleans the data and creates the train/test split
    train.py                      tunes XGBoost, logs to MLflow, saves the best model and its metrics
  deployment/
    app.py                        Streamlit app
    requirements.txt              dependencies of the app
    best_tourism_model.joblib     trained model (committed by the pipeline)
    model_metrics.json            scores, confusion matrix, lift and feature importance of that model
  requirements.txt                dependencies of the pipeline
.github/workflows/pipeline.yml    CI/CD pipeline
```

## How the pipeline works

Every push to `main` (or a manual start from the Actions tab) runs three jobs in sequence:

1. **register-dataset** - validates that all expected columns are present.
2. **data-prep** - cleans the data and creates train/test splits, passed on as a workflow artifact.
3. **model-training** - tunes the model, logs every run to MLflow, and commits the best model and its metrics report to `tourism_project/deployment/`.

The Streamlit app loads the committed model, so the live app always serves the latest trained model.

## Run locally

```
pip install -r tourism_project/requirements.txt
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
mlflow ui --port 5000 &
python tourism_project/model_building/train.py
pip install -r tourism_project/deployment/requirements.txt
streamlit run tourism_project/deployment/app.py
```
