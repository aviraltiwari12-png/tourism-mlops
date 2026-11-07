"""Model training: tune an XGBoost classifier, track the experiment with MLflow,
and save the best model together with a model_metrics.json report.

Run from the repository root (after prep.py has produced the split files):
    python tourism_project/model_building/train.py

Environment variables (optional):
    MLFLOW_TRACKING_URI  MLflow server address (default http://localhost:5000)
    MODEL_DIR            folder for the saved model (default tourism_project/deployment)
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "tourism-wellness-package"
MODEL_DIR = Path(os.getenv("MODEL_DIR", "tourism_project/deployment"))
MODEL_PATH = MODEL_DIR / "best_tourism_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"   # read by the notebook and the Streamlit app

CLASSIFICATION_THRESHOLD = 0.5   # probability above which a customer is flagged as a likely buyer
RANDOM_STATE = 42
SCORING = "f1"                   # tuning metric: balances catching buyers (recall) and wasted calls (precision)

# Hyperparameter grid (2 x 3 x 2 x 2 = 24 combinations, 5-fold CV each).
# Exploration showed this dataset rewards deeper trees and more boosting rounds
# (shallow trees under-fit), so the grid covers depths 6-12 and 200-400 rounds.
PARAM_GRID = {
    "xgbclassifier__n_estimators": [200, 400],
    "xgbclassifier__max_depth": [6, 9, 12],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.6, 0.8],
}


def load_splits():
    """Load the train/test splits produced by prep.py."""
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").iloc[:, 0]
    ytest = pd.read_csv("ytest.csv").iloc[:, 0]
    return Xtrain, Xtest, ytrain, ytest


def build_pipeline(Xtrain, ytrain):
    """Preprocessing + XGBoost in a single scikit-learn pipeline."""
    numeric_cols = Xtrain.select_dtypes(include="number").columns.tolist()
    categorical_cols = Xtrain.select_dtypes(exclude="number").columns.tolist()

    preprocessor = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
        (make_pipeline(SimpleImputer(strategy="most_frequent"),
                       OneHotEncoder(handle_unknown="ignore")), categorical_cols),
    )

    # Class imbalance: weight buyers (minority) by the non-buyer : buyer ratio
    scale_pos_weight = float((ytrain == 0).sum() / (ytrain == 1).sum())
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,  # parallelism is handled by GridSearchCV
    )
    return make_pipeline(preprocessor, model), scale_pos_weight


def evaluate(model, X, y, prefix):
    """Return (metrics, predicted labels, buyer probabilities); metrics = accuracy / precision / recall / F1 / ROC-AUC."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= CLASSIFICATION_THRESHOLD).astype(int)
    return {
        f"{prefix}_accuracy": accuracy_score(y, pred),
        f"{prefix}_precision": precision_score(y, pred),
        f"{prefix}_recall": recall_score(y, pred),
        f"{prefix}_f1": f1_score(y, pred),
        f"{prefix}_roc_auc": roc_auc_score(y, proba),
    }, pred, proba


def clean_params(params):
    """Drop the pipeline step prefix so MLflow shows readable parameter names."""
    return {k.replace("xgbclassifier__", ""): v for k, v in params.items()}


def write_step_summary(report) -> None:
    """Inside GitHub Actions, show the training results on the run's summary page."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    train, test, cm = report["train_metrics"], report["test_metrics"], report["confusion_matrix_test"]
    lines = ["## Model training results", "", "| Metric | Train | Test |", "|---|---|---|"]
    lines += [f"| {name} | {train[name]:.3f} | {test[name]:.3f} |" for name in test]
    lines += [
        "",
        f"- Best parameters: `{report['best_params']}` (cross-validated F1 {report['best_cv_f1']:.3f})",
        f"- Confusion matrix on the test set: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}",
        f"- Contacting the top 20% of customers by score reaches **{report['capture_rate_test']['top_20pct']:.0%}** of all buyers",
        f"- Model and `model_metrics.json` are committed to `tourism_project/deployment/` by the next step",
    ]
    with open(path, "a") as handle:
        handle.write("\n".join(lines) + "\n")


def capture_rate(y, proba, top_fraction):
    """Share of all buyers found by contacting only the highest-scored `top_fraction` of customers."""
    y, proba = np.asarray(y), np.asarray(proba)
    k = max(1, int(round(len(y) * top_fraction)))
    top = np.argsort(-proba, kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def aggregate_importances(pipeline, feature_cols):
    """Feature importance per ORIGINAL column (one-hot columns are summed back to their source)."""
    names = pipeline.named_steps["columntransformer"].get_feature_names_out()
    names = [re.sub(r"^pipeline-\d+__", "", name) for name in names]
    values = pipeline.named_steps["xgbclassifier"].feature_importances_
    totals = {}
    for name, value in zip(names, values):
        owner = max((c for c in feature_cols if name == c or name.startswith(f"{c}_")), key=len)
        totals[owner] = totals.get(owner, 0.0) + float(value)
    return dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))


def build_report(best_model, grid_search, scale_pos_weight, Xtrain, Xtest, ytrain, ytest,
                 train_metrics, test_metrics, test_proba, test_pred):
    """Collect everything worth knowing about the trained model in one JSON-friendly dict."""
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(ytest, test_pred).ravel())
    importances = aggregate_importances(best_model, Xtrain.columns.tolist())
    strip = lambda metrics: {k.split("_", 1)[1]: round(float(v), 4) for k, v in metrics.items()}
    return {
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "library_versions": {"xgboost": getattr(xgb, "__version__", "unknown"),
                             "scikit-learn": sklearn.__version__, "pandas": pd.__version__},
        "n_train": int(len(Xtrain)),
        "n_test": int(len(Xtest)),
        "buyer_share_train": round(float(ytrain.mean()), 4),
        "buyer_share_test": round(float(ytest.mean()), 4),
        "best_params": clean_params(grid_search.best_params_),
        "best_cv_f1": round(float(grid_search.best_score_), 4),
        "scale_pos_weight": round(scale_pos_weight, 3),
        "threshold": CLASSIFICATION_THRESHOLD,
        "train_metrics": strip(train_metrics),
        "test_metrics": strip(test_metrics),
        "confusion_matrix_test": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "capture_rate_test": {f"top_{int(f * 100)}pct": round(capture_rate(ytest, test_proba, f), 4)
                              for f in (0.1, 0.2, 0.3)},
        "feature_importance": {k: round(v, 4) for k, v in list(importances.items())[:10]},
    }


def main() -> None:
    Xtrain, Xtest, ytrain, ytest = load_splits()
    print(f"Loaded splits - train: {Xtrain.shape}, test: {Xtest.shape}")

    pipeline, scale_pos_weight = build_pipeline(Xtrain, ytrain)
    grid_search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        scoring=SCORING,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        verbose=1,
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost_grid_search"):
        grid_search.fit(Xtrain, ytrain)

        # Log every tuned parameter combination as a nested (child) run
        results = grid_search.cv_results_
        for i, params in enumerate(results["params"]):
            with mlflow.start_run(run_name=f"combination_{i:02d}", nested=True):
                mlflow.log_params(clean_params(params))
                mlflow.log_metric(f"cv_mean_{SCORING}", float(results["mean_test_score"][i]))
                mlflow.log_metric(f"cv_std_{SCORING}", float(results["std_test_score"][i]))

        # Log the best configuration and its performance on train and test data
        best_model = grid_search.best_estimator_
        mlflow.log_params(clean_params(grid_search.best_params_))
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 3))
        mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD)
        mlflow.log_param("tuning_metric", SCORING)
        mlflow.log_metric(f"best_cv_{SCORING}", float(grid_search.best_score_))

        train_metrics, _, _ = evaluate(best_model, Xtrain, ytrain, "train")
        test_metrics, test_pred, test_proba = evaluate(best_model, Xtest, ytest, "test")
        mlflow.log_metrics({k: float(v) for k, v in {**train_metrics, **test_metrics}.items()})

        # Save the best model and its metrics report, and attach both to the MLflow run
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        report = build_report(best_model, grid_search, scale_pos_weight, Xtrain, Xtest, ytrain, ytest,
                              train_metrics, test_metrics, test_proba, test_pred)
        METRICS_PATH.write_text(json.dumps(report, indent=2))
        write_step_summary(report)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(METRICS_PATH))

    print(f"\nBest parameters : {clean_params(grid_search.best_params_)}")
    print(f"Best CV {SCORING}   : {grid_search.best_score_:.4f}")
    print("\nTrain metrics:", {k: round(v, 4) for k, v in train_metrics.items()})
    print("Test metrics :", {k: round(v, 4) for k, v in test_metrics.items()})
    print(f"\nClassification report (test set, threshold={CLASSIFICATION_THRESHOLD}):")
    print(classification_report(ytest, test_pred, target_names=["No purchase", "Purchase"]))
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics report saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
