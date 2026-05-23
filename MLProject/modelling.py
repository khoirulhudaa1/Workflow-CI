from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "breast_cancer_preprocessing" / "breast_cancer_processed.csv"
MLRUNS_DIR = BASE_DIR / "mlruns"
TARGET_COLUMN = "diagnosis"
SPLIT_COLUMN = "split"
RANDOM_STATE = 42


def load_processed_dataset(data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset preprocessing tidak ditemukan: {data_path}")

    data = pd.read_csv(data_path)
    train_data = data[data[SPLIT_COLUMN] == "train"].drop(columns=[SPLIT_COLUMN])
    test_data = data[data[SPLIT_COLUMN] == "test"].drop(columns=[SPLIT_COLUMN])

    x_train = train_data.drop(columns=[TARGET_COLUMN])
    y_train = train_data[TARGET_COLUMN]
    x_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN]
    return x_train, x_test, y_train, y_test


def evaluate_model(model: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1_score": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
    }


def main() -> None:
    x_train, x_test, y_train, y_test = load_processed_dataset()

    if "MLFLOW_RUN_ID" not in os.environ and "MLFLOW_TRACKING_URI" not in os.environ:
        MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())

    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment("Breast Cancer Classification - Autolog")

    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    run_context = (
        mlflow.start_run()
        if "MLFLOW_RUN_ID" in os.environ
        else mlflow.start_run(run_name="random_forest_autolog")
    )

    with run_context as run:
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.log_param("dataset", DATA_PATH.name)
        mlflow.set_tag("submission_stage", "kriteria_2_basic")

        print(f"MLflow run_id: {run.info.run_id}")
        print(f"MLflow tracking URI: {MLRUNS_DIR.as_uri()}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
