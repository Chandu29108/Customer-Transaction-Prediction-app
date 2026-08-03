"""
Train and compare Logistic Regression, XGBoost, and an MLP classifier on
the Santander customer transaction dataset, then persist the best model
plus the fitted scaler/PCA and evaluation artifacts.

Run from the project root:
    python -m src.train
"""

import json
import logging

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_preprocessing import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(random_state=config.RANDOM_STATE, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model


def train_mlp(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 32),
            activation="relu",
            solver="adam",
            max_iter=200,
            early_stopping=True,
            random_state=config.RANDOM_STATE,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(name, model, X_train, y_train, X_test, y_test, results_dir=config.RESULTS_DIR):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    logger.info(
        "%s -> train acc: %.4f | test acc: %.4f | test F1: %.4f",
        name, train_acc, test_acc, test_f1,
    )

    report_path = results_dir / f"{name}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"{name} - Training Classification Report\n")
        f.write(classification_report(y_train, y_train_pred))
        f.write(f"\n{name} - Testing Classification Report\n")
        f.write(classification_report(y_test, y_test_pred))

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(results_dir / f"{name}_confusion_matrix.png", dpi=150)
    plt.close()

    return {
        "model_name": name,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_f1": train_f1,
        "test_f1": test_f1,
    }


def main():
    data = run_pipeline()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    trainers = {
        "logistic_regression": train_logistic_regression,
        "xgboost": train_xgboost,
        "mlp": train_mlp,
    }

    all_metrics = []
    trained_models = {}

    for name, train_fn in trainers.items():
        logger.info("Training %s ...", name)
        model = train_fn(X_train, y_train)
        trained_models[name] = model
        metrics = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        all_metrics.append(metrics)

    best = max(all_metrics, key=lambda m: m["test_f1"])
    best_name = best["model_name"]
    best_model = trained_models[best_name]
    logger.info("Best model: %s (test F1 = %.4f)", best_name, best["test_f1"])

    for name, model in trained_models.items():
        joblib.dump(model, config.MODELS_DIR / f"{name}.pkl")
    # Always save the champion under a fixed name so app.py doesn't need
    # to know in advance which model type won.
    joblib.dump(best_model, config.MODELS_DIR / "best_model.pkl")
    joblib.dump(data["scaler"], config.MODELS_DIR / "robust_scaler.pkl")
    joblib.dump(data["pca"], config.MODELS_DIR / "pca.pkl")

    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)

    with open(config.RESULTS_DIR / "best_model.json", "w") as f:
        json.dump({"best_model": best_name, "test_f1": best["test_f1"], "test_accuracy": best["test_accuracy"]}, f, indent=2)

    logger.info("Saved models to %s", config.MODELS_DIR)
    logger.info("Saved results to %s", config.RESULTS_DIR)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
