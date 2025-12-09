# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json
import mlflow
import mlflow.sklearn

DATA_PATH = "data/diabetes.csv"
MODEL_PATH = "model.pkl"
METRICS_PATH = "metrics.json"

def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Start MLflow experiment
    mlflow.set_experiment("pima-diabetes-experiment")

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("Accuracy:", accuracy)

        # 3. Log params and metrics to MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)

        # 4. Log model to MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        # 5. Save locally as well (for serving)
        joblib.dump(model, MODEL_PATH)

        # 6. Save metrics externally for DVC tracking
        with open(METRICS_PATH, "w") as f:
            json.dump({"accuracy": accuracy}, f)

if __name__ == "__main__":
    main()
