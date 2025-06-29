# retrain_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ml_model import MODEL_PATH, TRAINING_LOG, feature_names
from utils import log

def retrain_model(min_samples=20, test_size=0.2, random_state=42, verbose=True):
    try:
        if not pd.io.common.file_exists(TRAINING_LOG):
            log("❌ Training log not found.")
            return

        df = pd.read_csv(TRAINING_LOG)
        if len(df) < min_samples:
            log(f"⚠️ Not enough samples to retrain: {len(df)} found, {min_samples} required.")
            return

        X = df[feature_names()]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        joblib.dump(model, MODEL_PATH)

        log("✅ Model retrained and saved successfully.")
        if verbose:
            log(f"📊 Accuracy: {acc:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}")
            log(f"🔍 Confusion Matrix:\n{cm}")

    except Exception as e:
        log(f"❌ Model retraining failed: {e}")

if __name__ == "__main__":
    retrain_model()
