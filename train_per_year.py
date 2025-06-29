import os
import pandas as pd
import numpy as np
npNaN = np.nan
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from ml_model import feature_names
from feature_engineering import add_advanced_features
from sentiment import fetch_news_sentiment
from utils import log

def load_clean_ohlcv(file_path):
    df = pd.read_csv(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def train_model_on_year(file_path, year):
    log(f"📅 Training model for {year}")
    
    df = load_clean_ohlcv(file_path)
    df = df.sort_values("datetime").reset_index(drop=True)

    # Add technical features
    df = add_advanced_features(df)

    # Label target: +1 if next candle closes higher, else 0
    df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)

    # Additional engineered features
    df["ema_cross"] = np.sign(df["ema_fast"] - df["ema_slow"])
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Fetch sentiment once for the year (avoid per-row API hits)
    sentiment_score = fetch_news_sentiment("EURUSD", verbose=False)
    df["sentiment"] = sentiment_score

    # Prepare training data
    X = df[feature_names()]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save model
    model_filename = f"ml_model_{year}.joblib"
    joblib.dump(model, model_filename)
    log(f"✅ Model saved: {model_filename}")
    log(f"📊 Accuracy: {acc:.2f} | F1 Score: {f1:.2f}")

if __name__ == "__main__":
    data_folder = "mt4_data_cleaned"
    files = sorted([f for f in os.listdir(data_folder) if f.endswith(".csv")])

    for file in files:
        year = file.split("_")[-1].split(".")[0]  # e.g., cleaned_EURUSD_2022.csv → 2022
        file_path = os.path.join(data_folder, file)
        train_model_on_year(file_path, year)
