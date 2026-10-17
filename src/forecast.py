"""Reproducible, time-ordered evaluation for the DSE coursework dataset.

The target is the next available market-session close of UIC.  Features are
observations from the current session.  Rows are sorted by date before the
target is shifted, and the final 20% of dates are held out for evaluation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "dse_market_data.csv"
RESULTS_DIR = ROOT / "results"

TARGET = "UIC"
FEATURES = [
    "DWX",
    "DLX",
    "ALAQ",
    "NIC",
    "SII",
    "SKI",
    "UIC",
    "gold",
    "oil",
    "AROP",
    "SYP",
    "TC",
    "NC",
    "TD",
    "ND",
    "^IRX",
]


def load_market_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the consolidated CSV and return one row per ascending market date."""
    frame = pd.read_csv(path)
    if "Date" not in frame.columns:
        raise ValueError("The source CSV must contain a 'Date' column.")

    frame["Date"] = pd.to_datetime(frame["Date"], format="mixed", errors="coerce")
    frame = frame.dropna(subset=["Date"]).copy()

    required = [column for column in FEATURES if column in frame.columns]
    missing = sorted(set(FEATURES) - set(required))
    if missing:
        raise ValueError(f"Source CSV is missing required features: {missing}")

    # Thousands separators occur in some numeric CSV fields (for example DLX).
    for column in FEATURES:
        if frame[column].dtype == "object":
            frame[column] = frame[column].astype(str).str.replace(",", "", regex=False)
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Preserve the last source row when a market date appears more than once.
    frame["_source_order"] = np.arange(len(frame))
    frame = frame.sort_values(["Date", "_source_order"], kind="stable")
    frame = frame.drop_duplicates(subset="Date", keep="last")
    frame = frame.drop(columns="_source_order").reset_index(drop=True)
    return frame


def make_next_session_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create current-session features and next-session UIC target/date."""
    design = frame[["Date", *FEATURES]].copy()
    design["target_date"] = design["Date"].shift(-1)
    design["target"] = design[TARGET].shift(-1)
    design = design.dropna(subset=[*FEATURES, "target", "target_date"]).reset_index(drop=True)
    x = design[FEATURES].copy()
    y = design["target"].copy()
    target_dates = design["target_date"].copy()
    return x, y, target_dates


def _models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=1000,
                early_stopping=True,
                random_state=42,
            ),
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        pass
    return models


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train models on the first 80% of dates and evaluate on the last 20%."""
    x, y, target_dates = make_next_session_dataset(frame)
    split_at = int(math.floor(len(x) * 0.8))
    if split_at < 2 or split_at >= len(x):
        raise ValueError("Not enough dated observations for an 80/20 split.")

    x_train, x_test = x.iloc[:split_at], x.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
    dates_test = target_dates.iloc[split_at:]

    predictions = pd.DataFrame(
        {
            "target_date": dates_test.to_numpy(),
            "actual_UIC": y_test.to_numpy(),
            "persistence_baseline": x_test[TARGET].to_numpy(),
        }
    )
    metric_rows = []

    def add_metrics(name: str, actual: pd.Series | np.ndarray, predicted: np.ndarray) -> None:
        metric_rows.append(
            {
                "model": name,
                "MAE": mean_absolute_error(actual, predicted),
                "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
                "R2": r2_score(actual, predicted),
            }
        )

    add_metrics("Persistence baseline", y_test, predictions["persistence_baseline"].to_numpy())
    for name, model in _models().items():
        model.fit(x_train, y_train)
        predicted = np.asarray(model.predict(x_test), dtype=float)
        predictions[name] = predicted
        add_metrics(name, y_test, predicted)

    metrics = pd.DataFrame(metric_rows).sort_values("RMSE").reset_index(drop=True)
    return metrics, predictions


def save_results(frame: pd.DataFrame, metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS_DIR / "chronological_holdout_metrics.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "chronological_holdout_predictions.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(frame["Date"], frame[TARGET], color="#486581", linewidth=1.7)
    plt.title("UIC historical closing price")
    plt.xlabel("Market date")
    plt.ylabel("UIC close")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "uic_historical_price.png", dpi=160)
    plt.close()

    columns = ["actual_UIC", "persistence_baseline"]
    columns.extend(name for name in ["Linear regression", "Random forest", "XGBoost", "MLP"] if name in predictions)
    plt.figure(figsize=(12, 6))
    plt.plot(predictions["target_date"], predictions["actual_UIC"], label="Actual", color="#243b53", linewidth=2)
    for name in columns[1:]:
        plt.plot(predictions["target_date"], predictions[name], label=name, linewidth=1.2, alpha=0.9)
    plt.title("UIC next-session forecasts — chronological holdout")
    plt.xlabel("Target market date")
    plt.ylabel("UIC close")
    plt.legend(ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "chronological_holdout_forecasts.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to the consolidated market CSV")
    args = parser.parse_args()

    frame = load_market_data(args.data)
    metrics, predictions = evaluate(frame)
    save_results(frame, metrics, predictions)
    print(f"Rows after date cleanup: {len(frame)}")
    print(f"Date range: {frame['Date'].min().date()} to {frame['Date'].max().date()}")
    print(f"Chronological holdout observations: {len(predictions)}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if "XGBoost" not in metrics["model"].tolist():
        print("Optional model not run: install requirements-optional.txt to enable XGBoost.")
    print(f"Results saved to: {RESULTS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
