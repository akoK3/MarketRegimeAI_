import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from models import get_models


# ── Feature List ──────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "ATR", "ATR_pct",
    "RSI", "RSI_lag1", "RSI_lag5", "RSI_lag10",
    "MACD", "Signal", "MACD_hist", "MACD_lag1", "MACD_lag5",
    "Close_to_SMA20", "Close_to_SMA50", "Close_to_SMA200",
    "SMA20_minus_SMA50", "Close_to_SMA20_lag1", "Close_to_SMA20_lag5",
    "Vol_10d", "Vol_20d",
    "Volume_ratio", "Volume_change",
    "ROC_5d", "ROC_10d", "ROC_20d",
    "ATR_pct_lag1",
]


def run_evaluation(features_path, n_splits=5):
    # ── Load Data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(features_path, parse_dates=True, index_col=0)

    X = df[FEATURE_COLS].copy()
    y = df["Regime"].copy()

    # ── Clean ─────────────────────────────────────────────────────────────────
    X = X.dropna()
    y = y.loc[X.index].dropna()
    X = X.loc[y.index]

    # ── Label Encoding for XGBoost ────────────────────────────────────────────
    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y), index=y.index)
    print("Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

    # ── Walk-forward ──────────────────────────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = get_models()
    results = {name: {"acc": [], "f1": [], "logloss": []} for name in models}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"{'='*50}")

        is_xgb = name == "XGBoost"

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

            if is_xgb:
                y_train = y_encoded.iloc[train_idx]
                y_test = y_encoded.iloc[test_idx]
            else:
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])

            if is_xgb:
                sample_weights = compute_sample_weight("balanced", y_train)
                pipe.fit(X_train, y_train, model__sample_weight=sample_weights)
            else:
                pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            ll = log_loss(y_test, y_prob)

            results[name]["acc"].append(acc)
            results[name]["f1"].append(f1)
            results[name]["logloss"].append(ll)

            print(f"Fold {fold} | Acc: {acc:.4f} | Macro F1: {f1:.4f} | Log Loss: {ll:.4f} | Test size: {len(y_test)}")

        print(f"\n--- {name} Summary ---")
        print(f"Accuracy  : {np.mean(results[name]['acc']):.4f} ± {np.std(results[name]['acc']):.4f}")
        print(f"Macro F1  : {np.mean(results[name]['f1']):.4f} ± {np.std(results[name]['f1']):.4f}")
        print(f"Log Loss  : {np.mean(results[name]['logloss']):.4f} ± {np.std(results[name]['logloss']):.4f}")

    # ── Final Evaluation on Last Fold ─────────────────────────────────────────
    print(f"\n{'='*50}")
    print("FINAL EVALUATION — Last Fold Only")
    print(f"{'='*50}")

    train_idx, test_idx = list(tscv.split(X))[-1]

    for name, model in models.items():
        is_xgb = name == "XGBoost"

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

        if is_xgb:
            y_train = y_encoded.iloc[train_idx]
            y_test = y_encoded.iloc[test_idx]
            labels = list(range(len(le.classes_)))
            target_names = list(le.classes_)
        else:
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            labels = None
            target_names = None

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        if is_xgb:
            sample_weights = compute_sample_weight("balanced", y_train)
            pipe.fit(X_train, y_train, model__sample_weight=sample_weights)
        else:
            pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        print(f"\n--- {name} ---")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred, labels=labels))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

    return results


if __name__ == "__main__":
    run_evaluation("data/processed/SPY_features.csv")