"""
03_modeling.py
Pemodelan prediksi churn e-commerce.

Model yang dibandingkan:
- Logistic Regression (class_weight balanced, fitur numerik diskalakan)
- Random Forest (class_weight balanced_subsample)
- Gradient Boosting versi scikit-learn (HistGradientBoosting, class_weight balanced)
- XGBoost bila library tersedia di lingkungan (opsional, dilewati bila tidak ada)

Pembagian data: train/test 80/20 dengan stratifikasi label churn.
Kelas tidak seimbang (churn sekitar 16,8 persen), jadi model terbaik dipilih
berdasarkan PR-AUC pada data uji, bukan akurasi.

Output:
- outputs/model_metrics.csv      : tabel metrik semua model
- outputs/model_summary.json     : ringkasan model terbaik
- outputs/test_predictions.csv   : prediksi probabilitas churn pada data uji (model terbaik)
- outputs/feature_importance.csv : permutation importance fitur pada model terbaik
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "processed" / "churn_clean.csv"
OUT_METRICS = BASE / "outputs" / "model_metrics.csv"
OUT_SUMMARY = BASE / "outputs" / "model_summary.json"
OUT_PRED = BASE / "outputs" / "test_predictions.csv"
OUT_IMPORTANCE = BASE / "outputs" / "feature_importance.csv"

RANDOM_STATE = 42


def build_preprocessor(numeric_cols: list, categorical_cols: list, scale_numeric: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(num_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    target = "Churn"
    drop_cols = [target, "CustomerID"]
    X = df.drop(columns=drop_cols)
    y = df[target]

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {}

    lr = Pipeline([
        ("prep", build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    candidates["Logistic Regression"] = lr

    rf = Pipeline([
        ("prep", build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)),
        ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                         random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    candidates["Random Forest"] = rf

    gb = Pipeline([
        ("prep", build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)),
        ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE, class_weight="balanced")),
    ])
    candidates["Gradient Boosting"] = gb

    try:
        from xgboost import XGBClassifier

        xgb = Pipeline([
            ("prep", build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)),
            ("model", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                    subsample=0.9, colsample_bytree=0.9,
                                    scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
                                    eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)),
        ])
        candidates["XGBoost"] = xgb
        print("XGBoost tersedia dan ikut dibandingkan.")
    except ImportError:
        print("XGBoost tidak tersedia di lingkungan ini, dilewati.")

    rows = []
    fitted = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(np.asarray(y_test), proba)
        metrics = {"model": name, **metrics}
        rows.append(metrics)
        fitted[name] = (pipe, proba)
        print(name, metrics)

    metrics_df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(OUT_METRICS, index=False)
    print("\nTabel metrik tersimpan di", OUT_METRICS)
    print(metrics_df.to_string(index=False))

    best_name = str(metrics_df.iloc[0]["model"])
    best_pipe, best_proba = fitted[best_name]
    print("\nModel terbaik (PR-AUC tertinggi):", best_name)

    # Simpan prediksi data uji dari model terbaik untuk tahap rekomendasi.
    pred_df = pd.DataFrame({
        "CustomerID": df.loc[X_test.index, "CustomerID"].values,
        "churn_aktual": np.asarray(y_test),
        "probabilitas_churn": np.round(best_proba, 4),
        "model": best_name,
    })
    pred_df.to_csv(OUT_PRED, index=False)

    # Permutation importance pada data uji.
    perm = permutation_importance(best_pipe, X_test, y_test, n_repeats=10,
                                  random_state=RANDOM_STATE, scoring="average_precision")
    imp_df = pd.DataFrame({
        "fitur": X.columns,
        "importance_mean": np.round(perm.importances_mean, 4),
        "importance_std": np.round(perm.importances_std, 4),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    imp_df.to_csv(OUT_IMPORTANCE, index=False)
    print("\nFitur terpenting (permutation importance):")
    print(imp_df.head(10).to_string(index=False))

    summary = {
        "model_terbaik": best_name,
        "kriteria_pemilihan": "PR-AUC tertinggi pada data uji (kelas churn minoritas, 16,8 persen)",
        "ukuran_train": int(len(X_train)),
        "ukuran_test": int(len(X_test)),
        "metrik_model_terbaik": {k: float(metrics_df.iloc[0][k]) for k in
                                  ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]},
        "fitur_terpenting": imp_df.head(8)["fitur"].tolist(),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nRingkasan tersimpan di", OUT_SUMMARY)


if __name__ == "__main__":
    main()
