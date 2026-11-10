"""
04_rekomendasi.py
Segmentasi risiko churn dan simulasi sederhana dampak penanganan komplain.

Cara kerja:
1. Model terbaik dari tahap pemodelan (dipilih ulang di sini dengan resep yang sama,
   yaitu model dengan PR-AUC tertinggi pada data uji) dipakai untuk menghasilkan
   probabilitas risiko semua pelanggan dengan metode out-of-fold 5 fold:
   probabilitas tiap pelanggan berasal dari model yang TIDAK dilatih pada data
   pelanggan itu sendiri. Cara ini menghindari probabilitas ekstrem 0 atau 1 akibat
   model menghafal data latihnya, sehingga segmentasi mencerminkan risiko yang
   lebih realistis.
2. Pelanggan dibagi menjadi tiga segmen berdasarkan probabilitas churn:
   - Risiko tinggi : probabilitas 0,60 atau lebih
   - Risiko sedang : probabilitas 0,30 sampai di bawah 0,60
   - Risiko rendah : probabilitas di bawah 0,30
   Churn rate aktual tiap segmen ikut dilaporkan sebagai validasi segmentasi.
3. Simulasi penanganan komplain cepat (skenario batas atas dan skenario moderat):
   kelompok pelanggan yang pernah komplain punya churn rate jauh lebih tinggi.
   Skenario batas atas mengasumsikan penanganan komplain yang cepat dan tuntas mampu
   menekan churn rate kelompok komplain sampai sama dengan kelompok tanpa komplain.
   Skenario moderat mengasumsikan hanya separuh selisihnya yang bisa ditutup.
   Keduanya asumsi eksploratif, bukan jaminan hasil.

Output:
- outputs/segmentasi_risiko.csv
- outputs/segmentasi_stats.json
- outputs/simulasi_retensi.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "processed" / "churn_clean.csv"
SUMMARY_PATH = BASE / "outputs" / "model_summary.json"
OUT_SEG = BASE / "outputs" / "segmentasi_risiko.csv"
OUT_SEG_STATS = BASE / "outputs" / "segmentasi_stats.json"
OUT_SIM = BASE / "outputs" / "simulasi_retensi.json"

RANDOM_STATE = 42


def build_model(name: str, numeric_cols: list, categorical_cols: list) -> Pipeline:
    def prep(scale: bool) -> ColumnTransformer:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            num_steps.append(("scaler", StandardScaler()))
        return ColumnTransformer([
            ("num", Pipeline(num_steps), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ])

    if name == "Logistic Regression":
        return Pipeline([("prep", prep(True)),
                         ("model", LogisticRegression(max_iter=1000, class_weight="balanced",
                                                      random_state=RANDOM_STATE))])
    if name == "Random Forest":
        return Pipeline([("prep", prep(False)),
                         ("model", RandomForestClassifier(n_estimators=300,
                                                          class_weight="balanced_subsample",
                                                          random_state=RANDOM_STATE, n_jobs=-1))])
    if name == "Gradient Boosting":
        return Pipeline([("prep", prep(False)),
                         ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE,
                                                                  class_weight="balanced"))])
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return Pipeline([("prep", prep(False)),
                         ("model", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                                 subsample=0.9, colsample_bytree=0.9,
                                                 eval_metric="logloss",
                                                 random_state=RANDOM_STATE, n_jobs=-1))])
    raise ValueError(f"Model tidak dikenal: {name}")


def segmen(prob: float) -> str:
    if prob >= 0.60:
        return "Tinggi"
    if prob >= 0.30:
        return "Sedang"
    return "Rendah"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    best_name = summary["model_terbaik"]
    print("Model terbaik dari tahap pemodelan:", best_name)

    X = df.drop(columns=["Churn", "CustomerID"])
    y = df["Churn"]
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    model = build_model(best_name, numeric_cols, categorical_cols)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    print("Probabilitas risiko dihitung dengan out-of-fold 5 fold.")

    seg_df = pd.DataFrame({
        "CustomerID": df["CustomerID"],
        "probabilitas_churn": np.round(proba, 4),
        "segmen_risiko": [segmen(p) for p in proba],
        "churn_aktual": df["Churn"],
        "Complain": df["Complain"],
        "Tenure": df["Tenure"],
        "DaySinceLastOrder": df["DaySinceLastOrder"],
    })
    seg_df = seg_df.sort_values("probabilitas_churn", ascending=False).reset_index(drop=True)
    seg_df.to_csv(OUT_SEG, index=False)

    seg_stats = {}
    for name in ["Tinggi", "Sedang", "Rendah"]:
        sub = seg_df[seg_df["segmen_risiko"] == name]
        seg_stats[name] = {
            "jumlah_pelanggan": int(len(sub)),
            "churn_aktual": int(sub["churn_aktual"].sum()),
            "churn_rate_aktual_persen": round(float(sub["churn_aktual"].mean() * 100), 2) if len(sub) else 0.0,
        }
    total_churn = int(seg_df["churn_aktual"].sum())
    captured = seg_stats["Tinggi"]["churn_aktual"]
    seg_stats["porsi_churn_di_segmen_tinggi_persen"] = round(captured / total_churn * 100, 2) if total_churn else 0.0
    captured_ts = captured + seg_stats["Sedang"]["churn_aktual"]
    seg_stats["porsi_churn_di_tinggi_sedang_persen"] = round(captured_ts / total_churn * 100, 2) if total_churn else 0.0
    seg_stats["metode_probabilitas"] = "Out-of-fold 5 fold (cross_val_predict) dengan model terbaik"
    OUT_SEG_STATS.write_text(json.dumps(seg_stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Segmentasi risiko:")
    print(json.dumps(seg_stats, indent=2, ensure_ascii=False))

    # Simulasi penanganan komplain.
    grup_komplain = df[df["Complain"] == 1]
    grup_tanpa = df[df["Complain"] == 0]
    rate_komplain = float(grup_komplain["Churn"].mean())
    rate_tanpa = float(grup_tanpa["Churn"].mean())
    n_komplain = int(len(grup_komplain))
    selisih = rate_komplain - rate_tanpa
    simulasi = {
        "jumlah_pelanggan_pernah_komplain": n_komplain,
        "churn_rate_komplain_persen": round(rate_komplain * 100, 2),
        "churn_rate_tanpa_komplain_persen": round(rate_tanpa * 100, 2),
        "skenario_batas_atas": {
            "asumsi": "Penanganan komplain cepat dan tuntas menekan churn rate kelompok komplain "
                      "sampai sama dengan kelompok tanpa komplain.",
            "perkiraan_pelanggan_terselamatkan": int(round(n_komplain * selisih)),
        },
        "skenario_moderat": {
            "asumsi": "Penanganan komplain yang lebih baik hanya menutup separuh selisih churn rate.",
            "perkiraan_pelanggan_terselamatkan": int(round(n_komplain * selisih * 0.5)),
        },
        "catatan": "Simulasi eksploratif berbasis selisih churn rate observasional, bukan hasil "
                   "eksperimen. Angka ini batas perkiraan kasar untuk prioritas tindakan, bukan janji hasil.",
    }
    OUT_SIM.write_text(json.dumps(simulasi, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSimulasi retensi:")
    print(json.dumps(simulasi, indent=2, ensure_ascii=False))
    print("\nSegmentasi tersimpan di", OUT_SEG)


if __name__ == "__main__":
    main()
