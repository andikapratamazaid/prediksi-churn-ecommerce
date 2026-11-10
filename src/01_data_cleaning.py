"""
01_data_cleaning.py
Membersihkan dataset "Ecommerce Customer Churn Analysis and Prediction" (Kaggle).

Masalah data yang ditangani:
1. Nilai hilang (missing value) pada 7 kolom numerik, sekitar 251 sampai 307 baris per kolom.
   Ditangani dengan imputasi median per kolom.
2. Outlier salah ketik pada WarehouseToHome: nilai 126 dan 127 km (2 baris), padahal
   nilai wajarnya maksimal 36 km. Diperbaiki dengan mengurangi 100 (126 menjadi 26, 127 menjadi 27).
3. Label kategori ganda dengan arti sama:
   - PreferredLoginDevice: "Phone" sama dengan "Mobile Phone"
   - PreferredPaymentMode: "CC" sama dengan "Credit Card", "COD" sama dengan "Cash on Delivery"
   - PreferedOrderCat: "Mobile" sama dengan "Mobile Phone"

Input : data/raw/E Commerce Dataset.xlsx (sheet "E Comm")
Output: data/processed/churn_clean.csv dan outputs/cleaning_report.json
"""

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW_PATH = BASE / "data" / "raw" / "E Commerce Dataset.xlsx"
OUT_CSV = BASE / "data" / "processed" / "churn_clean.csv"
OUT_REPORT = BASE / "outputs" / "cleaning_report.json"

NUMERIC_COLS = [
    "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
    "Complain", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
    "DaySinceLastOrder", "CashbackAmount",
]

CATEGORY_FIXES = {
    "PreferredLoginDevice": {"Phone": "Mobile Phone"},
    "PreferredPaymentMode": {"CC": "Credit Card", "COD": "Cash on Delivery"},
    "PreferedOrderCat": {"Mobile": "Mobile Phone"},
}


def main() -> None:
    df = pd.read_excel(RAW_PATH, sheet_name="E Comm")
    report = {"baris_mentah": int(len(df)), "kolom": int(df.shape[1])}

    # 1. Perbaiki outlier salah ketik WarehouseToHome (nilai di atas 100 km tidak masuk akal
    #    untuk jarak gudang ke rumah pada dataset ini, nilai wajar maksimal 36 km).
    outlier_mask = df["WarehouseToHome"] > 100
    report["outlier_warehousetohome_diperbaiki"] = int(outlier_mask.sum())
    df.loc[outlier_mask, "WarehouseToHome"] = df.loc[outlier_mask, "WarehouseToHome"] - 100

    # 2. Samakan label kategori yang artinya sama.
    for col, mapping in CATEGORY_FIXES.items():
        df[col] = df[col].replace(mapping)

    # 3. Imputasi median untuk nilai hilang pada kolom numerik.
    missing_before = df.isna().sum()
    report["missing_sebelum"] = {
        col: int(missing_before[col]) for col in df.columns if missing_before[col] > 0
    }
    for col in NUMERIC_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    report["missing_sesudah"] = int(df.isna().sum().sum())

    # 4. Pastikan tipe data target bersih.
    df["Churn"] = df["Churn"].astype(int)

    report["baris_bersih"] = int(len(df))
    report["jumlah_churn"] = int(df["Churn"].sum())
    report["churn_rate_persen"] = round(float(df["Churn"].mean() * 100), 2)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Pembersihan selesai.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"CSV bersih tersimpan di {OUT_CSV}")


if __name__ == "__main__":
    main()
