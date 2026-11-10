"""
02_eda.py
Analisis eksplorasi data (EDA) untuk dataset churn e-commerce.

Menghasilkan grafik di outputs/figures/ dan ringkasan angka di outputs/eda_stats.json.
Semua angka dihitung dari data bersih hasil 01_data_cleaning.py.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "processed" / "churn_clean.csv"
FIG_DIR = BASE / "outputs" / "figures"
OUT_STATS = BASE / "outputs" / "eda_stats.json"

COLOR_STAY = "#2e7d32"
COLOR_CHURN = "#c62828"
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def churn_rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    t = df.groupby(col)["Churn"].agg(["mean", "count", "sum"]).reset_index()
    t["mean"] = (t["mean"] * 100).round(2)
    t = t.rename(columns={"mean": "churn_rate_persen", "count": "jumlah_pelanggan", "sum": "jumlah_churn"})
    return t


def bar_churn(df: pd.DataFrame, col: str, title: str, fname: str, xlabel: str = "") -> None:
    t = churn_rate_table(df, col)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(t[col].astype(str), t["churn_rate_persen"], color="#ef6c00", edgecolor="#333333", linewidth=0.4)
    for i, v in enumerate(t["churn_rate_persen"]):
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel(xlabel or col)
    ax.set_ylim(0, max(t["churn_rate_persen"]) * 1.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    stats: dict = {}

    stats["total_pelanggan"] = int(len(df))
    stats["jumlah_churn"] = int(df["Churn"].sum())
    stats["churn_rate_persen"] = round(float(df["Churn"].mean() * 100), 2)

    # 1. Churn berdasarkan status komplain.
    by_complain = churn_rate_table(df, "Complain")
    stats["churn_tanpa_komplain_persen"] = float(by_complain.loc[by_complain["Complain"] == 0, "churn_rate_persen"].iloc[0])
    stats["churn_dengan_komplain_persen"] = float(by_complain.loc[by_complain["Complain"] == 1, "churn_rate_persen"].iloc[0])
    stats["jumlah_pernah_komplain"] = int((df["Complain"] == 1).sum())
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    labels = ["Tidak pernah komplain", "Pernah komplain"]
    vals = [stats["churn_tanpa_komplain_persen"], stats["churn_dengan_komplain_persen"]]
    ax.bar(labels, vals, color=[COLOR_STAY, COLOR_CHURN], edgecolor="#333333", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_title("Churn rate berdasarkan status komplain")
    ax.set_ylabel("Churn rate (%)")
    ax.set_ylim(0, max(vals) * 1.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "churn_by_complain.png")
    plt.close(fig)

    # 2. Churn berdasarkan kelompok tenure (lama berlangganan, bulan).
    bins = [0, 1, 3, 6, 12, 24, 61]
    labels_t = ["0-1", "2-3", "4-6", "7-12", "13-24", "25+"]
    df["TenureGroup"] = pd.cut(df["Tenure"], bins=bins, labels=labels_t, include_lowest=True)
    bar_churn(df, "TenureGroup", "Churn rate berdasarkan lama berlangganan", "churn_by_tenure.png", "Tenure (bulan)")
    stats["tenure_rata2_bertahan_bulan"] = round(float(df.loc[df["Churn"] == 0, "Tenure"].mean()), 2)
    stats["tenure_rata2_churn_bulan"] = round(float(df.loc[df["Churn"] == 1, "Tenure"].mean()), 2)

    # 3. Churn berdasarkan tingkat kota.
    bar_churn(df, "CityTier", "Churn rate berdasarkan tingkat kota (CityTier)", "churn_by_citytier.png", "CityTier")
    stats["churn_rate_per_citytier"] = churn_rate_table(df, "CityTier").to_dict("records")

    # 4. Churn berdasarkan metode pembayaran favorit.
    bar_churn(df, "PreferredPaymentMode", "Churn rate berdasarkan metode pembayaran favorit", "churn_by_payment.png")
    stats["churn_rate_per_metode_bayar"] = churn_rate_table(df, "PreferredPaymentMode").to_dict("records")

    # 5. Paradoks skor kepuasan: rata-rata skor kepuasan pelanggan churn sedikit lebih tinggi.
    sat = df.groupby("Churn")["SatisfactionScore"].mean()
    stats["kepuasan_rata2_bertahan"] = round(float(sat.loc[0]), 2)
    stats["kepuasan_rata2_churn"] = round(float(sat.loc[1]), 2)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.bar(["Bertahan", "Churn"], [sat.loc[0], sat.loc[1]], color=[COLOR_STAY, COLOR_CHURN], edgecolor="#333333", linewidth=0.4)
    for i, v in enumerate([sat.loc[0], sat.loc[1]]):
        ax.text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_title("Rata-rata skor kepuasan (skala 1 sampai 5)")
    ax.set_ylabel("Skor kepuasan rata-rata")
    ax.set_ylim(0, 5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "satisfaction_paradox.png")
    plt.close(fig)

    # 6. Heatmap korelasi kolom numerik.
    num = df.drop(columns=["CustomerID", "TenureGroup"])
    num = num.select_dtypes(include="number")
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="left", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(corr.values[i, j]) > 0.55 else "black")
    ax.set_title("Heatmap korelasi antar kolom numerik")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png")
    plt.close(fig)

    # Korelasi terkuat dengan Churn (selain Churn itu sendiri).
    corr_churn = corr["Churn"].drop(labels=["Churn"]).sort_values(key=abs, ascending=False)
    stats["korelasi_terkuat_dengan_churn"] = {k: round(float(v), 3) for k, v in corr_churn.head(6).items()}

    # Rata-rata cashback dan kupon per status.
    stats["cashback_rata2_bertahan"] = round(float(df.loc[df["Churn"] == 0, "CashbackAmount"].mean()), 1)
    stats["cashback_rata2_churn"] = round(float(df.loc[df["Churn"] == 1, "CashbackAmount"].mean()), 1)
    stats["kupon_rata2_bertahan"] = round(float(df.loc[df["Churn"] == 0, "CouponUsed"].mean()), 2)
    stats["kupon_rata2_churn"] = round(float(df.loc[df["Churn"] == 1, "CouponUsed"].mean()), 2)
    stats["hari_sejak_order_terakhir_bertahan"] = round(float(df.loc[df["Churn"] == 0, "DaySinceLastOrder"].mean()), 1)
    stats["hari_sejak_order_terakhir_churn"] = round(float(df.loc[df["Churn"] == 1, "DaySinceLastOrder"].mean()), 1)

    OUT_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("EDA selesai. Grafik tersimpan di", FIG_DIR)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
