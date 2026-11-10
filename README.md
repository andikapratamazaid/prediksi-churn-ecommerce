# Prediksi Pelanggan Berhenti Belanja (E-Commerce Churn)

Project data science untuk memprediksi pelanggan e-commerce yang akan berhenti belanja (churn), lengkap dari pembersihan data, analisis eksplorasi, pemodelan machine learning, sampai segmentasi risiko dan rekomendasi retensi yang bisa ditindaklanjuti tim bisnis.

## Latar Belakang

Mempertahankan pelanggan lama hampir selalu lebih murah daripada mencari pelanggan baru. Masalahnya, tim retensi biasanya bekerja tanpa prioritas: semua pelanggan diperlakukan sama, padahal yang benar-benar akan pergi hanya sebagian kecil. Project ini menjawab tiga pertanyaan:

1. Pelanggan seperti apa yang paling banyak berhenti belanja?
2. Bisakah churn diprediksi dari perilaku dan riwayat pelanggan?
3. Pada kelompok pelanggan mana anggaran retensi sebaiknya difokuskan?

## Sumber Data

Dataset yang dipakai adalah **Ecommerce Customer Churn Analysis and Prediction** dari Kaggle:

- Tautan Kaggle: https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
- Isi: data 5.630 pelanggan sebuah perusahaan e-commerce dengan 20 kolom.
- File mentah diunduh dari mirror publik di GitHub karena Kaggle mewajibkan login untuk unduh langsung: https://raw.githubusercontent.com/miriamyi01/Data_Challenge/main/Exercise_2/E%20Commerce%20Dataset.xlsx (repositori https://github.com/miriamyi01/Data_Challenge)
- Verifikasi file hasil unduhan: sheet "E Comm" berisi tepat 5.630 baris dan 20 kolom sesuai daftar di bawah, dengan 948 pelanggan churn dan 4.682 pelanggan bertahan, churn rate 16,84%. Angka ini sama dengan yang terdokumentasi untuk dataset Kaggle aslinya.

Kolom dataset:

| Kelompok | Kolom |
| --- | --- |
| Identitas dan target | CustomerID, Churn |
| Profil pelanggan | Tenure (lama berlangganan, bulan), CityTier (tingkat kota 1 sampai 3), Gender, MaritalStatus, NumberOfAddress, WarehouseToHome (jarak gudang ke rumah, km) |
| Perilaku di aplikasi | PreferredLoginDevice, HourSpendOnApp, NumberOfDeviceRegistered, PreferedOrderCat (kategori paling sering dipesan), PreferredPaymentMode |
| Transaksi | OrderCount, OrderAmountHikeFromlastYear, CouponUsed, DaySinceLastOrder, CashbackAmount |
| Pengalaman layanan | SatisfactionScore (skala 1 sampai 5), Complain (pernah komplain atau tidak) |

## Tahapan dan Cara Menjalankan

Butuh Python 3.10 atau lebih baru. Instal dependensi lalu jalankan empat skrip secara berurutan dari folder project:

```bash
pip install -r requirements.txt
python src/01_data_cleaning.py
python src/02_eda.py
python src/03_modeling.py
python src/04_rekomendasi.py
```

| Skrip | Fungsi | Output utama |
| --- | --- | --- |
| `src/01_data_cleaning.py` | Pembersihan data | `data/processed/churn_clean.csv`, `outputs/cleaning_report.json` |
| `src/02_eda.py` | Analisis eksplorasi dan grafik | `outputs/figures/*.png`, `outputs/eda_stats.json` |
| `src/03_modeling.py` | Pemodelan dan evaluasi | `outputs/model_metrics.csv`, `outputs/feature_importance.csv`, `outputs/test_predictions.csv` |
| `src/04_rekomendasi.py` | Segmentasi risiko dan simulasi retensi | `outputs/segmentasi_risiko.csv`, `outputs/segmentasi_stats.json`, `outputs/simulasi_retensi.json` |

Dashboard statis hasil analisis tersedia di `dashboard/index.html` dan bisa dibuka langsung di browser.

## Pembersihan Data

Dataset ini tidak datang dalam keadaan bersih, dan itu bagian dari pekerjaannya:

- **Nilai hilang** pada 7 kolom numerik: Tenure 264 baris, DaySinceLastOrder 307 baris, OrderAmountHikeFromlastYear 265 baris, OrderCount 258 baris, CouponUsed 256 baris, HourSpendOnApp 255 baris, dan WarehouseToHome 251 baris. Ditangani dengan imputasi median per kolom.
- **Outlier salah ketik** pada WarehouseToHome: 2 baris bernilai 126 dan 127 km, padahal nilai wajar pada dataset ini maksimal 36 km. Diperbaiki menjadi 26 dan 27 km.
- **Label kategori ganda** dengan arti sama disatukan: "Phone" menjadi "Mobile Phone" pada PreferredLoginDevice, "CC" menjadi "Credit Card" dan "COD" menjadi "Cash on Delivery" pada PreferredPaymentMode, serta "Mobile" menjadi "Mobile Phone" pada PreferedOrderCat.

## Temuan Utama

Semua angka di bawah adalah hasil menjalankan skrip pada dataset ini.

1. **Komplain adalah sinyal churn terkuat kedua.** Pelanggan yang pernah komplain berhenti belanja pada tingkat **31,67%**, hampir tiga kali lipat dibanding pelanggan tanpa komplain yang hanya **10,93%**. Ada 1.604 dari 5.630 pelanggan yang pernah komplain.
2. **Churn adalah masalah pelanggan baru.** Rata-rata lama berlangganan pelanggan yang churn hanya **3,86 bulan**, sedangkan pelanggan yang bertahan rata-rata sudah **11,40 bulan**. Korelasi Tenure dengan churn adalah yang terkuat, minus 0,338.
3. **Ada paradoks skor kepuasan.** Rata-rata skor kepuasan pelanggan yang churn adalah **3,39 dari 5**, justru lebih tinggi daripada pelanggan yang bertahan, yaitu **3,00**. Skor survei kepuasan saja tidak bisa dipakai membaca risiko churn.
4. **Metode pembayaran dan tingkat kota ikut membedakan.** Churn tertinggi ada pada pengguna Cash on Delivery (**24,90%**) dan E wallet (22,80%), terendah pada pengguna Credit Card (14,21%). Berdasarkan kota, churn naik dari tier 1 (14,51%) ke tier 3 (**21,37%**).
5. **Cashback membedakan, kupon tidak.** Rata-rata cashback pelanggan bertahan 180,6 melawan 160,4 pada pelanggan churn. Sebaliknya pemakaian kupon hampir identik pada kedua kelompok (1,72 lawan 1,71), jadi kupon bukan pembeda utama retensi pada data ini.

## Hasil Model

Pembagian data 80% latih dan 20% uji dengan stratifikasi. Kelas churn adalah minoritas (16,84%), jadi ketiga model memakai penanganan ketidakseimbangan (class_weight balanced) dan model terbaik dipilih berdasarkan PR-AUC pada data uji, bukan akurasi.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| **Random Forest** | **0,9987** | **0,9939** | **0,9940** | 0,8684 | **0,9270** | **0,9769** |
| Gradient Boosting | 0,9970 | 0,9848 | 0,8624 | **0,9895** | 0,9216 | 0,9716 |
| Logistic Regression | 0,8854 | 0,6789 | 0,4411 | 0,8474 | 0,5802 | 0,7931 |

Model terbaik adalah **Random Forest**. Gradient Boosting layak dicatat karena recall-nya paling tinggi (0,9895): bila tujuan bisnisnya menangkap sebanyak mungkin calon churn dan biaya intervensi murah, model ini bisa lebih cocok walau precision-nya lebih rendah.

Fitur terpenting menurut permutation importance pada Random Forest: **Tenure (0,3439)**, **Complain (0,1328)**, PreferedOrderCat (0,0616), MaritalStatus (0,0592), dan NumberOfAddress (0,0423). Urutan ini sejalan dengan temuan EDA, jadi cerita analisis dan modelnya konsisten.

Catatan kejujuran soal performa: skor model pohon yang nyaris sempurna pada data uji menunjukkan label churn pada dataset publik ini mengikuti pola yang sangat mudah dipelajari mesin, kemungkinan karena dataset ini disusun atau dibangkitkan dengan pola yang teratur. Pada data operasional perusahaan yang sebenarnya, performa model churn umumnya jauh lebih rendah. Angka di sini adalah hasil yang benar pada dataset ini, bukan gambaran performa di lapangan.

## Segmentasi Risiko dan Rekomendasi Retensi

Probabilitas risiko semua pelanggan dihitung dengan metode out-of-fold 5 fold memakai Random Forest, sehingga probabilitas tiap pelanggan berasal dari model yang tidak dilatih pada data pelanggan itu sendiri.

| Segmen | Ambang probabilitas | Jumlah pelanggan | Churn aktual | Churn rate aktual |
| --- | --- | --- | --- | --- |
| Tinggi | 0,60 atau lebih | 702 | 691 | 98,43% |
| Sedang | 0,30 sampai 0,60 | 305 | 197 | 64,59% |
| Rendah | di bawah 0,30 | 4.623 | 60 | 1,30% |

Segmen tinggi dan sedang hanya berisi 1.007 pelanggan (17,9% basis pelanggan), tetapi mencakup **93,67%** dari seluruh pelanggan yang benar-benar churn. Segmentasinya terbukti memisahkan risiko dengan baik.

Rekomendasi retensi:

1. **Tangani komplain dengan cepat dan tuntas.** Simulasi sederhana berbasis selisih churn rate: bila penanganan komplain mampu menekan churn rate kelompok komplain dari 31,67% menjadi setara kelompok tanpa komplain 10,93%, perkiraan pelanggan yang terselamatkan sekitar **333 orang**. Skenario moderat, hanya separuh selisih yang tertutup, memberi angka sekitar **166 orang**. Ini asumsi eksploratif untuk prioritas tindakan, bukan janji hasil.
2. **Jaga pelanggan pada bulan-bulan pertama.** Program pendampingan awal dan pemicu pesanan kedua sebaiknya menyasar pelanggan dengan tenure di bawah 6 bulan.
3. **Fokuskan anggaran retensi pada segmen tinggi dan sedang**, bukan promo massal ke semua pelanggan.
4. **Jangan menilai risiko hanya dari skor kepuasan.** Gabungkan dengan riwayat komplain, tenure, dan hari sejak pesanan terakhir.
5. **Perbaiki pengalaman pengguna Cash on Delivery dan wilayah kota tier 3**, dua kelompok dengan churn rate tertinggi.

## Struktur Folder

```
prediksi-churn-ecommerce/
  data/
    raw/          File mentah hasil unduhan (E Commerce Dataset.xlsx)
    processed/    churn_clean.csv hasil pembersihan
  src/
    01_data_cleaning.py
    02_eda.py
    03_modeling.py
    04_rekomendasi.py
  outputs/
    figures/          Grafik EDA (6 berkas PNG)
    model_metrics.csv, feature_importance.csv, test_predictions.csv
    segmentasi_risiko.csv, segmentasi_stats.json, simulasi_retensi.json
    cleaning_report.json, eda_stats.json, model_summary.json
  dashboard/
    index.html        Dashboard statis (merujuk grafik di outputs/figures)
  requirements.txt
  README.md
```

## Batasan

- Data berasal dari **satu perusahaan e-commerce** dan merupakan snapshot satu periode, bukan data runtut waktu, jadi pola musiman dan perubahan perilaku tidak terlihat.
- Dataset ini **bukan data Indonesia** dan konteks pasarnya bisa berbeda, misalnya keberadaan metode pembayaran UPI dan struktur CityTier.
- Label churn sudah disediakan dataset; definisi persisnya (berapa lama tidak belanja dianggap churn) tidak terdokumentasi oleh penyedia dataset.
- XGBoost tidak dijalankan pada putaran ini karena library-nya tidak tersedia di lingkungan pengerjaan. Skrip pemodelan akan otomatis membandingkannya bila library tersebut terpasang. Sebagai gantinya, Gradient Boosting yang dipakai adalah HistGradientBoosting dari scikit-learn.
- Simulasi retensi berbasis selisih churn rate observasional, bukan hasil eksperimen terkontrol, jadi hanya layak dipakai sebagai perkiraan kasar untuk prioritas.

## Teknologi

Python, pandas, scikit-learn, matplotlib, openpyxl.
