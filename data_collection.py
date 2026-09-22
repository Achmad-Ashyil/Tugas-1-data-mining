"""
data_collection.py
-------------------
Skrip ingestion data untuk Capstone:
"Sistem Informasi Operasional Penerbangan & Kepuasan Penumpang
Bandara Internasional Sultan Hasanuddin Makassar"

Sumber data (Kaggle):
1. Airlines Dataset to Predict a Delay (jimschacko)
   https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay
   -> file: Airlines.csv

2. Airline Passenger Satisfaction (teejmahal20)
   https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
   -> file: train.csv, test.csv

Cara pakai:
  A. Download manual (paling gampang, tidak perlu API key):
     1. Download kedua dataset di atas dari Kaggle.
     2. Taruh semua file csv (Airlines.csv, train.csv, test.csv) ke folder data/raw/
     3. Jalankan: python data_collection.py

  B. Auto-download via Kaggle API (untuk reproducibility, opsional):
     1. Siapkan kaggle.json di ~/.kaggle/kaggle.json (Kaggle Account -> Create New Token)
     2. pip install -r requirements.txt
     3. Jalankan: python data_collection.py --download

Skrip ini AMAN dijalankan meskipun paket/kredensial kaggle belum ada:
kalau auto-download gagal, skrip hanya memberi peringatan dan tetap lanjut
membaca file csv yang sudah ada secara manual di data/raw/.
"""

import argparse
import os
import sqlite3
import sys

import pandas as pd

RAW_DIR = os.path.join("data", "raw")
DB_PATH = "bandara_makassar_local.db"

# dataset_id (sesuai slug di URL kaggle) -> daftar file csv yang dibutuhkan
DATASETS = {
    "jimschacko/airlines-dataset-to-predict-a-delay": ["Airlines.csv"],
    "teejmahal20/airline-passenger-satisfaction": ["train.csv", "test.csv"],
}

REQUIRED_FILES = [f for files in DATASETS.values() for f in files]


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)


def try_kaggle_download():
    """Coba download otomatis lewat Kaggle API.
    Kalau paket 'kaggle' belum terpasang atau kredensial belum diset,
    fungsi ini akan gagal dengan AMAN (tidak menghentikan program)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[INFO] Paket 'kaggle' belum terpasang (pip install kaggle). Lewati auto-download.")
        return

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"[INFO] Autentikasi Kaggle gagal ({e}).")
        print("       Lewati auto-download, silakan download manual ke folder data/raw/.")
        return

    for dataset_id in DATASETS:
        try:
            print(f"[DOWNLOAD] {dataset_id} ...")
            api.dataset_download_files(dataset_id, path=RAW_DIR, unzip=True)
        except Exception as e:
            print(f"[WARNING] Gagal download {dataset_id}: {e}")


def check_files():
    """Pastikan semua file csv yang dibutuhkan sudah ada di data/raw/."""
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(RAW_DIR, f))]
    if missing:
        print("\n[PERHATIAN] File berikut belum ditemukan di data/raw/:")
        for f in missing:
            print(f"  - {f}")
        print("\nSilakan download manual dari Kaggle, lalu taruh filenya persis di data/raw/:")
        for dataset_id, files in DATASETS.items():
            print(f"  https://www.kaggle.com/datasets/{dataset_id}")
            print(f"    -> {', '.join(files)}")
        return False
    return True


def load_to_sqlite():
    """Baca csv mentah, bersihkan sedikit, lalu simpan ke SQLite lokal."""
    conn = sqlite3.connect(DB_PATH)

    # 1. Airlines.csv -> tabel flight_delay
    airlines_path = os.path.join(RAW_DIR, "Airlines.csv")
    df_airlines = pd.read_csv(airlines_path)
    df_airlines.to_sql("flight_delay", conn, if_exists="replace", index=False)
    print(f"[OK] tabel 'flight_delay'         : {len(df_airlines):,} baris (dari Airlines.csv)")

    # 2. train.csv + test.csv -> digabung jadi tabel passenger_satisfaction
    train_path = os.path.join(RAW_DIR, "train.csv")
    test_path = os.path.join(RAW_DIR, "test.csv")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_train["split"] = "train"
    df_test["split"] = "test"
    df_satisfaction = pd.concat([df_train, df_test], ignore_index=True)

    # buang kolom index bawaan Kaggle kalau ada, biar rapi
    for col in ["Unnamed: 0", "id"]:
        if col in df_satisfaction.columns:
            df_satisfaction = df_satisfaction.drop(columns=col)

    df_satisfaction.to_sql("passenger_satisfaction", conn, if_exists="replace", index=False)
    print(f"[OK] tabel 'passenger_satisfaction': {len(df_satisfaction):,} baris (train+test)")

    conn.close()
    print(f"\n[SELESAI] Semua data tersimpan di {DB_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingestion data Capstone - Bandara Sultan Hasanuddin Makassar"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Coba auto-download dataset dari Kaggle API (butuh kaggle.json)",
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.download:
        try_kaggle_download()

    if not check_files():
        sys.exit(1)

    load_to_sqlite()


if __name__ == "__main__":
    main()
