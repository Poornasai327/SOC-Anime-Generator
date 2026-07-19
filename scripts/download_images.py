
"""
download_images_fast.py

Fast concurrent image downloader.

Requirements:
    pip install pandas requests tqdm
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import threading

import pandas as pd
import requests
from tqdm import tqdm

INPUT_CSV = Path("dataset/selected_dataset.csv")
OUTPUT_DIR = Path("dataset/images")
LABELS_CSV = Path("dataset/labels.csv")

MAX_WORKERS = 50      # Adjust to 16 if internet is slow
TIMEOUT = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

lock = threading.Lock()
records = []

def download(idx, row):
    url = row["sample_url"]
    ext = os.path.splitext(url)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"

    filename = f"{idx+1:06d}{ext}"
    path = OUTPUT_DIR / filename

    try:
        r = session.get(url, timeout=TIMEOUT)
        r.raise_for_status()

        with open(path, "wb") as f:
            f.write(r.content)

        with lock:
            records.append({
                "image": filename,
                "hair": row["hair"],
                "eyes": row["eyes"]
            })
        return True
    except Exception:
        return False

success = 0
failed = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [
        executor.submit(download, idx, row)
        for idx, (_, row) in enumerate(df.iterrows())
    ]

    for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
        if f.result():
            success += 1
        else:
            failed += 1

labels = pd.DataFrame(records).sort_values("image")
labels.to_csv(LABELS_CSV, index=False)

print("=" * 50)
print(f"Downloaded : {success:,}")
print(f"Failed     : {failed:,}")
print(f"Images     : {OUTPUT_DIR}")
print(f"Labels     : {LABELS_CSV}")
