from pathlib import Path
import csv
from datetime import datetime

DATASET_DIR = Path("dataset")
OUTPUT_FILE = DATASET_DIR / "metadata.csv"

rows = []

# Dataset train, validation, dan test
# Metadata hanya mencatat 120 gambar unik yang digunakan
# dalam proses pembagian dataset.
for split in ["train", "val", "test"]:
    for class_name in ["mobil", "motor"]:
        folder = DATASET_DIR / split / class_name

        if not folder.exists():
            print(f"Folder tidak ditemukan: {folder}")
            continue

        for image_path in sorted(folder.iterdir()):
            if image_path.is_file():
                rows.append({
                    "filename": str(image_path.relative_to(DATASET_DIR)).replace("\\", "/"),
                    "class": class_name,
                    "split": split,
                    "date": "unknown",
                    "lighting": "unknown",
                })

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["filename", "class", "split", "date", "lighting"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Metadata berhasil dibuat: {OUTPUT_FILE}")
print(f"Total gambar unik: {len(rows)}")
print(f"Tanggal pembuatan metadata: {datetime.now():%Y-%m-%d %H:%M:%S}")
