from pathlib import Path
import time
import csv

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/resnet18_partial.pth")
TEST_DIR = Path("dataset/test")

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WARMUP_RUNS = 10
MEASURE_RUNS = 100


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("RESNET-18 LATENCY MEASUREMENT")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Model : {MODEL_PATH}")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model = models.resnet18(weights=None)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)
model.eval()


# ============================================================
# FIND TEST IMAGE
# ============================================================

image_files = []

for class_folder in TEST_DIR.iterdir():

    if class_folder.is_dir():

        for image_path in class_folder.iterdir():

            if image_path.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp"
            ]:
                image_files.append(image_path)


if len(image_files) == 0:
    raise RuntimeError("Tidak ada gambar test ditemukan.")


# Gunakan satu gambar sebagai input latency
image_path = image_files[0]

print(f"Test image: {image_path}")


# ============================================================
# PREPARE IMAGE
# ============================================================

image = Image.open(image_path).convert("RGB")

input_tensor = transform(image).unsqueeze(0).to(DEVICE)


# ============================================================
# WARM-UP
# ============================================================

print("\nWarm-up...")

with torch.no_grad():

    for _ in range(WARMUP_RUNS):

        _ = model(input_tensor)


# ============================================================
# MEASURE LATENCY
# ============================================================

print("Measuring latency...")

times = []

with torch.no_grad():

    for _ in range(MEASURE_RUNS):

        start_time = time.perf_counter()

        _ = model(input_tensor)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000

        times.append(elapsed_ms)


average_latency = sum(times) / len(times)

minimum_latency = min(times)
maximum_latency = max(times)


# ============================================================
# SAVE RESULT
# ============================================================

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

output_file = RESULT_DIR / "latency.csv"

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "model",
        "device",
        "average_latency_ms",
        "min_latency_ms",
        "max_latency_ms",
        "measurement_runs"
    ])

    writer.writerow([
        "ResNet-18 Partial Fine-Tuning",
        str(DEVICE),
        average_latency,
        minimum_latency,
        maximum_latency,
        MEASURE_RUNS
    ])


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("LATENCY RESULT")
print("=" * 60)

print(f"Average latency : {average_latency:.2f} ms/image")
print(f"Minimum latency : {minimum_latency:.2f} ms/image")
print(f"Maximum latency : {maximum_latency:.2f} ms/image")
print(f"Measurements    : {MEASURE_RUNS}")
print(f"Device          : {DEVICE}")

print(f"\nResult saved to: {output_file}")
