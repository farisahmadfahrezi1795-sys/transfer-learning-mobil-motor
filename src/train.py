from pathlib import Path
import time
import csv
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("dataset")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_CLASSES = 2
IMAGE_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("TRANSFER LEARNING - RESNET-18")
print("=" * 60)
print(f"Device : {DEVICE}")
print(f"Epochs : {NUM_EPOCHS}")
print(f"Batch  : {BATCH_SIZE}")
print("=" * 60)


# ============================================================
# DATA TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=val_transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nClass mapping:")
print(train_dataset.class_to_idx)

print("\nDataset:")
print(f"Train      : {len(train_dataset)} images")
print(f"Validation : {len(val_dataset)} images")
print(f"Test       : {len(test_dataset)} images")


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(experiment):
    """
    experiment:
        feature  -> Feature Extraction
        partial  -> Partial Fine-Tuning
        scratch  -> Training from Scratch
    """

    if experiment == "scratch":
        print("\nCreating ResNet-18 from scratch...")
        model = models.resnet18(weights=None)

    else:
        print("\nCreating ResNet-18 with ImageNet pretrained weights...")
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)

    # Replace final classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    if experiment == "feature":

        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    # --------------------------------------------------------
    # PARTIAL FINE-TUNING
    # --------------------------------------------------------

    elif experiment == "partial":

        for param in model.parameters():
            param.requires_grad = False

        # Train final ResNet block
        for param in model.layer4.parameters():
            param.requires_grad = True

        # Train classifier
        for param in model.fc.parameters():
            param.requires_grad = True

    # --------------------------------------------------------
    # SCRATCH
    # --------------------------------------------------------

    elif experiment == "scratch":

        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    return model


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_experiment(experiment, learning_rate):

    print("\n")
    print("=" * 60)
    print(f"EXPERIMENT: {experiment.upper()}")
    print(f"Learning Rate: {learning_rate}")
    print("=" * 60)

    model = create_model(experiment)
    model = model.to(DEVICE)

    # Only optimize parameters that require gradients
    trainable_parameters = [
        param for param in model.parameters()
        if param.requires_grad
    ]

    optimizer = optim.Adam(
        trainable_parameters,
        lr=learning_rate
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )

    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    epoch_90 = None

    history = []

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_accuracy = correct / total

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss_total += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss_total / val_total
        val_accuracy = val_correct / val_total

        scheduler.step()

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

        # Epoch reaching 90%
        if epoch_90 is None and val_accuracy >= 0.90:
            epoch_90 = epoch + 1

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(
            f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy * 100:.2f}% "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy * 100:.2f}%"
        )

    training_time = time.time() - start_time

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    model.load_state_dict(best_model_state)

    model_path = MODEL_DIR / f"resnet18_{experiment}.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": train_dataset.class_to_idx,
            "experiment": experiment,
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch
        },
        model_path
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = test_correct / test_total

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_path = RESULT_DIR / f"history_{experiment}.csv"

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    print("\nResult:")
    print(f"Best validation accuracy : {best_val_accuracy * 100:.2f}%")
    print(f"Best epoch               : {best_epoch}")

    if epoch_90 is not None:
        print(f"Epoch >= 90%            : {epoch_90}")
    else:
        print("Epoch >= 90%            : Not reached")

    print(f"Test accuracy            : {test_accuracy * 100:.2f}%")
    print(f"Training time            : {training_time:.2f} seconds")
    print(f"Model saved              : {model_path}")

    return {
        "experiment": experiment,
        "learning_rate": learning_rate,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "epoch_90": epoch_90,
        "test_accuracy": test_accuracy,
        "training_time_seconds": training_time
    }


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

results = []

# 1. Feature Extraction
results.append(
    train_one_experiment(
        experiment="feature",
        learning_rate=1e-3
    )
)

# 2. Partial Fine-Tuning
results.append(
    train_one_experiment(
        experiment="partial",
        learning_rate=1e-4
    )
)

# 3. Training from Scratch
results.append(
    train_one_experiment(
        experiment="scratch",
        learning_rate=1e-3
    )
)


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_path = RESULT_DIR / "comparison.csv"

with open(
    comparison_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "experiment",
            "learning_rate",
            "best_val_accuracy",
            "best_epoch",
            "epoch_90",
            "test_accuracy",
            "training_time_seconds"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("ALL EXPERIMENTS COMPLETED")
print("=" * 60)

for result in results:

    print(
        f"{result['experiment']:10s} | "
        f"Val Acc: {result['best_val_accuracy'] * 100:6.2f}% | "
        f"Test Acc: {result['test_accuracy'] * 100:6.2f}% | "
        f"Time: {result['training_time_seconds']:8.2f}s"
    )

print("\nComparison saved to:")
print(comparison_path)

print("\nTraining completed.")