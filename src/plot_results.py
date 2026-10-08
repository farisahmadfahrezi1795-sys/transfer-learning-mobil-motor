from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULT_DIR = Path("results")


def plot_single(experiment):
    csv_file = RESULT_DIR / f"history_{experiment}.csv"

    df = pd.read_csv(csv_file)

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["train_accuracy"] * 100,
        marker="o",
        label="Train Accuracy"
    )

    plt.plot(
        df["epoch"],
        df["val_accuracy"] * 100,
        marker="o",
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy per Epoch - {experiment.title()}")
    plt.xticks(df["epoch"])
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_file = RESULT_DIR / f"accuracy_{experiment}.png"

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Graph saved: {output_file}")


def plot_comparison():
    experiments = ["feature", "partial", "scratch"]

    plt.figure(figsize=(9, 5))

    for experiment in experiments:

        csv_file = RESULT_DIR / f"history_{experiment}.csv"

        df = pd.read_csv(csv_file)

        plt.plot(
            df["epoch"],
            df["val_accuracy"] * 100,
            marker="o",
            label=experiment.title()
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy Comparison")
    plt.xticks(range(1, 11))
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_file = RESULT_DIR / "accuracy_comparison.png"

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Graph saved: {output_file}")


# Generate individual graphs
plot_single("feature")
plot_single("partial")
plot_single("scratch")

# Generate comparison graph
plot_comparison()

print("\nAll graphs generated successfully.")