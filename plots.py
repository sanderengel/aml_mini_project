import pandas as pd
import matplotlib.pyplot as plt


SEARCH_RESULTS_PATH = "data/results/aug_search_results_2.csv"
EPOCH_RESULTS_PATH = "data/results/aug_epoch_results_2.csv"
PLOTS_DIR = "data/results/plots"


def config_name(row):
    rotation = row["rotation"]
    crop_scale = row["crop_scale"]
    color_jitter = row["color_jitter"]
    erasing_prob = row["erasing_prob"]

    if rotation == 0 and crop_scale == 1.0 and color_jitter == 0 and erasing_prob == 0:
        return "Baseline"
    if rotation == 10 and crop_scale == 1.0 and color_jitter == 0 and erasing_prob == 0:
        return "Rotation 10°"
    if rotation == 20 and crop_scale == 1.0 and color_jitter == 0 and erasing_prob == 0:
        return "Rotation 20°"
    if rotation == 0 and crop_scale == 0.8 and color_jitter == 0 and erasing_prob == 0:
        return "Crop 0.8"
    if rotation == 0 and crop_scale == 1.0 and color_jitter == 0.2 and erasing_prob == 0:
        return "Color jitter 0.2"
    if rotation == 0 and crop_scale == 1.0 and color_jitter == 0 and erasing_prob == 0.5:
        return "Random erasing 0.5"

    return "All augmentations"


def main():
    search = pd.read_csv(SEARCH_RESULTS_PATH)
    epochs = pd.read_csv(EPOCH_RESULTS_PATH)

    search["config"] = search.apply(config_name, axis=1)
    epochs["config"] = epochs.apply(config_name, axis=1)

    # Plot 1: validation loss ranking
    ranked_loss = search.sort_values("avg_val_loss")

    plt.figure(figsize=(9, 5))
    plt.barh(ranked_loss["config"], ranked_loss["avg_val_loss"])
    plt.xlabel("Average validation loss")
    plt.title("Augmentation search: lower validation loss is better")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/aug_val_loss_ranking_2.png", dpi=200)
    plt.close()

    # Plot 2: validation accuracy ranking
    ranked_acc = search.sort_values("avg_val_acc", ascending=False)

    plt.figure(figsize=(9, 5))
    plt.barh(ranked_acc["config"], ranked_acc["avg_val_acc"])
    plt.xlabel("Average validation accuracy")
    plt.title("Augmentation search: higher validation accuracy is better")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/aug_val_acc_ranking_2.png", dpi=200)
    plt.close()

    # Average epoch results across folds
    curve = (
        epochs.groupby(["config", "epoch"], as_index=False)
        .agg(
            train_loss=("train_loss", "mean"),
            val_loss=("val_loss", "mean"),
            val_acc=("val_acc", "mean"),
        )
    )

    # Plot 3: validation loss curves
    plt.figure(figsize=(10, 6))

    for config, part in curve.groupby("config"):
        plt.plot(part["epoch"], part["val_loss"], marker="o", label=config)

    plt.xlabel("Epoch")
    plt.ylabel("Average validation loss")
    plt.title("Validation loss over epochs by augmentation")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/aug_val_loss_curves_2.png", dpi=200)
    plt.close()

    # Plot 4: train vs validation loss for best config
    best_config = ranked_loss.iloc[0]["config"]
    best_curve = curve[curve["config"] == best_config]

    plt.figure(figsize=(8, 5))
    plt.plot(best_curve["epoch"], best_curve["train_loss"], marker="o", label="Train loss")
    plt.plot(best_curve["epoch"], best_curve["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Overfitting check: {best_config}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/best_config_train_vs_val_2.png", dpi=200)
    plt.close()

    print(f"Plots saved to {PLOTS_DIR}")

    # Plot 5: train vs validation loss for all augmentations
    for config, part in curve.groupby("config"):
        plt.figure(figsize=(8, 5))

        plt.plot(part["epoch"], part["train_loss"], marker="o", label="Train loss")
        plt.plot(part["epoch"], part["val_loss"], marker="o", label="Validation loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Overfitting check: {config}")
        plt.legend()
        plt.tight_layout()

        safe_name = (
            config.lower()
            .replace(" ", "_")
            .replace("°", "deg")
            .replace(".", "_")
        )

        plt.savefig(f"{PLOTS_DIR}/overfitting_check_{safe_name}_2.png", dpi=200)
        plt.close()

if __name__ == "__main__":
    main()