"""
Detailed evaluation of a trained model.

Usage:
    python src/evaluate.py --checkpoint models/best-emotion-model-epoch=15-val_loss=0.6234.ckpt

Produces:
    - Per-class precision, recall, F1
    - Confusion matrix (saved as image)
    - Most confident wrong predictions (for error analysis)
"""

import argparse
import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datamodule import EmotionDataModule
from model import EmotionCNN
from dataset import EmotionDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--batch_size", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    emotions = EmotionDataset.EMOTIONS

    # =====================
    # LOAD MODEL AND DATA
    # =====================
    print("Loading model and data...")

    model = EmotionCNN.load_from_checkpoint(args.checkpoint, strict=False)
    model.eval()
    model.freeze()

    datamodule = EmotionDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
    )
    datamodule.setup()

    # =====================
    # COLLECT PREDICTIONS
    # =====================
    print("Running inference on test set...")

    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    test_loader = datamodule.test_dataloader()
    path_idx = 0

    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Track file paths for error analysis
            batch_size = len(labels)
            for i in range(batch_size):
                if path_idx < len(datamodule.test_dataset.image_paths):
                    all_paths.append(
                        datamodule.test_dataset.image_paths[path_idx]
                    )
                path_idx += 1

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # =====================
    # OVERALL METRICS
    # =====================
    overall_acc = (all_preds == all_labels).mean()

    print("\n" + "=" * 60)
    print(f"OVERALL ACCURACY: {overall_acc:.1%}")
    print("=" * 60)

    # =====================
    # PER-CLASS METRICS
    # =====================
    print(f"\n{'Emotion':>12s} | {'Precision':>9s} | {'Recall':>6s} | {'F1':>6s} | {'Support':>7s}")
    print("-" * 55)

    for i, emotion in enumerate(emotions):
        # True positives, false positives, false negatives
        tp = ((all_preds == i) & (all_labels == i)).sum()
        fp = ((all_preds == i) & (all_labels != i)).sum()
        fn = ((all_preds != i) & (all_labels == i)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) \
            if (precision + recall) > 0 else 0
        support = (all_labels == i).sum()

        print(f"{emotion:>12s} | {precision:>9.3f} | {recall:>6.3f} | {f1:>6.3f} | {support:>7d}")

    # =====================
    # CONFUSION MATRIX
    # =====================
    print("\nSaving confusion matrix to 'confusion_matrix.png'...")

    n_classes = len(emotions)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=emotions,
        yticklabels=emotions,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    # =====================
    # ERROR ANALYSIS
    # =====================
    # Find the most confident WRONG predictions
    # These are the most instructive errors to examine

    print("\nTop 20 Most Confident WRONG Predictions:")
    print(f"{'Confidence':>10s} | {'Predicted':>10s} | {'Actual':>10s} | Path")
    print("-" * 75)

    errors = []
    for i in range(len(all_preds)):
        if all_preds[i] != all_labels[i]:
            confidence = all_probs[i][all_preds[i]]
            errors.append({
                "confidence": confidence,
                "predicted": emotions[all_preds[i]],
                "actual": emotions[all_labels[i]],
                "path": all_paths[i] if i < len(all_paths) else "unknown",
            })

    # Sort by confidence descending — most confident mistakes first
    errors.sort(key=lambda x: x["confidence"], reverse=True)

    for error in errors[:20]:
        print(
            f"{error['confidence']:>10.1%} | "
            f"{error['predicted']:>10s} | "
            f"{error['actual']:>10s} | "
            f"{os.path.basename(error['path'])}"
        )

    print(f"\nTotal errors: {len(errors)} / {len(all_preds)} "
          f"({len(errors)/len(all_preds):.1%})")


if __name__ == "__main__":
    main()