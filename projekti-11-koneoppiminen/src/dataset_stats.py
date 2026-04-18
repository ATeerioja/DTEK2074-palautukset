"""
Dataset statistics and visualization.

Generates visual reports to understand your dataset before training.

Usage:
    python src/dataset_stats.py
    python src/dataset_stats.py --data_dir data/processed --save_dir reports/

Produces:
    1. Class distribution bar chart
    2. Sample grid per emotion
    3. Per-subject breakdown
    4. Pixel intensity histograms per emotion
"""

import argparse
import os
from collections import Counter, defaultdict
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt


EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset statistics")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--save_dir", type=str, default="reports")
    return parser.parse_args()


def scan_images(data_dir: str) -> list:
    """
    Scans processed directory and returns list of image info dicts.
    """
    images = []
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_dir):
            continue
        for filename in sorted(os.listdir(emotion_dir)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                # Try to extract subject
                match = re.match(r"^([a-zA-Z]+)_", filename)
                subject = match.group(1).lower() if match else "unknown"

                images.append({
                    "filepath": os.path.join(emotion_dir, filename),
                    "filename": filename,
                    "emotion": emotion,
                    "subject": subject,
                })
    return images


def plot_class_distribution(images: list, save_dir: str):
    """
    Bar chart of images per emotion class.
    """
    counts = Counter(img["emotion"] for img in images)

    fig, ax = plt.subplots(figsize=(10, 5))

    emotions = EMOTIONS
    values = [counts.get(e, 0) for e in emotions]
    colors = plt.cm.Set2(np.linspace(0, 1, len(emotions)))

    bars = ax.bar(emotions, values, color=colors, edgecolor="black", linewidth=0.5)

    # Add count labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution")
    ax.set_ylim(0, max(values) * 1.15)

    # Add ideal line
    ideal = len(images) / len(emotions)
    ax.axhline(y=ideal, color="red", linestyle="--", alpha=0.5,
               label=f"Ideal balanced ({ideal:.0f})")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_sample_grid(images: list, data_dir: str, save_dir: str):
    """
    Shows a grid of sample images for each emotion.
    5 random samples per emotion.
    """
    n_samples = 5
    fig, axes = plt.subplots(len(EMOTIONS), n_samples,
                             figsize=(n_samples * 2, len(EMOTIONS) * 2))

    for row, emotion in enumerate(EMOTIONS):
        emotion_images = [img for img in images if img["emotion"] == emotion]
        np.random.seed(42)

        if len(emotion_images) > n_samples:
            samples = np.random.choice(len(emotion_images), n_samples, replace=False)
        else:
            samples = range(len(emotion_images))

        for col in range(n_samples):
            ax = axes[row][col]

            if col < len(samples):
                img = cv2.imread(
                    emotion_images[samples[col]]["filepath"],
                    cv2.IMREAD_GRAYSCALE
                )
                if img is not None:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)

            ax.axis("off")
            if col == 0:
                ax.set_title(emotion, fontsize=10, fontweight="bold")

    plt.suptitle("Sample Images Per Emotion", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, "sample_grid.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_subject_breakdown(images: list, save_dir: str):
    """
    Stacked bar chart showing how many images each subject contributed per emotion.
    """
    subject_emotion_counts = defaultdict(lambda: defaultdict(int))

    for img in images:
        subject_emotion_counts[img["subject"]][img["emotion"]] += 1

    subjects = sorted(subject_emotion_counts.keys())

    if len(subjects) <= 1:
        print("  Skipping subject breakdown (only 1 subject)")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 1.5), 6))

    x = np.arange(len(subjects))
    width = 0.6
    bottom = np.zeros(len(subjects))
    colors = plt.cm.Set2(np.linspace(0, 1, len(EMOTIONS)))

    for i, emotion in enumerate(EMOTIONS):
        values = [subject_emotion_counts[s][emotion] for s in subjects]
        ax.bar(x, values, width, bottom=bottom, label=emotion, color=colors[i])
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_ylabel("Number of Images")
    ax.set_title("Images Per Subject Per Emotion")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    path = os.path.join(save_dir, "subject_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_intensity_histograms(images: list, save_dir: str):
    """
    Overlaid pixel intensity histograms per emotion.
    Reveals if different emotions have different brightness profiles.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(EMOTIONS)))

    for i, emotion in enumerate(EMOTIONS):
        emotion_images = [img for img in images if img["emotion"] == emotion]
        all_pixels = []

        # Sample up to 200 images to keep it fast
        sample = emotion_images[:200]

        for img_info in sample:
            img = cv2.imread(img_info["filepath"], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                all_pixels.extend(img.flatten().tolist())

        if all_pixels:
            ax.hist(all_pixels, bins=50, alpha=0.4, density=True,
                    color=colors[i], label=emotion)

    ax.set_xlabel("Pixel Intensity (0-255)")
    ax.set_ylabel("Density")
    ax.set_title("Pixel Intensity Distribution Per Emotion")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "intensity_histograms.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def print_summary_table(images: list):
    """
    Prints a detailed text summary to console.
    """
    counts = Counter(img["emotion"] for img in images)
    subjects = set(img["subject"] for img in images)

    print(f"\n{'='*50}")
    print(f"DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"Total images:    {len(images)}")
    print(f"Total subjects:  {len(subjects)}")
    print(f"Subjects:        {', '.join(sorted(subjects))}")
    print(f"Emotion classes: {len(counts)}")

    print(f"\n{'Emotion':>10s} | {'Count':>6s} | {'Pct':>6s} | {'Status':>10s}")
    print(f"{'-'*42}")

    total = len(images)
    for emotion in EMOTIONS:
        count = counts.get(emotion, 0)
        pct = count / total * 100 if total > 0 else 0

        if count == 0:
            status = "❌ MISSING"
        elif count < 50:
            status = "❌ LOW"
        elif count < 200:
            status = "⚠️  OK"
        else:
            status = "✅ GOOD"

        print(f"{emotion:>10s} | {count:>6d} | {pct:>5.1f}% | {status}")


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("Scanning dataset...")
    images = scan_images(args.data_dir)

    if not images:
        print(f"No images found in {args.data_dir}/")
        return

    print_summary_table(images)

    print(f"\nGenerating visualizations...")
    plot_class_distribution(images, args.save_dir)
    plot_sample_grid(images, args.data_dir, args.save_dir)
    plot_subject_breakdown(images, args.save_dir)
    plot_intensity_histograms(images, args.save_dir)

    print(f"\nAll reports saved to: {args.save_dir}/")


if __name__ == "__main__":
    main() 