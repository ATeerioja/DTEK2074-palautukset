"""
Automated Quality Assurance checks on the processed dataset.

Checks:
    1. Image integrity     — can every file be loaded?
    2. Image dimensions    — is everything 48x48?
    3. Channel count       — is everything grayscale (1 channel)?
    4. Pixel range         — any fully black or fully white images?
    5. Duplicate detection — any exact or near-duplicate images?
    6. Class balance       — how imbalanced is the dataset?
    7. Filename format     — does every file follow the naming convention?
    8. Minimum viable count — enough images per class to train?

Usage:
    python src/qa.py
    python src/qa.py --data_dir data/processed --fix
"""

import argparse
import os
import hashlib
from collections import Counter, defaultdict

import cv2
import numpy as np


EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EXPECTED_SIZE = 48
MIN_IMAGES_PER_CLASS = 50      # absolute minimum to attempt training
RECOMMENDED_PER_CLASS = 200    # recommended minimum


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset quality assurance")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--fix", action="store_true",
                        help="Attempt to auto-fix issues (resize, convert)")
    parser.add_argument("--remove_duplicates", action="store_true",
                        help="Remove detected duplicate images")
    return parser.parse_args()


class QualityChecker:
    """
    Runs comprehensive quality checks on the processed dataset.
    """

    def __init__(self, args):
        self.data_dir = args.data_dir
        self.auto_fix = args.fix
        self.remove_duplicates = args.remove_duplicates

        # Issue tracking
        self.issues = {
            "load_failed": [],
            "wrong_size": [],
            "not_grayscale": [],
            "too_dark": [],
            "too_bright": [],
            "low_contrast": [],
            "duplicates": [],
            "bad_filename": [],
        }

        self.all_passed = True

    def run_all_checks(self):
        """
        Runs all quality checks and prints a report.
        """
        print("=" * 60)
        print("DATASET QUALITY ASSURANCE")
        print("=" * 60)

        # Scan all images
        images = self._scan_all_images()
        print(f"\nTotal images found: {len(images)}")

        if len(images) == 0:
            print(f"ERROR: No images found in {self.data_dir}/")
            return

        # Run each check
        self._check_1_integrity(images)
        self._check_2_dimensions(images)
        self._check_3_channels(images)
        self._check_4_pixel_range(images)
        self._check_5_duplicates(images)
        self._check_6_class_balance(images)
        self._check_7_filename_format(images)
        self._check_8_minimum_counts(images)

        # Final report
        self._print_report()

    def _scan_all_images(self) -> list:
        """
        Finds all image files in the processed directory.
        """
        images = []
        for emotion in EMOTIONS:
            emotion_dir = os.path.join(self.data_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue

            for filename in sorted(os.listdir(emotion_dir)):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    images.append({
                        "filepath": os.path.join(emotion_dir, filename),
                        "filename": filename,
                        "emotion": emotion,
                    })

        return images

    # --- Check 1: Image Integrity ---

    def _check_1_integrity(self, images: list):
        """
        Verifies every image can be loaded without errors.
        """
        print("\n[Check 1] Image Integrity")
        failed = 0

        for img_info in images:
            img = cv2.imread(img_info["filepath"])
            if img is None:
                self.issues["load_failed"].append(img_info["filepath"])
                failed += 1

        if failed > 0:
            print(f"  ❌ FAILED: {failed} images could not be loaded")
            self.all_passed = False
        else:
            print(f"  ✅ PASSED: All {len(images)} images load correctly")

    # --- Check 2: Dimensions ---

    def _check_2_dimensions(self, images: list):
        """
        Verifies all images are exactly 48x48 pixels.
        """
        print("\n[Check 2] Image Dimensions")
        wrong = []

        for img_info in images:
            img = cv2.imread(img_info["filepath"])
            if img is None:
                continue

            h, w = img.shape[:2]
            if h != EXPECTED_SIZE or w != EXPECTED_SIZE:
                wrong.append({
                    "path": img_info["filepath"],
                    "actual": (w, h),
                })

                if self.auto_fix:
                    resized = cv2.resize(img, (EXPECTED_SIZE, EXPECTED_SIZE))
                    cv2.imwrite(img_info["filepath"], resized)

        if wrong:
            self.issues["wrong_size"] = wrong
            action = "Auto-fixed" if self.auto_fix else "FAILED"
            print(f"  ❌ {action}: {len(wrong)} images have wrong dimensions")
            for w in wrong[:5]:     # show first 5
                print(f"      {w['path']} → {w['actual']}")
            if len(wrong) > 5:
                print(f"      ... and {len(wrong) - 5} more")
            self.all_passed = False
        else:
            print(f"  ✅ PASSED: All images are {EXPECTED_SIZE}x{EXPECTED_SIZE}")

    # --- Check 3: Channels ---

    def _check_3_channels(self, images: list):
        """
        Verifies all images are single-channel grayscale.
        """
        print("\n[Check 3] Grayscale Check")
        not_gray = []

        for img_info in images:
            img = cv2.imread(img_info["filepath"])
            if img is None:
                continue

            if len(img.shape) == 3 and img.shape[2] != 1:
                # Check if it's actually grayscale stored as BGR
                # (all 3 channels identical)
                b, g, r = cv2.split(img)
                if np.array_equal(b, g) and np.array_equal(g, r):
                    # It's grayscale stored as 3-channel — fix it
                    if self.auto_fix:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(img_info["filepath"], gray)
                else:
                    not_gray.append(img_info["filepath"])
                    if self.auto_fix:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(img_info["filepath"], gray)

        if not_gray:
            self.issues["not_grayscale"] = not_gray
            action = "Auto-fixed" if self.auto_fix else "FAILED"
            print(f"  ❌ {action}: {len(not_gray)} images are not grayscale")
            self.all_passed = False
        else:
            print(f"  ✅ PASSED: All images are grayscale")

    # --- Check 4: Pixel Range ---

    def _check_4_pixel_range(self, images: list):
        """
        Checks for problematic images:
        - Too dark (mean < 20): face probably not visible
        - Too bright (mean > 235): overexposed
        - Low contrast (std < 10): almost uniform, no features
        """
        print("\n[Check 4] Pixel Range & Contrast")
        too_dark = []
        too_bright = []
        low_contrast = []

        for img_info in images:
            img = cv2.imread(img_info["filepath"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            mean_val = img.mean()
            std_val = img.std()

            if mean_val < 20:
                too_dark.append({
                    "path": img_info["filepath"],
                    "mean": mean_val,
                })
            elif mean_val > 235:
                too_bright.append({
                    "path": img_info["filepath"],
                    "mean": mean_val,
                })

            if std_val < 10:
                low_contrast.append({
                    "path": img_info["filepath"],
                    "std": std_val,
                })

        self.issues["too_dark"] = too_dark
        self.issues["too_bright"] = too_bright
        self.issues["low_contrast"] = low_contrast

        total_issues = len(too_dark) + len(too_bright) + len(low_contrast)

        if total_issues > 0:
            print(f"  ⚠️  WARNINGS:")
            if too_dark:
                print(f"      {len(too_dark)} images are very dark (mean < 20)")
            if too_bright:
                print(f"      {len(too_bright)} images are very bright (mean > 235)")
            if low_contrast:
                print(f"      {len(low_contrast)} images have very low contrast (std < 10)")
            print(f"      Consider removing these — they may hurt training.")
            self.all_passed = False
        else:
            print(f"  ✅ PASSED: All images have reasonable brightness and contrast")

    # --- Check 5: Duplicates ---

    def _check_5_duplicates(self, images: list):
        """
        Detects exact duplicate images using MD5 hashes
        and near-duplicates using pixel similarity.
        """
        print("\n[Check 5] Duplicate Detection")

        # Exact duplicates (same file content)
        hash_to_paths = defaultdict(list)

        for img_info in images:
            filepath = img_info["filepath"]
            try:
                with open(filepath, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                hash_to_paths[file_hash].append(filepath)
            except Exception:
                continue

        exact_dupes = {h: paths for h, paths in hash_to_paths.items()
                       if len(paths) > 1}

        # Near-duplicates (visually very similar)
        # Compare pixel values — expensive but thorough
        near_dupes = []
        print("  Checking for near-duplicates (this may take a moment)...")

        image_data = []
        for img_info in images:
            img = cv2.imread(img_info["filepath"], cv2.IMREAD_GRAYSCALE)
            if img is not None and img.shape == (EXPECTED_SIZE, EXPECTED_SIZE):
                image_data.append({
                    "path": img_info["filepath"],
                    "pixels": img.flatten().astype(np.float32),
                })

        # Only check within same emotion to keep it manageable
        emotion_groups = defaultdict(list)
        for item in image_data:
            emotion = os.path.basename(os.path.dirname(item["path"]))
            emotion_groups[emotion].append(item)

        for emotion, group in emotion_groups.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    # Mean Absolute Error between pixel values
                    mae = np.mean(np.abs(group[i]["pixels"] - group[j]["pixels"]))
                    if mae < 3.0:   # very similar
                        near_dupes.append((group[i]["path"], group[j]["path"], mae))

        total_dupes = sum(len(paths) - 1 for paths in exact_dupes.values())

        if exact_dupes or near_dupes:
            self.all_passed = False

            if exact_dupes:
                print(f"  ❌ Found {total_dupes} exact duplicates "
                      f"across {len(exact_dupes)} groups")
                for h, paths in list(exact_dupes.items())[:3]:
                    print(f"      Group:")
                    for p in paths:
                        print(f"        {p}")

                if self.remove_duplicates:
                    removed = 0
                    for h, paths in exact_dupes.items():
                        # Keep the first, remove the rest
                        for dup_path in paths[1:]:
                            os.remove(dup_path)
                            removed += 1
                    print(f"      Removed {removed} exact duplicates")

            if near_dupes:
                print(f"  ⚠️  Found {len(near_dupes)} near-duplicate pairs")
                for p1, p2, mae in near_dupes[:3]:
                    print(f"      MAE={mae:.1f}: {os.path.basename(p1)} ↔ "
                          f"{os.path.basename(p2)}")

            self.issues["duplicates"] = {
                "exact": total_dupes,
                "near": len(near_dupes),
            }
        else:
            print(f"  ✅ PASSED: No duplicates detected")

    # --- Check 6: Class Balance ---

    def _check_6_class_balance(self, images: list):
        """
        Analyzes class distribution and flags severe imbalance.
        """
        print("\n[Check 6] Class Balance")

        counts = Counter(img["emotion"] for img in images)
        total = len(images)

        print(f"\n  {'Emotion':>10s} | {'Count':>6s} | {'Percent':>7s} | Distribution")
        print(f"  {'-'*55}")

        max_count = max(counts.values()) if counts else 1

        for emotion in EMOTIONS:
            count = counts.get(emotion, 0)
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(count / max_count * 30) if max_count > 0 else ""
            print(f"  {emotion:>10s} | {count:>6d} | {pct:>6.1f}% | {bar}")

        # Check imbalance ratio
        if counts:
            max_c = max(counts.values())
            min_c = min(counts.values()) if len(counts) == len(EMOTIONS) else 0

            if min_c == 0:
                print(f"\n  ❌ CRITICAL: Some emotion classes have ZERO images!")
                self.all_passed = False
            elif max_c / max(min_c, 1) > 5:
                print(f"\n  ❌ SEVERE IMBALANCE: Largest class is "
                      f"{max_c/min_c:.1f}x the smallest")
                print(f"      This will cause the model to ignore minority classes.")
                print(f"      Solutions:")
                print(f"        1. Capture more data for small classes")
                print(f"        2. Use class weights in training (handled in Phase 3)")
                print(f"        3. Apply more augmentation to small classes")
                self.all_passed = False
            elif max_c / max(min_c, 1) > 2:
                print(f"\n  ⚠️  MODERATE IMBALANCE: Ratio = {max_c/min_c:.1f}x")
                print(f"      Class weights in Phase 3 should handle this.")
            else:
                print(f"\n  ✅ PASSED: Classes are reasonably balanced "
                      f"(ratio: {max_c/max(min_c,1):.1f}x)")

    # --- Check 7: Filename Format ---

    def _check_7_filename_format(self, images: list):
        """
        Verifies filenames follow the expected convention:
            subject_emotion_round_frame.jpg
        """
        print("\n[Check 7] Filename Format")
        import re

        pattern = re.compile(
            r"^[a-zA-Z]+_[a-zA-Z]+_\d+_\d+\.(jpg|jpeg|png)$",
            re.IGNORECASE
        )

        bad_names = []
        for img_info in images:
            if not pattern.match(img_info["filename"]):
                bad_names.append(img_info["filename"])

        if bad_names:
            self.issues["bad_filename"] = bad_names
            print(f"  ⚠️  WARNING: {len(bad_names)} files don't match expected format")
            print(f"      Expected: subject_emotion_round_frame.jpg")
            for name in bad_names[:5]:
                print(f"        {name}")
            if len(bad_names) > 5:
                print(f"        ... and {len(bad_names) - 5} more")
            print(f"      This may break subject-aware splitting in Phase 3.")
        else:
            print(f"  ✅ PASSED: All filenames follow naming convention")

    # --- Check 8: Minimum Counts ---

    def _check_8_minimum_counts(self, images: list):
        """
        Checks if there are enough images per class for meaningful training.
        """
        print("\n[Check 8] Minimum Viable Dataset Size")

        counts = Counter(img["emotion"] for img in images)
        total = len(images)

        below_minimum = [e for e in EMOTIONS if counts.get(e, 0) < MIN_IMAGES_PER_CLASS]
        below_recommended = [e for e in EMOTIONS
                             if MIN_IMAGES_PER_CLASS <= counts.get(e, 0) < RECOMMENDED_PER_CLASS]

        if below_minimum:
            self.all_passed = False
            print(f"  ❌ CRITICAL: These classes have fewer than "
                  f"{MIN_IMAGES_PER_CLASS} images:")
            for e in below_minimum:
                print(f"      {e}: {counts.get(e, 0)} images")
            print(f"      Training will likely fail or produce a useless model.")
            print(f"      Capture more data for these emotions.")
        elif below_recommended:
            print(f"  ⚠️  WARNING: These classes have fewer than "
                  f"{RECOMMENDED_PER_CLASS} images (recommended):")
            for e in below_recommended:
                print(f"      {e}: {counts.get(e, 0)} images")
            print(f"      Training will work but accuracy may suffer.")
        else:
            print(f"  ✅ PASSED: All classes have ≥{RECOMMENDED_PER_CLASS} images")

        print(f"\n  Total dataset: {total} images across {len(counts)} classes")
        if total < 500:
            print(f"  ⚠️  Overall dataset is small. Consider supplementing with FER2013.")

    # --- Final Report ---

    def _print_report(self):
        """
        Prints the final QA report.
        """
        print("\n" + "=" * 60)
        if self.all_passed:
            print("✅ ALL CHECKS PASSED — Dataset is ready for Phase 3")
        else:
            print("❌ ISSUES FOUND — Review the warnings above")

            total_issues = sum(
                len(v) if isinstance(v, list) else (1 if v else 0)
                for v in self.issues.values()
            )
            print(f"\nTotal issues: {total_issues}")
            print(f"\nRecommended actions:")

            if self.issues["load_failed"]:
                print(f"  1. Remove {len(self.issues['load_failed'])} "
                      f"corrupted files")
            if self.issues["wrong_size"]:
                print(f"  2. Rerun with --fix to resize images")
            if self.issues["too_dark"] or self.issues["too_bright"]:
                dark_bright = (len(self.issues['too_dark']) +
                               len(self.issues['too_bright']))
                print(f"  3. Review {dark_bright} dark/bright images")
            if self.issues["duplicates"]:
                print(f"  4. Rerun with --remove_duplicates")

        print("=" * 60)


def main():
    args = parse_args()
    checker = QualityChecker(args)
    checker.run_all_checks()


if __name__ == "__main__":
    main()