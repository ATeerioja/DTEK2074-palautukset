"""
Manual label verification tool.

Displays each processed face image and lets you:
    - CONFIRM the label (press SPACE or ENTER)
    - RELABEL it (press the first letter of the correct emotion)
    - REJECT it (press R)
    - GO BACK to the previous image (press B)
    - QUIT and save progress (press Q)

Progress is saved automatically so you can resume later.

Usage:
    python src/verify.py
    python src/verify.py --data_dir data/processed --start_from 500
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Key mappings for relabeling
# Press the key to assign that emotion
EMOTION_KEYS = {
    ord('a'): "angry",
    ord('d'): "disgust",
    ord('f'): "fear",
    ord('h'): "happy",
    ord('n'): "neutral",
    ord('s'): "sad",
    ord('u'): "surprise",     # s is taken by sad, use u for sUrprise
}


def parse_args():
    parser = argparse.ArgumentParser(description="Verify emotion labels")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--rejected_dir", type=str, default="data/rejected/manual")
    parser.add_argument("--progress_file", type=str, default="data/verify_progress.json")
    parser.add_argument("--start_from", type=int, default=0,
                        help="Image index to start from (overrides saved progress)")
    parser.add_argument("--display_size", type=int, default=300,
                        help="Display size for images (pixels)")
    return parser.parse_args()


class LabelVerifier:
    """
    Interactive tool for verifying and correcting emotion labels.
    """

    def __init__(self, args):
        self.data_dir = args.data_dir
        self.rejected_dir = args.rejected_dir
        self.progress_file = args.progress_file
        self.display_size = args.display_size

        os.makedirs(self.rejected_dir, exist_ok=True)

        # Load all images
        self.images = self._scan_images()

        # Load or initialize progress
        self.progress = self._load_progress()

        # Handle start_from override
        if args.start_from > 0:
            self.progress["current_index"] = args.start_from

        # Statistics
        self.session_stats = {
            "confirmed": 0,
            "relabeled": 0,
            "rejected": 0,
            "reviewed": 0,
        }

    def _scan_images(self) -> list:
        """
        Scans processed directory and returns list of image info dicts.
        """
        images = []

        for emotion in EMOTIONS:
            emotion_dir = os.path.join(self.data_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue

            for filename in sorted(os.listdir(emotion_dir)):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append({
                        "filepath": os.path.join(emotion_dir, filename),
                        "filename": filename,
                        "emotion": emotion,
                    })

        return images

    def _load_progress(self) -> dict:
        """
        Loads verification progress from file.
        """
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as f:
                progress = json.load(f)
            print(f"Resuming from image {progress['current_index']} "
                  f"/ {len(self.images)}")
            return progress

        return {
            "current_index": 0,
            "total_confirmed": 0,
            "total_relabeled": 0,
            "total_rejected": 0,
            "relabel_log": [],      # tracks what was changed
        }

    def _save_progress(self):
        """
        Saves verification progress to file.
        """
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def run(self):
        """
        Main verification loop.
        """
        total = len(self.images)
        idx = self.progress["current_index"]

        print(f"\nTotal images: {total}")
        print(f"Starting at:  {idx}")
        print(f"\nControls:")
        print(f"  SPACE/ENTER  = Confirm label is correct")
        print(f"  A            = Relabel as angry")
        print(f"  D            = Relabel as disgust")
        print(f"  F            = Relabel as fear")
        print(f"  H            = Relabel as happy")
        print(f"  N            = Relabel as neutral")
        print(f"  S            = Relabel as sad")
        print(f"  U            = Relabel as surprise")
        print(f"  R            = Reject (bad image)")
        print(f"  B            = Go back one image")
        print(f"  Q            = Quit and save progress")

        while idx < total:
            img_info = self.images[idx]

            # Load and display
            img = cv2.imread(img_info["filepath"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Could not load: {img_info['filepath']}")
                idx += 1
                continue

            # Create display canvas
            display = self._create_display(img, img_info, idx, total)
            cv2.imshow("Label Verification", display)

            # Wait for keypress
            key = cv2.waitKey(0) & 0xFF

            # --- Handle input ---
            if key == ord(' ') or key == 13:    # SPACE or ENTER
                # Confirm — do nothing, move forward
                self.session_stats["confirmed"] += 1
                self.progress["total_confirmed"] += 1
                idx += 1

            elif key in EMOTION_KEYS:
                # Relabel
                new_emotion = EMOTION_KEYS[key]
                if new_emotion != img_info["emotion"]:
                    self._relabel(img_info, new_emotion)
                    self.session_stats["relabeled"] += 1
                    self.progress["total_relabeled"] += 1
                idx += 1

            elif key == ord('r'):
                # Reject
                self._reject_image(img_info)
                self.session_stats["rejected"] += 1
                self.progress["total_rejected"] += 1
                idx += 1

            elif key == ord('b'):
                # Go back
                if idx > 0:
                    idx -= 1

            elif key == ord('q'):
                # Quit
                break

            # Update and save progress
            self.session_stats["reviewed"] += 1
            self.progress["current_index"] = idx
            self._save_progress()

        cv2.destroyAllWindows()
        self._print_stats()

    def _create_display(self, img: np.ndarray, img_info: dict,
                        idx: int, total: int) -> np.ndarray:
        """
        Creates a display image with the face and label information.
        """
        # Scale up the small 48x48 image for visibility
        display_img = cv2.resize(
            img,
            (self.display_size, self.display_size),
            interpolation=cv2.INTER_NEAREST     # no smoothing — see actual pixels
        )

        # Convert to BGR for colored annotations
        display = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

        # Add info panel on the right
        panel_w = 300
        panel = np.zeros((self.display_size, panel_w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)

        lines = [
            f"Image {idx + 1} / {total}",
            f"",
            f"Current label: {img_info['emotion'].upper()}",
            f"File: {img_info['filename']}",
            f"",
            f"--- Controls ---",
            f"SPACE  = Confirm",
            f"A = angry   D = disgust",
            f"F = fear    H = happy",
            f"N = neutral S = sad",
            f"U = surprise",
            f"R = Reject  B = Back",
            f"Q = Quit",
            f"",
            f"Session:",
            f"  Confirmed: {self.session_stats['confirmed']}",
            f"  Relabeled: {self.session_stats['relabeled']}",
            f"  Rejected:  {self.session_stats['rejected']}",
        ]

        for i, line in enumerate(lines):
            color = (0, 255, 128) if i == 2 else (255, 255, 255)
            cv2.putText(panel, line, (10, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Combine image and panel side by side
        combined = np.hstack([display, panel])

        return combined

    def _relabel(self, img_info: dict, new_emotion: str):
        """
        Moves an image from its current emotion folder to the correct one.
        """
        old_path = img_info["filepath"]
        new_dir = os.path.join(self.data_dir, new_emotion)
        os.makedirs(new_dir, exist_ok=True)

        # Update filename to reflect new label
        old_filename = img_info["filename"]
        # Replace the emotion part in the filename
        parts = old_filename.split("_")
        if len(parts) >= 4:
            parts[1] = new_emotion
            new_filename = "_".join(parts)
        else:
            new_filename = old_filename

        new_path = os.path.join(new_dir, new_filename)

        # Move file
        os.rename(old_path, new_path)

        # Log the change
        self.progress["relabel_log"].append({
            "original": old_path,
            "new": new_path,
            "old_emotion": img_info["emotion"],
            "new_emotion": new_emotion,
        })

        print(f"  Relabeled: {img_info['emotion']} → {new_emotion} "
              f"({old_filename})")

    def _reject_image(self, img_info: dict):
        """
        Moves a bad image to the rejected folder.
        """
        old_path = img_info["filepath"]
        new_path = os.path.join(self.rejected_dir, img_info["filename"])
        os.rename(old_path, new_path)
        print(f"  Rejected: {img_info['filename']}")

    def _print_stats(self):
        """
        Prints session statistics.
        """
        print("\n" + "=" * 50)
        print("VERIFICATION SESSION SUMMARY")
        print("=" * 50)
        print(f"  Reviewed:  {self.session_stats['reviewed']}")
        print(f"  Confirmed: {self.session_stats['confirmed']}")
        print(f"  Relabeled: {self.session_stats['relabeled']}")
        print(f"  Rejected:  {self.session_stats['rejected']}")
        print(f"\nAll-time totals:")
        print(f"  Confirmed: {self.progress['total_confirmed']}")
        print(f"  Relabeled: {self.progress['total_relabeled']}")
        print(f"  Rejected:  {self.progress['total_rejected']}")
        print(f"\nProgress saved to: {self.progress_file}")

        if self.session_stats["relabeled"] > 0:
            print(f"\nRelabel log saved. {self.session_stats['relabeled']} "
                  f"images were moved to new emotion folders.")


def main():
    args = parse_args()
    verifier = LabelVerifier(args)
    verifier.run()


if __name__ == "__main__":
    main()