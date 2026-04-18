"""
Face detection, alignment, and cropping pipeline.

Takes raw webcam frames from data/raw/ and produces
aligned, cropped, grayscale 48x48 faces in data/processed/.

Uses MediaPipe Face Detection + Face Mesh for landmark-based alignment.

Usage:
    python src/detect_and_crop.py
    python src/detect_and_crop.py --raw_dir data/raw --output_dir data/processed --size 48

Pipeline per image:
    1. Detect face bounding box (MediaPipe)
    2. Detect facial landmarks (MediaPipe Face Mesh)
    3. Align face using eye positions (affine transform)
    4. Crop to face region with margin
    5. Resize to target size (48x48)
    6. Convert to grayscale
    7. Save to emotion-labeled folder
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp


def parse_args():
    parser = argparse.ArgumentParser(description="Detect and crop faces")

    parser.add_argument("--raw_dir", type=str, default="data/raw",
                        help="Root folder containing subject subfolders")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Output folder for cropped faces")
    parser.add_argument("--rejected_dir", type=str, default="data/rejected",
                        help="Folder for images that failed processing")
    parser.add_argument("--size", type=int, default=48,
                        help="Output face size (square)")
    parser.add_argument("--margin", type=float, default=0.3,
                        help="Margin around face as fraction of face size")
    parser.add_argument("--min_confidence", type=float, default=0.7,
                        help="Minimum face detection confidence")

    return parser.parse_args()


class FaceProcessor:
    """
    Handles face detection, alignment, and cropping for all images.
    """

    # MediaPipe Face Mesh landmark indices
    # These specific indices correspond to eye centers
    LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]
    RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 373, 374, 380]

    def __init__(self, args):
        self.raw_dir = args.raw_dir
        self.output_dir = args.output_dir
        self.rejected_dir = args.rejected_dir
        self.target_size = args.size
        self.margin = args.margin
        self.min_confidence = args.min_confidence

        # Initialize MediaPipe
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode
        )
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self.landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path="face_landmarker.task"
                ),
                running_mode=RunningMode.IMAGE,
                num_faces=1
            )
        )

        # Statistics
        self.stats = {
            "total": 0,
            "success": 0,
            "no_face": 0,
            "low_confidence": 0,
            "alignment_failed": 0,
            "file_error": 0,
        }
        self.per_emotion_stats = defaultdict(int)

    def process_all(self):
        """
        Processes all images in all subject folders.
        """
        print("=" * 60)
        print("FACE DETECTION AND CROPPING PIPELINE")
        print("=" * 60)

        # Scan for all images
        image_tasks = self._scan_for_images()
        print(f"\nFound {len(image_tasks)} images to process.")

        if len(image_tasks) == 0:
            print(f"No images found in {self.raw_dir}/")
            print("Expected structure: data/raw/subjectname/subjectname_emotion_round_frame.jpg")
            return

        # Create output directories
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        for emotion in emotions:
            os.makedirs(os.path.join(self.output_dir, emotion), exist_ok=True)
        os.makedirs(self.rejected_dir, exist_ok=True)

        # Process with MediaPipe
        for i, task in enumerate(image_tasks):
          if (i + 1) % 100 == 0 or i == 0:
              print(f"  Processing {i + 1}/{len(image_tasks)}...")

          self._process_single_image(task)

          self._print_stats()

    def _scan_for_images(self) -> list:
        """
        Scans raw_dir for all images and parses their metadata from filenames.

        Expected filename format: subject_emotion_round_frame.jpg
        Example: alice_happy_01_003.jpg

        Returns list of dicts with parsed metadata.
        """
        tasks = []
        pattern = re.compile(
            r"^([a-zA-Z]+)_(\w+)_(\d+)_(\d+)\.(jpg|jpeg|png|bmp)$",
            re.IGNORECASE
        )

        for subject_dir in sorted(os.listdir(self.raw_dir)):
            subject_path = os.path.join(self.raw_dir, subject_dir)

            if not os.path.isdir(subject_path):
                continue

            for filename in sorted(os.listdir(subject_path)):
                match = pattern.match(filename)

                if match:
                    subject = match.group(1).lower()
                    emotion = match.group(2).lower()
                    round_num = match.group(3)
                    frame_num = match.group(4)

                    tasks.append({
                        "filepath": os.path.join(subject_path, filename),
                        "filename": filename,
                        "subject": subject,
                        "emotion": emotion,
                        "round": round_num,
                        "frame": frame_num,
                    })
                else:
                    print(f"  WARNING: Skipping unrecognized filename: {filename}")
                    print(f"           Expected format: subject_emotion_round_frame.jpg")

        return tasks

    def _process_single_image(self, task: dict) -> bool:
        """
        Processes a single image through the full pipeline.

        Returns True if successful, False otherwise.
        """
        self.stats["total"] += 1
        filepath = task["filepath"]
        emotion = task["emotion"]

        # --- Load image ---
        img = cv2.imread(filepath)
        if img is None:
            self.stats["file_error"] += 1
            self._reject(filepath, "file_error")
            return False

        # --- Run face mesh ---
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )

        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            self.stats["no_face"] += 1
            self._reject(filepath, "no_face")
            return False

        landmarks = result.face_landmarks[0]
        h, w = img.shape[:2]

        # --- Extract eye centers for alignment ---
        try:
            left_eye = self._get_eye_center(landmarks, self.LEFT_EYE_INDICES, w, h)
            right_eye = self._get_eye_center(landmarks, self.RIGHT_EYE_INDICES, w, h)
        except Exception:
            self.stats["alignment_failed"] += 1
            self._reject(filepath, "alignment_failed")
            return False

        # --- Align face ---
        aligned = self._align_face(img, left_eye, right_eye)

        if aligned is None:
            self.stats["alignment_failed"] += 1
            self._reject(filepath, "alignment_failed")
            return False

        # --- Re-detect face in aligned image to get tight crop ---
        # After alignment the face is upright, making detection more reliable
        aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

        mp_aligned = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=aligned_rgb
        )

        results_aligned = self.landmarker.detect(mp_aligned)

        if results_aligned.face_landmarks:
            face_crop = self._crop_face_from_landmarks(
                aligned,
                results_aligned.multi_face_landmarks[0]
            )
        else:
            # Fallback: center crop the aligned image
            face_crop = self._center_crop(aligned)

        # --- Resize to target ---
        face_resized = cv2.resize(
            face_crop,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA     # best for downscaling
        )

        # --- Convert to grayscale ---
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        # --- Save ---
        output_filename = (
            f"{task['subject']}_{emotion}_{task['round']}_{task['frame']}.jpg"
        )
        output_path = os.path.join(self.output_dir, emotion, output_filename)
        cv2.imwrite(output_path, face_gray)

        self.stats["success"] += 1
        self.per_emotion_stats[emotion] += 1
        return True

    def _get_eye_center(self, landmarks, indices: list, w: int, h: int) -> tuple:
      points = []
      for idx in indices:
          lm = landmarks[idx]
          points.append((lm.x * w, lm.y * h))

      center_x = sum(p[0] for p in points) / len(points)
      center_y = sum(p[1] for p in points) / len(points)

      return (center_x, center_y)

    def _align_face(self, img: np.ndarray,
                    left_eye: tuple, right_eye: tuple) -> np.ndarray:
        """
        Aligns the face so both eyes are on a horizontal line.

        This is critical because:
        - Head tilts add noise that isn't related to emotion
        - The model should focus on expression, not head angle
        - Consistent alignment makes features more comparable

        Process:
        1. Compute the angle between the eyes
        2. Rotate the image to make eyes horizontal
        3. Scale so eye distance is consistent
        """
        # Compute angle between eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Compute eye distance
        eye_distance = np.sqrt(dx ** 2 + dy ** 2)

        if eye_distance < 10:
            return None     # eyes too close — probably a bad detection

        # Desired eye distance in output (roughly 40% of image width)
        desired_eye_dist = self.target_size * 2.5
        scale = desired_eye_dist / eye_distance

        # Center point between eyes
        eyes_center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2,
        )

        # Rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(
            eyes_center, angle, scale
        )

        # Adjust translation so face is centered in output
        output_size = int(self.target_size * 4)     # generous canvas
        rotation_matrix[0, 2] += (output_size / 2 - eyes_center[0])
        rotation_matrix[1, 2] += (output_size / 2 - eyes_center[1])

        # Apply transformation
        aligned = cv2.warpAffine(
            img,
            rotation_matrix,
            (output_size, output_size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )

        return aligned

    def _crop_face_from_landmarks(self, img: np.ndarray,
                                  landmarks) -> np.ndarray:
        """
        Crops the face region using all landmark points to determine bounds.
        Adds a margin around the face.
        """
        h, w = img.shape[:2]

        # Get bounding box from all landmarks
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        face_w = x_max - x_min
        face_h = y_max - y_min

        # Add margin
        margin_x = face_w * self.margin
        margin_y = face_h * self.margin

        # Compute crop coordinates with boundary checking
        crop_x1 = max(0, int(x_min - margin_x))
        crop_y1 = max(0, int(y_min - margin_y))
        crop_x2 = min(w, int(x_max + margin_x))
        crop_y2 = min(h, int(y_max + margin_y))

        # Make square (use larger dimension)
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        size = max(crop_w, crop_h)

        # Center the square crop
        center_x = (crop_x1 + crop_x2) // 2
        center_y = (crop_y1 + crop_y2) // 2

        x1 = max(0, center_x - size // 2)
        y1 = max(0, center_y - size // 2)
        x2 = min(w, x1 + size)
        y2 = min(h, y1 + size)

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            return self._center_crop(img)

        return crop

    def _center_crop(self, img: np.ndarray) -> np.ndarray:
        """
        Fallback: takes a center crop of the image.
        Used when face re-detection fails after alignment.
        """
        h, w = img.shape[:2]
        size = min(h, w)
        y1 = (h - size) // 2
        x1 = (w - size) // 2
        return img[y1:y1 + size, x1:x1 + size]

    def _reject(self, filepath: str, reason: str):
        """
        Moves a failed image to the rejected folder with the reason.
        """
        filename = os.path.basename(filepath)
        reason_dir = os.path.join(self.rejected_dir, reason)
        os.makedirs(reason_dir, exist_ok=True)

        dest = os.path.join(reason_dir, filename)
        # Copy instead of move so raw data is preserved
        img = cv2.imread(filepath)
        if img is not None:
            cv2.imwrite(dest, img)

    def _print_stats(self):
        """
        Prints processing statistics.
        """
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)

        print(f"\nOverall:")
        print(f"  Total images:       {self.stats['total']:>5d}")
        print(f"  Successful:         {self.stats['success']:>5d}  "
              f"({self.stats['success']/max(self.stats['total'],1):.1%})")
        print(f"  No face detected:   {self.stats['no_face']:>5d}")
        print(f"  Low confidence:     {self.stats['low_confidence']:>5d}")
        print(f"  Alignment failed:   {self.stats['alignment_failed']:>5d}")
        print(f"  File errors:        {self.stats['file_error']:>5d}")

        print(f"\nPer emotion:")
        for emotion in sorted(self.per_emotion_stats.keys()):
            count = self.per_emotion_stats[emotion]
            print(f"  {emotion:>10s}: {count:>5d}")

        # Warnings
        rejected_total = (self.stats["no_face"] +
                          self.stats["alignment_failed"] +
                          self.stats["file_error"])

        if rejected_total > self.stats["total"] * 0.2:
            print(f"\n⚠️  WARNING: {rejected_total/self.stats['total']:.1%} "
                  f"of images were rejected.")
            print("    Check data/rejected/ to see why.")
            print("    Common causes:")
            print("      - Face not visible (turned away, too close)")
            print("      - Poor lighting (face in shadow)")
            print("      - Motion blur (moving during capture)")

        if self.stats["no_face"] > self.stats["total"] * 0.1:
            print(f"\n⚠️  HIGH 'no_face' RATE: {self.stats['no_face']} images")
            print("    Try lowering --min_confidence (default 0.7)")
            print("    Or check that subjects face the camera directly")


def main():
    args = parse_args()
    processor = FaceProcessor(args)
    processor.process_all()


if __name__ == "__main__":
    main()