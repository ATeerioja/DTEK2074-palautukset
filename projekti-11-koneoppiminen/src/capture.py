"""
Webcam capture session for emotion data collection.

Guides the subject through a structured recording session:
    - Displays emotion prompts on screen
    - Captures frames at a configurable rate
    - Saves with structured filenames for downstream processing

Usage:
    python src/capture.py --subject alice
    python src/capture.py --subject bob --fps 5 --hold_seconds 4 --rounds 3

Naming convention:
    {subject}_{emotion}_{round}_{frame}.jpg
    alice_happy_01_003.jpg

    This naming is critical — the datamodule in Phase 3
    uses the subject prefix for subject-aware train/test splitting.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


# =====================
# CONFIGURATION
# =====================

EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]

# Tips displayed to help subjects produce convincing expressions
EMOTION_TIPS = {
    "happy":    "Think of something funny. Smile with your eyes, not just your mouth.",
    "sad":      "Think of something disappointing. Let your face droop. Lower your gaze.",
    "angry":    "Think of something infuriating. Furrow your brows. Clench your jaw.",
    "surprise": "Imagine someone jumped out at you. Raise eyebrows. Open mouth wide.",
    "fear":     "Imagine hearing a loud crash at night. Wide eyes. Tense face.",
    "disgust":  "Imagine smelling something awful. Wrinkle your nose. Raise upper lip.",
    "neutral":  "Relax your face completely. No expression. Resting face.",
}

# Colors for the UI overlay (BGR format for OpenCV)
COLORS = {
    "bg":       (40, 40, 40),
    "text":     (255, 255, 255),
    "prompt":   (0, 255, 128),
    "warning":  (0, 128, 255),
    "countdown": (0, 200, 255),
    "recording": (0, 0, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Capture emotion data from webcam")

    parser.add_argument("--subject", type=str, required=True,
                        help="Subject name (e.g. alice). Used in filenames for splitting.")
    parser.add_argument("--output_dir", type=str, default="data/raw",
                        help="Root output directory")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (0 = default webcam)")
    parser.add_argument("--fps", type=int, default=5,
                        help="Frames to capture per second during recording")
    parser.add_argument("--hold_seconds", type=int, default=4,
                        help="How long to hold each expression")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of rounds through all emotions")
    parser.add_argument("--prep_seconds", type=int, default=3,
                        help="Countdown before each capture begins")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480],
                        help="Camera resolution (width height)")
    parser.add_argument("--vary_intensity", action="store_true",
                        help="Ask for mild/strong versions of each emotion")

    return parser.parse_args()


class CaptureSession:
    """
    Manages a complete data capture session for one subject.
    """

    def __init__(self, args):
        self.subject = args.subject.lower().strip()
        self.output_dir = os.path.join(args.output_dir, self.subject)
        self.camera_idx = args.camera
        self.fps = args.fps
        self.hold_seconds = args.hold_seconds
        self.rounds = args.rounds
        self.prep_seconds = args.prep_seconds
        self.resolution = tuple(args.resolution)
        self.vary_intensity = args.vary_intensity

        # Validate subject name — must be alphabetic for filename parsing
        if not self.subject.isalpha():
            raise ValueError(
                f"Subject name must be alphabetic only. Got: '{self.subject}'"
                "\nThis is required for subject-aware train/test splitting."
            )

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Build the session plan
        self.session_plan = self._build_session_plan()

        # Track statistics
        self.stats = {emotion: 0 for emotion in EMOTIONS}
        self.total_captured = 0

    def _build_session_plan(self) -> list:
        """
        Builds a randomized capture plan.

        Returns a list of dicts, each describing one capture segment:
            {"emotion": "happy", "round": 1, "intensity": "normal"}

        Randomizing the order prevents the subject from falling into
        a rhythm and producing less genuine expressions.
        """
        plan = []

        for round_num in range(1, self.rounds + 1):
            round_emotions = EMOTIONS.copy()
            np.random.shuffle(round_emotions)

            for emotion in round_emotions:
                if self.vary_intensity and emotion != "neutral":
                    # Capture both mild and strong versions
                    plan.append({
                        "emotion": emotion,
                        "round": round_num,
                        "intensity": "mild",
                    })
                    plan.append({
                        "emotion": emotion,
                        "round": round_num,
                        "intensity": "strong",
                    })
                else:
                    plan.append({
                        "emotion": emotion,
                        "round": round_num,
                        "intensity": "normal",
                    })

        return plan

    def run(self):
        """
        Runs the complete capture session.
        """
        # --- Open camera ---
        cap = cv2.VideoCapture(self.camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.camera_idx}. "
                "Try a different --camera index."
            )

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {actual_w}x{actual_h}")

        try:
            # --- Welcome screen ---
            self._show_welcome(cap, actual_w, actual_h)

            # --- Run through each segment ---
            total_segments = len(self.session_plan)

            for seg_idx, segment in enumerate(self.session_plan):
                emotion = segment["emotion"]
                round_num = segment["round"]
                intensity = segment["intensity"]

                print(f"\n[{seg_idx + 1}/{total_segments}] "
                      f"Round {round_num}: {emotion} ({intensity})")

                # Show preparation screen with countdown
                skipped = self._show_prep_screen(
                    cap, emotion, intensity, seg_idx, total_segments,
                    actual_w, actual_h
                )

                if skipped:
                    print(f"  Skipped by user.")
                    continue

                # Capture frames
                n_captured = self._capture_segment(
                    cap, emotion, round_num, intensity, actual_w, actual_h
                )

                self.stats[emotion] += n_captured
                self.total_captured += n_captured
                print(f"  Captured {n_captured} frames.")

            # --- Summary screen ---
            self._show_summary(cap, actual_w, actual_h)

        except KeyboardInterrupt:
            print("\n\nSession interrupted by user.")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_final_stats()

    def _show_welcome(self, cap, w, h):
        """
        Shows a welcome screen with instructions.
        Press SPACE to begin.
        """
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), COLORS["bg"], -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            lines = [
                f"EMOTION CAPTURE SESSION",
                f"",
                f"Subject: {self.subject}",
                f"Emotions: {len(EMOTIONS)}",
                f"Rounds: {self.rounds}",
                f"Total segments: {len(self.session_plan)}",
                f"Frames per segment: ~{self.fps * self.hold_seconds}",
                f"",
                f"INSTRUCTIONS:",
                f"1. You will see an emotion prompt",
                f"2. A countdown gives you time to prepare",
                f"3. When recording starts, hold the expression",
                f"4. Try to vary head angle slightly during each hold",
                f"5. Press 'S' during countdown to skip an emotion",
                f"6. Press 'Q' at any time to quit",
                f"",
                f"Press SPACE to begin",
            ]

            y_start = 30
            for i, line in enumerate(lines):
                color = COLORS["prompt"] if i == 0 else COLORS["text"]
                scale = 0.8 if i == 0 else 0.55
                cv2.putText(frame, line, (30, y_start + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

            cv2.imshow("Emotion Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                return
            elif key == ord('q'):
                raise KeyboardInterrupt

    def _show_prep_screen(self, cap, emotion, intensity, seg_idx, total,
                          w, h) -> bool:
        """
        Shows the emotion prompt with a countdown.

        Returns True if the user skipped this segment (pressed 'S').
        """
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            elapsed = time.time() - start_time
            remaining = self.prep_seconds - elapsed

            if remaining <= 0:
                return False    # countdown done, proceed to capture

            # Semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 120), COLORS["bg"], -1)
            frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

            # Progress
            progress_text = f"[{seg_idx + 1}/{total}]  Round {seg_idx // len(EMOTIONS) + 1}"
            cv2.putText(frame, progress_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)

            # Emotion prompt
            intensity_label = f" ({intensity})" if intensity != "normal" else ""
            prompt = f"GET READY: {emotion.upper()}{intensity_label}"
            cv2.putText(frame, prompt, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS["prompt"], 2)

            # Tip
            tip = EMOTION_TIPS.get(emotion, "")
            cv2.putText(frame, tip, (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text"], 1)

            # Countdown
            countdown_text = f"Starting in: {remaining:.1f}s"
            cv2.putText(frame, countdown_text, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["countdown"], 2)

            # Skip instruction
            cv2.putText(frame, "Press S to skip | Q to quit",
                        (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["warning"], 1)

            cv2.imshow("Emotion Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                return True     # skip this segment
            elif key == ord('q'):
                raise KeyboardInterrupt

    def _capture_segment(self, cap, emotion, round_num, intensity, w, h) -> int:
        """
        Captures frames for one emotion segment.

        Returns the number of frames successfully saved.
        """
        frame_interval = 1.0 / self.fps
        n_frames = self.fps * self.hold_seconds
        captured = 0
        start_time = time.time()

        for frame_idx in range(n_frames):
            # Wait for next capture time
            target_time = start_time + (frame_idx * frame_interval)
            now = time.time()
            if now < target_time:
                # Keep reading frames to drain the buffer, show live preview
                while time.time() < target_time:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    self._draw_recording_overlay(
                        frame, emotion, intensity, captured, n_frames,
                        time.time() - start_time, self.hold_seconds, w, h
                    )
                    cv2.imshow("Emotion Capture", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        raise KeyboardInterrupt

            # Capture the actual frame
            ret, frame = cap.read()
            if not ret:
                continue

            # Build filename
            # Format: subject_emotion_round_frame.jpg
            # Example: alice_happy_01_003.jpg
            filename = (
                f"{self.subject}_{emotion}_{round_num:02d}_{frame_idx:03d}.jpg"
            )
            filepath = os.path.join(self.output_dir, filename)

            # Save the raw, uncropped frame
            cv2.imwrite(filepath, frame)
            captured += 1

            # Show recording overlay
            self._draw_recording_overlay(
                frame, emotion, intensity, captured, n_frames,
                time.time() - start_time, self.hold_seconds, w, h
            )
            cv2.imshow("Emotion Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                raise KeyboardInterrupt

        return captured

    def _draw_recording_overlay(self, frame, emotion, intensity,
                                captured, total, elapsed, duration, w, h):
        """
        Draws the recording indicator and progress on the frame.
        """
        # Recording indicator (red dot)
        cv2.circle(frame, (30, 30), 12, COLORS["recording"], -1)
        cv2.putText(frame, "REC", (50, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["recording"], 2)

        # Emotion label
        intensity_label = f" ({intensity})" if intensity != "normal" else ""
        cv2.putText(frame, f"{emotion.upper()}{intensity_label}",
                    (120, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["prompt"], 2)

        # Frame counter
        cv2.putText(frame, f"Frame: {captured}/{total}",
                    (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)

        # Time progress bar
        progress = min(elapsed / duration, 1.0)
        bar_w = w - 20
        cv2.rectangle(frame, (10, h - 20), (10 + bar_w, h - 10),
                      COLORS["text"], 1)
        cv2.rectangle(frame, (10, h - 20),
                      (10 + int(bar_w * progress), h - 10),
                      COLORS["prompt"], -1)

    def _show_summary(self, cap, w, h):
        """
        Shows a summary of what was captured. Press SPACE to close.
        """
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), COLORS["bg"], -1)
            frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

            lines = [
                "SESSION COMPLETE",
                "",
                f"Subject: {self.subject}",
                f"Total frames captured: {self.total_captured}",
                "",
                "Per emotion:",
            ]

            for emotion in EMOTIONS:
                count = self.stats[emotion]
                bar = "#" * (count // 3)
                lines.append(f"  {emotion:>10s}: {count:>4d}  {bar}")

            lines.extend([
                "",
                f"Saved to: {self.output_dir}",
                "",
                "NEXT STEPS:",
                "1. Run detect_and_crop.py to extract faces",
                "2. Run verify.py to check labels",
                "3. Run qa.py for quality checks",
                "",
                "Press SPACE to close",
            ])

            for i, line in enumerate(lines):
                color = COLORS["prompt"] if i == 0 else COLORS["text"]
                scale = 0.7 if i == 0 else 0.5
                cv2.putText(frame, line, (30, 30 + i * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

            cv2.imshow("Emotion Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                return

    def _print_final_stats(self):
        """
        Prints final statistics to the console.
        """
        print("\n" + "=" * 50)
        print("CAPTURE SESSION SUMMARY")
        print("=" * 50)
        print(f"Subject:        {self.subject}")
        print(f"Output:         {self.output_dir}")
        print(f"Total frames:   {self.total_captured}")
        print(f"\nPer emotion:")
        for emotion in EMOTIONS:
            count = self.stats[emotion]
            print(f"  {emotion:>10s}: {count:>4d}")

        expected = self.fps * self.hold_seconds * self.rounds
        print(f"\nExpected per emotion: ~{expected}")

        low_count = [e for e in EMOTIONS if self.stats[e] < expected * 0.5]
        if low_count:
            print(f"\n⚠️  LOW COUNT WARNING for: {', '.join(low_count)}")
            print("    Consider running additional capture rounds for these.")


def main():
    args = parse_args()
    session = CaptureSession(args)

    print(f"\nCapture session for: {session.subject}")
    print(f"Plan: {len(session.session_plan)} segments")
    print(f"Expected frames per segment: {args.fps * args.hold_seconds}")
    print(f"Total expected frames: "
          f"~{len(session.session_plan) * args.fps * args.hold_seconds}")

    session.run()


if __name__ == "__main__":
    main()