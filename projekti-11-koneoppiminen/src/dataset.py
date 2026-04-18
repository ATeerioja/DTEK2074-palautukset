"""
Custom Dataset for 48x48 grayscale emotion images.

Expects a folder structure like:
    data/processed/
        happy/
            img001.jpg
            img002.jpg
        sad/
            img001.jpg
        ...

Each subfolder name becomes the class label.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    """
    Loads grayscale face images from labeled folders.
    Returns (image_tensor, label_index) pairs.
    """

    # Class-level constants so they're accessible everywhere
    EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    IMG_SIZE = 48

    def __init__(self, image_paths: list, labels: list, augment: bool = False):
        """
        Args:
            image_paths: list of full file paths to images
            labels:      list of integer labels (same length as image_paths)
            augment:     whether to apply training augmentations
        """
        assert len(image_paths) == len(labels), \
            f"Mismatch: {len(image_paths)} images but {len(labels)} labels"

        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- Load image as grayscale ---
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(
                f"Could not load image: {self.image_paths[idx]}"
            )

        # --- Resize to 48x48 ---
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

        # --- Augmentation (training only) ---
        if self.augment:
            img = self._apply_augmentations(img)

        # --- Normalize to [0, 1] ---
        img = img.astype(np.float32) / 255.0

        # --- Convert to tensor: shape (1, 48, 48) ---
        # The 1 is the single grayscale channel
        tensor = torch.from_numpy(img).unsqueeze(0)

        # --- Label ---
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return tensor, label

    def _apply_augmentations(self, img: np.ndarray) -> np.ndarray:
        """
        Simple augmentations using only OpenCV and NumPy.
        Applied randomly during training to reduce overfitting.
        """
        # Random horizontal flip (50% chance)
        # Faces are roughly symmetric, so this is safe
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)

        # Random brightness adjustment
        # Simulates different lighting conditions
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.7, 1.3)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)

        # Random rotation (-15 to +15 degrees)
        # Simulates slight head tilts
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            center = (self.IMG_SIZE // 2, self.IMG_SIZE // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(
                img, matrix, (self.IMG_SIZE, self.IMG_SIZE),
                borderMode=cv2.BORDER_REFLECT
            )

        # Random noise injection
        # Makes the model robust to camera noise
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img

    @classmethod
    def from_folder(cls, root_dir: str, augment: bool = False):
        """
        Convenience method to build a dataset from a folder structure.

        Args:
            root_dir: path to folder containing emotion subfolders
            augment:  whether to augment

        Returns:
            EmotionDataset instance
        """
        image_paths = []
        labels = []

        for emotion_name in cls.EMOTIONS:
            emotion_dir = os.path.join(root_dir, emotion_name)

            if not os.path.isdir(emotion_dir):
                print(f"WARNING: Missing folder for '{emotion_name}' at {emotion_dir}")
                continue

            label_idx = cls.EMOTIONS.index(emotion_name)

            for filename in os.listdir(emotion_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    filepath = os.path.join(emotion_dir, filename)
                    image_paths.append(filepath)
                    labels.append(label_idx)

        print(f"Loaded {len(image_paths)} images from {root_dir}")
        for i, emotion in enumerate(cls.EMOTIONS):
            count = labels.count(i)
            print(f"  {emotion:>10s}: {count:>5d} images")

        return cls(image_paths, labels, augment=augment)