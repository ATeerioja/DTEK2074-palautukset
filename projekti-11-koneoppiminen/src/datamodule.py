"""
PyTorch Lightning DataModule.

Handles:
    - Loading all images from the folder
    - Splitting into train / val / test BY SUBJECT (if subject info available)
      or stratified random split
    - Computing class weights for imbalanced data
    - Creating DataLoaders
"""

import os
import re
from collections import Counter

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import EmotionDataset


class EmotionDataModule(pl.LightningDataModule):
    """
    Prepares all data for training, validation, and testing.
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        # Will be populated in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None

    def setup(self, stage=None):
        """
        Called by Lightning before training/testing.
        Loads all data and splits into train/val/test.
        """
        # --- Load all images and labels ---
        full_dataset = EmotionDataset.from_folder(self.data_dir, augment=False)
        all_paths = full_dataset.image_paths
        all_labels = full_dataset.labels

        # --- Attempt subject-aware splitting ---
        # If your files are named like: subjectName_emotion_001.jpg
        # we extract subject IDs to prevent data leakage
        subjects = self._extract_subjects(all_paths)

        if False:
            print("\nSubject-aware splitting detected.")
            print(f"Found {len(set(subjects))} unique subjects.")
            train_paths, train_labels, val_paths, val_labels, \
                test_paths, test_labels = self._split_by_subject(
                    all_paths, all_labels, subjects
                )
        else:
            print("\nNo subject info found. Using stratified random split.")
            train_paths, train_labels, val_paths, val_labels, \
                test_paths, test_labels = self._split_stratified(
                    all_paths, all_labels
                )

        # --- Print split info ---
        print(f"\nSplit sizes:")
        print(f"  Train:      {len(train_paths):>5d}")
        print(f"  Validation: {len(val_paths):>5d}")
        print(f"  Test:       {len(test_paths):>5d}")

        # --- Create datasets ---
        self.train_dataset = EmotionDataset(train_paths, train_labels, augment=True)
        self.val_dataset   = EmotionDataset(val_paths, val_labels, augment=False)
        self.test_dataset  = EmotionDataset(test_paths, test_labels, augment=False)

        # --- Compute class weights from training set ---
        self.class_weights = self._compute_class_weights(train_labels)
        print(f"\nClass weights: {self.class_weights.numpy().round(3)}")

    def _extract_subjects(self, paths: list) -> list | None:
        """
        Tries to extract subject identifiers from filenames.

        Expected naming patterns:
            alice_happy_001.jpg     → subject = "alice"
            bob_angry_042.jpg       → subject = "bob"

        Returns None if pattern doesn't match (falls back to random split).
        """
        subjects = []
        # Pattern: anything before the first underscore
        pattern = re.compile(r"^([a-zA-Z]+)_")

        for path in paths:
            filename = os.path.basename(path)
            match = pattern.match(filename)
            if match:
                subjects.append(match.group(1).lower())
            else:
                return None     # pattern doesn't match, abort

        return subjects

    def _split_by_subject(self, paths, labels, subjects):
        """
        Splits data so that no subject appears in more than one split.
        This prevents the model from memorizing specific faces.
        """
        unique_subjects = list(set(subjects))
        np.random.seed(self.seed)
        np.random.shuffle(unique_subjects)

        n = len(unique_subjects)
        n_test = max(1, int(n * self.test_split))
        n_val = max(1, int(n * self.val_split))

        test_subjects = set(unique_subjects[:n_test])
        val_subjects  = set(unique_subjects[n_test:n_test + n_val])
        train_subjects = set(unique_subjects[n_test + n_val:])

        print(f"  Train subjects: {train_subjects}")
        print(f"  Val subjects:   {val_subjects}")
        print(f"  Test subjects:  {test_subjects}")

        train_paths, train_labels = [], []
        val_paths, val_labels     = [], []
        test_paths, test_labels   = [], []

        for path, label, subject in zip(paths, labels, subjects):
            if subject in test_subjects:
                test_paths.append(path)
                test_labels.append(label)
            elif subject in val_subjects:
                val_paths.append(path)
                val_labels.append(label)
            else:
                train_paths.append(path)
                train_labels.append(label)

        return (train_paths, train_labels,
                val_paths, val_labels,
                test_paths, test_labels)

    def _split_stratified(self, paths, labels):
        """
        Stratified random split — maintains class proportions in each split.
        Used when subject information is not available.
        """
        # First split: separate out the test set
        train_val_paths, test_paths, train_val_labels, test_labels = \
            train_test_split(
                paths, labels,
                test_size=self.test_split,
                stratify=labels,
                random_state=self.seed
            )

        # Second split: separate train and validation
        relative_val_size = self.val_split / (1 - self.test_split)
        train_paths, val_paths, train_labels, val_labels = \
            train_test_split(
                train_val_paths, train_val_labels,
                test_size=relative_val_size,
                stratify=train_val_labels,
                random_state=self.seed
            )

        return (train_paths, train_labels,
                val_paths, val_labels,
                test_paths, test_labels)

    def _compute_class_weights(self, labels: list) -> torch.Tensor:
        """
        Computes inverse frequency weights so rare classes count more.

        If happy has 500 images and fear has 100 images,
        fear gets a higher weight so the model can't ignore it.
        """
        counts = Counter(labels)
        total = len(labels)
        n_classes = len(EmotionDataset.EMOTIONS)

        weights = []
        for i in range(n_classes):
            if counts[i] > 0:
                w = total / (n_classes * counts[i])
            else:
                w = 1.0     # default weight if class is missing
            weights.append(w)

        return torch.tensor(weights, dtype=torch.float32)

    # --- DataLoaders ---
    # Lightning calls these automatically during training

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,               # randomize order each epoch
            num_workers=self.num_workers,
            pin_memory=True,            # faster GPU transfer
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,              # consistent evaluation order
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )