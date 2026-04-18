"""
PyTorch Lightning Module for Emotion Classification.

Contains:
    - CNN architecture (48x48 grayscale input → 7 emotion classes)
    - Training step
    - Validation step
    - Test step
    - Optimizer and scheduler configuration
    - Logging of all metrics
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
    ConfusionMatrix,
)


class EmotionCNN(pl.LightningModule):
    """
    Custom CNN for 48x48 grayscale facial emotion recognition.

    Architecture:
        3 convolutional blocks → global average pooling → classifier

    Each conv block:
        Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool → Dropout
    """

    def __init__(
        self,
        num_classes: int = 7,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: torch.Tensor = None,
        scheduler_patience: int = 5,
    ):
        super().__init__()

        # Save all __init__ arguments to self.hparams
        # This means Lightning will log them and save them with checkpoints
        self.save_hyperparameters(ignore=["class_weights"])

        # Store class weights (not a hyperparameter, it's data-dependent)
        self.class_weights = class_weights

        # =====================
        # CNN ARCHITECTURE
        # =====================

        # Block 1: (1, 48, 48) → (32, 24, 24)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),    # 48x48 → 48x48
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),   # 48x48 → 48x48
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 48x48 → 24x24
            nn.Dropout2d(0.25),
        )

        # Block 2: (32, 24, 24) → (64, 12, 12)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 24x24 → 24x24
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # 24x24 → 24x24
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 24x24 → 12x12
            nn.Dropout2d(0.25),
        )

        # Block 3: (64, 12, 12) → (128, 6, 6)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 12x12 → 12x12
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # 12x12 → 12x12
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 12x12 → 6x6
            nn.Dropout2d(0.25),
        )

        # Global Average Pooling: (128, 6, 6) → (128,)
        # Instead of flattening 128*6*6 = 4608 features,
        # GAP averages each channel into a single number.
        # This reduces parameters dramatically and prevents overfitting.
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier: (128,) → (7,)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # =====================
        # LOSS FUNCTION
        # =====================
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights     # None is fine — uses equal weights
        )

        # =====================
        # METRICS
        # =====================
        # We create separate metric objects for each phase
        # because Lightning tracks them independently

        # Training metrics
        self.train_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes
        )

        # Validation metrics
        self.val_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        # Test metrics
        self.test_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_confusion = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

        # Per-class metrics for detailed analysis
        self.test_f1_per_class = F1Score(
            task="multiclass", num_classes=num_classes, average="none"
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: tensor of shape (batch_size, 1, 48, 48)

        Returns:
            logits: tensor of shape (batch_size, 7)
                    raw scores (NOT probabilities)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    # =====================
    # TRAINING
    # =====================

    def training_step(self, batch, batch_idx):
        """
        Called for each batch during training.
        Lightning handles: zero_grad, backward, optimizer.step automatically.
        """
        images, labels = batch
        logits = self(images)                  # forward pass
        loss = self.criterion(logits, labels)  # compute loss

        # Compute predictions for metrics
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, labels)

        # Log metrics — Lightning collects these automatically
        self.log("train_loss", loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_accuracy,
                 on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # =====================
    # VALIDATION
    # =====================

    def validation_step(self, batch, batch_idx):
        """
        Called for each batch during validation.
        No gradients are computed (Lightning handles torch.no_grad).
        """
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)

        self.log("val_loss", loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1,
                 on_step=False, on_epoch=True, prog_bar=True)

    # =====================
    # TESTING
    # =====================

    def test_step(self, batch, batch_idx):
        """
        Called for each batch during testing.
        """
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_confusion(preds, labels)
        self.test_f1_per_class(preds, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

    # =====================
    # OPTIMIZER & SCHEDULER
    # =====================

    def configure_optimizers(self):
        """
        Lightning calls this to set up optimizer and scheduler.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",                    # reduce when val_loss stops decreasing
            factor=0.1,                    # new_lr = old_lr × 0.1
            patience=self.hparams.scheduler_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",     # watch this metric
                "interval": "epoch",       # check once per epoch
            },
        }