"""
Main training script.

Usage:
    python src/train.py
    python src/train.py --batch_size 32 --learning_rate 0.0005 --max_epochs 100

After training, the best model checkpoint is saved to models/
and TensorBoard logs are saved to logs/
"""

import argparse
import os
import sys

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datamodule import EmotionDataModule
from model import EmotionCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Train Emotion Recognition CNN")

    # Data
    parser.add_argument("--data_dir", type=str, default="/home/anton/projects/DTEK2074-palautukset/projekti-11-koneoppiminen/data/processed/",
                        help="Path to folder with emotion subfolders")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers. Set to 0 on Windows if errors occur")

    # Model
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Training
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="LR scheduler patience")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # =====================
    # REPRODUCIBILITY
    # =====================
    pl.seed_everything(args.seed, workers=True)

    # =====================
    # DATA
    # =====================
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    datamodule = EmotionDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Trigger data loading and splitting
    datamodule.setup()

    # =====================
    # MODEL
    # =====================
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)

    model = EmotionCNN(
        num_classes=7,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weights=datamodule.class_weights,
        scheduler_patience=args.scheduler_patience,
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:>10,}")
    print(f"Trainable parameters: {trainable_params:>10,}")

    # =====================
    # CALLBACKS
    # =====================

    callbacks = [
        # Save the best model based on validation loss
        ModelCheckpoint(
            dirpath="models/",
            filename="best-emotion-model-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,              # only keep the single best
            verbose=True,
        ),

        # Stop training if validation loss doesn't improve
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            mode="min",
            verbose=True,
        ),

        # Log learning rate changes
        LearningRateMonitor(logging_interval="epoch"),

        # Nice progress bar
        RichProgressBar(),
    ]

    # =====================
    # LOGGER
    # =====================

    logger = TensorBoardLogger(
        save_dir="logs/",
        name="emotion_recognition",
    )

    # =====================
    # TRAINER
    # =====================
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",           # GPU if available, else CPU
        devices=1,
        deterministic=True,            # reproducible results
        log_every_n_steps=10,
    )

    # =====================
    # TRAIN
    # =====================
    trainer.fit(model, datamodule=datamodule)

    # =====================
    # TEST
    # =====================
    print("\n" + "=" * 60)
    print("TESTING BEST MODEL")
    print("=" * 60)

    # Load the best checkpoint and test it
    trainer.test(
        model,
        datamodule=datamodule,
        ckpt_path="best",              # use the best saved checkpoint
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model saved to: {callbacks[0].best_model_path}")
    print(f"TensorBoard logs:    logs/emotion_recognition/")
    print(f"\nTo view logs run:")
    print(f"  tensorboard --logdir logs/")


if __name__ == "__main__":
    main()