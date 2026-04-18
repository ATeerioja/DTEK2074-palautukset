"""
Exports the trained model for real-time inference in Phase 4.

Produces two formats:
    1. Pure PyTorch state dict (.pt)     — for Python-based inference
    2. TorchScript (.pt)                 — for optimized / C++ inference

Usage:
    python src/export.py --checkpoint models/best-emotion-model-epoch=15-val_loss=0.6234.ckpt
"""

import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import EmotionCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Export model for deployment")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # =====================
    # LOAD FROM CHECKPOINT
    # =====================
    print(f"Loading checkpoint: {args.checkpoint}")
    model = EmotionCNN.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.freeze()

    # =====================
    # EXPORT 1: State Dict
    # =====================
    # Use this for Python inference in Phase 4
    state_dict_path = os.path.join(args.output_dir, "emotion_model_weights.pt")
    torch.save(model.state_dict(), state_dict_path)
    print(f"State dict saved to: {state_dict_path}")

    # =====================
    # EXPORT 2: TorchScript
    # =====================
    # Use this for optimized inference or non-Python deployment
    example_input = torch.randn(1, 1, 48, 48)    # batch=1, channels=1, 48x48
    scripted = torch.jit.trace(model, example_input)

    script_path = os.path.join(args.output_dir, "emotion_model_scripted.pt")
    scripted.save(script_path)
    print(f"TorchScript saved to: {script_path}")

    # =====================
    # VERIFY
    # =====================
    print("\nVerifying exported models...")

    # Test both produce the same output
    with torch.no_grad():
        original_output = model(example_input)
        scripted_output = scripted(example_input)

    diff = (original_output - scripted_output).abs().max().item()
    print(f"Max difference between original and scripted: {diff:.10f}")

    if diff < 1e-5:
        print("PASSED — Models are equivalent.")
    else:
        print("WARNING — Models differ. Check export.")

    # =====================
    # EXPORT PREPROCESSING CONFIG
    # =====================
    # Save a config so Phase 4 knows exactly how to preprocess
    config = {
        "image_size": 48,
        "channels": 1,                        # grayscale
        "normalize_range": [0.0, 1.0],        # divide by 255
        "emotions": ["angry", "disgust", "fear", "happy",
                     "neutral", "sad", "surprise"],
    }

    import json
    config_path = os.path.join(args.output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")

    # File sizes
    print(f"\nFile sizes:")
    for path in [state_dict_path, script_path, config_path]:
        size = os.path.getsize(path) / 1024
        print(f"  {os.path.basename(path):>35s}: {size:>8.1f} KB")


if __name__ == "__main__":
    main()