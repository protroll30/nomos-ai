"""Quick check: imports, CUDA, and optional tiny load (no full training). Run from repo root with venv active."""

from __future__ import annotations

import sys


def main() -> int:
    print("Python:", sys.version.split()[0])

    try:
        import torch
    except ImportError:
        print("FAIL: torch not installed")
        return 1
    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    try:
        from unsloth import FastLanguageModel
    except Exception as e:
        print("FAIL: unsloth import:", e)
        return 1
    print("OK: unsloth.FastLanguageModel imports")

    try:
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except Exception as e:
        print("FAIL: trl/transformers:", e)
        return 1
    print("OK: trl SFTTrainer + transformers TrainingArguments import")

    try:
        from datasets import load_dataset
    except Exception as e:
        print("FAIL: datasets:", e)
        return 1
    print("OK: datasets")

    print()
    print("Environment looks good for Unsloth. For training you need CUDA + enough VRAM.")
    print("Next: python scripts/train_unsloth_sft.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
