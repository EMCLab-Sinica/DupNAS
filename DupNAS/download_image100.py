#!/usr/bin/env python3

import os
import sys

from settings import Settings
from NASBase.load_image100 import load_image100_dataset


def count_images(directory: str) -> int:
    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    total = 0
    for root, _, files in os.walk(directory):
        total += sum(
            1
            for filename in files
            if os.path.splitext(filename)[1].lower() in valid_extensions
        )

    return total


def main() -> int:
    dataset_dir = Settings.NAS_SETTINGS_PER_DATASET["IMAGE100"]["TRAIN_DATADIR"]

    print("=" * 60)
    print("ImageNet-100 dataset preparation")
    print(f"Target directory: {dataset_dir}")
    print("=" * 60)

    try:
        train_dir, val_dir = load_image100_dataset(
            DATASET_DIR=dataset_dir
        )
    except Exception as error:
        print(f"\nERROR: Failed to prepare ImageNet-100: {error}")
        return 1

    if not os.path.isdir(train_dir):
        print(f"ERROR: Training directory does not exist: {train_dir}")
        return 1

    if not os.path.isdir(val_dir):
        print(f"ERROR: Validation directory does not exist: {val_dir}")
        return 1

    train_classes = [
        name
        for name in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, name))
    ]

    val_classes = [
        name
        for name in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, name))
    ]

    print("\nDataset preparation completed.")
    print(f"Training directory:   {train_dir}")
    print(f"Validation directory: {val_dir}")
    print(f"Training classes:     {len(train_classes)}")
    print(f"Validation classes:   {len(val_classes)}")
    print(f"Training images:      {count_images(train_dir)}")
    print(f"Validation images:    {count_images(val_dir)}")

    if len(train_classes) != 100:
        print(
            f"WARNING: Expected 100 training classes, "
            f"but found {len(train_classes)}."
        )

    if len(val_classes) != 100:
        print(
            f"WARNING: Expected 100 validation classes, "
            f"but found {len(val_classes)}."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())