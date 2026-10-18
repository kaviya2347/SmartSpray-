"""
FILE PATH: SmartSpray/scripts/analyze_splits.py

PURPOSE:
    Reproduces the exact train/validation/"test" splits used during training
    (notebooks/m3_classification_trials.ipynb) and verifies whether the
    notebook's 500-image "test" split genuinely overlaps with the training set.

    This is a data-integrity check, NOT the evaluation itself. It simply
    determines whether a truly held-out test set exists in this repository.

OUTPUT:
    Prints, for each class:
        - total images in prepared_dataset
        - train / val / test counts (as Keras split them)
        - number of test images that also appear in train or val

DEPENDENCIES:
    - tensorflow/keras (for ImageDataGenerator, matching the training env)
    - pathlib (stdlib)
"""

from pathlib import Path

import numpy as np

PREPARED_DIR = Path("models/classification/prepared_dataset")
IMAGE_SIZE = 224
BATCH_SIZE = 32
RANDOM_SEED = 42
CLASS_NAMES = ["early_blight", "healthy", "late_blight", "septoria_leaf_spot"]

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # noqa: E402


def main():
    print("ImageDataGenerator split analysis")
    print("=" * 60)

    datagen_train = ImageDataGenerator(
        preprocessing_function=lambda x: x,
        validation_split=0.2,
    )
    train_gen = datagen_train.flow_from_directory(
        PREPARED_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=RANDOM_SEED,
        shuffle=True,
    )

    val_gen = ImageDataGenerator(
        preprocessing_function=lambda x: x,
        validation_split=0.2,
    ).flow_from_directory(
        PREPARED_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=RANDOM_SEED,
        shuffle=False,
    )

    test_gen = ImageDataGenerator(
        preprocessing_function=lambda x: x,
        validation_split=0.1,
    ).flow_from_directory(
        PREPARED_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    print(f"Train samples: {train_gen.samples}")
    print(f"Val samples  : {val_gen.samples}")
    print(f"Test samples : {test_gen.samples}")

    # DirectoryIterator keeps the resolved file paths under .filepaths
    train_files = set(str(p) for p in train_gen.filepaths)
    val_files = set(str(p) for p in val_gen.filepaths)
    test_files = set(str(p) for p in test_gen.filepaths)

    print(f"\nTrain files: {len(train_files)}")
    print(f"Val files  : {len(val_files)}")
    print(f"Test files : {len(test_files)}")

    test_in_train = test_files & train_files
    test_in_val = test_files & val_files

    print(f"\nTest / Train : {len(test_in_train)}")
    print(f"Test / Val   : {len(test_in_val)}")
    print(f"Test / (Train+Val): {len(test_in_train | test_in_val)}")
    print(f"Test not in Train+Val: {len(test_files - train_files - val_files)}")

    # Per-class overlap breakdown.
    # Classify each test file by its parent directory name.
    def class_of(path: str) -> str:
        return Path(path).parent.name

    print("\nPer-class test overlap:")
    for cls_dir in CLASS_NAMES:
        cls_test = {p for p in test_files if class_of(p) == cls_dir}
        cls_train = {p for p in train_files if class_of(p) == cls_dir}
        cls_val = {p for p in val_files if class_of(p) == cls_dir}
        inter_train = len(cls_test & cls_train)
        inter_val = len(cls_test & cls_val)
        print(
            f"  {cls_dir:20s} test={len(cls_test):4d} "
            f"train={len(cls_train):4d} val={len(cls_val):4d} "
            f"| in_train={inter_train:4d} in_val={inter_val:4d}"
        )


if __name__ == "__main__":
    main()