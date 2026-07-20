"""
dataset.py

PyTorch Dataset for the Conditional DCGAN.

Loads:
    dataset/images/
    dataset/labels.csv

Returns:
    image, hair_label, eye_label
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image

Image.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_DIR, LABELS_FILE, IMAGE_SIZE


class AnimeDataset(Dataset):
    """
    PyTorch Dataset for the Anime Face Dataset.
    """

    def __init__(self) -> None:

        self.df = pd.read_csv(LABELS_FILE)

        # --------------------------------------------------
        # Label Encoding
        # --------------------------------------------------

        self.hair_classes: List[str] = sorted(self.df["hair"].unique())
        self.eye_classes: List[str] = sorted(self.df["eyes"].unique())

        self.hair_to_idx: Dict[str, int] = {
            label: idx
            for idx, label in enumerate(self.hair_classes)
        }

        self.eye_to_idx: Dict[str, int] = {
            label: idx
            for idx, label in enumerate(self.eye_classes)
        }

        self.idx_to_hair = {
            idx: label
            for label, idx in self.hair_to_idx.items()
        }

        self.idx_to_eye = {
            idx: label
            for label, idx in self.eye_to_idx.items()
        }

        # --------------------------------------------------
        # Image Transform
        # --------------------------------------------------

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):

        while True:

            row = self.df.iloc[index]

            image_path = IMAGE_DIR / row["image"]

            try:
                image = Image.open(image_path).convert("RGB")
                image = self.transform(image)

                hair_label = self.hair_to_idx[row["hair"]]
                eye_label = self.eye_to_idx[row["eyes"]]

                return (
                    image,
                    torch.tensor(hair_label, dtype=torch.long),
                    torch.tensor(eye_label, dtype=torch.long),
                )

            except (OSError, IOError):
                print(f"[WARNING] Skipping corrupted image: {image_path.name}")

                index = (index + 1) % len(self.df)
        


if __name__ == "__main__":

    dataset = AnimeDataset()

    print("=" * 50)
    print("Anime Dataset Summary")
    print("=" * 50)

    print(f"Total Images      : {len(dataset)}")
    print(f"Hair Classes      : {len(dataset.hair_classes)}")
    print(f"Eye Classes       : {len(dataset.eye_classes)}")

    print("\nHair Labels")
    print(dataset.hair_to_idx)

    print("\nEye Labels")
    print(dataset.eye_to_idx)

    image, hair, eye = dataset[0]

    print("\nSample")
    print(f"Image Shape       : {tuple(image.shape)}")
    print(f"Hair Label        : {hair.item()}")
    print(f"Eye Label         : {eye.item()}")