#    split   — 'train' or 'test'
#    label   — 'normal' or 'anomaly'
#    image   — relative path to the image
#    mask    — relative path to the mask (empty for "normal")"
#
# Modes:
#    TRAIN: pairs of (x1, x2) normal images with different augmentations
#    TEST:  single images + label + mask (if available)



import random
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ------------------------------------------------------------------
# Augmentation
# ------------------------------------------------------------------

def get_train_transforms(img_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_test_transforms(img_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_mask_transform(img_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class VisADataset(Dataset):
    """
    VisA Dataloader.
    """

    # All 12 categories VisA
    CATEGORIES = [
        "candle", "capsules", "cashew", "chewinggum", "fryum",
        "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
    ]

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        img_size: int = 256,
        pair_mode: bool = True,
    ):
        assert category in self.CATEGORIES, \
            f"Unknown category: {category}. Available: {self.CATEGORIES}"
        assert split in ("train", "test"), \
            f"split should be 'train' or 'test', recieved: {split}"

        self.root = Path(root)
        self.category = category
        self.split = split
        self.img_size = img_size
        self.pair_mode = pair_mode and (split == "train")

        # Transformations
        if split == "train":
            self.img_transform = get_train_transforms(img_size)
            self.img_transform2 = get_train_transforms(img_size)  # second branch — other augmentations
        else:
            self.img_transform = get_test_transforms(img_size)
            self.img_transform2 = None

        self.mask_transform = get_mask_transform(img_size)

        self.samples = self._load_csv()

    def _load_csv(self) -> pd.DataFrame:
        """Reading 1cls.csv and filter required items."""
        candidates = [
            self.root / "split_csv" / "1cls.csv",
            self.root / self.category / "split_csv" / "1cls.csv",
        ]
        csv_path = None
        for c in candidates:
            if c.exists():
                csv_path = c
                break

        if csv_path is None:
            found = list(self.root.rglob("1cls.csv"))
            if found:
                csv_path = found[0]

        if csv_path is None:
            raise FileNotFoundError(
                f"1cls.csv is nor found in {self.root}.\n"
            )

        df = pd.read_csv(csv_path)

        df.columns = df.columns.str.strip().str.lower()
        if "object" in df.columns:
            df = df[df["object"] == self.category]
        elif "category" in df.columns:
            df = df[df["category"] == self.category]

        df = df[df["split"] == self.split].reset_index(drop=True)

        if self.split == "train":
            df = df[df["label"] == "normal"].reset_index(drop=True)

        if len(df) == 0:
            raise RuntimeError(
                f"No data for category={self.category}, split={self.split}.\n"
                f"Check CSV: {csv_path}"
            )

        return df

    def _load_image(self, rel_path: str) -> Image.Image:
        """Loading image"""
        candidates = [
            self.root / rel_path,
            self.root / self.category / rel_path,
            Path(rel_path)
        ]
        for p in candidates:
            if p.exists():
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(
            f"Image not found: {rel_path}\n"
            f"Searched in: {[str(c) for c in candidates]}"
        )

    def _load_mask(self, rel_path: Optional[str]) -> Optional[torch.Tensor]:
        """Loads the anomaly mask (or returns None for normal values)."""
        if not rel_path or pd.isna(rel_path):
            return None

        candidates = [
            self.root / rel_path,
            self.root / self.category / rel_path,
        ]
        for p in candidates:
            if p.exists():
                mask = Image.open(p).convert("L")  # grayscale
                return self.mask_transform(mask)

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples.iloc[idx]

        img = self._load_image(row["image"])

        if self.pair_mode:
            # TRAIN: a pair (x1, x2) with different augmentations of the same image
            # We simulate a "normal reference" + "current normal"
            x1 = self.img_transform(img)
            x2 = self.img_transform2(img)

            # Sometimes we take a random second normal image from the dataset
            if random.random() < 0.5:
                idx2 = random.randint(0, len(self) - 1)
                img2 = self._load_image(self.samples.iloc[idx2]["image"])
                x2 = self.img_transform2(img2)

            return {
                "x1": x1,                       # tensor [3, H, W]
                "x2": x2,                       # tensor [3, H, W]
                "label": torch.tensor(0),       # 0 = normal (always in "train")
            }

        else:
            # TEST: image + label + mask
            x = self.img_transform(img)
            label = 0 if row["label"] == "normal" else 1
            mask = self._load_mask(row.get("mask"))

            return {
                "image": x,                              # tensor [3, H, W]
                "label": torch.tensor(label),            # 0 = normal, 1 = defect
                "mask": mask if mask is not None         # [1, H, W] or None
                        else torch.zeros(1, self.img_size, self.img_size),
                "has_mask": mask is not None,
                "image_path": str(row["image"]),
            }


# ------------------------------------------------------------------
# DataLoader helpers
# ------------------------------------------------------------------

def build_dataloaders(
    root: str,
    category: str,
    img_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Creates train and test DataLoaders for a single VisA category.

    Returns:
        train_loader — batches of (x1, x2) pairs of normal images
        test_loader  — batches of single labeled images
    """
    train_ds = VisADataset(root, category, split="train",
                           img_size=img_size, pair_mode=True)
    test_ds = VisADataset(root, category, split="test",
                          img_size=img_size, pair_mode=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,        
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,         
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[VisA] Category: {category}")
    print(f"  Train: {len(train_ds)} normal images")
    print(f"  Test:  {len(test_ds)} images "
          f"({(test_ds.samples['label']=='anomaly').sum()} anomalies)")

    return train_loader, test_loader
