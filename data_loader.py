import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import random

# ─── Dataset Class ───────────────────────────────────────────
class ECSSDDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def augment_data(self, image, mask):
        # Horizontal flip
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # Random crop
        if random.random() > 0.5:
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(180, 180))
            image = transforms.functional.crop(image, i, j, h, w)
            mask = transforms.functional.crop(mask, i, j, h, w)

        # Brightness variation
        if random.random() > 0.5:
            image = transforms.functional.adjust_brightness(
                image, brightness_factor=random.uniform(0.7, 1.3))

        # Contrast variation
        if random.random() > 0.5:
            image = transforms.functional.adjust_contrast(
                image, contrast_factor=random.uniform(0.7, 1.3))

        return image, mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.augment:
            image, mask = self.augment_data(image, mask)

        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        return image, mask


# ─── Load Data ───────────────────────────────────────────────
def get_dataloaders(data_dir, batch_size=16):
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "ground_truth_mask")

    image_paths = sorted([os.path.join(image_dir, f)
                          for f in os.listdir(image_dir)
                          if f.endswith(".jpg")])
    mask_paths = sorted([os.path.join(mask_dir, f)
                         for f in os.listdir(mask_dir)
                         if f.endswith(".png")])

    # Split: 70% train, 15% val, 15% test
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        image_paths, mask_paths, test_size=0.30, random_state=42)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks, test_size=0.50, random_state=42)

    # Augmentation vetëm për train
    train_dataset = ECSSDDataset(train_imgs, train_masks, augment=True)
    val_dataset = ECSSDDataset(val_imgs, val_masks, augment=False)
    test_dataset = ECSSDDataset(test_imgs, test_masks, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


# ─── Test ────────────────────────────────────────────────────
if __name__ == "__main__":
    data_dir = "data/ECSSD"
    train_loader, val_loader, test_loader = get_dataloaders(data_dir)
    images, masks = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape:  {masks.shape}")