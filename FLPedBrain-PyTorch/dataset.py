"""
Dataset classes for CDS (Centralized Data Sharing) brain segmentation.

Loads and concatenates all site data for centralized training baseline.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config


class PedBrainDataset(Dataset):
    """
    Dataset for 3D pediatric brain tumor segmentation.

    Data format:
        - xs_uint8: (N, 256, 256, 64) uint8 images
        - ys_uint8: (N, 256, 256, 64) uint8 segmentation masks
        - label_classes: (N,) int64 class labels (0-4)

    Preprocessing:
        - Normalize images to [0, 1]
        - Optional random cropping and resizing for augmentation
        - Transpose to PyTorch format (N, C, D, H, W)
    """

    def __init__(self, data_dir, site_ids=None, include_normals=True,
                 normals_file=None, augment=False, split='train',
                 normals_fraction=None):
        """
        Args:
            data_dir: Path to data directory
            site_ids: List of site IDs to load (e.g., ['TM', 'PH', ...])
            include_normals: Whether to include normal brain scans
            normals_file: Path to normals NPY file
            augment: Whether to apply data augmentation
            split: 'train' or 'val'
            normals_fraction: Fraction of normals to use relative to tumor samples
                              (e.g., 0.25 means use num_tumor_samples * 0.25 normals)
                              Set to None to use all normals.
        """
        self.data_dir = data_dir
        self.augment = augment
        self.split = split

        # Load and concatenate all data
        self.images = []
        self.masks = []
        self.labels = []

        num_tumor_samples = 0

        if site_ids:
            # Load individual site files
            for site_id in site_ids:
                suffix = 'train' if split == 'train' else 'val'
                file_path = os.path.join(data_dir, f'{site_id}_data_uint8_{suffix}.npy')
                if os.path.exists(file_path):
                    data = np.load(file_path, allow_pickle=True)
                    self.images.append(data['xs_uint8'])
                    self.masks.append(data['ys_uint8'])
                    self.labels.append(data['label_classes'])
                    num_tumor_samples += len(data['xs_uint8'])
                    print(f"Loaded {site_id}: {len(data['xs_uint8'])} samples")
        else:
            # Load combined file
            suffix = 'train' if split == 'train' else 'val'
            combined_file = os.path.join(data_dir, f'combined_data_uint8_{suffix}.npy')
            if os.path.exists(combined_file):
                data = np.load(combined_file, allow_pickle=True)
                self.images.append(data['xs_uint8'])
                self.masks.append(data['ys_uint8'])
                self.labels.append(data['label_classes'])
                num_tumor_samples = len(data['xs_uint8'])
                print(f"Loaded combined {split}: {len(data['xs_uint8'])} samples")

        # Include normal brain scans (with optional subsampling to match TF)
        if include_normals and normals_file and os.path.exists(normals_file):
            normals = np.load(normals_file, allow_pickle=True)
            if split == 'train':
                normals_x = normals['train_x']
                normals_seg = normals['train_seg']
                normals_y = normals['train_y']
            else:
                normals_x = normals['val_x']
                normals_seg = normals['val_seg']
                normals_y = normals['val_y']

            total_normals = len(normals_x)

            # Subsample normals if fraction specified (to match TF approach)
            if normals_fraction is not None and num_tumor_samples > 0:
                num_normals_to_use = int(num_tumor_samples * normals_fraction)
                num_normals_to_use = min(num_normals_to_use, total_normals)
                if num_normals_to_use < total_normals:
                    # Random sample without replacement
                    np.random.seed(42)  # For reproducibility
                    indices = np.random.choice(total_normals, num_normals_to_use, replace=False)
                    normals_x = normals_x[indices]
                    normals_seg = normals_seg[indices]
                    normals_y = normals_y[indices]
                    print(f"Subsampled normals ({split}): {num_normals_to_use}/{total_normals} "
                          f"(fraction={normals_fraction}, tumor_samples={num_tumor_samples})")

            self.images.append(normals_x)
            self.masks.append(normals_seg)
            self.labels.append(normals_y)
            print(f"Loaded normals ({split}): {len(normals_x)} samples")

        # Concatenate all data
        if self.images:
            self.images = np.concatenate(self.images, axis=0)
            self.masks = np.concatenate(self.masks, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
        else:
            raise ValueError(f"No data loaded for split '{split}'")

        print(f"Total {split} samples: {len(self.images)}")
        print(f"  Images shape: {self.images.shape}")
        print(f"  Masks shape: {self.masks.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        print(f"  Label distribution: {np.bincount(self.labels.astype(int), minlength=5)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        image = self.images[idx].astype(np.float32) / 255.0  # Normalize to [0, 1]
        mask = self.masks[idx].astype(np.float32)  # Already binary (0 or 1)
        label = int(self.labels[idx])

        # Apply augmentation
        if self.augment:
            image, mask = self._augment(image, mask)

        # Convert to PyTorch format: (H, W, D) -> (C, D, H, W)
        # Original: (256, 256, 64)
        # PyTorch: (1, 64, 256, 256)
        image = np.transpose(image, (2, 0, 1))  # (D, H, W)
        mask = np.transpose(mask, (2, 0, 1))    # (D, H, W)

        # Add channel dimension
        image = np.expand_dims(image, axis=0)  # (1, D, H, W)
        mask = np.expand_dims(mask, axis=0)    # (1, D, H, W)

        # Convert to tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        label = torch.tensor(label, dtype=torch.long)

        return image, mask, label

    def _augment(self, image, mask):
        """Apply data augmentation (random crop + resize)."""
        crop_size = config.CROP_SIZE  # 240

        # Random crop position
        max_delta = 256 - crop_size
        delta_x = np.random.randint(0, max_delta + 1)
        delta_y = np.random.randint(0, max_delta + 1)

        # Crop
        image_cropped = image[delta_x:delta_x + crop_size, delta_y:delta_y + crop_size, :]
        mask_cropped = mask[delta_x:delta_x + crop_size, delta_y:delta_y + crop_size, :]

        # Resize back to 256x256 using simple interpolation
        # Process each frame independently
        from scipy.ndimage import zoom

        zoom_factor = 256 / crop_size
        image_resized = zoom(image_cropped, (zoom_factor, zoom_factor, 1), order=1)
        mask_resized = zoom(mask_cropped, (zoom_factor, zoom_factor, 1), order=0)  # nearest for mask

        return image_resized, mask_resized


def get_dataloaders(batch_size=None, batch_size_eval=None, num_workers=4,
                    normals_fraction=None):
    """
    Create train and validation dataloaders with all data concatenated.

    Args:
        batch_size: Training batch size (default: from config)
        batch_size_eval: Evaluation batch size (default: from config)
        num_workers: Number of data loading workers
        normals_fraction: Fraction of normals to use (default: from config)
                          Set to 0 to exclude normals entirely.

    Returns:
        train_loader, val_loader
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if batch_size_eval is None:
        batch_size_eval = config.BATCH_SIZE_EVAL
    if normals_fraction is None:
        normals_fraction = getattr(config, 'NORMALS_FRACTION', None)

    # Training dataset: combined tumor data + normals
    train_dataset = PedBrainDataset(
        data_dir=config.DATA_DIR,
        site_ids=None,  # Use combined file
        include_normals=True,
        normals_file=config.NORMALS_TRAIN,
        augment=config.AUGMENT_TRAIN,
        split='train',
        normals_fraction=normals_fraction
    )

    # Validation dataset: combined validation + normals
    val_dataset = PedBrainDataset(
        data_dir=config.DATA_DIR,
        site_ids=None,  # Use combined file
        include_normals=True,
        normals_file=config.NORMALS_VAL,
        augment=False,
        split='val',
        normals_fraction=normals_fraction
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")

    train_loader, val_loader = get_dataloaders(batch_size=2, batch_size_eval=2, num_workers=0)

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test a batch
    for images, masks, labels in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Images: {images.shape}, dtype: {images.dtype}")
        print(f"  Masks: {masks.shape}, dtype: {masks.dtype}")
        print(f"  Labels: {labels.shape}, dtype: {labels.dtype}")
        print(f"  Label values: {labels}")
        break
