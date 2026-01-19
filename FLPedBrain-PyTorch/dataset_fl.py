"""
Dataset classes for Federated Learning brain segmentation.

Loads per-site data for FL training (each client has its own dataset).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config


class SiteDataset(Dataset):
    """
    Dataset for a single FL site/client.

    Each site has its own tumor data + normals. Normals can come from:
    1. Site-specific normal data (default): {site_id}_normals_uint8_{split}.npy
    2. Shared normal data (fallback): A single normals file shared across sites

    This matches the TF FL approach where each client gets:
    - All tumor samples from their site
    - Normal samples (either site-specific or randomly sampled from shared pool)

    Data format:
        - xs_uint8: (N, 256, 256, 64) uint8 images
        - ys_uint8: (N, 256, 256, 64) uint8 segmentation masks
        - label_classes: (N,) int64 class labels (0-4)
    """

    def __init__(self, data_dir, site_id, normals_file=None, augment=False,
                 split='train', random_seed=42, use_site_normals=True,
                 normals_fraction=1.0):
        """
        Args:
            data_dir: Path to data directory
            site_id: Site identifier (e.g., 'TM', 'PH', etc.)
            normals_file: Path to shared normals NPY file (used if site-specific not found)
            augment: Whether to apply data augmentation
            split: 'train' or 'val'
            random_seed: Random seed for reproducible normal sampling
            use_site_normals: If True (default), look for site-specific normals first
            normals_fraction: Fraction of normals to use relative to tumor samples
                              (1.0 = same number as tumors, 0.5 = half, etc.)
        """
        self.data_dir = data_dir
        self.site_id = site_id
        self.augment = augment
        self.split = split

        # Load site-specific tumor data
        suffix = 'train' if split == 'train' else 'val'
        site_file = os.path.join(data_dir, f'{site_id}_data_uint8_{suffix}.npy')

        if not os.path.exists(site_file):
            raise FileNotFoundError(f"Site data not found: {site_file}")

        site_data = np.load(site_file, allow_pickle=True)
        if isinstance(site_data, np.ndarray) and site_data.ndim == 0:
            site_data = site_data.item()

        self.images = site_data['xs_uint8']
        self.masks = site_data['ys_uint8']
        self.labels = site_data['label_classes']

        num_tumor_samples = len(self.images)
        print(f"Site {site_id} ({split}): {num_tumor_samples} tumor samples")

        # Try to load normals
        normals_loaded = False

        # Option 1: Site-specific normals 
        if use_site_normals:
            site_normals_file = os.path.join(data_dir, f'{site_id}_normals_uint8_{suffix}.npy')
            if os.path.exists(site_normals_file):
                site_normals = np.load(site_normals_file, allow_pickle=True)
                if isinstance(site_normals, np.ndarray) and site_normals.ndim == 0:
                    site_normals = site_normals.item()

                normals_x = site_normals['xs_uint8']
                normals_seg = site_normals['ys_uint8']
                normals_y = site_normals['label_classes']

                # Use fraction of normals relative to tumor samples
                num_normals_available = len(normals_x)
                num_normals_to_use = min(
                    int(num_tumor_samples * normals_fraction),
                    num_normals_available
                )

                if num_normals_to_use > 0:
                    np.random.seed(random_seed)
                    indices = np.random.choice(num_normals_available, num_normals_to_use, replace=False)

                    self.images = np.concatenate([self.images, normals_x[indices]], axis=0)
                    self.masks = np.concatenate([self.masks, normals_seg[indices]], axis=0)
                    self.labels = np.concatenate([self.labels, normals_y[indices]], axis=0)

                    print(f"  Added {num_normals_to_use} site-specific normals -> {len(self.images)} total samples")
                    normals_loaded = True

        # Option 2: Shared normals file (when normal data is public)
        if not normals_loaded and normals_file and os.path.exists(normals_file):
            normals = np.load(normals_file, allow_pickle=True)
            if isinstance(normals, np.ndarray) and normals.ndim == 0:
                normals = normals.item()

            if split == 'train':
                normals_x = normals['train_x']
                normals_seg = normals['train_seg']
                normals_y = normals['train_y']
            else:
                normals_x = normals['val_x']
                normals_seg = normals['val_seg']
                normals_y = normals['val_y']

            # Use fraction of normals relative to tumor samples
            np.random.seed(random_seed)
            num_normals_available = len(normals_x)
            num_normals_to_use = min(
                int(num_tumor_samples * normals_fraction),
                num_normals_available
            )

            if num_normals_to_use > 0:
                indices = np.random.choice(num_normals_available, num_normals_to_use, replace=False)

                self.images = np.concatenate([self.images, normals_x[indices]], axis=0)
                self.masks = np.concatenate([self.masks, normals_seg[indices]], axis=0)
                self.labels = np.concatenate([self.labels, normals_y[indices]], axis=0)

                print(f"  Added {num_normals_to_use} shared normals -> {len(self.images)} total samples")

        self.num_samples = len(self.images)
        print(f"  Label distribution: {np.bincount(self.labels.astype(int), minlength=5)}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load image and mask
        image = self.images[idx].astype(np.float32) / 255.0  # Normalize to [0, 1]
        mask = self.masks[idx].astype(np.float32)  # Already binary (0 or 1)
        label = int(self.labels[idx])

        # Apply augmentation (matching TF: 240x240 crop + resize to 256x256)
        if self.augment:
            image, mask = self._augment(image, mask)

        # Convert to PyTorch format: (H, W, D) -> (C, D, H, W)
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
        from scipy.ndimage import zoom

        zoom_factor = 256 / crop_size
        image_resized = zoom(image_cropped, (zoom_factor, zoom_factor, 1), order=1)
        mask_resized = zoom(mask_cropped, (zoom_factor, zoom_factor, 1), order=0)  # nearest for mask

        return image_resized, mask_resized


def get_site_dataloaders(site_ids, data_dir, normals_train_file=None, normals_val_file=None,
                         batch_size=8, batch_size_eval=2, num_workers=4, augment=True,
                         use_site_normals=True, normals_fraction=1.0):
    """
    Create dataloaders for all FL sites.

    Args:
        site_ids: List of site IDs (e.g., ['TM', 'PH', ...])
        data_dir: Path to data directory
        normals_train_file: Path to shared training normals file (fallback if no site-specific)
        normals_val_file: Path to validation normals file (optional)
        batch_size: Training batch size
        batch_size_eval: Evaluation batch size
        num_workers: Number of data loading workers
        augment: Whether to augment training data
        use_site_normals: If True (default), look for site-specific normals first
        normals_fraction: Fraction of normals to use relative to tumor samples

    Returns:
        site_train_loaders: Dict mapping site_id -> DataLoader
        site_num_samples: Dict mapping site_id -> number of samples
        val_loader: Single validation DataLoader (shared across all clients)
    """
    site_train_loaders = {}
    site_num_samples = {}

    for site_id in site_ids:
        try:
            dataset = SiteDataset(
                data_dir=data_dir,
                site_id=site_id,
                normals_file=normals_train_file,
                augment=augment,
                split='train',
                use_site_normals=use_site_normals,
                normals_fraction=normals_fraction
            )

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )

            site_train_loaders[site_id] = loader
            site_num_samples[site_id] = len(dataset)

        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    # Create shared validation loader (combined validation data)
    # This matches TF approach: evaluate on combined val set
    val_loader = None
    if normals_val_file:
        from dataset import PedBrainDataset
        val_dataset = PedBrainDataset(
            data_dir=data_dir,
            site_ids=None,  # Use combined file
            include_normals=True,
            normals_file=normals_val_file,
            augment=False,
            split='val',
            normals_fraction=None  # Use all normals for validation
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size_eval,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return site_train_loaders, site_num_samples, val_loader



# TODO: refactor need to move to config file
# Site data sizes from TF code (number of tumor samples per site)
# These are used for weighted FedAvg aggregation
SITE_DATA_SIZES = {
    'TM': 13,
    'PH': 55,
    'TO': 92,
    'UT': 129,
    'DU': 24,
    'CP': 96,
    'IN': 118,
    'ST': 328,
    'SE': 241,
    'CG': 150,
    'NY': 26,
    'CH': 14,
    'GO': 78,
    'BO': 19,
    'KC': 3,
    'DY': 28,
}

# Full list of site IDs
SITE_IDS = ['TM', 'PH', 'TO', 'UT', 'DU', 'CP', 'IN', 'ST', 'SE', 'CG', 'NY', 'CH', 'GO', 'BO', 'KC', 'DY']


if __name__ == "__main__":
    # Test dataset loading
    print("Testing FL dataset loading...")

    data_dir = config.DATA_DIR
    normals_file = config.NORMALS_TRAIN

    # Test single site with site-specific normals (default)
    site_id = 'ST'
    print(f"\n{'='*60}")
    print(f"Test 1: Site-specific normals (default)")
    print(f"{'='*60}")
    try:
        dataset = SiteDataset(
            data_dir=data_dir,
            site_id=site_id,
            normals_file=normals_file,
            augment=True,
            split='train',
            use_site_normals=True,  # Default: look for site-specific normals first
            normals_fraction=1.0
        )

        print(f"\nSite {site_id} dataset: {len(dataset)} samples")

        # Test a sample
        image, mask, label = dataset[0]
        print(f"Sample shapes: image={image.shape}, mask={mask.shape}, label={label}")

    except FileNotFoundError as e:
        print(f"Could not test: {e}")
        print("Per-site data files need to be copied to data directory")

    # Test with shared normals fallback
    print(f"\n{'='*60}")
    print(f"Test 2: Shared normals fallback")
    print(f"{'='*60}")
    try:
        dataset = SiteDataset(
            data_dir=data_dir,
            site_id=site_id,
            normals_file=normals_file,
            augment=True,
            split='train',
            use_site_normals=False,  # Force use of shared normals
            normals_fraction=0.5  # Use half as many normals as tumors
        )

        print(f"\nSite {site_id} dataset: {len(dataset)} samples")

    except FileNotFoundError as e:
        print(f"Could not test: {e}")
