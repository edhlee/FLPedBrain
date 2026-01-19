#!/usr/bin/env python
"""
Download pretrained I3D weights for the brain segmentation model.

Weights are from: https://github.com/piergiaj/pytorch-i3d
Originally converted from DeepMind's TensorFlow I3D trained on ImageNet + Kinetics.
"""

import os
import sys
import urllib.request
import hashlib

# Pretrained weight URLs from piergiaj/pytorch-i3d
WEIGHTS = {
    'rgb_imagenet': {
        'url': 'https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt',
        'filename': 'rgb_imagenet.pt',
        'description': 'RGB I3D pretrained on ImageNet (inflated from 2D)',
    },
    'rgb_charades': {
        'url': 'https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_charades.pt',
        'filename': 'rgb_charades.pt',
        'description': 'RGB I3D fine-tuned on Charades',
    },
}

# Alternative: hassony2/kinetics_i3d_pytorch (converted from original TF)
WEIGHTS_ALT = {
    'rgb_imagenet_kinetics': {
        'url': 'https://github.com/hassony2/kinetics_i3d_pytorch/raw/master/model/model_rgb.pth',
        'filename': 'model_rgb.pth',
        'description': 'RGB I3D pretrained on ImageNet + Kinetics (from TF conversion)',
    },
}


def download_file(url, dest_path, show_progress=True):
    """Download a file with progress indicator."""

    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, count * block_size * 100 // total_size)
            sys.stdout.write(f'\r  Downloading: {percent}%')
            sys.stdout.flush()

    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")

    try:
        if show_progress:
            urllib.request.urlretrieve(url, dest_path, progress_hook)
            print()  # newline after progress
        else:
            urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"\n  Error downloading: {e}")
        return False


def verify_file(filepath, min_size_mb=10):
    """Verify downloaded file exists and has reasonable size."""
    if not os.path.exists(filepath):
        return False

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if size_mb < min_size_mb:
        print(f"  Warning: File size ({size_mb:.1f} MB) seems too small")
        return False

    print(f"  File size: {size_mb:.1f} MB - OK")
    return True


def main():
    # Create pretrained weights directory
    weights_dir = os.path.join(os.path.dirname(__file__), 'pretrained_weights')
    os.makedirs(weights_dir, exist_ok=True)

    print("=" * 60)
    print("Downloading Pretrained I3D Weights")
    print("=" * 60)
    print(f"\nWeights directory: {weights_dir}\n")

    # Download primary weights (piergiaj)
    print("1. Downloading RGB ImageNet weights (piergiaj/pytorch-i3d)...")
    weight_info = WEIGHTS['rgb_imagenet']
    dest_path = os.path.join(weights_dir, weight_info['filename'])

    if os.path.exists(dest_path):
        print(f"  Already exists: {dest_path}")
        verify_file(dest_path)
    else:
        success = download_file(weight_info['url'], dest_path)
        if success:
            verify_file(dest_path)

    # Download alternative weights (hassony2 - includes Kinetics training)
    print("\n2. Downloading RGB ImageNet+Kinetics weights (hassony2/kinetics_i3d_pytorch)...")
    weight_info = WEIGHTS_ALT['rgb_imagenet_kinetics']
    dest_path = os.path.join(weights_dir, weight_info['filename'])

    if os.path.exists(dest_path):
        print(f"  Already exists: {dest_path}")
        verify_file(dest_path)
    else:
        success = download_file(weight_info['url'], dest_path)
        if success:
            verify_file(dest_path)

    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)

    # List downloaded files
    print("\nDownloaded files:")
    for f in os.listdir(weights_dir):
        filepath = os.path.join(weights_dir, f)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {f}: {size_mb:.1f} MB")

    print("\nUsage:")
    print("  # In config.py, set:")
    print(f"  PRETRAINED_WEIGHTS = '{os.path.join(weights_dir, 'rgb_imagenet.pt')}'")
    print("  USE_PRETRAINED = True")


if __name__ == "__main__":
    main()
