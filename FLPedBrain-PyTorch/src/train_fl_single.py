"""
FL Training - Single GPU per process version.

Each GPU handles 2 sites sequentially. Launch 8 processes with different GPU IDs.
Weights are synchronized via file system (like TF version).

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_fl_single.py --gpu-id 0 --num-gpus 8 ...
"""

import os
import sys
import time
import argparse
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# BF16 mixed precision
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import config
from dataset_fl import SiteDataset
from losses import CombinedLoss, dice_coefficient, F1Score


def get_model(use_pretrained=True, weights_path=None, freeze_encoder=False):
    """Get model based on configuration."""
    if use_pretrained:
        from model_pretrained import PedBrainNetPretrained, count_parameters
        model = PedBrainNetPretrained(
            num_classes=config.NUM_CLASSES,
            pretrained=True,
            weights_path=weights_path,
            freeze_encoder=freeze_encoder
        )
        model_name = "PedBrainNet_FL_Pretrained"
    else:
        from model import PedBrainNet, count_parameters
        model = PedBrainNet(num_classes=config.NUM_CLASSES)
        model_name = "PedBrainNet_FL_Scratch"

    return model, model_name, count_parameters


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch on a single site."""
    model.train()

    epoch_loss = 0.0
    epoch_dice = 0.0
    num_batches = len(train_loader)

    for images, masks, labels in train_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            seg_out, class_out, seg_logits, class_logits = model(images)
            total_loss, seg_loss, cls_loss = criterion(seg_out, masks, class_logits, labels)

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            dice = dice_coefficient((seg_out > 0.5).float(), masks)

        epoch_loss += total_loss.item()
        epoch_dice += dice.item()

    return epoch_loss / num_batches, epoch_dice / num_batches


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()

    epoch_loss = 0.0
    epoch_dice = 0.0
    num_batches = len(val_loader)

    f1_metric = F1Score(num_classes=config.NUM_CLASSES)
    all_predictions = []
    all_labels = []

    for images, masks, labels in val_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            seg_out, class_out, seg_logits, class_logits = model(images)
            total_loss, seg_loss, cls_loss = criterion(seg_out, masks, class_logits, labels)

        seg_pred_binary = (seg_out > 0.5).float()
        dice = dice_coefficient(seg_pred_binary, masks)

        f1_metric.update(class_out, labels)
        all_predictions.append(class_out.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

        epoch_loss += total_loss.item()
        epoch_dice += dice.item()

    per_class_f1, macro_f1 = f1_metric.compute()
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    accuracy = (all_predictions == all_labels).float().mean().item()

    return {
        'loss': epoch_loss / num_batches,
        'dice': epoch_dice / num_batches,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class_f1': per_class_f1
    }


def save_weights(model, path):
    """Save model weights to file."""
    state_dict = model.state_dict()
    torch.save(state_dict, path)


def load_weights(model, path):
    """Load model weights from file."""
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)


def fedavg_aggregate(weight_paths, scaling_factors, output_path):
    """FedAvg aggregation from weight files."""
    aggregated = None
    total_scale = sum(scaling_factors)

    for path, scale in zip(weight_paths, scaling_factors):
        state_dict = torch.load(path, map_location='cpu')
        if aggregated is None:
            aggregated = {k: v.float() * (scale / total_scale) for k, v in state_dict.items()}
        else:
            for k, v in state_dict.items():
                aggregated[k] += v.float() * (scale / total_scale)

    torch.save(aggregated, output_path)
    return aggregated


def wait_for_file(path, timeout=600, check_interval=1):
    """Wait for a file to exist."""
    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for {path}")
        time.sleep(check_interval)
    # Small delay to ensure file is fully written
    time.sleep(0.5)


def main(args):
    # Setup
    device = torch.device('cuda')
    gpu_id = args.gpu_id
    num_gpus = args.num_gpus

    # Determine which sites this GPU handles
    if args.sites:
        # Explicit site list provided (e.g., for warm start on ST,SE)
        all_sites = args.sites.split(',')
        if num_gpus == 1:
            my_sites = all_sites
        else:
            # Distribute sites evenly
            sites_per_gpu = len(all_sites) // num_gpus
            my_sites = all_sites[gpu_id * sites_per_gpu : (gpu_id + 1) * sites_per_gpu]
    else:
        # Use balanced FL_GPU_SITES allocation from config
        if hasattr(config, 'FL_GPU_SITES') and gpu_id < len(config.FL_GPU_SITES):
            my_sites = config.FL_GPU_SITES[gpu_id]
            # Flatten all sites for aggregation
            all_sites = [s for gpu_sites in config.FL_GPU_SITES for s in gpu_sites]
        else:
            # Fallback: equal distribution
            sites_per_gpu = len(config.TRAIN_SITE_IDS) // num_gpus
            my_sites = config.TRAIN_SITE_IDS[gpu_id * sites_per_gpu : (gpu_id + 1) * sites_per_gpu]
            all_sites = config.TRAIN_SITE_IDS

    print(f"GPU {gpu_id}: Handling sites {my_sites}")

    # Directories
    checkpoint_dir = config.CHECKPOINT_DIR
    weights_dir = os.path.join(checkpoint_dir, 'fl_weights')
    os.makedirs(weights_dir, exist_ok=True)

    # Create model
    model, model_name, count_parameters = get_model(
        use_pretrained=not args.no_pretrained,
        weights_path=config.PRETRAINED_WEIGHTS if not args.no_pretrained else None,
        freeze_encoder=args.freeze_encoder
    )
    model = model.to(device)

    if gpu_id == 0:
        print(f"Model: {model_name}")
        print(f"Parameters: {count_parameters(model, trainable_only=True):,}")

    # Loss
    criterion = CombinedLoss(dice_weight=args.dice_weight)

    # Load site data
    site_datasets = {}
    site_loaders = {}
    site_sizes = {}

    for site_id in my_sites:
        dataset = SiteDataset(
            data_dir=config.DATA_DIR,
            site_id=site_id,
            normals_file=config.NORMALS_TRAIN,
            augment=True,
            split='train'
        )
        site_datasets[site_id] = dataset
        site_loaders[site_id] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        site_sizes[site_id] = len(dataset)
        print(f"GPU {gpu_id}: Site {site_id} has {len(dataset)} samples")

    # Validation loader (only GPU 0 validates)
    val_loader = None
    if gpu_id == 0:
        from dataset import PedBrainDataset
        val_dataset = PedBrainDataset(
            data_dir=config.DATA_DIR,
            site_ids=None,
            include_normals=True,
            normals_file=config.NORMALS_VAL,
            augment=False,
            split='val',
            normals_fraction=args.normals_fraction
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size_eval,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    # Compute scaling factors (based on dataset sizes)
    total_samples = sum(site_sizes.values())

    # Training
    best_dice = 0.0
    global_weights_path = os.path.join(weights_dir, 'global_weights.pth')

    # Initialize global weights (GPU 0 saves initial weights or loads from resume)
    if gpu_id == 0:
        if args.resume and os.path.exists(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Loaded weights from checkpoint")
        save_weights(model, global_weights_path)
        print("Initialized global weights")

    # Barrier: wait for initial weights
    wait_for_file(global_weights_path)
    load_weights(model, global_weights_path)

    for fl_round in range(args.num_rounds):
        round_start = time.time()

        if gpu_id == 0:
            print(f"\n{'='*60}")
            print(f"FL Round {fl_round + 1}/{args.num_rounds}")
            print(f"{'='*60}")

        # Train on each of my sites
        for site_id in my_sites:
            # Load global weights
            load_weights(model, global_weights_path)

            # Create optimizer (fresh each round, like TF)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            # Train for local epochs
            for local_epoch in range(args.local_epochs):
                loss, dice = train_one_epoch(
                    model, site_loaders[site_id], criterion, optimizer, device
                )

            print(f"GPU {gpu_id} Site {site_id}: Loss={loss:.4f}, Dice={dice:.4f}")

            # Save local weights
            local_weights_path = os.path.join(weights_dir, f'round{fl_round}_site{site_id}.pth')
            save_weights(model, local_weights_path)

        # Barrier: GPU 0 waits for all sites, then aggregates
        if gpu_id == 0:
            # Wait for all site weights
            all_weight_paths = []
            all_scales = []

            for site_id in all_sites:
                path = os.path.join(weights_dir, f'round{fl_round}_site{site_id}.pth')
                wait_for_file(path, timeout=1200)
                all_weight_paths.append(path)
                # Get site size from filename or use uniform
                all_scales.append(1.0)  # Uniform for simplicity

            # Aggregate
            print("Aggregating weights...")
            fedavg_aggregate(all_weight_paths, all_scales, global_weights_path)

            # Validate
            load_weights(model, global_weights_path)
            val_metrics = validate(model, val_loader, criterion, device)

            print(f"Val - Dice: {val_metrics['dice']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['macro_f1']:.4f}")

            # Save best
            if val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
                torch.save({
                    'round': fl_round,
                    'model_state_dict': model.state_dict(),
                    'metrics': val_metrics
                }, os.path.join(checkpoint_dir, 'fl_best.pth'))
                print(f"New best Dice: {best_dice:.4f}")

            # Periodic save
            if (fl_round + 1) % args.save_every == 0:
                torch.save({
                    'round': fl_round,
                    'model_state_dict': model.state_dict(),
                    'metrics': val_metrics
                }, os.path.join(checkpoint_dir, f'fl_round{fl_round + 1}.pth'))

            # Log to CSV
            log_file = os.path.join(config.LOG_DIR, 'fl_training.csv')
            if fl_round == 0:
                with open(log_file, 'w') as f:
                    f.write('round,val_loss,val_dice,val_accuracy,val_f1\n')
            with open(log_file, 'a') as f:
                f.write(f"{fl_round},{val_metrics['loss']:.4f},{val_metrics['dice']:.4f},"
                        f"{val_metrics['accuracy']:.4f},{val_metrics['macro_f1']:.4f}\n")

            # Signal round complete
            with open(os.path.join(weights_dir, f'round{fl_round}_done.txt'), 'w') as f:
                f.write('done')

            print(f"Round {fl_round + 1} took {time.time() - round_start:.1f}s")

        else:
            # Non-zero GPUs wait for aggregation to complete
            done_file = os.path.join(weights_dir, f'round{fl_round}_done.txt')
            wait_for_file(done_file, timeout=1200)

    # Final
    if gpu_id == 0:
        print(f"\n{'='*60}")
        print("FL Training Complete!")
        print(f"Best Dice: {best_dice:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL Training - Single GPU')
    parser.add_argument('--gpu-id', type=int, required=True, help='GPU ID (0-7)')
    parser.add_argument('--num-gpus', type=int, default=8, help='Total number of GPUs')
    parser.add_argument('--sites', type=str, default=None,
                        help='Comma-separated site IDs (e.g., ST,SE for warm start). '
                             'Default: all 16 sites distributed across GPUs')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--batch-size-eval', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-rounds', type=int, default=50)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dice-weight', type=float, default=0.5)
    parser.add_argument('--normals-fraction', type=float, default=0.25)
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--freeze-encoder', action='store_true')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., warm start weights)')

    args = parser.parse_args()
    main(args)
