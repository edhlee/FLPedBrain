"""
Training script for CDS (Centralized Data Sharing) baseline.

Trains a 3D U-Net on all concatenated site data for brain tumor
segmentation and classification.
"""

import os
import sys
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# BF16 mixed precision 
# Enable TF32 for any FP32 operations (faster matmuls on Ampere+)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import config
from dataset import get_dataloaders
from losses import CombinedLoss, dice_coefficient, compute_metrics, F1Score


def get_model(use_pretrained=None, weights_path=None, freeze_encoder=None):
    """
    Get model based on configuration.

    Args:
        use_pretrained: Override config.USE_PRETRAINED
        weights_path: Override config.PRETRAINED_WEIGHTS
        freeze_encoder: Override config.FREEZE_ENCODER

    Returns:
        model, model_name
    """
    if use_pretrained is None:
        use_pretrained = config.USE_PRETRAINED
    if weights_path is None:
        weights_path = config.PRETRAINED_WEIGHTS
    if freeze_encoder is None:
        freeze_encoder = config.FREEZE_ENCODER

    if use_pretrained:
        from model_pretrained import PedBrainNetPretrained, count_parameters
        model = PedBrainNetPretrained(
            num_classes=config.NUM_CLASSES,
            pretrained=True,
            weights_path=weights_path,
            freeze_encoder=freeze_encoder
        )
        model_name = "PedBrainNet_CDS_Pretrained"
    else:
        from model import PedBrainNet, count_parameters
        model = PedBrainNet(num_classes=config.NUM_CLASSES)
        model_name = "PedBrainNet_CDS_Scratch"

    return model, model_name, count_parameters


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    """Train for one epoch."""
    model.train()

    epoch_loss = 0.0
    epoch_seg_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_dice = 0.0
    num_batches = len(train_loader)

    start_time = time.time()

    for batch_idx, (images, masks, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # BF16 mixed precision forward pass (no GradScaler needed for BF16)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            seg_out, class_out, seg_logits, class_logits = model(images)
            total_loss, seg_loss, cls_loss = criterion(seg_out, masks, class_logits, labels)

        # Standard backward pass (no gradient scaling needed for BF16)
        total_loss.backward()
        optimizer.step()

        # Compute Dice coefficient
        with torch.no_grad():
            dice = dice_coefficient((seg_out > 0.5).float(), masks)

        # Accumulate metrics
        epoch_loss += total_loss.item()
        epoch_seg_loss += seg_loss.item()
        epoch_cls_loss += cls_loss.item()
        epoch_dice += dice.item()

        # Log every 10 batches
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * config.BATCH_SIZE / elapsed
            print(f"  Batch [{batch_idx + 1}/{num_batches}] "
                  f"Loss: {total_loss.item():.4f} "
                  f"(Seg: {seg_loss.item():.4f}, Cls: {cls_loss.item():.4f}) "
                  f"Dice: {dice.item():.4f} "
                  f"({samples_per_sec:.1f} samples/s)")

    # Average metrics
    epoch_loss /= num_batches
    epoch_seg_loss /= num_batches
    epoch_cls_loss /= num_batches
    epoch_dice /= num_batches

    # Log to tensorboard
    global_step = epoch * num_batches
    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/SegLoss', epoch_seg_loss, epoch)
    writer.add_scalar('Train/ClsLoss', epoch_cls_loss, epoch)
    writer.add_scalar('Train/Dice', epoch_dice, epoch)

    return epoch_loss, epoch_seg_loss, epoch_cls_loss, epoch_dice


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, writer):
    """Validate the model."""
    model.eval()

    epoch_loss = 0.0
    epoch_seg_loss = 0.0
    epoch_cls_loss = 0.0
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

        # Compute metrics
        seg_pred_binary = (seg_out > 0.5).float()
        dice = dice_coefficient(seg_pred_binary, masks)

        # F1 score
        f1_metric.update(class_out, labels)
        all_predictions.append(class_out.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

        epoch_loss += total_loss.item()
        epoch_seg_loss += seg_loss.item()
        epoch_cls_loss += cls_loss.item()
        epoch_dice += dice.item()

    # Average metrics
    epoch_loss /= num_batches
    epoch_seg_loss /= num_batches
    epoch_cls_loss /= num_batches
    epoch_dice /= num_batches

    # Compute F1 scores
    per_class_f1, macro_f1 = f1_metric.compute()

    # Classification accuracy
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    accuracy = (all_predictions == all_labels).float().mean().item()

    # Log to tensorboard
    writer.add_scalar('Val/Loss', epoch_loss, epoch)
    writer.add_scalar('Val/SegLoss', epoch_seg_loss, epoch)
    writer.add_scalar('Val/ClsLoss', epoch_cls_loss, epoch)
    writer.add_scalar('Val/Dice', epoch_dice, epoch)
    writer.add_scalar('Val/Accuracy', accuracy, epoch)
    writer.add_scalar('Val/MacroF1', macro_f1, epoch)

    for i, f1 in enumerate(per_class_f1):
        writer.add_scalar(f'Val/F1_Class{i}', f1, epoch)

    return {
        'loss': epoch_loss,
        'seg_loss': epoch_seg_loss,
        'cls_loss': epoch_cls_loss,
        'dice': epoch_dice,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class_f1': per_class_f1
    }


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)
    print(f"Checkpoint saved: {path}")


def main(args):
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Setup device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Mixed Precision: BF16 + TF32 enabled")

    # Create model
    print("\nCreating model...")
    use_pretrained = not args.no_pretrained if hasattr(args, 'no_pretrained') else config.USE_PRETRAINED
    model, model_name, count_parameters = get_model(
        use_pretrained=use_pretrained,
        weights_path=config.PRETRAINED_WEIGHTS if use_pretrained else None,
        freeze_encoder=config.FREEZE_ENCODER
    )
    model = model.to(device)
    print(f"Model: {model_name}")
    print(f"Total parameters: {count_parameters(model, trainable_only=False):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

    # Create dataloaders
    print("\nLoading data...")
    normals_fraction = args.normals_fraction if args.normals_fraction is not None else None
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size or config.BATCH_SIZE,
        batch_size_eval=args.batch_size_eval or config.BATCH_SIZE_EVAL,
        num_workers=args.num_workers,
        normals_fraction=normals_fraction
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Loss function
    dice_weight = args.dice_weight if args.dice_weight is not None else config.DICE_LOSS_WEIGHT
    criterion = CombinedLoss(dice_weight=dice_weight)
    print(f"Loss weights: dice_weight={dice_weight}")

    # Optimizer with learning rate schedule
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Learning rate scheduler (exponential decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_DECAY_STEPS,
        gamma=config.LR_DECAY_RATE
    )

    # Load optimizer state if resuming
    if args.resume and os.path.exists(args.resume):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Adjust scheduler to correct epoch
        for _ in range(start_epoch):
            scheduler.step()

    # Determine number of epochs
    num_epochs = args.epochs if args.epochs is not None else config.NUM_EPOCHS

    # TensorBoard writer
    run_name = f"cds_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(os.path.join(config.LOG_DIR, run_name))

    # Log hyperparameters
    writer.add_text('Hyperparameters', f"""
    Batch Size: {args.batch_size or config.BATCH_SIZE}
    Learning Rate: {config.LEARNING_RATE}
    Dice Loss Weight: {config.DICE_LOSS_WEIGHT}
    Epochs: {num_epochs}
    """)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting CDS Baseline Training")
    print("=" * 60)

    best_dice = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # Train
        train_loss, train_seg_loss, train_cls_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        print(f"Train - Loss: {train_loss:.4f}, Seg: {train_seg_loss:.4f}, "
              f"Cls: {train_cls_loss:.4f}, Dice: {train_dice:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, writer)

        print(f"Val - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['macro_f1']:.4f}")
        print(f"Per-class F1: {[f'{f:.3f}' for f in val_metrics['per_class_f1']]}")

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LearningRate', current_lr, epoch)
        print(f"Learning rate: {current_lr:.6f}")

        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(config.CHECKPOINT_DIR, 'cds_best.pth')
            )
            print(f"New best Dice: {best_dice:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(config.CHECKPOINT_DIR, f'cds_epoch{epoch + 1}.pth')
            )

        # Log to CSV (detailed metrics including seg/cls breakdown)
        log_file = os.path.join(config.LOG_DIR, f'{run_name}.csv')
        if epoch == start_epoch:
            with open(log_file, 'w') as f:
                f.write('epoch,train_loss,train_seg_loss,train_cls_loss,train_dice,'
                        'val_loss,val_seg_loss,val_cls_loss,val_dice,val_accuracy,val_f1\n')
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{train_seg_loss:.4f},{train_cls_loss:.4f},{train_dice:.4f},"
                    f"{val_metrics['loss']:.4f},{val_metrics['seg_loss']:.4f},{val_metrics['cls_loss']:.4f},"
                    f"{val_metrics['dice']:.4f},{val_metrics['accuracy']:.4f},{val_metrics['macro_f1']:.4f}\n")

    # Save final model
    save_checkpoint(
        model, optimizer, num_epochs - 1, val_metrics,
        os.path.join(config.CHECKPOINT_DIR, 'cds_final.pth')
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Dice: {best_dice:.4f} at epoch {best_epoch + 1}")
    print("=" * 60)

    writer.close()

    # Generate training plots
    try:
        from plot_training import plot_training_curves
        print("\nGenerating training plots...")
        plot_training_curves(log_file, save=True)
    except Exception as e:
        print(f"Could not generate plots: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CDS Baseline Training')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Training batch size (default: from config)')
    parser.add_argument('--batch-size-eval', type=int, default=None,
                        help='Evaluation batch size (default: from config)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Train from scratch without pretrained weights')
    parser.add_argument('--dice-weight', type=float, default=None,
                        help='Weight for dice loss (default: from config, 0.5)')
    parser.add_argument('--normals-fraction', type=float, default=None,
                        help='Fraction of normals to use relative to tumor samples '
                             '(default: from config, 0.25 to match TF). Set to 0 to exclude normals.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')

    args = parser.parse_args()
    main(args)
