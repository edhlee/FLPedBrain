"""
Evaluation script for brain segmentation model.

Supports:
1. Evaluation on validation set with comprehensive metrics
2. Inference on DICOM files with segmentation overlay visualization

Computes comprehensive metrics including:
- Dice score (per-sample and mean)
- Classification accuracy and F1 score
- Per-class metrics
- Confusion matrix visualization
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import config
from dataset import get_dataloaders, PedBrainDataset
from losses import dice_coefficient, F1Score


def get_model(checkpoint_path, device, use_pretrained_arch=True):
    """Load model from checkpoint."""
    if use_pretrained_arch:
        from model_pretrained import PedBrainNetPretrained
        model = PedBrainNetPretrained(
            num_classes=config.NUM_CLASSES,
            pretrained=False
        )
    else:
        from model import PedBrainNet
        model = PedBrainNet(num_classes=config.NUM_CLASSES)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from epoch/round {checkpoint.get('epoch', checkpoint.get('round', 'unknown'))}")
        if 'metrics' in checkpoint:
            print(f"Checkpoint metrics: {checkpoint['metrics']}")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def evaluate_model(model, val_loader, device, save_dir=None):
    """
    Comprehensive evaluation of the model.

    Returns:
        Dictionary containing all metrics
    """
    model.eval()

    # Storage for metrics
    all_dice_scores = []
    all_predictions = []
    all_labels = []
    all_seg_pred = []
    all_seg_true = []

    # Per-class Dice scores
    class_dice_scores = {i: [] for i in range(config.NUM_CLASSES)}

    print("Evaluating model...")

    with torch.no_grad():
        for images, masks, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                seg_out, class_out, _, _ = model(images)

            # Binarize segmentation prediction
            seg_pred = (seg_out > 0.5).float()

            # Per-sample Dice score
            for i in range(images.size(0)):
                sample_dice = dice_coefficient(
                    seg_pred[i:i+1], masks[i:i+1]
                ).item()
                all_dice_scores.append(sample_dice)

                # Store per-class Dice
                label = labels[i].item()
                class_dice_scores[label].append(sample_dice)

            # Store predictions and labels
            all_predictions.append(class_out.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

            # Store some samples for visualization
            if len(all_seg_pred) < 10:
                all_seg_pred.append(seg_pred[0].cpu().numpy())
                all_seg_true.append(masks[0].cpu().numpy())

    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute overall metrics
    results = {
        'mean_dice': np.mean(all_dice_scores),
        'std_dice': np.std(all_dice_scores),
        'median_dice': np.median(all_dice_scores),
        'all_dice_scores': all_dice_scores,
    }

    # Per-class Dice scores
    print("\nPer-class Dice Scores:")
    class_names = config.CLASS_NAMES  # ["Normal Controls", "Ependymoma", "DIPG", "Medulloblastoma", "Pilocytic"]
    for i in range(config.NUM_CLASSES):
        if class_dice_scores[i]:
            mean_dice = np.mean(class_dice_scores[i])
            std_dice = np.std(class_dice_scores[i])
            n_samples = len(class_dice_scores[i])
            print(f"  Class {i} ({class_names[i]}): {mean_dice:.4f} ± {std_dice:.4f} (n={n_samples})")
            results[f'dice_class_{i}'] = mean_dice
            results[f'dice_std_class_{i}'] = std_dice
        else:
            print(f"  Class {i} ({class_names[i]}): No samples")
            results[f'dice_class_{i}'] = float('nan')

    # Classification metrics
    accuracy = (all_predictions == all_labels).mean()
    results['classification_accuracy'] = accuracy

    # Confusion matrix
    confusion_matrix = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=int)
    for pred, true in zip(all_predictions, all_labels):
        confusion_matrix[true, pred] += 1
    results['confusion_matrix'] = confusion_matrix

    # Per-class precision, recall, F1
    print("\nClassification Metrics:")
    per_class_f1 = []
    per_class_precision = []
    per_class_recall = []

    for c in range(config.NUM_CLASSES):
        tp = confusion_matrix[c, c]
        fp = confusion_matrix[:, c].sum() - tp
        fn = confusion_matrix[c, :].sum() - tp

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        per_class_f1.append(f1)
        per_class_precision.append(precision)
        per_class_recall.append(recall)

        print(f"  Class {c} ({class_names[c]}): P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    results['per_class_f1'] = per_class_f1
    results['per_class_precision'] = per_class_precision
    results['per_class_recall'] = per_class_recall
    results['macro_f1'] = np.mean(per_class_f1)

    # Save visualizations
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Plot Dice score distribution
        plt.figure(figsize=(10, 6))
        plt.hist(all_dice_scores, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(results['mean_dice'], color='r', linestyle='--',
                    label=f'Mean: {results["mean_dice"]:.4f}')
        plt.axvline(results['median_dice'], color='g', linestyle='--',
                    label=f'Median: {results["median_dice"]:.4f}')
        plt.xlabel('Dice Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Dice Scores')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'dice_distribution.png'), dpi=150)
        plt.close()

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

        # Plot per-class Dice scores
        plt.figure(figsize=(12, 6))
        x = np.arange(config.NUM_CLASSES)
        means = [results.get(f'dice_class_{i}', 0) for i in range(config.NUM_CLASSES)]
        stds = [results.get(f'dice_std_class_{i}', 0) for i in range(config.NUM_CLASSES)]
        plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.ylabel('Dice Score')
        plt.title('Per-class Dice Scores')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_class_dice.png'), dpi=150)
        plt.close()

        # Plot segmentation examples
        fig, axes = plt.subplots(min(5, len(all_seg_pred)), 3, figsize=(12, 4 * min(5, len(all_seg_pred))))
        if len(all_seg_pred) == 1:
            axes = axes.reshape(1, -1)

        for i in range(min(5, len(all_seg_pred))):
            # Select middle slice
            mid_slice = all_seg_pred[i].shape[1] // 2

            # Input (from dataset)
            axes[i, 0].set_title(f'Sample {i+1}: Prediction')
            axes[i, 0].imshow(all_seg_pred[i][0, mid_slice], cmap='gray')
            axes[i, 0].axis('off')

            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].imshow(all_seg_true[i][0, mid_slice], cmap='gray')
            axes[i, 1].axis('off')

            # Overlay
            axes[i, 2].set_title('Overlay (Green=TP, Red=FN, Blue=FP)')
            overlay = np.zeros((*all_seg_pred[i][0, mid_slice].shape, 3))
            pred = all_seg_pred[i][0, mid_slice] > 0.5
            true = all_seg_true[i][0, mid_slice] > 0.5
            overlay[pred & true] = [0, 1, 0]  # TP: green
            overlay[~pred & true] = [1, 0, 0]  # FN: red
            overlay[pred & ~true] = [0, 0, 1]  # FP: blue
            axes[i, 2].imshow(overlay)
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'segmentation_examples.png'), dpi=150)
        plt.close()

        print(f"\nVisualizations saved to: {save_dir}")

    return results


def print_summary(results):
    """Print a summary of evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print("\nSegmentation Metrics:")
    print(f"  Mean Dice Score: {results['mean_dice']:.4f} ± {results['std_dice']:.4f}")
    print(f"  Median Dice Score: {results['median_dice']:.4f}")

    print("\nClassification Metrics:")
    print(f"  Accuracy: {results['classification_accuracy']:.4f}")
    print(f"  Macro F1: {results['macro_f1']:.4f}")

    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

    print("=" * 60)


# =============================================================================
# DICOM Inference Functions
# =============================================================================

def load_dicom_volume(dicom_dir):
    """Load DICOM files from directory and create a 3D volume."""
    try:
        import pydicom
    except ImportError:
        print("pydicom not installed. Install with: pip install pydicom")
        return None, None

    dicom_files = sorted(Path(dicom_dir).glob('*.dcm'))
    if not dicom_files:
        dicom_files = sorted(Path(dicom_dir).glob('**/*.dcm'))

    if not dicom_files:
        print(f"No DICOM files found in {dicom_dir}")
        return None, None

    print(f"Found {len(dicom_files)} DICOM files")

    # Read all slices with instance numbers for proper ordering
    slices_data = []
    for dcm_path in dicom_files:
        ds = pydicom.dcmread(str(dcm_path))
        instance_num = getattr(ds, 'InstanceNumber', 0)
        slices_data.append((instance_num, ds.pixel_array.astype(np.float32), ds))

    # Sort by instance number
    slices_data.sort(key=lambda x: x[0])
    slices = [s[1] for s in slices_data]

    # Stack into volume (D, H, W)
    volume = np.stack(slices, axis=0)

    # Get metadata from first slice
    metadata = slices_data[0][2] if slices_data else None

    # Normalize to [0, 1]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-7)

    print(f"Volume shape: {volume.shape}")
    print(f"Volume range: [{volume.min():.3f}, {volume.max():.3f}]")

    return volume, metadata


def preprocess_volume_for_inference(volume, target_shape=(48, 240, 240)):
    """Preprocess volume for model inference."""
    from scipy.ndimage import zoom

    D, H, W = volume.shape
    target_D, target_H, target_W = target_shape

    # Resize if needed
    if (D, H, W) != target_shape:
        zoom_factors = (target_D / D, target_H / H, target_W / W)
        volume_resized = zoom(volume, zoom_factors, order=1)
        print(f"Resized from {(D, H, W)} to {volume_resized.shape}")
    else:
        volume_resized = volume

    # Add channel dimension and batch dimension
    # Model expects (B, C, D, H, W) where C=1
    tensor = torch.from_numpy(volume_resized).float().unsqueeze(0).unsqueeze(0)

    return tensor, volume


@torch.no_grad()
def inference_dicom(model, dicom_dir, device, save_dir):
    """Run inference on DICOM volume and save visualizations."""
    class_names = config.CLASS_NAMES

    # Load DICOM
    volume, metadata = load_dicom_volume(dicom_dir)
    if volume is None:
        return

    # Preprocess
    input_tensor, original_volume = preprocess_volume_for_inference(volume)
    input_tensor = input_tensor.to(device)

    print(f"Input tensor shape: {input_tensor.shape}")

    # Inference
    model.eval()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        seg_out, class_out, _, _ = model(input_tensor)

    # Convert from BFloat16 to float32 before numpy conversion
    seg_pred = (seg_out > 0.5).float().cpu().numpy()[0, 0]  # (D, H, W)
    seg_prob = seg_out.float().cpu().numpy()[0, 0]  # Probability map
    class_probs = torch.softmax(class_out.float(), dim=1).cpu().numpy()[0]
    pred_class = class_out.argmax(dim=1).item()

    print(f"\nPredicted class: {class_names[pred_class]} ({pred_class})")
    print(f"Class probabilities:")
    for i, (name, prob) in enumerate(zip(class_names, class_probs)):
        print(f"  {name}: {prob:.3f}")
    print(f"Segmentation: {seg_pred.sum():.0f} voxels predicted as tumor")

    # Resize prediction back to original volume size if needed
    if seg_pred.shape != original_volume.shape:
        from scipy.ndimage import zoom
        zoom_factors = tuple(o / p for o, p in zip(original_volume.shape, seg_pred.shape))
        seg_pred_original = zoom(seg_pred, zoom_factors, order=0)  # Nearest neighbor for binary
        seg_prob_original = zoom(seg_prob, zoom_factors, order=1)
    else:
        seg_pred_original = seg_pred
        seg_prob_original = seg_prob

    # Save visualizations
    os.makedirs(save_dir, exist_ok=True)

    # Save slice-by-slice with overlays
    num_slices = original_volume.shape[0]
    slices_with_tumor = []

    for slice_idx in range(num_slices):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        img_slice = original_volume[slice_idx]
        pred_slice = seg_pred_original[slice_idx]
        prob_slice = seg_prob_original[slice_idx]

        tumor_voxels = pred_slice.sum()
        if tumor_voxels > 0:
            slices_with_tumor.append(slice_idx)

        # Original
        axes[0].imshow(img_slice, cmap='gray')
        axes[0].set_title(f'Slice {slice_idx + 1}/{num_slices}')
        axes[0].axis('off')

        # Overlay
        axes[1].imshow(img_slice, cmap='gray')
        if tumor_voxels > 0:
            # Create colored overlay
            overlay = np.zeros((*img_slice.shape, 4))
            overlay[pred_slice > 0] = [1, 0, 0, 0.5]  # Red with alpha
            axes[1].imshow(overlay)
        axes[1].set_title(f'Segmentation Overlay ({int(tumor_voxels)} voxels)')
        axes[1].axis('off')

        # Probability heatmap
        im = axes[2].imshow(img_slice, cmap='gray')
        if prob_slice.max() > 0.1:
            prob_masked = np.ma.masked_where(prob_slice < 0.1, prob_slice)
            axes[2].imshow(prob_masked, cmap='hot', alpha=0.7, vmin=0, vmax=1)
        axes[2].set_title(f'Probability Map')
        axes[2].axis('off')

        plt.suptitle(f'Prediction: {class_names[pred_class]} (confidence: {class_probs[pred_class]:.2f})')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'slice_{slice_idx:03d}.png'), dpi=100, bbox_inches='tight')
        plt.close()

    # Save summary montage focusing on slices with tumor
    if slices_with_tumor:
        selected_slices = slices_with_tumor
    else:
        # If no tumor detected, show evenly spaced slices
        step = max(1, num_slices // 12)
        selected_slices = list(range(0, num_slices, step))

    n_cols = min(6, len(selected_slices))
    n_rows = (len(selected_slices) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, slice_idx in enumerate(selected_slices[:n_rows * n_cols]):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        ax.imshow(original_volume[slice_idx], cmap='gray')
        if seg_pred_original[slice_idx].sum() > 0:
            overlay = np.zeros((*original_volume[slice_idx].shape, 4))
            overlay[seg_pred_original[slice_idx] > 0] = [1, 0, 0, 0.5]
            ax.imshow(overlay)
        ax.set_title(f'Slice {slice_idx}', fontsize=8)
        ax.axis('off')

    # Hide unused axes
    for idx in range(len(selected_slices), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    plt.suptitle(f'Prediction: {class_names[pred_class]} (conf: {class_probs[pred_class]:.2f})\n'
                 f'Tumor voxels: {seg_pred_original.sum():.0f} | Slices with tumor: {len(slices_with_tumor)}',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'montage.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save 3D visualization (axial, coronal, sagittal views)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Find center of tumor or use volume center
    if seg_pred_original.sum() > 0:
        tumor_coords = np.where(seg_pred_original > 0)
        center_d = int(np.mean(tumor_coords[0]))
        center_h = int(np.mean(tumor_coords[1]))
        center_w = int(np.mean(tumor_coords[2]))
    else:
        center_d = original_volume.shape[0] // 2
        center_h = original_volume.shape[1] // 2
        center_w = original_volume.shape[2] // 2

    # Axial (D slice)
    axes[0, 0].imshow(original_volume[center_d], cmap='gray')
    axes[0, 0].set_title(f'Axial (slice {center_d})')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(original_volume[center_d], cmap='gray')
    if seg_pred_original[center_d].sum() > 0:
        overlay = np.zeros((*original_volume[center_d].shape, 4))
        overlay[seg_pred_original[center_d] > 0] = [1, 0, 0, 0.5]
        axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Axial + Overlay')
    axes[1, 0].axis('off')

    # Coronal (H slice)
    axes[0, 1].imshow(original_volume[:, center_h, :], cmap='gray', aspect='auto')
    axes[0, 1].set_title(f'Coronal (slice {center_h})')
    axes[0, 1].axis('off')

    axes[1, 1].imshow(original_volume[:, center_h, :], cmap='gray', aspect='auto')
    if seg_pred_original[:, center_h, :].sum() > 0:
        overlay = np.zeros((*original_volume[:, center_h, :].shape, 4))
        overlay[seg_pred_original[:, center_h, :] > 0] = [1, 0, 0, 0.5]
        axes[1, 1].imshow(overlay, aspect='auto')
    axes[1, 1].set_title('Coronal + Overlay')
    axes[1, 1].axis('off')

    # Sagittal (W slice)
    axes[0, 2].imshow(original_volume[:, :, center_w], cmap='gray', aspect='auto')
    axes[0, 2].set_title(f'Sagittal (slice {center_w})')
    axes[0, 2].axis('off')

    axes[1, 2].imshow(original_volume[:, :, center_w], cmap='gray', aspect='auto')
    if seg_pred_original[:, :, center_w].sum() > 0:
        overlay = np.zeros((*original_volume[:, :, center_w].shape, 4))
        overlay[seg_pred_original[:, :, center_w] > 0] = [1, 0, 0, 0.5]
        axes[1, 2].imshow(overlay, aspect='auto')
    axes[1, 2].set_title('Sagittal + Overlay')
    axes[1, 2].axis('off')

    plt.suptitle(f'3D Views - {class_names[pred_class]} (conf: {class_probs[pred_class]:.2f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'orthogonal_views.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved {num_slices} slice images to {save_dir}")
    print(f"Saved montage.png and orthogonal_views.png")

    # Save segmentation as numpy
    np.save(os.path.join(save_dir, 'segmentation.npy'), seg_pred_original)
    np.save(os.path.join(save_dir, 'probability_map.npy'), seg_prob_original)
    print(f"Saved segmentation.npy and probability_map.npy")

    # Save prediction summary
    with open(os.path.join(save_dir, 'prediction_summary.txt'), 'w') as f:
        f.write(f"DICOM Inference Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Input: {dicom_dir}\n")
        f.write(f"Volume shape: {original_volume.shape}\n\n")
        f.write(f"Predicted Class: {class_names[pred_class]}\n")
        f.write(f"Confidence: {class_probs[pred_class]:.4f}\n\n")
        f.write(f"Class Probabilities:\n")
        for name, prob in zip(class_names, class_probs):
            f.write(f"  {name}: {prob:.4f}\n")
        f.write(f"\nSegmentation:\n")
        f.write(f"  Total tumor voxels: {seg_pred_original.sum():.0f}\n")
        f.write(f"  Slices with tumor: {len(slices_with_tumor)}\n")
        if slices_with_tumor:
            f.write(f"  Tumor slice range: {min(slices_with_tumor)} - {max(slices_with_tumor)}\n")

    return {
        'predicted_class': pred_class,
        'class_name': class_names[pred_class],
        'class_probs': class_probs,
        'tumor_voxels': seg_pred_original.sum(),
        'slices_with_tumor': len(slices_with_tumor)
    }


def main(args):
    # Setup device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = get_model(args.checkpoint, device, use_pretrained_arch=not args.scratch_model)

    # Create output directory
    output_dir = args.output_dir or 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)

    # Run evaluation based on mode
    if args.mode in ['val', 'both']:
        print("\n" + "=" * 60)
        print("Evaluating on Validation Set")
        print("=" * 60)

        # Load data
        print("\nLoading validation data...")
        _, val_loader = get_dataloaders(
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size,
            num_workers=args.num_workers
        )
        print(f"Validation batches: {len(val_loader)}")

        # Evaluate
        val_save_dir = os.path.join(output_dir, 'validation')
        results = evaluate_model(model, val_loader, device, save_dir=val_save_dir)

        # Print summary
        print_summary(results)

        # Save results to file
        results_file = os.path.join(val_save_dir, 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n\n")

            f.write("Segmentation Metrics:\n")
            f.write(f"  Mean Dice Score: {results['mean_dice']:.4f} +/- {results['std_dice']:.4f}\n")
            f.write(f"  Median Dice Score: {results['median_dice']:.4f}\n\n")

            f.write("Classification Metrics:\n")
            f.write(f"  Accuracy: {results['classification_accuracy']:.4f}\n")
            f.write(f"  Macro F1: {results['macro_f1']:.4f}\n\n")

            f.write("Per-class Metrics:\n")
            class_names = config.CLASS_NAMES
            for i, name in enumerate(class_names):
                dice = results.get(f'dice_class_{i}', float('nan'))
                f1 = results['per_class_f1'][i]
                prec = results['per_class_precision'][i]
                rec = results['per_class_recall'][i]
                f.write(f"  {name}: Dice={dice:.4f}, P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}\n")

            f.write("\nConfusion Matrix:\n")
            f.write(str(results['confusion_matrix']) + "\n")

        print(f"\nValidation results saved to: {val_save_dir}")

    if args.mode in ['dicom', 'both']:
        print("\n" + "=" * 60)
        print("Inference on DICOM Volume")
        print("=" * 60)

        if not args.dicom_dir:
            print("ERROR: --dicom-dir required for DICOM inference")
            return

        dicom_save_dir = os.path.join(output_dir, 'dicom_inference')
        dicom_results = inference_dicom(model, args.dicom_dir, device, dicom_save_dir)

        if dicom_results:
            print(f"\nDICOM inference results saved to: {dicom_save_dir}")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Brain Segmentation Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['val', 'dicom', 'both'], default='both',
                        help='Evaluation mode: val (validation set), dicom (DICOM files), or both')
    parser.add_argument('--dicom-dir', type=str,
                        default='/home/edhlee/nvme_8TB/Jan2026/AX_T2_DRIVE_501/',
                        help='Directory containing DICOM files')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--scratch-model', action='store_true',
                        help='Use scratch model architecture instead of pretrained')

    args = parser.parse_args()
    main(args)
