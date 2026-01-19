"""
Loss functions and metrics for brain tumor segmentation.

Includes:
- Dice coefficient and Dice loss (for segmentation)
- Combined segmentation + classification loss
- F1 score for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coefficient(y_pred, y_true, smooth=1.0):
    """
    Compute Dice coefficient (Sørensen–Dice index).

    DICE = 2 * |X ∩ Y| / (|X| + |Y|)

    Args:
        y_pred: Predicted segmentation mask (B, 1, D, H, W), values in [0, 1]
        y_true: Ground truth mask (B, 1, D, H, W), values in [0, 1]
        smooth: Smoothing factor to prevent division by zero

    Returns:
        Mean Dice coefficient across batch
    """
    # Flatten spatial dimensions
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    y_true_flat = y_true.view(y_true.size(0), -1)

    # Compute intersection and union
    intersection = (y_pred_flat * y_true_flat).sum(dim=1)
    union = y_pred_flat.sum(dim=1) + y_true_flat.sum(dim=1)

    # Dice coefficient per sample
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.mean()


def dice_loss(y_pred, y_true, smooth=1.0):
    """
    Compute Dice loss as -log(dice_coefficient).

    This formulation is used in the original TensorFlow code.

    Args:
        y_pred: Predicted segmentation mask (B, 1, D, H, W)
        y_true: Ground truth mask (B, 1, D, H, W)
        smooth: Smoothing factor

    Returns:
        Dice loss value
    """
    dice = dice_coefficient(y_pred, y_true, smooth)
    return -torch.log(dice + 1e-7)


def dice_loss_v2(y_pred, y_true, smooth=0.1):
    """
    Alternative Dice loss formulation (from dice_coef_loss2 in TF code).

    Args:
        y_pred: Predicted segmentation mask (B, 1, D, H, W)
        y_true: Ground truth mask (B, 1, D, H, W)
        smooth: Smoothing factor

    Returns:
        Scaled Dice loss value
    """
    # Flatten spatial dimensions
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    y_true_flat = y_true.view(y_true.size(0), -1)

    intersection = (y_pred_flat * y_true_flat).sum(dim=1)
    p = y_pred_flat.sum(dim=1)
    t = y_true_flat.sum(dim=1)

    numerator = (intersection + smooth).sum()
    denominator = (t + p + smooth).mean()

    loss = -torch.log(2.0 * numerator + 1e-7) + torch.log(denominator + 1e-7)
    return loss / 20.0


class CombinedLoss(nn.Module):
    """
    Combined loss for segmentation and classification.

    Total Loss = dice_weight * dice_loss + classification_loss

    Args:
        dice_weight: Weight for segmentation loss (default 0.5)
        smooth: Smoothing factor for Dice loss
    """

    def __init__(self, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, seg_pred, seg_true, class_logits, class_true):
        """
        Compute combined loss.

        Args:
            seg_pred: Predicted segmentation (B, 1, D, H, W), sigmoid activated
            seg_true: Ground truth segmentation (B, 1, D, H, W)
            class_logits: Classification logits (B, num_classes), NOT softmax activated
            class_true: Ground truth class labels (B,)

        Returns:
            total_loss, seg_loss, class_loss
        """
        # Segmentation loss (Dice)
        seg_loss = dice_loss(seg_pred, seg_true, self.smooth)

        # Classification loss (Cross-Entropy)
        class_loss = self.ce_loss(class_logits, class_true)

        # Combined loss
        total_loss = self.dice_weight * seg_loss + class_loss

        return total_loss, seg_loss, class_loss


class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross-Entropy loss for segmentation.

    Useful when training with both losses for better gradient flow.
    """

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Predicted segmentation (B, 1, D, H, W), sigmoid activated
            y_true: Ground truth segmentation (B, 1, D, H, W)

        Returns:
            Combined loss value
        """
        dice = dice_loss(y_pred, y_true, self.smooth)
        bce = self.bce(y_pred, y_true)
        return self.dice_weight * dice + self.bce_weight * bce


def compute_metrics(seg_pred, seg_true, class_pred, class_true, threshold=0.5):
    """
    Compute evaluation metrics.

    Args:
        seg_pred: Predicted segmentation (B, 1, D, H, W)
        seg_true: Ground truth segmentation (B, 1, D, H, W)
        class_pred: Predicted class probabilities (B, num_classes)
        class_true: Ground truth class labels (B,)
        threshold: Threshold for binarizing segmentation predictions

    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # Binarize predictions
        seg_pred_binary = (seg_pred > threshold).float()

        # Dice coefficient
        dice = dice_coefficient(seg_pred_binary, seg_true).item()

        # Classification accuracy
        pred_labels = class_pred.argmax(dim=1)
        accuracy = (pred_labels == class_true).float().mean().item()

        # Per-class accuracy
        num_classes = class_pred.size(1)
        per_class_acc = []
        for c in range(num_classes):
            mask = class_true == c
            if mask.sum() > 0:
                class_acc = (pred_labels[mask] == c).float().mean().item()
            else:
                class_acc = float('nan')
            per_class_acc.append(class_acc)

    return {
        'dice': dice,
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc
    }


class F1Score:
    """
    Compute F1 score for multi-class classification.

    Accumulates predictions and computes F1 at the end.
    """

    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)

    def update(self, y_pred, y_true):
        """
        Update confusion matrix.

        Args:
            y_pred: Predicted class probabilities (B, num_classes)
            y_true: Ground truth labels (B,)
        """
        pred_labels = y_pred.argmax(dim=1).cpu()
        true_labels = y_true.cpu()

        for p, t in zip(pred_labels, true_labels):
            self.confusion_matrix[t, p] += 1

    def compute(self):
        """
        Compute per-class and macro F1 scores.

        Returns:
            per_class_f1: List of F1 scores per class
            macro_f1: Macro-averaged F1 score
        """
        per_class_f1 = []

        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = self.confusion_matrix[:, c].sum() - tp
            fn = self.confusion_matrix[c, :].sum() - tp

            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            per_class_f1.append(f1.item())

        macro_f1 = sum(per_class_f1) / self.num_classes

        return per_class_f1, macro_f1


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")

    # Create dummy data
    batch_size = 2
    seg_pred = torch.rand(batch_size, 1, 64, 256, 256)
    seg_true = torch.randint(0, 2, (batch_size, 1, 64, 256, 256)).float()
    class_logits = torch.randn(batch_size, 5)
    class_true = torch.randint(0, 5, (batch_size,))

    # Test Dice coefficient
    dice = dice_coefficient(seg_pred, seg_true)
    print(f"Dice coefficient: {dice.item():.4f}")

    # Test Dice loss
    d_loss = dice_loss(seg_pred, seg_true)
    print(f"Dice loss: {d_loss.item():.4f}")

    # Test combined loss
    criterion = CombinedLoss(dice_weight=0.5)
    total, seg, cls = criterion(seg_pred, seg_true, class_logits, class_true)
    print(f"Combined loss: {total.item():.4f} (seg: {seg.item():.4f}, cls: {cls.item():.4f})")

    # Test metrics
    class_pred = F.softmax(class_logits, dim=1)
    metrics = compute_metrics(seg_pred, seg_true, class_pred, class_true)
    print(f"Metrics: {metrics}")

    # Test F1 score
    f1_metric = F1Score(num_classes=5)
    f1_metric.update(class_pred, class_true)
    per_class_f1, macro_f1 = f1_metric.compute()
    print(f"Per-class F1: {per_class_f1}")
    print(f"Macro F1: {macro_f1:.4f}")
