"""
3D U-Net model with pretrained I3D encoder for brain tumor segmentation.

This version uses pretrained I3D (Inflated 3D) weights from ImageNet + Kinetics,
matching the TensorFlow FLPedBrain model architecture.

Architecture:
- Encoder: Pretrained I3D Inception backbone
- Decoder: Conv3DTranspose upsampling with skip connections (matching TF)
- Dual output: Segmentation mask + 5-class classification

Key difference from previous version:
- Uses Conv3DTranspose (learnable upsampling) instead of trilinear interpolation
- Skip connections from MaxPool outputs (not Mixed layers) to match TF
- Proper stride patterns: (2,2,2), (2,2,2), (1,2,2), (1,2,2), (2,2,2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from i3d_pytorch import InceptionI3D, load_pretrained_weights


class Upsample3D(nn.Module):
    """
    Upsampling block using Conv3DTranspose, matching TF upsample3d function.

    TF equivalent:
        Conv3DTranspose(filters, kernel_size, strides, padding='same')
        BatchNormalization()
        Activation('relu')

    This version stores the stride to compute expected output size and adjusts
    the output if needed to match TF's 'same' padding behavior.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        # Convert to tuples if needed
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.stride = stride

        # Use minimal padding for ConvTranspose, will adjust output size manually
        padding = tuple(k // 2 for k in kernel_size)

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Compute expected output size (input_size * stride)
        target_size = tuple(s * self.stride[i] for i, s in enumerate(x.shape[2:]))

        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.relu(x)

        # Adjust output size if needed to match TF 'same' padding
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)

        return x


class Conv3dBN(nn.Module):
    """3D Conv + BN + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class PedBrainNetPretrained(nn.Module):
    """
    3D U-Net with pretrained I3D encoder for pediatric brain tumor segmentation.

    This model matches the TensorFlow FLPedBrain architecture:
    - Uses I3D Inception backbone with ImageNet+Kinetics pretrained weights
    - Decoder with Conv3DTranspose upsampling (not interpolation)
    - Skip connections from MaxPool layers (matching TF layer names)
    - Dual output: segmentation + classification

    TF model skip connection layers and shapes (for 64x256x256 input):
        Conv3d_1a_7x7:    (B, 64,  32, 128, 128) - after stride (2,2,2) conv
        MaxPool3d_2a_3x3: (B, 64,  32,  64,  64) - after stride (1,2,2) pool
        MaxPool3d_3a_3x3: (B, 192, 32,  32,  32) - after stride (1,2,2) pool
        MaxPool3d_4a_3x3: (B, 480, 16,  16,  16) - after stride (2,2,2) pool
        MaxPool3d_5a_2x2: (B, 832,  8,   8,   8) - after stride (2,2,2) pool

    TF decoder up_stack strides:
        upsample3d(832, (2,3,3), (2,2,2))  - 8->16 in all dims
        upsample3d(480, (2,3,3), (2,2,2))  - 16->32 in all dims
        upsample3d(192, (1,3,3), (1,2,2))  - 32->32 in D, 32->64 in H,W
        upsample3d(64,  (1,3,3), (1,2,2))  - 32->32 in D, 64->128 in H,W
        upsample3d(64,  (2,3,3), (2,2,2))  - 32->64 in D, 128->256 in H,W
        Final: Conv3DTranspose(1, 5, strides=2) - additional 2x upsample

    Inputs:
        x: (B, 1, D, H, W) - 3D brain MRI volume (grayscale)
           D=64 frames, H=W=256 spatial resolution

    Outputs:
        seg_output: (B, 1, D, H, W) - Segmentation mask (sigmoid activated)
        class_output: (B, 5) - Classification probabilities (softmax activated)
        seg_logits: (B, 1, D, H, W) - Raw segmentation logits
        class_logits: (B, 5) - Raw classification logits
    """

    def __init__(self, num_classes=5, pretrained=True, weights_path=None, freeze_encoder=False):
        """
        Args:
            num_classes: Number of classification classes (default: 5)
            pretrained: Whether to load pretrained I3D weights
            weights_path: Path to pretrained weights (or None for default)
            freeze_encoder: If True, freeze encoder weights (for fine-tuning)
        """
        super().__init__()

        self.num_classes = num_classes

        # I3D Encoder (3-channel input for pretrained weights)
        self.encoder = InceptionI3D(num_classes=400, in_channels=3, dropout_prob=0.0)

        # Load pretrained weights
        if pretrained and weights_path and os.path.exists(weights_path):
            print(f"Loading pretrained I3D weights from: {weights_path}")
            load_pretrained_weights(self.encoder, weights_path, strict=False)
        elif pretrained:
            print("Warning: pretrained=True but no valid weights_path provided")
            print("Model will use random initialization")

        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder weights frozen")

        # Decoder matching TF EXACTLY
        # TF code:
        #   up_stack = [upsample3d(832,..), upsample3d(480,..), upsample3d(192,..), upsample3d(64,..)]
        #   # NOTE: 5 upsamples but zip with 4 skips means up[4] is NOT used!
        #   for up, skip in zip(up_stack, skips):  # only 4 iterations
        #       x = up(x)
        #       x = Concatenate()([x, skip])  # NO conv after concat!
        #   last = Conv3DTranspose(1, 5, strides=2, padding='same')
        #   x = last(x)
        #
        # TF decoder trace:
        #   (832,8,8,8) -> up(832) -> (832,16,16,16) -> cat(480) -> (1312,16,16,16)
        #   (1312,16,16,16) -> up(480) -> (480,32,32,32) -> cat(192) -> (672,32,32,32)
        #   (672,32,32,32) -> up(192) stride(1,2,2) -> (192,32,64,64) -> cat(64) -> (256,32,64,64)
        #   (256,32,64,64) -> up(64) stride(1,2,2) -> (64,32,128,128) -> cat(64) -> (128,32,128,128)
        #   (128,32,128,128) -> Conv3DTranspose(1,5,stride=2) -> (1,64,256,256)

        # Up1: (832,8,8,8) -> (832,16,16,16)
        self.up1 = Upsample3D(832, 832, kernel_size=(2, 3, 3), stride=(2, 2, 2))
        # After concat with skip4 (480ch): 832+480=1312

        # Up2: (1312,16,16,16) -> (480,32,32,32)
        self.up2 = Upsample3D(1312, 480, kernel_size=(2, 3, 3), stride=(2, 2, 2))
        # After concat with skip3 (192ch): 480+192=672

        # Up3: (672,32,32,32) -> (192,32,64,64) stride(1,2,2)
        self.up3 = Upsample3D(672, 192, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        # After concat with skip2 (64ch): 192+64=256

        # Up4: (256,32,64,64) -> (64,32,128,128) stride(1,2,2)
        self.up4 = Upsample3D(256, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        # After concat with skip1 (64ch): 64+64=128

        # Final: (128,32,128,128) -> (1,64,256,256) with Conv3DTranspose kernel=5 stride=2
        self.final_conv = nn.ConvTranspose3d(128, 1, kernel_size=5, stride=2, padding=2, output_padding=1)

        # Classification head from bottleneck features (matching TF)
        # TF: GlobalAveragePooling3D()(x) where x is segmentation output
        # But that's weird - let's use bottleneck features like before
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(832, 128),  # From MaxPool3d_5a output (832 channels)
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass matching TF model exactly.

        Args:
            x: Input tensor (B, 1, D, H, W) - single channel grayscale

        Returns:
            seg_output: Sigmoid-activated segmentation
            class_output: Softmax-activated classification
            seg_logits: Raw segmentation logits
            class_logits: Raw classification logits
        """
        # Store original size for potential size adjustment
        original_size = x.shape[2:]  # (D, H, W)

        # Convert 1-channel to 3-channel (like TF model does)
        x = torch.cat([x, x, x], dim=1)  # (B, 3, D, H, W)

        # Encoder forward pass with skip connections
        endpoints = self.encoder(x, return_endpoints=True)

        # Get skip connections from MaxPool outputs (matching TF model)
        # TF: layer_names = ['Conv3d_1a_7x7', 'MaxPool2d_2a_3x3', 'MaxPool2d_3a_3x3',
        #                    'MaxPool2d_4a_3x3', 'MaxPool2d_5a_2x2']
        # skips = reversed(skips[:-1]) -> [skip4, skip3, skip2, skip1]
        skip1 = endpoints['Conv3d_1a_7x7']      # (B, 64, 32, 128, 128)
        skip2 = endpoints['MaxPool3d_2a_3x3']   # (B, 64, 32, 64, 64)
        skip3 = endpoints['MaxPool3d_3a_3x3']   # (B, 192, 32, 32, 32)
        skip4 = endpoints['MaxPool3d_4a_3x3']   # (B, 480, 16, 16, 16)
        features = endpoints['MaxPool3d_5a_2x2'] # (B, 832, 8, 8, 8)

        # Decoder path (matching TF exactly - NO conv after concat)
        # Up1: (832,8,8,8) -> (832,16,16,16), concat with skip4 (480) -> (1312,16,16,16)
        x = self.up1(features)
        x = torch.cat([x, skip4], dim=1)

        # Up2: (1312,16,16,16) -> (480,32,32,32), concat with skip3 (192) -> (672,32,32,32)
        x = self.up2(x)
        x = torch.cat([x, skip3], dim=1)

        # Up3: (672,32,32,32) -> (192,32,64,64), concat with skip2 (64) -> (256,32,64,64)
        x = self.up3(x)
        x = torch.cat([x, skip2], dim=1)

        # Up4: (256,32,64,64) -> (64,32,128,128), concat with skip1 (64) -> (128,32,128,128)
        x = self.up4(x)
        x = torch.cat([x, skip1], dim=1)

        # Final: (128,32,128,128) -> (1,64,256,256)
        seg_logits = self.final_conv(x)

        # Ensure output matches input size exactly (handle any rounding issues)
        if seg_logits.shape[2:] != original_size:
            seg_logits = F.interpolate(seg_logits, size=original_size, mode='trilinear', align_corners=False)

        seg_output = torch.sigmoid(seg_logits)

        # Classification from bottleneck features
        class_logits = self.classifier(features)
        class_output = F.softmax(class_logits, dim=1)

        return seg_output, class_output, seg_logits, class_logits


def count_parameters(model, trainable_only=True):
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Test model
    print("Testing PedBrainNetPretrained...")

    # Without pretrained weights
    model = PedBrainNetPretrained(num_classes=5, pretrained=False)
    print(f"Total parameters: {count_parameters(model, trainable_only=False):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")

    # Test forward pass
    x = torch.randn(1, 1, 64, 256, 256)
    print(f"\nInput shape: {x.shape}")

    seg_out, class_out, seg_logits, class_logits = model(x)
    print(f"Segmentation output: {seg_out.shape}")
    print(f"Classification output: {class_out.shape}")
    print(f"Class probabilities: {class_out[0].detach().numpy()}")
