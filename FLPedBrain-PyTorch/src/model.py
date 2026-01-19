"""
3D U-Net model for brain tumor segmentation.

Architecture based on the TensorFlow I3D-UNet from FLPedBrain:
- Encoder: 3D convolutional backbone (similar to I3D but simplified for PyTorch)
- Decoder: Transposed convolutions with skip connections
- Dual output: Segmentation mask + 5-class classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3dBN(nn.Module):
    """3D Convolution + BatchNorm + ReLU block."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super().__init__()
        if padding == 'same':
            # Calculate padding for 'same' output size
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size, kernel_size)
            if isinstance(stride, int):
                stride = (stride, stride, stride)
            padding = tuple(k // 2 for k in kernel_size)

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionModule3D(nn.Module):
    """Simplified 3D Inception module."""

    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super().__init__()

        # 1x1 branch
        self.branch1 = Conv3dBN(in_channels, out_1x1, kernel_size=1)

        # 1x1 -> 3x3 branch
        self.branch2 = nn.Sequential(
            Conv3dBN(in_channels, reduce_3x3, kernel_size=1),
            Conv3dBN(reduce_3x3, out_3x3, kernel_size=3)
        )

        # 1x1 -> 3x3 (simulating 5x5) branch
        self.branch3 = nn.Sequential(
            Conv3dBN(in_channels, reduce_5x5, kernel_size=1),
            Conv3dBN(reduce_5x5, out_5x5, kernel_size=3)
        )

        # Max pool -> 1x1 branch
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            Conv3dBN(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class Encoder3D(nn.Module):
    """3D Encoder based on I3D-like architecture."""

    def __init__(self, in_channels=1):
        super().__init__()

        # Initial conv: (64, 256, 256) -> (32, 128, 128)
        self.conv1 = Conv3dBN(in_channels, 64, kernel_size=7, stride=2, padding=(3, 3, 3))

        # MaxPool: (32, 128, 128) -> (32, 64, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Conv block 2
        self.conv2a = Conv3dBN(64, 64, kernel_size=1)
        self.conv2b = Conv3dBN(64, 192, kernel_size=3)

        # MaxPool: (32, 64, 64) -> (32, 32, 32)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Inception 3b: 192 -> 256 (64+128+32+32)
        self.inception3b = InceptionModule3D(192, 64, 96, 128, 16, 32, 32)
        # Inception 3c: 256 -> 480 (128+192+96+64)
        self.inception3c = InceptionModule3D(256, 128, 128, 192, 32, 96, 64)

        # MaxPool: (32, 32, 32) -> (16, 16, 16)
        self.pool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Inception 4b: 480 -> 512 (192+208+48+64)
        self.inception4b = InceptionModule3D(480, 192, 96, 208, 16, 48, 64)
        # Inception 4c: 512 -> 512 (160+224+64+64)
        self.inception4c = InceptionModule3D(512, 160, 112, 224, 24, 64, 64)
        # Inception 4d: 512 -> 512 (128+256+64+64)
        self.inception4d = InceptionModule3D(512, 128, 128, 256, 24, 64, 64)
        # Inception 4e: 512 -> 528 (112+288+64+64)
        self.inception4e = InceptionModule3D(512, 112, 144, 288, 32, 64, 64)
        # Inception 4f: 528 -> 832 (256+320+128+128)
        self.inception4f = InceptionModule3D(528, 256, 160, 320, 32, 128, 128)

        # MaxPool: (16, 16, 16) -> (8, 8, 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Inception 5b: 832 -> 832 (256+320+128+128)
        self.inception5b = InceptionModule3D(832, 256, 160, 320, 32, 128, 128)
        # Inception 5c: 832 -> 1024 (384+384+128+128)
        self.inception5c = InceptionModule3D(832, 384, 192, 384, 48, 128, 128)

    def forward(self, x):
        # Returns skip connections for decoder
        skips = []

        # Stage 1: (B, 1, 64, 256, 256) -> (B, 64, 32, 128, 128)
        x = self.conv1(x)
        skips.append(x)

        # Stage 2: -> (B, 192, 32, 64, 64)
        x = self.pool1(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        skips.append(x)

        # Stage 3: -> (B, 480, 32, 32, 32)
        x = self.pool2(x)
        x = self.inception3b(x)
        x = self.inception3c(x)
        skips.append(x)

        # Stage 4: -> (B, 832, 16, 16, 16)
        x = self.pool3(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.inception4f(x)
        skips.append(x)

        # Stage 5: -> (B, 1024, 8, 8, 8)
        x = self.pool4(x)
        x = self.inception5b(x)
        x = self.inception5c(x)

        return x, skips


class Upsample3D(nn.Module):
    """Upsampling block with transposed convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        # Calculate padding for proper upsampling
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        padding = tuple(k // 2 for k in kernel_size)
        output_padding = tuple(s - 1 for s in stride)

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Decoder3D(nn.Module):
    """3D Decoder with skip connections."""

    def __init__(self):
        super().__init__()

        # Upsampling blocks (matching TF architecture)
        # 1024 -> 832, (8,8,8) -> (16,16,16)
        self.up1 = Upsample3D(1024, 832, kernel_size=(2, 3, 3), stride=(2, 2, 2))
        self.conv1 = Conv3dBN(832 + 832, 832, kernel_size=3)

        # 832 -> 480, (16,16,16) -> (32,32,32)
        self.up2 = Upsample3D(832, 480, kernel_size=(2, 3, 3), stride=(2, 2, 2))
        self.conv2 = Conv3dBN(480 + 480, 480, kernel_size=3)

        # 480 -> 192, (32,32,32) -> (32,64,64)
        self.up3 = Upsample3D(480, 192, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.conv3 = Conv3dBN(192 + 192, 192, kernel_size=3)

        # 192 -> 64, (32,64,64) -> (32,128,128)
        self.up4 = Upsample3D(192, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.conv4 = Conv3dBN(64 + 64, 64, kernel_size=3)

        # 64 -> 64, (32,128,128) -> (64,256,256)
        self.up5 = Upsample3D(64, 64, kernel_size=(2, 3, 3), stride=(2, 2, 2))

        # Final conv to segmentation output
        self.final_conv = nn.ConvTranspose3d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x, skips):
        # skips: [skip1, skip2, skip3, skip4] from encoder
        # Reverse skips for decoder
        skips = skips[::-1]

        # Up1: concat with skip4
        x = self.up1(x)
        x = torch.cat([x, skips[0]], dim=1)
        x = self.conv1(x)

        # Up2: concat with skip3
        x = self.up2(x)
        x = torch.cat([x, skips[1]], dim=1)
        x = self.conv2(x)

        # Up3: concat with skip2
        x = self.up3(x)
        x = torch.cat([x, skips[2]], dim=1)
        x = self.conv3(x)

        # Up4: concat with skip1
        x = self.up4(x)
        x = torch.cat([x, skips[3]], dim=1)
        x = self.conv4(x)

        # Up5: final upsampling
        x = self.up5(x)

        # Final conv
        x = self.final_conv(x)

        return x


class PedBrainNet(nn.Module):
    """
    3D U-Net for pediatric brain tumor segmentation with dual output.

    Inputs:
        x: (B, 1, D, H, W) - 3D brain MRI volume
           D=64 frames, H=W=256 spatial resolution

    Outputs:
        seg_output: (B, 1, D, H, W) - Segmentation mask (sigmoid activated)
        class_output: (B, 5) - Classification logits (softmax activated)
    """

    def __init__(self, num_classes=5):
        super().__init__()

        self.encoder = Encoder3D(in_channels=1)
        self.decoder = Decoder3D()

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Encoder
        features, skips = self.encoder(x)

        # Decoder for segmentation
        seg_logits = self.decoder(features, skips)
        seg_output = torch.sigmoid(seg_logits)

        # Classification from bottleneck features
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        class_logits = self.classifier(pooled)
        class_output = F.softmax(class_logits, dim=1)

        return seg_output, class_output, seg_logits, class_logits


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = PedBrainNet(num_classes=5)
    print(f"Total trainable parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(1, 1, 64, 256, 256)
    seg_out, class_out, _, _ = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Segmentation output shape: {seg_out.shape}")
    print(f"Classification output shape: {class_out.shape}")
