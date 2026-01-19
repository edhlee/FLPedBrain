"""
Inflated 3D ConvNet (I3D) implementation in PyTorch.

Based on the paper:
"Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
Joao Carreira, Andrew Zisserman
https://arxiv.org/abs/1705.07750

This implementation mirrors the TensorFlow Keras version used in FLPedBrain,
with support for pretrained weights from ImageNet + Kinetics.

Pretrained weights source: https://github.com/hassony2/kinetics_i3d_pytorch
(converted from DeepMind's TensorFlow implementation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Unit3D(nn.Module):
    """Basic unit: Conv3D + BatchNorm + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1, 1),
                 stride=(1, 1, 1), padding='same', activation=True, use_bn=True, use_bias=False):
        super().__init__()

        self.activation = activation
        self.use_bn = use_bn

        # Calculate padding for 'same'
        if padding == 'same':
            pad = tuple(k // 2 for k in kernel_size)
        else:
            pad = padding

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=pad, bias=use_bias
        )

        if use_bn:
            self.bn = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01)

        if activation:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class InceptionModule(nn.Module):
    """I3D Inception module with 4 branches."""

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: Input channels
            out_channels: List of [b0, b1a, b1b, b2a, b2b, b3] output channels
        """
        super().__init__()

        b0, b1a, b1b, b2a, b2b, b3 = out_channels

        # Branch 0: 1x1x1
        self.branch0 = Unit3D(in_channels, b0, kernel_size=(1, 1, 1))

        # Branch 1: 1x1x1 -> 3x3x3
        self.branch1 = nn.Sequential(
            Unit3D(in_channels, b1a, kernel_size=(1, 1, 1)),
            Unit3D(b1a, b1b, kernel_size=(3, 3, 3))
        )

        # Branch 2: 1x1x1 -> 3x3x3
        self.branch2 = nn.Sequential(
            Unit3D(in_channels, b2a, kernel_size=(1, 1, 1)),
            Unit3D(b2a, b2b, kernel_size=(3, 3, 3))
        )

        # Branch 3: MaxPool -> 1x1x1
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            Unit3D(in_channels, b3, kernel_size=(1, 1, 1))
        )

    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3D(nn.Module):
    """
    Inception-v1 Inflated 3D ConvNet.

    Matches the TensorFlow implementation layer-by-layer for weight compatibility.
    """

    # Endpoint names matching TensorFlow model
    ENDPOINTS = [
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
    ]

    def __init__(self, num_classes=400, in_channels=3, dropout_prob=0.0):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Layer definitions matching TensorFlow exactly
        self.layers_dict = nn.ModuleDict()

        # Conv3d_1a_7x7: stride (2,2,2)
        self.layers_dict['Conv3d_1a_7x7'] = Unit3D(
            in_channels, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2)
        )

        # MaxPool3d_2a_3x3: stride (1,2,2)
        self.layers_dict['MaxPool3d_2a_3x3'] = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        # Conv3d_2b_1x1
        self.layers_dict['Conv3d_2b_1x1'] = Unit3D(64, 64, kernel_size=(1, 1, 1))

        # Conv3d_2c_3x3
        self.layers_dict['Conv3d_2c_3x3'] = Unit3D(64, 192, kernel_size=(3, 3, 3))

        # MaxPool3d_3a_3x3: stride (1,2,2)
        self.layers_dict['MaxPool3d_3a_3x3'] = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        # Mixed_3b: 192 -> 256
        self.layers_dict['Mixed_3b'] = InceptionModule(192, [64, 96, 128, 16, 32, 32])

        # Mixed_3c: 256 -> 480
        self.layers_dict['Mixed_3c'] = InceptionModule(256, [128, 128, 192, 32, 96, 64])

        # MaxPool3d_4a_3x3: stride (2,2,2)
        self.layers_dict['MaxPool3d_4a_3x3'] = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1
        )

        # Mixed_4b: 480 -> 512
        self.layers_dict['Mixed_4b'] = InceptionModule(480, [192, 96, 208, 16, 48, 64])

        # Mixed_4c: 512 -> 512
        self.layers_dict['Mixed_4c'] = InceptionModule(512, [160, 112, 224, 24, 64, 64])

        # Mixed_4d: 512 -> 512
        self.layers_dict['Mixed_4d'] = InceptionModule(512, [128, 128, 256, 24, 64, 64])

        # Mixed_4e: 512 -> 528
        self.layers_dict['Mixed_4e'] = InceptionModule(512, [112, 144, 288, 32, 64, 64])

        # Mixed_4f: 528 -> 832
        self.layers_dict['Mixed_4f'] = InceptionModule(528, [256, 160, 320, 32, 128, 128])

        # MaxPool3d_5a_2x2: stride (2,2,2)
        self.layers_dict['MaxPool3d_5a_2x2'] = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0
        )

        # Mixed_5b: 832 -> 832
        self.layers_dict['Mixed_5b'] = InceptionModule(832, [256, 160, 320, 32, 128, 128])

        # Mixed_5c: 832 -> 1024
        self.layers_dict['Mixed_5c'] = InceptionModule(832, [384, 192, 384, 48, 128, 128])

        # Logits (optional, for standalone classification)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.logits = nn.Conv3d(1024, num_classes, kernel_size=(1, 1, 1), bias=True)

    def forward(self, x, return_endpoints=False):
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, T, H, W)
            return_endpoints: If True, return dict of intermediate features

        Returns:
            If return_endpoints: dict of {endpoint_name: features}
            Else: final logits
        """
        endpoints = {}

        for name in self.ENDPOINTS:
            x = self.layers_dict[name](x)
            endpoints[name] = x

        if return_endpoints:
            return endpoints

        # Classification head
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.logits(x)
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)

        return x

    def extract_features(self, x):
        """Extract features before classification head."""
        for name in self.ENDPOINTS:
            x = self.layers_dict[name](x)
        return x


def convert_pretrained_key(key):
    """
    Convert pretrained weight key (piergiaj/pytorch-i3d format) to our model format.

    Mapping:
        Conv3d_1a_7x7.conv3d.weight -> layers_dict.Conv3d_1a_7x7.conv.weight
        Mixed_3b.b0.conv3d.weight -> layers_dict.Mixed_3b.branch0.conv.weight
        Mixed_3b.b1a.conv3d.weight -> layers_dict.Mixed_3b.branch1.0.conv.weight
        Mixed_3b.b1b.conv3d.weight -> layers_dict.Mixed_3b.branch1.1.conv.weight
        etc.
    """
    # Replace conv3d with conv
    new_key = key.replace('.conv3d.', '.conv.')

    # Handle logits layer naming
    new_key = new_key.replace('logits.conv.', 'logits.')

    # Handle inception module branches
    # b0 -> branch0, b1a -> branch1.0, b1b -> branch1.1, b2a -> branch2.0, b2b -> branch2.1, b3b -> branch3.1
    # Note: piergiaj weights use b3b for the conv after maxpool (not b3)
    branch_map = {
        '.b0.': '.branch0.',
        '.b1a.': '.branch1.0.',
        '.b1b.': '.branch1.1.',
        '.b2a.': '.branch2.0.',
        '.b2b.': '.branch2.1.',
        '.b3b.': '.branch3.1.',  # 1x1 conv after MaxPool3d
    }
    for old, new in branch_map.items():
        new_key = new_key.replace(old, new)

    # Add layers_dict prefix for layer names
    layer_prefixes = ['Conv3d_1a_7x7', 'Conv3d_2b_1x1', 'Conv3d_2c_3x3',
                      'Mixed_3b', 'Mixed_3c', 'Mixed_4b', 'Mixed_4c',
                      'Mixed_4d', 'Mixed_4e', 'Mixed_4f', 'Mixed_5b', 'Mixed_5c']
    for prefix in layer_prefixes:
        if new_key.startswith(prefix):
            new_key = 'layers_dict.' + new_key
            break

    return new_key


def load_pretrained_weights(model, weights_path, strict=False):
    """
    Load pretrained I3D weights.

    Args:
        model: InceptionI3D model
        weights_path: Path to .pt or .pth file
        strict: If True, require exact key match
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)

    # Handle different weight formats
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # Convert keys to match our model architecture
    converted_state_dict = {}
    for key, value in state_dict.items():
        new_key = convert_pretrained_key(key)
        converted_state_dict[new_key] = value

    # Try to load weights
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=strict)

    loaded_count = len(state_dict) - len(unexpected)
    print(f"Loaded {loaded_count}/{len(state_dict)} pretrained weights")

    if missing and len(missing) < 20:
        print(f"Missing keys ({len(missing)}):")
        for k in missing:
            print(f"  {k}")
    elif missing:
        print(f"Missing keys: {len(missing)} (decoder weights, expected)")

    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"  {k}")

    return model


def convert_tf_to_pytorch_key(tf_key):
    """Convert TensorFlow weight key to PyTorch format."""
    # This function handles the key mapping between TF and PyTorch models
    # The hassony2 repo provides already-converted weights, so this may not be needed
    return tf_key


# Convenience functions for getting pretrained models
def i3d_rgb(pretrained=False, weights_path=None, **kwargs):
    """
    Create I3D model for RGB input.

    Args:
        pretrained: If True, load pretrained weights (requires weights_path)
        weights_path: Path to pretrained weights file
        **kwargs: Additional arguments for InceptionI3D
    """
    model = InceptionI3D(in_channels=3, **kwargs)

    if pretrained:
        if weights_path is None:
            # Default path
            weights_path = os.path.join(
                os.path.dirname(__file__),
                'pretrained_weights',
                'rgb_imagenet.pt'
            )
        if os.path.exists(weights_path):
            load_pretrained_weights(model, weights_path, strict=False)
            print(f"Loaded pretrained weights from: {weights_path}")
        else:
            print(f"Warning: Pretrained weights not found at {weights_path}")
            print("Download from: https://github.com/piergiaj/pytorch-i3d")

    return model


if __name__ == "__main__":
    # Test model
    model = InceptionI3D(num_classes=400, in_channels=3)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    x = torch.randn(1, 3, 64, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Test with endpoints
    endpoints = model(x, return_endpoints=True)
    print("\nEndpoint shapes:")
    for name, feat in endpoints.items():
        print(f"  {name}: {feat.shape}")
