from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math


def make_network(architecture: str, *args, **kwargs):
    """
    Builds a network by name. Preserves existing behavior for:
      - FCN / CNN
      - ResNets (with CIFAR stem tweaks via kernel_size / remove_maxpool)

    Adds:
      - VGG from torchvision: vgg11/13/16/19 (+ _bn variants)
        * If remove_maxpool=True, drops the first maxpool (CIFAR-friendly).
      - CIFAR WideResNet: wrn_16_{1,2}, wrn_40_{1,2} (local implementation)
    """
    # ---- existing branches (unchanged) ----
    if architecture == "fcn":
        return FCN(**kwargs)
    elif architecture == "cnn":
        return CNN(**kwargs)
    elif architecture.startswith("resnet"):
        net = getattr(torchvision.models, architecture)(num_classes=kwargs["out_features"])
        if "kernel_size" in kwargs:
            kernel_size = kwargs["kernel_size"]
            stride = kwargs.get("stride", 1)
            padding = kwargs.get("padding", kernel_size // 2)
            in_channels = kwargs.get("in_channels", 3)
            bias = net.conv1.bias is not None
            net.conv1 = nn.Conv2d(in_channels, net.conv1.out_channels, kernel_size, stride, padding, bias=bias)
        if kwargs.get("remove_maxpool", False):
            net.maxpool = nn.Identity()
        return net

    # ---- VGG from torchvision ----
    elif architecture in {
        "vgg11", "vgg13", "vgg16", "vgg19",
        "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"
    }:
        out_features = kwargs["out_features"]
        net = getattr(torchvision.models, architecture)(num_classes=out_features)

        # 1) CIFAR-friendly: drop the first MaxPool to avoid over-downsampling
        if kwargs.get("remove_maxpool", False):
            feats = net.features
            for i, m in enumerate(feats):
                if isinstance(m, nn.MaxPool2d):
                    feats[i] = nn.Identity()
                    break
            net.features = feats

        # 2) Use global average pooling to 1x1 instead of 7x7
        net.avgpool = nn.AdaptiveAvgPool2d(1)

        # 3) Replace the ImageNet classifier with a small CIFAR head (512 -> C)
        # torchvision VGG has classifier: [Linear(25088,4096), ReLU, Dropout, Linear(4096,4096), ReLU, Dropout, Linear(4096,C)]
        # We swap it for a single Linear on pooled 512-d features.
        net.classifier = nn.Linear(512, out_features)

        return net

    elif architecture == "vgg_c8":
        return VGG_C8(num_classes=kwargs.get("out_features", 100), use_bn=True)

    elif architecture == "vgg_c13":
        return VGG_C13(num_classes=kwargs.get("out_features", 100), use_bn=True)

    # ---- NEW: CIFAR WideResNet (local implementation) ----
    elif architecture in _WRN_MAP:
        depth, widen = _WRN_MAP[architecture]
        num_classes = kwargs.get("out_features", 100)
        drop_rate = kwargs.get("wrn_drop_rate", 0.0)
        net = WideResNet(depth=depth, widen_factor=widen, num_classes=num_classes, drop_rate=drop_rate)
        return net

    else:
        raise ValueError("Unrecognized network architecture:", architecture)


# ---------------- existing classes (unchanged) ----------------

class FCN(nn.Sequential):
    """Basic fully connected network class."""

    def __init__(self, sizes: List[int], nonlinearity: Union[str, type] = "ReLU", bn: bool = False, **layer_kwargs):
        super().__init__()
        nonl_class = getattr(nn, nonlinearity) if isinstance(nonlinearity, str) else nonlinearity

        layer_kwargs.setdefault("bias", not bn)
        for i, (s0, s1) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.add_module(f"Linear{i}", nn.Linear(s0, s1, **layer_kwargs))
            if bn:
                self.add_module(f"BN{i}", nn.BatchNorm1d(s1))
            if i < len(sizes) - 2:
                self.add_module(f"Nonlinarity{i}", nonl_class())


class CNN(nn.Sequential):
    """
    Basic CNN class with Conv/BN/Nonl/Maxpool blocks followed by a fully connected net.
    Batchnorm and maxpooling are optional and the latter can also only be included after every nth block.
    """

    def __init__(self, channels: List[int], lin_sizes: List[int], nonlinearity: Union[str, type] = "ReLU",
                 maxpool_freq: int = 1, conv_bn: bool = False, linear_bn: bool = False, kernel_size: int = 3,
                 **conv_kwargs):
        super().__init__()
        nonl_class = getattr(nn, nonlinearity) if isinstance(nonlinearity, str) else nonlinearity
        conv_kwargs.setdefault("bias", not conv_bn)
        for i, (c0, c1) in enumerate(zip(channels[:-1], channels[1:])):
            self.add_module(f"Conv{i}", nn.Conv2d(c0, c1, kernel_size, **conv_kwargs))
            if conv_bn:
                self.add_module(f"ConvBN{i}", nn.BatchNorm2d(c1))
            self.add_module(f"ConvNonlinearity{i}", nonl_class())
            if maxpool_freq and (i + 1) % maxpool_freq == 0:
                self.add_module(f"Maxpool{i//maxpool_freq}", nn.MaxPool2d(2, 2))
        self.add_module("Flatten", nn.Flatten())

        self.add_module("fc", FCN(lin_sizes, nonlinearity=nonlinearity, bn=linear_bn))


# ---------------- helpers for new architectures ----------------

def _vgg_drop_first_maxpool_(model: torchvision.models.VGG) -> None:
    """
    Replace the FIRST MaxPool2d in VGG.features with Identity (CIFAR-friendly).
    No-op if none found (shouldn't happen for standard VGGs).
    """
    feats = model.features
    for i, m in enumerate(feats):
        if isinstance(m, nn.MaxPool2d):
            feats[i] = nn.Identity()
            break
    model.features = feats


# ===== CIFAR WideResNet (Zagoruyko & Komodakis) =====

class WRNBasicBlock(nn.Module):
    """
    A WideResNet basic block with pre-activation and optional dropout.
    Named uniquely to avoid clashing with any existing 'BasicBlock'.
    """
    def __init__(self, in_planes: int, out_planes: int, stride: int, drop_rate: float = 0.0):
        super().__init__()
        self.equal_in_out = (in_planes == out_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.shortcut = None
        if not self.equal_in_out:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        residual = x if self.equal_in_out else out
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        return residual + out


def _wrn_make_group(n_blocks: int, in_planes: int, out_planes: int, stride: int, drop_rate: float) -> (nn.Sequential, int):
    layers = []
    for i in range(n_blocks):
        s = stride if i == 0 else 1
        layers.append(WRNBasicBlock(in_planes, out_planes, s, drop_rate))
        in_planes = out_planes
    return nn.Sequential(*layers), in_planes


class WideResNet(nn.Module):
    """
    CIFAR-style WideResNet with depth = 6n + 4 and widen_factor k.
    Downsampling occurs at the starts of group2 and group3 (stride=2).
    """
    def __init__(self, depth: int = 28, widen_factor: int = 10, num_classes: int = 100, drop_rate: float = 0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth should be 6n + 4."
        n = (depth - 4) // 6
        k = widen_factor
        widths = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.group1, in1 = _wrn_make_group(n, widths[0], widths[1], stride=1, drop_rate=drop_rate)
        self.group2, in2 = _wrn_make_group(n, in1,        widths[2], stride=2, drop_rate=drop_rate)
        self.group3, in3 = _wrn_make_group(n, in2,        widths[3], stride=2, drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(in3)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in3, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.relu(self.bn(x))
        x = self.pool(x).flatten(1)
        x = self.fc(x)  # logits
        return x


# Short names → (depth, widen_factor)
_WRN_MAP = {
    "wrn_16_1": (16, 1),
    "wrn_16_2": (16, 2),
    "wrn_40_1": (40, 1),
    "wrn_40_2": (40, 2),
}

def _vgg_block(in_c: int, out_c: int, n_convs: int = 2, use_bn: bool = True) -> nn.Sequential:
    layers = []
    c = in_c
    for _ in range(n_convs):
        conv = nn.Conv2d(c, out_c, kernel_size=3, padding=1, bias=not use_bn)
        layers.append(conv)
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True))
        c = out_c
    return nn.Sequential(*layers)

def _init_weights_he(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)

# ---------- VGG-C8 (CIFAR) ----------
class VGG_C8(nn.Module):
    """
    CIFAR-style VGG with 8 conv layers (2x per stage), BN by default, no first maxpool,
    global avg pool, and a single Linear head.
    """
    def __init__(self, num_classes: int = 100, use_bn: bool = True):
        super().__init__()
        # Stages (2 convs each): 64, 128, 256, 512
        self.stage1 = _vgg_block(3,   64, n_convs=2, use_bn=use_bn)
        self.pool1  = nn.Identity()              # drop first pool (CIFAR trick)
        self.stage2 = _vgg_block(64,  128, n_convs=2, use_bn=use_bn)
        self.pool2  = nn.MaxPool2d(2)            # 32->16
        self.stage3 = _vgg_block(128, 256, n_convs=2, use_bn=use_bn)
        self.pool3  = nn.MaxPool2d(2)            # 16->8
        self.stage4 = _vgg_block(256, 512, n_convs=2, use_bn=use_bn)
        self.pool4  = nn.AdaptiveAvgPool2d(1)    # 8->1
        self.classifier = nn.Linear(512, num_classes)

        _init_weights_he(self)

    def forward(self, x):
        x = self.stage1(x); x = self.pool1(x)
        x = self.stage2(x); x = self.pool2(x)
        x = self.stage3(x); x = self.pool3(x)
        x = self.stage4(x); x = self.pool4(x)
        x = x.flatten(1)
        return self.classifier(x)

# ---------- VGG-C13 (CIFAR) ----------
class VGG_C13(nn.Module):
    """
    CIFAR-style VGG-13 analogue (conv part like torchvision's VGG13: 2,2,2,2,2 convs across stages),
    BN by default, no first maxpool, global avg pool, single Linear head.
    Total conv layers = 10 (VGG13 has 13 "weight layers" incl. FC in the original).
    """
    def __init__(self, num_classes: int = 100, use_bn: bool = True):
        super().__init__()
        # Stages with conv counts: (64x2), (128x2), (256x2), (512x2), (512x2)
        self.stage1 = _vgg_block(3,    64, n_convs=2, use_bn=use_bn)
        self.pool1  = nn.Identity()              # drop first pool
        self.stage2 = _vgg_block(64,   128, n_convs=2, use_bn=use_bn)
        self.pool2  = nn.MaxPool2d(2)            # 32->16
        self.stage3 = _vgg_block(128,  256, n_convs=2, use_bn=use_bn)
        self.pool3  = nn.MaxPool2d(2)            # 16->8
        self.stage4 = _vgg_block(256,  512, n_convs=2, use_bn=use_bn)
        self.pool4  = nn.MaxPool2d(2)            # 8->4
        self.stage5 = _vgg_block(512,  512, n_convs=2, use_bn=use_bn)
        self.pool5  = nn.AdaptiveAvgPool2d(1)    # 4->1
        self.classifier = nn.Linear(512, num_classes)

        _init_weights_he(self)

    def forward(self, x):
        x = self.stage1(x); x = self.pool1(x)
        x = self.stage2(x); x = self.pool2(x)
        x = self.stage3(x); x = self.pool3(x)
        x = self.stage4(x); x = self.pool4(x)
        x = self.stage5(x); x = self.pool5(x)
        x = x.flatten(1)
        return self.classifier(x)


