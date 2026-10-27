"""
RepLK-BEV Backbone

Large-kernel 2D convolutional backbone for BEV feature maps.
Adapted from LargeKernel3D's CenterPoint backbone for dense 2D BEV grids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


# Large Kernel DW Conv
class LargeKernelDWConv(nn.Module):
    """
    Depthwise conv with a large kernel (e.g. 21×21) and a parallel small
    kernel (5×5) for reparameterization. At inference, both are fused
    into a single conv for zero overhead.
    """

    def __init__(self, channels, kernel_size, small_kernel=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.is_fused = False

        # Large depthwise conv + BN
        self.large_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.large_bn = nn.BatchNorm2d(channels)

        # Small depthwise conv + BN (reparameterizable branch)
        self.small_conv = nn.Conv2d(
            channels,
            channels,
            small_kernel,
            padding=small_kernel // 2,
            groups=channels,
            bias=False,
        )
        self.small_bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        if self.is_fused:
            return self.fused_conv(x)
        return self.large_bn(self.large_conv(x)) + self.small_bn(self.small_conv(x))

    @torch.no_grad()
    def fuse(self):
        """Merge both branches into one conv for inference."""
        if self.is_fused:
            return

        # Fold BN into weights for both branches
        large_w, large_b = self._fold_bn(self.large_conv, self.large_bn)
        small_w, small_b = self._fold_bn(self.small_conv, self.small_bn)

        # Pad small kernel to match large kernel size, then add
        pad = (self.kernel_size - small_w.shape[-1]) // 2
        small_w_padded = F.pad(small_w, [pad, pad, pad, pad])
        fused_w = large_w + small_w_padded
        fused_b = large_b + small_b

        # Create single fused conv
        self.fused_conv = nn.Conv2d(
            fused_w.shape[0],
            fused_w.shape[0],
            self.kernel_size,
            padding=self.kernel_size // 2,
            groups=fused_w.shape[0],
            bias=True,
        ).to(fused_w.device)
        self.fused_conv.weight.data.copy_(fused_w)
        self.fused_conv.bias.data.copy_(fused_b)

        # Clean up old layers
        del self.large_conv, self.large_bn, self.small_conv, self.small_bn
        self.is_fused = True

    @staticmethod
    def _fold_bn(conv, bn):
        """Fold BatchNorm into conv weights -> (weight, bias)."""
        scale = bn.weight / (bn.running_var + bn.eps).sqrt()
        weight = conv.weight * scale.view(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * scale
        return weight, bias


# RepLK Block
class RepLKBlock(nn.Module):
    """
    Single block: 1×1 expand -> DW large kernel -> 1×1 project -> residual add.
    """

    def __init__(self, in_ch, out_ch, kernel_size=13, expand=2, small_kernel=5):
        super().__init__()
        mid_ch = out_ch * expand

        # Main path
        self.expand = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(mid_ch)

        self.dw_conv = LargeKernelDWConv(mid_ch, kernel_size, small_kernel)

        self.project = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_ch)

        self.act = nn.GELU()

        # Shortcut (identity if channels match, 1×1 conv otherwise)
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.act(self.expand_bn(self.expand(x)))  # 1×1 expand
        out = self.act(self.dw_conv(out))  # DW large kernel
        out = self.project_bn(self.project(out))  # 1×1 project

        return self.act(out + identity)  # residual + activation


# Downsample
class Downsample(nn.Module):
    """Stride-2 spatial downsample: 3×3 conv -> BN -> GELU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# Stage
class Stage(nn.Module):
    """
    One stage = optional downsample + N RepLK blocks.
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        num_blocks,
        kernel_size,
        downsample=False,
        small_kernel=5,
        expand=2,
    ):
        super().__init__()
        layers = []

        if downsample:
            layers.append(Downsample(in_ch, out_ch))
            in_ch = out_ch

        for i in range(num_blocks):
            layers.append(
                RepLKBlock(
                    in_ch=in_ch if i == 0 else out_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    expand=expand,
                    small_kernel=small_kernel,
                )
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# RepLK-BEV Backbone
@MODELS.register_module()
class RepLKBEVBackbone(BaseModule):
    """
    RepLK-BEV: Large-Kernel 2D Backbone for BEV Feature Maps.
    """

    def __init__(
        self,
        in_channels,
        stem_channels=16,
        stage_channels=None,
        num_blocks=None,
        kernel_sizes=None,
        small_kernels=None,
        expand=2,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        # Defaults matching CenterPoint + LargeKernel3D
        if stage_channels is None:
            stage_channels = [16, 32, 64, 128]
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        if kernel_sizes is None:
            kernel_sizes = [21, 21, 13, 3]
        if small_kernels is None:
            small_kernels = [5, 5, 5, 3]

        self.num_stages = len(stage_channels)
        self.out_channels = stage_channels[-1]

        # Stem (c0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
        )

        # Stages (c1 -> c4)
        self.stages = nn.ModuleList()
        prev_ch = stem_channels
        for i in range(self.num_stages):
            self.stages.append(
                Stage(
                    in_ch=prev_ch,
                    out_ch=stage_channels[i],
                    num_blocks=num_blocks[i],
                    kernel_size=kernel_sizes[i],
                    downsample=(i > 0),  # stages 2, 3, 4 downsample
                    small_kernel=small_kernels[i],
                    expand=expand,
                )
            )
            prev_ch = stage_channels[i]

        # Weight initialization
        self.apply(self._init_weights)

    def forward(self, x):
        """(B, C, H, W) -> (B, C_out, H/8, W/8)"""
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            # Gradient checkpointing on high-resolution stages (512×512 and 256×256)

            if self.training and i < 2:
                x = grad_checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        return x

    def forward_multiscale(self, x):
        """Returns dict of per-stage feature maps (for FPN-style necks)."""
        features = {}
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features[f"stage{i+1}"] = x
        return features

    @torch.no_grad()
    def fuse_all(self):
        """Merge all reparam branches for inference deployment."""
        for stage in self.stages:
            for layer in stage.layers:
                if isinstance(layer, RepLKBlock):
                    layer.dw_conv.fuse()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# Multi-Scale Output Variant
@MODELS.register_module()
class RepLKBEVBackboneMultiScale(RepLKBEVBackbone):
    """RepLK backbone returning per-stage features as a list."""

    def forward(self, x):
        """(B, C, H, W) -> list[(B, Ci, Hi, Wi)] per stage."""
        outs = []
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            if self.training and i < 2:
                x = grad_checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
            outs.append(x)
        return outs
