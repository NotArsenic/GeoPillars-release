"""
BEV Feature Upsampler
=====================
Decoder-style upsampling module using ConvTranspose2d blocks.
Restores the spatial resolution from (H/8, W/8) back to (H, W)
while keeping the channel dimension unchanged.

Input:  (B, C, H/8, W/8)  - output from RepLK-BEV backbone
Output: (B, C, H,   W)    - full-resolution BEV feature map

Three upsample stages (each ×2):
    Stage 1: (B, C, H/8, W/8) -> (B, C, H/4, W/4)
    Stage 2: (B, C, H/4, W/4) -> (B, C, H/2, W/2)
    Stage 3: (B, C, H/2, W/2) -> (B, C, H,   W)
"""

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


class UpsampleBlock(nn.Module):
    """
    Single ×2 upsample block: ConvTranspose2d -> BatchNorm -> GELU -> Conv 3×3 -> BatchNorm -> GELU.

    The transposed conv doubles the spatial resolution.
    The follow-up 3×3 conv refines the upsampled features and reduces
    checkerboard artifacts that ConvTranspose2d can introduce.

    Args:
        in_channels:  input channel count
        out_channels: output channel count
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            # ×2 spatial upsample
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            # Refine / smooth
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


@MODELS.register_module()
class BEVUpsampler(BaseModule):
    """
    Three-stage ×2 upsampler: (B, C_in, H/8, W/8) -> (B, C_out, H, W).

    When ``out_channels`` < ``channels``, the first upsample stage reduces
    channels, so the two highest-resolution stages (H/2, H) run at the
    narrower width - significantly reducing activation memory.

    Args:
        channels:     input channel count (from backbone/NGFM)
        out_channels: output channel count fed to the detection head.
                      Defaults to ``channels`` (no reduction).
    """

    def __init__(self, channels=128, out_channels=None, num_stages=3, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        if out_channels is None:
            out_channels = channels
        self.out_channels = out_channels
        self.num_stages = num_stages

        # Build only the requested number of upsample stages.
        # Stage 1 reduces channels (channels -> out_channels); remaining stages
        # keep out_channels. Unused stage slots are simply absent.
        up_layers = [UpsampleBlock(channels, out_channels)]   # stage 1: reduce ch
        for _ in range(num_stages - 1):
            up_layers.append(UpsampleBlock(out_channels, out_channels))
        self.up_layers = nn.ModuleList(up_layers)

        self.apply(self._init_weights)

    def forward(self, x):
        """
        Args:
            x: (B, C_in, H_in, W_in) from RepLK-BEV backbone / NGFM

        Returns:
            (B, C_out, H_in * 2^num_stages, W_in * 2^num_stages) upsampled BEV features
        """
        for up in self.up_layers:
            x = up(x)
        return x

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

@MODELS.register_module()
class BEVUpsamplerToList(BEVUpsampler):
    """Adapter to wrap the upsampled tensor in a list for CenterHead."""
    def forward(self, x):
        return [super().forward(x)]
