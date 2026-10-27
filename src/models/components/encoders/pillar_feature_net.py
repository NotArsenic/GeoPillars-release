import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


class GroupedVectorAttention(nn.Module):
    """
    Intra-Pillar Vector Grouped Attention

    Computes attention based on the relative spatial relationships between points within a pillar.
    By projecting features into a lower-dimensional space for relation geometry, it captures micro-geometry efficiently.
    """

    def __init__(self, in_channels=64, groups=8):
        super().__init__()
        self.groups = groups
        self.channels_per_group = in_channels // groups

        # Project input down to 'G' dimensions.
        self.q_proj = nn.Linear(in_channels, groups, bias=False)
        self.k_proj = nn.Linear(in_channels, groups, bias=False)

        self.v_proj = nn.Linear(in_channels, in_channels, bias=False)

        # Learnable scaling factor for the spatial difference
        self.relation_scale = nn.Parameter(torch.ones(1, 1, 1, groups))

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        P, N, C = x.shape

        # Compute queries and keys for relation geometry (P, N, G)
        q = self.q_proj(x)
        k = self.k_proj(x)

        # Compute Directional Difference (P, N, N, G)
        # Using subtraction (q - k) directly encodes relative spatial orientation
        diff = q.unsqueeze(2) - k.unsqueeze(1)
        attn_logits = diff * self.relation_scale

        # Apply Sparsity Mask
        if mask is not None:

            # Expand mask to (P, 1, N, 1) to block padding points in the keys
            attn_mask = mask.unsqueeze(1).unsqueeze(-1)
            attn_logits = attn_logits.masked_fill(~attn_mask, -1e4)

        # Group-wise Attention Weights (P, N, N, G)
        attn_weights = F.softmax(attn_logits, dim=2)

        # Channel-Grouped Value Aggregation
        v = self.v_proj(x).view(P, N, self.groups, self.channels_per_group)

        out = torch.einsum("pnmg,pmgc->pngc", attn_weights, v)  # (P, N, G, C/G)
        out = out.contiguous().view(P, N, C)

        # 6. Residual Connection
        return x + self.gamma * out


@MODELS.register_module()
class GeoPillarFeatureNet(BaseModule):
    """
    Processes 11D geometric vectors, preserves micro-geometry via Vector Attention, and injects absolute spatial context.
    """

    def __init__(
        self,
        in_channels=11,
        out_channels=64,
        max_points=32,
        grid_size_x=512,
        grid_size_y=512,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.max_points = max_points
        self.out_channels = out_channels
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y

        # Initial expansion from 11D to high-dimensional space
        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
        self.norm1 = nn.BatchNorm1d(out_channels)

        # Micro-geometry extraction engine
        self.vector_attention = GroupedVectorAttention(
            in_channels=out_channels, groups=8
        )

        # Secondary processing after spatial relations are mapped
        self.linear2 = nn.Linear(out_channels, out_channels, bias=False)
        self.norm2 = nn.BatchNorm1d(out_channels)

        # 2D Absolute Positional Encoding for BEV context
        # Initialized as a learnable parameter map
        self.bev_pos_encoding = nn.Parameter(
            torch.randn(1, out_channels, grid_size_y, grid_size_x) * 0.01
        )

    def forward(self, features, num_points, coors):
        """
        Args:
            features: (P, N, C_in) The explicit spatial vectors.
            num_points: (P,) The actual point count in each pillar before padding.
            coors: (P, 4) Coordinates for each pillar (batch_idx, z, y, x).
        """
        P, N, _ = features.shape
        C = self.out_channels

        # Create Boolean Mask
        # True for valid points, False for padded zeros
        point_indices = (
            torch.arange(N, device=features.device).unsqueeze(0).expand(P, N)
        )
        mask = point_indices < num_points.unsqueeze(1)

        # Linear Expansion
        # Reshape to (P*N, C) for BatchNorm1d, then revert
        x = self.linear1(features).view(-1, C)
        x = F.relu(self.norm1(x)).view(P, N, C)

        # Extract Structural Fingerprint
        x = self.vector_attention(x, mask=mask)

        # Final Point Projection
        x = self.linear2(x).view(-1, C)
        x = F.relu(self.norm2(x)).view(P, N, C)

        # Apply mask before max pooling so padded points don't corrupt the max value
        x = x.masked_fill(~mask.unsqueeze(-1), -1e4)

        # Collapse Pillar to single vector (P, C)
        pillar_features, _ = torch.max(x, dim=1)

        # Scatter to 2D Pseudo-Image (BEV)
        batch_size = coors[:, 0].max().item() + 1
        canvas = torch.zeros(
            batch_size,
            C,
            self.grid_size_y,
            self.grid_size_x,
            dtype=pillar_features.dtype,
            device=pillar_features.device,
        )

        batch_idx = coors[:, 0].long()
        y_idx = coors[:, 2].long()
        x_idx = coors[:, 3].long()

        canvas[batch_idx, :, y_idx, x_idx] = pillar_features

        # Inject Absolute Macro-Semantic Context
        canvas = canvas + self.bev_pos_encoding

        return canvas
