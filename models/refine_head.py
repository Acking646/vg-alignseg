from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class RefineHead(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.token_proj = nn.Conv2d(token_dim, hidden_dim, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim + 2 * num_classes, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualConvBlock(hidden_dim, num_groups=num_groups) for _ in range(num_blocks)])
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(
        self,
        token_grid: torch.Tensor,
        coarse_logits: torch.Tensor,
        propagated_logits: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        bsz, n_views, grid_h, grid_w, token_dim = token_grid.shape
        _, _, _, _, num_classes = coarse_logits.shape

        tokens = token_grid.reshape(bsz * n_views, grid_h, grid_w, token_dim).permute(0, 3, 1, 2).contiguous()
        coarse = coarse_logits.reshape(bsz * n_views, grid_h, grid_w, num_classes).permute(0, 3, 1, 2).contiguous()
        propagated = propagated_logits.reshape(bsz * n_views, grid_h, grid_w, num_classes).permute(0, 3, 1, 2).contiguous()

        fused = self.token_proj(tokens)
        fused = torch.cat([fused, coarse, propagated], dim=1)
        fused = self.fuse(fused)
        fused = self.blocks(fused)

        lowres_logits = self.classifier(fused)
        fullres_logits = F.interpolate(lowres_logits, size=output_size, mode="bilinear", align_corners=False)

        lowres_logits = lowres_logits.permute(0, 2, 3, 1).reshape(bsz, n_views, grid_h, grid_w, num_classes).contiguous()
        fullres_logits = fullres_logits.reshape(bsz, n_views, num_classes, output_size[0], output_size[1]).contiguous()

        return {
            "lowres_logits": lowres_logits,
            "fullres_logits": fullres_logits,
        }
