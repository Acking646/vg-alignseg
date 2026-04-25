import torch
import torch.nn as nn


class CoarseHead(nn.Module):
    def __init__(self, token_dim: int, num_classes: int, hidden_dim: int = 256, num_groups: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(token_dim, hidden_dim, kernel_size=1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, token_grid: torch.Tensor) -> torch.Tensor:
        if token_grid.dim() != 5:
            raise ValueError(f"Expected token grid [B, N, H, W, C], got {tuple(token_grid.shape)}")

        bsz, n_views, grid_h, grid_w, channels = token_grid.shape
        x = token_grid.reshape(bsz * n_views, grid_h, grid_w, channels).permute(0, 3, 1, 2).contiguous()
        logits = self.net(x)
        logits = logits.permute(0, 2, 3, 1).reshape(bsz, n_views, grid_h, grid_w, -1).contiguous()
        return logits
