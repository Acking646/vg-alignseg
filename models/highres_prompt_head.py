from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CrossViewPrototypePrompt(nn.Module):
    """Class-wise source-view prototype prompts for target-view dense prediction.

    The module converts low-resolution part probabilities into per-class feature
    prototypes, aggregates prototypes from all other views, and scores each
    target-view token against those source prototypes. This is a lightweight
    mask-prompt analogue for the no-query multi-view part setting.
    """

    def __init__(self, temperature: float = 10.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))

    def forward(self, token_grid: torch.Tensor, lowres_logits: torch.Tensor) -> torch.Tensor:
        if token_grid.dim() != 5 or lowres_logits.dim() != 5:
            raise ValueError("Expected token_grid and lowres_logits with shape [B, N, H, W, C/K]")

        bsz, n_views, grid_h, grid_w, channels = token_grid.shape
        num_classes = lowres_logits.shape[-1]
        tokens = F.normalize(token_grid.reshape(bsz, n_views, grid_h * grid_w, channels), dim=-1)
        probs = lowres_logits.softmax(dim=-1).reshape(bsz, n_views, grid_h * grid_w, num_classes)

        weighted = torch.einsum("bnpc,bnpk->bnkc", tokens, probs)
        denom = probs.sum(dim=2).clamp_min(1e-6).unsqueeze(-1)
        prototypes = F.normalize(weighted / denom, dim=-1)

        if n_views > 1:
            source_sum = prototypes.sum(dim=1, keepdim=True) - prototypes
            source_prototypes = source_sum / float(n_views - 1)
        else:
            source_prototypes = prototypes
        source_prototypes = F.normalize(source_prototypes, dim=-1)

        temperature = self.log_temperature.exp().clamp(1.0, 100.0)
        proto_logits = temperature * torch.einsum("bnpc,bnkc->bnpk", tokens, source_prototypes)
        return proto_logits.reshape(bsz, n_views, grid_h, grid_w, num_classes).contiguous()


class PromptedHighResRefineHead(nn.Module):
    """High-resolution decoder for pixel-accurate multi-view part masks."""

    def __init__(
        self,
        token_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        use_prototype_logits: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_prototype_logits = use_prototype_logits

        logit_channels = 2 * num_classes + (num_classes if use_prototype_logits else 0)
        self.token_proj = nn.Conv2d(token_dim, hidden_dim, kernel_size=1)
        self.logit_proj = nn.Conv2d(logit_channels, hidden_dim // 2, kernel_size=1)
        self.lowres_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim + hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.SiLU(inplace=True),
            ResidualBlock(hidden_dim),
        )

        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 2, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim // 2),
            nn.SiLU(inplace=True),
        )

        self.fullres_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim + hidden_dim // 2 + 5, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.SiLU(inplace=True),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(
        self,
        token_grid: torch.Tensor,
        coarse_logits: torch.Tensor,
        propagated_logits: torch.Tensor,
        images: torch.Tensor,
        output_size: Tuple[int, int],
        prototype_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, n_views, grid_h, grid_w, token_dim = token_grid.shape
        _, _, _, _, num_classes = coarse_logits.shape
        height, width = output_size

        tokens = token_grid.reshape(bsz * n_views, grid_h, grid_w, token_dim).permute(0, 3, 1, 2).contiguous()
        coarse = coarse_logits.reshape(bsz * n_views, grid_h, grid_w, num_classes).permute(0, 3, 1, 2).contiguous()
        propagated = (
            propagated_logits.reshape(bsz * n_views, grid_h, grid_w, num_classes)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        logit_inputs = [coarse, propagated]
        if self.use_prototype_logits:
            if prototype_logits is None:
                prototype_logits = torch.zeros_like(coarse_logits)
            proto = (
                prototype_logits.reshape(bsz * n_views, grid_h, grid_w, num_classes)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            logit_inputs.append(proto)

        lowres = torch.cat([self.token_proj(tokens), self.logit_proj(torch.cat(logit_inputs, dim=1))], dim=1)
        lowres = self.lowres_fuse(lowres)
        lowres = F.interpolate(lowres, size=output_size, mode="bilinear", align_corners=False)

        image_flat = images.reshape(bsz * n_views, 3, height, width)
        rgb_features = self.rgb_encoder(image_flat)

        yy = torch.linspace(-1.0, 1.0, height, device=images.device, dtype=images.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=images.device, dtype=images.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(bsz * n_views, -1, -1, -1)

        fullres = torch.cat([lowres, rgb_features, image_flat, coords], dim=1)
        logits = self.fullres_fuse(fullres)
        return logits.reshape(bsz, n_views, num_classes, height, width).contiguous()
