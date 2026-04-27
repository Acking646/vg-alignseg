from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .highres_prompt_head import ResidualBlock


class ObjectAwarePrototypePrompt(nn.Module):
    """Object-level cross-view prototypes plus a small global prototype prior."""

    def __init__(self, token_dim: int, num_classes: int, temperature: float = 10.0, global_weight: float = 0.25) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.global_weight = global_weight
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))
        self.global_prototypes = nn.Parameter(torch.randn(num_classes, token_dim) * 0.02)
        self.presence_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, token_grid: torch.Tensor, lowres_logits: torch.Tensor) -> torch.Tensor:
        bsz, n_views, grid_h, grid_w, channels = token_grid.shape
        num_classes = lowres_logits.shape[-1]
        tokens = F.normalize(token_grid.reshape(bsz, n_views, grid_h * grid_w, channels), dim=-1)
        probs = lowres_logits.softmax(dim=-1).reshape(bsz, n_views, grid_h * grid_w, num_classes)

        weighted = torch.einsum("bnpc,bnpk->bnkc", tokens, probs)
        denom = probs.sum(dim=2).clamp_min(1e-6).unsqueeze(-1)
        view_prototypes = F.normalize(weighted / denom, dim=-1)

        if n_views > 1:
            source_prototypes = (view_prototypes.sum(dim=1, keepdim=True) - view_prototypes) / float(n_views - 1)
        else:
            source_prototypes = view_prototypes
        source_prototypes = F.normalize(source_prototypes, dim=-1)

        object_presence = probs.mean(dim=(1, 2))
        presence_gate = torch.sigmoid(8.0 * (object_presence + self.presence_bias.unsqueeze(0) - 0.01))
        local_logits = torch.einsum("bnpc,bnkc->bnpk", tokens, source_prototypes)
        local_logits = local_logits * presence_gate[:, None, None, :]

        global_prototypes = F.normalize(self.global_prototypes, dim=-1)
        global_logits = torch.einsum("bnpc,kc->bnpk", tokens, global_prototypes)

        temperature = self.log_temperature.exp().clamp(1.0, 100.0)
        proto_logits = temperature * (local_logits + self.global_weight * global_logits)
        return proto_logits.reshape(bsz, n_views, grid_h, grid_w, num_classes).contiguous()


class ASPP(nn.Module):
    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 2, 4, 6)) -> None:
        super().__init__()
        branches = []
        for dilation in dilations:
            if dilation == 1:
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(channels, channels, kernel_size=1),
                        nn.GroupNorm(8, channels),
                        nn.SiLU(inplace=True),
                    )
                )
            else:
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
                        nn.GroupNorm(8, channels),
                        nn.SiLU(inplace=True),
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(channels * len(dilations), channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            ResidualBlock(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(inplace=True),
    )


class DeepLabUNetRefineHead(nn.Module):
    """ASPP + U-Net style decoder for high-resolution part boundaries."""

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
        rgb0 = hidden_dim // 4
        rgb1 = hidden_dim // 2

        logit_channels = 2 * num_classes + (num_classes if use_prototype_logits else 0)
        self.token_proj = nn.Conv2d(token_dim, hidden_dim, kernel_size=1)
        self.logit_proj = nn.Sequential(
            nn.Conv2d(logit_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.lowres_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
            ResidualBlock(hidden_dim),
            ASPP(hidden_dim),
        )

        self.rgb_stem = conv_block(3, rgb0)
        self.rgb_down1 = nn.Sequential(nn.Conv2d(rgb0, rgb1, kernel_size=3, stride=2, padding=1), conv_block(rgb1, rgb1))
        self.rgb_down2 = nn.Sequential(
            nn.Conv2d(rgb1, hidden_dim, kernel_size=3, stride=2, padding=1),
            conv_block(hidden_dim, hidden_dim),
        )

        self.dec2 = nn.Sequential(conv_block(hidden_dim * 2, hidden_dim), ResidualBlock(hidden_dim))
        self.dec1 = nn.Sequential(conv_block(hidden_dim + rgb1, rgb1), ResidualBlock(rgb1))
        self.logit_full_proj = nn.Sequential(
            nn.Conv2d(logit_channels, rgb0, kernel_size=1),
            nn.GroupNorm(8, rgb0),
            nn.SiLU(inplace=True),
        )
        self.dec0 = nn.Sequential(
            conv_block(rgb1 + rgb0 + rgb0 + 5, rgb1),
            ResidualBlock(rgb1),
            ResidualBlock(rgb1),
        )
        self.classifier = nn.Conv2d(rgb1, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(rgb1, 1, kernel_size=1)

    def forward(
        self,
        token_grid: torch.Tensor,
        coarse_logits: torch.Tensor,
        propagated_logits: torch.Tensor,
        images: torch.Tensor,
        output_size: Tuple[int, int],
        prototype_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        lowres_logits = torch.cat(logit_inputs, dim=1)
        lowres = torch.cat([self.token_proj(tokens), self.logit_proj(lowres_logits)], dim=1)
        lowres = self.lowres_fuse(lowres)

        image_flat = images.reshape(bsz * n_views, 3, height, width)
        rgb0 = self.rgb_stem(image_flat)
        rgb1 = self.rgb_down1(rgb0)
        rgb2 = self.rgb_down2(rgb1)

        dec2 = F.interpolate(lowres, size=rgb2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([dec2, rgb2], dim=1))
        dec1 = F.interpolate(dec2, size=rgb1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([dec1, rgb1], dim=1))

        yy = torch.linspace(-1.0, 1.0, height, device=images.device, dtype=images.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=images.device, dtype=images.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(bsz * n_views, -1, -1, -1)

        full_logits = F.interpolate(lowres_logits, size=output_size, mode="bilinear", align_corners=False)
        full_logits = self.logit_full_proj(full_logits)
        dec0 = F.interpolate(dec1, size=output_size, mode="bilinear", align_corners=False)
        dec0 = self.dec0(torch.cat([dec0, rgb0, full_logits, image_flat, coords], dim=1))
        logits = self.classifier(dec0).reshape(bsz, n_views, num_classes, height, width).contiguous()
        boundary_logits = self.boundary_head(dec0).reshape(bsz, n_views, 1, height, width).contiguous()
        return logits, boundary_logits
