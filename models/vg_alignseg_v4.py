from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frozen_vggt import FrozenVGGTBackbone
from .highres_prompt_head import ResidualBlock
from .v3_prompt_head import ASPP, conv_block


class VGAlignSegV4PartTransfer(nn.Module):
    """Prompt-conditioned binary part transfer across the 8 object views.

    The model receives a source-view binary mask for one actor/part and predicts
    the same actor's binary mask in every view. Final multi-part segmentation is
    produced outside the model by running this head once per actor and composing
    the foreground logits.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        hidden_dim: int = 128,
        refinement_iters: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = FrozenVGGTBackbone(checkpoint_path=checkpoint_path)
        self.hidden_dim = hidden_dim
        self.refinement_iters = refinement_iters

        token_dim = self.backbone.token_dim
        self.token_proj = nn.Conv2d(token_dim, hidden_dim, kernel_size=1)
        self.prompt_proj = nn.Sequential(
            nn.Conv2d(5, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(inplace=True),
        )
        self.lowres_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim + hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
            ResidualBlock(hidden_dim),
            ASPP(hidden_dim),
        )

        rgb0 = hidden_dim // 4
        rgb1 = hidden_dim // 2
        self.rgb_stem = conv_block(3, rgb0)
        self.rgb_down1 = nn.Sequential(nn.Conv2d(rgb0, rgb1, kernel_size=3, stride=2, padding=1), conv_block(rgb1, rgb1))
        self.rgb_down2 = nn.Sequential(
            nn.Conv2d(rgb1, hidden_dim, kernel_size=3, stride=2, padding=1),
            conv_block(hidden_dim, hidden_dim),
        )
        self.dec2 = nn.Sequential(conv_block(hidden_dim * 2, hidden_dim), ResidualBlock(hidden_dim))
        self.dec1 = nn.Sequential(conv_block(hidden_dim + rgb1, rgb1), ResidualBlock(rgb1))
        self.full_prompt_proj = nn.Sequential(
            nn.Conv2d(7, rgb0, kernel_size=3, padding=1),
            nn.GroupNorm(8, rgb0),
            nn.SiLU(inplace=True),
        )
        self.dec0 = nn.Sequential(
            conv_block(rgb1 + rgb0 + rgb0 + 5, rgb1),
            ResidualBlock(rgb1),
            ResidualBlock(rgb1),
        )
        self.classifier = nn.Conv2d(rgb1, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(rgb1, 1, kernel_size=1)

        self.log_temperature = nn.Parameter(torch.log(torch.tensor(10.0)))
        self.point_log_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        images: torch.Tensor,
        source_mask: torch.Tensor,
        source_view: torch.Tensor,
        backbone_outputs: Optional[Dict[str, torch.Tensor]] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if backbone_outputs is None:
            backbone_outputs = self.backbone(images)
        return self.forward_from_backbone_outputs(
            images=images,
            backbone_outputs=backbone_outputs,
            source_mask=source_mask,
            source_view=source_view,
            output_size=output_size or images.shape[-2:],
        )

    def forward_from_backbone_outputs(
        self,
        images: torch.Tensor,
        backbone_outputs: Dict[str, torch.Tensor],
        source_mask: torch.Tensor,
        source_view: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        token_grid = backbone_outputs["token_grid"]
        point_grid = backbone_outputs["point_grid"]
        point_conf_grid = backbone_outputs["point_conf_grid"]
        bsz, n_views, grid_h, grid_w, token_dim = token_grid.shape
        height, width = output_size

        source_mask = self._format_source_mask(source_mask, height, width, images.device, images.dtype)
        source_view = source_view.to(device=images.device, dtype=torch.long).view(bsz)
        source_low = F.adaptive_avg_pool2d(source_mask, (grid_h, grid_w)).clamp(0.0, 1.0)
        source_prompt_low, source_indicator_low = self._source_prompt_maps(
            source_low, source_view, n_views, grid_h, grid_w
        )

        proto_logit = self._prototype_logit(token_grid, source_low, source_view)
        point_logit = self._point_logit(point_grid, point_conf_grid, source_low, source_view)
        prev_logit = proto_logit + point_logit

        logits = None
        boundary_logits = None
        iters = max(1, self.refinement_iters)
        for _ in range(iters):
            logits, boundary_logits = self._decode(
                token_grid=token_grid,
                images=images,
                source_mask=source_mask,
                source_prompt_low=source_prompt_low,
                source_indicator_low=source_indicator_low,
                proto_logit=proto_logit,
                point_logit=point_logit,
                prev_logit=prev_logit,
                output_size=output_size,
            )
            prev_logit = F.adaptive_avg_pool2d(
                logits.reshape(bsz * n_views, 1, height, width), (grid_h, grid_w)
            ).reshape(bsz, n_views, grid_h, grid_w)

        assert logits is not None and boundary_logits is not None
        return {
            **backbone_outputs,
            "binary_logits": logits,
            "boundary_logits": boundary_logits,
            "proto_logits_lowres": proto_logit,
            "point_logits_lowres": point_logit,
            "source_prompt_lowres": source_prompt_low,
        }

    @staticmethod
    def _format_source_mask(
        source_mask: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if source_mask.dim() == 2:
            source_mask = source_mask.unsqueeze(0).unsqueeze(0)
        elif source_mask.dim() == 3:
            source_mask = source_mask.unsqueeze(1)
        elif source_mask.dim() != 4:
            raise ValueError(f"Expected source_mask [B,H,W] or [B,1,H,W], got {tuple(source_mask.shape)}")
        source_mask = source_mask.to(device=device, dtype=dtype)
        if source_mask.shape[-2:] != (height, width):
            source_mask = F.interpolate(source_mask, size=(height, width), mode="nearest")
        return source_mask.clamp(0.0, 1.0)

    @staticmethod
    def _source_prompt_maps(
        source_low: torch.Tensor,
        source_view: torch.Tensor,
        n_views: int,
        grid_h: int,
        grid_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = source_low.shape[0]
        prompt = source_low.new_zeros((bsz, n_views, grid_h, grid_w))
        indicator = source_low.new_zeros((bsz, n_views, grid_h, grid_w))
        for batch_idx in range(bsz):
            view_idx = int(source_view[batch_idx].item())
            prompt[batch_idx, view_idx] = source_low[batch_idx, 0]
            indicator[batch_idx, view_idx].fill_(1.0)
        return prompt, indicator

    def _gather_source_view(self, value: torch.Tensor, source_view: torch.Tensor) -> torch.Tensor:
        bsz = value.shape[0]
        out = []
        for batch_idx in range(bsz):
            out.append(value[batch_idx, int(source_view[batch_idx].item())])
        return torch.stack(out, dim=0)

    def _prototype_logit(
        self,
        token_grid: torch.Tensor,
        source_low: torch.Tensor,
        source_view: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_views, grid_h, grid_w, channels = token_grid.shape
        tokens = F.normalize(token_grid, dim=-1)
        source_tokens = self._gather_source_view(tokens, source_view)
        mask = source_low[:, 0].reshape(bsz, grid_h, grid_w, 1)
        fg_weight = mask.clamp_min(0.0)
        bg_weight = (1.0 - mask).clamp_min(0.0)
        fg_proto = (source_tokens * fg_weight).sum(dim=(1, 2)) / fg_weight.sum(dim=(1, 2)).clamp_min(1e-6)
        bg_proto = (source_tokens * bg_weight).sum(dim=(1, 2)) / bg_weight.sum(dim=(1, 2)).clamp_min(1e-6)
        fg_proto = F.normalize(fg_proto, dim=-1)
        bg_proto = F.normalize(bg_proto, dim=-1)
        fg_sim = torch.einsum("bnhwc,bc->bnhw", tokens, fg_proto)
        bg_sim = torch.einsum("bnhwc,bc->bnhw", tokens, bg_proto)
        temperature = self.log_temperature.exp().clamp(1.0, 100.0)
        return temperature * (fg_sim - bg_sim)

    def _point_logit(
        self,
        point_grid: torch.Tensor,
        point_conf_grid: torch.Tensor,
        source_low: torch.Tensor,
        source_view: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_views, grid_h, grid_w, _ = point_grid.shape
        num_points = grid_h * grid_w
        points = point_grid.reshape(bsz, n_views * num_points, 3)
        mean = points.mean(dim=1, keepdim=True)
        std = points.std(dim=1, keepdim=True).clamp_min(1e-3)
        norm_points = ((point_grid.reshape(bsz, n_views, num_points, 3) - mean[:, None]) / std[:, None]).reshape(
            bsz, n_views, grid_h, grid_w, 3
        )
        source_points = self._gather_source_view(norm_points, source_view).reshape(bsz, num_points, 3)
        source_conf = self._gather_source_view(point_conf_grid, source_view).reshape(bsz, num_points)
        valid = (source_low[:, 0].reshape(bsz, num_points) > 0.05) & (source_conf > 1e-6)

        target_points = norm_points.reshape(bsz, n_views * num_points, 3)
        dists = torch.cdist(target_points, source_points, p=2)
        dists = dists.masked_fill(~valid[:, None, :], 1.0e4)
        min_dist = dists.min(dim=-1).values.reshape(bsz, n_views, grid_h, grid_w)
        no_prompt = ~valid.any(dim=1)
        if no_prompt.any():
            min_dist[no_prompt] = 1.0e4
        scale = F.softplus(self.point_log_scale).clamp(0.1, 20.0)
        return -scale * min_dist.clamp(max=20.0)

    def _decode(
        self,
        token_grid: torch.Tensor,
        images: torch.Tensor,
        source_mask: torch.Tensor,
        source_prompt_low: torch.Tensor,
        source_indicator_low: torch.Tensor,
        proto_logit: torch.Tensor,
        point_logit: torch.Tensor,
        prev_logit: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_views, grid_h, grid_w, token_dim = token_grid.shape
        height, width = output_size
        tokens = token_grid.reshape(bsz * n_views, grid_h, grid_w, token_dim).permute(0, 3, 1, 2).contiguous()
        low_prompt = torch.stack(
            [
                proto_logit,
                point_logit,
                source_prompt_low,
                source_indicator_low,
                torch.sigmoid(prev_logit),
            ],
            dim=2,
        ).reshape(bsz * n_views, 5, grid_h, grid_w)
        lowres = torch.cat([self.token_proj(tokens), self.prompt_proj(low_prompt)], dim=1)
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

        source_full = source_mask.new_zeros((bsz, n_views, height, width))
        source_indicator_full = source_mask.new_zeros((bsz, n_views, height, width))
        for batch_idx in range(bsz):
            view_idx = int(source_indicator_low[batch_idx].reshape(n_views, -1).sum(dim=1).argmax().item())
            source_full[batch_idx, view_idx] = source_mask[batch_idx, 0]
            source_indicator_full[batch_idx, view_idx].fill_(1.0)
        full_prompt = torch.stack(
            [
                F.interpolate(proto_logit.reshape(bsz * n_views, 1, grid_h, grid_w), output_size, mode="bilinear", align_corners=False)
                .reshape(bsz, n_views, height, width),
                F.interpolate(point_logit.reshape(bsz * n_views, 1, grid_h, grid_w), output_size, mode="bilinear", align_corners=False)
                .reshape(bsz, n_views, height, width),
                F.interpolate(source_prompt_low.reshape(bsz * n_views, 1, grid_h, grid_w), output_size, mode="nearest")
                .reshape(bsz, n_views, height, width),
                source_indicator_full,
                torch.sigmoid(
                    F.interpolate(prev_logit.reshape(bsz * n_views, 1, grid_h, grid_w), output_size, mode="bilinear", align_corners=False)
                    .reshape(bsz, n_views, height, width)
                ),
                source_full,
                (images.mean(dim=2) < 0.98).to(images.dtype),
            ],
            dim=2,
        ).reshape(bsz * n_views, 7, height, width)
        full_prompt = self.full_prompt_proj(full_prompt)

        dec0 = F.interpolate(dec1, size=output_size, mode="bilinear", align_corners=False)
        dec0 = self.dec0(torch.cat([dec0, rgb0, full_prompt, image_flat, coords], dim=1))
        logits = self.classifier(dec0).reshape(bsz, n_views, 1, height, width).contiguous()
        boundary_logits = self.boundary_head(dec0).reshape(bsz, n_views, 1, height, width).contiguous()
        return logits, boundary_logits
