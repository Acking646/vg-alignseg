import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENDORED_VGGT_ROOT = os.path.join(_REPO_ROOT, "vggt")
if os.path.isdir(_VENDORED_VGGT_ROOT) and _VENDORED_VGGT_ROOT not in sys.path:
    sys.path.insert(0, _VENDORED_VGGT_ROOT)

from vggt.models.vggt import VGGT


def _default_checkpoint_path() -> str:
    return os.path.join(_REPO_ROOT, "vggt", "weights", "VGGT-1B", "model.pt")


class FrozenVGGTBackbone(nn.Module):
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        token_layer_idx: int = -1,
        min_confidence: float = 1e-6,
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path or os.environ.get("VGGT_CKPT", _default_checkpoint_path())
        self.token_layer_idx = token_layer_idx
        self.min_confidence = min_confidence

        self.model = VGGT()
        self._load_weights()
        self.model.eval()
        self.model.requires_grad_(False)

        embed_dim = self.model.aggregator.camera_token.shape[-1]
        self.token_dim = 2 * embed_dim
        self.patch_start_idx = self.model.aggregator.patch_start_idx

    def _load_weights(self) -> None:
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"VGGT checkpoint not found at {self.checkpoint_path}. "
                "Set VGGT_CKPT or place model.pt under vggt/weights/VGGT-1B/."
            )
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if images.dim() != 5:
            raise ValueError(f"Expected images with shape [B, N, 3, H, W], got {tuple(images.shape)}")

        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)
            with torch.amp.autocast(device_type=images.device.type, enabled=False):
                point_map, point_conf = self.model.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                depth_map, depth_conf = self.model.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )

        token_grid = self._extract_token_grid(aggregated_tokens_list[self.token_layer_idx], patch_start_idx, images.shape)
        point_grid, point_conf_grid = self._pool_geometry(point_map, point_conf, token_grid.shape[2:4])
        depth_conf_grid = self._pool_confidence(depth_conf, token_grid.shape[2:4])

        return {
            "token_grid": token_grid,
            "point_map": point_map,
            "point_conf": point_conf,
            "depth_map": depth_map,
            "depth_conf": depth_conf,
            "point_grid": point_grid,
            "point_conf_grid": point_conf_grid,
            "depth_conf_grid": depth_conf_grid,
        }

    def _extract_token_grid(
        self,
        token_tensor: torch.Tensor,
        patch_start_idx: int,
        image_shape: torch.Size,
    ) -> torch.Tensor:
        bsz, n_views, _, token_dim = token_tensor.shape
        _, _, _, height, width = image_shape
        grid_h = height // self.model.aggregator.patch_size
        grid_w = width // self.model.aggregator.patch_size

        patch_tokens = token_tensor[:, :, patch_start_idx:, :]
        patch_tokens = patch_tokens.reshape(bsz, n_views, grid_h, grid_w, token_dim).contiguous()
        return patch_tokens

    def _pool_geometry(
        self,
        point_map: torch.Tensor,
        point_conf: torch.Tensor,
        output_size: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_views, height, width, _ = point_map.shape
        grid_h, grid_w = output_size

        points = point_map.permute(0, 1, 4, 2, 3).reshape(bsz * n_views, 3, height, width)
        conf = point_conf.reshape(bsz * n_views, 1, height, width)

        weighted_points = F.adaptive_avg_pool2d(points * conf, (grid_h, grid_w))
        pooled_conf = F.adaptive_avg_pool2d(conf, (grid_h, grid_w)).clamp_min(self.min_confidence)
        pooled_points = weighted_points / pooled_conf

        pooled_points = pooled_points.reshape(bsz, n_views, 3, grid_h, grid_w).permute(0, 1, 3, 4, 2).contiguous()
        pooled_conf = pooled_conf.reshape(bsz, n_views, grid_h, grid_w)
        return pooled_points, pooled_conf

    def _pool_confidence(self, confidence: torch.Tensor, output_size: torch.Size) -> torch.Tensor:
        bsz, n_views, height, width = confidence.shape
        grid_h, grid_w = output_size
        conf = confidence.reshape(bsz * n_views, 1, height, width)
        conf = F.adaptive_avg_pool2d(conf, (grid_h, grid_w))
        return conf.reshape(bsz, n_views, grid_h, grid_w)
