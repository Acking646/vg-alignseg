from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coarse_head import CoarseHead
from .frozen_vggt import FrozenVGGTBackbone
from .geometry_pruner import GeometryPruner
from .propagator import LogitPropagator
from .refine_head import RefineHead
from .sparse_align import SparseAlign


class VGAlignSegV1(nn.Module):
    def __init__(
        self,
        num_classes: int,
        checkpoint_path: Optional[str] = None,
        topk: int = 8,
        min_confidence: float = 0.05,
        coarse_hidden_dim: int = 256,
        refine_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.backbone = FrozenVGGTBackbone(checkpoint_path=checkpoint_path)
        self.coarse_head = CoarseHead(
            token_dim=self.backbone.token_dim,
            num_classes=num_classes,
            hidden_dim=coarse_hidden_dim,
        )
        self.geometry_pruner = GeometryPruner(topk=topk, min_confidence=min_confidence)
        self.sparse_align = SparseAlign()
        self.propagator = LogitPropagator()
        self.refine_head = RefineHead(
            token_dim=self.backbone.token_dim,
            num_classes=num_classes,
            hidden_dim=refine_hidden_dim,
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if images.dim() != 5:
            raise ValueError(f"Expected images with shape [B, N, 3, H, W], got {tuple(images.shape)}")

        bsz, n_views, _, height, width = images.shape
        backbone_outputs = self.backbone(images)

        token_grid = backbone_outputs["token_grid"]
        point_grid = backbone_outputs["point_grid"]
        point_conf_grid = backbone_outputs["point_conf_grid"]

        coarse_logits = self.coarse_head(token_grid)
        coarse_logits_fullres = self._upsample_lowres_logits(coarse_logits, (height, width))

        bsz, n_views, grid_h, grid_w, token_dim = token_grid.shape
        _, _, _, _, num_classes = coarse_logits.shape
        num_tokens = grid_h * grid_w

        flat_tokens = token_grid.reshape(bsz, n_views, num_tokens, token_dim)
        flat_points = point_grid.reshape(bsz, n_views, num_tokens, 3)
        flat_conf = point_conf_grid.reshape(bsz, n_views, num_tokens)
        flat_logits = coarse_logits.reshape(bsz, n_views, num_tokens, num_classes)

        propagated_by_view: List[torch.Tensor] = []
        pair_metadata: List[List[Dict[str, torch.Tensor]]] = []

        template = torch.zeros_like(flat_logits[:, 0])
        for target_idx in range(n_views):
            target_tokens = flat_tokens[:, target_idx]
            target_points = flat_points[:, target_idx]
            target_conf = flat_conf[:, target_idx]

            messages = []
            target_pair_meta = []
            for source_idx in range(n_views):
                if source_idx == target_idx:
                    continue

                source_tokens = flat_tokens[:, source_idx]
                source_points = flat_points[:, source_idx]
                source_conf = flat_conf[:, source_idx]
                source_logits = flat_logits[:, source_idx]

                prune_outputs = self.geometry_pruner(
                    target_points=target_points,
                    source_points=source_points,
                    target_conf=target_conf,
                    source_conf=source_conf,
                )
                align_outputs = self.sparse_align(
                    target_tokens=target_tokens,
                    source_tokens=source_tokens,
                    target_points=target_points,
                    source_points=source_points,
                    candidate_idx=prune_outputs["candidate_idx"],
                    candidate_mask=prune_outputs["candidate_mask"],
                )
                propagated_logits = self.propagator(
                    source_logits=source_logits,
                    alignment_weights=align_outputs["weights"],
                    candidate_idx=prune_outputs["candidate_idx"],
                )
                messages.append(propagated_logits)
                target_pair_meta.append(
                    {
                        "source_idx": torch.full((bsz,), source_idx, device=images.device, dtype=torch.long),
                        "candidate_idx": prune_outputs["candidate_idx"],
                        "candidate_mask": prune_outputs["candidate_mask"],
                        "weights": align_outputs["weights"],
                    }
                )

            propagated_by_view.append(self.propagator.aggregate(messages, template))
            pair_metadata.append(target_pair_meta)

        propagated_logits = torch.stack(propagated_by_view, dim=1)
        propagated_logits = propagated_logits.reshape(bsz, n_views, grid_h, grid_w, num_classes)
        propagated_logits_fullres = self._upsample_lowres_logits(propagated_logits, (height, width))

        refine_outputs = self.refine_head(
            token_grid=token_grid,
            coarse_logits=coarse_logits,
            propagated_logits=propagated_logits,
            output_size=(height, width),
        )

        return {
            **backbone_outputs,
            "coarse_logits_lowres": coarse_logits,
            "coarse_logits": coarse_logits_fullres,
            "propagated_logits_lowres": propagated_logits,
            "propagated_logits": propagated_logits_fullres,
            "final_logits_lowres": refine_outputs["lowres_logits"],
            "final_logits": refine_outputs["fullres_logits"],
            "pair_metadata": pair_metadata,
        }

    def _upsample_lowres_logits(self, logits: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        bsz, n_views, grid_h, grid_w, num_classes = logits.shape
        x = logits.reshape(bsz * n_views, grid_h, grid_w, num_classes).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x.reshape(bsz, n_views, num_classes, output_size[0], output_size[1]).contiguous()
