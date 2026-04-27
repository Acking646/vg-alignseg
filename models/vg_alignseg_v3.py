from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coarse_head import CoarseHead
from .frozen_vggt import FrozenVGGTBackbone
from .geometry_pruner import GeometryPruner
from .propagator import LogitPropagator
from .sparse_align import SparseAlign
from .v3_prompt_head import DeepLabUNetRefineHead, ObjectAwarePrototypePrompt


class VGAlignSegV3(nn.Module):
    """VG-AlignSeg V3 with object-aware prototypes and a stronger high-res decoder."""

    def __init__(
        self,
        num_classes: int,
        checkpoint_path: Optional[str] = None,
        topk: int = 8,
        min_confidence: float = 0.05,
        coarse_hidden_dim: int = 256,
        refine_hidden_dim: int = 128,
        use_prototypes: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_prototypes = use_prototypes

        self.backbone = FrozenVGGTBackbone(checkpoint_path=checkpoint_path)
        self.coarse_head = CoarseHead(
            token_dim=self.backbone.token_dim,
            num_classes=num_classes,
            hidden_dim=coarse_hidden_dim,
        )
        self.geometry_pruner = GeometryPruner(topk=topk, min_confidence=min_confidence)
        self.sparse_align = SparseAlign()
        self.propagator = LogitPropagator()
        self.prototype_prompt = ObjectAwarePrototypePrompt(
            token_dim=self.backbone.token_dim,
            num_classes=num_classes,
        )
        self.refine_head = DeepLabUNetRefineHead(
            token_dim=self.backbone.token_dim,
            num_classes=num_classes,
            hidden_dim=refine_hidden_dim,
            use_prototype_logits=use_prototypes,
        )

    def forward(
        self,
        images: torch.Tensor,
        backbone_outputs: Optional[Dict[str, torch.Tensor]] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if images.dim() != 5:
            raise ValueError(f"Expected images with shape [B, N, 3, H, W], got {tuple(images.shape)}")
        if backbone_outputs is None:
            backbone_outputs = self.backbone(images)
        return self.forward_from_backbone_outputs(images, backbone_outputs, output_size=output_size or images.shape[-2:])

    def forward_from_backbone_outputs(
        self,
        images: torch.Tensor,
        backbone_outputs: Dict[str, torch.Tensor],
        output_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        token_grid = backbone_outputs["token_grid"]
        point_grid = backbone_outputs["point_grid"]
        point_conf_grid = backbone_outputs["point_conf_grid"]

        coarse_logits = self.coarse_head(token_grid)
        coarse_logits_fullres = self._upsample_lowres_logits(coarse_logits, output_size)

        propagated_logits = self._propagate_logits(token_grid, point_grid, point_conf_grid, coarse_logits)
        propagated_logits_fullres = self._upsample_lowres_logits(propagated_logits, output_size)

        prototype_logits = None
        if self.use_prototypes:
            prototype_logits = self.prototype_prompt(token_grid, coarse_logits)
        final_logits, boundary_logits = self.refine_head(
            token_grid=token_grid,
            coarse_logits=coarse_logits,
            propagated_logits=propagated_logits,
            prototype_logits=prototype_logits,
            images=images,
            output_size=output_size,
        )

        return {
            **backbone_outputs,
            "coarse_logits_lowres": coarse_logits,
            "coarse_logits": coarse_logits_fullres,
            "propagated_logits_lowres": propagated_logits,
            "propagated_logits": propagated_logits_fullres,
            "prototype_logits_lowres": prototype_logits,
            "prototype_logits": None if prototype_logits is None else self._upsample_lowres_logits(prototype_logits, output_size),
            "boundary_logits": boundary_logits,
            "final_logits": final_logits,
        }

    def _propagate_logits(
        self,
        token_grid: torch.Tensor,
        point_grid: torch.Tensor,
        point_conf_grid: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_views, grid_h, grid_w, token_dim = token_grid.shape
        num_classes = coarse_logits.shape[-1]
        num_tokens = grid_h * grid_w

        flat_tokens = token_grid.reshape(bsz, n_views, num_tokens, token_dim)
        flat_points = point_grid.reshape(bsz, n_views, num_tokens, 3)
        flat_conf = point_conf_grid.reshape(bsz, n_views, num_tokens)
        flat_logits = coarse_logits.reshape(bsz, n_views, num_tokens, num_classes)

        propagated_by_view: List[torch.Tensor] = []
        template = torch.zeros_like(flat_logits[:, 0])

        for target_idx in range(n_views):
            target_tokens = flat_tokens[:, target_idx]
            target_points = flat_points[:, target_idx]
            target_conf = flat_conf[:, target_idx]

            messages = []
            for source_idx in range(n_views):
                if source_idx == target_idx:
                    continue
                prune_outputs = self.geometry_pruner(
                    target_points=target_points,
                    source_points=flat_points[:, source_idx],
                    target_conf=target_conf,
                    source_conf=flat_conf[:, source_idx],
                )
                align_outputs = self.sparse_align(
                    target_tokens=target_tokens,
                    source_tokens=flat_tokens[:, source_idx],
                    target_points=target_points,
                    source_points=flat_points[:, source_idx],
                    candidate_idx=prune_outputs["candidate_idx"],
                    candidate_mask=prune_outputs["candidate_mask"],
                )
                messages.append(
                    self.propagator(
                        source_logits=flat_logits[:, source_idx],
                        alignment_weights=align_outputs["weights"],
                        candidate_idx=prune_outputs["candidate_idx"],
                    )
                )
            propagated_by_view.append(self.propagator.aggregate(messages, template))

        propagated_logits = torch.stack(propagated_by_view, dim=1)
        return propagated_logits.reshape(bsz, n_views, grid_h, grid_w, num_classes).contiguous()

    @staticmethod
    def _upsample_lowres_logits(logits: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        bsz, n_views, grid_h, grid_w, num_classes = logits.shape
        x = logits.reshape(bsz * n_views, grid_h, grid_w, num_classes).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x.reshape(bsz, n_views, num_classes, output_size[0], output_size[1]).contiguous()
