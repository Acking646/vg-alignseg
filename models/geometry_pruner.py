from typing import Dict, Optional

import torch
import torch.nn as nn


class GeometryPruner(nn.Module):
    def __init__(self, topk: int = 8, min_confidence: float = 0.05) -> None:
        super().__init__()
        self.topk = topk
        self.min_confidence = min_confidence

    def forward(
        self,
        target_points: torch.Tensor,
        source_points: torch.Tensor,
        target_conf: Optional[torch.Tensor] = None,
        source_conf: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if target_points.dim() != 3 or source_points.dim() != 3:
            raise ValueError("Expected flattened point tensors with shape [B, P, 3]")

        distances = torch.cdist(target_points, source_points)
        valid = torch.isfinite(distances)

        if target_conf is not None:
            valid = valid & (target_conf.unsqueeze(-1) >= self.min_confidence)
        if source_conf is not None:
            valid = valid & (source_conf.unsqueeze(1) >= self.min_confidence)

        masked_distances = distances.masked_fill(~valid, float("inf"))
        topk = min(self.topk, source_points.shape[1])
        candidate_dist, candidate_idx = masked_distances.topk(topk, dim=-1, largest=False)
        candidate_mask = torch.isfinite(candidate_dist)

        return {
            "candidate_idx": candidate_idx,
            "candidate_dist": candidate_dist,
            "candidate_mask": candidate_mask,
        }
