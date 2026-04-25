from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_candidates(values: torch.Tensor, candidate_idx: torch.Tensor) -> torch.Tensor:
    batch_size, query_count, topk = candidate_idx.shape
    value_dim = values.shape[-1]

    expanded_values = values.unsqueeze(1).expand(-1, query_count, -1, -1)
    expanded_idx = candidate_idx.unsqueeze(-1).expand(-1, -1, -1, value_dim)
    gathered = torch.gather(expanded_values, dim=2, index=expanded_idx)
    return gathered


class SparseAlign(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 5.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def forward(
        self,
        target_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        target_points: torch.Tensor,
        source_points: torch.Tensor,
        candidate_idx: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        target_tokens = F.normalize(target_tokens, dim=-1)
        source_tokens = F.normalize(source_tokens, dim=-1)

        source_token_candidates = gather_candidates(source_tokens, candidate_idx)
        source_point_candidates = gather_candidates(source_points, candidate_idx)

        cosine = (target_tokens.unsqueeze(2) * source_token_candidates).sum(dim=-1)
        geom_dist = torch.norm(target_points.unsqueeze(2) - source_point_candidates, dim=-1)

        scores = self.alpha * cosine - self.beta * geom_dist
        scores = scores.masked_fill(~candidate_mask, -1e4)

        weights = torch.softmax(scores, dim=-1)
        weights = weights * candidate_mask.to(weights.dtype)

        valid_any = candidate_mask.any(dim=-1, keepdim=True)
        weights = torch.where(valid_any, weights, torch.zeros_like(weights))

        return {
            "weights": weights,
            "scores": scores,
            "source_token_candidates": source_token_candidates,
            "source_point_candidates": source_point_candidates,
        }
