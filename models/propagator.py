from typing import Iterable

import torch
import torch.nn as nn

from .sparse_align import gather_candidates


class LogitPropagator(nn.Module):
    def forward(
        self,
        source_logits: torch.Tensor,
        alignment_weights: torch.Tensor,
        candidate_idx: torch.Tensor,
    ) -> torch.Tensor:
        candidate_logits = gather_candidates(source_logits, candidate_idx)
        propagated = (alignment_weights.unsqueeze(-1) * candidate_logits).sum(dim=2)
        return propagated

    @staticmethod
    def aggregate(messages: Iterable[torch.Tensor], template: torch.Tensor) -> torch.Tensor:
        messages = list(messages)
        if not messages:
            return torch.zeros_like(template)
        return torch.stack(messages, dim=0).mean(dim=0)
