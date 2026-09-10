"""Relation-aware, signed graph deliberation used after evidence admission."""
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
from torch import nn


class VOIRDeliberation(nn.Module):
    """Two-layer signed message-passing engine with relation probabilities.

    Relation classes are support, contradiction, qualification and dependence.
    Callers must supply a hard comparability mask; masked pairs never interact.
    """
    relation_names = ("support", "contradiction", "qualification", "dependence")

    def __init__(self, dim: int = 256, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.relation = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 4))
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.head = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor, confidence: torch.Tensor, uncertainty: torch.Tensor,
                comparable: torch.Tensor) -> Dict[str, torch.Tensor]:
        # tokens (B,M,D); comparable (B,M,M), True only for clinically valid pairs.
        b, m, d = tokens.shape
        pair = torch.cat([tokens.unsqueeze(2).expand(-1, -1, m, -1), tokens.unsqueeze(1).expand(-1, m, -1, -1)], -1)
        relations = self.relation(pair).softmax(-1)
        relations = relations * comparable.unsqueeze(-1)
        support, contradiction, qualification, dependence = relations.unbind(-1)
        weight = confidence * (1 - uncertainty)
        signed = (support - contradiction) * (1 - dependence) * weight.unsqueeze(1)
        x = tokens
        # MHA blocks disallowed keys; all-false rows are made self-comparable safely.
        mask = ~comparable
        eye = torch.eye(m, dtype=torch.bool, device=tokens.device).unsqueeze(0)
        mask = mask & ~eye
        for attn, norm in zip(self.layers, self.norms):
            # MultiheadAttention expects a three-dimensional mask per head.
            attn_mask = mask.unsqueeze(1).expand(b, attn.num_heads, m, m).reshape(b * attn.num_heads, m, m)
            out, _ = attn(x, x, x, attn_mask=attn_mask)
            messages = torch.bmm(signed, x)
            x = norm(x + out + messages / max(m, 1))
        pooled = (x * weight.unsqueeze(-1)).sum(1) / weight.sum(1, keepdim=True).clamp_min(1e-6)
        logits = self.head(pooled).squeeze(-1)
        challenge = (weight.unsqueeze(2) * weight.unsqueeze(1) * contradiction).amax((1, 2))
        consistency = (signed.sum((1, 2)) / comparable.sum((1, 2)).clamp_min(1)).clamp(-1, 1)
        return {"logits": logits, "relations": relations, "challenge": challenge, "consistency": consistency, "embedding": pooled}
