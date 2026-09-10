"""Development-fitted calibration, conformal sets and VOIR action policy."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import torch


class SelectivePolicy:
    def __init__(self, acceptance_threshold: float = .8, challenge_threshold: float = .2, alpha: float = .1):
        self.acceptance_threshold, self.challenge_threshold, self.alpha = acceptance_threshold, challenge_threshold, alpha
        self.temperature, self.conformal_quantile = 1.0, None

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Fits conservative temperature and split-conformal quantile on development data only."""
        logits, labels = np.asarray(logits, float), np.asarray(labels, int)
        candidates = np.linspace(.25, 4., 100)
        losses = [np.mean(np.logaddexp(0, x / t) - labels * (x / t)) for t in candidates]
        self.temperature = float(candidates[int(np.argmin(losses))])
        probs = 1 / (1 + np.exp(-logits / self.temperature))
        scores = np.where(labels == 1, 1 - probs, probs)
        self.conformal_quantile = float(np.quantile(scores, min(1., np.ceil((len(scores)+1)*(1-self.alpha))/len(scores)), method="higher"))

    def decide(self, logits: torch.Tensor, challenge: torch.Tensor, coverage: torch.Tensor,
               gap_sensitivity: Optional[torch.Tensor] = None, requestable: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        temp = max(self.temperature, 1e-6)
        probability = torch.sigmoid(logits / temp)
        q = self.conformal_quantile if self.conformal_quantile is not None else 1 - self.alpha
        pred_set_size = ((probability >= q).long() + ((1 - probability) >= q).long())
        gap = torch.zeros_like(probability) if gap_sensitivity is None else gap_sensitivity
        readiness = (probability - .5).abs() * 2 * coverage * (1 - challenge) * (1 - gap)
        accept = (readiness >= self.acceptance_threshold) & (challenge <= self.challenge_threshold) & (pred_set_size <= 1)
        action = torch.full_like(pred_set_size, 0)  # 0 defer, 1 accept, 2 request
        action[accept] = 1
        if requestable is not None:
            action[(~accept) & requestable.bool()] = 2
        return {"probability": probability, "readiness": readiness, "prediction_set_size": pred_set_size, "action": action}
