"""
Federated Learning Client implementation (using Flower optionally, or standalone simulation).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from multimodal_cancer_detection.training.trainer import Trainer


class FLClient:
    """
    Simulation client for Federated Learning.
    Wraps a Trainer instance.
    """

    def __init__(
        self,
        client_id: int,
        trainer: Trainer,
    ):
        self.client_id = client_id
        self.trainer = trainer
        self.model = trainer.model
        self.device = trainer.device

    def get_parameters(self) -> List[np.ndarray]:
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy arrays."""
        state_dict = self.model.state_dict()
        keys = list(state_dict.keys())
        for k, v in zip(keys, parameters):
            state_dict[k] = torch.tensor(v).to(self.device)
        self.model.load_state_dict(state_dict)

    def train(self, epochs: int) -> Dict[str, Any]:
        """Train locally for E epochs."""
        self.trainer.epochs = self.trainer.start_epoch + epochs
        
        for epoch in range(self.trainer.start_epoch, self.trainer.epochs):
            metrics = self.trainer.train_epoch(epoch)
            
        self.trainer.start_epoch += epochs
        
        # Return last epoch metrics and num samples
        return {
            "metrics": metrics,
            "num_samples": len(self.trainer.train_loader.dataset)
        }

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate locally on validation set."""
        metrics, _ = self.trainer.evaluate(self.trainer.val_loader, prefix="val")
        return {
            "metrics": metrics,
            "num_samples": len(self.trainer.val_loader.dataset)
        }
