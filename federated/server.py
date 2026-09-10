"""
Federated Learning Server simulator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from multimodal_cancer_detection.federated.client import FLClient
from multimodal_cancer_detection.federated.strategies import aggregate_fedavg, aggregate_state_dicts


class FLServer:
    """
    Simulates an FL Server coordinating multiple local clients.
    """

    def __init__(
        self,
        global_model: torch.nn.Module,
        clients: List[FLClient],
        strategy: str = "fedavg",
    ):
        self.global_model = global_model
        self.clients = clients
        self.strategy = strategy
        self.global_parameters = [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]

    def run_scoped_rounds(self, num_rounds: int, local_epochs: int = 1,
                          shared_prefixes: tuple[str, ...] = ("deliberation.",)) -> Dict[str, list]:
        """VOIR federation: aggregate declared relation/deliberation state only.

        Source encoders and site calibration/thresholds remain local by design.
        A production secure-aggregation backend can consume the same scoped tensors.
        """
        history = {"round": [], "upload_bytes": [], "shared_keys": []}
        global_shared = {k: v.detach().cpu() for k, v in self.global_model.state_dict().items() if k.startswith(shared_prefixes)}
        for round_idx in range(1, num_rounds + 1):
            results = []
            for client in self.clients:
                client.set_scoped_state(global_shared)
                client.train(local_epochs)
                results.append((client.get_scoped_state(shared_prefixes), len(client.trainer.train_loader.dataset)))
            global_shared = aggregate_state_dicts(results, global_shared.keys())
            state = self.global_model.state_dict()
            state.update({k: v.to(next(self.global_model.parameters()).device) for k, v in global_shared.items()})
            self.global_model.load_state_dict(state)
            history["round"].append(round_idx)
            history["shared_keys"].append(sorted(global_shared))
            history["upload_bytes"].append(sum(v.numel() * v.element_size() for s, _ in results for v in s.values()))
        return history

    def run_rounds(self, num_rounds: int, local_epochs: int = 1) -> Dict[str, list]:
        """Run federated learning rounds."""
        history = {"round": [], "val_auroc": [], "val_loss": []}
        
        for round_idx in range(1, num_rounds + 1):
            print(f"\\n--- FL Round {round_idx}/{num_rounds} ---")
            
            # 1. Distribute global model to all clients
            for client in self.clients:
                client.set_parameters(self.global_parameters)
                
            # 2. Local Training
            client_results = []
            for i, client in enumerate(self.clients):
                res = client.train(epochs=local_epochs)
                client_results.append((client.get_parameters(), res["num_samples"]))
                print(f"Client {client.client_id} finished training. Loss: {res['metrics'].get('train_loss', 0):.4f}")
                
            # 3. Aggregate
            if self.strategy == "fedavg":
                self.global_parameters = aggregate_fedavg(client_results)
            else:
                raise NotImplementedError(f"Strategy {self.strategy} not implemented.")
                
            # Update global model for eval
            state_dict = self.global_model.state_dict()
            keys = list(state_dict.keys())
            for k, v in zip(keys, self.global_parameters):
                state_dict[k] = torch.tensor(v).to(next(self.global_model.parameters()).device)
            self.global_model.load_state_dict(state_dict)
            
            # 4. Global Evaluation (we can eval on first client's validation set assuming it's a global test set, 
            # or do average client val)
            total_loss = 0.0
            total_auroc = 0.0
            total_samples = 0
            
            for client in self.clients:
                res = client.evaluate()
                n = res["num_samples"]
                total_loss += res["metrics"].get("val_loss", 0) * n
                total_auroc += res["metrics"].get("val_auroc", 0) * n
                total_samples += n
                
            avg_loss = total_loss / total_samples
            avg_auroc = total_auroc / total_samples
            
            history["round"].append(round_idx)
            history["val_loss"].append(avg_loss)
            history["val_auroc"].append(avg_auroc)
            
            print(f"Round {round_idx} Global Eval - Loss: {avg_loss:.4f}, AUROC: {avg_auroc:.4f}")
            
        return history
