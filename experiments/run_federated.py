"""
Run a federated learning experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict
import copy

import pandas as pd
import torch
from torch.utils.data import DataLoader

from multimodal_cancer_detection.data.datasets.multimodal_dataset import MultimodalDataset, multimodal_collate_fn
from multimodal_cancer_detection.data.preprocessing.patient_matching import load_metadata
from multimodal_cancer_detection.federated.partition import create_federated_partitions
from multimodal_cancer_detection.federated.client import FLClient
from multimodal_cancer_detection.federated.server import FLServer
from multimodal_cancer_detection.models.multimodal_model import MultimodalModel
from multimodal_cancer_detection.training.trainer import Trainer
from multimodal_cancer_detection.utils.config import load_config, resolve_paths
from multimodal_cancer_detection.utils.seed import get_device, set_seed
from multimodal_cancer_detection.data.preprocessing.pathology import get_pathology_transforms
from multimodal_cancer_detection.data.preprocessing.radiology import get_radiology_transforms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/federated.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.smoke_test:
        overrides["experiment"] = {"smoke_test": True}

    config = load_config(["configs/base.yaml", args.config], overrides=overrides)
    config = resolve_paths(config, Path.cwd())

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    device = get_device(config.get("device", {}).get("use_cuda", True))

    fed_cfg = config.get("federated", {})
    num_clients = fed_cfg.get("num_clients", 3)
    num_rounds = fed_cfg.get("num_rounds", 10)
    local_epochs = fed_cfg.get("local_epochs", 1)

    dataset_cfg = config.get("dataset", {})
    metadata_csv = dataset_cfg.get("metadata_csv")
    df = load_metadata(metadata_csv, dataset_cfg.get("patient_id_col"), dataset_cfg.get("label_col"))

    if args.smoke_test:
        df = df.sample(min(32, len(df)), random_state=seed).reset_index(drop=True)
        num_rounds = 2
        local_epochs = 1

    # Simple split: 80% train to partition, 20% global test
    test_df = df.sample(frac=0.2, random_state=seed)
    train_df = df.drop(test_df.index).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    partitions = create_federated_partitions(
        train_df, 
        num_clients=num_clients, 
        partition_type=fed_cfg.get("partition_type", "iid"),
        label_col=dataset_cfg.get("label_col"),
        alpha=fed_cfg.get("alpha", 0.5),
        seed=seed
    )

    # Global model
    global_model = MultimodalModel(config=config, genomics_input_dim=None).to(device)

    clients = []
    
    # Shared transforms (simplification for FL script)
    rad_transform = get_radiology_transforms(config.get("radiology", {}), True)
    path_transform = get_pathology_transforms(config.get("pathology", {}), True)
    eval_rad = get_radiology_transforms(config.get("radiology", {}), False)
    eval_path = get_pathology_transforms(config.get("pathology", {}), False)

    for i in range(num_clients):
        client_df = partitions[i]
        
        # Simple client validation set (10% of client data)
        val_df = client_df.sample(frac=0.1, random_state=seed)
        train_df_c = client_df.drop(val_df.index).reset_index(drop=True)

        ds_kwargs = {
            "patient_id_col": dataset_cfg.get("patient_id_col"),
            "label_col": dataset_cfg.get("label_col"),
            "use_radiology": True,
            "use_pathology": True,
            "use_genomics": False,  # Simplified for federated script
            "radiology_path_col": dataset_cfg.get("radiology_path_col"),
            "pathology_path_col": dataset_cfg.get("pathology_path_col"),
            "patches_dir": config.get("pathology", {}).get("patches_dir"),
        }

        train_ds = MultimodalDataset(train_df_c, rad_transform, path_transform, **ds_kwargs)
        val_ds = MultimodalDataset(val_df, eval_rad, eval_path, **ds_kwargs)
        
        train_loader = DataLoader(train_ds, batch_size=config.get("training", {}).get("batch_size", 8), shuffle=True, collate_fn=multimodal_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=config.get("training", {}).get("batch_size", 8), shuffle=False, collate_fn=multimodal_collate_fn)

        client_model = copy.deepcopy(global_model)
        trainer = Trainer(
            model=client_model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        
        clients.append(FLClient(i, trainer))

    server = FLServer(global_model, clients, strategy=fed_cfg.get("strategy", "fedavg"))
    history = server.run_rounds(num_rounds, local_epochs)
    
    print("\\nFederated Learning Complete.")
    print(f"Final Global Val AUROC: {history['val_auroc'][-1]:.4f}")

if __name__ == "__main__":
    main()
