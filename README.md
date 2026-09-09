# Integrated Federated Multimodal AI for Early Cancer Detection

This repository contains a reproducible, research-grade PyTorch codebase for multimodal cancer detection using:
- **Radiology** (Images, e.g., MRI/CT)
- **Pathology** (Histopathology Whole Slide Images via Patch-based MIL)
- **Genomics** (Gene Expression Profiles)

It supports unimodal training, bimodal/trimodal fusion (concatenation or attention-based), and federated learning simulations. The codebase is designed to be highly configurable via YAML and is fully compatible with execution in Google Colab.

## Repository Structure

```
multimodal_cancer_detection/
├── configs/                 # YAML configurations for all experiments
│   ├── base.yaml            # Shared defaults
│   ├── radiology.yaml       # Radiology-specific settings
│   ├── pathology.yaml       # Pathology-specific settings
│   ├── genomics.yaml        # Genomics-specific settings
│   ├── multimodal.yaml      # Fusion and multimodal settings
│   └── federated.yaml       # FL settings
├── data/                    # Datasets and preprocessing
│   ├── datasets/            # PyTorch Dataset classes
│   └── preprocessing/       # Transforms, imputation, patient matching
├── evaluation/              # Metrics, plots, stats, explainability
├── experiments/             # Experiment execution scripts
│   ├── train_radiology.py   # Train unimodal radiology
│   ├── train_pathology.py   # Train unimodal pathology
│   ├── train_genomics.py    # Train unimodal genomics
│   ├── train_multimodal.py  # Train multimodal fusion model
│   ├── run_ablation.py      # Run full ablation study
│   └── run_federated.py     # Run federated learning simulation
├── federated/               # Federated learning logic (Server, Client, Strategy)
├── models/                  # PyTorch model architectures
│   ├── radiology_encoder.py # CNN backbones (e.g., DenseNet)
│   ├── pathology_encoder.py # Attention MIL patch encoder
│   ├── genomics_encoder.py  # MLP encoder for expression data
│   └── multimodal_model.py  # End-to-end fusion model
├── training/                # Generic Trainer, Callbacks, Losses
├── utils/                   # Config parsing, logging, checkpointing
└── requirements.txt         # Project dependencies
```

## Quickstart in Google Colab

The easiest way to run the experiments is via Google Colab.
You can open `notebooks/run_pipeline.ipynb` in Colab and execute the cells sequentially.

Alternatively, to run from the command line:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Unimodal Radiology
python -m multimodal_cancer_detection.experiments.train_radiology --config configs/radiology.yaml

# Run Multimodal Fusion
python -m multimodal_cancer_detection.experiments.train_multimodal --config configs/multimodal.yaml

# Run Federated Learning Simulation
python -m multimodal_cancer_detection.experiments.run_federated --config configs/federated.yaml

# Run Full Ablation Study
python -m multimodal_cancer_detection.experiments.run_ablation --config configs/base.yaml
```

*Note: For quick debugging without large datasets, append the `--smoke-test` flag to any run script.*

## Configuration

All hyperparameters, column names, paths, and training settings are fully decoupled from the code and specified in YAML files inside the `configs/` directory.

- The `dataset.metadata_csv` dictates which patients to load.
- Column mappings (e.g., `dataset.radiology_path_col`) tell the dataset exactly where to find images/files.
- Override defaults dynamically if needed.

## Evaluation Outputs

By default, all experiment scripts output to the `outputs/` directory:
- `outputs/metrics/`: JSON and CSV files of comprehensive metrics (AUROC, AUPRC, F1, etc.).
- `outputs/figures/`: Publication-ready plots (ROC curves, Confusion Matrices, Ablation Results) in PNG and PDF formats.
- `outputs/models/`: Saved model checkpoints and scaler states.
- `outputs/logs/`: Detailed training logs.
- `outputs/splits/`: Stratified train/val/test splits used during the experiment for reproducibility.
