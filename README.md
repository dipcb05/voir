# VOIR: Federated Multimodal Cancer Diagnosis

This repository implements the VOIR research pipeline for radiology, pathology,
and molecular evidence. It requires approved, real site-local clinical data.
Patient data, model outputs, and credentials are excluded from version control.

## Required data

Import data into `multimodal_cancer_detection/datasets/` before training:

```powershell
python -m multimodal_cancer_detection.data.ingest `
  --metadata C:\secure\metadata.csv `
  --genomics C:\secure\genomics.csv `
  --radiology-dir C:\secure\radiology `
  --pathology-dir C:\secure\pathology
```

See [datasets/README.md](datasets/README.md) for the required manifest schema.
The import command validates the manifest before copying files and refuses to
overwrite an existing dataset.

## Training

Install dependencies, then run a training entry point from the repository root:

```powershell
pip install -r multimodal_cancer_detection/requirements.txt
python -m multimodal_cancer_detection.experiments.train_multimodal --config multimodal_cancer_detection/configs/multimodal.yaml
```

The default configuration requires `datasets/metadata.csv` and
`datasets/genomics.csv`. A run fails if the real files are absent.

## Federated deployment

Federation is deployed at independent data-owning sites. The coordinator uses
`federated.server.start_production_server`; each site independently runs
`federated.client.start_production_client` with its local approved data and
mTLS credentials. This repository has no local federated run mode.

## Safety boundary

This is research software, not a clinical device. Do not use it for patient
care without validation, governance approval, security review, and appropriate
regulatory authorization.
