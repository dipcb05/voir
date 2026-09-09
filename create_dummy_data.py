import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import json

def create_dummy_data(base_dir="dummy_data", num_patients=50):
    base_dir = Path(base_dir)
    base_dir.mkdir(exist_ok=True)
    
    radiology_dir = base_dir / "radiology"
    radiology_dir.mkdir(exist_ok=True)
    
    pathology_dir = base_dir / "pathology"
    pathology_dir.mkdir(exist_ok=True)
    
    # 1. Metadata
    metadata = []
    
    for i in range(num_patients):
        pid = f"PAT_{i:03d}"
        label = np.random.randint(0, 2)
        
        # Radiology Image
        rad_path = radiology_dir / f"{pid}_rad.png"
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(rad_path)
        
        # Pathology Patches (Directory)
        pat_dir = pathology_dir / pid
        pat_dir.mkdir(exist_ok=True)
        num_patches = np.random.randint(1, 10)
        for j in range(num_patches):
            patch_path = pat_dir / f"patch_{j}.png"
            patch_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            patch_img.save(patch_path)
            
        metadata.append({
            "patient_id": pid,
            "label": label,
            "radiology_path": str(rad_path),
            "pathology_path": str(pat_dir)
        })
        
    df = pd.DataFrame(metadata)
    df.to_csv(base_dir / "metadata.csv", index=False)
    
    # 2. Genomics Expression Matrix
    # 50 patients, 100 genes
    gene_names = [f"GENE_{i}" for i in range(100)]
    pids = [f"PAT_{i:03d}" for i in range(num_patients)]
    expr_data = np.random.randn(num_patients, 100)
    
    expr_df = pd.DataFrame(expr_data, index=pids, columns=gene_names)
    expr_df.index.name = "patient_id"
    expr_df.to_csv(base_dir / "genomics.csv")
    
    print(f"Created dummy data at {base_dir}")

if __name__ == "__main__":
    create_dummy_data()
