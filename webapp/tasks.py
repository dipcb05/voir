"""Celery jobs. Workers own the only long-running execution path."""
import io, os, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path
import pandas as pd
from celery import Celery
from .config import settings
from .db import SessionLocal
from .models import Dataset, Job, ModelVersion
from .storage import get_file, put_path

celery = Celery("voir", broker=settings.redis_url, backend=settings.redis_url)

@celery.task(bind=True)
def validate_dataset(self, dataset_id: str):
    db=SessionLocal()
    try:
        dataset=db.get(Dataset,dataset_id)
        if not dataset: return
        dataset.status="READY"; db.commit()
    finally: db.close()

@celery.task(bind=True)
def train_model(self, job_id: str):
    """Train in an isolated worker directory and publish only an owned artifact."""
    db=SessionLocal()
    try:
        job=db.get(Job,job_id)
        if not job or job.cancel_requested: return
        job.status="RUNNING"; job.logs += "Worker accepted training job.\n"; db.commit()
        dataset=db.get(Dataset,job.dataset_id)
        if not dataset or dataset.owner_id != job.owner_id: raise RuntimeError("Dataset ownership contract failed")
        with tempfile.TemporaryDirectory(prefix=f"voir-{job.id}-") as tmp:
            work=Path(tmp); archive=io.BytesIO(); get_file(dataset.object_key,archive); archive.seek(0)
            with zipfile.ZipFile(archive) as z: z.extractall(work/"dataset")
            data=work/"dataset"; manifest=pd.read_csv(data/"metadata.csv")
            manifest["radiology_path"]=manifest["radiology_path"].map(lambda x:str(data/"radiology"/Path(str(x)).name))
            manifest["pathology_path"]=manifest["patient_id"].map(lambda x:str(data/"pathology"/str(x)))
            manifest.to_csv(data/"metadata.csv",index=False)
            output=work/"outputs"; output.mkdir()
            config=work/"job.yaml"
            mode_config="radiology.yaml" if job.model_mode=="radiology" else "multimodal.yaml"
            config.write_text("\n".join(["dataset:",f"  metadata_csv: '{data / 'metadata.csv'}'", "genomics:",f"  expression_matrix: '{data / 'genomics.csv'}'", "pathology:",f"  patches_dir: '{data / 'pathology'}'", "experiment:",f"  name: 'job_{job.id}'", "checkpoint:",f"  save_dir: '{output / 'models'}'", "evaluation:",f"  metrics_dir: '{output / 'metrics'}'",f"  figures_dir: '{output / 'figures'}'",f"  predictions_dir: '{output / 'predictions'}'", "logging:",f"  log_dir: '{output / 'logs'}'", "training:", "  epochs: 100"]))
            package_root=Path(__file__).resolve().parents[1]
            command=[sys.executable,"-m",f"multimodal_cancer_detection.experiments.train_{job.model_mode}","--config",str(config)]
            env=os.environ.copy(); env["PYTHONPATH"]=str(package_root.parent)
            result=subprocess.run(command,cwd=package_root,capture_output=True,text=True,timeout=job.config.get("timeout_seconds", 86400),env=env)
            job.logs += result.stdout[-10000:] + result.stderr[-10000:]
            if result.returncode != 0 or job.cancel_requested: raise RuntimeError("Training failed or was cancelled")
            checkpoints=list((output/"models").glob("*_best.pt"))
            if not checkpoints: raise RuntimeError("Training completed without a best checkpoint")
            checkpoint=checkpoints[0]; key=f"users/{job.owner_id}/models/{job.id}/{checkpoint.name}"; put_path(key,checkpoint)
            model=ModelVersion(owner_id=job.owner_id,job_id=job.id,mode=job.model_mode,checkpoint_key=key,config={"config_file":config.read_text()},metrics={})
            db.add(model); db.flush(); job.status="COMPLETED"; job.logs += f"Model {model.id} registered.\n"; db.commit()
    except Exception as exc:
        if 'job' in locals() and job:
            job.status="CANCELLED" if job.cancel_requested else "FAILED"; job.logs += f"\nERROR: {exc}\n"; db.commit()
        raise
    finally: db.close()
