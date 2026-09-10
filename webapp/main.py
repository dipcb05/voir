from pathlib import Path
import io, re
import tempfile
import torch
from PIL import Image
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address
from .db import Base, engine, get_db
from .models import AuditEvent, Dataset, Job, ModelVersion, Prediction, User
from .security import csrf, decode_token, hash_password, issue_token, require_csrf, verify_password
from .storage import put_file, get_file
from .tasks import train_model, validate_dataset
from .validation import validate_zip

app=FastAPI(title="VOIR Research Portal", docs_url=None, redoc_url=None)
limiter=Limiter(key_func=get_remote_address); app.state.limiter=limiter
root=Path(__file__).parent
app.mount("/static", StaticFiles(directory=root/"static"), name="static")
templates=Jinja2Templates(directory=root/"templates")

@app.on_event("startup")
def startup(): Base.metadata.create_all(engine)
def audit(db, actor, action, target=None, details=None): db.add(AuditEvent(actor_id=actor, action=action, target_id=target, details=details or {})); db.commit()
def current(request: Request, db: Session=Depends(get_db)):
    token=request.cookies.get("access");
    if not token: raise HTTPException(401,"Login required")
    user=db.get(User,decode_token(token)["sub"])
    if not user: raise HTTPException(401,"Unknown user")
    return user
def owned(obj, user):
    if obj is None or (obj.owner_id != user.id and not user.is_admin): raise HTTPException(404,"Not found")
    return obj

@app.get("/", response_class=HTMLResponse)
def home(request: Request): return templates.TemplateResponse("index.html", {"request":request})
@app.post("/auth/register")
@limiter.limit("5/minute")
def register(request:Request, email:str=Form(), password:str=Form(), db:Session=Depends(get_db)):
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+",email) or len(password)<12: raise HTTPException(422,"Use a valid email and a 12+ character password")
    if db.query(User).filter_by(email=email.lower()).first(): raise HTTPException(409,"Account already exists")
    u=User(email=email.lower(),password_hash=hash_password(password)); db.add(u); db.commit(); audit(db,u.id,"REGISTER",u.id)
    return {"ok":True}
@app.post("/auth/login")
@limiter.limit("5/minute")
def login(request:Request, email:str=Form(), password:str=Form(), db:Session=Depends(get_db)):
    u=db.query(User).filter_by(email=email.lower()).first()
    if not u or not verify_password(password,u.password_hash): raise HTTPException(401,"Invalid credentials")
    csrf_token=csrf(); response=JSONResponse({"ok":True,"csrf":csrf_token,"admin":u.is_admin}); response.set_cookie("access",issue_token(u.id,u.is_admin),httponly=True,secure=True,samesite="strict",max_age=1200); response.set_cookie("csrf",csrf_token,httponly=False,secure=True,samesite="strict",max_age=1200); audit(db,u.id,"LOGIN",u.id); return response
@app.post("/auth/logout")
def logout(request:Request,user:User=Depends(current)):
    require_csrf(request); r=JSONResponse({"ok":True}); r.delete_cookie("access"); r.delete_cookie("csrf"); return r

@app.get("/datasets")
def datasets(user:User=Depends(current),db:Session=Depends(get_db)): return [{"id":x.id,"name":x.name,"status":x.status,"manifest":x.manifest} for x in db.query(Dataset).filter_by(owner_id=user.id)]
@app.post("/datasets")
@limiter.limit("3/hour")
async def upload_dataset(request:Request, archive:UploadFile=File(), user:User=Depends(current), db:Session=Depends(get_db)):
    require_csrf(request)
    if archive.content_type not in {"application/zip","application/x-zip-compressed"}: raise HTTPException(415,"Upload a ZIP dataset")
    payload=await archive.read(); manifest=validate_zip(payload); dataset=Dataset(owner_id=user.id,name=Path(archive.filename or "dataset.zip").stem,object_key=f"users/{user.id}/datasets/",manifest=manifest)
    dataset.object_key += dataset.id+".zip"; put_file(dataset.object_key,io.BytesIO(payload),archive.content_type); db.add(dataset); db.commit(); validate_dataset.delay(dataset.id); audit(db,user.id,"DATASET_UPLOAD",dataset.id,manifest); return {"dataset_id":dataset.id,"status":dataset.status}

@app.post("/jobs")
def create_job(request:Request,dataset_id:str=Form(),model_mode:str=Form(),user:User=Depends(current),db:Session=Depends(get_db)):
    require_csrf(request); dataset=owned(db.get(Dataset,dataset_id),user)
    if dataset.status!="READY" or model_mode not in {"radiology","multimodal"}: raise HTTPException(422,"Dataset is not ready or model mode is invalid")
    job=Job(owner_id=user.id,dataset_id=dataset.id,model_mode=model_mode,config={}); db.add(job); db.commit(); train_model.delay(job.id); audit(db,user.id,"TRAIN_REQUEST",job.id); return {"job_id":job.id,"status":job.status}
@app.get("/jobs/{job_id}")
def job_status(job_id:str,user:User=Depends(current),db:Session=Depends(get_db)):
    x=owned(db.get(Job,job_id),user); return {"id":x.id,"status":x.status,"logs":x.logs,"cancel_requested":x.cancel_requested}
@app.post("/jobs/{job_id}/cancel")
def cancel(request:Request,job_id:str,user:User=Depends(current),db:Session=Depends(get_db)):
    require_csrf(request); x=owned(db.get(Job,job_id),user); x.cancel_requested=True; db.commit(); return {"ok":True}

@app.get("/models")
def models(user:User=Depends(current),db:Session=Depends(get_db)):
    rows=db.query(ModelVersion).filter((ModelVersion.owner_id==user.id)|(ModelVersion.published==True)).all(); return [{"id":x.id,"mode":x.mode,"published":x.published,"metrics":x.metrics} for x in rows]
@app.post("/admin/models/{model_id}/publish")
def publish(request:Request,model_id:str,user:User=Depends(current),db:Session=Depends(get_db)):
    require_csrf(request)
    if not user.is_admin: raise HTTPException(403,"Admin required")
    m=db.get(ModelVersion,model_id)
    if not m: raise HTTPException(404,"Model not found")
    m.published=True; db.commit(); audit(db,user.id,"PUBLISH_MODEL",m.id); return {"ok":True}

@app.post("/predictions")
@limiter.limit("20/hour")
async def predict(request:Request, model_id:str=Form(), image:UploadFile=File(), user:User=Depends(current), db:Session=Depends(get_db)):
    """Inference is intentionally restricted to completed radiology checkpoints."""
    require_csrf(request); model=db.get(ModelVersion,model_id)
    if not model or (model.owner_id != user.id and not model.published): raise HTTPException(404,"Model not found")
    if model.mode != "radiology": raise HTTPException(422,"Single-image inference requires a radiology model")
    if image.content_type not in {"image/jpeg","image/png","image/tiff"}: raise HTTPException(415,"Unsupported image type")
    # Worker-created artifacts are required; no heuristic or untrained fallback is permitted.
    if not model.checkpoint_key: raise HTTPException(409,"Model artifact is unavailable")
    payload=await image.read()
    try: uploaded=Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc: raise HTTPException(422,"Image content is invalid") from exc
    from multimodal_cancer_detection.models.radiology_encoder import RadiologyEncoder
    from multimodal_cancer_detection.data.preprocessing.radiology import get_radiology_transforms
    artifact=io.BytesIO(); get_file(model.checkpoint_key,artifact); artifact.seek(0)
    try:
        state=torch.load(artifact,map_location="cpu",weights_only=False)["model_state_dict"]
        network=RadiologyEncoder(pretrained=False,num_classes=2); network.load_state_dict(state); network.eval()
        tensor=get_radiology_transforms({},False)(uploaded).unsqueeze(0)
        with torch.no_grad(): probability=float(torch.sigmoid(network(tensor)["logits"])[0])
    except Exception as exc: raise HTTPException(409,"Checkpoint is incompatible with radiology inference") from exc
    result={"radiology_prediction":{"positive_probability":probability,"predicted_class":int(probability>=.5)},"model_id":model.id,"model_mode":"radiology","voir":{"action":"REQUEST_EVIDENCE","reason":"A single image is insufficient for full multimodal VOIR deliberation","missing_modalities":["pathology","genomics"]},"disclaimer":"Research use only; not for clinical decision-making."}
    prediction=Prediction(owner_id=user.id,model_id=model.id,result=result); db.add(prediction); db.commit(); audit(db,user.id,"PREDICTION_REQUEST",prediction.id); return result
