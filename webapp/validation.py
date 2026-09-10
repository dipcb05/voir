import csv, io, zipfile
from fastapi import HTTPException
from .config import settings

REQUIRED = {"metadata.csv", "genomics.csv"}
MANIFEST_COLUMNS = {"patient_id", "label", "index_time", "radiology_path", "pathology_path"}
def validate_zip(payload: bytes) -> dict:
    if len(payload) > settings.max_upload_bytes: raise HTTPException(413, "Upload exceeds size limit")
    try: archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc: raise HTTPException(422, "Dataset must be a valid ZIP archive") from exc
    infos = archive.infolist()
    if len(infos) > settings.max_zip_members or sum(i.file_size for i in infos) > settings.max_uncompressed_bytes: raise HTTPException(413, "Archive expansion limit exceeded")
    names = {i.filename for i in infos}
    if any(i.filename.startswith("/") or ".." in i.filename.split("/") for i in infos): raise HTTPException(422, "Unsafe archive path")
    if not REQUIRED <= names or not any(n.startswith("radiology/") for n in names) or not any(n.startswith("pathology/") for n in names): raise HTTPException(422, "ZIP requires metadata.csv, genomics.csv, radiology/, pathology/")
    try: rows = csv.DictReader(io.TextIOWrapper(archive.open("metadata.csv"), encoding="utf-8")); columns=set(rows.fieldnames or []); count=sum(1 for _ in rows)
    except Exception as exc: raise HTTPException(422, "metadata.csv is unreadable") from exc
    if not MANIFEST_COLUMNS <= columns or count < 2: raise HTTPException(422, "Invalid metadata.csv manifest")
    return {"rows":count, "members":len(infos), "columns":sorted(columns)}
