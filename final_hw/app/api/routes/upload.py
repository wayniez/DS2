"""Dataset upload endpoint (Section 5).

Validates and profiles an uploaded CSV, then creates a session that
subsequent /analysis requests reference by session_id.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.dependencies import SessionStore, get_session_store
from app.core.config import Settings, get_settings
from app.core.exceptions import DatasetError
from app.core.logging import get_logger
from app.models.schemas import UploadResponse
from app.tools.dataset import get_dataset_metadata, load_csv

logger = get_logger(__name__)

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    session_store: SessionStore = Depends(get_session_store),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    file_bytes = await file.read()
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the maximum upload size of {settings.max_upload_size_mb} MB.",
        )

    try:
        df = load_csv(file_bytes, max_rows=settings.max_dataset_rows)
        dataset_id = str(uuid.uuid4())
        metadata = get_dataset_metadata(df, dataset_id=dataset_id, filename=file.filename)
    except DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record = session_store.create_session(df=df, metadata=metadata, filename=file.filename)
    logger.info("Uploaded dataset '%s' as session %s", file.filename, record.session_id)

    return UploadResponse(session_id=record.session_id, dataset_id=record.dataset_id, metadata=metadata)
