"""FastAPI dependencies: session storage and shared singletons.

The session store is a simple in-memory, single-process dict. This is
an intentional scope decision for a portfolio project (Section 25 lists
"persistent user sessions" as a future extension) -- swapping this for
Redis/Postgres later means only touching this module, since routes only
ever go through `SessionStore`, never touch a dict directly.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.core.exceptions import DatasetNotFoundError
from app.core.logging import get_logger
from app.llm.base import LLMProvider
from app.llm.provider import get_llm_provider
from app.models.schemas import DatasetMetadata

logger = get_logger(__name__)


@dataclass
class SessionRecord:
    session_id: str
    dataset_id: str
    filename: str
    df: pd.DataFrame
    metadata: DatasetMetadata
    created_at: datetime = field(default_factory=datetime.utcnow)


class SessionStore:
    """Thread-safe in-memory store mapping session_id -> SessionRecord."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def create_session(self, df: pd.DataFrame, metadata: DatasetMetadata, filename: str) -> SessionRecord:
        session_id = str(uuid.uuid4())
        record = SessionRecord(
            session_id=session_id,
            dataset_id=metadata.dataset_id,
            filename=filename,
            df=df,
            metadata=metadata,
        )
        with self._lock:
            self._sessions[session_id] = record
        logger.info("Created session %s for dataset %s (%s)", session_id, metadata.dataset_id, filename)
        return record

    def get_session(self, session_id: str) -> SessionRecord:
        with self._lock:
            record = self._sessions.get(session_id)
        if record is None:
            raise DatasetNotFoundError(f"No dataset found for session_id '{session_id}'. Upload a CSV first.")
        return record

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


# Module-level singletons -- fine for a single-process FastAPI app of
# this scope; see SessionStore docstring re: future persistence.
_session_store = SessionStore()
_llm_provider: LLMProvider | None = None
_llm_provider_lock = threading.Lock()


def get_session_store() -> SessionStore:
    return _session_store


def get_llm_provider_dependency(settings: Settings = Depends(get_settings)) -> LLMProvider:
    """Return a process-wide LLMProvider instance, built lazily on first use
    (so the app can still start, e.g. for local dataset-only testing,
    even before LLM_API_KEY is configured).
    """
    global _llm_provider
    with _llm_provider_lock:
        if _llm_provider is None:
            _llm_provider = get_llm_provider(
                provider_name=settings.llm_provider,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                base_url=settings.llm_base_url or None,
            )
    return _llm_provider
