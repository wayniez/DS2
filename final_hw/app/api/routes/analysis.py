"""Analysis endpoint (Section 7).

Runs the AI Data Analyst agent loop for one question against a
previously uploaded dataset, identified by `session_id`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.agent.agent import AIDataAnalystAgent
from app.api.dependencies import SessionStore, get_llm_provider_dependency, get_session_store
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError, DatasetNotFoundError
from app.core.logging import get_logger
from app.llm.base import LLMProvider
from app.models.schemas import AnalysisRequest, AnalysisResponse

logger = get_logger(__name__)

router = APIRouter(tags=["analysis"])


@router.post("/analysis", response_model=AnalysisResponse)
def run_analysis(
    request: AnalysisRequest,
    session_store: SessionStore = Depends(get_session_store),
    llm_provider: LLMProvider = Depends(get_llm_provider_dependency),
    settings: Settings = Depends(get_settings),
) -> AnalysisResponse:
    try:
        record = session_store.get_session(request.session_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if record.dataset_id != request.dataset_id:
        raise HTTPException(
            status_code=404,
            detail="dataset_id does not match the dataset associated with this session_id.",
        )

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="`question` must not be empty.")

    agent = AIDataAnalystAgent(llm_provider=llm_provider, max_steps=settings.max_agent_steps)

    try:
        report = agent.run(
            session_id=record.session_id,
            dataset_id=record.dataset_id,
            df=record.df,
            dataset_metadata=record.metadata,
            question=request.question,
        )
    except AppError as exc:
        logger.exception("Agent run failed for session %s", request.session_id)
        raise HTTPException(status_code=502, detail=f"Analysis failed: {exc}") from exc

    return AnalysisResponse(report=report)
