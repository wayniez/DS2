"""FastAPI application entrypoint.

Run with: uvicorn app.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import analysis, health, upload
from app.core.exceptions import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AI Data Analyst",
    description=(
        "An autonomous AI-powered data analysis agent. The LLM plans and interprets; "
        "deterministic Python/SQL/ML tools do the actual computation."
    ),
    version="0.1.0",
)

# Permissive CORS for local development (Streamlit frontend running on a
# different port). Tighten this for any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(analysis.router)


@app.exception_handler(AppError)
async def app_error_handler(request, exc: AppError):  # noqa: ANN001
    """Fallback handler: any AppError subclass not already translated into
    an HTTPException by a route gets turned into a clean 400 response
    instead of a raw 500 traceback (Section 13).
    """
    from fastapi.responses import JSONResponse

    logger.warning("Unhandled AppError reached the top-level handler: %s", exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})
