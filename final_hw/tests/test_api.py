"""Integration tests for the FastAPI layer, using `TestClient`.

The LLM provider dependency is overridden with a scripted mock so these
tests never require a real LLM_API_KEY or network access (Section 18).
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_llm_provider_dependency
from app.llm.schemas import LLMResponse
from app.main import app
from tests.test_agent import ScriptedLLMProvider


@pytest.fixture
def client(churn_df: pd.DataFrame) -> TestClient:
    def _override_llm():
        return ScriptedLLMProvider(script=[LLMResponse(text="Grounded final answer from mocked LLM.")])

    app.dependency_overrides[get_llm_provider_dependency] = _override_llm
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def csv_bytes(churn_df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    churn_df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


class TestHealth:
    def test_health_endpoint(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestUpload:
    def test_upload_valid_csv(self, client: TestClient, csv_bytes: bytes) -> None:
        response = client.post(
            "/upload", files={"file": ("churn.csv", csv_bytes, "text/csv")}
        )
        assert response.status_code == 200
        body = response.json()
        assert "session_id" in body
        assert "dataset_id" in body
        assert body["metadata"]["n_rows"] > 0

    def test_upload_rejects_non_csv_extension(self, client: TestClient) -> None:
        response = client.post(
            "/upload", files={"file": ("data.txt", b"not,a,csv", "text/plain")}
        )
        assert response.status_code == 400

    def test_upload_rejects_empty_csv(self, client: TestClient) -> None:
        response = client.post(
            "/upload", files={"file": ("empty.csv", b"", "text/csv")}
        )
        assert response.status_code == 400

    def test_upload_rejects_malformed_csv(self, client: TestClient) -> None:
        # A file with no columns/no valid CSV structure.
        response = client.post(
            "/upload", files={"file": ("bad.csv", b"\x00\x01\x02\x03", "text/csv")}
        )
        assert response.status_code == 400


class TestAnalysis:
    def test_analysis_end_to_end(self, client: TestClient, csv_bytes: bytes) -> None:
        upload_response = client.post(
            "/upload", files={"file": ("churn.csv", csv_bytes, "text/csv")}
        )
        session_id = upload_response.json()["session_id"]
        dataset_id = upload_response.json()["dataset_id"]

        analysis_response = client.post(
            "/analysis",
            json={"session_id": session_id, "dataset_id": dataset_id, "question": "What is this dataset about?"},
        )
        assert analysis_response.status_code == 200
        report = analysis_response.json()["report"]
        assert report["answer_text"] == "Grounded final answer from mocked LLM."
        assert report["session_id"] == session_id

    def test_analysis_unknown_session_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/analysis",
            json={"session_id": "does-not-exist", "dataset_id": "does-not-exist", "question": "Hello?"},
        )
        assert response.status_code == 404

    def test_analysis_empty_question_returns_400(self, client: TestClient, csv_bytes: bytes) -> None:
        upload_response = client.post(
            "/upload", files={"file": ("churn.csv", csv_bytes, "text/csv")}
        )
        session_id = upload_response.json()["session_id"]
        dataset_id = upload_response.json()["dataset_id"]

        response = client.post(
            "/analysis",
            json={"session_id": session_id, "dataset_id": dataset_id, "question": "   "},
        )
        assert response.status_code == 400

    def test_analysis_mismatched_dataset_id_returns_404(self, client: TestClient, csv_bytes: bytes) -> None:
        upload_response = client.post(
            "/upload", files={"file": ("churn.csv", csv_bytes, "text/csv")}
        )
        session_id = upload_response.json()["session_id"]

        response = client.post(
            "/analysis",
            json={"session_id": session_id, "dataset_id": "wrong-dataset-id", "question": "Hello?"},
        )
        assert response.status_code == 404
