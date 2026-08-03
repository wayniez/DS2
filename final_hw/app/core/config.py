"""Application configuration.

All configuration is loaded from environment variables (optionally via a
.env file). Nothing sensitive (API keys, secrets) should ever be
hard-coded here.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings.

    Values are read from environment variables (case-insensitive) and
    fall back to the defaults below when not set. See `.env.example`
    for the full list of supported variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider -------------------------------------------------
    llm_provider: str = Field(default="anthropic")
    llm_api_key: str = Field(default="")
    llm_model: str = Field(default="claude-sonnet-4-6")
    llm_max_tokens: int = Field(default=2048)
    llm_base_url: str | None = Field(
        default=None,
        description=(
            "Override the default endpoint for OpenAI-compatible providers "
            "(groq/openrouter/ollama/openai). Leave unset to use each "
            "provider's default endpoint."
        ),
    )

    # --- Agent ----------------------------------------------------------
    max_agent_steps: int = Field(default=10)

    # --- App --------------------------------------------------------
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # --- Data storage -------------------------------------------------
    data_storage_dir: str = Field(default="./data/uploads")
    max_upload_size_mb: int = Field(default=50)
    max_dataset_rows: int = Field(default=200_000)

    # --- Frontend ---------------------------------------------------
    backend_url: str = Field(default="http://localhost:8000")

    @property
    def is_llm_configured(self) -> bool:
        """Whether a real LLM API key has been provided."""
        return bool(self.llm_api_key) and self.llm_api_key != "your-api-key-here"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Using lru_cache means the environment is parsed once per process,
    which is desirable for a stateless-config application like this one.
    """
    return Settings()
