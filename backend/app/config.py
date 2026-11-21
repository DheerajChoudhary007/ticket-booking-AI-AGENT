"""Application configuration. All secrets and tunables come from environment variables."""
from functools import lru_cache
from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Travel Booking Agent"
    app_version: str = "0.1.0"
    app_env: str = "development"
    app_timezone: str = "Asia/Kolkata"  # IANA name; "today" is always computed in this zone

    # LLM (provider-agnostic; see app/llm/factory.py)
    llm_provider: str = "google_genai"  # aliases "gemini" / "google" are accepted (see llm/factory.py)
    llm_model: str = "gemini-3.6-flash"
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    # None = don't send it. Gemini 3.x models reject sampling parameters (temperature/top_p/top_k),
    # so only set this if you switch to a model that supports it.
    llm_temperature: float | None = Field(default=None, ge=0, le=2)

    database_url: str = "sqlite:///./travel_agent.db"
    cors_origins: list[str] = ["http://localhost:5173"]

    @field_validator("app_timezone")
    @classmethod
    def _valid_timezone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (KeyError, ValueError, OSError):
            raise ValueError(f"unknown timezone '{v}' (use an IANA name such as 'Asia/Kolkata')") from None
        return v

    @property
    def llm_configured(self) -> bool:
        return self.llm_api_key is not None and bool(self.llm_api_key.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    return Settings()
