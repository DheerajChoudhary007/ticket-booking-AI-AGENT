"""API request/response models. Future-phase fields are present but null for now."""
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

MAX_MESSAGE_LENGTH = 2000


class MessageIn(BaseModel):
    message: str = Field(max_length=MAX_MESSAGE_LENGTH)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        return v


class ChatRequest(MessageIn):
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")

    def resolved_session_id(self) -> str:
        return self.session_id or uuid4().hex


class ChatResponse(BaseModel):
    session_id: str
    message: str
    workflow_status: str = "active"
    travel_request: dict[str, Any] | None = None
    results: list[dict[str, Any]] | None = None
    selected_option: dict[str, Any] | None = None
    booking_summary: dict[str, Any] | None = None
    booking_confirmation: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    llm_provider: str
    llm_configured: bool
