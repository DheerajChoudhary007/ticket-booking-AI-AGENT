"""Diagnostic endpoints, mounted only when APP_ENV=development (see main.py).

POST /api/debug/extract runs the extraction step on a single message, statelessly, so extraction
quality can be inspected over HTTP before the conversational workflow exists (Phase 3+).
"""
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from app.agent.extraction import extract_travel_info
from app.agent.schemas import Intent
from app.api.deps import llm_provider, now_provider
from app.api.schemas import MessageIn
from app.domain.issues import FieldIssue
from app.domain.validation import missing_fields

router = APIRouter(prefix="/api/debug", tags=["debug"])


class ExtractResponse(BaseModel):
    today: date
    timezone: str
    intent: Intent
    extracted: dict[str, Any]
    issues: list[FieldIssue]
    missing_fields: list[str]  # stateless: relative to this one message only


@router.post("/extract", response_model=ExtractResponse)
async def extract(
    req: MessageIn,
    make_llm: Callable[[], BaseChatModel] = Depends(llm_provider),
    clock: Callable[[], datetime] = Depends(now_provider),
) -> ExtractResponse:
    llm = make_llm()
    now = clock()
    result = await run_in_threadpool(extract_travel_info, llm, req.message, now)
    return ExtractResponse(
        today=now.date(),
        timezone=getattr(now.tzinfo, "key", str(now.tzinfo)),
        intent=result.intent,
        extracted=result.updates.model_dump(mode="json", exclude_defaults=True),
        issues=result.issues,
        missing_fields=missing_fields(result.updates),
    )
