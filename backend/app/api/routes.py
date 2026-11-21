from collections.abc import Callable

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from langchain_core.language_models import BaseChatModel

from app.api.deps import llm_provider
from app.api.schemas import ChatRequest, ChatResponse, HealthResponse
from app.config import Settings, get_settings
from app.services.chat_service import generate_reply

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.app_env,
        llm_provider=settings.llm_provider,
        llm_configured=settings.llm_configured,
    )


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(
    req: ChatRequest, make_llm: Callable[[], BaseChatModel] = Depends(llm_provider)
) -> ChatResponse:
    session_id = req.resolved_session_id()
    llm = make_llm()
    reply = await run_in_threadpool(generate_reply, llm, req.message)
    return ChatResponse(session_id=session_id, message=reply)
