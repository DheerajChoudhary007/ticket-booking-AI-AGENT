"""Chat orchestration. Phase 1: a plain LLM reply. Phase 5 replaces this with the LangGraph agent.

The LLM call is blocking, so the route runs it in a threadpool. Errors are normalised into
AppError subclasses so the API layer never sees provider-specific exceptions.
"""
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.errors import LLMError
from app.llm.utils import translate_llm_exception

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a friendly travel booking assistant for trains, buses and flights. "
    "The booking features are still being built, so for now chat helpfully about travel plans. "
    "Never claim to have searched inventory or made a booking."
)


def _text_of(content: object) -> str:
    """Normalise message content (str or list of content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b if isinstance(b, str) else b.get("text", "")
            for b in content
            if isinstance(b, str) or (isinstance(b, dict) and b.get("type", "text") == "text")
        ]
        return "".join(parts)
    return str(content)


def generate_reply(llm: BaseChatModel, user_message: str) -> str:
    try:
        result = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)])
    except Exception as exc:
        logger.exception("LLM call failed")
        raise translate_llm_exception(exc) from exc

    text = _text_of(result.content).strip()
    if not text:
        raise LLMError("The language model returned an empty response.")
    return text
