"""FastAPI dependencies. Both are overridable in tests via app.dependency_overrides."""
from collections.abc import Callable
from datetime import datetime

from langchain_core.language_models import BaseChatModel

from app.config import get_settings
from app.domain.dates import get_now
from app.llm.factory import get_llm


def llm_provider() -> Callable[[], BaseChatModel]:
    """A lazy LLM factory. The model is built inside the route, *after* request validation, so malformed
    requests get a 422 even when the LLM is not configured."""
    return get_llm


def now_provider() -> Callable[[], datetime]:
    """A clock returning timezone-aware 'now' in the configured timezone."""
    tz = get_settings().app_timezone
    return lambda: get_now(tz)
