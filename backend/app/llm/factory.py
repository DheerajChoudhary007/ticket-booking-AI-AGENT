"""Single place where an LLM is created. Nothing else imports provider-specific classes.

Default: Google Gemini via langchain-google-genai. Switching providers = changing LLM_PROVIDER /
LLM_MODEL / LLM_API_KEY in the environment (plus installing that provider's langchain package).
"""
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.config import Settings, get_settings
from app.errors import LLMNotConfiguredError

# Friendly names -> LangChain provider ids.
_PROVIDER_ALIASES = {"gemini": "google_genai", "google": "google_genai", "claude": "anthropic"}

# Provider -> kwarg name its LangChain integration uses for the API key.
_API_KEY_KWARG = {
    "google_genai": "google_api_key",
    "anthropic": "api_key",
    "openai": "api_key",
}


def normalize_provider(name: str) -> str:
    key = name.strip().lower()
    return _PROVIDER_ALIASES.get(key, key)


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    """Build a chat model from settings. Lazy: only called when a request needs it."""
    settings = settings or get_settings()
    if not settings.llm_configured:
        raise LLMNotConfiguredError("LLM_API_KEY is not set. Configure it in backend/.env.")

    provider = normalize_provider(settings.llm_provider)
    kwargs: dict[str, Any] = {
        "timeout": settings.llm_timeout_seconds,
        _API_KEY_KWARG.get(provider, "api_key"): settings.llm_api_key.get_secret_value(),  # type: ignore[union-attr]
    }
    # Only sent when explicitly configured: Gemini 3.x rejects sampling parameters, and the Gemini
    # integration only drops its own temperature default when none is passed explicitly.
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature

    return init_chat_model(settings.llm_model, model_provider=provider, **kwargs)
