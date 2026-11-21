"""Shared translation of provider-specific exceptions into AppError subclasses."""
from app.errors import AppError, LLMError, LLMTimeoutError


def translate_llm_exception(exc: Exception) -> AppError:
    """Provider SDKs raise many exception types; the API layer only ever sees AppError."""
    if isinstance(exc, AppError):
        return exc
    if "timeout" in type(exc).__name__.lower() or isinstance(exc, TimeoutError):
        return LLMTimeoutError("The language model took too long to respond. Please try again.")
    return LLMError("The language model request failed. Please try again.")
