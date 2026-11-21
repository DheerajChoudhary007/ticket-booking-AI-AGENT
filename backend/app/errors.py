"""Domain errors mapped to HTTP responses in one place (see main.py)."""


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class LLMNotConfiguredError(AppError):
    status_code = 503
    code = "llm_not_configured"


class LLMTimeoutError(AppError):
    status_code = 504
    code = "llm_timeout"


class LLMError(AppError):
    status_code = 502
    code = "llm_error"
