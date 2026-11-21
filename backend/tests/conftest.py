from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.api.deps import llm_provider, now_provider
from app.config import get_settings
from app.domain.dates import get_now
from app.main import create_app

# Saturday 21 Nov 2026, 10:00 in Asia/Kolkata. Fixed so date assertions are deterministic.
FIXED_NOW = get_now("Asia/Kolkata", datetime(2026, 11, 21, 4, 30, tzinfo=UTC))


class _StructuredRunnable:
    def __init__(self, outer: "StubLLM"):
        self.outer = outer

    def invoke(self, messages):
        self.outer.calls.append(messages)
        if self.outer.exc:
            raise self.outer.exc
        return self.outer.structured


class StubLLM:
    """Stand-in for a chat model. .invoke serves plain chat; .with_structured_output serves extraction."""

    def __init__(self, reply="Hello! Where would you like to travel?", exc: Exception | None = None, structured=None):
        self.reply, self.exc, self.structured, self.calls = reply, exc, structured, []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.exc:
            raise self.exc
        return AIMessage(content=self.reply)

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return _StructuredRunnable(self)


@pytest.fixture(autouse=True)
def clean_env(request, monkeypatch):
    """Isolate tests from the developer's real environment AND their .env file.

    Env vars outrank .env in pydantic-settings, so we pin them rather than merely deleting them.
    Tests marked `live` deliberately use the real configuration."""
    if request.node.get_closest_marker("live"):
        yield
        return
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "google_genai")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("APP_TIMEZONE", "Asia/Kolkata")
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def use_llm(app):
    def _use(stub: StubLLM) -> StubLLM:
        app.dependency_overrides[llm_provider] = lambda: (lambda: stub)
        return stub

    return _use


@pytest.fixture
def fixed_clock(app):
    app.dependency_overrides[now_provider] = lambda: (lambda: FIXED_NOW)
    return FIXED_NOW
