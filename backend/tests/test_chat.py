import pytest

from tests.conftest import StubLLM


def test_chat_generates_session_id_when_missing(client, use_llm):
    stub = use_llm(StubLLM("Sure, where to?"))
    r = client.post("/api/chat", json={"message": "I want to travel"})
    assert r.status_code == 200
    body = r.json()
    assert body["message"] == "Sure, where to?"
    assert len(body["session_id"]) == 32
    assert body["workflow_status"] == "active"
    assert len(stub.calls) == 1


def test_chat_preserves_given_session_id(client, use_llm):
    use_llm(StubLLM())
    r = client.post("/api/chat", json={"session_id": "abc-123", "message": "hi"})
    assert r.json()["session_id"] == "abc-123"


def test_chat_omits_future_fields_when_null(client, use_llm):
    use_llm(StubLLM())
    body = client.post("/api/chat", json={"message": "hi"}).json()
    assert "travel_request" not in body and "booking_confirmation" not in body


def test_chat_strips_message_before_sending(client, use_llm):
    stub = use_llm(StubLLM())
    client.post("/api/chat", json={"message": "   hello   "})
    assert stub.calls[0][-1].content == "hello"


def test_chat_handles_content_block_lists(client, use_llm):
    from langchain_core.messages import AIMessage

    stub = StubLLM()
    stub.invoke = lambda m: AIMessage(content=[{"type": "text", "text": "Part A. "}, {"type": "text", "text": "Part B."}])
    use_llm(stub)
    assert client.post("/api/chat", json={"message": "hi"}).json()["message"] == "Part A. Part B."


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": "   "},
        {"message": "x" * 2001},
        {"message": "hi", "session_id": "bad id!"},
        {"message": "hi", "session_id": "a" * 65},
        {"message": 123},
    ],
)
def test_chat_validation_errors(client, use_llm, payload):
    use_llm(StubLLM())
    r = client.post("/api/chat", json=payload)
    assert r.status_code == 422
    assert r.json()["error"] == "validation_error"


def test_chat_without_api_key_returns_503(client):
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 503
    assert r.json()["error"] == "llm_not_configured"


def test_chat_llm_failure_returns_502_without_leaking_details(client, use_llm):
    use_llm(StubLLM(exc=RuntimeError("secret upstream detail")))
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 502
    assert r.json()["error"] == "llm_error"
    assert "secret upstream detail" not in r.text


def test_chat_llm_timeout_returns_504(client, use_llm):
    class APITimeoutError(Exception):
        pass

    use_llm(StubLLM(exc=APITimeoutError("slow")))
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 504
    assert r.json()["error"] == "llm_timeout"


def test_chat_empty_llm_reply_returns_502(client, use_llm):
    use_llm(StubLLM(reply="   "))
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 502


def test_validation_error_takes_priority_over_missing_llm_config(client):
    """Regression: with no API key, an invalid request must still be 422, not 503."""
    r = client.post("/api/chat", json={"message": ""})
    assert r.status_code == 422


def test_chat_ignores_non_text_blocks_such_as_thinking(client, use_llm):
    from langchain_core.messages import AIMessage

    stub = StubLLM()
    stub.invoke = lambda m: AIMessage(
        content=[{"type": "thinking", "thinking": "internal reasoning"}, {"type": "text", "text": "Where to?"}, "!"]
    )
    use_llm(stub)
    assert client.post("/api/chat", json={"message": "hi"}).json()["message"] == "Where to?!"
