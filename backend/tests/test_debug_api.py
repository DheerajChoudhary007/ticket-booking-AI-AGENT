from app.agent.schemas import TravelExtraction
from app.main import create_app
from fastapi.testclient import TestClient
from tests.conftest import StubLLM


def post(client, message="hi"):
    return client.post("/api/debug/extract", json={"message": message})


def test_extract_endpoint_end_to_end(client, use_llm, fixed_clock):
    use_llm(
        StubLLM(
            structured=TravelExtraction(
                intent="new_travel_request", origin="Jaipur", destination="Delhi", travel_date_text="tomorrow",
                transport_mode="train", passenger_count=2,
            )
        )
    )
    r = post(client, "I need to travel from Jaipur to Delhi tomorrow by train for two people")
    assert r.status_code == 200
    body = r.json()
    assert body["today"] == "2026-11-21" and body["timezone"] == "Asia/Kolkata"
    assert body["intent"] == "new_travel_request"
    assert body["extracted"] == {
        "origin": "Jaipur", "destination": "Delhi", "travel_date": "2026-11-22",
        "transport_mode": "train", "passenger_count": 2,
    }
    assert body["issues"] == [] and body["missing_fields"] == []


def test_incomplete_request_reports_missing_fields(client, use_llm, fixed_clock):
    use_llm(StubLLM(structured=TravelExtraction(origin="Jaipur", destination="Delhi")))
    body = post(client).json()
    assert body["missing_fields"] == ["transport_mode", "travel_date", "passenger_count"]


def test_invalid_date_reported_as_issue(client, use_llm, fixed_clock):
    use_llm(StubLLM(structured=TravelExtraction(origin="Jaipur", travel_date="2026-11-01")))
    body = post(client).json()
    assert body["issues"][0]["code"] == "date_in_past" and "travel_date" in body["missing_fields"]
    assert "travel_date" not in body["extracted"]


def test_validation_precedes_llm_config(client):
    assert post(client, "").status_code == 422


def test_missing_key_returns_503(client):
    r = post(client)
    assert r.status_code == 503 and r.json()["error"] == "llm_not_configured"


def test_llm_failure_returns_502(client, use_llm, fixed_clock):
    use_llm(StubLLM(exc=RuntimeError("secret detail")))
    r = post(client)
    assert r.status_code == 502 and "secret detail" not in r.text


def test_endpoint_not_mounted_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    from app.config import get_settings

    get_settings.cache_clear()
    c = TestClient(create_app(), raise_server_exceptions=False)
    assert c.post("/api/debug/extract", json={"message": "hi"}).status_code == 404
