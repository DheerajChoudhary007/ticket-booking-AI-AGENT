def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is False
    assert {"version", "environment", "llm_provider"} <= body.keys()


def test_health_never_leaks_api_key(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("LLM_API_KEY", "sk-super-secret")
    get_settings.cache_clear()
    r = client.get("/api/health")
    assert r.json()["llm_configured"] is True
    assert "sk-super-secret" not in r.text


def test_unknown_route_404(client):
    assert client.get("/api/nope").status_code == 404
