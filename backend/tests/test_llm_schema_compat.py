"""Offline checks (no network, fake key) that the extraction schema is accepted by the Gemini integration."""
from langchain.chat_models import init_chat_model

from app.agent.schemas import TravelExtraction


def test_schema_binds_as_json_schema_response():
    llm = init_chat_model("gemini-3.6-flash", model_provider="google_genai", google_api_key="fake-key")
    bound = llm.with_structured_output(TravelExtraction).first
    assert bound.kwargs["response_mime_type"] == "application/json"
    schema = bound.kwargs["response_json_schema"]
    assert {"origin", "destination", "travel_date", "transport_mode", "passenger_count", "intent"} <= set(schema["properties"])
    assert not schema.get("required")  # nothing is mandatory for the model to emit


def test_gemini_3_drops_temperature_unless_explicit():
    llm = init_chat_model("gemini-3.6-flash", model_provider="google_genai", google_api_key="fake-key")
    assert llm.temperature is None
