import pytest
from pydantic import SecretStr

from app.config import Settings
from app.errors import LLMNotConfiguredError
from app.llm.factory import get_llm, normalize_provider


def settings(**over) -> Settings:
    base = dict(llm_provider="google_genai", llm_model="gemini-3.6-flash", llm_api_key=SecretStr("fake-key"))
    return Settings(**{**base, **over})


def test_defaults_are_gemini():
    s = Settings(_env_file=None)
    assert (s.llm_provider, s.llm_model) == ("google_genai", "gemini-3.6-flash")
    assert s.llm_temperature is None


def test_factory_requires_api_key():
    with pytest.raises(LLMNotConfiguredError):
        get_llm(Settings(llm_api_key=None))


def test_factory_builds_gemini_without_network():
    llm = get_llm(settings())
    assert type(llm).__name__ == "ChatGoogleGenerativeAI"
    assert llm.model == "gemini-3.6-flash"
    assert llm.timeout == 30.0


def test_temperature_is_not_sent_by_default():
    """Gemini 3.x rejects sampling parameters; the integration only omits temperature if we don't pass it."""
    assert get_llm(settings()).temperature is None


def test_temperature_is_passed_through_when_explicitly_configured():
    assert get_llm(settings(llm_temperature=0.2)).temperature == 0.2


def test_timeout_comes_from_settings():
    assert get_llm(settings(llm_timeout_seconds=7)).timeout == 7


@pytest.mark.parametrize("alias", ["gemini", "Gemini", "google", "GOOGLE_GENAI", " google_genai "])
def test_provider_aliases(alias):
    assert normalize_provider(alias) == "google_genai"
    assert type(get_llm(settings(llm_provider=alias))).__name__ == "ChatGoogleGenerativeAI"


def test_other_providers_pass_through_unchanged():
    assert normalize_provider("openai") == "openai" and normalize_provider("claude") == "anthropic"


def test_api_key_not_exposed_in_settings_repr_or_model_repr():
    s = settings()
    assert "fake-key" not in repr(s)
    assert "fake-key" not in repr(get_llm(s))
