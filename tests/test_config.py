"""
Tests for app/config.py — Task 1.1 acceptance criteria.

Run with:  pytest tests/test_config.py -v
"""
import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from app.config import Settings


class IsolatedSettings(Settings):
    """
    Inherits all fields and validators from the real Settings class.
    Overrides only model_config to skip .env so tests are not affected
    by the developer's local .env file or any CI environment secrets.
    """
    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )


_PROVIDER_KEYS = ("AMALITECH_API_KEY",)


@pytest.fixture(autouse=True)
def clear_provider_keys(monkeypatch):
    """Remove real provider keys from os.environ before every test."""
    for key in _PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)


# ── Happy path ────────────────────────────────────────────────────────────────

def test_loads_with_amalitech_key():
    s = IsolatedSettings(amalitech_api_key="test-key-123")
    assert s.amalitech_api_key == "test-key-123"


def test_defaults_are_correct():
    s = IsolatedSettings(amalitech_api_key="any-key")
    assert s.primary_provider == "openai"
    assert s.redis_host == "localhost"
    assert s.redis_port == 6379
    assert s.rate_limit_per_minute == 60
    assert s.relevance_threshold == 0.70


# ── Failure path ──────────────────────────────────────────────────────────────

def test_raises_when_no_provider_key_set():
    with pytest.raises(ValidationError) as exc_info:
        IsolatedSettings()
    assert "Boot failed" in str(exc_info.value)


def test_raises_when_key_is_empty_string():
    with pytest.raises(ValidationError) as exc_info:
        IsolatedSettings(amalitech_api_key="")
    assert "Boot failed" in str(exc_info.value)


def test_raises_when_key_is_whitespace():
    with pytest.raises(ValidationError) as exc_info:
        IsolatedSettings(amalitech_api_key="   ")
    assert "Boot failed" in str(exc_info.value)
