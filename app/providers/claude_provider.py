from app.providers.base import _AmaliCompatibleProvider


class ClaudeProvider(_AmaliCompatibleProvider):
    _default_model = "claude-3-5-haiku-20241022"
