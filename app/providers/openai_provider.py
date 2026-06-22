from app.providers.base import _AmaliCompatibleProvider


class OpenAIProvider(_AmaliCompatibleProvider):
    _default_model = "gpt-4o-mini"
