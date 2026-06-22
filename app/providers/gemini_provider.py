from app.providers.base import _AmaliCompatibleProvider


class GeminiProvider(_AmaliCompatibleProvider):
    _default_model = "gemini-1.5-flash"
