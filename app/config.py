from functools import lru_cache
from typing import Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # AI provider selection
    primary_provider: str = "openai"

    # Provider keys
    amalitech_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # Infrastructure
    redis_host: str = "localhost"
    redis_port: int = 6379
    rate_limit_per_minute: int = 60
    relevance_threshold: float = 0.70

    @model_validator(mode="after")
    def require_at_least_one_provider_key(self) -> "Settings":
        keys = [self.amalitech_api_key, self.anthropic_api_key, self.google_api_key]
        if not any(k for k in keys if k and k.strip()):
            raise ValueError(
                "Boot failed: set at least one provider key — "
                "AMALITECH_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
