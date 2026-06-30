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

    # AmaliAI gateway — the single entry point for all AI providers
    amalitech_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None

    # Infrastructure
    redis_host: str = "localhost"
    redis_port: int = 6379
    rate_limit_per_minute: int = 60
    relevance_threshold: float = 0.70

    @model_validator(mode="after")
    def require_amalitech_key(self) -> "Settings":
        if not self.amalitech_api_key or not self.amalitech_api_key.strip():
            raise ValueError(
                "Boot failed: AMALITECH_API_KEY must be set in .env"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
