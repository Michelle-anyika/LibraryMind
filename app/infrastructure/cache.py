import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CacheService:
    """
    Redis-backed cache with SHA-256 key hashing.
    Degrades to an in-memory dict if Redis is unavailable so the server
    never fails to start because of a missing cache layer.
    """

    def __init__(self, host: str = "localhost", port: int = 6379) -> None:
        self._fallback: dict[str, str] = {}
        self._using_fallback = False
        self._redis = None

        try:
            import redis.asyncio as aioredis  # noqa: PLC0415

            self._redis = aioredis.Redis(
                host=host,
                port=port,
                decode_responses=True,
                socket_connect_timeout=2,
            )
        except Exception as exc:  # ImportError or config error
            logger.warning(
                "Redis client could not be created — using in-memory fallback: %s", exc
            )
            self._using_fallback = True

    def _generate_key(self, text: str, namespace: str) -> str:
        """Return a deterministic 64-char hex key from namespace + text."""
        raw = f"{namespace}:{text}".encode()
        return hashlib.sha256(raw).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        """Return the deserialized value for *key*, or None on miss / error."""
        if self._using_fallback:
            raw = self._fallback.get(key)
            if raw is None:
                return None
            logger.debug("Cache HIT (in-memory) key=%s", key[:12])
            return json.loads(raw)

        try:
            raw = await self._redis.get(key)  # type: ignore[union-attr]
            if raw is None:
                return None
            logger.debug("Cache HIT (Redis) key=%s", key[:12])
            return json.loads(raw)
        except Exception as exc:
            logger.warning(
                "Redis GET failed — degrading to in-memory fallback: %s", exc
            )
            self._using_fallback = True
            return self._fallback.get(key) and json.loads(self._fallback[key])

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Serialize *value* as JSON and store it under *key* with *ttl* seconds."""
        serialized = json.dumps(value)

        if self._using_fallback:
            self._fallback[key] = serialized
            return

        try:
            await self._redis.setex(key, ttl, serialized)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning(
                "Redis SET failed — degrading to in-memory fallback: %s", exc
            )
            self._using_fallback = True
            self._fallback[key] = serialized
