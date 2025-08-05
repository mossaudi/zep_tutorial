# cache/enhanced_cache.py
"""Enhanced caching with Redis integration and intelligent policies."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional, Dict

# Try to import Redis, fallback to memory cache
try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from Bom_Chatbot.utils.cache import TTLCache


@dataclass
class CachePolicy:
    """Cache policy configuration."""
    default_ttl: int = 300
    hot_cache_size: int = 100
    enable_redis: bool = True
    redis_prefix: str = "bom_agent:"
    compression: bool = True


class IntelligentCache:
    """Multi-tier caching with hot/cold storage and Redis persistence."""

    def __init__(self, policy: CachePolicy = None, redis_url: str = None):
        self.policy = policy or CachePolicy()
        self.hot_cache = TTLCache(default_ttl=60)  # 1-minute hot cache
        self.access_count = {}
        self.hit_stats = {"hits": 0, "misses": 0}

        # Initialize Redis if available
        self.redis_client = None
        if REDIS_AVAILABLE and self.policy.enable_redis and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
            except Exception as e:
                print(f"Redis connection failed: {e}")

    def _generate_key(self, key: str) -> str:
        """Generate consistent cache key with prefix."""
        if isinstance(key, dict):
            key = json.dumps(key, sort_keys=True)

        # Hash long keys
        if len(key) > 100:
            key = hashlib.md5(key.encode()).hexdigest()

        return f"{self.policy.redis_prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent promotion."""
        cache_key = self._generate_key(key)

        # Check hot cache first
        hot_result = self.hot_cache.get(cache_key)
        if hot_result is not None:
            self.hit_stats["hits"] += 1
            self._track_access(cache_key)
            return hot_result

        # Check Redis cold cache
        if self.redis_client:
            try:
                cold_result = await self.redis_client.get(cache_key)
                if cold_result:
                    data = json.loads(cold_result)

                    # Promote to hot cache if frequently accessed
                    self._track_access(cache_key)
                    if self._should_promote(cache_key):
                        self.hot_cache.set(cache_key, data, ttl=60)

                    self.hit_stats["hits"] += 1
                    return data
            except Exception as e:
                print(f"Redis get error: {e}")

        self.hit_stats["misses"] += 1
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with intelligent placement."""
        cache_key = self._generate_key(key)
        ttl = ttl or self.policy.default_ttl

        # Always set in hot cache for immediate access
        self.hot_cache.set(cache_key, value, ttl=min(ttl, 300))

        # Set in Redis for persistence
        if self.redis_client:
            try:
                serialized = json.dumps(value, default=str)
                await self.redis_client.setex(cache_key, ttl, serialized)
            except Exception as e:
                print(f"Redis set error: {e}")

    def _track_access(self, key: str) -> None:
        """Track access patterns for promotion decisions."""
        self.access_count[key] = self.access_count.get(key, 0) + 1

    def _should_promote(self, key: str) -> bool:
        """Determine if item should be promoted to hot cache."""
        return self.access_count.get(key, 0) >= 2

    async def invalidate_pattern(self, pattern: str) -> None:
        """Invalidate cache entries matching pattern."""
        # Clear hot cache
        keys_to_remove = [k for k in self.hot_cache.cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self.hot_cache.cache[key]

        # Clear Redis cache
        if self.redis_client:
            try:
                cache_pattern = self._generate_key(pattern + "*")
                keys = await self.redis_client.keys(cache_pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                print(f"Redis invalidation error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_stats["hits"] + self.hit_stats["misses"]
        hit_rate = (self.hit_stats["hits"] / max(total_requests, 1)) * 100

        return {
            "hit_rate": round(hit_rate, 2),
            "total_hits": self.hit_stats["hits"],
            "total_misses": self.hit_stats["misses"],
            "hot_cache_size": len(self.hot_cache.cache),
            "redis_available": self.redis_client is not None,
            "most_accessed": sorted(
                self.access_count.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

