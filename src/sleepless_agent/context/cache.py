"""Context cache with TTL for caching expensive context extraction operations."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cached value with expiration."""

    value: T
    created_at: datetime
    expires_at: datetime
    cache_key: str


class ContextCache:
    """TTL-based cache for context extraction results.

    Caches expensive operations like directory traversal, config parsing,
    and git history to avoid repeated computation within a time window.
    """

    def __init__(self, default_ttl: timedelta = timedelta(minutes=5)):
        """Initialize the context cache.

        Args:
            default_ttl: Default time-to-live for cache entries
        """
        self.default_ttl = default_ttl
        self._cache: dict[str, CacheEntry[Any]] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    def _generate_key(self, prefix: str, path: Path, **kwargs: Any) -> str:
        """Generate a cache key from prefix, path, and options.

        Args:
            prefix: Key prefix (e.g., "structure", "conventions")
            path: Root path being analyzed
            **kwargs: Additional parameters that affect the result

        Returns:
            Cache key string
        """
        key_data = {
            "prefix": prefix,
            "path": str(path.resolve()),
            **{k: str(v) for k, v in sorted(kwargs.items())},
        }
        key_json = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_json.encode()).hexdigest()[:16]
        return f"{prefix}:{key_hash}"

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        entry = self._cache.get(key)

        if entry is None:
            self._stats["misses"] += 1
            return None

        now = datetime.now()
        if now >= entry.expires_at:
            # Entry has expired
            del self._cache[key]
            self._stats["evictions"] += 1
            self._stats["misses"] += 1
            logger.debug("context.cache.expired", key=key)
            return None

        self._stats["hits"] += 1
        logger.debug("context.cache.hit", key=key)
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None,
    ) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live (uses default if not specified)
        """
        now = datetime.now()
        ttl = ttl or self.default_ttl

        entry = CacheEntry(
            value=value,
            created_at=now,
            expires_at=now + ttl,
            cache_key=key,
        )
        self._cache[key] = entry
        logger.debug("context.cache.set", key=key, ttl_seconds=ttl.total_seconds())

    def invalidate(self, key: str) -> bool:
        """Remove a specific entry from the cache.

        Args:
            key: Cache key to invalidate

        Returns:
            True if the key was found and removed
        """
        if key in self._cache:
            del self._cache[key]
            logger.debug("context.cache.invalidated", key=key)
            return True
        return False

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove all entries with a given prefix.

        Args:
            prefix: Key prefix to match

        Returns:
            Number of entries invalidated
        """
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{prefix}:")]
        for key in keys_to_remove:
            del self._cache[key]

        if keys_to_remove:
            logger.debug("context.cache.prefix_invalidated", prefix=prefix, count=len(keys_to_remove))

        return len(keys_to_remove)

    def clear(self) -> int:
        """Clear all entries from the cache.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        logger.debug("context.cache.cleared", count=count)
        return count

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired_keys = [k for k, v in self._cache.items() if now >= v.expires_at]

        for key in expired_keys:
            del self._cache[key]
            self._stats["evictions"] += 1

        if expired_keys:
            logger.debug("context.cache.cleanup", expired=len(expired_keys))

        return len(expired_keys)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0

        return {
            "entries": len(self._cache),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "hit_rate": hit_rate,
        }

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._cache)


# Global cache instance
_global_cache: Optional[ContextCache] = None


def get_context_cache(ttl: Optional[timedelta] = None) -> ContextCache:
    """Get the global context cache instance.

    Args:
        ttl: Optional TTL to use if creating a new instance

    Returns:
        Global ContextCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = ContextCache(default_ttl=ttl or timedelta(minutes=5))
    return _global_cache


def reset_context_cache() -> None:
    """Reset the global context cache (primarily for testing)."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
    _global_cache = None
