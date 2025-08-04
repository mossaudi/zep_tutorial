# utils/cache.py - Recommended Implementation
import time
from typing import Any, Optional, Callable, Union
from functools import wraps


class TTLCache:
    """Time-to-live cache for expensive operations."""

    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self.cache:
            value, timestamp, ttl = self.cache[key]
            if time.time() - timestamp < ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Cache value with TTL."""
        ttl = ttl or self.default_ttl
        self.cache[key] = (value, time.time(), ttl)

    def invalidate(self, pattern: str = None):
        """Invalidate cache entries."""
        if pattern:
            keys_to_remove = [k for k in self.cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.cache[key]
        else:
            self.cache.clear()

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()


def cached_operation(cache_instance: Union[TTLCache, Callable[..., TTLCache]],
                     key_func: Optional[Callable] = None,
                     ttl: Optional[int] = None):
    """
    Decorator for caching expensive operations.

    Args:
        cache_instance: Either a TTLCache instance or a callable that returns one
        key_func: Function to generate cache key (optional)
        ttl: Time-to-live for cached values (optional)

    Example:
        @cached_operation(
            cache_instance=lambda self: self.cache,
            key_func=lambda self, **kwargs: f"key_{hash(str(kwargs))}"
        )
        def expensive_method(self, param1, param2="default"):
            return do_expensive_work(param1, param2)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Resolve cache instance
            try:
                if callable(cache_instance):
                    # Handle lambda functions like: lambda self: self.cache
                    if args:  # Method call with self
                        cache = cache_instance(args[0])
                    else:  # Function call without self
                        cache = cache_instance()
                else:
                    # Direct cache instance
                    cache = cache_instance

                # Validate cache instance
                if not hasattr(cache, 'get') or not hasattr(cache, 'set'):
                    raise ValueError(f"Cache instance must have 'get' and 'set' methods, got {type(cache)}")

            except Exception as e:
                print(f"❌ Cache Instance Resolution Error: {e}")
                # Fall back to executing without cache
                return func(*args, **kwargs)

            # Generate cache key
            try:
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    cache_key = f"{func.__name__}_{hash(str(args[1:]) + str(sorted(kwargs.items())))}"
            except Exception as e:
                print(f"❌ Cache Key Generation Error: {e}")
                # Fall back to executing without cache
                return func(*args, **kwargs)

            # Try to get from cache first
            try:
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            except Exception as e:
                print(f"❌ Cache Get Error: {e}")
                # Continue to execute function

            # Execute function
            result = func(*args, **kwargs)

            # Cache the result
            try:
                cache.set(cache_key, result, ttl)
            except Exception as e:
                print(f"❌ Cache Set Error: {e}")
                # Function succeeded, just couldn't cache

            return result

        return wrapper

    return decorator


# Alternative: Method-only version if you prefer the self.cache approach
def method_cached(key_func: Optional[Callable] = None, ttl: Optional[int] = None):
    """
    Simpler decorator that assumes the method's class has a 'cache' attribute.
    Only works with instance methods.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'cache'):
                raise AttributeError(f"{self.__class__.__name__} must have a 'cache' attribute")

            cache = self.cache

            # Generate cache key
            if key_func:
                cache_key = key_func(self, *args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"

            # Try cache first
            try:
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            except Exception as e:
                print(f"❌ Cache Get Error: {e}")

            # Execute and cache
            result = func(self, *args, **kwargs)
            try:
                cache.set(cache_key, result, ttl)
            except Exception as e:
                print(f"❌ Cache Set Error: {e}")

            return result

        return wrapper

    return decorator