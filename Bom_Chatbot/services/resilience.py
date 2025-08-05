# services/resilience.py
"""Enhanced error recovery and retry mechanisms."""

import asyncio
import random
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Any, Optional, Dict

from Bom_Chatbot.exceptions import AgentError, APIError
from Bom_Chatbot.services.memory_manager import ConversationMemory
from Bom_Chatbot.services.progress import get_progress_tracker


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retriable_exceptions: tuple = (APIError, ConnectionError, TimeoutError)


class CircuitBreaker:
    """Circuit breaker pattern for API resilience."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.progress = get_progress_tracker()

    def can_proceed(self) -> bool:
        """Check if request can proceed."""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            if self.last_failure_time and time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                self.progress.info("Circuit Breaker", "Moving to HALF_OPEN state")
                return True
            return False

        # HALF_OPEN state
        return True

    def on_success(self):
        """Handle successful request."""
        if self.state != "CLOSED":
            self.progress.success("Circuit Breaker", "Restored to CLOSED state")
        self.failure_count = 0
        self.state = "CLOSED"

    def on_failure(self):
        """Handle failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.progress.error("Circuit Breaker", f"OPEN - too many failures ({self.failure_count})")


def retry_with_backoff(config: RetryConfig = None):
    """Decorator for retry with exponential backoff."""
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            progress = get_progress_tracker()
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    if attempt > 0:
                        # Calculate delay with exponential backoff
                        delay = min(
                            config.base_delay * (config.exponential_base ** (attempt - 1)),
                            config.max_delay
                        )

                        # Add jitter to prevent thundering herd
                        if config.jitter:
                            delay *= (0.5 + random.random() * 0.5)

                        progress.info("Retry", f"Attempt {attempt + 1}/{config.max_attempts} in {delay:.1f}s")
                        await asyncio.sleep(delay)

                    result = await func(*args, **kwargs)

                    if attempt > 0:
                        progress.success("Retry", f"Succeeded on attempt {attempt + 1}")

                    return result

                except config.retriable_exceptions as e:
                    last_exception = e
                    progress.warning("Retry", f"Attempt {attempt + 1} failed: {str(e)}")

                    if attempt == config.max_attempts - 1:
                        progress.error("Retry", f"All {config.max_attempts} attempts failed")
                        break

                except Exception as e:
                    # Non-retriable exception
                    progress.error("Retry", f"Non-retriable error: {str(e)}")
                    raise

            # All attempts failed
            raise last_exception or AgentError("All retry attempts failed")

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # For sync functions, convert to async and run
            if asyncio.iscoroutinefunction(func):
                return asyncio.run(async_wrapper(*args, **kwargs))
            else:
                # Sync retry logic
                progress = get_progress_tracker()
                last_exception = None

                for attempt in range(config.max_attempts):
                    try:
                        if attempt > 0:
                            delay = min(
                                config.base_delay * (config.exponential_base ** (attempt - 1)),
                                config.max_delay
                            )
                            if config.jitter:
                                delay *= (0.5 + random.random() * 0.5)

                            progress.info("Retry", f"Attempt {attempt + 1}/{config.max_attempts} in {delay:.1f}s")
                            time.sleep(delay)

                        result = func(*args, **kwargs)
                        if attempt > 0:
                            progress.success("Retry", f"Succeeded on attempt {attempt + 1}")
                        return result

                    except config.retriable_exceptions as e:
                        last_exception = e
                        progress.warning("Retry", f"Attempt {attempt + 1} failed: {str(e)}")

                        if attempt == config.max_attempts - 1:
                            progress.error("Retry", f"All {config.max_attempts} attempts failed")
                            break

                    except Exception as e:
                        progress.error("Retry", f"Non-retriable error: {str(e)}")
                        raise

                raise last_exception or AgentError("All retry attempts failed")

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class FallbackHandler:
    """Handles fallback operations when primary methods fail."""

    def __init__(self, memory: ConversationMemory):
        self.memory = memory
        self.progress = get_progress_tracker()

    def handle_schematic_analysis_failure(self, image_url: str, error: Exception) -> Dict[str, Any]:
        """Fallback for schematic analysis failure."""
        self.progress.warning("Fallback", "Using manual component entry mode")

        return {
            "success": False,
            "error_message": str(error),
            "fallback_suggestion": "Please manually enter component details or try a different image format",
            "supported_formats": [".jpg", ".png", ".gif"],
            "manual_entry_template": {
                "components": [
                    {
                        "name": "Component_Name",
                        "part_number": "Part_Number",
                        "manufacturer": "Manufacturer",
                        "description": "Description"
                    }
                ]
            }
        }

    def handle_api_quota_exceeded(self, operation: str) -> Dict[str, Any]:
        """Handle API quota exceeded scenarios."""
        self.progress.error("Quota", f"API quota exceeded for {operation}")

        # Suggest alternatives based on context
        suggestions = []
        if self.memory.session_components:
            suggestions.append("Use existing session components for BOM creation")

        if self.memory.project_context.recent_boms:
            suggestions.append("Work with existing BOMs instead of creating new ones")

        return {
            "success": False,
            "error_type": "quota_exceeded",
            "operation": operation,
            "suggestions": suggestions,
            "retry_after": "1 hour",
            "alternative_actions": [
                "Use cached component data",
                "Manual component entry",
                "Export current session data"
            ]
        }