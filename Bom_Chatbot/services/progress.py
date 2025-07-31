# services/progress.py
"""Progress tracking service."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List


class ProgressLevel(Enum):
    """Progress message levels."""
    INFO = "info"
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ProgressMessage:
    """A progress message."""
    timestamp: str
    level: ProgressLevel
    step_name: str
    details: str

    def __str__(self) -> str:
        symbols = {
            ProgressLevel.INFO: '🔄',
            ProgressLevel.SUCCESS: '✅',
            ProgressLevel.ERROR: '❌',
            ProgressLevel.WARNING: '⚠️'
        }

        symbol = symbols.get(self.level, '📝')
        message = f"{symbol} [{self.timestamp}] {self.step_name}"
        if self.details:
            message += f": {self.details}"
        return message


class ProgressObserver(ABC):
    """Abstract base class for progress observers."""

    @abstractmethod
    def on_progress(self, message: ProgressMessage) -> None:
        """Handle a progress message."""
        pass


class ConsoleProgressObserver(ProgressObserver):
    """Console-based progress observer."""

    def on_progress(self, message: ProgressMessage) -> None:
        """Print progress message to console."""
        print(str(message))


class ProgressTracker:
    """Progress tracking service with observer pattern."""

    def __init__(self):
        self.observers: List[ProgressObserver] = []
        self.messages: List[ProgressMessage] = []

    def add_observer(self, observer: ProgressObserver) -> None:
        """Add a progress observer."""
        self.observers.append(observer)

    def remove_observer(self, observer: ProgressObserver) -> None:
        """Remove a progress observer."""
        if observer in self.observers:
            self.observers.remove(observer)

    def _notify_observers(self, message: ProgressMessage) -> None:
        """Notify all observers of a progress message."""
        self.messages.append(message)
        for observer in self.observers:
            observer.on_progress(message)

    def info(self, step_name: str, details: str = "") -> None:
        """Log an info progress message."""
        message = ProgressMessage(
            timestamp=time.strftime("%H:%M:%S"),
            level=ProgressLevel.INFO,
            step_name=step_name,
            details=details
        )
        self._notify_observers(message)

    def success(self, step_name: str, result: str = "") -> None:
        """Log a success progress message."""
        message = ProgressMessage(
            timestamp=time.strftime("%H:%M:%S"),
            level=ProgressLevel.SUCCESS,
            step_name=step_name,
            details=f"completed - {result}" if result else "completed"
        )
        self._notify_observers(message)

    def error(self, step_name: str, error: str) -> None:
        """Log an error progress message."""
        message = ProgressMessage(
            timestamp=time.strftime("%H:%M:%S"),
            level=ProgressLevel.ERROR,
            step_name=step_name,
            details=f"failed: {error}"
        )
        self._notify_observers(message)

    def warning(self, step_name: str, warning: str) -> None:
        """Log a warning progress message."""
        message = ProgressMessage(
            timestamp=time.strftime("%H:%M:%S"),
            level=ProgressLevel.WARNING,
            step_name=step_name,
            details=warning
        )
        self._notify_observers(message)

    def get_messages(self) -> List[ProgressMessage]:
        """Get all progress messages."""
        return self.messages.copy()

    def clear_messages(self) -> None:
        """Clear all progress messages."""
        self.messages.clear()


# Singleton instance for global access
_default_tracker = ProgressTracker()
_default_tracker.add_observer(ConsoleProgressObserver())


def get_progress_tracker() -> ProgressTracker:
    """Get the default progress tracker."""
    return _default_tracker


def configure_progress_tracker(tracker: ProgressTracker) -> None:
    """Configure a custom progress tracker."""
    global _default_tracker
    _default_tracker = tracker