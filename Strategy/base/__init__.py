"""Base interfaces for modular factor strategies."""

from .base_allocator import BaseAllocator
from .base_selector import BaseSelector
from .base_signal_generator import BaseSignalGenerator
from .base_window_strategy import BaseWindowStrategy

__all__ = [
    "BaseAllocator",
    "BaseSelector",
    "BaseSignalGenerator",
    "BaseWindowStrategy",
]
