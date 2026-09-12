"""Public API for StopWise."""

from .analyzer import (
    CompletionGenerator,
    StopWise,
    StopWiseError,
    Transport,
    analyze_conversation,
)
from .schemas import Action, Level, Signal, StopWiseResult
from .prompts import load_prompt, load_system_prompt

__all__ = [
    "Action",
    "CompletionGenerator",
    "Level",
    "load_prompt",
    "load_system_prompt",
    "Signal",
    "StopWise",
    "StopWiseError",
    "StopWiseResult",
    "Transport",
    "analyze_conversation",
]

__version__ = "0.2.0"
