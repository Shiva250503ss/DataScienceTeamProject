"""
DataPilot - Profiler & Cleaner Demo
Focused implementation with Profiler and Cleaner agents.
"""

from .agents.profiler import ProfilerAgent
from .agents.cleaner import CleanerAgent

__version__ = "1.0"
__all__ = [
    "ProfilerAgent",
    "CleanerAgent",
]
