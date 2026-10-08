"""evalforge: evaluate LLMs on task suites with pluggable models and scorers."""

from .models import get_model
from .runner import run_suite
from .suite import load_suite

__all__ = ["get_model", "load_suite", "run_suite"]
__version__ = "0.1.0"
