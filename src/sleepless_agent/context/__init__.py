"""Codebase context extraction for task prompt injection.

This module provides context-aware task execution by extracting:
- Directory structure (tree view)
- Project conventions (from config files)
- Git history (recent activity, active areas)

Example usage:
    from sleepless_agent.context import ContextExtractor, extract_context_for_task

    # Quick extraction with defaults
    context_text = extract_context_for_task(project_path)

    # Full control
    extractor = ContextExtractor(project_path)
    context = extractor.extract(include_git=False)
    print(context.format_full())
"""

from sleepless_agent.context.cache import ContextCache, get_context_cache, reset_context_cache
from sleepless_agent.context.extractor import CodebaseContext, ContextExtractor, extract_context_for_task

__all__ = [
    "ContextCache",
    "ContextExtractor",
    "CodebaseContext",
    "extract_context_for_task",
    "get_context_cache",
    "reset_context_cache",
]
