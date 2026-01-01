"""Context extractor for gathering codebase context for task injection."""

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Optional

from sleepless_agent.context.analyzers import ConventionsAnalyzer, GitAnalyzer, StructureAnalyzer
from sleepless_agent.context.analyzers.conventions import ProjectConventions
from sleepless_agent.context.analyzers.git import GitHistory
from sleepless_agent.context.analyzers.structure import DirectoryNode
from sleepless_agent.context.cache import ContextCache, get_context_cache
from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CodebaseContext:
    """Complete codebase context for injection into task prompts."""

    # Source path
    root_path: str

    # Extracted context
    structure: Optional[DirectoryNode] = None
    conventions: Optional[ProjectConventions] = None
    git_history: Optional[GitHistory] = None

    # Formatted strings (cached)
    _structure_text: Optional[str] = field(default=None, repr=False)
    _conventions_text: Optional[str] = field(default=None, repr=False)
    _git_text: Optional[str] = field(default=None, repr=False)

    @property
    def structure_text(self) -> str:
        """Get formatted structure text."""
        if self._structure_text is None and self.structure is not None:
            analyzer = StructureAnalyzer(Path(self.root_path))
            self._structure_text = analyzer.format_tree(self.structure)
        return self._structure_text or "(No structure available)"

    @property
    def conventions_text(self) -> str:
        """Get formatted conventions text."""
        if self._conventions_text is None and self.conventions is not None:
            analyzer = ConventionsAnalyzer(Path(self.root_path))
            self._conventions_text = analyzer.format_conventions(self.conventions)
        return self._conventions_text or "(No conventions available)"

    @property
    def git_text(self) -> str:
        """Get formatted git history text."""
        if self._git_text is None and self.git_history is not None:
            analyzer = GitAnalyzer(Path(self.root_path))
            self._git_text = analyzer.format_history(self.git_history)
        return self._git_text or "(No git history available)"

    def format_full(self, include_structure: bool = True, include_conventions: bool = True, include_git: bool = True) -> str:
        """Format the complete context for prompt injection.

        Args:
            include_structure: Include directory structure
            include_conventions: Include project conventions
            include_git: Include git history

        Returns:
            Formatted context string
        """
        sections = []

        if include_structure:
            sections.append(f"## Directory Structure\n```\n{self.structure_text}\n```")

        if include_conventions:
            sections.append(f"## {self.conventions_text}")

        if include_git:
            sections.append(f"## {self.git_text}")

        if not sections:
            return ""

        return "\n\n".join(sections)

    def format_compact(self) -> str:
        """Format a compact context summary for constrained prompts.

        Returns:
            Compact context string
        """
        parts = []

        if self.conventions:
            conv = self.conventions
            meta = []
            if conv.project_name:
                meta.append(f"Project: {conv.project_name}")
            if conv.python_version:
                meta.append(f"Python {conv.python_version}")
            if conv.test_framework:
                meta.append(f"Tests: {conv.test_framework}")
            if conv.linters:
                meta.append(f"Lint: {', '.join(conv.linters[:2])}")
            if meta:
                parts.append(" | ".join(meta))

        if self.git_history:
            gh = self.git_history
            git_info = [f"Branch: {gh.current_branch}"]
            if gh.is_dirty:
                git_info.append(f"{gh.uncommitted_changes} uncommitted")
            if gh.active_directories:
                active = [d for d, _ in gh.active_directories[:3]]
                git_info.append(f"Active: {', '.join(active)}")
            parts.append(" | ".join(git_info))

        return "\n".join(parts) if parts else "(No context)"


class ContextExtractor:
    """Extracts and caches codebase context for task prompts.

    This is the main interface for gathering context. It coordinates
    the individual analyzers and manages caching.
    """

    def __init__(
        self,
        root_path: Path,
        cache: Optional[ContextCache] = None,
        cache_ttl: timedelta = timedelta(minutes=5),
        max_depth: int = 4,
    ):
        """Initialize the context extractor.

        Args:
            root_path: Root directory to analyze
            cache: Cache instance (uses global cache if not provided)
            cache_ttl: TTL for cached context
            max_depth: Maximum directory depth for structure analysis
        """
        self.root_path = Path(root_path).resolve()
        self.cache = cache or get_context_cache(cache_ttl)
        self.cache_ttl = cache_ttl
        self.max_depth = max_depth

    def extract(
        self,
        include_structure: bool = True,
        include_conventions: bool = True,
        include_git: bool = True,
        force_refresh: bool = False,
    ) -> CodebaseContext:
        """Extract codebase context, using cache when available.

        Args:
            include_structure: Extract directory structure
            include_conventions: Extract project conventions
            include_git: Extract git history
            force_refresh: Bypass cache and re-extract

        Returns:
            CodebaseContext with extracted information
        """
        logger.debug(
            "context.extract.start",
            path=str(self.root_path),
            structure=include_structure,
            conventions=include_conventions,
            git=include_git,
        )

        context = CodebaseContext(root_path=str(self.root_path))

        if include_structure:
            context.structure = self._get_structure(force_refresh)

        if include_conventions:
            context.conventions = self._get_conventions(force_refresh)

        if include_git:
            context.git_history = self._get_git_history(force_refresh)

        logger.debug("context.extract.complete", path=str(self.root_path))
        return context

    def _get_structure(self, force_refresh: bool = False) -> Optional[DirectoryNode]:
        """Get directory structure, with caching.

        Args:
            force_refresh: Bypass cache

        Returns:
            DirectoryNode or None
        """
        cache_key = self.cache._generate_key("structure", self.root_path, max_depth=self.max_depth)

        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            analyzer = StructureAnalyzer(self.root_path, max_depth=self.max_depth)
            result = analyzer.analyze()
            self.cache.set(cache_key, result, self.cache_ttl)
            return result
        except Exception as e:
            logger.error("context.structure.error", path=str(self.root_path), error=str(e))
            return None

    def _get_conventions(self, force_refresh: bool = False) -> Optional[ProjectConventions]:
        """Get project conventions, with caching.

        Args:
            force_refresh: Bypass cache

        Returns:
            ProjectConventions or None
        """
        cache_key = self.cache._generate_key("conventions", self.root_path)

        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            analyzer = ConventionsAnalyzer(self.root_path)
            result = analyzer.analyze()
            self.cache.set(cache_key, result, self.cache_ttl)
            return result
        except Exception as e:
            logger.error("context.conventions.error", path=str(self.root_path), error=str(e))
            return None

    def _get_git_history(self, force_refresh: bool = False) -> Optional[GitHistory]:
        """Get git history, with caching.

        Args:
            force_refresh: Bypass cache

        Returns:
            GitHistory or None
        """
        cache_key = self.cache._generate_key("git", self.root_path)

        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            analyzer = GitAnalyzer(self.root_path)
            result = analyzer.analyze()
            self.cache.set(cache_key, result, self.cache_ttl)
            return result
        except Exception as e:
            logger.error("context.git.error", path=str(self.root_path), error=str(e))
            return None

    def invalidate_cache(self) -> int:
        """Invalidate all cached context for this path.

        Returns:
            Number of cache entries invalidated
        """
        count = 0
        for prefix in ["structure", "conventions", "git"]:
            key = self.cache._generate_key(prefix, self.root_path)
            if self.cache.invalidate(key):
                count += 1
        return count


def extract_context_for_task(
    project_path: Optional[Path] = None,
    compact: bool = False,
) -> str:
    """Convenience function to extract context for a task prompt.

    Args:
        project_path: Project root path (defaults to current directory)
        compact: Use compact format instead of full

    Returns:
        Formatted context string ready for prompt injection
    """
    if project_path is None:
        project_path = Path.cwd()

    extractor = ContextExtractor(project_path)
    context = extractor.extract()

    if compact:
        return context.format_compact()
    return context.format_full()
