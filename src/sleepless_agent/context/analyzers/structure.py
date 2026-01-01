"""Structure analyzer for extracting directory tree information."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DirectoryNode:
    """Represents a node in the directory tree."""

    name: str
    is_dir: bool
    children: list["DirectoryNode"] = field(default_factory=list)
    file_count: int = 0  # For directories: count of files inside


class StructureAnalyzer:
    """Analyzes and extracts directory structure from a codebase."""

    # Common directories to ignore
    DEFAULT_IGNORE_DIRS = {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        ".venv",
        "venv",
        ".env",
        "env",
        ".tox",
        ".nox",
        "dist",
        "build",
        "*.egg-info",
        ".eggs",
        ".coverage",
        "htmlcov",
        ".hypothesis",
    }

    # Common file patterns to ignore
    DEFAULT_IGNORE_FILES = {
        ".DS_Store",
        "*.pyc",
        "*.pyo",
        "*.so",
        "*.dylib",
        "*.dll",
        "*.exe",
        "*.o",
        "*.a",
        "*.class",
        "*.jar",
        "*.war",
    }

    def __init__(
        self,
        root_path: Path,
        max_depth: int = 4,
        ignore_dirs: Optional[set[str]] = None,
        ignore_files: Optional[set[str]] = None,
    ):
        """Initialize the structure analyzer.

        Args:
            root_path: Root directory to analyze
            max_depth: Maximum depth to traverse (default: 4)
            ignore_dirs: Additional directory patterns to ignore
            ignore_files: Additional file patterns to ignore
        """
        self.root_path = Path(root_path).resolve()
        self.max_depth = max_depth
        self.ignore_dirs = self.DEFAULT_IGNORE_DIRS.copy()
        self.ignore_files = self.DEFAULT_IGNORE_FILES.copy()

        if ignore_dirs:
            self.ignore_dirs.update(ignore_dirs)
        if ignore_files:
            self.ignore_files.update(ignore_files)

    def _should_ignore(self, path: Path, is_dir: bool) -> bool:
        """Check if a path should be ignored.

        Args:
            path: Path to check
            is_dir: Whether the path is a directory

        Returns:
            True if the path should be ignored
        """
        name = path.name
        patterns = self.ignore_dirs if is_dir else self.ignore_files

        for pattern in patterns:
            if pattern.startswith("*"):
                # Wildcard pattern - check suffix
                if name.endswith(pattern[1:]):
                    return True
            elif name == pattern:
                return True

        return False

    def analyze(self) -> DirectoryNode:
        """Analyze the directory structure.

        Returns:
            Root DirectoryNode representing the structure
        """
        logger.debug("context.structure.analyzing", root=str(self.root_path))
        root_node = self._build_tree(self.root_path, depth=0)
        logger.debug(
            "context.structure.complete",
            root=str(self.root_path),
            file_count=root_node.file_count,
        )
        return root_node

    def _build_tree(self, path: Path, depth: int) -> DirectoryNode:
        """Recursively build the directory tree.

        Args:
            path: Current path to process
            depth: Current depth level

        Returns:
            DirectoryNode for this path
        """
        node = DirectoryNode(name=path.name, is_dir=path.is_dir())

        if not path.is_dir():
            return node

        if depth >= self.max_depth:
            # At max depth, just count files
            try:
                node.file_count = sum(1 for _ in path.iterdir())
            except PermissionError:
                node.file_count = 0
            return node

        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            logger.debug("context.structure.permission_denied", path=str(path))
            return node

        total_files = 0
        for entry in entries:
            if self._should_ignore(entry, entry.is_dir()):
                continue

            child = self._build_tree(entry, depth + 1)
            node.children.append(child)

            if child.is_dir:
                total_files += child.file_count
            else:
                total_files += 1

        node.file_count = total_files
        return node

    def format_tree(self, node: Optional[DirectoryNode] = None, prefix: str = "") -> str:
        """Format the directory tree as a string.

        Args:
            node: Node to format (default: analyze and use root)
            prefix: Current line prefix for indentation

        Returns:
            Formatted tree string
        """
        if node is None:
            node = self.analyze()

        lines = []
        self._format_node(node, lines, prefix, is_last=True, is_root=True)
        return "\n".join(lines)

    def _format_node(
        self,
        node: DirectoryNode,
        lines: list[str],
        prefix: str,
        is_last: bool,
        is_root: bool = False,
    ) -> None:
        """Recursively format a node and its children.

        Args:
            node: Node to format
            lines: List to append formatted lines to
            prefix: Current prefix for this line
            is_last: Whether this is the last sibling
            is_root: Whether this is the root node
        """
        if is_root:
            connector = ""
            new_prefix = ""
        else:
            connector = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")

        # Format the node name
        if node.is_dir:
            suffix = "/" if not is_root else ""
            if node.file_count > 0 and not node.children:
                # Directory at max depth - show file count
                lines.append(f"{prefix}{connector}{node.name}{suffix} ({node.file_count} files)")
            else:
                lines.append(f"{prefix}{connector}{node.name}{suffix}")
        else:
            lines.append(f"{prefix}{connector}{node.name}")

        # Format children
        for i, child in enumerate(node.children):
            is_last_child = i == len(node.children) - 1
            self._format_node(child, lines, new_prefix, is_last_child)
