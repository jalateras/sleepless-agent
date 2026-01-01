"""Git analyzer for extracting repository history and activity."""

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GitCommit:
    """Represents a git commit."""

    hash: str
    short_hash: str
    author: str
    date: datetime
    subject: str
    files_changed: int = 0


@dataclass
class GitHistory:
    """Extracted git history information."""

    # Repository info
    current_branch: str = "unknown"
    remote_url: Optional[str] = None
    is_dirty: bool = False
    uncommitted_changes: int = 0

    # Recent activity
    recent_commits: list[GitCommit] = field(default_factory=list)

    # File activity (most recently changed files)
    recently_modified_files: list[str] = field(default_factory=list)

    # Active areas (directories with most recent changes)
    active_directories: list[tuple[str, int]] = field(default_factory=list)


class GitAnalyzer:
    """Analyzes git repository for history and activity patterns."""

    def __init__(self, root_path: Path, commit_limit: int = 10, file_limit: int = 20):
        """Initialize the git analyzer.

        Args:
            root_path: Root directory of the repository
            commit_limit: Maximum number of recent commits to fetch
            file_limit: Maximum number of recently modified files to track
        """
        self.root_path = Path(root_path).resolve()
        self.commit_limit = commit_limit
        self.file_limit = file_limit

    def _run_git(self, *args: str, check: bool = True) -> Optional[str]:
        """Run a git command and return output.

        Args:
            *args: Git command arguments
            check: Whether to raise on non-zero exit

        Returns:
            Command output or None on failure
        """
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.root_path,
                capture_output=True,
                text=True,
                check=check,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.debug("context.git.command_failed", args=args, error=e.stderr)
            return None
        except FileNotFoundError:
            logger.debug("context.git.not_found")
            return None

    def is_git_repo(self) -> bool:
        """Check if the root path is a git repository.

        Returns:
            True if this is a git repository
        """
        git_dir = self.root_path / ".git"
        return git_dir.exists() and git_dir.is_dir()

    def analyze(self) -> GitHistory:
        """Analyze git repository history.

        Returns:
            GitHistory with extracted information
        """
        logger.debug("context.git.analyzing", root=str(self.root_path))
        history = GitHistory()

        if not self.is_git_repo():
            logger.debug("context.git.not_a_repo", path=str(self.root_path))
            return history

        # Get current branch
        branch = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        if branch:
            history.current_branch = branch

        # Get remote URL
        remote = self._run_git("remote", "get-url", "origin", check=False)
        if remote:
            # Sanitize URL (remove credentials if present)
            history.remote_url = self._sanitize_url(remote)

        # Check for uncommitted changes
        status = self._run_git("status", "--porcelain")
        if status:
            history.is_dirty = True
            history.uncommitted_changes = len(status.split("\n"))

        # Get recent commits
        history.recent_commits = self._get_recent_commits()

        # Get recently modified files
        history.recently_modified_files = self._get_recently_modified_files()

        # Analyze active directories (pass the files list)
        history.active_directories = self._analyze_active_directories(history.recently_modified_files)

        logger.debug(
            "context.git.complete",
            branch=history.current_branch,
            commits=len(history.recent_commits),
            dirty=history.is_dirty,
        )
        return history

    def _sanitize_url(self, url: str) -> str:
        """Remove credentials from a git URL.

        Args:
            url: Git remote URL

        Returns:
            Sanitized URL
        """
        # Handle https://user:pass@host/repo format
        if "@" in url and "://" in url:
            protocol, rest = url.split("://", 1)
            if "@" in rest:
                host_and_path = rest.split("@", 1)[1]
                return f"{protocol}://{host_and_path}"
        return url

    def _get_recent_commits(self) -> list[GitCommit]:
        """Get recent commits with metadata.

        Returns:
            List of recent commits
        """
        commits = []

        # Format: hash|short_hash|author|date|subject
        log_format = "%H|%h|%an|%aI|%s"
        log_output = self._run_git(
            "log",
            f"--format={log_format}",
            f"-{self.commit_limit}",
            "--no-merges",
        )

        if not log_output:
            return commits

        for line in log_output.split("\n"):
            if not line:
                continue
            try:
                parts = line.split("|", 4)
                if len(parts) < 5:
                    continue

                commit_hash, short_hash, author, date_str, subject = parts

                # Parse ISO date
                try:
                    date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except ValueError:
                    date = datetime.now()

                # Get files changed count for this commit
                stat_output = self._run_git("diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash)
                files_changed = len(stat_output.split("\n")) if stat_output else 0

                commits.append(
                    GitCommit(
                        hash=commit_hash,
                        short_hash=short_hash,
                        author=author,
                        date=date,
                        subject=subject,
                        files_changed=files_changed,
                    )
                )
            except Exception as e:
                logger.debug("context.git.parse_commit_error", line=line, error=str(e))
                continue

        return commits

    def _get_recently_modified_files(self) -> list[str]:
        """Get list of recently modified tracked files.

        Returns:
            List of file paths (relative to repo root)
        """
        # Get files from recent commits
        files_output = self._run_git(
            "log",
            f"-{self.commit_limit}",
            "--name-only",
            "--format=",
            "--no-merges",
        )

        if not files_output:
            return []

        # Deduplicate while preserving order (most recent first)
        seen = set()
        files = []
        for line in files_output.split("\n"):
            line = line.strip()
            if line and line not in seen:
                # Verify file still exists
                if (self.root_path / line).exists():
                    seen.add(line)
                    files.append(line)
                    if len(files) >= self.file_limit:
                        break

        return files

    def _analyze_active_directories(self, recently_modified_files: list[str]) -> list[tuple[str, int]]:
        """Analyze which directories have the most recent activity.

        Args:
            recently_modified_files: List of recently modified file paths

        Returns:
            List of (directory, change_count) tuples, sorted by activity
        """
        from collections import Counter

        # Count directory occurrences in recently modified files
        dir_counts: Counter[str] = Counter()

        for file_path in recently_modified_files:
            path = Path(file_path)
            # Count the immediate parent directory
            if path.parent != Path("."):
                dir_counts[str(path.parent)] += 1

        # Return top directories sorted by count
        return dir_counts.most_common(10)

    def format_history(self, history: Optional[GitHistory] = None) -> str:
        """Format git history as a readable string.

        Args:
            history: History to format (default: analyze and use result)

        Returns:
            Formatted history string
        """
        if history is None:
            history = self.analyze()

        lines = ["Git History:"]
        lines.append(f"  Branch: {history.current_branch}")

        if history.is_dirty:
            lines.append(f"  Status: {history.uncommitted_changes} uncommitted changes")
        else:
            lines.append("  Status: Clean")

        if history.recent_commits:
            lines.append(f"\n  Recent Commits ({len(history.recent_commits)}):")
            for commit in history.recent_commits[:5]:
                date_str = commit.date.strftime("%Y-%m-%d")
                lines.append(f"    {commit.short_hash} ({date_str}) {commit.subject[:60]}")

        if history.active_directories:
            lines.append("\n  Active Areas:")
            for directory, count in history.active_directories[:5]:
                lines.append(f"    {directory}/ ({count} changes)")

        if history.recently_modified_files:
            lines.append(f"\n  Recently Modified ({len(history.recently_modified_files)} files):")
            for file_path in history.recently_modified_files[:10]:
                lines.append(f"    {file_path}")

        return "\n".join(lines)
