"""Local signal collector for project analysis."""

from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from sleepless_agent.monitoring.logging import get_logger
from sleepless_agent.orchestration.project_config import ProjectConfig
from sleepless_agent.orchestration.signals import (
    SignalSource,
    SignalType,
    WorkItem,
    TODOComment,
    TestFailure,
    CoverageGap,
)

logger = get_logger(__name__)


class LocalSignalCollector:
    """Collects signals from local project workspace.

    Scans for:
    - TODO/FIXME comments in code
    - Test failures (pytest, jest, etc.)
    - Coverage gaps
    - Broken imports
    - Stale git branches
    """

    # Patterns for TODO comments
    TODO_PATTERNS = [
        r"(?i)TODO[:\s]*(.+)",
        r"(?i)FIXME[:\s]*(.+)",
        r"(?i)HACK[:\s]*(.+)",
        r"(?i)XXX[:\s]*(.+)",
        r"(?i)NOTE[:\s]*(.+)",
    ]

    # File extensions to scan for TODOs
    CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".tsx", ".jsx",
        ".java", ".kt", ".go", ".rs", ".c", ".cpp", ".h",
        ".rb", ".php", ".swift", ".scala",
    }

    def __init__(self, project: ProjectConfig):
        """Initialize collector for a project.

        Args:
            project: Project configuration
        """
        self.project = project
        self.workspace = Path(project.local_path).expanduser()

    def collect_all(self) -> List[WorkItem]:
        """Collect all local signals for the project.

        Returns:
            List of WorkItems discovered
        """
        items = []

        # Scan for TODOs
        try:
            items.extend(self._scan_todos())
        except Exception as e:
            logger.warning(
                "collector.todos.failed",
                project_id=self.project.id,
                error=str(e),
            )

        # Check for test failures
        try:
            items.extend(self._check_test_failures())
        except Exception as e:
            logger.debug(
                "collector.tests.failed",
                project_id=self.project.id,
                error=str(e),
            )

        # Analyze coverage gaps
        try:
            items.extend(self._analyze_coverage())
        except Exception as e:
            logger.debug(
                "collector.coverage.failed",
                project_id=self.project.id,
                error=str(e),
            )

        # Check for broken imports
        try:
            items.extend(self._check_imports())
        except Exception as e:
            logger.debug(
                "collector.imports.failed",
                project_id=self.project.id,
                error=str(e),
            )

        # Check for stale branches
        try:
            items.extend(self._check_stale_branches())
        except Exception as e:
            logger.debug(
                "collector.branches.failed",
                project_id=self.project.id,
                error=str(e),
            )

        logger.info(
            "collector.local.complete",
            project_id=self.project.id,
            items_found=len(items),
        )

        return items

    def _scan_todos(self) -> List[WorkItem]:
        """Scan codebase for TODO/FIXME comments.

        Returns:
            List of WorkItems from TODO comments
        """
        items = []
        workspace_str = str(self.workspace)

        for root, dirs, files in os.walk(workspace_str):
            # Skip common directories to ignore
            dirs[:] = [d for d in dirs if d not in {
                ".git", ".venv", "venv", "node_modules",
                "__pycache__", ".pytest_cache", "dist", "build",
                ".next", ".nuxt", "target",
            }]

            for filename in files:
                filepath = Path(root) / filename
                if filepath.suffix not in self.CODE_EXTENSIONS:
                    continue

                try:
                    todos = self._extract_todos_from_file(filepath)
                    for todo in todos:
                        item = self._todo_to_work_item(todo, filepath)
                        if item:
                            items.append(item)
                except Exception as e:
                    logger.debug("collector.file_scan_failed", path=str(filepath), error=str(e))

        logger.debug(
            "collector.todos.scanned",
            project_id=self.project.id,
            todos_found=len(items),
        )
        return items

    def _extract_todos_from_file(self, filepath: Path) -> List[TODOComment]:
        """Extract TODO comments from a single file.

        Args:
            filepath: Path to the file

        Returns:
            List of TODOComment objects
        """
        todos = []

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
        except Exception:
            return todos

        for line_num, line in enumerate(lines, start=1):
            for pattern in self.TODO_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    prefix = match.group(0).split(":")[0].upper()
                    content = match.group(1).strip()
                    if content:
                        todos.append(TODOComment(
                            content=content,
                            file_path=str(filepath.relative_to(self.workspace)),
                            line_number=line_num,
                            prefix=prefix,
                        ))
                        # Only match first pattern per line
                        break

        return todos

    def _todo_to_work_item(self, todo: TODOComment, filepath: Path) -> Optional[WorkItem]:
        """Convert a TODO comment to a WorkItem.

        Args:
            todo: TODOComment object
            filepath: Full path to the file

        Returns:
            WorkItem or None if TODO should be ignored
        """
        # Filter out common ignorable TODOs
        content_lower = todo.content.lower()
        ignore_patterns = [
            "consider implementing",
            "maybe add",
            "could be improved",
            "nice to have",
            "someday",
            "later",
        ]
        if any(pattern in content_lower for pattern in ignore_patterns):
            return None

        # Estimate age from file modification time
        try:
            mtime = filepath.stat().st_mtime
            age_days = int((datetime.now().timestamp() - mtime) / 86400)
        except Exception:
            age_days = None

        # Determine signal type and urgency
        content_lower = todo.content.lower()
        if "fix" in content_lower or "bug" in content_lower:
            signal_type = SignalType.BUGFIX
            urgency = 60
        elif "security" in content_lower or "vulnerability" in content_lower:
            signal_type = SignalType.SECURITY
            urgency = 90
        elif "optimize" in content_lower or "slow" in content_lower:
            signal_type = SignalType.PERFORMANCE
            urgency = 40
        elif "test" in content_lower:
            signal_type = SignalType.TEST
            urgency = 50
        elif "refactor" in content_lower:
            signal_type = SignalType.REFACTOR
            urgency = 30
        elif "document" in content_lower or "doc" in content_lower:
            signal_type = SignalType.DOCUMENTATION
            urgency = 20
        else:
            signal_type = SignalType.MAINTENANCE
            urgency = 30

        # Boost urgency for FIXME vs TODO
        if todo.prefix == "FIXME":
            urgency = min(urgency + 20, 100)
        elif todo.prefix == "HACK":
            urgency = min(urgency + 10, 100)

        # Calculate confidence based on TODO clarity
        confidence = 0.8  # Base confidence
        if len(todo.content) > 50:  # Detailed TODO
            confidence = 0.9
        elif len(todo.content) < 10:  # Vague TODO
            confidence = 0.6

        return WorkItem(
            source=SignalSource.TODO,
            type=signal_type,
            title=f"{todo.prefix}: {todo.content[:50]}",
            description=todo.content,
            location=todo.file_path,
            line_number=todo.line_number,
            urgency=urgency,
            confidence=confidence,
            age_days=age_days,
            metadata={
                "prefix": todo.prefix,
                "full_line": self._get_line_with_context(filepath, todo.line_number),
            },
        )

    def _get_line_with_context(self, filepath: Path, line_number: int, context_lines: int = 2) -> str:
        """Get the TODO line with surrounding context.

        Args:
            filepath: Path to the file
            line_number: Line number of TODO
            context_lines: Number of lines before/after to include

        Returns:
            String with TODO and context
        """
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()

            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)

            result_lines = []
            for i in range(start, end):
                prefix = ">>> " if i == line_number - 1 else "    "
                result_lines.append(f"{prefix}{lines[i]}")

            return "\n".join(result_lines)
        except Exception:
            return ""

    def _check_test_failures(self) -> List[WorkItem]:
        """Check for test failures by running test suite.

        Returns:
            List of WorkItems for failing tests
        """
        items = []

        # Detect project type and run appropriate test command
        project_type = self._detect_project_type()

        if project_type == "python":
            items.extend(self._check_pytest_failures())
        elif project_type == "javascript":
            items.extend(self._check_jest_failures())
        else:
            logger.debug(
                "collector.tests.unsupported",
                project_id=self.project.id,
                project_type=project_type,
            )

        return items

    def _detect_project_type(self) -> Optional[str]:
        """Detect the project type based on files present.

        Returns:
            "python", "javascript", or None
        """
        if (self.workspace / "pyproject.toml").exists() or \
           (self.workspace / "setup.py").exists() or \
           (self.workspace / "requirements.txt").exists():
            return "python"
        elif (self.workspace / "package.json").exists():
            return "javascript"
        return None

    def _check_pytest_failures(self) -> List[WorkItem]:
        """Check for pytest failures.

        Returns:
            List of WorkItems for failing tests
        """
        items = []

        # Check if pytest config exists
        if not any([
            (self.workspace / "pyproject.toml").exists(),
            (self.workspace / "pytest.ini").exists(),
            (self.workspace / "setup.cfg").exists(),
            (self.workspace / "tox.ini").exists(),
        ]):
            return items

        try:
            # Run pytest with --tb=no to get summary only
            result = subprocess.run(
                ["pytest", "--tb=no", "-q"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                # Parse output for failed tests
                failed = self._parse_pytest_output(result.stdout + result.stderr)
                for failure in failed:
                    items.append(WorkItem(
                        source=SignalSource.TEST_FAILURE,
                        type=SignalType.BUGFIX,
                        title=f"Fix failing test: {failure.test_name}",
                        description=f"Test is failing:\n{failure.error_message}",
                        location=failure.file_path,
                        urgency=70,
                        confidence=0.95,
                        metadata={
                            "test_name": failure.test_name,
                            "error_type": failure.error_type,
                        },
                    ))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # pytest not available or tests take too long
            pass
        except Exception as e:
            logger.debug("collector.pytest.failed", error=str(e))

        return items

    def _parse_pytest_output(self, output: str) -> List[TestFailure]:
        """Parse pytest output for test failures.

        Args:
            output: pytest stdout/stderr

        Returns:
            List of TestFailure objects
        """
        failures = []

        # Look for FAILED lines
        for line in output.splitlines():
            if line.startswith("FAILED "):
                parts = line.split()
                if len(parts) >= 2:
                    test_name = parts[1]
                    failures.append(TestFailure(
                        test_name=test_name,
                        file_path=test_name.split("::")[0] if "::" in test_name else "unknown",
                        error_message=line,
                    ))

        return failures

    def _check_jest_failures(self) -> List[WorkItem]:
        """Check for jest test failures.

        Returns:
            List of WorkItems for failing tests
        """
        items = []

        try:
            result = subprocess.run(
                ["npm", "test", "--", "--verbose"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                # Parse jest output for failures
                for line in (result.stdout + result.stderr).splitlines():
                    if "●" in line and ("failed" in line or "passed" in line.lower()):
                        items.append(WorkItem(
                            source=SignalSource.TEST_FAILURE,
                            type=SignalType.BUGFIX,
                            title=f"Fix failing test: {line.strip()}",
                            description=line.strip(),
                            urgency=70,
                            confidence=0.9,
                        ))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        except Exception as e:
            logger.debug("collector.jest.failed", error=str(e))

        return items

    def _analyze_coverage(self) -> List[WorkItem]:
        """Analyze code coverage and identify gaps.

        Returns:
            List of WorkItems for coverage gaps
        """
        items = []

        # Get coverage target from project goals
        coverage_target = 80  # Default
        for goal in self.project.goals:
            if goal.type == "coverage" and goal.target:
                coverage_target = goal.target
                break

        # Try different coverage tools
        if self._detect_project_type() == "python":
            items.extend(self._analyze_python_coverage(coverage_target))
        elif self._detect_project_type() == "javascript":
            items.extend(self._analyze_jest_coverage(coverage_target))

        return items

    def _analyze_python_coverage(self, target_percent: float) -> List[WorkItem]:
        """Analyze Python coverage using coverage.py data.

        Args:
            target_percent: Target coverage percentage

        Returns:
            List of WorkItems for coverage gaps
        """
        items = []

        # Look for .coverage file
        coverage_file = self.workspace / ".coverage"
        if not coverage_file.exists():
            return items

        try:
            result = subprocess.run(
                ["coverage", "report", "--format=json"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0 and result.stdout:
                import json
                data = json.loads(result.stdout)

                # Check overall coverage
                overall_percent = data.get("percent_covered", 0)
                if overall_percent < target_percent:
                    gap = target_percent - overall_percent
                    items.append(WorkItem(
                        source=SignalSource.COVERAGE_GAP,
                        type=SignalType.TEST,
                        title=f"Improve test coverage from {overall_percent:.1f}% to {target_percent:.0f}%",
                        description=f"Current coverage is {overall_percent:.1f}%, target is {target_percent:.0f}%",
                        urgency=int(gap * 2),  # 2 urgency points per percent gap
                        confidence=0.95,
                        metadata={
                            "current_coverage": overall_percent,
                            "target_coverage": target_percent,
                            "gap_percent": gap,
                        },
                    ))

                # Check individual files with low coverage
                files_data = data.get("files", {})
                for file_path, file_data in files_data.items():
                    file_coverage = file_data.get("summary", {}).get("percent_covered", 100)
                    if file_coverage < target_percent - 10:  # Only flag significantly under-performing files
                        # Get relative path
                        try:
                            rel_path = Path(file_path).relative_to(self.workspace)
                        except ValueError:
                            rel_path = Path(file_path)

                        items.append(WorkItem(
                            source=SignalSource.COVERAGE_GAP,
                            type=SignalType.TEST,
                            title=f"Add tests for {rel_path}",
                            description=f"Coverage is {file_coverage:.1f}%, target is {target_percent:.0f}%",
                            location=str(rel_path),
                            urgency=int((target_percent - file_coverage)),
                            confidence=0.9,
                            metadata={
                                "file_coverage": file_coverage,
                                "target_coverage": target_percent,
                                "lines_uncovered": file_data.get("summary", {}).get("missing_lines", 0),
                            },
                        ))

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.debug("collector.coverage.failed", error=str(e))

        return items

    def _analyze_jest_coverage(self, target_percent: float) -> List[WorkItem]:
        """Analyze JavaScript/Jest coverage.

        Args:
            target_percent: Target coverage percentage

        Returns:
            List of WorkItems for coverage gaps
        """
        items = []

        # Look for coverage info in coverage/ directory
        coverage_dir = self.workspace / "coverage"
        if not coverage_dir.exists():
            return items

        coverage_file = coverage_dir / "coverage-final.json"
        if not coverage_file.exists():
            return items

        try:
            import json
            with open(coverage_file) as f:
                data = json.load(f)

            total = data.get("total", {})
            lines_pct = total.get("lines", {}).get("pct", 0)

            if lines_pct < target_percent:
                gap = target_percent - lines_pct
                items.append(WorkItem(
                    source=SignalSource.COVERAGE_GAP,
                    type=SignalType.TEST,
                    title=f"Improve test coverage from {lines_pct:.1f}% to {target_percent:.0f}%",
                    description=f"Current coverage is {lines_pct:.1f}%, target is {target_percent:.0f}%",
                    urgency=int(gap * 2),
                    confidence=0.9,
                    metadata={
                        "current_coverage": lines_pct,
                        "target_coverage": target_percent,
                    },
                ))
        except Exception as e:
            logger.debug("collector.jest_coverage.failed", error=str(e))

        return items

    def _check_imports(self) -> List[WorkItem]:
        """Check for broken imports by importing main module.

        Returns:
            List of WorkItems for broken imports
        """
        items = []

        if self._detect_project_type() != "python":
            return items

        # Try importing the main package
        try:
            # Find the main package (look for __init__.py in workspace root)
            init_files = list(self.workspace.glob("__init__.py"))
            if not init_files:
                return items

            # Try to import
            import sys
            import importlib.util

            # Add workspace to path temporarily
            sys.path.insert(0, str(self.workspace))

            try:
                spec = importlib.util.find_spec(self.workspace.name)
                if spec and spec.origin:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
            except ImportError as e:
                items.append(WorkItem(
                    source=SignalSource.BROKEN_IMPORT,
                    type=SignalType.BUGFIX,
                    title=f"Fix broken imports in {self.workspace.name}",
                    description=f"Import error: {str(e)}",
                    location=str(self.workspace),
                    urgency=80,
                    confidence=0.85,
                    metadata={"error": str(e)},
                ))
            finally:
                sys.path.remove(str(self.workspace))

        except Exception as e:
            logger.debug("collector.imports.failed", error=str(e))

        return items

    def _check_stale_branches(self) -> List[WorkItem]:
        """Check for stale git branches.

        Returns:
            List of WorkItems for stale branches
        """
        items = []

        git_dir = self.workspace / ".git"
        if not git_dir.exists():
            return items

        try:
            # Get list of branches
            result = subprocess.run(
                ["git", "branch", "-v"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    # Parse branch line: "* main    abc1234 Commit message"
                    if "HEAD" in line or "no branch" in line:
                        continue

                    parts = line.strip().split()
                    if len(parts) >= 3:
                        branch_name = parts[1]
                        # Stale if not main/master and not recently committed
                        if branch_name not in ["main", "master", "develop"]:
                            # Check last commit date (would need more parsing)
                            items.append(WorkItem(
                                source=SignalSource.STALE_BRANCH,
                                type=SignalType.MAINTENANCE,
                                title=f"Review and clean up branch: {branch_name}",
                                description=f"Branch appears to be stale. Consider merging or deleting.",
                                urgency=20,
                                confidence=0.5,
                                metadata={"branch": branch_name},
                            ))

        except Exception as e:
            logger.debug("collector.branches.failed", error=str(e))

        return items
