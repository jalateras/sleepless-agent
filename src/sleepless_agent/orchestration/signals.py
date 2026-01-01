"""Signal data models for project analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


class SignalSource(str, Enum):
    """Source of a work signal."""

    TODO = "todo"
    TEST_FAILURE = "test_failure"
    COVERAGE_GAP = "coverage_gap"
    BROKEN_IMPORT = "broken_import"
    STALE_BRANCH = "stale_branch"
    GITHUB_ISSUE = "github_issue"
    GITHUB_PR = "github_pr"
    CI_FAILURE = "ci_failure"
    DEPENDABOT = "dependabot"


class SignalType(str, Enum):
    """Type of work the signal represents."""

    BUGFIX = "bugfix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    TEST = "test"
    DOCUMENTATION = "documentation"
    MAINTENANCE = "maintenance"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class WorkItem:
    """A discovered work item from project analysis."""

    source: SignalSource
    type: SignalType
    title: str
    description: str

    # Location information
    location: Optional[str] = None  # File path, issue URL, etc.
    line_number: Optional[int] = None

    # Metadata
    urgency: int = 0  # 0-100
    confidence: float = 1.0  # 0-1
    age_days: Optional[int] = None

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timestamp
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate WorkItem fields."""
        if self.urgency < 0 or self.urgency > 100:
            raise ValueError(f"urgency must be 0-100, got {self.urgency}")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"confidence must be 0-1, got {self.confidence}")

    @property
    def priority_score(self) -> float:
        """Calculate overall priority score for ranking."""
        score = self.urgency * self.confidence

        # Boost for old items (stale signals)
        if self.age_days:
            if self.age_days > 30:
                score *= 1.5
            elif self.age_days > 14:
                score *= 1.2
            elif self.age_days > 7:
                score *= 1.1

        # Boost for security issues
        if self.type == SignalType.SECURITY:
            score *= 2.0

        # Boost for test failures
        if self.source == SignalSource.TEST_FAILURE:
            score *= 1.5

        return score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "source": self.source.value,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "line_number": self.line_number,
            "urgency": self.urgency,
            "confidence": self.confidence,
            "age_days": self.age_days,
            "priority_score": self.priority_score,
            "discovered_at": self.discovered_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class TODOComment:
    """A TODO/FIXME comment found in code."""

    content: str
    file_path: str
    line_number: int
    prefix: Literal["TODO", "FIXME", "HACK", "XXX", "NOTE"]

    @property
    def age_days(self) -> Optional[int]:
        """Calculate age in days if git info available."""
        # This would require git blame - expensive
        # For now, return None (age will be estimated from file mtime)
        return None


@dataclass
class TestFailure:
    """A failing test."""

    test_name: str
    file_path: str
    error_message: str
    error_type: Optional[str] = None  # AssertionError, ImportError, etc.


@dataclass
class CoverageGap:
    """A coverage gap in the codebase."""

    file_path: str
    coverage_percent: float
    target_percent: float
    gap_percent: float
    lines_uncovered: int
