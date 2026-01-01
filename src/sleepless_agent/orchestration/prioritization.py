"""Cross-project prioritization for intelligent task selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from enum import Enum

from sleepless_agent.monitoring.logging import get_logger
from sleepless_agent.orchestration.project_config import ProjectConfig, ProjectGoal
from sleepless_agent.orchestration.signals import WorkItem, SignalType

logger = get_logger(__name__)


class PriorityTier(Enum):
    """Priority tier for task classification."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class RankedTask:
    """A task with priority information."""
    description: str
    project_id: str
    project_name: str
    signal: WorkItem
    priority_score: float
    priority_tier: PriorityTier
    goal_alignment: float  # 0-1
    health_impact: float  # 0-1

    def __lt__(self, other: 'RankedTask') -> bool:
        """Sort by priority score (highest first)."""
        return self.priority_score > other.priority_score


@dataclass
class ProjectHealth:
    """Health status of a project."""
    project_id: str
    overall_health: float  # 0-1
    signal_health: float  # 0-1, based on active signals
    goal_progress: Dict[str, float]  # goal_type -> progress 0-1
    last_check: Optional[datetime] = None
    issues_count: int = 0
    stale_count: int = 0

    @property
    def health_tier(self) -> str:
        """Get health tier category."""
        if self.overall_health >= 0.8:
            return "excellent"
        elif self.overall_health >= 0.6:
            return "good"
        elif self.overall_health >= 0.4:
            return "fair"
        else:
            return "poor"


class GoalAlignmentScorer:
    """Calculates goal alignment for signals."""

    def score_alignment(self, signal: WorkItem, goals: List[ProjectGoal]) -> float:
        """Calculate how well a signal aligns with project goals.

        Args:
            signal: WorkItem to score
            goals: List of project goals

        Returns:
            Alignment score 0-1
        """
        if not goals:
            return 0.5  # Neutral if no goals defined

        best_score = 0.0

        for goal in goals:
            score = self._score_for_goal(signal, goal)
            best_score = max(best_score, score)

        return min(best_score, 1.0)

    def _score_for_goal(self, signal: WorkItem, goal: ProjectGoal) -> float:
        """Score alignment with a specific goal.

        Args:
            signal: WorkItem to score
            goal: Project goal

        Returns:
            Alignment score 0-1
        """
        base_score = 0.3  # Base alignment

        # Type-based matching
        type_match_scores = {
            SignalType.TEST: {"testing": 0.9, "coverage": 0.8},
            SignalType.BUGFIX: {"quality": 0.8, "stability": 0.7},
            SignalType.FEATURE: {"feature": 0.9},
            SignalType.DOCUMENTATION: {"documentation": 0.9},
            SignalType.PERFORMANCE: {"performance": 0.9},
            SignalType.SECURITY: {"security": 0.9},
            SignalType.REFACTOR: {"tech_debt": 0.8, "quality": 0.7},
        }

        if goal.type in type_match_scores.get(signal.type, {}):
            base_score += type_match_scores[signal.type][goal.type]

        # Keyword matching in description
        if goal.keywords:
            desc_lower = signal.description.lower()
            title_lower = signal.title.lower()
            for keyword in goal.keywords:
                if keyword.lower() in desc_lower or keyword.lower() in title_lower:
                    base_score += 0.1

        return min(base_score, 1.0)


class ConstraintValidator:
    """Validates signals against project constraints."""

    def validate(self, signal: WorkItem, constraints: List[str]) -> tuple[bool, Optional[str]]:
        """Check if a signal violates project constraints.

        Args:
            signal: WorkItem to validate
            constraints: List of constraint strings

        Returns:
            (is_valid, violation_reason) tuple
        """
        if not constraints:
            return True, None

        for constraint in constraints:
            violates, reason = self._check_constraint(signal, constraint)
            if violates:
                return False, reason

        return True, None

    def _check_constraint(self, signal: WorkItem, constraint: str) -> tuple[bool, Optional[str]]:
        """Check a single constraint.

        Args:
            signal: WorkItem to validate
            constraint: Constraint string

        Returns:
            (violates, reason) tuple
        """
        constraint_lower = constraint.lower()

        # Breaking changes protection
        if "breaking" in constraint_lower and "config" in constraint_lower:
            if signal.location and "config.yaml" in signal.location:
                return True, f"Breaking change to config.yaml not allowed"

        # Test requirement
        if "without tests" in constraint_lower or "no test" in constraint_lower:
            if signal.type == SignalType.TEST:
                return True, "Cannot add tests (constraint violation)"

        # Skip specific files
        if "skip" in constraint_lower and "file" in constraint_lower:
            parts = constraint_lower.split()
            if len(parts) >= 3:
                file_pattern = parts[2]
                if signal.location and file_pattern in signal.location:
                    return True, f"Skipping {file_pattern} files"

        return False, None


class ProjectHealthCalculator:
    """Calculates project health metrics."""

    def calculate(
        self,
        project: ProjectConfig,
        signals: List[WorkItem],
        goal_progress: Optional[Dict[str, float]] = None,
    ) -> ProjectHealth:
        """Calculate overall project health.

        Args:
            project: Project configuration
            signals: Current signals for the project
            goal_progress: Optional goal progress data

        Returns:
            ProjectHealth object
        """
        # Signal health: fewer stale/issues = better
        signal_health = self._calculate_signal_health(signals)

        # Goal health: average of goal progress
        goal_health = self._calculate_goal_health(project.goals, goal_progress)

        # Overall health: weighted average
        overall_health = (signal_health * 0.6) + (goal_health * 0.4)

        # Count issues
        issues_count = sum(1 for s in signals if s.type == SignalType.BUGFIX)
        stale_count = sum(1 for s in signals if s.age_days and s.age_days > 30)

        return ProjectHealth(
            project_id=project.id,
            overall_health=overall_health,
            signal_health=signal_health,
            goal_progress=goal_progress or {},
            last_check=datetime.now(timezone.utc),
            issues_count=issues_count,
            stale_count=stale_count,
        )

    def _calculate_signal_health(self, signals: List[WorkItem]) -> float:
        """Calculate health based on signal composition.

        Args:
            signals: List of work signals

        Returns:
            Health score 0-1
        """
        if not signals:
            return 1.0  # No signals = healthy

        # Penalize bugs and security issues
        penalty_signals = [s for s in signals if s.type in (SignalType.BUGFIX, SignalType.SECURITY)]
        penalty = min(len(penalty_signals) * 0.1, 0.5)

        # Penalize stale items
        stale_signals = [s for s in signals if s.age_days and s.age_days > 30]
        stale_penalty = min(len(stale_signals) * 0.05, 0.3)

        return max(1.0 - penalty - stale_penalty, 0.0)

    def _calculate_goal_health(
        self,
        goals: List[ProjectGoal],
        progress: Optional[Dict[str, float]]
    ) -> float:
        """Calculate health based on goal progress.

        Args:
            goals: List of project goals
            progress: Goal progress data

        Returns:
            Health score 0-1
        """
        if not goals:
            return 1.0  # No goals = neutral

        if not progress:
            return 0.7  # Unknown progress

        scores = []
        for goal in goals:
            goal_key = f"{goal.type}:{goal.metric or 'default'}"
            goal_progress = progress.get(goal_key, 0.5)
            scores.append(goal_progress)

        return sum(scores) / len(scores) if scores else 0.7


class CrossProjectPrioritizer:
    """Prioritizes tasks across multiple projects."""

    def __init__(self):
        """Initialize the prioritizer."""
        self.goal_scorer = GoalAlignmentScorer()
        self.constraint_validator = ConstraintValidator()
        self.health_calculator = ProjectHealthCalculator()

        # Project health cache
        self._health_cache: Dict[str, ProjectHealth] = {}

    def prioritize_signals(
        self,
        project_signals: Dict[str, List[WorkItem]],
        projects: Dict[str, ProjectConfig],
    ) -> List[RankedTask]:
        """Prioritize signals across all projects.

        Args:
            project_signals: Dict of project_id -> list of signals
            projects: Dict of project_id -> ProjectConfig

        Returns:
            List of RankedTask, sorted by priority
        """
        ranked_tasks = []

        for project_id, signals in project_signals.items():
            project = projects.get(project_id)
            if not project:
                continue

            # Calculate project health
            health = self.health_calculator.calculate(project, signals)
            self._health_cache[project_id] = health

            for signal in signals:
                # Validate constraints
                is_valid, violation = self.constraint_validator.validate(
                    signal, project.constraints
                )
                if not is_valid:
                    logger.debug(
                        "prioritizer.signal_filtered",
                        project_id=project_id,
                        reason=violation,
                    )
                    continue

                # Calculate scores
                goal_alignment = self.goal_scorer.score_alignment(signal, project.goals)
                health_impact = self._calculate_health_impact(signal, health)

                # Calculate final priority score
                priority_score = self._calculate_priority_score(
                    signal, goal_alignment, health_impact, project.priority_weight
                )

                # Generate task description
                description = self._generate_task_description(signal, project)

                # Determine priority tier
                priority_tier = self._classify_tier(priority_score, signal)

                ranked_tasks.append(RankedTask(
                    description=description,
                    project_id=project_id,
                    project_name=project.name,
                    signal=signal,
                    priority_score=priority_score,
                    priority_tier=priority_tier,
                    goal_alignment=goal_alignment,
                    health_impact=health_impact,
                ))

        # Sort by priority score
        ranked_tasks.sort()

        logger.info(
            "prioritizer.ranked",
            total_signals=sum(len(s) for s in project_signals.values()),
            valid_tasks=len(ranked_tasks),
        )

        return ranked_tasks

    def _calculate_priority_score(
        self,
        signal: WorkItem,
        goal_alignment: float,
        health_impact: float,
        project_weight: int,
    ) -> float:
        """Calculate final priority score.

        Args:
            signal: WorkItem to score
            goal_alignment: Goal alignment score 0-1
            health_impact: Health impact score 0-1
            project_weight: Project priority weight

        Returns:
            Final priority score
        """
        # Base score from signal
        base_score = signal.priority_score

        # Apply project weight (1-10, normalized)
        weight_multiplier = project_weight / 5.0

        # Goal alignment boost
        alignment_boost = 1.0 + (goal_alignment * 0.5)

        # Health impact boost
        health_boost = 1.0 + (health_impact * 0.3)

        final_score = base_score * weight_multiplier * alignment_boost * health_boost

        return final_score

    def _calculate_health_impact(self, signal: WorkItem, health: ProjectHealth) -> float:
        """Calculate how much this signal impacts project health.

        Args:
            signal: WorkItem to evaluate
            health: Current project health

        Returns:
            Health impact score 0-1
        """
        # High urgency signals have more health impact
        urgency_impact = signal.urgency / 100.0

        # Bugs and security issues have higher impact
        type_impact = {
            SignalType.SECURITY: 0.9,
            SignalType.BUGFIX: 0.7,
            SignalType.PERFORMANCE: 0.5,
        }.get(signal.type, 0.3)

        # Stale signals have higher impact on poor health projects
        staleness_impact = 0.0
        if signal.age_days and signal.age_days > 30:
            staleness_impact = 0.3
            if health.overall_health < 0.5:
                staleness_impact += 0.2

        return min(urgency_impact + type_impact + staleness_impact, 1.0)

    def _classify_tier(self, priority_score: float, signal: WorkItem) -> PriorityTier:
        """Classify signal into priority tier.

        Args:
            priority_score: Calculated priority score
            signal: WorkItem for context

        Returns:
            PriorityTier enum
        """
        # Security always at least HIGH
        if signal.type == SignalType.SECURITY:
            return PriorityTier.CRITICAL if priority_score > 100 else PriorityTier.HIGH

        if priority_score > 120:
            return PriorityTier.CRITICAL
        elif priority_score > 80:
            return PriorityTier.HIGH
        elif priority_score > 40:
            return PriorityTier.MEDIUM
        else:
            return PriorityTier.LOW

    def _generate_task_description(self, signal: WorkItem, project: ProjectConfig) -> str:
        """Generate task description with context.

        Args:
            signal: WorkItem to convert
            project: Project configuration

        Returns:
            Task description string
        """
        parts = []

        # Action verb
        verbs = {
            SignalType.BUGFIX: "Fix",
            SignalType.FEATURE: "Implement",
            SignalType.REFACTOR: "Refactor",
            SignalType.TEST: "Add tests for",
            SignalType.DOCUMENTATION: "Document",
            SignalType.MAINTENANCE: "Maintain",
            SignalType.SECURITY: "Fix security issue:",
            SignalType.PERFORMANCE: "Optimize",
        }
        parts.append(verbs.get(signal.type, "Address"))

        # Main description
        if signal.source.value == "todo":
            parts.append(signal.description)
        else:
            parts.append(signal.title)

        # Location
        if signal.location:
            parts.append(f"\nLocation: {signal.location}")
            if signal.line_number:
                parts.append(f":{signal.line_number}")

        # Project context
        parts.append(f"\nProject: {project.name}")

        return " ".join(parts)

    def get_project_health(self, project_id: str) -> Optional[ProjectHealth]:
        """Get cached health for a project.

        Args:
            project_id: Project identifier

        Returns:
            ProjectHealth if available, None otherwise
        """
        return self._health_cache.get(project_id)

    def get_all_project_health(self) -> Dict[str, ProjectHealth]:
        """Get all cached project health.

        Returns:
            Dict of project_id -> ProjectHealth
        """
        return self._health_cache.copy()

    def clear_health_cache(self) -> None:
        """Clear the health cache."""
        self._health_cache.clear()
