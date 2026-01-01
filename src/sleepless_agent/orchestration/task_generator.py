"""Task generation from work signals."""

from __future__ import annotations

from typing import List, Optional

from sleepless_agent.core.models import TaskPriority
from sleepless_agent.monitoring.logging import get_logger
from sleepless_agent.orchestration.project_config import ProjectConfig, ProjectGoal
from sleepless_agent.orchestration.signals import WorkItem, SignalType

logger = get_logger(__name__)


class ProjectTaskGenerator:
    """Converts work signals into well-scoped tasks with project context."""

    def __init__(self, project: ProjectConfig):
        """Initialize task generator for a project.

        Args:
            project: Project configuration
        """
        self.project = project

    def generate_tasks(self, signals: List[WorkItem]) -> List[str]:
        """Generate task descriptions from work signals.

        Args:
            signals: List of WorkItems to convert

        Returns:
            List of task description strings
        """
        tasks = []

        # Filter signals by project constraints
        filtered_signals = self._filter_by_constraints(signals)

        # Rank signals by priority
        ranked_signals = sorted(
            filtered_signals,
            key=lambda s: s.priority_score,
            reverse=True,
        )

        # Generate tasks from top signals
        for signal in ranked_signals:
            # Limit number of tasks per project per cycle
            if len(tasks) >= 5:
                break

            task = self._signal_to_task(signal)
            if task:
                tasks.append(task)

        logger.info(
            "task_generator.generated",
            project_id=self.project.id,
            signals_input=len(signals),
            tasks_output=len(tasks),
        )

        return tasks

    def _filter_by_constraints(self, signals: List[WorkItem]) -> List[WorkItem]:
        """Filter signals that violate project constraints.

        Args:
            signals: List of WorkItems

        Returns:
            Filtered list of WorkItems
        """
        filtered = []

        for signal in signals:
            # Check against constraints
            violates_constraint = False
            for constraint in self.project.constraints:
                constraint_lower = constraint.lower()
                # Simple keyword matching
                if "breaking" in constraint_lower and "config.yaml" in constraint_lower:
                    if signal.location and "config.yaml" in signal.location:
                        violates_constraint = True
                        break
                if "test" in constraint_lower and "without" in constraint_lower:
                    if signal.type == SignalType.TEST:
                        violates_constraint = True
                        break

            if not violates_constraint:
                filtered.append(signal)

        return filtered

    def _signal_to_task(self, signal: WorkItem) -> Optional[str]:
        """Convert a signal to a task description.

        Args:
            signal: WorkItem to convert

        Returns:
            Task description string or None
        """
        # Get goal alignment info
        goal_info = self._get_goal_alignment(signal)

        # Build task description
        description_parts = []

        # Add action verb based on signal type
        action = self._get_action_verb(signal)
        description_parts.append(action)

        # Add the main work
        if signal.source.value == "todo":
            description_parts.append(signal.description)
        else:
            description_parts.append(signal.title)

        # Add context
        if signal.location:
            description_parts.append(f"\nLocation: {signal.location}")
            if signal.line_number:
                description_parts.append(f":{signal.line_number}")

        # Add goal alignment
        if goal_info:
            description_parts.append(f"\nGoal: {goal_info}")

        # Add project context
        description_parts.append(f"\nProject: {self.project.name}")

        # Add metadata if available
        if signal.metadata.get("full_line"):
            description_parts.append(f"\nContext:\n{signal.metadata['full_line']}")

        return " ".join(description_parts)

    def _get_action_verb(self, signal: WorkItem) -> str:
        """Get appropriate action verb for signal type.

        Args:
            signal: WorkItem

        Returns:
            Action verb string
        """
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
        return verbs.get(signal.type, "Address")

    def _get_goal_alignment(self, signal: WorkItem) -> Optional[str]:
        """Find which project goal this signal aligns with.

        Args:
            signal: WorkItem

        Returns:
            Goal description or None
        """
        for goal in self.project.goals:
            # Match signal type to goal type
            if signal.type == SignalType.TEST and goal.type == "testing":
                return f"Improve test coverage (target: {goal.target}%)"
            elif signal.type == SignalType.TEST and goal.type == "coverage":
                return f"Improve coverage to {goal.target}%"
            elif signal.type == SignalType.PERFORMANCE and goal.type == "performance":
                return f"Improve {goal.metric} performance (target: {goal.target_ms}ms)"
            elif signal.type == SignalType.FEATURE and goal.type == "feature":
                return goal.description
            elif signal.type == SignalType.DOCUMENTATION and goal.type == "documentation":
                areas = ", ".join(goal.areas or [])
                return f"Improve documentation: {areas}"

        return None
