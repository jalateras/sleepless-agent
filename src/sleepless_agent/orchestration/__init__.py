"""Project orchestration for multi-project autonomous task management."""

from sleepless_agent.orchestration.project_config import (
    ProjectConfig,
    GitHubConfig,
    ProjectGoal,
    load_projects_from_config,
    validate_project_config,
)
from sleepless_agent.orchestration.orchestrator import ProjectOrchestrator
from sleepless_agent.orchestration.signals import (
    WorkItem,
    SignalSource,
    SignalType,
)
from sleepless_agent.orchestration.local_collector import LocalSignalCollector
from sleepless_agent.orchestration.github_collector import GitHubSignalCollector
from sleepless_agent.orchestration.task_generator import ProjectTaskGenerator
from sleepless_agent.orchestration.prioritization import (
    CrossProjectPrioritizer,
    GoalAlignmentScorer,
    ConstraintValidator,
    ProjectHealthCalculator,
    ProjectHealth,
    RankedTask,
    PriorityTier,
)

__all__ = [
    "ProjectConfig",
    "GitHubConfig",
    "ProjectGoal",
    "load_projects_from_config",
    "validate_project_config",
    "ProjectOrchestrator",
    "WorkItem",
    "SignalSource",
    "SignalType",
    "LocalSignalCollector",
    "GitHubSignalCollector",
    "ProjectTaskGenerator",
    "CrossProjectPrioritizer",
    "GoalAlignmentScorer",
    "ConstraintValidator",
    "ProjectHealthCalculator",
    "ProjectHealth",
    "RankedTask",
    "PriorityTier",
]
