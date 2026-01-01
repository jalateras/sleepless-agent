"""Project orchestration for multi-project autonomous task management."""

from sleepless_agent.orchestration.project_config import (
    ProjectConfig,
    GitHubConfig,
    ProjectGoal,
    load_projects_from_config,
    validate_project_config,
)

__all__ = [
    "ProjectConfig",
    "GitHubConfig",
    "ProjectGoal",
    "load_projects_from_config",
    "validate_project_config",
]
