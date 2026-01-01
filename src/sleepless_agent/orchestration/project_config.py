"""Project configuration dataclasses and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator

from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)


class GitHubConfig(BaseModel):
    """GitHub repository configuration for a project."""

    repo: str = Field(..., description="GitHub repo in 'owner/repo' format")
    default_branch: str = Field("main", description="Default branch name")
    sync_mode: Literal["pull", "push", "bidirectional"] = Field(
        "pull", description="Git sync mode"
    )
    check_dependencies: bool = Field(True, description="Check for dependency updates")
    auth_token: Optional[str] = Field(None, description="GitHub auth token (from env var)")

    @validator("repo")
    def validate_repo_format(cls, v: str) -> str:
        """Validate repo is in 'owner/repo' format."""
        if "/" not in v or v.count("/") != 1:
            raise ValueError(f"Invalid repo format: {v}. Expected 'owner/repo'")
        parts = v.split("/")
        if not parts[0] or not parts[1]:
            raise ValueError(f"Invalid repo format: {v}. Owner and repo name required")
        return v

    @validator("auth_token", pre=True, always=True)
    def resolve_auth_token(cls, v: Optional[str]) -> Optional[str]:
        """Resolve auth token from value or environment variable."""
        if v is not None:
            return v
        # Check for common GitHub token env vars
        for env_var in ["GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT"]:
            token = os.environ.get(env_var)
            if token:
                logger.debug("github.token.from_env", env_var=env_var)
                return token
        return None


class ProjectGoal(BaseModel):
    """A goal for a project that guides task generation."""

    type: Literal["coverage", "documentation", "performance", "feature", "testing"] = Field(
        ..., description="Type of goal"
    )
    target: Optional[float] = Field(None, description="Target value for numeric goals")
    current: Optional[float] = Field(None, description="Current value for progress tracking")
    description: Optional[str] = Field(None, description="Goal description for non-numeric goals")
    metric: Optional[str] = Field(None, description="Metric name for performance goals")
    areas: Optional[List[str]] = Field(None, description="Areas for documentation goals")

    @validator("areas")
    def validate_documentation_areas(cls, v: Optional[List[str]], values: Dict[str, Any]) -> Optional[List[str]]:
        """Validate documentation goals have areas specified."""
        if values.get("type") == "documentation" and not v:
            raise ValueError("Documentation goals must specify 'areas'")
        return v


class ProjectConfig(BaseModel):
    """Configuration for a single project in the orchestrator."""

    id: str = Field(..., description="Unique project identifier")
    name: str = Field(..., description="Human-readable project name")
    local_path: str = Field(..., description="Absolute path to local workspace")
    github: Optional[GitHubConfig] = Field(None, description="GitHub integration (optional)")
    goals: List[ProjectGoal] = Field(default_factory=list, description="Project goals")
    constraints: List[str] = Field(default_factory=list, description="Constraints on what work to do")
    check_interval_hours: int = Field(4, description="Hours between project checks")
    priority: Literal["low", "medium", "high", "urgent"] = Field("medium", description="Project priority")
    enabled: bool = Field(True, description="Whether this project is active")

    @validator("local_path")
    def validate_local_path(cls, v: str) -> str:
        """Validate local path exists and is absolute."""
        path = Path(v).expanduser().resolve()
        if not path.exists():
            logger.warning("project.path.not_found", path=str(path))
            # Don't fail - project might be created later or on different machine
        return str(path)

    @validator("check_interval_hours")
    def validate_check_interval(cls, v: int) -> int:
        """Ensure check interval is reasonable."""
        if v < 1:
            raise ValueError("check_interval_hours must be at least 1")
        if v > 168:  # 1 week
            raise ValueError("check_interval_hours must be less than 168 (1 week)")
        return v

    @property
    def priority_weight(self) -> int:
        """Get numeric weight for prioritization."""
        weights = {"low": 1, "medium": 5, "high": 10, "urgent": 20}
        return weights[self.priority]

    @property
    def has_github(self) -> bool:
        """Check if project has GitHub integration."""
        return self.github is not None

    @property
    def is_local_only(self) -> bool:
        """Check if project is local-only (no GitHub)."""
        return self.github is None


class OrchestratorConfig(BaseModel):
    """Top-level configuration for the project orchestrator."""

    version: str = Field("1.0", description="Config schema version")
    projects: List[ProjectConfig] = Field(default_factory=list, description="Project configurations")

    @validator("projects")
    def validate_project_ids(cls, v: List[ProjectConfig]) -> List[ProjectConfig]:
        """Ensure project IDs are unique."""
        ids = [p.id for p in v]
        if len(ids) != len(set(ids)):
            duplicates = [id for id in ids if ids.count(id) > 1]
            raise ValueError(f"Duplicate project IDs: {set(duplicates)}")
        return v

    @validator("projects")
    def validate_enabled_projects(cls, v: List[ProjectConfig]) -> List[ProjectConfig]:
        """Log warning if all projects are disabled."""
        if v and not any(p.enabled for p in v):
            logger.warning("orchestrator.all_projects_disabled")
        return v


def load_projects_from_config(config_path: str | Path) -> List[ProjectConfig]:
    """Load project configurations from a YAML file.

    Args:
        config_path: Path to projects.yaml configuration file

    Returns:
        List of validated ProjectConfig objects

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    import yaml

    config_file = Path(config_path).expanduser()
    if not config_file.exists():
        raise FileNotFoundError(f"Project config not found: {config_file}")

    try:
        with open(config_file) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_file}: {e}")

    try:
        config = OrchestratorConfig(**data)
    except Exception as e:
        raise ValueError(f"Config validation failed: {e}")

    # Filter to enabled projects
    enabled_projects = [p for p in config.projects if p.enabled]
    logger.info(
        "orchestrator.projects_loaded",
        total=len(config.projects),
        enabled=len(enabled_projects),
    )

    return enabled_projects


def validate_project_config(config_path: str | Path) -> bool:
    """Validate a project configuration file without loading.

    Args:
        config_path: Path to projects.yaml configuration file

    Returns:
        True if valid

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    load_projects_from_config(config_path)
    return True
