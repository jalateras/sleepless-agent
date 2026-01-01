"""Project orchestrator for multi-project autonomous task management."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from sleepless_agent.core.queue import TaskQueue
from sleepless_agent.core.models import TaskPriority
from sleepless_agent.monitoring.logging import get_logger

from sleepless_agent.orchestration.project_config import (
    ProjectConfig,
    load_projects_from_config,
)

logger = get_logger(__name__)


class ProjectOrchestrator:
    """Manages multiple projects, analyzes their state, and generates tasks.

    The orchestrator is responsible for:
    1. Loading project configurations from projects.yaml
    2. Periodically analyzing each project's state
    3. Generating tasks based on project goals and signals
    4. Prioritizing work across all projects
    5. Providing project health visibility
    """

    def __init__(
        self,
        *,
        config_path: str | Path,
        task_queue: TaskQueue,
        default_priority: TaskPriority = TaskPriority.THOUGHT,
        check_interval_seconds: int = 300,  # 5 minutes default
    ):
        """Initialize the project orchestrator.

        Args:
            config_path: Path to projects.yaml configuration file
            task_queue: Task queue for submitting generated tasks
            default_priority: Default priority for generated tasks
            check_interval_seconds: How often to check all projects
        """
        self.config_path = Path(config_path).expanduser()
        self.task_queue = task_queue
        self.default_priority = default_priority
        self.check_interval_seconds = check_interval_seconds

        # Project registry
        self._projects: Dict[str, ProjectConfig] = {}
        self._last_check_time: Dict[str, datetime] = {}

        # Load initial configuration
        self._load_projects()

        logger.info(
            "orchestrator.init",
            config_path=str(self.config_path),
            projects_count=len(self._projects),
        )

    def _load_projects(self) -> None:
        """Load or reload project configurations from file."""
        try:
            projects = load_projects_from_config(self.config_path)
            self._projects = {p.id: p for p in projects}
            logger.info(
                "orchestrator.projects_loaded",
                count=len(self._projects),
                project_ids=list(self._projects.keys()),
            )
        except FileNotFoundError:
            logger.warning("orchestrator.config_not_found", path=str(self.config_path))
            self._projects = {}
        except Exception as e:
            logger.error("orchestrator.config_load_failed", error=str(e))
            self._projects = {}

    def reload_projects(self) -> None:
        """Reload project configurations from file."""
        logger.info("orchestrator.reload_projects")
        self._load_projects()

    def get_project(self, project_id: str) -> Optional[ProjectConfig]:
        """Get a project configuration by ID.

        Args:
            project_id: Project identifier

        Returns:
            ProjectConfig if found, None otherwise
        """
        return self._projects.get(project_id)

    def get_all_projects(self) -> List[ProjectConfig]:
        """Get all enabled project configurations.

        Returns:
            List of ProjectConfig objects
        """
        return list(self._projects.values())

    def get_enabled_projects(self) -> List[ProjectConfig]:
        """Get enabled projects sorted by priority.

        Returns:
            List of enabled ProjectConfig objects, sorted by priority weight
        """
        projects = [p for p in self._projects.values() if p.enabled]
        return sorted(projects, key=lambda p: -p.priority_weight)

    def should_check_project(self, project: ProjectConfig) -> bool:
        """Determine if a project is due for analysis.

        Args:
            project: Project configuration

        Returns:
            True if project should be checked now
        """
        if project.id not in self._last_check_time:
            return True

        last_check = self._last_check_time[project.id]
        interval = timedelta(hours=project.check_interval_hours)

        return datetime.now(timezone.utc).replace(tzinfo=None) - last_check.replace(tzinfo=None) > interval

    async def analyze_all_projects(self) -> Dict[str, int]:
        """Analyze all projects that are due for checking.

        This is the main entry point called periodically by the daemon.
        It will:
        1. Identify projects due for analysis
        2. Collect signals from each project (local + GitHub)
        3. Generate tasks based on signals and goals
        4. Submit tasks to the task queue

        Returns:
            Dictionary with project_id -> tasks_generated count
        """
        results = {}

        for project in self.get_enabled_projects():
            if not self.should_check_project(project):
                continue

            try:
                tasks_created = await self._analyze_project(project)
                results[project.id] = tasks_created
                self._last_check_time[project.id] = datetime.now(timezone.utc)
            except Exception as e:
                logger.error(
                    "orchestrator.project_analysis_failed",
                    project_id=project.id,
                    error=str(e),
                )
                results[project.id] = 0

        return results

    async def _analyze_project(self, project: ProjectConfig) -> int:
        """Analyze a single project and generate tasks.

        Args:
            project: Project configuration

        Returns:
            Number of tasks generated
        """
        logger.debug(
            "orchestrator.analyzing_project",
            project_id=project.id,
            has_github=project.has_github,
        )

        # TODO: Implement actual signal collection
        # This will be done in future iterations:
        # - Scan for TODO/FIXME comments
        # - Check test failures
        # - Analyze coverage gaps
        # - Fetch GitHub issues/PRs
        # - Check CI status

        # For now, just log that we're checking
        logger.info(
            "orchestrator.project_checked",
            project_id=project.id,
            goals_count=len(project.goals),
        )

        return 0

    def get_project_health(self, project_id: str) -> Dict[str, any]:
        """Get health status for a specific project.

        Args:
            project_id: Project identifier

        Returns:
            Dictionary with health information
        """
        project = self.get_project(project_id)
        if not project:
            return {"status": "unknown", "error": "Project not found"}

        # TODO: Implement actual health reporting
        # This will be done in future iterations:
        # - Goal progress tracking
        # - Recent activity summary
        # - Active signals count
        # - Overall health status

        return {
            "status": "healthy",
            "project_id": project.id,
            "name": project.name,
            "priority": project.priority,
            "goals": [{"type": g.type, "target": g.target} for g in project.goals],
            "last_check": self._last_check_time.get(project.id),
        }

    def get_all_project_health(self) -> List[Dict[str, any]]:
        """Get health status for all projects.

        Returns:
            List of health dictionaries, sorted by priority
        """
        projects = self.get_enabled_projects()
        return [self.get_project_health(p.id) for p in projects]
