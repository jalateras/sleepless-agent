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
from sleepless_agent.orchestration.local_collector import LocalSignalCollector
from sleepless_agent.orchestration.github_collector import GitHubSignalCollector
from sleepless_agent.orchestration.task_generator import ProjectTaskGenerator
from sleepless_agent.orchestration.prioritization import (
    CrossProjectPrioritizer,
    ProjectHealth,
    RankedTask,
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

        # Cross-project prioritization
        self.prioritizer = CrossProjectPrioritizer()

        # Signals cache for health tracking
        self._signals_cache: Dict[str, List] = {}

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
        3. Prioritize signals across all projects using goals and health
        4. Submit top-ranked tasks to the task queue

        Returns:
            Dictionary with project_id -> tasks_generated count
        """
        # Step 1: Collect signals from all due projects
        all_signals: Dict[str, List] = {}
        projects_to_check = []

        enabled_projects = self.get_enabled_projects()
        logger.info(
            "orchestrator.analysis_starting",
            enabled_projects=len(enabled_projects),
            project_ids=[p.id for p in enabled_projects],
        )

        for project in enabled_projects:
            if not self.should_check_project(project):
                logger.debug(
                    "orchestrator.project_skipped",
                    project_id=project.id,
                    reason="not_due_for_check",
                    last_check=self._last_check_time.get(project.id),
                    interval_hours=project.check_interval_hours,
                )
                continue

            projects_to_check.append(project)

            try:
                signals = await self._collect_project_signals(project)
                all_signals[project.id] = signals
                self._last_check_time[project.id] = datetime.now(timezone.utc)
                self._signals_cache[project.id] = signals
            except Exception as e:
                logger.error(
                    "orchestrator.project_collection_failed",
                    project_id=project.id,
                    error=str(e),
                )
                all_signals[project.id] = []

        # Step 2: Prioritize across all projects
        ranked_tasks = self.prioritizer.prioritize_signals(
            all_signals,
            self._projects,
        )

        # Step 3: Submit top tasks with deduplication
        results = {p.id: 0 for p in projects_to_check}
        max_tasks_per_project = 1  # Only queue 1 task per project per cycle

        # Get existing pending task descriptions to avoid duplicates
        existing_pending = self.task_queue.get_pending_tasks()
        existing_descriptions = {t.description.lower()[:100] for t in existing_pending}

        # Track descriptions we're adding this cycle
        added_descriptions: set[str] = set()

        for ranked_task in ranked_tasks:
            # Stop if we've hit the per-project limit
            if results[ranked_task.project_id] >= max_tasks_per_project:
                continue

            # Skip if similar task already exists (check first 100 chars)
            desc_key = ranked_task.description.lower()[:100]
            if desc_key in existing_descriptions or desc_key in added_descriptions:
                logger.debug(
                    "orchestrator.task_skipped_duplicate",
                    project_id=ranked_task.project_id,
                    description_preview=ranked_task.description[:50],
                )
                continue

            try:
                # Map priority tier to TaskPriority
                task_priority = self._tier_to_priority(ranked_task.priority_tier)

                self.task_queue.add_task(
                    description=ranked_task.description,
                    priority=task_priority,
                    project_id=ranked_task.project_id,
                    project_name=ranked_task.project_name,
                )
                results[ranked_task.project_id] += 1
                added_descriptions.add(desc_key)
            except Exception as e:
                logger.error(
                    "orchestrator.task_creation_failed",
                    project_id=ranked_task.project_id,
                    error=str(e),
                )

        logger.info(
            "orchestrator.analysis_complete",
            projects_checked=len(projects_to_check),
            total_signals=sum(len(s) for s in all_signals.values()),
            ranked_tasks=len(ranked_tasks),
            tasks_submitted=sum(results.values()),
        )

        return results

    async def _collect_project_signals(self, project: ProjectConfig) -> List:
        """Collect all signals for a project.

        Args:
            project: Project configuration

        Returns:
            List of WorkItems
        """
        logger.debug(
            "orchestrator.collecting_signals",
            project_id=project.id,
            has_github=project.has_github,
        )

        signals = []

        # Collect local signals
        collector = LocalSignalCollector(project)
        signals.extend(collector.collect_all())

        # Collect GitHub signals if configured
        if project.has_github:
            try:
                github_collector = GitHubSignalCollector(project)
                signals.extend(github_collector.collect_all())
            except Exception as e:
                logger.error(
                    "orchestrator.github_collection_failed",
                    project_id=project.id,
                    error=str(e),
                )

        return signals

    def _tier_to_priority(self, tier) -> TaskPriority:
        """Convert priority tier to TaskPriority enum.

        Args:
            tier: PriorityTier enum

        Returns:
            TaskPriority enum value
        """
        from sleepless_agent.orchestration.prioritization import PriorityTier

        # Map PriorityTier to TaskPriority
        # TaskPriority has: SERIOUS (high), THOUGHT (medium), GENERATED (auto)
        tier_mapping = {
            PriorityTier.CRITICAL: TaskPriority.SERIOUS,
            PriorityTier.HIGH: TaskPriority.SERIOUS,
            PriorityTier.MEDIUM: TaskPriority.THOUGHT,
            PriorityTier.LOW: TaskPriority.GENERATED,
        }
        return tier_mapping.get(tier, TaskPriority.GENERATED)

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

        # Get health from prioritizer cache
        health = self.prioritizer.get_project_health(project_id)

        if not health:
            return {
                "status": "unknown",
                "project_id": project.id,
                "name": project.name,
                "priority": project.priority,
                "goals": [{"type": g.type, "target": g.target} for g in project.goals],
                "last_check": self._last_check_time.get(project_id),
            }

        return {
            "status": health.health_tier,
            "project_id": project.id,
            "name": project.name,
            "priority": project.priority,
            "overall_health": health.overall_health,
            "signal_health": health.signal_health,
            "goal_progress": health.goal_progress,
            "issues_count": health.issues_count,
            "stale_count": health.stale_count,
            "last_check": health.last_check or self._last_check_time.get(project_id),
        }

    def get_all_project_health(self) -> List[Dict[str, any]]:
        """Get health status for all projects.

        Returns:
            List of health dictionaries, sorted by priority
        """
        health_data = []

        for project in self.get_enabled_projects():
            health_info = self.get_project_health(project.id)
            health_data.append(health_info)

        # Sort by overall health (poorest first)
        health_data.sort(key=lambda h: h.get("overall_health", 1.0))

        return health_data
