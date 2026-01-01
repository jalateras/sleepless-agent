"""Auto-task generation mechanism that creates tasks when usage is below threshold"""

import json
import random
from datetime import datetime, timezone
from typing import Optional, TypeAlias

from claude_agent_sdk import (
    AssistantMessage,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ClaudeAgentOptions,
    ProcessError,
    ResultMessage,
    TextBlock,
    query,
)
from sqlalchemy.orm import Session

from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)

from sleepless_agent.core.models import Task, TaskPriority, TaskStatus, TaskType, GenerationHistory
from typing import TypeAlias

from sleepless_agent.scheduling.scheduler import BudgetManager
from sleepless_agent.storage.feedback import FeedbackStore
from sleepless_agent.utils.config import ConfigNode

AutoGenerationConfig: TypeAlias = ConfigNode
AutoTaskPromptConfig: TypeAlias = ConfigNode


class AutoTaskGenerator:
    """Generate tasks automatically when Claude Code usage is below configured threshold"""

    def __init__(
        self,
        db_session: Session,
        config: AutoGenerationConfig,
        budget_manager: BudgetManager,
        default_model: str,
        usage_command: str,
        threshold_day: float,
        threshold_night: float,
        night_start_hour: int = 20,
        night_end_hour: int = 8,
        feedback_store: Optional[FeedbackStore] = None,
    ):
        """Initialize auto-generator with database session and config"""
        self.session = db_session
        self.config = config
        self.budget_manager = budget_manager
        self.default_model = default_model
        self.usage_command = usage_command
        self.threshold_day = threshold_day
        self.threshold_night = threshold_night
        self.night_start_hour = night_start_hour
        self.night_end_hour = night_end_hour
        self.feedback_store = feedback_store

        # Validate prompt weights sum to approximately 1.0
        if self.config.prompts:
            total_weight = sum(float(p.weight) for p in self.config.prompts if p.weight is not None)
            assert abs(total_weight - 1.0) < 0.01, f"Prompt weights must sum to 1.0, got {total_weight:.3f}"
            logger.debug("autogen.weights.validated", total_weight=total_weight)

        # Track generation metadata
        self.last_generation_time: Optional[datetime] = None
        self._last_generation_source: Optional[str] = None

        # Cache for feedback weights (refreshed periodically)
        self._feedback_weights: dict[str, float] = {}
        self._feedback_weights_updated: Optional[datetime] = None

    async def check_and_generate(self) -> bool:
        """Check if conditions are met and generate a task if possible"""
        if not self.config.enabled:
            logger.debug("autogen.disabled")
            return False

        # Check usage threshold and ceiling
        if not self._should_generate():
            logger.debug("autogen.should_not_generate")
            return False

        # Try to generate a task
        logger.debug("autogen.attempting_generation")
        task = await self._generate_task()
        if task:
            logger.info("autogen.task.created", task_id=task.id, preview=task.description[:80], source=self._last_generation_source)
            return True

        logger.warning("autogen.generation_failed")
        return False

    def _should_generate(self) -> bool:
        """Check if usage is below pause threshold (use time-based thresholds)"""
        from sleepless_agent.monitoring.pro_plan_usage import ProPlanUsageChecker
        from sleepless_agent.scheduling.time_utils import is_nighttime

        try:
            checker = ProPlanUsageChecker(command=self.usage_command)
            threshold = self.threshold_night if is_nighttime(night_start_hour=self.night_start_hour, night_end_hour=self.night_end_hour) else self.threshold_day
            should_pause, _ = checker.check_should_pause(threshold_percent=threshold)

            # Generate only when NOT paused (i.e., usage < threshold)
            return not should_pause
        except Exception as e:
            logger.error("autogen.usage_check.failed", error=str(e))
            # Don't generate on error - fail safe
            return False

    def _determine_generation_mode(self, task_count: int) -> str:
        """Determine generation mode based on current task count

        Args:
            task_count: Number of pending + in_progress tasks

        Returns:
            Generation mode: "refine_focused", "balanced", or "new_friendly"
        """
        if task_count >= 5:
            return "refine_focused"
        elif task_count >= 2:
            return "balanced"
        else:
            return "new_friendly"

    def _sample_recent_tasks(self, limit: int = 5) -> list[Task]:
        """Sample recent tasks (completed or in_progress) to understand recent work

        Args:
            limit: Maximum number of tasks to sample

        Returns:
            List of recent tasks
        """
        tasks = self.session.query(Task).filter(
            Task.status.in_([TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS, TaskStatus.PENDING])
        ).order_by(Task.created_at.desc()).limit(limit).all()
        return tasks

    def _analyze_task_readmes(self, tasks: list[Task]) -> dict:
        """Analyze README files from tasks to extract status and recommendations

        Args:
            tasks: List of tasks to analyze

        Returns:
            Dictionary with aggregated information
        """
        from pathlib import Path
        import re

        analysis = {
            'partial_or_incomplete': [],
            'outstanding_items': [],
            'recommendations': [],
        }

        workspace_root = Path("./workspace")
        tasks_dir = workspace_root / "tasks"

        if not tasks_dir.exists():
            return analysis

        for task in tasks:
            # Find task workspace
            task_workspace = None
            for item in tasks_dir.iterdir():
                if item.is_dir() and item.name.startswith(f"{task.id}_"):
                    task_workspace = item
                    break

            if not task_workspace:
                continue

            readme_path = task_workspace / "README.md"
            if not readme_path.exists():
                continue

            try:
                content = readme_path.read_text()

                # Extract status
                status_match = re.search(r'## Status:\s*(\w+)', content)
                if status_match:
                    status = status_match.group(1)
                    if status in ['PARTIAL', 'INCOMPLETE']:
                        analysis['partial_or_incomplete'].append({
                            'task_id': task.id,
                            'description': task.description[:100],
                            'status': status
                        })

                # Extract outstanding items
                outstanding_section = re.search(r'## Outstanding Items\n(.*?)(?=##|$)', content, re.DOTALL)
                if outstanding_section:
                    items_text = outstanding_section.group(1).strip()
                    if items_text and items_text != '(None)':
                        # Extract list items
                        items = re.findall(r'^[-*]\s+(.+)$', items_text, re.MULTILINE)
                        analysis['outstanding_items'].extend(items[:3])  # Limit to 3 items

                # Extract recommendations
                rec_section = re.search(r'## Recommendations\n(.*?)(?=##|$)', content, re.DOTALL)
                if rec_section:
                    rec_text = rec_section.group(1).strip()
                    if rec_text and rec_text != '(None)':
                        # Extract list items
                        recs = re.findall(r'^[-*]\s+(.+)$', rec_text, re.MULTILINE)
                        analysis['recommendations'].extend(recs[:3])  # Limit to 3 items

            except Exception as e:
                logger.debug("autogen.readme_analysis.failed", task_id=task.id, error=str(e))
                continue

        return analysis

    def _gather_codebase_context(self) -> dict:
        """Gather codebase context for informed task generation

        Returns:
            Dictionary with context information
        """
        # Count current tasks
        pending_tasks = self.session.query(Task).filter(Task.status == TaskStatus.PENDING).count()
        in_progress_tasks = self.session.query(Task).filter(Task.status == TaskStatus.IN_PROGRESS).count()
        task_count = pending_tasks + in_progress_tasks

        # Determine generation mode
        mode = self._determine_generation_mode(task_count)

        # Sample recent tasks
        recent_tasks = self._sample_recent_tasks(limit=5)

        # Analyze READMEs
        readme_analysis = self._analyze_task_readmes(recent_tasks)

        # Build context summary
        recent_work_summary = []
        if readme_analysis['partial_or_incomplete']:
            partial_items = [f"Task #{item['task_id']}: {item['description'][:80]} (Status: {item['status']})"
                           for item in readme_analysis['partial_or_incomplete'][:3]]
            recent_work_summary.append("Incomplete/Partial tasks:\n- " + "\n- ".join(partial_items))

        if readme_analysis['outstanding_items']:
            recent_work_summary.append("Outstanding items from recent tasks:\n- " + "\n- ".join(readme_analysis['outstanding_items'][:5]))

        if readme_analysis['recommendations']:
            recent_work_summary.append("Recommendations from recent tasks:\n- " + "\n- ".join(readme_analysis['recommendations'][:5]))

        recent_work_text = "\n\n".join(recent_work_summary) if recent_work_summary else "No recent task history available"

        # Build list of available tasks for REFINE targeting
        available_tasks_text = self._format_available_tasks(recent_tasks)

        context = {
            'task_count': task_count,
            'pending_count': pending_tasks,
            'in_progress_count': in_progress_tasks,
            'mode': mode,
            'recent_work': recent_work_text,
            'available_tasks': available_tasks_text,
        }

        logger.debug("autogen.context.gathered", task_count=task_count, mode=mode,
                    partial_tasks=len(readme_analysis['partial_or_incomplete']),
                    available_tasks_count=len(recent_tasks))

        return context

    def _format_available_tasks(self, tasks: list[Task]) -> str:
        """Format list of tasks that can be refined

        Args:
            tasks: List of tasks to format

        Returns:
            Formatted string listing tasks with IDs and status
        """
        if not tasks:
            return "(No tasks available)"

        lines = []
        for task in tasks:
            status = task.status.value.upper() if task.status else "UNKNOWN"
            desc = task.description[:100]
            if len(task.description) > 100:
                desc += "..."
            lines.append(f"- Task #{task.id} ({status}): {desc}")

        return "\n".join(lines)

    async def _generate_task(self) -> Optional[Task]:
        """Generate a task using the configured prompt archetypes."""
        # Gather codebase context first
        context = self._gather_codebase_context()

        prompt_config = self._select_prompt(context)
        if not prompt_config:
            logger.warning("autogen.no_prompt_available")
            return None

        self._last_generation_source = prompt_config.name
        logger.debug("autogen.prompt.begin", prompt=prompt_config.name, task_count=context['task_count'], mode=context['mode'])

        task_desc = await self._generate_from_prompt(prompt_config, context)

        if not task_desc:
            return None

        # Check if generated task matches a suppressed failure pattern
        if self.feedback_store:
            try:
                if self.feedback_store.is_pattern_suppressed(task_desc):
                    logger.info(
                        "autogen.task.suppressed",
                        reason="matches_failure_pattern",
                        preview=task_desc[:80],
                    )
                    return None
            except Exception as e:
                logger.warning("autogen.pattern_check.failed", error=str(e))

        # Parse REFINE target task ID if present
        parsed_desc, refines_task_id = self._parse_refine_target(task_desc)

        # Parse task type from description (for AI-generated tasks with [NEW]/[REFINE] prefix)
        clean_desc, task_type = self._parse_task_type(parsed_desc)

        # All auto-generated tasks are low-priority
        priority = TaskPriority.GENERATED

        # Build task context with refines_task_id if applicable
        task_context = {}
        if refines_task_id is not None:
            task_context['refines_task_id'] = refines_task_id
            logger.debug("autogen.refine_target", task_id="pending", refines=refines_task_id)

        # Create task in database
        task = Task(
            description=clean_desc,
            priority=priority,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            context=json.dumps(task_context) if task_context else None,
        )
        self.session.add(task)
        self.session.flush()  # Get the ID

        # Record in generation history
        usage_percent = self.budget_manager.get_usage_percent()
        history = GenerationHistory(
            task_id=task.id,
            source=prompt_config.name,
            usage_percent_at_generation=usage_percent,
            source_metadata=json.dumps({
                "priority": priority.value,
                "task_type": task_type.value,
                "prompt_name": prompt_config.name,
                "task_count_at_generation": context['task_count'],
                "generation_mode": context['mode'],
                "refines_task_id": refines_task_id,
            })
        )
        self.session.add(history)
        self.session.commit()

        return task

    def _refresh_feedback_weights(self) -> None:
        """Refresh feedback weights from store if stale (older than 10 minutes)."""
        if not self.feedback_store:
            return

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if (
            self._feedback_weights_updated
            and (now - self._feedback_weights_updated).total_seconds() < 600
        ):
            return  # Cache is fresh

        try:
            self._feedback_weights = self.feedback_store.get_feedback_weights_by_source()
            self._feedback_weights_updated = now
            if self._feedback_weights:
                logger.debug(
                    "autogen.feedback_weights.refreshed",
                    weights=self._feedback_weights,
                )
        except Exception as e:
            logger.warning("autogen.feedback_weights.refresh_failed", error=str(e))

    def _select_prompt(self, context: dict) -> Optional[AutoTaskPromptConfig]:
        """Select a prompt configuration based on generation mode, feedback weights, and configured weights."""
        prompts = self.config.prompts or []
        logger.debug("autogen.select_prompt", prompt_count=len(prompts), mode=context['mode'])

        # Refresh feedback weights
        self._refresh_feedback_weights()

        mode = context['mode']

        # Filter prompts by mode first
        mode_prompts = [p for p in prompts if p.name == mode]

        # If no exact mode match, fall back to weighted selection
        if not mode_prompts:
            logger.debug("autogen.no_mode_match", mode=mode, falling_back_to_weighted=True)
            mode_prompts = prompts

        weighted_list: list[AutoTaskPromptConfig] = []

        for prompt in mode_prompts:
            base_weight = float(prompt.weight or 0)
            if base_weight <= 0:
                continue

            # Apply feedback multiplier if available
            feedback_multiplier = self._feedback_weights.get(prompt.name, 1.0)
            adjusted_weight = base_weight * feedback_multiplier

            # Scale by 10 and ensure at least 1 entry for positive weights
            entries = max(int(adjusted_weight * 10), 1) if adjusted_weight > 0 else 0
            weighted_list.extend([prompt] * entries)

            if feedback_multiplier != 1.0:
                logger.debug(
                    "autogen.weight_adjusted",
                    prompt=prompt.name,
                    base=base_weight,
                    multiplier=feedback_multiplier,
                    adjusted=adjusted_weight,
                )

        if not weighted_list:
            logger.warning("autogen.no_prompts_available", configured_prompts=len(prompts))
            return None

        selected = random.choice(weighted_list)
        logger.debug("autogen.prompt_selected", name=selected.name)
        return selected

    @staticmethod
    def _parse_refine_target(task_desc: str) -> tuple[str, Optional[int]]:
        """Parse task description to extract REFINE target task ID

        Args:
            task_desc: Raw task description (may include [REFINE:#<id>] prefix)

        Returns:
            Tuple of (clean_description, refines_task_id or None)

        Examples:
            "[REFINE:#2] Improve error handling" -> ("Improve error handling", 2)
            "[REFINE] General improvements" -> ("[REFINE] General improvements", None)
            "Normal task" -> ("Normal task", None)
        """
        import re

        task_desc = task_desc.strip()

        # Search for [REFINE:#<id>] pattern
        refine_match = re.search(r'\*?\*?\[REFINE:#(\d+)\]\*?\*?', task_desc, re.IGNORECASE)
        if refine_match:
            task_id = int(refine_match.group(1))
            # Extract everything after the tag
            start_pos = refine_match.end()
            clean_desc = task_desc[start_pos:].strip()
            return (clean_desc, task_id)

        # No target specified
        return (task_desc, None)

    @staticmethod
    def _parse_task_type(task_desc: str) -> tuple[str, TaskType]:
        """Parse task description to extract type prefix and clean description

        Args:
            task_desc: Raw task description (may include [NEW] or [REFINE] prefix)

        Returns:
            Tuple of (clean_description, task_type)
        """
        import re

        task_desc = task_desc.strip()

        # Search for [REFINE] tag anywhere in the description (including in markdown bold markers)
        # This includes both [REFINE] and [REFINE:#<id>] patterns
        refine_match = re.search(r'\*?\*?\[REFINE(?::#\d+)?\]\*?\*?', task_desc, re.IGNORECASE)
        if refine_match:
            # Extract everything after the tag
            start_pos = refine_match.end()
            clean_desc = task_desc[start_pos:].strip()
            return (clean_desc, TaskType.REFINE)

        # Search for [NEW] tag anywhere in the description
        new_match = re.search(r'\*?\*?\[NEW\]\*?\*?', task_desc, re.IGNORECASE)
        if new_match:
            # Extract everything after the tag
            start_pos = new_match.end()
            clean_desc = task_desc[start_pos:].strip()
            return (clean_desc, TaskType.NEW)

        # Default to NEW if no prefix found
        return (task_desc, TaskType.NEW)

    async def _generate_from_prompt(self, prompt_config: AutoTaskPromptConfig, context: dict) -> Optional[str]:
        """Execute the configured prompt via Claude and return the response."""
        try:
            return await self._run_prompt(prompt_config, context)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug(
                "autogen.prompt.execution_failed",
                prompt=prompt_config.name,
                error=str(exc),
            )
            return None

    async def _run_prompt(self, prompt_config: AutoTaskPromptConfig, context: dict) -> Optional[str]:
        """Stream the Claude response for the configured prompt."""
        prompt_text = prompt_config.prompt.strip()
        if not prompt_text:
            logger.debug("autogen.prompt.empty", prompt=prompt_config.name)
            return None

        # Inject context into prompt
        prompt_text = prompt_text.format(**context)

        options = self._build_options(self.default_model)
        text_segments: list[str] = []

        try:
            async for message in query(prompt=prompt_text, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text_segments.append(block.text)
        except (CLINotFoundError, ProcessError, CLIConnectionError, CLIJSONDecodeError) as exc:
            self._log_sdk_failure(exc, prompt_name=prompt_config.name)
            return None
        except Exception as exc:  # pragma: no cover - unexpected failure
            self._log_sdk_failure(exc, prompt_name=prompt_config.name, unexpected=True)
            return None

        full_text = "".join(text_segments).strip()
        return full_text or None

    def _build_options(self, model: str) -> ClaudeAgentOptions:
        """Create Claude SDK options for the prompt run.

        Sets cwd to workspace/shared to prevent the auto-generator from accessing
        workspace/data or other restricted directories during task generation.
        This ensures generated tasks don't reference files they won't have access to.
        """
        from pathlib import Path

        # Use shared directory as cwd to prevent access to workspace/data
        shared_dir = Path("./workspace/shared")
        shared_dir.mkdir(parents=True, exist_ok=True)

        return ClaudeAgentOptions(
            model=model,
            cwd=str(shared_dir),
            allowed_tools=[],  # No file access needed for task generation
        )

    @staticmethod
    def _log_sdk_failure(
        exc: Exception,
        *,
        prompt_name: str,
        unexpected: bool = False,
    ) -> None:
        """Emit a structured log entry for Claude SDK failures."""
        event = "claude_sdk.unexpected_error" if unexpected else "claude_sdk.error"
        logger.error(
            event,
            prompt=prompt_name,
            error=str(exc),
            exception_type=exc.__class__.__name__,
        )
