from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Set, TYPE_CHECKING

from sleepless_agent.monitoring.logging import get_logger

from sleepless_agent.core.models import TaskPriority
from sleepless_agent.core.retry import RetryConfig, RetryManager
from sleepless_agent.scheduling.scheduler import SmartScheduler
from sleepless_agent.core.queue import TaskQueue
from sleepless_agent.monitoring.report_generator import ReportGenerator, TaskMetrics
from sleepless_agent.monitoring.monitor import HealthMonitor, PerformanceLogger
from sleepless_agent.storage.git import GitManager
from sleepless_agent.storage.results import ResultManager
from sleepless_agent.storage.feedback import FeedbackStore
from sleepless_agent.core.executor import ClaudeCodeExecutor
from sleepless_agent.utils.exceptions import PauseException

if TYPE_CHECKING:
    from sleepless_agent.interfaces.bot import SlackBot

logger = get_logger(__name__)


class TaskRuntime:
    """Handle end-to-end execution of individual tasks."""

    def __init__(
        self,
        *,
        config,
        task_queue: TaskQueue,
        scheduler: SmartScheduler,
        claude: ClaudeCodeExecutor,
        results: ResultManager,
        git: GitManager,
        monitor: HealthMonitor,
        perf_logger: PerformanceLogger,
        report_generator: ReportGenerator,
        bot: Optional[SlackBot],
        live_status_tracker,
        feedback_store: Optional[FeedbackStore] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.config = config
        self.task_queue = task_queue
        self.scheduler = scheduler
        self.claude = claude
        self.results = results
        self.git = git
        self.monitor = monitor
        self.perf_logger = perf_logger
        self.report_generator = report_generator
        self.bot = bot
        self.live_status_tracker = live_status_tracker
        self.feedback_store = feedback_store

        # Initialize retry manager
        self.retry_config = retry_config or RetryConfig()
        self.retry_manager = RetryManager(
            config=self.retry_config,
            feedback_store=feedback_store,
        )

    async def execute(self, task) -> None:
        """Execute a single task asynchronously."""
        # Build context dict with only non-None values to reduce log noise
        context = {
            "task_id": task.id,
            "priority": task.priority.value if task.priority else "unknown",
        }
        if task.project_id:
            context["project_id"] = task.project_id
        if task.project_name:
            context["project_name"] = task.project_name

        task_log = logger.bind(**context)

        self.task_queue.mark_in_progress(task.id)

        task_log.info("=" * 80)
        task_log.info(
            "task.start",
            description=task.description,
        )

        start_time = time.time()
        result_output: str = ""
        files_modified: List[str] = []
        commands_executed: List[str] = []
        git_commit_sha: Optional[str] = None
        git_pr_url: Optional[str] = None
        workspace: Optional[Path] = None

        try:
            (
                result_output,
                files_modified,
                commands_executed,
                exit_code,
                usage_metrics,
                eval_status,
            ) = await self._run_task_with_timeout(task)

            processing_time = int(time.time() - start_time)
            usage_metrics = usage_metrics or {}
            files_modified = sorted(files_modified) if files_modified else []
            commands_executed = commands_executed or []

            # Log detailed metrics at DEBUG level to reduce noise
            task_log.debug(
                "task.executor.done",
                exit_code=exit_code,
                duration_s=processing_time,
                total_cost_usd=usage_metrics.get("total_cost_usd"),
                turns=usage_metrics.get("num_turns"),
                eval_status=eval_status,
            )

            if exit_code != 0:
                task_log.warning("task.exit_code", exit_code=exit_code)

            git_branch = self.git.determine_branch(task.project_id)
            workspace = self.claude.get_workspace_path(task.id, task.project_id)

            result = self.results.save_result(
                task_id=task.id,
                output=result_output,
                files_modified=files_modified,
                commands_executed=commands_executed,
                processing_time_seconds=processing_time,
                git_commit_sha=None,
                git_pr_url=None,
                git_branch=git_branch,
                workspace_path=str(workspace),
            )

            # Move to DEBUG - result saving is an internal detail
            task_log.debug(
                "task.result.saved",
                result_id=result.id,
                files=len(files_modified),
                commands=len(commands_executed),
            )

            if workspace and workspace.exists():
                self.claude.cleanup_workspace_caches(workspace)
                git_commit_sha = self._maybe_commit_changes(
                    task=task,
                    task_log=task_log,
                    workspace=workspace,
                    files_modified=files_modified,
                    result_output=result_output,
                    git_branch=git_branch,
                )

                if git_commit_sha:
                    self.results.update_result_commit_info(
                        result.id,
                        git_commit_sha=git_commit_sha,
                        git_pr_url=git_pr_url,
                        git_branch=git_branch,
                    )
                else:
                    task_log.debug("task.git.no_commit")
            else:
                task_log.warning("task.git.skipped", reason="workspace_missing")

            # Check evaluator status before marking as completed
            # Only mark as completed if evaluator says COMPLETE, or if evaluator is disabled
            if eval_status and eval_status.upper() in ["INCOMPLETE", "FAILED", "PARTIAL"]:
                task_log.warning(
                    "task.evaluator_incomplete",
                    eval_status=eval_status,
                    message="Task marked as failed due to evaluator status"
                )
                self.task_queue.mark_failed(task.id, f"Evaluator status: {eval_status}")
                self._log_failure_metrics(task=task, duration=processing_time, error=f"Evaluator: {eval_status}")
                task_log.info(
                    "task.complete",
                    status="failed",
                    duration_s=processing_time,
                    eval_status=eval_status,
                )
                task_log.info("=" * 80)
            else:
                self.task_queue.mark_completed(task.id, result_id=result.id)
                self._log_success_metrics(
                    task=task,
                    processing_time=processing_time,
                    files_modified=files_modified,
                    commands_executed=commands_executed,
                    git_commit_sha=git_commit_sha,
                    git_pr_url=git_pr_url,
                    usage_metrics=usage_metrics,
                    result_output=result_output,
                )
                task_log.info(
                    "task.complete",
                    status="completed",
                    duration_s=processing_time,
                    git_commit=git_commit_sha[:8] if git_commit_sha else None,
                    eval_status=eval_status,
                )
                task_log.info("=" * 80)
        except PauseException as pause:
            await self._handle_pause_exception(
                task=task,
                task_log=task_log,
                pause=pause,
                start_time=start_time,
                result_output=result_output,
                files_modified=files_modified,
                commands_executed=commands_executed,
                workspace=workspace,
            )
        except Exception as exc:
            processing_time = int(time.time() - start_time)
            error_str = str(exc)
            task_log.error("task.failure", error=error_str, duration_s=processing_time)

            # Record failure pattern for learning
            if self.feedback_store:
                try:
                    self.feedback_store.record_failure(error_str, task=task)
                except Exception as record_err:
                    task_log.debug(f"Failed to record failure pattern: {record_err}")

            # Check if we should retry
            retry_decision = self.retry_manager.should_retry(task, error_str)

            if retry_decision.should_retry:
                # Schedule retry
                await self._handle_retry(
                    task=task,
                    task_log=task_log,
                    error=error_str,
                    processing_time=processing_time,
                    retry_decision=retry_decision,
                )
            else:
                # Final failure - no more retries
                self.task_queue.mark_failed(task.id, error_str)
                self._log_failure_metrics(task=task, duration=processing_time, error=error_str)
                task_log.info(
                    "task.complete",
                    status="failed",
                    duration_s=processing_time,
                    error=error_str,
                    retry_info=retry_decision.reason,
                )
                task_log.info("=" * 80)

    async def _run_task_with_timeout(self, task):
        import json
        timeout = self.config.agent.task_timeout_seconds

        # Parse task context for workspace reuse
        task_context = None
        if task.context:
            try:
                task_context = json.loads(task.context)
            except (json.JSONDecodeError, TypeError):
                logger.warning("task.context.parse_failed", task_id=task.id, context=task.context)
                task_context = None

        try:
            return await asyncio.wait_for(
                self.claude.execute_task(
                    task_id=task.id,
                    description=task.description,
                    task_type="general",
                    priority=task.priority.value,
                    timeout=timeout,
                    project_id=task.project_id,
                    project_name=task.project_name,
                    workspace_task_type=task.task_type.value if task.task_type else None,
                    task_context=task_context,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            timeout_minutes = max(1, timeout // 60)
            logger.error("task.timeout", task_id=task.id, timeout_minutes=timeout_minutes)
            raise TimeoutError(f"Timed out after {timeout_minutes} minute(s)") from exc

    async def _handle_retry(
        self,
        *,
        task,
        task_log,
        error: str,
        processing_time: int,
        retry_decision,
    ) -> None:
        """Handle retry logic for a failed task.

        Updates the attempt count, notifies the user, waits for the backoff
        period, then re-executes the task.
        """
        from sleepless_agent.core.retry import RetryDecision

        # Increment attempt count in the database
        self.task_queue.increment_attempt_count(task.id)

        # Log the retry
        task_log.warning(
            "task.retry.scheduled",
            attempt=retry_decision.next_attempt,
            max_attempts=self.retry_config.max_attempts,
            delay_seconds=round(retry_decision.delay_seconds, 1),
            error=error[:200],
        )

        # Notify user if assigned
        if task.assigned_to and self.bot:
            try:
                retry_info = self.retry_manager.format_retry_info(task, error)
                self.bot.send_message(
                    task.assigned_to,
                    f"⚠️ Task #{task.id} failed: {error[:200]}\n"
                    f"🔄 {retry_info}"
                )
            except Exception as notify_err:
                task_log.debug(f"Failed to send retry notification: {notify_err}")

        # Wait for backoff period
        if retry_decision.delay_seconds > 0:
            task_log.info(
                "task.retry.waiting",
                delay_seconds=round(retry_decision.delay_seconds, 1),
            )
            await asyncio.sleep(retry_decision.delay_seconds)

        # Update task state - refresh from DB to get current attempt count
        refreshed_task = self.task_queue.get_task(task.id)
        if not refreshed_task:
            task_log.error("task.retry.task_not_found", task_id=task.id)
            return

        task_log.info(
            "task.retry.executing",
            attempt=retry_decision.next_attempt,
        )

        # Re-execute the task (this will call execute() again with updated attempt_count)
        await self.execute(refreshed_task)

    def _maybe_commit_changes(
        self,
        *,
        task,
        task_log,
        workspace: Path,
        files_modified: List[str],
        result_output: str,
        git_branch: str,
    ) -> Optional[str]:
        files_for_commit: Set[str] = set(files_modified or [])

        if task.priority in (TaskPriority.THOUGHT, TaskPriority.GENERATED):
            if not files_for_commit:
                summary_rel_path = self.git.write_summary_file(
                    workspace_path=workspace,
                    task_id=task.id,
                    priority=task.priority.value,
                    description=task.description,
                    result_output=result_output,
                )
                if summary_rel_path:
                    files_for_commit.add(summary_rel_path)
            commit_targets = self._collect_commit_targets(workspace, files_for_commit)
            if commit_targets:
                commit_message = f"Capture thought: #{task.id} {task.description[:60]}"
                commit_sha = self.git.commit_workspace_changes(
                    branch=git_branch,
                    workspace_path=workspace,
                    files=commit_targets,
                    message=commit_message,
                )
                # Keep at INFO but simplify - git commits are important
                task_log.info(
                    "task.git.commit",
                    branch=git_branch,
                    commit=commit_sha[:8] if commit_sha else None,  # Short commit hash
                    files=len(commit_targets),
                )
                return commit_sha
            return None

        if task.priority == TaskPriority.SERIOUS:
            if files_for_commit:
                is_valid, validation_msg = self.git.validate_changes(
                    workspace, sorted(files_for_commit)
                )
                if not is_valid:
                    task_log.warning("task.git.validation_failed", reason=validation_msg)
                    return None
                commit_targets = self._collect_commit_targets(workspace, files_for_commit)
                if commit_targets:
                    commit_message = f"Implement task #{task.id}: {task.description[:60]}"
                    commit_sha = self.git.commit_workspace_changes(
                        branch=git_branch,
                        workspace_path=workspace,
                        files=commit_targets,
                        message=commit_message,
                    )
                    task_log.info(
                        "task.git.commit",
                        branch=git_branch,
                        commit=commit_sha,
                        files=len(commit_targets),
                    )
                    return commit_sha
            return None

        commit_targets = self._collect_commit_targets(workspace, files_for_commit)
        if commit_targets:
            commit_message = f"Task #{task.id} update: {task.description[:60]}"
            commit_sha = self.git.commit_workspace_changes(
                branch=git_branch,
                workspace_path=workspace,
                files=commit_targets,
                message=commit_message,
            )
            task_log.info(
                "task.git.commit",
                branch=git_branch,
                commit=commit_sha,
                files=len(commit_targets),
            )
            return commit_sha
        return None

    def _collect_commit_targets(self, workspace_path: Path, base_files: Iterable[str]) -> List[str]:
        commit_paths: Set[str] = set()

        for entry in base_files:
            if entry:
                commit_paths.add(Path(entry).as_posix())

        try:
            for file_path in self.claude.list_workspace_files(workspace_path):
                commit_paths.add(Path(file_path).as_posix())
        except Exception as exc:
            logger.debug(f"Unable to enumerate workspace files in {workspace_path}: {exc}")

        tasks_root = workspace_path.parent
        if tasks_root.exists():
            rel_tasks = self._relative_to_workspace(workspace_path, tasks_root)
            if rel_tasks:
                commit_paths.add(rel_tasks)

        workspace_root = tasks_root.parent if tasks_root else workspace_path
        data_dir = workspace_root / "data"
        if data_dir.exists():
            rel_data = self._relative_to_workspace(workspace_path, data_dir)
            if rel_data:
                commit_paths.add(rel_data)

        return sorted(path for path in commit_paths if path and path != ".")

    def _relative_to_workspace(self, workspace: Path, target: Path) -> Optional[str]:
        try:
            rel_path = os.path.relpath(target, workspace)
        except Exception as exc:
            logger.debug(f"Failed to compute relative path from {workspace} to {target}: {exc}")
            return None

        if not rel_path or rel_path == ".":
            return None

        return Path(rel_path).as_posix()

    def _log_success_metrics(
        self,
        *,
        task,
        processing_time: int,
        files_modified: List[str],
        commands_executed: List[str],
        git_commit_sha: Optional[str],
        git_pr_url: Optional[str],
        usage_metrics: dict,
        result_output: str,
    ) -> None:
        try:
            git_info = None
            if git_commit_sha or git_pr_url:
                parts = []
                if git_commit_sha:
                    parts.append(f"Commit: {git_commit_sha[:8]}")
                if git_pr_url:
                    parts.append(f"PR: {git_pr_url}")
                git_info = " ".join(parts)

            task_metrics = TaskMetrics(
                task_id=task.id,
                description=task.description,
                priority=task.priority.value,
                status="completed",
                duration_seconds=processing_time,
                files_modified=len(files_modified),
                commands_executed=len(commands_executed),
                git_info=git_info,
            )
            self.report_generator.append_task_completion(task_metrics, project_id=task.project_id)
        except Exception as exc:
            logger.error("task.report.append_failed", error=str(exc))

        try:
            self.scheduler.record_task_usage(
                task_id=task.id,
                total_cost_usd=usage_metrics.get("total_cost_usd"),
                duration_ms=usage_metrics.get("duration_ms"),
                duration_api_ms=usage_metrics.get("duration_api_ms"),
                num_turns=usage_metrics.get("num_turns"),
                project_id=task.project_id,
            )
        except Exception as exc:
            logger.debug("scheduler.usage.record_failed", error=str(exc))

        try:
            self.monitor.record_task_completion(processing_time, success=True)
        except Exception as exc:
            logger.debug(f"Failed to record completion in health monitor for task {task.id}: {exc}")

        try:
            self.perf_logger.log_task_execution(
                task_id=task.id,
                description=task.description,
                priority=task.priority.value if task.priority else "unknown",
                duration_seconds=processing_time,
                success=True,
                files_modified=len(files_modified),
                commands_executed=len(commands_executed),
            )
        except Exception as exc:
            logger.debug(f"Failed to log metrics for task {task.id}: {exc}")

        if task.assigned_to and self.bot:
            try:
                priority_icon = {
                    TaskPriority.SERIOUS: "🔴",
                    TaskPriority.THOUGHT: "🟡",
                    TaskPriority.GENERATED: "🟢",
                }.get(task.priority, "ℹ️")

                files_info = f"\n📝 Files modified: {len(files_modified)}" if files_modified else ""
                commands_info = f"\n⚙️ Commands: {len(commands_executed)}" if commands_executed else ""
                git_info_display = ""
                if git_commit_sha:
                    git_info_display = f"\n✅ Committed: {git_commit_sha[:8]}"
                if git_pr_url:
                    git_info_display += f"\n🔗 PR: {git_pr_url}"

                output_limit = 3500
                truncated_output = result_output[:output_limit]
                if len(result_output) > output_limit:
                    truncated_output += "\n\n_[Output truncated - see result file for full content]_"

                message = (
                    f"{priority_icon} Task #{task.id} completed in {processing_time}s"
                    f"{files_info}{commands_info}{git_info_display}\n"
                    f"```{truncated_output}```"
                )
                self.bot.send_message(task.assigned_to, message)
            except Exception as exc:
                logger.debug(f"Failed to send completion notification for task {task.id}: {exc}")

        if self.live_status_tracker:
            try:
                self.live_status_tracker.clear(task.id)
            except Exception as exc:
                logger.debug(f"Failed to clear live status for task {task.id}: {exc}")

    def _log_failure_metrics(self, *, task, duration: int, error: str) -> None:
        try:
            task_metrics = TaskMetrics(
                task_id=task.id,
                description=task.description,
                priority=task.priority.value,
                status="failed",
                duration_seconds=duration,
                files_modified=0,
                commands_executed=0,
                error_message=error,
            )
            self.report_generator.append_task_completion(task_metrics, project_id=task.project_id)
        except Exception as exc:
            logger.error("task.report.append_failed", error=str(exc))

        try:
            self.monitor.record_task_completion(duration, success=False)
        except Exception as exc:
            logger.debug(f"Failed to record failure in health monitor for task {task.id}: {exc}")

        try:
            self.perf_logger.log_task_execution(
                task_id=task.id,
                description=task.description,
                priority=task.priority.value if task.priority else "unknown",
                duration_seconds=duration,
                success=False,
            )
        except Exception as exc:
            logger.debug(f"Failed to log failure metrics for task {task.id}: {exc}")

        if task.assigned_to and self.bot:
            try:
                self.bot.send_message(task.assigned_to, f"❌ Task #{task.id} failed: {error}")
            except Exception as exc:
                logger.debug(f"Failed to send failure notification for task {task.id}: {exc}")

        if self.live_status_tracker:
            try:
                self.live_status_tracker.clear(task.id)
            except Exception as exc:
                logger.debug(f"Failed to clear live status for failed task {task.id}: {exc}")

    async def _handle_pause_exception(
        self,
        *,
        task,
        task_log,
        pause: PauseException,
        start_time: float,
        result_output: str,
        files_modified: List[str],
        commands_executed: List[str],
        workspace: Optional[Path],
    ) -> None:
        reset_time_iso = pause.reset_time.isoformat() if pause.reset_time else None
        task_log.warning("task.pause.limit", usage_percent=pause.usage_percent, reset_at=reset_time_iso)

        try:
            result = self.results.save_result(
                task_id=task.id,
                output=result_output or "[Task completed before pause]",
                files_modified=files_modified,
                commands_executed=commands_executed,
                processing_time_seconds=int(time.time() - start_time),
                git_commit_sha=None,
                git_pr_url=None,
                git_branch=None,
                workspace_path=str(workspace) if workspace else "",
            )
            self.task_queue.mark_completed(task.id, result_id=result.id)
        except Exception as save_error:
            task_log.warning("task.pause.save_failed", error=str(save_error))

        if self.live_status_tracker:
            try:
                self.live_status_tracker.clear(task.id)
            except Exception as exc:
                logger.debug(f"Failed to clear live status for paused task {task.id}: {exc}")

        sleep_seconds = 0.0
        if pause.reset_time:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            sleep_seconds = max(0.0, (pause.reset_time - now).total_seconds())
            task_log.info("task.pause.reset_time", reset_at=reset_time_iso)
        else:
            task_log.info("task.pause.reset_time_missing")

        if task.assigned_to and self.bot:
            try:
                pause_message = (
                    f"⏸️  Pro plan usage limit reached ({pause.usage_percent:.0f}%)\n"
                    f"Task #{task.id} completed successfully\n"
                )
                if pause.reset_time:
                    pause_message += (
                        f"Pausing execution until {pause.reset_time.strftime('%H:%M:%S')}\n"
                        f"Will resume automatically in ~{sleep_seconds / 60:.0f} minutes"
                    )
                else:
                    pause_message += "Pausing execution until credits refresh."
                self.bot.send_message(task.assigned_to, pause_message)
            except Exception as exc:
                logger.debug(f"Failed to send pause notification for task {task.id}: {exc}")

        if sleep_seconds > 0 and pause.reset_time:
            task_log.critical(
                "task.pause.sleep",
                sleep_minutes=round(sleep_seconds / 60, 2),
                resume_at=reset_time_iso,
            )
            await asyncio.sleep(sleep_seconds)
            task_log.info("task.pause.resume")
            if task.assigned_to and self.bot:
                try:
                    self.bot.send_message(task.assigned_to, "▶️  Pro plan usage limit reset - resuming tasks")
                except Exception as exc:
                    logger.debug(f"Failed to send resume notification for task {task.id}: {exc}")
