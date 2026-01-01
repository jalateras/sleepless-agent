"""Tests for task runtime execution lifecycle.

This module tests the TaskRuntime class which handles end-to-end execution of
individual tasks. It verifies task execution, timeout handling, pause exception
handling, checkpoint integration, notification integration, stream integration,
and retry logic.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from sleepless_agent.core.models import (
    Base,
    CheckpointStatus,
    CheckpointType,
    Task,
    TaskPriority,
    TaskStatus,
    TaskType,
    init_db,
)
from sleepless_agent.core.queue import TaskQueue
from sleepless_agent.core.retry import RetryConfig, RetryDecision
from sleepless_agent.core.task_runtime import TaskRuntime
from sleepless_agent.monitoring.notifications import BlockerType, ExecutionPhase
from sleepless_agent.utils.exceptions import PauseException


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_db() -> Generator[str, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    engine = init_db(db_path)
    Base.metadata.create_all(engine)

    yield db_path


@pytest.fixture
def task_queue(temp_db: str) -> TaskQueue:
    """Create a TaskQueue for testing."""
    return TaskQueue(db_path=temp_db)


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock configuration."""
    config = MagicMock()
    config.agent.task_timeout_seconds = 300
    return config


@pytest.fixture
def mock_scheduler() -> MagicMock:
    """Create a mock SmartScheduler."""
    scheduler = MagicMock()
    scheduler.record_task_usage = MagicMock()
    return scheduler


@pytest.fixture
def mock_claude() -> MagicMock:
    """Create a mock ClaudeCodeExecutor."""
    claude = MagicMock()

    async def mock_execute(*args, **kwargs):
        return (
            "Task completed successfully",  # result_output
            ["src/main.py"],  # files_modified
            ["git add ."],  # commands_executed
            0,  # exit_code
            {"total_cost_usd": 0.05, "num_turns": 3},  # usage_metrics
            "COMPLETE",  # eval_status
        )

    claude.execute_task = AsyncMock(side_effect=mock_execute)
    claude.get_workspace_path = MagicMock(return_value=Path("/tmp/workspace"))
    claude.cleanup_workspace_caches = MagicMock()
    claude.list_workspace_files = MagicMock(return_value=[])
    return claude


@pytest.fixture
def mock_results() -> MagicMock:
    """Create a mock ResultManager."""
    results = MagicMock()
    mock_result = MagicMock()
    mock_result.id = 1
    results.save_result = MagicMock(return_value=mock_result)
    results.update_result_commit_info = MagicMock()
    return results


@pytest.fixture
def mock_git() -> MagicMock:
    """Create a mock GitManager."""
    git = MagicMock()
    git.determine_branch = MagicMock(return_value="feature/test")
    git.validate_changes = MagicMock(return_value=(True, None))
    git.commit_workspace_changes = MagicMock(return_value="abc123def456")
    git.write_summary_file = MagicMock(return_value=None)
    return git


@pytest.fixture
def mock_monitor() -> MagicMock:
    """Create a mock HealthMonitor."""
    monitor = MagicMock()
    monitor.record_task_completion = MagicMock()
    return monitor


@pytest.fixture
def mock_perf_logger() -> MagicMock:
    """Create a mock PerformanceLogger."""
    perf_logger = MagicMock()
    perf_logger.log_task_execution = MagicMock()
    return perf_logger


@pytest.fixture
def mock_report_generator() -> MagicMock:
    """Create a mock ReportGenerator."""
    report_gen = MagicMock()
    report_gen.append_task_completion = MagicMock()
    return report_gen


@pytest.fixture
def mock_bot() -> MagicMock:
    """Create a mock SlackBot."""
    bot = MagicMock()
    bot.send_message = MagicMock()
    return bot


@pytest.fixture
def mock_live_status_tracker() -> MagicMock:
    """Create a mock live status tracker."""
    tracker = MagicMock()
    tracker.clear = MagicMock()
    return tracker


@pytest.fixture
def mock_feedback_store() -> MagicMock:
    """Create a mock FeedbackStore."""
    store = MagicMock()
    store.record_failure = MagicMock()
    store.record_feedback = MagicMock()
    return store


@pytest.fixture
def mock_notification_manager() -> AsyncMock:
    """Create a mock NotificationManager."""
    manager = AsyncMock()
    manager.start_task_tracking = MagicMock()
    manager.stop_task_tracking = MagicMock()
    manager.start_heartbeat = AsyncMock()
    manager.notify_phase_transition = AsyncMock()
    manager.notify_completion = AsyncMock()
    manager.notify_blocker = AsyncMock()
    return manager


@pytest.fixture
def mock_stream_manager() -> AsyncMock:
    """Create a mock StreamManager."""
    manager = AsyncMock()
    manager.start_stream = AsyncMock(return_value="1234567890.123456")
    manager.append_output = AsyncMock()
    manager.finalize_stream = AsyncMock()
    manager.update_phase = AsyncMock()
    return manager


@pytest.fixture
def mock_checkpoint_manager() -> AsyncMock:
    """Create a mock CheckpointManager."""
    manager = AsyncMock()
    result = MagicMock()
    result.should_proceed = True
    result.status = CheckpointStatus.APPROVED
    manager.request_approval = AsyncMock(return_value=result)
    return manager


@pytest.fixture
def retry_config() -> RetryConfig:
    """Create a retry configuration for testing."""
    return RetryConfig(
        max_attempts=3,
        base_delay_seconds=0.01,  # Fast for tests
        max_delay_seconds=0.1,
        exponential_base=2.0,
        jitter_factor=0.0,
    )


@pytest.fixture
def runtime(
    mock_config: MagicMock,
    task_queue: TaskQueue,
    mock_scheduler: MagicMock,
    mock_claude: MagicMock,
    mock_results: MagicMock,
    mock_git: MagicMock,
    mock_monitor: MagicMock,
    mock_perf_logger: MagicMock,
    mock_report_generator: MagicMock,
    mock_bot: MagicMock,
    mock_live_status_tracker: MagicMock,
    mock_feedback_store: MagicMock,
    retry_config: RetryConfig,
) -> TaskRuntime:
    """Create a TaskRuntime with mock dependencies."""
    return TaskRuntime(
        config=mock_config,
        task_queue=task_queue,
        scheduler=mock_scheduler,
        claude=mock_claude,
        results=mock_results,
        git=mock_git,
        monitor=mock_monitor,
        perf_logger=mock_perf_logger,
        report_generator=mock_report_generator,
        bot=mock_bot,
        live_status_tracker=mock_live_status_tracker,
        feedback_store=mock_feedback_store,
        retry_config=retry_config,
    )


@pytest.fixture
def runtime_with_notifications(
    runtime: TaskRuntime,
    mock_notification_manager: AsyncMock,
) -> TaskRuntime:
    """Create a TaskRuntime with notification manager."""
    runtime.notification_manager = mock_notification_manager
    return runtime


@pytest.fixture
def runtime_with_streaming(
    runtime: TaskRuntime,
    mock_stream_manager: AsyncMock,
) -> TaskRuntime:
    """Create a TaskRuntime with stream manager."""
    runtime.stream_manager = mock_stream_manager
    return runtime


@pytest.fixture
def runtime_with_checkpoints(
    runtime: TaskRuntime,
    mock_checkpoint_manager: AsyncMock,
) -> TaskRuntime:
    """Create a TaskRuntime with checkpoint manager."""
    runtime.checkpoint_manager = mock_checkpoint_manager
    return runtime


@pytest.fixture
def sample_task(task_queue: TaskQueue) -> Task:
    """Create a sample task for testing."""
    return task_queue.add_task(
        description="Test task for runtime",
        priority=TaskPriority.THOUGHT,
    )


# -----------------------------------------------------------------------------
# Tests for execute() - Successful Completion
# -----------------------------------------------------------------------------


class TestExecuteSuccess:
    """Tests for TaskRuntime.execute() successful execution."""

    @pytest.mark.asyncio
    async def test_successful_task_completion(
        self, runtime: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() completes task successfully."""
        await runtime.execute(sample_task)

        # Verify task is marked completed
        updated = runtime.task_queue.get_task(sample_task.id)
        assert updated.status == TaskStatus.COMPLETED
        assert updated.result_id is not None

    @pytest.mark.asyncio
    async def test_marks_task_in_progress(
        self, runtime: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() marks task as in_progress."""
        # We need to check during execution, so we capture the state
        states_during_execution = []

        original_execute = runtime.claude.execute_task

        async def capture_state(*args, **kwargs):
            task = runtime.task_queue.get_task(sample_task.id)
            states_during_execution.append(task.status)
            return await original_execute(*args, **kwargs)

        runtime.claude.execute_task = AsyncMock(side_effect=capture_state)

        await runtime.execute(sample_task)

        # Task should have been IN_PROGRESS during execution
        assert TaskStatus.IN_PROGRESS in states_during_execution

    @pytest.mark.asyncio
    async def test_saves_result(
        self, runtime: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() saves result via ResultManager."""
        await runtime.execute(sample_task)

        runtime.results.save_result.assert_called_once()
        call_args = runtime.results.save_result.call_args
        assert call_args.kwargs["task_id"] == sample_task.id
        assert "output" in call_args.kwargs


# -----------------------------------------------------------------------------
# Tests for execute() - Failure Handling
# -----------------------------------------------------------------------------


class TestExecuteFailure:
    """Tests for TaskRuntime.execute() failure handling."""

    @pytest.mark.asyncio
    async def test_marks_task_failed_on_exception(
        self, runtime: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() marks task as failed on exception."""
        runtime.claude.execute_task = AsyncMock(
            side_effect=RuntimeError("Execution failed")
        )

        await runtime.execute(sample_task)

        # After all retries exhausted, task should be failed
        updated = runtime.task_queue.get_task(sample_task.id)
        assert updated.status == TaskStatus.FAILED
        assert "Execution failed" in updated.error_message

    @pytest.mark.asyncio
    async def test_records_failure_to_feedback_store(
        self, runtime: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() records failure pattern to feedback store."""
        runtime.claude.execute_task = AsyncMock(
            side_effect=RuntimeError("Connection timed out")
        )

        await runtime.execute(sample_task)

        runtime.feedback_store.record_failure.assert_called()


# -----------------------------------------------------------------------------
# Tests for execute() - Timeout Handling
# -----------------------------------------------------------------------------


class TestExecuteTimeout:
    """Tests for TaskRuntime.execute() timeout handling."""

    @pytest.mark.asyncio
    async def test_respects_timeout_configuration(
        self, runtime: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() respects timeout configuration."""
        # Set a very short timeout
        runtime.config.agent.task_timeout_seconds = 0.01

        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(1)  # Slower than timeout
            return ("", [], [], 0, {}, "COMPLETE")

        runtime.claude.execute_task = AsyncMock(side_effect=slow_execute)

        await runtime.execute(sample_task)

        # Should have timed out and marked as failed
        updated = runtime.task_queue.get_task(sample_task.id)
        assert updated.status == TaskStatus.FAILED
        assert "Timed out" in updated.error_message


# -----------------------------------------------------------------------------
# Tests for execute() - PauseException Handling
# -----------------------------------------------------------------------------


class TestExecutePauseException:
    """Tests for TaskRuntime.execute() PauseException handling."""

    @pytest.mark.asyncio
    async def test_handles_pause_exception(
        self, runtime: TaskRuntime, sample_task: Task, mock_bot: MagicMock
    ) -> None:
        """Verify execute() handles PauseException (usage limit hit)."""
        sample_task.assigned_to = "U123456"
        runtime.task_queue.update_task_description(
            sample_task.id, sample_task.description
        )

        pause_exc = PauseException(
            message="Pro plan usage limit reached",
            reset_time=datetime.now(timezone.utc).replace(tzinfo=None),
            usage_percent=95.0,
        )
        runtime.claude.execute_task = AsyncMock(side_effect=pause_exc)

        await runtime.execute(sample_task)

        # Task should be marked completed (work done before pause)
        updated = runtime.task_queue.get_task(sample_task.id)
        assert updated.status == TaskStatus.COMPLETED

        # Should notify user about pause
        mock_bot.send_message.assert_called()
        call_args = mock_bot.send_message.call_args
        assert "usage limit" in call_args.args[1].lower() or "pausing" in call_args.args[1].lower()


# -----------------------------------------------------------------------------
# Tests for execute() - Notification Integration
# -----------------------------------------------------------------------------


class TestExecuteNotifications:
    """Tests for TaskRuntime.execute() notification integration."""

    @pytest.mark.asyncio
    async def test_starts_notification_tracking(
        self, runtime_with_notifications: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() starts notification tracking."""
        await runtime_with_notifications.execute(sample_task)

        runtime_with_notifications.notification_manager.start_task_tracking.assert_called()

    @pytest.mark.asyncio
    async def test_notifies_phase_transition_on_start(
        self, runtime_with_notifications: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() notifies phase transition on start."""
        await runtime_with_notifications.execute(sample_task)

        runtime_with_notifications.notification_manager.notify_phase_transition.assert_called()
        call_args = runtime_with_notifications.notification_manager.notify_phase_transition.call_args
        assert call_args.kwargs["new_phase"] == ExecutionPhase.EXECUTING

    @pytest.mark.asyncio
    async def test_notifies_completion_on_success(
        self, runtime_with_notifications: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() notifies completion on success."""
        await runtime_with_notifications.execute(sample_task)

        runtime_with_notifications.notification_manager.notify_completion.assert_called()
        call_args = runtime_with_notifications.notification_manager.notify_completion.call_args
        assert call_args.kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_notifies_completion_on_failure(
        self, runtime_with_notifications: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() notifies completion on failure."""
        runtime_with_notifications.claude.execute_task = AsyncMock(
            side_effect=RuntimeError("Test failure")
        )

        await runtime_with_notifications.execute(sample_task)

        runtime_with_notifications.notification_manager.notify_completion.assert_called()
        call_args = runtime_with_notifications.notification_manager.notify_completion.call_args
        assert call_args.kwargs["success"] is False

    @pytest.mark.asyncio
    async def test_stops_notification_tracking_on_complete(
        self, runtime_with_notifications: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() stops notification tracking on complete."""
        await runtime_with_notifications.execute(sample_task)

        runtime_with_notifications.notification_manager.stop_task_tracking.assert_called()


# -----------------------------------------------------------------------------
# Tests for execute() - Stream Integration
# -----------------------------------------------------------------------------


class TestExecuteStreaming:
    """Tests for TaskRuntime.execute() stream integration."""

    @pytest.mark.asyncio
    async def test_starts_stream_for_assigned_task(
        self, runtime_with_streaming: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() starts stream for task with assigned user."""
        sample_task.assigned_to = "U123456"
        runtime_with_streaming.task_queue.update_task_description(
            sample_task.id, sample_task.description
        )

        await runtime_with_streaming.execute(sample_task)

        runtime_with_streaming.stream_manager.start_stream.assert_called()

    @pytest.mark.asyncio
    async def test_finalizes_stream_on_success(
        self, runtime_with_streaming: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() finalizes stream on success."""
        sample_task.assigned_to = "U123456"
        runtime_with_streaming.task_queue.update_task_description(
            sample_task.id, sample_task.description
        )

        await runtime_with_streaming.execute(sample_task)

        runtime_with_streaming.stream_manager.finalize_stream.assert_called()
        call_args = runtime_with_streaming.stream_manager.finalize_stream.call_args
        assert call_args.kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_finalizes_stream_on_failure(
        self, runtime_with_streaming: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() finalizes stream on failure."""
        sample_task.assigned_to = "U123456"
        runtime_with_streaming.task_queue.update_task_description(
            sample_task.id, sample_task.description
        )
        runtime_with_streaming.claude.execute_task = AsyncMock(
            side_effect=RuntimeError("Test failure")
        )

        await runtime_with_streaming.execute(sample_task)

        runtime_with_streaming.stream_manager.finalize_stream.assert_called()
        call_args = runtime_with_streaming.stream_manager.finalize_stream.call_args
        assert call_args.kwargs["success"] is False


# -----------------------------------------------------------------------------
# Tests for execute() - Checkpoint Integration
# -----------------------------------------------------------------------------


class TestExecuteCheckpoints:
    """Tests for TaskRuntime.execute() checkpoint integration."""

    @pytest.mark.asyncio
    async def test_triggers_checkpoint_before_commit(
        self, runtime_with_checkpoints: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() triggers checkpoint before commit."""
        sample_task.assigned_to = "U123456"
        runtime_with_checkpoints.task_queue.update_task_description(
            sample_task.id, sample_task.description
        )

        # Mock workspace exists
        with patch.object(Path, "exists", return_value=True):
            await runtime_with_checkpoints.execute(sample_task)

        runtime_with_checkpoints.checkpoint_manager.request_approval.assert_called()
        call_args = runtime_with_checkpoints.checkpoint_manager.request_approval.call_args
        assert call_args.kwargs["checkpoint_type"] == CheckpointType.PRE_COMMIT

    @pytest.mark.asyncio
    async def test_skips_commit_when_checkpoint_rejected(
        self, runtime_with_checkpoints: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() skips commit when checkpoint is rejected."""
        sample_task.assigned_to = "U123456"
        runtime_with_checkpoints.task_queue.update_task_description(
            sample_task.id, sample_task.description
        )

        # Configure checkpoint to be rejected
        result = MagicMock()
        result.should_proceed = False
        result.status = CheckpointStatus.REJECTED
        result.rejection_reason = "Not ready"
        runtime_with_checkpoints.checkpoint_manager.request_approval = AsyncMock(
            return_value=result
        )

        with patch.object(Path, "exists", return_value=True):
            await runtime_with_checkpoints.execute(sample_task)

        # Git commit should not be called
        runtime_with_checkpoints.git.commit_workspace_changes.assert_not_called()


# -----------------------------------------------------------------------------
# Tests for execute() - Retry Integration
# -----------------------------------------------------------------------------


class TestExecuteRetry:
    """Tests for TaskRuntime.execute() retry integration."""

    @pytest.mark.asyncio
    async def test_increments_attempt_count_on_retry(
        self, runtime: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify execute() increments attempt count on retry."""
        call_count = 0

        async def fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Transient error: rate limit exceeded")
            return ("Success", [], [], 0, {}, "COMPLETE")

        runtime.claude.execute_task = AsyncMock(side_effect=fail_then_succeed)

        await runtime.execute(sample_task)

        # Should have retried and succeeded
        updated = runtime.task_queue.get_task(sample_task.id)
        assert updated.status == TaskStatus.COMPLETED
        assert call_count == 2


# -----------------------------------------------------------------------------
# Tests for _handle_completion()
# -----------------------------------------------------------------------------


class TestHandleCompletion:
    """Tests for TaskRuntime completion handling."""

    @pytest.mark.asyncio
    async def test_records_success_metrics(
        self, runtime: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify completion records success metrics."""
        await runtime.execute(sample_task)

        runtime.monitor.record_task_completion.assert_called()
        call_args = runtime.monitor.record_task_completion.call_args
        assert call_args.kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_logs_task_execution(
        self, runtime: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify completion logs task execution."""
        await runtime.execute(sample_task)

        runtime.perf_logger.log_task_execution.assert_called()
        call_args = runtime.perf_logger.log_task_execution.call_args
        assert call_args.kwargs["task_id"] == sample_task.id
        assert call_args.kwargs["success"] is True


# -----------------------------------------------------------------------------
# Tests for _handle_failure()
# -----------------------------------------------------------------------------


class TestHandleFailure:
    """Tests for TaskRuntime failure handling."""

    @pytest.mark.asyncio
    async def test_records_failure_metrics(
        self, runtime: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify failure records failure metrics."""
        runtime.claude.execute_task = AsyncMock(
            side_effect=RuntimeError("Test failure")
        )

        await runtime.execute(sample_task)

        runtime.monitor.record_task_completion.assert_called()
        call_args = runtime.monitor.record_task_completion.call_args
        assert call_args.kwargs["success"] is False

    @pytest.mark.asyncio
    async def test_appends_failure_to_report(
        self, runtime: TaskRuntime, sample_task: Task
    ) -> None:
        """Verify failure appends to report generator."""
        runtime.claude.execute_task = AsyncMock(
            side_effect=RuntimeError("Test failure")
        )

        await runtime.execute(sample_task)

        runtime.report_generator.append_task_completion.assert_called()


# -----------------------------------------------------------------------------
# Tests for _classify_blocker()
# -----------------------------------------------------------------------------


class TestClassifyBlocker:
    """Tests for TaskRuntime._classify_blocker() method."""

    def test_classifies_rate_limit_errors(self, runtime: TaskRuntime) -> None:
        """Verify rate limit errors are classified correctly."""
        result = runtime._classify_blocker("Error: rate limit exceeded")
        assert result == BlockerType.RATE_LIMIT

    def test_classifies_timeout_errors(self, runtime: TaskRuntime) -> None:
        """Verify timeout errors are classified correctly."""
        result = runtime._classify_blocker("Request timed out")
        assert result == BlockerType.TIMEOUT

    def test_classifies_permission_errors(self, runtime: TaskRuntime) -> None:
        """Verify permission errors are classified correctly."""
        result = runtime._classify_blocker("Permission denied")
        assert result == BlockerType.PERMISSION_ERROR

    def test_classifies_missing_dependency_errors(self, runtime: TaskRuntime) -> None:
        """Verify missing dependency errors are classified correctly."""
        result = runtime._classify_blocker("Module not found: missing_pkg")
        assert result == BlockerType.MISSING_DEPENDENCY

    def test_classifies_api_errors(self, runtime: TaskRuntime) -> None:
        """Verify API errors are classified correctly."""
        result = runtime._classify_blocker("API error: 500 Internal Server Error")
        assert result == BlockerType.API_ERROR

    def test_classifies_unknown_errors(self, runtime: TaskRuntime) -> None:
        """Verify unknown errors are classified as UNKNOWN."""
        result = runtime._classify_blocker("Some random error")
        assert result == BlockerType.UNKNOWN


# -----------------------------------------------------------------------------
# Tests for _refine_prompt_with_error()
# -----------------------------------------------------------------------------


class TestRefinePromptWithError:
    """Tests for TaskRuntime._refine_prompt_with_error() method."""

    def test_appends_error_context(self, runtime: TaskRuntime) -> None:
        """Verify _refine_prompt_with_error appends error context."""
        original = "Fix the bug in main.py"
        error = "AttributeError: 'NoneType' object has no attribute 'items'"

        refined = runtime._refine_prompt_with_error(original, error, attempt=2)

        assert "Previous attempt" in refined
        assert "AttributeError" in refined
        assert "different approach" in refined

    def test_truncates_long_errors(self, runtime: TaskRuntime) -> None:
        """Verify _refine_prompt_with_error truncates long errors."""
        original = "Fix the bug"
        error = "x" * 1000

        refined = runtime._refine_prompt_with_error(original, error, attempt=2)

        # Error is truncated to 500 chars + template overhead (~170 chars)
        # Total should be well under original + 1000 error chars
        assert len(refined) < len(original) + 700  # Allow for template + truncated error

    def test_updates_existing_refinement(self, runtime: TaskRuntime) -> None:
        """Verify _refine_prompt_with_error updates existing refinement."""
        original = "Fix the bug"
        first_refinement = runtime._refine_prompt_with_error(
            original, "Error 1", attempt=2
        )

        # Second refinement should replace the first
        second_refinement = runtime._refine_prompt_with_error(
            first_refinement, "Error 2", attempt=3
        )

        # Should only have one refinement section
        assert second_refinement.count("Previous attempt") == 1
        assert "Error 2" in second_refinement


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------


class TestRuntimeEdgeCases:
    """Edge case tests for TaskRuntime."""

    @pytest.mark.asyncio
    async def test_edge_case_empty_task_description(
        self, runtime: TaskRuntime, task_queue: TaskQueue
    ) -> None:
        """Edge case: task with empty description."""
        task = task_queue.add_task(description="")

        await runtime.execute(task)

        # Should complete without error
        updated = task_queue.get_task(task.id)
        assert updated.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_edge_case_task_with_no_project_context(
        self, runtime: TaskRuntime, task_queue: TaskQueue
    ) -> None:
        """Edge case: task with no project context."""
        task = task_queue.add_task(
            description="Task without project",
            project_id=None,
            project_name=None,
        )

        await runtime.execute(task)

        updated = task_queue.get_task(task.id)
        assert updated.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_edge_case_executor_returns_empty_result(
        self, runtime: TaskRuntime, sample_task: Task
    ) -> None:
        """Edge case: executor returns empty result."""
        async def empty_result(*args, **kwargs):
            return ("", [], [], 0, {}, "COMPLETE")

        runtime.claude.execute_task = AsyncMock(side_effect=empty_result)

        await runtime.execute(sample_task)

        updated = runtime.task_queue.get_task(sample_task.id)
        assert updated.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_edge_case_evaluator_incomplete_status(
        self, runtime: TaskRuntime, sample_task: Task
    ) -> None:
        """Edge case: evaluator returns INCOMPLETE status."""
        async def incomplete_result(*args, **kwargs):
            return ("Partial work done", ["file.py"], [], 0, {}, "INCOMPLETE")

        runtime.claude.execute_task = AsyncMock(side_effect=incomplete_result)

        await runtime.execute(sample_task)

        # Task should be marked as failed due to incomplete evaluation
        updated = runtime.task_queue.get_task(sample_task.id)
        assert updated.status == TaskStatus.FAILED
        assert "Evaluator" in updated.error_message

    @pytest.mark.asyncio
    async def test_edge_case_workspace_does_not_exist(
        self, runtime: TaskRuntime, sample_task: Task
    ) -> None:
        """Edge case: workspace path does not exist."""
        runtime.claude.get_workspace_path = MagicMock(
            return_value=Path("/nonexistent/path")
        )

        await runtime.execute(sample_task)

        # Should still complete, just skip git operations
        updated = runtime.task_queue.get_task(sample_task.id)
        assert updated.status == TaskStatus.COMPLETED


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestRuntimeIntegration:
    """Integration tests for TaskRuntime."""

    @pytest.mark.asyncio
    async def test_full_task_lifecycle(
        self, runtime: TaskRuntime, task_queue: TaskQueue
    ) -> None:
        """Integration: full task lifecycle (create -> execute -> complete)."""
        # Create task
        task = task_queue.add_task(
            description="Integration test task",
            priority=TaskPriority.SERIOUS,
            project_id="test-project",
            project_name="Test Project",
        )

        assert task.status == TaskStatus.PENDING

        # Execute
        await runtime.execute(task)

        # Verify completion
        updated = task_queue.get_task(task.id)
        assert updated.status == TaskStatus.COMPLETED
        assert updated.result_id is not None
        assert updated.completed_at is not None

        # Verify metrics were recorded
        runtime.monitor.record_task_completion.assert_called()
        runtime.report_generator.append_task_completion.assert_called()
        runtime.scheduler.record_task_usage.assert_called()

    @pytest.mark.asyncio
    async def test_task_with_all_integrations(
        self,
        runtime: TaskRuntime,
        task_queue: TaskQueue,
        mock_notification_manager: AsyncMock,
        mock_stream_manager: AsyncMock,
        mock_checkpoint_manager: AsyncMock,
    ) -> None:
        """Integration: task with notifications, streaming, and checkpoints."""
        runtime.notification_manager = mock_notification_manager
        runtime.stream_manager = mock_stream_manager
        runtime.checkpoint_manager = mock_checkpoint_manager

        task = task_queue.add_task(
            description="Full integration test",
            priority=TaskPriority.SERIOUS,
        )
        task.assigned_to = "U123456"

        with patch.object(Path, "exists", return_value=True):
            await runtime.execute(task)

        # Verify all integrations were called
        mock_notification_manager.start_task_tracking.assert_called()
        mock_notification_manager.notify_phase_transition.assert_called()
        mock_notification_manager.notify_completion.assert_called()
        mock_notification_manager.stop_task_tracking.assert_called()

        mock_stream_manager.start_stream.assert_called()
        mock_stream_manager.finalize_stream.assert_called()

        mock_checkpoint_manager.request_approval.assert_called()
