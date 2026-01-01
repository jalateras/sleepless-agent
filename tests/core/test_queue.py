"""Tests for task queue management and state transitions.

This module tests the TaskQueue class which handles task lifecycle management,
priority ordering, and state transitions. It verifies valid and invalid state
transitions, database persistence, and edge cases.
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Generator

import pytest

from sleepless_agent.core.models import (
    Base,
    Task,
    TaskPriority,
    TaskStatus,
    init_db,
)
from sleepless_agent.core.queue import TaskQueue


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_db() -> Generator[str, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Initialize database schema
    engine = init_db(db_path)
    Base.metadata.create_all(engine)

    yield db_path


@pytest.fixture
def queue(temp_db: str) -> TaskQueue:
    """Create a TaskQueue for testing."""
    return TaskQueue(db_path=temp_db)


# -----------------------------------------------------------------------------
# Tests for add_task()
# -----------------------------------------------------------------------------


class TestAddTask:
    """Tests for TaskQueue.add_task() method."""

    def test_add_task_creates_task_in_pending_state(self, queue: TaskQueue) -> None:
        """Verify add_task creates a task in PENDING state."""
        task = queue.add_task(
            description="Test task",
            priority=TaskPriority.THOUGHT,
        )

        assert task.id is not None
        assert task.status == TaskStatus.PENDING
        assert task.description == "Test task"
        assert task.priority == TaskPriority.THOUGHT

    def test_add_task_with_all_parameters(self, queue: TaskQueue) -> None:
        """Verify add_task stores all provided parameters."""
        context = {"key": "value", "nested": {"inner": 123}}

        task = queue.add_task(
            description="Full task",
            priority=TaskPriority.SERIOUS,
            context=context,
            slack_user_id="U123456",
            slack_thread_ts="1234567890.123456",
            project_id="proj-001",
            project_name="Test Project",
        )

        assert task.description == "Full task"
        assert task.priority == TaskPriority.SERIOUS
        assert json.loads(task.context) == context
        assert task.assigned_to == "U123456"
        assert task.slack_thread_ts == "1234567890.123456"
        assert task.project_id == "proj-001"
        assert task.project_name == "Test Project"

    def test_add_task_sets_attempt_count_to_zero(self, queue: TaskQueue) -> None:
        """Verify new tasks have attempt_count = 0."""
        task = queue.add_task(description="Test task")

        assert task.attempt_count == 0

    def test_add_task_sets_created_at(self, queue: TaskQueue) -> None:
        """Verify new tasks have created_at timestamp."""
        before = datetime.now(timezone.utc).replace(tzinfo=None)
        task = queue.add_task(description="Test task")
        after = datetime.now(timezone.utc).replace(tzinfo=None)

        assert task.created_at is not None
        assert before <= task.created_at <= after

    def test_add_task_with_default_priority(self, queue: TaskQueue) -> None:
        """Verify add_task uses THOUGHT as default priority."""
        task = queue.add_task(description="Default priority task")

        assert task.priority == TaskPriority.THOUGHT


# -----------------------------------------------------------------------------
# Tests for Valid State Transitions
# -----------------------------------------------------------------------------


class TestValidStateTransitions:
    """Tests for valid task state transitions."""

    def test_pending_to_in_progress_via_mark_in_progress(
        self, queue: TaskQueue
    ) -> None:
        """Valid: PENDING -> IN_PROGRESS via mark_in_progress()."""
        task = queue.add_task(description="Test task")
        assert task.status == TaskStatus.PENDING

        updated = queue.mark_in_progress(task.id)

        assert updated is not None
        assert updated.status == TaskStatus.IN_PROGRESS
        assert updated.started_at is not None

    def test_in_progress_to_completed_via_mark_completed(
        self, queue: TaskQueue
    ) -> None:
        """Valid: IN_PROGRESS -> COMPLETED via mark_completed()."""
        task = queue.add_task(description="Test task")
        queue.mark_in_progress(task.id)

        updated = queue.mark_completed(task.id, result_id=42)

        assert updated is not None
        assert updated.status == TaskStatus.COMPLETED
        assert updated.completed_at is not None
        assert updated.result_id == 42

    def test_in_progress_to_failed_via_mark_failed(self, queue: TaskQueue) -> None:
        """Valid: IN_PROGRESS -> FAILED via mark_failed()."""
        task = queue.add_task(description="Test task")
        queue.mark_in_progress(task.id)

        updated = queue.mark_failed(task.id, error_message="Something went wrong")

        assert updated is not None
        assert updated.status == TaskStatus.FAILED
        assert updated.error_message == "Something went wrong"
        assert updated.completed_at is not None

    def test_pending_to_cancelled_via_cancel_task(self, queue: TaskQueue) -> None:
        """Valid: PENDING -> CANCELLED via cancel_task()."""
        task = queue.add_task(description="Test task")
        assert task.status == TaskStatus.PENDING

        updated = queue.cancel_task(task.id)

        assert updated is not None
        assert updated.status == TaskStatus.CANCELLED
        assert updated.deleted_at is not None


# -----------------------------------------------------------------------------
# Tests for Invalid State Transitions
# -----------------------------------------------------------------------------


class TestInvalidStateTransitions:
    """Tests for invalid task state transitions.

    Note: The current implementation does not enforce state transition validation
    at the database level. These tests document the expected behavior if such
    validation were added, and verify the current behavior.
    """

    def test_completed_cannot_transition_to_in_progress(
        self, queue: TaskQueue
    ) -> None:
        """Invalid: COMPLETED -> IN_PROGRESS should not change state meaningfully.

        Note: The current implementation will update the task, but this documents
        that logically a completed task should not go back to in_progress.
        """
        task = queue.add_task(description="Test task")
        queue.mark_in_progress(task.id)
        queue.mark_completed(task.id)

        # Verify task is completed
        completed_task = queue.get_task(task.id)
        assert completed_task.status == TaskStatus.COMPLETED

        # Attempting to mark in_progress again - implementation allows it
        # but semantically this is invalid
        updated = queue.mark_in_progress(task.id)

        # Current behavior: updates to IN_PROGRESS (no validation)
        # A stricter implementation would return None or raise an error
        assert updated is not None
        # Document current behavior
        assert updated.status == TaskStatus.IN_PROGRESS

    def test_cancelled_cannot_transition_to_pending(self, queue: TaskQueue) -> None:
        """Invalid: CANCELLED -> PENDING should not be possible.

        The current implementation does not have a method to transition back
        to PENDING, so this documents that no such operation exists.
        """
        task = queue.add_task(description="Test task")
        queue.cancel_task(task.id)

        # Verify task is cancelled
        cancelled_task = queue.get_task(task.id)
        assert cancelled_task.status == TaskStatus.CANCELLED

        # There is no un-cancel method, so this transition is prevented by design

    def test_failed_cannot_transition_to_completed(self, queue: TaskQueue) -> None:
        """Invalid: FAILED -> COMPLETED should preserve failed state.

        Note: Current implementation allows mark_completed on any task.
        This documents expected behavior.
        """
        task = queue.add_task(description="Test task")
        queue.mark_in_progress(task.id)
        queue.mark_failed(task.id, error_message="Failed")

        failed_task = queue.get_task(task.id)
        assert failed_task.status == TaskStatus.FAILED

        # Current behavior: allows completing a failed task
        updated = queue.mark_completed(task.id)

        # Document current behavior (implementation allows this)
        assert updated is not None

    def test_cancel_task_only_works_on_pending(self, queue: TaskQueue) -> None:
        """Verify cancel_task only cancels PENDING tasks."""
        task = queue.add_task(description="Test task")
        queue.mark_in_progress(task.id)

        # Try to cancel an in-progress task
        result = queue.cancel_task(task.id)

        # cancel_task checks status and only cancels PENDING tasks
        assert result is not None
        # Task should still be in_progress since cancel checks status
        assert result.status == TaskStatus.IN_PROGRESS


# -----------------------------------------------------------------------------
# Tests for get_next_task() / get_pending_tasks()
# -----------------------------------------------------------------------------


class TestGetNextTask:
    """Tests for retrieving the next task from the queue."""

    def test_get_pending_tasks_returns_highest_priority_first(
        self, queue: TaskQueue
    ) -> None:
        """Verify get_pending_tasks returns tasks sorted by priority."""
        # Add tasks in mixed order
        task_thought = queue.add_task(
            description="Thought task", priority=TaskPriority.THOUGHT
        )
        task_serious = queue.add_task(
            description="Serious task", priority=TaskPriority.SERIOUS
        )
        task_generated = queue.add_task(
            description="Generated task", priority=TaskPriority.GENERATED
        )

        pending = queue.get_pending_tasks(limit=10)

        # SERIOUS should come first, then THOUGHT, then GENERATED
        assert len(pending) == 3
        assert pending[0].id == task_serious.id
        assert pending[1].id == task_thought.id
        assert pending[2].id == task_generated.id

    def test_get_pending_tasks_returns_none_when_queue_empty(
        self, queue: TaskQueue
    ) -> None:
        """Verify get_pending_tasks returns empty list when no pending tasks."""
        pending = queue.get_pending_tasks()

        assert pending == []

    def test_get_pending_tasks_excludes_non_pending(self, queue: TaskQueue) -> None:
        """Verify get_pending_tasks excludes completed, failed, cancelled tasks."""
        task1 = queue.add_task(description="Will complete")
        task2 = queue.add_task(description="Will fail")
        task3 = queue.add_task(description="Will cancel")
        task4 = queue.add_task(description="Stays pending")

        queue.mark_in_progress(task1.id)
        queue.mark_completed(task1.id)

        queue.mark_in_progress(task2.id)
        queue.mark_failed(task2.id, "Error")

        queue.cancel_task(task3.id)

        pending = queue.get_pending_tasks()

        assert len(pending) == 1
        assert pending[0].id == task4.id

    def test_get_pending_tasks_respects_limit(self, queue: TaskQueue) -> None:
        """Verify get_pending_tasks respects the limit parameter."""
        for i in range(10):
            queue.add_task(description=f"Task {i}")

        pending = queue.get_pending_tasks(limit=3)

        assert len(pending) == 3

    def test_get_pending_tasks_orders_by_created_at_within_priority(
        self, queue: TaskQueue
    ) -> None:
        """Verify tasks with same priority are ordered by created_at."""
        task1 = queue.add_task(
            description="First", priority=TaskPriority.THOUGHT
        )
        time.sleep(0.01)  # Ensure different timestamps
        task2 = queue.add_task(
            description="Second", priority=TaskPriority.THOUGHT
        )
        time.sleep(0.01)
        task3 = queue.add_task(
            description="Third", priority=TaskPriority.THOUGHT
        )

        pending = queue.get_pending_tasks()

        # Should be ordered by created_at (oldest first)
        assert pending[0].id == task1.id
        assert pending[1].id == task2.id
        assert pending[2].id == task3.id


# -----------------------------------------------------------------------------
# Tests for attempt_count and timestamps
# -----------------------------------------------------------------------------


class TestAttemptCountAndTimestamps:
    """Tests for attempt counting and timestamp updates."""

    def test_mark_in_progress_increments_attempt_count(
        self, queue: TaskQueue
    ) -> None:
        """Verify mark_in_progress increments attempt_count."""
        task = queue.add_task(description="Test task")
        assert task.attempt_count == 0

        updated = queue.mark_in_progress(task.id)

        assert updated.attempt_count == 1

    def test_increment_attempt_count_method(self, queue: TaskQueue) -> None:
        """Verify increment_attempt_count increments properly."""
        task = queue.add_task(description="Test task")
        assert task.attempt_count == 0

        queue.increment_attempt_count(task.id)
        updated = queue.get_task(task.id)
        assert updated.attempt_count == 1

        queue.increment_attempt_count(task.id)
        updated = queue.get_task(task.id)
        assert updated.attempt_count == 2

    def test_mark_in_progress_sets_started_at(self, queue: TaskQueue) -> None:
        """Verify mark_in_progress sets started_at timestamp."""
        task = queue.add_task(description="Test task")
        assert task.started_at is None

        before = datetime.now(timezone.utc).replace(tzinfo=None)
        updated = queue.mark_in_progress(task.id)
        after = datetime.now(timezone.utc).replace(tzinfo=None)

        assert updated.started_at is not None
        assert before <= updated.started_at <= after

    def test_mark_completed_sets_completed_at(self, queue: TaskQueue) -> None:
        """Verify mark_completed sets completed_at timestamp."""
        task = queue.add_task(description="Test task")
        queue.mark_in_progress(task.id)

        before = datetime.now(timezone.utc).replace(tzinfo=None)
        updated = queue.mark_completed(task.id)
        after = datetime.now(timezone.utc).replace(tzinfo=None)

        assert updated.completed_at is not None
        assert before <= updated.completed_at <= after

    def test_mark_failed_sets_completed_at(self, queue: TaskQueue) -> None:
        """Verify mark_failed sets completed_at timestamp."""
        task = queue.add_task(description="Test task")
        queue.mark_in_progress(task.id)

        before = datetime.now(timezone.utc).replace(tzinfo=None)
        updated = queue.mark_failed(task.id, "Error message")
        after = datetime.now(timezone.utc).replace(tzinfo=None)

        assert updated.completed_at is not None
        assert before <= updated.completed_at <= after


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------


class TestQueueEdgeCases:
    """Edge case tests for task queue."""

    def test_edge_case_empty_task_description(self, queue: TaskQueue) -> None:
        """Edge case: empty task description should be handled."""
        task = queue.add_task(description="")

        assert task.id is not None
        assert task.description == ""
        assert task.status == TaskStatus.PENDING

    def test_edge_case_very_long_description(self, queue: TaskQueue) -> None:
        """Edge case: very long task description should be stored."""
        long_description = "x" * 10000  # 10KB description

        task = queue.add_task(description=long_description)

        retrieved = queue.get_task(task.id)
        assert retrieved.description == long_description

    def test_edge_case_get_nonexistent_task(self, queue: TaskQueue) -> None:
        """Edge case: get_task for non-existent ID returns None."""
        result = queue.get_task(99999)

        assert result is None

    def test_edge_case_mark_in_progress_nonexistent_task(
        self, queue: TaskQueue
    ) -> None:
        """Edge case: mark_in_progress on non-existent task returns None."""
        result = queue.mark_in_progress(99999)

        assert result is None

    def test_edge_case_mark_completed_nonexistent_task(
        self, queue: TaskQueue
    ) -> None:
        """Edge case: mark_completed on non-existent task returns None."""
        result = queue.mark_completed(99999)

        assert result is None

    def test_edge_case_mark_failed_nonexistent_task(self, queue: TaskQueue) -> None:
        """Edge case: mark_failed on non-existent task returns None."""
        result = queue.mark_failed(99999, "Error")

        assert result is None

    def test_edge_case_cancel_nonexistent_task(self, queue: TaskQueue) -> None:
        """Edge case: cancel_task on non-existent task returns None."""
        result = queue.cancel_task(99999)

        assert result is None

    def test_edge_case_task_ids_are_unique(self, queue: TaskQueue) -> None:
        """Verify task IDs are unique and auto-incrementing."""
        task1 = queue.add_task(description="Task 1")
        task2 = queue.add_task(description="Task 2")
        task3 = queue.add_task(description="Task 3")

        ids = [task1.id, task2.id, task3.id]

        # All IDs should be unique
        assert len(set(ids)) == 3

        # IDs should be sequential
        assert task2.id == task1.id + 1
        assert task3.id == task2.id + 1


# -----------------------------------------------------------------------------
# Tests for Queue Status and Statistics
# -----------------------------------------------------------------------------


class TestQueueStatus:
    """Tests for queue status and statistics methods."""

    def test_get_queue_status(self, queue: TaskQueue) -> None:
        """Verify get_queue_status returns correct counts."""
        # Create tasks in various states
        task1 = queue.add_task(description="Pending 1")
        task2 = queue.add_task(description="Pending 2")
        task3 = queue.add_task(description="Will complete")
        task4 = queue.add_task(description="Will fail")
        task5 = queue.add_task(description="Will cancel")

        queue.mark_in_progress(task3.id)
        queue.mark_completed(task3.id)

        queue.mark_in_progress(task4.id)
        queue.mark_failed(task4.id, "Error")

        queue.cancel_task(task5.id)

        status = queue.get_queue_status()

        assert status["total"] == 5
        assert status["pending"] == 2
        assert status["in_progress"] == 0
        assert status["completed"] == 1
        assert status["failed"] == 1

    def test_get_in_progress_tasks(self, queue: TaskQueue) -> None:
        """Verify get_in_progress_tasks returns only in-progress tasks."""
        task1 = queue.add_task(description="In progress 1")
        task2 = queue.add_task(description="In progress 2")
        task3 = queue.add_task(description="Pending")

        queue.mark_in_progress(task1.id)
        queue.mark_in_progress(task2.id)

        in_progress = queue.get_in_progress_tasks()

        assert len(in_progress) == 2
        assert all(t.status == TaskStatus.IN_PROGRESS for t in in_progress)


# -----------------------------------------------------------------------------
# Tests for Task Context
# -----------------------------------------------------------------------------


class TestTaskContext:
    """Tests for task context handling."""

    def test_get_task_context(self, queue: TaskQueue) -> None:
        """Verify get_task_context returns parsed JSON context."""
        context = {"key": "value", "number": 42}
        task = queue.add_task(description="With context", context=context)

        retrieved_context = queue.get_task_context(task.id)

        assert retrieved_context == context

    def test_get_task_context_returns_none_when_no_context(
        self, queue: TaskQueue
    ) -> None:
        """Verify get_task_context returns None when task has no context."""
        task = queue.add_task(description="No context")

        retrieved_context = queue.get_task_context(task.id)

        assert retrieved_context is None

    def test_get_task_context_for_nonexistent_task(self, queue: TaskQueue) -> None:
        """Verify get_task_context returns None for non-existent task."""
        retrieved_context = queue.get_task_context(99999)

        assert retrieved_context is None


# -----------------------------------------------------------------------------
# Tests for Update Methods
# -----------------------------------------------------------------------------


class TestUpdateMethods:
    """Tests for task update methods."""

    def test_update_priority(self, queue: TaskQueue) -> None:
        """Verify update_priority changes task priority."""
        task = queue.add_task(description="Test", priority=TaskPriority.THOUGHT)

        updated = queue.update_priority(task.id, TaskPriority.SERIOUS)

        assert updated.priority == TaskPriority.SERIOUS

    def test_update_task_description(self, queue: TaskQueue) -> None:
        """Verify update_task_description changes task description."""
        task = queue.add_task(description="Original description")

        updated = queue.update_task_description(task.id, "Updated description")

        assert updated.description == "Updated description"


# -----------------------------------------------------------------------------
# Tests for Project Operations
# -----------------------------------------------------------------------------


class TestProjectOperations:
    """Tests for project-related operations."""

    def test_get_projects_returns_project_info(self, queue: TaskQueue) -> None:
        """Verify get_projects returns project information with task counts."""
        queue.add_task(
            description="Task 1",
            project_id="proj-1",
            project_name="Project One",
        )
        queue.add_task(
            description="Task 2",
            project_id="proj-1",
            project_name="Project One",
        )
        queue.add_task(
            description="Task 3",
            project_id="proj-2",
            project_name="Project Two",
        )

        projects = queue.get_projects()

        assert len(projects) == 2

        proj1 = next(p for p in projects if p["project_id"] == "proj-1")
        assert proj1["project_name"] == "Project One"
        assert proj1["total_tasks"] == 2

    def test_get_project_by_id(self, queue: TaskQueue) -> None:
        """Verify get_project_by_id returns project details."""
        task = queue.add_task(
            description="Task 1",
            project_id="proj-test",
            project_name="Test Project",
        )

        project = queue.get_project_by_id("proj-test")

        assert project is not None
        assert project["project_id"] == "proj-test"
        assert project["project_name"] == "Test Project"
        assert project["total_tasks"] == 1

    def test_get_project_by_id_nonexistent(self, queue: TaskQueue) -> None:
        """Verify get_project_by_id returns None for non-existent project."""
        project = queue.get_project_by_id("nonexistent")

        assert project is None

    def test_delete_project_cancels_pending_tasks(self, queue: TaskQueue) -> None:
        """Verify delete_project soft-deletes pending tasks in project."""
        task1 = queue.add_task(
            description="Pending task",
            project_id="proj-delete",
        )
        task2 = queue.add_task(
            description="Another pending",
            project_id="proj-delete",
        )
        task3 = queue.add_task(
            description="In progress",
            project_id="proj-delete",
        )
        queue.mark_in_progress(task3.id)

        count = queue.delete_project("proj-delete")

        # Only pending tasks should be cancelled (2 of 3)
        assert count == 2

        # Verify states
        assert queue.get_task(task1.id).status == TaskStatus.CANCELLED
        assert queue.get_task(task2.id).status == TaskStatus.CANCELLED
        assert queue.get_task(task3.id).status == TaskStatus.IN_PROGRESS


# -----------------------------------------------------------------------------
# Tests for Timeout Handling
# -----------------------------------------------------------------------------


class TestTimeoutHandling:
    """Tests for task timeout handling."""

    def test_timeout_expired_tasks(self, queue: TaskQueue) -> None:
        """Verify timeout_expired_tasks marks old in-progress tasks as failed."""
        task = queue.add_task(description="Will timeout")
        queue.mark_in_progress(task.id)

        # Sleep briefly then check with a 1-second timeout
        # The task must have started_at older than max_age_seconds
        time.sleep(1.1)

        timed_out = queue.timeout_expired_tasks(max_age_seconds=1)

        assert len(timed_out) == 1
        assert timed_out[0].id == task.id
        assert timed_out[0].status == TaskStatus.FAILED
        assert "Timed out" in timed_out[0].error_message

    def test_timeout_expired_tasks_ignores_completed(self, queue: TaskQueue) -> None:
        """Verify timeout_expired_tasks ignores completed tasks."""
        task = queue.add_task(description="Completed")
        queue.mark_in_progress(task.id)
        queue.mark_completed(task.id)

        timed_out = queue.timeout_expired_tasks(max_age_seconds=0)

        assert len(timed_out) == 0

    def test_timeout_expired_tasks_with_zero_max_age(self, queue: TaskQueue) -> None:
        """Verify timeout_expired_tasks returns empty with max_age <= 0."""
        task = queue.add_task(description="Test")
        queue.mark_in_progress(task.id)

        # max_age_seconds <= 0 should return empty list
        timed_out = queue.timeout_expired_tasks(max_age_seconds=-1)

        assert timed_out == []


# -----------------------------------------------------------------------------
# Tests for Recent and Failed Tasks
# -----------------------------------------------------------------------------


class TestRecentAndFailedTasks:
    """Tests for retrieving recent and failed tasks."""

    def test_get_recent_tasks(self, queue: TaskQueue) -> None:
        """Verify get_recent_tasks returns tasks in reverse chronological order."""
        task1 = queue.add_task(description="First")
        time.sleep(0.01)
        task2 = queue.add_task(description="Second")
        time.sleep(0.01)
        task3 = queue.add_task(description="Third")

        recent = queue.get_recent_tasks(limit=10)

        # Most recent first
        assert recent[0].id == task3.id
        assert recent[1].id == task2.id
        assert recent[2].id == task1.id

    def test_get_failed_tasks(self, queue: TaskQueue) -> None:
        """Verify get_failed_tasks returns only failed tasks."""
        task1 = queue.add_task(description="Will fail 1")
        task2 = queue.add_task(description="Will fail 2")
        task3 = queue.add_task(description="Will complete")

        queue.mark_in_progress(task1.id)
        queue.mark_failed(task1.id, "Error 1")

        queue.mark_in_progress(task2.id)
        queue.mark_failed(task2.id, "Error 2")

        queue.mark_in_progress(task3.id)
        queue.mark_completed(task3.id)

        failed = queue.get_failed_tasks()

        assert len(failed) == 2
        assert all(t.status == TaskStatus.FAILED for t in failed)


# -----------------------------------------------------------------------------
# Tests for Pool Status
# -----------------------------------------------------------------------------


class TestPoolStatus:
    """Tests for connection pool status."""

    def test_get_pool_status(self, queue: TaskQueue) -> None:
        """Verify get_pool_status returns pool information."""
        status = queue.get_pool_status()

        assert "pool_class" in status
        assert "size" in status
        assert "checked_in" in status
        assert "checked_out" in status
