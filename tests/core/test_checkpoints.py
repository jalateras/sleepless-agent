"""Tests for checkpoint management for human-in-the-loop approval flows.

This module tests the CheckpointManager class which handles creation, resolution,
and waiting for approval checkpoints. It verifies Slack integration, timeout
handling, and database persistence.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Generator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sleepless_agent.core.checkpoints import (
    CheckpointConfig,
    CheckpointManager,
    CheckpointResult,
)
from sleepless_agent.core.models import (
    Base,
    Checkpoint,
    CheckpointStatus,
    CheckpointType,
    Task,
    TaskPriority,
    TaskStatus,
    init_db,
)


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
def default_config() -> CheckpointConfig:
    """Create a default checkpoint configuration."""
    return CheckpointConfig(
        enabled=True,
        post_plan=True,
        pre_commit=True,
        pre_pr=True,
        timeout_minutes=60,
        timeout_behavior="notify",
        poll_interval_seconds=0.01,  # Fast polling for tests
    )


@pytest.fixture
def disabled_config() -> CheckpointConfig:
    """Create a configuration with checkpoints disabled."""
    return CheckpointConfig(
        enabled=False,
        post_plan=False,
        pre_commit=False,
        pre_pr=False,
        timeout_minutes=60,
        timeout_behavior="notify",
    )


@pytest.fixture
def manager(temp_db: str, default_config: CheckpointConfig) -> CheckpointManager:
    """Create a CheckpointManager for testing."""
    return CheckpointManager(
        db_path=temp_db,
        config=default_config,
        slack_client=None,
    )


@pytest.fixture
def manager_with_slack(
    temp_db: str, default_config: CheckpointConfig
) -> tuple[CheckpointManager, MagicMock]:
    """Create a CheckpointManager with a mock Slack client."""
    mock_slack = MagicMock()
    mock_slack.chat_postMessage.return_value = {"ts": "1234567890.123456"}

    mgr = CheckpointManager(
        db_path=temp_db,
        config=default_config,
        slack_client=mock_slack,
    )
    return mgr, mock_slack


@pytest.fixture
def sample_task(temp_db: str) -> Task:
    """Create a sample task for testing."""
    task = Task(
        id=1,
        description="Test task for checkpoint testing",
        priority=TaskPriority.SERIOUS,
        status=TaskStatus.IN_PROGRESS,
    )
    return task


# -----------------------------------------------------------------------------
# Tests for CheckpointConfig
# -----------------------------------------------------------------------------


class TestCheckpointConfig:
    """Tests for CheckpointConfig class."""

    def test_is_checkpoint_enabled_when_globally_disabled(self) -> None:
        """Verify all checkpoint types are disabled when globally disabled."""
        config = CheckpointConfig(enabled=False, pre_commit=True, pre_pr=True)

        assert config.is_checkpoint_enabled(CheckpointType.POST_PLAN) is False
        assert config.is_checkpoint_enabled(CheckpointType.PRE_COMMIT) is False
        assert config.is_checkpoint_enabled(CheckpointType.PRE_PR) is False

    def test_is_checkpoint_enabled_per_type(self) -> None:
        """Verify individual checkpoint types can be enabled/disabled."""
        config = CheckpointConfig(
            enabled=True,
            post_plan=False,
            pre_commit=True,
            pre_pr=False,
        )

        assert config.is_checkpoint_enabled(CheckpointType.POST_PLAN) is False
        assert config.is_checkpoint_enabled(CheckpointType.PRE_COMMIT) is True
        assert config.is_checkpoint_enabled(CheckpointType.PRE_PR) is False

    def test_from_dict_creates_config(self) -> None:
        """Verify CheckpointConfig.from_dict() creates correct configuration."""
        config_dict = {
            "enabled": True,
            "global_defaults": {
                "post_plan": True,
                "pre_commit": False,
                "pre_pr": True,
            },
            "timeout_minutes": 30,
            "timeout_behavior": "abort",
            "poll_interval_seconds": 10.0,
        }

        config = CheckpointConfig.from_dict(config_dict)

        assert config.enabled is True
        assert config.post_plan is True
        assert config.pre_commit is False
        assert config.pre_pr is True
        assert config.timeout_minutes == 30
        assert config.timeout_behavior == "abort"
        assert config.poll_interval_seconds == 10.0

    def test_from_dict_with_defaults(self) -> None:
        """Verify CheckpointConfig.from_dict() uses defaults for missing keys."""
        config = CheckpointConfig.from_dict({})

        assert config.enabled is True
        assert config.post_plan is False
        assert config.pre_commit is True
        assert config.pre_pr is True
        assert config.timeout_minutes == 60
        assert config.timeout_behavior == "notify"


# -----------------------------------------------------------------------------
# Tests for create_checkpoint()
# -----------------------------------------------------------------------------


class TestCreateCheckpoint:
    """Tests for CheckpointManager.create_checkpoint() method."""

    def test_create_checkpoint_when_type_disabled(
        self, temp_db: str, sample_task: Task
    ) -> None:
        """Verify create_checkpoint returns None when checkpoint type is disabled."""
        config = CheckpointConfig(
            enabled=True,
            post_plan=False,  # Disabled
            pre_commit=True,
            pre_pr=True,
        )
        manager = CheckpointManager(temp_db, config)

        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.POST_PLAN,
            title="Test checkpoint",
        )

        assert checkpoint is None

    def test_create_checkpoint_when_globally_disabled(
        self, temp_db: str, disabled_config: CheckpointConfig, sample_task: Task
    ) -> None:
        """Verify create_checkpoint returns None when checkpoints globally disabled."""
        manager = CheckpointManager(temp_db, disabled_config)

        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Test checkpoint",
        )

        assert checkpoint is None

    def test_create_checkpoint_sets_correct_expiration_time(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Verify checkpoint expiration time is set correctly based on config."""
        before = datetime.now(timezone.utc).replace(tzinfo=None)

        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Test checkpoint",
        )

        after = datetime.now(timezone.utc).replace(tzinfo=None)

        assert checkpoint is not None
        assert checkpoint.expires_at is not None

        # Should be approximately timeout_minutes (60) from now
        expected_expires = before + timedelta(minutes=60)
        actual_expires = checkpoint.expires_at

        # Allow 1 second tolerance for test execution
        assert abs((actual_expires - expected_expires).total_seconds()) < 1

    def test_create_checkpoint_stores_details_as_json(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Verify checkpoint details are stored as JSON."""
        details = {
            "files": ["src/main.py", "src/utils.py"],
            "pr_title": "Add new feature",
        }

        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_PR,
            title="Review PR",
            details=details,
        )

        assert checkpoint is not None
        assert checkpoint.details is not None
        parsed_details = json.loads(checkpoint.details)
        assert parsed_details == details

    def test_create_checkpoint_persists_to_database(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Verify checkpoint is persisted and can be retrieved."""
        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Test persistence",
            channel_id="C123456",
            thread_ts="1234567890.123456",
        )

        assert checkpoint is not None

        # Retrieve from database
        retrieved = manager.get_checkpoint(checkpoint.id)

        assert retrieved is not None
        assert retrieved.id == checkpoint.id
        assert retrieved.title == "Test persistence"
        assert retrieved.channel_id == "C123456"
        assert retrieved.status == CheckpointStatus.PENDING


# -----------------------------------------------------------------------------
# Tests for request_approval()
# -----------------------------------------------------------------------------


class TestRequestApproval:
    """Tests for CheckpointManager.request_approval() method."""

    @pytest.mark.asyncio
    async def test_request_approval_sends_slack_message(
        self, manager_with_slack: tuple[CheckpointManager, MagicMock], sample_task: Task
    ) -> None:
        """Verify request_approval sends Slack message with buttons."""
        manager, mock_slack = manager_with_slack

        # Pre-resolve the checkpoint so wait_for_resolution returns immediately
        async def auto_approve():
            await asyncio.sleep(0.02)  # Small delay
            checkpoint = manager.get_pending_checkpoint(sample_task.id)
            if checkpoint:
                manager.resolve(checkpoint.id, CheckpointStatus.APPROVED, "U123")

        asyncio.create_task(auto_approve())

        result = await manager.request_approval(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Approve commit",
            channel_id="C123456",
        )

        # Verify Slack was called
        mock_slack.chat_postMessage.assert_called()
        call_args = mock_slack.chat_postMessage.call_args

        # Verify message structure
        assert call_args.kwargs["channel"] == "C123456"
        assert "blocks" in call_args.kwargs
        assert "Approval required" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_request_approval_skipped_when_disabled(
        self, temp_db: str, sample_task: Task
    ) -> None:
        """Verify request_approval returns SKIPPED when checkpoint type disabled."""
        config = CheckpointConfig(
            enabled=True,
            pre_commit=False,  # Disabled
        )
        manager = CheckpointManager(temp_db, config)

        result = await manager.request_approval(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Should be skipped",
        )

        assert result.status == CheckpointStatus.SKIPPED
        assert result.approved is True
        assert result.should_proceed is True


# -----------------------------------------------------------------------------
# Tests for resolve()
# -----------------------------------------------------------------------------


class TestResolve:
    """Tests for CheckpointManager.resolve() method."""

    def test_resolve_with_approval(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Verify resolve with approval updates status correctly."""
        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Test resolve",
        )
        assert checkpoint is not None

        resolved = manager.resolve(
            checkpoint_id=checkpoint.id,
            status=CheckpointStatus.APPROVED,
            resolved_by="U123456",
        )

        assert resolved is not None
        assert resolved.status == CheckpointStatus.APPROVED
        assert resolved.resolved_by == "U123456"
        assert resolved.resolved_at is not None

    def test_resolve_with_rejection(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Verify resolve with rejection updates status and reason."""
        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_PR,
            title="Test rejection",
        )
        assert checkpoint is not None

        resolved = manager.resolve(
            checkpoint_id=checkpoint.id,
            status=CheckpointStatus.REJECTED,
            resolved_by="U789",
            rejection_reason="Code needs more tests",
        )

        assert resolved is not None
        assert resolved.status == CheckpointStatus.REJECTED
        assert resolved.rejection_reason == "Code needs more tests"
        assert resolved.resolved_by == "U789"

    def test_resolve_already_resolved_checkpoint_is_idempotent(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Verify resolve on already-resolved checkpoint is idempotent."""
        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Test idempotent",
        )
        assert checkpoint is not None

        # First resolve
        first_resolve = manager.resolve(
            checkpoint_id=checkpoint.id,
            status=CheckpointStatus.APPROVED,
            resolved_by="U111",
        )
        first_resolved_at = first_resolve.resolved_at

        # Second resolve attempt (should not change anything)
        second_resolve = manager.resolve(
            checkpoint_id=checkpoint.id,
            status=CheckpointStatus.REJECTED,  # Try different status
            resolved_by="U222",
        )

        # Status should remain APPROVED from first resolve
        assert second_resolve.status == CheckpointStatus.APPROVED
        assert second_resolve.resolved_by == "U111"
        assert second_resolve.resolved_at == first_resolved_at

    def test_resolve_non_existent_checkpoint(
        self, manager: CheckpointManager
    ) -> None:
        """Verify resolve returns None for non-existent checkpoint ID."""
        result = manager.resolve(
            checkpoint_id=99999,  # Non-existent
            status=CheckpointStatus.APPROVED,
        )

        assert result is None


# -----------------------------------------------------------------------------
# Tests for expire_stale_checkpoints()
# -----------------------------------------------------------------------------


class TestExpireStaleCheckpoints:
    """Tests for CheckpointManager.expire_stale_checkpoints() method."""

    def test_expire_stale_checkpoints_finds_expired(
        self, temp_db: str, sample_task: Task
    ) -> None:
        """Verify expire_stale_checkpoints finds and expires old checkpoints."""
        # Create config with very short timeout
        config = CheckpointConfig(
            enabled=True,
            pre_commit=True,
            timeout_minutes=0,  # Immediate expiration
        )
        manager = CheckpointManager(temp_db, config)

        # Create checkpoint (will expire immediately)
        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Should expire",
        )
        assert checkpoint is not None

        # Wait a tiny bit to ensure expiration
        import time
        time.sleep(0.01)

        # Run expiration
        expired = manager.expire_stale_checkpoints()

        assert len(expired) == 1
        assert expired[0].id == checkpoint.id
        assert expired[0].status == CheckpointStatus.EXPIRED

    def test_expire_stale_checkpoints_ignores_resolved(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Verify expire_stale_checkpoints ignores already-resolved checkpoints."""
        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Already resolved",
        )
        assert checkpoint is not None

        # Resolve before expiration
        manager.resolve(checkpoint.id, CheckpointStatus.APPROVED)

        # Try to expire
        expired = manager.expire_stale_checkpoints()

        assert len(expired) == 0


# -----------------------------------------------------------------------------
# Tests for wait_for_resolution()
# -----------------------------------------------------------------------------


class TestWaitForResolution:
    """Tests for CheckpointManager.wait_for_resolution() method."""

    @pytest.mark.asyncio
    async def test_wait_for_resolution_timeout_proceed(
        self, temp_db: str, sample_task: Task
    ) -> None:
        """Verify wait_for_resolution with timeout_behavior='proceed' auto-approves."""
        config = CheckpointConfig(
            enabled=True,
            pre_commit=True,
            timeout_minutes=0,  # Immediate timeout
            timeout_behavior="proceed",
            poll_interval_seconds=0.01,
        )
        manager = CheckpointManager(temp_db, config, slack_client=None)

        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Test proceed timeout",
        )
        assert checkpoint is not None

        result = await manager.wait_for_resolution(checkpoint, sample_task)

        assert result.approved is True
        assert result.status == CheckpointStatus.EXPIRED
        assert result.should_proceed is True

    @pytest.mark.asyncio
    async def test_wait_for_resolution_timeout_abort(
        self, temp_db: str, sample_task: Task
    ) -> None:
        """Verify wait_for_resolution with timeout_behavior='abort' auto-rejects."""
        config = CheckpointConfig(
            enabled=True,
            pre_commit=True,
            timeout_minutes=0,  # Immediate timeout
            timeout_behavior="abort",
            poll_interval_seconds=0.01,
        )
        manager = CheckpointManager(temp_db, config, slack_client=None)

        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Test abort timeout",
        )
        assert checkpoint is not None

        result = await manager.wait_for_resolution(checkpoint, sample_task)

        assert result.approved is False
        assert result.status == CheckpointStatus.EXPIRED
        assert result.rejection_reason is not None
        assert "Timed out" in result.rejection_reason

    @pytest.mark.asyncio
    async def test_wait_for_resolution_timeout_notify(
        self, temp_db: str, sample_task: Task
    ) -> None:
        """Verify wait_for_resolution with timeout_behavior='notify' sends notification."""
        config = CheckpointConfig(
            enabled=True,
            pre_commit=True,
            timeout_minutes=0,  # Immediate timeout
            timeout_behavior="notify",
            poll_interval_seconds=0.01,
        )
        manager = CheckpointManager(temp_db, config, slack_client=None)

        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Test notify timeout",
        )
        assert checkpoint is not None

        result = await manager.wait_for_resolution(checkpoint, sample_task)

        assert result.approved is False
        assert result.status == CheckpointStatus.EXPIRED
        assert "manual review" in result.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_wait_for_resolution_returns_on_approval(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Verify wait_for_resolution returns when checkpoint is approved."""
        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Test approval resolution",
        )
        assert checkpoint is not None

        # Resolve in background
        async def resolve_later():
            await asyncio.sleep(0.02)
            manager.resolve(checkpoint.id, CheckpointStatus.APPROVED, "U123")

        asyncio.create_task(resolve_later())

        result = await manager.wait_for_resolution(checkpoint, sample_task)

        assert result.approved is True
        assert result.status == CheckpointStatus.APPROVED
        assert result.resolved_by == "U123"

    @pytest.mark.asyncio
    async def test_wait_for_resolution_returns_on_rejection(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Verify wait_for_resolution returns when checkpoint is rejected."""
        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Test rejection resolution",
        )
        assert checkpoint is not None

        # Reject in background
        async def reject_later():
            await asyncio.sleep(0.02)
            manager.resolve(
                checkpoint.id,
                CheckpointStatus.REJECTED,
                "U456",
                rejection_reason="Not ready yet",
            )

        asyncio.create_task(reject_later())

        result = await manager.wait_for_resolution(checkpoint, sample_task)

        assert result.approved is False
        assert result.status == CheckpointStatus.REJECTED
        assert result.rejection_reason == "Not ready yet"


# -----------------------------------------------------------------------------
# Tests for get_pending_checkpoint()
# -----------------------------------------------------------------------------


class TestGetPendingCheckpoints:
    """Tests for CheckpointManager.get_pending_checkpoint() method."""

    def test_get_pending_checkpoint_returns_pending(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Verify get_pending_checkpoint returns pending checkpoints for task."""
        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Pending checkpoint",
        )

        pending = manager.get_pending_checkpoint(sample_task.id)

        assert pending is not None
        assert pending.id == checkpoint.id
        assert pending.status == CheckpointStatus.PENDING

    def test_get_pending_checkpoint_excludes_resolved(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Verify get_pending_checkpoint excludes resolved checkpoints."""
        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Will be resolved",
        )
        assert checkpoint is not None

        # Resolve it
        manager.resolve(checkpoint.id, CheckpointStatus.APPROVED)

        pending = manager.get_pending_checkpoint(sample_task.id)

        assert pending is None

    def test_get_pending_checkpoint_for_nonexistent_task(
        self, manager: CheckpointManager
    ) -> None:
        """Verify get_pending_checkpoint returns None for task with no checkpoints."""
        pending = manager.get_pending_checkpoint(99999)

        assert pending is None


# -----------------------------------------------------------------------------
# Tests for CheckpointResult
# -----------------------------------------------------------------------------


class TestCheckpointResult:
    """Tests for CheckpointResult class."""

    def test_should_proceed_when_approved(self) -> None:
        """Verify should_proceed returns True when approved."""
        result = CheckpointResult(
            approved=True,
            status=CheckpointStatus.APPROVED,
        )

        assert result.should_proceed is True

    def test_should_proceed_when_skipped(self) -> None:
        """Verify should_proceed returns True when skipped."""
        result = CheckpointResult(
            approved=False,  # Not explicitly approved
            status=CheckpointStatus.SKIPPED,
        )

        assert result.should_proceed is True

    def test_should_not_proceed_when_rejected(self) -> None:
        """Verify should_proceed returns False when rejected."""
        result = CheckpointResult(
            approved=False,
            status=CheckpointStatus.REJECTED,
            rejection_reason="Rejected by user",
        )

        assert result.should_proceed is False

    def test_should_not_proceed_when_expired_abort(self) -> None:
        """Verify should_proceed returns False when expired with abort."""
        result = CheckpointResult(
            approved=False,
            status=CheckpointStatus.EXPIRED,
            rejection_reason="Timed out",
        )

        assert result.should_proceed is False


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------


class TestCheckpointEdgeCases:
    """Edge case tests for checkpoint management."""

    def test_edge_case_timeout_zero_minutes(
        self, temp_db: str, sample_task: Task
    ) -> None:
        """Edge case: timeout = 0 minutes should expire immediately."""
        config = CheckpointConfig(
            enabled=True,
            pre_commit=True,
            timeout_minutes=0,
        )
        manager = CheckpointManager(temp_db, config)

        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Zero timeout",
        )

        assert checkpoint is not None
        # Should be immediately at or past expiration
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        assert checkpoint.expires_at <= now + timedelta(seconds=1)

    def test_edge_case_empty_rejection_reason(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Edge case: empty rejection reason should be handled.

        The implementation only sets rejection_reason if it's truthy,
        so an empty string is not stored.
        """
        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Test empty reason",
        )
        assert checkpoint is not None

        resolved = manager.resolve(
            checkpoint_id=checkpoint.id,
            status=CheckpointStatus.REJECTED,
            resolved_by="U123",
            rejection_reason="",  # Empty string
        )

        # Empty string is falsy, so rejection_reason is not set
        assert resolved.rejection_reason is None

    def test_edge_case_none_rejection_reason(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Edge case: None rejection reason should be handled."""
        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Test none reason",
        )
        assert checkpoint is not None

        resolved = manager.resolve(
            checkpoint_id=checkpoint.id,
            status=CheckpointStatus.REJECTED,
            resolved_by="U123",
            rejection_reason=None,
        )

        assert resolved.rejection_reason is None

    def test_cleanup_old_checkpoints(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Verify cleanup_old_checkpoints removes old resolved checkpoints."""
        # Create and resolve a checkpoint
        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Old checkpoint",
        )
        assert checkpoint is not None
        manager.resolve(checkpoint.id, CheckpointStatus.APPROVED)

        # Cleanup with 0 days (should remove everything resolved)
        count = manager.cleanup_old_checkpoints(days=0)

        # Verify cleanup occurred
        retrieved = manager.get_checkpoint(checkpoint.id)
        assert retrieved is None or count > 0

    def test_update_message_ts(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Verify update_message_ts updates the Slack message timestamp."""
        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="Test message ts",
        )
        assert checkpoint is not None
        assert checkpoint.message_ts is None

        manager.update_message_ts(checkpoint.id, "1234567890.999999")

        # Retrieve and verify
        updated = manager.get_checkpoint(checkpoint.id)
        assert updated is not None
        assert updated.message_ts == "1234567890.999999"

    @pytest.mark.asyncio
    async def test_send_checkpoint_message_without_slack_client(
        self, manager: CheckpointManager, sample_task: Task
    ) -> None:
        """Verify send_checkpoint_message handles missing Slack client."""
        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="No Slack",
            channel_id="C123",
        )
        assert checkpoint is not None

        # Should return None without error
        result = await manager.send_checkpoint_message(checkpoint, sample_task)
        assert result is None

    @pytest.mark.asyncio
    async def test_send_checkpoint_message_without_channel(
        self, manager_with_slack: tuple[CheckpointManager, MagicMock], sample_task: Task
    ) -> None:
        """Verify send_checkpoint_message handles missing channel ID."""
        manager, mock_slack = manager_with_slack

        checkpoint = manager.create_checkpoint(
            task=sample_task,
            checkpoint_type=CheckpointType.PRE_COMMIT,
            title="No channel",
            channel_id=None,  # No channel
        )
        assert checkpoint is not None

        result = await manager.send_checkpoint_message(checkpoint, sample_task)

        assert result is None
        mock_slack.chat_postMessage.assert_not_called()
