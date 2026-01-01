"""Checkpoint management for human-in-the-loop approval flows."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from sleepless_agent.core.models import (
    Checkpoint,
    CheckpointStatus,
    CheckpointType,
    Task,
)
from sleepless_agent.monitoring.logging import get_logger
from sleepless_agent.storage.sqlite import SQLiteStore

logger = get_logger(__name__)


class CheckpointConfig:
    """Configuration for checkpoint behavior."""

    def __init__(
        self,
        enabled: bool = True,
        post_plan: bool = False,
        pre_commit: bool = True,
        pre_pr: bool = True,
        timeout_minutes: int = 60,
        timeout_behavior: str = "notify",  # proceed, abort, notify
        poll_interval_seconds: float = 5.0,
    ):
        self.enabled = enabled
        self.post_plan = post_plan
        self.pre_commit = pre_commit
        self.pre_pr = pre_pr
        self.timeout_minutes = timeout_minutes
        self.timeout_behavior = timeout_behavior
        self.poll_interval_seconds = poll_interval_seconds

    @classmethod
    def from_dict(cls, config: dict) -> "CheckpointConfig":
        """Create config from dictionary (e.g., from config.yaml)."""
        global_defaults = config.get("global_defaults", {})
        return cls(
            enabled=config.get("enabled", True),
            post_plan=global_defaults.get("post_plan", False),
            pre_commit=global_defaults.get("pre_commit", True),
            pre_pr=global_defaults.get("pre_pr", True),
            timeout_minutes=config.get("timeout_minutes", 60),
            timeout_behavior=config.get("timeout_behavior", "notify"),
            poll_interval_seconds=config.get("poll_interval_seconds", 5.0),
        )

    def is_checkpoint_enabled(self, checkpoint_type: CheckpointType) -> bool:
        """Check if a specific checkpoint type is enabled."""
        if not self.enabled:
            return False
        type_map = {
            CheckpointType.POST_PLAN: self.post_plan,
            CheckpointType.PRE_COMMIT: self.pre_commit,
            CheckpointType.PRE_PR: self.pre_pr,
        }
        return type_map.get(checkpoint_type, False)


class CheckpointResult:
    """Result of waiting for a checkpoint."""

    def __init__(
        self,
        approved: bool,
        status: CheckpointStatus,
        resolved_by: Optional[str] = None,
        rejection_reason: Optional[str] = None,
    ):
        self.approved = approved
        self.status = status
        self.resolved_by = resolved_by
        self.rejection_reason = rejection_reason

    @property
    def should_proceed(self) -> bool:
        """Whether execution should continue."""
        return self.approved or self.status == CheckpointStatus.SKIPPED


class CheckpointManager(SQLiteStore):
    """Manages checkpoint creation, waiting, and resolution.

    Checkpoints are approval gates that pause task execution until
    a user approves or rejects via Slack buttons. Supports configurable
    timeout behavior and persistence for daemon restart resilience.
    """

    def __init__(
        self,
        db_path: str,
        config: CheckpointConfig,
        slack_client: Optional[Any] = None,
    ):
        """Initialize checkpoint manager.

        Args:
            db_path: Path to SQLite database
            config: Checkpoint configuration
            slack_client: Slack WebClient for sending messages
        """
        super().__init__(db_path)
        self.config = config
        self.slack_client = slack_client

    def create_checkpoint(
        self,
        task: Task,
        checkpoint_type: CheckpointType,
        title: str,
        details: Optional[dict] = None,
        channel_id: Optional[str] = None,
        thread_ts: Optional[str] = None,
    ) -> Optional[Checkpoint]:
        """Create a new checkpoint for a task.

        Args:
            task: The task requiring approval
            checkpoint_type: Type of checkpoint (POST_PLAN, PRE_COMMIT, PRE_PR)
            title: Brief description for the approval prompt
            details: Additional context (files to commit, PR title, etc.)
            channel_id: Slack channel for the approval message
            thread_ts: Slack thread for context

        Returns:
            Created Checkpoint or None if checkpoints disabled
        """
        if not self.config.is_checkpoint_enabled(checkpoint_type):
            logger.debug(
                "checkpoint.skipped",
                task_id=task.id,
                checkpoint_type=checkpoint_type.value,
                reason="disabled_in_config",
            )
            return None

        def _op(session: Session) -> Checkpoint:
            expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
                minutes=self.config.timeout_minutes
            )
            checkpoint = Checkpoint(
                task_id=task.id,
                checkpoint_type=checkpoint_type,
                status=CheckpointStatus.PENDING,
                title=title,
                details=json.dumps(details) if details else None,
                channel_id=channel_id,
                thread_ts=thread_ts,
                expires_at=expires_at,
            )
            session.add(checkpoint)
            session.flush()
            return checkpoint

        checkpoint = self._run_write(_op)
        logger.info(
            "checkpoint.created",
            checkpoint_id=checkpoint.id,
            task_id=task.id,
            checkpoint_type=checkpoint_type.value,
            expires_at=checkpoint.expires_at.isoformat(),
        )
        return checkpoint

    def get_checkpoint(self, checkpoint_id: int) -> Optional[Checkpoint]:
        """Get checkpoint by ID."""

        def _op(session: Session) -> Optional[Checkpoint]:
            return session.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()

        return self._run_read(_op)

    def get_pending_checkpoint(self, task_id: int) -> Optional[Checkpoint]:
        """Get the pending checkpoint for a task, if any."""

        def _op(session: Session) -> Optional[Checkpoint]:
            return (
                session.query(Checkpoint)
                .filter(
                    Checkpoint.task_id == task_id,
                    Checkpoint.status == CheckpointStatus.PENDING,
                )
                .first()
            )

        return self._run_read(_op)

    def update_message_ts(self, checkpoint_id: int, message_ts: str) -> None:
        """Update the Slack message timestamp for a checkpoint."""

        def _op(session: Session) -> None:
            checkpoint = session.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
            if checkpoint:
                checkpoint.message_ts = message_ts

        self._run_write(_op)

    def resolve(
        self,
        checkpoint_id: int,
        status: CheckpointStatus,
        resolved_by: Optional[str] = None,
        rejection_reason: Optional[str] = None,
    ) -> Optional[Checkpoint]:
        """Resolve a checkpoint (approve, reject, or expire).

        Args:
            checkpoint_id: ID of the checkpoint to resolve
            status: Resolution status (APPROVED, REJECTED, EXPIRED)
            resolved_by: Slack user ID who resolved
            rejection_reason: Optional reason for rejection

        Returns:
            Updated Checkpoint or None if not found
        """

        def _op(session: Session) -> Optional[Checkpoint]:
            checkpoint = session.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
            if checkpoint and checkpoint.status == CheckpointStatus.PENDING:
                checkpoint.status = status
                checkpoint.resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
                checkpoint.resolved_by = resolved_by
                if rejection_reason:
                    checkpoint.rejection_reason = rejection_reason
            return checkpoint

        checkpoint = self._run_write(_op)
        if checkpoint:
            logger.info(
                "checkpoint.resolved",
                checkpoint_id=checkpoint_id,
                status=status.value,
                resolved_by=resolved_by,
            )
        return checkpoint

    def expire_stale_checkpoints(self) -> list[Checkpoint]:
        """Expire all checkpoints past their timeout."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        def _op(session: Session) -> list[Checkpoint]:
            stale = (
                session.query(Checkpoint)
                .filter(
                    Checkpoint.status == CheckpointStatus.PENDING,
                    Checkpoint.expires_at < now,
                )
                .all()
            )
            for checkpoint in stale:
                checkpoint.status = CheckpointStatus.EXPIRED
                checkpoint.resolved_at = now
            return stale

        expired = self._run_write(_op)
        if expired:
            logger.warning(
                "checkpoints.expired",
                count=len(expired),
                checkpoint_ids=[c.id for c in expired],
            )
        return expired

    def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """Delete resolved checkpoints older than specified days."""
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)

        def _op(session: Session) -> int:
            count = (
                session.query(Checkpoint)
                .filter(
                    Checkpoint.status != CheckpointStatus.PENDING,
                    Checkpoint.created_at < cutoff,
                )
                .delete()
            )
            return count

        count = self._run_write(_op)
        if count > 0:
            logger.info("checkpoints.cleanup", deleted_count=count, older_than_days=days)
        return count

    async def send_checkpoint_message(
        self,
        checkpoint: Checkpoint,
        task: Task,
    ) -> Optional[str]:
        """Send Slack message with Approve/Reject buttons.

        Args:
            checkpoint: The checkpoint to send message for
            task: The associated task

        Returns:
            Message timestamp if sent, None otherwise
        """
        if not self.slack_client:
            logger.warning("checkpoint.no_slack_client", checkpoint_id=checkpoint.id)
            return None

        if not checkpoint.channel_id:
            logger.warning("checkpoint.no_channel", checkpoint_id=checkpoint.id)
            return None

        # Parse details for context
        details = json.loads(checkpoint.details) if checkpoint.details else {}

        # Build Block Kit message
        blocks = self._build_checkpoint_blocks(checkpoint, task, details)

        try:
            response = self.slack_client.chat_postMessage(
                channel=checkpoint.channel_id,
                thread_ts=checkpoint.thread_ts,
                text=f"Approval required: {checkpoint.title}",
                blocks=blocks,
            )
            message_ts = response.get("ts")
            if message_ts:
                self.update_message_ts(checkpoint.id, message_ts)
            return message_ts
        except Exception as exc:
            logger.error(
                "checkpoint.send_failed",
                checkpoint_id=checkpoint.id,
                error=str(exc),
            )
            return None

    def _build_checkpoint_blocks(
        self,
        checkpoint: Checkpoint,
        task: Task,
        details: dict,
    ) -> list[dict]:
        """Build Slack Block Kit blocks for checkpoint message."""
        type_labels = {
            CheckpointType.POST_PLAN: "Plan Review",
            CheckpointType.PRE_COMMIT: "Pre-Commit Approval",
            CheckpointType.PRE_PR: "Pull Request Approval",
        }
        type_label = type_labels.get(checkpoint.checkpoint_type, "Approval Required")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f":clipboard: {type_label}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Task #{task.id}*: {task.description[:100]}{'...' if len(task.description) > 100 else ''}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{checkpoint.title}*",
                },
            },
        ]

        # Add details based on checkpoint type
        if checkpoint.checkpoint_type == CheckpointType.PRE_COMMIT:
            files = details.get("files", [])
            if files:
                file_list = "\n".join(f"• `{f}`" for f in files[:10])
                if len(files) > 10:
                    file_list += f"\n• ... and {len(files) - 10} more"
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Files to commit:*\n{file_list}",
                    },
                })

        elif checkpoint.checkpoint_type == CheckpointType.PRE_PR:
            pr_title = details.get("pr_title", "")
            base_branch = details.get("base_branch", "main")
            if pr_title:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*PR Title:* {pr_title}\n*Base Branch:* `{base_branch}`",
                    },
                })

        # Add timeout info
        timeout_mins = self.config.timeout_minutes
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":hourglass: Expires in {timeout_mins} minutes | Timeout action: _{self.config.timeout_behavior}_",
                }
            ],
        })

        # Add action buttons
        blocks.append({
            "type": "actions",
            "block_id": f"checkpoint_{checkpoint.id}",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Approve",
                        "emoji": True,
                    },
                    "style": "primary",
                    "action_id": "checkpoint_approve",
                    "value": str(checkpoint.id),
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Reject",
                        "emoji": True,
                    },
                    "style": "danger",
                    "action_id": "checkpoint_reject",
                    "value": str(checkpoint.id),
                },
            ],
        })

        return blocks

    async def wait_for_resolution(
        self,
        checkpoint: Checkpoint,
        task: Task,
    ) -> CheckpointResult:
        """Wait for a checkpoint to be resolved.

        Polls the database for resolution status and handles timeout
        according to configuration.

        Args:
            checkpoint: The checkpoint to wait for
            task: The associated task

        Returns:
            CheckpointResult indicating whether to proceed
        """
        # Send the approval message
        await self.send_checkpoint_message(checkpoint, task)

        # Poll for resolution
        while True:
            await asyncio.sleep(self.config.poll_interval_seconds)

            # Refresh checkpoint from database
            current = self.get_checkpoint(checkpoint.id)
            if not current:
                logger.error("checkpoint.not_found", checkpoint_id=checkpoint.id)
                return CheckpointResult(
                    approved=False,
                    status=CheckpointStatus.EXPIRED,
                )

            # Check if resolved
            if current.status != CheckpointStatus.PENDING:
                return CheckpointResult(
                    approved=current.status == CheckpointStatus.APPROVED,
                    status=current.status,
                    resolved_by=current.resolved_by,
                    rejection_reason=current.rejection_reason,
                )

            # Check for expiration
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            if current.expires_at and now >= current.expires_at:
                self.resolve(checkpoint.id, CheckpointStatus.EXPIRED)
                return self._handle_timeout(checkpoint, task)

    def _handle_timeout(
        self,
        checkpoint: Checkpoint,
        task: Task,
    ) -> CheckpointResult:
        """Handle checkpoint timeout according to configuration."""
        behavior = self.config.timeout_behavior

        if behavior == "proceed":
            logger.info(
                "checkpoint.timeout_proceed",
                checkpoint_id=checkpoint.id,
                task_id=task.id,
            )
            return CheckpointResult(
                approved=True,
                status=CheckpointStatus.EXPIRED,
            )

        elif behavior == "abort":
            logger.info(
                "checkpoint.timeout_abort",
                checkpoint_id=checkpoint.id,
                task_id=task.id,
            )
            return CheckpointResult(
                approved=False,
                status=CheckpointStatus.EXPIRED,
                rejection_reason="Timed out waiting for approval",
            )

        else:  # notify (default)
            logger.info(
                "checkpoint.timeout_notify",
                checkpoint_id=checkpoint.id,
                task_id=task.id,
            )
            # Notify user but don't proceed
            return CheckpointResult(
                approved=False,
                status=CheckpointStatus.EXPIRED,
                rejection_reason="Timed out - requires manual review",
            )

    async def request_approval(
        self,
        task: Task,
        checkpoint_type: CheckpointType,
        title: str,
        details: Optional[dict] = None,
        channel_id: Optional[str] = None,
        thread_ts: Optional[str] = None,
    ) -> CheckpointResult:
        """High-level method to request and wait for approval.

        Creates a checkpoint, sends the Slack message, and waits for
        resolution or timeout.

        Args:
            task: The task requiring approval
            checkpoint_type: Type of checkpoint
            title: Brief description
            details: Additional context
            channel_id: Slack channel
            thread_ts: Slack thread

        Returns:
            CheckpointResult indicating whether to proceed
        """
        # Check if checkpoint type is enabled
        if not self.config.is_checkpoint_enabled(checkpoint_type):
            return CheckpointResult(
                approved=True,
                status=CheckpointStatus.SKIPPED,
            )

        # Create the checkpoint
        checkpoint = self.create_checkpoint(
            task=task,
            checkpoint_type=checkpoint_type,
            title=title,
            details=details,
            channel_id=channel_id,
            thread_ts=thread_ts,
        )

        if not checkpoint:
            return CheckpointResult(
                approved=True,
                status=CheckpointStatus.SKIPPED,
            )

        # Wait for resolution
        return await self.wait_for_resolution(checkpoint, task)
