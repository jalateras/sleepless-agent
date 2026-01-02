"""Proactive progress notifications for task execution visibility."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)


class NotificationType(str, Enum):
    """Types of notifications sent during task execution."""

    PHASE_TRANSITION = "phase_transition"  # Task moved to new execution phase
    MILESTONE = "milestone"  # Significant progress point reached
    BLOCKER = "blocker"  # Something is blocking progress
    HEARTBEAT = "heartbeat"  # Periodic "still working" signal
    COMPLETION = "completion"  # Task finished (success or failure)


class BlockerType(str, Enum):
    """Classification of blockers for appropriate response."""

    RATE_LIMIT = "rate_limit"  # API rate limiting - will retry automatically
    API_ERROR = "api_error"  # Transient API error - will retry
    MISSING_DEPENDENCY = "missing_dependency"  # Needs manual intervention
    PERMISSION_ERROR = "permission_error"  # Needs manual intervention
    TIMEOUT = "timeout"  # Operation timed out
    UNKNOWN = "unknown"  # Unclassified blocker


class ExecutionPhase(str, Enum):
    """Phases of task execution for progress tracking."""

    QUEUED = "queued"
    PLANNING = "planning"
    EXECUTING = "executing"
    TESTING = "testing"
    COMMITTING = "committing"
    CREATING_PR = "creating_pr"
    COMPLETED = "completed"
    FAILED = "failed"


class NotificationConfig:
    """Configuration for notification behavior."""

    def __init__(
        self,
        enabled: bool = True,
        phase_notifications: bool = True,
        milestone_notifications: bool = True,
        blocker_notifications: bool = True,
        heartbeat_enabled: bool = True,
        heartbeat_interval_minutes: int = 10,
        batch_window_seconds: float = 5.0,
        min_phase_duration_seconds: float = 30.0,
    ):
        """Initialize notification configuration.

        Args:
            enabled: Master switch for all notifications
            phase_notifications: Send notifications on phase transitions
            milestone_notifications: Send notifications for milestones
            blocker_notifications: Send alerts for blockers
            heartbeat_enabled: Send periodic heartbeats for long tasks
            heartbeat_interval_minutes: Minutes between heartbeat messages
            batch_window_seconds: Time window for batching rapid notifications
            min_phase_duration_seconds: Minimum time in phase before notifying
        """
        self.enabled = enabled
        self.phase_notifications = phase_notifications
        self.milestone_notifications = milestone_notifications
        self.blocker_notifications = blocker_notifications
        self.heartbeat_enabled = heartbeat_enabled
        self.heartbeat_interval_minutes = heartbeat_interval_minutes
        self.batch_window_seconds = batch_window_seconds
        self.min_phase_duration_seconds = min_phase_duration_seconds

    @classmethod
    def from_dict(cls, config: dict) -> "NotificationConfig":
        """Create config from dictionary (e.g., from config.yaml)."""
        return cls(
            enabled=config.get("enabled", True),
            phase_notifications=config.get("phase_notifications", True),
            milestone_notifications=config.get("milestone_notifications", True),
            blocker_notifications=config.get("blocker_notifications", True),
            heartbeat_enabled=config.get("heartbeat_enabled", True),
            heartbeat_interval_minutes=config.get("heartbeat_interval_minutes", 10),
            batch_window_seconds=config.get("batch_window_seconds", 5.0),
            min_phase_duration_seconds=config.get("min_phase_duration_seconds", 30.0),
        )


class PendingNotification:
    """A notification queued for batching."""

    def __init__(
        self,
        notification_type: NotificationType,
        message: str,
        details: Optional[dict] = None,
        priority: int = 0,
    ):
        self.notification_type = notification_type
        self.message = message
        self.details = details or {}
        self.priority = priority
        self.created_at = datetime.now(timezone.utc).replace(tzinfo=None)


class NotificationManager:
    """Manages proactive notifications for task execution visibility.

    Provides phase transition updates, milestone notifications, blocker alerts,
    and heartbeat signals for long-running tasks. Supports notification batching
    to avoid spamming users with rapid updates.
    """

    def __init__(
        self,
        config: NotificationConfig,
        slack_client: Optional[Any] = None,
        default_channel: Optional[str] = None,
    ):
        """Initialize notification manager.

        Args:
            config: Notification configuration
            slack_client: Slack WebClient for sending messages
            default_channel: Fallback Slack channel for tasks without a channel
        """
        self.config = config
        self.slack_client = slack_client
        self.default_channel = default_channel

        # State tracking per task
        self._task_state: dict[int, dict] = {}
        self._pending_notifications: dict[int, list[PendingNotification]] = {}
        self._batch_tasks: dict[int, asyncio.Task] = {}
        self._heartbeat_tasks: dict[int, asyncio.Task] = {}

    def start_task_tracking(
        self,
        task_id: int,
        channel_id: Optional[str] = None,
        thread_ts: Optional[str] = None,
    ) -> None:
        """Begin tracking a task for notifications.

        Args:
            task_id: The task to track
            channel_id: Slack channel for notifications
            thread_ts: Slack thread for context
        """
        self._task_state[task_id] = {
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "current_phase": ExecutionPhase.QUEUED,
            "phase_started_at": datetime.now(timezone.utc).replace(tzinfo=None),
            "milestones": [],
            "blockers": [],
            "started_at": datetime.now(timezone.utc).replace(tzinfo=None),
        }
        self._pending_notifications[task_id] = []

        logger.debug(
            "notification.tracking_started",
            task_id=task_id,
            channel_id=channel_id,
        )

    def stop_task_tracking(self, task_id: int) -> None:
        """Stop tracking a task and cleanup resources.

        Args:
            task_id: The task to stop tracking
        """
        # Cancel any pending batch or heartbeat tasks
        if task_id in self._batch_tasks:
            self._batch_tasks[task_id].cancel()
            del self._batch_tasks[task_id]

        if task_id in self._heartbeat_tasks:
            self._heartbeat_tasks[task_id].cancel()
            del self._heartbeat_tasks[task_id]

        # Cleanup state
        self._task_state.pop(task_id, None)
        self._pending_notifications.pop(task_id, None)

        logger.debug("notification.tracking_stopped", task_id=task_id)

    async def notify_phase_transition(
        self,
        task_id: int,
        new_phase: ExecutionPhase,
        details: Optional[dict] = None,
    ) -> None:
        """Notify about a phase transition in task execution.

        Args:
            task_id: The task transitioning phases
            new_phase: The new execution phase
            details: Additional context about the transition
        """
        if not self.config.enabled or not self.config.phase_notifications:
            return

        state = self._task_state.get(task_id)
        if not state:
            logger.warning(
                "notification.untracked_task",
                task_id=task_id,
                phase=new_phase.value,
            )
            return

        old_phase = state["current_phase"]
        phase_duration = (
            datetime.now(timezone.utc).replace(tzinfo=None) - state["phase_started_at"]
        ).total_seconds()

        # Skip notification if phase was too brief (likely intermediate state)
        if phase_duration < self.config.min_phase_duration_seconds:
            logger.debug(
                "notification.phase_too_brief",
                task_id=task_id,
                old_phase=old_phase.value,
                new_phase=new_phase.value,
                duration_seconds=phase_duration,
            )

        # Update state
        state["current_phase"] = new_phase
        state["phase_started_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

        # Queue the notification
        phase_labels = {
            ExecutionPhase.QUEUED: ":inbox_tray: Queued",
            ExecutionPhase.PLANNING: ":thought_balloon: Planning",
            ExecutionPhase.EXECUTING: ":hammer_and_wrench: Executing",
            ExecutionPhase.TESTING: ":test_tube: Testing",
            ExecutionPhase.COMMITTING: ":floppy_disk: Committing",
            ExecutionPhase.CREATING_PR: ":git: Creating PR",
            ExecutionPhase.COMPLETED: ":white_check_mark: Completed",
            ExecutionPhase.FAILED: ":x: Failed",
        }
        label = phase_labels.get(new_phase, f":arrow_right: {new_phase.value}")

        notification = PendingNotification(
            notification_type=NotificationType.PHASE_TRANSITION,
            message=f"{label}",
            details={"old_phase": old_phase.value, "new_phase": new_phase.value, **(details or {})},
            priority=1,
        )
        await self._queue_notification(task_id, notification)

    async def notify_milestone(
        self,
        task_id: int,
        milestone: str,
        details: Optional[dict] = None,
    ) -> None:
        """Notify about a milestone reached in task execution.

        Args:
            task_id: The task that reached the milestone
            milestone: Description of the milestone
            details: Additional context
        """
        if not self.config.enabled or not self.config.milestone_notifications:
            return

        state = self._task_state.get(task_id)
        if not state:
            return

        state["milestones"].append({
            "milestone": milestone,
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None),
        })

        notification = PendingNotification(
            notification_type=NotificationType.MILESTONE,
            message=f":dart: {milestone}",
            details=details or {},
            priority=0,
        )
        await self._queue_notification(task_id, notification)

    async def notify_blocker(
        self,
        task_id: int,
        blocker_type: BlockerType,
        message: str,
        will_retry: bool = False,
        retry_after_seconds: Optional[int] = None,
        details: Optional[dict] = None,
    ) -> None:
        """Notify about a blocker encountered during execution.

        Args:
            task_id: The blocked task
            blocker_type: Classification of the blocker
            message: Human-readable description
            will_retry: Whether the system will automatically retry
            retry_after_seconds: Seconds until retry (if applicable)
            details: Additional context
        """
        if not self.config.enabled or not self.config.blocker_notifications:
            return

        state = self._task_state.get(task_id)
        if not state:
            return

        state["blockers"].append({
            "type": blocker_type.value,
            "message": message,
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None),
        })

        # Build notification with appropriate urgency
        blocker_icons = {
            BlockerType.RATE_LIMIT: ":hourglass_flowing_sand:",
            BlockerType.API_ERROR: ":warning:",
            BlockerType.MISSING_DEPENDENCY: ":package:",
            BlockerType.PERMISSION_ERROR: ":lock:",
            BlockerType.TIMEOUT: ":stopwatch:",
            BlockerType.UNKNOWN: ":question:",
        }
        icon = blocker_icons.get(blocker_type, ":warning:")

        # Build message with retry info
        full_message = f"{icon} *Blocker:* {message}"
        if will_retry:
            retry_info = f" (will retry"
            if retry_after_seconds:
                retry_info += f" in {retry_after_seconds}s"
            retry_info += ")"
            full_message += retry_info

        # Blockers bypass batching - send immediately
        notification = PendingNotification(
            notification_type=NotificationType.BLOCKER,
            message=full_message,
            details={
                "blocker_type": blocker_type.value,
                "will_retry": will_retry,
                "retry_after_seconds": retry_after_seconds,
                **(details or {}),
            },
            priority=10,  # High priority
        )

        # Send immediately for blockers
        await self._send_notification(task_id, notification)

    async def start_heartbeat(self, task_id: int) -> None:
        """Start sending periodic heartbeat notifications for a task.

        Args:
            task_id: The task to send heartbeats for
        """
        if not self.config.enabled or not self.config.heartbeat_enabled:
            return

        if task_id in self._heartbeat_tasks:
            return  # Already running

        async def heartbeat_loop():
            interval = self.config.heartbeat_interval_minutes * 60
            while True:
                await asyncio.sleep(interval)
                state = self._task_state.get(task_id)
                if not state:
                    break

                elapsed = (
                    datetime.now(timezone.utc).replace(tzinfo=None) - state["started_at"]
                ).total_seconds()
                elapsed_minutes = int(elapsed / 60)

                phase = state["current_phase"]
                notification = PendingNotification(
                    notification_type=NotificationType.HEARTBEAT,
                    message=f":heartbeat: Still working... ({elapsed_minutes}m, phase: {phase.value})",
                    details={"elapsed_minutes": elapsed_minutes, "phase": phase.value},
                    priority=-1,  # Low priority
                )
                await self._send_notification(task_id, notification)

        self._heartbeat_tasks[task_id] = asyncio.create_task(heartbeat_loop())
        logger.debug("notification.heartbeat_started", task_id=task_id)

    def stop_heartbeat(self, task_id: int) -> None:
        """Stop heartbeat notifications for a task.

        Args:
            task_id: The task to stop heartbeats for
        """
        if task_id in self._heartbeat_tasks:
            self._heartbeat_tasks[task_id].cancel()
            del self._heartbeat_tasks[task_id]
            logger.debug("notification.heartbeat_stopped", task_id=task_id)

    async def notify_completion(
        self,
        task_id: int,
        success: bool,
        summary: str,
        details: Optional[dict] = None,
    ) -> None:
        """Notify about task completion.

        Args:
            task_id: The completed task
            success: Whether the task succeeded
            summary: Brief summary of the outcome
            details: Additional context (PR URL, files modified, etc.)
        """
        if not self.config.enabled:
            return

        state = self._task_state.get(task_id)
        if not state:
            return

        # Stop heartbeat
        self.stop_heartbeat(task_id)

        # Calculate total duration
        elapsed = (
            datetime.now(timezone.utc).replace(tzinfo=None) - state["started_at"]
        ).total_seconds()
        elapsed_minutes = int(elapsed / 60)

        icon = ":white_check_mark:" if success else ":x:"
        status = "completed successfully" if success else "failed"

        message = f"{icon} Task {status} ({elapsed_minutes}m)\n{summary}"

        notification = PendingNotification(
            notification_type=NotificationType.COMPLETION,
            message=message,
            details={
                "success": success,
                "elapsed_minutes": elapsed_minutes,
                "milestones_count": len(state["milestones"]),
                "blockers_count": len(state["blockers"]),
                **(details or {}),
            },
            priority=5,
        )

        # Completion bypasses batching
        await self._send_notification(task_id, notification)

    async def _queue_notification(
        self,
        task_id: int,
        notification: PendingNotification,
    ) -> None:
        """Queue a notification for batching.

        Args:
            task_id: The task the notification is for
            notification: The notification to queue
        """
        if task_id not in self._pending_notifications:
            self._pending_notifications[task_id] = []

        self._pending_notifications[task_id].append(notification)

        # Start batch timer if not already running
        if task_id not in self._batch_tasks:
            self._batch_tasks[task_id] = asyncio.create_task(
                self._batch_and_send(task_id)
            )

    async def _batch_and_send(self, task_id: int) -> None:
        """Wait for batch window then send coalesced notifications.

        Args:
            task_id: The task to send batched notifications for
        """
        await asyncio.sleep(self.config.batch_window_seconds)

        pending = self._pending_notifications.get(task_id, [])
        if not pending:
            self._batch_tasks.pop(task_id, None)
            return

        # Sort by priority (highest first) and coalesce
        pending.sort(key=lambda n: n.priority, reverse=True)

        # For now, send all notifications as a single combined message
        # Could be smarter about coalescing similar types
        combined_message = "\n".join(n.message for n in pending)

        # Use highest priority notification as the base
        combined = PendingNotification(
            notification_type=pending[0].notification_type,
            message=combined_message,
            details={"batched_count": len(pending)},
            priority=pending[0].priority,
        )

        await self._send_notification(task_id, combined)

        # Clear pending
        self._pending_notifications[task_id] = []
        self._batch_tasks.pop(task_id, None)

    async def _send_notification(
        self,
        task_id: int,
        notification: PendingNotification,
    ) -> None:
        """Send a notification to Slack.

        Args:
            task_id: The task the notification is for
            notification: The notification to send
        """
        state = self._task_state.get(task_id)
        if not state:
            return

        channel_id = state.get("channel_id") or self.default_channel
        thread_ts = state.get("thread_ts")

        if not channel_id:
            logger.debug(
                "notification.no_channel",
                task_id=task_id,
                notification_type=notification.notification_type.value,
            )
            return

        if not self.slack_client:
            logger.debug(
                "notification.no_slack_client",
                task_id=task_id,
                notification_type=notification.notification_type.value,
            )
            return

        try:
            self.slack_client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=notification.message,
                unfurl_links=False,
                unfurl_media=False,
            )
            logger.debug(
                "notification.sent",
                task_id=task_id,
                notification_type=notification.notification_type.value,
            )
        except Exception as exc:
            logger.error(
                "notification.send_failed",
                task_id=task_id,
                notification_type=notification.notification_type.value,
                error=str(exc),
            )

    def get_task_summary(self, task_id: int) -> Optional[dict]:
        """Get a summary of notifications for a task.

        Args:
            task_id: The task to summarize

        Returns:
            Summary dict or None if task not tracked
        """
        state = self._task_state.get(task_id)
        if not state:
            return None

        elapsed = (
            datetime.now(timezone.utc).replace(tzinfo=None) - state["started_at"]
        ).total_seconds()

        return {
            "task_id": task_id,
            "current_phase": state["current_phase"].value,
            "elapsed_seconds": elapsed,
            "milestones_count": len(state["milestones"]),
            "blockers_count": len(state["blockers"]),
            "milestones": state["milestones"],
            "blockers": state["blockers"],
        }
