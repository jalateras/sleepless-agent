"""Real-time task streaming to Slack.

Streams Claude's output to Slack in real-time, updating a single message
as content arrives. Includes sliding window buffer for Slack's character
limit and rate limiting to avoid API throttling.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)


class StreamVerbosity(str, Enum):
    """Verbosity levels for streaming output."""

    OFF = "off"  # No streaming, only final result
    MINIMAL = "minimal"  # Phase changes only
    NORMAL = "normal"  # Phase changes + key actions (file edits, commands)
    VERBOSE = "verbose"  # Full output stream


@dataclass
class StreamConfig:
    """Configuration for streaming behavior."""

    enabled: bool = True
    verbosity: StreamVerbosity = StreamVerbosity.NORMAL
    update_interval_seconds: float = 2.0  # Minimum time between Slack updates
    max_message_length: int = 35000  # Slack limit is 40k, leave buffer
    truncation_indicator: str = "\n... (earlier output truncated) ...\n"
    phase_prefix_enabled: bool = True  # Show phase label in stream

    @classmethod
    def from_dict(cls, config: dict) -> "StreamConfig":
        """Create config from dictionary (e.g., from config.yaml)."""
        verbosity_str = config.get("verbosity", "normal")
        try:
            verbosity = StreamVerbosity(verbosity_str)
        except ValueError:
            verbosity = StreamVerbosity.NORMAL

        return cls(
            enabled=config.get("enabled", True),
            verbosity=verbosity,
            update_interval_seconds=config.get("update_interval_seconds", 2.0),
            max_message_length=config.get("max_message_length", 35000),
        )


@dataclass
class StreamState:
    """State for an active stream."""

    task_id: int
    channel_id: str
    thread_ts: Optional[str] = None
    message_ts: Optional[str] = None  # The message we're updating
    current_phase: str = "initializing"
    buffer: str = ""
    last_update_time: Optional[datetime] = None
    is_paused: bool = False
    total_chars_streamed: int = 0
    update_count: int = 0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )


class StreamManager:
    """Manages real-time streaming of task output to Slack.

    Features:
    - Sliding window buffer respecting Slack's character limit
    - Rate-limited updates to avoid API throttling
    - Verbosity levels for controlling output detail
    - Pause/resume functionality
    - Per-task stream state management
    """

    def __init__(
        self,
        config: StreamConfig,
        slack_client: Optional[Any] = None,
    ):
        """Initialize stream manager.

        Args:
            config: Streaming configuration
            slack_client: Slack WebClient for updating messages
        """
        self.config = config
        self.slack_client = slack_client

        # Active streams by task_id
        self._streams: dict[int, StreamState] = {}
        # Pending update tasks (for debouncing)
        self._update_tasks: dict[int, asyncio.Task] = {}
        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def start_stream(
        self,
        task_id: int,
        channel_id: str,
        thread_ts: Optional[str] = None,
        initial_message: str = ":arrows_counterclockwise: Starting task...",
    ) -> Optional[str]:
        """Start a new stream for a task.

        Creates an initial Slack message that will be updated as output arrives.

        Args:
            task_id: The task to stream
            channel_id: Slack channel for the stream
            thread_ts: Optional thread to stream in
            initial_message: Initial message content

        Returns:
            Message timestamp of the stream message, or None if failed
        """
        if not self.config.enabled or self.config.verbosity == StreamVerbosity.OFF:
            logger.debug("streaming.disabled", task_id=task_id)
            return None

        if not self.slack_client:
            logger.debug("streaming.no_slack_client", task_id=task_id)
            return None

        async with self._lock:
            # Stop any existing stream for this task
            if task_id in self._streams:
                await self._cleanup_stream(task_id)

            try:
                # Post initial message
                response = self.slack_client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=initial_message,
                    unfurl_links=False,
                    unfurl_media=False,
                )

                message_ts = response.get("ts")
                if not message_ts:
                    logger.error("streaming.no_message_ts", task_id=task_id)
                    return None

                # Create stream state
                self._streams[task_id] = StreamState(
                    task_id=task_id,
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    message_ts=message_ts,
                    buffer=initial_message,
                    last_update_time=datetime.now(timezone.utc).replace(tzinfo=None),
                )

                logger.info(
                    "streaming.started",
                    task_id=task_id,
                    channel_id=channel_id,
                    message_ts=message_ts,
                )
                return message_ts

            except Exception as exc:
                logger.error(
                    "streaming.start_failed",
                    task_id=task_id,
                    error=str(exc),
                )
                return None

    async def append_output(
        self,
        task_id: int,
        content: str,
        phase: Optional[str] = None,
    ) -> None:
        """Append output content to a stream.

        Content is buffered and updates are rate-limited to avoid
        hitting Slack API limits.

        Args:
            task_id: The task's stream
            content: Content to append
            phase: Optional phase label to update
        """
        if not self.config.enabled or self.config.verbosity == StreamVerbosity.OFF:
            return

        stream = self._streams.get(task_id)
        if not stream:
            logger.debug("streaming.no_stream", task_id=task_id)
            return

        if stream.is_paused:
            logger.debug("streaming.paused", task_id=task_id)
            return

        # Apply verbosity filtering
        if not self._should_stream_content(content, phase):
            return

        async with self._lock:
            # Update phase if provided
            if phase:
                stream.current_phase = phase

            # Append to buffer with truncation if needed
            new_content = self._format_content(content, phase, stream)
            stream.buffer = self._apply_sliding_window(
                stream.buffer + new_content,
                self.config.max_message_length,
            )
            stream.total_chars_streamed += len(content)

        # Schedule debounced update
        await self._schedule_update(task_id)

    async def update_phase(self, task_id: int, phase: str) -> None:
        """Update the current phase label for a stream.

        Args:
            task_id: The task's stream
            phase: New phase name
        """
        stream = self._streams.get(task_id)
        if not stream:
            return

        async with self._lock:
            stream.current_phase = phase

        # Add phase transition to stream if verbosity allows
        if self.config.verbosity in (StreamVerbosity.MINIMAL, StreamVerbosity.NORMAL, StreamVerbosity.VERBOSE):
            phase_labels = {
                "planner": ":thought_balloon: Planning",
                "worker": ":hammer_and_wrench: Executing",
                "evaluator": ":mag: Evaluating",
                "committing": ":floppy_disk: Committing",
                "completed": ":white_check_mark: Completed",
                "failed": ":x: Failed",
            }
            label = phase_labels.get(phase, f":arrow_right: {phase}")
            await self.append_output(task_id, f"\n\n{label}\n", phase=phase)

    def _should_stream_content(self, content: str, phase: Optional[str]) -> bool:
        """Check if content should be streamed based on verbosity.

        Args:
            content: Content to check
            phase: Current phase

        Returns:
            True if content should be streamed
        """
        verbosity = self.config.verbosity

        if verbosity == StreamVerbosity.OFF:
            return False

        if verbosity == StreamVerbosity.VERBOSE:
            return True

        if verbosity == StreamVerbosity.MINIMAL:
            # Only phase changes (handled separately in update_phase)
            return False

        # NORMAL verbosity: phase changes + key actions
        # Look for indicators of important actions
        important_patterns = [
            "Created file",
            "Modified file",
            "Deleted file",
            "Running command",
            "Executed",
            "Error:",
            "Warning:",
            "TODO:",
            "DONE:",
            "##",  # Markdown headers
        ]
        return any(pattern in content for pattern in important_patterns)

    def _format_content(
        self,
        content: str,
        phase: Optional[str],
        stream: StreamState,
    ) -> str:
        """Format content for display in stream.

        Args:
            content: Raw content
            phase: Current phase
            stream: Stream state

        Returns:
            Formatted content
        """
        # Simple formatting - just the content
        # Phase headers are added separately in update_phase
        return content

    def _apply_sliding_window(self, text: str, max_length: int) -> str:
        """Apply sliding window to keep text under max length.

        Keeps the most recent content, truncating from the beginning.

        Args:
            text: Full text
            max_length: Maximum allowed length

        Returns:
            Truncated text with indicator if truncated
        """
        if len(text) <= max_length:
            return text

        # Calculate how much to keep
        indicator = self.config.truncation_indicator
        keep_length = max_length - len(indicator)

        # Find a good break point (newline) near the truncation point
        truncated = text[-keep_length:]
        newline_pos = truncated.find("\n")
        if newline_pos > 0 and newline_pos < 200:
            truncated = truncated[newline_pos + 1:]

        return indicator + truncated

    async def _schedule_update(self, task_id: int) -> None:
        """Schedule a debounced update for a stream.

        Args:
            task_id: The task's stream
        """
        # Cancel any pending update
        if task_id in self._update_tasks:
            self._update_tasks[task_id].cancel()

        # Schedule new update after debounce interval
        self._update_tasks[task_id] = asyncio.create_task(
            self._debounced_update(task_id)
        )

    async def _debounced_update(self, task_id: int) -> None:
        """Execute a debounced update to Slack.

        Args:
            task_id: The task's stream
        """
        stream = self._streams.get(task_id)
        if not stream or not self.slack_client:
            return

        # Check if enough time has passed since last update
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if stream.last_update_time:
            elapsed = (now - stream.last_update_time).total_seconds()
            if elapsed < self.config.update_interval_seconds:
                await asyncio.sleep(self.config.update_interval_seconds - elapsed)

        # Update the Slack message
        try:
            self.slack_client.chat_update(
                channel=stream.channel_id,
                ts=stream.message_ts,
                text=stream.buffer,
            )
            stream.last_update_time = datetime.now(timezone.utc).replace(tzinfo=None)
            stream.update_count += 1

            logger.debug(
                "streaming.updated",
                task_id=task_id,
                update_count=stream.update_count,
                buffer_length=len(stream.buffer),
            )

        except Exception as exc:
            # Handle rate limiting gracefully
            error_str = str(exc)
            if "rate_limit" in error_str.lower() or "ratelimit" in error_str.lower():
                logger.warning(
                    "streaming.rate_limited",
                    task_id=task_id,
                    error=error_str,
                )
                # Back off and retry
                await asyncio.sleep(self.config.update_interval_seconds * 2)
                await self._schedule_update(task_id)
            else:
                logger.error(
                    "streaming.update_failed",
                    task_id=task_id,
                    error=error_str,
                )

        finally:
            self._update_tasks.pop(task_id, None)

    async def pause_stream(self, task_id: int) -> bool:
        """Pause streaming for a task.

        Args:
            task_id: The task to pause

        Returns:
            True if stream was paused
        """
        stream = self._streams.get(task_id)
        if not stream:
            return False

        async with self._lock:
            stream.is_paused = True
            # Append pause indicator
            stream.buffer += "\n\n:pause_button: *Streaming paused*"

        await self._force_update(task_id)
        logger.info("streaming.paused", task_id=task_id)
        return True

    async def resume_stream(self, task_id: int) -> bool:
        """Resume streaming for a task.

        Args:
            task_id: The task to resume

        Returns:
            True if stream was resumed
        """
        stream = self._streams.get(task_id)
        if not stream:
            return False

        async with self._lock:
            stream.is_paused = False
            # Append resume indicator
            stream.buffer += "\n\n:arrow_forward: *Streaming resumed*"

        await self._force_update(task_id)
        logger.info("streaming.resumed", task_id=task_id)
        return True

    def is_paused(self, task_id: int) -> bool:
        """Check if a stream is paused.

        Args:
            task_id: The task to check

        Returns:
            True if stream is paused
        """
        stream = self._streams.get(task_id)
        return stream.is_paused if stream else False

    async def _force_update(self, task_id: int) -> None:
        """Force an immediate update to Slack.

        Args:
            task_id: The task's stream
        """
        stream = self._streams.get(task_id)
        if not stream or not self.slack_client:
            return

        try:
            self.slack_client.chat_update(
                channel=stream.channel_id,
                ts=stream.message_ts,
                text=stream.buffer,
            )
            stream.last_update_time = datetime.now(timezone.utc).replace(tzinfo=None)
            stream.update_count += 1
        except Exception as exc:
            logger.error(
                "streaming.force_update_failed",
                task_id=task_id,
                error=str(exc),
            )

    async def finalize_stream(
        self,
        task_id: int,
        final_content: Optional[str] = None,
        success: bool = True,
    ) -> None:
        """Finalize a stream with optional final content.

        Args:
            task_id: The task's stream
            final_content: Optional content to append at the end
            success: Whether the task succeeded
        """
        stream = self._streams.get(task_id)
        if not stream:
            return

        async with self._lock:
            # Add final status
            status = ":white_check_mark: *Completed*" if success else ":x: *Failed*"
            stream.buffer += f"\n\n{status}"

            if final_content:
                stream.buffer = self._apply_sliding_window(
                    stream.buffer + f"\n{final_content}",
                    self.config.max_message_length,
                )

        await self._force_update(task_id)
        await self._cleanup_stream(task_id)

        logger.info(
            "streaming.finalized",
            task_id=task_id,
            total_chars=stream.total_chars_streamed,
            update_count=stream.update_count,
            success=success,
        )

    async def stop_stream(self, task_id: int) -> None:
        """Stop and cleanup a stream without finalizing.

        Args:
            task_id: The task's stream
        """
        await self._cleanup_stream(task_id)
        logger.debug("streaming.stopped", task_id=task_id)

    async def _cleanup_stream(self, task_id: int) -> None:
        """Cleanup stream resources.

        Args:
            task_id: The task's stream
        """
        # Cancel any pending update
        if task_id in self._update_tasks:
            self._update_tasks[task_id].cancel()
            del self._update_tasks[task_id]

        # Remove stream state
        self._streams.pop(task_id, None)

    def get_stream_stats(self, task_id: int) -> Optional[dict]:
        """Get statistics for a stream.

        Args:
            task_id: The task's stream

        Returns:
            Stats dict or None if no stream
        """
        stream = self._streams.get(task_id)
        if not stream:
            return None

        elapsed = (
            datetime.now(timezone.utc).replace(tzinfo=None) - stream.created_at
        ).total_seconds()

        return {
            "task_id": task_id,
            "phase": stream.current_phase,
            "buffer_length": len(stream.buffer),
            "total_chars_streamed": stream.total_chars_streamed,
            "update_count": stream.update_count,
            "is_paused": stream.is_paused,
            "elapsed_seconds": elapsed,
        }

    def get_active_streams(self) -> list[int]:
        """Get list of task IDs with active streams.

        Returns:
            List of task IDs
        """
        return list(self._streams.keys())
