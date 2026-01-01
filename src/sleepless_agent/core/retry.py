"""Retry management with exponential backoff and learning."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, TypeVar

from sleepless_agent.core.models import Task
from sleepless_agent.monitoring.logging import get_logger
from sleepless_agent.storage.feedback import FeedbackStore

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3  # Maximum total attempts (including first)
    base_delay_seconds: float = 30.0  # Initial delay before first retry
    max_delay_seconds: float = 3600.0  # Maximum delay (1 hour)
    exponential_base: float = 2.0  # Multiplier for exponential backoff
    jitter_factor: float = 0.1  # Random jitter to prevent thundering herd

    @classmethod
    def from_dict(cls, config: dict) -> "RetryConfig":
        """Create RetryConfig from a dictionary."""
        return cls(
            max_attempts=config.get("max_attempts", 3),
            base_delay_seconds=config.get("base_delay_seconds", 30.0),
            max_delay_seconds=config.get("max_delay_seconds", 3600.0),
            exponential_base=config.get("exponential_base", 2.0),
            jitter_factor=config.get("jitter_factor", 0.1),
        )


@dataclass
class RetryDecision:
    """Result of a retry decision."""

    should_retry: bool
    delay_seconds: float = 0.0
    reason: str = ""
    next_attempt: int = 0


class RetryManager:
    """Manages retry logic with exponential backoff and learning.

    Uses failure patterns from FeedbackStore to make intelligent retry decisions.
    Implements exponential backoff with jitter to prevent thundering herd problems.
    """

    def __init__(
        self,
        config: RetryConfig,
        feedback_store: Optional[FeedbackStore] = None,
    ):
        """Initialize retry manager.

        Args:
            config: Retry configuration
            feedback_store: Optional feedback store for pattern-based decisions
        """
        self.config = config
        self.feedback_store = feedback_store

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number using exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds before the next attempt
        """
        import random

        # Exponential backoff: base_delay * (exponential_base ^ attempt)
        delay = self.config.base_delay_seconds * (
            self.config.exponential_base ** attempt
        )

        # Cap at maximum delay
        delay = min(delay, self.config.max_delay_seconds)

        # Add jitter to prevent thundering herd
        jitter = delay * self.config.jitter_factor * (2 * random.random() - 1)
        delay = max(0, delay + jitter)

        return delay

    def should_retry(
        self,
        task: Task,
        error_message: str,
    ) -> RetryDecision:
        """Determine if a failed task should be retried.

        Args:
            task: The failed task
            error_message: The error message from the failure

        Returns:
            RetryDecision with should_retry, delay, and reason
        """
        current_attempts = task.attempt_count or 0
        next_attempt = current_attempts + 1

        # Check if we've exceeded max attempts
        if next_attempt >= self.config.max_attempts:
            return RetryDecision(
                should_retry=False,
                reason=f"Max attempts ({self.config.max_attempts}) exceeded",
                next_attempt=next_attempt,
            )

        # Use feedback store for pattern-based decision if available
        if self.feedback_store:
            should_retry = self.feedback_store.should_retry(
                error_message=error_message,
                current_attempts=current_attempts,
                max_attempts=self.config.max_attempts,
            )

            if not should_retry:
                return RetryDecision(
                    should_retry=False,
                    reason="Pattern suggests retry unlikely to succeed",
                    next_attempt=next_attempt,
                )

        # Calculate delay for this retry
        delay = self.calculate_delay(current_attempts)

        return RetryDecision(
            should_retry=True,
            delay_seconds=delay,
            reason=f"Retry {next_attempt}/{self.config.max_attempts}",
            next_attempt=next_attempt,
        )

    async def execute_with_retry(
        self,
        task: Task,
        execute_fn: Callable[[], T],
        on_retry: Optional[Callable[[Task, int, float], None]] = None,
    ) -> tuple[T, bool]:
        """Execute a function with retry logic.

        Args:
            task: The task being executed
            execute_fn: Async function to execute
            on_retry: Optional callback called before each retry (task, attempt, delay)

        Returns:
            Tuple of (result, success) where success indicates if it completed
            without exhausting retries

        Raises:
            Exception: If all retries are exhausted, raises the last exception
        """
        last_error: Optional[Exception] = None
        current_attempt = task.attempt_count or 0

        while current_attempt < self.config.max_attempts:
            try:
                result = await execute_fn()

                # Record success if we were retrying
                if current_attempt > 0 and self.feedback_store and last_error:
                    try:
                        self.feedback_store.record_retry_success(str(last_error))
                    except Exception as e:
                        logger.debug(f"Failed to record retry success: {e}")

                return result, True

            except Exception as e:
                last_error = e
                error_str = str(e)

                # Record the failure pattern
                if self.feedback_store:
                    try:
                        self.feedback_store.record_failure(error_str, task=task)
                    except Exception as record_error:
                        logger.debug(f"Failed to record failure pattern: {record_error}")

                # Check if we should retry
                decision = self.should_retry(task, error_str)

                if not decision.should_retry:
                    logger.info(
                        "retry.giving_up",
                        task_id=task.id,
                        attempts=current_attempt + 1,
                        reason=decision.reason,
                        error=error_str[:200],
                    )
                    raise

                # Log and wait before retry
                logger.warning(
                    "retry.scheduled",
                    task_id=task.id,
                    attempt=decision.next_attempt,
                    max_attempts=self.config.max_attempts,
                    delay_seconds=round(decision.delay_seconds, 1),
                    error=error_str[:200],
                )

                # Callback before retry
                if on_retry:
                    try:
                        on_retry(task, decision.next_attempt, decision.delay_seconds)
                    except Exception as callback_error:
                        logger.debug(f"Retry callback failed: {callback_error}")

                # Wait before retry
                await asyncio.sleep(decision.delay_seconds)

                current_attempt = decision.next_attempt

        # Should not reach here, but handle it gracefully
        if last_error:
            raise last_error
        raise RuntimeError("Retry exhausted without error")

    def get_next_retry_time(self, task: Task) -> Optional[datetime]:
        """Calculate when the next retry should occur.

        Args:
            task: The failed task

        Returns:
            datetime of next retry, or None if no retry should occur
        """
        current_attempts = task.attempt_count or 0

        if current_attempts >= self.config.max_attempts:
            return None

        delay = self.calculate_delay(current_attempts)
        return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=delay)

    def format_retry_info(self, task: Task, error_message: str) -> str:
        """Format a human-readable retry status message.

        Args:
            task: The failed task
            error_message: The error message

        Returns:
            Human-readable status string
        """
        decision = self.should_retry(task, error_message)

        if not decision.should_retry:
            return f"No retry: {decision.reason}"

        minutes = int(decision.delay_seconds // 60)
        seconds = int(decision.delay_seconds % 60)

        if minutes > 0:
            return f"Retry {decision.next_attempt}/{self.config.max_attempts} in {minutes}m {seconds}s"
        return f"Retry {decision.next_attempt}/{self.config.max_attempts} in {seconds}s"
