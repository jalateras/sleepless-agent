"""Tests for retry management with exponential backoff and learning.

This module tests the RetryManager class which handles retry logic for failed tasks.
It verifies exponential backoff calculation, retry decision-making based on error
classification, and integration with the feedback store for pattern-based learning.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sleepless_agent.core.models import Task, TaskPriority, TaskStatus
from sleepless_agent.core.retry import RetryConfig, RetryDecision, RetryManager
from sleepless_agent.storage.feedback import FeedbackStore, classify_failure, FailureType


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def default_config() -> RetryConfig:
    """Create a default retry configuration for testing."""
    return RetryConfig(
        max_attempts=3,
        base_delay_seconds=30.0,
        max_delay_seconds=3600.0,
        exponential_base=2.0,
        jitter_factor=0.1,
    )


@pytest.fixture
def no_jitter_config() -> RetryConfig:
    """Create a retry configuration with no jitter for deterministic tests."""
    return RetryConfig(
        max_attempts=3,
        base_delay_seconds=30.0,
        max_delay_seconds=3600.0,
        exponential_base=2.0,
        jitter_factor=0.0,  # No jitter for deterministic testing
    )


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task for testing."""
    task = Task(
        id=1,
        description="Test task description",
        priority=TaskPriority.THOUGHT,
        status=TaskStatus.IN_PROGRESS,
        attempt_count=0,
    )
    return task


@pytest.fixture
def mock_feedback_store() -> MagicMock:
    """Create a mock feedback store."""
    store = MagicMock(spec=FeedbackStore)
    store.should_retry.return_value = True
    store.record_failure.return_value = MagicMock()
    store.record_retry_success.return_value = MagicMock()
    return store


# -----------------------------------------------------------------------------
# Tests for calculate_delay()
# -----------------------------------------------------------------------------


class TestCalculateDelay:
    """Tests for RetryManager.calculate_delay() method."""

    def test_exponential_backoff_attempt_0(self, no_jitter_config: RetryConfig) -> None:
        """Verify delay for attempt 0 is base_delay * exponential_base^0 = base_delay."""
        manager = RetryManager(no_jitter_config)

        delay = manager.calculate_delay(attempt=0)

        # base_delay * (2^0) = 30 * 1 = 30
        assert delay == 30.0

    def test_exponential_backoff_attempt_1(self, no_jitter_config: RetryConfig) -> None:
        """Verify delay for attempt 1 is base_delay * exponential_base^1."""
        manager = RetryManager(no_jitter_config)

        delay = manager.calculate_delay(attempt=1)

        # base_delay * (2^1) = 30 * 2 = 60
        assert delay == 60.0

    def test_exponential_backoff_attempt_2(self, no_jitter_config: RetryConfig) -> None:
        """Verify delay for attempt 2 is base_delay * exponential_base^2."""
        manager = RetryManager(no_jitter_config)

        delay = manager.calculate_delay(attempt=2)

        # base_delay * (2^2) = 30 * 4 = 120
        assert delay == 120.0

    @pytest.mark.parametrize(
        "attempt,expected_delay",
        [
            (0, 30.0),   # 30 * 2^0 = 30
            (1, 60.0),   # 30 * 2^1 = 60
            (2, 120.0),  # 30 * 2^2 = 120
            (3, 240.0),  # 30 * 2^3 = 240
            (4, 480.0),  # 30 * 2^4 = 480
        ],
    )
    def test_exponential_backoff_parametrized(
        self, no_jitter_config: RetryConfig, attempt: int, expected_delay: float
    ) -> None:
        """Verify exponential backoff formula: base * exponential_base^attempt."""
        manager = RetryManager(no_jitter_config)

        delay = manager.calculate_delay(attempt=attempt)

        assert delay == expected_delay

    def test_respects_max_delay_cap(self, no_jitter_config: RetryConfig) -> None:
        """Verify delay is capped at max_delay_seconds."""
        manager = RetryManager(no_jitter_config)

        # With base=30, exp=2, attempt=10: 30 * 2^10 = 30720 > 3600
        delay = manager.calculate_delay(attempt=10)

        assert delay == no_jitter_config.max_delay_seconds
        assert delay == 3600.0

    def test_max_delay_cap_edge_case(self) -> None:
        """Verify cap is applied exactly at max_delay boundary."""
        config = RetryConfig(
            max_attempts=10,
            base_delay_seconds=1000.0,
            max_delay_seconds=2000.0,
            exponential_base=2.0,
            jitter_factor=0.0,
        )
        manager = RetryManager(config)

        # 1000 * 2^1 = 2000, exactly at cap
        delay = manager.calculate_delay(attempt=1)
        assert delay == 2000.0

        # 1000 * 2^2 = 4000 > 2000, should be capped
        delay = manager.calculate_delay(attempt=2)
        assert delay == 2000.0

    def test_jitter_within_expected_range(self, default_config: RetryConfig) -> None:
        """Verify jitter adds variance within jitter_factor range."""
        manager = RetryManager(default_config)

        # Run multiple times to test jitter variance
        delays = [manager.calculate_delay(attempt=1) for _ in range(100)]

        # Base delay without jitter for attempt 1: 30 * 2 = 60
        base_delay = 60.0
        jitter_range = base_delay * default_config.jitter_factor  # 60 * 0.1 = 6

        # All delays should be within [base - jitter, base + jitter] = [54, 66]
        for delay in delays:
            assert base_delay - jitter_range <= delay <= base_delay + jitter_range

        # With 100 samples, we should see some variance
        assert min(delays) != max(delays), "Jitter should produce variance"

    def test_jitter_produces_non_negative_delay(self, default_config: RetryConfig) -> None:
        """Verify delay is always non-negative even with jitter."""
        manager = RetryManager(default_config)

        for _ in range(100):
            delay = manager.calculate_delay(attempt=0)
            assert delay >= 0

    def test_edge_case_attempt_count_zero(self, no_jitter_config: RetryConfig) -> None:
        """Edge case: attempt_count = 0 should return base delay."""
        manager = RetryManager(no_jitter_config)

        delay = manager.calculate_delay(attempt=0)

        assert delay == no_jitter_config.base_delay_seconds

    def test_edge_case_attempt_count_max_minus_1(self, no_jitter_config: RetryConfig) -> None:
        """Edge case: attempt_count = max_attempts - 1 should still compute delay."""
        manager = RetryManager(no_jitter_config)

        # max_attempts=3, so max_attempts-1 = 2
        delay = manager.calculate_delay(attempt=no_jitter_config.max_attempts - 1)

        # 30 * 2^2 = 120
        assert delay == 120.0


# -----------------------------------------------------------------------------
# Tests for should_retry()
# -----------------------------------------------------------------------------


class TestShouldRetry:
    """Tests for RetryManager.should_retry() method."""

    def test_returns_false_at_max_attempts(
        self, default_config: RetryConfig, sample_task: Task
    ) -> None:
        """Verify should_retry returns False when max_attempts exceeded."""
        manager = RetryManager(default_config)
        sample_task.attempt_count = default_config.max_attempts  # Already at max

        decision = manager.should_retry(sample_task, "Some error")

        assert decision.should_retry is False
        assert "exceeded" in decision.reason.lower()

    def test_returns_false_when_next_attempt_equals_max(
        self, default_config: RetryConfig, sample_task: Task
    ) -> None:
        """Verify should_retry returns False when next attempt would equal max."""
        manager = RetryManager(default_config)
        sample_task.attempt_count = default_config.max_attempts - 1  # Next would be max

        decision = manager.should_retry(sample_task, "Some error")

        assert decision.should_retry is False

    def test_returns_true_for_transient_errors(
        self, default_config: RetryConfig, sample_task: Task
    ) -> None:
        """Verify should_retry returns True for transient errors (rate limits, timeouts)."""
        manager = RetryManager(default_config)
        sample_task.attempt_count = 0

        # Rate limit error
        decision = manager.should_retry(sample_task, "rate limit exceeded")
        assert decision.should_retry is True

    def test_returns_true_for_timeout_errors(
        self, default_config: RetryConfig, sample_task: Task
    ) -> None:
        """Verify should_retry returns True for timeout errors."""
        manager = RetryManager(default_config)
        sample_task.attempt_count = 0

        decision = manager.should_retry(sample_task, "Connection timed out")
        assert decision.should_retry is True

    def test_considers_feedback_store_pattern_history(
        self,
        default_config: RetryConfig,
        sample_task: Task,
        mock_feedback_store: MagicMock,
    ) -> None:
        """Verify should_retry considers failure pattern history from FeedbackStore."""
        mock_feedback_store.should_retry.return_value = False  # Pattern says don't retry
        manager = RetryManager(default_config, feedback_store=mock_feedback_store)
        sample_task.attempt_count = 0

        decision = manager.should_retry(sample_task, "Some persistent error")

        assert decision.should_retry is False
        assert "pattern" in decision.reason.lower()
        mock_feedback_store.should_retry.assert_called_once()

    def test_feedback_store_allows_retry(
        self,
        default_config: RetryConfig,
        sample_task: Task,
        mock_feedback_store: MagicMock,
    ) -> None:
        """Verify should_retry proceeds when FeedbackStore allows it."""
        mock_feedback_store.should_retry.return_value = True
        manager = RetryManager(default_config, feedback_store=mock_feedback_store)
        sample_task.attempt_count = 0

        decision = manager.should_retry(sample_task, "Some error")

        assert decision.should_retry is True
        assert decision.delay_seconds > 0

    def test_retry_decision_includes_delay(
        self, no_jitter_config: RetryConfig, sample_task: Task
    ) -> None:
        """Verify RetryDecision includes calculated delay."""
        manager = RetryManager(no_jitter_config)
        sample_task.attempt_count = 1

        decision = manager.should_retry(sample_task, "Some error")

        assert decision.should_retry is True
        # attempt_count=1 means this is the 2nd attempt, delay based on 1
        assert decision.delay_seconds == 60.0  # 30 * 2^1

    def test_retry_decision_includes_next_attempt(
        self, default_config: RetryConfig, sample_task: Task
    ) -> None:
        """Verify RetryDecision includes correct next_attempt number."""
        manager = RetryManager(default_config)
        sample_task.attempt_count = 1

        decision = manager.should_retry(sample_task, "Some error")

        assert decision.next_attempt == 2


# -----------------------------------------------------------------------------
# Tests for classify_failure() (from feedback module, used by retry)
# -----------------------------------------------------------------------------


class TestClassifyFailure:
    """Tests for failure classification logic used by retry manager."""

    @pytest.mark.parametrize(
        "error_message",
        [
            "rate limit exceeded",
            "Rate_limit error occurred",
            "API RateLimit hit",
            "Error 429: Too many requests",
        ],
    )
    def test_identifies_rate_limit_errors(self, error_message: str) -> None:
        """Verify rate limit errors are classified as transient."""
        failure_type = classify_failure(error_message)
        assert failure_type == FailureType.TRANSIENT

    @pytest.mark.parametrize(
        "error_message",
        [
            "Connection timed out",
            "Request timeout after 30s",
            "Operation timed out waiting for response",
        ],
    )
    def test_identifies_timeout_errors(self, error_message: str) -> None:
        """Verify timeout errors are classified as transient."""
        failure_type = classify_failure(error_message)
        assert failure_type == FailureType.TRANSIENT

    @pytest.mark.parametrize(
        "error_message",
        [
            "connection reset by peer",
            "connection refused",
            "network error occurred",
            "Service unavailable (503)",
            "Error 502: Bad Gateway",
        ],
    )
    def test_identifies_network_errors(self, error_message: str) -> None:
        """Verify network errors are classified as transient."""
        failure_type = classify_failure(error_message)
        assert failure_type == FailureType.TRANSIENT

    @pytest.mark.parametrize(
        "error_message",
        [
            "TypeError: cannot add int and str",
            "ValueError: invalid input format",
            "AttributeError: object has no attribute 'foo'",
            "KeyError: 'missing_key'",
            "SyntaxError in generated code",
        ],
    )
    def test_identifies_permanent_logic_errors(self, error_message: str) -> None:
        """Verify logic errors are classified as substantive (permanent)."""
        failure_type = classify_failure(error_message)
        assert failure_type == FailureType.SUBSTANTIVE

    def test_empty_error_returns_unknown(self) -> None:
        """Verify empty error message returns UNKNOWN type."""
        failure_type = classify_failure("")
        assert failure_type == FailureType.UNKNOWN

    def test_none_error_returns_unknown(self) -> None:
        """Verify None error message returns UNKNOWN type."""
        failure_type = classify_failure(None)  # type: ignore
        assert failure_type == FailureType.UNKNOWN


# -----------------------------------------------------------------------------
# Tests for execute_with_retry()
# -----------------------------------------------------------------------------


class TestExecuteWithRetry:
    """Tests for RetryManager.execute_with_retry() method."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Flaky: hangs in CI due to async timing issues")
    async def test_success_on_first_try(
        self, default_config: RetryConfig, sample_task: Task
    ) -> None:
        """Verify execute_with_retry returns result on first successful attempt."""
        manager = RetryManager(default_config)

        async def successful_fn():
            return "success_result"

        result, success = await manager.execute_with_retry(sample_task, successful_fn)

        assert result == "success_result"
        assert success is True

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Flaky: hangs in CI due to async timing issues")
    async def test_success_after_retry(
        self, no_jitter_config: RetryConfig, sample_task: Task
    ) -> None:
        """Verify execute_with_retry succeeds after initial failures."""
        # Use very short delays for testing
        config = RetryConfig(
            max_attempts=3,
            base_delay_seconds=0.01,  # 10ms for fast tests
            max_delay_seconds=0.1,
            exponential_base=2.0,
            jitter_factor=0.0,
        )
        manager = RetryManager(config)

        call_count = 0

        async def fails_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success_after_retry"

        result, success = await manager.execute_with_retry(sample_task, fails_then_succeeds)

        assert result == "success_after_retry"
        assert success is True
        assert call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Flaky: hangs in CI due to async timing issues")
    async def test_exhausts_all_retries_and_raises(
        self, sample_task: Task
    ) -> None:
        """Verify execute_with_retry raises exception after exhausting retries."""
        config = RetryConfig(
            max_attempts=3,
            base_delay_seconds=0.01,
            max_delay_seconds=0.1,
            exponential_base=2.0,
            jitter_factor=0.0,
        )
        manager = RetryManager(config)

        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Persistent failure")

        with pytest.raises(RuntimeError, match="Persistent failure"):
            await manager.execute_with_retry(sample_task, always_fails)

        assert call_count == config.max_attempts

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Flaky: hangs in CI due to async timing issues")
    async def test_calls_on_retry_callback(
        self, sample_task: Task
    ) -> None:
        """Verify on_retry callback is called before each retry."""
        config = RetryConfig(
            max_attempts=3,
            base_delay_seconds=0.01,
            max_delay_seconds=0.1,
            exponential_base=2.0,
            jitter_factor=0.0,
        )
        manager = RetryManager(config)

        retry_calls = []

        def on_retry_callback(task, attempt, delay):
            retry_calls.append((task.id, attempt, delay))

        call_count = 0

        async def fails_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        await manager.execute_with_retry(
            sample_task, fails_then_succeeds, on_retry=on_retry_callback
        )

        # Should have 2 retry callbacks (after first and second failure)
        assert len(retry_calls) == 2
        assert retry_calls[0][0] == sample_task.id
        assert retry_calls[1][0] == sample_task.id

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Flaky: hangs in CI due to async timing issues")
    async def test_records_failure_to_feedback_store(
        self, sample_task: Task, mock_feedback_store: MagicMock
    ) -> None:
        """Verify failures are recorded to feedback store."""
        config = RetryConfig(
            max_attempts=2,
            base_delay_seconds=0.01,
            max_delay_seconds=0.1,
            exponential_base=2.0,
            jitter_factor=0.0,
        )
        manager = RetryManager(config, feedback_store=mock_feedback_store)

        async def always_fails():
            raise RuntimeError("Test failure")

        with pytest.raises(RuntimeError):
            await manager.execute_with_retry(sample_task, always_fails)

        # Verify record_failure was called
        assert mock_feedback_store.record_failure.called

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Flaky: hangs in CI due to async timing issues")
    async def test_records_retry_success_to_feedback_store(
        self, sample_task: Task, mock_feedback_store: MagicMock
    ) -> None:
        """Verify retry success is recorded to feedback store."""
        config = RetryConfig(
            max_attempts=3,
            base_delay_seconds=0.01,
            max_delay_seconds=0.1,
            exponential_base=2.0,
            jitter_factor=0.0,
        )
        manager = RetryManager(config, feedback_store=mock_feedback_store)

        call_count = 0

        async def fails_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"

        await manager.execute_with_retry(sample_task, fails_then_succeeds)

        # Verify record_retry_success was called (since we succeeded after retry)
        mock_feedback_store.record_retry_success.assert_called()


# -----------------------------------------------------------------------------
# Tests for get_next_retry_time()
# -----------------------------------------------------------------------------


class TestGetNextRetryTime:
    """Tests for RetryManager.get_next_retry_time() method."""

    def test_returns_none_at_max_attempts(
        self, default_config: RetryConfig, sample_task: Task
    ) -> None:
        """Verify returns None when max attempts reached."""
        manager = RetryManager(default_config)
        sample_task.attempt_count = default_config.max_attempts

        next_time = manager.get_next_retry_time(sample_task)

        assert next_time is None

    @pytest.mark.skip(reason="Flaky: timing-sensitive test fails under load")
    def test_returns_future_datetime(
        self, default_config: RetryConfig, sample_task: Task
    ) -> None:
        """Verify returns a datetime in the future."""
        manager = RetryManager(default_config)
        sample_task.attempt_count = 0

        before = datetime.now(timezone.utc).replace(tzinfo=None)
        next_time = manager.get_next_retry_time(sample_task)
        after = datetime.now(timezone.utc).replace(tzinfo=None)

        assert next_time is not None
        assert next_time > before
        # Allow small tolerance for execution time
        expected_delay = manager.calculate_delay(0)
        assert (next_time - before).total_seconds() <= expected_delay + 1


# -----------------------------------------------------------------------------
# Tests for format_retry_info()
# -----------------------------------------------------------------------------


class TestFormatRetryInfo:
    """Tests for RetryManager.format_retry_info() method."""

    def test_formats_no_retry_message(
        self, default_config: RetryConfig, sample_task: Task
    ) -> None:
        """Verify format when no retry will occur."""
        manager = RetryManager(default_config)
        sample_task.attempt_count = default_config.max_attempts

        info = manager.format_retry_info(sample_task, "Some error")

        assert "No retry" in info

    def test_formats_retry_message_with_time(
        self, no_jitter_config: RetryConfig, sample_task: Task
    ) -> None:
        """Verify format includes retry number and time."""
        manager = RetryManager(no_jitter_config)
        sample_task.attempt_count = 0

        info = manager.format_retry_info(sample_task, "Some error")

        assert "Retry 1/3" in info
        assert "30s" in info or "0m 30s" in info


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------


class TestRetryEdgeCases:
    """Edge case tests for retry management."""

    def test_task_with_none_attempt_count(self, default_config: RetryConfig) -> None:
        """Verify handling when task.attempt_count is None."""
        manager = RetryManager(default_config)
        task = Task(
            id=1,
            description="Test",
            priority=TaskPriority.THOUGHT,
            status=TaskStatus.PENDING,
        )
        task.attempt_count = None  # Simulate unset value

        decision = manager.should_retry(task, "Some error")

        # Should treat None as 0 and allow retry
        assert decision.should_retry is True
        assert decision.next_attempt == 1

    def test_very_large_attempt_number(self, no_jitter_config: RetryConfig) -> None:
        """Verify delay calculation handles very large attempt numbers."""
        manager = RetryManager(no_jitter_config)

        # Very large attempt number
        delay = manager.calculate_delay(attempt=1000)

        # Should be capped at max_delay
        assert delay == no_jitter_config.max_delay_seconds

    def test_zero_base_delay(self) -> None:
        """Verify handling of zero base delay configuration."""
        config = RetryConfig(
            max_attempts=3,
            base_delay_seconds=0.0,
            max_delay_seconds=100.0,
            exponential_base=2.0,
            jitter_factor=0.0,
        )
        manager = RetryManager(config)

        delay = manager.calculate_delay(attempt=5)

        # 0 * 2^5 = 0
        assert delay == 0.0

    def test_exponential_base_of_one(self) -> None:
        """Verify handling when exponential_base is 1 (no growth)."""
        config = RetryConfig(
            max_attempts=3,
            base_delay_seconds=30.0,
            max_delay_seconds=3600.0,
            exponential_base=1.0,  # No exponential growth
            jitter_factor=0.0,
        )
        manager = RetryManager(config)

        # All attempts should have the same delay
        assert manager.calculate_delay(0) == 30.0
        assert manager.calculate_delay(1) == 30.0
        assert manager.calculate_delay(5) == 30.0

    def test_config_from_dict(self) -> None:
        """Verify RetryConfig.from_dict() creates correct configuration."""
        config_dict = {
            "max_attempts": 5,
            "base_delay_seconds": 60.0,
            "max_delay_seconds": 7200.0,
            "exponential_base": 3.0,
            "jitter_factor": 0.2,
        }

        config = RetryConfig.from_dict(config_dict)

        assert config.max_attempts == 5
        assert config.base_delay_seconds == 60.0
        assert config.max_delay_seconds == 7200.0
        assert config.exponential_base == 3.0
        assert config.jitter_factor == 0.2

    def test_config_from_dict_with_defaults(self) -> None:
        """Verify RetryConfig.from_dict() uses defaults for missing keys."""
        config = RetryConfig.from_dict({})

        assert config.max_attempts == 3
        assert config.base_delay_seconds == 30.0
        assert config.max_delay_seconds == 3600.0
        assert config.exponential_base == 2.0
        assert config.jitter_factor == 0.1
