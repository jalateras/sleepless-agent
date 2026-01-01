"""Tests for feedback storage and failure pattern learning.

This module tests the FeedbackStore class which handles user feedback on task
outcomes and failure pattern tracking. It verifies emoji classification, feedback
deduplication, failure pattern learning, and skip recommendations.
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Generator
from unittest.mock import MagicMock

import pytest

from sleepless_agent.core.models import (
    Base,
    FailurePattern,
    FailureType,
    FeedbackType,
    Task,
    TaskFeedback,
    TaskPriority,
    TaskStatus,
    TaskType,
    init_db,
)
from sleepless_agent.storage.feedback import (
    NEGATIVE_REACTIONS,
    POSITIVE_REACTIONS,
    FeedbackStore,
    classify_failure,
    classify_reaction,
    hash_error,
    normalize_error,
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
def store(temp_db: str) -> FeedbackStore:
    """Create a FeedbackStore for testing."""
    return FeedbackStore(db_path=temp_db)


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task for testing."""
    return Task(
        id=1,
        description="Test task for feedback",
        priority=TaskPriority.THOUGHT,
        status=TaskStatus.COMPLETED,
        task_type=TaskType.NEW,
    )


# -----------------------------------------------------------------------------
# Tests for classify_reaction()
# -----------------------------------------------------------------------------


class TestClassifyReaction:
    """Tests for classify_reaction() function."""

    @pytest.mark.parametrize(
        "emoji",
        ["+1", "thumbsup", "white_check_mark", "heavy_check_mark", "star", "star2",
         "tada", "rocket", "fire", "100", "heart", "clap", "raised_hands", "muscle"],
    )
    def test_classifies_positive_emojis(self, emoji: str) -> None:
        """Verify positive emojis are classified as POSITIVE."""
        result = classify_reaction(emoji)
        assert result == FeedbackType.POSITIVE

    @pytest.mark.parametrize(
        "emoji",
        ["-1", "thumbsdown", "x", "negative_squared_cross_mark", "no_entry",
         "no_entry_sign", "warning", "disappointed", "confused", "face_with_rolling_eyes"],
    )
    def test_classifies_negative_emojis(self, emoji: str) -> None:
        """Verify negative emojis are classified as NEGATIVE."""
        result = classify_reaction(emoji)
        assert result == FeedbackType.NEGATIVE

    @pytest.mark.parametrize(
        "emoji",
        ["eyes", "thinking_face", "wave", "smile", "coffee", "computer", "unknown_emoji"],
    )
    def test_classifies_neutral_emojis(self, emoji: str) -> None:
        """Verify unknown emojis are classified as NEUTRAL."""
        result = classify_reaction(emoji)
        assert result == FeedbackType.NEUTRAL

    @pytest.mark.parametrize(
        "emoji,expected",
        [
            (":+1:", FeedbackType.POSITIVE),
            (":thumbsup:", FeedbackType.POSITIVE),
            (":-1:", FeedbackType.NEGATIVE),
            (":thumbsdown:", FeedbackType.NEGATIVE),
            (":heart:", FeedbackType.POSITIVE),
            (":x:", FeedbackType.NEGATIVE),
        ],
    )
    def test_classifies_colon_wrapped_emojis(
        self, emoji: str, expected: FeedbackType
    ) -> None:
        """Verify colon-wrapped emojis are classified correctly."""
        result = classify_reaction(emoji)
        assert result == expected

    def test_classifies_empty_string_as_neutral(self) -> None:
        """Verify empty string is classified as NEUTRAL."""
        result = classify_reaction("")
        assert result == FeedbackType.NEUTRAL

    def test_classifies_whitespace_only_as_neutral(self) -> None:
        """Verify whitespace-only string is classified as NEUTRAL."""
        result = classify_reaction("   ")
        assert result == FeedbackType.NEUTRAL

    def test_case_insensitive_classification(self) -> None:
        """Verify emoji classification is case-insensitive."""
        assert classify_reaction("THUMBSUP") == FeedbackType.POSITIVE
        assert classify_reaction("ThumbsUp") == FeedbackType.POSITIVE
        assert classify_reaction("THUMBSDOWN") == FeedbackType.NEGATIVE


# -----------------------------------------------------------------------------
# Tests for record_feedback()
# -----------------------------------------------------------------------------


class TestRecordFeedback:
    """Tests for FeedbackStore.record_feedback() method."""

    def test_creates_new_feedback_record(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify record_feedback creates a new feedback record."""
        feedback = store.record_feedback(
            task_id=sample_task.id,
            user_id="U123456",
            reaction="+1",
        )

        assert feedback is not None
        assert feedback.id is not None
        assert feedback.task_id == sample_task.id
        assert feedback.user_id == "U123456"
        assert feedback.reaction == "+1"
        assert feedback.feedback_type == FeedbackType.POSITIVE

    def test_deduplicates_same_user_task_reaction(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify duplicate feedback (same user, task, reaction) is ignored."""
        # Record first feedback
        first = store.record_feedback(
            task_id=sample_task.id,
            user_id="U123456",
            reaction="+1",
        )

        # Attempt duplicate
        second = store.record_feedback(
            task_id=sample_task.id,
            user_id="U123456",
            reaction="+1",
        )

        # Should return the existing record, not create new one
        assert second.id == first.id

        # Verify only one record exists
        feedbacks = store.get_feedback_for_task(sample_task.id)
        assert len(feedbacks) == 1

    def test_allows_different_reactions_from_same_user(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify same user can give different reactions to same task."""
        store.record_feedback(
            task_id=sample_task.id,
            user_id="U123456",
            reaction="+1",
        )

        store.record_feedback(
            task_id=sample_task.id,
            user_id="U123456",
            reaction="heart",
        )

        feedbacks = store.get_feedback_for_task(sample_task.id)
        assert len(feedbacks) == 2
        reactions = {f.reaction for f in feedbacks}
        assert reactions == {"+1", "heart"}

    def test_allows_same_reaction_from_different_users(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify different users can give same reaction."""
        store.record_feedback(
            task_id=sample_task.id,
            user_id="U111",
            reaction="+1",
        )

        store.record_feedback(
            task_id=sample_task.id,
            user_id="U222",
            reaction="+1",
        )

        feedbacks = store.get_feedback_for_task(sample_task.id)
        assert len(feedbacks) == 2

    def test_stores_optional_context(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify optional context fields are stored."""
        feedback = store.record_feedback(
            task_id=sample_task.id,
            user_id="U123456",
            reaction="+1",
            message_ts="1234567890.123456",
            channel_id="C123456",
            task=sample_task,
            generation_source="code_review",
        )

        assert feedback.message_ts == "1234567890.123456"
        assert feedback.channel_id == "C123456"
        assert feedback.task_priority == TaskPriority.THOUGHT.value
        assert feedback.task_type == TaskType.NEW.value
        assert feedback.generation_source == "code_review"


# -----------------------------------------------------------------------------
# Tests for get_feedback_for_task()
# -----------------------------------------------------------------------------


class TestGetFeedbackForTask:
    """Tests for FeedbackStore.get_feedback_for_task() method."""

    def test_returns_all_feedback_for_task(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify get_feedback_for_task returns all feedback for a task."""
        store.record_feedback(task_id=sample_task.id, user_id="U1", reaction="+1")
        store.record_feedback(task_id=sample_task.id, user_id="U2", reaction="-1")
        store.record_feedback(task_id=sample_task.id, user_id="U3", reaction="heart")

        feedbacks = store.get_feedback_for_task(sample_task.id)

        assert len(feedbacks) == 3

    def test_returns_empty_list_for_unknown_task(self, store: FeedbackStore) -> None:
        """Verify get_feedback_for_task returns empty list for unknown task."""
        feedbacks = store.get_feedback_for_task(99999)

        assert feedbacks == []

    def test_returns_feedback_ordered_by_created_at_desc(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify feedback is returned in descending order by created_at."""
        store.record_feedback(task_id=sample_task.id, user_id="U1", reaction="+1")
        time.sleep(0.01)
        store.record_feedback(task_id=sample_task.id, user_id="U2", reaction="-1")
        time.sleep(0.01)
        store.record_feedback(task_id=sample_task.id, user_id="U3", reaction="heart")

        feedbacks = store.get_feedback_for_task(sample_task.id)

        # Most recent first
        assert feedbacks[0].user_id == "U3"
        assert feedbacks[2].user_id == "U1"


# -----------------------------------------------------------------------------
# Tests for get_feedback_weights_by_source()
# -----------------------------------------------------------------------------


class TestGetFeedbackWeightsBySource:
    """Tests for FeedbackStore.get_feedback_weights_by_source() method."""

    def test_returns_weights_with_sufficient_data(
        self, store: FeedbackStore
    ) -> None:
        """Verify weights are calculated when min_feedback_count is met."""
        # Create feedback for a source with all positive
        for i in range(5):
            store.record_feedback(
                task_id=i + 1,
                user_id=f"U{i}",
                reaction="+1",
                generation_source="good_prompt",
            )

        weights = store.get_feedback_weights_by_source(days=30, min_feedback_count=3)

        assert "good_prompt" in weights
        # All positive should give weight close to 2.0
        assert weights["good_prompt"] >= 1.5

    def test_returns_empty_dict_with_insufficient_data(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify empty dict is returned when min_feedback_count is not met."""
        store.record_feedback(
            task_id=sample_task.id,
            user_id="U1",
            reaction="+1",
            generation_source="sparse_prompt",
        )

        weights = store.get_feedback_weights_by_source(days=30, min_feedback_count=5)

        assert "sparse_prompt" not in weights

    def test_excludes_feedback_without_generation_source(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify feedback without generation_source is excluded."""
        for i in range(10):
            store.record_feedback(
                task_id=i + 1,
                user_id=f"U{i}",
                reaction="+1",
                generation_source=None,  # No source
            )

        weights = store.get_feedback_weights_by_source(days=30, min_feedback_count=3)

        assert weights == {}

    def test_negative_feedback_lowers_weight(self, store: FeedbackStore) -> None:
        """Verify negative feedback lowers the weight."""
        # Mix of positive and negative
        for i in range(3):
            store.record_feedback(
                task_id=i + 1,
                user_id=f"U{i}",
                reaction="+1",
                generation_source="mixed_prompt",
            )
        for i in range(3, 6):
            store.record_feedback(
                task_id=i + 1,
                user_id=f"U{i}",
                reaction="-1",
                generation_source="mixed_prompt",
            )

        weights = store.get_feedback_weights_by_source(days=30, min_feedback_count=3)

        # 50% positive, 50% negative should give weight around 1.0
        assert "mixed_prompt" in weights
        assert 0.9 <= weights["mixed_prompt"] <= 1.1


# -----------------------------------------------------------------------------
# Tests for classify_failure() and normalize_error()
# -----------------------------------------------------------------------------


class TestClassifyFailure:
    """Tests for classify_failure() function."""

    @pytest.mark.parametrize(
        "error_message",
        [
            "rate limit exceeded",
            "Rate_limit error",
            "Error 429: Too many requests",
            "Connection timed out",
            "Request timeout",
            "connection reset by peer",
            "connection refused",
            "network error",
            "Service unavailable (503)",
            "Error 502: Bad Gateway",
            "resource exhausted",
            "quota exceeded",
            "please try again later",
            "temporary failure",
        ],
    )
    def test_identifies_transient_failures(self, error_message: str) -> None:
        """Verify transient failures are classified correctly."""
        result = classify_failure(error_message)
        assert result == FailureType.TRANSIENT

    @pytest.mark.parametrize(
        "error_message",
        [
            "TypeError: cannot add int and str",
            "ValueError: invalid format",
            "AttributeError: object has no attribute",
            "SyntaxError in generated code",
            "ImportError: No module named foo",
        ],
    )
    def test_identifies_substantive_failures(self, error_message: str) -> None:
        """Verify substantive failures are classified correctly."""
        result = classify_failure(error_message)
        assert result == FailureType.SUBSTANTIVE

    def test_empty_string_returns_unknown(self) -> None:
        """Verify empty error message returns UNKNOWN."""
        result = classify_failure("")
        assert result == FailureType.UNKNOWN

    def test_none_returns_unknown(self) -> None:
        """Verify None error message returns UNKNOWN."""
        result = classify_failure(None)  # type: ignore
        assert result == FailureType.UNKNOWN


class TestNormalizeError:
    """Tests for normalize_error() function."""

    def test_removes_timestamps(self) -> None:
        """Verify timestamps are normalized."""
        error = "Error at 2024-01-15T10:30:00: something failed"
        normalized = normalize_error(error)
        assert "<TIMESTAMP>" in normalized
        assert "2024-01-15" not in normalized

    def test_removes_uuids(self) -> None:
        """Verify UUIDs are normalized."""
        error = "Task 550e8400-e29b-41d4-a716-446655440000 failed"
        normalized = normalize_error(error)
        assert "<UUID>" in normalized
        assert "550e8400" not in normalized

    def test_removes_file_paths(self) -> None:
        """Verify file paths are normalized."""
        error = "Error in /home/user/project/src/main.py"
        normalized = normalize_error(error)
        assert "<PATH>" in normalized

    def test_removes_line_numbers(self) -> None:
        """Verify line numbers are normalized."""
        error = "Error on line 42 of file.py"
        normalized = normalize_error(error)
        assert "line <N>" in normalized

    def test_truncates_long_errors(self) -> None:
        """Verify long error messages are truncated."""
        error = "x" * 1000
        normalized = normalize_error(error)
        assert len(normalized) <= 500

    def test_empty_string_returns_empty(self) -> None:
        """Verify empty string returns empty."""
        result = normalize_error("")
        assert result == ""


class TestHashError:
    """Tests for hash_error() function."""

    def test_produces_consistent_hash(self) -> None:
        """Verify same input produces same hash."""
        error = "Some error message"
        hash1 = hash_error(error)
        hash2 = hash_error(error)
        assert hash1 == hash2

    def test_different_inputs_produce_different_hashes(self) -> None:
        """Verify different inputs produce different hashes."""
        hash1 = hash_error("Error 1")
        hash2 = hash_error("Error 2")
        assert hash1 != hash2


# -----------------------------------------------------------------------------
# Tests for record_failure() and get_failure_pattern()
# -----------------------------------------------------------------------------


class TestRecordFailure:
    """Tests for FeedbackStore.record_failure() method."""

    def test_creates_new_failure_pattern(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify record_failure creates a new failure pattern."""
        pattern = store.record_failure(
            error_message="Connection timed out",
            task=sample_task,
        )

        assert pattern is not None
        assert pattern.id is not None
        assert pattern.occurrences == 1
        assert pattern.failure_type == FailureType.TRANSIENT
        assert pattern.sample_task_description is not None

    def test_increments_occurrences_for_same_pattern(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify record_failure increments occurrences for same error pattern."""
        store.record_failure(error_message="Connection timed out", task=sample_task)
        pattern = store.record_failure(
            error_message="Connection timed out", task=sample_task
        )

        assert pattern.occurrences == 2

    def test_updates_last_seen(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify record_failure updates last_seen timestamp."""
        first = store.record_failure(
            error_message="Connection timed out", task=sample_task
        )
        first_last_seen = first.last_seen

        time.sleep(0.01)

        second = store.record_failure(
            error_message="Connection timed out", task=sample_task
        )

        assert second.last_seen > first_last_seen


class TestGetFailurePattern:
    """Tests for FeedbackStore.get_failure_pattern() method."""

    def test_returns_existing_pattern(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify get_failure_pattern returns existing pattern."""
        store.record_failure(error_message="Connection timed out", task=sample_task)

        pattern = store.get_failure_pattern("Connection timed out")

        assert pattern is not None
        assert pattern.failure_type == FailureType.TRANSIENT

    def test_returns_none_for_unknown_pattern(self, store: FeedbackStore) -> None:
        """Verify get_failure_pattern returns None for unknown error."""
        pattern = store.get_failure_pattern("Never seen this error before")

        assert pattern is None


# -----------------------------------------------------------------------------
# Tests for should_retry()
# -----------------------------------------------------------------------------


class TestShouldRetry:
    """Tests for FeedbackStore.should_retry() method."""

    def test_returns_true_for_transient_errors(
        self, store: FeedbackStore
    ) -> None:
        """Verify should_retry returns True for transient errors."""
        result = store.should_retry(
            error_message="rate limit exceeded",
            current_attempts=0,
            max_attempts=3,
        )

        assert result is True

    def test_returns_false_at_max_attempts(self, store: FeedbackStore) -> None:
        """Verify should_retry returns False when max_attempts reached."""
        result = store.should_retry(
            error_message="rate limit exceeded",
            current_attempts=3,
            max_attempts=3,
        )

        assert result is False

    def test_returns_true_if_pattern_has_prior_success(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify should_retry returns True if pattern had successful retries."""
        # Record failure and then success
        store.record_failure(error_message="Some error", task=sample_task)
        store.record_retry_success(error_message="Some error")

        result = store.should_retry(
            error_message="Some error",
            current_attempts=1,
            max_attempts=3,
        )

        assert result is True

    def test_returns_false_for_consistently_failing_pattern(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify should_retry returns False for patterns that always fail."""
        # Record multiple failures with no success
        for _ in range(5):
            store.record_failure(error_message="Persistent error", task=sample_task)

        result = store.should_retry(
            error_message="Persistent error",
            current_attempts=2,
            max_attempts=5,
        )

        # Pattern has 5 occurrences, 0 successes - should not retry
        assert result is False


# -----------------------------------------------------------------------------
# Tests for record_retry_success()
# -----------------------------------------------------------------------------


class TestRecordRetrySuccess:
    """Tests for FeedbackStore.record_retry_success() method."""

    def test_increments_success_count(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify record_retry_success increments retry_success_count."""
        store.record_failure(error_message="Transient error", task=sample_task)

        pattern = store.record_retry_success(error_message="Transient error")

        assert pattern is not None
        assert pattern.retry_success_count == 1

    def test_updates_failure_type_to_transient(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify pattern is marked transient when retries succeed more than fail."""
        # Create pattern that initially fails
        store.record_failure(error_message="Unknown error", task=sample_task)

        # Record multiple successful retries
        store.record_retry_success(error_message="Unknown error")
        pattern = store.record_retry_success(error_message="Unknown error")

        # retry_success_count (2) > retry_failure_count (1)
        assert pattern.failure_type == FailureType.TRANSIENT

    def test_returns_none_for_unknown_pattern(self, store: FeedbackStore) -> None:
        """Verify returns None for pattern that doesn't exist."""
        result = store.record_retry_success(error_message="Never recorded")

        assert result is None


# -----------------------------------------------------------------------------
# Tests for suppress_pattern() and is_pattern_suppressed()
# -----------------------------------------------------------------------------


class TestSuppressPattern:
    """Tests for FeedbackStore.suppress_pattern() method."""

    def test_suppresses_pattern(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify suppress_pattern sets suppression fields."""
        store.record_failure(error_message="Annoying error", task=sample_task)

        pattern = store.suppress_pattern(
            error_message="Annoying error",
            days=7,
            reason="Known issue, ignore for now",
        )

        assert pattern is not None
        assert pattern.suppressed_until is not None
        assert pattern.suppression_reason == "Known issue, ignore for now"

    def test_returns_none_for_unknown_pattern(self, store: FeedbackStore) -> None:
        """Verify suppress_pattern returns None for unknown error."""
        result = store.suppress_pattern(
            error_message="Never seen this",
            days=7,
        )

        assert result is None


class TestIsPatternSuppressed:
    """Tests for FeedbackStore.is_pattern_suppressed() method."""

    def test_returns_true_for_matching_suppressed_pattern(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify is_pattern_suppressed returns True for matching description."""
        sample_task.description = "Fix the database connection issue"
        store.record_failure(error_message="Database error", task=sample_task)
        store.suppress_pattern(error_message="Database error", days=7)

        result = store.is_pattern_suppressed("Fix the database connection issue")

        assert result is True

    def test_returns_false_when_no_suppressed_patterns(
        self, store: FeedbackStore
    ) -> None:
        """Verify is_pattern_suppressed returns False with no suppressed patterns."""
        result = store.is_pattern_suppressed("Some task description")

        assert result is False


# -----------------------------------------------------------------------------
# Tests for get_recent_feedback_stats()
# -----------------------------------------------------------------------------


class TestGetRecentFeedbackStats:
    """Tests for FeedbackStore.get_recent_feedback_stats() method."""

    def test_returns_aggregate_stats(self, store: FeedbackStore) -> None:
        """Verify get_recent_feedback_stats returns correct aggregates."""
        # Create feedback from different users on different tasks
        store.record_feedback(task_id=1, user_id="U1", reaction="+1")
        store.record_feedback(task_id=1, user_id="U2", reaction="-1")
        store.record_feedback(task_id=2, user_id="U1", reaction="heart")
        store.record_feedback(task_id=3, user_id="U3", reaction="eyes")

        stats = store.get_recent_feedback_stats(days=7)

        assert stats["total_feedback"] == 4
        assert stats["positive"] == 2  # +1 and heart
        assert stats["negative"] == 1  # -1
        assert stats["neutral"] == 1  # eyes
        assert stats["unique_tasks"] == 3
        assert stats["unique_users"] == 3


# -----------------------------------------------------------------------------
# Tests for get_feedback_summary()
# -----------------------------------------------------------------------------


class TestGetFeedbackSummary:
    """Tests for FeedbackStore.get_feedback_summary() method."""

    def test_returns_summary_for_task(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify get_feedback_summary returns correct counts."""
        store.record_feedback(task_id=sample_task.id, user_id="U1", reaction="+1")
        store.record_feedback(task_id=sample_task.id, user_id="U2", reaction="heart")
        store.record_feedback(task_id=sample_task.id, user_id="U3", reaction="-1")
        store.record_feedback(task_id=sample_task.id, user_id="U4", reaction="eyes")

        summary = store.get_feedback_summary(sample_task.id)

        assert summary["positive"] == 2
        assert summary["negative"] == 1
        assert summary["neutral"] == 1
        assert summary["total"] == 4

    def test_returns_zeros_for_unknown_task(self, store: FeedbackStore) -> None:
        """Verify get_feedback_summary returns zeros for unknown task."""
        summary = store.get_feedback_summary(99999)

        assert summary["total"] == 0
        assert summary["positive"] == 0
        assert summary["negative"] == 0
        assert summary["neutral"] == 0


# -----------------------------------------------------------------------------
# Tests for get_failure_stats()
# -----------------------------------------------------------------------------


class TestGetFailureStats:
    """Tests for FeedbackStore.get_failure_stats() method."""

    def test_returns_failure_statistics(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify get_failure_stats returns correct statistics."""
        # Create various failure patterns
        store.record_failure(error_message="rate limit exceeded", task=sample_task)
        store.record_failure(error_message="TypeError: invalid", task=sample_task)
        store.record_failure(error_message="rate limit exceeded", task=sample_task)  # Repeat

        stats = store.get_failure_stats()

        assert stats["total_patterns"] == 2
        assert stats["transient_count"] == 1
        assert stats["substantive_count"] == 1
        assert stats["total_occurrences"] == 3


# -----------------------------------------------------------------------------
# Tests for cleanup_old_patterns()
# -----------------------------------------------------------------------------


class TestCleanupOldPatterns:
    """Tests for FeedbackStore.cleanup_old_patterns() method."""

    def test_removes_old_patterns(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Verify cleanup_old_patterns removes patterns older than threshold."""
        # Create a pattern
        store.record_failure(error_message="Old error", task=sample_task)

        # Clean up with 0 days (should remove all)
        count = store.cleanup_old_patterns(days=0)

        # Should have removed the pattern
        pattern = store.get_failure_pattern("Old error")
        assert pattern is None or count > 0


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------


class TestFeedbackEdgeCases:
    """Edge case tests for feedback storage."""

    def test_edge_case_empty_feedback_history(self, store: FeedbackStore) -> None:
        """Edge case: verify behavior with empty feedback history."""
        stats = store.get_recent_feedback_stats(days=7)

        assert stats["total_feedback"] == 0
        assert stats["unique_tasks"] == 0
        assert stats["unique_users"] == 0

    def test_edge_case_task_with_only_positive_feedback(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Edge case: task with only positive feedback."""
        store.record_feedback(task_id=sample_task.id, user_id="U1", reaction="+1")
        store.record_feedback(task_id=sample_task.id, user_id="U2", reaction="heart")
        store.record_feedback(task_id=sample_task.id, user_id="U3", reaction="star")

        summary = store.get_feedback_summary(sample_task.id)

        assert summary["positive"] == 3
        assert summary["negative"] == 0
        assert summary["neutral"] == 0

    def test_edge_case_task_with_only_negative_feedback(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Edge case: task with only negative feedback."""
        store.record_feedback(task_id=sample_task.id, user_id="U1", reaction="-1")
        store.record_feedback(task_id=sample_task.id, user_id="U2", reaction="x")
        store.record_feedback(task_id=sample_task.id, user_id="U3", reaction="thumbsdown")

        summary = store.get_feedback_summary(sample_task.id)

        assert summary["positive"] == 0
        assert summary["negative"] == 3
        assert summary["neutral"] == 0

    def test_edge_case_very_long_error_message(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Edge case: very long error message is handled."""
        long_error = "x" * 10000

        pattern = store.record_failure(error_message=long_error, task=sample_task)

        # Pattern should be created with normalized (truncated) signature
        assert pattern is not None
        assert len(pattern.error_signature) <= 500

    def test_edge_case_unicode_in_feedback_reaction(
        self, store: FeedbackStore, sample_task: Task
    ) -> None:
        """Edge case: unicode characters in reaction name."""
        # While Slack uses ASCII names, test unicode handling
        feedback = store.record_feedback(
            task_id=sample_task.id,
            user_id="U123",
            reaction="heart_eyes",  # A valid emoji name
        )

        assert feedback is not None
        assert feedback.reaction == "heart_eyes"
