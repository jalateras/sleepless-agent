"""Feedback storage for task outcome learning."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from sleepless_agent.core.models import (
    FailurePattern,
    FailureType,
    FeedbackType,
    Task,
    TaskFeedback,
)
from sleepless_agent.monitoring.logging import get_logger
from sleepless_agent.storage.sqlite import SQLiteStore

logger = get_logger(__name__)

# Mapping of Slack emoji names to feedback types
POSITIVE_REACTIONS = {
    "+1", "thumbsup", "white_check_mark", "heavy_check_mark",
    "star", "star2", "tada", "rocket", "fire", "100",
    "heart", "clap", "raised_hands", "muscle",
}

NEGATIVE_REACTIONS = {
    "-1", "thumbsdown", "x", "negative_squared_cross_mark",
    "no_entry", "no_entry_sign", "warning", "disappointed",
    "confused", "face_with_rolling_eyes",
}

# Patterns indicating transient failures (worth retrying)
TRANSIENT_ERROR_PATTERNS = [
    "rate limit",
    "rate_limit",
    "ratelimit",
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "connection error",
    "network error",
    "temporary failure",
    "service unavailable",
    "503",
    "502",
    "429",
    "resource exhausted",
    "quota exceeded",
    "try again",
    "retry",
]


def classify_reaction(emoji_name: str) -> FeedbackType:
    """Classify a Slack emoji reaction into a feedback type."""
    normalized = emoji_name.lower().replace(":", "").strip()
    if normalized in POSITIVE_REACTIONS:
        return FeedbackType.POSITIVE
    elif normalized in NEGATIVE_REACTIONS:
        return FeedbackType.NEGATIVE
    else:
        return FeedbackType.NEUTRAL


def classify_failure(error_message: str) -> FailureType:
    """Classify an error message into a failure type."""
    if not error_message:
        return FailureType.UNKNOWN

    error_lower = error_message.lower()
    for pattern in TRANSIENT_ERROR_PATTERNS:
        if pattern in error_lower:
            return FailureType.TRANSIENT

    return FailureType.SUBSTANTIVE


def normalize_error(error_message: str) -> str:
    """Normalize error message for pattern matching.

    Removes variable parts like timestamps, IDs, and paths to create
    a consistent signature for similar errors.
    """
    if not error_message:
        return ""

    import re

    normalized = error_message

    # Remove timestamps
    normalized = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '<TIMESTAMP>', normalized)

    # Remove UUIDs
    normalized = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '<UUID>', normalized)

    # Remove file paths
    normalized = re.sub(r'/[\w/.-]+', '<PATH>', normalized)

    # Remove line numbers
    normalized = re.sub(r'line \d+', 'line <N>', normalized)

    # Remove numeric IDs
    normalized = re.sub(r'\b\d{5,}\b', '<ID>', normalized)

    # Truncate to reasonable length
    return normalized[:500]


def hash_error(normalized_error: str) -> str:
    """Create a hash of the normalized error for deduplication."""
    return hashlib.sha256(normalized_error.encode()).hexdigest()


class FeedbackStore(SQLiteStore):
    """Store and query task feedback for learning.

    Handles both user feedback (reactions) and failure patterns
    to enable the agent to improve over time.
    """

    def __init__(self, db_path: str):
        """Initialize feedback store with database."""
        super().__init__(db_path)

    # -------------------------------------------------------------------------
    # Task Feedback Methods
    # -------------------------------------------------------------------------

    def record_feedback(
        self,
        task_id: int,
        user_id: str,
        reaction: str,
        *,
        message_ts: Optional[str] = None,
        channel_id: Optional[str] = None,
        task: Optional[Task] = None,
        generation_source: Optional[str] = None,
    ) -> TaskFeedback:
        """Record user feedback on a task outcome.

        Args:
            task_id: The task being rated
            user_id: Slack user ID providing feedback
            reaction: The emoji reaction name
            message_ts: Slack message timestamp (for deduplication)
            channel_id: Slack channel ID
            task: Optional task object for context snapshot
            generation_source: If auto-generated, which prompt was used

        Returns:
            The created TaskFeedback record
        """
        feedback_type = classify_reaction(reaction)

        def _op(session: Session) -> TaskFeedback:
            # Check for duplicate (same user, same task, same reaction)
            existing = session.query(TaskFeedback).filter(
                TaskFeedback.task_id == task_id,
                TaskFeedback.user_id == user_id,
                TaskFeedback.reaction == reaction,
            ).first()

            if existing:
                logger.debug(f"Duplicate feedback ignored: task={task_id}, user={user_id}")
                return existing

            feedback = TaskFeedback(
                task_id=task_id,
                user_id=user_id,
                feedback_type=feedback_type,
                reaction=reaction,
                message_ts=message_ts,
                channel_id=channel_id,
                task_priority=task.priority.value if task else None,
                task_type=task.task_type.value if task and task.task_type else None,
                generation_source=generation_source,
            )
            session.add(feedback)
            session.flush()
            return feedback

        feedback = self._run_write(_op)
        logger.info(
            f"Recorded {feedback_type.value} feedback for task {task_id} "
            f"(reaction={reaction}, user={user_id})"
        )
        return feedback

    def get_feedback_for_task(self, task_id: int) -> List[TaskFeedback]:
        """Get all feedback for a specific task."""

        def _op(session: Session) -> List[TaskFeedback]:
            return session.query(TaskFeedback).filter(
                TaskFeedback.task_id == task_id
            ).order_by(TaskFeedback.created_at.desc()).all()

        return self._run_read(_op)

    def get_feedback_summary(self, task_id: int) -> Dict[str, int]:
        """Get feedback summary for a task (counts by type)."""

        def _op(session: Session) -> Dict[str, int]:
            feedbacks = session.query(TaskFeedback).filter(
                TaskFeedback.task_id == task_id
            ).all()

            return {
                "positive": sum(1 for f in feedbacks if f.feedback_type == FeedbackType.POSITIVE),
                "negative": sum(1 for f in feedbacks if f.feedback_type == FeedbackType.NEGATIVE),
                "neutral": sum(1 for f in feedbacks if f.feedback_type == FeedbackType.NEUTRAL),
                "total": len(feedbacks),
            }

        return self._run_read(_op)

    def get_feedback_weights_by_source(
        self,
        days: int = 30,
        min_feedback_count: int = 3,
    ) -> Dict[str, float]:
        """Calculate feedback-weighted scores for each generation source.

        Used by auto-generator to prefer prompts that produce valuable tasks.

        Args:
            days: Look back period for feedback
            min_feedback_count: Minimum feedback to include source in weights

        Returns:
            Dict mapping generation_source to weight (0.1 to 2.0)
        """

        def _op(session: Session) -> Dict[str, float]:
            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)

            feedbacks = session.query(TaskFeedback).filter(
                TaskFeedback.created_at >= cutoff,
                TaskFeedback.generation_source.isnot(None),
            ).all()

            # Group by source
            source_scores: Dict[str, Dict[str, int]] = {}
            for fb in feedbacks:
                source = fb.generation_source
                if source not in source_scores:
                    source_scores[source] = {"positive": 0, "negative": 0, "total": 0}

                source_scores[source]["total"] += 1
                if fb.feedback_type == FeedbackType.POSITIVE:
                    source_scores[source]["positive"] += 1
                elif fb.feedback_type == FeedbackType.NEGATIVE:
                    source_scores[source]["negative"] += 1

            # Calculate weights
            weights: Dict[str, float] = {}
            for source, counts in source_scores.items():
                if counts["total"] < min_feedback_count:
                    continue

                # Score: positive_ratio - negative_ratio, mapped to [0.1, 2.0]
                positive_ratio = counts["positive"] / counts["total"]
                negative_ratio = counts["negative"] / counts["total"]
                raw_score = positive_ratio - negative_ratio  # Range: -1 to 1

                # Map to [0.1, 2.0]: score 0 -> 1.0, score 1 -> 2.0, score -1 -> 0.1
                weight = max(0.1, min(2.0, 1.0 + raw_score))
                weights[source] = round(weight, 2)

            return weights

        return self._run_read(_op)

    def get_recent_feedback_stats(self, days: int = 7) -> Dict[str, int]:
        """Get aggregate feedback statistics for recent period."""

        def _op(session: Session) -> Dict[str, int]:
            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)

            feedbacks = session.query(TaskFeedback).filter(
                TaskFeedback.created_at >= cutoff
            ).all()

            return {
                "total_feedback": len(feedbacks),
                "positive": sum(1 for f in feedbacks if f.feedback_type == FeedbackType.POSITIVE),
                "negative": sum(1 for f in feedbacks if f.feedback_type == FeedbackType.NEGATIVE),
                "neutral": sum(1 for f in feedbacks if f.feedback_type == FeedbackType.NEUTRAL),
                "unique_tasks": len(set(f.task_id for f in feedbacks)),
                "unique_users": len(set(f.user_id for f in feedbacks)),
            }

        return self._run_read(_op)

    # -------------------------------------------------------------------------
    # Failure Pattern Methods
    # -------------------------------------------------------------------------

    def record_failure(
        self,
        error_message: str,
        *,
        task: Optional[Task] = None,
    ) -> FailurePattern:
        """Record or update a failure pattern.

        Args:
            error_message: The error message from the failed task
            task: Optional task for context

        Returns:
            The created or updated FailurePattern record
        """
        normalized = normalize_error(error_message)
        pattern_hash = hash_error(normalized)
        failure_type = classify_failure(error_message)

        def _op(session: Session) -> FailurePattern:
            existing = session.query(FailurePattern).filter(
                FailurePattern.pattern_hash == pattern_hash
            ).first()

            now = datetime.now(timezone.utc).replace(tzinfo=None)

            if existing:
                existing.occurrences += 1
                existing.last_seen = now
                existing.retry_failure_count += 1
                return existing

            pattern = FailurePattern(
                pattern_hash=pattern_hash,
                error_signature=normalized,
                failure_type=failure_type,
                first_seen=now,
                last_seen=now,
                sample_task_description=task.description[:500] if task else None,
                sample_task_id=task.id if task else None,
            )
            session.add(pattern)
            session.flush()
            return pattern

        pattern = self._run_write(_op)
        logger.info(
            f"Recorded failure pattern: type={failure_type.value}, "
            f"occurrences={pattern.occurrences}, hash={pattern_hash[:8]}..."
        )
        return pattern

    def record_retry_success(self, error_message: str) -> Optional[FailurePattern]:
        """Record that a retry succeeded for this error pattern.

        This helps the system learn which failures are worth retrying.
        """
        normalized = normalize_error(error_message)
        pattern_hash = hash_error(normalized)

        def _op(session: Session) -> Optional[FailurePattern]:
            pattern = session.query(FailurePattern).filter(
                FailurePattern.pattern_hash == pattern_hash
            ).first()

            if pattern:
                pattern.retry_success_count += 1
                # If retries are now succeeding more than failing, mark as transient
                if pattern.retry_success_count > pattern.retry_failure_count:
                    pattern.failure_type = FailureType.TRANSIENT

            return pattern

        return self._run_write(_op)

    def get_failure_pattern(self, error_message: str) -> Optional[FailurePattern]:
        """Get the failure pattern for an error message if it exists."""
        normalized = normalize_error(error_message)
        pattern_hash = hash_error(normalized)

        def _op(session: Session) -> Optional[FailurePattern]:
            return session.query(FailurePattern).filter(
                FailurePattern.pattern_hash == pattern_hash
            ).first()

        return self._run_read(_op)

    def should_retry(self, error_message: str, current_attempts: int, max_attempts: int = 3) -> bool:
        """Determine if a failed task should be retried based on error pattern.

        Args:
            error_message: The error from the failed task
            current_attempts: How many attempts have been made
            max_attempts: Maximum allowed attempts

        Returns:
            True if task should be retried
        """
        if current_attempts >= max_attempts:
            return False

        failure_type = classify_failure(error_message)

        # Transient failures are always worth retrying
        if failure_type == FailureType.TRANSIENT:
            return True

        # Check if we have learned this pattern is actually transient
        pattern = self.get_failure_pattern(error_message)
        if pattern:
            # If retries have succeeded before, try again
            if pattern.retry_success_count > 0:
                return True
            # If this pattern consistently fails, don't retry
            if pattern.occurrences >= 3 and pattern.retry_success_count == 0:
                return False

        # For unknown patterns, retry once
        return current_attempts < 2

    def suppress_pattern(
        self,
        error_message: str,
        days: int = 7,
        reason: Optional[str] = None,
    ) -> Optional[FailurePattern]:
        """Suppress auto-generation of tasks matching this failure pattern.

        Args:
            error_message: The error to suppress
            days: How long to suppress (default 7 days)
            reason: Optional reason for suppression

        Returns:
            The updated pattern, or None if not found
        """
        normalized = normalize_error(error_message)
        pattern_hash = hash_error(normalized)

        def _op(session: Session) -> Optional[FailurePattern]:
            pattern = session.query(FailurePattern).filter(
                FailurePattern.pattern_hash == pattern_hash
            ).first()

            if pattern:
                pattern.suppressed_until = (
                    datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=days)
                )
                pattern.suppression_reason = reason

            return pattern

        return self._run_write(_op)

    def is_pattern_suppressed(self, task_description: str) -> bool:
        """Check if a task description matches any suppressed failure patterns.

        This is a simple heuristic check - more sophisticated matching
        could be implemented with embeddings or keyword extraction.
        """

        def _op(session: Session) -> bool:
            now = datetime.now(timezone.utc).replace(tzinfo=None)

            suppressed = session.query(FailurePattern).filter(
                FailurePattern.suppressed_until.isnot(None),
                FailurePattern.suppressed_until > now,
            ).all()

            # Simple keyword matching against sample task descriptions
            task_lower = task_description.lower()
            for pattern in suppressed:
                if pattern.sample_task_description:
                    sample_lower = pattern.sample_task_description.lower()
                    # Check for significant word overlap
                    sample_words = set(sample_lower.split())
                    task_words = set(task_lower.split())
                    common = sample_words & task_words
                    # If more than 50% of sample words appear in task, suppress
                    if len(sample_words) > 0 and len(common) / len(sample_words) > 0.5:
                        return True

            return False

        return self._run_read(_op)

    def get_failure_stats(self) -> Dict[str, any]:
        """Get aggregate statistics about failure patterns."""

        def _op(session: Session) -> Dict[str, any]:
            patterns = session.query(FailurePattern).all()

            now = datetime.now(timezone.utc).replace(tzinfo=None)

            return {
                "total_patterns": len(patterns),
                "transient_count": sum(1 for p in patterns if p.failure_type == FailureType.TRANSIENT),
                "substantive_count": sum(1 for p in patterns if p.failure_type == FailureType.SUBSTANTIVE),
                "total_occurrences": sum(p.occurrences for p in patterns),
                "currently_suppressed": sum(
                    1 for p in patterns
                    if p.suppressed_until and p.suppressed_until > now
                ),
                "retry_success_rate": (
                    sum(p.retry_success_count for p in patterns) /
                    max(1, sum(p.retry_success_count + p.retry_failure_count for p in patterns))
                ),
            }

        return self._run_read(_op)

    def cleanup_old_patterns(self, days: int = 90) -> int:
        """Remove failure patterns not seen in the specified period.

        Returns the count of deleted patterns.
        """

        def _op(session: Session) -> int:
            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)

            old_patterns = session.query(FailurePattern).filter(
                FailurePattern.last_seen < cutoff
            ).all()

            count = len(old_patterns)
            for pattern in old_patterns:
                session.delete(pattern)

            return count

        count = self._run_write(_op)
        if count:
            logger.info(f"Cleaned up {count} old failure patterns (>{days} days)")
        return count
