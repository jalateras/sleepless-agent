"""SQLAlchemy models for task queue and results"""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, DateTime, Enum as SQLEnum, Index, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

Base = declarative_base()


class TaskPriority(str, Enum):
    """Task priority levels"""
    THOUGHT = "thought"  # Low priority, experimental
    SERIOUS = "serious"  # High priority, needs completion
    GENERATED = "generated"  # Auto-generated backlog filler


class TaskStatus(str, Enum):
    """Task status states"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Task type: NEW (build from scratch) vs REFINE (improve existing code)"""
    NEW = "new"  # Create new functionality in empty workspace
    REFINE = "refine"  # Improve existing code (workspace pre-populated with source)


class Task(Base):
    """Task queue model"""
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    description = Column(Text, nullable=False)
    priority = Column(
        SQLEnum(
            TaskPriority,
            native_enum=False,
            validate_strings=True,
            create_constraint=False,
        ),
        default=TaskPriority.THOUGHT,
        nullable=False,
    )
    task_type = Column(
        SQLEnum(
            TaskType,
            native_enum=False,
            validate_strings=True,
            create_constraint=False,
        ),
        default=TaskType.NEW,
        nullable=True,  # Nullable for backwards compatibility
    )
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, nullable=False)

    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True)

    # Execution details
    attempt_count = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)
    result_id = Column(Integer, nullable=True)  # Reference to Result

    # Metadata
    context = Column(Text, nullable=True)  # JSON with additional context
    assigned_to = Column(String(255), nullable=True)  # Slack user ID
    slack_thread_ts = Column(String(255), nullable=True)  # Slack thread timestamp for updates

    # Project grouping - tasks with same project_id share workspace and context
    project_id = Column(String(255), nullable=True)  # Project identifier for context sharing
    project_name = Column(String(255), nullable=True)  # Human-readable project name

    def __repr__(self):
        return f"<Task(id={self.id}, type={self.task_type}, priority={self.priority}, status={self.status})>"

    # Define indexes for query optimization
    __table_args__ = (
        # Single-column indexes for frequently filtered columns
        Index('ix_task_status', 'status'),
        Index('ix_task_project_id', 'project_id'),
        Index('ix_task_created_at', 'created_at'),
        Index('ix_task_type', 'task_type'),

        # Composite indexes for common query patterns
        # Optimizes get_projects() and project-specific queries
        Index('ix_task_project_status', 'project_id', 'status'),

        # Optimizes get_pending_tasks() with status filter + created_at ordering
        Index('ix_task_status_created', 'status', 'created_at'),

        # Optimizes filtering by task type and status
        Index('ix_task_type_status', 'task_type', 'status'),
    )


class Result(Base):
    """Stores results from completed tasks"""
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, nullable=False)

    # Results
    output = Column(Text, nullable=True)  # Main output/response
    files_modified = Column(Text, nullable=True)  # JSON list of modified files
    commands_executed = Column(Text, nullable=True)  # JSON list of executed commands

    # Git integration
    git_commit_sha = Column(String(40), nullable=True)
    git_pr_url = Column(String(512), nullable=True)
    git_branch = Column(String(255), nullable=True)

    # Workspace
    workspace_path = Column(String(512), nullable=True)  # Path to isolated task workspace

    # Metadata
    processing_time_seconds = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Result(id={self.id}, task_id={self.task_id})>"


class UsageMetric(Base):
    """Track API usage and costs for budget management"""
    __tablename__ = "usage_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, nullable=False)

    # API usage details
    total_cost_usd = Column(Text, nullable=True)  # Stored as text to preserve precision
    duration_ms = Column(Integer, nullable=True)  # Total duration in milliseconds
    duration_api_ms = Column(Integer, nullable=True)  # API call duration
    num_turns = Column(Integer, nullable=True)  # Number of conversation turns

    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Project tracking
    project_id = Column(String(255), nullable=True)  # Link to project for aggregation

    def __repr__(self):
        return f"<UsageMetric(id={self.id}, task_id={self.task_id}, cost=${self.total_cost_usd})>"


class GenerationHistory(Base):
    """Track auto-generated tasks and their originating prompt archetype."""
    __tablename__ = "generation_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, nullable=False)  # Reference to created Task
    source = Column(String(50), nullable=False)  # Prompt name or generation source label
    usage_percent_at_generation = Column(Integer, nullable=False)  # Budget usage % when generated
    source_metadata = Column(Text, nullable=True)  # JSON with source-specific info
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<GenerationHistory(task_id={self.task_id}, source={self.source})>"


class TaskPool(Base):
    """Predefined pool of tasks for auto-generation"""
    __tablename__ = "task_pool"

    id = Column(Integer, primary_key=True, autoincrement=True)
    description = Column(Text, nullable=False)
    priority = Column(
        SQLEnum(
            TaskPriority,
            native_enum=False,
            validate_strings=True,
            create_constraint=False,
        ),
        default=TaskPriority.THOUGHT,
        nullable=False,
    )
    category = Column(String(100), nullable=True)
    used = Column(Integer, default=0, nullable=False)
    project_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<TaskPool(id={self.id}, priority={self.priority}, category={self.category})>"


class FeedbackType(str, Enum):
    """Types of user feedback on task outcomes"""
    POSITIVE = "positive"  # thumbs_up, white_check_mark
    NEGATIVE = "negative"  # thumbs_down, x
    NEUTRAL = "neutral"  # other reactions


class TaskFeedback(Base):
    """Store user feedback on task outcomes for learning.

    Captures Slack reactions on task completion messages to learn
    which task types and patterns are valuable to users. This feedback
    is used to weight auto-generation prompt selection.
    """
    __tablename__ = "task_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, nullable=False)  # Reference to Task
    user_id = Column(String(100), nullable=False)  # Slack user ID
    feedback_type = Column(
        SQLEnum(
            FeedbackType,
            native_enum=False,
            validate_strings=True,
            create_constraint=False,
        ),
        nullable=False,
    )
    reaction = Column(String(50), nullable=False)  # Original emoji name

    # Context snapshot for analysis
    task_priority = Column(String(50), nullable=True)  # Priority at feedback time
    task_type = Column(String(50), nullable=True)  # NEW or REFINE
    generation_source = Column(String(100), nullable=True)  # If auto-generated, which prompt

    # Slack context
    message_ts = Column(String(50), nullable=True)  # Message that was reacted to
    channel_id = Column(String(50), nullable=True)  # Channel where reaction occurred

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index('ix_feedback_task_id', 'task_id'),
        Index('ix_feedback_user_id', 'user_id'),
        Index('ix_feedback_type', 'feedback_type'),
        Index('ix_feedback_created', 'created_at'),
        # Composite for querying feedback patterns
        Index('ix_feedback_type_source', 'feedback_type', 'generation_source'),
    )

    def __repr__(self):
        return f"<TaskFeedback(task_id={self.task_id}, type={self.feedback_type}, reaction={self.reaction})>"


class FailureType(str, Enum):
    """Classification of task failure types"""
    TRANSIENT = "transient"  # Rate limits, timeouts, connection errors - worth retrying
    SUBSTANTIVE = "substantive"  # Wrong approach, missing deps - needs prompt refinement
    UNKNOWN = "unknown"  # Couldn't classify


class FailurePattern(Base):
    """Track failure patterns for learning and retry suppression.

    When tasks fail repeatedly with similar errors, this model stores
    the pattern to suppress auto-generation of similar tasks and to
    improve retry strategies.
    """
    __tablename__ = "failure_patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern_hash = Column(String(64), unique=True, nullable=False)  # SHA256 of normalized error
    error_signature = Column(String(500), nullable=False)  # Normalized error for display
    failure_type = Column(
        SQLEnum(
            FailureType,
            native_enum=False,
            validate_strings=True,
            create_constraint=False,
        ),
        default=FailureType.UNKNOWN,
        nullable=False,
    )

    # Occurrence tracking
    occurrences = Column(Integer, default=1, nullable=False)
    first_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Suppression control
    suppressed_until = Column(DateTime, nullable=True)  # If set, don't auto-generate similar
    suppression_reason = Column(String(255), nullable=True)

    # Related task context
    sample_task_description = Column(Text, nullable=True)  # Example task that triggered this
    sample_task_id = Column(Integer, nullable=True)

    # Learning metadata
    retry_success_count = Column(Integer, default=0, nullable=False)  # Times retry succeeded
    retry_failure_count = Column(Integer, default=0, nullable=False)  # Times retry failed again

    __table_args__ = (
        Index('ix_failure_pattern_hash', 'pattern_hash'),
        Index('ix_failure_type', 'failure_type'),
        Index('ix_failure_last_seen', 'last_seen'),
        Index('ix_failure_suppressed', 'suppressed_until'),
    )

    def __repr__(self):
        return f"<FailurePattern(hash={self.pattern_hash[:8]}..., type={self.failure_type}, count={self.occurrences})>"


def init_db(db_path: str) -> Session:
    """Initialize database and return session"""
    engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
    Base.metadata.create_all(engine)
    return engine
