"""Task queue management"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import case
from sqlalchemy.orm import Session

from sleepless_agent.monitoring.logging import get_logger
from sleepless_agent.storage.sqlite import SQLiteStore

from .models import Task, TaskPriority, TaskStatus

logger = get_logger(__name__)


class TaskQueue(SQLiteStore):
    """Task queue manager"""

    def __init__(self, db_path: str):
        """Initialize task queue with database"""
        super().__init__(db_path)

    def get_pool_status(self) -> dict:
        """Get connection pool status for monitoring.

        Returns:
            Dictionary with pool statistics including size and connections in use.
        """
        pool = self.engine.pool
        return {
            "pool_class": pool.__class__.__name__,
            "size": getattr(pool, "size", lambda: "N/A")() if callable(getattr(pool, "size", None)) else "N/A",
            "checked_in": getattr(pool, "checkedin", lambda: 0)() if callable(getattr(pool, "checkedin", None)) else 0,
            "checked_out": getattr(pool, "checkedout", lambda: 0)() if callable(getattr(pool, "checkedout", None)) else 0,
            "overflow": getattr(pool, "overflow", lambda: 0)() if callable(getattr(pool, "overflow", None)) else 0,
        }

    def add_task(
        self,
        description: str,
        priority: TaskPriority = TaskPriority.THOUGHT,
        context: Optional[dict] = None,
        slack_user_id: Optional[str] = None,
        slack_thread_ts: Optional[str] = None,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> Task:
        """Add new task to queue"""

        def _op(session: Session) -> Task:
            task = Task(
                description=description,
                priority=priority,
                context=json.dumps(context) if context else None,
                assigned_to=slack_user_id,
                slack_thread_ts=slack_thread_ts,
                project_id=project_id,
                project_name=project_name,
            )
            session.add(task)
            session.flush()
            return task

        task = self._run_write(_op)
        project_info = f" [Project: {project_name}]" if project_name else ""
        logger.info(f"Added task {task.id}: {description[:50]}...{project_info}")
        return task

    def get_task(self, task_id: int) -> Optional[Task]:
        """Get task by ID"""

        def _op(session: Session) -> Optional[Task]:
            return session.query(Task).filter(Task.id == task_id).first()

        return self._run_read(_op)

    def get_pending_tasks(self, limit: int = 10) -> List[Task]:
        """Get pending tasks sorted by priority"""

        def _op(session: Session) -> List[Task]:
            priority_order = case(
                (Task.priority == TaskPriority.SERIOUS.value, 0),
                (Task.priority == TaskPriority.THOUGHT.value, 1),
                else_=2,
            )
            return (
                session.query(Task)
                    .filter(Task.status == TaskStatus.PENDING)
                    .order_by(priority_order, Task.created_at)
                    .limit(limit)
                    .all()
            )

        return self._run_read(_op)

    def get_in_progress_tasks(self) -> List[Task]:
        """Get all in-progress tasks"""

        def _op(session: Session) -> List[Task]:
            return session.query(Task).filter(Task.status == TaskStatus.IN_PROGRESS).all()

        return self._run_read(_op)

    def mark_in_progress(self, task_id: int) -> Optional[Task]:
        """Mark task as in progress"""

        def _op(session: Session) -> Optional[Task]:
            task = session.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
                task.attempt_count += 1
            return task

        task = self._run_write(_op)
        if task:
            logger.info(f"Task {task_id} marked as in_progress")
        return task

    def mark_completed(self, task_id: int, result_id: Optional[int] = None) -> Optional[Task]:
        """Mark task as completed"""

        def _op(session: Session) -> Optional[Task]:
            task = session.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
                task.result_id = result_id
            return task

        task = self._run_write(_op)
        if task:
            logger.info(f"Task {task_id} marked as completed")
        return task

    def mark_failed(self, task_id: int, error_message: str) -> Optional[Task]:
        """Mark task as failed"""

        def _op(session: Session) -> Optional[Task]:
            task = session.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = TaskStatus.FAILED
                task.error_message = error_message
                if not task.completed_at:
                    task.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
            return task

        task = self._run_write(_op)
        if task:
            logger.error(f"Task {task_id} marked as failed: {error_message}")
        return task

    def increment_attempt_count(self, task_id: int) -> Optional[Task]:
        """Increment the attempt count for a task (used for retry tracking)."""

        def _op(session: Session) -> Optional[Task]:
            task = session.query(Task).filter(Task.id == task_id).first()
            if task:
                task.attempt_count = (task.attempt_count or 0) + 1
            return task

        task = self._run_write(_op)
        if task:
            logger.debug(f"Task {task_id} attempt count incremented to {task.attempt_count}")
        return task

    def cancel_task(self, task_id: int) -> Optional[Task]:
        """Cancel pending task (soft delete)"""

        def _op(session: Session) -> Optional[Task]:
            task = session.query(Task).filter(Task.id == task_id).first()
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                task.deleted_at = datetime.now(timezone.utc).replace(tzinfo=None)
            return task

        task = self._run_write(_op)
        if task and task.status == TaskStatus.CANCELLED:
            logger.info(f"Task {task_id} cancelled and moved to trash")
        return task

    def update_priority(self, task_id: int, priority: TaskPriority) -> Optional[Task]:
        """Update task priority"""

        def _op(session: Session) -> Optional[Task]:
            task = session.query(Task).filter(Task.id == task_id).first()
            if task:
                task.priority = priority
            return task

        task = self._run_write(_op)
        if task:
            logger.info(f"Task {task_id} priority updated to {priority}")
        return task

    def get_queue_status(self) -> dict:
        """Get overall queue status"""

        def _op(session: Session) -> dict:
            total = session.query(Task).count()
            pending = session.query(Task).filter(Task.status == TaskStatus.PENDING).count()
            in_progress = session.query(Task).filter(Task.status == TaskStatus.IN_PROGRESS).count()
            completed = session.query(Task).filter(Task.status == TaskStatus.COMPLETED).count()
            failed = session.query(Task).filter(Task.status == TaskStatus.FAILED).count()

            return {
                "total": total,
                "pending": pending,
                "in_progress": in_progress,
                "completed": completed,
                "failed": failed,
            }

        return self._run_read(_op)

    def get_task_context(self, task_id: int) -> Optional[dict]:
        """Get task context as dict"""
        task = self.get_task(task_id)
        if task and task.context:
            return json.loads(task.context)
        return None

    def get_projects(self) -> List[dict]:
        """Get all projects with task counts and status"""

        def _op(session: Session) -> List[dict]:
            projects = session.query(Task.project_id, Task.project_name).filter(
                Task.project_id.isnot(None)
            ).distinct().all()

            result = []
            for project_id, project_name in projects:
                if project_id is None:
                    continue

                tasks = session.query(Task).filter(Task.project_id == project_id).all()
                pending = sum(1 for t in tasks if t.status == TaskStatus.PENDING)
                in_progress = sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS)
                completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)

                result.append({
                    "project_id": project_id,
                    "project_name": project_name or project_id,
                    "total_tasks": len(tasks),
                    "pending": pending,
                    "in_progress": in_progress,
                    "completed": completed,
                })

            return sorted(result, key=lambda x: x["project_id"])

        return self._run_read(_op)

    def timeout_expired_tasks(self, max_age_seconds: int) -> List[Task]:
        """Mark in-progress tasks that exceed the timeout as failed and return them."""
        if max_age_seconds <= 0:
            return []

        def _op(session: Session) -> List[Task]:
            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=max_age_seconds)
            tasks = (
                session.query(Task)
                .filter(
                    Task.status == TaskStatus.IN_PROGRESS,
                    Task.started_at.isnot(None),
                    Task.started_at < cutoff,
                )
                .all()
            )

            if not tasks:
                return []

            now = datetime.now(timezone.utc).replace(tzinfo=None)
            for task in tasks:
                task.status = TaskStatus.FAILED
                task.completed_at = now
                task.error_message = (
                    f"Timed out after exceeding {max_age_seconds // 60} minute limit."
                )
            return tasks

        tasks = self._run_write(_op)
        if tasks:
            logger.warning(
                f"Timed out tasks: {[task.id for task in tasks]} (>{max_age_seconds}s)"
            )
        return tasks

    def get_project_by_id(self, project_id: str) -> Optional[dict]:
        """Get project info by ID"""

        def _op(session: Session) -> Optional[dict]:
            tasks = session.query(Task).filter(Task.project_id == project_id).all()
            if not tasks:
                return None

            first_task = tasks[0]
            pending = sum(1 for t in tasks if t.status == TaskStatus.PENDING)
            in_progress = sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS)
            completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)

            return {
                "project_id": project_id,
                "project_name": first_task.project_name or project_id,
                "total_tasks": len(tasks),
                "pending": pending,
                "in_progress": in_progress,
                "completed": completed,
                "failed": failed,
                "created_at": min(t.created_at for t in tasks),
                "tasks": [
                    {
                        "id": t.id,
                        "description": t.description[:50],
                        "status": t.status.value,
                        "priority": t.priority.value,
                        "created_at": t.created_at.isoformat(),
                    }
                    for t in sorted(tasks, key=lambda x: x.created_at, reverse=True)[:5]
                ],
            }

        return self._run_read(_op)

    def get_project_tasks(self, project_id: str) -> List[Task]:
        """Get all tasks for a project"""

        def _op(session: Session) -> List[Task]:
            return session.query(Task).filter(Task.project_id == project_id).order_by(
                Task.created_at.desc()
            ).all()

        return self._run_read(_op)

    def get_recent_tasks(self, limit: int = 10) -> List[Task]:
        """Get the most recent tasks across all projects."""

        def _op(session: Session) -> List[Task]:
            return (
                session.query(Task)
                .order_by(Task.created_at.desc())
                .limit(limit)
                .all()
            )

        return self._run_read(_op)

    def get_failed_tasks(self, limit: int = 10) -> List[Task]:
        """Get the most recent failed tasks."""

        def _op(session: Session) -> List[Task]:
            return (
                session.query(Task)
                .filter(Task.status == TaskStatus.FAILED)
                .order_by(Task.created_at.desc())
                .limit(limit)
                .all()
            )

        return self._run_read(_op)

    def delete_project(self, project_id: str) -> int:
        """Soft delete all tasks in a project (mark as CANCELLED). Returns count of affected tasks."""

        def _op(session: Session) -> int:
            tasks = session.query(Task).filter(Task.project_id == project_id).all()
            count = 0
            for task in tasks:
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                    task.deleted_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    count += 1
            return count

        count = self._run_write(_op)
        if count:
            logger.info(f"Soft deleted project {project_id}: {count} tasks moved to trash")
        return count
