"""Command line interface for Sleepless Agent."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from sleepless_agent.monitoring.logging import get_logger
logger = get_logger(__name__)

from sleepless_agent.context import extract_context_for_task
from sleepless_agent.tasks.utils import slugify_project
from sleepless_agent.templates import TemplateLoader, TemplateRegistry

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sleepless_agent.utils.config import get_config
from sleepless_agent.core.models import TaskPriority, TaskStatus, init_db
from sleepless_agent.core.queue import TaskQueue
from sleepless_agent.utils.display import format_age_seconds, format_duration, relative_time, shorten
from sleepless_agent.utils.live_status import LiveStatusTracker
from sleepless_agent.tasks.utils import prepare_task_creation
from sleepless_agent.monitoring.monitor import HealthMonitor
from sleepless_agent.monitoring.report_generator import ReportGenerator


@dataclass
class CLIContext:
    """Holds shared resources for CLI commands."""

    task_queue: TaskQueue
    monitor: HealthMonitor
    report_generator: ReportGenerator
    db_path: Path
    results_path: Path
    logs_dir: Path


def build_context(args: argparse.Namespace) -> CLIContext:
    """Create the CLI context using config values."""

    config = get_config()

    # Infer paths from workspace_root, similar to WorkspaceSetup._apply_workspace_root
    workspace_root = Path(config.agent.workspace_root).expanduser().resolve()
    data_dir = workspace_root / "data"
    db_path = data_dir / "tasks.db"
    results_path = data_dir / "results"
    logs_dir = db_path.parent  # Use same directory as db_path for metrics

    db_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    # Ensure database schema exists before instantiating the queue
    init_db(str(db_path))

    queue = TaskQueue(str(db_path))
    monitor = HealthMonitor(str(db_path), str(results_path))
    report_generator = ReportGenerator(base_path=str(db_path.parent / "reports"))

    return CLIContext(
        task_queue=queue,
        monitor=monitor,
        report_generator=report_generator,
        db_path=db_path,
        results_path=results_path,
        logs_dir=logs_dir,
    )




def command_task(
    ctx: CLIContext,
    description: str,
    priority: TaskPriority,
    project_name: Optional[str] = None,
    template_name: Optional[str] = None,
) -> int:
    """Create a task with the given priority, optionally from a template."""

    # Handle template expansion
    if template_name:
        loader = TemplateLoader()
        loader.load_all()
        template = loader.get_template(template_name)

        if not template:
            print(f"Template not found: {template_name}", file=sys.stderr)
            print("Use 'sle templates' to list available templates.", file=sys.stderr)
            return 1

        # Parse arguments from description (format: "key=value key2=value2" or positional for 'target')
        args = _parse_template_args(description, template)

        # Validate required arguments
        is_valid, errors = template.validate_args(args)
        if not is_valid:
            print(f"Template '{template_name}' validation failed:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            print(f"\nUsage: sle think --template={template_name} <target> [key=value ...]", file=sys.stderr)
            return 1

        # Extract codebase context if template requests it
        context = None
        if template.context_injection:
            try:
                context = extract_context_for_task(compact=False)
            except Exception as exc:
                logger.warning("context.extraction_failed", error=str(exc))

        # Expand the template with context
        description = template.expand(args, context=context)

        # Use template's priority if not overridden
        priority_map = {
            "urgent": TaskPriority.SERIOUS,
            "serious": TaskPriority.SERIOUS,
            "thought": TaskPriority.THOUGHT,
            "generated": TaskPriority.GENERATED,
        }
        if not project_name:  # Only use template priority if no project (no override)
            priority = priority_map.get(template.priority.lower(), TaskPriority.SERIOUS)

    if not description.strip():
        print("Description cannot be empty", file=sys.stderr)
        return 1

    (
        cleaned_description,
        final_project_name,
        project_id,
        note,
    ) = prepare_task_creation(description, project_override=project_name)

    task = ctx.task_queue.add_task(
        description=cleaned_description,
        priority=priority,
        project_id=project_id,
        project_name=final_project_name,
    )
    if priority == TaskPriority.SERIOUS:
        label = "Serious"
    elif priority == TaskPriority.THOUGHT:
        label = "Thought"
    else:
        label = "Generated"
    project_info = f" [Project: {final_project_name}]" if final_project_name else ""

    # Show template info if used
    template_info = f" (from template: {template_name})" if template_name else ""
    print(f"{label} task #{task.id} queued{project_info}{template_info}:")

    # Show truncated description for template tasks (full prompt can be long)
    if template_name and len(cleaned_description) > 200:
        print(f"{cleaned_description[:200]}...")
    else:
        print(cleaned_description)

    if note:
        print(note, file=sys.stderr)
    return 0


def _parse_template_args(description: str, template) -> dict[str, str]:
    """Parse template arguments from description string.

    Supports two formats:
    1. Positional: "src/module.py" -> assigns to first required param (usually 'target')
    2. Key-value: "target=src/module.py coverage=edge cases"

    Args:
        description: The argument string
        template: TaskTemplate instance

    Returns:
        Dictionary of argument name -> value
    """
    args: dict[str, str] = {}
    parts = description.split()

    # Check if any part contains '='
    has_kv = any("=" in part for part in parts)

    if has_kv:
        # Key-value mode: parse key=value pairs
        current_key = None
        current_value_parts = []

        for part in parts:
            if "=" in part and current_key is None:
                # New key=value pair
                key, _, value = part.partition("=")
                if current_key and current_value_parts:
                    args[current_key] = " ".join(current_value_parts)
                current_key = key
                current_value_parts = [value] if value else []
            elif current_key:
                # Continue current value
                current_value_parts.append(part)
            else:
                # First positional before any key=value
                required = template.get_required_parameters()
                if required and required[0].name not in args:
                    args[required[0].name] = part

        # Save last key-value pair
        if current_key and current_value_parts:
            args[current_key] = " ".join(current_value_parts)
    else:
        # Positional mode: entire description is the first required param
        required = template.get_required_parameters()
        if required:
            args[required[0].name] = description

    return args


def command_templates(ctx: CLIContext) -> int:
    """List available task templates."""

    console = Console()
    loader = TemplateLoader()
    loader.load_all()

    templates = loader.registry.list_all()

    if not templates:
        console.print("[yellow]No templates found.[/]")
        console.print("Templates are loaded from:")
        console.print("  • Built-in: package/templates/builtin/")
        console.print("  • User: ~/.sleepless/templates/")
        console.print("  • Project: .sleepless/templates/")
        return 0

    # Group by category
    by_category: dict[str, list] = {}
    for template in templates:
        cat = template.category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(template)

    console.print("\n[bold cyan]Available Task Templates[/]\n")

    for category in sorted(by_category.keys()):
        console.print(f"[bold magenta]{category.title()}[/]")

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column("Name", style="green bold")
        table.add_column("Params", style="yellow")
        table.add_column("Description")

        for template in sorted(by_category[category], key=lambda t: t.name):
            required_params = [p.name for p in template.get_required_parameters()]
            optional_params = [f"[{p.name}]" for p in template.get_optional_parameters()]
            all_params = required_params + optional_params
            params_str = ", ".join(all_params) if all_params else "-"

            table.add_row(template.name, params_str, template.description)

        console.print(table)
        console.print()

    console.print("[dim]Usage: sle think --template=<name> <target> [key=value ...][/]")
    console.print("[dim]Example: sle think --template=add-tests src/module.py[/]")
    console.print()

    return 0


def command_check(ctx: CLIContext) -> int:
    """Render an enriched system snapshot."""

    console = Console()

    config = get_config()
    timeout_seconds = getattr(config.agent, "task_timeout_seconds", 0)
    timed_out_tasks = []
    if timeout_seconds and timeout_seconds > 0:
        try:
            timed_out_tasks = ctx.task_queue.timeout_expired_tasks(timeout_seconds)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(f"Failed to enforce task timeout during check: {exc}")

    if timed_out_tasks:
        timeout_minutes = max(1, timeout_seconds // 60)
        console.print(
            f"[yellow]⏱️ Marked {len(timed_out_tasks)} task(s) as timed out "
            f"after exceeding {timeout_minutes} minute limit.[/]"
        )

    health = ctx.monitor.check_health()
    queue_status = ctx.task_queue.get_queue_status()
    entries = _load_metrics(ctx.logs_dir)
    metrics_summary = _summarize_metrics(entries)

    # Get Pro plan usage info with threshold (only show if usage > 0)
    pro_plan_usage_info = ""
    try:
        from sleepless_agent.monitoring.pro_plan_usage import ProPlanUsageChecker
        from sleepless_agent.scheduling.time_utils import is_nighttime
        checker = ProPlanUsageChecker(
            command=config.claude_code.usage_command,
        )
        usage_percent, _ = checker.get_usage()
        if usage_percent > 0:
            threshold = config.claude_code.threshold_night if is_nighttime(night_start_hour=config.claude_code.night_start_hour, night_end_hour=config.claude_code.night_end_hour) else config.claude_code.threshold_day
            pro_plan_usage_info = f" • Pro Usage: {usage_percent:.0f}% / {threshold:.0f}% limit"
    except Exception as exc:
        logger.debug(f"Could not fetch Pro plan usage for dashboard: {exc}")

    status = health.get("status", "unknown")
    status_lower = str(status).lower()
    status_emoji = {
        "healthy": "✅",
        "degraded": "⚠️",
        "unhealthy": "❌",
    }.get(status_lower, "❔")
    status_style = {
        "healthy": "green",
        "degraded": "yellow",
        "unhealthy": "red",
    }.get(status_lower, "grey50")

    system = health.get("system", {})
    db = health.get("database", {})
    storage = health.get("storage", {})

    header_text = Text()
    header_text.append(f"{status_emoji} Sleepless Agent Dashboard", style="bold bright_magenta")
    if pro_plan_usage_info:
        header_text.append(f"{pro_plan_usage_info}", style="dim")
    header_panel = Panel(Align.center(header_text), border_style=status_style)

    health_table = Table.grid(padding=(0, 2))
    health_table.add_column(justify="right", style="bold cyan")
    health_table.add_column(justify="left")
    health_table.add_row("Status", f"{status_emoji} [bold]{str(status).upper()}[/]")
    health_table.add_row("Uptime", health.get("uptime_human", "N/A"))
    health_table.add_row("CPU", f"{system.get('cpu_percent', 'N/A')}%")
    health_table.add_row("Memory", f"{system.get('memory_percent', 'N/A')}%")
    health_table.add_row("Queue Depth", f"{queue_status['pending']} pending / {queue_status['in_progress']} running")

    storage_table = Table.grid(padding=(0, 2))
    storage_table.add_column(justify="right", style="bold cyan")
    storage_table.add_column(justify="left")
    if db:
        storage_table.add_row(
            "Database",
            f"{db.get('size_mb', 'N/A')} MB · {format_age_seconds(db.get('modified_ago_seconds'))}",
        )
    else:
        storage_table.add_row("Database", "—")
    if storage:
        storage_table.add_row(
            "Results",
            f"{storage.get('count', 'N/A')} files · {storage.get('total_size_mb', 'N/A')} MB",
        )
    else:
        storage_table.add_row("Results", "—")

    health_panel = Panel(health_table, title="System Health", border_style=status_style)
    storage_panel = Panel(storage_table, title="Storage", border_style="cyan")

    queue_table = Table(box=box.ROUNDED, expand=True, title="Queue Summary")
    queue_table.add_column("State", style="bold cyan")
    queue_table.add_column("Count", justify="right", style="bold")
    queue_table.add_row("Pending", str(queue_status["pending"]))
    queue_table.add_row("In Progress", str(queue_status["in_progress"]))
    queue_table.add_row("Completed", str(queue_status["completed"]))
    queue_table.add_row("Failed", str(queue_status["failed"]))
    queue_table.add_row("Total", str(queue_status["total"]))
    queue_panel = Panel(queue_table, border_style="blue")

    metrics_table = Table(box=box.ROUNDED, expand=True, title="Performance")
    metrics_table.add_column("Window", style="bold cyan")
    metrics_table.add_column("Tasks", justify="right")
    metrics_table.add_column("Success", justify="right")
    metrics_table.add_column("Error", justify="right", style="red")
    metrics_table.add_column("Avg Time", justify="right")

    if metrics_summary["total"]:
        success_rate = metrics_summary["success_rate"]
        error_rate = 100 - success_rate if success_rate is not None else None
        success_style = "green" if success_rate and success_rate >= 80 else "yellow" if success_rate and success_rate >= 50 else "red"
        metrics_table.add_row(
            "All-Time",
            str(metrics_summary["total"]),
            f"[{success_style}]{success_rate:.1f}%[/]" if success_rate is not None else "—",
            f"{error_rate:.1f}%" if error_rate is not None else "—",
            format_duration(metrics_summary["avg_duration"]),
        )
    else:
        metrics_table.add_row("All-Time", "0", "—", "—", "—")

    if metrics_summary["recent_total"]:
        recent_rate = metrics_summary["recent_success_rate"]
        recent_error_rate = 100 - recent_rate if recent_rate is not None else None
        recent_style = "green" if recent_rate and recent_rate >= 80 else "yellow" if recent_rate and recent_rate >= 50 else "red"
        metrics_table.add_row(
            "Last 24h",
            str(metrics_summary["recent_total"]),
            f"[{recent_style}]{recent_rate:.1f}%[/]" if recent_rate is not None else "—",
            f"{recent_error_rate:.1f}%" if recent_error_rate is not None else "—",
            format_duration(metrics_summary["recent_avg_duration"]),
        )
    else:
        metrics_table.add_row("Last 24h", "0", "—", "—", "—")

    metrics_panel = Panel(metrics_table, border_style="yellow")

    # Live Status: Show daemon and executor info
    in_progress_tasks = ctx.task_queue.get_in_progress_tasks()
    workspace_path = Path(config.agent.workspace_root).resolve()

    # Get last activity from most recent task
    recent_for_activity = ctx.task_queue.get_recent_tasks(limit=1)
    last_activity = None
    if recent_for_activity:
        task = recent_for_activity[0]
        last_activity = task.completed_at or task.started_at or task.created_at

    # Load live status entries for detailed executor info
    live_status_path = ctx.db_path.parent / "live_status.json"
    live_tracker = LiveStatusTracker(live_status_path)
    live_entries = live_tracker.entries()

    live_table = Table.grid(padding=(0, 2))
    live_table.add_column(justify="right", style="bold cyan")
    live_table.add_column(justify="left", width=80)

    # Daemon info
    daemon_status = "🔄 Running" if in_progress_tasks else "⏸️  Idle"
    daemon_status_color = "green" if in_progress_tasks else "yellow"
    live_table.add_row("Daemon Status", f"[{daemon_status_color}]{daemon_status}[/]")

    if in_progress_tasks:
        current_task = in_progress_tasks[0]
        live_table.add_row("Current Task", f"#{current_task.id}: {shorten(current_task.description, limit=70)}")
        if current_task.started_at:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            elapsed = (now - current_task.started_at).total_seconds()
            live_table.add_row("Task Runtime", format_duration(elapsed))

        # Find matching live status entry
        matching_entry = None
        for entry in live_entries:
            if entry.task_id == current_task.id:
                matching_entry = entry
                break

        if matching_entry:
            # Show current phase with emoji
            phase_emojis = {
                "planner": "🔍",
                "worker": "⚙️",
                "evaluator": "✅",
                "initializing": "🚀",
                "completed": "✅",
                "error": "❌",
            }
            phase_emoji = phase_emojis.get(matching_entry.phase.lower(), "•")
            phase_display = f"{phase_emoji} {matching_entry.phase.title()}"
            live_table.add_row("Current Phase", f"[bold]{phase_display}[/]")

            # Show current prompt (truncated and wrapped)
            if matching_entry.prompt_preview:
                prompt_text = shorten(matching_entry.prompt_preview, limit=150)
                live_table.add_row("Current Prompt", f"[dim]{prompt_text}[/]")

            # Show current answer (truncated and wrapped)
            if matching_entry.answer_preview:
                answer_text = shorten(matching_entry.answer_preview, limit=150)
                live_table.add_row("Current Answer", f"[dim italic]{answer_text}[/]")
    else:
        live_table.add_row("Current Task", "—")

    # Executor info
    live_table.add_row("Active Workspace", str(workspace_path))
    if last_activity:
        live_table.add_row("Last Activity", relative_time(last_activity))
    else:
        live_table.add_row("Last Activity", "—")

    live_panel = Panel(live_table, title="Live Status", border_style="bright_cyan")

    projects = ctx.task_queue.get_projects()
    project_panel = None
    if projects:
        project_table = Table(
            title=f"Projects ({len(projects)})",
            box=box.ROUNDED,
            expand=True,
        )
        project_table.add_column("Project", style="bold cyan")
        project_table.add_column("Pending", justify="right")
        project_table.add_column("In Progress", justify="right")
        project_table.add_column("Completed", justify="right")
        project_table.add_column("Total", justify="right")

        for proj in projects:
            project_table.add_row(
                proj["project_name"],
                str(proj["pending"]),
                str(proj["in_progress"]),
                str(proj["completed"]),
                str(proj["total_tasks"]),
            )
        project_panel = Panel(project_table, border_style="bright_blue")

    # Adaptive Details panel: show errors if present, otherwise recent tasks
    failed_tasks = ctx.task_queue.get_failed_tasks(limit=5)
    details_panel = None
    if failed_tasks:
        # Show errors
        details_table = Table(
            title=f"Details (Errors: {len(failed_tasks)})",
            box=box.MINIMAL_DOUBLE_HEAD,
            expand=True,
        )
        details_table.add_column("ID", style="bold red")
        details_table.add_column("When", style="dim")
        details_table.add_column("Task", overflow="fold")
        details_table.add_column("Error", overflow="fold", style="red")

        for task in failed_tasks:
            error_preview = shorten(task.error_message, limit=100) if task.error_message else "Unknown error"
            details_table.add_row(
                str(task.id),
                relative_time(task.created_at),
                shorten(task.description, limit=80),
                error_preview,
            )
        details_panel = Panel(details_table, border_style="red")
    else:
        # Show recent tasks (any status)
        detail_tasks = ctx.task_queue.get_recent_tasks(limit=5)
        if detail_tasks:
            details_table = Table(
                title=f"Details (Recent: {len(detail_tasks)})",
                box=box.MINIMAL_DOUBLE_HEAD,
                expand=True,
            )
            details_table.add_column("ID", style="bold")
            details_table.add_column("Status")
            details_table.add_column("When", style="dim")
            details_table.add_column("Task", overflow="fold")

            status_icons = {
                TaskStatus.COMPLETED: "✅",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.PENDING: "🕒",
                TaskStatus.FAILED: "❌",
                TaskStatus.CANCELLED: "🗑️",
            }
            status_colors = {
                TaskStatus.COMPLETED: "green",
                TaskStatus.IN_PROGRESS: "cyan",
                TaskStatus.PENDING: "yellow",
                TaskStatus.FAILED: "red",
                TaskStatus.CANCELLED: "dim",
            }

            for task in detail_tasks:
                icon = status_icons.get(task.status, "•")
                status_color = status_colors.get(task.status, "white")
                details_table.add_row(
                    str(task.id),
                    f"[{status_color}]{icon} {task.status.value}[/]",
                    relative_time(task.created_at),
                    shorten(task.description, limit=100),
                )
            details_panel = Panel(details_table, border_style="blue")

    recent_tasks = ctx.task_queue.get_recent_tasks(limit=8)
    status_icons = {
        TaskStatus.COMPLETED: "✅",
        TaskStatus.IN_PROGRESS: "🔄",
        TaskStatus.PENDING: "🕒",
        TaskStatus.FAILED: "❌",
        TaskStatus.CANCELLED: "🗑️",
    }
    recent_table = Table(
        title="Recent Activity",
        box=box.MINIMAL_DOUBLE_HEAD,
        expand=True,
    )
    recent_table.add_column("ID", style="bold")
    recent_table.add_column("Status")
    recent_table.add_column("Priority")
    recent_table.add_column("Project", style="cyan")
    recent_table.add_column("When")
    recent_table.add_column("Summary", overflow="fold")

    for task in recent_tasks:
        icon = status_icons.get(task.status, "•")
        # Color-code status
        status_colors = {
            TaskStatus.COMPLETED: "green",
            TaskStatus.IN_PROGRESS: "cyan",
            TaskStatus.PENDING: "yellow",
            TaskStatus.FAILED: "red",
            TaskStatus.CANCELLED: "dim",
        }
        status_color = status_colors.get(task.status, "white")
        priority_value = None
        if isinstance(getattr(task, "priority", None), TaskPriority):
            priority_value = task.priority.value
        elif getattr(task, "priority", None):
            priority_value = str(task.priority)

        priority_display = "—"
        if priority_value:
            label = priority_value.replace("_", " ").title()
            if priority_value == TaskPriority.SERIOUS.value:
                priority_display = f"[red bold]{label}[/]"
            elif priority_value == TaskPriority.GENERATED.value:
                priority_display = f"[magenta]{label}[/]"
            elif priority_value == TaskPriority.THOUGHT.value:
                priority_display = f"[cyan]{label}[/]"
            else:
                priority_display = label

        recent_table.add_row(
            str(task.id),
            f"[{status_color}]{icon} {task.status.value}[/]",
            priority_display,
            task.project_name or task.project_id or "—",
            relative_time(task.created_at),
            shorten(task.description),
        )
    recent_panel = Panel(recent_table, border_style="orange1")

    # Print header
    console.print()
    console.print(header_panel)

    # Create 2x2 grid for System Health, Storage, Queue, Performance
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(health_panel, storage_panel)
    grid.add_row(queue_panel, metrics_panel)
    console.print(grid)

    # Print Live Status panel
    console.print()
    console.print(live_panel)
    if details_panel:
        console.print()
        console.print(details_panel)
    if project_panel:
        console.print()
        console.print(project_panel)
    console.print()
    console.print(recent_panel)

    return 0


def command_cancel(ctx: CLIContext, identifier: str | int) -> int:
    """Cancel a pending task or move a project to trash.

    If identifier is a task ID (integer), cancels that task and moves it to trash.
    If identifier is a project name/ID (string), moves the project and all its tasks to trash.
    Nothing is permanently deleted - everything goes to workspace/trash/.
    """

    # Try to parse as integer (task ID)
    if isinstance(identifier, int):
        task_id = identifier
        task = ctx.task_queue.cancel_task(task_id)
        if not task:
            print(f"Task #{task_id} not found or already running", file=sys.stderr)
            return 1
        print(f"Task #{task_id} moved to trash")
        return 0

    # Try to interpret as project ID
    identifier_str = str(identifier)
    try:
        # Check if it's an integer string
        task_id = int(identifier_str)
        task = ctx.task_queue.cancel_task(task_id)
        if not task:
            print(f"Task #{task_id} not found or already running", file=sys.stderr)
            return 1
        print(f"Task #{task_id} moved to trash")
        return 0
    except ValueError:
        # It's a project identifier, handle project soft deletion
        project_id = slugify_project(identifier_str)
        project = ctx.task_queue.get_project_by_id(project_id)

        if not project:
            print(f"Project not found: {identifier_str} (slug: {project_id})", file=sys.stderr)
            return 1

        # Confirm move to trash
        print(f"About to move project '{project['project_name']}' ({project_id}) to trash")
        print(f"This will move {project['total_tasks']} task(s) to trash")
        print(f"Workspace will be moved to: workspace/trash/project_{project_id}_{datetime.now(timezone.utc).replace(tzinfo=None).strftime('%Y%m%d_%H%M%S')}/")

        response = input("Continue? (y/N): ").strip().lower()
        if response != 'y':
            print("Cancelled")
            return 0

        # Soft delete tasks from database
        count = ctx.task_queue.delete_project(project_id)
        print(f"Moved {count} task(s) to trash in database")

        # Move workspace directory to trash
        workspace_path = Path("workspace") / "projects" / project_id
        if workspace_path.exists():
            trash_dir = Path("workspace") / "trash"
            trash_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
            trash_path = trash_dir / f"project_{project_id}_{timestamp}"
            workspace_path.rename(trash_path)
            print(f"Moved workspace to trash: {trash_path}")
        else:
            print(f"Workspace directory not found: {workspace_path} (no workspace to move)")

        return 0


def _load_metrics(logs_dir: Path) -> list[dict]:
    """Load metrics entries from metrics.jsonl if available."""

    metrics_file = logs_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return []

    entries: list[dict] = []
    with metrics_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:  # pragma: no cover - log noise
                logger.warning("Failed to parse metrics line: {}", line)
    return entries


def _parse_timestamp(timestamp: Optional[str]) -> Optional[datetime]:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _summarize_metrics(entries: list[dict]) -> dict:
    """Extract aggregate stats from metrics log entries."""
    total = len(entries)
    successes = sum(1 for e in entries if e.get("success"))
    durations = [
        e.get("duration_seconds")
        for e in entries
        if isinstance(e.get("duration_seconds"), (int, float))
    ]
    avg_duration = sum(durations) / len(durations) if durations else None

    cutoff = datetime.now(timezone.utc).replace(tzinfo=None).replace(tzinfo=None) - timedelta(hours=24)
    recent = []
    for entry in entries:
        ts = _parse_timestamp(entry.get("timestamp"))
        if ts and ts >= cutoff:
            recent.append(entry)

    recent_total = len(recent)
    recent_successes = sum(1 for e in recent if e.get("success"))
    recent_durations = [
        e.get("duration_seconds")
        for e in recent
        if isinstance(e.get("duration_seconds"), (int, float))
    ]
    recent_avg_duration = (
        sum(recent_durations) / len(recent_durations) if recent_durations else None
    )

    def rate(success_count: int, task_count: int) -> Optional[float]:
        if task_count == 0:
            return None
        return success_count / task_count * 100

    return {
        "total": total,
        "success_rate": rate(successes, total),
        "avg_duration": avg_duration,
        "recent_total": recent_total,
        "recent_success_rate": rate(recent_successes, recent_total),
        "recent_avg_duration": recent_avg_duration,
    }




def command_trash(ctx: CLIContext, subcommand: Optional[str] = None, identifier: Optional[str] = None) -> int:
    """Manage trash (list, restore, empty)."""

    if not subcommand:
        subcommand = "list"

    if subcommand == "list":
        trash_dir = Path("workspace") / "trash"
        if not trash_dir.exists():
            print("🗑️  Trash is empty")
            return 0

        items = list(trash_dir.iterdir())
        if not items:
            print("🗑️  Trash is empty")
            return 0

        print("🗑️  Trash Contents:")
        for item in sorted(items):
            if item.is_dir():
                size_mb = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024 * 1024)
                print(f"  📁 {item.name} ({size_mb:.1f} MB)")
        return 0

    elif subcommand == "restore":
        if not identifier:
            print("Usage: sle trash restore <project_id_or_name>", file=sys.stderr)
            return 1

        trash_dir = Path("workspace") / "trash"
        if not trash_dir.exists():
            print("🗑️  Trash is empty", file=sys.stderr)
            return 1

        # Find matching item in trash
        search_term = identifier.lower().replace(" ", "-")
        matching_items = [item for item in trash_dir.iterdir() if search_term in item.name.lower()]

        if not matching_items:
            print(f"Project not found in trash: {identifier}", file=sys.stderr)
            return 1

        if len(matching_items) > 1:
            print(f"Multiple matches found for '{identifier}'. Be more specific:", file=sys.stderr)
            for item in matching_items:
                print(f"  - {item.name}", file=sys.stderr)
            return 1

        trash_item = matching_items[0]

        # Extract project_id from trash item name (e.g., "project_myapp_20231015_120000")
        parts = trash_item.name.split("_")
        if parts[0] != "project":
            print(f"Invalid trash item format: {trash_item.name}", file=sys.stderr)
            return 1

        # Reconstruct project_id (everything except the last timestamp)
        project_id = "_".join(parts[1:-2])  # Remove "project" prefix and timestamp parts

        # Restore workspace
        workspace_path = Path("workspace") / "projects" / project_id
        if workspace_path.exists():
            print(f"Workspace already exists at {workspace_path}", file=sys.stderr)
            response = input("Overwrite? (y/N): ").strip().lower()
            if response != 'y':
                print("Cancelled")
                return 0
            shutil.rmtree(workspace_path)

        trash_item.rename(workspace_path)
        print(f"✅ Restored project '{project_id}' from trash")

        # Note: Tasks remain in CANCELLED status - user would need to manually update them
        print("⚠️  Note: Tasks remain in CANCELLED status. Update them manually if needed.")
        return 0

    elif subcommand == "empty":
        trash_dir = Path("workspace") / "trash"
        if not trash_dir.exists() or not list(trash_dir.iterdir()):
            print("🗑️  Trash is already empty")
            return 0

        print("About to permanently delete all items in trash")
        response = input("Continue? (y/N): ").strip().lower()
        if response != 'y':
            print("Cancelled")
            return 0

        count = 0
        for item in trash_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                count += 1

        print(f"✅ Deleted {count} item(s) from trash")
        return 0

    else:
        print(f"Unknown trash subcommand: {subcommand}", file=sys.stderr)
        print("Usage: sle trash list|restore|empty [identifier]", file=sys.stderr)
        return 1


def command_report(ctx: CLIContext, identifier: Optional[str] = None, list_reports: bool = False) -> int:
    """Unified report command - shows task details, daily reports, or project reports.

    Usage:
        sle report              # Today's daily report
        sle report 123          # Task #123 details
        sle report 2025-10-22   # Specific date's report
        sle report project-id   # Project report
        sle report --list       # List all reports
    """

    if list_reports:
        # List all reports
        daily_reports = ctx.report_generator.list_daily_reports()
        project_reports = ctx.report_generator.list_project_reports()

        if daily_reports:
            print("📅 Daily Reports:")
            for report_date in daily_reports:
                print(f"  • {report_date}")
        else:
            print("📅 No daily reports available")

        if project_reports:
            print("\n📦 Project Reports:")
            for project_id in project_reports:
                print(f"  • {project_id}")
        else:
            if daily_reports:
                print("\n📦 No project reports available")
            else:
                print("📦 No project reports available")

        return 0

    if not identifier:
        # Default: today's daily report
        date = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d")
        report = ctx.report_generator.get_daily_report(date)
        print(report)
        return 0

    # Auto-detect: is it a task ID (integer), date, or project ID?
    # First try to parse as integer (task ID)
    try:
        task_id = int(identifier)
        # It's a task ID, show task details
        task = ctx.task_queue.get_task(task_id)
        if not task:
            print(f"Task #{task_id} not found", file=sys.stderr)
            return 1

        print(f"Task #{task.id}")
        print(f"  Status  : {task.status.value}")
        print(f"  Priority: {task.priority.value}")
        print(f"  Created : {task.created_at}")
        if task.error_message:
            print(f"  Error   : {task.error_message}")
        if task.context:
            print("  Context :")
            try:
                context = json.loads(task.context)
                print(json.dumps(context, indent=2))
            except json.JSONDecodeError:
                print(f"    {task.context}")
        return 0
    except ValueError:
        pass

    # Try to parse as date (YYYY-MM-DD)
    try:
        datetime.strptime(identifier, "%Y-%m-%d")
        # It's a date
        report = ctx.report_generator.get_daily_report(identifier)
        print(report)
        return 0
    except ValueError:
        # Not a date, treat as project ID
        report = ctx.report_generator.get_project_report(identifier)
        print(report)
        return 0


def command_usage(ctx: CLIContext) -> int:
    """Show Claude Code Pro plan usage with visual bar."""

    console = Console()

    try:
        from sleepless_agent.monitoring.pro_plan_usage import ProPlanUsageChecker
        from sleepless_agent.scheduling.time_utils import is_nighttime

        config = get_config()
        checker = ProPlanUsageChecker(command=config.claude_code.usage_command)

        usage_percent, reset_time = checker.get_usage()

        # Determine current threshold based on time of day
        night_mode = is_nighttime(
            night_start_hour=config.claude_code.night_start_hour,
            night_end_hour=config.claude_code.night_end_hour,
        )
        current_threshold = (
            config.claude_code.threshold_night if night_mode else config.claude_code.threshold_day
        )
        period_label = "Night" if night_mode else "Day"

        # Format reset time
        if reset_time:
            reset_str = reset_time.strftime("%I:%M %p UTC")
        else:
            reset_str = "Unknown"

        # Determine status
        if usage_percent >= current_threshold:
            status_emoji = "🔴"
            status_text = "At Limit"
            status_style = "red bold"
        elif usage_percent >= current_threshold - 10:
            status_emoji = "🟡"
            status_text = "Near Limit"
            status_style = "yellow"
        elif usage_percent >= 50:
            status_emoji = "🟠"
            status_text = "Moderate"
            status_style = "orange1"
        else:
            status_emoji = "🟢"
            status_text = "Healthy"
            status_style = "green"

        # Build usage bar
        bar_length = 40
        filled = int(usage_percent / 100 * bar_length)
        empty = bar_length - filled
        usage_bar = "█" * filled + "░" * empty

        # Build table
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold cyan")
        table.add_column(justify="left")

        table.add_row("Usage", f"[{status_style}]{usage_percent:.1f}%[/] {usage_bar}")
        table.add_row("Status", f"{status_emoji} [{status_style}]{status_text}[/]")
        table.add_row("Period", f"{period_label} (threshold: {current_threshold:.0f}%)")
        table.add_row("Resets", reset_str)

        # Add warning if near or at limit
        warning_text = None
        if usage_percent >= current_threshold:
            warning_text = "⚠️  Usage at threshold - new task generation is paused until reset"
        elif usage_percent >= current_threshold - 10:
            remaining = current_threshold - usage_percent
            warning_text = f"ℹ️  {remaining:.1f}% remaining before threshold"

        panel = Panel(
            table,
            title=f"{status_emoji} Claude Code Usage",
            border_style=status_style.split()[0],  # Use first word of style for border
        )

        console.print()
        console.print(panel)

        if warning_text:
            console.print(f"[dim]{warning_text}[/]")

        console.print()
        return 0

    except Exception as e:
        console.print(f"[red]Failed to get usage: {e}[/]")
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Sleepless Agent command line interface")

    subparsers = parser.add_subparsers(dest="command", required=True)

    think_parser = subparsers.add_parser("think", help="Create a task or capture a thought")
    think_parser.add_argument("-p", "--project", help="Project name (optional). With -p: creates SERIOUS priority project task. Without -p: creates THOUGHT priority one-time task.")
    think_parser.add_argument("-t", "--template", help="Use a template (e.g., add-tests, code-review, refactor, document, debug)")
    think_parser.add_argument("description", nargs='*', help="Task/thought description or template arguments")

    subparsers.add_parser("templates", help="List available task templates")

    subparsers.add_parser("check", help="Show comprehensive system overview with rich output")
    subparsers.add_parser("usage", help="Show Claude Code Pro plan usage")

    cancel_parser = subparsers.add_parser("cancel", help="Move a task or project to trash")
    cancel_parser.add_argument("identifier", help="Task ID (integer) or project name/ID (string)")

    # Report command - unified for daily/project reports
    report_parser = subparsers.add_parser("report", help="Show task details, daily reports, or project reports (auto-detect)")
    report_parser.add_argument("identifier", nargs="?", help="Task ID (integer), report date (YYYY-MM-DD), or project ID (default: today)")
    report_parser.add_argument("--list", dest="list_reports", action="store_true", help="List all available reports")

    # Trash command - manage deleted items
    trash_parser = subparsers.add_parser("trash", help="Manage trash (list, restore, empty)")
    trash_parser.add_argument("subcommand", nargs="?", default="list", help="list (default) | restore | empty")
    trash_parser.add_argument("identifier", nargs="?", help="Project ID or name (for restore)")

    # Orchestrator commands
    orch_parser = subparsers.add_parser("orchestrator", help="Project orchestrator management")
    orch_subparsers = orch_parser.add_subparsers(dest="orchestrator_command", required=True)

    # orchestrator status
    orch_status_parser = orch_subparsers.add_parser("status", help="Show project configuration and status")
    orch_status_parser.add_argument("-p", "--project", help="Filter to specific project ID")
    orch_status_parser.add_argument("--config", help="Path to projects.yaml (default: ~/.sleepless-agent/data/projects.yaml)")
    orch_status_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # orchestrator goals
    orch_goals_parser = orch_subparsers.add_parser("goals", help="Show project goals and targets")
    orch_goals_parser.add_argument("-p", "--project", help="Filter to specific project ID")
    orch_goals_parser.add_argument("--config", help="Path to projects.yaml (default: ~/.sleepless-agent/data/projects.yaml)")
    orch_goals_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # orchestrator health
    orch_health_parser = orch_subparsers.add_parser("health", help="Show project health metrics")
    orch_health_parser.add_argument("-p", "--project", help="Filter to specific project ID")
    orch_health_parser.add_argument("--config", help="Path to projects.yaml (default: ~/.sleepless-agent/data/projects.yaml)")
    orch_health_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # orchestrator validate
    orch_validate_parser = orch_subparsers.add_parser("validate", help="Validate project configuration")
    orch_validate_parser.add_argument("-p", "--project", help="Filter to specific project ID")
    orch_validate_parser.add_argument("--config", help="Path to projects.yaml (default: ~/.sleepless-agent/data/projects.yaml)")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    ctx = build_context(args)

    if args.command == "think":
        description = " ".join(args.description).strip() if args.description else ""
        # For template mode, description can be empty if template has no required params
        if not description and not args.template:
            parser.error("think requires a description or --template")
        # Determine priority based on whether project is provided
        priority = TaskPriority.SERIOUS if args.project else TaskPriority.THOUGHT
        return command_task(ctx, description, priority, args.project, args.template)

    if args.command == "templates":
        return command_templates(ctx)

    if args.command == "check":
        return command_check(ctx)

    if args.command == "usage":
        return command_usage(ctx)

    if args.command == "cancel":
        return command_cancel(ctx, args.identifier)

    if args.command == "report":
        return command_report(ctx, args.identifier, args.list_reports)

    if args.command == "trash":
        return command_trash(ctx, args.subcommand, args.identifier)

    if args.command == "orchestrator":
        return command_orchestrator(args)

    parser.error(f"Unknown command: {args.command}")
    return 1


def command_orchestrator(args: argparse.Namespace) -> int:
    """Handle orchestrator subcommands."""
    from sleepless_agent.orchestration.project_config import load_projects_from_config

    # Determine config path
    if hasattr(args, 'config') and args.config:
        config_path = Path(args.config).expanduser()
    else:
        config_path = Path.home() / ".sleepless-agent" / "data" / "projects.yaml"

    # Load projects
    try:
        projects = load_projects_from_config(config_path)
    except FileNotFoundError:
        print(f"[yellow]Configuration file not found: {config_path}[/yellow]")
        print("Create a projects.yaml file or run: sle orchestrator init")
        return 1
    except Exception as e:
        print(f"[red]Error loading config: {e}[/red]")
        return 1

    # Filter by project if specified
    project_filter = getattr(args, 'project', None)
    if project_filter:
        projects = [p for p in projects if p.id == project_filter]
        if not projects:
            print(f"[red]Project '{project_filter}' not found[/red]")
            return 1

    console = Console()

    if args.orchestrator_command == "status":
        return _orchestrator_status(projects, args, console)
    elif args.orchestrator_command == "goals":
        return _orchestrator_goals(projects, args, console)
    elif args.orchestrator_command == "health":
        return _orchestrator_health(projects, args, console)
    elif args.orchestrator_command == "validate":
        return _orchestrator_validate(projects, console)
    else:
        print(f"[red]Unknown orchestrator command: {args.orchestrator_command}[/red]")
        return 1


def _orchestrator_status(projects, args, console) -> int:
    """Show project status."""
    if getattr(args, 'json', False):
        data = []
        for p in projects:
            data.append({
                "id": p.id,
                "name": p.name,
                "enabled": p.enabled,
                "priority": p.priority,
                "priority_weight": p.priority_weight,
                "check_interval_hours": p.check_interval_hours,
                "has_github": p.has_github,
                "goal_count": len(p.goals),
                "constraint_count": len(p.constraints),
            })
        import json
        print(json.dumps(data, indent=2))
    else:
        table = Table(title="Project Status")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Enabled", style="green")
        table.add_column("Priority", style="yellow")
        table.add_column("Weight", style="blue")
        table.add_column("GitHub", style="magenta")
        table.add_column("Goals", style="white")
        table.add_column("Constraints", style="white")

        for p in projects:
            table.add_row(
                p.id,
                p.name,
                "[green]✓[/green]" if p.enabled else "[red]✗[/red]",
                p.priority,
                str(p.priority_weight),
                "[green]✓[/green]" if p.has_github else "[red]✗[/red]",
                str(len(p.goals)),
                str(len(p.constraints)),
            )

        console.print(table)
    return 0


def _orchestrator_goals(projects, args, console) -> int:
    """Show project goals."""
    if getattr(args, 'json', False):
        data = []
        for p in projects:
            for goal in p.goals:
                data.append({
                    "project_id": p.id,
                    "project_name": p.name,
                    "type": goal.type,
                    "metric": goal.metric,
                    "target": goal.target,
                    "description": goal.description,
                    "keywords": goal.keywords or [],
                })
        import json
        print(json.dumps(data, indent=2))
    else:
        for p in projects:
            if not p.goals:
                console.print(f"[dim]No goals configured for {p.name}[/dim]")
                continue

            console.print(f"\n[bold cyan]{p.name} ({p.id})[/bold cyan]")

            table = Table()
            table.add_column("Type", style="green")
            table.add_column("Metric", style="blue")
            table.add_column("Target", style="yellow")
            table.add_column("Description", style="white")
            table.add_column("Keywords", style="dim")

            for goal in p.goals:
                table.add_row(
                    goal.type,
                    goal.metric or "N/A",
                    str(goal.target),
                    goal.description or "",
                    ", ".join(goal.keywords or []),
                )

            console.print(table)
    return 0


def _orchestrator_health(projects, args, console) -> int:
    """Show project health (placeholder - requires running orchestrator)."""
    console.print("[yellow]Project health requires the orchestrator to be running.[/yellow]")
    console.print("\n[bold]Project Summary:[/bold]")

    table = Table()
    table.add_column("Project", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Goals", style="yellow")
    table.add_column("Constraints", style="magenta")

    for p in projects:
        table.add_row(
            p.name,
            "[green]Enabled[/green]" if p.enabled else "[dim]Disabled[/dim]",
            str(len(p.goals)),
            str(len(p.constraints)),
        )

    console.print(table)
    console.print("\n[dim]Health metrics are calculated during orchestrator runs.[/dim]")
    return 0


def _orchestrator_validate(projects, console) -> int:
    """Validate project configuration."""
    errors_found = False

    for p in projects:
        console.print(f"\n[bold cyan]Validating {p.name} ({p.id})[/bold cyan]")

        errors = []
        warnings = []

        # Check required fields
        if not p.local_path:
            errors.append("Missing local_path")

        # Check if local path exists
        if p.local_path:
            local = Path(p.local_path).expanduser()
            if not local.exists():
                warnings.append(f"Local path does not exist: {local}")

        # Check GitHub config
        if p.github:
            if not p.github.repo:
                errors.append("GitHub configured but missing repo")
            else:
                if "/" not in p.github.repo:
                    errors.append(f"Invalid GitHub repo format: {p.github.repo}")

        # Check goals
        for goal in p.goals:
            if not goal.type:
                errors.append("Goal missing type")
            if goal.target is None:
                warnings.append(f"Goal '{goal.type}' has no target")

        # Display results
        if errors:
            errors_found = True
            console.print(f"  [red]Errors ({len(errors)}):[/red]")
            for error in errors:
                console.print(f"    [red]✗ {error}[/red]")
        else:
            console.print(f"  [green]No errors found[/green]")

        if warnings:
            console.print(f"  [yellow]Warnings ({len(warnings)}):[/yellow]")
            for warning in warnings:
                console.print(f"    [yellow]⚠ {warning}[/yellow]")

    return 1 if errors_found else 0


if __name__ == "__main__":  # pragma: no cover - manual execution
    sys.exit(main())
