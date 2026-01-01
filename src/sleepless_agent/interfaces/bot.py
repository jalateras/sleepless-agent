"""Slack bot interface for task management"""

import asyncio
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sleepless_agent.monitoring.logging import get_logger
logger = get_logger(__name__)

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

from sleepless_agent.utils.display import format_age_seconds, format_duration, relative_time, shorten
from sleepless_agent.core.models import TaskPriority, TaskStatus
from sleepless_agent.core.queue import TaskQueue
from sleepless_agent.tasks.utils import prepare_task_creation, slugify_project
from sleepless_agent.utils.live_status import LiveStatusTracker
from sleepless_agent.monitoring.report_generator import ReportGenerator
from sleepless_agent.chat import ChatSessionManager, ChatExecutor, ChatHandler
from sleepless_agent.storage.feedback import FeedbackStore, classify_reaction, FeedbackType
from sleepless_agent.core.models import CheckpointStatus
from sleepless_agent.context import extract_context_for_task
from sleepless_agent.templates import TemplateLoader
from sleepless_agent.interfaces.streaming import StreamManager


class SlackBot:
    """Slack bot for task management"""

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        task_queue: TaskQueue,
        scheduler=None,
        monitor=None,
        report_generator=None,
        live_status_tracker: Optional[LiveStatusTracker] = None,
        workspace_root: str = "./workspace",
        feedback_store: Optional[FeedbackStore] = None,
        checkpoint_manager=None,
        stream_manager: Optional[StreamManager] = None,
    ):
        """Initialize Slack bot"""
        self.bot_token = bot_token
        self.app_token = app_token
        self.task_queue = task_queue
        self.scheduler = scheduler
        self.monitor = monitor
        self.report_generator = report_generator
        self.live_status_tracker = live_status_tracker
        self.workspace_root = Path(workspace_root)
        self.feedback_store = feedback_store
        self.checkpoint_manager = checkpoint_manager
        self.stream_manager = stream_manager
        self.client = WebClient(token=bot_token)
        self.socket_mode_client = SocketModeClient(app_token=app_token, web_client=self.client)

        # Initialize chat mode components
        chat_sessions_path = self.workspace_root / "data" / "chat_sessions.json"
        self.chat_session_manager = ChatSessionManager(storage_path=chat_sessions_path)
        self.chat_executor = ChatExecutor(workspace_root=str(self.workspace_root))
        self.chat_handler = ChatHandler(
            session_manager=self.chat_session_manager,
            chat_executor=self.chat_executor,
            task_queue=self.task_queue,
            slack_client=self.client,
        )
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_thread: Optional[threading.Thread] = None

    def start(self):
        """Start bot and listen for events"""
        self.socket_mode_client.socket_mode_request_listeners.append(self.handle_event)
        self.socket_mode_client.connect()
        logger.info("Slack bot started and listening for events")

    def stop(self):
        """Stop bot"""
        self.socket_mode_client.close()
        logger.info("Slack bot stopped")

    def handle_event(self, client: SocketModeClient, req: SocketModeRequest):
        """Handle incoming Slack events"""
        try:
            # Acknowledge immediately to meet Slack's 3-second requirement
            client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))

            if req.type == "events_api":
                self.handle_events_api(req)
            elif req.type == "slash_commands":
                self.handle_slash_command(req)
            elif req.type == "interactive":
                self.handle_interactive_action(req)
        except Exception as e:
            logger.error(f"Error handling event: {e}")

    def handle_events_api(self, req: SocketModeRequest):
        """Handle events API"""
        event = req.payload.get("event", {})
        event_type = event.get("type")

        if event_type == "message":
            self.handle_message(event)
        elif event_type == "reaction_added":
            self.handle_reaction_added(event)

    def handle_message(self, event: dict):
        """Handle incoming messages"""
        # Ignore bot messages
        if event.get("bot_id"):
            return

        # Ignore message subtypes (edits, deletes, etc.)
        if event.get("subtype"):
            return

        channel = event.get("channel")
        user = event.get("user")
        text = event.get("text", "").strip()
        thread_ts = event.get("thread_ts")  # Parent thread timestamp if in a thread

        logger.debug(f"Message event: user={user}, channel={channel}, thread_ts={thread_ts}, text={text[:50] if text else 'empty'}...")

        # Check if this message is in a chat mode thread
        if thread_ts:
            session = self.chat_session_manager.get_session_by_thread(thread_ts)
            logger.debug(f"Thread message lookup: thread_ts={thread_ts}, session_found={session is not None}")
            if session:
                logger.debug(f"Session details: session_user={session.user_id}, event_user={user}, match={session.user_id == user}")
                if session.user_id == user:
                    logger.info(f"Chat mode message from {user} in thread {thread_ts}: {text[:50]}...")
                    self._handle_chat_message_async(session, text, channel, thread_ts)
                    return
                else:
                    logger.debug(f"User mismatch: session is for {session.user_id}, message from {user}")

        logger.info(f"Message from {user}: {text}")

    def handle_reaction_added(self, event: dict):
        """Handle reaction_added events to capture user feedback on task outcomes.

        When users react to task completion messages with thumbs up/down or similar
        emojis, we record this as feedback to improve future task generation.
        """
        if not self.feedback_store:
            logger.debug("Feedback store not configured, skipping reaction handling")
            return

        reaction = event.get("reaction", "")
        user_id = event.get("user", "")
        item = event.get("item", {})

        # Only handle reactions to messages
        if item.get("type") != "message":
            return

        channel_id = item.get("channel", "")
        message_ts = item.get("ts", "")

        # Classify the reaction early to avoid unnecessary API calls
        feedback_type = classify_reaction(reaction)
        if feedback_type == FeedbackType.NEUTRAL:
            logger.debug(f"Ignoring neutral reaction '{reaction}' from {user_id}")
            return

        logger.info(f"Reaction added: {reaction} by {user_id} on message {message_ts}")

        # Extract task ID from the message
        task_id = self._extract_task_id_from_message(channel_id, message_ts)
        if not task_id:
            logger.debug(f"Could not extract task ID from message {message_ts}")
            return

        # Get task context for enriched feedback
        task = self.task_queue.get_task(task_id)
        if not task:
            logger.debug(f"Task {task_id} not found for feedback")
            return

        # Determine generation source if task was auto-generated
        generation_source = None
        if task.priority == TaskPriority.GENERATED:
            # Check generation history for source
            generation_source = self._get_generation_source(task_id)

        # Record the feedback
        try:
            self.feedback_store.record_feedback(
                task_id=task_id,
                user_id=user_id,
                reaction=reaction,
                message_ts=message_ts,
                channel_id=channel_id,
                task=task,
                generation_source=generation_source,
            )
            logger.info(
                f"Recorded {feedback_type.value} feedback for task {task_id} "
                f"(reaction={reaction}, user={user_id})"
            )
        except Exception as e:
            logger.error(f"Failed to record feedback for task {task_id}: {e}")

    def _extract_task_id_from_message(self, channel_id: str, message_ts: str) -> Optional[int]:
        """Extract task ID from a Slack message.

        Looks for patterns like 'Task #123' or '#123' in the message text.
        Returns the task ID if found, None otherwise.
        """
        import re

        try:
            # Fetch the message content from Slack
            response = self.client.conversations_history(
                channel=channel_id,
                latest=message_ts,
                inclusive=True,
                limit=1,
            )

            messages = response.get("messages", [])
            if not messages:
                return None

            message = messages[0]
            text = message.get("text", "")

            # Also check blocks for task IDs (Block Kit messages)
            blocks = message.get("blocks", [])
            for block in blocks:
                if block.get("type") == "section":
                    block_text = block.get("text", {}).get("text", "")
                    text += " " + block_text

            # Look for task ID patterns: "Task #123", "#123", "task_id=123"
            patterns = [
                r'Task\s*#(\d+)',
                r'#(\d+)\b',
                r'task[_\s]?id[=:\s]+(\d+)',
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return int(match.group(1))

            return None

        except SlackApiError as e:
            logger.error(f"Failed to fetch message {message_ts} from {channel_id}: {e}")
            return None

    def _get_generation_source(self, task_id: int) -> Optional[str]:
        """Get the generation source (prompt archetype) for an auto-generated task."""
        try:
            # Query the generation_history table
            from sqlalchemy.orm import Session
            from sleepless_agent.core.models import GenerationHistory

            with self.task_queue._session_scope() as session:
                history = session.query(GenerationHistory).filter(
                    GenerationHistory.task_id == task_id
                ).first()
                return history.source if history else None
        except Exception as e:
            logger.debug(f"Could not get generation source for task {task_id}: {e}")
            return None

    def handle_interactive_action(self, req: SocketModeRequest):
        """Handle interactive component actions (buttons, menus, etc.)

        This handles Block Kit interactive components like the Approve/Reject
        buttons in checkpoint messages.
        """
        payload = req.payload
        action_type = payload.get("type")

        if action_type != "block_actions":
            logger.debug(f"Ignoring interactive action type: {action_type}")
            return

        actions = payload.get("actions", [])
        user = payload.get("user", {})
        user_id = user.get("id", "")
        channel = payload.get("channel", {})
        channel_id = channel.get("id", "")
        message = payload.get("message", {})
        message_ts = message.get("ts", "")
        response_url = payload.get("response_url", "")

        for action in actions:
            action_id = action.get("action_id", "")
            value = action.get("value", "")

            logger.info(f"Interactive action: {action_id} with value {value} from {user_id}")

            # Handle checkpoint approve/reject buttons
            if action_id in ("checkpoint_approve", "checkpoint_reject"):
                self._handle_checkpoint_action(
                    action_id=action_id,
                    checkpoint_id=value,
                    user_id=user_id,
                    channel_id=channel_id,
                    message_ts=message_ts,
                    response_url=response_url,
                )

            # Handle stream pause/resume buttons
            elif action_id in ("stream_pause", "stream_resume"):
                self._handle_stream_action(
                    action_id=action_id,
                    task_id=value,
                    user_id=user_id,
                    channel_id=channel_id,
                )

    def _handle_checkpoint_action(
        self,
        action_id: str,
        checkpoint_id: str,
        user_id: str,
        channel_id: str,
        message_ts: str,
        response_url: str,
    ):
        """Handle checkpoint approval or rejection from Slack button click."""
        if not self.checkpoint_manager:
            logger.warning("Checkpoint action received but no checkpoint_manager configured")
            self._update_checkpoint_message(
                response_url=response_url,
                text="Checkpoint system not configured",
                success=False,
            )
            return

        try:
            checkpoint_id_int = int(checkpoint_id)
        except ValueError:
            logger.error(f"Invalid checkpoint ID: {checkpoint_id}")
            return

        # Determine status based on action
        if action_id == "checkpoint_approve":
            status = CheckpointStatus.APPROVED
            action_text = "approved"
            emoji = ":white_check_mark:"
        else:
            status = CheckpointStatus.REJECTED
            action_text = "rejected"
            emoji = ":x:"

        # Resolve the checkpoint
        checkpoint = self.checkpoint_manager.resolve(
            checkpoint_id=checkpoint_id_int,
            status=status,
            resolved_by=user_id,
        )

        if not checkpoint:
            logger.warning(f"Checkpoint {checkpoint_id} not found or already resolved")
            self._update_checkpoint_message(
                response_url=response_url,
                text="Checkpoint already resolved or not found",
                success=False,
            )
            return

        # Get task info for the response
        task = self.task_queue.get_task(checkpoint.task_id)
        task_desc = task.description[:50] if task else "Unknown"

        # Update the original message to show resolution
        self._update_checkpoint_message(
            response_url=response_url,
            text=f"{emoji} Checkpoint {action_text} by <@{user_id}>",
            checkpoint=checkpoint,
            task_desc=task_desc,
            success=True,
        )

        logger.info(
            f"Checkpoint {checkpoint_id} {action_text} by {user_id} for task {checkpoint.task_id}"
        )

    def _handle_stream_action(
        self,
        action_id: str,
        task_id: str,
        user_id: str,
        channel_id: str,
    ):
        """Handle stream pause/resume from Slack button click."""
        if not self.stream_manager:
            logger.warning("Stream action received but no stream_manager configured")
            return

        try:
            task_id_int = int(task_id)
        except ValueError:
            logger.error(f"Invalid task ID for stream action: {task_id}")
            return

        # Get async loop for running coroutines
        loop = self._get_async_loop()

        if action_id == "stream_pause":
            future = asyncio.run_coroutine_threadsafe(
                self.stream_manager.pause_stream(task_id_int),
                loop,
            )
            success = future.result(timeout=5.0)
            if success:
                logger.info(f"Stream paused for task {task_id} by {user_id}")
            else:
                logger.warning(f"Failed to pause stream for task {task_id}")

        elif action_id == "stream_resume":
            future = asyncio.run_coroutine_threadsafe(
                self.stream_manager.resume_stream(task_id_int),
                loop,
            )
            success = future.result(timeout=5.0)
            if success:
                logger.info(f"Stream resumed for task {task_id} by {user_id}")
            else:
                logger.warning(f"Failed to resume stream for task {task_id}")

    def _update_checkpoint_message(
        self,
        response_url: str,
        text: str,
        success: bool = True,
        checkpoint=None,
        task_desc: str = "",
    ):
        """Update the checkpoint message after resolution."""
        import requests

        blocks = []

        if checkpoint:
            # Build updated message showing resolution
            type_labels = {
                "post_plan": "Plan Review",
                "pre_commit": "Pre-Commit Approval",
                "pre_pr": "Pull Request Approval",
            }
            type_label = type_labels.get(checkpoint.checkpoint_type.value, "Approval")
            status_emoji = ":white_check_mark:" if checkpoint.status == CheckpointStatus.APPROVED else ":x:"

            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{status_emoji} {type_label} - {checkpoint.status.value.upper()}",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Task #{checkpoint.task_id}*: {self._escape_slack(task_desc)}{'...' if len(task_desc) >= 50 else ''}",
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": text,
                        }
                    ],
                },
            ]
        else:
            # Simple error/info message
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": text,
                    },
                },
            ]

        try:
            response = requests.post(
                response_url,
                json={
                    "replace_original": True,
                    "blocks": blocks,
                    "text": text,
                },
                timeout=10,
            )
            if response.status_code != 200:
                logger.error(f"Failed to update checkpoint message: {response.text}")
        except Exception as e:
            logger.error(f"Failed to update checkpoint message: {e}")

    def handle_slash_command(self, req: SocketModeRequest):
        """Handle slash commands"""
        command = req.payload["command"]
        text = req.payload.get("text", "").strip()
        user = req.payload["user_id"]
        channel = req.payload["channel_id"]
        response_url = req.payload.get("response_url")

        logger.info(f"Slash command: {command} {text} from {user}")

        try:
            if command == "/task" or command == "/think":
                # Both commands now use unified handler with dynamic priority
                self.handle_think_command(text, user, channel, response_url)
            elif command == "/chat":
                self.handle_chat_command(text, user, channel, response_url)
            elif command == "/check":
                self.handle_check_command(response_url)
            elif command == "/cancel":
                self.handle_cancel_command(text, response_url)
            elif command == "/report":
                self.handle_report_command(text, response_url)
            elif command == "/trash":
                self.handle_trash_command(text, response_url)
            elif command == "/usage":
                self.handle_usage_command(response_url, channel)
            elif command == "/templates":
                self.handle_templates_command(response_url)
            else:
                self.send_response(response_url, f"Unknown command: {command}")
        except Exception as e:
            logger.error(f"Error executing {command}: {e}", exc_info=True)
            self.send_response(response_url, f"Error executing {command}: {str(e)}")

    def handle_think_command(
        self,
        args: str,
        user_id: str,
        channel_id: str,
        response_url: str,
    ):
        """Handle /think command - unified handler for both tasks and thoughts

        Usage: /think <description> [--project=<project_name>] [--template=<template_name>]

        With --project: Creates SERIOUS priority project task
        With --template: Uses a template to generate the task description
        Without --project: Creates THOUGHT priority one-time task
        """
        if not args:
            self.send_response(
                response_url,
                "Usage: /think <description> [--project=<project_name>] [--template=<template_name>]\n"
                "Use /templates to list available templates."
            )
            return

        # Check for --template flag
        template_name = None
        import re
        template_match = re.search(r'--template[=\s]+(\S+)', args)
        if template_match:
            template_name = template_match.group(1)
            # Remove the --template flag from args
            args = re.sub(r'--template[=\s]+\S+\s*', '', args).strip()

        # If template is specified, expand it
        if template_name:
            loader = TemplateLoader()
            loader.load_all()
            template = loader.get_template(template_name)

            if not template:
                self.send_response(
                    response_url,
                    f"Template not found: `{template_name}`\nUse /templates to list available templates."
                )
                return

            # Parse template arguments from remaining args
            template_args = self._parse_template_args_slack(args, template)

            # Validate required arguments
            is_valid, errors = template.validate_args(template_args)
            if not is_valid:
                error_text = f"Template `{template_name}` validation failed:\n"
                for error in errors:
                    error_text += f"• {error}\n"
                required_params = [p.name for p in template.get_required_parameters()]
                error_text += f"\nUsage: /think --template={template_name} <{', '.join(required_params)}>"
                self.send_response(response_url, error_text)
                return

            # Extract codebase context if template requests it
            context = None
            if template.context_injection:
                try:
                    context = extract_context_for_task(compact=False)
                except Exception as exc:
                    logger.warning("context.extraction_failed", error=str(exc))

            # Expand template with context
            args = template.expand(template_args, context=context)

            # Use template's priority
            priority_map = {
                "urgent": TaskPriority.SERIOUS,
                "serious": TaskPriority.SERIOUS,
                "thought": TaskPriority.THOUGHT,
                "generated": TaskPriority.GENERATED,
            }
            priority = priority_map.get(template.priority.lower(), TaskPriority.SERIOUS)

            # Create the task directly (no need to prepare_task_creation for templates)
            self._create_task(
                description=args.strip(),
                priority=priority,
                response_url=response_url,
                user_id=user_id,
                note=f"Created from template: {template_name}",
                project_name=None,
                project_id=None,
            )
            return

        # Standard flow (no template)
        (
            cleaned_description,
            project_name,
            project_id,
            note,
        ) = prepare_task_creation(args)

        if not cleaned_description.strip():
            self.send_response(response_url, "Please provide a description")
            return

        # Determine priority based on whether project is provided
        priority = TaskPriority.SERIOUS if project_id else TaskPriority.THOUGHT

        self._create_task(
            description=cleaned_description.strip(),
            priority=priority,
            response_url=response_url,
            user_id=user_id,
            note=note,
            project_name=project_name,
            project_id=project_id,
        )

    def _parse_template_args_slack(self, args: str, template) -> dict[str, str]:
        """Parse template arguments from Slack command args.

        Similar to CLI parsing but simpler.
        """
        result: dict[str, str] = {}
        parts = args.split()

        # Check if any part contains '='
        has_kv = any("=" in part for part in parts)

        if has_kv:
            # Key-value mode
            current_key = None
            current_value_parts = []

            for part in parts:
                if "=" in part:
                    # Save previous key-value pair
                    if current_key and current_value_parts:
                        result[current_key] = " ".join(current_value_parts)
                    # Start new key-value pair
                    key, _, value = part.partition("=")
                    current_key = key
                    current_value_parts = [value] if value else []
                elif current_key:
                    current_value_parts.append(part)

            # Save last key-value pair
            if current_key and current_value_parts:
                result[current_key] = " ".join(current_value_parts)
        else:
            # Positional mode: entire args is the first required param
            required = template.get_required_parameters()
            if required:
                result[required[0].name] = args

        return result

    def handle_templates_command(self, response_url: str):
        """Handle /templates command - list available task templates."""
        loader = TemplateLoader()
        loader.load_all()

        templates = loader.registry.list_all()

        if not templates:
            self.send_response(response_url, "No templates found.")
            return

        # Group by category
        by_category: dict[str, list] = {}
        for template in templates:
            cat = template.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(template)

        # Build Slack blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Available Task Templates",
                    "emoji": True,
                }
            }
        ]

        for category in sorted(by_category.keys()):
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{category.title()}*"
                }
            })

            template_lines = []
            for template in sorted(by_category[category], key=lambda t: t.name):
                required_params = [p.name for p in template.get_required_parameters()]
                params_str = f" `<{', '.join(required_params)}>`" if required_params else ""
                template_lines.append(f"• `{template.name}`{params_str} - {template.description}")

            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "\n".join(template_lines)
                }
            })

        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "Usage: `/think --template=<name> <target>`  •  Example: `/think --template=add-tests src/module.py`"
                }
            ]
        })

        self.send_response(response_url, "Available templates:", blocks=blocks)

    def handle_chat_command(
        self,
        args: str,
        user_id: str,
        channel_id: str,
        response_url: str,
    ):
        """Handle /chat command - interactive chat mode with Claude.

        Usage:
            /chat <project_name> - Start chat mode for a project
            /chat end - End current session
            /chat status - Check session status
            /chat help - Show help
        """
        result = self.chat_handler.handle_chat_command(
            args=args,
            user_id=user_id,
            channel_id=channel_id,
            response_url=response_url,
        )

        if result.get("action") == "start_session":
            # Create thread and start session
            success = self._start_chat_thread(
                user_id=result["user_id"],
                channel_id=result["channel_id"],
                project_id=result["project_id"],
                project_name=result["project_name"],
                response_url=response_url,
            )
            if not success:
                self.send_response(
                    response_url,
                    message="Failed to start chat mode. Check bot permissions.",
                )
        else:
            # Regular response
            self.send_response(
                response_url,
                message=result.get("text", ""),
                blocks=result.get("blocks"),
            )

    def _start_chat_thread(
        self,
        user_id: str,
        channel_id: str,
        project_id: str,
        project_name: str,
        response_url: str,
    ) -> bool:
        """Create a Slack thread to start the chat session.

        Returns True if successful, False otherwise.
        """
        try:
            # Create project workspace immediately (like -p flag behavior)
            workspace_path = self.workspace_root / "projects" / project_id
            workspace_path.mkdir(parents=True, exist_ok=True)

            # Create a README if it's a new project
            readme_path = workspace_path / "README.md"
            if not readme_path.exists():
                readme_content = f"# {project_name}\n\nProject created via chat mode.\n"
                readme_path.write_text(readme_content)
                logger.info(f"Created new project workspace: {workspace_path}")

            # Build welcome message
            welcome_text = (
                f"Chat mode started for project *{project_name}*.\n\n"
                f"Send messages in this thread to interact with Claude.\n"
                f"Type `exit` or use `/chat end` to end the session."
            )

            blocks = [
                self._block_header("Chat Mode Started"),
                self._block_section(welcome_text, markdown=True),
                self._block_context(f"Project: {project_name} | Session will timeout after 30 min of inactivity"),
            ]

            # Post message to channel to create thread
            response = self.client.chat_postMessage(
                channel=channel_id,
                text=f"Chat mode started for project {project_name}",
                blocks=blocks,
            )

            thread_ts = response["ts"]

            # Add active indicator emoji to the welcome message
            try:
                self.client.reactions_add(
                    channel=channel_id,
                    timestamp=thread_ts,
                    name="speech_balloon",  # 💬 emoji
                )
            except Exception as e:
                logger.debug(f"Could not add reaction: {e}")

            # Create session with thread_ts
            session = self.chat_session_manager.create_session(
                user_id=user_id,
                channel_id=channel_id,
                thread_ts=thread_ts,
                project_id=project_id,
                project_name=project_name,
                workspace_path=str(workspace_path),
            )

            logger.info(
                f"Chat session started: user={user_id}, project={project_name}, thread={thread_ts}"
            )

            # Send first instruction message in the thread
            self.client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=(
                    "👋 *I'm ready to help!*\n\n"
                    "Just type your message here and I'll respond.\n"
                    "I can read, write, and edit files in your project.\n\n"
                    "_Type `exit` when you're done._"
                ),
            )

            return True

        except Exception as e:
            logger.error(f"Failed to start chat thread: {e}", exc_info=True)
            return False

    def _handle_chat_message_async(
        self,
        session,
        text: str,
        channel: str,
        thread_ts: str,
    ):
        """Handle chat message asynchronously using a background event loop."""
        # Ensure we have an event loop running in a background thread
        if self._async_loop is None or not self._async_loop.is_running():
            self._async_loop = asyncio.new_event_loop()

            def run_loop():
                asyncio.set_event_loop(self._async_loop)
                self._async_loop.run_forever()

            self._async_thread = threading.Thread(target=run_loop, daemon=True)
            self._async_thread.start()

        # Schedule the async handler
        asyncio.run_coroutine_threadsafe(
            self.chat_handler.handle_chat_message(session, text, channel, thread_ts),
            self._async_loop,
        )

    def _create_task(
        self,
        description: str,
        priority: TaskPriority,
        response_url: str,
        user_id: str,
        note: Optional[str] = None,
        project_name: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """Create a task and send a Slack response"""
        try:
            task = self.task_queue.add_task(
                description=description,
                priority=priority,
                slack_user_id=user_id,
                project_id=project_id,
                project_name=project_name,
            )

            # Build Block Kit response
            blocks = []

            # Priority-based header
            if priority == TaskPriority.SERIOUS:
                header_text = "🔴 Serious Task Created"
                header_emoji = "🔴"
            elif priority == TaskPriority.THOUGHT:
                header_text = "🟡 Thought Captured"
                header_emoji = "💭"
            else:
                header_text = "🟢 Generated Task Created"
                header_emoji = "✨"

            blocks.append(self._block_header(header_text))

            # Task details
            fields = [
                {"label": "Task ID", "value": f"#{task.id}"},
                {"label": "Priority", "value": priority.value.capitalize()},
            ]
            if project_name:
                fields.append({"label": "Project", "value": project_name})

            blocks.append(self._block_section_fields(fields))

            # Description
            blocks.append(self._block_section(
                f"*Description:*\n{self._escape_slack(description)}",
                markdown=True
            ))

            # Note if present
            if note:
                blocks.append(self._block_context(f"ℹ️ {self._escape_slack(note)}"))

            # Fallback message
            project_info = f" [Project: {project_name}]" if project_name else ""
            fallback = f"{header_emoji} Task #{task.id} added to queue{project_info}: {description}"

            self.send_response(response_url, message=fallback, blocks=blocks)
            logger.info(f"Task {task.id} added by {user_id}" + (f" [Project: {project_name}]" if project_name else ""))

        except Exception as e:
            error_blocks = [
                self._block_header("❌ Error Creating Task"),
                self._block_section(f"Failed to add task: {str(e)}", markdown=True)
            ]
            self.send_response(response_url, message=f"Failed to add task: {str(e)}", blocks=error_blocks)
            logger.error(f"Failed to add task: {e}")

    def handle_check_command(self, response_url: str):
        """Handle /check command - comprehensive system status"""
        try:
            blocks = self._build_check_blocks()
            fallback_message = self._build_check_message()
            self.send_response(response_url, message=fallback_message, blocks=blocks)
        except Exception as e:
            self.send_response(response_url, f"Failed to get status: {str(e)}")
            logger.error(f"Failed to get status: {e}")

    def handle_cancel_command(self, identifier_str: str, response_url: str):
        """Handle /cancel command - move task or project to trash

        Usage: /cancel <task_id> or /cancel <project_id_or_name>
        """
        try:
            if not identifier_str:
                usage_blocks = [
                    self._block_header("ℹ️ Cancel Command Usage"),
                    self._block_section("*Usage:* `/cancel <task_id_or_project>`", markdown=True)
                ]
                self.send_response(response_url, message="Usage: /cancel <task_id_or_project>", blocks=usage_blocks)
                return

            # Try to parse as integer (task ID)
            try:
                task_id = int(identifier_str)
                task = self.task_queue.cancel_task(task_id)
                if task:
                    blocks = [
                        self._block_header("✅ Task Cancelled"),
                        self._block_section(
                            f"Task *#{task_id}* has been moved to trash",
                            markdown=True
                        ),
                        self._block_context(f"Task: {self._escape_slack(shorten(task.description, 100))}")
                    ]
                    self.send_response(response_url, message=f"Task #{task_id} moved to trash", blocks=blocks)
                else:
                    blocks = [
                        self._block_header("❌ Task Not Found"),
                        self._block_section(
                            f"Task *#{task_id}* not found or already running",
                            markdown=True
                        )
                    ]
                    self.send_response(response_url, message=f"Task #{task_id} not found or already running", blocks=blocks)
                return
            except ValueError:
                pass

            # Try to interpret as project ID
            project_id = slugify_project(identifier_str)
            project = self.task_queue.get_project_by_id(project_id)

            if not project:
                blocks = [
                    self._block_header("❌ Project Not Found"),
                    self._block_section(f"Project not found: *{self._escape_slack(identifier_str)}*", markdown=True)
                ]
                self.send_response(response_url, message=f"Project not found: {identifier_str}", blocks=blocks)
                return

            # Soft delete tasks from database
            count = self.task_queue.delete_project(project_id)

            # Move workspace to trash
            from datetime import datetime
            from pathlib import Path
            import shutil

            workspace_path = Path("workspace") / "projects" / project_id
            workspace_status = ""
            if workspace_path.exists():
                trash_dir = Path("workspace") / "trash"
                trash_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
                trash_path = trash_dir / f"project_{project_id}_{timestamp}"
                workspace_path.rename(trash_path)
                workspace_status = "Workspace moved to trash"
            else:
                workspace_status = "No workspace to move"

            # Build success blocks
            blocks = [
                self._block_header("🗑️ Project Cancelled"),
                self._block_section_fields([
                    {"label": "Project", "value": self._escape_slack(project["project_name"] or project_id)},
                    {"label": "Tasks Moved", "value": str(count)},
                ]),
                self._block_section(f"✅ {workspace_status}", markdown=True)
            ]

            fallback = f"✅ Moved {count} task(s) to trash. {workspace_status}"
            self.send_response(response_url, message=fallback, blocks=blocks)

        except Exception as e:
            error_blocks = [
                self._block_header("❌ Error Cancelling"),
                self._block_section(f"Failed to move to trash: {str(e)}", markdown=True)
            ]
            self.send_response(response_url, message=f"Failed to move to trash: {str(e)}", blocks=error_blocks)
            logger.error(f"Failed to cancel: {e}")

    def send_response(self, response_url: str, message: str = None, blocks: list = None, channel: str = None):
        """Send response to Slack

        Args:
            response_url: Slack response URL
            message: Plain text fallback message
            blocks: Block Kit blocks for rich formatting
            channel: Optional channel ID for fallback posting via WebClient
        """
        try:
            import requests
            payload = {
                "response_type": "in_channel"  # Make responses visible in channel
            }

            # If blocks provided, use them; otherwise use plain text
            if blocks:
                payload["blocks"] = blocks
            if message:
                payload["text"] = message

            # Ensure at least text or blocks are provided
            if not payload.get("text") and not payload.get("blocks"):
                payload["text"] = "No message"

            logger.info(f"send_response: posting to response_url, message_len={len(message or '')}, has_channel={channel is not None}")
            response = requests.post(
                response_url,
                json=payload,
                timeout=10,
            )

            # Check response status
            if response.status_code != 200:
                logger.error(f"send_response: failed status={response.status_code} body={response.text[:500]}")
                # Try fallback via WebClient if channel provided
                if channel and message:
                    logger.info(f"send_response: trying WebClient fallback to channel {channel}")
                    self._send_via_webclient(channel, message, blocks)
            else:
                logger.info(f"send_response: success status={response.status_code}")

        except Exception as e:
            logger.error(f"send_response: exception {e}")
            # Try fallback via WebClient if channel provided
            if channel and message:
                logger.info(f"send_response: trying WebClient fallback after exception")
                self._send_via_webclient(channel, message, blocks)

    def _send_via_webclient(self, channel: str, message: str, blocks: list = None):
        """Fallback method to send message via WebClient"""
        try:
            if blocks:
                self.client.chat_postMessage(channel=channel, text=message, blocks=blocks)
            else:
                self.client.chat_postMessage(channel=channel, text=message)
            logger.info(f"_send_via_webclient: success to {channel}")
        except Exception as e:
            logger.error(f"_send_via_webclient: failed {e}")

    def send_message(self, channel: str, message: str):
        """Send message to channel"""
        try:
            self.client.chat_postMessage(channel=channel, text=message, mrkdwn=True)
        except SlackApiError as e:
            logger.error(f"Failed to send message: {e}")

    def send_thread_message(self, channel: str, thread_ts: str, message: str):
        """Send message to thread"""
        try:
            self.client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=message,
                mrkdwn=True,
            )
        except SlackApiError as e:
            logger.error(f"Failed to send thread message: {e}")

    def handle_trash_command(self, args: str, response_url: str):
        """Handle /trash command - manage trash (list, restore, empty)

        Usage:
            /trash list       - Show trash contents
            /trash restore <project> - Restore project from trash
            /trash empty      - Permanently delete trash
        """
        from pathlib import Path
        import shutil

        subcommand = args.split()[0].lower() if args else "list"
        remaining_args = args[len(subcommand):].strip() if args else ""

        if subcommand == "list":
            try:
                trash_dir = Path("workspace") / "trash"
                if not trash_dir.exists():
                    blocks = [
                        self._block_header("🗑️ Trash"),
                        self._block_section("Trash is empty", markdown=True)
                    ]
                    self.send_response(response_url, message="🗑️ Trash is empty", blocks=blocks)
                    return

                items = list(trash_dir.iterdir())
                if not items:
                    blocks = [
                        self._block_header("🗑️ Trash"),
                        self._block_section("Trash is empty", markdown=True)
                    ]
                    self.send_response(response_url, message="🗑️ Trash is empty", blocks=blocks)
                    return

                blocks = [self._block_header("🗑️ Trash Contents")]

                # Build list of items
                for item in sorted(items):
                    if item.is_dir():
                        size_mb = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024 * 1024)
                        blocks.append(self._block_section(
                            f"📁 *{self._escape_slack(item.name)}*\n_{size_mb:.1f} MB_",
                            markdown=True
                        ))

                fallback = f"🗑️ Trash has {len(items)} item(s)"
                self.send_response(response_url, message=fallback, blocks=blocks)
            except Exception as e:
                error_blocks = [
                    self._block_header("❌ Error Listing Trash"),
                    self._block_section(f"Failed to list trash: {str(e)}", markdown=True)
                ]
                self.send_response(response_url, message=f"Failed to list trash: {str(e)}", blocks=error_blocks)
                logger.error(f"Failed to list trash: {e}")

        elif subcommand == "restore":
            try:
                if not remaining_args:
                    blocks = [
                        self._block_header("ℹ️ Restore Usage"),
                        self._block_section("*Usage:* `/trash restore <project_id_or_name>`", markdown=True)
                    ]
                    self.send_response(response_url, message="Usage: /trash restore <project_id_or_name>", blocks=blocks)
                    return

                trash_dir = Path("workspace") / "trash"
                if not trash_dir.exists():
                    blocks = [
                        self._block_header("🗑️ Trash Empty"),
                        self._block_section("Trash is empty", markdown=True)
                    ]
                    self.send_response(response_url, message="🗑️ Trash is empty", blocks=blocks)
                    return

                # Find matching item in trash
                search_term = remaining_args.lower().replace(" ", "-")
                matching_items = [item for item in trash_dir.iterdir() if search_term in item.name.lower()]

                if not matching_items:
                    blocks = [
                        self._block_header("❌ Project Not Found"),
                        self._block_section(f"Project not found in trash: *{self._escape_slack(remaining_args)}*", markdown=True)
                    ]
                    self.send_response(response_url, message=f"Project not found in trash: {remaining_args}", blocks=blocks)
                    return

                if len(matching_items) > 1:
                    blocks = [
                        self._block_header("⚠️ Multiple Matches"),
                        self._block_section(f"Multiple matches found for *{self._escape_slack(remaining_args)}*. Be more specific:", markdown=True)
                    ]
                    for item in matching_items:
                        blocks.append(self._block_context(f"• {self._escape_slack(item.name)}"))

                    fallback = f"Multiple matches found for '{remaining_args}'"
                    self.send_response(response_url, message=fallback, blocks=blocks)
                    return

                trash_item = matching_items[0]

                # Extract project_id from trash item name (e.g., "project_myapp_20231015_120000")
                parts = trash_item.name.split("_")
                if parts[0] != "project":
                    blocks = [
                        self._block_header("❌ Invalid Format"),
                        self._block_section(f"Invalid trash item format: *{self._escape_slack(trash_item.name)}*", markdown=True)
                    ]
                    self.send_response(response_url, message=f"Invalid trash item format: {trash_item.name}", blocks=blocks)
                    return

                # Reconstruct project_id (everything except the last timestamp)
                project_id = "_".join(parts[1:-2])  # Remove "project" prefix and timestamp parts

                # Restore workspace
                workspace_path = Path("workspace") / "projects" / project_id
                if workspace_path.exists():
                    blocks = [
                        self._block_header("⚠️ Workspace Exists"),
                        self._block_section(f"Workspace already exists at *{self._escape_slack(str(workspace_path))}*", markdown=True)
                    ]
                    self.send_response(response_url, message=f"Workspace already exists at {workspace_path}", blocks=blocks)
                    return

                trash_item.rename(workspace_path)

                blocks = [
                    self._block_header("✅ Project Restored"),
                    self._block_section(f"Project *{self._escape_slack(project_id)}* restored from trash", markdown=True),
                    self._block_context("⚠️ Note: Tasks remain in CANCELLED status. Update them manually if needed.")
                ]

                fallback = f"✅ Restored project '{project_id}' from trash"
                self.send_response(response_url, message=fallback, blocks=blocks)
            except Exception as e:
                error_blocks = [
                    self._block_header("❌ Error Restoring"),
                    self._block_section(f"Failed to restore from trash: {str(e)}", markdown=True)
                ]
                self.send_response(response_url, message=f"Failed to restore from trash: {str(e)}", blocks=error_blocks)
                logger.error(f"Failed to restore from trash: {e}")

        elif subcommand == "empty":
            try:
                trash_dir = Path("workspace") / "trash"
                if not trash_dir.exists() or not list(trash_dir.iterdir()):
                    blocks = [
                        self._block_header("🗑️ Trash"),
                        self._block_section("Trash is already empty", markdown=True)
                    ]
                    self.send_response(response_url, message="🗑️ Trash is already empty", blocks=blocks)
                    return

                count = 0
                for item in trash_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                        count += 1

                blocks = [
                    self._block_header("✅ Trash Emptied"),
                    self._block_section(f"Deleted *{count}* item(s) from trash", markdown=True)
                ]
                self.send_response(response_url, message=f"✅ Deleted {count} item(s) from trash", blocks=blocks)
            except Exception as e:
                error_blocks = [
                    self._block_header("❌ Error Emptying Trash"),
                    self._block_section(f"Failed to empty trash: {str(e)}", markdown=True)
                ]
                self.send_response(response_url, message=f"Failed to empty trash: {str(e)}", blocks=error_blocks)
                logger.error(f"Failed to empty trash: {e}")

        else:
            blocks = [
                self._block_header("ℹ️ Trash Command Usage"),
                self._block_section(
                    "*Usage:* `/trash list|restore|empty`\n\n"
                    "• `list` - Show trash contents\n"
                    "• `restore <project>` - Restore project from trash\n"
                    "• `empty` - Permanently delete all trash",
                    markdown=True
                )
            ]
            fallback = "Usage: /trash list|restore|empty"
            self.send_response(response_url, message=fallback, blocks=blocks)

    def handle_usage_command(self, response_url: str, channel: str = None):
        """Handle /usage command - show Claude Code Pro plan usage

        Usage: /usage
        """
        try:
            from sleepless_agent.monitoring.pro_plan_usage import ProPlanUsageChecker
            from sleepless_agent.utils.config import get_config
            from sleepless_agent.scheduling.time_utils import is_nighttime

            logger.debug("usage_command.start")

            config = get_config()
            checker = ProPlanUsageChecker(command=config.claude_code.usage_command)

            logger.debug("usage_command.fetching_usage")
            usage_percent, reset_time = checker.get_usage()
            logger.debug(f"usage_command.got_usage usage_percent={usage_percent}")

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
                from datetime import timezone as tz
                reset_local = reset_time.replace(tzinfo=tz.utc)
                reset_str = reset_local.strftime("%I:%M %p UTC")
            else:
                reset_str = "Unknown"

            # Determine status emoji based on usage level
            if usage_percent >= current_threshold:
                status_emoji = "🔴"
                status_text = "At Limit"
            elif usage_percent >= current_threshold - 10:
                status_emoji = "🟡"
                status_text = "Near Limit"
            elif usage_percent >= 50:
                status_emoji = "🟠"
                status_text = "Moderate"
            else:
                status_emoji = "🟢"
                status_text = "Healthy"

            # Build usage bar
            bar_length = 20
            filled = int(usage_percent / 100 * bar_length)
            empty = bar_length - filled
            usage_bar = "█" * filled + "░" * empty

            # Build blocks
            blocks = [
                self._block_header(f"{status_emoji} Claude Code Usage"),
                self._block_section(
                    f"*Usage:* `{usage_percent:.1f}%` {usage_bar}\n"
                    f"*Status:* {status_text}\n"
                    f"*Period:* {period_label} (threshold: {current_threshold:.0f}%)\n"
                    f"*Resets:* {reset_str}",
                    markdown=True
                ),
            ]

            # Add warning if near or at limit
            if usage_percent >= current_threshold:
                blocks.append(self._block_context(
                    "⚠️ Usage at threshold - new task generation is paused until reset"
                ))
            elif usage_percent >= current_threshold - 10:
                remaining = current_threshold - usage_percent
                blocks.append(self._block_context(
                    f"ℹ️ {remaining:.1f}% remaining before threshold"
                ))

            fallback = f"{status_emoji} Claude Usage: {usage_percent:.1f}% | Resets: {reset_str}"
            logger.debug(f"usage_command.sending_response fallback={fallback}")
            self.send_response(response_url, message=fallback, blocks=blocks, channel=channel)
            logger.info(f"usage_command.complete usage_percent={usage_percent:.1f}")

        except Exception as e:
            logger.error(f"usage_command.exception error={e}", exc_info=True)
            error_blocks = [
                self._block_header("❌ Error Getting Usage"),
                self._block_section(f"Failed to get usage: {str(e)}", markdown=True)
            ]
            self.send_response(response_url, message=f"Failed to get usage: {str(e)}", blocks=error_blocks, channel=channel)

    def handle_report_command(self, identifier: str, response_url: str):
        """Handle /report command - unified report handler (task/daily/project)

        Usage:
            /report              # Today's daily report
            /report 123          # Task #123 details
            /report 2025-10-22   # Specific date report
            /report <project>    # Project report
            /report --list       # List all available reports
        """
        try:
            if not self.report_generator:
                error_blocks = [
                    self._block_header("❌ Error"),
                    self._block_section("Report generator not available", markdown=True)
                ]
                self.send_response(response_url, message="Report generator not available", blocks=error_blocks)
                return

            args = identifier.strip() if identifier else ""

            # Check for --list flag
            if "--list" in args:
                daily_reports = self.report_generator.list_daily_reports()
                project_reports = self.report_generator.list_project_reports()

                blocks = [self._block_header("📊 Available Reports")]

                if daily_reports:
                    daily_list = "\n".join([f"• {report_date}" for report_date in daily_reports[:10]])
                    if len(daily_reports) > 10:
                        daily_list += f"\n• ... and {len(daily_reports) - 10} more"
                    blocks.append(self._block_section(f"*📅 Daily Reports:*\n{daily_list}", markdown=True))
                else:
                    blocks.append(self._block_section("*📅 Daily Reports:*\nNo daily reports available", markdown=True))

                if project_reports:
                    project_list = "\n".join([f"• {self._escape_slack(project_id)}" for project_id in project_reports[:10]])
                    if len(project_reports) > 10:
                        project_list += f"\n• ... and {len(project_reports) - 10} more"
                    blocks.append(self._block_section(f"*📦 Project Reports:*\n{project_list}", markdown=True))
                else:
                    blocks.append(self._block_section("*📦 Project Reports:*\nNo project reports available", markdown=True))

                fallback = f"Available: {len(daily_reports)} daily reports, {len(project_reports)} project reports"
                self.send_response(response_url, message=fallback, blocks=blocks)
                return

            # Determine if it's a date or project
            report_type = "daily"
            report_title = ""
            if not args:
                # Default: today's report
                from datetime import datetime
                date = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d")
                report = self.report_generator.get_daily_report(date)
                report_title = f"📅 Daily Report - {date}"
            else:
                # Try to parse as date
                try:
                    from datetime import datetime
                    datetime.strptime(args, "%Y-%m-%d")
                    report = self.report_generator.get_daily_report(args)
                    report_title = f"📅 Daily Report - {args}"
                except ValueError:
                    # Not a date, treat as project ID
                    report = self.report_generator.get_project_report(args)
                    report_title = f"📦 Project Report - {self._escape_slack(args)}"
                    report_type = "project"

            # Split report into sections for better formatting
            # Reports typically have lines starting with headers or separators
            max_length = 2500  # Leave room for formatting
            truncated = False

            if len(report) > max_length:
                report = report[:max_length]
                truncated = True

            # Build blocks
            blocks = [self._block_header(report_title)]

            # Split into smaller sections to avoid Slack's text block limit
            lines = report.split('\n')
            current_section = []
            section_length = 0

            for line in lines:
                line_length = len(line) + 1  # +1 for newline
                if section_length + line_length > 2900:  # Slack's limit is ~3000 per text block
                    if current_section:
                        blocks.append(self._block_section(
                            "\n".join(current_section),
                            markdown=False
                        ))
                    current_section = [line]
                    section_length = line_length
                else:
                    current_section.append(line)
                    section_length += line_length

            # Add remaining section
            if current_section:
                blocks.append(self._block_section(
                    "\n".join(current_section),
                    markdown=False
                ))

            if truncated:
                blocks.append(self._block_context("⚠️ Report truncated - use CLI for full content: `sle report`"))

            fallback = f"{report_title}\n{report}"
            self.send_response(response_url, message=fallback, blocks=blocks)

        except Exception as e:
            error_blocks = [
                self._block_header("❌ Error Getting Report"),
                self._block_section(f"Failed to get report: {str(e)}", markdown=True)
            ]
            self.send_response(response_url, message=f"Failed to get report: {str(e)}", blocks=error_blocks)
            logger.error(f"Failed to get report: {e}")

    def _block_header(self, text: str) -> dict:
        """Create a header block"""
        return {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": text
            }
        }

    def _block_divider(self) -> dict:
        """Create a divider block"""
        return {"type": "divider"}

    def _block_section(self, text: str, markdown: bool = False) -> dict:
        """Create a section block with text"""
        block = {
            "type": "section",
            "text": {
                "type": "mrkdwn" if markdown else "plain_text",
                "text": text
            }
        }
        # emoji is only valid for plain_text, not mrkdwn
        if not markdown:
            block["text"]["emoji"] = True
        return block

    def _block_section_fields(self, fields: list[dict]) -> dict:
        """Create a section block with fields

        Args:
            fields: List of dicts with 'label' and 'value' keys
        """
        field_blocks = []
        for field in fields:
            field_blocks.append({
                "type": "mrkdwn",
                "text": f"*{field['label']}*\n{field['value']}"
            })
        return {
            "type": "section",
            "fields": field_blocks
        }

    def _block_context(self, text: str) -> dict:
        """Create a context block for metadata"""
        return {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": text
                }
            ]
        }

    def _gather_status_data(self) -> dict:
        """Gather all status data for check command"""
        health = self.monitor.check_health() if self.monitor else {}
        status = str(health.get("status", "unknown"))
        status_emoji = {
            "healthy": "✅",
            "degraded": "⚠️",
            "unhealthy": "❌",
        }.get(status.lower(), "❔")

        system = health.get("system", {})

        def fmt_percent(value):
            if isinstance(value, (int, float)):
                return f"{float(value):.1f}"
            if value in (None, ""):
                return "N/A"
            return str(value)

        uptime = health.get("uptime_human", "< 1m")
        cpu_text = fmt_percent(system.get("cpu_percent"))
        mem_text = fmt_percent(system.get("memory_percent"))

        # Queue status
        queue_status = self.task_queue.get_queue_status()

        # Lifetime stats
        stats = None
        success_rate = None
        success_text = "—"
        if self.monitor:
            stats = self.monitor.get_stats()
            success_rate = stats.get("success_rate")
            success_text = f"{success_rate:.1f}%" if success_rate is not None else "—"

        # Live status entries
        live_entries = []
        if self.live_status_tracker:
            try:
                live_entries = self.live_status_tracker.entries()
            except Exception as exc:
                logger.debug(f"Live status unavailable: {exc}")
                live_entries = []

        # Tasks
        running_tasks = self.task_queue.get_in_progress_tasks()
        pending_tasks = self.task_queue.get_pending_tasks(limit=3)
        recent_tasks = self.task_queue.get_recent_tasks(limit=5)

        # Projects
        projects = self.task_queue.get_projects()
        projects_sorted = sorted(projects, key=lambda p: p["total_tasks"], reverse=True) if projects else []

        # Storage
        db = health.get("database", {})
        storage = health.get("storage", {})

        # Budget info
        budget_info = None
        if self.scheduler:
            try:
                budget_info = self.scheduler.get_credit_status()
            except Exception as exc:
                logger.debug(f"Failed to fetch credit status: {exc}")

        return {
            "status": status,
            "status_emoji": status_emoji,
            "uptime": uptime,
            "cpu_text": cpu_text,
            "mem_text": mem_text,
            "queue_status": queue_status,
            "stats": stats,
            "success_rate": success_rate,
            "success_text": success_text,
            "live_entries": live_entries,
            "running_tasks": running_tasks,
            "pending_tasks": pending_tasks,
            "recent_tasks": recent_tasks,
            "projects": projects,
            "projects_sorted": projects_sorted,
            "db": db,
            "storage": storage,
            "budget_info": budget_info,
        }

    def _build_check_blocks(self) -> list[dict]:
        """Build Block Kit blocks for status check response"""
        escape = self._escape_slack
        blocks = []

        # Gather all status data
        data = self._gather_status_data()

        # Header with status
        blocks.append(self._block_header(f"{data['status_emoji']} Sleepless Agent Status"))

        # System info
        uptime = escape(data['uptime'])
        blocks.append(self._block_section(
            f"Uptime: `{uptime}` · CPU: `{data['cpu_text']}%` · Memory: `{data['mem_text']}%`",
            markdown=True
        ))

        blocks.append(self._block_divider())

        # Queue section
        blocks.append(self._block_header("Queue"))
        queue_status = data['queue_status']
        queue_fields = [
            {"label": "Pending", "value": str(queue_status['pending'])},
            {"label": "In Progress", "value": str(queue_status['in_progress'])},
            {"label": "Completed", "value": str(queue_status['completed'])},
            {"label": "Failed", "value": str(queue_status['failed'])},
        ]
        blocks.append(self._block_section_fields(queue_fields))

        # Lifetime stats if available
        if data['stats']:
            stats = data['stats']
            lifetime_info = f"*Lifetime:* Completed `{stats['tasks_completed']}`, Failed `{stats['tasks_failed']}`, Success `{data['success_text']}`"
            if stats.get("avg_processing_time") is not None:
                lifetime_info += f" · Avg Duration `{format_duration(stats.get('avg_processing_time'))}`"
            blocks.append(self._block_section(lifetime_info, markdown=True))

        blocks.append(self._block_divider())

        # Active tasks section
        blocks.append(self._block_header("Active Tasks"))
        running_tasks = data['running_tasks']
        if running_tasks:
            for task in running_tasks[:3]:
                project = task.project_name or task.project_id or "—"
                project_text = escape(project)
                owner = f"<@{task.assigned_to}>" if task.assigned_to else "—"
                elapsed_seconds = (
                    (datetime.now(timezone.utc).replace(tzinfo=None) - task.started_at).total_seconds()
                    if task.started_at
                    else None
                )
                elapsed_text = format_duration(elapsed_seconds)
                description = escape(shorten(task.description, 80))
                task_text = f"*#{task.id}* `{project_text}` — {description}\n_Owner: {owner} · Elapsed: {elapsed_text}_"
                blocks.append(self._block_section(task_text, markdown=True))
        else:
            blocks.append(self._block_section("No active tasks", markdown=True))

        blocks.append(self._block_divider())

        # Pending tasks section
        blocks.append(self._block_header("Next Up"))
        pending_tasks = data['pending_tasks']
        if pending_tasks:
            for task in pending_tasks:
                project = task.project_name or task.project_id
                context_parts = []
                if project:
                    context_parts.append(f"`{escape(project)}`")
                context_parts.append(f"queued {relative_time(task.created_at)}")
                context = " · ".join(context_parts)
                description = escape(shorten(task.description, 80))
                priority = task.priority.value.capitalize()
                task_text = f"*#{task.id} {priority}* — {description}\n_{context}_"
                blocks.append(self._block_section(task_text, markdown=True))
        else:
            blocks.append(self._block_section("Queue is clear", markdown=True))

        # Projects section
        projects = data['projects']
        if projects:
            blocks.append(self._block_divider())
            blocks.append(self._block_header("Projects"))
            projects_sorted = data['projects_sorted']
            display_limit = 4
            for proj in projects_sorted[:display_limit]:
                name = escape(proj["project_name"] or proj["project_id"] or "—")
                proj_text = f"*{name}*\nPending: `{proj['pending']}` · Running: `{proj['in_progress']}` · Completed: `{proj['completed']}`"
                blocks.append(self._block_section(proj_text, markdown=True))
            if len(projects_sorted) > display_limit:
                blocks.append(self._block_context(f"… and {len(projects_sorted) - display_limit} more projects"))

        blocks.append(self._block_divider())

        # Storage section
        blocks.append(self._block_header("Storage"))
        db = data['db']
        storage = data['storage']
        storage_fields = []
        if db:
            if db.get("accessible"):
                storage_fields.append({
                    "label": "Database",
                    "value": f"{db.get('size_mb', 'N/A')} MB (updated {format_age_seconds(db.get('modified_ago_seconds'))})"
                })
            else:
                storage_fields.append({"label": "Database", "value": "unavailable"})
        if storage:
            if storage.get("accessible"):
                storage_fields.append({
                    "label": "Results",
                    "value": f"{storage.get('count', 0)} files · {storage.get('total_size_mb', 0)} MB"
                })
            else:
                storage_fields.append({"label": "Results", "value": "unavailable"})
        if storage_fields:
            blocks.append(self._block_section_fields(storage_fields))

        # Usage section
        budget_info = data['budget_info']

        if budget_info:
            blocks.append(self._block_divider())
            blocks.append(self._block_header("Usage"))
            budget = budget_info.get("budget", {})
            window = budget_info.get("current_window", {})

            period = "Night" if budget.get("is_nighttime") else "Day"
            remaining = budget.get("remaining_budget_usd")
            quota = budget.get("current_quota_usd")
            remaining_val = None
            quota_val = None
            try:
                if remaining is not None:
                    remaining_val = float(remaining)
                if quota is not None:
                    quota_val = float(quota)
            except (TypeError, ValueError):
                remaining_val = quota_val = None

            usage_fields = []
            if remaining_val is not None and quota_val is not None:
                usage_fields.append({
                    "label": f"{period} Period",
                    "value": f"${remaining_val:.2f} / ${quota_val:.2f}"
                })

            if window:
                executed = window.get("tasks_executed", 0) or 0
                remaining_minutes = window.get("time_remaining_minutes") or 0
                usage_fields.append({
                    "label": "Window",
                    "value": f"{executed} tasks · {format_duration(remaining_minutes * 60)} left"
                })

            if usage_fields:
                blocks.append(self._block_section_fields(usage_fields))

        blocks.append(self._block_divider())

        # Recent activity section
        blocks.append(self._block_header("Recent Activity"))
        recent_tasks = data['recent_tasks']
        if recent_tasks:
            status_icons = {
                TaskStatus.COMPLETED: "✅",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.PENDING: "🕒",
                TaskStatus.FAILED: "❌",
                TaskStatus.CANCELLED: "🗑️",
            }
            for task in recent_tasks:
                icon = status_icons.get(task.status, "•")
                description = escape(shorten(task.description, 70))
                status_label = task.status.value.replace('_', ' ')
                activity_text = f"{icon} *#{task.id}* {description}\n_{status_label} · {relative_time(task.created_at)}_"
                blocks.append(self._block_section(activity_text, markdown=True))
        else:
            blocks.append(self._block_section("No recent activity", markdown=True))

        return blocks

    def _build_check_message(self) -> str:
        escape = self._escape_slack

        # Gather all status data
        data = self._gather_status_data()

        lines: list[str] = []
        lines.append("*Sleepless Agent Status*")
        uptime = escape(data['uptime'])
        lines.append(
            f"{data['status_emoji']} *{escape(data['status'].upper())}* · "
            f"Uptime `{uptime}` · CPU `{data['cpu_text']}%` · Memory `{data['mem_text']}%`"
        )

        queue_status = data['queue_status']
        lines.append("")
        lines.append("*Queue*")
        lines.append(
            f"• Pending *{queue_status['pending']}* | "
            f"In progress *{queue_status['in_progress']}* | "
            f"Completed *{queue_status['completed']}* | "
            f"Failed *{queue_status['failed']}*"
        )

        if data['stats']:
            stats = data['stats']
            lines.append(
                f"• Lifetime: Completed *{stats['tasks_completed']}*, "
                f"Failed *{stats['tasks_failed']}*, Success {data['success_text']}"
            )
            avg_time = stats.get("avg_processing_time")
            if avg_time is not None:
                lines.append(f"• Avg Duration: {format_duration(avg_time)}")

        live_entries = data['live_entries']

        lines.append("")
        lines.append("*Live Sessions*")
        if live_entries:
            max_items = 3
            for entry in live_entries[:max_items]:
                try:
                    updated_dt = datetime.fromisoformat(entry.updated_at)
                except Exception:
                    updated_dt = None
                age_text = relative_time(updated_dt) if updated_dt else "just now"
                phase_text = escape(entry.phase.replace("_", " ").title())
                status_text = escape(entry.status.replace("_", " ").title())
                query_preview = escape(shorten(entry.prompt_preview or "—", 60))
                answer_preview = escape(shorten(entry.answer_preview or "—", 40))
                lines.append(
                    f"• #{entry.task_id} {phase_text} ({status_text}) — \"{query_preview}\" -> \"{answer_preview}\" [{age_text}]"
                )
            remaining = len(live_entries) - max_items
            if remaining > 0:
                lines.append(f"• ... {remaining} more session(s)")
        else:
            lines.append("• None")

        running_tasks = data['running_tasks']
        lines.append("")
        lines.append("*Active Tasks*")
        if running_tasks:
            for task in running_tasks[:3]:
                project = task.project_name or task.project_id or "—"
                project_text = escape(project)
                owner = f"<@{task.assigned_to}>" if task.assigned_to else "—"
                elapsed_seconds = (
                    (datetime.now(timezone.utc).replace(tzinfo=None) - task.started_at).total_seconds()
                    if task.started_at
                    else None
                )
                elapsed_text = format_duration(elapsed_seconds)
                description = escape(shorten(task.description, 80))
                lines.append(
                    f"• #{task.id} `{project_text}` — {description} "
                    f"(owner {owner}, elapsed {elapsed_text})"
                )
        else:
            lines.append("• None")

        pending_tasks = data['pending_tasks']
        lines.append("")
        lines.append("*Next Up*")
        if pending_tasks:
            for task in pending_tasks:
                project = task.project_name or task.project_id
                context_parts = []
                if project:
                    context_parts.append(f"`{escape(project)}`")
                context_parts.append(f"queued {relative_time(task.created_at)}")
                context = " · ".join(context_parts)
                description = escape(shorten(task.description, 80))
                priority = task.priority.value.capitalize()
                lines.append(f"• #{task.id} {priority} — {description} ({context})")
        else:
            lines.append("• Queue is clear")

        projects = data['projects']
        if projects:
            lines.append("")
            lines.append("*Projects*")
            projects_sorted = data['projects_sorted']
            display_limit = 4
            for proj in projects_sorted[:display_limit]:
                name = escape(proj["project_name"] or proj["project_id"] or "—")
                lines.append(
                    f"• {name} — pending {proj['pending']}, "
                    f"running {proj['in_progress']}, completed {proj['completed']}"
                )
            if len(projects_sorted) > display_limit:
                lines.append(f"• … and {len(projects_sorted) - display_limit} more")

        db = data['db']
        storage = data['storage']
        lines.append("")
        lines.append("*Storage*")
        if db:
            if db.get("accessible"):
                lines.append(
                    f"• DB: {db.get('size_mb', 'N/A')} MB "
                    f"(updated {format_age_seconds(db.get('modified_ago_seconds'))})"
                )
            else:
                lines.append("• DB: unavailable")
        if storage:
            if storage.get("accessible"):
                lines.append(
                    f"• Results: {storage.get('count', 0)} files · "
                    f"{storage.get('total_size_mb', 0)} MB"
                )
            else:
                lines.append("• Results: unavailable")

        budget_info = data['budget_info']

        if budget_info:
            budget = budget_info.get("budget", {})
            window = budget_info.get("current_window", {})
            lines.append("")
            lines.append("*Usage*")

            period = "Night" if budget.get("is_nighttime") else "Day"
            remaining = budget.get("remaining_budget_usd")
            quota = budget.get("current_quota_usd")
            remaining_val = None
            quota_val = None
            try:
                if remaining is not None:
                    remaining_val = float(remaining)
                if quota is not None:
                    quota_val = float(quota)
            except (TypeError, ValueError):
                remaining_val = quota_val = None

            if remaining_val is not None and quota_val is not None:
                lines.append(
                    f"• {period} period · Remaining ${remaining_val:.2f} / ${quota_val:.2f}"
                )

            if window:
                executed = window.get("tasks_executed", 0) or 0
                remaining_minutes = window.get("time_remaining_minutes") or 0
                lines.append(
                    f"• Window: {executed} tasks · "
                    f"{format_duration(remaining_minutes * 60)} left"
                )

        recent_tasks = data['recent_tasks']
        if recent_tasks:
            lines.append("")
            lines.append("*Recent Activity*")
            status_icons = {
                TaskStatus.COMPLETED: "✅",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.PENDING: "🕒",
                TaskStatus.FAILED: "❌",
                TaskStatus.CANCELLED: "🗑️",
            }
            for task in recent_tasks:
                icon = status_icons.get(task.status, "•")
                description = escape(shorten(task.description, 70))
                lines.append(
                    f"{icon} #{task.id} {description} — "
                    f"{task.status.value.replace('_', ' ')} ({relative_time(task.created_at)})"
                )

        return "\n".join(lines)

    def _escape_slack(self, text: Optional[str]) -> str:
        if text is None:
            return ""
        text = str(text)
        replacements = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        for char in ("*", "_", "`", "~"):
            text = text.replace(char, f"\\{char}")
        return text
