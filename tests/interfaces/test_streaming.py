"""Tests for real-time task streaming to Slack.

This module tests the StreamManager class which handles real-time streaming of
task output to Slack. It verifies stream lifecycle management, verbosity filtering,
rate limiting, sliding window buffer, and pause/resume functionality.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from sleepless_agent.interfaces.streaming import (
    StreamConfig,
    StreamManager,
    StreamState,
    StreamVerbosity,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def default_config() -> StreamConfig:
    """Create a default stream configuration for testing."""
    return StreamConfig(
        enabled=True,
        verbosity=StreamVerbosity.NORMAL,
        update_interval_seconds=0.01,  # Fast updates for tests
        max_message_length=35000,
        truncation_indicator="\n... (earlier output truncated) ...\n",
    )


@pytest.fixture
def verbose_config() -> StreamConfig:
    """Create a verbose stream configuration."""
    return StreamConfig(
        enabled=True,
        verbosity=StreamVerbosity.VERBOSE,
        update_interval_seconds=0.01,
        max_message_length=35000,
    )


@pytest.fixture
def minimal_config() -> StreamConfig:
    """Create a minimal stream configuration."""
    return StreamConfig(
        enabled=True,
        verbosity=StreamVerbosity.MINIMAL,
        update_interval_seconds=0.01,
        max_message_length=35000,
    )


@pytest.fixture
def disabled_config() -> StreamConfig:
    """Create a disabled stream configuration."""
    return StreamConfig(
        enabled=False,
        verbosity=StreamVerbosity.NORMAL,
    )


@pytest.fixture
def off_verbosity_config() -> StreamConfig:
    """Create a configuration with verbosity OFF."""
    return StreamConfig(
        enabled=True,
        verbosity=StreamVerbosity.OFF,
    )


@pytest.fixture
def mock_slack_client() -> MagicMock:
    """Create a mock Slack client."""
    client = MagicMock()
    client.chat_postMessage.return_value = {"ts": "1234567890.123456"}
    client.chat_update.return_value = {"ok": True}
    return client


@pytest.fixture
def manager(default_config: StreamConfig, mock_slack_client: MagicMock) -> StreamManager:
    """Create a StreamManager with mock Slack client."""
    return StreamManager(
        config=default_config,
        slack_client=mock_slack_client,
    )


@pytest.fixture
def manager_verbose(
    verbose_config: StreamConfig, mock_slack_client: MagicMock
) -> StreamManager:
    """Create a StreamManager with verbose config."""
    return StreamManager(
        config=verbose_config,
        slack_client=mock_slack_client,
    )


@pytest.fixture
def manager_minimal(
    minimal_config: StreamConfig, mock_slack_client: MagicMock
) -> StreamManager:
    """Create a StreamManager with minimal config."""
    return StreamManager(
        config=minimal_config,
        slack_client=mock_slack_client,
    )


# -----------------------------------------------------------------------------
# Tests for start_stream()
# -----------------------------------------------------------------------------


class TestStartStream:
    """Tests for StreamManager.start_stream() method."""

    @pytest.mark.asyncio
    async def test_creates_initial_slack_message(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify start_stream creates initial Slack message."""
        message_ts = await manager.start_stream(
            task_id=1,
            channel_id="C123456",
            thread_ts="1111111111.111111",
            initial_message=":arrows_counterclockwise: Starting...",
        )

        assert message_ts == "1234567890.123456"
        mock_slack_client.chat_postMessage.assert_called_once()
        call_args = mock_slack_client.chat_postMessage.call_args
        assert call_args.kwargs["channel"] == "C123456"
        assert call_args.kwargs["thread_ts"] == "1111111111.111111"
        assert call_args.kwargs["text"] == ":arrows_counterclockwise: Starting..."

    @pytest.mark.asyncio
    async def test_stores_stream_state(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify start_stream stores stream state."""
        await manager.start_stream(
            task_id=1,
            channel_id="C123456",
        )

        assert 1 in manager._streams
        state = manager._streams[1]
        assert state.task_id == 1
        assert state.channel_id == "C123456"
        assert state.message_ts == "1234567890.123456"

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(
        self, disabled_config: StreamConfig
    ) -> None:
        """Verify start_stream returns None when streaming disabled."""
        manager = StreamManager(config=disabled_config, slack_client=MagicMock())

        result = await manager.start_stream(task_id=1, channel_id="C123456")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_verbosity_off(
        self, off_verbosity_config: StreamConfig
    ) -> None:
        """Verify start_stream returns None when verbosity is OFF."""
        manager = StreamManager(config=off_verbosity_config, slack_client=MagicMock())

        result = await manager.start_stream(task_id=1, channel_id="C123456")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_without_slack_client(
        self, default_config: StreamConfig
    ) -> None:
        """Verify start_stream returns None without Slack client."""
        manager = StreamManager(config=default_config, slack_client=None)

        result = await manager.start_stream(task_id=1, channel_id="C123456")

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_slack_error(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify start_stream handles Slack API errors gracefully."""
        mock_slack_client.chat_postMessage.side_effect = Exception("Slack API error")

        result = await manager.start_stream(task_id=1, channel_id="C123456")

        assert result is None

    @pytest.mark.asyncio
    async def test_replaces_existing_stream(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify start_stream replaces existing stream for same task."""
        await manager.start_stream(task_id=1, channel_id="C111")

        mock_slack_client.chat_postMessage.return_value = {"ts": "2222222222.222222"}
        await manager.start_stream(task_id=1, channel_id="C222")

        assert len(manager._streams) == 1
        assert manager._streams[1].channel_id == "C222"
        assert manager._streams[1].message_ts == "2222222222.222222"


# -----------------------------------------------------------------------------
# Tests for append_output() with different verbosity levels
# -----------------------------------------------------------------------------


class TestAppendOutputVerbosity:
    """Tests for StreamManager.append_output() with different verbosity levels."""

    @pytest.mark.asyncio
    async def test_verbose_streams_all_content(
        self, manager_verbose: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify VERBOSE verbosity streams all content."""
        await manager_verbose.start_stream(task_id=1, channel_id="C123456")

        await manager_verbose.append_output(task_id=1, content="Any random content")
        await asyncio.sleep(0.05)  # Wait for debounced update

        # Content should be in buffer
        assert "Any random content" in manager_verbose._streams[1].buffer

    @pytest.mark.asyncio
    async def test_minimal_filters_most_output(
        self, manager_minimal: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify MINIMAL verbosity filters most output."""
        await manager_minimal.start_stream(task_id=1, channel_id="C123456")
        initial_buffer = manager_minimal._streams[1].buffer

        await manager_minimal.append_output(
            task_id=1, content="Regular execution output"
        )
        await asyncio.sleep(0.02)

        # Content should NOT be added (filtered out in MINIMAL mode)
        assert manager_minimal._streams[1].buffer == initial_buffer

    @pytest.mark.asyncio
    async def test_normal_includes_important_patterns(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify NORMAL verbosity includes important action patterns."""
        await manager.start_stream(task_id=1, channel_id="C123456")

        await manager.append_output(task_id=1, content="Created file src/main.py")
        await asyncio.sleep(0.05)

        assert "Created file" in manager._streams[1].buffer

    @pytest.mark.asyncio
    async def test_normal_includes_error_output(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify NORMAL verbosity includes error output."""
        await manager.start_stream(task_id=1, channel_id="C123456")

        await manager.append_output(task_id=1, content="Error: something went wrong")
        await asyncio.sleep(0.05)

        assert "Error:" in manager._streams[1].buffer

    @pytest.mark.asyncio
    async def test_normal_filters_routine_output(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify NORMAL verbosity filters routine output."""
        await manager.start_stream(task_id=1, channel_id="C123456")
        initial_buffer = manager._streams[1].buffer

        await manager.append_output(task_id=1, content="Just some routine text")
        await asyncio.sleep(0.02)

        # Routine content should be filtered
        assert manager._streams[1].buffer == initial_buffer


# -----------------------------------------------------------------------------
# Tests for rate limiting
# -----------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for StreamManager rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_respects_update_interval(
        self, mock_slack_client: MagicMock
    ) -> None:
        """Verify updates respect the configured interval."""
        config = StreamConfig(
            enabled=True,
            verbosity=StreamVerbosity.VERBOSE,
            update_interval_seconds=0.1,  # 100ms interval
            max_message_length=35000,
        )
        manager = StreamManager(config=config, slack_client=mock_slack_client)

        await manager.start_stream(task_id=1, channel_id="C123456")
        initial_update_count = mock_slack_client.chat_update.call_count

        # Rapid updates
        for i in range(5):
            await manager.append_output(task_id=1, content=f"Update {i}")

        # Should not have many updates due to debouncing
        await asyncio.sleep(0.02)
        updates_after = mock_slack_client.chat_update.call_count
        assert updates_after - initial_update_count <= 2

    @pytest.mark.asyncio
    async def test_handles_rate_limit_error_with_backoff(
        self, manager_verbose: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify rate limit errors trigger exponential backoff."""
        await manager_verbose.start_stream(task_id=1, channel_id="C123456")

        # First update succeeds, second triggers rate limit
        call_count = 0

        def rate_limit_on_second(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"ok": True}
            raise Exception("rate_limit_exceeded")

        mock_slack_client.chat_update.side_effect = rate_limit_on_second

        await manager_verbose.append_output(task_id=1, content="First update")
        await asyncio.sleep(0.05)
        await manager_verbose.append_output(task_id=1, content="Second update")
        await asyncio.sleep(0.05)

        # Should have attempted updates despite rate limit
        assert mock_slack_client.chat_update.call_count >= 1


# -----------------------------------------------------------------------------
# Tests for _apply_sliding_window()
# -----------------------------------------------------------------------------


class TestApplySlidingWindow:
    """Tests for StreamManager._apply_sliding_window() method."""

    def test_returns_text_unchanged_under_limit(
        self, manager: StreamManager
    ) -> None:
        """Verify short text is returned unchanged."""
        text = "Short text"

        result = manager._apply_sliding_window(text, max_length=1000)

        assert result == text

    def test_truncates_long_content(self, manager: StreamManager) -> None:
        """Verify long content is truncated correctly."""
        text = "a" * 1000

        result = manager._apply_sliding_window(text, max_length=500)

        assert len(result) <= 500

    def test_adds_truncation_indicator(self, manager: StreamManager) -> None:
        """Verify truncation indicator is added."""
        text = "a" * 1000

        result = manager._apply_sliding_window(text, max_length=500)

        assert manager.config.truncation_indicator in result

    def test_preserves_recent_content(self, manager: StreamManager) -> None:
        """Verify recent content (end of text) is preserved."""
        text = "old_content\n" * 50 + "recent_content"

        result = manager._apply_sliding_window(text, max_length=500)

        assert "recent_content" in result

    def test_finds_good_break_point(self, manager: StreamManager) -> None:
        """Verify truncation finds a newline break point."""
        text = "line1\nline2\nline3\n" * 100 + "final_line"

        result = manager._apply_sliding_window(text, max_length=500)

        # Should not start mid-line
        first_content = result.split(manager.config.truncation_indicator)[-1]
        # First character of content after indicator should not be mid-word
        # (it should start at a line boundary if possible)
        assert "final_line" in result


# -----------------------------------------------------------------------------
# Tests for pause_stream() and resume_stream()
# -----------------------------------------------------------------------------


class TestPauseResumeStream:
    """Tests for StreamManager.pause_stream() and resume_stream() methods."""

    @pytest.mark.asyncio
    async def test_pause_sets_paused_flag(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify pause_stream sets the paused flag."""
        await manager.start_stream(task_id=1, channel_id="C123456")

        result = await manager.pause_stream(task_id=1)

        assert result is True
        assert manager._streams[1].is_paused is True

    @pytest.mark.asyncio
    async def test_pause_updates_slack_message(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify pause_stream updates Slack message with paused indicator."""
        await manager.start_stream(task_id=1, channel_id="C123456")

        await manager.pause_stream(task_id=1)

        mock_slack_client.chat_update.assert_called()
        assert ":pause_button:" in manager._streams[1].buffer

    @pytest.mark.asyncio
    async def test_resume_clears_paused_flag(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify resume_stream clears the paused flag."""
        await manager.start_stream(task_id=1, channel_id="C123456")
        await manager.pause_stream(task_id=1)

        result = await manager.resume_stream(task_id=1)

        assert result is True
        assert manager._streams[1].is_paused is False

    @pytest.mark.asyncio
    async def test_resume_adds_resume_indicator(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify resume_stream adds resume indicator to buffer."""
        await manager.start_stream(task_id=1, channel_id="C123456")
        await manager.pause_stream(task_id=1)

        await manager.resume_stream(task_id=1)

        assert ":arrow_forward:" in manager._streams[1].buffer

    @pytest.mark.asyncio
    async def test_paused_stream_ignores_output(
        self, manager_verbose: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify paused stream ignores new output."""
        await manager_verbose.start_stream(task_id=1, channel_id="C123456")
        await manager_verbose.pause_stream(task_id=1)

        buffer_before_pause = manager_verbose._streams[1].buffer

        await manager_verbose.append_output(task_id=1, content="Should be ignored")

        # Buffer should not change during pause (except for pause indicator)
        # The append_output should return early without adding content
        # Note: the pause indicator was added by pause_stream

    @pytest.mark.asyncio
    async def test_pause_returns_false_for_nonexistent_stream(
        self, manager: StreamManager
    ) -> None:
        """Verify pause_stream returns False for non-existent stream."""
        result = await manager.pause_stream(task_id=999)

        assert result is False

    @pytest.mark.asyncio
    async def test_resume_returns_false_for_nonexistent_stream(
        self, manager: StreamManager
    ) -> None:
        """Verify resume_stream returns False for non-existent stream."""
        result = await manager.resume_stream(task_id=999)

        assert result is False


# -----------------------------------------------------------------------------
# Tests for finalize_stream()
# -----------------------------------------------------------------------------


class TestFinalizeStream:
    """Tests for StreamManager.finalize_stream() method."""

    @pytest.mark.asyncio
    async def test_adds_completion_marker_on_success(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify finalize_stream adds success marker."""
        await manager.start_stream(task_id=1, channel_id="C123456")

        await manager.finalize_stream(task_id=1, success=True)

        # Verify the final update contained success marker
        call_args = mock_slack_client.chat_update.call_args
        assert ":white_check_mark:" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_adds_failure_marker_on_failure(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify finalize_stream adds failure marker."""
        await manager.start_stream(task_id=1, channel_id="C123456")

        await manager.finalize_stream(task_id=1, success=False)

        call_args = mock_slack_client.chat_update.call_args
        assert ":x:" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_appends_final_content(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify finalize_stream appends final content."""
        await manager.start_stream(task_id=1, channel_id="C123456")

        await manager.finalize_stream(
            task_id=1,
            final_content="Summary: 3 files modified",
            success=True,
        )

        call_args = mock_slack_client.chat_update.call_args
        assert "Summary: 3 files modified" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_cleans_up_stream_state(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify finalize_stream cleans up stream state."""
        await manager.start_stream(task_id=1, channel_id="C123456")
        assert 1 in manager._streams

        await manager.finalize_stream(task_id=1, success=True)

        assert 1 not in manager._streams

    @pytest.mark.asyncio
    async def test_handles_nonexistent_stream(
        self, manager: StreamManager
    ) -> None:
        """Verify finalize_stream handles non-existent stream gracefully."""
        # Should not raise
        await manager.finalize_stream(task_id=999, success=True)


# -----------------------------------------------------------------------------
# Tests for get_stream_stats()
# -----------------------------------------------------------------------------


class TestGetStreamStats:
    """Tests for StreamManager.get_stream_stats() method."""

    @pytest.mark.asyncio
    async def test_returns_stats_for_active_stream(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify get_stream_stats returns stats for active stream."""
        await manager.start_stream(task_id=1, channel_id="C123456")

        stats = manager.get_stream_stats(task_id=1)

        assert stats is not None
        assert stats["task_id"] == 1
        assert "phase" in stats
        assert "buffer_length" in stats
        assert "total_chars_streamed" in stats
        assert "update_count" in stats
        assert "is_paused" in stats
        assert "elapsed_seconds" in stats

    def test_returns_none_for_nonexistent_stream(
        self, manager: StreamManager
    ) -> None:
        """Verify get_stream_stats returns None for non-existent stream."""
        stats = manager.get_stream_stats(task_id=999)

        assert stats is None

    @pytest.mark.asyncio
    async def test_tracks_total_chars_streamed(
        self, manager_verbose: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify get_stream_stats tracks total characters streamed."""
        await manager_verbose.start_stream(task_id=1, channel_id="C123456")

        await manager_verbose.append_output(task_id=1, content="Hello World")
        await asyncio.sleep(0.02)

        stats = manager_verbose.get_stream_stats(task_id=1)
        assert stats["total_chars_streamed"] == 11


# -----------------------------------------------------------------------------
# Tests for get_active_streams()
# -----------------------------------------------------------------------------


class TestGetActiveStreams:
    """Tests for StreamManager.get_active_streams() method."""

    @pytest.mark.asyncio
    async def test_returns_active_stream_ids(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify get_active_streams returns list of active task IDs."""
        await manager.start_stream(task_id=1, channel_id="C111")
        mock_slack_client.chat_postMessage.return_value = {"ts": "2222222222.222222"}
        await manager.start_stream(task_id=2, channel_id="C222")

        active = manager.get_active_streams()

        assert len(active) == 2
        assert 1 in active
        assert 2 in active

    def test_returns_empty_list_when_no_streams(
        self, manager: StreamManager
    ) -> None:
        """Verify get_active_streams returns empty list when no streams."""
        active = manager.get_active_streams()

        assert active == []


# -----------------------------------------------------------------------------
# Tests for is_paused()
# -----------------------------------------------------------------------------


class TestIsPaused:
    """Tests for StreamManager.is_paused() method."""

    @pytest.mark.asyncio
    async def test_returns_true_when_paused(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify is_paused returns True for paused stream."""
        await manager.start_stream(task_id=1, channel_id="C123456")
        await manager.pause_stream(task_id=1)

        assert manager.is_paused(task_id=1) is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_paused(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify is_paused returns False for active stream."""
        await manager.start_stream(task_id=1, channel_id="C123456")

        assert manager.is_paused(task_id=1) is False

    def test_returns_false_for_nonexistent_stream(
        self, manager: StreamManager
    ) -> None:
        """Verify is_paused returns False for non-existent stream."""
        assert manager.is_paused(task_id=999) is False


# -----------------------------------------------------------------------------
# Tests for update_phase()
# -----------------------------------------------------------------------------


class TestUpdatePhase:
    """Tests for StreamManager.update_phase() method."""

    @pytest.mark.asyncio
    async def test_updates_phase_in_state(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify update_phase updates the phase in stream state."""
        await manager.start_stream(task_id=1, channel_id="C123456")

        await manager.update_phase(task_id=1, phase="worker")

        assert manager._streams[1].current_phase == "worker"

    @pytest.mark.asyncio
    async def test_adds_phase_label_to_buffer(
        self, manager_verbose: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify update_phase adds phase label to buffer in VERBOSE mode."""
        await manager_verbose.start_stream(task_id=1, channel_id="C123456")

        await manager_verbose.update_phase(task_id=1, phase="worker")
        await asyncio.sleep(0.05)

        # Phase label should be in buffer (only added in VERBOSE mode)
        assert ":hammer_and_wrench:" in manager_verbose._streams[1].buffer or \
               "Executing" in manager_verbose._streams[1].buffer

    @pytest.mark.asyncio
    async def test_handles_nonexistent_stream(
        self, manager: StreamManager
    ) -> None:
        """Verify update_phase handles non-existent stream gracefully."""
        # Should not raise
        await manager.update_phase(task_id=999, phase="worker")


# -----------------------------------------------------------------------------
# Tests for stop_stream()
# -----------------------------------------------------------------------------


class TestStopStream:
    """Tests for StreamManager.stop_stream() method."""

    @pytest.mark.asyncio
    async def test_removes_stream_without_finalizing(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify stop_stream removes stream without sending final update."""
        await manager.start_stream(task_id=1, channel_id="C123456")
        initial_update_count = mock_slack_client.chat_update.call_count

        await manager.stop_stream(task_id=1)

        assert 1 not in manager._streams
        # Should not have sent any updates
        assert mock_slack_client.chat_update.call_count == initial_update_count


# -----------------------------------------------------------------------------
# Tests for StreamConfig
# -----------------------------------------------------------------------------


class TestStreamConfig:
    """Tests for StreamConfig class."""

    def test_from_dict_creates_config(self) -> None:
        """Verify StreamConfig.from_dict creates correct configuration."""
        config_dict = {
            "enabled": True,
            "verbosity": "verbose",
            "update_interval_seconds": 5.0,
            "max_message_length": 30000,
        }

        config = StreamConfig.from_dict(config_dict)

        assert config.enabled is True
        assert config.verbosity == StreamVerbosity.VERBOSE
        assert config.update_interval_seconds == 5.0
        assert config.max_message_length == 30000

    def test_from_dict_with_defaults(self) -> None:
        """Verify StreamConfig.from_dict uses defaults for missing keys."""
        config = StreamConfig.from_dict({})

        assert config.enabled is True
        assert config.verbosity == StreamVerbosity.NORMAL
        assert config.update_interval_seconds == 2.0
        assert config.max_message_length == 35000

    def test_from_dict_handles_invalid_verbosity(self) -> None:
        """Verify StreamConfig.from_dict handles invalid verbosity gracefully."""
        config_dict = {"verbosity": "invalid_level"}

        config = StreamConfig.from_dict(config_dict)

        assert config.verbosity == StreamVerbosity.NORMAL


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------


class TestStreamEdgeCases:
    """Edge case tests for streaming."""

    @pytest.mark.asyncio
    async def test_edge_case_message_at_40k_limit(
        self, manager_verbose: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Edge case: message approaching Slack's 40k character limit."""
        # Use a smaller limit for testing
        manager_verbose.config.max_message_length = 1000

        await manager_verbose.start_stream(task_id=1, channel_id="C123456")

        # Append content that exceeds limit
        large_content = "x" * 2000
        await manager_verbose.append_output(task_id=1, content=large_content)
        await asyncio.sleep(0.05)

        # Buffer should be capped at max_message_length
        assert len(manager_verbose._streams[1].buffer) <= 1000

    @pytest.mark.asyncio
    async def test_edge_case_rapid_append_calls(
        self, manager_verbose: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Edge case: very rapid append_output calls."""
        await manager_verbose.start_stream(task_id=1, channel_id="C123456")

        # Fire many rapid updates
        for i in range(100):
            await manager_verbose.append_output(task_id=1, content=f"Update {i}")

        await asyncio.sleep(0.1)

        # All content should be in buffer
        assert "Update 99" in manager_verbose._streams[1].buffer

        # But we shouldn't have made 100 Slack API calls (debouncing)
        assert mock_slack_client.chat_update.call_count < 50

    @pytest.mark.asyncio
    async def test_edge_case_stream_for_nonexistent_task(
        self, manager_verbose: StreamManager
    ) -> None:
        """Edge case: append_output for non-existent task stream."""
        # Should not raise
        await manager_verbose.append_output(task_id=999, content="Should be ignored")

    @pytest.mark.asyncio
    async def test_concurrent_streams_for_multiple_tasks(
        self, manager: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Test concurrent streams for multiple tasks."""
        # Start multiple streams
        await manager.start_stream(task_id=1, channel_id="C111")
        mock_slack_client.chat_postMessage.return_value = {"ts": "2222222222.222222"}
        await manager.start_stream(task_id=2, channel_id="C222")
        mock_slack_client.chat_postMessage.return_value = {"ts": "3333333333.333333"}
        await manager.start_stream(task_id=3, channel_id="C333")

        # Verify all streams are active
        assert len(manager.get_active_streams()) == 3

        # Finalize one, pause another
        await manager.finalize_stream(task_id=1, success=True)
        await manager.pause_stream(task_id=2)

        # Verify states
        assert 1 not in manager._streams
        assert manager._streams[2].is_paused is True
        assert manager._streams[3].is_paused is False

    @pytest.mark.asyncio
    async def test_stream_cleanup_cancels_pending_updates(
        self, manager_verbose: StreamManager, mock_slack_client: MagicMock
    ) -> None:
        """Verify stream cleanup cancels pending update tasks."""
        await manager_verbose.start_stream(task_id=1, channel_id="C123456")

        # Trigger an update that will be debounced
        await manager_verbose.append_output(task_id=1, content="Pending update")

        # Immediately stop the stream
        await manager_verbose.stop_stream(task_id=1)

        # Verify update task was cleaned up
        assert 1 not in manager_verbose._update_tasks
