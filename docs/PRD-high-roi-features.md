# PRD: sleepless-agent High-ROI Feature Suite

**Status:** Draft
**Author:** AI Product Manager
**Last Updated:** 2026-01-01
**Version:** 1.0
**Stakeholders:** Engineering, Product, DevOps

---

## Executive Summary

### Vision
Transform sleepless-agent from a capable autonomous AI daemon into an intelligent, self-improving system that learns from outcomes, communicates proactively, and adapts to user preferences. These 7 high-ROI features collectively enable the agent to work more effectively with less supervision while maintaining trust through transparency.

### Goals
1. **Increase task success rate by 25%** through feedback-driven learning and smart retries
2. **Reduce supervision overhead by 50%** via proactive notifications and optional checkpoints
3. **Accelerate task creation by 3x** with templates and codebase-aware context injection
4. **Improve user confidence** through real-time visibility into agent operations

### Current State
sleepless-agent v0.1.2 is a functional 24/7 autonomous AI daemon with:
- Multi-agent workflow (Planner -> Worker -> Evaluator)
- Slack and CLI interfaces
- Task queue with priority management
- Auto-generation when idle
- Isolated workspace management
- Git integration for commits/PRs

### Gap Analysis
The current system lacks:
- Feedback loops to learn from task outcomes
- Intelligent retry mechanisms for transient failures
- Human approval gates for critical operations
- Proactive status communication to Slack
- Templated workflows for common tasks
- Real-time streaming of agent output
- Automatic codebase context injection

---

## Problem Statement

### Current Pain Points

1. **No Learning from Outcomes**
   - Tasks complete or fail without systematic feedback collection
   - Auto-generation has no signal about what tasks are valuable
   - Same mistakes are repeated across similar tasks

2. **Blind Failures**
   - Failed tasks remain failed without retry attempts
   - No distinction between transient (API limits) vs. substantive (wrong approach) failures
   - Users discover failures only when checking status

3. **Trust Gap**
   - Users lack visibility into what the agent is doing
   - Critical operations (commits, PRs) happen without approval option
   - Progress is invisible until task completion

4. **Repetitive Task Creation**
   - Common workflows require manual description each time
   - No reuse of proven task patterns
   - Context that exists in codebase must be manually provided

---

## Success Metrics

### North Star Metric
**User Engagement Score:** Composite of tasks created, feedback provided, and time-to-first-intervention (target: 40% improvement)

### Primary Metrics
| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| Task Success Rate | ~60% | 75% | 3 months |
| Manual Intervention Rate | ~40% | 20% | 3 months |
| Time to Task Creation | 45 sec | 15 sec | 2 months |
| Feedback Collection Rate | 0% | 30% | 2 months |

### Guardrail Metrics (Do Not Degrade)
- Task execution latency (< 10% increase)
- System reliability (99.5% uptime)
- Claude Pro usage efficiency

---

## Feature Specifications

---

### Feature 1: Task Success Feedback Loop

#### Description
Enable users to provide feedback on task outcomes via Slack reactions, then use this feedback to improve auto-generation strategies and task execution patterns.

#### User Value
- Tasks become more valuable over time as the system learns preferences
- Auto-generated tasks align better with user expectations
- Provides sense of control and influence over agent behavior

#### Functional Requirements

**FR-1.1: Reaction Handler**
- Priority: P0
- Listen for reactions on task completion messages in Slack
- Capture thumbs-up and thumbs-down reactions
- Map reactions to task IDs via message metadata

**FR-1.2: Feedback Storage**
- Priority: P0
- Store feedback with full task context (description, project, priority, outcome)
- Schema: `task_feedback` table with `task_id`, `reaction`, `user_id`, `timestamp`
- Retain task context snapshot for analysis

**FR-1.3: Feedback-Weighted Generation**
- Priority: P1
- Query recent feedback when generating tasks
- Weight prompt selection toward patterns with positive feedback
- Suppress patterns consistently receiving negative feedback

**FR-1.4: Feedback Analytics**
- Priority: P2
- Track feedback trends per project, task type, priority
- Expose via `/think status` or dedicated command

#### Acceptance Criteria
- [ ] User can react with thumbs-up/down on task completion messages
- [ ] Reactions are stored within 5 seconds of being added
- [ ] Auto-generator queries feedback when selecting prompts
- [ ] Positive feedback increases weight for similar task patterns
- [ ] Negative feedback decreases weight (minimum floor of 0.1)

#### Technical Approach
```
Files to modify:
- src/sleepless_agent/interfaces/bot.py
  - Add reaction event handler in handle_events_api()
  - Map reaction to task via message metadata

- src/sleepless_agent/core/models.py
  - Add TaskFeedback model class

- src/sleepless_agent/storage/sqlite.py
  - Add feedback storage methods

- src/sleepless_agent/scheduling/auto_generator.py
  - Query feedback in _select_prompt()
  - Adjust weights based on feedback history
```

#### Effort Estimate: M (Medium)
- Reaction handling: S
- Storage schema: S
- Feedback integration: M

#### Dependencies
- None (can start immediately)

---

### Feature 2: Smart Task Retry with Learning

#### Description
Automatically retry failed tasks with intelligent analysis of failure causes, refined prompts, and configurable retry strategies. Learn from repeated failures to avoid similar problematic tasks.

#### User Value
- Transient failures (rate limits, timeouts) self-heal
- Higher overall task completion rate
- Reduced need for manual re-submission

#### Functional Requirements

**FR-2.1: Failure Classification**
- Priority: P0
- Analyze error messages to classify as transient vs. substantive
- Transient: rate limits, network errors, timeouts, resource exhaustion
- Substantive: invalid approach, missing dependencies, logic errors

**FR-2.2: Retry Strategy**
- Priority: P0
- Configurable max retries (default: 3)
- Exponential backoff for transient failures
- Prompt refinement for substantive failures

**FR-2.3: Prompt Refinement**
- Priority: P1
- On substantive failure, append error context to original prompt
- Include guidance: "Previous attempt failed with: {error}. Try a different approach."
- Track refinement attempts

**FR-2.4: Failure Pattern Learning**
- Priority: P2
- Track failure patterns across tasks (same error, similar descriptions)
- Suppress auto-generation of tasks matching failure patterns
- Decay suppression over time (allow retry after 7 days)

#### Acceptance Criteria
- [ ] Failed tasks with transient errors auto-retry with backoff
- [ ] Substantive failures include error context in retry prompt
- [ ] Max 3 retry attempts per task (configurable)
- [ ] Retry count tracked in task.attempt_count (existing field)
- [ ] Similar failing patterns suppressed in auto-generation

#### Technical Approach
```
Files to modify:
- src/sleepless_agent/core/task_runtime.py
  - Add retry logic in execute() after failure
  - Classify failure type from error message

- src/sleepless_agent/core/queue.py
  - Add method to requeue task with retry context

- src/sleepless_agent/core/models.py
  - Add FailurePattern model for learning

- src/sleepless_agent/scheduling/auto_generator.py
  - Check failure patterns before generating similar tasks

- src/sleepless_agent/config.yaml
  - Add retry configuration section
```

#### Effort Estimate: M (Medium)
- Failure classification: S
- Retry mechanism: M
- Learning system: M

#### Dependencies
- Feature 1 (feedback can inform whether retries were successful)

---

### Feature 3: Human-in-the-Loop Checkpoints

#### Description
Add optional approval gates at critical phases of task execution, allowing users to review and approve before significant actions (commits, PRs) occur.

#### User Value
- Maintains control over critical operations
- Builds trust in autonomous operations
- Catches issues before they reach repositories
- Configurable per-project for different risk levels

#### Functional Requirements

**FR-3.1: Checkpoint Types**
- Priority: P0
- Post-planning: "Here's my plan for task #{id}. Approve?"
- Pre-commit: "About to commit {n} files. OK?"
- Pre-PR: "Ready to create PR: {title}. Review?"

**FR-3.2: Approval Flow**
- Priority: P0
- Post checkpoint message to Slack thread
- Include Approve/Reject buttons (Block Kit)
- Wait for response with configurable timeout (default: 1 hour)
- Timeout behavior: configurable (proceed, abort, notify)

**FR-3.3: Configuration**
- Priority: P0
- Global defaults in config.yaml
- Per-project overrides via project metadata
- Checkpoint types independently toggleable

**FR-3.4: Checkpoint State Management**
- Priority: P1
- Persist checkpoint state for daemon restart resilience
- Resume waiting after restart
- Clear stale checkpoints after 24 hours

#### Acceptance Criteria
- [ ] Each checkpoint type can be enabled/disabled independently
- [ ] Slack message includes Approve/Reject interactive buttons
- [ ] Approval within timeout resumes execution
- [ ] Rejection cancels task with reason logged
- [ ] Timeout follows configured behavior
- [ ] Configuration supports global and per-project overrides

#### Technical Approach
```
Files to modify:
- src/sleepless_agent/interfaces/bot.py
  - Add interactive message handler for button actions
  - Add send_checkpoint_message() method

- src/sleepless_agent/core/executor.py
  - Insert checkpoint calls in execute_task()
  - After _execute_planner_phase() for post-planning
  - Before git commit in _maybe_commit_changes() style

- src/sleepless_agent/core/models.py
  - Add Checkpoint model for persistence

- src/sleepless_agent/config.yaml
  - Add checkpoints configuration section

New file:
- src/sleepless_agent/core/checkpoints.py
  - CheckpointManager class
  - Async wait_for_approval() method
```

#### Effort Estimate: L (Large)
- Checkpoint infrastructure: M
- Slack interactive buttons: M
- State persistence: S
- Configuration: S

#### Dependencies
- None (independent feature)

---

### Feature 4: Proactive Progress Notifications

#### Description
Push real-time progress updates to Slack threads as tasks execute, including phase transitions, milestones, and blockers.

#### User Value
- Visibility into long-running tasks
- Early awareness of blockers
- Confidence that the agent is making progress
- Context for when to intervene

#### Functional Requirements

**FR-4.1: Phase Transition Notifications**
- Priority: P0
- Notify on: initializing -> planner -> worker -> evaluator -> completed
- Include phase name and brief status
- Post to task's originating Slack thread (if exists)

**FR-4.2: Milestone Notifications**
- Priority: P1
- "Created {n} files: {file_list}"
- "Executed {n} commands successfully"
- "Made {n} API calls"
- Batch similar events (max 1 notification per type per minute)

**FR-4.3: Blocker Alerts**
- Priority: P0
- Rate limit hit: "Paused for {n} minutes due to rate limit"
- Resource wait: "Waiting for {resource}"
- Error encountered: "Hit issue: {brief_error}"

**FR-4.4: Heartbeat for Long Tasks**
- Priority: P2
- If no update for 5 minutes, send heartbeat
- "Still working on phase {phase}... ({elapsed} elapsed)"
- Configurable interval

**FR-4.5: Notification Preferences**
- Priority: P1
- Verbosity levels: minimal, normal, verbose
- Configurable per-project
- Respect quiet hours (night mode)

#### Acceptance Criteria
- [ ] Phase transitions trigger Slack thread updates
- [ ] File creation milestones batched and reported
- [ ] Rate limit blocks trigger immediate notification
- [ ] Heartbeat sent every 5 minutes for long tasks
- [ ] Verbosity configurable (minimal/normal/verbose)
- [ ] Thread context preserved from task creation

#### Technical Approach
```
Files to modify:
- src/sleepless_agent/interfaces/bot.py
  - Add send_progress_update() method
  - Store thread_ts in task context

- src/sleepless_agent/core/executor.py
  - Call notification hooks at phase boundaries
  - Integrate with existing _live_update() pattern

- src/sleepless_agent/core/task_runtime.py
  - Add notification calls for completion/failure

- src/sleepless_agent/utils/live_status.py
  - Extend to push to Slack (not just file storage)

New file:
- src/sleepless_agent/monitoring/notifications.py
  - NotificationManager class
  - Batching logic for milestones
  - Heartbeat scheduler

- src/sleepless_agent/config.yaml
  - Add notifications configuration section
```

#### Effort Estimate: M (Medium)
- Phase notifications: S
- Milestones with batching: M
- Heartbeat: S
- Configuration: S

#### Dependencies
- Requires task -> Slack thread mapping (already exists in context)

---

### Feature 5: Task Template Library

#### Description
Pre-defined templates for common workflows that can be invoked with a simple command, reducing repetitive task descriptions and ensuring best practices.

#### User Value
- Faster task creation for common patterns
- Consistent quality for templated workflows
- Extensible for project-specific templates
- Reduces learning curve for new users

#### Functional Requirements

**FR-5.1: Built-in Templates**
- Priority: P0
- `add-tests`: Add unit tests for specified file/module
- `code-review`: Review code for issues, suggest improvements
- `refactor`: Refactor specified code for clarity/performance
- `document`: Add/improve documentation for specified code
- `debug`: Investigate and fix specified issue

**FR-5.2: Template Invocation**
- Priority: P0
- CLI: `sleepless task --template=add-tests src/module.py`
- Slack: `/think --template=add-tests src/module.py`
- Template name validated against available templates

**FR-5.3: Template Format (YAML)**
- Priority: P0
- Name, description, category
- Prompt template with placeholders
- Default priority
- Optional: required parameters, validation

**FR-5.4: Custom Templates**
- Priority: P1
- Load from `~/.sleepless/templates/` or project `.sleepless/templates/`
- Override built-in templates with custom versions
- Template listing command: `/think templates`

**FR-5.5: Template Expansion**
- Priority: P0
- Substitute placeholders with provided arguments
- Inject codebase context if template requests it
- Validate required parameters before execution

#### Acceptance Criteria
- [ ] 5 built-in templates available out of box
- [ ] Templates invokable via CLI and Slack
- [ ] YAML format supports name, description, prompt, placeholders
- [ ] Custom templates loaded from user directories
- [ ] `/think templates` lists available templates
- [ ] Invalid template name returns helpful error

#### Technical Approach
```
New files:
- src/sleepless_agent/templates/
  - __init__.py
  - loader.py (TemplateLoader class)
  - registry.py (TemplateRegistry singleton)
  - builtin/
    - add_tests.yaml
    - code_review.yaml
    - refactor.yaml
    - document.yaml
    - debug.yaml

Files to modify:
- src/sleepless_agent/interfaces/bot.py
  - Parse --template flag in handle_think_command()
  - Add handle_templates_command()

- src/sleepless_agent/interfaces/cli.py
  - Add --template option to command_task()

- src/sleepless_agent/tasks/utils.py
  - Add expand_template() function
```

Template YAML structure:
```yaml
name: add-tests
description: Add unit tests for specified file or module
category: testing
priority: serious
parameters:
  - name: target
    required: true
    description: File or module path to test
prompt: |
  Create comprehensive unit tests for the following file:

  Target: {target}

  Requirements:
  - Use the project's existing test framework (detect from existing tests)
  - Cover happy paths and edge cases
  - Include docstrings explaining each test
  - Follow existing test naming conventions
```

#### Effort Estimate: M (Medium)
- Template format and loader: S
- Built-in templates: S
- CLI/Slack integration: M
- Custom template support: S

#### Dependencies
- Feature 7 (templates benefit from codebase-aware context)

---

### Feature 6: Real-Time Task Streaming

#### Description
Stream Claude's output to Slack in real-time as the agent works, showing thinking and actions as they happen.

#### User Value
- Immediate visibility into agent reasoning
- Catch issues early without waiting for completion
- Educational: users learn how agent approaches problems
- Engaging: feels interactive rather than batch

#### Functional Requirements

**FR-6.1: Stream to Slack**
- Priority: P0
- Stream planner, worker, evaluator output to Slack thread
- Update existing message (not spam new messages)
- Configurable update interval (default: 2 seconds)

**FR-6.2: Verbosity Levels**
- Priority: P1
- `off`: No streaming, only final result
- `minimal`: Phase changes only
- `normal`: Phase changes + key actions (file edits, commands)
- `verbose`: Full output stream

**FR-6.3: Buffer Management**
- Priority: P0
- Respect Slack's 40k character message limit
- Truncate old content, keep recent (sliding window)
- Add "... (truncated) ..." indicator

**FR-6.4: Rate Limit Handling**
- Priority: P0
- Buffer updates if approaching Slack API limits
- Batch rapid updates into single message edit
- Graceful degradation if rate limited

**FR-6.5: Stream Toggle**
- Priority: P1
- User can toggle streaming mid-task
- "Pause streaming" / "Resume streaming" buttons

#### Acceptance Criteria
- [ ] Agent output streams to Slack thread in real-time
- [ ] Message updates don't exceed Slack character limits
- [ ] Updates batched to avoid Slack rate limits
- [ ] Verbosity configurable per-task or globally
- [ ] Stream can be paused/resumed by user

#### Technical Approach
```
Files to modify:
- src/sleepless_agent/interfaces/bot.py
  - Add StreamManager integration
  - Handle pause/resume button actions

- src/sleepless_agent/core/executor.py
  - Hook _live_update() to also push to StreamManager
  - Pass Slack context (channel, thread_ts) to phases

New file:
- src/sleepless_agent/interfaces/streaming.py
  - StreamManager class
  - Sliding window buffer
  - Rate-limited message updater
  - Verbosity filtering

- src/sleepless_agent/config.yaml
  - Add streaming configuration section
```

#### Effort Estimate: M (Medium)
- StreamManager: M
- Slack integration: S
- Buffer management: S
- Rate limiting: S

#### Dependencies
- Feature 4 (shares Slack thread infrastructure)

---

### Feature 7: Codebase-Aware Context Injection

#### Description
Automatically inject relevant codebase context into task prompts, including repository structure, conventions, and recent history.

#### User Value
- Tasks execute with full codebase awareness
- Agent follows existing patterns automatically
- Reduces need for users to specify conventions
- Higher quality, more consistent output

#### Functional Requirements

**FR-7.1: Repository Structure Injection**
- Priority: P0
- Generate directory tree (configurable depth)
- Identify key files (README, package.json, pyproject.toml, etc.)
- Include file type distribution summary

**FR-7.2: Convention Extraction**
- Priority: P1
- Parse config files for project conventions
- Detect test framework (pytest, jest, etc.)
- Detect linting/formatting tools
- Extract from existing code patterns

**FR-7.3: Git History Context**
- Priority: P1
- Recent commits (last N, default 10)
- Active branches
- Uncommitted changes summary
- Contributors context

**FR-7.4: Pattern Analysis**
- Priority: P2
- Detect existing code patterns (naming, structure)
- Identify related files for context
- Smart file selection based on task description

**FR-7.5: Context Caching**
- Priority: P1
- Cache extracted context (TTL: 1 hour)
- Invalidate on git changes
- Lazy loading (generate only if template requests)

#### Acceptance Criteria
- [ ] Repository structure injected into prompts for REFINE tasks
- [ ] Conventions extracted from config files automatically
- [ ] Recent git history included in context
- [ ] Context cached with 1-hour TTL
- [ ] Context size respects token budget (max 4000 tokens)

#### Technical Approach
```
New file:
- src/sleepless_agent/context/
  - __init__.py
  - extractor.py (ContextExtractor class)
  - cache.py (ContextCache class)
  - analyzers/
    - structure.py (directory tree)
    - conventions.py (config parsing)
    - git.py (history extraction)
    - patterns.py (code pattern detection)

Files to modify:
- src/sleepless_agent/core/executor.py
  - Call ContextExtractor before each phase
  - Inject context into prompts
  - Reference existing _read_workspace_context() method

- src/sleepless_agent/scheduling/auto_generator.py
  - Use context in _gather_codebase_context() (already exists)
  - Extend with new extractors
```

Context injection format:
```markdown
## Repository Context

### Structure
```
project/
  src/
    module_a/
    module_b/
  tests/
  docs/
```

### Conventions
- Test Framework: pytest
- Linter: ruff
- Formatter: black
- Python Version: 3.11+

### Recent Activity
- 3 commits in last 24h
- Last commit: "Add user authentication" (2h ago)
- Active branch: feature/new-api

### Relevant Files
- src/module_a/handler.py (similar naming)
- tests/test_module_a.py (existing tests)
```

#### Effort Estimate: L (Large)
- Structure extraction: S
- Convention detection: M
- Git history: S
- Pattern analysis: L
- Caching: S

#### Dependencies
- None (independent feature, but enhances Feature 5)

---

## User Stories

### US-1: Feedback Provider
**As a** user who submitted a task
**I want to** react to the completion message with thumbs-up/down
**So that** the agent learns what tasks are valuable to me

**Acceptance Criteria:**
- Completion message has clear call-to-action for feedback
- Reaction is acknowledged with brief reply
- My feedback influences future auto-generation

### US-2: Hands-Off Operator
**As a** user who wants minimal intervention
**I want** failed tasks to automatically retry with refined prompts
**So that** transient failures don't require my attention

**Acceptance Criteria:**
- Rate limit failures retry after backoff
- Logic failures retry with error context
- I'm notified only if all retries exhaust

### US-3: Cautious Approver
**As a** user managing a production codebase
**I want to** approve commits and PRs before they're created
**So that** I maintain control over repository changes

**Acceptance Criteria:**
- I can enable checkpoints per-project
- Approve/Reject buttons appear in thread
- Timeout behavior is configurable

### US-4: Progress Watcher
**As a** user with a long-running task
**I want to** see progress updates in Slack
**So that** I know the agent is working and can spot issues early

**Acceptance Criteria:**
- Phase transitions appear in thread
- Blockers are immediately surfaced
- Heartbeat confirms activity on long tasks

### US-5: Template User
**As a** user who frequently runs similar tasks
**I want to** invoke pre-defined templates
**So that** I can create tasks quickly without repetitive typing

**Acceptance Criteria:**
- `/think --template=add-tests file.py` works
- Template list available via `/think templates`
- Custom templates loadable from my config

### US-6: Real-Time Observer
**As a** user curious about agent reasoning
**I want to** see Claude's output streamed in real-time
**So that** I can learn from and intervene in the process

**Acceptance Criteria:**
- Output appears in Slack as it's generated
- I can pause/resume streaming
- Verbosity is adjustable

### US-7: Context-Aware Task Creator
**As a** user working in an established codebase
**I want** the agent to automatically understand my conventions
**So that** generated code matches existing patterns

**Acceptance Criteria:**
- Agent detects my test framework
- Generated tests follow existing naming
- Recent git history informs task context

---

## Technical Architecture

### Integration Points

```
                                    +------------------+
                                    |   Slack API      |
                                    +--------+---------+
                                             |
                 +---------------------------+---------------------------+
                 |                           |                           |
         +-------v-------+           +-------v-------+           +-------v-------+
         | Reaction      |           | Interactive   |           | Streaming     |
         | Handler (F1)  |           | Buttons (F3)  |           | Manager (F6)  |
         +-------+-------+           +-------+-------+           +-------+-------+
                 |                           |                           |
                 v                           v                           v
         +-------+-------+           +-------+-------+           +-------+-------+
         | Feedback      |           | Checkpoint    |           | Notification  |
         | Storage       |           | Manager       |           | Manager (F4)  |
         +-------+-------+           +-------+-------+           +-------+-------+
                 |                           |                           |
                 +-------------+-------------+---------------------------+
                               |
                       +-------v-------+
                       |  TaskQueue    |
                       |  (enhanced)   |
                       +-------+-------+
                               |
                       +-------v-------+
                       | TaskRuntime   |
                       | (retry F2)    |
                       +-------+-------+
                               |
                       +-------v-------+
                       | Executor      |
                       | (context F7)  |
                       +-------+-------+
                               |
                       +-------v-------+
                       | Template      |
                       | Registry (F5) |
                       +---------------+
```

### Data Model Changes

```python
# New models in src/sleepless_agent/core/models.py

class TaskFeedback(Base):
    """Store user feedback on task outcomes"""
    __tablename__ = "task_feedback"

    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    user_id = Column(String(100), nullable=False)
    reaction = Column(String(20), nullable=False)  # thumbs_up, thumbs_down
    created_at = Column(DateTime, default=datetime.utcnow)
    context_snapshot = Column(Text, nullable=True)  # JSON of task at feedback time


class FailurePattern(Base):
    """Track failure patterns for learning"""
    __tablename__ = "failure_patterns"

    id = Column(Integer, primary_key=True)
    pattern_hash = Column(String(64), unique=True)  # Hash of normalized error
    error_type = Column(String(50))  # transient, substantive
    occurrences = Column(Integer, default=1)
    last_seen = Column(DateTime)
    suppressed_until = Column(DateTime, nullable=True)


class Checkpoint(Base):
    """Track pending approval checkpoints"""
    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    checkpoint_type = Column(String(50))  # post_plan, pre_commit, pre_pr
    status = Column(String(20))  # pending, approved, rejected, expired
    message_ts = Column(String(50))  # Slack message timestamp
    created_at = Column(DateTime)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String(100), nullable=True)
```

### Configuration Schema

```yaml
# Additions to config.yaml

checkpoints:
  enabled: true
  global_defaults:
    post_plan: false
    pre_commit: true
    pre_pr: true
  timeout_minutes: 60
  timeout_behavior: notify  # proceed, abort, notify

retry:
  enabled: true
  max_attempts: 3
  backoff_base_seconds: 60
  backoff_multiplier: 2
  transient_patterns:
    - "rate limit"
    - "timeout"
    - "connection reset"

notifications:
  enabled: true
  verbosity: normal  # minimal, normal, verbose
  heartbeat_interval_minutes: 5
  quiet_hours:
    start: 22
    end: 7

streaming:
  enabled: true
  verbosity: normal
  update_interval_seconds: 2
  max_message_length: 35000

templates:
  builtin_enabled: true
  custom_paths:
    - ~/.sleepless/templates/
    - .sleepless/templates/

context_injection:
  enabled: true
  cache_ttl_minutes: 60
  max_tokens: 4000
  include:
    structure: true
    conventions: true
    git_history: true
    patterns: true
```

---

## Implementation Iterations

### Iteration 1: Foundation (Weeks 1-2)
**Theme:** Core feedback and retry infrastructure

**Features:**
- F1: Task Success Feedback Loop (core)
- F2: Smart Task Retry (transient failures only)

**Goals:**
- Establish feedback collection pipeline
- Implement basic retry for rate limits/timeouts
- Create database schema for new models

**Acceptance Criteria:**
- [ ] Reactions captured and stored
- [ ] Transient failures trigger automatic retry
- [ ] Retry count visible in task status

**Deliverables:**
- TaskFeedback model and storage
- Reaction handler in bot.py
- RetryManager with backoff logic
- Database migrations

---

### Iteration 2: Control & Visibility (Weeks 3-4)
**Theme:** User control and awareness

**Features:**
- F3: Human-in-the-Loop Checkpoints
- F4: Proactive Progress Notifications

**Goals:**
- Enable approval gates for critical operations
- Push phase transitions to Slack threads
- Add blocker alerts

**Acceptance Criteria:**
- [ ] Pre-commit checkpoint working with Approve/Reject buttons
- [ ] Phase changes appear in Slack thread
- [ ] Rate limit hits trigger immediate notification

**Deliverables:**
- CheckpointManager class
- Interactive button handler
- NotificationManager class
- Heartbeat scheduler

**Dependencies:**
- Iteration 1 (uses feedback for notification preferences)

---

### Iteration 3: Acceleration (Weeks 5-6)
**Theme:** Faster, smarter task creation

**Features:**
- F5: Task Template Library
- F7: Codebase-Aware Context Injection (core)

**Goals:**
- Ship 5 built-in templates
- Auto-inject repo structure into prompts
- Extract and use project conventions

**Acceptance Criteria:**
- [ ] `/think --template=add-tests file.py` works
- [ ] Templates listed via `/think templates`
- [ ] REFINE tasks include repo structure
- [ ] Conventions detected from config files

**Deliverables:**
- Template loader and registry
- 5 built-in template YAML files
- ContextExtractor class
- Convention analyzers

**Dependencies:**
- None (can parallel with Iteration 2)

---

### Iteration 4: Polish & Learning (Weeks 7-8)
**Theme:** Advanced features and learning systems

**Features:**
- F2: Smart Task Retry (prompt refinement, learning)
- F1: Task Success Feedback Loop (weighted generation)
- F7: Codebase-Aware Context (patterns, caching)

**Goals:**
- Complete retry with prompt refinement
- Implement feedback-weighted auto-generation
- Add pattern analysis for context

**Acceptance Criteria:**
- [ ] Substantive failures include error context in retry
- [ ] Positive feedback increases prompt weight
- [ ] Failure patterns suppress similar auto-generation
- [ ] Context cached with 1-hour TTL

**Deliverables:**
- FailurePattern model and learning logic
- Feedback-weighted prompt selection
- Pattern analyzer
- Context cache

**Dependencies:**
- Iteration 1, 3

---

### Iteration 5: Real-Time Experience (Weeks 9-10)
**Theme:** Live visibility and streaming

**Features:**
- F6: Real-Time Task Streaming
- F4: Proactive Notifications (heartbeat, verbosity)
- F3: Checkpoints (stream toggle)

**Goals:**
- Stream agent output to Slack in real-time
- Add verbosity controls
- Enable stream pause/resume

**Acceptance Criteria:**
- [ ] Output streams to Slack as generated
- [ ] Verbosity levels work (minimal/normal/verbose)
- [ ] Pause/Resume buttons functional
- [ ] Buffer respects Slack limits

**Deliverables:**
- StreamManager class
- Sliding window buffer
- Rate-limited message updater
- Stream control buttons

**Dependencies:**
- Iteration 2 (notification infrastructure)

---

## Risks and Mitigations

### R1: Slack API Rate Limits
**Likelihood:** High | **Impact:** Medium
**Description:** Heavy streaming/notifications may hit Slack rate limits
**Mitigation:**
- Implement robust rate limiting in StreamManager
- Buffer updates and batch when approaching limits
- Graceful degradation (reduce verbosity under pressure)
- Monitor rate limit headers and adapt

### R2: Feedback Spam
**Likelihood:** Medium | **Impact:** Low
**Description:** Users may over-react or react inconsistently
**Mitigation:**
- Only count first reaction per user per task
- Decay feedback weight over time (recent = stronger)
- Require minimum feedback count before adjusting weights
- Admin override for weight resets

### R3: Checkpoint Abandonment
**Likelihood:** Medium | **Impact:** Medium
**Description:** Users may forget to approve, blocking tasks
**Mitigation:**
- Configurable timeout with sensible defaults
- Reminder notification before timeout
- Configurable timeout behavior (proceed/abort/notify)
- Daily cleanup of stale checkpoints

### R4: Context Token Budget
**Likelihood:** Medium | **Impact:** Low
**Description:** Injected context may consume excessive tokens
**Mitigation:**
- Strict token budget (max 4000 tokens)
- Priority ordering (structure > conventions > history)
- Smart truncation with "..." indicators
- Per-phase budget allocation

### R5: Template Maintenance
**Likelihood:** Low | **Impact:** Low
**Description:** Built-in templates may become stale
**Mitigation:**
- Version templates with PRD version
- Community contribution path for templates
- Template effectiveness tracking via feedback
- Regular review cycle (quarterly)

### R6: Retry Loops
**Likelihood:** Medium | **Impact:** High
**Description:** Tasks may retry infinitely on persistent issues
**Mitigation:**
- Hard cap on retries (max 3)
- Failure pattern learning suppresses similar tasks
- Exponential backoff increases wait time
- Alert after max retries exhausted

---

## Future Considerations

### Phase 2 Enhancements

1. **Multi-Task Orchestration**
   - Chain templates into workflows
   - Parallel task execution with coordination
   - Cross-task context sharing

2. **Advanced Learning**
   - ML-based feedback analysis
   - Prompt optimization from success patterns
   - Personalized task suggestions

3. **Team Features**
   - Shared template libraries
   - Team feedback aggregation
   - Role-based checkpoint permissions

4. **Integration Expansion**
   - GitHub Actions integration
   - Jira/Linear ticket sync
   - IDE plugin for direct invocation

5. **Analytics Dashboard**
   - Task success trends
   - Feedback patterns visualization
   - Cost/efficiency metrics
   - Template usage statistics

### Technical Debt to Address

1. **Unified Notification System**
   - Consolidate notification patterns across features
   - Abstract Slack-specific logic for multi-platform

2. **Context Extraction Optimization**
   - AST-based convention extraction
   - Incremental git history updates

3. **Testing Infrastructure**
   - Integration tests for Slack flows
   - Mocked feedback simulation
   - Template validation tests

---

## Appendix

### Open Questions

1. **Feedback granularity:** Should we support more reactions beyond thumbs-up/down? (e.g., heart for "exceptional", confused for "unclear")

2. **Checkpoint scope:** Should checkpoints apply to auto-generated tasks or only user-submitted?

3. **Template versioning:** How to handle breaking changes to template format?

4. **Streaming storage:** Should streamed output be persisted for replay/debugging?

### References

- [sleepless-agent Repository](./README.md)
- [Architecture Documentation](./docs/concepts/architecture.md)
- [Task Lifecycle](./docs/concepts/task-lifecycle.md)
- [Slack Setup Guide](./docs/guides/slack-setup.md)

### Glossary

| Term | Definition |
|------|------------|
| Checkpoint | Approval gate before critical operation |
| Transient Failure | Temporary error that may resolve on retry |
| Substantive Failure | Error requiring different approach |
| Template | Pre-defined task pattern with placeholders |
| Context Injection | Auto-adding codebase info to prompts |
| Feedback Loop | System for learning from user reactions |

---

*PRD Version: 1.0*
*Next Review: 2026-02-01*
*Approved By: [Pending]*
