# PRD: Project Orchestrator

**Status:** Draft
**Author:** AI Product Manager
**Last Updated:** 2026-01-01
**Version:** 1.0
**Stakeholders:** Engineering, Product, DevOps

---

## Executive Summary

### Vision
Transform sleepless-agent from a reactive task-execution engine into a proactive project-aware system that autonomously identifies, prioritizes, and executes work across multiple projects.

### Goals
1. **Eliminate manual task creation** for ongoing project maintenance and improvement
2. **Increase project velocity** by having the agent continuously identify and address technical debt, bugs, and opportunities
3. **Provide project-level visibility** into what work is being done and why
4. **Enable multi-project management** from a single sleepless-agent instance

### Current State
sleepless-agent v0.1.2 has:
- Task-centric architecture with `project_id` as a passive attribute
- Auto-generation creates generic tasks without project context
- Projects exist as containers (workspaces, git branches) but not active entities
- No project-level goals, state analysis, or priority management

### Gap Analysis
The current system lacks:
- Project configuration (goals, constraints, repo links)
- Project state analysis (what needs attention?)
- Project-aware task generation (tasks aligned with goals)
- Multi-project prioritization (which project needs attention first?)
- Dual-source context (local directory + GitHub repo)

---

## Problem Statement

### Current Pain Points

1. **Reactive Task Creation**
   - User must manually identify what needs work in each project
   - No systematic way to discover technical debt or improvement opportunities
   - Valuable maintenance work gets deferred until someone remembers to create a task

2. **Project Blindness**
   - Agent doesn't know project goals or constraints
   - Generated tasks may conflict with project priorities
   - No visibility into overall project health

3. **Multi-Project Friction**
   - Managing multiple projects requires context switching
   - Each project needs separate task creation workflow
   - No cross-project prioritization

4. **Underutilized Signals**
   - GitHub issues/PRs exist but aren't consumed
   - Test failures, TODOs, coverage reports go unnoticed
   - Valuable signals about project health are ignored

---

## Success Metrics

### North Star Metric
**Projects on Autopilot:** Percentage of configured projects that run for 30 days without manual task creation (target: 70%)

### Primary Metrics
| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| Manual Task Creation Rate | 100% | 30% | 2 months |
| Project Health Score Visibility | 0% | 100% | 1 month |
| Cross-Project Task Distribution | N/A | Balanced | 2 months |
| Signal-to-Task Conversion | 0% | 60% | 2 months |

### Guardrail Metrics (Do Not Degrade)
| Metric | Threshold |
|--------|-----------|
| Task execution latency | < 10% increase |
| Project analysis time | < 30 seconds per project |
| GitHub API usage | Within rate limits |

---

## Feature Specifications

---

### Feature 1: Project Configuration

#### Description
Define projects with goals, constraints, and local/remote sources in a declarative config file.

#### User Value
- Single source of truth for all project metadata
- Easy to add/remove projects
- Goals guide task generation toward valuable work

#### Configuration Schema

```yaml
projects:
  - id: sleepless-agent
    name: "Sleepless Agent"

    # Local workspace (where agent works)
    local_path: /Users/jima/workspace/sleepless-agent

    # Optional GitHub integration
    github:
      repo: jalateras/sleepless-agent
      default_branch: main
      sync_mode: pull  # pull | push | bidirectional
      check_dependencies: true

    # What this project needs
    goals:
      - type: coverage
        target: 80
        current: 65
      - type: documentation
        areas: ["README", "API docs", "examples"]
      - type: performance
        metric: "startup_time"
        target_ms: 500

    # What NOT to do
    constraints:
      - "No breaking changes to config.yaml"
      - "Maintain Python 3.11+ compatibility"
      - "Don't modify tests without corresponding code changes"

    # How often to check
    check_interval_hours: 4
    priority: high

  - id: local-ml-experiment
    name: "ML Prototype"
    local_path: /Users/jima/experiments/ml-prototype
    # No github - purely local
    goals:
      - type: feature
        description: "Implement image classification"
      - type: testing
        framework: pytest
```

#### Acceptance Criteria
- [ ] Projects load from config on daemon startup
- [ ] Config changes detected via file watcher
- [ ] Projects can be added/removed without restart
- [ ] Validation errors reported to user
- [ ] Both local-only and GitHub-enabled projects work

---

### Feature 2: Project State Analyzer

#### Description
Periodically analyze each project's state by examining local files and GitHub signals to identify what work is needed.

#### User Value
- Automatic discovery of issues, technical debt, and opportunities
- Proactive identification of work before it becomes urgent
- Data-driven prioritization based on project goals

#### Signal Sources

**Local Analysis:**
| Signal | Source | Action |
|--------|--------|--------|
| TODO/FIXME comments | Code search | Create refinement task |
| Test failures | pytest/jest output | Create fix task |
| Coverage gaps | .coverage/.pytest_cov | Create test task |
| Broken imports | Import analysis | Create fix task |
| Stale branches | git branch -v | Cleanup task |
| Uncommitted changes | git status | Reminder task |

**GitHub Analysis:**
| Signal | Source | Action |
|--------|--------|--------|
| Open issues | GitHub API | Triage/prioritize |
| Stale PRs | GitHub API | Reminder/merge task |
| Failed CI | Status API | Fix task |
| Dependabot alerts | Security API | Update task |
| Recent commits | Activity API | Review/regression check |
| Labels/milestones | Metadata | Task categorization |

#### Analysis Algorithm

```python
def analyze_project(project: ProjectConfig) -> List[WorkItem]:
    items = []

    # Local signal collection
    items.extend(analyzer.scan_todos(project.local_path))
    items.extend(analyzer.check_tests(project.local_path))
    items.extend(analyzer.measure_coverage(project.local_path))
    items.extend(analyzer.detect_breaking_changes(project.local_path))

    # GitHub signal collection (if configured)
    if project.github:
        items.extend(analyzer.fetch_issues(project.github))
        items.extend(analyzer.check_ci_status(project.github))
        items.extend(analyzer.get_dependabot_alerts(project.github))

    # Filter and prioritize against goals/constraints
    return prioritizer.rank(items, project.goals, project.constraints)
```

#### Acceptance Criteria
- [ ] Analyzes all projects within check_interval
- [ ] Local-only projects work without GitHub
- [ ] GitHub projects pull fresh data each cycle
- [ ] Signal collection is rate-limited and cached
- [ ] Analysis results logged for debugging

---

### Feature 3: Project-Aware Task Generator

#### Description
Convert analyzed signals into well-scoped tasks with project context and goal alignment.

#### User Value
- Generated tasks are relevant and valuable
- Tasks include project-specific context
- Work aligns with stated goals

#### Task Generation

```python
def generate_task(signal: WorkItem, project: ProjectConfig) -> Task:
    # Build task description with context
    description = build_task_description(
        signal=signal,
        project_context=get_project_context(project),
        goal_alignment=explain_goal_match(signal, project.goals)
    )

    # Determine task type and priority
    task_type = classify_task_type(signal)  # bugfix, feature, refactor, test
    priority = calculate_priority(signal, project)

    # Create task with project association
    return Task(
        description=description,
        project_id=project.id,
        project_name=project.name,
        priority=priority,
        task_type=map_to_task_type(task_type),
        context={
            'signal_source': signal.source,
            'goal_alignment': signal.matched_goal,
            'confidence': signal.confidence,
        }
    )
```

#### Task Description Examples

```python
# From TODO comment
"""
Fix authentication bug in login flow

Project: sleepless-agent
Source: TODO comment at src/auth.py:45

Context:
- This TODO was added 30 days ago
- Related to goal: "Improve test coverage to 80%" (auth module at 45%)
- Last touched by @jalateras in commit abc123

Action: Investigate the OAuth token refresh issue mentioned
in the TODO and implement a fix with tests.
"""

# From GitHub issue
"""
Implement dark mode support

Project: client-webapp
Source: GitHub issue #142 (opened by @designer, 5 days ago)

Context:
- Issue has 12 upvotes, labeled "enhancement" "good first issue"
- Related to goal: "Add dark mode support" (explicitly stated)
- No one has claimed this issue yet

Action: Implement the dark mode toggle as described in
the issue, following the project's design system.
"""

# From test coverage gap
"""
Add tests for payment module

Project: sleepless-agent
Source: Coverage report shows 23% coverage in payments/

Context:
- Goal: "Improve test coverage to 80%" (currently at 65% overall)
- Payment module is critical but under-tested
- Recent changes in this module (last commit: 3 days ago)

Action: Write comprehensive tests for the payment processing
logic, focusing on edge cases and error handling.
"""
```

#### Acceptance Criteria
- [ ] Generated tasks include project context
- [ ] Task descriptions explain WHY (goal alignment)
- [ ] Different signal types produce appropriate task formats
- [ ] Low-confidence signals are filtered or marked for review
- [ ] Tasks respect project constraints

---

### Feature 4: Multi-Project Prioritizer

#### Description
Rank work across all projects to determine which task to execute next based on project priority, urgency, and resource constraints.

#### User Value
- Important work happens first
- No single project hogs resources
- User can override when needed

#### Prioritization Algorithm

```python
def prioritize_tasks(all_tasks: List[Task], projects: List[ProjectConfig]) -> List[Task]:
    scores = []

    for task in all_tasks:
        project = get_project(task.project_id, projects)

        score = 0

        # Project priority weight
        score += PROJECT_PRIORITY_WEIGHT[project.priority]

        # Signal urgency
        if task.is_bugfix():
            score += 20
        elif task.is_security_issue():
            score += 50
        elif task.is_stale_signal(days=7):
            score += 10

        # Goal proximity
        goal = get_matched_goal(task, project)
        if goal and goal.near_target():
            score += 15

        # Project starvation (avoid neglect)
        recent_tasks_for_project = count_recent(project.id, days=24)
        if recent_tasks_for_project == 0:
            score += 5

        # User feedback (from Feature 1 of high-roi PRD)
        if task.project_id in positive_feedback_projects:
            score += 10

        scores.append((task, score))

    return [t for t, s in sorted(scores, key=lambda x: x[1], reverse=True)]
```

#### Acceptance Criteria
- [ ] Tasks across projects are interleaved fairly
- [ ] High-priority projects get more tasks
- [ ] Urgent signals (bugs, security) jump queue
- [ ] No project is completely starved
- [ ] User can manually adjust priorities

---

### Feature 5: Project Health Dashboard

#### Description
Provide visibility into each project's status, recent work, and progress toward goals.

#### User Value
- Know what the agent is working on for each project
- Track progress toward project goals
- Identify projects that need attention

#### Dashboard Sections

```
PROJECT HEALTH REPORT
Generated: 2026-01-01 14:30 UTC

[Sleepless Agent] 🟢 Healthy
  Priority: High | Last Checked: 2 hours ago
  Goals Progress:
    ✅ Coverage: 65% → 78% (target: 80%)
    🔄 Documentation: 3/5 areas complete
    ⚠️ Performance: Startup at 650ms (target: 500ms)

  Recent Activity:
    ✅ Task #142: Add tests for auth module (completed 4h ago)
    🔄 Task #143: Optimize startup time (in progress)
    📋 Task #144: Update API documentation (queued)

  Active Signals:
    - 3 TODO comments (oldest: 45 days)
    - 5 open GitHub issues (1 needs triage)
    - CI: All tests passing

[Client WebApp] 🟡 Needs Attention
  Priority: Medium | Last Checked: 6 hours ago
  Goals Progress:
    ⚠️ OAuth: Not started (due to auth library decision)
    ✅ Dark Mode: Implementation complete

  Recent Activity:
    ✅ Task #89: Implement dark mode (completed yesterday)
    ❌ Task #90: Fix login bug (failed 3x, needs user attention)

  Active Signals:
    - 2 failing tests
    - CI: Build failing on main branch
    - 1 stale PR (12 days old)

[ML Prototype] 🟢 Stable
  Priority: Low | Last Checked: 1 day ago
  Goals Progress:
    🔄 Image Classification: Model at 82% accuracy (target: 85%)

  Active Signals:
    - No TODOs
    - No GitHub (local-only project)
```

#### Acceptance Criteria
- [ ] Dashboard shows all configured projects
- [ ] Health status (🟢🟡🔴) reflects actual state
- [ ] Goals show progress metrics
- [ ] Recent activity is accurate
- [ ] Available via Slack command

---

## Data Model

### Project Configuration

```python
@dataclass
class GitHubConfig:
    repo: str  # "owner/repo"
    default_branch: str = "main"
    sync_mode: Literal["pull", "push", "bidirectional"] = "pull"
    check_dependencies: bool = True
    auth_token: Optional[str] = None  # From env var

@dataclass
class ProjectGoal:
    type: Literal["coverage", "documentation", "performance", "feature", "testing"]
    target: Optional[float] = None
    current: Optional[float] = None
    description: Optional[str] = None
    metric: Optional[str] = None

@dataclass
class ProjectConfig:
    id: str
    name: str
    local_path: str  # Absolute path to local workspace
    github: Optional[GitHubConfig] = None
    goals: List[ProjectGoal] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    check_interval_hours: int = 4
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    enabled: bool = True
```

### Work Signals

```python
@dataclass
class WorkItem:
    source: Literal["todo", "test_failure", "coverage", "github_issue", "ci_failure"]
    title: str
    description: str
    location: Optional[str] = None  # File path or issue URL
    urgency: int = 0  # 0-100
    confidence: float = 1.0  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     SleeplessAgent                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              ProjectOrchestrator (NEW)                │  │
│  │  ├── ProjectConfigLoader                             │  │
│  │  ├── ProjectStateAnalyzer                            │  │
│  │  │   ├── LocalSignalCollector                        │  │
│  │  │   └── GitHubSignalCollector                       │  │
│  │  ├── ProjectTaskGenerator                            │  │
│  │  ├── MultiProjectPrioritizer                         │  │
│  │  └── ProjectHealthReporter                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                               │
│                              ▼                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           AutoTaskGenerator (existing)               │  │
│  │  Now receives project-suggested tasks                │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                               │
│                              ▼                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  TaskQueue (existing)                │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                               │
│                              ▼                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  Executor (existing)                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└───────────────────────────────────────────────────────────────┘

External Sources:
  ├── projects.yaml (local config)
  ├── Local filesystem (TODOs, tests, coverage)
  └── GitHub API (issues, PRs, CI, alerts)
```

### File Locations

```
sleepless-agent/
├── config/
│   └── projects.yaml           # NEW: Project configurations
├── src/sleepless_agent/
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── project_config.py       # NEW: ProjectConfig dataclasses
│   │   ├── project_loader.py       # NEW: Load/validate projects.yaml
│   │   ├── signal_collector.py     # NEW: Collect local + GitHub signals
│   │   ├── task_generator.py       # NEW: Convert signals to tasks
│   │   ├── prioritizer.py          # NEW: Multi-project ranking
│   │   └── health_reporter.py      # NEW: Project health dashboard
│   └── core/
│       └── daemon.py               # MODIFIED: Wire ProjectOrchestrator
```

---

## Implementation Plan

### Iteration 1: Foundation (Week 1-2)

**Deliverables:**
- ProjectConfig dataclasses with validation
- projects.yaml schema and loader
- Basic ProjectOrchestrator skeleton
- Daemon integration for project loading

**Acceptance:**
- [ ] Projects load from config on startup
- [ ] Config validation catches errors
- [ ] File watcher reloads projects on change
- [ ] Projects accessible in daemon

### Iteration 2: Local Signals (Week 3-4)

**Deliverables:**
- LocalSignalCollector for TODO/FIXME comments
- Test failure detection from pytest/jest
- Coverage gap analysis
- Basic task generation from local signals

**Acceptance:**
- [ ] TODO comments generate tasks
- [ ] Test failures generate fix tasks
- [ ] Coverage gaps generate test tasks
- [ ] Generated tasks have project context

### Iteration 3: GitHub Integration (Week 5-6)

**Deliverables:**
- GitHubSignalCollector using PyGithub
- Issue/PR fetching and triaging
- CI status checking
- Dependabot alert consumption
- Sync modes (pull/push/bidirectional)

**Acceptance:**
- [ ] GitHub issues generate tasks
- [ ] Failed CI creates fix tasks
- [ ] Stale PRs generate reminders
- [ ] API usage respects rate limits

### Iteration 4: Prioritization & Goals (Week 7-8)

**Deliverables:**
- Multi-project prioritization algorithm
- Goal alignment scoring
- Constraint checking (what NOT to do)
- Starvation prevention

**Acceptance:**
- [ ] Tasks ranked across projects
- [ ] Goal-proximate work prioritized
- [ ] Constraints prevent invalid tasks
- [ ] Fair distribution across projects

### Iteration 5: Dashboard & Polish (Week 9-10)

**Deliverables:**
- ProjectHealthReporter
- Slack command for health report
- Progress tracking toward goals
- Metrics and logging

**Acceptance:**
- [ ] Health dashboard shows all projects
- [ ] Goal progress is accurate
- [ ] `/project-health` command works
- [ ] Historical tracking available

---

## Risks and Mitigations

### R1: GitHub API Rate Limits
**Likelihood:** High | **Impact:** Medium

**Mitigation:**
- Implement aggressive caching (ETag headers)
- Use conditional requests (`If-Modified-Since`)
- Batch requests where possible
- Graceful degradation when rate-limited
- Configurable check intervals

### R2: Signal Noise
**Likelihood:** High | **Impact:** Medium

**Mitigation:**
- Confidence scoring for each signal
- Minimum threshold before task generation
- User feedback loop (reject bad tasks)
- Whitelist/blacklist for signal types
- Manual review queue for low-confidence items

### R3: Project Goal Drift
**Likelihood:** Medium | **Impact:** High

**Mitigation:**
- Regular goal progress reporting
- Alert when goals are stale or unreachable
- Easy goal updates via config
- Suggest goal adjustments based on actual work

### R4: Local Path Issues
**Likelihood:** Medium | **Impact:** Low

**Mitigation:**
- Validate local_path exists on load
- Handle symlinks and relative paths
- Graceful handling of disappeared directories
- Clear error messages for path issues

### R5: Context Switching Overhead
**Likelihood:** Low | **Impact:** Medium

**Mitigation:**
- Batch work per project (do 2-3 tasks before switching)
- Warm project context (keep recent workspaces cached)
- Configurable task batch size

---

## Future Considerations

### Phase 2 Enhancements

1. **Smart Project Selection**
   - ML model to predict which projects need attention
   - Seasonal patterns (e.g., more maintenance after releases)
   - Dependency-aware prioritization

2. **Cross-Project Workflows**
   - "Update dependency across all projects"
   - "Apply security fix to all repos"
   - Template standardization

3. **Collaborative Filtering**
   - "Projects like yours also worked on..."
   - Recommended goals based on similar projects
   - Community project templates

4. **Advanced GitHub Integration**
   - Auto-labeling of issues
   - PR review automation
   - Release management tasks

---

## Open Questions

1. **Project isolation:** Should projects have separate budgets/cost limits?
2. **Task ownership:** Should generated tasks be auto-assigned or require approval?
3. **Goal evolution:** How should goals adapt as projects mature?
4. **Multi-tenant:** Should multiple users be able to share a sleepless-agent with different project sets?
5. **Local-only mode:** What features should work without any GitHub integration?

---

## Appendix

### Example projects.yaml

```yaml
# config/projects.yaml
version: "1.0"

projects:
  # Full-featured project with GitHub
  - id: sleepless-agent
    name: "Sleepless Agent"
    local_path: /Users/jima/workspace/sleepless-agent
    github:
      repo: jalateras/sleepless-agent
      default_branch: main
      sync_mode: pull
      check_dependencies: true
    goals:
      - type: coverage
        target: 80
        metric: "line_coverage"
      - type: documentation
        areas: ["README", "API", "examples"]
      - type: performance
        metric: "startup_time"
        target_ms: 500
    constraints:
      - "No breaking changes to config.yaml"
      - "Maintain Python 3.11+ compatibility"
    check_interval_hours: 4
    priority: high
    enabled: true

  # GitHub-only project (auto-clone)
  - id: oss-library
    name: "Open Source Library"
    local_path: null  # Auto-managed
    github:
      repo: user/oss-library
      default_branch: main
      sync_mode: bidirectional
    goals:
      - type: coverage
        target: 90
      - type: feature
        description: "Add async support"
    check_interval_hours: 8
    priority: medium
    enabled: true

  # Local-only project
  - id: side-project
    name: "Side Project"
    local_path: /Users/jima/projects/side-project
    goals:
      - type: feature
        description: "MVP implementation"
    check_interval_hours: 24
    priority: low
    enabled: true
```

### Signal-to-Task Examples

| Signal | Task Generated | Priority | Notes |
|--------|---------------|----------|-------|
| TODO: "Fix this race condition" | "Fix race condition in auth.py" | Medium | Old TODO + critical area |
| pytest: 3 tests failing | "Fix failing tests in payment module" | High | Blocking CI |
| GitHub issue #42: "Add dark mode" | "Implement dark mode support" | Medium | 15 upvotes, labeled enhancement |
| Coverage: 40% in utils/ | "Improve test coverage in utils/" | Low | Below project target |
| Dependabot: Critical security alert | "Update lodash to 4.17.21" | Urgent | Security |
| Stale PR: 12 days old | "Review and merge PR #89" | Medium | Needs attention |

