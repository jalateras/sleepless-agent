# Sleepless Agent - Complete Setup & Usage Guide

> A comprehensive step-by-step guide to setting up and using Sleepless Agent.

## What is Sleepless Agent?

Sleepless Agent is a **24/7 AI AgentOS** that turns your Claude Code Pro subscription into an autonomous development engine. It runs tasks while you sleep, processes ideas via Slack, manages Git commits automatically, and intelligently schedules work based on your Pro plan usage.

### Key Features

- **Multi-agent workflow**: Planner → Worker → Evaluator pattern for structured task execution
- **Usage-aware scheduling**: Runs aggressively at night (80% threshold), conservatively during day (20%)
- **Workspace isolation**: Each task gets its own directory, preventing conflicts
- **Event-driven daemon**: Async event loop that monitors queues, schedules tasks, and handles notifications
- **Interactive chat mode**: Real-time conversations with Claude in Slack threads
- **Automatic Git integration**: Commits, branches, and PRs created automatically

---

## Prerequisites Checklist

| Requirement | Version | How to Check |
|-------------|---------|--------------|
| Python | 3.11+ | `python --version` |
| Node.js | 16+ | `node --version` |
| Claude Code CLI | Latest | `claude --version` |
| Slack Workspace | Admin access | - |
| Git | Any | `git --version` |

---

## Step 1: Install System Dependencies

### macOS

```bash
# Install Python 3.11+ and Node.js via Homebrew
brew install python@3.11 node

# Verify
python3.11 --version
node --version
```

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# Install Node.js via NodeSource
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install nodejs
```

### Windows (WSL2)

```bash
# Inside WSL2 Ubuntu, follow Ubuntu instructions above
# Or use native Windows with installers from python.org and nodejs.org
```

---

## Step 2: Install Claude Code CLI

Claude Code CLI is the core execution engine that Sleepless Agent wraps.

```bash
# Install globally via npm
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version
```

### Authenticate Claude Code

```bash
# Login (opens browser for OAuth)
claude login

# Verify authentication and check your usage
claude /usage
```

> **Note:** Sleepless Agent monitors your Pro plan usage via `claude /usage`. Usage resets every 5 hours, not daily. The agent intelligently pauses when approaching limits to avoid interruption.

---

## Step 3: Install Sleepless Agent

### Option A: From PyPI (Recommended for users)

```bash
pip install sleepless-agent

# Verify installation
sle --version
```

### Option B: From Source (For development/customization)

```bash
# Clone repository
git clone https://github.com/context-machine-lab/sleepless-agent
cd sleepless-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Verify
sle --version
```

### Option C: Using Make (From source)

```bash
git clone https://github.com/context-machine-lab/sleepless-agent
cd sleepless-agent
make setup
```

---

## Step 4: Create Slack App

This is the most detailed step. Follow carefully.

### 4.1 Create the App

1. Go to **https://api.slack.com/apps**
2. Click **"Create New App"** → **"From scratch"**
3. Configure:
   - **App Name:** `Sleepless Agent`
   - **Workspace:** Select your workspace
4. Click **"Create App"**

### 4.2 Enable Socket Mode

Socket Mode allows real-time communication without a public endpoint.

1. Go to **Settings → Socket Mode**
2. Toggle **Enable Socket Mode** to ON
3. Create an app-level token:
   - **Token Name:** `sleepless-token`
   - **Scope:** `connections:write`
4. Click **Generate**
5. **SAVE THIS TOKEN!** It starts with `xapp-`

### 4.3 Create Slash Commands

Go to **Features → Slash Commands** and create each:

| Command | Short Description | Usage Hint |
|---------|-------------------|------------|
| `/think` | Submit a task or thought | `[description] [-p project_name]` |
| `/check` | Check system status and queue | (none) |
| `/usage` | Show Claude Code Pro plan usage | (none) |
| `/chat` | Start interactive chat mode | `<project_name> \| end \| status` |
| `/report` | View task reports | `[task_id \| date \| project_name]` |
| `/cancel` | Cancel a task or project | `<task_id \| project_name>` |
| `/trash` | Manage cancelled tasks | `<list \| restore <id> \| empty>` |

### 4.4 Set OAuth Scopes

Go to **Features → OAuth & Permissions** → **Bot Token Scopes** and add:

**Required Scopes:**

| Scope | Purpose |
|-------|---------|
| `chat:write` | Send messages |
| `chat:write.public` | Message public channels |
| `commands` | Receive slash commands |
| `channels:history` | Read channel history (for chat mode) |
| `groups:history` | Read private channel history |
| `im:history` | Read DM history |
| `reactions:write` | Add emoji reactions |
| `users:read` | Get user information |

### 4.5 Enable Event Subscriptions

Go to **Features → Event Subscriptions**:

1. Toggle **Enable Events** to ON
2. Under **Subscribe to bot events**, add:
   - `message.channels` (for chat mode in channels)
   - `message.groups` (for chat mode in private channels)
   - `app_mention` (for @mentions)

### 4.6 Install to Workspace

1. Go to **Settings → Install App**
2. Click **"Install to Workspace"**
3. Review and click **"Allow"**
4. **SAVE THE BOT TOKEN!** It starts with `xoxb-`

> **Chat Mode Architecture:** `/chat` creates a Slack thread where you converse with Claude in real-time. The agent maintains session state across messages in the thread. Each chat session has its own project workspace at `workspace/projects/<name>/`. Sessions auto-expire after 30 minutes of inactivity.

---

## Step 5: Configure Environment

### 5.1 Create Environment File

```bash
# If installed from source:
cd sleepless-agent
cp .env.example .env

# If installed from PyPI, create manually:
cat > .env << 'EOF'
# Slack Bot (Required)
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here

# Git (Optional)
GIT_USER_NAME=Sleepless Agent
GIT_USER_EMAIL=agent@sleepless.local

# Logging (Optional)
LOG_LEVEL=INFO
EOF
```

### 5.2 Add Your Tokens

Edit `.env` and replace the placeholder tokens:

```bash
SLACK_BOT_TOKEN=xoxb-ACTUAL-TOKEN-FROM-STEP-4.6
SLACK_APP_TOKEN=xapp-ACTUAL-TOKEN-FROM-STEP-4.2
```

### 5.3 Secure the File

```bash
chmod 600 .env
```

---

## Step 6: Configure Runtime Settings

The agent uses `config.yaml` for runtime behavior.

### 6.1 Locate/Create Config File

If installed from source, edit `src/sleepless_agent/config.yaml`. Otherwise, create `~/.sleepless/config.yaml`.

### 6.2 Key Configuration Options

```yaml
# Claude Code settings
claude_code:
  binary_path: claude              # Path to CLI
  model: claude-sonnet-4-5-20250929
  night_start_hour: 1              # Night mode starts (1 AM)
  night_end_hour: 9                # Night mode ends (9 AM)
  threshold_day: 20.0              # Pause at 20% during day
  threshold_night: 80.0            # Pause at 80% at night

# Git integration (disabled by default)
git:
  enabled: false                   # Set to true to enable
  use_remote_repo: true
  remote_repo_url: git@github.com:YOUR/REPO.git  # Change this!
  auto_create_repo: true

# Task settings
agent:
  workspace_root: ./workspace
  task_timeout_seconds: 1800       # 30 minutes per task

# Human approval checkpoints
checkpoints:
  enabled: true
  global_defaults:
    post_plan: false               # Don't require approval after planning
    pre_commit: true               # Require approval before commits
    pre_pr: true                   # Require approval before PRs
```

> **Smart Scheduling Logic:**
> - **Day mode (9 AM - 1 AM):** Conservative 20% threshold, leaves headroom for your work
> - **Night mode (1 AM - 9 AM):** Aggressive 80% threshold, maximizes autonomous work
> - The daemon checks `claude /usage` and pauses new task generation when threshold is reached
> - Running tasks complete; only new task generation pauses

---

## Step 7: Start the Agent

### 7.1 Run the Daemon

```bash
# Start the agent daemon
sle daemon
```

You should see:

```
INFO | Sleepless Agent starting...
INFO | Slack bot started and listening for events
```

### 7.2 Running Options

```bash
# Run with debug logging
SLEEPLESS_LOG_LEVEL=DEBUG sle daemon

# Run in background (Linux/macOS)
sle daemon &

# View logs
tail -f workspace/data/agent.log
```

### 7.3 Install as a Service (Production)

**macOS (launchd):**

```bash
make install-launchd
# Agent will auto-start on login
```

**Linux (systemd):**

```bash
make install-service
sudo systemctl start sleepless-agent
sudo systemctl enable sleepless-agent  # Auto-start on boot
```

---

## Step 8: Verify & Test

### 8.1 Invite Bot to Channel

In Slack:

```
/invite @sleepless-agent
```

### 8.2 Test Commands

```
/check
# Should show: Agent is running, queue status, etc.

/usage
# Should show: Claude Code Pro plan usage percentage

/think Research Python async patterns
# Should acknowledge and queue the task
```

### 8.3 Test Chat Mode

```
/chat my-first-project
```

This creates a thread. **Reply inside the thread**:

```
Create a hello world Python script
```

Claude will respond in the thread. To end:

```
exit
```

Or use `/chat end`.

---

## Step 9: Understanding the Usage

### Slash Commands Reference

| Command | Description | Example |
|---------|-------------|---------|
| `/think <text>` | Submit a random thought/idea | `/think Explore async patterns` |
| `/think -p <project> <text>` | Submit a serious task (creates PR) | `/think -p backend Add OAuth2` |
| `/check` | View system status and queue | `/check` |
| `/usage` | View Claude Pro plan usage | `/usage` |
| `/chat <project>` | Start interactive chat session | `/chat my-app` |
| `/chat end` | End current chat session | `/chat end` |
| `/chat status` | View active chat sessions | `/chat status` |
| `/report` | View today's task report | `/report` |
| `/report <id>` | View specific task details | `/report 42` |
| `/report --list` | List all available reports | `/report --list` |
| `/cancel <id>` | Cancel a task or project | `/cancel 5` |
| `/trash list` | List trashed items | `/trash list` |
| `/trash restore <id>` | Restore from trash | `/trash restore my-project` |
| `/trash empty` | Permanently delete trash | `/trash empty` |

### CLI Commands Reference

```bash
sle daemon          # Start the agent daemon
sle check           # Check status (from CLI)
sle think "text"    # Submit a thought
sle think "text" -p project  # Submit serious task
sle report 42       # View task report
sle cancel 5        # Cancel task
sle trash list      # List trashed items
```

### Makefile Commands

```bash
make run            # Start daemon
make dev            # Start with debug logging
make logs           # Follow live logs
make status         # Check daemon status
make stats          # View performance metrics
make db             # Query database
make db-reset       # Clear database
make clean          # Clean cache
make backup         # Backup workspace
```

---

## Step 10: Task Types & Workflows

### Random Thoughts (Default)

```
/think Explore Python async patterns
```

- Auto-commits to `thought-ideas` branch
- No PR created
- Lower priority
- Great for exploration and brainstorming

### Serious Tasks

```
/think -p backend Add OAuth2 authentication
```

- Creates feature branch: `feature/backend-<task_id>`
- Creates Pull Request after completion
- Higher priority
- Requires checkpoint approvals (pre-commit, pre-PR)

### Interactive Chat Mode

```
/chat my-project
```

- Real-time conversation with Claude
- Dedicated project workspace
- Claude can read/write/edit files
- Great for iterative development

> **Multi-Agent Workflow (for serious tasks):**
> 1. **Planner Agent** (max 10 turns): Analyzes task, creates execution plan
> 2. **Worker Agent** (max 30 turns): Executes the plan, writes code
> 3. **Evaluator Agent** (max 10 turns): Reviews output, suggests improvements
>
> This structured approach produces higher-quality results than single-agent execution.

---

## Step 11: Enable Git Integration (Optional)

### 11.1 Update config.yaml

```yaml
git:
  enabled: true                    # Enable Git features
  use_remote_repo: true
  remote_repo_url: git@github.com:yourusername/sleepless-workspace.git
  auto_create_repo: true
```

### 11.2 Configure Git

```bash
git config --global user.name "Sleepless Agent"
git config --global user.email "agent@sleepless.local"

# For PR creation (GitHub CLI)
gh auth login
```

### 11.3 Set Up SSH Key (for remote push)

```bash
# Generate key if needed
ssh-keygen -t ed25519 -C "sleepless-agent"

# Add to GitHub: Settings → SSH Keys
cat ~/.ssh/id_ed25519.pub
```

---

## Workspace Structure

```
workspace/
├── data/
│   ├── tasks.db           # SQLite database
│   ├── results/           # Task output JSON files
│   ├── reports/           # Daily markdown reports
│   ├── chat_sessions.json # Chat mode state
│   ├── agent.log          # Application logs
│   └── metrics.jsonl      # Performance metrics
├── tasks/
│   ├── task_1/            # Isolated workspace for task 1
│   ├── task_2/            # Isolated workspace for task 2
│   └── ...
├── projects/
│   ├── backend/           # Shared workspace for "backend" project
│   ├── my-app/            # Shared workspace for "my-app" project
│   └── ...
├── shared/                # Resources shared across all tasks
└── trash/                 # Soft-deleted projects
```

---

## Troubleshooting

### Bot Not Responding in Slack

1. **Verify Socket Mode is enabled** in Slack App settings
2. **Check tokens** in `.env`:
   ```bash
   cat .env | grep SLACK
   ```
3. **Confirm daemon is running**:
   ```bash
   pgrep -f sleepless_agent
   ```
4. **Check logs**:
   ```bash
   tail -f workspace/data/agent.log
   ```

### Chat Mode Not Working

1. **Ensure `message.channels` event is subscribed** (Slack App → Event Subscriptions)
2. **Bot must be in the channel**: `/invite @sleepless-agent`
3. **Messages must be in the thread**, not main channel

### Tasks Not Executing

1. **Check Claude CLI**:
   ```bash
   claude --version
   claude /usage
   ```
2. **Usage limit reached?**:
   ```bash
   /usage  # In Slack
   ```
3. **Review daemon logs** for errors

### "Permission Denied" Errors

```bash
# Fix workspace permissions
chmod -R 755 workspace/

# Fix .env permissions
chmod 600 .env
```

### Database Issues

```bash
# Reset database
make db-reset

# Or manually
rm workspace/data/tasks.db
sle daemon  # Will recreate
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────┐
│  SLEEPLESS AGENT - QUICK REFERENCE                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  START:        sle daemon                           │
│  STATUS:       /check  or  sle check                │
│  USAGE:        /usage                               │
│                                                     │
│  TASKS:                                             │
│    Random:     /think <description>                 │
│    Serious:    /think -p <project> <description>   │
│    Cancel:     /cancel <id>                         │
│                                                     │
│  CHAT MODE:                                         │
│    Start:      /chat <project-name>                 │
│    End:        /chat end  or type 'exit' in thread  │
│    Status:     /chat status                         │
│                                                     │
│  REPORTS:                                           │
│    Today:      /report                              │
│    Task:       /report <task_id>                    │
│    Project:    /report <project_name>               │
│    List:       /report --list                       │
│                                                     │
│  LOGS:         tail -f workspace/data/agent.log    │
│  STATS:        make stats                           │
│  BACKUP:       make backup                          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Read the architecture docs**: [Architecture Overview](concepts/architecture.md)
2. **Configure Git integration** for automatic commits and PRs: [Git Integration](guides/git-integration.md)
3. **Customize thresholds** in `config.yaml` for your schedule
4. **Set up as a service** for true 24/7 operation
5. **Explore task templates** in `templates/builtin/` for common workflows

---

## Getting Help

- [Troubleshooting Guide](troubleshooting.md)
- [FAQ](faq.md)
- [Discord Community](https://discord.gg/74my3Wkn)
- [GitHub Issues](https://github.com/context-machine-lab/sleepless-agent/issues)
