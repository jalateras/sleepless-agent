# Sleepless Agent Documentation

Welcome to the Sleepless Agent documentation! This guide will help you understand, configure, and use the 24/7 AI agent that works while you sleep.

## What is Sleepless Agent?

Sleepless Agent is an autonomous AI system that transforms your Claude Code Pro subscription into a 24/7 AgentOS. It processes tasks via Slack commands, manages isolated workspaces, and optimizes Claude usage across day and night cycles.

### Key Features

- **🤖 Continuous Operation**: Runs as a daemon, always ready for new tasks
- **💬 Slack Integration**: Submit and manage tasks through Slack commands
- **🎯 Hybrid Autonomy**: Auto-processes random thoughts, requires review for serious tasks
- **⚡ Smart Scheduling**: Optimizes execution based on priorities and usage limits
- **📊 Persistent Storage**: SQLite-backed task queue with full history
- **🏗️ Workspace Isolation**: Each task runs in its own isolated environment
- **📈 Usage Optimization**: Intelligent Pro plan usage management

## Documentation Overview

### 📚 Getting Started
- [**Complete Setup Guide**](complete-setup-guide.md) - Comprehensive step-by-step guide (start here!)
- [**Quickstart**](quickstart.md) - Get running in 5 minutes
- [**Installation**](installation.md) - Detailed setup instructions
- [**FAQ**](faq.md) - Frequently asked questions

### 🧠 Core Concepts
- [**Architecture**](concepts/architecture.md) - System design and components
- [**Task Lifecycle**](concepts/task-lifecycle.md) - How tasks flow through the system
- [**Workspace Isolation**](concepts/workspace-isolation.md) - Understanding isolated environments
- [**Scheduling**](concepts/scheduling.md) - Task prioritization and execution
- [**Pro Plan Management**](concepts/pro-plan-management.md) - Usage optimization strategies

### 📖 Guides
- [**Slack Setup**](guides/slack-setup.md) - Configure your Slack application
- [**Environment Setup**](guides/environment-setup.md) - Configure environment variables
- [**Git Integration**](guides/git-integration.md) - Automated commits and PRs
- [**Project Management**](guides/project-management.md) - Organizing tasks into projects
- [**Deployment**](guides/deployment.md) - Production deployment strategies

### 🎓 Tutorials
- [**First Task**](tutorials/first-task.md) - Create your first automated task
- [**Slack Workflows**](tutorials/slack-workflows.md) - Build effective command workflows
- [**Monitoring Tasks**](tutorials/monitoring-tasks.md) - Track execution and results
- [**Daily Reports**](tutorials/daily-reports.md) - Understanding generated reports
- [**Workspace Management**](tutorials/workspace-management.md) - Managing task workspaces

### 📋 Reference
- [**CLI Commands**](reference/api/cli-commands.md) - Complete command reference
- [**Slack Commands**](reference/api/slack-commands.md) - Slash command reference
- [**Configuration**](reference/configuration.md) - All configuration options
- [**Database Schema**](reference/database-schema.md) - SQLite schema details
- [**Environment Variables**](reference/environment-variables.md) - Variable reference

### 💡 Examples
- [**Basic Usage**](examples/basic-usage.md) - Simple usage patterns
- [**Slack Integration**](examples/slack-integration.md) - Slack workflow examples
- [**Advanced Workflows**](examples/advanced-workflows.md) - Complex task patterns
- [**Custom Executors**](examples/custom-executors.md) - Extending functionality

## Quick Links

- [GitHub Repository](https://github.com/context-machine-lab/sleepless-agent)
- [Discord Community](https://discord.gg/74my3Wkn)
- [Issue Tracker](https://github.com/context-machine-lab/sleepless-agent/issues)
- [Changelog](changelog.md)

## System Requirements

- Python 3.11+
- Claude Code CLI (`@anthropic-ai/claude-code`)
- Slack workspace with admin access
- SQLite 3
- Git (optional, for automation)

## Getting Help

- Check the [FAQ](faq.md) for common questions
- Review [Troubleshooting](troubleshooting.md) for solutions
- Join our [Discord](https://discord.gg/74my3Wkn) for community support
- Open an [issue](https://github.com/context-machine-lab/sleepless-agent/issues) for bugs

## Contributing

We welcome contributions! See our [Contributing Guide](https://github.com/context-machine-lab/sleepless-agent/blob/main/CONTRIBUTING.md) for details.

## License

Sleepless Agent is released under the [MIT License](https://github.com/context-machine-lab/sleepless-agent/blob/main/LICENSE).