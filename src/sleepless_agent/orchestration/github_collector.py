"""GitHub signal collector for project analysis."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from urllib.parse import quote

from sleepless_agent.monitoring.logging import get_logger
from sleepless_agent.orchestration.project_config import ProjectConfig, GitHubConfig
from sleepless_agent.orchestration.signals import (
    SignalSource,
    SignalType,
    WorkItem,
)

logger = get_logger(__name__)


class GitHubSignalCollector:
    """Collects signals from GitHub API.

    Fetches:
    - Open issues with triage
    - Stale pull requests
    - CI/CD status
    - Dependabot alerts
    """

    # Stale thresholds
    STALE_ISSUE_DAYS = 30
    STALE_PR_DAYS = 7

    def __init__(self, project: ProjectConfig):
        """Initialize collector for a project.

        Args:
            project: Project configuration with GitHub integration
        """
        if not project.github:
            raise ValueError("Project must have GitHub configuration")

        self.project = project
        self.github = project.github

        # Lazy load PyGithub
        self._github_client = None

    @property
    def github_client(self):
        """Lazy load GitHub client."""
        if self._github_client is None:
            try:
                from github import Github
                token = self.github.auth_token or os.environ.get("GITHUB_TOKEN")
                if token:
                    self._github_client = Github(token)
                else:
                    # Unauthenticated (rate-limited)
                    self._github_client = Github()
                logger.debug("github.client.connected", repo=self.github.repo)
            except ImportError:
                logger.warning("github.not_installed")
                self._github_client = None
        return self._github_client

    def collect_all(self) -> List[WorkItem]:
        """Collect all GitHub signals for the project.

        Returns:
            List of WorkItems discovered
        """
        items = []

        if not self.github_client:
            logger.warning(
                "collector.github.unavailable",
                project_id=self.project.id,
            )
            return items

        try:
            repo = self.github_client.get_repo(self.github.repo)

            # Collect issues
            items.extend(self._collect_issues(repo))

            # Collect PRs
            items.extend(self._collect_pull_requests(repo))

            # Collect CI failures (from PRs)
            items.extend(self._collect_ci_failures(repo))

            # Collect Dependabot alerts
            items.extend(self._collect_dependabot_alerts(repo))

        except Exception as e:
            logger.error(
                "collector.github.failed",
                project_id=self.project.id,
                error=str(e),
            )

        logger.info(
            "collector.github.complete",
            project_id=self.project.id,
            items_found=len(items),
        )

        return items

    def _collect_issues(self, repo) -> List[WorkItem]:
        """Collect signals from open issues.

        Args:
            repo: PyGithub Repository object

        Returns:
            List of WorkItems from issues
        """
        items = []

        try:
            # Get open issues, sorted by created date (newest first)
            issues = repo.get_issues(state="open", sort="created", direction="desc")

            for issue in issues.get_page(20):  # Limit to first 20
                # Skip pull requests (they're handled separately)
                if issue.pull_request:
                    continue

                # Calculate age
                created_at = issue.created_at.replace(tzinfo=None)
                age_days = (datetime.now() - created_at).days

                # Determine priority based on labels and reactions
                urgency = self._calculate_issue_urgency(issue, age_days)

                # Determine type based on labels
                signal_type = self._classify_issue(issue)

                # Check if needs attention
                needs_attention = self._issue_needs_attention(issue, age_days)

                if needs_attention:
                    # Get reactions count for confidence
                    reactions_count = sum(reactions.count for reactions in issue.get_reactions().raw_data)
                    confidence = min(0.5 + (reactions_count * 0.1), 1.0)

                    # Get assignees
                    assignees = [a.login for a in issue.assignees or []]
                    assignee_info = f"Assigned to: {', '.join(assignees)}" if assignees else "Unassigned"

                    # Get labels
                    labels = [label.name for label in issue.labels or []]
                    label_info = ", ".join(labels[:5]) if labels else "No labels"

                    items.append(WorkItem(
                        source=SignalSource.GITHUB_ISSUE,
                        type=signal_type,
                        title=f"Address GitHub issue: {issue.title[:60]}",
                        description=f"Issue #{issue.number}: {issue.title}\n\n{issue.body[:500] if issue.body else 'No description'}",
                        location=f"https://github.com/{self.github.repo}/issues/{issue.number}",
                        urgency=urgency,
                        confidence=confidence,
                        age_days=age_days,
                        metadata={
                            "issue_number": issue.number,
                            "labels": labels,
                            "assignees": assignees,
                            "reactions_count": reactions_count,
                            "html_url": issue.html_url,
                            "author": issue.user.login if issue.user else "unknown",
                        },
                    ))

        except Exception as e:
            logger.debug("collector.issues.failed", error=str(e))

        return items

    def _calculate_issue_urgency(self, issue, age_days: int) -> int:
        """Calculate urgency score for an issue.

        Args:
            issue: PyGithub Issue object
            age_days: Age in days

        Returns:
            Urgency score 0-100
        """
        urgency = 30  # Base urgency

        # Boost for old issues (stale)
        if age_days > self.STALE_ISSUE_DAYS:
            urgency += min(age_days - self.STALE_ISSUE_DAYS, 40)

        # Labels that increase urgency
        urgent_labels = {
            "bug": 20,
            "critical": 40,
            "security": 50,
            "urgent": 30,
            "high priority": 25,
        }

        for label in (issue.labels or []):
            label_lower = label.name.lower()
            for keyword, boost in urgent_labels.items():
                if keyword in label_lower:
                    urgency += boost
                    break

        # Comments indicate engagement
        comments_count = issue.comments or 0
        if comments_count > 5:
            urgency += 10

        return min(urgency, 100)

    def _classify_issue(self, issue) -> SignalType:
        """Classify an issue into a signal type.

        Args:
            issue: PyGithub Issue object

        Returns:
            SignalType enum value
        """
        for label in (issue.labels or []):
            label_lower = label.name.lower()
            if "bug" in label_lower or "error" in label_lower:
                return SignalType.BUGFIX
            elif "feature" in label_lower or "enhancement" in label_lower:
                return SignalType.FEATURE
            elif "documentation" in label_lower or "docs" in label_lower:
                return SignalType.DOCUMENTATION
            elif "performance" in label_lower or "slow" in label_lower:
                return SignalType.PERFORMANCE
            elif "security" in label_lower or "vulnerability" in label_lower:
                return SignalType.SECURITY
            elif "refactor" in label_lower or "tech debt" in label_lower:
                return SignalType.REFACTOR
            elif "test" in label_lower or "testing" in label_lower:
                return SignalType.TEST

        # Default based on title
        title_lower = issue.title.lower()
        if any(word in title_lower for word in ["fix", "bug", "error"]):
            return SignalType.BUGFIX
        elif any(word in title_lower for word in ["add", "implement", "feature"]):
            return SignalType.FEATURE
        elif any(word in title_lower for word in ["refactor", "clean", "improve"]):
            return SignalType.REFACTOR

        return SignalType.MAINTENANCE

    def _issue_needs_attention(self, issue, age_days: int) -> bool:
        """Determine if an issue needs attention/action.

        Args:
            issue: PyGithub Issue object
            age_days: Age in days

        Returns:
            True if issue should generate a task
        """
        # Always include stale issues
        if age_days > self.STALE_ISSUE_DAYS:
            return True

        # Include high-priority labels
        for label in (issue.labels or []):
            label_lower = label.name.lower()
            if any(keyword in label_lower for keyword in [
                "good first issue", "help wanted", "up-for-grabs"
            ]):
                return True

        # Skip if already assigned (someone is working on it)
        if issue.assignees:
            return False

        # Include if recently created (within 7 days)
        if age_days < 7:
            return True

        # Otherwise skip
        return False

    def _collect_pull_requests(self, repo) -> List[WorkItem]:
        """Collect signals from pull requests.

        Args:
            repo: PyGithub Repository object

        Returns:
            List of WorkItems from PRs
        """
        items = []

        try:
            # Get open PRs
            prs = repo.get_pulls(state="open", sort="created", direction="desc")

            for pr in prs.get_page(20):
                # Calculate age
                created_at = pr.created_at.replace(tzinfo=None)
                age_days = (datetime.now() - created_at).days

                # Flag stale PRs
                if age_days > self.STALE_PR_DAYS:
                    # Check if it has merge conflicts
                    mergeable = pr.mergeable
                    if mergeable is False:
                        # Has conflicts
                        items.append(WorkItem(
                            source=SignalSource.GITHUB_PR,
                            type=SignalType.MAINTENANCE,
                            title=f"Resolve merge conflicts in PR: {pr.title[:60]}",
                            description=f"PR #{pr.number} has merge conflicts and needs attention.\n\nAge: {age_days} days",
                            location=f"https://github.com/{self.github.repo}/pull/{pr.number}",
                            urgency=60,
                            confidence=0.9,
                            age_days=age_days,
                            metadata={
                                "pr_number": pr.number,
                                "has_conflicts": True,
                                "html_url": pr.html_url,
                                "author": pr.user.login if pr.user else "unknown",
                            },
                        ))
                    else:
                        # Just stale
                        items.append(WorkItem(
                            source=SignalSource.GITHUB_PR,
                            type=SignalType.MAINTENANCE,
                            title=f"Review and merge stale PR: {pr.title[:60]}",
                            description=f"PR #{pr.number} is {age_days} days old.\n\n{pr.title}",
                            location=f"https://github.com/{self.github.repo}/pull/{pr.number}",
                            urgency=40,
                            confidence=0.8,
                            age_days=age_days,
                            metadata={
                                "pr_number": pr.number,
                                "has_conflicts": False,
                                "html_url": pr.html_url,
                                "author": pr.user.login if pr.user else "unknown",
                                "reviewers": [r.login for r in pr.get_reviewers()] if pr.get_reviewers() else [],
                            },
                        ))

                # Check for unapproved PRs with no negative reviews
                reviews = list(pr.get_reviews())
                if reviews:
                    has_approval = any(r.state == "APPROVED" for r in reviews)
                    has_changes = any(r.state == "CHANGES_REQUESTED" for r in reviews)

                    if not has_approval and not has_changes and age_days > 3:
                        items.append(WorkItem(
                            source=SignalSource.GITHUB_PR,
                            type=SignalType.MAINTENANCE,
                            title=f"Review PR: {pr.title[:60]}",
                            description=f"PR #{pr.number} has no approvals yet.\n\n{pr.title}",
                            location=f"https://github.com/{self.github.repo}/pull/{pr.number}",
                            urgency=30,
                            confidence=0.7,
                            age_days=age_days,
                            metadata={
                                "pr_number": pr.number,
                                "reviews_count": len(reviews),
                                "html_url": pr.html_url,
                            },
                        ))

        except Exception as e:
            logger.debug("collector.prs.failed", error=str(e))

        return items

    def _collect_ci_failures(self, repo) -> List[WorkItem]:
        """Collect CI/CD failure signals from PRs.

        Args:
            repo: PyGithub Repository object

        Returns:
            List of WorkItems for CI failures
        """
        items = []

        try:
            # Check recent PRs with failed status
            prs = repo.get_pulls(state="open", sort="updated", direction="desc")

            for pr in prs.get_page(10):
                # Get combined status
                try:
                    combined_status = pr.get_commits().get_page(1)[-1].get_combined_status()

                    if combined_status.state == "failure":
                        # Get failed checks
                        failed_checks = [
                            check for check in combined_status.get_statuses()
                            if check.state == "failure"
                        ]

                        for check in failed_checks[:3]:  # Limit to top 3 failures
                            items.append(WorkItem(
                                source=SignalSource.CI_FAILURE,
                                type=SignalType.BUGFIX,
                                title=f"Fix CI failure: {check.context}",
                                description=f"CI check '{check.context}' failed for PR #{pr.number}.\n\nTarget: {check.target}\n{check.description[:200] if check.description else ''}",
                                location=f"https://github.com/{self.github.repo}/pull/{pr.number}",
                                urgency=75,
                                confidence=0.95,
                                metadata={
                                    "pr_number": pr.number,
                                    "check_context": check.context,
                                    "check_target": check.target,
                                    "html_url": check.html_url,
                                },
                            ))
                except Exception as e:
                    logger.debug("collector.ci.failed_for_pr", pr_number=pr.number, error=str(e))
                    continue

        except Exception as e:
            logger.debug("collector.ci.failed", error=str(e))

        return items

    def _collect_dependabot_alerts(self, repo) -> List[WorkItem]:
        """Collect Dependabot security alerts.

        Args:
            repo: PyGithub Repository object

        Returns:
            List of WorkItems for security alerts
        """
        items = []

        try:
            # Try to get dependabot alerts via GitHub API
            # Note: This requires the newer GitHub API (GraphQL sometimes works better)
            headers = {}
            token = self.github.auth_token or os.environ.get("GITHUB_TOKEN")
            if token:
                headers["Authorization"] = f"token {token}"

            import urllib.request
            import json

            api_url = f"https://api.github.com/repos/{self.github.repo}/dependabot/alerts"
            req = urllib.request.Request(api_url, headers=headers)

            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status == 200:
                        alerts = json.loads(response.read())

                        # Process only open, high/critical severity alerts
                        for alert in alerts[:10]:  # Limit to 10 alerts
                            if alert.get("state") != "open":
                                continue

                            severity = alert.get("security_advisory", {}).get("severity")
                            if severity not in ["high", "critical"]:
                                continue

                            adv = alert.get("security_advisory", {})
                            package = adv.get("package", {})
                            package_name = package.get("name", "unknown")

                            items.append(WorkItem(
                                source=SignalSource.DEPENDABOT,
                                type=SignalType.SECURITY,
                                title=f"Update dependency: {package_name}",
                                description=f"Security alert: {adv.get('summary', 'Security vulnerability')}\n\nSeverity: {severity}\nCVE: {', '.join(adv.get('cve_id', []))}",
                                location=alert.get("html_url"),
                                urgency=90 if severity == "critical" else 80,
                                confidence=0.98,
                                metadata={
                                    "alert_number": alert.get("number"),
                                    "package_name": package_name,
                                    "severity": severity,
                                    "affected_versions": alert.get("security_vulnerability", {}).get("package", {}).get("affected_versions", []),
                                },
                            ))
            except urllib.error.HTTPError as e:
                # 404 means no dependabot alerts (not an error)
                if e.code != 404:
                    logger.debug("collector.dependabot.http_error", code=e.code)
            except Exception as e:
                logger.debug("collector.dependabot.failed", error=str(e))

        except Exception as e:
            logger.debug("collector.dependabot.failed", error=str(e))

        return items
