"""Template loader for discovering and loading templates from various sources."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from sleepless_agent.monitoring.logging import get_logger
from sleepless_agent.templates.registry import TaskTemplate, TemplateRegistry

logger = get_logger(__name__)


class TemplateLoader:
    """Loads templates from filesystem and registers them.

    Loading order (later overrides earlier):
    1. Built-in templates (package/templates/builtin/)
    2. User templates (~/.sleepless/templates/)
    3. Project templates (.sleepless/templates/)
    """

    def __init__(
        self,
        registry: Optional[TemplateRegistry] = None,
        custom_paths: Optional[list[str]] = None,
    ):
        """Initialize loader.

        Args:
            registry: Template registry to populate (uses singleton if None)
            custom_paths: Additional paths to load templates from
        """
        self.registry = registry or TemplateRegistry()
        self.custom_paths = custom_paths or []
        self._loaded = False

    def load_all(self) -> int:
        """Load templates from all sources.

        Returns:
            Number of templates loaded
        """
        if self._loaded:
            return self.registry.template_count()

        initial_count = self.registry.template_count()

        # 1. Load built-in templates
        builtin_path = Path(__file__).parent / "builtin"
        self._load_from_directory(builtin_path, "builtin")

        # 2. Load user templates (~/.sleepless/templates/)
        user_path = Path.home() / ".sleepless" / "templates"
        if user_path.exists():
            self._load_from_directory(user_path, "user")

        # 3. Load project templates (.sleepless/templates/)
        project_path = Path.cwd() / ".sleepless" / "templates"
        if project_path.exists():
            self._load_from_directory(project_path, "project")

        # 4. Load from custom paths
        for custom_path in self.custom_paths:
            path = Path(custom_path).expanduser()
            if path.exists():
                self._load_from_directory(path, "custom")

        self._loaded = True
        loaded_count = self.registry.template_count() - initial_count

        logger.info(
            "templates.loaded",
            total=self.registry.template_count(),
            loaded_this_time=loaded_count,
        )

        return loaded_count

    def _load_from_directory(self, directory: Path, source: str) -> int:
        """Load all templates from a directory.

        Args:
            directory: Directory to load from
            source: Source identifier for logging

        Returns:
            Number of templates loaded
        """
        if not directory.exists():
            logger.debug("templates.dir_not_found", path=str(directory), source=source)
            return 0

        loaded = 0
        for template_file in directory.glob("*.yaml"):
            try:
                template = self._load_template_file(template_file)
                if template:
                    self.registry.register(template)
                    loaded += 1
            except Exception as exc:
                logger.warning(
                    "templates.load_failed",
                    file=str(template_file),
                    error=str(exc),
                )

        # Also check for .yml extension
        for template_file in directory.glob("*.yml"):
            try:
                template = self._load_template_file(template_file)
                if template:
                    self.registry.register(template)
                    loaded += 1
            except Exception as exc:
                logger.warning(
                    "templates.load_failed",
                    file=str(template_file),
                    error=str(exc),
                )

        logger.debug(
            "templates.loaded_from_dir",
            directory=str(directory),
            source=source,
            count=loaded,
        )

        return loaded

    def _load_template_file(self, path: Path) -> Optional[TaskTemplate]:
        """Load a single template from a YAML file.

        Args:
            path: Path to template file

        Returns:
            TaskTemplate or None if invalid
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            logger.debug("templates.empty_file", path=str(path))
            return None

        # Validate required fields
        required_fields = ["name", "prompt"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            logger.warning(
                "templates.missing_fields",
                path=str(path),
                missing=missing,
            )
            return None

        template = TaskTemplate.from_dict(data, source_path=str(path))

        # Validate template name (alphanumeric, hyphens, underscores)
        if not template.name.replace("-", "").replace("_", "").isalnum():
            logger.warning(
                "templates.invalid_name",
                path=str(path),
                name=template.name,
            )
            return None

        return template

    def reload(self) -> int:
        """Force reload all templates.

        Returns:
            Number of templates loaded
        """
        self._loaded = False
        self.registry.clear()
        return self.load_all()

    def get_template(self, name: str) -> Optional[TaskTemplate]:
        """Get a template by name, loading if needed.

        Args:
            name: Template name

        Returns:
            Template or None if not found
        """
        if not self._loaded:
            self.load_all()
        return self.registry.get(name)

    def expand_template(
        self,
        name: str,
        args: dict[str, str],
        context: Optional[str] = None,
    ) -> tuple[Optional[str], list[str]]:
        """Expand a template with arguments.

        Args:
            name: Template name
            args: Argument dictionary
            context: Optional codebase context

        Returns:
            Tuple of (expanded prompt or None, list of errors)
        """
        template = self.get_template(name)
        if not template:
            return (None, [f"Template not found: {name}"])

        is_valid, errors = template.validate_args(args)
        if not is_valid:
            return (None, errors)

        expanded = template.expand(args, context)
        return (expanded, [])
