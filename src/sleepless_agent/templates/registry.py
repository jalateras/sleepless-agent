"""Template registry for managing task templates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TemplateParameter:
    """A parameter definition for a template."""

    name: str
    required: bool = True
    description: str = ""
    default: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "TemplateParameter":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            required=data.get("required", True),
            description=data.get("description", ""),
            default=data.get("default"),
        )


@dataclass
class TaskTemplate:
    """A task template with placeholders for parameterized task creation."""

    name: str
    description: str
    prompt: str
    category: str = "general"
    priority: str = "serious"
    parameters: list[TemplateParameter] = field(default_factory=list)
    context_injection: bool = True  # Whether to inject codebase context
    source_path: Optional[str] = None  # Path to the template file

    @classmethod
    def from_dict(cls, data: dict, source_path: Optional[str] = None) -> "TaskTemplate":
        """Create template from dictionary (loaded from YAML)."""
        parameters = [
            TemplateParameter.from_dict(p) for p in data.get("parameters", [])
        ]
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            prompt=data.get("prompt", ""),
            category=data.get("category", "general"),
            priority=data.get("priority", "serious"),
            parameters=parameters,
            context_injection=data.get("context_injection", True),
            source_path=source_path,
        )

    def get_required_parameters(self) -> list[TemplateParameter]:
        """Get list of required parameters."""
        return [p for p in self.parameters if p.required]

    def get_optional_parameters(self) -> list[TemplateParameter]:
        """Get list of optional parameters."""
        return [p for p in self.parameters if not p.required]

    def validate_args(self, args: dict[str, str]) -> tuple[bool, list[str]]:
        """Validate provided arguments against template parameters.

        Args:
            args: Dictionary of argument name -> value

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        for param in self.get_required_parameters():
            if param.name not in args and param.default is None:
                errors.append(f"Missing required parameter: {param.name}")
        return (len(errors) == 0, errors)

    def expand(self, args: dict[str, str], context: Optional[str] = None) -> str:
        """Expand the template with provided arguments.

        Args:
            args: Dictionary of argument name -> value
            context: Optional codebase context to inject

        Returns:
            Expanded prompt string
        """
        # Build full args with defaults
        full_args = {}
        for param in self.parameters:
            if param.name in args:
                full_args[param.name] = args[param.name]
            elif param.default is not None:
                full_args[param.name] = param.default

        # Expand the prompt
        expanded = self.prompt
        for key, value in full_args.items():
            expanded = expanded.replace(f"{{{key}}}", value)

        # Inject context if enabled
        if self.context_injection and context:
            expanded = f"{context}\n\n---\n\n{expanded}"

        return expanded


class TemplateRegistry:
    """Singleton registry for managing task templates.

    Loads templates from:
    1. Built-in templates (shipped with package)
    2. User templates (~/.sleepless/templates/)
    3. Project templates (.sleepless/templates/)

    Later sources override earlier ones with same name.
    """

    _instance: Optional["TemplateRegistry"] = None

    def __new__(cls) -> "TemplateRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._templates: dict[str, TaskTemplate] = {}
        self._initialized = True

    def register(self, template: TaskTemplate) -> None:
        """Register a template.

        Args:
            template: Template to register
        """
        if template.name in self._templates:
            logger.debug(
                "template.override",
                name=template.name,
                old_source=self._templates[template.name].source_path,
                new_source=template.source_path,
            )
        self._templates[template.name] = template
        logger.debug("template.registered", name=template.name, source=template.source_path)

    def get(self, name: str) -> Optional[TaskTemplate]:
        """Get a template by name.

        Args:
            name: Template name

        Returns:
            Template or None if not found
        """
        return self._templates.get(name)

    def list_all(self) -> list[TaskTemplate]:
        """Get all registered templates.

        Returns:
            List of all templates
        """
        return list(self._templates.values())

    def list_by_category(self, category: str) -> list[TaskTemplate]:
        """Get templates by category.

        Args:
            category: Category to filter by

        Returns:
            List of templates in category
        """
        return [t for t in self._templates.values() if t.category == category]

    def get_categories(self) -> list[str]:
        """Get all unique categories.

        Returns:
            List of category names
        """
        return sorted(set(t.category for t in self._templates.values()))

    def clear(self) -> None:
        """Clear all templates (mainly for testing)."""
        self._templates.clear()

    def template_count(self) -> int:
        """Get number of registered templates."""
        return len(self._templates)

    def format_template_list(self) -> str:
        """Format templates as a readable list.

        Returns:
            Formatted string for display
        """
        if not self._templates:
            return "No templates available."

        lines = ["*Available Templates:*\n"]

        # Group by category
        by_category: dict[str, list[TaskTemplate]] = {}
        for template in self._templates.values():
            cat = template.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(template)

        for category in sorted(by_category.keys()):
            lines.append(f"\n*{category.title()}:*")
            for template in sorted(by_category[category], key=lambda t: t.name):
                params = ", ".join(p.name for p in template.get_required_parameters())
                param_str = f" ({params})" if params else ""
                lines.append(f"  • `{template.name}`{param_str} - {template.description}")

        return "\n".join(lines)
