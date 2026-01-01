"""Conventions analyzer for extracting project configuration and style conventions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from sleepless_agent.monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProjectConventions:
    """Extracted project conventions and configuration."""

    # Project metadata
    project_name: Optional[str] = None
    python_version: Optional[str] = None
    package_manager: Optional[str] = None  # pip, poetry, uv, etc.

    # Code style
    line_length: Optional[int] = None
    indent_style: Optional[str] = None  # "space" or "tab"
    indent_size: Optional[int] = None
    quote_style: Optional[str] = None  # "single" or "double"

    # Testing
    test_framework: Optional[str] = None  # pytest, unittest, etc.
    test_directory: Optional[str] = None

    # Linting/Formatting
    linters: list[str] = field(default_factory=list)
    formatters: list[str] = field(default_factory=list)

    # Type checking
    type_checker: Optional[str] = None  # mypy, pyright, etc.
    strict_typing: bool = False

    # Additional conventions
    docstring_style: Optional[str] = None  # google, numpy, sphinx
    naming_conventions: dict[str, str] = field(default_factory=dict)

    # Raw config files found
    config_files: list[str] = field(default_factory=list)


class ConventionsAnalyzer:
    """Analyzes project configuration files to extract conventions."""

    # Configuration files to look for
    CONFIG_FILES = [
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        ".editorconfig",
        ".pre-commit-config.yaml",
        "ruff.toml",
        ".ruff.toml",
        ".flake8",
        "mypy.ini",
        ".mypy.ini",
        "pyrightconfig.json",
        ".pylintrc",
        "pytest.ini",
        "conftest.py",
        "tox.ini",
        ".python-version",
        "Pipfile",
        "poetry.lock",
        "uv.lock",
        "requirements.txt",
    ]

    def __init__(self, root_path: Path):
        """Initialize the conventions analyzer.

        Args:
            root_path: Root directory of the project
        """
        self.root_path = Path(root_path).resolve()

    def analyze(self) -> ProjectConventions:
        """Analyze project configuration files and extract conventions.

        Returns:
            ProjectConventions with extracted information
        """
        logger.debug("context.conventions.analyzing", root=str(self.root_path))
        conventions = ProjectConventions()

        # Check which config files exist
        for config_file in self.CONFIG_FILES:
            config_path = self.root_path / config_file
            if config_path.exists():
                conventions.config_files.append(config_file)

        # Parse pyproject.toml if it exists (most informative)
        pyproject_path = self.root_path / "pyproject.toml"
        if pyproject_path.exists():
            self._parse_pyproject(pyproject_path, conventions)

        # Parse .editorconfig if it exists
        editorconfig_path = self.root_path / ".editorconfig"
        if editorconfig_path.exists():
            self._parse_editorconfig(editorconfig_path, conventions)

        # Infer package manager
        conventions.package_manager = self._detect_package_manager()

        # Detect test framework and directory
        self._detect_testing(conventions)

        # Detect linters and formatters
        self._detect_tools(conventions)

        logger.debug(
            "context.conventions.complete",
            config_files=len(conventions.config_files),
            package_manager=conventions.package_manager,
        )
        return conventions

    def _parse_pyproject(self, path: Path, conventions: ProjectConventions) -> None:
        """Parse pyproject.toml for project conventions.

        Args:
            path: Path to pyproject.toml
            conventions: Conventions object to update
        """
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore
            except ImportError:
                logger.debug("context.conventions.no_toml_parser")
                return

        try:
            content = path.read_text()
            data = tomllib.loads(content)
        except Exception as e:
            logger.debug("context.conventions.pyproject_parse_error", error=str(e))
            return

        # Extract project metadata
        project = data.get("project", {})
        conventions.project_name = project.get("name")
        requires_python = project.get("requires-python")
        if requires_python:
            conventions.python_version = requires_python

        # Check for ruff configuration
        ruff_config = data.get("tool", {}).get("ruff", {})
        if ruff_config:
            conventions.linters.append("ruff")
            conventions.formatters.append("ruff")
            if "line-length" in ruff_config:
                conventions.line_length = ruff_config["line-length"]
            lint_config = ruff_config.get("lint", {})
            if lint_config.get("select"):
                conventions.linters.append("ruff (extensive rules)")

        # Check for black configuration
        black_config = data.get("tool", {}).get("black", {})
        if black_config:
            conventions.formatters.append("black")
            if "line-length" in black_config:
                conventions.line_length = black_config["line-length"]

        # Check for mypy configuration
        mypy_config = data.get("tool", {}).get("mypy", {})
        if mypy_config:
            conventions.type_checker = "mypy"
            conventions.strict_typing = mypy_config.get("strict", False)

        # Check for pytest configuration
        pytest_config = data.get("tool", {}).get("pytest", {})
        if pytest_config:
            conventions.test_framework = "pytest"

        # Check for isort (indicates import sorting preference)
        isort_config = data.get("tool", {}).get("isort", {})
        if isort_config:
            conventions.formatters.append("isort")

    def _parse_editorconfig(self, path: Path, conventions: ProjectConventions) -> None:
        """Parse .editorconfig for style conventions.

        Args:
            path: Path to .editorconfig
            conventions: Conventions object to update
        """
        try:
            content = path.read_text()
        except Exception as e:
            logger.debug("context.conventions.editorconfig_read_error", error=str(e))
            return

        # Simple parser for .editorconfig
        current_section = None
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue

            if line.startswith("["):
                current_section = line[1:-1] if line.endswith("]") else line[1:]
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().lower()
                value = value.strip().lower()

                # Apply Python-specific or global settings
                if current_section in ["*", "*.py", "[*.py]"]:
                    if key == "indent_style":
                        conventions.indent_style = value
                    elif key == "indent_size":
                        try:
                            conventions.indent_size = int(value)
                        except ValueError:
                            pass
                    elif key == "max_line_length":
                        try:
                            conventions.line_length = int(value)
                        except ValueError:
                            pass

    def _detect_package_manager(self) -> Optional[str]:
        """Detect the package manager used by the project.

        Returns:
            Package manager name or None
        """
        if (self.root_path / "uv.lock").exists():
            return "uv"
        elif (self.root_path / "poetry.lock").exists():
            return "poetry"
        elif (self.root_path / "Pipfile.lock").exists():
            return "pipenv"
        elif (self.root_path / "Pipfile").exists():
            return "pipenv"
        elif (self.root_path / "requirements.txt").exists():
            return "pip"
        elif (self.root_path / "pyproject.toml").exists():
            # Check if pyproject.toml has poetry or other backend
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore
                except ImportError:
                    return None

            try:
                content = (self.root_path / "pyproject.toml").read_text()
                data = tomllib.loads(content)
                build_backend = data.get("build-system", {}).get("build-backend", "")
                if "poetry" in build_backend:
                    return "poetry"
                elif "hatchling" in build_backend:
                    return "hatch"
                elif "flit" in build_backend:
                    return "flit"
            except Exception:
                pass

        return None

    def _detect_testing(self, conventions: ProjectConventions) -> None:
        """Detect testing framework and directory.

        Args:
            conventions: Conventions object to update
        """
        # Check for common test directories
        for test_dir in ["tests", "test", "src/tests"]:
            if (self.root_path / test_dir).is_dir():
                conventions.test_directory = test_dir
                break

        # Detect test framework from files
        if (self.root_path / "pytest.ini").exists() or (self.root_path / "conftest.py").exists():
            conventions.test_framework = "pytest"
        elif conventions.test_directory:
            # Check for pytest markers in test files
            test_path = self.root_path / conventions.test_directory
            if test_path.exists():
                for py_file in test_path.glob("*.py"):
                    try:
                        content = py_file.read_text()
                        if "import pytest" in content or "@pytest" in content:
                            conventions.test_framework = "pytest"
                            break
                        elif "import unittest" in content:
                            conventions.test_framework = "unittest"
                            break
                    except Exception:
                        continue

    def _detect_tools(self, conventions: ProjectConventions) -> None:
        """Detect linting and formatting tools.

        Args:
            conventions: Conventions object to update
        """
        # Check for tool-specific config files
        if (self.root_path / ".flake8").exists():
            if "flake8" not in conventions.linters:
                conventions.linters.append("flake8")

        if (self.root_path / ".pylintrc").exists():
            if "pylint" not in conventions.linters:
                conventions.linters.append("pylint")

        if (self.root_path / "mypy.ini").exists() or (self.root_path / ".mypy.ini").exists():
            conventions.type_checker = "mypy"

        if (self.root_path / "pyrightconfig.json").exists():
            conventions.type_checker = "pyright"

        if (self.root_path / "ruff.toml").exists() or (self.root_path / ".ruff.toml").exists():
            if "ruff" not in conventions.linters:
                conventions.linters.append("ruff")
                conventions.formatters.append("ruff")

        # Check pre-commit config for additional tools
        precommit_path = self.root_path / ".pre-commit-config.yaml"
        if precommit_path.exists():
            try:
                content = precommit_path.read_text()
                # Simple detection from pre-commit hooks
                if "black" in content and "black" not in conventions.formatters:
                    conventions.formatters.append("black")
                if "ruff" in content and "ruff" not in conventions.linters:
                    conventions.linters.append("ruff")
                if "mypy" in content and not conventions.type_checker:
                    conventions.type_checker = "mypy"
                if "isort" in content and "isort" not in conventions.formatters:
                    conventions.formatters.append("isort")
            except Exception:
                pass

    def format_conventions(self, conventions: Optional[ProjectConventions] = None) -> str:
        """Format conventions as a readable string.

        Args:
            conventions: Conventions to format (default: analyze and use result)

        Returns:
            Formatted conventions string
        """
        if conventions is None:
            conventions = self.analyze()

        lines = ["Project Conventions:"]

        if conventions.project_name:
            lines.append(f"  Project: {conventions.project_name}")
        if conventions.python_version:
            lines.append(f"  Python: {conventions.python_version}")
        if conventions.package_manager:
            lines.append(f"  Package Manager: {conventions.package_manager}")

        # Style
        style_parts = []
        if conventions.line_length:
            style_parts.append(f"line-length={conventions.line_length}")
        if conventions.indent_style:
            size = conventions.indent_size or 4
            style_parts.append(f"{conventions.indent_style}s ({size})")
        if style_parts:
            lines.append(f"  Style: {', '.join(style_parts)}")

        # Testing
        if conventions.test_framework:
            test_info = conventions.test_framework
            if conventions.test_directory:
                test_info += f" ({conventions.test_directory}/)"
            lines.append(f"  Testing: {test_info}")

        # Tools
        if conventions.linters:
            lines.append(f"  Linters: {', '.join(conventions.linters)}")
        if conventions.formatters:
            lines.append(f"  Formatters: {', '.join(conventions.formatters)}")
        if conventions.type_checker:
            strict = " (strict)" if conventions.strict_typing else ""
            lines.append(f"  Type Checker: {conventions.type_checker}{strict}")

        # Config files
        if conventions.config_files:
            lines.append(f"  Config Files: {', '.join(conventions.config_files)}")

        return "\n".join(lines)
