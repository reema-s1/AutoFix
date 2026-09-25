"""Per-project configuration, loaded from ``autofix.toml`` at the project root.

Example::

    [build]
    command = ["gcc", "-std=c99", "-Wall", "-Werror", "-Isrc", "-o", "build/tests", "src/ring.c", "tests/test_ring.c"]
    timeout = 120

    [test]
    command = ["build/tests"]
    format = "minitest"      # "minitest" | "exit-code"
    timeout = 60

    [sandbox]
    protected = ["tests/**", "autofix.toml"]
    source_globs = ["**/*.c", "**/*.h", "**/*.cpp", "**/*.hpp"]
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_FILENAME = "autofix.toml"

DEFAULT_SOURCE_GLOBS = (
    "**/*.c",
    "**/*.h",
    "**/*.cc",
    "**/*.cpp",
    "**/*.cxx",
    "**/*.hpp",
    "**/*.hh",
    "**/CMakeLists.txt",
    "**/Makefile",
    "**/*.cmake",
)

TEST_FORMATS = ("minitest", "exit-code")


class ConfigError(ValueError):
    """Raised when a project configuration is missing or malformed."""


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    timeout: float

    def __post_init__(self) -> None:
        if not self.argv:
            raise ConfigError("command must be a non-empty list of arguments")
        if self.timeout <= 0:
            raise ConfigError("timeout must be positive")


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    build: CommandSpec
    test: CommandSpec
    test_format: str = "minitest"
    protected: tuple[str, ...] = ()
    source_globs: tuple[str, ...] = DEFAULT_SOURCE_GLOBS
    extra_ignored_dirs: tuple[str, ...] = ()

    @classmethod
    def load(cls, root: str | Path) -> "ProjectConfig":
        root = Path(root).resolve()
        path = root / CONFIG_FILENAME
        if not path.is_file():
            raise ConfigError(f"no {CONFIG_FILENAME} found in {root}")
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
        return cls.from_dict(root, data)

    @classmethod
    def from_dict(cls, root: Path, data: dict) -> "ProjectConfig":
        build = _command(data, "build", default_timeout=300)
        test = _command(data, "test", default_timeout=120)

        test_format = data.get("test", {}).get("format", "minitest")
        if test_format not in TEST_FORMATS:
            raise ConfigError(f"test.format must be one of {TEST_FORMATS}, got {test_format!r}")

        sandbox = data.get("sandbox", {})
        protected = tuple(sandbox.get("protected", ()))
        source_globs = tuple(sandbox.get("source_globs", DEFAULT_SOURCE_GLOBS))
        ignored = tuple(sandbox.get("ignored_dirs", ()))

        return cls(
            root=Path(root).resolve(),
            build=build,
            test=test,
            test_format=test_format,
            protected=protected,
            source_globs=source_globs,
            extra_ignored_dirs=ignored,
        )


def _command(data: dict, section: str, default_timeout: float) -> CommandSpec:
    table = data.get(section)
    if not isinstance(table, dict) or "command" not in table:
        raise ConfigError(f"[{section}] section with a 'command' list is required")
    argv = table["command"]
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        raise ConfigError(f"{section}.command must be a list of strings")
    return CommandSpec(argv=tuple(argv), timeout=float(table.get("timeout", default_timeout)))
