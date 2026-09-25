"""Path validation that confines every tool to the target project directory.

All paths the model supplies are untrusted. They are resolved against the
project root (following symlinks), and rejected if the resolved location
escapes the root. Writes are additionally checked against the project's
protected globs so the agent cannot "fix" a failing test by editing it.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path, PurePosixPath

# Directories never searched or listed: VCS metadata, build output, caches.
DEFAULT_IGNORED_DIRS = frozenset(
    {".git", ".hg", ".svn", "build", "out", "cmake-build-debug", "cmake-build-release",
     "__pycache__", ".venv", "node_modules", ".cache"}
)


class SandboxViolation(PermissionError):
    """Raised when a tool tries to touch something outside its allowed scope."""


class Sandbox:
    def __init__(
        self,
        root: str | os.PathLike[str],
        protected: tuple[str, ...] = (),
        ignored_dirs: frozenset[str] | set[str] = DEFAULT_IGNORED_DIRS,
    ) -> None:
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.protected = tuple(protected)
        self.ignored_dirs = frozenset(ignored_dirs)

    def resolve(self, user_path: str) -> Path:
        """Resolve a model-supplied path to an absolute path inside the root."""
        if not isinstance(user_path, str) or not user_path.strip():
            raise SandboxViolation("path must be a non-empty string")
        if "\x00" in user_path:
            raise SandboxViolation("path contains a NUL byte")

        candidate = Path(user_path.strip())
        if candidate.is_absolute() or candidate.drive:
            # Absolute paths are tolerated only if they already point inside the root.
            target = candidate.resolve()
        else:
            target = (self.root / candidate).resolve()

        if not target.is_relative_to(self.root):
            raise SandboxViolation(f"path escapes the project directory: {user_path}")
        return target

    def relative(self, path: Path) -> str:
        """Return a project-relative POSIX path, as shown to the model."""
        return path.resolve().relative_to(self.root).as_posix() or "."

    def is_protected(self, path: Path) -> bool:
        rel = self.relative(path)
        return any(_glob_match(rel, pattern) for pattern in self.protected)

    def check_readable(self, user_path: str) -> Path:
        target = self.resolve(user_path)
        if not target.exists():
            raise FileNotFoundError(f"no such file: {self.relative(target)}")
        if not target.is_file():
            raise IsADirectoryError(f"not a regular file: {self.relative(target)}")
        return target

    def check_writable(self, user_path: str) -> Path:
        target = self.resolve(user_path)
        if target == self.root:
            raise SandboxViolation("cannot write to the project root itself")
        if self.is_protected(target):
            raise SandboxViolation(
                f"{self.relative(target)} is protected and may not be modified "
                "(fix the code under test, not the tests or build configuration)"
            )
        if any(part in self.ignored_dirs for part in PurePosixPath(self.relative(target)).parts[:-1]):
            raise SandboxViolation(f"{self.relative(target)} is inside an ignored directory")
        return target

    def iter_files(self, globs: tuple[str, ...]) -> list[Path]:
        """List project files matching any of ``globs``, skipping ignored dirs."""
        matches: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in self.ignored_dirs)
            for name in sorted(filenames):
                path = Path(dirpath, name)
                rel = path.relative_to(self.root).as_posix()
                if any(_glob_match(rel, g) for g in globs):
                    resolved = path.resolve()
                    if resolved.is_relative_to(self.root):
                        matches.append(resolved)
        return matches


def _glob_match(rel_posix: str, pattern: str) -> bool:
    """Glob matching where ``**/`` also matches zero directories."""
    if fnmatch.fnmatchcase(rel_posix, pattern):
        return True
    if pattern.startswith("**/") and fnmatch.fnmatchcase(rel_posix, pattern[3:]):
        return True
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        return rel_posix == prefix or rel_posix.startswith(prefix + "/")
    return False
