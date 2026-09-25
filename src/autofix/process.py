"""Subprocess execution with timeouts and bounded output."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# Windows NTSTATUS codes surfaced as process exit codes when a program crashes.
_WINDOWS_CRASH_CODES = {
    0xC0000005: "access violation (segmentation fault)",
    0xC00000FD: "stack overflow",
    0xC0000094: "integer divide by zero",
    0xC000001D: "illegal instruction",
    0xC0000409: "stack buffer overrun / abort",
    0xC0000374: "heap corruption",
}

_POSIX_SIGNALS = {
    6: "SIGABRT (abort)",
    8: "SIGFPE (arithmetic error)",
    11: "SIGSEGV (segmentation fault)",
    4: "SIGILL (illegal instruction)",
    7: "SIGBUS (bus error)",
}


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int | None  # None if the process timed out
    output: str  # interleaved stdout+stderr
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def crash_reason(self) -> str | None:
        return describe_crash(self.exit_code)


def describe_crash(exit_code: int | None) -> str | None:
    if exit_code is None:
        return None
    # NTSTATUS values can surface either unsigned or as negative 32-bit ints.
    if windows := _WINDOWS_CRASH_CODES.get(exit_code & 0xFFFFFFFF):
        return windows
    if exit_code < 0:
        return _POSIX_SIGNALS.get(-exit_code, f"killed by signal {-exit_code}")
    if exit_code >= 128 + 1 and exit_code - 128 in _POSIX_SIGNALS:
        return _POSIX_SIGNALS[exit_code - 128]
    return None


def run_command(argv: tuple[str, ...] | list[str], cwd: Path, timeout: float) -> CommandResult:
    argv = tuple(argv)
    resolved = (_resolve_program(argv[0], cwd), *argv[1:])
    start = time.monotonic()
    try:
        proc = subprocess.run(
            resolved,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = _decode(exc.stdout or b"")
        return CommandResult(argv, None, output, time.monotonic() - start, timed_out=True)
    except FileNotFoundError:
        return CommandResult(argv, 127, f"command not found: {argv[0]}", time.monotonic() - start)
    except PermissionError as exc:
        return CommandResult(argv, 126, f"cannot execute {argv[0]}: {exc}", time.monotonic() - start)
    return CommandResult(argv, proc.returncode, _decode(proc.stdout), time.monotonic() - start)


def truncate_middle(text: str, limit: int) -> str:
    """Keep the head and tail of long output; compiler errors cluster at both ends."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    omitted = text[head : len(text) - tail].count("\n")
    return f"{text[:head]}\n... [{omitted} lines omitted] ...\n{text[len(text) - tail:]}"


def _resolve_program(program: str, cwd: Path) -> str:
    """Resolve project-relative executables such as ``build/tests``.

    On Windows, ``CreateProcess`` resolves relative program paths against the
    *parent's* working directory rather than ``cwd``, so make them absolute.
    """
    if not any(sep in program for sep in ("/", os.sep)) or Path(program).is_absolute():
        return program
    candidate = (cwd / program).resolve()
    if os.name == "nt" and not candidate.suffix and candidate.with_suffix(".exe").exists():
        return str(candidate.with_suffix(".exe"))
    return str(candidate)


def _decode(data: bytes | str) -> str:
    if isinstance(data, str):
        return data
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")
