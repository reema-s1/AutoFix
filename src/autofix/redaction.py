"""Scrub sensitive data from text before it reaches the model or a trace.

Build logs and source trees leak more than you would expect: absolute
paths reveal user names and machine layout, CI environments echo tokens,
and config files carry credentials. Every tool observation and every trace
event passes through :class:`Redactor` first.

Redaction rules, applied in order:

1. PEM private-key blocks.
2. Literal values of sensitive environment variables (``*_TOKEN``, ``*KEY*``, ...).
3. Credential-shaped strings (cloud keys, API tokens, JWTs, bearer tokens,
   ``password=...`` assignments, credentials embedded in URLs).
4. The project root, rewritten to project-relative paths so the model can
   still act on compiler locations.
5. Any remaining absolute path, reduced to ``<external>/<basename>``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SENSITIVE_ENV_NAME = re.compile(
    r"(KEY|TOKEN|SECRET|PASSW(OR)?D|PASSPHRASE|CREDENTIAL|AUTH|SESSION|COOKIE|PRIVATE|DSN)",
    re.IGNORECASE,
)
_MIN_SECRET_LEN = 6

_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)

# (label, pattern). A named ``secret`` group marks the part to replace; otherwise the
# whole match is replaced. Patterns are conservative so they do not eat source code.
_CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}")),
    ("api-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("slack-token", re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    ("bearer-token", re.compile(r"(?i)\bbearer\s+(?P<secret>[A-Za-z0-9._~+/\-]{16,}=*)")),
    ("url-credentials", re.compile(r"(?i)\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:(?P<secret>[^\s@/]+)@")),
    # password = "hunter22" / "api_key": "..." - only quoted values, so that C code
    # such as `token = next_token(p);` is left alone.
    (
        "secret-assignment",
        re.compile(
            r"(?i)\b[\w.\-]*(?:password|passwd|secret|api[_\-]?key|access[_\-]?key|auth[_\-]?token)[\w.\-]*"
            r"[\"']?\s*[:=]\s*([\"'])(?P<secret>[^\"'\s]{6,})\1"
        ),
    ),
    # Shell / .env style: DB_PASSWORD=..., GITHUB_TOKEN=...
    (
        "secret-assignment",
        re.compile(
            r"\b[A-Z0-9_]*(?:PASSWORD|PASSWD|SECRET|API_KEY|ACCESS_KEY|TOKEN)[A-Z0-9_]*"
            r"=(?P<secret>[^\s\"',;]{6,})"
        ),
    ),
)

_WINDOWS_ABS_PATH = re.compile(r"(?<![\w])[A-Za-z]:[\\/][^\s:\"'<>|*?,;()\[\]]*")
_POSIX_ABS_PATH = re.compile(r"(?<![\w.~:/<>\-])/(?:[\w.+\-@]+/)+[\w.+\-@]*")


class Redactor:
    def __init__(
        self,
        project_root: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        extra_secrets: tuple[str, ...] = (),
    ) -> None:
        self._root_patterns = _root_patterns(Path(project_root).resolve()) if project_root else []
        env = os.environ if env is None else env
        secrets = {
            name: value
            for name, value in env.items()
            if SENSITIVE_ENV_NAME.search(name) and len(value) >= _MIN_SECRET_LEN
        }
        # Longest first so a secret that contains another is replaced whole.
        self._env_secrets = sorted(secrets.items(), key=lambda kv: -len(kv[1]))
        self._extra_secrets = sorted((s for s in extra_secrets if s), key=len, reverse=True)

    def redact(self, text: str) -> str:
        if not text:
            return text
        text = _PRIVATE_KEY_BLOCK.sub("<redacted:private-key>", text)
        for name, value in self._env_secrets:
            text = text.replace(value, f"<redacted:env:{name}>")
        for value in self._extra_secrets:
            text = text.replace(value, "<redacted:secret>")
        for label, pattern in _CREDENTIAL_PATTERNS:
            text = pattern.sub(_credential_replacer(label), text)
        for pattern, replacement in self._root_patterns:
            text = pattern.sub(replacement, text)
        text = _WINDOWS_ABS_PATH.sub(_external_path, text)
        text = _POSIX_ABS_PATH.sub(_external_path, text)
        return text

    def redact_obj(self, value: Any) -> Any:
        """Recursively redact every string inside dicts, lists and tuples."""
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, Mapping):
            return {k: self.redact_obj(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self.redact_obj(v) for v in value)
        return value


def _credential_replacer(label: str):
    """Replace the ``secret`` group if the pattern has one, else the whole match."""

    def replace(match: re.Match[str]) -> str:
        if "secret" not in match.re.groupindex:
            return f"<redacted:{label}>"
        whole, (lo, hi) = match.group(0), match.span("secret")
        lo, hi = lo - match.start(), hi - match.start()
        return f"{whole[:lo]}<redacted:{label}>{whole[hi:]}"

    return replace


def _root_patterns(root: Path) -> list[tuple[re.Pattern[str], str]]:
    """Patterns mapping ``<root>/x`` to ``x`` and a bare ``<root>`` to ``.``."""
    variants = {str(root), root.as_posix()}
    if os.name == "nt":
        # Tools on Windows print either separator and sometimes a lowercase drive.
        variants |= {v[0].lower() + v[1:] for v in list(variants)}
    flags = re.IGNORECASE if os.name == "nt" else 0
    patterns = []
    for variant in sorted(variants, key=len, reverse=True):
        body = re.escape(variant).replace(r"\\", r"[\\/]").replace("/", r"[\\/]")
        patterns.append((re.compile(body + r"[\\/]", flags), ""))
        patterns.append((re.compile(body + r"(?![\w.\-])", flags), "."))
    return patterns


def _external_path(match: re.Match[str]) -> str:
    """Keep a file's basename (``stdio.h`` is useful context); drop directories,
    whose last component is often a user or machine name."""
    path = match.group(0)
    if path.endswith(("/", "\\")):
        return "<external>"
    basename = re.split(r"[\\/]", path)[-1]
    return f"<external>/{basename}" if "." in basename.strip(".") else "<external>"
