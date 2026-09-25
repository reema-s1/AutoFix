"""Seeded-bug catalog used for evaluation.

Each bug is a TOML file describing how to break a clean fixture project and
what the correct diagnosis is::

    id = "ringbuf-full-check"
    project = "ringbuf"
    category = "logic"          # build | logic | crash | memory
    symptom = "test_failure"    # build_failure | test_failure | crash
    root_cause = "ring_push compares count > cap ..."
    # Offline diagnosis check: every group must have at least one hit.
    keywords = [["ring_push"], ["full", "capacity", "=="]]

    [[mutation]]
    file = "src/ring.c"
    find = '''if (r->count == r->cap) {'''
    replace = '''if (r->count > r->cap) {'''

The clean project is the known-good fix, so a repair is judged by the
project's own tests rather than by matching a specific patch.
"""

from __future__ import annotations

import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = ("build", "logic", "crash", "memory")
SYMPTOMS = ("build_failure", "test_failure", "crash")


class BugSpecError(ValueError):
    pass


@dataclass(frozen=True)
class Mutation:
    file: str
    find: str
    replace: str


@dataclass(frozen=True)
class SeededBug:
    id: str
    project: str
    category: str
    symptom: str
    root_cause: str
    keywords: tuple[tuple[str, ...], ...]
    mutations: tuple[Mutation, ...]

    def keyword_match(self, diagnosis: str) -> bool:
        text = diagnosis.lower()
        return bool(self.keywords) and all(
            any(word.lower() in text for word in group) for group in self.keywords
        )


def load_bug(path: str | Path) -> SeededBug:
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise BugSpecError(f"{path}: {exc}") from exc

    missing = [k for k in ("id", "project", "category", "symptom", "root_cause", "mutation") if k not in data]
    if missing:
        raise BugSpecError(f"{path}: missing keys {missing}")
    if data["category"] not in CATEGORIES:
        raise BugSpecError(f"{path}: category must be one of {CATEGORIES}")
    if data["symptom"] not in SYMPTOMS:
        raise BugSpecError(f"{path}: symptom must be one of {SYMPTOMS}")
    if data["id"] != path.stem:
        raise BugSpecError(f"{path}: id {data['id']!r} must match the file name")

    mutations = tuple(Mutation(m["file"], m["find"], m["replace"]) for m in data["mutation"])
    if not mutations or any(m.find == m.replace or not m.find for m in mutations):
        raise BugSpecError(f"{path}: every mutation needs a non-empty find that differs from replace")
    keywords = tuple(tuple(group) for group in data.get("keywords", ()))
    return SeededBug(
        id=data["id"],
        project=data["project"],
        category=data["category"],
        symptom=data["symptom"],
        root_cause=" ".join(data["root_cause"].split()),
        keywords=keywords,
        mutations=mutations,
    )


def load_bugs(bugs_dir: str | Path, only: set[str] | None = None) -> list[SeededBug]:
    bugs = [load_bug(p) for p in sorted(Path(bugs_dir).glob("*.toml"))]
    if only:
        unknown = only - {b.id for b in bugs} - {b.project for b in bugs}
        if unknown:
            raise BugSpecError(f"unknown bug ids or projects: {sorted(unknown)}")
        bugs = [b for b in bugs if b.id in only or b.project in only]
    return bugs


def materialize(bug: SeededBug, fixtures_dir: str | Path, dest: str | Path) -> Path:
    """Copy the clean project to ``dest`` and apply the bug's mutations there."""
    source = Path(fixtures_dir) / bug.project
    if not (source / "autofix.toml").is_file():
        raise BugSpecError(f"{bug.id}: fixture project {bug.project!r} not found in {fixtures_dir}")
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("test_runner*", "build"))

    for mutation in bug.mutations:
        target = dest / mutation.file
        text = target.read_text(encoding="utf-8")
        occurrences = text.count(mutation.find)
        if occurrences != 1:
            raise BugSpecError(
                f"{bug.id}: mutation text must occur exactly once in {mutation.file}, found {occurrences}"
            )
        target.write_text(text.replace(mutation.find, mutation.replace), encoding="utf-8", newline="")
    return dest
