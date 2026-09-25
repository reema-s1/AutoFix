import os

import pytest

from autofix.sandbox import Sandbox, SandboxViolation


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "ring.c").write_text("int x;\n")
    (tmp_path / "src" / "ring.h").write_text("extern int x;\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ring.c").write_text("int main(void){return 0;}\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "gen.c").write_text("/* generated */\n")
    (tmp_path / "autofix.toml").write_text("")
    return tmp_path


@pytest.fixture
def sandbox(project):
    return Sandbox(project, protected=("tests/**", "autofix.toml"))


def test_resolves_relative_paths(sandbox, project):
    assert sandbox.resolve("src/ring.c") == (project / "src" / "ring.c").resolve()
    assert sandbox.resolve("./src/../src/ring.c") == (project / "src" / "ring.c").resolve()


@pytest.mark.parametrize("bad", ["../outside.c", "src/../../etc/passwd", "..", "src/../../../"])
def test_rejects_traversal(sandbox, bad):
    with pytest.raises(SandboxViolation):
        sandbox.resolve(bad)


def test_rejects_absolute_path_outside_root(sandbox, tmp_path_factory):
    other = tmp_path_factory.mktemp("elsewhere") / "secret.txt"
    other.write_text("nope")
    with pytest.raises(SandboxViolation):
        sandbox.resolve(str(other))


def test_accepts_absolute_path_inside_root(sandbox, project):
    inside = project / "src" / "ring.c"
    assert sandbox.resolve(str(inside)) == inside.resolve()


@pytest.mark.parametrize("bad", ["", "   ", "src/\x00ring.c"])
def test_rejects_malformed_paths(sandbox, bad):
    with pytest.raises(SandboxViolation):
        sandbox.resolve(bad)


def test_rejects_symlink_escape(sandbox, project, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "leak.txt").write_text("secret")
    link = project / "src" / "link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this platform")
    with pytest.raises(SandboxViolation):
        sandbox.check_readable("src/link/leak.txt")


def test_protected_paths_are_not_writable(sandbox):
    with pytest.raises(SandboxViolation, match="protected"):
        sandbox.check_writable("tests/test_ring.c")
    with pytest.raises(SandboxViolation, match="protected"):
        sandbox.check_writable("autofix.toml")
    assert sandbox.check_writable("src/ring.c").name == "ring.c"


def test_ignored_dirs_are_not_writable(sandbox):
    with pytest.raises(SandboxViolation, match="ignored"):
        sandbox.check_writable("build/gen.c")


def test_check_readable_reports_missing_and_dirs(sandbox):
    with pytest.raises(FileNotFoundError):
        sandbox.check_readable("src/missing.c")
    with pytest.raises(IsADirectoryError):
        sandbox.check_readable("src")


def test_iter_files_skips_ignored_dirs(sandbox):
    rels = [sandbox.relative(p) for p in sandbox.iter_files(("**/*.c", "**/*.h"))]
    assert rels == ["src/ring.c", "src/ring.h", "tests/test_ring.c"]
