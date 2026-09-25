import shutil
from pathlib import Path

import pytest

from autofix.config import ProjectConfig
from autofix.tools import Toolbox

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
PROJECTS = sorted(p.name for p in FIXTURES.iterdir() if (p / "autofix.toml").is_file())


def test_fixture_projects_exist():
    assert PROJECTS == ["lrucache", "ringbuf", "slidewin", "strkit"]


@pytest.mark.parametrize("project", PROJECTS)
def test_vendored_harness_matches_common_copy(project):
    common = (FIXTURES / "common" / "minitest.h").read_bytes()
    assert (FIXTURES / project / "tests" / "minitest.h").read_bytes() == common


@pytest.mark.needs_cc
@pytest.mark.parametrize("project", PROJECTS)
def test_clean_fixture_passes(project, tmp_path):
    work = tmp_path / project
    shutil.copytree(FIXTURES / project, work, ignore=shutil.ignore_patterns("test_runner*"))
    report = Toolbox(ProjectConfig.load(work)).test_report()
    assert report is not None, "clean fixture failed to build"
    assert report.all_passed, report.summary_line()
    assert len(report.passed) >= 7
