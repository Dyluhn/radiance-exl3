"""tools/gates/speed/run_pass.sh refuses bad input by name, with exit 2, before it touches a GPU or docker run.

Fast tier, host only: `python3 -m pytest tests/speed`. Each case stops at an argument or setup check that
comes before the quiet-host wait, the lock and the preflight; none starts a server. The positive control is
a real pass (notes: radiance-exl3 P0 speed).
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "gates" / "speed" / "run_pass.sh"
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="SKIPPED: no git on this machine")
needs_docker = pytest.mark.skipif(shutil.which("docker") is None, reason="SKIPPED: no docker on this machine")


def fake_src(root: Path, commit_ok=True) -> Path:
    """A radiance source tree with the four scripts; a git checkout at some commit (never the pin)."""
    src = root / "src"
    (src / "scripts").mkdir(parents=True)
    for s in ("ident.sh", "cstep2.sh", "fnpf.sh", "fnconc.sh"):
        (src / "scripts" / s).write_text("#!/bin/sh\nexit 0\n")
    if not commit_ok:
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
               "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run(["git", "init", "-q", str(src)], check=True, env=env)
        subprocess.run(["git", "-C", str(src), "add", "."], check=True, env=env)
        subprocess.run(["git", "-C", str(src), "commit", "-qm", "x"], check=True, env=env)
    return src


def run(args, tmp_path, **env):
    e = {**os.environ, "SPEED_IMAGE": "radiance-exl3-test/no-such-image:0", **env}
    e.setdefault("RADIANCE_SRC", str(fake_src(tmp_path)))
    return subprocess.run(["bash", str(TOOL), *args], capture_output=True, text=True, timeout=60, env=e)


def model(tmp_path) -> str:
    m = tmp_path / "m.rad"
    m.write_bytes(b"x")
    return str(m)


def test_wrong_argument_count(tmp_path):
    r = run(["only-one"], tmp_path)
    assert r.returncode == 2 and "usage" in r.stderr


def test_missing_container(tmp_path):
    r = run([str(tmp_path / "nope.rad"), "p1", str(tmp_path / "out")], tmp_path)
    assert r.returncode == 2 and "no such container" in r.stderr


def test_label_with_a_slash(tmp_path):
    r = run([model(tmp_path), "a/b", str(tmp_path / "out")], tmp_path)
    assert r.returncode == 2 and "label" in r.stderr


def test_plugin_home_that_is_not_a_directory(tmp_path):
    r = run([model(tmp_path), "p1", str(tmp_path / "out"), str(tmp_path / "nohome")], tmp_path)
    assert r.returncode == 2 and "plugin home" in r.stderr


def test_existing_pass_is_not_overwritten(tmp_path):
    (tmp_path / "out" / "p1").mkdir(parents=True)
    r = run([model(tmp_path), "p1", str(tmp_path / "out")], tmp_path)
    assert r.returncode == 2 and "refusing to overwrite" in r.stderr


@needs_docker  # the tool refuses a missing docker by name before it looks at the scripts
def test_radiance_src_without_the_scripts(tmp_path):
    (tmp_path / "empty").mkdir()
    r = run([model(tmp_path), "p1", str(tmp_path / "out")], tmp_path, RADIANCE_SRC=str(tmp_path / "empty"))
    assert r.returncode == 2 and "no scripts/ident.sh" in r.stderr


@needs_git
@needs_docker
def test_radiance_src_at_another_commit(tmp_path):
    src = fake_src(tmp_path / "g", commit_ok=False)
    r = run([model(tmp_path), "p1", str(tmp_path / "out")], tmp_path, RADIANCE_SRC=str(src))
    assert r.returncode == 2 and "not the pinned" in r.stderr


@needs_docker
def test_unknown_image(tmp_path):
    r = run([model(tmp_path), "p1", str(tmp_path / "out")], tmp_path)
    assert r.returncode == 2 and "no image radiance-exl3-test/no-such-image:0" in r.stderr
    assert not (tmp_path / "out" / "p1").exists()          # nothing was made before the refusal


def test_the_harness_scripts_are_executable():
    here = TOOL.parent
    for f in ("run_pass.sh", "summarize.py", "ttft.py", "hostload.py"):
        assert os.access(here / f, os.X_OK), f"{f} is not executable"
