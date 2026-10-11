"""tests/kld/rows_compare.py on hand-built row files: a shift it must find exactly, and none where none is."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOL = Path(__file__).with_name("rows_compare.py")


def make_ref(d: Path) -> int:
    docs = [{"source": "s", "tokens": 302, "score_from": 1, "row0": 0},     # 300 rows: blocks of 256 + 44
            {"source": "t", "tokens": 102, "score_from": 1, "row0": 300}]   # 100 rows
    (d / "kld.json").write_text(json.dumps({"rows": 400, "docs": docs}))
    return 400


def write_rows(path: Path, kl: np.ndarray) -> Path:
    rows = np.zeros((kl.size, 4), dtype="<f4")
    rows[:, 0], rows[:, 3] = kl, 1.0
    rows.tofile(path)
    return path


def compare(d: Path, a: Path, b: Path) -> dict:
    out = d / "cmp.json"
    r = subprocess.run([sys.executable, str(TOOL), str(d), str(a), str(b), "--json", str(out), "--draws", "200"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(out.read_text())


def test_identical_files_report_identical_and_a_zero_interval(tmp_path):
    n = make_ref(tmp_path)
    kl = np.random.default_rng(0).exponential(0.05, n)
    rep = compare(tmp_path, write_rows(tmp_path / "a.rows", kl), write_rows(tmp_path / "b.rows", kl))
    assert rep["identical"] is True
    assert rep["stats"]["b_minus_a"]["mean"] == {"value": 0.0, "ci95": [0.0, 0.0]}


def test_a_shift_in_one_source_is_found_there_and_only_there(tmp_path):
    n = make_ref(tmp_path)
    kl = np.random.default_rng(1).exponential(0.05, n).astype(np.float32)
    shifted = kl.copy()
    shifted[300:] += np.float32(0.25)                                 # source t only
    rep = compare(tmp_path, write_rows(tmp_path / "a.rows", kl), write_rows(tmp_path / "b.rows", shifted))
    src = rep["by_source_mean_b_minus_a"]
    assert src["s"]["value"] == 0.0 and src["s"]["ci95"] == [0.0, 0.0]
    assert abs(src["t"]["value"] - 0.25) < 1e-6
    assert abs(rep["stats"]["b_minus_a"]["mean"]["value"] - 0.0625) < 1e-6   # 100 of 400 rows moved by 0.25
    # a draw's t share is smallest when s draws its 256-row block twice: 0.25 * 100 / 612 = 0.0408
    assert rep["stats"]["b_minus_a"]["mean"]["ci95"][0] >= 0.25 * 100 / 612 - 1e-6
