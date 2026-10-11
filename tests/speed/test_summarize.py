"""tools/gates/speed/summarize.py on hand-built pass dirs. Fast tier, host only: `python3 -m pytest tests/speed`.

The fixtures are printed with the radiance scripts' own printf formats (scripts/cstep2.sh, fnpf.sh,
fnconc.sh at 1.3.0), so a format change upstream breaks these tests rather than the gate.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "gates" / "speed" / "summarize.py"
sys.path.insert(0, str(TOOL.parent))
import summarize as S  # noqa: E402


def cstep2_line(c, steps, tok, ms, acc, tag=""):
    return "%-4s %9d %9d %10.3f %10.2f %9.1f%s" % (c, steps, tok, tok / steps, ms, acc, tag)


def fnpf_line(label, pt, cached, secs):
    return "  %-6s %7d tok  %6d cached  %8.3f s  %8.1f tok/s   %s" % (label, pt, cached, secs, pt / secs, "")


def fnconc_line(tag, alone, beside, lw, pt):
    return "  %-10s decoders alone %6.2f s   beside a prefill %6.2f s  (%+5.1f%%)   prefill %6.2f s = %.0f tok/s" % (
        tag, alone, beside, 100 * (beside - alone) / alone, lw, pt / lw)


def make_pass(root: Path, label, verdict="valid", c8_ms=40.0, ident_tail="0", prefill_s=3.0):
    d = root / label
    (d / "steps").mkdir(parents=True)
    (d / "pass.json").write_text(json.dumps({"label": label, "verdict": verdict, "reasons": [], "digest": "sha256:abc",
                                             "flags": "--tp 2", "load_start": 1.0, "load_end": 2.0,
                                             "foreign_cores_mean": "0.4", "page_faults_at_stop": 0}))
    (d / "steps" / "cstep2.txt").write_text("\n".join([
        "%-4s %9s %9s %10s %10s %9s" % ("C", "steps", "tok", "tok/step", "ms/step", "accept%"),
        cstep2_line(1, 240, 600, 25.0, 50.0), cstep2_line(8, 150, 2400, c8_ms, 45.0)]) + "\n")
    for k, n in (("8k", 8003), ("24k", 24011)):
        (d / "steps" / f"fnpf-{k}.txt").write_text("\n".join([
            "server http://127.0.0.1:8100, target 8000 tokens, 3 reps, corpus 1 bytes",
            fnpf_line("warm", n, 0, 9.0), fnpf_line("rep 1", n, 0, prefill_s), fnpf_line("rep 2", n, 0, prefill_s + 0.1),
            fnpf_line("rep 3", n, 0, prefill_s - 0.1)]) + "\n")
    (d / "steps" / "fnconc.txt").write_text("\n".join([
        "4 decoders x 300 tokens, long prompt ~24000 tokens, server http://127.0.0.1:8100",
        fnconc_line("warm", 9.0, 30.0, 20.0, 24011), fnconc_line("pass1", 5.0, 6.0, 9.0, 24011),
        fnconc_line("pass2", 5.0, 6.5, 9.0, 24011)]) + "\n")
    for c, p50 in ((1, 80.0), (8, 600.0)):
        (d / "steps" / f"ttft-c{c}.json").write_text(json.dumps({"p50_ms": p50}))
    (d / "steps" / "ident.txt").write_text(
        "temp 0     aaaaaaaaaaaaaaaa  bbbbbbbbbbbbbbbb\ntemp 0.7   cccccccccccccccc  dddddddddddddddd\n"
        f"temp 1.0   eeeeeeeeeeeeeeee  ffffffffffffff{ident_tail}{ident_tail}\n")
    (d / "server.log").write_text("I VRAM budget, per card:\n  already held       785.55 MiB   measured\n"
                                  "    experts           17.13 GiB   63% of it\n    kv cache           9.52 GiB   35%\n"
                                  "I[0] mover: 14485 slab slots (64 shadow), 10146 pinned pool slots, 2.60 GiB static vram\n"
                                  "I[1] mover: 14533 slab slots (64 shadow), 10098 pinned pool slots, 2.55 GiB static vram\n")
    (d / "stats-ttft-c8.json").write_text(json.dumps({"draft_accept": 0.6, "experts": {
        "layer_units": [100, 100], "layer_resident": [90, 40], "routed": 1000, "routed_res": 950}}))
    return d


def test_cstep2_levels_the_script_flagged_are_not_numbers():
    text = "\n".join([cstep2_line(1, 240, 600, 25.0, 50.0), cstep2_line(2, 200, 900, 30.0, 48.0, "  <- count moved to 1"),
                      "C=4 never reached", "8       NO-STEPS"])
    got = S.parse_cstep2(text)
    assert got[1] == {"ms": 25.0, "tok_step": 2.5, "accept": 50.0}
    assert got[2] == {"bad": "count moved to 1"}
    assert got[4] == {"bad": "never reached"} and got[8] == {"bad": "NO-STEPS"}


def test_fnpf_leaves_out_the_warm_rep_and_refuses_cached_reps():
    text = "\n".join([fnpf_line("warm", 8000, 0, 8.0), fnpf_line("rep 1", 8000, 0, 2.0), fnpf_line("rep 2", 8000, 6000, 0.5),
                      fnpf_line("rep 3", 8000, 0, 4.0)])
    rate, why = S.parse_fnpf(text)
    assert rate == pytest.approx(3000.0)                 # median of 4000 and 2000; the 16000 of the cached rep is out
    assert why == ["rep 2: 6000 cached tokens"]


def test_fnpf_with_no_rep_is_no_number():
    assert S.parse_fnpf("  failed: connection refused\n") == (None, ["no rep"])


def test_fnconc_discards_the_warm_arm():
    text = "\n".join([fnconc_line("warm", 9.0, 30.0, 20.0, 24000), fnconc_line("pass1", 5.0, 6.0, 10.0, 24000),
                      fnconc_line("pass2", 5.0, 7.0, 8.0, 24000)])
    got = S.parse_fnconc(text)
    assert got["alone_s"] == 5.0 and got["beside_s"] == 6.5
    assert got["prefill_tok_s"] == pytest.approx((2400 + 3000) / 2)
    assert got["slowdown_pct"] == pytest.approx(30.0)


def test_spread_counts_valid_passes_only(tmp_path):
    passes = [S.load_pass(str(make_pass(tmp_path, "p1", c8_ms=40.0))),
              S.load_pass(str(make_pass(tmp_path, "p2", c8_ms=41.0))),
              S.load_pass(str(make_pass(tmp_path, "p3", verdict="contaminated", c8_ms=50.0)))]
    row = {r["metric"]: r for r in S.summarize(passes)["rows"]}["decode ms/step C=8"]
    assert row["values"] == [40.0, 41.0, 50.0]
    assert row["spread"] == pytest.approx(1.0) and row["spread_pct"] == pytest.approx(100 / 40.5)
    row_all = {r["metric"]: r for r in S.summarize(passes, include_all=True)["rows"]}["decode ms/step C=8"]
    assert row_all["spread"] == pytest.approx(10.0)


def test_ident_identical_only_when_every_hash_matches(tmp_path):
    same = [S.load_pass(str(make_pass(tmp_path / "a", f"p{i}"))) for i in range(3)]
    assert S.summarize(same)["ident_identical"] is True
    differ = same[:2] + [S.load_pass(str(make_pass(tmp_path / "b", "p9", ident_tail="1")))]
    assert S.summarize(differ)["ident_identical"] is False


def test_plan_report_is_read_from_the_server_log(tmp_path):
    m = S.load_pass(str(make_pass(tmp_path, "p1")))["metrics"]
    assert m["startup VRAM already held MiB"] == 785.55
    assert m["slab slots rank 0"] == 14485 and m["slab slots rank 1"] == 14533
    assert m["expert share of VRAM GiB/card"] == 17.13


def test_clock_medians_use_busy_samples_inside_the_window_only():
    rows = [{"t": 1.0, "sclk": 3000, "mclk": 1258, "busy": 99},
            {"t": 2.0, "sclk": 3100, "mclk": 1258, "busy": 99},
            {"t": 2.5, "sclk": 500, "mclk": 96, "busy": 0},        # idle sample inside: left out
            {"t": 9.0, "sclk": 100, "mclk": 96, "busy": 99}]       # busy but outside: left out
    got = S.clock_medians(rows, 0.5, 3.0)
    assert (got["sclk"], got["mclk"], got["n"]) == (3050, 1258, 2)


def test_a_dir_that_is_not_a_finished_pass_is_refused(tmp_path):
    (tmp_path / "x").mkdir()
    r = subprocess.run([sys.executable, str(TOOL), str(tmp_path / "x")], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "no pass.json" in r.stderr


def test_the_cli_prints_the_table_and_writes_json(tmp_path):
    dirs = [str(make_pass(tmp_path, f"p{i}", c8_ms=40.0 + i)) for i in range(3)]
    r = subprocess.run([sys.executable, str(TOOL), *dirs, "--json", str(tmp_path / "s.json")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "| decode ms/step C=8 | 40.00 | 41.00 | 42.00 | 41.00 | 2.000 | 4.88 |" in r.stdout
    assert json.loads((tmp_path / "s.json").read_text())["ident_identical"] is True


def test_residency_and_acceptance_are_read_from_the_last_stats_snapshot(tmp_path):
    m = S.load_pass(str(make_pass(tmp_path, "p1")))["metrics"]
    assert m["experts resident % (ttft-c8)"] == pytest.approx(65.0)
    assert m["weakest layer resident % (ttft-c8)"] == pytest.approx(40.0)
    assert m["routed-to-resident hit % (whole pass)"] == pytest.approx(95.0)
    assert m["MTP accept % (whole pass)"] == pytest.approx(60.0)


def test_idle_share_counts_samples_under_95_percent_inside_the_window():
    rows = [{"t": 1.0, "busy": 100}, {"t": 2.0, "busy": 45}, {"t": 3.0, "busy": 94}, {"t": 4.0, "busy": 99},
            {"t": 9.0, "busy": 0}]                                   # outside the window: left out
    assert S.idle_share(rows, 0.5, 4.5) == pytest.approx(50.0)
    assert S.idle_share([], 0, 1) is None


def test_foreign_names_rank_processes_by_mean_cores_in_the_window():
    hl = [{"t": 1.0, "top_foreign": [["cc1plus", 1.0], ["baloo_file", 0.5]]},
          {"t": 2.0, "top_foreign": [["baloo_file", 0.9]]},
          {"t": 9.0, "top_foreign": [["late", 5.0]]}]               # outside: left out
    assert S.foreign_names(hl, 0.5, 2.5) == [("baloo_file", 0.7), ("cc1plus", 0.5)]


def done_line(ttft, e2e):
    """A request line as radiance 1.3.0 logs it (core/server/server.cpp:732)."""
    return (f"I req cmpl-x done from 172.17.0.1:1 finish=stop prompt=57 (cache 0) gen=3000 ttft={ttft:.2f}s "
            f"180.0 tok/s {e2e:.2f}s")


def make_level_pass(root: Path, label, c8_ms, dip_at=None, c8_ttft=(0.11,) * 8):
    """A pass with cstep2 run once over C=1,2,4,8 in [100, 200] s; the server log puts the levels at
    C=1 [100, 120], C=2 [120, 140], C=4 [140, 160], C=8 [160, 200]. `dip_at` puts one 40%-busy clock
    sample that many seconds after C=8 starts."""
    d = make_pass(root, label, c8_ms=c8_ms)
    (d / "steps" / "cstep2.txt").write_text("\n".join([
        cstep2_line(1, 240, 600, 25.0, 50.0), cstep2_line(2, 240, 1200, 30.0, 50.0),
        cstep2_line(4, 240, 2400, 35.0, 50.0), cstep2_line(8, 150, 2400, c8_ms, 45.0)]) + "\n")
    (d / "windows.txt").write_text("ident 90 100\ncstep2 100 200\n")
    log = [done_line(0.05, 3.0)] * 6 + [done_line(0.07, 20.0)] + [done_line(0.06, 20.0)] * 2 + \
          [done_line(0.07, 20.0)] * 4 + [done_line(t, 40.0) for t in c8_ttft]
    (d / "server.log").write_text((d / "server.log").read_text() + "\n".join(log) + "\n")
    rows = [{"t": 100.0 + 0.5 * i, "sclk": 3000, "mclk": 1258, "busy": 100} for i in range(200)]
    if dip_at is not None:
        rows.append({"t": 160.0 + dip_at, "sclk": 2000, "mclk": 1258, "busy": 40})
    (d / "clocks-cardA.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return d


def test_level_windows_come_from_the_request_log():
    log = "\n".join([done_line(0.05, 3.0)] * 6 + [done_line(0.07, 20.0)] + [done_line(0.06, 19.0)] * 2 +
                    [done_line(0.07, 20.0)] * 4 + [done_line(0.11, 37.0)] * 8)
    lv = S.cstep2_levels(log, (100.0, 200.0))           # 96 s of levels + 4 s left: 1 s between levels
    assert [lv[c]["window"] for c in (1, 2, 4, 8)] == [(100.0, 120.0), (121.0, 140.0), (141.0, 161.0), (162.0, 199.0)]
    assert lv[2]["ttft"] == [0.06, 0.06]


def test_a_dip_inside_the_sampled_stretch_flags_the_level_and_leaves_the_others(tmp_path):
    p = S.load_pass(str(make_level_pass(tmp_path, "p1", 44.0, dip_at=6.0)))
    assert p["stalled"] == {"decode ms/step C=1": "", "decode ms/step C=2": "", "decode ms/step C=4": "",
                            "decode ms/step C=8": "stalled"}


def test_a_dip_during_the_ramp_is_not_a_stall(tmp_path):
    p = S.load_pass(str(make_level_pass(tmp_path, "p1", 40.0, dip_at=1.0)))
    assert p["stalled"]["decode ms/step C=8"] == ""


def test_requests_admitted_in_two_steps_flag_the_level(tmp_path):
    p = S.load_pass(str(make_level_pass(tmp_path, "p1", 42.0, c8_ttft=(0.06,) * 3 + (0.14,) * 5)))
    assert p["stalled"]["decode ms/step C=8"].startswith("split admission 0.06/0.06/0.06/0.14")
    assert p["stalled"]["decode ms/step C=4"] == ""


def test_the_unflagged_spread_leaves_flagged_values_out(tmp_path):
    passes = [S.load_pass(str(make_level_pass(tmp_path, "p1", 40.0))),
              S.load_pass(str(make_level_pass(tmp_path, "p2", 40.4))),
              S.load_pass(str(make_level_pass(tmp_path, "p3", 44.0, dip_at=6.0)))]
    s = S.summarize(passes)
    row = {r["metric"]: r for r in s["rows"]}["decode ms/step C=8"]
    assert row["spread"] == pytest.approx(4.0)
    assert row["unflagged"]["spread"] == pytest.approx(0.4)
    assert "| decode ms/step C=8 | 40.00 | 40.40 | 44.00* |" in S.render(s)
    assert "unflagged" not in {r["metric"]: r for r in s["rows"]}["decode ms/step C=1"]
