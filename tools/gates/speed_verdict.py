#!/usr/bin/env python3
"""speed_verdict.py -- one speed metric, EXL3 passes against the baseline's, by the rule pre-registered in
REQUIREMENTS R15-R17 (2026-10-11). Both arms are tools/gates/speed/summarize.py --json outputs (run_pass.sh dirs).

    speed_verdict.py BASE.json EXL3.json METRIC lower|higher [--min-passes 5] [--beside METRIC ...]

Per arm: the values of the passes summarize counted (verdict valid) whose level summarize did NOT flag (STALLED or
SPLIT). PASS: the EXL3 median is no worse than the worst baseline pass -- for a metric where lower is better
(ms/step, TTFT) median <= baseline max; where higher is better (tok/s) median >= baseline min. Inside the
baseline's [min, max] or better than every baseline pass: both pass, and the line says which (a faster container
is not a failure; notes/gates.md records that reading). SKIP when an arm has fewer than --min-passes counted passes
or fewer than 2 unflagged values. --beside prints other metrics' medians next to it (C=8: tokens/step, MTP accept).
Prints one line "PASS|FAIL|SKIP|<metric>|<detail>". Python standard library only.
"""
import argparse
import json
import statistics


def values(summary, metric):
    """(unflagged values of the counted passes, number of counted passes)."""
    counted = set(summary["counted"])
    labels = [p["label"] for p in summary["passes"]]
    row = next((r for r in summary["rows"] if r["metric"] == metric), None)
    if row is None:
        return [], len(counted)
    return [v for v, st, lab in zip(row["values"], row["stalled"], labels)
            if lab in counted and v is not None and not st], len(counted)


def verdict(base, exl3, metric, better, min_passes):
    b, nb = values(base, metric)
    e, ne = values(exl3, metric)
    if nb < min_passes or ne < min_passes:
        return "SKIP", f"{nb} baseline / {ne} EXL3 counted passes (need {min_passes} each)"
    if len(b) < 2 or len(e) < 2:
        return "SKIP", f"{len(b)} baseline / {len(e)} EXL3 unflagged values (need 2 each): rerun flagged passes"
    lo, hi, med = min(b), max(b), statistics.median(e)
    where = "inside" if lo <= med <= hi else ("better than every baseline pass" if (med < lo) == (better == "lower")
                                              else "WORSE than every baseline pass")
    ok = med <= hi if better == "lower" else med >= lo
    return ("PASS" if ok else "FAIL"), (f"EXL3 median {med:.4g} (n={len(e)}) {where}: baseline [{lo:.4g}, {hi:.4g}] "
                                        f"(median {statistics.median(b):.4g}, n={len(b)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("exl3")
    ap.add_argument("metric")
    ap.add_argument("better", choices=("lower", "higher"))
    ap.add_argument("--min-passes", type=int, default=5)
    ap.add_argument("--beside", nargs="*", default=[])
    a = ap.parse_args()
    base, exl3 = json.load(open(a.base)), json.load(open(a.exl3))
    state, detail = verdict(base, exl3, a.metric, a.better, a.min_passes)
    for m in a.beside:
        b, e = values(base, m)[0], values(exl3, m)[0]
        med = lambda v: f"{statistics.median(v):.4g}" if v else "--"  # noqa: E731
        detail += f"; {m}: baseline {med(b)}, EXL3 {med(e)}"
    print(f"{state}|{a.metric}|{detail}")


if __name__ == "__main__":
    main()
