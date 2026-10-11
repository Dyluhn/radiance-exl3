#!/usr/bin/env python3
"""p6_predict.py -- what P6's convert will reuse, write and weigh, from its --plan-only listing and the --reuse
container's entry table, before a byte is written.

    p6_predict.py PLAN_LOG BASELINE_LISTING BASELINE_BYTES [--json OUT]

PLAN_LOG          `rad-convert ... --plan-only -v` output: one line per planned weight,
                  `<name> <encoding> <size> <unit> as is|<quantiser> <options>` (tools/rad_convert.cpp:723 at 1.3.0)
BASELINE_LISTING  `rad-info -v <baseline.rad>` output: one line per entry,
                  `<name> <encoding> <shape> <size> <unit> @<offset> checkpoint|<quantiser> <options>`
BASELINE_BYTES    the baseline file's size (its last entry's footprint runs to the end of the file)

WHY: --plan-only stops before rad-convert matches weights against --reuse (that is in the write phase,
rad_convert.cpp:829-866), so the plan alone cannot say what will be copied. This applies the same test on what
both listings show -- same name, encoding spelling, quantiser and options text -- and cannot see the layer /
expert / slot / shape half of rad-convert's test; the convert's own `--reuse: N weight(s)` line is the check
for that, and tools/p6_convert.sh compares it with this prediction.

SIZE: a reused entry weighs what it weighs in the baseline (the distance to the next entry); a new one is its
planes, each rounded to 256 B, the entry to 4 KiB -- the writer's layout as the baseline shows it (its planes
start 256-aligned, its entries 4 KiB-aligned). Plus the baseline's header and tables (its data offset), an
estimate: the new container's strings differ slightly. A new weight whose encoding this cannot size, or that
the expected rule does not make (quantiser exl3), is listed and the exit status is 1.
"""
import argparse
import json
import math
import re
import sys

PLAN = re.compile(r"^D rad_convert\.cpp:\d+\s+(\S+)\s+(\S+)\s+\S+ \S+\s+(as is|\S+(?: \S+)?)\s*$")
ENTRY = re.compile(r"^      (\S+)\s+(\S+)\s+(\d+(?:x\d+)*)\s+\S+ \S+\s+@(\d+)\s+(checkpoint|\S+(?: \S+)?)\s*$")
DATA = re.compile(r"^  data\s+(\d+)\s")
PLAN_COUNT = re.compile(r"^I plan\s+(\d+) weight\(s\)")
PLANE = re.compile(r"([\w.]+)=(\w+)([\[{])(\*|\d+)x(\*|\d+)[\]}]")
BITS = {"bf16": 16, "f16": 16, "f32": 32, "i32": 32, "i8": 8, "u8": 8, "fp8_e4m3": 8, "fp8_e5m2": 8,
        **{f"u{k}": k for k in range(1, 8)}}
NEW_QUANTISER = "exl3"


def split_rule(text: str) -> tuple[str, str]:
    """'as is' / 'checkpoint' -> ('', ''); 'rtn a=b,c=d' -> ('rtn', 'a=b,c=d'); 'cast' -> ('cast', '')."""
    if text in ("as is", "checkpoint"):
        return "", ""
    q, _, o = text.partition(" ")
    return q, o


def read_plan(lines) -> dict:
    plan, count = {}, None
    for line in lines:
        if c := PLAN_COUNT.match(line):
            count = int(c[1])
        if m := PLAN.match(line.rstrip("\n")):
            plan[m[1]] = (m[2], *split_rule(m[3]))
    if count is None or count != len(plan):
        raise SystemExit(f"p6_predict: the plan log says {count} weight(s) and lists {len(plan)} (run --plan-only -v)")
    return plan


def read_baseline(lines, size: int) -> tuple[dict, int]:
    """{name: (encoding, quantiser, options, shape, footprint)} and the data offset."""
    rows, data0 = [], None
    for line in lines:
        line = line.rstrip("\n")
        if data0 is None and (d := DATA.match(line)):
            data0 = int(d[1])
        if m := ENTRY.match(line):
            rows.append((int(m[4]), m[1], m[2], *split_rule(m[5]), [int(x) for x in m[3].split("x")]))
    if data0 is None or not rows:
        raise SystemExit("p6_predict: the baseline listing has no data section line or no entries (rad-info -v?)")
    rows.sort()
    ends = [r[0] for r in rows[1:]] + [size]
    return {n: (e, q, o, s, end - off) for (off, n, e, q, o, s), end in zip(rows, ends)}, data0


def plane_bytes(dtype: str, bracket: str, r: str, c: str, rows: int, cols: int) -> int:
    if bracket == "{":                                    # a fixed table of r x c values
        n = int(r) * int(c)
    else:                                                 # a value every [r x c] block; * = the whole extent
        n = (1 if r == "*" else math.ceil(rows / int(r))) * (1 if c == "*" else math.ceil(cols / int(c)))
    return math.ceil(n * BITS[dtype] / 8)


def footprint(encoding: str, shape: list) -> int:
    """Bytes a new entry takes: each plane rounded to 256 B, the entry to 4 KiB."""
    rows, cols = (shape[0], shape[1]) if len(shape) == 2 else (1, shape[0])
    if ":" not in encoding:
        raise ValueError(f"cannot size encoding {encoding!r}")
    planes = PLANE.findall(encoding.split(":", 1)[1])
    if not planes or any(p[1] not in BITS for p in planes):
        raise ValueError(f"cannot size encoding {encoding!r}")
    total = sum(-(-plane_bytes(d, b, r, c, rows, cols) // 256) * 256 for _, d, b, r, c in planes)
    return -(-total // 4096) * 4096


def predict(plan: dict, base: dict, data0: int) -> dict:
    reused = [n for n, p in plan.items() if n in base and base[n][:3] == p]
    new = [n for n in plan if n not in set(reused)]
    unexpected, new_bytes, by_rule = [], 0, {}
    for n in new:
        enc, q, o = plan[n]
        by_rule[f"{q} {o}".strip() or "as is"] = by_rule.get(f"{q} {o}".strip() or "as is", 0) + 1
        try:
            if q != NEW_QUANTISER or n not in base:
                raise ValueError(f"made by '{q or 'as is'}', not the {NEW_QUANTISER} rule" if q != NEW_QUANTISER
                                 else "not in the baseline, so its shape is unknown here")
            new_bytes += footprint(enc, base[n][3])
        except ValueError as e:
            unexpected.append(f"{n}: {e}")
    reused_bytes = sum(base[n][4] for n in reused)
    return {"plan_weights": len(plan), "reused": len(reused), "written": len(new), "written_by_rule": by_rule,
            "reused_bytes": reused_bytes, "written_bytes": new_bytes, "header_bytes_estimate": data0,
            "predicted_bytes": data0 + reused_bytes + new_bytes, "baseline_entries": len(base),
            "baseline_entries_not_planned": len(set(base) - set(plan)), "unexpected": unexpected}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plan_log")
    ap.add_argument("baseline_listing")
    ap.add_argument("baseline_bytes", type=int)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    with open(a.plan_log, encoding="utf-8") as f:
        plan = read_plan(f)
    with open(a.baseline_listing, encoding="utf-8") as f:
        base, data0 = read_baseline(f, a.baseline_bytes)
    r = predict(plan, base, data0)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)
    print(f"p6_predict: {r['plan_weights']} planned weight(s): {r['reused']} reused ({r['reused_bytes']} B), "
          f"{r['written']} written ({r['written_bytes']} B: {r['written_by_rule']}); predicted "
          f"{r['predicted_bytes']} B ({r['predicted_bytes'] / 1e9:.2f} GB, {r['predicted_bytes'] / 2**30:.2f} GiB); "
          f"baseline entries not in the plan: {r['baseline_entries_not_planned']}")
    for u in r["unexpected"][:20]:
        print(f"p6_predict: UNEXPECTED {u}", file=sys.stderr)
    return 1 if r["unexpected"] else 0


if __name__ == "__main__":
    sys.exit(main())
