#!/usr/bin/env python3
"""p6_profile_diff.py -- tools/p6_profile.sh output -> device microseconds a DECODE STEP per op, per rank, per phase
(every level the run recorded in phases.txt: C=1, C=4, C=8 by default); with several profile dirs, the arms side by
side, and B - A when there are two. A phase one dir lacks is reported missing for it.

    p6_profile_diff.py <profile dir A> [<profile dir B>] [--json OUT] [--margin 3]

radiance's --profile-ops dump (core/runtime/ctx.cpp Ctx::dump_profile at 1.3.0) is CUMULATIVE, one row per op
INSTANCE: calls, total ms, us/call, site (dev = device event pair around the kernel; host = host clock around a
host-site launch, including what it waited on: page faults, link reads), join us, op, geometry, #index, first weight.
For each phase, the first and last dump of each rank whose timestamps fall inside [phase start + margin, phase end
- margin] are differenced row by row; the step count is the commonest call delta (an op runs once a step).
--robust: per op instance, the MEDIAN over the consecutive dump intervals inside the phase (each RADIANCE_PROFILE_EVERY
steps) instead of first-to-last: a stall of seconds (both cards idle, P0-speed's finding) lands on whichever op's event
pair is open and inflates that one row in the intervals it hits; the median of the intervals drops it. The stall time
itself is reported as `TOTAL - robust TOTAL`.
Groups: `routed experts: <op>` (first weight names a routed expert table: '_exps'), `shared expert: <op>`
('_shexp'), `host-site: <op>` (site host), else the op name. Standard library only. radiance-exl3 P6.
"""
import argparse
import collections
import json
import re
import sys
from datetime import datetime
from pathlib import Path

HDR = re.compile(r"per-op timing, rank (\d+)\.")
# calls, total ms, us/call, site, join us, [wait us: radiance 1.4.0 -- how long, a call, the op's lane sat idle before
# it started], op, geometry, #index, first weight
ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+(host|dev)\s+(\S+)\s+(?:([\d.]+)\s+)?([A-Za-z_]\S*)\s+(.*?)\s*#(\d+)"
                 r"\s+(\S+)\s*$")


def stamp(line):
    """(epoch seconds, text) from a `docker logs -t` line."""
    ts, _, text = line.partition(" ")
    t = datetime.fromisoformat(ts.rstrip("Z")[:26] + "+00:00").timestamp()
    return t, text


def dumps(log_path):
    """[(rank, t, {key: row})] in log order."""
    out, cur = [], None
    for line in open(log_path, errors="replace"):
        try:
            t, text = stamp(line.rstrip("\n"))
        except ValueError:
            continue
        m = HDR.search(text)
        if m:
            cur = (int(m.group(1)), t, {})
            out.append(cur)
            continue
        if cur is None:
            continue
        if text.strip().startswith("total"):
            cur = None
            continue
        r = ROW.match(text)
        if r:
            calls, ms, _, site, _, wait, op, geom, idx, wname = r.groups()
            cur[2][(op, geom, int(idx), wname)] = {"calls": int(calls), "ms": float(ms), "site": site,
                                                    "op": op, "wname": wname,
                                                    "wait_us": float(wait) * int(calls) if wait else 0.0}
    return out


def group(row):
    if "_exps" in row["wname"]:
        return f"routed experts: {row['op']}"
    if "_shexp" in row["wname"]:
        return f"shared expert: {row['op']}"
    if row["site"] == "host":
        return f"host-site: {row['op']}"
    return row["op"]


ROBUST = False


def per_interval(inside):
    """{key: [us a step in each consecutive interval]} over a rank's dumps inside a phase."""
    out = collections.defaultdict(list)
    for (_, _, a), (_, _, b) in zip(inside, inside[1:]):
        dcalls = collections.Counter(b[k]["calls"] - a.get(k, {"calls": 0})["calls"] for k in b)
        dcalls.pop(0, None)
        if not dcalls:
            continue
        steps = dcalls.most_common(1)[0][0]
        for k, row in b.items():
            prev = a.get(k, {"ms": 0.0, "wait_us": 0.0})
            out[k].append(((row["ms"] - prev["ms"]) * 1000.0 / steps, (row["wait_us"] - prev["wait_us"]) / steps))
    return out


def phase_table(prof_dir, margin):
    """{phase: {rank: {"steps": n, "us_per_step": {group: us}, "total_us": us}}}"""
    edges = dict(line.split() for line in open(Path(prof_dir) / "phases.txt"))
    ds = dumps(Path(prof_dir) / "server.log")
    out = {}
    for c in sorted({k[:-6] for k in edges if k.endswith("_start")}, key=lambda s: int(s[1:])):
        t0, t1 = float(edges[f"{c}_start"]) + margin, float(edges[f"{c}_end"]) - margin
        out[c] = {}
        for rank in sorted({d[0] for d in ds}):
            inside = [d for d in ds if d[0] == rank and t0 <= d[1] <= t1]
            if len(inside) < 2:
                out[c][rank] = {"error": f"{len(inside)} dump(s) inside the phase window"}
                continue
            a, b = inside[0][2], inside[-1][2]
            dcalls = collections.Counter(b[k]["calls"] - a.get(k, {"calls": 0})["calls"] for k in b)
            dcalls.pop(0, None)
            steps = dcalls.most_common(1)[0][0]
            per = collections.defaultdict(float)
            wper = collections.defaultdict(float)
            if ROBUST:
                for k, iv in per_interval(inside).items():
                    iv = sorted(iv)
                    per[group(b[k])] += iv[len(iv) // 2][0]
                    wper[group(b[k])] += sorted(w for _, w in iv)[len(iv) // 2]
            else:
                for k, row in b.items():
                    prev = a.get(k, {"ms": 0.0, "wait_us": 0.0})
                    dms = row["ms"] - prev["ms"]
                    if dms > 0:
                        per[group(row)] += dms * 1000.0 / steps
                        wper[group(row)] += (row["wait_us"] - prev["wait_us"]) / steps
            out[c][rank] = {"steps": steps, "dumps": [inside[0][1], inside[-1][1]],
                            "us_per_step": dict(per), "total_us": sum(per.values()),
                            "wait_per_step": dict(wper), "wait_total_us": sum(wper.values())}
    return out


def render(tabs, names):
    lines = []
    for c in sorted({c for t in tabs for c in t}, key=lambda s: int(s[1:])):
        for rank in sorted({r for t in tabs for r in t.get(c, {})}):
            cols = [t.get(c, {}).get(rank, {}) for t in tabs]
            if any("error" in x or not x for x in cols):
                lines.append(f"\n{c.upper()} rank {rank}: " + "; ".join(x.get("error", "missing") for x in cols))
                continue
            groups = sorted({g for x in cols for g in x["us_per_step"]},
                            key=lambda g: -max(x["us_per_step"].get(g, 0) for x in cols))
            head = " | ".join(f"{n} us/step" for n in names) + (" | B - A us" if len(tabs) == 2 else "")
            head += " | " + " | ".join(f"{n} wait us/step" for n in names)
            lines.append(f"\n{c.upper()} rank {rank} (steps differenced: {', '.join(str(x['steps']) for x in cols)})")
            lines.append(f"| op group | {head} |")
            lines.append("|---|" + "---|" * (2 * len(tabs) + (len(tabs) == 2)))
            for g in groups + ["TOTAL"]:
                v = [x["total_us"] if g == "TOTAL" else x["us_per_step"].get(g, 0.0) for x in cols]
                wv = [x.get("wait_total_us", 0.0) if g == "TOTAL" else x.get("wait_per_step", {}).get(g, 0.0) for x in cols]
                if g != "TOTAL" and max(v) < 5 and max(wv) < 5:
                    continue
                d = f" | {v[1] - v[0]:+.1f}" if len(v) == 2 else ""
                lines.append(f"| {g} | " + " | ".join(f"{x:.1f}" for x in v) + d + " | " +
                             " | ".join(f"{x:.1f}" for x in wv) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--robust", action="store_true", help="median over dump intervals per op instance")
    a = ap.parse_args()
    global ROBUST
    ROBUST = a.robust
    tabs = [phase_table(d, a.margin) for d in a.dirs]
    print(render(tabs, [Path(d).name for d in a.dirs]))
    if a.json:
        json.dump({Path(d).name: t for d, t in zip(a.dirs, tabs)}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
