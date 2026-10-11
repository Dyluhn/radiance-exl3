#!/usr/bin/env python3
"""summarize.py -- one calibration session folder (run.sh output) -> markdown tables + per-run values.

    summarize.py <session dir> [--json out.json]

Every number is the median over rounds with its [min, max]; each round is a fresh process. Clocks are
the clocks.py samples inside each instrument's run windows with the card at least 50% busy.
radiance-exl3 S0.11. Apache-2.0.
"""
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict


def rounds_of(d, tool):
    files = glob.glob(os.path.join(d, f"r*-{tool}.log"))
    return sorted(files, key=lambda f: int(re.search(r"/r(\d+)-", f).group(1)))


def med(v):
    return statistics.median(v), min(v), max(v)


def fmt(v, nd=1):
    m, lo, hi = med(v)
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def stream(d):
    vals = defaultdict(list)
    for f in rounds_of(d, "stream"):
        for m in re.finditer(r"^stream mode=(\w+) pitch=(\d+) tpr=(\d+) gbps=([\d.]+)", open(f).read(), re.M):
            vals[(m[1], int(m[2]), int(m[3]))].append(float(m[4]))
    return vals


def isarate(d, tool):
    vals = defaultdict(lambda: defaultdict(list))
    row = re.compile(r"^\s+(v_\w+(?: \w+)?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+([\d.]+)x", re.M)
    for f in rounds_of(d, tool):
        for m in row.finditer(open(f).read()):
            k = (m[1].strip(), int(m[2]))
            for name, g in (("wsimd", 3), ("ms", 4), ("ginstr", 5), ("tflops", 6), ("slots", 7), ("slots0", 8)):
                vals[k][name].append(float(m[g]))
    return vals


def sweep(d, tool):
    per_size, fit = defaultdict(list), defaultdict(list)
    for f in rounds_of(d, tool):
        t = open(f).read()
        for m in re.finditer(r"^sweep bytes=(\d+) us=([\d.]+) fit=([\d.]+) resid_us=([-+\d.]+) resid_pct=([-+\d.]+)", t, re.M):
            per_size[int(m[1])].append((float(m[2]), float(m[5])))
        m = re.search(r"^sweep fixed_us=([-\d.]+) slope_gbps=([\d.]+) rms_resid_pct=([\d.]+) points=\d+ empty_us=([\d.]+)", t, re.M)
        for i, name in enumerate(("fixed_us", "slope_gbps", "rms_resid_pct", "empty_us"), 1):
            fit[name].append(float(m[i]))
    return per_size, fit


def graphrate(d):
    vals = defaultdict(list)
    for f in rounds_of(d, "graphrate"):
        for m in re.finditer(r"^\s+(.+?)\s+submit\s+([\d.]+) us\s+\(\s*([\d.]+) us a dispatch\)", open(f).read(), re.M):
            vals[m[1].strip()].append(float(m[3]))
    return vals


def clocks(d):
    path = os.path.join(d, "clocks.jsonl")
    if not os.path.exists(path):
        return {}
    samples = [json.loads(l) for l in open(path)]
    windows = [l.split() for l in open(os.path.join(d, "windows.txt"))] if os.path.exists(os.path.join(d, "windows.txt")) else []
    by_tool = defaultdict(list)
    for label, t0, t1 in windows:
        tool = label.split("-", 1)[1]
        by_tool[tool] += [s for s in samples if float(t0) <= s["t"] <= float(t1) and s.get("busy", 100) >= 50]
    out = {}
    for tool, ss in by_tool.items():
        if not ss:
            continue
        out[tool] = {"src": ss[0]["src"], "n": len(ss), "sclk": med([s["sclk"] for s in ss]), "mclk": med([s["mclk"] for s in ss]),
                     "power_w": med([s["power_w"] for s in ss]) if "power_w" in ss[0] else None,
                     "tj_max": max(s["tj_c"] for s in ss) if "tj_c" in ss[0] else None,
                     "throttle": sorted({json.dumps(s["throttle"]) for s in ss if s.get("throttle")})}
    return out


def sweep_table(d, tool, js):
    per_size, fit = sweep(d, tool)
    if not fit:
        return
    print(f"\n### Dispatch {tool} (device us a dispatch, back to back; fit >= 1 MiB, weights 1/t^2)\n")
    print("| bytes | workgroups | us | resid % |")
    print("|---|---|---|---|")
    for S in sorted(per_size):
        print(f"| {S} | {S // 8192} | {fmt([x[0] for x in per_size[S]], 3)} | {fmt([x[1] for x in per_size[S]], 2)} |")
    print(f"\nfixed {fmt(fit['fixed_us'], 3)} us, slope {fmt(fit['slope_gbps'])} GB/s, "
          f"rms residual {fmt(fit['rms_resid_pct'], 2)} %" + (f", empty kernel {fmt(fit['empty_us'], 3)} us" if 0 in per_size else ""))
    js[tool] = dict(fit)


def main():
    d = sys.argv[1]
    js = {}
    sv = stream(d)
    print("### Stream (GB/s, read unless noted; median [min, max] over rounds)\n")
    tprs = sorted({k[2] for k in sv})
    print("| pitch B | " + " | ".join(f"TPR {t}" for t in tprs) + " |")
    print("|---|" + "---|" * len(tprs))
    for p in (1024, 5120, 1280, 640):
        if any(("read", p, t) in sv for t in tprs):
            print(f"| {p}{' (plain)' if p == 1024 else ''} | " + " | ".join(fmt(sv[("read", p, t)]) if ("read", p, t) in sv else "" for t in tprs) + " |")
    if ("copy", 1024, 32) in sv:
        print(f"\ncopy (read+write counted), plain: {fmt(sv[('copy', 1024, 32)])}")
    js["stream_plain_read"] = sv.get(("read", 1024, 32), [])
    js["stream_copy"] = sv.get(("copy", 1024, 32), [])
    js["stream_5120"] = {t: sv[("read", 5120, t)] for t in tprs if ("read", 5120, t) in sv}
    for tool in ("isarate", "isarate_novopd"):
        iv = isarate(d, tool)
        if not iv:
            continue
        print(f"\n### {tool} (slots = v_mul_f32 issue slots a WMMA, same LDS; slots@0 vs the 0-LDS VALU)\n")
        print("| instruction | LDS B | waves/SIMD | slots | slots@0 | TFLOP/s | Ginstr/s |")
        print("|---|---|---|---|---|---|---|")
        for (ins, lds), v in sorted(iv.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            print(f"| {ins} | {lds} | {statistics.median(v['wsimd']):.0f} | {fmt(v['slots'], 2)} | {fmt(v['slots0'], 2)} | "
                  f"{fmt(v['tflops'])} | {fmt(v['ginstr'])} |")
            js[f"{tool}|{ins}|{lds}"] = {k: v[k] for k in ("slots", "slots0", "tflops", "ginstr")}
    for tool in ("sweep", "sweepob"):
        sweep_table(d, tool, js)
    gv = graphrate(d)
    if gv:
        print("\n### graphrate (host submit, us a dispatch, 45 a sequence)\n")
        print("| arm | us a dispatch |")
        print("|---|---|")
        for k, v in gv.items():
            print(f"| {k} | {fmt(v, 2)} |")
        js["graphrate"] = dict(gv)
    ck = clocks(d)
    if ck:
        print("\n### Clocks while busy >= 50% (median [min, max])\n")
        print("| instrument | source | samples | sclk MHz | mclk MHz | power W | Tj max C | throttle |")
        print("|---|---|---|---|---|---|---|---|")
        for tool, c in sorted(ck.items()):
            f3 = lambda x: f"{x[0]:.0f} [{x[1]:.0f}, {x[2]:.0f}]" if x else "-"
            print(f"| {tool} | {c['src']} | {c['n']} | {f3(c['sclk'])} | {f3(c['mclk'])} | {f3(c['power_w'])} | {c['tj_max']} | "
                  f"{'; '.join(c['throttle']) or 'none'} |")
        js["clocks"] = ck
    if "--json" in sys.argv:
        json.dump(js, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
