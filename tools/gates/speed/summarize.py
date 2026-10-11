#!/usr/bin/env python3
"""summarize.py -- speed passes (run_pass.sh output dirs) -> one table, each number with its spread.

    summarize.py <pass dir>... [--all] [--json OUT.json]

One column per pass, then the median and the spread (max - min, and as % of the median) over the
passes whose verdict is `valid`; --all counts every pass that has the number. A pass that is not valid
is still shown, with its verdict, so a reader sees what was left out and why. A cstep2 level that the
script itself flagged (the running count moved, a prefill landed in the window, the level was never
reached) and a prefill rep that hit the prefix cache are not numbers: they print as `--` with the
reason in the notes. The spreads are the noise floors a later container is judged against
(radiance-exl3 REQUIREMENTS R15-R17).

Python standard library only. radiance-exl3 P0. Apache-2.0.
"""
import argparse
import json
import os
import re
import statistics

# ------------------------------------------------------------------ parsers (pure: text in, values out)


def parse_cstep2(text):
    """{C: {"ms": .., "tok_step": .., "accept": ..} or {"bad": reason}} from scripts/cstep2.sh output."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(.*)$", line)
        if m:
            c, tag = int(m.group(1)), m.group(7).strip()
            out[c] = {"bad": tag.lstrip("<- ")} if tag else \
                {"ms": float(m.group(5)), "tok_step": float(m.group(4)), "accept": float(m.group(6))}
            continue
        m = re.match(r"^C=(\d+) (busy|never reached)", line) or re.match(r"^(\d+)\s+(NO-STEPS)", line)
        if m:
            out[int(m.group(1))] = {"bad": m.group(2)}
    return out


def parse_fnpf(text):
    """(median tok/s over the reps, [reasons]) from scripts/fnpf.sh output; the warm rep is left out."""
    rates, why = [], []
    for m in re.finditer(r"^\s+(warm|rep \d+)\s+(\d+) tok\s+(\d+) cached\s+([\d.]+) s\s+([\d.]+) tok/s", text, re.M):
        if m.group(1) == "warm":
            continue
        if int(m.group(3)):
            why.append(f"{m.group(1)}: {m.group(3)} cached tokens")
            continue
        rates.append(float(m.group(5)))
    if not rates:
        why.append("no rep")
    return (statistics.median(rates) if rates else None), why


def parse_fnconc(text):
    """Means over the measured arms (the warm arm is discarded, as the script says) from fnconc.sh output."""
    arms = []
    pat = (r"^\s+(\S+)\s+decoders alone\s+([\d.]+) s\s+beside a prefill\s+([\d.]+) s\s+\(\s*([+-][\d.]+)%\)"
           r"\s+prefill\s+([\d.]+) s = (\d+) tok/s")
    for m in re.finditer(pat, text, re.M):
        if m.group(1) != "warm":
            arms.append([float(m.group(i)) for i in (2, 3, 4, 6)])
    if not arms:
        return None
    mean = [statistics.mean(a[i] for a in arms) for i in range(4)]
    return {"alone_s": mean[0], "beside_s": mean[1], "slowdown_pct": mean[2], "prefill_tok_s": mean[3]}


def parse_ident(text):
    """The six hashes, in order (temperature, question) from scripts/ident.sh output."""
    return [h for line in text.splitlines() if line.startswith("temp ") for h in line.split()[2:4]]


def parse_log(text):
    """The startup plan report: VRAM already held, expert/KV split, slab and pool slots per rank."""
    gib = lambda pat: [float(x) for x in re.findall(pat, text, re.M)]
    slots = {int(r): (int(a), int(b)) for r, a, b in
             re.findall(r"I\[(\d+)\] mover: (\d+) slab slots \(\d+ shadow\), (\d+) pinned pool slots", text)}
    return {"held_mib": gib(r"already held\s+([\d.]+) MiB"), "experts_gib": gib(r"^\s+experts\s+([\d.]+) GiB"),
            "kv_gib": gib(r"^\s+kv cache\s+([\d.]+) GiB"), "slab_slots": slots}


def residency(stats):
    """Expert residency from a /stats snapshot: resident fraction overall, its weakest layer, hit rate."""
    e = stats.get("experts") or {}
    units, res = e.get("layer_units") or [], e.get("layer_resident") or []
    if not units:
        return None
    frac = [r / u for r, u in zip(res, units) if u]
    return {"resident_frac": sum(res) / sum(units), "weakest_layer_frac": min(frac),
            "hit_rate": (e["routed_res"] / e["routed"]) if e.get("routed") else None,
            "layer_resident": res}


def clock_medians(rows, t0, t1):
    """Median sclk/mclk/power of the samples inside [t0, t1] with the card at least 50% busy."""
    r = [x for x in rows if t0 <= x["t"] <= t1 and (x.get("busy") or 0) >= 50]
    if not r:
        return None
    med = lambda k: statistics.median(x[k] for x in r if x.get(k) is not None)
    return {"sclk": med("sclk"), "mclk": med("mclk"), "power_w": med("power_w") if "power_w" in r[0] else None,
            "tj_max": max((x.get("tj_c") or 0) for x in r), "n": len(r)}


# ------------------------------------------------------------------ one pass dir -> {metric: value}


def read(path, default=""):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return default


def load_pass(d):
    p = json.loads(read(os.path.join(d, "pass.json"), "{}") or "{}")
    if not p:
        raise SystemExit(f"summarize: {d} has no pass.json -- not a finished run_pass.sh pass")
    m, notes = {}, []
    steps = os.path.join(d, "steps")
    for c, v in sorted(parse_cstep2(read(os.path.join(steps, "cstep2.txt"))).items()):
        if "bad" in v:
            notes.append(f"cstep2 C={c}: {v['bad']}")
            continue
        m[f"decode ms/step C={c}"] = v["ms"]
        m[f"tokens/step C={c}"] = v["tok_step"]
        m[f"MTP accept % C={c}"] = v["accept"]
    for k in ("8k", "24k"):
        rate, why = parse_fnpf(read(os.path.join(steps, f"fnpf-{k}.txt")))
        notes += [f"fnpf-{k} {w}" for w in why]
        if rate is not None:
            m[f"prefill tok/s {k.upper()}"] = rate
    fc = parse_fnconc(read(os.path.join(steps, "fnconc.txt")))
    if fc:
        m["fnconc prefill tok/s beside 4 decoders"] = fc["prefill_tok_s"]
        m["fnconc decoders' slowdown %"] = fc["slowdown_pct"]
        m["fnconc decoders alone s"] = fc["alone_s"]
    for c in (1, 8):
        t = json.loads(read(os.path.join(steps, f"ttft-c{c}.json"), "{}") or "{}")
        if t:
            m[f"TTFT p50 ms C={c}"] = t["p50_ms"]
    log = parse_log(read(os.path.join(d, "server.log")))
    if log["held_mib"]:
        m["startup VRAM already held MiB"] = max(log["held_mib"])
    if log["experts_gib"]:
        m["expert share of VRAM GiB/card"] = log["experts_gib"][0]
    for r, (slab, pool) in sorted(log["slab_slots"].items()):
        m[f"slab slots rank {r}"] = slab
    for name in ("ready", "ttft-c8"):
        st = json.loads(read(os.path.join(d, f"stats-{name}.json"), "{}") or "{}")
        res = residency(st)
        if res:
            m[f"experts resident % ({name})"] = 100 * res["resident_frac"]
            m[f"weakest layer resident % ({name})"] = 100 * res["weakest_layer_frac"]
            if name != "ready" and res["hit_rate"] is not None:
                m["routed-to-resident hit % (whole pass)"] = 100 * res["hit_rate"]
    windows = {}
    for line in read(os.path.join(d, "windows.txt")).splitlines():
        n, t0, t1 = line.split()
        windows[n] = (float(t0), float(t1))
    for f in sorted(os.listdir(d)):
        mm = re.match(r"clocks-(.+)\.jsonl$", f)
        if not mm:
            continue
        rows = [json.loads(x) for x in read(os.path.join(d, f)).splitlines() if x.strip()]
        for step in ("cstep2", "fnpf-24k"):
            if step in windows:
                c = clock_medians(rows, *windows[step])
                if c:
                    m[f"{mm.group(1)} sclk MHz ({step})"] = c["sclk"]
                    m[f"{mm.group(1)} mclk MHz ({step})"] = c["mclk"]
    m["load1 start"], m["load1 end"] = p.get("load_start"), p.get("load_end")
    try:
        m["non-server CPU cores (mean)"] = float(p.get("foreign_cores_mean"))
    except (TypeError, ValueError):
        pass
    return {"dir": d, "label": p.get("label", os.path.basename(d)), "verdict": p.get("verdict"),
            "reasons": p.get("reasons", []), "digest": p.get("digest"), "flags": p.get("flags"),
            "ident": parse_ident(read(os.path.join(steps, "ident.txt"))), "metrics": m, "notes": notes,
            "page_faults_at_stop": p.get("page_faults_at_stop")}


# ------------------------------------------------------------------ the table


def spread(values):
    """(median, max - min, % of the median) of the values, or None for fewer than two."""
    v = [x for x in values if x is not None]
    if len(v) < 2:
        return None
    med = statistics.median(v)
    return med, max(v) - min(v), (100 * (max(v) - min(v)) / med if med else float("nan"))


def summarize(passes, include_all=False):
    names = []
    for p in passes:
        names += [k for k in p["metrics"] if k not in names]
    counted = [p for p in passes if include_all or p["verdict"] == "valid"]
    rows = []
    for k in names:
        s = spread([p["metrics"].get(k) for p in counted])
        rows.append({"metric": k, "values": [p["metrics"].get(k) for p in passes],
                     "median": s[0] if s else None, "spread": s[1] if s else None, "spread_pct": s[2] if s else None})
    idents = {tuple(p["ident"]) for p in counted if p["ident"]}
    return {"passes": [{k: p[k] for k in ("label", "verdict", "reasons", "digest", "flags", "notes", "ident",
                                           "page_faults_at_stop")} for p in passes],
            "counted": [p["label"] for p in counted], "rows": rows,
            "ident_identical": (len(idents) == 1) if idents else None}


def fmt(x):
    if x is None:
        return "--"
    return f"{x:.0f}" if abs(x) >= 1000 else f"{x:.2f}" if abs(x) >= 10 else f"{x:.3f}"


def render(s):
    out = []
    labels = [p["label"] for p in s["passes"]]
    out.append("| metric | " + " | ".join(labels) + " | median | spread | spread % |")
    out.append("|---" * (len(labels) + 4) + "|")
    for r in s["rows"]:
        pct = "--" if r["spread_pct"] is None else f"{r['spread_pct']:.2f}"
        out.append(f"| {r['metric']} | " + " | ".join(fmt(v) for v in r["values"]) +
                   f" | {fmt(r['median'])} | {fmt(r['spread'])} | {pct} |")
    out.append("")
    out.append(f"median and spread over: {', '.join(s['counted']) or 'no valid pass'}")
    for p in s["passes"]:
        why = f" ({'; '.join(p['reasons'])})" if p["reasons"] else ""
        out.append(f"- {p['label']}: {p['verdict']}{why}; ident {' '.join(h[:8] for h in p['ident']) or '--'}; "
                   f"page faults at stop {p['page_faults_at_stop']}" + "".join(f"; {n}" for n in p["notes"]))
    out.append(f"ident identical across the counted passes: {s['ident_identical']}")
    digests = {p["digest"] for p in s["passes"]}
    flags = {p["flags"] for p in s["passes"]}
    out.append(f"image digest(s): {', '.join(sorted(str(d) for d in digests))}")
    out.append("flags: " + (" || ".join(sorted(str(f) for f in flags)) +
                            ("   <-- PASSES DIFFER IN FLAGS" if len(flags) > 1 else "")))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("passes", nargs="+")
    ap.add_argument("--all", action="store_true", help="count passes whatever their verdict")
    ap.add_argument("--json")
    a = ap.parse_args()
    s = summarize([load_pass(d) for d in a.passes], a.all)
    print(render(s))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(s, f, indent=1)


if __name__ == "__main__":
    main()
