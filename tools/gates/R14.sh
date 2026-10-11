#!/bin/sh
# R14 -- quality, by the rule pre-registered 2026-10-11 (REQUIREMENTS R14): paired per-position KL of the EXL3
# container and the baseline against the SAME independent bf16 record, tests/kld/rows_compare.py (paired
# 256-row-block bootstrap within docs, resampled within source, 2,000 draws, seed 0). B = EXL3, A = baseline.
#   mean        B - A <= 0 AND the upper 95% bound of B - A <= 0
#   each of the 7 sources, the non-agent total, p99, p99.9:   upper 95% bound of B - A <= +noise of that slice
#   top-1       B >= A - SE (SE of the paired top-1 difference = its 95% half-width / 1.96)
# The non-agent total is rows_compare.py's own mean over a copy of the record's manifest and both row files with
# the agent documents cut out (same protocol, no second bootstrap written here).
#
#   R14_REF=<record dir (kld.json)> R14_BASE_ROWS=<baseline .rows> R14_EXL3_ROWS=<EXL3 .rows> \
#   R14_NOISE=<json {slice: nats}> tools/gates/R14.sh
# R14_NOISE: the "+noise" of each slice, named "<source>", "non-agent", "p99", "p99_9". REQUIREMENTS states the
# instrument's resolution only roughly (+-0.002-0.003 overall, +-0.0006 non-agent), so the table must be
# pre-registered by the orchestrator BEFORE the EXL3 rows exist; without it those slices are NOT JUDGED (SKIP).
GATE=R14
. "$(dirname "$0")/lib.sh"
for v in R14_REF R14_BASE_ROWS R14_EXL3_ROWS; do
    eval "x=\${$v:-}"
    [ -n "$x" ] && [ -e "$x" ] || { part SKIP "inputs" "$v not given or missing ('$x'): the EXL3 container is scored at P6"; finish "nothing checked"; }
done
python3 -c 'import numpy' 2> /dev/null || { part SKIP "inputs" "python3 has no numpy"; finish "nothing checked"; }
W=$GATES_WORK/r14
mkdir -p "$W"
python3 "$REPO/tests/kld/rows_compare.py" "$R14_REF" "$R14_BASE_ROWS" "$R14_EXL3_ROWS" --json "$W/all.json" > "$W/all.txt" 2>&1 ||
    { part FAIL "rows_compare" "$(tail -2 "$W/all.txt")"; finish "no comparison"; }
# the non-agent total: the record and both row files without the agent documents
python3 - "$R14_REF" "$R14_BASE_ROWS" "$R14_EXL3_ROWS" "$W/nonagent" <<'PY' || { part FAIL "non-agent cut" "see above"; finish "no comparison"; }
import json, sys
from pathlib import Path
import numpy as np
ref, a, b, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
m = json.loads((ref / "kld.json").read_text())
keep, docs, row0 = [], [], 0
for d in m["docs"]:
    n = d["tokens"] - 1 - d["score_from"]
    if d["source"] != "agent":
        keep.append(np.arange(d["row0"], d["row0"] + n))
        docs.append({**d, "row0": row0})
        row0 += n
idx = np.concatenate(keep)
out.mkdir(parents=True, exist_ok=True)
(out / "kld.json").write_text(json.dumps({"rows": row0, "docs": docs}))
for src, name in ((a, "a.rows"), (b, "b.rows")):
    np.fromfile(src, "<f4").reshape(m["rows"], 4)[idx].tofile(out / name)
PY
python3 "$REPO/tests/kld/rows_compare.py" "$W/nonagent" "$W/nonagent/a.rows" "$W/nonagent/b.rows" --json "$W/nonagent.json" > "$W/nonagent.txt" 2>&1 ||
    { part FAIL "rows_compare (non-agent)" "$(tail -2 "$W/nonagent.txt")"; finish "no comparison"; }
python3 - "$W/all.json" "$W/nonagent.json" "${R14_NOISE:-}" > "$W/verdict.txt" <<'PY'
import json, sys
r, na = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
noise = json.load(open(sys.argv[3])) if sys.argv[3] else None
d = r["stats"]["b_minus_a"]
m = d["mean"]
print(("PASS" if m["value"] <= 0 and m["ci95"][1] <= 0 else "FAIL") + "|mean|B-A %+.6f [%+.6f, %+.6f] (A %.6f, B %.6f)" % (
    m["value"], *m["ci95"], r["stats"]["a"]["mean"]["value"], r["stats"]["b"]["mean"]["value"]))
t = d["top1"]
se = (t["ci95"][1] - t["ci95"][0]) / 2 / 1.96
print(("PASS" if t["value"] >= -se else "FAIL") + "|top-1|B-A %+.5f, SE %.5f (A %.4f, B %.4f)" % (
    t["value"], se, r["stats"]["a"]["top1"]["value"], r["stats"]["b"]["top1"]["value"]))
slices = {s: v for s, v in r["by_source_mean_b_minus_a"].items()}
slices["non-agent"] = na["stats"]["b_minus_a"]["mean"]
slices["p99"], slices["p99_9"] = d["p99"], d["p99_9"]
for s, v in slices.items():
    detail = "B-A %+.6f [%+.6f, %+.6f]" % (v["value"], *v["ci95"])
    if noise is None or s not in noise:
        print("SKIP|%s|%s; NOT JUDGED: no pre-registered noise for this slice (R14_NOISE)" % (s, detail))
    else:
        print(("PASS" if v["ci95"][1] <= noise[s] else "FAIL") + "|%s|%s <= +%g" % (s, detail, noise[s]))
PY
while IFS='|' read -r state name detail; do part "$state" "$name" "$detail"; done < "$W/verdict.txt"
finish "$(head -1 "$W/all.txt")"
