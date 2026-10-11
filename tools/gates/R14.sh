#!/bin/sh
# R14 -- quality, by the rule pre-registered 2026-10-11 before any EXL3 rows exist (REQUIREMENTS R14): paired
# per-position KL of the EXL3 container and the baseline against the SAME independent bf16 record, through
# tests/kld/rows_compare.py (256-row blocks within docs, resampled within source, 2,000 draws, seed 0). B = EXL3,
# A = baseline.
#   overall mean                          B - A <= 0 and the UPPER 95% bound of B - A <= 0
#   each of the 7 sources, the non-agent total, p99, p99.9
#                                         the LOWER 95% bound of B - A <= 0 (not detectably worse)
#   top-1, overall and per source         B >= A - 1 SE (SE = the 95% half-width of the paired top-1 difference / 1.96)
# Per-source means come from the whole-record run (its draws). Per-source top-1 and the non-agent total come from
# rows_compare.py over a cut of the record -- its manifest and both row files restricted to those documents -- so the
# protocol is rows_compare's own, not a second bootstrap written here.
#
#   R14_REF=<record dir (kld.json)> R14_BASE_ROWS=<baseline .rows> R14_EXL3_ROWS=<EXL3 .rows> tools/gates/R14.sh
GATE=R14
. "$(dirname "$0")/lib.sh"
for v in R14_REF R14_BASE_ROWS R14_EXL3_ROWS; do
    eval "x=\${$v:-}"
    [ -n "$x" ] && [ -e "$x" ] || { part SKIP "inputs" "$v not given or missing ('$x'): the EXL3 container is scored at P6"; finish "nothing checked"; }
done
python3 -c 'import numpy' 2> /dev/null || { part SKIP "inputs" "python3 has no numpy"; finish "nothing checked"; }
W=$GATES_WORK/r14
rm -rf "$W" && mkdir -p "$W"
compare() {   # compare <name> <record dir> <a.rows> <b.rows>
    python3 "$REPO/tests/kld/rows_compare.py" "$2" "$3" "$4" --json "$W/$1.json" > "$W/$1.txt" 2>&1 ||
        { part FAIL "rows_compare ($1)" "$(tail -2 "$W/$1.txt")"; finish "no comparison"; }
}
compare all "$R14_REF" "$R14_BASE_ROWS" "$R14_EXL3_ROWS"
# the cuts: non-agent, and each source alone
python3 - "$R14_REF" "$R14_BASE_ROWS" "$R14_EXL3_ROWS" "$W" > "$W/cuts.txt" <<'PY' || { part FAIL "record cuts" "$(tail -2 "$W/cuts.txt")"; finish "no comparison"; }
import json, sys
from pathlib import Path
import numpy as np
ref, a, b, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
m = json.loads((ref / "kld.json").read_text())
ra = np.fromfile(a, "<f4").reshape(m["rows"], 4)
rb = np.fromfile(b, "<f4").reshape(m["rows"], 4)
def cut(name, keep_doc):
    keep, docs, row0 = [], [], 0
    for d in m["docs"]:
        n = d["tokens"] - 1 - d["score_from"]
        if keep_doc(d):
            keep.append(np.arange(d["row0"], d["row0"] + n))
            docs.append({**d, "row0": row0})
            row0 += n
    idx = np.concatenate(keep)
    (out / name).mkdir(parents=True, exist_ok=True)
    (out / name / "kld.json").write_text(json.dumps({"rows": row0, "docs": docs}))
    ra[idx].tofile(out / name / "a.rows")
    rb[idx].tofile(out / name / "b.rows")
    print(name)
cut("non-agent", lambda d: d["source"] != "agent")
for s in sorted({d["source"] for d in m["docs"]}):
    cut("src-" + s, lambda d, s=s: d["source"] == s)
PY
for c in $(cat "$W/cuts.txt"); do compare "$c" "$W/$c" "$W/$c/a.rows" "$W/$c/b.rows"; done
python3 - "$W" > "$W/verdict.txt" <<'PY'
import json, sys
from pathlib import Path
W = Path(sys.argv[1])
r = json.loads((W / "all.json").read_text())
d = r["stats"]["b_minus_a"]
ci = lambda v: "B-A %+.6f [%+.6f, %+.6f]" % (v["value"], *v["ci95"])  # noqa: E731
m = d["mean"]
print(("PASS" if m["value"] <= 0 and m["ci95"][1] <= 0 else "FAIL") + "|overall mean|%s; upper bound must be <= 0 (A %.6f, B %.6f)" % (
    ci(m), r["stats"]["a"]["mean"]["value"], r["stats"]["b"]["mean"]["value"]))
lower = {f"source {s}": v for s, v in sorted(r["by_source_mean_b_minus_a"].items())}
lower["non-agent total"] = json.loads((W / "non-agent.json").read_text())["stats"]["b_minus_a"]["mean"]
lower["p99"], lower["p99.9"] = d["p99"], d["p99_9"]
for name, v in lower.items():
    print(("PASS" if v["ci95"][0] <= 0 else "FAIL") + "|%s|%s; lower bound must be <= 0" % (name, ci(v)))
def top1(name, rep):
    t = rep["stats"]["b_minus_a"]["top1"]
    se = (t["ci95"][1] - t["ci95"][0]) / 2 / 1.96
    print(("PASS" if t["value"] >= -se else "FAIL") + "|top-1 %s|B-A %+.5f, 1 SE %.5f (A %.4f, B %.4f)" % (
        name, t["value"], se, rep["stats"]["a"]["top1"]["value"], rep["stats"]["b"]["top1"]["value"]))
top1("overall", r)
for p in sorted(W.glob("src-*.json")):
    top1(p.stem[4:], json.loads(p.read_text()))
PY
while IFS='|' read -r state name detail; do part "$state" "$name" "$detail"; done < "$W/verdict.txt"
finish "$(head -1 "$W/all.txt")"
