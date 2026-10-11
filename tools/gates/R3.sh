#!/bin/sh
# R3 -- correct at the band tops (D15): exl3r4d against the NumPy transcription through P2's engine-style harness
# (tests/ref; poisoned margins, inputs unchanged, the layout hooks' round trip) on P2's band-top cases, M = 16 and
# M = 4096, run on the card through P4's container shim (tests/gpu/test_exl3r4d.py). rad-kbench cannot check these
# ops (it zero-fills WTAB operands, kfixture.cpp:312). Until exl3r4d declares rows above M = 16 (P5), the M = 4096
# cases have no device row and this gate FAILS when run -- which is what it is for.
#   tests/gpu/run_gpu.sh <pci> <lock> <log dir> -- tools/gates/R3.sh
GATE=R3
. "$(dirname "$0")/lib.sh"
why=$(device_session) || { part SKIP "band tops on the card" "$why"; finish "nothing checked"; }
out=$GATES_WORK/logs/r3-band-tops.out
python3 - "$REPO" "$REPO/$GATES_DEVICE_BUILD" "$EXL3_CONTAINER" "$GATES_WORK/r3" > "$out" 2>&1 <<'PY'
import sys
from argparse import Namespace
from pathlib import Path
repo, build, container, work = sys.argv[1:5]
sys.path.insert(0, f"{repo}/tests/gpu")
import test_exl3r4d as D   # P4's driver: its container shim around P2's harness
R = D.R
a = Namespace(harness=f"{build}/kernels/exl3ref/exl3ref_run", plugin=f"{build}/radiance_home/kernels/exl3r4d.so",
              container=container, domain="device", probes=None, golden=None, tol=4e-3)
D.in_container(a)
R.TOL = a.tol
Path(work).mkdir(parents=True, exist_ok=True)
ok = R.suite_band_tops(a, Path(work))
r = R.RESULTS
print(f"\nband-tops: {r['pass']} passed, {r['fail']} failed, {r['skip']} skipped")
sys.exit(0 if ok and r["fail"] == 0 else 1)
PY
if clean_tally "$out"; then part PASS "band tops M=16, M=4096 on the card" "$(tally "$out")"
else part FAIL "band tops M=16, M=4096 on the card" "$(tally "$out" || echo 'no tally'): $(grep -E '^(FAIL|SKIP)' "$out" | head -3 | tr '\n' ' ' | cut -c1-240)"; fi
finish "exl3r4d at the band tops"
