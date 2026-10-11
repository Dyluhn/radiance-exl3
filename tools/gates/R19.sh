#!/bin/sh
# R19 -- concurrency correctness: two sessions in flight are each the same as alone. radiance's own
# scripts/tiercross.sh against a server of the EXL3 container started with the tier flags its header gives
# (--checkpoint-slots 2 --prefix-cache-host-mib 512 --prefix-cache-disk-mib 16384 --prefix-cache-dir <dir>) and this
# repository's plugins (--kernels exl3r4d,exl3ref,libr4d,libavx,libref). Its own refusals (both prompts answered
# alike; an answer too short to compare) are FAILS of the instrument, reported as such.
#   R19_SERVER=<host:port of that server> RADIANCE_SRC=<src> tools/gates/R19.sh
GATE=R19
. "$(dirname "$0")/lib.sh"
[ -n "${R19_SERVER:-}" ] || { part SKIP "tiercross" "no server: R19_SERVER=<host:port> of the EXL3 container served with the tier flags (P6)"; finish "nothing checked"; }
[ -f "${RADIANCE_SRC:-}/scripts/tiercross.sh" ] || { part SKIP "tiercross" "RADIANCE_SRC has no scripts/tiercross.sh"; finish "nothing checked"; }
H=${R19_SERVER%:*} P=${R19_SERVER##*:} sh "$RADIANCE_SRC/scripts/tiercross.sh" > "$GATES_WORK/logs/r19-tiercross.out" 2>&1
rc=$?
fails=$(grep -cE '^  FAIL' "$GATES_WORK/logs/r19-tiercross.out"); oks=$(grep -cE '^  ok ' "$GATES_WORK/logs/r19-tiercross.out")
if [ $rc -eq 0 ] && [ "$fails" -eq 0 ] && [ "$oks" -gt 0 ]; then part PASS "tiercross" "$oks comparisons ok, 0 FAIL"
else part FAIL "tiercross" "rc=$rc, $oks ok, $fails FAIL: $(grep -E 'FAIL|tiercross:' "$GATES_WORK/logs/r19-tiercross.out" | head -3 | tr '\n' ' ')"; fi
finish "two sessions = each alone"
