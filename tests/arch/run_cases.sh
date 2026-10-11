#!/bin/sh
# run_cases.sh BIN -- run every case of a host test binary (`BIN --list`), each in its own process.
# Prints PASS / FAIL / SKIP a case. Exit 0 when none failed and at least one passed, 1 when any
# failed, 77 (ctest's SKIP) when every case skipped -- a run that checked nothing is not a pass.
set -u
bin=$1
cases=$("$bin" --list) || { echo "run_cases: $bin --list failed" >&2; exit 1; }
pass=0 fail=0 skip=0
for c in $cases; do
    "$bin" "$c"
    case $? in
        0)  pass=$((pass + 1)) ;;
        77) skip=$((skip + 1)); echo "SKIP $c" ;;
        *)  fail=$((fail + 1)); echo "FAIL $c" ;;
    esac
done
echo "run_cases: $pass passed, $fail failed, $skip skipped"
[ $fail -eq 0 ] || exit 1
[ $pass -gt 0 ] || exit 77
exit 0
