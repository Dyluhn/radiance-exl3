#!/bin/sh
# duplicate_key.sh RAD_INFO PLUGIN_HOME TWIN_HOME INSTALL_HOME
#
# The engine refuses a second plugin claiming ("qwen4exp", "exl3"), by name. TWIN_HOME holds this
# plugin under a second name (twin_plugin.cpp); PLUGIN_HOME holds the real one; INSTALL_HOME is the
# installed radiance's $RADIANCE_HOME (the kernels and the in-tree architectures).
#
# Two controls first, so the verdict cannot come from a broken build:
#   1. the plugin alone loads, at its key, beside the in-tree ("qwen4exp", "") one;
#   2. the twin alone loads -- so a refusal in the case below is the duplicate, not a bad twin.
# Then the case: both on the path, the loader's refusal names the key and both plugins.
#
# EXIT: 0 PASS, 1 FAIL, 77 SKIP (no rad-info to ask)
set -u
info=$1 ours=$2 twin=$3 inst=$4
[ -x "$info" ] || { echo "SKIP: no rad-info at '$info' (an install without tools)"; exit 77; }
plugins() { RADIANCE_HOME="$1" "$info" --plugins 2>&1; }
fail=0
check() {  # name, then the test
    name=$1; shift
    if "$@"; then echo "PASS $name"; else echo "FAIL $name"; echo "$out" | sed 's/^/      /'; fail=1; fi
}
listed() { echo "$out" | grep -qE "^ *[0-9]+ +$1 +[0-9.]+ +architecture +$2\$"; }

out=$(plugins "$ours:$inst"); st=$?
check "the plugin alone loads at (qwen4exp, exl3), beside the in-tree (qwen4exp, \"\")" \
    sh -c '[ "$0" -eq 0 ]' "$st"
check "  ...listed as qwen4exp_exl3" listed qwen4exp_exl3 'qwen4exp @ exl3'
check "  ...and the in-tree one still listed" listed qwen4exp_fp8 'qwen4exp @ \(unquantised\)'

out=$(plugins "$twin:$inst"); st=$?
check "the twin alone loads (control)" sh -c '[ "$0" -eq 0 ]' "$st"
check "  ...listed as qwen4exp_exl3_twin" listed qwen4exp_exl3_twin 'qwen4exp @ exl3'

out=$(plugins "$twin:$ours:$inst"); st=$?
want="architecture id 'qwen4exp' at quantisation 'exl3' is claimed by both 'qwen4exp_exl3_twin' and 'qwen4exp_exl3' -- refusing the second"
check "both on the path: the second claimant is refused by name (rad-info exit $st)" \
    sh -c 'printf "%s\n" "$0" | grep -qF "$1"' "$out" "$want"
echo "$out" | grep -F "claimed by both" | sed 's/^/      /'

[ $fail -eq 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
exit $fail
