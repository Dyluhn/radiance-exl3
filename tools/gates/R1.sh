#!/bin/sh
# R1 -- the ABI is pinned: configure FAILS unless the radiance found has RADIANCE_ABI_VERSION 15 (and, where the
# sources are copies of one release, that release); every plugin's rad_plugin_abi_version() returns the macro.
#
#   RADIANCE_PREFIX=<install of the pinned release> tools/gates/R1.sh
#
# The wrong installs are made, not found: SHIM prefixes beside GATES_WORK that link the real install's include/,
# share/ and bin/ and carry a copy of its CMake package -- (a) unchanged: the POSITIVE CONTROL, it must configure,
# so the shim itself is not what fails; (b) its config saying ABI 14; (c) its engine binary carrying another
# release. Every plugin directory is configured ALONE against (b), and the directories that copy one release's
# sources against (c). Then the loader: each built plugin listed by rad-info (the loader compares the ABI at load),
# and no source defines rad_plugin_abi_version() with a literal.
GATE=R1
. "$(dirname "$0")/lib.sh"
why=$(need_prefix) || { part SKIP "pinned install" "$why"; finish "nothing checked"; }
W=$GATES_WORK/r1
rm -rf "$W" && mkdir -p "$W"
pkg=$(ls -d "$RADIANCE_PREFIX"/lib*/cmake/radiance | head -1)

shim() {   # shim <dir> -- a prefix that is the real install but for what the caller then edits
    mkdir -p "$1/bin" "$1/lib/cmake"
    ln -s "$RADIANCE_PREFIX/include" "$1/include"
    ln -s "$RADIANCE_PREFIX/share" "$1/share"
    for f in "$RADIANCE_PREFIX"/bin/*; do ln -s "$f" "$1/bin/"; done
    cp -r "$pkg" "$1/lib/cmake/radiance"
}
configure() {   # configure <name> <source dir> <prefix> -> exit status of cmake; log in $W/<name>.log
    log=$W/$(printf '%s' "$1" | tr -c 'A-Za-z0-9.-' '_').log
    cmake -S "$2" -B "$log.b" -DCMAKE_PREFIX_PATH="$3" -DRAD_WITH_HIP=OFF > "$log" 2>&1
}
expect_refused() {   # expect_refused <name> <dir> <prefix> <pattern the refusal must contain>
    if configure "$1" "$REPO/$2" "$3"; then part FAIL "$1" "configured against $3 (log $log)"
    elif grep -q "$4" "$log"; then part PASS "$1" "refused: $(grep -m1 -A1 'CMake Error' "$log" | tail -1 | sed 's/^ *//' | cut -c1-110)"
    else part FAIL "$1" "failed, but not with '$4': $(grep -m1 -A2 'CMake Error' "$log" | tr '\n' ' ' | cut -c1-200)"
    fi
}

# the real install, and the shim of it unchanged (positive controls)
if configure pinned "$REPO" "$RADIANCE_PREFIX"; then
    part PASS "pinned install configures" "$(grep -m1 -E '^-- radiance [0-9.]+, ABI' "$log" | sed 's/: .*//; s/^-- //')"
else part FAIL "pinned install configures" "$(grep -m1 -A2 'CMake Error' "$log" | tr '\n' ' ')"; fi
shim "$W/same"
if configure same "$REPO" "$W/same"; then part PASS "control: the unchanged shim configures" "the shim is not what fails below"
else part FAIL "control: the unchanged shim configures" "see $log -- every refusal below is void"; fi

# ABI 14: every directory that builds against radiance refuses it, alone
shim "$W/abi14"
sed -i 's/^set(RADIANCE_ABI_VERSION 15)$/set(RADIANCE_ABI_VERSION 14)/' "$W/abi14/lib/cmake/radiance/radiance-config.cmake"
grep -q '^set(RADIANCE_ABI_VERSION 14)$' "$W/abi14/lib/cmake/radiance/radiance-config.cmake" ||
    part FAIL "ABI-14 shim" "the installed radiance-config.cmake has no 'set(RADIANCE_ABI_VERSION 15)' line to edit"
expect_refused "ABI 14: top level" . "$W/abi14" "ABI 14"
for d in quantizers/exl3 kernels/exl3ref kernels/exl3r4d architectures/qwen4exp_exl3 tests/arch; do
    [ -f "$REPO/$d/CMakeLists.txt" ] || { part SKIP "ABI 14: $d alone" "no $d/CMakeLists.txt in this checkout"; continue; }
    expect_refused "ABI 14: $d alone" "$d" "$W/abi14" "ABI 14"
done

# another release, same ABI: the directories that copy one release's sources refuse it
shim "$W/rel"
rm "$W/rel/bin/radiance" && printf '0.0.1\n' > "$W/rel/bin/radiance"
expect_refused "release 0.0.1: top level" . "$W/rel" "RADIANCE_VERSION pins radiance"
expect_refused "release 0.0.1: architectures/qwen4exp_exl3 alone" architectures/qwen4exp_exl3 "$W/rel" "is not that release"

# the loader: every plugin of a host build lists (it is refused at load on an ABI mismatch)
if host_build; then
    out=$(HIP_VISIBLE_DEVICES=-1 "$RADIANCE_PREFIX/bin/rad-info" --plugins --home "$GATES_BUILD/radiance_home" 2>&1)
    n=$(printf '%s\n' "$out" | grep -cE ' (exl3ref|exl3_quant|qwen4exp_exl3) ')
    so=$(find "$GATES_BUILD/radiance_home" -path '*/test' -prune -o -name '*.so' -print | wc -l)
    if [ "$n" -eq 3 ] && [ "$so" -eq 3 ]; then part PASS "loader loads every built plugin" "$n of $so listed by rad-info"
    else part FAIL "loader loads every built plugin" "$n of $so listed: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-300)"; fi
else part FAIL "host build" "see $GATES_WORK/logs/host-build.log"; fi

# rad_plugin_abi_version() returns the macro: every definition (the architecture's comes from radiance's
# RAD_ARCH_PLUGIN macro, which returns it) says `return RAD_ABI_VERSION;`. Control: a literal is caught.
not_macro() { grep -E 'rad_plugin_abi_version\(void\) *\{' | grep -v 'return RAD_ABI_VERSION;'; }
defs=$(cd "$REPO" && git grep -nE 'rad_plugin_abi_version\(void\) *\{' -- '*.c' '*.cc' '*.cpp' '*.h' '*.hip' ':!tests/probe')
macro=$(cd "$REPO" && git grep -lE '^RAD_ARCH_PLUGIN\(' -- '*.cpp' | tr '\n' ' ')
bad=$(printf '%s\n' "$defs" | not_macro)
caught=$(echo 'uint32_t rad_plugin_abi_version(void) { return 15; }' | not_macro)
if [ -z "$caught" ]; then part FAIL "rad_plugin_abi_version returns the macro" "the check misses a literal (control)"
elif [ -n "$bad" ]; then part FAIL "rad_plugin_abi_version returns the macro" "$bad"
else part PASS "rad_plugin_abi_version returns the macro" "$(printf '%s\n' "$defs" | grep -c .) definitions return RAD_ABI_VERSION; RAD_ARCH_PLUGIN in $macro; control (a literal) caught"
fi
finish "configure refuses ABI 14 in every plugin dir and another release where sources are copied; pinned install configures"
