#!/bin/sh
# R9 -- out of tree, the pattern of radiance's tests/oot_plugins.sh: against the INSTALLED radiance, every plugin
# directory of this repository is configured ALONE (find_package(radiance), no top level, nothing of the tree's
# build reused), built, and installed into a home of its own. Then, against the top-level build's home:
#   1. the same plugin files (kernels/, architectures/, quantizers/);
#   2. every kernel library declares the same vocabulary (rad-schemas, byte for byte);
#   3. the same registration (`rad-info --plugins --home`, without the path lines), no card visible.
# Control: the out-of-tree home less one plugin must compare DIFFERENT by 1 and 3 (the comparison can fail).
#
#   RADIANCE_PREFIX=<install> [RAD_WITH_HIP=OFF|ON|AUTO (default OFF)] tools/gates/R9.sh
# Run it inside the build image (RADIANCE_PREFIX=/stage/opt/radiance, RAD_WITH_HIP=AUTO) for the device library.
GATE=R9
. "$(dirname "$0")/lib.sh"
why=$(need_prefix) || { part SKIP "pinned install" "$why"; finish "nothing checked"; }
hip=${RAD_WITH_HIP:-OFF}
: "${GATES_BUILD:=$GATES_WORK/host-$hip}"
if [ ! -f "$GATES_BUILD/CMakeCache.txt" ]; then
    cmake -S "$REPO" -B "$GATES_BUILD" -DCMAKE_PREFIX_PATH="$RADIANCE_PREFIX" -DRAD_WITH_HIP="$hip" \
        > "$GATES_WORK/logs/r9-tree.log" 2>&1 || { part FAIL "top-level configure" "see $GATES_WORK/logs/r9-tree.log"; finish "no tree build"; }
fi
nice -n 19 cmake --build "$GATES_BUILD" -j "$GATES_JOBS" >> "$GATES_WORK/logs/r9-tree.log" 2>&1 ||
    { part FAIL "top-level build" "see $GATES_WORK/logs/r9-tree.log"; finish "no tree build"; }
TREE=$GATES_BUILD/radiance_home
W=$GATES_WORK/r9-$hip
H=$W/home
rm -rf "$W" && mkdir -p "$W/logs"
for d in quantizers/exl3 kernels/exl3ref kernels/exl3r4d architectures/qwen4exp_exl3; do
    [ -f "$REPO/$d/CMakeLists.txt" ] || continue
    n=$(basename "$d"); l=$W/logs/$n
    if cmake -S "$REPO/$d" -B "$W/b/$n" -DCMAKE_PREFIX_PATH="$RADIANCE_PREFIX" -DRAD_PLUGIN_INSTALL_HOME="$H" \
            -DRAD_WITH_HIP="$hip" > "$l.configure.log" 2>&1 &&
       nice -n 19 cmake --build "$W/b/$n" -j "$GATES_JOBS" > "$l.build.log" 2>&1 &&
       cmake --install "$W/b/$n" > "$l.install.log" 2>&1; then
        part PASS "$d alone" "configured, built and installed against $RADIANCE_PREFIX$(grep -h 'left out' "$l.configure.log" | sed 's/^-- / -- /' | head -1)"
    else
        part FAIL "$d alone" "$(grep -hE -m2 'CMake Error|error' "$l".*.log | tr '\n' ' ' | cut -c1-200)"
    fi
done

sets() { for k in kernels architectures quantizers; do [ -d "$1/$k" ] && (cd "$1/$k" && ls *.so 2>/dev/null | sed "s|^|$k/|"); done | sort; }
listing() { HIP_VISIBLE_DEVICES=-1 "$RADIANCE_PREFIX/bin/rad-info" --plugins --home "$1" 2>/dev/null | grep -vE '^plugins on|^     /|^     [^ ]+\.so$|^I device'; }
compare() {   # compare <home> -> prints what differs from the tree's, nothing when the same
    [ "$(sets "$TREE")" = "$(sets "$1")" ] || echo "files: tree [$(sets "$TREE" | tr '\n' ' ')] vs [$(sets "$1" | tr '\n' ' ')]"
    [ "$(listing "$TREE")" = "$(listing "$1")" ] || echo "registration differs"
}
diff=$(compare "$H")
if [ -z "$diff" ] && [ -n "$(sets "$H")" ]; then part PASS "same plugins, same registration" "$(sets "$H" | tr '\n' ' ')"
else part FAIL "same plugins, same registration" "${diff:-the out-of-tree home is empty}"; fi
for so in "$H"/kernels/*.so; do
    n=$(basename "$so")
    if "$RADIANCE_PREFIX/bin/rad-schemas" "$TREE/kernels/$n" > "$W/schemas.tree.$n" 2>&1 &&
       "$RADIANCE_PREFIX/bin/rad-schemas" "$so" > "$W/schemas.oot.$n" 2>&1 && cmp -s "$W/schemas.tree.$n" "$W/schemas.oot.$n"; then
        part PASS "$n declares the same vocabulary" "$(head -1 "$W/schemas.oot.$n")"
    else part FAIL "$n declares the same vocabulary" "diff $W/schemas.tree.$n $W/schemas.oot.$n"; fi
done
# the control: one plugin removed must be seen
cp -r "$H" "$W/less" && rm -f "$W/less/architectures/"*.so
ctl=$(compare "$W/less")
if [ -n "$ctl" ]; then part PASS "control: a home less the architecture compares different" "$(echo "$ctl" | tr '\n' ';' | cut -c1-160)"
else part FAIL "control: a home less the architecture compares different" "it compared the same: the check checks nothing"; fi
echo "  registration (out of tree):"; listing "$H" | sed 's/^/    /'
finish "each plugin dir alone against $RADIANCE_PREFIX (RAD_WITH_HIP=$hip) == the top-level build"
