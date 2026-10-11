#!/bin/sh
# R21 -- drift. (1) tests/upstream_drift.sh: the files the architecture was copied from are byte-identical at the
# pinned tag, and the tag still names the pinned commit (network; INFRA -> SKIP, never PASS); its negative control,
# the same check against v1.0.8, must FAIL. (2) `rad-schemas exl3ref.so` and `rad-schemas exl3r4d.so` declare the
# same ops, field for field (the header line, which names the library, aside); control: libref's differ. Run in
# BUILD_IMAGE, whose rad-schemas can load a HIP library. (3) docs/OPS-exl3.md against exl3ref (radiance's own
# tools/ops_doc_check.sh, ctest exl3ref_ops_doc; needs RADIANCE_SRC).
#   RADIANCE_PREFIX=<host install> [RADIANCE_SRC=<src>] tools/gates/R21.sh   (device build at GATES_DEVICE_BUILD)
GATE=R21
. "$(dirname "$0")/lib.sh"
L=$GATES_WORK/logs
sh "$REPO/tests/upstream_drift.sh" > "$L/r21-drift.out" 2>&1; rc=$?
case $rc in
    0) part PASS "upstream drift at the pin" "$(grep -c '^PASS ' "$L/r21-drift.out") file(s) byte-identical; $(head -1 "$L/r21-drift.out" | cut -c1-160)" ;;
    2) part SKIP "upstream drift at the pin" "$(tail -1 "$L/r21-drift.out" | cut -c1-200)" ;;
    *) part FAIL "upstream drift at the pin" "$(grep '^FAIL' "$L/r21-drift.out" | head -3 | tr '\n' ' ')" ;;
esac
sh "$REPO/tests/upstream_drift.sh" v1.0.8 > "$L/r21-drift-neg.out" 2>&1; rc=$?
case $rc in
    1) part PASS "control: drift against v1.0.8 fails" "$(grep -c '^FAIL ' "$L/r21-drift-neg.out") file(s) differ, as they must" ;;
    2) part SKIP "control: drift against v1.0.8 fails" "$(tail -1 "$L/r21-drift-neg.out" | cut -c1-200)" ;;
    *) part FAIL "control: drift against v1.0.8 fails" "exit $rc: the check does not fail when it must" ;;
esac

D=$REPO/$GATES_DEVICE_BUILD/radiance_home/kernels
if [ ! -f "$D/exl3r4d.so" ]; then part SKIP "rad-schemas exl3ref == exl3r4d" "no device build at $GATES_DEVICE_BUILD"
elif ! command -v docker > /dev/null || ! docker image inspect "$BUILD_IMAGE" > /dev/null 2>&1; then
    part SKIP "rad-schemas exl3ref == exl3r4d" "no docker or no image $BUILD_IMAGE"
else
    mkdir -p "$GATES_WORK/r21"
    in_image -v "$D":/k:ro -v "$GATES_WORK/r21":/out "$BUILD_IMAGE" sh -c \
        'for l in /k/exl3ref.so /k/exl3r4d.so /stage/opt/radiance/share/radiance/kernels/libref.so; do
             /stage/opt/radiance/bin/rad-schemas "$l" > /out/$(basename "$l").schemas 2>&1 || exit 1; done'
    rc=$?
    body() { tail -n +2 "$GATES_WORK/r21/$1.schemas"; }
    if [ $rc -ne 0 ] || [ ! -s "$GATES_WORK/r21/exl3r4d.so.schemas" ]; then part FAIL "rad-schemas exl3ref == exl3r4d" "rad-schemas failed: $(head -2 "$GATES_WORK/r21/"*.schemas | tr '\n' ' ')"
    elif [ "$(body exl3ref.so)" = "$(body exl3r4d.so)" ]; then
        if [ "$(body exl3ref.so)" != "$(body libref.so)" ]; then
            part PASS "rad-schemas exl3ref == exl3r4d" "$(head -1 "$GATES_WORK/r21/exl3ref.so.schemas") / $(head -1 "$GATES_WORK/r21/exl3r4d.so.schemas"): $(body exl3r4d.so | grep -cE '^  [a-z]') ops, $(body exl3r4d.so | wc -l) lines identical; control: libref's differ"
        else part FAIL "rad-schemas exl3ref == exl3r4d" "control: libref's schemas compare equal -- the comparison compares nothing"; fi
    else part FAIL "rad-schemas exl3ref == exl3r4d" "they differ: diff $GATES_WORK/r21/exl3ref.so.schemas $GATES_WORK/r21/exl3r4d.so.schemas"; fi
fi

if [ -n "${RADIANCE_SRC:-}" ] && why=$(need_prefix) && host_build; then
    ctest --test-dir "$GATES_BUILD" -R '^exl3ref_(ops_doc|schemas)$' -V > "$L/r21-docs.out" 2>&1; rc=$?
    if [ $rc -eq 0 ] && ! grep -q 'Skipped' "$L/r21-docs.out"; then part PASS "docs/OPS-exl3.md and include/exl3_abi.h == exl3ref" "ctest exl3ref_ops_doc, exl3ref_schemas passed"
    elif [ $rc -eq 0 ]; then part SKIP "docs/OPS-exl3.md == exl3ref" "$(grep -m1 SKIPPED "$L/r21-docs.out")"
    else part FAIL "docs/OPS-exl3.md and include/exl3_abi.h == exl3ref" "see $L/r21-docs.out"; fi
else part SKIP "docs/OPS-exl3.md == exl3ref" "needs RADIANCE_SRC and RADIANCE_PREFIX"; fi
finish "vendored files at the pin; one vocabulary in exl3ref, exl3r4d, the header and the docs"
