"""test_c_vs_ref.py -- the C oracle against its numpy twin (common/exl3_ref.py, a transcription of trellis.py).

    python3 test_c_vs_ref.py <libexl3_common.so>

W_hat bit for bit at K = 1..8 x every codebook; the canonical plane byte for byte both ways; W (f32, both
rotations) within a stated bound, whole and stacked. A negative control shows each comparison can fail.
Exit 0 pass, 1 fail, 77 SKIPPED (numpy missing).
"""
import ctypes as C
import os
import sys

try:
    import numpy as np
except ImportError:
    print("SKIPPED: numpy is not installed")
    sys.exit(77)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import exl3_ref as ref  # noqa: E402

W_REL_BOUND = 1e-5  # |W_c - W_numpy| / max|W|: two f32 roundings orders apart, measured ~1e-7

CB = {"3inst": 0, "mcg": 1, "mul1": 2}
P16 = C.POINTER(C.c_uint16)
P8 = C.POINTER(C.c_uint8)
PF = C.POINTER(C.c_float)


class View(C.Structure):
    _fields_ = [("rows", C.c_int64), ("cols", C.c_int64), ("bits", C.c_int), ("codebook", C.c_int),
                ("codes", P8), ("codes_row_bytes", C.c_int64), ("suh", P16), ("suh1", P16), ("svh", P16)]


def ptr(a, t):
    return a.ctypes.data_as(t)


failures = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        failures.append(what)


def main(lib_path):
    lib = C.CDLL(lib_path)
    lib.exl3_trellis_decode.argtypes = [P16, C.c_int64, C.c_int64, C.c_int, C.c_int, P16]
    lib.exl3_record_to_codes.argtypes = [P16, C.c_int64, C.c_int64, C.c_int, P8, C.c_int64]
    lib.exl3_codes_to_record.argtypes = [P8, C.c_int64, C.c_int64, C.c_int64, C.c_int, P16]
    lib.exl3_decode_rows.argtypes = [C.POINTER(View), C.c_int64, C.c_int64, PF]
    rng = np.random.default_rng(20261011)

    def c_what(tr, bits, cb):
        kt, nt, _ = tr.shape
        out = np.zeros((16 * kt, 16 * nt), dtype=np.uint16)
        assert lib.exl3_trellis_decode(ptr(tr, P16), kt, nt, bits, CB[cb], ptr(out, P16)) == 0
        return out

    def c_plane(tr, bits):
        kt, nt, _ = tr.shape
        rows, cols = 16 * nt, 16 * kt
        rb = (cols * bits + 7) // 8
        out = np.zeros((rows, rb), dtype=np.uint8)
        assert lib.exl3_record_to_codes(ptr(np.ascontiguousarray(tr), P16), rows, cols, bits, ptr(out, P8), rb) == 0
        return out

    def c_record(plane, rows, cols, bits):
        out = np.zeros(((cols // 16), (rows // 16), 16 * bits), dtype=np.int16)
        assert lib.exl3_codes_to_record(ptr(plane, P8), plane.shape[1], rows, cols, bits, ptr(out, P16)) == 0
        return out

    def c_weight(plane, rows, cols, bits, cb, suh, svh, suh1=None):
        out = np.zeros((rows, cols), dtype=np.float32)
        v = View(rows, cols, bits, CB[cb], ptr(plane, P8), plane.shape[1], ptr(suh, P16),
                 ptr(suh1, P16) if suh1 is not None else None, ptr(svh, P16))
        assert lib.exl3_decode_rows(C.byref(v), 0, rows, ptr(out, PF)) == 0
        return out

    # W_hat, bit for bit, every K and codebook
    bad = []
    for bits in range(1, 9):
        for cb in ref.CODEBOOKS:
            tr = ref.pack(rng.integers(0, 1 << bits, size=(3, 2, 256)), bits)
            if not np.array_equal(c_what(tr, bits, cb), ref.unpack_tiles(tr, bits, cb).view(np.uint16)):
                bad.append((bits, cb))
    check(not bad, f"W_hat C == numpy twin at K=1..8 x 3inst/mcg/mul1 (mismatching cases: {bad})")

    # the canonical plane, both directions
    bad = []
    for bits in range(1, 9):
        tr = ref.pack(rng.integers(0, 1 << bits, size=(4, 5, 256)), bits)
        plane = ref.codes_plane(tr, bits)
        if not np.array_equal(c_plane(tr, bits), plane) or not np.array_equal(c_record(plane, 80, 64, bits), tr):
            bad.append(bits)
    check(not bad, f"canonical plane C == numpy twin, record->plane and plane->record, K=1..8 (bad K: {bad})")

    # negative control: one symbol changed is seen by both comparisons
    tr = ref.pack(rng.integers(0, 16, size=(4, 5, 256)), 4)
    tr2 = tr.copy()
    tr2[2, 3, 7] ^= 0x0040
    check(not np.array_equal(c_what(tr2, 4, "mul1"), ref.unpack_tiles(tr, 4, "mul1").view(np.uint16))
          and not np.array_equal(c_plane(tr2, 4), ref.codes_plane(tr, 4)),
          "negative control: a flipped trellis bit is a mismatch in W_hat and in the plane")

    # W in f32, one projection: C [n, k] vs numpy reconstruct [k, n]
    worst = 0.0
    for bits in (2, 3, 4, 6, 8):
        for cb in ref.CODEBOOKS:
            tr, suh, svh = ref.random_tensor(256, 384, bits, rng)
            w = ref.reconstruct(tr, suh, svh, bits, cb)
            got = c_weight(ref.codes_plane(tr, bits), 384, 256, bits, cb, suh, svh)
            worst = max(worst, float(np.abs(got - w.T).max() / np.abs(w).max()))
    check(worst <= W_REL_BOUND, f"W C == numpy reconstruct within {W_REL_BOUND:g} of max|W| (worst {worst:.3g})")

    # stacked gate_up: [gate rows ; up rows], suh then suh.1
    tg, sg, vg = ref.random_tensor(256, 256, 4, rng)
    tu, su, vu = ref.random_tensor(256, 256, 4, rng)
    plane = np.concatenate([ref.codes_plane(tg, 4), ref.codes_plane(tu, 4)])
    svh = np.concatenate([vg, vu])
    want = np.concatenate([ref.reconstruct(tg, sg, vg, 4).T, ref.reconstruct(tu, su, vu, 4).T])
    got = c_weight(plane, 512, 256, 4, "mul1", sg, svh, su)
    rel = float(np.abs(got - want).max() / np.abs(want).max())
    check(rel <= W_REL_BOUND, f"stacked gate_up C == [gate; up] numpy within {W_REL_BOUND:g} (got {rel:.3g})")
    wrong = c_weight(plane, 512, 256, 4, "mul1", sg, svh, sg)
    rel = float(np.abs(wrong - want).max() / np.abs(want).max())
    check(rel > 1e-2, f"negative control: the up half read with gate's suh is far off (rel {rel:.3g})")

    print(f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
