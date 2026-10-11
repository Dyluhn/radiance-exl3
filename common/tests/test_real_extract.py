"""test_real_extract.py -- the C oracle on REAL turboderp tensors, not random symbols.

    EXL3_REAL_EXTRACT=<a .safetensors of exllamav3 tensors> python3 test_real_extract.py <libexl3_common.so>

For every `<name>.trellis` in the file (with its suh, svh and marker): W_hat from the C decoder == the numpy
twin bit for bit; the canonical plane round-trips to the same trellis bytes; W (both rotations) agrees with the
twin within 1e-5 of max|W|. The extract used during P1 is turboderp/Qwen3.8-Flash-Next-exl3 @ 55a732e layers
15-16 (6-bit shared expert, attention, GDN). SKIPPED (exit 77) without EXL3_REAL_EXTRACT or numpy.
"""
import json
import os
import struct
import sys

try:
    import numpy as np
except ImportError:
    print("SKIPPED: numpy is not installed")
    sys.exit(77)
path = os.environ.get("EXL3_REAL_EXTRACT", "")
if not path or not os.path.isfile(path):
    print("SKIPPED: set EXL3_REAL_EXTRACT to a safetensors file of exllamav3 tensors")
    sys.exit(77)

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, ".."))
import exl3_ref as ref  # noqa: E402
import test_c_vs_ref as cv  # noqa: E402  (the ctypes bindings)

DT = {"I16": np.int16, "F16": np.float16, "I32": np.int32}
MARK = {"mul1": 0x83DCD12D, "mcg": 0xCBAC1FED}


def tensors(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        out = {}
        for name, h in head.items():
            if name == "__metadata__":
                continue
            o = h["data_offsets"]
            f.seek(8 + n + o[0])
            out[name] = np.frombuffer(f.read(o[1] - o[0]), DT[h["dtype"]]).reshape(h["shape"])
        return out


def main(lib_path):
    import ctypes as C
    lib = C.CDLL(lib_path)
    lib.exl3_trellis_decode.argtypes = [cv.P16, C.c_int64, C.c_int64, C.c_int, C.c_int, cv.P16]
    lib.exl3_record_to_codes.argtypes = [cv.P16, C.c_int64, C.c_int64, C.c_int, cv.P8, C.c_int64]
    lib.exl3_codes_to_record.argtypes = [cv.P8, C.c_int64, C.c_int64, C.c_int64, C.c_int, cv.P16]
    lib.exl3_decode_rows.argtypes = [C.POINTER(cv.View), C.c_int64, C.c_int64, cv.PF]
    t = tensors(path)
    bad, worst, n = [], 0.0, 0
    for name in sorted(k[: -len(".trellis")] for k in t if k.endswith(".trellis")):
        tr = np.ascontiguousarray(t[name + ".trellis"])
        bits = tr.shape[-1] // 16
        cb = "mul1" if name + ".mul1" in t else "mcg" if name + ".mcg" in t else "3inst"
        if cb != "3inst" and int(t[f"{name}.{cb}"].view(np.uint32)) != MARK[cb]:
            bad.append((name, "marker"))
            continue
        kt, nt, _ = tr.shape
        what = np.zeros((16 * kt, 16 * nt), np.uint16)
        lib.exl3_trellis_decode(cv.ptr(tr, cv.P16), kt, nt, bits, cv.CB[cb], cv.ptr(what, cv.P16))
        if not np.array_equal(what, ref.unpack_tiles(tr, bits, cb).view(np.uint16)):
            bad.append((name, "W_hat"))
        rows, cols = 16 * nt, 16 * kt
        rb = (cols * bits + 7) // 8
        plane = np.zeros((rows, rb), np.uint8)
        back = np.zeros_like(tr)
        lib.exl3_record_to_codes(cv.ptr(tr, cv.P16), rows, cols, bits, cv.ptr(plane, cv.P8), rb)
        lib.exl3_codes_to_record(cv.ptr(plane, cv.P8), rb, rows, cols, bits, cv.ptr(back, cv.P16))
        if not np.array_equal(back, tr) or not np.array_equal(plane, ref.codes_plane(tr, bits)):
            bad.append((name, "plane"))
        if rows * cols <= 4 * 2560 * 2560:   # W for the smaller ones (the big q_proj is W_hat-checked only)
            suh, svh = np.ascontiguousarray(t[name + ".suh"]), np.ascontiguousarray(t[name + ".svh"])
            out = np.zeros((rows, cols), np.float32)
            v = cv.View(rows, cols, bits, cv.CB[cb], cv.ptr(plane, cv.P8), rb, cv.ptr(suh, cv.P16), None,
                        cv.ptr(svh, cv.P16))
            assert lib.exl3_decode_rows(C.byref(v), 0, rows, cv.ptr(out, cv.PF)) == 0
            w = ref.reconstruct(tr, suh, svh, bits, cb).T
            worst = max(worst, float(np.abs(out - w).max() / np.abs(w).max()))
        n += 1
        print(f"  {name}: K={bits} {cb} trellis {list(tr.shape)}")
    print(f"{n} real projection(s); mismatches {bad}; W worst {worst:.3g} of max|W| (bound 1e-5)")
    ok = n > 0 and not bad and worst <= 1e-5
    print("ok" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
