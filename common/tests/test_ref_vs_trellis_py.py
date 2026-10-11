"""test_ref_vs_trellis_py.py -- the numpy twin (common/exl3_ref.py) against the file it transcribes.

    EXL3_TRELLIS_PY=<exl3-moe>/exl3_moe/kernels/trellis.py python3 test_ref_vs_trellis_py.py

pack and unpack_tiles bit for bit at K = 1..8 x every codebook, reconstruct within 1e-6 of max|W|.
trellis.py needs torch and lives in another repository, so this is SKIPPED (exit 77) unless both are there;
the twin is also checked against the vllm-exl3 golden by the C tests on every run.
"""
import importlib.util
import os
import sys

path = os.environ.get("EXL3_TRELLIS_PY", "")
try:
    import numpy as np
    import torch
except ImportError as e:
    print(f"SKIPPED: {e.name} is not installed")
    sys.exit(77)
if not path or not os.path.isfile(path):
    print("SKIPPED: set EXL3_TRELLIS_PY to exl3-moe's exl3_moe/kernels/trellis.py")
    sys.exit(77)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import exl3_ref as ref  # noqa: E402

spec = importlib.util.spec_from_file_location("trellis", path)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

rng = np.random.default_rng(7)
bad = []
for bits in range(1, 9):
    for cb in ref.CODEBOOKS:
        syms = rng.integers(0, 1 << bits, size=(3, 2, 256))
        tr = ref.pack(syms, bits)
        if not np.array_equal(tr, t.pack(torch.from_numpy(syms), bits).numpy()):
            bad.append((bits, cb, "pack"))
        a = ref.unpack_tiles(tr, bits, cb).view(np.int16)
        b = t.unpack_tiles(torch.from_numpy(tr), bits, cb).view(torch.int16).numpy()
        if not np.array_equal(a, b):
            bad.append((bits, cb, "unpack"))
tr, suh, svh = ref.random_tensor(256, 384, 3, rng)
w = ref.reconstruct(tr, suh, svh, 3, "mul1")
wt = t.reconstruct(torch.from_numpy(tr), torch.from_numpy(suh), torch.from_numpy(svh), 3).numpy()
rel = float(np.abs(w - wt).max() / np.abs(wt).max())
print(f"pack/unpack mismatches: {bad}; reconstruct rel diff {rel:.3g}")
ok = not bad and rel <= 1e-6
print("ok" if ok else "FAIL")
sys.exit(0 if ok else 1)
