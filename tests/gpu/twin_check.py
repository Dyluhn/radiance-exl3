#!/usr/bin/env python3
"""twin_check.py <exl3r4d.so> -- every four-class kernel (interface v6's _mod4 rows: the kernel files compiled again
with EXL3R4D_NCLS 4) costs no more than its two-class twin: no larger private segment (scratch), no more VGPR
allocation granules, no more VGPR spills. (SGPR spills land in VGPR lanes, which the VGPR count already holds; a
few more of them appear in reading-arm kernels only and cost no memory traffic.) The two builds are the same source; a difference means the four-class class
selection changed code generation -- the argument block copied to scratch, or the planes' pointers made divergent
(waterfall loops around every buffer load), each measured at 2-3x the twin's time -- which no correctness suite can
see. Static: reads the code objects' metadata with the ROCm llvm tools (the build image). Exit 0 / 1, 77 without them.
"""
import os
import re
import subprocess
import sys
import tempfile

LLVM = os.environ.get("ROCM_LLVM", "/opt/rocm/llvm/bin")
TWIN = (("12Exl3r4dGemm4", "11Exl3r4dGemm"), ("15Exl3r4dPrefill4", "14Exl3r4dPrefill"))
VGPR_GRANULE = 8   # gfx12 wave32 allocates VGPRs in blocks of 8 (an occupancy step is a whole block)


def kernels(so, tmp):
    """{mangled name: {key: int}} over every offload bundle in the library's .hip_fatbin section."""
    fb = os.path.join(tmp, "fb.bin")
    subprocess.run(["objcopy", "-O", "binary", "--only-section=.hip_fatbin", so, fb], check=True)
    data = open(fb, "rb").read()
    starts = [m.start() for m in re.finditer(rb"__CLANG_OFFLOAD_BUNDLE__", data)]
    out = {}
    for i, a in enumerate(starts):
        b = starts[i + 1] if i + 1 < len(starts) else len(data)
        part, co = os.path.join(tmp, f"b{i}.bin"), os.path.join(tmp, f"b{i}.co")
        open(part, "wb").write(data[a:b])
        targets = subprocess.run([f"{LLVM}/clang-offload-bundler", "--list", "--type=o", f"--input={part}"],
                                 capture_output=True, text=True, check=True).stdout.split()
        for t in (t for t in targets if "amdgcn" in t):
            subprocess.run([f"{LLVM}/clang-offload-bundler", "--unbundle", "--type=o", f"--input={part}",
                            f"--targets={t}", f"--output={co}"], check=True)
            notes = subprocess.run([f"{LLVM}/llvm-readelf", "--notes", co], capture_output=True, text=True,
                                   check=True).stdout
            name = None
            for line in notes.splitlines():   # a kernel's keys after .name, alphabetically, until the next one
                m = re.match(r"\s*\.name:\s+(_Z\S+)", line)
                if m:
                    name = m.group(1)
                    out[name] = {}
                    continue
                m = re.match(r"\s*\.(private_segment_fixed_size|vgpr_count|vgpr_spill_count):\s+(\d+)", line)
                if m and name:
                    out[name][m.group(1)] = int(m.group(2))
    return out


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    if not os.access(f"{LLVM}/llvm-readelf", os.X_OK):
        print(f"SKIP: no {LLVM}/llvm-readelf")
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        k = kernels(sys.argv[1], tmp)
    four = [n for n in k if any(a in n for a, _ in TWIN)]
    if not four:
        print("FAIL: no four-class kernels in the library")
        return 1
    bad = 0
    for n in four:
        t = n
        for a, b in TWIN:
            t = t.replace(a, b)
        if t not in k:
            print(f"FAIL no two-class twin: {n}")
            bad += 1
            continue
        f, w = k[n], k[t]
        gran = lambda d: -(-d.get("vgpr_count", 0) // VGPR_GRANULE)  # noqa: E731
        why = [f"{key} {f.get(key, 0)} > {w.get(key, 0)}" for key in
               ("private_segment_fixed_size", "vgpr_spill_count") if f.get(key, 0) > w.get(key, 0)]
        if gran(f) > gran(w):
            why.append(f"vgpr_count {f.get('vgpr_count')} > {w.get('vgpr_count')} (a block more)")
        if why:
            print(f"FAIL {n}: " + "; ".join(why))
            bad += 1
    print(f"twin_check: {len(four)} four-class kernels against their two-class twins, {bad} costlier")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
