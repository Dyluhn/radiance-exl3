#!/usr/bin/env python3
"""twin_check.py <exl3r4d.so> -- every four-class kernel (interface v6's _mod4 rows: the kernel files compiled again
with EXL3R4D_NCLS 4) costs no more than its two-class twin: no larger private segment (scratch), no more VGPR
allocation granules, no more VGPR spills. (SGPR spills land in VGPR lanes, which the VGPR count already holds; a
few more of them appear in reading-arm kernels only and cost no memory traffic.) The two builds are the same source; a difference means the four-class class
selection changed code generation -- the argument block copied to scratch, or the planes' pointers made divergent
(waterfall loops around every buffer load), each measured at 2-3x the twin's time -- which no correctness suite can
see. The four-class build's dense gate_up grid (exl3r4d_gemm_dense_kernel, P10 L6) has no two-class form: its twin is
the two-class grid gate_up kernel at the same NPW, whose work each of its workgroups does.
Static: reads the code objects' metadata with the ROCm llvm tools (the build image). Exit 0 / 1, 77 without them.
"""
import os
import re
import subprocess
import sys
import tempfile

LLVM = os.environ.get("ROCM_LLVM", "/opt/rocm/llvm/bin")
TWIN = (("12Exl3r4dGemm4", "11Exl3r4dGemm"), ("15Exl3r4dPrefill4", "14Exl3r4dPrefill"))
VGPR_GRANULE = 8   # gfx12 wave32 allocates VGPRs in blocks of 8 (an occupancy step is a whole block)
# Kernels a costlier four-class build is accepted for, by name, with why. Only a kernel the shipped configuration never
# launches belongs here.
#   down's P1c form (STAGED 4, the in-GEMM rotation; launch: gscr needs !hrot_dn) runs only at h_rot 0, and the
#   architecture issues h_rot 1 at every tp. rc-1.3: 105 VGPRs vs the two-class twin's 103 -- K1's row stores
#   (st_row) took the two-class one down a block, the four-class one stayed at its tp4-perf count (105 = 105 there);
#   not the four-class kernarg touch (an EXL3R4D_KARG_TOUCH 0 build: the same 105 / 103).
ACCEPTED = {"_Z19exl3r4d_gemm_kernelILi4ELi2ELi1ELb0ELb0ELi4ELb0EEv12Exl3r4dGemm4"}
# Four-class kernels with no two-class form (P10-TP4S L1: gate_up's K over two workgroups at one tp4 token, its own
# search and slice reduction; (a'): prep_gu_mod4 reading the router's topk_ids): checked for no scratch and no VGPR
# spills instead.
NO_TWIN = {f"_Z22exl3r4d_gemm_xs_kernelILi{n}EEv12Exl3r4dGemm4" for n in (1, 2, 4)} | \
    {"_Z22exl3r4d_prep_id_kernelILi2EEv12Exl3r4dGemm4"}


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
        m = re.match(r"_Z\d+exl3r4d_gemm_dense_kernelILi(\d)EEv11Exl3r4dGemm$", t)
        if m:   # P10 L6's dense gate_up grid: its twin is the two-class grid gate_up kernel at the same NPW
            t = f"_Z19exl3r4d_gemm_kernelILi4ELi{m.group(1)}ELi2ELb0ELb0ELi0ELb0EEv11Exl3r4dGemm"
        if t not in k and n in NO_TWIN:   # a four-class form of its own: no scratch, no spills
            f = k[n]
            bad_nt = f.get("private_segment_fixed_size", 0) > 0 or f.get("vgpr_spill_count", 0) > 0
            print(f"{'FAIL' if bad_nt else 'no twin'} {n}: scratch {f.get('private_segment_fixed_size', 0)}, spills "
                  f"{f.get('vgpr_spill_count', 0)}, vgprs {f.get('vgpr_count', 0)}")
            bad += bad_nt
            continue
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
        if why and n in ACCEPTED:
            print(f"accepted {n}: " + "; ".join(why) + " (ACCEPTED: never launched in the shipped configuration)")
        elif why:
            print(f"FAIL {n}: " + "; ".join(why))
            bad += 1
    print(f"twin_check: {len(four)} four-class kernels against their two-class twins, {bad} costlier")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
