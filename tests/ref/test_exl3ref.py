#!/usr/bin/env python3
"""test_exl3ref.py -- kernels/exl3ref against an independent NumPy transcription (exl3_numpy.py).

    test_exl3ref.py --harness exl3ref_run --plugin exl3ref.so --work DIR [--golden exl3_golden.npz]
                    [--suite cases|band-tops]

Each case is written to DIR as files, run through the plugin by `exl3ref_run` (dlopen, the row table, opd_shape,
launch, poisoned margins on every buffer) and compared with the transcription:

- down to the BIT (bf16), gate_up to one f16 ulp (libm's expf against the correctly rounded one, the only
  arithmetic the two do not share) -- with the exact fraction printed;
- every row past expert_offset[n_expert] or with sorted_tok < 0, and every column past a narrower class's width,
  exactly zero (the output buffer is prefilled with NaN, so a row the op did not write cannot pass);
- no write outside an operand, no write to an input, the shape hook describing what the case built.

Controls, read before any verdict: the comparison must FAIL against a transcription with a deliberate error (the
gate's and the up half's input scales swapped; the parity classes swapped); one flipped trellis symbol must move
the flipped expert's rows and only those; the table and the stacked weight form, a row pitch, garbage in the
sorted rows past the live count and extra table entries must move nothing.

On P1's fixture (tools/exl3_fixture.py, generated into DIR): its layer-0 experts read from the EXL3 source,
compared with the transcription AND with the dense bf16 weights the fixture's skeleton holds (decoded by P1's own
reference), which checks the op computes x @ W and not merely what this transcription thinks.

Exit 0 all passed, 1 any failure, 77 nothing could run (no numpy). A check that cannot run prints SKIP, by name.
"""
import argparse
import copy
import json
import struct
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("SKIPPED: numpy is not installed")
    sys.exit(77)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exl3_numpy as X  # noqa: E402

RAD = {"OK": 0, "SHAPE": -1, "DTYPE": -2, "ALIGN": -3, "STRIDE": -4, "SCHEMA": -9, "INVAL": -12}
DT = {"f16": 2, "bf16": 3, "i32": 11, "u1": 22, "u2": 20, "u3": 24, "u4": 19, "u5": 26, "u6": 28, "u7": 30,
      "u8": 9}
# Operand positions (include/exl3_abi.h exl3_positions, v5): planes[class] = [codes, suh, suh1, svh, dn_suh] and for
# gate_up_down also [dn_codes, dn_svh] (down's weight).
GU, DN, PG, GD = "moe_exl3_gate_up", "moe_exl3_down", "moe_exl3_prep_gu", "moe_exl3_gate_up_down"
# The four-class forms (interface v6): experts by number mod 4, classes 1..3's tables after the routing.
GU4, DN4, PG4 = GU + "_mod4", DN + "_mod4", PG + "_mod4"
GD4 = GD + "_mod4"   # interface v7: the four-class gate_up_down
BASE = {GU4: GU, DN4: DN, PG4: PG, GD4: GD}
MOD4 = {GU: GU4, DN: DN4, PG: PG4, GD: GD4}
POS = {GU: dict(x=0, sorted=6, offset=7, y=13, n=14, planes={0: [1, 2, 3, 4, 5], 1: [8, 9, 10, 11, 12]}),
       DN: dict(x=0, sorted=4, offset=5, y=9, n=10, planes={0: [1, 2, None, 3, None], 1: [6, 7, None, 8, None]}),
       PG: dict(x=0, sorted=3, offset=4, y=7, n=8, planes={0: [None, 1, 2, None, None], 1: [None, 5, 6, None, None]}),
       GD: dict(x=0, sorted=8, offset=9, y=17, n=18, planes={0: [1, 2, 3, 4, 5, 6, 7], 1: [10, 11, 12, 13, 14, 15, 16]}),
       GU4: dict(x=0, sorted=6, offset=7, y=23, n=24, planes={0: [1, 2, 3, 4, 5], 1: [8, 9, 10, 11, 12],
                                                              2: [13, 14, 15, 16, 17], 3: [18, 19, 20, 21, 22]}),
       DN4: dict(x=0, sorted=4, offset=5, y=15, n=16, planes={0: [1, 2, None, 3, None], 1: [6, 7, None, 8, None],
                                                              2: [9, 10, None, 11, None], 3: [12, 13, None, 14, None]}),
       PG4: dict(x=0, sorted=3, offset=4, y=11, n=13, ids=12, planes={0: [None, 1, 2, None, None], 1: [None, 5, 6, None, None],
                                                              2: [None, 7, 8, None, None], 3: [None, 9, 10, None, None]}),
       GD4: dict(x=0, sorted=8, offset=9, y=31, n=32, planes={0: [1, 2, 3, 4, 5, 6, 7], 1: [10, 11, 12, 13, 14, 15, 16],
                                                              2: [17, 18, 19, 20, 21, 22, 23],
                                                              3: [24, 25, 26, 27, 28, 29, 30]})}


def base(op):
    """The two-class op a form is of (what it computes)."""
    return BASE.get(op, op)
PREP_KEYS = ("M", "K", "n_expert", "top_k", "dtype", "K_odd", "K_1", "K_2", "K_3")
TOL = None  # set from --tol in device mode
ABSENT_FLAG, IDX_CU_FLAG = 1, 4

RESULTS = {"pass": 0, "fail": 0, "skip": 0}


def verdict(ok, name, detail=""):
    RESULTS["pass" if ok else "fail"] += 1
    print(f"{'PASS' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}", flush=True)
    return ok


def skip(name, why):
    RESULTS["skip"] += 1
    print(f"SKIP {name}  {why}", flush=True)


# ----------------------------------------------------------------------------------------------- a case
class Case:
    """One launch: parameters, operands as arrays, the transcription's answer, and what to check."""

    def __init__(self, name, op, rng, *, M, top_k, n_expert, rows, cols, rows_odd=None, cols_odd=None, two=True,
                 bits=4, codebook="mul1", drop=0.0, n_protect=0, a_dtype="f16", experts=None, x=None, h_rot=0,
                 a_rot=0, classes=None):
        """For prep_gu, `rows` is still the stacked experts' 2 * width (they carry suh.1); the op reads only `cols`.
        `classes`: four (rows, cols), one a class by number mod 4 -- the _mod4 form of `op` (`rows`, `cols` unused)."""
        if classes:
            op = MOD4.get(op, op)
        self.name, self.op, self.gate_up, self.prep = name, op, base(op) == GU, base(op) == PG
        self.M, self.top_k, self.n_expert, self.bits, self.codebook = M, top_k, n_expert, bits, codebook
        self.n_class = 4 if classes else 2 if two else 1
        self.geom = list(classes) if classes else [(rows, cols), (rows_odd or rows, cols_odd or cols)]
        self.two = two
        self.T = M * top_k
        self.x_cols = max(c for _, c in self.geom)
        self.y_cols = 2 * self.x_cols if self.prep else max(r for r, _ in self.geom) // (2 if self.gate_up else 1)
        self.experts = experts or [X.random_expert(rng, *self.geom[e % self.n_class], bits, codebook,
                                                   self.gate_up or self.prep) for e in range(n_expert)]

        # The router: top_k distinct experts a token over the regular and the protected ones; a dropped slot is a
        # negative id or one past the range the sort owns. Protected experts sort last and are not this op's.
        n_total = n_expert + n_protect
        assert top_k <= n_total, f"{name}: top_k {top_k} distinct experts out of {n_total}"
        ids = np.stack([rng.permutation(n_total)[:top_k] for _ in range(M)]).astype(np.int32)
        if drop:
            hit = rng.random(ids.shape) < drop
            ids[hit] = np.where(rng.random(hit.sum()) < 0.5, -1, n_total + 3)
        self.ids = ids
        # prep_gu_mod4's optional topk_ids (P10-TP4S): "real" the router's own (above), "absent", or "garbage" (a hint
        # the plugin must not trust: the output may not move)
        self.ids_form = "real"
        self.sorted, off_full, _ = X.moe_scatter(ids, n_total, protect=tuple(range(n_expert, n_total)))
        self.off = off_full[: n_expert + 1].copy()
        self.a_dtype = a_dtype
        if x is not None:
            self.x = x
        elif self.gate_up or self.prep:
            self.x = rng.standard_normal((M, self.x_cols)).astype(np.float32)
        else:
            self.x = (rng.standard_normal((self.T, self.x_cols)) * 0.5).astype(np.float32)
        self.x = self.narrow_x(self.x)
        self.params = {"M": M, "N": self.geom[0][0], "K": self.geom[0][1], "n_expert": n_expert, "top_k": top_k,
                       "dtype": f"exl3k{bits}", "codebook": codebook}
        if classes:
            for q in (1, 2, 3):
                self.params.update({f"N_{q}": self.geom[q][0], f"K_{q}": self.geom[q][1]})
        elif two:
            self.params.update({"N_odd": rows_odd or rows, "K_odd": cols_odd or cols})
        self.params["h_rot"] = h_rot
        if self.gate_up:
            self.params["a_rot"] = a_rot
        if self.prep:  # prep_gu's own parameters: no N, no codebook, no flags
            self.params = {k: v for k, v in self.params.items() if k in PREP_KEYS}
        # gate_up's dn_suh tables: present with h_rot = 1 (None = that rule); `dn_nan` fills them with NaN, the
        # neutral control that proves h_rot = 0 does not read them.
        self.dn_present, self.dn_nan = None, False
        self.form, self.pitch, self.extra_entries, self.scale_rank = "table", None, 0, 3

    def narrow_x(self, x):
        if self.a_dtype == "bf16":
            return X.bf16_bits_to_f32(X.to_bf16_bits(x))
        return X.to_f16(x).astype(np.float32)

    def tables(self, experts=None):
        ex = experts or self.experts
        return [ex[c::self.n_class] for c in range(self.n_class)]

    def class_name(self, base_name, c):
        """A table's operand name: the class's suffix -- `_odd` for class 1 of the two-class ops, `_q` for class q of a
        _mod4 op."""
        return base_name + ("" if c == 0 else f"_{c}" if self.n_class == 4 else "_odd")

    @property
    def h_rot(self):
        return self.params.get("h_rot", 0)

    @property
    def a_rot(self):
        return self.params.get("a_rot", 0)

    def expected(self, swap_suh=False, experts=None, h_rot=None):
        ex = experts or self.experts
        hr = self.h_rot if h_rot is None else h_rot
        if self.prep:
            return X.prep_gu(self.x, ex, self.sorted, self.off, self.n_expert, self.top_k, self.x_cols,
                             swap_suh=swap_suh)
        if self.gate_up:
            return X.gate_up(self.x, ex, self.sorted, self.off, self.n_expert, self.top_k, self.y_cols,
                             swap_suh=swap_suh, h_rot=hr, a_rot=self.a_rot).view(np.uint16)
        return X.down(self.x, ex, self.sorted, self.off, self.n_expert, self.y_cols, h_rot=hr)

    def zero_rows(self):
        """Rows the op must leave zero: past the live count, or a negative sorted_tok."""
        live_end = int(self.off[self.n_expert])
        return [i for i in range(self.T) if i < int(self.off[0]) or i >= live_end or self.sorted[i] < 0]

    # ---- files
    def plane_tables(self, override=None):
        """[(operand position, name, dtype, dims, bytes)] of every plane table, in schema order; None for an
        absent odd table. override {(plane, class, entry): bytes} replaces one entry of one table."""
        out = []
        pos = POS[self.op]
        for c in range(len(pos["planes"])):
            for p, plane in enumerate(("codes", "suh", "suh1", "svh", "dn_suh")):
                k = pos["planes"][c][p]
                if k is None:
                    continue
                name = self.class_name(plane, c)
                dn_present = bool(self.h_rot) if self.dn_present is None else self.dn_present
                if c >= self.n_class or (p == 4 and not dn_present):
                    out.append((k, name, None, None, None))
                    continue
                R, C = self.geom[c]
                ex = self.experts[c::self.n_class]
                if p == 0:
                    ents = [X.pack_codes(e.sym, self.bits).tobytes() for e in ex]

                    dt, ent = f"u{self.bits}", [R, C]
                else:
                    if p == 4 and any(e.dn_suh is None for e in ex):
                        # No down experts paired with these: dn_suh from a private generator, so no shared
                        # random stream moves (version 2's seeded cases draw exactly what they drew).
                        own = np.random.default_rng(0xD5)
                        for e in ex:
                            if e.dn_suh is None:
                                w = e.rows // 2
                                e.dn_suh = ((own.integers(0, 2, w) * 2 - 1) * (0.5 + own.random(w)) * 0.01
                                            ).astype(np.float16)
                    vec = {1: lambda e: e.suh, 2: lambda e: e.suh1, 3: lambda e: e.svh, 4: lambda e: e.dn_suh}[p]
                    n = {1: C, 2: C, 3: R, 4: R // 2}[p]
                    if p == 4 and self.dn_nan:
                        ents = [np.full(n, np.nan, np.float16).tobytes() for _ in ex]
                    else:
                        ents = [vec(e).astype(np.float16).tobytes() for e in ex]
                    dt = "f16"
                    ent = [n] if self.scale_rank == 2 else ([R, 1] if p == 3 else [1, n])
                nbytes = len(ents[0]) if ents else X.plane_bytes(p, R, C, self.bits)
                for (pp, cc, j), b in (override or {}).items():
                    if pp == p and cc == c:
                        ents[j] = b
                ents += [bytes(np.random.default_rng(99 + p).integers(0, 256, nbytes, np.uint8))] * self.extra_entries
                out.append((k, name, dt, [len(ents)] + ent, b"".join(ents)))
        return out

    def write(self, d, sorted_tok=None, override=None):
        d.mkdir(parents=True, exist_ok=True)
        pos = POS[self.op]
        out = "xr" if self.prep else "h" if self.gate_up else "y"
        lines = [f"op {self.op}"]
        for k, v in self.params.items():
            lines.append(f"param {k} {'str' if isinstance(v, str) else 'int'} {v}")
        slots = [None] * pos["n"]
        xbits = X.to_bf16_bits(self.x) if self.a_dtype == "bf16" else X.to_f16(self.x).view(np.uint16)
        xbits.tofile(d / "x.bin")
        pitch = f" pitch={self.pitch}" if self.pitch else ""
        slots[pos["x"]] = f"opd {'h' if base(self.op) == DN else 'a'} in {self.a_dtype} 2 {self.x.shape[0]} {self.x.shape[1]} x.bin{pitch}"
        for k, name, dt, dims, blob in self.plane_tables(override):
            if dt is None:
                slots[k] = f"opd {name} absent"
                continue
            (d / f"{name}.bin").write_bytes(blob)
            slots[k] = f"opd {name} wtab {dt} {len(dims)} {' '.join(map(str, dims))} {name}.bin {self.form}"
        (self.sorted if sorted_tok is None else sorted_tok).astype(np.int32).tofile(d / "sorted.bin")
        self.off.astype(np.int32).tofile(d / "off.bin")
        slots[pos["sorted"]] = f"opd sorted_tok in i32 1 {self.T} sorted.bin"
        slots[pos["offset"]] = f"opd expert_offset in i32 1 {self.n_expert + 1} off.bin"
        ydt = "bf16" if base(self.op) == DN else "f16"
        slots[pos["y"]] = f"opd {out} out {ydt} 2 {self.T} {self.y_cols} -{pitch}"
        if "ids" in pos:
            if self.ids_form == "absent":
                slots[pos["ids"]] = "opd topk_ids absent"
            else:
                ids = self.ids if self.ids_form == "real" else \
                    np.random.default_rng(7).integers(-3, self.n_expert + 5, size=self.ids.shape)
                ids.astype(np.int32).tofile(d / "ids.bin")
                slots[pos["ids"]] = f"opd topk_ids in i32 2 {self.M} {self.top_k} ids.bin"
        (d / "case.txt").write_text("\n".join(lines + slots) + "\n")
        return d / "case.txt", d / f"out_{out}.bin"


def run(args, case, d, **write_kw):
    """Write, run; returns (rc, output uint16 [T, y_cols] or None, the harness's report)."""
    path, out = case.write(d, **write_kw)
    p = subprocess.run([args.harness, args.plugin, str(path), "--domain", args.domain], capture_output=True,
                       text=True, timeout=1800)
    if p.returncode == 77:
        print(p.stdout.strip())
        sys.exit(77)
    if p.returncode != 0:
        raise RuntimeError(f"harness failed ({p.returncode}): {p.stderr.strip()}")
    rep = {"shapes": {}, "layout": {}, "refused": []}
    for line in p.stdout.splitlines():
        f = line.split()
        if f[0] == "shape":
            dims = [] if f[5] == "-" else [int(v) for v in f[5].split(",")]
            rep["shapes"][int(f[1])] = dict(rc=int(f[2]), dtype=int(f[3]), dims=dims, flags=int(f[6]),
                                            fill=int(f[7]), idx_max=int(f[8]))
        elif f[0] == "layout":
            rep["layout"][int(f[1])] = " ".join(f[2:])
            if f[2] == "refused":
                rep["refused"].append(f"layout hook refused operand {f[1]} ({f[3]})")
        elif f[0] == "constraint":
            rep["refused"].append(f"constraint {f[1]} refused")
        elif f[0] == "row" and f[1] == "none":
            rep["refused"].append(f"the library has no {f[2]} row for {f[3]}")
        elif f[0] == "rc":
            rep["rc"] = None if f[1] == "none" else int(f[1])
        elif f[0] in ("redzone", "inputs"):
            rep[f[0]] = " ".join(f[1:])
    y = None
    if rep.get("rc") == 0 and out.exists():
        y = np.fromfile(out, np.uint16).reshape(case.T, case.y_cols)
    return rep.get("rc"), y, rep


# ----------------------------------------------------------------------------------------------- comparison
def ordered(b):
    """16-bit float bit patterns -> integers in value order, so a difference is a count of ulps."""
    b = b.astype(np.int64)
    return np.where(b & 0x8000, -(b & 0x7FFF), b & 0x7FFF)


def is_nan(b, kind):
    b = b.astype(np.int64)
    if kind == "f16":
        return ((b & 0x7C00) == 0x7C00) & ((b & 0x3FF) != 0)
    return ((b & 0x7F80) == 0x7F80) & ((b & 0x7F) != 0)


def compare(got, want, kind):
    """(max ulp, exact fraction, NaN mismatches)."""
    ng, nw = is_nan(got, kind), is_nan(want, kind)
    nan_bad = int((ng != nw).sum())
    both = ~(ng | nw)
    d = np.abs(ordered(got[both]) - ordered(want[both]))
    return (int(d.max()) if d.size else 0), float((d == 0).mean()) if d.size else 1.0, nan_bad


def values(b, kind):
    return b.view(np.float16).astype(np.float64) if kind == "f16" else X.bf16_bits_to_f32(b).astype(np.float64)


def agree(got, want, kind):
    """The host reference against the transcription: down to the bit, gate_up to one f16 ulp (libm's expf).
    A device library (--domain device): relative L2 within --tol, which the controls show is still far below
    any real error."""
    mu, ex, nb = compare(got, want, kind)
    if TOL is None:
        return nb == 0 and mu <= (1 if kind == "f16" else 0), f"max_ulp={mu} exact={ex:.6f} nan_mismatch={nb}"
    g, w = values(got, kind), values(want, kind)
    ok = np.isfinite(g) & np.isfinite(w)
    rel = float(np.linalg.norm(g[ok] - w[ok]) / max(np.linalg.norm(w[ok]), 1e-30))
    return nb == 0 and rel <= TOL, f"rel_l2={rel:.3e} (tol {TOL:g}) max_ulp={mu} exact={ex:.6f} nan_mismatch={nb}"


def check_case(args, case, d, label=None):
    """The full check of one case; returns (ok, output)."""
    label = label or case.name
    kind = "bf16" if base(case.op) == DN else "f16"
    rc, y, rep = run(args, case, d)
    if rep["refused"]:
        # The engine would not select this row for this geometry (its constraints, or its layout hook answering that
        # it does not read the encoding, rad_builder.cpp:536-563). A device library serving a subset is a skip, by
        # name; the reference serves every valid case, so there it is a failure.
        why = "; ".join(rep["refused"])
        if args.domain == "device":
            skip(f"{label}", f"the row does not serve this geometry: {why}")
            return True, None
        verdict(False, f"{label}: the reference serves the case", why)
        return False, None
    if not verdict(rc == 0, f"{label}: launch", f"rc={rc}"):
        return False, None
    ok = verdict(rep.get("redzone") == "ok" and rep.get("inputs") == "ok", f"{label}: margins and inputs",
                 f"redzone={rep.get('redzone')} inputs={rep.get('inputs')}")
    laid = {k: v for k, v in rep["layout"].items() if not v.startswith("as-is")}
    if laid:  # a check only where the row re-lays a table; exl3ref re-lays none
        ok &= verdict(all("unrelayout ok" in v for v in laid.values()), f"{label}: every re-laid plane inverts",
                      " ".join(f"{k}:{v}" for k, v in laid.items()))
    want = case.expected()
    good, detail = agree(y, want, kind)
    ok &= verdict(good, f"{label}: matches the transcription", detail)
    z = case.zero_rows()
    if case.prep:
        # prep_gu WRITES NOTHING outside the runs, nor past a row's 2 * C (include/exl3_abi.h, XR): still the prefill.
        outside = [i for i in range(case.T) if i < int(case.off[0]) or i >= int(case.off[case.n_expert])]
        ok &= verdict(not outside or bool((y[outside] == X.UNWRITTEN).all()),
                      f"{label}: rows outside the runs not written", f"{len(outside)} rows")
        tails = sum(int((y[i, 2 * case.geom[X.owner(case.off, i) % case.n_class][1]:] != X.UNWRITTEN).any())
                    for i in range(int(case.off[0]), int(case.off[case.n_expert])))
        ok &= verdict(tails == 0, f"{label}: columns past 2 * K_c not written", f"{tails} rows touched there")
        ok &= check_shapes(case, rep, label)
        return ok, y
    ok &= verdict(not z or not y[z].any(), f"{label}: dropped and past-live rows zero", f"{len(z)} rows")
    # A narrower class's tail columns.
    tails = 0
    for i in range(case.T):
        if i in z:
            continue
        e = X.owner(case.off, i)
        width = case.geom[e % case.n_class][0] // (2 if case.gate_up else 1)
        tails += int(y[i, width:].any())
    ok &= verdict(tails == 0, f"{label}: narrower class zero-filled", f"{tails} rows with a non-zero tail")
    ok &= check_shapes(case, rep, label)
    return ok, y


def check_shapes(case, rep, label):
    """The row's opd_shape against the case: same dtypes, extents, absence; valid routing fills."""
    s = rep["shapes"]
    pos = POS[case.op]
    # The input: f16 named (bf16 is the second form the token rows may take); prep_gu's xr is 2 * x_cols wide.
    want = {pos["x"]: (DT["f16"], [case.x.shape[0], 2 * case.x_cols if case.a_rot else case.x_cols]),
            pos["sorted"]: (DT["i32"], [case.T]), pos["offset"]: (DT["i32"], [case.n_expert + 1]),
            pos["y"]: (DT["bf16" if base(case.op) == DN else "f16"], [case.T, case.y_cols])}
    if "ids" in pos:
        want[pos["ids"]] = (DT["i32"], [case.M, case.top_k])
    absent = []
    for k, name, dt, dims, _ in case.plane_tables():
        if dt is None:
            absent.append(k)
        else:
            want[k] = (DT[dt], dims)   # [entries, rows, cols] of the plane: codes [R, C], suh [1, C], svh [R, 1]
    bad = []
    for k in range(pos["n"]):
        if s.get(k, {}).get("rc") != 0:
            bad.append(f"{k}:rc={s.get(k, {}).get('rc')}")
            continue
        if k in absent:
            if not s[k]["flags"] & ABSENT_FLAG:
                bad.append(f"{k} not described absent")
            continue
        dt, dims = want[k]
        if s[k]["dtype"] != dt or s[k]["dims"] != dims:
            bad.append(f"{k}: described {s[k]['dtype']} {s[k]['dims']}, built {dt} {dims}")
    if not s[pos["offset"]]["flags"] & IDX_CU_FLAG:
        bad.append("expert_offset not described as a cumulative run table")
    return verdict(not bad, f"{label}: opd_shape describes the case", "; ".join(bad))


# ----------------------------------------------------------------------------------------------- controls
def controls(args, case, y, d):
    """Read before the verdicts above count: the comparison can fail, and moves what it must and nothing else."""
    kind = "bf16" if base(case.op) == DN else "f16"
    ok = True
    if (case.gate_up and not case.a_rot) or case.prep:
        bad, detail = agree(y, case.expected(swap_suh=True), kind)
        ok &= verdict(not bad, f"{case.name}: control, swapped suh/suh.1 is caught", detail)
    if case.n_class == 2:
        swapped = [case.experts[e ^ 1] if (e ^ 1) < case.n_expert else case.experts[e] for e in range(case.n_expert)]
        bad, detail = agree(y, case.expected(experts=swapped), kind)
        ok &= verdict(not bad, f"{case.name}: control, swapped parity classes are caught", detail)
    if case.n_class == 4:
        # each expert read from the next class's entry of its group of four (the tables rotated by one class)
        rot = [case.experts[e - e % 4 + (e + 1) % 4] if e - e % 4 + (e + 1) % 4 < case.n_expert else case.experts[e]
               for e in range(case.n_expert)]
        bad, detail = agree(y, case.expected(experts=rot), kind)
        ok &= verdict(not bad, f"{case.name}: control, the four classes rotated by one are caught", detail)
        par = [case.experts[e ^ 2] if (e ^ 2) < case.n_expert else case.experts[e] for e in range(case.n_expert)]
        bad, detail = agree(y, case.expected(experts=par), kind)
        ok &= verdict(not bad, f"{case.name}: control, classes 0/2 and 1/3 swapped (a parity reading) are caught",
                      detail)

    # One flipped symbol moves the flipped expert's rows and nothing else.
    live = [e for e in range(case.n_expert) if any(case.sorted[i] >= 0 for i in range(case.off[e], case.off[e + 1]))]
    e = live[len(live) // 2]
    c, j = e % case.n_class, e // case.n_class
    if case.prep:   # no codes: the sign of one input scale of the gate half
        flipped = bytearray(case.experts[e].suh.astype(np.float16).tobytes())
        flipped[1] ^= 0x80
        plane = 1
    else:
        flipped = bytearray(X.pack_codes(case.experts[e].sym, case.bits).tobytes())
        flipped[0] ^= 0x01
        plane = 0
    rc, y2, _ = run(args, case, d / "flip", override={(plane, c, j): bytes(flipped)})
    own = np.zeros(case.T, bool)
    own[case.off[e]:case.off[e + 1]] = True
    moved_own = rc == 0 and bool((y2[own] != y[own]).any())
    moved_other = rc == 0 and bool((y2[~own] != y[~own]).any())
    ok &= verdict(moved_own and not moved_other,
                  f"{case.name}: control, one flipped {'suh sign' if case.prep else 'symbol'} of expert {e}",
                  f"rc={rc} own rows moved={moved_own} other rows moved={moved_other}")

    def same(label, **kw):
        saved = (case.form, case.pitch, case.extra_entries, case.scale_rank, case.dn_present, case.dn_nan)
        for k, v in kw.items():
            setattr(case, k, v)
        rc, y3, rep = run(args, case, d / label.replace(" ", "_").replace(",", ""))
        case.form, case.pitch, case.extra_entries, case.scale_rank, case.dn_present, case.dn_nan = saved
        return verdict(rc == 0 and np.array_equal(y3, y) and rep.get("redzone") == "ok",
                       f"{case.name}: neutral, {label}", f"rc={rc} identical={rc == 0 and np.array_equal(y3, y)}")

    ok &= same("stacked weight form", form="stacked")
    ok &= same("row pitch", pitch=case.x_cols + 48 if case.x_cols >= case.y_cols else case.y_cols + 48)
    ok &= same("extra table entries", extra_entries=2)
    ok &= same("scale tables declared rank 2", scale_rank=2)
    if case.gate_up and not case.h_rot:
        ok &= same("dn_suh tables present and NaN, h_rot 0 (not read)", dn_present=True, dn_nan=True)
    tail = case.sorted.copy()
    live_end = int(case.off[case.n_expert])
    tail[live_end:] = np.arange(case.T - live_end) % max(case.T, 1)
    rc, y4, _ = run(args, case, d / "tail", sorted_tok=tail)
    ok &= verdict(rc == 0 and np.array_equal(y4, y), f"{case.name}: neutral, valid indices past the live count",
                  f"rc={rc} identical={rc == 0 and np.array_equal(y4, y)}")
    return ok


# ----------------------------------------------------------------------------------------------- the layout path
def layout_path(args, case, y, d):
    """The harness's layout path through the two test plugins (tests/ref/layout_probe.c): codes re-laid by XOR,
    inverted, read back -- exl3ref's output to the bit -- and a wrong inverse caught."""
    if not args.probes:
        skip(f"{case.name}: harness layout path", "no --probes given")
        return True
    good_so, bad_so = args.probes
    saved = args.plugin
    try:
        args.plugin = good_so
        rc, y2, rep = run(args, case, d / "probe")
        codes = [POS[case.op]["planes"][c][0] for c in range(case.n_class)]
        laid_ok = all(rep["layout"].get(k) == "exl3probe.xor unrelayout ok" for k in codes)
        scales_as_is = all(v.startswith("as-is") for k, v in rep["layout"].items() if k not in codes)
        ok = verdict(rc == 0 and laid_ok and scales_as_is and np.array_equal(y2, y) and rep.get("redzone") == "ok"
                     and rep.get("inputs") == "ok", f"{case.name}: harness layout path, codes re-laid and inverted",
                     f"rc={rc} layout={rep['layout']} identical={rc == 0 and np.array_equal(y2, y)}")
        args.plugin = bad_so
        rc, _, rep = run(args, case, d / "probe_bad")
        caught = any("unrelayout FAIL" in v for v in rep["layout"].values())
        ok &= verdict(caught, f"{case.name}: control, a layout whose unrelayout does not invert is caught",
                      f"layout={rep['layout']}")
    finally:
        args.plugin = saved
    return ok


# ----------------------------------------------------------------------------------------------- EPI 1 (h_rot)
def chain(args, gu, dn_experts, d, label):
    """gate_up -> down at h_rot 0 and at h_rot 1 (interface v3, "EPI 1"): each op against the transcription, then
    the claims of the flag. gate_up's h_rot 1 output IS down's first step of its h_rot 0 output, to the bit; the
    pair's output is the same bits either way. Controls: the flag moves gate_up's output; the wrong expert's dn_suh
    is caught; down told h_rot 1 about an UNrotated h is caught."""
    ok = True
    saved = gu.params["h_rot"]
    gu.params["h_rot"] = 0
    ok0, h0 = check_case(args, gu, d / "gate_up0", f"{label} chain: gate_up h_rot 0")
    gu.params["h_rot"] = 1
    ok1, h1 = check_case(args, gu, d / "gate_up1", f"{label} chain: gate_up h_rot 1")
    ok &= ok0 and ok1
    if h0 is None or h1 is None:
        gu.params["h_rot"] = saved
        return ok        # a skipped op (device row not serving the geometry) skips its chain; a failure stays one
    # h1 against down's step 1 applied to exl3ref's own h0 (not the transcription's).
    step1 = np.zeros_like(h0)
    for i in range(gu.T):
        if i >= gu.off[gu.n_expert] or gu.sorted[i] < 0:
            continue
        e = X.owner(gu.off, i)
        W = gu.experts[e].rows // 2
        row = h0[i:i + 1, :W].view(np.float16).astype(np.float32)
        step1[i, :W] = X.to_f16(X._prologue(row, gu.experts[e].dn_suh)).view(np.uint16)[0]
    # On the host the same prologue code computes both sides: to the bit. A device library rotates in its own f32
    # order (its rotate128 / epilogue butterfly), so there it is within --tol like every other device comparison.
    if TOL is None:
        ok &= verdict(np.array_equal(h1, step1), f"{label} chain: gate_up h_rot 1 == down's step 1 of gate_up h_rot 0",
                      f"identical={np.array_equal(h1, step1)}")
    else:
        good, detail = agree(h1, step1, "f16")
        ok &= verdict(good, f"{label} chain: gate_up h_rot 1 == down's step 1 of gate_up h_rot 0 (within --tol)",
                      detail)
    ok &= verdict(not np.array_equal(h1, h0), f"{label} chain: control, h_rot moves gate_up's output",
                  f"identical={np.array_equal(h1, h0)}")
    # The partner a class count over (two, or four) is in the same class (same width) and is another expert.
    step = max(gu.n_class, 2)
    shifted = [gu.experts[e + step if e + step < gu.n_expert else e - step] for e in range(gu.n_expert)]
    keep = [ex.dn_suh for ex in gu.experts]
    for ex, sh in zip(gu.experts, [s.dn_suh for s in shifted]):
        ex.dn_suh = sh
    bad, detail = agree(h1, gu.expected(h_rot=1), "f16")
    for ex, k in zip(gu.experts, keep):
        ex.dn_suh = k
    ok &= verdict(not bad, f"{label} chain: control, another expert's dn_suh is caught", detail)

    def down_case(hbits, h_rot, name):
        return down_of(gu, dn_experts, hbits, h_rot, name)

    okd0, y0 = check_case(args, down_case(h0, 0, f"{label} chain: down h_rot 0"), d / "down0")
    okd1, y1 = check_case(args, down_case(h1, 1, f"{label} chain: down h_rot 1"), d / "down1")
    ok &= okd0 and okd1
    if (y0 is None or y1 is None) and okd0 and okd1:
        gu.params["h_rot"] = saved
        return ok        # skipped by name above
    ok &= verdict(y0 is not None and y1 is not None and np.array_equal(y0, y1),
                  f"{label} chain: the pair at h_rot 1 == the pair at h_rot 0, to the bit",
                  f"identical={y0 is not None and y1 is not None and np.array_equal(y0, y1)}")
    rc, yw, _ = run(args, down_case(h0, 1, f"{label} chain: down h_rot 1 on an unrotated h"), d / "down_wrong")
    ok &= verdict(rc == 0 and not np.array_equal(yw, y0),
                  f"{label} chain: control, down h_rot 1 on an unrotated h is caught",
                  f"rc={rc} identical={rc == 0 and np.array_equal(yw, y0)}")
    gu.params["h_rot"] = saved
    return ok


def chain_prep(args, gu, d, label):
    """prep_gu -> gate_up(a_rot 1) against gate_up(a_rot 0), at h_rot 0 and 1 (interface v4): prep_gu against the
    transcription (rows outside the runs untouched), then gate_up's output the same bits either way. Control: gate_up
    a_rot 1 handed an xr whose halves' scales are swapped is caught."""
    ok = True
    pg = copy.copy(gu)
    pg.name, pg.op, pg.gate_up, pg.prep = f"{label} prep: prep_gu", PG4 if gu.n_class == 4 else PG, False, True
    pg.y_cols = 2 * gu.x_cols
    pg.params = {k: v for k, v in gu.params.items() if k in PREP_KEYS}
    okp, xr = check_case(args, pg, d / "prep_gu")
    ok &= okp
    if xr is None:
        return ok        # skipped by name (a device library without the row), or failed above
    if pg.n_class == 4:   # topk_ids (P10-TP4S): a hint, the output may not depend on it
        for form in ("absent", "garbage"):
            q = copy.copy(pg)
            q.ids_form = form
            rc, xq, _ = run(args, q, d / f"prep_gu_ids_{form}")
            same = rc == 0 and xq is not None and np.array_equal(xq, xr)
            ok &= verdict(same, f"{label} prep: topk_ids {form}, the same xr to the bit", f"rc={rc} identical={same}")
    xr_vals = xr.view(np.float16).astype(np.float32)
    saved = dict(gu.params)
    for h in (0, 1):
        gu.params.update(h_rot=h, a_rot=0)
        ok0, y0 = check_case(args, gu, d / f"gate_up_a0_h{h}", f"{label} prep: gate_up a_rot 0 h_rot {h}")
        g1 = copy.copy(gu)
        g1.params = dict(gu.params, a_rot=1)
        g1.x, g1.a_dtype = xr_vals, "f16"
        ok1, y1 = check_case(args, g1, d / f"gate_up_a1_h{h}", f"{label} prep: gate_up a_rot 1 h_rot {h}")
        ok &= ok0 and ok1
        if y0 is None or y1 is None:
            continue
        ok &= verdict(np.array_equal(y0, y1), f"{label} prep: prep_gu -> gate_up(a_rot 1) == gate_up(a_rot 0) at "
                      f"h_rot {h}, to the bit", f"identical={np.array_equal(y0, y1)}")
        if h == 0:
            bad = copy.copy(g1)
            bad.x = pg.expected(swap_suh=True).view(np.float16).astype(np.float32)
            rc, yb, _ = run(args, bad, d / "gate_up_a1_swapped")
            ok &= verdict(rc == 0 and not np.array_equal(yb, y0),
                          f"{label} prep: control, gate_up a_rot 1 on an xr with swapped scales is caught",
                          f"rc={rc} identical={rc == 0 and np.array_equal(yb, y0)}")
    gu.params = saved
    return ok


# ----------------------------------------------------------------------------------------------- v5 gate_up_down
def down_of(gu, dn_experts, hbits, h_rot, name):
    """down over gate_up's routing and output h (f16 bits): [K, N / 2] a class, the experts `dn_experts`."""
    rows_gu = [gu.geom[c][0] for c in range(len(gu.geom))]
    dn = Case(name, DN, np.random.default_rng(0), M=gu.M, top_k=gu.top_k, n_expert=gu.n_expert,
              rows=gu.geom[0][1], cols=rows_gu[0] // 2, rows_odd=gu.geom[1][1] if gu.two else None,
              cols_odd=rows_gu[1] // 2 if gu.two else None, two=gu.two, bits=gu.bits, codebook=gu.codebook,
              experts=dn_experts, x=hbits.view(np.float16).astype(np.float32), h_rot=h_rot,
              classes=[(k, r // 2) for r, k in gu.geom] if gu.n_class == 4 else None)
    dn.ids, dn.sorted, dn.off = gu.ids, gu.sorted, gu.off
    return dn


class GdCase:
    """moe_exl3_gate_up_down (interface v5): down(h_rot 1) of gate_up(a_rot 0, h_rot 1) as one op, over a gate_up
    case's token rows, routing and experts and the down experts their dn_suh views. Writes the fused op's 18 operands
    from the two halves' own tables (gate_up's five planes a class, down's codes and svh as dn_codes / dn_svh). Over a
    four-class gate_up case, its _mod4 form (interface v7, 32 operands)."""

    def __init__(self, gu, dn_experts, name):
        self.gu = copy.copy(gu)
        self.gu.params = dict(gu.params, h_rot=1, a_rot=0)
        self.dn_experts, self.name, self.op = dn_experts, name, GD4 if gu.n_class == 4 else GD
        self.dn = down_of(self.gu, dn_experts, np.zeros((gu.T, gu.y_cols), np.uint16), 1, name)
        self.T, self.y_cols, self.n_expert, self.off, self.sorted = gu.T, self.dn.y_cols, gu.n_expert, gu.off, gu.sorted
        self.params = {k: v for k, v in gu.params.items() if k not in ("h_rot", "a_rot")}
        self.form = "table"

    def expected(self, dn_experts=None):
        g = self.gu
        h = X.gate_up(g.x, g.experts, g.sorted, g.off, g.n_expert, g.top_k, g.y_cols, h_rot=1, a_rot=0)
        return X.down(h.astype(np.float32), dn_experts or self.dn_experts, g.sorted, g.off, g.n_expert, self.y_cols,
                      h_rot=1)

    def zero_rows(self):
        return self.gu.zero_rows()

    def plane_tables(self, override=None, dn_override=None):
        """[(position, name, dtype, dims, bytes)] in the fused op's positions; override / dn_override as
        Case.plane_tables's, for gate_up's tables / down's."""
        P, nc = POS[self.op]["planes"], self.gu.n_class
        G, D = (GU4, DN4) if nc == 4 else (GU, DN)
        self.gu.form = self.dn.form = self.form
        out = []
        for k, name, dt, dims, blob in self.gu.plane_tables(override):
            c, p = next((c, p) for c in range(max(nc, 2)) for p in range(5) if POS[G]["planes"][c][p] == k)
            out.append((P[c][p], name, dt, dims, blob))
        for k, name, dt, dims, blob in self.dn.plane_tables(dn_override):
            c, p = next((c, p) for c in range(max(nc, 2)) for p in range(5) if POS[D]["planes"][c][p] == k)
            if p == 1:
                continue   # down's suh: the op reads it as gate_up's dn_suh view
            out.append((P[c][5 if p == 0 else 6], "dn_" + name, dt, dims, blob))
        return out

    def write(self, d, sorted_tok=None, override=None, dn_override=None):
        d.mkdir(parents=True, exist_ok=True)
        g, pos = self.gu, POS[self.op]
        lines = [f"op {self.op}"] + [f"param {k} {'str' if isinstance(v, str) else 'int'} {v}" for k, v in self.params.items()]
        slots = [None] * pos["n"]
        xbits = X.to_bf16_bits(g.x) if g.a_dtype == "bf16" else X.to_f16(g.x).view(np.uint16)
        xbits.tofile(d / "x.bin")
        slots[pos["x"]] = f"opd a in {g.a_dtype} 2 {g.x.shape[0]} {g.x.shape[1]} x.bin"
        for k, name, dt, dims, blob in self.plane_tables(override, dn_override):
            if dt is None:
                slots[k] = f"opd {name} absent"
                continue
            (d / f"{name}.bin").write_bytes(blob)
            slots[k] = f"opd {name} wtab {dt} {len(dims)} {' '.join(map(str, dims))} {name}.bin {self.form}"
        (g.sorted if sorted_tok is None else sorted_tok).astype(np.int32).tofile(d / "sorted.bin")
        g.off.astype(np.int32).tofile(d / "off.bin")
        slots[pos["sorted"]] = f"opd sorted_tok in i32 1 {self.T} sorted.bin"
        slots[pos["offset"]] = f"opd expert_offset in i32 1 {self.n_expert + 1} off.bin"
        slots[pos["y"]] = f"opd y out bf16 2 {self.T} {self.y_cols} -"
        (d / "case.txt").write_text("\n".join(lines + slots) + "\n")
        return d / "case.txt", d / "out_y.bin"


def chain_gd(args, gu, dn_experts, d, label):
    """moe_exl3_gate_up_down against the transcription AND against the separate ops through the same library --
    gate_up (a_rot 0, h_rot 1) then down (h_rot 1) -- to the bit (the op's definition, include/exl3_abi.h); margins,
    inputs, dropped rows zero, the shape hook. Controls: another expert's down weight is caught; one flipped down
    symbol moves its expert's rows and only those; the stacked weight form moves nothing."""
    g = GdCase(gu, dn_experts, f"{label} gate_up_down")
    rc, y, rep = run(args, g, d / "gd")
    if rep["refused"]:
        why = "; ".join(rep["refused"])
        if args.domain == "device":
            skip(g.name, f"the row does not serve this geometry: {why}")
            return True
        return verdict(False, f"{g.name}: the reference serves the case", why)
    if not verdict(rc == 0, f"{g.name}: launch", f"rc={rc}"):
        return False
    ok = verdict(rep.get("redzone") == "ok" and rep.get("inputs") == "ok", f"{g.name}: margins and inputs",
                 f"redzone={rep.get('redzone')} inputs={rep.get('inputs')}")
    laid = {k: v for k, v in rep["layout"].items() if not v.startswith("as-is")}
    if laid:
        ok &= verdict(all("unrelayout ok" in v for v in laid.values()), f"{g.name}: every re-laid plane inverts",
                      " ".join(f"{k}:{v}" for k, v in laid.items()))
    good, detail = agree(y, g.expected(), "bf16")
    ok &= verdict(good, f"{g.name}: matches the transcription", detail)
    # The separate ops, same library, same routing: the fused op is their composition to the bit.
    rc1, h, _ = run(args, g.gu, d / "sep_gate_up")
    rc2, ys, _ = (run(args, down_of(g.gu, dn_experts, h, 1, f"{label} sep down"), d / "sep_down") if rc1 == 0
                  else (rc1, None, None))
    same = rc2 == 0 and ys is not None and np.array_equal(y, ys)
    ok &= verdict(same, f"{g.name}: == gate_up(a_rot 0, h_rot 1) then down(h_rot 1), to the bit",
                  f"rc={rc1},{rc2} identical={same}")
    z = g.zero_rows()
    ok &= verdict(not z or not y[z].any(), f"{g.name}: dropped and past-live rows zero", f"{len(z)} rows")
    # opd_shape: every operand as built
    s, pos = rep["shapes"], POS[g.op]
    want = {pos["x"]: (DT["f16"], list(g.gu.x.shape)), pos["sorted"]: (DT["i32"], [g.T]),
            pos["offset"]: (DT["i32"], [g.n_expert + 1]), pos["y"]: (DT["bf16"], [g.T, g.y_cols])}
    absent = []
    for k, name, dt, dims, _ in g.plane_tables():
        if dt is None:
            absent.append(k)
        else:
            want[k] = (DT[dt], dims)
    bad = []
    for k in range(pos["n"]):
        if s.get(k, {}).get("rc") != 0:
            bad.append(f"{k}:rc={s.get(k, {}).get('rc')}")
        elif k in absent:
            if not s[k]["flags"] & ABSENT_FLAG:
                bad.append(f"{k} not described absent")
        elif (s[k]["dtype"], s[k]["dims"]) != want[k]:
            bad.append(f"{k}: described {s[k]['dtype']} {s[k]['dims']}, built {want[k][0]} {want[k][1]}")
    ok &= verdict(not bad, f"{g.name}: opd_shape describes the case", "; ".join(bad))
    # controls
    shifted = [dn_experts[e + 2 if e + 2 < g.n_expert else max(e - 2, 0)] for e in range(g.n_expert)]
    if g.n_expert > 2:
        bad_ok, detail = agree(y, g.expected(dn_experts=shifted), "bf16")
        ok &= verdict(not bad_ok, f"{g.name}: control, another expert's down weight is caught", detail)
    live = [e for e in range(g.n_expert) if any(g.sorted[i] >= 0 for i in range(g.off[e], g.off[e + 1]))]
    e = live[len(live) // 2]
    c, j = e % g.gu.n_class, e // g.gu.n_class
    flipped = bytearray(X.pack_codes(dn_experts[e].sym, g.gu.bits).tobytes())
    flipped[0] ^= 0x01
    rc, y2, _ = run(args, g, d / "flip", dn_override={(0, c, j): bytes(flipped)})
    own = np.zeros(g.T, bool)
    own[g.off[e]:g.off[e + 1]] = True
    mo, mx = rc == 0 and bool((y2[own] != y[own]).any()), rc == 0 and bool((y2[~own] != y[~own]).any())
    ok &= verdict(mo and not mx, f"{g.name}: control, one flipped down symbol of expert {e}",
                  f"rc={rc} own rows moved={mo} other rows moved={mx}")
    g.form = "stacked"
    rc, y3, _ = run(args, g, d / "stacked")
    g.form = "table"
    ok &= verdict(rc == 0 and np.array_equal(y3, y), f"{g.name}: neutral, stacked weight form",
                  f"rc={rc} identical={rc == 0 and np.array_equal(y3, y)}")
    return ok


def verify_routing(case, rng, group=4, mix=0.53):
    """Re-draw a case's routing as MTP verification routes (perfmoe --group G --group-mix R): tokens in runs of `group`
    share a draft, so each takes about `mix` of its top_k experts from the run's shared set and the rest at random --
    few experts with several rows each. Distinct experts a token; no drops."""
    n, k = case.n_expert, case.top_k
    ids = np.empty((case.M, k), np.int32)
    for g0 in range(0, case.M, group):
        shared = rng.permutation(n)[:k]
        for t in range(g0, min(g0 + group, case.M)):
            take = list(rng.permutation(shared)[:rng.binomial(k, mix)])
            rest = [e for e in rng.permutation(n) if e not in take][: k - len(take)]
            ids[t] = np.array(take + rest, np.int32)
    case.ids = ids
    case.sorted, off_full, _ = X.moe_scatter(ids, n, protect=())
    case.off = off_full[: n + 1].copy()
    return case


def chain_case(rng, label, M, top_k, n_expert, hidden, width, width_odd=None, drop=0.0, n_protect=0, bits=4,
               codebook="mul1", two=True, widths4=None):
    """A gate_up case and the down experts whose suh its dn_suh views (random, consistent). `widths4`: four widths,
    one a class by number mod 4 -- the _mod4 ops (`width` unused)."""
    gu = Case(label, "moe_exl3_gate_up", rng, M=M, top_k=top_k, n_expert=n_expert, rows=2 * width, cols=hidden,
              rows_odd=2 * width_odd if width_odd else None, drop=drop, n_protect=n_protect, bits=bits,
              codebook=codebook, two=two, classes=[(2 * w, hidden) for w in widths4] if widths4 else None)
    dn = [X.random_expert(rng, hidden, gu.geom[e % gu.n_class][0] // 2, bits, codebook, False)
          for e in range(n_expert)]
    for g_ex, d_ex in zip(gu.experts, dn):
        g_ex.dn_suh = d_ex.suh
    return gu, dn


# ----------------------------------------------------------------------------------------------- refusals
def harness_refusal(stdout):
    """(launch rc, or None when the harness stopped before launching; [refusals at selection or at layout])."""
    rc, why = None, []
    for l in stdout.splitlines():
        f = l.split()
        if not f:
            continue
        if f[0] == "rc" and f[1] != "none":
            rc = int(f[1])
        elif f[0] == "constraint":
            why.append(f"constraint {f[1]} refused")
        elif f[0] == "row" and len(f) > 3 and f[1] == "none":
            why.append(f"the library has no {f[2]} row for {f[3]}")
        elif f[0] == "layout" and len(f) > 3 and f[2] == "refused":
            why.append(f"layout hook refused operand {f[1]} ({f[3]})")
    return rc, why


def refused(name, rc, why, want):
    """A malformed case is refused by name. The reference (host) refuses it at its launch with the expected code
    whatever its constraints say -- the launch's own checks are the second wall and are tested as such. A device
    library may stop it before any launch -- its constraints or its layout hook, where the engine stops it -- and then
    its launch, if the harness reached it, must not return success."""
    if TOL is not None and any("has no" in w for w in why):
        # No row at all refuses nothing in particular: a check that checks nothing is a skip, never a pass.
        skip(name, "; ".join(why))
        return True
    if TOL is None or not why:
        extra = f"; also refused before launch: {'; '.join(why)}" if why else ""
        return verdict(rc == RAD[want], name, f"rc={rc} want {want} ({RAD[want]}){extra}")
    return verdict(rc is None or rc < 0, name, f"refused before launch: {'; '.join(why)} (launch rc={rc})")


def refusals(args, rng, work):
    base = dict(M=3, top_k=2, n_expert=4, rows=256, cols=256)
    cases = []

    def refuse(name, want, mutate, op="moe_exl3_gate_up", routing=False, **kw):
        """`routing`: the defect is in the routing operands' CONTENTS, which a device row can only see by reading them
        back to the host -- forbidden on the launch path (R6, rad_abi.h:97-98) -- so a device library skips it by name
        (tests/gpu carries the device refusal list); the reference, on the host, must still refuse it."""
        c = Case(f"refuse {name}", op, rng, **{**base, **kw})
        mutate(c)
        cases.append((c, want, routing))

    refuse("dtype exl3k1", "DTYPE", lambda c: c.params.update(dtype="exl3k1"))
    refuse("unknown codebook", "DTYPE", lambda c: c.params.update(codebook="mul2"))
    refuse("codes of another K", "DTYPE", lambda c: c.params.update(dtype="exl3k3"))
    refuse("N_odd without w_odd", "SHAPE", lambda c: c.params.update(N_odd=512), two=False)
    refuse("K not a multiple of 128", "SHAPE", lambda c: c.params.update(K=192, K_odd=192))
    refuse("gate_up N not a multiple of 256", "SHAPE", lambda c: c.params.update(N=384, N_odd=384))
    refuse("decreasing expert_offset", "SHAPE", lambda c: c.off.__setitem__(1, c.off[2] + 1), routing=True)
    refuse("expert_offset past T", "SHAPE", lambda c: c.off.__setitem__(-1, c.T + 1), routing=True)
    refuse("a sorted row past a's tokens", "SHAPE",
           lambda c: c.sorted.__setitem__(0, (c.M + 1) * c.top_k) if c.off[-1] > 0 else None, routing=True)
    refuse("output of the wrong width", "SHAPE", lambda c: setattr(c, "y_cols", c.y_cols + 16))
    refuse("bf16 activation into down", "DTYPE", lambda c: setattr(c, "a_dtype", "bf16"), op="moe_exl3_down")
    refuse("missing required parameter", "SCHEMA", lambda c: c.params.pop("codebook"))
    refuse("gate_up without h_rot", "SCHEMA", lambda c: c.params.pop("h_rot"))
    refuse("down without h_rot", "SCHEMA", lambda c: c.params.pop("h_rot"), op="moe_exl3_down")
    refuse("h_rot 2", "INVAL", lambda c: c.params.update(h_rot=2))
    refuse("down h_rot 2", "INVAL", lambda c: c.params.update(h_rot=2), op="moe_exl3_down")
    refuse("h_rot 1 without dn_suh", "INVAL", lambda c: (c.params.update(h_rot=1), setattr(c, "dn_present", False)))
    # a_rot and prep_gu (interface v4)
    refuse("gate_up without a_rot", "SCHEMA", lambda c: c.params.pop("a_rot"))
    refuse("a_rot 2", "INVAL", lambda c: c.params.update(a_rot=2))
    xr_like = lambda c, dt, w: (setattr(c, "x", np.zeros((c.T, w), np.float32)), setattr(c, "a_dtype", dt))  # noqa: E731
    refuse("a_rot 1, xr of bf16", "DTYPE", lambda c: xr_like(c, "bf16", 2 * c.x_cols), a_rot=1)
    refuse("a_rot 1, xr one half wide", "SHAPE", lambda c: xr_like(c, "f16", c.x_cols), a_rot=1)
    refuse("prep_gu without dtype", "SCHEMA", lambda c: c.params.pop("dtype"), op=PG)
    refuse("prep_gu K not a multiple of 128", "SHAPE", lambda c: c.params.update(K=192, K_odd=192), op=PG)
    refuse("prep_gu decreasing expert_offset", "SHAPE", lambda c: c.off.__setitem__(1, c.off[2] + 1), op=PG,
           routing=True)
    refuse("too few table entries", "SHAPE", lambda c: c.params.update(n_expert=6) or
           c.__setattr__("off", np.concatenate([c.off, [c.off[-1]] * 2]).astype(np.int32)))
    ok = True
    for i, (c, want, routing) in enumerate(cases):
        if routing and args.domain == "device":
            skip(c.name, "routing CONTENTS: a device row cannot read them on the launch path (R6); "
                         "the reference refuses it on the host")
            continue
        rc, _, rep = run(args, c, work / f"refuse{i}")
        ok &= refused(c.name, rc, rep["refused"], want)
    # A required operand absent: the operand is cut from the case file by hand.
    c = Case("refuse absent sorted_tok", "moe_exl3_gate_up", rng, **base)
    path, _ = c.write(work / "refuse_absent")
    text = path.read_text().splitlines()
    path.write_text("\n".join(l if not l.startswith("opd sorted_tok") else "opd sorted_tok absent" for l in text) + "\n")
    p = subprocess.run([args.harness, args.plugin, str(path), "--domain", args.domain], capture_output=True, text=True,
                       timeout=600)
    rc, why = harness_refusal(p.stdout)
    ok &= refused(c.name, rc, why, "INVAL")

    def edited(name, want, prefix, edit, op=GU, **kw):
        """A case whose operand line `prefix` is rewritten by `edit` -- a malformed table no Case builds."""
        c = Case(f"refuse {name}", op, rng, **{**base, **kw})
        path, _ = c.write(work / ("refuse_" + name.replace(" ", "_")))
        text = path.read_text().splitlines()
        path.write_text("\n".join(edit(l) if l.startswith(prefix) else l for l in text) + "\n")
        p = subprocess.run([args.harness, args.plugin, str(path), "--domain", args.domain], capture_output=True,
                           text=True, timeout=600)
        rc, why = harness_refusal(p.stdout)
        return refused(c.name, rc, why, want)

    ok &= edited("odd class partly absent", "INVAL", "opd suh_odd ", lambda l: "opd suh_odd absent")
    ok &= edited("suh table of bf16", "DTYPE", "opd suh ", lambda l: l.replace(" f16 ", " bf16 "))
    ok &= edited("codes table not [entries, rows, cols]", "SHAPE", "opd codes ",
                 lambda l: (lambda f: " ".join(f[:4] + ["2", f[5], str(int(f[6]) * int(f[7]))] + f[8:]))(l.split()))
    ok &= edited("svh table one row short", "SHAPE", "opd svh ",
                 lambda l: (lambda f: " ".join(f[:6] + [str(int(f[6]) - 16)] + f[7:]))(l.split()))
    # h_rot (interface v3)
    ok &= edited("h_rot 1, odd class without dn_suh_odd", "INVAL", "opd dn_suh_odd ",
                 lambda l: "opd dn_suh_odd absent", h_rot=1)
    ok &= edited("h_rot 1, dn_suh of bf16", "DTYPE", "opd dn_suh ", lambda l: l.replace(" f16 ", " bf16 "), h_rot=1)
    ok &= edited("h_rot 1, dn_suh 16 short", "SHAPE", "opd dn_suh ",
                 lambda l: (lambda f: " ".join(f[:7] + [str(int(f[7]) - 16)] + f[8:]))(l.split()), h_rot=1)
    ok &= edited("prep_gu xr of bf16", "DTYPE", "opd xr ", lambda l: l.replace(" f16 ", " bf16 "), op=PG)
    ok &= edited("prep_gu suh1 16 short", "SHAPE", "opd suh1 ",
                 lambda l: (lambda f: " ".join(f[:7] + [str(int(f[7]) - 16)] + f[8:]))(l.split()), op=PG)
    ok &= edited("prep_gu odd class partly absent", "INVAL", "opd suh_odd ", lambda l: "opd suh_odd absent", op=PG)
    # gate_up_down (interface v5): the fused op refuses what its halves refuse
    gu, dn = chain_case(rng, "refuse gd", M=3, top_k=2, n_expert=4, hidden=256, width=128)
    for i, (name, want, mutate) in enumerate((
            ("gate_up_down N not a multiple of 256", "SHAPE", lambda g: g.params.update(N=384)),
            ("gate_up_down without codebook", "SCHEMA", lambda g: g.params.pop("codebook")),
            ("gate_up_down dtype exl3k1", "DTYPE", lambda g: g.params.update(dtype="exl3k1")))):
        g = GdCase(gu, dn, f"refuse {name}")
        mutate(g)
        rc, _, rep = run(args, g, work / f"refuse_gd{i}")
        ok &= refused(g.name, rc, rep["refused"], want)
    return ok


# ----------------------------------------------------------------------------------------------- P1's fixture
def read_safetensors(path):
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    head = json.loads(raw[8:8 + n])
    np_dt = {"BF16": np.uint16, "F16": np.float16, "I16": np.int16, "I32": np.int32, "I64": np.int64,
             "F32": np.float32}
    out = {}
    for name, h in head.items():
        if name == "__metadata__":
            continue
        a, b = h["data_offsets"]
        out[name] = np.frombuffer(raw[8 + n + a:8 + n + b], np_dt[h["dtype"]]).reshape(h["shape"])
    return out


def plane_from_trellis(trellis, bits):
    """exllamav3 trellis [k/16, n/16, 16K] -> the canonical codes plane [n, k] of symbols."""
    kt, nt = trellis.shape[:2]
    sym = X.symbols_from_stream(trellis, bits)
    r, c = X.pos_to_tile()
    plane = np.zeros((nt * 16, kt * 16), np.uint32)
    for tk in range(kt):
        for tn in range(nt):
            plane[16 * tn + c, 16 * tk + r] = sym[tk, tn]
    return plane


def fixture(args, rng, work):
    """Layer 0 of P1's fixture, both ops, chained (down reads the h exl3ref wrote), against the transcription and
    against the skeleton's dense bf16 weights."""
    script = Path(__file__).resolve().parents[2] / "tools" / "exl3_fixture.py"
    if not script.exists():
        skip("fixture", f"no {script}")
        return True
    fx = work / "fixture"
    p = subprocess.run([sys.executable, str(script), "--out", str(fx), "--seed", "3"], capture_output=True,
                       text=True, timeout=600)
    if p.returncode != 0:
        return verdict(False, "fixture: generate", p.stderr.strip()[-400:])
    meta = json.loads((fx / "fixture.json").read_text())
    src = read_safetensors(fx / "exl3" / "model.safetensors")
    dense = read_safetensors(fx / "bf16" / "model.safetensors")
    bits, cb = meta["bits"], meta["codebook"]
    pre = "model.language_model.layers.0.mlp.experts"
    gate_up_ex, down_ex = [], []
    n_exp = sum(1 for k in src if k.startswith(pre) and k.endswith("gate_proj.trellis"))
    for e in range(n_exp):
        t = lambda proj, f: src[f"{pre}.{e}.{proj}.{f}"]  # noqa: E731
        gate = plane_from_trellis(t("gate_proj", "trellis"), bits)
        up = plane_from_trellis(t("up_proj", "trellis"), bits)
        gate_up_ex.append(X.Expert(np.concatenate([gate, up]), t("gate_proj", "suh"),
                                   np.concatenate([t("gate_proj", "svh"), t("up_proj", "svh")]),
                                   t("up_proj", "suh"), bits, cb))
        down_ex.append(X.Expert(plane_from_trellis(t("down_proj", "trellis"), bits), t("down_proj", "suh"),
                                t("down_proj", "svh"), None, bits, cb))
    for g_ex, d_ex in zip(gate_up_ex, down_ex):
        g_ex.dn_suh = d_ex.suh  # gate_up's view of down's suh: what h_rot = 1 reads
    top_k = 2
    rows_gu, hidden = gate_up_ex[0].rows, gate_up_ex[0].cols
    gu = Case("fixture gate_up", "moe_exl3_gate_up", rng, M=7, top_k=top_k, n_expert=n_exp, rows=rows_gu,
              cols=hidden, bits=bits, codebook=cb, drop=0.3, experts=gate_up_ex)
    ok, h = check_case(args, gu, work / "fx_gate_up")
    if h is None:
        return False
    hv = h.view(np.float16).astype(np.float32)
    dn = Case("fixture down", "moe_exl3_down", rng, M=7, top_k=top_k, n_expert=n_exp, rows=hidden,
              cols=rows_gu // 2, bits=bits, codebook=cb, experts=down_ex, x=hv)
    dn.ids, dn.sorted, dn.off = gu.ids, gu.sorted, gu.off
    ok2, y = check_case(args, dn, work / "fx_down")
    ok &= ok2
    if y is None:
        return False
    # Against the dense bf16 weights of the skeleton: y = x @ W, rounding points aside.
    wgu = X.bf16_bits_to_f32(dense[f"{pre}.gate_up_proj"])  # [E, 2w, h]
    wdn = X.bf16_bits_to_f32(dense[f"{pre}.down_proj"])     # [E, h, w]
    yd, hd = np.zeros_like(y, np.float32), np.zeros_like(hv)
    for i in range(gu.T):
        if gu.sorted[i] < 0 or i >= gu.off[n_exp]:
            continue
        e = X.owner(gu.off, i)
        x = gu.x[gu.sorted[i] // top_k].astype(np.float64)
        g, u = np.split(wgu[e].astype(np.float64) @ x, 2)
        hd[i] = g / (1 + np.exp(-g)) * u
        yd[i] = wdn[e].astype(np.float64) @ hv[i].astype(np.float64)
    rel = lambda a, b: float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))  # noqa: E731
    r_h, r_y = rel(hv, hd), rel(X.bf16_bits_to_f32(y), yd)
    # f16 rounding of the rotated input and bf16 weights in the dense form: ~1e-3 each; 2e-2 leaves headroom and
    # still fails a swapped half or a wrong scale by orders of magnitude.
    ok &= verdict(r_h < 2e-2 and r_y < 2e-2, "fixture: the ops compute x @ W of the skeleton's dense weights",
                  f"rel_l2 h={r_h:.2e} y={r_y:.2e}")
    gu.params["h_rot"] = 0
    ok &= chain(args, gu, down_ex, work / "fx_chain", "fixture")
    ok &= chain_prep(args, gu, work / "fx_prep", "fixture")
    ok &= chain_gd(args, gu, down_ex, work / "fx_gd", "fixture")
    return ok


# ----------------------------------------------------------------------------------------------- suites
def suite_cases(args, work):
    rng = np.random.default_rng(20261011)
    ok = True
    if args.golden and Path(args.golden).exists():
        res = X.selfcheck_decode(args.golden)
        ok &= verdict(all(r for _, r in res), "transcription decode vs the vllm-exl3 golden",
                      " ".join(f"{n}={'ok' if r else 'BAD'}" for n, r in res))
    else:
        skip("transcription decode vs the vllm-exl3 golden", f"no golden at {args.golden}")
    ok &= fixture(args, rng, work)

    main = [
        Case("gate_up two classes 3:2", "moe_exl3_gate_up", rng, M=9, top_k=4, n_expert=6, rows=768, cols=256,
             rows_odd=512, drop=0.1, n_protect=2),
        Case("down two classes 3:2", "moe_exl3_down", rng, M=9, top_k=4, n_expert=6, rows=256, cols=384,
             cols_odd=256, drop=0.1, n_protect=2),
    ]
    for c in main:
        good, y = check_case(args, c, work / c.name.replace(" ", "_"))
        ok &= good
        if y is not None:
            ok &= controls(args, c, y, work / (c.name.replace(" ", "_") + "_ctl"))
            if args.domain == "host":
                ok &= layout_path(args, c, y, work / (c.name.replace(" ", "_") + "_layout"))

    more = [
        Case("gate_up w_odd absent (tp1)", "moe_exl3_gate_up", rng, M=5, top_k=3, n_expert=5, rows=256, cols=256,
             two=False),
        Case("down w_odd absent (tp1)", "moe_exl3_down", rng, M=5, top_k=3, n_expert=5, rows=256, cols=128,
             two=False),
        Case("gate_up all slots of a token dropped", "moe_exl3_gate_up", rng, M=6, top_k=2, n_expert=4, rows=256,
             cols=256, drop=0.6),
        Case("down every slot dropped", "moe_exl3_down", rng, M=4, top_k=2, n_expert=4, rows=128, cols=128,
             drop=1.0),
        Case("gate_up bf16 a", "moe_exl3_gate_up", rng, M=4, top_k=2, n_expert=4, rows=256, cols=256,
             a_dtype="bf16"),
        Case("gate_up one expert (odd class empty)", "moe_exl3_gate_up", rng, M=3, top_k=1, n_expert=1, rows=256,
             cols=128),
    ]
    for bits, cb in ((2, "mul1"), (3, "mcg"), (5, "3inst"), (6, "mul1"), (8, "mcg"), (4, "3inst")):
        more.append(Case(f"gate_up K={bits} {cb}", "moe_exl3_gate_up", rng, M=3, top_k=2, n_expert=4, rows=256,
                         cols=128, bits=bits, codebook=cb))
        more.append(Case(f"down K={bits} {cb}", "moe_exl3_down", rng, M=3, top_k=2, n_expert=4, rows=128,
                         cols=128, bits=bits, codebook=cb))
    # The real model's per-rank geometry (Qwen3.8-Flash-Next: hidden 2560, expert 640, top-10): tp1 and the
    # 384/256 parity split of tp2 (PLAN.md D7).
    more += [
        Case("gate_up real tp1", "moe_exl3_gate_up", rng, M=2, top_k=10, n_expert=16, rows=1280, cols=2560),
        Case("gate_up real tp2 384/256", "moe_exl3_gate_up", rng, M=2, top_k=10, n_expert=16, rows=768,
             cols=2560, rows_odd=512),
        Case("down real tp2 384/256", "moe_exl3_down", rng, M=2, top_k=10, n_expert=16, rows=2560, cols=384,
             cols_odd=256),
    ]
    for c in more:
        if c.name.startswith("gate_up one expert"):
            # An odd class of no experts: the description declines (a tool skips it); the launch serves it.
            rc, y, rep = run(args, c, work / c.name.replace(" ", "_"))
            good, detail = agree(y, c.expected(), "f16") if y is not None else (False, "no output")
            ok &= verdict(rc == 0 and good, f"{c.name}: launch and matches", f"rc={rc} {detail}")
            ok &= verdict(rep["shapes"][2]["rc"] == -18, f"{c.name}: opd_shape declines the empty class",
                          f"rc={rep['shapes'][2]['rc']}")
            continue
        ok &= check_case(args, c, work / c.name.replace(" ", "_").replace("/", "_"))[0]
    # EPI 1 (h_rot, interface v3): the h_rot 1 gate_up through every control, then the chains.
    gu, dn = chain_case(rng, "gate_up h_rot 1, two classes 3:2", M=9, top_k=4, n_expert=6, hidden=256, width=384,
                        width_odd=256, drop=0.1, n_protect=2)
    gu.params["h_rot"] = 1
    good, y = check_case(args, gu, work / "gate_up_h_rot_1")
    ok &= good
    if y is not None:
        ok &= controls(args, gu, y, work / "gate_up_h_rot_1_ctl")
    gu.params["h_rot"] = 0
    ok &= chain(args, gu, dn, work / "chain_3_2", "two classes 3:2")
    ok &= chain_prep(args, gu, work / "prep_3_2", "two classes 3:2")
    ok &= chain_gd(args, gu, dn, work / "gd_3_2", "two classes 3:2")
    pg = Case("prep_gu two classes 3:2", PG, rng, M=9, top_k=4, n_expert=6, rows=768, cols=256, rows_odd=512,
              drop=0.1, n_protect=2)
    good, y = check_case(args, pg, work / "prep_gu_3_2")
    ok &= good
    if y is not None:
        ok &= controls(args, pg, y, work / "prep_gu_3_2_ctl")
    pg = Case("prep_gu bf16 a, odd tables absent", PG, rng, M=5, top_k=3, n_expert=5, rows=256, cols=256, two=False,
              a_dtype="bf16")
    ok &= check_case(args, pg, work / "prep_gu_bf16_tp1")[0]
    # A narrower odd class (K_odd < K): its rows use the first 2 * K_odd columns and leave the rest unwritten.
    pg = Case("prep_gu odd class narrower in K", PG, rng, M=6, top_k=2, n_expert=4, rows=256, cols=256, rows_odd=256,
              cols_odd=128)
    ok &= check_case(args, pg, work / "prep_gu_narrow_k")[0]
    gu, dn = chain_case(rng, "real tp2 384/256", M=2, top_k=10, n_expert=16, hidden=2560, width=384, width_odd=256)
    ok &= chain(args, gu, dn, work / "chain_real_tp2", "real tp2 384/256")
    ok &= chain_prep(args, gu, work / "prep_real_tp2", "real tp2 384/256")
    ok &= chain_gd(args, gu, dn, work / "gd_real_tp2", "real tp2 384/256")
    # (its own generator: the cases after it draw what they drew before v5)
    gu, dn = chain_case(np.random.default_rng(5), "real tp1", M=4, top_k=10, n_expert=16, hidden=2560, width=640,
                        two=False)
    ok &= chain_gd(args, gu, dn, work / "gd_real_tp1", "real tp1")
    gu, dn = chain_case(rng, "K=3 mcg", M=3, top_k=2, n_expert=4, hidden=128, width=128, bits=3, codebook="mcg")
    ok &= chain(args, gu, dn, work / "chain_k3_mcg", "K=3 mcg")
    ok &= chain_gd(args, gu, dn, work / "gd_k3_mcg", "K=3 mcg")
    ok &= refusals(args, rng, work)
    ok &= suite_mod4(args, work)
    return ok


# The tp4 rotation's per-rank widths (qwen4exp_exl3.cpp): rank r takes two of an expert's five 128-column blocks in
# class r and one in each other class.
TP4_WIDTHS = {r: [256 if q == r else 128 for q in range(4)] for r in range(4)}


def suite_mod4(args, work):
    """The _mod4 ops (interface v6): every class of every op against the transcription with the two-class cases'
    controls (and two of their own: the classes rotated by one, and read by parity, are caught), the layout path, the
    chains (h_rot, prep_gu -> a_rot) at the tp4 ranks' widths, classes of different K, a class with no experts, and
    their refusals. Its own generator: the cases before it draw what they drew before v6."""
    rng = np.random.default_rng(20261014)
    ok = True
    main = [
        Case("gate_up_mod4 rank 0 widths", GU, rng, M=9, top_k=4, n_expert=10, rows=0, cols=0, drop=0.1, n_protect=2,
             classes=[(2 * w, 256) for w in TP4_WIDTHS[0]]),
        Case("down_mod4 rank 2 widths", DN, rng, M=9, top_k=4, n_expert=10, rows=0, cols=0, drop=0.1, n_protect=2,
             classes=[(256, w) for w in TP4_WIDTHS[2]]),
        Case("prep_gu_mod4 classes of different K", PG, rng, M=9, top_k=4, n_expert=11, rows=0, cols=0, drop=0.1,
             n_protect=2, classes=[(256, 256), (256, 128), (256, 384), (256, 128)]),
    ]
    for c in main:
        good, y = check_case(args, c, work / label_of(c.name))
        ok &= good
        if y is not None:
            ok &= controls(args, c, y, work / (label_of(c.name) + "_ctl"))
            if args.domain == "host" and not c.prep:
                ok &= layout_path(args, c, y, work / (label_of(c.name) + "_layout"))
    more = [
        Case("gate_up_mod4 real tp4 rank 3", GU, rng, M=2, top_k=10, n_expert=16, rows=0, cols=0,
             classes=[(2 * w, 2560) for w in TP4_WIDTHS[3]]),
        Case("down_mod4 real tp4 rank 1", DN, rng, M=2, top_k=10, n_expert=16, rows=0, cols=0,
             classes=[(2560, w) for w in TP4_WIDTHS[1]]),
        Case("gate_up_mod4 bf16 a, n_expert 510 % 4", GU, rng, M=4, top_k=3, n_expert=6, rows=0, cols=0,
             a_dtype="bf16", classes=[(2 * w, 256) for w in TP4_WIDTHS[1]]),
        Case("down_mod4 K=3 mcg", DN, rng, M=3, top_k=2, n_expert=8, rows=0, cols=0, bits=3, codebook="mcg",
             classes=[(128, w) for w in TP4_WIDTHS[0]]),
    ]
    for c in more:
        ok &= check_case(args, c, work / label_of(c.name))[0]
    # Three experts: class 3 has none. The description declines (a tool skips it); the launch serves it.
    c = Case("gate_up_mod4 three experts (class 3 empty)", GU, rng, M=3, top_k=2, n_expert=3, rows=0, cols=0,
             classes=[(2 * w, 128) for w in TP4_WIDTHS[0]])
    rc, y, rep = run(args, c, work / label_of(c.name))
    if rep["refused"] and args.domain == "device":
        skip(c.name, "; ".join(rep["refused"]))
    else:
        good, detail = agree(y, c.expected(), "f16") if y is not None else (False, "no output")
        ok &= verdict(rc == 0 and good, f"{c.name}: launch and matches", f"rc={rc} {detail}")
        ok &= verdict(rep["shapes"][1]["rc"] == -18, f"{c.name}: opd_shape declines the empty class",
                      f"rc={rep['shapes'][1]['rc']}")
    for r in (0, 2):
        gu, dn = chain_case(rng, f"mod4 rank {r}", M=9, top_k=4, n_expert=10, hidden=256, width=0, drop=0.1,
                            n_protect=2, widths4=TP4_WIDTHS[r])
        ok &= chain(args, gu, dn, work / f"chain_mod4_r{r}", f"mod4 rank {r}")
        ok &= chain_prep(args, gu, work / f"prep_mod4_r{r}", f"mod4 rank {r}")
    gu, dn = chain_case(rng, "real tp4 rank 1", M=3, top_k=10, n_expert=16, hidden=2560, width=0,
                        widths4=TP4_WIDTHS[1])
    ok &= chain(args, gu, dn, work / "chain_real_tp4", "real tp4 rank 1")
    # v7: gate_up_down_mod4, the chain as one op over four classes -- against the transcription and the separate _mod4
    # ops to the bit, with the controls -- and over MTP-verify-like routing (own generator: the cases above unchanged)
    rng7 = np.random.default_rng(20261015)
    for r in (0, 3):
        gu, dn = chain_case(rng7, f"gd mod4 rank {r}", M=9, top_k=4, n_expert=10, hidden=256, width=0, drop=0.1,
                            n_protect=2, widths4=TP4_WIDTHS[r])
        ok &= chain_gd(args, gu, dn, work / f"gd_mod4_r{r}", f"mod4 rank {r}")
    gu, dn = chain_case(rng7, "gd mod4 verify routing", M=8, top_k=10, n_expert=16, hidden=256, width=0,
                        widths4=TP4_WIDTHS[1])
    verify_routing(gu, rng7)
    ok &= chain_gd(args, gu, dn, work / "gd_mod4_verify", "mod4 verify routing")
    ok &= chain_prep(args, gu, work / "prep_real_tp4", "real tp4 rank 1")
    ok &= refusals_mod4(args, rng, work)
    return ok


def label_of(name):
    return name.replace(" ", "_").replace("/", "_").replace(",", "").replace("%", "")


def refusals_mod4(args, rng, work):
    """What the _mod4 forms refuse beyond their two-class ops' list: a class's table absent (every class is
    required), a class geometry the format cannot hold, a class table too short, and the two-class op's flags."""
    ok = True
    base_kw = dict(M=3, top_k=2, n_expert=8, rows=0, cols=0, classes=[(256, 256)] * 4)
    cases = [
        ("gate_up_mod4 N_2 not a multiple of 256", "SHAPE", GU, lambda c: c.params.update(N_2=384)),
        ("down_mod4 K_3 not a multiple of 128", "SHAPE", DN, lambda c: c.params.update(K_3=192)),
        ("gate_up_mod4 without a_rot", "SCHEMA", GU, lambda c: c.params.pop("a_rot")),
        ("down_mod4 h_rot 2", "INVAL", DN, lambda c: c.params.update(h_rot=2)),
        ("prep_gu_mod4 K_1 not a multiple of 128", "SHAPE", PG, lambda c: c.params.update(K_1=192)),
    ]
    for i, (name, want, op, mutate) in enumerate(cases):
        kw = dict(base_kw, classes=[(256, 128)] * 4) if op == DN else base_kw
        c = Case(f"refuse {name}", op, rng, **kw)
        mutate(c)
        rc, _, rep = run(args, c, work / f"refuse_mod4_{i}")
        ok &= refused(c.name, rc, rep["refused"], want)

    def edited(name, want, prefix, edit, op=GU, **kw):
        c = Case(f"refuse {name}", op, rng, **{**base_kw, **kw})
        path, _ = c.write(work / ("refuse_" + label_of(name)))
        text = path.read_text().splitlines()
        path.write_text("\n".join(edit(l) if l.startswith(prefix) else l for l in text) + "\n")
        p = subprocess.run([args.harness, args.plugin, str(path), "--domain", args.domain], capture_output=True,
                           text=True, timeout=600)
        rc, why = harness_refusal(p.stdout)
        return refused(c.name, rc, why, want)

    ok &= edited("gate_up_mod4 class 2 codes absent", "INVAL", "opd codes_2 ", lambda l: "opd codes_2 absent")
    ok &= edited("prep_gu_mod4 class 3 suh1 absent", "INVAL", "opd suh1_3 ", lambda l: "opd suh1_3 absent", op=PG)
    ok &= edited("gate_up_mod4 h_rot 1, class 1 without dn_suh_1", "INVAL", "opd dn_suh_1 ",
                 lambda l: "opd dn_suh_1 absent", h_rot=1)
    ok &= edited("down_mod4 svh_3 table one row short", "SHAPE", "opd svh_3 ",
                 lambda l: (lambda f: " ".join(f[:6] + [str(int(f[6]) - 16)] + f[7:]))(l.split()), op=DN,
                 classes=[(256, 128)] * 4)
    return ok


def suite_band_tops(args, work):
    """Correct at the top of each band the device rows will declare (decode M <= 16, prefill up to the engine's
    --max-num-batched-tokens 4096; PLAN.md P4/P5), both ops, both h_rot: each band top a chain (gate_up then down,
    h_rot 0 and 1, every op against the transcription and the pair the same bits either way)."""
    rng = np.random.default_rng(4096)
    ok = True
    for M, top_k, n_expert in ((16, 10, 16), (4096, 2, 8)):
        gu, dn = chain_case(rng, f"band top M={M}", M=M, top_k=top_k, n_expert=n_expert, hidden=256, width=256,
                            width_odd=128, drop=0.02)
        ok &= chain(args, gu, dn, work / f"band_{M}", f"band top M={M}")
        ok &= chain_prep(args, gu, work / f"band_prep_{M}", f"band top M={M}")
        ok &= chain_gd(args, gu, dn, work / f"band_gd_{M}", f"band top M={M}")
    # The _mod4 ops (v6; gate_up_down_mod4, v7) at both band tops, rank 0's tp4 widths.
    rng4 = np.random.default_rng(40964)
    for M, top_k, n_expert in ((16, 10, 16), (4096, 2, 8)):
        gu, dn = chain_case(rng4, f"mod4 band top M={M}", M=M, top_k=top_k, n_expert=n_expert, hidden=256, width=0,
                            drop=0.02, widths4=TP4_WIDTHS[0])
        ok &= chain(args, gu, dn, work / f"band4_{M}", f"mod4 band top M={M}")
        ok &= chain_prep(args, gu, work / f"band4_prep_{M}", f"mod4 band top M={M}")
        ok &= chain_gd(args, gu, dn, work / f"band4_gd_{M}", f"mod4 band top M={M}")   # v7
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--harness", required=True)
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--golden")
    ap.add_argument("--suite", choices=("cases", "band-tops"), default="cases")
    ap.add_argument("--domain", choices=("host", "device"), default="host",
                    help="the rows to run: host (exl3ref, compared to the bit) or device (exl3r4d, within --tol)")
    ap.add_argument("--probes", nargs=2, metavar=("LAYOUT_PROBE", "LAYOUT_PROBE_BADINV"),
                    help="the two test plugins of tests/ref/layout_probe.c (the harness's layout path)")
    ap.add_argument("--tol", type=float, default=4e-3,
                    help="device mode: relative L2 against the transcription (PLAN.md D9's 4e-3)")
    args = ap.parse_args()
    global TOL
    TOL = args.tol if args.domain == "device" else None
    work = Path(args.work) / args.suite
    work.mkdir(parents=True, exist_ok=True)
    ok = suite_cases(args, work) if args.suite == "cases" else suite_band_tops(args, work)
    print(f"\n{args.suite}: {RESULTS['pass']} passed, {RESULTS['fail']} failed, {RESULTS['skip']} skipped")
    if RESULTS["pass"] + RESULTS["fail"] == 0:
        sys.exit(77)
    sys.exit(0 if ok and RESULTS["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
