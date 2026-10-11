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
DT = {"f16": 2, "bf16": 3, "u8": 9, "i32": 11}
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
                 bits=4, codebook="mul1", drop=0.0, n_protect=0, a_dtype="f16", experts=None, x=None):
        self.name, self.op, self.gate_up = name, op, op == "moe_exl3_gate_up"
        self.M, self.top_k, self.n_expert, self.bits, self.codebook = M, top_k, n_expert, bits, codebook
        self.n_class = 2 if two else 1
        self.geom = [(rows, cols), (rows_odd or rows, cols_odd or cols)]
        self.two = two
        self.T = M * top_k
        self.x_cols = max(cols, cols_odd or cols)
        self.y_cols = max(rows, rows_odd or rows) // (2 if self.gate_up else 1)
        self.experts = experts or [
            X.random_expert(rng, *self.geom[e % self.n_class], bits, codebook, self.gate_up) for e in range(n_expert)]
        # The router: top_k distinct experts a token over the regular and the protected ones; a dropped slot is a
        # negative id or one past the range the sort owns. Protected experts sort last and are not this op's.
        n_total = n_expert + n_protect
        assert top_k <= n_total, f"{name}: top_k {top_k} distinct experts out of {n_total}"
        ids = np.stack([rng.permutation(n_total)[:top_k] for _ in range(M)]).astype(np.int32)
        if drop:
            hit = rng.random(ids.shape) < drop
            ids[hit] = np.where(rng.random(hit.sum()) < 0.5, -1, n_total + 3)
        self.ids = ids
        self.sorted, off_full, _ = X.moe_scatter(ids, n_total, protect=tuple(range(n_expert, n_total)))
        self.off = off_full[: n_expert + 1].copy()
        self.a_dtype = a_dtype
        if x is not None:
            self.x = x
        elif self.gate_up:
            self.x = rng.standard_normal((M, self.x_cols)).astype(np.float32)
        else:
            self.x = (rng.standard_normal((self.T, self.x_cols)) * 0.5).astype(np.float32)
        self.x = self.narrow_x(self.x)
        self.params = {"M": M, "N": rows, "K": cols, "n_expert": n_expert, "top_k": top_k,
                       "dtype": f"exl3k{bits}", "codebook": codebook}
        if two:
            self.params.update({"N_odd": rows_odd or rows, "K_odd": cols_odd or cols})
        self.form, self.pitch, self.extra_entries = "table", None, 0

    def narrow_x(self, x):
        if self.a_dtype == "bf16":
            return X.bf16_bits_to_f32(X.to_bf16_bits(x))
        return X.to_f16(x).astype(np.float32)

    def tables(self, experts=None):
        ex = experts or self.experts
        return [ex[c::self.n_class] for c in range(self.n_class)]

    def expected(self, swap_suh=False, experts=None):
        ex = experts or self.experts
        if self.gate_up:
            return X.gate_up(self.x, ex, self.sorted, self.off, self.n_expert, self.top_k, self.y_cols,
                             swap_suh=swap_suh).view(np.uint16)
        return X.down(self.x, ex, self.sorted, self.off, self.n_expert, self.y_cols)

    def zero_rows(self):
        """Rows the op must leave zero: past the live count, or a negative sorted_tok."""
        live_end = int(self.off[self.n_expert])
        return [i for i in range(self.T) if i < int(self.off[0]) or i >= live_end or self.sorted[i] < 0]

    # ---- files
    def write(self, d, sorted_tok=None, records=None):
        d.mkdir(parents=True, exist_ok=True)
        out = "h" if self.gate_up else "y"
        xin = "a" if self.gate_up else "h"
        lines = [f"op {self.op}"]
        for k, v in self.params.items():
            lines.append(f"param {k} {'str' if isinstance(v, str) else 'int'} {v}")
        xbits = X.to_bf16_bits(self.x) if self.a_dtype == "bf16" else X.to_f16(self.x).view(np.uint16)
        xbits.tofile(d / "x.bin")
        pitch = f" pitch={self.pitch}" if self.pitch else ""
        lines.append(f"opd {xin} in {self.a_dtype} 2 {self.x.shape[0]} {self.x.shape[1]} x.bin{pitch}")
        recs = records or [[e.record() for e in tab] for tab in self.tables()]
        for c, name in enumerate(("w", "w_odd")):
            if c >= self.n_class:
                lines.append(f"opd {name} absent")
                continue
            blob = b"".join(recs[c])
            nbytes = X.rec_bytes(*self.geom[c], self.bits, self.gate_up)
            entries = len(recs[c])
            if self.extra_entries:
                blob += bytes(np.random.default_rng(99).integers(0, 256, nbytes * self.extra_entries, np.uint8))
                entries += self.extra_entries
            (d / f"{name}.bin").write_bytes(blob)
            lines.append(f"opd {name} wtab u8 2 {entries} {nbytes} {name}.bin {self.form}")
        (self.sorted if sorted_tok is None else sorted_tok).astype(np.int32).tofile(d / "sorted.bin")
        self.off.astype(np.int32).tofile(d / "off.bin")
        lines.append(f"opd sorted_tok in i32 1 {self.T} sorted.bin")
        lines.append(f"opd expert_offset in i32 1 {self.n_expert + 1} off.bin")
        ydt = "f16" if self.gate_up else "bf16"
        lines.append(f"opd {out} out {ydt} 2 {self.T} {self.y_cols} -{pitch}")
        (d / "case.txt").write_text("\n".join(lines) + "\n")
        return d / "case.txt", d / f"out_{out}.bin"


def run(args, case, d, **write_kw):
    """Write, run; returns (rc, output uint16 [T, y_cols] or None, the harness's report)."""
    path, out = case.write(d, **write_kw)
    p = subprocess.run([args.harness, args.plugin, str(path)], capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        raise RuntimeError(f"harness failed ({p.returncode}): {p.stderr.strip()}")
    rep = {"shapes": {}}
    for line in p.stdout.splitlines():
        f = line.split()
        if f[0] == "shape":
            dims = [] if f[5] == "-" else [int(v) for v in f[5].split(",")]
            rep["shapes"][int(f[1])] = dict(rc=int(f[2]), dtype=int(f[3]), dims=dims, flags=int(f[6]),
                                            fill=int(f[7]), idx_max=int(f[8]))
        elif f[0] == "rc":
            rep["rc"] = int(f[1])
        elif f[0] in ("redzone", "inputs"):
            rep[f[0]] = " ".join(f[1:])
    y = None
    if rep.get("rc") == 0 and out.exists():
        y = np.fromfile(out, np.uint16).reshape(case.T, case.y_cols)
    return rep["rc"], y, rep


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


def agree(got, want, kind):
    mu, ex, nb = compare(got, want, kind)
    return nb == 0 and mu <= (1 if kind == "f16" else 0), f"max_ulp={mu} exact={ex:.6f} nan_mismatch={nb}"


def check_case(args, case, d, label=None):
    """The full check of one case; returns (ok, output)."""
    label = label or case.name
    kind = "f16" if case.gate_up else "bf16"
    rc, y, rep = run(args, case, d)
    if not verdict(rc == 0, f"{label}: launch", f"rc={rc}"):
        return False, None
    ok = verdict(rep.get("redzone") == "ok" and rep.get("inputs") == "ok", f"{label}: margins and inputs",
                 f"redzone={rep.get('redzone')} inputs={rep.get('inputs')}")
    want = case.expected()
    good, detail = agree(y, want, kind)
    ok &= verdict(good, f"{label}: matches the transcription", detail)
    z = case.zero_rows()
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
    want = {0: (DT[case.a_dtype], [case.x.shape[0], case.x_cols]),
            3: (DT["i32"], [case.T]), 4: (DT["i32"], [case.n_expert + 1]),
            5: (DT["f16" if case.gate_up else "bf16"], [case.T, case.y_cols])}
    for c in range(2):
        if c < case.n_class:
            nbytes = X.rec_bytes(*case.geom[c], case.bits, case.gate_up)
            want[1 + c] = (DT["u8"], [len(case.experts[c::case.n_class]), nbytes])
    bad = []
    for k in range(6):
        if s.get(k, {}).get("rc") != 0:
            bad.append(f"{k}:rc={s.get(k, {}).get('rc')}")
            continue
        if k == 2 and case.n_class == 1:
            if not s[k]["flags"] & ABSENT_FLAG:
                bad.append("w_odd not described absent")
            continue
        dt, dims = want[k]
        if k == 0 and case.a_dtype == "bf16":
            dt = DT["f16"]  # the description names f16; bf16 is the second form the op reads
        if s[k]["dtype"] != dt or s[k]["dims"] != dims:
            bad.append(f"{k}: described {s[k]['dtype']} {s[k]['dims']}, built {dt} {dims}")
    if not s[4]["flags"] & IDX_CU_FLAG:
        bad.append("expert_offset not described as a cumulative run table")
    return verdict(not bad, f"{label}: opd_shape describes the case", "; ".join(bad))


# ----------------------------------------------------------------------------------------------- controls
def controls(args, case, y, d):
    """Read before the verdicts above count: the comparison can fail, and moves what it must and nothing else."""
    kind = "f16" if case.gate_up else "bf16"
    ok = True
    if case.gate_up:
        bad, detail = agree(y, case.expected(swap_suh=True), kind)
        ok &= verdict(not bad, f"{case.name}: control, swapped suh/suh.1 is caught", detail)
    if case.n_class == 2:
        swapped = [case.experts[e ^ 1] if (e ^ 1) < case.n_expert else case.experts[e] for e in range(case.n_expert)]
        bad, detail = agree(y, case.expected(experts=swapped), kind)
        ok &= verdict(not bad, f"{case.name}: control, swapped parity classes are caught", detail)

    # One flipped symbol moves the flipped expert's rows and nothing else.
    live = [e for e in range(case.n_expert) if any(case.sorted[i] >= 0 for i in range(case.off[e], case.off[e + 1]))]
    e = live[len(live) // 2]
    recs = [[ex.record() for ex in tab] for tab in case.tables()]
    c, j = e % case.n_class, e // case.n_class
    flipped = bytearray(recs[c][j])
    flipped[0] ^= 0x01
    recs[c][j] = bytes(flipped)
    rc, y2, _ = run(args, case, d / "flip", records=recs)
    own = np.zeros(case.T, bool)
    own[case.off[e]:case.off[e + 1]] = True
    moved_own = rc == 0 and bool((y2[own] != y[own]).any())
    moved_other = rc == 0 and bool((y2[~own] != y[~own]).any())
    ok &= verdict(moved_own and not moved_other, f"{case.name}: control, one flipped symbol of expert {e}",
                  f"rc={rc} own rows moved={moved_own} other rows moved={moved_other}")

    def same(label, **kw):
        saved = (case.form, case.pitch, case.extra_entries)
        for k, v in kw.items():
            setattr(case, k, v)
        rc, y3, rep = run(args, case, d / label.replace(" ", "_"))
        case.form, case.pitch, case.extra_entries = saved
        return verdict(rc == 0 and np.array_equal(y3, y) and rep.get("redzone") == "ok",
                       f"{case.name}: neutral, {label}", f"rc={rc} identical={rc == 0 and np.array_equal(y3, y)}")

    ok &= same("stacked weight form", form="stacked")
    ok &= same("row pitch", pitch=case.x_cols + 48 if case.x_cols >= case.y_cols else case.y_cols + 48)
    ok &= same("extra table entries", extra_entries=2)
    tail = case.sorted.copy()
    live_end = int(case.off[case.n_expert])
    tail[live_end:] = np.arange(case.T - live_end) % max(case.T, 1)
    rc, y4, _ = run(args, case, d / "tail", sorted_tok=tail)
    ok &= verdict(rc == 0 and np.array_equal(y4, y), f"{case.name}: neutral, valid indices past the live count",
                  f"rc={rc} identical={rc == 0 and np.array_equal(y4, y)}")
    return ok


# ----------------------------------------------------------------------------------------------- refusals
def refusals(args, rng, work):
    base = dict(M=3, top_k=2, n_expert=4, rows=256, cols=256)
    cases = []

    def refuse(name, want, mutate, op="moe_exl3_gate_up", **kw):
        c = Case(f"refuse {name}", op, rng, **{**base, **kw})
        mutate(c)
        cases.append((c, want))

    refuse("dtype exl3k1", "DTYPE", lambda c: c.params.update(dtype="exl3k1"))
    refuse("unknown codebook", "DTYPE", lambda c: c.params.update(codebook="mul2"))
    refuse("records of another K", "SHAPE", lambda c: c.params.update(dtype="exl3k3"))
    refuse("N_odd without w_odd", "SHAPE", lambda c: c.params.update(N_odd=512), two=False)
    refuse("K not a multiple of 128", "SHAPE", lambda c: c.params.update(K=192, K_odd=192))
    refuse("gate_up N not a multiple of 256", "SHAPE", lambda c: c.params.update(N=384, N_odd=384))
    refuse("decreasing expert_offset", "SHAPE", lambda c: c.off.__setitem__(1, c.off[2] + 1))
    refuse("expert_offset past T", "SHAPE", lambda c: c.off.__setitem__(-1, c.T + 1))
    refuse("a sorted row past a's tokens", "SHAPE",
           lambda c: c.sorted.__setitem__(0, (c.M + 1) * c.top_k) if c.off[-1] > 0 else None)
    refuse("output of the wrong width", "SHAPE", lambda c: setattr(c, "y_cols", c.y_cols + 16))
    refuse("bf16 activation into down", "DTYPE", lambda c: setattr(c, "a_dtype", "bf16"), op="moe_exl3_down")
    refuse("missing required parameter", "SCHEMA", lambda c: c.params.pop("codebook"))
    refuse("too few table entries", "SHAPE", lambda c: c.params.update(n_expert=6) or
           c.__setattr__("off", np.concatenate([c.off, [c.off[-1]] * 2]).astype(np.int32)))
    ok = True
    for i, (c, want) in enumerate(cases):
        rc, _, _ = run(args, c, work / f"refuse{i}")
        ok &= verdict(rc == RAD[want], c.name, f"rc={rc} want {want} ({RAD[want]})")
    # A required operand absent: the operand is cut from the case file by hand.
    c = Case("refuse absent sorted_tok", "moe_exl3_gate_up", rng, **base)
    path, _ = c.write(work / "refuse_absent")
    text = path.read_text().splitlines()
    path.write_text("\n".join(l if not l.startswith("opd sorted_tok") else "opd sorted_tok absent" for l in text) + "\n")
    p = subprocess.run([args.harness, args.plugin, str(path)], capture_output=True, text=True, timeout=600)
    rc = next((int(l.split()[1]) for l in p.stdout.splitlines() if l.startswith("rc ")), None)
    ok &= verdict(rc == RAD["INVAL"], c.name, f"rc={rc} want INVAL ({RAD['INVAL']})")
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
    ok &= refusals(args, rng, work)
    return ok


def suite_band_tops(args, work):
    """Correct at the top of each band the device rows will declare (decode M <= 16, prefill up to the engine's
    --max-num-batched-tokens 4096; PLAN.md P4/P5), both ops, the fixture's geometry and the real model's top-k."""
    rng = np.random.default_rng(4096)
    ok = True
    for M, top_k, n_expert in ((16, 10, 16), (4096, 2, 8)):
        for op, rows, cols, rows_odd, cols_odd in (("moe_exl3_gate_up", 512, 256, 256, None),
                                                   ("moe_exl3_down", 256, 256, None, 128)):
            c = Case(f"{op[len('moe_exl3_'):]} band top M={M}", op, rng, M=M, top_k=top_k, n_expert=n_expert, rows=rows,
                     cols=cols, rows_odd=rows_odd, cols_odd=cols_odd, drop=0.02)
            ok &= check_case(args, c, work / f"band_{op}_{M}")[0]
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--harness", required=True)
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--golden")
    ap.add_argument("--suite", choices=("cases", "band-tops"), default="cases")
    args = ap.parse_args()
    work = Path(args.work) / args.suite
    work.mkdir(parents=True, exist_ok=True)
    ok = suite_cases(args, work) if args.suite == "cases" else suite_band_tops(args, work)
    print(f"\n{args.suite}: {RESULTS['pass']} passed, {RESULTS['fail']} failed, {RESULTS['skip']} skipped")
    if RESULTS["pass"] + RESULTS["fail"] == 0:
        sys.exit(77)
    sys.exit(0 if ok and RESULTS["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
