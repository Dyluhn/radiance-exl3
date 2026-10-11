#!/usr/bin/env python3
"""test_exl3r4d.py -- the device rows (kernels/exl3r4d) against the NumPy transcription of the two ops.

    test_exl3r4d.py --harness exl3ref_run --plugin exl3r4d.so --work DIR --suite device|real [--extract F]

Every case runs through P2's engine-style harness (tests/ref/exl3ref_run.cpp, built with HIP), which loads the
library the way the engine does: dlopen, the row table, the row's opd_shape, its LAYOUT HOOKS (each codes entry
re-laid by relayout, inverted again by unrelayout and compared byte for byte), its scratch hook, the launch on a
HIP stream, 256-byte poisoned margins on every buffer, inputs checked unchanged. P2's case machinery
(tests/ref/test_exl3ref.py: the cases, the transcription, the controls) is reused as it is, read-only. rad-kbench
cannot do this at radiance 1.3.0: it zero-fills every weight-table operand (tools/kfixture.cpp:312; notes/p2.md).

suite device:
  - P1's fixture, both ops chained, against the transcription and the skeleton's dense bf16 weights;
  - P2's two-class cases (drops, protected rows past the live count) with P2's controls: swapped suh/suh.1 and
    swapped parity classes must FAIL the comparison; one flipped trellis symbol must move its expert's rows and
    nothing else; stacked scale tables, a row pitch, extra entries, rank-2 scale tables and valid indices past the
    live count must move nothing;
  - the decode band at the real model's per-rank geometry (hidden 2560, expert 640, top-10; tp1 and the 384/256
    parity split of tp2) at M = 1 .. 16: both the fused prep (rows in LDS) and the prep kernel (rows in scratch);
  - runs longer than one pass (a malformed but legal run table): several passes a run;
  - replay: the same case three times in three processes, byte-identical (R8);
  - refusals (R12): K != 4, mcg, 3inst at the layout hook (RAD_E_DTYPE), and what the launch refuses by name.
suite real: a real turboderp 4.05 tensor (layer 15's 4-bit indexer.index_qk_proj, 2560 -> 640, the routed
  experts' gate/up geometry) as gate and up, and its tiles re-tiled to 640 -> 2560 with the layer's real
  shared-expert down scales as down, against dense decode-and-multiply in float64: relative L2 <= 4e-3 (Dylan's
  bound, exl3-moe tests/test_real_layer.py). Without --extract: SKIPPED.

Exit 0 all passed, 1 any failure, 77 nothing could run.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ref"))
import test_exl3ref as R  # noqa: E402  (P2's; exits 77 itself when numpy is missing)
import exl3_numpy as X  # noqa: E402

np = R.np
verdict, skip = R.verdict, R.skip
GU, DN = "moe_exl3_gate_up", "moe_exl3_down"


def label(s):
    return s.replace(" ", "_").replace("/", "_").replace("=", "")


# ----------------------------------------------------------------------------------------------- device suite
def decode_band(args, rng, work):
    """The real per-rank geometry at decode M, both ops, fused prep and prep kernel."""
    ok = True
    geoms = [  # (op, name, rows, cols, rows_odd, cols_odd, M list)
        (GU, "tp1", 1280, 2560, None, None, (1, 6, 7, 16)),
        (GU, "tp2 384/256", 768, 2560, 512, None, (1, 2, 5, 6, 7, 10, 16)),
        (DN, "tp1", 2560, 640, None, None, (1, 6, 16)),
        (DN, "tp2 384/256", 2560, 384, None, 256, (1, 5, 16)),
    ]
    for op, name, rows, cols, ro, co, Ms in geoms:
        for M in Ms:
            c = R.Case(f"{op[9:]} real {name} M={M}", op, rng, M=M, top_k=10, n_expert=16, rows=rows, cols=cols,
                       rows_odd=ro, cols_odd=co, drop=0.02 if M >= 6 else 0.0, n_protect=2 if M == 16 else 0)
            ok &= R.check_case(args, c, work / label(c.name))[0]
    for M in (4, 12):  # bf16 activations, fused (4) and prep kernel (12)
        c = R.Case(f"gate_up real tp2 bf16 a M={M}", GU, rng, M=M, top_k=10, n_expert=16, rows=768, cols=2560,
                   rows_odd=512, a_dtype="bf16")
        ok &= R.check_case(args, c, work / label(c.name))[0]
    for op, rows, cols in ((GU, 1280, 2560), (DN, 2560, 640)):  # one class (the odd tables absent)
        c = R.Case(f"{op[9:]} real tp1 odd class absent M=3", op, rng, M=3, top_k=10, n_expert=16, rows=rows,
                   cols=cols, two=False)
        ok &= R.check_case(args, c, work / label(c.name))[0]
    return ok


def long_runs(args, rng, work):
    """Runs longer than a pass: all rows to expert 0, tokens repeated (a run table no router writes, but a legal
    one): the fused prep holds min(tokens, 16) rows a pass and the prep kernel 16, so these take several passes."""
    ok = True
    for op, M, top_k, rows, cols in ((GU, 4, 4, 512, 256), (GU, 7, 4, 768, 2560), (DN, 4, 4, 256, 256),
                                     (DN, 2, 20, 2560, 384)):
        c = R.Case(f"{op[9:]} one run of {M * top_k} rows, {M} tokens", op, rng, M=M, top_k=top_k,
                   n_expert=max(top_k, 4), rows=rows, cols=cols)
        c.sorted = np.arange(c.T, dtype=np.int32)
        c.off = np.full(c.n_expert + 1, c.T, np.int32)
        c.off[0] = 0
        ok &= R.check_case(args, c, work / label(c.name))[0]
    return ok


def replay(args, rng, work):
    """R8: one case, three processes, the same bytes."""
    ok = True
    for op, M, rows, cols, ro, co in ((GU, 16, 768, 2560, 512, None), (DN, 5, 2560, 384, None, 256)):
        c = R.Case(f"replay {op[9:]} M={M}", op, rng, M=M, top_k=10, n_expert=16, rows=rows, cols=cols,
                   rows_odd=ro, cols_odd=co, drop=0.05)
        outs = [R.run(args, c, work / label(f"{c.name} run{i}"))[1] for i in range(3)]
        same = all(o is not None for o in outs) and all(np.array_equal(outs[0], o) for o in outs[1:])
        ok &= verdict(same, f"{c.name}: three runs byte-identical")
    return ok


def run_rc(args, path):
    p = R.subprocess.run([args.harness, args.plugin, str(path), "--domain", args.domain], capture_output=True,
                         text=True, timeout=600)
    rep = {"rc": None, "refused": None}
    for line in p.stdout.splitlines():
        f = line.split()
        if f[:1] == ["rc"]:
            rep["rc"] = None if f[1] == "none" else int(f[1])
        if f[:1] == ["layout"] and len(f) > 3 and f[2] == "refused":
            rep["refused"] = int(f[3])
    return rep


def refusals(args, rng, work):
    """Nothing unimplemented returns success (R12). The layout hook refuses an encoding this library does not
    decode before anything is stored (as at declare); the launch refuses what it cannot take, by name."""
    ok = True
    for bits, cb in ((2, "mul1"), (3, "mul1"), (5, "mul1"), (6, "mul1"), (8, "mul1"), (4, "mcg"), (4, "3inst")):
        for op, rows, cols in ((GU, 256, 128), (DN, 128, 128)):
            c = R.Case(f"refuse {op[9:]} K={bits} {cb}", op, rng, M=3, top_k=2, n_expert=4, rows=rows, cols=cols,
                       bits=bits, codebook=cb)
            path, _ = c.write(work / label(c.name))
            rep = run_rc(args, path)
            ok &= verdict(rep["refused"] == R.RAD["DTYPE"] and rep["rc"] is None,
                          f"{c.name}: refused at the layout hook", f"refused={rep['refused']} rc={rep['rc']}")
    base = dict(M=3, top_k=2, n_expert=4, rows=256, cols=256)

    def launch_refuses(name, want, mutate=None, op=GU, edit=None):
        c = R.Case(f"refuse {name}", op, rng, **base)
        if mutate:
            mutate(c)
        path, _ = c.write(work / label(c.name))
        if edit:
            prefix, fn = edit
            text = path.read_text().splitlines()
            path.write_text("\n".join(fn(l) if l.startswith(prefix) else l for l in text) + "\n")
        rep = run_rc(args, path)
        got = rep["rc"] if rep["refused"] is None else rep["refused"]
        return verdict(got == R.RAD[want], c.name, f"got {got} want {want} ({R.RAD[want]})")

    ok &= launch_refuses("bf16 activation into down", "DTYPE", lambda c: setattr(c, "a_dtype", "bf16"), op=DN)
    ok &= launch_refuses("output of the wrong width", "SHAPE", lambda c: setattr(c, "y_cols", c.y_cols + 16))
    ok &= launch_refuses("missing codebook parameter", "SCHEMA", lambda c: c.params.pop("codebook"))
    ok &= launch_refuses("K not a multiple of 128", "SHAPE", lambda c: c.params.update(K=192, K_odd=192))
    ok &= launch_refuses("gate_up N not a multiple of 256", "SHAPE", lambda c: c.params.update(N=384, N_odd=384))
    ok &= launch_refuses("a row pitch the 8-byte loads cannot take", "ALIGN", lambda c: setattr(c, "pitch", 258))
    ok &= launch_refuses("odd class partly absent", "INVAL", edit=("opd suh_odd ", lambda l: "opd suh_odd absent"))
    ok &= launch_refuses("suh table of bf16", "DTYPE", edit=("opd suh ", lambda l: l.replace(" f16 ", " bf16 ")))
    ok &= launch_refuses("svh table one row short", "SHAPE",
                         edit=("opd svh ", lambda l: (lambda f: " ".join(f[:6] + [str(int(f[6]) - 16)] + f[7:]))(
                             l.split())))
    ok &= launch_refuses("codes table not [entries, rows, cols]", "SHAPE",
                         edit=("opd codes ", lambda l: (lambda f: " ".join(
                             f[:4] + ["2", f[5], str(int(f[6]) * int(f[7]))] + f[8:]))(l.split())))
    return ok


def suite_device(args, work):
    rng = np.random.default_rng(20261012)
    ok = R.fixture(args, rng, work)
    for c in (R.Case("gate_up two classes 3:2", GU, rng, M=9, top_k=4, n_expert=6, rows=768, cols=256,
                     rows_odd=512, drop=0.1, n_protect=2),
              R.Case("down two classes 3:2", DN, rng, M=9, top_k=4, n_expert=6, rows=256, cols=384, cols_odd=256,
                     drop=0.1, n_protect=2)):
        good, y = R.check_case(args, c, work / label(c.name))
        ok &= good
        if y is not None:
            ok &= R.controls(args, c, y, work / (label(c.name) + "_ctl"))
    ok &= decode_band(args, rng, work)
    ok &= long_runs(args, rng, work)
    ok &= replay(args, rng, work)
    ok &= refusals(args, rng, work)
    return ok


# ----------------------------------------------------------------------------------------------- real suite
def dense(ex):
    """W [rows, cols] (radiance's [n, k]) of one expert, float64: exllamav3's order -- had over k, x suh, had over
    n, x svh (common/exl3_decode.c decode_block), from the decoded W_hat. A stacked gate_up's up half takes suh1."""
    w = ex.what.astype(np.float32)
    w = X.had128(w)                                       # over k, along each row
    half = ex.rows // 2 if ex.suh1 is not None else ex.rows
    w[:half] *= ex.suh.astype(np.float32)[None, :]
    if ex.suh1 is not None:
        w[half:] *= ex.suh1.astype(np.float32)[None, :]
    w = X.had128(w.T).T                                   # over n, down each column
    return w.astype(np.float64) * ex.svh.astype(np.float64)[:, None]


def suite_real(args, work):
    if not args.extract or not Path(args.extract).exists():
        skip("real layer", f"no extract given or found ({args.extract!r}); pass --extract / -DEXL3_REAL_EXTRACT")
        return True
    t = R.read_safetensors(args.extract)
    pre = "model.language_model.layers.15."
    qk = pre + "self_attn.indexer.index_qk_proj."
    sd = pre + "mlp.shared_expert.down_proj."
    need = [qk + s for s in ("trellis", "suh", "svh", "mul1")] + [sd + s for s in ("suh", "svh")]
    missing = [n for n in need if n not in t]
    if missing:
        return verdict(False, "real layer: tensors present", f"missing {missing}")
    tr = t[qk + "trellis"]
    ok = verdict(tr.shape == (160, 40, 64) and int(t[qk + "mul1"]) & 0xFFFFFFFF == X_MUL1,
                 "real layer: index_qk_proj is 4-bit mul1, 2560 -> 640", f"trellis {tr.shape}")
    plane = R.plane_from_trellis(tr.view(np.uint16), 4).astype(np.uint32)   # [640, 2560] symbols
    suh, svh = t[qk + "suh"].astype(np.float16), t[qk + "svh"].astype(np.float16)
    gate_up = X.Expert(np.concatenate([plane, plane]), suh, np.concatenate([svh, svh]), suh, 4, "mul1")
    # down: the same real 4-bit tiles in the down geometry (k 640, n 2560), the layer's real down scales
    tiles = tr.reshape(-1, 64)[: 40 * 160].reshape(40, 160, 64)
    dplane = R.plane_from_trellis(tiles.view(np.uint16), 4).astype(np.uint32)  # [2560, 640]
    down = X.Expert(dplane, t[sd + "suh"].astype(np.float16), t[sd + "svh"].astype(np.float16), None, 4, "mul1")
    Wg = dense(gate_up)
    Wd = dense(down)
    rng = np.random.default_rng(15)
    rel = lambda a, b: float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))  # noqa: E731
    for M in (1, 5, 7, 16):
        gu = R.Case(f"real gate_up M={M}", GU, rng, M=M, top_k=1, n_expert=1, rows=1280, cols=2560,
                    experts=[gate_up], two=False)
        rc, h, _ = R.run(args, gu, work / label(gu.name))
        if not verdict(rc == 0 and h is not None, f"{gu.name}: launch", f"rc={rc}"):
            ok = False
            continue
        hv = h.view(np.float16).astype(np.float64)
        g, u = np.split(gu.x.astype(np.float64) @ Wg.T, 2, axis=1)
        h_ref = g / (1 + np.exp(-g)) * u                                   # rows: token i = sorted row i (top-1)
        r_h = rel(hv, h_ref[gu.sorted // 1])
        ok &= verdict(r_h <= args.tol, f"{gu.name}: device vs decode-and-multiply", f"rel_l2={r_h:.3e} (tol {args.tol:g})")
        dn = R.Case(f"real down M={M}", DN, rng, M=M, top_k=1, n_expert=1, rows=2560, cols=640, experts=[down],
                    two=False, x=hv.astype(np.float32))
        dn.sorted, dn.off = gu.sorted, gu.off
        rc, y, _ = R.run(args, dn, work / label(dn.name))
        if not verdict(rc == 0 and y is not None, f"{dn.name}: launch", f"rc={rc}"):
            ok = False
            continue
        y_ref = dn.x.astype(np.float64) @ Wd.T
        r_y = rel(X.bf16_bits_to_f32(y).astype(np.float64), y_ref)
        ok &= verdict(r_y <= args.tol, f"{dn.name}: device vs decode-and-multiply", f"rel_l2={r_y:.3e} (tol {args.tol:g})")
    # Negative control: the comparison fails against a reference whose up half has the wrong scales.
    bad = dense(X.Expert(gate_up.sym, suh, np.concatenate([svh, -svh]), suh, 4, "mul1"))
    g2, u2 = np.split(gu.x.astype(np.float64) @ bad.T, 2, axis=1)
    if h is not None:
        r_bad = rel(hv, (g2 / (1 + np.exp(-g2)) * u2)[gu.sorted])
        ok &= verdict(r_bad > 10 * args.tol, "real layer: control, a wrong reference is caught", f"rel_l2={r_bad:.3e}")
    return ok


X_MUL1 = 0x83DCD12D


def in_container(args):
    """Route P2's harness invocations (and only those) through `docker exec`, host paths under the repository
    rewritten to /work: R.run calls subprocess.run([harness, plugin, case, ...])."""
    repo = str(HERE.parents[1])
    real = R.subprocess

    def where(a):
        a = str(a)
        return "/work" + a[len(repo):] if a.startswith(repo) else a

    class Shim:
        PIPE = real.PIPE

        @staticmethod
        def run(cmd, **kw):
            if cmd and str(cmd[0]) == str(args.harness):
                cmd = ["docker", "exec", "-w", "/work", args.container] + [where(c) for c in cmd]
            return real.run(cmd, **kw)
    R.subprocess = Shim
    args.harness = str(Path(args.harness).resolve())
    args.plugin = str(Path(args.plugin).resolve())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--harness", required=True)
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--suite", choices=("device", "real"), default="device")
    ap.add_argument("--extract", default="")
    ap.add_argument("--tol", type=float, default=4e-3, help="relative L2 (PLAN.md D9's 4e-3)")
    ap.add_argument("--container", default="",
                    help="run the harness in this running container (tests/gpu/run_gpu.sh's $EXL3_CONTAINER), the "
                         "repository at /work there; this driver and NumPy stay on the host")
    args = ap.parse_args()
    if args.container:
        in_container(args)
    args.domain = "device"
    args.probes = None
    args.golden = None
    R.TOL = args.tol
    work = Path(args.work) / args.suite
    work.mkdir(parents=True, exist_ok=True)
    ok = suite_device(args, work) if args.suite == "device" else suite_real(args, work)
    res = R.RESULTS
    print(f"\n{args.suite}: {res['pass']} passed, {res['fail']} failed, {res['skip']} skipped")
    if res["pass"] + res["fail"] == 0:
        sys.exit(77)
    sys.exit(0 if ok and res["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
