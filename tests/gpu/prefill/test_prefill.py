#!/usr/bin/env python3
"""test_prefill.py -- exl3r4d's M > 16 rows (exl3r4d_moe_*_pf) against the NumPy transcription of the two ops.

    test_prefill.py --mode plan  --work DIR                     # host, no GPU: write every case, list the runs
    tests/gpu/prefill/run_cases.sh DIR <harness> <rowpick.so> <exl3r4d.so>   # under tests/gpu/run_gpu.sh (lock)
    test_prefill.py --mode check --work DIR                     # host, no GPU: read the runs back, judge

THREE PHASES SO THE SHARED GPU LOCK IS HELD ONLY FOR THE LAUNCHES. Building the cases and the transcription's
answers takes minutes of CPU at the band top (M = 4096, 40,960 routed rows); the launches take seconds. `plan` and
`check` build exactly the same cases from the same seed (every random draw happens in the same order: in plan mode
every harness call is answered as a successful launch, so the control flow -- which cases and which controls run --
is the one a passing check takes). `plan` writes the case files and the list of harness runs; `run_cases.sh` runs
P2's engine-style harness (tests/ref/exl3ref_run.cpp: dlopen, the row's opd_shape, its LAYOUT hooks with the
relayout -> unrelayout round trip, its scratch hook, the launch, 256-byte poisoned margins on every buffer, inputs
checked unchanged) on each, inside the session container; `check` replays the harness's saved output through P2's
own checks (tests/ref/test_exl3ref.py check_case / controls), read-only.

The harness takes an op's FIRST device row; tests/gpu/prefill/rowpick.c is a test-only shim plugin that exposes only
the rows whose name holds `_pf`, so these cases run the M > 16 rows (the decode rows come first in the table).

THE TRANSCRIPTION'S GEMM: P2's _gemm_rows sums k one step at a time (bit-exact with the host reference, minutes at the
band top); here it is swapped for an f32 BLAS product, which differs from it by ~1e-6 relative -- three orders below
the device tolerance (4e-3 relative L2) and below anything the controls catch (swapped scales: ~1.4).

Cases (the form each takes on the real model is decided by slots per expert, 2T <= 3 n_expert -> decode kernels):
  - P2's two-class cases with drops and protected rows, at M = 24 (prefill form), with P2's controls: swapped
    suh/suh.1 and swapped parity classes must FAIL; one flipped symbol must move its expert's rows and nothing else;
    the stacked form, a row pitch, extra entries, rank-2 scale tables, valid indices past the live count move nothing;
  - the real per-rank geometry (hidden 2560, expert 640, top-10; tp1 and the 384/256 split of tp2) at M = 17, 32, 64,
    512 (32 experts: prefill form) and M = 17 / 24 over 128 experts at tp2 (T = 170 / 240: the decode-like form at
    M > 16, then just past the crossover); group `tops` alone: the band top M = 4096 and the production chunk 2048;
  - runs longer than a map entry (one expert takes every row; skewed routing), rows before off[0], bf16 activations,
    one class (odd tables absent);
  - replay: one case three times in three processes, byte-identical (R8);
  - group `run`: the run form (no map) at T = 170 / 320 over 256 experts, tp2, and runs of 40 rows (several passes).
Exit 0 all passed, 1 any failure, 77 nothing could run.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "ref"))
import test_exl3ref as R  # noqa: E402  (P2's; exits 77 itself when numpy is missing)
import exl3_numpy as X  # noqa: E402

np = R.np
GU, DN = "moe_exl3_gate_up", "moe_exl3_down"
HARNESS = "exl3ref_run"   # the name R.run calls; the shim below answers for it


def fast_gemm(xr, w_hat):
    return (xr.astype(np.float32) @ w_hat.astype(np.float32).T).astype(np.float32)


X._gemm_rows = fast_gemm


def label(s):
    return s.replace(" ", "_").replace("/", "_").replace("=", "").replace(":", "")


class Replay:
    """Stands in for subprocess in P2's module. plan: record the case, answer 'launched' with a zero output.
    check: answer with what run_cases.sh saved beside the case."""
    PIPE = -1

    def __init__(self, mode):
        self.mode, self.runs = mode, []

    def run(self, cmd, **kw):
        case = Path(cmd[2])
        d = case.parent
        if self.mode == "plan":
            self.runs.append(str(d))
            lines = (d / "case.txt").read_text().splitlines()
            out = next(l for l in lines if l.split()[2:3] == ["out"])
            f = out.split()
            rows, cols = int(f[5]), int(f[6])   # opd <name> out <dtype> 2 <rows> <cols> ...
            name = f[1]
            np.zeros((rows, cols), np.uint16).tofile(d / f"out_{name}.bin")
            return Res(0, "rc 0\nredzone ok\ninputs ok\n", "")
        rc_file, out_file = d / "harness.rc", d / "harness.out"
        if not rc_file.exists():
            return Res(2, "", f"no saved run for {d} (run_cases.sh did not reach it)")
        return Res(int(rc_file.read_text().strip()), out_file.read_text(), (d / "harness.err").read_text())


class Res:
    def __init__(self, rc, out, err):
        self.returncode, self.stdout, self.stderr = rc, out, err


# ----------------------------------------------------------------------------------------------- the cases
def real_geoms():
    return [  # (op, name, rows, cols, rows_odd, cols_odd)
        (GU, "tp1", 1280, 2560, None, None),
        (GU, "tp2 384/256", 768, 2560, 512, None),
        (DN, "tp1", 2560, 640, None, None),
        (DN, "tp2 384/256", 2560, 384, None, 256),
    ]


def experts_for(rng, op, rows, cols, ro, co, n):
    geom = [(rows, cols), (ro or rows, co or cols)]
    return [X.random_expert(rng, *geom[e % 2], 4, "mul1", op == GU) for e in range(n)]


def two_class_controls(args, rng, work):
    ok = True
    for c in (R.Case("pf gate_up two classes 3:2", GU, rng, M=24, top_k=4, n_expert=6, rows=768, cols=256,
                     rows_odd=512, drop=0.1, n_protect=2),
              R.Case("pf down two classes 3:2", DN, rng, M=24, top_k=4, n_expert=6, rows=256, cols=384, cols_odd=256,
                     drop=0.1, n_protect=2)):
        good, y = R.check_case(args, c, work / label(c.name))
        ok &= good
        if y is not None:
            ok &= R.controls(args, c, y, work / (label(c.name) + "_ctl"))
    return ok


def real_band(args, rng, work):
    """The real per-rank geometry across the band (32 experts: the prefill form from M = 17 on)."""
    ok = True
    for op, name, rows, cols, ro, co in real_geoms():
        ex = experts_for(rng, op, rows, cols, ro, co, 32)
        for M in (17, 32, 64, 512):
            c = R.Case(f"{op[9:]} real {name} M={M}", op, rng, M=M, top_k=10, n_expert=32, rows=rows, cols=cols,
                       rows_odd=ro, cols_odd=co, drop=0.02, n_protect=2 if M == 64 else 0, experts=ex)
            ok &= R.check_case(args, c, work / label(c.name))[0]
    # Over 128 experts at tp2: M = 17 is T = 170 <= 192 (the decode kernels), M = 24 is T = 240 (the prefill form).
    for op, name, rows, cols, ro, co in real_geoms()[1::2]:
        ex = experts_for(rng, op, rows, cols, ro, co, 128)
        for M in (17, 24):
            c = R.Case(f"{op[9:]} real {name} 128 experts M={M}", op, rng, M=M, top_k=10, n_expert=128, rows=rows,
                       cols=cols, rows_odd=ro, cols_odd=co, drop=0.02, n_protect=2 if M == 24 else 0, experts=ex)
            ok &= R.check_case(args, c, work / label(c.name))[0]
    return ok


def band_tops(args, rng, work):
    """The band top (M = 4096: 40,960 routed rows) and the production chunk (M = 2048), 64 experts."""
    ok = True
    for (op, name, rows, cols, ro, co), M in zip(real_geoms(), (4096, 2048, 2048, 4096)):
        ex = experts_for(rng, op, rows, cols, ro, co, 64)
        c = R.Case(f"{op[9:]} real {name} M={M}", op, rng, M=M, top_k=10, n_expert=64, rows=rows, cols=cols,
                   rows_odd=ro, cols_odd=co, drop=0.02, experts=ex)
        ok &= R.check_case(args, c, work / label(c.name))[0]
    return ok


def run_form(args, rng, work):
    """The run form (no map: one workgroup an expert's whole run, prefill.c run_form) at the slots per expert a decode
    step at C = 5..8 with MTP depth 3 has on the real model (T = 170, 320 over 256 experts: 0.7 and 1.25 rows an
    expert), tp2 geometry; a run longer than 16 * MTX rows (all slots on 3 experts) takes several passes."""
    ok = True
    for op, name, rows, cols, ro, co in real_geoms()[1::2]:
        ex = experts_for(rng, op, rows, cols, ro, co, 256)
        for M in (17, 32):
            c = R.Case(f"{op[9:]} run form {name} 256 experts M={M}", op, rng, M=M, top_k=10, n_expert=256, rows=rows,
                       cols=cols, rows_odd=ro, cols_odd=co, drop=0.02, n_protect=2 if M == 32 else 0, experts=ex)
            ok &= R.check_case(args, c, work / label(c.name))[0]
        c = R.Case(f"{op[9:]} run form {name} long runs", op, rng, M=40, top_k=3, n_expert=256, rows=rows, cols=cols,
                   rows_odd=ro, cols_odd=co, experts=ex)
        ids = np.stack([rng.permutation(3) for _ in range(40)]).astype(np.int32)   # experts 0, 1, 2: 40 rows each
        c.ids = ids
        c.sorted, c.off, _ = X.moe_scatter(ids, 256)
        ok &= R.check_case(args, c, work / label(c.name))[0]
    return ok


def odd_runs(args, rng, work):
    """Run tables a router rarely or never writes, but legal: one expert owning every row (a run of T rows: many map
    entries), a skewed routing (a few hot experts), rows before off[0] (zeroed: no run owns them)."""
    ok = True
    for op, M, top_k, rows, cols in ((GU, 50, 4, 768, 2560), (DN, 50, 4, 2560, 384)):
        c = R.Case(f"{op[9:]} one run of {M * top_k} rows", op, rng, M=M, top_k=top_k, n_expert=8, rows=rows,
                   cols=cols)
        c.sorted = np.arange(c.T, dtype=np.int32)
        c.off = np.full(c.n_expert + 1, c.T, np.int32)
        c.off[0] = 0
        ok &= R.check_case(args, c, work / label(c.name))[0]
    for op, rows, cols, ro, co in ((GU, 768, 2560, 512, None), (DN, 2560, 384, None, 256)):
        c = R.Case(f"{op[9:]} skewed routing M=300", op, rng, M=300, top_k=10, n_expert=40, rows=rows, cols=cols,
                   rows_odd=ro, cols_odd=co)
        p = np.r_[np.full(4, 30.0), np.ones(36)]   # four hot experts take ~3/4 of the slots
        ids = np.stack([rng.choice(40, 10, replace=False, p=p / p.sum()) for _ in range(300)]).astype(np.int32)
        c.ids = ids
        c.sorted, c.off, _ = X.moe_scatter(ids, 40)
        ok &= R.check_case(args, c, work / label(c.name))[0]
    for op, rows, cols in ((GU, 512, 256), (DN, 256, 256)):
        c = R.Case(f"{op[9:]} rows before off[0]", op, rng, M=40, top_k=4, n_expert=6, rows=rows, cols=cols)
        live = int(c.off[c.n_expert])
        shift = 5
        c.sorted = np.r_[np.full(shift, -1, np.int32), c.sorted[: c.T - shift]].astype(np.int32)
        c.off = np.minimum(c.off + shift, c.T).astype(np.int32)
        c.off[c.n_expert] = min(live + shift, c.T)
        ok &= R.check_case(args, c, work / label(c.name))[0]
    for M in (32, 200):
        c = R.Case(f"gate_up bf16 a M={M}", GU, rng, M=M, top_k=10, n_expert=16, rows=768, cols=2560,
                   rows_odd=512, a_dtype="bf16")
        ok &= R.check_case(args, c, work / label(c.name))[0]
    for op, rows, cols in ((GU, 1280, 2560), (DN, 2560, 640)):
        c = R.Case(f"{op[9:]} odd class absent M=40", op, rng, M=40, top_k=10, n_expert=16, rows=rows, cols=cols,
                   two=False)
        ok &= R.check_case(args, c, work / label(c.name))[0]
    return ok


def replay(args, rng, work):
    ok = True
    for op, M, rows, cols, ro, co in ((GU, 512, 768, 2560, 512, None), (DN, 512, 2560, 384, None, 256)):
        c = R.Case(f"replay {op[9:]} M={M}", op, rng, M=M, top_k=10, n_expert=32, rows=rows, cols=cols,
                   rows_odd=ro, cols_odd=co, drop=0.05)
        outs = [R.run(args, c, work / label(f"{c.name} run{i}"))[1] for i in range(3)]
        same = all(o is not None for o in outs) and all(np.array_equal(outs[0], o) for o in outs[1:])
        ok &= R.verdict(same, f"{c.name}: three runs byte-identical")
    return ok


def dedup(work):
    """Hard-link identical operand files (the same expert set at several M): the case files of one group then take
    their unique bytes on disk, not one copy a case. A rewrite in check mode writes the same bytes through the link."""
    import hashlib
    import os
    seen = {}
    for f in sorted(work.rglob("*.bin")):
        if f.name.startswith("out_"):
            continue
        h = (f.stat().st_size, hashlib.sha256(f.read_bytes()).hexdigest())
        if h in seen:
            f.unlink()
            os.link(seen[h], f)
        else:
            seen[h] = f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=("plan", "check"), required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--tol", type=float, default=4e-3, help="relative L2 (PLAN.md D9's 4e-3)")
    ap.add_argument("--group", choices=("main", "tops", "run"), default="main",
                    help="main: everything but the band tops and the run form; tops: M = 2048 / 4096; run: the run form "
                         "over 256 experts (each group run apart: up to ~0.7 GB of case files)")
    args = ap.parse_args()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    shim = Replay(args.mode)
    R.subprocess = shim
    if args.mode == "plan":   # nothing is judged while planning: every launch is answered with zeros
        R.verdict = lambda ok, name, detail="": True
        R.skip = lambda name, why: None
        R.check_shapes = lambda case, rep, label: True
    args.harness, args.plugin, args.domain = HARNESS, "rowpick.so", "device"
    args.probes = args.golden = None
    R.TOL = args.tol
    rng = np.random.default_rng(20261012)
    if args.group == "tops":
        ok = band_tops(args, rng, work)
    elif args.group == "run":
        ok = run_form(args, rng, work)
    else:
        ok = two_class_controls(args, rng, work)
        ok &= real_band(args, rng, work)
        ok &= odd_runs(args, rng, work)
        ok &= replay(args, rng, work)
    if args.mode == "plan":
        dedup(work)
        (work / "runs.txt").write_text("\n".join(shim.runs) + "\n")
        print(f"planned {len(shim.runs)} harness runs in {work / 'runs.txt'}")
        return
    res = R.RESULTS
    print(f"\nprefill {args.group}: {res['pass']} passed, {res['fail']} failed, {res['skip']} skipped")
    if res["pass"] + res["fail"] == 0:
        sys.exit(77)
    sys.exit(0 if ok and res["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
