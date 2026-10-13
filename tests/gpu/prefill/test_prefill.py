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
  - group `run`: the run form (no map) at T = 170 / 320 over 256 experts, tp2, and runs of 40 rows (several passes);
  - group `hrot`: interface v3 ("EPI 1"), two GPU stages (plan, run, plan2, run, check): the pair at h_rot 1 == the
    pair at h_rot 0 to the bit, device against device -- the run form, the map form, tp1 and tp2.
--bits K (R23): every case of the group at K = 2, 3, 5 or 6 (mul1) instead of 4, the same checks and controls.
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
STAGE = 1                 # 2 in plan2 / check: the cases whose inputs are stage 1's device outputs


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
        if self.mode == "plan" or (self.mode == "plan2" and not (d / "harness.rc").exists()):
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


BITS = 4   # the codes' K of every case (--bits, R23)


def experts_for(rng, op, rows, cols, ro, co, n):
    geom = [(rows, cols), (ro or rows, co or cols)]
    return [X.random_expert(rng, *geom[e % 2], BITS, "mul1", op == GU) for e in range(n)]


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


def tp_band(args, rng, work):
    """Every rank's geometry at THREE and FOUR ranks (PLAN §0 G3; the box has two cards): even / odd widths of 256 /
    128, 256 / 256, 128 / 256 and 128 / 128 columns (the expert's 5 blocks split by parity, qwen4exp_exl3.cpp) --
    gate_up's narrowest class N = 256 and down's K = 128, which no tp1 / tp2 case reaches -- and 384 / 384, tp2's rank
    taking the extra block of both parities (QWEN4EXP_EXL3_EXTRA_RANK), across the band, 32 experts, through the
    production chunk."""
    ok = True
    for we, wo in ((256, 128), (256, 256), (128, 256), (128, 128), (384, 384)):
        for op, rows, cols, ro, co in ((GU, 2 * we, 2560, 2 * wo, None), (DN, 2560, we, None, wo)):
            ex = experts_for(rng, op, rows, cols, ro, co, 32)
            for M in (17, 64, 512, 2048):
                c = R.Case(f"{op[9:]} tp widths {we}/{wo} M={M}", op, rng, M=M, top_k=10, n_expert=32, rows=rows,
                           cols=cols, rows_odd=ro, cols_odd=co, drop=0.02, n_protect=2 if M == 64 else 0, experts=ex)
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


def hrot_chain(args, gu, dn_experts, d, label):
    """Interface v3 ("EPI 1") on the M > 16 rows. THE claim: the pair at h_rot 1 == the pair at h_rot 0, to the bit,
    device against device. Down's inputs are gate_up's DEVICE outputs, so this runs in two GPU stages (--mode plan,
    the launches, --mode plan2, the launches, --mode check):
      stage 1  gate_up at h_rot 0 and at h_rot 1, each against the transcription; h1 against NumPy's step 1 of h0:
               at most 1 f16 ulp on under 0.01% of the elements (measured: 1-81 of 0.04-10 M; the device's rotate128
               is not NumPy's to the bit -- a finding for the shared rotate128_math; numerically nil);
      stage 2  down at h_rot 0 on the device's h0 and at h_rot 1 on the device's h1: equal TO THE BIT (the pair),
               each against the transcription; control: down told h_rot 1 about the UNrotated h0 is caught."""
    ok = True
    gu.params["h_rot"] = 0
    ok0, h0 = R.check_case(args, gu, d / "gate_up0", f"{label}: gate_up h_rot 0")
    gu.params["h_rot"] = 1
    ok1, h1 = R.check_case(args, gu, d / "gate_up1", f"{label}: gate_up h_rot 1")
    gu.params["h_rot"] = 0
    ok &= ok0 and ok1
    if h0 is None or h1 is None:
        return False

    step1 = np.zeros_like(h0)
    for i in range(gu.T):
        if i >= gu.off[gu.n_expert] or gu.sorted[i] < 0:
            continue
        e = X.owner(gu.off, i)
        W = gu.experts[e].rows // 2
        row = h0[i:i + 1, :W].view(np.float16).astype(np.float32)
        step1[i, :W] = X.to_f16(X._prologue(row, gu.experts[e].dn_suh)).view(np.uint16)[0]
    diff = h1 != step1
    ulp = np.abs(R.ordered(h1).astype(np.int64) - R.ordered(step1).astype(np.int64))
    frac = float(diff.mean())
    ok &= R.verdict(int(ulp.max()) <= 1 and frac < 1e-4,
                    f"{label}: gate_up h_rot 1 vs NumPy's step 1 of its h_rot 0: at most 1 f16 ulp on < 0.01% of elements",
                    f"{int(diff.sum())} differ ({frac:.2e}), max {int(ulp.max())} ulp")
    ok &= R.verdict(not np.array_equal(h1, h0), f"{label}: control, h_rot moves gate_up's output")
    if STAGE < 2:
        return ok

    def down_case(hbits, h_rot, name):
        rows_gu = [gu.geom[c][0] for c in range(2)]
        dn = R.Case(name, "moe_exl3_down", np.random.default_rng(0), M=gu.M, top_k=gu.top_k, n_expert=gu.n_expert,
                    rows=gu.geom[0][1], cols=rows_gu[0] // 2, rows_odd=gu.geom[1][1] if gu.two else None,
                    cols_odd=rows_gu[1] // 2 if gu.two else None, two=gu.two, experts=dn_experts,
                    x=hbits.view(np.float16).astype(np.float32), h_rot=h_rot)
        dn.ids, dn.sorted, dn.off = gu.ids, gu.sorted, gu.off
        return dn
    okd0, y0 = R.check_case(args, down_case(h0, 0, f"{label}: down h_rot 0 on the device's h0"), d / "down0")
    okd1, y1 = R.check_case(args, down_case(h1, 1, f"{label}: down h_rot 1 on the device's h1"), d / "down1")
    ok &= okd0 and okd1
    ok &= R.verdict(y0 is not None and y1 is not None and np.array_equal(y0, y1),
                    f"{label}: THE PAIR at h_rot 1 == the pair at h_rot 0, to the bit",
                    f"identical={y0 is not None and y1 is not None and np.array_equal(y0, y1)}")
    rc, yw, _ = R.run(args, down_case(h0, 1, f"{label}: down h_rot 1 on an unrotated h"), d / "down_wrong")
    ok &= R.verdict(rc == 0 and yw is not None and y0 is not None and not np.array_equal(yw, y0),
                    f"{label}: control, down h_rot 1 on an unrotated h is caught")
    return ok


def hrot(args, rng, work):
    """v3 EPI 1 across the band: the run form (gate_up at T <= 1.5 n_expert), the map form, the band top's form."""
    ok = True
    for name, M, n_expert, wid, wodd in (("hrot two classes 3:2 M=24", 24, 6, 384, 256),
                                         ("hrot real tp2 run form M=24 (256 experts)", 24, 256, 384, 256),
                                         ("hrot real tp2 M=512", 512, 32, 384, 256),
                                         ("hrot real tp1 M=64", 64, 32, 640, None)):
        hidden = 256 if n_expert == 6 else 2560
        top_k = 4 if n_expert == 6 else 10
        gu, dn = R.chain_case(rng, name, M=M, top_k=top_k, n_expert=n_expert, hidden=hidden, width=wid,
                              width_odd=wodd, drop=0.05, n_protect=2 if n_expert == 6 else 0)
        ok &= hrot_chain(args, gu, dn, work / label(name), name)
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
    ap.add_argument("--mode", choices=("plan", "plan2", "check"), required=True,
                    help="plan: write every case; plan2: (group hrot) read stage 1's runs, write stage 2's; check")
    ap.add_argument("--work", required=True)
    ap.add_argument("--tol", type=float, default=4e-3, help="relative L2 (PLAN.md D9's 4e-3)")
    ap.add_argument("--group", choices=("main", "tops", "run", "hrot", "quick", "tp"), default="main",
                    help="main: everything but the band tops and the run form; tops: M = 2048 / 4096; run: the run form "
                         "over 256 experts (each group run apart: up to ~0.7 GB of case files)")
    ap.add_argument("--bits", type=int, choices=(2, 3, 4, 5, 6), default=4,
                    help="R23: the codes' K of every case in the group (mul1); 4 is what the groups were written at")
    args = ap.parse_args()
    global BITS
    BITS = args.bits
    if args.bits != 4:   # every case the group builds takes this K unless it names one (none does)
        base_case, base_chain = R.Case, R.chain_case

        class KCase(base_case):
            def __init__(self, *a, **kw):
                kw.setdefault("bits", args.bits)
                super().__init__(*a, **kw)

        R.Case = KCase
        R.chain_case = lambda *a, **kw: base_chain(*a, **{"bits": args.bits, **kw})
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    global STAGE
    STAGE = 1 if args.mode == "plan" else 2
    shim = Replay(args.mode)
    R.subprocess = shim
    if args.mode in ("plan", "plan2"):   # nothing is judged while planning: every planned launch answers zeros
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
    elif args.group == "hrot":
        ok = hrot(args, rng, work)
    elif args.group == "quick":   # the pre-timing bounds subset: real geometry, both ops, tp1 + tp2, M = 17..512
        ok = real_band(args, rng, work)
    elif args.group == "tp":      # every rank's widths at three and four ranks
        ok = tp_band(args, rng, work)
    else:
        ok = two_class_controls(args, rng, work)
        ok &= real_band(args, rng, work)
        ok &= odd_runs(args, rng, work)
        ok &= replay(args, rng, work)
    if args.mode in ("plan", "plan2"):
        dedup(work)
        (work / "runs.txt").write_text("\n".join(shim.runs) + "\n")
        print(f"planned {len(shim.runs)} harness runs in {work / 'runs.txt'}")
        return
    res = R.RESULTS
    print(f"\nprefill {args.group}{'' if args.bits == 4 else f' K={args.bits}'}: {res['pass']} passed, {res['fail']} failed, "
          f"{res['skip']} skipped")
    if res["pass"] + res["fail"] == 0:
        sys.exit(77)
    sys.exit(0 if ok and res["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
