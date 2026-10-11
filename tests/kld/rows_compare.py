#!/usr/bin/env python3
"""rows_compare.py -- two `--kld-out` row files scored against the same reference, compared position
by position: B - A with a paired block-bootstrap 95% interval, overall, per source, and on the tail.

    rows_compare.py REF_DIR A.json.rows B.json.rows [--json OUT] [--draws 2000] [--seed 0] [--block 256]

Each .rows is [rows, 4] f32 in reference row order: kl, nll_candidate, nll_reference, top1
(radiance core/kld.cpp report()). The record's kld.json gives each document's rows and source.

THE BOOTSTRAP (the protocol Dylan's kld_sweep reports used): every document's rows are cut into
consecutive blocks of --block rows (the last one shorter), because neighbouring positions are not
independent; a draw resamples, within each source, that source's blocks with replacement, the SAME
blocks for A and B (paired); a statistic is computed on the drawn rows. 95% interval = the 2.5th and
97.5th percentiles of --draws draws. Reported: mean KL, p99, p99.9 and top-1 for A, B and B - A;
per-source mean KL B - A. Equal files are reported as identical (sha256), with every interval [0, 0].
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def load_rows(path: Path, n_rows: int) -> np.ndarray:
    rows = np.fromfile(path, dtype="<f4")
    if rows.size != n_rows * 4:
        raise SystemExit(f"{path}: {rows.size // 4} rows, the reference has {n_rows}")
    return rows.reshape(n_rows, 4).astype(np.float64)


def blocks_by_source(docs: list, block: int) -> dict:
    """{source: [np.arange of row indices per block]} over every document's rows."""
    out = {}
    for d in docs:
        n = d["tokens"] - 1 - d["score_from"]
        for b0 in range(0, n, block):
            out.setdefault(d["source"], []).append(np.arange(d["row0"] + b0, d["row0"] + min(b0 + block, n)))
    return out


def stats(kl: np.ndarray, top1: np.ndarray) -> np.ndarray:
    return np.array([kl.mean(), np.quantile(kl, 0.99), np.quantile(kl, 0.999), top1.mean()])


NAMES = ["mean", "p99", "p99_9", "top1"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", type=Path)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", type=int, default=256)
    args = ap.parse_args()

    manifest = json.loads((args.ref / "kld.json").read_text())
    a, b = load_rows(args.a, manifest["rows"]), load_rows(args.b, manifest["rows"])
    sha = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.a, args.b)]
    groups = blocks_by_source(manifest["docs"], args.block)
    rng = np.random.default_rng(args.seed)

    draws = np.empty((args.draws, 3, 4))       # [draw, A/B/B-A, stat]
    src_draws = {s: np.empty(args.draws) for s in groups}
    for i in range(args.draws):
        picked = []
        for s, blocks in groups.items():
            idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
            src_draws[s][i] = b[idx, 0].mean() - a[idx, 0].mean()
            picked.append(idx)
        idx = np.concatenate(picked)
        sa, sb = stats(a[idx, 0], a[idx, 3]), stats(b[idx, 0], b[idx, 3])
        draws[i] = [sa, sb, sb - sa]
    point = np.array([stats(a[:, 0], a[:, 3]), stats(b[:, 0], b[:, 3])])
    point = np.vstack([point, point[1] - point[0]])
    lo, hi = np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)

    report = {"ref": str(args.ref), "a": str(args.a), "b": str(args.b), "a_sha256": sha[0], "b_sha256": sha[1],
              "identical": sha[0] == sha[1], "rows": manifest["rows"], "draws": args.draws, "seed": args.seed,
              "block": args.block, "max_abs_row_diff_kl": float(np.abs(b[:, 0] - a[:, 0]).max()),
              "stats": {arm: {n: {"value": float(point[k, j]), "ci95": [float(lo[k, j]), float(hi[k, j])]}
                              for j, n in enumerate(NAMES)} for k, arm in enumerate(["a", "b", "b_minus_a"])},
              "by_source_mean_b_minus_a": {}}
    print(f"rows {manifest['rows']}, {sum(len(v) for v in groups.values())} blocks of <= {args.block}, "
          f"{args.draws} draws, seed {args.seed}; identical files: {report['identical']}")
    for j, n in enumerate(NAMES):
        print(f"  {n:6s} A {point[0, j]:.6f}  B {point[1, j]:.6f}  B-A {point[2, j]:+.6f} [{lo[2, j]:+.6f}, {hi[2, j]:+.6f}]")
    for s in sorted(groups):
        rows = np.concatenate(groups[s])
        diff = float(b[rows, 0].mean() - a[rows, 0].mean())
        ci = [float(np.percentile(src_draws[s], 2.5)), float(np.percentile(src_draws[s], 97.5))]
        report["by_source_mean_b_minus_a"][s] = {"a": float(a[rows, 0].mean()), "b": float(b[rows, 0].mean()),
                                                 "value": diff, "ci95": ci, "rows": int(rows.size)}
        print(f"  {s:14s} {rows.size:6d} rows  A {a[rows, 0].mean():.5f}  B {b[rows, 0].mean():.5f}  "
              f"B-A {diff:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}]")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
