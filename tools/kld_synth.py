#!/usr/bin/env python3
"""kld_synth.py -- an independent bf16 reference (final hidden states + the bf16 lm_head) written as a
radiance KL reference record, the directory `radiance --kld-ref DIR` scores a candidate against.
The format is docs/KLD-RECORD-FORMAT.md (radiance core/kld.cpp, record_row); tests/kld/ holds this
file to radiance's own recorder and loader.

    kld_synth.py --hidden ref_final_hidden.npy --meta meta.json --docs heldout.jsonl \\
                 --lm-head model-000NN-of-000NN.safetensors --model NAME --out DIR \\
                 (--n-docs N | --doc-ids ID,ID,...) [--check-nll tokens_ref.npz] [--min-free-gib G]

INPUTS -- the outputs of a reference pass of r9v-gsq-exl3-depth/scripts/kld_sweep.py:
  --hidden     f32 [tokens, hidden]: the model's final hidden state at every position of every
               document, concatenated in meta.json order -- the output of the final hyper-connection
               mixer, whose hc_norm is the model's last RMS norm, so logits = hidden @ lm_head^T.
  --meta       meta.json: doc_ids, spans [[a, b), ...] into --hidden, tokens.
  --docs       the documents, JSONL: id, domain, text, ids (the stored tokenisation).
  --lm-head    a safetensors file holding `lm_head.weight` (bf16 [n_vocab, hidden]).
  --check-nll  the producer's per-position results (tokens_ref.npz, `nll_ref`): every row's NLL
               must reproduce it within --nll-tol, which proves hidden, head, positions and next
               tokens line up. Without it the record is written unchecked, and the log says so.

WHAT IS WRITTEN (DIR; refused if DIR already holds a kld.json, as radiance refuses):
  corpus.jsonl  the selected documents as radiance's --kld-corpus would take them:
                {"prompt": text, "score_from": 1, "source": domain}; its FNV-1a is the manifest's
  tokens.i32 logp.f16 top.bin rows.f32   the record, rows streamed in document order
  kld.json      last, via a rename, with a `synth` provenance object and each document's `id`
Every document scores positions [1, n - 1): the producer scores positions with >= 2 tokens of
context and a next token (kld_sweep.py: valid[a+1 : b-1]).

SELECTION (--n-docs N): stratified by `domain` -- each domain's share of N by largest remainder
(ties: the larger domain, then the name), at least one document per domain -- and within a domain
the SHORTEST documents first (ties: file order). Selected documents keep their file order.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

KFORMAT = "radiance-kld-1"   # core/kld.cpp:30, kFormat
TOP = 16                     # core/kld.h:65, kKldTop
SCORE_FROM = 1
CHUNK = 256                  # rows per matmul: ~1.7 GB of temporaries at a 248K vocabulary
TOP_DTYPE = np.dtype([("id", "<i4"), ("lp", "<f4")])   # struct TopEntry, kld.cpp:151


# ----------------------------------------------------------------------------- the row math (pure)

def top_entries(lp: np.ndarray) -> np.ndarray:
    """[R, TOP] TopEntry: the TOP largest lp per row, by lp descending then id ascending; among
    equal values at the cut the LOWER ids are kept (record_row's heap, kld.cpp:397-423)."""
    v = lp.shape[1]
    idx = np.argpartition(lp, v - TOP, axis=1)[:, v - TOP:]
    cut = np.take_along_axis(lp, idx, axis=1).min(axis=1)
    for r in np.flatnonzero((lp >= cut[:, None]).sum(axis=1) > TOP):   # a tie straddles the cut
        above = np.flatnonzero(lp[r] > cut[r])
        idx[r] = np.concatenate([above, np.flatnonzero(lp[r] == cut[r])[:TOP - above.size]])
    vals = np.take_along_axis(lp, idx, axis=1)
    order = np.lexsort((idx, -vals), axis=1)
    top = np.empty(lp.shape[0:1] + (TOP,), dtype=TOP_DTYPE)
    top["id"] = np.take_along_axis(idx, order, axis=1)
    top["lp"] = np.take_along_axis(vals, order, axis=1)
    return top


def record_rows(logits: np.ndarray, next_ids: np.ndarray):
    """f32 logits [R, V] and the next token of each row -> (logp f16 [R, V], top [R, TOP], rows f32
    [R, 2] = {nll, entropy}), exactly as record_row computes them (kld.cpp:384-430): lse in double,
    lp = (float)(logit - lse), f16 round-to-nearest-even, entropy over p = exp((double)lp)."""
    if logits.dtype != np.float32 or logits.ndim != 2 or logits.shape[1] < TOP:
        raise ValueError(f"logits must be f32 [rows, n_vocab >= {TOP}], got {logits.dtype} {logits.shape}")
    m = logits.max(axis=1)
    # expf of the f32 difference, correctly rounded (numpy's SIMD f32 exp is ~1 ulp off often enough
    # to move lse by ~3e-9 and flip an f16), summed in id order: then lp is radiance's to the bit.
    e = np.exp((logits - m[:, None]).astype(np.float64)).astype(np.float32)
    lse = m.astype(np.float64) + np.log(np.cumsum(e, axis=1, dtype=np.float64)[:, -1])
    del e
    lp = (logits.astype(np.float64) - lse[:, None]).astype(np.float32)
    lp64 = lp.astype(np.float64)
    ent = -(np.exp(lp64) * lp64).sum(axis=1)
    nll = lse - logits[np.arange(logits.shape[0]), next_ids].astype(np.float64)
    rows = np.stack([nll, ent], axis=1).astype("<f4")
    return lp.astype("<f2"), top_entries(lp), rows


def fnv1a64(lines: list) -> str:
    """The manifest's corpus_fnv1a64: FNV-1a 64 over each line's bytes, newlines excluded (kld.cpp:32-36, 219-222).
    Radiance's offset basis is 1469598103934665603 -- the standard one (14695981039346656037) without its
    last digit -- so published FNV test vectors do not apply; this mirrors radiance, not the standard."""
    h = 1469598103934665603
    for line in lines:
        for b in line:
            h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


# ----------------------------------------------------------------------------- selection (pure)

def select_docs(domains: list, lengths: list, n: int) -> list:
    """Indices of n documents: stratified by domain (largest remainder, >= 1 each), shortest first."""
    count = Counter(domains)
    if n < len(count) or n > len(domains):
        raise ValueError(f"--n-docs {n}: need between {len(count)} (one per domain) and {len(domains)}")
    quota = {d: n * c / len(domains) for d, c in count.items()}
    take = {d: max(1, math.floor(q)) for d, q in quota.items()}
    left = n - sum(take.values())
    if left < 0:
        raise ValueError(f"--n-docs {n}: the one-per-domain floor alone takes {sum(take.values())}")
    by_remainder = sorted(count, key=lambda d: (-(quota[d] - math.floor(quota[d])), -count[d], d))
    for d in by_remainder[:left]:
        take[d] += 1
    chosen = []
    for d in count:
        mine = sorted((i for i, x in enumerate(domains) if x == d), key=lambda i: (lengths[i], i))
        chosen += mine[:take[d]]
    return sorted(chosen)


# ----------------------------------------------------------------------------- inputs

def read_bf16(path: Path, name: str) -> np.ndarray:
    """One bf16 tensor of a safetensors file, widened to f32 exactly (the high half of an f32)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    if name not in header:
        raise SystemExit(f"{path}: no tensor {name!r} (has {sorted(k for k in header if k != '__metadata__')})")
    e = header[name]
    if e["dtype"] != "BF16" or len(e["shape"]) != 2:
        raise SystemExit(f"{path}: {name} is {e['dtype']} {e['shape']}, expected a 2-d BF16 matrix")
    begin, end = e["data_offsets"]
    raw = np.memmap(path, dtype="<u2", mode="r", offset=8 + n + begin, shape=tuple(e["shape"]))
    if end - begin != raw.nbytes:
        raise SystemExit(f"{path}: {name} spans {end - begin} B, its shape needs {raw.nbytes}")
    return (raw.astype(np.uint32) << 16).view(np.float32)


def load_docs(args) -> tuple[list, np.ndarray]:
    docs = [json.loads(line) for line in args.docs.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = json.loads(args.meta.read_text())
    hidden = np.load(args.hidden, mmap_mode="r")
    if meta["doc_ids"] != [d["id"] for d in docs]:
        raise SystemExit(f"{args.meta}: doc_ids are not {args.docs}'s documents in order")
    for d, (a, b) in zip(docs, meta["spans"]):
        if b - a != len(d["ids"]):
            raise SystemExit(f"{args.meta}: span {a}-{b} of {d['id']} != its {len(d['ids'])} ids")
        d["span"] = (a, b)
    if hidden.dtype != np.float32 or hidden.ndim != 2 or hidden.shape[0] != meta["tokens"]:
        raise SystemExit(f"{args.hidden}: {hidden.dtype} {hidden.shape}, expected f32 [{meta['tokens']}, hidden]")
    return docs, hidden


def pick(docs: list, args) -> list:
    if args.doc_ids:
        want = args.doc_ids.split(",")
        known = {d["id"]: i for i, d in enumerate(docs)}
        missing = [w for w in want if w not in known]
        if missing:
            raise SystemExit(f"--doc-ids: no such documents {missing}")
        return [docs[i] for i in sorted(known[w] for w in want)]
    return [docs[i] for i in select_docs([d["domain"] for d in docs], [len(d["ids"]) for d in docs], args.n_docs)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(16 << 20):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- the record

def check_room(out: Path, n_rows: int, n_vocab: int, min_free_gib: float) -> None:
    need = n_rows * (n_vocab * 2 + TOP * 8 + 8)
    free = shutil.disk_usage(out).free
    left = (free - need) / 2**30
    print(f"record: {n_rows} rows x {n_vocab} vocab = {need / 2**30:.2f} GiB; {free / 2**30:.1f} GiB free, "
          f"{left:.1f} GiB after", flush=True)
    if left < min_free_gib:
        raise SystemExit(f"{out}: the record would leave {left:.1f} GiB free, under --min-free-gib {min_free_gib}")


def write_record(out: Path, sel: list, hidden: np.ndarray, head: np.ndarray, nll_ref, nll_tol: float) -> dict:
    """Streams logp.f16, top.bin, rows.f32 in document order; returns the NLL check's numbers."""
    worst, total, n = 0.0, 0.0, 0
    t0, last, done = time.time(), 0.0, 0
    n_rows = sum(len(d["ids"]) - 1 - SCORE_FROM for d in sel)
    with open(out / "logp.f16", "wb") as f_lp, open(out / "top.bin", "wb") as f_top, open(out / "rows.f32", "wb") as f_rows:
        for d in sel:
            a, _ = d["span"]
            ids = np.asarray(d["ids"], dtype=np.int64)
            for p0 in range(SCORE_FROM, len(ids) - 1, CHUNK):
                p1 = min(p0 + CHUNK, len(ids) - 1)
                logits = np.ascontiguousarray(hidden[a + p0:a + p1]) @ head.T
                logp, top, rows = record_rows(logits, ids[p0 + 1:p1 + 1])
                f_lp.write(logp.tobytes())
                f_top.write(top.tobytes())
                f_rows.write(rows.tobytes())
                if nll_ref is not None:
                    diff = np.abs(rows[:, 0].astype(np.float64) - nll_ref[a + p0:a + p1])
                    worst, total, n = max(worst, float(diff.max())), total + float(diff.sum()), n + diff.size
                done += p1 - p0
                if time.time() - last > 15:
                    last = time.time()
                    print(f"record: {done} / {n_rows} rows, {done / (last - t0):.0f} rows/s", flush=True)
        for f in (f_lp, f_top, f_rows):
            f.flush()
            os.fsync(f.fileno())
    if nll_ref is None:
        print("record: NLL NOT CHECKED against the producer (no --check-nll)", flush=True)
        return {"checked": False}
    print(f"record: NLL vs producer over {n} rows: max |diff| {worst:.3g}, mean {total / n:.3g} (tol {nll_tol})", flush=True)
    if worst > nll_tol:
        raise SystemExit(f"NLL check FAILED: max |diff| {worst:.3g} > {nll_tol}; the record in {out} has no kld.json")
    return {"checked": True, "rows": n, "max_abs_diff": worst, "mean_abs_diff": total / n, "tol": nll_tol}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--lm-head", type=Path, required=True)
    ap.add_argument("--model", required=True, help="the reference's name in kld.json and every report")
    ap.add_argument("--out", type=Path, required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--n-docs", type=int)
    group.add_argument("--doc-ids", help="comma-separated document ids instead of the stratified selection")
    ap.add_argument("--check-nll", type=Path)
    ap.add_argument("--nll-tol", type=float, default=2e-3)
    ap.add_argument("--min-free-gib", type=float, default=25.0, help="refuse a record leaving less free space")
    args = ap.parse_args()

    if (args.out / "kld.json").exists():
        raise SystemExit(f"{args.out} already holds a reference (kld.json); write a new directory")
    docs, hidden = load_docs(args)
    sel = pick(docs, args)
    head = read_bf16(args.lm_head, "lm_head.weight")
    if head.shape[1] != hidden.shape[1] or head.shape[0] < TOP:
        raise SystemExit(f"lm_head {head.shape} does not fit hidden {hidden.shape}")
    nll_ref = np.load(args.check_nll)["nll_ref"] if args.check_nll else None
    if nll_ref is not None and nll_ref.shape[0] != hidden.shape[0]:
        raise SystemExit(f"{args.check_nll}: nll_ref has {nll_ref.shape[0]} positions, hidden {hidden.shape[0]}")
    n_rows = sum(len(d["ids"]) - 1 - SCORE_FROM for d in sel)
    print(f"selected {len(sel)} documents, {sum(len(d['ids']) for d in sel)} tokens, {n_rows} rows: "
          f"{dict(Counter(d['domain'] for d in sel))}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    check_room(args.out, n_rows, head.shape[0], args.min_free_gib)

    lines = [json.dumps({"prompt": d["text"], "score_from": SCORE_FROM, "source": d["domain"]},
                        ensure_ascii=False).encode("utf-8") for d in sel]
    (args.out / "corpus.jsonl").write_bytes(b"".join(line + b"\n" for line in lines))
    np.concatenate([np.asarray(d["ids"], dtype="<i4") for d in sel]).tofile(args.out / "tokens.i32")
    nll = write_record(args.out, sel, hidden, head, nll_ref, args.nll_tol)

    docs_json, row0 = [], 0
    for d in sel:
        docs_json.append({"source": d["domain"], "tokens": len(d["ids"]), "score_from": SCORE_FROM, "row0": row0, "id": d["id"]})
        row0 += len(d["ids"]) - 1 - SCORE_FROM
    manifest = {"format": KFORMAT, "model": args.model, "corpus": "corpus.jsonl", "corpus_fnv1a64": fnv1a64(lines),
                "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "n_vocab": head.shape[0],
                "top": TOP, "rows": row0, "docs": docs_json,
                "synth": {"tool": "kld_synth.py", "logits": "hidden @ bf16 lm_head^T in f32",
                          "selection": f"--doc-ids {args.doc_ids}" if args.doc_ids else f"--n-docs {args.n_docs}",
                          "hidden": {"file": args.hidden.name, "shape": list(hidden.shape)},
                          "meta_sha256": sha256_file(args.meta), "docs_sha256": sha256_file(args.docs),
                          "lm_head_sha256": sha256_file(args.lm_head), "nll_check": nll}}
    tmp = args.out / "kld.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1) + "\n")
    os.replace(tmp, args.out / "kld.json")
    print(f"record: {row0} rows over {len(sel)} documents in {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
