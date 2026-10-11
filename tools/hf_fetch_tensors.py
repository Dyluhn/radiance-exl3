#!/usr/bin/env python3
"""Fetch named tensors from a Hugging Face safetensors repo with HTTP Range requests, without
downloading the shards they sit in.

    hf_fetch_tensors.py REPO TENSOR [TENSOR ...] --out DIR [--revision REV]

For each tensor: the repo's model.safetensors.index.json names its shard; the shard's 8-byte
header length and JSON header are read with two Range requests; the tensor's byte range is then
streamed into DIR/<shard name>. Nothing is held in memory beyond one 8 MiB block.

WHAT IS VERIFIED, AND HOW STRONGLY:
  - always: the header's data_offsets span equals prod(shape) * itemsize(dtype); every Range
    response is 206 with exactly the bytes asked for; the shard's total size (Content-Range)
    equals the repo listing's size for that file.
  - when the requested tensors cover the WHOLE shard: the file written is the shard itself,
    byte for byte (its own header kept), and its sha256 must equal the listing's LFS sha256.
    Anything else is a subset file with a rebuilt header; it is size-verified only, and the
    record says so ("verified": "size").
The revision is resolved to a commit sha once and every request names that sha, so a branch that
moves mid-fetch cannot mix two commits. The provenance -- repo, sha, shard,
ranges, sizes, sha256 and the verification level beside each file, as DIR/<shard>.fetch.json.

Auth: HF_TOKEN from the environment, else $HF_HOME/token (default ~/.cache/huggingface/token),
else anonymous. Retries only transient network errors (3 attempts, backoff); a size or hash
mismatch is never retried.
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
BLOCK = 8 << 20
TIMEOUT_S = 60
ITEMSIZE = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "I16": 2,
            "F16": 2, "BF16": 2, "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}


def token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    path = home / "token"
    return path.read_text().strip() if path.is_file() else None


def request(url: str, headers: dict, attempts: int = 3):
    """urlopen with a timeout, retrying only connection-level failures and 5xx."""
    tok = token()
    if tok:
        headers = {**headers, "Authorization": f"Bearer {tok}"}
    for i in range(attempts):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT_S)
        except urllib.error.HTTPError as e:
            if e.code < 500 or i == attempts - 1:
                raise SystemExit(f"GET {url}: HTTP {e.code} {e.reason}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if i == attempts - 1:
                raise SystemExit(f"GET {url}: {e}")
        time.sleep(2 ** i)


def get_json(url: str):
    with request(url, {}) as r:
        return json.loads(r.read())


def resolve_url(repo: str, sha: str, path: str) -> str:
    return f"{HF}/{repo}/resolve/{sha}/{path}"


def get_range(url: str, start: int, end: int) -> tuple[bytes, int]:
    """Bytes [start, end) and the file's total size from Content-Range."""
    with request(url, {"Range": f"bytes={start}-{end - 1}"}) as r:
        body = r.read()
        total = check_range(r, start, end, len(body))
    return body, total


def check_range(r, start: int, end: int, got: int) -> int:
    if r.status != 206:
        raise SystemExit(f"{r.url}: expected 206 Partial Content for bytes {start}-{end - 1}, got {r.status}")
    cr = r.headers.get("Content-Range", "")
    want = f"bytes {start}-{end - 1}/"
    if not cr.startswith(want):
        raise SystemExit(f"{r.url}: Content-Range {cr!r}, expected {want}<total>")
    if got != end - start:
        raise SystemExit(f"{r.url}: got {got} bytes for a {end - start}-byte range")
    return int(cr[len(want):])


def read_header(url: str) -> tuple[bytes, dict, int]:
    """The shard's raw header bytes (length prefix included), parsed, and the shard's total size."""
    prefix, total = get_range(url, 0, 8)
    (n,) = struct.unpack("<Q", prefix)
    raw, _ = get_range(url, 8, 8 + n)
    return prefix + raw, json.loads(raw), total


def tensor_span(name: str, entry: dict) -> tuple[int, int]:
    begin, end = entry["data_offsets"]
    if entry["dtype"] not in ITEMSIZE:
        raise SystemExit(f"{name}: unknown dtype {entry['dtype']}")
    want = math.prod(entry["shape"]) * ITEMSIZE[entry["dtype"]]
    if end - begin != want:
        raise SystemExit(f"{name}: header span {end - begin} B != shape {entry['shape']} x {entry['dtype']} = {want} B")
    return begin, end


def stream_range(url: str, start: int, end: int, sink, sha) -> None:
    """Copy bytes [start, end) of url into the open file `sink`, feeding `sha`, with progress."""
    with request(url, {"Range": f"bytes={start}-{end - 1}"}) as r:
        got, t0, last = 0, time.time(), 0.0
        while chunk := r.read(BLOCK):
            sink.write(chunk)
            sha.update(chunk)
            got += len(chunk)
            if time.time() - last > 10:
                last = time.time()
                rate = got / max(last - t0, 1e-9) / 2**20
                print(f"  {got / 2**20:.0f} / {(end - start) / 2**20:.0f} MiB, {rate:.0f} MiB/s", flush=True)
        check_range(r, start, end, got)


def fetch_shard(repo: str, sha: str, shard: str, names: list, listing: dict, out: Path) -> dict:
    url = resolve_url(repo, sha, shard)
    header_bytes, header, total = read_header(url)
    if total != listing["size"]:
        raise SystemExit(f"{shard}: Content-Range total {total} B != repo listing {listing['size']} B")
    entries = {k: v for k, v in header.items() if k != "__metadata__"}
    missing = [n for n in names if n not in entries]
    if missing:
        raise SystemExit(f"{shard}: header has no {missing} (the index names this shard for them)")
    spans = {n: tensor_span(n, entries[n]) for n in sorted(names, key=lambda n: entries[n]["data_offsets"][0])}
    data0 = len(header_bytes)
    whole = set(names) == set(entries) and data0 + max(e for _, e in spans.values()) == total
    if whole:  # the shard itself: its own header, every byte checked against the LFS sha256
        head_out = header_bytes
    else:      # a subset: offsets renumbered in source order, provenance in __metadata__
        new, off = {}, 0
        for n, (b, e) in spans.items():
            new[n] = {"dtype": entries[n]["dtype"], "shape": entries[n]["shape"], "data_offsets": [off, off + e - b]}
            off += e - b
        new["__metadata__"] = {"format": "pt", "source": f"{repo}@{sha}/{shard}", "subset": "true"}
        raw = json.dumps(new, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        head_out = struct.pack("<Q", len(raw)) + raw
    dest = out / shard
    tmp = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256(head_out)
    with open(tmp, "wb") as f:
        f.write(head_out)
        for n, (b, e) in spans.items():
            print(f"{shard}: {n} {entries[n]['dtype']} {entries[n]['shape']} bytes {data0 + b}-{data0 + e - 1}", flush=True)
            stream_range(url, data0 + b, data0 + e, f, digest)
        f.flush()
        os.fsync(f.fileno())
    sha256 = digest.hexdigest()
    lfs = (listing.get("lfs") or {}).get("oid")
    if whole:
        if sha256 != lfs:
            raise SystemExit(f"{shard}: sha256 {sha256} != repo LFS sha256 {lfs}; kept {tmp} for inspection")
    os.replace(tmp, dest)
    return {"file": shard, "source_size": total, "tensors": {n: {"dtype": entries[n]["dtype"], "shape": entries[n]["shape"],
            "source_bytes": [data0 + b, data0 + e]} for n, (b, e) in spans.items()},
            "written_bytes": dest.stat().st_size, "sha256": sha256,
            "verified": "sha256 = repo LFS sha256 (whole shard)" if whole else "size"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("tensors", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sha = get_json(f"{HF}/api/models/{args.repo}/revision/{args.revision}")["sha"]
    listing = {e["path"]: e for e in get_json(f"{HF}/api/models/{args.repo}/tree/{sha}?recursive=1") if e["type"] == "file"}
    index = get_json(resolve_url(args.repo, sha, "model.safetensors.index.json"))["weight_map"]
    unknown = [t for t in args.tensors if t not in index]
    if unknown:
        raise SystemExit(f"{args.repo}@{sha}: the index names no {unknown}")
    by_shard = {}
    for t in args.tensors:
        by_shard.setdefault(index[t], []).append(t)
    print(f"{args.repo}@{sha}: {len(args.tensors)} tensor(s) in {len(by_shard)} shard(s) -> {args.out}", flush=True)
    for shard, names in by_shard.items():
        if shard not in listing:
            raise SystemExit(f"{args.repo}@{sha}: the index names shard {shard}, the repo listing does not have it")
        f = fetch_shard(args.repo, sha, shard, names, listing[shard], args.out)
        record = {"tool": "hf_fetch_tensors.py", "repo": args.repo, "revision_asked": args.revision, "sha": sha,
                  "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **f}
        (args.out / f"{shard}.fetch.json").write_text(json.dumps(record, indent=1) + "\n")
        print(f"{f['file']}: {f['written_bytes']} B, sha256 {f['sha256']}, verified: {f['verified']}")


if __name__ == "__main__":
    sys.exit(main())
