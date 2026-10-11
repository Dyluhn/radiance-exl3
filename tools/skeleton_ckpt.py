#!/usr/bin/env python3
"""skeleton_ckpt.py -- a safetensors checkpoint directory whose every tensor is a hole: the real shards'
headers, byte for byte, and no data. rad-convert reads it as the checkpoint; it costs ~0 disk.

    skeleton_ckpt.py SRC OUT [--repo REPO --revision REV]
    skeleton_ckpt.py SRC --check-whole        (writes nothing: are SRC's shards all there and whole?)

SRC holds the checkpoint's small files: config.json, model.safetensors.index.json, the tokenizer files.
For every shard the index names, OUT/<shard> is that shard's header followed by a hole as long as its
data region -- a sparse file whose apparent size is the real shard's. The header is read from SRC/<shard>
when that file is there (it must be the whole shard), else from the Hub by two HTTP Range requests
(--repo/--revision; tools/hf_fetch_tensors.py, which also honours HF_ENDPOINT and HF_TOKEN). Every other
top-level file of SRC (config, index, tokenizer, templates) is copied; hidden entries and *.safetensors
side files are not. The config is copied unchanged: the quantisation the convert selects is the
convert's to say (`rad-convert --set quantization_config.quant_method=...`, docs/STEP0.md S0.4).

WHY HOLES ARE ENOUGH (docs/STEP0.md S0.5, measured at radiance 1.3.0): `rad-convert --reuse` copies every
weight the reuse container holds at the same encoding, quantiser, options and shape without reading the
checkpoint, and an import quantiser (PLAN.md D2: `exl3`) ignores the rows it is handed. Anything that does
read the data reads zeros -- a convert's error report over a hole measures nothing, and says so.

CHECKS (each refuses by name; nothing is written for a shard that fails): every name the index gives a
shard is in that shard's header; every header span equals prod(shape) x itemsize and the spans tile the
data region without overlap; a local shard is exactly header + data long; a fetched header's shard is
exactly header + data long (the Hub's Content-Range total) and the Hub's index equals SRC's. OUT must not
exist, or must be a skeleton this tool wrote (OUT/skeleton.json); re-running over one is idempotent.

OUT/skeleton.json records the source (SRC's name, the repo and resolved commit when fetched) and each
shard's header sha256 and size. Exit 0 with a summary line: apparent bytes vs bytes on disk.
"""
import argparse
import hashlib
import json
import math
import shutil
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_fetch_tensors as hf  # noqa: E402

INDEX = "model.safetensors.index.json"
MARK = "skeleton.json"


def fail(msg: str):
    raise SystemExit(f"skeleton_ckpt: {msg}")


def parse_header(raw: bytes, where: str) -> tuple[dict, int]:
    """(tensor entries, data-region length) from a shard's header bytes, length prefix included."""
    (n,) = struct.unpack("<Q", raw[:8])
    if len(raw) != 8 + n:
        fail(f"{where}: header length prefix says {n} B, have {len(raw) - 8}")
    try:
        header = json.loads(raw[8:])
    except json.JSONDecodeError as e:
        fail(f"{where}: header is not JSON: {e}")
    entries = {k: v for k, v in header.items() if k != "__metadata__"}
    spans = []
    for name, e in entries.items():
        if e.get("dtype") not in hf.ITEMSIZE:
            fail(f"{where}: {name} has dtype {e.get('dtype')!r}, not a safetensors dtype")
        begin, end = e["data_offsets"]
        want = math.prod(e["shape"]) * hf.ITEMSIZE[e["dtype"]]
        if end - begin != want:
            fail(f"{where}: {name} spans {end - begin} B; shape {e['shape']} x {e['dtype']} is {want} B")
        spans.append((begin, end, name))
    at = 0
    for begin, end, name in sorted(spans):
        if begin != at:
            fail(f"{where}: {name} starts at data offset {begin}, the tensor before it ends at {at}")
        at = end
    return entries, at


def local_header(path: Path) -> bytes:
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            fail(f"{path}: shorter than a safetensors length prefix")
        (n,) = struct.unpack("<Q", prefix)
        return prefix + f.read(n)


def hub_header(repo: str, sha: str, shard: str) -> tuple[bytes, int]:
    """The shard's header bytes and the shard's total size, by two Range requests."""
    raw, _, total = hf.read_header(hf.resolve_url(repo, sha, shard))
    return raw, total


def read_src(src: Path) -> dict:
    for f in ("config.json", INDEX):
        if not (src / f).is_file():
            fail(f"{src} has no {f}; SRC is a checkpoint directory with its config and index")
    weight_map = json.loads((src / INDEX).read_text())["weight_map"]
    bad = sorted({s for s in weight_map.values() if "/" in s or not s.endswith(".safetensors")})
    if bad:
        fail(f"{src / INDEX} names shards that are not .safetensors files of the directory: {bad[:3]}")
    return weight_map


def check_out(out: Path, src: Path) -> None:
    if out.resolve() == src.resolve():
        fail(f"OUT is SRC ({src}); the skeleton is written beside the checkpoint, never over it")
    if out.exists() and any(out.iterdir()) and not (out / MARK).is_file():
        fail(f"{out} exists and is not a skeleton this tool wrote (no {MARK}); refusing to write into it")


def small_files(src: Path) -> list[Path]:
    return sorted(p for p in src.iterdir()
                  if p.is_file() and not p.name.startswith(".")
                  and (p.name == INDEX or ".safetensors" not in p.name))


def write_shard(dest: Path, raw: bytes, size: int) -> None:
    """The header, then a hole up to `size`: truncate() extends a file without allocating blocks."""
    with open(dest, "wb") as f:
        f.write(raw)
        f.truncate(size)


def build(src: Path, out: Path, repo: str | None, revision: str | None) -> dict:
    weight_map = read_src(src)
    check_out(out, src)
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)
    absent = sorted(s for s in by_shard if not (src / s).is_file())
    sha = None
    if absent:
        if not (repo and revision):
            fail(f"{len(absent)} shard(s) are not in {src} (e.g. {absent[0]}); pass --repo and --revision "
                 f"to read their headers from the Hub")
        sha = hf.get_json(f"{hf.HF}/api/models/{repo}/revision/{revision}")["sha"]
        hub_map = hf.get_json(hf.resolve_url(repo, sha, INDEX))["weight_map"]
        if hub_map != weight_map:
            fail(f"{repo}@{sha}'s {INDEX} differs from {src / INDEX}; SRC is another revision")

    out.mkdir(parents=True, exist_ok=True)
    record = {"tool": "skeleton_ckpt.py", "source": src.name, "repo": repo if absent else None,
              "revision": sha, "shards": {}}
    for i, shard in enumerate(sorted(by_shard), 1):
        if (src / shard).is_file():
            raw, origin = local_header(src / shard), "local"
            total = (src / shard).stat().st_size
        else:
            raw, total = hub_header(repo, sha, shard)
            origin = "hub"
        entries, data = parse_header(raw, f"{shard} ({origin})")
        missing = sorted(set(by_shard[shard]) - set(entries))
        if missing:
            fail(f"the index puts {missing[0]} (and {len(missing) - 1} more) in {shard}; its header does not "
                 f"have it")
        if total != len(raw) + data:
            fail(f"{shard} ({origin}) is {total} B; its header says {len(raw)} + {data} B")
        write_shard(out / shard, raw, total)
        record["shards"][shard] = {"header_sha256": hashlib.sha256(raw).hexdigest(), "header_bytes": len(raw),
                                   "size": total, "tensors": len(entries), "header_from": origin}
        print(f"[{i}/{len(by_shard)}] {shard}: {len(entries)} tensors, header {len(raw)} B ({origin}), "
              f"{total} B apparent", flush=True)
    for p in small_files(src):
        shutil.copyfile(p, out / p.name)
    (out / MARK).write_text(json.dumps(record, indent=1) + "\n")
    return record


def check_whole(src: Path) -> int:
    """The opposite check, for a checkpoint that is READ (P6's EXL3 source): every shard its index names is
    there, its header parses, every name is in it, and the file is header + data long and not sparse (a
    partial download). Returns the bytes checked; refuses by name."""
    weight_map = read_src(src)
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)
    total = 0
    for shard in sorted(by_shard):
        p = src / shard
        if not p.is_file():
            fail(f"{src} has no {shard} (its index names {len(by_shard[shard])} tensors there)")
        raw = local_header(p)
        entries, data = parse_header(raw, shard)
        missing = sorted(set(by_shard[shard]) - set(entries))
        if missing:
            fail(f"the index puts {missing[0]} (and {len(missing) - 1} more) in {shard}; its header does not have it")
        st = p.stat()
        if st.st_size != len(raw) + data:
            fail(f"{shard} is {st.st_size} B; its header says {len(raw)} + {data} B")
        if st.st_blocks * 512 < 0.99 * st.st_size:
            fail(f"{shard} holds {st.st_blocks * 512} B on disk of {st.st_size}: sparse, so not whole")
        total += st.st_size
    return total


def disk_usage(out: Path) -> tuple[int, int]:
    """(apparent bytes, bytes on disk) of the skeleton's shards."""
    shards = [p.stat() for p in out.glob("*.safetensors")]
    return sum(s.st_size for s in shards), sum(s.st_blocks * 512 for s in shards)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path, help="checkpoint dir: config.json, index, tokenizer (shards optional)")
    ap.add_argument("out", type=Path, nargs="?", help="the skeleton dir to write")
    ap.add_argument("--repo", help="Hub repo id to read absent shards' headers from")
    ap.add_argument("--revision", help="its revision (resolved to a commit once; every request names it)")
    ap.add_argument("--check-whole", action="store_true",
                    help="write nothing: check that SRC's shards are all there, whole and not sparse")
    a = ap.parse_args(argv)
    if not a.src.is_dir():
        fail(f"SRC {a.src} is not a directory")
    if a.check_whole:
        print(f"skeleton_ckpt: OK {a.src} is whole: {check_whole(a.src)} B of shards")
        return 0
    if a.out is None:
        fail("OUT is required (or --check-whole)")
    build(a.src, a.out, a.repo, a.revision)
    apparent, on_disk = disk_usage(a.out)
    print(f"skeleton_ckpt: OK {a.out}: shards {apparent} B apparent, {on_disk} B on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
