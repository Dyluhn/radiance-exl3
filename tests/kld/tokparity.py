#!/usr/bin/env python3
"""tokparity.py -- does radiance's tokeniser turn a KL record's documents back into the record's ids?

    tokparity.py RECORD_DIR TOKENIZER_JSON [--tokcheck PATH]

`--kld-ref` never re-tokenises (it reads tokens.i32), so a record made outside the engine is only
the text radiance would serve if radiance's encoder, run on each document's text the way
`--kld-corpus` runs it (control tokens recognised, nothing added in front: core/kld.cpp:234-237),
gives back exactly the stored ids. This runs radiance's own `rad-tokcheck` on every document of
RECORD_DIR/corpus.jsonl (written by tools/kld_synth.py) with the record's ids as the "huggingface"
side, and exits non-zero on any difference.

NEGATIVE CONTROL, every run: one document is checked again with its last id changed; rad-tokcheck
must report a difference. If it does not, the comparison is not comparing, and this exits non-zero.
--tokcheck defaults to `rad-tokcheck` on PATH.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def tokcheck(exe: str, tokenizer: Path, txt: Path, hf: Path, files: list) -> subprocess.CompletedProcess:
    cmd = [exe, "--hf-dir", str(hf), "--turns", "0", "--repeat", "1", str(tokenizer), *[str(txt / f) for f in files]]
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=600)   # it prints token bytes, not always whole UTF-8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", type=Path)
    ap.add_argument("tokenizer", type=Path)
    ap.add_argument("--tokcheck", default="rad-tokcheck")
    args = ap.parse_args()
    exe = shutil.which(args.tokcheck)
    if not exe:
        raise SystemExit(f"rad-tokcheck not found ({args.tokcheck!r}); pass --tokcheck PATH to a radiance install's bin/rad-tokcheck")

    docs = json.loads((args.record / "kld.json").read_text())["docs"]
    prompts = [json.loads(line)["prompt"] for line in (args.record / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    tokens = np.fromfile(args.record / "tokens.i32", dtype="<i4")
    if len(prompts) != len(docs) or tokens.size != sum(d["tokens"] for d in docs):
        raise SystemExit(f"{args.record}: corpus.jsonl ({len(prompts)} docs), kld.json ({len(docs)}) and "
                         f"tokens.i32 ({tokens.size} ids) do not describe the same documents")

    with tempfile.TemporaryDirectory() as tmp:
        txt, hf = Path(tmp, "txt"), Path(tmp, "hf")
        txt.mkdir()
        hf.mkdir()
        files, off = [], 0
        for i, (d, text) in enumerate(zip(docs, prompts)):
            name = f"{i:03d}-{d.get('id', 'doc')}.txt"
            (txt / name).write_bytes(text.encode("utf-8"))
            tokens[off:off + d["tokens"]].tofile(hf / f"{name}.ids")
            files.append(name)
            off += d["tokens"]
        real = tokcheck(exe, args.tokenizer, txt, hf, files)
        summary = real.stdout.strip().splitlines()[-1] if real.stdout.strip() else "(no output)"
        print(summary)

        planted = np.fromfile(hf / f"{files[0]}.ids", dtype="<i4")
        planted[-1] = (planted[-1] + 1) % 1000
        planted.tofile(hf / f"{files[0]}.ids")
        control = tokcheck(exe, args.tokenizer, txt, hf, files[:1])

    if real.returncode != 0:
        print(real.stdout + real.stderr, file=sys.stderr)
        print(f"FAIL: radiance's tokeniser does not reproduce the ids of {args.record} (see the differences above)", file=sys.stderr)
        return 1
    if control.returncode == 0:
        print(control.stdout + control.stderr, file=sys.stderr)
        print("FAIL: negative control -- a planted wrong id was NOT reported; the parity check is not comparing", file=sys.stderr)
        return 1
    print(f"PASS: {len(docs)} documents, {tokens.size} ids identical; negative control caught the planted id")
    return 0


if __name__ == "__main__":
    sys.exit(main())
