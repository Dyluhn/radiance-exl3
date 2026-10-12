#!/usr/bin/env python3
"""pool_headroom.py seqs LOG | pool LOG -- serve flags that leave a model about half its expert plane
off the card, with the prefill stager armed, from the engine's own VRAM budget table ("VRAM budget,
per card", core/mem/vram_budget.cpp) of a RESIDENT server's log. Nothing about the card is assumed.

  seqs LOG   LOG served --max-num-seqs 1: print the --max-num-seqs whose cache pays for the stager's two
             buffers and leaves about one sequence to lend. A lending cache pays for the buffers out of
             its own share (vram_budget.cpp:217-224: kv - 2 x one >= one sequence, one = a layer's
             pooled share), and whatever of it no live request holds is lent to the expert slab, where
             the mover promotes pooled experts back onto the card -- too large a cache and nothing is
             left pooled for the stager to copy. With half the plane pooled, the buffers are about half
             the plane: ceil(experts / 2 / kv) + 2 sequences.
  pool LOG   LOG served at that --max-num-seqs: print the --gpu-headroom-mib that makes the elastic
             share the cache's plus half the expert plane plus 16 MiB (headroom + claimable - static -
             kv - experts / 2 - 16: what the driver and code objects hold is measured afresh at every
             start and moved by up to ~10 MiB between two of them, and a cache that cannot cover its
             fixed costs after the stager's buffers is refused at load), then the --host-pool-mib that holds the whole plane plus 64 MiB (the
             default pool is sized for a resident plan, measured 768 KiB; an expert with no tier is
             refused at load).
Exit 1 when the table is not in LOG, 2 on bad usage."""
import math
import re
import sys

UNIT = {"B": 1 / 1048576, "KiB": 1 / 1024, "MiB": 1, "GiB": 1024}


def table(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    def mib(label):
        m = re.search(r"^\s+" + re.escape(label) + r"\s+([0-9.]+) (B|KiB|MiB|GiB)\b", text, re.M)
        if not m:
            raise KeyError(label)
        return float(m.group(1)) * UNIT[m.group(2)]
    return {k: mib(k) for k in ("--gpu-headroom-mib", "claimable", "static weights", "experts", "kv cache")}


def main(argv):
    if len(argv) != 2 or argv[0] not in ("seqs", "pool"):
        print(__doc__, file=sys.stderr)
        return 2
    try:
        a = table(argv[1])
    except KeyError as missing:
        print(f"pool_headroom: no '{missing.args[0]}' line in {argv[1]}", file=sys.stderr)
        return 1
    if argv[0] == "seqs":
        print(math.ceil(a["experts"] / 2 / a["kv cache"]) + 2)
    else:
        head = (a["--gpu-headroom-mib"] + a["claimable"] - a["static weights"] - a["kv cache"] -
                a["experts"] / 2 - 16)
        print(int(head), int(a["experts"]) + 64)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
