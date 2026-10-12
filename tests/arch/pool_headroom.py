#!/usr/bin/env python3
"""pool_headroom.py LOG -- the --gpu-headroom-mib that leaves the experts about half their plane, and
the --host-pool-mib that holds the half it pushes off the card.

Reads the engine's own VRAM budget table from a resident run's log ("VRAM budget, per card": the
headroom it ran with, claimable, static weights, and the elastic split's experts and kv cache, all
printed by core/mem/vram_budget.cpp) and prints the headroom that makes the elastic share the KV
cache's plus half the expert plane: headroom + claimable - static - kv - experts / 2, in MiB; then,
after a space, a pinned host pool of the whole expert plane plus 64 MiB (the default pool is sized for
what a resident plan needs -- measured 768 KiB -- and an expert with no tier is refused at load).
Nothing about the card is assumed; the numbers are the engine's, for this model on this card. Exit 1
when the table is not in the log."""
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
    try:
        t = [table(p) for p in argv]
    except KeyError as missing:
        print(f"pool_headroom: no '{missing.args[0]}' line in {' '.join(argv)}", file=sys.stderr)
        return 1
    a = t[0]
    head = a["--gpu-headroom-mib"] + a["claimable"] - a["static weights"] - a["kv cache"] - a["experts"] / 2
    print(int(head), int(max(x["experts"] for x in t)) + 64)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
