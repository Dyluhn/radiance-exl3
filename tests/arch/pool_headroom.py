#!/usr/bin/env python3
"""pool_headroom.py LOG -- the --gpu-headroom-mib that leaves the experts about half their plane.

Reads the engine's own VRAM budget table from a resident run's log ("VRAM budget, per card": the
headroom it ran with, claimable, static weights, and the elastic split's experts and kv cache, all
printed by core/mem/vram_budget.cpp) and prints the headroom that makes the elastic share the KV
cache's plus half the expert plane: headroom + claimable - static - kv - experts / 2, in MiB. Nothing
about the card is assumed; the numbers are the engine's, for this model on this card. Exit 1 when the
table is not in the log."""
import re
import sys

UNIT = {"B": 1 / 1048576, "KiB": 1 / 1024, "MiB": 1, "GiB": 1024}


def main(argv):
    text = open(argv[0], encoding="utf-8", errors="replace").read()
    def mib(label):
        m = re.search(r"^\s+" + re.escape(label) + r"\s+([0-9.]+) (B|KiB|MiB|GiB)\b", text, re.M)
        if not m:
            raise KeyError(label)
        return float(m.group(1)) * UNIT[m.group(2)]
    try:
        h, c, s = mib("--gpu-headroom-mib"), mib("claimable"), mib("static weights")
        e, k = mib("experts"), mib("kv cache")
    except KeyError as missing:
        print(f"pool_headroom: no '{missing.args[0]}' line in {argv[0]}", file=sys.stderr)
        return 1
    print(int(h + c - s - k - e / 2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
