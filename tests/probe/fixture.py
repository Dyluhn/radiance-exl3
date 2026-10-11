"""Step 0 probe fixtures: variants of a safetensors checkpoint directory (docs/STEP0.md, S0.4/S0.5).

  python3 -I fixture.py copy    SRC DST        copy a checkpoint dir (reflinked where the fs can)
  python3 -I fixture.py exl3cfg DIR            add quantization_config.quant_method = "exl3"
  python3 -I fixture.py poison  DIR REGEX...   overwrite matching tensors' bytes with 0xFF (bf16 NaN)
  python3 -I fixture.py holes   DIR            punch every shard's data region to a hole: headers
                                               intact, every tensor reads as zeros, no disk used

A poisoned tensor that is read produces NaN or 0xFF bytes downstream; a hole reads as zeros. Either
way an output that does not depend on them is the evidence that they were not read.
"""
import json
import os
import re
import struct
import subprocess
import sys


def shards(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".safetensors"))


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


def copy(src, dst):
    if os.path.exists(dst):
        sys.exit(f"fixture: {dst} exists")
    subprocess.run(["cp", "-r", "--reflink=auto", src, dst], check=True)
    for f in os.listdir(dst):
        os.chmod(os.path.join(dst, f), 0o644)


def exl3cfg(d):
    p = os.path.join(d, "config.json")
    with open(p) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = {"quant_method": "exl3"}
    with open(p, "w") as f:
        json.dump(cfg, f, indent=1)


def poison(d, patterns):
    rx = [re.compile(p) for p in patterns]
    n = total = 0
    for s in shards(d):
        base, h = header(s)
        with open(s, "r+b") as f:
            for name, t in h.items():
                if not any(r.search(name) for r in rx):
                    continue
                a, b = t["data_offsets"]
                f.seek(base + a)
                f.write(b"\xff" * (b - a))
                n += 1
                total += b - a
    print(f"poisoned {n} tensor(s), {total} bytes")
    if n == 0:
        sys.exit("fixture: no tensor matched")


def holes(d):
    for s in shards(d):
        base, _ = header(s)
        size = os.path.getsize(s)
        subprocess.run(["fallocate", "--punch-hole", "--keep-size", "-o", str(base), "-l",
                        str(size - base), s], check=True)
        print(f"{s}: data [{base}, {size}) punched")


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    cmd, d = argv[1], argv[2]
    if cmd == "copy" and len(argv) == 4:
        copy(d, argv[3])
    elif cmd == "exl3cfg":
        exl3cfg(d)
    elif cmd == "poison" and len(argv) > 3:
        poison(d, argv[3:])
    elif cmd == "holes":
        holes(d)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
