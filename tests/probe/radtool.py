"""Read a .rad container's directory (RAD2, abi/rad_format.h at radiance 1.3.0) -- Step 0 (S0.5, S0.12).

  python3 -I radtool.py sizes FILE   exact bytes per category; footprint = distance to the next entry
                                     (4 KiB unit padding is charged to the entry before it)
  python3 -I radtool.py cmp A B      every entry B shares with A by name: are its planes the same
                                     bytes? (S0.5: what --reuse copied)

Only the header, string blob, directory and plane table are read for `sizes`; `cmp` reads the
planes of the entries it compares.
"""
import hashlib
import struct
import sys
from collections import defaultdict

HDR = struct.Struct("<IIQ5Q16Q8Q")
ENT = struct.Struct("<3QII6q3iI3Q2Q")     # 136 bytes
PLN = struct.Struct("<QQ")
assert ENT.size == 136


class Rad:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            h = HDR.unpack(f.read(HDR.size))
            assert h[0] == 0x31444152 and h[1] == 2, (hex(h[0]), h[1])
            self.file_bytes = h[2]
            (str_off, str_bytes, _, _, dir_off, dir_n, plane_off, plane_n, _, _, _, _, _, _,
             self.data_off, self.data_bytes) = h[8:24]
            self.quant_off = h[5]
            f.seek(str_off)
            self.blob = f.read(str_bytes)
            f.seek(dir_off)
            raw = f.read(dir_n * ENT.size)
            f.seek(plane_off)
            praw = f.read(plane_n * PLN.size)
        self.planes = [PLN.unpack_from(praw, i * PLN.size) for i in range(plane_n)]
        self.entries = {}
        for i in range(dir_n):
            e = ENT.unpack_from(raw, i * ENT.size)
            name = self.s(e[0])
            self.entries[name] = dict(name=name, quantizer=self.s(e[1]), options=self.s(e[2]),
                                      layer=e[11], expert=e[12], slot=e[13], n_planes=e[14],
                                      plane_first=e[15], offset=e[16], bytes=e[17])

    def s(self, off):
        return self.blob[off:self.blob.index(b"\0", off)].decode()

    def plane_hash(self, e):
        hsh = hashlib.sha256()
        with open(self.path, "rb") as f:
            for k in range(e["n_planes"]):
                off, n = self.planes[e["plane_first"] + k]
                f.seek(off)
                hsh.update(f.read(n))
        return hsh.hexdigest()


def category(e):
    if e["expert"] >= 0:
        return f"expert ({e['quantizer'] or 'checkpoint'})"
    if "ple_ngram" in e["name"]:
        return "non-expert: n-gram table"
    return f"non-expert ({e['quantizer'] or 'checkpoint'})"


def sizes(path):
    r = Rad(path)
    ents = sorted(r.entries.values(), key=lambda e: e["offset"])
    end = r.data_off + r.data_bytes
    n, nbytes, foot = defaultdict(int), defaultdict(int), defaultdict(int)
    for i, e in enumerate(ents):
        nxt = ents[i + 1]["offset"] if i + 1 < len(ents) else end
        c = category(e)
        n[c] += 1
        nbytes[c] += e["bytes"]
        foot[c] += nxt - e["offset"]
    print(f"{path}: file {r.file_bytes} B, data_off {r.data_off}, quant '{r.s(r.quant_off)}', "
          f"{len(ents)} entries")
    print(f"{'category':40s} {'entries':>8s} {'bytes':>16s} {'footprint':>16s} {'GB':>9s} {'GiB':>9s}")
    for c in sorted(foot):
        print(f"{c:40s} {n[c]:8d} {nbytes[c]:16d} {foot[c]:16d} {foot[c] / 1e9:9.3f} "
              f"{foot[c] / 2**30:9.3f}")
    print(f"{'header and tables (before data_off)':40s} {'':8s} {'':16s} {r.data_off:16d} "
          f"{r.data_off / 1e9:9.3f} {r.data_off / 2**30:9.3f}")


def cmp(a_path, b_path):
    a, b = Rad(a_path), Rad(b_path)
    same, diff, only_b = defaultdict(int), defaultdict(list), defaultdict(int)
    for name, eb in b.entries.items():
        c = category(eb)
        ea = a.entries.get(name)
        if ea is None:
            only_b[c] += 1
        elif a.plane_hash(ea) == b.plane_hash(eb):
            same[c] += 1
        else:
            diff[c].append(name)
    for c in sorted(set(same) | set(diff) | set(only_b)):
        print(f"{c:40s} same {same[c]:5d}  differ {len(diff[c]):5d}  only in B {only_b[c]:5d}"
              + (f"  e.g. {diff[c][0]}" if diff[c] else ""))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "sizes":
        sizes(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "cmp":
        cmp(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
