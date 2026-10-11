#!/usr/bin/env python3
"""fixture_variant.py -- tools/exl3_fixture.py with the host end-to-end differential's three knobs:

    fixture_variant.py [--suh-scale S] [--lm-scale L] [--experts-rounding nearest|toward-zero]
                       [--check BF16_DIR] -- <exl3_fixture.py arguments>

  --suh-scale S   every expert projection's input scales `suh` times S, in the EXL3 source and in the
                  dense weights the skeleton's bf16 experts are rendered from alike (W is linear in
                  suh), so both are one model. The fixture's own suh (0.01 x 0.5..1.5) makes each
                  projection ~0.01 and the routed block ~0.1% of the residual stream: a comparison
                  over that model cannot see its experts (measured: another seed's experts scored as
                  rounding). S = 10 makes the block a share of the stream comparable to the rest.
  --lm-scale L    the lm_head times L (rounded to bf16 again): the fixture's logits are ~unit spread
                  over 248320 tokens, so its output is nearly flat (perplexity ~ vocab) and the
                  reference's f16 log-probabilities, stepped 2^-7 at -12.4, put a KL floor of ~3e-6
                  under everything (measured). A peaked output moves the floor far below the effect.
  --experts-rounding   how the skeleton's bf16 experts round the decoded weights: to nearest even
                  (the fixture's own) or TOWARD ZERO, the noise-floor variant below.
  --check DIR     prove that against a fixture made with the same knobs and nearest rounding, only
                  the routed experts differ: exit 1 if any other tensor does, or none of them does.

Same seed, same generator, same random stream: the knobs change values, never what is drawn.

WHY THIS VARIANT. The differential compares this plugin (EXL3 experts: the exact trellis weight, an
f16 intermediate) with the in-tree plugin over the fixture's bf16 rendering of the same weights
(bf16 weight, bf16 intermediate). The gap between the two is rounding, and its size is the
tolerance -- measured, not guessed: the in-tree plugin over two bf16 renderings of the same exact
weights (nearest vs toward zero) differs by one rounding step in about half of the experts' values,
the same scale as the bf16-weight half of the gap being tested.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import exl3_fixture as fx  # noqa: E402


def to_bf16_toward_zero(x):
    return (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def bf16_to_f32(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def stacked_bf16_toward_zero(self, prefix):
    gu = np.stack([np.concatenate([self.dense[(prefix, e, "gate_proj")], self.dense[(prefix, e, "up_proj")]])
                   for e in range(self.a.experts)])
    dn = np.stack([self.dense[(prefix, e, "down_proj")] for e in range(self.a.experts)])
    return to_bf16_toward_zero(gu), to_bf16_toward_zero(dn)


def tensors(directory):
    """name -> raw bytes of every tensor in a safetensors directory."""
    out = {}
    for f in sorted(Path(directory).glob("*.safetensors")):
        raw = f.read_bytes()
        n = struct.unpack("<Q", raw[:8])[0]
        head = json.loads(raw[8:8 + n])
        for name, d in head.items():
            if name == "__metadata__":
                continue
            a, b = d["data_offsets"]
            out[name] = raw[8 + n + a:8 + n + b]
    return out


def check(ref_dir, var_dir):
    ref, var = tensors(ref_dir), tensors(var_dir)
    if set(ref) != set(var):
        print(f"FAIL tensor names differ: {sorted(set(ref) ^ set(var))[:5]}")
        return 1
    experts = [n for n in ref if ".mlp.experts." in n]
    others = [n for n in ref if n not in experts]
    bad = [n for n in others if ref[n] != var[n]]
    moved = [n for n in experts if ref[n] != var[n]]
    print(f"variant: {len(others)} trunk tensors, {len(bad)} differ; {len(experts)} expert tensors, "
          f"{len(moved)} differ")
    return 1 if bad or not moved else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", help="a nearest-rounding fixture's bf16/ directory to compare against")
    p.add_argument("--suh-scale", type=float, default=1.0)
    p.add_argument("--lm-scale", type=float, default=1.0)
    p.add_argument("--experts-rounding", choices=["nearest", "toward-zero"], default="nearest")
    a, rest = p.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    if a.experts_rounding == "toward-zero":
        fx.Experts.stacked_bf16 = stacked_bf16_toward_zero
    if a.suh_scale != 1.0:
        random_tensor = fx.ref.random_tensor

        def scaled(k, n, bits, rng):
            tr, suh, svh = random_tensor(k, n, bits, rng)
            return tr, (suh.astype(np.float32) * a.suh_scale).astype(np.float16), svh
        fx.ref.random_tensor = scaled
    if a.lm_scale != 1.0:
        skeleton = fx.skeleton

        def scaled_skeleton(*args, **kw):
            t = skeleton(*args, **kw)
            dt, bits = t["lm_head.weight"]
            t["lm_head.weight"] = (dt, fx.ref.to_bf16_bits(bf16_to_f32(bits) * np.float32(a.lm_scale)))
            return t
        fx.skeleton = scaled_skeleton
    rc = fx.main(rest)
    if rc:
        return rc
    if a.check:
        out = Path(rest[rest.index("--out") + 1])
        return check(a.check, out / "bf16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
