#!/usr/bin/env python3
"""exl3_fixture.py -- the tiny EXL3 model every phase converts and runs: a Qwen3.8-Flash-Next (`qwen4_exp`)
of the same architecture, shrunk, whose routed experts are random EXL3 trellises.

    exl3_fixture.py --out DIR [--tokenizer DIR] [--defects] [geometry flags]

writes
    DIR/exl3/          the EXL3 source the `exl3` quantiser imports (its `src=`): turboderp's layout --
                       config.json with exllamav3's quantization_config and model.safetensors (+ index) holding
                       every routed expert, trunk and MTP, as {gate,up,down}_proj.{trellis,suh,svh,<marker>}
    DIR/bf16/          the bf16 SKELETON rad-convert reads (PLAN.md D3, D8): a complete checkpoint of the shrunk
                       model, config.json carrying quantization_config.quant_method "exl3", every weight random
                       bf16 EXCEPT the routed experts, which are bf16 of the EXL3 weights decoded (common/exl3_ref.py)
                       -- so a converter's error report on them measures bf16 rounding, ~2^-9, and nothing else
    DIR/exl3-defects/  (--defects) the source with ONE defect per layer-0 expert, for the quantiser's refusal
                       tests; experts 0 and 7 are clean (positive controls in the same directory)
    DIR/fixture.json   the geometry, seed, K, codebook, the defect table, and every file's sha256

Shape (defaults; PLAN.md §2, P1 brief): 2 layers (one gated-delta-net, one gated-attention:
full_attention_interval 2), 8 experts top-2, hidden 256, expert and shared-expert width 128, the MTP head with
its own 8 experts, every per-head dimension of the real model kept (24 x 256 query heads, 2 KV heads, 16/48
delta-net heads of 128, the n-gram PLE at its real head geometry with a 1000-row hash base). Vocab: the real
real model's 248320 when --tokenizer is given (its files are copied), else 1024 (no tokenizer is written
then; the skeleton is for declare-level use only); --vocab overrides either.

Every trellis symbol is uniform random: any symbol sequence is a valid EXL3 encoding (exllamav3 pack;
exl3_ref.pack). suh/svh are random signs x magnitudes near a real checkpoint's (trellis.py random_tensor).
Deterministic in --seed. Needs numpy only.
"""
import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import exl3_ref as ref  # noqa: E402

MARKERS = {"mul1": ("mul1", 0x83DCD12D), "mcg": ("mcg", 0xCBAC1FED), "3inst": None}

# Layer-0 expert -> the defect its source carries in exl3-defects/ (the quantiser must refuse each by name).
DEFECTS = {
    1: "gate_proj.mul1 holds 0x12345678, not the mul1 marker",
    2: "gate_proj carries the mcg marker (codebook mcg, the recipe says mul1)",
    3: "gate_proj has no marker (codebook 3inst)",
    4: "gate_proj.trellis last dim 16*K+8: half-integer K (exllamav3 1.5.2)",
    5: "gate_proj has a .bias",
    6: "up_proj carries both the mul1 and the mcg marker",
    7: None,
    0: None,
}
DEFECTS_MISSING = (3, "down_proj.svh is missing")  # layer-0 expert 3's down: a missing tensor, beside its gate defect


def text_config(a, vocab):
    """Flash-Next's text_config (turboderp 4.05 config.json, identical to the HF release's), shrunk."""
    return {
        "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 248044, "dtype": "bfloat16",
        "eos_token_id": 248044, "full_attention_interval": 2, "hc_count": 4, "hc_lowrank": 320,
        "head_dim": 256, "heads_per_ngram": 8, "hidden_act": "silu", "hidden_size": a.hidden,
        "indexer_budget": 2048, "indexer_compress_ratio": 4, "indexer_head_dim": 128, "indexer_kv_heads": 1,
        "indexer_n_heads": 4, "initializer_range": 0.02,
        "layer_types": ["full_attention" if (l + 1) % 2 == 0 else "linear_attention" for l in range(a.layers)],
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
        "linear_num_value_heads": 48, "linear_value_head_dim": 128, "make_ngram_vocab_size_divisible_by": 128,
        "mamba_ssm_dtype": "float32", "max_position_embeddings": 262144, "model_type": "qwen4_exp_text",
        "moe_intermediate_size": a.width,
        "mtp": {"hybrid": True, "layer_types": ["full_attention"], "mtp_use_hidden_state_from_layer": None,
                "num_hidden_layers": 1, "rope_theta": 10000000},
        "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False, "ngram_size": 3,
        "ngram_vocab_size_base": a.ngram_base, "num_attention_heads": 24, "num_experts": a.experts,
        "num_experts_per_tok": a.topk, "num_hidden_layers": a.layers, "num_key_value_heads": 2,
        "output_gate_type": "sigmoid", "output_router_logits": False, "pad_token_id": None,
        "partial_rotary_factor": 0.25, "ple_conv_kernel_size": 4, "ple_embed_dim": 2560, "ple_layer_ids": [2],
        "rms_norm_eps": 1e-06,
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
                            "rope_theta": 10000000, "rope_type": "default"},
        "router_aux_loss_coef": 0.001, "shared_expert_intermediate_size": a.width, "split_ngram_parts": 128,
        "tie_word_embeddings": False, "use_cache": True, "vocab_size": vocab,
    }


def config(a, vocab, quantization_config):
    # No vision_config: radiance declares a vision tower whenever vision_config.depth > 0, and the skeleton
    # writes none (the EXL3 work never touches it).
    return {"architectures": ["Qwen4ExpForConditionalGeneration"], "image_token_id": 248056,
            "language_model_only": False, "model_type": "qwen4_exp", "text_config": text_config(a, vocab),
            "tie_word_embeddings": False, "transformers_version": "5.8.0.dev0", "video_token_id": 248057,
            "vision_end_token_id": 248054, "vision_start_token_id": 248053,
            "quantization_config": quantization_config}


DT = {"BF16": "<u2", "F16": "<f2", "I16": "<i2", "I32": "<i4", "I64": "<i8", "F32": "<f4"}


def write_safetensors(tensors, path, metadata=None):
    """tensors: name -> (dtype tag, array). bf16 arrays are uint16 bit patterns. HF's layout: an 8-byte LE header
    length, the JSON header padded with spaces to 8 bytes, then the data in header order."""
    header, blobs, at = {}, [], 0
    if metadata:
        header["__metadata__"] = metadata
    for name, (dt, arr) in tensors.items():
        raw = np.ascontiguousarray(arr).astype(DT[dt], copy=False).tobytes()
        header[name] = {"dtype": dt, "shape": list(np.shape(arr)), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * ((8 - len(head) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for b in blobs:
            f.write(b)


def write_checkpoint(directory, tensors, cfg):
    directory.mkdir(parents=True, exist_ok=True)
    write_safetensors(tensors, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    index = {"metadata": {"total_size": (directory / "model.safetensors").stat().st_size},
             "weight_map": {name: "model.safetensors" for name in tensors}}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")


class Experts:
    """The routed experts of every MoE block, as EXL3 tensors and as the dense weights they decode to."""

    def __init__(self, a, rng):
        self.a, self.rng = a, rng
        # (checkpoint prefix of the block's experts, radiance layer index): trunk layers, then the MTP head at
        # layer index num_hidden_layers (radiance declares it there; arch qwen4exp_fp8.cpp `mtp.declare(.., n_layer)`)
        self.blocks = [(f"model.language_model.layers.{l}.mlp.experts", l) for l in range(a.layers)]
        self.blocks.append(("mtp.layers.0.mlp.experts", a.layers))
        self.exl3 = {}    # name -> (dtype, array): the EXL3 source
        self.dense = {}   # (block prefix, expert, proj) -> W in radiance's [n, k] view, f32

    def projection(self, prefix, e, proj, k, n):
        tr, suh, svh = ref.random_tensor(k, n, self.a.bits, self.rng)
        base = f"{prefix}.{e}.{proj}"
        self.exl3[f"{base}.trellis"] = ("I16", tr)
        self.exl3[f"{base}.suh"] = ("F16", suh)
        self.exl3[f"{base}.svh"] = ("F16", svh)
        if MARKERS[self.a.codebook]:
            tag, value = MARKERS[self.a.codebook]
            self.exl3[f"{base}.{tag}"] = ("I32", np.array(np.uint32(value).view(np.int32)))
        self.dense[(prefix, e, proj)] = ref.reconstruct(tr, suh, svh, self.a.bits, self.a.codebook).T

    def build(self):
        h, w = self.a.hidden, self.a.width
        for prefix, _ in self.blocks:
            for e in range(self.a.experts):
                self.projection(prefix, e, "gate_proj", h, w)
                self.projection(prefix, e, "up_proj", h, w)
                self.projection(prefix, e, "down_proj", w, h)
        return self

    def stacked_bf16(self, prefix):
        """The skeleton's stacked expert tensors: gate_up_proj [E, 2*width, hidden] (gate rows then up rows, the
        order radiance's gate_up weight has) and down_proj [E, hidden, width]."""
        gu = np.stack([np.concatenate([self.dense[(prefix, e, "gate_proj")], self.dense[(prefix, e, "up_proj")]])
                       for e in range(self.a.experts)])
        dn = np.stack([self.dense[(prefix, e, "down_proj")] for e in range(self.a.experts)])
        return ref.to_bf16_bits(gu), ref.to_bf16_bits(dn)


def is_prime(v):
    if v < 2 or (v % 2 == 0 and v != 2):
        return v == 2
    d = 3
    while d * d <= v:
        if v % d == 0:
            return False
        d += 2
    return True


def ngram_primes(base, heads):
    """The 1st..heads-th prime at or above base, the n-gram hash's per-head table sizes."""
    out, p = [], base - 1
    for _ in range(heads):
        p += 1
        while not is_prime(p):
            p += 1
        out.append(p)
    return out


def skeleton(a, vocab, experts, rng):
    """Every tensor of the shrunk model (the names radiance's qwen4exp architecture maps), bf16."""
    t = {}
    n, hc, low, hd = a.hidden, 4, 320, 256
    hn, q_dim, kv_dim = hc * n, 24 * hd, 2 * hd
    conv_dim, v_dim, idx_rows = 2 * 16 * 128 + 48 * 128, 48 * 128, (4 + 1) * 128
    ple_embed, ple_heads, shards = 2560, 16, 128

    def w(name, *shape):
        fan = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        t[name] = ("BF16", ref.to_bf16_bits(rng.normal(0.0, 1.0 / np.sqrt(fan), size=shape).astype(np.float32)))

    def const(name, n_, value):
        t[name] = ("BF16", ref.to_bf16_bits(np.full(n_, value, np.float32)))

    def hc_conn(pre, inject):
        const(f"{pre}.hc_norm.weight", hn, 0.0)
        w(f"{pre}.input_mix_weight_down.weight", low, hn)
        w(f"{pre}.input_mix_weight_up.weight", hn, low)
        if inject:
            w(f"{pre}.block_inject_weight.weight", hc, hn)

    def attention(sa):
        w(f"{sa}.q_proj.weight", 2 * q_dim, n)            # gated: half of it the sigmoid gate
        w(f"{sa}.k_proj.weight", kv_dim, n)
        w(f"{sa}.v_proj.weight", kv_dim, n)
        w(f"{sa}.o_proj.weight", n, q_dim)
        const(f"{sa}.q_norm.weight", hd, 1.0)
        const(f"{sa}.k_norm.weight", hd, 1.0)
        w(f"{sa}.indexer.index_qk_proj.weight", idx_rows, n)
        const(f"{sa}.indexer.q_layernorm.weight", 128, 1.0)
        const(f"{sa}.indexer.k_layernorm.weight", 128, 1.0)

    def moe(pre, prefix):
        w(f"{pre}.gate.weight", a.experts, n)
        gu, dn = experts.stacked_bf16(prefix)
        t[f"{pre}.experts.gate_up_proj"] = ("BF16", gu)
        t[f"{pre}.experts.down_proj"] = ("BF16", dn)
        w(f"{pre}.shared_expert.gate_proj.weight", a.width, n)
        w(f"{pre}.shared_expert.up_proj.weight", a.width, n)
        w(f"{pre}.shared_expert.down_proj.weight", n, a.width)
        w(f"{pre}.shared_expert_gate.weight", 1, n)

    w("lm_head.weight", vocab, n)
    w("model.language_model.embed_tokens.weight", vocab, n)
    hc_conn("model.language_model.hyper_connection_mixer", inject=False)
    for l in range(a.layers):
        pre = f"model.language_model.layers.{l}"
        hc_conn(f"{pre}.attn_hyper_connection", inject=True)
        hc_conn(f"{pre}.mlp_hyper_connection", inject=True)
        if (l + 1) % 2 == 0:
            attention(f"{pre}.self_attn")
        else:
            la = f"{pre}.linear_attn"
            w(f"{la}.in_proj_qkv.weight", conv_dim, n)
            w(f"{la}.in_proj_z.weight", v_dim, n)
            w(f"{la}.in_proj_a.weight", 48, n)
            w(f"{la}.in_proj_b.weight", 48, n)
            w(f"{la}.out_proj.weight", n, v_dim)
            w(f"{la}.conv1d.weight", conv_dim, 1, 4)
            const(f"{la}.A_log", 48, float(np.log(0.05)))
            const(f"{la}.dt_bias", 48, 0.0)
            const(f"{la}.norm.weight", 128, 1.0)
        moe(f"{pre}.mlp", f"{pre}.mlp.experts")
        if l == 1:                                          # ple_layer_ids [2] is 1-indexed: layer 1
            ple = f"{pre}.ple"
            w(f"{ple}.key_proj.weight", hn, ple_embed)
            w(f"{ple}.value_proj.weight", n, ple_embed)
            for k in ("norm_key", "norm_query", "norm_conv"):
                const(f"{ple}.{k}.weight", hn, 1.0)
            w(f"{ple}.conv1d.weight", hn, 1, 4)
            t[f"{ple}.ple_embedding.layer_multipliers"] = (
                "I64", rng.integers(1 << 62, (1 << 63) - 1, 3, dtype=np.int64))
            primes = ngram_primes(a.ngram_base, ple_heads)
            total = -(-sum(primes) // 128) * 128            # make_ngram_vocab_size_divisible_by
            t[f"{ple}.ple_embedding.ngram_heads_vocab_sizes"] = ("I64", np.array(primes, np.int64))
            t[f"{ple}.ple_embedding.ngram_heads_offsets"] = ("I64", np.cumsum([0] + primes[:-1]).astype(np.int64))
            table = rng.normal(0.0, 1.0 / np.sqrt(ple_embed // ple_heads),
                               size=(total, ple_embed // ple_heads)).astype(np.float32)
            rows = total // shards
            for i in range(shards):
                t[f"{ple}.ple_embedding.ngram_embedding.shard_{i}.weight"] = (
                    "BF16", ref.to_bf16_bits(table[i * rows:(i + 1) * rows]))
    const("mtp.pre_fc_norm_hidden.weight", hn, 0.0)
    const("mtp.pre_fc_norm_embedding.weight", n, 0.0)
    w("mtp.fc_hidden.weight", n, n)
    w("mtp.fc_embedding.weight", n, n)
    hc_conn("mtp.hyper_connection_mixer", inject=False)
    hc_conn("mtp.layers.0.attn_hyper_connection", inject=True)
    hc_conn("mtp.layers.0.mlp_hyper_connection", inject=True)
    attention("mtp.layers.0.self_attn")
    moe("mtp.layers.0.mlp", "mtp.layers.0.mlp.experts")
    return t


def defective(a, src):
    """The source with one defect per layer-0 expert (DEFECTS), for the refusal tests."""
    t = dict(src)
    pre = "model.language_model.layers.0.mlp.experts"
    g = lambda e, proj="gate_proj": f"{pre}.{e}.{proj}"  # noqa: E731
    t[f"{g(1)}.mul1"] = ("I32", np.array(np.int32(0x12345678)))
    del t[f"{g(2)}.mul1"]
    t[f"{g(2)}.mcg"] = ("I32", np.array(np.uint32(0xCBAC1FED).view(np.int32)))
    del t[f"{g(3)}.mul1"]
    tr = t[f"{g(4)}.trellis"][1]
    t[f"{g(4)}.trellis"] = ("I16", np.concatenate([tr, np.zeros(tr.shape[:-1] + (8,), np.int16)], -1))
    t[f"{g(5)}.bias"] = ("F16", np.zeros(a.width, np.float16))
    t[f"{g(6, 'up_proj')}.mcg"] = ("I32", np.array(np.uint32(0xCBAC1FED).view(np.int32)))
    del t[f"{g(DEFECTS_MISSING[0], 'down_proj')}.svh"]
    return t


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="directory to write exl3/, bf16/ (and exl3-defects/) into")
    ap.add_argument("--tokenizer", help="a directory with the real tokenizer files (tokenizer.json, ...), copied "
                                        "into bf16/")
    ap.add_argument("--vocab", type=int, help="vocab size (default: 248320 with --tokenizer, else 1024)")
    ap.add_argument("--defects", action="store_true", help="also write exl3-defects/")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--codebook", default="mul1", choices=ref.CODEBOOKS)
    ap.add_argument("--ngram-base", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.hidden % 128 or a.width % 128:
        ap.error("--hidden and --width must be multiples of 128 (EXL3's rotation block)")
    if not 1 <= a.bits <= 8:
        ap.error("--bits is 1..8")
    if a.defects and (a.experts < 8 or a.codebook != "mul1"):
        ap.error("--defects needs at least 8 experts and --codebook mul1")
    vocab = a.vocab or (248320 if a.tokenizer else 1024)   # 248320: the real model's (padded) vocab

    out = Path(a.out)
    rng = np.random.default_rng(a.seed)
    experts = Experts(a, rng).build()
    exl3_qc = {"quant_method": "exl3", "version": "1.4.4", "bits": float(a.bits), "head_bits": 6,
               "calibration": {"rows": 250, "cols": 2048}, "out_scales": "always", "codebook": a.codebook,
               "mtp_bits": a.bits}
    write_checkpoint(out / "exl3", experts.exl3, config(a, vocab, exl3_qc))
    write_checkpoint(out / "bf16", skeleton(a, vocab, experts, rng), config(a, vocab, {"quant_method": "exl3"}))
    if a.tokenizer:
        for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json",
                  "chat_template.jinja"):
            if (Path(a.tokenizer) / f).exists():
                shutil.copy2(Path(a.tokenizer) / f, out / "bf16" / f)
    if a.defects:
        write_checkpoint(out / "exl3-defects", defective(a, experts.exl3), config(a, vocab, exl3_qc))

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "fixture.json")
    manifest = {
        "tool": "tools/exl3_fixture.py", "seed": a.seed, "bits": a.bits, "codebook": a.codebook,
        "geometry": {"layers": a.layers, "experts": a.experts, "topk": a.topk, "hidden": a.hidden,
                     "width": a.width, "vocab": vocab, "moe_blocks": [p for p, _ in experts.blocks]},
        "radiance_layer_of_block": {p: l for p, l in experts.blocks},
        "defects": ({str(e): d for e, d in DEFECTS.items()} | {f"{DEFECTS_MISSING[0]}.down": DEFECTS_MISSING[1]})
        if a.defects else {},
        "sha256": {str(p.relative_to(out)): sha256(p) for p in files},
    }
    (out / "fixture.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in files)
    print(f"fixture at {out}: {a.layers} layers + MTP, {a.experts} experts, hidden {a.hidden}, width {a.width}, "
          f"K={a.bits} {a.codebook}, vocab {vocab}; {len(experts.exl3)} EXL3 tensors; {size / 2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
