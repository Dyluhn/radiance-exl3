# Troubleshooting radiance-exl3

Each entry gives what you see, why it happens and what to do. Lines starting `radiance: qwen4exp_exl3:` or
`I qwen4exp_exl3:` come from this plugin. Other lines come from the radiance engine. Run `rad-info --plugins`
with your `RADIANCE_HOME` first: it shows which plugin homes the engine sees.

## Versions and loading

**The build stops with "radiance at CMAKE_PREFIX_PATH has ABI N; these plugins are written against ABI 15".**
The plugins are pinned to one radiance release (`RADIANCE_VERSION`: 1.4.0, ABI 15). Point `CMAKE_PREFIX_PATH`
at an install of that release. A plugin built against another release is a different contract. The engine
will not load it, and if it did, the vendored architecture would not match the engine's.

**The engine does not serve the EXL3 container, or says no architecture takes ("qwen4exp", "exl3").**
The engine cannot see this plugin's home. Put the home first on the search path:
`RADIANCE_HOME=/path/to/home:/opt/radiance/share/radiance`. With the official image, mount the home and set the
variable (see README, "Using it"). `rad-info --plugins` should list `qwen4exp_exl3`, `exl3r4d`, `exl3ref` and
`exl3_quant`.

**The engine says an op `moe_exl3_*` has no kernel, or that only the reference kernels serve it.**
`exl3r4d` (the device kernels) is missing from the home, or the card is not one it was built for. The device
kernels are gfx12 (RDNA4, Radeon AI PRO R9700 / gfx1201). `exl3ref` is a host reference for tests: the engine accepts
reference kernels only with `--debug-accept-reference-kernels`, and they are far too slow to serve with. Check `kernels/exl3r4d.so` is in the home, and that the
cards are gfx1201.

## Wrong container

**`radiance: '<weight>' is <encoding>, and this plugin serves the routed experts as EXL3 ... This container is
not an EXL3 one`** or **`... that is a bf16 model, which the in-tree qwen4exp_fp8 serves`.**
The container's routed experts are not EXL3. This plugin claims the qwen4exp family's quantised slot, so with
its home on the path the engine offers it the container. Serve that container from a `RADIANCE_HOME` without
this plugin's home, or build an EXL3 container (README, "The container").

**`radiance: qwen4exp_exl3: this checkpoint's quantisation ('...') is imported weight by weight`.**
A GPTQ / AutoRound / Quark checkpoint goes to the family's one quantised plugin, which is this one while it is
installed. It serves EXL3 experts only. Serve that checkpoint without this plugin's home.

## Tensor parallelism

**`radiance: qwen4exp_exl3: --tp 8 is more ranks than an expert has 128-column blocks (5) ...`**
Each expert is 640 columns, five blocks of 128. Every rank must hold whole blocks of every expert, so the model
is served at `--tp 1`, `2`, `3` or `4` (the baseline fp8 container refuses `--tp 8` too). At `--tp 3` and
`--tp 4` the split is uneven: 2/2/1 and 2/1/1/1 blocks of each expert, alternating by expert parity. The
busiest ranks then carry about 20% more routed work than an even split would.

**`radiance: qwen4exp_exl3: layer N was declared with the four-class expert split ...`**
An internal error: this build always splits by parity. Report it with the full startup log.

## Memory

**`W one --max-model-len sequence's cache (X GiB) leaves the experts Y GiB, short of the Z GiB the pinned host
pool cannot hold: raise --host-pool-mib or lower --max-model-len`, then `E startup stopped: out of memory`.**
The engine's message. Every routed expert lives either in VRAM or in the pinned host pool (`--host-pool-mib`).
Whatever the pool cannot hold must fit in the VRAM left after the weights that always stay on the card and
one `--max-model-len` sequence of KV cache. Raise `--host-pool-mib` by at least `Z - Y` GiB (the machine needs
that much more free RAM, pinned), or lower `--max-model-len`. Configurations measured on 32 GB cards
(Radeon AI PRO R9700), with `--placement expert_tiered --kv-cache-dtype fp8`:

| ranks | `--host-pool-mib` | notes |
|---|---|---|
| `--tp 2` | 11793 | the same VRAM split the baseline container gets at 12288; KV about 5 GiB a card |
| `--tp 1` | 42010 | `--max-num-seqs 4 --max-num-batched-tokens 4096` |

`--gpu-headroom-mib` comes off the card's VRAM before any of this: a larger headroom needs a larger pool.

**The run is slow and the startup report says `weights-from-disk`.** Some weights are read from the
container on disk as they are needed, because neither VRAM nor the pool holds them. Raise `--host-pool-mib`
if the RAM allows it.

**`I qwen4exp_exl3: --weights-disk-tier: the decode band issues the one gate_up and down ...`.**
Informational. With the disk tier on, this plugin declares its decode experts as two ops, gate_up then down,
instead of the fused op, because the engine's file stager keys on ops that a decode pass issues. The output is
bit-identical. Decode is a few tenths of a ms a step slower (measured at tp2: about 0.3 ms at 1–2 concurrent).

## Prefill staging

**`I qwen4exp_exl3: prefill staging ON|OFF (automatic): a pooled expert's zero-copy reads a step of N tokens R
(top-k of E, 64-row entries) against 1.50 to stage; the serving cards' narrowest host path ... [...]`**
Informational. radiance's prefill stager copies each routed layer's pooled experts onto the card once per
prefill step. Without it, the kernels read them over the link, once per 64-row block of an expert's tokens. The
plugin stages when the expected reads per step, R, reach 1.5, i.e. when staging saves link traffic. That
depends on the step size (`--max-num-batched-tokens`) and the model's routing:

| setup | R | choice | measured (Radeon AI PRO R9700, one card on a PCIe 4.0 x4 path) |
|---|---|---|---|
| tp2, 2048 tokens a step | 1.00 | off | off 2.2x faster at 24K (3,726 vs 1,705 tok/s) |
| tp1, 4096 tokens a step | 1.96 | on | on 1.47x faster at 8K (607 vs 413 tok/s) |

`EXL3_PREFILL_STAGE=on`, `off` or `auto` (the default) overrides the choice. Any other value is ignored, and the
startup line says so. The host-path figure is printed for reference only. "unknown" or "could not be read"
there means the plugin could not read the PCI topology from sysfs (for example, in a container without
`/sys`). That changes nothing.

## Known engine issues

**`expert slab could not give back the loan ... wrong phase` on a long prompt at `--tp 1`.**
Seen at 24K-token prompts at tp1 with `--max-num-batched-tokens 4096`. The message comes from radiance's
activation arena, which lends memory to the expert slab, not from this plugin. Prompts of 8K were served. It
is open with the engine.

## Reporting a problem

Include the full startup log (every `I`/`W`/`E` line up to "ready"), the command line, `rad-info --plugins`,
the plugin home's `SHA256SUMS`, if you have one, and `RADIANCE_VERSION`.
