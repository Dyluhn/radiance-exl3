# radiance-exl3 (working name)

EXL3 routed experts for the radiance inference engine, as out-of-tree plugins: an `exl3` quantiser that imports
packed EXL3 planes, a host reference kernel library (`exl3ref`), a gfx12 device kernel library (`exl3r4d`) and a
`("qwen4exp", "exl3")` architecture plugin. Nothing in radiance is modified.

Pinned engine: `RADIANCE_VERSION` (release tag + commit). Plan and evidence: see `notes/` and `NOTES.md`.

## What you need

- **radiance 1.4.0** (`RADIANCE_VERSION`: commit dc339b4, plugin ABI 15), the release these plugins are built
  against. Another release refuses them at build time.
- **AMD gfx1201 cards** (RDNA4; Radeon AI PRO R9700, 32 GB). `exl3r4d` is compiled for gfx1201 only.
- **The EXL3 container** for Qwen3.8-Flash-Next (below), and RAM for the pinned host pool. See
  `docs/TROUBLESHOOTING.md`, "Memory", for measured `--host-pool-mib` values.

## Building the plugin home

The plugins are built against an installed radiance (its headers and CMake config) and install into a plugin
home of their own. They never install into radiance's.

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=<radiance 1.4.0 install prefix> -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build          # -> build/home/{architectures,kernels,quantizers}
```

`build/home` is the plugin home: `architectures/qwen4exp_exl3.so`, `kernels/exl3r4d.so`, `kernels/exl3ref.so`,
`quantizers/exl3_quant.so`. Device kernels need a ROCm toolchain that targets gfx1201 (we build in the same
image the radiance release is built in). `-DRAD_WITH_HIP=OFF` builds the host-only pieces.

## Using it

Put the home ahead of the engine's own on `RADIANCE_HOME`. With the official image:

```sh
docker run --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 \
    -v /path/to/build/home:/plugins:ro -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance \
    -v /path/to/models:/models:ro stilldeadcode/radiance:1.4.0 \
    --model /models/qwen3.8-flash-next-exl3.rad \
    --tp 2 --placement expert_tiered --host-pool-mib 11793 --kv-cache-dtype fp8 --max-num-batched-tokens 2048
```

`rad-info --plugins` with the same `RADIANCE_HOME` should list all four plugins. Every flag the baseline
container is served with applies unchanged. The plugin serves `--tp 1` to `--tp 4`. Engine runs exist for tp1 and tp2. tp3 and tp4 are checked by the
host tests and the kernel suites at every rank's widths. The startup log has
one `I qwen4exp_exl3:` line about prefill staging (`docs/TROUBLESHOOTING.md` says how to read it).

### Environment

| variable | values | what it does |
|---|---|---|
| `EXL3_PREFILL_STAGE` | `auto` (default), `on`, `off` | radiance's prefill stager for this model. `auto` stages when it saves link traffic (from the step size and the routing; the startup line gives the inputs). |

Everything else is a radiance flag or variable (`docs/GUIDE.md` in radiance).

## The container

The EXL3 container is radiance's published Qwen3.8-Flash-Next container (Hugging Face
`StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe`) with its routed experts replaced by turboderp's EXL3 experts
(`turboderp/Qwen3.8-Flash-Next-exl3`). Every other weight is byte-copied. `tools/p6_convert.sh` builds it: it runs
`rad-convert` with `recipes/qwen4exp-exl3.recipe` and `--reuse <baseline>` on a metadata-only skeleton of Qwen's
checkpoint. Its header lists the inputs and environment. A published copy of the EXL3 container is not linked
here yet.

## Known limitations / room to improve

Measured against the baseline container on two Radeon AI PRO R9700 cards (one behind a Gen5 x16 path, one
behind a Gen4 x4 path), same flags, same session.

- **tp2 decode (accepted as is):** MTP off: C=1 about 0.1 ms a step behind, C=2 tied, C=4 ahead, C=8 tied. At C=8
  the card behind the x4 path runs at its power limit (PPT0) with this plugin and not with the baseline, so its
  clocks drop. MTP on: ahead at every concurrency. Prefill 2.0–2.4x the baseline, TTFT lower.
- **tp1 (accepted as is, room to improve):** prefill on the x16 card 8K −4.3% / 24K −0.9% against the baseline
  (single passes; TTFT ahead). On the x4 card prefill and decode are ahead. TTFT at 8 concurrent read 3,350 vs
  2,954 ms, a noisy single pass. The prefetch-friendly expert layout (`QWEN4EXP_EXL3_PREP_SPLIT=0
  QWEN4EXP_EXL3_GD=0`, all four prefetch parts listed) lets radiance's next-layer prefetch work (about 47% of
  its copies used, where the shipped layout makes none). It measured 2,717 ms there, with equal prefill. Choosing
  it automatically when the prefetcher is active is not done.
- **radiance 1.4.0 KV-loan recall:** a long request can fail with `expert slab could not give back the loan`
  when the host pool is full. Give the pool room: `--host-pool-mib 43000` at tp1 served every request where 42010
  failed (`docs/TROUBLESHOOTING.md`).
- **tp3 and tp4** are covered by host tests and kernel suites over simulated layouts (every rank's expert
  widths), not by engine runs. tp4 splits each expert 2/1/1/1 blocks by parity: about 20% more routed work on
  the busiest ranks.
- **tp8 is refused** (an expert has five 128-column blocks), as the baseline refuses it.

## When something goes wrong

`docs/TROUBLESHOOTING.md`: every refusal this plugin prints, the engine messages you are likely to meet with it
(memory, plugin path, kernels), and what to do about each.

## Documents

- `docs/ARCH-qwen4exp_exl3.md`: the architecture plugin's build switches and what they require of the engine.
- `docs/OPS-exl3.md`: the ops (`include/exl3_abi.h`) and their contracts.
- `docs/TROUBLESHOOTING.md`: errors and what to do.
