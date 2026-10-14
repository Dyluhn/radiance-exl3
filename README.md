# radiance-exl3 (working name)

EXL3 routed experts for the radiance inference engine, as out-of-tree plugins: an `exl3` quantiser that imports
packed EXL3 planes, a host reference kernel library (`exl3ref`), a gfx12 device kernel library (`exl3r4d`) and a
`("qwen4exp", "exl3")` architecture plugin. Nothing in radiance is modified.

Pinned engine: `RADIANCE_VERSION` (release tag + commit). Plan and evidence: see `notes/` and `NOTES.md`.

## What you need

- **radiance 1.5.0** (`RADIANCE_VERSION`: commit b93b531, plugin ABI 15), the release these plugins are built
  against. Another release refuses them at build time. For radiance 1.4.0, use release v1.0.0 of these plugins.
- **AMD gfx1201 cards** (RDNA4; Radeon AI PRO R9700, 32 GB). `exl3r4d` is compiled for gfx1201 only.
- **The EXL3 container** for Qwen3.8-Flash-Next (below), and RAM for the pinned host pool. See
  `docs/TROUBLESHOOTING.md`, "Memory", for measured `--host-pool-mib` values.

## Building the plugin home

The plugins are built against an installed radiance (its headers, CMake config and engine binary) and install
into a plugin home of their own. They never install into radiance's.

You need:
- **a radiance 1.5.0 install prefix.** The official image carries one at `/opt/radiance`; copy it out:
  `cid=$(docker create stilldeadcode/radiance:1.5.0) && docker cp $cid:/opt/radiance ./radiance-1.5.0 && docker rm $cid`.
  A source build of the release (`RADIANCE_VERSION`) installed with `cmake --install` works too.
- **a ROCm toolchain whose clang targets gfx1201, and CMake 3.21 or newer.** Our builds use ROCm 7.2.4 (AMD clang
  22, CMake 3.28) on Ubuntu 24.04.

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=$PWD/radiance-1.5.0 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
cmake --install build          # -> build/home/{architectures,kernels,quantizers}
```

`build/home` is the plugin home: `architectures/qwen4exp_exl3.so`, `kernels/exl3r4d.so`, `kernels/exl3ref.so`,
`quantizers/exl3_quant.so`. Device kernels need a ROCm toolchain that targets gfx1201 (we build in the same
image the radiance release is built in). `-DRAD_WITH_HIP=OFF` builds the host-only pieces.
Built this way at the same commit, the four files are byte-identical to the ones the measurements in this README
were taken with.

`ctest --test-dir build -L host` runs the host tests. The Python ones need `python3` with `numpy`, and skip
without it. The engine-loading one skips where the prefix's `rad-info` cannot start (no ROCm runtime on the host).

## Using it

Put the home ahead of the engine's own on `RADIANCE_HOME`. With the official image:

```sh
docker run --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 \
    --init -p 8000:8000 \
    --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" \
    -v /path/to/build/home:/plugins:ro -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance \
    -v /path/to/models:/models:ro stilldeadcode/radiance:1.5.0 \
    --model /models/qwen3.8-flash-next-exl3.rad \
    --tp 2 --tp-wire wht6 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered \
    --host-pool-mib 11793 --kv-cache-dtype fp8 --max-num-batched-tokens 2048 --num-speculative-tokens 3
```

The server listens on port 8000 (`--port`), published on the host by `-p`. radiance's own docs use `--network
host` instead. That works with a rootful Docker daemon, but under rootless Docker the container then cannot see the
cards and the engine falls back to its host backend (`--tp 2 needs 2 devices ...; 1 are usable [host backend]`). This is the
baseline container's production configuration without its prefix-cache tiers. It needs two 32 GB cards and about
65 GiB of free RAM for pinned memory: the 11.5 GiB pool and the 47.7 GiB n-gram table.
Keep `--max-num-seqs` small: the engine's default of 256 reserves the per-sequence linear-attention state for 256
sequences (27.6 GiB), and startup stops with `vram_kv does not fit its fixed costs`.

`rad-info --plugins` with the same `RADIANCE_HOME` should list all four plugins. Every flag the baseline
container is served with applies unchanged. The plugin serves `--tp 1` to `--tp 4`. Engine runs exist for tp1 and tp2. tp3 and tp4 are checked by the
host tests and the kernel suites at every rank's widths. The startup log has
one `I qwen4exp_exl3:` line about prefill staging (`docs/TROUBLESHOOTING.md` says how to read it).

### Environment

| variable | values | what it does |
|---|---|---|
| `EXL3_PREFILL_STAGE` | `on` (default), `off`, `auto` | radiance's prefill stager for this model. `on` is radiance's own default, as the baseline is served. `auto` stages only when it saves link traffic (from the step size and the routing); it was measured on a PCIe 4.0 x4 path only and is meant for cards behind a slow link. |

Everything else is a radiance flag or variable (`docs/GUIDE.md` in radiance).

## The container

The EXL3 container is radiance's published Qwen3.8-Flash-Next container (Hugging Face
`StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe`) with its routed experts replaced by turboderp's EXL3 experts
(`turboderp/Qwen3.8-Flash-Next-exl3`). Every other weight is byte-copied. `tools/p6_convert.sh` builds it: it runs
`rad-convert` with `recipes/qwen4exp-exl3.recipe` and `--reuse <baseline>` on a metadata-only skeleton of Qwen's
checkpoint. Its header lists the inputs and environment. The published container is on Hugging Face:
[`Dyluhn/qwen3.8-flash-next-exl3`](https://huggingface.co/Dyluhn/qwen3.8-flash-next-exl3).

## Known limitations / room to improve

Measured against the baseline container on two Radeon AI PRO R9700 cards (one behind a Gen5 x16 path, one
behind a Gen4 x4 path), same flags, same session.
All numbers come from one machine (one card on PCIe Gen5 x16, one on Gen4 x4). Nothing in the plugin is tuned to those links, and the prefill/TTFT gap on other links has not been measured.

- **tp2 decode on radiance 1.5.0 (one same-session pair):** MTP off, ms a step at C=1/2/4/8: baseline 10.82 / 10.91 /
  12.06 (flagged by the harness: a card under 95% busy in that window) / 14.15, this plugin 10.97 / 10.97 / 11.91 / 14.09: C=1 0.15 ms behind, C=2 0.06 behind, C=4 and C=8 ahead
  (on 1.4.0: C=1 about 0.1 behind, C=2 tied, C=4 ahead, C=8 tied). At C=8 the card behind the x4 path runs at its
  power limit (PPT0) with this plugin and not with the baseline, so its clocks drop. MTP on, tokens a second for each
  sequence at C=1/2/4/8: baseline 168 / 125 / 98 / 69, this plugin 175 / 145 / 132 / 80. Prefill on 1.5.0: 3,010
  tok/s at 8K and 3,650 at 24K (the baseline measured 1,525–1,580 on 1.4.0), time to first token 118 ms (1 request)
  and 374 ms (8 concurrent).
- **tp1 (accepted as is, room to improve):** prefill on the x16 card 8K −4.3% / 24K −0.9% against the baseline
  (single passes; TTFT ahead). On the x4 card prefill and decode are ahead. TTFT at 8 concurrent read 3,350 vs
  2,954 ms, a noisy single pass. The prefetch-friendly expert layout (`QWEN4EXP_EXL3_PREP_SPLIT=0
  QWEN4EXP_EXL3_GD=0`, all four prefetch parts listed) lets radiance's next-layer prefetch work (about 47% of
  its copies used, where the shipped layout makes none). It measured 2,717 ms there, with equal prefill. Choosing
  it automatically when the prefetcher is active is not done.
- **radiance KV-loan recall (seen on 1.4.0):** a long request can fail with `expert slab could not give back the loan`
  when the host pool is full. Give the pool room: `--host-pool-mib 43000` at tp1 served every request where 42010
  failed (`docs/TROUBLESHOOTING.md`).
- **tp3 and tp4** are covered by host tests and kernel suites over simulated layouts (every rank's expert
  widths), not by engine runs. tp4 splits each expert 2/1/1/1 blocks by parity: about 20% more routed work on
  the busiest ranks.
- **tp8 is refused** (an expert has five 128-column blocks), as the baseline refuses it.

## Changes

- **Unreleased:** the fused gate_up_down kernel's hang is fixed at its cause (its workgroups no longer depend on
  every one of them being resident), so it covers steps of 4 to 16 rows again (v1.1.0: 6 to 8). A C=1 MTP step is
  0.26 ms shorter (radiance 1.5.0, tp2, three rounds). Same output.
- **v1.1.0** (radiance 1.5.0): built for radiance 1.5.0. Fixes a hang at 4 concurrent requests with MTP on, where
  the fused gate_up_down kernel could stall a 16-row step: the fused form now covers up to 8 rows, and 9 to 16 rows
  take the split kernels. Same output.
- **v1.0.1** (radiance 1.4.0): the same hang fix for radiance 1.4.0.
- **v1.0.0** (radiance 1.4.0): first release.

## When something goes wrong

`docs/TROUBLESHOOTING.md`: every refusal this plugin prints, the engine messages you are likely to meet with it
(memory, plugin path, kernels), and what to do about each.

## Documents

- `docs/ARCH-qwen4exp_exl3.md`: the architecture plugin's build switches and what they require of the engine.
- `docs/OPS-exl3.md`: the ops (`include/exl3_abi.h`) and their contracts.
- `docs/TROUBLESHOOTING.md`: errors and what to do.

## License

Apache-2.0 (`LICENSE`), Copyright 2026 Dylan Johnston. The EXL3 decode is transcribed from turboderp's
[exllamav3](https://github.com/turboderp-org/exllamav3) (MIT); its notice is in `NOTICE`. The model container is
under the Qwen Community License 1.0 (see its Hugging Face page).
