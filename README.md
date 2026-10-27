# radiance-exl3 (working name)

EXL3 routed experts for the radiance inference engine, as out-of-tree plugins: an `exl3` quantiser that imports
packed EXL3 planes, a host reference kernel library (`exl3ref`), a gfx12 device kernel library (`exl3r4d`) and a
`("qwen4exp", "exl3")` architecture plugin. Nothing in radiance is modified.

Pinned engine: `RADIANCE_VERSION` (release tag + commit). Plan and evidence: see `notes/` and `NOTES.md`.

## What you need

- **radiance 1.12.0** (`RADIANCE_VERSION`: commit 5a9ce49, plugin ABI 27), the release these plugins are built
  against. Another release refuses them at build time. For radiance 1.7.0 use release v1.3.0 of these plugins,
  for radiance 1.6.0 release v1.2.0,
  for radiance 1.5.0 release v1.1.0, and for radiance 1.4.0 release v1.0.1.
- **AMD gfx1201 cards** (RDNA4; Radeon AI PRO R9700, 32 GB). `exl3r4d` is compiled for gfx1201 only.
- **The EXL3 container** for Qwen3.8-Flash-Next (below), and RAM for the pinned host pool. See
  `docs/TROUBLESHOOTING.md`, "Memory", for measured `--host-pool-mib` values.

## Building the plugin home

The plugins are built against an installed radiance (its headers, CMake config and engine binary) and install
into a plugin home of their own. They never install into radiance's.

You need:
- **a radiance 1.12.0 install prefix.** The official image carries one at `/opt/radiance`; copy it out:
  `cid=$(docker create stilldeadcode/radiance:1.12.0) && docker cp $cid:/opt/radiance ./radiance-1.12.0 && docker rm $cid`.
  A source build of the release (`RADIANCE_VERSION`) installed with `cmake --install` works too.
- **a ROCm toolchain whose clang targets gfx1201, and CMake 3.21 or newer.** Our builds use ROCm 7.2.4 (AMD clang
  22, CMake 3.28) on Ubuntu 24.04.

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=$PWD/radiance-1.12.0 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
cmake --install build          # -> build/home/{architectures,kernels,quantizers}
```

`build/home` is the plugin home: `architectures/qwen4exp_exl3.so`, `kernels/exl3r4d.so`, `kernels/exl3ref.so`,
`quantizers/exl3_quant.so`. Device kernels need a ROCm toolchain that targets gfx1201 (we build in the same
image the radiance release is built in). `-DRAD_WITH_HIP=OFF` builds the host-only pieces.
The measurements below were taken with builds against radiance 1.6.0; they have not been repeated on 1.7.0.

`ctest --test-dir build -L host` runs the host tests. The Python ones need `python3` with `numpy`, and skip
without it. The engine-loading one skips where the prefix's `rad-info` cannot start (no ROCm runtime on the host).

## Using it

Put the home ahead of the engine's own on `RADIANCE_HOME`. With the official image:

```sh
docker run --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 \
    --init -p 8000:8000 \
    --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" \
    -v /path/to/build/home:/plugins:ro -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance \
    -v /path/to/models:/models:ro stilldeadcode/radiance:1.12.0 \
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
container is served with applies unchanged. The plugin serves `--tp 1` to `--tp 4`, splitting each expert across the
ranks as the baseline does (at `--tp 4` by the expert's number mod 4: every rank holds five 128-column blocks of every
four experts). Engine runs exist for tp1 and tp2. tp3 and tp4 are checked by the host tests and the kernel suites at
every rank's widths. The startup log has
one `I qwen4exp_exl3:` line about prefill staging (`docs/TROUBLESHOOTING.md` says how to read it).

**A card behind a slow PCIe link** (measured: PCIe 4.0 x4): add `-e EXL3_PREFILL_STAGE=auto`. By default radiance's
prefill stager copies every host-pool expert onto the card once per prefill step, which is right on fast links and
slow on a narrow one; `auto` stages only when that saves link traffic. On the measuring machine below (one card on
x4) tp2 prefill was about twice as fast with it (v1.1.0's default).

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

Measured on one machine against the baseline container, same flags, same session, radiance 1.6.0: two Radeon AI PRO
R9700 cards, one behind a PCIe Gen5 x16 path and one behind a Gen4 x4 path. On this machine the x4 card holds the
baseline back more than this plugin at high concurrency, so the C=4 and C=8 figures favour the plugin compared with a
machine whose cards all sit on fast links. Nothing in the plugin is tuned to these links.

- **BetterBench 0.6.0** (Dylan's benchmark: a corpus of distinct prompts over eight categories, temperature 0.7, cold
  prefix cache; tp2, MTP on, radiance 1.6.0, one run each): combined decode baseline 130.2, this plugin 139.5 tok/s;
  TTFT p50 264 vs 199 ms; total tokens a second at C=1/2/4/8/16 baseline 108 / 134 / 141 / 126 / 122, this plugin
  116 / 150 / 171 / 170 / 165; prefill 2K..64K baseline 1,033..1,365, this plugin 1,084..1,442 tok/s. Draft acceptance
  was the same for both (52-57 %).
- **tp2, radiance 1.6.0, MTP off** (ms a step at C=1/2/4/8, radiance's cstep2, one clean pass each): baseline 10.79 /
  10.91 / 12.04 / 14.07, this plugin 10.86 / 10.90 / 11.85 / 14.24 (C=8 flagged: a card under 95 % busy in that window).
  cstep2 sends the same prompt to every sequence, so at temperature 0 they route through the same experts; with a
  distinct prompt per sequence (more experts a step, as in real use) the same build read 10.65 / 11.89 / 13.91 against
  the baseline's 11.20 / 12.12 / 15.62 at C=1/2/4.
- **tp2, MTP on, cstep2** (tokens a second for each sequence at C=1/2/4/8/16, baseline two passes, this plugin one):
  baseline 164 / 121-122 / 93-98 / 67-90 / 45, this plugin 175 / 156 / 110 / 79 / 49. On a single prompt the draft
  acceptance of either model moved by 10-20 points with the prompt's text, so these single-prompt figures are less
  reliable than BetterBench's.
- **Prefill and time to first token** (tp2, default staging): prefill at 8K / 24K baseline 1,546 / 1,600-1,601, this plugin
  1,663-1,673 / 1,541-1,709 tok/s (two runs); TTFT for 1 request baseline 164-170, this plugin 112-114 ms. TTFT for 8
  concurrent requests depends on how the engine batches a burst against free sequence slots and moved by up to a few
  hundred ms between runs of the same build (baseline 1,205-1,209 ms, this plugin 1,133-1,609 ms over two runs).
- **Room to improve, measured:** after the routed experts of a step of more than 16 rows, the next memory-bound kernel
  runs 3-9 us slower than after the baseline's, which costs about 0.35-0.55 ms a step at C=8-16 (the rotated per-expert
  copies of the input rows this plugin writes); at C=1 with MTP a step is longer while the plugin's own kernels take
  less device time than the baseline's -- the remainder appears only with kernels running side by side, cause open.
- **tp1 (accepted as is, measured on radiance 1.4.0 and 1.5.0):** prefill on the x16 card 8K -4.3 % / 24K -0.9 % against
  the baseline (single passes; TTFT ahead). On the x4 card prefill and decode are ahead. The prefetch-friendly expert
  layout (`QWEN4EXP_EXL3_PREP_SPLIT=0 QWEN4EXP_EXL3_GD=0`) lets radiance's next-layer prefetch work; choosing it
  automatically when the prefetcher is active is not done.
- **radiance KV-loan recall (seen on 1.4.0):** a long request can fail with `expert slab could not give back the loan`
  when the host pool is full. Give the pool room: `--host-pool-mib 43000` at tp1 served every request where 42010
  failed (`docs/TROUBLESHOOTING.md`).
- **tp3 and tp4** are covered by host tests and kernel suites over simulated layouts (every rank's expert widths), not
  by engine runs here (two cards). tp4 rotates each expert's extra block over the four ranks by the expert's number
  mod 4, as the baseline does, so every rank carries the same routed work (14.7 GiB of expert bytes a rank; v1.1.0's
  parity split put 17.6 GiB on two of them); its decode band runs the four-class fused gate_up_down kernel, as tp2 runs
  the two-class one, and the split kernels below its band.
- **tp8 is refused** (an expert has five 128-column blocks), as the baseline refuses it.

## Changes

- **Unreleased** (radiance 1.12.0):
  - Built for radiance 1.12.0 (plugin ABI 27). Both copies three-way merged against 1.7.0
    (`architectures/qwen4exp_exl3/UPSTREAM.md`); the control vector's projection (`--control-vector`) comes with the
    architecture. Measured at tp2 on 2x R9700: greedy output identical to v1.3.0 on radiance 1.7.0 (8 of 8 prompts,
    MTP on); decode and prefill at its speed (one stream, no other load: 206 / 172 / 199 tok/s against 211 / 173 / 198,
    a 37K prefill 5,131 against 5,104 tok/s). A KL record over 6,129 positions moves from 1.7.0 to 1.12.0 by the
    same amount with and without the plugin (mean 0.057 here, 0.063 for an iq4r container on the in-tree
    architecture), so the difference is the engine's, not the port's.
- **v1.3.0** (radiance 1.7.0):
  - Built for radiance 1.7.0 (plugin ABI 17). The model code copied from radiance carries 1.7.0's changes (prefill
    spans, collectives on lane 1, QSA and MTP updates; `architectures/qwen4exp_exl3/UPSTREAM.md` lists them). Same
    output at tp2 as on 1.6.0: the KL rows over 52,132 positions are byte-identical (mean 0.046732). For radiance 1.6.0
    use v1.2.0.
  - tp4 at 1-2 rows: one token's gate_up splits each unit's K over two workgroups when they fit (18.2 -> 12.0 us,
    summation order only), and the four-class prep_gu runs as a 2-D grid that reads the router's expert ids instead
    of searching the offsets (1-3 us less a layer per rank). Kernel timings and per-rank suites, not four-card runs.
  - Measured on radiance 1.7.0 (tp2, production flags, distinct prompts, one pass each against the baseline): decode
    without MTP at C=1/3/4/8 10.70/12.80/14.16/18.23 vs 11.16/13.49/14.33/18.80 ms; prefill 8K/24K 1,734/1,743 vs
    1,542/1,616 tok/s; TTFT 110 vs 156 ms for one request, 1,115 vs 1,204 for eight; MTP decode at C=8/16 66.2/43.7 vs
    59.4/36.4 tok/s a sequence. Output identical to the 1.6.0 builds (temp-0 ident, MTP on and off).
  - The shared expert on a second lane in prefill passes (`QWEN4EXP_EXL3_SHARED_LANE 3`) is a build switch, off: it
    closed the TTFT gap on one card, but on radiance 1.7.0 at tp2 it is slower on every reading (prefill 24K 1,635 vs
    1,743 tok/s, TTFT 132 vs 110 ms, decode C=4 15.96 vs 14.16 ms).
  - `EXL3R4D_DIAG_FP8ACT` (a build switch, off): rounds every expert GEMM input through the baseline's fp8 quantiser,
    to measure what fp8 activations would cost. On the ref64 corpus with its agent documents left out, about a quarter
    of EXL3's KL gain over the baseline comes from keeping activations in f16; the rest is the weight format.
  - Decode: the fused gate_up_down launch starts its first gate_up units earlier and spreads them over the card's WGPs
    (1-3 us less a layer at 4-16 rows), and it now also serves 3-row steps (6-8 us less a layer than the split kernels
    there). tp2, distinct prompts: a 3-sequence decode step 12.77 vs 13.13 ms, 4 sequences 13.70 vs 13.83.
  - tp4: decode steps inside the fused band run a four-class fused gate_up_down kernel (interface v7:
    `moe_exl3_gate_up_down_mod4`): 5 to 18 us less a layer than the split kernels on every rank (kernel timings,
    M = 3 to 16). Below the band a 2-row step's gate_up no longer doubles up units on the card's WGPs (about 12 us less
    a layer), and the four-class kernels read their class records sooner (about 2 us a layer at M = 1). Still checked
    by host tests and per-rank kernel suites only, not engine runs on four cards.
  - Prefill: gate_up's map form takes whole experts longest first, and down streams its output past the cache when it
    is larger than the L2 (8 MiB): tp2 prefill +0.6-0.7 %.
  - Same output at tp2 as v1.2.0 (temp-0 ident). tp1-tp3 declare the fused op from 3 rows instead of 4; nothing else
    in their graphs changed.
- **v1.2.0** (radiance 1.6.0): built for radiance 1.6.0 (plugin ABI 16).
  - The fused gate_up_down kernel's hang is fixed at its cause: its workgroups no longer depend on every one of them
    being resident while other kernels hold part of the card. It covers steps of 4 to 16 rows again (v1.1.0: 6 to 8);
    a C=1 MTP step is 0.26 ms shorter (radiance 1.5.0, tp2, three interleaved rounds).
  - `--tp 4` splits the experts in four classes by number mod 4, as the baseline's `moe_gemm_q_mod4` does (interface
    v6: `moe_exl3_gate_up_mod4`, `moe_exl3_down_mod4`, `moe_exl3_prep_gu_mod4`): every rank holds the same expert bytes.
  - Prefill staging follows radiance's default (on). v1.1.0 turned it off at 2048-token steps by a rule measured on a
    PCIe 4.0 x4 card only; on fast links that cost 7-9 % of prefill. `EXL3_PREFILL_STAGE=auto` keeps v1.1.0's rule.
  - Decode at 2 tokens on tp1 splits K only when the grid fits on the card (98.6 vs 104.8 us a layer); the kernels are
    aligned to 4 KB (code placement had moved timings by up to 6 %).
  - Output at tp2 is identical to v1.1.0 (temp-0 ident). At tp1 a 2-token step sums in a different order.
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
