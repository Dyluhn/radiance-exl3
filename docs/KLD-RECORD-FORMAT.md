# The radiance KL reference record (`--kld-ref DIR`) — format spec

Source of truth: radiance **1.3.0** (`89cee7c`), `core/kld.h`, `core/kld.cpp`, `core/engine_kld.cpp`,
`core/config.cpp`. File:line citations below are at that commit. These three files are unchanged
between v1.0.8 and v1.3.0 (`git diff --stat v1.0.8 v1.3.0 -- core/kld.cpp core/kld.h core/engine_kld.cpp`
is empty). This document is what `tools/kld_synth.py` writes to; `tests/kld/` holds it to
radiance's own recorder and loader, so if this text and the engine ever disagree, the test fails.

## 1. Files

A record is one directory holding **five** files (the plan's §4 P0 names four; it missed
`rows.f32`, which `load_ref` requires — kld.cpp:334-338). All binary files are raw, headerless,
host byte order (little-endian on every supported host), and their sizes are checked exactly
(`open_sized`, kld.cpp:75-81).

| file | dtype | shape | bytes | content |
|---|---|---|---|---|
| `tokens.i32` | int32 | `[sum(docs.tokens)]` | 4 x tokens | every document's token ids, concatenated in `docs` order |
| `logp.f16` | IEEE binary16 | `[rows, n_vocab]` | 2 x rows x n_vocab | log p of every vocabulary id at every scored position |
| `top.bin` | `TopEntry` | `[rows, 16]` | 8 x 16 x rows | the 16 most likely ids with their exact f32 log p |
| `rows.f32` | float32 | `[rows, 2]` | 8 x rows | `{nll_ref, entropy}` per row |
| `kld.json` | JSON | — | — | the manifest; **written last** (kld.json.tmp, then rename; kld.cpp:516-525) |

A directory without `kld.json` is "a run that died" and is refused by `--kld-ref`
(kld.cpp:284-289) and written over by `--kld-record` (kld.cpp:256-263). A directory WITH one is
never overwritten by the engine.

## 2. Row order and `score_from`

Documents are in `kld.json` `docs` order. Document `d` has `tokens` = n ids, starting at
`off = sum of earlier docs' tokens` in `tokens.i32`, and scores positions

    p in [score_from, n - 1)          rows(d) = n - 1 - score_from  (must be >= 1)

The logits at position p predict token p + 1; the last token predicts nothing (kld.h:32-33).
Positions before `score_from` are still prefilled (they are context), just not scored. Row
`r = row0 + (p - score_from)`; `row0` of the first doc is 0 and each next doc's `row0` is the
previous `row0 + rows` (checked: kld.cpp:315-318). `rows` in the manifest is the sum (kld.cpp:323-327).

Dylan's held-out harness (`r9v-gsq-exl3-depth/scripts/kld_sweep.py:720`) scores
`valid[a+1 : b-1]` per doc span `[a, b)` — exactly `score_from = 1`.

## 3. Row contents (what `record_row` computes, kld.cpp:384-430)

For one row with f32 logits `l[0..V)` and next token `t`:

    m   = max_i l[i]                                   (f32)
    lse = (double)m + log( sum_i (double) expf(l[i] - m) )   (sum in double, id order)
    lp[i] = (float)((double)l[i] - lse)                (f32)

- `logp.f16[r, i] = rad_f32_to_f16(lp[i])` — round to nearest, ties to even, with IEEE subnormals,
  overflow (|x| >= 65520) to inf, |x| < 2^-25 to signed zero (abi/rad_plugin.h:245-279). This is
  numpy's `astype(float16)`.
- `top.bin[r]` = 16 x `struct TopEntry { int32_t id; float lp; }` (kld.cpp:151; 8 bytes, no padding).
  The 16 largest `lp` (f32, the exact value, NOT the f16), **sorted by lp descending, ties by lower
  id first**. Selection is a min-heap fed in id order that replaces its root only for a STRICTLY
  larger lp, so among equal values at the cut the lower ids are kept (kld.cpp:397-423). `top[0].id`
  is therefore the argmax with the lowest id among ties — the candidate's argmax rule. `kKldTop = 16`
  is a compile-time constant (kld.h:65) and the manifest must say `"top": 16`. With `n_vocab < 16`
  radiance would write uninitialised entries; a synthesiser must refuse it.
- `rows.f32[r] = { (float)(lse - (double)l[t]), (float)ent }` with
  `ent = -sum_i p_i * (double)lp[i]`, `p_i = exp((double)lp[i])`, terms with `p_i == 0` skipped.
  `rows.f32[r][0]` is the reference NLL the report's reference perplexity is built from.

## 4. `kld.json`

Written by `Kld::finish()` (kld.cpp:502-515) with `nlohmann::ordered_json`, `dump(1)` + newline.
Keys in this order:

| key | type | written as | read by `load_ref` (kld.cpp:283-342) |
|---|---|---|---|
| `format` | string | the constant `kFormat`, verbatim: **`radiance-kld-1`** (kld.cpp:30) | must equal it, else `RAD_E_FORMAT` |
| `model` | string | the recording run's `--model` | shown as "reference" in the report |
| `corpus` | string | the `--kld-corpus` path | copied into the report |
| `corpus_fnv1a64` | string | `%016llx` of FNV-1a 64 over every corpus line's bytes (newline excluded, empty lines included), offset basis 1469598103934665603, prime 1099511628211 (kld.cpp:32-36, 219-222) -- radiance's basis is the standard 14695981039346656037 missing its last digit, so published FNV vectors do not apply | not read |
| `created` | string | UTC, `%Y-%m-%dT%H:%M:%SZ` | not read |
| `n_vocab` | int | the model's vocabulary | must equal the candidate's `meta.n_vocab`, else `RAD_E_INVAL` |
| `top` | int | 16 | must equal `kKldTop` |
| `rows` | int | total scored rows | must equal the sum over `docs` |
| `docs` | array | one object per document, keys `source`, `tokens`, `score_from`, `row0` | `tokens`, `score_from`, `row0` required (`at`); `source` optional (`value`, default "") |

Keys `load_ref` does not read are ignored, so a synthesiser may add its own (kld_synth adds a
top-level `synth` provenance object and a per-document `id`). `source` is the report's grouping key.

## 5. What `--kld-ref DIR` does with it

- Flags (config.cpp:780-795): `--kld-record` and `--kld-ref` are exclusive; `--kld-ref` refuses
  `--kld-corpus` (the tokens come from `tokens.i32`, never re-tokenised); `--kld-out` only with
  `--kld-ref`.
- Engine setup in KL mode (engine_bringup.cpp:896-911, engine.cpp:413): speculation is forced off
  and an explicit `--num-speculative-tokens N > 0` is **refused** (`RAD_E_INVAL`); the prefix cache
  and step pipelining are off. The arch plugin's logits buffer must be f32 with at least
  `max-num-batched-tokens + max-num-seqs` rows (engine_kld.cpp:21-37); pinned host memory of
  `max-num-batched-tokens x width x 4` B per rank is taken for the copy (engine_kld.cpp:62-71).
- Every document is submitted at once, under one scheduler lock, as request `id = d + 1`,
  `max_tokens = 1`, temperature 0, `score = true` (engine_kld.cpp:77-100): step packing is then a
  function of the corpus and the flags alone, so a rerun batches identically.
- Every scored prefill position's full-vocabulary f32 logits (from all TP ranks' column shards)
  are scored by `score_row` (kld.cpp:440-488):

      lq[i] = l[i] - lse_q                      (candidate, f32 logits, lse in double)
      lp[i] = f16 row, with the 16 top.bin entries replaced by their exact f32 lp
      S = sum_i exp(lp[i]),  A = sum_i exp(lp[i]) * (lp[i] - lq[i])
      KL = A / S - log S                        (p renormalised over the row as stored)

  top-1 agreement = candidate argmax (first max, lowest id) == `top[0].id`; candidate NLL =
  `lse_q - l[t]`; reference NLL = `rows.f32[r][0]`.
- A row never returned by the engine fails the run (`RAD_E_STATE`, kld.cpp:492-497).

## 6. Outputs of a scoring run (kld.cpp:530-598)

- Log lines `kld: KLD mean ... median ... p99 ... p99.9 ... p99.99 ... max ...`, top-1 and PPL, and
  one line per source.
- `--kld-out FILE`: JSON with keys `model`, `reference`, `ref_dir`, `corpus`, `created`, `docs`,
  `percentile` ("linear interpolation between order statistics" — numpy's default), `all`, and
  `by_source` (`{source: summary + "docs"}`); a summary is `positions`,
  `kld {mean, median, p90, p95, p99, p99_9, p99_99, max}`, `top1_agreement`,
  `ppl {candidate, reference, ratio}`.
- `FILE.rows`: `[rows, 4]` float32, reference row order: `{kl, nll_candidate, nll_reference,
  top1 (1.0 / 0.0)}` — the paired per-position record two candidates are compared with.
- Without `--kld-out` nothing is written; the numbers are only in the log.

## 7. How `tools/kld_synth.py` maps Dylan's independent reference onto this

| record field | from |
|---|---|
| documents, ids | `heldout.jsonl` (`ids`, tokenised with HF `tokenizer.json` @ de4b8e4, `add_special_tokens=False`, no BOS); the producer's `meta.json` `doc_ids`/`spans` locate each doc's rows in `ref_final_hidden.npy` |
| `source` | the doc's `domain` (prose, code, math, agent, multilingual, niche, long) |
| `score_from` | 1 (= kld_sweep's `valid[a+1:b-1]`) |
| logits | `ref_final_hidden[pos] @ lm_head^T` in f32: the hidden state is the output of the model's final hyper-connection mixer, whose `hc_norm` is the model's last RMS norm (`r9v_gsq/fullmodel.py:462-467`, `hc_mix` :110-119; the checkpoint has no separate final norm); `lm_head` is the bf16 tensor widened to f32 (kld_sweep.py:866) |
| `n_vocab` | `lm_head` rows (248320) |
| `corpus`, `corpus_fnv1a64` | the `corpus.jsonl` kld_synth writes beside the record (`{"prompt": text, "score_from": 1, "source": domain}` per line), hashed the way `load_corpus` would |
