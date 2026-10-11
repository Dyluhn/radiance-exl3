#!/usr/bin/env python3
"""ttft.py -- time to first token against a live server, at a fixed prompt length and concurrency.

    ttft.py --url URL --corpus FILE --tokens N --conc C --reps R --out OUT.json [--warmup W] [--tag TAG]

WHAT ONE SAMPLE IS. C requests sent at once (a barrier releases them together), each a prompt of
EXACTLY N tokens, max_tokens 1, temperature 0, not streamed: the request returns when its first token
exists, so the client's wall from send to answer is the time to first token, queueing included. That
is what a user waits. The server's own `timings.prompt_ms` (from admission to first token, without the
queue) is kept beside it. R bursts are sent one after another; the summary is over all R x C requests.

WARM FIRST. W bursts go before them and are kept in the output but left out of the summary. Measured
on the baseline (2026-10-11, five passes): after the steps before it, the first ~4 single requests on
this text read 115-228 ms and the rest 105-110, the server's own prompt_ms moving with them -- the
server settling on new text, which a p50 of nine requests straddles.

WHY TOKEN IDS. The prompt goes as an array of token ids: a leading nonce (tokenised by the server's
own /tokenize, so it differs per request and the prefix cache cannot match from the start) and then
the corpus's ids, cut so the total is exactly N. A text prompt reaches N only through an assumed
bytes-per-token. Every answer must report prompt_tokens == N and no cached tokens, or the run fails:
a cache hit would time work that was never done.

Python standard library only. radiance-exl3 P0. Apache-2.0.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def tokenize(base, text):
    return post(base + "/tokenize", {"prompt": text, "add_special_tokens": False}, 120)["tokens"]


def pct(xs, q):
    xs = sorted(xs)
    return xs[round(q * (len(xs) - 1))]


def burst(base, prompts, timeout):
    """Send every prompt at once; return one row per request, in send order."""
    rows = [None] * len(prompts)
    gate = threading.Barrier(len(prompts))

    def one(i):
        body = {"prompt": prompts[i], "max_tokens": 1, "temperature": 0}
        gate.wait()
        t0 = time.perf_counter()
        out = post(base + "/v1/completions", body, timeout)
        t1 = time.perf_counter()
        usage = out.get("usage", {})
        rows[i] = {"ttft_ms": round((t1 - t0) * 1e3, 2),
                   "prompt_ms": (out.get("timings") or {}).get("prompt_ms"),
                   "prompt_tokens": usage.get("prompt_tokens"),
                   "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)}

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if any(r is None for r in rows):
        raise SystemExit("ttft: a request failed (see the traceback above)")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--corpus", required=True, help="text file whose tokens fill the prompts")
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--conc", type=int, required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=0, help="bursts sent first and left out of the summary")
    ap.add_argument("--tag", default="ttft")
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()

    with open(a.corpus, encoding="utf-8", errors="replace") as f:
        text = f.read(a.tokens * 16)            # far more bytes than N tokens need
    body_ids = tokenize(a.url, text)
    rows = []
    for rep in range(a.warmup + a.reps):
        prompts = []
        for i in range(a.conc):
            nonce = tokenize(a.url, f"{a.tag} {rep} {i} {time.time_ns()}\n\n")
            if len(nonce) + 16 > a.tokens or len(body_ids) < a.tokens:
                raise SystemExit(f"ttft: {a.tokens} tokens is too short for a nonce of {len(nonce)} "
                                 f"or the corpus ({len(body_ids)} tokens)")
            prompts.append(nonce + body_ids[:a.tokens - len(nonce)])
        for i, r in enumerate(burst(a.url, prompts, a.timeout)):
            rows.append({"rep": rep, "i": i, "warm": rep < a.warmup, **r})
        time.sleep(0.5)

    bad = [r for r in rows if r["prompt_tokens"] != a.tokens or r["cached"]]
    if bad:
        raise SystemExit(f"ttft: {len(bad)} answer(s) did not compute exactly {a.tokens} uncached tokens: {bad[0]}")
    kept = [r for r in rows if not r["warm"]]
    t = [r["ttft_ms"] for r in kept]
    pm = [r["prompt_ms"] for r in kept if r["prompt_ms"] is not None]
    res = {"tokens": a.tokens, "conc": a.conc, "reps": a.reps, "warmup": a.warmup, "n": len(kept),
           "p50_ms": round(statistics.median(t), 2), "p10_ms": pct(t, 0.1), "p90_ms": pct(t, 0.9),
           "server_prompt_ms_p50": round(statistics.median(pm), 2) if pm else None, "rows": rows}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"ttft C={a.conc} N={a.tokens}: p50 {res['p50_ms']:.1f} ms  p10 {res['p10_ms']:.1f}  p90 {res['p90_ms']:.1f}"
          f"  ({len(kept)} requests after {len(rows) - len(kept)} warm; server prompt_ms p50 {res['server_prompt_ms_p50']})")


if __name__ == "__main__":
    main()
