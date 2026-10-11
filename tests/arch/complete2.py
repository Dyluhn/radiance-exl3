#!/usr/bin/env python3
"""complete2.py -- serve a container on the host and take a greedy 2-token completion of every prompt in
a corpus, through the OpenAI completions endpoint: the DECODE path, which `--kld-ref` (prefill only)
does not reach -- the delta net's recurrent update, paged attention over the cache, the router and
prologue fusions at one token.

    complete2.py OUT.json CORPUS.jsonl -- <radiance command line, without --host/--port>

Writes {"prompts": [{"source", "tokens": [id, id], "text", "logprobs": [...]}...]}. Exit 0 when every
prompt completed; 1 when the server failed or any request did; 2 on bad usage. Only the standard
library: one server process, a free port it picks, a bounded wait for /health, and the process
stopped however this exits.
"""
import json
import socket
import subprocess
import sys
import time
import urllib.request


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port, body, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main(argv):
    if len(argv) < 4 or argv[2] != "--":
        print(__doc__, file=sys.stderr)
        return 2
    out, corpus, cmd = argv[0], argv[1], argv[3:]
    prompts = [json.loads(line) for line in open(corpus, encoding="utf-8") if line.strip()]
    port = free_port()
    log = open(out + ".log", "w", encoding="utf-8")
    server = subprocess.Popen(cmd + ["--host", "127.0.0.1", "--port", str(port)], stdout=log,
                              stderr=subprocess.STDOUT)
    try:
        for _ in range(120):   # bounded: a server that never comes up is a failure, not a hang
            if server.poll() is not None:
                print(f"complete2: the server exited ({server.returncode}); see {out}.log")
                return 1
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                break
            except OSError:
                time.sleep(0.5)
        else:
            print(f"complete2: no /health after 60 s; see {out}.log")
            return 1
        results = []
        for p in prompts:
            r = post(port, {"prompt": p["prompt"], "max_tokens": 2, "temperature": 0, "logprobs": 5,
                            "return_tokens_as_token_ids": True}, timeout=300)
            choice = r["choices"][0]
            lp = choice.get("logprobs") or {}
            results.append({"source": p.get("source", ""), "text": choice.get("text", ""),
                            "tokens": lp.get("tokens"), "token_logprobs": lp.get("token_logprobs"),
                            "top_logprobs": lp.get("top_logprobs"),
                            "finish_reason": choice.get("finish_reason")})
        json.dump({"cmd": cmd, "prompts": results}, open(out, "w", encoding="utf-8"), indent=1)
        return 0
    except OSError as e:
        print(f"complete2: a request failed: {e}; see {out}.log")
        return 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
