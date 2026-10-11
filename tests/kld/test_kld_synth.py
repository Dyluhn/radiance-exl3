"""tools/kld_synth.py against hand-computed rows, a naive per-row oracle, and radiance's own KL code.

Fast tier, host only: `python3 -m pytest tests/kld`. The radiance tests (marked by name) build
tests/kld/kld_oracle.cpp against a radiance host build and SKIP, saying why, unless
RADIANCE_SRC (the source tree at the pinned tag) and RADIANCE_BUILD (its host build tree, holding
core/librad_core.a) are set.
"""
import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
spec = importlib.util.spec_from_file_location("kld_synth", TOOLS / "kld_synth.py")
ks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ks)


# ----------------------------------------------------------------------------- the row, by hand

def test_hand_computed_row():
    # weights 1 x16, 2, 3, 4, 10 (sum 35): p = w / 35, log p = ln w - ln 35, ln 35 = 3.5553480614894135
    logits = np.log(np.array([1.0] * 16 + [2, 3, 4, 10])).astype(np.float32)[None, :]
    logp, top, rows = ks.record_rows(logits, np.array([18]))
    assert rows[0, 0] == pytest.approx(2.169053700369523, abs=1e-6)    # NLL of id 18: ln 35 - ln 4
    assert rows[0, 1] == pytest.approx(2.6052577871310066, abs=1e-6)   # ln 35 - (2ln2 + 3ln3 + 4ln4 + 10ln10) / 35
    assert top["id"][0].tolist() == [19, 18, 17, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert top["lp"][0, :5].tolist() == pytest.approx(
        [-1.2527629684953676, -2.169053700369523, -2.4567357728213037, -2.8622008809294682, -3.5553480614894135], abs=1e-6)
    assert logp[0, 0] == np.float16(-3.5546875)       # -3.5553 to the f16 grid of [2, 4), step 2^-9
    assert logp[0, 19] == np.float16(-1.2529296875)   # -1.25276 to the grid of [1, 2), step 2^-10


def test_all_equal_row_keeps_the_lowest_ids():
    logp, top, rows = ks.record_rows(np.zeros((1, 40), dtype=np.float32), np.array([0]))
    assert top["id"][0].tolist() == list(range(16))
    assert rows[0, 0] == pytest.approx(3.6888794541139363, abs=1e-6)   # ln 40


def test_refuses_a_vocabulary_smaller_than_the_top():
    with pytest.raises(ValueError, match="n_vocab >= 16"):
        ks.record_rows(np.zeros((1, 15), dtype=np.float32), np.array([0]))


def test_fnv1a64_uses_radiances_offset_basis_and_concatenates_lines():
    assert ks.fnv1a64([]) == "14650fb0739d0383"                  # 1469598103934665603, kld.cpp:32 (not the standard basis)
    assert ks.fnv1a64([b"a"]) == "44bd8ad473cd9906"              # (basis ^ 0x61) * 1099511628211 mod 2^64
    assert ks.fnv1a64([b"foo", b"bar"]) == ks.fnv1a64([b"foobar"])


# ----------------------------------------------------------------------------- selection

DOMAINS = ["a"] * 6 + ["b"] * 3 + ["c"]
LENGTHS = [50, 10, 40, 20, 30, 60, 7, 5, 9, 100]


def test_select_is_proportional_and_shortest_first():
    # quotas a 3.0, b 1.5, c 0.5 -> a 3, b 1, c 1 (the floor of one)
    assert ks.select_docs(DOMAINS, LENGTHS, 5) == [1, 3, 4, 7, 9]


def test_select_gives_the_remainder_to_the_largest_fraction():
    # quotas a 3.6, b 1.8, c 0.6 -> floors a 3, b 1, c 1 = 5; the sixth goes to b (.8 > .6, .6)
    assert ks.select_docs(DOMAINS, LENGTHS, 6) == [1, 3, 4, 6, 7, 9]


def test_select_refuses_when_the_floor_of_one_overshoots():
    # 4 documents over 3 domains, n = 3 is fine; 5 domains of one document each with n = 4 is not
    with pytest.raises(ValueError, match="one per domain"):
        ks.select_docs(["a", "b", "c", "d", "e", "e", "e", "e"], [1] * 8, 4)


def test_select_refuses_fewer_documents_than_domains():
    with pytest.raises(ValueError, match="one per domain"):
        ks.select_docs(DOMAINS, LENGTHS, 2)


# ----------------------------------------------------------------------------- a whole record

V, H = 64, 8


def write_bf16_safetensors(path: Path, name: str, w32: np.ndarray) -> np.ndarray:
    """w32 truncated to bf16 into a one-tensor safetensors file; returns the exact f32 of what was stored."""
    bits = (w32.astype(np.float32).view(np.uint32) >> 16).astype("<u2")
    header = json.dumps({name: {"dtype": "BF16", "shape": list(w32.shape), "data_offsets": [0, bits.nbytes]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bits.tobytes())
    return (bits.astype(np.uint32) << 16).view(np.float32)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """Three documents (x: 7 tokens, y: 5 and 9) as kld_sweep leaves them, and their hidden states."""
    d = tmp_path_factory.mktemp("inputs")
    rng = np.random.default_rng(0)
    lens = [7, 5, 9]
    docs = [{"id": f"doc{i}", "domain": dom, "text": f"text {i} é", "ids": rng.integers(0, V, n).tolist()}
            for i, (dom, n) in enumerate(zip(["x", "y", "y"], lens))]
    spans, a = [], 0
    for n in lens:
        spans.append([a, a + n])
        a += n
    (d / "docs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in docs))
    (d / "meta.json").write_text(json.dumps({"doc_ids": [x["id"] for x in docs], "spans": spans, "tokens": a}))
    hidden = rng.normal(0, 1, (a, H)).astype(np.float32)
    np.save(d / "hidden.npy", hidden)
    w = write_bf16_safetensors(d / "head.safetensors", "lm_head.weight", rng.normal(0, 1, (V, H)).astype(np.float32))
    return {"dir": d, "docs": docs, "spans": spans, "hidden": hidden, "w": w}


def naive_rows(hidden, w, ids, a, shift=0):
    """The oracle: one row at a time, in f64, straight from the definition."""
    out = []
    for p in range(1, len(ids) - 1):
        logit = hidden[a + p + shift].astype(np.float64) @ w.astype(np.float64).T
        lp = logit - (logit.max() + np.log(np.exp(logit - logit.max()).sum()))
        out.append((lp, -lp[ids[p + 1]], -(np.exp(lp) * lp).sum()))
    return out


def run_synth(inputs, out: Path, *extra):
    d = inputs["dir"]
    cmd = [sys.executable, str(TOOLS / "kld_synth.py"), "--hidden", str(d / "hidden.npy"), "--meta", str(d / "meta.json"),
           "--docs", str(d / "docs.jsonl"), "--lm-head", str(d / "head.safetensors"), "--model", "test-ref",
           "--out", str(out), "--min-free-gib", "0", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


@pytest.fixture(scope="module")
def record(inputs, tmp_path_factory):
    out = tmp_path_factory.mktemp("rec") / "ref"
    r = run_synth(inputs, out, "--doc-ids", "doc0,doc2")
    assert r.returncode == 0, r.stderr
    return out


def test_record_files_have_the_sizes_and_manifest_radiance_checks(record):
    m = json.loads((record / "kld.json").read_text())
    assert list(m)[:9] == ["format", "model", "corpus", "corpus_fnv1a64", "created", "n_vocab", "top", "rows", "docs"]
    assert (m["format"], m["n_vocab"], m["top"], m["rows"]) == ("radiance-kld-1", 64, 16, 5 + 7)
    assert [(x["source"], x["tokens"], x["score_from"], x["row0"]) for x in m["docs"]] == [("x", 7, 1, 0), ("y", 9, 1, 5)]
    sizes = {f: (record / f).stat().st_size for f in ["tokens.i32", "logp.f16", "top.bin", "rows.f32"]}
    assert sizes == {"tokens.i32": 16 * 4, "logp.f16": 12 * 64 * 2, "top.bin": 12 * 16 * 8, "rows.f32": 12 * 8}
    lines = (record / "corpus.jsonl").read_bytes().split(b"\n")[:-1]
    assert m["corpus_fnv1a64"] == ks.fnv1a64(lines)
    assert json.loads(lines[1]) == {"prompt": "text 2 é", "score_from": 1, "source": "y"}


def test_record_rows_match_the_naive_oracle_position_by_position(record, inputs):
    logp = np.fromfile(record / "logp.f16", dtype="<f2").reshape(-1, V).astype(np.float64)
    top = np.fromfile(record / "top.bin", dtype=ks.TOP_DTYPE).reshape(-1, 16)
    rows = np.fromfile(record / "rows.f32", dtype="<f4").reshape(-1, 2)
    want = [r for i in (0, 2) for r in naive_rows(inputs["hidden"], inputs["w"], inputs["docs"][i]["ids"], inputs["spans"][i][0])]
    assert np.fromfile(record / "tokens.i32", dtype="<i4").tolist() == inputs["docs"][0]["ids"] + inputs["docs"][2]["ids"]
    assert np.abs(logp - np.array([w[0] for w in want])).max() < 4e-3          # f16 of |lp| < 16: step <= 2^-7
    assert top["id"].tolist() == [np.argsort(-w[0], kind="stable")[:16].tolist() for w in want]
    assert np.abs(rows[:, 0] - [w[1] for w in want]).max() < 1e-5
    assert np.abs(rows[:, 1] - [w[2] for w in want]).max() < 1e-5


def test_the_oracle_comparison_catches_an_off_by_one_position(record, inputs):
    """Negative control: the same check fed the next position's hidden state must fail by a lot."""
    rows = np.fromfile(record / "rows.f32", dtype="<f4").reshape(-1, 2)
    shifted = naive_rows(inputs["hidden"], inputs["w"], inputs["docs"][0]["ids"], inputs["spans"][0][0], shift=1)
    assert np.abs(rows[:5, 0] - [w[1] for w in shifted]).max() > 0.1


def test_nll_check_passes_on_the_producers_numbers_and_refuses_wrong_ones(inputs, tmp_path):
    nll = np.full(inputs["hidden"].shape[0], np.nan, dtype=np.float32)
    for doc, (a, _) in zip(inputs["docs"], inputs["spans"]):
        nll[a + 1:a + len(doc["ids"]) - 1] = [w[1] for w in naive_rows(inputs["hidden"], inputs["w"], doc["ids"], a)]
    np.savez(tmp_path / "good.npz", nll_ref=nll)
    np.savez(tmp_path / "bad.npz", nll_ref=nll + 0.01)
    assert run_synth(inputs, tmp_path / "good", "--n-docs", "2", "--check-nll", str(tmp_path / "good.npz")).returncode == 0
    bad = run_synth(inputs, tmp_path / "bad", "--n-docs", "2", "--check-nll", str(tmp_path / "bad.npz"))
    assert bad.returncode != 0 and "NLL check FAILED" in bad.stderr
    assert not (tmp_path / "bad" / "kld.json").exists()


def test_refuses_to_write_over_a_reference(record, inputs):
    r = run_synth(inputs, record, "--doc-ids", "doc0")
    assert r.returncode != 0 and "already holds a reference" in r.stderr


# ----------------------------------------------------------------------------- radiance's own code

@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    src, build = os.environ.get("RADIANCE_SRC"), os.environ.get("RADIANCE_BUILD")
    if not (src and build):
        pytest.skip("SKIP: RADIANCE_SRC and RADIANCE_BUILD unset -- no radiance host build to hold kld_synth to")
    lib = Path(build) / "core" / "librad_core.a"
    if not lib.is_file():
        pytest.skip(f"SKIP: {lib} missing -- RADIANCE_BUILD is not a radiance host build tree")
    exe = tmp_path_factory.mktemp("oracle") / "kld_oracle"
    inc = [f"-I{src}/core", f"-I{src}/abi", f"-I{build}/rad_include", "-isystem", f"{src}/vendor",
           "-isystem", f"{src}/vendor/shim", "-isystem", f"{src}/vendor/llama/common"]
    cmd = [os.environ.get("CXX", "c++"), "-std=gnu++20", "-O2", *inc, str(Path(__file__).with_name("kld_oracle.cpp")), str(lib),
           "-ldl", f"{build}/vendor/librad_vendor_llama.a", f"{build}/vendor/librad_vendor_httplib.a", "-o", str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return exe


def test_radiance_records_the_same_bytes_from_the_same_logits(oracle, tmp_path):
    rng = np.random.default_rng(1)
    lens, vocab = [6, 11], 300
    tokens = rng.integers(0, vocab, sum(lens)).astype("<i4")
    logits = (rng.normal(0, 3, (sum(lens), vocab)) + 12 * (rng.random((sum(lens), vocab)) < 0.01)).astype(np.float32)
    logits[3, 100:140] = logits[3].max() + 1.0                      # 40 equal values straddling the top-16 cut
    logits[3, 7] = logits[3, 100] + 4.0                             # one above them: top = 7, then 100..114
    tokens.tofile(tmp_path / "tokens.i32")
    logits.tofile(tmp_path / "logits.f32")
    (tmp_path / "docs.txt").write_text("".join(f"{n} 1 s{i}\n" for i, n in enumerate(lens)))
    r = subprocess.run([str(oracle), "record", str(tmp_path / "rad"), str(vocab), str(tmp_path / "docs.txt"),
                        str(tmp_path / "tokens.i32"), str(tmp_path / "logits.f32")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    pos = [a + p for a, n in zip([0, lens[0]], lens) for p in range(1, n - 1)]
    logp, top, rows = ks.record_rows(logits[pos], tokens[[p + 1 for p in pos]])
    assert (tmp_path / "rad" / "logp.f16").read_bytes() == logp.tobytes()
    assert (tmp_path / "rad" / "top.bin").read_bytes() == top.tobytes()
    assert np.allclose(np.fromfile(tmp_path / "rad" / "rows.f32", dtype="<f4").reshape(-1, 2), rows, rtol=1e-6, atol=1e-6)


def run_score(oracle, ref: Path, vocab: int, logits: np.ndarray, out: Path):
    logits.astype(np.float32).tofile(out.with_suffix(".f32"))
    r = subprocess.run([str(oracle), "score", str(ref), str(vocab), str(out.with_suffix(".f32")), str(out)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(out.read_text()), np.fromfile(str(out) + ".rows", dtype="<f4").reshape(-1, 4)


def test_radiance_loads_a_synth_record_and_scores_the_reference_itself_at_zero(oracle, record, inputs, tmp_path):
    hidden, w = inputs["hidden"], inputs["w"]
    logits = np.concatenate([hidden[a:b] for (a, b) in (inputs["spans"][0], inputs["spans"][2])]) @ w.T
    rep, rows = run_score(oracle, record, V, logits, tmp_path / "same.json")
    assert rep["reference"] == "test-ref" and rep["all"]["positions"] == 12
    assert sorted(rep["by_source"]) == ["x", "y"]
    assert np.abs(rows[:, 0]).max() < 1e-6          # KL of the reference against itself (f16 storage only)
    assert rows[:, 3].tolist() == [1.0] * 12          # top-1 agrees everywhere
    assert np.abs(rows[:, 1] - rows[:, 2]).max() < 1e-5


def test_radiance_scores_a_perturbed_candidate_at_the_true_kl(oracle, record, inputs, tmp_path):
    """Positive control: a candidate that differs scores its f64 KL, within the f16 storage error."""
    hidden, w = inputs["hidden"], inputs["w"]
    spans = (inputs["spans"][0], inputs["spans"][2])
    ref = np.concatenate([hidden[a:b] for a, b in spans]) @ w.T
    cand = ref + np.random.default_rng(2).normal(0, 0.5, ref.shape).astype(np.float32)
    _, rows = run_score(oracle, record, V, cand, tmp_path / "pert.json")
    keep = [o + p for o, n in zip([0, 7], [7, 9]) for p in range(1, n - 1)]

    def lsm(x):
        x = x.astype(np.float64)
        return x - (x.max(1, keepdims=True) + np.log(np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True)))
    lp, lq = lsm(ref[keep]), lsm(cand[keep])
    kl = (np.exp(lp) * (lp - lq)).sum(1)
    assert kl.mean() > 0.05
    assert np.abs(rows[:, 0] - kl).max() < 2e-3 * kl.max()
