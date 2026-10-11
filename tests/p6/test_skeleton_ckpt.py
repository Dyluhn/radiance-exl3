"""tools/skeleton_ckpt.py on a small synthetic checkpoint: headers kept byte for byte, data all holes, and each
refusal by name. Run: python3 -m pytest tests/p6 (stdlib only; the Hub is replaced by a local directory)."""
import json
import os
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import skeleton_ckpt as sk  # noqa: E402

MiB = 1 << 20


def shard_bytes(tensors: dict, fill: int) -> bytes:
    """A safetensors file: {name: (dtype, shape)}, every data byte `fill` (non-zero: a real file)."""
    header, at = {"__metadata__": {"format": "pt"}}, 0
    for name, (dtype, shape) in tensors.items():
        n = sk.math.prod(shape) * sk.hf.ITEMSIZE[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + n]}
        at += n
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw + bytes([fill]) * at


SHARDS = {
    "model-00001-of-00002.safetensors": {"a.weight": ("BF16", [1024, 1024]), "a.bias": ("F32", [1024])},
    "model-00002-of-00002.safetensors": {"b.experts.gate_up_proj": ("BF16", [4, 512, 512]),
                                         "b.ids": ("I64", [3])},
}


def make_src(d: Path, with_shards=True) -> Path:
    d.mkdir()
    weight_map = {n: s for s, ts in SHARDS.items() for n in ts}
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (d / "config.json").write_text('{"model_type": "toy"}')
    (d / "tokenizer.json").write_text('{"model": {}}')
    (d / ".cache").mkdir()
    (d / "model-00001-of-00002.safetensors.fetch.json").write_text("{}")
    if with_shards:
        for s, ts in SHARDS.items():
            (d / s).write_bytes(shard_bytes(ts, 0x5A))
    return d


def header_len(path: Path) -> int:
    return 8 + struct.unpack("<Q", path.read_bytes()[:8])[0]


def test_every_header_is_kept_and_every_data_byte_is_a_hole(tmp_path):
    src = make_src(tmp_path / "src")
    out = tmp_path / "out"
    assert sk.main([str(src), str(out)]) == 0
    for s in SHARDS:
        real, skel = (src / s).read_bytes(), (out / s).read_bytes()
        h = header_len(src / s)
        assert len(skel) == len(real)
        assert skel[:h] == real[:h]
        assert skel[h:] == bytes(len(real) - h)
    apparent, on_disk = sk.disk_usage(out)
    _, src_on_disk = sk.disk_usage(src)
    assert apparent > 2 * MiB
    assert src_on_disk >= apparent - 64 * 1024     # positive control: a written file does occupy its blocks
    assert on_disk <= 64 * 1024                     # the skeleton: headers only
    for f in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        assert (out / f).read_bytes() == (src / f).read_bytes()
    assert not (out / ".cache").exists()
    assert not (out / "model-00001-of-00002.safetensors.fetch.json").exists()
    assert json.loads((out / "skeleton.json").read_text())["revision"] is None


def test_rerun_over_its_own_skeleton_is_idempotent(tmp_path):
    src = make_src(tmp_path / "src")
    out = tmp_path / "out"
    sk.main([str(src), str(out)])
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    assert sk.main([str(src), str(out)]) == 0
    assert {p.name: p.read_bytes() for p in out.iterdir()} == first


def test_absent_shard_header_comes_from_the_hub(tmp_path, monkeypatch):
    remote = make_src(tmp_path / "remote")
    src = make_src(tmp_path / "src", with_shards=False)
    calls = []

    def fake_hub_header(repo, sha, shard):
        calls.append((repo, sha, shard))
        return sk.local_header(remote / shard), (remote / shard).stat().st_size

    def fake_get_json(url):
        if "/api/models/" in url:
            return {"sha": "c0ffee"}
        return json.loads((remote / "model.safetensors.index.json").read_text())

    monkeypatch.setattr(sk, "hub_header", fake_hub_header)
    monkeypatch.setattr(sk.hf, "get_json", fake_get_json)
    out = tmp_path / "out"
    assert sk.main([str(src), str(out), "--repo", "org/toy", "--revision", "main"]) == 0
    assert sorted(c[2] for c in calls) == sorted(SHARDS)
    assert all(c[:2] == ("org/toy", "c0ffee") for c in calls)
    for s in SHARDS:
        h = header_len(remote / s)
        assert (out / s).read_bytes()[:h] == (remote / s).read_bytes()[:h]
        assert (out / s).stat().st_size == (remote / s).stat().st_size
    assert json.loads((out / "skeleton.json").read_text())["revision"] == "c0ffee"


def test_absent_shard_without_a_repo_is_refused_by_name(tmp_path):
    src = make_src(tmp_path / "src", with_shards=False)
    with pytest.raises(SystemExit, match="model-00001-of-00002.safetensors.*--repo and --revision"):
        sk.main([str(src), str(tmp_path / "out")])


def test_hub_index_of_another_revision_is_refused(tmp_path, monkeypatch):
    src = make_src(tmp_path / "src", with_shards=False)
    monkeypatch.setattr(sk.hf, "get_json", lambda url: {"sha": "c0ffee"} if "/api/" in url
                        else {"weight_map": {"a.weight": "model-00001-of-00002.safetensors"}})
    with pytest.raises(SystemExit, match="differs from"):
        sk.main([str(src), str(tmp_path / "out"), "--repo", "org/toy", "--revision", "main"])


def test_a_name_the_index_puts_in_a_shard_whose_header_lacks_it_is_refused(tmp_path):
    src = make_src(tmp_path / "src")
    ix = json.loads((src / "model.safetensors.index.json").read_text())
    ix["weight_map"]["c.weight"] = "model-00002-of-00002.safetensors"
    (src / "model.safetensors.index.json").write_text(json.dumps(ix))
    with pytest.raises(SystemExit, match="c.weight.*model-00002-of-00002.safetensors.*does not"):
        sk.main([str(src), str(tmp_path / "out")])


def test_a_span_that_disagrees_with_its_shape_is_refused(tmp_path):
    src = make_src(tmp_path / "src")
    bad = shard_bytes({"a.weight": ("BF16", [1024, 1024]), "a.bias": ("F32", [1024])}, 1)
    bad = bad.replace(b'"shape": [1024, 1024]', b'"shape": [1024, 1023]')
    (src / "model-00001-of-00002.safetensors").write_bytes(bad)
    with pytest.raises(SystemExit, match="a.weight spans 2097152 B; shape \\[1024, 1023\\]"):
        sk.main([str(src), str(tmp_path / "out")])


def test_a_truncated_local_shard_is_refused(tmp_path):
    src = make_src(tmp_path / "src")
    p = src / "model-00002-of-00002.safetensors"
    os.truncate(p, p.stat().st_size - 1)
    with pytest.raises(SystemExit, match="model-00002-of-00002.safetensors \\(local\\) is .* its header says"):
        sk.main([str(src), str(tmp_path / "out")])


def test_writing_into_a_directory_it_did_not_write_is_refused(tmp_path):
    src = make_src(tmp_path / "src")
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.bin").write_bytes(b"x")
    with pytest.raises(SystemExit, match="is not a skeleton this tool wrote"):
        sk.main([str(src), str(out)])
    assert (out / "keep.bin").read_bytes() == b"x"


def test_out_equal_to_src_is_refused(tmp_path):
    src = make_src(tmp_path / "src")
    (src / "skeleton.json").write_text("{}")        # even one that looks like a skeleton
    with pytest.raises(SystemExit, match="OUT is SRC"):
        sk.main([str(src), str(src)])


def test_check_whole_passes_a_complete_checkpoint(tmp_path, capsys):
    src = make_src(tmp_path / "src")
    assert sk.main([str(src), "--check-whole"]) == 0
    assert "is whole" in capsys.readouterr().out


def test_check_whole_refuses_its_own_skeleton_as_sparse(tmp_path):
    src = make_src(tmp_path / "src")
    sk.main([str(src), str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="model-00001-of-00002.safetensors holds .* sparse, so not whole"):
        sk.main([str(tmp_path / "out"), "--check-whole"])


def test_check_whole_refuses_a_missing_shard(tmp_path):
    src = make_src(tmp_path / "src")
    (src / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(SystemExit, match="has no model-00002-of-00002.safetensors"):
        sk.main([str(src), "--check-whole"])
