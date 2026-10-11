"""recipes/qwen4exp-exl3.recipe against the author's served recipe in the radiance source: exactly one rule differs
(the routed experts', now exl3), the protected experts still decide first, MTP experts are covered, the shared
expert is not. Needs the radiance 1.3.0 source: $RADIANCE_SRC, else <repo>/data/radiance-src; SKIPPED without it."""
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
OURS = REPO / "recipes" / "qwen4exp-exl3.recipe"
AUTHOR_REL = Path("data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe")
EXL3_RULE = ["blk.*.ffn_*_exps.*.weight", "exl3", "src=$EXL3_SRC", "bits=4", "codebook=mul1"]


def author_recipe() -> str:
    src = Path(os.environ.get("RADIANCE_SRC", REPO / "data" / "radiance-src"))
    if not (src / AUTHOR_REL).is_file():
        pytest.skip(f"SKIPPED: no radiance source with {AUTHOR_REL} (set RADIANCE_SRC)")
    return (src / AUTHOR_REL).read_text()


def rules(text: str) -> list[list[str]]:
    """core/format/recipe.cpp's reading: '#' to end of line is a comment, words split on whitespace."""
    return [line.split("#", 1)[0].split() for line in text.splitlines() if line.split("#", 1)[0].split()]


def glob(pattern: str, name: str) -> bool:
    """recipe_glob: '*' any run (dots included), '?' one character, everything else literal."""
    rx = "".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern)
    return re.fullmatch(rx, name) is not None


def decides(rs: list[list[str]], name: str) -> list[str] | None:
    return next((r for r in rs if glob(r[0], name)), None)


def differing(a: list[list[str]], b: list[list[str]]) -> list[int]:
    assert len(a) == len(b), f"{len(a)} rules vs {len(b)}"
    return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]


def test_exactly_the_routed_expert_rule_differs():
    a, b = rules(author_recipe()), rules(OURS.read_text())
    (i,) = differing(a, b)
    assert a[i][:2] == ["blk.*.ffn_*_exps.*.weight", "gptq"]
    assert b[i] == EXL3_RULE


def test_the_diff_check_sees_a_change_to_any_other_rule():
    """Negative control: one non-expert option changed in a copy is a second difference."""
    a, b = rules(author_recipe()), rules(OURS.read_text())
    mutated = [r[:] for r in b]
    j = next(k for k, r in enumerate(mutated) if r[0] == "output.weight")
    mutated[j][3] = "group=64"
    assert len(differing(a, mutated)) == 2


PROTECTED = [(34, 407), (34, 496), (44, 292), (44, 350), (46, 290), (46, 392),
             (47, 122), (47, 143), (47, 399), (47, 445)]


@pytest.mark.parametrize("proj", ["gate_up", "down"])
def test_protected_experts_stay_bf16(proj):
    rs = rules(OURS.read_text())
    for layer, expert in PROTECTED:
        assert decides(rs, f"blk.{layer}.ffn_{proj}_exps.{expert}.weight")[:3] == ["blk.%d.ffn_*_exps.%d.weight"
                                                                                    % (layer, expert), "cast",
                                                                                    "dtype=bf16"]


@pytest.mark.parametrize("name", ["blk.0.ffn_gate_up_exps.0.weight", "blk.47.ffn_down_exps.444.weight",
                                  "blk.34.ffn_gate_up_exps.406.weight",
                                  "blk.48.ffn_gate_up_exps.0.weight", "blk.48.ffn_down_exps.511.weight"])
def test_routed_experts_trunk_and_mtp_are_exl3(name):
    assert decides(rules(OURS.read_text()), name) == EXL3_RULE


@pytest.mark.parametrize("name", ["blk.0.ffn_gate_up_shexp.weight", "blk.48.ffn_down_shexp.weight",
                                  "blk.0.ffn_gate_inp.weight", "output.weight"])
def test_nothing_else_is_exl3(name):
    r = decides(rules(OURS.read_text()), name)
    assert r is None or r[1] != "exl3"


def test_no_machine_path_in_the_recipe():
    text = OURS.read_text()
    assert not re.search(r"/(var|home|mnt|tmp)/|qwen-storage|~/", text)
