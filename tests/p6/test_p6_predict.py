"""tools/p6_predict.py on hand-written listings in rad-convert's and rad-info's formats (radiance 1.3.0)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import p6_predict as pp  # noqa: E402

GPTQ = ("affine:codes=u4[1x1],scale=fp8_e4m3[1x64],scale.1=f32[*x*],table=f32{1x16}/fwht128",
        "gptq block2=*x*,calib=calib/w4nl-calib,cd=3,group=64")
EXL3_GU = "exl3:codes=u4[1x1],suh=f16[*x1],svh=f16[1x*],suh.1=f16[*x1]/mul1/h128"
EXL3_DN = "exl3:codes=u4[1x1],suh=f16[*x1],svh=f16[1x*]/mul1/h128"
EXL3_RULE = "exl3 bits=4,codebook=mul1,src=turboderp/Qwen3.8-Flash-Next-exl3"

BASELINE = f"""\
  data          20971520   113.57 GiB    4
      blk.0.attn_hc_norm.weight                    bf16                           10240               20.00 KiB  @20971520     checkpoint
      output.weight                                i8*bf16[1x128]                 248320x2560        615.72 MiB  @20992000  rtn clamp=sym,codes=i8,group=128,rule=search,scale=bf16
      blk.0.ffn_gate_up_exps.0.weight              {GPTQ[0]} 1280x2560            1.61 MiB  @666644480  {GPTQ[1]}
          plane codes        u4            1.56 MiB  @666644480
      blk.0.ffn_down_exps.0.weight                 {GPTQ[0]} 2560x640           825.31 KiB  @668336128  {GPTQ[1]}
"""
BASELINE_BYTES = 668336128 + 847872


def plan_log(output_rule="rtn clamp=sym,codes=i8,group=128,rule=search,scale=bf16", count=4):
    return f"""\
I plan     {count} weight(s): 3 quantised by the recipe, 1 kept as the checkpoint holds them
D rad_convert.cpp:723    blk.0.attn_hc_norm.weight                            bf16                                20.00 KiB  as is
D rad_convert.cpp:723    output.weight                                        i8*bf16[1x128]                     615.72 MiB  {output_rule}
D rad_convert.cpp:723    blk.0.ffn_gate_up_exps.0.weight                      {EXL3_GU}  1.57 MiB  {EXL3_RULE}
D rad_convert.cpp:723    blk.0.ffn_down_exps.0.weight                         {EXL3_DN}  806.25 KiB  {EXL3_RULE}
""".splitlines()


def run(plan_lines):
    base, data0 = pp.read_baseline(BASELINE.splitlines(), BASELINE_BYTES)
    return pp.predict(pp.read_plan(plan_lines), base, data0)


def test_unchanged_weights_are_reused_at_their_baseline_footprint_and_experts_sized_as_the_writer_lays_them():
    r = run(plan_log())
    assert (r["reused"], r["written"], r["unexpected"]) == (2, 2, [])
    assert r["reused_bytes"] == 20480 + 645652480          # offset deltas in the baseline
    # gate_up: codes 1280*2560/2 + suh 5120 + suh.1 5120 + svh 2560 -> 1,651,200 -> 4 KiB: 1,654,784 (docs/STEP0.md S0.12)
    # down: codes 2560*640/2 + suh 1280 + svh 5120 -> 825,600 -> 827,392
    assert r["written_bytes"] == 1654784 + 827392
    assert r["predicted_bytes"] == 20971520 + r["reused_bytes"] + r["written_bytes"]


def test_a_non_expert_weight_whose_options_changed_is_not_reused_and_is_flagged():
    r = run(plan_log(output_rule="rtn clamp=sym,codes=i8,group=64,rule=search,scale=bf16"))
    assert r["reused"] == 1
    assert len(r["unexpected"]) == 1 and r["unexpected"][0].startswith("output.weight: made by 'rtn'")


def test_a_plan_log_without_every_weight_listed_is_refused():
    with pytest.raises(SystemExit, match="says 5 weight\\(s\\) and lists 4"):
        pp.read_plan(plan_log(count=5))


def test_an_encoding_it_cannot_size_is_flagged_not_guessed():
    lines = [ln.replace(EXL3_DN, "exl3:codes=q4[1x1]/mul1/h128") for ln in plan_log()]
    r = run(lines)
    assert r["unexpected"] == ["blk.0.ffn_down_exps.0.weight: cannot size encoding 'exl3:codes=q4[1x1]/mul1/h128'"]
