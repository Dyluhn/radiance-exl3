# Where this plugin's code comes from

`qwen4exp_exl3.cpp` and `rad_block_moe_exl3.h` are COPIES of two radiance files, changed only where
EXL3 experts need it. Radiance is not modified (PLAN D11): the copies are built out of tree against an
installed radiance of the pinned release (`RADIANCE_VERSION` at the repository root).

| this directory | radiance file it was copied from |
|---|---|
| `qwen4exp_exl3.cpp`   | `arch/qwen4exp_fp8/qwen4exp_fp8.cpp` |
| `rad_block_moe_exl3.h` | `arch/common/rad_block_moe_fp8.h` |

Release `v1.3.0`, commit `89cee7ce414486c717aa2085c99e1f522e4c0ea6`, fetched from
`https://codeberg.org/StillDeadcode/radiance.git`. The first commit that added these files holds them
byte for byte as upstream ships them; every later change is in git history, so
`git log -p -- architectures/qwen4exp_exl3/` is the whole port.

## The pinned bytes

`tests/upstream_drift.sh` reads this block (`sha256  path`, the `sha256sum` format) and fails if the
release it fetches holds anything else at those paths:

```upstream-sha256
79766712ff673362c878c6f4835b2dd87630c38fccf5edcf8d6a2998829a9c11  arch/qwen4exp_fp8/qwen4exp_fp8.cpp
88f5b050d9997b8285b4e691e96078932efb4c5ed9e29e41b1f9c8bc7d54c1e4  arch/common/rad_block_moe_fp8.h
```

## Called, not copied

Every other block the model uses -- the gated residual, attention, delta net, PLE, QSA and the MTP
head (`arch/common/rad_block_mtp_hc.h`) -- is the INSTALLED header, compiled as it ships. The MTP head
names the fp8 MoE block as `MoeFP8`; in this plugin's translation unit that name is an alias for
`MoeEXL3` (see the includes in `qwen4exp_exl3.cpp`), so the head's routed experts are EXL3 with no
copy of the head. `CMakeLists.txt` refuses an installed radiance that is not the pinned release, so
a release bump is always deliberate: run the drift test, and read the release's changes to
`arch/common/` -- a head that asks the block for more than `MoeEXL3` has fails to compile here.

## Moving to a new radiance release

1. Put the new release in `RADIANCE_VERSION`.
2. `tests/upstream_drift.sh` names each copied file that changed and prints the `git diff` to port.
3. Port that diff into the copy here, then replace the hash above with the new release's.
