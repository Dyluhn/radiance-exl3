# Architecture port checks

These checks cover the out-of-tree architecture plugin against radiance 1.7.0 (ABI 17).
Run from the repository root; `PREFIX` names an installed radiance 1.7.0 prefix.

| Requirement | Check |
|---|---|
| Builds using only the installed plugin contract | `cmake -S architectures/qwen4exp_exl3 -B build-host/arch-oot -DCMAKE_PREFIX_PATH="$PREFIX" && cmake --build build-host/arch-oot -j 8` |
| Declares EXL3 planes, per-rank spans, protected experts and the MTP head correctly | `ctest --test-dir build-host -R '^arch_static' --output-on-failure` |
| Preserves mixed recurrent steps and the upstream prefill pipeline, including lane joins and separately issued collectives | `ctest --test-dir build-host -R '^arch_static' --output-on-failure` |
| The harness catches broken wiring and restores every mutated source | `python3 tests/arch/mutation_check.py build-host/tests/arch && git status --short` |
| Host-only build works without a GPU | `cmake -S . -B build-host -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_BUILD_TYPE=Release -DRAD_WITH_HIP=OFF && cmake --build build-host -j 8 && ctest --test-dir build-host -L host --output-on-failure` |
| The full HIP build has no warnings | Request the documented `.hip-build-request` bridge; require `exit 0` in `.hip-build-result` and zero compiler warnings in `.hip-build.log`. |
| Copies account for every upstream change and retain EXL3 behavior | Review the 1.6.0-to-1.7.0 diffs of both files and `arch/common/`; compare the pinned hashes with `sha256sum` of the two 1.7.0 source files. |
| Changed source contains no machine paths or device assumptions | `rg -n '/var/home|/home/dylan|qwen-storage|localhost|127\\.0\\.0\\.1|13:00\\.0|0000:|blackbox|R9700' architectures/qwen4exp_exl3 tests/arch` (review existing test defaults separately). |

GPU numerical, concurrency, tensor-parallel and repeatability checks require the coordinator's GPU runs.
Results and limits of these checks are recorded in `NOTES.md` after validation.
