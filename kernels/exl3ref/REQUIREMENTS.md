# exl3ref port checks

Run from the repository root with `RADIANCE_PREFIX` naming the installed pinned release.

| Requirement | Check |
|---|---|
| Build alone against ABI 17, with HIP disabled even when the prefix was built with HIP | `cmake -S kernels/exl3ref -B build-ref -DCMAKE_PREFIX_PATH="$RADIANCE_PREFIX" -DRAD_WITH_HIP=OFF && cmake --build build-ref` |
| Preserve the reference arithmetic, schemas, optional operands and band tops | `ctest --test-dir build-ref -L host --output-on-failure` |

The standalone configure command reproduced a failure on a host without ROCm: the harness linked
`RAD_HIP_RUNTIME_HERE-NOTFOUND` despite `RAD_WITH_HIP=OFF`. The result after the fix is recorded in
`NOTES.md`. GPU execution, concurrent serving and tensor-parallel engine checks belong to the device
library and the coordinator's hardware checks.
