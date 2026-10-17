# radiance 1.7.0 port

Checked against radiance 1.7.0 (`1e968f6`, ABI 17) on 2026-10-17, using GCC 16.2.1,
Release builds and the installed release prefix on a host without ROCm.

The ABI removes `RadKernelInfo.describe` and `RadLaunchDesc`. This library never used them and
initializes its rows by field name. The new joined-plane helpers do not apply: EXL3 passes each
plane as its own operand. The reference arithmetic and supported shapes stay unchanged.

The standalone host check in `REQUIREMENTS.md` failed before the CMake fix because the harness
linked `RAD_HIP_RUNTIME_HERE-NOTFOUND`. After the fix, configure, build and install exit 0 with
zero compiler or CMake warnings; `ctest -R '^exl3ref_schemas$'` passes 1/1. This uses the same
`RAD_WITH_HIP=OFF` choice the top-level build already applies. It affects the test harness only.

The whole-port audit and remaining hardware checks are in `docs/PORTING-1.7.0.md` at the repository root.
There were no tuning experiments or rejected kernel variants in this port.
