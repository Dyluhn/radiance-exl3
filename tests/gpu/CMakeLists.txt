# tests/gpu -- exl3r4d's checks (added by kernels/exl3r4d/CMakeLists.txt).
#
#   exl3r4d_layout       host  the stored codes form against common/exl3_layout, the hooks' claims and refusals
#   exl3r4d_device       gpu   the device rows against the NumPy transcription of the ops (tests/ref), through P2's
#                              engine-style harness (exl3ref_run built with HIP): poisoned margins, inputs unchanged,
#                              the layout hooks' relayout -> unrelayout round trip, controls, refusals, replay
#   exl3r4d_real_layer   gpu   a real turboderp 4-bit tensor (EXL3_REAL_EXTRACT) against decode-and-multiply
#
# A gpu test runs on whatever card the process sees: pin it (HIP_VISIBLE_DEVICES) and take the project's GPU lock
# around ctest yourself -- tests/gpu/run_gpu.sh does both, with the squatter check and the kernel-log watchdog.
# Without numpy, the HIP build or the extract, a test reports SKIPPED (exit 77), never passed.
set(EXL3R4D_DIR ${EXL3_ROOT}/kernels/exl3r4d)

add_executable(exl3r4d_layout_test layout_test.c ${EXL3R4D_DIR}/layout.c ${EXL3_ROOT}/common/exl3_decode.c
               ${EXL3_ROOT}/common/exl3_layout.c)
set_target_properties(exl3r4d_layout_test PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
target_compile_options(exl3r4d_layout_test PRIVATE -ffp-contract=off -Wall -Wextra -O2)
target_include_directories(exl3r4d_layout_test PRIVATE ${EXL3R4D_DIR} ${EXL3_ROOT}/include ${EXL3_ROOT}/common)
find_library(EXL3R4D_LIBM m)   # absent (and not needed) where libm is part of libc
target_link_libraries(exl3r4d_layout_test PRIVATE radiance::abi $<$<BOOL:${EXL3R4D_LIBM}>:${EXL3R4D_LIBM}>)
add_test(NAME exl3r4d_layout COMMAND exl3r4d_layout_test)
set_tests_properties(exl3r4d_layout PROPERTIES LABELS host)

if(EXL3R4D_HIP)
  # run_gpu.sh's proof of which card a session ran on (host API only)
  add_executable(exl3r4d_devcheck devcheck.cpp)
  target_link_libraries(exl3r4d_devcheck PRIVATE radiance::abi ${RAD_HIP_RUNTIME})
endif()

find_program(EXL3R4D_PYTHON NAMES python3 python)
set(EXL3_REAL_EXTRACT "" CACHE FILEPATH
    "a safetensors file holding turboderp 4.05 tensors (layer 15's indexer.index_qk_proj, shared_expert.down_proj)")
if(EXL3R4D_HIP AND TARGET exl3ref_run AND EXL3R4D_PYTHON)
  set(_drv ${EXL3R4D_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/test_exl3r4d.py --harness $<TARGET_FILE:exl3ref_run>
           --plugin $<TARGET_FILE:exl3r4d> --work ${CMAKE_CURRENT_BINARY_DIR}/work)
  add_test(NAME exl3r4d_device COMMAND ${_drv} --suite device)
  add_test(NAME exl3r4d_real_layer COMMAND ${_drv} --suite real --extract "${EXL3_REAL_EXTRACT}")
else()
  set(_why "SKIPPED: no device build of exl3r4d, no exl3ref_run, or no python3")
  add_test(NAME exl3r4d_device COMMAND sh -c "echo '${_why}'; exit 77")
  add_test(NAME exl3r4d_real_layer COMMAND sh -c "echo '${_why}'; exit 77")
endif()
set_tests_properties(exl3r4d_device exl3r4d_real_layer PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
