# tests/gpu/prefill -- exl3r4d's M > 16 rows (P5; added by kernels/exl3r4d/CMakeLists.txt).
#
#   exl3r4d_prefill   gpu   the M > 16 rows against the NumPy transcription through P2's engine-style harness (poisoned
#                           margins, inputs unchanged, the layout round trip, controls, replay) at M = 17 .. 4096;
#                           three phases (plan on the host, the launches, the check) -- test_prefill.py says why.
#
# rowpick: a TEST-ONLY shim plugin exposing one library's rows by name (the harness takes an op's first row). Never
# installed. A gpu test runs on whatever card the process sees: pin it and hold the project's GPU lock yourself, or
# use tests/gpu/run_gpu.sh with tests/gpu/prefill/run_cases.sh (what P5's sessions do).
add_library(exl3r4d_rowpick MODULE rowpick.c)
target_link_libraries(exl3r4d_rowpick PRIVATE radiance::abi ${CMAKE_DL_LIBS})
set_target_properties(exl3r4d_rowpick PROPERTIES PREFIX "" OUTPUT_NAME rowpick C_STANDARD 11 C_STANDARD_REQUIRED ON)
target_compile_options(exl3r4d_rowpick PRIVATE -Wall -Wextra)

if(EXL3R4D_HIP AND TARGET exl3ref_run AND EXL3R4D_PYTHON)
  set(_w ${CMAKE_CURRENT_BINARY_DIR}/work)
  set(_py ${EXL3R4D_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/test_prefill.py --work ${_w})
  add_test(NAME exl3r4d_prefill COMMAND sh -c
    "${_py} --mode plan && sh ${CMAKE_CURRENT_SOURCE_DIR}/run_cases.sh ${_w} $<TARGET_FILE:exl3ref_run> $<TARGET_FILE:exl3r4d_rowpick> $<TARGET_FILE:exl3r4d> && ${_py} --mode check")
else()
  add_test(NAME exl3r4d_prefill COMMAND sh -c "echo 'SKIPPED: no device build of exl3r4d, no exl3ref_run, or no python3'; exit 77")
endif()
set_tests_properties(exl3r4d_prefill PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
