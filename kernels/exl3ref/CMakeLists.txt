# exl3ref -- the host reference for the EXL3 routed-expert ops (moe_exl3_gate_up, moe_exl3_down).
#
# Builds on its own against an installed radiance (REQUIREMENTS R9), the same rad_add_plugin either
# way, as radiance's own libraries and RidgeFill's kernels/ do:
#
#   cmake -S kernels/exl3ref -B build-exl3ref -DCMAKE_PREFIX_PATH=<radiance install prefix>
#   cmake --build build-exl3ref && ctest --test-dir build-exl3ref -L host
#
# -DRADIANCE_SRC=<radiance source at the installed release> adds the docs check (ctest
# exl3ref_ops_doc: docs/OPS-exl3.md against what the library declares, by radiance's own
# tools/ops_doc_check.sh); without it that test reports SKIPPED. Host code only: there is nothing
# here for a device to run.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(exl3ref LANGUAGES C CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  enable_testing()
endif()
enable_language(C)

# R1: refuse at configure time what the loader would refuse at dlopen.
if(NOT RADIANCE_ABI_VERSION EQUAL 27)
  message(FATAL_ERROR "exl3ref is written against radiance ABI 27; the radiance found is ABI "
                      "${RADIANCE_ABI_VERSION}")
endif()

get_filename_component(EXL3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
foreach(_f include/exl3_abi.h include/exl3_encoding.h common/exl3_decode.c)
  if(NOT EXISTS "${EXL3_ROOT}/${_f}")
    message(FATAL_ERROR "exl3ref needs ${EXL3_ROOT}/${_f} (the repository around kernels/exl3ref)")
  endif()
endforeach()

# common/exl3_decode.c is compiled in, as every plugin that decodes does (common/CMakeLists.txt).
rad_add_plugin(exl3ref kernels
  SOURCES rows.c ref.c ${EXL3_ROOT}/common/exl3_decode.c
  INCLUDES ${EXL3_ROOT}/include ${EXL3_ROOT}/common)
set_target_properties(exl3ref PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
# -ffp-contract=off on the host code: a product and a sum are two roundings in the reference
# contract (abi/rad_plugin.h), and an FMA would make "the reference" depend on the compiler.
target_compile_options(exl3ref PRIVATE $<$<COMPILE_LANGUAGE:C>:-ffp-contract=off -Wall -Wextra>)
find_library(EXL3REF_LIBM m)
if(EXL3REF_LIBM)
  target_link_libraries(exl3ref PRIVATE ${EXL3REF_LIBM})
endif()

# ------------------------------------------------------------------ tests (tests/ref)
#
# Two TEST plugins, never installed: exl3ref's rows behind a layout hook that stores the codes planes
# XOR 0xFF, and the same with an unrelayout that does not invert -- the positive and the negative
# control for the harness's layout path, which a device library's check runs through.
set(RAD_PLUGIN_INSTALL OFF)
foreach(_probe layout_probe layout_probe_badinv)
  rad_add_plugin(${_probe} test
    SOURCES ${EXL3_ROOT}/tests/ref/layout_probe.c ref.c ${EXL3_ROOT}/common/exl3_decode.c
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${EXL3_ROOT}/include ${EXL3_ROOT}/common)
  set_target_properties(${_probe} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
  target_compile_options(${_probe} PRIVATE -ffp-contract=off -Wall -Wextra)
  if(EXL3REF_LIBM)
    target_link_libraries(${_probe} PRIVATE ${EXL3REF_LIBM})
  endif()
endforeach()
target_compile_definitions(layout_probe_badinv PRIVATE LAYOUT_PROBE_BAD_INVERSE=1)
unset(RAD_PLUGIN_INSTALL)

#
# exl3ref_run: loads the plugin as the engine does (dlopen, the row table, opd_shape, launch) and
# runs one case written by tests/ref/test_exl3ref.py, which compares the result with an independent
# NumPy transcription (tests/ref/exl3_numpy.py). A case that cannot run prints SKIP; a suite that
# checked nothing exits 77, which ctest reports as skipped.
set(EXL3REF_TESTS ${EXL3_ROOT}/tests/ref)
add_executable(exl3ref_run ${EXL3REF_TESTS}/exl3ref_run.cpp)
target_link_libraries(exl3ref_run PRIVATE radiance::abi ${CMAKE_DL_LIBS})
target_include_directories(exl3ref_run PRIVATE ${EXL3_ROOT}/include ${EXL3_ROOT}/common)
add_dependencies(exl3ref_run exl3ref)
# With a device build of radiance the same harness runs a library's DEVICE rows (`--domain device`,
# the exl3r4d check against this reference); without one it says SKIP and exits 77 when asked.
if(RAD_HIP_FOUND AND NOT (DEFINED RAD_WITH_HIP AND RAD_WITH_HIP STREQUAL "OFF"))
  target_compile_definitions(exl3ref_run PRIVATE EXL3_RUN_HIP=1)
  target_link_libraries(exl3ref_run PRIVATE ${RAD_HIP_RUNTIME})
endif()

find_program(EXL3REF_PYTHON NAMES python3 python)
if(EXL3REF_PYTHON)
  set(_py ${EXL3REF_PYTHON})
else()
  set(_py sh -c "echo SKIPPED: no python3 interpreter; exit 77" --)
endif()
set(_suite ${_py} ${EXL3REF_TESTS}/test_exl3ref.py --harness $<TARGET_FILE:exl3ref_run>
           --plugin $<TARGET_FILE:exl3ref> --golden ${EXL3_ROOT}/common/tests/data/exl3_golden.npz
           --work ${CMAKE_CURRENT_BINARY_DIR}/cases
           --probes $<TARGET_FILE:layout_probe> $<TARGET_FILE:layout_probe_badinv>)
add_test(NAME exl3ref_cases COMMAND ${_suite} --suite cases)
add_test(NAME exl3ref_band_tops COMMAND ${_suite} --suite band-tops)
set_tests_properties(exl3ref_cases PROPERTIES LABELS host SKIP_RETURN_CODE 77)
set_tests_properties(exl3ref_band_tops PROPERTIES LABELS "host;slow" SKIP_RETURN_CODE 77)

# The schemas this library exports are include/exl3_abi.h's, field for field.
add_test(NAME exl3ref_schemas COMMAND exl3ref_run $<TARGET_FILE:exl3ref> --schemas)
set_tests_properties(exl3ref_schemas PROPERTIES LABELS host)

set(RADIANCE_SRC "" CACHE PATH "radiance source checkout at the installed release (for the docs check)")
get_filename_component(_rad_prefix "${RADIANCE_INCLUDE_DIR}/../.." ABSOLUTE)
set(_rad_schemas "${_rad_prefix}/bin/rad-schemas")
if(RADIANCE_SRC AND EXISTS "${RADIANCE_SRC}/tools/ops_doc_check.sh")
  add_test(NAME exl3ref_ops_doc
           COMMAND sh ${RADIANCE_SRC}/tools/ops_doc_check.sh ${EXL3_ROOT}/docs/OPS-exl3.md
                   ${_rad_schemas} $<TARGET_FILE:exl3ref>)
else()
  add_test(NAME exl3ref_ops_doc
           COMMAND sh -c "echo 'SKIPPED: configure with -DRADIANCE_SRC=<radiance source> for the docs check'; exit 77")
endif()
set_tests_properties(exl3ref_ops_doc PROPERTIES LABELS host SKIP_RETURN_CODE 77)
