# tests/arch -- the qwen4exp_exl3 plugin's tests. Inside the repository's top-level project, or on
# its own against an installed radiance:
#
#   cmake -S tests/arch -B build -DCMAKE_PREFIX_PATH=<radiance install prefix> && cmake --build build
#   ctest --test-dir build -LE net        # host-only, seconds
#
# Labels: `host` needs nothing but the build; `net` fetches the pinned radiance tag (it skips, never
# passes, when it cannot).
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(qwen4exp_exl3_tests LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  enable_testing()
endif()
if(NOT TARGET qwen4exp_exl3)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../architectures/qwen4exp_exl3 qwen4exp_exl3)
endif()
set(_qx_repo "${CMAKE_CURRENT_LIST_DIR}/../..")

# ---- the declared graph and the MoE block's issues, against a recording builder (no core, no GPU).
# Built at both issue orders (QWEN4EXP_EXL3_PREP_SPLIT, rad_block_moe_exl3.h): every case holds for both.
add_executable(arch_static_test arch_static_test.cpp)
target_link_libraries(arch_static_test PRIVATE radiance::abi)
target_compile_definitions(arch_static_test PRIVATE QWEN4EXP_EXL3_PREP_SPLIT=0)
add_test(NAME arch_static
         COMMAND sh ${CMAKE_CURRENT_LIST_DIR}/run_cases.sh $<TARGET_FILE:arch_static_test>)
set_tests_properties(arch_static PROPERTIES LABELS host SKIP_RETURN_CODE 77)
add_executable(arch_static_test_split arch_static_test.cpp)
target_link_libraries(arch_static_test_split PRIVATE radiance::abi)
target_compile_definitions(arch_static_test_split PRIVATE QWEN4EXP_EXL3_PREP_SPLIT=1)
add_test(NAME arch_static_split
         COMMAND sh ${CMAKE_CURRENT_LIST_DIR}/run_cases.sh $<TARGET_FILE:arch_static_test_split>)
set_tests_properties(arch_static_split PROPERTIES LABELS host SKIP_RETURN_CODE 77)
# And with the decode chain as one op (QWEN4EXP_EXL3_GD 1, interface v5) over the split build.
add_executable(arch_static_test_gd arch_static_test.cpp)
target_link_libraries(arch_static_test_gd PRIVATE radiance::abi)
target_compile_definitions(arch_static_test_gd PRIVATE QWEN4EXP_EXL3_PREP_SPLIT=1 QWEN4EXP_EXL3_GD=1)
add_test(NAME arch_static_gd
         COMMAND sh ${CMAKE_CURRENT_LIST_DIR}/run_cases.sh $<TARGET_FILE:arch_static_test_gd>)
set_tests_properties(arch_static_gd PROPERTIES LABELS host SKIP_RETURN_CODE 77)
# And the GD build fusing only from 6 tokens (EXL3_GD_MIN_ROWS, include/exl3_abi.h: arch band and exl3r4d together).
add_executable(arch_static_test_gd6 arch_static_test.cpp)
target_link_libraries(arch_static_test_gd6 PRIVATE radiance::abi)
target_compile_definitions(arch_static_test_gd6 PRIVATE QWEN4EXP_EXL3_PREP_SPLIT=1 QWEN4EXP_EXL3_GD=1 EXL3_GD_MIN_ROWS=6)
add_test(NAME arch_static_gd6 COMMAND sh ${CMAKE_CURRENT_LIST_DIR}/run_cases.sh $<TARGET_FILE:arch_static_test_gd6>)
set_tests_properties(arch_static_gd6 PROPERTIES LABELS host SKIP_RETURN_CODE 77)
# And the GD build with prefill staging deliberately off (QWEN4EXP_EXL3_PREFILL_STAGE 0).
add_executable(arch_static_test_nostage arch_static_test.cpp)
target_link_libraries(arch_static_test_nostage PRIVATE radiance::abi)
target_compile_definitions(arch_static_test_nostage PRIVATE QWEN4EXP_EXL3_PREP_SPLIT=1 QWEN4EXP_EXL3_GD=1
                           QWEN4EXP_EXL3_PREFILL_STAGE=0)
add_test(NAME arch_static_nostage
         COMMAND sh ${CMAKE_CURRENT_LIST_DIR}/run_cases.sh $<TARGET_FILE:arch_static_test_nostage>)
set_tests_properties(arch_static_nostage PROPERTIES LABELS host SKIP_RETURN_CODE 77)
# And as a real build is, with no define: the one case that checks the shipped default (the others ran above).
add_executable(arch_static_test_default arch_static_test.cpp)
target_link_libraries(arch_static_test_default PRIVATE radiance::abi)
target_compile_definitions(arch_static_test_default PRIVATE ARCH_STATIC_DEFAULT_BUILD=1)
add_test(NAME arch_static_default
         COMMAND arch_static_test_default the_shipped_default_is_the_split_order)
set_tests_properties(arch_static_default PROPERTIES LABELS host)

# ---- the loader refuses a second ("qwen4exp", "exl3") by name. The twin is this plugin under a
# second name, built into a home of its own and never installed: in the plugin's build home it would
# make every run of that home a duplicate.
set(RAD_PLUGIN_INSTALL OFF)
rad_add_plugin(qwen4exp_exl3_twin architectures SOURCES ${CMAKE_CURRENT_LIST_DIR}/twin_plugin.cpp)
target_compile_options(qwen4exp_exl3_twin PRIVATE -fvisibility-inlines-hidden)
set_target_properties(qwen4exp_exl3_twin PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/twin_home/architectures)
# ---- host rows for the four fusions qwen4exp asks for -- the router pair and the attention prologue
# -- running libref's own rows (router_rows.cpp says why: a host-only engine cannot declare or
# convert qwen4exp without them). Its own home, never installed,
# named only in tests' --kernels; router_rows_test is its gate: byte-identical to libref's chain.
rad_add_plugin(exl3test_router kernels SOURCES ${CMAKE_CURRENT_LIST_DIR}/router_rows.cpp)
target_link_libraries(exl3test_router PRIVATE ${CMAKE_DL_LIBS})
# libref's own flag: an FMA would move the rope's last bit off libref's (its CMakeLists.txt says why).
target_compile_options(exl3test_router PRIVATE -ffp-contract=off)
set_target_properties(exl3test_router PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_home/kernels)
add_executable(router_rows_test router_rows_test.cpp)
target_link_libraries(router_rows_test PRIVATE radiance::abi ${CMAKE_DL_LIBS})
add_test(NAME arch_router_rows COMMAND router_rows_test $<TARGET_FILE:exl3test_router>)
set_tests_properties(arch_router_rows PROPERTIES LABELS host SKIP_RETURN_CODE 77
                     ENVIRONMENT "RADIANCE_HOME=${RADIANCE_HOME}")

# ---- the end-to-end differential's wiring controls: this plugin with a test-only mutation compiled
# in (QWEN4EXP_EXL3_TEST_MUTATION, rad_block_moe_exl3.h), each under the real plugin's file name in a
# home of its own, so that home put first shadows the real one. Never installed.
# ---- the two issue orders as homes of their own, for an engine A/B (P6's speed harness): each is
# this plugin under the real plugin's file name, the one at QWEN4EXP_EXL3_PREP_SPLIT 0 (one gate_up,
# the shared arm behind it) and the one at 1 (prep_gu, the shared arm, gate_up over xr, at <= 16
# rows). Put first in RADIANCE_HOME, either shadows the build's own. Never installed.
foreach(_qx_ord 0 1)
  rad_add_plugin(qwen4exp_exl3_order${_qx_ord} architectures
                 SOURCES ${CMAKE_CURRENT_LIST_DIR}/../../architectures/qwen4exp_exl3/qwen4exp_exl3.cpp)
  target_compile_definitions(qwen4exp_exl3_order${_qx_ord} PRIVATE QWEN4EXP_EXL3_PREP_SPLIT=${_qx_ord})
  target_compile_options(qwen4exp_exl3_order${_qx_ord} PRIVATE -fvisibility-inlines-hidden)
  set_target_properties(qwen4exp_exl3_order${_qx_ord} PROPERTIES OUTPUT_NAME qwen4exp_exl3
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/order${_qx_ord}_home/architectures)
endforeach()

# ...and the decode chain as one op (QWEN4EXP_EXL3_GD 1 over the split build: gate_up_down at <= 16 rows) as a
# third home, gd_home, for the same A/B. Never installed.
rad_add_plugin(qwen4exp_exl3_gd architectures
               SOURCES ${CMAKE_CURRENT_LIST_DIR}/../../architectures/qwen4exp_exl3/qwen4exp_exl3.cpp)
target_compile_definitions(qwen4exp_exl3_gd PRIVATE QWEN4EXP_EXL3_PREP_SPLIT=1 QWEN4EXP_EXL3_GD=1)
target_compile_options(qwen4exp_exl3_gd PRIVATE -fvisibility-inlines-hidden)
set_target_properties(qwen4exp_exl3_gd PROPERTIES OUTPUT_NAME qwen4exp_exl3
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gd_home/architectures)

# ...and that build with prefill staging deliberately off, gd_nostage_home/, for the staging-on vs -off A/B.
rad_add_plugin(qwen4exp_exl3_gd_nostage architectures
               SOURCES ${CMAKE_CURRENT_LIST_DIR}/../../architectures/qwen4exp_exl3/qwen4exp_exl3.cpp)
target_compile_definitions(qwen4exp_exl3_gd_nostage PRIVATE QWEN4EXP_EXL3_PREP_SPLIT=1 QWEN4EXP_EXL3_GD=1
                           QWEN4EXP_EXL3_PREFILL_STAGE=0)
target_compile_options(qwen4exp_exl3_gd_nostage PRIVATE -fvisibility-inlines-hidden)
set_target_properties(qwen4exp_exl3_gd_nostage PROPERTIES OUTPUT_NAME qwen4exp_exl3
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gd_nostage_home/architectures)

foreach(_qx_mut 1 2 3)
  rad_add_plugin(qwen4exp_exl3_mut${_qx_mut} architectures
                 SOURCES ${CMAKE_CURRENT_LIST_DIR}/../../architectures/qwen4exp_exl3/qwen4exp_exl3.cpp)
  target_compile_definitions(qwen4exp_exl3_mut${_qx_mut} PRIVATE QWEN4EXP_EXL3_TEST_MUTATION=${_qx_mut})
  target_compile_options(qwen4exp_exl3_mut${_qx_mut} PRIVATE -fvisibility-inlines-hidden)
  set_target_properties(qwen4exp_exl3_mut${_qx_mut} PROPERTIES OUTPUT_NAME qwen4exp_exl3
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mut${_qx_mut}_home/architectures)
endforeach()

get_filename_component(_qx_prefix "${RADIANCE_INCLUDE_DIR}/../.." ABSOLUTE)
find_program(QWEN4EXP_EXL3_RAD_INFO rad-info HINTS "${_qx_prefix}/bin" NO_DEFAULT_PATH)
add_test(NAME arch_duplicate_key
         COMMAND sh ${CMAKE_CURRENT_LIST_DIR}/duplicate_key.sh "${QWEN4EXP_EXL3_RAD_INFO}"
                 $<TARGET_FILE_DIR:qwen4exp_exl3>/.. ${CMAKE_CURRENT_BINARY_DIR}/twin_home
                 "${RADIANCE_HOME}")
set_tests_properties(arch_duplicate_key PROPERTIES LABELS host SKIP_RETURN_CODE 77)

# ---- the vendored files against the pinned release (tests/upstream_drift.sh), and its negative
# control: v1.0.8's copies of the same files differ, so checking against it must FAIL (exit 1) --
# not pass, and not "could not check" (2).
add_test(NAME upstream_drift COMMAND sh ${_qx_repo}/tests/upstream_drift.sh)
set_tests_properties(upstream_drift PROPERTIES LABELS net SKIP_RETURN_CODE 2)
add_test(NAME upstream_drift_negative_control
         COMMAND sh -c "sh \"$0\" v1.0.8; s=$?; [ $s -eq 1 ] && exit 0; [ $s -eq 2 ] && exit 2; exit 1"
                 ${_qx_repo}/tests/upstream_drift.sh)
set_tests_properties(upstream_drift_negative_control PROPERTIES LABELS net SKIP_RETURN_CODE 2)

# ---- the end-to-end differential (e2e_differential.sh says what and why). ~1 minute and ~2 GB of
# fixtures and containers on the host, so its own label, not `host`. Its inputs are paths on the
# machine running it, given at configure time; empty, the test SKIPS by name (77).
set(QWEN4EXP_EXL3_E2E_HOME "" CACHE PATH
    "a RADIANCE_HOME with this repository's built plugins (qwen4exp_exl3, exl3ref, exl3_quant)")
set(QWEN4EXP_EXL3_TOKENIZER "" CACHE PATH "a directory with the Qwen3.8 tokenizer files")
foreach(_qx_mode host gpu)
  add_test(NAME arch_e2e_${_qx_mode}
           COMMAND sh ${CMAKE_CURRENT_LIST_DIR}/e2e_differential.sh ${CMAKE_CURRENT_BINARY_DIR}/e2e-${_qx_mode})
  set(_qx_label e2e)
  if(_qx_mode STREQUAL "gpu")
    set(_qx_label gpu)
  endif()
  set_tests_properties(arch_e2e_${_qx_mode} PROPERTIES SKIP_RETURN_CODE 77 LABELS ${_qx_label}
    ENVIRONMENT "E2E_MODE=${_qx_mode};RADIANCE_PREFIX=${_qx_prefix};EXL3_HOME=${QWEN4EXP_EXL3_E2E_HOME};TEST_BUILD=${CMAKE_CURRENT_BINARY_DIR};TOKENIZER=${QWEN4EXP_EXL3_TOKENIZER}")
endforeach()
