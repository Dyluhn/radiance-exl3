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
add_executable(arch_static_test arch_static_test.cpp)
target_link_libraries(arch_static_test PRIVATE radiance::abi)
add_test(NAME arch_static
         COMMAND sh ${CMAKE_CURRENT_LIST_DIR}/run_cases.sh $<TARGET_FILE:arch_static_test>)
set_tests_properties(arch_static PROPERTIES LABELS host SKIP_RETURN_CODE 77)

# ---- the loader refuses a second ("qwen4exp", "exl3") by name. The twin is this plugin under a
# second name, built into a home of its own and never installed: in the plugin's build home it would
# make every run of that home a duplicate.
set(RAD_PLUGIN_INSTALL OFF)
rad_add_plugin(qwen4exp_exl3_twin architectures SOURCES ${CMAKE_CURRENT_LIST_DIR}/twin_plugin.cpp)
target_compile_options(qwen4exp_exl3_twin PRIVATE -fvisibility-inlines-hidden)
set_target_properties(qwen4exp_exl3_twin PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/twin_home/architectures)
# ---- three of libr4d's fusion schemas with no rows (fusion_schemas.cpp says why), for a host-only
# engine to declare qwen4exp at all. Its own home, never installed, named only in tests.
rad_add_plugin(exl3test_fusion_schemas kernels SOURCES ${CMAKE_CURRENT_LIST_DIR}/fusion_schemas.cpp)
set_target_properties(exl3test_fusion_schemas PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_home/kernels)

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
