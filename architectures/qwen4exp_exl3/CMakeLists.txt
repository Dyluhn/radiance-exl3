# qwen4exp_exl3 -- the ("qwen4exp", "exl3") architecture plugin. Built inside this repository's
# top-level project or ON ITS OWN against an installed radiance, the same way radiance's own plugin
# directories are:
#
#   cmake -S architectures/qwen4exp_exl3 -B build -DCMAKE_PREFIX_PATH=<radiance install prefix>
#
# The build tree's radiance_home/ is then a working $RADIANCE_HOME entry for it:
#   RADIANCE_HOME=build/radiance_home:<prefix>/share/radiance rad-info --plugins
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(qwen4exp_exl3 LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

# THE PIN, REFUSED AT CONFIGURE RATHER THAN AT dlopen. Two checks, because the ABI number alone cannot
# tell releases apart (it is 15 from 1.0.8 to 1.3.0), and these sources are copies of ONE release's
# files compiled against the installed headers of whatever release is found:
#   1. the ABI the loader will compare against;
#   2. the release: the installed engine must carry the version string RADIANCE_VERSION pins (the
#      repository root's; UPSTREAM.md says which files that pin governs). Said, not skipped in
#      silence, when the prefix has no engine binary to read it from.
if(NOT RADIANCE_ABI_VERSION EQUAL 27)
  message(FATAL_ERROR "qwen4exp_exl3 is built for radiance ABI 27; the radiance found "
                      "(${RADIANCE_INCLUDE_DIR}) is ABI ${RADIANCE_ABI_VERSION}")
endif()
set(_qx_pin_file "${CMAKE_CURRENT_LIST_DIR}/../../RADIANCE_VERSION")
if(NOT EXISTS "${_qx_pin_file}")
  message(FATAL_ERROR "qwen4exp_exl3: no RADIANCE_VERSION at ${_qx_pin_file}; the plugin's sources "
                      "are copies of one radiance release and that file says which")
endif()
file(STRINGS "${_qx_pin_file}" _qx_pin LIMIT_COUNT 1)
string(REGEX MATCH "^([0-9]+\\.[0-9]+\\.[0-9]+)" _qx_m "${_qx_pin}")
set(QWEN4EXP_EXL3_RADIANCE "${CMAKE_MATCH_1}")
if(NOT QWEN4EXP_EXL3_RADIANCE)
  message(FATAL_ERROR "qwen4exp_exl3: ${_qx_pin_file} does not start with a release (got '${_qx_pin}')")
endif()
get_filename_component(_qx_prefix "${RADIANCE_INCLUDE_DIR}/../.." ABSOLUTE)
if(EXISTS "${_qx_prefix}/bin/radiance")
  file(STRINGS "${_qx_prefix}/bin/radiance" _qx_versions REGEX "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  if(NOT QWEN4EXP_EXL3_RADIANCE IN_LIST _qx_versions)
    message(FATAL_ERROR "qwen4exp_exl3 is copied from radiance ${QWEN4EXP_EXL3_RADIANCE} "
                        "(RADIANCE_VERSION) and ${_qx_prefix}/bin/radiance is not that release "
                        "(it carries '${_qx_versions}'). Build against an install of "
                        "${QWEN4EXP_EXL3_RADIANCE}, or move the pin: tests/upstream_drift.sh.")
  endif()
  message(STATUS "qwen4exp_exl3: radiance ${QWEN4EXP_EXL3_RADIANCE} (ABI ${RADIANCE_ABI_VERSION}) at ${_qx_prefix}")
else()
  message(STATUS "qwen4exp_exl3: release check SKIPPED -- ${_qx_prefix}/bin/radiance does not exist "
                 "(a headers-only install); the ABI check still ran")
endif()

rad_add_plugin(qwen4exp_exl3 architectures SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/qwen4exp_exl3.cpp)
# The in-tree MTP head's inline members are compiled here over MoeEXL3 under the names the in-tree
# qwen4exp_fp8.so compiles over MoeFP8 (qwen4exp_exl3.cpp, at the includes). Hidden, each .so keeps
# its own copy however a process loads the two; the C exports are not inline and stay visible.
target_compile_options(qwen4exp_exl3 PRIVATE -fvisibility-inlines-hidden)
