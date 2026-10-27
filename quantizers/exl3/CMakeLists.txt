# quantizers/exl3 -- the `exl3` quantiser plugin (owner Q). Host code only; builds the same with or without HIP.
#
# Configures and builds ALONE against an installed radiance (REQUIREMENTS R9), as every radiance plugin
# directory does (docs/PLUGIN.md §13):
#
#   cmake -S quantizers/exl3 -B build-q -DCMAKE_PREFIX_PATH=<radiance install prefix>
#   cmake --build build-q && ctest --test-dir build-q -L host
#
# or as part of the repository's top-level build. It compiles common/'s two C files into itself (no
# library is shared between plugin directories) and includes include/exl3_encoding.h.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(exl3_quantizer LANGUAGES C CXX)
  # A $RADIANCE_HOME of this build's own, in front of the engine's at run time; set BEFORE find_package,
  # whose config would otherwise default it to the installed radiance's home.
  set(RAD_PLUGIN_INSTALL_HOME "${CMAKE_BINARY_DIR}/home" CACHE PATH "the \$RADIANCE_HOME this plugin installs under")
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  enable_testing()
endif()
enable_language(C)
if(NOT RADIANCE_ABI_VERSION EQUAL 27)
  message(FATAL_ERROR "the exl3 quantiser is written against radiance ABI 27; the radiance found has ABI "
                      "${RADIANCE_ABI_VERSION}. Point CMAKE_PREFIX_PATH at the release RADIANCE_VERSION pins.")
endif()

get_filename_component(EXL3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(EXL3_COMMON_C ${EXL3_ROOT}/common/exl3_decode.c ${EXL3_ROOT}/common/exl3_layout.c)
# The oracle's roundings are its own: no FMA contraction in the C (W_hat is exact either way).
set_source_files_properties(${EXL3_COMMON_C} PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

rad_add_plugin(exl3_quant quantizers
  SOURCES exl3_quant.cpp exl3_src.cpp ${EXL3_COMMON_C}
  INCLUDES ${EXL3_ROOT}/include ${EXL3_ROOT}/common)
set_target_properties(exl3_quant PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
target_compile_options(exl3_quant PRIVATE -Wall -Wextra -ffp-contract=off)
find_library(EXL3_LIBM m)
if(EXL3_LIBM)
  target_link_libraries(exl3_quant PRIVATE ${EXL3_LIBM})
endif()

# ---------------------------------------------------------------- tests (label `host`)
# ONE test: make the fixture in this build tree (tools/exl3_fixture.py, needs python3 + numpy, ~1 s), then run
# the plugin through its ABI against it. Without python3 or numpy it is SKIPPED (exit 77), never passed.
add_executable(exl3_quant_test tests/exl3_quant_test.cpp exl3_src.cpp ${EXL3_COMMON_C})
set_target_properties(exl3_quant_test PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
target_include_directories(exl3_quant_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${EXL3_ROOT}/include
                           ${EXL3_ROOT}/common ${EXL3_ROOT}/common/tests)
target_link_libraries(exl3_quant_test PRIVATE radiance::abi ${CMAKE_DL_LIBS})
if(EXL3_LIBM)
  target_link_libraries(exl3_quant_test PRIVATE ${EXL3_LIBM})
endif()
find_program(EXL3_PYTHON NAMES python3 python)
if(NOT EXL3_PYTHON)
  set(EXL3_PYTHON python3)   # absent: the test below says so and skips
endif()
# sh -c SCRIPT python tool fixture-dir test-binary plugin
add_test(NAME exl3_quant_test
         COMMAND sh -c "command -v \"$0\" >/dev/null && \"$0\" -c 'import numpy' 2>/dev/null || { echo \"SKIPPED: $0 with numpy is needed to make the fixture\"; exit 77; }; \"$0\" \"$1\" --out \"$2\" --defects && exec \"$3\" \"$4\" \"$2\""
                 ${EXL3_PYTHON} ${EXL3_ROOT}/tools/exl3_fixture.py ${CMAKE_CURRENT_BINARY_DIR}/fixture
                 $<TARGET_FILE:exl3_quant_test> $<TARGET_FILE:exl3_quant>)
set_tests_properties(exl3_quant_test PROPERTIES LABELS host SKIP_RETURN_CODE 77
                     ENVIRONMENT PYTHONDONTWRITEBYTECODE=1)
