# common/ -- the EXL3 oracle (exl3_decode) and the canonical-plane <-> tile-stream layout (exl3_layout).
# Pure C99, no radiance, no dependencies: configures and tests on its own,
#
#   cmake -S common -B build-common && cmake --build build-common && ctest --test-dir build-common -L host
#
# or from the repository's top level. Plugins that use it compile its two .c files into themselves
# (each plugin directory builds alone, REQUIREMENTS R9), so the library here is for tests and tools.
# Every test is labelled `host`; a test whose tool is missing exits 77 and ctest reports it SKIPPED.
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.21)
  project(exl3_common LANGUAGES C)
  enable_testing()
endif()
enable_language(C)

set(EXL3_COMMON_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/exl3_decode.c ${CMAKE_CURRENT_SOURCE_DIR}/exl3_layout.c)
# -ffp-contract=off: W's f32 roundings are the oracle's; an FMA would move them (W_hat is exact either way).
set(EXL3_COMMON_FLAGS -ffp-contract=off -Wall -Wextra)

find_library(EXL3_LIBM m)   # absent (and not needed) where libm is part of libc

function(exl3_common_target name kind)
  add_library(${name} ${kind} ${EXL3_COMMON_SOURCES})
  set_target_properties(${name} PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_compile_options(${name} PRIVATE ${EXL3_COMMON_FLAGS})
  target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  if(EXL3_LIBM)
    target_link_libraries(${name} PUBLIC ${EXL3_LIBM})
  endif()
endfunction()
exl3_common_target(exl3_common STATIC)
exl3_common_target(exl3_common_shared SHARED)   # what the Python cross-checks load through ctypes

foreach(t exl3_decode_test exl3_layout_test)
  add_executable(${t} tests/${t}.c)
  set_target_properties(${t} PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
  target_compile_options(${t} PRIVATE ${EXL3_COMMON_FLAGS})
  target_link_libraries(${t} PRIVATE exl3_common)
  add_test(NAME ${t} COMMAND ${t})
endforeach()

# The `python3` the shell would run (find_program searches PATH for the name, as `command -v` does;
# FindPython would take a versioned interpreter from an earlier PATH directory first). One without
# numpy turns the cross-checks into SKIPs; -DEXL3_PYTHON=... picks another.
find_program(EXL3_PYTHON NAMES python3 python)
set(_tests ${CMAKE_CURRENT_SOURCE_DIR}/tests)
if(EXL3_PYTHON)
  set(_py ${EXL3_PYTHON})
else()
  set(_py sh -c "echo SKIPPED: no python3 interpreter; exit 77" --)
endif()
add_test(NAME exl3_c_vs_ref COMMAND ${_py} ${_tests}/test_c_vs_ref.py $<TARGET_FILE:exl3_common_shared>)
add_test(NAME exl3_golden_header
         COMMAND ${_py} ${_tests}/gen_golden_h.py ${_tests}/data/exl3_golden.npz ${_tests}/exl3_golden.h --check)
# The twin against trellis.py itself: give EXL3_TRELLIS_PY (exl3-moe's exl3_moe/kernels/trellis.py) in the
# environment ctest runs in; SKIPPED otherwise.
add_test(NAME exl3_ref_vs_trellis_py COMMAND ${_py} ${_tests}/test_ref_vs_trellis_py.py)

set_tests_properties(exl3_decode_test exl3_layout_test exl3_c_vs_ref exl3_golden_header exl3_ref_vs_trellis_py
                     PROPERTIES LABELS host SKIP_RETURN_CODE 77)
