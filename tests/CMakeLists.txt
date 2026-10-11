# tests/ -- every test under tests/ in the top-level ctest, by label:
#
#   ctest -LE 'gpu|oot'   THE HOST SUITE: needs neither a card nor a rebuild of every plugin (REQUIREMENTS R10)
#   ctest -L gpu          needs a card (and exl3r4d): run it under the project's GPU lock with the card pinned
#   ctest -L oot          every plugin directory configured ALONE against the installed radiance (R9), minutes
#   ctest -L e2e          the end-to-end differential (~1 min, ~2.5 GB); configure -DQWEN4EXP_EXL3_TOKENIZER=<dir>
#   ctest -L slow         minutes rather than seconds
#   ctest -L net          fetches the pinned radiance tag; SKIPPED (never passed) without the network
#
# Where each directory's tests are declared:
#   tests/arch            its own CMakeLists.txt, added here
#   tests/ref             by kernels/exl3ref/CMakeLists.txt (the harness is built there): not added twice
#   tests/gpu             by kernels/exl3r4d/CMakeLists.txt
#   tests/kld, tests/p6, tests/speed   pytest files: one ctest per file, below
#   tests/probe           Step 0's throwaway probes, only with -DEXL3_STEP0_PROBES=ON (top level)
# A test that checked nothing exits 77 and ctest reports it SKIPPED, never passed.
get_filename_component(_exl3_repo "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
get_filename_component(_exl3_prefix "${RADIANCE_INCLUDE_DIR}/../.." ABSOLUTE)

# ---- tests/arch. Its end-to-end test needs a $RADIANCE_HOME holding this build's plugins: the build tree's own
# home, unless one is given (a normal variable shadows the cache entry tests/arch declares; policy CMP0126).
if(NOT QWEN4EXP_EXL3_E2E_HOME)
  set(QWEN4EXP_EXL3_E2E_HOME "${RAD_PLUGIN_OUT}")
endif()
add_subdirectory(arch)

# ---- pytest files. A file that imports numpy, run by a python3 without it, or a python3 without pytest: SKIPPED.
# A file whose every test skipped (pytest exits 0 for that): SKIPPED. Nothing is written into the source tree.
find_program(EXL3_TESTS_PYTHON NAMES python3 python)
if(NOT EXL3_TESTS_PYTHON)
  set(EXL3_TESTS_PYTHON python3)   # absent: each test below says so and skips
endif()
set(_exl3_pytest [=[
py=$0; f=$1
command -v "$py" > /dev/null || { echo "SKIPPED: no $py"; exit 77; }
"$py" -c 'import pytest' 2> /dev/null || { echo "SKIPPED: $py has no pytest"; exit 77; }
if grep -q 'import numpy' "$f" && ! "$py" -c 'import numpy' 2> /dev/null; then echo "SKIPPED: $py has no numpy"; exit 77; fi
out=$("$py" -m pytest -q -rs -p no:cacheprovider "$f" 2>&1); rc=$?
printf '%s\n' "$out"
[ $rc -eq 0 ] || exit $rc
printf '%s\n' "$out" | tail -1 | grep -q ' passed' || { echo "SKIPPED: no test in $f ran"; exit 77; }
]=])
set(_exl3_py_env PYTHONDONTWRITEBYTECODE=1)
if(RADIANCE_SRC)
  list(APPEND _exl3_py_env "RADIANCE_SRC=${RADIANCE_SRC}")
endif()
foreach(_dir kld p6 speed)
  file(GLOB _files CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${_dir}/test_*.py")
  foreach(_f ${_files})
    get_filename_component(_n "${_f}" NAME_WE)
    add_test(NAME py_${_dir}_${_n} COMMAND sh -c "${_exl3_pytest}" ${EXL3_TESTS_PYTHON} ${_f})
    set_tests_properties(py_${_dir}_${_n} PROPERTIES LABELS host SKIP_RETURN_CODE 77
                         ENVIRONMENT "${_exl3_py_env}")
  endforeach()
endforeach()

# ---- R9 as a test: each plugin directory configured, built and installed ALONE against the installed radiance,
# compared with this build's own home (tools/gates/R9.sh says what is compared).
add_test(NAME oot_plugins COMMAND sh ${_exl3_repo}/tools/gates/R9.sh)
set_tests_properties(oot_plugins PROPERTIES LABELS oot SKIP_RETURN_CODE 77 TIMEOUT 3600
  ENVIRONMENT "RADIANCE_PREFIX=${_exl3_prefix};GATES_WORK=${CMAKE_CURRENT_BINARY_DIR}/oot;RAD_WITH_HIP=${RAD_WITH_HIP}")
