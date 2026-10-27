# radiance-exl3 -- EXL3 routed experts for radiance, as out-of-tree plugins (PLAN.md §2).
#
# Build against an INSTALLED radiance (docs/PLUGIN.md §13):
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<radiance install prefix> [-DRAD_WITH_HIP=OFF]
#
# Each owner's directory is added only when it has a CMakeLists.txt, so any one of them configures
# and builds without the others existing yet. Plugins install into a $RADIANCE_HOME of this build's
# own (RAD_PLUGIN_INSTALL_HOME), put in front of the engine's at run time; they never install into
# the radiance installation.
cmake_minimum_required(VERSION 3.21)
project(radiance_exl3 LANGUAGES C CXX)

# Set BEFORE find_package, whose config otherwise defaults it to the installed radiance's own home.
set(RAD_PLUGIN_INSTALL_HOME "${CMAKE_BINARY_DIR}/home" CACHE PATH
    "the \$RADIANCE_HOME these plugins install under")
set(RAD_WITH_HIP "AUTO" CACHE STRING
    "ON | OFF | AUTO -- compile device rows (AUTO: when the installed radiance has HIP)")
set_property(CACHE RAD_WITH_HIP PROPERTY STRINGS ON OFF AUTO)
option(EXL3_STEP0_PROBES "build the Step 0 throwaway probe plugins in tests/probe" OFF)

find_package(radiance REQUIRED)

# ---------------------------------------------------------------- the pin (PLAN.md D11, D14)
# THE ABI: a plugin is loaded by this number; another one is another contract. find_package's own
# version IS the ABI number (27.0.0), so it cannot tell two releases apart -- hence the second check.
set(EXL3_RADIANCE_ABI 27)
if(NOT RADIANCE_ABI_VERSION EQUAL EXL3_RADIANCE_ABI)
  message(FATAL_ERROR
    "radiance at CMAKE_PREFIX_PATH has ABI ${RADIANCE_ABI_VERSION}; these plugins are written "
    "against ABI ${EXL3_RADIANCE_ABI}. Point CMAKE_PREFIX_PATH at an install of the release "
    "RADIANCE_VERSION names.")
endif()
# THE RELEASE: the vendored architecture is a copy of one release's qwen4exp_fp8.cpp, so the
# installed engine must be the release RADIANCE_VERSION pins. The release string lives in the engine
# binary (radiance core/CMakeLists.txt RAD_VERSION). A headers-only prefix has no binary: said, and
# the ABI check above still holds.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/RADIANCE_VERSION" _exl3_pin LIMIT_COUNT 1)
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" EXL3_RADIANCE_RELEASE "${_exl3_pin}")
if(NOT EXL3_RADIANCE_RELEASE)
  message(FATAL_ERROR "RADIANCE_VERSION does not start with a release number: '${_exl3_pin}'")
endif()
get_filename_component(_exl3_prefix "${RADIANCE_INCLUDE_DIR}/../.." ABSOLUTE)
set(_exl3_engine "${_exl3_prefix}/bin/radiance")
if(EXISTS "${_exl3_engine}")
  file(STRINGS "${_exl3_engine}" _exl3_versions REGEX "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  if(NOT EXL3_RADIANCE_RELEASE IN_LIST _exl3_versions)
    message(FATAL_ERROR
      "RADIANCE_VERSION pins radiance ${EXL3_RADIANCE_RELEASE}, and ${_exl3_engine} is not it "
      "(it carries '${_exl3_versions}'). Point CMAKE_PREFIX_PATH at an install of "
      "${EXL3_RADIANCE_RELEASE}.")
  endif()
  message(STATUS "radiance ${EXL3_RADIANCE_RELEASE}, ABI ${RADIANCE_ABI_VERSION}: ${_exl3_engine}")
else()
  message(STATUS "radiance release check SKIPPED: ${_exl3_engine} does not exist (headers-only "
                 "prefix); ABI ${RADIANCE_ABI_VERSION} checked")
endif()

rad_plugin_defaults()

# ---------------------------------------------------------------- device code
# -DRAD_WITH_HIP=OFF is the host-only build: no device sources even when the installed radiance has
# HIP. rad_add_plugin reads RAD_HIP_FOUND from this scope.
if(RAD_WITH_HIP STREQUAL "OFF")
  set(RAD_HIP_FOUND OFF)
elseif(RAD_WITH_HIP STREQUAL "ON" AND NOT RAD_HIP_FOUND)
  message(FATAL_ERROR "RAD_WITH_HIP=ON and the installed radiance was built without HIP")
endif()
if(RAD_HIP_FOUND)
  # The targets are the radiance install's (or -DRAD_GPU_TARGETS), never this file's: without this
  # CMake asks the build machine for a card, and a build container has none.
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${RAD_GPU_TARGETS}")
  endif()
  enable_language(HIP)
endif()
message(STATUS "radiance-exl3: device rows ${RAD_HIP_FOUND}, GPU targets '${RAD_GPU_TARGETS}', "
               "installs into ${RAD_PLUGIN_INSTALL_HOME}")

enable_testing()
foreach(_exl3_dir common quantizers/exl3 kernels/exl3ref kernels/exl3r4d
                  architectures/qwen4exp_exl3 tools tests)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_exl3_dir}/CMakeLists.txt")
    add_subdirectory(${_exl3_dir})
  endif()
endforeach()
if(EXL3_STEP0_PROBES)
  add_subdirectory(tests/probe)
endif()
