# exl3r4d -- the gfx1201 kernel library for the EXL3 routed-expert ops (moe_exl3_gate_up, moe_exl3_down), decode
# band. Built from the repository's top level, or on its own against an installed radiance (R9):
#
#   cmake -S kernels/exl3r4d -B build-exl3r4d -DCMAKE_PREFIX_PATH=<radiance install prefix>
#
# DEVICE ONLY. The library exists when the radiance found has a device build (RAD_HIP_FOUND), -DRAD_WITH_HIP is
# not OFF, and the build targets include gfx1201 (rad_plugin_gpu_targets); otherwise it is left out and said so --
# a host-only build has nothing for it to serve (its host twin is kernels/exl3ref). The layout conversions are host
# code and are tested in every build (tests/gpu/layout_test.c, label host).
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(exl3r4d LANGUAGES C CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  enable_testing()
  set(EXL3R4D_STANDALONE ON)
endif()
enable_language(C)

if(NOT RADIANCE_ABI_VERSION EQUAL 15)
  message(FATAL_ERROR "exl3r4d is written against radiance ABI 15; the radiance found is ABI "
                      "${RADIANCE_ABI_VERSION}")
endif()

get_filename_component(EXL3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
foreach(_f include/exl3_abi.h include/exl3_encoding.h common/exl3_decode.c common/exl3_layout.c)
  if(NOT EXISTS "${EXL3_ROOT}/${_f}")
    message(FATAL_ERROR "exl3r4d needs ${EXL3_ROOT}/${_f} (the repository around kernels/exl3r4d)")
  endif()
endforeach()
set(EXL3R4D_C_FLAGS -ffp-contract=off -Wall -Wextra)

set(EXL3R4D_HIP OFF)
if(RAD_HIP_FOUND AND NOT (DEFINED RAD_WITH_HIP AND RAD_WITH_HIP STREQUAL "OFF"))
  rad_plugin_gpu_targets(EXL3R4D_TARGETS "^gfx1201$")
  if(EXL3R4D_TARGETS)
    set(EXL3R4D_HIP ON)
  else()
    message(STATUS "exl3r4d: left out -- this build targets '${RAD_GPU_TARGETS}', and the library is gfx1201 code")
  endif()
else()
  message(STATUS "exl3r4d: left out -- no device build (RAD_HIP_FOUND=${RAD_HIP_FOUND}, RAD_WITH_HIP=${RAD_WITH_HIP})")
endif()

if(EXL3R4D_HIP)
  if(EXL3R4D_STANDALONE AND NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${EXL3R4D_TARGETS}")
  endif()
  if(EXL3R4D_STANDALONE)
    enable_language(HIP)
  endif()
  rad_add_plugin(exl3r4d kernels
    SOURCES rows.c launch.c layout.c ${EXL3_ROOT}/common/exl3_decode.c ${EXL3_ROOT}/common/exl3_layout.c
    HIP_SOURCES exl3r4d.hip
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${EXL3_ROOT}/include ${EXL3_ROOT}/common
    GPU_TARGETS ${EXL3R4D_TARGETS})
  set_target_properties(exl3r4d PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
  # -ffp-contract=off on the host code too (rad_hipify sets it for the device code): the layout code does no
  # arithmetic that could contract, and the flag keeps it that way.
  target_compile_options(exl3r4d PRIVATE $<$<COMPILE_LANGUAGE:C>:${EXL3R4D_C_FLAGS}>)
  string(REPLACE ";" " " _exl3r4d_bt "${EXL3R4D_TARGETS}")
  target_compile_definitions(exl3r4d PRIVATE "EXL3R4D_BUILD_TARGET=\"${_exl3r4d_bt}\"")
  find_library(EXL3R4D_LIBM m)   # common/exl3_decode.c's nearbyint; part of libc where absent
  if(EXL3R4D_LIBM)
    target_link_libraries(exl3r4d PRIVATE ${EXL3R4D_LIBM})
  endif()
endif()

add_subdirectory(${EXL3_ROOT}/tests/gpu ${CMAKE_CURRENT_BINARY_DIR}/tests-gpu)
