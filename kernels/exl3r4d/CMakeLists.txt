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

if(NOT RADIANCE_ABI_VERSION EQUAL 27)
  message(FATAL_ERROR "exl3r4d is written against radiance ABI 27; the radiance found is ABI "
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
            prefill.c                    # P5: the M > 16 band's hooks
    HIP_SOURCES exl3r4d.hip exl3r4d_prefill.hip
                exl3r4d_mod4.hip exl3r4d_prefill_mod4.hip   # v6: the same kernels at four classes
                exl3r4d_gd_mod4.hip                         # v7: the fused gate_up_down at four classes
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${EXL3_ROOT}/include ${EXL3_ROOT}/common
    GPU_TARGETS ${EXL3R4D_TARGETS})
  set_target_properties(exl3r4d PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
  # -ffp-contract=off on the host code too (rad_hipify sets it for the device code): the layout code does no
  # arithmetic that could contract, and the flag keeps it that way.
  target_compile_options(exl3r4d PRIVATE $<$<COMPILE_LANGUAGE:C>:${EXL3R4D_C_FLAGS}>)
  # P9-L7 (EXL3R4D_ALIGN_PAGE): every device function starts on a 4 KB boundary. Two builds with instruction-identical
  # kernels that differ only in where the kernels landed (the gate_up GEMM at 0x43000 vs 0x58a00) measured the tp1 M=3
  # split chain at 146.0 vs 155.2 us a layer (perfmoe, 0000:03:00.0, four sessions; the gate_up GEMM itself 88 vs 101 us
  # inside the chain, traced; equal alone and at every other shape tried); the second tree with 4 KB alignment read
  # 146.0 again (sK-1). Mechanism open (not the K loop crossing a page: a build whose loop crosses one is fast). The
  # cost is padding, < 4 KB a kernel. OFF puts the kernels wherever the linker packs them.
  option(EXL3R4D_ALIGN_PAGE "start every exl3r4d device function on a 4 KB boundary" ON)
  if(EXL3R4D_ALIGN_PAGE)
    target_compile_options(exl3r4d PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-falign-functions=4096>)
  endif()
  string(REPLACE ";" " " _exl3r4d_bt "${EXL3R4D_TARGETS}")
  target_compile_definitions(exl3r4d PRIVATE "EXL3R4D_BUILD_TARGET=\"${_exl3r4d_bt}\"")
  find_library(EXL3R4D_LIBM m)   # common/exl3_decode.c's nearbyint; part of libc where absent
  if(EXL3R4D_LIBM)
    target_link_libraries(exl3r4d PRIVATE ${EXL3R4D_LIBM})
  endif()
endif()

add_subdirectory(${EXL3_ROOT}/tests/gpu ${CMAKE_CURRENT_BINARY_DIR}/tests-gpu)
add_subdirectory(${EXL3_ROOT}/tests/gpu/prefill ${CMAKE_CURRENT_BINARY_DIR}/tests-gpu-prefill)   # P5
