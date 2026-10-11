# Step 0 throwaway probes (docs/STEP0.md). Built only with -DEXL3_STEP0_PROBES=ON, never installed,
# and into homes of their own, so no real $RADIANCE_HOME ever holds a row that computes nothing.
#   home/       exl3probe_quant, exl3probe_k, exl3probe_arch -- the probes
#   homebad/    exl3probe_quant_bad -- a decode that ignores suh.1 (S0.2's negative control)
#   home2tag/   exl3probe_k2 -- the same op with two stored forms (S0.10's negative control)
# run_probes.sh drives them.
set(RAD_PLUGIN_INSTALL OFF)

set(RAD_PLUGIN_OUT "${CMAKE_CURRENT_BINARY_DIR}/home")
rad_add_plugin(exl3probe_quant quantizers SOURCES exl3probe_quant.cc)
rad_add_plugin(exl3probe_k kernels SOURCES exl3probe_kern.cc)
rad_add_plugin(exl3probe_arch architectures SOURCES exl3probe_arch.cc)

set(RAD_PLUGIN_OUT "${CMAKE_CURRENT_BINARY_DIR}/homebad")
rad_add_plugin(exl3probe_quant_bad quantizers SOURCES exl3probe_quant.cc)
target_compile_definitions(exl3probe_quant_bad PRIVATE EXL3PROBE_BROKEN_DECODE=1)
# The same FILE name as the good one, so homebad/ first on $RADIANCE_HOME shadows home/'s
# (docs/PLUGIN.md: an earlier home's file of the same name wins) -- D8's fallback mechanism.
set_target_properties(exl3probe_quant_bad PROPERTIES OUTPUT_NAME exl3probe_quant)

set(RAD_PLUGIN_OUT "${CMAKE_CURRENT_BINARY_DIR}/home2tag")
rad_add_plugin(exl3probe_k2 kernels SOURCES exl3probe_kern.cc)
target_compile_definitions(exl3probe_k2 PRIVATE EXL3PROBE_TWO_TAGS=1)
