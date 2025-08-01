################################################################################
# Autogenerated by build_tools/bazel_to_cmake/bazel_to_cmake.py from           #
# compiler/plugins/target/ROCM/test/BUILD.bazel                                #
#                                                                              #
# Use iree_cmake_extra_content from iree/build_defs.oss.bzl to add arbitrary   #
# CMake-only content.                                                          #
#                                                                              #
# To disable autogeneration for this file entirely, delete this header.        #
################################################################################

iree_add_all_subdirs()

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "config_ukernel_argmax_gfx908.mlir"
    "config_ukernel_argmax_gfx942.mlir"
    "config_ukernel_data_tiled_mma_gfx942.mlir"
    "default_tuning_specs_amdgpu.mlir"
    "gpu_encoding_attrs.mlir"
    "lowering_strategy_from_tuning_spec.mlir"
    "ukernel_pipeline_transform.mlir"
  TOOLS
    FileCheck
    iree-opt
  DATA
    tuning_spec_mmt_tile_and_fuse.mlir
    tuning_spec_mmt_tile_and_fuse_default.mlir
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###

# NOTE: The following tests are CMake-only because they depend on AMD device
# bitcode libraries that are provided by custom CMake code in target/ROCM.
# By contrast, the above tests that only require ukernel bitcode are part of the
# Bazel build because ukernel bitcode is something that we generate ourselves.

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "external_function_validation.mlir"
    "smoketest.mlir"
    "smoketest_hsaco.mlir"
    "target_device_features.mlir"
  TOOLS
    FileCheck
    iree-opt
)
