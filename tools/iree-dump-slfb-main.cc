#include <cstdint>
#include <fstream>
#include <iostream>

#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/exsleratev2_builder.h"
#include "iree/schemas/exsleratev2_executable_def_builder.h"
#include "iree/schemas/exsleratev2_executable_def_reader.h"
#include "iree/schemas/exsleratev2_executable_def_verifier.h"
#include "iree/schemas/exsleratev2_reader.h"
#include "iree/schemas/exsleratev2_verifier.h"

namespace {

const char* llLayerKindName(uint8_t kind) {
  switch (kind) {
    case 0:
      return "conv";
    case 1:
      return "pool";
    case 2:
      return "matmul";
    case 3:
      return "add";
    case 4:
      return "fully_connected";
    case 5:
      return "cpu";
    case 6:
      return "skip";
    case 7:
      return "reduce";
    default:
      return "unknown";
  }
}

const char* llExecutionModeName(uint8_t mode) {
  switch (mode) {
    case 0:
      return "exslerate";
    case 1:
      return "cpu";
    case 2:
      return "skip";
    default:
      return "unknown";
  }
}

const char* llElementTypeName(uint8_t type) {
  switch (type) {
    case 0:
      return "i8";
    case 1:
      return "f32";
    case 2:
      return "i32";
    default:
      return "unknown";
  }
}

const char* llDataCategoryName(uint8_t category) {
  switch (category) {
    case 0:
      return "weights";
    case 1:
      return "bias";
    case 2:
      return "input";
    case 3:
      return "output";
    case 4:
      return "atomic_bank";
    case 5:
      return "atomic_offset";
    case 6:
      return "lifetime";
    case 7:
      return "bnweight";
    case 8:
      return "bnbias";
    case 9:
      return "lut";
    default:
      return "unknown";
  }
}

void* readEntireFile(const char* filename, size_t* out_size) {
  FILE* fp = fopen(filename, "rb");
  if (!fp) {
    printf("Error opening file for reading\n");
    return nullptr;
  }

  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  void* buffer = malloc(size);
  if (fread(buffer, 1, size, fp) != size) {
    printf("Error reading file\n");
    fclose(fp);
    free(buffer);
    return nullptr;
  }
  fclose(fp);

  *out_size = size;
  return buffer;
}

// Dumps the legacy exsleratev2_executable_def.fbs ("exsleratev2") schema,
// still the only format the runtime driver actually deserializes.
void deserializeFromSLFb(const void* buffer) {
  iree_exsleratev2_hal_exsleratev2_ExecutableDef_table_t executable =
      iree_exsleratev2_hal_exsleratev2_ExecutableDef_as_root(buffer);

  if (!executable) {
    printf("Invalid FlatBuffer format\n");
    return;
  }

  flatbuffers_string_vec_t entry_points =
      iree_exsleratev2_hal_exsleratev2_ExecutableDef_entry_points(executable);
  if (entry_points) {
    size_t count = flatbuffers_string_vec_len(entry_points);
    printf("Found %zu entry points:\n", count);
    for (size_t i = 0; i < count; i++) {
      const char* name = flatbuffers_string_vec_at(entry_points, i);
      printf("  %zu: %s\n", i, name ? name : "(null)");
    }
  }

  flatbuffers_uint8_vec_t cpu_code =
      iree_exsleratev2_hal_exsleratev2_ExecutableDef_cpu_code(executable);
  if (cpu_code) {
    printf("CPU Code size: %zu bytes\n", flatbuffers_uint8_vec_len(cpu_code));
  }

  const char* cpu_func_name =
      iree_exsleratev2_hal_exsleratev2_ExecutableDef_cpu_function_name(
          executable);
  if (cpu_func_name) {
    printf("CPU Function Name: %s\n", cpu_func_name);
  }

  iree_exsleratev2_hal_exsleratev2_LayerDef_vec_t layers =
      iree_exsleratev2_hal_exsleratev2_ExecutableDef_layers(executable);
  if (!layers) {
    printf("No layers found\n");
    return;
  }

  size_t layer_count =
      iree_exsleratev2_hal_exsleratev2_ExecutableDef_vec_len(layers);
  printf("Found %zu layers:\n", layer_count);

  for (size_t i = 0; i < layer_count; i++) {
    iree_exsleratev2_hal_exsleratev2_LayerDef_table_t layer =
        iree_exsleratev2_hal_exsleratev2_LayerDef_vec_at(layers, i);
    printf("Layer %zu: (ptr=%p)\n", i, (void*)layer);
    if (!layer) {
      printf("  ERROR: NULL layer pointer!\n");
      continue;
    }

    iree_exsleratev2_hal_exsleratev2_RegisterValue_vec_t csr_configs =
        iree_exsleratev2_hal_exsleratev2_LayerDef_csr_configs(layer);
    printf("  csr_configs ptr=%p\n", (void*)csr_configs);

    iree_exsleratev2_hal_exsleratev2_MemRefDef_vec_t mem_ref_defs_check =
        iree_exsleratev2_hal_exsleratev2_LayerDef_mem_ref_defs(layer);
    printf("  mem_ref_defs ptr=%p\n", (void*)mem_ref_defs_check);

    iree_exsleratev2_hal_exsleratev2_DataBufferDef_vec_t data_buffers_check =
        iree_exsleratev2_hal_exsleratev2_LayerDef_data_buffers(layer);
    printf("  data_buffers ptr=%p\n", (void*)data_buffers_check);

    iree_exsleratev2_hal_exsleratev2_InputTileData_table_t
        input_tile_data_check =
            iree_exsleratev2_hal_exsleratev2_LayerDef_input_tile_data(layer);
    printf("  input_tile_data ptr=%p\n", (void*)input_tile_data_check);

    uint32_t input_tile_buf =
        iree_exsleratev2_hal_exsleratev2_LayerDef_input_tile_buf(layer);
    uint32_t input_offset =
        iree_exsleratev2_hal_exsleratev2_LayerDef_input_offset(layer);
    uint32_t output_offset =
        iree_exsleratev2_hal_exsleratev2_LayerDef_output_offset(layer);
    uint32_t num_channel =
        iree_exsleratev2_hal_exsleratev2_LayerDef_num_channel(layer);
    uint32_t num_filter =
        iree_exsleratev2_hal_exsleratev2_LayerDef_num_filter(layer);

    printf("\n  Layer Scalar Fields:\n");
    printf("    input_tile_buf: %u\n", input_tile_buf);
    printf("    input_offset: %u\n", input_offset);
    printf("    output_offset: %u\n", output_offset);
    printf("    num_channel: %u\n", num_channel);
    printf("    num_filter: %u\n\n", num_filter);

    if (csr_configs) {
      size_t csr_count =
          iree_exsleratev2_hal_exsleratev2_RegisterValue_vec_len(csr_configs);
      printf("  CSR Configs (%zu):\n", csr_count);
      for (size_t j = 0; j < csr_count; j++) {
        iree_exsleratev2_hal_exsleratev2_RegisterValue_table_t reg =
            iree_exsleratev2_hal_exsleratev2_RegisterValue_vec_at(csr_configs,
                                                                  j);
        uint8_t value_type =
            iree_exsleratev2_hal_exsleratev2_RegisterValue_value_type(reg);
        uint32_t literal_value =
            iree_exsleratev2_hal_exsleratev2_RegisterValue_literal_value(reg);
        uint32_t memref_id =
            iree_exsleratev2_hal_exsleratev2_RegisterValue_memrefdef_id(reg);
        uint32_t offset =
            iree_exsleratev2_hal_exsleratev2_RegisterValue_csr_id(reg);

        printf("    %zu: type=%u, offset=0x%X, literal=%u, memref_id=%u\n", j,
               value_type, offset, literal_value, memref_id);
      }
    }

    iree_exsleratev2_hal_exsleratev2_MemRefDef_vec_t mem_ref_defs =
        iree_exsleratev2_hal_exsleratev2_LayerDef_mem_ref_defs(layer);
    if (mem_ref_defs) {
      size_t memref_count =
          iree_exsleratev2_hal_exsleratev2_MemRefDef_vec_len(mem_ref_defs);
      printf("  MemRef Defs (%zu):\n", memref_count);
      for (size_t j = 0; j < memref_count; j++) {
        iree_exsleratev2_hal_exsleratev2_MemRefDef_table_t memref =
            iree_exsleratev2_hal_exsleratev2_MemRefDef_vec_at(mem_ref_defs, j);
        uint32_t id = iree_exsleratev2_hal_exsleratev2_MemRefDef_id(memref);
        int8_t data_type =
            iree_exsleratev2_hal_exsleratev2_MemRefDef_data_type(memref);

        uint32_t alignment =
            iree_exsleratev2_hal_exsleratev2_MemRefDef_alignment(memref);

        flatbuffers_int32_vec_t shape_ptr =
            iree_exsleratev2_hal_exsleratev2_MemRefDef_shape(memref);
        size_t shape = flatbuffers_int32_vec_len(shape_ptr);

        printf("  MemRef %zu: id=%u, type=%d, alignment=%u, shape=%zu\n", j, id,
               data_type, alignment, shape);

        printf("    Shape: [");
        for (size_t k = 0; k < shape; ++k) {
          printf("%d", shape_ptr[k]);
          if (k < shape - 1) {
            printf(", ");
          }
        }
        printf("]\n");
      }
    }

    // Read InputTileData
    iree_exsleratev2_hal_exsleratev2_InputTileData_table_t input_tile_data =
        iree_exsleratev2_hal_exsleratev2_LayerDef_input_tile_data(layer);
    if (input_tile_data) {
      uint32_t tile_h =
          iree_exsleratev2_hal_exsleratev2_InputTileData_input_tile_height(
              input_tile_data);
      uint32_t tile_w =
          iree_exsleratev2_hal_exsleratev2_InputTileData_input_tile_width(
              input_tile_data);
      uint32_t out_tile_h =
          iree_exsleratev2_hal_exsleratev2_InputTileData_output_tile_height(
              input_tile_data);
      uint32_t out_tile_w =
          iree_exsleratev2_hal_exsleratev2_InputTileData_output_tile_width(
              input_tile_data);
      uint32_t input_tiled_buffer_size =
          iree_exsleratev2_hal_exsleratev2_InputTileData_input_tiled_buffer_size(
              input_tile_data);
      uint8_t fallback_mode =
          iree_exsleratev2_hal_exsleratev2_InputTileData_execution_mode(
              input_tile_data);
      flatbuffers_string_t kernel_name =
          iree_exsleratev2_hal_exsleratev2_InputTileData_kernel_name(
              input_tile_data);
      float quant_scale =
          iree_exsleratev2_hal_exsleratev2_InputTileData_quant_scale(
              input_tile_data);
      int32_t quant_zero_point =
          iree_exsleratev2_hal_exsleratev2_InputTileData_quant_zero_point(
              input_tile_data);
      float leaky_relu_alpha =
          iree_exsleratev2_hal_exsleratev2_InputTileData_leaky_relu_alpha(
              input_tile_data);
      float requant_output_scale =
          iree_exsleratev2_hal_exsleratev2_InputTileData_requant_output_scale(
              input_tile_data);
      float dequant_scale =
          iree_exsleratev2_hal_exsleratev2_InputTileData_dequant_scale(
              input_tile_data);
      uint64_t output_byte_size =
          iree_exsleratev2_hal_exsleratev2_InputTileData_output_byte_size(
              input_tile_data);
      uint8_t output_element_type =
          iree_exsleratev2_hal_exsleratev2_InputTileData_output_element_type(
              input_tile_data);
      uint32_t activation_binding_idx =
          iree_exsleratev2_hal_exsleratev2_InputTileData_activation_binding_idx(
              input_tile_data);
      bool has_filter_binding =
          iree_exsleratev2_hal_exsleratev2_InputTileData_has_filter_binding(
              input_tile_data);
      uint32_t filter_binding_idx =
          iree_exsleratev2_hal_exsleratev2_InputTileData_filter_binding_idx(
              input_tile_data);
      uint64_t filter_byte_offset =
          iree_exsleratev2_hal_exsleratev2_InputTileData_filter_byte_offset(
              input_tile_data);
      bool has_bias_binding =
          iree_exsleratev2_hal_exsleratev2_InputTileData_has_bias_binding(
              input_tile_data);
      uint32_t bias_binding_idx =
          iree_exsleratev2_hal_exsleratev2_InputTileData_bias_binding_idx(
              input_tile_data);
      uint64_t bias_byte_offset =
          iree_exsleratev2_hal_exsleratev2_InputTileData_bias_byte_offset(
              input_tile_data);
      uint64_t input_byte_size =
          iree_exsleratev2_hal_exsleratev2_InputTileData_input_byte_size(
              input_tile_data);
      uint8_t input_element_type =
          iree_exsleratev2_hal_exsleratev2_InputTileData_input_element_type(
              input_tile_data);

      printf("  InputTileData:\n");
      printf("    input_tile_height: %u\n", tile_h);
      printf("    input_tile_width: %u\n", tile_w);
      printf("    output_tile_height: %u\n", out_tile_h);
      printf("    output_tile_width: %u\n", out_tile_w);
      printf("    input_tiled_buffer_size: %u\n", input_tiled_buffer_size);
      printf("    execution_mode: %u ", fallback_mode);
      switch (fallback_mode) {
        case 0:
          printf("(hardware)\n");
          break;
        case 1:
          printf("(cpu)\n");
          break;
        case 2:
          printf("(skip)\n");
          break;
        default:
          printf("(unknown)\n");
          break;
      }
      printf("    kernel_name: %s\n", kernel_name ? kernel_name : "(empty)");
      printf("    quant_scale: %f\n", quant_scale);
      printf("    quant_zero_point: %d\n", quant_zero_point);
      printf("    leaky_relu_alpha: %f\n", leaky_relu_alpha);
      printf("    requant_output_scale: %f\n", requant_output_scale);
      printf("    dequant_scale: %f\n", dequant_scale);
      printf("    output_byte_size: %lu\n", output_byte_size);
      printf("    output_element_type: %u\n", output_element_type);
      printf("    activation_binding_idx: %u\n", activation_binding_idx);
      printf("    has_filter_binding: %s\n",
             has_filter_binding ? "true" : "false");
      printf("    filter_binding_idx: %u\n", filter_binding_idx);
      printf("    filter_byte_offset: %lu\n", filter_byte_offset);
      printf("    has_bias_binding: %s\n", has_bias_binding ? "true" : "false");
      printf("    bias_binding_idx: %u\n", bias_binding_idx);
      printf("    bias_byte_offset: %lu\n", bias_byte_offset);
      printf("    input_byte_size: %lu\n", input_byte_size);
      printf("    input_element_type: %u\n", input_element_type);
      bool skip_input_tiling =
          iree_exsleratev2_hal_exsleratev2_InputTileData_skip_input_tiling(
              input_tile_data);
      bool skip_output_detiling =
          iree_exsleratev2_hal_exsleratev2_InputTileData_skip_output_detiling(
              input_tile_data);
      bool output_layout_chw =
          iree_exsleratev2_hal_exsleratev2_InputTileData_output_layout_chw(
              input_tile_data);
      printf("    skip_input_tiling: %s\n",
             skip_input_tiling ? "true" : "false");
      printf("    skip_output_detiling: %s\n",
             skip_output_detiling ? "true" : "false");
      printf("    output_layout_chw: %s\n",
             output_layout_chw ? "true" : "false");
    }

    iree_exsleratev2_hal_exsleratev2_DataBufferDef_vec_t data_buffers =
        iree_exsleratev2_hal_exsleratev2_LayerDef_data_buffers(layer);
    if (data_buffers) {
      size_t buffer_count =
          iree_exsleratev2_hal_exsleratev2_DataBufferDef_vec_len(data_buffers);
      printf("  Data Buffers (%zu):\n", buffer_count);
      for (size_t j = 0; j < buffer_count; j++) {
        iree_exsleratev2_hal_exsleratev2_DataBufferDef_table_t data_buffer =
            iree_exsleratev2_hal_exsleratev2_DataBufferDef_vec_at(data_buffers,
                                                                  j);
        uint8_t category =
            iree_exsleratev2_hal_exsleratev2_DataBufferDef_category(
                data_buffer);

        flatbuffers_generic_t buffer =
            iree_exsleratev2_hal_exsleratev2_DataBufferDef_buffer(data_buffer);

        switch (category) {
          case 0: {
            printf("    %zu: category=%u (filter)\n", j, category);
            auto int8_buffer =
                (iree_exsleratev2_hal_exsleratev2_Int8Buffer_table_t)buffer;
            flatbuffers_int8_vec_t data =
                iree_exsleratev2_hal_exsleratev2_Int8Buffer_data(int8_buffer);

            printf("      Int8Buffer (%zu): [", flatbuffers_int8_vec_len(data));
            for (size_t k = 0; k < flatbuffers_int8_vec_len(data); k++) {
              printf("%d ", flatbuffers_int8_vec_at(data, k));
            }
            printf("]\n");
            break;
          }

          case 1: {
            printf("    %zu: category=%u (bias)\n", j, category);

            auto int32_buffer =
                (iree_exsleratev2_hal_exsleratev2_Int32Buffer_table_t)buffer;
            flatbuffers_int32_vec_t data =
                iree_exsleratev2_hal_exsleratev2_Int32Buffer_data(int32_buffer);

            printf("      Int32Buffer (%zu): [",
                   flatbuffers_int32_vec_len(data));
            for (size_t k = 0; k < flatbuffers_int32_vec_len(data); k++) {
              printf("%d ", flatbuffers_int32_vec_at(data, k));
            }
            printf("]\n");
            break;
          }

          case 2: {
            printf("    %zu: category=%u (input)\n", j, category);
            break;
          }

          case 3: {
            printf("    %zu: category=%u (output)\n", j, category);
            break;
          }

          case 8: {
            printf("    %zu: category=%u (bnbias)\n", j, category);
            auto int32_buffer =
                (iree_exsleratev2_hal_exsleratev2_Int32Buffer_table_t)buffer;
            flatbuffers_int32_vec_t data =
                iree_exsleratev2_hal_exsleratev2_Int32Buffer_data(int32_buffer);

            printf("      Int32Buffer (%zu): [",
                   flatbuffers_int32_vec_len(data));
            for (size_t k = 0; k < flatbuffers_int32_vec_len(data); k++) {
              printf("%d ", flatbuffers_int32_vec_at(data, k));
            }
            printf("]\n");
            break;
          }

          case 7: {
            printf("    %zu: category=%u (bnweight)\n", j, category);
            break;
          }

          case 9: {
            printf("    %zu: category=%u (lut)\n", j, category);
            break;
          }

          case 4: {
            printf("    %zu: category=%u (atomicBank)\n", j, category);
            auto uint32_buffer =
                (iree_exsleratev2_hal_exsleratev2_Uint32Buffer_table_t)buffer;
            flatbuffers_uint32_vec_t data =
                iree_exsleratev2_hal_exsleratev2_Uint32Buffer_data(
                    uint32_buffer);

            printf("      Uint32Buffer (%zu): [",
                   flatbuffers_uint32_vec_len(data));
            for (size_t k = 0; k < flatbuffers_uint32_vec_len(data); k++) {
              printf("%u ", flatbuffers_uint32_vec_at(data, k));
            }
            printf("]\n");
            break;
          }

          case 5: {
            printf("    %zu: category=%u (atomicOffset)\n", j, category);
            auto uint32_buffer =
                (iree_exsleratev2_hal_exsleratev2_Uint32Buffer_table_t)buffer;
            flatbuffers_uint32_vec_t data =
                iree_exsleratev2_hal_exsleratev2_Uint32Buffer_data(
                    uint32_buffer);

            printf("      Uint32Buffer (%zu): [",
                   flatbuffers_uint32_vec_len(data));
            for (size_t k = 0; k < flatbuffers_uint32_vec_len(data); k++) {
              printf("%u ", flatbuffers_uint32_vec_at(data, k));
            }
            printf("]\n");
            break;
          }

          case 6: {
            printf("    %zu: category=%u (lifetime)\n", j, category);
            auto uint32_buffer =
                (iree_exsleratev2_hal_exsleratev2_Uint32Buffer_table_t)buffer;
            flatbuffers_uint32_vec_t data =
                iree_exsleratev2_hal_exsleratev2_Uint32Buffer_data(
                    uint32_buffer);

            printf("      Uint32Buffer (%zu): [",
                   flatbuffers_uint32_vec_len(data));
            for (size_t k = 0; k < flatbuffers_uint32_vec_len(data); k++) {
              printf("%u ", flatbuffers_uint32_vec_at(data, k));
            }
            printf("]\n");
            break;
          }

          default:
            printf("    %zu: category=%u [WARN] Unknown DataBuffer category\n",
                   j, category);
            break;
        }
      }
    }
  }
}

// Dumps the newer, experimental exsleratev2.fbs ("exsleratev2_ll") schema
// produced by the EXSLHL->EXSLLL lowering path. As of this writing this
// serializer is opt-in at compile time and has no runtime consumer yet; it
// writes to a separate "<name>.exsl_ll.slfb" file alongside the legacy
// ".slfb".
void deserializeFromExslLlSLFb(const void* buffer) {
  iree_exsleratev2_hal_exsleratev2_ll_ExecutableDef_table_t executable =
      iree_exsleratev2_hal_exsleratev2_ll_ExecutableDef_as_root(buffer);

  if (!executable) {
    printf("Invalid FlatBuffer format (exsleratev2_ll)\n");
    return;
  }

  iree_exsleratev2_hal_exsleratev2_ll_EntryPointDef_vec_t entry_points =
      iree_exsleratev2_hal_exsleratev2_ll_ExecutableDef_entry_points(
          executable);
  if (entry_points) {
    size_t count =
        iree_exsleratev2_hal_exsleratev2_ll_EntryPointDef_vec_len(
            entry_points);
    printf("Found %zu entry points:\n", count);
    for (size_t i = 0; i < count; i++) {
      iree_exsleratev2_hal_exsleratev2_ll_EntryPointDef_table_t entry_point =
          iree_exsleratev2_hal_exsleratev2_ll_EntryPointDef_vec_at(
              entry_points, i);
      const char* name =
          iree_exsleratev2_hal_exsleratev2_ll_EntryPointDef_name(entry_point);
      uint32_t ordinal =
          iree_exsleratev2_hal_exsleratev2_ll_EntryPointDef_ordinal(
              entry_point);
      uint32_t layer_index =
          iree_exsleratev2_hal_exsleratev2_ll_EntryPointDef_layer_index(
              entry_point);
      printf("  %zu: %s (ordinal=%u, layer_index=%u)\n", i,
             name ? name : "(null)", ordinal, layer_index);
    }
  }

  flatbuffers_uint8_vec_t cpu_code =
      iree_exsleratev2_hal_exsleratev2_ll_ExecutableDef_cpu_code(executable);
  if (cpu_code) {
    printf("CPU Code size: %zu bytes\n", flatbuffers_uint8_vec_len(cpu_code));
  }

  const char* cpu_func_name =
      iree_exsleratev2_hal_exsleratev2_ll_ExecutableDef_cpu_function_name(
          executable);
  if (cpu_func_name) {
    printf("CPU Function Name: %s\n", cpu_func_name);
  }

  iree_exsleratev2_hal_exsleratev2_ll_LayerDef_vec_t layers =
      iree_exsleratev2_hal_exsleratev2_ll_ExecutableDef_layers(executable);
  if (!layers) {
    printf("No layers found\n");
    return;
  }

  size_t layer_count =
      iree_exsleratev2_hal_exsleratev2_ll_LayerDef_vec_len(layers);
  printf("Found %zu layers:\n", layer_count);

  for (size_t i = 0; i < layer_count; i++) {
    iree_exsleratev2_hal_exsleratev2_ll_LayerDef_table_t layer =
        iree_exsleratev2_hal_exsleratev2_ll_LayerDef_vec_at(layers, i);
    printf("Layer %zu:\n", i);
    if (!layer) {
      printf("  ERROR: NULL layer pointer!\n");
      continue;
    }

    uint8_t kind = iree_exsleratev2_hal_exsleratev2_ll_LayerDef_kind(layer);
    printf("  kind: %u (%s)\n", kind, llLayerKindName(kind));

    iree_exsleratev2_hal_exsleratev2_ll_ExecutionDef_table_t execution =
        iree_exsleratev2_hal_exsleratev2_ll_LayerDef_execution(layer);
    if (execution) {
      uint8_t mode =
          iree_exsleratev2_hal_exsleratev2_ll_ExecutionDef_mode(execution);
      const char* kernel_name =
          iree_exsleratev2_hal_exsleratev2_ll_ExecutionDef_kernel_name(
              execution);
      printf("  Execution: mode=%u (%s), kernel_name=%s\n", mode,
             llExecutionModeName(mode), kernel_name ? kernel_name : "(none)");
    }

    iree_exsleratev2_hal_exsleratev2_ll_TileDef_table_t tiles =
        iree_exsleratev2_hal_exsleratev2_ll_LayerDef_tiles(layer);
    if (tiles) {
      printf(
          "  Tiles: input=%ux%u output=%ux%u input_tile_buffer_size=%u\n",
          iree_exsleratev2_hal_exsleratev2_ll_TileDef_input_tile_height(
              tiles),
          iree_exsleratev2_hal_exsleratev2_ll_TileDef_input_tile_width(tiles),
          iree_exsleratev2_hal_exsleratev2_ll_TileDef_output_tile_height(
              tiles),
          iree_exsleratev2_hal_exsleratev2_ll_TileDef_output_tile_width(
              tiles),
          iree_exsleratev2_hal_exsleratev2_ll_TileDef_input_tile_buffer_size(
              tiles));
    }

    iree_exsleratev2_hal_exsleratev2_ll_QuantDef_table_t quant =
        iree_exsleratev2_hal_exsleratev2_ll_LayerDef_quant(layer);
    if (quant) {
      printf(
          "  Quant: scale=%f zero_point=%d leaky_relu_alpha=%f "
          "requant_output_scale=%f dequant_scale=%f\n",
          iree_exsleratev2_hal_exsleratev2_ll_QuantDef_quant_scale(quant),
          iree_exsleratev2_hal_exsleratev2_ll_QuantDef_quant_zero_point(
              quant),
          iree_exsleratev2_hal_exsleratev2_ll_QuantDef_leaky_relu_alpha(
              quant),
          iree_exsleratev2_hal_exsleratev2_ll_QuantDef_requant_output_scale(
              quant),
          iree_exsleratev2_hal_exsleratev2_ll_QuantDef_dequant_scale(quant));
    }

    iree_exsleratev2_hal_exsleratev2_ll_BindingDef_table_t bindings =
        iree_exsleratev2_hal_exsleratev2_ll_LayerDef_bindings(layer);
    if (bindings) {
      bool has_filter_binding =
          iree_exsleratev2_hal_exsleratev2_ll_BindingDef_has_filter_binding(
              bindings);
      bool has_bias_binding =
          iree_exsleratev2_hal_exsleratev2_ll_BindingDef_has_bias_binding(
              bindings);
      printf("  Bindings: activation_binding_index=%u\n",
             iree_exsleratev2_hal_exsleratev2_ll_BindingDef_activation_binding_index(
                 bindings));
      printf(
          "    filter: has_binding=%s index=%u byte_offset=%lu\n",
          has_filter_binding ? "true" : "false",
          iree_exsleratev2_hal_exsleratev2_ll_BindingDef_filter_binding_index(
              bindings),
          (unsigned long)
              iree_exsleratev2_hal_exsleratev2_ll_BindingDef_filter_byte_offset(
                  bindings));
      printf(
          "    bias: has_binding=%s index=%u byte_offset=%lu\n",
          has_bias_binding ? "true" : "false",
          iree_exsleratev2_hal_exsleratev2_ll_BindingDef_bias_binding_index(
              bindings),
          (unsigned long)
              iree_exsleratev2_hal_exsleratev2_ll_BindingDef_bias_byte_offset(
                  bindings));
    }

    iree_exsleratev2_hal_exsleratev2_ll_BufferInfoDef_table_t buffer_info =
        iree_exsleratev2_hal_exsleratev2_ll_LayerDef_buffer_info(layer);
    if (buffer_info) {
      uint8_t input_element_type =
          iree_exsleratev2_hal_exsleratev2_ll_BufferInfoDef_input_element_type(
              buffer_info);
      uint8_t output_element_type =
          iree_exsleratev2_hal_exsleratev2_ll_BufferInfoDef_output_element_type(
              buffer_info);
      printf(
          "  BufferInfo: input_byte_size=%lu (%s), "
          "output_byte_size=%lu (%s)\n",
          (unsigned long)
              iree_exsleratev2_hal_exsleratev2_ll_BufferInfoDef_input_byte_size(
                  buffer_info),
          llElementTypeName(input_element_type),
          (unsigned long)
              iree_exsleratev2_hal_exsleratev2_ll_BufferInfoDef_output_byte_size(
                  buffer_info),
          llElementTypeName(output_element_type));
    }

    iree_exsleratev2_hal_exsleratev2_ll_LayoutDef_table_t layout =
        iree_exsleratev2_hal_exsleratev2_ll_LayerDef_layout(layer);
    if (layout) {
      printf(
          "  Layout: skip_input_tiling=%s skip_output_detiling=%s "
          "output_layout_chw=%s\n",
          iree_exsleratev2_hal_exsleratev2_ll_LayoutDef_skip_input_tiling(
              layout)
              ? "true"
              : "false",
          iree_exsleratev2_hal_exsleratev2_ll_LayoutDef_skip_output_detiling(
              layout)
              ? "true"
              : "false",
          iree_exsleratev2_hal_exsleratev2_ll_LayoutDef_output_layout_chw(
              layout)
              ? "true"
              : "false");
    }

    iree_exsleratev2_hal_exsleratev2_ll_LayerMetaDef_table_t meta =
        iree_exsleratev2_hal_exsleratev2_ll_LayerDef_meta(layer);
    if (meta) {
      printf(
          "  Meta: input_tile_buffer=%u input_offset=%u output_offset=%u "
          "num_channel=%u num_filter=%u\n",
          iree_exsleratev2_hal_exsleratev2_ll_LayerMetaDef_input_tile_buffer(
              meta),
          iree_exsleratev2_hal_exsleratev2_ll_LayerMetaDef_input_offset(meta),
          iree_exsleratev2_hal_exsleratev2_ll_LayerMetaDef_output_offset(
              meta),
          iree_exsleratev2_hal_exsleratev2_ll_LayerMetaDef_num_channel(meta),
          iree_exsleratev2_hal_exsleratev2_ll_LayerMetaDef_num_filter(meta));
    }

    iree_exsleratev2_hal_exsleratev2_ll_CsrMapDef_table_t csr_map =
        iree_exsleratev2_hal_exsleratev2_ll_LayerDef_csr_map(layer);
    if (csr_map) {
      iree_exsleratev2_hal_exsleratev2_ll_CsrEntryDef_vec_t entries =
          iree_exsleratev2_hal_exsleratev2_ll_CsrMapDef_entries(csr_map);
      size_t csr_count =
          entries
              ? iree_exsleratev2_hal_exsleratev2_ll_CsrEntryDef_vec_len(
                    entries)
              : 0;
      printf("  CSR Map (%zu entries):\n", csr_count);
      for (size_t j = 0; j < csr_count; j++) {
        iree_exsleratev2_hal_exsleratev2_ll_CsrEntryDef_table_t entry =
            iree_exsleratev2_hal_exsleratev2_ll_CsrEntryDef_vec_at(entries,
                                                                    j);
        // Note: `name` is declared in the schema but is never populated by
        // the current serializer (address/value pairs only).
        printf("    %zu: address=0x%X value=%u\n", j,
               iree_exsleratev2_hal_exsleratev2_ll_CsrEntryDef_address(entry),
               iree_exsleratev2_hal_exsleratev2_ll_CsrEntryDef_value(entry));
      }
    }

    iree_exsleratev2_hal_exsleratev2_ll_DataBufferDef_vec_t data_buffers =
        iree_exsleratev2_hal_exsleratev2_ll_LayerDef_data_buffers(layer);
    if (data_buffers) {
      size_t buffer_count =
          iree_exsleratev2_hal_exsleratev2_ll_DataBufferDef_vec_len(
              data_buffers);
      printf("  Data Buffers (%zu):\n", buffer_count);
      for (size_t j = 0; j < buffer_count; j++) {
        iree_exsleratev2_hal_exsleratev2_ll_DataBufferDef_table_t data_buffer =
            iree_exsleratev2_hal_exsleratev2_ll_DataBufferDef_vec_at(
                data_buffers, j);
        uint8_t category =
            iree_exsleratev2_hal_exsleratev2_ll_DataBufferDef_category(
                data_buffer);
        uint8_t element_type =
            iree_exsleratev2_hal_exsleratev2_ll_DataBufferDef_element_type(
                data_buffer);
        printf("    %zu: category=%u (%s), element_type=%u (%s)\n", j,
               category, llDataCategoryName(category), element_type,
               llElementTypeName(element_type));

        // The producer picks i8_data vs. i32_data by the source buffer's
        // bit width, not by `category`; u32_data is declared but currently
        // never populated. Print whichever is actually present.
        flatbuffers_int8_vec_t i8_data =
            iree_exsleratev2_hal_exsleratev2_ll_DataBufferDef_i8_data(
                data_buffer);
        flatbuffers_int32_vec_t i32_data =
            iree_exsleratev2_hal_exsleratev2_ll_DataBufferDef_i32_data(
                data_buffer);
        flatbuffers_uint32_vec_t u32_data =
            iree_exsleratev2_hal_exsleratev2_ll_DataBufferDef_u32_data(
                data_buffer);

        if (i8_data) {
          size_t len = flatbuffers_int8_vec_len(i8_data);
          printf("      i8_data (%zu): [", len);
          for (size_t k = 0; k < len; k++) {
            printf("%d ", flatbuffers_int8_vec_at(i8_data, k));
          }
          printf("]\n");
        }
        if (i32_data) {
          size_t len = flatbuffers_int32_vec_len(i32_data);
          printf("      i32_data (%zu): [", len);
          for (size_t k = 0; k < len; k++) {
            printf("%d ", flatbuffers_int32_vec_at(i32_data, k));
          }
          printf("]\n");
        }
        if (u32_data) {
          size_t len = flatbuffers_uint32_vec_len(u32_data);
          printf("      u32_data (%zu): [", len);
          for (size_t k = 0; k < len; k++) {
            printf("%u ", flatbuffers_uint32_vec_at(u32_data, k));
          }
          printf("]\n");
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <path_to_flatbuffer_file>\n", argv[0]);
    return 1;
  }

  const char* filename = argv[1];
  size_t size = 0;
  void* buffer = readEntireFile(filename, &size);
  if (!buffer) {
    return 1;
  }

  int legacy_verify_ret =
      iree_exsleratev2_hal_exsleratev2_ExecutableDef_verify_as_root(buffer,
                                                                     size);
  if (legacy_verify_ret == flatcc_verify_ok) {
    printf("Detected schema: exsleratev2 (legacy)\n\n");
    deserializeFromSLFb(buffer);
    free(buffer);
    return 0;
  }

  int ll_verify_ret =
      iree_exsleratev2_hal_exsleratev2_ll_ExecutableDef_verify_as_root(buffer,
                                                                        size);
  if (ll_verify_ret == flatcc_verify_ok) {
    printf("Detected schema: exsleratev2_ll (experimental)\n\n");
    deserializeFromExslLlSLFb(buffer);
    free(buffer);
    return 0;
  }

  printf("Failed to verify FlatBuffer against either known schema:\n");
  printf("  exsleratev2:    %s\n",
         flatcc_verify_error_string(legacy_verify_ret));
  printf("  exsleratev2_ll: %s\n", flatcc_verify_error_string(ll_verify_ret));
  free(buffer);
  return 1;
}
