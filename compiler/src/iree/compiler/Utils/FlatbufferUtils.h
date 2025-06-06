// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_FLATBUFFERUTILS_H_
#define IREE_COMPILER_UTILS_FLATBUFFERUTILS_H_

#include <stddef.h>
#include <stdint.h>

#include <functional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

// clang-format off: order matters here as some of the LLVM includes conflict:
#include "iree/base/internal/flatcc/building.h"
#include "iree/base/internal/flatcc/debugging.h"
#include "iree/base/internal/flatcc/parsing.h"
// clang-format on

namespace mlir::iree_compiler {

// RAII wrapper for flatcc_builder_t; pass to functions requiring a builder.
//
// Usage:
//   FlatbufferBuilder builder;
//   // NOTE: FlatBuffers are built bottoms-up so we first generate our [uint8]:
//   auto dataRef = builder.streamUint8Vec(...);
//   // ... and then start the table that references it:
//   my_type_start_as_root(builder);
//   my_type_uint8_vec_field_add(builder, dataRef);
//   my_type_end_as_root(builder);
//   // ... and finally capture the results as an mlir::Attribute.
//   auto attr = builder.getBufferAttr(mlirContext);
class FlatbufferBuilder {
public:
  FlatbufferBuilder();
  ~FlatbufferBuilder();

  operator flatcc_builder_t *() { return &builder; }

  // Creates a string with the given string contents (including zeros).
  flatbuffers_string_ref_t createString(StringRef value) {
    if (value.empty())
      return 0;
    return flatbuffers_string_create(*this, value.data(), value.size());
  }

  // Creates a string vector containing all strings in the given range.
  template <typename RangeTy>
  flatbuffers_string_vec_ref_t createStringVec(RangeTy &&Range) {
    auto stringRefs = llvm::map_to_vector<8>(Range, [&](StringRef value) {
      return flatbuffers_string_create(*this, value.data(), value.size());
    });
    if (stringRefs.empty())
      return 0;
    return flatbuffers_string_vec_create(*this, stringRefs.data(),
                                         stringRefs.size());
  }

  // Creates an offset vector with the given values. The source values will not
  // be modified.
  flatbuffers_vec_ref_t createOffsetVec(ArrayRef<flatcc_builder_ref_t> values) {
    if (values.empty())
      return 0;
    return flatcc_builder_create_offset_vector(*this, values.data(),
                                               values.size());
  }

  // Creates an offset vector with the given values.
  // Unlike createOffsetVec this will destroy the input values array during
  // serialization but be much faster.
  flatbuffers_vec_ref_t
  createOffsetVecDestructive(SmallVectorImpl<flatcc_builder_ref_t> &values) {
    if (values.empty())
      return 0;
    return flatcc_builder_create_offset_vector_direct(*this, values.data(),
                                                      values.size());
  }

  // Creates an [int32] vec with the contents of the given range.
  template <typename RangeTy>
  flatbuffers_int32_vec_ref_t createInt32Vec(RangeTy &&Range) {
    if (Range.empty())
      return 0;
    flatbuffers_int32_vec_start(*this);
    for (int32_t v : Range) {
      flatbuffers_int32_vec_push_create(*this, v);
    }
    return flatbuffers_int32_vec_end(*this);
  }

  template <typename RangeTy>
  flatbuffers_int8_vec_ref_t createInt8Vec(RangeTy &&Range) {
    if (Range.empty())
      return 0;
    flatbuffers_int8_vec_start(*this);
    for (int8_t v : Range) {
      flatbuffers_int8_vec_push_create(*this, v);
    }
    return flatbuffers_int8_vec_end(*this);
  }

  template <typename RangeTy>
  flatbuffers_uint32_vec_ref_t createUInt32Vec(RangeTy &&Range) {
    if (Range.empty())
      return 0;
    flatbuffers_uint32_vec_start(*this);
    for (uint32_t v : Range) {
      flatbuffers_uint32_vec_push_create(*this, v);
    }
    return flatbuffers_uint32_vec_end(*this);
  }

  // Provides a raw_ostream that |fn| can use to directly stream into a [uint8]
  // in the FlatBuffer builder.
  //
  // Usage:
  //   auto ref = builder.streamUint8Vec([&](llvm::raw_ostream &stream) {
  //     stream << "foo";
  //     return true;
  //   });
  //   ...
  //   my_type_uint8_vec_field_add(builder, ref);  // use vec reference
  //   ...
  flatbuffers_uint8_vec_ref_t
  streamUint8Vec(std::function<bool(raw_ostream &stream)> fn,
                 size_t alignment = 16);

  // Captures the current contents of the flatbuffer builder and returns them
  // as a shaped `vector<SIZExi8>` dense attr. The builder is left unmodified.
  DenseIntElementsAttr getBufferAttr(MLIRContext *context);

  // Copies the current contents of the flatbuffer builder to the target output
  // stream. The builder is left unmodified.
  //
  // This is reduces a significant large allocation that can happen when trying
  // to stitch together all of the pages that were allocated in the emitter as
  // the FlatBuffer was constructed; here we can just walk over each page and
  // write it out in order without any allocations.
  LogicalResult copyToStream(llvm::raw_ostream &output);

  using print_json_fn_t = int (*)(flatcc_json_printer_t *ctx, const char *buf,
                                  size_t bufsiz);

  // Prints the FlatBuffer in its canonical JSON format to the given stream.
  // The builder is left unmodified.
  //
  // |pretty| enables newlines and indentation; somewhat useful for lit testing
  // (as large byte buffers end up with a byte per line!).
  //
  // |includeDefaults| will force all values, including those that would not
  // be serialized to the binary format due to the default value (0, etc) being
  // omitted.
  //
  // NOTE: JSON representations will also differ structurally from the binary
  // format as reused tables are printed wherever they are used as opposed to
  // referencing the same bytes; meaning that this can't be used to verify that
  // we are correctly memoizing strings/structures/etc.
  LogicalResult printJsonToStream(bool pretty, bool includeDefaults,
                                  print_json_fn_t printJsonFn,
                                  llvm::raw_ostream &output);

private:
  flatcc_builder_t builder;
};

// Allows streaming bytes directly into a FlatBuffer `[uint8]` field.
// The ostream runs in buffered mode and routes all writes into pages
// allocated by the FlatBuffer builder as we grow the output.
//
// Usage:
//   flatbuffers_uint8_vec_start(builder);
//   raw_flatbuffer_uint8_vec_ostream stream(builder);
//   stream << "foo";
//   stream.flush();  // *********** IMPORTANT ***********
//   flatbuffers_uint8_vec_ref_t ref = flatbuffers_uint8_vec_end(builder);
class raw_flatbuffer_uint8_vec_ostream : public llvm::raw_ostream {
public:
  explicit raw_flatbuffer_uint8_vec_ostream(flatcc_builder_t *builder)
      : raw_ostream(/*unbuffered=*/true), builder(builder) {}

  ~raw_flatbuffer_uint8_vec_ostream() override { flush(); }

private:
  void write_impl(const char *Ptr, size_t Size) override {
    flatbuffers_uint8_vec_append(builder,
                                 reinterpret_cast<const uint8_t *>(Ptr), Size);
    pos += Size;
  }

  uint64_t current_pos() const override { return pos - GetNumBytesInBuffer(); }

  flatcc_builder_t *builder;
  uint64_t pos = 0;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_FLATBUFFERUTILS_H_
