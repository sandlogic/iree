// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_ENCODING_UTILS_ELEMENTPACKINGUTILS_H_
#define IREE_COMPILER_DIALECT_ENCODING_UTILS_ELEMENTPACKINGUTILS_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir::iree_compiler {

/// Returns true if the given |bitWidth|, if appearing at runtime-kernel
/// interface, is less than a byte that should be tightly packed together.
bool needToPackSubByteElementBitWidth(unsigned bitWidth);

/// Returns true if the given |shapedType|, if appearing at runtime-kernel
/// interface, has sub-byte element types that should be tightly packed
/// together.
bool needToPackSubByteElements(RankedTensorType shapedType);

/// Legalizes the given |elementType| for storage.
///
/// In IREE, if compiling from the same source model, we control both the
/// runtime and kernel. For such cases, we perform tight packing for supported
/// sub-byte elements, and expand to the next power-of-two bit width for other
/// cases.
Type legalizeStorageElementType(Type elementType);

/// Legalizes the underlying element type of |shapedType| for storage, taking
/// into account the encoding attributes of |shapedType|, if present.
///
/// In IREE, if compiling from the same source model, we control both the
/// runtime and kernel. For such cases, we perform tight packing for supported
/// sub-byte elements, and expand to the next power-of-two bit width for other
/// cases.
Type legalizeStorageElementType(RankedTensorType shapedType);

/// Emits IR with the given |builder| to calculate the total number of bytes
/// required for the given |shapedType| in storage. Returns the value for the
/// final count on success; returns nullptr on failure. Dynamic dimensions in
/// |shapedType| have corresponding values in |dynamicDims|.
Value calculateStorageElementCountInBytes(Location loc,
                                          RankedTensorType shapedType,
                                          ValueRange dynamicDims,
                                          OpBuilder &builder);

/// Emits IR with the given |builder| to calculate the byte offset for the
/// element at the given |linearizedIndex|.
Value calculateStorageElementOffsetInBytes(Location loc,
                                           RankedTensorType originalType,
                                           Value linearizedIndex,
                                           OpBuilder &builder);

/// Registers dynamic output tile sizes (tileH, tileW) for an i8 activation
/// tensor with spatial shape (C, H, W).  Called by the Exsleratev2
/// tile-selector pass at preprocessing time so that
/// calculateStorageElementCountInBytes uses per-op tile sizes instead of
/// the hardware-default fallback (TILE_H=8, TILE_W=4).
void registerExsleratev2TileSizes(int64_t C, int64_t H, int64_t W,
                                  int64_t tileH, int64_t tileW);

/// Clears all registered tile sizes.  Call at the start of each compilation
/// to prevent stale entries from a previous run leaking into the next.
void clearExsleratev2TileSizes();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_ENCODING_UTILS_ELEMENTPACKINGUTILS_H_
