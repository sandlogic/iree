// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/Utils/ElementPackingUtils.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler {

static bool needToPackSubByteElementBitWidthImpl(unsigned bitWidth,
                                                 bool isPackedStorage) {
  // Enable i1 support if requested.
  if (isPackedStorage && bitWidth == 1) {
    return true;
  }
  // Require the original bit width to be some power of two for now to avoid
  // trickiness and weirdness of packing and cross-byte access.
  // Also disallow boolean values for now--they may require separate interface
  // choices.
  return bitWidth < 8 && llvm::isPowerOf2_32(bitWidth) && bitWidth != 1;
}

bool needToPackSubByteElements(RankedTensorType shapedType) {
  unsigned bitWidth = IREE::Util::getTypeBitWidth(shapedType.getElementType());
  // Two paths to enable packed storage for i1 tensors: the attribute or cl
  // option. The cl option will be dropped once frontend supports emitting
  // tensors with attributes.
  bool isPackedStorage = IREE::Encoding::hasPackedStorageAttr(shapedType);
  return needToPackSubByteElementBitWidthImpl(bitWidth, isPackedStorage);
}

Type legalizeStorageElementType(RankedTensorType tensorType) {
  bool isPackedStorage = IREE::Encoding::hasPackedStorageAttr(tensorType);
  Type elementType = tensorType.getElementType();
  // Only handle integers; floats in MLIR all have aligned widths (today).
  auto intType = dyn_cast<IntegerType>(elementType);
  if (!intType) {
    return elementType;
  }

  // For sub-byte elements, default to pack them into bytes.
  unsigned bitWidth = intType.getWidth();
  if (needToPackSubByteElementBitWidthImpl(bitWidth, isPackedStorage)) {
    return elementType;
  }

  // Otherwise, extend them to the next power-of-two bit width.
  unsigned alignedBitWidth =
      IREE::Util::getRoundedElementByteWidth(intType) * 8;
  if (alignedBitWidth == bitWidth) {
    return elementType;
  }
  return IntegerType::get(elementType.getContext(), alignedBitWidth,
                          intType.getSignedness());
}

Value calculateStorageElementCountInBytes(Location loc,
                                          RankedTensorType shapedType,
                                          ValueRange dynamicDims,
                                          OpBuilder &builder) {
 if (auto serializableEncodingAttr =
            IREE::Encoding::getSerializableAttr(shapedType)) {
      return serializableEncodingAttr.calculateStorageSizeInBytes(
          loc, builder, shapedType, dynamicDims);
    }

    

    const int64_t TILE_H = 8;
    const int64_t TILE_W = 4;
    const int64_t CHANNEL_SET_SIZE = 32;

    bool isPackedStorage = clEnableI1Support;
    Type alignedElementType = legalizeStorageElementTypeImpl(
        shapedType.getElementType(), isPackedStorage);
    unsigned elementBits = IREE::Util::getTypeBitWidth(alignedElementType);

    // Only apply tiling for i8 (signed int8) tensors
    bool shouldApplyTiling = shapedType.getElementType().isInteger(8);

  // Calculate all static dims first, if any.
  int64_t staticCount = 1;
  if (!needsPacking) {
    staticCount *= IREE::Util::getRoundedElementByteWidth(alignedElementType);
  }

    int64_t rank = shapedType.getRank();

    for (unsigned i = 0; i < rank; ++i) {
      if (!shapedType.isDynamicDim(i)) {
        int64_t dimSize = shapedType.getDimSize(i);

        // Only apply tiling for i8 tensors
        int64_t tileSize = 1;
        if (shouldApplyTiling) {
          if (rank == 4) {
            if (i == 1) tileSize = CHANNEL_SET_SIZE;
            else if (i == 2) tileSize = TILE_H;
            else if (i == 3) tileSize = TILE_W;
          } else if (rank == 3) {
            if (i == 0) tileSize = CHANNEL_SET_SIZE;
            else if (i == 1) tileSize = TILE_H;
            else if (i == 2) tileSize = TILE_W;
          } else if (rank == 2) {
            // MatMul: [M, N] -> only last dim (N) tiled to 32
            if (i == 1) tileSize = CHANNEL_SET_SIZE;
            // dim[0] (M) stays unchanged
          }
        }

        int64_t tiledDim = ((dimSize + tileSize - 1) / tileSize) * tileSize;
        staticCount *= tiledDim;
      }
    }

    // Scale by dynamic dims, if present.
    auto value =
        arith::ConstantIndexOp::create(builder, loc, staticCount).getResult();

    unsigned dynamicDimIdx = 0;
    for (unsigned i = 0; i < rank; ++i) {
      if (shapedType.isDynamicDim(i)) {
        Value dim = dynamicDims[dynamicDimIdx++];

        // Only apply tiling to dynamic dimensions for i8 tensors
        int64_t tileSize = 1;
        if (shouldApplyTiling) {
          if (rank == 4) {
            if (i == 1) tileSize = CHANNEL_SET_SIZE;
            else if (i == 2) tileSize = TILE_H;
            else if (i == 3) tileSize = TILE_W;
          } else if (rank == 3) {
            if (i == 0) tileSize = CHANNEL_SET_SIZE;
            else if (i == 1) tileSize = TILE_H;
            else if (i == 2) tileSize = TILE_W;
          } else if (rank == 2) {
            // MatMul: [M, N] -> only last dim (N) tiled to 32
            if (i == 1) tileSize = CHANNEL_SET_SIZE;
            // dim[0] (M) stays unchanged
          }
        }

        if (tileSize > 1) {
          Value tileSizeVal = arith::ConstantIndexOp::create(builder, loc, tileSize);
          Value tileSizeMinus1 = arith::ConstantIndexOp::create(builder, loc, tileSize - 1);
          Value sum = builder.createOrFold<arith::AddIOp>(loc, dim, tileSizeMinus1);
          Value divided = builder.createOrFold<arith::DivUIOp>(loc, sum, tileSizeVal);
          dim = builder.createOrFold<arith::MulIOp>(loc, divided, tileSizeVal);
        }

        value = builder.createOrFold<arith::MulIOp>(loc, value, dim);
      }
    }

    // Sub-byte packing requires putting multiple elements in the same byte.
    if (needToPackSubByteElementBitWidthImpl(elementBits, isPackedStorage)) {
      assert(8 % elementBits == 0);
      unsigned byteElements = 8 / elementBits;
      auto divisor = arith::ConstantIndexOp::create(builder, loc, byteElements);
      if (!isPackedStorage && dynamicDims.empty() &&
          (staticCount * elementBits) % 8 != 0) {
        return nullptr;
      }
      value = builder.createOrFold<arith::CeilDivUIOp>(loc, value, divisor);
    }

    return value;

}

Value calculateStorageElementOffsetInBytes(Location loc,
                                           RankedTensorType originalType,
                                           Value linearizedIndex,
                                           OpBuilder &builder) {
  bool isPackedStorage = IREE::Encoding::hasPackedStorageAttr(originalType);
  Type alignedElementType = legalizeStorageElementType(originalType);
  unsigned elementBits = IREE::Util::getTypeBitWidth(alignedElementType);

  // Sub-byte packing requires putting multiple elements in the same byte.
  if (needToPackSubByteElementBitWidthImpl(elementBits, isPackedStorage)) {
    Value byteElements =
        arith::ConstantIndexOp::create(builder, loc, 8 / elementBits);
    // TODO(antiagainst): We may want to emit runtime check to make sure this is
    // divisible.
    return builder.createOrFold<arith::DivUIOp>(loc, linearizedIndex,
                                                byteElements);
  }

  Value elementBytes = arith::ConstantIndexOp::create(
      builder, loc, IREE::Util::getRoundedElementByteWidth(alignedElementType));
  return builder.createOrFold<arith::MulIOp>(loc, linearizedIndex,
                                             elementBytes);
}

} // namespace mlir::iree_compiler
