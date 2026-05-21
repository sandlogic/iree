// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- CPUEncodingExternalModels.cpp --------------------------------------===//
//
// This file implements the following interfaces for CPU backends and the VMVX
// backend:
//
// - IREE::Encoding::LayoutResolverAttr
// - IREE::Encoding::SerializableAttr
// - IREE::Encoding::LayoutMaterializerAttr
// - IREE::Codegen::PackedLayoutMaterializerAttr
// - VerifiableTensorEncoding
//
// In these backends, we transpose narrow-N into narrow-M
// for a combination of reasons:
//
//   1. As linalg.matmul materializes into linalg.mmt4d, which has a transposed
//      RHS and therefore LHS<->RHS symmetry, transposeNarrowN is easy to
//      implement at that level.
//   2. We use ukernels, and this allows writing 2x fewer narrow ukernels.
//   3. Heuristics for cache-friendly dispatch tiling can get complex on CPU,
//      so it is nice that they have fewer narrow cases to consider.
//
// The only current exception to this is Arm SVE. It currently adheres to the
// canonical form of scalable vectorisation and keeps the N dimension to be
// scalable.
// This transposition is made easier by (and was all along part of the idea in)
// the RHS-transposition in mmt4d (the t in mmt4d), as generally with matrix
// multiplication
//
//   B * Transpose(A) == Transpose( A * Transpose(B) )
//
// so in mmt4d terms
//
//   mmt4d(B, A) == Transpose(mmt4d(A, B))
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/ExternalInterfaces/CPUEncodingExternalModels.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/ExternalInterfaces/Utils.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/IR/BuiltinAttributes.h"

#define DEBUG_TYPE "iree-codegen-materialize-encoding"

namespace mlir::iree_compiler::IREE::CPU {

using IREE::Codegen::MaterializeEncodingInfo;
using IREE::Codegen::TileMxNxK;
using IREE::Codegen::TileNxHxWxC;

namespace {

//===----------------------------------------------------------------------===//
// Utilities.
//===----------------------------------------------------------------------===//

static FailureOr<IREE::Codegen::ScalableTileFlags>
getScalableTileFlags(linalg::ContractionDimensions cDims,
                     IREE::Encoding::EncodingAttr encoding,
                     DictionaryAttr config) {
  // TODO(egebeysel): I think this isScalable*Enabled flag should be temporary
  // and the temporary SME flag should probably come next to it.
  if (!isAArch64(config) || !isScalableVectorizationEnabled()) {
    LDBG() << "Pre-conditions to enable scalable tiling are not met!";
    return failure();
  }

  std::optional<unsigned> mDim =
      cDims.m.empty() ? std::nullopt
                      : encoding.mapDimToOperandIndex(cDims.m[0]);
  std::optional<unsigned> nDim =
      cDims.n.empty() ? std::nullopt
                      : encoding.mapDimToOperandIndex(cDims.n[0]);
  std::optional<unsigned> kDim = encoding.mapDimToOperandIndex(cDims.k[0]);
  IREE::Codegen::ScalableTileFlags scalableTiles;
  // TODO(egebeysel): Add logic for SME.
  if (mDim.has_value()) {
    if (hasFeature(config, "+sme")) {
      LDBG() << "SME with data-tiling is not supported yet!";
      return failure();
    }
    scalableTiles.push_back(false);
  }
  if (nDim.has_value()) {
    scalableTiles.push_back(hasFeature(config, "+sve") ||
                            hasFeature(config, "+sve2"));
  }
  if (kDim.has_value()) {
    scalableTiles.push_back(false);
  }
  return scalableTiles;
}

static void transposeInPlace(MaterializeEncodingInfo &info) {
  // Vector cases: nothing to do.
  if (info.innerTileSizes.size() < 2) {
    return;
  }
  // Not a vector case, so all three arrays in `info` have size at least 2,
  // outerDimsPerm may have size 3 if there is a batch dimension, but in all
  // cases, the last 2 entries of each array are M and N, not batch.
  auto transpose = [](auto &a) { std::swap(a[a.size() - 2], a[a.size() - 1]); };
  transpose(info.innerDimsPos);
  transpose(info.innerTileSizes);
  transpose(info.outerDimsPerm);
  if (info.scalableTiles) {
    transpose(info.scalableTiles.value());
  }
}

static RankedTensorType
getExpandedType(RankedTensorType type, bool isBatched, bool isTransposed,
                SmallVectorImpl<ReassociationIndices> &ri) {
  if (!isBatched) {
    ri.assign({{0, 1}, {2, 3}});
    if (!isTransposed) {
      return RankedTensorType::get(
          {1, type.getDimSize(0), 1, type.getDimSize(1)},
          type.getElementType());
    }
    return RankedTensorType::get({type.getDimSize(0), 1, type.getDimSize(1), 1},
                                 type.getElementType());
  }

  ri.assign({{0}, {1, 2}, {3, 4}});
  if (!isTransposed) {
    return RankedTensorType::get(
        {type.getDimSize(0), 1, type.getDimSize(1), 1, type.getDimSize(2)},
        type.getElementType());
  }
  return RankedTensorType::get(
      {type.getDimSize(0), type.getDimSize(1), 1, type.getDimSize(2), 1},
      type.getElementType());
}

/// Given an input Value and a desired output element type, create and return
/// an element-wise linalg::GenericOp that extends the input Value to the
/// output element type. Returns `input` if casting is not needed.
static Value createElementWiseExtUIOp(OpBuilder &builder, Value input,
                                      Location loc, Type outElemType) {
  auto inputType = cast<RankedTensorType>(input.getType());
  if (inputType.getElementType() == outElemType) {
    return input;
  }
  SmallVector<AffineMap> maps(
      2, builder.getMultiDimIdentityMap(inputType.getRank()));
  SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(),
                                                 utils::IteratorType::parallel);
  auto castedType = inputType.clone(outElemType);
  SmallVector<OpFoldResult> inputMixedSizes =
      tensor::getMixedSizes(builder, loc, input);
  Value init =
      tensor::EmptyOp::create(builder, loc, inputMixedSizes, outElemType);
  return linalg::GenericOp::create(
             builder, loc, castedType, input, init, maps, iteratorTypes,
             [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
               Value castRes =
                   arith::ExtUIOp::create(b, nestedLoc, outElemType, args[0])
                       ->getResult(0);
               linalg::YieldOp::create(b, nestedLoc, castRes);
             })
      .getResult(0);
}

static Value getTileConvOperand(Value value, linalg::LinalgOp linalgOp,
                                OpBuilder &builder,
                                SmallVectorImpl<ReassociationIndices> &ri,
                                ArrayRef<Type> elemTypes, int operandIndex) {
  assert(linalgOp.getNumDpsInputs() == 2);
  assert(linalgOp.getNumDpsInits() == 1);
  auto convDims = linalg::inferConvolutionDims(linalgOp);
  Value expandedValue = value;
  return expandedValue;
}
/// If needed, expand and the input Value, and return the resulting input with
/// the canonical mmt4d input shape. If the input element type is unsigned,
/// create a producer Linalg::GenericOp on the input that unsigned extends the
/// input to the output element type. This extension is required to keep the
/// unsignedness information on the input for ukernels. If `transpose` is true,
/// the `linalgOp`'s indexing maps are transposed.
static Value getMmt4dOperand(Value value, linalg::LinalgOp linalgOp,
                             bool transpose, OpBuilder &builder,
                             SmallVectorImpl<ReassociationIndices> &ri,
                             ArrayRef<Type> elemTypes, int operandIdx) {
  assert(linalgOp.getNumDpsInputs() == 2);
  assert(linalgOp.getNumDpsInits() == 1);
  auto cDims = linalg::inferContractionDims(linalgOp);
  Location loc = linalgOp->getLoc();
  Value expandedValue = value;
  // If vecmat with non-rhs operandIdx or matvec with non-lhs operandIdx, the
  // operand is a vector and must be extended
  if ((cDims->m.empty() && operandIdx != 1) ||
      (cDims->n.empty() && operandIdx != 0)) {
    auto type = cast<RankedTensorType>(value.getType());
    RankedTensorType newType = getExpandedType(
        type, /*isBatched=*/!cDims->batch.empty(),
        /*isTransposed=*/operandIdx == 2 && (transpose ^ cDims->n.empty()), ri);
    expandedValue =
        tensor::ExpandShapeOp::create(builder, loc, newType, value, ri);
  }
  if (elemTypes[operandIdx].isUnsignedInteger()) {
    return createElementWiseExtUIOp(builder, expandedValue, loc,
                                    elemTypes.back());
  }
  return expandedValue;
}

/// Returns the best TileMxNxK from `enumeratedTiles` pool. If the
/// `hostDefinedUpperBound` is not empty, the chosen tile sizes can not be
/// greater than the values.
TileMxNxK chooseMatmulTile(ArrayRef<TileMxNxK> enumeratedTiles,
                           IREE::Encoding::MatmulNarrowDim narrowDim) {
  // Handle narrow-N by transposing to reduce to narrow-M. Note: the
  // enumeratedTiles currently only enumerate narrow-M cases.
  if (narrowDim.isN()) {
    narrowDim.dim = IREE::Encoding::MatmulNarrowDim::Dim::M;
    TileMxNxK tile = chooseMatmulTile(enumeratedTiles, narrowDim);
    std::swap(tile.M, tile.N);
    return tile;
  }
  // Handle kDynamic: currently this is only used with VMVX, where there is only
  // one enumerated tile and it has all three M/N/K dimensions dynamic, so for
  // now we only support that. Generalize that as needed when more dynamic tile
  // sizes are used outside of VMVX, e.g. perhaps some day with Arm SVE. Decide
  // how to incorporate the handling of kDynamic in the cost-model evaluation
  // below to decide when to prefer a dynamic vs a static tile shape.
  for (auto tile : enumeratedTiles) {
    if (ShapedType::isDynamic(tile.M) || ShapedType::isDynamic(tile.N) ||
        ShapedType::isDynamic(tile.K)) {
      assert(enumeratedTiles.size() == 1);
      assert(ShapedType::isDynamic(tile.M) && ShapedType::isDynamic(tile.N) &&
             ShapedType::isDynamic(tile.K));
      return tile;
    }
  }
  // We're going to "rate" the enumerated tiles.
  struct RatedTileMxNxK : TileMxNxK {
    RatedTileMxNxK() {}
    RatedTileMxNxK(TileMxNxK tile) : TileMxNxK(tile) {}
    // Penalize tiles that are wider in the M dimension than matmulNarrowM.
    int64_t paddingPenalty = 0;
    // Favor larger tiles, as long as they still minimize paddingPenalty.
    int64_t productMxNxK = 0;
  };
  SmallVector<RatedTileMxNxK> ratedTiles;
  ratedTiles.reserve(enumeratedTiles.size());
  int64_t bestPaddingPenalty = INT64_MAX;
  for (auto tile : enumeratedTiles) {
    RatedTileMxNxK ratedTile(tile);
    ratedTile.paddingPenalty = 0;
    // If we are choosing a tile for a narrow-M case, we want to minimize
    // padding along the M dimension.
    // The PowerOf2Ceil is so that we are OK with padding up to the next
    // power of two, we just try to avoid padding beyond that. For example,
    // if matmulNarrowM==7 and we have enumerated tiles with M=8,4,2,1, we
    // are OK with the tile that has M==8 even though it requires some padding.
    // Otherwise, we would be penalizing the tiles with M==8,4,2 and we would
    // end up selecting the vecmat tile (M==1) for that case!
    if (narrowDim) {
      ratedTile.paddingPenalty =
          std::max<int64_t>(tile.M - llvm::PowerOf2Ceil(narrowDim.size), 0);
    }
    ratedTile.productMxNxK = tile.M * tile.N * tile.K;
    ratedTiles.push_back(ratedTile);
    LDBG() << "candidate: "
           << llvm::interleaved(ArrayRef{tile.M, tile.N, tile.K})
           << " penalty:" << ratedTile.paddingPenalty;
    bestPaddingPenalty = std::min(bestPaddingPenalty, ratedTile.paddingPenalty);
  }
  RatedTileMxNxK bestRatedTile;
  for (auto ratedTile : ratedTiles) {
    // Choose only among tiles that minimize paddingPenalty. Among those,
    // maximize productMxNxK.
    if (ratedTile.paddingPenalty == bestPaddingPenalty &&
        bestRatedTile.productMxNxK < ratedTile.productMxNxK) {
      bestRatedTile = ratedTile;
    }
  }
  // Sanity check. This assert can only fail if there's a programming mistake
  // locally here.
  assert(bestRatedTile.paddingPenalty == bestPaddingPenalty);
  LDBG() << "bestRatedTile: "
         << llvm::interleaved(
                ArrayRef{bestRatedTile.M, bestRatedTile.N, bestRatedTile.K})
         << " penalty:" << bestRatedTile.paddingPenalty;
  return bestRatedTile;
}

TileNxHxWxC chooseConvTile(const SmallVector<TileNxHxWxC> &tiles) {
  // Choose the best tile for the convolution operation.
  for (const auto &tile : tiles) {
    return tile;
  }
  // If no tile matches the dimensions, return the first tile.
  return tiles.front();
}

Operation *lowerContractionOpWithEncoding(
    OpBuilder &builder, linalg::LinalgOp linalgOp, ValueRange operands,
    IREE::Encoding::LayoutMaterializerAttr layoutAttr) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return nullptr;
  }

  auto inputs = linalgOp.getDpsInputOperands();
  auto outputs = linalgOp.getDpsInits();

  auto lhsType = cast<RankedTensorType>(inputs[0]->get().getType());
  auto rhsType = cast<RankedTensorType>(inputs[1]->get().getType());
  auto resultType = cast<RankedTensorType>(outputs[0].getType());
  auto lhsEncoding = IREE::Encoding::getEncodingAttr(lhsType);
  auto rhsEncoding = IREE::Encoding::getEncodingAttr(rhsType);
  auto resultEncoding = IREE::Encoding::getEncodingAttr(resultType);
  if (!lhsEncoding || !rhsEncoding || !resultEncoding) {
    return nullptr;
  }

  if (lhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_LHS ||
      rhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_RHS ||
      resultEncoding.getOperandIndex().getValue() !=
          IREE::Encoding::MATMUL_RESULT) {
    return nullptr;
  }

  MaterializeEncodingInfo encodingInfo = {};
  if (auto packedLayoutAttr =
          dyn_cast<IREE::Codegen::PackedLayoutMaterializerAttr>(layoutAttr)) {
    encodingInfo = packedLayoutAttr.getEncodingInfo(
        cast<RankedTensorType>(linalgOp->getResultTypes()[0]));
  }

  if (isIdentityLayout(encodingInfo)) {
    return dropEncodingAndCloneOp(builder, linalgOp,
                                  operands.take_front(inputs.size()),
                                  operands.drop_front(inputs.size()));
  }

  bool transpose = isNarrowNResult(resultEncoding);
  // Do not transpose in case we have scalable tiles.
  transpose &= llvm::none_of(
      encodingInfo.scalableTiles.value_or(IREE::Codegen::ScalableTileFlags{}),
      [](bool flag) { return flag; });
  SmallVector<Type> elemTypes = lhsEncoding.getElementTypesArray();
  SmallVector<ReassociationIndices> ri;
  Value newLhs = getMmt4dOperand(operands[0], linalgOp, transpose, builder, ri,
                                 elemTypes, /*operandIdx=*/0);
  Value newRhs = getMmt4dOperand(operands[1], linalgOp, transpose, builder, ri,
                                 elemTypes, /*operandIdx=*/1);
  Value newResult = getMmt4dOperand(operands[2], linalgOp, transpose, builder,
                                    ri, elemTypes, /*operandIdx=*/2);
  if (transpose) {
    std::swap(newLhs, newRhs);
  }
  Type newResultType = newResult.getType();
  auto cDims = IREE::Encoding::getEncodingContractionDims(lhsEncoding);
  Operation *result;
  if (cDims->batch.empty()) {
    result = linalg::Mmt4DOp::create(builder, linalgOp.getLoc(), newResultType,
                                     ValueRange{newLhs, newRhs},
                                     ValueRange{newResult});
  } else {
    result = linalg::BatchMmt4DOp::create(
        builder, linalgOp.getLoc(), newResultType, ValueRange{newLhs, newRhs},
        ValueRange{newResult});
  }
  if (!ri.empty()) {
    result = tensor::CollapseShapeOp::create(builder, linalgOp->getLoc(),
                                             operands[2].getType(),
                                             result->getResult(0), ri);
  }
  return result;
}

FailureOr<Operation *> lowerConvolutionOpWithEncoding(
    OpBuilder &builder, linalg::LinalgOp linalgOp, ValueRange operands,
    IREE::Encoding::LayoutMaterializerAttr layoutAttr) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return failure();
  }
  auto inputs = linalgOp.getDpsInputOperands();
  auto outputs = linalgOp.getDpsInits();

  auto lhsType = cast<RankedTensorType>(inputs[0]->get().getType());
  auto rhsType = cast<RankedTensorType>(inputs[1]->get().getType());
  auto resultType = cast<RankedTensorType>(outputs[0].getType());
  auto inputEncoding = IREE::Encoding::getEncodingAttr(lhsType);
  auto filterEncoding = IREE::Encoding::getEncodingAttr(rhsType);
  auto resultEncoding = IREE::Encoding::getEncodingAttr(resultType);
  if (!inputEncoding || !filterEncoding || !resultEncoding) {
    return failure();
  }

  if (inputEncoding.getOperandIndex().getValue() != IREE::Encoding::CONV_LHS ||
      filterEncoding.getOperandIndex().getValue() != IREE::Encoding::CONV_RHS ||
      resultEncoding.getOperandIndex().getValue() !=
          IREE::Encoding::CONV_RESULT) {
    return failure();
  }

  MaterializeEncodingInfo encodingInfo = {};
  if (auto packedLayoutAttr =
          dyn_cast<IREE::Codegen::PackedLayoutMaterializerAttr>(layoutAttr)) {
    encodingInfo = packedLayoutAttr.getEncodingInfo(
        cast<RankedTensorType>(linalgOp->getResultTypes()[0]));
  }

  if (isIdentityLayout(encodingInfo)) {
    return dropEncodingAndCloneOp(builder, linalgOp,
                                  operands.take_front(inputs.size()),
                                  operands.drop_front(inputs.size()));
  }

  Operation *result;
  SmallVector<Type> elemTypes = inputEncoding.getElementTypesArray();
  SmallVector<utils::IteratorType> iterTypesVec =
      linalgOp.getIteratorTypesArray();
  iterTypesVec.append(2, utils::IteratorType::parallel);
  ArrayRef<utils::IteratorType> convertedIterType = iterTypesVec;

  SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
  SmallVector<AffineMap> convertedMaps;
  int64_t originalRank = linalgOp.getStaticLoopRanges().size();
  int64_t convertedRank = originalRank + 2;

  AffineExpr filterTile = builder.getAffineDimExpr(originalRank);
  AffineExpr channelTile = builder.getAffineDimExpr(originalRank + 1);

  for (auto [idx, map] : llvm::enumerate(linalgOp.getIndexingMapsArray())) {
    SmallVector<AffineExpr> results(map.getResults());
    if (idx == IREE::Encoding::CONV_RHS) {
      results.append({filterTile, channelTile});
    } else if (idx == IREE::Encoding::CONV_LHS ||
               idx == IREE::Encoding::CONV_RESULT) {
      results.append({filterTile});
    }
    convertedMaps.push_back(
        AffineMap::get(convertedRank, 0, results, builder.getContext()));
  }

  SmallVector<ReassociationIndices> ri;
  Value newLHS =
      getTileConvOperand(operands[0], linalgOp, builder, ri, elemTypes, 0);
  Value newRHS =
      getTileConvOperand(operands[1], linalgOp, builder, ri, elemTypes, 1);
  Value newResult =
      getTileConvOperand(operands[2], linalgOp, builder, ri, elemTypes, 2);
  Type newResultType = newResult.getType();
  result =
      linalg::GenericOp::create(
          builder, linalgOp.getLoc(), newResultType, ValueRange{newLHS, newRHS},
          ValueRange{newResult}, convertedMaps, convertedIterType,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            Value mul = arith::MulIOp::create(nestedBuilder, nestedLoc, args[0],
                                              args[1])
                            ->getResult(0);
            Value add =
                arith::AddIOp::create(nestedBuilder, nestedLoc, mul, args[2])
                    ->getResult(0);
            linalg::YieldOp::create(nestedBuilder, nestedLoc, add);
          })
          .getOperation();
  return result;
}

//===----------------------------------------------------------------------===//
// EXSLERATEV2-specific conv lowering with full tiled layouts.
//
// Hardware tile constants (tiled_conv_kernel.cc):
//   CHANNEL_SET_SIZE = FILTER_SET_SIZE = 32
//   default spatial tile: tileH=4, tileW=8
//
// Packed layouts (all NHWC):
//   input  [IH, IW, C]     -> [IH/tH, IW/tW, C/32,  tH, tW, 32]
//   filter [KH, KW, C, F]  -> [F/32,  KH,    KW,    C/32, 32, 32]
//   output [OH, OW, F]     -> [OH/tH, OW/tW, F/32,  tH, tW, 32]
//
// 10-D loop nest (matches run_tiled_convolution_impl loops):
//   d0=ty_cube d1=tx_cube d2=fs d3=ky d4=kx d5=cs d6=ly d7=lx d8=filt d9=c
//===----------------------------------------------------------------------===//

static constexpr int64_t kExslTileH = 4;
static constexpr int64_t kExslTileW = 8;
static constexpr int64_t kExslChSet = 32;   // CHANNEL_SET_SIZE
static constexpr int64_t kExslFiltSet = 32; // FILTER_SET_SIZE

// Return the coefficient of affine dim `dimIdx` in `expr`.
// Handles d*c, c*d, and additive compositions thereof.
static int64_t exslExtractStride(AffineExpr expr, unsigned dimIdx) {
  if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
    return dim.getPosition() == dimIdx ? 1 : 0;
  }
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (bin.getKind() == AffineExprKind::Mul) {
      if (auto ld = dyn_cast<AffineDimExpr>(bin.getLHS())) {
        if (auto rc = dyn_cast<AffineConstantExpr>(bin.getRHS())) {
          return ld.getPosition() == dimIdx ? rc.getValue() : 0;
        }
      }
      if (auto rd = dyn_cast<AffineDimExpr>(bin.getRHS())) {
        if (auto lc = dyn_cast<AffineConstantExpr>(bin.getLHS())) {
          return rd.getPosition() == dimIdx ? lc.getValue() : 0;
        }
      }
    }
    if (bin.getKind() == AffineExprKind::Add) {
      return exslExtractStride(bin.getLHS(), dimIdx) +
             exslExtractStride(bin.getRHS(), dimIdx);
    }
  }
  return 0;
}

// Creates the 10-D tiled linalg.generic for EXSLERATEV2 convolution.
// Supports both NHWC (input [H,W,C]) and NCHW (input [C,H,W]) formats.
// Operands in `operands` are expected to already be in packed form (emitted
// by the materialization framework before calling lowerOp).
static FailureOr<Operation *> lowerExsleratev2ConvolutionOpWithEncoding(
    OpBuilder &builder, linalg::LinalgOp linalgOp, ValueRange operands,
    IREE::Encoding::LayoutMaterializerAttr layoutAttr) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return failure();
  }

  auto inputs = linalgOp.getDpsInputOperands();
  auto outputs = linalgOp.getDpsInits();
  if (inputs.size() < 2 || outputs.size() != 1) {
    return failure();
  }

  // Require filter (RHS) to carry CONV_RHS encoding to identify conv ops.
  auto rhsType = cast<RankedTensorType>(inputs[1]->get().getType());
  auto filterEnc = IREE::Encoding::getEncodingAttr(rhsType);
  if (!filterEnc ||
      filterEnc.getOperandIndex().getValue() != IREE::Encoding::CONV_RHS) {
    return failure();
  }

  // Standard 2-D conv has 6 loop dims: OH, OW, F, KH, KW, C.
  int64_t origRank =
      static_cast<int64_t>(linalgOp.getStaticLoopRanges().size());
  if (origRank != 6) {
    return failure();
  }

  SmallVector<AffineMap> origMaps = linalgOp.getIndexingMapsArray();

  // Detect input format from the LHS indexing map:
  //   NHWC [H, W, C]: last result is a pure dim expr (channel)
  //   NCHW [C, H, W]: first result is a pure dim expr (channel)
  AffineMap lhsMap = origMaps[0];
  bool isNHWC = dyn_cast<AffineDimExpr>(
                    lhsMap.getResult(lhsMap.getNumResults() - 1)) != nullptr;
  bool isNCHW =
      !isNHWC && dyn_cast<AffineDimExpr>(lhsMap.getResult(0)) != nullptr;
  if (!isNHWC && !isNCHW) {
    return failure();
  }

  // Stride lives at result[0]/result[1] for NHWC, result[1]/result[2] for NCHW.
  int64_t strideH = isNHWC ? exslExtractStride(lhsMap.getResult(0), 0)
                           : exslExtractStride(lhsMap.getResult(1), 0);
  int64_t strideW = isNHWC ? exslExtractStride(lhsMap.getResult(1), 1)
                           : exslExtractStride(lhsMap.getResult(2), 1);
  if (strideH == 0) {
    strideH = 1;
  }
  if (strideW == 0) {
    strideW = 1;
  }

  // Build 10-D affine maps.
  // Loop dims: d0=ty_cube, d1=tx_cube, d2=fs, d3=ky, d4=kx, d5=cs, d6=ly,
  // d7=lx, d8=filt, d9=c iy = d0*(tH*sH) + d6*sH + d3   (raw input H index) ix
  // = d1*(tW*sW) + d7*sW + d4   (raw input W index)
  auto d = [&](unsigned i) { return builder.getAffineDimExpr(i); };
  AffineExpr iy = d(0) * (kExslTileH * strideH) + d(6) * strideH + d(3);
  AffineExpr ix = d(1) * (kExslTileW * strideW) + d(7) * strideW + d(4);

  // Input:  [IH/tH, IW/tW, IC/32, tH, tW, 32]
  AffineMap inputMap =
      AffineMap::get(10, 0,
                     {iy.floorDiv(kExslTileH), ix.floorDiv(kExslTileW), d(5),
                      iy % kExslTileH, ix % kExslTileW, d(9)},
                     builder.getContext());

  // Filter: [F/32, KH, KW, C/32, 32, 32]
  AffineMap filterMap = AffineMap::get(
      10, 0, {d(2), d(3), d(4), d(5), d(8), d(9)}, builder.getContext());

  // Output: [OH/tH, OW/tW, F/32, tH, tW, 32]
  AffineMap outputMap = AffineMap::get(
      10, 0, {d(0), d(1), d(2), d(6), d(7), d(8)}, builder.getContext());

  SmallVector<utils::IteratorType> iterTypes = {
      utils::IteratorType::parallel,  // d0: ty_cube
      utils::IteratorType::parallel,  // d1: tx_cube
      utils::IteratorType::parallel,  // d2: fs
      utils::IteratorType::reduction, // d3: ky
      utils::IteratorType::reduction, // d4: kx
      utils::IteratorType::reduction, // d5: cs
      utils::IteratorType::parallel,  // d6: ly
      utils::IteratorType::parallel,  // d7: lx
      utils::IteratorType::parallel,  // d8: filt_idx
      utils::IteratorType::reduction, // d9: c
  };

  // Packed operands pre-packed by MaterializeDeviceEncoding:
  //   operands[0]:              input  [IH/tH, IW/tW, IC/32, tH, tW, 32]
  //   operands[1]:              filter [F/32, KH, KW, IC/32, 32, 32]
  //   operands[2..N-1]:         extra inputs (e.g. zero points) — 0D, passed through
  //   operands[inputs.size()]:  output [OH/tH, OW/tW, F/32, tH, tW, 32]
  Value packedIn = operands[0];
  Value packedFilter = operands[1];
  size_t numExtraInputs = inputs.size() - 2;
  Value packedOut = operands[inputs.size()];
  Location loc = linalgOp.getLoc();

  // The inputMap uses (iy.floorDiv(tH), ix.floorDiv(tW)) where iy/ix can
  // exceed the packed tile count by 1 for the rightmost tile when the kernel
  // stencil extends past the tile boundary.  Pad the packed input by
  // ceil((kH-1)/tH) and ceil((kW-1)/tW) extra tiles so the MLIR linalg
  // verifier sees in-bounds accesses.  The hardware kernel handles partial
  // last tiles at runtime, so these extra zero elements are never accessed.
  {
    auto packedFilterTy = cast<RankedTensorType>(packedFilter.getType());
    // Packed filter layout: [F/32, KH, KW, IC/32, 32, 32]
    int64_t kH = packedFilterTy.getShape()[1];
    int64_t kW = packedFilterTy.getShape()[2];
    int64_t extraH = (kH + kExslTileH - 2) / kExslTileH;
    int64_t extraW = (kW + kExslTileW - 2) / kExslTileW;
    if (extraH > 0 || extraW > 0) {
      auto packedInTy = cast<RankedTensorType>(packedIn.getType());
      SmallVector<int64_t> paddedShape(packedInTy.getShape());
      paddedShape[0] += extraH;
      paddedShape[1] += extraW;
      auto paddedTy =
          RankedTensorType::get(paddedShape, packedInTy.getElementType());
      Value zeroPad = arith::ConstantOp::create(
          builder, loc, builder.getZeroAttr(packedInTy.getElementType()));
      SmallVector<OpFoldResult> low(6, builder.getIndexAttr(0));
      SmallVector<OpFoldResult> high = {
          builder.getIndexAttr(extraH), builder.getIndexAttr(extraW),
          builder.getIndexAttr(0),      builder.getIndexAttr(0),
          builder.getIndexAttr(0),      builder.getIndexAttr(0)};
      packedIn = tensor::PadOp::create(builder, loc, paddedTy, packedIn, low,
                                       high, zeroPad);
    }
  }

  auto packedOutType = cast<RankedTensorType>(packedOut.getType());
  Type i32Ty = builder.getI32Type();

  // Build input list and affine maps: packed tensors first, then extra 0D inputs.
  SmallVector<Value> genericInputs = {packedIn, packedFilter};
  SmallVector<AffineMap> allMaps = {inputMap, filterMap};
  AffineMap zeroMap = AffineMap::get(10, 0, {}, builder.getContext());
  for (size_t i = 2; i < inputs.size(); i++) {
    genericInputs.push_back(operands[i]);
    allMaps.push_back(zeroMap);
  }
  allMaps.push_back(outputMap);

  // Index of the output accumulator block arg inside the generic body.
  size_t accumArgIdx = 2 + numExtraInputs;

  // 10-D tiled generic matching the tiled_conv2d kernel loop structure.
  // Extra zero-point inputs (if any) are carried through with 0D maps and
  // ignored in the body — hardware lowering via emitCSRFromTiledConv handles
  // quantization separately from the body computation.
  Value result10D =
      linalg::GenericOp::create(
          builder, loc, packedOutType, genericInputs,
          ValueRange{packedOut},
          ArrayRef<AffineMap>(allMaps), iterTypes,
          [&](OpBuilder &nb, Location nb_loc, ValueRange args) {
            Value lhs = arith::ExtSIOp::create(nb, nb_loc, i32Ty, args[0])
                            ->getResult(0);
            Value rhs = arith::ExtSIOp::create(nb, nb_loc, i32Ty, args[1])
                            ->getResult(0);
            Value mul =
                arith::MulIOp::create(nb, nb_loc, lhs, rhs)->getResult(0);
            Value add =
                arith::AddIOp::create(nb, nb_loc, mul, args[accumArgIdx])
                    ->getResult(0);
            linalg::YieldOp::create(nb, nb_loc, add);
          })
          .getResult(0);
  return result10D.getDefiningOp();
}

//===----------------------------------------------------------------------===//
// Interface methods implementation for iree_cpu.cpu_encoding_resolver.
//===----------------------------------------------------------------------===//

// Enumerate tile sizes to choose from on riscv32.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTileRiscv32(DictionaryAttr config) {
  if (hasUkernel(config)) {
    return {
        TileMxNxK{8, 8, 4}, // Some reasonable tile shape.
        TileMxNxK{4, 8, 4}, // Truncation of the above.
        TileMxNxK{2, 8, 4}, // Truncation of the above.
        TileMxNxK{1, 8, 4}, // Truncation of the above.
    };
  }
  // Fallback - no architecture-optimized tile size for this case.
  return {};
}
// RISC-V has vector register length extensions: zvl128b, zvl256b etc.
// If these extension are specified in target cpu feature,
// they can be used to determine VLEN. This function assumes that
// 'v' feature is present
size_t getRISCVVVlenFromCPUFeatures(DictionaryAttr config) {
  // If +zvl* feature is not explicitly specified,
  // fallback to +zvl128b, as spec specifies minimum VLEN
  // of 128b for the V extension: https://rb.gy/p8rbzv
  size_t vlen;
  if (hasFeature(config, "+zvl65536b")) {
    vlen = 65536;
  } else if (hasFeature(config, "+zvl32768b")) {
    vlen = 32768;
  } else if (hasFeature(config, "+zvl16384b")) {
    vlen = 16384;
  } else if (hasFeature(config, "+zvl8192b")) {
    vlen = 8192;
  } else if (hasFeature(config, "+zvl4096b")) {
    vlen = 4096;
  } else if (hasFeature(config, "+zvl2048b")) {
    vlen = 2048;
  } else if (hasFeature(config, "+zvl1024b")) {
    vlen = 1024;
  } else if (hasFeature(config, "+zvl512b")) {
    vlen = 512;
  } else if (hasFeature(config, "+zvl256b")) {
    vlen = 256;
  } else {
    vlen = 128;
  }
  return vlen;
}
// Enumerate tile sizes to choose from on riscv64.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTileRiscv64(TypeRange elementTypes, DictionaryAttr config) {

  // Data-Tiling is only implemented for the V extension
  if (!hasFeature(config, "+v")) {
    return {};
  }
  size_t vlen = getRISCVVVlenFromCPUFeatures(config);
  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];
  if (lhs.isF32() && rhs.isF32() && out.isF32()) {
    // VLEN-aware Tile size selection
    // One concern that needs to be addressed here is that
    // for larger VLENs tile sizes would be very large
    // leading to a very high padding overhead
    int N0 = vlen / 8;
    return {
        TileMxNxK{7, N0, 1}, // Aim to use vfmacc, 100% register utilization.
        TileMxNxK{4, N0, 1}, // Truncation of the above.
        TileMxNxK{2, N0, 1}, // Truncation of the above.
        TileMxNxK{1, N0, 1}, // Truncation of the above.
    };
  }
  if (lhs.isF16() && rhs.isF16()) {
    int N0 = vlen / 8;
    if (hasFeature(config, "+zvfh")) {
      return {
          TileMxNxK{7, N0, 1},
          TileMxNxK{4, N0, 1}, // Truncation of the above.
          TileMxNxK{2, N0, 1}, // Truncation of the above.
          TileMxNxK{1, N0, 1}, // Truncation of the above.
      };
    }
    if (hasFeature(config, "+zvfhmin")) {
      return {
          TileMxNxK{6, N0, 1},
          TileMxNxK{4, N0, 1}, // Truncation of the above.
          TileMxNxK{2, N0, 1}, // Truncation of the above.
          TileMxNxK{1, N0, 1}, // Truncation of the above.
      };
    }
  }
  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

// Enumerate tile sizes to choose from on arm64.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK> enumerateMatmulTileArm64(TypeRange elementTypes,
                                                       DictionaryAttr config) {
  // For SVE and scalable vectors, this methods selects base sizes that match
  // the NEON fixed-width sizes.
  // TODO: Add SME inner tile sizes and corresponding tests.
  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];

  if (out.isF32() || out.isF16() || out.isBF16()) {
    if (lhs.isBF16() && rhs.isBF16() && (out.isBF16() || out.isF32()) &&
        hasFeature(config, "+bf16")) {
      return {
          TileMxNxK{8, 8, 4}, // Aim to use BFMMLA.
          TileMxNxK{4, 8, 4}, // Truncation of the above.
          TileMxNxK{2, 8, 4}, // Truncation of the above.
          TileMxNxK{1, 8, 4}, // Truncation of the above.
      };
    }
    if (isa<FloatType>(lhs) && isa<FloatType>(rhs)) {
      // Note: 16-bit floating point types currently use the same tile size as
      // f32. This makes sense when either (1) the accumulator is f32, or (2)
      // the arithmetic will have to expand f16 to f32 in registers. We may
      // reconsider when taking advantage of native f16/bf16 arithmetic when the
      // accumulator itself is f16/bf16, as we could typically have a 2x wider
      // tile in that case. However, on current CPUs, the existing tiles seem
      // wide enough already to approach peak performance.
      return {
          TileMxNxK{8, 8, 1}, // Aim to use FMLA or FMLAL.
          TileMxNxK{4, 8, 1}, // Truncation of the above.
          TileMxNxK{2, 8, 1}, // Truncation of the above.
          TileMxNxK{1, 8, 1}, // Truncation of the above.
      };
    }
  }
  if (lhs.isSignlessInteger(8) && rhs.isSignlessInteger(8) &&
      out.isSignlessInteger(32)) {
    if (hasFeature(config, "+i8mm")) {
      return {
          TileMxNxK{8, 8, 8}, // Aim to use SMMLA.
          TileMxNxK{4, 8, 8}, // Truncation of the above.
          TileMxNxK{2, 8, 8}, // Truncation of the above.
          TileMxNxK{1, 8, 8}, // Truncation of the above.
      };
    }
    if (hasFeature(config, "+dotprod")) {
      return {
          TileMxNxK{8, 8, 4}, // Aim to use SDOT.
          TileMxNxK{4, 8, 4}, // Truncation of the above.
          TileMxNxK{2, 8, 4}, // Truncation of the above.
          TileMxNxK{1, 8, 4}, // Truncation of the above.
      };
    }
  }
  if (lhs.isSignlessInteger(8) && rhs.isSignlessInteger(4) &&
      out.isSignlessInteger(32)) {
    if (hasFeature(config, "+i8mm")) {
      return {
          TileMxNxK{4, 8, 16},
          TileMxNxK{2, 8, 16},
          TileMxNxK{1, 8, 16},
      };
    }
    if (hasFeature(config, "+dotprod")) {
      return {
          TileMxNxK{8, 8, 8},
          TileMxNxK{4, 8, 8},
          TileMxNxK{2, 8, 8},
          TileMxNxK{1, 8, 8},
      };
    }
    return {
        TileMxNxK{4, 16, 2},
        TileMxNxK{2, 16, 2},
        TileMxNxK{1, 16, 2},
    };
  }
  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

// Enumerate tile sizes to choose from on x86-64.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK> enumerateMatmulTileX86_64(TypeRange elementTypes,
                                                        DictionaryAttr config) {
  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];

  if (out.isF32() || out.isF16() || out.isBF16()) {
    if (lhs.isBF16() && rhs.isBF16() && (out.isBF16() || out.isF32())) {
      if (hasFeature(config, "+avx512bf16")) {
        return {
            TileMxNxK{16, 16, 2}, // Aim to use VDPBF16PS (zmm).
            TileMxNxK{8, 16, 2},  // Truncation of the above.
            TileMxNxK{4, 16, 2},  // Truncation of the above.
            TileMxNxK{2, 16, 2},  // Truncation of the above.
            TileMxNxK{1, 16, 2},  // Truncation of the above.
        };
      }
    }
    if (isa<FloatType>(lhs) && isa<FloatType>(rhs)) {
      // Note: 16-bit floating point types currently use the same tile size as
      // f32. This makes sense when either (1) the accumulator is f32, or (2)
      // the arithmetic will have to expand f16 to f32 in registers. We may
      // reconsider when taking advantage of native f16/bf16 arithmetic when the
      // accumulator itself is f16/bf16.
      if (hasFeature(config, "+avx512f")) {
        return {
            TileMxNxK{16, 16, 1}, // Aim to use VFMADD* (zmm).
            TileMxNxK{8, 16, 1},  // Truncation of the above.
            TileMxNxK{4, 16, 1},  // Truncation of the above.
            TileMxNxK{2, 16, 1},  // Truncation of the above.
            TileMxNxK{1, 16, 1},  // Truncation of the above.
        };
      }
      if (hasFeature(config, "+avx")) {
        // Note: for good performance, most +avx users will also want to add
        // +fma, but that's a local instruction selection detail and the tile
        // layout is unaffected, as there are enough registers even with the
        // need for intermediate product registers when +fma is not used.
        return {
            TileMxNxK{8, 8, 1}, // Aim to use VFMADD* (ymm).
            TileMxNxK{4, 8, 1}, // Truncation of the above.
            TileMxNxK{2, 8, 1}, // Truncation of the above.
            TileMxNxK{1, 8, 1}, // Truncation of the above.
        };
      }
      // SSE fallback.
      return {
          TileMxNxK{8, 4, 1}, // Aim to use MULPS/ADDPS (xmm).
          TileMxNxK{4, 4, 1}, // Truncation of the above.
          TileMxNxK{2, 4, 1}, // Truncation of the above.
          TileMxNxK{1, 4, 1}, // Truncation of the above.
      };
    }
  }

  if (out.isSignlessInteger(32) &&
      ((lhs.isSignlessInteger(8) && rhs.isSignlessInteger(8)) ||
       (lhs.isSignlessInteger(16) && rhs.isSignlessInteger(16)))) {
    if (hasFeature(config, "+avx512vnni")) {
      // This is the same tile size as with VPMADDWD as the only difference
      // is that VPDPWSSD accumulates. VPDPBUSD would call for {16, 16, 4} but
      // we can't easily use it because of its unsigned*signed semantics.
      return {
          TileMxNxK{16, 16, 2}, // Aim to use VPDPWSSD (zmm).
          TileMxNxK{8, 16, 2},  // Truncation of the above.
          TileMxNxK{4, 16, 2},  // Truncation of the above.
          TileMxNxK{2, 16, 2},  // Truncation of the above.
          TileMxNxK{1, 16, 2},  // Truncation of the above.
      };
    }
    if (hasFeature(config, "+avx512bw")) {
      return {
          TileMxNxK{16, 16, 2}, // Aim to use VPMADDWD (zmm).
          TileMxNxK{8, 16, 2},  // Truncation of the above.
          TileMxNxK{4, 16, 2},  // Truncation of the above.
          TileMxNxK{2, 16, 2},  // Truncation of the above.
          TileMxNxK{1, 16, 2},  // Truncation of the above.
      };
    }
    if (hasFeature(config, "+avx2")) {
      return {
          TileMxNxK{8, 8, 2}, // Aim to use VPMADDWD (ymm).
          TileMxNxK{4, 8, 2}, // Truncation of the above.
          TileMxNxK{2, 8, 2}, // Truncation of the above.
          TileMxNxK{1, 8, 2}, // Truncation of the above.
      };
    }
    // SSE fallback.
    return {
        TileMxNxK{8, 4, 2}, // Aim to use PMADDWD (xmm).
        TileMxNxK{4, 4, 2}, // Truncation of the above.
        TileMxNxK{2, 4, 2}, // Truncation of the above.
        TileMxNxK{1, 4, 2}, // Truncation of the above.
    };
  }

  if (out.isSignlessInteger(32) && lhs.isSignlessInteger(16) &&
      rhs.isUnsignedInteger(4)) {
    // Experimental s16u4s32 case. Focusing only on the vecmat case for now.
    if (hasFeature(config, "+avx512vnni")) {
      return {
          TileMxNxK{1, 32, 8}, // Aim to use VPDPBUSD (zmm).
      };
    }
  }

  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

static SmallVector<TileMxNxK>
enumerateCPUMatmulTiles(IREE::Encoding::EncodingAttr encoding,
                        DictionaryAttr config) {
  // Enumerate available tile shapes for the given encoding and config.
  SmallVector<Type> elementTypes = encoding.getElementTypesArray();
  if (isAArch64(config)) {
    return enumerateMatmulTileArm64(elementTypes, config);
  }
  if (isX86_64(config)) {
    return enumerateMatmulTileX86_64(elementTypes, config);
  }
  if (isRISCV32(config)) {
    return enumerateMatmulTileRiscv32(config);
  }
  if (isRISCV64(config)) {
    return enumerateMatmulTileRiscv64(elementTypes, config);
  }
  return {};
}

static SmallVector<TileNxHxWxC>
enumerateExsleratev2ConvTiles(IREE::Encoding::EncodingAttr encoding,
                              DictionaryAttr config) {
  // Fallback - no architecture-optimized tile size for this case.
  return {TileNxHxWxC{32, 1, 1, 32}};
}

struct CPUEncodingPackedLayoutMaterializerAttr
    : PackedLayoutMaterializerAttrExternalModelBase<
          CPUEncodingPackedLayoutMaterializerAttr, CPUEncodingResolverAttr> {

  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<CPUEncodingResolverAttr>(attr).getConfiguration();
  }

  MaterializeEncodingInfo getEncodingInfoImpl(Attribute attr,
                                              RankedTensorType type) const {
    auto layoutAttr = cast<CPUEncodingResolverAttr>(attr);

    auto encoding =
        dyn_cast_if_present<IREE::Encoding::EncodingAttr>(type.getEncoding());

    MaterializeEncodingInfo info;
    if (!encoding) {
      return info;
    }

    // We only know about contractions with {Batch, M, N, K} <= 1 at the moment.
    auto cDims = getEncodingContractionDims(encoding);
    if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
        cDims->n.size() > 1 || cDims->k.size() > 1) {
      return info;
    }

    SmallVector<TileMxNxK> enumeratedTileMxNxK =
        enumerateCPUMatmulTiles(encoding, layoutAttr.getConfiguration());
    if (enumeratedTileMxNxK.empty()) {
      return info;
    }
    auto narrowDim = IREE::Encoding::getPo2MatmulNarrowDim(encoding);
    // Choose a final matmul TileMxNxK from the above-enumerated tile shapes,
    // taking narrow dimensions into account.
    TileMxNxK chosenTileMxNxK =
        chooseMatmulTile(enumeratedTileMxNxK, narrowDim);
    FailureOr<MaterializeEncodingInfo> maybeEncodingInfo =
        getEncodingInfoForMatmul(encoding, chosenTileMxNxK);
    if (failed(maybeEncodingInfo)) {
      return info;
    }
    info = std::move(maybeEncodingInfo.value());
    FailureOr<IREE::Codegen::ScalableTileFlags> scalableFlags =
        getScalableTileFlags(*cDims, encoding, layoutAttr.getConfiguration());
    if (succeeded(scalableFlags)) {
      info.scalableTiles = std::move(scalableFlags);
    }
    if (IREE::Encoding::isNarrowNResult(encoding) &&
        llvm::none_of(info.scalableTiles.value_or(Codegen::ScalableTileFlags{}),
                      [](bool flag) { return flag; })) {
      transposeInPlace(info);
    }
    return info;
  }
};

struct CPUEncodingResolverMaterializerAttr final
    : EncodingLayoutMaterializerAttrExternalModelBase<
          CPUEncodingResolverMaterializerAttr, CPUEncodingResolverAttr> {

  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto layoutAttr = cast<CPUEncodingResolverAttr>(attr);
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }
    if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
      return lowerFillOpWithResolvedLayouts(b, fillOp, convertedResTypes,
                                            convertedOperands);
    }
    // Scaled contraction (MX matmul) is not yet supported on CPU, so we drop
    // the encoding and clone the op as-is.
    if (IREE::LinalgExt::isaScaledContractionOpInterface(linalgOp)) {
      int64_t numInputs = linalgOp.getNumDpsInputs();
      return dropEncodingAndCloneOp(b, linalgOp,
                                    convertedOperands.take_front(numInputs),
                                    convertedOperands.drop_front(numInputs));
    }
    if (linalg::isaContractionOpInterface(linalgOp)) {
      return lowerContractionOpWithEncoding(
          b, linalgOp, convertedOperands,
          cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      return lowerGenericOpWithResolvedLayouts(
          b, genericOp, convertedResTypes, convertedOperands,
          cast<IREE::Encoding::LayoutMaterializerAttr>(attr));
    }
    return nullptr;
  }
};

struct CPULayoutResolverAttr final
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<
          CPULayoutResolverAttr, CPUEncodingResolverAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    MLIRContext *ctx = attr.getContext();
    SmallVector<NamedAttribute> configItems;
    if (std::optional<StringRef> cpuFeatures = getConfigCpuFeatures(config)) {
      addConfigCpuFeatures(ctx, cpuFeatures.value(), configItems);
    }
    if (std::optional<StringRef> targetTriple = getConfigTargetTriple(config)) {
      addConfigTargetTriple(ctx, targetTriple.value(), configItems);
    }
    storeNamedAttrIfPresent(configItems, config, "ukernels");
    return CPUEncodingResolverAttr::get(ctx,
                                        DictionaryAttr::get(ctx, configItems));
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    MLIRContext *ctx = attr.getContext();
    return CPUEncodingResolverAttr::get(ctx, getPackedLayoutImpl(attr, type));
  }
};

struct CPUSerializableAttr final
    : IREE::Encoding::SerializableAttr::ExternalModel<CPUSerializableAttr,
                                                      CPUEncodingResolverAttr> {
  bool isSerialized(Attribute attr) const {
    auto configuration = cast<CPUEncodingResolverAttr>(attr).getConfiguration();
    return configuration && configuration.contains(kEncodingInfoAttrName);
  }

  Value calculateStorageSizeInBytes(Attribute attr, Location loc,
                                    OpBuilder &builder, RankedTensorType type,
                                    ValueRange dynamicDims) const {
    return calculatePackedStorageSizeInBytesImpl(attr, loc, builder, type,
                                                 dynamicDims);
  }
};

struct CPUEncodingResolverVerifier
    : mlir::VerifiableTensorEncoding::ExternalModel<CPUEncodingResolverVerifier,
                                                    CPUEncodingResolverAttr> {

  LogicalResult
  verifyEncoding(Attribute attr, ArrayRef<int64_t> shape, Type elementType,
                 function_ref<InFlightDiagnostic()> emitError) const {
    auto packedLayoutMaterializerAttr =
        cast<Codegen::PackedLayoutMaterializerAttr>(attr);
    return packedLayoutMaterializerAttr.verifyPackedLayoutWithType(
        shape, elementType, emitError);
  }
};

//===----------------------------------------------------------------------===//
// Interface methods implementation for iree_cpu.vmvx_encoding_resolver.
//===----------------------------------------------------------------------===//

// Enumerate tile sizes to choose from when no specific architecture is
// targeted. For narrow-{M,N} cases, this only enumerates on narrow M. The
// narrow-N cases are handled by transposition in chooseMatmulTile.
 static SmallVector<TileMxNxK>
enumerateVMVXMatmulTiles(linalg::ContractionDimensions cDims,
                         IREE::Encoding::EncodingAttr encoding,
                         DictionaryAttr config) {
  bool hasUkernelSupport = hasUkernel(config);

  // TODO(hanchung): The ukernel path does not support 3d
  // codegen.query_tile_sizes op, so we disable dynamic tile shapes for
  // batch_matmul. Also, they are not set up for narrow M/N matmul, so it is
  // disabled when it is the case.
  if (!cDims.batch.empty() || getPo2MatmulNarrowDim(encoding)) {
    hasUkernelSupport = false;
  }
  if (hasUkernelSupport) {
    // VMVX+ukernel uses dynamic tile shapes.
    return {TileMxNxK{ShapedType::kDynamic, ShapedType::kDynamic,
                      ShapedType::kDynamic}};
  }

  return {
      TileMxNxK{8, 8, 4}, // Some vaguely reasonable tile shape.
      TileMxNxK{4, 8, 4}, // Truncation of the above.
      TileMxNxK{2, 8, 4}, // Truncation of the above.
      TileMxNxK{1, 8, 4}, // Truncation of the above.
  };
}

struct VMVXEncodingPackedLayoutMaterializerAttr final
    : PackedLayoutMaterializerAttrExternalModelBase<
          VMVXEncodingPackedLayoutMaterializerAttr, VMVXEncodingResolverAttr> {

  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<VMVXEncodingResolverAttr>(attr).getConfiguration();
  }

  MaterializeEncodingInfo getEncodingInfoImpl(Attribute attr,
                                              RankedTensorType type) const {
    auto layoutAttr = cast<VMVXEncodingResolverAttr>(attr);

    auto encoding =
        dyn_cast_if_present<IREE::Encoding::EncodingAttr>(type.getEncoding());

    MaterializeEncodingInfo info;
    if (!encoding) {
      return info;
    }

    // We only know about contractions with {Batch, M, N, K} <= 1 at the moment.
    auto cDims = getEncodingContractionDims(encoding);
    if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
        cDims->n.size() > 1 || cDims->k.size() > 1) {
      return info;
    }

    SmallVector<TileMxNxK> enumeratedTileMxNxK = enumerateVMVXMatmulTiles(
        cDims.value(), encoding, layoutAttr.getConfiguration());
    if (enumeratedTileMxNxK.empty()) {
      return info;
    }
    auto narrowDim = IREE::Encoding::getPo2MatmulNarrowDim(encoding);
    // Choose a final matmul TileMxNxK from the above-enumerated tile shapes,
    // taking narrow dimensions into account.
    TileMxNxK chosenTileMxNxK =
        chooseMatmulTile(enumeratedTileMxNxK, narrowDim);
    FailureOr<MaterializeEncodingInfo> maybeEncodingInfo =
        getEncodingInfoForMatmul(encoding, chosenTileMxNxK);
    if (failed(maybeEncodingInfo)) {
      return info;
    }
    info = std::move(maybeEncodingInfo.value());
    if (IREE::Encoding::isNarrowNResult(encoding)) {
      transposeInPlace(info);
    }
    return info;
  }
};

struct VMVXEncodingResolverMaterializerAttr final
    : EncodingLayoutMaterializerAttrExternalModelBase<
          VMVXEncodingResolverMaterializerAttr, VMVXEncodingResolverAttr> {

  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto layoutAttr = cast<VMVXEncodingResolverAttr>(attr);
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }
    if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
      return lowerFillOpWithResolvedLayouts(b, fillOp, convertedResTypes,
                                            convertedOperands);
    }
    if (linalg::isaContractionOpInterface(linalgOp)) {
      return lowerContractionOpWithEncoding(
          b, linalgOp, convertedOperands,
          cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      return lowerGenericOpWithResolvedLayouts(
          b, genericOp, convertedResTypes, convertedOperands,
          cast<IREE::Encoding::LayoutMaterializerAttr>(attr));
    }
    return nullptr;
  }
};

struct VMVXLayoutResolverAttr final
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<
          VMVXLayoutResolverAttr, VMVXEncodingResolverAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    MLIRContext *ctx = attr.getContext();
    SmallVector<NamedAttribute> configItems;
    storeNamedAttrIfPresent(configItems, config, "ukernels");
    return VMVXEncodingResolverAttr::get(ctx,
                                         DictionaryAttr::get(ctx, configItems));
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    MLIRContext *ctx = attr.getContext();
    return VMVXEncodingResolverAttr::get(
        ctx, getPackedLayoutImpl(attr, type, /*addEncodingAttr=*/true));
  }
};

struct VMVXSerializableAttr final
    : IREE::Encoding::SerializableAttr::ExternalModel<
          VMVXSerializableAttr, VMVXEncodingResolverAttr> {
  bool isSerialized(Attribute attr) const {
    auto configuration =
        cast<VMVXEncodingResolverAttr>(attr).getConfiguration();
    return configuration && configuration.contains(kEncodingInfoAttrName);
  }

  Value calculateStorageSizeInBytes(Attribute attr, Location loc,
                                    OpBuilder &builder, RankedTensorType type,
                                    ValueRange dynamicDims) const {
    return calculatePackedStorageSizeInBytesImpl(attr, loc, builder, type,
                                                 dynamicDims);
  }
};

struct VMVXEncodingResolverVerifier
    : mlir::VerifiableTensorEncoding::ExternalModel<
          VMVXEncodingResolverVerifier, VMVXEncodingResolverAttr> {
  LogicalResult
  verifyEncoding(Attribute attr, ArrayRef<int64_t> shape, Type elementType,
                 function_ref<InFlightDiagnostic()> emitError) const {
    auto packedLayoutMaterializerAttr =
        cast<Codegen::PackedLayoutMaterializerAttr>(attr);
    return packedLayoutMaterializerAttr.verifyPackedLayoutWithType(
        shape, elementType, emitError);
  }
};

// Interface for Exsleratev2
struct Exsleratev2EncodingPackedLayoutMaterializerAttr
    : public PackedLayoutMaterializerAttrExternalModelBase<
          Exsleratev2EncodingPackedLayoutMaterializerAttr,
          Exsleratev2EncodingResolverAttr> {
  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<Exsleratev2EncodingResolverAttr>(attr).getConfiguration();
  }

  // Return tiling info for conv input and filter operands.
  // Output (CONV_RESULT) is left flat; lowerOp handles its pack/unpack
  // manually so that downstream requant ops see plain types.
  //   CONV_LHS  [IH, IW, C]    -> [IH/tH, IW/tW, C/32,  tH, tW, 32]
  //   CONV_RHS  [KH, KW, C, F] -> [F/32,  KH,    KW,    C/32, 32, 32]
  //   CONV_RESULT               -> identity (no pack)
  MaterializeEncodingInfo getEncodingInfoImpl(Attribute attr,
                                              RankedTensorType type) const {
    auto encoding = llvm::dyn_cast_or_null<IREE::Encoding::EncodingAttr>(
        type.getEncoding());
    MaterializeEncodingInfo info;
    if (!encoding) {
      return info;
    }

    // Only apply EXSLERATEV2 tiling to convolution encodings.
    if (failed(IREE::Encoding::getEncodingConvDims(encoding))) {
      return info;
    }

    // Detect input format from the LHS user_indexing_map.
    // Only handle 6D conv loops (OH,OW,F,KH,KW,C). Pool ops have 5D loops
    // and would false-match the NCHW heuristic — guard by num dims.
    SmallVector<AffineMap> rootMaps = encoding.getRootMaps();
    if (rootMaps.empty()) {
      return info;
    }
    AffineMap lhsMap = rootMaps[0];
    if (lhsMap.getNumDims() != 6) {
      return info;
    }
    bool isNHWC =
        lhsMap.getNumResults() > 0 &&
        dyn_cast<AffineDimExpr>(lhsMap.getResult(lhsMap.getNumResults() - 1));
    bool isNCHW = !isNHWC && lhsMap.getNumResults() > 0 &&
                  dyn_cast<AffineDimExpr>(lhsMap.getResult(0));
    if (!isNHWC && !isNCHW) {
      return info;
    }

    unsigned opIdx = encoding.getOperandIndex().getValue().getZExtValue();
    if (opIdx == IREE::Encoding::CONV_LHS) {
      if (isNHWC) {
        // NHWC [H, W, C] → [H/tH, W/tW, C/32, tH, tW, 32]
        info.innerDimsPos = {0, 1, 2};
        info.innerTileSizes = {kExslTileH, kExslTileW, kExslChSet};
      } else {
        // NCHW [C, H, W] → [H/tH, W/tW, C/32, tH, tW, 32] (spatial-tile-major,
        // matches runtime)
        info.innerDimsPos = {1, 2, 0};
        info.innerTileSizes = {kExslTileH, kExslTileW, kExslChSet};
        info.outerDimsPerm = {1, 2, 0};
      }
    } else if (opIdx == IREE::Encoding::CONV_RHS) {
      // Detect actual filter layout from rootMaps[1] (filter indexing map).
      // For 6D conv (d0=OH, d1=OW, d2=F, d3=KH, d4=KW, d5=C):
      //   HWCF [KH,KW,C,F]: first result = d3 (reduction, position >= 3)
      //   FHWC [F,KH,KW,C]: first result = d2 (parallel, position < 3)
      bool isFHWC = false;
      if (rootMaps.size() > 1 && rootMaps[1].getNumResults() > 0) {
        if (auto dimExpr =
                dyn_cast<AffineDimExpr>(rootMaps[1].getResult(0)))
          isFHWC = (dimExpr.getPosition() < 3); // F is parallel, pos 0–2
      }
      if (isNHWC && isFHWC) {
        // FHWC [F, KH, KW, C] → [F/32, KH, KW, C/32, 32, 32]
        // Tile dim 0 (F) and dim 3 (C); outer order already correct — no perm.
        info.innerDimsPos = {0, 3};
        info.innerTileSizes = {kExslFiltSet, kExslChSet};
      } else if (isNHWC) {
        // HWCF [KH, KW, C, F] → [F/32, KH, KW, C/32, 32, 32]
        info.outerDimsPerm = {3, 0, 1, 2};
        info.innerDimsPos = {3, 2};
        info.innerTileSizes = {kExslFiltSet, kExslChSet};
      } else {
        // NCHW filter [F, C, KH, KW] → [F/32, KH, KW, C/32, 32, 32]
        info.outerDimsPerm = {0, 2, 3, 1};
        info.innerDimsPos = {0, 1};
        info.innerTileSizes = {kExslFiltSet, kExslChSet};
      }
    } else if (opIdx == IREE::Encoding::CONV_RESULT) {
      if (type.getRank() == 1) {
        // 1D bias [F] → [F/32, 32].
        // outerDimsPerm must be {0} not empty — empty causes applyPermutation
        // to drop the outer F/32 dim in lowerGenericOpWithResolvedLayouts,
        // producing a 1-result map for a 2D tensor (SIGSEGV).
        info.innerDimsPos = {0};
        info.innerTileSizes = {kExslFiltSet};
        info.outerDimsPerm = {0};
      } else if (type.getRank() == 3) {
        // 3D conv output [F, OH, OW] (NCHW) or [OH, OW, F] (NHWC) → packed
        // to [OH/tH, OW/tW, F/32, tH, tW, 32], matching the tiled_conv2d
        // output format.
        if (isNHWC) {
          info.innerDimsPos = {0, 1, 2};
          info.innerTileSizes = {kExslTileH, kExslTileW, kExslFiltSet};
          // outerDimsPerm must be explicit identity — empty perm causes
          // outInverseOuterDimsPerm[] OOB access in lowerGenericOpWithResolvedLayouts.
          info.outerDimsPerm = {0, 1, 2};
        } else {
          info.innerDimsPos = {1, 2, 0};
          info.innerTileSizes = {kExslTileH, kExslTileW, kExslFiltSet};
          info.outerDimsPerm = {1, 2, 0};
        }
      }
    }
    return info;
  }
};

struct Exsleratev2EncodingResolverMaterializerAttr final
    : EncodingLayoutMaterializerAttrExternalModelBase<
          Exsleratev2EncodingResolverMaterializerAttr,
          Exsleratev2EncodingResolverAttr> {

  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto layoutAttr = cast<Exsleratev2EncodingResolverAttr>(attr);
    auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }

    if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
      return lowerFillOpWithResolvedLayouts(b, fillOp, convertedResTypes,
                                            convertedOperands);
    }

    if (linalg::isaContractionOpInterface(linalgOp)) {
      return lowerContractionOpWithEncoding(
          b, linalgOp, convertedOperands,
          cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
    }

    FailureOr<Operation *> newOp = lowerExsleratev2ConvolutionOpWithEncoding(
        b, linalgOp, convertedOperands,
        cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
    if (succeeded(newOp)) {
      return newOp.value();
    }

    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      if (Operation *r = lowerGenericOpWithResolvedLayouts(
              b, genericOp, convertedResTypes, convertedOperands,
              cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr))) {
        return r;
      }
    }

    return nullptr;
  }
};

struct Exsleratev2LayoutResolverAttr final
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<
          Exsleratev2LayoutResolverAttr, Exsleratev2EncodingResolverAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    MLIRContext *ctx = attr.getContext();
    SmallVector<NamedAttribute> configItems;
    return Exsleratev2EncodingResolverAttr::get(
        ctx, DictionaryAttr::get(ctx, configItems));
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    MLIRContext *ctx = attr.getContext();
    return Exsleratev2EncodingResolverAttr::get(
        ctx, getPackedLayoutImpl(attr, type, /*addEncodingAttr=*/true));
  }
};

struct Exsleratev2SerializableAttr final
    : IREE::Encoding::SerializableAttr::ExternalModel<
          Exsleratev2SerializableAttr, Exsleratev2EncodingResolverAttr> {

  bool isSerialized(Attribute attr) const {
    auto configuration =
        cast<Exsleratev2EncodingResolverAttr>(attr).getConfiguration();
    return configuration && configuration.contains(kEncodingInfoAttrName);
  }

  Value calculateStorageSizeInBytes(Attribute attr, Location loc,
                                    OpBuilder &builder, RankedTensorType type,
                                    ValueRange dynamicDims) const {
    return calculatePackedStorageSizeInBytesImpl(attr, loc, builder, type,
                                                 dynamicDims);
  }
};

} // namespace

void registerCPUEncodingExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::CPU::IREECPUDialect *dialect) {
        IREE::CPU::CPUEncodingResolverAttr::attachInterface<
            CPUEncodingPackedLayoutMaterializerAttr,
            CPUEncodingResolverMaterializerAttr, CPULayoutResolverAttr,
            CPUSerializableAttr, CPUEncodingResolverVerifier>(*ctx);
        IREE::CPU::VMVXEncodingResolverAttr::attachInterface<
            VMVXEncodingPackedLayoutMaterializerAttr,
            VMVXEncodingResolverMaterializerAttr, VMVXLayoutResolverAttr,
            VMVXSerializableAttr, VMVXEncodingResolverVerifier>(*ctx);
        IREE::CPU::Exsleratev2EncodingResolverAttr::attachInterface<
            Exsleratev2EncodingPackedLayoutMaterializerAttr,
            Exsleratev2EncodingResolverMaterializerAttr,
            Exsleratev2LayoutResolverAttr, Exsleratev2SerializableAttr>(*ctx);
      });
}

} // namespace mlir::iree_compiler::IREE::CPU
