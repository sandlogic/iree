// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- Exsleratev2EncodingExternalModels.cpp ------------------------------===//
//
// This file implements the encoding-materialization interfaces for the
// EXSLERATEV2 backend. It provides the full tiled-convolution layouts and
// lowering that were previously part of CPUEncodingExternalModels.cpp, kept
// here so the EXSLERATEV2-specific packing/lowering is decoupled from the CPU
// and VMVX resolvers.
//
// Implements for iree_cpu.exsleratev2_encoding_resolver:
//   - IREE::Encoding::LayoutResolverAttr
//   - IREE::Encoding::SerializableAttr
//   - IREE::Encoding::LayoutMaterializerAttr
//   - IREE::Codegen::PackedLayoutMaterializerAttr
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/ExternalInterfaces/Exsleratev2EncodingExternalModels.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/ExternalInterfaces/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"

#define DEBUG_TYPE "iree-codegen-materialize-encoding"

namespace mlir::iree_compiler::IREE::CPU {

using IREE::Codegen::MaterializeEncodingInfo;
using IREE::Codegen::TileMxNxK;

namespace {

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

// A stride of 0 means the affine map carried no explicit stride coefficient;
// treat that as the unit stride.
static int64_t defaultStride(int64_t stride) {
  return stride == 0 ? 1 : stride;
}

// Pads the packed activation with extra spatial *tiles* so the linalg.generic
// verifier sees in-bounds accesses. The body index map reads
// (iy.floorDiv(tH), iw.floorDiv(tW)), and iy/iw can exceed the packed tile
// count by up to one tile on the rightmost/bottom edge when the kernel stencil
// extends past the tile boundary. We add ceil((k-1)/tile) halo tiles on the H
// and W outer dims. The hardware kernel handles partial last tiles at runtime,
// so the halo elements are never actually read; `padValue` only has to be a
// value that cannot corrupt the reduction if it ever were (0 for a mul-add
// accumulator, the max-reduction identity for a max pool).
// Returns `packedIn` unchanged when no halo is needed.
static Value padPackedInputHalo(OpBuilder &builder, Location loc,
                                Value packedIn, int64_t kH, int64_t kW,
                                TypedAttr padValue) {
  int64_t extraH = (kH + kExslTileH - 2) / kExslTileH;
  int64_t extraW = (kW + kExslTileW - 2) / kExslTileW;
  if (extraH == 0 && extraW == 0) {
    return packedIn;
  }
  auto packedInTy = cast<RankedTensorType>(packedIn.getType());
  // Packed layout is [IH/tH, IW/tW, C/32, tH, tW, 32]; only the outer H/W tile
  // counts (dims 0, 1) grow.
  SmallVector<int64_t> paddedShape(packedInTy.getShape());
  paddedShape[0] += extraH;
  paddedShape[1] += extraW;
  auto paddedTy =
      RankedTensorType::get(paddedShape, packedInTy.getElementType());
  Value pad = arith::ConstantOp::create(builder, loc, padValue);
  SmallVector<OpFoldResult> low(6, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> high = {
      builder.getIndexAttr(extraH), builder.getIndexAttr(extraW),
      builder.getIndexAttr(0),      builder.getIndexAttr(0),
      builder.getIndexAttr(0),      builder.getIndexAttr(0)};
  return tensor::PadOp::create(builder, loc, paddedTy, packedIn, low, high,
                               pad);
}

// Emits the in-body read of the packed activation shared by the conv and pool
// lowerings. Reconstructs the raw (untiled) spatial coordinates from the loop
// indices,
//   iy = tyBase*(tH*sH) + innerY*sH + ky ,
//   iw = txBase*(tW*sW) + innerX*sW + kx ,
// splits each into its (outer tile, inner offset) pair, and extracts from the
// packed [IH/tH, IW/tW, C/32, tH, tW, 32] layout at
//   [iy/tH, iw/tW, chan, iy%tH, iw%tW, innerC].
// The activation is read via tensor.extract (off the operand list) so it can be
// left unpadded without tripping the linalg.generic operand-extent verifier
// (cf. iree-org/iree#23746).
static Value emitPackedInputExtract(OpBuilder &nb, Location loc, Value packedIn,
                                    Value tyBase, Value txBase, Value innerY,
                                    Value innerX, Value ky, Value kx,
                                    Value chan, Value innerC, int64_t strideH,
                                    int64_t strideW) {
  auto cstIdx = [&](int64_t v) -> Value {
    return arith::ConstantIndexOp::create(nb, loc, v);
  };
  Value iy = arith::AddIOp::create(
      nb, loc,
      arith::AddIOp::create(
          nb, loc,
          arith::MulIOp::create(nb, loc, tyBase, cstIdx(kExslTileH * strideH)),
          arith::MulIOp::create(nb, loc, innerY, cstIdx(strideH))),
      ky);
  Value iw = arith::AddIOp::create(
      nb, loc,
      arith::AddIOp::create(
          nb, loc,
          arith::MulIOp::create(nb, loc, txBase, cstIdx(kExslTileW * strideW)),
          arith::MulIOp::create(nb, loc, innerX, cstIdx(strideW))),
      kx);
  Value iyDiv = arith::DivUIOp::create(nb, loc, iy, cstIdx(kExslTileH));
  Value iyRem = arith::RemUIOp::create(nb, loc, iy, cstIdx(kExslTileH));
  Value iwDiv = arith::DivUIOp::create(nb, loc, iw, cstIdx(kExslTileW));
  Value iwRem = arith::RemUIOp::create(nb, loc, iw, cstIdx(kExslTileW));
  return tensor::ExtractOp::create(
      nb, loc, packedIn,
      ValueRange{iyDiv, iwDiv, chan, iyRem, iwRem, innerC});
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
  int64_t rawStrideH = isNHWC ? exslExtractStride(lhsMap.getResult(0), 0)
                              : exslExtractStride(lhsMap.getResult(1), 0);
  int64_t rawStrideW = isNHWC ? exslExtractStride(lhsMap.getResult(1), 1)
                              : exslExtractStride(lhsMap.getResult(2), 1);
  int64_t strideH = defaultStride(rawStrideH);
  int64_t strideW = defaultStride(rawStrideW);

  // Build 10-D affine maps for the filter and output operands. The activation
  // is read OFF the operand list via tensor.extract inside the body (so it is
  // not a bound-checked linalg operand, cf. iree-org/iree#23746); the in-body
  // raw input indices are
  //   iy = d0*(tH*sH) + d6*sH + d3 ,  iw = d1*(tW*sW) + d7*sW + d4 .
  // Loop dims: d0=ty_cube, d1=tx_cube, d2=fs, d3=ky, d4=kx, d5=cs, d6=ly,
  // d7=lx, d8=filt, d9=c.
  auto d = [&](unsigned i) { return builder.getAffineDimExpr(i); };

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
  //   operands[2..N-1]:         extra inputs (e.g. zero points) — 0D, passed
  //   through operands[inputs.size()]:  output [OH/tH, OW/tW, F/32, tH, tW, 32]
  Value packedIn = operands[0];
  Value packedFilter = operands[1];
  size_t numExtraInputs = inputs.size() - 2;
  Value packedOut = operands[inputs.size()];
  Location loc = linalgOp.getLoc();

  // Add the spatial halo tiles the stencil may reach. Kernel extent comes from
  // the packed filter layout [F/32, KH, KW, IC/32, 32, 32]. Pad value is 0: the
  // halo never contributes to the mul-add accumulator at runtime.
  {
    auto packedFilterTy = cast<RankedTensorType>(packedFilter.getType());
    int64_t kH = packedFilterTy.getShape()[1];
    int64_t kW = packedFilterTy.getShape()[2];
    Type elemTy = cast<RankedTensorType>(packedIn.getType()).getElementType();
    packedIn = padPackedInputHalo(builder, loc, packedIn, kH, kW,
                                  builder.getZeroAttr(elemTy));
  }

  auto packedOutType = cast<RankedTensorType>(packedOut.getType());
  Type i32Ty = builder.getI32Type();

  // Build input list and affine maps: filter first, then extra 0D inputs.
  // The activation (packedIn) is NOT a mapped operand; it is captured and read
  // via tensor.extract in the body (off-operand / implicit-pad form). This lets
  // the activation be unpadded without tripping the linalg.generic verifier's
  // operand-extent inference. Only filter + output bound the 10 loop dims.
  SmallVector<Value> genericInputs = {packedFilter};
  SmallVector<AffineMap> allMaps = {filterMap};
  AffineMap zeroMap = AffineMap::get(10, 0, {}, builder.getContext());
  for (size_t i = 2; i < inputs.size(); i++) {
    genericInputs.push_back(operands[i]);
    allMaps.push_back(zeroMap);
  }
  allMaps.push_back(outputMap);

  // Body args: [filter, extra..., output_accumulator].
  size_t accumArgIdx = 1 + numExtraInputs;
  Value capturedIn = packedIn;

  // 10-D tiled generic matching the tiled_conv2d kernel loop structure.
  // Extra zero-point inputs (if any) are carried through with 0D maps and
  // ignored in the body — hardware lowering via emitCSRFromTiledConv handles
  // quantization separately from the body computation.
  Value result10D =
      linalg::GenericOp::create(
          builder, loc, packedOutType, genericInputs, ValueRange{packedOut},
          ArrayRef<AffineMap>(allMaps), iterTypes,
          [&](OpBuilder &nb, Location nb_loc, ValueRange args) {
            auto idx = [&](unsigned dim) -> Value {
              return linalg::IndexOp::create(nb, nb_loc, dim);
            };
            // Loop dims: d0=ty_cube d1=tx_cube d2=fs d3=ky d4=kx d5=cs d6=ly
            //            d7=lx d8=filt d9=c.
            Value inVal = emitPackedInputExtract(
                nb, nb_loc, capturedIn, /*tyBase=*/idx(0), /*txBase=*/idx(1),
                /*innerY=*/idx(6), /*innerX=*/idx(7), /*ky=*/idx(3),
                /*kx=*/idx(4), /*chan=*/idx(5), /*innerC=*/idx(9), strideH,
                strideW);
            Value lhs = arith::ExtSIOp::create(nb, nb_loc, i32Ty, inVal);
            Value rhs = arith::ExtSIOp::create(nb, nb_loc, i32Ty, args[0]);
            Value mul = arith::MulIOp::create(nb, nb_loc, lhs, rhs);
            Value add =
                arith::AddIOp::create(nb, nb_loc, mul, args[accumArgIdx]);
            linalg::YieldOp::create(nb, nb_loc, add);
          })
          .getResult(0);
  Operation *result10DOp = result10D.getDefiningOp();
  // Stamp the conv strides so the NPU codegen can recover them: with the
  // activation off the operand list there is no input indexing map for
  // inferConvolutionDims to read.
  result10DOp->setAttr("exsleratev2.conv_stride",
                       builder.getDenseI64ArrayAttr({strideH, strideW}));
  // Propagate the dropped spatial pad and the true unpadded input dims (set by
  // SetEncoding when the activation was encoded unpadded) so emitCSR can
  // program the hardware halo and the correct input size.
  if (auto padAttr = linalgOp->getAttr("exsleratev2.conv_pad")) {
    result10DOp->setAttr("exsleratev2.conv_pad", padAttr);
  }
  if (auto inHwAttr = linalgOp->getAttr("exsleratev2.conv_in_hw")) {
    result10DOp->setAttr("exsleratev2.conv_in_hw", inHwAttr);
  }
  // True unpadded channel count (the packed layout rounds C up to the
  // 32-element inner tile, so a sub-32 channel count like C=3 must be carried
  // explicitly).
  if (auto inCAttr = linalgOp->getAttr("exsleratev2.conv_in_c")) {
    result10DOp->setAttr("exsleratev2.conv_in_c", inCAttr);
  }
  return result10DOp;
}

// Creates the tiled linalg.generic for EXSLERATEV2 matmul. MatMul runs on the
// NPU conv datapath with KH=KW=1 and 1x1 spatial tiles, so the packed
// operands (produced by getEncodingInfoForMatmul with {M=1, N=32, K=32}) are
// all rank-4:
//   LHS    [M, K/32, 1, 32]    (M -> spatial H, W=1)
//   RHS    [N/32, K/32, 32, 32] (N -> filter set, K -> channel set)
//   RESULT [M, N/32, 1, 32]    (M -> spatial H, W=1)
// The 5-D loop nest matches the collapsed 10-D conv nest (KH=KW=1, tiles 1x1):
//   d0=ty  d1=fs  d2=cs  d3=filt  d4=c
static FailureOr<Operation *> lowerExsleratev2MatmulOpWithEncoding(
    OpBuilder &builder, linalg::LinalgOp linalgOp, ValueRange operands,
    IREE::Encoding::LayoutMaterializerAttr layoutAttr) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return failure();
  }

  auto inputs = linalgOp.getDpsInputOperands();
  auto outputs = linalgOp.getDpsInits();
  if (inputs.size() != 2 || outputs.size() != 1) {
    return failure();
  }

  // Require the RHS to carry a matmul encoding to identify matmul ops.
  auto rhsType = cast<RankedTensorType>(inputs[1]->get().getType());
  auto rhsEnc = IREE::Encoding::getEncodingAttr(rhsType);
  if (!rhsEnc ||
      rhsEnc.getOperandIndex().getValue() != IREE::Encoding::MATMUL_RHS) {
    return failure();
  }

  // Packed operands pre-packed by MaterializeDeviceEncoding:
  //   operands[0]:              LHS    [M, K/32, 1, 32]
  //   operands[1]:              RHS    [N/32, K/32, 32, 32]
  //   operands[inputs.size()]:  RESULT [M, N/32, 1, 32]
  Value packedIn = operands[0];
  Value packedFilter = operands[1];
  Value packedOut = operands[inputs.size()];
  auto packedInTy = cast<RankedTensorType>(packedIn.getType());
  auto packedFilterTy = cast<RankedTensorType>(packedFilter.getType());
  auto packedOutTy = cast<RankedTensorType>(packedOut.getType());
  if (packedInTy.getRank() != 4 || packedFilterTy.getRank() != 4 ||
      packedOutTy.getRank() != 4) {
    return failure();
  }

  Location loc = linalgOp.getLoc();
  auto d = [&](unsigned i) { return builder.getAffineDimExpr(i); };
  AffineExpr zero = builder.getAffineConstantExpr(0);

  // Loop dims: d0=ty(M), d1=fs(N/32), d2=cs(K/32), d3=filt(32), d4=c(32).
  // LHS [M, K/32, 1, 32]:   {d0, d2, 0, d4}
  AffineMap inputMap = AffineMap::get(5, 0, {d(0), d(2), zero, d(4)},
                                      builder.getContext());
  // RHS [N/32, K/32, 32, 32]: {d1, d2, d3, d4}. Note the RHS pack sets
  // outer_dims_perm=[1,0] (N before K), so the packed dim order is
  // [N/32, K/32, 32, 32] = [fs, cs, filt, c], not [cs, fs, ...].
  AffineMap filterMap = AffineMap::get(5, 0, {d(1), d(2), d(3), d(4)},
                                       builder.getContext());
  // RESULT [M, N/32, 1, 32]: {d0, d1, 0, d3}
  AffineMap outputMap = AffineMap::get(5, 0, {d(0), d(1), zero, d(3)},
                                       builder.getContext());

  SmallVector<utils::IteratorType> iterTypes = {
      utils::IteratorType::parallel,  // d0: ty
      utils::IteratorType::parallel,  // d1: fs
      utils::IteratorType::reduction, // d2: cs
      utils::IteratorType::parallel,  // d3: filt
      utils::IteratorType::reduction, // d4: c
  };

  // The accumulator element type may be wider than the multiplied operands
  // (e.g. i8/i32 inputs with an i64 accumulator for MatMulInteger). Extend the
  // product to the init element type before the add.
  Type i32Ty = builder.getI32Type();
  Type accTy = packedOutTy.getElementType();
  auto extToI32 = [&](OpBuilder &nb, Location nb_loc, Value v) -> Value {
    if (v.getType() == i32Ty) return v;
    return arith::ExtSIOp::create(nb, nb_loc, i32Ty, v)->getResult(0);
  };
  Value result =
      linalg::GenericOp::create(
          builder, loc, packedOutTy,
          ValueRange{packedIn, packedFilter}, ValueRange{packedOut},
          ArrayRef<AffineMap>{inputMap, filterMap, outputMap}, iterTypes,
          [&](OpBuilder &nb, Location nb_loc, ValueRange args) {
            Value lhs = extToI32(nb, nb_loc, args[0]);
            Value rhs = extToI32(nb, nb_loc, args[1]);
            Value mul =
                arith::MulIOp::create(nb, nb_loc, lhs, rhs)->getResult(0);
            if (mul.getType() != accTy) {
              mul = arith::ExtSIOp::create(nb, nb_loc, accTy, mul)
                        ->getResult(0);
            }
            Value add =
                arith::AddIOp::create(nb, nb_loc, mul, args[2])->getResult(0);
            linalg::YieldOp::create(nb, nb_loc, add);
          })
          .getResult(0);
  Operation *resultOp = result.getDefiningOp();
  // Record the original logical dims so the CSR lowering can recover M, K, N
  // without tracing the pack/unpack chain (the packed operands only carry the
  // padded set-multiple shapes). A [M,K] x [K,N] matmul:
  //   LHS [M, K] -> exsleratev2.matmul_in_shape = [M, K]
  //   RHS [K, N] -> exsleratev2.matmul_weight_shape = [K, N]
  //   RESULT    -> exsleratev2.matmul_out_shape = [M, N]
  auto lhsType = cast<RankedTensorType>(inputs[0]->get().getType());
  auto rhsOrigType = cast<RankedTensorType>(inputs[1]->get().getType());
  auto outType = cast<RankedTensorType>(outputs[0].getType());
  if (lhsType.getRank() == 2 && rhsOrigType.getRank() == 2 &&
      outType.getRank() == 2 && lhsType.hasStaticShape() &&
      rhsOrigType.hasStaticShape() && outType.hasStaticShape()) {
    resultOp->setAttr(
        "exsleratev2.matmul_in_shape",
        builder.getDenseI64ArrayAttr(
            {lhsType.getDimSize(0), lhsType.getDimSize(1)}));
    resultOp->setAttr(
        "exsleratev2.matmul_weight_shape",
        builder.getDenseI64ArrayAttr(
            {rhsOrigType.getDimSize(0), rhsOrigType.getDimSize(1)}));
    resultOp->setAttr(
        "exsleratev2.matmul_out_shape",
        builder.getDenseI64ArrayAttr(
            {outType.getDimSize(0), outType.getDimSize(1)}));
  }
  return resultOp;
}

// Creates the tiled linalg.generic for an EXSLERATEV2 max-pooling op. Mirrors
// the convolution lowering above but for the pooling signature (no filter/RHS,
// no channel mixing, i8->i8 max reduction). Operands are expected to already be
// in packed form (activation + output packed; the shape-only window operand
// stays unpacked and only bounds the KH/KW reduction loops).
//
// Packed layouts (NHWC):
//   input  [IH, IW, C] -> [IH/tH, IW/tW, C/32, tH, tW, 32]
//   output [OH, OW, C] -> [OH/tH, OW/tW, C/32, tH, tW, 32]
//
// 8-D loop nest: d0=ty_cube d1=tx_cube d2=cs d3=ky d4=kx d5=ly d6=lx d7=c
static FailureOr<Operation *> lowerExsleratev2PoolingOpWithEncoding(
    OpBuilder &builder, linalg::LinalgOp linalgOp, ValueRange operands,
    IREE::Encoding::LayoutMaterializerAttr layoutAttr) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return failure();
  }

  auto inputs = linalgOp.getDpsInputOperands();
  auto outputs = linalgOp.getDpsInits();
  // Pool has exactly two inputs: activation + shape-only window.
  if (inputs.size() != 2 || outputs.size() != 1) {
    return failure();
  }

  // Activation (LHS) must carry CONV_LHS; the window (input 1) must be
  // UNENCODED. This is precisely what distinguishes a pool (window unencoded)
  // from a conv (filter carries CONV_RHS), so conv never reaches here.
  auto lhsType = cast<RankedTensorType>(inputs[0]->get().getType());
  auto lhsEnc = IREE::Encoding::getEncodingAttr(lhsType);
  if (!lhsEnc ||
      lhsEnc.getOperandIndex().getValue() != IREE::Encoding::CONV_LHS) {
    return failure();
  }
  auto winType = cast<RankedTensorType>(inputs[1]->get().getType());
  if (IREE::Encoding::getEncodingAttr(winType)) {
    return failure();
  }

  // Pool has a 5-D loop nest (OH, OW, C, KH, KW).
  if (static_cast<int64_t>(linalgOp.getStaticLoopRanges().size()) != 5) {
    return failure();
  }

  // Only NHWC pooling: activation map's last result is the pure channel dim.
  SmallVector<AffineMap> origMaps = linalgOp.getIndexingMapsArray();
  AffineMap lhsMap = origMaps[0];
  if (lhsMap.getNumResults() == 0 ||
      !dyn_cast<AffineDimExpr>(lhsMap.getResult(lhsMap.getNumResults() - 1))) {
    return failure();
  }

  // Strides live at result[0]/result[1] of the activation map (NHWC).
  int64_t strideH = defaultStride(exslExtractStride(lhsMap.getResult(0), 0));
  int64_t strideW = defaultStride(exslExtractStride(lhsMap.getResult(1), 1));

  // Element type must be integer (max via arith.maxsi, quantized VGG16) or
  // float (max via arith.maximumf, e.g. YOLOv4's dequantized f32 pools).
  Type elemTy = cast<RankedTensorType>(outputs[0].getType()).getElementType();
  bool isFloatPool = llvm::isa<FloatType>(elemTy);
  if (!elemTy.isInteger() && !isFloatPool) {
    return failure();
  }

  Location loc = linalgOp.getLoc();
  auto d = [&](unsigned i) { return builder.getAffineDimExpr(i); };

  // Window (2x2) bounds the two reduction loops: (d0..d7) -> (d3, d4).
  AffineMap windowMap =
      AffineMap::get(8, 0, {d(3), d(4)}, builder.getContext());
  // Output: [OH/tH, OW/tW, C/32, tH, tW, 32].
  AffineMap outputMap = AffineMap::get(
      8, 0, {d(0), d(1), d(2), d(5), d(6), d(7)}, builder.getContext());

  SmallVector<utils::IteratorType> iterTypes = {
      utils::IteratorType::parallel,  // d0: ty_cube
      utils::IteratorType::parallel,  // d1: tx_cube
      utils::IteratorType::parallel,  // d2: cs
      utils::IteratorType::reduction, // d3: ky
      utils::IteratorType::reduction, // d4: kx
      utils::IteratorType::parallel,  // d5: ly
      utils::IteratorType::parallel,  // d6: lx
      utils::IteratorType::parallel,  // d7: c
  };

  // Packed operands: operands[0]=activation, operands[1]=window (unpacked),
  // operands[2]=output.
  Value packedIn = operands[0];
  Value window = operands[1];
  Value packedOut = operands[inputs.size()];

  // Add the spatial halo tiles the stencil may reach (same scheme as conv).
  // Kernel extent comes from the shape-only window operand. Pad value is the
  // max-reduction identity (−inf for float, signed min for int) so a halo read
  // (never taken at runtime) can never win the max.
  {
    auto winTy = cast<RankedTensorType>(window.getType());
    int64_t kH = winTy.getShape()[0];
    int64_t kW = winTy.getShape()[1];
    Type padElemTy =
        cast<RankedTensorType>(packedIn.getType()).getElementType();
    TypedAttr minAttr;
    if (auto fTy = llvm::dyn_cast<FloatType>(padElemTy)) {
      minAttr = builder.getFloatAttr(
          fTy, APFloat::getInf(fTy.getFloatSemantics(), /*Negative=*/true));
    } else {
      unsigned bw = padElemTy.getIntOrFloatBitWidth();
      minAttr = builder.getIntegerAttr(padElemTy, APInt::getSignedMinValue(bw));
    }
    packedIn = padPackedInputHalo(builder, loc, packedIn, kH, kW, minAttr);
  }

  auto packedOutType = cast<RankedTensorType>(packedOut.getType());
  Value capturedIn = packedIn;

  Value result8D =
      linalg::GenericOp::create(
          builder, loc, packedOutType, ValueRange{window},
          ValueRange{packedOut}, ArrayRef<AffineMap>{windowMap, outputMap},
          iterTypes,
          [&](OpBuilder &nb, Location nb_loc, ValueRange args) {
            auto idx = [&](unsigned dim) -> Value {
              return linalg::IndexOp::create(nb, nb_loc, dim);
            };
            // Loop dims: d0=ty_cube d1=tx_cube d2=cs d3=ky d4=kx d5=ly d6=lx
            //            d7=c.
            Value inVal = emitPackedInputExtract(
                nb, nb_loc, capturedIn, /*tyBase=*/idx(0), /*txBase=*/idx(1),
                /*innerY=*/idx(5), /*innerX=*/idx(6), /*ky=*/idx(3),
                /*kx=*/idx(4), /*chan=*/idx(2), /*innerC=*/idx(7), strideH,
                strideW);
            // args = [window_val (ignored), out_accumulator].
            Value maxVal;
            if (isFloatPool) {
              maxVal = arith::MaximumFOp::create(nb, nb_loc, args[1], inVal);
            } else {
              maxVal = arith::MaxSIOp::create(nb, nb_loc, args[1], inVal);
            }
            linalg::YieldOp::create(nb, nb_loc, maxVal);
          })
          .getResult(0);

  Operation *result8DOp = result8D.getDefiningOp();
  // Stamp pool metadata so the later Exsleratev2PoolLowering can recover the
  // logical (unpacked) dims from the packed op: strides, kernel window, and the
  // true input H/W/C taken from the pre-pack activation type.
  result8DOp->setAttr("exsleratev2.pool_stride",
                      builder.getDenseI64ArrayAttr({strideH, strideW}));
  {
    auto winTy = cast<RankedTensorType>(window.getType());
    result8DOp->setAttr("exsleratev2.pool_kernel",
                        builder.getDenseI64ArrayAttr(
                            {winTy.getShape()[0], winTy.getShape()[1]}));
  }
  if (lhsType.getRank() >= 3) {
    result8DOp->setAttr(
        "exsleratev2.pool_in_hw",
        builder.getDenseI64ArrayAttr(
            {lhsType.getShape()[0], lhsType.getShape()[1]}));
    result8DOp->setAttr(
        "exsleratev2.pool_in_c",
        builder.getDenseI64ArrayAttr({lhsType.getShape()[2]}));
  }
  return result8DOp;
}

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
      // Not a convolution: check whether this is a matmul encoding. MatMul
      // runs on the NPU conv datapath (KH=KW=1), so the data-tiled layout uses
      // 1x1 spatial tiles and 32-element channel/filter sets:
      //   LHS [M, K]   -> [M/1,  K/32, 1, 32]   (M -> spatial H, W=1)
      //   RHS [K, N]   -> [N/32, K/32, 32, 32]  (transposed to filter [N,K])
      //   RESULT [M, N]-> [M/1,  N/32, 1, 32]   (M -> spatial H, W=1)
      if (encoding.getOpType().getValue() ==
          IREE::Encoding::EncodingOpType::matmul) {
        auto matmulInfo = IREE::Codegen::getEncodingInfoForMatmul(
            encoding, TileMxNxK{1, 32, 32});
        if (succeeded(matmulInfo)) {
          return std::move(matmulInfo.value());
        }
      }
      return info;
    }

    // Detect input format from the LHS user_indexing_map.
    // Handle 6D conv loops (OH,OW,F,KH,KW,C) and 5D pool loops (OH,OW,C,KH,KW).
    // NHWC detection (last LHS result is the pure channel dim) holds for both;
    // pool only ever has CONV_LHS / CONV_RESULT operands (no CONV_RHS filter).
    SmallVector<AffineMap> rootMaps = encoding.getRootMaps();
    if (rootMaps.empty()) {
      return info;
    }
    AffineMap lhsMap = rootMaps[0];
    if (lhsMap.getNumDims() != 5 && lhsMap.getNumDims() != 6) {
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
        if (auto dimExpr = dyn_cast<AffineDimExpr>(rootMaps[1].getResult(0))) {
          isFHWC = (dimExpr.getPosition() < 3); // F is parallel, pos 0–2
        }
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
          // outInverseOuterDimsPerm[] OOB access in
          // lowerGenericOpWithResolvedLayouts.
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

    // MatMul runs on the NPU conv datapath (KH=KW=1): lower it to the
    // EXSLERATEV2 tiled 1x1-conv generic, not the generic mmt4d path.
    if (linalg::isaContractionOpInterface(linalgOp)) {
      FailureOr<Operation *> matmulOp = lowerExsleratev2MatmulOpWithEncoding(
          b, linalgOp, convertedOperands,
          cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
      if (succeeded(matmulOp)) {
        return matmulOp.value();
      }
    }

    // Try the convolution resolver first: it only matches ops whose filter
    // carries a CONV_RHS encoding. Contraction/matmul data tiling is not
    // enabled for EXSLERATEV2 yet (see AnnotateDataTilingHints), so no matmul
    // reaches here; such an op falls through to nullptr below.
    FailureOr<Operation *> newOp = lowerExsleratev2ConvolutionOpWithEncoding(
        b, linalgOp, convertedOperands,
        cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
    if (succeeded(newOp)) {
      return newOp.value();
    }

    // Pooling: activation carries CONV_LHS but there is no CONV_RHS filter, so
    // the conv resolver above returned failure. A pool op is packed the same
    // way (activation + result), with a max reduction and no weights.
    FailureOr<Operation *> poolOp = lowerExsleratev2PoolingOpWithEncoding(
        b, linalgOp, convertedOperands,
        cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
    if (succeeded(poolOp)) {
      return poolOp.value();
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

void registerExsleratev2EncodingExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::CPU::IREECPUDialect *dialect) {
        IREE::CPU::Exsleratev2EncodingResolverAttr::attachInterface<
            Exsleratev2EncodingPackedLayoutMaterializerAttr,
            Exsleratev2EncodingResolverMaterializerAttr,
            Exsleratev2LayoutResolverAttr, Exsleratev2SerializableAttr>(*ctx);
      });
}

} // namespace mlir::iree_compiler::IREE::CPU
