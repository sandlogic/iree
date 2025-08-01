// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Transforms.h - Transformations common to all backends --------------===//
//
// Defines transformations that are common to backends
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CODEGEN_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_CODEGEN_TRANSFORMS_TRANSFORMS_H_

#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

/// Get the `offsets`, `sizes` and `strides` for a `storeOp` (or `loadOp`). This
/// method clones the operations that generate the `Value`s used for
/// specifying the offsets, sizesm strides and dynamic dims of the
/// `storeOp/loadOp` at the insertion point to avoid use-def violations.
struct SliceAndDynamicDims {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<Value> dynamicDims;
};
SliceAndDynamicDims
cloneOffsetsSizesAndStrides(OpBuilder &builder,
                            IREE::TensorExt::DispatchTensorStoreOp storeOp);
SliceAndDynamicDims
cloneOffsetsSizesAndStrides(OpBuilder &builder,
                            IREE::TensorExt::DispatchTensorLoadOp loadOp);

/// Creates an allocation in the entry block of the function if the size is
/// statically bounded. For a static allocation, it returns an allocation
/// of the same size but in the entry basic block. For dynamic (still bounded)
/// allocations creates an allocation, and inserts a subview to match the
/// dynamic shape of the allocation. Returns std::nullopt if the method
/// couldnt creat an allocation in the entry block.
template <typename AllocLikeOpType>
std::optional<Value> hoistOneStaticallyBoundAllocation(
    mlir::FunctionOpInterface funcOp, OpBuilder &builder, Location loc,
    MemRefType allocaType, ValueRange dynamicSizes,
    std::optional<uint64_t> alignment,
    std::optional<vector::VscaleRange> vscaleRange = std::nullopt);

/// Hoists `allocaOp` to the entry block of the function if the size is
/// statically bounded. For a static allocation, it returns an allocation
/// of the same size but in the entry basic block. For dynamic (still bounded)
/// allocations creates an allocation, and inserts a subview to match the
/// dynamic shape of the allocation. The method returns a value, but
/// does not replace the uses of the `allocaOp`.
template <typename AllocLikeOpType>
std::optional<Value> hoistOneStaticallyBoundAllocation(
    mlir::FunctionOpInterface funcOp, OpBuilder &builder,
    AllocLikeOpType allocaOp,
    std::optional<vector::VscaleRange> vscaleRange = std::nullopt);

/// Traverse funcOp and try to hoist every AllocaOp to the entry block of the
/// function if the size is statically bounded.
template <typename AllocLikeOpType>
void hoistStaticallyBoundAllocationsInFunc(
    RewriterBase &rewriter, mlir::FunctionOpInterface funcOp,
    std::optional<vector::VscaleRange> vscaleRange = std::nullopt);

/// Insert patterns to perform folding of AffineMinOp by matching the
/// pattern generated by tile and distribute. Try to fold a affine.min op by
/// matching the following form:
/// ```
/// scf.for %iv = %lb to %ub step %step
///   %affine.min affine_map<(d0, d1) -> (N, d0 - d1)>(%ub, %iv)
/// ```
/// With N a compile time constant. This operations can be replace by
/// `%cN = arith.constant N : index` if we can prove that %lb, %step and %ub
/// are divisible by N.
void populateAffineMinSCFCanonicalizationPattern(RewritePatternSet &patterns);

/// Populate patterns that hoist perfectly nested scf.forall ops from parent
/// scf.for ops.
void populateForallLoopHoistingPattern(RewritePatternSet &patterns);

using GetMinMaxExprFn =
    std::function<std::optional<std::pair<AffineExpr, AffineExpr>>(
        Value value, SmallVectorImpl<Value> &dims,
        SmallVectorImpl<Value> &symbols)>;

/// Insert pattern to remove single iteration loop. The pattern will detect
/// single iteration loops based on the range returned ValueBoundsOpInterface.
void populateRemoveSingleIterationLoopPattern(RewritePatternSet &patterns);

// Group of Alloc operations that have overlapping liveranges.
using AliasGroup = SmallVector<Operation *>;

/// Analyze the liverange of the given allocs and set them in individual groups
/// if they don't overlap.
/// The algorithm is a simplistic memory allocation solution. It sorts
/// allocations into alias groups. Everytime two alloc's liverange interfers
/// they are merge into the same group. If a new alloc is part of multiple alias
/// groups all those are merged into one. At the end we are left with groups of
/// allocations that are disjoint and can use the same memory.
void analyseAllocsForPacking(mlir::FunctionOpInterface funcOp,
                             ArrayRef<Operation *> allocs,
                             SmallVector<AliasGroup> &aliasGroups);

/// Pack groups of allocations into a unique large i8 allocation and use
/// memref.view to separate the indivudual allocations. This allows re-using
/// memory across alias groups.
void packAllocs(OpBuilder &builder, mlir::FunctionOpInterface funcOp,
                ArrayRef<AliasGroup> aliasGroups);

/// Materialize the backward slice starting at the values in `workgroupCount`
/// at the current insertion point of the `rewriter`. The leaves of the slice
/// are expected to be `iree_tensor_ext.workload.ordinal` ops that
/// are replaced with the corresponding `workloadVals`. Returns the
/// values corresponding to `workgroupCount` materialized at the insertion
/// point.
FailureOr<SmallVector<OpFoldResult>> materializeWorkgroupCountComputation(
    RewriterBase &rewriter, mlir::FunctionOpInterface entryPointFn,
    ArrayRef<OpFoldResult> workgroupCount, ValueRange workloadVals);

/// Lower the workgroup count region for the default code-generation path in
/// IREE. Given the list `workgroupCount` (fastest varying dimension innermost)
/// as computed within the `entryPointFn`, clones a backward slice of the
/// computation starting at these values and ending with
/// `flow.dispatch.constant_ordinal` into the workgroup count region on the
/// `hal.executable.export` op corresponding to the `entryPointFn`. Also removes
/// the `flow.dispatch.constant_ordinal` operations from within the
/// `entryPointFn`. Expects the workgroup count region of the corresponding
/// `hal.executable.export` to contain the
/// `flow.dispatch.workgroup_count_slice` operation as a placeholder for the
/// computation to compute the number of workgroups. In absence of this
/// operation, this method does nothing assuming that the workgroup count
/// computation has already been resolved.
LogicalResult lowerWorkgroupCountFromSliceOp(
    RewriterBase &rewriter,
    IREE::TensorExt::DispatchWorkgroupCountFromSliceOp workgroupCountOp,
    mlir::FunctionOpInterface entryPointFn,
    ArrayRef<OpFoldResult> workgroupCount,
    int maxWorkgroupParallelDims = kNumMaxParallelDims);

/// Wrapper around `lowerWorkgroupCountFromSliceOp` method that
/// takes the `iree_tensor_ext.dispatch.workgroup_count_from_slice` op
/// as an argument. Looks up the `hal.executable.export` operation
/// and finds the `iree_tensor_ext.dispatch.workgroup_count_from_slice` op to
/// lower.
LogicalResult lowerWorkgroupCountFromSliceOp(
    RewriterBase &rewriter, mlir::FunctionOpInterface entryPointFn,
    ArrayRef<OpFoldResult> workgroupCount,
    int maxWorkgroupParallelDims = kNumMaxParallelDims);

/// Helper to perform LICM on loops nested within |target| that are guaranteed
/// to have at least one trip. Additionally LICM on `scf.forall` ops with
/// mapping attributes are excluded as their trip count is unclear until
/// resolution.
void moveLoopInvariantCodeFromGuaranteedLoops(Operation *target);

/// Populate the pattern to fold `scf.forall` created by split reduction
/// and scf.forall created from workgroup mapping.
void populateFoldSplitReductionAndWorkgroupMappingLoops(
    RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Transformations exposed as patterns, moved from upstream MLIR as IREE still
// heavily relies on patterns that compose through filters.
// TODO: Deprecate all the code below.
//===----------------------------------------------------------------------===//
///
/// Linalg promotion patterns.
///
/// Apply the `promoteSubViews` transformation as a pattern.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `promoteSubViews` for more details.
struct LinalgBasePromotionPattern : public RewritePattern {
  /// Entry point to match any LinalgOp
  /// OpInterface. MatchAnyOpTag-based constructor
  /// with a mandatory `filter`.
  LinalgBasePromotionPattern(
      MLIRContext *context, LinalgTransformationFilter f,
      linalg::LinalgPromotionOptions options = linalg::LinalgPromotionOptions(),
      PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
        filter(std::move(f)), options(std::move(options)) {}
  /// Entry point to match a specific Linalg op.
  LinalgBasePromotionPattern(
      StringRef opName, MLIRContext *context,
      linalg::LinalgPromotionOptions options,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : RewritePattern(opName, benefit, context, {}), filter(std::move(f)),
        options(std::move(options)) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, op)))
      return failure();
    if (failed(promoteSubviewsPrecondition(op, options)))
      return failure();

    // TODO: We cannot use root update here. This
    // pattern is creating other ops, so if the
    // promotion fails, those need to be cleaned
    // up, which doesnt seem to be happening here.
    // So to fail properly, we should be cloning
    // the op and deleting the previous op. This
    // needs more investigation.
    rewriter.startOpModification(op);
    std::optional<linalg::LinalgOp> promotedOp =
        promoteSubViews(rewriter, cast<linalg::LinalgOp>(op), options);
    if (!promotedOp) {
      rewriter.cancelOpModification(op);
      return op->emitError("subview promotion failed");
    }
    rewriter.finalizeOpModification(op);
    filter.replaceLinalgTransformationFilter(rewriter, op);
    return success();
  }

private:
  /// LinalgTransformMarker handles special
  /// attribute manipulations.
  LinalgTransformationFilter filter;
  /// Promotion options.
  linalg::LinalgPromotionOptions options;
};

template <typename OpTy>
struct LinalgPromotionPattern : public LinalgBasePromotionPattern {
  /// SFINAE: This constructor can only trigger for
  /// concrete ops that have a static
  /// `getOperationName` method.
  template <typename ConcreateOpTy = OpTy>
  LinalgPromotionPattern(
      MLIRContext *context, linalg::LinalgPromotionOptions options,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : LinalgBasePromotionPattern(OpTy::getOperationName(), context, options,
                                   f, benefit) {}
  /// This constructor is available to anyone.
  LinalgPromotionPattern(
      StringRef opName, MLIRContext *context,
      linalg::LinalgPromotionOptions options,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : LinalgBasePromotionPattern(opName, context, options, f, benefit) {}
};

/// Tiles LinalgOp ops that match filter.
LogicalResult tileLinalgOpsWithFilter(mlir::FunctionOpInterface funcOp,
                                      scf::SCFTilingOptions options,
                                      LinalgTransformationFilter filter);
/// Distributes LinalgOp ops that match filter.
LogicalResult
distributeLinalgOpsWithFilter(mlir::FunctionOpInterface funcOp,
                              linalg::LinalgTilingOptions tilingOptions,
                              LinalgTransformationFilter filter);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_TRANSFORMS_TRANSFORMS_H_
