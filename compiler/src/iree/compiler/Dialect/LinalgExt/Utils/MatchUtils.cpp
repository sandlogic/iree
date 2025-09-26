// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "iree-linalgext-utils-matchutils"

// This is based on the upstream implementation of the
// Linalg::ContractionOpInterface.

namespace mlir::iree_compiler::IREE::LinalgExt {

namespace {
constexpr auto par = utils::IteratorType::parallel;
constexpr auto red = utils::IteratorType::reduction;
} // namespace

namespace detail {
enum class MatchContractionResult {
  Success = 0,
  NotLinalgOp,
  WrongNumOperands,
  NoReduction,
  NotProjectedPermutations,
  NotAddMul
};

enum class MatchEXSLTiledConvolutionResult {
  Success = 0,
  NotLinalgOp,
  WrongNoWeightDims,
  WrongNumOperands,
  NotAddMul
};

}

template <typename T>
static T getAffineExprOfType(AffineExpr lhs, AffineExpr rhs) {
  return isa<T>(lhs) ? cast<T>(lhs) : (isa<T>(rhs) ? cast<T>(rhs) : nullptr);
}

namespace {
/// Walk the indexing expressions for input of a convolution operation to verify
/// its of the right form, either
/// - AffineDimExpr
/// - AffineDimExpr (`*` (AffineSymbolExpr | AffineConstantExpr))?
///      (`+` AffineDimExpr (`*` (AffineSymbolExpr | AffineConstantExpr))?)*
///
/// classifies the AffineDimExpr as convolved dimensions or unconvolved
/// dimensions and verifies each dimension occurs only once.
struct ConvAccessExprWalker
    : public AffineExprVisitor<ConvAccessExprWalker, LogicalResult> {
  // Stores dimensions used in expressions of the above form.
  llvm::SmallDenseSet<int64_t> convolvedDims;
  // Stores the dual mapping between LHS and RHS of convolution exprs.
  llvm::SmallDenseMap<int64_t, int64_t> convolvedDimMapping;
  // Stores single use dimensions used by an AffineDimExpr.
  llvm::SmallDenseSet<int64_t> unConvolvedDims;
  // Stores a mapping from convolved dims to their coefficient.
  llvm::SmallDenseMap<int64_t, AffineExpr> strideAndDilationMapping;

  // Removes dims with multiple uses in the source input map from dimension
  // sets tracked by this walker.
  void clearMultiUseDims(AffineMap map) {
    for (int dimPos = 0, e = map.getNumDims(); dimPos < e; ++dimPos) {
      if (llvm::count_if(map.getResults(), [dimPos](AffineExpr e) {
            return e.isFunctionOfDim(dimPos);
          }) > 1) {
        convolvedDims.erase(dimPos);
        unConvolvedDims.erase(dimPos);
        // If a duplicate dim is marked as convolved, the pair of the duplicate
        // dim must be removed from the map as well.
        auto it = convolvedDimMapping.find(dimPos);
        if (it != convolvedDimMapping.end()) {
          int64_t pairedDim = it->second;
          convolvedDims.erase(pairedDim);
          unConvolvedDims.erase(pairedDim);
          strideAndDilationMapping.erase(pairedDim);
          convolvedDimMapping.erase(dimPos);
          convolvedDimMapping.erase(pairedDim);
        }
      }
    }
  }

  LogicalResult visitDimExpr(AffineDimExpr dimExpr) {
    unsigned position = dimExpr.getPosition();
    if (unConvolvedDims.count(position) || convolvedDims.count(position)) {
      return failure();
    }
    unConvolvedDims.insert(position);
    return success();
  }

  LogicalResult visitSymbolExpr(AffineSymbolExpr expr) { return failure(); }

  LogicalResult visitConstantExpr(AffineConstantExpr expr) { return failure(); }

  LogicalResult visitAffineBinaryOpExpr(AffineBinaryOpExpr binaryExpr) {
    // In pre-order visit, top level op has to be an add op.
    if (binaryExpr.getKind() != AffineExprKind::Add)
      return failure();
    auto lhsDimPos = getDimExprOrMulExprDimPos(binaryExpr.getLHS());
    auto rhsDimPos = getDimExprOrMulExprDimPos(binaryExpr.getRHS());
    if (failed(lhsDimPos) || failed(rhsDimPos))
      return failure();
    convolvedDimMapping[*lhsDimPos] = *rhsDimPos;
    convolvedDimMapping[*rhsDimPos] = *lhsDimPos;
    return success();
  }

  FailureOr<int64_t> getDimExprOrMulExprDimPos(AffineExpr expr) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      int64_t dim = dimExpr.getPosition();
      if (convolvedDims.count(dim) || unConvolvedDims.count(dim))
        return failure();
      // Stride/dilation for this dim is implicitly 1.
      strideAndDilationMapping[dim] =
          getAffineConstantExpr(1, expr.getContext());
      convolvedDims.insert(dim);
      return dim;
    }
    if (auto symbolMulExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
      if (symbolMulExpr.getKind() != AffineExprKind::Mul)
        return failure();
      auto lhsExpr = symbolMulExpr.getLHS();
      auto rhsExpr = symbolMulExpr.getRHS();
      // Check for symbol expression.
      AffineExpr mulExpr =
          getAffineExprOfType<AffineSymbolExpr>(lhsExpr, rhsExpr);
      // If there was no symbol expr, check for constant expression.
      if (!mulExpr) {
        mulExpr = getAffineExprOfType<AffineConstantExpr>(lhsExpr, rhsExpr);
      }
      auto dimExpr = getAffineExprOfType<AffineDimExpr>(lhsExpr, rhsExpr);
      if (!mulExpr || !dimExpr)
        return failure();
      int64_t dim = dimExpr.getPosition();
      if (convolvedDims.count(dim) || unConvolvedDims.count(dim))
        return failure();
      strideAndDilationMapping[dim] = mulExpr;
      convolvedDims.insert(dim);
      return dim;
    }
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ScaledContractionOpInterface implementation
//===----------------------------------------------------------------------===//

/// If the value is defined by a chain of unary side-effect free, go up the
/// use-def chain until the first value that isn't defined by such an op.
// TODO: relax to multi-operands with constants, which are technically unary ops
// as needed (e.g. add5).
static Value getSourceSkipUnary(Value value) {
  Operation *op = value.getDefiningOp();
  while (op && op->getNumOperands() == 1) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface || !iface.hasNoEffect()) {
      break;
    }
    value = op->getOperand(0);
    op = value.getDefiningOp();
  }
  return value;
}

/// Returns true if the two operations are ones supported by scaled mfmas.
static bool isValidScaledMmaPair(Operation *add, Operation *mul) {
  constexpr std::pair<StringRef, StringRef> validPairs[4] = {
      {arith::AddFOp::getOperationName(), arith::MulFOp::getOperationName()},
      {arith::AddIOp::getOperationName(), arith::MulIOp::getOperationName()},
      {complex::AddOp::getOperationName(), complex::MulOp::getOperationName()},
      {arith::OrIOp::getOperationName(), arith::AndIOp::getOperationName()}};
  return llvm::any_of(validPairs,
                      [add, mul](std::pair<StringRef, StringRef> p) {
                        return p.first == add->getName().getStringRef() &&
                               p.second == mul->getName().getStringRef();
                      });
}

/// Given an `indexingMap` and its corresponding `iterators`, returns
/// the positions of the iterators of type `iter` that are indexed by
/// the `indexingMap` as a permutation. This is useful to infer various
/// subcomputations on a `LinalgOp`. This is performed by looking up
/// each result in the `indexingMap` and determining whether:
///   - It is a single AffineDimExpr.
///   - It is the only result involving this AffineDimExpr.
static llvm::SmallDenseSet<int64_t>
findPermutationsIndexingOperand(AffineMap indexingMap,
                                ArrayRef<utils::IteratorType> iterators,
                                utils::IteratorType iter) {
  assert(iterators.size() == indexingMap.getNumDims());
  llvm::SmallDenseSet<int64_t> res;
  for (AffineExpr e : indexingMap.getResults()) {
    if (auto d = dyn_cast<AffineDimExpr>(e)) {
      if (iterators[d.getPosition()] == iter &&
          llvm::count_if(indexingMap.getResults(), [d](AffineExpr e) {
            return e.isFunctionOfDim(d.getPosition());
          }) == 1)
        res.insert(d.getPosition());
    }
  }
  return res;
}

/// Infer the iterator types from the init affine map. This looks at which dims
/// are present in the map results, and returns an iterator types array with
/// parallel types for dims that are present, and reduction types for dims that
/// are not present.
static FailureOr<SmallVector<utils::IteratorType>>
inferIteratorsFromOutMap(AffineMap map) {
  if (!map.isProjectedPermutation()) {
    return failure();
  }
  SmallVector<utils::IteratorType> iterators(map.getNumDims(), red);
  for (auto expr : map.getResults()) {
    if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
      iterators[dim.getPosition()] = par;
    }
  }
  return iterators;
}

bool isScaledContractionBody(Block &block) {
  if (block.empty() || !block.back().mightHaveTrait<OpTrait::IsTerminator>()) {
    LDBG() << "no terminator in the block";
    return false;
  }
  if (block.getNumArguments() != 5) {
    LDBG() << "expected block with 3 arguments";
    return false;
  }

  Operation *terminator = block.getTerminator();
  if (terminator->getNumOperands() != 1) {
    LDBG() << "expected terminator with 1 operand";
    return false;
  }

  // Skip cast-like ops and ops with no memory effects
  auto getSourceSkipIrrelevant = [](Value value) -> Value {
    Operation *op = value.getDefiningOp();
    while (op) {
      if ((op->getNumOperands() == 1) ||
          !isa<arith::ExtFOp, arith::TruncFOp, arith::ScalingExtFOp,
               arith::ScalingTruncFOp>(op)) {
        break;
      }
      auto iface = dyn_cast<MemoryEffectOpInterface>(op);
      if (!iface || !iface.hasNoEffect()) {
        break;
      }
      value = op->getOperand(0);
      op = value.getDefiningOp();
    }
    return value;
  };
  Value yielded = getSourceSkipUnary(terminator->getOperand(0));
  Operation *reductionOp = yielded.getDefiningOp();
  if (reductionOp->getNumResults() != 1 || reductionOp->getNumOperands() != 2) {
    LDBG() << "expected reduction op to be binary";
    return false;
  }

  Value reductionLHS = getSourceSkipUnary(reductionOp->getOperand(0));
  Value reductionRHS = getSourceSkipUnary(reductionOp->getOperand(1));

  if (reductionLHS != block.getArgument(4) &&
      reductionRHS != block.getArgument(4)) {
    LDBG() << "expected reduction to take block argument #4 as one of the "
              "operands (modulo unary casts)";
    return false;
  }

  Value contributed = getSourceSkipUnary(
      isa<BlockArgument>(reductionLHS) ? reductionRHS : reductionLHS);
  Operation *elementwiseOp = contributed.getDefiningOp();
  if (!elementwiseOp || elementwiseOp->getNumResults() != 1 ||
      elementwiseOp->getNumOperands() != 2) {
    LDBG() << "expected elementwise op to be binary";
    return false;
  }
  if (!isValidScaledMmaPair(reductionOp, elementwiseOp)) {
    LDBG() << "expected reduction/elementwise op kind not satisfied";
    return false;
  }

  Value elementwiseLHS = getSourceSkipIrrelevant(elementwiseOp->getOperand(0));
  Value elementwiseRHS = getSourceSkipIrrelevant(elementwiseOp->getOperand(1));
  if ((elementwiseLHS == block.getArgument(0) &&
       elementwiseRHS == block.getArgument(1)) ||
      (elementwiseLHS == block.getArgument(1) &&
       elementwiseRHS == block.getArgument(0))) {
    return true;
  }

  LDBG() << "expected elementwise op to apply to block arguments (modulo unary "
            "casts)";
  return false;
}

static FailureOr<ScaledContractionDimensions>
inferScaledContractionDimsImpl(ArrayRef<AffineMap> indexingMaps,
                               ArrayRef<utils::IteratorType> iterators) {
  llvm::SmallDenseSet<int64_t> a =
      findPermutationsIndexingOperand(indexingMaps[0], iterators, par);
  llvm::SmallDenseSet<int64_t> b =
      findPermutationsIndexingOperand(indexingMaps[1], iterators, par);
  llvm::SmallDenseSet<int64_t> c =
      findPermutationsIndexingOperand(indexingMaps[4], iterators, par);
  // A & C - B are the iterators involved in an outer-product along A (the LHS).
  llvm::SmallDenseSet<int64_t> ac = a;
  llvm::set_intersect(ac, c);
  llvm::set_subtract(ac, b);
  // B & C - A are the iterators involved in an outer-product along B (the RHS).
  llvm::SmallDenseSet<int64_t> bc = b;
  llvm::set_intersect(bc, c);
  llvm::set_subtract(bc, a);
  // A & B & C are the "batch" dimensions.
  llvm::SmallDenseSet<int64_t> batches = a;
  llvm::set_intersect(batches, b);
  llvm::set_intersect(batches, c);

  // Scale A & Scale B is k reduction dimension.
  llvm::SmallDenseSet<int64_t> sa =
      findPermutationsIndexingOperand(indexingMaps[2], iterators, red);
  llvm::SmallDenseSet<int64_t> sb =
      findPermutationsIndexingOperand(indexingMaps[3], iterators, red);
  llvm::set_intersect(sa, sb);

  // A red & B red - Scale A & Scale B is k_b reduction dimension.
  llvm::SmallDenseSet<int64_t> ra =
      findPermutationsIndexingOperand(indexingMaps[0], iterators, red);
  llvm::SmallDenseSet<int64_t> rb =
      findPermutationsIndexingOperand(indexingMaps[1], iterators, red);
  llvm::set_intersect(ra, rb);
  llvm::set_subtract(ra, sa);

  // Return each set in sorted order.
  ScaledContractionDimensions dimensions{
      llvm::to_vector_of<unsigned, 2>(batches),
      llvm::to_vector_of<unsigned, 2>(ac), llvm::to_vector_of<unsigned, 2>(bc),
      llvm::to_vector_of<unsigned, 2>(sa), llvm::to_vector_of<unsigned, 2>(ra)};
  llvm::sort(dimensions.batch);
  llvm::sort(dimensions.m);
  llvm::sort(dimensions.n);
  llvm::sort(dimensions.k);
  llvm::sort(dimensions.kB);
  return dimensions;
}

FailureOr<ScaledContractionDimensions>
inferScaledContractionDims(ArrayRef<AffineMap> indexingMaps) {
  if (indexingMaps.size() != 5) {
    return failure();
  }
  auto iterators = inferIteratorsFromOutMap(indexingMaps[4]);
  if (failed(iterators)) {
    return failure();
  }
  return inferScaledContractionDimsImpl(indexingMaps, iterators.value());
}

FailureOr<ScaledContractionDimensions>
inferScaledContractionDims(linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumDpsInits() != 1 || linalgOp.getNumDpsInputs() != 4) {
    return failure();
  }
  return inferScaledContractionDimsImpl(linalgOp.getIndexingMapsArray(),
                                        linalgOp.getIteratorTypesArray());
}

detail::MatchContractionResult
isScaledContractionImpl(Operation *op,
                        ScaledContractionDimensions *dimensions) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return detail::MatchContractionResult::NotLinalgOp;
  }
  if (linalgOp.getNumDpsInputs() != 4 || linalgOp.getNumDpsInits() != 1) {
    return detail::MatchContractionResult::WrongNumOperands;
  }
  auto mapRange = linalgOp.getIndexingMapsArray();
  if (linalgOp.getNumReductionLoops() == 0) {
    return detail::MatchContractionResult::NoReduction;
  }
  if (llvm::any_of(mapRange,
                   [](AffineMap m) { return !m.isProjectedPermutation(); })) {
    return detail::MatchContractionResult::NotProjectedPermutations;
  }
  // TODO: more fields than add/mul.
  // clang-format off
  if (!isScaledContractionBody(*linalgOp.getBlock())) {
    return detail::MatchContractionResult::NotAddMul;
  }
  // clang-format on
  if (dimensions) {
    FailureOr<ScaledContractionDimensions> res =
        inferScaledContractionDims(linalgOp);
    assert(succeeded(res) && "unexpected failure to infer contraction dims");
    *dimensions = *res;
  }
  return detail::MatchContractionResult::Success;
}

bool isaScaledContractionOpInterface(linalg::LinalgOp linalgOp) {
  if (!linalgOp) {
    return false;
  }
  Operation *op = linalgOp.getOperation();
  return isScaledContractionImpl(op) == detail::MatchContractionResult::Success;
}

detail::MatchEXSLTiledConvolutionResult
isEXSLTiledConvolutionInterfaceImpl(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return detail::MatchEXSLTiledConvolutionResult::NotLinalgOp;
  }

  if (linalgOp.getNumDpsInputs() < 2 || linalgOp.getNumDpsInits() != 1)
    return detail::MatchEXSLTiledConvolutionResult::WrongNumOperands;

  auto indexingMaps = linalgOp.getIndexingMapsArray();

  if (indexingMaps[1].getResults().size() != 6)
    return detail::MatchEXSLTiledConvolutionResult::WrongNoWeightDims;

  return detail::MatchEXSLTiledConvolutionResult::Success;
}

bool isaEXSLTileConvolutionOpInterface(linalg::LinalgOp linalgOp) {
  if (!linalgOp) {
    return false;
  }
  return isEXSLTiledConvolutionInterfaceImpl(linalgOp.getOperation()) ==
         detail::MatchEXSLTiledConvolutionResult::Success;
}

FailureOr<linalg::ConvolutionDimensions> inferConvolutionDimsImpl(
    ArrayRef<AffineMap> indexingMaps, ArrayRef<utils::IteratorType> iterators,
    ConvAccessExprWalker &inputExprWalker, bool allowEmptyConvolvedDims) {
  if (indexingMaps.size() < 3) {
    return failure();
  }
  llvm::SmallDenseSet<int64_t> filterDims =
      findPermutationsIndexingOperand(indexingMaps[1], iterators, par);
  llvm::SmallDenseSet<int64_t> outputDims =
      findPermutationsIndexingOperand(indexingMaps[2], iterators, par);

  // unConvolvedDims & outputDims - filterDims are the batch iterators.
  llvm::SmallDenseSet<int64_t> batch = inputExprWalker.unConvolvedDims;
  llvm::set_intersect(batch, outputDims);
  llvm::set_subtract(batch, filterDims);

  // convolvedDims & outputDims are the output image iterators.
  llvm::SmallDenseSet<int64_t> oi = inputExprWalker.convolvedDims;
  llvm::set_intersect(oi, outputDims);

  // filterDims & outputDims - unConvolvedDims are the output channel iterators.
  llvm::SmallDenseSet<int64_t> oc = filterDims;
  llvm::set_intersect(oc, outputDims);
  llvm::set_subtract(oc, inputExprWalker.unConvolvedDims);

  // filterDims & outputDims & unConvolvedDims are the depth iterators.
  llvm::SmallDenseSet<int64_t> depth = filterDims;
  llvm::set_intersect(depth, outputDims);
  llvm::set_intersect(depth, inputExprWalker.unConvolvedDims);

  llvm::SmallDenseSet<int64_t> filterReducedDims =
      findPermutationsIndexingOperand(indexingMaps[1], iterators, red);

  // convolvedDims & filterReducedDims are the filter loop iterators.
  llvm::SmallDenseSet<int64_t> fl = inputExprWalker.convolvedDims;
  llvm::set_intersect(fl, filterReducedDims);

  // unConvolvedDims & filterReducedDims are the input channel iterators.
  llvm::SmallDenseSet<int64_t> ic = inputExprWalker.unConvolvedDims;
  llvm::set_intersect(ic, filterReducedDims);

  if (oi.empty() && !allowEmptyConvolvedDims)
    return failure();

  // Return each set in sorted order.
  linalg::ConvolutionDimensions dimensions{
      SmallVector<unsigned, 2>(batch.begin(), batch.end()),
      SmallVector<unsigned, 2>(oi.begin(), oi.end()),
      SmallVector<unsigned, 2>(oc.begin(), oc.end()),
      SmallVector<unsigned, 2>(fl.begin(), fl.end()),
      SmallVector<unsigned, 2>(ic.begin(), ic.end()),
      SmallVector<unsigned, 2>(depth.begin(), depth.end()),
      /*strides=*/SmallVector<int64_t, 2>{},
      /*dilations=*/SmallVector<int64_t, 2>{}};
  llvm::sort(dimensions.batch);
  llvm::sort(dimensions.outputImage);
  llvm::sort(dimensions.outputChannel);
  llvm::sort(dimensions.filterLoop);
  llvm::sort(dimensions.inputChannel);
  llvm::sort(dimensions.depth);

  return dimensions;
}

FailureOr<linalg::ConvolutionDimensions>
inferConvolutionDims(ArrayRef<AffineMap> indexingMaps) {
  if (indexingMaps.size() < 3) {
    return failure();
  }
  auto iterators = inferIteratorsFromOutMap(indexingMaps[2]);
  if (failed(iterators)) {
    return failure();
  }

  ConvAccessExprWalker inputExprWalker;
  for (AffineExpr expr : indexingMaps[0].getResults())
    (void)inputExprWalker.visit(expr);
  inputExprWalker.clearMultiUseDims(indexingMaps[0]);

  return inferConvolutionDimsImpl(indexingMaps, iterators.value(),
                                  inputExprWalker,
                                  /*allowEmptyConvolvedDims=*/false);
}

}; // namespace mlir::iree_compiler::IREE::LinalgExt
