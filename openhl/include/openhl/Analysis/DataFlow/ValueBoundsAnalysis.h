//===-- ValueBoundsAnalysis.h - Enterprise Value Bounds Infrastructure -===//
//
// Part of the OpenHL Infrastructure Project, under the Apache License v2.0
// with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides an ultra-scalable Value Bounds and Shape Analysis
/// infrastructure for OpenHL dialect ecosystems. It interfaces with MLIR's
/// ValueBoundsConstraintSet to compute upper/lower numerical bounds, symbolic
/// relationships, and index equivalences across dynamic shapes and values.
///
//===----------------------------------------------------------------------===//

#ifndef OPENHL_ANALYSIS_DATAFLOW_VALUEBOUNDSANALYSIS_H
#define OPENHL_ANALYSIS_DATAFLOW_VALUEBOUNDSANALYSIS_H

#include "mlir/Analysis/ValueBoundsOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/RWMutex.h"
#include <optional>

namespace openhl {
namespace analysis {

//===----------------------------------------------------------------------===//
// Symbolic Bound Relation Enum
//===----------------------------------------------------------------------===//

/// Represents relational comparisons between two dynamic values or shape dimensions.
enum class PresburgerRelation {
  EQ, // Value A == Value B
  LT, // Value A <  Value B
  LE, // Value A <= Value B
  GT, // Value A >  Value B
  GE, // Value A >= Value B
  NE  // Value A != Value B (Unproven/Unknown overlap)
};

//===----------------------------------------------------------------------===//
// Constant Bound Result
//===----------------------------------------------------------------------===//

/// Struct representing the computed integer bounds for a value or dynamic dimension.
struct ConstantValueBounds {
  std::optional<int64_t> lowerBound;
  std::optional<int64_t> upperBound;

  bool isConstant() const {
    return lowerBound.has_value() && upperBound.has_value() &&
           *lowerBound == *upperBound;
  }

  std::optional<int64_t> getConstantValue() const {
    if (isConstant())
      return *lowerBound;
    return std::nullopt;
  }
};

//===----------------------------------------------------------------------===//
// Core Value Bounds Analysis Engine
//===----------------------------------------------------------------------===//

/// High-performance thread-safe analysis engine that computes bounds on SSA values 
/// and dynamic tensor/memref dimensions using integer polyhedral sets.
class OpenHLValueBoundsAnalysis {
public:
  explicit OpenHLValueBoundsAnalysis(mlir::Operation *rootOp);
  ~OpenHLValueBoundsAnalysis() = default;

  // Non-copyable, non-movable for analysis safety
  OpenHLValueBoundsAnalysis(const OpenHLValueBoundsAnalysis &) = delete;
  OpenHLValueBoundsAnalysis &operator=(const OpenHLValueBoundsAnalysis &) = delete;

  /// Computes the constant lower and upper bounds for a given SSA index value.
  ConstantValueBounds computeConstantBounds(mlir::Value value);

  /// Computes the constant lower and upper bounds for a dynamic dimension of a Tensor/MemRef.
  ConstantValueBounds computeConstantDimensionBounds(mlir::Value shapedValue,
                                                       unsigned dimIndex);

  /// Compares two SSA scalar values symbolically using IntegerRelation analysis.
  bool compareValues(mlir::Value valueA,
                     PresburgerRelation relation,
                     mlir::Value valueB);

  /// Compares two shaped value dimensions symbolically (e.g., dim(A, 0) == dim(B, 1)).
  bool compareDimensions(mlir::Value shapedValueA, unsigned dimA,
                         PresburgerRelation relation,
                         mlir::Value shapedValueB, unsigned dimB);

  /// Returns true if two shaped values are guaranteed to have identical runtime shapes.
  bool areShapesEquivalent(mlir::Value shapedValueA, mlir::Value shapedValueB);

  /// Invalidate internal caching when passes mutate IR.
  void invalidate();

private:
  /// Low-level constraint set builder helper.
  mlir::LogicalResult populateConstraints(
      mlir::ValueBoundsConstraintSet &cstrSet,
      mlir::Value value,
      std::optional<unsigned> dim = std::nullopt);

  /// Root scope operation (ModuleOp or FunctionOp)
  mlir::Operation *rootOperation;

  /// RW Mutex for parallel thread queries during multi-threaded optimization passes
  mutable llvm::sys::SmartRWMutex<true> boundsLock;

  /// Cache for constant bounds queries (Value -> Bounds)
  llvm::DenseMap<mlir::Value, ConstantValueBounds> valueBoundsCache;
  
  /// Cache for dynamic shape dimensions ((Value, Dim) -> Bounds)
  llvm::DenseMap<std::pair<mlir::Value, unsigned>, ConstantValueBounds> dimBoundsCache;
};

//===----------------------------------------------------------------------===//
// Value Bounds Analysis Manager (Pass Integration)
//===----------------------------------------------------------------------===//

/// Managed wrapper interface for seamless execution inside MLIR Passes.
class ValueBoundsAnalysisManager {
public:
  explicit ValueBoundsAnalysisManager(mlir::Operation *rootOp)
      : analysisEngine(rootOp) {}

  /// Access the underlying analysis engine
  OpenHLValueBoundsAnalysis &getEngine() { return analysisEngine; }

  /// Helper utility to attempt dynamic shape specialization based on bounds
  mlir::LogicalResult deriveStaticShapeIfPossible(
      mlir::Value shapedValue,
      llvm::SmallVectorImpl<int64_t> &inferredShapes);

private:
  OpenHLValueBoundsAnalysis analysisEngine;
};

} // namespace analysis
} // namespace openhl

#endif // OPENHL_ANALYSIS_DATAFLOW_VALUEBOUNDSANALYSIS_H
