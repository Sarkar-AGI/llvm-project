//===-- AliasAnalysis.h - Enterprise Alias Analysis Infrastructure -------===//
//
// Part of the OpenHL Infrastructure Project, under the Apache License v2.0
// with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines an enterprise-grade Alias Analysis infrastructure for
/// the OpenHL dialect ecosystem. It provides memory reference querying,
/// underlying allocation tracing, and multi-tier alias provider composition.
///
//===----------------------------------------------------------------------===//

#ifndef OPENHL_ANALYSIS_ALIASANALYSIS_H
#define OPENHL_ANALYSIS_ALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/RWMutex.h"
#include <memory>
#include <optional>

namespace openhl {
namespace analysis {

//===----------------------------------------------------------------------===//
// Alias Result & ModRef Definitions
//===----------------------------------------------------------------------===//

/// Categorizes the relationship between two memory locations.
using AliasResult = mlir::AliasResult;

/// Defines the modifications and reads an operation can perform on memory.
using ModRefResult = mlir::ModRefResult;

//===----------------------------------------------------------------------===//
// Abstract Alias Analysis Provider
//===----------------------------------------------------------------------===//

/// Abstract base class for all OpenHL alias analysis implementations.
/// Permits composable analysis chains (e.g., Basic, Shape-based, Interprocedural).
class AliasAnalysisProvider {
public:
  virtual ~AliasAnalysisProvider() = default;

  /// Query alias relationship between two values.
  virtual AliasResult alias(mlir::Value vA, mlir::Value vB) = 0;

  /// Query whether an operation modifies or references a given memory location.
  virtual ModRefResult getModRef(mlir::Operation *op, mlir::Value location) = 0;

  /// Returns true if the memory allocated by this value is guaranteed not to
  /// escape the local thread or function scope.
  virtual bool isLocalNoEscape(mlir::Value value) = 0;
};

//===----------------------------------------------------------------------===//
// Underlying Object Tracing Engine
//===----------------------------------------------------------------------===//

/// High-performance helper utility to trace back SSA values to their underlying
/// origin memory allocations across views, subviews, and cast operations.
class MemoryOriginTracer {
public:
  /// Recursively strips view-like operations (Cast, SubView, Reshape, ExtractSlice)
  /// to locate the root allocation value.
  static mlir::Value getUnderlyingObject(mlir::Value value, unsigned maxDepth = 32);

  /// Traces all potential root allocation sources for a value in presence of
  /// control-flow phi nodes or select ops.
  static void getUnderlyingObjects(
      mlir::Value value,
      llvm::SmallVectorImpl<mlir::Value> &objects,
      unsigned maxDepth = 32);

  /// Returns true if the root value represents an isolated, distinct memory allocation.
  static bool isDistinctAllocation(mlir::Value value);
};

//===----------------------------------------------------------------------===//
// Enterprise Basic Alias Analysis Provider
//===----------------------------------------------------------------------===//

/// Primary alias analysis engine that integrates MLIR interfaces with custom
/// high-level dialect heuristics.
class OpenHLBasicAliasAnalysis : public AliasAnalysisProvider {
public:
  explicit OpenHLBasicAliasAnalysis(mlir::Operation *rootOp);
  ~OpenHLBasicAliasAnalysis() override = default;

  // Non-copyable, non-movable for safety within AnalysisManager.
  OpenHLBasicAliasAnalysis(const OpenHLBasicAliasAnalysis &) = delete;
  OpenHLBasicAliasAnalysis &operator=(const OpenHLBasicAliasAnalysis &) = delete;

  /// Core alias query implementation.
  AliasResult alias(mlir::Value vA, mlir::Value vB) override;

  /// ModRef query implementation using SideEffectInterfaces.
  ModRefResult getModRef(mlir::Operation *op, mlir::Value location) override;

  /// Returns true if value is verified non-escaping.
  bool isLocalNoEscape(mlir::Value value) override;

  /// Clears internal caches when IR undergoes localized mutations.
  void invalidateCache();

private:
  /// Evaluates aliasing between view operations on the same underlying root allocation.
  AliasResult aliasSameBaseViews(mlir::Value vA, mlir::Value vB);

  /// Top-level operation context (ModuleOp, FunctionOp, etc.).
  mlir::Operation *rootOperation;

  /// Read-Write lock for thread-safe concurrent queries in parallel pass pipelines.
  mutable llvm::sys::SmartRWMutex<true> cacheLock;

  /// Query memoization cache to guarantee O(1) repeated lookup performance.
  llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, AliasResult> aliasCache;
  llvm::DenseMap<std::pair<mlir::Operation *, mlir::Value>, ModRefResult> modRefCache;
};

//===----------------------------------------------------------------------===//
// Composite Alias Analysis Pipeline Manager
//===----------------------------------------------------------------------===//

/// Pipeline manager that chains multiple `AliasAnalysisProvider` passes together
/// using the Chain-of-Responsibility pattern.
class AliasAnalysisManager {
public:
  explicit AliasAnalysisManager(mlir::Operation *rootOp);

  /// Add a custom analysis provider to the pipeline.
  template <typename T, typename... Args>
  void addProvider(Args &&...args) {
    providers.push_back(std::make_unique<T>(std::forward<Args>(args)...));
  }

  /// Run alias query across all registered providers from finest to coarsest granularity.
  AliasResult alias(mlir::Value vA, mlir::Value vB) const;

  /// Run ModRef query across all registered providers.
  ModRefResult getModRef(mlir::Operation *op, mlir::Value location) const;

  /// Returns true if any provider confirms the memory location is non-escaping.
  bool isLocalNoEscape(mlir::Value value) const;

  /// Clear all provider caches upon IR invalidation.
  void invalidate();

private:
  mlir::Operation *rootOperation;
  llvm::SmallVector<std::unique_ptr<AliasAnalysisProvider>, 4> providers;
};

} // namespace analysis
} // namespace openhl

#endif // OPENHL_ANALYSIS_ALIASANALYSIS_H
