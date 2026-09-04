//===-- MemoryDependence.h - Enterprise .hl Memory Dependence Engine -----===//
//
// Part of the OpenHL Infrastructure Project (.hl Ecosystem)
// Licensing under Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the core Memory Dependence Analysis engine for the OpenHL
/// (.hl) IR ecosystem. It evaluates Read-After-Write (RAW), Write-After-Read
/// (WAR), Write-After-Write (WAW), and Read-After-Read (RAR) memory hazards,
/// dependence vectors, and loop iteration distance vectors.
///
//===----------------------------------------------------------------------===//

#ifndef OPENHL_ANALYSIS_MEMORY_MEMORYDEPENDENCE_H
#define OPENHL_ANALYSIS_MEMORY_MEMORYDEPENDENCE_H

#include "openhl/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/RWMutex.h"
#include <memory>
#include <optional>

namespace openhl {
namespace memory {

//===----------------------------------------------------------------------===//
// Dependence Type Classification
//===----------------------------------------------------------------------===//

/// Categorizes the classical memory hazard relationship between two operations.
enum class DependenceType : uint8_t {
  RAW, /// Read-After-Write  (True dependence: Writer executes before Reader)
  WAR, /// Write-After-Read  (Anti dependence: Reader executes before Writer)
  WAW, /// Write-After-Write (Output dependence: Writer executes before Writer)
  RAR, /// Read-After-Read   (No hazard: Concurrent reads allowed)
  None /// Proven to be completely independent (No Memory Overlap)
};

//===----------------------------------------------------------------------===//
// Loop Dependence Distance Vector
//===----------------------------------------------------------------------===//

/// Represents iteration space loop distances for nested loop optimizations.
struct DependenceDistanceVector {
  /// Distance per loop depth level (e.g., [0, 1] -> independent in outer loop, 
  /// dependent on next iteration of inner loop).
  llvm::SmallVector<std::optional<int64_t>, 4> steps;

  /// Returns true if dependence exists entirely within the same loop iteration across all depths.
  bool isLoopIndependent() const {
    for (auto step : steps) {
      if (!step.has_value() || *step != 0)
        return false;
    }
    return true;
  }

  /// Returns true if dependence crosses loop iteration boundaries (Loop-Carried Dependence).
  bool isLoopCarried() const { return !isLoopIndependent(); }
};

//===----------------------------------------------------------------------===//
// Memory Dependence Result Representation
//===----------------------------------------------------------------------===//

/// Detailed structural information holding dependence relationship metrics between two ops.
struct MemoryDependenceResult {
  DependenceType type{DependenceType::None};
  mlir::Operation *sourceOp{nullptr};       /// First executing operation in program order
  mlir::Operation *destinationOp{nullptr};  /// Second executing operation
  mlir::Value accessedBuffer;               /// Root memory buffer value involved
  std::optional<DependenceDistanceVector> distanceVector;

  /// Returns true if a true hazard exists that blocks instruction reordering or fusion.
  bool hasHazard() const {
    return type == DependenceType::RAW || 
           type == DependenceType::WAR || 
           type == DependenceType::WAW;
  }
};

//===----------------------------------------------------------------------===//
// Enterprise Memory Dependence Engine (.hl)
//===----------------------------------------------------------------------===//

/// Main analysis engine responsible for evaluating memory operation conflicts,
/// polyhedral loop bounds dependencies, and instruction reordering permissions.
class HLMemoryDependenceAnalysis {
public:
  explicit HLMemoryDependenceAnalysis(mlir::Operation *rootOp,
                                       analysis::AliasAnalysisManager &aliasMgr);
  ~HLMemoryDependenceAnalysis() = default;

  // Non-copyable, non-movable for static safety
  HLMemoryDependenceAnalysis(const HLMemoryDependenceAnalysis &) = delete;
  HLMemoryDependenceAnalysis &operator=(const HLMemoryDependenceAnalysis &) = delete;

  /// Analyzes memory hazard dependence between two specific memory-access operations.
  MemoryDependenceResult evaluateDependence(mlir::Operation *opA, mlir::Operation *opB);

  /// Returns true if a loop nest contains no loop-carried memory hazards (Safely Parallelizable).
  bool isLoopParallelizable(mlir::LoopLikeOpInterface loopOp);

  /// Collects all memory dependencies within a basic block or region.
  void getDependenciesInRegion(
      mlir::Region &region,
      llvm::SmallVectorImpl<MemoryDependenceResult> &results);

  /// Checks if two memory operations within a loop nest can be safely reordered or fused.
  bool canSafelyReorder(mlir::Operation *opA, mlir::Operation *opB);

  /// Invalidate internal query caches when IR undergoes transformation passes.
  void invalidate();

private:
  /// Evaluates dependence using polyhedral affine integer set constraint solvers if applicable.
  std::optional<DependenceDistanceVector> computeAffineDistance(
      mlir::Operation *opA,
      mlir::Operation *opB);

  /// Helper to verify whether an operation accesses memory locations.
  bool hasMemoryEffects(mlir::Operation *op) const;

  /// Top-level operation context (.hl ModuleOp or FunctionOp)
  mlir::Operation *rootOperation;

  /// Reference to the underlying Alias Analysis Engine
  analysis::AliasAnalysisManager &aliasManager;

  /// Read-Write lock for multi-threaded parallel pass queries
  mutable llvm::sys::SmartRWMutex<true> analysisLock;

  /// Cache for pair queries ((OpA, OpB) -> DependenceResult)
  llvm::DenseMap<std::pair<mlir::Operation *, mlir::Operation *>, MemoryDependenceResult> dependenceCache;
};

//===----------------------------------------------------------------------===//
// Memory Dependence Manager (Pass Integration)
//===----------------------------------------------------------------------===//

/// Manager interface used inside `.hl` optimization passes (`Transforms/`) to automate
/// loop vectorization checking, software pipelining, and safe node scheduling.
class HLMemoryDependenceManager {
public:
  explicit HLMemoryDependenceManager(mlir::Operation *rootOp,
                                     analysis::AliasAnalysisManager &aliasMgr)
      : dependenceEngine(rootOp, aliasMgr) {}

  /// Access the underlying analysis engine
  HLMemoryDependenceAnalysis &getEngine() { return dependenceEngine; }

  /// Helper utility to test if a loop is vectorizable along a target iteration dimension
  bool isVectorizableLoop(mlir::LoopLikeOpInterface loopOp);

private:
  HLMemoryDependenceAnalysis dependenceEngine;
};

} // namespace memory
} // namespace openhl

#endif // OPENHL_ANALYSIS_MEMORY_MEMORYDEPENDENCE_H
