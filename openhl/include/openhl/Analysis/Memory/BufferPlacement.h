//===-- BufferPlacement.h - Enterprise .hl Buffer Allocation Engine -------===//
//
// Part of the OpenHL Infrastructure Project (.hl Ecosystem)
// Licensing under Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the core Buffer Placement and Memory Hoisting analysis 
/// engine for the OpenHL (.hl) IR file format. It performs explicit lifetime
/// analysis, buffer placement optimization, memory hierarchy selection, and
/// buffer alias reuse for custom high-performance ML workloads.
///
//===----------------------------------------------------------------------===//

#ifndef OPENHL_ANALYSIS_MEMORY_BUFFERPLACEMENT_H
#define OPENHL_ANALYSIS_MEMORY_BUFFERPLACEMENT_H

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/RWMutex.h"
#include <memory>
#include <optional>

namespace openhl {
namespace memory {

//===----------------------------------------------------------------------===//
// OpenHL Memory Hierarchy Tiers
//===----------------------------------------------------------------------===//

/// Defines target hardware memory spaces for allocated `.hl` buffers.
enum class HLMemorySpace : uint8_t {
  GlobalDRAM = 0,   /// Default off-chip high-capacity device memory
  L2Cache    = 1,   /// Shared multi-core on-chip cache
  SRAM Scratchpad = 2, /// Local high-bandwidth scratchpad memory (Vector/Matrix units)
  RegisterFile   = 3    /// Ultra-fast thread-local register space
};

//===----------------------------------------------------------------------===//
// Buffer Lifetime Interval
//===----------------------------------------------------------------------===//

/// Tracks the precise operational lifecycle of a `.hl` buffer allocation.
struct BufferLifetime {
  mlir::Value allocValue;               /// SSA value representing the allocation
  mlir::Operation *startOperation;      /// Operation where buffer is allocated
  mlir::Operation *lastUserOperation;   /// Operation representing last read/write access
  HLMemorySpace memorySpace;            /// Assigned target memory space
  int64_t requiredSizeBytes{-1};        /// Peak byte size (-1 if dynamic)
  bool escapesScope{false};             /// True if buffer escapes current function scope

  /// Returns true if two buffer lifecycles overlap in execution sequence.
  bool overlaps(const BufferLifetime &other) const;
};

//===----------------------------------------------------------------------===//
// Enterprise Buffer Placement Engine (.hl)
//===----------------------------------------------------------------------===//

/// Primary memory analysis infrastructure responsible for calculating placement,
/// loop hoisting, automatic deallocation insertion points, and buffer aliasing.
class HLBufferPlacementAnalysis {
public:
  explicit HLBufferPlacementAnalysis(mlir::Operation *rootOp);
  ~HLBufferPlacementAnalysis() = default;

  // Non-copyable, non-movable for analysis safety.
  HLBufferPlacementAnalysis(const HLBufferPlacementAnalysis &) = delete;
  HLBufferPlacementAnalysis &operator=(const HLBufferPlacementAnalysis &) = delete;

  /// Main analysis entry point for evaluating a .hl Module or Function.
  mlir::LogicalResult buildAnalysis();

  /// Returns the optimal position to insert a deallocation for a given buffer.
  mlir::Operation *getOptimalDeallocPoint(mlir::Value allocValue) const;

  /// Returns the optimal loop-hoisted scope for allocation if buffer size is static.
  mlir::Operation *getOptimalPlacementScope(mlir::Value allocValue) const;

  /// Returns calculated buffer lifetime metrics for a given allocation SSA value.
  std::optional<BufferLifetime> getBufferLifetime(mlir::Value allocValue) const;

  /// Evaluates whether two distinct allocations can safely share the same physical memory space.
  bool canReuseBuffer(mlir::Value allocA, mlir::Value allocB) const;

  /// Calculates total peak memory footprint in bytes for a target memory space.
  int64_t calculatePeakMemoryUsage(HLMemorySpace space) const;

  /// Invalidate internal analysis states when IR undergoes transformation passes.
  void invalidate();

private:
  /// Identifies all allocation operations (`openhl.alloc`, `memref.alloc`) in current `.hl` IR.
  void collectAllocations(mlir::Operation *root);

  /// Computes precise liveness ranges using MLIR Liveness infrastructure.
  void computeLifetimes();

  /// Identifies potential loop invariant allocations to hoist out of nested loops.
  mlir::Operation *findHoistTarget(mlir::Value allocValue, mlir::Operation *currentScope);

  /// Top-level operation context (.hl ModuleOp or FunctionOp)
  mlir::Operation *rootOperation;

  /// MLIR Liveness analysis driver
  std::unique_ptr<mlir::Liveness> liveness;

  /// Read-Write lock for safe multithreaded pass queries
  mutable llvm::sys::SmartRWMutex<true> analysisLock;

  /// Collection of discovered allocation SSA values
  llvm::SetVector<mlir::Value> allocations;

  /// Mapping from allocation value to calculated lifetime interval
  llvm::DenseMap<mlir::Value, BufferLifetime> lifetimeMap;

  /// Mapping from allocation to optimal deallocation insertion point
  llvm::DenseMap<mlir::Value, mlir::Operation *> deallocPlacementMap;
};

//===----------------------------------------------------------------------===//
// Buffer Placement Manager (.hl Transformation Interface)
//===----------------------------------------------------------------------===//

/// Interface wrapper for `.hl` optimization passes (`Transforms/`) to automate
/// buffer hoisting, placement, and explicit deallocation insertion.
class HLBufferPlacementManager {
public:
  explicit HLBufferPlacementManager(mlir::Operation *rootOp)
      : placementEngine(rootOp) {}

  /// Executes analysis and performs transformations directly on `.hl` IR.
  mlir::LogicalResult runPlacementPipeline();

  /// Access the underlying analysis engine
  HLBufferPlacementAnalysis &getEngine() { return placementEngine; }

private:
  HLBufferPlacementAnalysis placementEngine;
};

} // namespace memory
} // namespace openhl

#endif // OPENHL_ANALYSIS_MEMORY_BUFFERPLACEMENT_H
