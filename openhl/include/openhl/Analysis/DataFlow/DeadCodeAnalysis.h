//===-- DeadCodeAnalysis.h - Ultra Scalable Dead Code Analysis Framework --===//
//
// Part of the OpenHL Infrastructure Project, under the Apache License v2.0
// with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides an ultra-scale, enterprise-grade Dead Code Analysis
/// framework for OpenHL dialect infrastructures. It extends MLIR's sparse
/// and dense dataflow framework to analyze executable control flow, dead
/// SSA values, unreachable blocks, and side-effect-free unused operations.
///
//===----------------------------------------------------------------------===//

#ifndef OPENHL_ANALYSIS_DATAFLOW_DEADCODEANALYSIS_H
#define OPENHL_ANALYSIS_DATAFLOW_DEADCODEANALYSIS_H

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/RWMutex.h"
#include <memory>

namespace openhl {
namespace analysis {

//===----------------------------------------------------------------------===//
// Liveness State Definition
//===----------------------------------------------------------------------===//

/// Represents the liveness state of an SSA Value, Block, or Control Edge.
class LivenessState : public mlir::dataflow::AnalysisState {
public:
  using AnalysisState::AnalysisState;

  enum class LiveLattice {
    Dead,      /// Completely unreachable / unused.
    MaybeLive, /// Conditionally reachable based on branch conditions.
    Live       /// Known to be live and reachable.
  };

  /// Build liveness state bound to a program point or value.
  explicit LivenessState(mlir::DataFlowSolver::Position anchor)
      : mlir::dataflow::AnalysisState(anchor), state(LiveLattice::Dead) {}

  /// Returns true if the associated element is determined to be live.
  bool isLive() const { return state == LiveLattice::Live; }
  
  /// Returns true if the element is dead.
  bool isDead() const { return state == LiveLattice::Dead; }

  /// Meets current state with a new state. Returns change status for solver convergence.
  mlir::ChangeResult meet(LiveLattice newState);

  /// Meets current state with another lattice instance.
  mlir::ChangeResult join(const LivenessState &rhs);

  void print(llvm::raw_ostream &os) const override;

private:
  LiveLattice state{LiveLattice::Dead};
};

//===----------------------------------------------------------------------===//
// Enterprise Dead Code Analysis Engine
//===----------------------------------------------------------------------===//

/// Comprehensive dead-code analysis solver component capable of interprocedural
/// analysis, control-flow reachability evaluation, and memory-effect auditing.
class OpenHLDeadCodeAnalysis : public mlir::dataflow::DeadCodeAnalysis {
public:
  explicit OpenHLDeadCodeAnalysis(mlir::DataFlowSolver &solver);
  ~OpenHLDeadCodeAnalysis() override = default;

  // Non-copyable, non-movable for static analysis integrity.
  OpenHLDeadCodeAnalysis(const OpenHLDeadCodeAnalysis &) = delete;
  OpenHLDeadCodeAnalysis &operator=(const OpenHLDeadCodeAnalysis &) = delete;

  /// Initializes analysis state starting from top-level module/function entry points.
  mlir::LogicalResult initialize(mlir::Operation *topLevelOp) override;

  /// Main transfer function for evaluating operational live status.
  mlir::LogicalResult visit(mlir::ProgramPoint point) override;

  /// Evaluates whether a given block is executable in the control flow graph.
  bool isBlockExecutable(mlir::Block *block) const;

  /// Evaluates whether an operation is dead (pure with no live results/uses).
  bool isOperationDead(mlir::Operation *op) const;

  /// Evaluates whether an SSA Value has active, live uses in reachable code.
  bool isValueLive(mlir::Value value) const;

  /// Queries whether an interprocedural call graph edge is active.
  bool isCallSiteExecutable(mlir::CallOpInterface callOp) const;

private:
  /// Evaluates branch condition operations to prune non-executable successor paths.
  void processBranchOperation(
      mlir::BranchOpInterface branchOp,
      mlir::Block *currentBlock);

  /// Evaluates regional control flow operations (e.g., scf.if, scf.for, custom loops).
  void processRegionBranchOperation(
      mlir::RegionBranchOpInterface regionOp,
      mlir::ProgramPoint point);

  /// Audits whether an operation has critical side-effects preventing elimination.
  bool hasCriticalSideEffects(mlir::Operation *op) const;

  /// Thread-safe read-write mutex for parallel solver workers.
  mutable llvm::sys::SmartRWMutex<true> analysisLock;

  /// Memoized cache of validated dead operations for fast external querying.
  mutable llvm::DenseMap<mlir::Operation *, bool> deadOpCache;
  mutable llvm::DenseSet<mlir::Block *> executableBlocks;
};

//===----------------------------------------------------------------------===//
// Dead Code Analysis Query Driver
//===----------------------------------------------------------------------===//

/// Query interface used inside Pass pipelines (`Transforms/`) to retrieve
/// verified dead code analysis metrics.
class DeadCodeAnalysisManager {
public:
  explicit DeadCodeAnalysisManager(mlir::Operation *rootOp);

  /// Runs the full dataflow analysis solver pipeline.
  mlir::LogicalResult runAnalysis();

  /// Collects all dead operations in top-down order ready for safe deletion.
  void getDeadOperations(llvm::SmallVectorImpl<mlir::Operation *> &deadOps) const;

  /// Collects all unreachable basic blocks for CFG simplification passes.
  void getUnreachableBlocks(llvm::SmallVectorImpl<mlir::Block *> &deadBlocks) const;

  /// Query direct liveness status of an SSA Value.
  bool isDead(mlir::Value value) const;

  /// Query direct liveness status of an Operation.
  bool isDead(mlir::Operation *op) const;

  /// Invalidate internal solver state when IR mutates.
  void invalidate();

private:
  mlir::Operation *rootOperation;
  mlir::DataFlowSolver solver;
  OpenHLDeadCodeAnalysis *deadCodeAnalysis{nullptr};
  bool isExecuted{false};
};

} // namespace analysis
} // namespace openhl

#endif // OPENHL_ANALYSIS_DATAFLOW_DEADCODEANALYSIS_H￼Enter
