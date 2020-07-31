//===- HiIPUSyncAnalysis.h - Defines SyncAnalysis interface----------------===//
//
// Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for SyncAnalysis that is used for LICM in HiIPU.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HiIPU_HiIPUSYNCANALYSIS_H
#define LLVM_LIB_TARGET_HiIPU_HiIPUSYNCANALYSIS_H

#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include <map>
#include <memory>
#include <set>

namespace llvm {

struct event;
class FunctionPass;
class SyncAnalysis;
class HiIPUSyncAnalysis;

FunctionPass *createHiIPUSyncAnalysisWrapperPass();
// PipeMode and AddressSpace are defined in HiIPU.h. However, we can not just
// include them from that file since Sync Analysis is currently implemented as a
// target-independent pass.
// Target-independent files can not see target-dependent files while compiling
// llvm (this is convenient by definition)
enum PipeMode {
  S = 0,
  V,
  M,
  MTE1,
  MTE2,
  MTE3,
  ALL,
  SYNC,
  UNDEFINED,
};
enum AddressSpace {
  SBUF = 0,
  GM = 1,
  CBUF = 2,
  CA = 3,
  CB = 4,
  CC = 5,
  UBUF = 6,
};
typedef enum {
  EVENT_WAIT,
  EVENT_SET,
  EVENT_BARRIER,
  EVENT_UNDEFINED,
} event_t;

static std::map<const unsigned, event_t> Intr2Event = {
    {Intrinsic::hivm_BARRIER, EVENT_BARRIER},
    {Intrinsic::hivm_SET_FLAG_IMM, EVENT_SET},
    {Intrinsic::hivm_WAIT_FLAG_IMM, EVENT_WAIT},
    {Intrinsic::hivm_SET_FLAG_REG, EVENT_SET},
    {Intrinsic::hivm_WAIT_FLAG_REG, EVENT_WAIT},
};

std::string eventTypeToString(event_t type);
std::string pipeToString(PipeMode type);
PipeMode intrinsic2Pipe(IntrinsicInst *II);

class Event {
public:
  Event();
  Event(Instruction *I);
  ~Event() = default;
  void print(raw_ostream &OS) const;
  void dump() const;

  Instruction *Instr = nullptr;
  PipeMode ProducerPipe = UNDEFINED;
  PipeMode ConsumerPipe = UNDEFINED;
  event_t Type = EVENT_UNDEFINED;
  unsigned Id = UINT_MAX;
  std::set<std::shared_ptr<Event>> EventPairs;
  std::set<Instruction *> Producers;
  std::set<Instruction *> Consumers;
};

class SyncAnalysis {
  std::map<PipeMode, std::set<Instruction *>> Instrs;
  std::set<std::shared_ptr<Event>> Events;

  SyncAnalysis(SyncAnalysis const &) = delete;
  SyncAnalysis &operator=(SyncAnalysis const &) = delete;
  SyncAnalysis(SyncAnalysis &&) = delete;
  SyncAnalysis &operator=(SyncAnalysis &&) = delete;

public:
  SyncAnalysis() = default;
  virtual ~SyncAnalysis() = default;

  void construct_sync_buckets(Function &F);
  void pair_events();
  void populate_producers(const DominatorTree *DT, const LoopInfo *LI);
  void populate_consumers(const DominatorTree *DT, const LoopInfo *LI);
  void updateModRefInfoForUnknownInsts(AliasSetTracker *CurAST, Loop *L);
  bool isIndependent(Instruction *I1, Instruction *I2);
  void print(raw_ostream &OS) const;
  void dump() const;
};

class HiIPUSyncAnalysisWrapperPass : public FunctionPass {
  std::unique_ptr<SyncAnalysis> SA;

  HiIPUSyncAnalysisWrapperPass(HiIPUSyncAnalysisWrapperPass const &) = delete;
  HiIPUSyncAnalysisWrapperPass &
  operator=(HiIPUSyncAnalysisWrapperPass const &) = delete;
  HiIPUSyncAnalysisWrapperPass(HiIPUSyncAnalysisWrapperPass &&) = delete;
  HiIPUSyncAnalysisWrapperPass &
  operator=(HiIPUSyncAnalysisWrapperPass &&) = delete;

public:
  static char ID; // Pass ID
  HiIPUSyncAnalysisWrapperPass() : FunctionPass(ID), SA(new SyncAnalysis()) {}
  virtual ~HiIPUSyncAnalysisWrapperPass(){};

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
  }

  StringRef getPassName() const override { return "HiIPU sync analysis"; }
  bool runOnFunction(Function &F) override;
  SyncAnalysis *getSyncAnalysis() { return SA.get(); }
  SyncAnalysis const *getSyncAnalysis() const { return SA.get(); }
};
} // end namespace llvm

#endif
