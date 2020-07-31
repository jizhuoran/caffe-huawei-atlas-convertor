//===-- AutoTuning.h - Auto-Tuning-----------------------------------------===//
//
// Copyright (C) 2018-2019. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file defines Auto Tuning related functions, models and interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_SUPPORT_AUTOTUNING_H_
#define LIB_SUPPORT_AUTOTUNING_H_

#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace autotuning {

const std::string UNDEFINED =
    "undefined"; // constant string to represent undefined fields

enum CodeRegionType { Other, Module, Loop, MachineBasicBlock };

/// This class represents a range in the source code including
/// the directory path, file name, function name, line numbers and the type that
/// this code region is presenting for.
class CodeRegion {

public:
  CodeRegion(); // default constructor with empty information
  // concrete constructor
  CodeRegion(const std::string &Name, const std::string &FileName,
             const std::string &FuncName, const CodeRegionType &Type);
  CodeRegion(const std::string &Name, const std::string &FileName,
             const std::string &FuncName, const CodeRegionType &Type,
             int StartLine, int EndLine);
  ~CodeRegion() = default;

  bool operator==(const CodeRegion &CR) const;
  bool operator<(const CodeRegion &CR) const;
  inline bool operator!=(const CodeRegion &CR) const { return !(*this == CR); };
  inline bool operator>(const CodeRegion &CR) const { return CR < *this; };
  inline bool operator<=(const CodeRegion &CR) const { return !(*this > CR); };
  inline bool operator>=(const CodeRegion &CR) const { return !(*this < CR); };

  explicit operator bool() const { return !IsEmpty; }

  static const std::string inline CodeRegionTypeToString(
      CodeRegionType CRType) {
    switch (CRType) {
    case autotuning::CodeRegionType::MachineBasicBlock:
      return "machine_basic_block";
    case autotuning::CodeRegionType::Loop:
      return "loop";
    case autotuning::CodeRegionType::Module:
      return "module";
    default:
      return "other";
    }
  }

  // getters and setters
  int getEndLine() const;
  void setEndLine(int EndLine);
  const std::string &getFileName() const;
  void setFileName(const std::string &FileName);
  const std::string &getFuncName() const;
  void setFuncName(const std::string &FuncName);
  int getStartLine() const;
  void setStartLine(int StartLine);
  const CodeRegionType &getType() const;

  const std::string getStringType() const;
  void setType(const CodeRegionType &Type);
  const std::string &getName() const;
  void setName(const std::string &Name);

private:
  bool IsEmpty = true;
  std::string Name;
  std::string FileName; // The file name of this code region.
  std::string FuncName; // The function name of this code region if any.
  CodeRegionType Type;  // The type of this code region. Options are program,
                        // function, loop, instruction,  and other
  int StartLine;        // The start line of this cod region
  int EndLine;          // The end line of this cod region
};

} // end namespace autotuning

namespace std {
template <>
// implement hash for CodeRegion data type in std namespace
struct hash<autotuning::CodeRegion> {
  std::size_t operator()(const autotuning::CodeRegion &CR) const {
    return llvm::hash_combine(CR.getName(), CR.getFuncName(), CR.getFileName(),
                              CR.getType());
  }
};
} // namespace std

namespace autotuning {

class ParameterBase {
public:
  virtual ~ParameterBase() {}
  enum ParameterKind {
    PK_PARAMETER,
  };
  ParameterKind getKind() const { return Kind; }

  ParameterBase(ParameterKind K) : Kind(K) {}

private:
  const ParameterKind Kind;
};

template <typename T> class Parameter : public ParameterBase {
public:
  Parameter(const T &RHS) : ParameterBase(PK_PARAMETER), Value(RHS) {}
  ~Parameter() = default;
  const T &get() const { return Value; }
  void setValue(const T &RHS) { Value = RHS; }

  static bool classof(const ParameterBase *P) {
    return P->getKind() == PK_PARAMETER;
  }

private:
  T Value;
};

/// This class manages parameters of one codeRegion
class ParameterManager {

public:
  // add a param into this ParameterManager
  template <typename T> void add(const std::string &ParamName, T ParamValue) {
    std::shared_ptr<ParameterBase> Param(new Parameter<T>(ParamValue));
    this->Parameters[ParamName] = Param;
  }

  // look up the value of a parameter by name in this ParameterManager
  // the found value will be assigned to the reference variable "value".
  // return true if the parameter exits in this ParameterManager,
  // and false otherwise.
  template <typename T>
  bool findbyName(const std::string &ParamName, T &Value) {

    auto Iterator = Parameters.find(ParamName);
    if (Iterator == Parameters.end())
      return false;

    auto ParamPtr = llvm::dyn_cast<Parameter<T>>(Iterator->second.get());
    if (ParamPtr) {
      Value = ParamPtr->get();
      return true;
    } else {
      return false;
    }
  }

private:
  std::unordered_map<std::string, std::shared_ptr<ParameterBase>> Parameters;
};

///  This class represents a transformation performed by LLVM.
class Transformation {

  // A auto-increment Number based on creations of transformation objects.
  static int CurrentOrder;

public:
  Transformation(const std::string &TransName, const std::string &Category,
                 const std::string &Status, const std::string &Description,
                 const CodeRegion &CR);
  ~Transformation() = default;
  // getters and setters
  const CodeRegion &getCodeRegion() const;
  void setCodeRegion(const CodeRegion &CR);
  const std::string &getDescription() const;
  void setDescription(const std::string &Description);
  void appendDescription(const std::string &Description);
  int getOrder() const;
  void setOrder(int Order);
  const std::string &getStatus() const;
  void setStatus(const std::string &Status);
  const std::string &getTransName() const;
  void setTransName(const std::string &TransName);
  const std::string &getCategory() const;
  void setCategory(const std::string &Category);

private:
  int Order; // the order number of this transformation, based the creating
             // order of the Transformation class

  std::string Category;    // the category of this transformation
  std::string TransName;   // the name of this transformation
  std::string Description; // the description of this transformation
  std::string Status; // the status of this transformation: success or failure
  CodeRegion CR;      // the code region of this transformation
};

/// this class is an interface for elements representing some code regions in
/// LLVM (eg. loop function and basic block) to enable auto-tuning.
class AutoTuningEnabledContainer {

public:
  virtual ~AutoTuningEnabledContainer(){};

  // abstract method
  virtual void initCodeRegion() = 0;
  // getter and setters
  const CodeRegion &getCodeRegion();

  void setCodeRegion(const CodeRegion &CR);

private:
  CodeRegion CR;
};

/// This boolean indicates if the auto-tuning mode is enabled
/// it is set to true if the any of the following command line options
/// (auto-tuning-input, auto-tuning-result and auto-tuning-opp)
/// is specified.
extern bool AutoTuningEnabled;

/// this method is to look up values of parameters that correspond to an
/// AutoTuningEnabledContainer from global ConfigLookUpMap. the parameters being
/// looked up are specified in paramsMap where the key is a parameter's name and
/// the value of is a variable reference which will be assigned to a new lookup
/// value.
template <typename T>
bool lookUpParams(const std::string &ParamsName, T &Value,
                  AutoTuningEnabledContainer *Container);

/// if no AutoTuningEnabledContainer given, it will pick up parameters for
/// the current module being compiled.
template <typename T>
bool lookUpParams(const std::string &ParamsName, T &Value);

/// append a Transformation result into the global TransformationOutputList
void enqueueTransformationResult(const std::string &Category,
                                 const std::string &Status,
                                 const std::string &TransName,
                                 const std::string &Description,
                                 AutoTuningEnabledContainer *Container);

/// append a tuning opportunity into the global CodeRegionOutputList
void enqueueTuningOpportunity(AutoTuningEnabledContainer *Container);

/// initialize the AutoTuningEnabledContainer for auto-tuning.
void initAutoTuningContainer(AutoTuningEnabledContainer *Container);

// Initialization and finalization hooks

/// do initialization for auto-tuning. it method should only be used in main
/// function.
void initAutoTuning(AutoTuningEnabledContainer *Module);
/// do finalization for auto-tuning. it method should only be used in main
/// function.
void endAutoTuning();

} // end namespace autotuning

#endif /* LIB_SUPPORT_AUTOTUNING_H_ */
