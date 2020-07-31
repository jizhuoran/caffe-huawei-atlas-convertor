//===-- XMLManager.h - Manage XML reading and writing----------------------===//
//
// This file defines some XML reading and writing utilities. It defines
// functions that are used to read from XML input files to get parameters
// when performing transformations, and report transformations information
// into output XML files.
//
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_LLVM_SUPPORT_XMLMANAGER_H_
#define INCLUDE_LLVM_SUPPORT_XMLMANAGER_H_

#include "tinyxml2.h"
#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations
namespace autotuning {
class ParameterManager;
class CodeRegion;
class Transformation;
} // namespace autotuning

namespace llvm {
class Error;
}

/// Read a list of parameters from XML input file.
/// Return true on success and false on failure
llvm::Error readFromXML(
    const std::string &InputName, const std::string &ModuleID,
    std::unordered_map<autotuning::CodeRegion, autotuning::ParameterManager>
        &ConfigPerCodeRegionMap);

/// Write a list of results into XML output files.
/// Return true on success and false on failure
template <typename T>
llvm::Error writeToXML(std::vector<T> ModelList, const std::string &TopNodeName,
                       const std::string &Path, const std::string &ModuleID);

#endif /* INCLUDE_LLVM_SUPPORT_XMLMANAGER_H_ */
