#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

auto_schedule distribution rules

Add your rules to pattern mapping here so cce auto_schedule can distribute your operators
to corresponding schedule correctly
"""
from te import tvm

from .cce_schedule_distribution_rules import OpPatternRules, OpSubPatternRules, OpSpecialRules
from .cce_schedule_declarations import OpFlags, OpPatterns, OpSubPatterns, OpSpecTypes


# Define all rules for the pattern here
# Be aware: ***rules at the front have a higher priority***
# WARNING:  ***Python 3.6 and lower version has an unordered dict impl***
# WARNING:  ***Priority control will be broken on those python env***
##############################################################################
#                HOW   TO   ADD   A   NEW   PATTERN   RULE                   #
#                                                                            #
# 1. Add your pattern in cce_schedule_declarations.py OpPatterns class       #
# 2. Add your custom tag in OpFlags class, if needed                         #
# 3. Add your rule here, in cce_schedule_mappings.OP_PATTERN_RULES           #
# 4. Add your pattern2flag mapping in cce_schedule_mapping.SIMPLE_MAPPING    #
#    Only if you need to use simple_pattern_rule                             #
##############################################################################
# (Advanced) If you want to add your custom rule, see
# cce_schedule_distribution_rules.py OpPatternRules class

##############################################################################
#                    WHAT IS SIMPLE PATTERN RULE ?                           #
# simple_pattern_rule is a built_in rule used for one-step pattern setup     #
# As its name, the logic is simple:                                          #
# If any of the computeOp tag matched your pattern's simple mapping flag     #
# (Which is configured in cce_schedule_mappings.SIMPLE_MAPPING)              #
# simple_pattern_rule will return True for your pattern                      #
##############################################################################

# Read cce_schedule_distribution_rules.py OpPatternRules class
# for all available rules
OP_PATTERN_RULES = {
    # Custom op patterns have the highest priority
    OpPatternRules.bn_reduce_pattern_rule: OpPatterns.BN_REDUCE_PATTERN,
    OpPatternRules.bn_update_pattern_rule: OpPatterns.BN_UPDATE_PATTERN,
    OpPatternRules.softmax_cross_entropy_with_logits_pattern_rule:
        OpPatterns.SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_PATTERN,
    OpPatternRules.l2_normalize_pattern_rule: OpPatterns.L2_NORMALIZE_PATTERN,
    OpPatternRules.l2_loss_pattern_rule: OpPatterns.L2_LOSS_PATTERN,
    OpPatternRules.bn_update_grad_pattern_rule: OpPatterns.BN_UPDATE_GRAD_PATTERN,
    OpPatternRules.bn_grad_reduce_pattern_rule: OpPatterns.BN_GRAD_REDUCE_PATTERN,
    OpPatternRules.layer_norm_grad_pattern_rule: OpPatterns.LAYER_NORM_GRAD_PATTERN,
    OpPatternRules.l2loss_mul_addn_pattern_rule: OpPatterns.L2LOSS_MUL_ADDN_PATTERN,
    # Softmax pattern can override normal pattern in old cce_schedule.py
    OpPatternRules.softmax_pattern_rule: OpPatterns.SOFTMAX_PATTERN,
    # Convolution phase
    OpPatternRules.simple_pattern_rule_conv_phase:
    (OpPatterns.CONV_PATTERN, OpPatterns.CONV3D_BACKPROP_INPUT_PATTERN,
     OpPatterns.CONV3D_BACKPROP_FILTER_PATTERN, OpPatterns.CONV3D_PATTERN,
     OpPatterns.CONV2D_BACKPROP_INPUT_PATTERN,
     OpPatterns.CONV2D_BACKPROP_FILTER_PATTERN,
     OpPatterns.DEPTHWISECONV_PATTERN),
    # Normal schedules
    OpPatternRules.simple_pattern_rule: (OpPatterns.MATMUL_V2_PATTERN,
                                         OpPatterns.MATMUL_PATTERN,
                                         OpPatterns.POOL2D_PATTERN,
                                         OpPatterns.MAX_POOL_PATTERN,
                                         OpPatterns.READ_SELECT_PATTERN,
                                         OpPatterns.WRITE_SELECT_PATTERN,
                                         OpPatterns.STRIDED_READ_PATTERN,
                                         OpPatterns.STRIDED_WRITE_PATTERN),
    OpPatternRules.single_only_pattern_rule: (OpPatterns.SEGMENT_PATTERN,
                                              OpPatterns.INPLACE_PATTERN,
                                              OpPatterns.PURE_BROADCAST_PATTERN),
    OpPatternRules.reduce_pattern_rule: OpPatterns.REDUCE_PATTERN,
    OpPatternRules.elewise_pattern_rule: OpPatterns.ELEMWISE_PATTERN,

    OpPatternRules.simple_pattern_rule_quant_phase: (OpPatterns.ASCEND_ANTI_QUANT_PATTERN,
                                                     OpPatterns.ASCEND_REQUANT_S16_PATTERN,
                                                     OpPatterns.ASCEND_REQUANT_PATTERN,
                                                     OpPatterns.ASCEND_DEQUANT_S16_PATTERN,
                                                     OpPatterns.ASCEND_DEQUANT_PATTERN,
                                                     OpPatterns.ASCEND_QUANT_PATTERN)
}
OP_SUBPATTERN_RULES = {
    OpSubPatternRules.reduce_5hdc_sub_pattern_rule: OpSubPatterns.REDUCE_5HDC_PATTERN,
    OpSubPatternRules.reduce_atomic_sub_pattern_rule: OpSubPatterns.REDUCE_ATOMIC_PATTERN,
    OpSubPatternRules.simple_pattern_rule: (OpSubPatterns.CMPSEL_PATTERN,)
}
OP_SPEC_RULES = {
    OpSpecialRules.relu_grad_v2_operation_rule: OpSpecTypes.RELU_GRAD_V2,
    OpSpecialRules.relu_v2_operation_rule: OpSpecTypes.RELU_V2,
    OpSpecialRules.normalize_scale_operation_rule: OpSpecTypes.NORMALIZE_SCALE,
    OpSpecialRules.only_broadcast_operation_rule: OpSpecTypes.ONLY_BROADCAST_TYPE,
    OpSpecialRules.mvn_operation_rule: OpSpecTypes.MVN,
    OpSpecialRules.bn_ext2_operation_rule: OpSpecTypes.BN_EXT2,
    OpSpecialRules.reduce_mean_2d_aligned_mid_reduce_no_cast_pattern_rule:
    OpSpecTypes.REDUCE_MEAN_2D_ALIGNED_MID_REDUCE_NO_CAST,
}
SIMPLE_MAPPING = {
    OpPatterns.CONV_PATTERN: OpFlags.convolution_flag,
    OpPatterns.CONV2D_BACKPROP_FILTER_PATTERN: OpFlags.conv2d_backprop_filter_flag,
    OpPatterns.CONV2D_BACKPROP_INPUT_PATTERN: OpFlags.conv2d_backprop_input_flag,
    OpPatterns.CONV3D_BACKPROP_INPUT_PATTERN: OpFlags.conv3d_backprop_input_flag,
    OpPatterns.MATMUL_PATTERN: OpFlags.matmul_flag,
    OpPatterns.MATMUL_V2_PATTERN: OpFlags.matmul_v2_flag,
    OpPatterns.POOL2D_PATTERN: (OpFlags.pool2d_avg_flag,
                                OpFlags.pool2d_max_flag,
                                OpFlags.pool2d_gap_flag,
                                OpFlags.pool2d_gmp_flag),
    OpPatterns.MAX_POOL_PATTERN: OpFlags.max_pool_flag,
    OpPatterns.ASCEND_QUANT_PATTERN: OpFlags.ascend_quant_flag,
    OpPatterns.ASCEND_DEQUANT_PATTERN: OpFlags.ascend_dequant_flag,
    OpPatterns.ASCEND_ANTI_QUANT_PATTERN: OpFlags.ascend_anti_quant_flag,
    OpPatterns.ASCEND_DEQUANT_S16_PATTERN: OpFlags.ascend_dequant_s16_flag,
    OpPatterns.ASCEND_REQUANT_PATTERN: OpFlags.ascend_requant_flag,
    OpPatterns.ASCEND_REQUANT_S16_PATTERN: OpFlags.ascend_requant_s16_flag,
    OpPatterns.DEPTHWISECONV_PATTERN: OpFlags.depthwise_conv_flag,
    OpPatterns.CONV3D_PATTERN: OpFlags.conv3d_flag,
    OpPatterns.SEGMENT_PATTERN: OpFlags.segment_flag,
    OpPatterns.INPLACE_PATTERN: OpFlags.inplace_flag,
    OpPatterns.PURE_BROADCAST_PATTERN:OpFlags.broadcast_flag,
    OpPatterns.READ_SELECT_PATTERN: OpFlags.read_select_flag,
    OpPatterns.WRITE_SELECT_PATTERN: OpFlags.write_select_flag,
    OpPatterns.STRIDED_READ_PATTERN: OpFlags.strided_read_flag,
    OpPatterns.STRIDED_WRITE_PATTERN: OpFlags.strided_write_flag,
    OpSubPatterns.CMPSEL_PATTERN: OpFlags.cmpsel_flag,
    OpPatterns.CONV3D_BACKPROP_FILTER_PATTERN:
    OpFlags.conv3d_backprop_filter_flag
}


class OpPatternRecognizer:
    """Pattern recognizer(distributor) core functions, Usually you shouldn't modify them"""
    @staticmethod
    def get_pattern(flags: dict, input_tensors: list,
                    output_tensors: list, tensor_list: list, tensor_map: dict):
        """Match pattern based on defined rules and received compute statistics"""
        # If nothing matches, default pattern would be opaque pattern
        matched_pattern = OpPatternRecognizer._apply_pattern_rules(flags,
                                                                   input_tensors,
                                                                   output_tensors,
                                                                   tensor_list,
                                                                   tensor_map)
        matched_subpattern = OpPatternRecognizer.apply_subpattern_rules(flags,
                                                                        input_tensors,
                                                                        output_tensors,
                                                                        tensor_list,
                                                                        tensor_map)
        matched_special_op = OpPatternRecognizer.apply_spec_rules(flags,
                                                                  input_tensors,
                                                                  output_tensors,
                                                                  tensor_list,
                                                                  tensor_map)
        return matched_pattern, matched_subpattern, matched_special_op

    @staticmethod
    def apply_spec_rules(flags, input_tensors, output_tensors, tensor_list, tensor_map):
        """Subroutine for applying special type rules"""
        matched_special_op = None
        for rule, target_pattern in OP_SPEC_RULES.items():
            if matched_special_op is not None:
                break
            # One rule for multiple patterns
            if isinstance(target_pattern, tuple):
                for pattern in target_pattern:
                    if rule(flags, pattern, SIMPLE_MAPPING, input_tensors,
                            output_tensors, tensor_list, tensor_map):
                        matched_special_op = pattern
                        break
            elif rule(flags, input_tensors, output_tensors,
                      tensor_list, tensor_map) and isinstance(target_pattern, OpSpecTypes):
                # One rule for one pattern
                matched_special_op = OP_SPEC_RULES[rule]
                break
            elif not isinstance(target_pattern, OpSpecTypes):
                raise ValueError("Wrong Subpattern rule dictionary format: " +
                                 "SpecType expected but received " + str(type(target_pattern)))
        return matched_special_op

    @staticmethod
    def apply_subpattern_rules(flags, input_tensors, output_tensors, tensor_list, tensor_map):
        """Subroutine for applying subpattern rules"""
        matched_subpattern = None
        for rule, target_pattern in OP_SUBPATTERN_RULES.items():
            if matched_subpattern is not None:
                break
            # One rule for multiple patterns
            if isinstance(target_pattern, tuple):
                for pattern in target_pattern:
                    if rule(flags, pattern, SIMPLE_MAPPING, input_tensors,
                            output_tensors, tensor_list, tensor_map):
                        matched_subpattern = pattern
                        break
            elif rule(flags, input_tensors, output_tensors,
                      tensor_list, tensor_map) and isinstance(target_pattern, OpSubPatterns):
                # One rule for one pattern
                matched_subpattern = OP_SUBPATTERN_RULES[rule]
                break
            elif not isinstance(target_pattern, OpSubPatterns):
                raise ValueError("Wrong Subpattern rule dictionary format: " +
                                 "SubPattern expected but received " + str(type(target_pattern)))
        return matched_subpattern

    @staticmethod
    def _apply_pattern_rules(flags, input_tensors, output_tensors, tensor_list, tensor_map):
        """Subroutine for applying pattern rules"""
        matched_pattern = OpPatterns.OPAQUE_PATTERN
        for rule, target_pattern in OP_PATTERN_RULES.items():
            if matched_pattern != OpPatterns.OPAQUE_PATTERN:
                break
            # One rule for multiple patterns
            if isinstance(target_pattern, tuple):
                for pattern in target_pattern:
                    if rule(flags, pattern, SIMPLE_MAPPING, input_tensors,
                            output_tensors, tensor_list, tensor_map):
                        matched_pattern = pattern
                        break
            elif rule(flags, input_tensors, output_tensors,
                      tensor_list, tensor_map) and isinstance(target_pattern, OpPatterns):
                # One rule for one pattern
                matched_pattern = OP_PATTERN_RULES[rule]
                break
            elif not isinstance(target_pattern, OpPatterns):
                raise ValueError("Wrong Subpattern rule dictionary format: " +
                                 "Pattern expected but received " + str(type(target_pattern)))
        return matched_pattern

    @classmethod
    def get_compute_statistics(cls, all_tensors):
        """Get statistics of all compute tensors"""
        flags = {
            "total": {"count": 0, "tensors": []},
            None: {"count": 0, "tensors": []},
            "unknown": {"count": 0, "tensors": []},
        }
        for index in OpFlags:
            index = index.value
            flags[index] = {"count": 0, "tensors": []}
        for tensor in all_tensors:
            flag = tensor.op.tag
            cls.count_flag(flags, flag, tensor)
        return flags

    @classmethod
    def count_flag(cls, flags: dict, flag: str, current_tensor: tvm.tensor.Tensor) -> None:
        """Helper function for counting compute tensor flag"""
        # Always add 1 count for total_flag when there is a flag
        cls.add_flag(flags, "total", current_tensor)
        found_flag = False
        # Add 1 count for corresponding flag
        for key in flags:
            if flag is None or key is None:
                continue
            if flag.find(key) >= 0:
                found_flag = True
                cls.add_flag(flags, key, current_tensor)
        # When there isn't a match, if flag is empty, add 1 for None, or add 1 for unknown
        if not found_flag:
            if flag is None or flag == "":
                cls.add_flag(flags, None, current_tensor)
            else:
                cls.add_flag(flags, "unknown", current_tensor)

    @staticmethod
    def add_flag(flags, flag, tensor):
        """Utility function for adding flag counter"""
        flags[flag]["count"] += 1
        if tensor not in flags[flag]["tensors"]:
            flags[flag]["tensors"].append(tensor)
