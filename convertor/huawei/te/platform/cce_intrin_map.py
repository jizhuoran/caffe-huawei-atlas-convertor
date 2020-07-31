#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Runtime function related hooks
"""


# pylint: disable=useless-object-inheritance, too-few-public-methods
class OpIntrinInfo(object):
    """
    OpIntrinInfo
    """

    def __init__(self, op_index, op_tag, intrin=None):
        self.op_index = op_index
        self.op_tag = op_tag
        if intrin is None:
            self.intrin = op_tag
        else:
            self.intrin = intrin

    def __str__(self):
        return "OpIntrinInfo, self.op_tag: %s, op_index: %s, intrin: %s!" % (
            self.op_tag, self.op_index, self.intrin)


OP_INTRIN_INFO_LIST = [
    OpIntrinInfo(0, "mem_copy", "dma_copy"),
    OpIntrinInfo(1, "elewise_single_log", "vector_ln"),
    OpIntrinInfo(2, "elewise_single_exp", "vector_exp"),
    OpIntrinInfo(3, "elewise_single_rec", "vector_rec"),
    OpIntrinInfo(4, "elewise_single_VS_add", "vector_adds"),
    OpIntrinInfo(5, "elewise_single_VS_mul", "vector_muls"),
    OpIntrinInfo(6, "elewise_single_abs", "vector_abs"),
    OpIntrinInfo(7, "elewise_single_relu", "vector_relu"),
    OpIntrinInfo(8, "elewise_single_not", "vector_not"),
    OpIntrinInfo(9, "elewise_single_sqrt", "vector_sqrt"),
    OpIntrinInfo(10, "elewise_binary_add", "vector_add"),
    OpIntrinInfo(11, "elewise_binary_sub", "vector_sub"),
    OpIntrinInfo(12, "elewise_binary_mul", "vector_mul"),
    OpIntrinInfo(13, "elewise_binary_min", "vector_min"),
    OpIntrinInfo(14, "elewise_binary_max", "vector_max"),
    OpIntrinInfo(15, "elewise_binary_or", "vector_or"),
    OpIntrinInfo(16, "elewise_binary_and", "vector_and"),
    OpIntrinInfo(17, "elewise_binary_scalar_axpy", "vector_multiple"),
    OpIntrinInfo(18, "elewise_multiple_mla", "vector_multiple"),
    OpIntrinInfo(19, "elewise_multiple_madd", "vector_multiple"),
    OpIntrinInfo(20, "elewise_multiple_maddrelu", "vector_multiple"),
    OpIntrinInfo(21, "reduce_sum_last", "vector_reduce_sum"),
    OpIntrinInfo(22, "reduce_sum_nist", "vector_reduce_sum"),
    OpIntrinInfo(23, "reduce_min_last", "vector_reduce_min"),
    OpIntrinInfo(24, "reduce_min_nist", "vector_reduce_min"),
    OpIntrinInfo(25, "reduce_max_last", "vector_reduce_max"),
    OpIntrinInfo(26, "reduce_max_nist", "vector_reduce_max"),
    OpIntrinInfo(27, "reduce_prod_last", "reduce_last_axis_reduce_prod"),
    OpIntrinInfo(28, "reduce_prod_nist", "vector_mul"),
    OpIntrinInfo(29, "broadcast_for_tensor", "unified_broadcast"),
    OpIntrinInfo(30, "elewise_single_cast", "vector_conv"),
    OpIntrinInfo(31, "elewise_single_round"),
    OpIntrinInfo(32, "elewise_single_ceil"),
    OpIntrinInfo(33, "elewise_single_floor"),
    OpIntrinInfo(34, "elewise_single_trunc"),
    OpIntrinInfo(35, "segment_sum"),
    OpIntrinInfo(36, "load2d", "dma_copy"),
    OpIntrinInfo(37, "transpose_true", "dma_copy"),
    OpIntrinInfo(38, "matmul", "mad"),
    OpIntrinInfo(39, "set_fmatrix", "set_fmatrix"),
    OpIntrinInfo(40, "im2col", "im2col"),
    OpIntrinInfo(41, "conv_mad", "mad"),
    OpIntrinInfo(42, "out_to_l1", "dma_copy"),
    OpIntrinInfo(43, "vector_dup", "vector_dup"),
    OpIntrinInfo(44, "l1_to_l0", "dma_copy"),
    OpIntrinInfo(45, "mov_backup", "mov_backup"),
    OpIntrinInfo(46, "broadcast", "broadcast"),
    OpIntrinInfo(47, "elewise_binary_div", "vector_div"),
    OpIntrinInfo(48, "elewise_single_VS_max", "vector_maxs"),
    OpIntrinInfo(49, "elewise_single_VS_min", "vector_mins"),
    OpIntrinInfo(50, "elewise_single_rsqrt", "vector_rsqrt"),
    OpIntrinInfo(51, "elewise_binary_vcmpv_gt", "vector_gt"),
    OpIntrinInfo(52, "elewise_binary_vcmpv_ge", "vector_ge"),
    OpIntrinInfo(53, "elewise_binary_vcmpv_lt", "vector_lt"),
    OpIntrinInfo(54, "elewise_binary_vcmpv_le", "vector_le"),
    OpIntrinInfo(55, "elewise_binary_vcmpv_eq", "vector_eq"),
    OpIntrinInfo(56, "elewise_binary_vcmpv_ne", "vector_ne"),
    OpIntrinInfo(57, "vector_auto"),
    OpIntrinInfo(58, "emit_insn_elewise_multiple_sel", "elewise_multiple_sel"),
    OpIntrinInfo(59, "emit_insn_elewise_binary_cmp", "elewise_binary_cmp"),
    OpIntrinInfo(60, "elewise_binary_cmpsel", "vector_cmpsel"),
    OpIntrinInfo(61, "elewise_single_VS_cond"),
    OpIntrinInfo(62, "elewise_binary_logic"),
    OpIntrinInfo(63, "elewise_binary_compare_lt"),
    OpIntrinInfo(64, "elewise_binary_compare_gt"),
    OpIntrinInfo(65, "elewise_single_VS_mul_with_reg_in_quant"),
    OpIntrinInfo(66, "elewise_single_VS_adds_with_reg"),
    OpIntrinInfo(67, "elewise_single_VS_mul_with_reg_sqrt_in_quant"),
    OpIntrinInfo(68, "elewise_single_VS_mul_with_reg"),
    OpIntrinInfo(69, "elewise_single_VS_add_with_reg"),
    OpIntrinInfo(70, "elewise_single_diagonal"),
    OpIntrinInfo(71, "tuple_reduce_sum_nist", "vector_reduce_sum"),
    OpIntrinInfo(72, "tuple_reduce_sum_last", "vector_reduce_sum"),
    OpIntrinInfo(73, "concat", "dma_copy"),
    OpIntrinInfo(74, "strided_slice_d", "dma_copy"),
    OpIntrinInfo(75, "split_com", "dma_copy"),
]


def get_op_intrin_map():
    """
    get_op_intrin_map
    """
    op_intrin_key_index_map = {}
    op_intrin_key_tag_map = {}
    for op_intrin_info in OP_INTRIN_INFO_LIST:
        op_index = op_intrin_info.op_index
        op_tag = op_intrin_info.op_tag
        op_intrin_key_index_map[op_index] = op_intrin_info
        op_intrin_key_tag_map[op_tag] = op_intrin_info
    return op_intrin_key_index_map, op_intrin_key_tag_map


OP_INTRIN_KEY_INDEX, OP_INTRIN_KEY_TAG = get_op_intrin_map()
