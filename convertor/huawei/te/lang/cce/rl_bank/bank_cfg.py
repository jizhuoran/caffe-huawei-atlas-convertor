#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Define some config for rl bank
"""
from te import platform as cceconf

AXIS_NUM = 8

DTYPE_INDEX = {
    'int8': 1,
    'uint8': 2,
    'int16': 3,
    'uint16': 4,
    'int32': 5,
    'uint32': 6,
    'int64': 7,
    'uint64': 8,
    'float16': 9,
    'bfloat16': 10,
    'float32': 11,
    'float64': 12,
    'bool': 13,
    'unknown': 0
}

TAG_INDEX = {
    '': 0,
    'mem_copy': 0,
    'elewise_single_log': 1,
    'elewise_single_exp': 2,
    'elewise_single_rec': 3,
    'elewise_single_VS_add': 4,
    'elewise_single_VS_mul': 5,
    'elewise_single_abs': 6,
    'elewise_single_relu': 7,
    'elewise_single_not': 8,
    'elewise_single_sqrt': 9,
    'elewise_binary_add': 10,
    'elewise_binary_sub': 11,
    'elewise_binary_mul': 12,
    'elewise_binary_min': 13,
    'elewise_binary_max': 14,
    'elewise_binary_or': 15,
    'elewise_binary_and': 16,
    'elewise_binary_scalar_axpy': 17,
    'elewise_multiple_mla': 18,
    'elewise_multiple_madd': 19,
    'elewise_multiple_maddrelu': 20,
    'reduce_sum': 21,
    'reduce_sum_nist': 22,
    'reduce_min': 23,
    'reduce_min_nist': 24,
    'reduce_max': 25,
    'reduce_max_nist': 26,
    'reduce_prod': 27,
    'reduce_prod_nist': 28,
    'broadcast_for_tensor': 29,
    'elewise_single_cast': 30,
    'elewise_single_round': 31,
    'elewise_single_ceil': 32,
    'elewise_single_floor': 33,
    'elewise_single_trunc': 34,
    'segment_sum': 35,
    'load2d': 36,
    'transpose_true': 37,
    'matmul': 38,
    'set_fmatrix': 39,
    'im2col': 40,
    'conv_mad': 41,
    'out_to_l1': 42,
    'vector_dup': 43,
    'l1_to_l0': 44,
    'mov_backup': 45,
    'broadcast': 46,
    'elewise_binary_div': 47,
    'elewise_single_VS_max': 48,
    'elewise_single_VS_min': 49,
    'elewise_single_rsqrt': 50,
    'elewise_binary_vcmpv_gt': 51,
    'elewise_binary_vcmpv_ge': 52,
    'elewise_binary_vcmpv_lt': 53,
    'elewise_binary_vcmpv_le': 54,
    'elewise_binary_vcmpv_eq': 55,
    'elewise_binary_vcmpv_ne': 56,
    'vector_auto': 57,
    'emit_insn_elewise_multiple_sel': 58,
    'emit_insn_elewise_binary_cmp': 59,
    'elewise_binary_cmpsel': 60,
    'elewise_single_VS_cond': 61,
    'elewise_binary_logic': 62,
    'elewise_binary_compare_lt': 63,
    'elewise_binary_compare_gt': 64,
    'elewise_single_VS_mul_with_reg_in_quant': 65,
    'elewise_single_VS_adds_with_reg': 66,
    'elewise_single_VS_mul_with_reg_sqrt_in_quant': 67,
    'elewise_single_VS_mul_with_reg': 68,
    'elewise_single_VS_add_with_reg': 69,
    'elewise_single_diagonal': 70,
    'tuple_reduce_sum': 71,
    'concat': 72,
    'elewise_empty_intrin': 73,
    'strided_slice_d': 74,
    'split_com': 75
}

INTRIN_MAP = {
    0: 'dma_copy',
    1: 'vector_ln',
    2: 'vector_exp',
    3: 'elewise_single_rec',
    4: 'vector_adds',
    5: 'vector_muls',
    6: 'vector_abs',
    7: 'vector_relu',
    8: 'vector_not',
    9: 'vector_sqrt',
    10: 'vector_add',
    11: 'vector_sub',
    12: 'vector_mul',
    13: 'vector_min',
    14: 'vector_max',
    15: 'vector_or',
    16: 'vector_and',
    17: 'vector_multiple',
    18: 'phony_insn',
    19: 'vector_mul_with_broadcast',
    20: 'vector_add_with_broadcast',
    21: 'vector_reduce_sum',
    22: 'replace_output',
    23: 'vector_reduce_min',
    24: 'reuse_input',
    25: 'vector_reduce_max',
    26: 'vector_mul_with_broadcast_enhance',
    27: 'reduce_last_axis_reduce_prod',
    28: 'vector_mul',
    29: 'broadcast_for_tensor',
    30: 'vector_conv',
    31: 'elewise_single_round',
    32: 'elewise_single_ceil',
    33: 'elewise_single_floor',
    34: 'elewise_single_trunc',
    35: 'segment_sum',
    36: 'vector_add_with_broadcast_enhance',
    37: 'vector_rec',
    38: 'mad',
    39: 'set_fmatrix',
    40: 'im2col',
    41: 'mad',
    42: 'dma_copy',
    43: 'vector_dup',
    44: 'dma_copy',
    45: 'mov_backup',
    46: 'broadcast',
    47: 'vector_div',
    48: 'vector_maxs',
    49: 'vector_mins',
    50: 'vector_rsqrt',
    51: 'vector_gt',
    52: 'vector_ge',
    53: 'vector_lt',
    54: 'vector_le',
    55: 'vector_eq',
    56: 'vector_ne',
    57: 'vector_auto',
    58: 'elewise_multiple_sel',
    59: 'elewise_binary_cmp',
    60: 'vector_cmpsel',
    61: 'elewise_single_VS_cond',
    62: 'elewise_binary_logic',
    63: 'elewise_binary_compare_lt',
    64: 'elewise_binary_compare_gt',
    65: 'elewise_single_VS_mul_with_reg_in_quant',
    66: 'elewise_single_VS_adds_with_reg',
    67: 'elewise_single_VS_mul_with_reg_sqrt_in_quant',
    68: 'elewise_single_VS_mul_with_reg',
    69: 'elewise_single_VS_add_with_reg',
    70: 'elewise_single_diagonal',
    71: 'vector_dichotomy_add_for_bn_reduce',
    72: 'broadcast_for_tensor_opt',
    73: 'vector_dichotomy_add',
    74: 'unified_broadcast',
    75: 'vector_sub_with_broadcast_enhance'
}

PRIMITIVE_DICT = {
    0: 'cache_read',
    1: 'cache_write',
    2: 'double_buffer',
    3: 'compute_inline',
    4: 'get_axis',
    5: 'get_reduce_axis',
    6: 'split',
    7: 'split_nparts',
    8: 'reorder',
    9: 'compute_at',
    10: 'bind',
    11: 'emit_insn',
    12: 'broadcast_axis_offset',
    13: 'storage_align',
    14: 'cce_special',
    15: 'fuse',
    16: 'pragma',
    17: 'rfactor',
    18: 'set_scope',
}

SCOPE_DICT = {
    0: cceconf.scope_gm,
    1: cceconf.scope_ubuf,
    2: cceconf.scope_cbuf,
    3: cceconf.scope_ca,
    4: cceconf.scope_cb,
    5: cceconf.scope_cc,
    6: cceconf.scope_reg,
    7: cceconf.scope_aicpu,
    8: "",
}

MODE_RUNTIME = "runtime"
MODE_OFFLINE = "offline"


class ScheduleTarget():  # pylint: disable=too-few-public-methods
    '''
    ScheduleTarget
    '''

    def __init__(self, name, obj, axes):
        self.name = name.strip()
        self.obj = obj
        # reduce axis shoule be after comm aixs
        self.axes = axes


class Axis():  # pylint: disable=too-few-public-methods
    '''
    Axis
    '''

    def __init__(self, name, obj):
        self.name = name.strip()
        self.obj = obj

    def update_name(self, new_name):
        new_name = new_name.strip()
        if isinstance(self.name, str):
            self.name = [self.name, new_name]
        elif isinstance(self.name, list):
            self.name.append(new_name)
        else:
            raise RuntimeError("axis name must be string or list!")
