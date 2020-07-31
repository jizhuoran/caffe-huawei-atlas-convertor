#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

register the cce buffer info
"""
from __future__ import absolute_import as _abs

from te import tvm

from .cce_conf import cceProduct
from . import cce_params as param

# pylint: disable=invalid-name
# add default product, default value is 200
# get the CceProductParams instance
cur_cce_product_params = cceProduct("1.1.xxx.xxx")


@tvm.register_func("te.cce.cur_buf_params")
def cur_product_params(name):
    """ api for c++ pass to get current product params"""
    ret = cur_cce_product_params.getParams(name)
    return tvm.const(ret, 'int32')


# The memory information for the compiler
# ub 32B alignedn, L1 and L0 2*16^2B aligned
L1_UNIT_BITS = 2*16*16*8
L1_MAX_SIMD_BITS = 2*16*16*8


@tvm.register_func("tvm.info.mem.%s" % param.scope_cbuf)
def mem_info_l1_buffer():
    """
    make node info L1 buffer
    """
    return tvm.make.node("MemoryInfo",
                         unit_bits=L1_UNIT_BITS,
                         max_simd_bits=L1_MAX_SIMD_BITS,
                         max_num_bits=cur_cce_product_params.getParams("L1_Buffer")*8,
                         head_address=tvm.const(0, 'int32'))

@tvm.register_func("tvm.info.mem.%s" % param.scope_cbuf_fusion)
def mem_info_l1_fusion_buffer():
    """
    make node info L1 buffer
    """
    return tvm.make.node("MemoryInfo",
                         unit_bits=L1_UNIT_BITS,
                         max_simd_bits=L1_MAX_SIMD_BITS,
                         max_num_bits=cur_cce_product_params.getParams("L1_Buffer")*8,
                         head_address=tvm.const(0, 'int32'))


UB_UNIT_BITS = 32*8
UB_MAX_SIMD_BITS = 32*8


@tvm.register_func("tvm.info.mem.%s" % param.scope_ubuf)
def mem_info_ub_buffer():
    """
    make node info UB buffer
    """
    return tvm.make.node("MemoryInfo",
                         unit_bits=UB_UNIT_BITS,
                         max_simd_bits=UB_MAX_SIMD_BITS,
                         max_num_bits=cur_cce_product_params.getParams("Unified_Buffer")*8,
                         head_address=tvm.const(0, 'int32'))

# The memory information for the compiler
# smask 32B aligned
SMASK_UNIT_BITS = 32*8
SMASK_MAX_SIMD_BITS = 32*8


@tvm.register_func("tvm.info.mem.%s" % param.scope_smask)
def mem_info_smask_buffer():
    """
    make node info SMASK buffer
    """
    return tvm.make.node("MemoryInfo",
                         unit_bits=SMASK_UNIT_BITS,
                         max_simd_bits=SMASK_MAX_SIMD_BITS,
                         max_num_bits=cur_cce_product_params.getParams("SMASK_Buffer")*8,
                         head_address=tvm.const(0, 'int32'))

L0A_UNIT_BITS = 2*16*16*8
L0A_MAX_SIMD_BITS = 2*16*16*8


@tvm.register_func("tvm.info.mem.%s" % param.scope_ca)
def mem_info_l0a_buffer():
    """
    make node info L0A buffer
    """
    return tvm.make.node("MemoryInfo",
                         unit_bits=L0A_UNIT_BITS,
                         max_simd_bits=L0A_MAX_SIMD_BITS,
                         max_num_bits=cur_cce_product_params.getParams("L0A_Buffer")*8,
                         head_address=tvm.const(0, 'int32'))


L0B_UNIT_BITS = 2*16*16*8
L0B_MAX_SIMD_BITS = 2*16*16*8


@tvm.register_func("tvm.info.mem.%s" % param.scope_cb)
def mem_info_l0b_buffer():
    """
    make node info L0B buffer
    """
    return tvm.make.node("MemoryInfo",
                         unit_bits=L0B_UNIT_BITS,
                         max_simd_bits=L0B_MAX_SIMD_BITS,
                         max_num_bits=cur_cce_product_params.getParams("L0B_Buffer")*8,
                         head_address=tvm.const(0, 'int32'))


L0C_UNIT_BITS = 2*16*16*8
L0C_MAX_SIMD_BITS = 2*16*16*8


@tvm.register_func("tvm.info.mem.%s" % param.scope_cc)
def mem_info_l0c_buffer():
    """
    make node info L0C buffer
    """
    return tvm.make.node("MemoryInfo",
                         unit_bits=L0C_UNIT_BITS,
                         max_simd_bits=L0C_MAX_SIMD_BITS,
                         max_num_bits=cur_cce_product_params.getParams("L0C_Buffer")*8,
                         head_address=tvm.const(0, 'int32'))


REG_UNIT_BITS = 16
REG_MAX_SIMD_BITS = 64
REG_MAX_NUM_BITS = 64*3200


@tvm.register_func("tvm.info.mem.%s" % param.scope_reg)
def mem_info_reg_buffer():
    """
    make node info Reg buffer
    """
    return tvm.make.node("MemoryInfo",
                         unit_bits=REG_UNIT_BITS,
                         max_simd_bits=REG_MAX_SIMD_BITS,
                         max_num_bits=REG_MAX_NUM_BITS,
                         head_address=tvm.const(0, 'int32'))


AICPU_UNIT_BITS = 16
AICPU_MAX_SIMD_BITS = 64
AICPU_MAX_NUM_BITS = 16*1024*1024 # AICPU stack memory limit is 2M


@tvm.register_func("tvm.info.mem.%s" % param.scope_aicpu)
def mem_info_ai_cpu():
    """
    make node info Ai_CPU
    """
    return tvm.make.node("MemoryInfo",
                         unit_bits=AICPU_UNIT_BITS,
                         max_simd_bits=AICPU_MAX_SIMD_BITS,
                         max_num_bits=AICPU_MAX_NUM_BITS,
                         head_address=tvm.const(0, 'int32'))
