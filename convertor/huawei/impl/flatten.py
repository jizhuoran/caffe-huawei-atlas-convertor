#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

flatten
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from te import tvm
from te import platform as tbe_platform
from te.platform.cce_build import build_config
from topi.cce import util

# max block num
MAX_BLOCK = 65535


# pylint: disable=too-many-locals
def _tile_axis(data_list, shape, dtype):
    """calculate the tile parameters.
    """
    sch = data_list[0]
    data_ub = data_list[1]
    data_out = data_list[2]
    ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
    dtype_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
    total_cnt = ub_size // dtype_size
    ele_cnt = shape[0]
    axis = 0
    factor = ele_cnt
    if ele_cnt > total_cnt:
        factor = total_cnt
    core_num = ele_cnt // factor
    if core_num <= MAX_BLOCK:
        axis_outer, axis_inner = sch[data_out].split(
            data_out.op.axis[axis], factor=factor)
        if core_num != 1:
            sch[data_out].bind(axis_outer, tvm.thread_axis('blockIdx.x'))
        sch[data_ub].compute_at(sch[data_out], axis_outer)
        sch[data_ub].emit_insn(data_ub.op.axis[axis], 'dma_copy')
        sch[data_out].emit_insn(axis_inner, 'dma_copy')
    else:
        factor_new = 1
        core_num_new = 1
        for i in reversed(list(range(1, MAX_BLOCK))):
            if core_num % i == 0:
                factor_new = core_num // i
                core_num_new = i
                break
        axis_outer, axis_inner = sch[data_out].split(
            data_out.op.axis[axis], factor=factor)
        last_outer, last_inner = sch[data_out].split(
            axis_outer, factor=factor_new)
        if core_num_new != 1:
            sch[data_out].bind(last_outer, tvm.thread_axis('blockIdx.x'))
        sch[data_ub].compute_at(sch[data_out], last_inner)
        sch[data_ub].emit_insn(data_ub.op.axis[axis], 'dma_copy')
        sch[data_out].emit_insn(axis_inner, 'dma_copy')

    return sch


# pylint: disable=invalid-name,unnecessary-lambda,unused-argument
@util.check_input_type(dict, dict, str)
def flatten(x, y, kernel_name="flatten"):
    """return a copy of the tensor collapsed into one dimension.

    Parameters
    ----------
    x : dict
        shape and dtype of input.
    y : dict
        shape and dtype of output.
    kernel_name : str
        kernel name, default value is "flatten"

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    dtype_lower = dtype.lower()
    check_list = ("int8", "int16", "int32", "int64", "uint8", "uint16",
                  "uint32", "uint64", "float16", "float32")

    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_dtype_rule(dtype_lower, check_list)
    util.check_kernel_name(kernel_name)

    size = 1
    for i, _ in enumerate(shape):
        size = size * shape[i]

    shape_new = [size]
    data = tvm.placeholder(shape_new, name="data", dtype=dtype_lower)
    data_ub = tvm.compute(shape_new, lambda *i: data(*i), name='data_ub')
    res = tvm.compute(shape_new, lambda *i: data_ub(*i), name='res')

    sch = tvm.create_schedule(res.op)
    sch[data_ub].set_scope(tbe_platform.scope_ubuf)

    sch_new = _tile_axis([sch, data_ub, res], shape_new, dtype_lower)

    with build_config:
        tvm.build(sch_new, [data, res], "cce", name=kernel_name)
