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

store_to_gm
"""

from functools import reduce as functools_reduce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_build import build_config
from te import platform as cce

PARA_LIST_LEN = 5


# pylint: disable=locally-disabled,unused-argument,unnecessary-lambda
@fusion_manager.register("store_to_gm")
def store_to_gm_compute(input_tensor, output_x, kernel_name="store_to_gm"):
    """
    calculating data

    Parameters
    ----------
    input_tensor : TVM tensor
        the input tensor
    output_x : dict
        dict of output_x, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "store_to_gm"

    Returns
    -------
    res
    """
    input_shape = input_tensor.shape
    res_ub = tvm.compute(input_shape, lambda *indice: input_tensor(*indice), name="res_ub")
    res = tvm.compute(input_shape, lambda *indice: res_ub(*indice), name="res")

    return res, res_ub

def _tilling_axis(valid_shape, input_dtype):
    ub_size_bytes = cce.CceProductParams().getParams("Unified_Buffer") - 32
    dtype_bytes_size = cce.cce_intrin.get_bit_len(input_dtype) // 8

    total_ele = ub_size_bytes // dtype_bytes_size
    split_axis = 0
    split_factor = 1

    for i, _ in enumerate(valid_shape):
        ele_cnt = int(functools_reduce(lambda x, y: x*y, valid_shape[i:]))
        if ele_cnt <= total_ele:
            split_axis = i - 1
            split_factor = total_ele // ele_cnt
            break
        elif i == len(valid_shape) - 1:
            split_axis = i
            split_factor = total_ele
            break

    if split_axis < 0:
        split_axis = 0
        split_factor = valid_shape[0]

    return split_axis, split_factor


def store_to_gm(input_x, output_x, kernel_name="store_to_gm"):
    """
    copy data from l1 to ddr (l1 --> ub --> ddr)

    Parameters
    ----------
    input_x : TVM tensor
        the input tensor
    output_x : dict
        dict of output_x, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "store_to_gm"

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype")

    input_tensor = tvm.placeholder(input_shape,
                                   name="input_tensor",
                                   dtype=input_dtype)

    res, res_ub = store_to_gm_compute(input_tensor, output_x, kernel_name=kernel_name)
    sch = tvm.create_schedule([res.op])

    split_axis, split_factor = _tilling_axis(input_shape, input_dtype)
    axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis], factor=split_factor)

    sch[res_ub].compute_at(sch[res], axis_outer)
    sch[input_tensor].set_scope(cce.scope_cbuf_fusion)
    sch[res_ub].set_scope(cce.scope_ubuf)

    sch[res_ub].emit_insn(res_ub.op.axis[split_axis], 'dma_copy')
    sch[res].emit_insn(axis_inner, 'dma_copy')

    tensor_list = [input_tensor, res]

    with build_config:
        tvm.build(sch, tensor_list, "cce", name=kernel_name)



