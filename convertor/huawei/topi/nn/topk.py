#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

topk
"""

from __future__ import absolute_import as _abs

from te import platform as cce
from te import tvm


def topk_ir(input_tensor, output, indices, k, need_sorted):
    ib = tvm.ir_builder.create()
    p_in = ib.buffer_ptr(input_tensor)
    p_out = ib.buffer_ptr(output)
    p_idx = ib.buffer_ptr(indices)

    n_dim = len(input_tensor.shape)
    last_dim = input_tensor.shape[n_dim - 1]
    size = 1
    for dim in input_tensor.shape:
        size = size*dim
    loop_cnt = size/last_dim

    tmp_val_array = ib.allocate(output.dtype, (last_dim,), name="tmp_val_array",
                                scope=cce.scope_aicpu)
    max_val = ib.allocate(output.dtype, (1,), name="max_val", scope=cce.scope_aicpu)

    tmp_idx_array = ib.allocate("int32", (last_dim,), name="tmp_idx_array", scope=cce.scope_aicpu)
    max_idx = ib.allocate("int32", (1,), name="max_idx", scope=cce.scope_aicpu)

    swp = ib.allocate("int32", (1,), name="swp", scope=cce.scope_aicpu)

    with ib.for_range(0, loop_cnt, for_type="serial", name="i") as i:
        offset = i*last_dim
        offset_k = i*k
        with ib.for_range(0, last_dim, for_type="serial", name="j") as j:
            tmp_idx_array[j] = j
            tmp_val_array[j] = p_in[offset + j]
        with ib.for_range(0, k, for_type="serial", name="m") as m:
            max_val[0] = tmp_val_array[m]
            max_idx[0] = tmp_idx_array[m]
            swp[0] = m
            with ib.for_range(0, last_dim - m, for_type="serial", name="n") as n:
                with ib.if_scope(tmp_val_array[n + m] > max_val[0]):
                    max_val[0] = tmp_val_array[n + m]
                    max_idx[0] = tmp_idx_array[n + m]
                    swp[0] = n + m
                with ib.if_scope(tvm.all(tmp_val_array[n + m] == max_val[0],
                                         max_idx[0] > tmp_idx_array[n + m])):
                    max_val[0] = tmp_val_array[n + m]
                    max_idx[0] = tmp_idx_array[n + m]
                    swp[0] = n + m
            tmp_val_array[swp[0]] = tmp_val_array[m]
            tmp_val_array[m] = max_val[0]

            tmp_idx_array[swp[0]] = tmp_idx_array[m]
            tmp_idx_array[m] = max_idx[0]
            with ib.if_scope(tvm.const(need_sorted == True,'uint1')):
                p_out[offset_k + m] = tmp_val_array[m]
                p_idx[offset_k + m] = tmp_idx_array[m]
            with ib.else_scope():
                p_out[offset_k + k - m - 1] = tmp_val_array[m]
                p_idx[offset_k + k - m - 1] = tmp_idx_array[m]
    return ib.get()


@tvm.tag_scope(tag="topk")
def compute_topk_cce(input_tensor, k, need_sorted=True):
    """cce topk arithmetic operator

    Parameters
    ----------
    input_tensor : tvm.Tensor
        n-D, with shape [a1, a2, ..., an]

    k : int
		0-an. K result values from last dimension

    need_sorted: bool, optional
        whether to sort result values in descending order

    Returns
    -------
    values : tvm.Tensor
        n-D, with shape [a1, a2, ..., k], The k largest elements along each last dimensional slice.
    indices : tvm.Tensor
        n-D, with th same shape with values, the indices of values.
    """
    shape = tuple(input_tensor.shape)
    shape_output = shape[:-1] + (k,)

    values, indices = tvm.extern([shape_output, shape_output], [input_tensor],
                                 lambda ins, outs: topk_ir(ins[0], output=outs[0], indices=outs[1],
                                                           k=k, need_sorted=need_sorted),
                                 dtype=[input_tensor.dtype, "int32"], name="topk")

    return values, indices
