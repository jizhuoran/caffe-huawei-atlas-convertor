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

split_v_d
"""
from __future__ import absolute_import

from functools import reduce as functools_reduce
import te.lang.cce
from te import tvm
from te.platform.cce_build import build_config
from topi.cce import util
from impl.copy_only import copy_only
from impl.split_last_dim import split_last_dim
from impl.split_last_dim import check_use_last_dim_branch
from impl.split_d import SplitMov
from impl.split_d import SplitLastDimVnv
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

# vtranspose can deal 16*16
TRANSPOSE_SIZE = 256


# pylint: disable=locally-disabled,unused-argument,too-many-arguments,too-many-locals
def split_v_d_compute(input_value,
                      output_data,
                      size_splits,
                      split_dim,
                      num_split,
                      kernel_name="split_v_d"):
    """Split a tensor into len(size_splits) tensors along one dimension.

    Parameters
    ----------
    input_value: TVM tensor
        input tensor.
    output_data: list or tuple
        the list of output tensor.
    size_splits: list or tuple
        a Python list containing the sizes of each output tensor
        along `split_dim`.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        the number of outputs.
    kernel_name: str
        cce kernel name, default value is "split_v_d".

    Returns
    -------
    output_shape_list: list
        the list of output shapes.
    output_tensor_list: list
        the list of output tensors, output tensor type is TVM tensor.
    """
    output_shape_list, output_tensor_list = te.lang.cce.split_compute_com(
        input_value, split_dim, size_splits)

    return output_shape_list, output_tensor_list


def op_select_format(input_value,
                     output_data,
                     size_splits,
                     split_dim,
                     num_split,
                     kernel_name="split_v_d"):
    """Split a tensor into len(size_splits) tensors along one dimension.

    Parameters
    ----------
    input_value: dict
        the dict of input tensor.
    output_data: list or tuple
        the list of output tensor.
    size_splits: list or tuple
        a Python list containing the sizes of each output tensor
        along `split_dim`.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        used to specify the number of outputs.
    kernel_name: str
        cce kernel name, default value is "split_v_d".

    Returns
    -------
    None.
    """
    dtype = input_value.get("dtype").lower()
    if dtype == "int8":
        c0_len = 32
    else:
        c0_len = 16
    output_org_shape_list = []
    output_org_format_list = []
    is_support_5hd = True
    support_ori_format = ["NCHW", "NHWC"]
    input_ori_shape = input_value.get("ori_shape")
    input_ori_format = input_value.get("ori_format")
    split_dim = split_dim % len(input_ori_shape)

    for _, output_dict in enumerate(output_data):
        ori_format = output_dict.get("ori_format").upper()
        ori_shape = output_dict.get("ori_shape")
        output_org_shape_list.append(ori_shape)
        output_org_format_list.append(ori_format)

        if ori_format not in support_ori_format or len(ori_shape) != 4:
            is_support_5hd = False
            break

        # when split_d by N,H,W, support NC1HWC0
        if ori_format[split_dim] != "C":
            break

        # when split_d by C, but output size not C0 align donot support NC1HWC0
        if ori_format == "NCHW" and ori_shape[1] % c0_len != 0:
            is_support_5hd = False
            break

        if ori_format == "NHWC" and ori_shape[3] % c0_len != 0:
            is_support_5hd = False
            break

    is_support_nz = False
    if input_ori_format[0] == "N" and split_dim == 0:
        is_support_nz = True

    dtype_base = [
        "float16", "float", "int32", "int8", "int16", "int64", "uint8",
        "uint16", "uint32", "uint64"
    ]
    dtype_5hd = [
        "float16", "float", "int32", "int8", "int16", "uint16", "uint32"
    ]
    dtype_base_out = dtype_base.copy()
    format_base_out = ["ND"] * len(dtype_base)

    if is_support_5hd:
        dtype_base_out = dtype_base_out + dtype_5hd
        format_base_out = format_base_out + ["NC1HWC0"] * len(dtype_5hd)

    if is_support_nz:
        dtype_base_out = dtype_base_out + dtype_base
        format_base_out = format_base_out + ["FRACTAL_NZ"] * len(dtype_base)

    dtype_str = ','.join(dtype_base_out)
    format_str = ','.join(format_base_out)

    input0 = gen_param(
        classify="input0", name="x", datatype=dtype_str, format=format_str)
    output0 = gen_param(
        classify="output0", name="y", datatype=dtype_str, format=format_str)
    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


# pylint: disable=locally-disabled,too-many-branches,too-many-statements
@util.check_input_type(dict, (list, tuple), (list, tuple), int, int, str)
def split_v_d(input_value,
              output_data,
              size_splits,
              split_dim,
              num_split,
              kernel_name="split_v_d"):
    """Split a tensor into len(size_splits) tensors along one dimension.

    Parameters
    ----------
    input_value: dict
        the dict of input tensor.
    output_data: list or tuple
        the list of output tensor.
    size_splits: list or tuple
        a Python list containing the sizes of each output tensor
        along `split_dim`.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        used to specify the number of outputs.
    kernel_name: str
        cce kernel name, default value is "split_v_d".

    Returns
    -------
    None.
    """
    input_format = input_value.get("format")
    ori_format = input_value.get("ori_format")
    if input_format == "NC1HWC0":
        split_dim = util.axis_transfrom_5d(split_dim, ori_format)
        if split_dim == 1:
            size_splits = list(size_splits)
            size_splits = [size // 16 for size in size_splits]

    shape = input_value.get("shape")
    dtype = input_value.get("dtype")
    dtype_lower = dtype.lower()
    check_list = ("int8", "int16", "int32", "int64", "uint8", "uint16",
                  "uint32", "uint64", "float16", "float32")

    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_dtype_rule(dtype_lower, check_list)
    util.check_kernel_name(kernel_name)

    shape_len = len(shape)
    split_dim = util.axis_check(shape_len, split_dim)

    dim = shape[split_dim]
    if len(size_splits) + 1 == num_split or len(size_splits) == 0:
        spilt_list = []
        split_sum = 0
        if len(size_splits) != 0:
            for i, _ in enumerate(size_splits):
                spilt_list.append(size_splits[i])
                split_sum = split_sum + size_splits[i]
            if dim - split_sum > 0:
                spilt_list.append(dim - split_sum)
        else:
            batch = dim / num_split
            for i in range(0, num_split):
                spilt_list.append(int(batch))
        size_splits = spilt_list

    size_sum = 0
    for size in size_splits:
        if size < 1:
            raise RuntimeError(
                "The size (%d) of size_splits must be greater or equal to %d" %
                (size, 1))
        size_sum = size_sum + size
    if size_sum != shape[split_dim]:
        raise RuntimeError(
            "The sum size (%d) of size_splits must be equal to the length of "
            "split_dim (%d)" % (size_sum, shape[split_dim]))
    if len(size_splits) != num_split:
        raise RuntimeError(
            "The length (%d) of size_splits must be equal to num_split(%d)" %
            (len(size_splits), num_split))

    if num_split == 1:
        copy_only(input_value, input_value, kernel_name)
        return

    split_mov = SplitMov(shape, dtype_lower, split_dim, num_split, size_splits,
                         kernel_name)
    new_shape = split_mov.input_shape
    new_split_dim = split_mov.split_dim
    new_size_splits = split_mov.size_splits
    new_output_shapes = split_mov.output_shapes
    input_size = functools_reduce(lambda x, y: x * y, new_shape)
    last_dim_same = True
    input_last_dim = new_output_shapes[0][-1]
    for i, _ in enumerate(new_output_shapes):
        if input_last_dim != new_output_shapes[i][-1]:
            last_dim_same = False
            break

    if dtype_lower == "float16" and new_split_dim == len(new_shape) - 1 and \
            last_dim_same and new_size_splits[0] == 1 and num_split <= 16 \
            and input_size >= TRANSPOSE_SIZE * num_split:
        split_vnc = SplitLastDimVnv(new_shape, dtype_lower, new_output_shapes,
                                    new_split_dim, num_split, kernel_name)
        split_vnc.split_last_dim_vnc_compute()
        return

    if check_use_last_dim_branch(new_shape, dtype_lower, new_split_dim,
                                 num_split, new_size_splits):
        split_last_dim(new_shape, dtype_lower, new_split_dim, num_split,
                       new_size_splits, kernel_name)
        return

    if split_mov.check_whether_use_split_mov():
        split_mov.split_mov_compute()
        return

    data = tvm.placeholder(shape, name="data", dtype=dtype_lower)
    output_shape_list, output_tensor_list = split_v_d_compute(
        data, output_data, size_splits, split_dim, num_split,
        kernel_name)

    sch, build_list = te.lang.cce.split_schedule_com(data, split_dim,
                                                     output_shape_list,
                                                     output_tensor_list)

    with build_config:
        tvm.build(sch, build_list, "cce", name=kernel_name)
