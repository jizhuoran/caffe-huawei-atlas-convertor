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

unsorted_segment_min_d
"""
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util

#General limitation of the size for input shape
SHAPE_SIZE_LIMIT = 2**30
#block length in number
BLOCK_LENGTH = 32
#max ub size
UB_SIZE_MAX = 293952
# the max num of each axis of shape
MAX_SHAPE_NUM = 1000000


# pylint: disable=unused-argument,invalid-name
def check_supported(x,
                    segment_ids,
                    y,
                    num_segments,
                    kernel_name="unsorted_segment_min_d"):
    """
    fusion pass test if num_segments is int32
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    segment_ids_shape = segment_ids.get("shape")
    segment_ids_dtype = segment_ids.get("dtype")
    check_list = ("float16", "float32", "int32")
    if dtype.lower() not in check_list:
        return False
    util.check_dtype_rule(dtype, check_list)
    check_list_ids = ("int32")
    if segment_ids_dtype.lower() not in check_list_ids:
        return False
    if num_segments <= 0:
        return False
    first_shape = int(shape[0])
    ids_length = int(segment_ids_shape[0])
    if first_shape != ids_length:
        return False
    total_ub_size = (num_segments + first_shape)*BLOCK_LENGTH + (
        (BLOCK_LENGTH // 2 - first_shape %
         (BLOCK_LENGTH // 4)) + first_shape)*(BLOCK_LENGTH // 8)
    if total_ub_size > UB_SIZE_MAX // 2:
        return False
    return True

# pylint: disable=unused-argument,invalid-name,no-member
@fusion_manager.register("unsorted_segment_min_d")
def unsorted_segment_min_d_compute(x,
                                   segment_ids,
                                   y,
                                   num_segments,
                                   kernel_name="unsorted_segment_min_d"):
    """
    compute for unsorted_segment_min_d_compute
    """
    res = te.lang.cce.unsorted_segment_min(x, segment_ids, num_segments)
    return res

# pylint: disable =too-many-locals
@util.check_input_type(dict, dict, dict, int, str)
def unsorted_segment_min_d(x,
                           segment_ids,
                           y,
                           num_segments,
                           kernel_name="unsorted_segment_min_d"):
    """
    Operation and Schedule for unsorted_segment_min_d.

    Parameters
    ----------
    x: dict
        shape and dtype of input.
        dtype only support float16, float32, int32

    segment_ids : dict
        should be the size of the first dimension
        need not cover all values in the full range of valid values
        dtype only support int32

    y: dict
        shape and dtype of output.

    num_segments : the dimension of the first axis of
                   the output tensor(>= max(segment_ids) + 1)

    kernel_name : cce kernel name,
                  default value is "unsorted_segment_min_d"

    Returns
    -------
        None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    segment_ids_shape = segment_ids.get("shape")
    segment_ids_dtype = segment_ids.get("dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape, max_shape_num=MAX_SHAPE_NUM)
    util.check_shape_rule(segment_ids_shape, max_shape_num=MAX_SHAPE_NUM)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
    util.check_shape_size(segment_ids_shape, SHAPE_SIZE_LIMIT)

    check_list = ("float16", "float32", "int32")
    util.check_dtype_rule(dtype, check_list)

    check_list_ids = ("int32",)
    util.check_dtype_rule(segment_ids_dtype, check_list_ids)
    min_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmin", "float32")
    if dtype == "float32" and not min_support:
        raise RuntimeError(
            "Input dtype only support float16 while input dtype is float32")
    if num_segments <= 0:
        raise RuntimeError(
            "unsorted_segment_min_d only support num_segments"
            " greater than 0, while num_segments is %d" %
            (num_segments))

    first_shape = int(shape[0])
    ids_length = int(segment_ids_shape[0])
    if first_shape != ids_length:
        raise RuntimeError(
            "unsorted_segment_min_d only supports inputs[0]"
            "equal to segment_ids_shape[0], while inputs[0] is %d, "
            "segment_ids_shape[0] is %d" %
            (first_shape, ids_length))
    total_ub_size = (num_segments + first_shape)*BLOCK_LENGTH + (
        (BLOCK_LENGTH // 2 - first_shape %
         (BLOCK_LENGTH // 4)) + first_shape)*(BLOCK_LENGTH // 8)
    if total_ub_size > UB_SIZE_MAX // 2:
        raise RuntimeError("unsorted_segment_min_d num_segments=%d,"
                           "shape[0]=%d, greater than UB_SIZE_MAX" %
                           (num_segments, shape[0]))

    dtype = dtype.lower()

    data_inputs = tvm.placeholder(shape, name="data_inputs", dtype=dtype)
    data_segments_id = tvm.placeholder(segment_ids_shape,
                                       name="data_segments_id",
                                       dtype=segment_ids_dtype)
    with tvm.target.cce():
        res = unsorted_segment_min_d_compute(data_inputs, data_segments_id,
                                             y, num_segments, kernel_name)

        sch = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [data_inputs, data_segments_id, res]
    }
    te.lang.cce.cce_build_code(sch, config)
