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

segment compute
"""
from te import tvm

from .broadcast_compute import broadcast
from .elewise_compute import __binary_elewise_op
from .util import dtype_check_decorator, shape_to_list, check_input_tensor_shape


@dtype_check_decorator
def unsorted_segment_sum(tensor, segment_ids, num_segments, init_value=0):
    """
    calculate segment sum, return a new tensor which is the sum along segments of a tensor.
    only support float16, int32

    Parameters
    ----------
    tensor : tvm.tensor
        input tensor

    segment_ids : list
        index of each segment

    Returns
    -------
    tensor : segment_sum(tensor , segment_ids)
    """
    return __segment_op(tensor, segment_ids, num_segments, init_value, tensor.dtype, "segment_sum")


@dtype_check_decorator
def unsorted_segment_mean(tensor, segment_ids, num_segments, init_value=0):
    """
    calculate segment mean, return a new tensor which is the mean along segments of a tensor.
    only support float16, int32.

    Parameters
    ----------
    tensor : tvm.tensor
        input tensor

    segment_ids : list
        index of each segment

    Returns
    -------
    tensor : segment_mean(tensor , segment_ids)
    """
    return __segment_op(tensor, segment_ids, num_segments, init_value, tensor.dtype, "segment_mean")


@dtype_check_decorator
def unsorted_segment_prod(tensor, segment_ids, num_segments, init_value=0):
    """
    calculate segment prod, return a new tensor which is the prod along segments of a tensor.
    only support float16, int32.

    Parameters
    ----------
    tensor : tvm.tensor
        input tensor

    segment_ids : list
        index of each segment

    Returns
    -------
    tensor : segment_prod(tensor , segment_ids)
    """
    if isinstance(segment_ids, tvm.tensor.Tensor):
        init_value = 1
        return __segment_tensor_op(tensor, segment_ids, num_segments,
                                   init_value, tensor.dtype, "segmentensor_prod")
    return __segment_op(tensor, segment_ids, num_segments, init_value,
                        tensor.dtype, "segment_prod")


@dtype_check_decorator
def unsorted_segment_min(tensor, segment_ids, num_segments, init_value=0):
    """
    calculate segment min, return a new tensor which is the min along segments of a tensor.
    only support float16, int32

    Parameters
    ----------
    tensor : tvm.tensor
        input tensor

    segment_ids : list
        index of each segment

    Returns
    -------
    tensor : segment_min(tensor , segment_ids)
    """
    if isinstance(segment_ids, tvm.tensor.Tensor):
        return __segment_tensor_op(tensor, segment_ids, num_segments,
                                   init_value, tensor.dtype, "segmentensor_min")
    return __segment_op(tensor, segment_ids, num_segments,
                        init_value, tensor.dtype, "segment_min")


@dtype_check_decorator
def unsorted_segment_max(tensor, segment_ids, num_segments, init_value=0):
    """
    calculate segment max, return a new tensor which is the max along segments of a tensor.
     only support float16, int32

    Parameters
    ----------
    tensor : tvm.tensor
        input tensor

    segment_ids : list
        index of each segment

    Returns
    -------
    tensor : segment_max(tensor , segment_ids)
    """
    return __segment_op(tensor, segment_ids, num_segments, init_value, tensor.dtype, "segment_max")


# pylint: disable=too-many-arguments, unused-argument
def __segment_tensor_op(tensor, segment_ids, num_segments, init_value, output_dtype, segment_op):
    """
    factory method of segment operations
    """
    shape_tensor = shape_to_list(tensor.shape)
    shape_ids = shape_to_list(segment_ids.shape)
    reduce_k = tvm.reduce_axis((0, shape_ids[0]), "reduce_k")
    segment_dtype = ["float16", "float32", "int32"]
    if tensor.dtype not in segment_dtype:
        raise RuntimeError("data not support this dtype")
    if segment_ids.dtype != "int32":
        raise RuntimeError("segment_ids not support this dtype")
    if not isinstance(num_segments, int):
        raise RuntimeError("num_segments should be a scalar and dtype should be int")

    if shape_ids[0] != shape_tensor[0]:
        raise RuntimeError("the rank of segment_ids should be equal to"
                           "the rank of data's first dimension")

    if len(shape_ids) != 1:
        raise RuntimeError("segment_ids's shape should be 1 dimension")

    if not (isinstance(tensor, tvm.tensor.Tensor) and isinstance(segment_ids, tvm.tensor.Tensor)):
        raise RuntimeError("The input type must be tvm.tensor")

    def __segment_select(indices):
        tmp = tvm.select(indices[0] == segment_ids[reduce_k], tensor[(reduce_k,) + indices[1:]],
                         tvm.const(init_value, tensor.dtype))
        return tmp

    if segment_op == "segmentensor_min":
        lambda_func = lambda *indices: tvm.min(__segment_select(indices), axis=[reduce_k])
    elif segment_op == "segmentensor_prod":
        lambda_func = lambda *indices: tvm.prod(__segment_select(indices), axis=[reduce_k])
    else:
        raise RuntimeError("operation %s not support yet" % segment_op)

    name = "data_" + segment_op.split("_")[-2] + '_' + tensor.name.split("_")[-1]
    shape_tensor[0] = num_segments
    with tvm.tag_scope(
            segment_op + "|" + str(num_segments) + "|" + str(init_value)):
        tmp = tvm.compute(shape_tensor, lambda_func, name=name)
    return tmp


# pylint: disable=too-many-locals, too-many-statements, too-many-arguments
def __segment_op(tensor, segment_ids, num_segments, init_value, output_dtype, segment_op):
    """
    factory method of segment operations
    """
    # pylint: disable=consider-merging-isinstance
    if not isinstance(num_segments, int):
        raise RuntimeError("the type of num_segments must be int")
    if not isinstance(init_value, (int, float)):
        raise RuntimeError("the type of init_value must be int or float")

    def __segment_compute(indices):
        """compute_func of unsorted segment mean arithmetic operator

        """
        unique_id = []
        for i in segment_ids:
            if i not in unique_id:
                unique_id.append(i)

        def __compute_outer_dim(i):
            new_segment_id = list(segment_ids)[:]
            if i in unique_id:
                idx = new_segment_id.index(i)
                new_segment_id[idx] = -1
                tmp = tensor[(idx,) + indices[1:]].astype(output_dtype)
                for _ in range(segment_ids.count(i) - 1):
                    new_segment_id[idx] = -1
                    idx = new_segment_id.index(i)
                    if segment_op in ["segment_sum", "segment_mean"]:
                        tmp = tensor[(idx,) + indices[1:]].astype(output_dtype) + tmp
                    elif segment_op == "segment_prod":
                        tmp = tensor[(idx,) + indices[1:]].astype(output_dtype)*tmp
                    elif segment_op == "segment_min":
                        tmp = tvm.min(tensor[(idx,) + indices[1:]].astype(output_dtype), tmp)
                    elif segment_op == "segment_max":
                        tmp = tvm.max(tensor[(idx,) + indices[1:]].astype(output_dtype), tmp)
                    else:
                        raise RuntimeError("operation %s not support yet" % segment_op)
                if segment_op == "segment_mean":
                    tmp = tmp // tvm.const(segment_ids.count(i), output_dtype)
            else:
                tmp = tvm.const(init_value, tensor.dtype)
            return tmp

        res = __compute_outer_dim(0)
        for i in range(num_segments)[1:]:
            res = tvm.select(indices[0] == i, __compute_outer_dim(i), res)
        return res

    shape = shape_to_list(tensor.shape)
    check_input_tensor_shape(shape)

    # check
    if len(segment_ids) != shape[0]:
        raise RuntimeError("the rank of segment_ids should be equal to"
                           "the rank of data's first dimension")
    if (max(segment_ids) + 1) > num_segments:
        raise RuntimeError("num_segments must be larger than max value of segment_ids," \
                           "while num_segments is %d and max value of segment_ids is %d"
                           % (num_segments, max(segment_ids)))

    name = "data_" + segment_op.split("_")[-2] + '_' + tensor.name.split("_")[-1]

    if max(segment_ids) < 0:
        spec_dtype_list = ["int8", "uint8"]

        output_shape = [num_segments] + shape[1:]
        init_value_const = tvm.const(init_value, output_dtype)

        init_value_tmp = broadcast(init_value_const, output_shape)
        input_tmp_zero = __binary_elewise_op(tensor, tensor, "elewise_binary_sub")
        with tvm.tag_scope("broadcast_for_tensor"):
            output_tmp_zero = tvm.compute(output_shape,
                                          lambda *indices: input_tmp_zero[(0,) + indices[1:]],
                                          name="output_tmp")

        if output_dtype in spec_dtype_list:
            with tvm.tag_scope("segment_elewise_special"):
                tmp = tvm.compute(output_shape,
                                  lambda *indices: output_tmp_zero[indices] + init_value_tmp[
                                      indices],
                                  name=name)
        else:
            tmp = __binary_elewise_op(output_tmp_zero, init_value_tmp, "elewise_binary_add")
    else:
        lambda_func = lambda *indices: __segment_compute(indices)
        shape[0] = num_segments
        str_segment_ids = ",".join([str(i) for i in segment_ids])
        with tvm.tag_scope(
                segment_op + "|" + str_segment_ids + "|"
                + str(num_segments) + "|" + str(init_value)):
            tmp = tvm.compute(shape, lambda_func, name=name)

    return tmp
