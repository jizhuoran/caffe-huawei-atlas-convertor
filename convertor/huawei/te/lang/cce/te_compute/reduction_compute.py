#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

reduction compute
"""

# pylint: disable=import-error
from decorator import decorator
from te import tvm
from te.platform import intrinsic_check_support

from .cast_compute import _cast
from .elewise_compute import vmuls
from .util import shape_to_list
from .util import refine_axis
from .util import is_cast_support
from .util import check_input_tensor_shape
from .util import reduce_axis_check
from .util import auto_cast_tensor
from .util import dsl_support_dtype


# pylint: disable=too-many-branches
@decorator
def auto_cast_of_reduce(func, *args, **kwargs):
    '''
    auto cast dectorator.
    Before calling elewise api, check the input tensor is supported by the intr.
    If not supported, casting the input tensor to supported dtype.
    (On condition that the cast type is supported.
    If the cast type is not supported,raising a RuntimeError).
    '''
    intr = func.__name__

    if intr == "sum":
        intr = "reduce_sum"

    def _is_last_axis(shape, axis):
        local_axis = []
        for i in axis:
            new_axis = i
            if i < 0:
                new_axis = i + len(shape)
            local_axis.append(new_axis)

        return len(shape) - 1 in local_axis

    if len(args) == 3 or len(args) == 4:
        if not isinstance(args[0], tvm.tensor.Tensor):
            raise RuntimeError("The first input type must be tvm.tensor")

        raw_tensor = args[0]
        axis = args[1]
        keepdims = args[2]
        priority_flag = False
        if len(args) == 4:
            priority_flag = args[3]

        if isinstance(axis, (tuple, list)):
            axis = axis
        else:
            axis = [axis]

        shape_len = len(raw_tensor.shape)
        axis = reduce_axis_check(shape_len, axis)

        is_last_axis = _is_last_axis(raw_tensor.shape, axis)

        supported_dtypes = dsl_support_dtype(intr)
        if not supported_dtypes:
            raise RuntimeError("%s is not supported!" % intr)

        # 1. reduce_max/min last v100 with priority_flag
        #    or v200, support float32
        vcmax_support_fp32 = intrinsic_check_support("Intrinsic_vcmax",
                                                     "float32")
        support_fp32 = (vcmax_support_fp32 or priority_flag)
        if intr in ("reduce_max", "reduce_min") and \
                is_last_axis and (not support_fp32):
            supported_dtypes = list(set(supported_dtypes) - set(("float32",)))

        # 2. reduce_max/min/sum nlst support int32
        if intr in ("reduce_max", "reduce_min", "reduce_sum") and \
                (not is_last_axis):
            supported_dtypes.append("int32")

        temp_tensor = auto_cast_tensor(raw_tensor, intr, supported_dtypes)


        return func(temp_tensor, axis, keepdims)

    return func(*args, **kwargs)


NAME_INDEX = [0]


# pylint: disable=redefined-builtin
@auto_cast_of_reduce
def sum(raw_tensor, axis, keepdims=False):
    """
    calculate sum of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    axis : int or list
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    Returns
    -------
    res : wrapped_tensor
    """
    return _single_reduce_op(raw_tensor, axis, "reduce_sum", keepdims)


@auto_cast_of_reduce
def reduce_min(raw_tensor, axis, keepdims=False, priority_flag=False):
    """
    calculate reduce_min of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    axis : int or list
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    Returns
    -------
    res : wrapped_tensor
    """
    return _single_reduce_op(raw_tensor, axis, "reduce_min", keepdims)


# pylint: disable=unused-argument
@auto_cast_of_reduce
def reduce_max(raw_tensor, axis, keepdims=False, priority_flag=False):
    """
    calculate reduce_max of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    axis : int or list
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    priority_flag : supported 1(precision) and 0(performance)
    Returns
    -------
    res : wrapped_tensor
    """
    return _single_reduce_op(raw_tensor, axis, "reduce_max", keepdims)


@auto_cast_of_reduce
def reduce_prod(raw_tensor, axis, keepdims=False):
    """
    calculate reduce_prod of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    axis : int
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    Returns
    -------
    res : wrapped_tensor
    """
    return _single_reduce_op(raw_tensor, axis, "reduce_prod", keepdims)


def _single_reduce_op(input_tensor, # pylint: disable=too-many-statements
                      axis, in_op, keepdims=False):
    """
    factory method of single reduce operations
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    """
    if axis is None:
        raise RuntimeError("The axis is None!")

    check_input_tensor_shape(input_tensor)

    def __reduce_compute(data_shape, axis, tensor, func):
        def compute_func(*indice):
            count_indice = 0
            count_reduce = 0
            res_indice = []
            for index in range(len(data_shape)):
                if index not in axis:
                    res_indice.append(indice[count_indice])
                    count_indice += 1
                else:
                    res_indice.append(reduce_axises[count_reduce])
                    count_reduce += 1
                    if keepdims:
                        count_indice += 1

            return func(tensor(*res_indice), axis=reduce_axises)

        reduce_axises = []
        for index, axis_num in enumerate(axis):
            reduce_axises.append(
                tvm.reduce_axis((0, data_shape[axis_num]),
                                name='k' + str(index + 1)))
        res_reshape = []
        for index, shape_l in enumerate(data_shape):
            if index not in axis:
                res_reshape.append(shape_l)
            else:
                if keepdims:
                    res_reshape.append(1)

        # all axis reduce, the dim is 1
        if not res_reshape:
            res_reshape.append(1)

        name = "reduce_" + str(NAME_INDEX[0])
        NAME_INDEX[0] += 1

        reduce_res = tvm.compute(res_reshape, compute_func, name=name)
        return reduce_res

    def __get_reduce_fun(in_op):
        if in_op.lower() == "reduce_min":
            reduce_func = tvm.min
        elif in_op.lower() == "reduce_max":
            reduce_func = tvm.max
        elif in_op.lower() == "reduce_sum":
            reduce_func = tvm.sum
        elif in_op.lower() == "reduce_prod":
            reduce_func = tvm.prod
        else:
            raise RuntimeError("Not Support yet for op %s." % in_op)
        return reduce_func

    reduce_func = __get_reduce_fun(in_op)

    op_tensor = input_tensor
    shape = shape_to_list(op_tensor.shape)
    res_axis = refine_axis(axis, shape)

    if keepdims:
        axis = res_axis[:]
        axis_for_loop = axis.copy()
        for index in axis_for_loop:
            if int(input_tensor.shape[index]) == 1:
                axis.remove(index)
    if not axis:
        res = vmuls(input_tensor, tvm.const(1, dtype=input_tensor.dtype))
        return res

    for i in axis:
        is_last_axis = (i == (len(shape) - 1))
        if is_last_axis:
            break

    with tvm.tag_scope(in_op.lower()):
        res = __reduce_compute(shape, axis, op_tensor, reduce_func)

    return res


@decorator
def auto_cast_of_tuple_reduce(func, *args, **kwargs):
    '''
    auto cast dectorator.
    Before calling elewise api, check the input tensor is supported by the intr.
    If not supported, casting the input tensor to supported dtype.
    (On condition that the cast type is supported.
    If the cast type is not supported,raising a RuntimeError).
    '''
    func_name = func.__name__
    supported_types = ("float16", "float32")
    if func_name != "tuple_sum":
        raise RuntimeError("function name must be tuple_sum")

    def _is_last_axis(shape, axis):
        if isinstance(axis, (tuple, list)):
            local_axis = axis
        else:
            local_axis = [axis]
        return len(shape) - 1 in local_axis

    def _check_tensor(tensor_list):
        if len(tensor_list) != 2:
            raise RuntimeError("Tuple reduce input tensors must be 2.")
        shape1 = shape_to_list(tensor_list[0].shape)
        shape2 = shape_to_list(tensor_list[1].shape)
        if shape1 != shape2:
            raise RuntimeError(
                "Tuple reduce input tensors must have same shape.")

    def _deal_tensor_dtype(raw_tensor, supported_types):
        dtype = raw_tensor.dtype
        if func_name == "tuple_sum" and not _is_last_axis(raw_tensor.shape,
                                                          axis):
            supported_types = supported_types + ("int32",)
        dealed_tensor = raw_tensor
        if dtype not in supported_types:
            if "float32" in supported_types and is_cast_support(dtype,
                                                                "float32"):
                dealed_tensor = _cast(raw_tensor, "float32")
            else:
                dealed_tensor = _cast(raw_tensor, "float16")
        return dealed_tensor

    if len(args) == 3:
        if not isinstance(args[0], (tuple, list)):
            raise RuntimeError("The first input type must be list or tuple")

        raw_tensor_list = args[0]
        axis = args[1]
        keepdims = args[2]

        _check_tensor(raw_tensor_list)

        temp_tensor_list = []
        for raw_tensor in raw_tensor_list:
            temp_tensor = _deal_tensor_dtype(raw_tensor, supported_types)
            temp_tensor_list.append(temp_tensor)

        return func(temp_tensor_list, axis, keepdims)

    return func(*args, **kwargs)


@auto_cast_of_tuple_reduce
def tuple_sum(input_tensor_list, axis, keepdims=False):
    """
    calculate sum of raw_tensor, only support float16
    Parameters
    ----------
    input_tensor_list : wrapped_tensor or tvm.tensor list that each tensor has same reduce operation
    axis : int or list
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    Returns
    -------
    res : wrapped_tensor
    """
    return _tuple_reduce_op(input_tensor_list, axis, "tuple_reduce_sum",
                            keepdims)


def _tuple_reduce_op(input_tensor_list, axis, in_op, keepdims=False):
    """
    factory method of tuple reduce operations
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    """
    if axis is None:
        raise RuntimeError("The axis is None!")

    check_input_tensor_shape(input_tensor_list[0])

    if axis in ((), []):
        res = []
        for tensor in input_tensor_list:
            temp_res = vmuls(tensor, tvm.const(1, dtype=tensor.dtype))
            res.append(temp_res)
        return res

    def __tuple_reduce_compute(data_shape, axis, tensor_list, func):
        def compute_func(*indice):
            """
            compute_func
            """
            count_indice = 0
            count_reduce = 0
            res_indice = []
            for index in range(len(data_shape)):
                if index not in axis:
                    res_indice.append(indice[count_indice])
                    count_indice += 1
                else:
                    res_indice.append(reduce_axises[count_reduce])
                    count_reduce += 1
                    if keepdims:
                        count_indice += 1

            return func(
                (tensor_list[0](*res_indice), tensor_list[1](*res_indice)),
                axis=reduce_axises)

        reduce_axises = []
        for index, axis_num in enumerate(axis):
            reduce_axises.append(
                tvm.reduce_axis((0, data_shape[axis_num]),
                                name='k' + str(index + 1)))
        res_reshape = []
        for index, shape_l in enumerate(data_shape):
            if index not in axis:
                res_reshape.append(shape_l)
            else:
                if keepdims:
                    res_reshape.append(1)

        # all axis reduce, the dim is 1
        if not res_reshape:
            res_reshape.append(1)

        name = "reduce_" + str(NAME_INDEX[0])
        NAME_INDEX[0] += 1

        reduce_res = tvm.compute(res_reshape, compute_func, name=name)
        return reduce_res

    tuple_sum_func = tvm.comm_reducer(lambda x, y: (x[0] + y[0], x[1] + y[1]),
                                      lambda t0, t1: (tvm.const(0, dtype=t0),
                                                      tvm.const(0, dtype=t1)),
                                      name="tuple_sum")

    if in_op.lower() == "tuple_reduce_sum":
        reduce_func = tuple_sum_func
    else:
        raise RuntimeError("Not Support yet for op %s." % in_op)

    op_tensor = input_tensor_list[0]
    shape = shape_to_list(op_tensor.shape)
    res_axis = refine_axis(axis, shape)
    for i in res_axis:
        is_last_axis = (i == (len(shape) - 1))
        if is_last_axis:
            break

    with tvm.tag_scope(in_op.lower()):
        res = __tuple_reduce_compute(shape, res_axis, input_tensor_list,
                                     reduce_func)

    return res
