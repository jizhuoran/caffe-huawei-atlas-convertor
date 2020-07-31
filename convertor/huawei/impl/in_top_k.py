#!/usr/bin/env python
# -*- coding:utf-8 -*-
# pylint: disable=too-many-lines,too-many-statements
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

in_top_k
"""
from te import platform as tbe_platform
from te import tik
from topi.cce import util

SHAPE_SIZE_LIMIT = 1 << 30

# size of useful UB buffer
UB_SIZE_BYTES = \
    tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
# The size of each vector instruction processing data
V_SIZE_BYTES = 256
# max repeat of vector calc
V_MAX_REPEAT = 255
# size of one block
BLOCK_SIZE = 32
FP_MIN = -1


# pylint: disable=unused-argument
def check_supported(predictions,
                    targets,
                    precision,
                    k,
                    kernel_name='in_top_k'):
    """
    support aicpu route
    """
    prediction_dtype = predictions.get("dtype").lower()
    target_dtype = targets.get("dtype").lower()
    if prediction_dtype != "float32":
        return False
    if target_dtype != "int32":
        return False

    return True


# pylint: disable=unused-argument,too-many-locals
@util.check_input_type(dict, dict, dict, int, str)
def in_top_k(predictions, targets, precision, k, kernel_name="in_top_k"):
    """
    the main function of in_top_k

    Parameters
    ----------
    predictions: dict of predictions
                 include keys(shape and dtype)
    targets: dict of targets
             include keys(shape and dtype)
    precision: dict of precision
               reserved output
    k: the k value of top k
    kernel_name: kernel_name
    Returns
    -------
    tik_instance: tik_instance
    """
    predictions_shape = predictions.get("shape")
    target_shape = targets.get("shape")
    prediction_dtype = predictions.get("dtype").lower()
    target_dtype = targets.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_tensor_shape_size(predictions_shape)
    util.check_tensor_shape_size(target_shape)
    util.check_dtype_rule(prediction_dtype, ("float32"))
    util.check_dtype_rule(target_dtype, ("int32"))
    if not tbe_platform.cce_conf.api_check_support("tik.vconv", "f322f16"):
        raise RuntimeError("this product does not supported float32")
    # the predictions is 2-dimensional.
    # the targets is 1-dimensional.
    if len(predictions_shape) != 2:
        raise RuntimeError("predcitions must be 2-dimensional")
    if len(target_shape) != 1:
        raise RuntimeError("targets must be 1-dimensional")
    if predictions_shape[0] != target_shape[0]:
        raise RuntimeError("First dimension of predictions must match "
                           "length of targets.")

    row = predictions_shape[0]
    column = predictions_shape[1]

    # float32 is 4 bytes.
    element_bytes = 4
    coexisting_tensor_num = 5

    # the number of elements on a block.
    block_element = BLOCK_SIZE // element_bytes
    column_aligned = ((column + block_element - 1) // block_element *
                      block_element)
    # the max size of a tensor on ub in the compute schedule of in_top_k.
    max_tensor_size = UB_SIZE_BYTES // coexisting_tensor_num
    # the number of rows UB can deal each time.
    core_row_capicity = max_tensor_size // (column_aligned * element_bytes)

    if k <= 0 or k > column:
        return _in_top_k_special_k(predictions, targets, k, kernel_name)
    if core_row_capicity > 0:
        if row <= BLOCK_SIZE:
            return _in_top_k_single_core(predictions, targets, k, kernel_name)
        if core_row_capicity < BLOCK_SIZE:
            return _in_top_k_mul_core_v2(predictions, targets, k, kernel_name)
        return _in_top_k_mul_core(predictions, targets, k, kernel_name)

    return _in_top_k_tiling_column(predictions, targets, k, kernel_name)


# pylint: disable=too-many-locals
def _in_top_k_special_k(predictions, targets, k, kernel_name):
    """
    the _in_top_k_special_k function of the in_top_k

    Parameters
    ----------
    predictions: dict of predictions
                 include keys(shape and dtype)
    targets: dict of targets
             include keys(shape and dtype)
    precision: dict of precision
               reserved output
    k: the k value of top k
    kernel_name: kernel_name
    Returns
    -------
    tik_instance: tik_instance
    """
    target_shape = targets.get("shape")
    predictions_shape = predictions.get("shape")
    row = target_shape[0]
    prediction_dtype = "float32"
    target_dtype = "int32"
    precision_dtype = "uint8"
    if k <= 0:
        element = 0
    else:
        element = 1
    copy_repeat_times = (row + BLOCK_SIZE - 1) // BLOCK_SIZE

    tik_instance = tik.Tik()

    prediction_tensor = tik_instance.Tensor(prediction_dtype,
                                            predictions_shape,
                                            name="prediction_tensor",
                                            scope=tik.scope_gm)
    target_tensor = tik_instance.Tensor(target_dtype,
                                        target_shape,
                                        name="target_tensor",
                                        scope=tik.scope_gm)

    tensor_output = tik_instance.Tensor(precision_dtype,
                                        target_shape,
                                        name="tensor_output",
                                        scope=tik.scope_gm)

    tensor_ub = tik_instance.Tensor("float16", (BLOCK_SIZE,),
                                    name="tensor_ub",
                                    scope=tik.scope_ubuf)
    tensor_output_ub = tik_instance.Tensor(precision_dtype, (BLOCK_SIZE,),
                                           name="tensor_output_ub",
                                           scope=tik.scope_ubuf)

    prediction_ub = tik_instance.Tensor(precision_dtype, (8,),
                                        name="prediction_ub",
                                        scope=tik.scope_ubuf)
    target_ub = tik_instance.Tensor(target_dtype, (8,),
                                    name="target_ub",
                                    scope=tik.scope_ubuf)

    tik_instance.data_move(prediction_ub, prediction_tensor, 0, 1, 1, 0, 0)

    tik_instance.data_move(target_ub, target_tensor, 0, 1, 1, 0, 0)

    tik_instance.vector_dup(BLOCK_SIZE, tensor_ub, element, 1, 1, 1)
    tik_instance.vconv(BLOCK_SIZE, '', tensor_output_ub, tensor_ub, 1, 1, 1, 8,
                       8)
    with tik_instance.for_range(0, copy_repeat_times) as i:
        tik_instance.data_move(tensor_output[i * BLOCK_SIZE], tensor_output_ub,
                               0, 1, 1, 0, 0)
    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[prediction_tensor, target_tensor],
                          outputs=[tensor_output])
    return tik_instance


# pylint: disable=too-many-locals,too-many-branches
def _in_top_k_inter_process(shape_info, tensor, tik_instance, k):
    """
    the _in_top_k_inter_process function

    Parameters
    ----------
    shape_info: dict
                include keys(core_loop, outer_loop, core_nums,
                             row_remainder, row_nums, column,
                             column_aligned, split_rows_nums)
    tensor: dict
             include keys(prediction_tensor and target_tensor)
    tik_instance: tik_instance
    k: the k value of top k
    Returns
    -------
    tensor_output_ub: tensor
    """
    core_loop = shape_info.get("core_loop")
    outer_loop = shape_info.get("outer_loop")
    core_nums = shape_info.get("core_nums")
    row_remainder = shape_info.get("row_remainder")
    row_nums = shape_info.get("row_nums")
    column = shape_info.get("column")
    column_aligned = shape_info.get("column_aligned")
    split_rows_nums = shape_info.get("split_rows_nums")
    prediction_tensor = tensor.get("prediction_tensor")
    target_tensor = tensor.get("target_tensor")

    half_mask_value = 64
    mask_value = 128
    element_bytes = 4
    # 1 float32 = 2 float16
    carry = 2
    block_element = BLOCK_SIZE // element_bytes
    shape = (row_nums, column_aligned)
    prediction_dtype = "float32"
    target_dtype = "int32"
    core_row_num = tik_instance.Scalar("int64")

    # step 0: set some shape value of tensor in UB.
    with tik_instance.if_scope(outer_loop < core_nums - 1):
        core_row_num.set_as(row_nums)
    with tik_instance.else_scope():
        core_row_num.set_as(row_remainder)

    prediction_tensor_ub = tik_instance.Tensor(prediction_dtype,
                                               (row_nums, column_aligned),
                                               name="prediction_tensor_ub",
                                               scope=tik.scope_ubuf)
    row_nums_align = (row_nums + block_element -
                      1) // block_element * block_element
    target_ub = tik_instance.Tensor(target_dtype, (row_nums_align,),
                                    name="target_ub",
                                    scope=tik.scope_ubuf)
    index = core_loop * split_rows_nums + outer_loop * row_nums
    tik_instance.data_move(target_ub, target_tensor[index], 0, 1,
                           (core_row_num + block_element - 1) // block_element,
                           0, 0)

    if column % block_element == 0:
        length = tik_instance.Scalar("int64")
        length.set_as(core_row_num * column_aligned)
        index = (core_loop * split_rows_nums +
                 outer_loop * row_nums) * column_aligned
        tik_instance.data_move(prediction_tensor_ub, prediction_tensor[index],
                               0, 1, length // block_element, 0, 0)
    else:
        index = (core_loop * split_rows_nums + outer_loop * row_nums) * column
        with tik_instance.for_range(0, core_row_num) as i:
            tik_instance.data_move(prediction_tensor_ub[i * column_aligned],
                                   prediction_tensor[index + i * column], 0, 1,
                                   column_aligned // block_element, 0, 0)

        # dirty data set as FP_MIN, for example, A[:,1:15] is target data,
        # then set A[:,15:16] as FP_MIN.
        reg_data = tik_instance.Scalar(prediction_dtype)
        reg_data.set_as(FP_MIN)
        with tik_instance.for_range(0, core_row_num) as i:
            with tik_instance.for_range(0, column_aligned - column) as j:
                prediction_tensor_ub[i, column + j] = reg_data

    # step 1: index the predictions's elements according to the targets.
    # the result is in tensor data_ub.
    tensor_size = row_nums * column_aligned
    data_ub = tik_instance.Tensor(prediction_dtype,
                                  shape,
                                  name="data_ub",
                                  scope=tik.scope_ubuf)

    # set the number of repeat.
    column_reduce_times = int(column_aligned // half_mask_value)
    column_reduce_remainder = int(column_aligned % half_mask_value)

    if column_reduce_times > 0:
        with tik_instance.for_range(0, core_row_num) as i:
            scalar_target = tik_instance.Scalar(target_dtype, "scalar_target")
            scalar_target.set_as(target_ub[i])
            scalar_value = tik_instance.Scalar(prediction_dtype)
            scalar_value.set_as(prediction_tensor_ub[i, scalar_target])
            tik_instance.vector_dup(half_mask_value,
                                    data_ub[i * column_aligned], scalar_value,
                                    column_reduce_times, 1, 8, 0)
            if column_reduce_remainder != 0:
                tik_instance.vector_dup(
                    column_reduce_remainder,
                    data_ub[i * column_aligned +
                            column_reduce_times * half_mask_value],
                    scalar_value, 1, 1, 1, 0)
    else:
        with tik_instance.for_range(0, core_row_num) as i:
            scalar_target = tik_instance.Scalar(target_dtype, "scalar_target")
            scalar_target.set_as(target_ub[i])
            scalar_value = tik_instance.Scalar(prediction_dtype)
            scalar_value.set_as(prediction_tensor_ub[i * column_aligned +
                                                     scalar_target])
            tik_instance.vector_dup(column_reduce_remainder,
                                    data_ub[i * column_aligned], scalar_value,
                                    1, 1, 1, 0)

    # step 2, prediction_tensor subtract data_ub.
    column_remainder = int(column % half_mask_value)
    reduce_times = int(column // half_mask_value)

    repeat_times = int((carry * tensor_size) // mask_value)
    tail_mask = int((carry * tensor_size) % mask_value)

    half_sub = tik_instance.Tensor("float16", (shape[0], shape[1] * carry),
                                   name="half_sub",
                                   scope=tik.scope_ubuf)

    if reduce_times > 0:
        with tik_instance.for_range(0, core_row_num) as i:
            tik_instance.vsub(half_mask_value,
                              prediction_tensor_ub[i * column_aligned],
                              prediction_tensor_ub[i * column_aligned],
                              data_ub[i * column_aligned],
                              reduce_times, 1, 1, 1, 8, 8, 8)
            if column_remainder != 0:
                index = reduce_times * half_mask_value \
                        + i * column_aligned
                tik_instance.vsub(column_remainder, prediction_tensor_ub[index],
                                  prediction_tensor_ub[index], data_ub[index],
                                  1, 1, 1, 1, 8, 8, 8)

    else:
        with tik_instance.for_range(0, core_row_num) as i:
            tik_instance.vsub(column_remainder,
                              prediction_tensor_ub[i * column_aligned],
                              prediction_tensor_ub[i * column_aligned],
                              data_ub[i * column_aligned],
                              1, 1, 1, 1, 8, 8, 8)

    if repeat_times > 0:
        tik_instance.vector_dup(mask_value, half_sub, 0, repeat_times, 1, 8)
    if tail_mask != 0:
        tik_instance.vector_dup(tail_mask, half_sub[repeat_times * mask_value],
                                0, 1, 1, 8)

    if column_reduce_times > 0:
        with tik_instance.for_range(0, core_row_num) as i:
            tik_instance.vconv(half_mask_value, '',
                               half_sub[i * column_aligned * carry],
                               prediction_tensor_ub[i * column_aligned],
                               column_reduce_times, 1, 1, 4, 8)
        if column_reduce_remainder != 0:
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.vconv(
                    column_reduce_remainder, '',
                    half_sub[column_reduce_times * half_mask_value +
                             carry * i * column_aligned],
                    prediction_tensor_ub[i * column_aligned +
                                         column_reduce_times *
                                         half_mask_value], 1, 1, 1, 4, 8)
    else:
        with tik_instance.for_range(0, core_row_num) as i:
            tik_instance.vconv(column_reduce_remainder, '',
                               half_sub[carry * i * column_aligned],
                               prediction_tensor_ub[i * column_aligned], 1, 1,
                               1, 4, 8)

    # step 3, if half_sub[i, j] > 0, then the according element data_sign[i, j]
    # set 1, else 0.
    column_reduce_times = int(column_aligned // mask_value)
    column_reduce_remainder = int(column_aligned % mask_value)

    data_sign = tik_instance.Tensor("float16", (shape[0], (shape[1] * carry)),
                                    name="data_sign",
                                    scope=tik.scope_ubuf)
    data_zeros = tik_instance.Tensor("float16", (mask_value, 1),
                                     name="data_zeros",
                                     scope=tik.scope_ubuf)
    if repeat_times > 0:
        tik_instance.vector_dup(mask_value, data_sign, 0, repeat_times, 1, 8)
    if tail_mask != 0:
        tik_instance.vector_dup(tail_mask,
                                data_sign[repeat_times * mask_value], 0, 1, 1,
                                8)

    zero = tik_instance.Scalar(dtype="float16", name="zeros")
    zero.set_as(0)
    tik_instance.vector_dup(mask_value, data_zeros, zero, 1, 1, 1, 0)
    one = tik_instance.Scalar(dtype="float16", name="ones")
    one.set_as(1)
    data_ones = tik_instance.Tensor("float16", (mask_value, 1),
                                    name="data_ones",
                                    scope=tik.scope_ubuf)
    tik_instance.vector_dup(mask_value, data_ones, one, 1, 1, 1, 0)

    if column_reduce_times > 0:
        with tik_instance.for_range(0, core_row_num) as i:
            with tik_instance.for_range(0, column_reduce_times) as j:
                srcmask = tik_instance.vcmp_gt(
                    mask_value,
                    half_sub[i * column_aligned * carry + mask_value * j],
                    data_zeros, 1, 1)
                tik_instance.vsel(
                    mask_value, 0,
                    data_sign[i * column_aligned * carry + mask_value * j],
                    srcmask, data_ones, data_zeros, 1, 1, 1, 1)
            if column_reduce_remainder != 0:
                srcmask = tik_instance.vcmp_gt(
                    column_reduce_remainder,
                    half_sub[i * column_aligned * carry +
                             column_reduce_times * mask_value], data_zeros, 1,
                    1)
                tik_instance.vsel(
                    column_reduce_remainder, 0,
                    data_sign[i * column_aligned * carry +
                              column_reduce_times * mask_value], srcmask,
                    data_ones, data_zeros, 1, 1, 1, 1)
    else:
        with tik_instance.for_range(0, core_row_num) as i:
            srcmask = tik_instance.vcmp_gt(
                column_reduce_remainder, half_sub[i * column_aligned * carry],
                data_zeros, 1, 1)
            tik_instance.vsel(column_reduce_remainder, 0,
                              data_sign[i * column_aligned * carry], srcmask,
                              data_ones, data_zeros, 1, 1, 1, 1)

    # step 4: do reduce sum in each row of data_sign to count the number
    # larger than the element indexing from the target.
    column_reduce_times = int(column_aligned // half_mask_value)
    column_reduce_remainder = int(column_aligned % half_mask_value)

    if column_reduce_times > 0:
        with tik_instance.for_range(0, core_row_num) as i:
            tik_instance.vconv(half_mask_value, '',
                               prediction_tensor_ub[i * column_aligned],
                               data_sign[i * column_aligned * carry],
                               column_reduce_times, 1, 1, 8, 4)
        if column_reduce_remainder != 0:
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.vconv(
                    column_reduce_remainder, '',
                    prediction_tensor_ub[column_reduce_times * half_mask_value
                                         + i * column_aligned],
                    data_sign[carry * i * column_aligned
                              + column_reduce_times * half_mask_value],
                    1, 1, 1, 8, 4)
    else:
        with tik_instance.for_range(0, core_row_num) as i:
            tik_instance.vconv(column_reduce_remainder, '',
                               prediction_tensor_ub[i * column_aligned],
                               data_sign[carry * i * column_aligned],
                               1, 1, 1, 8, 4)

    if column_reduce_remainder != 0:
        reduce_mask = column_reduce_times + 1
    else:
        reduce_mask = column_reduce_times

    data_bool = tik_instance.Tensor(
        "float32", ((row_nums + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE,),
        name="data_bool",
        scope=tik.scope_ubuf)

    if reduce_mask == 1:
        if column_reduce_remainder != 0:
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.vcadd(column_reduce_remainder, data_bool[i],
                                   prediction_tensor_ub[i * column_aligned],
                                   1, 1, 1, 1, 0)
        else:
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.vcadd(half_mask_value, data_bool[i],
                                   prediction_tensor_ub[i * column_aligned],
                                   1, 1, 1, 1, 0)
    elif reduce_mask <= half_mask_value:
        tensor_sec_reduce = tik_instance.Tensor("float32",
                                                (row_nums, half_mask_value),
                                                name="tensor_sec_reduce",
                                                scope=tik.scope_ubuf)
        zeros_init = tik_instance.Scalar("float32")
        zeros_init.set_as(0)
        if column_reduce_remainder != 0:
            tik_instance.vector_dup(half_mask_value, tensor_sec_reduce,
                                    zeros_init, row_nums, 1, 8, 0)
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.vcadd(half_mask_value,
                                   tensor_sec_reduce[i * half_mask_value],
                                   prediction_tensor_ub[i * column_aligned],
                                   column_reduce_times, 1, 1, 8, 0)
                tik_instance.vcadd(
                    column_reduce_remainder,
                    tensor_sec_reduce[i * half_mask_value
                                      + column_reduce_times],
                    prediction_tensor_ub[i * column_aligned
                                         + column_reduce_times
                                         * half_mask_value],
                    1, 1, 1, 1, 0)
            tik_instance.vcadd(reduce_mask, data_bool, tensor_sec_reduce,
                               row_nums, 1, 1, 8, 0)
        else:
            tik_instance.vector_dup(half_mask_value, tensor_sec_reduce,
                                    zeros_init, row_nums, 1, 8, 0)
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.vcadd(half_mask_value,
                                   tensor_sec_reduce[i * half_mask_value],
                                   prediction_tensor_ub[i * column_aligned],
                                   column_reduce_times, 1, 1, 8, 0)
            tik_instance.vcadd(reduce_mask, data_bool, tensor_sec_reduce,
                               row_nums, 1, 1, 8, 0)
    else:
        tensor_third_reduce = tik_instance.Tensor("float32",
                                                  (row_nums, V_SIZE_BYTES),
                                                  name="tensor_third_reduce",
                                                  scope=tik.scope_ubuf)
        tensor_sec_reduce = tik_instance.Tensor("float32",
                                                (row_nums, half_mask_value),
                                                name="tensor_sec_reduce",
                                                scope=tik.scope_ubuf)

        zeros_init = tik_instance.Scalar("float32")
        zeros_init.set_as(0)

        tik_instance.vector_dup(half_mask_value, tensor_third_reduce,
                                zeros_init, row_nums * 4, 1, 8, 0)
        tik_instance.vector_dup(half_mask_value, tensor_sec_reduce,
                                zeros_init, row_nums, 1, 8, 0)
        if column_reduce_remainder != 0:
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.vcadd(half_mask_value,
                                   tensor_third_reduce[i * V_SIZE_BYTES],
                                   prediction_tensor_ub[i * column_aligned],
                                   column_reduce_times, 1, 1, 8, 0)
                tik_instance.vcadd(
                    column_reduce_remainder,
                    tensor_third_reduce[i * V_SIZE_BYTES + column_reduce_times],
                    prediction_tensor_ub[i * column_aligned +
                                         column_reduce_times * half_mask_value],
                    1, 1, 1, 8, 0)
                tik_instance.vcadd(half_mask_value,
                                   tensor_sec_reduce[i * half_mask_value],
                                   tensor_third_reduce[i * V_SIZE_BYTES],
                                   4, 1, 1, 8, 0)
            tik_instance.vcadd(half_mask_value, data_bool, tensor_sec_reduce,
                               row_nums, 1, 1, 8, 0)
        else:
            with tik_instance.for_range(0, core_row_num) as i:
                tik_instance.vcadd(half_mask_value,
                                   tensor_third_reduce[i * V_SIZE_BYTES],
                                   prediction_tensor_ub[i * column_aligned],
                                   column_reduce_times, 1, 1, 8, 0)
                tik_instance.vcadd(half_mask_value,
                                   tensor_sec_reduce[i * half_mask_value],
                                   tensor_third_reduce[i * V_SIZE_BYTES],
                                   4, 1, 1, 8, 0)
            tik_instance.vcadd(half_mask_value, data_bool, tensor_sec_reduce,
                               row_nums, 1, 1, 8, 0)

    # if data_bool[i] < k, then the tensor_output[i] is true,
    # else is false(represented by 0 and 1).
    data_k = tik_instance.Tensor("float32", ((row_nums + BLOCK_SIZE - 1)
                                             // BLOCK_SIZE * BLOCK_SIZE,),
                                 name="data_k",
                                 scope=tik.scope_ubuf)
    tensor_bool = tik_instance.Tensor(
        "float16", ((row_nums + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE,),
        name="tensor_bool",
        scope=tik.scope_ubuf)

    k_conv_dtype = tik_instance.Scalar("float32")
    k_conv_dtype.set_as(k)

    repeat_times = int(row_nums // half_mask_value)
    repeat_remainder = int(row_nums % half_mask_value)

    if repeat_remainder == 0:
        tik_instance.vector_dup(half_mask_value, data_k, k_conv_dtype,
                                repeat_times, 1, 8, 0)
        tik_instance.vsub(half_mask_value, data_bool, data_k, data_bool,
                          repeat_times, 1, 1, 1, 8, 8, 8)
        tik_instance.vconv(half_mask_value, '', tensor_bool,
                           data_bool, repeat_times, 1, 1, 4, 8)

    else:
        if repeat_times == 0:
            tik_instance.vector_dup(row_nums, data_k, k_conv_dtype,
                                    1, 1, 1, 0)
            tik_instance.vsub(repeat_remainder, data_bool, data_k, data_bool,
                              1, 1, 1, 1, 8, 8, 8)
            tik_instance.vconv(repeat_remainder, '', tensor_bool,
                               data_bool, 1, 1, 1, 8, 8)
        else:
            index = repeat_times * half_mask_value
            tik_instance.vector_dup(half_mask_value, data_k, k_conv_dtype,
                                    repeat_times, 1, 8, 0)
            tik_instance.vector_dup(repeat_remainder, data_k[index],
                                    k_conv_dtype, 1, 1, 1, 0)
            tik_instance.vsub(half_mask_value, data_bool, data_k, data_bool,
                              repeat_times, 1, 1, 1, 8, 8, 8)
            tik_instance.vconv(half_mask_value, '', tensor_bool,
                               data_bool, repeat_times, 1, 1, 4, 8)
            tik_instance.vsub(repeat_remainder, data_bool[index],
                              data_k[index], data_bool[index],
                              1, 1, 1, 1, 8, 8, 8)
            tik_instance.vconv(repeat_remainder, '', tensor_bool[index],
                               data_bool[index], 1, 1, 1, 4, 8)

    tensor_zeros = tik_instance.Tensor("float16", (BLOCK_SIZE,),
                                       name="zeros",
                                       scope=tik.scope_ubuf)
    tensor_ones = tik_instance.Tensor("float16", (BLOCK_SIZE,),
                                      name="ones",
                                      scope=tik.scope_ubuf)

    zeros_uint8 = tik_instance.Scalar("float16")
    zeros_uint8.set_as(0)
    tik_instance.vector_dup(BLOCK_SIZE, tensor_zeros, zeros_uint8, 1, 1, 1, 0)
    ones_uint8 = tik_instance.Scalar("float16")
    ones_uint8.set_as(1)
    tik_instance.vector_dup(BLOCK_SIZE, tensor_ones, ones_uint8, 1, 1, 1, 0)

    data_bool_ub = tik_instance.Tensor(
        "float16", ((row_nums + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE,),
        name="data_bool_ub",
        scope=tik.scope_ubuf)
    tensor_output_ub = tik_instance.Tensor(
        "uint8", ((row_nums + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE,),
        name="tensor_output_ub",
        scope=tik.scope_ubuf)

    cmp_times = int(row_nums // BLOCK_SIZE)
    cmp_rem = int(row_nums % BLOCK_SIZE)
    if cmp_times > 0:
        with tik_instance.for_range(0, cmp_times) as i:
            cmp_mask = tik_instance.vcmp_gt(BLOCK_SIZE,
                                            tensor_bool[i * BLOCK_SIZE],
                                            tensor_zeros, 1, 1)
            tik_instance.vsel(BLOCK_SIZE, 0, data_bool_ub[i * BLOCK_SIZE],
                              cmp_mask, tensor_ones, tensor_zeros, 1, 1, 1, 1)
            tik_instance.vconv(BLOCK_SIZE, '',
                               tensor_output_ub[i * BLOCK_SIZE],
                               data_bool_ub[i * BLOCK_SIZE], 1, 1, 1, 8, 8)

    if cmp_rem != 0:
        cmp_mask = tik_instance.vcmp_gt(cmp_rem,
                                        tensor_bool[cmp_times * BLOCK_SIZE],
                                        tensor_zeros, 1, 1)
        tik_instance.vsel(cmp_rem, 0, data_bool_ub[cmp_times * BLOCK_SIZE],
                          cmp_mask, tensor_ones, tensor_zeros, 1, 1, 1, 1)
        tik_instance.vconv(cmp_rem, '',
                           tensor_output_ub[cmp_times * BLOCK_SIZE],
                           data_bool_ub[cmp_times * BLOCK_SIZE], 1, 1, 1, 8, 8)
    return tensor_output_ub


# pylint: disable=too-many-locals,too-many-branches
def _in_top_k_column_process(shape_info, tensor, tik_instance):
    """
    the _in_top_k_column_process function
    the process of _in_top_k_tiling_column

    Parameters
    ----------
    shape_info: dict
                include keys(core_loop, outer_loop, inner_loop,
                             split_column_nums, column, column_num,
                             column_size, column_size)
    tensor: dict
             include keys(prediction_tensor, target_tensor and scalar_value)
    tik_instance: tik_instance
    Returns
    -------
    data_bool: tensor
    """
    core_loop = shape_info.get("core_loop")
    outer_loop = shape_info.get("outer_loop")
    inner_loop = shape_info.get("inner_loop")
    split_column_nums = shape_info.get("split_column_nums")
    column = shape_info.get("column")
    column_num = shape_info.get("column_num")
    column_size = shape_info.get("column_size")
    prediction_tensor = tensor.get("prediction_tensor")
    scalar_value = tensor.get("scalar_value")
    half_mask_value = 64
    mask_value = 128
    element_bytes = 4
    carry = 2
    block_element = BLOCK_SIZE / element_bytes
    prediction_dtype = "float32"

    # step 1: do vector_dup in data_ub, in this situation, row_nums is 1.
    column_aligned = (column + block_element -
                      1) // block_element * block_element
    shape = (1, column_num)
    prediction_tensor_ub = tik_instance.Tensor(prediction_dtype,
                                               (1, column_num),
                                               name="prediction_tensor_ub",
                                               scope=tik.scope_ubuf)
    index = (core_loop * BLOCK_SIZE + outer_loop) * column
    tik_instance.data_move(prediction_tensor_ub,
                           prediction_tensor[index + inner_loop * column_num],
                           0, 1, int(column_num // block_element), 0, 0)

    # dirty data set as FP_MIN, for example, A[:, 1:15] is target data,
    # then set A[:, 15:16] as FP_MIN.
    with tik_instance.if_scope(inner_loop == split_column_nums - 1):
        reg_data = tik_instance.Scalar(prediction_dtype)
        reg_data.set_as(FP_MIN)
        index = column - inner_loop * column_size
        with tik_instance.for_range(0, column_aligned - column) as j:
            prediction_tensor_ub[0, index + j] = reg_data

    data_ub = tik_instance.Tensor(prediction_dtype, (1, column_num),
                                  name="data_ub",
                                  scope=tik.scope_ubuf)

    # set the number of repeat.
    column_reduce_times = int(column_num // half_mask_value)
    column_reduce_remainder = int(column_num % half_mask_value)

    if column_reduce_times >= 1:
        tik_instance.vector_dup(half_mask_value, data_ub, scalar_value,
                                column_reduce_times, 1, 8, 0)

    if column_reduce_remainder != 0:
        tik_instance.vector_dup(column_reduce_remainder,
                                data_ub[column_reduce_times * half_mask_value],
                                scalar_value, 1, 1, 1, 0)

    # step 2: prediction_tensor_ub subtract data_ub.
    half_repeat_times = int(column_num // half_mask_value)
    half_tail_mask = int(column_num % half_mask_value)

    half_sub = tik_instance.Tensor("float16", (shape[0], shape[1] * carry),
                                   name="half_sub",
                                   scope=tik.scope_ubuf)

    half_remainder_mask = int((column - (split_column_nums - 1) *
                               column_size) % half_mask_value)
    half_repeat = int((column - (split_column_nums - 1) *
                       column_size) // half_mask_value)

    if half_remainder_mask != 0:
        index = half_repeat * half_mask_value
        column_remainder = tik_instance.Scalar("int64")
        with tik_instance.if_scope(inner_loop == split_column_nums - 1):
            if half_repeat_times > 0:
                tik_instance.vsub(half_mask_value, prediction_tensor_ub,
                                  prediction_tensor_ub, data_ub,
                                  half_repeat, 1, 1, 1, 8, 8, 8)
            column_remainder.set_as(half_remainder_mask)
            tik_instance.vsub(column_remainder, prediction_tensor_ub[index],
                              prediction_tensor_ub[index], data_ub[index],
                              1, 1, 1, 1, 8, 8, 8)
    if half_tail_mask != 0:
        with tik_instance.if_scope(inner_loop < split_column_nums - 1):
            if half_repeat_times > 0:
                tik_instance.vsub(half_mask_value, prediction_tensor_ub,
                                  prediction_tensor_ub, data_ub,
                                  half_repeat_times, 1, 1, 1, 8, 8, 8)
            index = half_repeat_times * half_mask_value
            tik_instance.vsub(half_tail_mask, prediction_tensor_ub[index],
                              prediction_tensor_ub[index], data_ub[index],
                              1, 1, 1, 1, 8, 8, 8)

    repeat_times = int((carry * column_num) // mask_value)
    tail_mask = int((carry * column_num) % mask_value)

    if repeat_times > 0:
        tik_instance.vector_dup(mask_value, half_sub, 0, repeat_times, 1, 8)
    if tail_mask != 0:
        tik_instance.vector_dup(tail_mask, half_sub[repeat_times * mask_value],
                                0, 1, 1, 8)

    if column_reduce_times > 0:
        tik_instance.vconv(half_mask_value, '', half_sub, prediction_tensor_ub,
                           column_reduce_times, 1, 1, 4, 8)
        if column_reduce_remainder != 0:
            index = half_mask_value * column_reduce_times
            tik_instance.vconv(
                column_reduce_remainder, '',
                half_sub[index], prediction_tensor_ub[index], 1,
                1, 1, 4, 8)
    else:
        tik_instance.vconv(column_reduce_remainder, '', half_sub,
                           prediction_tensor_ub, 1, 1, 1, 4, 8)

    # step 3: if half_sub[i, j] > 0, then the accoding element data_sign[i, j]
    # set 1, else set 0.
    column_reduce_times = int(column_num // mask_value)
    column_reduce_remainder = int(column_num % mask_value)

    data_sign = tik_instance.Tensor("float16", (shape[0], shape[1] * carry),
                                    name="data_sign",
                                    scope=tik.scope_ubuf)
    data_zeros = tik_instance.Tensor("float16", (mask_value, 1),
                                     name="data_zeros",
                                     scope=tik.scope_ubuf)
    if repeat_times > 0:
        tik_instance.vector_dup(mask_value, data_sign, 0, repeat_times, 1, 8)
    if tail_mask != 0:
        tik_instance.vector_dup(tail_mask,
                                data_sign[repeat_times * mask_value], 0, 1, 1,
                                8)

    zero = tik_instance.Scalar(dtype="float16", name="zeros")
    zero.set_as(0)
    tik_instance.vector_dup(mask_value, data_zeros, zero, 1, 1, 1, 0)
    one = tik_instance.Scalar(dtype="float16", name="ones")
    one.set_as(1)
    data_ones = tik_instance.Tensor("float16", (mask_value, 1),
                                    name="data_ones",
                                    scope=tik.scope_ubuf)
    tik_instance.vector_dup(mask_value, data_ones, one, 1, 1, 1, 0)

    if column_reduce_times > 0:
        with tik_instance.for_range(0, column_reduce_times) as j:
            srcmask = tik_instance.vcmp_gt(mask_value,
                                           half_sub[mask_value * j],
                                           data_zeros, 1, 1)
            tik_instance.vsel(mask_value, 0, data_sign[mask_value * j],
                              srcmask, data_ones, data_zeros, 1, 1, 1, 1)
        if column_reduce_remainder != 0:
            srcmask = tik_instance.vcmp_gt(
                column_reduce_remainder,
                half_sub[column_reduce_times * mask_value], data_zeros, 1, 1)
            tik_instance.vsel(column_reduce_remainder, 0,
                              data_sign[column_reduce_times * mask_value],
                              srcmask, data_ones, data_zeros, 1, 1, 1, 1)
    else:
        srcmask = tik_instance.vcmp_gt(column_reduce_remainder, half_sub,
                                       data_zeros, 1, 1)
        tik_instance.vsel(column_reduce_remainder, 0, data_sign, srcmask,
                          data_ones, data_zeros, 1, 1, 1, 1)

    column_reduce_times = int(column_num // half_mask_value)
    column_reduce_remainder = int(column_num % half_mask_value)

    if column_reduce_times > 0:
        tik_instance.vconv(half_mask_value, '',
                           prediction_tensor_ub,
                           data_sign,
                           column_reduce_times, 1, 1, 8, 4)
        if column_reduce_remainder != 0:
            tik_instance.vconv(
                column_reduce_remainder, '',
                prediction_tensor_ub[column_reduce_times * half_mask_value],
                data_sign[column_reduce_times *
                          half_mask_value], 1, 1, 1, 8, 4)
    else:
        tik_instance.vconv(column_reduce_remainder, '',
                           prediction_tensor_ub,
                           data_sign,
                           1, 1, 1, 8, 4)

    if column_reduce_remainder != 0:
        reduce_mask = column_reduce_times + 1
    else:
        reduce_mask = column_reduce_times

    data_bool = tik_instance.Tensor(
        "float32", (1,),
        name="data_bool",
        scope=tik.scope_ubuf)

    if reduce_mask == 1:
        if column_reduce_remainder != 0:
            tik_instance.vcadd(column_reduce_remainder, data_bool,
                               prediction_tensor_ub,
                               1, 1, 1, 1, 0)
        else:
            tik_instance.vcadd(half_mask_value, data_bool,
                               prediction_tensor_ub,
                               1, 1, 1, 1, 0)
    elif reduce_mask <= half_mask_value:
        tensor_sec_reduce = tik_instance.Tensor("float32",
                                                (1, half_mask_value),
                                                name="tensor_sec_reduce",
                                                scope=tik.scope_ubuf)
        zeros_init = tik_instance.Scalar("float32")
        zeros_init.set_as(0)
        if column_reduce_remainder != 0:
            tik_instance.vector_dup(half_mask_value, tensor_sec_reduce,
                                    zeros_init, 1, 1, 8, 0)
            tik_instance.vcadd(half_mask_value,
                               tensor_sec_reduce,
                               prediction_tensor_ub,
                               column_reduce_times, 1, 1, 8, 0)
            tik_instance.vcadd(
                column_reduce_remainder,
                tensor_sec_reduce[column_reduce_times],
                prediction_tensor_ub[column_reduce_times * half_mask_value],
                1, 1, 1, 1, 0)
            tik_instance.vcadd(reduce_mask, data_bool, tensor_sec_reduce,
                               1, 1, 1, 8, 0)
        else:
            tik_instance.vector_dup(half_mask_value, tensor_sec_reduce,
                                    zeros_init, 1, 1, 8, 0)
            tik_instance.vcadd(half_mask_value,
                               tensor_sec_reduce,
                               prediction_tensor_ub,
                               column_reduce_times, 1, 1, 8, 0)
            tik_instance.vcadd(reduce_mask, data_bool, tensor_sec_reduce,
                               1, 1, 1, 8, 0)
    else:
        tensor_third_reduce = tik_instance.Tensor("float32",
                                                  (1, V_SIZE_BYTES),
                                                  name="tensor_third_reduce",
                                                  scope=tik.scope_ubuf)
        tensor_sec_reduce = tik_instance.Tensor("float32",
                                                (1, half_mask_value),
                                                name="tensor_sec_reduce",
                                                scope=tik.scope_ubuf)

        zeros_init = tik_instance.Scalar("float32")
        zeros_init.set_as(0)
        tik_instance.vector_dup(half_mask_value, tensor_third_reduce,
                                zeros_init, 4, 1, 8, 0)
        tik_instance.vector_dup(half_mask_value, tensor_sec_reduce,
                                zeros_init, 1, 1, 8, 0)
        if column_reduce_remainder != 0:
            tik_instance.vcadd(half_mask_value,
                               tensor_third_reduce,
                               prediction_tensor_ub,
                               column_reduce_times, 1, 1, 8, 0)
            tik_instance.vcadd(
                column_reduce_remainder,
                tensor_third_reduce[column_reduce_times],
                prediction_tensor_ub[column_reduce_times * half_mask_value],
                1, 1, 1, 1, 0)
            tik_instance.vcadd(half_mask_value,
                               tensor_sec_reduce,
                               tensor_third_reduce,
                               4, 1, 1, 8, 0)
            tik_instance.vcadd(half_mask_value, data_bool, tensor_sec_reduce,
                               1, 1, 1, 8, 0)
        else:
            tik_instance.vcadd(half_mask_value,
                               tensor_third_reduce,
                               prediction_tensor_ub,
                               column_reduce_times, 1, 1, 8, 0)
            tik_instance.vcadd(half_mask_value,
                               tensor_sec_reduce,
                               tensor_third_reduce,
                               4, 1, 1, 8, 0)
            tik_instance.vcadd(half_mask_value, data_bool, tensor_sec_reduce,
                               1, 1, 1, 8, 0)

    return data_bool


# pylint: disable=too-many-locals
def _in_top_k_single_core(predictions, targets, k, kernel_name):
    """
    the _in_top_k_single_core function

    Parameters
    ----------
    predictions: dict of predictions
                 include keys(shape and dtype)
    targets: dict of targets
             include keys(shape and dtype)
    k: the k value of top k
    kernel_name: kernel_name
    Returns
    -------
    tik_instance: tik_instance
    """
    element_bytes = 4
    block_element = BLOCK_SIZE // element_bytes
    predictions_shape = predictions.get("shape")
    target_shape = targets.get("shape")
    row = predictions_shape[0]
    column = predictions_shape[1]
    element_bytes = 4
    coexisting_tensor_num = 5
    column_aligned = ((column + block_element - 1) // block_element *
                      block_element)
    max_tensor_size = UB_SIZE_BYTES // coexisting_tensor_num
    core_row_capicity = max_tensor_size // (column_aligned * element_bytes)
    split_row_times = (row + core_row_capicity - 1) // core_row_capicity
    row_nums = (row + split_row_times - 1) // split_row_times
    row_remainder = row - (split_row_times - 1) * row_nums
    prediction_dtype = "float32"
    target_dtype = "int32"
    core_nums = split_row_times
    core_loop = 0
    split_rows_nums = 0

    tik_instance = tik.Tik()

    prediction_tensor = tik_instance.Tensor(prediction_dtype,
                                            predictions_shape,
                                            name="prediction_tensor",
                                            scope=tik.scope_gm)
    target_tensor = tik_instance.Tensor(target_dtype,
                                        target_shape,
                                        name="target_tensor",
                                        scope=tik.scope_gm)
    tensor_output = tik_instance.Tensor("uint8", (row,),
                                        name="tensor_output",
                                        scope=tik.scope_gm)

    # copy predictions to ub from gm,
    # if the last dimension is not divided by 32bytes, just aligned.
    with tik_instance.for_range(0, split_row_times) as outer_loop:
        shape_info = {
            "core_loop": core_loop,
            "outer_loop": outer_loop,
            "core_nums": core_nums,
            "row_remainder": row_remainder,
            "row_nums": row_nums,
            "column": column,
            "column_aligned": column_aligned,
            "split_rows_nums": split_rows_nums
        }
        tensor = {
            "prediction_tensor": prediction_tensor,
            "target_tensor": target_tensor
        }

        tensor_output_ub = _in_top_k_inter_process(shape_info, tensor,
                                                   tik_instance, k)

        length = tik_instance.Scalar("int64")
        length.set_as(row_nums + BLOCK_SIZE - 1)
        tik_instance.data_move(tensor_output[outer_loop * row_nums],
                               tensor_output_ub, 0, 1, length // BLOCK_SIZE, 0,
                               0)
    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[prediction_tensor, target_tensor],
                          outputs=[tensor_output])
    return tik_instance


# pylint: disable=too-many-locals
def _in_top_k_mul_core(predictions, targets, k, kernel_name):
    """
    the _in_top_k_mul_core function

    Parameters
    ----------
    predictions: dict of predictions
                 include keys(shape and dtype)
    targets: dict of targets
             include keys(shape and dtype)
    k: the k value of top k
    kernel_name: kernel_name
    Returns
    -------
    tik_instance: tik_instance
    """
    element_bytes = 4
    coexisting_tensor_num = 5
    predictions_shape = predictions.get("shape")
    target_shape = targets.get("shape")
    prediction_dtype = "float32"
    target_dtype = "int32"

    row = predictions_shape[0]
    column = predictions_shape[1]
    block_element = BLOCK_SIZE // element_bytes
    column_aligned = ((column + block_element - 1) // block_element *
                      block_element)
    max_tensor_size = UB_SIZE_BYTES // coexisting_tensor_num
    core_row_capicity = max_tensor_size // (column_aligned * element_bytes)
    core_loop = 0
    split_rows_nums = 0
    tik_instance = tik.Tik()
    mini_cloud_core_nums = \
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
    if core_row_capicity < row:
        row_block_num = (row + BLOCK_SIZE - 1) // BLOCK_SIZE
        core_block_nums = core_row_capicity // BLOCK_SIZE
        if row_block_num <= mini_cloud_core_nums:
            # align row, and then split it
            row_nums = BLOCK_SIZE
            core_nums = (row + row_nums - 1) // row_nums
            tail_block = core_nums - 1
            row_remainder = row - row_nums * tail_block
        else:
            # align row, and then split it
            row_nums = core_block_nums * BLOCK_SIZE
            core_nums = (row + row_nums - 1) // row_nums
            tail_block = core_nums - 1
            row_remainder = row - row_nums * tail_block
    else:
        row_split_num = (row + BLOCK_SIZE - 1) // BLOCK_SIZE
        if row % BLOCK_SIZE == 0 and row_split_num > mini_cloud_core_nums:
            block_size_times = row_split_num // mini_cloud_core_nums
            tail_block = row_split_num % mini_cloud_core_nums
            row_nums = BLOCK_SIZE * block_size_times
            core_nums = mini_cloud_core_nums
            if tail_block == 0:
                row_remainder = block_size_times * BLOCK_SIZE
            else:
                row_remainder = (block_size_times - 1) * BLOCK_SIZE
        else:
            # align row, and then split it
            row_nums = BLOCK_SIZE
            core_nums = (row + row_nums - 1) // row_nums
            tail_block = core_nums - 1
            row_remainder = row - row_nums * tail_block

    prediction_tensor = tik_instance.Tensor(prediction_dtype,
                                            predictions_shape,
                                            name="prediction_tensor",
                                            scope=tik.scope_gm)
    target_tensor = tik_instance.Tensor(target_dtype,
                                        target_shape,
                                        name="target_tensor",
                                        scope=tik.scope_gm)
    tensor_output = tik_instance.Tensor("uint8", (row,),
                                        name="tensor_output",
                                        scope=tik.scope_gm)
    # copy predictions to ub from gm,
    # if the last dimension is not divided by 32bytes,just aligned
    with tik_instance.for_range(0, core_nums,
                                block_num=core_nums) as outer_loop:
        shape_info = {
            "core_loop": core_loop,
            "outer_loop": outer_loop,
            "core_nums": core_nums,
            "row_remainder": row_remainder,
            "row_nums": row_nums,
            "column": column,
            "column_aligned": column_aligned,
            "split_rows_nums": split_rows_nums
        }
        tensor = {
            "prediction_tensor": prediction_tensor,
            "target_tensor": target_tensor
        }
        tensor_output_ub = _in_top_k_inter_process(shape_info, tensor,
                                                   tik_instance, k)
        length = tik_instance.Scalar("int64")
        with tik_instance.if_scope(outer_loop < tail_block):
            length.set_as(row_nums + BLOCK_SIZE - 1)
        with tik_instance.else_scope():
            length.set_as(row_remainder + BLOCK_SIZE - 1)
        tik_instance.data_move(tensor_output[outer_loop * row_nums],
                               tensor_output_ub, 0, 1, length // BLOCK_SIZE, 0,
                               0)
    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[prediction_tensor, target_tensor],
                          outputs=[tensor_output])
    return tik_instance


# pylint: disable=too-many-locals
def _in_top_k_mul_core_v2(predictions, targets, k, kernel_name):
    """
    the _in_top_k_mul_core_v2 function

    Parameters
    ----------
    predictions: dict of predictions
                 include keys(shape and dtype)
    targets: dict of targets
             include keys(shape and dtype)
    k: the k value of top k
    kernel_name: kernel_name
    Returns
    -------
    tik_instance: tik_instance
    """
    element_bytes = 4
    coexisting_tensor_num = 5
    predictions_shape = predictions.get("shape")
    target_shape = targets.get("shape")
    prediction_dtype = "float32"
    target_dtype = "int32"
    row = predictions_shape[0]
    column = predictions_shape[1]

    block_element = BLOCK_SIZE // element_bytes
    column_aligned = ((column + block_element - 1) // block_element *
                      block_element)
    max_tensor_size = UB_SIZE_BYTES // coexisting_tensor_num
    core_row_capicity = max_tensor_size // (column_aligned * element_bytes)

    tik_instance = tik.Tik()

    split_core_nums = (row + BLOCK_SIZE - 1) // BLOCK_SIZE
    split_rows_nums = BLOCK_SIZE
    split_remainder = row - (split_core_nums - 1) * BLOCK_SIZE
    non_tail_core = split_core_nums - 1

    row_nums = core_row_capicity
    single_core_times = (BLOCK_SIZE + core_row_capicity -
                         1) // core_row_capicity

    non_tail_block = single_core_times - 1
    row_remainder = BLOCK_SIZE - row_nums * non_tail_block

    last_single_core_times = (split_remainder + core_row_capicity -
                              1) // core_row_capicity
    last_non_tail_block = last_single_core_times - 1
    last_row_remainder = split_remainder - \
                         last_non_tail_block * core_row_capicity

    prediction_tensor = tik_instance.Tensor(prediction_dtype,
                                            predictions_shape,
                                            name="prediction_tensor",
                                            scope=tik.scope_gm)
    target_tensor = tik_instance.Tensor(target_dtype,
                                        target_shape,
                                        name="target_tensor",
                                        scope=tik.scope_gm)
    tensor_output = tik_instance.Tensor("uint8", (row,),
                                        name="tensor_output",
                                        scope=tik.scope_gm)

    with tik_instance.for_range(0, split_core_nums,
                                block_num=split_core_nums) as core_loop:

        tensor = {
            "prediction_tensor": prediction_tensor,
            "target_tensor": target_tensor
        }

        # step 0: set some shape value of tensor in UB
        outer_core_num = tik_instance.Scalar("int64")
        output_ub = tik_instance.Tensor("uint8", (BLOCK_SIZE,),
                                        name="output_ub",
                                        scope=tik.scope_ubuf)
        with tik_instance.if_scope(core_loop < non_tail_core):
            outer_core_num.set_as(split_rows_nums)
            with tik_instance.for_range(0, single_core_times) as outer_loop:
                core_row_num = tik_instance.Scalar("int64")
                with tik_instance.if_scope(outer_loop < non_tail_block):
                    core_row_num.set_as(row_nums)
                with tik_instance.else_scope():
                    core_row_num.set_as(row_remainder)
                shape_info = {
                    "core_loop": core_loop,
                    "outer_loop": outer_loop,
                    "core_nums": single_core_times,
                    "row_remainder": row_remainder,
                    "row_nums": row_nums,
                    "column": column,
                    "column_aligned": column_aligned,
                    "split_rows_nums": split_rows_nums
                }
                tensor_output_ub = _in_top_k_inter_process(
                    shape_info, tensor, tik_instance, k)
                with tik_instance.for_range(0, core_row_num) as i:
                    output_ub[outer_loop * row_nums + i] = tensor_output_ub[i]

        with tik_instance.else_scope():
            outer_core_num.set_as(split_remainder)
            with tik_instance.for_range(0,
                                        last_single_core_times) as outer_loop:
                core_row_num = tik_instance.Scalar("int64")
                shape_info = {
                    "core_loop": core_loop,
                    "outer_loop": outer_loop,
                    "core_nums": last_single_core_times,
                    "row_remainder": last_row_remainder,
                    "row_nums": row_nums,
                    "column": column,
                    "column_aligned": column_aligned,
                    "split_rows_nums": split_rows_nums
                }
                with tik_instance.if_scope(outer_loop < last_non_tail_block):
                    core_row_num.set_as(row_nums)
                with tik_instance.else_scope():
                    core_row_num.set_as(last_row_remainder)
                tensor_output_ub = _in_top_k_inter_process(
                    shape_info, tensor, tik_instance, k)
                with tik_instance.for_range(0, core_row_num) as i:
                    output_ub[outer_loop * row_nums + i] = tensor_output_ub[i]
        index = core_loop * BLOCK_SIZE
        tik_instance.data_move(tensor_output[index], output_ub, 0, 1, 1, 0, 0)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[prediction_tensor, target_tensor],
                          outputs=[tensor_output])
    return tik_instance


# pylint: disable=too-many-locals
def _in_top_k_column_inner_loop(shape_info, tensor, tik_instance, k):
    """
    the _in_top_k_column_inner_loop function

    Parameters
    ----------
    shape_info: dict
                include keys(core_loop, outer_loop, column, column_size,
                             single_core_times, split_column_nums)
    tensor: dict
             include keys(prediction_tensor and target_tensor)
    tik_instance: tik_instance
    Returns
    -------
    tensor_output_temp_ub: tensor
    """
    core_loop = shape_info.get("core_loop")
    outer_loop = shape_info.get("outer_loop")
    column = shape_info.get("column")
    column_size = shape_info.get("column_size")
    last_column_size = shape_info.get("last_column_size")
    single_core_times = shape_info.get("single_core_times")
    split_column_nums = shape_info.get("split_column_nums")
    target_tensor = tensor.get("target_tensor")
    prediction_tensor = tensor.get("prediction_tensor")
    prediction_dtype = "float32"
    target_dtype = "int32"
    element_bytes = 4
    block_element = BLOCK_SIZE / element_bytes

    target_ub = tik_instance.Tensor(target_dtype, (block_element,),
                                    name="target_ub",
                                    scope=tik.scope_ubuf)
    data_temp_ub = tik_instance.Tensor(prediction_dtype, (block_element,),
                                       name="data_temp_ub",
                                       scope=tik.scope_ubuf)
    tensor_output_ub_temp = tik_instance.Tensor("uint8", (BLOCK_SIZE,),
                                                name="tensor_output_ub",
                                                scope=tik.scope_ubuf)
    # copy target_tensor to UB
    # just need 1 element each time, but read 8 elements
    index = core_loop * single_core_times + outer_loop
    tik_instance.data_move(target_ub, target_tensor[index], 0, 1, 1, 0, 0)
    scalar_target = tik_instance.Scalar(target_dtype, "scalar_target")
    scalar_target.set_as(target_ub[0])
    index = core_loop * single_core_times * column + column * outer_loop
    tik_instance.data_move(data_temp_ub,
                           prediction_tensor[index + scalar_target], 0, 1, 1,
                           0, 0)

    # get the value according to the target of each row
    scalar_value = tik_instance.Scalar(prediction_dtype)
    scalar_value.set_as(data_temp_ub[0])
    tensor["scalar_value"] = scalar_value
    bool_src = tik_instance.Tensor(prediction_dtype, (1,),
                                   name="bool_src",
                                   scope=tik.scope_ubuf)
    tik_instance.vector_dup(1, bool_src, 0, 1, 1, 1)
    # record the number of element which is larger than target element
    bool_sum = tik_instance.Tensor(prediction_dtype, (1,),
                                   name="bool_sum",
                                   scope=tik.scope_ubuf)
    tik_instance.vector_dup(1, bool_sum, 0, 1, 1, 1)

    with tik_instance.for_range(0, split_column_nums) as inner_loop:
        with tik_instance.if_scope(inner_loop < split_column_nums - 1):
            shape_info["inner_loop"] = inner_loop
            shape_info["column_num"] = column_size
            data_bool = _in_top_k_column_process(shape_info, tensor,
                                                 tik_instance)

            tik_instance.vadd(1, bool_sum, bool_src, data_bool, 1, 1, 1, 1, 8,
                              8, 8)
            sum_scalar = tik_instance.Scalar(prediction_dtype,
                                             name="sum_scalar")
            sum_scalar.set_as(0)
            tik_instance.vadds(1, bool_src, bool_sum, sum_scalar, 1, 1, 1, 8,
                               8, 0)
        with tik_instance.else_scope():
            shape_info["inner_loop"] = inner_loop
            shape_info["column_num"] = last_column_size
            data_bool = _in_top_k_column_process(shape_info, tensor,
                                                 tik_instance)

            tik_instance.vadd(1, bool_sum, bool_src, data_bool, 1, 1, 1, 1, 8,
                              8, 8)
            sum_scalar = tik_instance.Scalar(prediction_dtype,
                                             name="sum_scalar")
            sum_scalar.set_as(0)
            tik_instance.vadds(1, bool_src, bool_sum, sum_scalar, 1, 1, 1, 8,
                               8, 0)

    tensor_k = tik_instance.Tensor(prediction_dtype, (8,),
                                   name="tensor_k",
                                   scope=tik.scope_ubuf)
    tensor_zeros = tik_instance.Tensor("float16", (8,),
                                       name="zeros",
                                       scope=tik.scope_ubuf)
    tensor_ones = tik_instance.Tensor("float16", (8,),
                                      name="ones",
                                      scope=tik.scope_ubuf)

    k_conv_dtype = tik_instance.Scalar(prediction_dtype)
    k_conv_dtype.set_as(k)
    tik_instance.vector_dup(1, tensor_k, k_conv_dtype, 1, 1, 1, 0)

    tensor_sub_float = tik_instance.Tensor(prediction_dtype, (1,),
                                           name="tensor_sub_float",
                                           scope=tik.scope_ubuf)
    tensor_sub_half = tik_instance.Tensor("float16", (1,),
                                          name="tensor_sub_half",
                                          scope=tik.scope_ubuf)
    tik_instance.vsub(1, tensor_sub_float, bool_sum, tensor_k, 1, 1, 1, 1, 8,
                      8, 8)
    tik_instance.vconv(1, '', tensor_sub_half, tensor_sub_float, 1, 1, 1, 8, 8)

    zeros_half = tik_instance.Scalar("float16")
    zeros_half.set_as(0)
    tik_instance.vector_dup(1, tensor_zeros, zeros_half, 1, 1, 1, 0)
    ones_half = tik_instance.Scalar("float16")
    ones_half.set_as(1)
    tik_instance.vector_dup(1, tensor_ones, ones_half, 1, 1, 1, 0)

    data_bool_ub = tik_instance.Tensor("float16", (1,),
                                       name="data_bool_ub",
                                       scope=tik.scope_ubuf)
    cmp_mask = tik_instance.vcmp_lt(1, tensor_sub_half, tensor_zeros, 1, 1)
    tik_instance.vsel(1, 0, data_bool_ub, cmp_mask, tensor_ones, tensor_zeros,
                      1, 1, 1, 1)
    tik_instance.vconv(1, '', tensor_output_ub_temp, data_bool_ub, 1, 1, 1, 8,
                       8)

    return tensor_output_ub_temp


# pylint: disable=too-many-locals
def _in_top_k_tiling_column(predictions, targets, k, kernel_name):
    """
    the _in_top_k_tiling_column function

    Parameters
    ----------
    predictions: dict of predictions
                 include keys(shape and dtype)
    targets: dict of targets
             include keys(shape and dtype)
    k: the k value of top k
    kernel_name: kernel_name
    Returns
    -------
    tik_instance: tik_instance
    """
    predictions_shape = predictions.get("shape")
    target_shape = targets.get("shape")
    prediction_dtype = predictions.get("dtype").lower()
    target_dtype = targets.get("dtype").lower()
    column = predictions_shape[1]

    tik_instance = tik.Tik()

    element_bytes = 4
    coexisting_tensor_num = 5
    block_element = BLOCK_SIZE / element_bytes
    column_aligned = ((column + block_element - 1) // block_element *
                      block_element)
    max_tensor_size = UB_SIZE_BYTES // coexisting_tensor_num // \
                      block_element * block_element - block_element

    split_column_nums = (column_aligned * element_bytes + max_tensor_size -
                         1) // max_tensor_size
    column_num_temp = column_aligned // split_column_nums
    column_num = (column_num_temp + block_element - 1) // \
                 block_element * block_element + block_element
    last_column_size = int(column_aligned - (column_num *
                                             (split_column_nums - 1)))

    row = predictions_shape[0]
    if row < BLOCK_SIZE:
        split_row_times = 1
        single_core_times = row
        row_remainder = row
    else:
        split_row_times = (row + BLOCK_SIZE - 1) // BLOCK_SIZE
        single_core_times = BLOCK_SIZE
        row_remainder = row - (split_row_times - 1) * BLOCK_SIZE

    prediction_tensor = tik_instance.Tensor(prediction_dtype,
                                            predictions_shape,
                                            name="prediction_tensor",
                                            scope=tik.scope_gm)
    target_tensor = tik_instance.Tensor(target_dtype,
                                        target_shape,
                                        name="target_tensor",
                                        scope=tik.scope_gm)

    move_data_count = tik_instance.Scalar(dtype="int64",
                                          name="move_data_count")
    move_data_count.set_as(0)
    tensor_output = tik_instance.Tensor("uint8", (predictions_shape[0],),
                                        name="tensor_output",
                                        scope=tik.scope_gm)

    with tik_instance.for_range(0, split_row_times,
                                block_num=split_row_times) as core_loop:
        output_ub = tik_instance.Tensor("uint8", (BLOCK_SIZE,),
                                        name="output_ub",
                                        scope=tik.scope_ubuf)
        with tik_instance.if_scope(core_loop < split_row_times - 1):
            with tik_instance.for_range(0, single_core_times) as outer_loop:
                shape_info = {
                    "core_loop": core_loop,
                    "outer_loop": outer_loop,
                    "single_core_times": single_core_times,
                    "split_column_nums": split_column_nums,
                    "column_size": column_num,
                    "column": column,
                    "column_num": last_column_size,
                    "last_column_size": last_column_size
                }
                tensor = {
                    "prediction_tensor": prediction_tensor,
                    "target_tensor": target_tensor
                }
                tensor_output_temp_ub = _in_top_k_column_inner_loop(
                    shape_info, tensor, tik_instance, k)
                output_ub[outer_loop] = tensor_output_temp_ub[0]
            index = core_loop * single_core_times
            tik_instance.data_move(tensor_output[index], output_ub, 0, 1, 1, 1,
                                   1)

        with tik_instance.else_scope():
            with tik_instance.for_range(0, row_remainder) as outer_loop:
                shape_info = {
                    "core_loop": core_loop,
                    "outer_loop": outer_loop,
                    "single_core_times": single_core_times,
                    "split_column_nums": split_column_nums,
                    "column_size": column_num,
                    "column": column,
                    "column_num": last_column_size,
                    "last_column_size": last_column_size
                }
                tensor = {
                    "prediction_tensor": prediction_tensor,
                    "target_tensor": target_tensor
                }
                tensor_output_temp_ub = _in_top_k_column_inner_loop(
                    shape_info, tensor, tik_instance, k)
                output_ub[outer_loop] = tensor_output_temp_ub[0]
            index = core_loop * single_core_times
            tik_instance.data_move(tensor_output[index], output_ub, 0, 1, 1, 1,
                                   1)
    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[prediction_tensor, target_tensor],
                          outputs=[tensor_output])
    return tik_instance
