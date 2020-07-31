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

strided slice
"""

import copy
import math
from functools import reduce as functools_reduce

from te.lang.cce.rl_bank import rl_bank
from te import platform as tbe_platform
from te import tvm
from te.platform.cce_build import build_config
from topi.cce import util
from te.platform import insn_cmd
from te.platform.fusion_manager import fusion_manager

from .strided_slice_for_last_dim import strided_slice_last_dim
from .copy_only import copy_only
from .strided_slice_two_turn_one import strided_slice_two_turn_one
from .strided_slice_fast_last_dim import strided_slice_last_dim_only
from .strided_slice_last_dim_one import strided_slice_last_dim_one

# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648


def _shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    result_list = []
    for i in shape:
        result_list.append(i.value)

    return result_list


def _fill_list_with_ones(length):
    """
    fill a list array with ones
    """
    result_list = [1] * length

    return result_list


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-branches,too-many-statements
def _init_parameter(input_list, begin_shape, end_shape, stride_shape,
                    begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                    shrink_axis_mask):
    """
    init the begin and end parameters.
    1.support when begin shape and end shape is less than input shape (new axis mask = 0 or != 0)
    2.support when the value of begin and end is negative
    3.support when the begin mask and end mask is not 0
    4.support when ellipsis mask is not 0
    5.support when stride shape is less than input shape
    6.when new axis mask or shrink axis mask is not 0, change corresponding begin/end/stride
    """

    formal_begin_len = len(begin_shape)
    input_len = len(input_list)
    input_calc = copy.deepcopy(input_list)
    if new_axis_mask != 0:
        for i, _ in enumerate(input_calc):
            if new_axis_mask & 2**i == 2**i:
                input_calc.insert(i, 1)

    i = 0
    # if begin and end shape is not equal to input shape, then append begin and end
    if ellipsis_mask != 0:
        if end_mask != 0 and input_len != formal_begin_len:
            end_mask *= end_mask * 2**(input_len - formal_begin_len - 1)
        if begin_mask != 0 and input_len != formal_begin_len:
            begin_mask *= begin_mask * 2**(input_len - formal_begin_len - 1)
        for i, _ in enumerate(input_calc):
            if (ellipsis_mask & 2**i) == 2**i:
                ellipsis_dim = i
                begin_shape[i] = 0
                end_shape[i] = input_calc[i]
                stride_shape[i] = 1
                if len(begin_shape) < input_len:
                    for j in range(1, input_len - formal_begin_len + 1):
                        begin_shape.insert(ellipsis_dim + j, 0)
                        end_shape.insert(ellipsis_dim + j,
                                         input_calc[ellipsis_dim + j])
                        stride_shape.insert(ellipsis_dim + j, 1)

    while len(begin_shape) < input_len:
        begin_shape.append(0)
        end_shape.append(input_calc[formal_begin_len + i])
        stride_shape.append(1)
        i += 1

    # if the value of begin and end is negative, transform to the positive value
    for i, _ in enumerate(zip(begin_shape, end_shape, input_calc)):
        if stride_shape[i] > 0:
            if begin_shape[i] >= 0 and end_shape[i] >= 0:
                if begin_shape[i] >= end_shape[i]:
                    end_shape[i] = begin_shape[i]
                if begin_shape[i] < end_shape[i]:
                    if begin_shape[i] >= input_calc[i]:
                        end_shape[i] = begin_shape[i] = input_calc[i]
                    if begin_shape[i] < input_calc[i] and \
                           end_shape[i] >= input_calc[i]:
                        end_shape[i] = input_calc[i]
            if begin_shape[i] < 0 and end_shape[i] > 0:
                begin_shape[i] = begin_shape[i] + input_calc[i]
                if begin_shape[i] >= end_shape[i]:
                    begin_shape[i] = end_shape[i]
                if begin_shape[i] < end_shape[i]:
                    if begin_shape[i] <= 0:
                        begin_shape[i] = 0
                    if begin_shape[i] < input_calc[i] and \
                           end_shape[i] > input_calc[i]:
                        end_shape[i] = input_calc[i]
            if begin_shape[i] >= 0 and end_shape[i] < 0:
                end_shape[i] = end_shape[i] + input_calc[i]
                if end_shape[i] <= begin_shape[i]:
                    begin_shape[i] = end_shape[i]
            if begin_shape[i] < 0 and end_shape[i] < 0:
                begin_shape[i] = begin_shape[i] + input_calc[i]
                end_shape[i] = end_shape[i] + input_calc[i]
                if begin_shape[i] >= 0 and end_shape[i] >= 0:
                    if begin_shape[i] >= end_shape[i]:
                        begin_shape[i] = end_shape[i]
                if begin_shape[i] >= 0 and end_shape[i] < 0:
                    begin_shape[i] = end_shape[i]
                if begin_shape[i] < 0 and end_shape[i] >= 0:
                    begin_shape[i] = 0
                if begin_shape[i] < 0 and end_shape[i] < 0:
                    begin_shape[i] = 0
                    end_shape[i] = 0
        if stride_shape[i] < 0:
            if begin_shape[i] >= 0 and end_shape[i] >= 0:
                if begin_shape[i] <= end_shape[i]:
                    begin_shape[i] = end_shape[i]
                if begin_shape[i] > end_shape[i]:
                    if stride_shape[i] == -1:
                        if begin_shape[i] >= input_calc[i]:
                            begin_shape[i] = input_calc[i] - 1
                    else:
                        begin_shape[i] = input_calc[i]
            if begin_shape[i] >= 0 and end_shape[i] < 0:
                end_shape[i] = end_shape[i] + input_calc[i]
                if end_shape[i] >= 0:
                    if end_shape[i] >= begin_shape[i]:
                        end_shape[i] = begin_shape[i]
                    if end_shape[i] < begin_shape[i]:
                        if stride_shape[i] == -1:
                            if begin_shape[i] >= input_calc[i]:
                                begin_shape[i] = input_calc[i] - 1
                        else:
                            if begin_shape[i] >= input_calc[i]:
                                begin_shape[i] = input_calc[i]
                if end_shape[i] < 0:
                    begin_shape[i] = begin_shape[i] + 1
                    end_shape[i] = 0
            if begin_shape[i] < 0 and end_shape[i] >= 0:
                begin_shape[i] = begin_shape[i] + input_calc[i]
                if begin_shape[i] >= 0:
                    if begin_shape[i] <= end_shape[i]:
                        begin_shape[i] = end_shape[i]
                if begin_shape[i] < 0:
                    begin_shape[i] = end_shape[i]
            if begin_shape[i] < 0 and end_shape[i] < 0:
                end_shape[i] = end_shape[i] + input_calc[i]
                begin_shape[i] = begin_shape[i] + input_calc[i]
                if begin_shape[i] >= 0 and end_shape[i] >= 0:
                    if begin_shape[i] <= end_shape[i]:
                        begin_shape[i] = end_shape[i]
                    if begin_shape[i] > end_shape[i]:
                        if begin_shape[i] >= input_calc[i]:
                            begin_shape[i] = input_calc[i] - 1
                if begin_shape[i] >= 0 and end_shape[i] < 0:
                    begin_shape[i] = begin_shape[i] + 1
                    end_shape[i] = 0
                if begin_shape[i] < 0 and end_shape[i] >= 0:
                    begin_shape[i] = end_shape[i]
                if begin_shape[i] < 0 and end_shape[i] < 0:
                    begin_shape[i] = end_shape[i]

    if shrink_axis_mask != 0:
        for i, _ in enumerate(input_calc):
            if shrink_axis_mask & 2**i == 2**i:
                input_calc[i] = 0

    # if the begin mask or end mask or ellipsis is not 0, update begin and end shape
    for i, _ in \
            enumerate(zip(begin_shape, end_shape, input_calc)):
        if (begin_mask & 2**i) == 2**i:
            if stride_shape[i] > 0:
                begin_shape[i] = 0
            else:
                begin_shape[i] = input_calc[i]

        if (end_mask & 2**i) == 2**i:
            if stride_shape[i] > 0:
                end_shape[i] = input_calc[i]
            else:
                end_shape[i] = 0
        if (ellipsis_mask & 2**i) == 2**i:
            begin_shape[i] = 0
            end_shape[i] = input_calc[i]
            stride_shape[i] = 1

    i = 0
    # if new axis mask is not 0, need to update begin and end shape
    if new_axis_mask != 0:
        while len(begin_shape) < len(input_calc):
            begin_shape.append(0)
            end_shape.append(input_calc[input_len + i])
            i = i + 1
    # if stride shape is less than begin shape, fill with 1
    if len(stride_shape) < len(begin_shape):
        while len(stride_shape) < len(begin_shape):
            stride_shape.append(1)

    # if begin or end is greater than length, update the value
    for i, _ in enumerate(zip(stride_shape, input_calc)):
        if stride_shape[i] > 0:
            if end_shape[i] > input_calc[i]:
                end_shape[i] = input_calc[i]
        if stride_shape[i] < 0:
            if begin_shape[i] >= input_calc[i]:
                begin_shape[i] = input_calc[i] - 1
    # in order not to get the output dim is 0 or negative,and change
    # end shape when shrink axis mask is not 0
    for i, _ in enumerate(begin_shape):
        if (new_axis_mask & (2**i)) == 2**i:
            begin_shape[i] = 0
            end_shape[i] = 1
            stride_shape[i] = 1
        if shrink_axis_mask & 2**i == 2**i:
            end_shape[i] = begin_shape[i] + 1

    return begin_shape, end_shape, stride_shape


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-branches,unused-argument
def strided_slice_d_compute(input_data,
                            output_x,
                            begin,
                            end,
                            stride_shape=None,
                            begin_mask=0,
                            end_mask=0,
                            ellipsis_mask=0,
                            new_axis_mask=0,
                            shrink_axis_mask=0,
                            kernel_name="strided_slice_d"):
    """
    extracts a slice of size (end-begin)/stride from the given input_data.

    Parameters:
    ----------
    input_data: TVM tensor.
        Tensor to be segmented.
    output_x : dict
        shape and dtype of out
    begin: list.
        represents the index of the first value to select.
    end: list.
        represents the index of the last value to select.
    stride_shape: list or tuple.
        step length to select.
    begin_mask: int
        a bitmask where a bit i being 1 means to ignore the begin value and instead use the
        largest interval possible.
    end_mask: int
        analogous to `begin_mask`.
    ellipsis_mask: int
        a bitmask where bit `i` being 1 means the `i`th position is actually an ellipsis.
    new_axis_mask: int
        a bitmask where bit `i` being 1 means the `i`th specification creates a
        new shape 1 dimension.
    shrink_axis_mask: int
        a bitmask where bit `i` implies that the `i`th specification should shrink
        the dimensionality.

    Returns
    -------
    Computational process for TVM compute.
    """
    select_run_branch = 0
    begin_shape = copy.deepcopy(begin)
    end_shape = copy.deepcopy(end)
    stride_shape = copy.deepcopy(stride_shape)
    input_list = _shape_to_list(input_data.shape)
    # update begin_shape, end_shape
    begin_shape, end_shape, stride_shape = _init_parameter(
        input_list, begin_shape, end_shape, stride_shape, begin_mask, end_mask,
        ellipsis_mask, new_axis_mask, shrink_axis_mask)

    if stride_shape[-1] != 1:
        raise RuntimeError("Only support strides with 1 at last value.")

    # Estimate the value of shrink_axis_mask, to avoid the dimension of output with size 0
    shrink_axis_mask_temp = 0
    diff_begin_end = []

    for i, _ in enumerate(zip(end_shape, begin_shape)):
        diff_begin_end.append(end_shape[i] - begin_shape[i])

    for i, diff_begin_end_i in enumerate(diff_begin_end):

        if diff_begin_end_i == 0 and (new_axis_mask & 2**i) != 2**i:
            shrink_axis_mask_temp = shrink_axis_mask_temp + 2**i
    shrink_axis_mask = shrink_axis_mask | shrink_axis_mask_temp

    # handle only shrink_axis_mask
    if shrink_axis_mask > 0 and new_axis_mask == 0:
        for i, _ in enumerate(zip(begin_shape, end_shape)):
            if (shrink_axis_mask & 2**i) == 2**i:
                if begin_shape[i] < 0:
                    begin_shape[i] = input_list[i] + begin_shape[i]
                end_shape[i] = begin_shape[i] + 1

    output_shape = [
        int(math.ceil((end - begin) / (stride * 1.0)))
        for begin, end, stride in zip(begin_shape, end_shape, stride_shape)
    ]

    if shrink_axis_mask > 0 or new_axis_mask > 0:
        select_run_branch = 1

    # To construct output_shape_shrink_axis accord to shrink_axis_mask
    if shrink_axis_mask > 0:
        shrink_flag = 0
        for i, _ in enumerate(input_list):
            if (shrink_axis_mask & 2**i) == 2**i:
                del output_shape[i - shrink_flag]
                shrink_flag += 1

    # AICore don't support sclar
    if not output_shape:
        output_shape = [1]

    def _map_index_norm(*index):
        """
        calculate normal index by strided and begin parameters.
        """
        for i, _ in enumerate(zip(begin_shape, stride_shape)):
            if i == 0:
                index_org = (index[i] * stride_shape[i] + begin_shape[i], )
            else:
                index_org = index_org + (index[i] * stride_shape[i] +
                                         begin_shape[i], )
        return index_org

    def _map_index_new_or_shrink_axis(*index):
        """
        calculate index by strided and begin parameters when new axis mask
        or shrink axis mask is not 0.
        """
        location = 0
        index_org_axis = None
        for i, _ in enumerate(zip(begin_shape, stride_shape)):
            if (new_axis_mask & 2**i) == 2**i:
                location += 1
            elif (shrink_axis_mask & 2**i) == 2**i:
                if i == 0:
                    index_org_axis = (0 + begin_shape[i], )
                else:
                    index_org_axis = index_org_axis + (0 + begin_shape[i], )
            else:
                if index_org_axis is None:
                    index_org_axis = (index[location] * stride_shape[i] +
                                      begin_shape[i], )
                else:
                    index_org_axis = index_org_axis + \
                                     (index[location] * stride_shape[i] + begin_shape[i],)
                location += 1
        return index_org_axis

    # normal situation, new axis mask == 0 and shrink axis mask == 0
    if select_run_branch == 0:
        output = tvm.compute(output_shape,
                             lambda *i: input_data(*_map_index_norm(*i)),
                             name='output', tag='strided_slice_d|1')

    # new axis mask != 0 and shrink axis mask != 0
    elif select_run_branch == 1:
        output = tvm.compute(output_shape, lambda *i: input_data(
            *_map_index_new_or_shrink_axis(*i)),
                             name='output',
                             tag='strided_slice_d|2'
                             )

    return [output, output_shape]


# pylint: disable=locally-disabled,too-many-return-statements
def _check_parameter(input_shape, begin, end, strides, ellipsis_mask,
                     new_axis_mask, shrink_axis_mask):
    """
    check if the input parameters shape
    """
    ellipsis_dim = 0
    if len(end) != len(begin):
        print("end shape,begin shape length mismatch!")
        return False

    if strides is not None and new_axis_mask == 0 and shrink_axis_mask == 0:
        if len(end) != len(strides) or len(begin) != len(strides):
            print("end shape and strides length mismatch!")
            return False
        for i, _ in enumerate(begin):
            if strides[i] == 0:
                print("strides should be non zero")
                return False

    if ellipsis_mask != 0:
        for i, _ in enumerate(input_shape):
            if (ellipsis_mask & 2**i) == 2**i:
                ellipsis_dim += 1
        if ellipsis_dim > 1:
            print("only suppot 1 dim of ellipsis")
            return False
    return True


def _tilling_axis(shape, dtype):
    """
    split axis and return split_factor
    """
    # minus 1024 (Bytes) to avoid overflow
    ub_size_bytes = \
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) - 1024
    # Convert byts to Bytes
    dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8

    # 32 means one block size(32 Bytes), divide by 32 to get the numbers of data that
    # can be stored in one block.
    flag = 32 // dtype_bytes_size
    element_new = math.ceil(shape[-1] / flag) * flag
    shape_new = []
    for i in shape:
        shape_new.append(i)
    shape_new[-1] = int(element_new)

    total_ele = ub_size_bytes // dtype_bytes_size
    split_axis = 0
    split_factor = 1
    for i, _ in enumerate(shape_new):
        ele_cnt = functools_reduce(lambda x, y: x * y, shape_new[i:])
        if ele_cnt <= total_ele:
            split_axis = i - 1
            split_factor = total_ele // ele_cnt
            break

    if shape_new[-1] > total_ele:
        split_axis = len(shape_new) - 1
        split_factor = total_ele

    if split_axis < 0:
        split_axis = 0
        split_factor = shape_new[0]

    # if the shape[-1]=1 and have two or more dims >1 and tilling axis is not itself ,
    # then change the split axis to itself,and fatcor is its value
    flag_first = -1
    flag_second = -1
    if shape[-1] == 1:
        for i, item in enumerate(reversed(shape)):
            if item > 1:
                # the first dim greater than 1 in reverse order
                flag_first = i
                break
        for i, item in enumerate(reversed(shape)):
            if i <= flag_first:
                continue
            if item > 1:
                # the second dim greater than 1 in reverse order
                flag_second = i
                break
        if flag_first != -1 and flag_second != -1 and split_axis != len(
                shape_new) - 1 - flag_first:
            split_axis = len(shape_new) - 1 - flag_first
            split_factor = shape_new[split_axis]

    return split_axis, split_factor


def _get_align_axis(out_shape):
    """
    get the axis_info when applying the align
    """
    flag = -1
    if out_shape[-1] != 1:
        axis = len(out_shape) - 2
    else:
        for i, item in enumerate(reversed(out_shape)):
            if item > 1:
                # the first dim greater than 1 in reverse order
                flag = i
                break
        if flag in (-1, 0):
            axis = 0
        else:
            axis = len(out_shape) - flag - 1

    return axis


def _get_target_core_num(first_axis_size):
    cloud_core_num = 32
    target_core_num = cloud_core_num
    for i in reversed(list(range(1, cloud_core_num + 1))):
        if first_axis_size % i == 0:
            target_core_num = i
            break

    return target_core_num


def _check_prime_number(num):
    """
    check the num belongs to prime number
    """
    res = -1
    max_num = num - 1
    for i in range(2, max_num):
        if num % i == 0:
            res = i
            if i > 32:
                break

    return res


def _get_core_num_last_axis(input_shape, dtype):
    """
    return the four return args:
    the first axis factor, the first_inner facotr, the second_axis factor
    """
    axis_num = input_shape[0]
    core_flag = True

    core_axis_list = [11520, 2880, 1280]
    if axis_num < 32:
        core_flag = False
        axis_inner = 1
    if axis_num % 32 == 0:
        core_flag = False
        axis_inner = axis_num // 32
        if axis_num in core_axis_list:
            axis_inner = axis_num // 20

    res = _check_prime_number(axis_num)

    if res == -1 and core_flag:
        axis_inner = axis_num
    elif core_flag:
        axis_inner = res

    total_number = axis_inner * input_shape[1]
    ub_number, block_number = _get_ub_block_num(dtype)
    # return 3 num and 1 bool
    if total_number < ub_number:
        if total_number % block_number == 0:
            return axis_inner, axis_inner, input_shape[
                1], True, "axis_outer_outer"
        if total_number % block_number != 0:
            return axis_inner, axis_inner, input_shape[
                1], False, "axis_outer_outer"
    # total_number > ub_number
    stride = 0 - block_number

    if input_shape[1] > ub_number:
        for i in range(input_shape[1], 1, stride):
            if i < ub_number:
                axis2_inner = i
                break

        cores = axis2_inner % block_number == 0

        return axis_inner, axis_inner, axis2_inner, cores, "axis_two"

    axis_inner2 = -1
    for i in range(axis_inner, 1, stride):
        cur_total = i * input_shape[1]
        if cur_total < ub_number:
            axis_inner2 = i
            break
    cores = (axis_inner2 * input_shape[1]) % block_number == 0

    if axis_inner2 == -1:
        axis_inner2 = axis_inner
        compute_axis = "axis_inner_inner"
    else:
        compute_axis = "axis_inner_outer"

    return axis_inner, axis_inner2, input_shape[1], cores, compute_axis


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=unused-argument,too-many-locals, unused-variable
def _schedule_last_axis(sch, shape, in_data, output, dtype):
    """
    schedule for the last axis situation
    """
    # the four return args is the first axis factor, the first_inner facotr, the second_axis factor
    axis_inner_ft, axis_inner2_ft, axis2_inner_ft, cores, compute_axis = _get_core_num_last_axis(
        shape, dtype)

    axis_outer, axis_inner = sch[output].split(output.op.axis[0],
                                               factor=axis_inner_ft)
    axis_inner_outer, axis_inner_inner = sch[output].split(
        axis_inner, factor=axis_inner2_ft)
    axis_two_outter, axis_two_inner = sch[output].split(output.op.axis[1],
                                                        factor=axis2_inner_ft)

    input_axis_outer, input_axis_inner = sch[in_data].split(
        in_data.op.axis[0], factor=axis_inner_ft)
    input_axis_inner_outer, input_axis_inner_inner = sch[in_data].split(
        input_axis_inner, factor=axis_inner2_ft)
    input_axis_two_outter, input_axis_two_inner = sch[in_data].split(
        in_data.op.axis[1], factor=axis2_inner_ft)

    if compute_axis == "axis_inner_inner":
        sch[in_data].compute_at(sch[output], axis_inner_inner)
        sch[in_data].emit_insn(input_axis_two_inner, insn_cmd.DMA_COPY)  # gm-ub
        sch[output].emit_insn(axis_two_inner, insn_cmd.DMA_COPY)  # ub-gm
    elif compute_axis == "axis_inner_outer":
        sch[in_data].compute_at(sch[output], axis_inner_outer)
        sch[in_data].emit_insn(input_axis_inner_inner, insn_cmd.DMA_COPY)  # gm-ub
        sch[output].emit_insn(axis_inner_inner, insn_cmd.DMA_COPY)  # ub-gm
    elif compute_axis == "axis_two":
        sch[in_data].compute_at(sch[output], axis_two_outter)
        sch[in_data].emit_insn(input_axis_two_inner, insn_cmd.DMA_COPY)  # gm-ub
        sch[output].emit_insn(axis_two_inner, insn_cmd.DMA_COPY)  # ub-gm
    else:
        sch[in_data].compute_at(sch[output], axis_outer)
        sch[in_data].emit_insn(input_axis_inner_inner, insn_cmd.DMA_COPY)  # gm-ub
        sch[output].emit_insn(axis_inner_inner, insn_cmd.DMA_COPY)  # ub-gm

    if cores:
        thread_block = tvm.thread_axis("blockIdx.x")
        sch[output].bind(axis_outer, thread_block)


def _check_last_axis_situation(input_shape, begin, end, strides):
    """
    check the iput args for last_dim_schedule
    """
    result = False
    axis_list = [184320, 46080, 11520, 2880, 1280]
    if len(input_shape) != 2:
        result = False
    elif len(begin) != 2:
        result = False
    elif strides[0] != 1 or strides[1] != 1:
        result = False
    elif begin[1] != 0:
        result = False
    elif input_shape[0] in axis_list:
        if input_shape[1] == 1 or input_shape[1] == 4:
            result = True

    return result


def _get_ub_block_num(dtype):
    """
    get the ub_size for dtype, get the block_size for dtype
    """
    ub_size_bytes = \
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) - 1024
    # Convert byts to Bytes
    dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
    ub_number = ub_size_bytes // dtype_bytes_size
    block_number = 32 // dtype_bytes_size

    return ub_number, block_number


def _get_multicore(input_shape, dtype, split_axis, split_factor):
    """
     judge the input args multicore situation
    """
    length = len(input_shape) - 1
    ub_number, block_number = _get_ub_block_num(dtype)
    result = False
    if split_axis == length:
        last_number = input_shape[length] % split_factor
        if last_number == 0:
            result = split_factor % block_number == 0
        else:
            result = split_factor % block_number == 0 and last_number % block_number == 0
    elif input_shape[length] % block_number == 0:
        result = True
    else:
        result = False

    return result


def _check_tik_branch(sch_input_shape, output_shape, begin, end, strides):
    """
    check last dim
    """
    for i in strides:
        if i != 1:
            return False
    result = True
    sch_input_shape = list(sch_input_shape)

    last_dim = sch_input_shape[len(sch_input_shape) - 1]
    if (len(sch_input_shape) - len(output_shape)) == 1:
        length = len(output_shape)
    elif len(output_shape) == len(sch_input_shape):
        length = len(output_shape) - 1
    else:
        return False

    for i in range(0, length):
        if sch_input_shape[i] != output_shape[i]:
            result = False
            break
    if last_dim == begin[len(begin) - 1]:
        result = False

    return result

# pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-locals
@util.check_input_type(dict, dict, (list, tuple), (list, tuple), (list, tuple),
                       int, int, int, int, int, str)
def strided_slice_d(input_x,
                    output_x,
                    begin,
                    end,
                    strides=None,
                    begin_mask=0,
                    end_mask=0,
                    ellipsis_mask=0,
                    new_axis_mask=0,
                    shrink_axis_mask=0,
                    kernel_name="strided_slice_d"):
    """
    Extracts a strided slice of a tensor (generalized python array indexing).
    Roughly speaking, this op extracts a slice of size (end-begin)/stride
    from the given input_ tensor.
    Starting at the location specified by begin the slice continues
     by adding stride to the index
    until all dimensions are not less than end. Note that a stride
    can be negative, which causes a reverse slice.

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x : dict
        shape and dtype of out
    begin: list.
        represents the index of the first value to select.
    end: list.
        represents the index of the last value to select.
    strides: list or tuple.
        step length to select.
    begin_mask: int
        a bitmask where a bit i being 1 means to ignore the begin
        value and instead use the
        largest interval possible.
    end_mask: int
        analogous to `begin_mask`.
    ellipsis_mask: int
        a bitmask where bit `i` being 1 means the `i`th position
        is actually an ellipsis.
    new_axis_mask: int
        a bitmask where bit `i` being 1 means the `i`th specification creates a
        new shape 1 dimension.
    shrink_axis_mask: int
        a bitmask where bit `i` implies that the `i`th specification
        should shrink the dimensionality.
    kernel_name : str
        cce kernel name, default value is "strided_slice_d"

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "uint8", "bool", "int8")

    util.check_dtype_rule(input_dtype, check_list)
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_shape)
    util.check_shape_size(input_shape, SHAPE_SIZE_LIMIT)

    begin = list(begin)
    end = list(end)

    if not _check_parameter(input_shape, begin, end, strides, ellipsis_mask,
                            new_axis_mask, shrink_axis_mask):
        raise RuntimeError("Parameter Invalid!")

    if strides is None:
        strides = _fill_list_with_ones(len(input_shape))
    else:
        strides = list(strides)

    input_tensor = tvm.placeholder(input_shape,
                                   dtype=input_dtype,
                                   name='input_tensor')

    [output, out_shape] = strided_slice_d_compute(input_tensor,
                                                  output_x,
                                                  begin,
                                                  end,
                                                  strides,
                                                  begin_mask,
                                                  end_mask,
                                                  ellipsis_mask,
                                                  new_axis_mask,
                                                  shrink_axis_mask,
                                                  kernel_name=kernel_name)

    # pylint: disable=locally-disabled,unnecessary-lambda
    out_tensor = tvm.compute(out_shape,
                             lambda *i: output(*i),
                             name='out_tensor',
                             tag='strided_slice_d|3')

    input_size = functools_reduce(lambda x, y: x * y, input_shape[0:])
    out_size = functools_reduce(lambda x, y: x * y, out_shape[0:])

    output_dtype = output_x.get("dtype").lower()
    output_shape = output_x.get("shape")
    if input_size == out_size:
        if output_dtype == "bool":
            input_x["dtype"] = "int8"
            output_x["dtype"] = "int8"
        if len(output_shape) == 0:
            output_x["shape"] = (1,)
        copy_only(input_x, output_x, kernel_name)
        return

    # for RL tune getting res
    fusion_manager.set_op_res(out_tensor)

    ret, sch = rl_bank.query_rl_bank([out_tensor])
    if ret and sch:
        with build_config:
            tvm.build(sch, [input_tensor, out_tensor], "cce", name=kernel_name)
        return

    sch = tvm.create_schedule(out_tensor.op)
    sch[output].set_scope(tbe_platform.scope_ubuf)

    sch_input_shape = []
    for dim in output.shape:
        sch_input_shape.append(dim.value)
    check_result = _check_last_axis_situation(sch_input_shape, begin, end,
                                              strides)
    if check_result:
        _schedule_last_axis(sch, sch_input_shape, output, out_tensor,
                            input_dtype)
        with build_config:
            tvm.build(sch, [input_tensor, out_tensor], "cce", name=kernel_name)
        return

    if _check_tik_branch(input_shape, output_shape, begin, end, strides):
        begin_shape = copy.deepcopy(begin)
        end_shape = copy.deepcopy(end)
        stride_shape = list(strides)
        stride_shape = copy.deepcopy(stride_shape)
        input_list = list(input_shape)
        # update begin_shape, end_shape
        begin_shape, end_shape, stride_shape = _init_parameter(input_list, begin_shape, end_shape,
                                                               stride_shape, begin_mask, end_mask,
                                                               ellipsis_mask, new_axis_mask,
                                                               shrink_axis_mask)
        head_size = 1
        for i in range(0, (len(input_shape) - 1)):
            head_size = head_size * input_shape[i]
        if input_dtype == "float32" and input_shape[-1] == 2 and \
           begin_shape[len(begin_shape) - 1] == 0  and end_shape[len(begin_shape) - 1] == 1 \
           and head_size > 128:
            strided_slice_two_turn_one(input_x, output_x, kernel_name)
            return
        if input_list[-1] == 85 and output_shape[-1] == 80:
            strided_slice_last_dim_only(input_shape, input_dtype,
                                        output_shape, begin_shape,
                                        kernel_name)
            return
        res = strided_slice_last_dim(input_shape, input_dtype,
                                     output_shape, begin_shape,
                                     end_shape, stride_shape,
                                     kernel_name)
        if res:
            return
        else:
            res1 = strided_slice_last_dim_one(input_shape, input_dtype,
                                              output_shape, begin_shape,
                                              kernel_name)
            if res1:
                return

    split_axis, split_factor = _tilling_axis(out_shape, dtype=input_dtype)
    core_state = _get_multicore(out_shape, input_dtype, split_axis,
                                split_factor)
    axis_outer, axis_inner = sch[out_tensor].split(
        out_tensor.op.axis[split_axis], factor=split_factor)

    if split_axis == 0:
        core_num = _get_target_core_num(out_shape[split_axis] // split_factor)
        axis_outer_outer, axis_outer_inter = sch[out_tensor].split(
            axis_outer, nparts=core_num)
    else:
        core_num = _get_target_core_num(out_shape[0])
        axis_outer_outer, axis_outer_inter = sch[out_tensor].split(
            out_tensor.op.axis[0], nparts=core_num)
        for i in range(1, split_axis):
            axis_outer_inter = sch[out_tensor].fuse(axis_outer_inter,
                                                    out_tensor.op.axis[i])
        axis_outer_inter = sch[out_tensor].fuse(axis_outer_inter, axis_outer)

    sch[output].compute_at(sch[out_tensor], axis_outer_inter)

    sch[output].emit_insn(output.op.axis[0], insn_cmd.DMA_COPY)  # gm-ub
    if len(out_shape) >= 2:
        # Convert bytes to Bytes
        dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(input_dtype) // 8
        # 32 means one block size(32 Bytes), divide by 32 to
        # get the numbers of data that
        # can be stored in one block.
        element = 32 // dtype_bytes_size
        align_axis = _get_align_axis(out_shape)
        sch[output].storage_align(output.op.axis[align_axis], element, 0)

    if core_state:
        thread_block = tvm.thread_axis("blockIdx.x")
        sch[out_tensor].bind(axis_outer_outer, thread_block)

    sch[out_tensor].emit_insn(axis_inner, insn_cmd.DMA_COPY)  # ub-gm

    with build_config:
        tvm.build(sch, [input_tensor, out_tensor], "cce", name=kernel_name)
