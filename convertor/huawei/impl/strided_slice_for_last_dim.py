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

NPUClearFloatStatus
"""

from te import tik
from te import platform as tbe_platform

# constant 8
NUM_EIGHT = 8


# pylint: disable=invalid-name, too-many-locals, unused-argument
# pylint: disable=too-many-arguments, unused-variable, too-many-return-statements
# pylint: disable=too-many-branches, too-many-statements
def strided_slice_last_dim(input_shape, dtype, output_shape, begin, end, stride, kernel_name):
    """
    the main function of npu_clear_float_status

    Parameters
    ----------
    addr: dict,shape and datatype,datatype supports float32
    data: dict,shape and datatype,datatype supports float32
    kernel_name: cce kernel name, default value is "n_p_u_clear_float_status"

    Returns
    -------
    tik_instance: tik_instance
    """
    tik_instance = tik.Tik()
    aicore_num = \
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
    input_size = 1
    for i in input_shape:
        input_size = input_size * i
    if dtype == "float16":
        type_block_num = 16
    elif dtype == "float32":
        type_block_num = 8
    else:
        return False

    output_size = 1
    output_length = len(output_shape)
    for i in range(0, output_length):
        output_size = output_size * output_shape[i]

    if len(output_shape) != len(input_shape):
        consecutive_num = 1
    else:
        consecutive_num = output_shape[len(output_shape) - 1]

    ouput_core_data_num = output_size // aicore_num  #
    input_core_data_num = input_size // aicore_num  #

    output_group_1 = ouput_core_data_num  #
    input_group_1 = ouput_core_data_num // consecutive_num * input_shape[len(input_shape) - 1]  #
    output_group_2 = 0  #
    input_group_2 = 0  #

    tail_core_num = 0  #
    total_core_num = aicore_num  #
    tail_flag = False
    if output_size % aicore_num != 0 or ouput_core_data_num % type_block_num != 0:
        aicore_num = aicore_num - 1
        ouput_core_data_num = output_size // aicore_num
        input_core_data_num = input_size // aicore_num
        if output_size % aicore_num != 0 or ouput_core_data_num % type_block_num != 0:
            if output_size // aicore_num == 0:
                aicore_num = 1
            output_group_1 = ouput_core_data_num
            output_group_1 = (output_group_1 // type_block_num) * type_block_num

            output_group_2 = output_size - (output_group_1 * aicore_num)
            input_group_1 = output_group_1 // consecutive_num * input_shape[len(input_shape) - 1]
            input_group_2 = input_size - (input_group_1 * aicore_num)

            tail_core_num = 1  #
            total_core_num = aicore_num + tail_core_num
            tail_flag = True
        else:
            output_group_1 = ouput_core_data_num  #
            input_group_1 = ouput_core_data_num // consecutive_num \
                            * input_shape[len(input_shape) - 1]  #
            output_group_2 = 0  #
            input_group_2 = 0  #
            total_core_num = aicore_num  #

    def _get_ub_block_num():
        """
        get the ub_size for dtype, get the block_size for dtype
        """
        ub_size_bytes = \
            tbe_platform.cce_conf.get_soc_spec(
                tbe_platform.cce_conf.UB_SIZE) - 1024
        # Convert byts to Bytes
        dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
        ub_number = ub_size_bytes // dtype_bytes_size
        block_number = 32 // dtype_bytes_size

        return ub_number, block_number

    def _get_split_axis(input_size, output_size):
        ub_number, block_number = _get_ub_block_num()
        total_num = (input_size + output_size)
        result = 1
        find_flag = False
        for result in range(1, output_size):
            if (total_num // result) <= ub_number and (output_size % result) == 0 and (
                    (output_size // result) % type_block_num) == 0 and \
                    (output_size // consecutive_num) % result == 0:
                find_flag = True
                break

        return result, find_flag

    # internal split factor
    split_factor_group_1, find_flag_1 = _get_split_axis(input_group_1, output_group_1)

    split_factor_group_2, find_flag_2 = _get_split_axis(input_group_2, output_group_2)

    if not find_flag_1:
        return False

    if split_factor_group_2 != 1:
        return False

    if split_factor_group_2 != 1:
        return False

    if output_group_1 % consecutive_num != 0 or output_group_2 % consecutive_num != 0:
        return False

    if input_group_1 % input_shape[len(input_shape) - 1] != 0:
        return False

    if input_group_2 % input_shape[len(input_shape) - 1] != 0:
        return False

    if input_group_1 == 0 or output_group_1 == 0:
        return False

    if consecutive_num > 100:
        return False

    # can't change
    start_num = begin[len(begin) - 1]
    # can't change
    len_burst = input_shape[len(input_shape) - 1]

    # gm_size
    output_data = tik_instance.Tensor(dtype, (output_size,), name="output_data",
                                      scope=tik.scope_gm)
    input_data = tik_instance.Tensor(dtype, (input_size,), name="input_data",
                                     scope=tik.scope_gm)

    input_ub_size = 0
    output_ub_size = 0

    if input_group_2 > input_group_1:
        input_ub_size = ((input_group_2 + type_block_num - 1) // type_block_num) \
                        * type_block_num
        output_ub_size = ((output_group_2 + type_block_num - 1) // type_block_num) \
                         * type_block_num
    else:
        input_ub_size = input_size // aicore_num // split_factor_group_1
        output_ub_size = output_size // aicore_num // split_factor_group_1

    # ub_size change
    input_data_ub = tik_instance.Tensor(dtype,
                                        (input_ub_size,),
                                        name="input_data_ub",
                                        scope=tik.scope_ubuf)
    # ub_size change
    output_data_ub = tik_instance.Tensor(dtype,
                                         (output_ub_size,),
                                         name="output_data_ub",
                                         scope=tik.scope_ubuf)

    if (output_group_2 % type_block_num) == 0:
        group_2_length = output_group_2 // type_block_num
    else:
        group_2_length = (output_group_2 // type_block_num) + 1

    if (input_group_2 % type_block_num) == 0:
        group_2_input_length = input_group_2 // type_block_num
    else:
        group_2_input_length = (input_group_2 // type_block_num) + 1

    with tik_instance.for_range(0, total_core_num,
                                block_num=total_core_num) as total_cycle:
        with tik_instance.if_scope(total_cycle < aicore_num):
            with tik_instance.for_range(0, split_factor_group_1)as axis_outer:
                if (input_group_1 % type_block_num) == 0:
                    stride_length = input_group_1 // type_block_num
                else:
                    stride_length = (input_group_1 // type_block_num) + 1
                stride_length = stride_length // split_factor_group_1
                tik_instance.data_move(input_data_ub,
                                       input_data[(total_cycle * input_group_1)
                                                  + (axis_outer * input_group_1
                                                     // split_factor_group_1)],
                                       0, 1,
                                       stride_length,
                                       0, 0)
                max_num = output_group_1 // consecutive_num \
                          // split_factor_group_1
                with tik_instance.for_range(0, max_num)as group:
                    for cur_num in range(0, consecutive_num):
                        output_data_ub[group * consecutive_num + cur_num].set_as(
                            input_data_ub[group * len_burst + start_num + cur_num])
                gm_deviation = axis_outer * output_group_1 \
                               // split_factor_group_1
                output_data_src1 = total_cycle * output_group_1 + gm_deviation

                if (output_group_1 % type_block_num) == 0:
                    output_stride_length = output_group_1 // type_block_num
                else:
                    output_stride_length = (output_group_1 // type_block_num) + 1

                output_data_burlen = output_stride_length // split_factor_group_1
                tik_instance.data_move(output_data[output_data_src1],
                                       output_data_ub, 0, 1,
                                       output_data_burlen,
                                       0, 0)
        if tail_flag:
            with tik_instance.else_scope():
                with tik_instance.for_range(0, split_factor_group_2)as axis_outer:
                    input_deviation = axis_outer * input_group_2 \
                                      // split_factor_group_2
                    stride_input_dev = group_2_input_length \
                                       // split_factor_group_2
                    tik_instance.data_move(input_data_ub, input_data[
                        (aicore_num * input_group_1) + input_deviation],
                                           0, 1, stride_input_dev, 0, 0)
                    max_num = output_group_2 // consecutive_num \
                              // split_factor_group_2
                    with tik_instance.for_range(0, max_num)as group:
                        for cur_num in range(0, consecutive_num):
                            output_data_ub[group * consecutive_num + cur_num].set_as(
                                input_data_ub[group * len_burst + start_num + cur_num])
                    tik_instance.data_move(output_data[aicore_num * output_group_1 + (
                        axis_outer * output_group_2 // split_factor_group_2)], output_data_ub,
                                           0, 1, group_2_length, 0, 0)

    tik_instance.BuildCCE(kernel_name, inputs=[input_data], outputs=[output_data])

    return tik_instance
