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

def strided_slice_only_last_dim(input_shape, dtype, output_shape, begin, kernel_name):
    tik_instance = tik.Tik()
    aicore_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
    def _get_ub_block_num():
        """
        get the ub_size for dtype, get the block_size for dtype
        """
        ub_size_bytes = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) - 8192
        # Convert byts to Bytes
        dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
        ub_number = ub_size_bytes // dtype_bytes_size
        block_number = 32 // dtype_bytes_size

        return ub_number, block_number

    ub_number,type_block_number = _get_ub_block_num()

    input_size = 1
    for i in input_shape:
        input_size = input_size * i
    if dtype == "float16":
        one_repeats_block_number = 128
        expect_last_dim = 96
    elif dtype == "float32":
        one_repeats_block_number = 64
        expect_last_dim = 88
    else:
        return False

    output_size = 1
    output_length = len(output_shape)
    for i in range(0, output_length):
        output_size = output_size * output_shape[i]

    output_list = list(output_shape)
    input_list = list(input_shape)
    consecutive = output_list[len(output_shape) - 1]
    len_burst = input_list[len(input_shape) - 1]
    start = begin[len(begin) - 1]

    tail_len_burst = input_size // len_burst
    for i in range(1, tail_len_burst):
        tail_len_burst_one = len_burst * i
        if input_size % tail_len_burst_one == 0:
            core_num = input_size // tail_len_burst_one
        else:
            core_num = (input_size // tail_len_burst_one) + 1
        if core_num <= aicore_num:
            input_len_stride = i
            break

    ub_block_dim = expect_last_dim // type_block_number

    input_ub_size = expect_last_dim * input_len_stride
    output_ub_size = consecutive * input_len_stride

    output_length_stride  = output_ub_size // type_block_number
    input_ub_len_stride = input_ub_size // type_block_number

    scalar = 1.0

    tail_input_data = input_size - (core_num - 1) * tail_len_burst_one
    input_tail_stride = tail_input_data // len_burst
    tail_input_size = input_tail_stride * expect_last_dim
    tail_output_size = input_tail_stride * consecutive
    tail_output_stride_length = tail_output_size // type_block_number

    if input_len_stride > 255:
        return False

    input_data = tik_instance.Tensor(dtype, (input_size, ),
                                     name="input_data", scope = tik.scope_gm)
    output_data = tik_instance.Tensor(dtype, (output_size, ),
                                      name="output_data", scope = tik.scope_gm)
    input_ub_data = tik_instance.Tensor(dtype, (input_ub_size, ),
                                        name="input_ub_data", scope = tik.scope_ubuf)
    output_ub_data = tik_instance.Tensor(dtype, (output_ub_size, ),
                                         name="output_ub_data", scope=tik.scope_ubuf)

    if dtype == "float32":
        with tik_instance.for_range(0, core_num, block_num=core_num) as total_cycle:
            with tik_instance.if_scope(total_cycle < core_num -1):
                with tik_instance.for_range(0, input_len_stride) as group:
                    tik_instance.data_move(input_ub_data[group * expect_last_dim],
                                           input_data[(total_cycle * tail_len_burst_one)
                                                      + (group * len_burst) + start], 0, 1, ub_block_dim, 0, 0)
                tik_instance.vmuls(64, output_ub_data, input_ub_data, scalar, input_len_stride, 1, 1, 10, 11)
                tik_instance.vmuls(16, output_ub_data[64], input_ub_data[64], scalar, input_len_stride, 1, 1, 10, 11)
                tik_instance.data_move(output_data[total_cycle * output_ub_size],
                                       output_ub_data, 0, 1, output_length_stride, 0, 0)

        with tik_instance.if_scope(total_cycle == core_num -1):
            with tik_instance.for_range(0, input_tail_stride) as group1:
                tik_instance.data_move(input_ub_data[group1 * expect_last_dim],
                                      input_data[(core_num-1) * tail_len_burst_one
                                                 + (group1 * len_burst) + start], 0, 1, ub_block_dim, 0, 0)
            tik_instance.vmuls(64, output_ub_data, input_ub_data, scalar, input_tail_stride, 1, 1, 10, 11)
            tik_instance.vmuls(16, output_ub_data[64], input_ub_data[64], scalar, input_tail_stride, 1, 1, 10, 11)
            tik_instance.data_move(output_data[(core_num-1) * output_length_stride * type_block_number],
                                   output_ub_data, 0, 1, tail_output_stride_length, 0, 0)

    if dtype == "float16":
        with tik_instance.for_range(0, core_num, block_num=core_num) as total_cycle:
            with tik_instance.if_scope(total_cycle < core_num -1):
                with tik_instance.for_range(0, input_len_stride) as group:
                    tik_instance.data_move(input_ub_data[group * expect_last_dim],
                                           input_data[(total_cycle * tail_len_burst_one)
                                                      + (group * len_burst) + start], 0, 1, ub_block_dim, 0, 0)
                tik_instance.vmuls(80, output_ub_data, input_ub_data, scalar, input_len_stride, 1, 1, 5, 6)
                tik_instance.data_move(output_data[total_cycle * output_ub_size],
                                       output_ub_data, 0, 1, output_length_stride, 0, 0)

        with tik_instance.if_scope(total_cycle == core_num -1):
            with tik_instance.for_range(0, input_tail_stride) as group1:
                tik_instance.data_move(input_ub_data[group1 * expect_last_dim],
                                       input_data[(core_num - 1) * tail_len_burst_one
                                                  + (group1 * len_burst) + start], 0, 1, ub_block_dim, 0, 0)
            tik_instance.vmuls(80, output_ub_data, input_ub_data, scalar, input_tail_stride, 1, 1, 5, 6)
            tik_instance.data_move(output_data[(core_num - 1) * output_length_stride * type_block_number],
                                   output_ub_data, 0, 1, tail_output_stride_length, 0, 0)

    tik_instance.BuildCCE(kernel_name, inputs=[input_data], outputs=[output_data])
    return tik_instance
