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

# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def strided_slice_last_dim_only(input_shape, dtype, output_shape, begin, kernel_name):
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

    ub_number, type_block_number = _get_ub_block_num()

    if dtype == "float16":
        expect_cycle = 16
        src_stride = 80
        dst_stride = 75
        one_repeat_block_number = 80
        expect_last_dim = 96
        scalar = tik_instance.Scalar(dtype=dtype, name="scalar")
        scalar.set_as(1.0)
    elif dtype == "float32":
        expect_cycle = 8
        src_stride = 75
        dst_stride = 70
        one_repeat_block_number = 64
        expect_last_dim = 88
        scalar = tik_instance.Scalar(dtype=dtype, name="scalar")
        scalar.set_as(1.0)
    else:
        return False

    input_size = 1
    for i in input_shape:
        input_size = input_size * i
    output_size = 1
    output_length = len(output_shape)
    for i in range(0, output_length):
        output_size = output_size * output_shape[i]

    output_list = list(output_shape)
    input_list = list(input_shape)
    consecutive = output_list[len(output_shape) - 1]
    len_burst = input_list[len(input_shape) - 1]
    start = begin[len(begin) - 1]

    for i in range(0, ub_number):
        input_size_one = ub_number - i
        if input_size_one % len_burst == 0:
            if (input_size_one // len_burst) % expect_cycle == 0:
                max_data_num = i
                break

    max_input_size = ub_number - max_data_num

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

    for j in range(0, tail_len_burst):
        input_len_stride_one = input_len_stride - j
        can_bound_input_size = input_len_stride_one * (core_num - 1) * len_burst
        if input_len_stride_one % (type_block_number) == 0:
            if can_bound_input_size <= input_size:
                add_num = j
                break

    input_len_stride_two = input_len_stride - add_num
    max_ub_stride = max_input_size // len_burst
    max_move_one = max_ub_stride // expect_cycle
    input_pre_data = input_len_stride_two * len_burst
    output_ub_size = consecutive * input_len_stride_two
    output_length_stride = output_ub_size // type_block_number
    tail_input_data = input_size - (core_num - 1) * input_pre_data
    tail_input_stride_two = tail_input_data // len_burst

    if input_len_stride_two > max_ub_stride:
        loop_index = input_len_stride_two // max_ub_stride
        tail_data_ub_index = input_len_stride_two % max_ub_stride
        tail_data_ub_one = input_len_stride_two - loop_index * max_ub_stride
        tail_move_one = tail_data_ub_one // type_block_number
        tail_output_data = tail_move_one * expect_cycle * consecutive
        tail_output_stride_length = tail_output_data // type_block_number

    if tail_input_stride_two > max_ub_stride:
        tail_loop_index = tail_input_stride_two // max_ub_stride
        tail_data_index = tail_input_stride_two % max_ub_stride
        tail_last_move_one = tail_data_index // type_block_number

        tail_last_output_data = tail_last_move_one * expect_cycle * consecutive
        tail_last_output_stride_length = tail_last_output_data // type_block_number
        tail_one_last_move = tail_data_index - tail_last_move_one * expect_cycle

    if input_len_stride_two <= max_ub_stride:
        one_move = input_len_stride_two // type_block_number

    if tail_input_stride_two <= max_ub_stride:
        tail_one_move = tail_input_stride_two // expect_cycle
        tail_one_last_move = tail_input_stride_two - tail_one_move * expect_cycle
        tail_one_move_stride = (expect_cycle * tail_one_move) * consecutive // type_block_number

    input_data = tik_instance.Tensor(dtype, input_shape,
                                     name="input_data", scope=tik.scope_gm)
    output_data = tik_instance.Tensor(dtype, output_shape,
                                      name="output_data", scope=tik.scope_gm)
    if tail_one_last_move == 0:
        input_ub_data = tik_instance.Tensor(dtype, (max_input_size, ),
                                            name="input_ub_data", scope=tik.scope_ubuf)
    else:
        ub_block_dim_one = expect_last_dim // type_block_number
        input_ub_data = tik_instance.Tensor(dtype, (max_input_size, ),
                                            name="input_ub_data", scope=tik.scope_ubuf)
        output_ub_data = tik_instance.Tensor(dtype, (tail_one_last_move * consecutive, ),
                                             name="output_ub_data", scope=tik.scope_ubuf)

    ub_block_dim = consecutive // type_block_number
    max_input_output_length_stride = \
        (max_move_one * expect_cycle * consecutive) // type_block_number

    if input_len_stride_two <= max_ub_stride and tail_input_stride_two <= max_ub_stride:
        with tik_instance.for_range(0, core_num, block_num=core_num) as total_cycle:
            with tik_instance.if_scope(total_cycle < core_num -1):
                with tik_instance.for_range(0, expect_cycle) as group:
                    tik_instance.data_move(input_ub_data[group * consecutive],
                                           input_data[total_cycle * input_pre_data
                                                      + group * len_burst+start],
                                           0, one_move, ub_block_dim, src_stride, dst_stride)
                tik_instance.data_move(output_data[total_cycle * output_ub_size],
                                       input_ub_data, 0, 1, output_length_stride, 0, 0)
            with tik_instance.if_scope(total_cycle == core_num - 1):
                with tik_instance.for_range(0, expect_cycle) as group:
                    if tail_one_move != 0:
                        tik_instance.data_move(input_ub_data[group * consecutive],
                                               input_data[(core_num - 1) * input_pre_data
                                                          + group * len_burst + start],
                                               0, tail_one_move, ub_block_dim,
                                               src_stride, dst_stride)
                if tail_one_move_stride != 0:
                    tik_instance.data_move(output_data[(core_num - 1) * output_ub_size],
                                           input_ub_data, 0, 1, tail_one_move_stride, 0, 0)
                if tail_one_last_move:
                    with tik_instance.for_range(0, tail_one_last_move) as group:
                        tik_instance.data_move(input_ub_data[group * expect_last_dim],
                                               input_data[total_cycle * input_pre_data +
                                                          expect_cycle * tail_one_move * len_burst
                                                          + (group * len_burst) + start],
                                               0, 1, ub_block_dim_one, 0, 0)
                    if dtype == "float32":
                        tik_instance.vmuls(one_repeat_block_number, output_ub_data,
                                           input_ub_data, scalar, tail_one_last_move, 1, 1, 10, 11)
                        tik_instance.vmuls(16, output_ub_data[64], input_ub_data[64],
                                           scalar, tail_one_last_move, 1, 1, 10, 11)
                    elif dtype == "float16":
                        tik_instance.vmuls(one_repeat_block_number, output_ub_data,
                                           input_ub_data, scalar, tail_one_last_move, 1, 1, 5, 6)
                    tik_instance.data_move(output_data[(core_num - 1) * output_ub_size
                                                       + tail_one_move_stride * type_block_number],
                                           output_ub_data, 0, 1,
                                           tail_one_last_move * consecutive // type_block_number,
                                           0, 0)

    if input_len_stride_two <= max_ub_stride and tail_input_stride_two > max_ub_stride:
        with tik_instance.for_range(0, core_num, block_num=core_num) as total_cycle:
            with tik_instance.if_scope(total_cycle < core_num -1):
                with tik_instance.for_range(0, expect_cycle) as group:
                    tik_instance.data_move(input_ub_data[group * consecutive],
                                           input_data[total_cycle * input_pre_data
                                                      + group * len_burst+start],
                                           0, one_move, ub_block_dim, src_stride, dst_stride)
                tik_instance.data_move(output_data[total_cycle * output_ub_size],
                                       input_ub_data, 0, 1, output_length_stride, 0, 0)

        with tik_instance.if_scope(total_cycle == core_num - 1):
            with tik_instance.for_range(0, tail_loop_index) as loop:
                with tik_instance.for_range(0, expect_cycle) as group:
                    tik_instance.data_move(input_ub_data[group * consecutive],
                                           input_data[(core_num - 1) * input_pre_data
                                                      + loop * max_input_size
                                                      + group * len_burst + start],
                                           0, max_move_one, ub_block_dim, src_stride, dst_stride)
                tik_instance.data_move(output_data[(core_num - 1) * output_ub_size
                                                   + loop * max_ub_stride * consecutive],
                                       input_ub_data, 0, 1,
                                       max_input_output_length_stride, 0, 0)
            if tail_data_index:
                with tik_instance.for_range(0, expect_cycle) as group:
                    tik_instance.data_move(input_ub_data[group * consecutive],
                                           input_data[(core_num - 1) * input_pre_data
                                                      + tail_loop_index * max_input_size
                                                      + group * len_burst + start],
                                           0, tail_last_move_one,
                                           ub_block_dim, src_stride, dst_stride)
                tik_instance.data_move(output_data[(core_num - 1) * output_ub_size
                                                   + tail_loop_index * max_ub_stride * consecutive],
                                       input_ub_data, 0, 1,
                                       tail_last_output_stride_length, 0, 0)
                if tail_one_last_move:
                    with tik_instance.for_range(0, tail_one_last_move) as group:
                        tik_instance.data_move(input_ub_data[group * expect_last_dim],
                                               input_data[(core_num - 1) * input_pre_data
                                                          + tail_loop_index * max_input_size
                                                          + expect_cycle
                                                          * len_burst * tail_last_move_one
                                                          + group * len_burst + start],
                                               0, 1, ub_block_dim_one, 0, 0)
                        if dtype == "float32":
                            tik_instance.vmuls(one_repeat_block_number, output_ub_data,
                                               input_ub_data, scalar, tail_one_last_move,
                                               1, 1, 10, 11)
                            tik_instance.vmuls(16, output_ub_data[64], input_ub_data[64],
                                               scalar, tail_one_last_move, 1, 1, 10, 11)
                        elif dtype == "float16":
                            tik_instance.vmuls(one_repeat_block_number, output_ub_data,
                                               input_ub_data, scalar, tail_one_last_move,
                                               1, 1, 5, 6)
                        tik_instance.data_move(output_data[(core_num - 1) * output_ub_size
                                                           + tail_loop_index * max_ub_stride * consecutive
                                                           + tail_last_output_stride_length * type_block_number],
                                               output_ub_data, 0, 1,
                                               tail_one_last_move * consecutive // type_block_number,
                                               0, 0)

    if input_len_stride_two > max_ub_stride and tail_input_stride_two > max_ub_stride:
        with tik_instance.for_range(0, core_num, block_num=core_num) as total_cycle:
            with tik_instance.if_scope(total_cycle < core_num -1):
                with tik_instance.for_range(0, loop_index) as loop:
                    with tik_instance.for_range(0, expect_cycle) as group:
                        tik_instance.data_move(input_ub_data[group * consecutive],
                                               input_data[total_cycle * input_pre_data
                                                          + loop * max_input_size
                                                          + group * len_burst + start],
                                               0, max_move_one, ub_block_dim, src_stride, dst_stride)
                    tik_instance.data_move(output_data[total_cycle * output_ub_size +
                                                       loop * max_ub_stride * consecutive],
                                           input_ub_data, 0, 1,
                                           max_input_output_length_stride, 0, 0)

                if tail_data_ub_index:
                    with tik_instance.for_range(0, expect_cycle) as group:
                        tik_instance.data_move(input_ub_data[group * consecutive],
                                               input_data[total_cycle * input_pre_data
                                                          + loop_index * max_input_size
                                                          + group * len_burst + start],
                                               0, tail_move_one, ub_block_dim, src_stride, dst_stride)
                    tik_instance.data_move(output_data[total_cycle * output_ub_size
                                                       + loop_index * max_ub_stride * consecutive],
                                           input_ub_data, 0, 1,
                                           tail_output_stride_length, 0, 0)

            with tik_instance.if_scope(total_cycle == core_num - 1):
                with tik_instance.for_range(0, tail_loop_index) as loop:
                    with tik_instance.for_range(0, expect_cycle) as group:
                        tik_instance.data_move(input_ub_data[group * consecutive],
                                               input_data[(core_num - 1) * input_pre_data
                                                          + loop * max_input_size
                                                          + group * len_burst + start],
                                               0, max_move_one, ub_block_dim,
                                               src_stride, dst_stride)
                    tik_instance.data_move(output_data[(core_num - 1) * output_ub_size
                                                       + loop * max_ub_stride * consecutive],
                                           input_ub_data, 0, 1,
                                           max_input_output_length_stride, 0, 0)
                if tail_data_index:
                    with tik_instance.for_range(0, expect_cycle) as group:
                        tik_instance.data_move(input_ub_data[group * consecutive],
                                               input_data[(core_num - 1) * input_pre_data
                                                          + tail_loop_index * max_input_size
                                                          + group * len_burst + start],
                                               0, tail_last_move_one, ub_block_dim, src_stride, dst_stride)
                    tik_instance.data_move(output_data[(core_num - 1) * output_ub_size
                                                       + tail_loop_index * max_ub_stride * consecutive],
                                           input_ub_data, 0, 1,
                                           tail_last_output_stride_length, 0, 0)
                    if tail_one_last_move:
                        with tik_instance.for_range(0, tail_one_last_move) as group:
                            tik_instance.data_move(input_ub_data[group * expect_last_dim],
                                                   input_data[(core_num - 1) * input_pre_data
                                                              + tail_loop_index * max_input_size +
                                                              expect_cycle * len_burst * tail_last_move_one
                                                              + group * len_burst + start],
                                                   0, 1, ub_block_dim_one, 0, 0)
                            if dtype == "float32":
                                tik_instance.vmuls(one_repeat_block_number, output_ub_data,
                                                   input_ub_data, scalar, tail_one_last_move,
                                                   1, 1, 10, 11)
                                tik_instance.vmuls(16, output_ub_data[64], input_ub_data[64],
                                                   scalar, tail_one_last_move, 1, 1, 10, 11)
                            elif dtype == "float16":
                                tik_instance.vmuls(one_repeat_block_number, output_ub_data,
                                                   input_ub_data, scalar, tail_one_last_move,
                                                   1, 1, 5, 6)
                            tik_instance.data_move(output_data[(core_num - 1) * output_ub_size
                                                   + tail_loop_index * max_ub_stride * consecutive +
                                                   tail_last_output_stride_length * type_block_number],
                                                   output_ub_data, 0, 1,
                                                   tail_one_last_move * consecutive // type_block_number,
                                                   0, 0)

    if input_len_stride_two > max_ub_stride and tail_input_stride_two <= max_ub_stride:
        with tik_instance.for_range(0, core_num, block_num=core_num) as total_cycle:
            with tik_instance.if_scope(total_cycle < core_num -1):
                with tik_instance.for_range(0, loop_index) as loop:
                    with tik_instance.for_range(0, expect_cycle) as group:
                        tik_instance.data_move(input_ub_data[group * consecutive],
                                               input_data[total_cycle * input_pre_data
                                                          + loop * max_input_size
                                                          + group * len_burst + start],
                                               0, max_move_one, ub_block_dim,
                                               src_stride, dst_stride)
                    tik_instance.data_move(output_data[total_cycle * output_ub_size
                                                       + loop * max_ub_stride * consecutive],
                                           input_ub_data, 0, 1,
                                           max_input_output_length_stride, 0, 0)
                if tail_data_ub_index:
                    with tik_instance.for_range(0, expect_cycle) as group:
                        tik_instance.data_move(input_ub_data[group * consecutive],
                                               input_data[total_cycle * input_pre_data
                                                          + loop_index * max_input_size
                                                          + group * len_burst+start],
                                               0, tail_move_one,
                                               ub_block_dim, src_stride, dst_stride)
                    tik_instance.data_move(output_data[total_cycle * output_ub_size
                                                       + loop_index * max_ub_stride * consecutive],
                                           input_ub_data, 0, 1,
                                           tail_output_stride_length, 0, 0)

            with tik_instance.if_scope(total_cycle == core_num - 1):
                with tik_instance.for_range(0, expect_cycle) as group:
                    tik_instance.data_move(input_ub_data[group * consecutive],
                                           input_data[(core_num - 1) * input_pre_data
                                                      + group * len_burst + start],
                                           0, tail_one_move, ub_block_dim,
                                           src_stride, dst_stride)
                tik_instance.data_move(output_data[(core_num - 1) * output_ub_size],
                                       input_ub_data, 0, 1,
                                       tail_one_move_stride, 0, 0)
                if tail_one_last_move:
                    with tik_instance.for_range(0, tail_one_last_move) as group:
                        tik_instance.data_move(input_ub_data[group * expect_last_dim],
                                               input_data[total_cycle * input_pre_data +
                                                          expect_cycle * tail_one_move * len_burst
                                               + (group * len_burst) + start],
                                               0, 1, ub_block_dim_one, 0, 0)
                    if dtype == "float32":
                        tik_instance.vmuls(one_repeat_block_number, output_ub_data,
                                           input_ub_data, scalar, tail_one_last_move,
                                           1, 1, 10, 11)
                        tik_instance.vmuls(16, output_ub_data[64], input_ub_data[64],
                                           scalar, tail_one_last_move, 1, 1, 10, 11)
                    if dtype == "float16":
                        tik_instance.vmuls(one_repeat_block_number, output_ub_data,
                                           input_ub_data, scalar, tail_one_last_move,
                                           1, 1, 5, 6)
                    tik_instance.data_move(output_data[(core_num - 1) * output_ub_size
                                           + tail_one_move_stride * type_block_number],
                                           output_ub_data,
                                           0, 1,
                                           tail_one_last_move * consecutive // type_block_number,
                                           0, 0)

    tik_instance.BuildCCE(kernel_name, inputs=[input_data], outputs=[output_data], enable_l2=False)
    return tik_instance
