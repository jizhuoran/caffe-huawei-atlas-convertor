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

# pylint: disable=invalid-name, too-many-locals, unused-argument
# pylint: disable=too-many-arguments, unused-variable, too-many-return-statements
# pylint: disable=too-many-branches, too-many-statements
def strided_slice_last_dim_one(input_shape, dtype, output_shape, begin, kernel_name):
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

    ub_number, type_block_num = _get_ub_block_num()
    max_ub_number = ub_number // 2

    input_len = len(input_shape)
    output_len = len(output_shape)
    if input_len != output_len:
        return False
    for i in range(0, input_len-1):
        if input_shape[i] != output_shape[i]:
            return False
    if input_len == 1 or output_len == 0:
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

    start_num = begin[len(begin) - 1]

    type_block_num_one = type_block_num
    len_burst_one = len_burst * type_block_num_one
    if input_size % len_burst_one == 0:
        core_num = input_size // len_burst_one
    else:
        core_num = (input_size // len_burst_one) + 1

    if core_num > aicore_num:
        for i in range(2, input_size):
            type_block_num_two = type_block_num * i
            len_burst_one = len_burst * type_block_num_two
            if input_size % len_burst_one == 0:
                core_num = input_size // len_burst_one
            else:
                core_num = (input_size // len_burst_one) + 1
            if core_num <= aicore_num:
                break

    input_len_burst = len_burst_one // type_block_num
    max_num = len_burst_one // len_burst
    input_ub_len_one = consecutive * max_num

    tail_num = input_size - len_burst_one*(core_num - 1)
    max_num_one = tail_num // len_burst
    output_tail_size = max_num_one*consecutive
    max_ub_data = input_ub_len_one + len_burst_one
    last_max_ub_data = tail_num + output_tail_size

    if max_ub_data <= ub_number and last_max_ub_data <= ub_number:
        input_ub_data = tik_instance.Tensor(dtype, (len_burst_one,),
                                            name="input_ub_data", scope=tik.scope_ubuf)
        input_ub_data_one = tik_instance.Tensor(dtype, (input_ub_len_one,),
                                                name="input_ub_data_one", scope=tik.scope_ubuf)
        input_len_burst = len_burst_one // type_block_num
        input_ub_length = input_ub_len_one // type_block_num
        if tail_num % type_block_num == 0:
            tail_len_burst = tail_num // type_block_num
        else:
            tail_len_burst = (tail_num // type_block_num) + 1
        if output_tail_size % type_block_num == 0:
            tail_len_output_burst = output_tail_size // type_block_num
        else:
            tail_len_output_burst = (output_tail_size // type_block_num) + 1

    if max_ub_data > ub_number:
        if input_ub_len_one <= max_ub_number and len_burst_one > max_ub_number:
            input_ub_data_one = tik_instance.Tensor(dtype, (input_ub_len_one, ),
                                                    name="input_ub_data_one", scope=tik.scope_ubuf)
            input_ub_length = input_ub_len_one // type_block_num

            input_ub_size = ub_number - input_ub_len_one
            for i in range(0, input_ub_size):
                input_ub_size_one = input_ub_size - i
                input_ub_size_two = input_ub_size_one % len_burst
                input_ub_size_three = input_ub_size_one // len_burst
                if input_ub_size_two == 0 and (input_ub_size_three % type_block_num) == 0:
                    sub_num = i
                    break
            use_input_ub_size = input_ub_size - sub_num
            input_ub_data = tik_instance.Tensor(dtype, (use_input_ub_size,),
                                                name="input_ub_data", scope=tik.scope_ubuf)

            use_input_length = use_input_ub_size // type_block_num
            max_use_number = use_input_ub_size // len_burst
            max_use_number_one = max_use_number *consecutive
            use_output_length = max_use_number_one // type_block_num

            loop_index = len_burst_one // use_input_ub_size

            last_input_ub_data = len_burst_one - loop_index * use_input_ub_size

            last_input_ub_data_length = last_input_ub_data // type_block_num
            max_last_use_number = last_input_ub_data // len_burst
            last_output_ub_data = max_last_use_number * consecutive
            last_output_ub_data_length = last_output_ub_data // type_block_num

            last_core_loop_index = tail_num // use_input_ub_size

            last_core_input_ub_data = tail_num % use_input_ub_size

            if last_core_input_ub_data % type_block_num == 0:
                last_core_input_ub_data_length = last_core_input_ub_data // type_block_num
            else:
                last_core_input_ub_data_length = (last_core_input_ub_data // type_block_num) + 1

            last_core_tail_max_use_number = last_core_input_ub_data // len_burst
            last_core_output_ub_data = last_core_tail_max_use_number * consecutive
            if last_core_output_ub_data % type_block_num == 0:
                last_core_output_ub_data_length = last_core_output_ub_data // type_block_num
            else:
                last_core_output_ub_data_length = (last_core_output_ub_data // type_block_num) + 1

        if input_ub_len_one > max_ub_number and len_burst_one > max_ub_number:
            for i in range(0, max_ub_number):
                max_ub_size = max_ub_number - i
                can_div_max_ub_size = max_ub_size // len_burst
                if max_ub_size % len_burst == 0:
                    if can_div_max_ub_size % type_block_num == 0:
                        sub_num = i
                        break
            max_ub_size_last = max_ub_number - sub_num
            input_ub_data = tik_instance.Tensor(dtype, (max_ub_size_last,),
                                                name="input_ub_data", scope=tik.scope_ubuf)
            max_ub_size_last_one = max_ub_size_last // len_burst * consecutive

            both_ub_data_length = max_ub_size_last // type_block_num
            max_both_number = max_ub_size_last // len_burst

            max_out_ub_size = max_ub_size_last // len_burst *consecutive
            max_out_ub_size_stride = max_out_ub_size // type_block_num

            input_ub_data_one = tik_instance.Tensor(dtype, (max_ub_size_last_one, ),
                                                    name="input_ub_data_one", scope=tik.scope_ubuf)
            loop_index = len_burst_one // max_ub_size_last

            tail_len_burst_one = len_burst_one % max_ub_size_last


            tail_len_burst_data_length = tail_len_burst_one // type_block_num

            tail_max_both_number = tail_len_burst_one // len_burst
            tail_output_ub_data = tail_max_both_number * consecutive

            tail_output_data_length = tail_output_ub_data // type_block_num

            last_core_loop_index = tail_num // max_ub_size_last
            last_core_burst_one = tail_num % max_ub_size_last
            if last_core_burst_one == 0:
                last_core_len_burst_data_length = last_core_burst_one // type_block_num
            else:
                last_core_len_burst_data_length = (last_core_burst_one // type_block_num) + 1
            last_core_tail_max_both_number = last_core_burst_one // len_burst
            last_core_tail_output_ub_number = last_core_tail_max_both_number * consecutive
            if last_core_tail_output_ub_number % type_block_num == 0:
                last_core_tail_output_data_length = \
                    last_core_tail_output_ub_number // type_block_num
            else:
                last_core_tail_output_data_length = \
                    (last_core_tail_output_ub_number // type_block_num) + 1

    input_data = tik_instance.Tensor(dtype, (input_size, ),
                                     name="input_data", scope=tik.scope_gm)
    output_data = tik_instance.Tensor(dtype, (output_size,),
                                      name="output_data", scope=tik.scope_gm)

    if max_ub_data <= ub_number and last_max_ub_data <= ub_number:
        with tik_instance.for_range(0, core_num, block_num=core_num) as total_cycle:
            with tik_instance.if_scope(total_cycle < core_num - 1):
                tik_instance.data_move(input_ub_data,
                                       input_data[total_cycle * len_burst_one],
                                       0, 1, input_len_burst, 0, 0)
                with tik_instance.for_range(0, max_num) as group:
                    for cur_num in range(0, consecutive):
                        input_ub_data_one[group*consecutive + cur_num]\
                            .set_as(input_ub_data[group * len_burst + start_num + cur_num])
                tik_instance.data_move(output_data[total_cycle * input_ub_len_one],
                                       input_ub_data_one, 0, 1, input_ub_length, 0, 0)

            with tik_instance.if_scope(total_cycle == core_num - 1):
                tik_instance.data_move(input_ub_data,
                                       input_data[(core_num-1) * len_burst_one],
                                       0, 1, tail_len_burst, 0, 0)
                with tik_instance.for_range(0, max_num_one) as group:
                    for cur_num in range(0, consecutive):
                        input_ub_data_one[group * consecutive + cur_num]\
                            .set_as(input_ub_data[group * len_burst + start_num + cur_num])
                tik_instance.data_move(output_data[(core_num-1) * input_ub_len_one],
                                       input_ub_data_one, 0, 1, tail_len_output_burst, 0, 0)
    if max_ub_data > ub_number:
        if input_ub_len_one > max_ub_number and len_burst_one > max_ub_number:
            with tik_instance.for_range(0, core_num, block_num=core_num) as total_cycle:
                with tik_instance.if_scope(total_cycle < core_num - 1):
                    with tik_instance.for_range(0, loop_index) as loop:
                        tik_instance.data_move(input_ub_data,
                                               input_data[total_cycle * len_burst_one
                                                          + loop * max_ub_size_last],
                                               0, 1, both_ub_data_length, 0, 0)
                        with tik_instance.for_range(0, max_both_number) as group:
                            with tik_instance.for_range(0, consecutive) as cur_num:
                                input_ub_data_one[group * consecutive + cur_num]\
                                    .set_as(input_ub_data[group * len_burst + start_num + cur_num])
                        tik_instance.data_move(output_data[total_cycle * input_ub_len_one
                                                           + loop * max_ub_size_last_one],
                                               input_ub_data_one, 0, 1,
                                               max_out_ub_size_stride, 0, 0)

                    with tik_instance.if_scope(tail_len_burst_one > 0):
                        tik_instance.data_move(input_ub_data,
                                               input_data[total_cycle * len_burst_one
                                                          + loop_index * max_ub_size_last],
                                               0, 1, tail_len_burst_data_length, 0, 0)
                        with tik_instance.for_range(0, tail_max_both_number) as group:
                            with tik_instance.for_range(0, consecutive) as cur_num:
                                input_ub_data_one[group * consecutive + cur_num]\
                                    .set_as(input_ub_data[group * len_burst + start_num + cur_num])
                        tik_instance.data_move(output_data[total_cycle * input_ub_len_one
                                                           + loop_index * max_ub_size_last_one],
                                               input_ub_data_one, 0, 1,
                                               tail_output_data_length, 0, 0)

                with tik_instance.if_scope(total_cycle == core_num -1):
                    with tik_instance.for_range(0, last_core_loop_index) as loop:
                        tik_instance.data_move(input_ub_data,
                                               input_data[(core_num-1) * len_burst_one
                                                          + loop * max_ub_size_last],
                                               0, 1, both_ub_data_length, 0, 0)
                        with tik_instance.for_range(0, max_both_number) as group:
                            with tik_instance.for_range(0, consecutive) as cur_num:
                                input_ub_data_one[group * consecutive + cur_num]\
                                    .set_as(input_ub_data[group * len_burst + start_num + cur_num])
                        tik_instance.data_move(output_data[(core_num - 1) * input_ub_len_one
                                                           + loop * max_ub_size_last_one],
                                               input_ub_data_one, 0, 1,
                                               max_out_ub_size_stride, 0, 0)

                    with tik_instance.if_scope(last_core_burst_one > 0):
                        tik_instance.data_move(input_ub_data,
                                               input_data[(core_num-1) * len_burst_one
                                                          + last_core_loop_index
                                                          * max_ub_size_last],
                                               0, 1, last_core_len_burst_data_length, 0, 0)
                        with tik_instance.for_range(0, last_core_tail_max_both_number) as group:
                            with tik_instance.for_range(0, consecutive) as cur_num:
                                input_ub_data_one[group * consecutive + cur_num]\
                                    .set_as(input_ub_data[group * len_burst + start_num + cur_num])
                        tik_instance.data_move(output_data[(core_num-1) * input_ub_len_one
                                                           + last_core_loop_index
                                                           * max_ub_size_last_one],
                                               input_ub_data_one, 0, 1,
                                               last_core_tail_output_data_length, 0, 0)

        if input_ub_len_one <= max_ub_number and len_burst_one > max_ub_number:
            with tik_instance.for_range(0, core_num, block_num=core_num) as total_cycle:
                with tik_instance.if_scope(total_cycle < core_num - 1):
                    with tik_instance.for_range(0, loop_index) as loop:
                        tik_instance.data_move(input_ub_data,
                                               input_data[total_cycle * len_burst_one
                                                          + loop * use_input_ub_size],
                                               0, 1, use_input_length, 0, 0)
                        with tik_instance.for_range(0, max_use_number) as group:
                            with tik_instance.for_range(0, consecutive) as cur_num:
                                input_ub_data_one[group * consecutive + cur_num]\
                                    .set_as(input_ub_data[group * len_burst + start_num + cur_num])
                        tik_instance.data_move(output_data[total_cycle * input_ub_len_one
                                                           + loop * max_use_number_one],
                                               input_ub_data_one, 0, 1,
                                               use_output_length, 0, 0)

                    with tik_instance.if_scope(last_input_ub_data > 0):
                        tik_instance.data_move(input_ub_data,
                                               input_data[total_cycle * len_burst_one
                                                          + loop_index * use_input_ub_size],
                                               0, 1, last_input_ub_data_length, 0, 0)
                        with tik_instance.for_range(0, max_last_use_number) as group:
                            with tik_instance.for_range(0, consecutive) as cur_num:
                                input_ub_data_one[group * consecutive + cur_num]\
                                    .set_as(input_ub_data[group * len_burst + start_num +cur_num])
                        tik_instance.data_move(output_data[total_cycle * input_ub_len_one
                                                           + loop_index * max_use_number_one],
                                               input_ub_data_one, 0, 1,
                                               last_output_ub_data_length, 0, 0)

                with tik_instance.if_scope(total_cycle == core_num - 1):
                    with tik_instance.for_range(0, last_core_loop_index) as loop:
                        tik_instance.data_move(input_ub_data,
                                               input_data[total_cycle * len_burst_one
                                                          + loop * use_input_ub_size],
                                               0, 1, use_input_length, 0, 0)
                        with tik_instance.for_range(0, max_use_number) as group:
                            with tik_instance.for_range(0, consecutive) as cur_num:
                                input_ub_data_one[group * consecutive + cur_num]\
                                    .set_as(input_ub_data[group * len_burst + start_num + cur_num])
                        tik_instance.data_move(output_data[total_cycle * input_ub_len_one
                                                           + loop * max_use_number_one],
                                               input_ub_data_one, 0, 1, use_output_length, 0, 0)

                    with tik_instance.if_scope(last_core_input_ub_data > 0):
                        tik_instance.data_move(input_ub_data,
                                               input_data[total_cycle * len_burst_one
                                                          + loop_index * use_input_ub_size],
                                               0, 1, last_core_input_ub_data_length, 0, 0)
                        with tik_instance.for_range(0, last_core_tail_max_use_number) as group:
                            with tik_instance.for_range(0, consecutive) as cur_num:
                                input_ub_data_one[group * consecutive + cur_num]\
                                    .set_as(input_ub_data[group * len_burst + start_num + cur_num])
                        tik_instance.data_move(output_data[total_cycle * input_ub_len_one
                                                           + last_core_loop_index
                                                           * max_use_number_one],
                                               input_ub_data_one, 0, 1,
                                               last_core_output_ub_data_length, 0, 0)

    tik_instance.BuildCCE(kernel_name, inputs=[input_data], outputs=[output_data])
    return tik_instance
