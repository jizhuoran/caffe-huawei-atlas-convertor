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

nd_2_zn
"""
from functools import reduce as functools_reduce
from te import platform as tbe_platform
from topi.cce import util
from te import tik

# available ub size
TOTAL_UB_MEMORY = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
# available number of cores
MAX_CORE_NUM = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
# bytes of type int8
SIZE_ONE_BYTES = 1
# bytes of type float16
SIZE_TWO_BYTES = 2
# size of the cube unit
CUBE_SIZE = 16
# size of the cube unit of last axis when dtype is int8
CUBE_SIZE_2 = 32
# minimum unit of data_move: 32Bytes
DATA_MOVE_MIN_UNIT = 32
# maximum repeat number
MAX_REPEATS = 255
# maximum blk stride
MAX_STRIDE_BLK = 65535
# maximum mask
MAX_MASK = 128


# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,invalid-name
# pylint: disable=unused-argument,too-many-lines,too-many-statements
def _cal_core(tik_instance, total_core_loop_num, num_core, core_number):
    """
    calculate the loop number on each core
    """
    core_loop = tik_instance.Scalar("uint64")
    sum_core = tik_instance.Scalar("uint64")

    with tik_instance.if_scope(num_core < total_core_loop_num % MAX_CORE_NUM):
        core_loop.set_as((total_core_loop_num + core_number - 1) //
                         core_number)
        sum_core.set_as(core_loop * num_core)

    with tik_instance.else_scope():
        core_loop.set_as(total_core_loop_num // core_number)
        sum_core.set_as((core_loop + 1) *
                        (total_core_loop_num % MAX_CORE_NUM) +
                        core_loop * (num_core - total_core_loop_num %
                                     MAX_CORE_NUM))

    return core_loop, sum_core


def _set_core_num(loop_number):
    """
    set the block_num
    """
    if loop_number < MAX_CORE_NUM:
        return loop_number

    return MAX_CORE_NUM


def _cal_core_loop(tik_instance, num_data_one_loop, core_loop, ub_ori):
    """
    calculate the number of loops and remainder on each core
    """

    align_loop = tik_instance.Scalar("uint64")
    align_loop.set_as(ub_ori // num_data_one_loop)
    with tik_instance.if_scope(align_loop * num_data_one_loop > ub_ori):
        align_loop.set_as(align_loop - 1)

    remainder = tik_instance.Scalar("uint64")
    remainder.set_as(core_loop % align_loop)
    with tik_instance.if_scope(remainder == 0):
        remainder.set_as(align_loop)

    return align_loop, remainder


def _cal_core_loop_python(num_data_one_loop, core_loop, ub_ori):
    """
    calculate the number of loops and remainder on each core and return python
    variable
    """
    align_loop = ub_ori // num_data_one_loop

    if align_loop * num_data_one_loop > ub_ori:
        align_loop = align_loop - 1

    remainder = core_loop % align_loop

    return align_loop, remainder


def _cast_dtype(tik_instance, dst, src, cast_repeat_time,
                cast_remainder, cast_case):
    """
    cast the data form int8 to float16 and from float16 to int8
    """
    if cast_case == "int8_2_float16":
        tik_instance.vconv(MAX_MASK, 'none', dst, src, cast_repeat_time,
                           1, 1, 8, 4, None)
        with tik_instance.if_scope(cast_remainder != 0):
            tik_instance.vconv(cast_remainder, 'none',
                               dst[cast_repeat_time * MAX_MASK],
                               src[cast_repeat_time * MAX_MASK],
                               1, 1, 1, 8, 4, None)
    elif cast_case == "float16_2_int8":
        tik_instance.vconv(MAX_MASK, 'none', dst, src, cast_repeat_time,
                           1, 1, 4, 8, None)
        with tik_instance.if_scope(cast_remainder != 0):
            tik_instance.vconv(cast_remainder, 'none',
                               dst[cast_repeat_time * MAX_MASK],
                               src[cast_repeat_time * MAX_MASK],
                               1, 1, 1, 4, 8, None)


class ND2ZNComputeInt8:
    """
    Rearranges data from ND format into FRACTAL_Zn format

    Functions
    ----------
    __init__:
        initialize some properties
    set_tik_instance:
        set tik_instance
    set_src_dst_tensor:
        set input and output tensor
    cal_core_loop:
        calculate the loop number on each core
    set_format_transfer_case:
        divide the transfer case from nd to nz
    vector_dup_zero_python
        vector_dup zeros when dup_number is python variable
    data_rearrange_case_zero:
        rearrange data when UB can put in last axis * 32 data and
        the shape of dst is 4-D and last axis % 32 == 0 or
        16 < last axis % 32 < 32
    data_rearrange_case_one:
        rearrange data when UB can put in last axis * 32 data and
        the shape of dst is 4-D and 0 < last axis % 32 <= 16
    data_rearrange_case_two:
        rearrange data when UB can not put in last axis * 32 data and
        the shape of dst is 4-D and last axis % 32 == 0 or
        16 < last axis % 32 < 32
    data_rearrange_case_three:
        rearrange data when UB can not put in last axis * 32 data and
        the shape of dst is 4-D and 0 < last axis % 32 <= 16
    def format_transfer_case_zero:
        the transfer process when the transfer case is 0
    def format_transfer_case_one:
        the transfer process when the transfer case is 1
    def format_transfer_case_two:
        the transfer process when the transfer case is 2
    def format_transfer_case_three:
        the transfer process when the transfer case is 3
    def remainder_case_zero:
        the transfer process of remainder when the transfer case is 3 and
        last axis % 32 == 0 or 17 < last axis % 32 < 32
    def remainder_case_one:
        the transfer process of remainder when the transfer case is 3 and
        0 < last axis % 32 <= 16
    nd_2_Zn_compute:
        the overall transfer process
    get_tik_instance:
        obtain tik instance

    Returns
    -------
    None
    """
    def __init__(self, src_shape, dtype, kernel_name):
        """
        initialize some properties
        """
        self.src_shape_ori = src_shape[:]
        self.src_shape = src_shape[:]
        self.dtype = dtype
        self.kernel_name = kernel_name
        if len(src_shape) == 1:
            self.src_shape = [1, src_shape[0]]
        self.dst_shape = self.src_shape[:]
        self.dst_shape[-2:] = [(self.src_shape[-2] + CUBE_SIZE_2 - 1) //
                               CUBE_SIZE_2,
                               (self.src_shape[-1] + CUBE_SIZE - 1) //
                               CUBE_SIZE,
                               CUBE_SIZE, CUBE_SIZE_2]
        self.num_byte = SIZE_ONE_BYTES
        self.cast_num_byte = SIZE_TWO_BYTES

        # the number of data that can be moved in each data_move
        self.num_data = DATA_MOVE_MIN_UNIT // self.num_byte
        # the number of float16 data that can be moved in each data_move
        self.cast_num_data = DATA_MOVE_MIN_UNIT // self.cast_num_byte
        util.check_shape_rule(self.dst_shape)
        util.check_tensor_shape_size(self.dst_shape)
        # the number of data that UB can put in
        self.ub_memory = min(TOTAL_UB_MEMORY, 252 * 1024) // self.cast_num_byte // 4
        self.src_gm = None
        self.dst_gm = None

    def set_tik_instance(self):
        """
        set tik_instance
        """
        tik_instance = tik.Tik()
        self.set_src_dst_tensor(tik_instance)

        return tik_instance

    def set_src_dst_tensor(self, tik_instance):
        """
        set input and output tensor
        """
        src_element_number = functools_reduce(lambda x1, x2: x1 * x2,
                                              self.src_shape_ori[:])
        dst_element_number = functools_reduce(lambda x1, x2: x1 * x2,
                                              self.dst_shape[:])
        self.src_gm = tik_instance.Tensor(self.dtype,
                                          (src_element_number,),
                                          name="src_gm",
                                          scope=tik.scope_gm)
        self.dst_gm = tik_instance.Tensor(self.dtype,
                                          (dst_element_number,),
                                          name="dst_gm",
                                          scope=tik.scope_gm)

    def set_format_transfer_case(self):
        """
        divide the transfer case from nd to nz
        """
        if len(self.dst_shape) == 4:
            is_four_d = 1
        else:
            is_four_d = (functools_reduce(lambda x1, x2: x1 * x2,
                                          self.src_shape[:-2]) == 1)

        if is_four_d:
            if self.src_shape[-1] % CUBE_SIZE_2 == 0:
                format_transfer_case = 0
                if self.dst_shape[-3] * self.dst_shape[-1] * \
                        self.dst_shape[-2] > self.ub_memory:
                    format_transfer_case = 3
            elif 0 < self.src_shape[-1] % CUBE_SIZE_2 <= CUBE_SIZE:
                format_transfer_case = 1
                if self.dst_shape[-3] * self.dst_shape[-1] * \
                        self.dst_shape[-2] + CUBE_SIZE_2 * CUBE_SIZE > \
                        self.ub_memory:
                    format_transfer_case = 3
            else:
                format_transfer_case = 2
                if self.dst_shape[-3] * self.dst_shape[-1] * \
                        self.dst_shape[-2] > self.ub_memory:
                    format_transfer_case = 3
        else:
            raise RuntimeError("nd_2_zn only support 2D now "
                               "when dtype is int8")

        return format_transfer_case

    def vector_dup_zero_python(self, tik_instance, ub_trans, dup_number,
                               offset):
        """
        vector_dup zeros when dup_number is python variable
        """
        scalar_zero = tik_instance.Scalar(dtype="float16", init_value=0.0)
        repeat_number = dup_number // MAX_MASK
        tail = dup_number % MAX_MASK

        with tik_instance.for_range(0, repeat_number // MAX_REPEATS) as \
                num_repeat_loop:
            tik_instance.vector_dup(MAX_MASK,
                                    ub_trans[MAX_MASK * MAX_REPEATS *
                                             num_repeat_loop + offset],
                                    scalar_zero,
                                    MAX_REPEATS,
                                    self.cast_num_byte // 2,
                                    MAX_MASK // self.cast_num_data)
        if repeat_number % MAX_REPEATS != 0:
            tik_instance.vector_dup(MAX_MASK,
                                    ub_trans[repeat_number // MAX_REPEATS *
                                             MAX_MASK * MAX_REPEATS + offset],
                                    scalar_zero,
                                    repeat_number % MAX_REPEATS,
                                    self.cast_num_byte // 2,
                                    MAX_MASK // self.cast_num_data)
        if tail != 0:
            tik_instance.vector_dup(tail,
                                    ub_trans[MAX_MASK * repeat_number +
                                             offset],
                                    scalar_zero,
                                    1,
                                    self.cast_num_byte // 2,
                                    MAX_MASK // self.cast_num_data)

    def data_rearrange_case_zero(self, tik_instance, ub_ori, ub_cast_fp16,
                                 ub_trans, ub_cast_int8, loop_num, is_last):
        """
        rearrange data when UB can put in last axis * 32 data and
        the shape of dst is 4-D and 0 < last axis % 32 <= 16
        """
        cast_repeat_time = tik_instance.Scalar("uint64")
        cast_remainder = tik_instance.Scalar("uint64")
        with tik_instance.if_scope(is_last == 1):
            if (self.src_shape[-2] % CUBE_SIZE_2) == 0:
                cast_repeat_time.set_as(loop_num * CUBE_SIZE_2 *
                                        self.dst_shape[-3] *
                                        self.dst_shape[-2] // MAX_MASK)
                cast_remainder.set_as(loop_num * CUBE_SIZE_2 *
                                      self.dst_shape[-3] *
                                      self.dst_shape[-2] % MAX_MASK)
            else:
                cast_repeat_time.set_as((loop_num * CUBE_SIZE_2 - CUBE_SIZE_2 +
                                         self.src_shape[-2] % CUBE_SIZE_2) *
                                        self.dst_shape[-3] *
                                        self.dst_shape[-2] // MAX_MASK)
                cast_remainder.set_as((loop_num * CUBE_SIZE_2 - CUBE_SIZE_2 +
                                       self.src_shape[-2] % CUBE_SIZE_2) *
                                      self.dst_shape[-3] * self.dst_shape[-2] %
                                      MAX_MASK)
        with tik_instance.else_scope():
            cast_repeat_time.set_as(loop_num * CUBE_SIZE_2 *
                                    self.dst_shape[-3] * self.dst_shape[-2] //
                                    MAX_MASK)
            cast_remainder.set_as(loop_num * CUBE_SIZE_2 * self.dst_shape[-3] *
                                  self.dst_shape[-2] % MAX_MASK)
        # cast the data from int8 to float16
        _cast_dtype(tik_instance, ub_cast_fp16, ub_ori, cast_repeat_time,
                    cast_remainder, "int8_2_float16")
        num_row_one_loop = loop_num * CUBE_SIZE_2
        scalar_zero = tik_instance.Scalar(dtype="float16", init_value=0.0)
        # last axis padding zero
        if self.src_shape[-1] % CUBE_SIZE != 0:
            mask = 0
            for i, _ in enumerate(range(CUBE_SIZE -
                                        self.src_shape[-1] % CUBE_SIZE)):
                mask += 2 ** (CUBE_SIZE - 1 - i)

            with tik_instance.for_range(0, num_row_one_loop // MAX_REPEATS) \
                    as num_repeat:
                tik_instance.vector_dup([0, mask],
                                        ub_cast_fp16[(MAX_REPEATS *
                                                      num_repeat + 1) *
                                                     self.dst_shape[-3] *
                                                     self.dst_shape[-2] -
                                                     CUBE_SIZE],
                                        scalar_zero, MAX_REPEATS,
                                        0, self.dst_shape[-3] *
                                        self.dst_shape[-2] //
                                        self.cast_num_data)
            with tik_instance.if_scope(num_row_one_loop % MAX_REPEATS != 0):
                tik_instance.vector_dup([0, mask],
                                        ub_cast_fp16[((num_row_one_loop //
                                                       MAX_REPEATS) *
                                                      MAX_REPEATS + 1) *
                                                     self.dst_shape[-3] *
                                                     self.dst_shape[-2] -
                                                     CUBE_SIZE],
                                        scalar_zero,
                                        num_row_one_loop % MAX_REPEATS,
                                        0, self.dst_shape[-3] *
                                        self.dst_shape[-2] //
                                        self.cast_num_data)
        # second last axis padding zero
        if (self.src_shape[-2] % CUBE_SIZE_2) != 0:
            with tik_instance.if_scope(is_last == 1):
                dup_number = (CUBE_SIZE_2 - self.src_shape[-2] %
                              CUBE_SIZE_2) * self.dst_shape[-3] *\
                             self.dst_shape[-2]
                offset = ((loop_num - 1) * self.dst_shape[-1] +
                          self.src_shape[-2] % CUBE_SIZE_2) * \
                         self.dst_shape[-3] * self.dst_shape[-2]
                self.vector_dup_zero_python(tik_instance, ub_cast_fp16,
                                            dup_number, offset)

        # data rearrange
        with tik_instance.for_range(0, loop_num) as num_row:
            offset = num_row * self.dst_shape[-3] * self.dst_shape[-2] * \
                     self.dst_shape[-1]
            dst_list_low = [ub_trans[i * CUBE_SIZE_2 + offset]
                            for i in range(CUBE_SIZE)]
            src_list_low = [ub_cast_fp16[i * self.dst_shape[-3] *
                                         self.dst_shape[-2] + offset]
                            for i in range(CUBE_SIZE)]
            dst_list_high = [ub_trans[i * CUBE_SIZE_2 + CUBE_SIZE + offset]
                             for i in range(16)]
            src_list_high = [ub_cast_fp16[i * self.dst_shape[-3] *
                                          self.dst_shape[-2] +
                                          CUBE_SIZE * self.dst_shape[-3] *
                                          self.dst_shape[-2] + offset]
                             for i in range(CUBE_SIZE)]
            if self.dst_shape[-3] == 1:

                tik_instance.vnchwconv(False, False,
                                       dst_list_low, src_list_low,
                                       1, 0, 0)
                tik_instance.vnchwconv(False, False,
                                       dst_list_high, src_list_high,
                                       1, 0, 0)
            else:
                tik_instance.vnchwconv(False, False,
                                       dst_list_low, src_list_low,
                                       self.dst_shape[-3], CUBE_SIZE_2,
                                       self.cast_num_byte // 2)
                tik_instance.vnchwconv(False, False,
                                       dst_list_high, src_list_high,
                                       self.dst_shape[-3], CUBE_SIZE_2,
                                       self.cast_num_byte // 2)

        cast_repeat_time.set_as(loop_num * CUBE_SIZE_2 * self.dst_shape[-3] *
                                self.dst_shape[-2] // MAX_MASK)
        cast_remainder.set_as(loop_num * CUBE_SIZE * self.dst_shape[-3] *
                              self.dst_shape[-2] % MAX_MASK)
        # cast the data from float16 to int8
        _cast_dtype(tik_instance, ub_cast_int8, ub_trans, cast_repeat_time,
                    cast_remainder, "float16_2_int8")

    def data_rearrange_case_one(self, tik_instance, ub_ori, ub_cast_fp16,
                                ub_trans, ub_cast_int8, loop_num, is_last):
        """
        rearrange data when UB can put in last axis * 32 data and
        the shape of dst is 4-D and last axis % 32 == 0 or
        16 < last axis % 32 < 32
        """
        cast_repeat_time = tik_instance.Scalar("uint64")
        cast_remainder = tik_instance.Scalar("uint64")
        with tik_instance.if_scope(is_last == 1):
            if (self.src_shape[-2] % CUBE_SIZE_2) == 0:
                cast_repeat_time.set_as(loop_num * CUBE_SIZE_2 *
                                        (self.dst_shape[-3] + 1) *
                                        self.dst_shape[-2] // MAX_MASK)
                cast_remainder.set_as(loop_num * CUBE_SIZE_2 *
                                      (self.dst_shape[-3] + 1) *
                                      self.dst_shape[-2] % MAX_MASK)
            else:
                cast_repeat_time.set_as((loop_num * CUBE_SIZE_2 - CUBE_SIZE_2 +
                                         self.src_shape[-2] % CUBE_SIZE_2) *
                                        (self.dst_shape[-3] + 1) *
                                        self.dst_shape[-2] // MAX_MASK)
                cast_remainder.set_as((loop_num * CUBE_SIZE_2 - CUBE_SIZE_2 +
                                       self.src_shape[-2] % CUBE_SIZE_2) *
                                      (self.dst_shape[-3] + 1) *
                                      self.dst_shape[-2] % MAX_MASK)
        with tik_instance.else_scope():
            cast_repeat_time.set_as(loop_num * CUBE_SIZE_2 *
                                    (self.dst_shape[-3] + 1) *
                                    self.dst_shape[-2] // MAX_MASK)
            cast_remainder.set_as(loop_num * CUBE_SIZE_2 *
                                  (self.dst_shape[-3] + 1) *
                                  self.dst_shape[-2] % MAX_MASK)
        # cast the data from int8 to float16
        _cast_dtype(tik_instance, ub_cast_fp16, ub_ori, cast_repeat_time,
                    cast_remainder, "int8_2_float16")
        num_row_one_loop = loop_num * CUBE_SIZE_2
        scalar_zero = tik_instance.Scalar(dtype="float16", init_value=0.0)
        # last axis padding zero
        if self.src_shape[-1] % CUBE_SIZE != 0:
            mask = 0
            for i, _ in enumerate(range(CUBE_SIZE -
                                        self.src_shape[-1] % CUBE_SIZE)):
                mask += 2 ** (CUBE_SIZE - 1 - i)

            with tik_instance.for_range(0, num_row_one_loop // MAX_REPEATS) \
                    as num_repeat:
                tik_instance.vector_dup([0, mask],
                                        ub_cast_fp16[(MAX_REPEATS *
                                                      num_repeat + 1) *
                                                     (self.dst_shape[-3] + 1) *
                                                     self.dst_shape[-2] -
                                                     CUBE_SIZE_2],
                                        scalar_zero, MAX_REPEATS,
                                        0, (self.dst_shape[-3] + 1) *
                                        self.dst_shape[-2] //
                                        self.cast_num_data)
            with tik_instance.if_scope(num_row_one_loop % MAX_REPEATS != 0):
                tik_instance.vector_dup([0, mask],
                                        ub_cast_fp16[((num_row_one_loop //
                                                       MAX_REPEATS) *
                                                      MAX_REPEATS + 1) *
                                                     (self.dst_shape[-3] + 1) *
                                                     self.dst_shape[-2] -
                                                     CUBE_SIZE_2],
                                        scalar_zero,
                                        num_row_one_loop % MAX_REPEATS,
                                        0, (self.dst_shape[-3] + 1) *
                                        self.dst_shape[-2] //
                                        self.cast_num_data)
        # second last axis padding zero
        if self.src_shape[-2] % CUBE_SIZE_2 != 0:
            with tik_instance.if_scope(is_last == 1):
                dup_number = (CUBE_SIZE_2 - self.src_shape[-2] %
                              CUBE_SIZE_2) * (self.dst_shape[-3] + 1) *\
                             self.dst_shape[-2]
                offset = ((loop_num - 1) * self.dst_shape[-1] +
                          self.src_shape[-2] % CUBE_SIZE_2) *  \
                         (self.dst_shape[-3] + 1) * self.dst_shape[-2]
                self.vector_dup_zero_python(tik_instance, ub_cast_fp16,
                                            dup_number, offset)

        # data rearrange
        with tik_instance.for_range(0, loop_num) as num_row:
            src_offset = num_row * (self.dst_shape[-3] + 1) *\
                         self.dst_shape[-2] * self.dst_shape[-1]
            dst_offset = num_row * self.dst_shape[-3] * self.dst_shape[-2] * \
                         self.dst_shape[-1]
            dst_list_low = [ub_trans[i * CUBE_SIZE_2 + dst_offset]
                            for i in range(CUBE_SIZE)]
            src_list_low = [ub_cast_fp16[i * (self.dst_shape[-3] + 1) *
                                         self.dst_shape[-2] + src_offset]
                            for i in range(CUBE_SIZE)]
            dst_list_high = [ub_trans[i * CUBE_SIZE_2 + CUBE_SIZE + dst_offset]
                             for i in range(16)]
            src_list_high = [ub_cast_fp16[i * (self.dst_shape[-3] + 1) *
                                          self.dst_shape[-2] +
                                          CUBE_SIZE *
                                          (self.dst_shape[-3] + 1) *
                                          self.dst_shape[-2] + src_offset]
                             for i in range(CUBE_SIZE)]

            if self.dst_shape[-3] == 1:
                tik_instance.vnchwconv(False, False,
                                       dst_list_low, src_list_low,
                                       1, 0, 0)
                tik_instance.vnchwconv(False, False,
                                       dst_list_high, src_list_high,
                                       1, 0, 0)
            else:
                tik_instance.vnchwconv(False, False,
                                       dst_list_low, src_list_low,
                                       self.dst_shape[-3], CUBE_SIZE_2,
                                       self.cast_num_byte // 2)
                tik_instance.vnchwconv(False, False,
                                       dst_list_high, src_list_high,
                                       self.dst_shape[-3], CUBE_SIZE_2,
                                       self.cast_num_byte // 2)

        cast_repeat_time.set_as(loop_num * CUBE_SIZE_2 * self.dst_shape[-3] *
                                self.dst_shape[-2] // MAX_MASK)
        cast_remainder.set_as(loop_num * CUBE_SIZE * self.dst_shape[-3] *
                              self.dst_shape[-2] % MAX_MASK)
        # cast the data from float16 to int8
        _cast_dtype(tik_instance, ub_cast_int8, ub_trans, cast_repeat_time,
                    cast_remainder, "float16_2_int8")

    def data_rearrange_case_two(self, tik_instance, ub_ori, ub_cast_fp16,
                                ub_trans, ub_cast_int8,
                                num_loop_time, loop_num, is_last):
        """
        rearrange data when UB can not put in last axis * 32 data and
        the shape of dst is 4-D and last axis % 32 == 0 or
        16 < last axis % 32 < 32
        """
        cast_repeat_time = tik_instance.Scalar("uint64")
        cast_remainder = tik_instance.Scalar("uint64")
        with tik_instance.if_scope(num_loop_time == self.dst_shape[-4] - 1):
            if (self.src_shape[-2] % CUBE_SIZE_2) == 0:
                cast_repeat_time.set_as(loop_num * CUBE_SIZE *
                                        self.dst_shape[-1] // MAX_MASK)
                cast_remainder.set_as(loop_num * CUBE_SIZE *
                                      self.dst_shape[-1] % MAX_MASK)
            else:
                cast_repeat_time.set_as((self.src_shape[-2] % CUBE_SIZE_2) *
                                        loop_num * CUBE_SIZE // MAX_MASK)
                cast_remainder.set_as((self.src_shape[-2] % CUBE_SIZE_2) *
                                      loop_num * CUBE_SIZE % MAX_MASK)
        with tik_instance.else_scope():
            cast_repeat_time.set_as(loop_num * CUBE_SIZE *
                                    self.dst_shape[-1] // MAX_MASK)
            cast_remainder.set_as(loop_num * CUBE_SIZE * self.dst_shape[-1] %
                                  MAX_MASK)
        # cast the data from int8 to float16
        _cast_dtype(tik_instance, ub_cast_fp16, ub_ori, cast_repeat_time,
                    cast_remainder, "int8_2_float16")
        scalar_zero = tik_instance.Scalar(dtype="float16", init_value=0.0)
        # last axis padding zero
        if self.src_shape[-1] % CUBE_SIZE != 0:
            with tik_instance.if_scope(is_last == 1):
                mask = 0
                for i, _ in enumerate(range(CUBE_SIZE -
                                            self.src_shape[-1] % CUBE_SIZE)):
                    mask += 2 ** (CUBE_SIZE - 1 - i)

                tik_instance.vector_dup([0, mask],
                                        ub_cast_fp16[loop_num * CUBE_SIZE -
                                                     CUBE_SIZE],
                                        scalar_zero, CUBE_SIZE_2,
                                        0, loop_num * CUBE_SIZE //
                                        self.cast_num_data)
        # second last axis padding zero
        if (self.src_shape[-2] % CUBE_SIZE_2) != 0:
            with tik_instance.if_scope(num_loop_time ==
                                       self.dst_shape[-4] - 1):
                dup_number = (CUBE_SIZE_2 - self.src_shape[-2] %
                              CUBE_SIZE_2) * loop_num * self.dst_shape[-2]
                offset = (self.src_shape[-2] % CUBE_SIZE_2) * loop_num * \
                         self.dst_shape[-2]
                self.vector_dup_zero_python(tik_instance, ub_cast_fp16,
                                            dup_number, offset)
        # data rearrange
        dst_list_low = [ub_trans[i * CUBE_SIZE_2] for i in range(CUBE_SIZE)]
        src_list_low = [ub_cast_fp16[i * loop_num * self.dst_shape[-2]]
                        for i in range(CUBE_SIZE)]
        dst_list_high = [ub_trans[i * CUBE_SIZE_2 + CUBE_SIZE]
                         for i in range(16)]
        src_list_high = [ub_cast_fp16[i * loop_num * self.dst_shape[-2] +
                                      CUBE_SIZE * loop_num *
                                      self.dst_shape[-2]]
                         for i in range(CUBE_SIZE)]
        if loop_num == 1:
            tik_instance.vnchwconv(False, False, dst_list_low, src_list_low,
                                   1, 0, 0)
            tik_instance.vnchwconv(False, False, dst_list_high, src_list_high,
                                   1, 0, 0)
        else:
            tik_instance.vnchwconv(False, False, dst_list_low, src_list_low,
                                   loop_num, CUBE_SIZE_2,
                                   self.cast_num_byte // 2)
            tik_instance.vnchwconv(False, False, dst_list_high, src_list_high,
                                   loop_num, CUBE_SIZE_2,
                                   self.cast_num_byte // 2)

        cast_repeat_time.set_as(CUBE_SIZE * loop_num * self.dst_shape[-1] //
                                MAX_MASK)
        cast_remainder.set_as(CUBE_SIZE * loop_num * self.dst_shape[-1] %
                              MAX_MASK)
        # cast the data from float16 to int8
        _cast_dtype(tik_instance, ub_cast_int8, ub_trans, cast_repeat_time,
                    cast_remainder, "float16_2_int8")

    def data_rearrange_case_three(self, tik_instance, ub_ori, ub_cast_fp16,
                                  ub_trans, ub_cast_int8,
                                  num_loop_time, loop_num, is_last):
        """
        rearrange data when UB can not put in last axis * 32 data and
        the shape of dst is 4-D and 0 < last axis % 32 <= 16
        """
        cast_repeat_time = tik_instance.Scalar("uint64")
        cast_remainder = tik_instance.Scalar("uint64")
        with tik_instance.if_scope(num_loop_time == self.dst_shape[-4] - 1):
            if (self.src_shape[-2] % CUBE_SIZE_2) == 0:
                cast_repeat_time.set_as((loop_num + 1) * CUBE_SIZE *
                                        self.dst_shape[-1] // MAX_MASK)
                cast_remainder.set_as((loop_num + 1) * CUBE_SIZE *
                                      self.dst_shape[-1] % MAX_MASK)
            else:
                cast_repeat_time.set_as((self.src_shape[-2] % CUBE_SIZE_2) *
                                        (loop_num + 1) * CUBE_SIZE // MAX_MASK)
                cast_remainder.set_as((self.src_shape[-2] % CUBE_SIZE_2) *
                                      (loop_num + 1) * CUBE_SIZE % MAX_MASK)
        with tik_instance.else_scope():
            cast_repeat_time.set_as((loop_num + 1) * CUBE_SIZE *
                                    self.dst_shape[-1] // MAX_MASK)
            cast_remainder.set_as((loop_num + 1) * CUBE_SIZE *
                                  self.dst_shape[-1] % MAX_MASK)
        # cast the data from int8 to float16
        _cast_dtype(tik_instance, ub_cast_fp16, ub_ori, cast_repeat_time,
                    cast_remainder, "int8_2_float16")
        scalar_zero = tik_instance.Scalar(dtype="float16", init_value=0.0)
        # last axis padding zero
        if self.src_shape[-1] % CUBE_SIZE != 0:
            with tik_instance.if_scope(is_last == 1):
                mask = 0
                for i, _ in enumerate(range(CUBE_SIZE -
                                            self.src_shape[-1] % CUBE_SIZE)):
                    mask += 2 ** (CUBE_SIZE - 1 - i)

                tik_instance.vector_dup([0, mask],
                                        ub_cast_fp16[(loop_num + 1) *
                                                     CUBE_SIZE - CUBE_SIZE_2],
                                        scalar_zero, CUBE_SIZE_2,
                                        0, (loop_num + 1) * CUBE_SIZE //
                                        self.cast_num_data)
        # second last axis padding zero
        if (self.src_shape[-2] % CUBE_SIZE_2) != 0:
            with tik_instance.if_scope(num_loop_time ==
                                       self.dst_shape[-4] - 1):
                dup_number = (CUBE_SIZE_2 - self.src_shape[-2] %
                              CUBE_SIZE_2) * (loop_num + 1) *\
                             self.dst_shape[-2]
                offset = (self.src_shape[-2] % CUBE_SIZE_2) *\
                         (loop_num + 1) * self.dst_shape[-2]
                self.vector_dup_zero_python(tik_instance, ub_cast_fp16,
                                            dup_number, offset)
        # data rearrange
        dst_list_low = [ub_trans[i * CUBE_SIZE_2] for i in range(CUBE_SIZE)]
        src_list_low = [ub_cast_fp16[i * (loop_num + 1) * self.dst_shape[-2]]
                        for i in range(CUBE_SIZE)]
        dst_list_high = [ub_trans[i * CUBE_SIZE_2 + CUBE_SIZE]
                         for i in range(16)]
        src_list_high = [ub_cast_fp16[i * (loop_num + 1) * self.dst_shape[-2] +
                                      CUBE_SIZE * (loop_num + 1) *
                                      self.dst_shape[-2]]
                         for i in range(CUBE_SIZE)]
        if loop_num == 1:
            tik_instance.vnchwconv(False, False, dst_list_low, src_list_low,
                                   1, 0, 0)
            tik_instance.vnchwconv(False, False, dst_list_high, src_list_high,
                                   1, 0, 0)
        else:
            tik_instance.vnchwconv(False, False, dst_list_low, src_list_low,
                                   loop_num, CUBE_SIZE_2,
                                   self.cast_num_byte // 2)
            tik_instance.vnchwconv(False, False, dst_list_high, src_list_high,
                                   loop_num, CUBE_SIZE_2,
                                   self.cast_num_byte // 2)

        cast_repeat_time.set_as(CUBE_SIZE * loop_num * self.dst_shape[-1] //
                                MAX_MASK)
        cast_remainder.set_as(CUBE_SIZE * loop_num * self.dst_shape[-1] %
                              MAX_MASK)
        # cast the data from float16 to int8
        _cast_dtype(tik_instance, ub_cast_int8, ub_trans, cast_repeat_time,
                    cast_remainder, "float16_2_int8")

    def format_transfer_case_zero(self, tik_instance):
        """
        the transfer process when UB can put in 16 * last axis data,
        last axis is 32B align and the shape of dst is 4-D
        """
        ub_ori_data = self.ub_memory
        ub_trans_data = ub_ori_data
        total_core_loop_num = self.dst_shape[-4]

        core_number = _set_core_num(total_core_loop_num)
        num_data_one_loop = self.dst_shape[-3] * self.dst_shape[-2] * \
                            self.dst_shape[-1]

        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            ub_ori = tik_instance.Tensor("int8",
                                         (ub_ori_data,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_cast_fp16 = tik_instance.Tensor("float16",
                                               (ub_trans_data,),
                                               name="ub_cast_fp16",
                                               scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor("float16",
                                           (ub_trans_data,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            ub_cast_int8 = tik_instance.Tensor("int8",
                                               (ub_trans_data,),
                                               name="ub_cast_int8",
                                               scope=tik.scope_ubuf)
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            align_loop, remainder = _cal_core_loop(tik_instance,
                                                   num_data_one_loop,
                                                   core_loop, ub_ori_data)
            with tik_instance.for_range(0, core_loop) as num_core_loop:

                total_core_loop = sum_core + num_core_loop
                num_fourth_last_axis = total_core_loop
                is_last = tik_instance.Scalar("uint64", init_value=0)
                with tik_instance.if_scope(num_fourth_last_axis ==
                                           self.dst_shape[-4] - 1):
                    is_last.set_as(1)
                src_ub_index = 0

                with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                                   align_loop == 0,
                                                   num_core_loop !=
                                                   core_loop - 1)):
                    src_gm_index = num_fourth_last_axis * num_data_one_loop - \
                                   (align_loop - 1) * num_data_one_loop
                    tik_instance.data_move(ub_ori[src_ub_index],
                                           self.src_gm[src_gm_index],
                                           0, 1,
                                           align_loop * num_data_one_loop //
                                           self.num_data, 0, 0)
                    self.data_rearrange_case_zero(tik_instance, ub_ori,
                                                  ub_cast_fp16, ub_trans,
                                                  ub_cast_int8, align_loop,
                                                  is_last)
                    dst_gm_index = (num_fourth_last_axis -
                                    (align_loop - 1)) * num_data_one_loop
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_cast_int8[0],
                                           0, 1,
                                           align_loop * num_data_one_loop //
                                           self.num_data, 0, 0)

                with tik_instance.if_scope(num_core_loop == core_loop - 1):
                    src_gm_index = num_fourth_last_axis * num_data_one_loop - \
                                   (remainder - 1) * num_data_one_loop
                    with tik_instance.if_scope(is_last == 1):
                        if self.src_shape[-2] % CUBE_SIZE_2 != 0:
                            tik_instance.data_move(ub_ori[src_ub_index],
                                                   self.src_gm[src_gm_index],
                                                   0, 1,
                                                   (remainder *
                                                    num_data_one_loop -
                                                    (CUBE_SIZE_2 -
                                                     self.src_shape[-2] %
                                                     CUBE_SIZE_2) *
                                                    self.dst_shape[-3] *
                                                    self.dst_shape[-2]) //
                                                   self.num_data, 0, 0)
                        else:
                            tik_instance.data_move(ub_ori[src_ub_index],
                                                   self.src_gm[src_gm_index],
                                                   0, 1,
                                                   remainder *
                                                   num_data_one_loop //
                                                   self.num_data, 0, 0)
                    with tik_instance.else_scope():
                        tik_instance.data_move(ub_ori[src_ub_index],
                                               self.src_gm[src_gm_index],
                                               0, 1,
                                               remainder * num_data_one_loop //
                                               self.num_data, 0, 0)
                    self.data_rearrange_case_zero(tik_instance, ub_ori,
                                                  ub_cast_fp16, ub_trans,
                                                  ub_cast_int8, remainder,
                                                  is_last)
                    dst_gm_index = (num_fourth_last_axis - (remainder - 1)) * \
                                   num_data_one_loop
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_cast_int8[0],
                                           0, 1,
                                           remainder * num_data_one_loop //
                                           self.num_data, 0, 0)

        return tik_instance

    def format_transfer_case_one(self, tik_instance):
        """
        the transfer process when UB can put in 6 * last axis data,
         0 < last axis % 32 <= 16 and the shape of dst is 4-D
        """
        ub_ori_data = self.ub_memory
        ub_trans_data = ub_ori_data
        total_core_loop_num = self.dst_shape[-4]

        core_number = _set_core_num(total_core_loop_num)
        num_data_one_loop = self.dst_shape[-3] * self.dst_shape[-2] * \
                            self.dst_shape[-1]

        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            ub_ori = tik_instance.Tensor("int8",
                                         (ub_ori_data,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_cast_fp16 = tik_instance.Tensor("float16",
                                               (ub_trans_data,),
                                               name="ub_cast_fp16",
                                               scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor("float16",
                                           (ub_trans_data,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            ub_cast_int8 = tik_instance.Tensor("int8",
                                               (ub_trans_data,),
                                               name="ub_cast_int8",
                                               scope=tik.scope_ubuf)
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            align_loop, remainder = _cal_core_loop(tik_instance,
                                                   num_data_one_loop +
                                                   CUBE_SIZE * CUBE_SIZE_2,
                                                   core_loop, ub_ori_data)
            with tik_instance.for_range(0, core_loop) as num_core_loop:
                total_core_loop = sum_core + num_core_loop
                num_fourth_last_axis = total_core_loop
                is_last = tik_instance.Scalar("uint64", init_value=0)
                src_gm_index = num_fourth_last_axis * self.src_shape[-1] * \
                               CUBE_SIZE_2
                src_ub_index = (num_core_loop % align_loop) * \
                               (num_data_one_loop + CUBE_SIZE_2 * CUBE_SIZE)
                with tik_instance.if_scope(num_fourth_last_axis ==
                                           self.dst_shape[-4] - 1):
                    is_last.set_as(1)
                    if self.src_shape[-2] % CUBE_SIZE_2 != 0:
                        with tik_instance.for_range(0, self.src_shape[-2] %
                                                    CUBE_SIZE_2) as \
                                num_cube_row:
                            tik_instance.data_move(ub_ori[src_ub_index +
                                                          num_cube_row *
                                                          self.dst_shape[-2] *
                                                          (self.dst_shape[-3] +
                                                           1)],
                                                   self.src_gm
                                                   [src_gm_index +
                                                    num_cube_row *
                                                    self.src_shape[-1]],
                                                   0, 1,
                                                   (self.dst_shape[-3] *
                                                    self.dst_shape[-2] +
                                                    self.num_data - 1) //
                                                   self.num_data, 0, 0)
                    else:
                        with tik_instance.for_range(0, CUBE_SIZE_2) as \
                                num_cube_row:
                            tik_instance.data_move(ub_ori[src_ub_index +
                                                          num_cube_row *
                                                          self.dst_shape[-2] *
                                                          (self.dst_shape[-3] +
                                                           1)],
                                                   self.src_gm
                                                   [src_gm_index +
                                                    num_cube_row *
                                                    self.src_shape[-1]],
                                                   0, 1,
                                                   (self.dst_shape[-3] *
                                                    self.dst_shape[-2] +
                                                    self.num_data - 1) //
                                                   self.num_data, 0, 0)
                with tik_instance.else_scope():
                    with tik_instance.for_range(0, CUBE_SIZE_2) as \
                            num_cube_row:
                        tik_instance.data_move(ub_ori[src_ub_index +
                                                      num_cube_row *
                                                      self.dst_shape[-2] *
                                                      (self.dst_shape[-3] +
                                                       1)],
                                               self.src_gm
                                               [src_gm_index +
                                                num_cube_row *
                                                self.src_shape[-1]],
                                               0, 1,
                                               (self.dst_shape[-3] *
                                                self.dst_shape[-2] +
                                                self.num_data - 1) //
                                               self.num_data, 0, 0)

                with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                                   align_loop == 0,
                                                   num_core_loop !=
                                                   core_loop - 1)):
                    self.data_rearrange_case_one(tik_instance, ub_ori,
                                                 ub_cast_fp16, ub_trans,
                                                 ub_cast_int8, align_loop,
                                                 is_last)

                    dst_gm_index = (num_fourth_last_axis -
                                    (align_loop - 1)) * num_data_one_loop
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_cast_int8[0],
                                           0, 1,
                                           align_loop * num_data_one_loop //
                                           self.num_data, 0, 0)

                with tik_instance.if_scope(num_core_loop == core_loop - 1):

                    self.data_rearrange_case_one(tik_instance, ub_ori,
                                                 ub_cast_fp16, ub_trans,
                                                 ub_cast_int8, remainder,
                                                 is_last)
                    dst_gm_index = (num_fourth_last_axis - (remainder - 1)) * \
                                   num_data_one_loop
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_cast_int8[0],
                                           0, 1,
                                           remainder * num_data_one_loop //
                                           self.num_data, 0, 0)

        return tik_instance

    def format_transfer_case_two(self, tik_instance):
        """
        the transfer process when UB can put in 16 * last axis data,
         16 < last axis % 32 < 32 and the shape of dst is 4-D
        """
        ub_ori_data = self.ub_memory
        ub_trans_data = ub_ori_data
        total_core_loop_num = self.dst_shape[-4]

        core_number = _set_core_num(total_core_loop_num)
        num_data_one_loop = self.dst_shape[-3] *  \
                            self.dst_shape[-2] * self.dst_shape[-1]

        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            ub_ori = tik_instance.Tensor("int8",
                                         (ub_ori_data,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_cast_fp16 = tik_instance.Tensor("float16",
                                               (ub_trans_data,),
                                               name="ub_cast_fp16",
                                               scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor("float16",
                                           (ub_trans_data,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            ub_cast_int8 = tik_instance.Tensor("int8",
                                               (ub_trans_data,),
                                               name="ub_cast_int8",
                                               scope=tik.scope_ubuf)
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            align_loop, remainder = _cal_core_loop(tik_instance,
                                                   num_data_one_loop,
                                                   core_loop, ub_ori_data)
            with tik_instance.for_range(0, core_loop) as num_core_loop:
                total_core_loop = sum_core + num_core_loop
                num_fourth_last_axis = total_core_loop
                is_last = tik_instance.Scalar("uint64", init_value=0)
                src_gm_index = num_fourth_last_axis * self.src_shape[-1] * \
                               CUBE_SIZE_2
                src_ub_index = (num_core_loop % align_loop) * \
                               num_data_one_loop
                with tik_instance.if_scope(num_fourth_last_axis ==
                                           self.dst_shape[-4] - 1):
                    is_last.set_as(1)
                    if self.src_shape[-2] % CUBE_SIZE_2 != 0:
                        with tik_instance.for_range(0, self.src_shape[-2] %
                                                    CUBE_SIZE_2) as \
                                num_cube_row:
                            tik_instance.data_move(ub_ori[src_ub_index +
                                                          num_cube_row *
                                                          self.dst_shape[-2] *
                                                          self.dst_shape[-3]],
                                                   self.src_gm
                                                   [src_gm_index +
                                                    num_cube_row *
                                                    self.src_shape[-1]],
                                                   0, 1,
                                                   (self.dst_shape[-3] *
                                                    self.dst_shape[-2] +
                                                    self.num_data - 1) //
                                                   self.num_data, 0, 0)
                    else:
                        with tik_instance.for_range(0, CUBE_SIZE_2) as \
                                num_cube_row:
                            tik_instance.data_move(ub_ori[src_ub_index +
                                                          num_cube_row *
                                                          self.dst_shape[-2] *
                                                          self.dst_shape[-3]],
                                                   self.src_gm
                                                   [src_gm_index +
                                                    num_cube_row *
                                                    self.src_shape[-1]],
                                                   0, 1,
                                                   (self.dst_shape[-3] *
                                                    self.dst_shape[-2] +
                                                    self.num_data - 1) //
                                                   self.num_data, 0, 0)
                with tik_instance.else_scope():
                    with tik_instance.for_range(0, CUBE_SIZE_2) as \
                            num_cube_row:
                        tik_instance.data_move(ub_ori[src_ub_index +
                                                      num_cube_row *
                                                      self.dst_shape[-2] *
                                                      self.dst_shape[-3]],
                                               self.src_gm
                                               [src_gm_index +
                                                num_cube_row *
                                                self.src_shape[-1]],
                                               0, 1,
                                               (self.dst_shape[-3] *
                                                self.dst_shape[-2] +
                                                self.num_data - 1) //
                                               self.num_data, 0, 0)

                with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                                   align_loop == 0,
                                                   num_core_loop !=
                                                   core_loop - 1)):
                    self.data_rearrange_case_zero(tik_instance, ub_ori,
                                                  ub_cast_fp16, ub_trans,
                                                  ub_cast_int8, align_loop,
                                                  is_last)

                    dst_gm_index = (num_fourth_last_axis -
                                    (align_loop - 1)) * num_data_one_loop
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_cast_int8[0],
                                           0, 1,
                                           align_loop * num_data_one_loop //
                                           self.num_data, 0, 0)

                with tik_instance.if_scope(num_core_loop == core_loop - 1):

                    self.data_rearrange_case_zero(tik_instance, ub_ori,
                                                  ub_cast_fp16, ub_trans,
                                                  ub_cast_int8, remainder,
                                                  is_last)
                    dst_gm_index = (num_fourth_last_axis - (remainder - 1)) * \
                                   num_data_one_loop
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_cast_int8[0],
                                           0, 1,
                                           remainder * num_data_one_loop //
                                           self.num_data, 0, 0)

        return tik_instance

    def format_transfer_case_three(self, tik_instance):
        """
        the transfer process when UB can not put in second last axis * 32 data
        and the shape of dst is 4-D
        """
        self.ub_memory = self.ub_memory - self.ub_memory % \
                         (CUBE_SIZE_2 * CUBE_SIZE_2)
        ub_ori_data = self.ub_memory
        ub_trans_data = ub_ori_data
        # loop_col is divisible by 2
        loop_col, loop_remainder = _cal_core_loop_python(
            CUBE_SIZE_2 * CUBE_SIZE, self.dst_shape[-3], ub_ori_data)
        num_data_one_loop = self.dst_shape[-4] * self.dst_shape[-3] * \
                            self.dst_shape[-2] * self.dst_shape[-1]
        loop_times = self.dst_shape[-4]
        if len(self.dst_shape) == 4:
            total_core_loop_num = loop_times
        else:
            total_core_loop_num = functools_reduce(lambda x1, x2: x1 * x2,
                                                   self.dst_shape[:-4]) * \
                                  loop_times
        core_number = _set_core_num(total_core_loop_num)

        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            ub_ori = tik_instance.Tensor("int8",
                                         (ub_ori_data,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_cast_fp16 = tik_instance.Tensor("float16",
                                               (ub_trans_data,),
                                               name="ub_cast_fp16",
                                               scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor("float16",
                                           (ub_trans_data,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            ub_cast_int8 = tik_instance.Tensor("int8",
                                               (ub_trans_data,),
                                               name="ub_cast_int8",
                                               scope=tik.scope_ubuf)
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            with tik_instance.for_range(0, core_loop) as num_core_loop:
                total_core_loop = sum_core + num_core_loop
                num_loop_time = total_core_loop % loop_times
                num_outer_axis = (total_core_loop - num_loop_time) //\
                                 loop_times
                is_last = tik_instance.Scalar("uint64", init_value=0)
                src_ub_index = 0

                with tik_instance.for_range(
                    0, self.dst_shape[-3] // loop_col) as num_loop_cube:
                    if self.src_shape[-1] % CUBE_SIZE_2 != 0 or \
                            (self.src_shape[-1] - loop_col * CUBE_SIZE) // \
                            self.num_data > MAX_STRIDE_BLK:
                        if self.src_shape[-2] % CUBE_SIZE_2 != 0:
                            with tik_instance.if_scope(num_loop_time ==
                                                       self.dst_shape[-4] - 1):
                                with tik_instance.for_range(
                                    0, self.src_shape[-2] % CUBE_SIZE_2)\
                                    as num_cube_row:
                                    src_gm_index = num_outer_axis * \
                                                   self.src_shape[-1] * \
                                                   self.src_shape[-2] + \
                                                   (num_loop_time *
                                                    CUBE_SIZE_2 +
                                                    num_cube_row) * \
                                                   self.src_shape[-1] + \
                                                   num_loop_cube * loop_col * \
                                                   CUBE_SIZE
                                    tik_instance.data_move(ub_ori
                                                           [loop_col *
                                                            CUBE_SIZE *
                                                            num_cube_row],
                                                           self.src_gm
                                                           [src_gm_index],
                                                           0, 1,
                                                           loop_col *
                                                           self.num_byte //
                                                           2, 0, 0)
                            with tik_instance.else_scope():
                                with tik_instance.for_range(0, CUBE_SIZE_2) \
                                        as num_cube_row:
                                    src_gm_index = num_outer_axis * \
                                                   self.src_shape[-1] * \
                                                   self.src_shape[-2] + \
                                                   (num_loop_time *
                                                    CUBE_SIZE_2 +
                                                    num_cube_row) * \
                                                   self.src_shape[-1] + \
                                                   num_loop_cube * loop_col * \
                                                   CUBE_SIZE
                                    tik_instance.data_move(ub_ori
                                                           [loop_col *
                                                            CUBE_SIZE *
                                                            num_cube_row],
                                                           self.src_gm
                                                           [src_gm_index],
                                                           0, 1,
                                                           loop_col *
                                                           self.num_byte // 2,
                                                           0, 0)
                        else:
                            with tik_instance.for_range(0, CUBE_SIZE_2) as \
                                    num_cube_row:
                                src_gm_index = num_outer_axis * \
                                               self.src_shape[-1] * \
                                               self.src_shape[-2] + \
                                               (num_loop_time * CUBE_SIZE_2 +
                                                num_cube_row) * \
                                               self.src_shape[-1] + \
                                               num_loop_cube * loop_col * \
                                               CUBE_SIZE
                                tik_instance.data_move(ub_ori
                                                       [loop_col * CUBE_SIZE *
                                                        num_cube_row],
                                                       self.src_gm
                                                       [src_gm_index],
                                                       0, 1,
                                                       loop_col *
                                                       self.num_byte // 2,
                                                       0, 0)
                    else:
                        src_gm_index = num_outer_axis * self.src_shape[-1] * \
                                       self.src_shape[-2] + num_loop_time * \
                                       CUBE_SIZE_2 * self.src_shape[-1] + \
                                       num_loop_cube * loop_col * CUBE_SIZE
                        if self.src_shape[-2] % CUBE_SIZE_2 != 0:
                            with tik_instance.if_scope(num_loop_time ==
                                                       self.dst_shape[-4] - 1):
                                tik_instance.data_move(ub_ori[src_ub_index],
                                                       self.src_gm
                                                       [src_gm_index],
                                                       0,
                                                       self.src_shape[-2] %
                                                       CUBE_SIZE_2,
                                                       loop_col *
                                                       self.num_byte // 2,
                                                       (self.src_shape[-1] -
                                                        loop_col *
                                                        CUBE_SIZE) //
                                                       self.num_data, 0)
                            with tik_instance.else_scope():
                                tik_instance.data_move(ub_ori[src_ub_index],
                                                       self.src_gm
                                                       [src_gm_index],
                                                       0, CUBE_SIZE_2,
                                                       loop_col *
                                                       self.num_byte // 2,
                                                       (self.src_shape[-1] -
                                                        loop_col *
                                                        CUBE_SIZE) //
                                                       self.num_data, 0)
                        else:
                            tik_instance.data_move(ub_ori[src_ub_index],
                                                   self.src_gm
                                                   [src_gm_index],
                                                   0, CUBE_SIZE_2,
                                                   loop_col * self.num_byte //
                                                   2,
                                                   (self.src_shape[-1] -
                                                    loop_col * CUBE_SIZE) //
                                                   self.num_data, 0)
                    with tik_instance.if_scope(tik.all(num_loop_cube ==
                                                       self.dst_shape[-3] //
                                                       loop_col - 1,
                                                       self.dst_shape[-3] %
                                                       loop_col == 0)):
                        is_last.set_as(1)
                    self.data_rearrange_case_two(tik_instance, ub_ori,
                                                 ub_cast_fp16, ub_trans,
                                                 ub_cast_int8,
                                                 num_loop_time,
                                                 loop_col, is_last)

                    dst_gm_index = num_outer_axis * num_data_one_loop + \
                                   num_loop_time * self.dst_shape[-1] * \
                                   self.dst_shape[-2] * self.dst_shape[-3] + \
                                   num_loop_cube * loop_col * \
                                   self.dst_shape[-1] * self.dst_shape[-2]

                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_cast_int8[0],
                                           0, 1,
                                           loop_col * self.dst_shape[-1] *
                                           self.dst_shape[-2] // self.num_data,
                                           0, 0)

                if loop_remainder != 0:
                    if self.src_shape[-1] % CUBE_SIZE_2 == 0 or \
                            CUBE_SIZE < self.src_shape[-1] % CUBE_SIZE_2 < \
                            CUBE_SIZE_2:
                        self.remainder_case_zero(tik_instance, ub_ori,
                                                 ub_cast_fp16, ub_trans,
                                                 ub_cast_int8, is_last,
                                                 num_outer_axis,
                                                 num_loop_time,
                                                 self.dst_shape[-3] //
                                                 loop_col,
                                                 loop_col, loop_remainder)
                    else:
                        self.remainder_case_one(tik_instance, ub_ori,
                                                ub_cast_fp16, ub_trans,
                                                ub_cast_int8, is_last,
                                                num_outer_axis,
                                                num_loop_time,
                                                self.dst_shape[-3] // loop_col,
                                                loop_col, loop_remainder)

        return tik_instance

    def remainder_case_zero(self, tik_instance, ub_ori, ub_cast_fp16, ub_trans,
                            ub_cast_int8, is_last, num_outer_axis,
                            num_loop_time, count_loop, loop_col,
                            loop_remainder):
        """
        the transfer process of remainder when the transfer case is 3 and
        last axis % 32 == 0 or 17 < last axis % 32 < 32
        """
        num_data_one_loop = self.dst_shape[-4] * self.dst_shape[-3] * \
                            self.dst_shape[-2] * self.dst_shape[-1]
        src_ub_index = 0
        is_last.set_as(1)
        if self.src_shape[-1] % CUBE_SIZE_2 != 0 or \
                (self.src_shape[-1] - loop_remainder * CUBE_SIZE) //  \
                self.num_data > MAX_STRIDE_BLK:
            if self.src_shape[-2] % CUBE_SIZE_2 != 0:
                with tik_instance.if_scope(num_loop_time ==
                                           self.dst_shape[-4] - 1):
                    with tik_instance.for_range(0, self.src_shape[-2] %
                                                CUBE_SIZE_2) as num_cube_row:
                        src_gm_index = num_outer_axis * self.src_shape[-1] * \
                                       self.src_shape[-2] + \
                                       (num_loop_time * CUBE_SIZE_2 +
                                        num_cube_row) * self.src_shape[-1] + \
                                       count_loop * loop_col * CUBE_SIZE
                        tik_instance.data_move(ub_ori[loop_remainder *
                                                      CUBE_SIZE *
                                                      num_cube_row],
                                               self.src_gm[src_gm_index],
                                               0, 1,
                                               (loop_remainder * CUBE_SIZE +
                                                self.num_data - 1) //
                                               self.num_data, 0, 0)
                with tik_instance.else_scope():
                    with tik_instance.for_range(0, CUBE_SIZE_2) \
                            as num_cube_row:
                        src_gm_index = num_outer_axis * self.src_shape[-1] * \
                                       self.src_shape[-2] + \
                                       (num_loop_time * CUBE_SIZE_2 +
                                        num_cube_row) * self.src_shape[-1] + \
                                       count_loop * loop_col * CUBE_SIZE
                        tik_instance.data_move(ub_ori[loop_remainder *
                                                      CUBE_SIZE *
                                                      num_cube_row],
                                               self.src_gm[src_gm_index],
                                               0, 1,
                                               (loop_remainder * CUBE_SIZE +
                                                self.num_data - 1) //
                                               self.num_data, 0, 0)
            else:
                with tik_instance.for_range(0, CUBE_SIZE_2) as num_cube_row:
                    src_gm_index = num_outer_axis * self.src_shape[-1] * \
                                   self.src_shape[-2] + \
                                   (num_loop_time * CUBE_SIZE_2 +
                                    num_cube_row) * self.src_shape[-1] +\
                                   count_loop * loop_col * CUBE_SIZE
                    tik_instance.data_move(ub_ori[loop_remainder * CUBE_SIZE *
                                                  num_cube_row],
                                           self.src_gm[src_gm_index],
                                           0, 1,
                                           (loop_remainder * CUBE_SIZE +
                                            self.num_data - 1) //
                                           self.num_data, 0, 0)
        else:
            src_gm_index = num_outer_axis * self.src_shape[-1] * \
                           self.src_shape[-2] + num_loop_time * CUBE_SIZE_2 * \
                           self.src_shape[-1] + count_loop * loop_col *\
                           CUBE_SIZE
            if self.src_shape[-2] % CUBE_SIZE_2 != 0:
                with tik_instance.if_scope(num_loop_time ==
                                           self.dst_shape[-4] - 1):
                    tik_instance.data_move(ub_ori[src_ub_index],
                                           self.src_gm[src_gm_index], 0,
                                           self.src_shape[-2] % CUBE_SIZE_2,
                                           loop_remainder * self.num_byte // 2,
                                           (self.src_shape[-1] -
                                            loop_remainder * CUBE_SIZE +
                                            self.num_data - 1) //
                                           self.num_data,
                                           0)
                with tik_instance.else_scope():
                    tik_instance.data_move(ub_ori[src_ub_index],
                                           self.src_gm[src_gm_index],
                                           0, CUBE_SIZE_2,
                                           loop_remainder * self.num_byte // 2,
                                           (self.src_shape[-1] -
                                            loop_remainder * CUBE_SIZE +
                                            self.num_data - 1) //
                                           self.num_data,
                                           0)
            else:
                tik_instance.data_move(ub_ori[src_ub_index],
                                       self.src_gm[src_gm_index],
                                       0, CUBE_SIZE_2,
                                       loop_remainder * self.num_byte // 2,
                                       (self.src_shape[-1] - loop_remainder *
                                        CUBE_SIZE + self.num_data - 1) //
                                       self.num_data, 0)

        self.data_rearrange_case_two(tik_instance, ub_ori, ub_cast_fp16,
                                     ub_trans, ub_cast_int8, num_loop_time,
                                     loop_remainder, is_last)

        dst_gm_index = num_outer_axis * num_data_one_loop + num_loop_time * \
                       self.dst_shape[-1] * self.dst_shape[-2] *  \
                       self.dst_shape[-3] + count_loop * loop_col *  \
                       self.dst_shape[-1] * self.dst_shape[-2]

        tik_instance.data_move(self.dst_gm[dst_gm_index],
                               ub_cast_int8[0],
                               0, 1,
                               loop_remainder * self.dst_shape[-1] *
                               self.dst_shape[-2] // self.num_data,
                               0, 0)

    def remainder_case_one(self, tik_instance, ub_ori, ub_cast_fp16,
                           ub_trans, ub_cast_int8, is_last, num_outer_axis,
                           num_loop_time, count_loop, loop_col,
                           loop_remainder):
        """
        the transfer process of remainder when the transfer case is 3 and
        0 < last axis % 32 <= 16
        """
        num_data_one_loop = self.dst_shape[-4] * self.dst_shape[-3] * \
                            self.dst_shape[-2] * self.dst_shape[-1]
        is_last.set_as(1)

        if self.src_shape[-2] % CUBE_SIZE_2 != 0:
            with tik_instance.if_scope(num_loop_time ==
                                       self.dst_shape[-4] - 1):
                with tik_instance.for_range(0, self.src_shape[-2] %
                                            CUBE_SIZE_2) as num_cube_row:
                    src_gm_index = num_outer_axis * self.src_shape[-1] * \
                                   self.src_shape[-2] + \
                                   (num_loop_time * CUBE_SIZE_2 +
                                    num_cube_row) * self.src_shape[-1] +\
                                   count_loop * loop_col * CUBE_SIZE
                    tik_instance.data_move(ub_ori[(loop_remainder + 1) *
                                                  CUBE_SIZE * num_cube_row],
                                           self.src_gm[src_gm_index],
                                           0, 1,
                                           (loop_remainder * CUBE_SIZE +
                                            self.num_data - 1) //
                                           self.num_data,
                                           0, 0)
            with tik_instance.else_scope():
                with tik_instance.for_range(0, CUBE_SIZE_2) as num_cube_row:
                    src_gm_index = num_outer_axis * self.src_shape[-1] *  \
                                   self.src_shape[-2] + \
                                   (num_loop_time * CUBE_SIZE_2 +
                                    num_cube_row) * self.src_shape[-1] +\
                                   count_loop * loop_col * CUBE_SIZE
                    tik_instance.data_move(ub_ori[(loop_remainder + 1) *
                                                  CUBE_SIZE * num_cube_row],
                                           self.src_gm[src_gm_index],
                                           0, 1,
                                           (loop_remainder * CUBE_SIZE +
                                            self.num_data - 1) //
                                           self.num_data,
                                           0, 0)
        else:
            with tik_instance.for_range(0, CUBE_SIZE_2) as num_cube_row:
                src_gm_index = num_outer_axis * self.src_shape[-1] *\
                               self.src_shape[-2] + \
                               (num_loop_time * CUBE_SIZE_2 + num_cube_row) * \
                               self.src_shape[-1] + count_loop * loop_col * \
                               CUBE_SIZE
                tik_instance.data_move(ub_ori[(loop_remainder + 1) *
                                              CUBE_SIZE * num_cube_row],
                                       self.src_gm[src_gm_index],
                                       0, 1,
                                       (loop_remainder * CUBE_SIZE +
                                        self.num_data - 1) // self.num_data,
                                       0, 0)

        self.data_rearrange_case_three(tik_instance, ub_ori, ub_cast_fp16,
                                       ub_trans, ub_cast_int8, num_loop_time,
                                       loop_remainder, is_last)

        dst_gm_index = num_outer_axis * num_data_one_loop + num_loop_time * \
                       self.dst_shape[-1] * self.dst_shape[-2] * \
                       self.dst_shape[-3] + count_loop * loop_col * \
                       self.dst_shape[-1] * self.dst_shape[-2]

        tik_instance.data_move(self.dst_gm[dst_gm_index],
                               ub_cast_int8[0],
                               0, 1,
                               loop_remainder * self.dst_shape[-1] *
                               self.dst_shape[-2] // self.num_data,
                               0, 0)

    def nd_2_zn_compute(self):
        """
        the overall data move process
        """
        tik_instance = self.set_tik_instance()
        format_transfer_case = self.set_format_transfer_case()
        if format_transfer_case == 0:
            tik_instance = self.format_transfer_case_zero(tik_instance)
        elif format_transfer_case == 1:
            tik_instance = self.format_transfer_case_one(tik_instance)
        elif format_transfer_case == 2:
            tik_instance = self.format_transfer_case_two(tik_instance)
        elif format_transfer_case == 3:
            tik_instance = self.format_transfer_case_three(tik_instance)
        return tik_instance

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        tik_instance = self.nd_2_zn_compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.src_gm],
                              outputs=[self.dst_gm])

        return tik_instance


@util.check_input_type(dict, dict, str, str, str)
def nd_2_zn_int8(src, dst, src_format, dst_format, kernel_name="nd_2_zn"):
    """
    algorithm: nd_2_Zn

    Parameters
    ----------
    src: dict
        dict with keys(shape, dtype) of src
    dst: dict
        dict with keys(shape, dtype) of dst
    src_format: str
        data format of src
    dst_format: str
        data format of dst
    kernel_name: str
        kernel name, default value is "nd_2_zn"

    Returns
    -------
    tik_instance: tik_instance
    """
    src_shape = src.get("shape")
    src_dtype = src.get("dtype").lower()
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(src_shape)
    util.check_tensor_shape_size(src_shape)
    check_list = ("int8")
    util.check_dtype_rule(src_dtype, check_list)

    if src_format.upper() != "ND":
        raise RuntimeError("nd_2_zn only support %s "
                           "while src format is %s" %
                           ("ND", src_format))

    if dst_format.upper() != "FRACTAL_Z":
        raise RuntimeError("nd_2_zn only support %s "
                           "while dst format is %s" %
                           ("FRACTAL_nZ", dst_format))

    src_shape = list(src_shape)
    nd_2_zn_template_int8 = ND2ZNComputeInt8(src_shape, src_dtype, kernel_name)

    return nd_2_zn_template_int8.get_tik_instance()
