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

nd_2_nz
"""
from functools import reduce as functools_reduce
from te import platform as tbe_platform
from topi.cce import util
from te import tik

# available ub size
TOTAL_UB_MEMORY = tbe_platform.cce_conf.get_soc_spec(
    tbe_platform.cce_conf.UB_SIZE)
# available number of cores
MAX_CORE_NUM = tbe_platform.cce_conf.get_soc_spec(
    tbe_platform.cce_conf.CORE_NUM)
# bytes of type int8
SIZE_ONE_BYTES = 1
# bytes of type float16
SIZE_TWO_BYTES = 2
# bytes of type float32
SIZE_FOUR_BYTES = 4
# size of the cube unit
CUBE_SIZE = 16
# size of the cube unit of last axis when dtype is int8
CUBE_SIZE_2 = 32
# minimum unit of data_move: 32Bytes
DATA_MOVE_MIN_UNIT = 32
# maximum repeat number
MAX_REPEATS = 255
# maximum burst number
MAX_BURST_NUMBER = 4095
# maximum rep stride
MAX_STRIDE_REP = 255
# maximum blk stride
MAX_STRIDE_BLK = 65535
# maximum mask
MAX_MASK = 128
# number of cubes processed by one vadds instruction
NUM_CUBE = MAX_MASK // CUBE_SIZE


# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-many-branches
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
                        core_loop *
                        (num_core - total_core_loop_num % MAX_CORE_NUM))

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
    align_loop.set_as((ub_ori + num_data_one_loop - 1) // num_data_one_loop)
    with tik_instance.if_scope((align_loop - 1) * core_loop *
                               num_data_one_loop >= ub_ori):
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
    remainder = core_loop % align_loop

    if align_loop > core_loop:
        align_loop = core_loop
        remainder = 0

    return align_loop, remainder


def _cal_core_loop_python_one(num_data_one_loop, core_loop, ub_ori):
    """
    calculate the number of loops and remainder on each core in another case
    and return python variable
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


class ND2NzCompute:
    """
    Rearranges data from ND format into FRACTAL_NZ format

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
    data_rearrange_case_zero:
        rearrange data when UB can put in second last axis * last axis data and
        the shape of dst is not 4-D
    data_rearrange_case_one:
        rearrange data when UB can not put in second last axis * last axis data
    data_rearrange_case_two:
        rearrange data when UB can not put in second last axis * 16 data
    data_rearrange_case_three:
        rearrange data when UB can put in last axis * 16 data and
        the shape of dst is 4-D
    data_rearrange_case_four:
        rearrange data when UB can not put in last axis * 16 data and
        the shape of dst is 4-D
    data_rearrange_case_five:
        rearrange data when UB can not put in last axis * 16 data and
        the shape of dst is 4-D
    def format_transfer_case_zero:
        the transfer process when the transfer case is 0
    def format_transfer_case_one:
        the transfer process when the transfer case is 1
    def format_transfer_case_two:
        the transfer process when the transfer case is 2
    def format_transfer_case_three:
        the transfer process when the transfer case is 3
    def format_transfer_case_four:
        the transfer process when the transfer case is 4
    def format_transfer_case_five:
        the transfer process when the transfer case is 5
    def format_transfer_case_six:
        the transfer process when the transfer case is 6
    def format_transfer_case_seven:
        the transfer process when the transfer case is 7
    def format_transfer_case_eight:
        the transfer process when the transfer case is 8
    nd_2_nz_compute:
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
        self.dst_shape[-2:] = [(self.src_shape[-1] + CUBE_SIZE - 1) //
                               CUBE_SIZE,
                               (self.src_shape[-2] + CUBE_SIZE - 1) //
                               CUBE_SIZE,
                               CUBE_SIZE, CUBE_SIZE]
        self.num_byte = SIZE_TWO_BYTES
        self.vadds_mask = MAX_MASK
        if self.dtype == "float32":
            self.num_byte = SIZE_FOUR_BYTES
            self.vadds_mask = MAX_MASK // 2

        # the number of data that can be moved in each data_move
        self.num_data = DATA_MOVE_MIN_UNIT // self.num_byte
        util.check_shape_rule(self.dst_shape)
        # the number of data that UB can put in
        self.ub_memory = min(TOTAL_UB_MEMORY, 252 * 1024) // self.num_byte // 2
        self.dst_gm = None
        self.src_gm = None

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
        format_transfer_case = 0
        if self.src_shape[-1] % CUBE_SIZE == 0:
            if self.dst_shape[-4] * self.dst_shape[-1] * self.dst_shape[-3] * \
                    (self.dst_shape[-2] + 1) > self.ub_memory:
                format_transfer_case = 2
        else:
            format_transfer_case = 1
            if self.dst_shape[-4] * self.dst_shape[-1] * self.dst_shape[-3] * \
                    (self.dst_shape[-2] + 1) > self.ub_memory:
                format_transfer_case = 3
        if (CUBE_SIZE * CUBE_SIZE * self.dst_shape[-3] + CUBE_SIZE) > \
                self.ub_memory:
            format_transfer_case = 4
            if (self.dst_shape[-3] - 1) * self.dst_shape[-1] * \
                    self.dst_shape[-2] // self.num_data <= MAX_STRIDE_BLK and \
                    ((CUBE_SIZE + 1) * CUBE_SIZE * self.dst_shape[-4]) >= \
                    self.ub_memory:
                format_transfer_case = 7
        is_four_d = 0
        if len(self.dst_shape) == 4:
            is_four_d = 1
        else:
            is_four_d = (functools_reduce(lambda x1, x2: x1 * x2,
                                          self.src_shape[:-2]) == 1)

        if is_four_d:
            if self.dst_shape[-4] * self.dst_shape[-1] * \
                    (self.dst_shape[-2] + 1) <= self.ub_memory and \
                    self.src_shape[-1] % CUBE_SIZE == 0:
                format_transfer_case = 5
            elif self.dst_shape[-4] * self.dst_shape[-1] * \
                    (self.dst_shape[-2] + 1) <= self.ub_memory and \
                    self.src_shape[-1] % CUBE_SIZE != 0:
                format_transfer_case = 6
            elif self.dst_shape[-4] * self.dst_shape[-1] * \
                    (self.dst_shape[-2] + 1) > self.ub_memory and \
                    (self.dst_shape[-3] - 1) * self.dst_shape[-1] * \
                    self.dst_shape[-2] // self.num_data <= MAX_STRIDE_BLK:
                format_transfer_case = 7
            if self.dst_shape[-4] * self.dst_shape[-1] * \
                    (self.dst_shape[-2] + 1) <= self.ub_memory // 2 and \
                    self.src_shape[-2] % (MAX_CORE_NUM * CUBE_SIZE) == 0 and \
                    self.src_shape[-1] % CUBE_SIZE == 0 and\
                    self.src_shape[-2] // (MAX_CORE_NUM * CUBE_SIZE) >= 2:
                format_transfer_case = 8

        return format_transfer_case

    def data_rearrange_case_zero(self, tik_instance, ub_ori, ub_trans,
                                 loop_num):
        """
        rearrange data when UB can put in second last axis * last axis data and
        the shape of dst is not 4-D
        """
        num_row_one_loop = loop_num * CUBE_SIZE * self.dst_shape[-3]
        scalar_zero = tik_instance.Scalar(dtype=self.dtype, init_value=0.0)
        if self.src_shape[-1] % CUBE_SIZE != 0:
            mask = 0
            for i, _ in enumerate(range(CUBE_SIZE -
                                        self.src_shape[-1] % CUBE_SIZE)):
                mask += 2 ** (CUBE_SIZE - 1 - i)

            with tik_instance.for_range(0, num_row_one_loop // MAX_REPEATS) \
                    as num_repeat:
                tik_instance.vector_dup([0, mask],
                                        ub_ori[(MAX_REPEATS * num_repeat + 1) *
                                               self.dst_shape[-4] *
                                               self.dst_shape[-1] - CUBE_SIZE],
                                        scalar_zero, MAX_REPEATS,
                                        0, self.dst_shape[-4] *
                                        self.dst_shape[-1] // self.num_data)
            with tik_instance.if_scope(num_row_one_loop % MAX_REPEATS != 0):
                tik_instance.vector_dup([0, mask],
                                        ub_ori[((num_row_one_loop //
                                                 MAX_REPEATS) *
                                                MAX_REPEATS + 1) *
                                               self.dst_shape[-4] *
                                               self.dst_shape[-1] - CUBE_SIZE],
                                        scalar_zero,
                                        num_row_one_loop % MAX_REPEATS,
                                        0, self.dst_shape[-4] *
                                        self.dst_shape[-1] // self.num_data)
        if (self.src_shape[-2] % CUBE_SIZE) != 0:
            with tik_instance.for_range(0, loop_num) as num_loop_index:
                with tik_instance.for_range(
                    0, self.dst_shape[-4] // NUM_CUBE) as num_col_cube:
                    tik_instance.vector_dup(self.vadds_mask,
                                            ub_ori[num_loop_index *
                                                   self.dst_shape[-3] *
                                                   self.dst_shape[-2] *
                                                   self.dst_shape[-4] *
                                                   self.dst_shape[-1] +
                                                   ((self.dst_shape[-3] - 1) *
                                                    self.dst_shape[-2] +
                                                    self.src_shape[-2] %
                                                    CUBE_SIZE) *
                                                   self.dst_shape[-1] *
                                                   self.dst_shape[-4] +
                                                   num_col_cube * NUM_CUBE *
                                                   CUBE_SIZE],
                                            scalar_zero,
                                            CUBE_SIZE - self.src_shape[-2] %
                                            CUBE_SIZE,
                                            self.num_byte // 2,
                                            self.dst_shape[-4] *
                                            self.dst_shape[-1] //
                                            self.num_data)
                    if self.vadds_mask == MAX_MASK // 2:
                        tik_instance.vector_dup(self.vadds_mask,
                                                ub_ori[num_loop_index *
                                                       self.dst_shape[-3] *
                                                       self.dst_shape[-2] *
                                                       self.dst_shape[-4] *
                                                       self.dst_shape[-1] +
                                                       ((self.dst_shape[-3] -
                                                         1) *
                                                        self.dst_shape[-2] +
                                                        self.src_shape[-2] %
                                                        CUBE_SIZE) *
                                                       self.dst_shape[-1] *
                                                       self.dst_shape[-4] +
                                                       num_col_cube *
                                                       NUM_CUBE * CUBE_SIZE +
                                                       CUBE_SIZE // 2],
                                                scalar_zero,
                                                CUBE_SIZE -
                                                self.src_shape[-2] %
                                                CUBE_SIZE,
                                                self.num_byte // 2,
                                                self.dst_shape[-4] *
                                                self.dst_shape[-1] //
                                                self.num_data)
                if self.dst_shape[-4] % NUM_CUBE != 0:
                    tik_instance.vector_dup((self.dst_shape[-4] % NUM_CUBE) *
                                            CUBE_SIZE * self.vadds_mask //
                                            MAX_MASK,
                                            ub_ori[num_loop_index *
                                                   self.dst_shape[-3] *
                                                   self.dst_shape[-2] *
                                                   self.dst_shape[-4] *
                                                   self.dst_shape[-1] +
                                                   ((self.dst_shape[-3] - 1) *
                                                    self.dst_shape[-2] +
                                                    self.src_shape[-2] %
                                                    CUBE_SIZE) *
                                                   self.dst_shape[-1] *
                                                   self.dst_shape[-4] +
                                                   self.dst_shape[-4] //
                                                   NUM_CUBE * NUM_CUBE *
                                                   CUBE_SIZE],
                                            scalar_zero,
                                            CUBE_SIZE - self.src_shape[-2] %
                                            CUBE_SIZE,
                                            self.num_byte // 2,
                                            self.dst_shape[-4] *
                                            self.dst_shape[-1] //
                                            self.num_data)
                    if self.vadds_mask == MAX_MASK // 2:
                        tik_instance.vector_dup((self.dst_shape[-4] %
                                                 NUM_CUBE) * CUBE_SIZE *
                                                self.vadds_mask // MAX_MASK,
                                                ub_ori[num_loop_index *
                                                       self.dst_shape[-3] *
                                                       self.dst_shape[-2] *
                                                       self.dst_shape[-4] *
                                                       self.dst_shape[-1] +
                                                       ((self.dst_shape[-3] -
                                                         1) *
                                                        self.dst_shape[-2] +
                                                        self.src_shape[-2] %
                                                        CUBE_SIZE) *
                                                       self.dst_shape[-1] *
                                                       self.dst_shape[-4] +
                                                       self.dst_shape[-4] //
                                                       NUM_CUBE * NUM_CUBE *
                                                       CUBE_SIZE +
                                                       CUBE_SIZE // 2],
                                                scalar_zero,
                                                CUBE_SIZE -
                                                self.src_shape[-2] %
                                                CUBE_SIZE,
                                                self.num_byte // 2,
                                                self.dst_shape[-4] *
                                                self.dst_shape[-1] //
                                                self.num_data)

        with tik_instance.for_range(0, loop_num) as num_loop_index:
            with tik_instance.for_range(0, self.dst_shape[-4] // NUM_CUBE) as \
                    num_col_cube:
                with tik_instance.for_range(
                    0, CUBE_SIZE * self.dst_shape[-3] // MAX_REPEATS) \
                    as num_repeat_one:
                    tik_instance.vadds(self.vadds_mask,
                                       ub_trans[num_loop_index *
                                                self.dst_shape[-3] *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-4] *
                                                self.dst_shape[-1] +
                                                num_loop_index *
                                                self.dst_shape[-4] *
                                                self.dst_shape[-1] +
                                                num_col_cube * NUM_CUBE *
                                                self.dst_shape[-3] *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                MAX_REPEATS * num_repeat_one *
                                                CUBE_SIZE + MAX_MASK *
                                                num_col_cube],
                                       ub_ori[num_loop_index *
                                              self.dst_shape[-4] *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-2] *
                                              self.dst_shape[-3] +
                                              MAX_REPEATS * num_repeat_one *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-4] +
                                              num_col_cube * NUM_CUBE *
                                              CUBE_SIZE],
                                       scalar_zero, MAX_REPEATS,
                                       self.dst_shape[-3] *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       self.dst_shape[-4] *
                                       self.dst_shape[-1] //
                                       self.num_data)
                    if self.vadds_mask == MAX_MASK // 2:
                        tik_instance.vadds(self.vadds_mask,
                                           ub_trans[num_loop_index *
                                                    self.dst_shape[-3] *
                                                    self.dst_shape[-2] *
                                                    self.dst_shape[-4] *
                                                    self.dst_shape[-1] +
                                                    num_loop_index *
                                                    self.dst_shape[-4] *
                                                    self.dst_shape[-1] +
                                                    num_col_cube * NUM_CUBE *
                                                    self.dst_shape[-3] *
                                                    self.dst_shape[-2] *
                                                    self.dst_shape[-1] +
                                                    MAX_REPEATS *
                                                    num_repeat_one *
                                                    CUBE_SIZE + MAX_MASK *
                                                    num_col_cube +
                                                    CUBE_SIZE // 2],
                                           ub_ori[num_loop_index *
                                                  self.dst_shape[-4] *
                                                  self.dst_shape[-1] *
                                                  self.dst_shape[-2] *
                                                  self.dst_shape[-3] +
                                                  MAX_REPEATS *
                                                  num_repeat_one *
                                                  self.dst_shape[-1] *
                                                  self.dst_shape[-4] +
                                                  num_col_cube * NUM_CUBE *
                                                  CUBE_SIZE + CUBE_SIZE // 2],
                                           scalar_zero, MAX_REPEATS,
                                           self.dst_shape[-3] *
                                           self.dst_shape[-2] *
                                           self.dst_shape[-1] //
                                           self.num_data +
                                           self.num_byte // 2,
                                           self.num_byte // 2,
                                           self.num_byte // 2,
                                           self.dst_shape[-4] *
                                           self.dst_shape[-1] // self.num_data)
                with tik_instance.if_scope((CUBE_SIZE * self.dst_shape[-3]) %
                                           MAX_REPEATS != 0):
                    tik_instance.vadds(self.vadds_mask,
                                       ub_trans[num_loop_index *
                                                self.dst_shape[-3] *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-4] *
                                                self.dst_shape[-1] +
                                                num_loop_index *
                                                self.dst_shape[-4] *
                                                self.dst_shape[-1] +
                                                num_col_cube * NUM_CUBE *
                                                self.dst_shape[-3] *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                (MAX_REPEATS *
                                                 ((CUBE_SIZE *
                                                   self.dst_shape[-3]) //
                                                  MAX_REPEATS)) * CUBE_SIZE +
                                                MAX_MASK * num_col_cube],
                                       ub_ori[num_loop_index *
                                              self.dst_shape[-4] *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-2] *
                                              self.dst_shape[-3] +
                                              (MAX_REPEATS *
                                               ((CUBE_SIZE *
                                                 self.dst_shape[-3]) //
                                                MAX_REPEATS)) *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-4] +
                                              num_col_cube * NUM_CUBE *
                                              CUBE_SIZE],
                                       scalar_zero,
                                       (CUBE_SIZE * self.dst_shape[-3]) %
                                       MAX_REPEATS,
                                       self.dst_shape[-3] *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       self.dst_shape[-4] *
                                       self.dst_shape[-1] // self.num_data)
                    if self.vadds_mask == MAX_MASK // 2:
                        tik_instance.vadds(self.vadds_mask,
                                           ub_trans[num_loop_index *
                                                    self.dst_shape[-3] *
                                                    self.dst_shape[-2] *
                                                    self.dst_shape[-4] *
                                                    self.dst_shape[-1] +
                                                    num_loop_index *
                                                    self.dst_shape[-4] *
                                                    self.dst_shape[-1] +
                                                    num_col_cube * NUM_CUBE *
                                                    self.dst_shape[-3] *
                                                    self.dst_shape[-2] *
                                                    self.dst_shape[-1] +
                                                    (MAX_REPEATS *
                                                     ((CUBE_SIZE *
                                                       self.dst_shape[-3]) //
                                                      MAX_REPEATS)) *
                                                    CUBE_SIZE +
                                                    MAX_MASK * num_col_cube +
                                                    CUBE_SIZE // 2],
                                           ub_ori[num_loop_index *
                                                  self.dst_shape[-4] *
                                                  self.dst_shape[-1] *
                                                  self.dst_shape[-2] *
                                                  self.dst_shape[-3] +
                                                  (MAX_REPEATS *
                                                   ((CUBE_SIZE *
                                                     self.dst_shape[-3]) //
                                                    MAX_REPEATS)) *
                                                  self.dst_shape[-1] *
                                                  self.dst_shape[-4] +
                                                  num_col_cube * NUM_CUBE *
                                                  CUBE_SIZE + CUBE_SIZE // 2],
                                           scalar_zero,
                                           (CUBE_SIZE * self.dst_shape[-3]) %
                                           MAX_REPEATS,
                                           self.dst_shape[-3] *
                                           self.dst_shape[-2] *
                                           self.dst_shape[-1] //
                                           self.num_data + self.num_byte // 2,
                                           self.num_byte // 2,
                                           self.num_byte // 2,
                                           self.dst_shape[-4] *
                                           self.dst_shape[-1] //
                                           self.num_data)
            if self.dst_shape[-4] % NUM_CUBE != 0:
                with tik_instance.for_range(
                    0, CUBE_SIZE * self.dst_shape[-3] // MAX_REPEATS) \
                    as num_repeat_one:
                    tik_instance.vadds((self.dst_shape[-4] % NUM_CUBE) *
                                       CUBE_SIZE * self.vadds_mask // MAX_MASK,
                                       ub_trans[num_loop_index *
                                                self.dst_shape[-3] *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-4] *
                                                self.dst_shape[-1] +
                                                num_loop_index *
                                                self.dst_shape[-4] *
                                                self.dst_shape[-1] +
                                                self.dst_shape[-4] //
                                                NUM_CUBE *
                                                NUM_CUBE * self.dst_shape[-3] *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                MAX_REPEATS * num_repeat_one *
                                                CUBE_SIZE +
                                                self.dst_shape[-4] //
                                                NUM_CUBE *
                                                MAX_MASK],
                                       ub_ori[num_loop_index *
                                              self.dst_shape[-4] *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-2] *
                                              self.dst_shape[-3] +
                                              MAX_REPEATS * num_repeat_one *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-4] +
                                              self.dst_shape[-4] // NUM_CUBE *
                                              NUM_CUBE * CUBE_SIZE],
                                       scalar_zero, MAX_REPEATS,
                                       self.dst_shape[-3] *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       self.dst_shape[-4] *
                                       self.dst_shape[-1] // self.num_data)
                    if self.vadds_mask == MAX_MASK // 2:
                        tik_instance.vadds((self.dst_shape[-4] % NUM_CUBE) *
                                           CUBE_SIZE * self.vadds_mask //
                                           MAX_MASK,
                                           ub_trans[num_loop_index *
                                                    self.dst_shape[-3] *
                                                    self.dst_shape[-2] *
                                                    self.dst_shape[-4] *
                                                    self.dst_shape[-1] +
                                                    num_loop_index *
                                                    self.dst_shape[-4] *
                                                    self.dst_shape[-1] +
                                                    self.dst_shape[-4] //
                                                    NUM_CUBE * NUM_CUBE *
                                                    self.dst_shape[-3] *
                                                    self.dst_shape[-2] *
                                                    self.dst_shape[-1] +
                                                    MAX_REPEATS *
                                                    num_repeat_one *
                                                    CUBE_SIZE +
                                                    self.dst_shape[-4] //
                                                    NUM_CUBE * MAX_MASK +
                                                    CUBE_SIZE // 2],
                                           ub_ori[num_loop_index *
                                                  self.dst_shape[-4] *
                                                  self.dst_shape[-1] *
                                                  self.dst_shape[-2] *
                                                  self.dst_shape[-3] +
                                                  MAX_REPEATS *
                                                  num_repeat_one *
                                                  self.dst_shape[-1] *
                                                  self.dst_shape[-4] +
                                                  self.dst_shape[-4] //
                                                  NUM_CUBE * NUM_CUBE *
                                                  CUBE_SIZE + CUBE_SIZE // 2],
                                           scalar_zero, MAX_REPEATS,
                                           self.dst_shape[-3] *
                                           self.dst_shape[-2] *
                                           self.dst_shape[-1] //
                                           self.num_data + self.num_byte // 2,
                                           self.num_byte // 2,
                                           self.num_byte // 2,
                                           self.dst_shape[-4] *
                                           self.dst_shape[-1] //
                                           self.num_data)
                with tik_instance.if_scope((CUBE_SIZE * self.dst_shape[-3]) %
                                           MAX_REPEATS != 0):
                    tik_instance.vadds((self.dst_shape[-4] % NUM_CUBE) *
                                       CUBE_SIZE * self.vadds_mask // MAX_MASK,
                                       ub_trans[num_loop_index *
                                                self.dst_shape[-3] *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-4] *
                                                self.dst_shape[-1] +
                                                num_loop_index *
                                                self.dst_shape[-4] *
                                                self.dst_shape[-1] +
                                                self.dst_shape[-4] //
                                                NUM_CUBE * NUM_CUBE *
                                                self.dst_shape[-3] *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                (MAX_REPEATS *
                                                 ((CUBE_SIZE *
                                                   self.dst_shape[-3]) //
                                                  MAX_REPEATS)) * CUBE_SIZE +
                                                self.dst_shape[-4] //
                                                NUM_CUBE * MAX_MASK],
                                       ub_ori[num_loop_index *
                                              self.dst_shape[-4] *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-2] *
                                              self.dst_shape[-3] +
                                              (MAX_REPEATS *
                                               ((CUBE_SIZE *
                                                 self.dst_shape[-3]) //
                                                MAX_REPEATS)) *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-4] +
                                              self.dst_shape[-4] // NUM_CUBE *
                                              NUM_CUBE * CUBE_SIZE],
                                       scalar_zero,
                                       (CUBE_SIZE * self.dst_shape[-3]) %
                                       MAX_REPEATS,
                                       self.dst_shape[-3] *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       self.dst_shape[-4] *
                                       self.dst_shape[-1] // self.num_data)
                    if self.vadds_mask == MAX_MASK // 2:
                        tik_instance.vadds((self.dst_shape[-4] % NUM_CUBE) *
                                           CUBE_SIZE * self.vadds_mask //
                                           MAX_MASK,
                                           ub_trans[num_loop_index *
                                                    self.dst_shape[-3] *
                                                    self.dst_shape[-2] *
                                                    self.dst_shape[-4] *
                                                    self.dst_shape[-1] +
                                                    num_loop_index *
                                                    self.dst_shape[-4] *
                                                    self.dst_shape[-1] +
                                                    self.dst_shape[-4] //
                                                    NUM_CUBE * NUM_CUBE *
                                                    self.dst_shape[-3] *
                                                    self.dst_shape[-2] *
                                                    self.dst_shape[-1] +
                                                    (MAX_REPEATS *
                                                     ((CUBE_SIZE *
                                                       self.dst_shape[-3]) //
                                                      MAX_REPEATS)) *
                                                    CUBE_SIZE +
                                                    self.dst_shape[-4] //
                                                    NUM_CUBE * MAX_MASK +
                                                    CUBE_SIZE // 2],
                                           ub_ori[num_loop_index *
                                                  self.dst_shape[-4] *
                                                  self.dst_shape[-1] *
                                                  self.dst_shape[-2] *
                                                  self.dst_shape[-3] +
                                                  (MAX_REPEATS *
                                                   ((CUBE_SIZE *
                                                     self.dst_shape[-3]) //
                                                    MAX_REPEATS)) *
                                                  self.dst_shape[-1] *
                                                  self.dst_shape[-4] +
                                                  self.dst_shape[-4] //
                                                  NUM_CUBE * NUM_CUBE *
                                                  CUBE_SIZE + CUBE_SIZE // 2],
                                           scalar_zero,
                                           (CUBE_SIZE * self.dst_shape[-3]) %
                                           MAX_REPEATS,
                                           self.dst_shape[-3] *
                                           self.dst_shape[-2] *
                                           self.dst_shape[-1] //
                                           self.num_data + self.num_byte // 2,
                                           self.num_byte // 2,
                                           self.num_byte // 2,
                                           self.dst_shape[-4] *
                                           self.dst_shape[-1] //
                                           self.num_data)

    def data_rearrange_case_one(self, tik_instance, ub_ori, ub_trans,
                                col_cube_num, is_last):
        """
        rearrange data when UB can not put in second last axis * last axis data
        """
        scalar_zero = tik_instance.Scalar(dtype=self.dtype, init_value=0.0)
        if self.src_shape[-1] % CUBE_SIZE != 0:
            with tik_instance.if_scope(is_last == 1):
                mask = 0
                for i, _ in enumerate(range(CUBE_SIZE - self.src_shape[-1] %
                                            CUBE_SIZE)):
                    mask += 2 ** (CUBE_SIZE - 1 - i)
                with tik_instance.for_range(0, self.src_shape[-2]) \
                        as num_row:
                    tik_instance.vector_dup([0, mask],
                                            ub_ori[(num_row + 1) *
                                                   col_cube_num *
                                                   self.dst_shape[-1] -
                                                   CUBE_SIZE],
                                            scalar_zero, 1, 0, 0)
        if self.src_shape[-2] % CUBE_SIZE != 0:
            with tik_instance.for_range(0, col_cube_num // NUM_CUBE) as \
                    num_col_cube:
                tik_instance.vector_dup(self.vadds_mask,
                                        ub_ori[((self.dst_shape[-3] - 1) *
                                                self.dst_shape[-2] +
                                                self.src_shape[-2] %
                                                CUBE_SIZE) *
                                               self.dst_shape[-1] *
                                               col_cube_num + num_col_cube *
                                               NUM_CUBE * CUBE_SIZE],
                                        scalar_zero, CUBE_SIZE -
                                        self.src_shape[-2] % CUBE_SIZE,
                                        self.num_byte // 2,
                                        col_cube_num * self.dst_shape[-1] //
                                        self.num_data)
                if self.vadds_mask == MAX_MASK // 2:
                    tik_instance.vector_dup(self.vadds_mask,
                                            ub_ori[((self.dst_shape[-3] - 1) *
                                                    self.dst_shape[-2] +
                                                    self.src_shape[-2] %
                                                    CUBE_SIZE) *
                                                   self.dst_shape[-1] *
                                                   col_cube_num +
                                                   num_col_cube * NUM_CUBE *
                                                   CUBE_SIZE + CUBE_SIZE // 2],
                                            scalar_zero, CUBE_SIZE -
                                            self.src_shape[-2] % CUBE_SIZE,
                                            self.num_byte // 2,
                                            col_cube_num *
                                            self.dst_shape[-1] //
                                            self.num_data)
            with tik_instance.if_scope(col_cube_num % NUM_CUBE != 0):
                tik_instance.vector_dup((col_cube_num % NUM_CUBE) * CUBE_SIZE *
                                        self.vadds_mask // MAX_MASK,
                                        ub_ori[((self.dst_shape[-3] - 1) *
                                                self.dst_shape[-2] +
                                                self.src_shape[-2] %
                                                CUBE_SIZE) *
                                               self.dst_shape[-1] *
                                               col_cube_num + col_cube_num //
                                               NUM_CUBE * NUM_CUBE *
                                               CUBE_SIZE],
                                        scalar_zero, CUBE_SIZE -
                                        self.src_shape[-2] % CUBE_SIZE,
                                        self.num_byte // 2,
                                        col_cube_num * self.dst_shape[-1] //
                                        self.num_data)
                if self.vadds_mask == MAX_MASK // 2:
                    tik_instance.vector_dup((col_cube_num % NUM_CUBE) *
                                            CUBE_SIZE * self.vadds_mask //
                                            MAX_MASK,
                                            ub_ori[((self.dst_shape[-3] - 1) *
                                                    self.dst_shape[-2] +
                                                    self.src_shape[-2] %
                                                    CUBE_SIZE) *
                                                   self.dst_shape[-1] *
                                                   col_cube_num +
                                                   col_cube_num // NUM_CUBE *
                                                   NUM_CUBE * CUBE_SIZE +
                                                   CUBE_SIZE // 2],
                                            scalar_zero, CUBE_SIZE -
                                            self.src_shape[-2] % CUBE_SIZE,
                                            self.num_byte // 2,
                                            col_cube_num *
                                            self.dst_shape[-1] //
                                            self.num_data)
        with tik_instance.for_range(0, col_cube_num // NUM_CUBE) as \
                num_col_cube:
            with tik_instance.for_range(
                0, CUBE_SIZE * self.dst_shape[-3] // MAX_REPEATS) \
                as num_repeat_one:
                tik_instance.vadds(self.vadds_mask,
                                   ub_trans[num_col_cube * NUM_CUBE *
                                            self.dst_shape[-3] *
                                            self.dst_shape[-2] *
                                            self.dst_shape[-1] + MAX_REPEATS *
                                            num_repeat_one * CUBE_SIZE +
                                            num_col_cube * MAX_MASK],
                                   ub_ori[MAX_REPEATS * num_repeat_one *
                                          self.dst_shape[-1] * col_cube_num +
                                          num_col_cube * NUM_CUBE * CUBE_SIZE],
                                   scalar_zero, MAX_REPEATS,
                                   self.dst_shape[-3] *
                                   self.dst_shape[-2] *
                                   self.dst_shape[-1] // self.num_data +
                                   self.num_byte // 2, self.num_byte // 2,
                                   self.num_byte // 2,
                                   col_cube_num * self.dst_shape[-1] //
                                   self.num_data)
                if self.vadds_mask == MAX_MASK // 2:
                    tik_instance.vadds(self.vadds_mask,
                                       ub_trans[num_col_cube * NUM_CUBE *
                                                self.dst_shape[-3] *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                MAX_REPEATS *
                                                num_repeat_one * CUBE_SIZE +
                                                num_col_cube * MAX_MASK +
                                                CUBE_SIZE // 2],
                                       ub_ori[MAX_REPEATS * num_repeat_one *
                                              self.dst_shape[-1] *
                                              col_cube_num + num_col_cube *
                                              NUM_CUBE * CUBE_SIZE +
                                              CUBE_SIZE // 2],
                                       scalar_zero, MAX_REPEATS,
                                       self.dst_shape[-3] *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       col_cube_num * self.dst_shape[-1] //
                                       self.num_data)
            with tik_instance.if_scope((CUBE_SIZE * self.dst_shape[-3]) %
                                       MAX_REPEATS != 0):
                tik_instance.vadds(self.vadds_mask,
                                   ub_trans[num_col_cube * NUM_CUBE *
                                            self.dst_shape[-3] *
                                            self.dst_shape[-2] *
                                            self.dst_shape[-1] +
                                            (MAX_REPEATS *
                                             ((CUBE_SIZE *
                                               self.dst_shape[-3]) //
                                              MAX_REPEATS)) * CUBE_SIZE +
                                            num_col_cube * MAX_MASK],
                                   ub_ori[(MAX_REPEATS *
                                           ((CUBE_SIZE * self.dst_shape[-3]) //
                                            MAX_REPEATS)) *
                                          self.dst_shape[-1] *
                                          col_cube_num + num_col_cube *
                                          NUM_CUBE * CUBE_SIZE],
                                   scalar_zero,
                                   (CUBE_SIZE * self.dst_shape[-3]) %
                                   MAX_REPEATS,
                                   self.dst_shape[-3] *
                                   self.dst_shape[-2] *
                                   self.dst_shape[-1] // self.num_data +
                                   self.num_byte // 2, self.num_byte // 2,
                                   self.num_byte // 2,
                                   col_cube_num * self.dst_shape[-1] //
                                   self.num_data)
                if self.vadds_mask == MAX_MASK // 2:
                    tik_instance.vadds(self.vadds_mask,
                                       ub_trans[num_col_cube * NUM_CUBE *
                                                self.dst_shape[-3] *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                (MAX_REPEATS *
                                                 ((CUBE_SIZE *
                                                   self.dst_shape[-3]) //
                                                  MAX_REPEATS)) * CUBE_SIZE +
                                                num_col_cube * MAX_MASK +
                                                CUBE_SIZE // 2],
                                       ub_ori[(MAX_REPEATS *
                                               ((CUBE_SIZE *
                                                 self.dst_shape[-3]) //
                                                MAX_REPEATS)) *
                                              self.dst_shape[-1] *
                                              col_cube_num +
                                              num_col_cube * NUM_CUBE *
                                              CUBE_SIZE + CUBE_SIZE // 2],
                                       scalar_zero,
                                       (CUBE_SIZE * self.dst_shape[-3]) %
                                       MAX_REPEATS,
                                       self.dst_shape[-3] *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       col_cube_num * self.dst_shape[-1] //
                                       self.num_data)
        with tik_instance.if_scope(col_cube_num % NUM_CUBE != 0):
            with tik_instance.for_range(
                0, CUBE_SIZE * self.dst_shape[-3] // MAX_REPEATS) \
                as num_repeat_one:
                tik_instance.vadds((col_cube_num % NUM_CUBE) * CUBE_SIZE *
                                   self.vadds_mask // MAX_MASK,
                                   ub_trans[col_cube_num // NUM_CUBE *
                                            NUM_CUBE * self.dst_shape[-3] *
                                            self.dst_shape[-2] *
                                            self.dst_shape[-1] + MAX_REPEATS *
                                            num_repeat_one * CUBE_SIZE +
                                            col_cube_num // NUM_CUBE *
                                            MAX_MASK],
                                   ub_ori[MAX_REPEATS * num_repeat_one *
                                          self.dst_shape[-1] * col_cube_num +
                                          col_cube_num // NUM_CUBE * NUM_CUBE *
                                          CUBE_SIZE],
                                   scalar_zero, MAX_REPEATS,
                                   self.dst_shape[-3] *
                                   self.dst_shape[-2] *
                                   self.dst_shape[-1] // self.num_data +
                                   self.num_byte // 2, self.num_byte // 2,
                                   self.num_byte // 2,
                                   col_cube_num * self.dst_shape[-1] //
                                   self.num_data)
                if self.vadds_mask == MAX_MASK // 2:
                    tik_instance.vadds((col_cube_num % NUM_CUBE) * CUBE_SIZE *
                                       self.vadds_mask // MAX_MASK,
                                       ub_trans[col_cube_num // NUM_CUBE *
                                                NUM_CUBE * self.dst_shape[-3] *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                MAX_REPEATS * num_repeat_one *
                                                CUBE_SIZE + col_cube_num //
                                                NUM_CUBE * MAX_MASK +
                                                CUBE_SIZE // 2],
                                       ub_ori[MAX_REPEATS * num_repeat_one *
                                              self.dst_shape[-1] *
                                              col_cube_num + col_cube_num //
                                              NUM_CUBE * NUM_CUBE *
                                              CUBE_SIZE + CUBE_SIZE // 2],
                                       scalar_zero, MAX_REPEATS,
                                       self.dst_shape[-3] *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       col_cube_num * self.dst_shape[-1] //
                                       self.num_data)
            with tik_instance.if_scope((CUBE_SIZE * self.dst_shape[-3]) %
                                       MAX_REPEATS != 0):
                tik_instance.vadds((col_cube_num % NUM_CUBE) * CUBE_SIZE *
                                   self.vadds_mask // MAX_MASK,
                                   ub_trans[col_cube_num // NUM_CUBE *
                                            NUM_CUBE * self.dst_shape[-3] *
                                            self.dst_shape[-2] *
                                            self.dst_shape[-1] +
                                            (MAX_REPEATS *
                                             ((CUBE_SIZE *
                                               self.dst_shape[-3]) //
                                              MAX_REPEATS)) * CUBE_SIZE +
                                            col_cube_num // NUM_CUBE *
                                            MAX_MASK],
                                   ub_ori[(MAX_REPEATS *
                                           ((CUBE_SIZE * self.dst_shape[-3]) //
                                            MAX_REPEATS)) *
                                          self.dst_shape[-1] *
                                          col_cube_num + col_cube_num //
                                          NUM_CUBE * NUM_CUBE *
                                          CUBE_SIZE],
                                   scalar_zero,
                                   (CUBE_SIZE * self.dst_shape[-3]) %
                                   MAX_REPEATS,
                                   self.dst_shape[-3] *
                                   self.dst_shape[-2] *
                                   self.dst_shape[-1] // self.num_data +
                                   self.num_byte // 2, self.num_byte // 2,
                                   self.num_byte // 2,
                                   col_cube_num * self.dst_shape[-1] //
                                   self.num_data)
                if self.vadds_mask == MAX_MASK // 2:
                    tik_instance.vadds((col_cube_num % NUM_CUBE) * CUBE_SIZE *
                                       self.vadds_mask // MAX_MASK,
                                       ub_trans[col_cube_num // NUM_CUBE *
                                                NUM_CUBE * self.dst_shape[-3] *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                (MAX_REPEATS *
                                                 ((CUBE_SIZE *
                                                   self.dst_shape[-3]) //
                                                  MAX_REPEATS)) * CUBE_SIZE +
                                                col_cube_num // NUM_CUBE *
                                                MAX_MASK + CUBE_SIZE // 2],
                                       ub_ori[(MAX_REPEATS *
                                               ((CUBE_SIZE *
                                                 self.dst_shape[-3]) //
                                                MAX_REPEATS)) *
                                              self.dst_shape[-1] *
                                              col_cube_num + col_cube_num //
                                              NUM_CUBE * NUM_CUBE *
                                              CUBE_SIZE + CUBE_SIZE // 2],
                                       scalar_zero,
                                       (CUBE_SIZE * self.dst_shape[-3]) %
                                       MAX_REPEATS,
                                       self.dst_shape[-3] *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       col_cube_num * self.dst_shape[-1] //
                                       self.num_data)

    def data_rearrange_case_two(self, tik_instance, ub_ori, num_loop_time,
                                loop_row, is_last):
        """
        rearrange data when UB can not put in second last axis * 16 data
        """
        scalar_zero = tik_instance.Scalar(dtype=self.dtype, init_value=0.0)
        with tik_instance.if_scope(num_loop_time == self.dst_shape[-4] - 1):
            if self.src_shape[-1] % CUBE_SIZE != 0:
                mask = 0
                for i, _ in enumerate(range(CUBE_SIZE - self.src_shape[-1] %
                                            CUBE_SIZE)):
                    mask += 2 ** (CUBE_SIZE - 1 - i)

                with tik_instance.for_range(0, loop_row * CUBE_SIZE //
                                            MAX_REPEATS) as num_repeat:
                    tik_instance.vector_dup([0, mask],
                                            ub_ori[MAX_REPEATS * num_repeat *
                                                   self.dst_shape[-1]],
                                            scalar_zero, MAX_REPEATS,
                                            0, self.dst_shape[-1] //
                                            self.num_data)
                with tik_instance.if_scope(loop_row * CUBE_SIZE %
                                           MAX_REPEATS != 0):
                    tik_instance.vector_dup([0, mask],
                                            ub_ori[(loop_row * CUBE_SIZE //
                                                    MAX_REPEATS) *
                                                   MAX_REPEATS *
                                                   self.dst_shape[-1]],
                                            scalar_zero, loop_row * CUBE_SIZE %
                                            MAX_REPEATS, 0,
                                            self.dst_shape[-1] //
                                            self.num_data)

        with tik_instance.if_scope(is_last == 1):
            if self.src_shape[-2] % CUBE_SIZE != 0:
                tik_instance.vector_dup(CUBE_SIZE,
                                        ub_ori[((loop_row - 1) *
                                                self.dst_shape[-2] +
                                                self.src_shape[-2] %
                                                CUBE_SIZE) *
                                               self.dst_shape[-1]],
                                        scalar_zero, CUBE_SIZE -
                                        self.src_shape[-2] % CUBE_SIZE, 0,
                                        self.num_byte // 2)

    def data_rearrange_case_three(self, tik_instance, ub_ori, ub_trans,
                                  loop_num, is_last):
        """
        rearrange data when UB can put in last axis * 16 data and
        the shape of dst is 4-D
        """
        num_row_one_loop = loop_num * CUBE_SIZE
        scalar_zero = tik_instance.Scalar(dtype=self.dtype, init_value=0.0)
        if self.src_shape[-1] % CUBE_SIZE != 0:
            mask = 0
            for i, _ in enumerate(range(CUBE_SIZE -
                                        self.src_shape[-1] % CUBE_SIZE)):
                mask += 2 ** (CUBE_SIZE - 1 - i)

            with tik_instance.for_range(0, num_row_one_loop // MAX_REPEATS) \
                    as num_repeat:
                tik_instance.vector_dup([0, mask],
                                        ub_ori[(MAX_REPEATS * num_repeat + 1) *
                                               self.dst_shape[-4] *
                                               self.dst_shape[-1] - CUBE_SIZE],
                                        scalar_zero, MAX_REPEATS,
                                        0, self.dst_shape[-4] *
                                        self.dst_shape[-1] // self.num_data)
            with tik_instance.if_scope(num_row_one_loop % MAX_REPEATS != 0):
                tik_instance.vector_dup([0, mask],
                                        ub_ori[((num_row_one_loop //
                                                 MAX_REPEATS) *
                                                MAX_REPEATS + 1) *
                                               self.dst_shape[-4] *
                                               self.dst_shape[-1] - CUBE_SIZE],
                                        scalar_zero,
                                        num_row_one_loop % MAX_REPEATS,
                                        0, self.dst_shape[-4] *
                                        self.dst_shape[-1] // self.num_data)
        with tik_instance.if_scope(is_last == 1):
            if (self.src_shape[-2] % CUBE_SIZE) != 0:
                with tik_instance.for_range(0, self.dst_shape[-4] //
                                            NUM_CUBE) as num_col_cube:
                    tik_instance.vector_dup(self.vadds_mask,
                                            ub_ori[((loop_num - 1) *
                                                    self.dst_shape[-2] +
                                                    self.src_shape[-2] %
                                                    CUBE_SIZE) *
                                                   self.dst_shape[-1] *
                                                   self.dst_shape[-4] +
                                                   num_col_cube * NUM_CUBE *
                                                   CUBE_SIZE],
                                            scalar_zero,
                                            CUBE_SIZE - self.src_shape[-2] %
                                            CUBE_SIZE,
                                            self.num_byte // 2,
                                            self.dst_shape[-4] *
                                            self.dst_shape[-1] //
                                            self.num_data)
                    if self.vadds_mask == MAX_MASK // 2:
                        tik_instance.vector_dup(self.vadds_mask,
                                                ub_ori[((loop_num - 1) *
                                                        self.dst_shape[-2] +
                                                        self.src_shape[-2] %
                                                        CUBE_SIZE) *
                                                       self.dst_shape[-1] *
                                                       self.dst_shape[-4] +
                                                       num_col_cube *
                                                       NUM_CUBE *
                                                       CUBE_SIZE +
                                                       CUBE_SIZE // 2],
                                                scalar_zero,
                                                CUBE_SIZE -
                                                self.src_shape[-2] % CUBE_SIZE,
                                                self.num_byte // 2,
                                                self.dst_shape[-4] *
                                                self.dst_shape[-1] //
                                                self.num_data)
                if self.dst_shape[-4] % NUM_CUBE != 0:
                    tik_instance.vector_dup((self.dst_shape[-4] % NUM_CUBE) *
                                            CUBE_SIZE * self.vadds_mask //
                                            MAX_MASK,
                                            ub_ori[((loop_num - 1) *
                                                    self.dst_shape[-2] +
                                                    self.src_shape[-2] %
                                                    CUBE_SIZE) *
                                                   self.dst_shape[-1] *
                                                   self.dst_shape[-4] +
                                                   self.dst_shape[-4] //
                                                   NUM_CUBE * NUM_CUBE *
                                                   CUBE_SIZE],
                                            scalar_zero,
                                            CUBE_SIZE - self.src_shape[-2] %
                                            CUBE_SIZE,
                                            self.num_byte // 2,
                                            self.dst_shape[-4] *
                                            self.dst_shape[-1] //
                                            self.num_data)
                    if self.vadds_mask == MAX_MASK // 2:
                        tik_instance.vector_dup((self.dst_shape[-4] %
                                                 NUM_CUBE) * CUBE_SIZE *
                                                self.vadds_mask // MAX_MASK,
                                                ub_ori[((loop_num - 1) *
                                                        self.dst_shape[-2] +
                                                        self.src_shape[-2] %
                                                        CUBE_SIZE) *
                                                       self.dst_shape[-1] *
                                                       self.dst_shape[-4] +
                                                       self.dst_shape[-4] //
                                                       NUM_CUBE * NUM_CUBE *
                                                       CUBE_SIZE +
                                                       CUBE_SIZE // 2],
                                                scalar_zero,
                                                CUBE_SIZE -
                                                self.src_shape[-2] % CUBE_SIZE,
                                                self.num_byte // 2,
                                                self.dst_shape[-4] *
                                                self.dst_shape[-1] //
                                                self.num_data)

        with tik_instance.for_range(0, self.dst_shape[-4] // NUM_CUBE) \
                as num_col_cube:
            with tik_instance.for_range(0, CUBE_SIZE * loop_num //
                                        MAX_REPEATS) as num_repeat_one:
                tik_instance.vadds(self.vadds_mask,
                                   ub_trans[num_col_cube * NUM_CUBE *
                                            loop_num * self.dst_shape[-2] *
                                            self.dst_shape[-1] +
                                            MAX_REPEATS *
                                            num_repeat_one * CUBE_SIZE +
                                            MAX_MASK * num_col_cube],
                                   ub_ori[MAX_REPEATS * num_repeat_one *
                                          self.dst_shape[-1] *
                                          self.dst_shape[-4] +
                                          num_col_cube * MAX_MASK],
                                   scalar_zero, MAX_REPEATS, loop_num *
                                   self.dst_shape[-2] *
                                   self.dst_shape[-1] // self.num_data +
                                   self.num_byte // 2, self.num_byte // 2,
                                   self.num_byte // 2,
                                   self.dst_shape[-4] * self.dst_shape[-1] //
                                   self.num_data)
                if self.vadds_mask == MAX_MASK // 2:
                    tik_instance.vadds(self.vadds_mask,
                                       ub_trans[num_col_cube * NUM_CUBE *
                                                loop_num *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                MAX_REPEATS *
                                                num_repeat_one * CUBE_SIZE +
                                                MAX_MASK * num_col_cube +
                                                CUBE_SIZE // 2],
                                       ub_ori[MAX_REPEATS * num_repeat_one *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-4] +
                                              num_col_cube * MAX_MASK +
                                              CUBE_SIZE // 2],
                                       scalar_zero, MAX_REPEATS, loop_num *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       self.dst_shape[-4] *
                                       self.dst_shape[-1] // self.num_data)
            with tik_instance.if_scope((CUBE_SIZE * loop_num) %
                                       MAX_REPEATS != 0):
                tik_instance.vadds(self.vadds_mask,
                                   ub_trans[num_col_cube * NUM_CUBE *
                                            loop_num * self.dst_shape[-2] *
                                            self.dst_shape[-1] +
                                            MAX_REPEATS *
                                            ((CUBE_SIZE * loop_num) //
                                             MAX_REPEATS) * CUBE_SIZE +
                                            MAX_MASK * num_col_cube],
                                   ub_ori[MAX_REPEATS *
                                          ((CUBE_SIZE * loop_num) //
                                           MAX_REPEATS) *
                                          self.dst_shape[-1] *
                                          self.dst_shape[-4] + num_col_cube *
                                          MAX_MASK],
                                   scalar_zero,
                                   (CUBE_SIZE * loop_num) % MAX_REPEATS,
                                   loop_num *
                                   self.dst_shape[-2] *
                                   self.dst_shape[-1] // self.num_data +
                                   self.num_byte // 2, self.num_byte // 2,
                                   self.num_byte // 2,
                                   self.dst_shape[-4] * self.dst_shape[-1] //
                                   self.num_data)
                if self.vadds_mask == MAX_MASK // 2:
                    tik_instance.vadds(self.vadds_mask,
                                       ub_trans[num_col_cube * NUM_CUBE *
                                                loop_num *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                MAX_REPEATS *
                                                ((CUBE_SIZE * loop_num) //
                                                 MAX_REPEATS) * CUBE_SIZE +
                                                MAX_MASK * num_col_cube +
                                                CUBE_SIZE // 2],
                                       ub_ori[MAX_REPEATS *
                                              ((CUBE_SIZE * loop_num) //
                                               MAX_REPEATS) *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-4] +
                                              num_col_cube * MAX_MASK +
                                              CUBE_SIZE // 2],
                                       scalar_zero,
                                       (CUBE_SIZE * loop_num) % MAX_REPEATS,
                                       loop_num *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       self.dst_shape[-4] *
                                       self.dst_shape[-1] //
                                       self.num_data)
        if self.dst_shape[-4] % NUM_CUBE != 0:
            with tik_instance.for_range(0, CUBE_SIZE * loop_num //
                                        MAX_REPEATS) as num_repeat_one:
                tik_instance.vadds((self.dst_shape[-4] % NUM_CUBE) *
                                   CUBE_SIZE * self.vadds_mask // MAX_MASK,
                                   ub_trans[self.dst_shape[-4] // NUM_CUBE *
                                            NUM_CUBE * loop_num *
                                            self.dst_shape[-2] *
                                            self.dst_shape[-1] + MAX_REPEATS *
                                            num_repeat_one * CUBE_SIZE +
                                            self.dst_shape[-4] // NUM_CUBE *
                                            MAX_MASK],
                                   ub_ori[MAX_REPEATS * num_repeat_one *
                                          self.dst_shape[-1] *
                                          self.dst_shape[-4] +
                                          self.dst_shape[-4] // NUM_CUBE *
                                          MAX_MASK],
                                   scalar_zero, MAX_REPEATS, loop_num *
                                   self.dst_shape[-2] *
                                   self.dst_shape[-1] // self.num_data +
                                   self.num_byte // 2, self.num_byte // 2,
                                   self.num_byte // 2,
                                   self.dst_shape[-4] * self.dst_shape[-1] //
                                   self.num_data)
                if self.vadds_mask == MAX_MASK // 2:
                    tik_instance.vadds((self.dst_shape[-4] % NUM_CUBE) *
                                       CUBE_SIZE * self.vadds_mask // MAX_MASK,
                                       ub_trans[self.dst_shape[-4] //
                                                NUM_CUBE * NUM_CUBE *
                                                loop_num *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                MAX_REPEATS *
                                                num_repeat_one * CUBE_SIZE +
                                                self.dst_shape[-4] //
                                                NUM_CUBE * MAX_MASK +
                                                CUBE_SIZE // 2],
                                       ub_ori[MAX_REPEATS * num_repeat_one *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-4] +
                                              self.dst_shape[-4] // NUM_CUBE *
                                              MAX_MASK + CUBE_SIZE // 2],
                                       scalar_zero, MAX_REPEATS, loop_num *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       self.dst_shape[-4] *
                                       self.dst_shape[-1] // self.num_data)
            with tik_instance.if_scope((CUBE_SIZE * loop_num) %
                                       MAX_REPEATS != 0):
                tik_instance.vadds((self.dst_shape[-4] % NUM_CUBE) *
                                   CUBE_SIZE * self.vadds_mask // MAX_MASK,
                                   ub_trans[self.dst_shape[-4] // NUM_CUBE *
                                            NUM_CUBE * loop_num *
                                            self.dst_shape[-2] *
                                            self.dst_shape[-1] +
                                            MAX_REPEATS *
                                            ((CUBE_SIZE * loop_num) //
                                             MAX_REPEATS) * CUBE_SIZE +
                                            self.dst_shape[-4] // NUM_CUBE *
                                            MAX_MASK],
                                   ub_ori[MAX_REPEATS *
                                          ((CUBE_SIZE * loop_num) //
                                           MAX_REPEATS) *
                                          self.dst_shape[-1] *
                                          self.dst_shape[-4] +
                                          self.dst_shape[-4] // NUM_CUBE *
                                          MAX_MASK],
                                   scalar_zero,
                                   (CUBE_SIZE * loop_num) % MAX_REPEATS,
                                   loop_num *
                                   self.dst_shape[-2] *
                                   self.dst_shape[-1] // self.num_data +
                                   self.num_byte // 2, self.num_byte // 2,
                                   self.num_byte // 2,
                                   self.dst_shape[-4] * self.dst_shape[-1] //
                                   self.num_data)
                if self.vadds_mask == MAX_MASK // 2:
                    tik_instance.vadds((self.dst_shape[-4] % NUM_CUBE) *
                                       CUBE_SIZE * self.vadds_mask // MAX_MASK,
                                       ub_trans[self.dst_shape[-4] //
                                                NUM_CUBE * NUM_CUBE *
                                                loop_num *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                MAX_REPEATS *
                                                ((CUBE_SIZE * loop_num) //
                                                 MAX_REPEATS) * CUBE_SIZE +
                                                self.dst_shape[-4] //
                                                NUM_CUBE * MAX_MASK +
                                                CUBE_SIZE // 2],
                                       ub_ori[MAX_REPEATS *
                                              ((CUBE_SIZE * loop_num) //
                                               MAX_REPEATS) *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-4] +
                                              self.dst_shape[-4] // NUM_CUBE *
                                              MAX_MASK + CUBE_SIZE // 2],
                                       scalar_zero,
                                       (CUBE_SIZE * loop_num) % MAX_REPEATS,
                                       loop_num *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       self.dst_shape[-4] *
                                       self.dst_shape[-1] //
                                       self.num_data)

    def data_rearrange_case_four(self, tik_instance, ub_ori, ub_trans,
                                 num_loop_time, loop_num, is_last):
        """
        rearrange data when UB can not put in last axis * 16 data and
        the shape of dst is 4-D
        """
        scalar_zero = tik_instance.Scalar(dtype=self.dtype, init_value=0.0)
        with tik_instance.if_scope(is_last == 1):
            if self.src_shape[-1] % CUBE_SIZE != 0:
                mask = 0
                for i, _ in enumerate(range(CUBE_SIZE -
                                            self.src_shape[-1] % CUBE_SIZE)):
                    mask += 2 ** (CUBE_SIZE - 1 - i)

                tik_instance.vector_dup([0, mask],
                                        ub_ori[loop_num * CUBE_SIZE -
                                               CUBE_SIZE],
                                        scalar_zero, CUBE_SIZE,
                                        0, loop_num * CUBE_SIZE //
                                        self.num_data)
        if (self.src_shape[-2] % CUBE_SIZE) != 0:
            with tik_instance.if_scope(num_loop_time == self.dst_shape[-3] - 1):
                with tik_instance.for_range(0, loop_num // NUM_CUBE) \
                        as num_col_cube:
                    tik_instance.vector_dup(self.vadds_mask,
                                            ub_ori[(self.src_shape[-2] %
                                                    CUBE_SIZE) *
                                                   self.dst_shape[-1] *
                                                   loop_num + num_col_cube *
                                                   NUM_CUBE *
                                                   CUBE_SIZE],
                                            scalar_zero,
                                            CUBE_SIZE - self.src_shape[-2] %
                                            CUBE_SIZE,
                                            self.num_byte // 2,
                                            loop_num * self.dst_shape[-1] //
                                            self.num_data)
                    if self.vadds_mask == MAX_MASK // 2:
                        tik_instance.vector_dup(self.vadds_mask,
                                                ub_ori[(self.src_shape[-2] %
                                                        CUBE_SIZE) *
                                                       self.dst_shape[-1] *
                                                       loop_num +
                                                       num_col_cube *
                                                       NUM_CUBE * CUBE_SIZE +
                                                       CUBE_SIZE // 2],
                                                scalar_zero,
                                                CUBE_SIZE -
                                                self.src_shape[-2] % CUBE_SIZE,
                                                self.num_byte // 2,
                                                loop_num *
                                                self.dst_shape[-1] //
                                                self.num_data)
                if loop_num % NUM_CUBE != 0:
                    tik_instance.vector_dup((loop_num % NUM_CUBE) *
                                            CUBE_SIZE * self.vadds_mask //
                                            MAX_MASK,
                                            ub_ori[(self.src_shape[-2] %
                                                    CUBE_SIZE) *
                                                   self.dst_shape[-1] *
                                                   loop_num + loop_num //
                                                   NUM_CUBE * NUM_CUBE *
                                                   CUBE_SIZE],
                                            scalar_zero,
                                            CUBE_SIZE - self.src_shape[-2] %
                                            CUBE_SIZE,
                                            self.num_byte // 2,
                                            loop_num * self.dst_shape[-1] //
                                            self.num_data)
                    if self.vadds_mask == MAX_MASK // 2:
                        tik_instance.vector_dup((loop_num % NUM_CUBE) *
                                                CUBE_SIZE * self.vadds_mask //
                                                MAX_MASK,
                                                ub_ori[(self.src_shape[-2] %
                                                        CUBE_SIZE) *
                                                       self.dst_shape[-1] *
                                                       loop_num + loop_num //
                                                       NUM_CUBE * NUM_CUBE *
                                                       CUBE_SIZE +
                                                       CUBE_SIZE // 2],
                                                scalar_zero,
                                                CUBE_SIZE -
                                                self.src_shape[-2] %
                                                CUBE_SIZE,
                                                self.num_byte // 2,
                                                loop_num *
                                                self.dst_shape[-1] //
                                                self.num_data)
        with tik_instance.for_range(0, loop_num // NUM_CUBE) \
                as num_col_cube:
            tik_instance.vadds(self.vadds_mask,
                               ub_trans[num_col_cube * NUM_CUBE *
                                        self.dst_shape[-2] *
                                        self.dst_shape[-1] +
                                        MAX_MASK * num_col_cube],
                               ub_ori[num_col_cube * MAX_MASK],
                               scalar_zero, CUBE_SIZE,
                               self.dst_shape[-2] * self.dst_shape[-1] //
                               self.num_data + self.num_byte // 2,
                               self.num_byte // 2, self.num_byte // 2,
                               loop_num * self.dst_shape[-1] // self.num_data)
            if self.vadds_mask == MAX_MASK // 2:
                tik_instance.vadds(self.vadds_mask,
                                   ub_trans[num_col_cube * NUM_CUBE *
                                            self.dst_shape[-2] *
                                            self.dst_shape[-1] +
                                            MAX_MASK * num_col_cube +
                                            CUBE_SIZE // 2],
                                   ub_ori[num_col_cube * MAX_MASK +
                                          CUBE_SIZE // 2],
                                   scalar_zero, CUBE_SIZE,
                                   self.dst_shape[-2] * self.dst_shape[-1] //
                                   self.num_data + self.num_byte // 2,
                                   self.num_byte // 2, self.num_byte // 2,
                                   loop_num * self.dst_shape[-1] //
                                   self.num_data)
        if loop_num % NUM_CUBE != 0:
            tik_instance.vadds((loop_num % NUM_CUBE) * CUBE_SIZE *
                               self.vadds_mask // MAX_MASK,
                               ub_trans[loop_num // NUM_CUBE * NUM_CUBE *
                                        self.dst_shape[-2] *
                                        self.dst_shape[-1] + loop_num //
                                        NUM_CUBE * MAX_MASK],
                               ub_ori[loop_num // NUM_CUBE * MAX_MASK],
                               scalar_zero, CUBE_SIZE,
                               self.dst_shape[-2] * self.dst_shape[-1] //
                               self.num_data + self.num_byte // 2,
                               self.num_byte // 2, self.num_byte // 2,
                               loop_num * self.dst_shape[-1] // self.num_data)
            if self.vadds_mask == MAX_MASK // 2:
                tik_instance.vadds((loop_num % NUM_CUBE) * CUBE_SIZE *
                                   self.vadds_mask // MAX_MASK,
                                   ub_trans[loop_num // NUM_CUBE *
                                            NUM_CUBE * self.dst_shape[-2] *
                                            self.dst_shape[-1] +
                                            loop_num // NUM_CUBE *
                                            MAX_MASK + CUBE_SIZE // 2],
                                   ub_ori[loop_num // NUM_CUBE *
                                          MAX_MASK + CUBE_SIZE // 2],
                                   scalar_zero, CUBE_SIZE,
                                   self.dst_shape[-2] * self.dst_shape[-1] //
                                   self.num_data +
                                   self.num_byte // 2, self.num_byte // 2,
                                   self.num_byte // 2,
                                   loop_num * self.dst_shape[-1] //
                                   self.num_data)

    def data_rearrange_case_five(self, tik_instance, ub_ori, ub_trans,
                                 loop_num):
        """
        rearrange data when UB // 2 can put in last axis * 16 data and
        the shape of dst is 4-D
        """
        scalar_zero = tik_instance.Scalar(dtype=self.dtype, init_value=0.0)
        with tik_instance.for_range(0, self.dst_shape[-4] // NUM_CUBE) \
                as num_col_cube:
            with tik_instance.for_range(0, CUBE_SIZE * loop_num //
                                        MAX_REPEATS) as num_repeat_one:
                tik_instance.vadds(self.vadds_mask,
                                   ub_trans[num_col_cube * NUM_CUBE *
                                            loop_num * self.dst_shape[-2] *
                                            self.dst_shape[-1] +
                                            MAX_REPEATS *
                                            num_repeat_one * CUBE_SIZE +
                                            MAX_MASK * num_col_cube],
                                   ub_ori[MAX_REPEATS * num_repeat_one *
                                          self.dst_shape[-1] *
                                          self.dst_shape[-4] +
                                          num_col_cube * MAX_MASK],
                                   scalar_zero, MAX_REPEATS, loop_num *
                                   self.dst_shape[-2] *
                                   self.dst_shape[-1] // self.num_data +
                                   self.num_byte // 2, self.num_byte // 2,
                                   self.num_byte // 2,
                                   self.dst_shape[-4] * self.dst_shape[-1] //
                                   self.num_data)
                if self.vadds_mask == MAX_MASK // 2:
                    tik_instance.vadds(self.vadds_mask,
                                       ub_trans[num_col_cube * NUM_CUBE *
                                                loop_num *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                MAX_REPEATS *
                                                num_repeat_one * CUBE_SIZE +
                                                MAX_MASK * num_col_cube +
                                                CUBE_SIZE // 2],
                                       ub_ori[MAX_REPEATS * num_repeat_one *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-4] +
                                              num_col_cube * MAX_MASK +
                                              CUBE_SIZE // 2],
                                       scalar_zero, MAX_REPEATS, loop_num *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       self.dst_shape[-4] *
                                       self.dst_shape[-1] // self.num_data)
            with tik_instance.if_scope((CUBE_SIZE * loop_num) %
                                       MAX_REPEATS != 0):
                tik_instance.vadds(self.vadds_mask,
                                   ub_trans[num_col_cube * NUM_CUBE *
                                            loop_num * self.dst_shape[-2] *
                                            self.dst_shape[-1] +
                                            MAX_REPEATS *
                                            ((CUBE_SIZE * loop_num) //
                                             MAX_REPEATS) * CUBE_SIZE +
                                            MAX_MASK * num_col_cube],
                                   ub_ori[MAX_REPEATS *
                                          ((CUBE_SIZE * loop_num) //
                                           MAX_REPEATS) *
                                          self.dst_shape[-1] *
                                          self.dst_shape[-4] + num_col_cube *
                                          MAX_MASK],
                                   scalar_zero,
                                   (CUBE_SIZE * loop_num) % MAX_REPEATS,
                                   loop_num *
                                   self.dst_shape[-2] *
                                   self.dst_shape[-1] // self.num_data +
                                   self.num_byte // 2, self.num_byte // 2,
                                   self.num_byte // 2,
                                   self.dst_shape[-4] * self.dst_shape[-1] //
                                   self.num_data)
                if self.vadds_mask == MAX_MASK // 2:
                    tik_instance.vadds(self.vadds_mask,
                                       ub_trans[num_col_cube * NUM_CUBE *
                                                loop_num *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                MAX_REPEATS *
                                                ((CUBE_SIZE * loop_num) //
                                                 MAX_REPEATS) * CUBE_SIZE +
                                                MAX_MASK * num_col_cube +
                                                CUBE_SIZE // 2],
                                       ub_ori[MAX_REPEATS *
                                              ((CUBE_SIZE * loop_num) //
                                               MAX_REPEATS) *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-4] +
                                              num_col_cube * MAX_MASK +
                                              CUBE_SIZE // 2],
                                       scalar_zero,
                                       (CUBE_SIZE * loop_num) % MAX_REPEATS,
                                       loop_num *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       self.dst_shape[-4] *
                                       self.dst_shape[-1] //
                                       self.num_data)
        if self.dst_shape[-4] % NUM_CUBE != 0:
            with tik_instance.for_range(0, CUBE_SIZE * loop_num //
                                        MAX_REPEATS) as num_repeat_one:
                tik_instance.vadds((self.dst_shape[-4] % NUM_CUBE) *
                                   CUBE_SIZE * self.vadds_mask // MAX_MASK,
                                   ub_trans[self.dst_shape[-4] // NUM_CUBE *
                                            NUM_CUBE * loop_num *
                                            self.dst_shape[-2] *
                                            self.dst_shape[-1] + MAX_REPEATS *
                                            num_repeat_one * CUBE_SIZE +
                                            self.dst_shape[-4] // NUM_CUBE *
                                            MAX_MASK],
                                   ub_ori[MAX_REPEATS * num_repeat_one *
                                          self.dst_shape[-1] *
                                          self.dst_shape[-4] +
                                          self.dst_shape[-4] // NUM_CUBE *
                                          MAX_MASK],
                                   scalar_zero, MAX_REPEATS, loop_num *
                                   self.dst_shape[-2] *
                                   self.dst_shape[-1] // self.num_data +
                                   self.num_byte // 2, self.num_byte // 2,
                                   self.num_byte // 2,
                                   self.dst_shape[-4] * self.dst_shape[-1] //
                                   self.num_data)
                if self.vadds_mask == MAX_MASK // 2:
                    tik_instance.vadds((self.dst_shape[-4] % NUM_CUBE) *
                                       CUBE_SIZE * self.vadds_mask // MAX_MASK,
                                       ub_trans[self.dst_shape[-4] //
                                                NUM_CUBE * NUM_CUBE *
                                                loop_num *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                MAX_REPEATS *
                                                num_repeat_one * CUBE_SIZE +
                                                self.dst_shape[-4] //
                                                NUM_CUBE * MAX_MASK +
                                                CUBE_SIZE // 2],
                                       ub_ori[MAX_REPEATS * num_repeat_one *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-4] +
                                              self.dst_shape[-4] // NUM_CUBE *
                                              MAX_MASK + CUBE_SIZE // 2],
                                       scalar_zero, MAX_REPEATS, loop_num *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       self.dst_shape[-4] *
                                       self.dst_shape[-1] // self.num_data)
            with tik_instance.if_scope((CUBE_SIZE * loop_num) %
                                       MAX_REPEATS != 0):
                tik_instance.vadds((self.dst_shape[-4] % NUM_CUBE) *
                                   CUBE_SIZE * self.vadds_mask // MAX_MASK,
                                   ub_trans[self.dst_shape[-4] // NUM_CUBE *
                                            NUM_CUBE * loop_num *
                                            self.dst_shape[-2] *
                                            self.dst_shape[-1] +
                                            MAX_REPEATS *
                                            ((CUBE_SIZE * loop_num) //
                                             MAX_REPEATS) * CUBE_SIZE +
                                            self.dst_shape[-4] // NUM_CUBE *
                                            MAX_MASK],
                                   ub_ori[MAX_REPEATS *
                                          ((CUBE_SIZE * loop_num) //
                                           MAX_REPEATS) *
                                          self.dst_shape[-1] *
                                          self.dst_shape[-4] +
                                          self.dst_shape[-4] // NUM_CUBE *
                                          MAX_MASK],
                                   scalar_zero,
                                   (CUBE_SIZE * loop_num) % MAX_REPEATS,
                                   loop_num *
                                   self.dst_shape[-2] *
                                   self.dst_shape[-1] // self.num_data +
                                   self.num_byte // 2, self.num_byte // 2,
                                   self.num_byte // 2,
                                   self.dst_shape[-4] * self.dst_shape[-1] //
                                   self.num_data)
                if self.vadds_mask == MAX_MASK // 2:
                    tik_instance.vadds((self.dst_shape[-4] % NUM_CUBE) *
                                       CUBE_SIZE * self.vadds_mask // MAX_MASK,
                                       ub_trans[self.dst_shape[-4] //
                                                NUM_CUBE * NUM_CUBE *
                                                loop_num *
                                                self.dst_shape[-2] *
                                                self.dst_shape[-1] +
                                                MAX_REPEATS *
                                                ((CUBE_SIZE * loop_num) //
                                                 MAX_REPEATS) * CUBE_SIZE +
                                                self.dst_shape[-4] //
                                                NUM_CUBE * MAX_MASK +
                                                CUBE_SIZE // 2],
                                       ub_ori[MAX_REPEATS *
                                              ((CUBE_SIZE * loop_num) //
                                               MAX_REPEATS) *
                                              self.dst_shape[-1] *
                                              self.dst_shape[-4] +
                                              self.dst_shape[-4] // NUM_CUBE *
                                              MAX_MASK + CUBE_SIZE // 2],
                                       scalar_zero,
                                       (CUBE_SIZE * loop_num) % MAX_REPEATS,
                                       loop_num *
                                       self.dst_shape[-2] *
                                       self.dst_shape[-1] // self.num_data +
                                       self.num_byte // 2, self.num_byte // 2,
                                       self.num_byte // 2,
                                       self.dst_shape[-4] *
                                       self.dst_shape[-1] //
                                       self.num_data)

    def format_transfer_case_zero(self, tik_instance):
        """
        the transfer process when UB can put in
        second last axis * last axis data, last axis is 32B align, and
        the shape of dst is not 4-D
        """
        ub_ori_data = self.ub_memory
        ub_trans_data = ub_ori_data
        if len(self.dst_shape) == 4:
            total_core_loop_num = 1
        else:
            total_core_loop_num = functools_reduce(lambda x1, x2: x1 * x2,
                                                   self.dst_shape[:-4])
        core_number = _set_core_num(total_core_loop_num)
        num_data_one_loop = self.dst_shape[-4] * self.dst_shape[-3] * \
                            self.dst_shape[-2] * self.dst_shape[-1]

        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (ub_ori_data,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (ub_trans_data,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            align_loop, remainder = _cal_core_loop(tik_instance,
                                                   num_data_one_loop +
                                                   self.dst_shape[-4] *
                                                   self.dst_shape[-1],
                                                   core_loop, ub_ori_data)
            with tik_instance.for_range(0, core_loop) as num_core_loop:

                total_core_loop = sum_core + num_core_loop
                num_outer_axis = total_core_loop
                src_gm_index = num_outer_axis * self.src_shape[-1] * \
                               self.src_shape[-2]
                if self.src_shape[-2] % CUBE_SIZE != 0:
                    tik_instance.data_move(ub_ori[(num_core_loop %
                                                   align_loop) *
                                                  num_data_one_loop],
                                           self.src_gm[src_gm_index],
                                           0, 1,
                                           self.dst_shape[-1] *
                                           self.dst_shape[-4] *
                                           self.src_shape[-2] // self.num_data,
                                           0, 0)
                    with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                                       align_loop == 0,
                                                       num_core_loop !=
                                                       core_loop - 1)):

                        self.data_rearrange_case_zero(tik_instance, ub_ori,
                                                      ub_trans, align_loop)
                        dst_gm_index = num_outer_axis * num_data_one_loop - \
                                       (align_loop - 1) * num_data_one_loop
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans[0], 0, align_loop *
                                               self.dst_shape[-4],
                                               num_data_one_loop //
                                               self.dst_shape[-4] //
                                               self.num_data,
                                               self.num_byte // 2, 0)
                    with tik_instance.if_scope(num_core_loop == core_loop - 1):
                        self.data_rearrange_case_zero(tik_instance, ub_ori,
                                                      ub_trans, remainder)
                        dst_gm_index = num_outer_axis * num_data_one_loop - \
                                       (remainder - 1) * num_data_one_loop
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans[0], 0, remainder *
                                               self.dst_shape[-4],
                                               num_data_one_loop //
                                               self.dst_shape[-4] //
                                               self.num_data,
                                               self.num_byte // 2, 0)
                else:
                    with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                                       align_loop == 0,
                                                       num_core_loop !=
                                                       core_loop - 1)):
                        tik_instance.data_move(ub_ori[0],
                                               self.src_gm[src_gm_index - \
                                                           (align_loop - 1) *
                                                           num_data_one_loop],
                                               0, 1,
                                               align_loop *
                                               self.dst_shape[-1] *
                                               self.dst_shape[-4] *
                                               self.src_shape[-2] //
                                               self.num_data,
                                               0, 0)
                        self.data_rearrange_case_zero(tik_instance, ub_ori,
                                                      ub_trans, align_loop)
                        dst_gm_index = num_outer_axis * num_data_one_loop - \
                                       (align_loop - 1) * num_data_one_loop
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans[0], 0, align_loop *
                                               self.dst_shape[-4],
                                               num_data_one_loop //
                                               self.dst_shape[-4] //
                                               self.num_data,
                                               self.num_byte // 2, 0)
                    with tik_instance.if_scope(num_core_loop == core_loop - 1):
                        tik_instance.data_move(ub_ori[0],
                                               self.src_gm[src_gm_index - \
                                                           (remainder - 1) *
                                                           num_data_one_loop],
                                               0, 1,
                                               remainder * self.dst_shape[-1] *
                                               self.dst_shape[-4] *
                                               self.src_shape[-2] //
                                               self.num_data,
                                               0, 0)
                        self.data_rearrange_case_zero(tik_instance, ub_ori,
                                                      ub_trans, remainder)
                        dst_gm_index = num_outer_axis * num_data_one_loop - \
                                       (remainder - 1) * num_data_one_loop
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans[0], 0, remainder *
                                               self.dst_shape[-4],
                                               num_data_one_loop //
                                               self.dst_shape[-4] //
                                               self.num_data,
                                               self.num_byte // 2, 0)

        return tik_instance

    def format_transfer_case_one(self, tik_instance):
        """
        the transfer process when UB can put in
        second last axis * last axis data, last axis is not 32B align, and
        the shape of dst is not 4-D
        """
        ub_ori_data = self.ub_memory
        ub_trans_data = ub_ori_data

        if len(self.dst_shape) == 4:
            total_core_loop_num = 1
        else:
            total_core_loop_num = functools_reduce(lambda x1, x2: x1 * x2,
                                                   self.dst_shape[:-4])
        core_number = _set_core_num(total_core_loop_num)
        num_data_one_loop = self.dst_shape[-4] * self.dst_shape[-3] * \
                            self.dst_shape[-2] * self.dst_shape[-1]

        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (ub_ori_data,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (ub_trans_data,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            with tik_instance.for_range(0, core_loop) as num_core_loop:
                total_core_loop = sum_core + num_core_loop
                num_outer_axis = total_core_loop
                align_loop, remainder = _cal_core_loop(tik_instance,
                                                       num_data_one_loop +
                                                       self.dst_shape[-4] *
                                                       self.dst_shape[-1],
                                                       core_loop,
                                                       ub_ori_data)
                src_gm_index = num_outer_axis * self.src_shape[-1] * \
                               self.src_shape[-2]
                src_ub_index = (num_core_loop % align_loop) * num_data_one_loop

                with tik_instance.for_range(0, self.dst_shape[-3]) as num_cube:
                    with tik_instance.for_range(0, CUBE_SIZE) as num_cube_row:
                        with tik_instance.if_scope(num_cube * CUBE_SIZE +
                                                   num_cube_row >
                                                   self.src_shape[-2]):
                            pass
                        with tik_instance.else_scope():
                            tik_instance.data_move(ub_ori[src_ub_index +
                                                          (num_cube *
                                                           CUBE_SIZE +
                                                           num_cube_row) *
                                                          self.dst_shape[-1] *
                                                          self.dst_shape[-4]],
                                                   self.src_gm
                                                   [src_gm_index +
                                                    (num_cube * CUBE_SIZE +
                                                     num_cube_row) *
                                                    self.src_shape[-1]],
                                                   0, 1,
                                                   num_data_one_loop //
                                                   self.dst_shape[-2] //
                                                   self.dst_shape[-3] //
                                                   self.num_data, 0, 0)
                # move data from ub to gm when ub is full
                with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                                   align_loop == 0,
                                                   num_core_loop !=
                                                   core_loop - 1)):
                    self.data_rearrange_case_zero(tik_instance, ub_ori,
                                                  ub_trans, align_loop)
                    dst_gm_index = num_outer_axis * num_data_one_loop - \
                                   (align_loop - 1) * num_data_one_loop
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_trans[0], 0, align_loop *
                                           self.dst_shape[-4],
                                           num_data_one_loop //
                                           self.dst_shape[-4] // self.num_data,
                                           self.num_byte // 2, 0)
                # move the remaining data
                with tik_instance.if_scope(num_core_loop == core_loop - 1):
                    self.data_rearrange_case_zero(tik_instance, ub_ori,
                                                  ub_trans, remainder)
                    dst_gm_index = num_outer_axis * num_data_one_loop - \
                                   (remainder - 1) * num_data_one_loop
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_trans[0], 0,
                                           remainder * self.dst_shape[-4],
                                           num_data_one_loop //
                                           self.dst_shape[-4] // self.num_data,
                                           self.num_byte // 2, 0)

        return tik_instance

    def format_transfer_case_two(self, tik_instance):
        """
        the transfer process when UB can not put in
        second last axis * last axis data and last axis is 32B align
        """
        ub_ori_data = self.ub_memory
        ub_trans_data = ub_ori_data
        loop_memory = ub_ori_data - ub_ori_data % \
                      (CUBE_SIZE * CUBE_SIZE * self.dst_shape[-3] + CUBE_SIZE)
        num_data_one_loop = self.dst_shape[-4] * self.dst_shape[-3] * \
                            self.dst_shape[-2] * self.dst_shape[-1]
        num_data_one_loop_padding = num_data_one_loop + self.dst_shape[-4] * \
                                    self.dst_shape[-1]
        loop_times = (num_data_one_loop_padding + loop_memory - 1) // \
                     loop_memory
        if len(self.dst_shape) == 4:
            total_core_loop_num = loop_times
        else:
            total_core_loop_num = functools_reduce(lambda x1, x2: x1 * x2,
                                                   self.dst_shape[:-4]) * \
                                  loop_times
        core_number = _set_core_num(total_core_loop_num)

        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (ub_ori_data,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (ub_trans_data,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            src_ub_index = 0

            with tik_instance.for_range(0, core_loop) as num_core_loop:
                total_core_loop = sum_core + num_core_loop
                num_loop_time = total_core_loop % loop_times
                num_outer_axis = (total_core_loop - num_loop_time) // \
                                 loop_times

                handling_times = tik_instance.Scalar("uint64")
                is_last = tik_instance.Scalar("uint64")
                is_last.set_as(0)
                handling_times.set_as(loop_memory //
                                      (CUBE_SIZE * CUBE_SIZE *
                                       self.dst_shape[-3] + CUBE_SIZE))
                with tik_instance.if_scope(num_loop_time == loop_times - 1):
                    if num_data_one_loop_padding % loop_memory == 0:
                        remainder = loop_memory
                    else:
                        remainder = num_data_one_loop_padding % loop_memory
                    handling_times.set_as((remainder + CUBE_SIZE * CUBE_SIZE *
                                           self.dst_shape[-3] +
                                           CUBE_SIZE - 1) //
                                          (CUBE_SIZE * CUBE_SIZE *
                                           self.dst_shape[-3] + CUBE_SIZE))
                    is_last.set_as(1)
                src_gm_index = num_outer_axis * self.src_shape[-1] * \
                               self.src_shape[-2] + loop_memory // \
                               (CUBE_SIZE * self.dst_shape[-3] + 1) * \
                               num_loop_time
                with tik_instance.for_range(0, self.src_shape[-2] //
                                            MAX_BURST_NUMBER) as num_repeat:
                    tik_instance.data_move(ub_ori[src_ub_index +
                                                  MAX_BURST_NUMBER *
                                                  num_repeat *
                                                  (loop_memory //
                                                   (CUBE_SIZE *
                                                    self.dst_shape[-3] + 1))],
                                           self.src_gm[src_gm_index +
                                                       MAX_BURST_NUMBER *
                                                       num_repeat *
                                                       self.src_shape[-1]],
                                           0, MAX_BURST_NUMBER,
                                           handling_times * self.num_byte // 2,
                                           (self.src_shape[-1] -
                                            handling_times * CUBE_SIZE +
                                            self.num_data - 1) //
                                           self.num_data, 0)
                with tik_instance.if_scope(self.src_shape[-2] %
                                           MAX_BURST_NUMBER != 0):
                    tik_instance.data_move(ub_ori[src_ub_index +
                                                  (self.src_shape[-2] //
                                                   MAX_BURST_NUMBER) *
                                                  MAX_BURST_NUMBER *
                                                  (loop_memory //
                                                   (CUBE_SIZE *
                                                    self.dst_shape[-3] + 1))],
                                           self.src_gm[src_gm_index +
                                                       (self.src_shape[-2] //
                                                        MAX_BURST_NUMBER) *
                                                       MAX_BURST_NUMBER *
                                                       self.src_shape[-1]], 0,
                                           self.src_shape[-2] %
                                           MAX_BURST_NUMBER,
                                           handling_times * self.num_byte // 2,
                                           (self.src_shape[-1] -
                                            handling_times * CUBE_SIZE +
                                            self.num_data - 1) //
                                           self.num_data, 0)
                self.data_rearrange_case_one(tik_instance, ub_ori, ub_trans,
                                             handling_times, is_last)
                dst_gm_index = num_outer_axis * num_data_one_loop + \
                               loop_memory // (CUBE_SIZE * CUBE_SIZE *
                                               self.dst_shape[-3] +
                                               CUBE_SIZE) * \
                               (CUBE_SIZE * CUBE_SIZE *
                                self.dst_shape[-3]) * num_loop_time
                tik_instance.data_move(self.dst_gm[dst_gm_index], ub_trans[0],
                                       0, handling_times,
                                       CUBE_SIZE * self.dst_shape[-3] *
                                       CUBE_SIZE // self.num_data,
                                       self.num_byte // 2, 0)

        return tik_instance

    def format_transfer_case_three(self, tik_instance):
        """
        the transfer process when UB can not put in
        second last axis * last axis data and last axis is not 32B align
        """
        ub_ori_data = self.ub_memory
        ub_trans_data = ub_ori_data

        loop_memory = ub_ori_data - ub_ori_data % \
                      (CUBE_SIZE * CUBE_SIZE * self.dst_shape[-3] + CUBE_SIZE)
        num_data_one_loop = self.dst_shape[-4] * self.dst_shape[-3] * \
                            self.dst_shape[-2] * self.dst_shape[-1]
        num_data_one_loop_padding = num_data_one_loop + self.dst_shape[-4] * \
                                    self.dst_shape[-1]
        loop_times = (num_data_one_loop_padding + loop_memory - 1) // \
                     loop_memory
        if len(self.dst_shape) == 4:
            total_core_loop_num = loop_times
        else:
            total_core_loop_num = functools_reduce(lambda x1, x2: x1 * x2,
                                                   self.dst_shape[:-4]) * \
                                  loop_times
        core_number = _set_core_num(total_core_loop_num)

        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (ub_ori_data,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (ub_trans_data,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:
                total_core_loop = sum_core + num_core_loop
                num_loop_time = total_core_loop % loop_times
                num_outer_axis = (total_core_loop - num_loop_time) // \
                                 loop_times

                handling_times = tik_instance.Scalar("uint64")
                is_last = tik_instance.Scalar("uint64")
                is_last.set_as(0)
                handling_times.set_as(loop_memory // (CUBE_SIZE * CUBE_SIZE *
                                                      self.dst_shape[-3] +
                                                      CUBE_SIZE))

                with tik_instance.if_scope(num_loop_time == loop_times - 1):
                    if num_data_one_loop_padding % loop_memory == 0:
                        remainder = loop_memory
                    else:
                        remainder = num_data_one_loop_padding % loop_memory
                    handling_times.set_as((remainder + CUBE_SIZE * CUBE_SIZE *
                                           self.dst_shape[-3] + CUBE_SIZE -
                                           1) // (CUBE_SIZE * CUBE_SIZE *
                                                  self.dst_shape[-3] +
                                                  CUBE_SIZE))
                    is_last.set_as(1)

                src_ub_index = 0
                src_gm_index = num_outer_axis * self.src_shape[-1] * \
                               self.src_shape[-2] + loop_memory // \
                               (CUBE_SIZE * CUBE_SIZE * self.dst_shape[-3] +
                                CUBE_SIZE) * num_loop_time * CUBE_SIZE
                with tik_instance.for_range(0, self.src_shape[-2]) as num_cube:
                    tik_instance.data_move(ub_ori[src_ub_index +
                                                  num_cube *
                                                  handling_times *
                                                  self.dst_shape[-1]],
                                           self.src_gm[src_gm_index +
                                                       num_cube *
                                                       self.src_shape[-1]],
                                           0, 1,
                                           (handling_times * CUBE_SIZE +
                                            self.num_data - 1) //
                                           self.num_data, 0, 0)

                self.data_rearrange_case_one(tik_instance, ub_ori, ub_trans,
                                             handling_times, is_last)
                dst_gm_index = num_outer_axis * num_data_one_loop + \
                               loop_memory // \
                               (CUBE_SIZE * CUBE_SIZE * self.dst_shape[-3] +
                                CUBE_SIZE) * (CUBE_SIZE * CUBE_SIZE *
                                              self.dst_shape[-3]) * \
                               num_loop_time
                tik_instance.data_move(self.dst_gm[dst_gm_index], ub_trans[0],
                                       0, handling_times,
                                       CUBE_SIZE * self.dst_shape[-3] *
                                       CUBE_SIZE // self.num_data,
                                       self.num_byte // 2, 0)

        return tik_instance

    def format_transfer_case_four(self, tik_instance):
        """
        the transfer process when UB can not put in second last axis * 16 data
        """
        ub_ori_data = self.ub_memory
        loop_row, loop_remainder = _cal_core_loop(tik_instance,
                                                  CUBE_SIZE * CUBE_SIZE,
                                                  self.dst_shape[-3],
                                                  ub_ori_data)
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
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (ub_ori_data,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            with tik_instance.for_range(0, core_loop) as num_core_loop:
                total_core_loop = sum_core + num_core_loop
                num_loop_time = total_core_loop % loop_times
                num_outer_axis = (total_core_loop - num_loop_time) // \
                                 loop_times
                is_last = tik_instance.Scalar("uint64")
                is_last.set_as(0)
                src_ub_index = 0
                count_loop = tik_instance.Scalar("uint64")
                count_loop.set_as(0)
                with tik_instance.for_range(0, self.dst_shape[-3]) as num_cube:
                    with tik_instance.if_scope(
                        tik.all(num_cube % loop_row == 0,
                                num_cube != self.dst_shape[-3] -
                                loop_remainder)):
                        if self.src_shape[-1] % CUBE_SIZE != 0 or \
                                (self.src_shape[-1] - CUBE_SIZE) // \
                                self.num_data > MAX_STRIDE_BLK:
                            with tik_instance.for_range(0, loop_row) \
                                    as num_loop_row:
                                with tik_instance.for_range(0, CUBE_SIZE) \
                                        as num_cube_row:
                                    src_gm_index = num_outer_axis * \
                                                   self.src_shape[-1] * \
                                                   self.src_shape[-2] + \
                                                   num_loop_time * \
                                                   CUBE_SIZE + \
                                                   ((count_loop * loop_row +
                                                     num_loop_row) *
                                                    CUBE_SIZE +
                                                    num_cube_row) * \
                                                   self.src_shape[-1]
                                    tik_instance.data_move(ub_ori
                                                           [src_ub_index +
                                                            (num_loop_row *
                                                             CUBE_SIZE +
                                                             num_cube_row) *
                                                            CUBE_SIZE],
                                                           self.src_gm
                                                           [src_gm_index],
                                                           0, 1,
                                                           self.num_byte // 2,
                                                           0, 0)
                        else:
                            src_gm_index = num_outer_axis * \
                                           self.src_shape[-1] * \
                                           self.src_shape[-2] + \
                                           num_loop_time * CUBE_SIZE + \
                                           (count_loop * loop_row *
                                            CUBE_SIZE) * self.src_shape[-1]
                            tik_instance.data_move(ub_ori[src_ub_index],
                                                   self.src_gm
                                                   [src_gm_index],
                                                   0, loop_row * CUBE_SIZE,
                                                   self.num_byte // 2,
                                                   (self.src_shape[-1] -
                                                    CUBE_SIZE) //
                                                   self.num_data, 0)
                        count_loop.set_as(count_loop + 1)
                        self.data_rearrange_case_two(tik_instance, ub_ori,
                                                     num_loop_time, loop_row,
                                                     is_last)
                        dst_gm_index = num_outer_axis * num_data_one_loop + \
                                       loop_row * (count_loop - 1) * \
                                       CUBE_SIZE * CUBE_SIZE + \
                                       num_loop_time * self.dst_shape[-3] * \
                                       self.dst_shape[-2] * self.dst_shape[-1]
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_ori[0], 0, 1,
                                               loop_row * self.dst_shape[-2] *
                                               self.dst_shape[-1] //
                                               self.num_data, 0, 0)
                    with tik_instance.if_scope(num_cube == self.dst_shape[-3] -
                                               loop_remainder):
                        is_last.set_as(1)
                        if self.src_shape[-1] % CUBE_SIZE != 0 or \
                                (self.src_shape[-1] - CUBE_SIZE) // \
                                self.num_data > MAX_STRIDE_BLK:
                            with tik_instance.for_range(0, loop_remainder) \
                                    as num_loop_row:
                                with tik_instance.for_range(0, CUBE_SIZE) \
                                        as num_cube_row:
                                    with tik_instance.if_scope(
                                        ((count_loop * loop_row +
                                          num_loop_row) * CUBE_SIZE +
                                         num_cube_row) >
                                        self.src_shape[-2]):
                                        pass
                                    with tik_instance.else_scope():
                                        src_gm_index = num_outer_axis * \
                                                       self.src_shape[-1] * \
                                                       self.src_shape[-2] + \
                                                       num_loop_time * \
                                                       CUBE_SIZE + \
                                                       ((count_loop *
                                                         loop_row +
                                                         num_loop_row) *
                                                        CUBE_SIZE +
                                                        num_cube_row) * \
                                                       self.src_shape[-1]
                                        tik_instance.data_move(
                                            ub_ori[src_ub_index +
                                                   (num_loop_row * CUBE_SIZE +
                                                    num_cube_row) *
                                                   CUBE_SIZE],
                                            self.src_gm[src_gm_index],
                                            0, 1, self.num_byte // 2, 0, 0)
                        else:
                            src_gm_index = num_outer_axis * \
                                           self.src_shape[-1] * \
                                           self.src_shape[-2] + \
                                           num_loop_time * CUBE_SIZE + \
                                           (count_loop * loop_row *
                                            CUBE_SIZE) * self.src_shape[-1]
                            if self.src_shape[-2] % CUBE_SIZE == 0:
                                tik_instance.data_move(ub_ori[0],
                                                       self.src_gm
                                                       [src_gm_index], 0,
                                                       loop_remainder *
                                                       CUBE_SIZE,
                                                       self.num_byte // 2,
                                                       (self.src_shape[-1] -
                                                        CUBE_SIZE) //
                                                       self.num_data, 0)
                            else:
                                tik_instance.data_move(ub_ori[0],
                                                       self.src_gm
                                                       [src_gm_index], 0,
                                                       loop_remainder *
                                                       CUBE_SIZE -
                                                       (CUBE_SIZE -
                                                        self.src_shape[-2] %
                                                        CUBE_SIZE),
                                                       self.num_byte // 2,
                                                       (self.src_shape[-1] -
                                                        CUBE_SIZE) //
                                                       self.num_data, 0)
                        self.data_rearrange_case_two(tik_instance, ub_ori,
                                                     num_loop_time,
                                                     loop_remainder, is_last)
                        dst_gm_index = num_outer_axis * num_data_one_loop + \
                                       loop_row * count_loop * CUBE_SIZE * \
                                       CUBE_SIZE + num_loop_time * \
                                       self.dst_shape[-3] * \
                                       self.dst_shape[-2] * \
                                       self.dst_shape[-1]
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_ori[0], 0, 1,
                                               loop_remainder *
                                               self.dst_shape[-2] *
                                               self.dst_shape[-1] //
                                               self.num_data, 0, 0)

        return tik_instance

    def format_transfer_case_five(self, tik_instance):
        """
        the transfer process when UB can put in
        16 * last axis data, last axis is 32B align and
        the shape of dst is 4-D
        """
        ub_ori_data = self.ub_memory
        ub_trans_data = ub_ori_data
        total_core_loop_num = self.dst_shape[-3]

        core_number = _set_core_num(total_core_loop_num)
        num_data_one_loop = self.dst_shape[-4] * \
                            self.dst_shape[-2] * self.dst_shape[-1]

        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (ub_ori_data,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (ub_trans_data,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            align_loop, remainder = _cal_core_loop(tik_instance,
                                                   num_data_one_loop,
                                                   core_loop, ub_ori_data)
            src_ub_index = 0
            with tik_instance.for_range(0, core_loop) as num_core_loop:

                total_core_loop = sum_core + num_core_loop
                num_third_last_axis = total_core_loop
                is_last = tik_instance.Scalar("uint64")
                is_last.set_as(0)
                with tik_instance.if_scope(num_third_last_axis ==
                                           self.dst_shape[-3] - 1):
                    is_last.set_as(1)
                with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                                   align_loop == 0,
                                                   num_core_loop !=
                                                   core_loop - 1)):
                    src_gm_index = num_third_last_axis * num_data_one_loop - \
                                   (align_loop - 1) * num_data_one_loop
                    tik_instance.data_move(ub_ori[src_ub_index],
                                           self.src_gm[src_gm_index],
                                           0, 1,
                                           align_loop * num_data_one_loop //
                                           self.num_data, 0, 0)
                    self.data_rearrange_case_three(tik_instance, ub_ori,
                                                   ub_trans, align_loop,
                                                   is_last)
                    with tik_instance.if_scope(
                        (self.dst_shape[-3] - align_loop) *
                        self.dst_shape[-1] * self.dst_shape[-2] //
                        self.num_data > MAX_STRIDE_BLK):
                        with tik_instance.for_range(0, self.dst_shape[-4]) \
                                as num_col_cube:
                            dst_gm_index = num_third_last_axis * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] - \
                                           (align_loop - 1) * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] + \
                                           num_col_cube * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] * \
                                           self.dst_shape[-3]
                            tik_instance.data_move(self.dst_gm[dst_gm_index],
                                                   ub_trans
                                                   [align_loop *
                                                    self.dst_shape[-1] *
                                                    self.dst_shape[-2] *
                                                    num_col_cube +
                                                    CUBE_SIZE * num_col_cube],
                                                   0, 1,
                                                   align_loop *
                                                   self.dst_shape[-1] *
                                                   self.dst_shape[-2] //
                                                   self.num_data, 0, 0)
                    with tik_instance.else_scope():
                        dst_gm_index = num_third_last_axis * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2] - \
                                       (align_loop - 1) * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2]
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans[0], 0,
                                               self.dst_shape[-4],
                                               (align_loop *
                                                self.dst_shape[-1] *
                                                self.dst_shape[-2] +
                                                self.num_data - 1) //
                                               self.num_data,
                                               self.num_byte // 2,
                                               (self.dst_shape[-3] -
                                                align_loop) *
                                               self.dst_shape[-1] *
                                               self.dst_shape[-2] //
                                               self.num_data)
                with tik_instance.if_scope(num_core_loop == core_loop - 1):

                    src_gm_index = num_third_last_axis * num_data_one_loop - \
                                   (remainder - 1) * num_data_one_loop
                    with tik_instance.if_scope(is_last == 1):
                        if self.src_shape[-2] % CUBE_SIZE != 0:
                            tik_instance.data_move(ub_ori[src_ub_index],
                                                   self.src_gm[src_gm_index],
                                                   0, 1,
                                                   (remainder *
                                                    num_data_one_loop -
                                                    (CUBE_SIZE -
                                                     self.src_shape[-2] %
                                                     CUBE_SIZE) *
                                                    self.dst_shape[-4] *
                                                    self.dst_shape[-1]) //
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
                    self.data_rearrange_case_three(tik_instance, ub_ori,
                                                   ub_trans, remainder,
                                                   is_last)
                    with tik_instance.if_scope(
                        (self.dst_shape[-3] - remainder) *
                        self.dst_shape[-1] * self.dst_shape[-2] //
                        self.num_data > MAX_STRIDE_BLK):
                        with tik_instance.for_range(0, self.dst_shape[-4]) \
                                as num_col_cube:
                            dst_gm_index = num_third_last_axis * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] - \
                                           (remainder - 1) * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] + \
                                           num_col_cube * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] * \
                                           self.dst_shape[-3]
                            tik_instance.data_move(self.dst_gm[dst_gm_index],
                                                   ub_trans
                                                   [remainder *
                                                    self.dst_shape[-1] *
                                                    self.dst_shape[-2] *
                                                    num_col_cube +
                                                    CUBE_SIZE * num_col_cube],
                                                   0, 1,
                                                   remainder *
                                                   self.dst_shape[-1] *
                                                   self.dst_shape[-2] //
                                                   self.num_data, 0, 0)
                    with tik_instance.else_scope():
                        dst_gm_index = num_third_last_axis * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2] - \
                                       (remainder - 1) * self.dst_shape[-1] * \
                                       self.dst_shape[-2]

                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans[0], 0,
                                               self.dst_shape[-4],
                                               (remainder *
                                                self.dst_shape[-1] *
                                                self.dst_shape[-2] +
                                                self.num_data - 1) //
                                               self.num_data,
                                               self.num_byte // 2,
                                               (self.dst_shape[-3] -
                                                remainder) *
                                               self.dst_shape[-1] *
                                               self.dst_shape[-2] //
                                               self.num_data)

        return tik_instance

    def format_transfer_case_six(self, tik_instance):
        """
        the transfer process when UB can put in
        16 * last axis data, last axis is not 32B align and
        the shape of dst is 4-D
        """
        ub_ori_data = self.ub_memory
        ub_trans_data = ub_ori_data
        total_core_loop_num = self.dst_shape[-3]

        core_number = _set_core_num(total_core_loop_num)
        num_data_one_loop = self.dst_shape[-4] * \
                            self.dst_shape[-2] * self.dst_shape[-1]

        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (ub_ori_data,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (ub_trans_data,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            align_loop, remainder = _cal_core_loop(tik_instance,
                                                   num_data_one_loop,
                                                   core_loop, ub_ori_data)
            with tik_instance.for_range(0, core_loop) as num_core_loop:

                total_core_loop = sum_core + num_core_loop
                num_third_last_axis = total_core_loop
                src_gm_index = num_third_last_axis * self.src_shape[-1] * \
                               CUBE_SIZE
                src_ub_index = (num_core_loop % align_loop) * \
                               num_data_one_loop
                with tik_instance.if_scope(num_core_loop == core_loop - 1):
                    with tik_instance.if_scope(num_third_last_axis ==
                                               self.dst_shape[-3] - 1):
                        if self.src_shape[-2] % CUBE_SIZE != 0:
                            with tik_instance.for_range(0, self.src_shape[-2] %
                                                        CUBE_SIZE) as \
                                    num_cube_row:
                                tik_instance.data_move(ub_ori
                                                       [src_ub_index +
                                                        num_cube_row *
                                                        self.dst_shape[-1] *
                                                        self.dst_shape[-4]],
                                                       self.src_gm
                                                       [src_gm_index +
                                                        num_cube_row *
                                                        self.src_shape[-1]],
                                                       0, 1,
                                                       self.dst_shape[-1] *
                                                       self.dst_shape[-4] //
                                                       self.num_data, 0, 0)
                        else:
                            with tik_instance.for_range(0, CUBE_SIZE) as \
                                    num_cube_row:
                                tik_instance.data_move(ub_ori
                                                       [src_ub_index +
                                                        num_cube_row *
                                                        self.dst_shape[-1] *
                                                        self.dst_shape[-4]],
                                                       self.src_gm
                                                       [src_gm_index +
                                                        num_cube_row *
                                                        self.src_shape[-1]],
                                                       0, 1,
                                                       self.dst_shape[-1] *
                                                       self.dst_shape[-4] //
                                                       self.num_data, 0, 0)
                    with tik_instance.else_scope():
                        with tik_instance.for_range(0, CUBE_SIZE) as \
                                num_cube_row:
                            tik_instance.data_move(ub_ori[src_ub_index +
                                                          num_cube_row *
                                                          self.dst_shape[-1] *
                                                          self.dst_shape[-4]],
                                                   self.src_gm
                                                   [src_gm_index +
                                                    num_cube_row *
                                                    self.src_shape[-1]],
                                                   0, 1,
                                                   self.dst_shape[-1] *
                                                   self.dst_shape[-4] //
                                                   self.num_data, 0, 0)
                with tik_instance.else_scope():
                    with tik_instance.for_range(0, CUBE_SIZE) as num_cube_row:
                        tik_instance.data_move(ub_ori[src_ub_index +
                                                      num_cube_row *
                                                      self.dst_shape[-1] *
                                                      self.dst_shape[-4]],
                                               self.src_gm[src_gm_index +
                                                           num_cube_row *
                                                           self.src_shape[-1]],
                                               0, 1,
                                               self.dst_shape[-1] *
                                               self.dst_shape[-4] //
                                               self.num_data, 0, 0)
                is_last = tik_instance.Scalar("uint64")
                is_last.set_as(0)
                with tik_instance.if_scope(num_third_last_axis ==
                                           self.dst_shape[-3] - 1):
                    is_last.set_as(1)
                with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                                   align_loop == 0,
                                                   num_core_loop !=
                                                   core_loop - 1)):
                    self.data_rearrange_case_three(tik_instance, ub_ori,
                                                   ub_trans, align_loop,
                                                   is_last)
                    with tik_instance.if_scope(
                        (self.dst_shape[-3] - align_loop) *
                        self.dst_shape[-1] * self.dst_shape[-2] //
                        self.num_data > MAX_STRIDE_BLK):
                        with tik_instance.for_range(0, self.dst_shape[-4]) \
                                as num_col_cube:

                            dst_gm_index = num_third_last_axis * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] - \
                                           (align_loop - 1) * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] + \
                                           num_col_cube * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] * \
                                           self.dst_shape[-3]
                            tik_instance.data_move(self.dst_gm[dst_gm_index],
                                                   ub_trans
                                                   [align_loop *
                                                    self.dst_shape[-1] *
                                                    self.dst_shape[-2] *
                                                    num_col_cube +
                                                    CUBE_SIZE * num_col_cube],
                                                   0, 1,
                                                   align_loop *
                                                   self.dst_shape[-1] *
                                                   self.dst_shape[-2] //
                                                   self.num_data, 0, 0)
                    with tik_instance.else_scope():
                        dst_gm_index = num_third_last_axis * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2] - \
                                       (align_loop - 1) * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2]
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans[0], 0,
                                               self.dst_shape[-4],
                                               (align_loop *
                                                self.dst_shape[-1] *
                                                self.dst_shape[-2] +
                                                self.num_data - 1) //
                                               self.num_data,
                                               self.num_byte // 2,
                                               (self.dst_shape[-3] -
                                                align_loop) *
                                               self.dst_shape[-1] *
                                               self.dst_shape[-2] //
                                               self.num_data)
                with tik_instance.if_scope(num_core_loop == core_loop - 1):

                    self.data_rearrange_case_three(tik_instance, ub_ori,
                                                   ub_trans, remainder,
                                                   is_last)
                    with tik_instance.if_scope(
                        (self.dst_shape[-3] - remainder) *
                        self.dst_shape[-1] * self.dst_shape[-2] //
                        self.num_data > MAX_STRIDE_BLK):
                        with tik_instance.for_range(0, self.dst_shape[-4]) \
                                as num_col_cube:

                            dst_gm_index = num_third_last_axis * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] - \
                                           (remainder - 1) * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] + \
                                           num_col_cube * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] * \
                                           self.dst_shape[-3]
                            tik_instance.data_move(self.dst_gm[dst_gm_index],
                                                   ub_trans
                                                   [remainder *
                                                    self.dst_shape[-1] *
                                                    self.dst_shape[-2] *
                                                    num_col_cube +
                                                    CUBE_SIZE * num_col_cube],
                                                   0, 1,
                                                   remainder *
                                                   self.dst_shape[-1] *
                                                   self.dst_shape[-2] //
                                                   self.num_data, 0, 0)
                    with tik_instance.else_scope():
                        dst_gm_index = num_third_last_axis * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2] - \
                                       (remainder - 1) * self.dst_shape[-1] * \
                                       self.dst_shape[-2]

                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans[0], 0,
                                               self.dst_shape[-4],
                                               (remainder *
                                                self.dst_shape[-1] *
                                                self.dst_shape[-2] +
                                                self.num_data - 1) //
                                               self.num_data,
                                               self.num_byte // 2,
                                               (self.dst_shape[-3] -
                                                remainder) *
                                               self.dst_shape[-1] *
                                               self.dst_shape[-2] //
                                               self.num_data)

        return tik_instance

    def format_transfer_case_seven(self, tik_instance):
        """
        the transfer process when UB can not put in second last axis * 16 data
        and the shape of dst is 4-D
        """
        def data_move_case_zero(tik_instance, ub_ori, ub_trans,
                                is_last, num_outer_axis, num_loop_time,
                                loop_time, loop_col, loop_len):
            """
            the process of date move
            """
            with tik_instance.if_scope(tik.all(loop_time ==
                                               self.dst_shape[-4] //
                                               loop_col - 1,
                                               self.dst_shape[-4] % loop_col ==
                                               0)):
                is_last.set_as(1)
            num_data_one_loop = self.dst_shape[-4] * self.dst_shape[-3] * \
                                self.dst_shape[-2] * self.dst_shape[-1]
            src_ub_index = 0
            if self.src_shape[-1] % CUBE_SIZE != 0 or \
                    (self.src_shape[-1] - loop_len * CUBE_SIZE) // \
                    self.num_data > MAX_STRIDE_BLK:
                if self.src_shape[-2] % CUBE_SIZE != 0:
                    with tik_instance.if_scope(num_loop_time ==
                                               self.dst_shape[-3] - 1):
                        with tik_instance.for_range(0,
                                                    self.src_shape[-2] %
                                                    CUBE_SIZE) as num_cube_col:
                            src_gm_index = num_outer_axis *\
                                           self.src_shape[-1] * \
                                           self.src_shape[-2] + \
                                           (num_loop_time * CUBE_SIZE +
                                            num_cube_col) *\
                                           self.src_shape[-1] + loop_time *\
                                           loop_col * CUBE_SIZE
                            tik_instance.data_move(ub_ori[loop_len *
                                                          CUBE_SIZE *
                                                          num_cube_col],
                                                   self.src_gm[src_gm_index],
                                                   0, 1,
                                                   loop_len * self.num_byte //
                                                   2, 0, 0)
                    with tik_instance.else_scope():
                        with tik_instance.for_range(0, CUBE_SIZE) \
                                as num_cube_col:
                            src_gm_index = num_outer_axis * \
                                           self.src_shape[-1] * \
                                           self.src_shape[-2] + \
                                           (num_loop_time * CUBE_SIZE +
                                            num_cube_col) *\
                                           self.src_shape[-1] + loop_time *\
                                           loop_col * CUBE_SIZE
                            tik_instance.data_move(ub_ori[loop_len *
                                                          CUBE_SIZE *
                                                          num_cube_col],
                                                   self.src_gm[src_gm_index],
                                                   0, 1,
                                                   loop_len * self.num_byte //
                                                   2, 0, 0)
                else:
                    with tik_instance.for_range(0, CUBE_SIZE) as num_cube_col:
                        src_gm_index = num_outer_axis * self.src_shape[-1] * \
                                       self.src_shape[-2] + \
                                       (num_loop_time * CUBE_SIZE +
                                        num_cube_col) * self.src_shape[-1] + \
                                       loop_time * loop_col * CUBE_SIZE
                        tik_instance.data_move(ub_ori[loop_len * CUBE_SIZE *
                                                      num_cube_col],
                                               self.src_gm[src_gm_index],
                                               0, 1,
                                               loop_len * self.num_byte //
                                               2, 0, 0)
            else:
                src_gm_index = num_outer_axis * self.src_shape[-1] * \
                               self.src_shape[-2] + num_loop_time *\
                               CUBE_SIZE * self.src_shape[-1] +\
                               loop_time * loop_col * CUBE_SIZE
                if self.src_shape[-2] % CUBE_SIZE != 0:
                    with tik_instance.if_scope(num_loop_time ==
                                               self.dst_shape[-3] - 1):
                        tik_instance.data_move(ub_ori[src_ub_index],
                                               self.src_gm[src_gm_index], 0,
                                               self.src_shape[-2] % CUBE_SIZE,
                                               loop_len * self.num_byte // 2,
                                               (self.src_shape[-1] -
                                                loop_len * CUBE_SIZE) //
                                               self.num_data,
                                               0)
                    with tik_instance.else_scope():
                        tik_instance.data_move(ub_ori[src_ub_index],
                                               self.src_gm[src_gm_index],
                                               0, CUBE_SIZE,
                                               loop_len * self.num_byte // 2,
                                               (self.src_shape[-1] -
                                                loop_len * CUBE_SIZE) //
                                               self.num_data,
                                               0)
                else:
                    tik_instance.data_move(ub_ori[src_ub_index],
                                           self.src_gm[src_gm_index],
                                           0, CUBE_SIZE,
                                           loop_len * self.num_byte // 2,
                                           (self.src_shape[-1] - loop_len *
                                            CUBE_SIZE) // self.num_data, 0)
            self.data_rearrange_case_four(tik_instance, ub_ori,
                                          ub_trans, num_loop_time,
                                          loop_len, is_last)

            if((self.dst_shape[-3] - 1) * self.dst_shape[-1] *
               self.dst_shape[-2] // self.num_data > MAX_STRIDE_BLK):
                with tik_instance.for_range(0, loop_len) as \
                        num_col_cube:
                    dst_gm_index = num_outer_axis * num_data_one_loop + \
                                   num_loop_time * self.dst_shape[-1] * \
                                   self.dst_shape[-2] + \
                                   (loop_time * loop_col + num_col_cube) * \
                                   self.dst_shape[-1] * self.dst_shape[-2] * \
                                   self.dst_shape[-3]
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_trans[num_col_cube *
                                                    CUBE_SIZE *
                                                    (CUBE_SIZE + 1)],
                                           0, 1,
                                           self.dst_shape[-1] *
                                           self.dst_shape[-2] //
                                           self.num_data,
                                           0, 0)
            else:
                dst_gm_index = num_outer_axis * num_data_one_loop + \
                               num_loop_time * self.dst_shape[-1] * \
                               self.dst_shape[-2] + loop_time * \
                               loop_col * self.dst_shape[-1] * \
                               self.dst_shape[-2] * \
                               self.dst_shape[-3]
                tik_instance.data_move(self.dst_gm[dst_gm_index],
                                       ub_trans[0],
                                       0, loop_len,
                                       self.dst_shape[-1] *
                                       self.dst_shape[-2] // self.num_data,
                                       self.num_byte // 2,
                                       (self.dst_shape[-3] - 1) *
                                       self.dst_shape[-1] *
                                       self.dst_shape[-2] // self.num_data)
        if self.src_shape[-2] // (MAX_CORE_NUM * CUBE_SIZE) < 2:
            ub_ori_data = self.ub_memory
            ub_trans_data = ub_ori_data
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (ub_ori_data,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (ub_trans_data,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            loop_col, loop_remainder = _cal_core_loop_python_one(
                CUBE_SIZE * (CUBE_SIZE + 1), self.dst_shape[-4], ub_ori_data)
            loop_times = self.dst_shape[-3]
            if len(self.dst_shape) == 4:
                total_core_loop_num = loop_times
            else:
                total_core_loop_num = functools_reduce(lambda x1, x2: x1 * x2,
                                                       self.dst_shape[:-4]) * \
                                      loop_times
            core_number = _set_core_num(total_core_loop_num)

            with tik_instance.for_range(0, core_number, block_num=core_number) \
                    as num_core:

                core_loop, sum_core = _cal_core(tik_instance,
                                                total_core_loop_num,
                                                num_core, core_number)
                with tik_instance.for_range(0, core_loop) as num_core_loop:
                    total_core_loop = sum_core + num_core_loop
                    num_loop_time = total_core_loop % loop_times
                    num_outer_axis = (total_core_loop - num_loop_time) // \
                                     loop_times
                    is_last = tik_instance.Scalar("uint64")
                    is_last.set_as(0)
                    with tik_instance.for_range(0, self.dst_shape[-4] //
                                                loop_col) as num_cube:
                        data_move_case_zero(tik_instance, ub_ori,
                                            ub_trans, is_last,
                                            num_outer_axis,
                                            num_loop_time, num_cube,
                                            loop_col, loop_col)

                    if loop_remainder != 0:
                        is_last.set_as(1)
                        data_move_case_zero(tik_instance, ub_ori,
                                            ub_trans, is_last, num_outer_axis,
                                            num_loop_time,
                                            self.dst_shape[-4] // loop_col,
                                            loop_col, loop_remainder)
        else:
            ub_ori_data = self.ub_memory // 2
            ub_trans_data = ub_ori_data
            loop_col, loop_remainder = _cal_core_loop_python_one(
                CUBE_SIZE * (CUBE_SIZE + 1), self.dst_shape[-4], ub_ori_data)
            loop_times = self.dst_shape[-3]
            if len(self.dst_shape) == 4:
                total_core_loop_num = loop_times
            else:
                total_core_loop_num = functools_reduce(lambda x1, x2: x1 * x2,
                                                       self.dst_shape[:-4]) * \
                                      loop_times
            core_number = _set_core_num(total_core_loop_num)

            with tik_instance.for_range(0, core_number, block_num=core_number) \
                    as num_core:

                core_loop, sum_core = _cal_core(tik_instance,
                                                total_core_loop_num,
                                                num_core, core_number)
                with tik_instance.for_range(0, core_loop, thread_num=2)\
                        as num_core_loop:
                    ub_ori = tik_instance.Tensor(self.dtype,
                                                 (ub_ori_data,),
                                                 name="ub_ori",
                                                 scope=tik.scope_ubuf)
                    ub_trans = tik_instance.Tensor(self.dtype,
                                                   (ub_trans_data,),
                                                   name="ub_trans",
                                                   scope=tik.scope_ubuf)
                    total_core_loop = sum_core + num_core_loop
                    num_loop_time = total_core_loop % loop_times
                    num_outer_axis = (total_core_loop - num_loop_time) // \
                                     loop_times
                    is_last = tik_instance.Scalar("uint64")
                    is_last.set_as(0)
                    with tik_instance.for_range(0, self.dst_shape[-4] //
                                                loop_col) as num_cube:
                        data_move_case_zero(tik_instance, ub_ori,
                                            ub_trans, is_last, num_outer_axis,
                                            num_loop_time, num_cube,
                                            loop_col, loop_col)

                    if loop_remainder != 0:
                        is_last.set_as(1)
                        data_move_case_zero(tik_instance, ub_ori,
                                            ub_trans, is_last, num_outer_axis,
                                            num_loop_time,
                                            self.dst_shape[-4] // loop_col,
                                            loop_col, loop_remainder)

        return tik_instance

    def format_transfer_case_eight(self, tik_instance):
        """
        the transfer process when UB // 2 can put in
        16 * last axis data, last axis is 32B align and
        the shape of dst is 4-D
        """
        ub_ori_data = self.ub_memory // 2

        ub_trans_data = ub_ori_data
        total_core_loop_num = self.dst_shape[-3]

        core_number = _set_core_num(total_core_loop_num)
        num_data_one_loop = self.dst_shape[-4] * \
                            self.dst_shape[-2] * self.dst_shape[-1]

        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            core_loop = total_core_loop_num // core_number
            sum_core = num_core*core_loop
            align_loop, remainder = _cal_core_loop_python(num_data_one_loop,
                                                          core_loop,
                                                          ub_ori_data -
                                                          self.dst_shape[-4] *
                                                          self.dst_shape[-1])
            core_loop_num = core_loop // align_loop
            thread_number = 1 if core_loop_num == 1 else 2
            src_ub_index = 0
            with tik_instance.for_range(0, core_loop_num,
                                        thread_num=thread_number)\
                    as num_core_loop:
                ub_ori = tik_instance.Tensor(self.dtype,
                                             (ub_ori_data,),
                                             name="ub_ori",
                                             scope=tik.scope_ubuf)
                ub_trans = tik_instance.Tensor(self.dtype,
                                               (ub_trans_data,),
                                               name="ub_trans",
                                               scope=tik.scope_ubuf)
                total_core_loop = sum_core + num_core_loop * align_loop
                num_third_last_axis = total_core_loop
                src_gm_index = num_third_last_axis * num_data_one_loop
                tik_instance.data_move(ub_ori[src_ub_index],
                                       self.src_gm[src_gm_index],
                                       0, 1,
                                       align_loop * num_data_one_loop //
                                       self.num_data, 0, 0)
                self.data_rearrange_case_five(tik_instance, ub_ori,
                                              ub_trans, align_loop)
                if (self.dst_shape[-3] - align_loop) * self.dst_shape[-1] *\
                        self.dst_shape[-2] // self.num_data > MAX_STRIDE_BLK:
                    with tik_instance.for_range(0, self.dst_shape[-4]) \
                            as num_col_cube:
                        dst_gm_index = num_third_last_axis * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2] + \
                                       num_col_cube * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2] * \
                                       self.dst_shape[-3]
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans
                                               [align_loop *
                                                self.dst_shape[-1] *
                                                self.dst_shape[-2] *
                                                num_col_cube +
                                                CUBE_SIZE * num_col_cube],
                                               0, 1,
                                               align_loop *
                                               self.dst_shape[-1] *
                                               self.dst_shape[-2] //
                                               self.num_data, 0, 0)
                else:
                    dst_gm_index = num_third_last_axis * \
                                   self.dst_shape[-1] * \
                                   self.dst_shape[-2]
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_trans[0], 0,
                                           self.dst_shape[-4],
                                           (align_loop *
                                            self.dst_shape[-1] *
                                            self.dst_shape[-2] +
                                            self.num_data - 1) //
                                           self.num_data,
                                           self.num_byte // 2,
                                           (self.dst_shape[-3] -
                                            align_loop) *
                                           self.dst_shape[-1] *
                                           self.dst_shape[-2] //
                                           self.num_data)
            if remainder != 0:
                ub_ori = tik_instance.Tensor(self.dtype,
                                             (ub_ori_data,),
                                             name="ub_ori",
                                             scope=tik.scope_ubuf)
                ub_trans = tik_instance.Tensor(self.dtype,
                                               (ub_trans_data,),
                                               name="ub_trans",
                                               scope=tik.scope_ubuf)
                num_third_last_axis = sum_core + core_loop_num * align_loop
                src_gm_index = num_third_last_axis * num_data_one_loop
                tik_instance.data_move(ub_ori[src_ub_index],
                                       self.src_gm[src_gm_index],
                                       0, 1,
                                       remainder *
                                       num_data_one_loop //
                                       self.num_data, 0, 0)
                self.data_rearrange_case_five(tik_instance, ub_ori,
                                              ub_trans, remainder)
                if (self.dst_shape[-3] - remainder) * self.dst_shape[-1] * \
                        self.dst_shape[-2] // self.num_data > MAX_STRIDE_BLK:
                    with tik_instance.for_range(0, self.dst_shape[-4]) \
                            as num_col_cube:
                        dst_gm_index = num_third_last_axis * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2] + \
                                       num_col_cube * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2] * \
                                       self.dst_shape[-3]
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans
                                               [remainder *
                                                self.dst_shape[-1] *
                                                self.dst_shape[-2] *
                                                num_col_cube +
                                                CUBE_SIZE * num_col_cube],
                                               0, 1,
                                               remainder *
                                               self.dst_shape[-1] *
                                               self.dst_shape[-2] //
                                               self.num_data, 0, 0)
                else:
                    dst_gm_index = num_third_last_axis * \
                                   self.dst_shape[-1] * \
                                   self.dst_shape[-2]

                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_trans[0], 0,
                                           self.dst_shape[-4],
                                           (remainder *
                                            self.dst_shape[-1] *
                                            self.dst_shape[-2] +
                                            self.num_data - 1) //
                                           self.num_data,
                                           self.num_byte // 2,
                                           (self.dst_shape[-3] -
                                            remainder) *
                                           self.dst_shape[-1] *
                                           self.dst_shape[-2] //
                                           self.num_data)

        return tik_instance

    def nd_2_nz_compute(self):
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
        elif format_transfer_case == 4:
            tik_instance = self.format_transfer_case_four(tik_instance)
        elif format_transfer_case == 5:
            tik_instance = self.format_transfer_case_five(tik_instance)
        elif format_transfer_case == 6:
            tik_instance = self.format_transfer_case_six(tik_instance)
        elif format_transfer_case == 7:
            tik_instance = self.format_transfer_case_seven(tik_instance)
        elif format_transfer_case == 8:
            tik_instance = self.format_transfer_case_eight(tik_instance)
        return tik_instance

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        tik_instance = self.nd_2_nz_compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.src_gm],
                              outputs=[self.dst_gm])

        return tik_instance


class ND2NzComputeInt8:
    """
    Rearranges data from ND format into FRACTAL_NZ format

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
    vector_dup_zero:
        vector_dup zeros when dup_number is python variable
    data_rearrange_case_zero:
        rearrange data when UB can put in last axis * 16 data and
        the shape of dst is 4-D
    data_rearrange_case_one:
        rearrange data when UB can not put in last axis * 16 data
    def format_transfer_case_zero:
        the transfer process when the transfer case is 0
    def format_transfer_case_one:
        the transfer process when the transfer case is 1
    def format_transfer_case_two:
        the transfer process when the transfer case is 2
    def data_move_case_zero:
        the data move process of the transfer case is 2
    nd_2_nz_compute:
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
        self.dst_shape[-2:] = [(self.src_shape[-1] + CUBE_SIZE_2 - 1) //
                               CUBE_SIZE_2,
                               (self.src_shape[-2] + CUBE_SIZE - 1) //
                               CUBE_SIZE,
                               CUBE_SIZE,
                               CUBE_SIZE_2]
        self.num_byte = SIZE_ONE_BYTES
        self.cast_num_byte = SIZE_TWO_BYTES
        self.vadds_mask = MAX_MASK

        # the number of data that can be moved in each data_move
        self.num_data = DATA_MOVE_MIN_UNIT // self.num_byte
        # the number of float16 data that can be moved in each data_move
        self.cast_num_data = DATA_MOVE_MIN_UNIT // self.cast_num_byte
        util.check_shape_rule(self.dst_shape)
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
                if self.dst_shape[-4] * self.dst_shape[-1] * \
                        (self.dst_shape[-2] + 1) > self.ub_memory:
                    format_transfer_case = 2
            else:
                format_transfer_case = 1
                if self.dst_shape[-4] * self.dst_shape[-1] * \
                        (self.dst_shape[-2] + 1) > self.ub_memory:
                    format_transfer_case = 2
        else:
            raise RuntimeError("ND2Nz only support 2D now when dtype is int8")

        return format_transfer_case

    def vector_dup_zero(self, tik_instance, ub_trans, dup_number, offset):
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
        rearrange data when UB can put in last axis * 16 data and
        the shape of dst is 4-D
        """
        cast_repeat_time = tik_instance.Scalar("uint64")
        cast_remainder = tik_instance.Scalar("uint64")
        with tik_instance.if_scope(is_last == 1):
            if (self.src_shape[-2] % CUBE_SIZE) == 0:
                cast_repeat_time.set_as(loop_num * CUBE_SIZE *
                                        self.dst_shape[-4] *
                                        self.dst_shape[-1] // MAX_MASK)
                cast_remainder.set_as(loop_num * CUBE_SIZE *
                                      self.dst_shape[-4] *
                                      self.dst_shape[-1] % MAX_MASK)
            else:
                cast_repeat_time.set_as((loop_num * CUBE_SIZE - CUBE_SIZE +
                                         self.src_shape[-2] % CUBE_SIZE) *
                                        self.dst_shape[-4] *
                                        self.dst_shape[-1] // MAX_MASK)
                cast_remainder.set_as((loop_num * CUBE_SIZE - CUBE_SIZE +
                                       self.src_shape[-2] % CUBE_SIZE) *
                                      self.dst_shape[-4] * self.dst_shape[-1] %
                                      MAX_MASK)
        with tik_instance.else_scope():
            cast_repeat_time.set_as(loop_num * CUBE_SIZE * self.dst_shape[-4] *
                                    self.dst_shape[-1] // MAX_MASK)
            cast_remainder.set_as(loop_num * CUBE_SIZE * self.dst_shape[-4] *
                                  self.dst_shape[-1] % MAX_MASK)
        # cast the data from int8 to float16
        _cast_dtype(tik_instance, ub_cast_fp16, ub_ori, cast_repeat_time,
                    cast_remainder, "int8_2_float16")

        num_row_one_loop = loop_num * CUBE_SIZE
        scalar_zero = tik_instance.Scalar(dtype="float16", init_value=0.0)

        if self.src_shape[-1] % CUBE_SIZE_2 != 0:
            mask = 0
            for i, _ in enumerate(range(CUBE_SIZE_2 -
                                        self.src_shape[-1] % CUBE_SIZE_2)):
                mask += 2 ** (CUBE_SIZE_2 - 1 - i)

            with tik_instance.for_range(0, num_row_one_loop // MAX_REPEATS) \
                    as num_repeat:
                tik_instance.vector_dup([0, mask],
                                        ub_cast_fp16[(MAX_REPEATS *
                                                      num_repeat + 1) *
                                                     self.dst_shape[-4] *
                                                     self.dst_shape[-1] -
                                                     CUBE_SIZE_2],
                                        scalar_zero, MAX_REPEATS,
                                        self.cast_num_byte // 2,
                                        self.dst_shape[-4] *
                                        self.dst_shape[-1] //
                                        self.cast_num_data)
            with tik_instance.if_scope(num_row_one_loop % MAX_REPEATS != 0):
                tik_instance.vector_dup([0, mask],
                                        ub_cast_fp16[((num_row_one_loop //
                                                       MAX_REPEATS) *
                                                      MAX_REPEATS + 1) *
                                                     self.dst_shape[-4] *
                                                     self.dst_shape[-1] -
                                                     CUBE_SIZE_2],
                                        scalar_zero,
                                        num_row_one_loop % MAX_REPEATS,
                                        0, self.dst_shape[-4] *
                                        self.dst_shape[-1] //
                                        self.cast_num_data)

        with tik_instance.if_scope(is_last == 1):
            if (self.src_shape[-2] % CUBE_SIZE) != 0:
                dup_number = (CUBE_SIZE - self.src_shape[-2] % CUBE_SIZE) * \
                             self.dst_shape[-1] * self.dst_shape[-4]
                offset = ((loop_num - 1) * self.dst_shape[-2] +
                          self.src_shape[-2] % CUBE_SIZE) * \
                         self.dst_shape[-1] * self.dst_shape[-4]
                self.vector_dup_zero(tik_instance, ub_cast_fp16,
                                     dup_number, offset)

        with tik_instance.for_range(0, self.dst_shape[-4]) as num_col_cube:
            with tik_instance.for_range(0, CUBE_SIZE * loop_num //
                                        MAX_REPEATS) as num_repeat_one:
                tik_instance.vadds(CUBE_SIZE_2,
                                   ub_trans[num_col_cube * loop_num *
                                            self.dst_shape[-2] *
                                            self.dst_shape[-1] +
                                            MAX_REPEATS *
                                            num_repeat_one * CUBE_SIZE_2 +
                                            CUBE_SIZE_2 * num_col_cube],
                                   ub_cast_fp16[MAX_REPEATS * num_repeat_one *
                                                self.dst_shape[-1] *
                                                self.dst_shape[-4] +
                                                num_col_cube * CUBE_SIZE_2],
                                   scalar_zero, MAX_REPEATS,
                                   self.cast_num_byte // 2,
                                   self.cast_num_byte // 2,
                                   self.cast_num_byte,
                                   self.dst_shape[-4] * self.dst_shape[-1] //
                                   self.cast_num_data)

            with tik_instance.if_scope((CUBE_SIZE * loop_num) %
                                       MAX_REPEATS != 0):
                tik_instance.vadds(CUBE_SIZE_2,
                                   ub_trans[num_col_cube * loop_num *
                                            self.dst_shape[-2] *
                                            self.dst_shape[-1] +
                                            (CUBE_SIZE * loop_num) //
                                            MAX_REPEATS * MAX_REPEATS *
                                            CUBE_SIZE_2 + CUBE_SIZE_2 *
                                            num_col_cube],
                                   ub_cast_fp16[(CUBE_SIZE * loop_num) //
                                                MAX_REPEATS * MAX_REPEATS *
                                                self.dst_shape[-1] *
                                                self.dst_shape[-4] +
                                                num_col_cube * CUBE_SIZE_2],
                                   scalar_zero,
                                   (CUBE_SIZE * loop_num) % MAX_REPEATS,
                                   self.cast_num_byte // 2,
                                   self.cast_num_byte // 2,
                                   self.cast_num_byte,
                                   self.dst_shape[-4] * self.dst_shape[-1] //
                                   self.cast_num_data)

        cast_repeat_time.set_as((loop_num * CUBE_SIZE + 1) *
                                self.dst_shape[-4] * self.dst_shape[-1] //
                                MAX_MASK)
        cast_remainder.set_as((loop_num * CUBE_SIZE + 1) * self.dst_shape[-4] *
                              self.dst_shape[-1] % MAX_MASK)
        # cast the data from float16 to int8
        _cast_dtype(tik_instance, ub_cast_int8, ub_trans, cast_repeat_time,
                    cast_remainder, "float16_2_int8")

    def data_rearrange_case_one(self, tik_instance, ub_ori, ub_cast_fp16,
                                ub_trans, ub_cast_int8,
                                num_loop_time, loop_num, is_last):
        """
        rearrange data when UB can not put in last axis * 16 data
        """
        cast_repeat_time = tik_instance.Scalar("uint64")
        cast_remainder = tik_instance.Scalar("uint64")
        with tik_instance.if_scope(num_loop_time == self.dst_shape[-3] - 1):
            if (self.src_shape[-2] % CUBE_SIZE) == 0:
                cast_repeat_time.set_as(loop_num * self.dst_shape[-1] *
                                        self.dst_shape[-2] // MAX_MASK)
                cast_remainder.set_as(loop_num * self.dst_shape[-1] *
                                      self.dst_shape[-2] % MAX_MASK)
            else:
                cast_repeat_time.set_as((self.src_shape[-2] % CUBE_SIZE) *
                                        loop_num * self.dst_shape[-1] //
                                        MAX_MASK)
                cast_remainder.set_as((self.src_shape[-2] % CUBE_SIZE) *
                                      loop_num * self.dst_shape[-1] %
                                      MAX_MASK)
        with tik_instance.else_scope():
            cast_repeat_time.set_as(loop_num * self.dst_shape[-1] *
                                    self.dst_shape[-2] // MAX_MASK)
            cast_remainder.set_as(loop_num * self.dst_shape[-1] *
                                  self.dst_shape[-2] % MAX_MASK)
        # cast the data from int8 to float16
        _cast_dtype(tik_instance, ub_cast_fp16, ub_ori, cast_repeat_time,
                    cast_remainder, "int8_2_float16")

        scalar_zero = tik_instance.Scalar(dtype="float16", init_value=0.0)
        with tik_instance.if_scope(is_last == 1):
            if self.src_shape[-1] % CUBE_SIZE_2 != 0:
                mask = 0
                for i, _ in enumerate(range(CUBE_SIZE_2 -
                                            self.src_shape[-1] % CUBE_SIZE_2)):
                    mask += 2 ** (CUBE_SIZE_2 - 1 - i)

                tik_instance.vector_dup([0, mask],
                                        ub_cast_fp16[loop_num * CUBE_SIZE_2 -
                                                     CUBE_SIZE_2],
                                        scalar_zero, CUBE_SIZE,
                                        self.cast_num_byte // 2,
                                        loop_num * CUBE_SIZE_2 //
                                        self.cast_num_data)
        with tik_instance.if_scope(num_loop_time == self.dst_shape[-3] - 1):
            if (self.src_shape[-2] % CUBE_SIZE) != 0:
                dup_number = (CUBE_SIZE - self.src_shape[-2] % CUBE_SIZE) * \
                             self.dst_shape[-1] * loop_num
                offset = (self.src_shape[-2] % CUBE_SIZE) * \
                         self.dst_shape[-1] * loop_num
                self.vector_dup_zero(tik_instance, ub_cast_fp16,
                                     dup_number, offset)
        with tik_instance.for_range(0, loop_num) as num_col_cube:
            tik_instance.vadds(CUBE_SIZE_2,
                               ub_trans[num_col_cube *
                                        self.dst_shape[-2] *
                                        self.dst_shape[-1] +
                                        CUBE_SIZE_2 * num_col_cube],
                               ub_cast_fp16[num_col_cube * CUBE_SIZE_2],
                               scalar_zero, CUBE_SIZE,
                               self.cast_num_byte // 2,
                               self.cast_num_byte // 2,
                               self.cast_num_byte,
                               loop_num * self.dst_shape[-1] //
                               self.cast_num_data)

        cast_repeat_time.set_as((CUBE_SIZE + 1) * loop_num *
                                self.dst_shape[-1] // MAX_MASK)
        cast_remainder.set_as((CUBE_SIZE + 1) * loop_num * self.dst_shape[-1] %
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
        total_core_loop_num = self.dst_shape[-3]

        core_number = _set_core_num(total_core_loop_num)
        num_data_one_loop = self.dst_shape[-4] * \
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
            src_ub_index = 0
            with tik_instance.for_range(0, core_loop) as num_core_loop:

                total_core_loop = sum_core + num_core_loop
                num_third_last_axis = total_core_loop
                is_last = tik_instance.Scalar("uint64")
                is_last.set_as(0)
                with tik_instance.if_scope(num_third_last_axis ==
                                           self.dst_shape[-3] - 1):
                    is_last.set_as(1)
                with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                                   align_loop == 0,
                                                   num_core_loop !=
                                                   core_loop - 1)):
                    src_gm_index = num_third_last_axis * num_data_one_loop - \
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
                    with tik_instance.if_scope(
                        (self.dst_shape[-3] - align_loop) *
                        self.dst_shape[-1] * self.dst_shape[-2] //
                        self.num_data > MAX_STRIDE_BLK):
                        with tik_instance.for_range(0, self.dst_shape[-4]) \
                                as num_col_cube:
                            dst_gm_index = num_third_last_axis * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] - \
                                           (align_loop - 1) * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] + \
                                           num_col_cube * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] * \
                                           self.dst_shape[-3]
                            tik_instance.data_move(self.dst_gm[dst_gm_index],
                                                   ub_cast_int8
                                                   [align_loop *
                                                    self.dst_shape[-1] *
                                                    self.dst_shape[-2] *
                                                    num_col_cube +
                                                    CUBE_SIZE_2 *
                                                    num_col_cube],
                                                   0, 1,
                                                   align_loop *
                                                   self.dst_shape[-1] *
                                                   self.dst_shape[-2] //
                                                   self.num_data, 0, 0)
                    with tik_instance.else_scope():
                        dst_gm_index = num_third_last_axis * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2] - \
                                       (align_loop - 1) * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2]
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_cast_int8[0], 0,
                                               self.dst_shape[-4],
                                               align_loop *
                                               self.dst_shape[-1] *
                                               self.dst_shape[-2] //
                                               self.num_data,
                                               self.num_byte,
                                               (self.dst_shape[-3] -
                                                align_loop) *
                                               self.dst_shape[-1] *
                                               self.dst_shape[-2] //
                                               self.num_data)
                with tik_instance.if_scope(num_core_loop == core_loop - 1):

                    src_gm_index = num_third_last_axis * num_data_one_loop - \
                                   (remainder - 1) * num_data_one_loop
                    with tik_instance.if_scope(is_last == 1):
                        if self.src_shape[-2] % CUBE_SIZE != 0:
                            tik_instance.data_move(ub_ori[src_ub_index],
                                                   self.src_gm[src_gm_index],
                                                   0, 1,
                                                   (remainder *
                                                    num_data_one_loop -
                                                    (CUBE_SIZE -
                                                     self.src_shape[-2] %
                                                     CUBE_SIZE) *
                                                    self.dst_shape[-4] *
                                                    self.dst_shape[-1]) //
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
                    with tik_instance.if_scope(
                        (self.dst_shape[-3] - remainder) *
                        self.dst_shape[-1] * self.dst_shape[-2] //
                        self.num_data > MAX_STRIDE_BLK):
                        with tik_instance.for_range(0, self.dst_shape[-4]) \
                                as num_col_cube:
                            dst_gm_index = num_third_last_axis * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] - \
                                           (remainder - 1) * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] + \
                                           num_col_cube * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] * \
                                           self.dst_shape[-3]
                            tik_instance.data_move(self.dst_gm[dst_gm_index],
                                                   ub_cast_int8
                                                   [remainder *
                                                    self.dst_shape[-1] *
                                                    self.dst_shape[-2] *
                                                    num_col_cube +
                                                    CUBE_SIZE_2 *
                                                    num_col_cube],
                                                   0, 1,
                                                   remainder *
                                                   self.dst_shape[-1] *
                                                   self.dst_shape[-2] //
                                                   self.num_data, 0, 0)
                    with tik_instance.else_scope():
                        dst_gm_index = num_third_last_axis * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2] - \
                                       (remainder - 1) * self.dst_shape[-1] * \
                                       self.dst_shape[-2]

                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_cast_int8[0], 0,
                                               self.dst_shape[-4],
                                               remainder * self.dst_shape[-1] *
                                               self.dst_shape[-2] //
                                               self.num_data,
                                               self.num_byte,
                                               (self.dst_shape[-3] -
                                                remainder) *
                                               self.dst_shape[-1] *
                                               self.dst_shape[-2] //
                                               self.num_data)

        return tik_instance

    def format_transfer_case_one(self, tik_instance):
        """
        the transfer process when UB can put in 16 * last axis data,
        last axis % 32 != 0 and the shape of dst is 4-D
        """
        ub_ori_data = self.ub_memory
        ub_trans_data = ub_ori_data
        total_core_loop_num = self.dst_shape[-3]

        core_number = _set_core_num(total_core_loop_num)
        num_data_one_loop = self.dst_shape[-4] * \
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
                num_third_last_axis = total_core_loop
                src_gm_index = num_third_last_axis * self.src_shape[-1] * \
                               CUBE_SIZE
                src_ub_index = (num_core_loop % align_loop) * num_data_one_loop
                with tik_instance.if_scope(num_third_last_axis ==
                                           self.dst_shape[-3] - 1):
                    if self.src_shape[-2] % CUBE_SIZE != 0:
                        with tik_instance.for_range(
                            0, self.src_shape[-2] % CUBE_SIZE) as num_cube_row:
                            tik_instance.data_move(ub_ori
                                                   [src_ub_index +
                                                    num_cube_row *
                                                    self.dst_shape[-1] *
                                                    self.dst_shape[-4]],
                                                   self.src_gm
                                                   [src_gm_index +
                                                    num_cube_row *
                                                    self.src_shape[-1]],
                                                   0, 1,
                                                   self.dst_shape[-1] *
                                                   self.dst_shape[-4] //
                                                   self.num_data, 0, 0)
                    else:
                        with tik_instance.for_range(0, CUBE_SIZE) as \
                                num_cube_row:
                            tik_instance.data_move(ub_ori
                                                   [src_ub_index +
                                                    num_cube_row *
                                                    self.dst_shape[-1] *
                                                    self.dst_shape[-4]],
                                                   self.src_gm
                                                   [src_gm_index +
                                                    num_cube_row *
                                                    self.src_shape[-1]],
                                                   0, 1,
                                                   self.dst_shape[-1] *
                                                   self.dst_shape[-4] //
                                                   self.num_data, 0, 0)
                with tik_instance.else_scope():
                    with tik_instance.for_range(0, CUBE_SIZE) as \
                            num_cube_row:
                        tik_instance.data_move(ub_ori[src_ub_index +
                                                      num_cube_row *
                                                      self.dst_shape[-1] *
                                                      self.dst_shape[-4]],
                                               self.src_gm
                                               [src_gm_index +
                                                num_cube_row *
                                                self.src_shape[-1]],
                                               0, 1,
                                               self.dst_shape[-1] *
                                               self.dst_shape[-4] //
                                               self.num_data, 0, 0)
                is_last = tik_instance.Scalar("uint64")
                is_last.set_as(0)
                with tik_instance.if_scope(num_third_last_axis ==
                                           self.dst_shape[-3] - 1):
                    is_last.set_as(1)
                with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                                   align_loop == 0,
                                                   num_core_loop !=
                                                   core_loop - 1)):
                    self.data_rearrange_case_zero(tik_instance, ub_ori,
                                                  ub_cast_fp16, ub_trans,
                                                  ub_cast_int8, align_loop,
                                                  is_last)
                    with tik_instance.if_scope(
                        (self.dst_shape[-3] - align_loop) *
                        self.dst_shape[-1] * self.dst_shape[-2] //
                        self.num_data > MAX_STRIDE_BLK):
                        with tik_instance.for_range(0, self.dst_shape[-4]) \
                                as num_col_cube:

                            dst_gm_index = num_third_last_axis * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] - \
                                           (align_loop - 1) * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] + \
                                           num_col_cube * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] * \
                                           self.dst_shape[-3]
                            tik_instance.data_move(self.dst_gm[dst_gm_index],
                                                   ub_cast_int8
                                                   [align_loop *
                                                    self.dst_shape[-1] *
                                                    self.dst_shape[-2] *
                                                    num_col_cube +
                                                    CUBE_SIZE_2 *
                                                    num_col_cube],
                                                   0, 1,
                                                   align_loop *
                                                   self.dst_shape[-1] *
                                                   self.dst_shape[-2] //
                                                   self.num_data, 0, 0)
                    with tik_instance.else_scope():
                        dst_gm_index = num_third_last_axis * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2] - \
                                       (align_loop - 1) * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2]
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_cast_int8[0], 0,
                                               self.dst_shape[-4],
                                               align_loop *
                                               self.dst_shape[-1] *
                                               self.dst_shape[-2] //
                                               self.num_data,
                                               self.num_byte,
                                               (self.dst_shape[-3] -
                                                align_loop) *
                                               self.dst_shape[-1] *
                                               self.dst_shape[-2] //
                                               self.num_data)
                with tik_instance.if_scope(num_core_loop == core_loop - 1):

                    self.data_rearrange_case_zero(tik_instance, ub_ori,
                                                  ub_cast_fp16, ub_trans,
                                                  ub_cast_int8, remainder,
                                                  is_last)
                    with tik_instance.if_scope(
                        (self.dst_shape[-3] - remainder) *
                        self.dst_shape[-1] * self.dst_shape[-2] //
                        self.num_data > MAX_STRIDE_BLK):
                        with tik_instance.for_range(0, self.dst_shape[-4]) \
                                as num_col_cube:
                            dst_gm_index = num_third_last_axis * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] - \
                                           (remainder - 1) * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] + \
                                           num_col_cube * \
                                           self.dst_shape[-1] * \
                                           self.dst_shape[-2] * \
                                           self.dst_shape[-3]
                            tik_instance.data_move(self.dst_gm[dst_gm_index],
                                                   ub_cast_int8
                                                   [remainder *
                                                    self.dst_shape[-1] *
                                                    self.dst_shape[-2] *
                                                    num_col_cube +
                                                    CUBE_SIZE_2 *
                                                    num_col_cube],
                                                   0, 1,
                                                   remainder *
                                                   self.dst_shape[-1] *
                                                   self.dst_shape[-2] //
                                                   self.num_data, 0, 0)
                    with tik_instance.else_scope():
                        dst_gm_index = num_third_last_axis * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2] - \
                                       (remainder - 1) * \
                                       self.dst_shape[-1] * \
                                       self.dst_shape[-2]

                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_cast_int8[0], 0,
                                               self.dst_shape[-4],
                                               remainder * self.dst_shape[-1] *
                                               self.dst_shape[-2] //
                                               self.num_data, self.num_byte,
                                               (self.dst_shape[-3] -
                                                remainder) *
                                               self.dst_shape[-1] *
                                               self.dst_shape[-2] //
                                               self.num_data)

        return tik_instance

    def format_transfer_case_two(self, tik_instance):
        """
        the transfer process when UB can not put in last axis * 16 data
        """
        self.ub_memory = self.ub_memory - self.ub_memory % \
                         (CUBE_SIZE_2 * (CUBE_SIZE + 1))
        ub_ori_data = self.ub_memory
        ub_trans_data = ub_ori_data
        loop_col, loop_remainder = _cal_core_loop_python_one(
            CUBE_SIZE_2 * (CUBE_SIZE + 1), self.dst_shape[-4], ub_ori_data)
        loop_times = self.dst_shape[-3]
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
                num_outer_axis = (total_core_loop - num_loop_time) // \
                                 loop_times
                is_last = tik_instance.Scalar("uint64")
                is_last.set_as(0)
                with tik_instance.for_range(0, self.dst_shape[-4] //
                                            loop_col) as num_cube:
                    self.data_move_case_zero(tik_instance, ub_ori,
                                             ub_cast_fp16, ub_trans,
                                             ub_cast_int8, is_last,
                                             num_outer_axis, num_loop_time,
                                             num_cube,
                                             loop_col, loop_col)

                if loop_remainder != 0:
                    is_last.set_as(1)
                    self.data_move_case_zero(tik_instance, ub_ori,
                                             ub_cast_fp16, ub_trans,
                                             ub_cast_int8, is_last,
                                             num_outer_axis, num_loop_time,
                                             self.dst_shape[-4] // loop_col,
                                             loop_col, loop_remainder)

        return tik_instance

    def data_move_case_zero(self, tik_instance, ub_ori, ub_cast_fp16, ub_trans,
                            ub_cast_int8, is_last, num_outer_axis,
                            num_loop_time, loop_time, loop_col, loop_len):
        """
        the data move process of the transfer case is 2
        """
        with tik_instance.if_scope(tik.all(loop_time == self.dst_shape[-4] //
                                           loop_col - 1,
                                           self.dst_shape[-4] % loop_col ==
                                           0)):
            is_last.set_as(1)
        num_data_one_loop = self.dst_shape[-4] * self.dst_shape[-3] * \
                            self.dst_shape[-2] * self.dst_shape[-1]
        src_ub_index = 0
        if self.src_shape[-1] % CUBE_SIZE_2 != 0 or \
                (self.src_shape[-1] - loop_len * CUBE_SIZE_2) // \
                self.num_data > MAX_STRIDE_BLK:
            with tik_instance.if_scope(num_loop_time ==
                                       self.dst_shape[-3] - 1):
                if self.src_shape[-2] % CUBE_SIZE != 0:
                    with tik_instance.for_range(0, self.src_shape[-2] %
                                                CUBE_SIZE) as num_cube_col:
                        src_gm_index = num_outer_axis * self.src_shape[-1] * \
                                       self.src_shape[-2] + \
                                       (num_loop_time * CUBE_SIZE +
                                        num_cube_col) * self.src_shape[-1] + \
                                       loop_time * loop_col * CUBE_SIZE_2
                        tik_instance.data_move(ub_ori[loop_len *
                                                      CUBE_SIZE_2 *
                                                      num_cube_col],
                                               self.src_gm[src_gm_index],
                                               0, 1,
                                               loop_len * CUBE_SIZE_2 //
                                               self.num_data, 0, 0)
                else:
                    with tik_instance.for_range(0, CUBE_SIZE) \
                            as num_cube_col:
                        src_gm_index = num_outer_axis * self.src_shape[-1] * \
                                       self.src_shape[-2] + \
                                       (num_loop_time * CUBE_SIZE +
                                        num_cube_col) * self.src_shape[-1] + \
                                       loop_time * loop_col * CUBE_SIZE_2
                        tik_instance.data_move(ub_ori[loop_len *
                                                      CUBE_SIZE_2 *
                                                      num_cube_col],
                                               self.src_gm[src_gm_index],
                                               0, 1,
                                               loop_len * CUBE_SIZE_2 //
                                               self.num_data, 0, 0)
            with tik_instance.else_scope():
                with tik_instance.for_range(0, CUBE_SIZE) as num_cube_col:
                    src_gm_index = num_outer_axis * self.src_shape[-1] * \
                                   self.src_shape[-2] + \
                                   (num_loop_time * CUBE_SIZE +
                                    num_cube_col) * self.src_shape[-1] + \
                                   loop_time * loop_col * CUBE_SIZE_2
                    tik_instance.data_move(ub_ori[loop_len * CUBE_SIZE_2 *
                                                  num_cube_col],
                                           self.src_gm[src_gm_index],
                                           0, 1,
                                           loop_len * CUBE_SIZE_2 //
                                           self.num_data, 0, 0)
        else:
            src_gm_index = num_outer_axis * self.src_shape[-1] * \
                           self.src_shape[-2] + num_loop_time * CUBE_SIZE * \
                           self.src_shape[-1] + loop_time * loop_col * \
                           CUBE_SIZE_2
            with tik_instance.if_scope(num_loop_time ==
                                       self.dst_shape[-3] - 1):
                if self.src_shape[-2] % CUBE_SIZE != 0:
                    tik_instance.data_move(ub_ori[src_ub_index],
                                           self.src_gm[src_gm_index], 0,
                                           self.src_shape[-2] % CUBE_SIZE,
                                           loop_len,
                                           (self.src_shape[-1] -
                                            loop_len * CUBE_SIZE_2) //
                                           self.num_data,
                                           0)
                else:
                    tik_instance.data_move(ub_ori[src_ub_index],
                                           self.src_gm[src_gm_index],
                                           0, CUBE_SIZE,
                                           loop_len,
                                           (self.src_shape[-1] -
                                            loop_len * CUBE_SIZE_2) //
                                           self.num_data,
                                           0)
            with tik_instance.else_scope():
                tik_instance.data_move(ub_ori[src_ub_index],
                                       self.src_gm[src_gm_index],
                                       0, CUBE_SIZE,
                                       loop_len,
                                       (self.src_shape[-1] - loop_len *
                                        CUBE_SIZE_2) // self.num_data, 0)

        self.data_rearrange_case_one(tik_instance, ub_ori, ub_cast_fp16,
                                     ub_trans, ub_cast_int8, num_loop_time,
                                     loop_len, is_last)

        if((self.dst_shape[-3] - 1) * self.dst_shape[-1] *
           self.dst_shape[-2] // self.num_data > MAX_STRIDE_BLK):
            with tik_instance.for_range(0, loop_len) as \
                    num_col_cube:
                dst_gm_index = num_outer_axis * num_data_one_loop + \
                               num_loop_time * self.dst_shape[-1] * \
                               self.dst_shape[-2] + \
                               (loop_time * loop_col + num_col_cube) * \
                               self.dst_shape[-1] * self.dst_shape[-2] * \
                               self.dst_shape[-3]
                tik_instance.data_move(self.dst_gm[dst_gm_index],
                                       ub_cast_int8[num_col_cube *
                                                    CUBE_SIZE_2 *
                                                    (CUBE_SIZE + 1)],
                                       0, 1,
                                       self.dst_shape[-1] *
                                       self.dst_shape[-2] //
                                       self.num_data,
                                       0, 0)
        else:
            dst_gm_index = num_outer_axis * num_data_one_loop + \
                           num_loop_time * self.dst_shape[-1] * \
                           self.dst_shape[-2] + loop_time * \
                           loop_col * self.dst_shape[-1] * \
                           self.dst_shape[-2] * \
                           self.dst_shape[-3]
            tik_instance.data_move(self.dst_gm[dst_gm_index],
                                   ub_cast_int8[0],
                                   0, loop_len,
                                   self.dst_shape[-1] * self.dst_shape[-2] //
                                   self.num_data, self.num_byte,
                                   (self.dst_shape[-3] - 1) *
                                   self.dst_shape[-1] *
                                   self.dst_shape[-2] // self.num_data)

    def nd_2_nz_compute(self):
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
        return tik_instance

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        tik_instance = self.nd_2_nz_compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.src_gm],
                              outputs=[self.dst_gm])

        return tik_instance


@util.check_input_type(dict, dict, str, str, str)
def nd_2_nz(src, dst, src_format, dst_format, kernel_name="nd_2_nz"):
    """
    algorithm: nd_2_nz

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
        kernel name, default value is "nd_2_nz"

    Returns
    -------
    tik_instance: tik_instance
    """
    src_shape = src.get("shape")
    src_dtype = src.get("dtype").lower()
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(src_shape)
    check_list = ("float16", "float32", "int8")
    util.check_dtype_rule(src_dtype, check_list)

    if src_format.upper() not in {"NHWC", "NCHW", "ND"}:
        raise RuntimeError("The src_format of ND2Nz"
                           "only support NHWC, NCHW, ND.")

    if dst_format.upper() != "FRACTAL_NZ":
        raise RuntimeError("The dat_format of ND2Nz"
                           "only support FRACTAL_NZ.")

    src_shape = list(src_shape)
    if src_dtype == "int8":
        nd_2_nz_template_int8 = ND2NzComputeInt8(src_shape, src_dtype,
                                                 kernel_name)
        return nd_2_nz_template_int8.get_tik_instance()
    else:
        nd_2_nz_template = ND2NzCompute(src_shape, src_dtype, kernel_name)
        return nd_2_nz_template.get_tik_instance()
