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
http: // www.apache.org/licenses/LICENSE-2.0

hwcn_2_fractal_z_c04
"""

from functools import reduce as functools_reduce
from te import platform as tbe_platform
from topi.cce import util
from te import tik

# available ub size
TOTAL_UB_MEMORY = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
# available number of cores
MAX_CORE_NUM = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
# bytes of type float16
SIZE_TWO_BYTES = 2
# size of the cube unit
CUBE_SIZE = 16
# minimum unit of data_move: 32Bytes
DATA_MOVE_MIN_UNIT = 32
# maximum repeat number
MAX_REPEATS = 255
# maximum blk stride
MAX_STRIDE_BLK = 65535
# maximum mask
MAX_MASK = 128
# the value of C0
C0 = 4


# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-lines
# pylint: disable=too-many-arguments,unused-argument
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


class HWCN2FRACTALZC04Compute:
    """
    Rearranges data from HWCN format into FRACTAL_Z_C04 format

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
        divide the transfer case from hwcn to fractal_z_c04
    vector_dup_zero:
        vector_dup zeros when dup_number is Scalar
    vector_dup_zero_python:
        vector_dup zeros when dup_number is python variable
    data_rearrange_case_zero:
        rearrange data when UB can put in N1 * N0 * X0 data
    data_rearrange_case_one:
        rearrange data when UB can not put in N1 * N0 * X0 data
    data_move_gm2ub_n_align_zero:
        move data from gm to UB when UB can
        put in N1 * N0 * X0 data and N % 16 == 0
    data_move_gm2ub_n_align_one:
        move data from gm to UB when UB can not
        put in N1 * N0 * X0 data and N % 16 == 0
    data_move_case_zero:
        the data_move process when UB can put in N1 * N0 * X0 data and
        N % 16 == 0
    data_move_case_one:
        the data_move process when UB can put in N1 * N0 * X0 data and
        N % 16 != 0
    data_move_case_two:
        the data_move process when UB can not put in N1 * N0 * X0 data
    format_transfer_case_zero:
        the transfer process when UB can put in N1 * N0 * X0 data
    format_transfer_case_one:
        the transfer process when UB can not put in N1 * N0 * X0 data
    hwcn_2_fractal_z_c04_compute:
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
        self.src_shape = src_shape
        self.dtype = dtype
        self.kernel_name = kernel_name
        self.dst_shape = [(self.src_shape[0] * self.src_shape[1] * C0 +
                           CUBE_SIZE - 1) // CUBE_SIZE,
                          (self.src_shape[3] + CUBE_SIZE - 1) // CUBE_SIZE,
                          CUBE_SIZE, CUBE_SIZE]

        self.num_byte = SIZE_TWO_BYTES
        self.mask = MAX_MASK
        # the number of data that can be moved in each data_move
        self.num_data = DATA_MOVE_MIN_UNIT // self.num_byte
        util.check_shape_rule(self.dst_shape)
        util.check_tensor_shape_size(self.dst_shape)
        # the number of data that UB can put in
        self.ub_memory = min(TOTAL_UB_MEMORY, 252 * 1024) // self.num_byte // 2
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
                                              self.src_shape[:])
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
        divide the transfer case from hwcn to fractal_z_c04
        """
        format_transfer_case = 0
        if self.dst_shape[1] * self.dst_shape[2] * self.dst_shape[3] > \
                self.ub_memory:
            format_transfer_case = 1

        return format_transfer_case

    def vector_dup_zero(self, tik_instance, ub_trans, dup_number, offset):
        """
        vector_dup zeros when dup_number is Scalar
        """
        scalar_zero = tik_instance.Scalar(dtype=self.dtype, init_value=0.0)
        repeat_number = dup_number // MAX_MASK
        tail = dup_number % MAX_MASK

        with tik_instance.for_range(0, repeat_number // MAX_REPEATS) as \
                num_repeat_loop:
            tik_instance.vector_dup(MAX_MASK,
                                    ub_trans[MAX_MASK * MAX_REPEATS *
                                             num_repeat_loop + offset],
                                    scalar_zero,
                                    MAX_REPEATS,
                                    self.num_byte // 2,
                                    MAX_MASK // self.num_data)
        with tik_instance.if_scope(repeat_number % MAX_REPEATS != 0):
            tik_instance.vector_dup(MAX_MASK,
                                    ub_trans[repeat_number // MAX_REPEATS *
                                             MAX_MASK * MAX_REPEATS + offset],
                                    scalar_zero,
                                    repeat_number % MAX_REPEATS,
                                    self.num_byte // 2,
                                    MAX_MASK // self.num_data)
        with tik_instance.if_scope(tail != 0):
            tik_instance.vector_dup(tail,
                                    ub_trans[MAX_MASK * repeat_number +
                                             offset],
                                    scalar_zero,
                                    1,
                                    self.num_byte // 2,
                                    MAX_MASK // self.num_data)

    def vector_dup_zero_python(self, tik_instance, ub_trans, dup_number,
                               offset):
        """
        vector_dup zeros when dup_number is python variable
        """
        scalar_zero = tik_instance.Scalar(dtype=self.dtype, init_value=0.0)
        repeat_number = dup_number // MAX_MASK
        tail = dup_number % MAX_MASK

        with tik_instance.for_range(0, repeat_number // MAX_REPEATS) as \
                num_repeat_loop:
            tik_instance.vector_dup(MAX_MASK,
                                    ub_trans[MAX_MASK * MAX_REPEATS *
                                             num_repeat_loop + offset],
                                    scalar_zero,
                                    MAX_REPEATS,
                                    self.num_byte // 2,
                                    MAX_MASK // self.num_data)
        if repeat_number % MAX_REPEATS != 0:
            tik_instance.vector_dup(MAX_MASK,
                                    ub_trans[repeat_number // MAX_REPEATS *
                                             MAX_MASK * MAX_REPEATS + offset],
                                    scalar_zero,
                                    repeat_number % MAX_REPEATS,
                                    self.num_byte // 2,
                                    MAX_MASK // self.num_data)
        if tail != 0:
            tik_instance.vector_dup(tail,
                                    ub_trans[MAX_MASK * repeat_number +
                                             offset],
                                    scalar_zero,
                                    1,
                                    self.num_byte // 2,
                                    MAX_MASK // self.num_data)

    def data_rearrange_case_zero(self, tik_instance, ub_ori, ub_trans,
                                 loop_number, is_x_padding):
        """
        rearrange data when UB can put in N1 * N0 * X0 data
        """
        if self.src_shape[2] == C0:
            with tik_instance.if_scope(is_x_padding == 1):
                chw = self.src_shape[0] * self.src_shape[1] *\
                      self.src_shape[2]
                dup_number = (CUBE_SIZE - chw % CUBE_SIZE) *\
                             self.dst_shape[1] * self.dst_shape[2]
                offset = loop_number * self.dst_shape[1] *\
                         self.dst_shape[2] * self.dst_shape[3] -\
                         (CUBE_SIZE - chw % CUBE_SIZE) * \
                         self.dst_shape[1] * self.dst_shape[2]
                self.vector_dup_zero_python(tik_instance, ub_ori,
                                            dup_number, offset)

        if self.src_shape[3] % CUBE_SIZE != 0:
            scalar_zero = tik_instance.Scalar(dtype=self.dtype,
                                              init_value=0.0)
            mask = 0
            for i, _ in enumerate(range(CUBE_SIZE - self.src_shape[3] %
                                        CUBE_SIZE)):
                mask += 2 ** (CUBE_SIZE - 1 - i)

            with tik_instance.for_range(
                0, loop_number * CUBE_SIZE // MAX_REPEATS) as num_repeat:
                offset = (num_repeat * MAX_REPEATS + 1) * self.dst_shape[1] * \
                         self.dst_shape[2] - CUBE_SIZE
                tik_instance.vector_dup([0, mask],
                                        ub_ori[offset],
                                        scalar_zero,
                                        MAX_REPEATS, 0,
                                        self.dst_shape[1] *
                                        self.dst_shape[2] // self.num_data)
            with tik_instance.if_scope(
                loop_number * CUBE_SIZE % MAX_REPEATS != 0):
                offset = (loop_number * CUBE_SIZE // MAX_REPEATS *
                          MAX_REPEATS + 1) * self.dst_shape[1] *\
                         self.dst_shape[2] - CUBE_SIZE
                tik_instance.vector_dup([0, mask],
                                        ub_ori[offset],
                                        scalar_zero,
                                        loop_number * CUBE_SIZE % MAX_REPEATS,
                                        0,
                                        self.dst_shape[1] *
                                        self.dst_shape[2] //
                                        self.num_data)

        with tik_instance.for_range(0, loop_number) as num_cn_loop:
            offset = num_cn_loop * self.dst_shape[1] * self.dst_shape[2] * \
                     self.dst_shape[3]
            dst_list = [ub_trans[i * CUBE_SIZE + offset] for i in range(16)]
            src_list = [ub_ori[i * self.dst_shape[1] * self.dst_shape[2] +
                               offset] for i in range(16)]
            if self.dst_shape[1] == 1:
                tik_instance.vnchwconv(False, False, dst_list, src_list,
                                       1, 0, 0)
            else:
                tik_instance.vnchwconv(False, False, dst_list, src_list,
                                       self.dst_shape[1],
                                       CUBE_SIZE * CUBE_SIZE // self.num_data,
                                       self.num_byte // 2)

    def data_rearrange_case_one(self, tik_instance, ub_ori, ub_trans,
                                is_last, num_x0, loop_len):
        """
        rearrange data when UB can not put in N1 * N0 * X0 data
        """
        chw = self.src_shape[0] * self.src_shape[1] * C0

        if self.src_shape[2] == C0 and chw % CUBE_SIZE != 0:
            with tik_instance.if_scope(num_x0 == self.dst_shape[0] - 1):
                dup_number = (CUBE_SIZE - chw % CUBE_SIZE) * loop_len * \
                             self.dst_shape[2]
                offset = (chw % CUBE_SIZE) * loop_len * self.dst_shape[2]
                self.vector_dup_zero_python(tik_instance, ub_ori,
                                            dup_number, offset)

        if self.src_shape[3] % CUBE_SIZE != 0:
            with tik_instance.if_scope(is_last == 1):
                scalar_zero = tik_instance.Scalar(dtype=self.dtype,
                                                  init_value=0.0)
                mask = 0
                for i, _ in enumerate(range(CUBE_SIZE - self.src_shape[3] %
                                            CUBE_SIZE)):
                    mask += 2 ** (CUBE_SIZE - 1 - i)
                offset = loop_len * self.dst_shape[2] - CUBE_SIZE
                tik_instance.vector_dup([0, mask],
                                        ub_ori[offset],
                                        scalar_zero,
                                        CUBE_SIZE, 0,
                                        loop_len * self.dst_shape[2] //
                                        self.num_data)

        dst_list = [ub_trans[i * CUBE_SIZE] for i in range(16)]
        src_list = [ub_ori[i * loop_len * self.dst_shape[2]]
                    for i in range(16)]
        if loop_len == 1:
            tik_instance.vnchwconv(False, False, dst_list, src_list,
                                   1, 0, 0)
        else:
            tik_instance.vnchwconv(False, False, dst_list, src_list,
                                   loop_len,
                                   CUBE_SIZE * CUBE_SIZE // self.num_data,
                                   self.num_byte // 2)

    def data_move_gm2ub_n_align_zero(self, tik_instance, ub_ori, loop_number,
                                     num_x0, num_data_one_loop):
        """
        move data from gm to UB when UB can put in N1 * N0 * X0 data and
        N % 16 == 0
        """
        if self.src_shape[2] == C0:
            src_gm_index = num_x0 * num_data_one_loop - \
                           (loop_number - 1) * num_data_one_loop
            src_ub_index = 0
            tik_instance.data_move(ub_ori[src_ub_index],
                                   self.src_gm[src_gm_index],
                                   0, 1,
                                   loop_number * num_data_one_loop //
                                   self.num_data, 0, 0)
        else:
            src_gm_index = (num_x0 - loop_number + 1) * self.src_shape[2] * \
                           self.src_shape[3] * CUBE_SIZE // C0
            src_ub_index = 0
            tik_instance.data_move(ub_ori[src_ub_index],
                                   self.src_gm[src_gm_index],
                                   0, CUBE_SIZE // C0 * loop_number,
                                   self.src_shape[2] *
                                   self.src_shape[3] // self.num_data,
                                   0,
                                   (C0 - self.src_shape[2]) *
                                   self.src_shape[3] // self.num_data)

    def data_move_gm2ub_n_align_one(self, tik_instance, ub_ori, src_gm_index,
                                    loop_len, cn_number):
        """
        move data from gm to UB when UB can not put in N1 * N0 * X0 data and
        N % 16 == 0
        """
        if self.src_shape[2] == C0:
            tik_instance.data_move(ub_ori[0],
                                   self.src_gm[src_gm_index],
                                   0, C0 * cn_number,
                                   loop_len * self.dst_shape[2] //
                                   self.num_data,
                                   (self.src_shape[3] - loop_len *
                                    self.dst_shape[2]) // self.num_data,
                                   0)
        else:
            with tik_instance.for_range(0, cn_number) as num_cn:
                tik_instance.data_move(ub_ori[num_cn * C0 * loop_len *
                                              self.dst_shape[2]],
                                       self.src_gm[src_gm_index + num_cn *
                                                   self.src_shape[2] *
                                                   self.src_shape[3]],
                                       0, self.src_shape[2],
                                       loop_len * self.dst_shape[2] //
                                       self.num_data,
                                       (self.src_shape[3] - loop_len *
                                        self.dst_shape[2]) // self.num_data,
                                       0)

    def data_move_case_zero(self, tik_instance, ub_ori, ub_trans, core_loop,
                            sum_core, align_loop, remainder,
                            num_data_one_loop):
        """
        the data_move process when UB can put in N1 * N0 * X0 data and
        N % 16 == 0
        """
        is_x_padding = tik_instance.Scalar("uint64", init_value=0)
        with tik_instance.for_range(0, core_loop) as num_core_loop:
            total_core_loop = sum_core + num_core_loop
            num_x0 = total_core_loop

            with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                               align_loop == 0,
                                               num_core_loop !=
                                               core_loop - 1)):
                if self.src_shape[2] != C0:
                    self.vector_dup_zero(tik_instance, ub_ori,
                                         align_loop * num_data_one_loop, 0)

                self.data_move_gm2ub_n_align_zero(tik_instance, ub_ori,
                                                  align_loop, num_x0,
                                                  num_data_one_loop)
                self.data_rearrange_case_zero(tik_instance, ub_ori, ub_trans,
                                              align_loop, is_x_padding)
                dst_gm_index = (num_x0 - (align_loop - 1)) * num_data_one_loop
                tik_instance.data_move(self.dst_gm[dst_gm_index],
                                       ub_trans[0],
                                       0, 1,
                                       align_loop * num_data_one_loop //
                                       self.num_data,
                                       0, 0)

            with tik_instance.if_scope(num_core_loop == core_loop - 1):
                # zero padding if C != 4
                if self.src_shape[2] != C0:
                    self.vector_dup_zero(tik_instance, ub_ori,
                                         remainder * num_data_one_loop, 0)

                if self.src_shape[0] * self.src_shape[1] * C0 % CUBE_SIZE != 0:
                    cn_number = CUBE_SIZE // C0 * (remainder - 1) +\
                                self.src_shape[0] * self.src_shape[1] %\
                                (CUBE_SIZE // C0)
                    with tik_instance.if_scope(num_x0 == self.dst_shape[0] -
                                               1):
                        is_x_padding.set_as(1)
                        if self.src_shape[2] == C0:
                            src_gm_index = num_x0 * num_data_one_loop - \
                                           (remainder - 1) * num_data_one_loop
                            src_ub_index = 0
                            burst_number = (remainder * num_data_one_loop -
                                            (CUBE_SIZE - self.src_shape[0] *
                                             self.src_shape[1] *
                                             self.src_shape[2] % CUBE_SIZE) *
                                            self.dst_shape[1] *
                                            self.dst_shape[2]) // self.num_data
                            tik_instance.data_move(ub_ori[src_ub_index],
                                                   self.src_gm[src_gm_index],
                                                   0, 1, burst_number, 0, 0)
                        else:
                            src_gm_index = (num_x0 - remainder + 1) * \
                                           self.src_shape[2] *\
                                           self.src_shape[3] * \
                                           CUBE_SIZE // C0
                            src_ub_index = 0
                            tik_instance.data_move(ub_ori[src_ub_index],
                                                   self.src_gm[src_gm_index],
                                                   0, cn_number,
                                                   self.src_shape[2] *
                                                   self.src_shape[3] //
                                                   self.num_data,
                                                   0,
                                                   (C0 - self.src_shape[2]) *
                                                   self.src_shape[3] //
                                                   self.num_data)
                    with tik_instance.else_scope():
                        self.data_move_gm2ub_n_align_zero(tik_instance, ub_ori,
                                                          remainder, num_x0,
                                                          num_data_one_loop)

                else:
                    self.data_move_gm2ub_n_align_zero(tik_instance, ub_ori,
                                                      remainder, num_x0,
                                                      num_data_one_loop)
                self.data_rearrange_case_zero(tik_instance, ub_ori, ub_trans,
                                              remainder, is_x_padding)
                dst_gm_index = (num_x0 - (remainder - 1)) * num_data_one_loop
                tik_instance.data_move(self.dst_gm[dst_gm_index],
                                       ub_trans[0],
                                       0, 1,
                                       remainder * num_data_one_loop //
                                       self.num_data,
                                       0, 0)

    def data_move_case_one(self, tik_instance, ub_ori, ub_trans, core_loop,
                           sum_core, align_loop, remainder, num_data_one_loop):
        """
        the data_move process when UB can put in N1 * N0 * X0 data and
        N % 16 != 0
        """
        is_x_padding = tik_instance.Scalar("uint64", init_value=0)
        with tik_instance.for_range(0, core_loop) as num_core_loop:

            total_core_loop = sum_core + num_core_loop
            num_x0 = total_core_loop
            # zero padding if C != 4
            with tik_instance.if_scope(num_core_loop % align_loop == 0):
                if self.src_shape[2] != C0:
                    self.vector_dup_zero(tik_instance, ub_ori,
                                         align_loop * num_data_one_loop, 0)

            src_gm_index = num_x0 * self.src_shape[3] * self.src_shape[2] * \
                           CUBE_SIZE // C0
            src_ub_index = (num_core_loop % align_loop) * num_data_one_loop
            if C0 * self.src_shape[0] * self.src_shape[1] % CUBE_SIZE != 0:
                with tik_instance.if_scope(num_x0 == self.dst_shape[0] - 1):
                    is_x_padding.set_as(1)
                    with tik_instance.for_range(0, self.src_shape[0] *
                                                self.src_shape[1] %
                                                (CUBE_SIZE // C0)) as num_cn:
                        with tik_instance.for_range(0, self.src_shape[2])\
                                as num_row:
                            tik_instance.data_move(ub_ori[src_ub_index +
                                                          (num_cn * C0 +
                                                           num_row) *
                                                          self.dst_shape[1] *
                                                          self.dst_shape[2]],
                                                   self.src_gm
                                                   [src_gm_index +
                                                    (num_cn *
                                                     self.src_shape[2] +
                                                     num_row) *
                                                    self.src_shape[3]],
                                                   0, 1,
                                                   self.dst_shape[1] *
                                                   self.dst_shape[2] //
                                                   self.num_data, 0, 0)
                with tik_instance.else_scope():
                    with tik_instance.for_range(0, CUBE_SIZE // C0) as num_cn:
                        with tik_instance.for_range(0, self.src_shape[2])\
                                as num_row:
                            tik_instance.data_move(ub_ori[src_ub_index +
                                                          (num_cn * C0 +
                                                           num_row) *
                                                          self.dst_shape[1] *
                                                          self.dst_shape[2]],
                                                   self.src_gm
                                                   [src_gm_index +
                                                    (num_cn *
                                                     self.src_shape[2] +
                                                     num_row) *
                                                    self.src_shape[3]],
                                                   0, 1,
                                                   self.dst_shape[1] *
                                                   self.dst_shape[2] //
                                                   self.num_data, 0, 0)
            else:
                with tik_instance.for_range(0, CUBE_SIZE // C0) as num_cn:
                    with tik_instance.for_range(0, self.src_shape[2])\
                            as num_row:
                        tik_instance.data_move(ub_ori[src_ub_index +
                                                      (num_cn * C0 +
                                                       num_row) *
                                                      self.dst_shape[1] *
                                                      self.dst_shape[2]],
                                               self.src_gm[src_gm_index +
                                                           (num_cn *
                                                            self.src_shape[2] +
                                                            num_row) *
                                                           self.src_shape[3]],
                                               0, 1,
                                               self.dst_shape[1] *
                                               self.dst_shape[2] //
                                               self.num_data, 0, 0)

            with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                               align_loop == 0,
                                               num_core_loop !=
                                               core_loop - 1)):
                self.data_rearrange_case_zero(tik_instance, ub_ori, ub_trans,
                                              align_loop, is_x_padding)
                dst_gm_index = (num_x0 - (align_loop - 1)) * num_data_one_loop
                tik_instance.data_move(self.dst_gm[dst_gm_index],
                                       ub_trans[0],
                                       0, 1,
                                       align_loop * num_data_one_loop //
                                       self.num_data,
                                       0, 0)

            with tik_instance.if_scope(num_core_loop == core_loop - 1):
                self.data_rearrange_case_zero(tik_instance, ub_ori, ub_trans,
                                              remainder, is_x_padding)
                dst_gm_index = (num_x0 - (remainder - 1)) * num_data_one_loop
                tik_instance.data_move(self.dst_gm[dst_gm_index],
                                       ub_trans[0],
                                       0, 1,
                                       remainder * num_data_one_loop //
                                       self.num_data,
                                       0, 0)

    def data_move_case_two(self, tik_instance, ub_ori, ub_trans, is_last,
                           num_x0, loop_time, loop_n, loop_len):
        """
        the data_move process when UB can put not in N1 * N0 * X0 data
        """
        ori_hwc = self.src_shape[0] * self.src_shape[1] * C0
        src_gm_index = num_x0 * self.src_shape[3] * self.src_shape[2] * \
                       CUBE_SIZE // C0 + loop_time * loop_n * CUBE_SIZE
        # zero padding if C != 4
        if self.src_shape[2] != C0:
            self.vector_dup_zero_python(tik_instance, ub_ori,
                                        loop_len * CUBE_SIZE * CUBE_SIZE, 0)

        if self.src_shape[3] % CUBE_SIZE != 0 or\
                (self.src_shape[3] - loop_len * self.dst_shape[2]) // \
                self.num_data > MAX_STRIDE_BLK:
            if ori_hwc % CUBE_SIZE != 0:
                with tik_instance.if_scope(num_x0 == self.dst_shape[0] - 1):
                    with tik_instance.for_range(0, self.src_shape[0] *
                                                self.src_shape[1] %
                                                (CUBE_SIZE // C0)) as num_cn:
                        with tik_instance.for_range(0, self.src_shape[2]) \
                                as num_row:
                            tik_instance.data_move(ub_ori[(num_cn * C0 +
                                                           num_row) *
                                                          loop_len *
                                                          self.dst_shape[2]],
                                                   self.src_gm
                                                   [src_gm_index +
                                                    (num_cn *
                                                     self.src_shape[2] +
                                                     num_row) *
                                                    self.src_shape[3]],
                                                   0, 1,
                                                   loop_len *
                                                   self.dst_shape[2] //
                                                   self.num_data,
                                                   0, 0)

                with tik_instance.else_scope():
                    with tik_instance.for_range(0, CUBE_SIZE // C0) as num_cn:
                        with tik_instance.for_range(0, self.src_shape[2]) \
                                as num_row:
                            tik_instance.data_move(ub_ori[(num_cn * C0 +
                                                           num_row) *
                                                          loop_len *
                                                          self.dst_shape[2]],
                                                   self.src_gm
                                                   [src_gm_index +
                                                    (num_cn *
                                                     self.src_shape[2] +
                                                     num_row) *
                                                    self.src_shape[3]],
                                                   0, 1,
                                                   loop_len *
                                                   self.dst_shape[2] //
                                                   self.num_data,
                                                   0, 0)
            else:
                with tik_instance.for_range(0, CUBE_SIZE // C0) as num_cn:
                    with tik_instance.for_range(0, self.src_shape[2]) \
                            as num_row:
                        tik_instance.data_move(ub_ori[(num_cn * C0 +
                                                       num_row) * loop_len *
                                                      self.dst_shape[2]],
                                               self.src_gm[src_gm_index +
                                                           (num_cn *
                                                            self.src_shape[2] +
                                                            num_row) *
                                                           self.src_shape[3]],
                                               0, 1,
                                               loop_len * self.dst_shape[2] //
                                               self.num_data,
                                               0, 0)
        else:
            if ori_hwc % CUBE_SIZE != 0:
                with tik_instance.if_scope(num_x0 == self.dst_shape[0] - 1):
                    cn_number = self.src_shape[0] * self.src_shape[1] % \
                                (CUBE_SIZE // C0)
                    self.data_move_gm2ub_n_align_one(tik_instance, ub_ori,
                                                     src_gm_index, loop_len,
                                                     cn_number)

                with tik_instance.else_scope():
                    self.data_move_gm2ub_n_align_one(tik_instance, ub_ori,
                                                     src_gm_index, loop_len,
                                                     C0)
            else:
                self.data_move_gm2ub_n_align_one(tik_instance, ub_ori,
                                                 src_gm_index, loop_len, C0)

        self.data_rearrange_case_one(tik_instance, ub_ori, ub_trans,
                                     is_last, num_x0, loop_len)

        dst_gm_index = num_x0 * self.dst_shape[1] * self.dst_shape[2] * \
                       self.dst_shape[3] + loop_time * loop_n *\
                       self.dst_shape[2] * self.dst_shape[3]
        tik_instance.data_move(self.dst_gm[dst_gm_index],
                               ub_trans[0],
                               0, 1,
                               loop_len * self.dst_shape[2] *
                               self.dst_shape[3] // self.num_data,
                               0, 0)

        return tik_instance

    def format_transfer_case_zero(self, tik_instance):
        """
        the transfer process when UB can put in N1 * N0 * X0 data
        """
        ub_ori_data = self.ub_memory
        ub_trans_data = ub_ori_data
        # divide the core according to X0
        total_core_loop_num = self.dst_shape[0]
        core_number = _set_core_num(total_core_loop_num)
        num_data_one_loop = self.dst_shape[1] * self.dst_shape[2] * \
                            self.dst_shape[3]

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
            if self.src_shape[3] % CUBE_SIZE == 0:
                self.data_move_case_zero(tik_instance, ub_ori, ub_trans,
                                         core_loop, sum_core, align_loop,
                                         remainder, num_data_one_loop)
            else:
                self.data_move_case_one(tik_instance, ub_ori, ub_trans,
                                        core_loop, sum_core, align_loop,
                                        remainder, num_data_one_loop)
        return tik_instance

    def format_transfer_case_one(self, tik_instance):
        """
        the transfer process when UB can not put in N1 * N0 * X0 data
        """
        ub_ori_data = self.ub_memory - self.ub_memory % (CUBE_SIZE * CUBE_SIZE)
        ub_trans_data = ub_ori_data
        loop_n, loop_remainder = _cal_core_loop_python(
            CUBE_SIZE * CUBE_SIZE, self.dst_shape[1], ub_ori_data)
        # divide the core according to X0
        total_core_loop_num = self.dst_shape[0]
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
                num_x0 = total_core_loop
                is_last = tik_instance.Scalar("uint64", init_value=0)
                with tik_instance.for_range(0, self.dst_shape[1] // loop_n) \
                        as num_n_loop:
                    with tik_instance.if_scope(tik.all(num_n_loop ==
                                                       self.dst_shape[1] //
                                                       loop_n - 1,
                                                       self.dst_shape[1] %
                                                       loop_n == 0)):
                        is_last.set_as(1)
                    self.data_move_case_two(tik_instance, ub_ori, ub_trans,
                                            is_last, num_x0, num_n_loop,
                                            loop_n, loop_n)

                if loop_remainder != 0:
                    is_last.set_as(1)
                    self.data_move_case_two(tik_instance, ub_ori, ub_trans,
                                            is_last, num_x0,
                                            self.dst_shape[1] // loop_n,
                                            loop_n, loop_remainder)

        return tik_instance

    def hwcn_2_fractal_z_c04_compute(self):
        """
        the overall data move process
        """
        tik_instance = self.set_tik_instance()
        format_transfer_case = self.set_format_transfer_case()
        if format_transfer_case == 0:
            tik_instance = self.format_transfer_case_zero(tik_instance)
        elif format_transfer_case == 1:
            tik_instance = self.format_transfer_case_one(tik_instance)
        return tik_instance

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        tik_instance = self.hwcn_2_fractal_z_c04_compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.src_gm],
                              outputs=[self.dst_gm])

        return tik_instance


@util.check_input_type(dict, dict, str, str, str)
def hwcn_2_fractal_z_c04(src, dst, src_format, dst_format,
                         kernel_name="hwcn_2_fractal_z_c04"):
    """
    algorithm: hwcn_2_fractal_z_c04

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
        kernel name, default value is "hwcn_2_fractal_z_c04"

    Returns
    -------
    tik_instance: tik_instance
    """
    src_shape = src.get("shape")
    src_dtype = src.get("dtype").lower()
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(src_shape)
    util.check_tensor_shape_size(src_shape)
    check_list = ("float16")
    util.check_dtype_rule(src_dtype, check_list)
    if len(src_shape) != 4:
        raise RuntimeError("hwcn_2_fractal_z_c04 only support 4D "
                           "while src shape is %s" %
                           ", ".join(src_shape))

    if src_shape[2] > 4:
        raise RuntimeError("hwcn_2_fractal_z_c04 only support C <= 4 "
                           "while src shape is %s" %
                           ", ".join(src_shape))

    if src_format.upper() != "HWCN":
        raise RuntimeError("hwcn_2_fractal_z_c04 only support %s "
                           "while src format is %s" %
                           ("HWCN", src_format))

    if dst_format.upper() != "FRACTAL_Z_C04":
        raise RuntimeError("hwcn_2_fractal_z_c04 only support %s "
                           "while dst format is %s" %
                           ("FRACTAL_Z_C04", dst_format))

    src_shape = list(src_shape)

    hwcn_2_fractal_z_c04_template = HWCN2FRACTALZC04Compute(src_shape,
                                                            src_dtype,
                                                            kernel_name)
    return hwcn_2_fractal_z_c04_template.get_tik_instance()
