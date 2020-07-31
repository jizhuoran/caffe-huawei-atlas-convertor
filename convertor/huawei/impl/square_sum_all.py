#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

square_sum_all
"""

import math
import operator
from functools import reduce as functools_reduce

import te.platform.cce_params as cce_params
from te import tik
from te import platform as tbe_platform
from topi.cce import util

SHAPE_SIZE_LIMIT = 2**30

# Each core processing data num greater than the size
# we can get better performace from experience
MINIMUM_DATA_NUM_EACH_CORE = 1024


# pylint: disable=too-many-instance-attributes
class SquareSumAll():
    """
    Function: use to store square_sum_all base parameters
    """

    # pylint: disable=too-many-statements
    def __init__(self, input_x, input_y, kernel_name):
        """
        Init square_sum_all base  parameters

        Parameters
        ----------
        input_x: dict
            data of input_x
            datatype suports float32
        input_y: dict
            data of input_y
            datatype suports float32
        kernel_name: str
            the name of the operator

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik()
        self.device_core_num = tbe_platform.cce_conf.get_soc_spec(\
            tbe_platform.cce_conf.CORE_NUM)
        self.input_x_shape = input_x.get("shape")
        self.input_x_dtype = input_x.get("dtype").lower()
        self.input_y_shape = input_y.get("shape")
        self.input_y_dtype = input_y.get("dtype").lower()

        self.shape_one_dim = (functools_reduce(operator.mul,
                                               self.input_x_shape), )
        self.input_x_num = self.shape_one_dim[0]

        self.ub_size_bytes = (tbe_platform.cce_conf.get_soc_spec(\
            tbe_platform.cce_conf.UB_SIZE) - 8192)

        self.kernel_name = kernel_name
        self.dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.input_x_dtype) // 8
        one_block_bytes_size = cce_params.VECTOR_INST_BLOCK_WIDTH // \
                               cce_params.VECTOR_INST_BLOCK_NUM
        self.data_each_block = one_block_bytes_size // self.dtype_bytes_size

        self.vector_process_bytes = 256

        self.vector_mask_max = cce_params.VECTOR_INST_BLOCK_NUM * \
                               self.data_each_block

        if self.input_x_num < self.data_each_block:
            self.block_num = 1
        else:
            ai_core_num = self.device_core_num
            temp_num = math.ceil(self.input_x_num / MINIMUM_DATA_NUM_EACH_CORE)
            if temp_num < 32:
                self.block_num = temp_num
            else:
                self.block_num = ai_core_num

        self.data_num_each_core = self.input_x_num // self.block_num
        self.remain_core = self.input_x_num % self.block_num
        self.process_data_num_each_core = self.data_num_each_core

        self.check_param()

        self.input_x_gm = self.tik_instance.Tensor(self.input_x_dtype,
                                                   self.shape_one_dim,
                                                   name="input_x_gm",
                                                   scope=tik.scope_gm)
        self.input_y_gm = self.tik_instance.Tensor(self.input_y_dtype,
                                                   self.shape_one_dim,
                                                   name="input_y_gm",
                                                   scope=tik.scope_gm)
        self.output_x_gm = self.tik_instance.Tensor(self.input_x_dtype, (1, ),
                                                    name="output_x_gm",
                                                    scope=tik.scope_gm,
                                                    is_atomic_add=True)
        self.output_y_gm = self.tik_instance.Tensor(self.input_y_dtype, (1, ),
                                                    name="output_y_gm",
                                                    scope=tik.scope_gm,
                                                    is_atomic_add=True)

        self.input_x_ub = None
        self.input_y_ub = None
        self.every_process_data_num = None
        self.process_times = None
        self.core_tail_num = None

    def check_param(self):
        """
        Check parameter

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        util.check_kernel_name(self.kernel_name)
        util.check_shape_rule(self.input_x_shape)
        util.check_shape_rule(self.input_y_shape)

        util.check_shape_size(self.input_x_shape, SHAPE_SIZE_LIMIT)
        util.check_shape_size(self.input_y_shape, SHAPE_SIZE_LIMIT)

        check_list_dtype = ("float32")

        util.check_dtype_rule(self.input_x_dtype, check_list_dtype)
        util.check_dtype_rule(self.input_y_dtype, check_list_dtype)

        add_support = tbe_platform.cce_conf.api_check_support(
            "tik.vadd", "float32")

        if self.input_x_dtype != self.input_y_dtype:
            raise RuntimeError(
                "input_x and input_y do not have the same dtype")

        if self.input_x_dtype == "float32" and not add_support:
            raise RuntimeError(
                "Input dtype is float32, but do not support on the platform")

    # pylint: disable=too-many-arguments
    def vector_add(self, mask, des_offset, src1_offset, src2_offset,
                   repeat_times):
        """
        Execute the vector add calculation

        Parameters
        ----------
        mask: int
            the mask of instruction
        des_offset: int
            destination address offset
        src1_offset: int
            src1 address offset
        src2_offset: int
            src2 address offset
        repeat_times: int
            the repeat times of instruction
        Returns
        -------
        None
        """
        self.tik_instance.vadd(mask, self.input_x_ub[des_offset],
                               self.input_x_ub[src1_offset],
                               self.input_x_ub[src2_offset], repeat_times, 1,
                               1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.input_y_ub[des_offset],
                               self.input_y_ub[src1_offset],
                               self.input_y_ub[src2_offset], repeat_times, 1,
                               1, 1, 8, 8, 8)

    def reduce_sum(self, calc_num):
        """
        Execute add calculation

        Parameters
        ----------
        calc_num: int
            the number of tensor elements in add calculation
        Returns
        calc_num: int
            the number of tensor elements in add calculation next time
        """

        # ensured the data address aligned 32b after tensor divided by 2
        align_value = self.data_each_block * 2
        tail_num = calc_num % (align_value)
        calc_num = (calc_num // align_value) * align_value
        total_sum_num = calc_num

        calc_num = calc_num // 2
        add_loop = calc_num // (self.vector_mask_max * 255)
        calc_offset = 0
        if add_loop > 0:
            with self.tik_instance.for_range(0, add_loop) as add_index:
                calc_offset = add_index * self.vector_mask_max * 255
                self.vector_add(self.vector_mask_max, calc_offset, calc_offset,
                                calc_offset + calc_num, 255)
            calc_offset = add_loop * self.vector_mask_max * 255
        repeat_time = (calc_num %
                       (self.vector_mask_max * 255)) // self.vector_mask_max
        if repeat_time > 0:
            self.vector_add(self.vector_mask_max, calc_offset, calc_offset,
                            calc_offset + calc_num, repeat_time)
        last_num = calc_num % self.vector_mask_max
        if last_num > 0:
            calc_offset += repeat_time * self.vector_mask_max
            self.vector_add(last_num, calc_offset, calc_offset,
                            calc_offset + calc_num, 1)
        if tail_num > 0:
            last_num = tail_num % self.vector_mask_max
            self.vector_add(last_num, 0, 0, total_sum_num, 1)
        return calc_num

    def init_ub_tensor(self, process_data_num):
        """
        Compute the ub size of tensors

        Parameters
        ----------
        process_data_num: int
            the number of process data each core

        Returns
        -------
        None
        """
        process_data_num_each_core = process_data_num
        every_process_data_num = process_data_num
        ub_max_num = self.ub_size_bytes // self.dtype_bytes_size

        if process_data_num_each_core > ub_max_num // 2:
            every_process_data_num = ub_max_num // 2

        self.every_process_data_num = every_process_data_num
        self.process_times = process_data_num_each_core // \
                             every_process_data_num
        self.core_tail_num = process_data_num_each_core % \
                             every_process_data_num

        flag = self.data_each_block
        assign_ub_shape = (math.ceil(every_process_data_num / flag) * flag, )

        self.input_x_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                                   assign_ub_shape,
                                                   name="input_x_ub",
                                                   scope=tik.scope_ubuf)
        self.input_y_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                                   assign_ub_shape,
                                                   name="input_y_ub",
                                                   scope=tik.scope_ubuf)

    def execute_square_sum_all(self, data_offset):
        """
        execute square_sum operation

        Parameters
        ----------
        data_offset: int
            the offset of data address in different core

        Returns
        -------
        None
        """
        temp_offset = data_offset
        with self.tik_instance.for_range(0, self.process_times) as i:
            data_offset = data_offset + i * self.every_process_data_num
            self.calc_op(self.every_process_data_num, data_offset)

        data_offset = temp_offset + self.every_process_data_num * \
                      self.process_times
        if self.core_tail_num > 0:
            self.calc_op(self.core_tail_num, data_offset)

    # pylint: disable=too-many-arguments,
    def vector_mul(self, mask, des_offset, src1_offset, src2_offset,
                   repeat_times):
        """
        Execute the vector mul calculation

        Parameters
        ----------
        mask: int
            the mask of instruction
        des_offset: int
            destination address offset
        src1_offset: int
            src1 address offset
        src2_offset: int
            src2 address offset
        repeat_times: int
            the repeat times of instruction
        Returns
        -------
        None
        """
        self.tik_instance.vmul(mask, self.input_x_ub[des_offset],
                               self.input_x_ub[src1_offset],
                               self.input_x_ub[src2_offset], repeat_times, 1,
                               1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.input_y_ub[des_offset],
                               self.input_y_ub[src1_offset],
                               self.input_y_ub[src2_offset], repeat_times, 1,
                               1, 1, 8, 8, 8)

    def calc_op(self, calc_num, offset):
        """
        every process square_sum

        Parameters
        ----------
        calc_num: int
            the number of every process data
        offset: int
            the offset of data address

        Returns
        -------
        None
        """
        burst_len = math.ceil(calc_num / self.data_each_block)
        self.tik_instance.data_move(self.input_x_ub, self.input_x_gm[offset],
                                    0, 1, burst_len, 0, 0)
        self.tik_instance.data_move(self.input_y_ub, self.input_y_gm[offset],
                                    0, 1, burst_len, 0, 0)

        calc_loop = calc_num // (self.vector_mask_max * 255)
        calc_offset = 0
        if calc_loop > 0:
            with self.tik_instance.for_range(0, calc_loop) as add_index:
                calc_offset = add_index * self.vector_mask_max * 255
                self.vector_mul(self.vector_mask_max, calc_offset, calc_offset,
                                calc_offset, 255)
            calc_offset = self.vector_mask_max * 255 * (calc_loop)

        repeat_time = (calc_num % (self.vector_mask_max * 255) //
                       self.vector_mask_max)

        if repeat_time > 0:
            self.vector_mul(self.vector_mask_max, calc_offset, calc_offset,
                            calc_offset, repeat_time)
        last_num = calc_num % self.vector_mask_max
        if last_num > 0:
            calc_offset += repeat_time * self.vector_mask_max
            self.vector_mul(last_num, calc_offset, calc_offset, calc_offset, 1)
        while calc_num > self.vector_process_bytes // self.dtype_bytes_size:
            calc_num = self.reduce_sum(calc_num)
            if (calc_num <=
                    self.vector_process_bytes // self.dtype_bytes_size):
                break
        vcadd_mask = calc_num
        self.tik_instance.vcadd(vcadd_mask, self.input_x_ub, self.input_x_ub,
                                1, 1, 1, 8)
        self.tik_instance.vcadd(vcadd_mask, self.input_y_ub, self.input_y_ub,
                                1, 1, 1, 8)

        self.tik_instance.set_atomic_add(1)
        self.tik_instance.data_move(self.output_x_gm, self.input_x_ub, 0, 1, 1,
                                    0, 0)
        self.tik_instance.data_move(self.output_y_gm, self.input_y_ub, 0, 1, 1,
                                    0, 0)
        self.tik_instance.set_atomic_add(0)

    def square_sum_all_operator(self):
        """
        SquareSumAll operation

        Parameters
        ----------
        None

        Returns:

        ----------
        tik_instance: tik instance
        """
        if self.block_num > 1:
            process_data_num = self.data_num_each_core
            process_data_extern_num = self.data_num_each_core + self.remain_core
            with self.tik_instance.for_range(
                    0, self.block_num, block_num=self.block_num) as loop_index:
                move_offset = loop_index * self.data_num_each_core
                with self.tik_instance.if_scope(loop_index == self.block_num -
                                                1):
                    self.init_ub_tensor(process_data_extern_num)
                    self.execute_square_sum_all(move_offset)
                with self.tik_instance.else_scope():
                    self.init_ub_tensor(process_data_num)
                    self.execute_square_sum_all(move_offset)
        else:
            self.init_ub_tensor(self.data_num_each_core)
            move_offset = 0
            self.execute_square_sum_all(move_offset)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.input_x_gm, self.input_y_gm),
                                   outputs=(self.output_x_gm,
                                            self.output_y_gm),
                                   enable_l2=False)

        return self.tik_instance

# pylint: disable=unused-argument
def square_sum_all(input_x,
                   input_y,
                   output_x,
                   output_y,
                   kernel_name="square_sum"):
    """
    calculating square_sum

    Parameters
    ----------
    input_x: dict
        input tensor contains shape and dtype attributes.
        only support float32.
    input_y: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype and shape as 'input_x'.
    output_x: dict
        output tensor contains shape and dtype attributes.
        Must have the same dtype  as 'input_x'.
    output_y: dict
        output tensor contains shape and dtype attributes.
        Must have the same dtype as 'input_x'.

    Returns
    -------
    None
    """

    square_sum_all_res = SquareSumAll(input_x, input_y, kernel_name)
    square_sum_all_res.square_sum_all_operator()
