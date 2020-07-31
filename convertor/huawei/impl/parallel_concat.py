#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

parallel_concact
"""
# pylint: disable=import-error
from te import tik
from te import platform as tbe_platform
from topi.cce import util
from impl import common_util

# pylint: disable=redefined-builtin
if "reduce" not in dir(__builtins__):
    from functools import reduce

# available ub size
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(
    tbe_platform.cce_conf.UB_SIZE)


def ceil_align(ori_num, divider):
    '''
    dst_num = (ori_num + divider -1) / divider * divider
    :param ori_num: original number
    :param divider: divider
    :return:
    '''
    return (ori_num + divider - 1) // divider * divider


def floor_align(ori_num, divider):
    '''
    dst_num = ori_num // divider * divider
    :param ori_num: original number
    :param divider: divider
    :return:
    '''
    return ori_num // divider * divider


def ceil_divide(ori_num, divider):
    '''
    dst_num = (ori_num + divider -1) / divider
    :param ori_num: original number
    :param divider: divider
    :return:
    '''
    return (ori_num + divider - 1) // divider


def _check_param(input_values, shape, kernel_name):
    """
    check the parameters is valid, if one is invalid,then raise error

    Parameters
    ----------
    input_values : A list of `dict`.dict include keys shape and dtype
    shape : list of the output shape
    kernel_name : cce kernel name, default value is "parallel_concat"
    Returns
    -------
    None
    """
    check_list = ("int8", "int16", "int32", "int64", "uint8",
                  "uint16", "uint32", "uint64", "float16", "float32")
    if shape[0] != len(input_values):
        raise RuntimeError(
            "first dim of output shape must be equal to the num of"
            "input tensors,"
            "the first dim of output shape is {0}, "
            "while the num of input tensors is {1}".format(
                shape[0], len(input_values))
        )
    first_shape = input_values[0].get("shape")
    first_dtype = input_values[0].get("dtype")
    for i, input_dict in enumerate(input_values):
        shape_input = input_dict.get("shape")
        dtype_input = input_dict.get("dtype").lower()
        util.check_shape_rule(shape_input)
        util.check_tensor_shape_size(shape_input)
        util.check_dtype_rule(dtype_input, check_list)
        if shape_input != first_shape:
            raise RuntimeError(
                "the input tensors shape must be same, "
                "while the {}th input shape is {}, "
                "not same with the first shape {}".format(i, shape_input, first_shape)
            )
        if shape_input[0] != 1:
            raise RuntimeError(
                "the first dim of all input tensor must be 1, "
                "while the {}th input shape is {}".format(i, shape_input)
            )
        if first_dtype != dtype_input:
            raise RuntimeError(
                "the input tensors dtype must be same,"
                "while the {}th input dtype is {}, "
                "not same with the first dtype {}".format(i, dtype_input, first_dtype)
            )
    if list(shape[1:]) != list(first_shape[1:]):
        raise RuntimeError(
            "the input shape {} do not match the output shape {}".format(first_shape, shape))
    util.check_kernel_name(kernel_name)

# pylint: disable=too-many-locals,invalid-name,unused-argument
@util.check_input_type((list, tuple), dict, (list, tuple), int, str)
def parallel_concat(values, output_data, shape, num, kernel_name="parallel_concat"):
    """
    algorithm: parallel_concat
    Concatenates tensors along one dimension.
    Parameters
    ----------
    values : A list of `dict`.dict include keys shape and dtype
    output_data: dict of output data, dict include keys shape and dtype
    shape : list of the output shape
    num: The nums of values
    kernel_name : cce kernel name, default value is "parallel_concat"
    Returns
    -------
    None
    """
    if len(values) != num:
        raise RuntimeError('The size of input and num must be same.')

    _check_param(values, shape, kernel_name)
    fun = ParallelConcat(values, shape, kernel_name)
    return fun.parallel_concat_compute()


class ParallelConcatBase:
    """
    Function: use to store parallel_concat base parameters
    Modify : 2019-12-10
    """

    def __init__(self, input_data, shape, kernel_name):
        """
        init parallel_concat base parameters
        Parameters
        ----------
        input_data: shape and data type,data type supports float16,float32,
                    int32,uint32,int16,uint16,int8,uint8,int64,uint64
        shape: list of output shape
        kernel_name: cce kernel name, default value is "parallel_concat"
        """
        self.data_shape = []
        self.data_dtype = []
        for _, input_dict in enumerate(input_data):
            shape_input = input_dict.get("shape")
            dtype_input = (input_dict.get("dtype")).lower()
            self.data_shape.append(shape_input)
            self.data_dtype.append(dtype_input)
        self.output_shape = shape
        self.kernel_name = kernel_name
        self.dtype_size = common_util.get_data_size(self.data_dtype[0])
        self.product_core_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        self.tik_instance = tik.Tik()

    def get_input_size(self):
        """
        get the size of input data

        Parameters
        ----------
        Returns
        -------
        size: the size of input data
        """
        return reduce(lambda x1, x2: x1 * x2, self.data_shape[0])

    def get_output_size(self):
        """
        get the size of output data

        Parameters
        ----------
        Returns
        -------
        size: the size of output data
        """
        return reduce(lambda x1, x2: x1 * x2, self.output_shape)


# pylint: disable=too-many-instance-attributes
class ParallelConcat(ParallelConcatBase):
    """
    Function: use to store patallel_concat compute parameters
    Modify : 2019-12-10
    """
    def __init__(self, input_data, shape, kernel_name):
        """
        init parallel_concat base parameters
        Parameters
        ----------
        input_data: shape and data type,data type supports float16,float32,
                    int32,uint32,int16,uint16,int8,uint8
        shape: list of output shape
        kernel_name: cce kernel name, default value is "parallel_concat"
        """
        super(ParallelConcat, self).__init__(input_data, shape, kernel_name)
        self.data_gm_in = []
        self.single_input_element_num = self.get_input_size()
        self.output_size = self.get_output_size()
        self.input_tensor_num = len(self.data_shape)
        for i in range(self.input_tensor_num):
            self.data_gm_in.append(self.tik_instance.Tensor(
                self.data_dtype[0], self.data_shape[0],
                scope=tik.scope_gm, name="data_gm_in_{}".format(i)))
        self.data_gm_out = self.tik_instance.Tensor(
            self.data_dtype[0], self.output_shape, scope=tik.scope_gm, name="data_gm_out")
        self.tiling_case, self.single_tensor_size, self.max_single_tensor_size_all_core, \
        self.max_single_tensor_size_each_core = self.get_buf_info()

    def get_buf_info(self):
        """
        get the ub buffer size
        Returns
        -------
        tiling_case: the index of tiling shape
        buf_size: the ub buffer size
        """
        single_tensor_size = self.single_input_element_num * self.dtype_size
        max_single_tensor_size_each_core = floor_align(UB_SIZE//self.input_tensor_num, 32)
        max_single_tensor_size_all_core = max_single_tensor_size_each_core * self.product_core_num
        if single_tensor_size <= max_single_tensor_size_each_core:
            tiling_case = 1
        elif max_single_tensor_size_each_core < single_tensor_size <= \
                max_single_tensor_size_all_core:
            tiling_case = 2
        else:
            tiling_case = 3
        return tiling_case, \
               single_tensor_size, \
               max_single_tensor_size_all_core, \
               max_single_tensor_size_each_core

    # pylint: disable=too-many-branches,too-many-statements
    def parallel_concat_compute(self):
        """
        describe the ParallelConcat calculation process
        Returns
        -------
        tik_instance: tik_instance
        """
        if self.tiling_case == 1:
            single_tensor_buf_size_needed = ceil_align(self.single_tensor_size, 32)
            data_ub = self.tik_instance.Tensor(
                self.data_dtype[0], (single_tensor_buf_size_needed//self.dtype_size,),
                scope=tik.scope_ubuf, name="data_ubuf")
            for i in range(self.input_tensor_num):
                # move input from gm to ub
                self.tik_instance.data_move(data_ub[0], self.data_gm_in[i][0], 0, 1,
                                            single_tensor_buf_size_needed//32, 0, 0)
                # move data from ub to gm
                gm_out_offset = i * self.single_input_element_num
                self.tik_instance.data_move(self.data_gm_out[gm_out_offset],
                                            data_ub[0], 0, 1,
                                            single_tensor_buf_size_needed//32, 0, 0)
        elif self.tiling_case == 2:
            block_num_needed = ceil_divide(self.single_tensor_size,
                                           self.max_single_tensor_size_each_core)
            tail = self.single_tensor_size % self.max_single_tensor_size_each_core
            # the easiest situation: tail == 0
            if tail == 0:
                with self.tik_instance.for_range(0, block_num_needed,
                                                 thread_num=1,
                                                 block_num=block_num_needed) as block_idx:
                    data_ub = []
                    for i in range(self.input_tensor_num):
                        data_ub.append(self.tik_instance.Tensor(
                            self.data_dtype[0],
                            (self.max_single_tensor_size_each_core//self.dtype_size,),
                            scope=tik.scope_ubuf, name="data_ubuf_{}".format(i)))
                    for i in range(self.input_tensor_num):

                        # move input from gm to ub
                        gm_in_offset = (block_idx * self.max_single_tensor_size_each_core) \
                                       // self.dtype_size
                        self.tik_instance.data_move(
                            data_ub[i][0],
                            self.data_gm_in[i][gm_in_offset],
                            0, 1, self.max_single_tensor_size_each_core//32, 0, 0)
                        # move data from ub to gm
                        gm_out_offset = (i * self.single_input_element_num) + \
                                        (block_idx * self.max_single_tensor_size_each_core) \
                                        // self.dtype_size
                        self.tik_instance.data_move(
                            self.data_gm_out[gm_out_offset],
                            data_ub[i][0], 0, 1,
                            self.max_single_tensor_size_each_core//32, 0, 0)
            # tail != 0 but align with 32B
            elif self.single_tensor_size % 32 == 0:
                data_ub = []
                for i in range(self.input_tensor_num):
                    data_ub.append(self.tik_instance.Tensor(
                        self.data_dtype[0],
                        (self.max_single_tensor_size_each_core//self.dtype_size,),
                        scope=tik.scope_ubuf, name="data_ubuf_{}".format(i)))
                with self.tik_instance.for_range(0, block_num_needed,
                                                 block_num=block_num_needed) as block_idx:

                    for i in range(self.input_tensor_num):

                        with self.tik_instance.if_scope(block_idx != block_num_needed-1):
                            # move input from gm to ub
                            gm_in_offset = (block_idx * self.max_single_tensor_size_each_core) \
                                           // self.dtype_size
                            self.tik_instance.data_move(
                                data_ub[i][0],
                                self.data_gm_in[i][gm_in_offset],
                                0, 1, self.max_single_tensor_size_each_core//32, 0, 0)
                            # move data from ub to gm
                            gm_out_offset = (i * self.single_input_element_num) + \
                                            (block_idx * self.max_single_tensor_size_each_core) \
                                            // self.dtype_size
                            self.tik_instance.data_move(
                                self.data_gm_out[gm_out_offset],
                                data_ub[i][0], 0, 1,
                                self.max_single_tensor_size_each_core//32, 0, 0)
                        # last core handle the tail
                        with self.tik_instance.else_scope():
                            burst_len = tail
                            # move input from gm to ub
                            gm_in_offset = (block_idx * self.max_single_tensor_size_each_core) \
                                           // self.dtype_size
                            self.tik_instance.data_move(data_ub[i][0],
                                                        self.data_gm_in[i][gm_in_offset],
                                                        0, 1, burst_len//32, 0, 0)
                            # move data from ub to gm
                            gm_out_offset = (i * self.single_input_element_num) + \
                                            (block_idx *
                                             self.max_single_tensor_size_each_core) \
                                            // self.dtype_size
                            self.tik_instance.data_move(self.data_gm_out[gm_out_offset],
                                                        data_ub[i][0], 0, 1,
                                                        burst_len//32, 0, 0)
            # tail != 0 and not align with 32B
            else:
                # multi core handle the segments align with 32B
                with self.tik_instance.for_range(0, block_num_needed-1,
                                                 block_num=block_num_needed-1) as block_idx:
                    data_ub = []
                    for i in range(self.input_tensor_num):
                        data_ub.append(self.tik_instance.Tensor(
                            self.data_dtype[0],
                            (self.max_single_tensor_size_each_core//self.dtype_size,),
                            scope=tik.scope_ubuf, name="data_ubuf_{}".format(i)))
                    for i in range(self.input_tensor_num):

                        # move input from gm to ub
                        gm_in_offset = (block_idx * self.max_single_tensor_size_each_core) \
                                       // self.dtype_size
                        self.tik_instance.data_move(
                            data_ub[i][0],
                            self.data_gm_in[i][gm_in_offset],
                            0, 1,
                            self.max_single_tensor_size_each_core//32, 0, 0)
                        # move data from ub to gm
                        gm_out_offset = (i * self.single_input_element_num) + \
                                        (block_idx * self.max_single_tensor_size_each_core) \
                                        // self.dtype_size
                        self.tik_instance.data_move(
                            self.data_gm_out[gm_out_offset],
                            data_ub[i][0], 0, 1,
                            self.max_single_tensor_size_each_core//32, 0, 0)
                data_ub = []
                for j in range(self.input_tensor_num):
                    data_ub.append(self.tik_instance.Tensor(
                        self.data_dtype[0],
                        (self.max_single_tensor_size_each_core//self.dtype_size,),
                        scope=tik.scope_ubuf, name="data_ubuf_{}".format(j)))
                # single core handle the tail
                for j in range(self.input_tensor_num):
                    burst_len = ceil_align(tail, 32)
                    # move input from gm to ub
                    gm_in_offset = \
                        ((block_num_needed-1) * self.max_single_tensor_size_each_core) \
                        // self.dtype_size
                    self.tik_instance.data_move(data_ub[j][0],
                                                self.data_gm_in[j][gm_in_offset],
                                                0, 1, burst_len//32, 0, 0)
                    # move data from ub to gm
                    gm_out_offset = (j * self.single_input_element_num) + gm_in_offset
                    self.tik_instance.data_move(self.data_gm_out[gm_out_offset],
                                                data_ub[j][0], 0, 1,
                                                burst_len//32, 0, 0)
                for k in range(self.input_tensor_num-1):
                    burst_len = 32
                    # move input from gm to ub
                    gm_in_offset = 0
                    self.tik_instance.data_move(data_ub[k][0],
                                                self.data_gm_in[k+1][gm_in_offset],
                                                0, 1, burst_len//32, 0, 0)
                    # move data from ub to gm
                    gm_out_offset = (k+1) * self.single_input_element_num
                    self.tik_instance.data_move(self.data_gm_out[gm_out_offset],
                                                data_ub[k][0], 0, 1,
                                                burst_len//32, 0, 0)

        # tiling case == 3
        else:

            block_num_needed = ceil_divide(self.single_tensor_size,
                                           self.max_single_tensor_size_each_core)
            loop_block = block_num_needed//self.product_core_num
            block_num_needed_last = block_num_needed - self.product_core_num * loop_block
            tail = self.single_tensor_size % self.max_single_tensor_size_each_core
            # the easiest situation: tail == 0
            if tail == 0:
                with self.tik_instance.for_range(0, self.product_core_num,
                                                 block_num=self.product_core_num) as block_idx:
                    data_ub = []
                    for i in range(self.input_tensor_num):
                        data_ub.append(self.tik_instance.Tensor(
                            self.data_dtype[0],
                            (self.max_single_tensor_size_each_core//self.dtype_size,),
                            scope=tik.scope_ubuf, name="data_ubuf_{}".format(i)))
                    # the former loop
                    for loop_block_i in range(loop_block):
                        for i in range(self.input_tensor_num):
                            # move input from gm to ub
                            gm_in_offset = ((loop_block_i * self.product_core_num + block_idx) *
                                            self.max_single_tensor_size_each_core)//self.dtype_size
                            self.tik_instance.data_move(
                                data_ub[i][0],
                                self.data_gm_in[i][gm_in_offset],
                                0, 1, self.max_single_tensor_size_each_core//32, 0, 0)
                            # move data from ub to gm
                            gm_out_offset = (i * self.single_input_element_num) + gm_in_offset
                            self.tik_instance.data_move(
                                self.data_gm_out[gm_out_offset],
                                data_ub[i][0], 0, 1,
                                self.max_single_tensor_size_each_core//32, 0, 0)
                    if block_num_needed_last > 0:
                        # handle the segments cannot be divide by product_core_num
                        with self.tik_instance.if_scope(
                                block_idx < block_num_needed_last):
                            for i in range(self.input_tensor_num):
                                # move input from gm to ub
                                gm_in_offset = \
                                    ((self.product_core_num * loop_block + block_idx) *
                                     self.max_single_tensor_size_each_core)//self.dtype_size
                                self.tik_instance.data_move(
                                    data_ub[i][0],
                                    self.data_gm_in[i][gm_in_offset],
                                    0, 1, self.max_single_tensor_size_each_core//32, 0, 0)
                                # move data from ub to gm
                                gm_out_offset = (i * self.single_input_element_num) + gm_in_offset
                                self.tik_instance.data_move(
                                    self.data_gm_out[gm_out_offset],
                                    data_ub[i][0], 0, 1,
                                    self.max_single_tensor_size_each_core//32, 0, 0)
            # tail != 0 but align with 32B
            elif self.single_tensor_size % 32 == 0:
                # tail is in big loop
                if block_num_needed_last == 0:
                    with self.tik_instance.for_range(0, self.product_core_num,
                                                     thread_num=1,
                                                     block_num=self.product_core_num) as block_idx:
                        data_ub = []
                        for i in range(self.input_tensor_num):
                            data_ub.append(self.tik_instance.Tensor(
                                self.data_dtype[0],
                                (self.max_single_tensor_size_each_core//self.dtype_size,),
                                scope=tik.scope_ubuf, name="data_ubuf_{}".format(i)))
                        # the former loop
                        for loop_block_i in range(loop_block-1):
                            for i in range(self.input_tensor_num):
                                # move input from gm to ub
                                gm_in_offset = ((self.product_core_num * loop_block_i + block_idx) *
                                                self.max_single_tensor_size_each_core) \
                                               // self.dtype_size
                                self.tik_instance.data_move(
                                    data_ub[i][0],
                                    self.data_gm_in[i][gm_in_offset],
                                    0, 1, self.max_single_tensor_size_each_core//32, 0, 0)
                                # move data from ub to gm
                                gm_out_offset = (i * self.single_input_element_num) + gm_in_offset
                                self.tik_instance.data_move(
                                    self.data_gm_out[gm_out_offset],
                                    data_ub[i][0], 0, 1,
                                    self.max_single_tensor_size_each_core//32, 0, 0)
                        # last special loop, last core of the 32 core not full
                        for i in range(self.input_tensor_num):
                            # the former 31 core
                            with self.tik_instance.if_scope(
                                    block_idx != self.product_core_num-1):
                                # move input from gm to ub
                                gm_in_offset = \
                                    ((loop_block-1) * self.product_core_num + block_idx) * \
                                    self.max_single_tensor_size_each_core \
                                    // self.dtype_size
                                self.tik_instance.data_move(
                                    data_ub[i][0],
                                    self.data_gm_in[i][gm_in_offset],
                                    0, 1, self.max_single_tensor_size_each_core//32, 0, 0)
                                # move data from ub to gm
                                gm_out_offset = (i * self.single_input_element_num) + gm_in_offset
                                self.tik_instance.data_move(
                                    self.data_gm_out[gm_out_offset],
                                    data_ub[i][0], 0, 1,
                                    self.max_single_tensor_size_each_core//32, 0, 0)
                            # the last core
                            with self.tik_instance.else_scope():
                                burst_len = tail
                                # move input from gm to ub
                                gm_in_offset = (block_num_needed - 1) * \
                                               self.max_single_tensor_size_each_core \
                                               // self.dtype_size
                                self.tik_instance.data_move(data_ub[i][0],
                                                            self.data_gm_in[i][gm_in_offset],
                                                            0, 1, burst_len//32, 0, 0)
                                # move data from ub to gm
                                gm_out_offset = (i * self.single_input_element_num) + gm_in_offset
                                self.tik_instance.data_move(self.data_gm_out[gm_out_offset],
                                                            data_ub[i][0], 0, 1,
                                                            burst_len//32, 0, 0)
                # tail is out of big loop
                else:
                    with self.tik_instance.for_range(0, self.product_core_num,
                                                     thread_num=1,
                                                     block_num=self.product_core_num) as block_idx:
                        data_ub = []
                        for i in range(self.input_tensor_num):
                            data_ub.append(self.tik_instance.Tensor(
                                self.data_dtype[0],
                                (self.max_single_tensor_size_each_core//self.dtype_size,),
                                scope=tik.scope_ubuf, name="data_ubuf_{}".format(i)))
                        # the former loop
                        for loop_block_i in range(loop_block):
                            for i in range(self.input_tensor_num):
                                # move input from gm to ub
                                gm_in_offset = ((self.product_core_num * loop_block_i + block_idx) *
                                                self.max_single_tensor_size_each_core) \
                                               // self.dtype_size
                                self.tik_instance.data_move(
                                    data_ub[i][0],
                                    self.data_gm_in[i][gm_in_offset],
                                    0, 1, self.max_single_tensor_size_each_core//32, 0, 0)
                                # move data from ub to gm
                                gm_out_offset = (i * self.single_input_element_num) + gm_in_offset
                                self.tik_instance.data_move(
                                    self.data_gm_out[gm_out_offset],
                                    data_ub[i][0], 0, 1,
                                    self.max_single_tensor_size_each_core//32, 0, 0)
                        # last special loop, last core of the n(n<32) core not full
                        # the former cores
                        with self.tik_instance.if_scope(block_idx < block_num_needed_last-1):
                            for i in range(self.input_tensor_num):
                                # move input from gm to ub
                                gm_in_offset = (loop_block * self.product_core_num + block_idx) * \
                                               self.max_single_tensor_size_each_core \
                                               // self.dtype_size
                                self.tik_instance.data_move(
                                    data_ub[i][0],
                                    self.data_gm_in[i][gm_in_offset],
                                    0, 1, self.max_single_tensor_size_each_core//32, 0, 0)
                                # move data from ub to gm
                                gm_out_offset = (i * self.single_input_element_num) + gm_in_offset
                                self.tik_instance.data_move(
                                    self.data_gm_out[gm_out_offset],
                                    data_ub[i][0], 0, 1,
                                    self.max_single_tensor_size_each_core//32, 0, 0)

                    data_ub = []
                    for i in range(self.input_tensor_num):
                        data_ub.append(self.tik_instance.Tensor(
                            self.data_dtype[0],
                            (self.max_single_tensor_size_each_core//self.dtype_size,),
                            scope=tik.scope_ubuf, name="data_ubuf_{}".format(i)))
                    # the last core
                    for i in range(self.input_tensor_num):
                        burst_len = tail
                        # move input from gm to ub
                        gm_in_offset = (block_num_needed-1) * \
                                       self.max_single_tensor_size_each_core//self.dtype_size
                        self.tik_instance.data_move(data_ub[i][0],
                                                    self.data_gm_in[i][gm_in_offset],
                                                    0, 1, burst_len//32, 0, 0)
                        # move data from ub to gm
                        gm_out_offset = (i * self.single_input_element_num) + gm_in_offset
                        self.tik_instance.data_move(self.data_gm_out[gm_out_offset],
                                                    data_ub[i][0], 0, 1,
                                                    burst_len//32, 0, 0)

            # tail != 0 and not align with 32B
            else:
                # multi core handle the segments align with 32B
                # tail is in big loop
                if block_num_needed_last == 0:
                    with self.tik_instance.for_range(0, self.product_core_num,
                                                     thread_num=1,
                                                     block_num=self.product_core_num) as block_idx:
                        data_ub = []
                        for i in range(self.input_tensor_num):
                            data_ub.append(self.tik_instance.Tensor(
                                self.data_dtype[0],
                                (self.max_single_tensor_size_each_core//self.dtype_size,),
                                scope=tik.scope_ubuf, name="data_ubuf_{}".format(i)))
                        # the former loop
                        for loop_block_i in range(loop_block-1):
                            for i in range(self.input_tensor_num):
                                # move input from gm to ub
                                gm_in_offset = ((self.product_core_num * loop_block_i + block_idx) *
                                                self.max_single_tensor_size_each_core) \
                                               // self.dtype_size
                                self.tik_instance.data_move(
                                    data_ub[i][0],
                                    self.data_gm_in[i][gm_in_offset],
                                    0, 1, self.max_single_tensor_size_each_core//32, 0, 0)
                                # move data from ub to gm
                                gm_out_offset = (i * self.single_input_element_num) + gm_in_offset
                                self.tik_instance.data_move(
                                    self.data_gm_out[gm_out_offset],
                                    data_ub[i][0], 0, 1,
                                    self.max_single_tensor_size_each_core//32, 0, 0)
                        # last special loop, last core of the 32 core not full
                        for i in range(self.input_tensor_num):
                            # the former 31 core
                            with self.tik_instance.if_scope(
                                    block_idx != self.product_core_num-1):
                                # move input from gm to ub
                                gm_in_offset = \
                                    ((loop_block-1) * self.product_core_num + block_idx) * \
                                    self.max_single_tensor_size_each_core \
                                    // self.dtype_size
                                self.tik_instance.data_move(
                                    data_ub[i][0],
                                    self.data_gm_in[i][gm_in_offset],
                                    0, 1, self.max_single_tensor_size_each_core//32, 0, 0)
                                # move data from ub to gm
                                gm_out_offset = (i * self.single_input_element_num) + gm_in_offset
                                self.tik_instance.data_move(
                                    self.data_gm_out[gm_out_offset],
                                    data_ub[i][0], 0, 1,
                                    self.max_single_tensor_size_each_core//32, 0, 0)
                    # single core handle the tail
                    data_ub = []
                    for i in range(self.input_tensor_num):
                        data_ub.append(self.tik_instance.Tensor(
                            self.data_dtype[0],
                            (self.max_single_tensor_size_each_core//self.dtype_size,),
                            scope=tik.scope_ubuf, name="data_ubuf_{}".format(i)))
                    for i in range(self.input_tensor_num):
                        burst_len = ceil_align(tail, 32)
                        # move input from gm to ub
                        gm_in_offset = (block_num_needed-1) * \
                                       self.max_single_tensor_size_each_core//self.dtype_size
                        self.tik_instance.data_move(data_ub[i][0],
                                                    self.data_gm_in[i][gm_in_offset],
                                                    0, 1, burst_len//32, 0, 0)
                        # move data from ub to gm
                        gm_out_offset = (i * self.single_input_element_num) + gm_in_offset
                        self.tik_instance.data_move(self.data_gm_out[gm_out_offset],
                                                    data_ub[i][0], 0, 1,
                                                    burst_len//32, 0, 0)
                    for k in range(self.input_tensor_num-1):
                        burst_len = 32
                        # move input from gm to ub
                        gm_in_offset = 0
                        self.tik_instance.data_move(data_ub[k][0],
                                                    self.data_gm_in[k+1][gm_in_offset],
                                                    0, 1, burst_len//32, 0, 0)
                        # move data from ub to gm
                        gm_out_offset = (k+1) * self.single_input_element_num
                        self.tik_instance.data_move(self.data_gm_out[gm_out_offset],
                                                    data_ub[k][0], 0, 1,
                                                    burst_len//32, 0, 0)
                # tail is out of big loop
                else:
                    with self.tik_instance.for_range(0, self.product_core_num,
                                                     thread_num=1,
                                                     block_num=self.product_core_num) as block_idx:
                        data_ub = []
                        for i in range(self.input_tensor_num):
                            data_ub.append(self.tik_instance.Tensor(
                                self.data_dtype[0],
                                (self.max_single_tensor_size_each_core//self.dtype_size,),
                                scope=tik.scope_ubuf, name="data_ubuf_{}".format(i)))
                        # the former loop
                        for loop_block_i in range(loop_block):
                            for i in range(self.input_tensor_num):
                                # move input from gm to ub
                                gm_in_offset = ((self.product_core_num * loop_block_i + block_idx) *
                                                self.max_single_tensor_size_each_core) \
                                               // self.dtype_size
                                self.tik_instance.data_move(
                                    data_ub[i][0],
                                    self.data_gm_in[i][gm_in_offset],
                                    0, 1, self.max_single_tensor_size_each_core//32, 0, 0)
                                # move data from ub to gm
                                gm_out_offset = (i * self.single_input_element_num) + gm_in_offset
                                self.tik_instance.data_move(
                                    self.data_gm_out[gm_out_offset],
                                    data_ub[i][0], 0, 1,
                                    self.max_single_tensor_size_each_core//32, 0, 0)
                        # last special loop, last core of the n(n<32) core not full
                        # the former cores
                        with self.tik_instance.if_scope(block_idx < block_num_needed_last-1):
                            for i in range(self.input_tensor_num):
                                # move input from gm to ub
                                gm_in_offset = (loop_block * self.product_core_num + block_idx) * \
                                               self.max_single_tensor_size_each_core \
                                               // self.dtype_size
                                self.tik_instance.data_move(
                                    data_ub[i][0],
                                    self.data_gm_in[i][gm_in_offset],
                                    0, 1, self.max_single_tensor_size_each_core//32, 0, 0)
                                # move data from ub to gm
                                gm_out_offset = (i * self.single_input_element_num) + gm_in_offset
                                self.tik_instance.data_move(
                                    self.data_gm_out[gm_out_offset],
                                    data_ub[i][0], 0, 1,
                                    self.max_single_tensor_size_each_core//32, 0, 0)
                    # single core handle the tail
                    data_ub = []
                    for i in range(self.input_tensor_num):
                        data_ub.append(self.tik_instance.Tensor(
                            self.data_dtype[0],
                            (self.max_single_tensor_size_each_core//self.dtype_size,),
                            scope=tik.scope_ubuf, name="data_ubuf_{}".format(i)))
                    for i in range(self.input_tensor_num):
                        burst_len = ceil_align(tail, 32)
                        # move input from gm to ub
                        gm_in_offset = (block_num_needed-1) * \
                                       self.max_single_tensor_size_each_core//self.dtype_size
                        self.tik_instance.data_move(data_ub[i][0],
                                                    self.data_gm_in[i][gm_in_offset],
                                                    0, 1, burst_len//32, 0, 0)
                        # move data from ub to gm
                        gm_out_offset = (i * self.single_input_element_num) + gm_in_offset
                        self.tik_instance.data_move(self.data_gm_out[gm_out_offset],
                                                    data_ub[i][0], 0, 1,
                                                    burst_len//32, 0, 0)
                    for k in range(self.input_tensor_num-1):
                        burst_len = 32
                        # move input from gm to ub
                        gm_in_offset = 0
                        self.tik_instance.data_move(data_ub[k][0],
                                                    self.data_gm_in[k+1][gm_in_offset],
                                                    0, 1, burst_len//32, 0, 0)
                        # move data from ub to gm
                        gm_out_offset = (k+1) * self.single_input_element_num
                        self.tik_instance.data_move(self.data_gm_out[gm_out_offset],
                                                    data_ub[k][0], 0, 1,
                                                    burst_len//32, 0, 0)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=self.data_gm_in,
                                   outputs=[self.data_gm_out],
                                   enable_l2=False)
        return self.tik_instance
