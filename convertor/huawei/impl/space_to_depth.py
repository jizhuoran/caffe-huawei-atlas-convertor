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

space_to_depth
"""
from te import tik
from te import platform as tbe_platform
from topi.cce import util
from impl import common_util
from impl import constant_util as constant

# pylint: disable=redefined-builtin
if "reduce" not in dir(__builtins__):
    from functools import reduce

# the minimum value of block_size
BLOCK_SIZE_MIN = 2

# the dim count is four
DIM_CNT_FOUR = 4

# the format of NHWC
NHWC_STR = "NHWC"

# the maximum value of  burst_n
MAX_BURST_N = 4095

# the maximum value of  stride
MAX_STRIDE = 65535

# available ub size: 248KB
UB_SIZE = 248 * 1024


# pylint: disable=invalid-name,unused-argument,too-many-locals
@util.check_input_type(dict, dict, int, str, str)
def space_to_depth(x,
                   y,
                   block_size,
                   data_format="NHWC",
                   kernel_name="space_to_depth"):
    """
    the main function of space_to_depth
    Parameters
    ----------
    x: dict,shape and data type,data type supports float16,float32,int32,
       uint32,int16,uint16,int8,uint8,int64,uint64
    y: dict,shape and data type,data type supports float16,float32,int32,
       uint32,int16,uint16,int8,uint8,int64,uint64
    block_size: must be greater than one. It indicates the block size
    data_format: only support NHWC
    kernel_name: cce kernel name, default value is "space_to_batch"

    Returns
    -------
    tik_instance: tik_instance
    """
    _check_param(x, y, block_size, data_format, kernel_name)
    fun = SpaceToDepth(x, block_size, kernel_name)
    return fun.space_to_depth_compute()


def _check_param(x, y, block_size, data_format, kernel_name):
    """
    check the parameters is valid, if one is invalid,then raise error

    Parameters
    ----------
    x: dict,shape and data type,data type supports float16,float32,int32,
       uint32,int16,uint16,int8,uint8,int64,uint64
    y: dict,shape and data type,data type supports float16,float32,int32,
       uint32,int16,uint16,int8,uint8,int64,uint64
    block_size: must be greater than one. It indicates the block size
    data_format: only support NHWC
    kernel_name: cce kernel name, default value is "space_to_depth"
    Returns
    -------
    None
    """
    if data_format != NHWC_STR:
        raise RuntimeError("data_format only supported NHWC")
    shape = x.get("shape")
    dtype = x.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_dtype_rule(
        dtype, (constant.DATA_TYPE_FP32, constant.DATA_TYPE_FP16,
                constant.DATA_TYPE_INT32, constant.DATA_TYPE_INT16,
                constant.DATA_TYPE_INT8, constant.DATA_TYPE_UINT32,
                constant.DATA_TYPE_UINT64, constant.DATA_TYPE_INT64,
                constant.DATA_TYPE_UINT16, constant.DATA_TYPE_UINT8))

    if len(shape) != DIM_CNT_FOUR:
        raise RuntimeError("The input x only supported 4D")

    if block_size < BLOCK_SIZE_MIN:
        raise RuntimeError("the attr block_size must be greater than one")

    if shape[1] % block_size != 0 or shape[2] % block_size != 0:
        raise RuntimeError(
            "both height and width must be divisible by block_size")
    output_shape = (shape[0], shape[1] // block_size, shape[2] // block_size,
                    shape[3] * block_size * block_size)
    util.check_shape_rule(output_shape)
    util.check_tensor_shape_size(output_shape)


class SpaceToDepthBase:
    """
    Function: use to store space_to_depth base parameters
    Modify : 2019-10-28
    """
    def __init__(self, input_data, block_size, kernel_name):
        """
        init space_to_depth base parameters
        Parameters
        ----------
        input_data: shape and data type,data type supports float16,float32,
                    int32,uint32,int16,uint16,int8,uint8,int64,uint64
        block_size: must be greater than one. It indicates the block size
        kernel_name: cce kernel name, default value is "space_to_depth"
        """
        self.input_shape = input_data.get("shape")
        self.dtype = input_data.get("dtype").lower()
        self.dtype_size = common_util.get_data_size(self.dtype)
        self.block_size = block_size
        self.tik_instance = tik.Tik()
        self.kernel_name = kernel_name
        self.output_shape = (self.input_shape[0],
                             self.input_shape[1] // block_size,
                             self.input_shape[2] // block_size,
                             self.input_shape[3] * block_size * block_size)

    def get_input_size(self, start=0, end=4):
        """
        get the size of input data

        Parameters
        ----------
        start: the start dim used to calculate the size of input data
        end: the end dim used to calculate the size of input data
        Returns
        -------
        size: the size of input data
        """
        return reduce(lambda x1, x2: x1 * x2, self.input_shape[start:end])

    def get_output_size(self, start=0, end=4):
        """
        get the size of output data

        Parameters
        ----------
        start: the start dim used to calculate the size of output data
        end: the end dim used to calculate the size of output data
        Returns
        -------
        size: the size of output data
        """
        return reduce(lambda x1, x2: x1 * x2, self.output_shape[start:end])


class SpaceToDepthComputeParam:
    """
    Function: use to store space_to_depth compute parameters
    Modify : 2019-10-28
    """
    def __init__(self, n_dim, h_dim, w_dim, bs_info):
        """
        init space_to_depth dim parameters
        Parameters
        ----------
        n_dim: the dim of n
        h_dim: the dim of h
        w_dim: the dim of w
        bs_info: the dim of block_size
        """
        self.n_dim = n_dim
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.bs_h_dim = bs_info[0]
        self.ub_dim = bs_info[1]

    def get_n_dim(self):
        """
        get the dim of n
        Returns
        -------
        n_dim: the dim of n
        """
        return self.n_dim

    def get_h_dim(self):
        """
        get the dim of h
        Returns
        -------
        h_dim: the dim of h
        """
        return self.h_dim


class SpaceToDepth(SpaceToDepthBase):
    """
    Function: use to store space_to_depth compute parameters
    Modify : 2019-10-28
    """
    def __init__(self, input_data, block_size, kernel_name):
        """
        init space_to_depth base parameters
        Parameters
        ----------
        input_data: shape and data type,data type supports float16,float32,
                    int32,uint32,int16,uint16,int8,uint8,int64,uint64
        block_size: must be greater than one. It indicates the block size
        kernel_name: cce kernel name, default value is "space_to_batch"
        """
        super(SpaceToDepth, self).__init__(input_data, block_size, kernel_name)
        b_size = constant.BLOCK_SIZE
        self.element_size = b_size // self.dtype_size
        self.bs_c_greater_32 = bool(
            self.block_size * self.input_shape[3] * self.dtype_size > b_size)
        self.data_gm_out = self.tik_instance.Tensor(
            self.dtype, (self.get_output_size() + b_size,),
            scope=tik.scope_gm,
            name="data_gm_out")
        self.data_gm_in = self.tik_instance.Tensor(
            self.dtype, (self.get_input_size() + b_size,),
            scope=tik.scope_gm,
            name="data_gm_in")
        self.tiling_shape = (self.input_shape[0],
                             self.input_shape[1] // self.block_size,
                             self.input_shape[2] // self.block_size,
                             self.block_size, self.get_bs_c_align_size())
        self.tiling_case, self.buf_size = self.get_buf_info()

    def get_tiling_shape_size(self, start=0, end=5):
        """
        get the tiling shape size
        Parameters
        ----------
        start: the start dim
        end: the end dim

        Returns
        -------
        size: the size of [start:end]
        """
        if start == end:
            size = 1
        else:
            size = reduce(lambda x1, x2: x1 * x2, self.tiling_shape[start:end])
        return size

    def get_bs_c_align_size(self):
        """
        get the align size of block_size * c
        Returns
        -------
        size: the align size of block_size * c
        """
        bs_c_align_flag = bool((self.block_size * self.input_shape[3]) %
                               self.element_size == 0)
        if bs_c_align_flag:
            size = self.input_shape[3] * self.block_size
        else:
            size = (self.input_shape[3] * self.block_size +
                    self.element_size - 1) // \
                   self.element_size * self.element_size
        return size

    def get_bs_c(self):
        """
        get the size of block_size * c
        Returns
        -------
        size: the size of block_size * c
        """
        return self.input_shape[3] * self.block_size

    def get_buf_info(self):
        """
        get the ub buffer size
        Returns
        -------
        tiling_case: the index of tiling shape
        buf_size: the ub buffer size
        """
        size_for_tail = 32
        all_ub_buffer = UB_SIZE // self.dtype_size - size_for_tail
        buf_size = all_ub_buffer
        tiling_case = len(self.tiling_shape)
        for i in range(0, len(self.tiling_shape)):
            buf_size_needed = reduce(lambda x1, x2: x1 * x2,
                                     self.tiling_shape[i:])
            is_buffer_large_enough = all_ub_buffer // buf_size_needed
            if is_buffer_large_enough > 0:
                tiling_case = i
                buf_size = is_buffer_large_enough * buf_size_needed
                break
        if tiling_case == 0:
            buf_size = self.get_tiling_shape_size(0, 5)
        if tiling_case == 4:
            buf_size_needed = reduce(lambda x1, x2: x1 * x2,
                                     self.tiling_shape[tiling_case:])
            buf_num = buf_size // \
                      reduce(lambda x1, x2: x1 * x2,
                             self.tiling_shape[tiling_case:5])
            buf_size = buf_size_needed
            for i in range(buf_num, 0, -1):
                if self.block_size % i == 0:
                    buf_size = i * buf_size_needed
                    break
        return tiling_case, buf_size

    def move_tail_less_32_loop(self, loop_info, index_info, tensor,
                               offset_flag):
        """
        describe the loop calculation process when move tail less 32
        """
        loop_n = loop_info[0]
        loop_r = loop_info[1]
        burst_n = loop_info[2]
        ub_index = index_info[0]
        ub_offset = index_info[2]
        if offset_flag:
            loop = loop_n
        else:
            loop = loop_n - burst_n
        with self.tik_instance.for_range(0, loop) as l_j:
            with self.tik_instance.for_range(0,
                                             self.get_bs_c()) as l_i:
                if offset_flag:
                    index = ub_index + \
                            (burst_n - 1 - loop_n + 1 + l_j) * ub_offset + l_i
                else:
                    index = (index_info[1] - loop_n + burst_n) * \
                            ub_offset + l_i
                tensor[1][loop_r + l_j * self.get_bs_c() + l_i].set_as(
                    tensor[0][index])
        if loop_r > 0:
            with self.tik_instance.for_range(0, loop_r) as l_k:
                if offset_flag:
                    index = ub_index + (burst_n - 1 - loop_n) * ub_offset + \
                            self.get_bs_c() - loop_r + l_k
                else:
                    index = (index_info[1] - loop_n + burst_n - 1) * \
                            ub_offset + self.get_bs_c() - loop_r + l_k
                tensor[1][l_k].set_as(tensor[0][index])

    def move_tail_less_32(self, data_ub, index_info, burst_info, offset_info):
        """
        block_size * c  is not aligned with the 32B, but less than 32B
        Parameters
        ----------
        data_ub: the ub tensor
        index_info: the index of ub and gm
        burst_info: the burst n and the burst length
        offset_info: the offset of ub and gm

        Returns
        -------
        None
        """
        ub_index = index_info[0]
        out_index = index_info[1]
        burst_n = burst_info[0]
        burst_len = burst_info[1]
        ub_offset = offset_info[0]
        out_offset = offset_info[1]
        block_num = self.get_block_num()
        offset = self.element_size - self.get_bs_c()
        offset_align = self.tik_instance.Scalar(
            dtype=constant.DATA_TYPE_UINT64, name="offset_align")
        offset_align.set_as((offset + self.get_bs_c() - 1) // self.get_bs_c())
        buf_num = self.buf_size // \
                  reduce(lambda x1, x2: x1 * x2,
                         self.tiling_shape[self.tiling_case:
                                           len(self.tiling_shape)])
        loop_n = self.tik_instance.Scalar(dtype=constant.DATA_TYPE_UINT64,
                                          name="loop_n")
        loop_n.set_as(self.element_size // self.get_bs_c())
        loop_r = self.element_size % self.get_bs_c()
        if block_num == 1:
            with self.tik_instance.for_range(0, burst_n) as n_i:
                self.tik_instance.tensor_mov(
                    self.data_gm_out[out_index + n_i * out_offset],
                    data_ub[ub_index + n_i * ub_offset], "",
                    constant.DEFAULT_NBURST, burst_len, constant.STRIDE_ZERO,
                    constant.STRIDE_ZERO)
        else:
            data_ub1 = self.tik_instance.Tensor(self.dtype,
                                                (self.element_size,),
                                                scope=tik.scope_ubuf,
                                                name="data_ub1")
            with self.tik_instance.if_scope(offset_align < burst_n):
                with self.tik_instance.for_range(0, burst_info[0] -
                                                 offset_align) as n_i:
                    self.tik_instance.tensor_mov(
                        self.data_gm_out[out_index + n_i * out_offset],
                        data_ub[ub_index + n_i * ub_offset], "", 1, burst_len,
                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                self.move_tail_less_32_loop((loop_n, loop_r, burst_n),
                                            (ub_index, buf_num, ub_offset),
                                            (data_ub, data_ub1),
                                            True)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, burst_n) as l_j:
                    with self.tik_instance.for_range(0,
                                                     self.get_bs_c()) as l_i:
                        data_ub1[loop_r +
                                 (loop_n - burst_n) * self.get_bs_c() +
                                 l_j * self.get_bs_c() + l_i].set_as(
                                     data_ub[l_j * ub_offset + l_i])
                self.move_tail_less_32_loop((loop_n, loop_r, burst_n),
                                            (ub_index, buf_num, ub_offset),
                                            (data_ub, data_ub1),
                                            False)
            self.tik_instance.tensor_mov(
                self.data_gm_out[out_index +
                                 (burst_n - 1) * self.get_bs_c() - offset],
                data_ub1, "", constant.DEFAULT_NBURST,
                constant.DEFAULT_BURST_LEN, constant.STRIDE_ZERO,
                constant.STRIDE_ZERO)

    def move_tail_great_32(self, data_ub, index_info, burst_info, offset_info):
        """
        block_size * c  is not aligned with the 32B, but greater than 32B
        Parameters
        ----------
        data_ub: the ub tensor
        index_info: the index of ub and gm
        burst_info: the burst n and the burst length
        offset_info: the offset of ub and gm

        Returns
        -------
        None
        """
        ub_index = index_info[0]
        out_index = index_info[1]
        burst_len = burst_info[1]
        ub_offset = offset_info[0]
        with self.tik_instance.for_range(0, burst_info[0]) as n_i:
            with self.tik_instance.if_scope(
                    tik.any(n_i < burst_info[0] - 1,
                            self.get_block_num() == 1)):
                self.tik_instance.tensor_mov(
                    self.data_gm_out[out_index + n_i * offset_info[1]],
                    data_ub[ub_index + n_i * ub_offset], "",
                    constant.DEFAULT_NBURST, burst_len, constant.STRIDE_ZERO,
                    constant.STRIDE_ZERO)
            with self.tik_instance.else_scope():
                loop_n = self.get_bs_c() // self.element_size
                loop_r = self.get_bs_c() % self.element_size
                offset = self.element_size - loop_r
                self.tik_instance.tensor_mov(
                    self.data_gm_out[out_index + n_i * offset_info[1]],
                    data_ub[ub_index + n_i * ub_offset], "", 1, loop_n,
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                if loop_r > 0:
                    data_ub1 = self.tik_instance.Tensor(self.dtype,
                                                        (self.element_size,),
                                                        scope=tik.scope_ubuf,
                                                        name="data_ub1")
                    with self.tik_instance.for_range(0,
                                                     self.element_size) as i:
                        data_ub1[i].set_as(
                            data_ub[ub_index + n_i * ub_offset +
                                    loop_n * self.element_size - offset + i])
                    self.tik_instance.tensor_mov(
                        self.data_gm_out[out_index + n_i * offset_info[1] +
                                         loop_n * self.element_size - offset],
                        data_ub1, "", constant.DEFAULT_NBURST,
                        constant.DEFAULT_BURST_LEN, constant.STRIDE_ZERO,
                        constant.STRIDE_ZERO)

    def move_ub_to_gm(self, data_ub, index_info, burst_info, offset_info):
        """
        move the data from ub to gm
        Parameters
        ----------
        data_ub: the ub tensor
        index_info: the index of ub and gm
        burst_info: the burst n and the burst length
        offset_info: the offset of ub and gm

        Returns
        -------
        None
        """
        ub_index = index_info[0]
        out_index = index_info[1]
        burst_n = burst_info[0]
        burst_len = burst_info[1]
        bs_c_align_flag = bool(
            (self.block_size * self.input_shape[3] * self.dtype_size) %
            constant.BLOCK_SIZE == 0)
        if self.tiling_case == 5 or bs_c_align_flag:
            loop_n = self.tik_instance.Scalar(dtype=constant.DATA_TYPE_UINT64,
                                              name="ub_to_gm_loop_n")
            loop_r = self.tik_instance.Scalar(dtype=constant.DATA_TYPE_UINT64,
                                              name="ub_to_gm_loop_r")
            loop_n.set_as(burst_n // MAX_BURST_N)
            loop_r.set_as(burst_n % MAX_BURST_N)
            num = MAX_BURST_N * burst_len * self.element_size
            with self.tik_instance.for_range(0, loop_n) as i:
                self.tik_instance.tensor_mov(
                    self.data_gm_out[out_index + num * i],
                    data_ub[ub_index + num * i], "", MAX_BURST_N, burst_len,
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            with self.tik_instance.if_scope(loop_r > 0):
                self.tik_instance.tensor_mov(
                    self.data_gm_out[out_index + num * loop_n],
                    data_ub[ub_index + num * loop_n], "", loop_r, burst_len,
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        else:
            if self.bs_c_greater_32:
                self.move_tail_great_32(data_ub, index_info, burst_info,
                                        offset_info)
            else:
                self.move_tail_less_32(data_ub, index_info, burst_info,
                                       offset_info)

    def move_gm_to_ub(self, data_ub, index_info, burst_info, offset_info):
        """
        move data from gm to ub
        Parameters
        ----------
        data_ub: the ub tensor
        index_info: the index of ub and gm
        burst_info: burst n and burst length
        offset_info: the offset of ub and gm

        Returns
        -------
        None
        """
        in_index = index_info[1]
        burst_n = burst_info[0]
        burst_len = burst_info[1]
        bs_c_align_flag = bool(
            (self.block_size * self.input_shape[3] * self.dtype_size) %
            constant.BLOCK_SIZE == 0)
        if burst_info[2] > MAX_STRIDE or not bs_c_align_flag:
            with self.tik_instance.for_range(0, burst_n) as n_i:
                self.tik_instance.tensor_mov(
                    data_ub[index_info[0] + n_i * offset_info[0]],
                    self.data_gm_in[in_index + n_i * offset_info[1]], "", 1,
                    burst_len, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        else:
            loop_n = self.tik_instance.Scalar(dtype=constant.DATA_TYPE_UINT64,
                                              name="gm_to_ub_loop_n")
            loop_r = self.tik_instance.Scalar(dtype=constant.DATA_TYPE_UINT64,
                                              name="gm_to_ub_loop_r")
            loop_n.set_as(burst_n // MAX_BURST_N)
            loop_r.set_as(burst_n % MAX_BURST_N)
            ub_num = MAX_BURST_N * burst_len * self.element_size
            src_num = MAX_BURST_N * self.input_shape[2] * self.input_shape[3]
            with self.tik_instance.for_range(0, loop_n) as i:
                self.tik_instance.tensor_mov(
                    data_ub[index_info[0] + ub_num * i],
                    self.data_gm_in[in_index + src_num * i], "", MAX_BURST_N,
                    burst_len, constant.STRIDE_ZERO, burst_info[2])
            with self.tik_instance.if_scope(loop_r > 0):
                self.tik_instance.tensor_mov(
                    data_ub[index_info[0] + ub_num * loop_n],
                    self.data_gm_in[in_index + src_num * loop_n], "", loop_r,
                    burst_len, constant.STRIDE_ZERO, burst_info[2])

    def move_data_tiling_5(self, data_ub, in_index, block_i, param):
        """
        move data from input to output when tiling case is 5
        """
        self.move_gm_to_ub(
            data_ub, (0, in_index),
            (1, self.buf_size * self.dtype_size // constant.BLOCK_SIZE, 0),
            (0, 0))
        self.move_ub_to_gm(
            data_ub, (0, self.get_out_index(block_i, param)),
            (1, self.buf_size * self.dtype_size // constant.BLOCK_SIZE, 0),
            (0, 0))
        with self.tik_instance.if_scope(
                param.ub_dim == self.get_bs_c() // self.buf_size - 1):
            last_n = (self.get_bs_c() % self.buf_size) // self.element_size
            last_r = (self.get_bs_c() % self.buf_size) % self.element_size
            if last_n > 0:
                self.move_gm_to_ub(data_ub, (0, in_index + self.buf_size),
                                   (1, last_n, 0), (0, 0))
                self.move_ub_to_gm(data_ub,
                                   (0, self.get_out_index(block_i, param) +
                                    self.buf_size), (1, last_n, 0), (0, 0))
            if last_r > 0:
                offset = self.element_size - last_r
                data_ub1 = self.tik_instance.Tensor(self.dtype,
                                                    (self.element_size,),
                                                    scope=tik.scope_ubuf,
                                                    name="data_ub1")
                if last_n > 0:
                    with self.tik_instance.for_range(0, offset) as i:
                        data_ub1[offset - 1 - i].set_as(
                            data_ub[last_n * self.element_size - 1 - i])
                else:
                    with self.tik_instance.for_range(0, offset) as i:
                        data_ub1[offset - 1 - i].set_as(
                            data_ub[self.buf_size - 1 - i])
                self.move_gm_to_ub(data_ub, (0, in_index + self.buf_size +
                                             last_n * self.element_size),
                                   (1, 1, 0), (0, 0))
                with self.tik_instance.for_range(0, last_r) as j:
                    data_ub1[offset + j].set_as(data_ub[j])
                self.move_ub_to_gm(
                    data_ub1,
                    (0, self.get_out_index(block_i, param) +
                     self.buf_size + last_n * self.element_size - offset),
                    (1, 1, 0), (0, 0))

    def move_data(self, block_num, block_i, loop_info, data_ub):
        """
        move data from input to output
        Parameters
        ----------
        block_num: all block number
        block_i: the current block number
        loop_info: the all loop number and the current loop number
        data_ub: the ub tensor

        Returns
        -------
        None
        """
        param = self.get_compute_param(block_i, loop_info[0])
        in_index = param.n_dim * self.get_input_size(1) \
                   + param.h_dim * self.block_size * self.get_input_size(2) \
                   + param.w_dim * self.block_size * self.get_input_size(3) \
                   + param.bs_h_dim * self.get_input_size(2) \
                   + param.ub_dim * self.buf_size
        src_stride = (self.input_shape[2] - self.block_size) * \
                     self.input_shape[3] * self.dtype_size // \
                     constant.BLOCK_SIZE
        if self.tiling_case == 0:
            self.move_gm_to_ub(
                data_ub,
                (loop_info[0] * self.get_tiling_shape_size(3, 5), in_index),
                (self.block_size, self.get_bs_c_align_size() *
                 self.dtype_size // constant.BLOCK_SIZE, src_stride),
                (self.get_bs_c_align_size(), self.get_input_size(2)))
            with self.tik_instance.if_scope(loop_info[0] == loop_info[1] - 1):
                self.move_ub_to_gm(
                    data_ub, (0, 0),
                    (self.get_tiling_shape_size(
                        self.tiling_case, 4), self.get_bs_c_align_size() *
                     self.dtype_size // constant.BLOCK_SIZE, 0),
                    (self.get_bs_c_align_size(),
                     self.block_size * self.input_shape[3]))
        elif self.tiling_case == 5:
            self.move_data_tiling_5(data_ub, in_index, block_i, param)
        elif self.tiling_case == 4:
            buf_num = self.buf_size // \
                      reduce(lambda x1, x2: x1 * x2,
                             self.tiling_shape[self.tiling_case:
                                               len(self.tiling_shape)])
            self.move_gm_to_ub(
                data_ub, (0, in_index),
                (buf_num, self.get_bs_c_align_size() * self.dtype_size //
                 constant.BLOCK_SIZE, src_stride),
                (self.get_bs_c_align_size(), self.get_input_size(2)))
            self.move_ub_to_gm(data_ub,
                               (0, self.get_out_index(block_i, param)),
                               (buf_num, self.get_bs_c_align_size() *
                                self.dtype_size // constant.BLOCK_SIZE, 0),
                               (self.get_bs_c_align_size(),
                                self.block_size * self.input_shape[3]))
        else:
            buf_num = self.buf_size // \
                      self.get_tiling_shape_size(self.tiling_case,
                                                 len(self.tiling_shape))
            loop_num = buf_num * self.get_tiling_shape_size(
                self.tiling_case, 3)
            self.move_gm_to_ub(
                data_ub,
                ((loop_info[0] % loop_num) * self.get_bs_c_align_size() *
                 self.block_size, in_index),
                (self.block_size, self.get_bs_c_align_size() *
                 self.dtype_size // constant.BLOCK_SIZE, src_stride),
                (self.get_bs_c_align_size(), self.get_input_size(2)))
            with self.tik_instance.if_scope(block_i < block_num - 1):
                with self.tik_instance.if_scope(loop_info[0] == loop_info[1] -
                                                1):
                    self.move_ub_to_gm(
                        data_ub, (0, self.get_out_index(block_i, param)),
                        (loop_num * self.block_size,
                         self.get_bs_c_align_size() * self.dtype_size //
                         constant.BLOCK_SIZE, 0),
                        (self.get_bs_c_align_size(),
                         self.block_size * self.input_shape[3]))
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(loop_info[0] == loop_num - 1):
                    self.move_ub_to_gm(
                        data_ub, (0, self.get_out_index(block_i, param)),
                        (loop_num * self.block_size,
                         self.get_bs_c_align_size() * self.dtype_size //
                         constant.BLOCK_SIZE, 0),
                        (self.get_bs_c_align_size(),
                         self.block_size * self.input_shape[3]))
                with self.tik_instance.if_scope(loop_info[1] > loop_num):
                    with self.tik_instance.if_scope(
                            loop_info[0] == loop_info[1] - 1):
                        self.move_ub_to_gm(
                            data_ub, (0, self.get_out_index(block_num, param)),
                            ((loop_info[1] - loop_num) * self.block_size,
                             self.get_bs_c_align_size() * self.dtype_size //
                             constant.BLOCK_SIZE, 0),
                            (self.get_bs_c_align_size(),
                             self.block_size * self.input_shape[3]))

    def compute_on_each_core(self, block_num, block_i):
        """
        calculated on each core
        Parameters
        ----------
        block_num: all block number
        block_i: the current block number

        Returns
        -------
        None
        """
        loop = self.get_loop_num(block_num, block_i)
        data_ub = self.tik_instance.Tensor(self.dtype, (self.buf_size,),
                                           scope=tik.scope_ubuf,
                                           name="data_ub")
        with self.tik_instance.for_range(0, loop) as loop_i:
            self.move_data(block_num, block_i, (loop_i, loop), data_ub)

    def get_out_index(self, block_i, param):
        """
        get the index of gm for move data from ub to gm
        Parameters
        ----------
        block_i: the current block
        param:the dim info of n,h,w,bs_h,ub_dim

        Returns
        -------
        index: the index of gm for move data from ub to gm
        """
        n_dim = 0
        h_dim = 0
        w_dim = 0
        bs_h_dim = 0
        ub_dim = 0
        buf_num = self.buf_size // \
                  self.get_tiling_shape_size(self.tiling_case,
                                             len(self.tiling_shape))
        if self.tiling_case == 1:
            index = block_i * buf_num * self.get_tiling_shape_size(1, 3)
            n_dim = index // self.get_tiling_shape_size(1, 3)
            h_dim = index % self.get_tiling_shape_size(1, 3) // \
                    self.output_shape[2]
            w_dim = index % self.get_tiling_shape_size(1, 3) % \
                    self.output_shape[2]
        elif self.tiling_case == 2:
            index = block_i * buf_num * self.output_shape[2]
            n_dim = index // self.get_tiling_shape_size(1, 3)
            h_dim = index % self.get_tiling_shape_size(1, 3) // \
                    self.output_shape[2]
            w_dim = index % self.get_tiling_shape_size(1, 3) % \
                    self.output_shape[2]
        elif self.tiling_case == 3:
            n_dim = (block_i * buf_num) // self.get_output_size(1, 3)
            h_dim = (block_i * buf_num) % self.get_output_size(1, 3) // \
                    self.output_shape[2]
            w_dim = (block_i * buf_num) % self.get_output_size(1, 3) % \
                    self.output_shape[2]
        elif self.tiling_case == 4:
            n_dim = param.get_n_dim()
            h_dim = param.get_h_dim()
            w_dim = param.w_dim
            bs_h_dim = param.bs_h_dim
        elif self.tiling_case == 5:
            n_dim = param.n_dim
            h_dim = param.h_dim
            w_dim = param.w_dim
            bs_h_dim = param.bs_h_dim
            ub_dim = param.ub_dim
        return n_dim * self.get_output_size(1) + \
               h_dim * self.get_output_size(2) + \
               w_dim * self.get_output_size(3) + \
               bs_h_dim * self.block_size * self.input_shape[3] + \
               ub_dim * self.buf_size

    def get_compute_param(self, block_i, loop_i):
        """
        get the dim info
        Parameters
        ----------
        block_i: the current block
        loop_i: the current loop

        Returns
        -------
        param: the dim info of n,h,w,bs_h,ub_dim
        """
        bs_h_dim = 0
        ub_dim = 0
        n_dim = 0
        h_dim = 0
        w_dim = 0
        if self.tiling_case == 0:
            n_dim = loop_i // self.get_tiling_shape_size(1, 3)
            h_dim = loop_i % self.get_tiling_shape_size(1, 3) // \
                    self.output_shape[2]
            w_dim = loop_i % self.get_tiling_shape_size(1, 3) % \
                    self.output_shape[2]
        elif self.tiling_case == 5:
            num = self.get_bs_c() // self.buf_size
            h_w_b_n = self.get_tiling_shape_size(1, 4) * num
            w_b_n = self.get_tiling_shape_size(2, 4) * num
            b_n = self.block_size * num
            n_dim = block_i // h_w_b_n
            h_dim = block_i % h_w_b_n // w_b_n
            w_dim = block_i % h_w_b_n % w_b_n // b_n
            bs_h_dim = block_i % h_w_b_n % w_b_n % b_n // num
            ub_dim = block_i % h_w_b_n % w_b_n % b_n % num
        else:
            buf_num = self.buf_size // \
                      self.get_tiling_shape_size(self.tiling_case,
                                                 len(self.tiling_shape))
            if self.tiling_case == 1:
                index = block_i * buf_num * \
                        self.get_tiling_shape_size(1, 3) + loop_i
                n_dim = index // self.get_tiling_shape_size(1, 3)
                h_dim = index % self.get_tiling_shape_size(1, 3) // \
                        self.output_shape[2]
                w_dim = index % self.get_tiling_shape_size(1, 3) % \
                        self.output_shape[2]
            elif self.tiling_case == 2:
                index = block_i * buf_num * self.output_shape[2] + loop_i
                n_dim = index // self.get_tiling_shape_size(1, 3)
                h_dim = index % self.get_tiling_shape_size(1, 3) // \
                        self.output_shape[2]
                w_dim = index % self.get_tiling_shape_size(1, 3) % \
                        self.output_shape[2]
            elif self.tiling_case == 3:
                index = block_i * buf_num + loop_i
                n_dim = index // self.get_tiling_shape_size(1, 3)
                h_dim = index % self.get_tiling_shape_size(1, 3) // \
                        self.output_shape[2]
                w_dim = index % self.get_tiling_shape_size(1, 3) % \
                        self.output_shape[2]
            elif self.tiling_case == 4:
                index = block_i * buf_num
                n_dim = index // self.get_tiling_shape_size(1, 4)
                h_dim = index % self.get_tiling_shape_size(1, 4) \
                        // self.get_tiling_shape_size(2, 4)
                w_dim = index % self.get_tiling_shape_size(1, 4) \
                        % self.get_tiling_shape_size(2, 4) // self.block_size
                bs_h_dim = index % self.get_tiling_shape_size(1, 4) \
                           % self.get_tiling_shape_size(2, 4) % self.block_size
        param = SpaceToDepthComputeParam(n_dim, h_dim, w_dim,
                                         (bs_h_dim, ub_dim))
        return param

    def get_loop_num(self, block_num, block_i):
        """
        get the loop number
        Parameters
        ----------
        block_num: all block number
        block_i: cur block number

        Returns
        -------
        num: the number of loop
        """
        if self.tiling_case == 0:
            loop_num = self.get_output_size(self.tiling_case, 3)
        elif self.tiling_case == 4 or self.tiling_case == 5:
            loop_num = 1
        else:
            buf_num = self.buf_size // \
                      self.get_tiling_shape_size(self.tiling_case,
                                                 len(self.tiling_shape))
            last_num = self.get_tiling_shape_size(
                0, self.tiling_case) - block_num * buf_num
            last_flag = (block_i + 1) // block_num
            loop_num = (buf_num + last_flag * last_num) * \
                       self.get_tiling_shape_size(self.tiling_case, 3)
        return loop_num

    def get_block_num(self):
        """
        get the block number
        Returns
        -------
        num: the block number
        """
        if self.tiling_case == 0:
            block_num = 1
        elif self.tiling_case == 5:
            num = (self.input_shape[3] * self.block_size) // self.buf_size
            block_num = self.get_tiling_shape_size(0, 4) * num
        else:
            num = self.get_tiling_shape_size(0, len(self.tiling_shape))
            block_num = num // self.buf_size
        return block_num

    def space_to_depth_compute(self):
        """
        describe the SpaceToDepth calculation process
        Returns
        -------
        tik_instance: tik_instance
        """
        all_block_num = self.get_block_num()
        block_dim = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        if all_block_num < block_dim:
            with self.tik_instance.for_range(
                    0, all_block_num, block_num=all_block_num) as block_i:
                self.compute_on_each_core(all_block_num, block_i)
        else:
            kernel_loop = all_block_num // block_dim
            kernel_r = all_block_num % block_dim
            with self.tik_instance.for_range(0, block_dim,
                                             block_num=block_dim) as block_i:
                if kernel_loop > 0:
                    with self.tik_instance.for_range(0, kernel_loop) as loop_i:
                        self.compute_on_each_core(
                            all_block_num, loop_i + kernel_loop * block_i)
                with self.tik_instance.if_scope(block_i < kernel_r):
                    self.compute_on_each_core(
                        all_block_num, kernel_loop * block_dim + block_i)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.data_gm_in],
                                   outputs=[self.data_gm_out],
                                   enable_l2=False)
        return self.tik_instance
