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

arg_max_with_value
"""
from te import tik
from te import platform as tbe_platform
from topi.cce import util

# define a scalar, value = -(2**16 - 1)
SCALAR_MIN_FP16 = -(2 ** 16 - 1)
# define a scalar, value = -(2**32 - 1)
SCALAR_MIN_FP32 = -(2 ** 31 - 1)
# max set_mask_int64 value
MAX_MASK_INT64 = 2 ** 64 - 1
# max segment len
MAX_SEGMENT_LEN = 2048 * 4
# int32 num in 8*block
OUT_MASK = 64
# max int32 output num
OUT_MAX_NUM = 2048 * 4
# 0101 mask value
MASK_0_1 = 6148914691236517205


def _get_ceil_int(int1, int2):
    """Get Ceil Int

    Parameters
    ----------
    int1: int
        input int 1
    int2: int
        input int 2

    Returns
    -------
    ceil_int: int
    """
    _result = int1 // int2
    if int1 % int2 == 0:
        ceil_int = _result
    else:
        ceil_int = _result + 1

    return ceil_int


# pylint: disable=unused-argument,invalid-name,useless-object-inheritance
@util.check_input_type(dict, dict, dict, int, str)
def arg_max_with_value(x, index, value, dimension,
                       kernel_name="arg_max_with_value"):
    """
    Generate arg_max_with_value operator use arg_max_with_value

    Parameters
    ----------
    x: dict
        data of input.
        source data type, support "float16", "float32"
    index: dict
        index of output.
    value: dict
        value of output.
    dimension: int
        the axis value for reverse
    kernel_name: str
        kernel name, default value is "reverse_ext2"

    Returns
    -------
    tik_instance
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    axis_list = dimension

    _param_check(shape_x, dtype_x, axis_list, kernel_name)
    max_index = Argmax(shape_x, dtype_x, axis_list, kernel_name)

    tik_instance = max_index.argmax_compute()

    return tik_instance


def _param_check(shape_x, dtype_x, axis, kernel_name):
    """check param

    Parameters
    ----------
    shape_x: list
        input shape
    dtype_x: str
        input dtype
    axis: int
        axis int num
    kernel_name: str
        kernel_name string

    Returns
    -------
    None
    """
    util.check_shape_rule(shape_x, max_dim=8)
    util.check_tensor_shape_size(shape_x)
    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype_x.lower(), check_list)
    axis = util.axis_check(len(shape_x), axis)
    util.check_kernel_name(kernel_name)


class ArgmaxBase(object):
    """
       Function: use to store arg_max_with_value base parameters
    """

    def __init__(self, shape_x, dtype_x, axis, kernel_name):
        """
        init argmaxwithvalue base parameters

        Parameters
        ----------
        shape_x: list
            shape of input x
        dtype_x: str
            dtype_x of input x
        axis: int
            process axis
        kernel_name: str
            kernel_name

        Returns
        -------
        None
        """
        self.tik_instance = None
        self.product_core_num = 0
        self.shape_x = list(shape_x)
        self.dtype_x = dtype_x
        self.axis = axis
        self.kernel_name = kernel_name
        self.set_tik_product()

    def get_instance(self):
        """
        init arg_max_with_value parameters

        Parameters
        ----------
        None
        Returns
        -------
        tik_instance: tik_instance
        """
        return self.tik_instance

    def set_tik_product(self):
        """
        init arg_max_with_value  parameters

        Parameters
        ----------
        None
        Returns
        -------
        tik_instance: tik_instance
        """
        self.product_core_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        self.tik_instance = tik.Tik()

# pylint: disable=too-many-instance-attributes
class Argmax(ArgmaxBase):
    """
       Function: use to store arg_max_with_value schedule parameters
    """

    def __init__(self, shape_x, dtype_x, axis, kernel_name):
        """
        init argmaxwithvalue base parameters

        Parameters
        ----------
        shape_x: list
            shape of input x
        dtype_x: str
            dtype_x of input x
        axis: int
            process axis
        kernel_name: str
            kernel_name

        Returns
        -------
        None
        """
        self.result_gm_value = None
        self.result_gm = None
        self.ub_result_int32 = None
        self.ub_result_value = None
        self.result_int32 = None
        self.result_float32 = None
        self.result_out_scalar = None
        super(Argmax, self).__init__(shape_x, dtype_x, axis, kernel_name)
        self.dtype_x = dtype_x
        dtype_bytes_size = 2 if dtype_x == "float16" else 4
        self.data_each_block = 32 // dtype_bytes_size
        self.data_each_vector = self.data_each_block * 8
        shape_len = len(shape_x)
        axis = axis % shape_len
        # To initialize the data.
        self.argmax_axis = axis
        self.first_dim_size = 1
        self.last_dim_size = 1
        self.axis_size = 1
        self.gm_result_size = 0
        self.full_mask = self.data_each_vector

        self.segment = MAX_SEGMENT_LEN
        self.out_mask = OUT_MASK

        self.c_align_ubsize = shape_x[-1]
        if axis < len(self.shape_x) - 1:
            i = 0
            while i < axis:
                self.first_dim_size = self.first_dim_size * shape_x[i]
                i = i + 1
            self.axis_size = shape_x[axis]
            i = axis + 1
            while i < len(shape_x):
                self.last_dim_size = self.last_dim_size * shape_x[i]
                i = i + 1
            self.gm_result_size = self.first_dim_size * self.last_dim_size
            self.repeat_times = \
                (self.last_dim_size * dtype_bytes_size + 255) // 256

        else:
            i = 0
            while i < len(shape_x) - 1:
                self.first_dim_size = self.first_dim_size * shape_x[i]
                i = i + 1
            self.axis_size = shape_x[axis]
            self.repeat_times = \
                (self.axis_size * dtype_bytes_size + 255) // 256
            self.gm_result_size = \
                self.first_dim_size + 2 * self.repeat_times + 15

        self.thread_num = 1
        if self.first_dim_size != 1:
            self.thread_num = 2

        self.data_gm = self.tik_instance.Tensor(
            self.dtype_x,
            (self.first_dim_size * self.axis_size * self.last_dim_size,),
            name="data_gm",
            scope=tik.scope_gm)

    def argmax_compute(self):
        """
        argmax_compute

        Parameters
        ----------

        Returns
        -------
        result : tik_instance
            self.tik_instance
        """
        # if not need split
        self.result_gm_value = self.tik_instance.Tensor(
            self.dtype_x, (self.gm_result_size,),
            name="result_gm_value",
            scope=tik.scope_gm)
        self.result_gm = self.tik_instance.Tensor(
            "int32", (self.gm_result_size,),
            name="result_gm",
            scope=tik.scope_gm)

        if self.argmax_axis < len(self.shape_x) - 1:
            self.argmax_not_last_axis()
        else:
            self.argmax_last_axis()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=(self.data_gm,),
            outputs=(self.result_gm, self.result_gm_value))
        return self.tik_instance

    def get_cut_info_by_lastdims(self, dim_size, core_num):
        """
        get_cut_info_by_lastdims

        Parameters
        ----------
        dim_size : int
            dim_size
        core_num : int
            core_num

        Returns
        -------
        result : list
            [core_used, core_seg]
        """
        core_seg = _get_ceil_int(dim_size, core_num)
        core_seg = _get_ceil_int(core_seg, self.data_each_vector)
        core_seg = self.data_each_vector * core_seg
        core_used = _get_ceil_int(dim_size, core_seg)

        return core_used, core_seg

    # pylint: disable=too-many-locals
    def argmax_not_last_axis(self):
        """
        scedule for argmax_not_last_axis

        Parameters
        ----------

        Returns
        -------
        None
        """
        core_number = self.product_core_num
        # core size 1
        core_one, core_one_seg = \
            self.get_cut_info_by_lastdims(self.last_dim_size, core_number)
        offset = 0
        if self.first_dim_size >= core_one:
            segment_loop = self.last_dim_size // self.segment
            segment_tail = self.last_dim_size % self.segment
            segment_tail_data = segment_tail
            # calcu tail
            if segment_tail % self.data_each_block != 0 and segment_loop != 0:
                segment_tail_data = \
                    (segment_tail // self.data_each_block) * \
                    self.data_each_block + \
                    (self.data_each_block
                     if segment_tail % self.data_each_block != 0 else 0)
                offset = 0 + segment_tail - segment_tail_data

            if segment_tail != 0 and segment_tail < self.data_each_block and \
                    segment_loop == 0:
                core_number = 1
            core_number_all = self.first_dim_size
            core_loop = core_number_all // core_number
            core_over = core_number_all - (core_loop * 32)

            with self.tik_instance.for_range(
                    0, core_number, block_num=core_number) as num_core_i:
                with self.tik_instance.for_range(0, core_loop) as num_core_j:
                    first_i = core_loop * num_core_i + num_core_j
                    self.compute_argmax_not_last_axis_cut_by_first_dim(
                        first_i, segment_loop, offset, segment_tail,
                        segment_tail_data)
                with self.tik_instance.if_scope(num_core_i < core_over):
                    first_i = core_loop * core_number + num_core_i
                    self.compute_argmax_not_last_axis_cut_by_first_dim(
                        first_i, segment_loop, offset, segment_tail,
                        segment_tail_data)
        else:
            core_tail = core_one_seg * core_one - self.last_dim_size
            with self.tik_instance.for_range(0, self.first_dim_size) as num_i:
                if core_tail == 0:
                    with self.tik_instance.for_range(
                            0, core_one, block_num=core_one) as core_id:
                        offset_in = \
                            num_i * self.axis_size * self.last_dim_size + \
                            core_id * core_one_seg
                        offset_out = num_i * self.last_dim_size + \
                                     core_id * core_one_seg
                        self.compute_argmax_not_last_axis_cut_by_last_dim(
                            core_one_seg, offset_in, offset_out)
                else:
                    with self.tik_instance.for_range(
                            0, core_one, block_num=core_one) as core_id:
                        offset_in = \
                            num_i * self.axis_size * self.last_dim_size + \
                            core_id * core_one_seg
                        offset_out = num_i * self.last_dim_size + \
                                     core_id * core_one_seg
                        with self.tik_instance.if_scope(
                                core_id < core_one - 1):
                            self.compute_argmax_not_last_axis_cut_by_last_dim(
                                core_one_seg, offset_in, offset_out)
                        with self.tik_instance.else_scope():
                            tail_data = \
                                self.last_dim_size - \
                                core_one_seg * (core_one - 1)
                            self.compute_argmax_not_last_axis_cut_by_last_dim(
                                tail_data, offset_in, offset_out)

    def compute_argmax_not_last_axis_cut_by_last_dim(self, data_segment,
                                                     in_offset, out_offset):
        """
        compute for last_axis

        Parameters
        ----------
        data_segment : int
            data len for process
        in_offset : int
            gm addr begin offset
        out_offset : int
            gm addr end offset

        Returns
        -------
        None
        """
        segment_loop = data_segment // self.segment
        segment_tail = data_segment % self.segment
        with self.tik_instance.for_range(0, segment_loop) as segm_i:
            gm_in_offset = in_offset + self.segment * segm_i
            gm_out_offset = out_offset + self.segment * segm_i
            self.do_not_last(self.segment, gm_in_offset, gm_out_offset)
        if segment_tail != 0:
            segment_tail_data = \
                _get_ceil_int(segment_tail, self.data_each_vector) * \
                self.data_each_vector
            offset = segment_tail_data - segment_tail
            gm_in_offset = in_offset + self.segment * segment_loop - offset
            gm_out_offset = out_offset + self.segment * segment_loop - offset
            self.do_not_last(segment_tail_data, gm_in_offset, gm_out_offset)

    # pylint: disable=too-many-arguments
    def compute_argmax_not_last_axis_cut_by_first_dim(
            self, first_i, segment_loop, offset, segment_tail,
            segment_tail_data):
        """
        compute when cut by first_dim

        Parameters
        ----------
        first_i : int
            data len for process
        segment_loop : int
            gm addr begin offset
        offset : int
            gm addr end offset
        segment_tail : int
            segment_tail
        segment_tail_data :int
            segment_tail_data

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, segment_loop) as segm_i:
            gm_in_offset = first_i * self.axis_size * self.last_dim_size + \
                           segm_i * self.segment
            gm_out_offset = first_i * self.last_dim_size + \
                            segm_i * self.segment
            self.do_not_last(self.segment, gm_in_offset, gm_out_offset)

        if segment_tail != 0 and segment_tail_data % 8 == 0:
            gm_in_offset = first_i * self.axis_size * self.last_dim_size + \
                           segment_loop * self.segment + offset
            gm_out_offset = first_i * self.last_dim_size + \
                            segment_loop * self.segment + offset
            self.do_not_last(segment_tail_data, gm_in_offset, gm_out_offset)

        elif segment_tail != 0 and segment_tail_data > 8:
            # last_axis < segment and not 8 alagn
            pro_len = _get_ceil_int(segment_tail_data, 2)
            pro_len = _get_ceil_int(pro_len, 8) * 8
            offset = segment_tail_data - pro_len
            gm_in_offset = first_i * self.axis_size * self.last_dim_size + \
                           segment_loop * self.segment
            gm_out_offset = first_i * self.last_dim_size + \
                            segment_loop * self.segment
            self.do_not_last(pro_len, gm_in_offset, gm_out_offset)
            gm_in_offset = first_i * self.axis_size * self.last_dim_size + \
                           segment_loop * self.segment + offset
            gm_out_offset = first_i * self.last_dim_size + \
                            segment_loop * self.segment + offset
            self.do_not_last(pro_len, gm_in_offset, gm_out_offset)

        elif segment_tail != 0:
            # one core if last_axis < 8
            gm_in_offset = first_i * self.axis_size * self.last_dim_size + \
                           segment_loop * self.segment
            gm_out_offset = first_i * self.last_dim_size + \
                            segment_loop * self.segment
            self.do_not_last(segment_tail_data, gm_in_offset, gm_out_offset)

    # pylint: disable=too-many-locals
    def do_not_last(self, segment, gm_in_offset, gm_out_offset):
        """
        process for a segment when arg not last dim

        Parameters
        ----------
        segment : int
            data len for process
        gm_in_offset : int
            gm addr begin offset
        gm_out_offset : int
            gm addr end offset

        Returns
        -------
        None
        """
        ub_a = self.tik_instance.Tensor(
            self.dtype_x, (self.segment,), name="ub_a", scope=tik.scope_ubuf)
        ub_c = self.tik_instance.Tensor(
            "int32", (self.segment,), name="ub_c", scope=tik.scope_ubuf)
        data_segment = segment
        nbust_len = _get_ceil_int(data_segment, self.data_each_block)
        self.tik_instance.data_move(ub_a, self.data_gm[gm_in_offset], 0, 1,
                                    nbust_len, 0, 0)
        # Init out
        repeat = _get_ceil_int(data_segment, self.data_each_block * 8)
        self.tik_instance.vector_dup(OUT_MASK, ub_c, 0, _get_ceil_int(
            data_segment, OUT_MASK), 1, 8)

        thread_num = 2 if self.axis_size > 2 else 1
        with self.tik_instance.for_range(
                1, self.axis_size, thread_num=thread_num) as axis_i:
            ub_b = self.tik_instance.Tensor(
                self.dtype_x, (self.segment,),
                name="ub_b",
                scope=tik.scope_ubuf)
            ub_mask = self.tik_instance.Tensor(
                "uint64", (self.segment // OUT_MASK,),
                name="ub_mask",
                scope=tik.scope_ubuf)
            self.tik_instance.data_move(
                ub_b, self.data_gm[gm_in_offset + axis_i * self.last_dim_size],
                0, 1, nbust_len, 0, 0)

            self.tik_instance.vcmpv_lt(ub_mask, ub_a, ub_b, repeat, 1, 1, 8, 8)
            int64_num = _get_ceil_int(data_segment, OUT_MASK)
            with self.tik_instance.for_range(0, int64_num) as i:
                mask_l = self.tik_instance.Scalar("uint64")
                mask_l.set_as(ub_mask[i])
                with self.tik_instance.if_scope(mask_l != 0):
                    self.tik_instance.vector_dup([mask_l, mask_l],
                                                 ub_c[i * OUT_MASK],
                                                 axis_i, 1, 1, 8)

            self.tik_instance.vmax(self.full_mask, ub_a, ub_a, ub_b, repeat, 1,
                                   1, 1, 8, 8, 8)
        nbust_len_out = _get_ceil_int(data_segment, 8)
        self.tik_instance.data_move(self.result_gm[gm_out_offset], ub_c, 0, 1,
                                    nbust_len_out, 0, 0)
        nbust_len_out = _get_ceil_int(data_segment, self.data_each_block)
        self.tik_instance.data_move(self.result_gm_value[gm_out_offset], ub_a,
                                    0, 1, nbust_len_out, 0, 0)

    def get_tiling_info(self):
        """
        get_tiling_info when arg with last dim

        Parameters
        ----------
        None

        Returns
        -------
        result : list
            buf_size, loop_times, over_size, align_flag
        """
        if self.dtype_x == "float16":
            self.segment = MAX_SEGMENT_LEN * 2
        segment_size = self.segment
        align_flag = ((self.c_align_ubsize % segment_size) != 0)
        if segment_size <= self.c_align_ubsize:
            buf_size = segment_size
            loop_times = self.c_align_ubsize // segment_size
            over_size = self.c_align_ubsize - (loop_times * segment_size)
        else:
            loop_times = 0
            buf_size = self.c_align_ubsize
            over_size = buf_size
        return buf_size, loop_times, over_size, align_flag

    def argmax_last_axis(self):
        """
        scedule then do last axis

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        core_number = self.product_core_num
        core_number_all = self.first_dim_size
        if core_number_all < self.data_each_block:
            core_number = 1
        core_segment = core_number_all // core_number
        if core_segment == 0:
            core_segment = 1
        core_segment = _get_ceil_int(
            core_segment, self.data_each_block) * self.data_each_block
        core_num_used = _get_ceil_int(core_number_all, core_segment)
        core_segment_tail = core_number_all % core_segment
        with self.tik_instance.for_range(
                0, core_num_used, block_num=core_num_used) as n_i:
            if core_segment_tail == 0:
                self.compute_argmax_last_axis(n_i, core_segment, core_segment)

            if core_segment_tail != 0:
                with self.tik_instance.if_scope(n_i < (core_num_used - 1)):
                    self.compute_argmax_last_axis(n_i, core_segment,
                                                  core_segment)
                with self.tik_instance.else_scope():
                    self.compute_argmax_last_axis(n_i, core_segment_tail,
                                                  core_segment)

    def compute_argmax_last_axis(self, n_i, core_segment, segment_core):
        """
        compute arg when do last axis

        Parameters
        ----------
        n_i : int
            the first loop index
        core_segment : int
            segment process len
        segment_core : int
            the total segment index

        Returns
        -------
        None
        """
        ub_buf_size, loop_times, over_size, align_flag = self.get_tiling_info()
        self.ub_result_int32 = self.tik_instance.Tensor(
            "int32", (MAX_SEGMENT_LEN,),
            name="ub_result_int32",
            scope=tik.scope_ubuf)
        self.ub_result_value = self.tik_instance.Tensor(
            self.dtype_x, (MAX_SEGMENT_LEN,),
            name="ub_result_value",
            scope=tik.scope_ubuf)
        thread_num = 2 if core_segment > 1 else 1

        def _run(segment_len, segment_index):
            with self.tik_instance.for_range(
                    0, segment_len, thread_num=thread_num) as core_i:
                index = core_i + MAX_SEGMENT_LEN * segment_index
                offset = n_i * segment_core + index
                self.result_int32 = self.tik_instance.Scalar("int32")
                self.result_int32.set_as(0)
                self.result_float32 = self.tik_instance.Scalar("float32")
                self.result_float32.set_as(SCALAR_MIN_FP32)
                self.result_out_scalar = self.tik_instance.Scalar(self.dtype_x)
                if self.dtype_x == "float16":
                    argmax_func = self.do_argmax_last_axis
                    self.result_out_scalar.set_as(SCALAR_MIN_FP16)
                else:
                    argmax_func = self.do_argmax_last_axis_fp32
                    self.result_out_scalar.set_as(SCALAR_MIN_FP32)
                if loop_times != 0:
                    with self.tik_instance.for_range(0, loop_times) as loop:
                        argmax_func(ub_buf_size, loop, offset)
                if align_flag:
                    argmax_func(over_size, loop_times, offset)
                self.ub_result_int32[core_i] = self.result_int32
                self.ub_result_value[core_i] = self.result_out_scalar
            gm_out_offset = n_i * segment_core + \
                            MAX_SEGMENT_LEN * segment_index
            out_nbust = _get_ceil_int(segment_len, 8)
            self.tik_instance.data_move(self.result_gm[gm_out_offset],
                                        self.ub_result_int32, 0, 1,
                                        out_nbust, 0, 0)
            out_nbust = _get_ceil_int(segment_len, self.data_each_block)
            self.tik_instance.data_move(self.result_gm_value[gm_out_offset],
                                        self.ub_result_value, 0, 1,
                                        out_nbust, 0, 0)

        _loop_segment = core_segment // MAX_SEGMENT_LEN
        _loop_segment_tail = core_segment % MAX_SEGMENT_LEN
        with self.tik_instance.for_range(
                0, _loop_segment) as _loop:
            _run(MAX_SEGMENT_LEN, _loop)
        if _loop_segment_tail != 0:
            _run(_loop_segment_tail, _loop_segment)

    # pylint: disable=too-many-locals,too-many-statements
    def do_argmax_last_axis(self, ub_buf_size, loop, n_i):
        """
        do arg in one segment fo float16

        Parameters
        ----------
        ub_buf_size : int
            process len
        loop : int
            segment index in one core
        n_i : int
            the first loop index

        Returns
        -------
        None
        """
        ub_result = self.tik_instance.Tensor(
            self.dtype_x, (_get_ceil_int(ub_buf_size, self.data_each_vector) *
                           self.data_each_vector,),
            name="ub_result",
            scope=tik.scope_ubuf)
        ub_result_int32 = self.tik_instance.Tensor(
            "int32", (16,), name="ub_result_int32", scope=tik.scope_ubuf)
        ub_data = self.tik_instance.Tensor(
            self.dtype_x, (_get_ceil_int(ub_buf_size, self.data_each_vector) *
                           self.data_each_vector,),
            name="ub_data",
            scope=tik.scope_ubuf)
        offset = loop * self.segment + n_i * self.axis_size
        self.tik_instance.data_move(ub_data, self.data_gm[offset], 0, 1,
                                    _get_ceil_int(ub_buf_size,
                                                  self.data_each_block), 0, 0)

        def _calu_mask_by_one_zero(_len):
            _mask_h, _mask_l = 0, 0
            if _len > 32:
                _mask_l = MASK_0_1
                for i in range(_len - 32):
                    _mask_h = _mask_h + 2 ** (2 * i)
            else:
                _mask_h = 0
                for i in range(_len):
                    _mask_l = _mask_l + 2 ** (2 * i)
            return _mask_h, _mask_l

        def _get_tail_mask(tail_len):
            if tail_len <= OUT_MASK:
                mask = 2 ** tail_len - 1
                mask_h = MAX_MASK_INT64
                mask_l = MAX_MASK_INT64 - mask
            else:
                mask_l = 0
                mask = 2 ** (tail_len - OUT_MASK) - 1
                mask_h = MAX_MASK_INT64 - mask
            return mask_h, mask_l

        tail = ub_buf_size % self.data_each_vector
        if tail != 0:
            mask_h, mask_l = _get_tail_mask(tail)
            _offset = ub_buf_size // (self.data_each_vector)
            self.tik_instance.vector_dup(
                [mask_h, mask_l], ub_data[_offset * self.data_each_vector],
                SCALAR_MIN_FP16, 1, 1, 8)

        repeat_times = _get_ceil_int(ub_buf_size, self.data_each_vector)
        self.tik_instance.vcmax(self.data_each_vector, ub_result, ub_data,
                                repeat_times, 1, 1, 8)

        if repeat_times > 64:
            _repeat_times = _get_ceil_int(repeat_times, 64)
            _repeat_tail = (repeat_times * 2) % self.data_each_vector
            if _repeat_tail != 0:
                mask_h, mask_l = _get_tail_mask(_repeat_tail)
                _offset = repeat_times * 2 // self.data_each_vector
                self.tik_instance.vector_dup(
                    [mask_h, mask_l],
                    ub_result[_offset * self.data_each_vector],
                    SCALAR_MIN_FP16, 1, 1, 8)
            repeat_times = _get_ceil_int(repeat_times, 64)
            ub_second_result = self.tik_instance.Tensor(
                self.dtype_x,
                (_get_ceil_int(repeat_times, self.data_each_vector) *
                 self.data_each_vector,),
                name="ub_second_result",
                scope=tik.scope_ubuf)
            self.tik_instance.vcmax([MASK_0_1,
                                     MASK_0_1],
                                    ub_second_result, ub_result,
                                    _repeat_times, 1, 1, 8)

            ub_third_result = self.tik_instance.Tensor(
                self.dtype_x,
                (_get_ceil_int(_repeat_times, self.data_each_vector) *
                 self.data_each_vector,),
                name="ub_third_result",
                scope=tik.scope_ubuf)

            _mask = _calu_mask_by_one_zero(repeat_times % 64)
            self.tik_instance.vcmax(_mask,
                                    ub_third_result, ub_second_result,
                                    1, 1, 1, 8)
            third_max_index = self.tik_instance.Scalar("uint16")
            third_max_index.set_as(ub_third_result[1])
            second_max_index = self.tik_instance.Scalar("uint16")
            second_max_index.set_as(ub_second_result[third_max_index + 1])
            last_max_index = self.tik_instance.Scalar("uint16")
            last_max_index.set_as(
                ub_result[third_max_index * 64 + second_max_index + 1])
            max_index = self.tik_instance.Scalar("uint16")
            max_index.set_as(
                third_max_index * 64 * 64 + second_max_index * 64 + \
                last_max_index)

        elif repeat_times > 1:
            _repeat_tail = repeat_times % 64
            _mask = _calu_mask_by_one_zero(_repeat_tail)
            if _repeat_tail == 0:
                _mask = [MASK_0_1, MASK_0_1]
            ub_second_result = self.tik_instance.Tensor(
                self.dtype_x,
                (_get_ceil_int(repeat_times, self.data_each_vector) *
                 self.data_each_vector,),
                name="ub_second_result",
                scope=tik.scope_ubuf)
            self.tik_instance.vcmax(_mask,
                                    ub_second_result, ub_result,
                                    1, 1, 1, 8)
            second_max_index = self.tik_instance.Scalar("uint16")
            second_max_index.set_as(ub_second_result[1])
            last_max_index = self.tik_instance.Scalar("uint16")
            last_max_index.set_as(ub_result[second_max_index + 1])
            max_index = self.tik_instance.Scalar("uint16")
            max_index.set_as(second_max_index * 64 + last_max_index)
        else:
            max_index = self.tik_instance.Scalar("uint16")
            max_index.set_as(ub_result[1])

        max_index_int32 = self.tik_instance.Scalar("int32")
        max_index_int32.set_as(max_index)
        ub_result_cmp = self.tik_instance.Tensor(
            self.dtype_x, (self.data_each_vector,),
            name="ub_result_cmp",
            scope=tik.scope_ubuf)
        ub_result_cmp[0].set_as(self.result_out_scalar)
        ub_result_cmp[1].set_as(ub_data[max_index_int32])
        ub_result_int32[0].set_as(self.result_int32)
        ub_result_int32[1].set_as(max_index_int32 + loop * self.segment)
        self.tik_instance.vcmax(2, ub_result_cmp, ub_result_cmp,
                                1, 1, 1, 8)
        max_index1 = self.tik_instance.Scalar("uint16")
        max_index1.set_as(ub_result_cmp[1])
        self.result_int32.set_as(ub_result_int32[max_index1])
        self.result_out_scalar.set_as(ub_result_cmp[0])

    def do_argmax_last_axis_fp32(self, ub_buf_size, loop, n_i):
        """
        do arg in one segment fo float32

        Parameters
        ----------
        ub_buf_size : int
            process len
        loop : int
            segment index in one core
        n_i : int
            the first loop index

        Returns
        -------
        None
        """
        segment = ub_buf_size
        _ub_size = [
            self.data_each_block * _get_ceil_int(self.segment,
                                                 self.data_each_block)
        ]
        ub_index_int32 = self.tik_instance.Tensor(
            "int32", _ub_size, name="ub_index_int32", scope=tik.scope_ubuf)
        ub_data = self.tik_instance.Tensor(
            self.dtype_x, _ub_size, name="ub_data", scope=tik.scope_ubuf)
        ub_max_64 = self.tik_instance.Tensor(
            self.dtype_x, [self.data_each_vector],
            name="ub_max_64",
            scope=tik.scope_ubuf)
        cmp_mask_ub = self.tik_instance.Tensor(
            "uint64", [
                _get_ceil_int(
                    _get_ceil_int(self.segment, self.data_each_vector),
                    self.data_each_vector) * self.data_each_vector
            ],
            name="cmp_mask_ub",
            scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(self.data_each_vector, ub_max_64,
                                     SCALAR_MIN_FP32, 1, 1, 8)
        nbust = _get_ceil_int(segment, self.data_each_block)
        offset = loop * self.segment + n_i * self.axis_size
        repeat = _get_ceil_int(segment, self.data_each_vector)
        self.tik_instance.data_move(ub_data, self.data_gm[offset], 0, 1, nbust,
                                    0, 0)
        tail = ub_buf_size % self.data_each_vector
        if tail != 0:
            mask_h = 0
            mask = 2 ** tail - 1
            mask_l = MAX_MASK_INT64 - mask
            _offset = ub_buf_size // self.data_each_vector
            self.tik_instance.vector_dup(
                [mask_h, mask_l], ub_data[_offset * self.data_each_vector],
                SCALAR_MIN_FP32, 1, 1, 8)
        self.tik_instance.vmax(self.data_each_vector, ub_max_64, ub_data,
                               ub_max_64, repeat, 1, 1, 1, 0, 8, 0)
        self.tik_instance.vcmpv_eq(cmp_mask_ub, ub_max_64, ub_data, repeat, 1,
                                   1, 0, 8)
        self.tik_instance.vector_dup(self.data_each_vector, ub_index_int32, 0,
                                     1, 1, 8)
        with self.tik_instance.for_range(0, repeat) as i:
            index = repeat - 1 - i
            mask_l = self.tik_instance.Scalar("uint64")
            mask_l.set_as(cmp_mask_ub[index])
            with self.tik_instance.if_scope(mask_l != 0):
                self.tik_instance.vector_dup([mask_l, mask_l], ub_index_int32,
                                             index * self.data_each_vector, 1,
                                             1, 8)
        # get one value from 64
        max_value = self.tik_instance.Scalar(self.dtype_x)
        max_index = self.tik_instance.Scalar("int32")
        max_value.set_as(ub_max_64[0])
        max_index.set_as(ub_index_int32[0])
        scalar_valid = self.data_each_vector \
            if segment > self.data_each_vector else segment
        with self.tik_instance.for_range(1, scalar_valid) as i:
            max_cmp_value = self.tik_instance.Scalar(self.dtype_x)
            max_cmp_index = self.tik_instance.Scalar("int32")
            max_cmp_value.set_as(ub_max_64[i])
            with self.tik_instance.if_scope(max_cmp_value > max_value):
                max_cmp_index.set_as(ub_index_int32[i])
                max_value.set_as(ub_max_64[i])
                max_index.set_as(max_cmp_index + i)
        with self.tik_instance.if_scope(max_value > self.result_out_scalar):
            self.result_out_scalar.set_as(max_value)
            self.result_int32.set_as(max_index + loop * self.segment)
