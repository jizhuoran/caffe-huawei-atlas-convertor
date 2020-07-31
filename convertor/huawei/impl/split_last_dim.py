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

split_for_last_dim
"""

from te import tik
from te import platform as tbe_platform
from topi.cce import util

# 1111000011110000111100001111000011110000111100001111000011110000
MASK_FOR_11110000 = 17361641481138401520
# 0000111100001111000011110000111100001111000011110000111100001111
MASK_FOR_00001111 = 1085102592571150095


def _apply_mem(tik_instance, dtype, shape, name, scope=tik.scope_ubuf):
    """apply mem fuc

    Parameters
    ----------
    tik_instance: tik_instance
        tik_instance
    dtype: str
        ub dtype
    shape: list
        ub shape
    name: str
        ub name
    scope: scope
        scope_ubuf or scope_gm

    Returns
    -------
    Tensor: Tensor
    """
    return tik_instance.Tensor(dtype, shape, name=name, scope=scope)


def _get_ceil_int(int1, int2):
    """get cel for input1 and input2
    """
    if int1 == 0:
        return 1
    _result = int1 // int2
    if int1 % int2 == 0:
        return _result

    return _result + 1


# pylint: disable=locally-disabled,too-many-instance-attributes,too-many-arguments,unused-argument
class SplitLastDim():
    """Function: use to finish SplitLastDim main functions
    """

    def __init__(self, shape, dtype, split_dim, num_split, size_splits):
        """init SplitLastDim parameters
        """
        self.src_shape = shape
        self.src_dtype = dtype
        self.data_size = util.check_tensor_shape_size(list(self.src_shape))
        self.split_dim = split_dim
        self.num_split = num_split
        self.split_dim_size = self.src_shape[self.split_dim]

        self.data_size_first_dim = self.data_size // self.split_dim_size
        self.split_output_dim_size = \
            self.src_shape[self.split_dim] // self.num_split
        self.output_size = \
            self.split_output_dim_size * self.data_size_first_dim
        # get dtype size, float16 size = 2 byte   / float32 size = 4 byte
        self.dtype_size = \
            tbe_platform.cce_intrin.get_bit_len(self.src_dtype) // 8
        # get one block data size, block align len
        # the len in one block = 16 fp16 and = 8 fp32
        self.data_len_one_block = 32 // self.dtype_size
        self.data_len_one_vector = self.data_len_one_block * 8

        self.ub_availble = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.UB_SIZE) - 8 * 1024
        self.ub_max_data = self.ub_availble // self.dtype_size
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        self.max_dims = 1
        self.segment_len = 1
        self.out_ub = None
        self.out_ub_1 = None
        self.index_reg = None
        self.index_reg_1 = None

        # input and output tensor in gm
        self.src_gm = self.tik_instance.Tensor(
            self.src_dtype, [self.data_size_first_dim, self.split_dim_size],
            name="src_gm",
            scope=tik.scope_gm)
        self.dst_gm_list = []

        for _, i in enumerate(range(num_split)):
            dst_gm = self.tik_instance.Tensor(
                self.src_dtype,
                [self.data_size_first_dim, self.split_output_dim_size],
                name="dst_gm_" + str(i),
                scope=tik.scope_gm)
            self.dst_gm_list.append(dst_gm)

    def split_last_dim_less_block(self):
        """copy all data from src to des
        """
        # core scedule
        self.max_dims = 256 // 2
        inner_loop = self.data_len_one_block
        core_len = _get_ceil_int(self.data_size_first_dim, inner_loop)
        core_len = _get_ceil_int(core_len, self.core_num)
        if core_len == 0:
            core_len = 1

        dims_per_core = core_len * inner_loop
        core_used = self.data_size_first_dim // dims_per_core
        if self.data_size_first_dim % dims_per_core != 0:
            core_used = core_used + 1
        tail_dims_core = \
            self.data_size_first_dim - (core_used - 1)*dims_per_core

        self.segment_len = self.max_dims * self.split_dim_size
        if self.split_output_dim_size == 4 and self.data_len_one_block == 8:
            split_fuc = self.proc_4_with_fp32
        else:
            split_fuc = self.proc_default
        # for core loop
        with self.tik_instance.for_range(
                0, core_used, block_num=core_used) as _core_index:
            core_dims_offset = _core_index * dims_per_core
            if tail_dims_core != dims_per_core:
                # for copy segment loop
                with self.tik_instance.if_scope(_core_index < (core_used - 1)):
                    split_fuc(dims_per_core, core_dims_offset)

                with self.tik_instance.else_scope():
                    split_fuc(tail_dims_core, core_dims_offset)
            else:
                split_fuc(dims_per_core, core_dims_offset)

    def proc_4_with_fp32(self, work_dims, core_offset):
        """when output size = 4 run this do_spilt_four_fp32
        0 1 2  3  4  5  6  7
        8 9 10 11 12 13 14 15

        step1:
            copy gm(+0) --> ub1
            copy gm(+4) --> ub2
        step2:
            set_mask(00001111)
            output[0:4] = vadds(ub1(0:4),0)
            set_mask(11110000)
            output[4:8] = vadds(ub2(4:8),0)
        step3:
            copy output to gm
        """
        segment_loop = work_dims * self.split_dim_size // self.segment_len
        segment_tail_len = \
            work_dims * self.split_dim_size - segment_loop*self.segment_len

        def _run_one_segment(_segment_len, _segment_index):
            """_run_one_segment
            """
            # get gm offset
            offset = \
                core_offset*self.split_dim_size + \
                _segment_index*self.segment_len
            out_offset = \
                core_offset*self.split_output_dim_size \
                + _segment_index*self.max_dims*self.split_output_dim_size

            # apply ub for data
            data_ub = _apply_mem(self.tik_instance, self.src_dtype,
                                 [self.segment_len], "data_ub")
            # apply ub for data_1
            data_ub_1 = _apply_mem(self.tik_instance, self.src_dtype,
                                   [self.segment_len], "data_ub_1")
            # calcu len for copy
            burst_len = _get_ceil_int(_segment_len, self.data_len_one_block)
            # copy data from gm to ub1
            self.tik_instance.data_move(data_ub, self.src_gm[offset], 0, 1,
                                        burst_len, 0, 0)
            burst_len = _get_ceil_int(_segment_len - 4, self.data_len_one_block)
            # copy data from gm to ub2
            self.tik_instance.data_move(
                data_ub_1, self.src_gm[offset + self.split_output_dim_size], 0,
                1, burst_len, 0, 0)

            # apply ub to save output
            self.out_ub = \
                _apply_mem(self.tik_instance, self.src_dtype,
                           [self.max_dims*self.split_output_dim_size],
                           "out_ub")
            self.out_ub_1 = \
                _apply_mem(self.tik_instance, self.src_dtype,
                           [self.max_dims*self.split_output_dim_size],
                           "out_ub_1")
            # do split_d use adds_4_to_ub
            self.adds_4_to_ub(_segment_len, [data_ub, data_ub_1], out_offset)

        with self.tik_instance.for_range(0, segment_loop) as _segment_loop:
            _run_one_segment(self.segment_len, _segment_loop)

        if segment_tail_len != 0:
            # process tail data
            _run_one_segment(segment_tail_len, segment_loop)

    # pylint: disable=locally-disabled,too-many-locals
    def adds_4_to_ub(self, segment_len, data_ub_list, out_offset):
        """used adds 0 to move data from input ub to output ub
        """
        data_ub, data_ub_1 = data_ub_list
        dst_m0 = 1
        src_m0 = self.split_dim_size // self.data_len_one_block * 2
        dst_m1 = 8
        src_m1 = self.split_dim_size * 8 * 2 // self.data_len_one_block
        mask1_scalar = self.tik_instance.Scalar(dtype="uint64")
        mask2_scalar = self.tik_instance.Scalar(dtype="uint64")
        mask1_scalar.set_as(MASK_FOR_00001111)
        mask2_scalar.set_as(MASK_FOR_11110000)
        work_dim = segment_len // self.split_dim_size
        repeat_time = _get_ceil_int(work_dim // 2, 8)
        nbust = _get_ceil_int(work_dim * self.split_output_dim_size,
                              self.data_len_one_block)

        def process_one_output(_index, _out_ub):
            if _index % 2 == 0:
                # the output index is odd number
                first_ub = data_ub
                second_ub = data_ub_1
                first_offset = (_index // 2) * self.data_len_one_block
                second_offset = first_offset + self.split_dim_size - 8
            else:
                # the output index is even number
                first_ub = data_ub_1
                second_ub = data_ub
                first_offset = (_index // 2) * self.data_len_one_block
                second_offset = first_offset + self.split_dim_size
            data_ub_first = first_ub[first_offset]
            data_ub_second = second_ub[second_offset]
            # the max value of src_m1 is 255,
            # when src_m1 > 255, connot use repeat for vadds
            if src_m1 <= 255:
                # conditons: src_m1 <= 255 vadds use repeat
                self.tik_instance.vadds([mask1_scalar, mask1_scalar], _out_ub,
                                        data_ub_first, 0, repeat_time, dst_m0,
                                        src_m0, dst_m1, src_m1)
                self.tik_instance.vadds([mask2_scalar, mask2_scalar], _out_ub,
                                        data_ub_second, 0, repeat_time, dst_m0,
                                        src_m0, dst_m1, src_m1)
                self.tik_instance.data_move(
                    self.dst_gm_list[_index][out_offset], _out_ub, 0, 1, nbust,
                    0, 0)
            elif repeat_time == 1:
                # conditons: src_m1 > 255 and repeat_time is equal to 0
                # vector cmd "vadds" ignore src_m1
                self.tik_instance.vadds([mask1_scalar, mask1_scalar], _out_ub,
                                        data_ub_first, 0, repeat_time, dst_m0,
                                        src_m0, dst_m1, 8)
                self.tik_instance.vadds([mask2_scalar, mask2_scalar], _out_ub,
                                        data_ub_second, 0, repeat_time, dst_m0,
                                        src_m0, dst_m1, 8)
            else:
                # vadds 0 one by one
                for _, i in enumerate(range(repeat_time)):
                    data_ub_first = \
                        first_ub[i*src_m1*self.data_len_one_block
                                 + first_offset]
                    data_ub_second = \
                        second_ub[second_offset
                                  + i*src_m1*self.data_len_one_block]
                    self.tik_instance.vadds(
                        [mask1_scalar, mask1_scalar],
                        _out_ub[i * dst_m1 * self.data_len_one_block],
                        data_ub_first, 0, 1, dst_m0, src_m0, dst_m1, 8)
                    self.tik_instance.vadds(
                        [mask2_scalar, mask2_scalar],
                        _out_ub[i * dst_m1 * self.data_len_one_block],
                        data_ub_second, 0, 1, dst_m0, src_m0, dst_m1, 8)

            self.tik_instance.data_move(self.dst_gm_list[_index][out_offset],
                                        _out_ub, 0, 1, nbust, 0, 0)

        for _, output_index in enumerate(range(self.num_split // 2)):
            process_one_output(output_index * 2, self.out_ub)
            process_one_output(output_index * 2 + 1, self.out_ub_1)
        if self.num_split % 2 == 1:
            process_one_output(self.num_split - 1, self.out_ub)

    def proc_default(self, work_dims, core_offset):
        """run this do_spilt use scalar
        """
        segment_loop = work_dims * self.split_dim_size // self.segment_len
        segment_tail_len = \
            work_dims * self.split_dim_size - segment_loop * self.segment_len

        def _run_one_segment(_segment_len, _segment_index):
            # calcu gm offset
            offset = core_offset*self.split_dim_size + \
                     _segment_index*self.segment_len
            out_offset = \
                core_offset*self.split_output_dim_size \
                + _segment_index * self.max_dims * self.split_output_dim_size
            # copy from gm to ub
            data_ub = _apply_mem(self.tik_instance, self.src_dtype,
                                 [self.segment_len], "data_ub")
            nbust = _get_ceil_int(_segment_len, self.data_len_one_block)
            self.tik_instance.data_move(data_ub, self.src_gm[offset], 0, 1,
                                        nbust, 0, 0)
            self.out_ub = \
                _apply_mem(self.tik_instance, self.src_dtype,
                           [self.max_dims*self.split_output_dim_size],
                           "out_ub")
            self.out_ub_1 = \
                _apply_mem(self.tik_instance, self.src_dtype,
                           [self.max_dims*self.split_output_dim_size],
                           "out_ub_1")
            self.index_reg = [
                self.tik_instance.Scalar(dtype=self.src_dtype)
                for _, _ in enumerate(range(8))
            ]
            self.index_reg_1 = [
                self.tik_instance.Scalar(dtype=self.src_dtype)
                for _, _ in enumerate(range(8))
            ]
            self.scalar_to_ub(_segment_len, data_ub, out_offset)

        with self.tik_instance.for_range(0, segment_loop) as _segment_loop:
            _run_one_segment(self.segment_len, _segment_loop)

        if segment_tail_len != 0:
            _run_one_segment(segment_tail_len, segment_loop)

    def scalar_to_ub(self, segment_len, data_ub, out_offset):
        """used scalar to move data from input ub to output ub
        """
        first_loop = segment_len // self.split_dim_size
        if segment_len % self.split_dim_size != 0:
            first_loop = first_loop + 8

        def _run_one_output(_index, _out_ub):
            with self.tik_instance.for_range(0, _get_ceil_int(first_loop,
                                                              8)) as i:
                with self.tik_instance.for_range(
                        0, self.split_output_dim_size) as j:
                    with self.tik_instance.for_range(0, 8) as k:
                        _input_index = \
                            (i*8 + k)*self.split_dim_size \
                            + _index*self.split_output_dim_size + j
                        _out_index = (i * 8 +
                                      k) * self.split_output_dim_size + j
                        _out_ub[_out_index].set_as(data_ub[_input_index])
            nbust = _get_ceil_int(first_loop * self.split_output_dim_size,
                                  self.data_len_one_block)
            self.tik_instance.data_move(self.dst_gm_list[_index][out_offset],
                                        _out_ub, 0, 1, nbust, 0, 0)

        for _, output_index in enumerate(range(self.num_split // 2)):
            _run_one_output(output_index * 2, self.out_ub)
            _run_one_output(output_index * 2 + 1, self.out_ub_1)
        if self.num_split % 2 == 1:
            _run_one_output(self.num_split - 1, self.out_ub)

    def split_last_dim_with_blocks(self):
        """copy all data from src to des
        """
        # core scedule
        many_copy_num = self.ub_max_data // 2
        self.max_dims = many_copy_num // self.split_output_dim_size
        dims_per_core = _get_ceil_int(self.data_size_first_dim, self.core_num)
        if dims_per_core == 0:
            dims_per_core = 1

        core_used = self.data_size_first_dim // dims_per_core
        if self.data_size_first_dim % dims_per_core != 0:
            core_used = core_used + 1
        tail_dims_core = \
            self.data_size_first_dim - (core_used - 1)*dims_per_core

        self.segment_len = self.max_dims * self.split_output_dim_size
        # for core loop
        with self.tik_instance.for_range(
                0, core_used, block_num=core_used) as _core_index:
            # for copy segment loop
            core_dims_offset = _core_index * dims_per_core
            if tail_dims_core != dims_per_core:
                with self.tik_instance.if_scope(_core_index < (core_used - 1)):
                    self.data_move_with_blocks(dims_per_core, core_dims_offset)

                with self.tik_instance.else_scope():
                    self.data_move_with_blocks(tail_dims_core, core_dims_offset)
            else:
                self.data_move_with_blocks(dims_per_core, core_dims_offset)

    def data_move_with_blocks(self, work_dims, core_offset):
        """copy all data from src to des the last size is 32B align
        """
        segment_loop = \
            work_dims * self.split_output_dim_size // self.segment_len
        segment_tail_len = \
            work_dims * self.split_output_dim_size \
            - segment_loop * self.segment_len
        # copy from gm to ub
        data_ub = _apply_mem(self.tik_instance, self.src_dtype,
                             [self.segment_len], "data_ub")
        data_ub_1 = _apply_mem(self.tik_instance, self.src_dtype,
                               [self.segment_len], "data_ub_1")

        def _run_one_segment(_segment_index, _segment_len):
            offset = core_offset*self.split_dim_size + \
                     _segment_index*self.max_dims*self.split_dim_size
            out_offset = \
                core_offset*self.split_output_dim_size \
                + _segment_index*self.max_dims*self.split_output_dim_size
            len_burst = self.split_output_dim_size // self.data_len_one_block
            n_burst = _get_ceil_int(_segment_len, self.data_len_one_block)
            n_burst = _get_ceil_int(n_burst, len_burst)
            src_stride = _get_ceil_int(
                (self.split_dim_size - self.split_output_dim_size),
                self.data_len_one_block)
            out_n_burst = _get_ceil_int(_segment_len, self.data_len_one_block)

            def _run_one_output(_index):
                if _index % 2 == 0:
                    src_ub = data_ub
                else:
                    src_ub = data_ub_1
                src_offset = offset + _index * self.split_output_dim_size
                self.tik_instance.data_move(src_ub, self.src_gm[src_offset], 0,
                                            n_burst, len_burst, src_stride, 0)
                self.tik_instance.data_move(
                    self.dst_gm_list[_index][out_offset], src_ub, 0, 1,
                    out_n_burst, 0, 0)

            for _, i in enumerate(range(self.num_split // 2)):
                _run_one_output(i * 2)
                _run_one_output(i * 2 + 1)
            if self.num_split % 2 == 1:
                _run_one_output(self.num_split - 1)

        with self.tik_instance.for_range(0, segment_loop) as _segment_loop:
            _run_one_segment(_segment_loop, self.segment_len)

        if segment_tail_len != 0:
            _run_one_segment(segment_loop, segment_tail_len)

    def run_tik(self, kernel_name):
        """cal tik_instance according
        """
        if self.split_output_dim_size % self.data_len_one_block == 0:
            self.split_last_dim_with_blocks()
        else:
            self.split_last_dim_less_block()

        self.tik_instance.BuildCCE(
            kernel_name=kernel_name,
            inputs=[self.src_gm],
            outputs=self.dst_gm_list)
        return self.tik_instance


def check_whether_lastdim(shape, split_dim):
    """check whether the shape and axis is split_d by last dim
    """

    if len(shape) == 1 or split_dim != len(shape) - 1:
        return False

    return True


def check_whether_equal_split(size_splits):
    """check whether split_v == split_d
    """
    size_set = list(set(size_splits))
    if len(size_set) == 1:
        return True

    return False


def check_use_last_dim_branch(shape,
                              dtype,
                              split_dim,
                              num_split,
                              size_splits=None):
    """check whether use new tik branch for last dim tp split_d
    """
    # check whether split_d by last dim
    is_last_dim = check_whether_lastdim(shape, split_dim)

    # check whether in support_dtype
    support_dtype = ("float16", "float32")
    is_dtype_support = dtype in support_dtype

    # check whether the value in size_splits must be equal
    is_split = check_whether_equal_split(size_splits)

    # check the size in new branch condition
    split_l = SplitLastDim(shape, dtype, split_dim, num_split, size_splits)
    half_ub = split_l.ub_max_data // 2
    out_split_size = shape[split_dim] // num_split
    is_shape_support = ((out_split_size % 8 == 0 and out_split_size < half_ub)
                        or out_split_size < 8)

    return is_shape_support and is_dtype_support and is_split and is_last_dim


# pylint: disable=locally-disabled,unused-argument,too-many-arguments
def split_last_dim(shape, dtype, split_dim, num_split, size_splits,
                   kernel_name):
    """Split a tensor into len(size_splits) tensors along last dimension.

    Parameters
    ----------
    shape: list or tuple
        the shape of input tensor.
    dtype: str
        the dtype of input tensor.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        used to specify the number of outputs.
    size_splits: list or tuple
        a Python list containing the sizes of each output tensor
        along `split_dim`.
    kernel_name: str
        cce kernel name.

    Returns
    -------
    None.
    """
    res = SplitLastDim(shape, dtype, split_dim, num_split, size_splits)

    return res.run_tik(kernel_name)
