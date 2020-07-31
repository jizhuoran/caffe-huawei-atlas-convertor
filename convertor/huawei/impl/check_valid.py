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

check_valid
"""
from te import tik
from te import platform as tbe_platform
from topi.cce import util
from impl import common_util

# count of shape dim
DIM_CNT = 2
# the second dim num
DIM_SECOND_SIZE = 4
# minimum value for float16
FLOAT16_MINIMUM = 2**(-24)
FLOAT16_SCALAR = 2**12


@util.check_input_type(dict, dict, dict, str)
def check_valid(bbox_tensor, img_metas, valid_tensor,
                kernel_name="check_valid"):
    """
    check_valid compute

    Parameters
    ----------
    bbox_tensor : dict
        the dict info of bbox_tensor
    img_metas : dict
        the dict info of img_metas
    valid_tensor: list
        the dict info of valid_tensor
    kernel_name : str
        kernel name, default value is "check_valid"

    Returns
    -------
    None
    """
    bbox_shape = bbox_tensor.get("shape")
    bbox_dtype = bbox_tensor.get("dtype").lower()
    # check for value > 0
    util.check_shape_rule(bbox_shape)
    util.check_tensor_shape_size(bbox_shape)

    bbox_dtype_check_list = [
        "float16",
    ]
    util.check_dtype_rule(bbox_dtype, bbox_dtype_check_list)
    valid_shape = valid_tensor.get("shape")

    img_metas_dtype = img_metas.get("dtype").lower()
    img_metas_shape = img_metas.get("shape")
    util.check_shape_rule(img_metas_shape)
    util.check_tensor_shape_size(img_metas_shape)
    if img_metas_dtype != bbox_dtype:
        raise RuntimeError(
            "The type of img_metas should be same to bbox_tensor!")

    if len(bbox_shape) != DIM_CNT:
        raise RuntimeError("the length of bbox_shape must be 2,\
            while it is: %d" % len(bbox_shape))
    if bbox_shape[-1] != DIM_SECOND_SIZE:
        raise RuntimeError(
            "the second dim must be 4, while it's: %d" % bbox_shape[-1])

    if bbox_shape[0] != valid_shape[0]:
        raise RuntimeError(
            "the dim-0 must be equal between 'bbox_tensor' and 'valid_tensor'")

    util.check_kernel_name(kernel_name)
    cvd = CheckValid(bbox_tensor, img_metas, valid_tensor, kernel_name)
    return cvd.check_valid()


# pylint: disable=useless-object-inheritance,too-many-instance-attributes
class CheckValid(object):
    """
    object of CheckValid
    """
    def __init__(self, bbox_tensor, img_metas, valid_tensor, kernel_name):
        self.kernel_name = kernel_name
        self.bbox_shape = bbox_tensor.get("shape")
        self.bbox_dtype = bbox_tensor.get("dtype").lower()
        self.bbox_dtype_size = common_util.get_data_size(self.bbox_dtype)

        self.valid_shape = valid_tensor.get("shape")
        self.valid_dtype_size = 1

        # select operations only handle 128 elements once time.
        self.__default_rows_per_job = 32 * 4 * 1

        self.job_num = self.__calc_job_num()

        self.img_metas = img_metas
        self.tik_instance = tik.Tik()

        # buffer for threshold extract
        self.img_metas_gm = self.tik_instance.Tensor(
            "float16", (16,), name="img_metas_gm", scope=tik.scope_gm)
        self.img_metas_ub = self.tik_instance.Tensor(
            "float16", (16,), name="img_metas_ub", scope=tik.scope_ubuf)
        self.threshold_h = self.tik_instance.Scalar("float16", "threshold_h")
        self.threshold_w = self.tik_instance.Scalar("float16", "threshold_w")
        self.__extract_threshold_as_scalar()

        # input bbox tensor from caller fp16, 128;  fp32, 64
        self.bbox_tensor_gm = self.tik_instance.Tensor(
            "float16",
            self.bbox_shape,
            name="bbox_tensor_gm",
            scope=tik.scope_gm)

        # return buffer, gm be whole
        self.data_ret_int8_gm = self.tik_instance.Tensor(
            "int8",
            self.valid_shape,
            name="data_ret_int8_gm",
            scope=tik.scope_gm)
        self.padded_bytes = 0
        self.last_job_row_aligned = self.__calc_last_job_row()

        # each job used buffer maximum
        self.job_buf_row = self.get_job_buffer_row(
        )

        self.quad_flag_ub = self.tik_instance.Tensor(
            "float16", (self.job_buf_row * 4,),
            name="quad_flag_ub",
            scope=tik.scope_ubuf)
        self.quad_flags_sum_ub = self.tik_instance.Tensor(
            "float16", (self.job_buf_row * 4,),
            name="quad_flags_sum_ub",
            scope=tik.scope_ubuf)
        # need set value before each-times using!
        # this buffer will used in multi-time.
        # contains threshold and the transform tmp for return.
        self.quad_threshold_ub = self.tik_instance.Tensor(
            "float16", (self.job_buf_row * 4,),
            name="quad_threshold_ub",
            scope=tik.scope_ubuf)

        self.ones_ub = self.tik_instance.Tensor(
            "float16", (self.job_buf_row, 4),
            name="ones_ub",
            scope=tik.scope_ubuf)
        self.zeros_ub = self.tik_instance.Tensor(
            "float16", (self.job_buf_row, 4),
            name="zeros_ub",
            scope=tik.scope_ubuf)

        _repeat_time = max(4 * self.job_buf_row // 128, 1)
        _process_elem_count = self.get_handle_num_with_clip_128(4)

        self.tik_instance.vector_dup(_process_elem_count, self.ones_ub, 1,
                                     _repeat_time, 1, 8)
        self.tik_instance.vector_dup(_process_elem_count, self.zeros_ub, 0,
                                     _repeat_time, 1, 8)

        self.data_ret_int8_ub = self.tik_instance.Tensor(
            "int8", (self.job_buf_row, 1),
            name="data_ret_int8_ub",
            scope=tik.scope_ubuf)
        self.data_ret_mask_ub = self.tik_instance.Tensor(
            "uint16", (self.job_buf_row * 4 // 16,),
            name="data_ret_mask_ub",
            scope=tik.scope_ubuf)
        self.data_ret_ub = self.tik_instance.Tensor(
            "float16", (self.job_buf_row, 1),
            name="data_ret_ub",
            scope=tik.scope_ubuf)
        self.ret_unfold_half_ub = self.tik_instance.Tensor(
            "float16", (self.job_buf_row * 2,),
            name="ret_unfold_half_ub",
            scope=tik.scope_ubuf)

        self.bbox_tensor_ub = self.tik_instance.Tensor(
            "float16", (self.job_buf_row, 4),
            name="bbox_tensor_ub",
            scope=tik.scope_ubuf)

    def get_handle_num_with_clip_128(self, column):
        """get get_handle_num_with_clip_128

        Parameters
        ----------
        column : int
            column number

        Returns
        -------
        result : int
            min(self.job_buf_row * column, 128)
        """
        return min(self.job_buf_row * column, 128)

    def clear_quad_flags(self):
        """clear_quad_flags

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        _repeat_time = max(4 * self.job_buf_row // 128, 1)
        _process_elem_count = self.get_handle_num_with_clip_128(4)

        self.tik_instance.vector_dup(_process_elem_count, self.quad_flag_ub, 0,
                                     _repeat_time, 1, 8)
        self.tik_instance.vector_dup(
            _process_elem_count, self.quad_flags_sum_ub, 0, _repeat_time, 1, 8)

    def get_job_buffer_row(self):
        """get_job_buffer_row

        Parameters
        ----------
        None

        Returns
        -------
        result : self.__default_rows_per_job
        """
        return self.__default_rows_per_job

    def get_job_num(self):
        """get_job_num

        Parameters
        ----------
        None

        Returns
        -------
        result : self.job_num
        """
        return self.job_num

    def __calc_last_job_row(self):
        """__calc_last_job_row

        Parameters
        ----------
        None

        Returns
        -------
        result : last_job_bytes // (self.bbox_dtype_size * self.bbox_shape[1])
        """
        over_row = self.bbox_shape[0] % self.__default_rows_per_job

        # if last job is equal to the default job row.
        if over_row < 1:
            return self.__default_rows_per_job

        over_bytes = over_row * self.bbox_shape[1] * self.bbox_dtype_size

        _align_size = 32
        align_32bytes_count = over_bytes // _align_size

        if over_bytes % _align_size > 0:
            align_32bytes_count += 1

        last_job_bytes = _align_size * align_32bytes_count
        # total less than 32B
        self.padded_bytes = last_job_bytes - over_bytes

        return last_job_bytes // (self.bbox_dtype_size * self.bbox_shape[1])


    def __calc_job_num(self):
        """__calc_job_num

        Parameters
        ----------
        None

        Returns
        -------
        result : job_num
        """
        if self.bbox_shape[0] % self.__default_rows_per_job > 0:
            job_num = self.bbox_shape[0] // self.__default_rows_per_job + 1
        else:
            job_num = self.bbox_shape[0] // self.__default_rows_per_job
        return job_num

    def __extract_threshold_as_scalar(self):
        """__extract_threshold_as_scalar

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tik_instance.data_move(self.img_metas_ub, self.img_metas_gm, 0, 1,
                                    1, 0, 0)

        img_metas_dummy_ub = self.tik_instance.Tensor(
            "float16", (16,), name="img_metas_dummy_ub", scope=tik.scope_ubuf)

        scalar_ratio = self.tik_instance.Scalar("float16", "scalar_ratio")
        scalar_ratio.set_as(self.img_metas_ub[2])

        # scalar, 0.25&floor, or 0.15 & round
        self.tik_instance.vmuls(
            16,
            img_metas_dummy_ub,
            self.img_metas_ub,
            scalar_ratio,
            1,
            1,
            1,
            8,
            8)

        self.threshold_h.set_as(img_metas_dummy_ub[0])
        self.threshold_w.set_as(img_metas_dummy_ub[1])

    def get_last_job_header(self, head_type):
        """get_last_job_header

        Parameters
        ----------
        head_type : str
            head_type
        Returns
        -------
        result : int
            last_align_job_head
        """
        if head_type == "bbox":
            _unit = 4
        elif head_type == "valid":
            _unit = 1
        else:
            raise KeyError(
                "only support 'bbox' or 'valid', got {}.".format(head_type))
        element_total_num = self.bbox_shape[0] * _unit
        if self.job_num > 1:
            last_align_job_head = -self.job_buf_row * _unit + element_total_num
        else:
            last_align_job_head = -self.last_job_row_aligned * _unit + \
                                  element_total_num

        # if only one job, we should cut in at the start.
        if last_align_job_head < 0:
            last_align_job_head = 0

        return last_align_job_head

    def __move_job_bbox_to_ub(self, job_index=0, inverted=False):
        """__move_job_bbox_to_ub

        Parameters
        ----------
        job_index : int
            job_index
        inverted : bool
            inverted

        Returns
        -------
        None
        """
        # fp32--/8--4B. fp16-/16--2B, int8-/32--1B
        with self.tik_instance.if_scope(inverted is False):
            _burst_fp16 = max(self.job_buf_row * 4 // 16, 1)

            self.tik_instance.data_move(
                self.bbox_tensor_ub,
                self.bbox_tensor_gm[self.job_buf_row * job_index * 4], 0, 1,
                _burst_fp16, 0, 0)
        with self.tik_instance.else_scope():
            bbox_head = self.get_last_job_header("bbox")
            if self.job_num > 1:
                _burst_fp16 = max(self.job_buf_row * 4 // 16, 1)
            else:
                _burst_fp16 = max(self.last_job_row_aligned * 4 // 16, 1)

            self.tik_instance.data_move(self.bbox_tensor_ub,
                                        self.bbox_tensor_gm[bbox_head], 0, 1,
                                        _burst_fp16, 0, 0)

    def __move_once_job_ret_to_gm(self, job_index=0, inverted=False):
        """__move_once_job_ret_to_gm

        Parameters
        ----------
        job_index : int
            job_index
        inverted : bool
            inverted

        Returns
        -------
        None
        """
        # int8 as result
        with self.tik_instance.if_scope(inverted is False):
            _burst_int8 = max(self.job_buf_row // 32, 1)

            self.tik_instance.data_move(
                self.data_ret_int8_gm[self.job_buf_row * job_index],
                self.data_ret_int8_ub, 0, 1, _burst_int8, 0, 0)
        with self.tik_instance.else_scope():
            valid_head = self.get_last_job_header("valid")
            if self.job_num > 1:
                _burst_int8 = max(
                    self.job_buf_row * self.valid_dtype_size // 32, 1)
            else:
                _burst_int8 = max(
                    self.last_job_row_aligned * self.valid_dtype_size // 32, 1)

                # owing to the output type is int8, different with input fp16
                #  | * * * * * * * * * *(32B)
                #                * * * * * * * * *(32B) |
                # small shape, when last job output value non-align
                _over_in_single_job = self.last_job_row_aligned * \
                                      self.valid_dtype_size % 32

                if _over_in_single_job > 0:
                    _burst_int8 += 1  # hard over write with one more  block

            self.tik_instance.data_move(self.data_ret_int8_gm[valid_head],
                                        self.data_ret_int8_ub, 0, 1,
                                        _burst_int8, 0, 0)

    def calc_col_ge_flag(self):
        """calculate for each column, and add this column'result into
        quad_flags_sum

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # operation on succession
        _repeat_time = max(self.job_buf_row * 4 // 128, 1)

        _deal_elem_num = self.get_handle_num_with_clip_128(4)
        rep_offset_fp16 = _deal_elem_num * 2 // 32

        self.tik_instance.vector_dup(_deal_elem_num, self.zeros_ub, 0,
                                     _repeat_time, 1, 8)

        self.tik_instance.vmax(_deal_elem_num, self.ones_ub,
                               self.bbox_tensor_ub, self.zeros_ub, _repeat_time,
                               1, 1, 1, rep_offset_fp16, rep_offset_fp16,
                               rep_offset_fp16)

        # 'zeros_ub' as x - max(0 ,x)
        self.tik_instance.vsub(_deal_elem_num, self.zeros_ub,
                               self.bbox_tensor_ub, self.ones_ub, _repeat_time,
                               1, 1, 1, rep_offset_fp16, rep_offset_fp16,
                               rep_offset_fp16)

        self.tik_instance.vabs(_deal_elem_num, self.ones_ub, self.zeros_ub,
                               _repeat_time, 1, 1, rep_offset_fp16,
                               rep_offset_fp16)

        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub,
                                     FLOAT16_MINIMUM, _repeat_time, 1,
                                     rep_offset_fp16)

        self.tik_instance.vmin(_deal_elem_num, self.zeros_ub, self.ones_ub,
                               self.quad_threshold_ub, _repeat_time, 1, 1, 1,
                               rep_offset_fp16, rep_offset_fp16,
                               rep_offset_fp16)

        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub,
                                     FLOAT16_SCALAR, _repeat_time, 1,
                                     rep_offset_fp16)

        self.tik_instance.vmul(_deal_elem_num, self.ones_ub,
                               self.quad_threshold_ub, self.zeros_ub,
                               _repeat_time, 1, 1, 1, rep_offset_fp16,
                               rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vmul(_deal_elem_num, self.zeros_ub,
                               self.quad_threshold_ub, self.ones_ub,
                               _repeat_time, 1, 1, 1, rep_offset_fp16,
                               rep_offset_fp16, rep_offset_fp16)

        # -1
        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, 1,
                                     _repeat_time, 1, rep_offset_fp16)

        self.tik_instance.vsub(_deal_elem_num, self.ones_ub, self.zeros_ub,
                               self.quad_threshold_ub, _repeat_time, 1, 1, 1,
                               rep_offset_fp16, rep_offset_fp16,
                               rep_offset_fp16)

        # abs
        self.tik_instance.vabs(_deal_elem_num, self.quad_flag_ub, self.ones_ub,
                               _repeat_time, 1, 1, rep_offset_fp16,
                               rep_offset_fp16)

        # step 4, add once unfold flag into sum flags
        self.__add_col_flag_to_sum()

    def calc_col_lt_flag(self):
        """
        calculate for each column, and add this column'result into
        quad_flags_sum
        """
        _repeat_time = max(self.job_buf_row * 4 // 128, 1)
        _deal_elem_num = self.get_handle_num_with_clip_128(4)

        rep_offset_fp16 = _deal_elem_num * 2 // 32

        _mask64 = 0xAAAAAAAAAAAAAAAA
        self.tik_instance.vector_dup([_mask64, _mask64], self.quad_threshold_ub,
                                     self.threshold_h, _repeat_time, 1,
                                     rep_offset_fp16)

        _mask64 = 0x5555555555555555
        self.tik_instance.vector_dup([_mask64, _mask64], self.quad_threshold_ub,
                                     self.threshold_w, _repeat_time, 1,
                                     rep_offset_fp16)

        self.tik_instance.vsub(_deal_elem_num, self.zeros_ub,
                               self.quad_threshold_ub, self.bbox_tensor_ub,
                               _repeat_time, 1, 1, 1, rep_offset_fp16,
                               rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub,
                                     FLOAT16_MINIMUM, _repeat_time, 1, 8)

        self.tik_instance.vmin(_deal_elem_num, self.ones_ub, self.zeros_ub,
                               self.quad_threshold_ub, _repeat_time, 1, 1, 1,
                               rep_offset_fp16, rep_offset_fp16,
                               rep_offset_fp16)

        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, 0,
                                     _repeat_time, 1, rep_offset_fp16)
        self.tik_instance.vmax(_deal_elem_num, self.zeros_ub, self.ones_ub,
                               self.quad_threshold_ub, _repeat_time, 1, 1, 1,
                               rep_offset_fp16, rep_offset_fp16,
                               rep_offset_fp16)

        # mul 2 times
        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub,
                                     FLOAT16_SCALAR, _repeat_time, 1,
                                     rep_offset_fp16)

        self.tik_instance.vmul(_deal_elem_num, self.ones_ub,
                               self.quad_threshold_ub, self.zeros_ub,
                               _repeat_time, 1, 1, 1, rep_offset_fp16,
                               rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vmul(_deal_elem_num, self.quad_flag_ub,
                               self.quad_threshold_ub, self.ones_ub,
                               _repeat_time, 1, 1, 1, rep_offset_fp16,
                               rep_offset_fp16, rep_offset_fp16)

        # step 4, add once unfold flag into sum flags
        self.__add_col_flag_to_sum()

    def __add_col_flag_to_sum(self):
        _repeat_time = max(self.job_buf_row * 4 // 128, 1)
        _process_elem_count = self.get_handle_num_with_clip_128(4)
        rep_offset_fp16 = _process_elem_count * 2 // 32

        self.tik_instance.vadd(
            _process_elem_count,
            self.quad_flags_sum_ub,
            self.quad_flag_ub,
            self.quad_flags_sum_ub,
            _repeat_time,  # repeat
            1,
            1,
            1,
            rep_offset_fp16,
            rep_offset_fp16,
            rep_offset_fp16)

    def merge_successive_four_elem_to_one_val(self):
        """merge successive four elements elements into once."""
        _repeat_time = max(self.job_buf_row * 4 // 128, 1)
        _process_elem_count = self.get_handle_num_with_clip_128(4)
        rep_offset_fp16 = _process_elem_count * 2 // 32

        self.tik_instance.vcpadd(
            _process_elem_count,
            self.ret_unfold_half_ub,
            self.quad_flags_sum_ub,
            _repeat_time,  # repeat
            1,
            1,
            rep_offset_fp16)

        self.tik_instance.vcpadd(
            _process_elem_count, self.data_ret_ub, self.ret_unfold_half_ub,
            max(_repeat_time // 2, 1), 1, 1, rep_offset_fp16)

    def transform_to_one_or_zero(self):
        """transform_to_one_or_zero

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # method1 [N, 1] * 0.25 --> (0, 1)  floor
        # method2 [N, 1] * 0.15 --> (0, 1)  round
        # 128 as the maximum elements could been processed in once command
        _repeat_time = max(self.job_buf_row // 128, 1)
        _process_elem_count = self.get_handle_num_with_clip_128(1)
        rep_offset_fp16 = _process_elem_count * 2 // 32
        rep_offset_int8 = _process_elem_count * 1 // 32

        self.tik_instance.vmuls(
            _process_elem_count,
            self.quad_threshold_ub,
            self.data_ret_ub,
            0.067,  # scalar, 0.25&floor, or 0.15 & round
            _repeat_time,  # repeat
            1,
            1,
            rep_offset_fp16,
            rep_offset_fp16)

        # (0,1,2,3)/4 --> 0/1
        # RuntimeError: v100 mini doesn't support float16 to int8 with floor mode.
        # mini support 'none' only, ei, round mode, 0.15*3-->0; 0.15*4 --> 1
        # cloud support the floor mode
        self.tik_instance.vconv(_process_elem_count, 'none',
                                self.data_ret_int8_ub, self.quad_threshold_ub,
                                _repeat_time, 1, 1, rep_offset_int8,
                                rep_offset_fp16)

    def check_valid_compute(self, job_index, inverted=False):
        """entrance for each core

        Parameters
        ----------
        job_index : int
            job_index
        inverted : bool
            inverted

        Returns
        -------
        resut : instance
            tik_instance
        """
        self.clear_quad_flags()  # must clear it in each core
        self.__move_job_bbox_to_ub(job_index, inverted)
        with self.tik_instance.for_range(0, 2) as n_col:
            # col=0,1, point to x1, y1
            with self.tik_instance.if_scope(
                    n_col == 0):
                self.calc_col_ge_flag()

            with self.tik_instance.else_scope():
                self.calc_col_lt_flag()

        self.merge_successive_four_elem_to_one_val()
        self.transform_to_one_or_zero()

        self.__move_once_job_ret_to_gm(job_index, inverted)

    def check_valid(self):
        """main entrance for check valid operation

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        _max_threshold = self.threshold_w
        _max_threshold = self.threshold_h

        # cloud 32 core; mini 2 core
        core_number = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        total_job_num = self.get_job_num()
        base_loop_per_core = total_job_num // core_number
        core_over = total_job_num - (base_loop_per_core * core_number)

        with self.tik_instance.for_range(
                0, core_number, block_num=core_number) as num_core_i:
            with self.tik_instance.for_range(0,
                                             base_loop_per_core) as num_core_j:
                self.check_valid_compute(
                    base_loop_per_core * num_core_i + num_core_j)

            # last schedule
            with self.tik_instance.if_scope(num_core_i < core_over):
                with self.tik_instance.if_scope(num_core_i == core_over - 1):
                    # last core of whole task
                    self.check_valid_compute(
                        base_loop_per_core * core_number + num_core_i, True)
                with self.tik_instance.else_scope():
                    self.check_valid_compute(
                        base_loop_per_core * core_number + num_core_i)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.bbox_tensor_gm, self.img_metas_gm],
            outputs=[self.data_ret_int8_gm])
        return self.tik_instance
