#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

resize_bilinear_grad
"""

from te import tik
from topi.cce import util
from te import platform as tbe_platform

def _ceil_div(value, factor):
    """
    caculate floor value of div

    Parameters
    ----------
    value: dtype of int or float
        original value
    factor: dtype of int or float
        dividend value

    Returns
    -------
    value: dtype of int or float
    """
    if value % factor == 0:
        quotient = value // factor
    else:
        quotient = value // factor + 1

    return quotient


def calc_segment(tik_instance, total_seg, seg_index, seg_len):
    """calc one block gradient

    Parameters
    ----------
    tik_instance: class
    total_seg: int
    seg_index: int
    seg_len: float

    Returns
    -------
    ret_seg_len:int
    """
    left_seg_len = tik_instance.Scalar(dtype="int64")
    ret_seg_len = tik_instance.Scalar(dtype="int64")
    seg_gap = tik_instance.Scalar(dtype="int64", init_value=seg_len)
    left_seg_len.set_as(total_seg - seg_index * seg_len)
    tik_instance.scalar_min(ret_seg_len, left_seg_len, seg_gap)
    return ret_seg_len


def clear_ub(tik_instance, dst_ub):
    """clear ub to zero

    Parameters
    ----------
    tik_instance: class
    dst_ub: destinatiob ub

    Returns
    -------
    None
    """
    shape = dst_ub.shape
    data_len = 1
    for i in shape:
        data_len = data_len * i
    dst_ub.reshape((data_len,))

    total_repeat_times = data_len // 64
    tail = data_len % 64
    vector_dup_times = (total_repeat_times + 254) // 255
    with tik_instance.for_range(0, vector_dup_times) as i:
        repeat_times = calc_segment(tik_instance, total_repeat_times,
                                    i, 255)
        tik_instance.vector_dup(64, dst_ub[i * 255 * 64], 0,
                                repeat_times, 1, 8)

    if tail > 0:
        tik_instance.vector_dup(tail, dst_ub[total_repeat_times * 64],
                                0, 1, 1, 8)

    dst_ub.reshape(shape)


# pylint: disable-msg=invalid-name,unused-argument,too-many-arguments
@util.check_input_type(dict, dict, (list, tuple), bool, bool, str)
def resize_nearest_neighbor_grad_d_h(
        grads, y, size, align_corners=False, half_pixel_centers=False,
        kernel_name="resize_nearest_neighbor_grad"):
    """
    The resize_nearest_neighbor_grad_d operator divides the kernel according
    to the H axis

    Parameters
    ----------
    grads: dict
        dict with keys(shape and dtype) of input grads
    y: dict
        dict with keys(shape and dtype) of output y
    size: list
        (orig_height, orig_width)
    align_corners: bool
        whether align_corners
    half_pixel_centers: bool
        whether open half_pixel_centers
    kernel_name: str
        kernel_name

    Returns
    -------
    tik_instance: tik_instance
    """

    resize_bilinear_grad_reslut = ResizeNearestGrad(grads, y,
                                                    align_corners,
                                                    half_pixel_centers)

    return resize_bilinear_grad_reslut.tik_instance_function(kernel_name)


# pylint: disable=too-many-instance-attributes
class ResizeNearestGrad():
    """
       Function: use to finish MaxPoolWithargmax main functions
       Modify : 2019-11-06
    """

    def __init__(self, grads, images, align_corners=False, half_pixel_centers=False):
        """
        init MaxPoolWithargmax parameters

        Parameters
        ----------
        grads : dict
            dict with keys(shape and dtype) of grads
        images : dict
            dict with keys(shape and dtype) of images
        align_corners : bool
            decide how to calculate for scale
        half_pixel_centers: bool
            whether open half_pixel_centers

        Returns
        -------
        None
        """
        self.grads_shape = grads.get("shape")
        self.grads_dtype = grads.get("dtype").lower()
        self.images_shape = images.get("shape")
        self.images_dtype = images.get("dtype").lower()
        self.align_corners = align_corners
        self.tik_instance = tik.Tik()
        self.half_pixel_centers = half_pixel_centers
        self.batch_size = self.grads_shape[0]
        self.c1_size = self.grads_shape[1]
        self.in_size_h = self.grads_shape[2]
        self.in_size_w = self.grads_shape[3]
        self.c_block_size = self.grads_shape[4]
        self.nc1 = self.batch_size * self.c1_size
        self.out_size_h = self.images_shape[2]
        self.out_size_w = self.images_shape[3]
        self.w_batch = 512
        self.w_in_loop = _ceil_div(self.in_size_w, self.w_batch)
        self.w_in_tail = self.in_size_w % self.w_batch
        if self.w_in_tail == 0:
            self.w_in_tail = self.w_batch
        height_in = self.in_size_h
        weight_in = self.in_size_w
        height_out = self.out_size_h
        weight_out = self.out_size_w
        # input and output
        if (self.align_corners) and (height_in > 1):
            height_in -= 1
            height_out -= 1

        if (self.align_corners) and (weight_in > 1):
            weight_in -= 1
            weight_out -= 1

        self.scale_h = float(height_out) / float(height_in)
        self.scale_w = float(weight_out) / float(weight_in)

        self.grads_gm = self.tik_instance.Tensor(self.grads_dtype,
                                                 self.grads_shape,
                                                 name="grads_gm",
                                                 scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.images_dtype,
                                                  self.images_shape,
                                                  name="output_gm",
                                                  scope=tik.scope_gm,
                                                  is_atomic_add=True)

    def location_h(self, reg_index_y, reg_cur_index):
        """
        Calculate h coordinates
        """
        h_floor_buf = self.tik_instance.Tensor("int32", (8,),
                                               name="h_floor_buf",
                                               scope=tik.scope_ubuf)
        h_scale_buf = self.tik_instance.Tensor("float32", (8,),
                                               name="h_scale_buf",
                                               scope=tik.scope_ubuf)
        h_block_buf = self.tik_instance.Tensor("int32", (8,),
                                               name="h_scale_buf",
                                               scope=tik.scope_ubuf)

        self.tik_instance.vector_dup(
            8, h_block_buf, reg_cur_index, 1, 1, 8)
        self.tik_instance.vconv(8, "", h_scale_buf[0],
                                h_block_buf[0], 1, 1, 1, 8, 8)
        if not self.half_pixel_centers:
            self.tik_instance.vmuls(8, h_scale_buf, h_scale_buf,
                                    self.scale_h, 1, 1, 1, 8, 8)
        else:
            self.tik_instance.vadds(8, h_scale_buf, h_scale_buf,
                                    float(0.5), 1, 1, 1, 8, 8)
            self.tik_instance.vmuls(8, h_scale_buf, h_scale_buf,
                                    self.scale_h, 1, 1, 1, 8, 8)
        cmd = "round" if self.align_corners else "floor"
        self.tik_instance.vconv(8, cmd, h_floor_buf[0],
                                h_scale_buf[0], 1, 1, 1, 8, 8)

        const_value_int32 = self.tik_instance.Tensor("int32", (8,),
                                                     name="const_value_int32",
                                                     scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(8, const_value_int32,
                                     self.images_shape[2] - 1, 1, 1, 8)
        self.tik_instance.vmin(8, h_floor_buf, h_floor_buf, const_value_int32,
                               1, 1, 1, 1, 8, 8, 8)
        reg_index_y.set_as(h_floor_buf[0])

    def move_out(self, ub_output, nc1_index, reg_index_y):
        """
        move data output
        """
        self.tik_instance.set_atomic_add(1)
        self.tik_instance.data_move(
            self.output_gm[(nc1_index * self.out_size_h + reg_index_y) *
                           self.out_size_w * self.c_block_size],
            ub_output[0], 0, 1, self.out_size_w * 2, 0, 0)

        self.tik_instance.set_atomic_add(0)

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def tik_instance_function(self, kernel_name):
        """
        tik_instance_function

        Parameters
        ----------
        kernel_name: str
            kernel_name

        Returns
        -------
        tik_instance
        """

        core_counts = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        # (N,M) to (N1,M1) and w_out small
        h_per_core = self.in_size_h // core_counts + (
            1 if self.in_size_h % core_counts > 0 else 0)
        if (self.in_size_h % core_counts == 0) or \
                (self.in_size_h % h_per_core == 0):
            is_same_core = 0
        else:
            is_same_core = 1

        core_num = self.in_size_h // h_per_core + (
            0 if self.in_size_h // core_counts == 0 else is_same_core)
        with self.tik_instance.for_range(0, core_num, block_num=core_num) \
                as core_index:

            with self.tik_instance.if_scope(core_index != core_num - 1):
                with self.tik_instance.for_range(0, h_per_core) as h_in_index:
                    self.fun_w_out_small(core_index, h_per_core, h_in_index)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(
                        0, self.in_size_h - core_index * h_per_core) \
                        as h_in_index:
                    self.fun_w_out_small(core_index, h_per_core, h_in_index)

        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(self.grads_gm),
                                   outputs=(self.output_gm))

        return self.tik_instance

    def equal(self, list_w_num, reg_index_y, reg_cur_index, ub_output):
        """
        repeat are all the same
        """

        repeat = list_w_num[0]
        self.location_h(reg_index_y, reg_cur_index)

        if self.in_size_w > 512 and (512 % repeat) == 0:
            one_num = (512 // repeat)
            reg_repeat = one_num // 8
            location_512 = self.w_batch // repeat
        elif self.in_size_w > 512 and (512 % repeat) != 0:
            if (512 // repeat) >= 8:
                self.w_batch = ((512 // repeat) // 8) * 8 * repeat

            else:
                self.w_batch = (512 // repeat) * repeat

            self.w_in_loop = _ceil_div(self.in_size_w, self.w_batch)
            self.w_in_tail = self.in_size_w % self.w_batch
            if self.w_in_tail == 0:
                self.w_in_tail = self.w_batch
            location_512 = self.w_batch // repeat
            reg_repeat = (location_512 // 8)
            one_num = self.w_batch // repeat
        elif self.in_size_w <= 512:
            reg_repeat = self.out_size_w // 8
            tail_reg = self.out_size_w % 8
            location_512 = self.out_size_w
            one_num = self.out_size_w - tail_reg
            if reg_repeat == 0:
                one_num = self.out_size_w
        reg_repeat1 = (self.in_size_w -
                       (self.w_in_loop - 1) * self.w_batch) // repeat // 8
        tail1 = (self.in_size_w -
                 (self.w_in_loop - 1) * self.w_batch) // repeat % 8
        ub_input = self.tik_instance.Tensor(
            "float32", (self.w_batch, self.c_block_size),
            name="ub_input", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, self.nc1) as nc1_index:
            clear_ub(self.tik_instance, ub_output)
            with self.tik_instance.for_range(0, self.w_in_loop) as loop_index:
                with self.tik_instance.if_scope(
                        loop_index != (self.w_in_loop - 1)):
                    self.tik_instance.data_move(
                        ub_input[0],
                        self.grads_gm[
                            (nc1_index * self.in_size_h +
                             reg_cur_index) * self.in_size_w *
                            self.c_block_size +
                            loop_index * self.w_batch * self.c_block_size],
                        0, 1, (self.w_batch * self.c_block_size + 7) // 8,
                        0, 0)

                    if 8 * 2 * repeat <= 255 and reg_repeat != 0:
                        with self.tik_instance.for_range(0, 2) as i:
                            with self.tik_instance.for_range(0, repeat) as j:
                                self.tik_instance.vadd(
                                    64, ub_output[8 * i +
                                                  one_num * 16 * loop_index],
                                    ub_input[2 * j * 8 + 8 * i],
                                    ub_output[8 * i +
                                              one_num * 16 * loop_index],
                                    reg_repeat, 2, repeat * 2,
                                    2, 16, 8 * 2 * repeat, 16)

                    elif 8 * 2 * repeat > 255 and reg_repeat != 0:
                        with self.tik_instance.for_range(0, 2) as i:
                            with self.tik_instance.for_range(0, repeat) as j:
                                with self.tik_instance.for_range(
                                        0, reg_repeat) as k:
                                    self.tik_instance.vadd(
                                        64, ub_output[
                                            8 * i + 16 * k * 8 +
                                            one_num * 16 * loop_index],
                                        ub_input[2 * j * 8 + 8 * i +
                                                 2 * k * 8 * repeat * 8],
                                        ub_output[8 * i + 16 * k * 8 +
                                                  one_num * 16 * loop_index],
                                        1, 2, repeat * 2, 2, 0, 0, 0)
                    elif reg_repeat == 0:
                        with self.tik_instance.for_range(0, 2) as i:
                            with self.tik_instance.for_range(0, repeat) as j:
                                self.tik_instance.vadd(
                                    8 * location_512,
                                    ub_output[8 * i +
                                              one_num * 16 * loop_index],
                                    ub_input[2 * j * 8 + 8 * i],
                                    ub_output[8 * i +
                                              one_num * 16 * loop_index],
                                    1, 2, repeat * 2, 2, 0, 0, 0)

                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        ub_input[0],
                        self.grads_gm[(nc1_index *
                                       self.in_size_h + reg_cur_index) *
                                      self.in_size_w * self.c_block_size +
                                      loop_index * self.w_batch *
                                      self.c_block_size],
                        0, 1, (self.w_in_tail) * 2, 0, 0)

                    if 8 * 2 * repeat <= 255 and reg_repeat1 != 0:
                        with self.tik_instance.for_range(0, 2) as i:
                            with self.tik_instance.for_range(0, repeat) as j:
                                self.tik_instance.vadd(
                                    64, ub_output[8 * (i) +
                                                  one_num * 16 * loop_index],
                                    ub_input[2 * j * 8 + 8 * i],
                                    ub_output[8 * i +
                                              one_num * 16 * loop_index],
                                    reg_repeat1, 2, repeat * 2, 2, 16,
                                    8 * 2 * repeat, 16)
                                if tail1 != 0:
                                    self.tik_instance.vadd(
                                        8 * tail1,
                                        ub_output[
                                            8 * i + one_num * 16 * loop_index +
                                            reg_repeat1 * 8 * 16],
                                        ub_input[
                                            2 * j * 8 + 8 * i +
                                            2 * reg_repeat1 * 8 * repeat * 8],
                                        ub_output[
                                            8 * i + 16 * reg_repeat1 * 8 +
                                            one_num * 16 * loop_index],
                                        1, 2, repeat * 2, 2, 0, 0, 0)
                    elif 8 * 2 * repeat <= 255 and reg_repeat1 == 0:
                        with self.tik_instance.for_range(0, 2) as i:
                            with self.tik_instance.for_range(0, repeat) as j:
                                self.tik_instance.vadd(
                                    8 * tail1, ub_output[
                                        8 * i + one_num * 16 * loop_index],
                                    ub_input[2 * j * 8 + 8 * i],
                                    ub_output[8 * i +
                                              one_num * 16 * loop_index],
                                    1, 2, repeat * 2, 2, 0, 0, 0)
                    elif 8 * 2 * repeat > 255 and reg_repeat1 != 0:
                        with self.tik_instance.for_range(0, 2) as i:
                            with self.tik_instance.for_range(0, repeat) as j:
                                with self.tik_instance.for_range(
                                        0, reg_repeat1) as k:
                                    self.tik_instance.vadd(
                                        64, ub_output[
                                            8 * i + 16 * k * 8 +
                                            one_num * 16 * loop_index],
                                        ub_input[16 * j  + 8 * i +
                                                 128 * k  * repeat],
                                        ub_output[8 * i + 128 * k  +
                                                  one_num * 16 * loop_index],
                                        reg_repeat1, 2, repeat * 2, 2, 0, 0, 0)
                                if tail1 != 0:
                                    self.tik_instance.vadd(
                                        8 * tail1, ub_output[
                                            8 * i + 128 * (reg_repeat1) +
                                            one_num * 16 * loop_index],
                                        ub_input[16 * j + 8 * i +
                                                 128 * reg_repeat1 * repeat],
                                        ub_output[8 * i + 128 * reg_repeat1 +
                                                  one_num * 16 * loop_index],
                                        1, 2, repeat * 2, 2, 0, 0, 0)

                    elif 8 * 2 * repeat > 255 and reg_repeat1 == 0:
                        with self.tik_instance.for_range(0, 2) as i:
                            with self.tik_instance.for_range(0, repeat) as j:
                                self.tik_instance.vadd(
                                    8 * tail1, ub_output[
                                        one_num * 16 * loop_index + 8 * i],
                                    ub_input[2 * j * 8 + 8 * i],
                                    ub_output[one_num * 16 * loop_index +
                                              8 * i],
                                    1, 2, repeat * 2, 2, 0, 0, 0)
            self.move_out(ub_output, nc1_index, reg_index_y)

    def repeat_numpy(self, list_w_num, list_num, loop_index, repeat_num):
        """
        Calculate the number of repeats
        """
        while loop_index < self.w_in_loop:
            if loop_index != self.w_in_loop - 1:
                order = self.w_batch * loop_index
                list_w_fp = []
                list_w_int = []
                while order < self.w_batch * (loop_index + 1):
                    list_w_fp.append(float(order))

                    if self.align_corners is not True:
                        if self.half_pixel_centers is not True:
                            int_w = int(float(order) * self.scale_w)
                        else:
                            int_w = int((float(order) + float(0.5)) * self.scale_w)
                    else:
                        int_w = round(float(order) * self.scale_w)

                    if int_w > (self.out_size_w - 1):
                        int_w = self.out_size_w - 1

                    list_w_int.append(int_w)
                    list_num.append(int_w)
                    order += 1

                list_w_int_new = list(set(list_w_int))

                list_w_int_new.sort()

                for i in list_w_int_new:
                    list_w_num.append(list_w_int.count(i))

            else:
                order = self.w_batch * loop_index
                list_w_fp = []
                list_w_int = []
                while order < self.in_size_w:
                    list_w_fp.append(float(order))
                    if self.align_corners is not True:
                        if self.half_pixel_centers is not True:
                            int_w = int(float(order) * self.scale_w)
                        else:
                            int_w = int((float(order) + float(0.5)) * self.scale_w)
                    else:
                        int_w = round(float(order) * self.scale_w)

                    if int_w > (self.out_size_w - 1):
                        int_w = self.out_size_w - 1
                    list_w_int.append(int_w)
                    list_num.append(int_w)
                    order += 1
                list_w_int_new = list(set(list_w_int))
                list_w_int_new.sort()
                for i in list_w_int_new:
                    list_w_num.append(list_w_int.count(i))
            loop_index += 1
        repeat_num_tmp = list(set(list_num))
        repeat_num_tmp.sort()
        for i in repeat_num_tmp:
            repeat_num.append(list_num.count(i))

    # pylint: disable=too-many-locals, too-many-arguments
    def not_equal(self, list_w_num, list_num, reg_index_y, reg_cur_index,
                  ub_output):
        """
        repeat is not the same
        """
        w_offline_num = self.tik_instance.Tensor(
            "int32", (len(list_w_num),),
            name="w_offline_num", scope=tik.scope_ubuf)
        reg_w_tmp = self.tik_instance.Scalar(dtype="int32")
        reg_w_tmp1 = self.tik_instance.Scalar(dtype="int32")
        reg_w_tmp2 = self.tik_instance.Scalar(dtype="int32")

        number = 0
        for j in list_w_num:
            reg_w_tmp.set_as(j)
            w_offline_num[number].set_as(reg_w_tmp)
            number = number + 1
        reg_w_out_begin = self.tik_instance.Tensor(
            "int32", (self.w_in_loop,),
            name="reg_w_out_begin", scope=tik.scope_ubuf)
        reg_w_out_end = self.tik_instance.Tensor(
            "int32", (self.w_in_loop,),
            name="reg_w_out_end", scope=tik.scope_ubuf)

        for i in range(0, self.w_in_loop):
            c = list_num[i * self.w_batch]
            if i * self.w_batch + self.w_batch - 1 <= self.in_size_w - 1:
                c1 = list_num[i * 512 + 511]
            else:
                c1 = list_num[self.in_size_w - 1]
            reg_w_tmp1.set_as(c)
            reg_w_tmp2.set_as(c1)

            reg_w_out_begin[i].set_as(reg_w_tmp1)
            reg_w_out_end[i].set_as(reg_w_tmp2)

        self.location_h(reg_index_y, reg_cur_index)
        ub_input = self.tik_instance.Tensor(
            "float32", (self.w_batch, self.c_block_size),
            name="ub_input", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, self.nc1) as nc1_index:
            clear_ub(self.tik_instance, ub_output)
            reg_w_num = self.tik_instance.Scalar(dtype="int32")
            reg_w_value = self.tik_instance.Scalar(dtype="int32")
            reg_repeat = self.tik_instance.Scalar(dtype="int32")
            reg_w_num.set_as(0)
            with self.tik_instance.for_range(0, self.w_in_loop) as loop_index:
                reg_w_value.set_as(0)

                with self.tik_instance.if_scope(
                        loop_index != (self.w_in_loop - 1)):
                    self.tik_instance.data_move(
                        ub_input[0],
                        self.grads_gm[
                            (nc1_index * self.in_size_h +
                             reg_cur_index) * self.in_size_w *
                            self.c_block_size +
                            loop_index * self.w_batch * self.c_block_size],
                        0, 1, (self.w_batch * self.c_block_size + 7) // 8, 0,
                        0)
                    reg_w_tmp1.set_as(reg_w_out_begin[loop_index])
                    reg_w_tmp2.set_as(reg_w_out_end[loop_index])
                    with self.tik_instance.for_range(
                            reg_w_tmp1,
                            reg_w_tmp2 + 1) as w_out_index:
                        reg_repeat.set_as(w_offline_num[reg_w_num])

                        self.tik_instance.vadd(
                            16, ub_output[w_out_index * self.c_block_size],
                            ub_input[reg_w_value * 16],
                            ub_output[w_out_index * self.c_block_size],
                            reg_repeat, 1, 1, 1, 0, 2, 0)
                        reg_w_value.set_as(reg_w_value + reg_repeat)
                        reg_w_num.set_as(reg_w_num + 1)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        ub_input[0],
                        self.grads_gm[(nc1_index *
                                       self.in_size_h + reg_cur_index) *
                                      self.in_size_w * self.c_block_size +
                                      loop_index * self.w_batch *
                                      self.c_block_size],
                        0, 1, (self.w_in_tail) * 2, 0, 0)

                    reg_w_tmp1.set_as(reg_w_out_begin[loop_index])
                    reg_w_tmp2.set_as(reg_w_out_end[loop_index])
                    with self.tik_instance.for_range(
                            reg_w_tmp1, reg_w_tmp2 + 1) as w_out_index:
                        reg_repeat.set_as(w_offline_num[reg_w_num])
                        self.tik_instance.vadd(
                            16, ub_output[w_out_index * self.c_block_size],
                            ub_input[reg_w_value * 16],
                            ub_output[w_out_index * self.c_block_size],
                            reg_repeat, 1, 1, 1, 0, 2, 0)

                        reg_w_value.set_as(reg_w_value + reg_repeat)
                        reg_w_num.set_as(reg_w_num + 1)
            self.move_out(ub_output, nc1_index, reg_index_y)

    def onetime(self, list_w_num, reg_index_y, reg_cur_index,
                ub_output):
        """
        Loop once
        """

        self.location_h(reg_index_y, reg_cur_index)
        ub_input = self.tik_instance.Tensor(
            "float32", (self.w_batch, self.c_block_size),
            name="ub_input", scope=tik.scope_ubuf)
        i = 0
        with self.tik_instance.for_range(0, self.nc1) as nc1_index:
            clear_ub(self.tik_instance, ub_output)
            self.tik_instance.data_move(
                ub_input[0],
                self.grads_gm[
                    (nc1_index * self.in_size_h +
                     reg_cur_index) * self.in_size_w * self.c_block_size],
                0, 1, (self.w_in_tail) * 2, 0, 0)
            for w_out_index in range(0, self.out_size_w):
                self.tik_instance.vadd(
                    16, ub_output[w_out_index * self.c_block_size],
                    ub_input[i * 16],
                    ub_output[w_out_index * self.c_block_size],
                    list_w_num[w_out_index], 1, 1, 1, 0, 2, 0)
                i = i + list_w_num[w_out_index]
            self.move_out(ub_output, nc1_index, reg_index_y)

    # pylint: disable=too-many-locals, too-many-arguments,too-many-statements
    def fun_w_out_small(self, core_index, h_per_core, h_in_index):
        """
        funtion for other scene

        Parameters
        ----------
        core_index: int
            index of core
        h_per_core: int
            number of h in per core
        h_in_index: int
            index of h
        l1_xpos: tensor
            x pos in l1
        l1_xscale: tensor
            x scale in l1
        one_value_buf: tensor
            tensor of one value

        Returns
        -------
        none
        """
        reg_index_y = self.tik_instance.Scalar(dtype="int32")
        reg_cur_index = self.tik_instance.Scalar(dtype="int32")
        reg_cur_index.set_as(core_index * h_per_core + h_in_index)
        list_w_num = []
        list_num = []
        loop_index = 0
        repeat_num = []
        self.repeat_numpy(list_w_num, list_num, loop_index, repeat_num)
        ub_output = self.tik_instance.Tensor(
            "float32", (self.out_size_w, self.c_block_size),
            name="ub_output", scope=tik.scope_ubuf)
        if len(set(repeat_num)) == 1 and repeat_num[0] < 128:
            self.equal(list_w_num, reg_index_y, reg_cur_index, ub_output)
        elif self.w_in_loop == 1:
            self.onetime(list_w_num, reg_index_y, reg_cur_index, ub_output)
        else:
            self.not_equal(list_w_num, list_num, reg_index_y, reg_cur_index,
                           ub_output)
