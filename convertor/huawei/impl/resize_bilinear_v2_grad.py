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

resize_bilinear_v2_grad
"""

from te import tik
from topi.cce import util
from te import platform as tbe_platform

# parameters for vector instruct
MASK = 64
DSTSTRIDEM0 = 1
SRC0STRIDEM0 = 1
SRC1STRIDEM0 = 1
DSTSTRIDEM1 = 8
SRC0STRIDEM1 = 8
SRC1STRIDEM1 = 8

# get available ub size
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
# get available l1 size
L1_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L1_SIZE)


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


def _check_param(grads, images, kernel_name, align_corners, half_pixel_centers):
    """
    check parameters, if one is invalid, then raise error

    Parameters
    ----------
    grads : dict
        dict with keys(shape and dtype) of grads
    images : dict
        dict with keys(shape and dtype) of images
    align_corners : bool
        decide how to calculate for scale
    half_pixel_centers : bool
        decide how to calculate for location
    kernel_name : str
        kernel name, default value is "resize_bilinear_v2_grad"

    Returns
    -------
    None
    """
    if half_pixel_centers:
        if align_corners:
            raise RuntimeError("If half_pixel_centers is True, "
                               "align_corners must be False.")
    grads_shape = grads.get("shape")
    grads_dtype = grads.get("dtype")
    images_shape = images.get("shape")
    images_dtype = images.get("dtype")
    data_limit = ((1 << 31) - 1) // (4 if images_dtype == "float32" else 2)
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(grads_shape)
    util.check_shape_rule(images_shape)
    util.check_shape_size(grads_shape, data_limit)
    util.check_shape_size(images_shape, data_limit)
    check_list_grads = ("float32")
    check_list_images = ("float32")
    util.check_dtype_rule(grads_dtype.lower(), check_list_grads)
    util.check_dtype_rule(images_dtype.lower(), check_list_images)


# pylint: disable=unused-argument,invalid-name, unused-variable
def check_supported(grads, images, y, align_corners=False,
                    kernel_name="resize_bilinear_v2_grad"):
    """
    algorithm:resize_bilinear_v2_grad
    Operation for resize_bilinear_v2_grad

    Parameters
    ----------
    grads : dict
        dict with keys(shape and dtype) of grads
    images : dict
        dict with keys(shape and dtype) of images
    y : dict
        dict with keys(shape and dtype) of output
    align_corners : bool
        decide how to calculate for scale
    kernel_name : str
        kernel name, default value is "resize_bilinear_v2_grad"

    Returns
    -------
    check_supported: bool
    """
    grads_shape = grads.get("shape")
    format_grads = grads.get("format")
    if format_grads == "NHWC":
        in_size_h = grads_shape[1]
        in_size_w = grads_shape[2]
    elif format_grads in ("NCHW", "NC1HWC0"):
        in_size_h = grads_shape[2]
        in_size_w = grads_shape[3]
    else:
        raise RuntimeError("The format of grads is not supported")

    try:
        if in_size_h > 10000 or in_size_w > 10000:
            return False

    except RuntimeError as e:
        return False

    return True

# pylint: disable=unused-argument,too-many-lines,invalid-name,too-many-arguments
@util.check_input_type(dict, dict, dict, bool, bool, str)
def resize_bilinear_v2_grad(grads, images, y, align_corners=False, half_pixel_centers=False,
                            kernel_name="resize_bilinear_v2_grad"):
    """
    algorithm:resize_bilinear_v2_grad
    Operation for resize_bilinear_v2_grad

    Parameters
    ----------
    grads : dict
        dict with keys(shape and dtype) of grads
    images : dict
        dict with keys(shape and dtype) of images
    y : dict
        dict with keys(shape and dtype) of output
    align_corners : bool
        decide how to calculate for scale
    half_pixel_centers : bool
        decide how to calculate for location
    kernel_name : str
        kernel name, default value is "resize_bilinear_v2_grad"

    Returns
    -------
    None
    """
    _check_param(grads, images, kernel_name, align_corners, half_pixel_centers)
    resize_bilinear_grad_reslut = ResizeBilinearGrad(grads, images,
                                                     align_corners, half_pixel_centers)

    return resize_bilinear_grad_reslut.tik_instance_function(kernel_name)


# pylint: disable=too-many-instance-attributes
class ResizeBilinearGrad():
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

        Returns
        -------
        None
        """
        self.grads_shape = grads.get("shape")
        self.grads_dtype = grads.get("dtype").lower()
        self.images_shape = images.get("shape")
        self.images_dtype = images.get("dtype").lower()
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers
        self.tik_instance = tik.Tik()

        self.batch_size = self.grads_shape[0]
        self.c1_size = self.grads_shape[1]
        self.in_size_h = self.grads_shape[2]
        self.in_size_w = self.grads_shape[3]
        self.c_block_size = self.grads_shape[4]
        self.nc1 = self.batch_size*self.c1_size

        self.out_size_h = self.images_shape[2]
        self.out_size_w = self.images_shape[3]

        self.w_in_loop = _ceil_div(self.in_size_w, 256)
        self.w_in_tail = self.in_size_w % 256
        if self.w_in_tail == 0:
            self.w_in_tail = 256
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

        self.scale_h = float(height_out)/float(height_in)
        self.scale_w = float(weight_out)/float(weight_in)
        self.weight_out = weight_out
        self.weight_in = weight_in
        output_gm_shape = (self.batch_size, self.c1_size, self.images_shape[2],
                           self.images_shape[3], self.c_block_size)
        self.grads_gm = self.tik_instance.Tensor(self.grads_dtype,
                                                 self.grads_shape,
                                                 name="grads_gm",
                                                 scope=tik.scope_gm)
        self.images_gm = self.tik_instance.Tensor(self.images_dtype,
                                                  self.images_shape,
                                                  name="images_gm",
                                                  scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.images_dtype,
                                                  output_gm_shape,
                                                  name="output_gm",
                                                  scope=tik.scope_gm,
                                                  is_atomic_add=True)

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

        core_counts = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        h_per_core = self.out_size_h // core_counts + \
                     (1 if self.out_size_h % core_counts > 0 else 0)
        if (self.out_size_h % core_counts == 0) or \
                (self.out_size_h % h_per_core == 0):
            is_same_core = 0
        else:
            is_same_core = 1

        core_num = self.out_size_h // h_per_core + \
                   (0 if self.out_size_h // core_counts == 0
                    else is_same_core)
        # for special shape
        if self.images_shape == (2, 1, 48, 72, 16) and \
                self.grads_shape == (2, 1, 768, 1152, 16) and not self.half_pixel_centers:
            self.fun_special()
        # (N,M) to (N,M)
        elif self.out_size_h == self.in_size_h and \
                self.out_size_w == self.in_size_w:
            with self.tik_instance.for_range(0, core_num, block_num=core_num) \
                    as core_index:
                with self.tik_instance.if_scope(core_index != core_num - 1):
                    with self.tik_instance.for_range(0, h_per_core) \
                            as h_out_index:
                        with self.tik_instance.for_range(0, self.nc1) \
                                as nc1_index:
                            self.fun_in_and_out_same(
                                nc1_index, core_index, h_per_core, h_out_index)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(
                            0, self.out_size_h - (core_num - 1)*h_per_core) \
                            as h_out_index:
                        with self.tik_instance.for_range(0, self.nc1) \
                                as nc1_index:
                            self.fun_in_and_out_same(
                                nc1_index, core_index, h_per_core, h_out_index)
        # (N,M) to (1,1)
        elif self.out_size_h == 1 and self.out_size_w == 1:
            # caculate block number
            nc1 = self.batch_size*self.c1_size
            nc1_size = nc1 // core_counts + (1 if nc1 % core_counts > 0 else 0)
            if (nc1 % core_counts == 0) or (nc1 % nc1_size == 0):
                is_same_core = 0
            else:
                is_same_core = 1

            block_dim = nc1 // nc1_size + (0 if nc1 // core_counts == 0
                                           else is_same_core)
            with self.tik_instance.for_range(0, block_dim, block_num=block_dim)\
                    as block_index:
                if self.in_size_w*self.in_size_h*self.c_block_size*4 < UB_SIZE \
                        and self.in_size_h < 256 and self.in_size_w*2 < 256:
                    self.fun_n_to_one_small(block_index, block_dim, nc1_size)

                else:
                    self.fun_n_to_one_big(block_index, block_dim, nc1_size)
        # (1,1) to (N,M)
        elif self.in_size_h == 1 and self.in_size_w == 1 and not self.half_pixel_centers:
            with self.tik_instance.for_range(0, core_num, block_num=core_num) \
                    as core_index:
                with self.tik_instance.if_scope(core_index == 0):
                    with self.tik_instance.for_range(0, self.nc1) as nc1_index:
                        self.fun_one_to_n(core_index, nc1_index, h_per_core)
        # (N,M) to (N1,M1) and w_out small
        elif self.in_size_w > self.out_size_w and self.out_size_w <= 600 \
                and self.scale_w*255 > 1:
            h_per_core = self.in_size_h // core_counts + (
                1 if self.in_size_h % core_counts > 0 else 0)
            if (self.in_size_h % core_counts == 0) \
                    or (self.in_size_h % h_per_core == 0):
                is_same_core = 0
            else:
                is_same_core = 1

            core_num = self.in_size_h // h_per_core + (
                0 if self.in_size_h // core_counts == 0 else is_same_core)
            with self.tik_instance.for_range(0, core_num, block_num=core_num) \
                    as core_index:
                l1_xpos = self.tik_instance.Tensor(
                    "int32", (self.w_in_loop*256, 8), name="l1_xpos",
                    scope=tik.scope_cbuf)
                l1_xscale = self.tik_instance.Tensor(
                    "float32", (self.w_in_loop*512, 8), name="l1_xscale",
                    scope=tik.scope_cbuf)
                self.fun_location(l1_xpos, l1_xscale)
                one_value_buf = self.tik_instance.Tensor(
                    "float32", (8, 8), name="one_value_buf",
                    scope=tik.scope_ubuf)
                self.tik_instance.vector_dup(MASK, one_value_buf, float(1),
                                             1, 1, 8)
                with self.tik_instance.if_scope(core_index != core_num - 1):
                    with self.tik_instance.for_range(0, h_per_core) \
                            as h_in_index:
                        self.fun_w_out_small(core_index, h_per_core, h_in_index,
                                             l1_xpos, l1_xscale, one_value_buf)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(
                            0, self.in_size_h - core_index*h_per_core) \
                            as h_in_index:
                        self.fun_w_out_small(core_index, h_per_core, h_in_index,
                                             l1_xpos, l1_xscale, one_value_buf)
        # (N,M) to (N1,M1)
        else:
            # fun general
            h_per_core = self.in_size_h // core_counts + (
                1 if self.in_size_h % core_counts > 0 else 0)
            if (self.in_size_h % core_counts == 0) \
                    or (self.in_size_h % h_per_core == 0):
                is_same_core = 0
            else:
                is_same_core = 1

            core_num = self.in_size_h // h_per_core + (
                0 if self.in_size_h // core_counts == 0 else is_same_core)

            with self.tik_instance.for_range(0, core_num, block_num=core_num) \
                    as core_index:
                l1_xpos = self.tik_instance.Tensor(
                    "int32", (self.w_in_loop*256, 8), name="l1_xpos",
                    scope=tik.scope_cbuf)
                l1_xscale = self.tik_instance.Tensor(
                    "float32", (self.w_in_loop*512, 8), name="l1_xscale",
                    scope=tik.scope_cbuf)
                self.fun_location(l1_xpos, l1_xscale)
                one_value_buf = self.tik_instance.Tensor(
                    "float32", (8, 8), name="one_value_buf",
                    scope=tik.scope_ubuf)
                self.tik_instance.vector_dup(MASK, one_value_buf, float(1),
                                             1, 1, 8)
                with self.tik_instance.if_scope(core_index != core_num - 1):
                    with self.tik_instance.for_range(0, h_per_core) \
                            as h_in_index:
                        self.fun_other(core_index, h_per_core, h_in_index,
                                       l1_xpos, l1_xscale, one_value_buf)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(
                            0, self.in_size_h - core_index*h_per_core) \
                            as h_in_index:
                        self.fun_other(core_index, h_per_core, h_in_index,
                                       l1_xpos, l1_xscale, one_value_buf)
        images_buf = self.tik_instance.Tensor("float32", (8,),
                                              name="one_value_buf",
                                              scope=tik.scope_ubuf)
        self.tik_instance.data_move(images_buf[0], self.images_gm[0],
                                    0, 1, 1, 0, 0)
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(self.grads_gm, self.images_gm),
                                   outputs=(self.output_gm))
        return self.tik_instance

    def fun_location(self, l1_xpos, l1_xscale):
        """
        funtion for location in w direction

        Parameters
        ----------
        l1_xpos: tensor
        l1_xscale: tensor

        Returns
        -------
        none
        """
        const_1 = self.tik_instance.Tensor("float32", (8, 8),
                                           name="const_1",
                                           scope=tik.scope_ubuf)
        const_0 = self.tik_instance.Tensor("float32", (8, 8),
                                           name="const_0",
                                           scope=tik.scope_ubuf)
        index_256 = self.tik_instance.Tensor("float32", (256, 8),
                                             name="index_256",
                                             scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(MASK, const_1, float(1), 1, 1, 8)
        self.tik_instance.vector_dup(MASK, const_0, 0, 1, 1, 8)
        int32_256_ub = self.tik_instance.Tensor("int32", (256, 8),
                                                name="int32_256_ub",
                                                scope=tik.scope_ubuf)
        scale_512_x = self.tik_instance.Tensor("float32", (512, 8),
                                               name="scale_512_x",
                                               scope=tik.scope_ubuf)
        const_weight = self.tik_instance.Tensor("float32", (8, 8),
                                                name="const_weight",
                                                scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(MASK, const_weight, float(self.weight_in), 1, 1, 8)
        #x zuobiao
        with self.tik_instance.for_range(0, self.w_in_loop) as w_index:
            with self.tik_instance.for_range(0, 256) as num_index:
                self.tik_instance.vector_dup(8, index_256[num_index*8],
                                             w_index*256+num_index, 1, 1, 8)
            if self.half_pixel_centers:
                self.tik_instance.vadds(MASK, index_256, index_256, float(0.5),
                                        32, 1, 1, 8, 8)
            self.tik_instance.vmuls(MASK, scale_512_x, index_256, float(self.weight_out),
                                    32, 1, 1, 8, 8)
            self.tik_instance.vdiv(MASK, scale_512_x, scale_512_x, const_weight,
                                   32, 1, 1, 1, 8, 8, 0)
            if self.half_pixel_centers:
                self.tik_instance.vadds(MASK, scale_512_x, scale_512_x, float(-0.5),
                                        32, 1, 1, 8, 8)
                self.tik_instance.vmax(MASK, scale_512_x[0], scale_512_x[0], const_0[0],
                                       32, 1, 1, 1, 8, 8, 0)
            self.tik_instance.vconv(MASK, "floor", int32_256_ub[0],
                                    scale_512_x[0], 32, 1, 1, 8, 8)
            self.tik_instance.vconv(MASK, "", scale_512_x[2048],
                                    int32_256_ub[0], 32, 1, 1, 8, 8)
            self.tik_instance.data_move(l1_xpos[w_index*256*8], int32_256_ub[0],
                                        0, 1, 256, 0, 0)
            self.tik_instance.vsub(MASK, scale_512_x[2048],
                                   scale_512_x[0], scale_512_x[2048],
                                   32, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vsub(MASK, scale_512_x[0],
                                   const_1[0], scale_512_x[2048],
                                   32, 1, 1, 1, 8, 0, 8)
            self.tik_instance.data_move(l1_xscale[w_index*512*8],
                                        scale_512_x[0], 0, 1, 512, 0, 0)

    def fun_in_and_out_same(self, nc1_index, core_index, h_per_core,
                            h_out_index):
        """
        funtion for same in and same out

        Parameters
        ----------
        nc1_index: int
            index of n*c1
        core_index: int
            index of core
        h_per_core: int
            h number of per core
        h_out_index: int
            indec of put index of h

        Returns
        -------
        none
        """
        if self.in_size_w*self.c_block_size*4 < UB_SIZE/2:
            ub_output = self.tik_instance.Tensor(
                "float32", (self.in_size_w, self.c_block_size),
                name="ub_output", scope=tik.scope_ubuf)
            self.tik_instance.data_move(
                ub_output[0], self.grads_gm[(nc1_index*self.in_size_h +
                                             core_index*h_per_core +
                                             h_out_index)*self.in_size_w*16],
                0, 1, self.in_size_w*2, 0, 0)
            self.tik_instance.data_move(
                self.output_gm[(nc1_index*self.out_size_h + core_index *
                                h_per_core + h_out_index)*self.out_size_w*16],
                ub_output[0], 0, 1, self.out_size_w*2, 0, 0)
        else:
            w_size_ub = UB_SIZE // (2*4*self.c_block_size)
            ub_output = self.tik_instance.Tensor(
                "float32", (w_size_ub, self.c_block_size),
                name="ub_output", scope=tik.scope_ubuf)
            w_num_ub = _ceil_div(self.in_size_w, w_size_ub)
            if w_num_ub > 1:
                thread_num = 2
            else:
                thread_num = 1

            with self.tik_instance.for_range(
                    0, w_num_ub, thread_num=thread_num) as w_num_index:
                with self.tik_instance.if_scope(w_num_index != w_num_ub - 1):
                    self.tik_instance.data_move(
                        ub_output[0], self.grads_gm[
                            ((nc1_index*self.in_size_h + core_index*h_per_core +
                              h_out_index)*self.in_size_w + w_num_index *
                             w_size_ub)*16], 0, 1, w_size_ub*2, 0, 0)
                    self.tik_instance.data_move(
                        self.output_gm[
                            ((nc1_index*self.out_size_h + core_index*h_per_core
                              + h_out_index)*self.out_size_w + w_num_index *
                             w_size_ub)*16], ub_output[0], 0, 1, w_size_ub*2,
                        0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        ub_output[0],
                        self.grads_gm[((nc1_index*self.in_size_h +
                                        core_index*h_per_core + h_out_index) *
                                       self.in_size_w + w_num_index*w_size_ub) *
                                      16],
                        0, 1, (self.in_size_w - w_num_index*w_size_ub)*2, 0, 0)

                    self.tik_instance.data_move(
                        self.output_gm[((nc1_index*self.out_size_h +
                                         core_index*h_per_core + h_out_index) *
                                        self.out_size_w + w_num_index *
                                        w_size_ub)*16], ub_output[0], 0, 1,
                        (self.in_size_w - w_num_index*w_size_ub)*2, 0, 0)

    def fun_n_to_one_big(self, block_index, block_dim, nc1_size):
        """
        funtion for one to n with big shape

        Parameters
        ----------
        block_index: int
            index of core
        block_dim: int
            number of block
        nc1_size: int
            size of n*c1

        Returns
        -------
        none
        """
        ub_output_tmp = self.tik_instance.Tensor(
            "float32", (4, self.c_block_size), name="ub_output_tmp",
            scope=tik.scope_ubuf)
        ub_output = self.tik_instance.Tensor(
            "float32", (1, self.c_block_size), name="ub_output",
            scope=tik.scope_ubuf)
        ub_input = self.tik_instance.Tensor(
            "float32", (240*4, self.c_block_size), name="ub_input",
            scope=tik.scope_ubuf)
        input_num = _ceil_div(self.in_size_h*self.in_size_w*16, 240*64)
        if input_num > 1:
            thread_num = 2
        else:
            thread_num = 1

        nc1 = self.batch_size*self.c1_size
        with self.tik_instance.if_scope(block_index != block_dim - 1):
            with self.tik_instance.for_range(0, nc1_size) as nc1_index:
                self.tik_instance.vector_dup(MASK, ub_output_tmp, 0.0, 1, 1, 8)
                self.tik_instance.vector_dup(16, ub_output, 0.0, 1, 1, 8)
                with self.tik_instance.for_range(
                        0, input_num, thread_num=thread_num) as input_index:
                    with self.tik_instance.if_scope(
                            input_index != input_num - 1):
                        self.tik_instance.data_move(
                            ub_input[0],
                            self.grads_gm[(block_index*nc1_size + nc1_index) *
                                          self.in_size_h*self.in_size_w*16 +
                                          input_index*64*240],
                            0, 1, 8*240, 0, 0)
                        self.tik_instance.vadd(MASK, ub_output_tmp[0],
                                               ub_input[0], ub_output_tmp[0],
                                               240, 1, 1, 1, 0, 8, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(
                            ub_input[0],
                            self.grads_gm[(block_index*nc1_size + nc1_index) *
                                          self.in_size_h*self.in_size_w*16 +
                                          input_index*64*240],
                            0, 1, (self.in_size_h*self.in_size_w -
                                   input_index*4*240)*2, 0, 0)
                        with self.tik_instance.for_range(
                                0, self.in_size_h*self.in_size_w -
                                input_index*4*240) as tmp_index:
                            self.tik_instance.vadd(16, ub_output[0],
                                                   ub_input[tmp_index*16],
                                                   ub_output[0], 1, 1,
                                                   1, 1, 0, 2, 0)
                        self.tik_instance.vadd(16, ub_output[0],
                                               ub_output_tmp[0],
                                               ub_output[0],
                                               4, 1, 1, 1, 0, 2, 0)
                self.tik_instance.data_move(
                    self.output_gm[(block_index*nc1_size + nc1_index)*16],
                    ub_output[0], 0, 1, 2, 0, 0)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, nc1 - (block_dim - 1)*nc1_size)\
                    as nc1_index:
                self.tik_instance.vector_dup(MASK, ub_output_tmp, 0.0, 1, 1, 8)
                self.tik_instance.vector_dup(16, ub_output, 0.0, 1, 1, 8)
                with self.tik_instance.for_range(
                        0, input_num, thread_num=thread_num) as input_index:
                    with self.tik_instance.if_scope(
                            input_index != input_num - 1):
                        self.tik_instance.data_move(
                            ub_input[0],
                            self.grads_gm[(block_index*nc1_size + nc1_index) *
                                          self.in_size_h*self.in_size_w*16 +
                                          input_index*64*240],
                            0, 1, 8*240, 0, 0)
                        self.tik_instance.vadd(MASK, ub_output_tmp[0],
                                               ub_input[0], ub_output_tmp[0],
                                               240, 1, 1, 1, 0, 8, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(
                            ub_input[0],
                            self.grads_gm[(block_index*nc1_size + nc1_index) *
                                          self.in_size_h*self.in_size_w*16 +
                                          input_index*64*240],
                            0, 1, (self.in_size_h*self.in_size_w -
                                   input_index*4*240)*2, 0, 0)
                        with self.tik_instance.for_range(
                                0, self.in_size_h*self.in_size_w -
                                input_index*4*240) as tmp_index:
                            self.tik_instance.vadd(16, ub_output[0],
                                                   ub_input[tmp_index*16],
                                                   ub_output[0], 1, 1,
                                                   1, 1, 0, 8, 0)
                        self.tik_instance.vadd(16, ub_output[0],
                                               ub_output_tmp[0],
                                               ub_output[0],
                                               4, 1, 1, 1, 0, 2, 0)
                self.tik_instance.data_move(
                    self.output_gm[(block_index*nc1_size + nc1_index)*16],
                    ub_output[0], 0, 1, 2, 0, 0)

    def fun_n_to_one_small(self, block_index, block_dim, nc1_size):
        """
        funtion for one to n with small shape

        Parameters
        ----------
        block_index: int
            index of core
        block_dim: int
            number of block
        nc1_size: int
            size of n*c1

        Returns
        -------
        none
        """
        nc1 = self.batch_size*self.c1_size
        in_size_w_num = _ceil_div(self.in_size_w, 4)
        with self.tik_instance.if_scope(block_index != block_dim - 1):
            with self.tik_instance.for_range(0, nc1_size) as nc1_index:
                ub_input = self.tik_instance.Tensor(
                    "float32", (self.in_size_h, self.in_size_w,
                                self.c_block_size),
                    name="ub_input", scope=tik.scope_ubuf)
                self.tik_instance.data_move(
                    ub_input[0], self.grads_gm[(block_index*nc1_size +
                                                nc1_index) * self.in_size_h *
                                               self.in_size_w*16],
                    0, 1, self.in_size_h*self.in_size_w*2, 0, 0)
                if in_size_w_num > 1:
                    with self.tik_instance.for_range(0, in_size_w_num) \
                            as w_in_index:
                        with self.tik_instance.if_scope(
                                w_in_index != in_size_w_num - 1):
                            if self.in_size_h != 1:
                                self.tik_instance.vadd(
                                    MASK, ub_input[w_in_index*64],
                                    ub_input[w_in_index*64 + self.in_size_w*16],
                                    ub_input[w_in_index*64], self.in_size_h-1,
                                    1, 1, 1, 0, self.in_size_w*2, 0)
                            self.tik_instance.vadd(
                                16, ub_input[w_in_index*64],
                                ub_input[w_in_index*64+16],
                                ub_input[w_in_index*64], 3,
                                1, 1, 1, 0, 2, 0)
                        with self.tik_instance.else_scope():
                            if self.in_size_h != 1:
                                self.tik_instance.vadd((self.in_size_w - (
                                    in_size_w_num-1)*4)*16,
                                                       ub_input[w_in_index*64],
                                                       ub_input[w_in_index*64 +
                                                                self.in_size_w *
                                                                16],
                                                       ub_input[w_in_index*64],
                                                       self.in_size_h-1, 1,
                                                       1, 1, 0, self.in_size_w *
                                                       2, 0)
                            if self.in_size_w-(in_size_w_num-1)*4 > 1:
                                self.tik_instance.vadd(
                                    16, ub_input[w_in_index*64],
                                    ub_input[w_in_index*64+16],
                                    ub_input[w_in_index*64],
                                    self.in_size_w-(in_size_w_num-1)*4-1,
                                    1, 1, 1, 0, 2, 0)
                    self.tik_instance.vadd(
                        16, ub_input[0], ub_input[64], ub_input[0],
                        in_size_w_num-1, 1, 1, 1, 0, 8, 0)
                else:
                    if self.in_size_h != 1:
                        self.tik_instance.vadd(
                            self.in_size_w*16, ub_input[0],
                            ub_input[self.in_size_w*16],
                            ub_input[0], self.in_size_h-1, 1,
                            1, 1, 0, self.in_size_w*2, 0)
                    if self.in_size_w != 1:
                        self.tik_instance.vadd(
                            16, ub_input[0], ub_input[16],
                            ub_input[0], self.in_size_w-1,
                            1, 1, 1, 0, 2, 0)
                self.tik_instance.data_move(
                    self.output_gm[(block_index*nc1_size+nc1_index)*16],
                    ub_input[0], 0, 1, 2, 0, 0)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(
                    0, nc1 - (block_dim - 1)*nc1_size) as nc1_index:
                ub_input = self.tik_instance.Tensor(
                    "float32", (self.in_size_h, self.in_size_w,
                                self.c_block_size),
                    name="ub_output", scope=tik.scope_ubuf)
                self.tik_instance.data_move(
                    ub_input[0],
                    self.grads_gm[(block_index*nc1_size + nc1_index) *
                                  self.in_size_h*self.in_size_w*16],
                    0, 1, self.in_size_h*self.in_size_w*2, 0, 0)
                if in_size_w_num > 1:
                    with self.tik_instance.for_range(0, in_size_w_num) \
                            as w_in_index:
                        with self.tik_instance.if_scope(
                                w_in_index != in_size_w_num - 1):
                            if self.in_size_h != 1:
                                self.tik_instance.vadd(
                                    MASK, ub_input[w_in_index*64],
                                    ub_input[w_in_index*64 + self.in_size_w*16],
                                    ub_input[w_in_index*64], self.in_size_h-1,
                                    1, 1, 1, 0, self.in_size_w*2, 0)
                            self.tik_instance.vadd(
                                16, ub_input[w_in_index*64],
                                ub_input[w_in_index*64+16],
                                ub_input[w_in_index*64], 3, 1, 1, 1, 0, 2, 0)
                        with self.tik_instance.else_scope():
                            if self.in_size_h != 1:
                                self.tik_instance.vadd((self.in_size_w-(
                                    in_size_w_num-1)*4)*16,
                                                       ub_input[w_in_index*64],
                                                       ub_input[w_in_index*64 +
                                                                self.in_size_w *
                                                                16],
                                                       ub_input[w_in_index*64],
                                                       self.in_size_h-1, 1,
                                                       1, 1, 0, self.in_size_w *
                                                       2, 0)
                            if self.in_size_w-(in_size_w_num-1)*4 > 1:
                                self.tik_instance.vadd(
                                    16, ub_input[w_in_index*64],
                                    ub_input[w_in_index*64+16],
                                    ub_input[w_in_index*64],
                                    self.in_size_w-(in_size_w_num-1)*4-1,
                                    1, 1, 1, 0, 2, 0)
                    self.tik_instance.vadd(
                        16, ub_input[0], ub_input[64], ub_input[0],
                        in_size_w_num-1, 1, 1, 1, 0, 8, 0)
                else:
                    if self.in_size_h != 1:
                        self.tik_instance.vadd(
                            self.in_size_w*16, ub_input[0],
                            ub_input[self.in_size_w*16],
                            ub_input[0], self.in_size_h-1, 1,
                            1, 1, 0, self.in_size_w*2, 0)
                    if self.in_size_w != 1:
                        self.tik_instance.vadd(
                            16, ub_input[0], ub_input[16], ub_input[0],
                            self.in_size_w-1, 1, 1, 1, 0, 2, 0)
                self.tik_instance.data_move(
                    self.output_gm[(block_index*nc1_size+nc1_index)*16],
                    ub_input[0], 0, 1, 2, 0, 0)

    def fun_one_to_n(self, core_index, nc1_index, h_per_core):
        """
        funtion for one to n

        Parameters
        ----------
        block_index: int
            index of core
        block_dim: int
            number of block
        nc1_size: int
            size of n*c1

        Returns
        -------
        none
        """
        ub_output = self.tik_instance.Tensor(
            "float32", (1, self.c_block_size),
            name="ub_output", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, h_per_core) as h_out_index:
            with self.tik_instance.if_scope(
                    tik.all(core_index == 0, h_out_index == 0)):
                self.tik_instance.data_move(
                    ub_output[0], self.grads_gm[nc1_index*16],
                    0, 1, 2, 0, 0)
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(
                    self.output_gm[(nc1_index*self.out_size_h +
                                    core_index*h_per_core + h_out_index) *
                                   self.out_size_w*16], ub_output[0], 0, 1,
                    2, 0, 0)
                self.tik_instance.set_atomic_add(0)

    # pylint: disable=too-many-locals, too-many-arguments,too-many-statements
    def fun_w_out_small(self, core_index, h_per_core, h_in_index,
                        l1_xpos, l1_xscale, one_value_buf):
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
        reg_cur_index.set_as(core_index*h_per_core + h_in_index)
        list_w_num = []
        loop_index = 0
        while loop_index < self.w_in_loop:
            if loop_index != self.w_in_loop - 1:
                order = 256*loop_index
                list_w_fp = []
                list_w_int = []
                if self.half_pixel_centers:
                    while order < 256*(loop_index+1):
                        list_w_fp.append(float(order))
                        list_w_int.append(int(max((float(order) + 0.5)*self.scale_w - 0.5, 0)))
                        order += 1
                else:
                    while order < 256*(loop_index+1):
                        list_w_fp.append(float(order))
                        list_w_int.append(int(float(order)*self.scale_w))
                        order += 1
                list_w_int_new = list(set(list_w_int))
                list_w_int_new.sort()
                for i in list_w_int_new:
                    list_w_num.append(list_w_int.count(i))
            else:
                order = 256*loop_index
                list_w_fp = []
                list_w_int = []
                if self.half_pixel_centers:
                    while order < self.in_size_w:
                        list_w_fp.append(float(order))
                        list_w_int.append(int(max((float(order) + 0.5)*self.scale_w - 0.5, 0)))
                        order += 1
                else:
                    while order < self.in_size_w:
                        list_w_fp.append(float(order))
                        list_w_int.append(int(float(order)*self.scale_w))
                        order += 1
                list_w_int_new = list(set(list_w_int))
                list_w_int_new.sort()
                for i in list_w_int_new:
                    list_w_num.append(list_w_int.count(i))
            loop_index += 1

        w_offline_num = self.tik_instance.Tensor(
            "int32", (len(list_w_num),),
            name="w_offline_num", scope=tik.scope_ubuf)
        reg_w_tmp = self.tik_instance.Scalar(dtype="int32")
        number = 0
        for j in list_w_num:
            reg_w_tmp.set_as(j)
            w_offline_num[number].set_as(reg_w_tmp)
            number = number + 1

        out_size_w_num = _ceil_div(self.out_size_w, 4)
        ub_output = self.tik_instance.Tensor(
            "float32", (out_size_w_num*4, self.c_block_size),
            name="ub_output", scope=tik.scope_ubuf)
        ub_output_2 = self.tik_instance.Tensor(
            "float32", (out_size_w_num*4, self.c_block_size),
            name="ub_output_2", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, self.nc1) as nc1_index:
            self.tik_instance.vector_dup(
                MASK, ub_output, 0.0, out_size_w_num, 1, 8)
            self.tik_instance.vector_dup(
                MASK, ub_output_2, 0.0, out_size_w_num, 1, 8)
            h_floor_buf = self.tik_instance.Tensor("int32", (8,),
                                                   name="h_floor_buf",
                                                   scope=tik.scope_ubuf)
            h_floor_buf_fp = self.tik_instance.Tensor("float32", (8,),
                                                      name="h_floor_buf_fp",
                                                      scope=tik.scope_ubuf)
            h_scale_buf = self.tik_instance.Tensor("float32", (8,),
                                                   name="h_scale_buf",
                                                   scope=tik.scope_ubuf)
            h_block_buf = self.tik_instance.Tensor("int32", (8,),
                                                   name="h_scale_buf",
                                                   scope=tik.scope_ubuf)
            one_u_u_buf = self.tik_instance.Tensor("float32", (2, 8),
                                                   name="one_u_u_buf",
                                                   scope=tik.scope_ubuf)
            const_0 = self.tik_instance.Tensor("float32", (8,),
                                               name="const_0",
                                               scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(
                8, h_block_buf, reg_cur_index, 1, 1, 8)
            self.tik_instance.vconv(8, "", h_scale_buf[0],
                                    h_block_buf[0], 1, 1, 1, 8, 8)
            if self.half_pixel_centers:
                self.tik_instance.vadds(8, h_scale_buf, h_scale_buf,
                                        float(0.5), 1, 1, 1, 8, 8)
            self.tik_instance.vmuls(8, h_scale_buf, h_scale_buf,
                                    self.scale_h, 1, 1, 1, 8, 8)
            if self.half_pixel_centers:
                self.tik_instance.vector_dup(8, const_0, 0, 1, 1, 8)
                self.tik_instance.vadds(8, h_scale_buf, h_scale_buf,
                                        float(-0.5), 1, 1, 1, 8, 8)
                self.tik_instance.vmax(8, h_scale_buf[0], h_scale_buf[0], const_0[0],
                                       1, 1, 1, 1, 8, 8, 0)
            self.tik_instance.vconv(8, "floor", h_floor_buf[0],
                                    h_scale_buf[0], 1, 1, 1, 8, 8)
            self.tik_instance.vconv(8, "", h_floor_buf_fp[0],
                                    h_floor_buf[0], 1, 1, 1, 8, 8)
            self.tik_instance.vsub(8, one_u_u_buf[8],
                                   h_scale_buf[0], h_floor_buf_fp[0],
                                   1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vsub(8, one_u_u_buf[0],
                                   one_value_buf[0], one_u_u_buf[8],
                                   1, 1, 1, 1, 8, 8, 8)
            reg_index_y.set_as(h_floor_buf[0])

            one_out = self.tik_instance.Tensor(
                "float32", (4*256, self.c_block_size),
                name="one_out", scope=tik.scope_ubuf)
            scale_512_ub_x = self.tik_instance.Tensor(
                "float32", (512, 8), name="scale_512_ub_x",
                scope=tik.scope_ubuf)
            int32_256_ub_x = self.tik_instance.Tensor(
                "int32", (256, 8), name="int32_256_ub_x",
                scope=tik.scope_ubuf)
            uv_ub = self.tik_instance.Tensor(
                "float32", (4*256, 8), name="uv_ub", scope=tik.scope_ubuf)

            reg_w_out_begin = self.tik_instance.Scalar(dtype="int32")
            reg_w_out_end = self.tik_instance.Scalar(dtype="int32")
            reg_w_num = self.tik_instance.Scalar(dtype="int32")
            reg_w_value = self.tik_instance.Scalar(dtype="int32")
            reg_repeat = self.tik_instance.Scalar(dtype="int32")
            self.tik_instance.vector_dup(
                MASK, one_out[0], float(0), 128, 1, 8)
            self.tik_instance.vector_dup(
                MASK, one_out[8192], float(0), 128, 1, 8)
            reg_w_num.set_as(0)
            with self.tik_instance.for_range(0, self.w_in_loop) \
                    as loop_index:
                reg_w_value.set_as(0)
                self.tik_instance.data_move(
                    int32_256_ub_x, l1_xpos[loop_index*256*8], 0, 1,
                    256, 0, 0)
                self.tik_instance.data_move(
                    scale_512_ub_x, l1_xscale[loop_index*512*8], 0, 1,
                    512, 0, 0)
                with self.tik_instance.if_scope(
                        loop_index != self.w_in_loop - 1):
                    ub_input = self.tik_instance.Tensor(
                        "float32", (256, self.c_block_size),
                        name="ub_input", scope=tik.scope_ubuf)
                    self.tik_instance.data_move(
                        ub_input[0],
                        self.grads_gm[(nc1_index *
                                       self.in_size_h + reg_cur_index) *
                                      self.in_size_w*16 +
                                      loop_index*256*16],
                        0, 1, 512, 0, 0)

                    self.tik_instance.vmul(
                        MASK, uv_ub[0], one_u_u_buf[0], scale_512_ub_x[0],
                        64, 1, 0, 1, 8, 0, 8)
                    self.tik_instance.vmul(
                        MASK, uv_ub[2*256*8], one_u_u_buf[8],
                        scale_512_ub_x[0], 64, 1, 0, 1, 8, 0, 8)
                    with self.tik_instance.for_range(0, 2) as repeat_index:
                        self.tik_instance.vmul(
                            MASK, one_out[256*32*repeat_index],
                            uv_ub[256*16*repeat_index], ub_input[0],
                            32, 4, 1, 2, 32, 8, 16)
                        self.tik_instance.vmul(
                            MASK, one_out[256*32*repeat_index+8],
                            uv_ub[256*16*repeat_index], ub_input[8],
                            32, 4, 1, 2, 32, 8, 16)
                        self.tik_instance.vmul(
                            MASK, one_out[256*32*repeat_index+16],
                            uv_ub[256*16*repeat_index + 256*8], ub_input[0],
                            32, 4, 1, 2, 32, 8, 16)
                        self.tik_instance.vmul(
                            MASK, one_out[256*32*repeat_index+24],
                            uv_ub[256*16*repeat_index + 256*8], ub_input[8],
                            32, 4, 1, 2, 32, 8, 16)
                    reg_w_out_begin.set_as(int32_256_ub_x[0])
                    reg_w_out_end.set_as(int32_256_ub_x[2047])
                    with self.tik_instance.for_range(
                            reg_w_out_begin, reg_w_out_end + 1) as w_out_index:
                        reg_repeat.set_as(w_offline_num[reg_w_num])
                        with self.tik_instance.if_scope(
                                w_out_index != (self.out_size_w - 1)):
                            self.tik_instance.vadd(
                                32, ub_output[w_out_index*16],
                                one_out[reg_w_value*32],
                                ub_output[w_out_index*16],
                                reg_repeat, 1, 1, 1, 0, 4, 0)
                            self.tik_instance.vadd(
                                32, ub_output_2[w_out_index*16],
                                one_out[256*32+reg_w_value*32],
                                ub_output_2[w_out_index*16],
                                reg_repeat, 1, 1, 1, 0, 4, 0)
                        with self.tik_instance.else_scope():
                            self.tik_instance.vadd(
                                16, ub_output[w_out_index*16],
                                one_out[reg_w_value*32],
                                ub_output[w_out_index*16],
                                reg_repeat, 1, 1, 1, 0, 4, 0)
                            self.tik_instance.vadd(
                                16, ub_output[w_out_index*16],
                                one_out[reg_w_value*32+16],
                                ub_output[w_out_index*16],
                                reg_repeat, 1, 1, 1, 0, 4, 0)
                            self.tik_instance.vadd(
                                16, ub_output_2[w_out_index*16],
                                one_out[256*32+reg_w_value*32],
                                ub_output_2[w_out_index*16],
                                reg_repeat, 1, 1, 1, 0, 4, 0)
                            self.tik_instance.vadd(
                                16, ub_output_2[w_out_index*16],
                                one_out[256*32+reg_w_value*32+16],
                                ub_output_2[w_out_index*16],
                                reg_repeat, 1, 1, 1, 0, 4, 0)
                        reg_w_value.set_as(reg_w_value + reg_repeat)
                        reg_w_num.set_as(reg_w_num + 1)

                    self.tik_instance.vector_dup(
                        MASK, one_out[0], float(0), 128, 1, 8)
                    self.tik_instance.vector_dup(
                        MASK, one_out[8192], float(0), 128, 1, 8)
                with self.tik_instance.else_scope():
                    ub_input = self.tik_instance.Tensor(
                        "float32", (256, self.c_block_size),
                        name="ub_input", scope=tik.scope_ubuf)
                    self.tik_instance.data_move(
                        ub_input[0],
                        self.grads_gm[(nc1_index*self.in_size_h +
                                       reg_cur_index)*self.in_size_w*16 +
                                      loop_index*256*16],
                        0, 1, self.w_in_tail*2, 0, 0)

                    self.tik_instance.vmul(
                        MASK, uv_ub[0], one_u_u_buf[0], scale_512_ub_x[0],
                        64, 1, 0, 1, 8, 0, 8)
                    self.tik_instance.vmul(
                        MASK, uv_ub[2*256*8], one_u_u_buf[8],
                        scale_512_ub_x[0], 64, 1, 0, 1, 8, 0, 8)
                    with self.tik_instance.for_range(0, 2) as repeat_index:
                        self.tik_instance.vmul(
                            MASK, one_out[256*32*repeat_index],
                            uv_ub[256*16*repeat_index], ub_input[0],
                            32, 4, 1, 2, 32, 8, 16)
                        self.tik_instance.vmul(
                            MASK, one_out[256*32*repeat_index+8],
                            uv_ub[256*16*repeat_index], ub_input[8],
                            32, 4, 1, 2, 32, 8, 16)
                        self.tik_instance.vmul(
                            MASK, one_out[256*32*repeat_index+16],
                            uv_ub[256*16*repeat_index + 256*8], ub_input[0],
                            32, 4, 1, 2, 32, 8, 16)
                        self.tik_instance.vmul(
                            MASK, one_out[256*32*repeat_index+24],
                            uv_ub[256*16*repeat_index + 256*8], ub_input[8],
                            32, 4, 1, 2, 32, 8, 16)

                    reg_w_out_begin.set_as(int32_256_ub_x[0])
                    with self.tik_instance.for_range(
                            reg_w_out_begin, self.out_size_w) as w_out_index:
                        reg_repeat.set_as(w_offline_num[reg_w_num])
                        with self.tik_instance.if_scope(
                                w_out_index != (self.out_size_w - 1)):
                            self.tik_instance.vadd(
                                32, ub_output[w_out_index*16],
                                one_out[reg_w_value*32],
                                ub_output[w_out_index*16],
                                reg_repeat, 1, 1, 1, 0, 4, 0)
                            self.tik_instance.vadd(
                                32, ub_output_2[w_out_index*16],
                                one_out[256*32+reg_w_value*32],
                                ub_output_2[w_out_index*16],
                                reg_repeat, 1, 1, 1, 0, 4, 0)
                        with self.tik_instance.else_scope():
                            self.tik_instance.vadd(
                                16, ub_output[w_out_index*16],
                                one_out[reg_w_value*32],
                                ub_output[w_out_index*16],
                                reg_repeat, 1, 1, 1, 0, 4, 0)
                            self.tik_instance.vadd(
                                16, ub_output[w_out_index*16],
                                one_out[reg_w_value*32+16],
                                ub_output[w_out_index*16],
                                reg_repeat, 1, 1, 1, 0, 4, 0)
                            self.tik_instance.vadd(
                                16, ub_output_2[w_out_index*16],
                                one_out[256*32+reg_w_value*32],
                                ub_output_2[w_out_index*16],
                                reg_repeat, 1, 1, 1, 0, 4, 0)
                            self.tik_instance.vadd(
                                16, ub_output_2[w_out_index*16],
                                one_out[256*32+reg_w_value*32+16],
                                ub_output_2[w_out_index*16],
                                reg_repeat, 1, 1, 1, 0, 4, 0)
                        reg_w_value.set_as(reg_w_value + reg_repeat)
                        reg_w_num.set_as(reg_w_num + 1)
                    self.tik_instance.vector_dup(
                        MASK, one_out[0], float(0), 128, 1, 8)
                    self.tik_instance.vector_dup(
                        MASK, one_out[8192], float(0), 128, 1, 8)
            #move data output
            self.tik_instance.set_atomic_add(1)
            self.tik_instance.data_move(
                self.output_gm[(nc1_index*self.out_size_h + reg_index_y) *
                               self.out_size_w*self.c_block_size],
                ub_output[0], 0, 1, self.out_size_w*2, 0, 0)
            with self.tik_instance.if_scope(
                    reg_index_y != self.out_size_h - 1):
                self.tik_instance.data_move(
                    self.output_gm[(nc1_index*self.out_size_h +
                                    reg_index_y + 1) * self.out_size_w *
                                   self.c_block_size], ub_output_2[0],
                    0, 1, self.out_size_w*2, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(
                    self.output_gm[(nc1_index*self.out_size_h +
                                    reg_index_y) * self.out_size_w *
                                   self.c_block_size], ub_output_2[0],
                    0, 1, self.out_size_w*2, 0, 0)
            self.tik_instance.set_atomic_add(0)

    # pylint: disable=too-many-locals, too-many-arguments,too-many-statements
    def fun_other(self, core_index, h_per_core, h_in_index,
                  l1_xpos, l1_xscale, one_value_buf):
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
        reg_cur_index.set_as(core_index*h_per_core + h_in_index)
        if self.out_size_w <= 600:
            out_size_w_num = _ceil_div(self.out_size_w, 4)
            ub_output = self.tik_instance.Tensor(
                "float32", (out_size_w_num*4, self.c_block_size),
                name="ub_output", scope=tik.scope_ubuf)
            ub_output_2 = self.tik_instance.Tensor(
                "float32", (out_size_w_num*4, self.c_block_size),
                name="ub_output_2", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.nc1) as nc1_index:
                self.tik_instance.vector_dup(
                    MASK, ub_output, 0.0, out_size_w_num, 1, 8)
                self.tik_instance.vector_dup(
                    MASK, ub_output_2, 0.0, out_size_w_num, 1, 8)
                h_floor_buf = self.tik_instance.Tensor("int32", (8,),
                                                       name="h_floor_buf",
                                                       scope=tik.scope_ubuf)
                h_floor_buf_fp = self.tik_instance.Tensor("float32", (8,),
                                                          name="h_floor_buf_fp",
                                                          scope=tik.scope_ubuf)
                h_scale_buf = self.tik_instance.Tensor("float32", (8,),
                                                       name="h_scale_buf",
                                                       scope=tik.scope_ubuf)
                h_block_buf = self.tik_instance.Tensor("int32", (8,),
                                                       name="h_scale_buf",
                                                       scope=tik.scope_ubuf)
                one_u_u_buf = self.tik_instance.Tensor("float32", (2, 8),
                                                       name="one_u_u_buf",
                                                       scope=tik.scope_ubuf)
                const_0 = self.tik_instance.Tensor("float32", (8,),
                                                   name="const_0",
                                                   scope=tik.scope_ubuf)
                self.tik_instance.vector_dup(
                    8, h_block_buf, reg_cur_index, 1, 1, 8)
                self.tik_instance.vconv(8, "", h_scale_buf[0],
                                        h_block_buf[0], 1, 1, 1, 8, 8)
                if self.half_pixel_centers:
                    self.tik_instance.vadds(8, h_scale_buf, h_scale_buf,
                                            float(0.5), 1, 1, 1, 8, 8)
                self.tik_instance.vmuls(8, h_scale_buf, h_scale_buf,
                                        self.scale_h, 1, 1, 1, 8, 8)
                if self.half_pixel_centers:
                    self.tik_instance.vector_dup(8, const_0, 0, 1, 1, 8)
                    self.tik_instance.vadds(8, h_scale_buf, h_scale_buf,
                                            float(-0.5), 1, 1, 1, 8, 8)
                    self.tik_instance.vmax(8, h_scale_buf[0], h_scale_buf[0], const_0[0],
                                           1, 1, 1, 1, 8, 8, 0)
                self.tik_instance.vconv(8, "floor", h_floor_buf[0],
                                        h_scale_buf[0], 1, 1, 1, 8, 8)
                self.tik_instance.vconv(8, "", h_floor_buf_fp[0],
                                        h_floor_buf[0], 1, 1, 1, 8, 8)
                self.tik_instance.vsub(8, one_u_u_buf[8],
                                       h_scale_buf[0], h_floor_buf_fp[0],
                                       1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vsub(8, one_u_u_buf[0],
                                       one_value_buf[0], one_u_u_buf[8],
                                       1, 1, 1, 1, 8, 8, 8)
                reg_index_y.set_as(h_floor_buf[0])

                one_out = self.tik_instance.Tensor(
                    "float32", (4*256, self.c_block_size),
                    name="one_out", scope=tik.scope_ubuf)
                scale_512_ub_x = self.tik_instance.Tensor(
                    "float32", (512, 8), name="scale_512_ub_x",
                    scope=tik.scope_ubuf)
                int32_256_ub_x = self.tik_instance.Tensor(
                    "int32", (256, 8), name="int32_256_ub_x",
                    scope=tik.scope_ubuf)
                uv_ub = self.tik_instance.Tensor(
                    "float32", (4*256, 8), name="uv_ub", scope=tik.scope_ubuf)
                reg_index_x = self.tik_instance.Scalar(dtype="int32")
                self.tik_instance.vector_dup(
                    MASK, one_out[0], float(0), 128, 1, 8)
                self.tik_instance.vector_dup(
                    MASK, one_out[8192], float(0), 128, 1, 8)
                with self.tik_instance.for_range(0, self.w_in_loop) \
                        as loop_index:
                    self.tik_instance.data_move(
                        int32_256_ub_x, l1_xpos[loop_index*256*8], 0, 1,
                        256, 0, 0)
                    self.tik_instance.data_move(
                        scale_512_ub_x, l1_xscale[loop_index*512*8], 0, 1,
                        512, 0, 0)
                    with self.tik_instance.if_scope(
                            loop_index != self.w_in_loop - 1):
                        ub_input = self.tik_instance.Tensor(
                            "float32", (256, self.c_block_size),
                            name="ub_input", scope=tik.scope_ubuf)
                        self.tik_instance.data_move(
                            ub_input[0],
                            self.grads_gm[(nc1_index *
                                           self.in_size_h + reg_cur_index) *
                                          self.in_size_w*16 +
                                          loop_index*256*16],
                            0, 1, 512, 0, 0)

                        self.tik_instance.vmul(
                            MASK, uv_ub[0], one_u_u_buf[0], scale_512_ub_x[0],
                            64, 1, 0, 1, 8, 0, 8)
                        self.tik_instance.vmul(
                            MASK, uv_ub[2*256*8], one_u_u_buf[8],
                            scale_512_ub_x[0], 64, 1, 0, 1, 8, 0, 8)
                        with self.tik_instance.for_range(0, 2) as repeat_index:
                            self.tik_instance.vmul(
                                MASK, one_out[256*32*repeat_index],
                                uv_ub[256*16*repeat_index], ub_input[0],
                                32, 4, 1, 2, 32, 8, 16)
                            self.tik_instance.vmul(
                                MASK, one_out[256*32*repeat_index+8],
                                uv_ub[256*16*repeat_index], ub_input[8],
                                32, 4, 1, 2, 32, 8, 16)
                            self.tik_instance.vmul(
                                MASK, one_out[256*32*repeat_index+16],
                                uv_ub[256*16*repeat_index + 256*8], ub_input[0],
                                32, 4, 1, 2, 32, 8, 16)
                            self.tik_instance.vmul(
                                MASK, one_out[256*32*repeat_index+24],
                                uv_ub[256*16*repeat_index + 256*8], ub_input[8],
                                32, 4, 1, 2, 32, 8, 16)
                        with self.tik_instance.for_range(0, 256) as w_in_index:
                            reg_index_x.set_as(int32_256_ub_x[w_in_index*8])
                            with self.tik_instance.if_scope(
                                    reg_index_x != (self.out_size_w - 1)):
                                self.tik_instance.vadd(
                                    32, ub_output[reg_index_x*16],
                                    one_out[w_in_index*32],
                                    ub_output[reg_index_x*16],
                                    1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vadd(
                                    32, ub_output_2[reg_index_x*16],
                                    one_out[256*32+w_in_index*32],
                                    ub_output_2[reg_index_x*16],
                                    1, 1, 1, 1, 8, 8, 8)
                            with self.tik_instance.else_scope():
                                self.tik_instance.vadd(
                                    16, ub_output[reg_index_x*16],
                                    one_out[w_in_index*32],
                                    ub_output[reg_index_x*16],
                                    1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vadd(
                                    16, ub_output[reg_index_x*16],
                                    ub_output[reg_index_x*16],
                                    one_out[w_in_index*32+16],
                                    1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vadd(
                                    16, ub_output_2[reg_index_x*16],
                                    one_out[256*32+w_in_index*32],
                                    ub_output_2[reg_index_x*16],
                                    1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vadd(
                                    16, ub_output_2[reg_index_x*16],
                                    ub_output_2[reg_index_x*16],
                                    one_out[256*32+w_in_index*32+16],
                                    1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.vector_dup(
                            MASK, one_out[0], float(0), 128, 1, 8)
                        self.tik_instance.vector_dup(
                            MASK, one_out[8192], float(0), 128, 1, 8)
                    with self.tik_instance.else_scope():
                        ub_input = self.tik_instance.Tensor(
                            "float32", (256, self.c_block_size),
                            name="ub_input", scope=tik.scope_ubuf)
                        self.tik_instance.data_move(
                            ub_input[0],
                            self.grads_gm[(nc1_index*self.in_size_h +
                                           reg_cur_index)*self.in_size_w*16 +
                                          loop_index*256*16],
                            0, 1, self.w_in_tail*2, 0, 0)

                        self.tik_instance.vmul(
                            MASK, uv_ub[0], one_u_u_buf[0], scale_512_ub_x[0],
                            64, 1, 0, 1, 8, 0, 8)
                        self.tik_instance.vmul(
                            MASK, uv_ub[2*256*8], one_u_u_buf[8],
                            scale_512_ub_x[0], 64, 1, 0, 1, 8, 0, 8)
                        with self.tik_instance.for_range(0, 2) as repeat_index:
                            self.tik_instance.vmul(
                                MASK, one_out[256*32*repeat_index],
                                uv_ub[256*16*repeat_index], ub_input[0],
                                32, 4, 1, 2, 32, 8, 16)
                            self.tik_instance.vmul(
                                MASK, one_out[256*32*repeat_index+8],
                                uv_ub[256*16*repeat_index], ub_input[8],
                                32, 4, 1, 2, 32, 8, 16)
                            self.tik_instance.vmul(
                                MASK, one_out[256*32*repeat_index+16],
                                uv_ub[256*16*repeat_index + 256*8], ub_input[0],
                                32, 4, 1, 2, 32, 8, 16)
                            self.tik_instance.vmul(
                                MASK, one_out[256*32*repeat_index+24],
                                uv_ub[256*16*repeat_index + 256*8], ub_input[8],
                                32, 4, 1, 2, 32, 8, 16)

                        with self.tik_instance.for_range(0, self.w_in_tail) \
                                as w_in_index:
                            reg_index_x.set_as(int32_256_ub_x[w_in_index*8])
                            with self.tik_instance.if_scope(
                                    reg_index_x != (self.out_size_w - 1)):
                                self.tik_instance.vadd(
                                    32, ub_output[reg_index_x*16],
                                    one_out[w_in_index*32],
                                    ub_output[reg_index_x*16],
                                    1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vadd(
                                    32, ub_output_2[reg_index_x*16],
                                    one_out[256*32+w_in_index*32],
                                    ub_output_2[reg_index_x*16],
                                    1, 1, 1, 1, 8, 8, 8)
                            with self.tik_instance.else_scope():
                                self.tik_instance.vadd(
                                    16, ub_output[reg_index_x*16],
                                    one_out[w_in_index*32],
                                    ub_output[reg_index_x*16],
                                    1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vadd(
                                    16, ub_output[reg_index_x*16],
                                    ub_output[reg_index_x*16],
                                    one_out[w_in_index*32+16],
                                    1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vadd(
                                    16, ub_output_2[reg_index_x*16],
                                    one_out[256*32+w_in_index*32],
                                    ub_output_2[reg_index_x*16],
                                    1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vadd(
                                    16, ub_output_2[reg_index_x*16],
                                    ub_output_2[reg_index_x*16],
                                    one_out[256*32+w_in_index*32+16],
                                    1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.vector_dup(
                            MASK, one_out[0], float(0), 128, 1, 8)
                        self.tik_instance.vector_dup(
                            MASK, one_out[8192], float(0), 128, 1, 8)
                #move data output
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(
                    self.output_gm[(nc1_index*self.out_size_h + reg_index_y) *
                                   self.out_size_w*self.c_block_size],
                    ub_output[0], 0, 1, self.out_size_w*2, 0, 0)
                with self.tik_instance.if_scope(
                        reg_index_y != self.out_size_h - 1):
                    self.tik_instance.data_move(
                        self.output_gm[(nc1_index*self.out_size_h +
                                        reg_index_y + 1) * self.out_size_w *
                                       self.c_block_size], ub_output_2[0],
                        0, 1, self.out_size_w*2, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        self.output_gm[(nc1_index*self.out_size_h +
                                        reg_index_y) * self.out_size_w *
                                       self.c_block_size], ub_output_2[0],
                        0, 1, self.out_size_w*2, 0, 0)
                self.tik_instance.set_atomic_add(0)
        else:
            cut_num_w = _ceil_div(self.out_size_w, 600)
            ub_output = self.tik_instance.Tensor(
                "float32", (600, self.c_block_size),
                name="ub_output", scope=tik.scope_ubuf)
            ub_output_2 = self.tik_instance.Tensor(
                "float32", (600, self.c_block_size),
                name="ub_output_2", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.nc1) as nc1_index:
                with self.tik_instance.for_range(0, cut_num_w) as cut_w_index:
                    self.tik_instance.vector_dup(
                        MASK, ub_output[0], 0.0, 150, 1, 8)
                    self.tik_instance.vector_dup(
                        MASK, ub_output_2[0], 0.0, 150, 1, 8)
                    with self.tik_instance.if_scope(
                            cut_w_index != cut_num_w - 1):
                        h_floor_buf = self.tik_instance.Tensor(
                            "int32", (8,), name="h_floor_buf",
                            scope=tik.scope_ubuf)
                        h_floor_buf_fp = self.tik_instance.Tensor(
                            "float32", (8,), name="h_floor_buf_fp",
                            scope=tik.scope_ubuf)
                        h_scale_buf = self.tik_instance.Tensor(
                            "float32", (8,), name="h_scale_buf",
                            scope=tik.scope_ubuf)
                        h_block_buf = self.tik_instance.Tensor(
                            "int32", (8,), name="h_scale_buf",
                            scope=tik.scope_ubuf)
                        one_u_u_buf = self.tik_instance.Tensor(
                            "float32", (2, 8), name="one_u_u_buf",
                            scope=tik.scope_ubuf)
                        const_0 = self.tik_instance.Tensor("float32", (8,),
                                                           name="const_0",
                                                           scope=tik.scope_ubuf)
                        self.tik_instance.vector_dup(
                            8, h_block_buf, reg_cur_index, 1, 1, 8)
                        self.tik_instance.vconv(8, "", h_scale_buf[0],
                                                h_block_buf[0], 1, 1, 1, 8, 8)
                        if self.half_pixel_centers:
                            self.tik_instance.vadds(8, h_scale_buf, h_scale_buf,
                                                    float(0.5), 1, 1, 1, 8, 8)
                        self.tik_instance.vmuls(
                            8, h_scale_buf, h_scale_buf, self.scale_h, 1,
                            1, 1, 8, 8)
                        if self.half_pixel_centers:
                            self.tik_instance.vector_dup(8, const_0, 0, 1, 1, 8)
                            self.tik_instance.vadds(8, h_scale_buf, h_scale_buf,
                                                    float(-0.5), 1, 1, 1, 8, 8)
                            self.tik_instance.vmax(8, h_scale_buf[0], h_scale_buf[0], const_0[0],
                                                   1, 1, 1, 1, 8, 8, 0)
                        self.tik_instance.vconv(
                            8, "floor", h_floor_buf[0], h_scale_buf[0],
                            1, 1, 1, 8, 8)
                        self.tik_instance.vconv(8, "", h_floor_buf_fp[0],
                                                h_floor_buf[0], 1, 1, 1, 8, 8)
                        self.tik_instance.vsub(8, one_u_u_buf[8],
                                               h_scale_buf[0],
                                               h_floor_buf_fp[0],
                                               1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.vsub(8, one_u_u_buf[0],
                                               one_value_buf[0],
                                               one_u_u_buf[8],
                                               1, 1, 1, 1, 8, 8, 8)
                        reg_index_y.set_as(h_floor_buf[0])
                        #calc output
                        one_out = self.tik_instance.Tensor(
                            "float32", (4*256, self.c_block_size),
                            name="one_out", scope=tik.scope_ubuf)
                        scale_512_ub_x = self.tik_instance.Tensor(
                            "float32", (512, 8),
                            name="scale_512_ub_x", scope=tik.scope_ubuf)
                        int32_256_ub_x = self.tik_instance.Tensor(
                            "int32", (256, 8), name="int32_256_ub_x",
                            scope=tik.scope_ubuf)
                        uv_ub = self.tik_instance.Tensor(
                            "float32", (4*256, 8), name="uv_ub",
                            scope=tik.scope_ubuf)
                        reg_index_x = self.tik_instance.Scalar(dtype="int32")
                        reg_index_w = self.tik_instance.Scalar(dtype="int32")
                        self.tik_instance.vector_dup(
                            MASK, one_out[0], float(0), 128, 1, 8)
                        self.tik_instance.vector_dup(
                            MASK, one_out[8192], float(0), 128, 1, 8)
                        with self.tik_instance.for_range(0, self.w_in_loop) \
                                as loop_index:
                            self.tik_instance.data_move(
                                int32_256_ub_x, l1_xpos[loop_index*256*8],
                                0, 1, 256, 0, 0)
                            self.tik_instance.data_move(
                                scale_512_ub_x, l1_xscale[loop_index*512*8],
                                0, 1, 512, 0, 0)
                            ub_input = self.tik_instance.Tensor(
                                "float32", (256, self.c_block_size),
                                name="ub_input", scope=tik.scope_ubuf)
                            with self.tik_instance.if_scope(
                                    loop_index != self.w_in_loop - 1):
                                self.tik_instance.data_move(
                                    ub_input[0],
                                    self.grads_gm[(nc1_index*self.in_size_h +
                                                   reg_cur_index) *
                                                  self.in_size_w*16 +
                                                  loop_index*256*16],
                                    0, 1, 512, 0, 0)

                                self.tik_instance.vmul(
                                    MASK, uv_ub[0], one_u_u_buf[0],
                                    scale_512_ub_x[0],
                                    64, 1, 0, 1, 8, 0, 8)
                                self.tik_instance.vmul(
                                    MASK, uv_ub[2*256*8], one_u_u_buf[8],
                                    scale_512_ub_x[0],
                                    64, 1, 0, 1, 8, 0, 8)
                                with self.tik_instance.for_range(0, 2) \
                                        as repeat_index:
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index],
                                        uv_ub[256*16*repeat_index], ub_input[0],
                                        32, 4, 1, 2, 32, 8, 16)
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index+8],
                                        uv_ub[256*16*repeat_index],
                                        ub_input[8], 32, 4, 1, 2, 32, 8, 16)
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index+16],
                                        uv_ub[256*16*repeat_index + 256*8],
                                        ub_input[0], 32, 4, 1, 2, 32, 8, 16)
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index+24],
                                        uv_ub[256*16*repeat_index + 256*8],
                                        ub_input[8], 32, 4, 1, 2, 32, 8, 16)
                                with self.tik_instance.for_range(0, 256) \
                                        as w_in_index:
                                    reg_index_x.set_as(
                                        int32_256_ub_x[w_in_index*8])
                                    reg_index_w.set_as(reg_index_x % 600)
                                    with self.tik_instance.if_scope(
                                            tik.all(
                                                reg_index_x >= cut_w_index*600,
                                                reg_index_x < (cut_w_index + 1) *
                                                600)):
                                        with self.tik_instance.if_scope(
                                                reg_index_w != 599):
                                            self.tik_instance.vadd(
                                                32, ub_output[reg_index_w*16],
                                                one_out[w_in_index*32],
                                                ub_output[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                            self.tik_instance.vadd(
                                                32, ub_output_2[reg_index_w*16],
                                                one_out[256*32+w_in_index*32],
                                                ub_output_2[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                        with self.tik_instance.else_scope():
                                            self.tik_instance.vadd(
                                                16, ub_output[reg_index_w*16],
                                                one_out[w_in_index*32],
                                                ub_output[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                            self.tik_instance.vadd(
                                                16, ub_output_2[reg_index_w*16],
                                                one_out[256*32+w_in_index*32],
                                                ub_output_2[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                    with self.tik_instance.if_scope(
                                            reg_index_x == cut_w_index*600 - 1):
                                        self.tik_instance.vadd(
                                            16, ub_output[0],
                                            one_out[w_in_index*32+16],
                                            ub_output[0], 1, 1, 1, 1, 8, 8, 8)
                                        self.tik_instance.vadd(
                                            16, ub_output_2[0],
                                            one_out[256*32+w_in_index*32+16],
                                            ub_output_2[0], 1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vector_dup(
                                    MASK, one_out[0], float(0), 128, 1, 8)
                                self.tik_instance.vector_dup(
                                    MASK, one_out[8192], float(0), 128, 1, 8)
                            with self.tik_instance.else_scope():
                                self.tik_instance.data_move(
                                    ub_input[0],
                                    self.grads_gm[(nc1_index*self.in_size_h +
                                                   reg_cur_index) *
                                                  self.in_size_w*16 +
                                                  loop_index*256*16],
                                    0, 1, self.w_in_tail*2, 0, 0)

                                self.tik_instance.vmul(
                                    MASK, uv_ub[0], one_u_u_buf[0],
                                    scale_512_ub_x[0], 64, 1, 0, 1, 8, 0, 8)
                                self.tik_instance.vmul(
                                    MASK, uv_ub[2*256*8], one_u_u_buf[8],
                                    scale_512_ub_x[0], 64, 1, 0, 1, 8, 0, 8)
                                with self.tik_instance.for_range(0, 2) \
                                        as repeat_index:
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index],
                                        uv_ub[256*16*repeat_index], ub_input[0],
                                        32, 4, 1, 2, 32, 8, 16)
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index+8],
                                        uv_ub[256*16*repeat_index], ub_input[8],
                                        32, 4, 1, 2, 32, 8, 16)
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index+16],
                                        uv_ub[256*16*repeat_index + 256*8],
                                        ub_input[0], 32, 4, 1, 2, 32, 8, 16)
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index+24],
                                        uv_ub[256*16*repeat_index + 256*8],
                                        ub_input[8], 32, 4, 1, 2, 32, 8, 16)

                                with self.tik_instance.for_range(
                                        0, self.w_in_tail) as w_in_index:
                                    reg_index_x.set_as(
                                        int32_256_ub_x[w_in_index*8])
                                    reg_index_w.set_as(reg_index_x % 600)
                                    with self.tik_instance.if_scope(
                                            tik.all(
                                                reg_index_x >= cut_w_index*600,
                                                reg_index_x < (cut_w_index + 1) *
                                                600)):
                                        with self.tik_instance.if_scope(
                                                reg_index_w != 599):
                                            self.tik_instance.vadd(
                                                32, ub_output[reg_index_w*16],
                                                one_out[w_in_index*32],
                                                ub_output[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                            self.tik_instance.vadd(
                                                32, ub_output_2[reg_index_w*16],
                                                one_out[256*32+w_in_index*32],
                                                ub_output_2[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                        with self.tik_instance.else_scope():
                                            self.tik_instance.vadd(
                                                16, ub_output[reg_index_w*16],
                                                one_out[w_in_index*32],
                                                ub_output[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                            self.tik_instance.vadd(
                                                16, ub_output_2[reg_index_w*16],
                                                one_out[256*32+w_in_index*32],
                                                ub_output_2[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                    with self.tik_instance.if_scope(
                                            reg_index_x == cut_w_index*600 - 1):
                                        self.tik_instance.vadd(
                                            16, ub_output[0], ub_output[0],
                                            one_out[w_in_index*32+16],
                                            1, 1, 1, 1, 8, 8, 8)
                                        self.tik_instance.vadd(
                                            16, ub_output_2[0],
                                            one_out[256*32+w_in_index*32+16],
                                            ub_output_2[0], 1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vector_dup(
                                    MASK, one_out[0], float(0), 128, 1, 8)
                                self.tik_instance.vector_dup(
                                    MASK, one_out[8192], float(0), 128, 1, 8)
                        #move data output
                        self.tik_instance.set_atomic_add(1)
                        self.tik_instance.data_move(
                            self.output_gm[((nc1_index*self.out_size_h +
                                             reg_index_y) *
                                            self.out_size_w + cut_w_index*600) *
                                           self.c_block_size], ub_output[0],
                            0, 1, 600*2, 0, 0)
                        with self.tik_instance.if_scope(
                                reg_index_y != self.out_size_h - 1):
                            self.tik_instance.data_move(
                                self.output_gm[((nc1_index*self.out_size_h +
                                                 reg_index_y + 1) *
                                                self.out_size_w +
                                                cut_w_index*600) *
                                               self.c_block_size],
                                ub_output_2[0], 0, 1, 600*2, 0, 0)
                        with self.tik_instance.else_scope():
                            self.tik_instance.data_move(
                                self.output_gm[((nc1_index*self.out_size_h +
                                                 reg_index_y) *
                                                self.out_size_w +
                                                cut_w_index*600) *
                                               self.c_block_size],
                                ub_output_2[0], 0, 1, 600*2, 0, 0)
                        self.tik_instance.set_atomic_add(0)
                    with self.tik_instance.else_scope():
                        h_floor_buf = self.tik_instance.Tensor(
                            "int32", (8,), name="h_floor_buf",
                            scope=tik.scope_ubuf)
                        h_floor_buf_fp = self.tik_instance.Tensor(
                            "float32", (8,), name="h_floor_buf_fp",
                            scope=tik.scope_ubuf)
                        h_scale_buf = self.tik_instance.Tensor(
                            "float32", (8,), name="h_scale_buf",
                            scope=tik.scope_ubuf)
                        h_block_buf = self.tik_instance.Tensor(
                            "int32", (8,), name="h_scale_buf",
                            scope=tik.scope_ubuf)
                        one_u_u_buf = self.tik_instance.Tensor(
                            "float32", (2, 8), name="one_u_u_buf",
                            scope=tik.scope_ubuf)
                        const_0 = self.tik_instance.Tensor("float32", (8,),
                                                           name="const_0",
                                                           scope=tik.scope_ubuf)
                        self.tik_instance.vector_dup(
                            8, h_block_buf, reg_cur_index, 1, 1, 8)
                        self.tik_instance.vconv(8, "", h_scale_buf[0],
                                                h_block_buf[0], 1, 1, 1, 8, 8)
                        if self.half_pixel_centers:
                            self.tik_instance.vadds(8, h_scale_buf, h_scale_buf,
                                                    float(0.5), 1, 1, 1, 8, 8)
                        self.tik_instance.vmuls(8, h_scale_buf, h_scale_buf,
                                                self.scale_h, 1,
                                                1, 1, 8, 8)
                        if self.half_pixel_centers:
                            self.tik_instance.vector_dup(8, const_0, 0, 1, 1, 8)
                            self.tik_instance.vadds(8, h_scale_buf, h_scale_buf,
                                                    float(-0.5), 1, 1, 1, 8, 8)
                            self.tik_instance.vmax(8, h_scale_buf[0], h_scale_buf[0], const_0[0],
                                                   1, 1, 1, 1, 8, 8, 0)
                        self.tik_instance.vconv(8, "floor", h_floor_buf[0],
                                                h_scale_buf[0], 1, 1, 1, 8, 8)
                        self.tik_instance.vconv(8, "", h_floor_buf_fp[0],
                                                h_floor_buf[0], 1, 1, 1, 8, 8)
                        self.tik_instance.vsub(8, one_u_u_buf[8],
                                               h_scale_buf[0],
                                               h_floor_buf_fp[0],
                                               1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.vsub(8, one_u_u_buf[0],
                                               one_value_buf[0], one_u_u_buf[8],
                                               1, 1, 1, 1, 8, 8, 8)
                        reg_index_y.set_as(h_floor_buf[0])
                        #calc
                        one_out = self.tik_instance.Tensor(
                            "float32", (4*256, self.c_block_size),
                            name="one_out", scope=tik.scope_ubuf)
                        scale_512_ub_x = self.tik_instance.Tensor(
                            "float32", (512, 8),
                            name="scale_512_ub_x", scope=tik.scope_ubuf)
                        int32_256_ub_x = self.tik_instance.Tensor(
                            "int32", (256, 8),
                            name="int32_256_ub_x",
                            scope=tik.scope_ubuf)
                        uv_ub = self.tik_instance.Tensor(
                            "float32", (4*256, 8),
                            name="uv_ub", scope=tik.scope_ubuf)
                        reg_index_x = self.tik_instance.Scalar(dtype="int32")
                        reg_index_w = self.tik_instance.Scalar(dtype="int32")
                        self.tik_instance.vector_dup(
                            MASK, one_out[0], float(0), 128, 1, 8)
                        self.tik_instance.vector_dup(
                            MASK, one_out[8192], float(0), 128, 1, 8)
                        with self.tik_instance.for_range(0, self.w_in_loop) \
                                as loop_index:
                            self.tik_instance.data_move(
                                int32_256_ub_x, l1_xpos[loop_index*256*8],
                                0, 1, 256, 0, 0)
                            self.tik_instance.data_move(
                                scale_512_ub_x, l1_xscale[loop_index*512*8],
                                0, 1, 512, 0, 0)
                            ub_input = self.tik_instance.Tensor(
                                "float32", (256, self.c_block_size),
                                name="ub_input", scope=tik.scope_ubuf)
                            with self.tik_instance.if_scope(
                                    loop_index != self.w_in_loop - 1):
                                self.tik_instance.data_move(
                                    ub_input[0],
                                    self.grads_gm[(nc1_index*self.in_size_h +
                                                   reg_cur_index) *
                                                  self.in_size_w*16 +
                                                  loop_index*256*16],
                                    0, 1, 512, 0, 0)

                                self.tik_instance.vmul(
                                    MASK, uv_ub[0], one_u_u_buf[0],
                                    scale_512_ub_x[0],
                                    64, 1, 0, 1, 8, 0, 8)
                                self.tik_instance.vmul(
                                    MASK, uv_ub[2*256*8], one_u_u_buf[8],
                                    scale_512_ub_x[0],
                                    64, 1, 0, 1, 8, 0, 8)
                                with self.tik_instance.for_range(0, 2) \
                                        as repeat_index:
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index],
                                        uv_ub[256*16*repeat_index], ub_input[0],
                                        32, 4, 1, 2, 32, 8, 16)
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index+8],
                                        uv_ub[256*16*repeat_index], ub_input[8],
                                        32, 4, 1, 2, 32, 8, 16)
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index+16],
                                        uv_ub[256*16*repeat_index + 256*8],
                                        ub_input[0], 32, 4, 1, 2, 32, 8, 16)
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index+24],
                                        uv_ub[256*16*repeat_index + 256*8],
                                        ub_input[8], 32, 4, 1, 2, 32, 8, 16)
                                with self.tik_instance.for_range(0, 256) \
                                        as w_in_index:
                                    reg_index_x.set_as(
                                        int32_256_ub_x[w_in_index*8])
                                    reg_index_w.set_as(reg_index_x % 600)
                                    with self.tik_instance.if_scope(
                                            tik.all(
                                                reg_index_x >= cut_w_index*600,
                                                reg_index_x < (cut_w_index + 1) *
                                                600)):
                                        with self.tik_instance.if_scope(
                                                reg_index_x != (
                                                    self.out_size_w - 1)):
                                            self.tik_instance.vadd(
                                                32, ub_output[reg_index_w*16],
                                                one_out[w_in_index*32],
                                                ub_output[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                            self.tik_instance.vadd(
                                                32, ub_output_2[reg_index_w*16],
                                                one_out[256*32+w_in_index*32],
                                                ub_output_2[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                        with self.tik_instance.else_scope():
                                            self.tik_instance.vadd(
                                                16, ub_output[reg_index_w*16],
                                                one_out[w_in_index*32],
                                                ub_output[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                            self.tik_instance.vadd(
                                                16, ub_output[reg_index_w*16],
                                                one_out[w_in_index*32 + 16],
                                                ub_output[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                            self.tik_instance.vadd(
                                                16, ub_output_2[reg_index_w*16],
                                                one_out[256*32+w_in_index*32],
                                                ub_output_2[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                            self.tik_instance.vadd(
                                                16, ub_output_2[reg_index_w*16],
                                                one_out[
                                                    256*32+w_in_index*32+16],
                                                ub_output_2[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                    with self.tik_instance.if_scope(
                                            reg_index_x == cut_w_index*600 - 1):
                                        self.tik_instance.vadd(
                                            16, ub_output[0],
                                            one_out[w_in_index*32 + 16],
                                            ub_output[0], 1, 1, 1, 1, 8, 8, 8)
                                        self.tik_instance.vadd(
                                            16, ub_output_2[0],
                                            one_out[256*32 + w_in_index*32+16],
                                            ub_output_2[0], 1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vector_dup(
                                    MASK, one_out[0], float(0), 128, 1, 8)
                                self.tik_instance.vector_dup(
                                    MASK, one_out[8192], float(0), 128, 1, 8)
                            with self.tik_instance.else_scope():
                                self.tik_instance.data_move(
                                    ub_input[0],
                                    self.grads_gm[(nc1_index*self.in_size_h +
                                                   reg_cur_index) *
                                                  self.in_size_w*16 +
                                                  loop_index*256*16],
                                    0, 1, self.w_in_tail*2, 0, 0)

                                self.tik_instance.vmul(
                                    MASK, uv_ub[0], one_u_u_buf[0],
                                    scale_512_ub_x[0], 64, 1, 0, 1, 8, 0, 8)
                                self.tik_instance.vmul(
                                    MASK, uv_ub[2*256*8], one_u_u_buf[8],
                                    scale_512_ub_x[0], 64, 1, 0, 1, 8, 0, 8)
                                with self.tik_instance.for_range(0, 2) \
                                        as repeat_index:
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index],
                                        uv_ub[256*16*repeat_index], ub_input[0],
                                        32, 4, 1, 2, 32, 8, 16)
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index+8],
                                        uv_ub[256*16*repeat_index], ub_input[8],
                                        32, 4, 1, 2, 32, 8, 16)
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index+16],
                                        uv_ub[256*16*repeat_index + 256*8],
                                        ub_input[0], 32, 4, 1, 2, 32, 8, 16)
                                    self.tik_instance.vmul(
                                        MASK, one_out[256*32*repeat_index+24],
                                        uv_ub[256*16*repeat_index + 256*8],
                                        ub_input[8], 32, 4, 1, 2, 32, 8, 16)
                                with self.tik_instance.for_range(
                                        0, self.w_in_tail) as w_in_index:
                                    reg_index_x.set_as(
                                        int32_256_ub_x[w_in_index*8])
                                    reg_index_w.set_as(reg_index_x % 600)
                                    with self.tik_instance.if_scope(
                                            tik.all(
                                                reg_index_x >= cut_w_index*600,
                                                reg_index_x < (cut_w_index + 1) *
                                                600)):
                                        with self.tik_instance.if_scope(
                                                reg_index_x != (
                                                    self.out_size_w - 1)):
                                            self.tik_instance.vadd(
                                                32, ub_output[reg_index_w*16],
                                                one_out[w_in_index*32],
                                                ub_output[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                            self.tik_instance.vadd(
                                                32, ub_output_2[reg_index_w*16],
                                                one_out[256*32+w_in_index*32],
                                                ub_output_2[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                        with self.tik_instance.else_scope():
                                            self.tik_instance.vadd(
                                                16, ub_output[reg_index_w*16],
                                                one_out[w_in_index*32],
                                                ub_output[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                            self.tik_instance.vadd(
                                                16, ub_output[reg_index_w*16],
                                                one_out[w_in_index*32+16],
                                                ub_output[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                            self.tik_instance.vadd(
                                                16, ub_output_2[reg_index_w*16],
                                                one_out[256*32 + w_in_index*32],
                                                ub_output_2[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                            self.tik_instance.vadd(
                                                16, ub_output_2[reg_index_w*16],
                                                one_out[256*32 +
                                                        w_in_index*32 + 16],
                                                ub_output_2[reg_index_w*16],
                                                1, 1, 1, 1, 8, 8, 8)
                                    with self.tik_instance.if_scope(
                                            reg_index_x == cut_w_index*600 - 1):
                                        self.tik_instance.vadd(
                                            16, ub_output[0],
                                            one_out[w_in_index*32+16],
                                            ub_output[0], 1, 1, 1, 1, 8, 8, 8)
                                        self.tik_instance.vadd(
                                            16, ub_output_2[0],
                                            one_out[256*32+w_in_index*32+16],
                                            ub_output_2[0], 1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vector_dup(
                                    MASK, one_out[0], float(0), 128, 1, 8)
                                self.tik_instance.vector_dup(
                                    MASK, one_out[8192], float(0), 128, 1, 8)
                        #move data output
                        self.tik_instance.set_atomic_add(1)
                        self.tik_instance.data_move(
                            self.output_gm[((nc1_index*self.out_size_h +
                                             reg_index_y) *
                                            self.out_size_w +
                                            cut_w_index*600) *
                                           self.c_block_size],
                            ub_output[0], 0, 1, (self.out_size_w -
                                                 cut_w_index*600)*2, 0, 0)
                        with self.tik_instance.if_scope(
                                reg_index_y != self.out_size_h - 1):
                            self.tik_instance.data_move(
                                self.output_gm[((nc1_index*self.out_size_h +
                                                 reg_index_y + 1) *
                                                self.out_size_w +
                                                cut_w_index*600) *
                                               self.c_block_size],
                                ub_output_2[0], 0, 1, (self.out_size_w -
                                                       cut_w_index*600)*2, 0, 0)
                        with self.tik_instance.else_scope():
                            self.tik_instance.data_move(
                                self.output_gm[((nc1_index*self.out_size_h +
                                                 reg_index_y) *
                                                self.out_size_w +
                                                cut_w_index*600) *
                                               self.c_block_size],
                                ub_output_2[0], 0, 1, (self.out_size_w -
                                                       cut_w_index*600)*2, 0, 0)
                        self.tik_instance.set_atomic_add(0)

    # pylint: disable=too-many-locals,too-many-statements
    def fun_special(self):
        """
        funtion special shape

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        n_c1_h = self.nc1*self.out_size_h
        core_num = 32
        h_per_core = n_c1_h // core_num

        list_w_fp = []
        list_w_int = []
        list_w_num = []
        order = 0
        while order < self.in_size_w:
            list_w_fp.append(float(order))
            list_w_int.append(int(float(order)*self.scale_w))
            order = order + 1

        for i in set(list_w_int):
            list_w_num.append(list_w_int.count(i))

        with self.tik_instance.for_range(
                0, core_num, block_num=core_num) as core_index:
            w_offline_num = self.tik_instance.Tensor(
                "int32", (len(list_w_num),),
                name="w_offline_num", scope=tik.scope_ubuf)
            reg_w_tmp = self.tik_instance.Scalar(dtype="int32")
            number = 0
            for j in list_w_num:
                reg_w_tmp.set_as(j)
                w_offline_num[number].set_as(reg_w_tmp)
                number = number + 1

            one_value_buf = self.tik_instance.Tensor("float32", (8, 8),
                                                     name="one_value_buf",
                                                     scope=tik.scope_ubuf)
            ub_out = self.tik_instance.Tensor(
                "float32", (self.in_size_w, 8),
                name="ub_output", scope=tik.scope_ubuf)
            w_offline_buf_floor = self.tik_instance.Tensor(
                "int32", (self.in_size_w, 8),
                name="w_offline_buf_floor", scope=tik.scope_ubuf)
            w_offline_buf_floorfp = self.tik_instance.Tensor(
                "float32", (self.in_size_w, 8),
                name="w_offline_buf_floorfp", scope=tik.scope_ubuf)
            w_offline_1_v_v_buf = self.tik_instance.Tensor(
                "float32", (self.in_size_w, 16),
                name="w_offline_1_v_v_buf", scope=tik.scope_ubuf)

            self.tik_instance.vector_dup(MASK, one_value_buf, float(1), 1, 1, 8)

            with self.tik_instance.for_range(0, self.in_size_w) as num_index:
                self.tik_instance.vector_dup(
                    8, ub_out[num_index*8], num_index, 1, 1, 8)

            self.tik_instance.vmuls(MASK, ub_out, ub_out, self.scale_w, 144,
                                    1, 1, 8, 8)
            self.tik_instance.vconv(MASK, "floor", w_offline_buf_floor[0],
                                    ub_out[0], 144, 1, 1, 8, 8)
            self.tik_instance.vconv(MASK, "", w_offline_buf_floorfp[0],
                                    w_offline_buf_floor[0], 144, 1, 1, 8, 8)
            self.tik_instance.vsub(MASK, w_offline_1_v_v_buf[9216],
                                   ub_out[0], w_offline_buf_floorfp[0],
                                   144, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vsub(MASK, w_offline_1_v_v_buf[0],
                                   one_value_buf[0], w_offline_1_v_v_buf[9216],
                                   144, 1, 0, 1, 8, 0, 8)
            self.tik_instance.vector_dup(MASK, ub_out, float(0), 54, 1, 8)
            grad_l1 = self.tik_instance.Tensor(
                "float32", (10, 1152, 16),
                name="grad_l1", scope=tik.scope_cbuf)
            nc1_index = core_index // 16
            core_index2 = core_index % 16
            with self.tik_instance.for_range(0, 7) as h1_index:
                with self.tik_instance.if_scope((core_index % 16) != 0):
                    # move data to L1
                    with self.tik_instance.if_scope((core_index % 16) < 15):
                        with self.tik_instance.if_scope(h1_index < 6):
                            self.tik_instance.data_move(
                                grad_l1[0],
                                self.grads_gm[(nc1_index*self.in_size_h +
                                               core_index2*49 +
                                               h1_index*10 - 16) *
                                              self.in_size_w*16],
                                0, 1, 23040, 0, 0)
                        with self.tik_instance.else_scope():
                            self.tik_instance.data_move(
                                grad_l1[0],
                                self.grads_gm[(nc1_index*self.in_size_h +
                                               core_index2*49 + h1_index*10 -
                                               16)*self.in_size_w*16],
                                0, 1, 11520, 0, 0)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.if_scope(h1_index < 4):
                            self.tik_instance.data_move(
                                grad_l1[0],
                                self.grads_gm[(nc1_index*self.in_size_h +
                                               core_index2*49 +
                                               h1_index*10 - 16) *
                                              self.in_size_w*16],
                                0, 1, 23040, 0, 0)
                        with self.tik_instance.if_scope(h1_index == 4):
                            self.tik_instance.data_move(
                                grad_l1[0],
                                self.grads_gm[(nc1_index*self.in_size_h +
                                               core_index2*49 +
                                               h1_index*10 - 16) *
                                              self.in_size_w*16],
                                0, 1, 20736, 0, 0)
                    # calc
                    with self.tik_instance.for_range(0, 10) as h2_index:
                        with self.tik_instance.if_scope(
                                core_index2*49 +
                                h1_index*10 - 16 < 768 - h2_index):
                            h_ceil_buf = self.tik_instance.Tensor(
                                "int32", (8,), name="h_ceil_buf",
                                scope=tik.scope_ubuf)
                            h_floor_buf = self.tik_instance.Tensor(
                                "int32", (8,), name="h_floor_buf",
                                scope=tik.scope_ubuf)
                            h_floor_buf_fp = self.tik_instance.Tensor(
                                "float32", (8,), name="h_floor_buf_fp",
                                scope=tik.scope_ubuf)
                            h_scale_buf = self.tik_instance.Tensor(
                                "float32", (8,), name="h_scale_buf",
                                scope=tik.scope_ubuf)
                            h_block_buf = self.tik_instance.Tensor(
                                "int32", (8,), name="h_scale_buf",
                                scope=tik.scope_ubuf)
                            one_u_u_buf = self.tik_instance.Tensor(
                                "float32", (2, 8), name="one_u_u_buf",
                                scope=tik.scope_ubuf)
                            reg_index_h = self.tik_instance.Scalar(
                                dtype="int32")
                            reg_h0 = self.tik_instance.Scalar(dtype="int32")
                            reg_h1 = self.tik_instance.Scalar(dtype="int32")
                            reg_w0 = self.tik_instance.Scalar(dtype="int32")
                            reg_cur_index = self.tik_instance.Scalar(
                                dtype="int32")
                            reg_ub_h0 = self.tik_instance.Scalar(dtype="int32")
                            reg_ub_h1 = self.tik_instance.Scalar(dtype="int32")
                            reg_index_h.set_as(
                                core_index2*49 + h1_index*10 - 16 + h2_index)
                            self.tik_instance.vector_dup(
                                8, h_block_buf, reg_index_h, 1, 1, 8)
                            self.tik_instance.vconv(8, "", h_scale_buf[0],
                                                    h_block_buf[0],
                                                    1, 1, 1, 8, 8)
                            self.tik_instance.vmuls(8, h_scale_buf, h_scale_buf,
                                                    self.scale_h, 1,
                                                    1, 1, 8, 8)
                            self.tik_instance.vconv(8, "floor", h_floor_buf[0],
                                                    h_scale_buf[0],
                                                    1, 1, 1, 8, 8)
                            self.tik_instance.vconv(8, "ceil", h_ceil_buf[0],
                                                    h_scale_buf[0],
                                                    1, 1, 1, 8, 8)
                            self.tik_instance.vconv(8, "", h_floor_buf_fp[0],
                                                    h_floor_buf[0],
                                                    1, 1, 1, 8, 8)
                            self.tik_instance.vsub(8, one_u_u_buf[8],
                                                   h_scale_buf[0],
                                                   h_floor_buf_fp[0],
                                                   1, 1, 1, 1, 8, 8, 8)
                            self.tik_instance.vsub(8, one_u_u_buf[0],
                                                   one_value_buf[0],
                                                   one_u_u_buf[8],
                                                   1, 1, 1, 1, 8, 8, 8)
                            reg_h0.set_as(h_floor_buf[0])
                            reg_h1.set_as(h_ceil_buf[0])
                            with self.tik_instance.if_scope(reg_h1 > 47):
                                reg_h1.set_as(47)
                            with self.tik_instance.for_range(
                                    0, h_per_core) as n_c1_h_index:
                                reg_cur_index.set_as(
                                    core_index2*3 + n_c1_h_index)
                                with self.tik_instance.if_scope(
                                        tik.any(reg_h0 == reg_cur_index,
                                                reg_h1 == reg_cur_index)):
                                    reg_w_num = self.tik_instance.Scalar(
                                        dtype="int32")
                                    list_w_value = self.tik_instance.Scalar(
                                        dtype="int32")
                                    reg_w_num.set_as(0)
                                    with self.tik_instance.for_range(
                                            0, len(list_w_num)) as w_num_index:
                                        list_w_value.set_as(
                                            w_offline_num[w_num_index])
                                        grad_ub = self.tik_instance.Tensor(
                                            "float32", (24, 16), name="grad_ub",
                                            scope=tik.scope_ubuf)
                                        one_v_v_buf = self.tik_instance.Tensor(
                                            "float32", (24, 16),
                                            name="one_v_v_buf",
                                            scope=tik.scope_ubuf)
                                        uv_value_buf = self.tik_instance.Tensor(
                                            "float32", (24*2, 16),
                                            name="uv_value_buf",
                                            scope=tik.scope_ubuf)
                                        self.tik_instance.data_move(
                                            grad_ub[0],
                                            grad_l1[h2_index*1152*16 +
                                                    reg_w_num*16],
                                            0, 1, list_w_value*2, 0, 0)
                                        self.tik_instance.data_move(
                                            one_v_v_buf[0],
                                            w_offline_1_v_v_buf[reg_w_num*8],
                                            0, 2, list_w_value,
                                            1152 - list_w_value, 0)
                                        self.tik_instance.vmul(
                                            MASK, uv_value_buf[0],
                                            one_u_u_buf[0], one_v_v_buf[0],
                                            5, 1, 0, 1, 8, 0, 8)
                                        self.tik_instance.vmul(
                                            MASK, uv_value_buf[384],
                                            one_u_u_buf[8], one_v_v_buf[0],
                                            5, 1, 0, 1, 8, 0, 8)
                                        reg_w_num.set_as(
                                            reg_w_num + list_w_value)
                                        grad_tmp_buf = self.tik_instance.Tensor(
                                            "float32", (24*4, 16),
                                            name="grad_tmp_buf",
                                            scope=tik.scope_ubuf)
                                        ub_tmp_buf = self.tik_instance.Tensor(
                                            "float32", (4, 16),
                                            name="ub_tmp_buf",
                                            scope=tik.scope_ubuf)
                                        self.tik_instance.vector_dup(
                                            MASK, ub_tmp_buf, float(0), 1, 1, 8)
                                        with self.tik_instance.for_range(0, 4) \
                                                as repeat_index:
                                            with self.tik_instance.if_scope(
                                                    repeat_index <= 1):
                                                self.tik_instance.vmul(
                                                    MASK,
                                                    grad_tmp_buf[
                                                        24*16*repeat_index],
                                                    uv_value_buf[list_w_value * 8 *
                                                                 repeat_index],
                                                    grad_ub[0],
                                                    3, 2, 1, 2, 16, 8, 16)
                                                self.tik_instance.vmul(
                                                    MASK,
                                                    grad_tmp_buf[
                                                        24*16*repeat_index+8],
                                                    uv_value_buf[list_w_value * 8 *
                                                                 repeat_index],
                                                    grad_ub[8],
                                                    3, 2, 1, 2, 16, 8, 16)
                                            with self.tik_instance.else_scope():
                                                self.tik_instance.vmul(
                                                    MASK,
                                                    grad_tmp_buf[
                                                        24*16*repeat_index],
                                                    uv_value_buf[(24*2 + (
                                                        repeat_index - 2) *
                                                                  list_w_value) *
                                                                 8],
                                                    grad_ub[0],
                                                    3, 2, 1, 2, 16, 8, 16)
                                                self.tik_instance.vmul(
                                                    MASK,
                                                    grad_tmp_buf[
                                                        24*16*repeat_index+8],
                                                    uv_value_buf[(24*2 + (
                                                        repeat_index - 2) *
                                                                  list_w_value) *
                                                                 8],
                                                    grad_ub[8],
                                                    3, 2, 1, 2, 16, 8, 16)
                                        self.tik_instance.vadd(
                                            32, ub_tmp_buf[0],
                                            grad_tmp_buf[0], ub_tmp_buf[0],
                                            list_w_value, 2, 48, 2, 0, 2, 0)
                                        self.tik_instance.vadd(
                                            32, ub_tmp_buf[8],
                                            grad_tmp_buf[8], ub_tmp_buf[8],
                                            list_w_value, 2, 48, 2, 0, 2, 0)
                                        reg_w0.set_as(w_num_index)
                                        with self.tik_instance.if_scope(
                                                reg_h0 == reg_cur_index):
                                            reg_ub_h0.set_as(((reg_h0 * 72) +
                                                              reg_w0) * 16 -
                                                             core_index2*3456)
                                            self.tik_instance.vadd(
                                                32, ub_out[reg_ub_h0],
                                                ub_out[reg_ub_h0],
                                                ub_tmp_buf[0],
                                                1, 1, 1, 1, 8, 8, 8)
                                        with self.tik_instance.if_scope(
                                                reg_h1 == reg_cur_index):
                                            reg_ub_h1.set_as(((reg_h1 * 72) +
                                                              reg_w0) * 16 -
                                                             core_index2*3456)
                                            self.tik_instance.vadd(
                                                32, ub_out[reg_ub_h1],
                                                ub_out[reg_ub_h1],
                                                ub_tmp_buf[32],
                                                1, 1, 1, 1, 8, 8, 8)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        grad_l1[0],
                        self.grads_gm[(nc1_index*self.in_size_h +
                                       core_index2*49 + h1_index*7) *
                                      self.in_size_w*16],
                        0, 1, 16128, 0, 0)
                    with self.tik_instance.for_range(0, 7) as h2_index:
                        h_ceil_buf = self.tik_instance.Tensor(
                            "int32", (8,), name="h_ceil_buf",
                            scope=tik.scope_ubuf)
                        h_floor_buf = self.tik_instance.Tensor(
                            "int32", (8,), name="h_floor_buf",
                            scope=tik.scope_ubuf)
                        h_floor_buf_fp = self.tik_instance.Tensor(
                            "float32", (8,), name="h_floor_buf_fp",
                            scope=tik.scope_ubuf)
                        h_scale_buf = self.tik_instance.Tensor(
                            "float32", (8,), name="h_scale_buf",
                            scope=tik.scope_ubuf)
                        h_block_buf = self.tik_instance.Tensor(
                            "int32", (8,), name="h_scale_buf",
                            scope=tik.scope_ubuf)
                        one_u_u_buf = self.tik_instance.Tensor(
                            "float32", (2, 8), name="one_u_u_buf",
                            scope=tik.scope_ubuf)
                        reg_index_h = self.tik_instance.Scalar(dtype="int32")
                        reg_h0 = self.tik_instance.Scalar(dtype="int32")
                        reg_h1 = self.tik_instance.Scalar(dtype="int32")
                        reg_w0 = self.tik_instance.Scalar(dtype="int32")
                        reg_cur_index = self.tik_instance.Scalar(dtype="int32")
                        reg_ub_h0 = self.tik_instance.Scalar(dtype="int32")
                        reg_ub_h1 = self.tik_instance.Scalar(dtype="int32")
                        reg_index_h.set_as(
                            core_index2*49 + h1_index*7 + h2_index)
                        self.tik_instance.vector_dup(
                            8, h_block_buf, reg_index_h, 1, 1, 8)
                        self.tik_instance.vconv(8, "", h_scale_buf[0],
                                                h_block_buf[0], 1, 1, 1, 8, 8)
                        self.tik_instance.vmuls(8, h_scale_buf, h_scale_buf,
                                                self.scale_h, 1,
                                                1, 1, 8, 8)
                        self.tik_instance.vconv(8, "floor", h_floor_buf[0],
                                                h_scale_buf[0], 1, 1, 1, 8, 8)
                        self.tik_instance.vconv(8, "ceil", h_ceil_buf[0],
                                                h_scale_buf[0], 1, 1, 1, 8, 8)

                        self.tik_instance.vconv(8, "", h_floor_buf_fp[0],
                                                h_floor_buf[0], 1, 1, 1, 8, 8)
                        self.tik_instance.vsub(8, one_u_u_buf[8],
                                               h_scale_buf[0],
                                               h_floor_buf_fp[0],
                                               1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.vsub(8, one_u_u_buf[0],
                                               one_value_buf[0], one_u_u_buf[8],
                                               1, 1, 1, 1, 8, 8, 8)
                        reg_h0.set_as(h_floor_buf[0])
                        reg_h1.set_as(h_ceil_buf[0])
                        with self.tik_instance.if_scope(reg_h1 > 47):
                            reg_h1.set_as(47)
                        with self.tik_instance.for_range(0, h_per_core) \
                                as n_c1_h_index:
                            reg_cur_index.set_as(core_index2*3 + n_c1_h_index)
                            with self.tik_instance.if_scope(
                                    tik.any(reg_h0 == reg_cur_index,
                                            reg_h1 == reg_cur_index)):
                                reg_w_num = self.tik_instance.Scalar(
                                    dtype="int32")
                                list_w_value = self.tik_instance.Scalar(
                                    dtype="int32")
                                reg_w_num.set_as(0)
                                with self.tik_instance.for_range(
                                        0, len(list_w_num)) as w_num_index:
                                    list_w_value.set_as(
                                        w_offline_num[w_num_index])
                                    grad_ub = self.tik_instance.Tensor(
                                        "float32", (24, 16), name="grad_ub",
                                        scope=tik.scope_ubuf)
                                    one_v_v_buf = self.tik_instance.Tensor(
                                        "float32", (24, 16), name="one_v_v_buf",
                                        scope=tik.scope_ubuf)
                                    uv_value_buf = self.tik_instance.Tensor(
                                        "float32", (24*2, 16),
                                        name="uv_value_buf",
                                        scope=tik.scope_ubuf)
                                    self.tik_instance.data_move(
                                        grad_ub[0],
                                        grad_l1[h2_index*1152*16+reg_w_num*16],
                                        0, 1, list_w_value*2, 0, 0)
                                    self.tik_instance.data_move(
                                        one_v_v_buf[0],
                                        w_offline_1_v_v_buf[reg_w_num*8],
                                        0, 2, list_w_value,
                                        1152-list_w_value, 0)
                                    self.tik_instance.vmul(
                                        MASK, uv_value_buf[0], one_u_u_buf[0],
                                        one_v_v_buf[0],
                                        5, 1, 0, 1, 8, 0, 8)
                                    self.tik_instance.vmul(
                                        MASK, uv_value_buf[384], one_u_u_buf[8],
                                        one_v_v_buf[0],
                                        5, 1, 0, 1, 8, 0, 8)
                                    reg_w_num.set_as(reg_w_num + list_w_value)
                                    grad_tmp_buf = self.tik_instance.Tensor(
                                        "float32", (24*4, 16),
                                        name="grad_tmp_buf",
                                        scope=tik.scope_ubuf)
                                    ub_tmp_buf = self.tik_instance.Tensor(
                                        "float32", (4, 16), name="ub_tmp_buf",
                                        scope=tik.scope_ubuf)
                                    self.tik_instance.vector_dup(
                                        MASK, ub_tmp_buf, float(0), 1, 1, 8)
                                    with self.tik_instance.for_range(0, 4) \
                                            as repeat_index:
                                        with self.tik_instance.if_scope(
                                                repeat_index <= 1):
                                            self.tik_instance.vmul(
                                                MASK,
                                                grad_tmp_buf[24 * 16 *
                                                             repeat_index],
                                                uv_value_buf[list_w_value * 8 *
                                                             repeat_index],
                                                grad_ub[0],
                                                3, 2, 1, 2, 16, 8, 16)
                                            self.tik_instance.vmul(
                                                MASK,
                                                grad_tmp_buf[24 * 16 *
                                                             repeat_index+8],
                                                uv_value_buf[list_w_value * 8 *
                                                             repeat_index],
                                                grad_ub[8],
                                                3, 2, 1, 2, 16, 8, 16)
                                        with self.tik_instance.else_scope():
                                            self.tik_instance.vmul(
                                                MASK,
                                                grad_tmp_buf[24 * 16 *
                                                             repeat_index],
                                                uv_value_buf[(24*2+(
                                                    repeat_index-2) *
                                                              list_w_value)*8],
                                                grad_ub[0],
                                                3, 2, 1, 2, 16, 8, 16)
                                            self.tik_instance.vmul(
                                                MASK,
                                                grad_tmp_buf[24 * 16 *
                                                             repeat_index+8],
                                                uv_value_buf[(24*2+(
                                                    repeat_index-2) *
                                                              list_w_value)*8],
                                                grad_ub[8],
                                                3, 2, 1, 2, 16, 8, 16)
                                    self.tik_instance.vadd(
                                        32, ub_tmp_buf[0], grad_tmp_buf[0],
                                        ub_tmp_buf[0],
                                        list_w_value,
                                        2, 48, 2, 0, 2, 0)
                                    self.tik_instance.vadd(
                                        32, ub_tmp_buf[8], grad_tmp_buf[8],
                                        ub_tmp_buf[8],
                                        list_w_value, 2, 48, 2, 0, 2, 0)
                                    reg_w0.set_as(w_num_index)
                                    with self.tik_instance.if_scope(
                                            reg_h0 == reg_cur_index):
                                        reg_ub_h0.set_as(((reg_h0 * 72) +
                                                          reg_w0) * 16 -
                                                         core_index2*3456)
                                        self.tik_instance.vadd(
                                            32, ub_out[reg_ub_h0],
                                            ub_out[reg_ub_h0], ub_tmp_buf[0],
                                            1, 1, 1, 1, 8, 8, 8)
                                    with self.tik_instance.if_scope(
                                            reg_h1 == reg_cur_index):
                                        reg_ub_h1.set_as(((reg_h1 * 72) +
                                                          reg_w0) * 16 -
                                                         core_index2*3456)
                                        self.tik_instance.vadd(
                                            32, ub_out[reg_ub_h1],
                                            ub_out[reg_ub_h1], ub_tmp_buf[32],
                                            1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.data_move(self.output_gm[(core_index * 3)*72*16],
                                        ub_out[0], 0, 1, 432, 0, 0)
