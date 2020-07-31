#!/usr/bin/python
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

max_pool_with_argmax
"""

from te import tik
from topi.cce import util
from impl import max_pool_with_argmax_resnet50 as resnet50
from te import platform as tbe_platform

# min value of fp16
MIN_VALUE_FP16 = -65504.0
# define dilation size
DILATION = 1
# parameters for vector instruct
MASK = 128
REPEAT_2 = 2
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
    caculate ceil value of div

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


def _check_param(input_x, ksize, strides, padding, kernel_name):
    """
    check parameters, if one is invalid, then raise error

    Parameters
    ----------
    input_x: dict
        shape and datatype
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    padding: str
        value from `SAME`, `VALID`
    kernel_name: str

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_shape)
    util.check_tensor_shape_size(input_shape)
    util.check_dtype_rule(input_dtype, ("float16",))

    # the format of input_x must be NC1HWC0
    if len(input_shape) != 5:
        raise RuntimeError("invalid shape params, input feature map must be "
                           "5D format in kernel.")
    # get shape info of feature map in NC1HWC0 format
    in_size_h = input_shape[2]
    in_size_w = input_shape[3]
    c_block_size = input_shape[4]

    if c_block_size != 16:
        raise RuntimeError("invalid featur map shape params, "
                           "C0 must be equal to 16")

    if len(ksize) != 4:
        raise RuntimeError("Invalid ksize params, ksize dim must be 4.")

    if ksize[0] != 1 or ksize[3] != 1:
        raise RuntimeError("MaxPoolWithArgmax only supports pooling "
                           "across width/height, and other ksize "
                           "dimension should be one")
    if len(strides) != 4:
        raise RuntimeError("Invalid strides params, strides dim must be 4.")

    if strides[0] != 1 or strides[3] != 1:
        raise RuntimeError("MaxPoolWithArgmax only supports pooling across "
                           "width/height, and other strides dimension "
                           "should be one")

    if ksize[1] >= in_size_h or ksize[2] >= in_size_w:
        raise RuntimeError("can not support global pooling now")

    if ksize[1] * ksize[2] > 255:
        raise RuntimeError("invalid window params, window_h*window_w "
                           "should be <= 255")

    if padding not in ("SAME", "VALID"):
        raise RuntimeError("MaxPool can only support SAME or VALID "
                           "padding mode.")


# pylint: disable=too-many-arguments,unused-argument,too-many-lines
@util.check_input_type(dict, dict, dict, (list, tuple), (list, tuple), str, str)
def max_pool_with_argmax(input_x, output_y, output_argmax, ksize, strides,
                         padding, kernel_name="max_pool_with_argmax"):
    """
    Performs max pooling on the input and outputs both max values and indices.

    Parameters
    ----------
    input_x: dict
        shape and datatype
    output_y: dict
        The max pooled output tensor.
    output_argmax: dict
        the max values chosen for each output.
    ksize: list or tuple
        The size of the window for each dimension of the input tensor.
    strides: list or tuple
        The stride of the sliding window for each dimension of the input tensor.
    padding: str
        The type of padding algorithm to use.
    kernel_name: str
        kernel_name, default value is 'max_pool_with_argmax'

    Returns
    -------
    max_pool_reslut: reslut of maxpool
    """
    _check_param(input_x, ksize, strides, padding, kernel_name)
    if resnet50.is_max_pool_with_argmax_param(input_x, ksize, strides, padding):
        return resnet50.max_pool_with_argmax(input_x, ksize,
                                             strides, padding, kernel_name)
    max_pool_reslut = MaxPoolWithargmax(input_x, ksize, strides, padding)

    return max_pool_reslut.tik_instance_function(kernel_name)


# pylint: disable=too-many-instance-attributes,too-few-public-methods
class MaxPoolWithargmax():
    """
       Function: use to finish MaxPoolWithargmax main functions
       Modify : 2019-10-16
    """

    def __init__(self, input_x, ksize, strides, padding):
        """
        init MaxPoolWithargmax parameters

        Parameters
        ----------
        input_x: dict
            shape and datatype
        ksize: list or tuple
            The size of the window for each dimension of the input tensor.
        strides: list or tuple
            The stride of the sliding window of the input tensor.
        padding: str
            The type of padding algorithm to use.

        Returns
        -------
        None
        """
        self.input_shape = input_x.get("shape")
        self.input_dtype = input_x.get("dtype").lower()
        self.tik_instance = tik.Tik()

        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.batch_size = self.input_shape[0]
        self.c1_size = self.input_shape[1]
        self.in_size_h = self.input_shape[2]
        self.in_size_w = self.input_shape[3]
        self.c_block_size = self.input_shape[4]

        self.window_h = self.ksize[1]
        self.window_w = self.ksize[2]
        self.stride_h = self.strides[1]
        self.stride_w = self.strides[2]
        self.nc1 = self.batch_size * self.c1_size
        # scalar for load3d
        self.scalar_source_h = self.tik_instance.Scalar(dtype="int64")
        self.scalar_source_w = self.tik_instance.Scalar(dtype="int64")

        # caculate pad and output size
        self.pad, self.out_size_h, self.out_size_w = \
            self._calc_out_size_and_pad()
        # output_shape
        self.fmap_img2col_h = self.out_size_h * self.out_size_w
        self.fmap_img2col_w = self.window_h * self.window_w
        self.fmap_img2col_h_num = _ceil_div(self.fmap_img2col_h,
                                            self.c_block_size)
        mask_tmp = self.fmap_img2col_h_num * 16 - self.fmap_img2col_h
        self.mask_zero = 2 ** 16 - 2**(16 - mask_tmp)

        if self.input_dtype == "float16":
            self.pad_value = MIN_VALUE_FP16
        # famp is NC1HWC0 format
        fmap_gm_shape = (self.batch_size, self.c1_size, self.in_size_h,
                         self.in_size_w, self.c_block_size)

        output_gm_shape = (self.batch_size, self.c1_size, self.out_size_h,
                           self.out_size_w, self.c_block_size)
        output_mask_gm_shape = (
            self.batch_size, self.c1_size, self.fmap_img2col_w,
            (self.fmap_img2col_h_num + 1) * self.c_block_size)
        # input and output
        self.input_fmap_gm = self.tik_instance.Tensor(self.input_dtype,
                                                      fmap_gm_shape,
                                                      name="input_fmap_gm",
                                                      scope=tik.scope_gm)
        self.output_max_gm = self.tik_instance.Tensor(self.input_dtype,
                                                      output_gm_shape,
                                                      name="output_max_gm",
                                                      scope=tik.scope_gm)
        self.output_mask_gm = self.tik_instance.Tensor("uint16",
                                                       output_mask_gm_shape,
                                                       name="output_mask_gm",
                                                       scope=tik.scope_gm)

    # pylint: disable=too-many-locals, too-many-function-args
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
        # caculate if need cutH or cutW
        # caculate block number
        core_counts = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        need_cut_h, need_cut_h_w, need_cut = self._check_if_need_cut_h_w()
        if need_cut_h or need_cut:
            cut_h_size, cut_stride, cut_h_num = \
                self._calc_cut_h_size_fun(need_cut)
            flag_cut_h = False
            out_size_cut_h = \
                (cut_h_size - self.window_h + self.stride_h) // self.stride_h
            fmap_img2col_cut_h = self.out_size_w * out_size_cut_h
            if (fmap_img2col_cut_h % 16) == 0:
                flag_cut_h = True
                nc1_cuth = self.nc1 * cut_h_num
            else:
                nc1_cuth = self.nc1

            nc1_cuth_size = nc1_cuth // core_counts + (
                1 if nc1_cuth % core_counts > 0 else 0)
            if (nc1_cuth % core_counts == 0) or (nc1_cuth % nc1_cuth_size == 0):
                is_same_percore = 0
            else:
                is_same_percore = 1

            block_dim = nc1_cuth // nc1_cuth_size + (
                0 if nc1_cuth // core_counts == 0 else is_same_percore)
            with self.tik_instance. \
                for_range(0, block_dim, block_num=block_dim) \
                as block_index:
                with self.tik_instance.if_scope(
                        block_index != block_dim - 1):
                    with self.tik_instance.for_range(0, nc1_cuth_size) \
                        as nc1_cuth_index:
                        # size of ub is not enough, need cutH
                        if need_cut_h_w:
                            self._fun_need_cut_h_w(block_index,
                                                   nc1_cuth_index,
                                                   cut_h_size,
                                                   cut_stride,
                                                   cut_h_num,
                                                   nc1_cuth_size,
                                                   flag_cut_h)
                        else:
                            self._fun_only_cut_h(
                                block_index, nc1_cuth_index, cut_h_size,
                                cut_stride, cut_h_num, nc1_cuth_size, flag_cut_h)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(
                            0, nc1_cuth - (block_dim - 1) * nc1_cuth_size) \
                        as nc1_cuth_index:
                        # size of ub is not enough, need cutH
                        if need_cut_h_w:
                            self._fun_need_cut_h_w(block_index,
                                                   nc1_cuth_index,
                                                   cut_h_size,
                                                   cut_stride,
                                                   cut_h_num,
                                                   nc1_cuth_size, flag_cut_h)
                        else:
                            self._fun_only_cut_h(
                                block_index, nc1_cuth_index, cut_h_size,
                                cut_stride, cut_h_num, nc1_cuth_size, flag_cut_h)
        # no need cut
        else:
            nc1_size = self.nc1 // core_counts + (
                1 if self.nc1 % core_counts > 0 else 0)
            if (self.nc1 % core_counts == 0) or (self.nc1 % nc1_size == 0):
                is_same_percore = 0
            else:
                is_same_percore = 1

            block_dim = self.nc1 // nc1_size + (
                0 if self.nc1 // core_counts == 0 else is_same_percore)

            with self.tik_instance. \
                for_range(0, block_dim, block_num=block_dim) \
                as block_index:
                with self.tik_instance.if_scope(
                        block_index != block_dim - 1):
                    with self.tik_instance.for_range(0, nc1_size) \
                        as nc1_index:
                        self._fun_no_cut(block_index, nc1_index, nc1_size)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(
                            0, self.nc1 - (block_dim - 1) * nc1_size) \
                        as nc1_index:
                        self._fun_no_cut(block_index, nc1_index, nc1_size)

        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(self.input_fmap_gm),
                                   outputs=(self.output_max_gm,
                                            self.output_mask_gm))
        return self.tik_instance

    def _check_if_need_cut_h_w(self):
        """
        funtion check if need cutH or cutW

        Parameters
        ----------
            none

        Returns
        -------
        need_cut_h: bool
        need_cut_h_w: bool
        need_cut: bool

        """
        need_cut_h = False
        need_cut_h_w = False
        need_cut = False
        ub_size_used_max = self.out_size_h * self.out_size_w * 16 * \
                           self.window_h * self.window_w * 2
        ub_size_cut_h_max = self.out_size_w * 16 * self.window_h * self.window_w * 2

        if ub_size_used_max > (UB_SIZE / 2):
            need_cut_h = True

        if ub_size_cut_h_max > (UB_SIZE / 2):
            need_cut_h_w = True

        if self.window_h * self.in_size_w * self.c_block_size * 2 > L1_SIZE:
            raise RuntimeError(
                "cutC0 is needed and this scene is not supported")

        if not need_cut_h:
            if self.in_size_h * self.in_size_w * self.c_block_size * 2 > L1_SIZE:
                need_cut = True

        return need_cut_h, need_cut_h_w, need_cut

    # pylint: disable=too-many-locals
    def _fun_no_cut(self, block_index, nc1_index, nc1_size):
        """
        funtion while no need cut H

        Parameters
        ----------
        block_index: index of block
        nc1_index: index of nc1

        Returns
        -------
        none
        """

        fmap_l1_shape = (self.in_size_h, self.in_size_w, self.c_block_size)
        input_fmap_l1 = self.tik_instance.Tensor(self.input_dtype,
                                                 fmap_l1_shape,
                                                 name="input_fmap_l1",
                                                 scope=tik.scope_cbuf)
        fmap_img2col_shape_ub = (self.fmap_img2col_h_num * 16, self.window_h,
                                 self.window_w, self.c_block_size)
        fmap_img2col_ub = self.tik_instance.Tensor(self.input_dtype,
                                                   fmap_img2col_shape_ub,
                                                   name="fmap_img2col_ub",
                                                   scope=tik.scope_ubuf)
        mask_shape_ub = (self.window_h, self.window_w, self.fmap_img2col_h_num,
                         self.c_block_size)
        mask_ub = self.tik_instance.Tensor("uint16", mask_shape_ub,
                                           name="mask_ub", scope=tik.scope_ubuf)
        data_x_max = self.tik_instance.Tensor("float16",
                                              (self.fmap_img2col_h_num, 16, 16),
                                              name="data_x_max",
                                              scope=tik.scope_ubuf)
        # copy input fmap from gm to l1
        gm_l1_burst_len = int(self.in_size_h * self.in_size_w *
                              self.c_block_size // 16)
        self.tik_instance.data_move(
            input_fmap_l1,
            self.input_fmap_gm[(block_index * nc1_size + nc1_index) *
                               self.in_size_h * self.in_size_w *
                               self.c_block_size],
            0, 1, gm_l1_burst_len, 0, 0)
        with self.tik_instance.for_range(0, self.fmap_img2col_h_num) as h_index:
            source_h = (((h_index * 256 * self.fmap_img2col_w) /
                         (16 * self.fmap_img2col_w)) / self.out_size_w) * \
                       self.stride_h - self.pad[2]
            source_w = (((h_index * 256 * self.fmap_img2col_w) /
                         (16 * self.fmap_img2col_w)) % self.out_size_w) * \
                       self.stride_w - self.pad[0]
            self.scalar_source_h.set_as(source_h)
            self.scalar_source_w.set_as(source_w)
            self.tik_instance.load3dv1(
                fmap_img2col_ub[h_index * 256 * self.fmap_img2col_w],
                input_fmap_l1[0], self.pad,
                self.in_size_h, self.in_size_w, 0, 0, 0, self.scalar_source_w,
                self.scalar_source_h, self.stride_w, self.stride_h,
                self.window_w, self.window_h, 1, 1, 1, 0,
                self.fmap_img2col_w, 0, self.pad_value)
        if self.fmap_img2col_w != 1:
            self._calc_max_and_mask(self.fmap_img2col_h_num, fmap_img2col_ub,
                                    data_x_max, mask_ub)
            # move max output to gm
            self.tik_instance.data_move(
                self.output_max_gm[(block_index * nc1_size + nc1_index) *
                                   self.out_size_h * self.out_size_w *
                                   self.c_block_size],
                data_x_max[0], 0, 1, self.fmap_img2col_h, 0, 0)
            self._remove_repeated_fun(mask_ub)
        else:
            # move max output to gm
            self.tik_instance.data_move(
                self.output_max_gm[(block_index * nc1_size + nc1_index) *
                                   self.out_size_h * self.out_size_w *
                                   self.c_block_size],
                fmap_img2col_ub[0], 0, 1, self.fmap_img2col_h, 0, 0)
            self._dup_mask_fun(mask_ub, mask_shape_ub)

        with self.tik_instance.for_range(0, self.fmap_img2col_w) as w_index:
            offset_output_mask = \
                (block_index * nc1_size + nc1_index) * \
                (self.fmap_img2col_h_num + 1) * self.fmap_img2col_w * \
                self.c_block_size
            if self.mask_zero != 0 and self.fmap_img2col_w != 1:
                self.tik_instance.vector_dup([0, self.mask_zero], mask_ub[
                    w_index * self.fmap_img2col_h_num * self.c_block_size +
                    self.fmap_img2col_h_num * 16 - 16], 0, 1, 1, 8)

            self.tik_instance.data_move(
                self.output_mask_gm[
                    offset_output_mask + w_index *
                    (self.fmap_img2col_h_num + 1) * 16],
                mask_ub[w_index * self.fmap_img2col_h_num * self.c_block_size],
                0, 1, self.fmap_img2col_h_num, 0, 0)

    # pylint: disable=too-many-locals,too-many-statements
    def _calc_only_cut_h(self, cut_h_index, cut_h_size, cut_stride, cut_h_num,
                         input_fmap_l1, fmap_img2col_ub, fmap_img2col_cut_h,
                         mask_shape_ub, nc1_num):
        """
        calc only cut H

        Parameters
        ----------
        cut_h_index: index of cuth
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        input_fmap_l1: fmag in l1
        fmap_img2col_ub: fmag in ub
        fmap_img2col_cut_h: fmag cutH
        mask_shape_ub: shape of mask
        nc1_num: num of n*c1

        Returns
        -------
        none
        """
        fmap_img2col_cut_h_num = _ceil_div(fmap_img2col_cut_h, 16)
        mask_ub = self.tik_instance.Tensor("uint16", mask_shape_ub,
                                           name="mask_ub", scope=tik.scope_ubuf)
        data_x_max = self.tik_instance.Tensor("float16", (
            fmap_img2col_cut_h_num, 16, 16), name="data_x_max",
                                              scope=tik.scope_ubuf)
        with self.tik_instance.if_scope(cut_h_index != 0):
            with self.tik_instance.if_scope(cut_h_index != (cut_h_num - 1)):
                # copy input fmap from gm to l1
                gm_l1_burst_len = int(cut_h_size * self.in_size_w)
                with self.tik_instance.if_scope(cut_h_index == (cut_h_num - 2)):
                    len_tmp = min(cut_h_size, (self.in_size_h + self.pad[2] -
                                               cut_stride * (cut_h_num - 2)))
                    gm_l1_burst_len_1 = len_tmp * self.in_size_w
                    pad_max = max(0, self.pad[3] - cut_stride)
                    self.tik_instance.data_move(
                        input_fmap_l1,
                        self.input_fmap_gm[
                            nc1_num * self.in_size_h * self.in_size_w *
                            self.c_block_size + (cut_h_index * cut_stride -
                                                 self.pad[2]) * self.in_size_w *
                            self.c_block_size], 0, 1, gm_l1_burst_len_1, 0, 0)
                    with self.tik_instance. \
                            for_range(0, fmap_img2col_cut_h_num) as h_index:
                        source_h = (((h_index * 256 * self.fmap_img2col_w)
                                     // (16 * self.fmap_img2col_w)) //
                                    self.out_size_w) * self.stride_h
                        source_w = (((h_index * 256 * self.fmap_img2col_w) //
                                     (16 * self.fmap_img2col_w)) %
                                    self.out_size_w) * self.stride_w - \
                                   self.pad[0]
                        self.scalar_source_h.set_as(source_h)
                        self.scalar_source_w.set_as(source_w)
                        self.tik_instance.load3dv1(
                            fmap_img2col_ub[h_index * 256 *
                                            self.fmap_img2col_w],
                            input_fmap_l1[0],
                            (self.pad[0], self.pad[1], 0, pad_max),
                            cut_h_size - pad_max, self.in_size_w, 0, 0, 0,
                            self.scalar_source_w,
                            self.scalar_source_h, self.stride_w,
                            self.stride_h, self.window_w, self.window_h, 1,
                            1, 1, 0, self.fmap_img2col_w, 0,
                            self.pad_value)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(cut_h_index != 1):
                        self.tik_instance.data_move(
                            input_fmap_l1,
                            self.input_fmap_gm[
                                nc1_num * self.in_size_h * self.in_size_w *
                                self.c_block_size + (cut_h_index * cut_stride -
                                                     self.pad[2]) * self.in_size_w *
                                self.c_block_size], 0, 1, gm_l1_burst_len, 0, 0)
                        with self.tik_instance. \
                                for_range(0, fmap_img2col_cut_h_num) as h_index:
                            source_h = (((h_index * 256 * self.fmap_img2col_w)
                                         // (16 * self.fmap_img2col_w)) //
                                        self.out_size_w) * self.stride_h
                            source_w = (((h_index * 256 * self.fmap_img2col_w) //
                                         (16 * self.fmap_img2col_w)) %
                                        self.out_size_w) * self.stride_w - \
                                       self.pad[0]
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)
                            self.tik_instance.load3dv1(
                                fmap_img2col_ub[h_index * 256 *
                                                self.fmap_img2col_w],
                                input_fmap_l1[0],
                                (self.pad[0], self.pad[1], 0, 0),
                                cut_h_size, self.in_size_w, 0, 0, 0,
                                self.scalar_source_w,
                                self.scalar_source_h, self.stride_w,
                                self.stride_h, self.window_w, self.window_h, 1,
                                1, 1, 0, self.fmap_img2col_w, 0,
                                self.pad_value)
                    with self.tik_instance.else_scope():
                        # copy input fmap from gm to l1
                        len_tmp1 = min(cut_h_size, (cut_h_size - self.pad[2] + cut_stride))
                        pad_max1 = max(0, self.pad[2] - cut_stride)
                        gm_l1_burst_len2 = int(len_tmp1 * self.in_size_w)
                        gm_tem = max(0, cut_stride - self.pad[2])
                        self.tik_instance.data_move(input_fmap_l1, self.input_fmap_gm[
                            nc1_num * self.in_size_h * self.in_size_w *
                            self.c_block_size + gm_tem * self.in_size_w *
                            self.c_block_size], 0, 1, gm_l1_burst_len2, 0, 0)
                        with self.tik_instance.for_range(
                                0, fmap_img2col_cut_h_num) as h_index:
                            source_h = (((h_index * 256 * self.fmap_img2col_w) /
                                         (16 * self.fmap_img2col_w)) /
                                        self.out_size_w) * self.stride_h - pad_max1
                            source_w = (((h_index * 256 * self.fmap_img2col_w) /
                                         (16 * self.fmap_img2col_w)) %
                                        self.out_size_w) * self.stride_w - self.pad[0]
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)
                            self.tik_instance.load3dv1(
                                fmap_img2col_ub[h_index * 256 * self.fmap_img2col_w],
                                input_fmap_l1[0],
                                (self.pad[0], self.pad[1],
                                 pad_max1, 0),
                                (cut_h_size - pad_max1),
                                self.in_size_w, 0,
                                0, 0, self.scalar_source_w,
                                self.scalar_source_h,
                                self.stride_w, self.stride_h,
                                self.window_w, self.window_h, 1,
                                1, 1, 0, self.fmap_img2col_w, 0,
                                self.pad_value)
                if self.fmap_img2col_w != 1:
                    self._calc_max_and_mask(fmap_img2col_cut_h_num,
                                            fmap_img2col_ub,
                                            data_x_max, mask_ub)
                    # move max output to gm
                    gm_max_burst_len = int(fmap_img2col_cut_h)
                    self.tik_instance.data_move(
                        self.output_max_gm[
                            nc1_num * self.out_size_h *
                            self.out_size_w * self.c_block_size +
                            cut_h_index * fmap_img2col_cut_h *
                            self.c_block_size],
                        data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                    self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h)
                else:
                    # move max output to gm
                    gm_max_burst_len = int(fmap_img2col_cut_h)
                    self.tik_instance.data_move(
                        self.output_max_gm[
                            nc1_num * self.out_size_h *
                            self.out_size_w * self.c_block_size +
                            cut_h_index * fmap_img2col_cut_h *
                            self.c_block_size],
                        fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                    self._dup_mask_fun(mask_ub, mask_shape_ub)
                with self.tik_instance.for_range(
                        0, self.fmap_img2col_w) as w_index:
                    offset_output_mask = \
                        nc1_num * (self.fmap_img2col_h_num + 1) * \
                        self.fmap_img2col_w * self.c_block_size + \
                        cut_h_index * fmap_img2col_cut_h
                    self.tik_instance.data_move(
                        self.output_mask_gm[
                            offset_output_mask +
                            w_index * (self.fmap_img2col_h_num + 1) *
                            16], mask_ub[w_index *
                                         fmap_img2col_cut_h_num *
                                         self.c_block_size],
                        0, 1, fmap_img2col_cut_h_num, 0, 0)
            with self.tik_instance.else_scope():
                cut_h_tail = self.in_size_h + self.pad[2] - cut_stride * \
                             (cut_h_num - 1)
                if cut_h_tail > cut_h_size:
                    cut_h_tail = cut_h_size
                out_size_h_tail = \
                    (cut_h_tail - self.window_h + self.stride_h +
                     self.pad[3]) // self.stride_h
                fmap_img2col_h_tail = self.out_size_w * out_size_h_tail
                fmap_img2col_h_tail_num = _ceil_div(fmap_img2col_h_tail, 16)
                # copy input fmap from gm to l1
                gm_l1_burst_len = int(cut_h_tail * self.in_size_w *
                                      self.c_block_size // 16)
                self.tik_instance.data_move(
                    input_fmap_l1,
                    self.input_fmap_gm[
                        nc1_num * self.in_size_h * self.in_size_w *
                        self.c_block_size + (cut_h_index * cut_stride -
                                             self.pad[2]) * self.in_size_w *
                        self.c_block_size], 0, 1,
                    gm_l1_burst_len, 0, 0)
                with self.tik_instance.for_range(
                        0, fmap_img2col_h_tail_num) as h_index:
                    source_h = (((h_index * 256 * self.fmap_img2col_w)
                                 / (16 * self.fmap_img2col_w)) /
                                self.out_size_w) * self.stride_h
                    source_w = (((h_index * 256 *
                                  self.fmap_img2col_w) /
                                 (16 * self.fmap_img2col_w)) %
                                self.out_size_w) * self.stride_w - \
                               self.pad[0]
                    self.scalar_source_h.set_as(source_h)
                    self.scalar_source_w.set_as(source_w)
                    self.tik_instance.load3dv1(
                        fmap_img2col_ub[h_index * 256 *
                                        self.fmap_img2col_w],
                        input_fmap_l1[0],
                        (self.pad[0], self.pad[1], 0, self.pad[3]),
                        cut_h_tail, self.in_size_w, 0, 0, 0,
                        self.scalar_source_w,
                        self.scalar_source_h, self.stride_w,
                        self.stride_h, self.window_w, self.window_h, 1,
                        1, 1, 0, self.fmap_img2col_w, 0,
                        self.pad_value)
                if self.fmap_img2col_w != 1:
                    self._calc_max_and_mask(fmap_img2col_h_tail_num,
                                            fmap_img2col_ub,
                                            data_x_max, mask_ub,
                                            fmap_img2col_cut_h_num)
                    # move max output to gm
                    gm_max_burst_len = int(fmap_img2col_h_tail)
                    self.tik_instance.data_move(
                        self.output_max_gm[
                            nc1_num * self.out_size_h *
                            self.out_size_w * self.c_block_size +
                            cut_h_index * fmap_img2col_cut_h *
                            self.c_block_size],
                        data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                    self._remove_repeated_fun(mask_ub, fmap_img2col_h_tail, 0,
                                              0, fmap_img2col_cut_h)
                else:
                    # move max output to gm
                    gm_max_burst_len = int(fmap_img2col_h_tail)
                    self.tik_instance.data_move(
                        self.output_max_gm[
                            nc1_num * self.out_size_h *
                            self.out_size_w * self.c_block_size +
                            cut_h_index * fmap_img2col_cut_h *
                            self.c_block_size],
                        fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                    self._dup_mask_fun(mask_ub, mask_shape_ub)
                mask_cut = fmap_img2col_h_tail_num * 16 - fmap_img2col_h_tail
                mask_zero_cut = 2 ** 16 - 2**(16 - mask_cut)
                with self.tik_instance.for_range(
                        0, self.fmap_img2col_w) as w_index:
                    offset_output_mask = \
                        nc1_num * (self.fmap_img2col_h_num + 1) * \
                        self.fmap_img2col_w * self.c_block_size + \
                        cut_h_index * fmap_img2col_cut_h
                    if mask_zero_cut != 0 and self.fmap_img2col_w != 1:
                        self.tik_instance.vector_dup([
                            0, mask_zero_cut], mask_ub[
                                w_index * fmap_img2col_cut_h_num * self.c_block_size +
                                fmap_img2col_h_tail_num * 16 - 16], 0, 1, 1, 8)

                    self.tik_instance.data_move(
                        self.output_mask_gm[
                            offset_output_mask +
                            w_index * (self.fmap_img2col_h_num + 1) * 16],
                        mask_ub[w_index * fmap_img2col_cut_h_num *
                                self.c_block_size],
                        0, 1, fmap_img2col_h_tail_num, 0, 0)
        with self.tik_instance.else_scope():
            # copy input fmap from gm to l1
            gm_l1_burst_len = int(
                (cut_h_size - self.pad[2]) * self.in_size_w *
                self.c_block_size // 16)
            self.tik_instance.data_move(input_fmap_l1, self.input_fmap_gm[
                nc1_num * self.in_size_h * self.in_size_w *
                self.c_block_size], 0, 1, gm_l1_burst_len, 0, 0)
            with self.tik_instance.for_range(
                    0, fmap_img2col_cut_h_num) as h_index:
                source_h = (((h_index * 256 * self.fmap_img2col_w) /
                             (16 * self.fmap_img2col_w)) /
                            self.out_size_w) * self.stride_h - self.pad[2]
                source_w = (((h_index * 256 * self.fmap_img2col_w) /
                             (16 * self.fmap_img2col_w)) %
                            self.out_size_w) * self.stride_w - self.pad[0]
                self.scalar_source_h.set_as(source_h)
                self.scalar_source_w.set_as(source_w)
                self.tik_instance.load3dv1(
                    fmap_img2col_ub[h_index * 256 * self.fmap_img2col_w],
                    input_fmap_l1[0],
                    (self.pad[0], self.pad[1],
                     self.pad[2], 0),
                    (cut_h_size - self.pad[2]),
                    self.in_size_w, 0,
                    0, 0, self.scalar_source_w,
                    self.scalar_source_h,
                    self.stride_w, self.stride_h,
                    self.window_w, self.window_h, 1,
                    1, 1, 0, self.fmap_img2col_w, 0,
                    self.pad_value)
            if self.fmap_img2col_w != 1:
                self._calc_max_and_mask(fmap_img2col_cut_h_num,
                                        fmap_img2col_ub, data_x_max, mask_ub)
                # move max output to gm
                gm_max_burst_len = int(fmap_img2col_cut_h)
                self.tik_instance.data_move(
                    self.output_max_gm[
                        nc1_num * self.out_size_h * self.out_size_w *
                        self.c_block_size],
                    data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h)
            else:
                # move max output to gm
                gm_max_burst_len = int(fmap_img2col_cut_h)
                self.tik_instance.data_move(
                    self.output_max_gm[
                        nc1_num * self.out_size_h * self.out_size_w *
                        self.c_block_size],
                    fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                self._dup_mask_fun(mask_ub, mask_shape_ub)
            with self.tik_instance.for_range(
                    0, self.fmap_img2col_w) as w_index:
                offset_output_mask = \
                    nc1_num * (self.fmap_img2col_h_num + 1) * \
                    self.fmap_img2col_w * self.c_block_size + \
                    cut_h_index * fmap_img2col_cut_h
                self.tik_instance.data_move(
                    self.output_mask_gm[
                        offset_output_mask + w_index *
                        (self.fmap_img2col_h_num + 1) * 16],
                    mask_ub[w_index *
                            fmap_img2col_cut_h_num *
                            self.c_block_size], 0,
                    1, fmap_img2col_cut_h_num, 0, 0)

    # pylint: disable=too-many-locals,too-many-statements
    def _fun_only_cut_h(self, block_index, nc1_cuth_index, cut_h_size,
                        cut_stride, cut_h_num, nc1_cuth_size, flag_cut_h):
        """
        funtion only cut H

        Parameters
        ----------
        block_index: index of block
        nc1_cuth_index: index of nc1_cuth
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        nc1_cuth_size: size of nc1_cuth
        flag_cut_h: bool

        Returns
        -------
        none
        """
        fmap_l1_shape = (cut_h_size, self.in_size_w, self.c_block_size)
        input_fmap_l1 = self.tik_instance.Tensor(self.input_dtype,
                                                 fmap_l1_shape,
                                                 name="input_fmap_l1",
                                                 scope=tik.scope_cbuf)
        out_size_cut_h = \
            (cut_h_size - self.window_h + self.stride_h) // self.stride_h
        fmap_img2col_cut_h = self.out_size_w * out_size_cut_h
        fmap_img2col_cut_h_num = _ceil_div(fmap_img2col_cut_h, 16)
        fmap_img2col_shape_ub = (fmap_img2col_cut_h_num * 16, self.window_h,
                                 self.window_w, self.c_block_size)
        fmap_img2col_ub = self.tik_instance.Tensor(self.input_dtype,
                                                   fmap_img2col_shape_ub,
                                                   name="fmap_img2col_ub",
                                                   scope=tik.scope_ubuf)
        mask_shape_ub = (self.window_h, self.window_w, fmap_img2col_cut_h_num,
                         self.c_block_size)
        if flag_cut_h:
            cut_h_index = (block_index * nc1_cuth_size + nc1_cuth_index) % cut_h_num
            nc1_num = (block_index * nc1_cuth_size + nc1_cuth_index) // cut_h_num
            self._calc_only_cut_h(cut_h_index, cut_h_size, cut_stride, cut_h_num,
                                  input_fmap_l1, fmap_img2col_ub, fmap_img2col_cut_h,
                                  mask_shape_ub, nc1_num)
        else:
            nc1_num = block_index * nc1_cuth_size + nc1_cuth_index
            with self.tik_instance.for_range(0, cut_h_num) as cut_h_index:
                self._calc_only_cut_h(cut_h_index, cut_h_size, cut_stride, cut_h_num,
                                      input_fmap_l1, fmap_img2col_ub, fmap_img2col_cut_h,
                                      mask_shape_ub, nc1_num)

    # pylint: disable=too-many-statements,too-many-branches
    def _calc_need_cut_h_w(self, nc1_num, cut_h_size, cut_h_num, cut_h_index, cut_stride):
        """
        funtion need cut H and W while l1 not enough

        Parameters
        ----------
        nc1_num: num of n*c1
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        cut_h_index: index of cuth

        Returns
        -------
        none
        """
        cut_w_size, cut_w_stride, cut_w_num = self._calc_cut_w_size_fun()
        fmap_l1_shape = (cut_h_size, self.in_size_w, self.c_block_size)
        input_fmap_l1 = self.tik_instance.Tensor(self.input_dtype,
                                                 fmap_l1_shape,
                                                 name="input_fmap_l1",
                                                 scope=tik.scope_cbuf)
        with self.tik_instance.for_range(0, cut_w_num) as cut_w_index:
            out_size_cut_h = (cut_h_size - self.window_h +
                              self.stride_h) // self.stride_h
            fmap_img2col_cut_h = self.out_size_w * out_size_cut_h
            out_size_cut_w = (cut_w_size - self.window_w +
                              self.stride_w) // self.stride_w
            fmap_img2col_cut_w = out_size_cut_w
            fmap_img2col_cut_w_num = _ceil_div(fmap_img2col_cut_w, 16)
            fmap_img2col_shape_ub = (
                fmap_img2col_cut_w_num * 16, self.window_h,
                self.window_w, self.c_block_size)
            fmap_img2col_ub = \
                self.tik_instance.Tensor(self.input_dtype,
                                         fmap_img2col_shape_ub,
                                         name="fmap_img2col_ub",
                                         scope=tik.scope_ubuf)
            mask_shape_ub = (self.window_h, self.window_w,
                             fmap_img2col_cut_w_num,
                             self.c_block_size)
            mask_ub = self.tik_instance.Tensor("uint16",
                                               mask_shape_ub,
                                               name="mask_ub",
                                               scope=tik.scope_ubuf)
            data_x_max = self.tik_instance.Tensor("float16",
                                                  (fmap_img2col_cut_w_num,
                                                   16, 16),
                                                  name="data_x_max",
                                                  scope=tik.scope_ubuf)
            with self.tik_instance.if_scope(cut_h_index != 0):
                with self.tik_instance.if_scope(
                        cut_h_index != (cut_h_num - 1)):
                    # copy input fmap from gm to l1
                    gm_l1_burst_len = int(cut_h_size * self.in_size_w *
                                          self.c_block_size // 16)
                    self.tik_instance.data_move(
                        input_fmap_l1,
                        self.input_fmap_gm[
                            nc1_num * self.in_size_h * self.in_size_w *
                            self.c_block_size +
                            (cut_h_index * cut_stride - self.pad[2]) *
                            self.in_size_w *
                            self.c_block_size],
                        0, 1, gm_l1_burst_len, 0, 0)
                    with self.tik_instance.if_scope(cut_w_index != 0):
                        with self.tik_instance.if_scope(
                                cut_w_index != (cut_w_num - 1)):
                            with self.tik_instance. \
                                for_range(0, fmap_img2col_cut_w_num) \
                                as h_index:
                                source_h = 0
                                source_w = \
                                    (((h_index * 256 * self.fmap_img2col_w) /
                                      (16 * self.fmap_img2col_w)) %
                                     out_size_cut_w) * self.stride_w + \
                                    cut_w_stride * cut_w_index - self.pad[0]
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)
                                self.tik_instance.load3dv1(
                                    fmap_img2col_ub[
                                        h_index * 256 *
                                        self.fmap_img2col_w],
                                    input_fmap_l1[0],
                                    (self.pad[0], self.pad[1], 0, 0),
                                    cut_h_size, self.in_size_w,
                                    0, 0, 0, self.scalar_source_w,
                                    self.scalar_source_h,
                                    self.stride_w, self.stride_h,
                                    self.window_w, self.window_h,
                                    1, 1, 1, 0, self.fmap_img2col_w,
                                    0, self.pad_value)
                            if self.fmap_img2col_w != 1:
                                self._calc_max_and_mask(fmap_img2col_cut_w_num,
                                                        fmap_img2col_ub,
                                                        data_x_max, mask_ub)
                                # move max output to gm
                                gm_max_burst_len = int(fmap_img2col_cut_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.out_size_h *
                                        self.out_size_w * self.c_block_size +
                                        cut_h_index * self.out_size_w *
                                        self.c_block_size +
                                        cut_w_index * fmap_img2col_cut_w *
                                        self.c_block_size],
                                    data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._remove_repeated_fun(mask_ub,
                                                          fmap_img2col_cut_h,
                                                          fmap_img2col_cut_w)
                            else:
                                # move max output to gm
                                gm_max_burst_len = int(fmap_img2col_cut_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.out_size_h *
                                        self.out_size_w * self.c_block_size +
                                        cut_h_index * self.out_size_w *
                                        self.c_block_size +
                                        cut_w_index * fmap_img2col_cut_w *
                                        self.c_block_size],
                                    fmap_img2col_ub[0], 0, 1, gm_max_burst_len,
                                    0, 0)
                                self._dup_mask_fun(mask_ub, mask_shape_ub)
                            with self.tik_instance. \
                                for_range(0, self.fmap_img2col_w) \
                                as w_index:
                                offset_output_mask = \
                                    nc1_num * (self.fmap_img2col_h_num + 1) * \
                                    self.fmap_img2col_w * \
                                    self.c_block_size + \
                                    cut_h_index * fmap_img2col_cut_h + \
                                    cut_w_index * fmap_img2col_cut_w
                                self.tik_instance.data_move(
                                    self.output_mask_gm[
                                        offset_output_mask
                                        + w_index *
                                        (self.fmap_img2col_h_num
                                         + 1) * 16],
                                    mask_ub[w_index * fmap_img2col_cut_w_num *
                                            self.c_block_size],
                                    0, 1, fmap_img2col_cut_w_num, 0, 0)
                        with self.tik_instance.else_scope():
                            cut_w_tail = self.in_size_w + self.pad[
                                0] - cut_w_stride * (cut_w_num - 1)
                            if cut_w_tail > cut_w_size:
                                cut_w_tail = cut_w_size
                            out_size_tail_w = (cut_w_tail - self.window_w +
                                               self.stride_w +
                                               self.pad[1]) // self.stride_w
                            fmap_img2col_tail_w = out_size_tail_w
                            fmap_img2col_tail_w_num = _ceil_div(
                                fmap_img2col_tail_w, 16)
                            with self.tik_instance.for_range(
                                    0, fmap_img2col_tail_w_num) as h_index:
                                source_h = 0
                                source_w = \
                                    (((h_index * 256 *
                                       self.fmap_img2col_w) /
                                      (16 * self.fmap_img2col_w)) %
                                     out_size_tail_w
                                    ) * self.stride_w + \
                                    cut_w_stride * cut_w_index - self.pad[0]
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)
                                self.tik_instance.load3dv1(
                                    fmap_img2col_ub[
                                        h_index * 256 * self.fmap_img2col_w],
                                    input_fmap_l1[0],
                                    (self.pad[0], self.pad[1], 0, 0),
                                    cut_h_size, self.in_size_w,
                                    0, 0, 0, self.scalar_source_w,
                                    self.scalar_source_h,
                                    self.stride_w, self.stride_h,
                                    self.window_w, self.window_h,
                                    1, 1, 1, 0, self.fmap_img2col_w,
                                    0, self.pad_value)
                            if self.fmap_img2col_w != 1:
                                self._calc_max_and_mask(fmap_img2col_tail_w_num,
                                                        fmap_img2col_ub,
                                                        data_x_max, mask_ub,
                                                        fmap_img2col_cut_w_num)
                                # move max output to gm
                                gm_max_burst_len = int(fmap_img2col_tail_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.out_size_h * self.out_size_w *
                                        self.c_block_size + cut_h_index *
                                        self.out_size_w * self.c_block_size +
                                        cut_w_index * fmap_img2col_cut_w *
                                        self.c_block_size],
                                    data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._remove_repeated_fun(mask_ub,
                                                          fmap_img2col_cut_h,
                                                          fmap_img2col_tail_w,
                                                          fmap_img2col_cut_w)
                            else:
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.out_size_h * self.out_size_w *
                                        self.c_block_size + cut_h_index *
                                        self.out_size_w * self.c_block_size +
                                        cut_w_index * fmap_img2col_cut_w *
                                        self.c_block_size],
                                    fmap_img2col_ub[0], 0, 1, gm_max_burst_len,
                                    0, 0)
                                self._dup_mask_fun(mask_ub, mask_shape_ub)
                            with self.tik_instance.for_range(
                                    0, self.fmap_img2col_w) as w_index:
                                offset_output_mask = \
                                    nc1_num * (self.fmap_img2col_h_num + 1) * \
                                    self.fmap_img2col_w * \
                                    self.c_block_size + cut_h_index * \
                                    fmap_img2col_cut_h + cut_w_index * \
                                    fmap_img2col_cut_w
                                self.tik_instance.data_move(
                                    self.output_mask_gm[
                                        offset_output_mask
                                        + w_index *
                                        (self.fmap_img2col_h_num
                                         + 1) * 16],
                                    mask_ub[w_index *
                                            fmap_img2col_cut_w_num *
                                            self.c_block_size],
                                    0, 1, fmap_img2col_tail_w_num, 0, 0)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.for_range(
                                0, fmap_img2col_cut_w_num) as h_index:
                            source_h = 0
                            source_w = (((h_index * 256 *
                                          self.fmap_img2col_w) /
                                         (16 * self.fmap_img2col_w))
                                        % out_size_cut_w
                                       ) * self.stride_w - self.pad[0]
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)
                            self.tik_instance.load3dv1(
                                fmap_img2col_ub[h_index * 256 *
                                                self.fmap_img2col_w],
                                input_fmap_l1[0],
                                (self.pad[0], self.pad[1], 0, 0),
                                cut_h_size, self.in_size_w,
                                0, 0, 0, self.scalar_source_w,
                                self.scalar_source_h,
                                self.stride_w, self.stride_h,
                                self.window_w, self.window_h, 1,
                                1, 1, 0, self.fmap_img2col_w,
                                0, self.pad_value)
                        if self.fmap_img2col_w != 1:
                            self._calc_max_and_mask(fmap_img2col_cut_w_num,
                                                    fmap_img2col_ub, data_x_max,
                                                    mask_ub)
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.out_size_h * self.out_size_w *
                                    self.c_block_size + cut_h_index *
                                    self.out_size_w * self.c_block_size],
                                data_x_max[0], 0, 1,
                                gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub,
                                                      fmap_img2col_cut_h,
                                                      fmap_img2col_cut_w)
                        else:
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.out_size_h * self.out_size_w *
                                    self.c_block_size + cut_h_index *
                                    self.out_size_w * self.c_block_size],
                                fmap_img2col_ub[0], 0, 1,
                                gm_max_burst_len, 0, 0)
                            self._dup_mask_fun(mask_ub, mask_shape_ub)
                        with self.tik_instance.for_range(
                                0, self.fmap_img2col_w) as w_index:
                            offset_output_mask = \
                                nc1_num * (self.fmap_img2col_h_num + 1) * \
                                self.fmap_img2col_w * self.c_block_size + \
                                cut_h_index * fmap_img2col_cut_h
                            self.tik_instance.data_move(
                                self.output_mask_gm[
                                    offset_output_mask + w_index *
                                    (self.fmap_img2col_h_num + 1) * 16],
                                mask_ub[w_index * fmap_img2col_cut_w_num *
                                        self.c_block_size],
                                0, 1, fmap_img2col_cut_w_num, 0, 0)
                with self.tik_instance.else_scope():
                    # copy input fmap from gm to l1
                    if self.in_size_h - cut_stride * (cut_h_num - 1) + \
                        self.pad[2] <= cut_h_size:
                        gm_l1_burst_len = int((self.in_size_h - cut_stride *
                                               (cut_h_num - 1) + self.pad[2]) *
                                              self.in_size_w *
                                              self.c_block_size // 16)
                    else:
                        gm_l1_burst_len = int(cut_h_size * self.in_size_w *
                                              self.c_block_size // 16)
                    self.tik_instance.data_move(
                        input_fmap_l1, self.input_fmap_gm[
                            nc1_num * self.in_size_h *
                            self.in_size_w * self.c_block_size +
                            (cut_h_index * cut_stride - self.pad[2]) *
                            self.in_size_w * self.c_block_size], 0,
                        1, gm_l1_burst_len, 0, 0)
                    with self.tik_instance.if_scope(cut_w_index != 0):
                        with self.tik_instance.if_scope(
                                cut_w_index != (cut_w_num - 1)):
                            with self.tik_instance.for_range(
                                    0, fmap_img2col_cut_w_num) as h_index:
                                source_h = 0
                                source_w = \
                                    (((h_index * 256 *
                                       self.fmap_img2col_w) /
                                      (16 * self.fmap_img2col_w)) %
                                     out_size_cut_w
                                    ) * self.stride_w + \
                                    cut_w_stride * cut_w_index - self.pad[0]
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)
                                self.tik_instance.load3dv1(
                                    fmap_img2col_ub[
                                        h_index * 256 *
                                        self.fmap_img2col_w],
                                    input_fmap_l1[0],
                                    (self.pad[0], self.pad[1],
                                     0, self.pad[3]),
                                    (cut_h_size -
                                     self.pad[3]), self.in_size_w,
                                    0, 0, 0,
                                    self.scalar_source_w,
                                    self.scalar_source_h,
                                    self.stride_w, self.stride_h,
                                    self.window_w, self.window_h,
                                    1, 1, 1, 0,
                                    self.fmap_img2col_w, 0,
                                    self.pad_value)
                            if self.fmap_img2col_w != 1:
                                self._calc_max_and_mask(fmap_img2col_cut_w_num,
                                                        fmap_img2col_ub,
                                                        data_x_max, mask_ub)
                                # move max output to gm
                                gm_max_burst_len = int(fmap_img2col_cut_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.out_size_h *
                                        self.out_size_w * self.c_block_size +
                                        cut_h_index * out_size_cut_h *
                                        self.out_size_w * self.c_block_size +
                                        cut_w_index * fmap_img2col_cut_w *
                                        self.c_block_size],
                                    data_x_max[0], 0, 1,
                                    gm_max_burst_len, 0, 0)
                                self._remove_repeated_fun(mask_ub,
                                                          fmap_img2col_cut_h,
                                                          fmap_img2col_cut_w)
                            else:
                                # move max output to gm
                                gm_max_burst_len = int(fmap_img2col_cut_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.out_size_h *
                                        self.out_size_w * self.c_block_size +
                                        cut_h_index * out_size_cut_h *
                                        self.out_size_w * self.c_block_size +
                                        cut_w_index * fmap_img2col_cut_w *
                                        self.c_block_size],
                                    fmap_img2col_ub[0], 0, 1,
                                    gm_max_burst_len, 0, 0)
                                self._dup_mask_fun(mask_ub, mask_shape_ub)
                            with self.tik_instance.for_range(
                                    0, self.fmap_img2col_w) as w_index:
                                offset_output_mask = \
                                    nc1_num * (self.fmap_img2col_h_num + 1) * \
                                    self.fmap_img2col_w * \
                                    self.c_block_size + \
                                    cut_h_index * fmap_img2col_cut_h + \
                                    cut_w_index * fmap_img2col_cut_w
                                self.tik_instance.data_move(
                                    self.output_mask_gm[
                                        offset_output_mask + w_index *
                                        (self.fmap_img2col_h_num + 1) * 16],
                                    mask_ub[w_index * fmap_img2col_cut_w_num *
                                            self.c_block_size],
                                    0, 1, fmap_img2col_cut_w_num, 0, 0)
                        with self.tik_instance.else_scope():
                            cut_w_tail = self.in_size_w + self.pad[
                                0] - cut_w_stride * (cut_w_num - 1)
                            if cut_w_tail > cut_w_size:
                                cut_w_tail = cut_w_size
                            out_size_tail_w = (cut_w_tail - self.window_w +
                                               self.stride_w +
                                               self.pad[1]) // self.stride_w
                            fmap_img2col_tail_w = out_size_tail_w
                            fmap_img2col_tail_w_num = _ceil_div(
                                fmap_img2col_tail_w, 16)
                            with self.tik_instance.for_range(
                                    0, fmap_img2col_tail_w_num) as h_index:
                                source_h = 0
                                source_w = \
                                    (((h_index * 256 *
                                       self.fmap_img2col_w) /
                                      (16 * self.fmap_img2col_w)) %
                                     out_size_tail_w
                                    ) * self.stride_w + cut_w_stride * \
                                    cut_w_index - self.pad[0]
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)
                                self.tik_instance.load3dv1(
                                    fmap_img2col_ub[
                                        h_index * 256 *
                                        self.fmap_img2col_w],
                                    input_fmap_l1[0],
                                    (self.pad[0], self.pad[1],
                                     0, self.pad[3]),
                                    (cut_h_size - self.pad[3]),
                                    self.in_size_w, 0, 0, 0,
                                    self.scalar_source_w,
                                    self.scalar_source_h,
                                    self.stride_w, self.stride_h,
                                    self.window_w, self.window_h,
                                    1, 1, 1, 0, self.fmap_img2col_w,
                                    0, self.pad_value)
                            if self.fmap_img2col_w != 1:
                                self._calc_max_and_mask(fmap_img2col_tail_w_num,
                                                        fmap_img2col_ub,
                                                        data_x_max, mask_ub,
                                                        fmap_img2col_cut_w_num)
                                # move max output to gm
                                gm_max_burst_len = int(fmap_img2col_tail_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.out_size_h *
                                        self.out_size_w * self.c_block_size +
                                        cut_h_index * out_size_cut_h *
                                        self.out_size_w * self.c_block_size +
                                        cut_w_index * fmap_img2col_cut_w *
                                        self.c_block_size],
                                    data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._remove_repeated_fun(mask_ub,
                                                          fmap_img2col_cut_h,
                                                          fmap_img2col_tail_w,
                                                          fmap_img2col_cut_w)
                            else:
                                # move max output to gm
                                gm_max_burst_len = int(fmap_img2col_tail_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.out_size_h *
                                        self.out_size_w * self.c_block_size +
                                        cut_h_index * out_size_cut_h *
                                        self.out_size_w * self.c_block_size +
                                        cut_w_index * fmap_img2col_cut_w *
                                        self.c_block_size],
                                    fmap_img2col_ub[0], 0, 1, gm_max_burst_len,
                                    0, 0)
                                self._dup_mask_fun(mask_ub, mask_shape_ub)
                            mask_cut_w = fmap_img2col_tail_w_num * 16 - fmap_img2col_tail_w
                            mask_zero_w = 2 ** 16 - 2**(16 - mask_cut_w)
                            with self.tik_instance.for_range(
                                    0, self.fmap_img2col_w) as w_index:
                                offset_output_mask = \
                                    nc1_num * (self.fmap_img2col_h_num + 1) * \
                                    self.fmap_img2col_w * \
                                    self.c_block_size + \
                                    cut_h_index * fmap_img2col_cut_h + \
                                    cut_w_index * fmap_img2col_cut_w
                                if mask_zero_w != 0 and self.fmap_img2col_w != 1:
                                    self.tik_instance.vector_dup([
                                        0, mask_zero_w], mask_ub[
                                            w_index * fmap_img2col_cut_w_num * self.c_block_size +
                                            fmap_img2col_tail_w_num * 16 - 16], 0, 1, 1, 8)

                                self.tik_instance.data_move(
                                    self.output_mask_gm[
                                        offset_output_mask + w_index *
                                        (self.fmap_img2col_h_num + 1) * 16],
                                    mask_ub[w_index * fmap_img2col_cut_w_num *
                                            self.c_block_size],
                                    0, 1, fmap_img2col_tail_w_num, 0, 0)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.for_range(
                                0, fmap_img2col_cut_w_num) as h_index:
                            source_h = 0
                            source_w = \
                                (((h_index * 256 *
                                   self.fmap_img2col_w) /
                                  (16 * self.fmap_img2col_w)) %
                                 out_size_cut_w) * \
                                self.stride_w - self.pad[0]
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)
                            self.tik_instance.load3dv1(
                                fmap_img2col_ub[h_index * 256 *
                                                self.fmap_img2col_w],
                                input_fmap_l1[0],
                                (self.pad[0], self.pad[1], 0, self.pad[3]),
                                (cut_h_size - self.pad[3]),
                                self.in_size_w, 0, 0,
                                0, self.scalar_source_w,
                                self.scalar_source_h,
                                self.stride_w, self.stride_h,
                                self.window_w, self.window_h, 1,
                                1, 1, 0, self.fmap_img2col_w,
                                0, self.pad_value)
                        if self.fmap_img2col_w != 1:
                            self._calc_max_and_mask(fmap_img2col_cut_w_num,
                                                    fmap_img2col_ub, data_x_max,
                                                    mask_ub)
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.out_size_h * self.out_size_w *
                                    self.c_block_size + cut_h_index *
                                    self.out_size_w * self.c_block_size],
                                data_x_max[0], 0, 1,
                                gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub,
                                                      fmap_img2col_cut_h,
                                                      fmap_img2col_cut_w)
                        else:
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.out_size_h * self.out_size_w *
                                    self.c_block_size + cut_h_index *
                                    self.out_size_w * self.c_block_size],
                                fmap_img2col_ub[0], 0, 1,
                                gm_max_burst_len, 0, 0)
                            self._dup_mask_fun(mask_ub, mask_shape_ub)
                        with self.tik_instance.for_range(
                                0, self.fmap_img2col_w) as w_index:
                            offset_output_mask = \
                                nc1_num * (self.fmap_img2col_h_num + 1) * \
                                self.fmap_img2col_w * self.c_block_size + \
                                cut_h_index * fmap_img2col_cut_h
                            self.tik_instance.data_move(
                                self.output_mask_gm[
                                    offset_output_mask + w_index *
                                    (self.fmap_img2col_h_num + 1) * 16],
                                mask_ub[w_index * fmap_img2col_cut_w_num *
                                        self.c_block_size],
                                0, 1, fmap_img2col_cut_w_num, 0, 0)
            with self.tik_instance.else_scope():
                # copy input fmap from gm to l1
                gm_l1_burst_len = int(
                    (cut_h_size - self.pad[2]) * self.in_size_w *
                    self.c_block_size // 16)
                self.tik_instance.data_move(
                    input_fmap_l1,
                    self.input_fmap_gm[
                        nc1_num * self.in_size_h * self.in_size_w * self.c_block_size],
                    0, 1, gm_l1_burst_len, 0, 0)
                with self.tik_instance.if_scope(cut_w_index != 0):
                    with self.tik_instance.if_scope(
                            cut_w_index != (cut_w_num - 1)):
                        with self.tik_instance.for_range(
                                0, fmap_img2col_cut_w_num) as h_index:
                            source_h = -self.pad[2]
                            source_w = \
                                (((h_index * 256 *
                                   self.fmap_img2col_w) /
                                  (16 * self.fmap_img2col_w))
                                 % out_size_cut_w
                                ) * self.stride_w + cut_w_stride * \
                                cut_w_index - self.pad[0]
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)
                            self.tik_instance.load3dv1(
                                fmap_img2col_ub[h_index * 256 *
                                                self.fmap_img2col_w],
                                input_fmap_l1[0],
                                (self.pad[0], self.pad[1], self.pad[2], 0),
                                (cut_h_size - self.pad[2]),
                                self.in_size_w, 0, 0, 0,
                                self.scalar_source_w, self.scalar_source_h,
                                self.stride_w, self.stride_h,
                                self.window_w, self.window_h, 1,
                                1, 1, 0, self.fmap_img2col_w,
                                0, self.pad_value)
                        if self.fmap_img2col_w != 1:
                            self._calc_max_and_mask(fmap_img2col_cut_w_num,
                                                    fmap_img2col_ub, data_x_max,
                                                    mask_ub)
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.out_size_h * self.out_size_w *
                                    self.c_block_size +
                                    cut_w_index * fmap_img2col_cut_w *
                                    self.c_block_size],
                                data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub,
                                                      fmap_img2col_cut_h,
                                                      fmap_img2col_cut_w)
                        else:
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.out_size_h * self.out_size_w *
                                    self.c_block_size +
                                    cut_w_index * fmap_img2col_cut_w *
                                    self.c_block_size],
                                fmap_img2col_ub[0], 0, 1, gm_max_burst_len,
                                0, 0)
                            self._dup_mask_fun(mask_ub, mask_shape_ub)
                        with self.tik_instance.for_range(
                                0, self.fmap_img2col_w) as w_index:
                            offset_output_mask = \
                                nc1_num * (self.fmap_img2col_h_num + 1) * \
                                self.fmap_img2col_w * self.c_block_size + \
                                cut_h_index * fmap_img2col_cut_h + \
                                cut_w_index * fmap_img2col_cut_w
                            self.tik_instance.data_move(
                                self.output_mask_gm[
                                    offset_output_mask +
                                    w_index * (self.fmap_img2col_h_num + 1) *
                                    16],
                                mask_ub[w_index * fmap_img2col_cut_w_num *
                                        self.c_block_size],
                                0, 1, fmap_img2col_cut_w_num, 0, 0)
                    with self.tik_instance.else_scope():
                        cut_w_tail = self.in_size_w + self.pad[
                            0] - cut_w_stride * (cut_w_num - 1)
                        if cut_w_tail > cut_w_size:
                            cut_w_tail = cut_w_size
                        out_size_tail_w = (cut_w_tail - self.window_w +
                                           self.stride_w +
                                           self.pad[1]) // self.stride_w
                        fmap_img2col_tail_w = out_size_tail_w
                        fmap_img2col_tail_w_num = _ceil_div(
                            fmap_img2col_tail_w, 16)
                        with self.tik_instance. \
                            for_range(0, fmap_img2col_tail_w_num) \
                            as h_index:
                            source_h = -self.pad[2]
                            source_w = \
                                (((h_index * 256 * self.fmap_img2col_w) /
                                  (16 * self.fmap_img2col_w)) %
                                 out_size_tail_w) * self.stride_w + \
                                cut_w_stride * cut_w_index - self.pad[0]
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)
                            self.tik_instance.load3dv1(
                                fmap_img2col_ub[
                                    h_index * 256 *
                                    self.fmap_img2col_w],
                                input_fmap_l1[0],
                                (self.pad[0], self.pad[1],
                                 self.pad[2], 0), (cut_h_size - self.pad[2]),
                                self.in_size_w, 0, 0,
                                0, self.scalar_source_w,
                                self.scalar_source_h,
                                self.stride_w, self.stride_h,
                                self.window_w, self.window_h, 1,
                                1, 1, 0, self.fmap_img2col_w,
                                0, self.pad_value)
                        if self.fmap_img2col_w != 1:
                            self._calc_max_and_mask(fmap_img2col_tail_w_num,
                                                    fmap_img2col_ub, data_x_max,
                                                    mask_ub,
                                                    fmap_img2col_cut_w_num)
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_tail_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.out_size_h * self.out_size_w *
                                    self.c_block_size + cut_w_index *
                                    fmap_img2col_cut_w * self.c_block_size],
                                data_x_max[0], 0, 1,
                                gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub,
                                                      fmap_img2col_cut_h,
                                                      fmap_img2col_tail_w,
                                                      fmap_img2col_cut_w)
                        else:
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_tail_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.out_size_h * self.out_size_w *
                                    self.c_block_size + cut_w_index *
                                    fmap_img2col_cut_w * self.c_block_size],
                                fmap_img2col_ub[0], 0, 1,
                                gm_max_burst_len, 0, 0)
                            self._dup_mask_fun(mask_ub, mask_shape_ub)
                        mask_cut_w = fmap_img2col_tail_w_num * 16 - fmap_img2col_tail_w
                        mask_zero_w = 2 ** 16 - 2**(16 - mask_cut_w)
                        with self.tik_instance.for_range(
                                0, self.fmap_img2col_w) as w_index:
                            offset_output_mask = \
                                nc1_num * (self.fmap_img2col_h_num + 1) * \
                                self.fmap_img2col_w * self.c_block_size + \
                                cut_h_index * fmap_img2col_cut_h + \
                                cut_w_index * fmap_img2col_cut_w
                            if mask_zero_w != 0 and self.fmap_img2col_w != 1\
                                    and cut_h_num == 1:
                                self.tik_instance.vector_dup([
                                    0, mask_zero_w], mask_ub[
                                        w_index * fmap_img2col_cut_w_num * self.c_block_size +
                                        fmap_img2col_tail_w_num * 16 - 16], 0, 1, 1, 8)

                            self.tik_instance.data_move(
                                self.output_mask_gm[
                                    offset_output_mask +
                                    w_index *
                                    (self.fmap_img2col_h_num + 1) * 16],
                                mask_ub[w_index * fmap_img2col_cut_w_num *
                                        self.c_block_size],
                                0, 1, fmap_img2col_tail_w_num, 0, 0)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(
                            0, fmap_img2col_cut_w_num) as h_index:
                        source_h = -self.pad[2]
                        source_w = (((h_index * 256 *
                                      self.fmap_img2col_w) /
                                     (16 * self.fmap_img2col_w)) %
                                    out_size_cut_w) * \
                                   self.stride_w - self.pad[0]
                        self.scalar_source_h.set_as(source_h)
                        self.scalar_source_w.set_as(source_w)
                        self.tik_instance.load3dv1(
                            fmap_img2col_ub[
                                h_index * 256 *
                                self.fmap_img2col_w],
                            input_fmap_l1[0],
                            (self.pad[0], self.pad[1], self.pad[2], 0),
                            (cut_h_size - self.pad[2]),
                            self.in_size_w, 0, 0, 0,
                            self.scalar_source_w,
                            self.scalar_source_h,
                            self.stride_w, self.stride_h,
                            self.window_w, self.window_h, 1, 1, 1,
                            0, self.fmap_img2col_w, 0, self.pad_value)
                    if self.fmap_img2col_w != 1:
                        self._calc_max_and_mask(fmap_img2col_cut_w_num,
                                                fmap_img2col_ub,
                                                data_x_max, mask_ub)
                        # move max output to gm
                        gm_max_burst_len = int(fmap_img2col_cut_w)
                        self.tik_instance.data_move(
                            self.output_max_gm[nc1_num * self.out_size_h *
                                               self.out_size_w *
                                               self.c_block_size],
                            data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                        self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h,
                                                  fmap_img2col_cut_w)
                    else:
                        # move max output to gm
                        gm_max_burst_len = int(fmap_img2col_cut_w)
                        self.tik_instance.data_move(
                            self.output_max_gm[nc1_num * self.out_size_h *
                                               self.out_size_w *
                                               self.c_block_size],
                            fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                        self._dup_mask_fun(mask_ub, mask_shape_ub)
                    with self.tik_instance. \
                        for_range(0, self.fmap_img2col_w) as w_index:
                        offset_output_mask = \
                            nc1_num * (self.fmap_img2col_h_num + 1) * \
                            self.fmap_img2col_w * self.c_block_size + \
                            cut_h_index * fmap_img2col_cut_h
                        self.tik_instance.data_move(
                            self.output_mask_gm[offset_output_mask +
                                                w_index *
                                                (self.fmap_img2col_h_num +
                                                 1) * 16],
                            mask_ub[w_index * fmap_img2col_cut_w_num *
                                    self.c_block_size],
                            0, 1, fmap_img2col_cut_w_num, 0, 0)

    # pylint: disable=too-many-statements,too-many-branches
    def _fun_need_cut_h_w(self, block_index, nc1_cuth_index, cut_h_size,
                          cut_stride, cut_h_num, nc1_cuth_size, flag_cut_h):
        """
        funtion need cut H and W while l1 not enough

        Parameters
        ----------
        block_index: index of block
        nc1_index: index of nc1
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        flag_cut_h:bool

        Returns
        -------
        none
        """
        if flag_cut_h:
            cut_h_index = (block_index * nc1_cuth_size + nc1_cuth_index) % cut_h_num
            nc1_num = (block_index * nc1_cuth_size + nc1_cuth_index) // cut_h_num
            self._calc_need_cut_h_w(nc1_num, cut_h_size, cut_h_num, cut_h_index, cut_stride)
        else:
            nc1_num = block_index * nc1_cuth_size + nc1_cuth_index
            with self.tik_instance.for_range(0, cut_h_num) as cut_h_index:
                self._calc_need_cut_h_w(nc1_num, cut_h_size, cut_h_num, cut_h_index, cut_stride)

    def _calc_out_size_and_pad(self):
        """
        caculate output size and padding size

        Parameters
        ----------
        none

        Returns
        -------
        pad: include pad_t, pad_b, pad_l, pad_r
        out_size_h: out_size in h direction
        out_size_w: out_size in w direction
        """
        # pad_l, pad_r, pad_t, pad_b is for pad on the left, right, top, bottom
        pad_l, pad_r, pad_t, pad_b = 0, 0, 0, 0

        if self.padding == "SAME":
            # Hout = ceil(Hi, Sh), Wout = ceil(Wi, Sw)
            out_size_h = (self.in_size_h + self.stride_h - 1) // self.stride_h
            out_size_w = (self.in_size_w + self.stride_w - 1) // self.stride_w

            # get total pad rows or pad columns
            pad_rows = (out_size_h - 1) * self.stride_h + \
                       ((self.window_h - 1) * DILATION + 1) - self.in_size_h
            pad_cols = (out_size_w - 1) * self.stride_w + \
                       ((self.window_w - 1) * DILATION + 1) - self.in_size_w

            # pad_rows and pad_columns is odd or even number
            if pad_rows % 2 == 0:
                pad_t = pad_rows // 2
                pad_b = pad_rows // 2
            else:
                pad_t = pad_rows // 2
                pad_b = pad_rows - pad_t

            if pad_cols % 2 == 0:
                pad_l = pad_cols // 2
                pad_r = pad_cols // 2
            else:
                pad_l = pad_cols // 2
                pad_r = pad_cols - pad_l

            if pad_t < 0:
                pad_t = 0

            if pad_b < 0:
                pad_b = 0

            if pad_l < 0:
                pad_l = 0

            if pad_r < 0:
                pad_r = 0

        # caculate output size in VALID mode
        if self.padding == "VALID":
            # Hout = ceil(Hi - Fh + 1, Sh), Wout = ceil(Wi - Fw + 1, Sw)
            out_size_h = (self.in_size_h - self.window_h + 1 +
                          (self.stride_h - 1)) // self.stride_h
            out_size_w = (self.in_size_w - self.window_w + 1 +
                          (self.stride_w - 1)) // self.stride_w
        pad = (pad_l, pad_r, pad_t, pad_b)

        return pad, out_size_h, out_size_w

    # pylint: disable=too-many-branches
    def _calc_cut_h_size_fun(self, need_cut=False):
        """
        caculate cut_h size

        Parameters
        ----------
        need_cut :bool
            if need cut

        Returns
        -------
        cut_h_size: cut size
        cut_stride: cut stride
        fh_loop: loop number
        """
        img2col_w = self.window_h * self.window_w * 16
        img2col_h = UB_SIZE / 2 / (img2col_w * 2 + (32 * 5))
        if self.window_h >= self.stride_h:
            cut_h_size = ((img2col_h //
                           (((self.in_size_w + self.pad[0] + self.pad[1])) //
                            self.stride_w + 1)) - 1) * self.stride_h + \
                         self.window_h - self.stride_h
            if cut_h_size < self.window_h:
                cut_h_size = self.window_h
            cut_stride = cut_h_size - (self.window_h - self.stride_h)
        else:
            cut_h_size = ((img2col_h //
                           (((self.in_size_w + self.pad[0] + self.pad[1])) //
                            self.stride_w + 1)) - 1) * self.stride_h
            if cut_h_size < self.window_h:
                cut_h_size = self.window_h
                cut_stride = self.stride_h
            else:
                cut_stride = cut_h_size

        if cut_h_size >= cut_stride:
            fh_loop = _ceil_div(
                ((self.in_size_h + self.pad[2] + self.pad[3]) - cut_h_size),
                cut_stride) + 1
        else:
            if (self.in_size_h + self.pad[2] + self.pad[3]) % cut_stride == 0:
                fh_loop = (self.in_size_h + self.pad[2] + self.pad[
                    3]) // cut_stride
            else:
                fh_loop = _ceil_div(
                    (self.in_size_h + self.pad[2] + self.pad[3]),
                    cut_stride)

        if cut_h_size * self.in_size_w * self.c_block_size * 2 > L1_SIZE:
            need_cut = True

        if need_cut:
            cut_h_size = self.window_h
            cut_stride = self.stride_h
            if cut_h_size >= cut_stride:
                fh_loop = _ceil_div(
                    ((self.in_size_h + self.pad[2] + self.pad[3]) - cut_h_size),
                    cut_stride) + 1
            else:
                if (self.in_size_h + self.pad[2] + self.pad[3]) % \
                    cut_stride == 0:
                    fh_loop = (self.in_size_h + self.pad[2] + self.pad[
                        3]) // cut_stride
                else:
                    fh_loop = _ceil_div(
                        (self.in_size_h + self.pad[2] + self.pad[3]),
                        cut_stride)

        if self.padding == "VALID":
            if cut_h_size == self.window_h:
                if cut_h_size >= cut_stride:
                    fh_loop = (self.in_size_h - cut_h_size + cut_stride) // cut_stride
                else:
                    fh_loop = self.in_size_h // cut_stride
            else:
                if cut_h_size >= cut_stride:
                    fh_loop = (self.in_size_h - cut_h_size + cut_stride) // cut_stride
                else:
                    fh_loop = self.in_size_h // cut_stride
                if (self.in_size_h % cut_stride) >= self.window_h:
                    fh_loop = fh_loop + 1

        return int(cut_h_size), int(cut_stride), int(fh_loop)

    def _calc_cut_w_size_fun(self):
        """
        caculate cut_w size

        Parameters
        ----------
        none

        Returns
        -------
        cut_w_size: cut size
        cut_w_stride: cut stride
        fw_loop: loop number
        """
        img2col_w = self.window_h * self.window_w * 16
        img2col_h = UB_SIZE / 2 / (img2col_w * 2 + (32 * 5))
        if self.window_w >= self.stride_w:
            cut_w_size = (img2col_h // 1 - 1) * self.stride_w + self.window_w - \
                         self.stride_w
            cut_w_stride = cut_w_size - (self.window_w - self.stride_w)
        else:
            cut_w_size = (img2col_h // 1 - 1) * self.stride_w
            cut_w_stride = cut_w_size

        if cut_w_size < self.window_w:
            raise RuntimeError(
                "cutC0 is needed and this scene is not supported")

        if cut_w_size >= cut_w_stride:
            fw_loop = _ceil_div(
                ((self.in_size_w + self.pad[0] + self.pad[1]) - cut_w_size),
                cut_w_stride) + 1
        else:
            if (self.in_size_w + self.pad[0] + self.pad[1]) % cut_w_stride == 0:
                fw_loop = (self.in_size_w + self.pad[0] + self.pad[1]) // \
                          cut_w_stride
            else:
                fw_loop = _ceil_div(
                    (self.in_size_w + self.pad[0] + self.pad[1]),
                    cut_w_stride)

        if self.padding == "VALID":
            if cut_w_size == self.window_w:
                if cut_w_size >= cut_w_stride:
                    fw_loop = (self.in_size_w - cut_w_size + cut_w_stride) // cut_w_stride
                else:
                    fw_loop = self.in_size_w // cut_w_size
            else:
                if cut_w_size >= cut_w_stride:
                    fw_loop = (self.in_size_w - cut_w_size + cut_w_stride) // cut_w_stride
                else:
                    fw_loop = self.in_size_w // cut_w_stride
                if (self.in_size_w % cut_w_stride) >= self.window_w:
                    fw_loop = fw_loop + 1

        return int(cut_w_size), int(cut_w_stride), int(fw_loop)

    def _calc_max_fun(self, data_input, data_input_ub, index_w, index_h):
        """
        caculate max of data_input

        Parameters
        ----------
        data_input: input data
        data_input_ub: input data in ub
        index_w: input size in w direction
        index_h: input size in h direction

        Returns
        -------
        data_input: output tensor
        """
        self.tik_instance.vmax(
            MASK, data_input[index_h * 256], data_input[index_h * 256],
            data_input_ub[index_w * 256 + index_h * self.fmap_img2col_w * 256],
            REPEAT_2, DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0, DSTSTRIDEM1,
            SRC0STRIDEM1, SRC1STRIDEM1)
        return data_input

    def _calc_mask_fun(self, data_input_max, data_input_ub, index_w, index_h,
                       fmap_h_num, mask_ub):
        """
        caculate mask of data_input_max

        Parameters
        ----------
        data_input_max: max value in input data
        data_input_ub: input data in ub
        index_w: index of w
        index_h: index of h
        fmap_h_num: num of fmap in h
        mask_ub: mask in ub

        Returns
        -------
        mask_ub: mask in ub
        """
        self.tik_instance.vcmpv_eq(mask_ub[index_w * fmap_h_num * 16 +
                                           index_h * 16],
                                   data_input_ub[index_w * 256 + index_h *
                                                 self.fmap_img2col_w * 256],
                                   data_input_max[index_h * 256], REPEAT_2,
                                   SRC0STRIDEM0, SRC1STRIDEM0, SRC0STRIDEM1,
                                   SRC1STRIDEM1)
        return mask_ub

    def _calc_max_and_mask(self, fmap_h_num, fmap_img2col_ub, data_x_max,
                           mask_ub, fmap_img2col_cut_w_num=0,
                           fmap_h_tail_num=0):
        """
        caculate max and mask of data_input

        Parameters
        ----------
        fmap_h_num: num of fmap_img2col_h
        fmap_img2col_ub: fmap in ub
        data_x_max: max value in input data
        mask_ub: mask in ub
        fmap_img2col_cut_w_num: cut number of w
        fmap_h_tail_num: num of h tail

        Returns
        -------
        data_input_ub: output tensor
        """
        scalar_repeat_times = int(fmap_h_num * 2)
        repeat_times = _ceil_div(scalar_repeat_times, 254)
        # dup 8*blocks init 1 into a buffer:
        if scalar_repeat_times > 255:
            with self.tik_instance.for_range(0, repeat_times) as repeat_index:
                with self.tik_instance.if_scope(repeat_index !=
                                                (repeat_times - 1)):
                    self.tik_instance.vector_dup(
                        MASK, data_x_max[repeat_index * 254 * 128],
                        MIN_VALUE_FP16,
                        254, DSTSTRIDEM0, SRC0STRIDEM1)
                with self.tik_instance.else_scope():
                    self.tik_instance.vector_dup(
                        MASK, data_x_max[repeat_index * 254 * 128],
                        MIN_VALUE_FP16,
                        (scalar_repeat_times - repeat_index * 254),
                        DSTSTRIDEM0, SRC0STRIDEM1)
        else:
            self.tik_instance.vector_dup(MASK, data_x_max, MIN_VALUE_FP16,
                                         scalar_repeat_times, DSTSTRIDEM0,
                                         SRC0STRIDEM1)
        # do max
        with self.tik_instance.for_range(0, self.fmap_img2col_w) as index_w:
            with self.tik_instance.for_range(0, fmap_h_num) as index_h:
                # the first 128
                data_x_max = self._calc_max_fun(data_x_max, fmap_img2col_ub,
                                                index_w, index_h)
        # do mask
        with self.tik_instance.for_range(0, self.fmap_img2col_w) as index_w:
            with self.tik_instance.for_range(0, fmap_h_num) as index_h:
                if fmap_img2col_cut_w_num == 0:
                    if fmap_h_tail_num == 0:
                        mask_ub = self._calc_mask_fun(data_x_max,
                                                      fmap_img2col_ub,
                                                      index_w, index_h,
                                                      fmap_h_num, mask_ub)
                    else:
                        mask_ub = self._calc_mask_fun(data_x_max,
                                                      fmap_img2col_ub,
                                                      index_w, index_h,
                                                      fmap_h_tail_num, mask_ub)
                else:
                    mask_ub = self._calc_mask_fun(data_x_max, fmap_img2col_ub,
                                                  index_w, index_h,
                                                  fmap_img2col_cut_w_num,
                                                  mask_ub)

    def _remove_repeated_fun(self, mask_ub, fmap_img2col_cut_h=0,
                             fmap_img2col_cut_w=0, fmap_img2col_tail_w=0,
                             fmap_img2col_tail_h=0):
        """
        caculate max and mask of data_input

        Parameters
        ----------
        fmap_img2col_h: size of fmap_img2col_h
        mask_ub: mask in ub
        fmap_img2col_cut_h: size of fmap_img2col_cut_h
        fmap_img2col_cut_w: size of fmap_img2col_cut_w
        fmap_img2col_tail_w: size of fmap_img2col_tail_w
        fmap_img2col_tail_h: size of tail_h

        Returns
        -------
        data_input_ub: output tensor
        """
        if fmap_img2col_cut_h != 0:
            if fmap_img2col_cut_w != 0:
                fmap_img2col_h_num = _ceil_div(fmap_img2col_cut_w, 16)
            else:
                fmap_img2col_h_num = _ceil_div(fmap_img2col_cut_h, 16)
        else:
            fmap_img2col_h_num = _ceil_div(self.fmap_img2col_h, 16)

        mask_or_shape_ub = (fmap_img2col_h_num, 16)
        mask_or = self.tik_instance.Tensor(
            "uint16", mask_or_shape_ub, name="mask_or", scope=tik.scope_ubuf)
        mask_not = self.tik_instance.Tensor(
            "uint16", mask_or_shape_ub, name="mask_not", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, self.fmap_img2col_w) as index_w:
            with self.tik_instance.if_scope(index_w > 0):
                if fmap_img2col_tail_w == 0:
                    if fmap_img2col_tail_h == 0:
                        self.tik_instance.vor(
                            16, mask_or[0],
                            mask_ub[index_w * fmap_img2col_h_num * 16],
                            mask_or[0],
                            fmap_img2col_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0)
                        self.tik_instance.vand(
                            16, mask_ub[index_w * fmap_img2col_h_num * 16],
                            mask_not[0],
                            mask_ub[index_w * fmap_img2col_h_num * 16],
                            fmap_img2col_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0)
                    else:
                        fmap_img2col_tail_num = _ceil_div(fmap_img2col_tail_h, 16)
                        self.tik_instance.vor(
                            16, mask_or[0],
                            mask_ub[index_w * fmap_img2col_tail_num * 16],
                            mask_or[0],
                            fmap_img2col_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0)
                        self.tik_instance.vand(
                            16, mask_ub[index_w * fmap_img2col_tail_num * 16],
                            mask_not[0],
                            mask_ub[index_w * fmap_img2col_tail_num * 16],
                            fmap_img2col_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0)
                else:
                    fmap_img2col_tail_num = _ceil_div(fmap_img2col_tail_w, 16)
                    self.tik_instance.vor(
                        16, mask_or[0],
                        mask_ub[index_w * fmap_img2col_tail_num * 16],
                        mask_or[0],
                        fmap_img2col_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                        SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0)
                    self.tik_instance.vand(
                        16, mask_ub[index_w * fmap_img2col_tail_num * 16],
                        mask_not[0],
                        mask_ub[index_w * fmap_img2col_tail_num * 16],
                        fmap_img2col_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                        SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0)
                self.tik_instance.vnot(16, mask_not[0], mask_or[0],
                                       fmap_img2col_h_num, SRC0STRIDEM0,
                                       SRC1STRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0)
            with self.tik_instance.else_scope():
                self.tik_instance.vnot(16, mask_not[0], mask_ub[0],
                                       fmap_img2col_h_num, SRC0STRIDEM0,
                                       SRC1STRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0)
                self.tik_instance.data_move(mask_or[0], mask_ub[0], 0, 1,
                                            fmap_img2col_h_num, 0, 0)

    def _dup_mask_fun(self, mask_ub, mask_shape_ub):
        """
         caculate max and mask of data_input

         Parameters
         ----------
         mask_ub: mask in ub
         mask_shape_ub: shape of mask_ub

         Returns
         -------
         none
         """
        scalar_repeat_times = mask_shape_ub[2]
        repeat_times = _ceil_div(scalar_repeat_times, 240)
        # dup 8*blocks init 1 into a buffer:
        if scalar_repeat_times > 240:
            with self.tik_instance.for_range(0, repeat_times) as repeat_index:
                with self.tik_instance.if_scope(repeat_index !=
                                                (repeat_times - 1)):
                    self.tik_instance.vector_dup(
                        MASK, mask_ub[repeat_index * 240 * 16],
                        65535, 30, DSTSTRIDEM0, SRC0STRIDEM1)
                with self.tik_instance.else_scope():
                    self.tik_instance.vector_dup(
                        16, mask_ub[repeat_index * 240 * 16],
                        65535, (scalar_repeat_times - repeat_index * 240),
                        DSTSTRIDEM0, DSTSTRIDEM0)
        else:
            self.tik_instance.vector_dup(16, mask_ub, 65535,
                                         scalar_repeat_times, DSTSTRIDEM0,
                                         DSTSTRIDEM0)
