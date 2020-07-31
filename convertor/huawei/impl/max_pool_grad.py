"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

maxpool_grad
"""

from __future__ import absolute_import
import math
from te import tik
from topi.cce import util
from te.platform.cce_conf import CceProductParams
from te import platform as tbe_platform
from te import platform as cce


def ceil_div(num, divisor):
    """calcu use ceil_div

    Parameters
    ----------
    num: int
        input num
    divisor: int
        input divisor

    returns
    -------
    result: int
        num // divisor
    """
    if num % divisor != 0:
        return num // divisor + 1
    return num // divisor


def cal_shape_ele(shape):
    """calcu element nums

    Parameters
    ----------
    shape: list
         input shape list

    returns
    -------
    result: int
        the total num of shape
    """
    reduce_ = 1
    for i in shape:
        reduce_ *= int(i)
    return reduce_


def cal_byte_size(shape, dtype):
    """calcu tensor size

    Parameters
    ----------
    shape: list
        input shape list
    dtype: str
        input data dtype
    returns
    -------
    result: int
        the total num of shape
    """
    if dtype == "float16":
        return cal_shape_ele(shape) * 2
    elif dtype == "float32":
        return cal_shape_ele(shape) * 4
    else:
        raise RuntimeError("Not support shape now")


# MIN VALUE OF FP16
MIN_VALUE_FP16 = -65500.0
# MIN VALUE OF POSIRIVE FP32
MIN_VALUE_FP32 = 1.18e-38
# VALID MASK BITS FOR 128
MASK128_VALUE = 128
# VALID MASK BITS FOR 64
MASK64_VALUE = 64
# REPEAT ONE TIMES
REPEAT_1 = 1
# REPEAT TWO TIMES
REPEAT_2 = 2
# DSTSTRIDEM0
DSTSTRIDEM0 = 1
# SRC0STRIDEM0
SRC0STRIDEM0 = 1
# SRC1STRIDEM0
SRC1STRIDEM0 = 1
# DSTSTRIDEM1
DSTSTRIDEM1 = 8
# SRC0STRIDEM1
SRC0STRIDEM1 = 8
# SRC1STRIDEM1
SRC1STRIDEM1 = 8
# MAX_VECTOR_REPEATE_TIME
MAX_VECTOR_REPEATE_TIME = 255
# VECTOR FP16 SIZE
VECTOR_FP16_SIZE = 128
# VECTOR FP32 SIZE
VECTOR_FP32_SIZE = 64
# BLOCK SIZE(32B)
BLOCK_SIZE = 32
# UB SIZE
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
# L1 SIZE
L1_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L1_SIZE)
C0 = 16
# BLOCK NUMS
CORE_NUM = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

# pylint: disable = too-many-arguments
def check_param(ori_input, ori_output, grad, ksize, strides, padding,
                kernel_name):
    """
    check parameters, if one is invalid, then raise error

    Parameters
    ----------
    ori_input: dict
        shape and data type of ori_input
    ori_output: dict
        shape and data type of ori_output
    grad: dict
        shape and data type of grad
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
    ori_input_shape = ori_input.get("shape")
    ori_input_dtype = ori_input.get("dtype").lower()
    ori_output_shape = ori_output.get("shape")
    grad_shape = grad.get("shape")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(ori_input_shape)
    util.check_tensor_shape_size(ori_input_shape)
    util.check_dtype_rule(ori_input_dtype, ("float16",))
    # the format of input_x must be NC1HWC0
    if len(ori_input_shape) != 5:
        raise RuntimeError("invalid shape params, input feature map must be "
                           "5D format in kernel.")
    if len(ori_output_shape) != 5:
        raise RuntimeError("invalid shape params, forward output must be "
                           "5D format in kernel.")
    if len(grad_shape) != 5:
        raise RuntimeError("invalid shape params, update grad must be "
                           "5D format in kernel.")

    if grad_shape != ori_output_shape:
        raise RuntimeError(
            "invalid shape params, update grad must be same shape as forward output")

    if grad_shape[-1] != 16 or ori_input_shape[-1] != 16:
        raise RuntimeError("invalid shape params, C0 must be equal to 16.")

    if ori_output_shape[:2] != ori_input_shape[:2]:
        raise RuntimeError(
            "invalid shape params, N axis and C1 axis should be same.")

    if len(ksize) != 4 or len(strides) != 4:
        raise RuntimeError("Invalid ksize or strides params,"
                           " ksize dim must be 4.")

    if ksize[0] != 1 or ksize[3] != 1:
        raise RuntimeError("MaxPoolGRAD only supports pooling "
                           "across width/height, and other ksize "
                           "dimension should be one")

    if strides[0] != 1 or strides[3] != 1:
        raise RuntimeError("MaxPoolGRAD only supports pooling across "
                           "width/height, and other strides dimension "
                           "should be one")

    if padding not in ("SAME", "VALID"):
        raise RuntimeError("MaxPoolGrad can only support SAME or VALID "
                           "padding mode.")
    if strides[1] > 63 or strides[3] > 63 or strides[1] < 1 or strides[3] < 1:
        raise RuntimeError("Invalid stride parameters, stride size should be in [1, 63]")
    if ksize[1] > 255 or ksize[3] > 255 or strides[1] < 1 or strides[3] < 1:
        raise RuntimeError("Invalid kernel parameters, kernel size should be in [1, 255]")


# pylint: disable = too-many-arguments,,unused-argument,invalid-name
@util.check_input_type(dict, dict, dict, dict,
                       (tuple, list), (tuple, list), str, str, str)
def max_pool_grad(ori_input,
                  ori_output,
                  grad,
                  y,
                  ksize,
                  strides,
                  padding,
                  data_format="NHWC",
                  kernel_name="maxpoolgrad"):
    """
    main function of maxpool_grad

    Parameters
    ----------
    ori_input: dict
        shape and data type of ori_input
    ori_output: dict
        shape and data type of ori_output
    grad: dict
        shape and data type of grad
    y: dict
        shape and data type of y
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    padding: str
        value from `SAME`, `VALID`
    data_format: str
        value from `NCHW`, `NHWC`
    kernel_name: str

    Returns
    -------
    return the tik api function
    """
    ori_format = ori_input.get("ori_format")
    if ori_format == "NCHW":
        ksize = (ksize[0], ksize[2], ksize[3], ksize[1])
        strides = (strides[0], strides[2], strides[3], strides[1])
    check_param(ori_input, ori_output, grad, ksize, strides, padding,
                kernel_name)
    ori_input_shape = ori_input.get("shape")
    ori_output_shape = ori_output.get("shape")
    grad_shape = grad.get("shape")
    dtype = ori_input.get("dtype").lower()
    maxpoolgrad = MaxpoolGrad(ori_input_shape, ori_output_shape, grad_shape,
                              dtype, ksize, strides, padding)
    return maxpoolgrad.tik_instance_function(kernel_name)


# pylint: disable = too-many-instance-attributes
class MaxpoolGrad(object):
    """
    MaxpoolGrad  Object include all fuction and paras
    """

    def __init__(self, ori_input, ori_output, grad, dtype, ksize, strides,
                 padding):
        self.ori_input_shape = ori_input
        self.ori_output_shape = ori_output
        self.grad_shape = grad
        self.dtype = dtype
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.n = ori_input[0]
        self.c1 = ori_input[1]
        self.hi = ori_input[2]
        self.wi = ori_input[3]

        self.ho = grad[2]
        self.wo = grad[3]
        self.stride_h = strides[1]
        self.stride_w = strides[2]
        self.kh = ksize[1]
        self.kw = ksize[2]
        self.pad = None
        self.pad_top = None
        self.pad_bottom = None
        self.pad_left = None
        self.pad_right = None
        self.pad_value = None
        self.ho_block = None
        self.hi_block = None
        self.tile_h_to_block = False

        self.tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))
        self.ori_input_gm = self.tik_instance.Tensor(dtype,
                                                     self.ori_input_shape,
                                                     name='ori_input_gm',
                                                     scope=tik.scope_gm)
        self.ori_output_gm = self.tik_instance.Tensor(dtype,
                                                      self.ori_output_shape,
                                                      name='ori_output_gm',
                                                      scope=tik.scope_gm)
        self.grad_gm = self.tik_instance.Tensor(dtype, self.grad_shape,
                                                name='grad_gm',
                                                scope=tik.scope_gm)
        self.res_gm = self.tik_instance.Tensor(dtype, (
            self.n, self.c1, self.hi, self.wi, C0), name='res_gm',
                                               scope=tik.scope_gm)

        self.scalar_esp = self.tik_instance.Scalar(dtype='float32',
                                                   name='scalar_esp')
        self.scalar_one = self.tik_instance.Scalar(dtype='float32',
                                                   name='scalar_one')
        self.scalar_zero = self.tik_instance.Scalar(dtype='float32',
                                                    name='scalar_zero')
        self.scalar_zero_fp16 = self.tik_instance.Scalar(dtype='float16',
                                                         name='scalar_zero_fp16')
        self.offset_gm = self.tik_instance.Scalar(dtype='int64',
                                                  name='offset_gm')
        self.actual_pad_top = self.tik_instance.Scalar(dtype='int64',
                                                       name='actual_pad_top')
        self.actual_pad_bottom = self.tik_instance.Scalar(dtype='int64',
                                                          name='actual_pad_bottom')
        self.row_effective = self.tik_instance.Scalar(dtype='int64',
                                                      name='row_effective')
        # define some sclar
        self.scalar_zero_fp16.set_as(0)
        self.scalar_zero.set_as(0)
        self.scalar_esp.set_as(1.18e-38)
        self.scalar_one.set_as(1)

    # pylint: disable = too-many-locals
    def _padding_mode(self, ori_input, ksize, strides, padding):
        _, _, fmap_h, fmap_w, _ = ori_input
        _, kernel_h, kernel_w, _ = ksize
        _, self.stride_h, self.stride_w, _ = strides
        if padding == 'VALID':
            ho = int(math.ceil((fmap_h - kernel_h + 1) * 1.0 / self.stride_h))
            wo = int(math.ceil((fmap_w - kernel_w + 1) * 1.0 / self.stride_w))
            pad_top = pad_left = pad_bottom = pad_right = 0

        if padding == 'SAME':
            ho = (fmap_h + self.stride_h - 1) // self.stride_h
            wo = (fmap_w + self.stride_w - 1) // self.stride_w
            pad_h = max((ho - 1) * self.stride_h + kernel_h - fmap_h, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_w = max((wo - 1) * self.stride_w + kernel_w - fmap_w, 0)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
        return wo, ho, pad_left, pad_right, pad_top, pad_bottom

    def _vector_dup(self, src, src_start, shape, dup_reg, dtype):
        ele_num = cal_shape_ele(shape)
        if dtype == "float16":
            total_repeate_time = ele_num // VECTOR_FP16_SIZE
            remain_ele = ele_num % VECTOR_FP16_SIZE
            mask_value = VECTOR_FP16_SIZE
        elif dtype == "float32":
            total_repeate_time = ele_num // VECTOR_FP32_SIZE
            remain_ele = ele_num % VECTOR_FP32_SIZE
            mask_value = VECTOR_FP32_SIZE
        else:
            raise RuntimeError("Not support dtype")
        repeate_max_time = total_repeate_time // MAX_VECTOR_REPEATE_TIME
        remain_repeate_time = total_repeate_time % MAX_VECTOR_REPEATE_TIME

        if repeate_max_time > 0:
            with self.tik_instance.for_range(0, repeate_max_time) as loop1:
                self.tik_instance.vector_dup(mask_value, src[
                    src_start + loop1 * MAX_VECTOR_REPEATE_TIME * mask_value],
                                             dup_reg, MAX_VECTOR_REPEATE_TIME,
                                             1, 8)
        if remain_repeate_time > 0:
            self.tik_instance.vector_dup(mask_value, src[
                src_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME * mask_value],
                                         dup_reg, remain_repeate_time, 1, 8)
        if remain_ele > 0:
            self.tik_instance.vector_dup(remain_ele, src[
                src_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME * mask_value +
                remain_repeate_time * mask_value], dup_reg, 1, 1, 8)

    def _vconv(self, src, src_start, dst, dst_start, ele_num, src_dtype):
        total_repeate_time = ele_num // VECTOR_FP32_SIZE
        remain_ele = ele_num % VECTOR_FP32_SIZE
        mask_value = VECTOR_FP32_SIZE

        repeate_max_time = total_repeate_time // MAX_VECTOR_REPEATE_TIME
        remain_repeate_time = total_repeate_time % MAX_VECTOR_REPEATE_TIME

        if src_dtype == 'float16':
            src_stride, dst_stride = 4, 8
            if repeate_max_time > 0:
                with self.tik_instance.for_range(0, repeate_max_time) as loop1:
                    self.tik_instance.vconv(
                        MASK64_VALUE, "",
                        dst[
                            dst_start + loop1 * MAX_VECTOR_REPEATE_TIME * mask_value],
                        src[
                            src_start + loop1 * MAX_VECTOR_REPEATE_TIME * mask_value],
                        MAX_VECTOR_REPEATE_TIME, 1, 1, dst_stride, src_stride)
            if remain_repeate_time > 0:
                self.tik_instance.vconv(
                    MASK64_VALUE, "",
                    dst[
                        dst_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME * mask_value],
                    src[
                        src_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME * mask_value],
                    remain_repeate_time, 1, 1, dst_stride, src_stride)
            if remain_ele > 0:
                self.tik_instance.vconv(
                    remain_ele, "",
                    dst[dst_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME *
                        mask_value + remain_repeate_time * mask_value],
                    src[src_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME *
                        mask_value + remain_repeate_time * mask_value],
                    1, 1, 1, dst_stride, src_stride)

        else:
            src_stride, dst_stride = 8, 4
            if repeate_max_time > 0:
                with self.tik_instance.for_range(0, repeate_max_time) as loop1:
                    self.tik_instance.vconv(
                        MASK64_VALUE, "",
                        dst[
                            dst_start + loop1 * MAX_VECTOR_REPEATE_TIME * mask_value],
                        src[
                            src_start + loop1 * MAX_VECTOR_REPEATE_TIME * mask_value],
                        MAX_VECTOR_REPEATE_TIME, 1, 1, dst_stride, src_stride)
            if remain_repeate_time > 0:
                self.tik_instance.vconv(
                    MASK64_VALUE, "",
                    dst[
                        dst_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME * mask_value],
                    src[
                        src_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME * mask_value],
                    remain_repeate_time, 1, 1, dst_stride, src_stride)
            if remain_ele > 0:
                self.tik_instance.vconv(
                    remain_ele, "",
                    dst[dst_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME *
                        mask_value + remain_repeate_time * mask_value],
                    src[src_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME *
                        mask_value + remain_repeate_time * mask_value],
                    1, 1, 1, dst_stride, src_stride)

    def _vector_op(self, operator, src1, src2, dst, dtype, ele_num, stride_cofig=None, offset=None):
        if dtype == "float16":
            repeate_times = ele_num // VECTOR_FP16_SIZE
            remain_ele = ele_num % VECTOR_FP16_SIZE
            mask = VECTOR_FP16_SIZE
        else:
            repeate_times = ele_num // VECTOR_FP32_SIZE
            remain_ele = ele_num % VECTOR_FP32_SIZE
            mask = VECTOR_FP32_SIZE

        repeat_max_loop = repeate_times // MAX_VECTOR_REPEATE_TIME
        remain_max_loop = repeate_times % MAX_VECTOR_REPEATE_TIME

        if operator == "vmuls":
            if offset:
                dst_offset = offset[0]
                src1_offset = offset[1]
            else:
                dst_offset = 0
                src1_offset = 0

            if stride_cofig is None:
                stride_cofig = 1, 1, 8, 8
            if repeat_max_loop > 0:
                self.tik_instance.vmuls(mask, dst[dst_offset], src1[src1_offset],
                                        src2, 255,
                                        stride_cofig[0], stride_cofig[1],
                                        stride_cofig[2], stride_cofig[3])
                dst_offset += BLOCK_SIZE // (tbe_platform.cce_intrin.get_bit_len(
                    dst.dtype.lower()) // 8) * stride_cofig[2] * 255
                src1_offset += BLOCK_SIZE // (tbe_platform.cce_intrin.get_bit_len(
                    src1.dtype.lower()) // 8) * stride_cofig[3] * 255
            if remain_max_loop > 0:
                self.tik_instance.vmuls(mask, dst[dst_offset], src1[src1_offset],
                                        src2, remain_max_loop,
                                        stride_cofig[0], stride_cofig[1],
                                        stride_cofig[2], stride_cofig[3])
                dst_offset += BLOCK_SIZE // (tbe_platform.cce_intrin.get_bit_len(
                    dst.dtype.lower()) // 8) * stride_cofig[2] * remain_max_loop
                src1_offset += BLOCK_SIZE // (tbe_platform.cce_intrin.get_bit_len(
                    src1.dtype.lower()) // 8) * stride_cofig[3] * remain_max_loop
            if remain_ele > 0:
                self.tik_instance.vmuls(remain_ele, dst[dst_offset], src1[src1_offset],
                                        src2, 1,
                                        stride_cofig[0], stride_cofig[1],
                                        stride_cofig[2], stride_cofig[3])

        if operator == "vadd":
            if stride_cofig is None:
                stride_cofig = 1, 1, 1, 8, 8, 8
            dst_offset = 0
            src1_offset = 0
            src2_offset = 0
            if repeat_max_loop > 0:
                self.tik_instance.vadd(mask, dst[dst_offset], src1[src1_offset], src2[src2_offset],
                                       255,
                                       stride_cofig[0], stride_cofig[1],
                                       stride_cofig[2], stride_cofig[3],
                                       stride_cofig[4], stride_cofig[5])
                dst_offset += BLOCK_SIZE // (tbe_platform.cce_intrin.get_bit_len(
                    dst.dtype.lower()) // 8) * stride_cofig[3] * 255
                src1_offset += BLOCK_SIZE // (tbe_platform.cce_intrin.get_bit_len(
                    src1.dtype.lower()) // 8) * stride_cofig[4] * 255
                src2_offset += BLOCK_SIZE // (tbe_platform.cce_intrin.get_bit_len(
                    src2.dtype.lower()) // 8) * stride_cofig[5] * 255
            if remain_max_loop > 0:
                self.tik_instance.vadd(mask, dst[dst_offset], src1[src1_offset], src2[src2_offset],
                                       remain_max_loop,
                                       stride_cofig[0], stride_cofig[1],
                                       stride_cofig[2], stride_cofig[3],
                                       stride_cofig[4], stride_cofig[5])
                dst_offset += BLOCK_SIZE // (tbe_platform.cce_intrin.get_bit_len(
                    dst.dtype.lower()) // 8) * stride_cofig[3] * remain_max_loop
                src1_offset += BLOCK_SIZE // (tbe_platform.cce_intrin.get_bit_len(
                    src1.dtype.lower()) // 8) * stride_cofig[4] * remain_max_loop
                src2_offset += BLOCK_SIZE // (tbe_platform.cce_intrin.get_bit_len(
                    src2.dtype.lower()) // 8) * stride_cofig[5] * remain_max_loop
            if remain_ele > 0:
                self.tik_instance.vadd(remain_ele, dst[dst_offset], src1[src1_offset],
                                       src2[src2_offset], 1,
                                       stride_cofig[0], stride_cofig[1],
                                       stride_cofig[2], stride_cofig[3],
                                       stride_cofig[4], stride_cofig[5])

    def _calc_mask(self, index_h, index_w, mask_shape, ori_output_ub, ori_input_col_ub,
                  mask_or, mask_not):
        mask_ori = self.tik_instance.Tensor('uint16', mask_shape,
                                            name='mask_ori',
                                            scope=tik.scope_ubuf)
        mask_ub = self.tik_instance.Tensor('uint16', mask_shape,
                                           name='mask_ori',
                                           scope=tik.scope_ubuf)

        with self.tik_instance.if_scope(tik.all(index_h == 0, index_w == 0)):
            self.tik_instance.vcmpv_eq(mask_ub, ori_output_ub, ori_input_col_ub,
                                       cal_shape_ele(ori_output_ub.shape) // VECTOR_FP16_SIZE,
                                       1, 1, 8, 8)

            self.tik_instance.data_move(mask_or[0],
                                        mask_ub[0], 0, 1,
                                        cal_shape_ele(mask_ub.shape) // 16, 0, 0)

            self.tik_instance.vnot(MASK128_VALUE, mask_not, mask_ub,
                                   cal_shape_ele(mask_ub.shape) // VECTOR_FP16_SIZE,
                                   1, 1, 8, 8)

        with self.tik_instance.else_scope():
            self.tik_instance.vcmpv_eq(mask_ori, ori_output_ub, ori_input_col_ub,
                                       cal_shape_ele(ori_output_ub.shape) // VECTOR_FP16_SIZE,
                                       1, 1, 8, 8)

            mask_ori = mask_ori.reinterpret_cast_to("uint16")
            self.tik_instance.vand(MASK128_VALUE, mask_ub, mask_not, mask_ori,
                                   cal_shape_ele(mask_ub.shape) // VECTOR_FP16_SIZE,
                                   1, 1, 1, 8, 8, 8)

            mask_or = mask_or.reinterpret_cast_to("uint16")
            self.tik_instance.vor(MASK128_VALUE, mask_or, mask_or, mask_ub,
                                  cal_shape_ele(mask_ub.shape) // VECTOR_FP16_SIZE,
                                  1, 1, 1, 8, 8, 8)
            self.tik_instance.vnot(MASK128_VALUE, mask_not, mask_or,
                                   cal_shape_ele(mask_ub.shape) // VECTOR_FP16_SIZE,
                                   1, 1, 8, 8)

        return mask_ub

    def _vsel_grad_col(self, mask_ub, grad_ub):
        grad_sel_ub = self.tik_instance.Tensor("float16", grad_ub.shape,
                                               name="col2img_fp16_ub", scope=tik.scope_ubuf)
        temp_zero = self.tik_instance.Tensor("float16", (MASK128_VALUE,),
                                             name="temp_zero", scope=tik.scope_ubuf)
        self._vector_dup(temp_zero, 0, temp_zero.shape, self.scalar_zero_fp16, "float16")

        # vsel
        with self.tik_instance.for_range(0, grad_ub.shape[0]) as mask_index:
            fractal_repeat = C0 * C0 // VECTOR_FP16_SIZE
            with self.tik_instance.for_range(0, fractal_repeat) as fractal_index:
                mask_type_bit_size = tbe_platform.cce_intrin.get_bit_len("uint16")
                mask_offset = (mask_index * fractal_repeat + fractal_index) * \
                              MASK128_VALUE // mask_type_bit_size

                cmpmask = self.tik_instance.mov_tensor_to_cmpmask(mask_ub[mask_offset])
                grad_ub_offset = (mask_index * fractal_repeat + fractal_index) * MASK128_VALUE
                self.tik_instance.vsel(MASK128_VALUE, 0, grad_sel_ub[grad_ub_offset], cmpmask,
                                       grad_ub[grad_ub_offset], temp_zero,
                                       1, 1, 1, 1, 8, 8, 8)

        return grad_sel_ub

    def _load3d(self, index_h, index_w, start_h, end_h, ori_input_col_ub,
               ori_input_l1, start_pos_h, each_process_hi, each_process_wi,
               repeat_times, pad, pad_value, wo_offset, each_process_hi_block):
        """
        load3d function

        Parameters
        index_h: scalar
            start pos of kernel
        index_w: scalar
            start pos of kernel
        start_h: scalar
            start row number of Ho
        end_h: scalar
            end row number of Ho
        ori_input_col_ub: tensor
            col ub
        ori_input_l1: tensor
            ori input tensor on L1
        start_pos_h:
            start pos of row number on Ho
        each_process_hi: int
            number of rows on Hi processed each loop
        each_process_wi: int
            number of rows on Wi processed each loop
        repeat_times: int
            repeate times of doing load3d
        pad: list
            pad value
        wo_offset: scalar
            offset on W direction
        ----------
        Returns
        ori_input_col_ub: tensor
            load3d's result
        -------
        """
        pad_left, pad_right, pad_top, pad_bottom = pad
        # load3d
        with self.tik_instance.if_scope(start_h <= pad_top):
            self.actual_pad_top.set_as(pad_top - start_h)
            with self.tik_instance.if_scope(end_h < each_process_hi_block + pad_top):
                self.tik_instance.load3dv1(ori_input_col_ub[0],
                                           ori_input_l1[0],
                                           (pad_left, pad_right,
                                            self.actual_pad_top, 0),
                                           each_process_hi - self.actual_pad_top,
                                           each_process_wi, 0, index_w, index_h,
                                           wo_offset, -self.actual_pad_top,
                                           self.stride_w, self.stride_h,
                                           self.kw, self.kh, 1, 1, 1, 1,
                                           repeat_times, 0, pad_value)
            with self.tik_instance.else_scope():
                self.actual_pad_bottom.set_as(end_h - each_process_hi_block - pad_top)
                self.tik_instance.load3dv1(ori_input_col_ub[0],
                                           ori_input_l1[0],
                                           (pad_left, pad_right,
                                            self.actual_pad_top,
                                            self.actual_pad_bottom),
                                           each_process_hi -
                                           self.actual_pad_top -
                                           self.actual_pad_bottom,
                                           each_process_wi, 0, index_w, index_h,
                                           wo_offset, -self.actual_pad_top,
                                           self.stride_w, self.stride_h,
                                           self.kw, self.kh, 1, 1, 1, 1,
                                           repeat_times,
                                           0, pad_value)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(end_h < each_process_hi_block + pad_top):
                self.tik_instance.load3dv1(ori_input_col_ub[0],
                                           ori_input_l1[
                                               start_pos_h * self.wi * C0],
                                           (pad_left, pad_right, 0, 0),
                                           each_process_hi, each_process_wi, 0,
                                           index_w, index_h, wo_offset, 0,
                                           self.stride_w, self.stride_h,
                                           self.kw, self.kh, 1, 1, 1, 1,
                                           repeat_times, 0, pad_value)

            with self.tik_instance.else_scope():
                self.actual_pad_bottom.set_as(end_h - each_process_hi_block - pad_top)
                self.tik_instance.load3dv1(ori_input_col_ub[0],
                                           ori_input_l1[
                                               start_pos_h * self.wi * C0],
                                           (pad_left, pad_right, 0,
                                            self.actual_pad_bottom),
                                           each_process_hi - self.actual_pad_bottom,
                                           each_process_wi, 0,
                                           index_w, index_h, wo_offset, 0,
                                           self.stride_w, self.stride_h,
                                           self.kw, self.kh, 1, 1, 1, 1,
                                           repeat_times, 0, pad_value)
        return ori_input_col_ub

    def _data_move_ub(self, ori_input_shape, ori_output_shape, input_data_num,
                     output_data_nums, src_input_offset, src_output_offset):
        if ori_input_shape:
            ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape,
                                                    name='ori_input_l1',
                                                    scope=tik.scope_cbuf)
            self.tik_instance.data_move(ori_input_l1[0],
                                        self.ori_input_gm[src_input_offset], 0, 1,
                                        input_data_num // 16, 0, 0)
        else:
            ori_input_l1 = None

        # mov actual ori output to ub

        ori_output_ub = self.tik_instance.Tensor(self.dtype, ori_output_shape,
                                                 name='ori_output_ub',
                                                 scope=tik.scope_ubuf)
        self._vector_dup(ori_output_ub, 0, ori_output_shape,
                        self.scalar_zero_fp16, "float16")
        self.tik_instance.data_move(ori_output_ub[0],
                                    self.ori_output_gm[src_output_offset],
                                    0, 1, output_data_nums // 16, 0, 0)

        # mov ori grad to ub
        grad_ub = self.tik_instance.Tensor(self.dtype, ori_output_shape,
                                           name='grad_ub', scope=tik.scope_ubuf)
        self._vector_dup(grad_ub, 0, ori_output_shape,
                        self.scalar_zero_fp16, "float16")
        self.tik_instance.data_move(grad_ub[0],
                                    self.grad_gm[src_output_offset],
                                    0, 1, output_data_nums // 16, 0, 0)

        return ori_input_l1, ori_output_ub, grad_ub

    def _mov_func(self, cut_ho_nums_index, cut_ho_nums, remain_ho_nums, each_process_ho,
                 each_process_hi, each_valid_ho, col2img_fp32_ub, temp_tensor_ub, pad):
        pad_left, pad_right, pad_top, pad_bottom = pad
        wi = self.wi + pad_left + pad_right
        pad_top_rows = self.tik_instance.Scalar(dtype="int64", name='pad_top_rows')
        pad_top_rows.set_as(pad_top - cut_ho_nums_index * each_process_ho * self.stride_h)
        self.tik_instance.scalar_max(pad_top_rows, pad_top_rows, 0)
        each_valid_hi = each_valid_ho * self.stride_h - pad_top_rows

        col2img_fp16_ub = self.tik_instance.Tensor("float16",
                                                   col2img_fp32_ub.shape,
                                                   name="col2img_fp16_ub",
                                                   scope=tik.scope_ubuf)
        self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                   cal_shape_ele(col2img_fp32_ub.shape), "float32")

        with self.tik_instance.if_scope(
                tik.all(cut_ho_nums_index < cut_ho_nums - 1, each_valid_hi > 0)):
            self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                        col2img_fp16_ub[pad_top_rows * wi * C0 + pad_left * C0], 0,
                                        each_valid_hi, self.wi * C0 // 16,
                                        pad_left + pad_right, 0)

            self.offset_gm.set_as(self.offset_gm + each_valid_hi * self.wi * C0)

        if remain_ho_nums == 0:
            with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums - 1):
                if cut_ho_nums - 1 == 0:
                    last_valid_hi = self.hi
                else:
                    last_valid_hi = self.hi - (
                            (cut_ho_nums - 1) * each_process_ho * self.stride_h - pad_top)
                if last_valid_hi <= each_process_hi:
                    self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                col2img_fp16_ub[
                                                    pad_top_rows * wi * C0 + pad_left * C0],
                                                0,
                                                last_valid_hi, self.wi * C0 // 16,
                                                pad_left + pad_right, 0)
                else:
                    self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                col2img_fp16_ub[
                                                    pad_top_rows * wi * C0 + pad_left * C0],
                                                0,
                                                each_process_hi, self.wi * C0 // 16,
                                                pad_left + pad_right, 0)
                    remain_hi = last_valid_hi - each_process_hi
                    temp_zero = self.tik_instance.Tensor("float16",
                                                         (remain_hi, self.wi * C0),
                                                         name='temp_zero',
                                                         scope=tik.scope_ubuf)
                    self._vector_dup(temp_zero, 0, temp_zero.shape,
                                    self.scalar_zero_fp16, "float16")
                    self.tik_instance.data_move(
                        self.res_gm[self.offset_gm + each_process_hi * self.wi * C0],
                        temp_zero,
                        0,
                        1, remain_hi * self.wi * C0 // 16,
                        0, 0)
                self.offset_gm.set_as(self.offset_gm + last_valid_hi * self.wi * C0)
        else:
            with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums - 1):
                self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                            col2img_fp16_ub[pad_top_rows * wi * C0 + pad_left * C0],
                                            0,
                                            each_valid_hi, self.wi * C0 // 16,
                                            pad_left + pad_right, 0)
                self.offset_gm.set_as(self.offset_gm + each_valid_hi * self.wi * C0)
            if isinstance(cut_ho_nums_index, int):
                if cut_ho_nums == 0:
                    last_valid_hi = self.hi
                else:
                    last_valid_hi = self.hi - (
                            cut_ho_nums * each_process_ho * self.stride_h - pad_top)
                if last_valid_hi <= each_process_hi:
                    self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                col2img_fp16_ub[
                                                    pad_top_rows * wi * C0 + pad_left * C0],
                                                0,
                                                last_valid_hi, self.wi * C0 // 16,
                                                pad_left + pad_right, 0)
                else:
                    self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                col2img_fp16_ub[
                                                    pad_top_rows * wi * C0 + pad_left * C0],
                                                0,
                                                each_process_hi, self.wi * C0 // 16,
                                                pad_left + pad_right, 0)
                    remain_hi = last_valid_hi - each_process_hi
                    temp_zero = self.tik_instance.Tensor("float16",
                                                         (remain_hi, self.wi * C0),
                                                         name='temp_zero',
                                                         scope=tik.scope_ubuf)
                    self._vector_dup(temp_zero, 0, temp_zero.shape,
                                    self.scalar_zero_fp16, "float16")
                    self.tik_instance.data_move(
                        self.res_gm[self.offset_gm + each_process_hi * self.wi * C0],
                        temp_zero,
                        0,
                        1, remain_hi * self.wi * C0 // 16,
                        0, 0)
                self.offset_gm.set_as(self.offset_gm + last_valid_hi * self.wi * C0)

        if self.kh > self.stride_h:
            self._vector_op("vmuls", col2img_fp32_ub, 1.0, temp_tensor_ub, "float32",
                           cal_shape_ele(temp_tensor_ub.shape),
                           None, [0, each_process_ho * self.stride_h * wi * C0])

    def _move_func_block(self, cut_ho_nums_index, cut_ho_nums, start_h, end_h, each_process_ho,
                        valid_hi_block, col2img_fp32_ub, temp_tensor_ub,
                        remained_hi, remain, pad):
        pad_left, pad_right, pad_top, pad_bottom = pad
        wi = self.wi + pad_left + pad_right
        mov_len_h = self.tik_instance.Scalar(dtype='int64', name='mov_len_h')
        hi_max = self.tik_instance.Scalar(dtype='int64', name='hi_max')
        hi_min = self.tik_instance.Scalar(dtype='int64', name='hi_min')
        self.tik_instance.scalar_max(hi_min, pad_top, start_h)
        hi_max.set_as(valid_hi_block + pad_top)
        self.tik_instance.scalar_min(hi_max, hi_max, end_h)
        mov_len_h.set_as(hi_max - hi_min)

        col2img_fp16_ub = self.tik_instance.Tensor("float16", col2img_fp32_ub.shape,
                                                   name="col2img_fp16_ub", scope=tik.scope_ubuf)
        self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                   cal_shape_ele(col2img_fp32_ub.shape), "float32")
        with self.tik_instance.if_scope(end_h > pad_top):
            with self.tik_instance.if_scope(start_h < pad_top + valid_hi_block):
                with self.tik_instance.if_scope(mov_len_h > 0):
                    self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                col2img_fp16_ub[(hi_min - start_h) *
                                                                wi * C0 +
                                                                pad_left * C0],
                                                0, mov_len_h, self.wi * C0 // 16,
                                                pad_left + pad_right, 0)
                    self.offset_gm.set_as(self.offset_gm + mov_len_h * self.wi * C0)
                    remained_hi.set_as(remained_hi - mov_len_h)

                with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums - 1):
                    with self.tik_instance.if_scope(remain == 0):
                        with self.tik_instance.if_scope(remained_hi > 0):
                            temp_zero = self.tik_instance.Tensor(
                                "float16", (1, self.wi, C0), name="temp_zero",
                                scope=tik.scope_ubuf)
                            self._vector_dup(temp_zero,
                                            0,
                                            temp_zero.shape,
                                            self.scalar_zero_fp16,
                                            temp_zero.dtype)

                            with self.tik_instance.for_range(0, remained_hi) as index_0:
                                self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                            temp_zero, 0,
                                                            1, self.wi * C0 // 16, 0, 0)
                                self.offset_gm.set_as(self.offset_gm + self.wi * C0)
                with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums):
                    with self.tik_instance.if_scope(remained_hi > 0):
                        temp_zero = self.tik_instance.Tensor(
                            "float16", (1, self.wi, C0), name="temp_zero",
                            scope=tik.scope_ubuf)
                        self._vector_dup(temp_zero,
                                        0,
                                        temp_zero.shape,
                                        self.scalar_zero_fp16,
                                        temp_zero.dtype)
                        with self.tik_instance.for_range(0, remained_hi) as index_0:
                            self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                        temp_zero, 0,
                                                        1, self.wi * C0 // 16, 0, 0)
                            self.offset_gm.set_as(self.offset_gm + self.wi * C0)

        if self.kh > self.stride_h:
            self._vector_op("vmuls", col2img_fp32_ub, 1.0, temp_tensor_ub, "float32",
                           cal_shape_ele(temp_tensor_ub.shape),
                           None, [0, each_process_ho * self.stride_h * wi * C0])
        return remained_hi

    def _tilling_factor(self, ori_input_shape, pad):
        pad_left, pad_right, _, _ = pad
        C0 = ori_input_shape[-1]
        input_l1_size = cal_byte_size(ori_input_shape, self.dtype)
        # each type of buffer's bit size
        fp16_data_size = tbe_platform.cce_intrin.get_bit_len("float16") // 8
        fp32_data_size = tbe_platform.cce_intrin.get_bit_len("float32") // 8
        uint16_data_size = tbe_platform.cce_intrin.get_bit_len("uint16") // 8

        if input_l1_size >= L1_SIZE:
            need_cut_L1 = True
        else:
            need_cut_L1 = False

        if self.kh > self.stride_h:
            each_process_hi = self.kh
        else:
            each_process_hi = self.stride_h

        # define eaxtra_size for some case
        # if each_process_wo is not 32B alined and 256B alined
        extra_size = (VECTOR_FP16_SIZE * C0 + C0) * fp16_data_size
        if self.kw > self.stride_w:
            extra_size += each_process_hi * (self.kw - self.stride_w) * C0 * \
                          (fp16_data_size + fp32_data_size)

        # calculate col ub size
        # There are two col ub, one is fp32, other is fp16, shape is (each_hi, each_wi, C0)
        # Here we need calculate each_wi to judge if need cut wo or cut ho.
        # self.kw > self.stride_w, each_wi = (each_process_wo - 1) * self.stride_w + self.kw
        # self.kw <= self.stride_w, each_wi = each_process_wo * self.stride_w

        col_ub_size_times = each_process_hi * self.stride_w * C0 * (fp16_data_size + fp32_data_size)

        # calculate mask ub size
        # There are for mask buffer on ub, each is (math.ceil(each_wo_16 * C0 // 128) * 128, )
        # Here each_wo_16 is ceil to 16 times.
        mask_ub_size_times = uint16_data_size * 4

        # calculate tensor ub siez
        # There are five tensor buffer on UB, each is (each_wo_16, 16, C0), one is fp32, others
        # are fp16. Here each_wo_16 is ceil to 16 times.
        tensor_ub_size_time = C0 * (4 * fp16_data_size + fp32_data_size)

        # some temp size
        # at most have 4 temp buffer on ub, each is (128, ), dtype is float16
        temp_ub_size = fp16_data_size * self.wi * C0

        each_process_wo = (UB_SIZE - temp_ub_size) * 1.0 / \
                          (col_ub_size_times + mask_ub_size_times + tensor_ub_size_time) - \
                          pad_left - pad_right
        each_process_wo = int(each_process_wo // 16 * 16)

        if each_process_wo >= self.wo:
            wi = self.wi + pad_left + pad_right
            extra_size = 0
            if self.kh > self.stride_h:
                extra_size += (self.kh - self.stride_h) * wi * C0 * (
                        fp16_data_size + 2 * fp32_data_size)
            # calculate col ub size
            # There are two col ub, one is fp32, other is fp16, shape is (each_hi, wi, C0)
            # self.kh > self.stride_h, each_hi = (each_process_ho - 1) * self.stride_h + self.kh
            # self.kh <= self.stride_h, each_hi = each_process_ho * self.stride_h
            col_ub_size_times = self.stride_h * wi * C0 * (fp16_data_size + fp32_data_size)

            mask_ub_size_times = self.wo * uint16_data_size * 4

            tensor_ub_size_time = self.wo * C0 * (4 * fp16_data_size + fp32_data_size)

            temp_ub_size = fp16_data_size * self.wo * C0
            each_process_ho = (UB_SIZE - temp_ub_size - extra_size) // (
                    col_ub_size_times + mask_ub_size_times +
                    tensor_ub_size_time)
            each_process_ho = each_process_ho
            if each_process_ho <= 0:
                each_process_ho = 1
                each_process_wo = self.wo
        else:
            each_process_ho = 0

        if each_process_ho >= self.ho:
            need_cut_Ho = False
            need_cut_Wo = False
            return need_cut_L1, need_cut_Ho, need_cut_Wo, self.ho, 0
        elif each_process_ho > 1:
            need_cut_Ho = True
            need_cut_Wo = False
            return need_cut_L1, need_cut_Ho, need_cut_Wo, each_process_ho, 0
        else:
            need_cut_Ho = True
            need_cut_Wo = True
            return True, need_cut_Ho, need_cut_Wo, 1, each_process_wo

    def _not_tilling(self, n_index, c1_index,
                    each_process_ho_block, each_process_hi_block,
                    mov_len_ho, mov_len_hi,
                    start_ho_index, start_hi_index,
                    start_threshold,
                    offset_gm_block, shape, pad):

        pad_left, pad_right, pad_top, pad_bottom = pad
        shape_ho, shape_wo, shape_hi, shape_wi = shape

        howo_ceil16 = ceil_div(shape_ho * shape_wo, 16)

        wi = self.wi + self.pad_left + self.pad_right
        hi = shape_hi + self.pad_top + self.pad_bottom

        # define col res
        col2img_ub_shape = (hi, wi, C0)
        col2img_fp32_ub = self.tik_instance.Tensor("float32",
                                                   col2img_ub_shape,
                                                   name="col2img_fp32_ub",
                                                   scope=tik.scope_ubuf)

        ori_input_shape = (hi, self.wi, C0)
        ori_output_shape = (howo_ceil16, 16, C0)
        input_data_nums = mov_len_hi * self.wi * C0
        output_data_nums = mov_len_ho * self.wo * C0

        src_input_offset = ((n_index * self.c1 + c1_index) * self.hi + start_hi_index) * \
                           self.wi * C0
        src_output_offset = ((n_index * self.c1 + c1_index) * self.ho + start_ho_index) * \
                            self.wo * C0
        repeate_time = (mov_len_ho * self.wo + 15) // 16

        ori_input_l1, ori_output_ub, grad_ub = self._data_move_ub(
            ori_input_shape, ori_output_shape, input_data_nums,
            output_data_nums, src_input_offset, src_output_offset)
        ori_input_col_ub = self.tik_instance.Tensor(self.dtype,
                                                    ori_output_shape,
                                                    name='ori_input_col_ub',
                                                    scope=tik.scope_ubuf)
        mask_shape = (ceil_div(cal_shape_ele(ori_output_ub.shape[:2]), MASK128_VALUE) *
                      MASK128_VALUE,)
        mask_not = self.tik_instance.Tensor("uint16",
                                            mask_shape,
                                            name='mask_not',
                                            scope=tik.scope_ubuf)
        mask_or = self.tik_instance.Tensor("uint16",
                                           mask_shape,
                                           name='mask_or',
                                           scope=tik.scope_ubuf)

        # init col2img_fp32_ub, if not the first one and have overlap, dump
        # the overlap part to col2img_fp32_ub, here we process whole ho, so
        # no need to move overlap part
        self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                        self.scalar_zero, "float32")

        with self.tik_instance.for_range(0, self.kh, thread_num=1) as index_h:
            with self.tik_instance.for_range(0, self.kw, thread_num=1) as index_w:
                self.tik_instance.load3dv1(ori_input_col_ub[0],
                                           ori_input_l1[0],
                                           (pad_left, pad_right, pad_top, pad_bottom),
                                           mov_len_hi, self.wi, 0,
                                           index_w, index_h,
                                           -pad_left, -pad_top,
                                           self.stride_w, self.stride_h,
                                           self.kw, self.kh, 1, 1, 1, 1,
                                           repeate_time, 0, self.pad_value)

                # calculate mask here
                mask_shape = (ceil_div(cal_shape_ele(ori_output_ub.shape[:2]), MASK128_VALUE) *
                              MASK128_VALUE,)
                mask_ub = self._calc_mask(index_h, index_w, mask_shape, ori_output_ub,
                                         ori_input_col_ub, mask_or, mask_not)
                grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub)
                grad_sel_ub_fp32 = self.tik_instance.Tensor("float32",
                                                            grad_sel_ub.shape,
                                                            name='grad_sel_ub_fp32',
                                                            scope=tik.scope_ubuf)
                self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                           cal_shape_ele(grad_sel_ub.shape), "float16")
                with self.tik_instance.for_range(0, mov_len_ho) as ho_idx:
                    col_index = index_h * wi * C0 + index_w * C0 + wi * C0 * self.stride_h * ho_idx
                    mask_index = self.wo * C0 * ho_idx
                    self._vector_op("vadd", col2img_fp32_ub[col_index], grad_sel_ub_fp32[mask_index],
                                   col2img_fp32_ub[col_index], "float32", self.wo * C0 // 2,
                                   stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                 self.stride_w * 16, self.stride_w * 16, 16))
                    self._vector_op("vadd", col2img_fp32_ub[col_index + 8],
                                   grad_sel_ub_fp32[mask_index + 8],
                                   col2img_fp32_ub[col_index + 8], "float32", self.wo * C0 // 2,
                                   stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                 self.stride_w * 16, self.stride_w * 16, 16))

        col2img_fp16_ub = self.tik_instance.Tensor("float16",
                                                   col2img_ub_shape,
                                                   name="col2img_fp16_ub",
                                                   scope=tik.scope_ubuf)
        self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                   cal_shape_ele(col2img_fp32_ub.shape), "float32")

        if offset_gm_block is None:
            pad_top_offset = pad_top * wi * C0
            self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                        col2img_fp16_ub[pad_top_offset + pad_left * C0],
                                        0, self.hi, self.wi * C0 // 16,
                                        pad_left + pad_right, 0)
        else:
            with self.tik_instance.if_scope(start_threshold > pad_top):
                self.tik_instance.data_move(self.res_gm[offset_gm_block],
                                            col2img_fp16_ub[
                                                start_threshold * wi * C0 + pad_left * C0],
                                            0, each_process_hi_block, self.wi * C0 // 16,
                                            pad_left + pad_right, 0)

            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.res_gm[offset_gm_block],
                                            col2img_fp16_ub[pad_top * wi * C0 + pad_left * C0],
                                            0, each_process_hi_block, self.wi * C0 // 16,
                                            pad_left + pad_right, 0)

    def _tilling_ho_only(self, each_process_ho, n_index, c1_index,
                        each_process_ho_block, each_process_hi_block,
                        mov_len_ho, mov_len_hi,
                        start_ho_index, start_hi_index,
                        start_threshold,
                        offset_gm_block, shape, pad):

        pad_left, pad_right, pad_top, pad_bottom = pad
        shape_ho, shape_wo, shape_hi, shape_wi = shape
        cut_ho_nums = mov_len_ho // each_process_ho
        remain_ho_nums = mov_len_ho % each_process_ho
        wi = self.wi + pad_left + pad_right

        if self.kh > self.stride_h:
            each_process_hi = (each_process_ho - 1) * self.stride_h + self.kh
            temp_size = ((self.kh - self.stride_h), wi, C0)

        else:
            each_process_hi = each_process_ho * self.stride_h
            temp_size = (1, 16, C0)
        temp_tensor_ub = self.tik_instance.Tensor("float32", temp_size,
                                                  name="temp_tensor_ub", scope=tik.scope_ubuf)

        each_process_ho_wo_div16 = ceil_div(each_process_ho * self.wo, 16)
        ori_input_shape = (shape_hi, self.wi, C0)
        ori_output_shape = (each_process_ho_wo_div16, 16, C0)
        ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape, name='ori_input_l1',
                                                scope=tik.scope_cbuf)

        col2img_ub_shape = (each_process_hi, wi, C0)
        col2img_fp32_ub = self.tik_instance.Tensor("float32", col2img_ub_shape,
                                                   name="col2img_fp32_ub", scope=tik.scope_ubuf)

        input_data_nums = mov_len_hi * self.wi * C0
        self.tik_instance.data_move(ori_input_l1[0],
                                    self.ori_input_gm[
                                        ((n_index * self.c1 + c1_index) * self.hi + start_hi_index
                                         ) * self.wi * C0],
                                    0, 1, input_data_nums // 16, 0, 0)

        if offset_gm_block is not None:
            self.offset_gm.set_as(offset_gm_block)
            remained_hi = self.tik_instance.Scalar(dtype='int64', name='remained_hi')
            remained_hi.set_as(each_process_hi_block)

        def process_ho(output_data_nums, cut_ho_nums_index, each_valid_ho, remained_hi):
            self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                            self.scalar_zero, "float32")
            if self.kh > self.stride_h:
                with self.tik_instance.if_scope(cut_ho_nums_index > 0):
                    self._vector_op("vmuls", temp_tensor_ub, 1.0, col2img_fp32_ub,
                                   temp_tensor_ub.dtype, cal_shape_ele(temp_tensor_ub.shape))

            start_h = self.tik_instance.Scalar(dtype='int64', name='start_h')
            end_h = self.tik_instance.Scalar(dtype='int64', name='end_h')
            start_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h)
            end_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h + each_process_hi)

            src_output_offset = ((n_index * self.c1 + c1_index) * self.ho + start_ho_index +
                                 each_process_ho * cut_ho_nums_index) * self.wo * C0
            _, ori_output_ub, grad_ub = self._data_move_ub(None,
                                                          ori_output_shape,
                                                          None,
                                                          output_data_nums,
                                                          None,
                                                          src_output_offset)

            ori_input_col_ub = self.tik_instance.Tensor(self.dtype,
                                                        ori_output_shape,
                                                        name='ori_input_col_ub',
                                                        scope=tik.scope_ubuf)
            mask_shape = (ceil_div(cal_shape_ele(ori_output_ub.shape[:2]), MASK128_VALUE) *
                          MASK128_VALUE,)
            mask_not = self.tik_instance.Tensor("uint16",
                                                mask_shape,
                                                name='mask_not',
                                                scope=tik.scope_ubuf)
            mask_or = self.tik_instance.Tensor("uint16",
                                               mask_shape,
                                               name='mask_or',
                                               scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.kh, thread_num=1) as index_h:
                with self.tik_instance.for_range(0, self.kw, thread_num=1) as index_w:
                    self._load3d(index_h, index_w, start_h, end_h,
                                ori_input_col_ub,
                                ori_input_l1, start_h - pad_top,
                                each_process_hi, self.wi,
                                each_process_ho_wo_div16,
                                pad, self.pad_value, -pad_left, mov_len_hi)
                    mask_shape = (ceil_div(cal_shape_ele(ori_output_ub.shape[:2]), MASK128_VALUE) *
                                  MASK128_VALUE,)
                    mask_ub = self._calc_mask(index_h, index_w, mask_shape, ori_output_ub,
                                             ori_input_col_ub, mask_or, mask_not)
                    grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub)
                    grad_sel_ub_fp32 = self.tik_instance.Tensor("float32",
                                                                grad_sel_ub.shape,
                                                                name='grad_sel_ub_fp32',
                                                                scope=tik.scope_ubuf)
                    self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                               cal_shape_ele(grad_sel_ub.shape), "float16")

                    with self.tik_instance.for_range(0, each_valid_ho) as h_idx:
                        col_index = index_h * wi * C0 + index_w * C0 + \
                                    wi * C0 * self.stride_h * h_idx
                        mask_idx = self.wo * C0 * h_idx
                        self._vector_op("vadd", col2img_fp32_ub[col_index],
                                       grad_sel_ub_fp32[mask_idx],
                                       col2img_fp32_ub[col_index], "float32", self.wo * C0 // 2,
                                       stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                     self.stride_w * 16, self.stride_w * 16, 16))
                        self._vector_op("vadd", col2img_fp32_ub[col_index + 8],
                                       grad_sel_ub_fp32[mask_idx + 8],
                                       col2img_fp32_ub[col_index + 8], "float32", self.wo * C0 // 2,
                                       stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                     self.stride_w * 16, self.stride_w * 16, 16))

            if self.tile_h_to_block:
                start_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h)
                end_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h +
                             each_process_ho * self.stride_h)
                with self.tik_instance.if_scope(remain_ho_nums == 0):
                    with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums - 1):
                        end_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h + (
                                each_process_ho - 1) * self.stride_h + self.kh)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums):
                        end_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h + (
                                each_process_ho - 1) * self.stride_h + self.kh)

                remained_hi = self._move_func_block(cut_ho_nums_index, cut_ho_nums,
                                                   start_h, end_h, each_process_ho,
                                                   each_process_hi_block, col2img_fp32_ub,
                                                   temp_tensor_ub,
                                                   remained_hi, remain_ho_nums,
                                                   (pad_left, pad_right, start_threshold,
                                                    pad_bottom))


            else:
                self._mov_func(cut_ho_nums_index, cut_ho_nums, remain_ho_nums, each_process_ho,
                              each_process_hi, each_valid_ho, col2img_fp32_ub, temp_tensor_ub, pad)
            return remained_hi

        with self.tik_instance.for_range(0, cut_ho_nums) as cut_ho_nums_index:
            output_data_nums = each_process_ho * self.wo * C0
            if self.tile_h_to_block:
                remained_hi = process_ho(output_data_nums, cut_ho_nums_index, each_process_ho,
                                         remained_hi)
            else:
                process_ho(output_data_nums, cut_ho_nums_index, each_process_ho, None)

        if not self.tile_h_to_block:
            if remain_ho_nums > 0:
                output_data_nums = remain_ho_nums * self.wo * C0
                process_ho(output_data_nums, cut_ho_nums, remain_ho_nums, None)
        else:
            with self.tik_instance.if_scope(remain_ho_nums > 0):
                output_data_nums = remain_ho_nums * self.wo * C0
                process_ho(output_data_nums, cut_ho_nums, remain_ho_nums, remained_hi)

    def _tilling_l1_ho_only(self, each_process_ho, n_index, c1_index,
                           each_process_ho_block, each_process_hi_block,
                           mov_len_ho, mov_len_hi,
                           start_ho_index, start_hi_index,
                           start_threshold,
                           offset_gm_block, shape, pad):
        pad_left, pad_right, pad_top, pad_bottom = pad
        shape_ho, shape_wo, shape_hi, shape_wi = shape
        wi = self.wi + pad_left + pad_right
        # if cut self.ho, every time process each_process_ho * self.wo
        cut_ho_nums = mov_len_ho // each_process_ho
        remain_ho_nums = mov_len_ho % each_process_ho

        start_h = self.tik_instance.Scalar(dtype='int64', name='start_h')
        end_h = self.tik_instance.Scalar(dtype='int64', name='end_h')
        hi_max = self.tik_instance.Scalar(dtype='int64', name='hi_max')
        hi_min = self.tik_instance.Scalar(dtype='int64', name='hi_min')
        mov_len_h = self.tik_instance.Scalar(dtype='int64', name='mov_len_h')
        start_pos_h = self.tik_instance.Scalar(dtype='int64', name='start_pos_h')

        # each loop process each_process_ho * self.wo * CO
        if self.kh > self.stride_h:
            each_process_hi = (each_process_ho - 1) * self.stride_h + self.kh
            temp_size = ((self.kh - self.stride_h), wi, C0)

        else:
            each_process_hi = each_process_ho * self.stride_h
            temp_size = (1, 16, C0)
        temp_tensor_ub = self.tik_instance.Tensor("float32", temp_size,
                                                  name="temp_tensor_ub", scope=tik.scope_ubuf)

        each_process_ho_wo_div16 = ceil_div(each_process_ho * self.wo, 16)
        # define col res, init to zero
        col2img_ub_shape = (each_process_hi, wi, C0)
        col2img_fp32_ub = self.tik_instance.Tensor("float32", col2img_ub_shape,
                                                   name="col2img_fp32_ub",
                                                   scope=tik.scope_ubuf)
        self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                        self.scalar_zero, "float32")

        # one times the number of (each_process_hi, self.wi, C0) can be storaged on L1
        n_each_process_hi_block = \
            (L1_SIZE // 2 - cal_shape_ele((each_process_hi, self.wi, C0))) // \
            (each_process_ho * self.stride_h * self.wi * C0) + 1
        # times of process all (each_process_hi, self.wi, C0) blocks
        n_hi_block = cut_ho_nums // n_each_process_hi_block
        # remains of (each_process_hi, self.wi, C0) blocks
        remain_hi_block = cut_ho_nums % n_each_process_hi_block
        if offset_gm_block is not None:
            self.offset_gm.set_as(offset_gm_block)
            remained_hi = self.tik_instance.Scalar(dtype='int64', name='remained_hi')
            remained_hi.set_as(each_process_hi_block)

        def process_tiling_l1_ho(n_hi_block_index, n_each_process_hi_block_index, start_h, end_h,
                                 start_pos_h, src_output_offset,
                                 output_data_nums, each_valid_ho, remained_hi, remain):
            # init col2img buffer each loop
            self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                            self.scalar_zero, "float32")
            if self.kh > self.stride_h:
                with self.tik_instance.if_scope(
                        tik.any(n_hi_block_index > 0, n_each_process_hi_block_index > 0)):
                    self._vector_op("vmuls", temp_tensor_ub, 1.0, col2img_fp32_ub,
                                   temp_tensor_ub.dtype, cal_shape_ele(temp_tensor_ub.shape))
            ori_output_shape = (ceil_div(each_process_ho * self.wo, 16), 16, C0)
            _, ori_output_ub, grad_ub = self._data_move_ub(None,
                                                          ori_output_shape,
                                                          None,
                                                          output_data_nums,
                                                          None,
                                                          src_output_offset)

            ori_input_col_ub = self.tik_instance.Tensor(self.dtype,
                                                        ori_output_shape,
                                                        name='ori_input_col_ub',
                                                        scope=tik.scope_ubuf)
            mask_shape = (ceil_div(cal_shape_ele(ori_output_ub.shape[:2]), MASK128_VALUE) *
                          MASK128_VALUE,)
            mask_not = self.tik_instance.Tensor("uint16",
                                                mask_shape,
                                                name='mask_not',
                                                scope=tik.scope_ubuf)
            mask_or = self.tik_instance.Tensor("uint16",
                                               mask_shape,
                                               name='mask_or',
                                               scope=tik.scope_ubuf)

            with self.tik_instance.for_range(0, self.kh, thread_num=1) as index_h:
                with self.tik_instance.for_range(0, self.kw, thread_num=1) as index_w:
                    self._load3d(index_h, index_w, start_h, end_h,
                                ori_input_col_ub,
                                ori_input_l1, start_pos_h,
                                each_process_hi, self.wi,
                                each_process_ho_wo_div16,
                                pad, self.pad_value,
                                -pad_left, self.hi)

                    mask_ub = self._calc_mask(index_h, index_w, mask_shape, ori_output_ub,
                                             ori_input_col_ub, mask_or, mask_not)
                    grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub)
                    grad_sel_ub_fp32 = self.tik_instance.Tensor("float32",
                                                                grad_sel_ub.shape,
                                                                name='grad_sel_ub_fp32',
                                                                scope=tik.scope_ubuf)
                    self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                               cal_shape_ele(grad_sel_ub.shape), "float16")
                    with self.tik_instance.for_range(0, each_valid_ho) as h_idx:
                        col_index = index_h * wi * C0 + index_w * C0 + \
                                    wi * C0 * self.stride_h * h_idx
                        mask_idx = self.wo * C0 * h_idx
                        self._vector_op("vadd", col2img_fp32_ub[col_index],
                                       grad_sel_ub_fp32[mask_idx],
                                       col2img_fp32_ub[col_index], "float32", self.wo * C0 // 2,
                                       stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                     self.stride_w * 16, self.stride_w * 16, 16))
                        self._vector_op("vadd", col2img_fp32_ub[col_index + 8],
                                       grad_sel_ub_fp32[mask_idx + 8],
                                       col2img_fp32_ub[col_index + 8], "float32", self.wo * C0 // 2,
                                       stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                     self.stride_w * 16, self.stride_w * 16, 16))

            cut_ho_nums_index = n_hi_block_index * n_each_process_hi_block + \
                                n_each_process_hi_block_index
            if self.tile_h_to_block:
                remained_hi = self._move_func_block(cut_ho_nums_index, cut_ho_nums,
                                                   start_h, end_h, each_process_ho,
                                                   each_process_hi_block, col2img_fp32_ub,
                                                   temp_tensor_ub,
                                                   remained_hi, remain,
                                                   (pad_left, pad_right, start_threshold,
                                                    pad_bottom))
            else:
                self._mov_func(cut_ho_nums_index, cut_ho_nums, remain_ho_nums, each_process_ho,
                              each_process_hi, each_valid_ho, col2img_fp32_ub, temp_tensor_ub, pad)
            return remained_hi

        with self.tik_instance.for_range(0, n_hi_block) as n_hi_block_index:
            start_h.set_as(
                n_hi_block_index * n_each_process_hi_block * each_process_ho * self.stride_h)
            end_h.set_as(n_hi_block_index * n_each_process_hi_block *
                         each_process_ho * self.stride_h + each_process_hi +
                         (n_each_process_hi_block - 1) * each_process_ho * self.stride_h)
            ori_input_shape = (
                each_process_hi + (n_each_process_hi_block - 1) * each_process_ho * self.stride_h,
                self.wi, C0)
            ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape,
                                                    name='ori_input_l1',
                                                    scope=tik.scope_cbuf)
            self.tik_instance.scalar_max(hi_min, pad_top, start_h)
            hi_max.set_as(self.hi + pad_top)
            self.tik_instance.scalar_min(hi_max, hi_max, end_h)
            mov_len_h.set_as(hi_max - hi_min)
            self.tik_instance.data_move(ori_input_l1[0],
                                        self.ori_input_gm[((n_index * self.c1 + c1_index) *
                                                           self.hi + start_hi_index +
                                                           hi_min - pad_top) *
                                                          self.wi * C0],
                                        0, mov_len_h, self.wi * C0 // 16, 0, 0)

            with self.tik_instance.for_range(0, n_each_process_hi_block) as \
                    n_each_process_hi_block_index:
                start_h.set_as((n_hi_block_index * n_each_process_hi_block +
                                n_each_process_hi_block_index) *
                               each_process_ho * self.stride_h)
                end_h.set_as((n_hi_block_index * n_each_process_hi_block +
                              n_each_process_hi_block_index) *
                             each_process_ho * self.stride_h +
                             each_process_ho * self.stride_h)

                start_pos_h.set_as(
                    n_hi_block_index * n_each_process_hi_block * each_process_ho * self.stride_h)
                with self.tik_instance.if_scope(start_pos_h > pad_top):
                    start_pos_h.set_as(n_each_process_hi_block_index * each_process_ho *
                                       self.stride_h)
                with self.tik_instance.else_scope():
                    start_pos_h.set_as(n_each_process_hi_block_index * each_process_ho *
                                       self.stride_h - pad_top)
                self.tik_instance.scalar_max(start_pos_h, start_pos_h, 0)

                output_data_nums = each_process_ho * self.wo * C0
                src_output_offset = ((n_index * self.c1 + c1_index) * self.ho + start_ho_index +
                                     each_process_ho * (n_hi_block_index *
                                                        n_each_process_hi_block +
                                                        n_each_process_hi_block_index)) * \
                                    self.wo * C0

                if self.tile_h_to_block:
                    remain0 = self.tik_instance.Scalar(dtype='int64', name='remain0')
                    remain1 = self.tik_instance.Scalar(dtype='int64', name='remain')
                    remain0.set_as(remain_hi_block)
                    remain1.set_as(remain_ho_nums)
                    self.tik_instance.scalar_max(remain0, remain0, remain1)
                    remained_hi = process_tiling_l1_ho(n_hi_block_index,
                                                       n_each_process_hi_block_index,
                                                       start_h, end_h, start_pos_h,
                                                       src_output_offset,
                                                       output_data_nums, each_process_ho,
                                                       remained_hi, remain0)
                else:
                    process_tiling_l1_ho(n_hi_block_index, n_each_process_hi_block_index,
                                         start_h, end_h, start_pos_h, src_output_offset,
                                         output_data_nums, each_process_ho, None, None)

        if offset_gm_block is None:
            if remain_hi_block != 0:
                start_h.set_as(
                    n_hi_block * n_each_process_hi_block * each_process_ho * self.stride_h)
                end_h.set_as(
                    n_hi_block * n_each_process_hi_block * each_process_ho * self.stride_h +
                    each_process_hi + (remain_hi_block - 1) * each_process_ho * self.stride_h)

                ori_input_shape = (each_process_hi + (remain_hi_block - 1) *
                                   each_process_ho * self.stride_h, self.wi, C0)
                ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape,
                                                        name='ori_input_l1',
                                                        scope=tik.scope_cbuf)

                self.tik_instance.scalar_max(hi_min, pad_top, start_h)
                hi_max.set_as(self.hi + pad_top)
                self.tik_instance.scalar_min(hi_max, hi_max, end_h)
                mov_len_h.set_as(hi_max - hi_min)
                self.tik_instance.data_move(ori_input_l1[0],
                                            self.ori_input_gm[
                                                ((n_index * self.c1 + c1_index) * self.hi +
                                                 hi_min - pad_top) * self.wi * C0],
                                            0, mov_len_h, self.wi * C0 // 16, 0, 0)

                with self.tik_instance.for_range(0, remain_hi_block) as remain_hi_n_index:
                    output_data_nums = each_process_ho * self.wo * C0
                    src_output_offset = ((n_index * self.c1 + c1_index) * self.ho +
                                         each_process_ho * (n_hi_block * n_each_process_hi_block +
                                                            remain_hi_n_index)) * self.wo * C0
                    start_h.set_as((n_hi_block * n_each_process_hi_block + remain_hi_n_index) *
                                   each_process_ho * self.stride_h)
                    end_h.set_as((n_hi_block * n_each_process_hi_block + remain_hi_n_index) *
                                 each_process_ho * self.stride_h + each_process_ho * self.stride_h)
                    if n_hi_block == 0:
                        start_pos_h.set_as(
                            remain_hi_n_index * each_process_ho * self.stride_h - pad_top)
                    else:
                        start_pos_h.set_as(remain_hi_n_index * each_process_ho * self.stride_h)

                    process_tiling_l1_ho(n_hi_block, remain_hi_n_index, start_h, end_h,
                                         start_pos_h, src_output_offset,
                                         output_data_nums, each_process_ho, None, None)
            if remain_ho_nums != 0:
                input_data_num = (self.hi + pad_top - cut_ho_nums * each_process_ho *
                                  self.stride_h) * self.wi * C0
                ori_input_shape = (
                    self.hi + pad_top - cut_ho_nums * each_process_ho * self.stride_h, self.wi, C0)
                ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape,
                                                        name='ori_input_l1', scope=tik.scope_cbuf)
                start_h.set_as((n_index * self.c1 + c1_index) * self.hi * self.wi *
                               C0 + (cut_ho_nums * each_process_ho *
                                     self.stride_h - pad_top) * self.wi * C0)

                self.tik_instance.data_move(ori_input_l1[0],
                                            self.ori_input_gm[(n_index * self.c1 + c1_index) *
                                                              self.hi * self.wi *
                                                              C0 + (cut_ho_nums * each_process_ho *
                                                                    self.stride_h - pad_top) *
                                                              self.wi * C0],
                                            0, 1, input_data_num // 16, 0, 0)

                each_process_ho_wo_div16 = (remain_ho_nums * self.wo + 15) // 16
                start_h.set_as(cut_ho_nums * each_process_ho * self.stride_h)
                if self.stride_h >= self.kh:
                    each_process_hi = remain_ho_nums * self.stride_h
                    end_h.set_as(cut_ho_nums * each_process_ho * self.stride_h + (
                            remain_ho_nums - 1) * self.stride_h + self.kh)

                else:
                    each_process_hi = (remain_ho_nums - 1) * self.stride_h + self.kh
                    end_h.set_as(
                        cut_ho_nums * each_process_ho * self.stride_h +
                        remain_ho_nums * self.stride_h)
                each_process_hi = (remain_ho_nums - 1) * self.stride_h + self.kh

                output_data_nums = remain_ho_nums * self.wo * C0
                src_output_offset = ((n_index * self.c1 + c1_index) * self.ho +
                                     each_process_ho * cut_ho_nums) * self.wo * C0

                process_tiling_l1_ho(n_hi_block, remain_hi_block, start_h, end_h,
                                     0, src_output_offset,
                                     output_data_nums, remain_ho_nums, None, None)
        else:
            with self.tik_instance.if_scope(remain_hi_block != 0):
                start_h.set_as(
                    n_hi_block * n_each_process_hi_block * each_process_ho * self.stride_h)
                end_h.set_as(
                    n_hi_block * n_each_process_hi_block * each_process_ho * self.stride_h +
                    each_process_hi + (remain_hi_block - 1) * each_process_ho * self.stride_h)

                ori_input_shape = (each_process_hi + (n_each_process_hi_block - 1) *
                                   each_process_ho * self.stride_h, self.wi, C0)
                ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape,
                                                        name='ori_input_l1',
                                                        scope=tik.scope_cbuf)

                self.tik_instance.scalar_max(hi_min, pad_top, start_h)
                hi_max.set_as(self.hi + pad_top)
                self.tik_instance.scalar_min(hi_max, hi_max, end_h)
                mov_len_h.set_as(hi_max - hi_min)
                self.tik_instance.data_move(ori_input_l1[0],
                                            self.ori_input_gm[
                                                ((n_index * self.c1 + c1_index) *
                                                 self.hi + start_hi_index +
                                                 hi_min - pad_top) * self.wi * C0],
                                            0, mov_len_h, self.wi * C0 // 16, 0, 0)

                with self.tik_instance.for_range(0, remain_hi_block) as remain_hi_n_index:
                    output_data_nums = each_process_ho * self.wo * C0
                    src_output_offset = ((n_index * self.c1 + c1_index) * self.ho + start_ho_index +
                                         each_process_ho * (n_hi_block * n_each_process_hi_block +
                                                            remain_hi_n_index)) * self.wo * C0
                    start_h.set_as((n_hi_block * n_each_process_hi_block + remain_hi_n_index) *
                                   each_process_ho * self.stride_h)
                    end_h.set_as((n_hi_block * n_each_process_hi_block + remain_hi_n_index) *
                                 each_process_ho * self.stride_h + each_process_ho * self.stride_h)
                    with self.tik_instance.if_scope(n_hi_block == 0):
                        start_pos_h.set_as(
                            remain_hi_n_index * each_process_ho * self.stride_h - pad_top)
                    with self.tik_instance.else_scope():
                        start_pos_h.set_as(remain_hi_n_index * each_process_ho * self.stride_h)

                    remained_hi = process_tiling_l1_ho(n_hi_block, remain_hi_n_index, start_h,
                                                       end_h,
                                                       start_pos_h, src_output_offset,
                                                       output_data_nums, each_process_ho,
                                                       remained_hi, remain_ho_nums)

            with self.tik_instance.if_scope(remain_ho_nums != 0):
                input_data_num = (mov_len_hi + pad_top - cut_ho_nums * each_process_ho *
                                  self.stride_h) * self.wi * C0
                ori_input_shape = (each_process_hi, self.wi, C0)
                ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape,
                                                        name='ori_input_l1', scope=tik.scope_cbuf)
                start_h.set_as((n_index * self.c1 + c1_index) * self.hi * self.wi *
                               C0 + (cut_ho_nums * each_process_ho *
                                     self.stride_h - pad_top) * self.wi * C0)

                self.tik_instance.data_move(ori_input_l1[0],
                                            self.ori_input_gm[((n_index * self.c1 + c1_index) *
                                                               self.hi + start_hi_index) * self.wi *
                                                              C0 + (cut_ho_nums * each_process_ho *
                                                                    self.stride_h - pad_top) *
                                                              self.wi * C0],
                                            0, 1, input_data_num // 16, 0, 0)

                each_process_ho_wo_div16 = (remain_ho_nums * self.wo + 15) // 16
                start_h.set_as(cut_ho_nums * each_process_ho * self.stride_h)
                if self.stride_h >= self.kh:
                    each_process_hi = remain_ho_nums * self.stride_h
                    end_h.set_as(cut_ho_nums * each_process_ho * self.stride_h + (
                            remain_ho_nums - 1) * self.stride_h + self.kh)

                else:
                    each_process_hi = (remain_ho_nums - 1) * self.stride_h + self.kh
                    end_h.set_as(
                        cut_ho_nums * each_process_ho * self.stride_h +
                        remain_ho_nums * self.stride_h)
                each_process_hi = (remain_ho_nums - 1) * self.stride_h + self.kh

                output_data_nums = remain_ho_nums * self.wo * C0
                src_output_offset = ((n_index * self.c1 + c1_index) * self.ho + start_ho_index +
                                     each_process_ho * cut_ho_nums) * self.wo * C0

                process_tiling_l1_ho(n_hi_block, remain_hi_block, start_h, end_h,
                                     0, src_output_offset,
                                     output_data_nums, remain_ho_nums, remained_hi, 0)

    def _tilling_l1_ho_wo(self, each_process_wo, n_index, c1_index,
                         each_process_ho_block, each_process_hi_block,
                         mov_len_ho, mov_len_hi,
                         start_ho_index, start_hi_index,
                         start_threshold,
                         offset_gm_block, shape, pad):
        pad_left, pad_right, pad_top, pad_bottom = pad
        start_h = self.tik_instance.Scalar(dtype='int64', name='start_h')
        end_h = self.tik_instance.Scalar(dtype='int64', name='end_h')
        hi_max = self.tik_instance.Scalar(dtype='int64', name='hi_max')
        hi_min = self.tik_instance.Scalar(dtype='int64', name='hi_min')
        mov_len_h = self.tik_instance.Scalar(dtype='int64', name='mov_len_h')
        start_pos_h = self.tik_instance.Scalar(dtype='int64', name='start_pos_h')
        start_w = self.tik_instance.Scalar(dtype='int64', name='start_h')
        end_w = self.tik_instance.Scalar(dtype='int64', name='end_w')
        wi_max = self.tik_instance.Scalar(dtype='int64', name='wi_max')
        wi_min = self.tik_instance.Scalar(dtype='int64', name='wi_min')
        mov_len_w = self.tik_instance.Scalar(dtype='int64', name='mov_len_w')
        start_pos_w = self.tik_instance.Scalar(dtype='int64', name='start_pos_w')
        remained_hi = self.tik_instance.Scalar(dtype='int64', name='remained_hi')
        remained_hi.set_as(each_process_hi_block)

        cut_wo_nums = self.wo // each_process_wo
        remain_wo_nums = self.wo % each_process_wo

        if self.stride_w >= self.kw:
            each_process_wi = each_process_wo * self.stride_w
        else:
            each_process_wi = (each_process_wo - 1) * self.stride_w + self.kw

        if self.stride_h >= self.kh:
            each_process_hi = self.stride_h
        else:
            each_process_hi = self.kh
        each_process_wo_div16 = ceil_div(each_process_wo, 16)

        # define col res, init to zero
        col2img_ub_shape = (each_process_hi, each_process_wi, C0)
        col2img_fp32_ub = self.tik_instance.Tensor("float32", col2img_ub_shape,
                                                   name="col2img_fp32_ub",
                                                   scope=tik.scope_ubuf)
        if offset_gm_block is not None:
            self.offset_gm.set_as(offset_gm_block)

        if self.stride_h < self.kh:
            overlap_l1_shape = (self.kh - self.stride_h, (self.wi + pad_left + pad_right) * C0)

            overlap_l1 = self.tik_instance.Tensor('float32', overlap_l1_shape,
                                                  name='overlap_l1',
                                                  scope=tik.scope_cbuf)
            overlap_l1_h, overlap_l1_w = overlap_l1_shape

        with self.tik_instance.for_range(0, mov_len_ho, thread_num=1) as ho_index:
            start_h.set_as(ho_index * self.stride_h)
            end_h.set_as(ho_index * self.stride_h + each_process_hi)
            # mov actual non pad ori input to L1
            ori_input_shape = (each_process_hi, self.wi + pad_left + pad_right, C0)
            ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape,
                                                    name='ori_input_l1',
                                                    scope=tik.scope_cbuf)

            self.tik_instance.scalar_max(hi_min, pad_top, start_h)
            hi_max.set_as(mov_len_hi + pad_top)
            self.tik_instance.scalar_min(hi_max, hi_max, end_h)
            mov_len_h.set_as(hi_max - hi_min)

            self.tik_instance.data_move(
                ori_input_l1[0],
                self.ori_input_gm[((n_index * self.c1 + c1_index) *
                                   self.hi + start_hi_index + hi_min - pad_top) * self.wi * C0],
                0, mov_len_h, self.wi * C0 // 16, 0, 0)

            offset_gm_inside = self.tik_instance.Scalar(dtype='int64',
                                                        name='offset_gm_inside')
            offset_gm_inside.set_as(self.offset_gm)

            # init col2img after every looph
            self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                            self.scalar_zero, "float32")

            with self.tik_instance.for_range(0, cut_wo_nums, thread_num=1) as cut_wo_nums_index:
                if self.kh > self.stride_h:
                    with self.tik_instance.if_scope(ho_index != 0):
                        with self.tik_instance.if_scope(cut_wo_nums_index == 0):
                            with self.tik_instance.for_range(0,
                                                             self.kh - self.stride_h) as index_khs:
                                self.tik_instance.data_move(
                                    col2img_fp32_ub[index_khs * each_process_wi * C0],
                                    overlap_l1[index_khs * overlap_l1_w],
                                    0, 1, each_process_wi * C0 // 8,
                                    0, 0)

                        with self.tik_instance.else_scope():
                            start_pos = (each_process_wi - self.stride_w * each_process_wo) * C0
                            with self.tik_instance.for_range(0,
                                                             self.kh - self.stride_h) as index_khs:
                                self.tik_instance.data_move(
                                    col2img_fp32_ub[index_khs * each_process_wi * C0 + start_pos],
                                    overlap_l1[index_khs * overlap_l1_w +
                                               cut_wo_nums_index * each_process_wo *
                                               self.stride_w * C0 + start_pos],
                                    0, 1, self.stride_w * each_process_wo * C0 // 8,
                                    0, 0)

                ori_output_shape = (each_process_wo_div16, 16, C0)
                output_data_nums = each_process_wo * C0
                src_output_offset = ((n_index * self.c1 + c1_index) * self.ho +
                                     start_ho_index + ho_index) * self.wo * C0 + \
                                    cut_wo_nums_index * each_process_wo * C0
                _, ori_output_ub, grad_ub = self._data_move_ub(
                    None, ori_output_shape, None,
                    output_data_nums, None, src_output_offset)

                ori_input_col_ub = self.tik_instance.Tensor(self.dtype,
                                                            ori_output_shape,
                                                            name='ori_input_col_ub',
                                                            scope=tik.scope_ubuf)
                start_w.set_as(
                    cut_wo_nums_index * each_process_wo * self.stride_w)
                end_w.set_as(
                    cut_wo_nums_index * each_process_wo * self.stride_w + each_process_wi)

                # load3d to get col
                wo_offset = self.tik_instance.Scalar(dtype='int64',
                                                     name='wo_offset')
                wo_offset.set_as(
                    each_process_wo * cut_wo_nums_index * self.stride_w - pad_left)

                mask_shape = (ceil_div(cal_shape_ele(ori_output_ub.shape[:2]), MASK128_VALUE) *
                              MASK128_VALUE,)
                mask_not = self.tik_instance.Tensor("uint16",
                                                    mask_shape,
                                                    name='mask_not',
                                                    scope=tik.scope_ubuf)
                mask_or = self.tik_instance.Tensor("uint16",
                                                   mask_shape,
                                                   name='mask_or',
                                                   scope=tik.scope_ubuf)

                with self.tik_instance.for_range(0, self.kh,
                                                 thread_num=1) as index_h:
                    with self.tik_instance.for_range(0, self.kw,
                                                     thread_num=1) as index_w:
                        ori_input_col_ub = self._load3d(index_h, index_w,
                                                       start_h, end_h,
                                                       ori_input_col_ub,
                                                       ori_input_l1, 0,
                                                       each_process_hi, self.wi,
                                                       each_process_wo_div16,
                                                       pad, self.pad_value,
                                                       wo_offset, mov_len_hi)
                        mask_ub = self._calc_mask(index_h, index_w, mask_shape, ori_output_ub,
                                                 ori_input_col_ub, mask_or, mask_not)
                        grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub)
                        grad_sel_ub_fp32 = self.tik_instance.Tensor("float32",
                                                                    grad_sel_ub.shape,
                                                                    name='grad_sel_ub_fp32',
                                                                    scope=tik.scope_ubuf)
                        self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                                   cal_shape_ele(grad_sel_ub.shape), "float16")

                        with self.tik_instance.for_range(0, 1) as h_idx:
                            col_index = index_h * each_process_wi * C0 + index_w * C0 + \
                                        each_process_wi * C0 * self.stride_h * h_idx
                            mask_idx = each_process_wo * C0 * h_idx

                            self._vector_op("vadd", col2img_fp32_ub[col_index],
                                           grad_sel_ub_fp32[mask_idx],
                                           col2img_fp32_ub[col_index], "float32",
                                           each_process_wo * C0 // 2,
                                           stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                         self.stride_w * 16, self.stride_w * 16,
                                                         16))
                            self._vector_op("vadd", col2img_fp32_ub[col_index + 8],
                                           grad_sel_ub_fp32[mask_idx + 8],
                                           col2img_fp32_ub[col_index + 8], "float32",
                                           each_process_wo * C0 // 2,
                                           stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                         self.stride_w * 16, self.stride_w * 16,
                                                         16))

                col2img_fp16_ub = self.tik_instance.Tensor("float16",
                                                           col2img_ub_shape,
                                                           name="col2img_fp16_ub",
                                                           scope=tik.scope_ubuf)
                self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                           cal_shape_ele(col2img_fp32_ub.shape), "float32")
                # set h direction's paras
                start_h.set_as(ho_index * self.stride_h)
                end_h.set_as(start_h + self.stride_h)
                with self.tik_instance.if_scope(ho_index == mov_len_ho - 1):
                    end_h.set_as(start_h + self.kh)
                if offset_gm_block is not None:
                    with self.tik_instance.if_scope(start_threshold > pad_top):
                        self.tik_instance.scalar_max(hi_min, start_threshold, start_h)
                        hi_max.set_as(each_process_hi_block + start_threshold)
                    with self.tik_instance.else_scope():
                        self.tik_instance.scalar_max(hi_min, pad_top, start_h)
                        hi_max.set_as(each_process_hi_block + pad_top)
                else:
                    self.tik_instance.scalar_max(hi_min, pad_top, start_h)
                    hi_max.set_as(each_process_hi_block + pad_top)
                self.tik_instance.scalar_min(hi_max, hi_max, end_h)
                mov_len_h.set_as(hi_max - hi_min)
                start_pos_h.set_as(hi_min - ho_index * self.stride_h)

                # set w direction's paras
                start_w.set_as(cut_wo_nums_index * each_process_wo * self.stride_w)
                end_w.set_as(cut_wo_nums_index * each_process_wo *
                             self.stride_w + each_process_wo * self.stride_w)
                self.tik_instance.scalar_max(wi_min, pad_left, start_w)
                self.tik_instance.scalar_min(wi_max, self.wi + pad_left, end_w)
                mov_len_w.set_as(wi_max - wi_min)
                start_pos_w.set_as(wi_min - cut_wo_nums_index *
                                   each_process_wo * self.stride_w)

                with self.tik_instance.if_scope(
                        tik.all(cut_wo_nums_index < cut_wo_nums - 1, mov_len_h > 0)):
                    self.tik_instance.data_move(self.res_gm[offset_gm_inside],
                                                col2img_fp16_ub[start_pos_h * each_process_wi *
                                                                C0 + start_pos_w * C0], 0,
                                                mov_len_h, mov_len_w * C0 // 16,
                                                each_process_wi - mov_len_w,
                                                self.wi - mov_len_w)
                    offset_gm_inside.set_as(offset_gm_inside + mov_len_w * C0)
                    self.offset_gm.set_as(self.offset_gm + mov_len_h * mov_len_w * C0)

                with self.tik_instance.if_scope(tik.all(mov_len_h > 0, mov_len_w > 0)):
                    if remain_wo_nums == 0:
                        with self.tik_instance.if_scope(cut_wo_nums_index == cut_wo_nums - 1):
                            last_valid_wi = self.wi - ((cut_wo_nums - 1) * each_process_wo *
                                                       self.stride_w - pad_left)
                            last_valid_wi = min(last_valid_wi, self.wi)

                            if last_valid_wi <= each_process_wi:
                                self.tik_instance.data_move(
                                    self.res_gm[offset_gm_inside],
                                    col2img_fp16_ub[start_pos_h * each_process_wi *
                                                    C0 + start_pos_w * C0],
                                    0, mov_len_h,
                                    last_valid_wi * C0 // 16,
                                    each_process_wi - last_valid_wi, self.wi - last_valid_wi)
                                offset_gm_inside.set_as(offset_gm_inside + last_valid_wi * C0)
                                self.offset_gm.set_as(
                                    self.offset_gm + mov_len_h * last_valid_wi * C0)
                            else:
                                self.tik_instance.data_move(self.res_gm[offset_gm_inside],
                                                            col2img_fp16_ub[
                                                                start_pos_h * each_process_wi *
                                                                C0 + start_pos_w * C0], 0,
                                                            mov_len_h, each_process_wi * C0 // 16,
                                                            0,
                                                            self.wi - each_process_wi)
                                offset_gm_inside.set_as(offset_gm_inside + each_process_wi * C0)

                                remain_wi = last_valid_wi - each_process_wi
                                temp_zero = self.tik_instance.Tensor("float16",
                                                                     (remain_wi * C0,),
                                                                     name='temp_zero',
                                                                     scope=tik.scope_ubuf)
                                self._vector_dup(temp_zero, 0, temp_zero.shape,
                                                self.scalar_zero_fp16, temp_zero.dtype)
                                with self.tik_instance.for_range(0, mov_len_h) as index_0:
                                    self.tik_instance.data_move(
                                        self.res_gm[offset_gm_inside + index_0 * self.wi * C0],
                                        temp_zero, 0, 1,
                                        cal_shape_ele(temp_zero.shape) // 16, 0, 0)
                                offset_gm_inside.set_as(offset_gm_inside + remain_wi * C0)
                                self.offset_gm.set_as(
                                    self.offset_gm + mov_len_h * last_valid_wi * C0)
                    else:
                        with self.tik_instance.if_scope(cut_wo_nums_index == cut_wo_nums - 1):
                            self.tik_instance.data_move(self.res_gm[offset_gm_inside],
                                                        col2img_fp16_ub[
                                                            start_pos_h * each_process_wi *
                                                            C0 + start_pos_w * C0], 0,
                                                        mov_len_h, mov_len_w * C0 // 16,
                                                        each_process_wi - mov_len_w,
                                                        self.wi - mov_len_w)
                            offset_gm_inside.set_as(offset_gm_inside + mov_len_w * C0)
                            self.offset_gm.set_as(self.offset_gm + mov_len_h * mov_len_w * C0)

                # move back to init col2img_fp16 tensor
                with self.tik_instance.if_scope(cut_wo_nums_index < cut_wo_nums - 1):
                    # mov h overlap to L1
                    if self.kh > self.stride_h:
                        with self.tik_instance.for_range(0, self.kh - self.stride_h) as index_s:
                            self.tik_instance.data_move(
                                overlap_l1[index_s * overlap_l1_w +
                                           cut_wo_nums_index * each_process_wo *
                                           self.stride_w * C0],
                                col2img_fp32_ub[self.stride_h *
                                                each_process_wi * C0 + each_process_wi *
                                                C0 * index_s],
                                0, 1, self.stride_w * each_process_wo * C0 // 8, 0, 0)

                    if self.kw > self.stride_w:
                        with self.tik_instance.for_range(0, self.kh) as index_kh:
                            offset = [index_kh * each_process_wi * C0,
                                      index_kh * each_process_wi * C0 +
                                      self.stride_w * each_process_wo * C0]
                            self._vector_op("vmuls", col2img_fp32_ub, 1.0, col2img_fp32_ub,
                                           col2img_fp32_ub.dtype,
                                           (each_process_wi - each_process_wo * self.stride_w) * C0,
                                           None, offset)
                        with self.tik_instance.for_range(0, self.kh) as index_kh:
                            self._vector_dup(col2img_fp32_ub,
                                            index_kh * each_process_wi * C0 +
                                            (each_process_wi - self.stride_w *
                                             each_process_wo) * C0,
                                            (self.stride_w * each_process_wo * C0,),
                                            self.scalar_zero,
                                            col2img_fp32_ub.dtype)

                    else:
                        self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                                        self.scalar_zero, col2img_fp32_ub.dtype)

                with self.tik_instance.else_scope():
                    if remain_wo_nums > 0:
                        # mov h overlap to L1
                        if self.kh > self.stride_h:
                            with self.tik_instance.for_range(0, self.kh - self.stride_h) as index_s:
                                self.tik_instance.data_move(
                                    overlap_l1[index_s * overlap_l1_w +
                                               cut_wo_nums_index * each_process_wo *
                                               self.stride_w * C0],
                                    col2img_fp32_ub[self.stride_h *
                                                    each_process_wi * C0 + each_process_wi *
                                                    C0 * index_s],
                                    0, 1, self.stride_w * each_process_wo * C0 // 8, 0, 0)

                        if self.kw > self.stride_w:
                            with self.tik_instance.for_range(0, self.kh) as index_kh:
                                offset = [index_kh * each_process_wi * C0,
                                          index_kh * each_process_wi * C0 +
                                          self.stride_w * each_process_wo * C0]
                                self._vector_op("vmuls", col2img_fp32_ub, 1.0, col2img_fp32_ub,
                                               col2img_fp32_ub.dtype,
                                               (each_process_wi - each_process_wo *
                                                self.stride_w) * C0,
                                               None, offset)
                            with self.tik_instance.for_range(0, self.kh) as index_kh:
                                self._vector_dup(col2img_fp32_ub,
                                                index_kh * each_process_wi * C0 +
                                                (each_process_wi - self.stride_w *
                                                 each_process_wo) * C0,
                                                (self.stride_w * each_process_wo * C0,),
                                                self.scalar_zero,
                                                col2img_fp32_ub.dtype)
                        else:
                            self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                                            self.scalar_zero, col2img_fp32_ub.dtype)
                    else:
                        if self.kh > self.stride_h:
                            with self.tik_instance.for_range(
                                    0, self.kh - self.stride_h) as index_s:
                                self.tik_instance.data_move(
                                    overlap_l1[index_s * overlap_l1_w + (cut_wo_nums - 1) *
                                               each_process_wo * self.stride_w * C0],
                                    col2img_fp32_ub[self.stride_h * each_process_wi * C0 +
                                                    each_process_wi * C0 * index_s],
                                    0, 1, each_process_wi * C0 // 8, 0, 0)

            if remain_wo_nums:
                each_process_remain_div16 = ceil_div(remain_wo_nums, 16)
                if self.kw > self.stride_w:
                    each_process_remain_wi = (remain_wo_nums - 1) * self.stride_w + self.kw
                else:
                    each_process_remain_wi = remain_wo_nums * self.stride_w

                start_h.set_as(ho_index * self.stride_h)
                end_h.set_as(ho_index * self.stride_h + each_process_hi)

                # move l1 to UB to init the col2img_fp16 tensor. if stride < kernel, there has
                # overlap between each ho.
                if self.kh > self.stride_h:
                    with self.tik_instance.if_scope(ho_index != 0):
                        start_pos = (each_process_wi - self.stride_w * each_process_wo) * C0
                        with self.tik_instance.for_range(0, self.kh - self.stride_h) as index_khs:
                            self.tik_instance.data_move(
                                col2img_fp32_ub[index_khs * each_process_wi * C0 + start_pos],
                                overlap_l1[index_khs * overlap_l1_w +
                                           cut_wo_nums * each_process_wo *
                                           self.stride_w * C0 + start_pos],
                                0, 1, self.stride_w * remain_wo_nums * C0 // 8, 0, 0)

                # mov forward output and grad to UB
                ori_output_ub_shape = (each_process_wo_div16, 16, C0)
                ori_output_ub = self.tik_instance.Tensor(self.dtype,
                                                         ori_output_ub_shape,
                                                         name='ori_output_ub',
                                                         scope=tik.scope_ubuf)
                self.tik_instance.data_move(
                    ori_output_ub,
                    self.ori_output_gm[((n_index * self.c1 + c1_index) *
                                        self.ho + start_ho_index + ho_index) * self.wo * C0 +
                                       cut_wo_nums * each_process_wo * C0],
                    0, 1, remain_wo_nums * C0 // 16, 0, 0)

                grad_ub = self.tik_instance.Tensor(self.dtype, ori_output_ub_shape,
                                                   name='grad_ub',
                                                   scope=tik.scope_ubuf)

                self.tik_instance.data_move(
                    grad_ub, self.grad_gm[((n_index * self.c1 + c1_index) *
                                           self.ho + start_ho_index + ho_index) * self.wo * C0 +
                                          cut_wo_nums * each_process_wo * C0],
                    0, 1, remain_wo_nums * C0 // 16, 0, 0)

                ori_input_col_ub_shape = (each_process_wo_div16 * 16, C0)
                ori_input_col_ub = self.tik_instance.Tensor(self.dtype,
                                                            ori_input_col_ub_shape,
                                                            name='ori_input_col_ub',
                                                            scope=tik.scope_ubuf)

                wo_offset = self.tik_instance.Scalar(dtype='int64',
                                                     name='wo_offset')
                wo_offset.set_as(each_process_wo * cut_wo_nums * self.stride_w - pad_left)
                mask_shape = (ceil_div(cal_shape_ele(ori_output_ub.shape[:2]), MASK128_VALUE) *
                              MASK128_VALUE,)

                mask_not = self.tik_instance.Tensor("uint16",
                                                    mask_shape,
                                                    name='mask_not',
                                                    scope=tik.scope_ubuf)
                mask_or = self.tik_instance.Tensor("uint16",
                                                   mask_shape,
                                                   name='mask_or',
                                                   scope=tik.scope_ubuf)

                # do load3d, calculate mask and grad_x, here we loop kh and kw, so each loop
                # it process one row of output, image output as a window slide on kernel window
                with self.tik_instance.for_range(0, self.kh) as index_h:
                    with self.tik_instance.for_range(0, self.kw) as index_w:
                        self._load3d(index_h, index_w,
                                    start_h, end_h,
                                    ori_input_col_ub,
                                    ori_input_l1, 0, each_process_hi,
                                    self.wi,
                                    each_process_remain_div16,
                                    pad, self.pad_value,
                                    wo_offset, mov_len_hi)
                        mask_ub = self._calc_mask(index_h, index_w, mask_shape, ori_output_ub,
                                                 ori_input_col_ub, mask_or, mask_not)
                        grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub)
                        grad_sel_ub_fp32 = self.tik_instance.Tensor("float32",
                                                                    grad_sel_ub.shape,
                                                                    name='grad_sel_ub_fp32',
                                                                    scope=tik.scope_ubuf)
                        self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                                   cal_shape_ele(grad_sel_ub.shape), "float16")

                        # each procee ho is 1, so here loop value is 1
                        with self.tik_instance.for_range(0, 1) as h_idx:
                            col_index = index_h * each_process_wi * C0 + index_w * C0 + \
                                        each_process_wi * C0 * self.stride_h * h_idx
                            mask_idx = each_process_wo * C0 * h_idx
                            self._vector_op("vadd", col2img_fp32_ub[col_index],
                                           grad_sel_ub_fp32[mask_idx],
                                           col2img_fp32_ub[col_index], "float32",
                                           remain_wo_nums * C0 // 2,
                                           stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                         self.stride_w * 16, self.stride_w * 16,
                                                         16))
                            self._vector_op("vadd", col2img_fp32_ub[col_index + 8],
                                           grad_sel_ub_fp32[mask_idx + 8],
                                           col2img_fp32_ub[col_index + 8], "float32",
                                           remain_wo_nums * C0 // 2,
                                           stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                         self.stride_w * 16, self.stride_w * 16,
                                                         16))

                col2img_fp16_ub = self.tik_instance.Tensor("float16",
                                                           col2img_ub_shape,
                                                           name="col2img_fp16_ub",
                                                           scope=tik.scope_ubuf)
                self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                           cal_shape_ele(col2img_fp32_ub.shape), "float32")

                # set h direction's paras
                start_h.set_as(ho_index * self.stride_h)
                end_h.set_as(start_h + self.stride_h)
                with self.tik_instance.if_scope(ho_index == mov_len_ho - 1):
                    end_h.set_as(start_h + self.kh)

                if offset_gm_block is not None:
                    with self.tik_instance.if_scope(start_threshold > pad_top):
                        self.tik_instance.scalar_max(hi_min, start_threshold, start_h)
                        hi_max.set_as(each_process_hi_block + start_threshold)
                    with self.tik_instance.else_scope():
                        self.tik_instance.scalar_max(hi_min, pad_top, start_h)
                        hi_max.set_as(each_process_hi_block + pad_top)
                else:
                    self.tik_instance.scalar_max(hi_min, pad_top, start_h)
                    hi_max.set_as(each_process_hi_block + pad_top)
                self.tik_instance.scalar_min(hi_max, hi_max, end_h)
                mov_len_h.set_as(hi_max - hi_min)

                # set w direction's paras
                start_w.set_as(cut_wo_nums * each_process_wo * self.stride_w)
                end_w.set_as(cut_wo_nums * each_process_wo *
                             self.stride_w + remain_wo_nums * self.stride_w)
                self.tik_instance.scalar_max(wi_min, pad_left, start_w)
                self.tik_instance.scalar_min(wi_max, self.wi + pad_left, end_w)
                mov_len_w.set_as(wi_max - wi_min)

                with self.tik_instance.if_scope(mov_len_h > 0):
                    last_valid_wi = self.wi - (
                            cut_wo_nums * each_process_wo * self.stride_w - pad_left)
                    last_valid_wi = min(last_valid_wi, self.wi)
                    if last_valid_wi <= each_process_wi:
                        self.tik_instance.data_move(
                            self.res_gm[offset_gm_inside],
                            col2img_fp16_ub[start_pos_h * each_process_wi * C0],
                            0, mov_len_h,
                            last_valid_wi * C0 // 16,
                            each_process_wi - last_valid_wi, self.wi - last_valid_wi)
                        offset_gm_inside.set_as(offset_gm_inside + last_valid_wi * C0)
                        self.offset_gm.set_as(self.offset_gm + mov_len_h * last_valid_wi * C0)
                    else:
                        self.tik_instance.data_move(self.res_gm[offset_gm_inside],
                                                    col2img_fp16_ub[start_pos_h * each_process_wi *
                                                                    C0], 0,
                                                    mov_len_h, each_process_wi * C0 // 16,
                                                    0,
                                                    self.wi - each_process_wi)
                        offset_gm_inside.set_as(offset_gm_inside + each_process_wi * C0)

                        remain_wi = last_valid_wi - each_process_wi
                        temp_zero = self.tik_instance.Tensor("float16",
                                                             (remain_wi * C0,),
                                                             name='temp_zero',
                                                             scope=tik.scope_ubuf)
                        self._vector_dup(temp_zero, 0, temp_zero.shape,
                                        self.scalar_zero_fp16, temp_zero.dtype)
                        with self.tik_instance.for_range(0, mov_len_h) as index_0:
                            self.tik_instance.data_move(
                                self.res_gm[offset_gm_inside + index_0 * self.wi * C0],
                                temp_zero, 0, 1,
                                cal_shape_ele(temp_zero.shape) // 16, 0, 0)
                        offset_gm_inside.set_as(offset_gm_inside + remain_wi * C0)
                        self.offset_gm.set_as(self.offset_gm + mov_len_h * last_valid_wi * C0)

                if self.kh > self.stride_h:
                    with self.tik_instance.for_range(0, self.kh - self.stride_h) as index_s:
                        self.tik_instance.data_move(
                            overlap_l1[index_s * overlap_l1_w + cut_wo_nums *
                                       each_process_wo * self.stride_w * C0],
                            col2img_fp32_ub[self.stride_h * each_process_wi *
                                            C0 + each_process_wi * C0 * index_s],
                            0, 1, each_process_remain_wi * C0 // 8, 0, 0)
                if self.kw <= self.stride_w:
                    self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                                    self.scalar_zero, col2img_fp32_ub.dtype)

            with self.tik_instance.if_scope(mov_len_h > 0):
                remained_hi.set_as(remained_hi - mov_len_h)

            with self.tik_instance.if_scope(tik.all(ho_index == mov_len_ho - 1, remained_hi > 0)):
                temp_zero = self.tik_instance.Tensor("float16",
                                                     (self.wi, C0),
                                                     name="temp_zero",
                                                     scope=tik.scope_ubuf)
                self._vector_dup(temp_zero, 0,
                                temp_zero.shape,
                                self.scalar_zero_fp16,
                                temp_zero.dtype)
                with self.tik_instance.for_range(0, remained_hi) as repeate_index:
                    self.tik_instance.data_move(
                        self.res_gm[self.offset_gm],
                        temp_zero, 0,
                        1, cal_shape_ele(temp_zero.shape) // 16, 0, 0)
                    self.offset_gm.set_as(self.offset_gm +
                                          cal_shape_ele(
                                              temp_zero.shape))

    def _get_core_divlist(self):
        div_list = []
        for i in range(1, CORE_NUM + 1):
            if CORE_NUM % i == 0:
                if CORE_NUM // i not in div_list:
                    div_list.append(CORE_NUM // i)
        return div_list

    def _if_block(self, ho_outer, ho_inner):
        if ho_inner <= 1:
            return False
        if self.stride_h >= self.kh:
            return True
        else:
            overlap_num = math.ceil((self.kh - self.stride_h) * 1.0 / self.stride_h)
            if self.kh > self.stride_h:
                shape_hi = (ho_inner + overlap_num - 1) * self.stride_h + self.kh
            else:
                shape_hi = ho_inner * self.stride_h
            need_cut_L1, need_cut_Ho, need_cut_Wo, \
            each_process_ho, each_process_wo = \
                self._tilling_factor((shape_hi, self.wi, C0), self.pad)

            times = math.ceil(ho_inner * 1.0 / each_process_ho)
            overlaps = overlap_num * times
            if (overlaps + ho_inner) * 1.0 / ho_inner >= ho_outer:
                return False
            return True

    def _get_block_num(self, block_num):
        if block_num > CORE_NUM:
            real_block = CORE_NUM
            block_cycle = (block_num + CORE_NUM - 1) // CORE_NUM
        else:
            real_block = block_num
            block_cycle = 1
        return real_block, block_cycle

    def tik_instance_function(self, kernel_name):
        """
        main function of tik_instance
        """
        block_num = self.n * self.c1
        real_block, block_cycle = self._get_block_num(block_num)
        if self.dtype == "float16":
            self.pad_value = MIN_VALUE_FP16
        else:
            raise RuntimeError("Only Support float16 Now!")
        pad_calc_wo, pad_calc_ho, pad_left, pad_right, pad_top, pad_bottom = \
            self._padding_mode(self.ori_input_shape, self.ksize, self.strides,
                              self.padding)
        self.pad = (int(pad_left), int(pad_right), int(pad_top), int(pad_bottom))
        self.pad_left, self.pad_right, self.pad_top, self.pad_bottom = self.pad
        if pad_calc_ho != self.ho or pad_calc_wo != self.wo:
            raise RuntimeError("Wrong ori_output shape")

        # nc do block is not enough, but ncho is enough
        # real_block == 32 or block_num * self.ho < 32
        if real_block == CORE_NUM:
            need_cut_L1, need_cut_Ho, need_cut_Wo, \
            each_process_ho, each_process_wo = \
                self._tilling_factor((self.hi, self.wi, C0), self.pad)
            with self.tik_instance.for_range(0, real_block, block_num=real_block) as block_index:
                with self.tik_instance.for_range(0, block_cycle) as cycle_index:
                    n_index = self.tik_instance.Scalar(dtype='int64', name='n_axis')
                    c1_index = self.tik_instance.Scalar(dtype='int64', name='c1_index')
                    index_sum = self.tik_instance.Scalar(dtype='int64', name='index_sum')
                    index_sum.set_as(block_index * block_cycle + cycle_index)
                    with self.tik_instance.if_scope(index_sum < block_num):
                        n_index.set_as(index_sum // self.c1)
                        c1_index.set_as(index_sum % self.c1)
                        shape = (self.ho, self.wo, self.hi, self.wi)
                        self.offset_gm.set_as(
                            (n_index * self.c1 + c1_index) * self.hi * self.wi * C0)
                        if need_cut_L1 and need_cut_Ho and need_cut_Wo:
                            self._tilling_l1_ho_wo(each_process_wo, n_index, c1_index,
                                                  self.ho, self.hi,
                                                  self.ho, self.hi,
                                                  0, 0,
                                                  0,
                                                  None, shape, self.pad)
                        elif need_cut_L1 and need_cut_Ho:
                            self._tilling_l1_ho_only(each_process_ho, n_index, c1_index,
                                                    self.ho, self.hi,
                                                    self.ho, self.hi,
                                                    0, 0,
                                                    0,
                                                    None, shape, self.pad)

                        elif need_cut_Ho:
                            self._tilling_ho_only(each_process_ho, n_index, c1_index,
                                                 self.ho, self.hi,
                                                 self.ho, self.hi,
                                                 0, 0,
                                                 0,
                                                 None, shape, self.pad)
                        else:
                            self._not_tilling(n_index, c1_index,
                                             self.ho, self.hi,
                                             self.ho, self.hi,
                                             0, 0,
                                             0, None, shape, self.pad)

        else:
            # calculate how to tiling H to block nums
            div_list = self._get_core_divlist()
            block_num_inner, block_num_outer = 0, 0
            for i in div_list:
                if block_num >= i:
                    if self.ho >= CORE_NUM // i:
                        block_num_outer = i
                        block_num_inner = (block_num + i - 1) // i
                        break
            if block_num * self.ho < CORE_NUM:
                ho_outer = self.ho
                block_num_outer = block_num
                block_num_inner = 1
            else:
                if block_num_outer == 0:
                    ho_outer = CORE_NUM // block_num
                    block_num_outer = block_num
                    block_num_inner = 1
                else:
                    ho_outer = CORE_NUM // block_num_outer
            ho_inner = int(math.ceil(self.ho * 1.0 / ho_outer))
            # ho inner > 2, do tilling Ho
            ho_outer = int(math.ceil(self.ho * 1.0 / ho_inner))

            if self._if_block(ho_outer, ho_inner):
                self.tile_h_to_block = True
                with self.tik_instance.for_range(0, block_num_outer * ho_outer,
                                                 block_num=block_num_outer *
                                                           ho_outer) as block_outer_index:
                    with self.tik_instance.for_range(0, block_num_inner) as block_innner_index:
                        nc1_index = self.tik_instance.Scalar(dtype='int64', name='nc1_index')
                        nc1_index.set_as(
                            block_outer_index // ho_outer * block_num_inner + block_innner_index)
                        with self.tik_instance.if_scope(nc1_index < block_num):
                            n_index = self.tik_instance.Scalar(dtype='int64', name='n_index')
                            c1_index = self.tik_instance.Scalar(dtype='int64', name='c1_index')
                            ho_outer_index = self.tik_instance.Scalar(dtype='int64',
                                                                      name='ho_outer_index')
                            offset_gm_block = self.tik_instance.Scalar(dtype='int64',
                                                                       name='offset_gm_block')
                            n_index.set_as(nc1_index // self.c1)
                            c1_index.set_as(nc1_index % self.c1)
                            ho_outer_index.set_as(block_outer_index % ho_outer)

                            start_hi_index = self.tik_instance.Scalar(dtype='int64',
                                                                      name='start_hi_index')
                            start_ho_index = self.tik_instance.Scalar(dtype='int64',
                                                                      name='start_ho_index')
                            actual_start_ho_index = self.tik_instance.Scalar(
                                dtype='int64', name='actual_start_ho_index')
                            actual_start_hi_index = self.tik_instance.Scalar(
                                dtype='int64', name='actual_start_hi_index')
                            each_process_ho_block = self.tik_instance.Scalar(
                                dtype='int64', name='each_process_ho_block')
                            each_process_hi_block = self.tik_instance.Scalar(
                                dtype='int64', name='each_process_hi_block')
                            pad_top_block = self.tik_instance.Scalar(dtype='int64',
                                                                     name='pad_top_block')
                            pad_bottom_block = self.tik_instance.Scalar(dtype='int64',
                                                                        name='pad_bottom_block')
                            start_threshold = self.tik_instance.Scalar(dtype='int64',
                                                                       name='start_threshold')
                            mov_len_ho = self.tik_instance.Scalar(dtype='int64', name='mov_len_ho')
                            mov_len_hi = self.tik_instance.Scalar(dtype='int64', name='mov_len_hi')

                            # each block's start ho pos and hi pos
                            # calculate the offset gm
                            start_ho_index.set_as(ho_outer_index * ho_inner)
                            start_hi_index.set_as(start_ho_index * self.stride_h)
                            actual_start_ho_index.set_as(start_ho_index)
                            actual_start_hi_index.set_as(start_hi_index)
                            start_threshold.set_as(0)

                            with self.tik_instance.if_scope(start_hi_index <= pad_top):
                                offset_gm_block.set_as(
                                    (n_index * self.c1 + c1_index) * self.hi * self.wi * C0)
                                pad_top_block.set_as(pad_top - start_hi_index)
                                self.tik_instance.scalar_max(start_threshold, start_threshold,
                                                             pad_top_block)
                                actual_start_hi_index.set_as(0)
                            with self.tik_instance.else_scope():
                                offset_gm_block.set_as(((n_index * self.c1 + c1_index) * self.hi +
                                                        start_hi_index - pad_top) * self.wi * C0)
                                pad_top_block.set_as(0)
                                actual_start_hi_index.set_as(actual_start_hi_index - pad_top)

                            with self.tik_instance.if_scope(ho_outer_index != ho_outer - 1):
                                each_process_ho_block.set_as(ho_inner)
                            with self.tik_instance.else_scope():
                                each_process_ho_block.set_as(self.ho - ho_inner * (ho_outer - 1))
                            mov_len_ho.set_as(each_process_ho_block)
                            mov_len_hi.set_as(each_process_ho_block * self.stride_h)

                            if self.stride_h < self.kh:
                                overlap = self.kh - self.stride_h
                                overlap_num = int(math.ceil(overlap * 1.0 / self.stride_h))

                                actual_start_hi_index.set_as(
                                    (start_ho_index - overlap_num) * self.stride_h)

                                with self.tik_instance.if_scope(actual_start_hi_index <= 0):
                                    actual_start_hi_index.set_as(0)
                                    actual_start_ho_index.set_as(0)
                                    pad_top_block.set_as(pad_top)
                                    mov_len_ho.set_as(start_ho_index + each_process_ho_block)
                                    start_threshold.set_as(start_ho_index * self.stride_h)
                                    self.tik_instance.scalar_max(start_threshold, start_threshold,
                                                                 pad_top_block)

                                with self.tik_instance.else_scope():
                                    pad_top_block.set_as(pad_top - actual_start_hi_index)
                                    self.tik_instance.scalar_max(pad_top_block, pad_top_block, 0)
                                    actual_start_ho_index.set_as(start_ho_index - overlap_num)
                                    with self.tik_instance.if_scope(
                                            actual_start_hi_index <= pad_top):
                                        actual_start_hi_index.set_as(0)
                                    with self.tik_instance.else_scope():
                                        actual_start_hi_index.set_as(
                                            actual_start_hi_index - pad_top)
                                    mov_len_ho.set_as(overlap_num + each_process_ho_block)
                                    start_threshold.set_as(overlap_num * self.stride_h)
                                mov_len_hi.set_as(
                                    (mov_len_ho - 1) * self.stride_h + self.kh)

                            with self.tik_instance.if_scope(start_hi_index < pad_top):
                                each_process_hi_block.set_as(
                                    each_process_ho_block * self.stride_h - (
                                            pad_top - start_hi_index))
                            with self.tik_instance.else_scope():
                                each_process_hi_block.set_as(
                                    each_process_ho_block * self.stride_h)

                            with self.tik_instance.if_scope(
                                    actual_start_ho_index + mov_len_ho > self.ho):
                                mov_len_ho.set_as(self.ho - actual_start_ho_index)

                            with self.tik_instance.if_scope(
                                    actual_start_hi_index + mov_len_hi < self.hi):
                                pad_bottom_block.set_as(0)
                            with self.tik_instance.else_scope():
                                pad_bottom_block.set_as(
                                    actual_start_hi_index + mov_len_hi - self.hi)
                                mov_len_hi.set_as(self.hi - actual_start_hi_index)

                            with self.tik_instance.if_scope(ho_outer_index == ho_outer - 1):
                                each_process_hi_block.set_as(self.hi + pad_top - start_hi_index)
                            with self.tik_instance.if_scope(
                                    start_hi_index + each_process_hi_block > self.hi + pad_top):
                                each_process_hi_block.set_as(self.hi + pad_top - start_hi_index)

                            pad = (pad_left, pad_right, pad_top_block, pad_bottom_block)
                            if self.kh > self.stride_h:
                                shape_ho = ho_inner + overlap_num
                                shape_hi = (ho_inner + overlap_num - 1) * self.stride_h + self.kh
                            else:
                                shape_ho = ho_inner
                                shape_hi = ho_inner * self.stride_h
                            if self.hi - ho_inner * self.stride_h * ho_outer > 0:
                                shape_hi += (self.hi - ho_inner * self.stride_h * ho_outer)
                            shape = (shape_ho, self.wo, shape_hi, self.wi)

                            need_cut_L1, need_cut_Ho, need_cut_Wo, \
                            each_process_ho, each_process_wo = \
                                self._tilling_factor((shape_hi, self.wi, C0), self.pad)

                            if need_cut_L1 and need_cut_Ho and need_cut_Wo:
                                self._tilling_l1_ho_wo(each_process_wo, n_index, c1_index,
                                                      each_process_ho_block, each_process_hi_block,
                                                      mov_len_ho, mov_len_hi,
                                                      actual_start_ho_index, actual_start_hi_index,
                                                      start_threshold,
                                                      offset_gm_block, shape, pad)
                            elif need_cut_L1 and need_cut_Ho:
                                self._tilling_l1_ho_only(each_process_ho, n_index, c1_index,
                                                        each_process_ho_block,
                                                        each_process_hi_block,
                                                        mov_len_ho, mov_len_hi,
                                                        actual_start_ho_index,
                                                        actual_start_hi_index,
                                                        start_threshold,
                                                        offset_gm_block, shape, pad)

                            elif need_cut_Ho:
                                self._tilling_ho_only(each_process_ho, n_index, c1_index,
                                                     each_process_ho_block, each_process_hi_block,
                                                     mov_len_ho, mov_len_hi,
                                                     actual_start_ho_index, actual_start_hi_index,
                                                     start_threshold,
                                                     offset_gm_block, shape, pad)
                            else:
                                self._not_tilling(n_index, c1_index,
                                                 each_process_ho_block, each_process_hi_block,
                                                 mov_len_ho, mov_len_hi,
                                                 actual_start_ho_index, actual_start_hi_index,
                                                 start_threshold,
                                                 offset_gm_block, shape, pad)

            else:
                nc1 = self.n * self.c1
                block = CORE_NUM
                while nc1 % block != 0:
                    block = block - 1
                nc1 = nc1 // block

                need_cut_L1, need_cut_Ho, need_cut_Wo, \
                each_process_ho, each_process_wo = \
                    self._tilling_factor((self.hi, self.wi, C0), self.pad)
                with self.tik_instance.for_range(0, block, block_num=block) as block_index:
                    with self.tik_instance.for_range(0, nc1, thread_num=1) as nc1_index:
                        self.offset_gm.set_as((block_index * nc1 + nc1_index) *
                                              self.hi * self.wi * C0)
                        n_index = (block_index * nc1 + nc1_index) // self.c1
                        c1_index = (block_index * nc1 + nc1_index) % self.c1

                        shape = (self.ho, self.wo, self.hi, self.wi)
                        if need_cut_L1 and need_cut_Ho and need_cut_Wo:
                            self._tilling_l1_ho_wo(each_process_wo, n_index, c1_index,
                                                  self.ho, self.hi,
                                                  self.ho, self.hi,
                                                  0, 0,
                                                  0,
                                                  None, shape, self.pad)
                        elif need_cut_L1 and need_cut_Ho:
                            self._tilling_l1_ho_only(each_process_ho, n_index, c1_index,
                                                    self.ho, self.hi,
                                                    self.ho, self.hi,
                                                    0, 0,
                                                    0,
                                                    None, shape, self.pad)

                        elif need_cut_Ho:
                            self._tilling_ho_only(each_process_ho, n_index, c1_index,
                                                 self.ho, self.hi,
                                                 self.ho, self.hi,
                                                 0, 0,
                                                 0,
                                                 None, shape, self.pad)
                        else:
                            self._not_tilling(n_index, c1_index,
                                             self.ho, self.hi,
                                             self.ho, self.hi,
                                             0, 0,
                                             0,
                                             None, shape, self.pad)

        self.tik_instance.BuildCCE(kernel_name=kernel_name, inputs=(
            self.ori_input_gm, self.ori_output_gm, self.grad_gm),
                                   outputs=(self.res_gm), enable_l2=False)
        return self.tik_instance
