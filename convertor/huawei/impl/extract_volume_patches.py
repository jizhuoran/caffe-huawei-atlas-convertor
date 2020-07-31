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

extract_volume_patches
"""
# pylint: disable=locally-disabled, too-many-lines
import re
import json
import os
import stat
import math
from functools import reduce as func_reduce
from te import tvm
from te import platform as tbe_platform
from te.platform import insn_cmd
from te.lang.cce.te_compute import common
from te.platform.fusion_manager import fusion_manager
from topi.cce import util

SHAPE_SIZE_LIMIT = 2 ** 30
BLOCK_SIZE = 16
BLOCK_SIZE_FP16 = 16
BLOCK_SIZE_INT8 = 32

DOUBLE_BUFFER = 2
FP16_SIZE = 2
INT8_SIZE = 1

MAX_UB_SIZE = tbe_platform.CceProductParams().getParams("Unified_Buffer")
MAX_L1_SIZE = tbe_platform.CceProductParams().getParams("L1_Buffer")

# pylint: disable=locally-disabled, too-many-arguments, invalid-name, too-many-boolean-expressions
# pylint: disable=locally-disabled, too-many-locals, too-many-statements, too-many-branches
def check_shape_and_format_vailded(input_x, output_y, ksizes, strides, padding,
                                   kernel_name):
    """
    check whether the input param valid or not
    """
    shape_x = input_x.get("ori_shape")
    shape_y = output_y.get("shape")

    if len(ksizes) == 5 and len(strides) == 5:
        _, kernel_d, kernel_h, kernel_w, _ = ksizes
        _, stride_d, stride_h, stride_w, _ = strides
    else:
        raise RuntimeError("the length of ksizes, strides must be must be 5")

    if kernel_h > 255 or kernel_h < 1 or kernel_w > 255 or kernel_w < 1:
        raise RuntimeError("Invalid kernel params, kernel size must be [1, 255].")
    if stride_h > 63 or stride_h < 1 or stride_w > 63 or stride_w < 1:
        raise RuntimeError("Invalid strides params, stride size must be [1, 63].")

    if padding not in ("SAME", "VALID"):
        raise RuntimeError("Padding must be SAME or VALID.")

    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)

    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_y, SHAPE_SIZE_LIMIT)

    dtype_x = input_x.get("dtype").lower()
    dtype_y = output_y.get("dtype").lower()

    util.check_dtype_rule(dtype_x, ("float16", "uint8", "int8"))
    util.check_dtype_rule(dtype_y, ("float16", "uint8", "int8"))
    if dtype_x != dtype_y:
        raise RuntimeError("dtype_x and dtype_y must be same.")

    util.check_kernel_name(kernel_name)
    fmap_n, fmap_d, fmap_h, fmap_w, fmap_c = shape_x

    out_d, _, _ = \
        common.tf_get_windowed_output_size_verbose(fmap_d, kernel_d, stride_d,
                                                   padding)
    out_h, _, _ = \
        common.tf_get_windowed_output_size_verbose(fmap_h, kernel_h, stride_h,
                                                   padding)
    out_w, pad_left, pad_right = \
        common.tf_get_windowed_output_size_verbose(fmap_w, kernel_w, stride_w,
                                                   padding)

    # check whether fmap_d, fmap_h and fmap_w are valid
    dilation_rate = 1
    for input_size, kernel_size, stride in ((fmap_d, kernel_d, stride_d),
                                            (fmap_h, kernel_h, stride_h),
                                            (fmap_w, kernel_w, stride_w)):
        effective_kernel_size = (kernel_size - 1) * dilation_rate + 1
        if padding == "SAME":
            output_size = (input_size + stride - 1) // stride
            padding_needed = (output_size - 1) * stride + \
                             effective_kernel_size - input_size
            if padding_needed < 0:
                raise RuntimeError("Not supported shape.")

    BLOCK_SIZE_ALIGN = BLOCK_SIZE_FP16 if dtype_x == "float16" else BLOCK_SIZE_INT8
    dtype_size = 2 if dtype_x == "float16" else 1

    if (kernel_h * fmap_w * BLOCK_SIZE_ALIGN * dtype_size) > MAX_UB_SIZE:
        raise RuntimeError("Not Supported tiling shape.")

    # cloud out_size_h = 1 or out_size_w = 1, img2col does not act normally
    if util.get_product_version() == util.VERSION_CLOUD and \
            (out_h != 1 or out_w != 1):
        if fmap_w + pad_left + pad_right - kernel_w < stride_w:
            raise RuntimeError("Invalid params in the platform of cloud, "
                               "it must be fmap_w + pad_l + pad_r - kernel_w "
                               ">= stride_w")

    expect_shape_y = (fmap_n, out_d, out_h, out_w, kernel_d*kernel_h*kernel_w*fmap_c)

    if list(expect_shape_y) != list(shape_y):
        raise RuntimeError("shape_y not support.")

    if kernel_d <= 0 or kernel_h <= 0 or kernel_w <= 0 or \
            stride_d <= 0 or stride_h <= 0 or stride_w <= 0:
        raise RuntimeError("kernel_d <= 0 or kernel_h <= 0 or kernel_w <= 0 "
                           "or stride_d <= 0 or stride_h <= 0 or stride_w <= 0")

    if len(shape_x) != 5 or len(shape_y) != 5:
        raise RuntimeError("the length of shape must be 5!")

    fmap_c1 = (fmap_c + BLOCK_SIZE_ALIGN - 1) // BLOCK_SIZE_ALIGN
    fmap_c0 = BLOCK_SIZE_ALIGN
    type_size = 2 if dtype_x == "float16" else 1
    cut_h = BLOCK_SIZE // math.gcd(out_w, BLOCK_SIZE) * stride_h + kernel_h
    if cut_h > fmap_h:
        cut_h = fmap_h
    if (cut_h * fmap_w * fmap_c1 * fmap_c0 * type_size * DOUBLE_BUFFER > MAX_L1_SIZE):
        raise RuntimeError(
            "Input size is too large load to L1, while cut h, need size: %d" %
            (cut_h * fmap_w * fmap_c1 * fmap_c0 * type_size * DOUBLE_BUFFER))

# pylint: disable=locally-disabled, too-many-arguments, invalid-name
def _ceil_to(value, ceil_value):
    """
    Return the least multiple of ceil_value integer number(output > 0)
    which is greater than or equal to x.
    """
    if ceil_value <= 0:
        return value
    return ((value + ceil_value - 1) // ceil_value) * ceil_value


# pylint: disable=locally-disabled, too-many-arguments, invalid-name
# pylint: disable=locally-disabled, too-many-locals, too-many-statements
def _din_img2col(image_patches_res_shape, A, padding, stride_d):
    """
    calculate din_img2col tensor
    Parameters
    ----------
    A : feature map

    image_patches_res_shape : shape of A_din_img2col

    padding: the padding shape

    stride_d: the stride value in d
    -------
    Returns : A_din_img2col tensor
    """

    def _din_img2col_compute(indices, A, padding, stride_d):
        """
        calculate din_img2col tensor
        Parameters
        ----------
        indices : indices in lambda function

        A : feature map

        padding: the padding shape

        stride_d: the stride value in d
        -------
        Returns : A_din_img2col tvm lambda function
        """
        _, in_d, _, _, _, _, _ = A.shape
        n, do, howo1, howo0, kd, c1, khkw, c0 = indices
        padding_top, _, _, _ = padding

        n_index = n
        d_index = do * stride_d + kd
        howo1_index = howo1
        howo0_index = howo0
        c1_index = c1
        khkw_index = khkw
        c0_index = c0

        return tvm.select(
            tvm.any(d_index < padding_top,
                    d_index > in_d.value + padding_top - 1),
            tvm.const(0, A.dtype),
            A(n_index, d_index - padding_top, howo1_index, howo0_index, c1_index, khkw_index,
              c0_index))

    return tvm.compute(
        image_patches_res_shape,
        lambda *indices: _din_img2col_compute(indices, A, padding, stride_d),
        name='image_patches_res',
        tag='image_patches_res')


# pylint: disable=locally-disabled, too-many-arguments, invalid-name
# pylint: disable=locally-disabled, too-many-locals, too-many-statements
def _img2col(input_img, col_shape, filter_h, filter_w, pad, stride):
    """
    calculate im2col tensor
    Parameters
    ----------
    input_img : feature map

    col_shape : shape of im2col tensor

    filter_h: the filter value in h

    filter_w: the filter value in w

    pad: the pad shape, list

    stride: the stride value, list

    -------
    Returns : im2col tensor
    """

    def _img2col_compute(input_img, indices, filter_w, pad, stride):
        """
        calculate im2col tensor
        Parameters
        ----------
        input_img: feature map

        indices: indices in lambda function

        filter_w: the filter value in w

        pad: the pad shape, list

        stride: the stride value, list

        -------
        Returns:  im2col tvm lambda function
        """
        _, _, _, fmap_h, fmap_w, _ = input_img.shape
        col_n, col_d, col_howo, col_c1, col_hw, col_ww, col_c0 = indices
        stride_h, stride_w = stride
        pad_top, _, pad_left, pad_right = pad

        output_w = (fmap_w.value + pad_left + pad_right - filter_w) \
                   // stride_w + 1

        img_n_index = col_n
        img_d_index = col_d
        img_c1_index = col_c1
        img_h_index = (col_howo // output_w) * stride_h + col_hw
        img_w_index = (col_howo % output_w) * stride_w + col_ww
        img_c0_index = col_c0

        return tvm.select(
            tvm.any(img_h_index < pad_top,
                    img_h_index > fmap_h.value + pad_top - 1,
                    img_w_index < pad_left,
                    img_w_index > fmap_w.value + pad_left - 1),
            tvm.const(0, input_img.dtype),
            input_img(img_n_index, img_d_index, img_c1_index, img_h_index - pad_top,
                      img_w_index - pad_left, img_c0_index))

    return tvm.compute(
        col_shape,
        lambda *indices: _img2col_compute(input_img, indices, filter_w,
                                          pad, stride),
        name='im2col_row_major',
        tag='im2col_fractal',
        attrs={
            'kernel_h': filter_h,
            'kernel_w': filter_w,
            'padding': pad,
            'stride': stride
        })


# pylint: disable=locally-disabled, too-many-arguments, invalid-name
# pylint: disable=locally-disabled, too-many-locals, too-many-statements
def _im2col_fractal(A_im2col_shape, A):
    """
    calculate im2col_fractal tensor

    Parameters
    ----------
    A_im2col_shape: shape of A_im2col

    A: feature map
    -------
    Returns : A_im2col_fractal tensor
    """

    def __im2col_fractal_indices(indices, A):
        """
        calculate im2col_fractal tvm lambda function
        Parameters
        ----------
        indices: indices in lambda function

        A: feature map
        -------
        Returns : im2col_fractal tvm lambda function
        """
        _, _, hw, _, kernel_h, kernel_w, _ = A.shape
        batch_size, d, i1, j1, j0, i0 = indices

        if A.dtype in ("int8", "uint8"):
            BLOCK_SIZE_ALIGN = BLOCK_SIZE_INT8
        else:
            BLOCK_SIZE_ALIGN = BLOCK_SIZE_FP16  # 16

        n_index = batch_size
        d_index = d
        hw_index = i1 * BLOCK_SIZE + i0
        c1_index = (((j1 * BLOCK_SIZE_ALIGN + j0) // BLOCK_SIZE_ALIGN) //
                    kernel_w.value) // kernel_h.value
        kh_index = (((j1 * BLOCK_SIZE_ALIGN + j0) // BLOCK_SIZE_ALIGN) //
                    kernel_w.value) % kernel_h.value
        kw_index = ((j1 * BLOCK_SIZE_ALIGN + j0) // BLOCK_SIZE_ALIGN) % kernel_w.value
        c0_index = (j1 * BLOCK_SIZE_ALIGN + j0) % BLOCK_SIZE_ALIGN

        return tvm.select(
            tvm.any(hw_index < 0, hw_index > hw.value - 1),
            tvm.const(0, A.dtype),
            A(n_index, d_index, hw_index, c1_index, kh_index, kw_index, c0_index))

    return tvm.compute(
        A_im2col_shape,
        lambda *indices: __im2col_fractal_indices(indices, A),
        name='im2col_fractal',
        tag='im2col_fractal')


def _tilling_axis(shape, dtype):
    """
    calculate the split parameters according to different shapes

    Parameters
    ----------
    shape : list or tuple
        shape of tensor
    dtype : string
        buffer date type

    Returns
    -------
    split_axis : the target axis that is used for spliting the tensor to find
        the maximum amount of data can be stored and processed every time on UB.
    split_factor : the factor used when spliting the target axis.
        For example, for data of float16, [1024, 1024, 256] will be split to
        [1024, 7, 164, 256], UB processes 164*256 elements every time.
        In this case, the split_axis is 1 and the split_factor is 164.
    """
    # ub_size_bytes is the size of the UB expressed by bytes(mod 8 bits).
    ub_size_bytes = MAX_UB_SIZE - 1 * 1024
    # dtype_bytes_size for float16 is 2, for float32 is 4
    dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
    # total_ele is the maximum amount of data that can be stored in UB.
    total_ele = ub_size_bytes // dtype_bytes_size

    # To initialize the split_axis and the split_factor.
    split_axis = 0
    split_factor = 1

    # To find the appropriate axis from the first one to the last
    # by comparing the amount of the elements of the split tensor with
    # the maximum amount of data that can be stored in UB.
    for index, _ in enumerate(shape):
        i = index
        ele_cnt = 1
        while i < len(shape):
            ele_cnt = ele_cnt * shape[i].astype("int64")
            i += 1
        ele_cnt = ele_cnt.value
        if ele_cnt <= total_ele:
            split_axis = index - 1
            split_factor = total_ele // ele_cnt
            break

    # when the last axis is still over the size of UB, we choose to split the
    # last axis, and the split_factor is set as the maximum amount of data
    # that can be stored in UB. For example, [10, 10, 256000] will be
    # split to [10, 10, 7, 42154]
    if shape[-1].value > total_ele:
        split_axis = len(shape) - 1
        split_factor = total_ele

    # when the amount of the elements of the tensor is less than the size of UB,
    # it means UB can process the whole tensor in one time. But the split_axis
    # has already been set to "-1", split_axis and split_factor
    # should be initialized into "0" and shape[0]
    if split_axis < 0:
        split_axis = 0
        split_factor = shape[0]

    return split_axis, split_factor

# pylint: disable=locally-disabled, too-many-arguments, invalid-name, too-many-branches
# pylint: disable=locally-disabled, too-many-locals, too-many-statements
def _get_load3d_tiling(fmap_shape, ksize, strides, padding, max_l1_valid_size,
                       max_next_valid_size, dtype):
    """
    get load3d tiling in davinci.
    ----------
        fmap_shape:
             The shape before load3d, should be (n, di, c1, hi, wi, c0).
        ksize:
             kernel sizes of load3d, should be (kernel_d, kernel_h, kernel_w).
        strides:
             strides of load3d, should be (stride_d, stride_h, stride_w).
        padding:
             "SAME" or "VALID"
        max_l1_valid_size:
            The max buffer size which can used before load3d.
        max_next_valid_size:
            The max buffer size which can used after load3d.
        dtype:
            "float16" or others.
    Returns
    -------
        is_tiling_valid:
            True or False.
        shape_in_l1:
            (n, di, c1, hi, wi, c0).
        is_l1_double_buffer:
            True or False or None.
        shape_after_load3d:
            (n, do, howo, c1, kd, khkw, c0), howo is a multiple of c0.
        is_l0_ub_double_buffer:
            True or False or None
    """
    data_size = tbe_platform.cce_intrin.get_bit_len(dtype.lower()) // 8  # 8bits = 1bytes
    max_l1_valid_num = max_l1_valid_size // data_size
    max_next_valid_num = max_next_valid_size // data_size

    fmap_n, fmap_d, fmap_c1, fmap_h, fmap_w, fmap_c0 = fmap_shape
    fmap_n, fmap_d, fmap_c1, fmap_h, fmap_w, fmap_c0 = \
        fmap_n.value, fmap_d.value, fmap_c1.value, fmap_h.value, fmap_w.value, fmap_c0.value
    kernel_d, kernel_h, kernel_w = ksize
    stride_d, stride_h, stride_w = strides
    output_d, _, _ = common.tf_get_windowed_output_size_verbose(
        fmap_d, kernel_d, stride_d, padding.upper())
    output_h, _, _ = common.tf_get_windowed_output_size_verbose(
        fmap_h, kernel_h, stride_h, padding.upper())
    output_w, _, _ = common.tf_get_windowed_output_size_verbose(
        fmap_w, kernel_w, stride_w, padding.upper())

    l1_n = 1  # init param
    l1_di = fmap_d  # init param
    l1_c1 = 1  # init param
    l1_hi = fmap_h
    l1_wi = fmap_w
    l1_c0 = fmap_c0
    max_dihiwi_l1 = max_l1_valid_num // fmap_c0
    max_dihi_l1 = max_dihiwi_l1 // fmap_w
    max_ho_l1 = (max_dihi_l1 // 1 - (kernel_h - stride_h)) // stride_h
    max_do_l1 = (max_dihi_l1 // 1 - (kernel_d - stride_d)) // stride_d

    # The memory space of l1 is not enough.
    if max_dihiwi_l1 < 1:
        return False, None, None, None, None
    if max_ho_l1 < 1 or max_do_l1 < 1:
        # not supported tiling wi in l1 now! must repeat in vertical.
        return False, None, None, None, None

    # see if we can do double buffer in l1

    l1_double_buffer = False
    min_com_multi = output_w * BLOCK_SIZE // math.gcd(output_w, BLOCK_SIZE)
    if max_ho_l1 >= min_com_multi // output_w * DOUBLE_BUFFER and \
                    max_do_l1 >= min_com_multi // output_w * DOUBLE_BUFFER:
        max_ho_l1 = max_ho_l1 // DOUBLE_BUFFER
        max_do_l1 = max_do_l1 // DOUBLE_BUFFER
        max_dihiwi_l1 = max_dihiwi_l1 // DOUBLE_BUFFER
        max_dihi_l1 = max_dihiwi_l1 // DOUBLE_BUFFER
        l1_double_buffer = True

    # l1 memory is enough to put the whole feature map.
    if max_ho_l1 >= output_h:
        # max_di_l1 = max_ho_l1 // output_h
        max_ho_l1 = output_h
        l1_hi = fmap_h

    else:  # not enough to put the whole feature map
        wo_gcd_c0 = math.gcd(output_w, BLOCK_SIZE)
        ho_gcd_c0 = BLOCK_SIZE // wo_gcd_c0
        if max_ho_l1 < ho_gcd_c0:
            return False, None, None, None, None
        max_ho_l1 = max_ho_l1 // ho_gcd_c0 * ho_gcd_c0
        l1_hi = max_ho_l1 * stride_h + kernel_h - stride_h
    l1_di = min(fmap_d, max_dihi_l1 // l1_hi) if max_dihi_l1 // l1_hi >= 1 else 1
    max_do_l1 = ((max_dihi_l1 // l1_hi) - kernel_d + 1) // stride_d
    if max_do_l1 < 1:
        max_do_l1 = 1

    howo_pad = _ceil_to(output_h * output_w, BLOCK_SIZE)
    howo_block = howo_pad // BLOCK_SIZE
    l0ub_n = 1
    l0ub_do = output_d
    l0ub_c1 = 1
    # The value of l0ub_howo must be multiplied by c0 later.
    l0ub_howo = howo_block
    l0ub_kd = kernel_d
    l0ub_khkw = kernel_h * kernel_w
    l0ub_c0 = fmap_c0
    l0_double_buffer = False

    max_dohowokdkhkw_l0ub = max_next_valid_num // fmap_c0 // fmap_c0
    # The memory space of l0/ub is not enough.
    if max_dohowokdkhkw_l0ub < 1:
        return False, None, None, None, None
    # see if we can do double buffer in l0/ub.
    if max_dohowokdkhkw_l0ub >= DOUBLE_BUFFER:
        max_dohowokdkhkw_l0ub = max_dohowokdkhkw_l0ub // DOUBLE_BUFFER
        l0_double_buffer = True

    # l0/ub memory is enough to put the whole col.
    if max_dohowokdkhkw_l0ub >= output_d * howo_block * kernel_d * kernel_h * kernel_w:
        pass
    # not enough to put whole kernel
    elif max_dohowokdkhkw_l0ub < kernel_h * kernel_w:
        l0ub_do = 1
        l0ub_howo = 1
        l0ub_khkw = max_dohowokdkhkw_l0ub
        l0ub_kd = 1
    # enough to put a whole khkw, but not enough for kd * khkw
    elif max_dohowokdkhkw_l0ub < kernel_d * kernel_h * kernel_w:
        l0ub_do = 1
        l0ub_howo = 1
        l0ub_khkw = kernel_h * kernel_w
        l0ub_kd = max_dohowokdkhkw_l0ub // (kernel_h * kernel_w)
        if l0ub_kd == 0:
            l0ub_kd = 1
    # enough to put a whole kernel, but not enough for howo
    elif max_dohowokdkhkw_l0ub < howo_block * kernel_d * kernel_h * kernel_w:
        l0ub_do = 1
        l0ub_howo = max_dohowokdkhkw_l0ub // (kernel_d * kernel_h * kernel_w)
        if l0ub_howo == 0:
            l0ub_howo = 1
    # enough to put a whole kernel and howo, but not enough for dohowo
    else:
        l0ub_do = max_dohowokdkhkw_l0ub // (howo_block * kernel_d * kernel_h * kernel_w)
        if l0ub_do == 0:
            l0ub_do = 1
    l0ub_howo *= fmap_c0  # multiplied by c0
    # get min howo in l1 and l0/ub
    l0ub_howo = min(l0ub_howo, max(max_ho_l1 * output_w, 16))
    l0ub_do = min(l0ub_do, max_do_l1)

    tile_l1_wi = fmap_w
    tile_l1_hi = (l0ub_howo + output_w - 1) // output_w * stride_h + kernel_h - 1
    tile_l1_di = max(l0ub_kd, l0ub_do)
    tile_l1_c0 = l0ub_c0
    if tile_l1_wi * tile_l1_hi * tile_l1_di * tile_l1_c0 * DOUBLE_BUFFER > max_l1_valid_num:
        l0ub_kd = min(max_l1_valid_num // (DOUBLE_BUFFER * tile_l1_wi * tile_l1_hi * tile_l1_c0),
                      kernel_d)
        if l0ub_kd == 0:
            l0ub_kd = 1
        tile_l1_di = l0ub_kd
        l0ub_do = 1
        if tile_l1_wi * tile_l1_hi * tile_l1_c0 * DOUBLE_BUFFER > max_l1_valid_num:
            l0ub_howo = BLOCK_SIZE
            tile_l1_hi = (l0ub_howo + output_w - 1) // output_w * stride_h + kernel_h - 1
    if tile_l1_wi * tile_l1_hi * tile_l1_di * tile_l1_c0 * DOUBLE_BUFFER > max_l1_valid_num:
        raise RuntimeError("L1 Size is not enough")

    return True, (l1_n, l1_di, l1_c1, l1_hi, l1_wi, l1_c0), \
           l1_double_buffer, \
           (l0ub_n, l0ub_do, l0ub_howo, l0ub_c1, l0ub_kd, l0ub_khkw, l0ub_c0), \
           l0_double_buffer

# pylint: disable=locally-disabled, too-many-arguments, unused-argument,
# pylint: disable=locally-disabled, unnecessary-lambda, too-many-locals
def _extract_volume_patches_compute_6hd(data_input, fmap_c, ksizes, strides, padding):
    """
    calculating data

    Parameters
    ----------
    data_input : TVM tensor
        the placeholder of input_x
    ksizes: input attr
    strides: input attr
    padding: input attr

    Returns
    -------
    out_res: output tensor
    workspace_res: workspace result
    """
    # fmap's format is NDC1HWC0
    fmap_shape = data_input.shape
    original_Cin = fmap_c
    fmap_in_l1 = tvm.compute(fmap_shape, lambda *i: data_input(*i), name="fmap_in_l1")

    _, filter_d, filter_h, filter_w, _ = ksizes
    _, stride_d, stride_h, stride_w, _ = strides
    fmap_batch, fmap_d, fmap_c1, fmap_h, fmap_w, fmap_c0 = fmap_shape
    out_h, padding_h_before, padding_h_after = common.tf_get_windowed_output_size_verbose(
        fmap_h.value, filter_h,
        stride_h, padding)
    out_w, padding_w_before, padding_w_after = common.tf_get_windowed_output_size_verbose(
        fmap_w.value, filter_w,
        stride_w, padding)
    pad = (padding_h_before, padding_h_after, padding_w_before, padding_w_after)
    stride = (stride_h, stride_w)
    # set_fmatrix, VM shape
    fmap_vm_shape = (fmap_batch, fmap_d, out_h * out_w, fmap_c1, filter_h, filter_w, fmap_c0)
    # fmap_in_l1: [N, D, C1 ,H, W, C0]
    fmap_im2col = _img2col(fmap_in_l1, fmap_vm_shape, filter_h, filter_w, pad, stride)
    HoWo = ((out_h * out_w + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    # load 3D, L1 to UB
    if fmap_in_l1.dtype in ("int8", "uint8"):
        BLOCK_SIZE_ALIGN = BLOCK_SIZE_INT8  # 32
    else:
        BLOCK_SIZE_ALIGN = BLOCK_SIZE_FP16  # 16

    fractal_shape = (
        fmap_batch, fmap_d, HoWo // BLOCK_SIZE, fmap_c1 * filter_h * filter_w, BLOCK_SIZE,
        BLOCK_SIZE_ALIGN)
    fmap_fractal = _im2col_fractal(fractal_shape, fmap_im2col)

    # fmap_fractal_transpose
    # transpose BLOCK_SIZE
    # [UB]fmap_fractal:
    #  (fmap_batch, fmap_d, HoWo // BLOCK_SIZE, fmap_c1 * filter_h * filter_w,
    #  BLOCK_SIZE, BLOCK_SIZE_ALIGN) ->
    # [UB]image_patches:
    #  (fmap_batch, fmap_d, HoWo // BLOCK_SIZE, BLOCK_SIZE,
    #  fmap_c1 * filter_h * filter_w, BLOCK_SIZE_ALIGN)
    image_patches_shape = (
        fmap_batch, fmap_d, HoWo // BLOCK_SIZE, BLOCK_SIZE, fmap_c1 * filter_h * filter_w,
        BLOCK_SIZE_ALIGN)
    image_patches = tvm.compute(image_patches_shape,
                                lambda n, d, howo1, howo0, c1khkw, c0: fmap_fractal[
                                    n, d, howo1, c1khkw, howo0, c0],
                                name="image_patches")

    # image_patches_split_c1
    # split c1 & khkw
    # [UB]image_patches:
    #  (fmap_batch, fmap_d, HoWo // BLOCK_SIZE, BLOCK_SIZE,
    #  fmap_c1 * filter_h * filter_w, BLOCK_SIZE_ALIGN) ->
    # [UB]image_patches_split_c1:
    #  (fmap_batch, fmap_d, HoWo // BLOCK_SIZE, BLOCK_SIZE,
    #  fmap_c1, filter_h * filter_w, BLOCK_SIZE_ALIGN)
    image_patches_split_c1_shape = (
        fmap_batch, fmap_d, HoWo // BLOCK_SIZE, BLOCK_SIZE, fmap_c1, filter_h * filter_w,
        BLOCK_SIZE_ALIGN)
    image_patches_split_c1 = tvm.compute(image_patches_split_c1_shape,
                                         lambda n, d, howo1, howo0, c1, khkw, c0: image_patches[
                                             n, d, howo1,
                                             howo0, c1 * filter_h * filter_w + khkw, c0],
                                         name="image_patches_split_c1")

    # 2nd filter and stride
    filter_h_2nd = filter_d
    stride_h_2nd = stride_d
    filter_w_2nd = 1
    stride_w_2nd = 1
    image_patches_reshape_shape = (
        fmap_batch, fmap_d, HoWo, fmap_c1 * filter_h * filter_w * BLOCK_SIZE_ALIGN)
    _, fmap_h_2nd, fmap_w_2nd, _ = image_patches_reshape_shape
    out_h_2nd, padding_h_before_2nd, padding_h_after_2nd = \
        common.tf_get_windowed_output_size_verbose(fmap_h_2nd.value,
                                                   filter_h_2nd,
                                                   stride_h_2nd,
                                                   padding)
    dout = out_h_2nd
    _, padding_w_before_2nd, padding_w_after_2nd = \
        common.tf_get_windowed_output_size_verbose(fmap_w_2nd,
                                                   filter_w_2nd,
                                                   stride_w_2nd,
                                                   padding)
    pad_2nd = (padding_h_before_2nd, padding_h_after_2nd, padding_w_before_2nd, padding_w_after_2nd)

    # image_patches_res
    # expand d axis
    # [UB]image_patches_split_c1:
    #  (fmap_batch, fmap_d, HoWo // BLOCK_SIZE, BLOCK_SIZE,
    #  fmap_c1, filter_h * filter_w, BLOCK_SIZE) ->
    # [UB]image_patches_res:
    #  (fmap_batch, dout, HoWo // BLOCK_SIZE, BLOCK_SIZE, filter_d,
    #  fmap_c1, filter_h * filter_w, BLOCK_SIZE_ALIGN)
    image_patches_res_shape = (
        fmap_batch, dout, HoWo // BLOCK_SIZE, BLOCK_SIZE, filter_d, fmap_c1, filter_h * filter_w,
        BLOCK_SIZE_ALIGN)
    image_patches_res = _din_img2col(image_patches_res_shape, image_patches_split_c1, pad_2nd,
                                     stride_d)

    # workspace_res
    # dma from ub to workspace and transpose
    # [UB]image_patches_res:
    #  (fmap_batch, dout, HoWo // BLOCK_SIZE, BLOCK_SIZE, filter_d,
    #  fmap_c1, filter_h * filter_w, BLOCK_SIZE_ALIGN) ->
    # [WorkSpace]workspace_res:
    #  (fmap_batch, dout, out_h*out_w, filter_d, filter_h * filter_w, fmap_c1, BLOCK_SIZE_ALIGN)
    workspace_shape = (
        fmap_batch, dout, out_h * out_w, filter_d, filter_h * filter_w, fmap_c1, BLOCK_SIZE_ALIGN)
    workspace_res = tvm.compute(workspace_shape,
                                lambda n, do, howo, kd, khkw, c1, c0: image_patches_res[
                                    n, do, howo // BLOCK_SIZE, howo % BLOCK_SIZE, kd, c1, khkw, c0],
                                name="workspace_res")

    # ub_res
    # dma from workspace to ub and merge c1c0
    # [WorkSpace]workspace_res:
    #  (fmap_batch, dout, out_h*out_w, filter_d, filter_h * filter_w, fmap_c1, BLOCK_SIZE_ALIGN) ->
    # [UB]ub_res:
    #  (fmap_batch, dout, out_h*out_w, filter_d, filter_h * filter_w, fmap_c1 * BLOCK_SIZE_ALIGN)
    ub_res_shape = (
        fmap_batch, dout, out_h * out_w, filter_d, filter_h * filter_w, fmap_c1 * BLOCK_SIZE_ALIGN)
    ub_res = tvm.compute(ub_res_shape, lambda n, do, howo, kd, khkw, c1c0: workspace_res[
        n, do, howo, kd, khkw, c1c0 // BLOCK_SIZE_ALIGN, c1c0 % BLOCK_SIZE_ALIGN],
                         name="ub_res")

    # out_res
    # remove redundant c', and split kh and kw, from ub to gm
    # [UB]ub_res: (fmap_batch, dout, out_h*out_w, filter_d,
    #  filter_h * filter_w, fmap_c1 * BLOCK_SIZE) ->
    # [GM]out_res: (fmap_batch, dout, out_h*out_w, filter_d,
    #  filter_h * filter_w, original_Cin)
    out_res_shape = (fmap_batch, dout, out_h * out_w, filter_d, filter_h * filter_w, original_Cin)

    extract_params = {}
    extract_params["padding_mode"] = padding
    extract_params["original_Cin"] = original_Cin
    extract_params["out_d"] = out_h_2nd
    extract_params["out_h"] = out_h
    extract_params["out_w"] = out_w
    extract_params["fmap_shape"] = fmap_shape
    extract_params["ksizes"] = (filter_d, filter_h, filter_w)
    extract_params["strides"] = (stride_d, stride_h, stride_w)
    extract_params["pad"] = pad
    extract_params["fmap_vm_shape"] = fmap_vm_shape
    extract_params["fractal_shape"] = fractal_shape
    extract_params["HoWo"] = HoWo
    extract_params["ub_res_shape"] = ub_res_shape

    setfmatrix_dict = {"conv_kernel_h": filter_h,
                       "conv_kernel_w": filter_w,
                       "conv_padding_top": padding_h_before,
                       "conv_padding_left": padding_w_before,
                       "conv_padding_right": padding_w_after,
                       "conv_stride_h": stride_h,
                       "conv_stride_w": stride_w,
                       "conv_fm_c": fmap_c1 * fmap_c0,
                       "conv_fm_h": fmap_h,
                       "conv_fm_w": fmap_w}
    out_res = tvm.compute(out_res_shape, lambda *i: ub_res[i], name="out_res",
                          attrs={'extract_params': extract_params,
                                 'setfmatrix_dict': setfmatrix_dict})

    return out_res, workspace_res


# pylint: disable=locally-disabled, too-many-arguments, unused-argument,
# pylint: disable=locally-disabled, unnecessary-lambda, too-many-locals
@fusion_manager.register("extract_volume_patches")
def extract_volume_patches_compute(data_input, fmap_c, ksizes, strides, padding):
    """
    ops compute

    Parameters
    ----------
    data_input:  TVM tensor
        the placeholder of input_x
    ksizes: input attr
    strides: input attr
    padding: input attr
    Returns
    -------
    compute results
    """
    output_image_patches = _extract_volume_patches_compute_6hd(data_input,
                                                               fmap_c, ksizes, strides, padding)

    return output_image_patches


# pylint: disable=locally-disabled, too-many-statements, too-many-locals
def extract_volume_patches_schedule(res, sch_list):
    """
    extract_image_patches schedule

    Parameters
    ----------
    res: the multi-results in the operator
    sch_list: schedule list

    Returns
    -------
    None
    """
    sch = sch_list[0]

    ub_res = res.op.input_tensors[0]
    workspace_res = ub_res.op.input_tensors[0]
    image_patches_res = workspace_res.op.input_tensors[0]
    image_patches_split_c1 = image_patches_res.op.input_tensors[0]
    image_patches = image_patches_split_c1.op.input_tensors[0]
    fmap_fractal = image_patches.op.input_tensors[0]
    fmap_im2col = fmap_fractal.op.input_tensors[0]
    fmap_in_l1 = fmap_im2col.op.input_tensors[0]

    setfmatrix_map = res.op.attrs['setfmatrix_dict']
    setfmatrix_dict = {}
    for key, value in setfmatrix_map.items():
        if hasattr(value, "value"):
            setfmatrix_dict[key] = value.value
        else:
            setfmatrix_dict[key] = value

    extract_map = res.op.attrs['extract_params']
    extract_params = {}
    for key, value in extract_map.items():
        if hasattr(value, "value"):
            extract_params[key] = value.value
        else:
            extract_params[key] = value

    fmap_shape = extract_params["fmap_shape"]
    (filter_d, filter_h, filter_w) = extract_params["ksizes"]
    (filter_d, filter_h, filter_w) = (filter_d.value, filter_h.value, filter_w.value)
    (stride_d, stride_h, stride_w) = extract_params["strides"]
    (stride_d, stride_h, stride_w) = (stride_d.value, stride_h.value, stride_w.value)
    (fmap_batch, fmap_d, fmap_c1, fmap_h, fmap_w, _) = fmap_shape
    fmap_batch, fmap_d, fmap_c1, fmap_h, fmap_w = \
        fmap_batch.value, fmap_d.value, fmap_c1.value, fmap_h.value, fmap_w.value
    padding = extract_params["padding_mode"]
    original_Cin = extract_params["original_Cin"]
    out_d = extract_params['out_d']
    out_w = extract_params['out_w']
    HoWo = extract_params["HoWo"]
    ub_res_shape = extract_params["ub_res_shape"]

    sch[fmap_in_l1].set_scope(tbe_platform.scope_cbuf)
    sch[fmap_im2col].set_scope(tbe_platform.scope_cbuf)
    sch[fmap_fractal].set_scope(tbe_platform.scope_ubuf)
    sch[image_patches_res].set_scope(tbe_platform.scope_ubuf)
    sch[workspace_res].set_scope(tbe_platform.scope_gm)
    sch[ub_res].set_scope(tbe_platform.scope_ubuf)

    # compute inline
    sch[image_patches].compute_inline()
    sch[image_patches_split_c1].compute_inline()

    # align to 32B
    dtype_input = ub_res.dtype
    if dtype_input in ("int8" ,"uint8"):
        BLOCK_SIZE_ALIGN = BLOCK_SIZE_INT8
    else:
        BLOCK_SIZE_ALIGN = BLOCK_SIZE  # 16

    # c1 * BlockSize must be integer multiple of 16B
    sch[fmap_im2col].buffer_align((1, 1), (1, 1), (out_w, out_w), (1, 1), (1, 1), (1, 1),
                                  (1, BLOCK_SIZE_ALIGN))
    sch[fmap_fractal].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, BLOCK_SIZE),
                                   (1, BLOCK_SIZE_ALIGN))
    sch[image_patches_res].buffer_align((1, 1), (1, 1), (1, 1), (1, BLOCK_SIZE), (1, 1), (1, 1),
                                        (1, 1), (1, BLOCK_SIZE_ALIGN))
    sch[ub_res].storage_align(ub_res.op.axis[4], BLOCK_SIZE_ALIGN, 0)

    # compute tiling params
    fmap_fractal_size = \
        fmap_batch * fmap_d * HoWo * fmap_c1 * filter_h * filter_w * BLOCK_SIZE_ALIGN
    image_patches_res_size = \
        fmap_batch * out_d * HoWo * filter_d * fmap_c1 * filter_h * filter_w * BLOCK_SIZE_ALIGN

    total_size = fmap_fractal_size + image_patches_res_size

    max_next_valid_size = MAX_UB_SIZE * image_patches_res_size // total_size

    max_l1_valid_size = tbe_platform.CceProductParams().getParams("L1_Buffer")

    is_tiling_valid, shape_in_l1, is_l1_double_buffer, \
    shape_after_load3d, is_l0_ub_double_buffer = \
        _get_load3d_tiling(fmap_shape, (filter_d, filter_h, filter_w),
                           (stride_d, stride_h, stride_w), padding, max_l1_valid_size,
                           max_next_valid_size, dtype_input)

    if (is_tiling_valid, shape_in_l1, is_l1_double_buffer, shape_after_load3d,
        is_l0_ub_double_buffer) == (False, None, None, None, None):
        raise RuntimeError(
            "Not supported fmap shape = (%u, %u, %u, %u, %u, %u),"
            " kernel = (1, %u, %u, %u, 1),"
            " stride = (1, %u, %u, %u, 1)" %
            (fmap_shape[0], fmap_shape[1], fmap_shape[2], fmap_shape[3],
             fmap_shape[4], fmap_shape[5], filter_d, filter_h, filter_w, stride_d, stride_h,
             stride_w))

    (_, ub_do, ub_howo, _, ub_kd, ub_khkw, _) = shape_after_load3d

    # for load3d emit_insn
    _, fmap_im2col_d_inner = sch[fmap_im2col].split(fmap_im2col.op.axis[1], factor=1)
    _, fmap_fractal_d_inner = sch[fmap_fractal].split(fmap_fractal.op.axis[1], factor=1)
    if original_Cin % BLOCK_SIZE_ALIGN != 0:
        # cut workspace_res
        workspace_res_n_outer, workspace_res_n_inner = sch[workspace_res].split(
            workspace_res.op.axis[0], factor=1)
        workspace_res_do_outer, workspace_res_do_inner = sch[workspace_res].split(
            workspace_res.op.axis[1],
            factor=ub_do)
        workspace_res_howo_outer, workspace_res_howo_inner = sch[workspace_res].split(
            workspace_res.op.axis[2],
            factor=ub_howo)
        workspace_res_kd_outer, workspace_res_kd_inner = sch[workspace_res].split(
            workspace_res.op.axis[3],
            factor=ub_kd)
        workspace_res_khkw_outer, workspace_res_khkw_inner = sch[workspace_res].split(
            workspace_res.op.axis[4],
            factor=ub_khkw)
        workspace_res_c1_outer, workspace_res_c1_inner = sch[workspace_res].split(
            workspace_res.op.axis[5], factor=1)

        sch[workspace_res].reorder(workspace_res_n_outer, workspace_res_do_outer,
                                   workspace_res_c1_outer,
                                   workspace_res_howo_outer, workspace_res_kd_outer,
                                   workspace_res_khkw_outer,
                                   workspace_res_n_inner, workspace_res_do_inner,
                                   workspace_res_c1_inner,
                                   workspace_res_howo_inner, workspace_res_kd_inner,
                                   workspace_res_khkw_inner,
                                   workspace_res.op.axis[6])

        split_axis, split_factor = _tilling_axis(ub_res_shape[1:], dtype_input)

        # cut res
        res_n_outer, res_n_inner = sch[res].split(res.op.axis[0], factor=1)
        res_axis_outer, res_axis_inner = sch[res].split(res.op.axis[split_axis+1],
                                                        factor=split_factor)


        # for compute_at
        sch[fmap_in_l1].compute_at(sch[workspace_res], workspace_res_kd_outer)
        sch[fmap_im2col].compute_at(sch[workspace_res], workspace_res_kd_outer)

        sch[fmap_fractal].compute_at(sch[workspace_res], workspace_res_khkw_outer)
        sch[image_patches_res].compute_at(sch[workspace_res], workspace_res_khkw_outer)

        sch[ub_res].compute_at(sch[res], res_axis_outer)

        # for emit_insn
        sch[fmap_in_l1].emit_insn(fmap_in_l1.op.axis[0], insn_cmd.DMA_COPY)
        sch[fmap_im2col].emit_insn(fmap_im2col_d_inner, 'set_fmatrix', setfmatrix_dict)
        sch[fmap_fractal].emit_insn(fmap_fractal_d_inner, insn_cmd.IM2COL)

        sch[image_patches_res].emit_insn(image_patches_res.op.axis[0], insn_cmd.DMA_COPY)
        sch[workspace_res].emit_insn(workspace_res_n_inner, insn_cmd.DMA_COPY)
        sch[ub_res].emit_insn(ub_res.op.axis[0], insn_cmd.DMA_COPY)
        sch[res].emit_insn(res_axis_inner, insn_cmd.DMA_COPY)

        # for double buffer
        if is_l0_ub_double_buffer:
            sch[fmap_fractal].double_buffer()
            sch[image_patches_res].double_buffer()
            sch[fmap_im2col].double_buffer()
        if is_l1_double_buffer:
            sch[fmap_in_l1].double_buffer()

        # for multi cores
        res_fused_axis = sch[res].fuse(res_n_outer)
        workspace_fused_axis = sch[workspace_res].fuse(workspace_res_n_outer)
        block_idx = tvm.thread_axis('blockIdx.x')
        real_multi_num = fmap_shape[0].value
        device_core_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        if real_multi_num >= device_core_num:
            workspace_res_fused_axis_outer, _ = sch[workspace_res].split(workspace_fused_axis,
                                                                         nparts=device_core_num)
            res_fused_axis_outer, _ = sch[res].split(res_fused_axis, nparts=device_core_num)
            sch[workspace_res].bind(workspace_res_fused_axis_outer, block_idx)
            sch[res].bind(res_fused_axis_outer, block_idx)
        else:
            sch[workspace_res].bind(workspace_fused_axis, block_idx)
            sch[res].bind(res_fused_axis, block_idx)

    else:
        # cut workspace_res
        sch[workspace_res].compute_inline()
        sch[ub_res].compute_inline()

        # cut res
        res_n_outer, res_n_inner = sch[res].split(res.op.axis[0], factor=1)
        res_do_outer, res_do_inner = sch[res].split(res.op.axis[1], factor=ub_do)
        res_howo_outer, res_howo_inner = sch[res].split(res.op.axis[2], factor=ub_howo)
        res_kd_outer, res_kd_inner = sch[res].split(res.op.axis[3], factor=ub_kd)
        res_khkw_outer, res_khkw_inner = sch[res].split(res.op.axis[4], factor=ub_khkw)
        res_c_outer, res_c_inner = sch[res].split(res.op.axis[5], factor=BLOCK_SIZE_ALIGN)

        sch[res].reorder(res_n_outer, res_do_outer, res_c_outer,
                         res_howo_outer, res_kd_outer, res_khkw_outer,
                         res_n_inner, res_do_inner, res_c_inner,
                         res_howo_inner, res_kd_inner, res_khkw_inner)

        # for compute_at
        sch[fmap_in_l1].compute_at(sch[res], res_khkw_outer)
        sch[fmap_im2col].compute_at(sch[res], res_khkw_outer)

        sch[fmap_fractal].compute_at(sch[res], res_khkw_outer)
        sch[image_patches_res].compute_at(sch[res], res_khkw_outer)


        # for emit_insn
        sch[fmap_in_l1].emit_insn(fmap_in_l1.op.axis[0], insn_cmd.DMA_COPY)
        sch[fmap_im2col].emit_insn(fmap_im2col_d_inner, 'set_fmatrix', setfmatrix_dict)
        sch[fmap_fractal].emit_insn(fmap_fractal_d_inner, insn_cmd.IM2COL)
        sch[image_patches_res].emit_insn(image_patches_res.op.axis[0], insn_cmd.DMA_COPY)
        sch[res].emit_insn(res_n_inner, insn_cmd.DMA_COPY)

        # for double buffer
        if is_l0_ub_double_buffer:
            sch[fmap_fractal].double_buffer()
            sch[image_patches_res].double_buffer()
            sch[fmap_im2col].double_buffer()
        if is_l1_double_buffer:
            sch[fmap_in_l1].double_buffer()

        # for multi cores
        res_shape = [int(i.value) for i in res.shape]
        res_n, res_do, _, _, _, _ = res_shape
        real_multi_num = res_n
        device_core_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        res_fused_axis = sch[res].fuse(res_n_outer)

        if real_multi_num >= device_core_num:
            res_fused_axis_outer, _ = sch[res].split(res_fused_axis, nparts=device_core_num)
            block_idx = tvm.thread_axis('blockIdx.x')
            sch[res].bind(res_fused_axis_outer, block_idx)


# pylint: disable=too-many-arguments, too-many-locals
@util.check_input_type(dict, dict, (list, tuple), (list, tuple), str, str)
def extract_volume_patches(input_x, output_y, ksizes, strides, padding,
                           kernel_name="extract_volume_patches"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same type as input
    ksizes: input attr
    strides: input attr
    padding: input attr
    kernel_name : str
        kernel name, default value is "extract_volume_patches"

    Returns
    -------
    None
    """
    check_shape_and_format_vailded(input_x, output_y, ksizes, strides, padding,
                                   kernel_name)
    ori_shape_5d = input_x.get("ori_shape")
    fmap_n = ori_shape_5d[0]
    fmap_d = ori_shape_5d[1]
    fmap_h = ori_shape_5d[2]
    fmap_w = ori_shape_5d[3]
    fmap_c = ori_shape_5d[4]
    dtype = input_x.get("dtype")
    input_dtype = dtype.lower()
    BLOCK_SIZE_ALIGN = BLOCK_SIZE_FP16 if input_dtype == "float16" else BLOCK_SIZE_INT8
    shape = (fmap_n, fmap_d, (fmap_c + BLOCK_SIZE_ALIGN - 1) // BLOCK_SIZE_ALIGN,
             fmap_h, fmap_w, BLOCK_SIZE_ALIGN)

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
    res, workspace_res = extract_volume_patches_compute(data_input,
                                                        fmap_c, ksizes, strides, padding)

    sch = tvm.create_schedule(res.op)

    extract_volume_patches_schedule(res, [sch])

    def _write_workspace_info(workspace_list, kernel_name):
        """
        write workspace information
        """
        def write_code(wkspace_dict, fname):
            """
            write information into json
            """
            fname = os.path.realpath(fname)
            if fname.startswith(os.getcwd()):
                if os.path.exists(fname):
                    with open(fname, "r") as f:
                        load_dict = json.load(f)
                    load_dict.update(wkspace_dict)
                    with open(fname, "w") as f:
                        json.dump(load_dict,
                                  f,
                                  sort_keys=True,
                                  indent=4,
                                  separators=(',', ':'))

        def shape_to_list(shape):
            """
            translate tvm.shape to list type in python
            """
            tmp = []
            for i in shape:
                tmp.append(i.value)
            return tmp

        def get_data_width(dtype):
            """
            get data_width
            """
            m = re.search(r'\d+', dtype)
            if m:
                return int(m.group(0)) // 8
            return 0

        num = len(workspace_list)
        if num:
            shape_list = [shape_to_list(i.shape) for i in workspace_list]
            total_size = [
                func_reduce(lambda x, y: x * y, list_i)
                for list_i in shape_list
            ]

            total_size = [i * get_data_width(j.dtype)
                          for i, j in zip(total_size, workspace_list)]
            if not os.path.exists("kernel_meta"):
                os.mkdir("kernel_meta")
                os.chmod("kernel_meta",
                         stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP)
            wkspace_dict = {"workspace": {"num": num, "size": total_size}}
            write_code(wkspace_dict, "kernel_meta/" + kernel_name + ".json")

    with tbe_platform.build_config:
        tvm.build(sch, [data_input, res, workspace_res], "cce", name=kernel_name)
        _write_workspace_info([workspace_res], kernel_name)
