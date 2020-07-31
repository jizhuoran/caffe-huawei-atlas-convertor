#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import
from te import tvm
from te.platform import intrinsic_check_support
import te.platform.cce_params as cce_params
from .elewise_compute import vadds, vmuls, vmax, vmin, vabs, vmul, vrec
from .broadcast_compute import broadcast
# pylint: disable=redefined-builtin
from .cast_compute import _cast, floor, round, round_d
from .util import check_input_tensor_shape
from .util import DTYPE_MAP

BLOCK_SIZE = cce_params.BLOCK_REDUCE
BLOCK_INT8_SIZE = cce_params.BLOCK_REDUCE_INT8


def round_to(data, max_value, min_value):
    """
    round data to [min_value,max_value]

    Parameters
    ----------
    data : tvm.tensor
        tensors need to change dtype

    max_value/min_value : float
        the range of res

    Returns
    -------
    tensor : tvm.tensor ,elements in tensor is in range [min_value,max_value]
    """
    if isinstance(data, tvm.tensor.Tensor):
        check_input_tensor_shape(data)
    data_tmp = vmuls(data, 0)
    data_min = vadds(data_tmp, min_value)
    data_max = vadds(data_tmp, max_value)
    data1 = vmax(data, data_min)
    data1 = vmin(data1, data_max)
    return data1


def cast_to_round(data, dtype):
    """
    Parameters
    ----------
    data : tvm.tensor
        tensors need to change dtype

    dtype : string
        dst dtype need to cast to

    Returns
    -------
    tensor : tvm.tensor
    """
    dtype = dtype.lower()
    if dtype != "int32":
        raise RuntimeError("The cast input type must be tvm.tensor")

    src_dtype = data.dtype.lower()
    cast_type = DTYPE_MAP[src_dtype] + "2s32a"
    is_support_round_d = intrinsic_check_support("Intrinsic_vconv", cast_type)
    if not is_support_round_d:
        return cast_to(data, dtype)

    return round_d(data)


# pylint: disable=too-many-locals, invalid-name
def cast_to(data, dtype, f1628IntegerFlag=True):
    """
    a wrapped cast operations , cast data to the type of dtype

    Parameters
    ----------
    data : tvm.tensor
        tensors need to change dtype

    dtype : string
        dst dtype need to cast to

    f1628IntegerFlag : bool
        before fp16->int8/uint8, the data is all interger or not. default value
        is False.

    Returns
    -------
    tensor : tvm.tensor
    """
    if isinstance(data, tvm.tensor.Tensor):
        data_dtype = getattr(data, 'dtype')
    else:
        raise RuntimeError("The cast input type must be tvm.tensor")
    check_input_tensor_shape(data)

    if (data_dtype.lower() == "float32") and (dtype == "int32"):
        fp32_max = tvm.const(2147483648, dtype="float32")
        fp32_min = tvm.const(2**(-31), dtype="float32")
        data1 = round_to(data, 0.5, -0.5)
        new_data = vmuls(data1, fp32_max)
        tmp2 = vabs(new_data)
        tmp3 = vadds(tmp2, fp32_min)
        fp32_res = vmul(new_data, vrec(tmp3))
        sign_res = round(fp32_res)  # get the sign data[-1,0,1]

        abs_data = vabs(data)
        floor_data = floor(abs_data)  # get the floor data
        res = vmul(floor_data, sign_res)
        return res

    if (data_dtype.lower() == "float16") and (dtype == "int32"):
        fp16_max = tvm.const(32768, dtype="float16")
        fp16_min = tvm.const(2**(-15), dtype="float16")

        data1 = round_to(data, 0.5, -0.5)

        new_data = vmuls(data1, fp16_max)
        tmp2 = vabs(new_data)
        tmp3 = vadds(tmp2, fp16_min)
        fp16_res = vmul(new_data, vrec(tmp3))
        sign_res = round(fp16_res)

        floor_data = floor(vabs(data))
        res = vmul(floor_data, sign_res)
        return res

    if (data_dtype.lower() == "float16") and (dtype in (
            'int8', 'uint8')) and not f1628IntegerFlag:
        fp16_half = tvm.const(-0.5, dtype="float16")
        data = vadds(data, fp16_half)

    if data_dtype == dtype:
        return data

    return _cast(data, dst_dtype=dtype, is_auto_cast=False)


# pylint: disable=too-many-arguments
def img2col(input_img,
            col_shape,
            filter_h,
            filter_w,
            pad,
            stride,
            tag=None,
            padding_value=0.0):
    """
    img2col
    """

    # pylint: disable=too-many-locals
    def _img2col_compute(input_img, indices, filter_w, pad, stride,
                         padding_value):
        # fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0
        _, _, fmap_h, fmap_w, _ = input_img.shape
        col_n, col_howo, col_c1, col_hw, col_ww, col_c0 = indices
        stride_h, stride_w = stride
        pad_top, _, pad_left, pad_right = pad

        output_w = (fmap_w.value + pad_left + pad_right - filter_w) \
                   // stride_w + 1

        img_n_index = col_n
        img_c1_index = col_c1
        img_h_index = (col_howo // output_w) * stride_h + col_hw
        img_w_index = (col_howo % output_w) * stride_w + col_ww
        img_c0_index = col_c0

        return tvm.select(
            tvm.any(img_h_index < pad_top,
                    img_h_index > fmap_h.value + pad_top - 1,
                    img_w_index < pad_left,
                    img_w_index > fmap_w.value + pad_left - 1),
            tvm.const(padding_value, 'float16'),
            input_img(img_n_index, img_c1_index, img_h_index - pad_top,
                      img_w_index - pad_left, img_c0_index))

    if tag is None:
        tag = 'im2col_row_major'
    return tvm.compute(
        col_shape,
        lambda *indices: _img2col_compute(input_img, indices, filter_w, pad,
                                          stride, padding_value),
        name='im2col_row_major',
        tag=tag,
        attrs={
            'kernel_h': filter_h,
            'kernel_w': filter_w,
            'padding': pad,
            'stride': stride
        })


# pylint: disable=too-many-arguments
def img2col_with_block_pad(img_tensor, filter_h, filter_w, output_h, output_w,
                           pad, stride):
    """
    img2col_with_block_pad
    """
    # img2col NC1HWC0
    img_shape = list(img_tensor.shape)
    output_hw = output_h * output_w
    img_shape = list(img_shape)
    col_shape = (img_shape[0], output_hw, img_shape[1], filter_h, filter_w,
                 img_shape[4])
    col = img2col(img_tensor, col_shape, filter_h, filter_w, pad, stride)

    output_hw_pad = (output_hw + BLOCK_SIZE - 1) \
                    // BLOCK_SIZE * BLOCK_SIZE
    col_pad_shape = (img_shape[0], output_hw_pad, img_shape[1], filter_h,
                     filter_w, img_shape[4])
    col_pad = tvm.compute(
        col_pad_shape,
        lambda n, howo_mad, c1, kh, kw, c0: tvm.select(
            howo_mad < output_hw, col(n, howo_mad, c1, kh, kw, c0),
            tvm.const(0, img_tensor.dtype)),
        name='A_im2col_after_load3d',
        tag='im2col_row_major')

    return col, col_pad


# pylint: disable=too-many-arguments
def im2col_6d(input_img,
              col_shape,
              filter_h,
              filter_w,
              pad,
              stride,
              padding_value=0.0):
    """
    im2col_6d
    """

    # pylint: disable=too-many-locals
    def _im2col_compute(input_img, indices, filter_w, pad, stride,
                        padding_value):
        # fmap_n, fmap_cg, fmap_c1, fmap_h, fmap_w, fmap_c0
        sixd_flag = 0
        if (len(input_img.shape)) == 6:
            _, _, _, fmap_h, fmap_w, _ = input_img.shape
            sixd_flag = 1
        else:
            _, _, fmap_h, fmap_w, _ = input_img.shape
        col_n, col_cg, col_howo, col_c1, col_hw, col_ww, col_c0 = indices
        stride_h, stride_w = stride
        pad_top, _, pad_left, pad_right = pad

        output_w = (fmap_w.value + pad_left + pad_right - filter_w) \
                   // stride_w + 1

        img_n_index = col_n
        img_c1_index = col_c1
        img_h_index = (col_howo // output_w) * stride_h + col_hw
        img_w_index = (col_howo % output_w) * stride_w + col_ww
        img_c0_index = col_c0

        if sixd_flag == 1:
            return tvm.select(
                tvm.any(img_h_index < pad_top,
                        img_h_index > fmap_h.value + pad_top - 1,
                        img_w_index < pad_left,
                        img_w_index > fmap_w.value + pad_left - 1),
                tvm.const(padding_value, input_img.dtype),
                input_img(img_n_index, col_cg, img_c1_index,
                          img_h_index - pad_top, img_w_index - pad_left,
                          img_c0_index))
        else:
            return tvm.select(
                tvm.any(img_h_index < pad_top,
                        img_h_index > fmap_h.value + pad_top - 1,
                        img_w_index < pad_left,
                        img_w_index > fmap_w.value + pad_left - 1),
                tvm.const(padding_value, input_img.dtype),
                input_img(img_n_index, col_cg, img_h_index - pad_top,
                          img_w_index - pad_left, img_c0_index))

    return tvm.compute(
        col_shape,
        lambda *indices: _im2col_compute(input_img, indices, filter_w, pad,
                                         stride, padding_value),
        name='im2col_row_major',
        tag='im2col_row_major',
        attrs={
            'kernel_h': filter_h,
            'kernel_w': filter_w,
            'padding': pad,
            'stride': stride
        })


def im2col_fractal(a_im2col_shape, in_a, dst='ca', tag=None):
    """
    im2col_fractal
    """
    last_dim = in_a.shape[-1]

    # pylint: disable=too-many-locals
    def __im2col_fractal_indices(indices, in_a):
        _, h_w, _, kernel_h, kernel_w, _ = in_a.shape
        if dst == 'ca':
            batch_size, i_1, j_1, i_0, j_0 = indices
        else:
            batch_size, i_1, j_1, j_0, i_0 = indices

        n_index = batch_size
        hw_index = i_1 * BLOCK_SIZE + i_0
        c1_index = (((j_1 * last_dim + j_0) // last_dim) //
                    kernel_w.value) // kernel_h.value
        kh_index = (((j_1 * last_dim + j_0) // last_dim) //
                    kernel_w.value) % kernel_h.value
        kw_index = ((j_1 * last_dim + j_0) // last_dim) % kernel_w.value
        c0_index = (j_1 * last_dim + j_0) % last_dim

        return tvm.select(
            tvm.any(hw_index < 0, hw_index > h_w.value - 1),
            tvm.const(0.0, 'float16'),
            in_a(n_index, hw_index, c1_index, kh_index, kw_index, c0_index))

    if tag is None:
        tag = 'im2col_fractal'
    return tvm.compute(
        a_im2col_shape,
        lambda *indices: __im2col_fractal_indices(indices, in_a),
        name='im2col_fractal',
        tag=tag)


def im2col_fractal_6d(a_im2col_shape, in_a):
    """
    im2col_fractal_6d
    """
    last_dim = in_a.shape[-1]

    # pylint: disable=too-many-locals
    def __im2col_fractal_indices(indices, in_a):
        _, c_g, h_w, _, kernel_h, kernel_w, _ = in_a.shape
        batch_size, c_g, i_1, j_1, i_0, j_0 = indices

        n_index = batch_size
        hw_index = i_1 * BLOCK_SIZE + i_0
        c1_index = (((j_1 * last_dim + j_0) // last_dim) //
                    kernel_w.value) // kernel_h.value
        kh_index = (((j_1 * last_dim + j_0) // last_dim) //
                    kernel_w.value) % kernel_h.value
        kw_index = ((j_1 * last_dim + j_0) // last_dim) % kernel_w.value
        c0_index = (j_1 * last_dim + j_0) % last_dim

        return tvm.select(
            tvm.any(hw_index < 0, hw_index > h_w.value - 1),
            tvm.const(0.0, in_a.dtype),
            in_a(n_index, c_g, hw_index, c1_index, kh_index, kw_index,
                 c0_index))

    return tvm.compute(
        a_im2col_shape,
        lambda *indices: __im2col_fractal_indices(indices, in_a),
        name='im2col_fractal',
        tag='im2col_fractal')


def mad(mad_shape, in_a, in_b, res_type):
    """
    mad
    """
    if res_type == "int32" or res_type == "uint32":
        r_k0 = tvm.reduce_axis((0, BLOCK_INT8_SIZE), name='k0')
    else:
        r_k0 = tvm.reduce_axis((0, BLOCK_SIZE), name='k0')
    r_k1 = tvm.reduce_axis((0, in_b.shape[1]), name='k1')
    # If tag set to 'gemv', computeOp return tensor of specific layout.
    # e.g. gemv of 1x32, tensor C is 1x32 but occupy 16x32 fractal matrix size.
    # gemv of 2x32 also occupy 16x32.
    if res_type == "float16":
        crmode = 'f162f16'
    else:
        crmode = 'f162f32'
    return tvm.compute(
        mad_shape,
        lambda n, cg, j1, i, j0: tvm.sum(in_a[
            n, cg, i // BLOCK_SIZE, r_k1, i % BLOCK_SIZE, r_k0].astype(
                res_type) * in_b[cg, r_k1, j1, j0, r_k0].astype(res_type),
                                         axis=[r_k1, r_k0]),
        name='mad',
        tag='gemm',
        attrs={'mode': crmode})


# pylint: disable=invalid-name
def tf_get_windowed_output_size(input_size, filter_size, stride, padding_type):
    """
    get output and padding size using tensorflow padding rule

    Parameters
    ----------
    input_size : int, feature map size

    filter_size : int, filter size

    stride: int, stride size

    padding_type: string, support "SAME", "VALID" or "EXPLICIT"

    Returns
    -------
    output_size: int, output feature map size

    padding_size: int, feature map padding size
    """
    if padding_type == 'EXPLICIT':
        raise RuntimeError(
            "tf_get_windowed_output_size does not handle "
            "EXPLITCIT padding; call tf_get_windowed_output_size_verbose "
            "instead.")

    # pylint: disable=invalid-name
    output_size, padding_size, _ = tf_get_windowed_output_size_verbose(
        input_size, filter_size, stride, padding_type)

    return output_size, padding_size


# pylint: disable=invalid-name
def tf_get_windowed_output_size_verbose(input_size, filter_size, stride,
                                        padding_type):
    """
    get output and padding size using tensorflow padding rule

    Parameters
    ----------
    input_size : int, feature map size

    filter_size : int, filter size

    stride: int, stride size

    padding_type: string, support "SAME", "VALID" or "EXPLICIT"

    Returns
    -------
    output_size: int, output feature map size

    padding_before: int, feature map padding before size

    padding_after: int, feature map padding after size
    """
    dilation_rate = 1

    (output_size, padding_before,
     padding_after) = tf_get_windowed_output_size_verbose_v2(
         input_size, filter_size, dilation_rate, stride, padding_type)

    return output_size, padding_before, padding_after


def tf_get_windowed_output_size_verbose_v2(input_size, filter_size,
                                           dilation_rate, stride,
                                           padding_type):
    """
    get output and padding size using tensorflow padding rule

    Parameters
    ----------
    input_size : int, feature map size

    filter_size : int, filter size

    dilation_rate: int, dilation rate

    stride: int, stride size

    padding_type: string, support "SAME", "VALID" or "EXPLICIT"

    Returns
    -------
    output_size: int, output feature map size

    padding_before: int, feature map padding before size

    padding_after: int, feature map padding after size
    """
    if stride <= 0:
        raise RuntimeError("Stride must be > 0, but got", stride)

    if dilation_rate < 1:
        raise RuntimeError("Dilation rate must be >= 1, but got",
                           dilation_rate)

    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    if padding_type == "VALID":
        output_size = (input_size - effective_filter_size + stride) // stride
        padding_before = 0
        padding_after = 0
    elif padding_type == "SAME":
        output_size = (input_size + stride - 1) // stride
        padding_needed = max(0, (output_size - 1) * stride +
                             effective_filter_size - input_size)
        padding_before = padding_needed // 2
        padding_after = padding_needed - padding_before
    else:
        raise RuntimeError("Unsupported padding type", padding_type)

    return output_size, padding_before, padding_after


def calculate_one_or_zero(input_tensor, shape, dtype):
    """
    if input_tensor>0, then output is 1, or input_tensor <=0, then output is 0

    Parameters
    ----------
    input_tensor: TVM tensor
        input_tensor tensor
    shape: list or tuple
        the shape of input_tensor
    dtype: tr
        he dtype of input_tensor

    returns
    ----------
    result: TVM tensor
        a tensor all value is 1 or 0
    """
    # define help constant. use help_min*help_rec_one*help_rec_sec to get the
    # result 1
    if dtype == "float32":
        help_min = tvm.const(2**(-126), "float32")
        help_rec_one = tvm.const(2**38, "float32")
        help_rec_sec = tvm.const(2**44, "float32")
    elif dtype == "float16":
        help_min = tvm.const(2**(-24), "float16")
        help_rec_one = tvm.const(2**12, "float16")
        help_rec_sec = help_rec_one
    elif dtype == "int32":
        help_min = tvm.const(1, "int32")
        help_rec_one = help_min
        help_rec_sec = help_min

    # broadcast constant to tensor to do vmul
    help_tensor = broadcast(help_min, shape, dtype)
    help_zero_tensor = broadcast(tvm.const(0, dtype), shape, dtype)
    help_rec_one_tensor = broadcast(help_rec_one, shape, dtype)
    help_rec_sec_tensor = broadcast(help_rec_sec, shape, dtype)

    # process to get tmp_min_y in (input_tensor, help_tensor)
    tmp_min_y = vmin(input_tensor, help_tensor)
    # process to get tmp_max_y in (help_zero_tensor, help_tensor)
    tmp_max_y = vmax(tmp_min_y, help_zero_tensor)
    result_tmp = vmul(tmp_max_y, help_rec_one_tensor)
    if dtype == "float32":
        result_tmp = vmul(result_tmp, help_rec_sec_tensor)
    result = vmul(result_tmp, help_rec_sec_tensor)

    return result
