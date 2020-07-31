#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

max_pool_v200
"""
from __future__ import division
from te import tvm


class MaxPoolParam:
    """
    class of ConvParam
    """

    def __init__(self):
        pass

    tensor_map = {"is_conv_pool_fused": False}

    @staticmethod
    def update_tensormap(map_key, map_value):
        """
        update the tensor map
        """
        MaxPoolParam.tensor_map[map_key] = map_value

    @staticmethod
    def get_tensormap():
        """
        get then tensor map
        """
        return MaxPoolParam.tensor_map


def shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    tmp = []
    for i in shape:
        tmp.append(i.value)
    return tmp

NAME = "pooling2d_max_"
OP_TAG = "pooling2d_max_"
C0_SIZE = 16

@tvm.target.generic_func
def max_pool_v200(input_data, ksize, strides, pad_mode="VALID",
                  padding=None):
    """
    Performs max pooling on the input.

    Parameters
    ----------
    input_data: tensor
        tensor of input_data.
    dtype: str
        input and output data type.
    ksize: list or tuple
        A list of `ints` that has length 2
        The size of the window for H, W dimension of the input tensor
    strides: list or tuple
        A list of `ints` that has length 4.
        The stride of the sliding window of the input tensor
    padding: str
        A `string` from: "SAME", "VALID"`.The type of padding algorithm to use
    data_format: str
        default:"NC1MC0", stands for the format of output of convolution
    kernel_name: str
        kernel name, default value is 'max_pool_v200'

    Returns:
    -------
    res:
        The result of max pooling
    """

    def _compute_max_pooling(input_data, width_in, padding):
        # compute max_pooling
        input_shape = shape_to_list(input_data.shape)

        data_pad, row_max_shape, max_pool_res_shape, input_5d_data = \
            max_pooling_input(input_shape, width_in, padding)
        out_col_max, ret_out_list = \
            max_pooling(data_pad, row_max_shape, max_pool_res_shape)
        trans_line_data = tvm.compute(input_5d_data.shape,
                                      lambda *i: input_5d_data[i],
                                      name=NAME + 'trans_line_data',
                                      tag=OP_TAG + "trans_line_data")
        MaxPoolParam.update_tensormap("max_pool_tensors", ret_out_list)
        MaxPoolParam.update_tensormap("trans_line_data", trans_line_data)
        # get tiling factor
        tiling_pool_value = [max_pool_res_shape[2],
                             max_pool_res_shape[3],
                             ksize[0],
                             width_in]
        MaxPoolParam.update_tensormap("tiling_pool_value", tiling_pool_value)
        width_out = max_pool_res_shape[3]
        input_shape[1] = input_shape[3] // C0_SIZE
        input_shape[3] = C0_SIZE

        ub_reshape_shape = (input_shape[0],
                            input_shape[1],
                            max_pool_res_shape[2],
                            max_pool_res_shape[3],
                            input_shape[3])

        ub_reshape = tvm.compute(ub_reshape_shape,
                                 lambda n, c1, h, w, c0:
                                 out_col_max[n, 0,
                                             h,
                                             w,
                                             c1 * C0_SIZE + c0]
                                 + tvm.const(0.0, dtype=out_col_max.dtype),
                                 name=NAME + 'ub_reshape',
                                 tag=OP_TAG + "ub_reshape")
        MaxPoolParam.update_tensormap("ub_reshape", ub_reshape)
        trans_vn_node = tvm.compute(ub_reshape.shape,
                                    lambda n, c1, h, w, c0:
                                    ub_reshape[n, c1, h, w, c0] +
                                    trans_line_data[n, 0, 0, 0, c1 * 16 + c0],
                                    name=NAME + 'trans_vn_node',
                                    tag=OP_TAG + "trans_vn_node")
        MaxPoolParam.update_tensormap("trans_vn_node", trans_vn_node)
        res_shape = (input_shape[0],
                     input_shape[1],
                     max_pool_res_shape[2] * max_pool_res_shape[3],
                     input_shape[3])
        res = tvm.compute(res_shape,
                          lambda n, c1, m, c0:
                          trans_vn_node[n, c1,
                                      m // width_out,
                                      m % width_out,
                                      c0],
                          name=NAME + 'max_pool_res',
                          tag=OP_TAG + "max_pool_res")
        MaxPoolParam.update_tensormap("max_pool_res", res)

        return res

    def max_pooling_input(input_shape, width_in, padding):
        # get the 5HD input
        input_shape[3] = input_shape[3] * input_shape[1]
        input_shape[1] = 1
        input_5d_shape = (input_shape[0],
                          1,
                          input_shape[2] // width_in, width_in,
                          input_shape[3])
        input_5d_data = tvm.compute(input_5d_shape,
                                    lambda n, c1, h, w, c0:
                                    input_data[n,
                                               c0 // C0_SIZE,
                                               h * width_in + w,
                                               c0 % C0_SIZE] +
                                    tvm.const(0.0, dtype=input_data.dtype),
                                    name=NAME + 'input_5d_data',
                                    tag=OP_TAG + "input_5d_data")
        MaxPoolParam.update_tensormap("input_5d_data", input_5d_data)
        MaxPoolParam.update_tensormap("strides", strides)
        if pad_mode == "SAME":
            h_out, w_out, padding = \
                cal_padding_pooling(input_5d_shape,
                                    strides,
                                    ksize,
                                    dilation)

        if pad_mode == "VALID":
            h_out = (input_5d_shape[2] - ksize[0] + 1 + (strides[0] - 1)
                     + padding[0] + padding[1]) // strides[0]
            w_out = (input_5d_shape[3] - ksize[1] + 1 + (strides[1] - 1)
                     + padding[2] + padding[3]) // strides[1]

        if padding == [0, 0, 0, 0] and pad_mode == "VALID":
            data_pad = input_5d_data
            pad_shape = input_5d_shape
        else:
            data_pad, pad_shape = \
                max_pooling_padding(input_5d_data, padding, dtype)

        row_max_shape = [input_shape[0],
                         input_shape[1],
                         h_out,
                         pad_shape[3],
                         input_shape[3]]
        max_pool_res_shape = [input_shape[0],
                              input_shape[1],
                              h_out,
                              w_out,
                              input_shape[3]]

        return data_pad, row_max_shape, max_pool_res_shape, input_5d_data

    def max_pooling(data_pad, row_max_shape, max_pool_res_shape):
        # compute row max
        row_max_res, ret_row_max_pool_tensors = \
            compute_row_optimization(data_pad, row_max_shape, strides, ksize)

        # compute col max and get final res
        col_max_res, ret_col_max_pool_tensors = \
            compute_col_optimization(row_max_res,
                                     max_pool_res_shape,
                                     strides,
                                     ksize)
        # get list of vector max tensors
        ret_out_list = ret_row_max_pool_tensors + ret_col_max_pool_tensors

        return col_max_res, ret_out_list

    _check_para(ksize, strides)

    dilation = [1, 1]
    dtype = input_data.dtype

    # hard code to get conv_res attrs!!!
    conv_res = input_data.op.input_tensors[0]
    width_in = conv_res.op.attrs["width_out"].value

    res = _compute_max_pooling(input_data, width_in, padding)
    MaxPoolParam.update_tensormap("is_conv_pool_fused", False)
    return res


def _check_para(window, strides):
    """
    check the window and stride
    """
    window_h, window_w = window
    if (window_h not in range(1, 8)) or (window_w not in range(1, 8)):
        raise RuntimeError("pooling window size must be in [1,7].")
    if window_h != window_w:
        raise RuntimeError("pooling window size must be same in w and h.")

    stride_h, stride_w = strides
    if (stride_h not in range(1, 4)) or (stride_w not in range(1, 4)):
        raise RuntimeError("pooling stride size must be in [1,3].")


def compute_col_optimization(data, max_pool_res_shape, stride, window):
    """
    cal the max in cols
    """
    col_max_tensors = []
    input_shape = shape_to_list(data.shape)
    if window[1] < 4:
        col_max_tensors, out_col_max = \
            compute_window_less(data,
                                max_pool_res_shape,
                                False,
                                stride,
                                window)

    if window[1] in (4, 5, 6, 7):
        tmp_w_value = input_shape[3] - 1 \
            if stride[1] % 2 else input_shape[3] // 2
        tmp_shape = [input_shape[0],
                     input_shape[1],
                     input_shape[2],
                     tmp_w_value,
                     input_shape[4]]
        stride_tmp = 1 if stride[1] % 2 else 2
        out_col_max = compute_tmp_max(stride_tmp, tmp_shape,
                                      data, False, "col_temp_max")
        col_max_tensors.append(out_col_max)

        out_col_mid_max = compute_mid_max(max_pool_res_shape,
                                          out_col_max,
                                          False,
                                          "col_max_mid",
                                          stride)
        col_max_tensors.append(out_col_mid_max)

        if window[1] in (6, 7):
            jump_value = 4 if stride[1] % 2 else 2
            stride_value = stride[1] if stride[1] % 2 else 1
            out_col_max = \
                tvm.compute(max_pool_res_shape,
                            lambda i, j, h, w, c:
                            tvm.max(out_col_mid_max(i, j, h, w, c),
                                    out_col_max(i, j, h,
                                                w * stride_value
                                                + jump_value,
                                                c)),
                            name=NAME + "col_max_temp",
                            tag=OP_TAG + "col_max_temp")
            col_max_tensors.append(out_col_max)
        else:
            out_col_max = out_col_mid_max

        if window[1] in (5, 7):
            out_col_max = \
                tvm.compute(max_pool_res_shape,
                            lambda i2, j2, h2, w2, c2:
                            tvm.max(out_col_max(i2, j2, h2, w2, c2),
                                    data(i2, j2, h2,
                                         stride[1] * w2 + window[1] - 1,
                                         c2)),
                            name=NAME + "col_max",
                            tag=OP_TAG + "col_max")
            col_max_tensors.append(out_col_max)

    return out_col_max, col_max_tensors


def compute_row_optimization(data, row_max_shape, stride, window):
    """
    cal the max in cols
    """
    row_max_tensors = []
    input_shape = shape_to_list(data.shape)
    if window[0] < 4:
        row_max_tensors, out_row_max = \
            compute_window_less(data,
                                row_max_shape,
                                True,
                                stride,
                                window)

    if window[0] in (4, 5, 6, 7):
        tmp_h_value = input_shape[2] - 1 \
            if stride[0] % 2 else input_shape[2] // 2
        tmp_shape = [input_shape[0],
                     input_shape[1],
                     tmp_h_value,
                     input_shape[3],
                     input_shape[4]]
        stride_tmp = 1 if stride[0] % 2 else 2
        out_row_max = compute_tmp_max(stride_tmp, tmp_shape,
                                      data, True, "row_temp_max")
        row_max_tensors.append(out_row_max)

        out_row_mid_max = compute_mid_max(row_max_shape,
                                          out_row_max,
                                          True,
                                          "row_max_mid",
                                          stride)
        row_max_tensors.append(out_row_mid_max)

        if window[0] in (6, 7):
            jump_value = 4 if stride[0] % 2 else 2
            stride_value = stride[0] if stride[0] % 2 else 1
            out_row_max = \
                tvm.compute(row_max_shape, lambda i, j, h, w, c:
                            tvm.max(out_row_mid_max(i, j, h, w, c),
                                    out_row_max(i, j,
                                                h * stride_value
                                                + jump_value,
                                                w, c)),
                            name=NAME + "row_max_temp",
                            tag=OP_TAG + "row_max_temp")
            row_max_tensors.append(out_row_max)
        else:
            out_row_max = out_row_mid_max

        if window[0] in (5, 7):
            out_row_max = \
                tvm.compute(row_max_shape,
                            lambda i2, j2, h2, w2, c2:
                            tvm.max(out_row_max(i2, j2, h2, w2, c2),
                                    data(i2, j2,
                                         stride[0] * h2 + window[0] - 1,
                                         w2, c2)),
                            name=NAME + "row_max",
                            tag=OP_TAG + "row_max")
            row_max_tensors.append(out_row_max)

    return out_row_max, row_max_tensors


def max_pooling_padding(input_data, padding, dtype):
    """
    padding 0 at left, right, up, down
    """
    data_shape = shape_to_list(input_data.shape)
    padding_top, padding_bottom, padding_left, padding_right = padding

    pad_shape = [data_shape[0],
                 data_shape[1],
                 data_shape[2] + padding_top + padding_bottom,
                 data_shape[3] + padding_left + padding_right,
                 data_shape[4]]
    h_last = data_shape[2] + padding_top
    w_last = data_shape[3] + padding_left
    zero_value = tvm.const(0, dtype)
    pad_align = pad_shape
    pad_align[3] = (pad_shape[3] + C0_SIZE - 1) // C0_SIZE * C0_SIZE

    pad_data = \
        tvm.compute(
            pad_align,
            lambda n, c1, h, w, c0:
            tvm.select(
                w > padding_left - 1,
                tvm.select(w < w_last,
                           tvm.select(h > padding_top - 1,
                                      tvm.select(h < h_last,
                                                 input_data(n,
                                                            c1,
                                                            h -
                                                            padding_top,
                                                            w -
                                                            padding_left,
                                                            c0),
                                                 ),
                                      ),
                           ),
                ),
            name=NAME + "max_pooling_pad_data",
            tag=OP_TAG + "max_pooling_pad_data")

    pad_top = tvm.compute(pad_align,
                          lambda *i:
                          tvm.select(i[2] < padding_top, zero_value),
                          name="max_pooling_pad_top")
    pad_bottom = tvm.compute(pad_align,
                             lambda *i:
                             tvm.select(i[2] >= h_last, zero_value),
                             name="max_pooling_pad_bottom")
    pad_left = tvm.compute(pad_align,
                           lambda *i:
                           tvm.select(i[3] < padding_left, zero_value),
                           name="max_pooling_pad_left")
    pad_right = tvm.compute(pad_align,
                            lambda *i:
                            tvm.select(i[3] >= w_last, zero_value),
                            name="max_pooling_pad_right")
    pad_vn = tvm.compute(pad_align,
                         lambda *i:
                         pad_data[i] + pad_top[i] + pad_bottom[i] +
                         pad_left[i] + pad_right[i],
                         name="max_pooling_pad_vn")

    MaxPoolParam.update_tensormap("max_pooling_pad_data", pad_data)
    MaxPoolParam.update_tensormap("max_pooling_pad_top", pad_top)
    MaxPoolParam.update_tensormap("max_pooling_pad_bottom", pad_bottom)
    MaxPoolParam.update_tensormap("max_pooling_pad_left", pad_left)
    MaxPoolParam.update_tensormap("max_pooling_pad_right", pad_right)
    MaxPoolParam.update_tensormap("max_pooling_pad_vn", pad_vn)

    return pad_vn, pad_shape


def cal_padding_pooling(input_5d_shape, stride, window, dilation):
    """
    cal the padding , h_out and w_out when padding mode is same
    """
    h_out = (input_5d_shape[2] + stride[0] - 1) // stride[0]
    w_out = (input_5d_shape[3] + stride[1] - 1) // stride[1]

    padding_rows = (h_out - 1) * stride[0] + \
                   ((window[0] - 1) * dilation[0] + 1) - input_5d_shape[2]
    padding_cols = (w_out - 1) * stride[1] + \
                   ((window[1] - 1) * dilation[1] + 1) - input_5d_shape[3]

    if padding_rows % 2 == 0:
        padding_top = padding_rows // 2
        padding_bottom = padding_rows // 2
    else:
        padding_top = padding_rows // 2
        padding_bottom = padding_rows - padding_top

    if padding_cols % 2 == 0:
        padding_left = padding_cols // 2
        padding_right = padding_cols // 2
    else:
        padding_left = padding_cols // 2
        padding_right = padding_cols - padding_left

    if padding_top < 0:
        padding_top = 0

    if padding_bottom < 0:
        padding_bottom = 0

    if padding_left < 0:
        padding_left = 0

    if padding_right < 0:
        padding_right = 0

    padding = [padding_top, padding_bottom, padding_left, padding_right]

    return h_out, w_out, padding


def compute_window_less(data, data_shape, is_row, stride, window):
    """
    cal the max whne window is less 4
    """
    max_tensors = []
    if is_row:
        if window[0] == 1:
            out_max = tvm.compute(data_shape,
                                  lambda i, j, h, w, c:
                                  tvm.max(data(i, j, stride[0] * h, w, c),
                                          data(i, j, stride[0] * h, w, c)),
                                  name=NAME + "row_max",
                                  tag=OP_TAG + "row_max")
            max_tensors.append(out_max)

        if window[0] in (2, 3):
            out_max = compute_tmp_max(stride[0], data_shape,
                                      data, True, "row_temp_max")
            max_tensors.append(out_max)
            if window[0] == 3:
                out_max = \
                    tvm.compute(data_shape,
                                lambda i2, j2, h2, w2, c2:
                                tvm.max(out_max(i2, j2, h2, w2, c2),
                                        data(i2, j2,
                                             stride[0] * h2 + window[0] - 1,
                                             w2, c2)),
                                name=NAME + "row_max",
                                tag=OP_TAG + "row_max")
                max_tensors.append(out_max)
    else:
        if window[1] == 1:
            out_max = tvm.compute(data_shape,
                                  lambda i, j, h, w, c:
                                  tvm.max(data(i, j, h, stride[1] * w, c),
                                          data(i, j, h, stride[1] * w, c)),
                                  name=NAME + "col_max",
                                  tag=OP_TAG + "col_max")
            max_tensors.append(out_max)

        if window[1] in (2, 3):
            out_max = compute_tmp_max(stride[1], data_shape,
                                      data, False, "col_temp_max")
            max_tensors.append(out_max)
            if window[1] == 3:
                out_max = \
                    tvm.compute(data_shape,
                                lambda i2, j2, h2, w2, c2:
                                tvm.max(out_max(i2, j2, h2, w2, c2),
                                        data(i2, j2, h2,
                                             stride[1] * w2 + window[1] - 1,
                                             c2)),
                                name=NAME + "col_max",
                                tag=OP_TAG + "col_max")
                max_tensors.append(out_max)
    return max_tensors, out_max


def compute_tmp_max(stride_tmp, cal_shape, data, is_row, tmp_max_name):
    """
    compute tmp data
    """
    if is_row:
        tmp_out_max = \
            tvm.compute(cal_shape,
                        lambda i, j, h, w, c:
                        tvm.max(data(i, j, h * stride_tmp, w, c),
                                data(i, j, h * stride_tmp + 1, w, c)),
                        name=tmp_max_name,
                        tag=OP_TAG + tmp_max_name)
    else:
        tmp_out_max = \
            tvm.compute(cal_shape,
                        lambda i, j, h, w, c:
                        tvm.max(data(i, j, h, w * stride_tmp, c),
                                data(i, j, h, w * stride_tmp + 1, c)),
                        name=tmp_max_name,
                        tag=OP_TAG + tmp_max_name)

    return tmp_out_max


def compute_mid_max(cal_shape, tmp_max, is_row, mid_max_name, stride):
    """
    compute tmp data
    """
    if is_row:
        jump_mid = 2 if stride[0] % 2 else 1
        stride_mid = stride[0] if stride[0] % 2 else 1
        out_mid_max = \
            tvm.compute(cal_shape,
                        lambda i, j, h, w, c:
                        tvm.max(tmp_max(i, j,
                                        h * stride_mid,
                                        w, c),
                                tmp_max(i, j,
                                        h * stride_mid + jump_mid,
                                        w, c)),
                        name=NAME + mid_max_name,
                        tag=OP_TAG + mid_max_name)
    else:
        jump_mid = 2 if stride[1] % 2 else 1
        stride_mid = stride[1] if stride[1] % 2 else 1
        out_mid_max = \
            tvm.compute(cal_shape,
                        lambda i, j, h, w, c:
                        tvm.max(tmp_max(i, j, h, w * stride_mid, c),
                                tmp_max(i, j, h,
                                        w * stride_mid + jump_mid,
                                        c)),
                        name=NAME + mid_max_name,
                        tag=OP_TAG + mid_max_name)
    return out_mid_max
