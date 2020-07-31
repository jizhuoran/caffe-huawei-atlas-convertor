#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

Compute of depthwise conv2d.
"""
from te import tvm
import te.platform.cce_params as cce_params
from topi.cce import util

BLOCK_SIZE = cce_params.BLOCK_REDUCE

def depthwise_conv2d_native_v200_compute(fmap, weight, res_dtype, stride, pad):
    """
    algorithm: depthwise_conv2d_compute

    calculating  depthwise convolution compute

    Parameters
    ----------
    fmap : feature map placehold
        5-D shape of input tensor [N, C1, H, W, C0]

    weight : filter placehold
        4-D shape of filter tensor [k, C1, C0, KH*KW]

    res_dtype : dtype of result
        float16 float32 int32 is supported.

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    pad : Tensorflow 'SAME' model or 'VALID' model

    Returns
    -------
    depthwise_res : result tensor
       forward depthwise result of out
    """
    fmap_shape = [int(i.value) for i in fmap.shape]
    weight_shape = [int(i.value) for i in weight.shape]
    util.check_shape_rule(fmap_shape)
    util.check_shape_rule(weight_shape)
    mad_out_dtype = res_dtype

    kernel_h = 3
    kernel_w = 3
    _, stride_h, stride_w, _ = stride
    fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0 = fmap_shape
    filter_k, _, _, _ = weight_shape
    pad_top, pad_bottom, pad_left, pad_right = pad
    output_h = fmap_h + pad_top + pad_bottom - 2
    output_w = fmap_w + pad_left + pad_right - 2
    half = 2
    half_fmap_c0 = fmap_c0 // half
    pad_in_h = pad_top + pad_bottom

    fmap_pad_shape = (fmap_n, fmap_c1, fmap_h + pad_in_h, fmap_w, fmap_c0)
    fmap_pad = tvm.compute(
        fmap_pad_shape,
        lambda n, c1, h, w, c0: tvm.select(
            h < pad_top, tvm.const(0, fmap.dtype),
            tvm.select(h > fmap_h + pad_top - 1, tvm.const(0, fmap.dtype),
                       fmap[n, c1, h - pad_top, w, c0])),
        name="fmap_pad")

    fmap_split_shape = (fmap_n, fmap_c1, fmap_h + pad_in_h, fmap_w,
                        half_fmap_c0)
    fmap_low, fmap_high = tvm.compute(
        fmap_split_shape, lambda n, c1, h, w, c0:
        (fmap_pad[n, c1, h, w, c0], fmap_pad[n, c1, h, w, c0 + half_fmap_c0]),
        "split")

    fmap_new_shape = (fmap_n, fmap_c1, half, fmap_h + pad_in_h, fmap_w,
                      half_fmap_c0)
    fmap_new = tvm.compute(
        fmap_new_shape,
        lambda n, c1, c2, h, w, c0: tvm.select(
            c2 == 0, fmap_low(n, c1, h, w, c0), fmap_high(n, c1, h, w, c0)),
        name="fmap_new")
    fmap_pad_hw_shape = (fmap_n, fmap_c1, half, fmap_h + pad_in_h,
                         fmap_w + pad_left + pad_right, half_fmap_c0)
    fmap_pad_hw = tvm.compute(fmap_pad_hw_shape,
                              lambda n, c1, c2, ho, wo, c0: tvm.select(
                                  tvm.any(wo < pad_left, wo > fmap_w + pad_left
                                          - 1), tvm.const(0.0, fmap.dtype),
                                  fmap_new[n, c1, c2, ho, wo - pad_left, c0]),
                              name="fmap_pad_hw")

    k_h = tvm.reduce_axis((0, kernel_h), name='k_h')
    k_w = tvm.reduce_axis((0, kernel_w), name='k_w')

    if fmap.dtype == "float16":
        mad_shape = (fmap_n, fmap_c1, 1, filter_k, output_h, output_w, fmap_c0)
        matmul = tvm.compute(
            mad_shape,
            lambda n, c1, c2, k, ho, wo, c0: tvm.sum((fmap_pad_hw[
                n, c1, c0 // half_fmap_c0, ho * stride_h + k_h, wo * stride_w +
                k_w, c0 % half_fmap_c0] * weight[k, c1, c0, k_h * kernel_w +
                                                 k_w]).astype(mad_out_dtype),
                                                     axis=[k_h, k_w]),
            name="matmul")
        if mad_out_dtype == "float32":
            biasadd = tvm.compute(mad_shape,
                                  lambda n, c1, c2, k, ho, wo, c0: matmul[
                                      n, c1, c2, k, ho, wo, c0] +
                                  ((weight[k, c1, c0, kernel_h * kernel_w]) |
                                   (weight[k, c1, c0, kernel_h * kernel_w + 1]
                                    << 16)).astype(mad_out_dtype),
                                  name="biasadd")
        else:
            biasadd = tvm.compute(
                mad_shape,
                lambda n, c1, c2, k, ho, wo, c0: matmul[n, c1, c2, k, ho, wo,
                                                        c0] +
                (weight[k, c1, c0, kernel_h * kernel_w]).astype(mad_out_dtype),
                name="biasadd")
    else:
        mad_out_dtype = "int32"
        mad_shape = (fmap_n, fmap_c1, half, filter_k, output_h, output_w,
                     half_fmap_c0)
        matmul = tvm.compute(
            mad_shape,
            lambda n, c1, c2, k, ho, wo, c0: tvm.sum((fmap_pad_hw[
                n, c1, c2, ho * stride_h + k_h, wo * stride_w + k_w, c0
            ] * weight[k, c1, c2 * half_fmap_c0 + c0, k_h * kernel_w + k_w]).
                                                     astype(mad_out_dtype),
                                                     axis=[k_h, k_w]),
            name="matmul")
        biasadd = tvm.compute(
            mad_shape,
            lambda n, c1, c2, k, ho, wo, c0: matmul[n, c1, c2, k, ho, wo, c0] +
            ((weight[k, c1, c2 * half_fmap_c0 + c0, kernel_h * kernel_w]) |
             (weight[k, c1, c2 * half_fmap_c0 + c0, kernel_h * kernel_w + 1] <<
              8) | (weight[k, c1, c2 * half_fmap_c0 + c0, kernel_h * kernel_w +
                           2] << 16) |
             (weight[k, c1, c2 * half_fmap_c0 + c0, kernel_h * kernel_w + 3] <<
              24)).astype(mad_out_dtype),
            name="biasadd")

    mad_res = biasadd
    if mad_out_dtype not in ("float16",):
        res_dtype = biasadd.dtype
    depthwise_cast = tvm.compute(
        mad_res.shape,
        lambda *index: mad_res(*index).astype(res_dtype),
        name='depthwise_cast')
    if mad_out_dtype in ("float16",):
        res_shape = [fmap_n, fmap_c1, 1, 1, filter_k, output_h, output_w,
                     fmap_c0]
        depthwise_res = tvm.compute(
            res_shape,
            lambda n, c1, c2, c3, k, h, w, c0: depthwise_cast(
                n, c1, c2, k, h, w, c0).astype(res_dtype),
            name='depthwise_res',
            attrs={
                'kernel_h': kernel_h,
                'kernel_w': kernel_w,
                'padding': pad,
                'stride': stride
            })
    else:
        res_shape = list(depthwise_cast.shape)
        res_shape.insert(3, half)
        res_shape[-1] = res_shape[-1] // half
        depthwise_res = tvm.compute(
            res_shape,
            lambda n, c1, c2, c3, k, h, w, c0: depthwise_cast(
                n, c1, c2, k, h, w,
                (c3 * res_shape[-1]) + c0).astype(res_dtype),
            name='depthwise_res',
            attrs={
                'kernel_h': kernel_h,
                'kernel_w': kernel_w,
                'padding': pad,
                'stride': stride
            })

    return depthwise_res
