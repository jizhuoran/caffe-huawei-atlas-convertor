#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

ndc1hwc0_2_ndhwc
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from topi.cce import util
from impl.five_2_four import five_2_four


def _check_parameters(src, dst, src_format, dst_format, kernel_name):
    """
    check the parameters including src_shape, dst_shape,
    src_format, dst_format, dtype and kernel_name

    """
    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype")
    dtype_dst = dst.get("dtype")

    if src_format.lower() != "ndc1hwc0":
        raise RuntimeError("src_format must be NDC1HWC0 !")

    if dst_format.lower() != "ndhwc":
        raise RuntimeError("dst_format must be NDHWC!")

    util.check_kernel_name(kernel_name)
    check_list = ("float16",)
    util.check_dtype_rule(dtype, check_list)
    if dtype != dtype_dst:
        raise RuntimeError("dtype of src and dst are different !")

    util.check_shape_rule(src_shape, 6, 6)
    util.check_shape_rule(dst_shape, 5, 5)
    util.check_tensor_shape_size(src_shape)
    util.check_tensor_shape_size(dst_shape)

    if src_shape[5] != 16:
        raise RuntimeError(
            "the last dimension of src_shape is not 16, c0 must be 16 !")

    if src_shape[0] != dst_shape[0] or src_shape[1] != dst_shape[1]\
            or src_shape[3] != dst_shape[2] or src_shape[4] != dst_shape[3]:
        raise RuntimeError("the shape of src and dst not match, "
                           "the 1st,2nd,4th,5th dimension of src_shape and "
                           "the 1st,2nd,3rd,4th dimension of dst_shape "
                           "must be the same !")
    c_dst = dst_shape[4]

    c_1 = src_shape[2]
    c_0 = src_shape[5]
    if not ((c_dst <= c_1*c_0) and (c_dst > (c_1 - 1)*c_0)):
        raise RuntimeError("c must be less than or equal to c1*c0,"
                           "and greater than ((c1 - 1)*c0 )!")


# pylint: disable=locally-disabled, too-many-locals
@util.check_input_type(dict, dict, str, str, str)
def ndc1hwc0_2_ndhwc(src, dst, src_format, dst_format,
                     kernel_name='ndc1hwc0_2_ndhwc'):
    """
    algorithm: five_2_four
    calculating: change data format from NC1HWC0 to NCHW/NHWC

    Parameters
    ----------
    src: dict
        contains shape and dtype information of input tensor
    dst: dict
        contains shape and dtype information of output tensor
    src_format: str
        represents the format of input tensor, only support "NDC1HWC0"
    dst_format: str
        represents the format of output tensor, only support "NDHWC"
    kernel_name: str
        cce kernel name, default value is "ndc1hwc0_2_ndhwc"

    Returns
    -------
    None
    """
    _check_parameters(src, dst, src_format, dst_format, kernel_name)
    shape_ndc1hwc0 = list(src.get("shape"))
    shape_ndhwc = list(dst.get("shape"))
    n_e, d_e, c_1, h_i, w_i, c_0 = shape_ndc1hwc0
    c_i = shape_ndhwc[4]

    src_shape = [n_e*d_e, c_1, h_i, w_i, c_0]
    dst_shape = [n_e*d_e, h_i, w_i, c_i]

    src_new = src.copy()
    dst_new = dst.copy()
    src_new["shape"] = src_shape
    dst_new["shape"] = dst_shape

    five_2_four(src_new, dst_new, "NC1HWC0", "NHWC", kernel_name)
