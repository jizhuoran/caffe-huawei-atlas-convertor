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

format_transfer
"""
from __future__ import absolute_import
from topi.cce import util
from impl import four_2_five
from impl import five_2_four
from impl import zn_2_nchw
from impl import nchw_hwcn_zn
from impl import zn_2_hwcn
from impl import depthwise_weight_4d_2_6d
from impl import depthwise_weight_6d_2_4d
from impl import trans_data_2d
from impl import nz_2_nd
from impl import nd_2_nz
from impl import four_2_five_int8
from impl import five_2_four_int8
from impl import transpose_d
from impl import nd_2_zn_int8
from impl import ndhwc_2_ndc1hwc0
from impl import ndc1hwc0_2_ndhwc
from impl import nhwc_2_fractal_z_c04
from impl import nchw_2_fractal_z_c04
from impl import hwcn_2_fractal_z_c04
from impl import four_2_five_c04
from impl import dhwcn_2_fractal_z_3d
from impl import fractal_z_3d_2_dhwcn


# pylint: disable=locally-disabled,redefined-builtin,too-many-statements
def check_whether_2d(format, input_dict):
    """Check whether the 4D is 2D extend to 4D

    Parameters
    ----------
    format: str
        source data format, can be NHWC, NCHW, FRACTAL_Zn etc.
    input_dict: dict
        shape and dtype of output, should be same shape and type as input

    Returns
    -------
    is_2d : bool
        is_2d
    """
    is_2d = False
    shape = input_dict.get("shape")
    if not (len(list(format)) == len(shape) and len(shape) == 4):
        return is_2d

    dict_zip = dict(zip(list(format), shape))
    if dict_zip["H"] == 1 and dict_zip["W"] == 1 and \
            dict_zip["C"] % 16 == 0:
        is_2d = True

    return is_2d

# pylint: disable=locally-disabled,too-many-branches
@util.check_input_type(dict, dict, str, str, str)
def trans_data(src, dst, src_format, dst_format,
               kernel_name='trans_data'):
    """
    algorithm: format_transfer
    doing format_transfer for various data format
    only support NHWC/NCHW to NC1HWC0 and NC1HWC0 to NHWC/NCHW
    NCHW to FRACTAL_Zn or FRACTAL_Zn to NCHW
    HWCN to FRACTAL_Zn or FRACTAL_Zn to HWCN

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst: dict
        shape and dtype of output, should be same shape and type as input
    src_format: str
        source data format, can be NHWC, NCHW, FRACTAL_Zn etc.
    dst_format: str
        target data format, can be NC1HWC0, NCHW, FRACTAL_Zn etc.
    kernel_name: str
        kernel name, default value is "format_transfer"

    Returns
    -------
    None
    """
    if (src_format.upper() == "NHWC" or src_format.upper() == "NCHW") \
            and dst_format.upper() == "NC1HWC0":
        if check_whether_2d(src_format.upper(), src):
            trans_data_2d(src, dst, src_format, dst_format, kernel_name)
        else:
            if src.get("dtype") == "int8" or src.get("dtype") == "bool" \
                    or src.get("dtype") == "uint8":
                four_2_five_int8.four_2_five(src, dst, src_format,
                                             dst_format, kernel_name)
            else:
                four_2_five.four_2_five(src, dst, src_format,
                                        dst_format, kernel_name)
    elif src_format.upper() == "NC1HWC0" \
            and (dst_format.upper() == "NHWC" or dst_format.upper() == "NCHW"):
        if check_whether_2d(dst_format.upper(), dst):
            trans_data_2d(src, dst, src_format, dst_format, kernel_name)
        else:
            if src.get("dtype") == "int8" or src.get("dtype") == "bool" \
                    or src.get("dtype") == "uint8":
                five_2_four_int8.five_2_four(src, dst, src_format,
                                             dst_format, kernel_name)
            else:
                five_2_four.five_2_four(src, dst, src_format,
                                        dst_format, kernel_name)
    elif src_format.upper() == "NCHW" \
            and (dst_format.upper() == "FRACTAL_ZN"
                 or dst_format.upper() == "FRACTAL_Z"):
        nchw_hwcn_zn.nchw_hwcn_zn(src, dst, src_format,
                                  dst_format, kernel_name)
    elif src_format.upper() == "ND" \
            and (dst_format.upper() == "FRACTAL_ZN"
                 or dst_format.upper() == "FRACTAL_Z"):
        nd_2_zn_int8.nd_2_zn_int8(src, dst, src_format,
                                  dst_format, kernel_name)
    elif (src_format.upper() == "FRACTAL_ZN"
          or src_format.upper() == "FRACTAL_Z") \
            and dst_format.upper() == "NCHW":
        zn_2_nchw.zn_2_nchw(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "HWCN" \
            and (dst_format.upper() == "FRACTAL_ZN"
                 or dst_format.upper() == "FRACTAL_Z"
                 or dst_format.upper() == "FRACTAL_ZN_LSTM"):
        nchw_hwcn_zn.nchw_hwcn_zn(src, dst, src_format,
                                  dst_format, kernel_name)
    elif (src_format.upper() == "FRACTAL_ZN"
          or src_format.upper() == "FRACTAL_Z") \
            and dst_format.upper() == "HWCN":
        zn_2_hwcn.zn_2_hwcn(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "HWCN" \
            and dst_format.upper() == "C1HWNCOC0":
        depthwise_weight_4d_2_6d(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "C1HWNCOC0" \
            and dst_format.upper() == "HWCN":
        depthwise_weight_6d_2_4d(src, dst, src_format, dst_format, kernel_name)
    elif (src_format.upper() == "NHWC" or src_format.upper() == "NCHW"\
          or src_format.upper() == "ND") and \
            dst_format.upper() == "FRACTAL_NZ":
        nd_2_nz.nd_2_nz(src, dst, src_format, dst_format, kernel_name)
    elif (src_format.upper() == "FRACTAL_NZ" or
          src_format == "FORMAT_FRACTAL_Nz") and \
            (dst_format in ("ND", "NHWC", "NCHW")):
        nz_2_nd.nz_2_nd(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "NCHW" and dst_format.upper() == "NHWC":
        transpose_d(src, dst, [0, 2, 3, 1], kernel_name)
    elif src_format.upper() == "NCHW" and dst_format.upper() == "HWCN":
        transpose_d(src, dst, [2, 3, 1, 0], kernel_name)
    elif src_format.upper() == "NHWC" and dst_format.upper() == "NCHW":
        transpose_d(src, dst, [0, 3, 1, 2], kernel_name)
    elif src_format.upper() == "NHWC" and dst_format.upper() == "HWCN":
        transpose_d(src, dst, [1, 2, 3, 0], kernel_name)
    elif src_format.upper() == "HWCN" and dst_format.upper() == "NCHW":
        transpose_d(src, dst, [3, 2, 0, 1], kernel_name)
    elif src_format.upper() == "HWCN" and dst_format.upper() == "NHWC":
        transpose_d(src, dst, [3, 0, 1, 2], kernel_name)
    elif src_format.upper() == "CHWN" and dst_format.upper() == "NCHW":
        transpose_d(src, dst, [3, 0, 1, 2], kernel_name)
    elif src_format.upper() == "CHWN" and dst_format.upper() == "NHWC":
        transpose_d(src, dst, [3, 1, 2, 0], kernel_name)
    elif src_format.upper() == "CHWN" and dst_format.upper() == "HWCN":
        transpose_d(src, dst, [1, 2, 0, 3], kernel_name)
    elif src_format.upper() == "NDHWC" and dst_format.upper() == "NDC1HWC0":
        ndhwc_2_ndc1hwc0.ndhwc_2_ndc1hwc0(src, dst, src_format,
                                          dst_format, kernel_name)
    elif src_format.upper() == "NDC1HWC0" and dst_format.upper() == "NDHWC":
        ndc1hwc0_2_ndhwc.ndc1hwc0_2_ndhwc(src, dst, src_format,
                                          dst_format, kernel_name)
    elif src_format.upper() == "NHWC" and \
            dst_format.upper() == "FRACTAL_Z_C04":
        nhwc_2_fractal_z_c04.nhwc_2_fractal_z_c04(src, dst, src_format,
                                                  dst_format, kernel_name)
    elif src_format.upper() == "NCHW" and \
            dst_format.upper() == "FRACTAL_Z_C04":
        nchw_2_fractal_z_c04.nchw_2_fractal_z_c04(src, dst, src_format,
                                                  dst_format, kernel_name)
    elif src_format.upper() == "HWCN" and \
            dst_format.upper() == "FRACTAL_Z_C04":
        hwcn_2_fractal_z_c04.hwcn_2_fractal_z_c04(src, dst, src_format,
                                                  dst_format, kernel_name)
    elif (src_format.upper() in ["NHWC", "NCHW", "HWCN"]) and \
            dst_format.upper() == "NC1HWC0_C04":
        four_2_five_c04.four_2_five_c04(src, dst, src_format,
                                        dst_format, kernel_name)
    elif src_format.upper() == "DHWCN" and \
            dst_format.upper() == "FRACTAL_Z_3D":
        dhwcn_2_fractal_z_3d.dhwcn_2_fractal_z_3d(src, dst, src_format,
                                                  dst_format, kernel_name)
    elif src_format.upper() == "FRACTAL_Z_3D"\
            and dst_format.upper() == "DHWCN":
        fractal_z_3d_2_dhwcn.fractal_z_3d_2_dhwcn(src, dst, src_format,
                                                  dst_format, kernel_name)
    else:
        raise RuntimeError("not support this kind of format transfer !")
