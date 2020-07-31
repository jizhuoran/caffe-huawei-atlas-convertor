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

depthwise conv2d.
"""

from te import tvm
import te.platform.cce_params as cce_params
from te.lang.cce.te_compute import common
from topi.cce import util
import te.lang.cce
from te import tvm
from te import platform as cce
import te.lang.cce.te_compute.common as common
import te.platform.cce_params as cce_params
from topi.cce import util
from topi import generic

# cube min cell 16
BLOCK_SIZE = cce.cce_params.BLOCK_REDUCE

# shape's dim of input and output must be 4
FEATURE_MAP_DIM = 5

# shape's dim of filter must be 4
FILTER_DIM = 4

# shape's dim of strides must be 4
STRIDES_DIM = 4

# c0 value in float16 dtype
C0_16 = 16

# c0 value in uint8 or int8 dtype
C0_32 = 32


#check if all element equal
def check_element_equal(intput, element):
    for value in intput:
        if value != element:
            return False
    return True


@util.check_input_type(dict, dict, dict, dict, dict, (list, tuple),
                       (list, tuple), (list, tuple), str, int, str)
def depthwise_conv2d_native_v200(
        x,
        filter,
        bias,
        offset_w,
        out,
        strides,
        dilations=[1, 1, 1, 1],
        pads=[1, 1, 1, 1],
        data_format='NHWC',
        offset_a=0,
        kernel_name="depthwise_conv2d_native_v200",
):
    """
    algorithm: depthwise conv2d

    calculating  depthwise convolution

    Parameters
    ----------
    x : a dict of featureMap
        {"shape", "dtype"}
        shape of input tensor [N, C1, H, W, C0],
        support float16, uint8, int8.

    filter : a dict of filter
        {"shape", "format", "dtype"}
        shape of filter tensor [K, C1, C0, W],
        support float16, uint8, int8.
        K is depthwise_multiplier, only 1.

    bias : a dict of bias
           {"shape", "format", "dtype"}
           shape of input tensor [N]

    offset_w : a dict of offset
            {"shape", "format", "dtype"}

    out : a dict of output
        {"shape", "dtype"}
        shape of input tensor [N, C1, Ho, Wo, C0],
        support float16. float32 int32.

    strides : a list of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]

    pads :  a list of four ints
        pads size, [padding_top, padding_bottom, padding_left, padding_right]

    dilations : a list/tuple of four ints
        dilation size, or [1, dilation_height, dilation_width, 1]

    data_format : a str of featuremap original shape
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]

    kernel_name : str
       cce kernel name

    Returns
    -------
    None

    """
    shape_w = filter.get("shape")
    shape_in = x.get("shape")
    output_dtype = out.get("dtype")
    in_dtype = x.get("dtype")
    w_dtype = filter.get("dtype")
    fmap_data_format = x.get("format")
    util.check_kernel_name(kernel_name)
    util.check_dtype_rule(in_dtype.lower(), ('float16', "int8", "uint8"))
    util.check_dtype_rule(w_dtype.lower(), ('float16', "int8", "uint8"))
    util.check_dtype_rule(output_dtype.lower(),
                          ('float16', "float32", "int32"))
    util.check_shape_rule(shape_in, FEATURE_MAP_DIM, FEATURE_MAP_DIM)
    util.check_shape_rule(shape_w, FILTER_DIM, FILTER_DIM)
    util.check_shape_rule(strides, STRIDES_DIM, FEATURE_MAP_DIM)
    dtype_support_list = [("float16", "float16", "float16"),
                          ("float16", "float16", "float32"),
                          ("uint8", "uint8", "int32"),
                          ("uint8", "int8", "int32"),
                          ("int8", "int8", "int32")]
    if (in_dtype, w_dtype, output_dtype) not in dtype_support_list:
        raise RuntimeError("unsupport dtype combination, only support,"
                           "fp162fp16, fp162fp32, u8s8, u8u8, s8s8")

    if shape_in[1] != shape_w[1]:
        raise RuntimeError("shape_in channel and filter channel,"
                           "must be equal")

    if w_dtype.lower() in ("int8", "uint8"):
        if shape_in[-1] != C0_32 or shape_w[-1] != C0_32 or shape_w[
                -2] != C0_32:
            raise RuntimeError("shape_in[-1], shape_w[-1],"
                               "and shape_w[-2] must be 32,"
                               "when input dtype is in (int8,uint8)")
    else:
        if shape_in[-1] != C0_16 or shape_w[-1] != C0_16 or shape_w[
                -2] != C0_16:
            raise RuntimeError("shape_in[-1], shape_w[-1],"
                               "and shape_w[-2] must be 16,"
                               "when input dtype is in float16")

    if (not check_element_equal(strides, 1)):
        raise RuntimeError("depthwise_conv2d_native_v200 only support"
                           " strides is [1,1,1,1] or (1,1,1,1)")

    if (not check_element_equal(dilations, 1)):
        raise RuntimeError("depthwise_conv2d_native_v200 only support"
                           " dilations is [1,1,1,1] or (1,1,1,1)")

    if (not check_element_equal(pads, 0) and not check_element_equal(pads, 1)):
        raise RuntimeError("depthwise_conv2d_native_v200 only support"
                           " pads is [1,1,1,1] or [0,0,0,0]")

    if shape_w[0] != 1:
        raise RuntimeError("""depthwise_conv2d_native_v200 only support"""
                           """ k is 1""")

    if shape_in[2] < 3 or shape_in[3] < 3:
        raise RuntimeError("depthwise_conv2d_native_v200 valid only support"
                           " fmap h >= 3 and fmap w >= 3")

    fmap_placeholder = tvm.placeholder(shape_in,
                                       dtype=in_dtype.lower(),
                                       name='fmap')
    filter_placeholder = tvm.placeholder(shape_w,
                                         dtype=w_dtype.lower(),
                                         name='filter')
    out = te.lang.cce.te_compute.depthwise_conv2d_native_v200_compute(
        fmap_placeholder, filter_placeholder, output_dtype.lower(), strides,
        pads)
    sch = te.lang.cce.te_schedule.depthwise_conv2d_native_v200_schedule(out)

    with cce.build_config:
        tvm.build(sch, [fmap_placeholder, filter_placeholder, out],
                  "cce",
                  name=kernel_name)
