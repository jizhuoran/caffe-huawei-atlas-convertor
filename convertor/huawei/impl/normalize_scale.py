"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

normalize_scale
"""

import te.lang.cce
from te import tvm, platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


# pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
# pylint: disable=locally-disabled,too-many-arguments,protected-access
# pylint: disable=locally-disabled,too-many-branches
@fusion_manager.register("normalize_scale")
def normalize_scale_compute(x1, x2, x3, y,
                            across_spatial=True, eps=1e-10,
                            kernel_name="normalize_scale"):
    """
    calculating data

    Parameters
    ----------
    x1 : TVM tensor
        the placeholder of x1
    x2 : TVM tensor
        the placeholder of x2
    x3 : TVM tensor
        the placeholder of x3
    y : dict
        dict of y, include keys(shape and dtype, format)
    across_spatial: bool
        indicates whether standardization should cross spatial locations.
        Default(True)
    eps: float
        prevent dividing by 0.
        Default(1e-10)
    kernel_name : str
        kernel name, default value is "normalize_scale"

    Returns
    -------
    output tensor
    """

    # set intermediate dtype
    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
        # hisi es, cs
        intermediate_dtype = "float16"
        dtype_cast_mapping = {"int8": "float16"}
        dtype_reverse_cast_mapping = {"float16": "int8"}
    else:
        # mini, cloud
        intermediate_dtype = "float32"
        dtype_cast_mapping = {"int8": "float16", "float16": "float32"}
        dtype_reverse_cast_mapping = {"float16": "int8",
                                      "float32": "float16"}

    x1_shape = te.lang.cce.util.shape_to_list(x1.shape)

    x1_cast = x1
    while x1_cast.dtype in dtype_cast_mapping:
        x1_cast = te.lang.cce.cast_to(x1_cast,
                                      dtype_cast_mapping[x1_cast.dtype])
    x2_cast = x2
    while x2_cast.dtype in dtype_cast_mapping:
        x2_cast = te.lang.cce.cast_to(x2_cast,
                                      dtype_cast_mapping[x2_cast.dtype])

    x3_cast = x3
    while x3_cast.dtype in dtype_cast_mapping:
        x3_cast = te.lang.cce.cast_to(x3_cast,
                                      dtype_cast_mapping[x3_cast.dtype])

    x1_sqr_sum = te.lang.cce.vadds(x3_cast,
                                   tvm.const(eps, dtype=intermediate_dtype))

    x2_cast_broadcast = te.lang.cce.broadcast(x2_cast, x1_shape)

    x1_scaled = te.lang.cce.vmul(x1_cast, x2_cast_broadcast)

    if cce_product in ("Ascend910",):
        # for cloud
        x1_sqr_sum_sqrt = te.lang.cce.vsqrt(x1_sqr_sum)
        x1_sqr_sum_sqrt_broadcast = te.lang.cce.broadcast(x1_sqr_sum_sqrt,
                                                          x1_shape)
        x1_normalized = te.lang.cce.vdiv(x1_scaled, x1_sqr_sum_sqrt_broadcast)
    elif cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "Ascend610", "Ascend620") and \
            hasattr(te.lang.cce, "te_compute") and \
            hasattr(te.lang.cce.te_compute, "elewise_compute") and \
            hasattr(te.lang.cce.te_compute.elewise_compute,
                    "__binary_elewise_op"):
        # customized for hisi-es
        x1_sqr_sum_sqrt = te.lang.cce.vsqrt(x1_sqr_sum)
        x1_sqr_sum_sqrt_broadcast = te.lang.cce.broadcast(x1_sqr_sum_sqrt,
                                                          x1_shape)
        x1_normalized = te.lang.cce.te_compute.elewise_compute. \
            __binary_elewise_op(x1_scaled, x1_sqr_sum_sqrt_broadcast,
                                "elewise_binary_div")
    elif cce_product in ("Ascend310",):
        # customized for mini, using newton
        x1_sqr_sum_sqrt = te.lang.cce.vsqrt(x1_sqr_sum)

        for _ in range(1):
            res = te.lang.cce.vdiv(x1_sqr_sum, x1_sqr_sum_sqrt)
            res = te.lang.cce.vadd(res, x1_sqr_sum_sqrt)
            res = te.lang.cce.vmuls(res, tvm.const(0.5, intermediate_dtype))
            x1_sqr_sum_sqrt = res
        x1_sqr_sum_rsqrt = te.lang.cce.vrec(x1_sqr_sum_sqrt)
        x1_sqr_sum_rsqrt_broadcast = te.lang.cce.broadcast(x1_sqr_sum_rsqrt,
                                                           x1_shape)
        x1_normalized = te.lang.cce.vmul(x1_scaled, x1_sqr_sum_rsqrt_broadcast)
    else:
        # for mini and hisi-es
        x1_sqr_sum_rsqrt = te.lang.cce.vrsqrt(x1_sqr_sum)
        x1_sqr_sum_rsqrt_broadcast = te.lang.cce.broadcast(x1_sqr_sum_rsqrt,
                                                           x1_shape)
        x1_normalized = te.lang.cce.vmul(x1_scaled, x1_sqr_sum_rsqrt_broadcast)

    x1_normalized_cast = x1_normalized
    while x1_normalized_cast.dtype != x1.dtype and \
            x1_normalized_cast.dtype in dtype_reverse_cast_mapping:
        x1_normalized_cast = te.lang.cce.cast_to(x1_normalized_cast,
                                                 dtype_reverse_cast_mapping[
                                                     x1_normalized_cast.dtype])

    return x1_normalized_cast


def check_format(data_format, data_format_3):
    """
    check the format for x1 and x3

    Parameters
    ----------
    data_format : str
        the format for x1
    data_format_3 : str
        the format for x3

    Returns
    -------
    None
    """

    if data_format != data_format_3:
        raise RuntimeError("x1.format is not match with x3.format")

    if data_format not in ("NCHW", "NHWC"):
        raise RuntimeError("x1/x3.format only support NCHW or NHWC")


def check_dtype(dtype_1, dtype_3):
    """
    check the dtype for x1, x3

    Parameters
    ----------
    dtype_1 : str
        dtype for x1
    dtype_3 : str
        dtype for x3

    Returns
    -------
    None
    """

    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
        # hisi es, cs
        util.check_dtype_rule(dtype_1, ("int8", "float16",))
        util.check_dtype_rule(dtype_3, ("int8", "float16",))
    else:
        util.check_dtype_rule(dtype_1, ("int8", "float16", "float32",))
        util.check_dtype_rule(dtype_3, ("int8", "float16", "float32",))


def check_shape_1(shape_1):
    """
    check the shape for x1

    Parameters
    ----------
    shape_1 : list or tuple
        shape for x1

    Returns
    -------
    None
    """

    util.check_shape_rule(shape_1)
    util.check_tensor_shape_size(shape_1)

    if len(shape_1) != 4:
        raise RuntimeError("x1.shape only support 4D Tensor")


def check_shape_2(shape_1, data_format, channel_shared):
    """
    check the shape for x2

    Parameters
    ----------
    shape_1 : list or tuple
        shape for x1
    data_format : str
        format for x1
    channel_shared: bool
        used to control whether x2 are shared by multiple channels.
        Default(True)

    Returns
    -------
    the expand shape for x2, used for placeholder
    """

    if channel_shared:
        shape_2 = [1, 1, 1, 1]
    elif data_format == "NCHW":
        shape_2 = [1, shape_1[1], 1, 1]
    elif data_format == "NHWC":
        shape_2 = [1, 1, 1, shape_1[3]]

    return shape_2


def check_shape_3(shape_1, shape_3, data_format, across_spatial):
    """
    check the shape for x3

    Parameters
    ----------
    shape_1 : list or tuple
        shape for x1
    shape_3 : list or tuple
        shape for x3
    data_format : str
        format for x1 and x3
    across_spatial: bool
        indicates whether standardization should cross spatial locations.
        Default(True)

    Returns
    -------
    None
    """

    util.check_shape_rule(shape_3)
    util.check_tensor_shape_size(shape_3)

    if len(shape_3) != 4:
        raise RuntimeError("x3.shape only support 4D Tensor")

    if across_spatial:
        if not (shape_3[0] == shape_1[0] and shape_3[1] == 1 and
                shape_3[2] == 1 and shape_3[3] == 1):
            raise RuntimeError("x3.shape is not match with x1.shape")
    elif data_format == "NCHW":
        if not (shape_3[0] == shape_1[0] and shape_3[1] == 1 and
                shape_3[2] == shape_1[2] and shape_3[3] == shape_1[3]):
            raise RuntimeError("x3.shape is not match with x1.shape")
    elif data_format == "NHWC":
        if not (shape_3[0] == shape_1[0] and shape_3[1] == shape_1[1] and
                shape_3[2] == shape_1[2] and shape_3[3] == 1):
            raise RuntimeError("x3.shape is not match with x1.shape")


# pylint: disable=locally-disabled,invalid-name,too-many-arguments
# pylint: disable=locally-disabled,too-many-locals
@util.check_input_type(dict, dict, dict, dict, bool, bool, float, str)
def normalize_scale(x1, x2, x3, y, across_spatial=True,
                    channel_shared=True, eps=1e-10,
                    kernel_name="normalize_scale"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        shape and dtype, format of input 1
    x2 : dict
        shape and dtype, format of input 2
    x3 : dict
        shape and dtype, format of input 3
    y : dict
        shape and dtype, format of output,
        should be same shape and type, format as input 1
    across_spatial: bool
        indicates whether standardization should cross spatial locations.
        Default(True)
    channel_shared: bool
        used to control whether x2 are shared by multiple channels.
        Default(True)
    eps: float
        prevent dividing by 0.
        Default(1e-10)
    kernel_name : str
        kernel name, default value is "normalize_scale"

    Returns
    -------
    None
    """

    shape_1 = x1.get("shape")
    dtype_1 = x1.get("dtype").lower()
    data_format = x1.get("format")

    shape_3 = x3.get("shape")
    dtype_3 = x3.get("dtype").lower()
    data_format_3 = x3.get("format")

    util.check_kernel_name(kernel_name)

    check_format(data_format, data_format_3)
    check_dtype(dtype_1, dtype_3)
    check_shape_1(shape_1)
    check_shape_3(shape_1, shape_3, data_format, across_spatial)

    # the expand shape for x2, used for placeholder
    shape_2 = check_shape_2(shape_1, data_format, channel_shared)
    dtype_2 = dtype_1

    data_x1 = tvm.placeholder(shape_1, name="data_1", dtype=dtype_1)
    data_x2 = tvm.placeholder(shape_2, name="data_2", dtype=dtype_2)
    data_x3 = tvm.placeholder(shape_3, name="data_3", dtype=dtype_3)
    res = normalize_scale_compute(data_x1, data_x2, data_x3, y,
                                  across_spatial, eps, kernel_name)

    # pylint: disable=no-member
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "need_build": True,
              "name": kernel_name,
              "tensor_list": [data_x1, data_x2, data_x3, res]}

    te.lang.cce.cce_build_code(sch, config)
