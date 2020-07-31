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

ascend_anti_quant
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
import topi
from te.platform.cce_build import build_config


# define the tag of anti_quant
ASCEND_ANTI_QUANT_TAG = "anti_quant"
BLOCK_VALUE = 16

# pylint: disable=too-many-arguments,invalid-name,unused-argument
# pylint: disable=unnecessary-lambda
def _check_params(x, y, scale, offset, sqrt_mode, kernel_name):
    """
    check the parameters including shape, dtype, kernel_name, attr.
    """
    shape = x.get("shape")
    x_format = x.get("format")
    x_dtype = x.get("dtype").lower()
    if x_format != "NC1HWC0":
        raise RuntimeError("ascend anti quant only support NC1HWC0")
    if len(shape) != 5:
        raise RuntimeError(
            "ascend anti quant only support the length of shape is 5")
    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_kernel_name(kernel_name)

    if x_dtype not in ("int8",):
        raise RuntimeError("ascend anti quant only support int8")
    if not isinstance(sqrt_mode, bool):
        raise RuntimeError("ascend anti quant, sqrt_mode must be bool")


def _reform_compute_generate(tensor, in_shape, out_shape, scale_val):
    """
    generate lambda func

    Parameters
    ----------
    tensor : input tensor
    in_shape : the shape of input tensor
    out_shape :the shape of output tensor
    scale_val : the value of scale

    Returns
    -------
    res lambda_func
    """
    in_shape = list(in_shape)
    out_shape = list(out_shape)
    n_dim = len(in_shape)

    def lambda_func(*indice):
        new_indice = [indice[0],
                      (indice[1] * out_shape[n_dim - 1] +
                       indice[n_dim - 1])
                      // in_shape[n_dim - 1]] \
                     + list(indice[2:n_dim - 1]) \
                     + [(indice[1] * out_shape[n_dim - 1] +
                         indice[n_dim - 1])
                        % in_shape[n_dim - 1]]

        return tensor(*new_indice) * scale_val

    return lambda_func


def _reform_by_vmuls(input_tensor, input_shape, output_shape, scale_val):
    """
    5 dim input tensor C0 change

    Parameters
    ----------
    input_tensor : input tensor
    input_shape : the shape of input tensor
    output_shape :the shape of output tensor
    scale_val : the value of scale

    Returns
    -------
    res tensor
    """
    vmuls_vector = tvm.compute(output_shape,
                               _reform_compute_generate(
                                   input_tensor, input_shape,
                                   output_shape, scale_val),
                               name='reform_by_vmuls')

    return vmuls_vector


@fusion_manager.register("ascend_anti_quant")
def ascend_anti_quant_compute(x, y, scale, offset, sqrt_mode=False,
                              kernel_name="ascend_anti_quant"):
    """
    int8 -> float16/float32

    Parameters:
    ----------
    x : the tensor of input
    y : the dict of output
    scale : the data of scale
    offset : the data of offset
    sqrt_mode : the sqrt mode when true the result to do sqrt
    kernel_name : cce kernel name, default value is "ascend_anti_quant"

    Returns:
    -------
    None
    """
    in_shape = te.lang.cce.util.shape_to_list(x.shape)
    out_shape = (in_shape[0], in_shape[1]*2, in_shape[2],
                 in_shape[3], in_shape[4] // 2)

    input_ub = tvm.compute(in_shape, lambda *i: x(*i), name="input_ub")
    # cast int8 to fp16
    cast_f16_ub = tvm.compute(in_shape,
                              lambda *indice: topi.cast(
                                  input_ub(*indice),
                                  "float16"),
                              name='cast_f16_ub')

    # add offset
    offset_value = tvm.const(offset, "float16")
    offset_ub = tvm.compute(
        in_shape,
        lambda *indice: cast_f16_ub(*indice) + offset_value,
        name="offset_ub")

    scale_value = tvm.const(scale, "float16")
    if sqrt_mode:
        scale_sqrt_ub = tvm.compute(
            in_shape,
            lambda *indice: offset_ub(*indice) * scale_value,
            name="scale_sqrt_ub")
        scale_ub = _reform_by_vmuls(scale_sqrt_ub, in_shape, out_shape,
                                    scale_value)
    else:
        # mul scale and convert 32 to 16 of C0
        scale_ub = _reform_by_vmuls(offset_ub, in_shape, out_shape, scale_value)

    ori_shape = y.get('ori_shape')
    ori_format = y.get('ori_format')
    if ori_format == "NHWC":
        ori_shape = [ori_shape[0], ori_shape[3], ori_shape[1], ori_shape[2]]

    ori_c = ori_shape[1]
    # remove pad
    if ori_c % 32 > 0 and ori_c % 32 <= 16:
        tmp_res = tvm.compute(out_shape, lambda *indice: scale_ub(*indice),
                              name="tmp_res")

        align_shape = [ori_shape[0],
                       (ori_shape[1] + BLOCK_VALUE - 1) // BLOCK_VALUE,
                       ori_shape[2], ori_shape[3], BLOCK_VALUE]

        res = tvm.compute(align_shape, lambda *indice: tmp_res(*indice),
                          name="res", tag=ASCEND_ANTI_QUANT_TAG,
                          attrs={'scale': scale,
                                 'sqrt_mode': sqrt_mode,
                                 'offset': offset})
    else:
        res = tvm.compute(out_shape, lambda *indice: scale_ub(*indice),
                          name="res", tag=ASCEND_ANTI_QUANT_TAG,
                          attrs={'scale': scale,
                                 'sqrt_mode': sqrt_mode,
                                 'offset': offset})

    return res


# pylint: disable=too-many-arguments,invalid-name,unused-argument
@util.check_input_type(dict, dict, (int, float), (int, float), bool, str)
def ascend_anti_quant(x, y, scale, offset, sqrt_mode=False,
                      kernel_name="ascend_anti_quant"):
    """
    int8 -> float16

    Parameters:
    ----------
    x : the dict of input, format is NC1HWC0
    y : the dict of output, format is NC1HWC0
    scale : the data of scale
    offset : the data of offset
    sqrt_mode : the sqrt mode when true the result to do sqrt
    kernel_name : cce kernel name, default value is "ascend_anti_quant"

    Returns:
    -------
    None
    """
    _check_params(x, y, scale, offset, sqrt_mode, kernel_name)
    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    input_x = tvm.placeholder(input_shape,
                              name="input_x",
                              dtype=input_dtype)

    res = ascend_anti_quant_compute(input_x, y, scale, offset, sqrt_mode,
                                    kernel_name)
    with tvm.target.cce():
        generic.auto_schedule(res)
