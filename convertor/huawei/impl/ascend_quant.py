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

ascend_quant
"""
import te.lang.cce
import topi
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
from topi import generic
from topi.cce.util import is_lhisi_version

# define the tag of quant
ASCEND_QUANT_TAG = "quant"


# pylint: disable=too-many-arguments,invalid-name,unused-argument
# pylint: disable=unnecessary-lambda
def _check_params(x, y, scale, offset, sqrt_mode, round_mode, kernel_name):
    """
    check the parameters including shape, dtype, kernel_name, attr.
    """
    shape = x.get("shape")
    x_format = x.get("format")
    dtype = x.get("dtype").lower()
    format_list = ["NC1HWC0", "FRACTAL_NZ"]
    if x_format not in format_list:
        raise RuntimeError(
            "ascend quant only support [NC1HWC0, FRACTAL_NZ]")
    if x_format == "NC1HWC0":
        if len(shape) != 5:
            raise RuntimeError(
                "ascend quant only support the length of shape is 4 or 5")
    if x_format == "FRACTAL_NZ":
        if len(shape) != 4:
            raise RuntimeError(
                "ascend quant only support the length of shape is 4 or 5")
    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_kernel_name(kernel_name)
    if is_lhisi_version():
        # es
        check_list = ["float16"]
    else:
        check_list = ["float16", "float32"]

    if dtype not in check_list:
        raise RuntimeError("ascend quant only support %s"
                           % (",".join(check_list)))
    round_mode_list = ["Round", "Ceil", "Floor", "Trunc"]
    if round_mode not in round_mode_list:
        raise RuntimeError(
            "ascend quant only support %s while" % (",".join(round_mode_list)))


def _reform_compute_generate(tensor, in_shape, out_shape, val_info,
                             tensor_format):
    """
    generate lambda func
    Parameters
    ----------
    tensor : input tensor

    in_shape : the shape of input tensor

    out_shape :the shape of output tensor

    val_info : the val info of offset,scale

    tensor_format: the format of input tensor

    Returns
    -------
    res lambda_func
    """
    in_shape = list(in_shape)
    out_shape = list(out_shape)
    n_dim = len(in_shape)

    c0_index = n_dim - 1
    c1_index = 1
    if tensor_format == "FRACTAL_NZ":
        c1_index = 0

    def lambda_func(*indice):
        new_indice = [0] * n_dim
        for i in range(n_dim):
            if i == c0_index:
                new_indice[i] = (indice[c1_index] * out_shape[c0_index] +
                                 indice[c0_index]) % in_shape[c0_index]
            elif i == c1_index:
                new_indice[i] = (indice[c1_index] * out_shape[c0_index] +
                                 indice[c0_index]) // in_shape[c0_index]
            else:
                new_indice[i] = indice[i]

        if val_info[0]:
            return tensor(*new_indice) + val_info[1]

        return tensor(*new_indice) * val_info[2]

    return lambda_func


def _reform_by_vadds(input_tensor, input_shape, output_shape, offset_val,
                     tensor_format):
    """
    5 dim input tensor C0 change
    Parameters
    ----------
    input_tensor : input tensor

    input_shape : the shape of input tensor

    output_shape :the shape of output tensor

    offset_val : the val of offset

    tensor_format: the format of input tensor

    Returns
    -------
    res tensor
    """
    vadds_vector = tvm.compute(output_shape,
                               _reform_compute_generate(
                                   input_tensor, input_shape,
                                   output_shape, (True, offset_val, -1),
                                   tensor_format),
                               name='reform_by_vadds')

    return vadds_vector


def _reform_by_vmuls(input_tensor, input_shape, output_shape, scale_val,
                     tensor_format):
    """
    5 dim input tensor C0 change
    Parameters
    ----------
    input_tensor : input tensor

    input_shape : the shape of input tensor

    output_shape :the shape of output tensor

    scale_val : the val of scale

    tensor_format: the format of input tensor

    Returns
    -------
    res tensor
    """
    vmuls_vector = tvm.compute(output_shape,
                               _reform_compute_generate(
                                   input_tensor, input_shape,
                                   output_shape, (False, -1, scale_val),
                                   tensor_format),
                               name='reform_by_vmuls')

    return vmuls_vector


def _compute_scale(in_tensor, in_shape, out_shape, attr_list, tensor_format):
    """
    the compute of scale
    Parameters
    ----------
    in_tensor : input tensor

    in_shape : the shape of input tensor

    out_shape :the shape of output tensor

    attr_list : the attr list

    tensor_format: the format of input tensor

    Returns
    -------
    res tensor
    """
    scale = attr_list[0]
    offset = attr_list[1]
    sqrt_mode = attr_list[2]
    if scale != 1:
        scale_value = tvm.const(scale, "float16")
        scale_ub = _reform_by_vmuls(in_tensor, in_shape, out_shape,
                                    scale_value, tensor_format)
        if sqrt_mode:
            scale_sqrt_ub = tvm.compute(
                out_shape,
                lambda *indice: scale_ub(*indice) * scale_value,
                name="scale_sqrt_ub")
            res = _compute_offset(scale_sqrt_ub, in_shape, out_shape,
                                  (offset, False, scale), tensor_format)
        else:
            res = _compute_offset(scale_ub, in_shape, out_shape,
                                  (offset, False, scale), tensor_format)
    else:
        res = _compute_offset(in_tensor, in_shape, out_shape,
                              (offset, True, scale), tensor_format)
    return res


def _compute_offset(in_tensor, in_shape, out_shape, attr_list, tensor_format):
    """
    the compute of scale
    Parameters
    ----------
    in_tensor : input tensor

    in_shape : the shape of input tensor

    out_shape :the shape of output tensor

    attr_list : the attr list

    tensor_format: the format of input tensor

    Returns
    -------
    res tensor
    """
    offset = attr_list[0]
    reform_flag = attr_list[1]
    scale = attr_list[2]
    if offset != 0 or scale == 1:
        offset_value = tvm.const(offset, "float16")
        if reform_flag:
            offset_ub = _reform_by_vadds(in_tensor, in_shape, out_shape,
                                         offset_value, tensor_format)
        else:
            offset_ub = tvm.compute(
                out_shape,
                lambda *indice: in_tensor(*indice) + offset_value,
                name="offset_ub")
        cast_i8_ub = tvm.compute(out_shape,
                                 lambda *indice: topi.cast(
                                     offset_ub(*indice), "int8"),
                                 name='cast_i8_ub')
    else:
        cast_i8_ub = tvm.compute(out_shape,
                                 lambda *indice: topi.cast(
                                     in_tensor(*indice),
                                     "int8"),
                                 name='cast_i8_ub')
    return cast_i8_ub


def get_shape_info(in_shape, tensor_format):
    """
    the compute of scale
    Parameters
    ----------
    in_shape : the shape of input tensor

    tensor_format : the format of output tensor

    Returns
    -------
    read_shape, out_shape
    """
    c0_index = len(in_shape) - 1
    c1_index = 1
    c1_dim = in_shape[1]
    if tensor_format == "FRACTAL_NZ":
        c1_index = 0
        c1_dim = in_shape[0]
    out_shape = in_shape[:]
    read_shape = in_shape[:]
    read_shape[c1_index] = read_shape[c1_index] + 1 * (c1_dim % 2)
    for dim, _ in enumerate(in_shape):
        if dim == c0_index:
            out_shape[dim] = in_shape[dim] * 2
        if dim == c1_index:
            out_shape[dim] = in_shape[dim] // 2 + 1 * (c1_dim % 2)
    return read_shape, out_shape


@fusion_manager.register("ascend_quant")
def ascend_quant_compute(x, y, scale, offset, sqrt_mode=False,
                         round_mode="Round", kernel_name="ascend_quant"):
    """
    float16/float32 -> int8

    Parameters:
    ----------
    x : the tensor of input

    y : the dict of output

    scale : the data of scale

    offset : the data of offset

    sqrt_mode : the sqrt mode when true the result to do sqrt

    round_mode : the data conversion mode

    kernel_name : cce kernel name, default value is "ascend_quant"

    Returns:
    -------
    None
    """
    dtype = x.dtype
    in_shape = te.lang.cce.util.shape_to_list(x.shape)
    tensor_format = "NC1HWC0"
    # N C1 H*W for conv/matmul;
    # C1,N1,N0,C0 for matmul;
    # N,C1,H,W,C0 for depthwise_conv
    if x.op.attrs:
        if 'format' in x.op.attrs:
            # NZ format,UB convergence scenario, input shape C1,N1,N0,C0
            # the single-operator scenario, input shape 1,C1,N1*N0,C0
            tensor_format = x.op.attrs['format']
    c1_dim = in_shape[1]
    c1_index = 1
    if tensor_format == "FRACTAL_NZ":
        c1_dim = in_shape[0]
        c1_index = 0

    read_shape, out_shape = get_shape_info(in_shape, tensor_format)
    if c1_dim % 2 == 0:
        input_ub = tvm.compute(in_shape, lambda *i: x(*i),
                               name="input_ub",
                               attrs={"c_out": c1_dim})
    else:
        input_ub = tvm.compute(read_shape,
                               lambda *indice: tvm.select(
                                   indice[c1_index] <= in_shape[
                                       c1_index] - 1,
                                   x(*indice),
                                   tvm.const(0, dtype=dtype)),
                               name='input_ub',
                               attrs={"c_out": c1_dim})
    if dtype == "float32":
        cast_f16_ub = tvm.compute(read_shape,
                                  lambda *indice: topi.cast(
                                      input_ub(*indice),
                                      "float16"),
                                  name='cast_f16_ub')
        cast_i8_ub = _compute_scale(
            cast_f16_ub, in_shape, out_shape, (scale, offset, sqrt_mode),
            tensor_format)
    else:
        cast_i8_ub = _compute_scale(
            input_ub, in_shape, out_shape, (scale, offset, sqrt_mode),
            tensor_format)
    res = tvm.compute(out_shape, lambda *indice: cast_i8_ub(*indice),
                      name="res", tag=ASCEND_QUANT_TAG,
                      attrs={'scale': scale,
                             'sqrt_mode': sqrt_mode,
                             'offset': offset,
                             'round_mode': round_mode})
    return res


@util.check_input_type(dict, dict, (int, float), (int, float), bool, str, str)
def ascend_quant(x, y, scale, offset, sqrt_mode=False, round_mode="Round",
                 kernel_name="ascend_quant"):
    """
    float16/float32 -> int8

    Parameters:
    ----------
    x : the dict of input

    y : the dict of output

    scale : the data of scale

    offset : the data of offset

    sqrt_mode : the sqrt mode when true the result to do sqrt

    round_mode : the data conversion mode

    kernel_name : cce kernel name, default value is "ascend_quant"

    Returns:
    -------
    None
    """
    _check_params(x, y, scale, offset, sqrt_mode, round_mode, kernel_name)
    shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    input_format = x.get("format")
    if input_format == "NC1HWC0":
        # change to N,C1,H*W,C0
        input_shape = (shape[0],
                       shape[1],
                       shape[2] * shape[3],
                       shape[4])
    else:
        # nz change to 1,C1,N1*N0,C0 equivalence N,C1,H*W,C0
        input_shape = (1,
                       shape[0],
                       shape[1] * shape[2],
                       shape[3])
    input_x = tvm.placeholder(input_shape,
                              name="input_x",
                              dtype=input_dtype)

    res = ascend_quant_compute(input_x, y, scale, offset, sqrt_mode,
                               round_mode, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_x, res]}

    te.lang.cce.cce_build_code(sch, config)
