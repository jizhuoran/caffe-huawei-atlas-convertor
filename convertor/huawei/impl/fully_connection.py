#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License == distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

fully_connection
"""
from __future__ import absolute_import
import te.lang.cce
from te import tvm
from topi.cce import util
from te.platform.fusion_manager import fusion_manager
from topi import generic

NoneType = type(None)


# pylint: disable=locally-disabled,too-many-arguments, too-many-locals
# pylint: disable=too-many-statements, invalid-name, unused-argument
def fully_connection_check_rule(x, w, b, offset_w, y, num_output, transpose, axis, offset_x,
                                kernel_name="fully_connection"):
    """check input params"""
    # x info
    shape_x = x.get('shape')
    dtype_x = x.get('dtype')
    format_x = x.get('format')
    km_shape = shape_x[1] * shape_x[2] * shape_x[3] * shape_x[4]

    if shape_x[-1] not in (16, 32):
        raise RuntimeError("no_quant x C0 must be 16!, quant x C0 must be 32!")

    util.check_dtype_rule(dtype_x, ['float16', 'int8'])

    if format_x != 'NC1HWC0':
        raise RuntimeError("x format must be NC1HWC0!")

    # w info
    shape_w = w.get('shape')
    dtype_w = w.get('dtype')
    format_w = w.get('format')

    util.check_dtype_rule(dtype_w, ['float16', 'int8'])

    if format_w != 'FRACTAL_Z':
        raise RuntimeError("w format must be FRACTAL_Z!")

    # format shape info
    if dtype_x == 'float16' and (shape_w[2] != 16 or shape_w[3] != 16):
        raise RuntimeError("for no quant, w last two dims must be 16!")
    if dtype_x == 'int8' and (shape_w[2] != 16 or shape_w[3] != 32):
        raise RuntimeError("for quant, w last two dims must be 16 and 32!")

    kn_shape = shape_w[0] * shape_w[3]
    n_shape = shape_w[1] * shape_w[2]

    # Check shape
    if km_shape != kn_shape:
        raise RuntimeError("KM must equal to KN!")

    # b info
    if b is not None:
        shape_b = b.get('shape')
        dtype_b = b.get('dtype')
        format_b = b.get('format')
        b_size = shape_b[1] * shape_b[4]

        # Check info
        util.check_dtype_rule(dtype_b, ['float16', 'int32'])
        if format_b != 'NC1HWC0':
            raise RuntimeError("For FullyConnection, b input format must be NC1HWC0!")
        if b_size != n_shape:
            raise RuntimeError("For bias, the C1*C0 must equal to aligned_Cout!")

    # axis info
    if axis not in (1, -3):
        raise RuntimeError("axis only support 1,-3 when reduce from channel!")


@fusion_manager.register("fully_connection")
def fully_connection_compute(x, w, b, offset_w, y, num_output, transpose, axis, offset_x,
                             kernel_name="fully_connection"):
    """
    x : the tensor of input x

    w: the tensor of intput w

    b: the tensor of bias

    offset_w : unused

    num_output : output neuron number

    transpose: is weight transpose

    axis: the beginning reduce axis, reduce axis is [axis, last_axis]

    offset_x: unused

    Returns:
    -------
    None
    """
    format_a = 'ND'
    format_b = 'fractal'

    quantize_params = None
    if offset_w is not None:
        raise RuntimeError("For FullyConnection, tensor offset_w must be None!")

    result = te.lang.cce.matmul(tensor_a=x, tensor_b=w, trans_a=False, trans_b=transpose,
                                format_a=format_a, format_b=format_b, alpha_num=1.0, beta_num=0.0,
                                dst_dtype='float16', tensor_bias=b, quantize_params=quantize_params)
    return result


@util.check_input_type(dict, dict, (dict, NoneType), (dict, NoneType),
                       dict, int, bool, int, int, str)
def fully_connection(x, w, b, offset_w, y, num_output, transpose, axis, offset_x,
                     kernel_name="fully_connection"):
    """
    x : the dict of input x

    w: the dict of intput w

    b: the dict of bias

    offset_w : unused

    num_output : output neuron number

    transpose: is weight transpose

    axis: the beginning reduce axis, reduce axis is [axis, last_axis]

    offset_x: unused

    Returns:
    -------
    None
    """
    # Check params
    fully_connection_check_rule(x, w, b, offset_w, y, num_output, transpose, axis, offset_x,
                                kernel_name="fully_connection")

    # x info
    shape_x = x.get('shape')
    dtype_x = x.get('dtype')

    shape_x_final = (shape_x[0], shape_x[1] * shape_x[2] * shape_x[3] * shape_x[4])
    tensor_x = tvm.placeholder(shape_x_final, dtype=dtype_x, name='tensor_a')

    # w info
    shape_w = w.get('shape')
    dtype_w = w.get('dtype')
    tensor_w = tvm.placeholder(shape_w, dtype=dtype_w, name='tensor_b')

    # b info
    if b is not None:
        shape_b = b.get('shape')
        dtype_b = b.get('dtype')
        shape_bias = (shape_b[1] * shape_b[4],)
        tensor_b = tvm.placeholder(shape_bias, dtype=dtype_b, name='tensor_bias')
    else:
        tensor_b = None

    # offset_w info
    if offset_w is None:
        tensor_offset_w = None
    else:
        raise RuntimeError("offset_w must be None!")

    # Compute
    result = fully_connection_compute(tensor_x, tensor_w, tensor_b, tensor_offset_w, y,
                                      num_output, False, axis, offset_x)

    # Schedule
    with tvm.target.cce():
        schedule = generic.auto_schedule(result)

    # CCE build
    if b is not None:
        tensor_list = [tensor_x, tensor_w, tensor_b, result]
    else:
        tensor_list = [tensor_x, tensor_w, result]

    config = {"print_ir": False, "need_build": True, "need_print": True,
              "name": kernel_name, "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(schedule, config)
