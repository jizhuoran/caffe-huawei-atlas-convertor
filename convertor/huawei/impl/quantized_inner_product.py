#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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

quantized_inner_product
"""
from __future__ import absolute_import
from functools import reduce as functools_reduce
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
from topi import generic

SHAPE_SIZE_LIMIT = 2 ** 31 - 1
NoneType = type(None)


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals,invalid-name
# pylint: disable=locally-disabled,too-many-branches,unused-argument,too-many-statements
# pylint: disable=locally-disabled,simplifiable-if-expression
def quantized_inner_product_check_rule(x, w, b, scale_q, offset_q,
                                       scale_deq_req, offset_req, y, quant_algo,
                                       scale_sqrt,
                                       num_output, transpose, bias_term, axis,
                                       kernel_name="quantized_inner_product"):
    """
    Check the legality of each entry
    """
    # x info
    shape_x = x.get('shape')
    dtype_x = x.get('dtype')
    format_x = x.get('format')
    m_shape = shape_x[0]
    km_shape = shape_x[1] * shape_x[2] * shape_x[3] * shape_x[4]

    if functools_reduce(lambda x, y: x * y, shape_x) >= SHAPE_SIZE_LIMIT:
        raise RuntimeError("The shape_x exceed 32 bit limitations! ")

    if shape_x[-1] != 32:
        raise RuntimeError("For non_quant 'NC1HWC0' x, the C0 must be 32!")

    util.check_dtype_rule(dtype_x, ['uint8'])

    if format_x != 'NC1HWC0':
        raise RuntimeError("For IP situation, x format must be NC1HWC0!")

    # gevm
    is_gevm = m_shape == 1
    if is_gevm:
        if km_shape % 512 != 0:
            raise RuntimeError("for quant_gevm, KM/KN must be multi of 512!")

    # w info
    shape_w = w.get('shape')
    dtype_w = w.get('dtype')
    format_w = w.get('format')

    if functools_reduce(lambda x, y: x * y, shape_w) >= SHAPE_SIZE_LIMIT:
        raise RuntimeError("The shape_w exceed 32 bit limitations! ")

    util.check_dtype_rule(dtype_w, ['int8'])

    if format_w != 'FRACTAL_Z':
        raise RuntimeError(
            "For quant IP situation, w format must be FRACTAL_Z!")

    if shape_w[2] != 16 or shape_w[3] != 32:
        raise RuntimeError(
            "For quant IP situation, last two dim must be 16 and 32!")

    kn_shape = shape_w[0] * shape_w[3]
    n_shape = shape_w[1] * shape_w[2]

    # Check shape
    if km_shape != kn_shape:
        raise RuntimeError("KM of input_x must be equal to KN of input_w!")

    # y info
    shape_y = y.get('shape')
    dtype_y = y.get('dtype')
    format_y = y.get('format')

    if shape_y[-1] != 16:
        raise RuntimeError("For Quant 'NC1HWC0' y, the C0 must be 32!")

    util.check_dtype_rule(dtype_y, ['float16'])

    if format_y != 'NC1HWC0':
        raise RuntimeError("For IP situation, y format must be NC1HWC0!")

    # b info
    if bias_term:
        shape_b = b.get('shape')
        dtype_b = b.get('dtype')
        format_b = b.get('format')
        b_size = shape_b[1] * shape_b[4]
        # Check info
        util.check_dtype_rule(dtype_b, ['int32'])
        if format_b != 'NC1HWC0':
            raise RuntimeError("For IP situation, b format must be NC1HWC0!")
        if b_size != n_shape:
            raise RuntimeError(
                "For bias, the C1*C0 must equal to aligned_Cout!")
    else:
        if b is not None:
            raise RuntimeError("for bias_term false, the b must be an None!")

    if transpose:
        raise RuntimeError("for quantized IP, only support transpose false")


def quantized_params_check(scale_q, offset_q, scale_deq_req, offset_req,
                           quant_algo, scale_sqrt):
    """
    Check the legality of each entry
    """
    quantize_params = {}

    if scale_q is not None:
        raise RuntimeError("IP quantized not support scale_q!")
    if offset_q is not None:
        raise RuntimeError("IP quantized not support offset_q!")
    if offset_req is not None:
        raise RuntimeError("IP quantized not support offset_req!")

    if quant_algo[0] not in (0, 1):
        raise RuntimeError("quant_algo[0] only support 0 and 1!")
    if quant_algo[0] == 0:
        quantize_params['quantize_alg'] = 'NON_OFFSET'
    elif quant_algo[0] == 1:
        quantize_params['quantize_alg'] = 'HALF_OFFSET_A'

    if quant_algo[1] != 0:
        raise RuntimeError("quant_algo[1] only support 0!")
    quantize_params['scale_mode_out'] = 'SCALAR'

    if scale_sqrt[0] != 0 or scale_sqrt[2] != 0:
        raise RuntimeError("scale_sqrt[0] and scale_sqrt[2] must be 0!")

    if scale_sqrt[1] == 0:
        quantize_params['sqrt_mode_out'] = 'NON_SQRT'
    elif scale_sqrt[1] == 1:
        quantize_params['sqrt_mode_out'] = 'SQRT'
    else:
        raise RuntimeError("scale_sqrt[1] only support 0 and 1!")

    quantize_params['scale_drq'] = scale_deq_req

    return quantize_params


@fusion_manager.register("quantized_inner_product")
def quantized_inner_product_compute(x, w, b, scale_q, offset_q, scale_deq_req,
                                    offset_req, y, quant_algo, scale_sqrt,
                                    num_output, transpose, bias_term, axis,
                                    kernel_name="quantized_inner_product"):
    """
    quantized_inner_product's compute interface
    """
    format_a = 'ND'
    format_b = 'fractal'

    quantize_params = quantized_params_check(scale_q, offset_q, scale_deq_req,
                                             offset_req, quant_algo,
                                             scale_sqrt)

    result = te.lang.cce.matmul(tensor_a=x, tensor_b=w, trans_a=False,
                                trans_b=transpose,
                                format_a=format_a, format_b=format_b,
                                alpha_num=1.0, beta_num=0.0,
                                dst_dtype='float16', tensor_bias=b,
                                quantize_params=quantize_params)

    return result


# pylint: disable=locally-disabled,too-many-arguments, too-many-locals, too-many-statements
@util.check_input_type(dict, dict, (dict, NoneType), (dict, NoneType),
                       (dict, NoneType), (dict, NoneType),
                       (dict, NoneType), dict, (list, tuple), (list, tuple),
                       int, bool, bool, int, str)
def quantized_inner_product(x, w, b, scale_q, offset_q, scale_deq_req,
                            offset_req, y, quant_algo, scale_sqrt,
                            num_output, transpose, bias_term, axis,
                            kernel_name="quantized_inner_product"):
    """
    :param x: dict,shape and dtype of input
    :param w: dict,shape and dtype of input
    :param b: dict,shape and dtype of input
    :param scale_q: dict,shape and dtype of input
    :param offset_q: dict,shape and dtype of input
    :param scale_deq_req: dict,shape and dtype of input
    :param offset_req: dict,shape and dtype of input
    :param y: dict,shape and dtype of output
    :param quant_algo: attr,listint
    :param scale_sqrt: attr,listint
    :param num_output: attr,int
    :param transpose: attr,bool
    :param bias_term: attr,bool
    :param axis: attr,int
    """
    # Check params
    quantized_inner_product_check_rule(x, w, b, scale_q, offset_q,
                                       scale_deq_req, offset_req, y, quant_algo,
                                       scale_sqrt,
                                       num_output, transpose, bias_term, axis)
    # x info
    shape_x = x.get('shape')
    dtype_x = x.get('dtype')

    shape_x_final = (shape_x[0],
                     shape_x[1] * shape_x[2] * shape_x[3] * shape_x[4])

    tensor_x = tvm.placeholder(shape_x_final, dtype=dtype_x, name='tensor_a')

    # w info
    shape_w = w.get('shape')
    dtype_w = w.get('dtype')

    tensor_w = tvm.placeholder(shape_w, dtype=dtype_w, name='tensor_b')

    # b info
    if bias_term:
        shape_b = b.get('shape')
        dtype_b = b.get('dtype')
        # choose bias for template
        b_size = shape_b[1] * shape_b[4]
        shape_bias = (1, b_size)
        tensor_b = tvm.placeholder(shape_bias, dtype=dtype_b,
                                   name='tensor_bias')
    else:
        tensor_b = None

    shape_req = scale_deq_req.get('shape')
    dtype_req = scale_deq_req.get('dtype')

    scale_drq = tvm.placeholder(shape_req, dtype=dtype_req, name='scale_drq')

    # quantize params info
    quantize_params = quantized_params_check(scale_q, offset_q, scale_drq,
                                             offset_req, quant_algo, scale_sqrt)
    # Compute
    result = quantized_inner_product_compute(tensor_x, tensor_w, tensor_b,
                                             scale_q, offset_q, scale_drq,
                                             offset_req,
                                             y, quant_algo, scale_sqrt,
                                             num_output, transpose, bias_term,
                                             axis)
    # Schedule
    with tvm.target.cce():
        schedule = generic.auto_schedule(result)

    # CCE build
    if bias_term:
        tensor_list = [tensor_x, tensor_w, tensor_b,
                       quantize_params['scale_drq'], result]
    else:
        tensor_list = [tensor_x, tensor_w, quantize_params['scale_drq'], result]

    config = {"print_ir": False, "need_build": True, "need_print": True,
              "name": kernel_name, "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(schedule, config)
