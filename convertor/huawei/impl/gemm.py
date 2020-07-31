#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License == distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

gemm
"""
from __future__ import absolute_import

from math import ceil
# pylint: disable=import-error
import te.lang.cce
import te.platform.cce_params as cce
from te import tvm
from topi import generic
from topi.cce import util

from impl.matmul_vector import matmul_vector_cce
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

ALPHA_BETA_SHAPE = [1]
NoneType = type(None)

ND_THRESHOLD = 0


def op_select_format(input_x1, input_x2, # pylint: disable=too-many-arguments
                     alpha, beta, bias=None, output_y=None, trans_a=False,
                     trans_b=False, kernel_name="gemm"):
    """
    select format dynamically
    """
    def _select_format(params):
        input_x1 = params[0]
        input_x2 = params[1]
        shape_a = input_x1.get("ori_shape")
        shape_b = input_x2.get("ori_shape")
        src_dtype = input_x1.get("dtype")
        need_transdata = not (
            shape_a[0] <= ND_THRESHOLD and
            shape_a[1] <= ND_THRESHOLD and
            shape_b[1] <= ND_THRESHOLD and
            src_dtype == "float16" and
            shape_a[0] % 16 == 0 and
            shape_a[1] % 16 == 0 and shape_b[1] % 16 == 0
        )
        if need_transdata:
            input0 = gen_param(
                classify="input0",
                name="a",
                datatype="float16,float16,int8,int8",
                format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ",
            )
            input1 = gen_param(
                classify="input1",
                name="b",
                datatype="float16,float16,int8,int8",
                format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_Z,FRACTAL_Z",
            )
            input2 = gen_param(
                classify="input2",
                name="c",
                datatype="float32,float16,int32,float32",
                format="FRACTAL_NZ,FRACTAL_NZ,ND,FRACTAL_NZ",
            )
            output0 = gen_param(
                classify="output0",
                name="out",
                datatype="float32,float16,int32,float32",
                format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ",
            )
        else:
            input0 = gen_param(
                classify="input0",
                name="a",
                datatype="float16,float16,int8,int8",
                format="ND,ND,ND,ND",
            )
            input1 = gen_param(
                classify="input1",
                name="b",
                datatype="float16,float16,int8,int8",
                format="ND,ND,ND,ND",
            )
            input2 = gen_param(
                classify="input2",
                name="c",
                datatype="float32,float16,int32,float32",
                format="ND,ND,ND,ND",
            )
            output0 = gen_param(
                classify="output0",
                name="out",
                datatype="float32,float16,int32,float32",
                format="ND,ND,ND,ND",
            )
        input3 = gen_param(
            classify="input3",
            name="alpha",
            datatype="float32,float16,int32,float32",
            format="ND,ND,ND,ND",
        )
        input4 = gen_param(
            classify="input4",
            name="beta",
            datatype="float32,float16,int32,float32",
            format="ND,ND,ND,ND",
        )
        return [input0, input1, input2, input3, input4, output0]

    params = [input_x1, input_x2, alpha, beta, bias, output_y, trans_a,
              trans_b, kernel_name]
    param_list = _select_format(params)
    return get_dynamic_param_in_json(param_list)

# pylint: disable=locally-disabled,too-many-arguments,too-many-branches, too-many-statements, too-many-locals,
def _shape_check(
        shape_a, shape_b, shape_bias, src_dtype,
        trans_a, trans_b, alpha_dtype, beta_dtype, dst_dtype
):
    """
    Check the given input if legal

    Parameters:
    shape_a: list or tuple
            Shape of the first tensor a with rank > 1
    shape_b:  list or tuple
            Shape of the second tensor b with the same type with a,
            and shape_a, shape_b must be 2 dims
    shape_bias: list or tuple
            Shape of bias, only support the input data format with ND
    src_dtype: str
            The data type of input, support "float32", "float16"
    trans_a: bool
            If True, shape_a == transposed before multiplication
    trans_b: bool
            If True, shape_b == transposed before multiplication

    Returns None
    """
    if alpha_dtype != beta_dtype:
        raise RuntimeError("dtype of alpha and beta are not same!")

    if alpha_dtype != dst_dtype:
        raise RuntimeError("dtype of alpha/beta and dst are not same!")

    if src_dtype == "int8":
        if dst_dtype not in ["int32", "float32"]:
            raise RuntimeError(
                "not support src_type:{} and dst_type:{}".format(
                    src_dtype, dst_dtype,
                )
            )
    elif src_dtype == "float16":
        if dst_dtype not in ["float16", "float32"]:
            raise RuntimeError(
                "not support src_type:{} and dst_type:{}".format(
                    src_dtype, dst_dtype,
                )
            )

    shape_len = len(shape_a)
    src_dtype = src_dtype.lower()

    check_list = ("float16", "int8")

    if src_dtype not in check_list:
        raise RuntimeError("gemm_cce only support float16,int8")

    if shape_len != len(shape_b):
        raise RuntimeError("length of a and b are not equal")

    if shape_len != 2:
        raise RuntimeError(
            "length of shape must be 2, "
            "more than 2 dimensions should use batch_matmul now!"
        )

    if trans_a:
        m_shape = shape_a[shape_len - 1]
        km_shape = shape_a[shape_len - 2]
    else:
        m_shape = shape_a[shape_len - 2]
        km_shape = shape_a[shape_len - 1]

    if trans_b:
        kn_shape = shape_b[shape_len - 1]
        n_shape = shape_b[shape_len - 2]
    else:
        kn_shape = shape_b[shape_len - 2]
        n_shape = shape_b[shape_len - 1]

    if km_shape != kn_shape:
        raise RuntimeError("reduce axis not same")

    if shape_bias:
        if len(shape_bias) == shape_len:
            if shape_bias[-2:] != [m_shape, n_shape]:
                raise RuntimeError(
                    "non broadcast bias shape must be same as output shape")
        else:
            raise RuntimeError("unsupport input shape now for batch bias case")


def _get_bias_element(shape_bias_element):
    bias_length = shape_bias_element
    if bias_length % 16 == 0:
        return bias_length
    bias_length = (bias_length // 16) * 16 + 16
    return bias_length


def _get_bias(shape_bias):
    for index, value in enumerate(shape_bias):
        shape_bias[index] = _get_bias_element(value)
    return shape_bias


def _get_input_shape_a(shape_x, dtype):
    dim_a = shape_x[0]
    dim_b = shape_x[1]
    res = list()
    block_in = cce.BLOCK_IN

    if dtype == "float16":
        block_reduce = cce.BLOCK_REDUCE
    else:
        block_reduce = cce.BLOCK_REDUCE_INT8

    res.append(ceil(dim_a/block_in)*block_in)
    res.append(ceil(dim_b/block_reduce)*block_reduce)
    return res


def _get_input_shape_b(shape_x, dtype):
    dim_a = shape_x[0]
    dim_b = shape_x[1]
    res = list()
    block_out = cce.BLOCK_OUT

    if dtype == "float16":
        block_reduce = cce.BLOCK_REDUCE
    else:
        block_reduce = cce.BLOCK_REDUCE_INT8

    res.append(ceil(dim_a/block_reduce)*block_reduce)
    res.append(ceil(dim_b/block_out)*block_out)
    return res

# pylint: disable=unused-argument, too-many-return-statements
def check_supported(input_x1, input_x2, alpha, beta, bias=None, output_y=None,
                    trans_a=False, trans_b=False, kernel_name="gemm"):
    """check support"""
    shape_a = input_x1.get("shape")
    shape_b = input_x2.get("shape")
    src_dtype = input_x1.get("dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_a)
    util.check_shape_rule(shape_b)

    if src_dtype == "float16":
        if len(shape_a) != 2 and len(shape_b) != 2:
            return False

        if trans_a:
            k_shape = shape_a[0]
        else:
            k_shape = shape_a[1]

        if trans_b:
            k_b_shape = shape_b[1]
        else:
            k_b_shape = shape_b[0]

        if k_shape != k_b_shape:
            return False

    return True


# pylint: disable=locally-disabled,too-many-arguments, too-many-locals, too-many-statements
@util.check_input_type(dict, dict, dict, dict, dict, dict, bool, bool, str)
def gemm(input_x1, input_x2, bias, alpha, beta, output_y=None, trans_a=False,
         trans_b=False, kernel_name="gemm"):
    """
    calculating  matrix multiplication with bias, C = alpha*A*B + beta*bias, support input
    data with Nz format.

    Parameters:
    input_x1: dict
            shape and dtype of tensor_a
    input_x2: dict
            shape and dtype of tensor_b
    alpha: shape and dtype of alpha
    beta: shape and dtype of beta
    bias: dict
            Shape of bias, support the input data format with Nz/ND in different scenes
    trans_a:
            whether transpose a
            only support false
    trans_b:
            whether transpose b
            only support false
    Returns
    -------
    None
    """
    if output_y is None:
        output_y = {}
    # 现阶段，trans_a/trans_b不由算子层处理
    trans_a = False
    trans_b = False
    if trans_a or trans_b:
        raise RuntimeError("not support transpose now!")

    shape_a = input_x1.get("ori_shape")
    shape_b = input_x2.get("ori_shape")

    src_dtype = input_x1.get("dtype").lower()
    dst_dtype = output_y.get("dtype").lower()

    if shape_a is not None:
        if len(shape_a) < 2:
            shape_a = input_x1.get("shape")

    if shape_b is not None:
        if len(shape_b) < 2:
            shape_b = input_x2.get("shape")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_a)
    util.check_shape_rule(shape_b)


    alpha_dtype = alpha.get("dtype")
    beta_dtype = beta.get("dtype")

    shape_bias = bias.get("ori_shape")
    shape_bias = list(shape_bias)

    _shape_check(
        shape_a, shape_b, shape_bias, src_dtype,
        trans_a, trans_b, alpha_dtype, beta_dtype, dst_dtype
    )

    if input_x1.get("format") != "ND":
        shape_a = _get_input_shape_a(list(shape_a), src_dtype)
        shape_b = _get_input_shape_b(list(shape_b), src_dtype)
    if bias.get("format") != "ND":
        shape_bias = _get_bias(shape_bias)

    if input_x1.get("format") == "FRACTAL_NZ":
        shape_a = [shape_a[1], shape_a[0]]
        trans_a = bool(1 - trans_a)

    if input_x2.get("format") == "FRACTAL_NZ":
        shape_b = [shape_b[1], shape_b[0]]
        trans_b = bool(1 - trans_b)

    if bias is None or not bool(bias):
        raise RuntimeError("unsupport bias is None")

    if src_dtype in ["float32", "int32"]:
        matmul_vector_cce(shape_a, shape_b, src_dtype, trans_a, trans_b,
                          shape_bias, kernel_name)
        return

    m_shape = shape_a[len(shape_a) - 2]
    km_shape = shape_a[len(shape_a) - 1]
    kn_shape = shape_b[len(shape_a) - 2]
    n_shape = shape_b[len(shape_a) - 1]

    if src_dtype == "float16":
        block_reduce = cce.BLOCK_REDUCE
    else:
        block_reduce = cce.BLOCK_REDUCE_INT8

    block_in = cce.BLOCK_IN
    block_out = cce.BLOCK_OUT

    if trans_a:
        shape_a_temp = (
            m_shape // block_reduce, km_shape // block_in, block_in,
            block_reduce
        )
    else:
        shape_a_temp = (
            m_shape // block_in, km_shape // block_reduce, block_in,
            block_reduce
        )

    if trans_b:
        shape_b_temp = (
            kn_shape // block_out, n_shape // block_reduce, block_reduce,
            block_out
        )
    else:
        shape_b_temp = (
            kn_shape // block_reduce, n_shape // block_out, block_out,
            block_reduce
        )

    if input_x1.get("format") == "FRACTAL_Z":
        shape_a_temp = (
            shape_a_temp[0], shape_a_temp[1], shape_a_temp[2], shape_a_temp[3]
        )
        format_a = "fractal"
    elif input_x1.get("format") == "FRACTAL_NZ":
        shape_a_temp = (
            shape_a_temp[0], shape_a_temp[1], shape_a_temp[2], shape_a_temp[3]
        )
        format_a = "FRACTAL_NZ"
    else:
        shape_a_temp = (shape_a[len(shape_a) - 2], shape_a[len(shape_a) - 1])
        format_a = "ND"

    if input_x2.get("format") == "FRACTAL_Z":
        shape_b_temp = (
            shape_b_temp[0], shape_b_temp[1], shape_b_temp[2], shape_b_temp[3]
        )
        format_b = "fractal"
    elif input_x2.get("format") == "FRACTAL_NZ":
        shape_b_temp = (
            shape_b_temp[0], shape_b_temp[1], shape_b_temp[2], shape_b_temp[3]
        )
        format_b = "FRACTAL_NZ"
    else:
        shape_b_temp = (shape_b[len(shape_b) - 2], shape_b[len(shape_b) - 1])
        format_b = "ND"

    # 获取Nz格式的bias shape
    if bias.get("format") != "ND":
        shape_bias_temp = (
            shape_bias[1] // block_out, shape_bias[0] // block_in, block_in,
            block_out,
            )
    else:
        shape_bias_temp = shape_bias

    tensor_a = tvm.placeholder(shape_a_temp, name='tensor_a',
                               dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b_temp, name='tensor_b',
                               dtype=src_dtype)

    tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name='tensor_alpha',
                                   dtype=alpha_dtype)
    tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name='tensor_beta',
                                  dtype=alpha_dtype)

    tensor_bias = tvm.placeholder(shape_bias_temp, name='tensor_bias',
                                  dtype=dst_dtype)

    result = te.lang.cce.gemm(
        tensor_a, tensor_b, tensor_alpha, tensor_beta, trans_a, trans_b,
        format_a=format_a, format_b=format_b, dst_dtype=dst_dtype,
        tensor_bias=tensor_bias,
    )

    with tvm.target.cce():
        schedule = generic.auto_schedule(result)

    tensor_list = [tensor_a, tensor_b, tensor_bias,
                   tensor_alpha, tensor_beta, result]

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list,
              }

    te.lang.cce.cce_build_code(schedule, config)
