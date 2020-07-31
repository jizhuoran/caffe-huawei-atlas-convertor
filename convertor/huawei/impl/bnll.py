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

bnll
  Oo_description :
  do element-wise bnll operation

  bnll(x, y, kernel_name = "bnll")

  Supportive_dtype_format :
   ["float16", "float32"]
   ["ND"]

"""
# pylint: disable=E0401
# pylint: disable=C0412
# pylint: disable=W0613

from functools import reduce as reduceIns
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform

CONST_ZERO = 0.0
CONST_ONE = 1.0
CONST_NEGATIVE_ONE = -1.0


def _bnll_compute(data, dtype):

    scalar_zero = tvm.const(CONST_ZERO, dtype)

    negative_data = te.lang.cce.vmins(data, scalar_zero)
    positive_data = te.lang.cce.vmaxs(data, scalar_zero)

    data_reverse = te.lang.cce.vaxpy(positive_data, negative_data, tvm.const(CONST_NEGATIVE_ONE, dtype))

    res = te.lang.cce.vexp(data_reverse)
    res = te.lang.cce.vadds(res, tvm.const(CONST_ONE, dtype))
    res = te.lang.cce.vlog(res)
    res = te.lang.cce.vadd(res, positive_data)

    return res


@fusion_manager.register("bnll")
def _bnll_computer(input_x, product):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "bnll"

    Returns
    -------
    output tensor
    """
    dtype = input_x.dtype

    if dtype == "float16" and product not in ("Hi3796CV300ES"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
        d_dtype = "float32"
    else:
        d_dtype = "float16"

    res = _bnll_compute(input_x, d_dtype)

    if dtype == "float16" and product not in ("Hi3796CV300ES"):
        res = te.lang.cce.cast_to(res, "float16")

    return res


@util.check_input_type(dict, dict, str)
def bnll(input_x, output_y, kernel_name="bnll"):
    """
    calculating data
    algrithm: y=x+log(1+exp(-x)) if x>0; y=log(1+exp(x)) otherwise

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "bnll"

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    input_dtype = dtype.lower()

    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_kernel_name(kernel_name)

    check_list = ("float16", "float32")
    util.check_dtype_rule(input_dtype, check_list)

    product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if product in ["Ascend310", "Hi3796CV300ES"] and input_dtype == "float32":
        raise RuntimeError(
            "Input dtype only support float16 while input dtype is float32")

    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape)
    data_input = tvm.placeholder(fuseshape, name="data_input", dtype=input_dtype)

    res = _bnll_computer(data_input, product)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "bool_storage_as_1bit": False,
              "tensor_list": [data_input, res]}

    te.lang.cce.cce_build_code(schedule, config)
