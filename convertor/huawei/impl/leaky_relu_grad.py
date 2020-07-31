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

leaky_relu_grad
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import refine_shapes_for_broadcast
from topi import generic
from topi.cce import util

# define a scalar , value = 0
SCALAR_ZERO = 0
# define a scalar , value = -1
NEGATIVE_ONE = -1

# pylint: disable=unused-argument,invalid-name,too-many-locals
@fusion_manager.register("leaky_relu_grad")
def leaky_relu_grad_compute(g, x, y, negative_slope=0,
                            kernel_name="leaky_relu_grad"):
    """
    calculate the backpropagation of leaky_relu operation
    y = gradients(x>0) or negative_slope*gradients(x<=0).

    Parameters
    ----------
    g : TVM tensor
        the placeholder of input g
    x : TVM tensor
        the placeholder of input x
    y : dict
        dict of output y, include keys(shape and dtype)
    negative_slope : float or int
        allow non-zero slope for negative inputs to speed up optimization
    kernel_name : str
        kernel name, default value is "leaky_relu_grad"

    Returns
    -------
    res: TVM tensor
        the result of leaky_relu_grad_compute
    """

    shape_list = util.produce_shapes(
        te.lang.cce.util.shape_to_list(g.shape),
        te.lang.cce.util.shape_to_list(x.shape))
    util.check_tensor_shape_size(shape_list[2])

    dtype = g.dtype
    g = te.lang.cce.broadcast(g, shape_list[2])
    x = te.lang.cce.broadcast(x, shape_list[2])

    if dtype == "float32":
        help_min = tvm.const(2 ** (-126), "float32")
        help_rec_one = tvm.const(2 ** 38, "float32")
        help_rec_sec = tvm.const(2 ** 44, "float32")
    elif dtype == "float16":
        help_min = tvm.const(2 ** (-24), "float16")
        help_rec_one = tvm.const(2 ** 12, "float16")
        help_rec_sec = help_rec_one

    tmp_min_x = te.lang.cce.vmins(x, help_min)
    tmp_max_x = te.lang.cce.vmaxs(tmp_min_x, tvm.const(SCALAR_ZERO, "float32"))
    tmp_mul_x = te.lang.cce.vmuls(tmp_max_x, help_rec_one)

    if dtype == "float32":
        tmp_mul_x = te.lang.cce.vmuls(tmp_mul_x, help_rec_sec)

    result_tmp_right = te.lang.cce.vmuls(tmp_mul_x, help_rec_sec)

    result_sub = te.lang.cce.vadds(result_tmp_right, tvm.const(NEGATIVE_ONE,
                                                               "float32"))
    result_abs = te.lang.cce.vabs(result_sub)
    result_tmp_left = te.lang.cce.vmuls(result_abs, negative_slope)

    result_tmp = te.lang.cce.vadd(result_tmp_left, result_tmp_right)

    res = te.lang.cce.vmul(g, result_tmp)
    return res

@util.check_input_type(dict, dict, dict, (int, float), str)
def leaky_relu_grad(g, x, y, negative_slope=0, kernel_name="leaky_relu_grad"):
    """
    calculate the backpropagation of leaky_relu operation
    y = gradients(x>0) or negative_slope*gradients(x<=0).
    support dtype:float16,float32

    Parameters
    ----------
    g : dict
        the backpropagated gradients to the corresponding leaky_relu operation
    x : dict
        the x passed as output of leaky_relu operation
    y : dict
        the output of leaky_relu back propagation
    negative_slope : float or int
        allow non-zero slope for negative inputs to speed up optimization
    kernel_name : str
        kernel name, default value is "leaky_relu_grad"

    Returns
    -------
    None
    """

    shape_g = g.get("shape")
    shape_x = x.get("shape")
    dtype_g = g.get("dtype").lower()
    dtype_x = x.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_g)
    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_g)
    util.check_tensor_shape_size(shape_x)

    shape_list = util.produce_shapes(shape_g, shape_x)
    util.check_tensor_shape_size(shape_list[2])

    # check input tensor data_type
    check_list = ["float16", "float32"]
    util.check_dtype_rule(dtype_g, check_list)
    util.check_dtype_rule(dtype_x, check_list)
    util.compare_tensor_dict_key(g, x, "dtype")

    shape_g, shape_x = refine_shapes_for_broadcast(shape_list[0],
                                                   shape_list[1])
    data_g = tvm.placeholder(shape_g, name="data_g", dtype=dtype_g)
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_g)
    res = leaky_relu_grad_compute(data_g, data_x, y,
                                  negative_slope, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_g, data_x, res]}

    te.lang.cce.cce_build_code(schedule, config)
