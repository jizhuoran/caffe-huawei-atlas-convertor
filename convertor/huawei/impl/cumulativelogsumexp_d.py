"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

cumulative_logsumexp
"""
from topi.cce import util
from impl.cum_computer import get_computer_by_ctype
from te.utils.op_utils import check_op_params, check_dtype, \
    REQUIRED_INPUT, OPTION_OUTPUT, REQUIRED_ATTR_INT,\
    OPTION_ATTR_BOOL, KERNEL_NAME

# the computer type
COMPUTE_TYPE = "logsumexp"


# pylint: disable=locally-disabled, unused-argument,invalid-name
# pylint: disable=locally-disabled, too-many-arguments, not-callable
@check_op_params(REQUIRED_INPUT, OPTION_OUTPUT, REQUIRED_ATTR_INT,
                 OPTION_ATTR_BOOL, OPTION_ATTR_BOOL, KERNEL_NAME)
def cumulative_logsumexp_d(x, y, axis, exclusive=False, reverse=False,
                           kernel_name="cumulative_logsumexp_d"):
    """
    Compute the cumulative logsumexp of the input tensor along `axis`.

    Parameters
    ----------
    x: dict, shape and dtype, must be in ('float16', 'float32', 'float64')
    y: the dict of output
    axis: a number of int32 or int 64, cumulative axis, must be in the range
    [-rank(x), rank(x))
    exclusive: if `True`, perform exclusive cumsum
    reverse: a `bool` (default: False)
    kernel_name: kernel name

    Returns
    -------
    tik_instance: tik_instance

    """
    shape = x.get("shape")
    if axis < 0:
        axis = len(shape) + axis
    check_param(x, axis, kernel_name)

    cumlogsumexp_template = get_computer_by_ctype(
        x, axis, kernel_name, COMPUTE_TYPE)
    cumlogsumexp_template.set_ext_params(exclusive, reverse)

    return cumlogsumexp_template.get_tik_instance()


def check_param(input_x, axis, kernel_name):
    """
    check the parameters is valid, if one is invalid,then raise error

    Parameters
    ----------
    input_x: dict,shape and datatype
    axis: cumulative axis
    kernel_name: kernel_name
    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_shape)
    util.check_tensor_shape_size(input_shape)
    check_dtype(input_dtype, ("float16", "float32"))

    if axis < len(input_shape) * (-1) or axis >= len(input_shape):
        raise RuntimeError("axis must be in the range [%d, %d). but is %d "
                           % (len(input_shape) * (-1), len(input_shape), axis))
