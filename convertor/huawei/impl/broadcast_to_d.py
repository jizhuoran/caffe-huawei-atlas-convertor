#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

broadcast_to_d

  Op_description :
    Broadcast an array for a compatible shape.

    # broadcast_to_d(
    #   x,
    #   y,
    #   shape,
    #   kernel_name='broadcast_to_d')

  Supportive_dtype_format :
    ['float32', 'float16', 'int8', 'uint8', 'int32']
    ['ND', 'NCHW', 'NHWC']

  Constraint :
    [1] All : `shape` must be an 1-D 'int' tensor.
    [2] All : shape size limit is 2147483648.
"""

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from topi import generic
from topi.cce import util

NUM_ONE = 1


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("broadcast_to_d")
def broadcast_to_compute(x,
                         y,
                         shape,
                         kernel_name='broadcast_to_d'):
    """
    Process broadcast_to operator.

    Parameters:
    ----------
    x : the input tensor.

    y : the dict of output.

    shape : the desired output shape.

    kernel_name : cce kernel name, default value is "broadcast_to_d".

    Returns:
    -------
    output_tensor : tensor after broadcast_to.
    """

    dtype = x.dtype
    shape_in = x.shape

    # te.lang.cce.broadcast supports float16, float32, int32.
    # so convert int8, uint8 to float16
    if dtype in ('int8', 'uint8'):
        x = te.lang.cce.cast_to(x, 'float16')

    python_shape_in = [int(x) for x in shape_in]
    if list(python_shape_in) == list(shape):
        if dtype == "int32":
            # te.lang.cce.vmuls supports float16, float32. int8, uint8, int32 will
            # be converted to float16. This will cause the data to be truncated.
            # so use te.lang.cce.vmul.
            value_one = tvm.const(NUM_ONE, dtype=dtype)
            value_one_tensor = te.lang.cce.broadcast(value_one, shape)
            output_tensor = te.lang.cce.vmul(x, value_one_tensor)
        else:
            output_tensor = te.lang.cce.vmuls(x, NUM_ONE)
    else:
        output_tensor = te.lang.cce.broadcast(x, shape, dtype)

    # convert float16 back to int8, uint8
    if dtype in ('int8', 'uint8'):
        return te.lang.cce.cast_to(output_tensor, dtype, f1628IntegerFlag=True)

    return output_tensor


def _check_shape_compatibility(shape_in, shape):
    """
    Check if the shape of input tensor is compatible with output tensor.

    Parameters:
    ----------
    shape_in : shape of input tensor.

    shape : shape of output tensor.

    Returns:
    -------
    comp_shape_in : new shape_in compatible with shape.
    """

    try:
        comp_shape_in, comp_shape, shape_max = broadcast_shapes(
            shape_in, shape)
        if comp_shape != shape_max:
            raise ValueError('shape_in is not compatible with shape_out.')
    except RuntimeError:
        raise ValueError('shape_in is not compatible with shape_out.')

    return comp_shape_in


@util.check_input_type(dict, dict, (list, tuple), str)
def broadcast_to_d(x,
                   y,
                   shape,
                   kernel_name="broadcast_to_d"):
    """
    Broadcast an array for a compatible shape.

    Parameters:
    ----------
    x : the dict of input. support data type: float32, float16, int8, uint8, int32.

    y : the dict of output.

    shape : shape of output tensor.

    kernel_name : cce kernel name, default value is "broadcast_to_d".

    Returns:
    -------
    None
    """

    util.check_kernel_name(kernel_name)

    check_list = ('float32', 'float16', 'int8', 'uint8', 'int32')
    dtype = x.get('dtype')
    check_dtype(dtype, check_list)
    inp_dtype = dtype.lower()

    shape_in = x.get('shape')
    check_shape(shape_in)
    check_shape(shape)

    compatible_shape_in = _check_shape_compatibility(shape_in, shape)

    var = tvm.placeholder(compatible_shape_in, inp_dtype, name='data_input')

    with tvm.target.cce():
        res = broadcast_to_compute(var, y, shape, kernel_name)

        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [var, res]}
    te.lang.cce.cce_build_code(sch, config)
