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

tensor_scatter_add
"""
from topi.tensor_scatter import TensorScatter
from te.utils.op_utils import *

# pylint: disable=too-many-arguments,invalid-name
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_OUTPUT, KERNEL_NAME)
def tensor_scatter_add(x,
                       indices,
                       updates,
                       y,
                       kernel_name="tensor_scatter_add"):
    """
    Applies sparse addition to individual values or slices in a Variable.

    Parameters
    ----------
    x: dict
        data of input.
        source data type, support "int8", "uint8", "int32", "float16", "float32"
    indices: dict
         A tensor of indices into var, support "int32"
    updates: dict
        data of updates
        source data type should ne same as var
    y: dict
        data of output.
    kernel_name: str
        kernel name, default value is "tensor_scatter_add"

    Returns:
    None
    """
    tensor_scatter = TensorScatter(x, indices, updates, y, kernel_name, "vadd")

    tensor_scatter.tensor_scatter_operator()
