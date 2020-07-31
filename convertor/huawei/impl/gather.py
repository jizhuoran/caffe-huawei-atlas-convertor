#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

gather
"""
from topi.cce import util
from impl.gather_v2_d import gather_v2_d

# pylint: disable=locally-disabled,unused-argument,invalid-name
@util.check_input_type(dict, dict, dict, bool, str)
def gather(x, indices, y, validate_indices=True, kernel_name="gather"):
    """Gather slices from `params` according to `indices`.`indices` must be an
    integertensor of any dimension (usually 0-D or 1-D).Produces an output
    tensor with shape `indices.shape + params.shape[1:]`.

    Parameters
    ----------
    x: dict
        dict with keys(shape and dtype) of x
    indices: dict
        dict with keys(shape and dtype) of indices
    y: dict
        dict with keys(shape and dtype) of output
    validate_indices: bool
        An optional `bool`. Defaults to `True`
    kernel_name: str
        kernel name, default value is "pow"

    Returns
    -------
    None
    """
    x_shape = x.get("shape")
    x_dtype = x.get("dtype").lower()
    indices_shape = indices.get("shape")
    indices_dtype = indices.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(x_shape)
    util.check_tensor_shape_size(x_shape)
    util.check_shape_rule(indices_shape)
    util.check_tensor_shape_size(indices_shape)
    dtype_list = ("int8", "int16", "int32", "int64", "uint8",
                  "uint16", "uint32", "uint64", "float16", "float32")
    util.check_dtype_rule(indices_dtype, ("int32", "int64"))
    util.check_dtype_rule(x_dtype, dtype_list)

    gather_v2_d(x, indices, y, 0, kernel_name)
