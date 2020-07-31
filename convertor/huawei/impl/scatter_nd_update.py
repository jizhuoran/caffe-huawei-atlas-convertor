#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

scatter_nd_update
"""
from topi.scatter import Scatter
from topi.cce import util


# pylint: disable=too-many-arguments,unused-argument
@util.check_input_type(dict, dict, dict, dict, bool, str)
def scatter_nd_update(var,
                      indices,
                      updates,
                      var_out,
                      use_locking=False,
                      kernel_name="scatter_nd_update"):
    """
    Update the variable referenced by resource.

    Parameters
    ----------
    var: dict
        data of input.
        source data type, support "int8", "uint8", "float16", "float32"
    indices: dict
         A tensor of indices into var, support "int32"
    updates: dict
        data of updates
        source data type should ne same as var
    var_out: dict
        data of output.
    use_locking: bool
        not used in this compute
    kernel_name: str
        kernel name, default value is "scatter_nd_add"

    Returns:
    None
    """
    scatter_nd = Scatter(var, indices, updates, var_out, True, kernel_name,
                         "update")

    scatter_nd.scatter_operator()
