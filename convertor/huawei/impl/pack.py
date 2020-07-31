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

pack
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from impl.concat_v2_d import concat_v2_d


# pylint: disable = locally-disabled,invalid-name,too-many-arguments
# pylint: disable = unused-argument,no-member

@util.check_input_type((list, tuple), dict, int, str)
def check_supported(x, y, axis, kernel_name="pack"):
    """
    support aicpu route
    """
    if axis == -1 or axis == len((x[0].get("shape"))):
        return False
    return True


def pack(x, y, axis, kernel_name="pack"):
    """
    algorithm: pack
    Concatenates tensors along one dimension.
    Parameters
    ----------
    x : A list of `dict`.dict include keys shape and dtype
    y: dict of output_data, dict include keys shape and dtype
    axis : int, in the range [-rank(values), rank(values)
    kernel_name : cce kernel name, default value is "pack"
    Returns
    -------
    None
    """
    check_list = ("int8", "int16", "int32", "int64", "uint8",
                  "uint16", "uint32", "uint64", "float16", "float32")
    data = []
    for i, input_dict in enumerate(x):
        shape_input = input_dict.get("shape")
        util.check_shape_rule(shape_input)
        util.check_tensor_shape_size(shape_input)
        util.check_dtype_rule(input_dict.get("dtype").lower(), check_list)
        input_dtype = (input_dict.get("dtype")).lower()
        data.append(tvm.placeholder(shape_input, name="data_%d" % i,
                                    dtype=input_dtype))
    util.check_kernel_name(kernel_name)

    if axis < -len((x[0].get("shape")))-1 or axis > len((x[0].get("shape"))):
        raise RuntimeError(
            "pack axis must be in [-%d , %d), "
            "actual is %d" % (len(x[0].get("shape"))+1,
                              len(x[0].get("shape"))+1, axis))

    if axis < -1:
        axis = axis + 1
    concat_v2_d(x, y, axis, kernel_name)
