"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0


CCE configuration constants
"""

from __future__ import absolute_import as _abs
from te import tvm


def cc_device_exp(input_data, p_scale, p_shift, p_base, p_shape):
    """Take exp of input_data by cc_device_api.(DeviceExp)

    Parameters
    ----------
    input_data: Tensor
        Input argument.

    p_scale: ptr of Tensor
        default [1]

    p_shift: ptr of Tensor
        default [0]

    p_base: ptr of Tensor
        default [-1]

    p_shape: ptr of Tensor
        default [1]

    Returns
    -------
    y : Expr
        The result.
    """
    return tvm.call_pure_intrin(input_data.dtype, "DeviceExp", input_data, p_scale, p_shift,
                                p_base, p_shape)


def cc_device_log(input_data, p_scale, p_shift, p_base, p_shape):
    """Take log of input_data by cc_device_api(DeviceLog).

    Parameters
    ----------
    input_data: Tensor
        Input argument.

    p_scale: ptr of Tensor
        default [1]

    p_shift: ptr of Tensor
        default [0]

    p_base: ptr of Tensor
        default [-1]

    p_shape: ptr of Tensor
        default [1]

    Returns
    -------
    y : Expr
        The result.
    """
    return tvm.call_pure_intrin(input_data.dtype, "DeviceLog", input_data, p_scale, p_shift,
                                p_base, p_shape)
