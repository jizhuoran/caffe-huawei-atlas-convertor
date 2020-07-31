# /usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

NPUAllocFloatStatus
"""

from te import platform as tbe_platform
from topi.cce import util
from te import tik

#constant 8
NUM_EIGHT = 8

# pylint: disable=invalid-name, too-many-locals, unused-argument
@util.check_input_type(dict, str)
def n_p_u_alloc_float_status(data, kernel_name="n_p_u_alloc_float_status"):
    """
    the main function of n_p_u_alloc_float_status

    Parameters
    ----------
    data: dict,shape and datatype,datatype supports float32
    kernel_name: cce kernel name, default value is "n_p_u_alloc_float_status"

    Returns
    -------
    tik_instance: tik_instance
    """
    tik_instance = tik.Tik()
    data_output = tik_instance.Tensor("float32", (NUM_EIGHT,),
                                      name="data_output", scope=tik.scope_gm)
    data_ub = tik_instance.Tensor("float32", (NUM_EIGHT,), name="data_ub",
                                  scope=tik.scope_ubuf)
    tik_instance.vector_dup(NUM_EIGHT, data_ub, 0, 1, 1, 1)
    tik_instance.data_move(data_output, data_ub, 0, 1, 1, 0, 0)
    tik_instance.BuildCCE(kernel_name, inputs=[], outputs=[data_output])
    return tik_instance
