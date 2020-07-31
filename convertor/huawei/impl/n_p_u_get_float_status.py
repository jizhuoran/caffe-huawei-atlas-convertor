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

GetFloatStatus
"""

from te import platform as tbe_platform
from topi.cce import util
from te import tik

#constant 8
NUM_EIGHT = 8

# pylint:disable=invalid-name,too-many-locals,unused-argument,unused-variable
@util.check_input_type(dict, dict, str)
def n_p_u_get_float_status(addr, data, kernel_name="n_p_u_get_float_status"):
    """
    the main function of n_p_u_get_float_status

    Parameters
    ----------
    addr: dict,shape and datatype,datatype supports float32
    data: dict,shape and datatype,datatype supports float32
    kernel_name: cce kernel name, default value is "n_p_u_get_float_status"

    Returns
    -------
    tik_instance: tik_instance
    """
    tik_instance = tik.Tik()
    aicore_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

    input_addr = tik_instance.Tensor("float32", (NUM_EIGHT,), name="input_addr",
                                     scope=tik.scope_gm)
    output_data = tik_instance.Tensor("float32", (NUM_EIGHT,),
                                      name="output_data",
                                      scope=tik.scope_gm)

    with tik_instance.for_range(0, aicore_num, block_num=aicore_num) as cycle:
        status_in_scalar_buffer = tik_instance.Scalar(dtype="uint64",
                                                      init_value=0)
        tik_instance.get_overflow_status(status_in_scalar_buffer)
        data_ub = tik_instance.Tensor("float32", (NUM_EIGHT,), name="data_ub",
                                      scope=tik.scope_ubuf)
        tik_instance.vector_dup(NUM_EIGHT, data_ub, 1, 1, 0, 0)
        data_ub_input = tik_instance.Tensor("float16", (38400,),
                                            name="data_ub_input",
                                            scope=tik.scope_ubuf)
        tik_instance.vector_dup(128, data_ub_input, 3, 150, 8, 1)
        tik_instance.vector_dup(128, data_ub_input, 4, 150, 8, 1)
        tik_instance.vector_dup(128, data_ub_input, 5, 150, 8, 1)
        tik_instance.vector_dup(128, data_ub_input, 6, 150, 8, 1)
        tik_instance.vector_dup(128, data_ub_input, 7, 150, 8, 1)
        tik_instance.vector_dup(128, data_ub_input, 8, 150, 8, 1)
        tik_instance.vector_dup(128, data_ub_input, 9, 150, 8, 1)
        with tik_instance.if_scope(status_in_scalar_buffer == 1):
            tik_instance.data_move(input_addr, data_ub, 0, 1, 1, 0, 0)
        tik_instance.vector_dup(NUM_EIGHT, data_ub, 0, 1, 0, 0)
        tik_instance.data_move(output_data, data_ub, 0, 1, 1, 0, 0)
    tik_instance.BuildCCE(kernel_name, inputs=[input_addr],
                          outputs=[output_data])
    return tik_instance
