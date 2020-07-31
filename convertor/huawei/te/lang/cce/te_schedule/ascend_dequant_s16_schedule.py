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

ascend_dequant_s16
"""
from te import tvm


def ascend_dequant_s16_schedule(res, input_tensors):
    """
    the schedule processes of dequant_s16

    Parameters
    ----------
    res: the placeholder of result
    input_tensors: the placeholder of input

    Returns
    -------
    the result of schedule
    """
    sch = tvm.create_schedule(res.op)
    return sch
