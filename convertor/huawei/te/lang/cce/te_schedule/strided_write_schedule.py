"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

strided_write
"""
from te import tvm


def strided_write_schedule(res, input_tensors):
    """
    the schedule processes of strided_write

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
