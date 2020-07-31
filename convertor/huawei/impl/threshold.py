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

threshold
"""

from functools import reduce as reduceIns
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi.generic import cce
from topi.cce import util
from te import platform as tbe_platform

# define a scalar , value = 1.0
SCALAR_POSITIVE_ONE = 1.0
# define a scalar , value = 0.0
SCALAR_ZERO = 0.0



@fusion_manager.register("threshold")
def threshold_compute(input_x, input_y, output_y, kernel_name="threshold"):
    """
    compare data with threshold,x > threshold ? 1; 0

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    input_y: dict
        shape and dtype of output, should be broadcast shape and type as input
    output_y: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is threshold
    Returns
    -------
    res: if data is bigger than threshold return 1,else return 0
    """
    # define a scalar , value = 10000000.0
    SCALAR_TEN_MILLION = 10000000.0
    # switch zoom value
    input_dtype = input_x.dtype
    if input_dtype == "float16":
        SCALAR_TEN_MILLION = 10000.0

    sub_threshold = te.lang.cce.vadds(input_x, -input_y)
    zero_data = te.lang.cce.vmaxs(sub_threshold, SCALAR_ZERO)
    one_data = te.lang.cce.vmins(zero_data, SCALAR_POSITIVE_ONE)

    zoom_data_tmp = te.lang.cce.vmuls(one_data, SCALAR_TEN_MILLION)
    res_tmp = te.lang.cce.vmins(zoom_data_tmp, SCALAR_POSITIVE_ONE)

    zoom_data = te.lang.cce.vmuls(res_tmp, SCALAR_TEN_MILLION)
    res = te.lang.cce.vmins(zoom_data, SCALAR_POSITIVE_ONE)

    return res


@util.check_input_type(dict, dict, float, str)
def threshold(input_x, output_y, threshold=0.0, kernel_name="threshold"):
    """
    algorithm: threshold
    compare data with threshold: x > threshold ? 1; 0

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32, int32
    output_y: dict
        shape and dtype of output, should be broadcast shape and type as input
    threshold: scalar
        parameter of the operator
    kernel_name : str
        kernel name, default value is "threshold"

    Returns
    -------
    None
    """

    # check shape
    shape = input_x.get("shape")
    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)

    # check kernel_name
    util.check_kernel_name(kernel_name)

    # check data type
    check_tuple = ("float16", "float32")
    input_data_type = input_x.get("dtype").lower()
    util.check_dtype_rule(input_data_type, check_tuple)

    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x * y, shape)
    data_x = tvm.placeholder(fuseshape, name="data_x", dtype=input_data_type)
    res = threshold_compute(data_x, threshold, output_y, kernel_name)

    with tvm.target.cce():
        schedule = cce.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "need_build": False,
              "tensor_list": (data_x, res)}
    te.lang.cce.cce_build_code(schedule, config)
