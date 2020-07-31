#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

data_format_dim_map

  Op_description :
    Returns the dimension index in the destination data format given the one in.

    # data_format_dim_map(
    #   x,
    #   y,
    #   src_format,
    #   dst_format,
    #   kernel_name='data_format_dim_map')

  Supportive_dtype_format :
    ['int32']
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : `x` Must be in the range [-4, 4).
    [2] All : `src_format` and `dst_format` must be length of 4.
    [3] All : shape size limit is 2147483648.
"""

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import refine_shape_axes
import topi
from topi.cce import util

# mod rhs
MOD_RHS = 4
# quarter
QUARTER = 0.25


def _data_format_dim_map_mod(data):
    """
    mod function based on TF

    Parameters
    ----------
    data: the shape of input, only support int32.

    Returns
    -------
    data mod by 4
    """

    data = te.lang.cce.cast_to(data, 'float16')
    data = te.lang.cce.vadds(data, MOD_RHS)
    data_div_4 = te.lang.cce.vmuls(data, QUARTER)
    data_floor = te.lang.cce.floor(data_div_4)
    data_floor = te.lang.cce.cast_to(data_floor, 'float16')
    data_mul_4 = te.lang.cce.vmuls(data_floor, MOD_RHS)
    data_mod = te.lang.cce.vsub(data, data_mul_4)
    return data_mod


def _dimension_index(data_mod, ind):
    """
    dimension index function

    Parameters
    ----------
    data_mod: the data after modulo
    ind: mapping of index

    Returns
    -------
    dimension index
    """

    is_zero = te.lang.cce.vcmp(data_mod, 0., 'eq')
    is_one = te.lang.cce.vcmp(data_mod, 1., 'eq')
    is_two = te.lang.cce.vcmp(data_mod, 2., 'eq')
    return te.lang.cce.cast_to(te.lang.cce.vsel(is_zero, ind[0], \
                               te.lang.cce.vsel(is_one, ind[1], \
                               te.lang.cce.vsel(is_two, ind[2], ind[3]))), \
                               "int32")


# pylint: disable=locally-disabled,unused-argument,invalid-name
@fusion_manager.register("data_format_dim_map")
def _data_format_dim_map_compute(x,
                                 y,
                                 src_format='NHWC',
                                 dst_format='NCHW',
                                 kernel_name='data_format_dim_map'):
    """
    Parameters
    ----------
    x: the dict of input, only support int32.
    y : the dict of y, reserved parameter, not used now.
    src_format : the original type of x, default value is "NHWC" (optional).
    dst_format : the aim type of x, default value is "NCHW" (optional).
    kernel_name : cce kernel name, default value is "data_format_dim_map" (optional).

    Returns
    -------
    Tensor after dataformatdimmap compute
    """

    data_mod = _data_format_dim_map_mod(x)
    src_format = src_format.upper()
    dst_format = dst_format.upper()

    ind = [0] * len(src_format)
    for i, src in enumerate(src_format):
        for j, dst in enumerate(dst_format):
            if src == dst:
                ind[i] = j
                break

    return _dimension_index(data_mod, ind)


@util.check_input_type(dict, dict, str, str, str)
def data_format_dim_map(x,
                        y,
                        src_format="NHWC",
                        dst_format="NCHW",
                        kernel_name="data_format_dim_map"):
    """
    Returns the dimension index in the destination data format given the one in.

    Parameters
    ----------
    x : A Tensor with each element as a dimension index in source data format.
        Must be the following types: `int32`. Must be in the range [-4, 4).
    y : Shape and dtype of y, reserved parameter, not used now.
    src_format : An optional `string`. Defaults to `"NHWC"`. source data format.
    dst_format : An optional `string`. Defaults to `"NCHW"`. destination data format.
    kernel_name : CCE kernel name, default value is "data_format_dim_map" (optional).

    Returns
    -------
    None
    """

    shape_input = x.get("shape")
    dtype_input = x.get("dtype")

    # check kernel name, shape, size, dtype
    util.check_kernel_name(kernel_name)
    check_shape(shape_input)
    shape_input, _ = refine_shape_axes(shape_input, [])
    check_list = ("int32", )
    dtype_input = dtype_input.lower()
    check_dtype(dtype_input, check_list)

    # check length of format
    if len(src_format) != 4:
        raise ValueError(
            "source format must of length 4, received src_format = %s" %
            src_format)

    if len(dst_format) != 4:
        raise ValueError(
            "destination format must of length 4, received dst_format = %s" %
            dst_format)
    # get data and compute
    data_input = tvm.placeholder(shape_input,
                                 dtype=dtype_input,
                                 name="data_input")
    res = _data_format_dim_map_compute(data_input, y, src_format, dst_format,
                                       kernel_name)

    with tvm.target.cce():
        sch = topi.generic.auto_schedule(res)
    config = {
        "name": kernel_name,
        "print_ir": False,
        "tensor_list": (data_input, res),
        "bool_storage_as_1bit": False
    }
    te.lang.cce.cce_build_code(sch, config)
