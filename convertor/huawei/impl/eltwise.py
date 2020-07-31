#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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

eltwise
"""
from functools import reduce as reduceIns

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

SHAPE_SIZE_LIMIT = 2147483648

# pylint: disable=unidiomatic-typecheck,too-many-branches,too-many-locals
# pylint: disable=no-member,dangerous-default-value,invalid-name
@fusion_manager.register("eltwise")
def eltwise_compute(x, y, mode=1, coeff=[], kernel_name="eltwise"):
    '''
    Compute elementwise operation
    '''
    tensor_num = len(x)
    inp_dtype = x[0].dtype
    data0_tmp = x[0]

    if mode == 1:
        if len(coeff) != 0 and len(coeff) != tensor_num:
            raise RuntimeError("lenth of coeff must equal to tensor_num or 0.")
        if len(coeff) == tensor_num:
            if type(coeff[0]) != int and type(coeff[0]) != float:
                raise RuntimeError("ele of coeff must be a number.")
            if coeff[0] != 1:
                coeff1 = tvm.const(coeff[0], dtype=inp_dtype)
                data0_tmp = te.lang.cce.vmuls(data0_tmp, coeff1)

    if tensor_num == 1:
        const_val_0 = tvm.const(0, dtype=inp_dtype)
        data0_tmp = te.lang.cce.vadds(data0_tmp, const_val_0)
        res = data0_tmp
    elif tensor_num > 1:
        for i in range(1, tensor_num):
            datan_tmp = x[i]
            if mode == 0:
                data0_tmp = te.lang.cce.vmul(data0_tmp, datan_tmp)
            elif mode == 2:
                data0_tmp = te.lang.cce.vmax(data0_tmp, datan_tmp)
            else:
                if len(coeff) == 0:
                    data0_tmp = te.lang.cce.vadd(data0_tmp, datan_tmp)
                elif coeff[i] == 1:
                    data0_tmp = te.lang.cce.vadd(data0_tmp, datan_tmp)
                else:
                    coeff2 = tvm.const(coeff[i], dtype=inp_dtype)
                    datan_tmp = te.lang.cce.vmuls(datan_tmp, coeff2)
                    data0_tmp = te.lang.cce.vadd(data0_tmp, datan_tmp)
        res = data0_tmp
    return res


def _eltwise_check_para(x, y, mode=1, coeff=[],
                        kernel_name="eltwise"):

    util.check_kernel_name(kernel_name)

    shape = x[0].get("shape")
    dtype = x[0].get("dtype").lower()

    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)

    dtype_check_list = ["float16", "float32"]
    if not dtype in dtype_check_list:
        raise RuntimeError("dtype is not support,"
                           "only support fp16 and fp32.")

    tensor_num = len(x)
    if tensor_num < 1 or tensor_num > 32:
        raise RuntimeError("tensor_num need in range [1,32].")

    # all input data should be same shape and dtype
    if tensor_num > 1:
        for i in range(1, tensor_num):

            shape_tmp = x[i].get("shape")
            dtype_tmp = x[i].get("dtype").lower()

            if shape_tmp != shape:
                raise RuntimeError("input shape is not same.")

            if dtype_tmp != dtype:
                raise RuntimeError("input dtype is not same.")

    shape_output = y.get("shape")
    util.check_shape_rule(shape_output)
    util.check_shape_size(shape_output, SHAPE_SIZE_LIMIT)
    if shape_output != shape:
        raise RuntimeError("output shape is not same with input.")

    dtype_output = y.get("dtype").lower()
    if dtype_output != dtype:
        raise RuntimeError("output dtype is not same with input.")

    #mode type must be 0, 1 or 2
    op_list = (0, 1, 2)
    if mode not in op_list:
        raise RuntimeError("mode is wrong, only suppor 0, 1, 2.")


@util.check_input_type((list, tuple), dict, int, (list, tuple), str)
def eltwise(x, y, mode=1, coeff=[], kernel_name="eltwise"):
    """
    Compute elementwise modes, such as 0:PRODUCT, 1:SUM and 2:MAX

    Parameters
    ----------
    x : the list of input data, it's element is dict:{"shape":[], "dtype":""}

    y : the dict of output

    mode : 0:product,1:sum,2:max;default is 1:sum.

    coeff : input_num should be equal with coeff size.

    kernel_name : cce kernel name, default value is "eltwise"

    Returns
    -------
    None

    """

    _eltwise_check_para(x, y, mode=mode,
                        coeff=coeff, kernel_name=kernel_name)
    shape = x[0].get("shape")
    dtype = x[0].get("dtype").lower()
    tensor_num = len(x)

    shape = util.shape_refine(shape)
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape)

    tlist = []
    with tvm.target.cce():
        for i in range(0, tensor_num):
            datan_name = 'data%d' % i
            datan_tmp = tvm.placeholder(fuseshape, name=datan_name, dtype=dtype)
            tlist.append(datan_tmp)

        y_datan_name = 'y_data'
        data_y_tmp = tvm.placeholder(fuseshape, name=y_datan_name, dtype=dtype)

        res = eltwise_compute(tlist, data_y_tmp,
                              mode, coeff, kernel_name)
        sch = generic.auto_schedule(res)
    tlist.append(res)

    config = {"print_ir": False,
              "need_build": False,
              "name": kernel_name,
              "tensor_list": tlist}
    te.lang.cce.cce_build_code(sch, config)
