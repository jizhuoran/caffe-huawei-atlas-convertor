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

concat_offset_d
"""
from te import platform as tbe_platform
from topi.cce import util
from te import tik

# 256B can store up to 64 numbers when the data is int32 type
NUM64 = 64

# const vlaue 2
VALUE_TWO = 2

# The maximum value of the mask corresponding to 8 blocks
MAX_MASK8 = 255

# pylint: disable=locally-disabled,unused-argument, invalid-name
@util.check_input_type((list, tuple), (list, tuple), int, str)
def concat_offset_d(x, y, concat_dim, kernel_name="concat_offset_d"):
    """
    Compute the concat offset of the input tensor along `concat_dim`.

    Parameters
    ----------
    concat_dim: a number of int32, The dimension along which to concatenate,
                must be in the range [-rank(shape), rank(shape))
    x: list of dict, dict include shape and dtype, dtype must be in ('int32')
    y: list of dict, dict include shape and dtype, dtype must be in ('int32')
    kernel_name: kernel name

    Returns
    -------
    concat_offset_d_compute.compute(): the result of compute

    """
    dict_num = len(x)
    input0_rank = x[0].get("shape")[0]
    if concat_dim < 0:
        concat_dim = input0_rank + concat_dim

    concat_offset_d_check(x, concat_dim, input0_rank, dict_num, kernel_name)
    concat_offset_d_compute = ConcatOffsetDCompute(concat_dim,
                                                   input0_rank, dict_num,
                                                   kernel_name)
    return concat_offset_d_compute.compute()


def concat_offset_d_check(x, concat_dim, input0_rank, dict_num, kernel_name):
    """
    check the parameters is valid, if one is invalid,then raise error
    Parameters
    ----------
    x: list of tensor
    concat_dim: The dimension along which to concatenate,
                must be in the range [-rank(shape), rank(shape))
    input0_rank: The rank of the first input shape in the list
    dict_num: The number of tensor in the list
    kernel_name: kernel_name
    Returns
    -------
    None
    """

    util.check_kernel_name(kernel_name)
    if dict_num < 2:
        raise RuntimeError("The number of elements in the list "
                           "should be no less than two")
    for i in range(0, dict_num):
        shape_input = x[i].get("shape")
        if shape_input[0] != input0_rank:
            raise RuntimeError("input_%d : should contain %d elements,but "
                               "got %d" % (i, input0_rank,
                                           shape_input[0]))
        elif shape_input[0] > 8:
            raise RuntimeError("the shape of input_%d should be not bigger"
                               " than 8" % i)
        elif concat_dim >= shape_input[0]:
            raise RuntimeError("Concat dim is greater or equal to input0_rank,"
                               "the concat_dim is %d, the input0_rank is %d"
                               % (concat_dim, input0_rank))
        elif concat_dim < 0:
            concat_dim = concat_dim - input0_rank
            raise RuntimeError("Concat dim is not less than zero,"
                               "the concat_dim is %d, the input0_rank is %d"
                               % (concat_dim, input0_rank))
        util.check_shape_rule(shape_input, max_dim=1)
        util.check_tensor_shape_size(shape_input)
        util.check_dtype_rule(x[i].get("dtype").lower(), ("int32",))


class ConcatOffsetDCompute(object):
    def __init__(self, concat_dim, input0_rank, dict_num, kernel_name):
        """
        init the input param

        Parameters
        ----------
        input0_rank: The rank of the first input shape in the list
        dict_num: the number of tensor
        kernel_name: kernel name

        """
        self.concat_dim = concat_dim
        self.kernel_name = kernel_name
        self.input0_rank = input0_rank
        self.dict_num = dict_num
        self.dtype = "int32"

    def compute(self):
        """
        describe the concat_offset calculation process

        Returns
        -------
        tik_instance: the instance of tik

        """
        data_input = []
        data_output = []
        tik_instance = tik.Tik()
        cdim_mask = MAX_MASK8 - pow(VALUE_TWO, int(self.concat_dim))
        for i in range(self.dict_num):
            data_input.append(tik_instance.Tensor(
                self.dtype, [self.input0_rank],
                name="".join(["data_input", str(i)]), scope=tik.scope_gm))
            data_output.append(tik_instance.Tensor(
                self.dtype, [self.input0_rank],
                name="".join(["data_output", str(i)]), scope=tik.scope_gm))
        data_row1_ub = tik_instance.Tensor(self.dtype, [NUM64],
                                           name="data_row1_ub",
                                           scope=tik.scope_ubuf)
        data_row2_ub = tik_instance.Tensor(self.dtype, [NUM64],
                                           name="data_row2_ub",
                                           scope=tik.scope_ubuf)
        tik_instance.vector_dup(NUM64, data_row1_ub, 0, 1, 1, 1)

        for m in range(self.dict_num):
            tik_instance.data_move(data_output[m], data_row1_ub, 0, 1, 1, 0, 0)
            tik_instance.data_move(data_row2_ub, data_input[m], 0, 1, 1, 0, 0)
            tik_instance.vadd(8, data_row1_ub, data_row1_ub, data_row2_ub,
                              1, 1, 1, 1, 0, 0, 0)
            tik_instance.vector_dup([0, cdim_mask], data_row1_ub,
                                    0, 1, 0, 0)

        tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=data_input,
                              outputs=data_output)
        return tik_instance
