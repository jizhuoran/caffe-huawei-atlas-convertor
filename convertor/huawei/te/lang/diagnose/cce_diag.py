#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use 
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

cce diag function:we can use this function to build and generate llvm code on cpu
"""
import numpy as np
from te import tvm
from topi.util import get_const_tuple



def cce_diag(config_map=None):
    """
    API of building or printing lower code, just can be used when device is CCE

    Parameters
    ----------
    config_map : dict, default is {} and use default configration

        key_words:

            print_ir : if need print lower IR code, default is True

            need_build : if need build, default is True

            name : kernel name, default is cce_op

            ref_input_func : the python reference input data function

            ref_output_func : the python reference output data  function

            rtol : the relative tolerance

    Returns
    -------
    None
    """
    # for pylint, otherwise "Dangerous default value {} as argument"
    if config_map is None:
        config_map = {}

    def _build(sch, tensor_list, kernel_name='cce_diag_op'):
        target = 'llvm'
        mod = tvm.build(sch, tensor_list, target, name=kernel_name)
        return mod

    def _lower(sch, tensor_list):
        print(tvm.lower(sch, tensor_list, simple_mode=True))

    def _check(mod, input_tensors, output_tensors, ref_input_func, ref_output_func, rtol):
        ctx = tvm.context('llvm', 0)
        if not ctx.exist:
            raise RuntimeError('Only support diagnose on CPU now')
        np_inputs = []
        tvm_inputs = []
        tvm_outputs = []
        if ref_input_func is None:
            for tensor in input_tensors:
                np_inputs.append(
                    np.random.uniform(size=get_const_tuple(tensor.shape)).astype(tensor.dtype))
        else:
            np_inputs = ref_input_func(*input_tensors)
        for np_input in np_inputs:
            tvm_inputs.append(tvm.nd.array(np_input, ctx))

        for tensor in output_tensors:
            tvm_outputs.append(
                tvm.nd.array(np.zeros(get_const_tuple(tensor.shape), dtype=tensor.dtype), ctx))

        mod(*(tvm_inputs + tvm_outputs))
        if ref_output_func is None:
            print([output.asnumpy() for output in tvm_outputs])
        else:
            np_output = ref_output_func(*np_inputs)
            if rtol is None:
                np.testing.assert_allclose(tvm_outputs[0].asnumpy(), np_output, rtol=1e-3)
            else:
                np.testing.assert_allclose(tvm_outputs[0].asnumpy(), np_output, rtol=rtol)
            print('CHECK PASS')

    input_tensors = config_map["input_tensors"]
    output_tensors = config_map["output_tensors"]
    ref_input_func = config_map["ref_input_func"]
    ref_output_func = config_map["ref_output_func"]
    rtol = config_map["rtol"]

    sch = tvm.create_schedule([x.op for x in output_tensors])
    if config_map["print_ir"]:
        _lower(sch, input_tensors + output_tensors)

    mod = _build(sch, input_tensors + output_tensors, config_map["name"])

    _check(mod, input_tensors, output_tensors, ref_input_func, ref_output_func, rtol)
