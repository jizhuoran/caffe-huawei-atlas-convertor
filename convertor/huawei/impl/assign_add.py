#!/usr/bin/env python
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

assignadd
"""

from functools import reduce as function_reduce

from te import platform as cce
from te.platform import insn_cmd
from te import tvm
from te.platform.cce_build import build_config
from topi.cce import util

# shape limit for int64: 1
SHAPE_SIZE_LIMIT_INT64 = 1
# shape limit for the others: 2**31 - 1
SHAPE_SIZE_LIMIT_OTHER = 2147483647
# available ub size
UB_SIZE_B = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE)


def _tilling_axis(shape, dtype):
    """
    calculate the split parameters according to different shapes

    Parameters
    ----------
    shape : list or tuple
        shape of tensor
    dtype : string
        buffer date type

    Returns
    -------
    split_axis : the target axis that is used for spliting the tensor to find
        the maximum amount of data can be stored and processed every time on UB.
    split_factor : the factor used when spliting the target axis.
        For example, for data of float16, [1024, 1024, 256] will be split to
        [1024, 7, 164, 256], UB processes 164*256 elements every time.
        In this case, the split_axis is 1 and the split_factor is 164.
    """
    # ub_size_bytes is the size of the UB expressed by bytes(mod 8 bits).
    ub_size_bytes = UB_SIZE_B - 1*1024
    # dtype_bytes_size for float16 is 2, for float32 is 4
    dtype_bytes_size = cce.cce_intrin.get_bit_len(dtype) // 8
    # total_ele is the maximum amount of data that can be stored in UB.
    if dtype in ("int8", "uint8"):
        dtype_bytes_size_fp16 = cce.cce_intrin.get_bit_len("float16") // 8
        total_ele = ub_size_bytes //\
                    (dtype_bytes_size + dtype_bytes_size_fp16) // 3
    else:
        total_ele = ub_size_bytes // dtype_bytes_size // 3

    # To initialize the split_axis and the split_factor.
    split_axis = 0
    split_factor = 1

    # To find the appropriate axis from the first one to the last
    # by comparing the amount of the elements of the split tensor with
    # the maximum amount of data that can be stored in UB.
    for index, _ in enumerate(shape):
        ele_cnt = function_reduce(lambda x, y: x*y, shape[index:])
        if ele_cnt <= total_ele:
            split_axis = index - 1
            split_factor = total_ele // ele_cnt
            break

    # when the last axis is still over the size of UB, we choose to split the
    # last axis, and the split_factor is set as the maximum amount of data
    # that can be stored in UB. For example, [10, 10, 256000] will be
    # split to [10, 10, 7, 42154]
    if shape[-1] > total_ele:
        split_axis = len(shape) - 1
        split_factor = total_ele

    # when the amount of the elements of the tensor is less than the size of UB,
    # it means UB can process the whole tensor in one time. But the split_axis
    # has already been set to "-1", split_axis and split_factor
    # should be initialized into "0" and shape[0]
    if split_axis < 0:
        split_axis = 0
        split_factor = shape[0]

    return split_axis, split_factor


def _new_alloc(tvm_ir, dtype, shape, name, scope):
    """
    allocate new buffer for ir builder

    Parameters
    ----------
    tvm_ir : tvm.ir_builder
        Developer API of IR node builder make function
    dtype : string
        buffer date type
    shape : list of int
        buffer shape
    name : string
        buffer name
    scope : string
        buffer memory scope

    Returns
    -------
    new_buffer : tvm.schedule.Buffer
        Symbolic data buffer
    """
    buf_var = tvm_ir.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope,
                                 data=buf_var)

    return new_buffer


def _kernel_ir(_, src):
    """
    algorithm: assignadd
    calculating data's add: a = a + b in IR build method.

    Parameters
    ----------
    src : data_a and data_b, the same address as dst.

    Returns
    -------
    tvm_ir.get() : the ir_builder created
        Developer API of IR node builder make function.
    """
    tvm_ir = tvm.ir_builder.create()

    input_a = src[0]
    input_b = src[1]

    input_a_ub = _new_alloc(tvm_ir, input_a.dtype, input_a.shape,
                            "input_a_ub", scope=cce.scope_ubuf)
    input_b_ub = _new_alloc(tvm_ir, input_b.dtype, input_b.shape,
                            "input_b_ub", scope=cce.scope_ubuf)
    reg = tvm_ir.allocate("int64", (2,), name='reg', scope=cce.scope_reg)
    tvm_ir.emit(tvm.call_extern(input_a.dtype, "copy_gm_to_ubuf",
                                input_a_ub.access_ptr("w"),
                                input_a.access_ptr("r"),
                                0, 1, 1, 0, 0))

    tvm_ir.emit(tvm.call_extern(input_b.dtype, "copy_gm_to_ubuf",
                                input_b_ub.access_ptr("w"),
                                input_b.access_ptr("r"),
                                0, 1, 1, 0, 0))

    tvm_ir.emit(tvm.call_extern(
        input_a_ub.dtype, "reg_mov",
        tvm.call_extern(reg.dtype, "reg", reg[0]),
        input_a_ub.access_ptr('r', offset=0)
    ))

    tvm_ir.emit(tvm.call_extern(
        input_b_ub.dtype, "reg_mov",
        tvm.call_extern(reg.dtype, "reg", reg[1]),
        input_b_ub.access_ptr('r', offset=0)
    ))

    reg[0] = reg[0] + reg[1]

    tvm_ir.emit(tvm.call_extern(
        input_a_ub.dtype, "reg_mov",
        input_a_ub.access_ptr('w', offset=0),
        tvm.call_extern(reg.dtype, "reg", reg[0])
    ))

    tvm_ir.emit(tvm.call_extern(input_a.dtype, "copy_ubuf_to_gm",
                                input_a.access_ptr('w'),
                                input_a_ub.access_ptr("r"),
                                0, 1, 1, 0, 0))

    return tvm_ir.get()


# pylint: disable=locally-disabled,too-many-locals,unnecessary-lambda
# pylint: disable=locally-disabled,too-many-statements
def _compute_assignadd(shape_x, shape_y, dtype):
    """
    assignadd compute function for int8, uint8, int32, float16, float32

    Parameters
    ----------
    shape_x : list or tuple
        shape of data_1.
    shape_y : list or tuple
        shape of data_2.
    dtype : str
        the data type.

    Returns
    -------
    sch: tvm.schedule
        the compute schedule
    data_a: tvm.tensor
        tensor of data_1
    data_b: tvm.tensor
        tensor of data_2
    res: tvm.tensor
        tensor of result
    """
    data_a = tvm.placeholder(shape_x, dtype=dtype, name='data_a')
    data_b = tvm.placeholder(shape_y, dtype=dtype, name='data_b')
    data_a_ub = tvm.compute(shape_x, lambda *i: data_a(*i), name='data_a_ub')
    data_b_ub = tvm.compute(shape_y, lambda *i: data_b(*i), name='data_b_ub')
    if dtype in ("int8", "uint8"):
        data_a_cast = tvm.compute(shape_x,
                                  lambda *i: data_a_ub(*i).astype("float16"),
                                  name="data_a_cast")
        data_b_cast = tvm.compute(shape_y,
                                  lambda *i: data_b_ub(*i).astype("float16"),
                                  name="data_b_cast")
    else:
        data_a_cast = data_a_ub
        data_b_cast = data_b_ub
    res_ub = tvm.compute(shape_x, lambda *i: data_a_cast(*i) + data_b_cast(*i),
                         name='res_ub.local.UB')
    if dtype in ("int8", "uint8"):
        res_ub_cast = tvm.compute(shape_x, lambda *i: res_ub(*i).astype(dtype),
                                  name="res_ub_cast")
    else:
        res_ub_cast = res_ub
    res = tvm.compute(shape_x, lambda *i: res_ub_cast(*i), name='res')
    sch = tvm.create_schedule(res.op)

    sch[data_a_ub].set_scope(cce.scope_ubuf)
    sch[data_b_ub].set_scope(cce.scope_ubuf)
    sch[data_a_cast].set_scope(cce.scope_ubuf)
    sch[data_b_cast].set_scope(cce.scope_ubuf)
    sch[res_ub].set_scope(cce.scope_ubuf)
    sch[res_ub_cast].set_scope(cce.scope_ubuf)

    # shape_x is same as shape_y
    # choose a appropriate method of tiling the tensor
    split_axis, split_factor = _tilling_axis(shape_x, dtype=dtype)
    axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis],
                                            factor=split_factor)

    if split_axis == 0:
        core_num = shape_x[split_axis] // split_factor
    else:
        core_num = shape_x[0]

    if core_num <= 65534:
        if split_axis == 0:
            sch[res].bind(axis_outer, tvm.thread_axis('blockIdx.x'))
        else:
            sch[res].bind(res.op.axis[0], tvm.thread_axis('blockIdx.x'))

    sch[data_a_ub].compute_at(sch[res], axis_outer)
    sch[data_b_ub].compute_at(sch[res], axis_outer)
    sch[data_a_cast].compute_at(sch[res], axis_outer)
    sch[data_b_cast].compute_at(sch[res], axis_outer)
    sch[res_ub].compute_at(sch[res], axis_outer)
    sch[res_ub_cast].compute_at(sch[res], axis_outer)

    sch[data_a].reused_by(res)

    sch[data_a_ub].emit_insn(data_a_ub.op.axis[split_axis], insn_cmd.DMA_COPY)
    sch[data_b_ub].emit_insn(data_b_ub.op.axis[split_axis], insn_cmd.DMA_COPY)
    if dtype in ("int8", "uint8"):
        sch[data_a_cast].emit_insn(data_a_cast.op.axis[split_axis],
                                   insn_cmd.CAST)
        sch[data_b_cast].emit_insn(data_b_cast.op.axis[split_axis],
                                   insn_cmd.CAST)
    sch[res_ub].emit_insn(res_ub.op.axis[split_axis], insn_cmd.ADD)
    if dtype in ("int8", "uint8"):
        sch[res_ub_cast].emit_insn(res_ub_cast.op.axis[split_axis],
                                   insn_cmd.CAST)
    sch[res].emit_insn(axis_inner, insn_cmd.DMA_COPY)

    return sch, data_a, data_b, res


def _update_shape(shape_x, shape_y):
    """
    update the shape of shape_x and shape_y

    """
    if len(shape_x) >= 2 and shape_x[0] == 1:
        len_s = len(shape_x)
        flag = len_s - 1
        for i in range(len_s):
            if shape_x[i] != 1:
                flag = i
                break

        shape_x = shape_x[flag:]
        shape_y = shape_y[flag:]

    return shape_x, shape_y


# pylint: disable=locally-disabled,too-many-arguments, unused-argument
@util.check_input_type(dict, dict, dict, str)
def assign_add(ref, value, output, kernel_name="assign_add"):
    """
    algorithm: assign_add
    update ref by adding value to it
    calculating data's add, a = a + b

    Parameters
    ----------
    ref: dict
        dict of input_ref, include shape and dtype,
    value: dict
        dict of input_value, include shape and dtype,
        Must have the same shape and dtype as input_ref
    output: dict
        dict of output
    kernel_name : str
        cce kernel name, default value is assign_add

    Returns
    -------
    None
    """
    # check if the parameter is valid
    shape_x = util.scalar2tensor_one(ref.get("shape"))
    shape_y = util.scalar2tensor_one(value.get("shape"))
    dtype = ref.get("dtype").lower()
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    check_list = ("float16", "float32", "int8", "uint8", "int32", "int64")
    util.check_dtype_rule(dtype, check_list)

    # process the data of int64
    if dtype == "int64":
        util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT_INT64)
        util.check_shape_size(shape_y, SHAPE_SIZE_LIMIT_INT64)
        data_a = tvm.placeholder(shape_x, dtype=dtype, name="data_a")
        data_b = tvm.placeholder(shape_y, dtype=dtype, name="data_b")
        res = tvm.extern([shape_x, shape_y], [data_a, data_b],
                         lambda ins, outs: _kernel_ir(outs, ins), name="res",
                         dtype=dtype)
        sch = tvm.create_schedule(res.op)

    # process the data of float16 or float32
    else:
        util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT_OTHER)
        util.check_shape_size(shape_y, SHAPE_SIZE_LIMIT_OTHER)

        if shape_x != shape_y:
            raise RuntimeError(
                "shape_x and shape_y must be the same")

        shape_x, shape_y = _update_shape(shape_x, shape_y)

        sch, data_a, data_b, res = _compute_assignadd(shape_x, shape_y, dtype)

    with build_config:
        tvm.build(sch, [data_a, data_b, res], "cce", name=kernel_name)
