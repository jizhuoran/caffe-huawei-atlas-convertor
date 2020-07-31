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

bn_training_reduce
"""
from __future__ import division

import math

import te.lang.cce
import te.platform.cce_params as cce_params
import te.platform.cce_emitinsn_params as cce_emitinsn_params

from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.platform import cce_util
from te.platform.cce_build import build_config

from topi import generic
from topi.cce import util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# pylint: disable=locally-disabled,unused-argument,invalid-name
# pylint: disable=locally-disabled,redefined-builtin
def op_select_format(x, sum, square_sum,
                     kernel_name="bn_training_reduce"):
    """
    select format dynamically
    """
    origin_format = x.get("ori_format").upper()
    origin_shape = x.get("ori_shape")

    # can support Nz + ND
    if origin_format == "NCHW" and len(origin_shape) == 4 \
            and origin_shape[0] == 1 and origin_shape[2] == 1:
        input0 = gen_param(classify="input0", name="x",
                           datatype="float16,float,float16,float",
                           format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        output0 = gen_param(classify="output0", name="sum",
                            datatype="float,float,float,float",
                            format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        output1 = gen_param(classify="output1", name="square_sum",
                            datatype="float,float,float,float",
                            format="NCHW,NCHW,NC1HWC0,NC1HWC0")
    # support 5HD + 5HD
    else:
        input0 = gen_param(classify="input0", name="x",
                           datatype="float16,float",
                           format="NC1HWC0,NC1HWC0")
        output0 = gen_param(classify="output0", name="sum",
                            datatype="float,float",
                            format="NC1HWC0,NC1HWC0")
        output1 = gen_param(classify="output1", name="square_sum",
                            datatype="float,float",
                            format="NC1HWC0,NC1HWC0")

    param_list = [input0, output0, output1]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _check_format(data_format, origin_foramt):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    data_format: str
        data format of data
    origin_foramt: str
        origin format of data

    Returns
    -------
    None
    """
    if data_format.upper() not in ("NC1HWC0", "NCHW"):
        raise RuntimeError("The data format only supports NC1HWC0 and NCHW.")
    if data_format.upper() == "NCHW":
        if origin_foramt not in ("NCHW",):
            raise RuntimeError("The origin format only supports "
                               "NCHW when format is NCHW")


def _reduce_compute_5hd(x):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_reduce compute
    """
    square_x = te.lang.cce.vmul(x, x)

    axis = [0, 2, 3]
    sum_x, square_sum_x = te.lang.cce.tuple_sum([x, square_x], axis, True)

    res = [sum_x, square_sum_x]

    return res


def _reduce_compute_nd(x, sum):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: dict
        dict of sum, A `Tensor`. Sum of x.

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_reduce compute
    """
    origin_format = sum.get("ori_format")
    shape = te.lang.cce.util.shape_to_list(x.shape)
    axis = list(range(len(shape)))

    if origin_format == "NCHW":
        axis.pop(1)

    for _, i in enumerate(range(len(shape))):
        if shape[i] == 1 and i in axis:
            axis.remove(i)

    square_x = te.lang.cce.vmul(x, x)
    sum_x = te.lang.cce.sum(x, axis, False)
    square_sum_x = te.lang.cce.sum(square_x, axis, False)

    # Output has been reversed because of binary_reduce_output_reversed
    res = [square_sum_x, sum_x]

    return res


# Schedule for ND
# Including definition for several operator specific intrinsic instructions
# bn_reduce_sum
# pylint: disable=locally-disabled,too-many-locals,unused-variable
@tvm.register_func("tvm.intrin.cce.bn_reduce_sum")
def bn_reduce_sum(stmt_op):
    """
    Collapse second input tensor to one repeat
    and use vcadd to calculate sum to output
    """
    # Get input and output buffers
    input_size_list = [1]
    for_extents = []
    ir_builder = tvm.ir_builder.create()
    cce_util.get_init_op(stmt_op)

    def _post_order_for(_stmt):
        if isinstance(_stmt, tvm.stmt.For):
            input_size_list[0] = input_size_list[0] * _stmt.extent.value
            for_extents.append(_stmt.extent.value)

    tvm.ir_pass.IRTransform(stmt_op, None, _post_order_for, ["For"])
    ins, outs = \
        cce_util.get_buffer(stmt_op, need_unique=True, need_origin_adress=True)
    in_buffer = ins[1]
    out_buffer = outs[0]
    input_size = input_size_list[0]

    # Check if input can be collapsed into one repeat
    vector_inst_one_repeat_size = \
        cce_params.VECTOR_INST_BLOCK_WIDTH // \
        cce_util.get_align_factor(in_buffer.dtype)[1]


    # get reduce_axis shape
    if len(for_extents) == 1:
        input_reduce_axis_shape = for_extents[0]
        ub_loop_num = 1
    else:
        input_reduce_axis_shape = for_extents[0]
        ub_loop_num = for_extents[1]

    collapse_loop_num = \
        math.log(input_reduce_axis_shape / vector_inst_one_repeat_size, 2)

    # judge reduce_shape is remaining or not after dichotomy add
    remain_flag = False
    collapse_repeat = 0
    if not collapse_loop_num.is_integer():
        collapse_repeat = int(math.pow(2, int(collapse_loop_num)))
        out_of_collapse_repeat = \
            input_reduce_axis_shape / vector_inst_one_repeat_size - \
            collapse_repeat
        if not out_of_collapse_repeat.is_integer():
            raise RuntimeError("Input size is not aligned:",
                               input_reduce_axis_shape)
        remain_flag = True

    # Do Emit Insn
    def collapse(ir_b, buffer, current_size):
        """Function to do emit insn"""
        repeat = current_size // 2 / vector_inst_one_repeat_size
        tail_flag = False
        if not repeat.is_integer():
            tail_flag = True
        repeat = int(repeat)

        ir_b.emit(tvm.call_extern(
            buffer.dtype,
            "vadd",
            buffer.access_ptr("rw", offset=0),
            buffer.access_ptr("r", offset=0),
            buffer.access_ptr("r", offset=8),
            repeat, 1, 2, 2, 8, 16, 16))

        # solve tail vadd
        if tail_flag:
            tail_mask = \
                (current_size - repeat * 2 * vector_inst_one_repeat_size) // 2
            te.platform.cce_intrin_md.reset_mask_insn(ir_builder,
                                                      in_buffer.dtype,
                                                      tail_mask)
            ir_b.emit(tvm.call_extern(
                buffer.dtype,
                "vadd",
                buffer.access_ptr("rw",
                                  offset=repeat*vector_inst_one_repeat_size),
                buffer.access_ptr("r",
                                  offset=repeat*2*vector_inst_one_repeat_size),
                buffer.access_ptr("r",
                                  offset=repeat*2*vector_inst_one_repeat_size +
                                  8),
                1, 1, 2, 2, 0, 0, 0))
            te.platform.cce_intrin_md.reset_mask_insn(ir_builder,
                                                      in_buffer.dtype)
        return current_size // 2

    # emit vadd
    cur_size = input_size
    for loop in range(int(collapse_loop_num)):
        cur_size = collapse(ir_builder, in_buffer, cur_size)

    if remain_flag:
        # solve remain repeat
        mask_bits = \
            input_reduce_axis_shape / collapse_repeat - \
            vector_inst_one_repeat_size
        add_repeat_stride = int(8 + mask_bits / 8)
        te.platform.cce_intrin_md.reset_mask_insn(ir_builder,
                                                  in_buffer.dtype, mask_bits)
        ir_builder.emit(tvm.call_extern(
            in_buffer.dtype,
            "vadd",
            in_buffer.access_ptr("rw", offset=0),
            in_buffer.access_ptr("r", offset=0),
            in_buffer.access_ptr("r", offset=vector_inst_one_repeat_size),
            ub_loop_num, 1, 1, 1,
            add_repeat_stride,
            add_repeat_stride,
            add_repeat_stride))

        # emit vcadd for remain
        te.platform.cce_intrin_md.reset_mask_insn(ir_builder, in_buffer.dtype)
        ir_builder.emit(tvm.call_extern(
            in_buffer.dtype,
            "vcadd",
            out_buffer.access_ptr("rw", offset=0),
            in_buffer.access_ptr("r", offset=0),
            ub_loop_num, 1, 1, add_repeat_stride))
    else:
        # emit vcadd for no remain
        ir_builder.emit(tvm.call_extern(
            in_buffer.dtype,
            "vcadd",
            out_buffer.access_ptr("rw", offset=0),
            in_buffer.access_ptr("r", offset=0), ub_loop_num, 1, 1, 8))

    return ir_builder.get()


# pylint: disable=locally-disabled,too-many-locals
@tvm.register_func("tvm.intrin.cce.binary_reduce_output_reversed")
def binary_reduce_output(stmt_op):
    """Move reduce results to two destinations"""
    # Get input and output buffers
    input_size_list = [1]
    ir_builder = tvm.ir_builder.create()

    def _post_order_for(_stmt):
        if isinstance(_stmt, tvm.stmt.For):
            input_size_list[0] = input_size_list[0] * _stmt.extent.value

    def new_alloc(tvm_ib, dtype, shape, name, scope):
        """Funtion to alloc mem"""
        buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name,
                                     scope=scope, data=buf_var)
        return new_buffer
    _ = tvm.ir_pass.IRTransform(stmt_op, None, _post_order_for, ["For"])
    ins, outs = cce_util.get_buffer(stmt_op)
    # Alloc second buffer for binary collection
    out_buffer_sec = \
        cce_emitinsn_params.cceEmitParamsIns.get_param("binary_reduce"
                                                       "_output_buffer")
    in_buffer = ins[0], ins[1]
    out_buffer = outs[0], out_buffer_sec
    input_size = input_size_list[0]
    output_size = input_size
    block_unit = cce_util.get_align_factor(in_buffer[0].dtype)[0]
    remain_buffer = new_alloc(ir_builder, out_buffer[0].dtype, (block_unit,),
                              "copy_part_0", cce_params.scope_ubuf)
    remain_buffer_sec = new_alloc(ir_builder, out_buffer[1].dtype,
                                  (block_unit,), "copy_part_1",
                                  cce_params.scope_ubuf)
    burst_len = output_size // block_unit
    remains = output_size - burst_len * block_unit
    remains_fill = block_unit - remains
    if output_size < block_unit:
        raise RuntimeError("DMA Copy cannot move less than 32 Byte data"
                           " without corrupting original data")

    # Main part
    global_offset = out_buffer[0].elem_offset
    ir_builder.emit(
        tvm.call_extern(out_buffer[0].dtype,
                        "copy_ubuf_to_gm",
                        out_buffer[0].access_ptr("rw"),
                        in_buffer[1].access_ptr("r"),
                        0,
                        1,
                        burst_len, 0, 0))
    ir_builder.emit(
        tvm.call_extern(out_buffer[1].dtype,
                        "copy_ubuf_to_gm",
                        out_buffer[1].access_ptr("rw", offset=global_offset),
                        in_buffer[0].access_ptr("r"),
                        0,
                        1,
                        burst_len, 0, 0))
    # Remain part
    if remains > 0:
        with ir_builder.for_range(0, block_unit, name="copy_part_fill_loop") \
                as reg_mov_loop:
            ir_builder.emit(tvm.call_extern(
                remain_buffer.dtype, "reg_mov",
                remain_buffer.access_ptr("rw", offset=reg_mov_loop),
                in_buffer[1].access_ptr("r",
                                        offset=burst_len * block_unit -
                                        remains_fill + reg_mov_loop)))
            ir_builder.emit(tvm.call_extern(
                remain_buffer_sec.dtype, "reg_mov",
                remain_buffer_sec.access_ptr("rw", offset=reg_mov_loop),
                in_buffer[0].access_ptr("r",
                                        offset=burst_len *
                                        block_unit -
                                        remains_fill + reg_mov_loop)))
        ir_builder.emit(
            tvm.call_extern(out_buffer[0].dtype,
                            "copy_ubuf_to_gm",
                            out_buffer[0].access_ptr("rw",
                                                     offset=burst_len *
                                                     block_unit - remains_fill),
                            remain_buffer.access_ptr("r"),
                            0,
                            1,
                            1, 0, 0))
        ir_builder.emit(
            tvm.call_extern(out_buffer[1].dtype,
                            "copy_ubuf_to_gm",
                            out_buffer[1].access_ptr("rw",
                                                     offset=global_offset +
                                                     burst_len * block_unit -
                                                     remains_fill),
                            remain_buffer_sec.access_ptr("r"),
                            0,
                            1,
                            1, 0, 0))
    return ir_builder.get()


# pylint: disable=locally-disabled,too-many-branches
def bn_training_reduce_schedule_nd(res):
    """bn_training_reduce schedule method"""
    cce_emitinsn_params.cceEmitParamsIns.clear_param()
    # Prepare extra tensors
    # Step 1: Get two output tensors
    # Step 2: Merge two output tensors into Dummy
    # Step 3: Move UB data to GM tensor
    output_first = res[0]  # Square Sum
    output_second = res[1]  # Sum
    final_output = tvm.compute(output_first.shape,
                               lambda *indices: output_first(*indices) +
                               output_second(*indices),
                               name="DummyYummySweety")
    is_cast = False
    if "cast" in output_second.op.input_tensors[0].name:
        is_cast = True
    # Prepare block split parameters
    # First, try to split N axis with core_num, no toleration to remains
    # If there exists remains, check if N axis is shorter than core_num
    # If no rule applies, raise RuntimeError
    axis_n_size = int(res[0].shape[1])
    core_num = int(te.platform.CceProductParams().getParams("Device_core_num"))
    # Try block split to core_num
    core_split_remain = int(axis_n_size % core_num)

    element_size = cce_util.get_align_factor(output_first.dtype)[1]
    block_element_num = te.platform.cce_intrin_md.ALIGNMENT_BYTES // 2
    if int(core_split_remain) == 0 and axis_n_size // core_num > \
            block_element_num:
        block_split_part = int(core_num)
        block_split_factor = int(axis_n_size // core_num)
    elif axis_n_size // core_num < block_element_num:
        block_split_part = math.ceil(axis_n_size / block_element_num)
        block_split_factor = axis_n_size // block_split_part
    else:
        raise RuntimeError("Unable to get block split factor, "
                           "need backup plan: " + str(axis_n_size))
    # Calculate UB split
    ub_size = te.platform.CceProductParams().getParams("Unified_Buffer") // 2
    reduce_data_num = 1
    reduce_data_factor = 2
    if is_cast:
        reduce_data_factor = 3
    for reduce_axis in output_first.op.reduce_axis:
        reduce_data_num *= int(reduce_axis.dom.extent)
    reduce_data_num *= reduce_data_factor
    max_possible_loop = ub_size // (element_size * reduce_data_num)
    for loop in range(max_possible_loop - 1, 0, -1):
        if block_split_factor % loop == 0:
            actual_loop = loop
            break

    # Find all tensors
    if is_cast:
        # With Cast, prepare tensor parameters
        mul_tensor = output_first.op.input_tensors[0]
        cast_tensor = mul_tensor.op.input_tensors[0]
        res_input = cast_tensor.op.input_tensors[0]
        input_tensor_next = [cast_tensor]  # First compute tensor is cast_tensor
        ub_tensors = [cast_tensor, mul_tensor, output_first, output_second]
    else:
        # Without Cast, prepare tensor parameters
        cast_tensor = None
        mul_tensor = output_first.op.input_tensors[0]
        res_input = mul_tensor.op.input_tensors[0]
        input_tensor_next = [mul_tensor, output_second]  # First compute tensor is cast_tensor
        ub_tensors = [mul_tensor, output_first, output_second]

    # Create original schedule
    sch = tvm.create_schedule(final_output.op)
    # ////////////////////////////////////
    # ///////// DataFlow Control /////////
    # ////////////////////////////////////
    # Read input in
    input_tensor_ub = sch.cache_read(res_input, cce_params.scope_ubuf, input_tensor_next)
    ub_tensors.append(input_tensor_ub)
    # Compute procedure in ubuf
    for ub_tens in ub_tensors:
        sch[ub_tens].set_scope(cce_params.scope_ubuf)
    # ////////////////////////////////////
    # //////// Split axis Control ////////
    # ////////////////////////////////////
    outer, inner = \
        sch[final_output].split(sch[final_output].op.axis[1],
                                nparts=block_split_part)
    ub_outer, ub_inner = sch[final_output].split(inner, factor=actual_loop)
    sch[final_output].bind(outer, tvm.thread_axis("blockIdx.x"))
    # ////////////////////////////////////
    # ///////// Compute Control //////////
    # ////////////////////////////////////
    compute_at_axis = ub_outer
    for ub_tens in ub_tensors:
        sch[ub_tens].compute_at(sch[final_output], compute_at_axis)
    # ////////////////////////////////////
    # //////////// EmitInsn //////////////
    # ////////////////////////////////////

    def emit_on_self(tensor, axisnum=0, op='dma_copy'):
        """Do emit insn"""
        sch[tensor].emit_insn(sch[tensor].op.axis[axisnum], op)

    def emit_on_self_ex(tensor, axis, op='dma_copy'):
        """Do emit insn"""
        sch[tensor].emit_insn(axis, op)

    # Fake results
    emit_on_self(input_tensor_ub, 0)
    if is_cast:
        emit_on_self(cast_tensor, 0, cast_tensor.op.tag.split('|')[0])
    emit_on_self(mul_tensor, 0, mul_tensor.op.tag)

    sch[output_first].pragma(sch[output_first].op.axis[1], "emit_insn", "bn_reduce_sum")
    sch[output_second].pragma(sch[output_second].op.axis[1], "emit_insn", "bn_reduce_sum")
    sch[output_first].double_buffer()
    sch[output_second].double_buffer()

    emit_on_self_ex(final_output, ub_inner, "binary_reduce_output_reversed")

    def new_alloc(dtype, shape, name):
        """Alloc mem"""
        new_buffer = tvm.decl_buffer(shape, dtype, name=name,
                                     scope="", data=None)
        return new_buffer

    out_buffer_sec = new_alloc(final_output.dtype,
                               (block_split_factor,),
                               "reduce_sec_output_gm")
    cce_emitinsn_params.cceEmitParamsIns.insert_param(
        "binary_reduce_output_buffer", out_buffer_sec)
    tensor_list = [res_input,
                   final_output,
                   out_buffer_sec]

    return sch, tensor_list


@fusion_manager.register("bn_training_reduce")
def bn_training_reduce_compute(x, sum, square_sum,
                               kernel_name="bn_training_reduce"):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_reduce compute
    """
    if x.dtype == "float16":
        x = te.lang.cce.cast_to(x, "float32")
    data_format = sum.get("format")
    if data_format == "NC1HWC0":
        res = _reduce_compute_5hd(x)
    else:
        res = _reduce_compute_nd(x, sum)
    return res


@util.check_input_type(dict, dict, dict, str)
def bn_training_reduce(x, sum, square_sum,
                       kernel_name="bn_training_reduce"):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    dtype_x = x.get("dtype")

    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_x)
    util.check_dtype_rule(dtype_x.lower(), ("float16", "float32"))

    data_format = x.get("format")
    origin_format = x.get("ori_format")
    _check_format(data_format, origin_format)

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())

    res = bn_training_reduce_compute(x_input, sum, square_sum,
                                     kernel_name=kernel_name)
    if data_format == "NC1HWC0":
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
    else:
        sch, tensor_list = bn_training_reduce_schedule_nd(res)

        with build_config:
            tvm.build(sch, tensor_list, "cce", name=kernel_name)
        return
    tensor_list = [x_input] + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
