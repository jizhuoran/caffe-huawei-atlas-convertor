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

reduce atomic schedule
"""
import math
import te.lang.cce
from te import tvm
from te.platform import cce_emitinsn_params
from te.platform import cce_util


@tvm.register_func("tvm.intrin.cce.dichotomy_reduce_block_mean")
def dichotomy_reduce_block_mean(stmt_op):  # pylint: disable=too-many-locals
    """Collapse second input tensor to one repeat and use vcadd to calculate sum to output"""
    # Get input and output buffers
    input_size = [1]
    for_extents = []
    ir_builder = tvm.ir_builder.create()

    def _post_order_for(_stmt):
        if isinstance(_stmt, tvm.stmt.For):
            input_size[0] *= _stmt.extent.value
            for_extents.append(_stmt.extent.value)

    tvm.ir_pass.IRTransform(stmt_op, None, _post_order_for, ["For"])
    ins, outs = cce_util.get_buffer(stmt_op)
    in_buffer = ins[1]
    out_buffer = outs[0]
    input_size = input_size[0]

    # Check if input can be collapsed into one repeat
    vector_inst_one_repeat_size = te.platform.cce_params.VECTOR_INST_BLOCK_WIDTH \
        // cce_util.get_align_factor(in_buffer.dtype)[1]
    collapse_loop_num = math.log(input_size / vector_inst_one_repeat_size, 2)
    if not collapse_loop_num.is_integer():
        collapse_repeat = int(math.pow(2, int(collapse_loop_num)))
        total_repeat = input_size // vector_inst_one_repeat_size
        block_size = te.platform.cce_util.get_align_factor(in_buffer.dtype)[0]
        remain_block = (input_size - total_repeat * vector_inst_one_repeat_size) / block_size
        remain_block_size = int(remain_block * block_size)
        out_of_collapse_repeat = total_repeat - collapse_repeat
        if not remain_block.is_integer():
            raise RuntimeError("Input size is not aligned:", input_size)
        out_of_collapse_repeat = int(out_of_collapse_repeat)
        te.platform.cce_intrin_md.reset_mask_insn(ir_builder, in_buffer.dtype)
        ir_builder.emit(tvm.call_extern(
            in_buffer.dtype,
            "vadd",
            in_buffer.access_ptr("rw", offset=0),
            in_buffer.access_ptr("r", offset=0),
            in_buffer.access_ptr("r", offset=vector_inst_one_repeat_size * collapse_repeat),
            out_of_collapse_repeat, 1, 1, 1, 8, 8, 8))
        if remain_block_size > 0:
            te.platform.cce_intrin_md.reset_mask_insn(ir_builder, in_buffer.dtype,
                                                      bits=remain_block_size)
            ir_builder.emit(tvm.call_extern(
                in_buffer.dtype,
                "vadd",
                in_buffer.access_ptr("rw", offset=0),
                in_buffer.access_ptr("r", offset=0),
                in_buffer.access_ptr("r", offset=vector_inst_one_repeat_size * collapse_repeat +
                                     vector_inst_one_repeat_size * out_of_collapse_repeat),
                1, 1, 1, 1, 8, 8, 8))
        te.platform.cce_intrin_md.reset_mask_insn(ir_builder, in_buffer.dtype)
        input_size = collapse_repeat * vector_inst_one_repeat_size
    # Do Emit Insn

    def collapse(ir_b, buffer, current_size):
        repeat = current_size // 2 // vector_inst_one_repeat_size
        ir_b.emit(tvm.call_extern(
            buffer.dtype,
            "vadd",
            buffer.access_ptr("rw", offset=0),
            buffer.access_ptr("r", offset=0),
            buffer.access_ptr("r", offset=current_size // 2),
            repeat, 1, 1, 1, 8, 8, 8))
        return current_size // 2

    cur_size = input_size
    for _ in range(int(collapse_loop_num)):
        cur_size = collapse(ir_builder, in_buffer, cur_size)
    target_size = cce_emitinsn_params.cceEmitParamsIns.get_param("block_split_factor")
    loop_num = int(math.log(vector_inst_one_repeat_size // target_size, 2))
    for _ in range(loop_num):
        half_size = cur_size // 2
        te.platform.cce_intrin_md.reset_mask_insn(ir_builder, in_buffer.dtype, bits=half_size)
        ir_builder.emit(tvm.call_extern(
                        in_buffer.dtype,
                        "vadd",
                        in_buffer.access_ptr("rw", offset=0),
                        in_buffer.access_ptr("r", offset=0),
                        in_buffer.access_ptr("r", offset=half_size),
                        1, 1, 1, 1, 8, 8, 8))
        cur_size = half_size

    te.platform.cce_intrin_md.reset_mask_insn(ir_builder, in_buffer.dtype)
    ir_builder.emit(tvm.call_extern(
        in_buffer.dtype,
        "vadd",
        out_buffer.access_ptr("rw", offset=0),
        in_buffer.access_ptr("r", offset=0),
        out_buffer.access_ptr("rw", offset=0), 1, 1, 1, 1, 8, 8, 8))

    return ir_builder.get()


def reduce_mean_mid_reduce_high_performance_schedule(outs,  # pylint: disable=R0914, R0912, R0915
                                                     sch_list):
    """For middle reduce like 32, 65535, 16, axis=1"""
    # Special reversed search, if there is branch, raise Error
    res = outs[0]

    def _reversed_search_tensor(_tensor, _tensor_list, _placeholder_list, tensor_dict):
        if _tensor not in _tensor_list:
            _tensor_list.append(_tensor)
        if len(_tensor.op.input_tensors) > 1:
            raise RuntimeError("reduce_mean shouldn't have branches:")
        if len(_tensor.op.input_tensors) == 1:
            src = _tensor.op.input_tensors[0]
            if src in tensor_dict:
                tensor_dict[src].append(_tensor)
            else:
                tensor_dict[src] = [_tensor]
            _reversed_search_tensor(src, _tensor_list, _placeholder_list, tensor_dict)
        else:
            _placeholder_list.append(_tensor)
    # Find all tensors
    final_output = res
    tensor_list = []
    tensor_src_dst_dict = {}
    placeholder_list = []
    _reversed_search_tensor(final_output, tensor_list, placeholder_list, tensor_src_dst_dict)
    reduce_axis = [len(placeholder_list[0].shape) - 2]  # Controlled by distribution rule
    mul_tensor = None
    sum_tensor = None
    for tensor in tensor_list:
        if "reduce_sum" in tensor.op.tag:
            sum_tensor = tensor
        elif "VS_mul" in tensor.op.tag:
            mul_tensor = tensor
    is_keepdims = len(sum_tensor.shape) == len(sum_tensor.op.input_tensors[0].shape)
    # Existence of Branches is impossible here, hence only one placeholder presents
    placeholder = placeholder_list[0]
    original_shape = tuple(map(int, placeholder.shape))
    # ////////////////////////////////
    # //         Schedule           //
    # ////////////////////////////////
    cce_emitinsn_params.cceEmitParamsIns.clear_param()
    sch = sch_list[0]
    block_elem_num, element_byte_size = te.platform.cce_util.get_align_factor(res.dtype)
    # Double block_elem_num for DMA performance
    block_elem_num *= 2
    # Get maximum core num
    core_num = int(te.platform.get_soc_spec("CORE_NUM"))
    # available block split axes are all axes except reduce axes
    possible_axes = [axis for axis, _ in enumerate(original_shape) if axis not in reduce_axis]
    # Get UB size
    ub_size = te.platform.get_soc_spec("UB_SIZE")
    # ////////////////////////////////
    # //    Tiling Calculation      //
    # ////////////////////////////////
    block_split_axis, block_split_factor, block_split_last_axis, block_split_nparts,\
        maximum_loop = tiling_calculation(block_elem_num,
                                          core_num,
                                          element_byte_size,
                                          original_shape,
                                          possible_axes,
                                          ub_size)
    # ////////////////////////////////
    # //      DataFlow Control      //
    # ////////////////////////////////
    # Set data on UB
    for tensor in [sum_tensor, mul_tensor]:
        if tensor not in (res,):
            sch[tensor].set_scope(te.platform.cce_params.scope_ubuf)
    # Read data on GM
    placeholder_ub = sch.cache_read(placeholder,
                                    te.platform.cce_params.scope_ubuf,
                                    tensor_src_dst_dict[placeholder])
    # Write data to GM
    res_ub = sch.cache_write(res, te.platform.cce_params.scope_ubuf)
    # ////////////////////////////////
    # //        Do Blk split        //
    # ////////////////////////////////
    if not is_keepdims and block_split_last_axis:
        block_split_axis -= 1
    if block_split_factor <= 1:
        # No block split needed
        block_outer, block_inner = sch[res].split(sch[res].op.axis[block_split_axis],
                                                  factor=1)
    else:
        block_outer, block_inner = sch[res].split(sch[res].op.axis[block_split_axis],
                                                  nparts=block_split_nparts)
        sch[res].bind(block_outer, tvm.thread_axis("blockIdx.x"))
    reduce_outer, reduce_inner = sch[res_ub].split(sch[res_ub].op.reduce_axis[0],
                                                   factor=maximum_loop)
    sch[res_ub].reorder(reduce_outer, reduce_inner, sch[res_ub].op.axis[-1])
    # ////////////////////////////////
    # //      Compute Control       //
    # ////////////////////////////////
    # compute at sum always
    for tensor in [placeholder, placeholder_ub, mul_tensor]:
        sch[tensor].compute_at(sch[res_ub], reduce_outer)
    for tensor in [res_ub]:
        sch[tensor].compute_at(sch[res], block_outer)
    # ////////////////////////////////
    # //         Emit Insn          //
    # ////////////////////////////////

    def emit_on_self_ex(_tensor, axis, operation):
        sch[_tensor].emit_insn(axis, operation)

    def emit_on_self(_tensor, axisnum=0, operation='dma_copy'):
        emit_on_self_ex(_tensor, sch[_tensor].op.axis[axisnum], operation)

    emit_on_self(placeholder_ub)
    emit_on_self_ex(res_ub, reduce_inner, "dichotomy_reduce_block_mean")
    emit_on_self(mul_tensor, operation=mul_tensor.op.tag.split('|')[0])
    if block_split_last_axis:
        cce_emitinsn_params.cceEmitParamsIns.insert_param("block_split_factor",
                                                          block_split_factor)
    else:
        cce_emitinsn_params.cceEmitParamsIns.insert_param("block_split_factor",
                                                          original_shape[-1])
    emit_on_self_ex(res, axis=block_inner, operation="dma_copy")
    return sch


def tiling_calculation(block_elem_num, core_num, element_byte_size,
                       original_shape, possible_axes, ub_size):
    """Extracted from main function in order to avoid static checks"""
    # get best block split axis, currently supports last or first axis only
    block_split_last_axis = False
    if len(possible_axes) == 1:
        block_split_last_axis = True
        block_split_axis = possible_axes[-1]
        # If block_split_axis is not sufficient for all core, change maximum core num to axis size
        estimate_axis_size = original_shape[block_split_axis] / block_elem_num
        if estimate_axis_size < 0.5:
            raise RuntimeError("reduce mean high performance schedule doesn't support aligned")
        estimate_axis_size = math.ceil(estimate_axis_size)
        block_split_nparts = min(core_num, estimate_axis_size)
        # Calculate block split estimated result, if no axis available, it will be 1
        estimate_block_split_factor = max(1.0,
                                          original_shape[block_split_axis] / block_split_nparts)
        if estimate_block_split_factor.is_integer():
            # Exact division achieved
            block_split_factor = int(estimate_block_split_factor)
        else:
            raise RuntimeError("reduce mid mean high performance schedule doesn't support "
                               "unaligned data:" + str(original_shape))
    else:
        block_split_axis = possible_axes[0]
        block_split_nparts = min(core_num, original_shape[block_split_axis])
        block_split_factor = original_shape[block_split_axis] / block_split_nparts
    # ////////////////////////////////
    # //     Vector Reduce Opt      //
    # ////////////////////////////////
    # Size of UB needed for each loop
    if block_split_last_axis:
        ub_usage_factor = 8 * block_split_factor * element_byte_size
    else:
        ub_usage_factor = 8 * original_shape[-1] * element_byte_size
    maximum_loop = ub_size // ub_usage_factor
    return block_split_axis, block_split_factor, block_split_last_axis,\
        block_split_nparts, maximum_loop
