#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Pure broadcast intrin
"""
import math

import te

from te import tvm

from te.platform.cce_util import get_align_factor
from .util import get_least_common_multiple
from te.platform.cce_intrin_md import reset_mask_insn
from te.platform.cce_intrin_md import reset_mask_insn_inverted
from te.platform.cce_intrin_md import apply_for_new_alloc
from te.platform.cce_params import scope_reg, scope_ubuf

DMA_MODE = 0
VECTOR_MODE = 1
VECTOR_ENHANCED_MODE = 2
UINT8_MAXIMUM = 255


def last_axis_broadcast(*args):  # pylint: disable=too-many-locals, too-many-statements
    """Do last axis broadcast"""
    ir_builder, index, input_buffer, output_buffer, \
        broadcast_src, broadcast_factor = args
    ###########################################################################
    # There are currently 3 ways to do last axis broadcast:
    # 1. Direct scalar move DSM
    #        Use reg_mov to copy all elements one by one
    #        Advantage: Good for tiny broadcast such as 1, 2 to 2, 2
    #                   Stable support for all scene
    #        Disadvantage: Slowest
    #        Prerequisite: No
    #        Cycle is broadcast_src * broadcast_factor
    #        Extra storage needed: registers
    # 2. Last axis full aligned broadcast LFA
    #        Use reg_mov along with vector_dup to do full aligned broadcast
    #        Advantage: Fastest and simple for aligned broadcast
    #        Disadvantage: Supports only aligned last axis broadcast
    #        Prerequisite: Aligned or broadcast_src == 1
    #        Cycle is broadcast_src
    #        Plus broadcast_src * broadcast_factor // rep
    # 3. Mask grouping broadcast MG
    #        Calculate mask for each one of the element, then do vector_dup
    #        Advantage: Faster than DSM when dealing with unaligned data
    #        Disadvantage: May use too much registers (stack)
    #                      Supports only unaligned data larger than 1/2 block
    #        Prerequisite: src is larger than 1/2 block
    #        Cycle is broadcast_src
    #        Plus broadcast_src * (ceil(broadcast_factor / rep) + 1)
    #        Extra storage needed: registers
    # 4. Last axis transpose broadcast LAT
    ###########################################################################
    # Calculate broadcast information
    dtype_byte_size = get_align_factor(input_buffer.dtype)[1]
    align_factor = 32  # Currently all of our chip requires 32Byte aligned data for performance
    algorithm_score_dict = {
        "DSM": [True, common_unaligned_broadcast_last_axis, dsm_cycle_estimation],
        "LFA": [False, broadcast_last_axis_aligned, lfa_cycle_estimation],
        "MG": [False, mask_grouping_broadcast, mg_cycle_estimation],
        "LAT": [False, last_axis_transpose_broadcast, lat_cycle_estimation],
    }
    # Rule 1: For aligned data, enable Last axis full aligned broadcast algorithm
    #         Enable DSM if not aligned
    if int(broadcast_factor * dtype_byte_size % align_factor) == 0 or broadcast_src == 1:
        algorithm_score_dict["LFA"][0] = True
    # Rule 2: For unaligned broadcast_src >= 1 block, enable MG broadcast algorithm
    if broadcast_factor * dtype_byte_size >= align_factor and \
            not algorithm_score_dict["LFA"][0]:
        algorithm_score_dict["MG"][0] = True
    # Rule 3: Enable LAT for binary16 unaligned broadcast
    if dtype_byte_size == 2 and \
            not algorithm_score_dict["LFA"][0]:
        algorithm_score_dict["LAT"][0] = True
    # Collect all enabled algorithm
    enabled_algorithm = []
    for algorithm_info in algorithm_score_dict.values():
        if algorithm_info[0] is True:
            enabled_algorithm.append(algorithm_info)
    # Final stage: Select and execute algorithm
    if len(enabled_algorithm) == 1:
        result_buffer = enabled_algorithm[0][1](ir_builder, index, input_buffer, output_buffer,
                                                broadcast_src, broadcast_factor)
    else:
        best_algorithm = None
        current_cycle = -1
        # Find the best algorithm, only the algorithm with a lower cycle can stay
        for algorithm_info in enabled_algorithm:
            my_cycle = algorithm_info[2](input_buffer.dtype, broadcast_src, 1,
                                         broadcast_factor)
            if best_algorithm is None or current_cycle < 0 or my_cycle < current_cycle:
                best_algorithm = algorithm_info[1]
                current_cycle = my_cycle
        result_buffer = best_algorithm(ir_builder, index, input_buffer, output_buffer,
                                       broadcast_src, broadcast_factor)
    return result_buffer


def mid_axis_broadcast(*args):  # pylint: disable=too-many-locals, too-many-statements
    """Do mid axis broadcast"""
    ir_builder, index, input_buffer, output_buffer, \
        broadcast_src, broadcast_unit, broadcast_factor = args
    ###########################################################################
    # There are currently 3 ways to do mid axis broadcast:
    # 1. Direct scalar move DSM
    #        Use reg_mov to copy all elements one by one
    #        Advantage: Good for tiny broadcast such as 1, 2 to 2, 2
    #        Prerequisite: No
    #        Cycle is broadcast_src * broadcast_unit * broadcast_factor
    #        Extra storage needed: 0
    # 2. Full aligned broadcast FA
    #        Use copy_ubuf_to_ubuf to copy all elements block by block
    #        Advantage: Super fast for aligned data
    #        Prerequisite: Aligned data
    #        Cycle is broadcast_src * broadcast_unit * broadcast_factor // repeat_factor // depth
    #        Extra storage needed: 0
    # 3. Semi aligned broadcast SA
    #        Use DSM to align data, then do copy_ub_to_ub and transpose back
    #        Prerequisite: broadcast_unit * broadcast_facotr % align_factor == 0
    #        Cycle is ceil((broadcast_src * broadcast_unit) / (2 * rep))
    #        Plus broadcast_factor * broadcast_src
    #        Plus ceil((broadcast_src * broadcast_unit * broadcast_factor) / (2 * rep))
    #        Extra storage needed: align to 256 Byte
    # 4. Compound scalar move CSM
    ###########################################################################
    # Calculate broadcast information
    dtype_byte_size = get_align_factor(input_buffer.dtype)[1]
    align_factor = 32  # Currently all of our chip requires 32Byte aligned data for performance
    algorithm_score_dict = {
        "DSM": [True, common_unaligned_broadcast, dsm_cycle_estimation],
        "FA": [False, full_aligned_broadcast, fa_cycle_estimation],
        "SA": [False, semi_aligned_broadcast, sa_cycle_estimation],
        "CSM": [False, compound_unaligned_broadcast, csm_cycle_estimation]
    }
    # Rule 1: For aligned data, enable FA
    if int(broadcast_unit * dtype_byte_size % align_factor) == 0:
        algorithm_score_dict["FA"][0] = True
    # Rule 2: For aligned data after broadcast, enable SA
    if int(broadcast_unit * dtype_byte_size * broadcast_factor % align_factor) == 0 or \
            broadcast_src == 1:
        algorithm_score_dict["SA"][0] = True
    # Rule 3: For 16bit aligned data, enable CSM
    if int(broadcast_unit * dtype_byte_size % 2) == 0:
        algorithm_score_dict["CSM"][0] = True
    # Collect all enabled algorithm
    enabled_algorithm = []
    for algorithm_info in algorithm_score_dict.values():
        if algorithm_info[0] is True:
            enabled_algorithm.append(algorithm_info)
    # Final stage: Select and execute algorithm
    if len(enabled_algorithm) == 1:
        result_buffer = enabled_algorithm[0][1](ir_builder, index, input_buffer, output_buffer,
                                                broadcast_src, broadcast_unit, broadcast_factor)
    else:
        best_algorithm = None
        current_cycle = -1
        # Find the best algorithm, only the algorithm with a lower cycle can stay
        for algorithm_info in enabled_algorithm:
            my_cycle = algorithm_info[2](input_buffer.dtype, broadcast_src, broadcast_unit,
                                         broadcast_factor)
            if best_algorithm is None or current_cycle < 0 or my_cycle < current_cycle:
                best_algorithm = algorithm_info[1]
                current_cycle = my_cycle
        result_buffer = best_algorithm(ir_builder, index, input_buffer, output_buffer,
                                       broadcast_src, broadcast_unit, broadcast_factor)
    return result_buffer


def mask_grouping_broadcast(*args):  # pylint: disable=too-many-locals, too-many-statements
    """MGB broadcast algorithm"""
    ir_builder, _, input_tensor, output_tensor, \
        broadcast_src, broadcast_factor = args
    # Dtype
    dtype = input_tensor.dtype
    # Block unit num
    unit_per_block, _ = get_align_factor(dtype)
    # Broadcast 1 to broadcast_len
    broadcast_len = broadcast_factor
    # Broadcast count, number of float you need to broadcast
    broadcast_count = broadcast_src
    # number of elements processed per vector op
    vector_inst_one_repeat_size = te.platform.cce_params.VECTOR_INST_BLOCK_WIDTH \
        // get_align_factor(dtype)[1]

    # Save regs
    current_reg_buf = []
    current_buf_indexes = []

    def read_nums_to_reg(num_indexes, _current_reg_buf, _current_buf_indexes):
        if _current_reg_buf:
            if tuple(_current_buf_indexes) == num_indexes:
                return _current_reg_buf[0]
        _reg = _current_reg_buf[0]
        i = 0
        num_index_dict = {}
        for num_index in num_indexes:
            num_index_dict[num_index] = num_index
        _pattern = find_pattern(num_index_dict)
        if _pattern is None:
            for num_index in num_indexes:
                ir_builder.emit(tvm.call_extern(
                    input_tensor.dtype, "reg_mov",
                    tvm.call_extern(_reg.dtype, "reg", _reg[i]),
                    input_tensor.access_ptr("r", offset=num_index)))
                i += 1
        elif len(num_indexes) == 1:
            ir_builder.emit(tvm.call_extern(
                input_tensor.dtype, "reg_mov",
                tvm.call_extern(_reg.dtype, "reg", _reg[0]),
                input_tensor.access_ptr("r", offset=num_indexes[0])))
        else:
            with ir_builder.for_range(0, len(num_indexes), name="loop_regidx") as loop_idx:
                first_reg = num_indexes[0]
                ir_builder.emit(tvm.call_extern(
                    input_tensor.dtype, "reg_mov",
                    tvm.call_extern(_reg.dtype, "reg", _reg[loop_idx]),
                    input_tensor.access_ptr("r", offset=first_reg + loop_idx * _pattern[0])))
        if _current_reg_buf:
            _current_buf_indexes.clear()
            for i in num_indexes:
                _current_buf_indexes.append(i)
        else:
            _current_reg_buf.append(_reg)
            for i in num_indexes:
                _current_buf_indexes.append(i)
        return _reg

    # Collect all instruction with their mask
    # Example: {mask: {line_index: address}, }
    mask2insn = {}
    for broadcast_index in range(broadcast_count):
        start, mid, end = get_instr(vector_inst_one_repeat_size, broadcast_index,
                                    broadcast_count,
                                    broadcast_len,
                                    unit_per_block)
        if start is not None:
            if not start[1] in mask2insn:
                mask2insn[start[1]] = {}
            mask2insn[start[1]][start[2]] = start[0]
        if mid is not None:
            if not mid[1] in mask2insn:
                mask2insn[mid[1]] = {}
            mask2insn[mid[1]][mid[2]] = mid[0]
        if end is not None:
            if not end[1] in mask2insn:
                mask2insn[end[1]] = {}
            mask2insn[end[1]][end[2]] = end[0]
    # Emit insn
    needed_reg = 0
    for _, addrs in mask2insn.items():
        if len(addrs.keys()) > needed_reg:
            needed_reg = len(addrs.keys())
    reg_buffer = ir_builder.allocate(input_tensor.dtype,
                                     (needed_reg,),
                                     name="reg_buf",
                                     scope=scope_reg)
    current_reg_buf.append(reg_buffer)
    for mask, addrs in mask2insn.items():
        mask_insn = reset_mask_insn
        if mask == 0:
            raise RuntimeError("Broadcast to zero is meaningless")
        if mask < 0:
            mask = -mask
            mask_insn = reset_mask_insn_inverted
        # Get repeat and remain
        rpt = mask // vector_inst_one_repeat_size
        remain = mask % vector_inst_one_repeat_size
        # Maximum mask is vector_inst_one_repeat_size
        mask = min(vector_inst_one_repeat_size, mask)
        # Prepare regs
        reg = read_nums_to_reg(tuple(addrs.keys()), current_reg_buf, current_buf_indexes)
        # Initialize reg index and mask
        addr_index_in_mask = 0
        mask_insn(ir_builder, dtype, bits=mask, ref=unit_per_block)
        # Find pattern if possible
        pattern = find_pattern(addrs)
        broadcast_for_tensor_unaligned_emit_insn((addr_index_in_mask, addrs, dtype, ir_builder,
                                                  mask_insn, [output_tensor], pattern, reg, remain,
                                                  rpt, unit_per_block))
    reset_mask_insn(ir_builder, dtype)
    return output_tensor


def get_instr(vector_inst_one_repeat_size,  # pylint: disable=too-many-locals
              line_index, total_line, unit_per_line, align_factor):
    """Generally, each line needs two instr, broadcast me and remain for next Except last line"""
    # Start address for each line
    start_addr = line_index * unit_per_line
    _remain = start_addr % align_factor
    is_aligned = _remain == 0
    last_aligned_block_index = start_addr // align_factor
    last_aligned = last_aligned_block_index * align_factor
    next_aligned = (last_aligned_block_index + 1) * align_factor
    # Real part
    if is_aligned:
        # Aligned line always start at last_aligned (which is its start addr)
        real_start_addr = last_aligned
        # Aligned line always do full unit
        real_start_mask = unit_per_line
    else:
        # Unaligned line always do partial and start at next_aligned
        real_start_addr = next_aligned
        real_start_mask = start_addr + unit_per_line - next_aligned

    # It is possible for this_line to have tail part
    this_line_rpts = real_start_mask // vector_inst_one_repeat_size
    this_line_tail = real_start_mask % vector_inst_one_repeat_size
    this_line_mask = this_line_rpts * vector_inst_one_repeat_size
    if this_line_rpts == 0:
        this_line_part = None
    else:
        this_line_part = (real_start_addr, this_line_mask, line_index)
    if this_line_tail > 0:
        this_line_tail_start = real_start_addr + this_line_mask
        this_line_tail_part = (this_line_tail_start, this_line_tail, line_index)
    else:
        this_line_tail_part = None
    # tail part for next line
    no_tail = False
    if total_line == line_index + 1:
        # Last line doesn't need to care about next line
        no_tail = True
    elif total_line <= line_index:
        raise RuntimeError("Broadcast unaligned enhancement failure: Line exceeded")
    else:
        # Calculate and see whether next line has a tail
        if (start_addr + unit_per_line) % align_factor == 0:
            no_tail = True
    if no_tail:
        next_line_part = None
    else:
        next_line_start_addr = start_addr + unit_per_line
        next_line_remain = next_line_start_addr % align_factor
        next_line_last_aligned_block_index = next_line_start_addr // align_factor
        next_line_last_aligned = next_line_last_aligned_block_index * align_factor
        tail_mask_inverted = next_line_remain
        next_line_part = (next_line_last_aligned, -tail_mask_inverted, line_index + 1)
    result = (this_line_part, this_line_tail_part, next_line_part)
    return result


def find_pattern(_addrs):
    """This function is used to find the pattern of address"""
    last_addrs = None
    last_index = None
    distance = None
    index_stride = None
    for line_index in _addrs:
        if last_addrs is None:
            last_addrs = _addrs[line_index]
            last_index = line_index
            continue
        if distance is None:
            distance = _addrs[line_index] - last_addrs
            index_stride = line_index - last_index
            last_addrs = _addrs[line_index]
            last_index = line_index
        else:
            current_distance = _addrs[line_index] - last_addrs
            current_stride = line_index - last_index
            if distance == current_distance and index_stride == current_stride:
                last_addrs = _addrs[line_index]
                last_index = line_index
                continue
            else:
                return None
    return distance, index_stride


def broadcast_for_tensor_unaligned_emit_insn(inputs):  # pylint: disable=too-many-locals
    """Emit Insn for the previous function"""
    addr_index_in_mask, addrs, dtype, ir_builder, \
        mask_insn, outs, pattern, reg, remain, rpt, \
        unit_per_block = inputs
    if pattern is None:
        for _, addr in addrs.items():
            if rpt > 0:
                # noinspection PyTypeChecker
                ir_builder.emit(tvm.call_extern
                                (dtype,
                                 'vector_dup',
                                 outs[0].access_ptr("rw", offset=addr),
                                 reg[addr_index_in_mask], rpt, 1, 1, 8, 8))
            if remain > 0:
                mask_insn(ir_builder, dtype, bits=remain, ref=unit_per_block)
                # noinspection PyTypeChecker
                ir_builder.emit(tvm.call_extern
                                (dtype,
                                 'vector_dup',
                                 outs[0].access_ptr("rw", offset=addr),
                                 reg[addr_index_in_mask], 1, 1, 1, 8, 8))
            addr_index_in_mask += 1
    elif len(addrs) == 1:
        first_line = tuple(addrs.keys())[0]
        first_addr = addrs[first_line]
        addr_offset = first_addr
        if rpt > 0:
            # noinspection PyTypeChecker
            ir_builder.emit(tvm.call_extern
                            (dtype,
                             'vector_dup',
                             outs[0].access_ptr("rw", offset=addr_offset),
                             reg[0], rpt, 1, 1, 8, 8))
        if remain > 0:
            mask_insn(ir_builder, dtype, bits=remain, ref=unit_per_block)
            # noinspection PyTypeChecker
            ir_builder.emit(tvm.call_extern
                            (dtype,
                             'vector_dup',
                             outs[0].access_ptr("rw", offset=addr_offset),
                             reg[0], 1, 1, 1, 8, 8))
    else:
        with ir_builder.for_range(0, len(addrs), name="loop_idx") as loop_idx:
            first_line = tuple(addrs.keys())[0]
            first_addr = addrs[first_line]
            addr_offset = first_addr + loop_idx * pattern[0]
            if rpt > 0:
                # noinspection PyTypeChecker
                ir_builder.emit(tvm.call_extern
                                (dtype,
                                 'vector_dup',
                                 outs[0].access_ptr("rw", offset=addr_offset),
                                 reg[loop_idx], rpt, 1, 1, 8, 8))
            if remain > 0:
                mask_insn(ir_builder, dtype, bits=remain, ref=unit_per_block)
                # noinspection PyTypeChecker
                ir_builder.emit(tvm.call_extern
                                (dtype,
                                 'vector_dup',
                                 outs[0].access_ptr("rw", offset=addr_offset),
                                 reg[loop_idx], 1, 1, 1, 8, 8))


def broadcast_last_axis_aligned(*args):  # pylint: disable=too-many-locals
    """LFA broadcast algorithm"""
    ir_builder, _, input_buffer, output_buffer, \
        broadcast_src, broadcast_factor = args
    ins, outs = ([input_buffer], [output_buffer])

    loop_count = broadcast_src
    broadcast_len = broadcast_factor

    intrinsic_cmd = "vector_dup"
    reg = ir_builder.allocate(outs[0].dtype, (1,), name="reg_buf",
                              scope=te.platform.cce_params.scope_reg)
    reset_mask = 1
    with ir_builder.for_range(0, loop_count, name="idx") as idx:
        ir_builder.emit(tvm.call_extern(
            outs[0].dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[0]),
            ins[0].access_ptr("rw", offset=idx), ))

        te.platform.cce_intrin_md.vec_broadcast_opt(ir_builder,
                                                    intrinsic_cmd,
                                                    outs,
                                                    broadcast_len, reset_mask, [reg[0]], idx)
    reset_mask_insn(ir_builder, outs[0].dtype)
    return output_buffer


def common_unaligned_broadcast_last_axis(*args):  # pylint: disable=too-many-locals
    """API for translating last axis DSM to common DSM"""
    ir_builder, index, input_buffer, output_buffer, \
        broadcast_src, broadcast_factor = args
    return common_unaligned_broadcast(ir_builder, index, input_buffer, output_buffer,
                                      broadcast_src, 1, broadcast_factor)


def common_unaligned_broadcast(*args):  # pylint: disable=too-many-locals
    """DSM Broadcast algorithm"""
    ir_builder, _, input_buffer, output_buffer, \
        broadcast_src, broadcast_unit, broadcast_factor = args
    for_loop = broadcast_src
    # Be aware of using too much registers!
    maximum_reg_buffer = 8
    actual_reg_num = None
    for reg_num in range(maximum_reg_buffer, 0, -1):
        if broadcast_src % reg_num == 0 or broadcast_src == reg_num:
            actual_reg_num = reg_num
            break
    reg_buffer = ir_builder.allocate(input_buffer.dtype, (actual_reg_num,),
                                     name="common_reg_buf", scope=scope_reg)
    for_loop = for_loop // actual_reg_num
    if for_loop > 0:
        with ir_builder.for_range(0, for_loop, name="broadcast_unit_index") as idx:
            with ir_builder.for_range(0,
                                      broadcast_unit,
                                      name="broadcast_element_index") as __idx:
                for reg_idx in range(actual_reg_num):
                    src_offset = (broadcast_src - idx - 1) * broadcast_unit \
                                 + __idx \
                                 - reg_idx * broadcast_unit \
                                 - idx * (actual_reg_num - 1) * broadcast_unit
                    src_address = input_buffer.access_ptr("r", offset=src_offset)
                    ir_builder.emit(tvm.call_extern(input_buffer.dtype, "reg_mov",
                                                    tvm.call_extern(reg_buffer.dtype,
                                                                    "reg",
                                                                    reg_buffer[reg_idx]),
                                                    src_address))

                with ir_builder.for_range(0,
                                          broadcast_factor,
                                          name="broadcast_factor_index") as _idx:
                    for reg_idx in range(actual_reg_num):
                        dst_offset = (broadcast_src - idx - 1 - reg_idx) \
                                     * broadcast_unit \
                                     * broadcast_factor \
                                     + _idx * broadcast_unit\
                                     + __idx \
                                     - idx \
                                     * (actual_reg_num - 1) \
                                     * broadcast_factor \
                                     * broadcast_unit
                        dst_address = output_buffer.access_ptr("rw", offset=dst_offset)
                        ir_builder.emit(tvm.call_extern(
                            input_buffer.dtype, "reg_mov",
                            dst_address,
                            tvm.call_extern(reg_buffer.dtype,
                                            "reg",
                                            reg_buffer[reg_idx])))
    return output_buffer


def compound_unaligned_broadcast(*args):  # pylint: disable=too-many-locals
    """CSM Broadcast algorithm"""
    ir_builder, idx, input_buffer, output_buffer, \
        broadcast_src, broadcast_unit, broadcast_factor = args
    # Compound mode selection
    compound_mode_dict = {
        2: "uint16_t",
        4: "int32",
        8: "int64"
    }
    compound_mode = -1
    dtype_size = get_align_factor(input_buffer.dtype)[1]
    for mode in compound_mode_dict:
        if broadcast_unit * dtype_size % mode == 0 and mode > compound_mode:
            compound_mode = mode
    if dtype_size == compound_mode:
        return common_unaligned_broadcast(*args)
    # Compound generation
    compound_factor = compound_mode // dtype_size
    input_buffer_byte_size = get_buffer_shape(input_buffer) * dtype_size
    input_buffer_compound_size = input_buffer_byte_size // compound_mode
    input_buffer_com = tvm.decl_buffer((input_buffer_compound_size,),
                                       compound_mode_dict[compound_mode],
                                       name=input_buffer.name,
                                       data=input_buffer.data,
                                       offset_factor=input_buffer.offset_factor,
                                       data_alignment=input_buffer.data_alignment,
                                       scope=scope_ubuf,
                                       elem_offset=0)
    output_buffer_byte_size = get_buffer_shape(output_buffer) * dtype_size
    output_buffer_compound_size = output_buffer_byte_size // compound_mode
    output_buffer_com = tvm.decl_buffer((output_buffer_compound_size,),
                                        compound_mode_dict[compound_mode],
                                        name=output_buffer.name,
                                        data=output_buffer.data,
                                        offset_factor=output_buffer.offset_factor,
                                        data_alignment=output_buffer.data_alignment,
                                        scope=scope_ubuf,
                                        elem_offset=0)
    # Do DSM with generated compound
    common_unaligned_broadcast(ir_builder, idx, input_buffer_com, output_buffer_com,
                               broadcast_src, broadcast_unit // compound_factor,
                               broadcast_factor)
    return output_buffer


def full_aligned_broadcast(*args, remain=0):  # pylint: disable=too-many-locals
    """FA Broadcast algorithm"""
    ir_builder, _, input_buffer, output_buffer, \
        broadcast_src, broadcast_unit, broadcast_factor = args
    dtype_block_size, dtype_byte_size = get_align_factor(input_buffer.dtype)
    # Check if aligned
    if broadcast_unit % dtype_block_size != 0 or remain % dtype_block_size != 0:
        if broadcast_src != 1:
            raise RuntimeError("Full aligned broadcast supports aligned broadcast only!")
    # For larger broadcast_unit, use copy_ubuf_to_ubuf
    # For larger broadcast_factor, use vor
    # For larget broadcast_src, use enhanced vor
    no_enhanced = False
    if remain > 0:
        no_enhanced = True
    selected = full_aligned_broadcast_selection(broadcast_src, broadcast_unit, broadcast_factor,
                                                input_buffer.dtype, no_enhanced=no_enhanced)
    if selected == DMA_MODE:
        return full_aligned_broadcast_dma(*args, remain=remain)
    if selected == VECTOR_MODE:
        return full_aligned_broadcast_vector(*args, remain=remain)
    if input_buffer != output_buffer:
        return full_aligned_broadcast_vector_enhanced(*args)
    print("[BroadcastIntrin][Warning] FA broadcast vector enhanced mode disabled!")
    no_enhanced = True
    if full_aligned_broadcast_selection(broadcast_src, broadcast_unit, broadcast_factor,
                                        input_buffer.dtype,
                                        no_enhanced=no_enhanced) == DMA_MODE:
        return full_aligned_broadcast_dma(*args, remain=remain)
    return full_aligned_broadcast_vector(*args, remain=remain)


def full_aligned_broadcast_selection(broadcast_src, broadcast_unit, broadcast_factor,
                                     dtype, no_enhanced=False):
    """Determine possible FA broadcast algorithm"""
    dtype_block_size, dtype_byte_size = get_align_factor(dtype)
    vector_insn_one_repeat_size = te.platform.cce_params.VECTOR_INST_BLOCK_WIDTH // dtype_byte_size
    std_src = broadcast_unit // dtype_block_size * broadcast_factor
    std_unit = broadcast_src * broadcast_factor
    std_factor = math.ceil(broadcast_unit / vector_insn_one_repeat_size) * broadcast_src
    if no_enhanced or not (is_support_full_aligned_broadcast_vector_enhanced(broadcast_unit,
                                                                             broadcast_factor,
                                                                             dtype)):
        std_src = 99999999999999
    std_min = min(std_src, std_unit, std_factor)
    if std_min == std_src:
        return VECTOR_ENHANCED_MODE
    if std_min == std_unit:
        return DMA_MODE
    if std_min == std_factor:
        return VECTOR_MODE
    raise RuntimeError("FA broadcast selection failed")


def is_support_full_aligned_broadcast_vector_enhanced(broadcast_unit, broadcast_factor, dtype):
    """Check if enhanced vector mode FA broadcast is supported"""
    _, dtype_byte_size = get_align_factor(dtype)
    b16_factor = dtype_byte_size // 2
    broadcast_unit_16 = broadcast_unit * b16_factor
    block_size_16 = 16
    broadcast_unit_16_block_num = broadcast_unit_16 // block_size_16  # 32Byte Block -> 16 b16
    if broadcast_unit_16_block_num * 8 * broadcast_factor <= UINT8_MAXIMUM:
        return True
    return False


def full_aligned_broadcast_dma(*args, remain):  # pylint: disable=too-many-locals
    """Use copy_ubuf_to_ubuf when broadcast_unit is larger"""
    ir_builder, _, input_buffer, output_buffer, \
        broadcast_src, broadcast_unit, broadcast_factor = args
    dtype_block_size, dtype_byte_size = get_align_factor(input_buffer.dtype)
    block_num = broadcast_unit // dtype_block_size
    for_loop = broadcast_src
    # vor needs to treat original and target buffer as binary16
    b16_factor = dtype_byte_size // 2
    input_buffer_1d_size = get_buffer_shape(input_buffer)
    input_buffer_16_size = input_buffer_1d_size * b16_factor
    input_buffer_16 = tvm.decl_buffer((input_buffer_16_size,), "uint16_t",
                                      name=input_buffer.name,
                                      data=input_buffer.data,
                                      offset_factor=input_buffer.offset_factor,
                                      data_alignment=input_buffer.data_alignment,
                                      scope=scope_ubuf,
                                      elem_offset=0)
    output_buffer_1d_size = get_buffer_shape(output_buffer)
    output_buffer_16_size = output_buffer_1d_size * b16_factor
    output_buffer_16 = tvm.decl_buffer((output_buffer_16_size,), "uint16_t",
                                       name=output_buffer.name,
                                       data=output_buffer.data,
                                       offset_factor=output_buffer.offset_factor,
                                       data_alignment=output_buffer.data_alignment,
                                       scope=scope_ubuf,
                                       elem_offset=0)
    vor_buffer = apply_for_new_alloc(ir_builder, "uint16_t",
                                     (128,),
                                     scope_ubuf)
    init_vor_buffer(ir_builder, vor_buffer)

    if input_buffer == output_buffer:
        for_loop -= 1
    if for_loop > 0:
        with ir_builder.for_range(0, for_loop, name="broadcast_src_index") as idx:
            with ir_builder.for_range(0, broadcast_factor, name="broadcast_factor_index") as _idx:
                src_address = input_buffer.access_ptr("r", offset=(broadcast_src - idx - 1)
                                                      * broadcast_unit)
                dst_address = output_buffer.access_ptr("rw", offset=(broadcast_src - idx - 1)
                                                       * (broadcast_unit * broadcast_factor
                                                          + remain)
                                                       + _idx * broadcast_unit)
                # noinspection PyTypeChecker
                ir_builder.emit(tvm.call_extern(
                    output_buffer.dtype, 'copy_ubuf_to_ubuf',
                    dst_address, src_address,
                    0, 1, block_num, 0, 0))
            if remain > 0:
                vector_insn_factory_normal(ir_builder, "vor", output_buffer_16,
                                           input_buffer_16, vor_buffer, remain * b16_factor,
                                           src1_stride=0, src1_repstr=0,
                                           src_offset=(broadcast_src - idx - 1)
                                           * broadcast_unit * b16_factor,
                                           dst_offset=(broadcast_src - idx - 1)
                                           * (broadcast_unit * broadcast_factor * b16_factor
                                              + remain * b16_factor)
                                           + broadcast_factor * broadcast_unit * b16_factor)
    if input_buffer == output_buffer:
        if remain > 0:
            # noinspection PyTypeChecker
            vector_insn_factory_normal(ir_builder, "vor", output_buffer_16,
                                       input_buffer_16, vor_buffer, remain * b16_factor,
                                       src1_stride=0, src1_repstr=0,
                                       dst_offset=broadcast_factor * broadcast_unit * b16_factor)
        with ir_builder.for_range(0,
                                  broadcast_factor - 1,
                                  name="broadcast_factor_index") as _idx:
            src_address = input_buffer.access_ptr("r",
                                                  offset=0)
            dst_address = output_buffer.access_ptr("rw",
                                                   offset=(_idx + 1) * broadcast_unit)
            # noinspection PyTypeChecker
            ir_builder.emit(tvm.call_extern(
                output_buffer.dtype, 'copy_ubuf_to_ubuf',
                dst_address, src_address,
                0, 1, block_num, 0, 0))
    reset_mask_insn(ir_builder, output_buffer.dtype)
    return output_buffer


def full_aligned_broadcast_vector(*args, remain=0):  # pylint: disable=too-many-locals
    """Use vor when broadcast_factor is larger"""
    ir_builder, _, input_buffer, output_buffer, \
        broadcast_src, broadcast_unit, broadcast_factor = args
    _, dtype_byte_size = get_align_factor(input_buffer.dtype)
    for_loop = broadcast_src
    # vor needs to treat original and target buffer as binary16
    b16_factor = dtype_byte_size // 2
    input_buffer_1d_size = get_buffer_shape(input_buffer)
    input_buffer_16_size = input_buffer_1d_size * b16_factor
    input_buffer_16 = tvm.decl_buffer((input_buffer_16_size,), "uint16_t",
                                      name=input_buffer.name,
                                      data=input_buffer.data,
                                      offset_factor=input_buffer.offset_factor,
                                      data_alignment=input_buffer.data_alignment,
                                      scope=scope_ubuf,
                                      elem_offset=0)
    output_buffer_1d_size = get_buffer_shape(output_buffer)
    output_buffer_16_size = output_buffer_1d_size * b16_factor
    output_buffer_16 = tvm.decl_buffer((output_buffer_16_size,), "uint16_t",
                                       name=output_buffer.name,
                                       data=output_buffer.data,
                                       offset_factor=output_buffer.offset_factor,
                                       data_alignment=output_buffer.data_alignment,
                                       scope=scope_ubuf,
                                       elem_offset=0)
    vor_buffer = apply_for_new_alloc(ir_builder, "uint16_t",
                                     (128,),
                                     scope_ubuf)
    init_vor_buffer(ir_builder, vor_buffer)
    # Calculate vector instruction parameters
    broadcast_unit_16 = broadcast_unit * b16_factor
    remain_16 = remain * b16_factor
    broadcast_unit_block_size = broadcast_unit_16 // 16  # 32Byte Block -> 16 b16
    vector_insn_one_repeat_size_16 = te.platform.cce_params.VECTOR_INST_BLOCK_WIDTH // 2
    num_of_instruction = broadcast_unit_16 // vector_insn_one_repeat_size_16
    remain_instruction_size = broadcast_unit_16 % vector_insn_one_repeat_size_16
    if for_loop > 0:
        with ir_builder.for_range(0, for_loop - 1, name="broadcast_src_index") as idx:
            if remain > 0:
                vector_insn_factory_normal(ir_builder, "vor", output_buffer_16,
                                           input_buffer_16, vor_buffer, remain_16,
                                           src1_stride=0, src1_repstr=0,
                                           src_offset=(broadcast_src - idx - 1)
                                           * broadcast_unit_16,
                                           dst_offset=(broadcast_src - idx - 1)
                                           * (broadcast_unit_16 * broadcast_factor + remain_16)
                                           + broadcast_factor * broadcast_unit_16)
            if remain_instruction_size > 0:
                vector_insn_factory_normal(ir_builder, "vor", output_buffer_16,
                                           input_buffer_16, vor_buffer,
                                           remain_instruction_size * broadcast_factor,
                                           src1_stride=0, dst_repstr=broadcast_unit_block_size,
                                           src_repstr=0, src1_repstr=0,
                                           src_offset=(broadcast_src - idx)
                                           * broadcast_unit_16
                                           - remain_instruction_size,
                                           dst_offset=(broadcast_src - idx - 1)
                                           * (broadcast_unit_16 * broadcast_factor + remain_16)
                                           + num_of_instruction * vector_insn_one_repeat_size_16,
                                           _repeat_size=remain_instruction_size)
            if num_of_instruction > 0:
                for i in range(num_of_instruction):
                    vector_insn_factory_normal(ir_builder, "vor", output_buffer_16,
                                               input_buffer_16, vor_buffer,
                                               vector_insn_one_repeat_size_16 * broadcast_factor,
                                               src1_stride=0, dst_repstr=broadcast_unit_block_size,
                                               src_repstr=0, src1_repstr=0,
                                               src_offset=(broadcast_src - idx)
                                               * broadcast_unit_16
                                               - (i + 1)
                                               * vector_insn_one_repeat_size_16
                                               - remain_instruction_size,
                                               dst_offset=(broadcast_src - idx - 1)
                                               * (broadcast_unit_16 * broadcast_factor + remain_16)
                                               + (num_of_instruction - i - 1)
                                               * vector_insn_one_repeat_size_16)
        idx = for_loop - 1
        if remain > 0:
            vector_insn_factory_normal(ir_builder, "vor", output_buffer_16,
                                       input_buffer_16, vor_buffer, remain_16,
                                       src1_stride=0, src1_repstr=0,
                                       src_offset=(broadcast_src - idx - 1)
                                       * broadcast_unit_16,
                                       dst_offset=(broadcast_src - idx - 1)
                                       * (broadcast_unit_16 * broadcast_factor + remain_16)
                                       + broadcast_factor * broadcast_unit_16)
        if remain_instruction_size > 0:
            vector_insn_factory_normal(ir_builder, "vor", output_buffer_16,
                                       input_buffer_16, vor_buffer,
                                       remain_instruction_size * broadcast_factor,
                                       src1_stride=0, dst_repstr=broadcast_unit_block_size,
                                       src_repstr=0, src1_repstr=0,
                                       src_offset=(broadcast_src - idx)
                                       * broadcast_unit_16
                                       - remain_instruction_size,
                                       dst_offset=(broadcast_src - idx - 1)
                                       * (broadcast_unit_16 * broadcast_factor + remain_16)
                                       + num_of_instruction * vector_insn_one_repeat_size_16,
                                       _repeat_size=remain_instruction_size)
        if num_of_instruction > 0:
            for i in range(num_of_instruction):
                vector_insn_factory_normal(ir_builder, "vor", output_buffer_16,
                                           input_buffer_16, vor_buffer,
                                           vector_insn_one_repeat_size_16 * broadcast_factor,
                                           src1_stride=0, dst_repstr=broadcast_unit_block_size,
                                           src_repstr=0, src1_repstr=0,
                                           src_offset=(broadcast_src - idx)
                                           * broadcast_unit_16
                                           - (i + 1)
                                           * vector_insn_one_repeat_size_16
                                           - remain_instruction_size,
                                           dst_offset=(broadcast_src - idx - 1)
                                           * (broadcast_unit_16 * broadcast_factor + remain_16)
                                           + (num_of_instruction - i - 1)
                                           * vector_insn_one_repeat_size_16)
    return output_buffer


def full_aligned_broadcast_vector_enhanced(*args):  # pylint: disable=too-many-locals
    """Use vor enhanced when broadcast_src is larger"""
    ir_builder, _, input_buffer, output_buffer, \
        broadcast_src, broadcast_unit, broadcast_factor = args
    _, dtype_byte_size = get_align_factor(input_buffer.dtype)
    if input_buffer == output_buffer:
        raise RuntimeError("Full aligned broadcast vector enhanced mode needs safety buffer")
    for_loop = broadcast_factor
    # vor needs to treat original and target buffer as binary16
    b16_factor = dtype_byte_size // 2
    input_buffer_1d_size = get_buffer_shape(input_buffer)
    input_buffer_16_size = input_buffer_1d_size * b16_factor
    input_buffer_16 = tvm.decl_buffer((input_buffer_16_size,), "uint16_t",
                                      name=input_buffer.name,
                                      data=input_buffer.data,
                                      offset_factor=input_buffer.offset_factor,
                                      data_alignment=input_buffer.data_alignment,
                                      scope=scope_ubuf,
                                      elem_offset=0)
    output_buffer_1d_size = get_buffer_shape(output_buffer)
    output_buffer_16_size = output_buffer_1d_size * b16_factor
    output_buffer_16 = tvm.decl_buffer((output_buffer_16_size,), "uint16_t",
                                       name=output_buffer.name,
                                       data=output_buffer.data,
                                       offset_factor=output_buffer.offset_factor,
                                       data_alignment=output_buffer.data_alignment,
                                       scope=scope_ubuf,
                                       elem_offset=0)
    vor_buffer = apply_for_new_alloc(ir_builder, "uint16_t",
                                     (128,),
                                     scope_ubuf)
    init_vor_buffer(ir_builder, vor_buffer)
    # Calculate vector instruction parameters
    broadcast_unit_16 = broadcast_unit * b16_factor
    block_size_16 = 16
    broadcast_unit_16_block_num = broadcast_unit_16 // block_size_16  # 32Byte Block -> 16 b16
    vector_insn_one_repeat_size_16 = te.platform.cce_params.VECTOR_INST_BLOCK_WIDTH // 2
    num_of_instruction = broadcast_unit_16 // block_size_16
    num_of_repeat = broadcast_src * block_size_16 // vector_insn_one_repeat_size_16
    remain_repeat_size = broadcast_src * block_size_16 % vector_insn_one_repeat_size_16
    if for_loop > 0:
        with ir_builder.for_range(0, for_loop, name="broadcast_factor_index") as idx:
            if remain_repeat_size > 0:
                for i in range(num_of_instruction):
                    vector_insn_factory_normal(ir_builder, "vor", output_buffer_16,
                                               input_buffer_16, vor_buffer,
                                               remain_repeat_size,
                                               dst_stride=broadcast_unit_16_block_num
                                               * broadcast_factor,
                                               src_stride=broadcast_unit_16_block_num,
                                               src1_stride=0,
                                               dst_repstr=0,
                                               src_repstr=0,
                                               src1_repstr=0,
                                               src_offset=num_of_repeat
                                               * vector_insn_one_repeat_size_16 * num_of_instruction
                                               + i * block_size_16,
                                               dst_offset=num_of_repeat
                                               * vector_insn_one_repeat_size_16 * broadcast_factor
                                               * num_of_instruction
                                               + i * block_size_16
                                               + idx * broadcast_unit_16,
                                               _repeat_size=remain_repeat_size)
            for i in range(num_of_instruction):
                vector_insn_factory_normal(ir_builder, "vor", output_buffer_16,
                                           input_buffer_16, vor_buffer,
                                           num_of_repeat * vector_insn_one_repeat_size_16,
                                           dst_stride=broadcast_unit_16_block_num
                                           * broadcast_factor,
                                           src_stride=broadcast_unit_16_block_num,
                                           src1_stride=0,
                                           dst_repstr=broadcast_unit_16_block_num * 8
                                           * broadcast_factor,
                                           src_repstr=broadcast_unit_16_block_num * 8,
                                           src1_repstr=0,
                                           src_offset=i * block_size_16,
                                           dst_offset=i * block_size_16
                                           + idx * broadcast_unit_16)
    return output_buffer


def semi_aligned_broadcast(*args):  # pylint: disable=too-many-locals, too-many-statements
    """SA Broadcast algorithm"""
    ir_builder, index, input_buffer, output_buffer, \
        broadcast_src, broadcast_unit, broadcast_factor = args
    # Get block size of input tensor
    dtype_block_size, dtype_byte_size = get_align_factor(input_buffer.dtype)
    # Get least common multiple for alignment factor
    least_common_multiple = get_least_common_multiple(broadcast_unit, dtype_block_size)
    factor_after = math.ceil(least_common_multiple / broadcast_unit)
    # Get full aligned parameters
    remain = (broadcast_unit * broadcast_factor) % least_common_multiple
    repeat_time = (broadcast_unit * broadcast_factor) // least_common_multiple
    # Declare for extra buffer if it is possible to select FA vector_enhanced algorithm
    selection = full_aligned_broadcast_selection(broadcast_src, least_common_multiple, repeat_time,
                                                 input_buffer.dtype)
    original_output_buffer = output_buffer
    if selection != DMA_MODE or \
            selection != VECTOR_MODE:
        output_buffer = apply_for_new_alloc(ir_builder, input_buffer.dtype,
                                            (broadcast_src * broadcast_unit * factor_after,),
                                            scope_ubuf)

    # Do DSM for alignment, broadcast broadcast_unit to least_common_multiple
    # Use CSM if possible
    if int(broadcast_unit * dtype_byte_size % 2) == 0:
        input_buffer = compound_unaligned_broadcast(ir_builder, index, input_buffer, output_buffer,
                                                    broadcast_src, broadcast_unit, factor_after)
    else:
        input_buffer = common_unaligned_broadcast(ir_builder, index, input_buffer, output_buffer,
                                                  broadcast_src, broadcast_unit, factor_after)
    # Do full aligned
    full_aligned_broadcast(ir_builder, index, input_buffer, original_output_buffer,
                           broadcast_src, least_common_multiple, repeat_time, remain=remain)
    return original_output_buffer


def last_axis_transpose_broadcast(*args):  # pylint: disable=too-many-locals, too-many-statements
    """Last Axis Transpose Broadcast algorithm"""
    ir_builder, index, input_buffer, output_buffer, \
        broadcast_src, broadcast_factor = args
    # Get original buffer dtype factor compared with binary16
    input_dtype_block_size, input_dtype_byte_size = get_align_factor(input_buffer.dtype)
    dtype_factor = input_dtype_byte_size // get_align_factor("uint16")[1]
    if dtype_factor != 1:
        raise RuntimeError("LAT broadcast algorithm supports binary16 only")
    # Get original buffer block num
    unit_block_num = broadcast_src // input_dtype_block_size
    unit_remain = broadcast_src - unit_block_num * input_dtype_block_size
    repeat_time = broadcast_src // input_dtype_block_size
    # vtranspose needs to treat original buffer as binary16
    input_buffer_1d_size = get_buffer_shape(input_buffer)
    input_buffer_16_size = input_buffer_1d_size
    input_buffer_16 = tvm.decl_buffer((input_buffer_16_size,), "uint16_t",
                                      name=input_buffer.name,
                                      data=input_buffer.data,
                                      offset_factor=input_buffer.offset_factor,
                                      data_alignment=input_buffer.data_alignment,
                                      scope=scope_ubuf,
                                      elem_offset=0)
    output_buffer_1d_size = get_buffer_shape(output_buffer)
    output_buffer_16_size = output_buffer_1d_size
    output_buffer_16 = tvm.decl_buffer((output_buffer_16_size,), "uint16_t",
                                       name=output_buffer.name,
                                       data=output_buffer.data,
                                       offset_factor=output_buffer.offset_factor,
                                       data_alignment=output_buffer.data_alignment,
                                       scope=scope_ubuf,
                                       elem_offset=0)
    mid_buffer = apply_for_new_alloc(ir_builder, "uint16_t",
                                     (16, 16),
                                     scope_ubuf)
    mid_buffer_target = apply_for_new_alloc(ir_builder, "uint16_t",
                                            (16, 16, broadcast_factor),
                                            scope_ubuf)
    vor_buffer = apply_for_new_alloc(ir_builder, "uint16_t",
                                     (128,),
                                     scope_ubuf)
    # Do emit insn
    reset_mask_insn(ir_builder, "uint16_t")
    # Use vor if dtype factor is 1
    init_vor_buffer(ir_builder, vor_buffer)
    if repeat_time > 0:
        safe_const = 256
        safe_repeat_need = math.ceil(safe_const / broadcast_factor / 16)
        safe_repeat_threshold = repeat_time - safe_repeat_need
        if safe_repeat_threshold > 0:
            with ir_builder.for_range(0, safe_repeat_threshold, name="transpose_counter") as \
                    transpose_counter:
                ir_builder.emit(tvm.call_extern(
                    "uint16_t", "vtranspose",
                    mid_buffer.access_ptr("rw", offset=0),
                    input_buffer_16.access_ptr("r", offset=transpose_counter * 16)
                ))
                # Do broadcast with vand when dtype factor is 1
                # noinspection PyTypeChecker
                ir_builder.emit(tvm.call_extern(
                    "uint16_t", "vor",
                    mid_buffer_target.access_ptr("rw", offset=0),
                    mid_buffer.access_ptr("r", offset=0),
                    vor_buffer.access_ptr("r", offset=0),
                    broadcast_factor, broadcast_factor, 1, 0, 1, 0, 0
                ))
                # noinspection PyTypeChecker
                ir_builder.emit(tvm.call_extern(
                    "uint16_t", "vor",
                    mid_buffer_target.access_ptr("rw", offset=broadcast_factor * 16 * 8),
                    mid_buffer.access_ptr("r", offset=8 * 16),
                    vor_buffer.access_ptr("r", offset=0),
                    broadcast_factor, broadcast_factor, 1, 0, 1, 0, 0
                ))
                # Transpose back
                for i in range(broadcast_factor):
                    ir_builder.emit(tvm.call_extern(
                        "uint16_t", "vtranspose",
                        output_buffer_16.access_ptr("rw", offset=transpose_counter * 16
                                                    * broadcast_factor + i * 16),
                        mid_buffer_target.access_ptr("r", offset=i * 16 * 16)
                    ))
        if safe_repeat_threshold <= 0:
            with ir_builder.for_range(0, repeat_time, name="transpose_counter") as \
                    transpose_counter:
                ir_builder.emit(tvm.call_extern(
                    "uint16_t", "vtranspose",
                    mid_buffer.access_ptr("rw", offset=0),
                    input_buffer_16.access_ptr("r", offset=transpose_counter * 16)
                ))
                # Do broadcast with vand when dtype factor is 1
                # noinspection PyTypeChecker
                ir_builder.emit(tvm.call_extern(
                    "uint16_t", "vor",
                    mid_buffer_target.access_ptr("rw", offset=0),
                    mid_buffer.access_ptr("r", offset=0),
                    vor_buffer.access_ptr("r", offset=0),
                    broadcast_factor, broadcast_factor, 1, 0, 1, 0, 0
                ))
                # noinspection PyTypeChecker
                ir_builder.emit(tvm.call_extern(
                    "uint16_t", "vor",
                    mid_buffer_target.access_ptr("rw", offset=broadcast_factor * 16 * 8),
                    mid_buffer.access_ptr("r", offset=8 * 16),
                    vor_buffer.access_ptr("r", offset=0),
                    broadcast_factor, broadcast_factor, 1, 0, 1, 0, 0
                ))
                # Transpose back
                for i in range(broadcast_factor):
                    ir_builder.emit(tvm.call_extern(
                        "uint16_t", "vtranspose",
                        mid_buffer_target.access_ptr("rw", offset=i * 16),
                        mid_buffer_target.access_ptr("r", offset=i * 16 * 16)
                    ))
                # Move back
                vector_insn_factory_normal(ir_builder, "vor", output_buffer_16,
                                           mid_buffer_target, vor_buffer, broadcast_factor * 16,
                                           src1_stride=0, src1_repstr=0,
                                           dst_offset=transpose_counter * 16
                                           * broadcast_factor)
        elif safe_repeat_threshold > 0:
            for transpose_counter in range(safe_repeat_threshold, repeat_time, 1):
                ir_builder.emit(tvm.call_extern(
                    "uint16_t", "vtranspose",
                    mid_buffer.access_ptr("rw", offset=0),
                    input_buffer_16.access_ptr("r", offset=transpose_counter * 16)
                ))
                # Do broadcast with vand when dtype factor is 1
                # noinspection PyTypeChecker
                ir_builder.emit(tvm.call_extern(
                    "uint16_t", "vor",
                    mid_buffer_target.access_ptr("rw", offset=0),
                    mid_buffer.access_ptr("r", offset=0),
                    vor_buffer.access_ptr("r", offset=0),
                    broadcast_factor, broadcast_factor, 1, 0, 1, 0, 0
                ))
                # noinspection PyTypeChecker
                ir_builder.emit(tvm.call_extern(
                    "uint16_t", "vor",
                    mid_buffer_target.access_ptr("rw", offset=broadcast_factor * 16 * 8),
                    mid_buffer.access_ptr("r", offset=8 * 16),
                    vor_buffer.access_ptr("r", offset=0),
                    broadcast_factor, broadcast_factor, 1, 0, 1, 0, 0
                ))
                # Transpose back
                for i in range(broadcast_factor):
                    ir_builder.emit(tvm.call_extern(
                        "uint16_t", "vtranspose",
                        mid_buffer_target.access_ptr("rw", offset=i * 16),
                        mid_buffer_target.access_ptr("r", offset=i * 16 * 16)
                    ))
                # Move back
                vector_insn_factory_normal(ir_builder, "vor", output_buffer_16,
                                           mid_buffer_target, vor_buffer, broadcast_factor * 16,
                                           src1_stride=0, src1_repstr=0,
                                           dst_offset=transpose_counter * 16
                                           * broadcast_factor)
    if unit_remain > 0:
        ir_builder.emit(tvm.call_extern(
            "uint16_t", "vtranspose",
            mid_buffer.access_ptr("rw", offset=0),
            input_buffer_16.access_ptr("r", offset=repeat_time * 16)
        ))
        # Do broadcast with vand when dtype factor is 1
        # noinspection PyTypeChecker
        ir_builder.emit(tvm.call_extern(
            "uint16_t", "vor",
            mid_buffer_target.access_ptr("rw", offset=0),
            mid_buffer.access_ptr("r", offset=0),
            vor_buffer.access_ptr("r", offset=0),
            broadcast_factor, broadcast_factor, 1, 0, 1, 0, 0
        ))
        # noinspection PyTypeChecker
        ir_builder.emit(tvm.call_extern(
            "uint16_t", "vor",
            mid_buffer_target.access_ptr("rw", offset=broadcast_factor * 16 * 8),
            mid_buffer.access_ptr("r", offset=8 * 16),
            vor_buffer.access_ptr("r", offset=0),
            broadcast_factor, broadcast_factor, 1, 0, 1, 0, 0
        ))
        # Transpose back
        for i in range(broadcast_factor):
            ir_builder.emit(tvm.call_extern(
                "uint16_t", "vtranspose",
                mid_buffer_target.access_ptr("rw", offset=i * 16),
                mid_buffer_target.access_ptr("r", offset=i * 16 * 16)
            ))
        # Move back
        vector_insn_factory_normal(ir_builder, "vor", output_buffer_16,
                                   mid_buffer_target, vor_buffer, broadcast_factor * unit_remain,
                                   src1_stride=0, src1_repstr=0,
                                   dst_offset=repeat_time * 16
                                   * broadcast_factor)
    return output_buffer


def init_vor_buffer(ir_builder, buffer):
    """Initialize buffer for vand move"""
    # noinspection PyTypeChecker
    ir_builder.emit(tvm.call_extern(
        "uint16_t", "vector_dup",
        buffer.access_ptr("rw", offset=0),
        tvm.const(0, "uint16_t"),
        1, 1, 1, 8, 8
    ))


def get_buffer_shape(buffer):
    """Get 1D size of a buffer"""
    size = 1
    for i in buffer.shape:
        size *= i
    return size


def vector_insn_factory_normal(ir_b, cmd, dst_buffer, src_buffer,  # pylint: disable=R0913, R0914
                               src1_buffer, elem, dst_stride=1, src_stride=1, src1_stride=1,
                               dst_repstr=8, src_repstr=8, src1_repstr=8,
                               dst_offset=0, src_offset=0, src1_offset=0, _repeat_size=128):
    """Generate normal vector intrin, factory function"""
    block_size, dtype_size = get_align_factor(src_buffer.dtype)
    repeat_size = min(te.platform.cce_params.VECTOR_INST_BLOCK_WIDTH // dtype_size, _repeat_size)
    repeat = elem // repeat_size
    rem = elem - repeat * repeat_size
    # Remain part
    if rem > 0:
        reset_mask_insn(ir_b, dst_buffer.dtype, bits=rem)
        # noinspection PyTypeChecker
        ir_b.emit(tvm.call_extern(
            dst_buffer.dtype,
            cmd,
            dst_buffer.access_ptr("rw", offset=repeat * dst_repstr * block_size
                                  + dst_offset),
            src_buffer.access_ptr("rw", offset=repeat * src_repstr * block_size
                                  + src_offset),
            src1_buffer.access_ptr("r", offset=repeat * src1_repstr * block_size
                                   + src1_offset),
            1, dst_stride, src_stride, src1_stride, dst_repstr, src_repstr, src1_repstr))
    if repeat > 255:
        reset_mask_insn(ir_b, dst_buffer.dtype, bits=repeat_size)
        outer_repeat_times = repeat // 255
        remain_repeat_times = repeat % 255
        for outer_repeat in range(outer_repeat_times):
            # noinspection PyTypeChecker
            ir_b.emit(tvm.call_extern(
                dst_buffer.dtype,
                cmd,
                dst_buffer.access_ptr("rw", offset=outer_repeat * 255
                                      * dst_repstr * block_size
                                      + dst_offset),
                src_buffer.access_ptr("rw", offset=outer_repeat * 255
                                      * src_repstr * block_size
                                      + src_offset),
                src1_buffer.access_ptr("r", offset=outer_repeat * 255
                                       * src1_repstr * block_size
                                       + src1_offset),
                255, dst_stride, src_stride, src1_stride, dst_repstr, src_repstr, src1_repstr))
        # noinspection PyTypeChecker
        ir_b.emit(tvm.call_extern(
            dst_buffer.dtype,
            cmd,
            dst_buffer.access_ptr("rw", offset=outer_repeat_times * 255
                                  * dst_repstr * block_size
                                  + dst_offset),
            src_buffer.access_ptr("rw", offset=outer_repeat_times * 255
                                  * src_repstr * block_size
                                  + src_offset),
            src1_buffer.access_ptr("r", offset=outer_repeat_times * 255
                                   * src1_repstr * block_size
                                   + src1_offset),
            remain_repeat_times, dst_stride, src_stride, src1_stride, dst_repstr, src_repstr,
            src1_repstr))
    elif repeat > 0:
        reset_mask_insn(ir_b, dst_buffer.dtype, bits=repeat_size)
        # noinspection PyTypeChecker
        ir_b.emit(tvm.call_extern(
            dst_buffer.dtype,
            cmd,
            dst_buffer.access_ptr("rw", offset=0 + dst_offset),
            src_buffer.access_ptr("rw", offset=0 + src_offset),
            src1_buffer.access_ptr("r", offset=0 + src1_offset),
            repeat, dst_stride, src_stride, src1_stride, dst_repstr, src_repstr, src1_repstr))
    reset_mask_insn(ir_b, dst_buffer.dtype)


def lfa_cycle_estimation(dtype, broadcast_src, broadcast_unit, broadcast_factor):
    """Estimation function for lfa(last_axis) algorithm cycle performance"""
    list([broadcast_unit]).clear()  # Use it once to avoid static checks
    dtype_byte_size = get_align_factor(dtype)[1]
    vector_repeat_size = te.platform.cce_params.VECTOR_INST_BLOCK_WIDTH // dtype_byte_size
    # Base cycle: read each broadcast source number into register, each num costs 1 cycle
    cycle_base = broadcast_src
    # Action cycle, broadcast one number to target shape using masked vector_dup
    cycle_action = math.ceil(broadcast_factor / vector_repeat_size / 2)
    # For cycle, do action cycle for each number
    cycle_for = cycle_action * broadcast_src
    # Total cycle
    cycle_total = cycle_for + cycle_base
    return cycle_total


def mg_cycle_estimation(dtype, broadcast_src, broadcast_unit, broadcast_factor):
    """Estimation function for mg(last_axis) algorithm cycle performance"""
    list([broadcast_unit]).clear()  # Use it once to avoid static checks
    dtype_byte_size = get_align_factor(dtype)[1]
    vector_repeat_size = te.platform.cce_params.VECTOR_INST_BLOCK_WIDTH // dtype_byte_size
    # Base cycle: read each broadcast source number into register, each num costs 1 cycle
    cycle_base = broadcast_src
    # Action cycle, broadcast one number to target shape using masked vector_dup
    # This algorithm usually deals with unaligned data, default to having remain part
    rpt_num = math.ceil(math.ceil(broadcast_factor / vector_repeat_size) / 2)
    main_action = broadcast_src * rpt_num
    remain_action = broadcast_src
    cycle_total = cycle_base + main_action + remain_action
    return cycle_total


def dsm_cycle_estimation(dtype, broadcast_src, broadcast_unit, broadcast_factor):
    """Estimation function for dsm algorithm cycle performance"""
    list([dtype]).clear()  # Use it once to avoid static checks
    # Base cycle: read each broadcast source number into register, each num costs 1 cycle
    cycle_base = broadcast_src
    # Action cycle, broadcast one number to target shape using reg_mov
    cycle_action = broadcast_factor * broadcast_unit
    # For cycle, do action cycle for each number
    cycle_for = cycle_action * broadcast_src
    # Total cycle
    cycle_total = cycle_for + cycle_base
    return cycle_total


def lat_cycle_estimation(dtype, broadcast_src, broadcast_unit, broadcast_factor):
    """Estimation function for lat(last_axis) algorithm cycle performance"""
    if broadcast_unit != 1:
        raise RuntimeError("LAT Broadcast algorithm is for last axis only!")
    dtype_block_size = get_align_factor(dtype)[0]
    block_num = math.ceil(broadcast_src / dtype_block_size)
    # Base cycle: move each broadcast source block into mid buffer,
    #             do transpose for each move, each costs 1 cycle
    cycle_base = 1
    # Use vor to broadcast each block to target factor
    cycle_base += broadcast_factor
    # Transpose broadcasted blocks back
    cycle_base += broadcast_factor
    cycle_total = block_num * cycle_base
    return cycle_total


def fa_cycle_estimation(dtype, broadcast_src, broadcast_unit, broadcast_factor):
    """Estimation function for fa(mid_axis) algorithm cycle performance"""
    list([broadcast_unit]).clear()  # Use it once to avoid static checks
    dtype_byte_size = get_align_factor(dtype)[1]
    vector_repeat_size = te.platform.cce_params.VECTOR_INST_BLOCK_WIDTH // dtype_byte_size
    # Base cycle: none
    cycle_base = 0
    # Action cycle, broadcast one number to target shape using masked vector_dup
    cycle_action = math.ceil(broadcast_factor * broadcast_unit / vector_repeat_size / 2)
    # For cycle, do action cycle for each number
    cycle_for = cycle_action * broadcast_src
    # Total cycle
    cycle_total = cycle_for + cycle_base
    return cycle_total


def sa_cycle_estimation(dtype, broadcast_src, broadcast_unit, broadcast_factor):
    """Estimation function for sa(mid_axis) algorithm cycle performance"""
    dtype_block_size, dtype_byte_size = get_align_factor(dtype)
    least_common_multiple = get_least_common_multiple(broadcast_unit, dtype_block_size)
    factor_after = math.ceil(least_common_multiple / broadcast_unit)
    factor_final = math.ceil((broadcast_unit * broadcast_factor) / (broadcast_unit * factor_after))
    # Base cycle: read each broadcast source number into register, each num costs 1 cycle
    if int(broadcast_unit * dtype_byte_size % 2) == 0:
        cycle_base = csm_cycle_estimation(dtype, broadcast_src, broadcast_unit, factor_after)
    else:
        cycle_base = dsm_cycle_estimation(dtype, broadcast_src, broadcast_unit, factor_after)
    # Action cycle, broadcast one number to target shape using full aligned algorithm
    cycle_action = fa_cycle_estimation(dtype, broadcast_src,
                                       broadcast_unit * factor_after, factor_final)
    # For cycle, already computed in action cycle calculation stage
    cycle_for = cycle_action * 1
    # Total cycle
    cycle_total = cycle_for + cycle_base
    return cycle_total


def csm_cycle_estimation(dtype, broadcast_src, broadcast_unit, broadcast_factor):
    """Estimation function for csm(mid_axis) algorithm cycle performance"""
    # Compound mode selection
    compound_mode_dict = {
        2: "uint16_t",
        4: "int32",
        8: "int64"
    }
    compound_mode = -1
    dtype_size = get_align_factor(dtype)[1]
    for mode in compound_mode_dict:
        if broadcast_unit * dtype_size % mode == 0 and mode > compound_mode:
            compound_mode = mode
    compound_mode_factor = compound_mode // dtype_size
    return dsm_cycle_estimation(dtype, broadcast_src,
                                broadcast_unit, broadcast_factor) // compound_mode_factor
