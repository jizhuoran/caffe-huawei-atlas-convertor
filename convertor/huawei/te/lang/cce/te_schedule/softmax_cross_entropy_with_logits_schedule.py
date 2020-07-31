#!/usr/bin/env python
# -*- coding:utf-8 -*-
# pylint: disable=too-many-lines
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

batch_normalization_forward_training_reduce
"""
from __future__ import absolute_import

import math
import te.lang.cce
from te import tvm
from te import platform as cce
from te.platform import cce_util
from te.platform import cce_emitinsn_params
from .util import get_align_factor
from .util import get_nearest_factor
from .util import gen_reversed_subgraph_list
from .util import DTYPE_WIDTH_MAP


def get_mask_fp16_skip_one(length):
    """
    calculate MASK in cce for skip one half
    """
    length = int(length)
    len1 = max(length - 32, 0)
    len2 = min(length, 32)
    mask1 = 0
    mask2 = 0
    for _ in range(len1):
        mask1 = mask1 * 4 + 1
    for _ in range(len2):
        mask2 = mask2 * 4 + 1
    return mask1, mask2


def reset_mask_insn(ib_expr, type_, bits=128, mask_func=None):
    """
    :describe: caculate the mask, and set vector mask
    :param ib_expr: ir builder
    :param type_: the type of mask dst
    :param bits: the bit of mask, default : 128
    """
    # argmin/argmax has his own set_mask func
    if mask_func is not None:
        mask1, mask2 = mask_func(bits)
    else:
        mask1, mask2 = cce_util.set_mask(bits)

    ib_expr.emit(tvm.call_extern(
        type_, "set_vector_mask", tvm.const(mask1, dtype="uint64"),
        tvm.const(mask2, dtype="uint64")))


# pylint: disable=too-many-locals
@tvm.register_func("tvm.intrin.cce.reg_mov_single")
def reg_mov_single(stmt_op):
    """
    elewise_binary_phony which will eliminate its second input tensor
    """
    ins, outs, _, _ = cce_util.get_dma_buffer(stmt_op)
    ir_builder = tvm.ir_builder.create()

    # Move second input to out
    ir_builder.emit(tvm.call_extern(
        ins[1].dtype, "reg_mov",
        outs[0].access_ptr("rw"),
        ins[1].access_ptr("r")))
    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.reduce_last_axis_reduce_sum_2")
def reduce_last_axis_reduce_sum_2(tensor_op):
    """
    reduce last axis reduce sum 2
    """
    return reduce_last_axis_max_and_sum(tensor_op, "vcadd")


@tvm.register_func("tvm.intrin.cce.reduce_last_axis_reduce_max_2")
def reduce_last_axis_reduce_max_2(tensor_op):
    """
    reduce last axis reduce max 2
    """
    return reduce_last_axis_max_and_sum(tensor_op, "vcmax")


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def reduce_last_axis_max_and_sum(tensor_op, intrin_cmd):
    """
    reduce last axis reduce sum and max
    """
    ins, outs = cce_util.get_buffer(tensor_op)
    ib_expr = tvm.ir_builder.create()

    for_extent_vals = []
    for_vars = []

    def _post_order_for(tensor_op):
        """
        post order
        """
        if isinstance(tensor_op, tvm.stmt.For):
            for_extent_vals.append(tensor_op.extent.value)
            for_vars.append(tensor_op.loop_var)

    _ = tvm.ir_pass.IRTransform(tensor_op, None, _post_order_for, ["For"])

    reduce_axis_len = for_extent_vals[0]
    k1_size = for_vars[0]

    src_buffer = ins[1]
    res_buffer = outs[0]
    dtype = src_buffer.dtype

    if dtype == "float16":
        vector_inst_one_repeat_size = 128
    else:
        vector_inst_one_repeat_size = 64

    if len(for_extent_vals) == 1:
        repeat_time = reduce_axis_len // vector_inst_one_repeat_size
        remain_size = reduce_axis_len % vector_inst_one_repeat_size

        if repeat_time > 0:
            reset_mask_insn(ib_expr, dtype, bits=vector_inst_one_repeat_size)

            ib_expr.emit(tvm.call_extern(
                res_buffer.dtype, intrin_cmd,
                res_buffer.access_ptr("rw", offset=0),
                src_buffer.access_ptr("r", offset=-k1_size),
                repeat_time,
                1,
                1,
                8))

        if remain_size > 0:
            reset_mask_insn(ib_expr, dtype, bits=remain_size)

            src_offset = repeat_time * vector_inst_one_repeat_size - k1_size
            ib_expr.emit(tvm.call_extern(
                res_buffer.dtype, intrin_cmd,
                res_buffer.access_ptr("rw", offset=repeat_time),
                src_buffer.access_ptr("r", offset=src_offset),
                1,
                1,
                1,
                8))

        if repeat_time > 1 or (repeat_time == 1 and remain_size > 0):
            if remain_size > 0:
                reset_mask_insn(ib_expr, dtype, bits=repeat_time + 1)
            else:
                reset_mask_insn(ib_expr, dtype, bits=repeat_time)
            ib_expr.emit(tvm.call_extern(
                res_buffer.dtype, intrin_cmd,
                res_buffer.access_ptr("rw", offset=0),
                res_buffer.access_ptr("r", offset=0),
                1,
                1,
                1,
                8))

        reset_mask_insn(ib_expr, res_buffer.dtype)

        return ib_expr.get()

    loop_len = for_extent_vals[1]
    tmp_buf_len = (reduce_axis_len + 15) // 16 * 16

    def new_alloc(ib_expr, dtype, shape, name, scope):
        """
        new alloc
        """
        buf_var = ib_expr.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name,
                                     scope=scope, data=buf_var)

        return new_buffer

    src_buffer = ins[1]
    res_buffer = outs[0]

    tmp_buf = new_alloc(ib_expr, src_buffer.dtype, (tmp_buf_len,),
                        'tmp_buf', scope=cce.scope_ubuf)

    repeat_time = reduce_axis_len // vector_inst_one_repeat_size
    remain_size = reduce_axis_len % vector_inst_one_repeat_size

    factor = 1
    is_vcmax = False
    if intrin_cmd == "vcmax":
        factor = 2
        is_vcmax = True
    with ib_expr.for_range(0, loop_len, name="idx") as loop:
        with ib_expr.for_range(0, reduce_axis_len, name="idx_2") as idx_2:
            ib_expr.emit(tvm.call_extern(
                tmp_buf.dtype, "reg_mov",
                tmp_buf.access_ptr("rw", offset=idx_2),
                src_buffer.access_ptr(
                    "r", offset=loop*reduce_axis_len + idx_2 - k1_size)
            ))

        if repeat_time > 0:
            reset_mask_insn(ib_expr, dtype, bits=vector_inst_one_repeat_size)
            ib_expr.emit(tvm.call_extern(
                tmp_buf.dtype, intrin_cmd,
                tmp_buf.access_ptr("rw", offset=0),
                tmp_buf.access_ptr("r", offset=0),
                repeat_time,
                1,
                1,
                8))

        if remain_size > 0:
            reset_mask_insn(ib_expr, dtype, bits=remain_size)
            ib_expr.emit(tvm.call_extern(
                tmp_buf.dtype, intrin_cmd,
                tmp_buf.access_ptr("rw", offset=repeat_time * factor),
                tmp_buf.access_ptr("r", offset=repeat_time * vector_inst_one_repeat_size),
                1,
                1,
                1,
                8))

        if is_vcmax:
            vcmax_repeat_size = (vector_inst_one_repeat_size // 2)
            total_time = repeat_time
            if remain_size > 0:
                total_time = total_time + 1
            sub_repeat_time = total_time // vcmax_repeat_size
            sub_remain_size = total_time % vcmax_repeat_size

            if sub_repeat_time > 0:
                reset_mask_insn(ib_expr, dtype, bits=vcmax_repeat_size,
                                mask_func=get_mask_fp16_skip_one)
                ib_expr.emit(tvm.call_extern(
                    tmp_buf.dtype, intrin_cmd,
                    tmp_buf.access_ptr("rw", offset=0),
                    tmp_buf.access_ptr("r", offset=0),
                    sub_repeat_time,
                    1, 1, 8))
            if total_time > 1 and sub_remain_size > 0:
                reset_mask_insn(ib_expr, dtype, bits=sub_remain_size,
                                mask_func=get_mask_fp16_skip_one)
                ib_expr.emit(tvm.call_extern(
                    tmp_buf.dtype, intrin_cmd,
                    tmp_buf.access_ptr("rw", offset=sub_repeat_time * factor),
                    tmp_buf.access_ptr("r",
                                       offset=sub_repeat_time
                                       * vector_inst_one_repeat_size),
                    1, 1, 1, 8))
            if sub_repeat_time > 1 or (sub_repeat_time == 1 and sub_remain_size > 0):
                if sub_remain_size > 0:
                    reset_mask_insn(ib_expr, dtype, bits=sub_repeat_time + 1,
                                    mask_func=get_mask_fp16_skip_one)
                else:
                    reset_mask_insn(ib_expr, dtype, bits=sub_repeat_time,
                                    mask_func=get_mask_fp16_skip_one)

                ib_expr.emit(tvm.call_extern(
                    tmp_buf.dtype, intrin_cmd,
                    tmp_buf.access_ptr("rw", offset=0),
                    tmp_buf.access_ptr("r", offset=0),
                    1, 1, 1, 8))
        else:
            if repeat_time > 1 or (repeat_time == 1 and remain_size > 0):
                if remain_size > 0:
                    reset_mask_insn(ib_expr, dtype, bits=repeat_time + 1)
                else:
                    reset_mask_insn(ib_expr, dtype, bits=repeat_time)

                ib_expr.emit(tvm.call_extern(
                    tmp_buf.dtype, intrin_cmd,
                    tmp_buf.access_ptr("rw", offset=0),
                    tmp_buf.access_ptr("r", offset=0),
                    1, 1, 1, 8))

        ib_expr.emit(tvm.call_extern(
            res_buffer.dtype, "reg_mov",
            res_buffer.access_ptr("rw", offset=loop),
            tmp_buf.access_ptr("rw"), ))

    reset_mask_insn(ib_expr, res_buffer.dtype)

    return ib_expr.get()


@tvm.register_func("tvm.intrin.cce.dma_copy_softmax_cewl")
def dma_copy_softmax_cewl(tensor_op):
    """
    dma copy for softmax cewl
    store the reduce res of every loop,
    and do the ub_to_gm when the last loop
    """
    ib_expr = tvm.ir_builder.create()

    ins, outs = cce_util.get_buffer(tensor_op)
    src_buffer = ins[0]
    dst_buffer = outs[0]

    ib_expr = tvm.ir_builder.create()
    intrin_name = "copy_ubuf_to_gm"

    def new_alloc(ib_expr, dtype, shape, name, scope):
        """
        new alloc
        """
        buf_var = ib_expr.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)

        return new_buffer, buf_var

    out_var1 = []
    len_var1 = []

    def _post_order_add(stmt_in):
        if isinstance(stmt_in, tvm.expr.Add):
            var = stmt_in.b
            if isinstance(var, tvm.expr.Var) and var.name == "i0.inner":
                out_var1.append(var)
    _ = tvm.ir_pass.IRTransform(tensor_op, None, _post_order_add, ["Add"])

    dtype = src_buffer.dtype

    if not out_var1:
        ib_expr.emit(
            tvm.call_extern(
                dtype, intrin_name,
                dst_buffer.access_ptr('r', offset=0),
                src_buffer.access_ptr('r', offset=0),
                0, 1, 1, 0, 0))
        return ib_expr.get()

    out_var = out_var1[0]

    def _post_order_mul(stmt_in):
        if isinstance(stmt_in, tvm.expr.Mul):
            var = stmt_in.b
            if isinstance(var, tvm.expr.IntImm):
                len_var1.append(var.value)
    _ = tvm.ir_pass.IRTransform(tensor_op, None, _post_order_mul, ["Mul"])

    tmp_buf_len = len_var1[0]

    if dtype == "float16":
        dma_copy_size_one_block = 16
    else:
        dma_copy_size_one_block = 8

    if tmp_buf_len < dma_copy_size_one_block:
        raise RuntimeError("buf_len must be greater than %d" % dma_copy_size_one_block)

    tmp_buf, buf_var = new_alloc(ib_expr, src_buffer.dtype, (tmp_buf_len,),
                                 'tmp_buf', scope=cce.scope_ubuf)

    ib_expr.scope_attr(buf_var.asnode(), "pragma_buffer_index",
                       tvm.call_extern("int64", "buffer_index", 1000))

    ib_expr.emit(tvm.call_extern(
        tmp_buf.dtype, "reg_mov",
        tmp_buf.access_ptr("rw", offset=out_var),
        src_buffer.access_ptr("r")
    ))

    with ib_expr.if_scope(out_var == tmp_buf_len - 1):
        repeat_time = tmp_buf_len // dma_copy_size_one_block
        remain_size = tmp_buf_len % dma_copy_size_one_block
        ib_expr.emit(
            tvm.call_extern(
                dst_buffer.dtype, intrin_name,
                dst_buffer.access_ptr('r', offset=-1*out_var),
                tmp_buf.access_ptr('r', offset=0),
                0, 1, repeat_time, 0, 0))

        if remain_size > 0:
            tmp_buf_2, _ = new_alloc(ib_expr, src_buffer.dtype,
                                     (dma_copy_size_one_block,),
                                     'tmp_buf_2',
                                     scope=cce.scope_ubuf)

            res_offset = (repeat_time - 1)*dma_copy_size_one_block + remain_size

            with ib_expr.for_range(0, dma_copy_size_one_block,
                                   name="loop") as _loop:
                ib_expr.emit(tvm.call_extern(
                    dtype, "reg_mov",
                    tmp_buf_2.access_ptr("rw", offset=_loop),
                    tmp_buf.access_ptr("r", offset=res_offset)
                ))

            ib_expr.emit(
                tvm.call_extern(dst_buffer.dtype, intrin_name,
                                dst_buffer.access_ptr('rw', offset=-1 * out_var + res_offset),
                                tmp_buf_2.access_ptr('r', offset=0),
                                0, 1, 1, 0, 0))

    stmt = ib_expr.get()

    stmt = tvm.make.AttrStmt(None, "pragma_buffer_non_reuse",
                             tvm.call_extern("int64", "buffer_non_reuse", 1000),
                             stmt)
    return stmt


def logits_schedule(res, input_tensors):
    """
    logits schedule
    """
    if len(res[0].shape) == 4:
        return logits_nchw_schedule(res, input_tensors)
    return logits_2d_schedule(res, input_tensors)


def get_npart_factor(n_h_w, dtype_bytes, block_dim):
    '''
    get_npart_factor
    '''
    if n_h_w*dtype_bytes >= block_dim*32:
        npart_factor = block_dim
    else:
        npart_factor = n_h_w*dtype_bytes // 32

    return npart_factor


def is_vector_reduce(c_size, split_factor):
    '''
    is_vector_reduce
    '''
    if c_size % 16 == 0 or split_factor == 1:
        return True

    return False


def logits_2d_schedule(res, input_tensors): # pylint: disable=unused-argument
    '''
    softmax_cross_entropy_with_logits schedule for nchw format data
    :param data_features: input tensor 1
    :param data_labels: input tensor2
    :param res: res tensor, include two tensor
    :return: sch
    '''
    # pylint: too-many-locals,
    # pylint: too-many-branches
    # pylint: too-many-statements
    output_loss = res[0]
    output_backprop = res[1]
    shape = [var.value for var in output_backprop.shape]
    c_size = int(shape[1])
    current_csize_maximum = 15360
    if c_size > current_csize_maximum:
        return logits_2d_schedule_large_axis(res, input_tensors)

    dtype = output_backprop.dtype

    reduce_ext = te.lang.cce.sum(output_backprop, axis=-1)
    out = te.lang.cce.vadd(reduce_ext, output_loss)

    sch = tvm.create_schedule(out.op)

    tensor_list_map = {}
    tensor_list_dst_tensor_map = {}
    gen_reversed_subgraph_list(out, tensor_list_map,
                               tensor_list_dst_tensor_map)

    input_tensor_dst_tensor_map = {}
    mid_tensor_dst_tensor_map = {}
    mid_out_tensor_list = [output_loss, output_backprop]
    mid_out_buffer_tensor_list = {}

    for tensor in tensor_list_dst_tensor_map:
        if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
            input_tensor_dst_tensor_map[tensor] = \
                tensor_list_dst_tensor_map[tensor]
        else:
            mid_tensor_dst_tensor_map[tensor] = \
                tensor_list_dst_tensor_map[tensor]

    input_tensor_buffer_tensor_map = {}
    for tensor in input_tensor_dst_tensor_map:
        tensor_ub = sch.cache_read(tensor, cce.scope_ubuf,
                                   input_tensor_dst_tensor_map[tensor])
        input_tensor_buffer_tensor_map[tensor] = tensor_ub

    for tensor in mid_out_tensor_list:
        tensor_ub = sch.cache_read(tensor, cce.scope_ubuf,
                                   mid_tensor_dst_tensor_map[tensor])
        mid_out_buffer_tensor_list[tensor] = tensor_ub

    mid_tensor_buffer_tensor_map = {}
    for tensor in mid_tensor_dst_tensor_map:
        tensor_ub = sch.cache_write(tensor, cce.scope_ubuf)
        mid_tensor_buffer_tensor_map[tensor] = tensor_ub

    out_ub = sch.cache_write(out, cce.scope_ubuf)

    for tensor in mid_tensor_dst_tensor_map:
        if tensor not in mid_out_tensor_list:
            sch[tensor].compute_inline()

    n_h_w = shape[0]

    max_ub_count = get_max_ub_count(dtype, len(shape))

    block_dim = cce.get_soc_spec("CORE_NUM")
    _, dtype_bytes = get_align_factor(dtype)

    npart_factor = get_npart_factor(n_h_w, dtype_bytes, block_dim)
    if npart_factor == 0:
        return None, []

    if dtype == "float16":
        min_num_size_one_core = 16
    elif dtype == "float32":
        min_num_size_one_core = 8
    else:
        raise RuntimeError("Unsupported dtype!")

    if n_h_w > c_size:
        threshold_size = 512

        block_split_inner_size = n_h_w // npart_factor

        while 0 < block_split_inner_size < min_num_size_one_core:
            npart_factor -= 1

            while n_h_w % npart_factor != 0:
                npart_factor -= 1
            block_split_inner_size = n_h_w // npart_factor

        if npart_factor <= 1:
            return None, []

        if block_split_inner_size < min_num_size_one_core:
            npart_factor = 1
            block_split_inner_size = n_h_w // npart_factor
        else:
            while block_split_inner_size >= min_num_size_one_core and \
                    block_split_inner_size * c_size * dtype_bytes < threshold_size and \
                    npart_factor > 1:
                npart_factor -= 1
                while npart_factor > 0 and n_h_w % npart_factor > 1:
                    npart_factor -= 1
                if npart_factor > 0:
                    block_split_inner_size = n_h_w // npart_factor

    else:
        while n_h_w % npart_factor != 0:
            npart_factor -= 1

        if npart_factor <= 1:
            return None, []

        block_split_inner_size = n_h_w // npart_factor

        threshold_size = 8 * 1024

        if block_split_inner_size * c_size * dtype_bytes < threshold_size:
            npart_factor = 1
            block_split_inner_size = n_h_w // npart_factor

    split_factor, npart_factor = get_ub_tiling_2d(
        shape, npart_factor, block_split_inner_size, max_ub_count)

    is_need_workspace = (npart_factor > 1 and \
                        split_factor < min_num_size_one_core) or\
                        c_size > max_ub_count
    if is_need_workspace:
        return logits_2d_schedule_large_axis_workspace(res, input_tensors)

    res.append(out)

    block_res_outer, block_res_inner = sch[out].split(out.op.axis[0],
                                                      nparts=npart_factor)

    res_outer, res_inner = sch[out].split(block_res_inner,
                                          factor=split_factor)

    block = tvm.thread_axis("blockIdx.x")
    sch[out].bind(block_res_outer, block)
    compute_at_axis = res_outer

    for tensor in input_tensor_dst_tensor_map:
        tensor_ub = input_tensor_buffer_tensor_map[tensor]
        sch[tensor_ub].compute_at(sch[out], compute_at_axis)

    for tensor in mid_out_tensor_list:
        tensor_ub = mid_out_buffer_tensor_list[tensor]
        sch[tensor].compute_at(sch[out], compute_at_axis)
        sch[tensor_ub].compute_at(sch[out], compute_at_axis)

    for tensor in mid_tensor_dst_tensor_map:
        tensor_ub = mid_tensor_buffer_tensor_map[tensor]
        sch[tensor_ub].compute_at(sch[out], compute_at_axis)

    sch[out_ub].compute_at(sch[out], compute_at_axis)

    phony_tensor_list = [reduce_ext]

    for i in input_tensor_buffer_tensor_map:
        buffer_tensor = input_tensor_buffer_tensor_map[i]
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], "dma_copy")

    cce_emitinsn_params.cceEmitParamsIns.insert_param(
        "broadcast_axis_offset", 1)

    for i in mid_tensor_buffer_tensor_map:
        mid_tensor = mid_tensor_buffer_tensor_map[i]
        if i in phony_tensor_list:
            sch[mid_tensor].emit_insn(mid_tensor.op.axis[0], "phony_insn")
        else:
            if i.op.tag == "reduce_sum":
                if is_vector_reduce(c_size, split_factor):
                    insn = "vector_reduce_sum"
                else:
                    insn = "reduce_last_axis_reduce_sum_2"
            elif i.op.tag == "reduce_max":
                if is_vector_reduce(c_size, split_factor):
                    insn = "vector_reduce_max"
                else:
                    insn = "reduce_last_axis_reduce_max_2"
            else:
                insn = _get_emit_insn_map(i)

            emit_insn_axis = 0
            if i.op.tag.find("broadcast") != -1:
                emit_insn_axis = 1

            sch[mid_tensor].emit_insn(mid_tensor.op.axis[emit_insn_axis], insn)

    for i in mid_out_tensor_list:
        sch[i].emit_insn(i.op.axis[0], "dma_copy")
        if i in mid_out_buffer_tensor_list.keys():
            phony_read_buffer = mid_out_buffer_tensor_list[i]
            sch[phony_read_buffer].emit_insn(phony_read_buffer.op.axis[0],
                                             "phony_insn")

    sch[out_ub].emit_insn(out_ub.op.axis[0], "phony_insn")
    sch[out].emit_insn(res_inner, "phony_insn")

    return sch, []


# pylint: disable=too-many-locals, too-many-branches, too-many-statements, unused-argument, unnecessary-lambda, invalid-name
def logits_2d_schedule_large_axis(res, input_tensors):
    """For large reduce axis such as (64, 17191)"""
    output_loss = res[0]
    output_backprop = res[1]
    c_size = int(res[1].shape[-1])
    current_csize_maximum = 20000
    if c_size > current_csize_maximum:
        return logits_2d_schedule_large_axis_workspace(res, input_tensors)
    # This node can be used to collect reduce result
    reduce_ext = te.lang.cce.sum(output_backprop, axis=-1, keepdims=True)
    fake_fuse_node = te.lang.cce.vadd(reduce_ext, output_loss)
    fake_output_node = tvm.compute(fake_fuse_node.shape, lambda *indices: fake_fuse_node(*indices))
    shape = fake_output_node.shape
    res.clear()
    res.append(fake_output_node)
    res.append(output_backprop)
    # ////////////////////////////////
    # //         Schedule           //
    # ////////////////////////////////
    cce_emitinsn_params.cceEmitParamsIns.clear_param()
    sch = tvm.create_schedule(fake_output_node.op)
    block_elem_num = te.platform.cce_util.get_align_factor(output_loss.dtype)[0]
    # Get all tensors
    tensor_list_map = {}
    tensor_list_dst_tensor_map = {}
    gen_reversed_subgraph_list(fake_output_node, tensor_list_map,
                                tensor_list_dst_tensor_map)
    tensor_list = list(tensor_list_map.values())
    # Get all placeholders, remove L1 workspace
    placeholder_list = tensor_list[:]
    for dst_tensors in tensor_list_dst_tensor_map.values():
        for tensor in dst_tensors:
            if tensor in placeholder_list:
                placeholder_list.remove(tensor)
    placeholder_list = tuple(
        tensor for tensor in placeholder_list if "L1_workspace" not in tensor.name)
    out_list = (fake_output_node,)
    ub_list = tuple(tensor for tensor in tensor_list if tensor not in placeholder_list + out_list)
    # ////////////////////////////////
    # //     Split Calculation      //
    # ////////////////////////////////
    # ////////////////////////////////
    # //     Block split(Core)      //
    # ////////////////////////////////
    # Get maximum core num
    core_num = int(te.platform.get_soc_spec("CORE_NUM"))
    # available block split axis is always 0 for this operator as the shape will always be rank 2
    block_split_axis_index = 0
    total_block_split_num = int(shape[0])
    # If block_split_axis is not sufficient for all core, change maximum core num to axis size
    block_split_nparts = min(core_num, total_block_split_num)
    # Calculate block split estimated result, if no axis available, it will be 1
    estimate_block_split_factor = max(1.0, total_block_split_num / block_split_nparts)
    not_aligned = False
    if total_block_split_num * int(output_backprop.shape[1]) < 512 or int(
            output_backprop.shape[1]) < block_elem_num:
        # Total shape size or last axis too small, abort block split
        block_split_factor = -1
    elif estimate_block_split_factor < block_elem_num:
        # block_split_factor too small, adjust to a higher number
        not_aligned = True
        block_split_factor = block_elem_num
    elif estimate_block_split_factor.is_integer():
        # Exact division achieved
        block_split_factor = int(estimate_block_split_factor)
    else:
        # Remainder present
        block_split_factor = math.ceil(total_block_split_num / block_split_nparts)
    # ////////////////////////////////
    # //      UB split(Tiling)      //
    # ////////////////////////////////
    # Reduce axis is always too large if schedule runs here
    # So, factor is always 1
    # Axis is always block inner
    # ////////////////////////////////
    # //      DataFlow Control      //
    # ////////////////////////////////
    # Set data on UB
    for tensor in ub_list:
        if tensor not in (output_backprop,):
            sch[tensor].set_scope(cce.scope_ubuf)
    # Read data on GM
    placeholders_ub = tuple(sch.cache_read(placeholder,
                                           cce.scope_ubuf,
                                           tensor_list_dst_tensor_map[placeholder]) for placeholder
                            in placeholder_list)
    # Write data to GM
    output_backprop_ub = sch.cache_write(output_backprop, cce.scope_ubuf)
    # ////////////////////////////////
    # //        Do Blk split        //
    # ////////////////////////////////
    if block_split_factor < 1:
        # No block split needed
        block_outer, block_inner = sch[fake_output_node].split(
            sch[fake_output_node].op.axis[block_split_axis_index], nparts=1)
    else:
        if not_aligned:
            block_outer, block_inner = sch[fake_output_node].split(
                sch[fake_output_node].op.axis[block_split_axis_index],
                factor=block_split_factor)
        else:
            block_outer, block_inner = sch[fake_output_node].split(
                sch[fake_output_node].op.axis[block_split_axis_index],
                nparts=block_split_nparts)
        sch[fake_output_node].bind(block_outer, tvm.thread_axis("blockIdx.x"))

    # ////////////////////////////////
    # //        Do UB split         //
    # ////////////////////////////////
    # UB split is always unavailable
    # ////////////////////////////////
    # //      Compute Control       //
    # ////////////////////////////////
    compute_at_axis = sch[fake_fuse_node].op.axis[0]
    for tensor in ub_list + placeholders_ub + (output_backprop_ub,):
        if tensor != fake_fuse_node:
            sch[tensor].compute_at(sch[fake_fuse_node], compute_at_axis)
    sch[fake_fuse_node].compute_at(sch[fake_output_node], block_outer)

    # ////////////////////////////////
    # //         Emit Insn          //
    # ////////////////////////////////

    def emit_on_self_ex(tensor, axis, op_name):
        sch[tensor].emit_insn(axis, op_name)

    def emit_on_self(tensor, axisnum=0, op_name='dma_copy'):
        emit_on_self_ex(tensor, sch[tensor].op.axis[axisnum], op_name)

    for tensor in placeholders_ub:
        emit_on_self(tensor)
    for tensor in ub_list + (output_backprop_ub,):
        insn = _get_emit_insn_map(tensor)
        if tensor not in [fake_fuse_node, reduce_ext, output_backprop]:
            emit_on_self(tensor, -1, op_name=insn)
    emit_on_self(output_backprop, -1)
    emit_on_self(reduce_ext, -1, "phony_insn")
    sch[fake_fuse_node].emit_insn(sch[fake_fuse_node].op.axis[-1], "reg_mov_single")
    sch[fake_output_node].emit_insn(block_inner, "dma_copy")
    return sch, []


def get_max_ub_count(dtype, dim):
    """
    caculate the max element num loaded in UB buffer
    :return: max element num loaded in UB buffer
    """
    total_size = cce.get_soc_spec("UB_SIZE") // 2  # div 2 for align to fp16
    dtype_size = DTYPE_WIDTH_MAP.get(dtype)
    total_size = total_size // dtype_size

    if dtype == "float32":
        total_width = 6
    else:
        if dim == 4:
            total_width = 10
        else:
            total_width = 12

    align_to = 128

    max_bound = total_width * align_to
    max_ub_count = int(total_size // max_bound * align_to)

    return max_ub_count


# pylint: disable=too-many-branches
def get_ub_tiling(shape, block_tiling_axis, block_tiling_inner_loop, max_ub_count):
    """
    get ub tiling
    """
    last_axis = len(shape) - 1
    ub_split_inner = 1
    ub_split_axis = 0
    if block_tiling_axis < 0 or block_tiling_axis > last_axis:
        return ub_split_axis, ub_split_inner

    bound_size = max_ub_count
    split_axis = block_tiling_axis
    step = -1
    shape_c = shape[1]
    temp_size = shape_c
    need_split = False

    for i in range(last_axis, block_tiling_axis + step, step):
        # step C axis
        if i == 1:
            continue

        temp_size = temp_size * shape[i]
        if temp_size >= bound_size:
            split_axis = i
            temp_size = temp_size / shape[i]
            need_split = True
            break

    split_size = 1
    # split the split axis
    if need_split:
        for i in range(1, shape[split_axis] + 1, 1):
            if (temp_size * i) == bound_size:
                split_size = i
                break
            if (temp_size * i) > bound_size:
                split_size = i - 1
                break
        if split_size < 1:
            return None, None

        if shape[split_axis] % split_size != 0:
            while shape[split_axis] % split_size != 0:
                split_size -= 1
    else:
        split_size = block_tiling_inner_loop

    if split_axis == block_tiling_axis and split_size > block_tiling_inner_loop:
        split_size = block_tiling_inner_loop

    ub_split_inner = split_size
    ub_split_axis = split_axis

    return ub_split_axis, ub_split_inner


def get_ub_tiling_2d(shape, npart_factor,
                     block_tiling_inner_loop, max_ub_count):
    """
    get ub tiling 2d
    """
    step = -1
    shape_c = shape[1]
    temp_size = shape_c
    bound_size = max_ub_count

    threshold_size = 128
    one_block_size = 32
    byte_size_fp32 = 4
    if shape_c > threshold_size and\
        shape_c*byte_size_fp32 % one_block_size != 0:
        return 1, 1

    split_size = 1
    for i in range(block_tiling_inner_loop, 0, step):
        if temp_size * i <= bound_size:
            split_size = i
            while block_tiling_inner_loop % split_size != 0:
                split_size -= 1
            break

    return split_size, npart_factor


# pylint: disable=too-many-locals, too-many-branches, too-many-statements, unused-argument
def logits_nchw_schedule(res, input_tensors):
    '''
    softmax_cross_entropy_with_logits schedule for nchw format data
    :param data_features: input tensor 1
    :param data_labels: input tensor2
    :param res: res tensor, include two tensor
    :return: sch
    '''
    output_loss = res[0]
    output_backprop = res[1]

    reduce_ext = te.lang.cce.sum(output_backprop, axis=1, keepdims=True)
    out = te.lang.cce.vadd(reduce_ext, output_loss)

    res.append(out)
    sch = tvm.create_schedule(out.op)

    tensor_list_map = {}
    tensor_list_dst_tensor_map = {}
    gen_reversed_subgraph_list(out, tensor_list_map, tensor_list_dst_tensor_map)

    input_tensor_dst_tensor_map = {}
    mid_tensor_dst_tensor_map = {}
    mid_out_tensor_list = [output_loss, output_backprop]
    broadcast_not_last_axis_tensors = []
    mid_out_buffer_tensor_list = {}
    reduce_tensor_list = []

    for tensor in tensor_list_dst_tensor_map:
        if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
            input_tensor_dst_tensor_map[tensor] = tensor_list_dst_tensor_map[tensor]
        else:
            mid_tensor_dst_tensor_map[tensor] = tensor_list_dst_tensor_map[tensor]

        if tensor.op.tag.find("broadcast") != -1:
            broadcast_not_last_axis_tensors.append(tensor)

        if tensor.op.tag.find("reduce") != -1:
            reduce_tensor_list.append(tensor)

    input_tensor_buffer_tensor_map = {}
    for tensor in input_tensor_dst_tensor_map:
        tensor_ub = sch.cache_read(tensor, cce.scope_ubuf, input_tensor_dst_tensor_map[tensor])
        input_tensor_buffer_tensor_map[tensor] = tensor_ub

    for tensor in mid_out_tensor_list:
        tensor_ub = sch.cache_read(tensor, cce.scope_ubuf, mid_tensor_dst_tensor_map[tensor])
        mid_out_buffer_tensor_list[tensor] = tensor_ub

    mid_tensor_buffer_tensor_map = {}
    for tensor in mid_tensor_dst_tensor_map:
        if tensor not in broadcast_not_last_axis_tensors:
            tensor_ub = sch.cache_write(tensor, cce.scope_ubuf)
            mid_tensor_buffer_tensor_map[tensor] = tensor_ub

    out_ub = sch.cache_write(out, cce.scope_ubuf)

    for tensor in mid_tensor_dst_tensor_map:
        if tensor not in mid_out_tensor_list:
            sch[tensor].compute_inline()

    # reorder
    block_tiling_axis = 2
    for tensor in reduce_tensor_list:
        tensor_ub = mid_tensor_buffer_tensor_map[tensor]
        reduce_reorder_list = []
        for i in range(len(tensor_ub.op.axis)):
            reduce_reorder_list.append(tensor_ub.op.axis[i])

        reduce_reorder_list.insert(block_tiling_axis, tensor_ub.op.reduce_axis[-1])
        sch[tensor_ub].reorder(*reduce_reorder_list)

    dtype = output_backprop.dtype
    shape_x = [s.value for s in output_backprop.shape]
    batch = shape_x[0]
    shape_h = shape_x[2]
    shape_w = shape_x[3]

    max_ub_count = get_max_ub_count(dtype, len(shape_x))

    block_split_axis = 0

    block_dim = cce.get_soc_spec("CORE_NUM")

    # N greater than or equal to  block_dim
    if batch >= block_dim and batch % block_dim == 0:
        npart_factor = block_dim
        block_res_outer, block_res_inner = sch[out].split(out.op.axis[0], nparts=npart_factor)
        block_split_inner_size = shape_x[block_split_axis] // block_dim
    elif shape_h * shape_w >= block_dim:
        # cut H or W
        block_split_axis = 2

        if shape_h > block_dim and shape_h % block_dim == 0:
            npart_factor = block_dim
        elif shape_w % block_dim == 0:
            npart_factor = block_dim
            while shape_h % npart_factor != 0:
                npart_factor -= 1
        else:
            return None, []

        block_res_outer, block_res_inner = sch[out].split(out.op.axis[2], nparts=npart_factor)
        block_split_inner_size = shape_x[block_split_axis] // npart_factor
    else:
        return None, []

    ub_split_axis, ub_split_inner = get_ub_tiling(
        shape_x, block_split_axis, block_split_inner_size, max_ub_count)

    split_factor = ub_split_inner
    if ub_split_axis == block_split_axis:
        if block_split_inner_size % split_factor != 0:
            while block_split_inner_size % split_factor != 0:
                split_factor -= 1

        res_outer, res_inner = sch[out].split(block_res_inner, factor=split_factor)
    else:
        res_outer, res_inner = sch[out].split(out.op.axis[ub_split_axis],
                                              factor=split_factor)

    onetime_process_size = split_factor
    for i in range(len(shape_x) - 1, ub_split_axis, -1):
        onetime_process_size *= shape_x[i]

    if dtype == "float32":
        onetime_process_size *= 4
    else:
        onetime_process_size *= 2

    # not block align
    if onetime_process_size % 32 != 0:
        return None, []

    block = tvm.thread_axis("blockIdx.x")
    sch[out].bind(block_res_outer, block)
    compute_at_axis = res_outer

    for tensor in input_tensor_dst_tensor_map:
        tensor_ub = input_tensor_buffer_tensor_map[tensor]
        sch[tensor_ub].compute_at(sch[out], compute_at_axis)

    for tensor in mid_out_tensor_list:
        tensor_ub = mid_out_buffer_tensor_list[tensor]
        sch[tensor].compute_at(sch[out], compute_at_axis)
        sch[tensor_ub].compute_at(sch[out], compute_at_axis)

    for tensor in mid_tensor_dst_tensor_map:
        if tensor not in broadcast_not_last_axis_tensors:
            tensor_ub = mid_tensor_buffer_tensor_map[tensor]
            sch[tensor_ub].compute_at(sch[out], compute_at_axis)

    sch[out_ub].compute_at(sch[out], compute_at_axis)

    phony_tensor_list = [reduce_ext]

    for i in input_tensor_buffer_tensor_map:
        buffer_tensor = input_tensor_buffer_tensor_map[i]
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[2], "dma_copy")

    for i in mid_tensor_buffer_tensor_map:
        mid_tensor = mid_tensor_buffer_tensor_map[i]
        if i in phony_tensor_list:
            sch[mid_tensor].emit_insn(mid_tensor.op.axis[0], "phony_insn")
        else:
            insn = _get_emit_insn_map(i)
            emit_insn_axis = 0
            if i.op.tag.find("reduce") != -1:
                emit_insn_axis = 2

            sch[mid_tensor].emit_insn(mid_tensor.op.axis[emit_insn_axis], insn)

    for i in mid_out_tensor_list:
        sch[i].emit_insn(i.op.axis[0], "dma_copy")
        if i in mid_out_buffer_tensor_list.keys():
            phony_read_buffer = mid_out_buffer_tensor_list[i]
            sch[phony_read_buffer].emit_insn(phony_read_buffer.op.axis[0], "phony_insn")

    sch[out_ub].emit_insn(out_ub.op.axis[0], "phony_insn")
    sch[out].emit_insn(res_inner, "phony_insn")

    return sch, []


def _get_tiling_large_axis_workspace(shape, dtype):
    """
    get tiling of large axie with workspace
    the size of one core must be greater equal min_num_size_one_core
    """
    batch = shape[0]
    core_num = int(te.platform.get_soc_spec("CORE_NUM"))

    # block tiling
    if dtype == "float16":
        min_num_size_one_core = 16
    elif dtype == "float32":
        min_num_size_one_core = 8

    if batch < min_num_size_one_core:
        block_factor = batch
        block_outer = 1
    else:
        block_factor = batch
        for i in range(batch, min_num_size_one_core - 1, -1):
            if (batch + i - 1) // i > core_num:
                break
            block_factor = i

        # for odd size
        if block_factor == batch:
            if batch > core_num*min_num_size_one_core:
                block_factor = (batch + core_num - 1) // core_num
            else:
                block_factor = min_num_size_one_core

        block_outer = (batch + block_factor - 1) // block_factor

    # ub tiling
    c_size = shape[1]
    dim = len(shape)
    max_ub_count = get_max_ub_count(dtype, dim)

    temp_size = c_size
    while temp_size > max_ub_count:
        temp_size = temp_size - 1

    ub_factor = get_nearest_factor(c_size, temp_size)

    return block_outer, block_factor, ub_factor


def logits_2d_schedule_large_axis_workspace(res, input_tensors):
    """
    when c size is too large, multi core with workspace.
    the template is produced by RL search, then generalization
    """
    output_loss = res[0]
    output_backprop = res[1]

    dtype = output_loss.dtype
    shape = [var.value for var in output_backprop.shape]

    is_with_cast = False
    if dtype == "float16":
        is_with_cast = True

    if is_with_cast:
        output_loss_reduce = output_loss.op.input_tensors[0]
        output_loss_input = output_loss_reduce.op.input_tensors[0]
        output_loss_fp32 = te.lang.cce.sum(output_loss_input, axis=1, keepdims=True)
        output_loss = te.lang.cce.cast_to(output_loss_fp32, dtype)
    else:
        output_loss_input = output_loss.op.input_tensors[0]
        output_loss = te.lang.cce.sum(output_loss_input, axis=1, keepdims=True)

    res[0] = output_loss

    reduce_2_broadcast = te.lang.cce.broadcast(output_loss, shape)
    add_0 = te.lang.cce.vadd(reduce_2_broadcast, output_backprop)

    res.append(add_0)

    input_labels = input_tensors[1]
    shape_labels = [var.value for var in input_labels.shape]

    is_labels_broadcast = False
    if shape != shape_labels:
        is_labels_broadcast = True

    s = tvm.create_schedule([add_0.op])

    sch_list = [s]
    if is_with_cast:
        spec_node_list = _schedule_workspace_fp16(
            sch_list, add_0, shape, dtype, is_labels_broadcast)

        return sch_list[0], spec_node_list

    if is_labels_broadcast:
        spec_node_list = _schedule_workspace_fp32_broad(
            sch_list, add_0, shape, dtype)

        return sch_list[0], spec_node_list

    sub_7 = add_0.op.input_tensors[1]
    reduce_2 = reduce_2_broadcast.op.input_tensors[0]

    data_labels = sub_7.op.input_tensors[1]

    div_2 = sub_7.op.input_tensors[0]
    mul_6 = reduce_2.op.input_tensors[0]
    exp_1 = div_2.op.input_tensors[0]
    broadcast_tensor_1 = div_2.op.input_tensors[1]
    mul_5 = mul_6.op.input_tensors[0]
    sub_0 = exp_1.op.input_tensors[0]
    reduce_1 = broadcast_tensor_1.op.input_tensors[0]
    sub_4 = mul_5.op.input_tensors[1]

    data_features = sub_0.op.input_tensors[0]

    cast_1 = sub_0.op.input_tensors[1]
    log_3 = sub_4.op.input_tensors[1]
    broadcast_tensor_0 = cast_1.op.input_tensors[0]
    reduce_0 = broadcast_tensor_0.op.input_tensors[0]
    cast_0 = reduce_0.op.input_tensors[0]

    # cache_read/cache_write code
    data_labels_ub_000 = s.cache_read(data_labels, 'local.UB', [mul_5])
    data_labels_ub_001 = s.cache_read(data_labels, 'local.UB', [sub_7])
    data_features_ub_000 = s.cache_read(data_features, 'local.UB', [cast_0])
    data_features_ub_001 = s.cache_read(data_features, 'local.UB', [sub_0])

    cast_0_ub = s.cache_write(cast_0, 'local.UB')
    reduce_0_ub = s.cache_write(reduce_0, 'local.UB')
    broadcast_tensor_0_ub = s.cache_write(broadcast_tensor_0, 'local.UB')
    cast_1_ub = s.cache_write(cast_1, 'local.UB')
    sub_0_ub_000 = s.cache_read(sub_0, 'local.UB', [exp_1])
    sub_0_ub_001 = s.cache_read(sub_0, 'local.UB', [sub_4])
    sub_0_ub = s.cache_write(sub_0, 'local.UB')
    exp_1_ub_000 = s.cache_read(exp_1, 'local.UB', [reduce_1])
    exp_1_ub_001 = s.cache_read(exp_1, 'local.UB', [div_2])
    exp_1_ub = s.cache_write(exp_1, 'local.UB')
    reduce_1_ub = s.cache_write(reduce_1, 'local.UB')
    broadcast_tensor_1_ub_000 = s.cache_read(broadcast_tensor_1, 'local.UB', [log_3])
    broadcast_tensor_1_ub_001 = s.cache_read(broadcast_tensor_1, 'local.UB', [div_2])
    broadcast_tensor_1_ub = s.cache_write(broadcast_tensor_1, 'local.UB')
    log_3_ub = s.cache_write(log_3, 'local.UB')
    sub_4_ub = s.cache_write(sub_4, 'local.UB')
    mul_5_ub = s.cache_write(mul_5, 'local.UB')
    mul_6_ub = s.cache_write(mul_6, 'local.UB')
    reduce_2_ub = s.cache_write(reduce_2, 'local.UB')
    div_2_ub = s.cache_write(div_2, 'local.UB')
    sub_7_ub = s.cache_write(sub_7, 'local.UB')

    # compute_inline code
    s[cast_0].compute_inline()
    s[reduce_0].compute_inline()
    s[broadcast_tensor_0].compute_inline()
    s[cast_1].compute_inline()
    s[reduce_1].compute_inline()
    s[log_3].compute_inline()
    s[sub_4].compute_inline()
    s[mul_5].compute_inline()
    s[mul_6].compute_inline()
    s[reduce_2_broadcast].compute_inline()
    s[div_2].compute_inline()

    block_outer, block_factor, ub_factor = \
        _get_tiling_large_axis_workspace(shape, dtype)

    # split code
    reduce_0_ub_axis_0 = s[reduce_0_ub].op.axis[0]
    reduce_0_ub_axis_1 = s[reduce_0_ub].op.axis[1]
    reduce_0_ub_reduce_axis_0_o, reduce_0_ub_reduce_axis_0_i = \
        s[reduce_0_ub].split(s[reduce_0_ub].op.reduce_axis[0], factor=ub_factor)
    sub_0_axis_0_o = s[sub_0].op.axis[0]
    sub_0_axis_1_o, sub_0_axis_1_i = s[sub_0].split(s[sub_0].op.axis[1], factor=ub_factor)
    exp_1_axis_0_o = s[exp_1].op.axis[0]
    exp_1_axis_1_o, exp_1_axis_1_i = s[exp_1].split(s[exp_1].op.axis[1], factor=ub_factor)
    reduce_1_ub_axis_0 = s[reduce_1_ub].op.axis[0]
    reduce_1_ub_axis_1 = s[reduce_1_ub].op.axis[1]
    reduce_1_ub_reduce_axis_0_o, reduce_1_ub_reduce_axis_0_i = \
        s[reduce_1_ub].split(s[reduce_1_ub].op.reduce_axis[0], factor=ub_factor)
    broadcast_tensor_1_axis_0_o = s[broadcast_tensor_1].op.axis[0]
    broadcast_tensor_1_axis_1_o, broadcast_tensor_1_axis_1_i = \
        s[broadcast_tensor_1].split(s[broadcast_tensor_1].op.axis[1],
                                    factor=ub_factor)
    reduce_2_ub_axis_0 = s[reduce_2_ub].op.axis[0]
    reduce_2_ub_reduce_axis_0_o, reduce_2_ub_reduce_axis_0_i = \
        s[reduce_2_ub].split(s[reduce_2_ub].op.reduce_axis[0], factor=ub_factor)
    add_0_axis_0_o = s[add_0].op.axis[0]
    add_0_axis_1_o, add_0_axis_1_i = s[add_0].split(s[add_0].op.axis[1], factor=ub_factor)
    add_0_axis_1_n1_o, add_0_axis_1_i = s[add_0].split(add_0_axis_1_i, factor=ub_factor)

    # reorder code
    s[reduce_0_ub].reorder(
        reduce_0_ub_reduce_axis_0_o,
        reduce_0_ub_axis_0,
        reduce_0_ub_axis_1,
        reduce_0_ub_reduce_axis_0_i)
    s[sub_0].reorder(sub_0_axis_0_o, sub_0_axis_1_o, sub_0_axis_1_i)
    s[exp_1].reorder(exp_1_axis_0_o, exp_1_axis_1_o, exp_1_axis_1_i)
    s[reduce_1_ub].reorder(
        reduce_1_ub_reduce_axis_0_o,
        reduce_1_ub_axis_0,
        reduce_1_ub_axis_1,
        reduce_1_ub_reduce_axis_0_i)
    s[broadcast_tensor_1].reorder(
        broadcast_tensor_1_axis_0_o,
        broadcast_tensor_1_axis_1_o,
        broadcast_tensor_1_axis_1_i)
    s[reduce_2_ub].reorder(
        reduce_2_ub_reduce_axis_0_o,
        reduce_2_ub_axis_0,
        reduce_2_ub_reduce_axis_0_i)
    s[add_0].reorder(add_0_axis_0_o, add_0_axis_1_o,
                     add_0_axis_1_n1_o, add_0_axis_1_i)

    # compute_at code
    add_0_axis_0_o, add_0_axis_0_n1_o = s[add_0].split(add_0_axis_0_o,
                                                       factor=block_factor)

    s[data_labels_ub_001].compute_at(s[add_0], add_0_axis_1_n1_o)
    s[data_labels_ub_000].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[data_features_ub_001].compute_at(s[sub_0], sub_0_axis_1_o)
    s[data_features_ub_000].compute_at(s[reduce_0_ub], reduce_0_ub_reduce_axis_0_o)

    s[cast_0_ub].compute_at(s[reduce_0_ub], reduce_0_ub_reduce_axis_0_o)
    s[reduce_0_ub].compute_at(s[sub_0], sub_0_axis_0_o)
    s[broadcast_tensor_0_ub].compute_at(s[sub_0], sub_0_axis_1_o)
    s[cast_1_ub].compute_at(s[sub_0], sub_0_axis_1_o)
    s[sub_0_ub].compute_at(s[sub_0], sub_0_axis_1_o)
    s[sub_0].compute_at(s[add_0], add_0_axis_0_n1_o)
    s[sub_0_ub_001].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[sub_0_ub_000].compute_at(s[exp_1], exp_1_axis_1_o)
    s[exp_1_ub].compute_at(s[exp_1], exp_1_axis_1_o)
    s[exp_1].compute_at(s[add_0], add_0_axis_0_n1_o)
    s[exp_1_ub_001].compute_at(s[add_0], add_0_axis_1_n1_o)
    s[exp_1_ub_000].compute_at(s[reduce_1_ub], reduce_1_ub_reduce_axis_0_o)
    s[reduce_1_ub].compute_at(s[broadcast_tensor_1], broadcast_tensor_1_axis_0_o)
    s[broadcast_tensor_1_ub].compute_at(s[broadcast_tensor_1], broadcast_tensor_1_axis_1_o)
    s[broadcast_tensor_1].compute_at(s[add_0], add_0_axis_0_n1_o)
    s[broadcast_tensor_1_ub_001].compute_at(s[add_0], add_0_axis_1_n1_o)
    s[broadcast_tensor_1_ub_000].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[log_3_ub].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[sub_4_ub].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[mul_5_ub].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[mul_6_ub].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[reduce_2_ub].compute_at(s[add_0], add_0_axis_0_n1_o)
    s[div_2_ub].compute_at(s[add_0], add_0_axis_1_n1_o)
    s[sub_7_ub].compute_at(s[add_0], add_0_axis_1_n1_o)

    s[reduce_2].compute_at(s[add_0], add_0_axis_0_n1_o)
    s[sub_7].compute_at(s[add_0], add_0_axis_1_n1_o)

    # bind code
    if block_outer > 1:
        block = tvm.thread_axis("blockIdx.x")
        s[add_0].bind(add_0_axis_0_o, block)

    # emit_insn code
    s[data_labels_ub_001].emit_insn(s[data_labels_ub_001].op.axis[0], 'dma_copy')
    s[data_labels_ub_000].emit_insn(s[data_labels_ub_000].op.axis[0], 'dma_copy')
    s[data_features_ub_001].emit_insn(s[data_features_ub_001].op.axis[0], 'dma_copy')
    s[data_features_ub_000].emit_insn(s[data_features_ub_000].op.axis[0], 'dma_copy')
    s[cast_0_ub].emit_insn(s[cast_0_ub].op.axis[0], 'vector_conv')
    s[reduce_0_ub].emit_insn(reduce_0_ub_axis_0, 'vector_reduce_max')
    cce_emitinsn_params.cceEmitParamsIns.del_param('broadcast_axis_offset')
    cce_emitinsn_params.cceEmitParamsIns.insert_param('broadcast_axis_offset', 1)
    s[cast_1_ub].emit_insn(s[cast_1_ub].op.axis[0], 'vector_conv')
    s[sub_0_ub].emit_insn(s[sub_0_ub].op.axis[0], 'vector_sub')
    s[sub_0].emit_insn(sub_0_axis_1_i, 'dma_copy')
    s[sub_0_ub_001].emit_insn(s[sub_0_ub_001].op.axis[0], 'dma_copy')
    s[sub_0_ub_000].emit_insn(s[sub_0_ub_000].op.axis[0], 'dma_copy')
    s[exp_1_ub].emit_insn(s[exp_1_ub].op.axis[0], 'vector_exp')
    s[exp_1].emit_insn(exp_1_axis_1_i, 'dma_copy')
    s[exp_1_ub_001].emit_insn(s[exp_1_ub_001].op.axis[0], 'dma_copy')
    s[exp_1_ub_000].emit_insn(s[exp_1_ub_000].op.axis[0], 'dma_copy')
    s[reduce_1_ub].emit_insn(reduce_1_ub_axis_0, 'vector_reduce_sum')
    cce_emitinsn_params.cceEmitParamsIns.del_param('broadcast_axis_offset')
    cce_emitinsn_params.cceEmitParamsIns.insert_param('broadcast_axis_offset', 1)
    s[broadcast_tensor_1].emit_insn(broadcast_tensor_1_axis_1_i, 'dma_copy')
    s[broadcast_tensor_1_ub_001].emit_insn(s[broadcast_tensor_1_ub_001].op.axis[0], 'dma_copy')
    s[broadcast_tensor_1_ub_000].emit_insn(s[broadcast_tensor_1_ub_000].op.axis[0], 'dma_copy')
    s[log_3_ub].emit_insn(s[log_3_ub].op.axis[0], 'vector_ln')
    s[sub_4_ub].emit_insn(s[sub_4_ub].op.axis[0], 'vector_sub')
    s[mul_5_ub].emit_insn(s[mul_5_ub].op.axis[0], 'vector_mul')
    s[mul_6_ub].emit_insn(s[mul_6_ub].op.axis[0], 'vector_muls')
    s[reduce_2_ub].emit_insn(reduce_2_ub_axis_0, 'vector_reduce_sum')

    s[sub_7].emit_insn(s[sub_7].op.axis[0], 'dma_copy')
    if block_outer == 1:
        s[reduce_2].emit_insn(s[reduce_2].op.axis[0], 'dma_copy')
    else:
        s[reduce_2].emit_insn(s[reduce_2].op.axis[0],
                              'dma_copy_softmax_cewl')

    s[div_2_ub].emit_insn(s[div_2_ub].op.axis[0], 'vector_div')
    s[sub_7_ub].emit_insn(s[sub_7_ub].op.axis[0], 'vector_sub')
    s[add_0].emit_insn(add_0_axis_1_i, 'phony_insn')

    s[broadcast_tensor_0_ub].emit_insn(s[broadcast_tensor_0_ub].op.axis[1],
                                       'unified_broadcast')
    s[broadcast_tensor_1_ub].emit_insn(s[broadcast_tensor_1_ub].op.axis[1],
                                       'unified_broadcast')

    # storage_align code
    s[data_labels_ub_001].storage_align(s[data_labels_ub_001].op.axis[0], 8, 0)
    s[data_labels_ub_000].storage_align(s[data_labels_ub_000].op.axis[0], 8, 0)
    s[data_features_ub_001].storage_align(s[data_features_ub_001].op.axis[0], 8, 0)
    s[data_features_ub_000].storage_align(s[data_features_ub_000].op.axis[0], 8, 0)
    s[cast_0_ub].storage_align(s[cast_0_ub].op.axis[0], 16, 0)
    s[broadcast_tensor_0_ub].storage_align(s[broadcast_tensor_0_ub].op.axis[0], 16, 0)
    s[cast_1_ub].storage_align(s[cast_1_ub].op.axis[0], 8, 0)
    s[sub_0_ub].storage_align(s[sub_0_ub].op.axis[0], 8, 0)
    s[sub_0_ub_001].storage_align(s[sub_0_ub_001].op.axis[0], 8, 0)
    s[sub_0_ub_000].storage_align(s[sub_0_ub_000].op.axis[0], 8, 0)
    s[exp_1_ub].storage_align(s[exp_1_ub].op.axis[0], 8, 0)
    s[exp_1_ub_001].storage_align(s[exp_1_ub_001].op.axis[0], 8, 0)
    s[exp_1_ub_000].storage_align(s[exp_1_ub_000].op.axis[0], 8, 0)
    s[broadcast_tensor_1_ub].storage_align(s[broadcast_tensor_1_ub].op.axis[0], 8, 0)
    s[broadcast_tensor_1_ub_001].storage_align(s[broadcast_tensor_1_ub_001].op.axis[0], 8, 0)
    s[broadcast_tensor_1_ub_000].storage_align(s[broadcast_tensor_1_ub_000].op.axis[0], 8, 0)
    s[log_3_ub].storage_align(s[log_3_ub].op.axis[0], 8, 0)
    s[sub_4_ub].storage_align(s[sub_4_ub].op.axis[0], 8, 0)
    s[mul_5_ub].storage_align(s[mul_5_ub].op.axis[0], 8, 0)
    s[mul_6_ub].storage_align(s[mul_6_ub].op.axis[0], 8, 0)
    s[div_2_ub].storage_align(s[div_2_ub].op.axis[0], 8, 0)
    s[sub_7_ub].storage_align(s[sub_7_ub].op.axis[0], 8, 0)

    s.cce_special = dict()
    spec_node_list = []

    spec_node_list = [sub_0, exp_1, broadcast_tensor_1]
    # spec_node_list
    s.cce_special["tensor_list"] = spec_node_list
    # the origin out tensor list
    s.cce_special["orign_out_tensor"] = [reduce_2, sub_7]
    # the real out tensor list
    s.cce_special["real_out_tensor"] = [reduce_2, sub_7]

    return s, spec_node_list


def _schedule_workspace_fp32_broad(sch_list, add_0, shape, dtype):
    """
    schedule for fp32 input and input labels need to broadcast
    :param sch_list:
    :param add_0:
    :param shape:
    :param dtype:
    :return:
    """
    s = sch_list[0]

    sub_7 = add_0.op.input_tensors[1]

    reduce_2_broadcast = add_0.op.input_tensors[0]
    reduce_2 = reduce_2_broadcast.op.input_tensors[0]
    data_labels_broad = sub_7.op.input_tensors[1]
    data_labels = data_labels_broad.op.input_tensors[0]

    div_2 = sub_7.op.input_tensors[0]
    mul_6 = reduce_2.op.input_tensors[0]
    exp_1 = div_2.op.input_tensors[0]
    broadcast_tensor_1 = div_2.op.input_tensors[1]
    mul_5 = mul_6.op.input_tensors[0]
    sub_0 = exp_1.op.input_tensors[0]
    reduce_1 = broadcast_tensor_1.op.input_tensors[0]
    sub_4 = mul_5.op.input_tensors[1]

    data_features = sub_0.op.input_tensors[0]

    cast_1 = sub_0.op.input_tensors[1]
    log_3 = sub_4.op.input_tensors[1]
    broadcast_tensor_0 = cast_1.op.input_tensors[0]
    reduce_0 = broadcast_tensor_0.op.input_tensors[0]
    cast_0 = reduce_0.op.input_tensors[0]

    # cache_read/cache_write code
    data_labels_ub = s.cache_read(data_labels, 'local.UB', [data_labels_broad])
    data_labels_broad_ub = s.cache_write(data_labels_broad, 'local.UB')
    data_labels_broad_ub_000 = s.cache_read(data_labels_broad, 'local.UB', [mul_5])
    data_labels_broad_ub_001 = s.cache_read(data_labels_broad, 'local.UB', [sub_7])

    data_features_ub_000 = s.cache_read(data_features, 'local.UB', [cast_0])
    data_features_ub_001 = s.cache_read(data_features, 'local.UB', [sub_0])

    cast_0_ub = s.cache_write(cast_0, 'local.UB')
    reduce_0_ub = s.cache_write(reduce_0, 'local.UB')
    broadcast_tensor_0_ub = s.cache_write(broadcast_tensor_0, 'local.UB')
    cast_1_ub = s.cache_write(cast_1, 'local.UB')
    sub_0_ub_000 = s.cache_read(sub_0, 'local.UB', [exp_1])
    sub_0_ub_001 = s.cache_read(sub_0, 'local.UB', [sub_4])
    sub_0_ub = s.cache_write(sub_0, 'local.UB')
    exp_1_ub_000 = s.cache_read(exp_1, 'local.UB', [reduce_1])
    exp_1_ub_001 = s.cache_read(exp_1, 'local.UB', [div_2])
    exp_1_ub = s.cache_write(exp_1, 'local.UB')
    reduce_1_ub = s.cache_write(reduce_1, 'local.UB')
    broadcast_tensor_1_ub_000 = s.cache_read(broadcast_tensor_1, 'local.UB', [log_3])
    broadcast_tensor_1_ub_001 = s.cache_read(broadcast_tensor_1, 'local.UB', [div_2])
    broadcast_tensor_1_ub = s.cache_write(broadcast_tensor_1, 'local.UB')
    log_3_ub = s.cache_write(log_3, 'local.UB')
    sub_4_ub = s.cache_write(sub_4, 'local.UB')
    mul_5_ub = s.cache_write(mul_5, 'local.UB')
    mul_6_ub = s.cache_write(mul_6, 'local.UB')
    reduce_2_ub = s.cache_write(reduce_2, 'local.UB')
    div_2_ub = s.cache_write(div_2, 'local.UB')
    sub_7_ub = s.cache_write(sub_7, 'local.UB')

    # compute_inline code
    s[cast_0].compute_inline()
    s[reduce_0].compute_inline()
    s[broadcast_tensor_0].compute_inline()
    s[cast_1].compute_inline()
    s[reduce_1].compute_inline()
    s[log_3].compute_inline()
    s[sub_4].compute_inline()
    s[mul_5].compute_inline()
    s[mul_6].compute_inline()
    s[reduce_2_broadcast].compute_inline()
    s[div_2].compute_inline()

    block_outer, block_factor, ub_factor = \
        _get_tiling_large_axis_workspace(shape, dtype)

    # split code
    reduce_0_ub_axis_0 = s[reduce_0_ub].op.axis[0]
    reduce_0_ub_axis_1 = s[reduce_0_ub].op.axis[1]
    reduce_0_ub_reduce_axis_0_o, reduce_0_ub_reduce_axis_0_i = \
        s[reduce_0_ub].split(s[reduce_0_ub].op.reduce_axis[0], factor=ub_factor)
    sub_0_axis_0_o = s[sub_0].op.axis[0]
    sub_0_axis_1_o, sub_0_axis_1_i = s[sub_0].split(s[sub_0].op.axis[1], factor=ub_factor)
    exp_1_axis_0_o = s[exp_1].op.axis[0]
    exp_1_axis_1_o, exp_1_axis_1_i = s[exp_1].split(s[exp_1].op.axis[1], factor=ub_factor)
    reduce_1_ub_axis_0 = s[reduce_1_ub].op.axis[0]
    reduce_1_ub_axis_1 = s[reduce_1_ub].op.axis[1]
    reduce_1_ub_reduce_axis_0_o, reduce_1_ub_reduce_axis_0_i = \
        s[reduce_1_ub].split(s[reduce_1_ub].op.reduce_axis[0], factor=ub_factor)
    broadcast_tensor_1_axis_0_o = s[broadcast_tensor_1].op.axis[0]
    broadcast_tensor_1_axis_1_o, broadcast_tensor_1_axis_1_i = \
        s[broadcast_tensor_1].split(s[broadcast_tensor_1].op.axis[1],
                                    factor=ub_factor)
    reduce_2_ub_axis_0 = s[reduce_2_ub].op.axis[0]
    reduce_2_ub_reduce_axis_0_o, reduce_2_ub_reduce_axis_0_i = \
        s[reduce_2_ub].split(s[reduce_2_ub].op.reduce_axis[0], factor=ub_factor)
    add_0_axis_0_o = s[add_0].op.axis[0]
    add_0_axis_1_o, add_0_axis_1_i = s[add_0].split(s[add_0].op.axis[1], factor=ub_factor)
    add_0_axis_1_n1_o, add_0_axis_1_i = s[add_0].split(add_0_axis_1_i, factor=ub_factor)

    # reorder code
    s[reduce_0_ub].reorder(
        reduce_0_ub_reduce_axis_0_o,
        reduce_0_ub_axis_0,
        reduce_0_ub_axis_1,
        reduce_0_ub_reduce_axis_0_i)
    s[sub_0].reorder(sub_0_axis_0_o, sub_0_axis_1_o, sub_0_axis_1_i)
    s[exp_1].reorder(exp_1_axis_0_o, exp_1_axis_1_o, exp_1_axis_1_i)
    s[reduce_1_ub].reorder(
        reduce_1_ub_reduce_axis_0_o,
        reduce_1_ub_axis_0,
        reduce_1_ub_axis_1,
        reduce_1_ub_reduce_axis_0_i)
    s[broadcast_tensor_1].reorder(
        broadcast_tensor_1_axis_0_o,
        broadcast_tensor_1_axis_1_o,
        broadcast_tensor_1_axis_1_i)
    s[reduce_2_ub].reorder(
        reduce_2_ub_reduce_axis_0_o,
        reduce_2_ub_axis_0,
        reduce_2_ub_reduce_axis_0_i)
    s[add_0].reorder(add_0_axis_0_o, add_0_axis_1_o,
                     add_0_axis_1_n1_o, add_0_axis_1_i)

    # compute_at code
    add_0_axis_0_o, add_0_axis_0_n1_o = s[add_0].split(add_0_axis_0_o,
                                                       factor=block_factor)

    data_labels_broad_axis_1_o, data_labels_broad_axis_1_i = \
        s[data_labels_broad].split(s[data_labels_broad].op.axis[1], factor=ub_factor)

    s[data_labels_ub].compute_at(s[data_labels_broad], data_labels_broad_axis_1_o)
    s[data_labels_broad_ub].compute_at(s[data_labels_broad], data_labels_broad_axis_1_o)
    s[data_labels_broad].compute_at(s[add_0], add_0_axis_0_n1_o)

    s[data_labels_broad_ub_001].compute_at(s[add_0], add_0_axis_1_n1_o)
    s[data_labels_broad_ub_000].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[data_features_ub_001].compute_at(s[sub_0], sub_0_axis_1_o)
    s[data_features_ub_000].compute_at(s[reduce_0_ub], reduce_0_ub_reduce_axis_0_o)

    s[cast_0_ub].compute_at(s[reduce_0_ub], reduce_0_ub_reduce_axis_0_o)
    s[reduce_0_ub].compute_at(s[sub_0], sub_0_axis_0_o)
    s[broadcast_tensor_0_ub].compute_at(s[sub_0], sub_0_axis_1_o)
    s[cast_1_ub].compute_at(s[sub_0], sub_0_axis_1_o)
    s[sub_0_ub].compute_at(s[sub_0], sub_0_axis_1_o)
    s[sub_0].compute_at(s[add_0], add_0_axis_0_n1_o)
    s[sub_0_ub_001].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[sub_0_ub_000].compute_at(s[exp_1], exp_1_axis_1_o)
    s[exp_1_ub].compute_at(s[exp_1], exp_1_axis_1_o)
    s[exp_1].compute_at(s[add_0], add_0_axis_0_n1_o)
    s[exp_1_ub_001].compute_at(s[add_0], add_0_axis_1_n1_o)
    s[exp_1_ub_000].compute_at(s[reduce_1_ub], reduce_1_ub_reduce_axis_0_o)
    s[reduce_1_ub].compute_at(s[broadcast_tensor_1], broadcast_tensor_1_axis_0_o)
    s[broadcast_tensor_1_ub].compute_at(s[broadcast_tensor_1], broadcast_tensor_1_axis_1_o)
    s[broadcast_tensor_1].compute_at(s[add_0], add_0_axis_0_n1_o)
    s[broadcast_tensor_1_ub_001].compute_at(s[add_0], add_0_axis_1_n1_o)
    s[broadcast_tensor_1_ub_000].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[log_3_ub].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[sub_4_ub].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[mul_5_ub].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[mul_6_ub].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[reduce_2_ub].compute_at(s[add_0], add_0_axis_0_n1_o)
    s[div_2_ub].compute_at(s[add_0], add_0_axis_1_n1_o)
    s[sub_7_ub].compute_at(s[add_0], add_0_axis_1_n1_o)

    s[reduce_2].compute_at(s[add_0], add_0_axis_0_n1_o)
    s[sub_7].compute_at(s[add_0], add_0_axis_1_n1_o)

    # bind code
    if block_outer > 1:
        block = tvm.thread_axis("blockIdx.x")
        s[add_0].bind(add_0_axis_0_o, block)

    # emit_insn code
    cce_emitinsn_params.cceEmitParamsIns.del_param('broadcast_axis_offset')
    cce_emitinsn_params.cceEmitParamsIns.insert_param('broadcast_axis_offset', 1)
    s[data_labels_ub].emit_insn(s[data_labels_ub].op.axis[0], 'dma_copy')
    s[data_labels_broad_ub].emit_insn(s[data_labels_broad_ub].op.axis[1],
                                      'unified_broadcast')
    s[data_labels_broad].emit_insn(data_labels_broad_axis_1_i, 'dma_copy')
    s[data_labels_broad_ub_001].emit_insn(s[data_labels_broad_ub_001].op.axis[0], 'dma_copy')
    s[data_labels_broad_ub_000].emit_insn(s[data_labels_broad_ub_000].op.axis[0], 'dma_copy')
    s[data_features_ub_001].emit_insn(s[data_features_ub_001].op.axis[0], 'dma_copy')
    s[data_features_ub_000].emit_insn(s[data_features_ub_000].op.axis[0], 'dma_copy')
    s[cast_0_ub].emit_insn(s[cast_0_ub].op.axis[0], 'vector_conv')
    s[reduce_0_ub].emit_insn(reduce_0_ub_axis_0, 'vector_reduce_max')
    cce_emitinsn_params.cceEmitParamsIns.del_param('broadcast_axis_offset')
    cce_emitinsn_params.cceEmitParamsIns.insert_param('broadcast_axis_offset', 1)
    s[cast_1_ub].emit_insn(s[cast_1_ub].op.axis[0], 'vector_conv')
    s[sub_0_ub].emit_insn(s[sub_0_ub].op.axis[0], 'vector_sub')
    s[sub_0].emit_insn(sub_0_axis_1_i, 'dma_copy')
    s[sub_0_ub_001].emit_insn(s[sub_0_ub_001].op.axis[0], 'dma_copy')
    s[sub_0_ub_000].emit_insn(s[sub_0_ub_000].op.axis[0], 'dma_copy')
    s[exp_1_ub].emit_insn(s[exp_1_ub].op.axis[0], 'vector_exp')
    s[exp_1].emit_insn(exp_1_axis_1_i, 'dma_copy')
    s[exp_1_ub_001].emit_insn(s[exp_1_ub_001].op.axis[0], 'dma_copy')
    s[exp_1_ub_000].emit_insn(s[exp_1_ub_000].op.axis[0], 'dma_copy')
    s[reduce_1_ub].emit_insn(reduce_1_ub_axis_0, 'vector_reduce_sum')
    cce_emitinsn_params.cceEmitParamsIns.del_param('broadcast_axis_offset')
    cce_emitinsn_params.cceEmitParamsIns.insert_param('broadcast_axis_offset', 1)
    s[broadcast_tensor_1].emit_insn(broadcast_tensor_1_axis_1_i, 'dma_copy')
    s[broadcast_tensor_1_ub_001].emit_insn(s[broadcast_tensor_1_ub_001].op.axis[0], 'dma_copy')
    s[broadcast_tensor_1_ub_000].emit_insn(s[broadcast_tensor_1_ub_000].op.axis[0], 'dma_copy')
    s[log_3_ub].emit_insn(s[log_3_ub].op.axis[0], 'vector_ln')
    s[sub_4_ub].emit_insn(s[sub_4_ub].op.axis[0], 'vector_sub')
    s[mul_5_ub].emit_insn(s[mul_5_ub].op.axis[0], 'vector_mul')
    s[mul_6_ub].emit_insn(s[mul_6_ub].op.axis[0], 'vector_muls')
    s[reduce_2_ub].emit_insn(reduce_2_ub_axis_0, 'vector_reduce_sum')

    s[sub_7].emit_insn(s[sub_7].op.axis[0], 'dma_copy')
    if block_outer == 1:
        s[reduce_2].emit_insn(s[reduce_2].op.axis[0], 'dma_copy')
    else:
        s[reduce_2].emit_insn(s[reduce_2].op.axis[0],
                              'dma_copy_softmax_cewl')

    s[div_2_ub].emit_insn(s[div_2_ub].op.axis[0], 'vector_div')
    s[sub_7_ub].emit_insn(s[sub_7_ub].op.axis[0], 'vector_sub')
    s[add_0].emit_insn(add_0_axis_1_i, 'phony_insn')


    s[broadcast_tensor_0_ub].emit_insn(s[broadcast_tensor_0_ub].op.axis[1],
                                       'unified_broadcast')
    s[broadcast_tensor_1_ub].emit_insn(s[broadcast_tensor_1_ub].op.axis[1],
                                       'unified_broadcast')

    # storage_align code
    s[data_labels_broad_ub_000].storage_align(s[data_labels_broad_ub_000].op.axis[0], 8, 0)
    s[data_labels_broad_ub_001].storage_align(s[data_labels_broad_ub_001].op.axis[0], 8, 0)
    s[data_labels_ub].storage_align(s[data_labels_ub].op.axis[0], 8, 0)
    s[data_labels_broad_ub].storage_align(s[data_labels_broad_ub].op.axis[0], 8, 0)
    s[data_features_ub_001].storage_align(s[data_features_ub_001].op.axis[0], 8, 0)
    s[data_features_ub_000].storage_align(s[data_features_ub_000].op.axis[0], 8, 0)
    s[cast_0_ub].storage_align(s[cast_0_ub].op.axis[0], 16, 0)
    s[broadcast_tensor_0_ub].storage_align(s[broadcast_tensor_0_ub].op.axis[0], 16, 0)
    s[cast_1_ub].storage_align(s[cast_1_ub].op.axis[0], 8, 0)
    s[sub_0_ub].storage_align(s[sub_0_ub].op.axis[0], 8, 0)
    s[sub_0_ub_001].storage_align(s[sub_0_ub_001].op.axis[0], 8, 0)
    s[sub_0_ub_000].storage_align(s[sub_0_ub_000].op.axis[0], 8, 0)
    s[exp_1_ub].storage_align(s[exp_1_ub].op.axis[0], 8, 0)
    s[exp_1_ub_001].storage_align(s[exp_1_ub_001].op.axis[0], 8, 0)
    s[exp_1_ub_000].storage_align(s[exp_1_ub_000].op.axis[0], 8, 0)
    s[broadcast_tensor_1_ub].storage_align(s[broadcast_tensor_1_ub].op.axis[0], 8, 0)
    s[broadcast_tensor_1_ub_001].storage_align(s[broadcast_tensor_1_ub_001].op.axis[0], 8, 0)
    s[broadcast_tensor_1_ub_000].storage_align(s[broadcast_tensor_1_ub_000].op.axis[0], 8, 0)
    s[log_3_ub].storage_align(s[log_3_ub].op.axis[0], 8, 0)
    s[sub_4_ub].storage_align(s[sub_4_ub].op.axis[0], 8, 0)
    s[mul_5_ub].storage_align(s[mul_5_ub].op.axis[0], 8, 0)
    s[mul_6_ub].storage_align(s[mul_6_ub].op.axis[0], 8, 0)
    s[div_2_ub].storage_align(s[div_2_ub].op.axis[0], 8, 0)
    s[sub_7_ub].storage_align(s[sub_7_ub].op.axis[0], 8, 0)

    s.cce_special = dict()

    spec_node_list = [data_labels_broad, sub_0, exp_1, broadcast_tensor_1]
    # spec_node_list
    s.cce_special["tensor_list"] = spec_node_list
    # the origin out tensor list
    s.cce_special["orign_out_tensor"] = [reduce_2, sub_7]
    # the real out tensor list
    s.cce_special["real_out_tensor"] = [reduce_2, sub_7]

    sch_list[0] = s
    return spec_node_list


def _schedule_workspace_fp16(sch_list, add_0, shape,
                             dtype, is_labels_broadcast):
    """
    schedule for fp16 input
    :param sch_list:
    :param add_0:
    :param shape:
    :param dtype:
    :return:
    """
    s = sch_list[0]

    sub_7_cast = add_0.op.input_tensors[1]
    sub_7 = sub_7_cast.op.input_tensors[0]
    reduce_2_broadcast = add_0.op.input_tensors[0]
    reduce_2_cast = reduce_2_broadcast.op.input_tensors[0]
    reduce_2 = reduce_2_cast.op.input_tensors[0]
    data_labels_cast = sub_7.op.input_tensors[1]
    if is_labels_broadcast:
        data_labels_broad = data_labels_cast.op.input_tensors[0]
        data_labels = data_labels_broad.op.input_tensors[0]
    else:
        data_labels = data_labels_cast.op.input_tensors[0]

    div_2 = sub_7.op.input_tensors[0]
    mul_6 = reduce_2.op.input_tensors[0]
    exp_1 = div_2.op.input_tensors[0]
    broadcast_tensor_1 = div_2.op.input_tensors[1]
    mul_5 = mul_6.op.input_tensors[0]
    sub_0 = exp_1.op.input_tensors[0]
    reduce_1 = broadcast_tensor_1.op.input_tensors[0]
    sub_4 = mul_5.op.input_tensors[1]

    data_features_cast = sub_0.op.input_tensors[0]
    data_features = data_features_cast.op.input_tensors[0]

    cast_1 = sub_0.op.input_tensors[1]
    log_3 = sub_4.op.input_tensors[1]
    broadcast_tensor_0 = cast_1.op.input_tensors[0]
    reduce_0 = broadcast_tensor_0.op.input_tensors[0]
    cast_0 = reduce_0.op.input_tensors[0]

    # cache_read/cache_write code

    if is_labels_broadcast:
        data_labels_ub = s.cache_read(data_labels, 'local.UB', [data_labels_broad])
        data_labels_broad_ub = s.cache_write(data_labels_broad, 'local.UB')
        s[data_labels_broad].compute_inline()
    else:
        data_labels_ub = s.cache_read(data_labels, 'local.UB', [data_labels_cast])
    data_labels_ub_000 = s.cache_read(data_labels_cast, 'local.UB', [mul_5])
    data_labels_ub_001 = s.cache_read(data_labels_cast, 'local.UB', [sub_7])
    data_labels_cast_ub = s.cache_write(data_labels_cast, 'local.UB')

    data_features_ub = s.cache_read(data_features, 'local.UB', [data_features_cast])
    data_features_ub_000 = s.cache_read(data_features_cast, 'local.UB', [cast_0])
    data_features_ub_001 = s.cache_read(data_features_cast, 'local.UB', [sub_0])
    data_features_cast_ub = s.cache_write(data_features_cast, 'local.UB')

    cast_0_ub = s.cache_write(cast_0, 'local.UB')
    reduce_0_ub = s.cache_write(reduce_0, 'local.UB')
    broadcast_tensor_0_ub = s.cache_write(broadcast_tensor_0, 'local.UB')
    cast_1_ub = s.cache_write(cast_1, 'local.UB')
    sub_0_ub_000 = s.cache_read(sub_0, 'local.UB', [exp_1])
    sub_0_ub_001 = s.cache_read(sub_0, 'local.UB', [sub_4])
    sub_0_ub = s.cache_write(sub_0, 'local.UB')
    exp_1_ub_000 = s.cache_read(exp_1, 'local.UB', [reduce_1])
    exp_1_ub_001 = s.cache_read(exp_1, 'local.UB', [div_2])
    exp_1_ub = s.cache_write(exp_1, 'local.UB')
    reduce_1_ub = s.cache_write(reduce_1, 'local.UB')
    broadcast_tensor_1_ub_000 = s.cache_read(broadcast_tensor_1, 'local.UB', [log_3])
    broadcast_tensor_1_ub_001 = s.cache_read(broadcast_tensor_1, 'local.UB', [div_2])
    broadcast_tensor_1_ub = s.cache_write(broadcast_tensor_1, 'local.UB')
    log_3_ub = s.cache_write(log_3, 'local.UB')
    sub_4_ub = s.cache_write(sub_4, 'local.UB')
    mul_5_ub = s.cache_write(mul_5, 'local.UB')
    mul_6_ub = s.cache_write(mul_6, 'local.UB')
    reduce_2_ub = s.cache_write(reduce_2, 'local.UB')
    div_2_ub = s.cache_write(div_2, 'local.UB')
    sub_7_ub = s.cache_write(sub_7, 'local.UB')
    sub_7_cast_ub = s.cache_write(sub_7_cast, 'local.UB')
    reduce_2_cast_ub = s.cache_write(reduce_2_cast, 'local.UB')

    # compute_inline code
    s[cast_0].compute_inline()
    s[reduce_0].compute_inline()
    s[broadcast_tensor_0].compute_inline()
    s[cast_1].compute_inline()
    s[reduce_1].compute_inline()
    s[log_3].compute_inline()
    s[sub_4].compute_inline()
    s[mul_5].compute_inline()
    s[mul_6].compute_inline()
    s[reduce_2_broadcast].compute_inline()
    s[div_2].compute_inline()
    s[sub_7].compute_inline()
    s[reduce_2].compute_inline()

    if is_labels_broadcast:
        s[data_labels_broad].compute_inline()

    block_outer, block_factor, ub_factor = \
        _get_tiling_large_axis_workspace(shape, dtype)

    # split code
    reduce_0_ub_axis_0 = s[reduce_0_ub].op.axis[0]
    reduce_0_ub_axis_1 = s[reduce_0_ub].op.axis[1]
    reduce_0_ub_reduce_axis_0_o, reduce_0_ub_reduce_axis_0_i = \
        s[reduce_0_ub].split(s[reduce_0_ub].op.reduce_axis[0], factor=ub_factor)
    sub_0_axis_0_o = s[sub_0].op.axis[0]
    sub_0_axis_1_o, sub_0_axis_1_i = s[sub_0].split(s[sub_0].op.axis[1], factor=ub_factor)
    exp_1_axis_0_o = s[exp_1].op.axis[0]
    exp_1_axis_1_o, exp_1_axis_1_i = s[exp_1].split(s[exp_1].op.axis[1], factor=ub_factor)
    reduce_1_ub_axis_0 = s[reduce_1_ub].op.axis[0]
    reduce_1_ub_axis_1 = s[reduce_1_ub].op.axis[1]
    reduce_1_ub_reduce_axis_0_o, reduce_1_ub_reduce_axis_0_i = \
        s[reduce_1_ub].split(s[reduce_1_ub].op.reduce_axis[0], factor=ub_factor)
    broadcast_tensor_1_axis_0_o = s[broadcast_tensor_1].op.axis[0]
    broadcast_tensor_1_axis_1_o, broadcast_tensor_1_axis_1_i = \
        s[broadcast_tensor_1].split(s[broadcast_tensor_1].op.axis[1],
                                    factor=ub_factor)
    reduce_2_ub_axis_0 = s[reduce_2_ub].op.axis[0]
    reduce_2_ub_reduce_axis_0_o, reduce_2_ub_reduce_axis_0_i = \
        s[reduce_2_ub].split(s[reduce_2_ub].op.reduce_axis[0], factor=ub_factor)
    add_0_axis_0_o = s[add_0].op.axis[0]
    add_0_axis_1_o, add_0_axis_1_i = s[add_0].split(s[add_0].op.axis[1], factor=ub_factor)
    add_0_axis_1_n1_o, add_0_axis_1_i = s[add_0].split(add_0_axis_1_i, factor=ub_factor)

    # reorder code
    s[reduce_0_ub].reorder(
        reduce_0_ub_reduce_axis_0_o,
        reduce_0_ub_axis_0,
        reduce_0_ub_axis_1,
        reduce_0_ub_reduce_axis_0_i)
    s[sub_0].reorder(sub_0_axis_0_o, sub_0_axis_1_o, sub_0_axis_1_i)
    s[exp_1].reorder(exp_1_axis_0_o, exp_1_axis_1_o, exp_1_axis_1_i)
    s[reduce_1_ub].reorder(
        reduce_1_ub_reduce_axis_0_o,
        reduce_1_ub_axis_0,
        reduce_1_ub_axis_1,
        reduce_1_ub_reduce_axis_0_i)
    s[broadcast_tensor_1].reorder(
        broadcast_tensor_1_axis_0_o,
        broadcast_tensor_1_axis_1_o,
        broadcast_tensor_1_axis_1_i)
    s[reduce_2_ub].reorder(
        reduce_2_ub_reduce_axis_0_o,
        reduce_2_ub_axis_0,
        reduce_2_ub_reduce_axis_0_i)
    s[add_0].reorder(add_0_axis_0_o, add_0_axis_1_o,
                     add_0_axis_1_n1_o, add_0_axis_1_i)

    # compute_at code
    add_0_axis_0_o, add_0_axis_0_n1_o = s[add_0].split(add_0_axis_0_o,
                                                       factor=block_factor)

    data_labels_cast_axis_1_o, data_labels_cast_axis_1_i = \
        s[data_labels_cast].split(s[data_labels_cast].op.axis[1], factor=ub_factor)

    data_features_cast_axis_1_o, data_features_cast_axis_1_i = \
        s[data_features_cast].split(s[data_features_cast].op.axis[1], factor=ub_factor)

    s[data_labels_ub].compute_at(s[data_labels_cast], data_labels_cast_axis_1_o)
    s[data_labels_cast_ub].compute_at(s[data_labels_cast], data_labels_cast_axis_1_o)

    if is_labels_broadcast:
        s[data_labels_broad_ub].compute_at(s[data_labels_cast],
                                           data_labels_cast_axis_1_o)

    s[data_labels_cast].compute_at(s[add_0], add_0_axis_0_n1_o)

    s[data_features_ub].compute_at(s[data_features_cast], data_features_cast_axis_1_o)
    s[data_features_cast_ub].compute_at(s[data_features_cast], data_features_cast_axis_1_o)
    s[data_features_cast].compute_at(s[add_0], add_0_axis_0_n1_o)

    s[data_labels_ub_001].compute_at(s[add_0], add_0_axis_1_n1_o)
    s[data_labels_ub_000].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[data_features_ub_001].compute_at(s[sub_0], sub_0_axis_1_o)
    s[data_features_ub_000].compute_at(s[reduce_0_ub], reduce_0_ub_reduce_axis_0_o)

    s[cast_0_ub].compute_at(s[reduce_0_ub], reduce_0_ub_reduce_axis_0_o)
    s[reduce_0_ub].compute_at(s[sub_0], sub_0_axis_0_o)
    s[broadcast_tensor_0_ub].compute_at(s[sub_0], sub_0_axis_1_o)
    s[cast_1_ub].compute_at(s[sub_0], sub_0_axis_1_o)
    s[sub_0_ub].compute_at(s[sub_0], sub_0_axis_1_o)
    s[sub_0].compute_at(s[add_0], add_0_axis_0_n1_o)
    s[sub_0_ub_001].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[sub_0_ub_000].compute_at(s[exp_1], exp_1_axis_1_o)
    s[exp_1_ub].compute_at(s[exp_1], exp_1_axis_1_o)
    s[exp_1].compute_at(s[add_0], add_0_axis_0_n1_o)
    s[exp_1_ub_001].compute_at(s[add_0], add_0_axis_1_n1_o)
    s[exp_1_ub_000].compute_at(s[reduce_1_ub], reduce_1_ub_reduce_axis_0_o)
    s[reduce_1_ub].compute_at(s[broadcast_tensor_1], broadcast_tensor_1_axis_0_o)
    s[broadcast_tensor_1_ub].compute_at(s[broadcast_tensor_1], broadcast_tensor_1_axis_1_o)
    s[broadcast_tensor_1].compute_at(s[add_0], add_0_axis_0_n1_o)
    s[broadcast_tensor_1_ub_001].compute_at(s[add_0], add_0_axis_1_n1_o)
    s[broadcast_tensor_1_ub_000].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[log_3_ub].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[sub_4_ub].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[mul_5_ub].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[mul_6_ub].compute_at(s[reduce_2_ub], reduce_2_ub_reduce_axis_0_o)
    s[reduce_2_ub].compute_at(s[add_0], add_0_axis_0_n1_o)
    s[div_2_ub].compute_at(s[add_0], add_0_axis_1_n1_o)
    s[sub_7_ub].compute_at(s[add_0], add_0_axis_1_n1_o)

    s[reduce_2_cast_ub].compute_at(s[add_0], add_0_axis_0_n1_o)
    s[reduce_2_cast].compute_at(s[add_0], add_0_axis_0_n1_o)

    s[sub_7_cast_ub].compute_at(s[add_0], add_0_axis_1_n1_o)
    s[sub_7_cast].compute_at(s[add_0], add_0_axis_1_n1_o)

    # bind code
    if block_outer > 1:
        block = tvm.thread_axis("blockIdx.x")
        s[add_0].bind(add_0_axis_0_o, block)

    # emit_insn code
    s[data_labels_ub_001].emit_insn(s[data_labels_ub_001].op.axis[0], 'dma_copy')
    s[data_labels_ub_000].emit_insn(s[data_labels_ub_000].op.axis[0], 'dma_copy')
    s[data_features_ub_001].emit_insn(s[data_features_ub_001].op.axis[0], 'dma_copy')
    s[data_features_ub_000].emit_insn(s[data_features_ub_000].op.axis[0], 'dma_copy')
    s[cast_0_ub].emit_insn(s[cast_0_ub].op.axis[0], 'vector_conv')
    s[reduce_0_ub].emit_insn(reduce_0_ub_axis_0, 'vector_reduce_max')
    cce_emitinsn_params.cceEmitParamsIns.del_param('broadcast_axis_offset')
    cce_emitinsn_params.cceEmitParamsIns.insert_param('broadcast_axis_offset', 1)
    s[cast_1_ub].emit_insn(s[cast_1_ub].op.axis[0], 'vector_conv')
    s[sub_0_ub].emit_insn(s[sub_0_ub].op.axis[0], 'vector_sub')
    s[sub_0].emit_insn(sub_0_axis_1_i, 'dma_copy')
    s[sub_0_ub_001].emit_insn(s[sub_0_ub_001].op.axis[0], 'dma_copy')
    s[sub_0_ub_000].emit_insn(s[sub_0_ub_000].op.axis[0], 'dma_copy')
    s[exp_1_ub].emit_insn(s[exp_1_ub].op.axis[0], 'vector_exp')
    s[exp_1].emit_insn(exp_1_axis_1_i, 'dma_copy')
    s[exp_1_ub_001].emit_insn(s[exp_1_ub_001].op.axis[0], 'dma_copy')
    s[exp_1_ub_000].emit_insn(s[exp_1_ub_000].op.axis[0], 'dma_copy')
    s[reduce_1_ub].emit_insn(reduce_1_ub_axis_0, 'vector_reduce_sum')
    cce_emitinsn_params.cceEmitParamsIns.del_param('broadcast_axis_offset')
    cce_emitinsn_params.cceEmitParamsIns.insert_param('broadcast_axis_offset', 1)
    s[broadcast_tensor_1].emit_insn(broadcast_tensor_1_axis_1_i, 'dma_copy')
    s[broadcast_tensor_1_ub_001].emit_insn(s[broadcast_tensor_1_ub_001].op.axis[0], 'dma_copy')
    s[broadcast_tensor_1_ub_000].emit_insn(s[broadcast_tensor_1_ub_000].op.axis[0], 'dma_copy')
    s[log_3_ub].emit_insn(s[log_3_ub].op.axis[0], 'vector_ln')
    s[sub_4_ub].emit_insn(s[sub_4_ub].op.axis[0], 'vector_sub')
    s[mul_5_ub].emit_insn(s[mul_5_ub].op.axis[0], 'vector_mul')
    s[mul_6_ub].emit_insn(s[mul_6_ub].op.axis[0], 'vector_muls')
    s[reduce_2_ub].emit_insn(reduce_2_ub_axis_0, 'vector_reduce_sum')

    s[data_labels_ub].emit_insn(s[data_labels_ub].op.axis[0], 'dma_copy')

    if is_labels_broadcast:
        cce_emitinsn_params.cceEmitParamsIns.del_param('broadcast_axis_offset')
        cce_emitinsn_params.cceEmitParamsIns.insert_param(
            'broadcast_axis_offset', 1)
        s[data_labels_broad_ub].emit_insn(s[data_labels_broad_ub].op.axis[1],
                                          'unified_broadcast')

    s[data_labels_cast_ub].emit_insn(s[data_labels_cast_ub].op.axis[0], 'vector_conv')
    s[data_labels_cast].emit_insn(data_labels_cast_axis_1_i, 'dma_copy')
    s[data_features_ub].emit_insn(s[data_features_ub].op.axis[0], 'dma_copy')
    s[data_features_cast_ub].emit_insn(s[data_features_cast_ub].op.axis[0], 'vector_conv')
    s[data_features_cast].emit_insn(data_features_cast_axis_1_i, 'dma_copy')

    s[reduce_2_cast_ub].emit_insn(s[reduce_2_cast_ub].op.axis[0], 'vector_conv')
    s[sub_7_cast_ub].emit_insn(s[sub_7_cast_ub].op.axis[0], 'vector_conv')
    s[sub_7_cast].emit_insn(s[sub_7_cast].op.axis[0], 'dma_copy')

    if block_outer == 1:
        s[reduce_2_cast].emit_insn(s[reduce_2_cast].op.axis[0], 'dma_copy')
    else:
        s[reduce_2_cast].emit_insn(s[reduce_2_cast].op.axis[0],
                                   'dma_copy_softmax_cewl')

    s[div_2_ub].emit_insn(s[div_2_ub].op.axis[0], 'vector_div')
    s[sub_7_ub].emit_insn(s[sub_7_ub].op.axis[0], 'vector_sub')
    s[add_0].emit_insn(add_0_axis_1_i, 'phony_insn')

    s[broadcast_tensor_0_ub].emit_insn(s[broadcast_tensor_0_ub].op.axis[1],
                                       'unified_broadcast')
    s[broadcast_tensor_1_ub].emit_insn(s[broadcast_tensor_1_ub].op.axis[1],
                                       'unified_broadcast')

    # storage_align code
    s[data_labels_ub_001].storage_align(s[data_labels_ub_001].op.axis[0], 8, 0)
    s[data_labels_ub_000].storage_align(s[data_labels_ub_000].op.axis[0], 8, 0)
    s[data_features_ub_001].storage_align(s[data_features_ub_001].op.axis[0], 8, 0)
    s[data_features_ub_000].storage_align(s[data_features_ub_000].op.axis[0], 8, 0)
    s[cast_0_ub].storage_align(s[cast_0_ub].op.axis[0], 16, 0)
    s[broadcast_tensor_0_ub].storage_align(s[broadcast_tensor_0_ub].op.axis[0], 16, 0)
    s[cast_1_ub].storage_align(s[cast_1_ub].op.axis[0], 8, 0)
    s[sub_0_ub].storage_align(s[sub_0_ub].op.axis[0], 8, 0)
    s[sub_0_ub_001].storage_align(s[sub_0_ub_001].op.axis[0], 8, 0)
    s[sub_0_ub_000].storage_align(s[sub_0_ub_000].op.axis[0], 8, 0)
    s[exp_1_ub].storage_align(s[exp_1_ub].op.axis[0], 8, 0)
    s[exp_1_ub_001].storage_align(s[exp_1_ub_001].op.axis[0], 8, 0)
    s[exp_1_ub_000].storage_align(s[exp_1_ub_000].op.axis[0], 8, 0)
    s[broadcast_tensor_1_ub].storage_align(s[broadcast_tensor_1_ub].op.axis[0], 8, 0)
    s[broadcast_tensor_1_ub_001].storage_align(s[broadcast_tensor_1_ub_001].op.axis[0], 8, 0)
    s[broadcast_tensor_1_ub_000].storage_align(s[broadcast_tensor_1_ub_000].op.axis[0], 8, 0)
    s[log_3_ub].storage_align(s[log_3_ub].op.axis[0], 8, 0)
    s[sub_4_ub].storage_align(s[sub_4_ub].op.axis[0], 8, 0)
    s[mul_5_ub].storage_align(s[mul_5_ub].op.axis[0], 8, 0)
    s[mul_6_ub].storage_align(s[mul_6_ub].op.axis[0], 8, 0)
    s[div_2_ub].storage_align(s[div_2_ub].op.axis[0], 8, 0)
    s[sub_7_ub].storage_align(s[sub_7_ub].op.axis[0], 8, 0)

    if is_labels_broadcast:
        s[data_labels_broad_ub].storage_align(s[data_labels_broad_ub].op.axis[0], 8, 0)

    s[data_labels_ub].storage_align(s[data_labels_ub].op.axis[0], 8, 0)
    s[data_labels_cast_ub].storage_align(s[data_labels_cast_ub].op.axis[0], 8, 0)
    s[data_features_ub].storage_align(s[data_features_ub].op.axis[0], 8, 0)
    s[data_features_cast_ub].storage_align(s[data_features_cast_ub].op.axis[0], 8, 0)
    s[reduce_2_cast_ub].storage_align(s[reduce_2_cast_ub].op.axis[0], 8, 0)
    s[sub_7_cast_ub].storage_align(s[sub_7_cast_ub].op.axis[0], 8, 0)

    s.cce_special = dict()
    spec_node_list = [data_labels_cast, data_features_cast,
                      sub_0, exp_1, broadcast_tensor_1]

    # spec_node_list
    s.cce_special["tensor_list"] = spec_node_list
    # the origin out tensor list
    s.cce_special["orign_out_tensor"] = [reduce_2, sub_7]
    # the real out tensor list
    s.cce_special["real_out_tensor"] = [reduce_2, sub_7]

    sch_list[0] = s
    return spec_node_list


def _get_emit_insn_map(tensor):
    """
    get emit insn map
    """
    insn_map = {"elewise_single_cast": "vector_conv",
                "elewise_single_VS_max": "vector_maxs",
                "elewise_single_VS_min": "vector_mins",
                "elewise_single_log": "vector_ln",
                "elewise_single_exp": "vector_exp",
                "elewise_single_relu": "vector_relu",
                "elewise_single_abs": "vector_abs",
                "elewise_single_not": "vector_not",
                "elewise_single_sqrt": "vector_sqrt",
                "elewise_single_rsqrt": "vector_rsqrt",
                "elewise_binary_mul": "vector_mul",
                "elewise_single_VS_mul": "vector_muls",
                "elewise_binary_div": "vector_div",
                "elewise_binary_add": "vector_add",
                "elewise_single_VS_add": "vector_adds",
                "elewise_binary_min": "vector_min",
                "elewise_binary_max": "vector_max",
                "elewise_binary_sub": "vector_sub",
                "reduce_max": "vector_reduce_max",
                "reduce_sum": "vector_reduce_sum",
                "broadcast_for_tensor": "broadcast_for_tensor",
                "last_axis_reduce_max": "last_axis_reduce_max",
                "last_axis_reduce_sum_reuse": "last_axis_reduce_sum_reuse",
                "elewise_binary_sub_scalar_L1": "elewise_binary_sub_scalar_L1",
                "elewise_get_L1_workspace": "elewise_get_L1_workspace",
                }

    if tensor.op.tag.find("|") != -1:
        str_list = tensor.op.tag.split("|")
        insn = insn_map.get(str_list[0])
    else:
        insn = insn_map.get(tensor.op.tag)
    return insn
