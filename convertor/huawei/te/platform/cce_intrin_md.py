#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Runtime function related hooks
"""
# pylint: disable=import-error
from __future__ import absolute_import as _abs
import math
from functools import reduce
from functools import cmp_to_key
from te import tvm
from te.tik.tik_lib.tik_check_util import get_context_msg
from . import cce_conf
from . import cce_emitinsn_params
from . import cce_params as cce
from . import cce_util
from . import conv_buffer_ex

# max repeat time
MAX_CAL_TIMES = 254
# the limit of the nesting depth in the generated CCE,
# if num_segments > MAX_BRACKET_DEPTH, stackoverflow,
# the number was tested by experiment
MAX_BRACKET_DEPTH = 250

BITSIZE_OF_FP16 = 1
BITSIZE_OF_FP32 = 2

# Alignment requirement
ALIGNMENT_BYTES = 32

CALL_TYPE = "handle"

# pylint: too-many-return-statements,too-few-public-methods,too-many-arguments,no-member
# pylint: too-many-statements,no-self-use,too-many-lines,too-many-instance-attributes,too-many-branches
@tvm.register_func("tvm.intrin.cce.inplace_add")
def inplace_add(stmt_op):
    """
    inplace_add
    :stmt_op: the emit_insn stmt
    :return:
    """
    return vec_inplace(stmt_op, "inplace_add")


@tvm.register_func("tvm.intrin.cce.inplace_sub")
def inplace_sub(stmt_op):
    """
    inplace_sub
    :stmt_op: the emit_insn stmt
    :return:
    """
    return vec_inplace(stmt_op, "inplace_sub")


@tvm.register_func("tvm.intrin.cce.inplace_update")
def inplace_update(stmt_op):
    """
    inplace_update
    :stmt_op: the emit_insn stmt
    :return:
    """
    return vec_inplace(stmt_op, "inplace_update")


def vec_inplace(stmt_op, intrinsic_cmd):
    """
    vec_inplace
    :stmt_op: the emit_insn stmt
    :return:
    """
    inplace_ids_all = list(cce_emitinsn_params.cceEmitParamsIns.get_param("inplace_ids"))
    ins, outs = cce_util.get_inplace_buffer(stmt_op, len(inplace_ids_all), intrinsic_cmd)
    outer_axis_var = cce_util.get_segment_outer_axis_var(stmt_op)
    scalar_const = cce_util.get_inplace_var(stmt_op)
    size = cce_util.get_op_lenth(stmt_op)
    inplace_ids = cce_util.get_inplace_ids(stmt_op)

    ir_builder = tvm.ir_builder.create()

    # 1.Tensor vs Scalar: tvm.select((i0.outer == 0), (dataA.local.UB[0]+6.296875h),
    #   dataA.local.UB[0])
    if len(outs[0].shape) == 1 and scalar_const is not None:
        scalar_buf = cce_util.apply_for_new_alloc(ir_builder, outs[0].dtype, (1,), \
                                                  scope=cce.scope_ubuf)
        vec_broadcast(ir_builder, op_cmd="vector_dup", dst_buffers=[scalar_buf], \
                      op_length=1, reset_mask=1, extern_args=[scalar_const])
        with ir_builder.if_scope(outer_axis_var == inplace_ids[0]):
            inplace_intrin_ib_emit(ir_builder, stmt_op, ins[0], scalar_buf, outs[0], \
                                   size, intrinsic_cmd)
        with ir_builder.else_scope():
            inplace_ub2ub_ib_emit(ir_builder, outs[0], ins[0], size)
        return ir_builder.get()

    # 2. Tensor vs Tensor:
    # After IR PASS, donot have tvm.select, and inplace_add/sub/update res = ins[0]
    if (inplace_ids is None) and (len(ins) == 1):
        inplace_ub2ub_ib_emit(ir_builder, outs[0], ins[0], size)

    # After IR PASS, donot have tvm.select,
    # and inplace_add/sub res = ins[0] -/+ ins[1] -/+ ins[2] ...
    elif (inplace_ids is None) and (len(ins) >= 2):
        inplace_intrin_ib_emit(ir_builder, stmt_op, ins[0], ins[1], outs[0], size, intrinsic_cmd)
        for loops in range(2, len(ins)):
            inplace_intrin_ib_emit(ir_builder, stmt_op, outs[0], ins[loops], outs[0], \
                                   size, intrinsic_cmd)

    # After IR PASS, it have tvm.select
    else:
        def recurisive_inplace_ib_gen(idx):
            """
            recurisive_inplace_ib_gen
            """
            if idx >= len(inplace_ids):
                inplace_ub2ub_ib_emit(ir_builder, outs[0], ins[0], size)
                return

            with ir_builder.if_scope(outer_axis_var == inplace_ids[idx]):
                rhs_idx = inplace_ids_all.index(inplace_ids[idx])
                inplace_ids_all[rhs_idx] = -1
                inplace_intrin_ib_emit(ir_builder, stmt_op, ins[0], ins[rhs_idx+1], outs[0], \
                                       size, intrinsic_cmd)
                for _ in range(inplace_ids_all.count(inplace_ids[idx])):
                    rhs_idx = inplace_ids_all.index(inplace_ids[idx])
                    inplace_ids_all[rhs_idx] = -1
                    inplace_intrin_ib_emit(ir_builder, stmt_op, outs[0], ins[rhs_idx+1], outs[0], \
                                           size, intrinsic_cmd)
            with ir_builder.else_scope():
                recurisive_inplace_ib_gen(idx+1)

        def unrecurisive_inplace_ib_gen():
            """
            unrecurisive_inplace_ib_gen
            """
            inplace_ub2ub_ib_emit(ir_builder, outs[0], ins[0], size)
            for idx, _ in enumerate(inplace_ids):
                with ir_builder.if_scope(outer_axis_var == inplace_ids[idx]):
                    rhs_idx = inplace_ids_all.index(inplace_ids[idx])
                    inplace_ids_all[rhs_idx] = -1
                    inplace_intrin_ib_emit(ir_builder, stmt_op, ins[0], ins[rhs_idx+1], outs[0], \
                                           size, intrinsic_cmd)
                    for _ in range(inplace_ids_all.count(inplace_ids[idx])):
                        rhs_idx = inplace_ids_all.index(inplace_ids[idx])
                        inplace_ids_all[rhs_idx] = -1
                        inplace_intrin_ib_emit(ir_builder, stmt_op, outs[0], ins[rhs_idx+1], \
                                               outs[0], size, intrinsic_cmd)

        # the limit of the nesting depth in the generated CCE,
        # if nesting depth  > MAX_BRACKET_DEPTH,
        # stack will overflow, so in this case we generated cce with non-nesting
        if len(inplace_ids) > MAX_BRACKET_DEPTH:
            unrecurisive_inplace_ib_gen()
        else:
            recurisive_inplace_ib_gen(0)

    return ir_builder.get()


def inplace_ub2ub_ib_emit(ir_builder, dst, src, size):
    """
    inplace_ub2ub_ib_emit
    """
    align_factor = cce_util.get_data_alignment(dst.dtype)
    len_burst = (size+align_factor - 1)//align_factor
    # copy 2048KB max once, larger than UB 256KB
    ir_builder.emit(tvm.call_extern(
        dst.dtype, "copy_ubuf_to_ubuf",
        dst.access_ptr("rw"),
        src.access_ptr("r"),
        0,  # sid
        1,  # nburst
        len_burst,  # burst len
        0,  # src_stride
        0  # dst_stride
    ))

# pylint: disable=unused-argument
def inplace_intrin_ib_emit(ir_builder, stmt_op, src0, src1, dst, size, intrinsic_cmd):
    """
    inplace_intrin_ib_emit
    """

    def _vec_intrin_ib_emit(cmd_str):
        block_ele_num = (256*8)//cce_util.get_bits_of(dst.dtype)
        repeat_times = size//block_ele_num  # for compiler 5.6.2
        last_one_length = size - repeat_times*block_ele_num

        # 1.compute by group and each group include 128 elements
        if repeat_times > 0:
            local_repeat_times = repeat_times
            repeat_offset = 0
            while local_repeat_times > 0:
                tmp_repeat_times = min(local_repeat_times, MAX_CAL_TIMES)

                # compute 63.5KB max once, need repeat compute
                ir_builder.emit(tvm.call_extern(
                    dst.dtype, cmd_str,
                    dst.access_ptr('w', offset=repeat_offset),
                    src0.access_ptr('r', offset=repeat_offset),
                    src1.access_ptr('r', offset=repeat_offset),
                    tmp_repeat_times, 1, 1, 1, 8, 8, 8))

                local_repeat_times -= MAX_CAL_TIMES
                repeat_offset += block_ele_num*tmp_repeat_times

        # 2.if the remain of size after repet time
        if last_one_length > 0:
            # set mask
            reset_mask_insn(ir_builder, dst.dtype, bits=last_one_length)
            ir_builder.emit(tvm.call_extern(
                dst.dtype, cmd_str,
                dst.access_ptr('w', offset=repeat_times*block_ele_num),
                src0.access_ptr('r', offset=repeat_times*block_ele_num),
                src1.access_ptr('r', offset=repeat_times*block_ele_num),
                1, 1, 1, 1, 8, 8, 8))

        # 3. reset mask
        reset_mask_insn(ir_builder, dst.dtype)

    if intrinsic_cmd == "inplace_sub":
        _vec_intrin_ib_emit("vsub")
    elif intrinsic_cmd == "inplace_add":
        _vec_intrin_ib_emit("vadd")
    elif intrinsic_cmd == "inplace_update":
        align_factor = cce_util.get_data_alignment(dst.dtype)
        len_burst = (size+align_factor - 1)//align_factor
        # copy 2048KB max once, larger than UB 256KB
        ir_builder.emit(tvm.call_extern(
            dst.dtype, "copy_ubuf_to_ubuf",
            dst.access_ptr("rw"),
            src1.access_ptr("r"),
            0,  # sid
            1,  # nburst
            len_burst,  # burst len
            0,  # src_stride
            0  # dst_stride
        ))
    else:
        raise RuntimeError("not support intrinsic_cmd: %s"%intrinsic_cmd)


@tvm.register_func("tvm.intrin.cce.dma_copy")
def dma_copy(stmt_op):
    """
    dma_copy
    """
    ins, outs, if_var, sel_var = cce_util.get_dma_buffer(stmt_op)
    ir_builder = tvm.ir_builder.create()

    def _gen_sel(ib_in, input_local, vars_local):
        if vars_local:
            with ir_builder.if_scope(tvm.ir_pass.Simplify(vars_local[0].condition)):
                if len(vars_local) > 1:
                    _gen_sel(ib_in, input_local[:len(input_local) - 1], vars_local[1:])
                else:
                    cce_util.dma_copy(ir_builder, input_local[0], outs[0])
            with ir_builder.else_scope():
                cce_util.dma_copy(ir_builder, input_local[-1], outs[0])

    if if_var != []:
        def _gen_if(ib_in, vars_local):
            if vars_local:
                with ir_builder.if_scope(tvm.ir_pass.Simplify(vars_local[0].condition)):
                    if len(vars_local) == 1:
                        if sel_var != []:
                            _gen_sel(ib_in, ins, sel_var)
                        else:
                            cce_util.dma_copy(ir_builder, ins[0], outs[0])
                    else:
                        _gen_if(ib_in, vars_local[1:])

        _gen_if(ir_builder, if_var)
    elif sel_var != []:
        _gen_sel(ir_builder, ins, sel_var)
    else:
        cce_util.dma_copy(ir_builder, ins[0], outs[0])
    return ir_builder.get()


# pylint: disable=too-many-locals
@tvm.register_func("tvm.intrin.cce.elewise_binary_phony")
def elewise_binary_phony(stmt_op):
    """
    elewise_binary_phony which will eliminate its second input tensor
    """
    ins, outs, _, _ = cce_util.get_dma_buffer(stmt_op)
    ir_builder = tvm.ir_builder.create()

    def new_alloc(ir_builder, dtype, shape, name, scope):
        """
        new_alloc
        """
        buf_var = ir_builder.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)

        return new_buffer
    # Move first input to out
    dtype = ins[0].dtype
    total_element = 0
    for dim in ins[0].shape:
        if total_element == 0:
            total_element = dim
        else:
            total_element *= dim
    _block_unit_size = ALIGNMENT_BYTES // cce_util.get_align_factor(dtype)[1]
    total_block = int(total_element) // int(_block_unit_size)
    remain = int(total_element % _block_unit_size)
    if total_block > 0:
        ir_builder.emit(tvm.call_extern(
            ins[0].dtype, "copy_ubuf_to_gm",
            outs[0].access_ptr("rw"),
            ins[0].access_ptr("r"),
            0,
            1,
            total_block,
            0,
            0))
    if remain > 0 and total_block > 0:
        # Roll back for remaining data
        roll_back_size = _block_unit_size - remain
        # Allocate reg buffer needed for holding src data
        reg = new_alloc(ir_builder,
                        ins[0].dtype,
                        (_block_unit_size,),
                        "copy_part",
                        scope=cce.scope_ubuf)
        # reg_mov src data
        with ir_builder.for_range(0, _block_unit_size, name="reg_idx") as reg_idx:
            ir_builder.emit(tvm.call_extern(
                ins[0].dtype, "reg_mov",
                reg.access_ptr("rw", offset=reg_idx),
                ins[0].access_ptr("r", offset=total_block*_block_unit_size-roll_back_size+reg_idx)))
        ir_builder.emit(tvm.call_extern(
            ins[0].dtype, "copy_ubuf_to_gm",
            outs[0].access_ptr("rw", offset=total_block*_block_unit_size-roll_back_size),
            reg.access_ptr("r"),
            0,
            1,
            1,
            0,
            0))
    if remain > 0 and total_block == 0:
        ir_builder.emit(tvm.call_extern(
            ins[0].dtype, "copy_ubuf_to_gm",
            outs[0].access_ptr("rw", offset=0),
            ins[0].access_ptr("r", offset=0),
            0,
            1,
            1,
            0,
            0))
    # Phony input for second input
    phony_buffer = new_alloc(ir_builder,
                             ins[1].dtype,
                             (8,),
                             "elewise_binary_phony_buffer",
                             cce.scope_ubuf)
    ir_builder.emit(tvm.call_extern(
        ins[1].dtype, "copy_gm_to_ubuf",
        phony_buffer.access_ptr("rw"),
        ins[1].access_ptr("r"),
        1,
        1,
        1,
        1,
        1))
    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.group_gm_to_ub")
def group_gm_to_ub(stmt_op):
    """
    this fuction can replace such this ir to intric:
    for (i0.inner, 0, 10) {
        for (i1, 0, 10) {
            var[((i0.inner*10)+i1)] = outputs.local.UB.v0[((i0.inner*10)+i1)]
            mom[((i0.inner*10)+i1)] = outputs.local.UB.v1[((i0.inner*10)+i1)]
            ms[((i0.inner*10)+i1)] = outputs.local.UB.v2[((i0.inner*10)+i1)]
            outputs.v3[((i0.inner*10)+i1)] = outputs.local.UB.v3[((i0.inner*10)+i1)]
        }
    }
    :param stmt_op: the emit_insn stmt
    :return: intrin it
    """
    ins, outs = cce_util.get_buffer(stmt_op)

    if len(ins) != len(outs):
        raise RuntimeError("dma copy dst buffer num must equal to src buffer num")

    if not ins:
        raise RuntimeError("the dma ir is wrong")

    size = cce_util.get_op_lenth(stmt_op)
    ir_builder = tvm.ir_builder.create()

    for i, _ in enumerate(ins):
        align_factor = cce_util.get_data_alignment(outs[i].dtype)
        len_burst = (size+align_factor - 1)//align_factor
        ir_builder.emit(tvm.call_extern(
            outs[i].dtype, "copy_ubuf_to_gm",
            outs[i].access_ptr("rw"),
            ins[i].access_ptr("r"),
            0,
            1,
            len_burst,
            0,
            0))

    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.elewise_single_abs")
def elewise_single_abs(stmt_op):
    """
    :param stmt_op: the stmt of for with abs
    :return: the intric stmt what we want
    """
    return vec_single_elewise(stmt_op, "vabs")


@tvm.register_func("tvm.intrin.cce.elewise_single_log")
def elewise_single_log(stmt_op):
    """
     :param stmt_op: the stmt of for with log
     :return: the intric stmt what we want
     """
    return vec_single_elewise(stmt_op, "vln")


@tvm.register_func("tvm.intrin.cce.elewise_single_exp")
def elewise_single_exp(stmt_op):
    """
        :param stmt_op: the stmt of for with exp
        :return: the intric stmt what we want
    """
    return vec_single_elewise(stmt_op, "vexp")


@tvm.register_func("tvm.intrin.cce.elewise_single_relu")
def elewise_single_relu(stmt_op):
    """
        :param stmt_op: the stmt of for with relu
        :return: the intric stmt what we want
    """
    return vec_single_elewise(stmt_op, "vrelu")


@tvm.register_func("tvm.intrin.cce.elewise_single_not")
def elewise_single_not(stmt_op):
    """
        :param stmt_op: the stmt of for with relu
        :return: the intric stmt what we want
    """
    return vec_single_elewise(stmt_op, "vnot")


@tvm.register_func("tvm.intrin.cce.elewise_single_rec")
def elewise_single_rec(stmt_op):
    """
        :param stmt_op: the stmt of for with rec
        :return: the intric stmt what we want
    """
    return vec_single_elewise(stmt_op, "vrec")


@tvm.register_func("tvm.intrin.cce.elewise_single_sqrt")
def elewise_single_sqrt(stmt_op):
    """
        :param stmt_op: the stmt of for with rec
        :return: the intric stmt what we want
    """
    return vec_single_elewise(stmt_op, "vsqrt")


@tvm.register_func("tvm.intrin.cce.elewise_single_rsqrt")
def elewise_single_rsqrt(stmt_op):
    """
        :param stmt_op: the stmt of for with rec
        :return: the intric stmt what we want
    """
    return vec_single_elewise(stmt_op, "vrsqrt")


@tvm.register_func("tvm.intrin.cce.elewise_single_round")
def elewise_single_round(stmt_op):
    """
        :param stmt_op: the stmt of for with round
        :return: the intric stmt what we want
    """
    dst_stride_m1 = 8
    src_stride_m1 = 4
    intrinsic_cmd = "vconv_f162s32r"
    repeat_cal_dtype = "int32"
    src_dtype, _ = cce_util.get_src_dst_type(stmt_op)

    if src_dtype == "float32":
        intrinsic_cmd = "vconv_f322s32r"
        src_stride_m1 = 8

    args = [1, 1, dst_stride_m1, src_stride_m1]

    return vec_single_elewise(stmt_op, intrinsic_cmd, args, repeat_cal_dtype)


@tvm.register_func("tvm.intrin.cce.elewise_single_floor")
def elewise_single_floor(tensor_op):
    """
        :param tensor_op: the stmt of for with floor
        :return: the intric stmt what we want
    """
    dst_stride_m1 = 8
    src_stride_m1 = 4
    intrinsic_cmd = "vconv_f162s32f"
    repeat_cal_dtype = "int32"
    src_dtype, _ = cce_util.get_src_dst_type(tensor_op)

    if src_dtype == "float32":
        intrinsic_cmd = "vconv_f322s32f"
        src_stride_m1 = 8

    args = [1, 1, dst_stride_m1, src_stride_m1]
    return vec_single_elewise(tensor_op, intrinsic_cmd, args, repeat_cal_dtype)


@tvm.register_func("tvm.intrin.cce.elewise_single_ceil")
def elewise_single_ceil(tensor_op):
    """
        :param tensor_op: the stmt of for with ceil
        :return: the intric stmt what we want
    """
    dst_stride_m1 = 8
    src_stride_m1 = 4
    intrinsic_cmd = "vconv_f162s32c"
    repeat_cal_dtype = "int32"
    src_dtype, _ = cce_util.get_src_dst_type(tensor_op)

    if src_dtype == "float32":
        intrinsic_cmd = "vconv_f322s32c"
        src_stride_m1 = 8

    args = [1, 1, dst_stride_m1, src_stride_m1]
    return vec_single_elewise(tensor_op, intrinsic_cmd, args, repeat_cal_dtype)


@tvm.register_func("tvm.intrin.cce.elewise_single_trunc")
def elewise_single_trunc(tensor_op):
    """
         :param tensor_op: the stmt of for with trunc
         :return: the intric stmt what we want
     """
    dst_stride_m1 = 8
    src_stride_m2 = 4
    intrinsic_cmd = "vconv_f162s32z"
    repeat_cal_dtype = "int32"
    src_dtype, _ = cce_util.get_src_dst_type(tensor_op)

    if src_dtype == "float32":
        intrinsic_cmd = "vconv_f322s32z"
        src_stride_m2 = 8

    args = [1, 1, dst_stride_m1, src_stride_m2]
    return vec_single_elewise(tensor_op, intrinsic_cmd, args, repeat_cal_dtype)


@tvm.register_func("tvm.intrin.cce.elewise_single_cast_s322fp16")
def elewise_single_cast_s322fp16(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return elewise_single_cast(tensor_op)


@tvm.register_func("tvm.intrin.cce.elewise_single_cast")
def elewise_single_cast(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    ins, outs = cce_util.get_buffer(tensor_op)
    src_dtype = ins[0].dtype
    dst_dtype = outs[0].dtype
    src_conv_type = cce_util.dtype2ccetype(src_dtype)
    dst_conv_type = cce_util.dtype2ccetype(dst_dtype)
    intrinsic_cmd = "vconv_"+src_conv_type+"2"+dst_conv_type
    if src_conv_type == 's32' and dst_conv_type == 'f16':
        intrinsic_cmd = "vconv_deq"

    dst_strid_m1 = 8
    src_strid_m1 = 8
    src_bit_len = cce_util.get_bits_of(src_dtype)
    dst_bit_len = cce_util.get_bits_of(dst_dtype)
    repeat_cal_dtype = src_conv_type
    if src_bit_len > dst_bit_len:
        dst_strid_m1 = 4
        repeat_cal_dtype = src_conv_type
    elif src_bit_len < dst_bit_len:
        src_strid_m1 = 4
        repeat_cal_dtype = dst_conv_type
    args = [1, 1, dst_strid_m1, src_strid_m1]
    return vec_single_elewise(tensor_op, intrinsic_cmd, args, repeat_cal_dtype)


def copy_ubuf_to_ubuf_case(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    ins, outs = cce_util.get_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()
    dtype = ins[0].dtype

    lens = cce_util.get_op_lenth(tensor_op)
    align_factor = cce_util.get_align_of(dtype)

    ir_builder.emit(tvm.call_extern(
        outs[0].dtype, "copy_ubuf_to_ubuf",
        outs[0].access_ptr("rw"),
        ins[0].access_ptr("r"),
        0,  # sid
        lens//align_factor,  #
        1,  # len
        0,  # src_stride
        0  # dst_stride
    ))
    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.elewise_binary_scalar_axpy")
def elewise_binary_scalar_axpy(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    ins, _ = cce_util.get_buffer(tensor_op)
    if len(ins) == 1:
        return copy_ubuf_to_ubuf_case(tensor_op)
    return vec_VSsingle_elewise(tensor_op, "vaxpy",
                                cce_util.get_binary_scalar_axpy_extern_args(tensor_op))


@tvm.register_func("tvm.intrin.cce.elewise_single_VS_mul")
def elewise_single_VS_mul(tensor_op):
    # pylint: disable=invalid-name
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    ins, _ = cce_util.get_buffer(tensor_op)
    if not ins:
        return broadcast(tensor_op)
    return vec_VSsingle_elewise(tensor_op, "vmuls", \
                                cce_util.get_elewise_single_vs_extern_args(tensor_op))


@tvm.register_func("tvm.intrin.cce.elewise_single_VS_mul_with_reg_in_quant")
def elewise_single_VS_mul_with_reg_in_quant(tensor_op):
    # pylint: disable=invalid-name
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    ins, outs = cce_util.get_buffer(tensor_op)
    if not ins:
        return broadcast(tensor_op)

    ir_builder = tvm.ir_builder.create()
    reg = ir_builder.allocate(outs[0].dtype, (1,), name="reg_buf_vmuls", scope=cce.scope_reg)
    ir_builder.emit(tvm.call_extern(
        outs[0].dtype, "reg_mov",
        tvm.call_extern(reg.dtype, "reg", reg[0]),
        ins[1].access_ptr("rw"),
    ))
    ir_builder.emit(vec_VSsingle_elewise(tensor_op, "vmuls", [reg[0]]))

    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.elewise_single_VS_adds_with_reg")
def elewise_single_VS_adds_with_reg(tensor_op):
    # pylint: disable=invalid-name
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    ins, outs = cce_util.get_buffer(tensor_op)
    if not ins:
        return broadcast(tensor_op)

    ir_builder = tvm.ir_builder.create()
    reg = ir_builder.allocate(outs[0].dtype, (1,), name="reg_buf_vadds", scope=cce.scope_reg)
    ir_builder.emit(tvm.call_extern(
        outs[0].dtype, "reg_mov",
        tvm.call_extern(reg.dtype, "reg", reg[0]),
        ins[1].access_ptr("rw"),
    ))
    ir_builder.emit(vec_VSsingle_elewise(tensor_op, "vadds", [reg[0]]))

    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.elewise_single_VS_mul_with_reg_sqrt_in_quant")
def elewise_single_VS_mul_with_reg_sqrt_in_quant(tensor_op):
    # pylint: disable=invalid-name
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    ins, outs = cce_util.get_buffer(tensor_op)
    if not ins:
        return broadcast(tensor_op)

    ir_builder = tvm.ir_builder.create()
    reg = ir_builder.allocate(outs[0].dtype, (1,), name="reg_buf_vmuls_sqrt", scope=cce.scope_reg)
    ir_builder.emit(tvm.call_extern(
        outs[0].dtype, "reg_mov",
        tvm.call_extern(reg.dtype, "reg", reg[0]),
        ins[1].access_ptr("r"),
    ))
    # get the load
    load = []

    def _post_order_load(tensor_op):
        if isinstance(tensor_op, tvm.expr.Load):
            load.append(tensor_op)
            return None
        return None

    _ = tvm.ir_pass.IRTransform(tensor_op, None, _post_order_load, ["Load"])

    # Transform the op A = B*C to B = B*C
    # Then we get the vmuls(B, B, reg[0]... ) stmt
    def _post_order_store(_op):
        if isinstance(_op, tvm.stmt.Store):
            new_load = tvm.stmt.Store(ins[0].data, _op.value, load[0].index, tvm.const(3, "uint1"))
            return new_load
        return None

    new_op = tvm.ir_pass.IRTransform(tensor_op, None, _post_order_store, ["Store", "Load"])
    ir_builder.emit(vec_VSsingle_elewise(new_op, "vmuls", [reg[0]]))
    ir_builder.emit(vec_VSsingle_elewise(tensor_op, "vmuls", [reg[0]]))

    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.elewise_single_VS_mul_with_reg")
def elewise_single_VS_mul_with_reg(tensor_op):
    # pylint: disable=invalid-name
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    ins, _ = cce_util.get_buffer(tensor_op)
    if not ins:
        return broadcast(tensor_op)
    args = []

    def _post_order(tensor_op):
        if isinstance(tensor_op, tvm.stmt.Store):
            args.append(tensor_op.value.b)

    _ = tvm.ir_pass.IRTransform(tensor_op, None, _post_order, ["Store"])

    return vec_VSsingle_elewise(tensor_op, "vmuls", [args[0]])


@tvm.register_func("tvm.intrin.cce.elewise_single_VS_add_with_reg")
def elewise_single_VS_add_with_reg(tensor_op):
    # pylint: disable=invalid-name
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    ins, _ = cce_util.get_buffer(tensor_op)
    if not ins:
        return broadcast(tensor_op)
    args = []

    def _post_order(stmt_op):
        if isinstance(stmt_op, tvm.stmt.Store):
            args.append(stmt_op.value.b)

    _ = tvm.ir_pass.IRTransform(tensor_op, None, _post_order, ["Store"])

    return vec_VSsingle_elewise(tensor_op, "vadds", [args[0]])

# pylint: disable=too-many-locals
@tvm.register_func("tvm.intrin.cce.elewise_single_diagonal")
def elewise_single_diagonal(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    body = tensor_op

    def _gen_stmt_in(stmt_op):
        if isinstance(stmt_op, tvm.stmt.AttrStmt):
            stmt_op = stmt_op.body
            stmt_op = _gen_stmt_in(stmt_op)
        if isinstance(stmt_op, tvm.stmt.For):
            return stmt_op
        return None

    body = _gen_stmt_in(body)
    # For reset0, add set_vector_mask and vector_dup
    ins, outs = cce_util.get_buffer(body)
    out_0 = outs[0]
    if not ins:
        return broadcast(body)
    ir_builder = tvm.ir_builder.create()
    mask1 = 1
    dst_stridem0 = 1
    src_stridem0 = 1
    dst_stridem1 = 32
    src_stridem1 = 8
    init_mask = 0x000000000000ffff
    mask0 = 0x01

    for_var = []
    select_var = []

    def _post_order_select(stmt_op):
        if isinstance(stmt_op, tvm.expr.Select):
            select_var.append(stmt_op.condition.a)

    def _post_order_for(stmt_op):
        if isinstance(stmt_op, tvm.stmt.For):
            for_var.append(stmt_op.extent.value)

    _ = tvm.ir_pass.IRTransform(body, None, _post_order_for, ["For"])
    _ = tvm.ir_pass.IRTransform(body, None, _post_order_select, ["Select"])
    if not select_var:
        raise RuntimeError("elewise_single_diagonal select_var is empty ")
    if not for_var:
        raise RuntimeError("elewise_single_diagonal for_var is empty ")
    if len(for_var) == 1:
        repeat = 1  # for(1)
    else:
        repeat = for_var[1]
    mask1 = tvm.const(mask1, "uint64")
    s_l = tvm.expr.Call(mask1.dtype, "shift_left", [mask0, select_var[0]], tvm.expr.Call.Intrinsic,
                        None, 0)
    b_n = tvm.expr.Call(mask1.dtype, "bitwise_not", [s_l], tvm.expr.Call.Intrinsic, None, 0)
    b_a = tvm.expr.Call(mask1.dtype, "bitwise_and", [init_mask, b_n], tvm.expr.Call.Intrinsic, None,
                        0)
    ir_builder.emit(tvm.call_extern(
        outs[0].dtype, "set_vector_mask", b_a,
        b_a))
    ir_builder.emit(tvm.call_extern(
        out_0.dtype, 'vector_dup',
        out_0.access_ptr('rw', offset=0),
        tvm.const(0, out_0.dtype),
        repeat,
        dst_stridem0,  # dst stridem0
        src_stridem0,
        dst_stridem1,  # dst stridem1
        src_stridem1
    ))
    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.elewise_single_VS_add")
def elewise_single_VS_add(tensor_op):
    # pylint: disable=invalid-name
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    ins, outs = cce_util.get_buffer(tensor_op, True)
    if not ins:
        return broadcast(tensor_op)
    if (len(ins) != 1) or (len(outs) != 1):
        raise RuntimeError("elewise_single_VS_add only support ONE src buffer and ONE dst buffer ")

    def _post_order(tensor_op):
        if isinstance(tensor_op, tvm.stmt.Store):
            if hasattr(tensor_op.value, 'b') and cce_util.is_const(tensor_op.value.b):
                return tensor_op
            tensor_op = tvm.make.Store(tensor_op.buffer_var,
                                       tensor_op.value+tvm.const(0, tensor_op.value.dtype),
                                       tensor_op.index,
                                       tensor_op.predicate)
        return tensor_op

    # For stmt like A=B+0 or A=B*1, simplify() will optimize it to A=B.
    # Before emit intrinsics of `vadds`, remember transform it back to A=B+0.
    new_op = tvm.ir_pass.IRTransform(tensor_op, None, _post_order, ["Store"])
    return vec_VSsingle_elewise(new_op, "vadds", cce_util.get_elewise_single_vs_extern_args(new_op))


@tvm.register_func("tvm.intrin.cce.elewise_binary_add")
def elewise_binary_add(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_binary_elewise(tensor_op, "vadd")


@tvm.register_func("tvm.intrin.cce.elewise_binary_sub")
def elewise_binary_sub(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_binary_elewise(tensor_op, "vsub")


@tvm.register_func("tvm.intrin.cce.elewise_binary_mul")
def elewise_binary_mul(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_binary_elewise(tensor_op, "vmul")

@tvm.register_func("tvm.intrin.cce.elewise_binary_min")
def elewise_binary_min(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_binary_elewise(tensor_op, "vmin")


@tvm.register_func("tvm.intrin.cce.elewise_binary_max")
def elewise_binary_max(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_binary_elewise(tensor_op, "vmax")


@tvm.register_func("tvm.intrin.cce.elewise_binary_or")
def elewise_binary_or(tensro_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    ins, _ = cce_util.get_buffer(tensro_op, True)
    if len(ins) == 1:  # if inputs are same
        return copy_ubuf_to_ubuf_case(tensro_op)
    return vec_binary_elewise(tensro_op, "vor")


@tvm.register_func("tvm.intrin.cce.elewise_binary_and")
def elewise_binary_and(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    ins, _ = cce_util.get_buffer(tensor_op, True)
    if len(ins) == 1:  # if inputs are same
        return copy_ubuf_to_ubuf_case(tensor_op)
    return vec_binary_elewise(tensor_op, "vand")


@tvm.register_func("tvm.intrin.cce.elewise_binary_cmp")
def elewise_binary_cmp(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_binary_cmp(tensor_op)


@tvm.register_func("tvm.intrin.cce.elewise_multiple_mla")
def elewise_multiple_mla(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_multiple_elewise(tensor_op, "vmla")


@tvm.register_func("tvm.intrin.cce.elewise_multiple_madd")
def elewise_multiple_madd(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_multiple_elewise(tensor_op, "vmadd")


@tvm.register_func("tvm.intrin.cce.elewise_multiple_maddrelu")
def elewise_multiple_maddrelu(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_multiple_elewise(tensor_op, "vmaddrelu")


@tvm.register_func("tvm.intrin.cce.elewise_multiple_sel")
def elewise_multiple_sel(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_multiple_sel(tensor_op)


@tvm.register_func("tvm.intrin.cce.reduce_last_axis_reduce_max")
def reduce_max_last_axis(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_reduce_last_axis(tensor_op, "vmax")


@tvm.register_func("tvm.intrin.cce.reduce_last_axis_reduce_min")
def reduce_min_last_axis(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_reduce_last_axis(tensor_op, "vmin")


@tvm.register_func("tvm.intrin.cce.reduce_last_axis_reduce_sum")
def reduce_sum_last_axis(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_reduce_last_axis(tensor_op, "vsum")


@tvm.register_func("tvm.intrin.cce.reduce_last_axis_reduce_prod")
def reduce_prod_last_axis(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_reduce_last_axis(tensor_op, "vmul")


@tvm.register_func("tvm.intrin.cce.reduce_nlst_axis_reduce_max")
def reduce_max_nlst_axis(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_reduce_nlst_axis(tensor_op, "vmax")


@tvm.register_func("tvm.intrin.cce.reduce_nlst_axis_reduce_min")
def reduce_min_nlst_axis(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_reduce_nlst_axis(tensor_op, "vmin")


@tvm.register_func("tvm.intrin.cce.reduce_nlst_axis_reduce_sum")
def reduce_sum_nlst_axis(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_reduce_nlst_axis(tensor_op, "vadd")


@tvm.register_func("tvm.intrin.cce.reduce_nlst_axis_reduce_prod")
def reduce_prod_nlst_axis(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_reduce_nlst_axis(tensor_op, "vmul")


@tvm.register_func("tvm.intrin.cce.arg_reduce_last_axis_argmax")
def arg_reduce_last_axis_argmax(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_arg_reduce_last_axis(tensor_op, "argmax")


@tvm.register_func("tvm.intrin.cce.arg_reduce_last_axis_argmin")
def arg_reduce_last_axis_argmin(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_arg_reduce_last_axis(tensor_op, "argmin")


@tvm.register_func("tvm.intrin.cce.arg_reduce_nlst_axis_argmax")
def arg_reduce_nlst_axis_argmax(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_arg_reduce_nlst_axis(tensor_op, "argmax")


@tvm.register_func("tvm.intrin.cce.arg_reduce_nlst_axis_argmin")
def arg_reduce_nlst_axis_argmin(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_arg_reduce_nlst_axis(tensor_op, "argmin")


@tvm.register_func("tvm.intrin.cce.set_padding_ex")
def set_padding_ex(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    block_size_in_b8 = cce.CUBE_MKN["uint8"]["mac"][1]
    offset_pad_left_shift_8bit = cce.CUBE_MKN["uint8"]["mac"][0]*cce.CUBE_MKN["uint8"]["mac"][2]

    def emit_dma_gm_to_ubuf(ir_builder, src, dst, sid=0, n_burst=1, len_burst=0, \
                            src_stride=0, dst_stride=0):
        """
        emit_dma_gm_to_ubuf
        """
        sid = cce_util.get_dma_sid("Sid_copy_gm_to_ubuf")

        ir_builder.emit(tvm.call_extern(
            dst.dtype, "copy_gm_to_ubuf",
            dst,  # dst buffer
            src,  # src buffer
            sid,
            n_burst,
            len_burst,
            src_stride,
            dst_stride
        ))

    def new_alloc(ir_builder, dtype, shape, name, scope):
        """
        new_alloc
        """
        buf_var = ir_builder.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)

        return new_buffer

    ir_builder = tvm.ir_builder.create()

    offset_pad_ubuf = new_alloc(ir_builder, "uint8", (block_size_in_b8,), 'OffsetPad_ubuf',
                                scope=cce.scope_ubuf)

    src_buf = list(conv_buffer_ex.offsetPad.items())
    src = src_buf[0][1].access_ptr('r')
    dst = offset_pad_ubuf.access_ptr('w')
    # DataFlow: move OffsetPad OUT -> UB
    emit_dma_gm_to_ubuf(ir_builder, src, dst, len_burst=1)

    reg = ir_builder.allocate("uint8", (1,), name="reg_buf_padding", scope=cce.scope_reg)
    ir_builder.emit(tvm.call_extern("uint8", "reg_mov", tvm.call_extern(reg.dtype, "reg", reg[0]),
                                    offset_pad_ubuf.access_ptr("r")))
    # let higher 8bit is same with lower 8bit
    padding = reg[0]*offset_pad_left_shift_8bit+reg[0]
    ir_builder.emit(tvm.call_extern(
        "float16", "set_padding",
        padding
    ))

    # reset pad
    offset_pad_rst = 0
    padding = offset_pad_rst
    ibe = tvm.ir_builder.create()
    ibe.emit(tvm.call_extern("uint8", "set_padding", padding))
    return tvm.make.Block(tvm.make.Block(ir_builder.get(), tensor_op), ibe.get())


@tvm.register_func("tvm.intrin.cce.padding_end")
def padding_end(tensor_op):
    """Insert Set Padding reset stmt

        Parameters
        ----------
        tensor_op : inset location

        Returns
        -------
        ret : `|-op
               |-set_padding(0)

    """
    ir_builder = tvm.ir_builder.create()
    ir_builder.emit(tensor_op)

    # The return value is useless.
    ir_builder.emit(tvm.call_extern(
        "float16", "set_padding",
        0,
    ))
    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.segment_max")
def segment_max(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_segment(tensor_op, "segment_max")


@tvm.register_func("tvm.intrin.cce.segment_min")
def segment_min(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_segment(tensor_op, "segment_min")


@tvm.register_func("tvm.intrin.cce.segment_sum")
def segment_sum(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_segment(tensor_op, "segment_sum")


@tvm.register_func("tvm.intrin.cce.segment_mean")
def segment_mean(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_segment(tensor_op, "segment_mean")


@tvm.register_func("tvm.intrin.cce.segment_prod")
def segment_prod(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_segment(tensor_op, "segment_prod")


@tvm.register_func("tvm.intrin.cce.reg_mov")
def reg_mov(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    op_length = cce_util.get_op_lenth(tensor_op)
    ir_builder = tvm.ir_builder.create()
    is_for_init, init_val = cce_util.get_init_val(tensor_op)
    if op_length == 1:
        ins, outs = cce_util.get_buffer(tensor_op)
        out = outs[0]
        ir_builder.emit(tvm.call_extern(
            out.dtype, "reg_mov",
            out.access_ptr('w'),
            tvm.call_extern(out.dtype, "reg", init_val[0])
        ))
    else:
        op_length = cce_util.get_align_oplength(tensor_op)
        ins, outs = cce_util.get_buffer(tensor_op, buffer_shape=(op_length,))
        if is_for_init is False:
            raise RuntimeError("reg_mov must is init stmt")
        args = [1, 1, 8, 8]
        reset_mask = []
        vec_cmd_factory(ir_builder, "vector_dup", ins, outs, op_length, \
                        reset_mask, [init_val[0]], args)
    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.broadcast_for_tensor")
def broadcast_for_tensor(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    intrinsic_cmd = "vector_dup"
    ins, outs = cce_util.get_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()
    reg = ir_builder.allocate(outs[0].dtype, (1,), name="reg_buf", scope=cce.scope_reg)
    ir_builder.emit(tvm.call_extern(
        outs[0].dtype, "reg_mov",
        tvm.call_extern(reg.dtype, "reg", reg[0]),
        ins[0].access_ptr("rw"), ))
    reset_mask = 1

    op_length = cce_util.get_op_lenth(tensor_op)
    align_factor = cce_util.get_data_alignment(outs[0].dtype)

    if op_length%align_factor != 0 and not isinstance(ins[0].elem_offset, tvm.expr.IntImm):
        vec_broadcast_for_no_align(ir_builder, outs[0], op_length, [reg[0]])
        return ir_builder.get()

    vec_broadcast(ir_builder, intrinsic_cmd, outs, cce_util.get_op_lenth(tensor_op), \
                  reset_mask, [reg[0]])

    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.broadcast")
def broadcast(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    intrinsic_cmd = "vector_dup"
    _, outs = cce_util.get_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()
    reset_mask = 1
    is_for_init, init_val = cce_util.get_init_val(tensor_op)
    if not is_for_init:
        raise RuntimeError("reg_mov must is init stmt")

    vec_broadcast(ir_builder, intrinsic_cmd, outs, cce_util.get_op_lenth(tensor_op), \
                  reset_mask, [init_val[0]])

    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.mov_backup")
def mov_backup(tensor_op):
    """This mov is helpful to procss non-aligned data.

        Parameters
        ----------
        tensor_op : the copp stmt of for

        Returns
        -------
        ret : tvm.Tensor
    """
    ins, outs = cce_util.get_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()
    align_factor = cce_util.get_data_alignment(outs[0].dtype)

    size = cce_util.get_op_lenth(tensor_op)
    # if the the size can div align_factor, we copy the result from ub to gm directly
    # else we should backup for the old result in gm.
    # so we copy the old result form gm to ub for backup.then copy the new result from ub to gm.
    # at last we copy the old result form ub to gm for recovery
    if size%align_factor == 0:
        len_burst = size//align_factor
        ir_builder.emit(tvm.call_extern(
            outs[0].dtype, "copy_ubuf_to_gm",
            outs[0].access_ptr("rw"),
            ins[0].access_ptr("r"),
            0,
            1,
            len_burst,
            0,
            0))
    else:
        # because the copy_gm_to_ubuf need address align ,
        # so the copy size is the multiple of align_factor
        len_burst = (size+(align_factor - 1))//align_factor
        tmp_buffer = apply_for_new_alloc(ir_builder, ins[0].dtype, (align_factor,))
        sid = cce_util.get_dma_sid("Sid_copy_gm_to_ubuf")
        ir_builder.emit(tvm.call_extern(
            outs[0].dtype, "copy_gm_to_ubuf",
            tmp_buffer.access_ptr("w"),
            outs[0].access_ptr("rw", offset=size),
            sid,
            1,
            1,
            0,
            0))
        ir_builder.emit(tvm.call_extern(
            outs[0].dtype, "copy_ubuf_to_gm",
            outs[0].access_ptr("rw"),
            ins[0].access_ptr("r"),
            0,
            1,
            len_burst,
            0,
            0))
        ir_builder.emit(tvm.call_extern(
            outs[0].dtype, "copy_ubuf_to_gm",
            outs[0].access_ptr("rw", offset=size),
            tmp_buffer.access_ptr("r"),
            0,
            1,
            1,
            0,
            0))
    return ir_builder.get()


def split_op(ir_builder, repeat_times, src_block, dst_block, idx, func):
    """
    :describe: you can use this function to caculate the emit repeat times and
     emit the intrin by the function params f
    :param ir_builder: ir builder
    :param repeat_times: the repeat times
    :param src_block: src block
    :param dst_block: dst block
    :param func: a function, the intrin you want to emit
    """
    local_repeat_times = repeat_times
    src_roffset = 0
    dst_roffset = 0
    while local_repeat_times > 0:
        if local_repeat_times > MAX_CAL_TIMES:
            tmp_repeat_times = MAX_CAL_TIMES
        else:
            tmp_repeat_times = local_repeat_times

        ir_builder.emit(func(src_roffset, dst_roffset, tmp_repeat_times, idx))
        local_repeat_times -= MAX_CAL_TIMES
        src_roffset += src_block*tmp_repeat_times
        dst_roffset += dst_block*tmp_repeat_times


def reset_mask_insn(ir_builder, type_, bits=128, mask_func=None, ref=None):
    """
    :describe: caculate the mask, and set vector mask
    :param ir_builder: ir builder
    :param type_: the type of mask dst
    :param bits: the bit of mask, default : 128
    """
    # argmin/argmax has his own set_mask func
    if mask_func is not None:
        mask1, mask2 = mask_func(bits)
    else:
        mask1, mask2 = cce_util.set_mask(bits)

    ir_builder.emit(tvm.call_extern(
        type_, "set_vector_mask", tvm.const(mask1, dtype="uint64"),
        tvm.const(mask2, dtype="uint64")))

def reset_mask_insn_inverted(ir_builder, type_, bits, ref=None, mask_func=None):
    """
    :describe: caculate the mask, and set vector mask
    :param ir_builder: ir builder
    :param type_: the type of mask dst
    :param bits: the bit of mask, default : 128
    """
    # argmin/argmax has his own set_mask func
    if bits > ref:
        raise RuntimeError("Reference mask should be larger than target")
    if mask_func is not None:
        mask1, mask2 = mask_func(ref)
        mask1_, mask2_ = mask_func(bits)
        mask1, mask2 = (mask1 - mask1_, mask2 - mask2_)
    else:
        mask1, mask2 = cce_util.set_mask(ref)
        mask1_, mask2_ = cce_util.set_mask(bits)
        mask1, mask2 = (mask1 - mask1_, mask2 - mask2_)

    ir_builder.emit(tvm.call_extern(
        type_, "set_vector_mask", tvm.const(mask1, dtype="uint64"),
        tvm.const(mask2, dtype="uint64")))


def segment_intrin(ir_builder, ins, outs, size, segment_ids, num_segments,
                   init_value, tensor_op, outer_axis):
    """
    :describe: use this func to replace the for_segment_stmt by vcmd
    :param ir_builder: ir builder
    :param ins: input buffer
    :param outs: output buffer
    :param size: repeat times
    :param segment_ids:segment_ids
    :param num_segments:num_segments
    :param init_value: segment_sum : 0, segment_min: Max, segment_min: Min
    :param tensor_op: segment_max,segment_sum,segment_min...
    :param outer_axis: the outer axis
    :return:ir
    """
    # pylint: disable=too-many-locals

    # sort the ins by offset, becase we will use the ins[0] and index to caculate ins[1],ins[2].....
    def ins_key(ele):
        """
        ins_key
        """
        elem_offset = ele.elem_offset
        # segment intrin expected constant value as buffer elem_offset.
        # Things changed when keeping for(1) in scheduleOps.
        # There may be for(i1.c, 0, 1) loops around.
        # Just return 64 for ((i1.c*2)+64), 0 for (i1.c*2).
        while isinstance(elem_offset, tvm.expr.Add):
            if cce_util.is_const(elem_offset.a):
                elem_offset = elem_offset.a
                break
            if cce_util.is_const(elem_offset.b):
                elem_offset = elem_offset.b
                break
            elem_offset = elem_offset.b
        if not cce_util.is_const(elem_offset):
            return 0
        return elem_offset.value

    ins.sort(key=ins_key)

    if not ins:
        shape = (len(ins), outs[0].shape[0])
        buffer_0 = tvm.decl_buffer(shape, outs[0].dtype,
                                   name=outs[0].name,
                                   data=outs[0].data,
                                   offset_factor=1,
                                   data_alignment=16,
                                   scope=cce.scope_ubuf,
                                   elem_offset=0)
    else:
        shape = (len(ins), ins[0].shape[0])
        buffer_0 = tvm.decl_buffer(shape, ins[0].dtype,
                                   name=ins[0].name,
                                   data=ins[0].data,
                                   offset_factor=1,
                                   data_alignment=16,
                                   scope=cce.scope_ubuf,
                                   elem_offset=0)

    out_0 = outs[0]

    # base on the diff data type, get the align_factor
    align_factor = 0
    dtype = buffer_0.dtype
    if dtype in ('int8', 'uint8'):
        align_factor = 32
    elif dtype == "float16":
        align_factor = 16
    else:
        align_factor = 8

    valid_list = [idx for idx in range(len(segment_ids)) if segment_ids[idx] >= 0]
    segment_ids = segment_ids[valid_list[0]:]

    # caculate the real size, because we may align the buffer
    align_vector = ((size+align_factor - 1)//align_factor)*align_factor

    repeat_cal_dtype = out_0.dtype
    block_ele_num = (256*8)//cce_util.get_bits_of(repeat_cal_dtype)
    repeat_times = size//block_ele_num  # for compiler 5.6.2
    last_one_length = size - repeat_times*block_ele_num

    cmd_str = tensor_op.split('_')[-1]
    if cmd_str in ['sum', 'mean']:
        v_cmd = 'vadd'
    elif cmd_str == 'prod':
        v_cmd = 'vmul'
    elif cmd_str == 'min':
        v_cmd = 'vmin'
    elif cmd_str == 'max':
        v_cmd = 'vmax'
    else:
        raise RuntimeError("operation %s not support yet"%tensor_op)

    def segment_ib_emit(ir_builder, i_dim):
        """
        :descripe: make the stmt : if(i0 < i_dim){....}
        :param ir_builder: ir builder
        :param i_dim: the outer_axis scope ,such as i0 < i_dim
        :return:
        """
        # pylint: disable=too-many-locals

        # get the ids from segment_ids that not repeat
        unique_id = []
        for i in segment_ids:
            if i not in unique_id:
                unique_id.append(i)

        dst_stridem0 = 1
        src0_stridem0 = 1
        src1_stridem0 = 1
        dst_stridem1 = 8
        src0_stridem1 = 8
        src1_stridem1 = 8

        if i_dim in unique_id:
            # if there are at least two same segment id,
            # then we will use the at least two data to compute
            if segment_ids.count(i_dim) > 1:
                if repeat_times > 0:
                    local_repeat_times = repeat_times
                    repeat_offset = 0
                    new_segment_id = list(segment_ids)[:]
                    idx0 = new_segment_id.index(i_dim)
                    new_segment_id[idx0] = -1
                    idx1 = new_segment_id.index(i_dim)
                    new_segment_id[idx1] = -1

                    offset0 = idx0*align_vector+repeat_offset
                    offset1 = idx1*align_vector+repeat_offset

                    while local_repeat_times > 0:
                        if local_repeat_times > MAX_CAL_TIMES:
                            tmp_repeat_times = MAX_CAL_TIMES
                        else:
                            tmp_repeat_times = local_repeat_times

                        ir_builder.emit(tvm.call_extern(
                            out_0.dtype, v_cmd,
                            out_0.access_ptr('w', offset=repeat_offset),
                            buffer_0.access_ptr('r', offset=offset0),
                            buffer_0.access_ptr('r', offset=offset1),
                            tmp_repeat_times,
                            dst_stridem0,  # dst stridem0
                            src0_stridem0,
                            src1_stridem0,
                            dst_stridem1,  # dst stridem1
                            src0_stridem1,
                            src1_stridem1
                        ))
                        local_repeat_times -= MAX_CAL_TIMES
                        repeat_offset += block_ele_num*tmp_repeat_times

                    # if there are at least 3 same segment id,
                    # we will use the all same segment id of data
                    if segment_ids.count(i_dim) > 2:
                        local_repeat_times = repeat_times

                        def insn_reduce(src_roffset, dst_roffset, tmp_repeat_times, idx):
                            """
                            insn_reduce
                            """
                            return tvm.call_extern(
                                out_0.dtype, v_cmd,
                                out_0.access_ptr('rw', offset=dst_roffset),
                                out_0.access_ptr('rw', offset=dst_roffset),
                                buffer_0.access_ptr('r', offset=idx*align_vector+src_roffset),
                                tmp_repeat_times,
                                dst_stridem0,  # dst stridem0
                                src0_stridem0,
                                src1_stridem0,
                                dst_stridem1,  # dst stridem1
                                src0_stridem1,
                                src1_stridem1
                            )

                        for _ in range(segment_ids.count(i_dim) - 2):
                            idx = new_segment_id.index(i_dim)
                            new_segment_id[idx] = -1

                            split_op(ir_builder, repeat_times, block_ele_num, block_ele_num, idx,
                                     insn_reduce)

                    if segment_ids.count(i_dim) > 1 and (cmd_str == 'mean'):
                        local_repeat_times = repeat_times
                        dst_stridem0 = 1
                        src_stridem0 = 1
                        dst_stridem1 = 8
                        src_stridem1 = 8

                        def insn_avg(src_roffset, dst_roffset, tmp_repeat_times, idx):
                            # pylint: disable=unused-argument
                            """
                            insn_avg
                            """
                            return tvm.call_extern(
                                out_0.dtype, 'vmuls',
                                out_0.access_ptr('w', offset=dst_roffset),
                                out_0.access_ptr('w', offset=dst_roffset),
                                tvm.const(1.0 / segment_ids.count(i_dim), out_0.dtype),
                                1,
                                dst_stridem0,  # dst stridem0
                                src_stridem0,
                                dst_stridem1,  # dst stridem1
                                src_stridem1
                            )

                        split_op(ir_builder, repeat_times, block_ele_num, \
                                 block_ele_num, 0, insn_avg)

                # if the remain of size after repet time
                if last_one_length > 0:
                    new_segment_id = list(segment_ids)[:]
                    idx0 = new_segment_id.index(i_dim)
                    new_segment_id[idx0] = -1
                    idx1 = new_segment_id.index(i_dim)
                    new_segment_id[idx1] = -1

                    reset_mask_insn(ir_builder, out_0.dtype, bits=last_one_length)
                    ir_builder.emit(tvm.call_extern(
                        out_0.dtype, v_cmd,
                        out_0.access_ptr('w', offset=repeat_times*block_ele_num),
                        buffer_0.access_ptr('r',
                                            offset=idx0*align_vector+repeat_times*block_ele_num),
                        buffer_0.access_ptr('r',
                                            offset=idx1*align_vector+repeat_times*block_ele_num),
                        1,
                        dst_stridem0,  # dst stridem0
                        src0_stridem0,
                        src1_stridem0,
                        dst_stridem1,  # dst stridem1
                        src0_stridem1,
                        src1_stridem1
                    ))

                    if segment_ids.count(i_dim) > 2:
                        for _ in range(segment_ids.count(i_dim) - 2):
                            idx = new_segment_id.index(i_dim)
                            new_segment_id[idx] = -1

                            ir_builder.emit(tvm.call_extern(
                                out_0.dtype, v_cmd,
                                out_0.access_ptr('rw', offset=repeat_times*block_ele_num),
                                out_0.access_ptr('rw', offset=repeat_times*block_ele_num),
                                buffer_0.access_ptr(
                                    'r', offset=idx*align_vector+repeat_times*block_ele_num),
                                1,
                                dst_stridem0,  # dst stridem0
                                src0_stridem0,
                                src1_stridem0,
                                dst_stridem1,  # dst stridem1
                                src0_stridem1,
                                src1_stridem1
                            ))

                    if segment_ids.count(i_dim) > 1 and (cmd_str == 'mean'):
                        local_repeat_times = repeat_times
                        repeat_offset = 0
                        dst_stridem0 = 1
                        src_stridem0 = 1
                        dst_stridem1 = 8
                        src_stridem1 = 8

                        ir_builder.emit(tvm.call_extern(
                            out_0.dtype, 'vmuls',
                            out_0.access_ptr('w', offset=repeat_times*block_ele_num),
                            out_0.access_ptr('w', offset=repeat_times*block_ele_num),
                            tvm.const(1.0 / segment_ids.count(i_dim), out_0.dtype),
                            1,
                            dst_stridem0,  # dst stridem0
                            src_stridem0,
                            dst_stridem1,  # dst stridem1
                            src_stridem1
                        ))

                reset_mask_insn(ir_builder, out_0.dtype)

            # if the segment_ids is unique, we will set B = A[idx] directly
            else:
                idx = segment_ids.index(i_dim)
                len_burst = align_vector//align_factor
                ir_builder.emit(tvm.call_extern(
                    out_0.dtype, 'copy_ubuf_to_ubuf',
                    out_0.access_ptr('w'),
                    buffer_0.access_ptr('r', offset=idx*align_vector),
                    0,
                    1,
                    len_burst,
                    0,
                    0))
                reset_mask_insn(ir_builder, out_0.dtype)
        # if i_dim is not in unique_id,we will set B = init_value
        else:
            tmp_repeat_times = 1
            dst_stridem0 = 1
            src_stridem0 = 1
            dst_stridem1 = 8
            src_stridem1 = 8

            if repeat_times > 0:
                def insn_vector_dup(src_roffset, dst_roffset, tmp_repeat_times, idx):
                    # pylint: disable=unused-argument
                    """
                    insn_vector_dup
                    """
                    return (tvm.call_extern(
                        out_0.dtype, 'vector_dup',
                        out_0.access_ptr('w', offset=dst_roffset),
                        tvm.const(init_value, out_0.dtype),
                        tmp_repeat_times,
                        dst_stridem0,  # dst stridem0
                        src_stridem0,
                        dst_stridem1,  # dst stridem1
                        src_stridem1
                    ))

                split_op(ir_builder, repeat_times, block_ele_num, block_ele_num, 0, \
                         insn_vector_dup)
            if last_one_length > 0:
                reset_mask_insn(ir_builder, out_0.dtype, bits=last_one_length)
                ir_builder.emit(tvm.call_extern(
                    out_0.dtype, 'vector_dup',
                    out_0.access_ptr('w', offset=repeat_times*block_ele_num),
                    tvm.const(init_value, out_0.dtype),
                    1,
                    dst_stridem0,  # dst stridem0
                    src_stridem0,
                    dst_stridem1,  # dst stridem1
                    src_stridem1
                ))

            reset_mask_insn(ir_builder, out_0.dtype)

    def segment_ib_gen(ir_builder, num_segments):
        """
        descripe : the function to generate IR using ir_builder
        """
        # if only one segment,the outer_axis is Null and no need to add if stmt
        # the limit of the nesting depth in the generated CCE,
        # if nesting depth  > MAX_BRACKET_DEPTH,
        # stack will overflow, so in this case we generated cce with non-nesting
        if num_segments > MAX_BRACKET_DEPTH or num_segments == 1:
            unrecurisive_segment_ib_gen(ir_builder, num_segments)
        else:
            recursive_segment_ib_gen(ir_builder, 0, num_segments)

    def unrecurisive_segment_ib_gen(ir_builder, num_segments):
        """
        descripe : non_recurisive situation to generate only if statement with non-nesting
        """
        if num_segments == 1:
            segment_ib_emit(ir_builder, 0)
        else:
            # change the stmt from recursion to non-recursion
            for index in range(num_segments):
                with ir_builder.if_scope(outer_axis == index):
                    segment_ib_emit(ir_builder, index)

    def recursive_segment_ib_gen(ir_builder, start_idx, num_segments):
        """
        descripe : recurisive situation to generate embed if-else with nesting,
                   to decrease the scalar operation
        """
        # start_idx is the value of the segment_id the embed if_else has counted to
        with ir_builder.if_scope(outer_axis == start_idx):
            segment_ib_emit(ir_builder, start_idx)
            if start_idx == num_segments - 1:
                return
        with ir_builder.else_scope():
            recursive_segment_ib_gen(ir_builder, start_idx+1, num_segments)

    segment_ib_gen(ir_builder, num_segments)


def apply_for_new_alloc(ir_builder, dtype, shape, scope=cce.scope_ubuf):
    """
    :descripe: apply an scope buffer block,thie block size is decided by dtype and shape
    :param ir_builder: ir builder
    :param dtype: buffer dtype
    :param shape: buffer shape
    :param scope: buffer scope,such as ub
    :return:
    """
    buf_var = ir_builder.allocate(dtype, shape, name="tmp_buf", scope=scope)
    tmp_buffer = tvm.decl_buffer(shape, buf_var.dtype,
                                 name="tmp_buf",
                                 scope=cce.scope_ubuf,
                                 data=buf_var)
    return tmp_buffer


def vec_single_elewise(tensor_op, intrinsic_cmd, args=None, repeat_cal_dtype=None):
    """

    :param tensor_op:
    :param intrinsic_cmd: the intrinsic_cmd. such as vmax/min...
    :param args: [1,1,8,8].[1,1,8,8,8]
    :param repeat_cal_dtype:the type of deal one times
    :return:
    """
    iter_var = None
    pad = None
    if isinstance(tensor_op, tvm.stmt.AttrStmt):
        iter_var, pad, tensor_op = cce_util.get_pad_info(tensor_op)

    op_len = cce_util.get_op_lenth(tensor_op)

    ins, outs = cce_util.get_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()

    if args is None:
        args = [1, 1, 8, 8]
    reset_mask = 1

    if intrinsic_cmd == "vconv_deq":
        ir_builder.emit(tvm.call_extern("float16", "set_deqscale", tvm.const(1, dtype="float16")))

    vec_cmd_factory(ir_builder, intrinsic_cmd, [ins[0]], [outs[0]], op_len,
                    reset_mask, [], args, iter_var, pad, repeat_cal_dtype)
    return ir_builder.get()


def vec_multiple_elewise(tensor_op, intrinsic_cmd, args=None):
    """

    :param tensor_op:
    :param intrinsic_cmd: the intrinsic_cmd. such as vmla/vmadd...
    :param args: [1,1,8,8].[1,1,8,8,8]
    :param repeat_cal_dtype:the type of deal one times
    :return:
    """

    iter_var = None
    pad = None
    if isinstance(tensor_op, tvm.stmt.AttrStmt):
        iter_var, pad, tensor_op = cce_util.get_pad_info(tensor_op)

    # get the data size
    op_len = cce_util.get_op_lenth(tensor_op)
    ins, outs = cce_util.get_buffer(tensor_op, True)

    if (len(ins) != 3) or (len(outs) != 1):
        raise RuntimeError("vec_binary_elewise only support Three src buffer and ONE dst buffer ")

    if args is None:
        args = [1, 1, 1, 8, 8, 8]

    ir_builder = tvm.ir_builder.create()
    reset_mask = 1

    vec_cmd_factory(ir_builder, intrinsic_cmd, [ins[0], ins[1]], [ins[2]], op_len,
                    reset_mask, [], args, iter_var, pad)
    # for storage rewrite reuse,outs[0] and ins[1] is the same buffer
    ir_builder.emit(tvm.call_extern(outs[0].dtype, "rewrite_inplace", outs[0].access_ptr("w"),
                                    ins[2].access_ptr("r")))
    return ir_builder.get()

# pylint: disable=too-many-locals, invalid-name
def vec_VSsingle_elewise(tensor_op, intrinsic_cmd, extern_args, args=None):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    iter_var = None
    pad = None
    if isinstance(tensor_op, tvm.stmt.AttrStmt):
        iter_var, pad, tensor_op = cce_util.get_pad_info(tensor_op)

    op_len = cce_util.get_op_lenth(tensor_op)
    ins, outs = cce_util.get_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()

    _, _, coef = cce_util.get_opshape_tilesize(tensor_op)

    src_stride = 1
    dst_stride = 1
    src_repeat_stride = 8
    dst_repeat_stride = 8

    # coef is 2 means vmuls(dst, src, scalar)
    # coef is 3 means vmuls(dst, src, reg)
    if len(coef) == 2 or len(coef) == 3:
        if coef[0] >= coef[1]:
            dst_stride = coef[0]//coef[1]
            dst_repeat_stride = dst_stride*8
        else:
            src_stride = coef[1]//coef[0]
            src_repeat_stride = src_stride*8

    if args is None:
        args = [dst_stride,
                src_stride,
                dst_repeat_stride,
                src_repeat_stride]

    reset_mask = 1
    # vmuls and vadds need default value, in case of scalar operand is simplified
    if intrinsic_cmd == "vmuls" and isinstance(extern_args, type(None)):
        extern_args = [tvm.const(cce.DEFAULT_MUL_VALUE, dtype=ins[0].dtype)]
    if intrinsic_cmd == "vadds" and isinstance(extern_args, type(None)):
        extern_args = [tvm.const(cce.DEFAULT_ADD_VALUE, dtype=ins[0].dtype)]

    if intrinsic_cmd != "vaxpy":
        vec_cmd_factory(ir_builder, intrinsic_cmd, [ins[0]], [outs[0]], op_len, reset_mask,
                        extern_args, args, iter_var, pad)
    else:
        if len(ins) < 2:
            raise RuntimeError("vaxpy only support TWO src buffer buffer at least")
        vec_cmd_factory(ir_builder, intrinsic_cmd, [ins[0]], [ins[1]], op_len, reset_mask,
                        extern_args, args, iter_var, pad)

        # for storage rewrite reuse,outs[0] and ins[1] is the same buffer
        ir_builder.emit(tvm.call_extern(outs[0].dtype, "rewrite_inplace", outs[0].access_ptr("w"),
                                        ins[1].access_ptr("r")))

    return ir_builder.get()


def vec_binary_elewise(tensor_op, intrinsic_cmd, args=None):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    iter_var = None
    pad = None
    if isinstance(tensor_op, tvm.stmt.AttrStmt):
        iter_var, pad, tensor_op = cce_util.get_pad_info(tensor_op)

    shape, tile_size, _ = cce_util.get_opshape_tilesize(tensor_op)
    ins, outs = cce_util.get_buffer(tensor_op)

    if (len(outs) == 1) and (not ins):
        return broadcast(tensor_op)
    if (len(ins) != 2) or (len(outs) != 1):
        raise RuntimeError(
            "vec_binary_elewise only support TWO src buffer and ONE dst buffer ")
    if args is None:
        args = [1, 1, 1, 8, 8, 8]

    ir_builder = tvm.ir_builder.create()
    reset_mask = 1

    if shape[1] == shape[2]:
        vec_cmd_factory(ir_builder, intrinsic_cmd, ins, outs, shape[0],
                        reset_mask, [], args, iter_var, pad)
    else:
        bias_accmulate_on_ub_factory(
            ir_builder, intrinsic_cmd, ins, outs, shape, tile_size, reset_mask, args)

    return ir_builder.get()


def vec_binary_elewise_with_ext(tensor_op, intrinsic_cmd, extern_args, args=None):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    iter_var = None
    pad = None
    if isinstance(tensor_op, tvm.stmt.AttrStmt):
        iter_var, pad, tensor_op = cce_util.get_pad_info(tensor_op)

    shape, tile_size, _ = cce_util.get_opshape_tilesize(tensor_op)
    ins, outs = cce_util.get_buffer(tensor_op, True)

    if (len(outs) == 1) and (not ins):
        return broadcast(tensor_op)
    if (len(ins) != 2) or (len(outs) != 1):
        raise RuntimeError(
            "vec_binary_elewise only support TWO src buffer and ONE dst buffer ")
    if args is None:
        args = [1, 1, 1, 8, 8, 8]

    ir_builder = tvm.ir_builder.create()
    reset_mask = 1

    if shape[1] == shape[2]:
        vec_cmd_factory(ir_builder, intrinsic_cmd, ins, outs, shape[0],
                        reset_mask, extern_args, args, iter_var, pad)
    else:
        bias_accmulate_on_ub_factory(
            ir_builder, intrinsic_cmd, ins, outs, shape, tile_size, reset_mask, args)

    return ir_builder.get()


def vec_reduce_nlst_axis(tensor_op, intrinsic_cmd, args=None):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    op_len = cce_util.get_op_lenth(tensor_op)

    is_for_init, init_val = cce_util.get_init_val(tensor_op)
    if is_for_init:
        intrinsic_cmd = "vector_dup"
        args = [1, 1, 8, 8]
    else:
        args = [1, 1, 1, 8, 8, 8]

    ins, outs = cce_util.get_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()

    reset_mask = []

    if intrinsic_cmd == "vector_dup":
        vec_cmd_factory(ir_builder, intrinsic_cmd, ins, outs, op_len, \
                        reset_mask, [init_val[0]], args)
    else:
        vec_cmd_factory(ir_builder, intrinsic_cmd, ins, outs, op_len, reset_mask, [], args)
    return ir_builder.get()


def vec_reduce_last_axis(tensor_op, intrinsic_cmd):
    """
    vec_reduce_last_axis
    """
    init_op = cce_util.get_init_op(tensor_op)
    reduce_op = cce_util.get_reduce_op(tensor_op)
    ir_builder = tvm.ir_builder.create()

    if init_op is not None:
        _, init_val = cce_util.get_init_val(init_op)
        args = [1, 1, 8, 8]
        ins, outs = cce_util.get_buffer(init_op)
        op_len = cce_util.get_op_lenth(init_op)
        vec_cmd_factory(ir_builder, "vector_dup", ins, outs, op_len, [], [init_val[0]], args)

    if reduce_op is not None:
        iter_var = None
        pad = None
        if isinstance(tensor_op, tvm.stmt.AttrStmt):
            iter_var, pad, tensor_op = cce_util.get_pad_info(tensor_op)
        op_len = cce_util.get_op_lenth(reduce_op)

        ins, outs = cce_util.get_buffer(reduce_op)
        for i, _ in enumerate(ins):
            if tvm.ir_pass.Equal(ins[i].data, outs[0].data) and \
                    tvm.ir_pass.Equal(ins[i].elem_offset, outs[0].elem_offset):
                del ins[i]
                break

        reduce_last_axis(ir_builder, intrinsic_cmd, ins, outs, (op_len,), \
                         outs[0].dtype, iter_var, pad)
    return ir_builder.get()


def vec_broadcast_for_no_align(ir_builder, dst_buffer, op_length, extern_args):
    """
    vec_broadcast_for_no_align
    """
    # when the dst_buffer is not 32B aligned, use scalar operation to vector dup data.
    with ir_builder.for_range(0, op_length, name="idx") as idx:
        ir_builder.emit(tvm.call_extern(
            dst_buffer.dtype, "reg_mov",
            dst_buffer.access_ptr("rw", offset=idx),
            tvm.call_extern(dst_buffer.dtype, "reg", extern_args[0])
        ))


def vec_broadcast(ir_builder, op_cmd, dst_buffers, op_length,
                  reset_mask, extern_args, args=None, iter_var=None, pad=None):
    """
    vec_broadcast
    """
    if (len(dst_buffers) != 1) or (len(extern_args) != 1):
        raise RuntimeError("vec_single_elewise only support ONE src buffer and ONE dst buffer ")
    if args is None:
        args = [1, 1, 8, 8]

    if isinstance(extern_args[0], int):
        extern_args[0] = tvm.const(extern_args[0], "int32")
    elif isinstance(extern_args[0], float):
        extern_args[0] = tvm.const(extern_args[0], "float32")

    vec_cmd_factory(ir_builder, op_cmd, [], dst_buffers, op_length,
                    reset_mask, extern_args, args, iter_var, pad)


def vec_segment(tensor_op, intrinsic_cmd):
    """
    vec_segment
    """
    ins, outs = cce_util.get_buffer(tensor_op)
    outer_axis_var = cce_util.get_segment_outer_axis_var(tensor_op)
    segment_ids = cce_emitinsn_params.cceEmitParamsIns.get_param("segment_ids")
    init_value = cce_emitinsn_params.cceEmitParamsIns.get_param("segment_init_value")
    num_segments = cce_emitinsn_params.cceEmitParamsIns.get_param("num_segments")

    ir_builder = tvm.ir_builder.create()
    segment_intrin(ir_builder, ins, outs, cce_util.get_op_lenth(tensor_op), segment_ids,
                   num_segments, init_value, intrinsic_cmd, outer_axis_var)

    return ir_builder.get()


def arg_get_unique_ins(ins, outs):
    """
     get the unique ins that ins can not be same as outs
    :param ins:
    :param outs:
    :return:
    """

    # del same buffer
    def is_same(input_local, outlist):
        """
        is input in outlist
        """
        for elem in outlist:
            if tvm.ir_pass.Equal(input_local.data, elem.data) and \
                    tvm.ir_pass.Equal(input_local.elem_offset, elem.elem_offset):
                return True
        return False

    def ins_key(ele):
        """
        ins_key
        """
        elem_offset = ele.elem_offset
        # segment intrin expected constant value as buffer elem_offset.
        # Things changed when keeping for(1) in scheduleOps.
        # There may be for(i1.c, 0, 1) loops around.
        # Just return 64 for ((i1.c*2)+64), 0 for (i1.c*2).
        while isinstance(elem_offset, tvm.expr.Add):
            if cce_util.is_const(elem_offset.a):
                elem_offset = elem_offset.a
                break
            if cce_util.is_const(elem_offset.b):
                elem_offset = elem_offset.b
                break
            elem_offset = elem_offset.b
        if not cce_util.is_const(elem_offset):
            return 0
        return elem_offset.value

    # del same buffer with outs
    ins = [i for i in ins if not is_same(i, outs)]

    # get the first ins that elem_offset is zero
    ins.sort(key=ins_key)
    if len(ins) > 1:
        ins = [ins[0]]

    return ins

# pylint: disable=too-many-locals
def vec_arg_reduce_last_axis(tensor_op, intrinsic_cmd):
    """
    :param tensor_op: the for_stmt with arg
    :param intrinsic_cmd: arg_max or arg_min
    :return: ir
    """
    split_out_axis_var = None
    pad = None
    # represent we has add pad in realize pass
    if isinstance(tensor_op, tvm.stmt.AttrStmt):
        split_out_axis, pad, tensor_op = cce_util.get_pad_info(tensor_op)
        if split_out_axis is not None:
            split_out_axis_var = split_out_axis.var
    else:
        split_out_axis_var = cce_util.get_arg_outer_axis_var(tensor_op)
    length = cce_util.get_op_lenth(tensor_op)

    if intrinsic_cmd == "argmax":
        intrin_cmd = 'vcmax'
    else:
        intrin_cmd = 'vcmin'

    ins, outs = cce_util.get_buffer(tensor_op)

    ins = arg_get_unique_ins(ins, outs)

    if (len(ins) != 1) or (len(outs) != 2):
        raise RuntimeError(
            "vec_arg_reduce_last_axis only support One src buffer and Two dst buffer ")

    dst_stride = 1
    src0_stride = 1
    src1_stride = 8

    def vccmd_cal(ir_builder, src_buffer, tmp_buffer, src_address_offset, \
                  dst_address_offset, total_len, is_for_first_stage):
        '''
        compute the max/min value and index of the batch data
        '''
        local_total_len = total_len
        # argmin/argmax has his own set_mask func
        if is_for_first_stage:
            reduce_len = 128
            mask_func = cce_util.set_mask
        else:
            reduce_len = 64
            mask_func = cce_util.set_mask_argmax

        repeat_times = local_total_len//reduce_len
        remain_len = local_total_len - repeat_times*reduce_len

        # compute by group and each group include 128 elements
        if repeat_times > 0:
            if not is_for_first_stage:
                reset_mask_insn(ir_builder, tmp_buffer.dtype, reduce_len, mask_func)

            def insn_arg(src_roffset, dst_roffset, tmp_repeat_times, idx):
                # pylint: disable=unused-argument
                """
                insn_arg
                """
                return tvm.call_extern(
                    tmp_buffer.dtype, intrin_cmd,
                    tmp_buffer.access_ptr("rw", offset=dst_address_offset+dst_roffset),
                    src_buffer.access_ptr("r", offset=src_address_offset+src_roffset),
                    tmp_repeat_times,
                    dst_stride,
                    src0_stride,
                    src1_stride
                )

            split_op(ir_builder, repeat_times, reduce_len, 2, 0, insn_arg)

        # process reamin datas
        if remain_len > 0:
            reset_mask_insn(ir_builder, tmp_buffer.dtype, remain_len, mask_func)
            ir_builder.emit(tvm.call_extern(
                tmp_buffer.dtype, intrin_cmd,
                tmp_buffer.access_ptr("rw", offset=dst_address_offset+2*repeat_times),
                src_buffer.access_ptr("r", offset=src_address_offset+repeat_times*128),
                1,
                dst_stride,
                src0_stride,
                src1_stride))

    def update_(local_length, dst_offset_last):
        '''
        update offset infomation
        '''
        dst_offset = dst_offset_last
        src_offset = dst_offset
        dst_offset += (local_length+7)//8*16
        tmp_data = {}
        tmp_data["src_offset"] = src_offset
        tmp_data["dst_offset"] = dst_offset
        tmp_data["length"] = local_length
        return tmp_data

    def cal_index(ir_builder, record_data, buffer_local):
        '''
        compute the real index.
        the index after calling vccmd_cal is uint8 which is the index in 128 elements.
        so we need to restore the index to real index in the batch and all input data.
        '''
        reg = ir_builder.allocate("uint64", (2,), name="reg_buf", scope=cce.scope_reg)
        reg[1] = tvm.const(0, "uint64")
        for data in record_data[::-1][:-1]:
            ir_builder.emit(tvm.call_extern(
                "uint8", "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                buffer_local.access_ptr("rw",
                                        offset=tvm.const(data["src_offset"]+1,
                                                         "uint64")+reg[1])))

            reg[1] = reg[1]*tvm.const(64, "uint64")+reg[0]
        data = record_data[0]
        ir_builder.emit(tvm.call_extern(
            "uint8", "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[0]),
            buffer_local.access_ptr("rw",
                                    offset=tvm.const(data["src_offset"]+1,
                                                     "uint64")+reg[1])))

        ir_builder.emit(tvm.call_extern(
            "float16", "reg_mov",
            buffer_local.access_ptr("rw"),
            buffer_local.access_ptr("rw",
                                    offset=tvm.const(data["src_offset"],
                                                     "uint64")+reg[1])))

        # in the 2nd stage, the index is in even bit and mask is 0101.. ,
        # that is to say the index has multiplied 2 ,so there is multiply 64 instead of 128
        reg[1] = reg[1]*tvm.const(64, "uint64")+reg[0]
        if split_out_axis_var is not None:
            reg[1] = reg[1]+split_out_axis_var.astype("uint64")*tvm.const(length, "uint64")

        return reg[1]

    # update current value of min/max number
    def cache_val_and_idx(ir_builder, reg_res, tmp_val_buffer, offset):
        '''
        cache the max/min value and index of one batch data
        for first times, the value write to (tmp_buffer+0), the index write to (tmp_buffer+2)
        for other times, the value write to (tmp_buffer+1), the index write to (tmp_buffer+4)
        note: tmp_buffer is float16*
        '''

        ir_builder.emit(tvm.call_extern(
            "float16", "reg_mov",
            tmp_val_buffer.access_ptr("rw", offset=offset),
            tmp_val_buffer.access_ptr("rw")))

        ir_builder.emit(tvm.call_extern(
            "int32", "reg_mov",
            tmp_val_buffer.access_ptr("rw", offset=2+2*offset),
            tvm.call_extern(reg_res.dtype, "reg", reg_res)))

    # update current index of min/max number
    def renew_idx(ir_builder, tmp_buffer, idx_buffer, val_buffer):
        '''
        from batch 2,compare the value of this batch with the value of previous batch
        the loaction of data is:
        address:  | buffer+0  | buffer+1 | buffer+2 | buffer+3 | buffer+4 | buffer+5 |
        data:     | value_0   | value_1  |       index_0       |      index_1        |
        note:
            value_0 is value of previous batch (float16)
            value_1 is value of this batch (float16)
            index_0 is index of previous batch (int32)
            index_1 is index of this batch (int32)
            buffer(float16*) is the starting address of input data
        compore two value, and the result max/min value in (buffer+0) and
        the result index is in (buffer+1)(uint8) that value is 0 or 1.
        get the index in buffer+1 and decide the final index is index_0 or index_1
        '''
        reg = ir_builder.allocate("uint64", (1,), name="reg_buf", scope=cce.scope_reg)

        # get the max/min value of previous batch from val_buffer and write to tmp_buffer
        ir_builder.emit(tvm.call_extern(
            "float16", "reg_mov",
            tmp_buffer.access_ptr("rw"),
            val_buffer.access_ptr("r"), ))

        # get the max/min index of previous batch from idx_buffer and write to tmp_buffer
        ir_builder.emit(tvm.call_extern(
            "int32", "reg_mov",
            tmp_buffer.access_ptr("rw", offset=2),
            idx_buffer.access_ptr("r")))

        # set mask to 11, make 2 value compare
        reset_mask_insn(ir_builder, tmp_buffer.dtype, bits=2)
        ir_builder.emit(tvm.call_extern(
            tmp_buffer.dtype, intrin_cmd,
            tmp_buffer.access_ptr("rw"),
            tmp_buffer.access_ptr("r"),
            1,
            dst_stride,
            src0_stride,
            src1_stride))

        # compare 2 value(the max/min value of previous batch and the max/min value of this batch)
        ir_builder.emit(tvm.call_extern(
            "uint8", "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[0]),
            tmp_buffer.access_ptr("rw", offset=1)))

        # get the max/min index of two values in ((*float16)tmp_buffer+1)
        # and write to reg[0], reg[0] = 0 or 1
        ir_builder.emit(tvm.call_extern(
            "int32", "reg_mov",
            tmp_buffer.access_ptr("rw", offset=2),
            tmp_buffer.access_ptr("rw", offset=2+reg[0]*2)))

    # get the final index of two indexes in ((*float16)tmp_buffer+2+2*reg[0])
    # and write to ((*float16)tmp_buffer+2)
    def write_res(ir_builder, idx_buffer, val_buffer, tmp_buffer):
        '''
        after getting the max/min value and index of each batch,
        write the value and index to val_vuffer and idx_buffer
        '''
        # get the max/min value of all already computed data from tmp_buffer
        # and write it to val_buffer

        ir_builder.emit(tvm.call_extern(
            val_buffer.dtype, "reg_mov",
            val_buffer.access_ptr("rw"),
            tmp_buffer.access_ptr("r")))

        # get the max/min index of all already computed data from tmp_buffer
        # and write it to val_buffer
        ir_builder.emit(tvm.call_extern(
            idx_buffer.dtype, "reg_mov",
            idx_buffer.access_ptr("rw"),
            tmp_buffer.access_ptr("r", offset=2)))

    ins_0 = ins[0]
    idx_b = outs[0]
    val_b = outs[1]

    ir_builder = tvm.ir_builder.create()

    def instr(flag, pad=None):
        """
        insn_arg
        """
        record_data = []
        local_length = length

        if pad is not None:
            local_length = local_length - pad.value

        src_offset = 0
        dst_offeset = 0

        tmp_factor = 16
        tmp_shape = [((i.value+tmp_factor - 1)//tmp_factor) for i in ins_0.shape]
        tmp_shape[-1] = 2
        total_size = local_length
        while total_size > 1:
            total_size = (total_size + 63) // 64
            tmp_shape[-1] += (total_size + 7) // 8 * 16
        tmp_buffer = apply_for_new_alloc(ir_builder, ins_0.dtype, tmp_shape)
        ir_builder.allocate("uint64", (1,), name="reg_buf", scope=cce.scope_reg)
        # stage1: split the data to many gropus,every group include 128 elements,
        # compute the max/min value and index in each group
        vccmd_cal(ir_builder, ins_0, tmp_buffer, src_offset, dst_offeset, local_length,
                  is_for_first_stage=True)
        local_length = (local_length+127)//128

        tmp_data = update_(local_length, dst_offeset)
        record_data.append(tmp_data)
        while local_length > 1:
            # stage2: compute the max/min value and index of intermediate result
            vccmd_cal(ir_builder, tmp_buffer, tmp_buffer, tmp_data["src_offset"],
                      tmp_data["dst_offset"], local_length, is_for_first_stage=False)
            local_length = (local_length+63)//64
            tmp_data = update_(local_length, tmp_data["dst_offset"])
            record_data.append(tmp_data)

        # compute the real index
        index_reg = cal_index(ir_builder, record_data, tmp_buffer)
        if flag == "body":
            cache_val_and_idx(ir_builder, index_reg, tmp_buffer, 1)
            renew_idx(ir_builder, tmp_buffer, idx_b, val_b)
            write_res(ir_builder, idx_b, val_b, tmp_buffer)
        elif flag == "init":
            cache_val_and_idx(ir_builder, index_reg, tmp_buffer, 0)
            write_res(ir_builder, idx_b, val_b, tmp_buffer)
        reset_mask_insn(ir_builder, tmp_buffer.dtype)

    # if has split the axis
    if split_out_axis_var is not None:
        # if pad is not None, represents that we had add pad in realize pass
        if pad is not None and int(pad.value) != 0:
            # if pad is not None, we only need caculate the length - pad.value in last forloop
            with ir_builder.if_scope(split_out_axis_var == split_out_axis.dom.extent - 1):
                instr("body", pad)
            with ir_builder.else_scope():
                with ir_builder.if_scope(split_out_axis_var > 0):
                    instr("body")
                # if split_out_axis_var = 0, should init
                with ir_builder.else_scope():
                    instr("init")
        else:
            with ir_builder.if_scope(split_out_axis_var > 0):
                instr("body")
            # if split_out_axis_var = 0, should init
            with ir_builder.else_scope():
                instr("init")
    else:
        instr("init")

    return ir_builder.get()

# pylint: disable=too-many-locals
def vec_arg_reduce_nlst_axis(tensor_op, intrinsic_cmd):
    """
    :param tensor_op: the for_stmt whit arg
    :param intrinsic_cmd: arg_max or arg_min
    :return:
    """
    if isinstance(tensor_op, tvm.stmt.AttrStmt):
        _, _, tensor_op = cce_util.get_pad_info(tensor_op)

    length = cce_util.get_op_lenth(tensor_op)

    if intrinsic_cmd.lower() == "argmax":
        v_cmd = 'vmax'
        vcmp_intrin = "vcmp_gt"
    else:
        v_cmd = 'vmin'
        vcmp_intrin = "vcmp_lt"

    ins, outs = cce_util.get_buffer(tensor_op)

    ins = arg_get_unique_ins(ins, outs)

    is_for_init, _ = cce_util.get_init_val(tensor_op)

    stride_inside = 1
    stride_outside = 8
    total_len = length
    buist_length = 16

    # update the index of the current max/min value
    def update_val(ir_builder, src_val_buffer, dst_val_buffer, is_for_init):
        """
        update_val
        """
        local_total_len = total_len
        if is_for_init:
            burst = (length+buist_length - 1)//buist_length
            ir_builder.emit(tvm.call_extern(
                dst_val_buffer.dtype, "copy_ubuf_to_ubuf",
                dst_val_buffer.access_ptr("rw"),
                src_val_buffer.access_ptr("r"),
                0,  # sid
                1,  # nburst
                burst,  # burst len
                stride_inside,  # src_stride
                stride_inside  # dst_stride
            ))
        else:
            src_buffers = [src_val_buffer, dst_val_buffer]
            dst_buffers = [dst_val_buffer]
            args = [1, 1, 1, 8, 8, 8]
            vec_cmd_factory(ir_builder, v_cmd, src_buffers, dst_buffers, local_total_len,
                            False, [], args)

    # update the index of the current index of max/min value
    def update_idx(ir_builder, src_val_buffer, dst_val_buffer, dst_idx_buffer, \
                   idx_var, is_for_init):
        """
        update_idx
        """
        # pylint: disable=too-many-locals
        # by default, set all 64 bits of the mask to 1,  18446744073709551615 = 2**64-1
        default_template_mask = 18446744073709551615

        def idx_cal(ir_builder, mask_buffer, dst_idx_buffer, idx_var, offset_mask, offset_idx,
                    template_mask=default_template_mask):
            """
            idx_cal
            """
            if template_mask != 0:
                ir_builder.emit(tvm.call_extern(
                    "uint64", "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    mask_buffer.access_ptr("rw", offset=offset_mask)))

                if template_mask != default_template_mask:
                    reg[0] = reg[0] & tvm.const(template_mask, "uint64")
                with ir_builder.if_scope(reg[0] != 0):
                    ir_builder.emit(tvm.call_extern(
                        dst_idx_buffer.dtype, "set_vector_mask", tvm.const(0, dtype="uint64"),
                        reg[0]))
                    ir_builder.emit(tvm.call_extern(
                        dst_idx_buffer.dtype, "vector_dup",
                        dst_idx_buffer.access_ptr("rw", offset=offset_idx),
                        idx_var,
                        1,
                        stride_inside,  # dst stridem0
                        stride_inside,
                        stride_outside,  # dst stridem1
                        stride_outside))

        local_total_len = total_len

        # init the beginning current min/max value
        if is_for_init:
            op_cmd = 'vector_dup'
            dst_buffers = [dst_idx_buffer]
            op_length = local_total_len
            if_reset_mask = True
            extern_args = [idx_var]
            vec_broadcast(ir_builder, op_cmd, dst_buffers, op_length,
                          if_reset_mask, extern_args)
        # update the index and value of the min/max
        else:
            reg = ir_builder.allocate("uint64", (1,), name="reg_buf", scope=cce.scope_reg)
            tmp_buffer = apply_for_new_alloc(ir_builder, "uint64", (2,))  # 128 bit length
            block_ele_num_idx = 64
            block_ele_num_cmp = 128
            repeat_times_cmp = length//block_ele_num_cmp
            last_one_length = length - repeat_times_cmp*block_ele_num_cmp
            if repeat_times_cmp > 0:
                with ir_builder.for_range(0, repeat_times_cmp, name="i_inner") as i_inner:
                    ir_builder.emit(tvm.call_extern(
                        dst_val_buffer.dtype, vcmp_intrin,
                        src_val_buffer.access_ptr("r", offset=i_inner*block_ele_num_cmp),
                        dst_val_buffer.access_ptr("r", offset=i_inner*block_ele_num_cmp),
                        1,
                        0,  # dst1 stridem0, fake
                        stride_inside,  # src0 stridem0
                        stride_inside,  # src1 stridem0
                        0,  # dst1 stridem0, fake
                        stride_outside,  # src0 stridem1
                        stride_outside  # src1 stridem1
                    ))
                    ir_builder.emit(tvm.call_extern(
                        tmp_buffer.dtype, "get_cmpmask",
                        tmp_buffer.access_ptr("wr")))

                    idx_cal(ir_builder, tmp_buffer, dst_idx_buffer, idx_var, 0,
                            i_inner*block_ele_num_cmp)
                    idx_cal(ir_builder, tmp_buffer, dst_idx_buffer, idx_var, 1,
                            i_inner*block_ele_num_cmp+block_ele_num_idx)
                    reset_mask_insn(ir_builder, dst_val_buffer.dtype)

            if last_one_length > 0:
                reset_mask_insn(ir_builder, dst_val_buffer.dtype, bits=last_one_length)
                ir_builder.emit(tvm.call_extern(
                    dst_val_buffer.dtype, vcmp_intrin,
                    src_val_buffer.access_ptr("r", offset=repeat_times_cmp*block_ele_num_cmp),
                    dst_val_buffer.access_ptr("r", offset=repeat_times_cmp*block_ele_num_cmp),
                    1,
                    0,  # dst1 stridem0, fake
                    stride_inside,  # src0 stridem0
                    stride_inside,  # src1 stridem0
                    0,  # dst1 stridem0, fake
                    stride_outside,  # src0 stridem1
                    stride_outside  # src1 stridem1
                ))
                ir_builder.emit(tvm.call_extern(
                    tmp_buffer.dtype, "get_cmpmask",
                    tmp_buffer.access_ptr("wr")))

                mask1, mask2 = cce_util.set_mask(last_one_length)

                idx_cal(ir_builder, tmp_buffer, dst_idx_buffer, idx_var, 0,
                        repeat_times_cmp*block_ele_num_cmp, template_mask=mask2)
                idx_cal(ir_builder, tmp_buffer, dst_idx_buffer, idx_var,
                        1, repeat_times_cmp*block_ele_num_cmp+block_ele_num_idx,
                        template_mask=mask1)

    ir_builder = tvm.ir_builder.create()

    if is_for_init:
        return ir_builder.get()

    ins_0 = ins[0]
    idx_b = outs[0]
    val_b = outs[1]

    out_axis_var = cce_util.get_argnlst_outaxis(tensor_op)

    def instr(flag):
        """
        instr
        """
        # if i = 0, we should set out_index = init_index. out_value = init_value
        if flag == "body":
            reset_mask_insn(ir_builder, val_b.dtype)
            update_idx(ir_builder, ins_0, val_b, idx_b, out_axis_var, False)
            reset_mask_insn(ir_builder, val_b.dtype)
            update_val(ir_builder, ins_0, val_b, False)
        elif flag == "init":
            update_idx(ir_builder, ins_0, val_b, idx_b, 0, True)
            update_val(ir_builder, ins_0, val_b, True)

    with ir_builder.if_scope(out_axis_var > 0):
        instr("body")
    with ir_builder.else_scope():
        instr("init")

    return ir_builder.get()

# pylint: disable=too-many-locals
def vec_repeat_elewise(ir_builder, op_cmd, src_buffers, dst_buffers, local_total_len, cal_once_len,
                       reset_mask, extern_args, args):
    """
    vec_repeat_elewise
    """
    dst_dtype = dst_buffers[0].dtype
    repeat_times = local_total_len//cal_once_len
    remain_len = local_total_len - repeat_times*cal_once_len
    reduce_factor = 1

    def __apply_for_new_alloc(ir_builder, dtype, shape, scope=cce.scope_ubuf):
        buf_var = ir_builder.allocate(dtype, shape, name="tmp_buf", scope=scope)
        tmp_buffer = tvm.decl_buffer(shape, buf_var.dtype,
                                     name="tmp_buf",
                                     scope=scope,
                                     data=buf_var)
        return tmp_buffer

    if op_cmd.lower().find("vcmp_") != -1:
        dup_repeat_times = local_total_len//cal_once_len
        scalar_buf = cce_util.apply_for_new_alloc(ir_builder, dst_dtype,
                                                  src_buffers[0].shape,
                                                  scope=cce.scope_ubuf)
        ir_builder.emit(tvm.call_extern(dst_dtype, "vector_dup",
                                        scalar_buf.access_ptr("rw"),
                                        extern_args[0],
                                        dup_repeat_times if (dup_repeat_times != 0) else 1,
                                        1, 1, 8, 8))
        bias_buf = cce_util.apply_for_new_alloc(ir_builder, dst_dtype,
                                                src_buffers[0].shape,
                                                scope=cce.scope_ubuf)
        ir_builder.emit(tvm.call_extern(dst_dtype, "vector_dup",
                                        bias_buf.access_ptr("rw"),
                                        tvm.const(0, dtype=dst_dtype),
                                        dup_repeat_times if (dup_repeat_times != 0) else 1,
                                        1, 1, 8, 8))
    elif op_cmd == 'vcond':
        temp_thredhold = tvm.const(float(extern_args[1]), dst_dtype)
        temp_bias = tvm.const(float(extern_args[2]), dst_dtype)
        thred_buf = __apply_for_new_alloc(ir_builder, dst_dtype,
                                          (cal_once_len,), scope=cce.scope_ubuf)
        ir_builder.emit(tvm.call_extern(
            dst_dtype, "vector_dup",
            thred_buf.access_ptr("rw"),
            temp_thredhold, 1, 1, 1, 8, 8))

        bias_buf = __apply_for_new_alloc(ir_builder, dst_dtype, (cal_once_len,),
                                         scope=cce.scope_ubuf)
        ir_builder.emit(tvm.call_extern(
            dst_dtype, "vector_dup",
            bias_buf.access_ptr("rw"),
            temp_bias, 1, 1, 1, 8, 8))
    elif op_cmd == 'vlogic':
        temp_a0 = __apply_for_new_alloc(ir_builder, "float16",
                                        src_buffers[0].shape,
                                        scope=cce.scope_ubuf)
        temp_a1 = __apply_for_new_alloc(ir_builder, "float16",
                                        src_buffers[0].shape,
                                        scope=cce.scope_ubuf)
        if extern_args[0] == 'or' or extern_args[0] == 'not':
            thred_buf = __apply_for_new_alloc(ir_builder, "float16",
                                              src_buffers[0].shape,
                                              scope=cce.scope_ubuf)
            vec_cmd_factory(ir_builder, "vector_dup", [], [thred_buf],
                            local_total_len, 1,
                            [tvm.const(0.0, "float16")],
                            [1, 1, 8, 8])

            bias_buf = __apply_for_new_alloc(ir_builder, "float16",
                                             src_buffers[0].shape,
                                             scope=cce.scope_ubuf)
            vec_cmd_factory(ir_builder, "vector_dup", [], [bias_buf],
                            local_total_len, 1,
                            [tvm.const(1.0, "float16")],
                            [1, 1, 8, 8])
            reset_mask_insn(ir_builder, dst_dtype)
            temp_andor_out = __apply_for_new_alloc(ir_builder, "float16",
                                                   src_buffers[0].shape,
                                                   scope=cce.scope_ubuf)
        temp_out = __apply_for_new_alloc(ir_builder, "float16",
                                         src_buffers[0].shape,
                                         scope=cce.scope_ubuf)
    if repeat_times > 0:
        if op_cmd.find("vcmp_") != -1:
            with ir_builder.for_range(0, repeat_times,
                                      name="cmp_index") as cmp_index:
                repeat_offset = cal_once_len*cmp_index
                ir_builder.emit(tvm.call_extern(dst_dtype,
                                                op_cmd,
                                                src_buffers[0].access_ptr(
                                                    "r", offset=repeat_offset),
                                                scalar_buf.access_ptr(
                                                    "r", offset=repeat_offset),
                                                1, 1, 1, 1, 8, 8, 8))
                ir_builder.emit(tvm.call_extern(dst_dtype,
                                                "vsel",
                                                dst_buffers[0].access_ptr(
                                                    "rw", offset=repeat_offset),
                                                src_buffers[1].access_ptr(
                                                    "r", offset=repeat_offset),
                                                bias_buf.access_ptr("r"),
                                                1, 1, 1, 1, 8, 8, 8))
        else:
            def insn_concat_args(src_roffset, dst_roffset, tmp_repeat_times,
                                 idx):
                # pylint: disable=unused-argument
                """
                insn_concat_args
                """
                # for pylint, reserve unified entry
                tmp_args = cce_util.concat_args(src_buffers, dst_buffers,
                                                src_roffset, dst_roffset,
                                                tmp_repeat_times,
                                                extern_args, args)
                return tvm.call_extern(dst_dtype, op_cmd, *tmp_args)

            if op_cmd in ('vcond', 'vcmpsel', 'vlogic'):
                local_repeat_times = repeat_times
                repeat_src_offset = 0
                repeat_dst_offset = 0
                while local_repeat_times > 0:
                    if local_repeat_times > MAX_CAL_TIMES:
                        tmp_repeat_times = MAX_CAL_TIMES
                    else:
                        tmp_repeat_times = local_repeat_times

                    if op_cmd == 'vcond':
                        # vsel only can process 128 numbers
                        with ir_builder.for_range(0,
                                                  tmp_repeat_times,
                                                  name="cmp_index") as cmp_index:
                            tmp_src_repeat_offset = cal_once_len*cmp_index
                            tmp_dst_repeat_offset = (cal_once_len//reduce_factor)*cmp_index
                            ir_builder.emit(tvm.call_extern(
                                dst_dtype, "vcmp_"+extern_args[0],
                                src_buffers[0].access_ptr(
                                    "r", offset=tmp_src_repeat_offset),
                                thred_buf.access_ptr("r"), 1, 1, 1, 1, 8, 8, 8))

                            ir_builder.emit(tvm.call_extern(
                                dst_dtype, "vsel",
                                dst_buffers[0].access_ptr(
                                    "rw", offset=tmp_dst_repeat_offset),
                                src_buffers[0].access_ptr(
                                    "r", offset=tmp_src_repeat_offset),
                                bias_buf.access_ptr("r"),
                                1, 1, 1, 1, 8, 8, 8))
                    elif op_cmd == 'vcmpsel':
                        # vsel only can process 128 numbers
                        with ir_builder.for_range(0, tmp_repeat_times,
                                                  name="cmp_index") as cmp_index:
                            tmp_src_repeat_offset = cal_once_len*cmp_index
                            tmp_dst_repeat_offset = (cal_once_len//reduce_factor)*cmp_index
                            ir_builder.emit(tvm.call_extern(
                                dst_dtype, "vcmp_"+extern_args[0],
                                src_buffers[0].access_ptr(
                                    "r",
                                    offset=tmp_src_repeat_offset),
                                src_buffers[1].access_ptr(
                                    "r",
                                    offset=tmp_src_repeat_offset),
                                1, 1, 1, 1, 8, 8, 8))

                            ir_builder.emit(tvm.call_extern(
                                dst_dtype, "vsel",
                                dst_buffers[0].access_ptr(
                                    "rw",
                                    offset=tmp_dst_repeat_offset),
                                src_buffers[0].access_ptr(
                                    "r",
                                    offset=tmp_src_repeat_offset),
                                src_buffers[1].access_ptr(
                                    "r",
                                    offset=tmp_src_repeat_offset),
                                1, 1, 1, 1, 8, 8, 8))
                    elif op_cmd == 'vlogic':
                        ir_builder.emit(tvm.call_extern(
                            "float16", "vconv_s82f16",
                            temp_a0.access_ptr("w", offset=repeat_src_offset),
                            src_buffers[0].access_ptr(
                                "r",
                                offset=repeat_src_offset),
                            tmp_repeat_times, 1, 1, 8, 4))
                        if extern_args[0] != 'not':
                            ir_builder.emit(tvm.call_extern(
                                "float16", "vconv_s82f16",
                                temp_a1.access_ptr(
                                    "w",
                                    offset=repeat_src_offset),
                                src_buffers[1].access_ptr(
                                    "r",
                                    offset=repeat_src_offset),
                                tmp_repeat_times, 1, 1, 8, 4))

                        if extern_args[0] == 'and':
                            ir_builder.emit(tvm.call_extern(
                                "float16", "vmul",
                                temp_out.access_ptr("rw",
                                                    offset=repeat_dst_offset),
                                temp_a0.access_ptr("r",
                                                   offset=repeat_src_offset),
                                temp_a1.access_ptr("r",
                                                   offset=repeat_src_offset),
                                tmp_repeat_times, 1, 1, 1, 8, 8, 8))
                        elif extern_args[0] == 'or' or extern_args[0] == 'not':
                            if extern_args[0] == 'or':
                                ir_builder.emit(
                                    tvm.call_extern(
                                        "float16", "vadd",
                                        temp_andor_out.access_ptr(
                                            "rw",
                                            offset=repeat_dst_offset),
                                        temp_a0.access_ptr(
                                            "r",
                                            offset=repeat_src_offset),
                                        temp_a1.access_ptr(
                                            "r",
                                            offset=repeat_src_offset),
                                        tmp_repeat_times, 1, 1, 1, 8, 8, 8))
                            elif extern_args[0] == 'not':
                                ir_builder.emit(
                                    tvm.call_extern(
                                        "float16", "vsub",
                                        temp_andor_out.access_ptr(
                                            "rw",
                                            offset=repeat_dst_offset),
                                        temp_a0.access_ptr(
                                            "r",
                                            offset=repeat_src_offset),
                                        bias_buf.access_ptr("r"),
                                        tmp_repeat_times, 1, 1, 1, 8, 8, 8))

                            with ir_builder.for_range(
                                    0, tmp_repeat_times,
                                    name="cmp_index") as cmp_index:
                                tmp_src_repeat_offset = cal_once_len*cmp_index
                                tmp_dst_repeat_offset = (cal_once_len//reduce_factor)*cmp_index
                                ir_builder.emit(tvm.call_extern(
                                    "float16", "vcmp_eq",
                                    temp_andor_out.access_ptr(
                                        "r",
                                        offset=tmp_src_repeat_offset),
                                    thred_buf.access_ptr("r"),
                                    1, 1, 1, 1, 8, 8, 8))

                                ir_builder.emit(tvm.call_extern(
                                    "float16", "vsel",
                                    temp_out.access_ptr(
                                        "rw",
                                        offset=tmp_dst_repeat_offset),
                                    temp_andor_out.access_ptr(
                                        "r",
                                        offset=tmp_src_repeat_offset),
                                    bias_buf.access_ptr("r"),
                                    1, 1, 1, 1, 8, 8, 8))

                        ir_builder.emit(tvm.call_extern(
                            "float16", "vconv_f162s8",
                            dst_buffers[0].access_ptr(
                                "rw", offset=repeat_dst_offset),
                            temp_out.access_ptr("r", offset=repeat_dst_offset),
                            tmp_repeat_times, 1, 1, 4, 8))

                    local_repeat_times -= MAX_CAL_TIMES
                    repeat_src_offset += cal_once_len*tmp_repeat_times
                    repeat_dst_offset += cal_once_len*tmp_repeat_times
            else:
                split_op(ir_builder, repeat_times, cal_once_len, cal_once_len,
                         0, insn_concat_args)

    if remain_len > 0:
        reset_mask_insn(ir_builder, dst_dtype, bits=remain_len)
        repeat_src_offset = repeat_times*cal_once_len*args[1]
        repeat_dst_offset = repeat_times*cal_once_len*args[0]
        if op_cmd.find("vcmp_") != -1:
            ir_builder.emit(tvm.call_extern(dst_dtype,
                                            op_cmd,
                                            src_buffers[0].access_ptr(
                                                "r",
                                                offset=repeat_src_offset),
                                            scalar_buf.access_ptr("r"),
                                            1, 1, 1, 1, 8, 8, 8))
            ir_builder.emit(tvm.call_extern(dst_dtype,
                                            "vsel",
                                            dst_buffers[0].access_ptr(
                                                "rw",
                                                offset=repeat_dst_offset),
                                            src_buffers[1].access_ptr(
                                                "r",
                                                offset=repeat_src_offset),
                                            bias_buf.access_ptr("r"),
                                            1, 1, 1, 1, 8, 8, 8))
        elif op_cmd == 'vcond':
            ir_builder.emit(tvm.call_extern(
                dst_dtype, "vcmp_"+extern_args[0],
                src_buffers[0].access_ptr("r", offset=repeat_src_offset),
                thred_buf.access_ptr("r"), 1, 1, 1, 1, 8, 8, 8))
            ir_builder.emit(tvm.call_extern(
                dst_dtype, "vsel",
                dst_buffers[0].access_ptr("rw", offset=repeat_dst_offset),
                src_buffers[0].access_ptr("r", offset=repeat_src_offset),
                bias_buf.access_ptr("r"),
                1, 1, 1, 1, 8, 8, 8))
        elif op_cmd == 'vcmpsel':
            ir_builder.emit(tvm.call_extern(
                dst_dtype, "vcmp_"+extern_args[0],
                src_buffers[0].access_ptr("r", offset=repeat_src_offset),
                src_buffers[1].access_ptr("r", offset=repeat_src_offset),
                1, 1, 1, 1, 8, 8, 8))
            ir_builder.emit(tvm.call_extern(
                dst_dtype, "vsel",
                dst_buffers[0].access_ptr("rw", offset=repeat_dst_offset),
                src_buffers[0].access_ptr("r", offset=repeat_src_offset),
                src_buffers[1].access_ptr("r", offset=repeat_src_offset),
                1, 1, 1, 1, 8, 8, 8))
        elif op_cmd == 'vlogic':
            ir_builder.emit(tvm.call_extern(
                "float16", "vconv_s82f16",
                temp_a0.access_ptr("rw", offset=repeat_src_offset),
                src_buffers[0].access_ptr("r", offset=repeat_src_offset),
                1, 1, 1, 8, 4))
            if extern_args[0] != 'not':
                ir_builder.emit(tvm.call_extern(
                    "float16", "vconv_s82f16",
                    temp_a1.access_ptr("rw", offset=repeat_src_offset),
                    src_buffers[1].access_ptr("r", offset=repeat_src_offset),
                    1, 1, 1, 8, 4))

            if extern_args[0] == 'and':
                ir_builder.emit(tvm.call_extern("float16", "vmul",
                                                temp_out.access_ptr(
                                                    "rw",
                                                    offset=repeat_dst_offset),
                                                temp_a0.access_ptr(
                                                    "r",
                                                    offset=repeat_src_offset),
                                                temp_a1.access_ptr(
                                                    "r",
                                                    offset=repeat_src_offset),
                                                1, 1, 1, 1, 8, 8, 8))
            elif extern_args[0] == 'or' or extern_args[0] == 'not':
                if extern_args[0] == 'or':
                    ir_builder.emit(tvm.call_extern(
                        "float16", "vadd",
                        temp_andor_out.access_ptr(
                            "rw",
                            offset=repeat_dst_offset),
                        temp_a0.access_ptr(
                            "r",
                            offset=repeat_src_offset),
                        temp_a1.access_ptr(
                            "r",
                            offset=repeat_src_offset),
                        1, 1, 1, 1, 8, 8, 8))
                elif extern_args[0] == 'not':
                    ir_builder.emit(tvm.call_extern(
                        "float16", "vsub",
                        temp_andor_out.access_ptr(
                            "rw",
                            offset=repeat_dst_offset),
                        temp_a0.access_ptr(
                            "r",
                            offset=repeat_src_offset),
                        bias_buf.access_ptr("r"),
                        1, 1, 1, 1, 8, 8, 8))
                ir_builder.emit(tvm.call_extern(
                    "float16", "vcmp_eq",
                    temp_andor_out.access_ptr("r", offset=repeat_src_offset),
                    thred_buf.access_ptr("r"),
                    1, 1, 1, 1, 8, 8, 8))

                ir_builder.emit(tvm.call_extern(
                    "float16", "vsel",
                    temp_out.access_ptr("rw", offset=repeat_src_offset),
                    temp_andor_out.access_ptr("r", offset=repeat_src_offset),
                    bias_buf.access_ptr("r"),
                    1, 1, 1, 1, 8, 8, 8))

            ir_builder.emit(tvm.call_extern(
                "float16", "vconv_f162s8",
                dst_buffers[0].access_ptr("rw", offset=repeat_dst_offset),
                temp_out.access_ptr("r", offset=repeat_dst_offset),
                1, 1, 1, 4, 8))
        else:
            tmp_args = cce_util.concat_args(src_buffers, dst_buffers,
                                            repeat_src_offset,
                                            repeat_dst_offset, 1, extern_args,
                                            args)
            ir_builder.emit(tvm.call_extern(dst_dtype, op_cmd, *tmp_args))
        if reset_mask is not None:
            reset_mask_insn(ir_builder, dst_dtype)

# pylint: disable=too-many-locals
def vec_cmd_factory(ir_builder, op_cmd, src_buffers, dst_buffers, op_length,
                    reset_mask, extern_args, args, iter_var=None, pad=None,
                    repeat_cal_dtype=None):
    """
    factory function for generate commond , only support elewise, broadcast,
    vcg, do not support VCOP

    ib: instance of ir_builder

    op_cmd : string
        commond type

    src_buffers : list
        contains source buffers

    dst_buffers : list
        contains dst buffers

    op_length : int
        data length

    reset_mask : int or bool
        if reset_mask == True:
            means want to add reset mask to 128 at the end of commond if mask is changed,
            and pipeline id follows pipeline_count_list changes
        if reset_mask == False:
            means NOT want to add reset mask to 128 at the end of commond
        if reset_mask is int:
            means want to add reset mask to 128 at the end of commond if mask is changed,
            and pipeline id is reset_mask*8

    extern_args : list
        external args in VS or broadcast commond

    args : list
        commond args like the strides in vector commond
    """
    block_len = 256

    if not src_buffers:
        src_dtype = dst_buffers[0].dtype
    else:
        src_dtype = src_buffers[0].dtype

    if repeat_cal_dtype is None:
        repeat_cal_dtype = src_dtype

    cal_bit_len = cce_util.get_bits_of(repeat_cal_dtype)

    # logic use vmul vadd vsub vcmp vsel, only support float16, so
    # 1. int8 -> float16 2.  mul vadd vsub vcmp vsel 3. float16 ->int8
    # bit len use float16 length
    if op_cmd == 'vlogic':
        cal_bit_len = 16
    cal_once_len = block_len*8//cal_bit_len

    # in reduce last axis case , we add pad for speel_num so that speel_num = iter_var*op_length
    # if iter_var != None, and pad is not zero, we should sub the pad when
    # caculating in the last forloop
    if iter_var is not None and pad is not None and int(pad.value) != 0:
        with ir_builder.if_scope(iter_var < iter_var.dom.extent - 1):
            local_total_len = op_length
            vec_repeat_elewise(ir_builder, op_cmd, src_buffers, dst_buffers,
                               local_total_len, cal_once_len,
                               reset_mask, extern_args, args)
        with ir_builder.else_scope():
            local_total_len = op_length - int(pad.value)
            vec_repeat_elewise(ir_builder, op_cmd, src_buffers, dst_buffers,
                               local_total_len, cal_once_len,
                               reset_mask, extern_args, args)
    else:
        local_total_len = op_length
        vec_repeat_elewise(ir_builder, op_cmd, src_buffers, dst_buffers,
                           local_total_len, cal_once_len,
                           reset_mask,
                           extern_args, args)

# pylint: disable=unused-argument, too-many-locals
def bias_accmulate_on_ub_factory(ir_builder, op_cmd, src_buffers, dst_buffers,
                                 shape, vector_tile_size, reset_mask, args):
    """
    factory function for generate commond , only support vector add used to accmulate the bias on UB

    ib: instance of ir_builder

    op_cmd : string
        commond type

    src_buffers : list
        contains source buffers

    dst_buffers : list
        contains dst buffers

    shape : list
        contains shape information of both inputs and output

    reset_mask : int or bool
        if reset_mask == True:
            means want to add reset mask to 128 at the end of commond if mask is changed,
            and pipeline id follows pipeline_count_list changes
        if reset_mask == False:
            means NOT want to add reset mask to 128 at the end of commond
        if reset_mask is int:
            means want to add reset mask to 128 at the end of commond if mask is changed,
            and pipeline id is reset_mask*8

    args : list
        commond args like the strides in vector commond
    """
    # unused func args, for pylint
    if src_buffers[0].dtype != src_buffers[1].dtype:
        raise RuntimeError("src0 dtype[%s] not same with src1 dtype[%s]."
                           % (src_buffers[0].dtype, src_buffers[1].dtype))

    src_dtype = src_buffers[0].dtype
    dst_dtype = dst_buffers[0].dtype

    element_bitwidth = cce_util.get_bits_of(src_dtype)
    # Currently, we only support the broadcast of fp16
    if element_bitwidth != 16:
        raise RuntimeError("element bitwidth for fp16 should be 16, curr is [%d]."
                           % element_bitwidth)
    if cce.VECTOR_INST_BLOCK_WIDTH % element_bitwidth != 0:
        raise RuntimeError("vector block width[%d] should be multi 16 times."
                           % cce.VECTOR_INST_BLOCK_WIDTH)

    element_num_in_one_block = cce.VECTOR_INST_BLOCK_WIDTH//element_bitwidth
    element_num_in_one_repeat = element_num_in_one_block*cce.VECTOR_INST_BLOCK_NUM

    repeat_who = (shape[1] < shape[2])

    src_buffers = [src_buffers[0], src_buffers[1]] \
        if repeat_who else [src_buffers[1], src_buffers[0]]
    element_num_in_op0 = shape[1] if repeat_who else shape[2]
    element_num_in_op1 = shape[2] if repeat_who else shape[1]

    if element_num_in_op1 % element_num_in_op0 != 0:
        raise RuntimeError("element_num_in_op1[%d] shoud be multi element_num_in_op0[%d]."
                           % (element_num_in_op1, element_num_in_op0))

    broadcast_number = element_num_in_op1//element_num_in_op0
    vector_ins_number = element_num_in_op0//element_num_in_one_block
    reset_mask_insn(ir_builder, dst_dtype, bits=128)

    def emit_brc_vector(local_broadcast_number, offset_index):
        '''
        emit_brc_vector
        '''
        # pylint: disable=too-many-locals
        src0_stride_m0 = 0  # data element stride of 1 means repeat-used data
        src1_stride_m0 = 1  # data element stride of 1 means continuous data
        dst_stride_m0 = 1  # data element stride of 1 means continuous data

        src0_stride_m1 = 0  # block stride of 0 means repeat-used data
        src1_stride_m1 = 8  # block stride of 8 means continuous data
        dst_stride_m1 = 8  # block stride of 8 means continuous data

        base_src0_roffset = 0
        base_src1_roffset = offset_index*element_num_in_one_repeat
        base_dst_roffset = offset_index*element_num_in_one_repeat

        args = [dst_stride_m0, src0_stride_m0, src1_stride_m0,
                dst_stride_m1, src0_stride_m1, src1_stride_m1]

        def insn_concat_args(src0_roffset, src1_roffset, dst_roffset,
                             tmp_repeat_times):
            '''
            insn_concat_args
            '''
            tmp_args = []
            for i in dst_buffers:
                tmp_args.append(i.access_ptr("wr", offset=dst_roffset))

            tmp_args.append(src_buffers[0].access_ptr("r", offset=src0_roffset))
            tmp_args.append(src_buffers[1].access_ptr("r", offset=src1_roffset))
            tmp_args.append(tmp_repeat_times)
            tmp_args += args

            return tvm.call_extern(dst_dtype, op_cmd, *tmp_args)

        if vector_ins_number > 0:
            with ir_builder.for_range(0, vector_ins_number,
                                      name="i_inner") as i_inner:
                src0_roffset = base_src0_roffset+i_inner*element_num_in_one_block

                aligned_broadcast_number = broadcast_number
                if broadcast_number%16 != 0:
                    aligned_broadcast_number = (broadcast_number//16+1)*16
                src1_roffset = base_src1_roffset+i_inner*\
                               element_num_in_one_block*aligned_broadcast_number
                dst_roffset = base_dst_roffset+i_inner*\
                              element_num_in_one_block*broadcast_number

                ir_builder.emit(insn_concat_args(src0_roffset, src1_roffset,
                                                 dst_roffset,
                                                 local_broadcast_number))

    outer_for_range = broadcast_number//(cce.VECTOR_INST_BLOCK_NUM*\
                                           cce.VECTOR_INST_MAX_REPEAT_TIMES)
    with ir_builder.for_range(0, outer_for_range, name="i_outer") as i_outer:
        emit_brc_vector(cce.VECTOR_INST_MAX_REPEAT_TIMES, i_outer*cce.VECTOR_INST_MAX_REPEAT_TIMES)

    remain_broadcast_number = broadcast_number%\
                              (cce.VECTOR_INST_BLOCK_NUM*cce.VECTOR_INST_MAX_REPEAT_TIMES)
    if remain_broadcast_number != 0:
        # ratio part
        ratio = remain_broadcast_number//cce.VECTOR_INST_BLOCK_NUM
        emit_brc_vector(ratio, outer_for_range*cce.VECTOR_INST_MAX_REPEAT_TIMES)
        # remain part
        remain = remain_broadcast_number%cce.VECTOR_INST_BLOCK_NUM
        if remain != 0:
            reset_mask_insn(ir_builder, dst_dtype, bits=(remain*element_num_in_one_block))
            emit_brc_vector(1, outer_for_range*cce.VECTOR_INST_MAX_REPEAT_TIMES+ratio)

    if reset_mask is not None:
        reset_mask_insn(ir_builder, dst_dtype, bits=128)

# pylint: disable=too-many-locals
def vec_repeat_reduce_bisec(ir_builder, vcg_cmd, v_cmd, vc_cmd, operator_a,
                            operator_b, total_len, cross_element, dtype):
    # pylint: disable=unused-argument
    '''
    insn_concat_args
    '''
    src0_stride_m0 = 1
    src1_stride_m0 = 1
    dst_stride_m0 = 1
    src0_stride_m1 = 16
    src1_stride_m1 = 16
    dst_stride_m1 = 8

    emit_cmd = v_cmd

    def new_alloc(tvm_ib, dtype, shape, name, scope):
        buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)
        return new_buffer

    local_total_len = total_len
    max_temp_buffer = new_alloc(ir_builder, operator_b.dtype, (total_len // 2,),
                                'max_temp_buffer', scope=cce.scope_ubuf)

    operator_src = operator_a
    operator_dst = max_temp_buffer

    while local_total_len > cross_element:
        repeat_times = local_total_len //2// cross_element
        remain_len = local_total_len-repeat_times*cross_element*2

        reset_mask_insn(ir_builder, operator_b.dtype, bits=cross_element)
        if repeat_times != 0:
            ir_builder.emit(tvm.call_extern(
                operator_b.dtype, emit_cmd,
                operator_dst.access_ptr("rw", offset=0),
                operator_src.access_ptr("r", offset=0),
                operator_src.access_ptr("r", offset=cross_element),
                repeat_times,
                dst_stride_m0,
                src0_stride_m0,
                src1_stride_m0,
                dst_stride_m1,
                src0_stride_m1,
                src1_stride_m1))
            if remain_len != 0:
                remain_left_times = remain_len//cross_element
                remain_left_len = remain_len - remain_left_times*cross_element
                if remain_left_times != 0:
                    ir_builder.emit(tvm.call_extern(
                        operator_b.dtype, emit_cmd,
                        operator_dst.access_ptr("rw", offset=0),
                        operator_dst.access_ptr("r", offset=0),
                        operator_src.access_ptr("r", offset=repeat_times*cross_element*2),
                        1,
                        dst_stride_m0,
                        src0_stride_m0,
                        src1_stride_m0,
                        dst_stride_m1,
                        src0_stride_m1,
                        src1_stride_m1))
                if remain_left_len != 0:
                    reset_mask_insn(ir_builder, operator_b.dtype, bits=remain_left_len)
                    ir_builder.emit(tvm.call_extern(
                        operator_b.dtype, emit_cmd,
                        operator_dst.access_ptr("rw", offset=0),
                        operator_dst.access_ptr("r", offset=0),
                        operator_src.access_ptr("r",
                                                offset=repeat_times*cross_element*2 +
                                                cross_element*remain_left_times),
                        1,
                        dst_stride_m0,
                        src0_stride_m0,
                        src1_stride_m0,
                        dst_stride_m1,
                        src0_stride_m1,
                        src1_stride_m1))
        else:
            remain_left_times = remain_len//cross_element
            remain_left_len = remain_len - remain_left_times*cross_element
            reset_mask_insn(ir_builder, operator_b.dtype, bits=remain_left_len)
            if remain_left_times != 0:
                ir_builder.emit(tvm.call_extern(
                    operator_b.dtype, emit_cmd,
                    operator_src.access_ptr("rw", offset=0),
                    operator_src.access_ptr("r", offset=0),
                    operator_src.access_ptr("r", offset=remain_left_times*cross_element),
                    1,
                    dst_stride_m0,
                    src0_stride_m0,
                    src1_stride_m0,
                    dst_stride_m1,
                    src0_stride_m1,
                    src1_stride_m1))
            local_total_len = remain_left_times * cross_element
            break
        ##swap the src addr and dst addr
        operator_dst, operator_src = operator_src, operator_dst

        local_total_len = repeat_times * cross_element
    #here must be 64 element left,
    # we know that ths ub address is 32B aligned,for fp32 the min element num is 8
    if local_total_len%8 == 0:
        while local_total_len > 8:
            remain_len = local_total_len //2
            reset_mask_insn(ir_builder, operator_b.dtype, bits=remain_len)
            ir_builder.emit(tvm.call_extern(
                operator_b.dtype, emit_cmd,
                operator_dst.access_ptr("rw", offset=0),
                operator_src.access_ptr("r", offset=0),
                operator_src.access_ptr("r", offset=remain_len),
                1,
                dst_stride_m0,
                src0_stride_m0,
                src1_stride_m0,
                dst_stride_m1,
                src0_stride_m1,
                src1_stride_m1))

            operator_dst, operator_src = operator_src, operator_dst

            local_total_len = remain_len
    #here use reg_mov
    with ir_builder.for_range(0, local_total_len -1, name="idx") as idx:
        ir_builder.emit(tvm.call_extern(
            operator_b.dtype, "reg_mov",
            operator_dst.access_ptr("rw"),
            operator_src.access_ptr("r",offset=idx+1)))
        reset_mask_insn(ir_builder, operator_b.dtype, bits=1)
        ir_builder.emit(tvm.call_extern(
            operator_b.dtype, emit_cmd,
            operator_src.access_ptr("rw", offset=0),
            operator_dst.access_ptr("r", offset=0),
            operator_src.access_ptr("r", offset=0),
            1,
            dst_stride_m0,
            src0_stride_m0,
            src1_stride_m0,
            dst_stride_m1,
            src0_stride_m1,
            src1_stride_m1))
    ##this for that there will be out loop, like 251*3868,every 3868 we get the max value,but after
    #251 loop ,we should get the the max value in the 251 num,as follows:
    v_cmd_mask = 2
    reset_mask_insn(ir_builder, operator_b.dtype, v_cmd_mask)

    ir_builder.emit(tvm.call_extern(
        operator_b.dtype, "reg_mov",
        operator_dst.access_ptr("rw",),
        operator_b.access_ptr("r")))
    ir_builder.emit(tvm.call_extern(
            operator_b.dtype, emit_cmd,
            operator_src.access_ptr("rw", offset=0),
            operator_dst.access_ptr("r", offset=0),
            operator_src.access_ptr("r", offset=0),
            1,
            dst_stride_m0,
            src0_stride_m0,
            src1_stride_m0,
            dst_stride_m1,
            src0_stride_m1,
            src1_stride_m1))

    ir_builder.emit(tvm.call_extern(
        operator_b.dtype, "reg_mov",
        operator_b.access_ptr("rw"),
        operator_src.access_ptr("r")))
    reset_mask_insn(ir_builder, operator_b.dtype)

# pylint: disable=too-many-locals
def vec_repeat_reduce(ir_builder, vcg_cmd, v_cmd, vc_cmd, operator_a,
                      operator_b, total_len, cross_element, dtype):
    # pylint: disable=unused-argument
    '''
    insn_concat_args
    '''
    # for pylint, unused args
    src0_stride = 1
    src1_stride = 8
    dst_stride = 1

    local_total_len = total_len
    emit_cmd = vcg_cmd
    res_block_size = 8
    if vc_cmd == "vcadd":
        emit_cmd = vc_cmd
        res_block_size = 1

    def insn_reduce(src_roffset, dst_roffset, tmp_repeat_times, idx):
        # pylint: disable=unused-argument
        '''
        insn_reduce
        '''
        # for pylint, reserve unified entry

        return tvm.call_extern(
            operator_b.dtype, emit_cmd,
            operator_a.access_ptr("rw", offset=dst_roffset),
            operator_a.access_ptr("r", offset=src_roffset),
            tmp_repeat_times,
            dst_stride,
            src0_stride,
            src1_stride)

    while local_total_len > cross_element:
        repeat_times = local_total_len//cross_element
        remain_len = local_total_len - repeat_times*cross_element

        if repeat_times > 0:
            split_op(ir_builder, repeat_times, cross_element, res_block_size, 0, insn_reduce)
        if remain_len > 0:
            reset_mask_insn(ir_builder, operator_b.dtype, bits=remain_len)
            ir_builder.emit(tvm.call_extern(
                operator_b.dtype, emit_cmd,
                operator_a.access_ptr("rw", offset=repeat_times*res_block_size),
                operator_a.access_ptr("r", offset=repeat_times*cross_element),
                1,
                dst_stride,
                src0_stride,
                src1_stride))

            if vc_cmd == "vcadd":
                if dtype == "float32":
                    local_total_len_tmp = repeat_times+(remain_len+63)//64
                else:
                    local_total_len_tmp = repeat_times+(remain_len+127)//128
            else:
                local_total_len_tmp = repeat_times*8+(remain_len+15)//16
            if local_total_len_tmp > cross_element:  # last time no need to set mask
                reset_mask_insn(ir_builder, operator_b.dtype)

        if vc_cmd == "vcadd":
            if dtype == "float32":
                local_total_len = repeat_times+(remain_len+63)//64
            else:
                local_total_len = repeat_times+(remain_len+127)//128
        else:
            local_total_len = repeat_times*8+(remain_len+15)//16

    if local_total_len > 1:
        reset_mask_insn(ir_builder, operator_b.dtype, bits=local_total_len)
        ir_builder.emit(tvm.call_extern(
            operator_b.dtype, vc_cmd,
            operator_a.access_ptr("rw"),
            operator_a.access_ptr("r"),
            1,
            dst_stride,
            src0_stride,
            src1_stride))

    v_cmd_mask = 2

    reset_mask_insn(ir_builder, operator_b.dtype, v_cmd_mask)

    ir_builder.emit(tvm.call_extern(
        operator_b.dtype, "reg_mov",
        operator_a.access_ptr("rw", offset=1),
        operator_b.access_ptr("r")))

    ir_builder.emit(tvm.call_extern(
        operator_b.dtype, vc_cmd,
        operator_a.access_ptr("rw"),
        operator_a.access_ptr("r"),
        1,
        dst_stride,
        src0_stride,
        src1_stride))

    ir_builder.emit(tvm.call_extern(
        operator_b.dtype, "reg_mov",
        operator_b.access_ptr("rw"),
        operator_a.access_ptr("r")))

    reset_mask_insn(ir_builder, operator_b.dtype)


def compute_prod_segmentation(ir_builder, total_len, operator_a, operator_c):
    '''
    compute_prod_segmentation
    '''
    dst_stride_m0 = 1
    src0_stride_m0 = 1
    src1_stride_m0 = 1
    dst_stride_m1 = 0
    src0_stride_m1 = 8
    src1_stride_m1 = 0
    repeat_times = total_len//128
    remain_len = total_len - repeat_times*128
    src_offset = 0
    if repeat_times > 0:
        local_repeat_times = repeat_times
        while local_repeat_times > 0:
            if local_repeat_times > 255:
                tmp_repeat_times = 255
            else:
                tmp_repeat_times = local_repeat_times
            with ir_builder.new_scope():
                ir_builder.emit(tvm.call_extern(
                    operator_c.dtype, "vmul",
                    operator_c.access_ptr("rw"),
                    operator_a.access_ptr("r", offset=src_offset),
                    operator_c.access_ptr("r"),
                    tmp_repeat_times,
                    dst_stride_m0, src0_stride_m0, src1_stride_m0,
                    dst_stride_m1, src0_stride_m1, src1_stride_m1))
                src_offset += 128*tmp_repeat_times
                local_repeat_times -= 255
    if remain_len > 0:
        reset_mask_insn(ir_builder, operator_c.dtype, bits=remain_len)

        with ir_builder.new_scope():
            ir_builder.emit(tvm.call_extern(
                operator_c.dtype, "vmul",
                operator_c.access_ptr("rw"),
                operator_a.access_ptr("r", offset=src_offset),
                operator_c.access_ptr("r"),
                1,
                dst_stride_m0, src0_stride_m0, src1_stride_m0,
                dst_stride_m1, src0_stride_m1, src1_stride_m1))

        reset_mask_insn(ir_builder, operator_c.dtype, bits=128)


def compute_last_128_numbers(ir_builder, operator_c, d_buf):
    '''
    compute_last_128_numbers
    '''
    dst_stride_m0 = 1
    src0_stride_m0 = 1
    src1_stride_m0 = 1
    dst_stride_m1 = 8
    src0_stride_m1 = 8
    src1_stride_m1 = 8

    def fold_mul_1(num):
        '''
        fold mul of 128 data.
        The data split two parts. Then, the two half data mul by vmul after setting mask.
        '''
        reset_mask_insn(ir_builder, operator_c.dtype, bits=num)

        with ir_builder.new_scope():
            ir_builder.emit(tvm.call_extern(
                operator_c.dtype, "vmul",
                operator_c.access_ptr("rw"),
                operator_c.access_ptr("r"),
                operator_c.access_ptr("r", offset=num),
                1,
                dst_stride_m0, src0_stride_m0, src1_stride_m0,
                dst_stride_m1, src0_stride_m1, src1_stride_m1))

    # 64*64
    if operator_c.dtype == "float16":
        fold_mul_1(64)
    # 32*32
    fold_mul_1(32)
    # 16*16
    fold_mul_1(16)

    # last 16 numbers
    reg = ir_builder.allocate(operator_c.dtype, (2,), name="reg_buf", scope=cce.scope_reg)

    def fold_mul_2(num):
        '''
        fold mul of last 16 data.
        Because of alignment constraints,the remanent data must split to two parts.
        The front half stay at source address,and the back half move to buffer address.
        Then, the two half data mul by vmul after setting mask.
        '''
        with ir_builder.for_range(0, num, name="irb_i") as irb_i:
            ir_builder.emit(tvm.call_extern(operator_c.dtype, "reg_mov",
                                            tvm.call_extern(reg.dtype, "reg", reg[0]),
                                            operator_c.access_ptr("rw", offset=(irb_i+num))))

            with ir_builder.new_scope():
                ir_builder.emit(tvm.call_extern(
                    reg.dtype, "reg_mov",
                    d_buf.access_ptr("rw", offset=irb_i),
                    tvm.call_extern(reg.dtype, "reg", reg[0])))

        reset_mask_insn(ir_builder, operator_c.dtype, bits=num)
        with ir_builder.new_scope():
            ir_builder.emit(tvm.call_extern(
                operator_c.dtype, "vmul",
                operator_c.access_ptr("rw"),
                operator_c.access_ptr("r"),
                d_buf.access_ptr("r"),
                1,
                dst_stride_m0, src0_stride_m0, src1_stride_m0,
                dst_stride_m1, src0_stride_m1, src1_stride_m1))

    # 8*8
    fold_mul_2(8)
    # 4*4
    fold_mul_2(4)
    # 2*2
    fold_mul_2(2)
    # 1*1
    fold_mul_2(1)

    reset_mask_insn(ir_builder, operator_c.dtype, bits=128)

# pylint: disable=too-many-locals
def vec_repeat_reduce_prod(ir_builder, operator_a, operator_b, total_len, dtype):
    '''
    vec_repeat_reduce_prod
    '''
    dst_stride_m0 = 1  # dst stridem0
    src0_stride_m0 = 1  # src0 stridem0
    src1_stride_m0 = 1  # src1 stridem0
    dst_stride_m1 = 8  # dst stridem1
    src0_stride_m1 = 8  # src0 stridem1
    src1_stride_m1 = 8  # src1 stridem1

    stride_inside = 1
    stride_outside = 8

    buf_var = ir_builder.allocate(dtype, (128,), "c_buf", scope="local.UB")
    c_buf = tvm.decl_buffer((128,), dtype, "c_buf", scope="local.UB", data=buf_var)
    d_buf_var = ir_builder.allocate(dtype, (128,), "d_buf", scope="local.UB")
    d_buf = tvm.decl_buffer((128,), dtype, "d_buf", scope="local.UB", data=d_buf_var)

    ir_builder.emit(tvm.call_extern(
        c_buf.dtype, "vector_dup",
        c_buf.access_ptr("rw", offset=0),
        tvm.const(1.0, dtype=dtype),
        1,
        stride_inside, stride_inside,
        stride_outside, stride_outside))

    # Compute the product segment
    compute_prod_segmentation(ir_builder, total_len, operator_a, c_buf)
    # Compute the product of the last 128 numbers
    compute_last_128_numbers(ir_builder, c_buf, d_buf)

    # reset_mask_insn(ib, B.dtype, bits=16)

    ir_builder.emit(tvm.call_extern(
        d_buf.dtype, "reg_mov",
        d_buf.access_ptr("rw", offset=0),
        operator_b.access_ptr("r")))

    reset_mask_insn(ir_builder, operator_b.dtype, bits=1)
    with ir_builder.new_scope():
        ir_builder.emit(tvm.call_extern(
            d_buf.dtype, "vmul",
            d_buf.access_ptr("rw"),
            d_buf.access_ptr("r"),
            c_buf.access_ptr("r"),
            1,
            dst_stride_m0, src0_stride_m0, src1_stride_m0,
            dst_stride_m1, src0_stride_m1, src1_stride_m1))

    ir_builder.emit(tvm.call_extern(
        operator_b.dtype, "reg_mov",
        operator_b.access_ptr("rw"),
        d_buf.access_ptr("r")))

    reset_mask_insn(ir_builder, operator_b.dtype, bits=128)

# pylint: disable=too-many-locals
def reduce_last_axis(ir_builder, op_cmd, src_buffers, dst_buffers, data_shape, dtype, iter_var=None,
                     pad=None):
    """
    factory funtion for last axis reduction operations. For tensorize

    Parameters
    ----------
    ir_builder : ir builder
        ir builder
    op_cmd : string
        operation type, supports reduce_sum, reduce_min, reduce_max, reduce_prod

    src_buffers : buffer
        input buffers

    dst_buffers : buffer
        output buffers

    data_shape : tuple or list
        The arrays to concatenate

    dtype : string
        The source data type

    Returns
    -------
    ret : None

    Example :
    -------
        tensorize code like :
            for (k1, 0, 402) {
                A0.local.UB[1024] = (A0.local.UB[1024]+A0.local.UB[(k1+512)])
            }
    """
    op_dict = {"vsum": (tvm.sum, "add"),
               "vmin": (tvm.min, "min"),
               "vmax": (tvm.max, "max"),
               "vmul": (tvm.prod, "mul")}

    op_attr = op_dict.get(op_cmd)
    if not op_attr:
        raise RuntimeError("op %s not support yet"%op_cmd)

    _, cmd = op_attr

    if cmd in ("add", "max", "min", "mul"):
        if dtype not in ('float16', 'float32'):
            raise ValueError(
                "reduce_last_axis only support float16 and float32 while dtype is %s"%dtype)
    else:
        if dtype != "float16":
            raise ValueError("reduce_last_axis only support float16 while dtype is %s"%dtype)

    vcg_cmd = "vcg"+cmd
    v_cmd = "v"+cmd
    vc_cmd = "vc"+cmd

    buffer_a = src_buffers[0]  # A_buf
    buffer_b = dst_buffers[0]  # B_buf

    cross_element = 128
    if cmd in ("add", "max", "min") and dtype == "float32":
        cross_element = 64


    # in reduce last axis case , we add pad for speel_num so that
    # speel_num = iter_var*op_length
    # if iter_var != None, and pad is not zero,
    # we should sub the pad when caculating in the last forloop
    if iter_var is not None and pad is not None and int(pad.value) != 0:
        with ir_builder.if_scope(iter_var < iter_var.dom.extent - 1):
            total_len = data_shape[-1]
            if cmd == "mul":
                vec_repeat_reduce_prod(ir_builder, buffer_a, buffer_b, total_len, dtype)
            elif dtype == "float32" and cmd in ("max", "min"):
                vec_repeat_reduce_bisec(ir_builder, vcg_cmd, v_cmd, vc_cmd, buffer_a,
                                        buffer_b, total_len, cross_element, dtype)
            else:
                vec_repeat_reduce(ir_builder, vcg_cmd, v_cmd, vc_cmd, buffer_a,
                                  buffer_b, total_len, cross_element, dtype)
        with ir_builder.else_scope():
            total_len = data_shape[-1] - int(pad.value)
            if cmd == "mul":
                vec_repeat_reduce_prod(ir_builder, buffer_a, buffer_b, total_len, dtype)
            elif dtype == "float32" and cmd in ("max", "min"):
                vec_repeat_reduce_bisec(ir_builder, vcg_cmd, v_cmd, vc_cmd, buffer_a,
                                        buffer_b, total_len, cross_element, dtype)
            else:
                vec_repeat_reduce(ir_builder, vcg_cmd, v_cmd, vc_cmd, buffer_a,
                                  buffer_b, total_len, cross_element, dtype)
    else:
        total_len = data_shape[-1]
        if cmd == "mul":
            vec_repeat_reduce_prod(ir_builder, buffer_a, buffer_b, total_len, dtype)
        elif dtype == "float32" and cmd in ("max", "min"):
            vec_repeat_reduce_bisec(ir_builder, vcg_cmd, v_cmd, vc_cmd, buffer_a,
                                    buffer_b, total_len, cross_element, dtype)
        else:
            vec_repeat_reduce(ir_builder, vcg_cmd, v_cmd, vc_cmd, buffer_a,
                              buffer_b, total_len, cross_element, dtype)

# pylint: disable=too-many-locals
# intrin for psroialign begin
@tvm.register_func("tvm.intrin.cce.dma_copy_res_for_batch_core")
def dma_copy_res_for_batch_core(tensor_op):
    """
    psroialign dma copy res for multi core
    """
    ins, outs, _, _ = cce_util.get_dma_buffer(tensor_op)
    src = ins[0]
    dst = outs[0]
    roi_num_per_batch = get_emitinsn_params("roi_num_per_batch")
    cur_batch = get_emitinsn_params("thread_block")
    c0_times = get_emitinsn_params("c0_times")

    dst_offset = c0_times*cur_batch*roi_num_per_batch*cce.C0_SIZE*\
                 get_emitinsn_params("loop_num_out")

    ir_builder = tvm.ir_builder.create()
    elem_width = cce_util.get_type_bits(dst.dtype)
    elem_bytes = cce.GLB_ELEM_BYTES
    _, _, nburst, burst, src_stride, dst_stride = cce_util.get_mov_pattern(
        src, elem_width, elem_bytes, dst, allow_fold=True)
    cce_util.dma_dependency_scope(src, dst, ir_builder)

    sid = 0

    ir_builder.emit(tvm.call_extern(
        dst.dtype, "copy_ubuf_to_gm",
        dst.access_ptr("w", offset=dst_offset),  # dst buffer
        src.access_ptr("r"),  # src buffer
        sid,
        nburst,
        burst,
        src_stride,
        dst_stride
    ))

    return ir_builder.get()

@tvm.register_func("tvm.intrin.cce.dma_copy_for_non_32_align")
def dma_copy_for_non_32_align(tensor_op):
    """
    psroialign dma copy res for multi core
    """
    ins, outs, _, _ = cce_util.get_dma_buffer(tensor_op, is_storage_align=True)
    src = ins[0]
    dst = outs[0]

    shape = []

    def _get_shape(stmt_op):
        if isinstance(stmt_op, tvm.stmt.For):
            shape.append(stmt_op.extent)
            if isinstance(stmt_op.body, tvm.stmt.For):
                _get_shape(stmt_op.body)

    while tensor_op is not None and isinstance(tensor_op, tvm.stmt.AttrStmt):
        tensor_op = tensor_op.body

    _get_shape(tensor_op)

    if not shape:
        shape.append(tvm.const(1))

    def _shape_mul(shape):
        if not shape:
            return 1
        return reduce(lambda x, y: x*y, shape)

    reg_idx_out_len = int(_shape_mul(shape[:len(shape)-1]))
    reg_idx_in_len = int(shape[-1])

    def get_align_factor(dtype):
        if dtype in ('int8', 'uint8'):
            align_factor = 32
        elif dtype in ('float16', 'int16', 'uint16'):
            align_factor = 16
        else:
            align_factor = 8
        return align_factor

    align_factor = get_align_factor(dst.dtype)

    ir_builder = tvm.ir_builder.create()
    repeat = (reg_idx_out_len * reg_idx_in_len - align_factor) // reg_idx_in_len + 1
    if reg_idx_out_len * reg_idx_in_len - align_factor <= 0:
        repeat = 0
    if repeat != 0:
        with ir_builder.for_range(0, repeat, name="repeat_index") as repeat_index:
            ir_builder.emit(tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr("w", offset=repeat_index*reg_idx_in_len),
                src.access_ptr("r", offset=repeat_index*align_factor),
                0,
                1,
                1,
                0,
                0
            ))

    def get_repeat_temp(repeat):
        if repeat > 0:
            repeat_temp = repeat - 1
        else:
            repeat_temp = 0
        return repeat_temp

    buf_var = ir_builder.allocate(src.dtype, (align_factor,), "c_buf", scope="local.UB")
    c_buf = tvm.decl_buffer((align_factor,), src.dtype, "c_buf", scope="local.UB", data=buf_var)
    reg = ir_builder.allocate(outs[0].dtype, (1,), name="reg", scope=cce.scope_reg)
    remain_cout_remain = align_factor % reg_idx_in_len
    remain_cout_cout = align_factor // reg_idx_in_len
    if remain_cout_remain != 0:
        with ir_builder.for_range(0, remain_cout_remain, name="remain_cout_remain_index") as \
                remain_cout_remain_index:
            repeat_temp = get_repeat_temp(repeat)
            offset = repeat_temp * align_factor + (reg_idx_in_len - remain_cout_remain) + \
                     remain_cout_remain_index
            ir_builder.emit(tvm.call_extern(
                dst.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                src.access_ptr("r", offset=offset)))
            ir_builder.emit(tvm.call_extern(
                dst.dtype, "reg_mov",
                c_buf.access_ptr("rw", offset=remain_cout_remain_index),
                tvm.call_extern(reg.dtype, "reg", reg[0])))
    if remain_cout_cout != 0:
        start_index = 0
        if repeat > 0:
            if remain_cout_remain == 0:
                start_index = repeat - 1
            else:
                start_index = repeat
        with ir_builder.for_range(0, remain_cout_cout, name="remain_cout_cout_index") as \
                remain_cout_cout_index:
            with ir_builder.for_range(0, reg_idx_in_len, name="reg_idx_in_index") as \
                    reg_idx_in_index:
                offset_in = (start_index + remain_cout_cout_index) * align_factor + reg_idx_in_index
                ir_builder.emit(tvm.call_extern(
                    dst.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    src.access_ptr("r", offset=offset_in)))
                offset_out = remain_cout_remain + remain_cout_cout_index * reg_idx_in_len + \
                             reg_idx_in_index
                ir_builder.emit(tvm.call_extern(
                    dst.dtype, "reg_mov",
                    c_buf.access_ptr("rw", offset=offset_out),
                    tvm.call_extern(reg.dtype, "reg", reg[0])))
    dst_offset = reg_idx_out_len * reg_idx_in_len - align_factor
    ir_builder.emit(tvm.call_extern(
        dst.dtype, "copy_ubuf_to_gm",
        dst.access_ptr("w", offset=dst_offset),
        c_buf.access_ptr("r"),
        0,
        1,
        1,
        0,
        0
    ))

    return ir_builder.get()

def get_emitinsn_params(name):
    """
    :param name:  the key want to get param from dsl
    :return:  the value need
    """
    return cce_emitinsn_params.cceEmitParamsIns.get_param(name)

# pylint: disable=too-many-locals
@tvm.register_func("tvm.intrin.cce.psalign_four_2_five")
def psalign_four_2_five(tensor_op):
    """
    :param tensor_op: the stmt of for with psroialign
    :return: the intric stmt what we want
    """
    ins, outs, _, _ = cce_util.get_dma_buffer(tensor_op)

    is_four_to_five_big = get_emitinsn_params("is_four_to_five_big")
    c_dim = get_emitinsn_params("c_dim")
    fm_num_per_batch_in = get_emitinsn_params("fm_num_per_batch_in")
    fm_num_per_batch_out = get_emitinsn_params("fm_num_per_batch_out")
    batch_size = get_emitinsn_params("batch_size")
    loop_num_out = get_emitinsn_params("loop_num_out")
    inner_loop_times = get_emitinsn_params("inner_loop_times")
    loop_num = get_emitinsn_params("loop_num")

    gm_to_ub_burst_length = get_emitinsn_params("gm_to_ub_burst_length")
    gm_to_ub_burst_length_l = get_emitinsn_params("gm_to_ub_burst_length_l")
    ub_addr_stride = get_emitinsn_params("ub_addr_stride")
    ub_addr_stride_l = get_emitinsn_params("ub_addr_stride_l")
    out_addr_stride = get_emitinsn_params("out_addr_stride")

    src_addr_stride = get_emitinsn_params("src_addr_stride")
    src_addr_stride_l = get_emitinsn_params("src_addr_stride_l")
    c0_addr_offset = get_emitinsn_params("c0_addr_offset")

    ub_to_out_burst_length = get_emitinsn_params("ub_to_out_burst_length")
    ub_to_out_burst_length_l = get_emitinsn_params("ub_to_out_burst_length_l")
    dest_addr_stride = get_emitinsn_params("dest_addr_stride")
    dest_addr_stride_l = get_emitinsn_params("dest_addr_stride_l")

    src_offset = get_emitinsn_params("src_offset")
    dest_offset = get_emitinsn_params("dest_offset")

    nchw_repeat = get_emitinsn_params("nchw_repeat")
    nchw_repeat_l = get_emitinsn_params("nchw_repeat_l")

    one_vec_size = get_emitinsn_params("one_vec_size")
    one_vec_size_l = get_emitinsn_params("one_vec_size_l")

    ub_one_buffer_size = get_emitinsn_params("UB_ONE_BUFFER_SIZE")

    cc_block_size = 32
    cc_block_per_repeat = 8
    block_num = 8

    input_c1 = (c_dim+cce.C0_SIZE - 1)//cce.C0_SIZE

    src = ins[0]
    gm_buffer = tvm.decl_buffer([], "float16",
                                name=src.name,
                                data=src.data,
                                offset_factor=src.offset_factor,
                                data_alignment=src.data_alignment,
                                scope=cce_util.get_buf_scope(src.name),
                                elem_offset=0)
    dst = outs[0]  # use for copy_ubuf_to_gm
    dst_buffer = tvm.decl_buffer([], "float16",
                                 name=dst.name,
                                 data=dst.data,
                                 offset_factor=dst.offset_factor,
                                 data_alignment=dst.data_alignment,
                                 scope=cce_util.get_buf_scope(dst.name),
                                 elem_offset=0)

    ir_builder = tvm.ir_builder.create()

    # thread_block is roi in single batch, or batch in multi batch
    thread_block = get_emitinsn_params("thread_block")
    device_core_num = cce_conf.get_soc_spec("CORE_NUM")
    fm_out_thread_offset = (thread_block%device_core_num)*fm_num_per_batch_out
    fm_in_thread_offset = 0
    if batch_size == 1:
        # just use 2 cores for single batch
        ir_builder.scope_attr(thread_block, "thread_extent", 2)
    else:
        fm_in_thread_offset = thread_block*fm_num_per_batch_in
        ir_builder.scope_attr(thread_block, "thread_extent", batch_size)

    # reg[0] src_addr_stride
    # reg[1] destAddrStride
    # reg[2] toUbBurstLength
    # reg[3] ub_addr_stride
    # reg[4] srcStride
    # reg[5] one_vec_size
    # reg[6] NCHWRepeat
    # reg[7] ub_to_out_burst_length
    # reg[8] toOutDestStride

    reg_num = 9 if input_c1 < 2 else 10
    reg = ir_builder.allocate("uint64", (reg_num,), name="reg",
                              scope=cce.scope_reg)

    with ir_builder.for_range(0, loop_num_out, name="n") as loop_var_n:
        reg[0] = tvm.const(0, "uint64")
        reg[1] = tvm.const(0, "uint64")

        with ir_builder.for_range(0, loop_num, name="c") as loop_var_c:
            with ir_builder.if_scope(((loop_var_c+1)%inner_loop_times) == 0):
                reg[2] = tvm.const(gm_to_ub_burst_length_l, "uint64")
                reg[3] = tvm.const(ub_addr_stride_l, "uint64")
                reg[4] = tvm.const(src_addr_stride_l, "uint64")
                reg[5] = tvm.const(one_vec_size_l, "uint64")
                reg[6] = tvm.const(nchw_repeat_l, "uint64")
                reg[7] = tvm.const(ub_to_out_burst_length_l, "uint64")
                reg[8] = tvm.const(dest_addr_stride_l, "uint64")
            with ir_builder.else_scope():
                reg[2] = tvm.const(gm_to_ub_burst_length, "uint64")
                reg[3] = tvm.const(ub_addr_stride, "uint64")
                reg[4] = tvm.const(src_addr_stride, "uint64")
                reg[5] = tvm.const(one_vec_size, "uint64")
                reg[6] = tvm.const(nchw_repeat, "uint64")
                reg[7] = tvm.const(ub_to_out_burst_length, "uint64")
                reg[8] = tvm.const(dest_addr_stride, "uint64")

            dst = apply_for_new_alloc(ir_builder, "float16", (32,), "local.UB")
            in_buf = gm_buffer
            out_buf = dst

            in_buf_offset = fm_in_thread_offset+loop_var_n*src_offset
            if is_four_to_five_big:
                in_buf_offset = in_buf_offset+reg[0]
            else:
                in_buf_offset = in_buf_offset+loop_var_c*c0_addr_offset

            if input_c1 < 2:
                # c_dim <= 16,copy time is c_dim
                c0_times = c_dim
                with ir_builder.for_range(0, c0_times, name="c0") as loop_var_c0:
                    intrin_name = "copy_gm_to_ubuf"
                    ir_builder.emit(tvm.call_extern(
                        out_buf.dtype, intrin_name,
                        out_buf.access_ptr("rw", offset=loop_var_c0*reg[3]),
                        in_buf.access_ptr(
                            "rw",
                            offset=in_buf_offset+loop_var_c0*out_addr_stride),
                        0, 1, reg[2], 0, 0))
            else:
                # c_dim > 16, previous copy time is 16,
                # last copy time is c_dim%16
                with ir_builder.if_scope(
                        loop_var_c < (input_c1 - 1)*inner_loop_times):
                    reg[9] = tvm.const(cce.C0_SIZE, "uint64")
                with ir_builder.else_scope():
                    reg[9] = tvm.const(c_dim - (input_c1 - 1)*cce.C0_SIZE,
                                       "uint64")
                with ir_builder.for_range(0, reg[9], name="c0") as loop_var_c0:
                    intrin_name = "copy_gm_to_ubuf"
                    ir_builder.emit(tvm.call_extern(
                        out_buf.dtype, intrin_name,
                        out_buf.access_ptr("rw", offset=loop_var_c0*reg[3]),
                        in_buf.access_ptr(
                            "rw",
                            offset=in_buf_offset+loop_var_c0*out_addr_stride),
                        0, 1, reg[2], 0, 0))

            if is_four_to_five_big:
                reg[0] = reg[0]+reg[4]
                if loop_num > 1:
                    with ir_builder.if_scope(
                            ((loop_var_c+1)%inner_loop_times) == 0):
                        reg[0] = reg[0]+tvm.const(c0_addr_offset, "uint64")

            # use scatter_vnchwconv_b16 do 4D->5D
            addr_array = ir_builder.allocate("uint64", (32,),
                                             name="addr_array2",
                                             scope=cce.scope_reg)
            addr_array_buf = tvm.decl_buffer((32,), dtype="uint64_t",
                                             name="addr_array_buf",
                                             scope=cce.scope_reg,
                                             data=addr_array)
            src0_offset = 8*0
            src1_offset = 8*1
            dst0_offset = 8*2
            dst1_offset = 8*3
            with ir_builder.for_range(0, 8, name="i") as i:
                ir_builder.emit(tvm.call_extern("uint64", "reg_mov",
                                                tvm.call_extern(
                                                    addr_array.dtype,
                                                    "reg",
                                                    addr_array[
                                                        src0_offset+i]),
                                                i*reg[5]
                                                ))
                ir_builder.emit(tvm.call_extern("uint64", "reg_mov",
                                                tvm.call_extern(
                                                    addr_array.dtype,
                                                    "reg", addr_array[
                                                        src1_offset+i]),
                                                (i+8)*reg[5]
                                                ))
                ir_builder.emit(tvm.call_extern("uint64", "reg_mov",
                                                tvm.call_extern(
                                                    addr_array.dtype,
                                                    "reg", addr_array[
                                                        dst0_offset+i]),
                                                ub_one_buffer_size +
                                                i*cc_block_size
                                                ))
                ir_builder.emit(tvm.call_extern("uint64", "reg_mov",
                                                tvm.call_extern(
                                                    addr_array.dtype,
                                                    "reg", addr_array[
                                                        dst1_offset+i]),
                                                ub_one_buffer_size +
                                                (i+cc_block_per_repeat) *
                                                cc_block_size
                                                ))
            ir_builder.emit(tvm.call_extern("int32",
                                            "set_va_reg_sb",
                                            "VA0",
                                            addr_array_buf.access_ptr(
                                                "rw",
                                                offset=src0_offset)
                                            ))
            ir_builder.emit(tvm.call_extern("int32",
                                            "set_va_reg_sb",
                                            "VA1",
                                            addr_array_buf.access_ptr(
                                                "rw",
                                                offset=src1_offset)
                                            ))
            ir_builder.emit(tvm.call_extern("int32",
                                            "set_va_reg_sb",
                                            "VA2",
                                            addr_array_buf.access_ptr(
                                                "rw",
                                                offset=dst0_offset)
                                            ))
            ir_builder.emit(tvm.call_extern("int32",
                                            "set_va_reg_sb",
                                            "VA3",
                                            addr_array_buf.access_ptr(
                                                "rw",
                                                offset=dst1_offset)
                                            ))
            ir_builder.emit(tvm.call_extern("int32",
                                            "scatter_vnchwconv_b16",
                                            "VA2",
                                            "VA0",
                                            reg[6],
                                            2*block_num,
                                            1))

            # copy 5D res to gm
            out_buf = dst_buffer
            in_buf = dst
            intrin_name = "copy_ubuf_to_gm"
            ir_builder.emit(tvm.call_extern(
                out_buf.dtype, intrin_name,
                out_buf.access_ptr("rw",
                                   offset=((fm_out_thread_offset +
                                            loop_var_n*dest_offset+reg[1]))),
                in_buf.access_ptr("rw", offset=ub_one_buffer_size//2),
                0, 1, reg[7], 0, 0))

            reg[1] = reg[1]+reg[8]
    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.psalign_construct_seq")
def psalign_construct_seq(tensor_op):
    """
    :param tensor_op: the stmt of for with psroialign
    :return: the intric stmt what we want
    """
    _, outs, _, _ = cce_util.get_dma_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()
    roi_delta_buf = outs[0]

    reg_int32 = ir_builder.allocate("int32", (1,), name="reg",
                                    scope=cce.scope_reg)
    with ir_builder.for_range(0, 128, name="idx") as idx:
        with ir_builder.new_scope():
            reg_int32[0] = idx
        ir_builder.emit(tvm.call_extern(
            roi_delta_buf.dtype, "reg_mov",
            roi_delta_buf.access_ptr("rw", offset=idx),
            tvm.call_extern(reg_int32.dtype, "reg", reg_int32[0])
        ))

    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.phony_insn")
def phony_insn(tensor_op):
    # pylint: disable=unused-argument
    """
    :param tensor_op: the stmt of for with psroialign
    :return: the intric stmt what we want
    """
    ir_builder = tvm.ir_builder.create()
    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.psalign_roi_transform_vtrans")
def psalign_roi_transform_vtrans(tensor_op):
    """
    :param tensor_op: the stmt of for with psroialign
    :return: the intric stmt what we want
    """
    ins, outs, _, _ = cce_util.get_dma_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()
    if isinstance(tensor_op.body.extent, tvm.expr.IntImm):
        roi_num_aligned = tensor_op.body.extent.value

    repeat_times = roi_num_aligned//128

    src = ins[0]
    dst = outs[0]
    src_buf = tvm.decl_buffer(src.shape, "uint16_t",
                              name=src.name,
                              data=src.data,
                              offset_factor=src.offset_factor,
                              data_alignment=src.data_alignment,
                              scope=cce_util.get_buf_scope(src.name),
                              elem_offset=0)
    dst_buf = tvm.decl_buffer(dst.shape, "uint16_t",
                              name=dst.name,
                              data=dst.data,
                              offset_factor=dst.offset_factor,
                              data_alignment=dst.data_alignment,
                              scope=cce_util.get_buf_scope(dst.name),
                              elem_offset=dst.elem_offset)

    with ir_builder.for_range(0, 8*repeat_times, name="roi_index") as roi_index:
        ir_builder.emit(tvm.call_extern(
            dst_buf.dtype, "vtranspose",
            dst_buf.access_ptr("rw", offset=256*roi_index),  # dst buffer
            src_buf.access_ptr("r", offset=256*roi_index)  # src buffer
        ))

    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.psalign_roi_transform_vadd")
def psalign_roi_transform_vadd(tensor_op):
    """
    :param tensor_op: the stmt of for with psroialign
    :return: the intric stmt what we want
    """
    ins, outs, _, _ = cce_util.get_dma_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()
    roi_vtrans = ins[0]
    zero_ub = ins[1]
    roi_128_index = outs[0]
    if isinstance(tensor_op.extent, tvm.expr.IntImm):
        roi_num_aligned = tensor_op.extent.value

    repeat_times = roi_num_aligned//128

    dst_strid_m1 = 0
    src0_strid_m1 = 0
    if repeat_times > 1:
        dst_strid_m1 = 8
        src0_strid_m1 = 128

    ir_builder.emit(tvm.call_extern(
        roi_128_index.dtype, 'vadd',
        roi_128_index.access_ptr('w'),
        roi_vtrans.access_ptr('r'),
        zero_ub.access_ptr('r'),
        repeat_times,
        1, 16, 1, dst_strid_m1, src0_strid_m1, 1))

    return ir_builder.get()


# tvm.select(A xx B, C, D) -> vcmp(A, B) then vsel(dst, C, D)
@tvm.register_func("tvm.intrin.cce.vector_vcmp")
def vector_vcmp(tensor_op):
    """
    :param tensor_op: the stmt of for with psroialign
    :return: the intric stmt what we want
    """
    ins, _, _, _ = cce_util.get_dma_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()
    src0 = ins[0]
    src1 = ins[1]
    if isinstance(tensor_op.body.value.condition, tvm.expr.LT):
        cmd = 'vcmp_lt'
    elif isinstance(tensor_op.body.value.condition, tvm.expr.GT):
        cmd = 'vcmp_gt'

    ir_builder.emit(tvm.call_extern(
        src0.dtype, cmd,
        src0.access_ptr('r'),
        src1.access_ptr('r'),
        1, 1, 1, 1, 8, 8, 8))

    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.vector_vsel")
def vector_vsel(tensor_op):
    """
    :param tensor_op: the stmt of for with psroialign
    :return: the intric stmt what we want
    """
    ins, outs, _, _ = cce_util.get_dma_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()
    dst = outs[0]
    src0 = ins[2]
    src1 = ins[3]

    ir_builder.emit(tvm.call_extern(
        dst.dtype, 'vsel',
        dst.access_ptr('w'),
        src0.access_ptr('r'),
        src1.access_ptr('r'),
        1, 1, 1, 1, 8, 8, 8))
    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.vector_vsel_dummy")
def vector_vsel_dummy(tensor_op):
    """
    :param tensor_op: the stmt of for with psroialign
    :return: the intric stmt what we want
    """
    ins, outs, _, _ = cce_util.get_dma_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()
    dst = outs[0]
    src0 = ins[2]
    src1 = ins[3]

    # this instrinc is dummy for adding pipe_barrier(PIPE_V);
    # between vcmp and vsel
    ir_builder.emit(tvm.call_extern(
        src0.dtype, 'vadds',
        src0.access_ptr('w'),
        src0.access_ptr('r'),
        tvm.const(0, dtype=src0.dtype),
        1, 1, 1, 8, 8))

    ir_builder.emit(tvm.call_extern(
        dst.dtype, 'vsel',
        dst.access_ptr('rw'),
        src0.access_ptr('r'),
        src1.access_ptr('r'),
        1, 1, 1, 1, 8, 8, 8))
    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.psalign_xypos_scale_with_reg")
def psalign_xypos_scale_with_reg(tensor_op):
    """
    :param tensor_op: the stmt of for with psroialign
    :return: the intric stmt what we want
    """
    ins, outs, _, _ = cce_util.get_dma_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()

    index_arr_fp32_buf = ins[0]  # indexArr
    roi_grid_wh_fp32_buf = ins[1]  # deltaW/H
    x_y_start_end_fp32_buf = ins[2]  # x/yStartFp32
    x_ypos_fp32_buf = outs[0]
    reg_fp32 = ir_builder.allocate("float32", (1,), name="reg",
                                   scope=cce.scope_reg)
    cur_roi_offset = get_emitinsn_params("roi_loop_axis")
    cur_batch = get_emitinsn_params("thread_block")
    batch_size = get_emitinsn_params("batch_size")
    roi_num_per_batch = get_emitinsn_params("roi_num_per_batch")
    if batch_size > 1:
        roi_batch_offset = cur_batch*roi_num_per_batch
    else:
        roi_batch_offset = 0

    ir_builder.emit(tvm.call_extern(
        roi_grid_wh_fp32_buf.dtype, "reg_mov",
        tvm.call_extern(reg_fp32.dtype, "reg", reg_fp32[0]),
        roi_grid_wh_fp32_buf.access_ptr(
            "r",
            offset=cur_roi_offset+roi_batch_offset)
    ))

    ir_builder.emit(tvm.call_extern(
        x_ypos_fp32_buf.dtype, 'vmuls',
        x_ypos_fp32_buf.access_ptr('rw'),
        index_arr_fp32_buf.access_ptr('r'),
        reg_fp32[0],
        2, 1, 1, 8, 8))

    ir_builder.emit(tvm.call_extern(
        x_y_start_end_fp32_buf.dtype, "reg_mov",
        tvm.call_extern(reg_fp32.dtype, "reg", reg_fp32[0]),
        x_y_start_end_fp32_buf.access_ptr(
            "r",
            offset=cur_roi_offset+roi_batch_offset)
    ))

    ir_builder.emit(tvm.call_extern(
        x_ypos_fp32_buf.dtype, 'vadds',
        x_ypos_fp32_buf.access_ptr('rw'),
        x_ypos_fp32_buf.access_ptr('r'),
        reg_fp32[0],
        2, 1, 1, 8, 8))

    return ir_builder.get()

# pylint: disable=too-many-locals
@tvm.register_func("tvm.intrin.cce.psalign_roi_pooling_reg_mov")
def psalign_roi_pooling_reg_mov(tensor_op):
    """
    :param tensor_op: the stmt of for with psroialign
    :return: the intric stmt what we want
    """
    ins, outs, _, _ = cce_util.get_dma_buffer(tensor_op)

    cur_c0 = get_emitinsn_params("c0_times_axis_fused")
    c0_times = get_emitinsn_params("c0_times")
    fm_h = get_emitinsn_params("fm_h")
    fm_w = get_emitinsn_params("fm_w")
    group_size = get_emitinsn_params("group_size")
    sample_height = get_emitinsn_params("sample_height")
    sample_width = get_emitinsn_params("sample_width")
    fm_num_per_batch_out = get_emitinsn_params("fm_num_per_batch_out")

    # thread_block is roi in single batch, or batch in multi batch
    thread_block = get_emitinsn_params("thread_block")

    per_c0_size = fm_w*fm_h*cce.C0_SIZE

    # four2five out result of feature map need offset for multi core
    device_core_num = cce_conf.get_soc_spec("CORE_NUM")
    fm_out_thread_offset = (thread_block%device_core_num)*fm_num_per_batch_out
    # the max space that one bin can be occupied
    max_bin_size = ((fm_w*fm_h+group_size ** 2 - 1)//(group_size ** 2))*cce.C0_SIZE

    min_bin_ub_size = cce.ELEMENTS_VECTOR_OP_FP16*cce.C0_SIZE

    # when fm_h/w is too small, the max_bin_size will be small
    # this could lead to the bin buf is reused
    # so to prevent bin buf from being reused, we need to limit the min size of bin buf.
    # the min_bin_ub_size is ELEMENTS_VECTOR_OP_FP16*C0_SIZE
    if max_bin_size < min_bin_ub_size:
        max_bin_size = min_bin_ub_size

    if c0_times > 1:
        cur_c0_offset = per_c0_size*cur_c0
    else:
        cur_c0_offset = 0

    ir_builder = tvm.ir_builder.create()
    # input tensor
    fm_gm = ins[0]  # feature map
    lx_buf = ins[1]
    ly_buf = ins[2]
    hx_buf = ins[3]
    hy_buf = ins[4]
    xlow_int_buf = ins[5]
    xhigh_int_buf = ins[6]
    ylow_int_buf = ins[7]
    yhigh_int_buf = ins[8]

    x_start_int32_buf = ins[9]
    y_start_int32_buf = ins[10]
    roi_grid_w_int32_buf = ins[11]
    roi_grid_h_int32_buf = ins[12]

    res_addr_buf = outs[0]

    reg = ir_builder.allocate("uint64", (12,), name="reg", scope=cce.scope_reg)
    reset_mask_insn(ir_builder, res_addr_buf.dtype, bits=16)

    with ir_builder.for_range(0, group_size, name='ph') as loop_var_ph:
        # skip pooledH which has been computed ,
        # offset for ylowInt, yhighInt, ly, hy
        offset_ph = loop_var_ph*sample_height

        ir_builder.emit(tvm.call_extern(
            y_start_int32_buf.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[1]),
            y_start_int32_buf.access_ptr("r", offset=loop_var_ph),
        ))

        ir_builder.emit(tvm.call_extern(
            roi_grid_h_int32_buf.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[3]),
            roi_grid_h_int32_buf.access_ptr("r", offset=loop_var_ph),
        ))

        with ir_builder.for_range(0, group_size, name='pw') as loop_var_pw:
            # reg[0]      roiXStart
            # reg[1]      roiYStart
            # reg[2]      roiBinWidthInt
            # reg[3]      roiBinHeightInt
            ir_builder.emit(tvm.call_extern(
                x_start_int32_buf.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                x_start_int32_buf.access_ptr("r", offset=loop_var_pw),
            ))
            ir_builder.emit(tvm.call_extern(
                roi_grid_w_int32_buf.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[2]),
                roi_grid_w_int32_buf.access_ptr("r", offset=loop_var_pw),
            ))

            dest_offset = 128 - 16  # for vmax dst addr
            dest_addr_buffer = apply_for_new_alloc(ir_builder, "float16_t",
                                                   (max_bin_size+dest_offset,),
                                                   cce.scope_ubuf)

            ele_offset_per_c0_block = \
                (loop_var_ph*group_size+loop_var_pw)*fm_h*fm_w*cce.C0_SIZE*c0_times
            with ir_builder.for_range(0, reg[3], name="h") as loop_var_h:
                ir_builder.emit(tvm.call_extern(
                    outs[0].dtype, "copy_gm_to_ubuf",
                    dest_addr_buffer.access_ptr(
                        "w",
                        offset=loop_var_h*reg[2]*cce.C0_SIZE+dest_offset),
                    fm_gm.access_ptr(
                        "r",
                        offset=fm_out_thread_offset +
                        ele_offset_per_c0_block +
                        reg[1]*fm_w*cce.C0_SIZE +
                        reg[0]*cce.C0_SIZE +
                        loop_var_h*fm_w*cce.C0_SIZE +
                        cur_c0_offset),
                    0, 1, reg[2]*cce.C0_SIZE//16, 0, 0))
            # skip pooledW which has been computed ,
            # offset for xlowInt, xhighInt, lx, hx
            offset_pw = loop_var_pw*sample_width

            reg_half = ir_builder.allocate("float16", (4,),
                                           name="reg_half",
                                           scope=cce.scope_reg)
            reg_half[0] = tvm.const(0, dtype="float16")
            reg_half[1] = tvm.const(0, dtype="float16")
            reg_half[2] = tvm.const(0, dtype="float16")
            reg_half[3] = tvm.const(0, dtype="float16")

            # all in bin interpolation
            # input:  binW*binH*C0SIZE size of feature map
            # output: C0SIZE of pooling result
            # reg[4]     curXLow         uint64
            # reg[5]     curXHigh        uint64
            # reg[6]     curYLow         uint64
            # reg[7]     curYHigh        uint64
            reg[4] = tvm.const(0, dtype="uint64")
            reg[5] = tvm.const(0, dtype="uint64")
            reg[6] = tvm.const(0, dtype="uint64")
            reg[7] = tvm.const(0, dtype="uint64")

            len_c1 = 128
            len_c2 = 2*len_c1
            len_c3 = 3*len_c1
            grid_h = sample_height
            grid_w = sample_width
            res_ub_offset = (loop_var_ph*group_size+loop_var_pw)*cce.C0_SIZE

            # vector_dup all compute result
            # dump minimum fp16 to compare with result
            ir_builder.emit(tvm.call_extern(
                res_addr_buf.dtype, 'vector_dup',
                res_addr_buf.access_ptr('w', offset=res_ub_offset),
                tvm.const(-65503.0, dtype="float16"),
                1, 1, 1, 8, 8))

            with ir_builder.for_range(0, grid_h, name="grid_h") as loop_var_gh:
                ir_builder.emit(tvm.call_extern(
                    ylow_int_buf.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[6]),
                    ylow_int_buf.access_ptr("r", offset=offset_ph+loop_var_gh),
                ))
                # curYHigh euqals to yhighInt0 plus gh;
                ir_builder.emit(tvm.call_extern(
                    yhigh_int_buf.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[7]),
                    yhigh_int_buf.access_ptr("r", offset=offset_ph+loop_var_gh),
                ))

                # lyCur equals to ly0 plus gh;
                ir_builder.emit(tvm.call_extern(
                    ly_buf.dtype, "reg_mov",
                    tvm.call_extern(reg_half.dtype, "reg", reg_half[1]),
                    ly_buf.access_ptr("r", offset=offset_ph+loop_var_gh),
                ))

                # hyCur equals to hy0 plus gh;
                ir_builder.emit(tvm.call_extern(
                    hy_buf.dtype, "reg_mov",
                    tvm.call_extern(reg_half.dtype, "reg", reg_half[3]),
                    hy_buf.access_ptr("r", offset=offset_ph+loop_var_gh),
                ))

                reg[6] = reg[6] - reg[1]
                reg[7] = reg[7] - reg[1]

                with ir_builder.for_range(0, grid_w,
                                          name="grid_w") as loop_var_gw:
                    out_addr_buf = apply_for_new_alloc(ir_builder, "float16_t",
                                                       (128,),
                                                       cce.scope_ubuf)
                    ir_builder.emit(tvm.call_extern(
                        out_addr_buf.dtype, 'vector_dup',
                        out_addr_buf.access_ptr('w'),
                        tvm.const(0.0, dtype="float16"),
                        1, 1, 1, 8, 8))

                    temp_addr_buf = apply_for_new_alloc(ir_builder, "float16_t",
                                                        (4, 128), cce.scope_ubuf)
                    ir_builder.emit(tvm.call_extern(
                        temp_addr_buf.dtype, 'vector_dup',
                        temp_addr_buf.access_ptr('w'),
                        tvm.const(0.0, dtype="float16"),
                        1, 1, 1, 8, 8))
                    # reg[4]     curXLow         uint64
                    # reg[5]     curXHigh        uint64
                    ir_builder.emit(tvm.call_extern(
                        xlow_int_buf.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[4]),
                        xlow_int_buf.access_ptr("r",
                                                offset=offset_pw+loop_var_gw),
                    ))

                    ir_builder.emit(tvm.call_extern(
                        xhigh_int_buf.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[5]),
                        xhigh_int_buf.access_ptr("r",
                                                 offset=offset_pw+loop_var_gw),
                    ))

                    # lxCur
                    ir_builder.emit(tvm.call_extern(
                        lx_buf.dtype, "reg_mov",
                        tvm.call_extern(reg_half.dtype, "reg", reg_half[0]),
                        lx_buf.access_ptr("r", offset=offset_pw+loop_var_gw),
                    ))

                    # hxCur
                    ir_builder.emit(tvm.call_extern(
                        hx_buf.dtype, "reg_mov",
                        tvm.call_extern(reg_half.dtype, "reg", reg_half[2]),
                        hx_buf.access_ptr("r", offset=offset_pw+loop_var_gw),
                    ))

                    # reg[4]     curXLow         uint64
                    # reg[5]     curXHigh        uint64
                    reg[4] = reg[4] - reg[0]
                    reg[5] = reg[5] - reg[0]

                    # reg[0]      roiXStart
                    # reg[1]      roiYStart
                    # reg[2]      roiBinWidthInt
                    # reg[3]      roiBinHeightInt

                    # reg[8]      idx1            uint64
                    # reg[9]      idx2            uint64
                    # reg[10]      idx3            uint64
                    # reg[11]      idx4            uint64

                    # reg[4]     curXLow         uint64
                    # reg[5]     curXHigh        uint64
                    # reg[6]     curYLow         uint64
                    # reg[7]     curYHigh        uint64

                    reg[8] = (reg[6]*reg[2] + reg[4])*\
                             tvm.const(cce.C0_SIZE, dtype="uint64")
                    reg[9] = (reg[6]*reg[2] + reg[5])*\
                             tvm.const(cce.C0_SIZE, dtype="uint64")
                    reg[10] = (reg[7]*reg[2] + reg[4])*\
                              tvm.const(cce.C0_SIZE, dtype="uint64")
                    reg[11] = (reg[7]*reg[2] + reg[5])*\
                              tvm.const(cce.C0_SIZE, dtype="uint64")

                    # align_value
                    # equals to m1*hx*hy+m2*lx*hy+m3*hx*ly+m4*lx*ly
                    # equals to (m1*hx+m2*lx)*hy+(m3*hx+m4*lx)*ly
                    # m1 is point of top left, m2 is point of top right
                    # m3 is point of bottom left, m4 is point of bottom right

                    # reg_half[0]     lx
                    # reg_half[1]     ly
                    # reg_half[2]     hx
                    # reg_half[3]     hy

                    # m1*hx
                    ir_builder.emit(tvm.call_extern(
                        temp_addr_buf.dtype, 'vmuls',
                        temp_addr_buf.access_ptr('w'),
                        dest_addr_buffer.access_ptr('r',
                                                    offset=reg[8]+dest_offset),
                        reg_half[2],
                        1, 1, 1, 8, 8))

                    # m2*lx
                    ir_builder.emit(tvm.call_extern(
                        temp_addr_buf.dtype, 'vmuls',
                        temp_addr_buf.access_ptr('w', offset=len_c1),
                        dest_addr_buffer.access_ptr('r',
                                                    offset=reg[9]+dest_offset),
                        reg_half[0], 1, 1, 1, 8, 8))

                    # m3*hx
                    ir_builder.emit(tvm.call_extern(
                        temp_addr_buf.dtype, 'vmuls',
                        temp_addr_buf.access_ptr('w', offset=len_c2),
                        dest_addr_buffer.access_ptr('r',
                                                    offset=reg[10]+dest_offset),
                        reg_half[2], 1, 1, 1, 8, 8))

                    # m4*lx
                    ir_builder.emit(tvm.call_extern(
                        temp_addr_buf.dtype, 'vmuls',
                        temp_addr_buf.access_ptr('w', offset=len_c3),
                        dest_addr_buffer.access_ptr('r',
                                                    offset=reg[11]+dest_offset),
                        reg_half[0], 1, 1, 1, 8, 8))

                    # m1*hx+m2*lx
                    ir_builder.emit(tvm.call_extern(
                        temp_addr_buf.dtype, 'vadd',
                        temp_addr_buf.access_ptr('w'),
                        temp_addr_buf.access_ptr('r'),
                        temp_addr_buf.access_ptr('r', offset=len_c1),
                        1, 1, 1, 1, 8, 8, 8))

                    # m3*hx+m4*lx
                    ir_builder.emit(tvm.call_extern(
                        temp_addr_buf.dtype, 'vadd',
                        temp_addr_buf.access_ptr('w', offset=len_c2),
                        temp_addr_buf.access_ptr('r', offset=len_c2),
                        temp_addr_buf.access_ptr('r', offset=len_c3),
                        1, 1, 1, 1, 8, 8, 8))

                    # (m1*hx+m2*lx)*hy
                    ir_builder.emit(tvm.call_extern(
                        temp_addr_buf.dtype, 'vmuls',
                        temp_addr_buf.access_ptr('w'),
                        temp_addr_buf.access_ptr('r'),
                        reg_half[3],
                        1, 1, 1, 8, 8))

                    # (m3*hx+m4*lx)*ly
                    ir_builder.emit(tvm.call_extern(
                        temp_addr_buf.dtype, 'vmuls',
                        temp_addr_buf.access_ptr('w', offset=len_c2),
                        temp_addr_buf.access_ptr('r', offset=len_c2),
                        reg_half[1],
                        1, 1, 1, 8, 8))

                    # add (m1*hx+m2*lx)*hy to out
                    # add (m3*hx+m4*lx)*ly to out

                    ir_builder.emit(tvm.call_extern(
                        out_addr_buf.dtype, 'vadd',
                        out_addr_buf.access_ptr('w'),
                        out_addr_buf.access_ptr('r'),
                        temp_addr_buf.access_ptr('r'),
                        1, 1, 1, 1, 8, 8, 8))

                    ir_builder.emit(tvm.call_extern(
                        out_addr_buf.dtype, 'vadd',
                        out_addr_buf.access_ptr('w'),
                        out_addr_buf.access_ptr('r'),
                        temp_addr_buf.access_ptr('r', offset=len_c2),
                        1, 1, 1, 1, 8, 8, 8))

                    ir_builder.emit(tvm.call_extern(
                        res_addr_buf.dtype, 'vmax',
                        res_addr_buf.access_ptr('w', offset=res_ub_offset),
                        res_addr_buf.access_ptr('r', offset=res_ub_offset),
                        out_addr_buf.access_ptr('r'),
                        1, 1, 1, 1, 8, 8, 8))

    reset_mask_insn(ir_builder, res_addr_buf.dtype, bits=128)
    return ir_builder.get()

# pylint: disable=too-many-locals
@tvm.register_func("tvm.intrin.cce.psalign_roi_valid")
def psalign_roi_valid(tensor_op):
    """
    :param tensor_op: the stmt of for with psroialign
    :return: the intric stmt what we want
    """
    ins, _, _, _ = cce_util.get_dma_buffer(tensor_op)
    src_buf = ins[0]
    dst_buf = ins[1]
    # src_buf and dst_buf's sequence is uncertain
    if isinstance(tensor_op.body.value.condition, tvm.expr.GT):
        cmd = "vcmpv_gt"
        src_strid_m0 = 1
        src_strid_m1 = 8
        dst_strid_m0 = 0
        dst_strid_m1 = 0
    elif isinstance(tensor_op.body.value.condition, tvm.expr.LT):
        cmd = "vcmpv_lt"
        src_strid_m0 = 0
        src_strid_m1 = 0
        dst_strid_m0 = 1
        dst_strid_m1 = 8

    batch_size = get_emitinsn_params("batch_size")
    thread_block = get_emitinsn_params("thread_block")
    roi_num_per_core = get_emitinsn_params("roi_num_per_batch")
    roi_num_loop = roi_num_per_core
    valide_roi_reg_idx = thread_block
    if batch_size == 1:
        valide_roi_reg_idx = 0
        roi_num_per_core = roi_num_per_core//2
        psalign_roi_num_reg = get_emitinsn_params("psalign_roi_num_reg")
        roi_num_loop = psalign_roi_num_reg[thread_block]

    vcmpv_repeat = (roi_num_per_core+7)//8  # compare 8 rois per repeat

    ir_builder = tvm.ir_builder.create()
    psalign_valid_roi_reg = ir_builder.allocate("uint64", (batch_size,),
                                                name="psalign_valid_roi_reg",
                                                scope=cce.scope_reg)
    cce_emitinsn_params.cceEmitParamsIns.insert_param("psalign_valid_roi_reg",
                                                      psalign_valid_roi_reg)

    reg_type = "uint16"
    flag_reg = ir_builder.allocate(reg_type, (2,), name="flag_reg",
                                   scope=cce.scope_reg)
    flag_reg[0] = tvm.const(-1, reg_type)
    flag_reg[1] = tvm.const(0, reg_type)
    cmp_res_buf = apply_for_new_alloc(ir_builder, "uint8", [roi_num_per_core*2, ],
                                      scope=cce.scope_ubuf)
    skip_tsh = 32  # skip one block threshold
    psalign_valid_roi_reg[valide_roi_reg_idx] = tvm.const(0, "uint64")
    ir_builder.emit(tvm.call_extern(
        "uint8", cmd,
        cmp_res_buf.access_ptr("w", offset=skip_tsh),
        src_buf.access_ptr("r",
                           offset=roi_num_per_core*16*valide_roi_reg_idx),
        dst_buf.access_ptr("r"),
        vcmpv_repeat, 1, src_strid_m0, dst_strid_m0,
        8, src_strid_m1, dst_strid_m1))

    # the invalid roi after valid roi and continuous, and
    # the invalid roi all vale is -65504,
    # so when find the first invlaid roi, break,
    # the roi after this roi all invalid roi
    with ir_builder.for_range(0, roi_num_loop, name="roi_index") as roi_index:
        ir_builder.emit(tvm.call_extern(reg_type, "reg_mov",
                                        tvm.call_extern(flag_reg.dtype, "reg",
                                                        flag_reg[1]),
                                        cmp_res_buf.access_ptr(
                                            "r",
                                            offset=skip_tsh+roi_index*2)))
        with ir_builder.if_scope((flag_reg[0] &
                                  flag_reg[1]) == tvm.const(0, dtype=reg_type)):
            ir_builder.emit(tvm.call_extern("uint64", "break"))
        psalign_valid_roi_reg[valide_roi_reg_idx] += tvm.const(1, "uint64")

    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.psalign_break_invalid_roi")
def psalign_break_invalid_roi(tensor_op):
    # pylint: disable=unused-argument
    """
    :param tensor_op: the stmt of for with psroialign
    :return: the intric stmt what we want
    """
    thread_block = get_emitinsn_params("thread_block")
    cur_roi = get_emitinsn_params("roi_loop_axis")
    batch_size = get_emitinsn_params("batch_size")
    ir_builder = tvm.ir_builder.create()
    psalign_valid_roi_reg = get_emitinsn_params("psalign_valid_roi_reg")
    roi_pad_num = get_emitinsn_params("roi_pad_num")

    if batch_size == 1:
        psalign_roi_num_reg = get_emitinsn_params("psalign_roi_num_reg")
        valid_roi_num = psalign_valid_roi_reg[0]
    else:
        valid_roi_num = psalign_valid_roi_reg[thread_block]

    if roi_pad_num == 1:
        with ir_builder.if_scope(psalign_roi_num_reg[thread_block] <= cur_roi):
            ir_builder.emit(tvm.call_extern("uint64", "break"))

    with ir_builder.if_scope(valid_roi_num <= cur_roi):
        ir_builder.emit(tvm.call_extern("uint64", "break"))

    return ir_builder.get()

# pylint: disable=too-many-locals
@tvm.register_func("tvm.intrin.cce.psalign_dma_copy_invalid_roi")
def psalign_dma_copy_invalid_roi(tensor_op):
    """
    :param tensor_op: the stmt of for with psroialign
    :return: the intric stmt what we want
    """
    ins, _, _, _ = cce_util.get_dma_buffer(tensor_op)
    out_buf = ins[0]

    psalign_valid_roi_reg = get_emitinsn_params("psalign_valid_roi_reg")
    roi_num_per_core = get_emitinsn_params("roi_num_per_batch")
    c0_times = get_emitinsn_params("c0_times")
    group_size = get_emitinsn_params("group_size")
    batch_size = get_emitinsn_params("batch_size")
    thread_block = get_emitinsn_params("thread_block").var
    pooled_area = group_size ** 2

    roi_copy_thread_offset = 0

    if batch_size == 1:
        psalign_roi_num_reg = get_emitinsn_params("psalign_roi_num_reg")
        roi_num_loop = psalign_roi_num_reg[2]
        valid_roi_num = psalign_valid_roi_reg[0]
        roi_num_per_core = roi_num_per_core//2
        roi_copy_thread_offset = thread_block*roi_num_per_core*c0_times
    else:
        roi_num_loop = roi_num_per_core
        valid_roi_num = psalign_valid_roi_reg[thread_block]

    ir_builder = tvm.ir_builder.create()
    dst_buf = tvm.decl_buffer(shape=out_buf.shape,
                              dtype=out_buf.dtype,
                              name=out_buf.name,
                              scope=out_buf.scope,
                              data=out_buf.data,
                              elem_offset=0,
                              offset_factor=out_buf.offset_factor)

    invalid_roi_num = roi_num_loop - valid_roi_num
    dup_0_repeat = ((pooled_area*16 - 1)//128)+1
    dup_0_buf = apply_for_new_alloc(ir_builder, "float16", (dup_0_repeat*128,),
                                    scope=cce.scope_ubuf)

    dup_0_repeat_time = dup_0_repeat//255
    dup_0_repeat_tail = dup_0_repeat%255
    if dup_0_repeat_time > 0:
        for i in range(dup_0_repeat_time):
            ir_builder.emit(tvm.call_extern("float16", "vector_dup",
                                            dup_0_buf.access_ptr(
                                                "w",
                                                offset=32640*i),
                                            tvm.const(0, dtype="float16"),
                                            255, 1, 1, 8, 8))

    if dup_0_repeat_tail > 0:
        ir_builder.emit(tvm.call_extern("float16", "vector_dup",
                                        dup_0_buf.access_ptr(
                                            "w",
                                            offset=32640*dup_0_repeat_time),
                                        tvm.const(0, dtype="float16"),
                                        dup_0_repeat_tail, 1, 1, 8, 8))

    sid = 0
    n_burst = 1
    len_burst = pooled_area
    dst_strid = 0
    src_strid = 0
    src_offset = 0
    intrin_cmd = "copy_ubuf_to_gm"
    with ir_builder.for_range(0, invalid_roi_num, name="roi_index") as roi_index:
        with ir_builder.for_range(0, c0_times, name="c0_index") as c0_index:
            dst_offset = (roi_copy_thread_offset+valid_roi_num*c0_times +
                          roi_index*c0_times+c0_index)*pooled_area*cce.C0_SIZE
            ir_builder.emit(tvm.call_extern(
                dst_buf.dtype, intrin_cmd,
                dst_buf.access_ptr("w", offset=dst_offset),
                dup_0_buf.access_ptr("r", offset=src_offset),
                sid, n_burst, len_burst, src_strid, dst_strid))
    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.psalign_roi_cmplt_sel")
def psalign_roi_cmplt_sel(tensor_op):
    """
    :param tensor_op: the stmt of for with psroialign
    :return: the intric stmt what we want
    """
    ins, outs, _, _ = cce_util.get_dma_buffer(tensor_op)
    roi_start_buf = ins[0]
    roi_start_add_one_buf = ins[1]
    roi_end_buf = outs[0]
    roi_aligned_num = roi_start_buf.shape[0]
    ir_builder = tvm.ir_builder.create()

    # vsel only can process 128 numbers, so when roi_aligned_num big than 128,
    # we split multi times use vsel
    for i in range(roi_aligned_num.value//128):
        ir_builder.emit(tvm.call_extern(
            roi_start_buf.dtype, "vcmp_lt",
            roi_start_buf.access_ptr("r", offset=i*128),
            roi_start_add_one_buf.access_ptr("r", offset=i*128),
            1, 1, 1, 1, 8, 8, 8))

        ir_builder.emit(tvm.call_extern(
            roi_end_buf.dtype, "vsel",
            roi_end_buf.access_ptr("w", offset=i*128),
            roi_start_add_one_buf.access_ptr("r", offset=i*128),
            roi_start_buf.access_ptr("r", offset=i*128),
            1, 1, 1, 1, 8, 8, 8))
    return ir_builder.get()

# pylint: disable=too-many-locals
@tvm.register_func("tvm.intrin.cce.dma_copy_for_roi_multi_core")
def dma_copy_for_roi_multi_core(tensor_op):
    """
    :param tensor_op: the stmt of for with psroialign, just for roi multi core template
    :return: the intric stmt what we want
    """
    ins, outs, _, _ = cce_util.get_dma_buffer(tensor_op)
    src = ins[0]
    dst = outs[0]

    roi_num_per_core = get_emitinsn_params("roi_num_per_batch")//2
    roi_pad_num = get_emitinsn_params("roi_pad_num")

    thread_block = get_emitinsn_params("thread_block").var
    roi_copy_thread_offset = thread_block*roi_num_per_core*16

    ir_builder = tvm.ir_builder.create()

    psalign_roi_num_reg = ir_builder.allocate("uint64", (3,),
                                              name="psalign_roi_num_reg",
                                              scope=cce.scope_reg)
    cce_emitinsn_params.cceEmitParamsIns.insert_param("psalign_roi_num_reg",
                                                      psalign_roi_num_reg)
    psalign_roi_num_reg[0] = tvm.const(roi_num_per_core, dtype='uint64')
    psalign_roi_num_reg[1] = tvm.const(roi_num_per_core - roi_pad_num,
                                       dtype='uint64')
    psalign_roi_num_reg[2] = psalign_roi_num_reg[thread_block]

    sid = 0
    nburst = 1
    burst = psalign_roi_num_reg[thread_block]
    src_stride = 0
    dst_stride = 0

    ir_builder.emit(tvm.call_extern(
        dst.dtype, "copy_gm_to_ubuf",
        dst.access_ptr("w"),  # dst buffer
        src.access_ptr("r", offset=roi_copy_thread_offset),  # src buffer
        sid,
        nburst,
        burst,
        src_stride,
        dst_stride
    ))

    return ir_builder.get()


# intrin for psroialign end
@tvm.register_func("tvm.intrin.cce.dma_padding")
def dma_padding(tensor_op):
    '''
    dma_padding
    '''
    ins, outs, _, sel_var = cce_util.get_dma_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()

    obuf = tvm.decl_buffer(outs[0].shape, outs[0].dtype,
                           name=outs[0].name+'.Local.UB',
                           data=outs[0].data,
                           offset_factor=outs[0].offset_factor,
                           data_alignment=outs[0].data_alignment,
                           scope=cce.scope_ubuf,
                           strides=outs[0].strides,
                           elem_offset=outs[0].elem_offset)
    if sel_var != []:
        with ir_builder.if_scope(tvm.ir_pass.Simplify(sel_var[0].condition)):
            if cce_util.is_const(sel_var[0].true_value):
                vec_broadcast(ir_builder, "vector_dup", [obuf],
                              cce_util.get_op_lenth(tensor_op), 1, [0])
            else:
                cce_util.dma_copy(ir_builder, ins[0], obuf)
        with ir_builder.else_scope():
            if cce_util.is_const(sel_var[0].true_value):
                cce_util.dma_copy(ir_builder, ins[0], obuf)
            else:
                vec_broadcast(ir_builder, "vector_dup", [obuf],
                              cce_util.get_op_lenth(tensor_op), 1, [0])
    else:
        cce_util.dma_copy(ir_builder, ins[0], obuf)

    return ir_builder.get()

# pylint: disable=too-many-locals
def vec_binary_cmpsel(tensor_op, extern_args, args=None):
    # pylint: disable=unused-argument
    '''
    vec_binary_cmpsel
    '''
    # for pylint
    ins, outs = cce_util.get_buffer(tensor_op)

    if not ins:
        raise RuntimeError("vec_cmp_sel at least has one src")

    if len(outs) != 1:
        raise RuntimeError("vec_cmp_sel only has one dst")

    size = cce_util.get_op_lenth(tensor_op)
    block_len = 256
    src_dtype = ins[0].dtype

    ir_builder = tvm.ir_builder.create()

    repeat_cal_dtype = src_dtype
    cal_bit_len = cce_util.get_bits_of(repeat_cal_dtype)
    cal_once_len = block_len*8//cal_bit_len

    dst_dtype = outs[0].dtype
    repeat_times = size//cal_once_len
    remain_len = size - repeat_times*cal_once_len

    is_scalar_flag_list = [False, False, False, False]
    for i in range(1, len(extern_args)):
        if extern_args[i] is not None:
            scalar_buffer = cce_util.apply_for_new_alloc(ir_builder, src_dtype,
                                                         (128,), cce.scope_ubuf,
                                                         "scalar"+str(i))
            vec_cmd_factory(ir_builder, "vector_dup", [], [scalar_buffer],
                            cal_once_len, 1,
                            [tvm.const(extern_args[i], dtype=src_dtype)],
                            [1, 1, 8, 8])
            ins.insert(i - 1, scalar_buffer)
            is_scalar_flag_list[i - 1] = True

    if repeat_times > 0:
        with ir_builder.for_range(0, repeat_times, name="cmp_index") as cmp_index:
            repeat_offset = cal_once_len*cmp_index
            repeat_offset_list = [repeat_offset, repeat_offset, repeat_offset,
                                  repeat_offset]
            for i, _ in enumerate(is_scalar_flag_list):
                if is_scalar_flag_list[i] is True:
                    repeat_offset_list[i] = 0
            # cmp the input1 and input2
            ir_builder.emit(tvm.call_extern(dst_dtype,
                                            "vcmp_"+extern_args[0],
                                            ins[0].access_ptr(
                                                "r",
                                                offset=repeat_offset_list[0]),
                                            ins[1].access_ptr(
                                                "r",
                                                offset=repeat_offset_list[1]),
                                            1, 1, 1, 1, 8, 8, 8))
            # sel the input3 and input4
            ir_builder.emit(tvm.call_extern(dst_dtype,
                                            "vsel",
                                            outs[0].access_ptr(
                                                "rw",
                                                offset=repeat_offset),
                                            ins[2].access_ptr(
                                                "r",
                                                offset=repeat_offset_list[2]),
                                            ins[3].access_ptr(
                                                "r",
                                                offset=repeat_offset_list[3]),
                                            1, 1, 1, 1, 8, 8, 8))

    if remain_len > 0:
        repeat_offset = repeat_times*cal_once_len
        repeat_offset_list = [repeat_offset, repeat_offset, repeat_offset,
                              repeat_offset]
        for i, _ in enumerate(is_scalar_flag_list):
            if is_scalar_flag_list[i] is True:
                repeat_offset_list[i] = 0
        # cmp the input1 and input2
        reset_mask_insn(ir_builder, dst_dtype, bits=remain_len)
        ir_builder.emit(tvm.call_extern(dst_dtype,
                                        "vcmp_"+extern_args[0],
                                        ins[0].access_ptr(
                                            "r",
                                            offset=repeat_offset_list[0]),
                                        ins[1].access_ptr(
                                            "r",
                                            offset=repeat_offset_list[1]),
                                        1, 1, 1, 1, 8, 8, 8))
        # sel the input3 and input4
        ir_builder.emit(tvm.call_extern(dst_dtype,
                                        "vsel",
                                        outs[0].access_ptr(
                                            "rw",
                                            offset=repeat_offset),
                                        ins[2].access_ptr(
                                            "r",
                                            offset=repeat_offset_list[2]),
                                        ins[3].access_ptr(
                                            "r",
                                            offset=repeat_offset_list[3]),
                                        1, 1, 1, 1, 8, 8, 8))

    reset_mask_insn(ir_builder, dst_dtype)

    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.vector_cmpsel")
def elewise_binary_cmpsel(tensor_op):
    '''
    elewise_binary_cmpsel
    '''
    args = cce_util.get_cmp_sel_args(tensor_op)
    return vec_binary_cmpsel(tensor_op, args)


@tvm.register_func("tvm.intrin.cce.elewise_binary_logic")
def elewise_binary_logic(tensor_op):
    '''
    elewise_binary_logic
    '''
    args = cce_util.get_logic_args(tensor_op)
    if args[0] == 'not':
        return vec_VSsingle_elewise(tensor_op, "vlogic", args)
    return vec_binary_elewise_with_ext(tensor_op, "vlogic", args)

def _get_decl_buffer(buffer_tensors):
    """
    get buffer by decl_buffer for clear index in loop
    """
    ten_tmp = []
    for ten_buffer in buffer_tensors:
        buffers_ten = tvm.decl_buffer(shape=ten_buffer.shape,
                                      dtype=ten_buffer.dtype,
                                      name=ten_buffer.name,
                                      scope=ten_buffer.scope,
                                      data=ten_buffer.data,
                                      elem_offset=0,
                                      offset_factor=ten_buffer.offset_factor)
        ten_tmp.append(buffers_ten)

    return ten_tmp

# pylint: disable=too-many-locals
def vec_binary_cmp(tensor_op):
    '''
    vec_binary_cmp
    '''
    args = cce_util.get_cmp_args(tensor_op)
    if args[1] == 'bool':
        src_buffers, dst_buffers = cce_util.get_buffer(tensor_op, True)
    else:
        src_buffers, dst_buffers = cce_util.get_cmp_bit_buffer(tensor_op)

    if len(dst_buffers) != 1:
        raise RuntimeError("vec_binary_cmp only support ONE dst buffer ")
    src_buffers = cce_util.del_ins_reuse_out(src_buffers, dst_buffers[0])
    if len(src_buffers) > 2:
        raise RuntimeError("vec_binary_cmp only support ONE or TWO src buffer ")
    # decl_buffer for clear index in loop
    src_buffers = _get_decl_buffer(src_buffers)
    dst_buffers = _get_decl_buffer(dst_buffers)

    size = cce_util.get_op_lenth(tensor_op)
    op_cmd = "vcmp_"+args[0]
    ir_builder = tvm.ir_builder.create()

    block_len = 256

    src_dtype = src_buffers[0].dtype

    repeat_cal_dtype = src_dtype
    cal_bit_len = cce_util.get_bits_of(repeat_cal_dtype)
    cal_once_len = block_len*8//cal_bit_len

    dst_dtype = dst_buffers[0].dtype
    repeat_times = size//cal_once_len
    remain_len = size - repeat_times*cal_once_len

    if args[2] is not None:
        scalar_buffer = cce_util.apply_for_new_alloc(ir_builder, src_dtype, (128,),
                                                     cce.scope_ubuf, "scalar")
        vec_cmd_factory(ir_builder, "vector_dup", [], [scalar_buffer], cal_once_len, 1,
                        [tvm.const(args[2], dtype=src_dtype)], [1, 1, 8, 8])
        src_buffers.insert(0, scalar_buffer)
    if args[3] is not None:
        scalar_buffer = cce_util.apply_for_new_alloc(ir_builder, src_dtype, (128,),
                                                     cce.scope_ubuf, "scalar")
        vec_cmd_factory(ir_builder, "vector_dup", [], [scalar_buffer], cal_once_len, 1,
                        [tvm.const(args[3], dtype=src_dtype)], [1, 1, 8, 8])
        src_buffers.insert(1, scalar_buffer)

    if args[1] == 'bool':
        ones_buffer = cce_util.apply_for_new_alloc(ir_builder, "float16", (128,),
                                                   name="ones",
                                                   scope=cce.scope_ubuf)
        zeros_buffer = cce_util.apply_for_new_alloc(ir_builder, "float16", (128,),
                                                    name="zeros",
                                                    scope=cce.scope_ubuf)
        fp16_res_buffer = cce_util.apply_for_new_alloc(ir_builder, "float16",
                                                       dst_buffers[0].shape,
                                                       name="fp16_res",
                                                       scope=cce.scope_ubuf)
        vec_cmd_factory(ir_builder, "vector_dup", [], [ones_buffer], 128, 1,
                        [tvm.const(1.0, "float16")],
                        [1, 1, 8, 8])
        vec_cmd_factory(ir_builder, "vector_dup", [], [zeros_buffer], 128, 1,
                        [tvm.const(0.0, "float16")],
                        [1, 1, 8, 8])
        if repeat_times > 0:
            with ir_builder.for_range(0, repeat_times, name="cmp_index") as cmp_index:
                repeat_offset = cal_once_len*cmp_index
                first_src_repeat_offset = repeat_offset
                second_src_repeat_offset = repeat_offset
                if args[2] is not None:
                    first_src_repeat_offset = 0
                if args[3] is not None:
                    second_src_repeat_offset = 0
                # cmp the input1 and input2
                ir_builder.emit(tvm.call_extern(dst_dtype,
                                                op_cmd,
                                                src_buffers[0].access_ptr(
                                                    "r",
                                                    offset=first_src_repeat_offset),
                                                src_buffers[1].access_ptr(
                                                    "r",
                                                    offset=second_src_repeat_offset),
                                                1, 1, 1, 1, 8, 8, 8))

                reset_mask_insn(ir_builder, "float16", bits=cal_once_len)
                ir_builder.emit(tvm.call_extern("float16",
                                                "vsel",
                                                fp16_res_buffer.access_ptr(
                                                    "rw",
                                                    offset=repeat_offset),
                                                ones_buffer.access_ptr("r"),
                                                zeros_buffer.access_ptr("r"),
                                                1, 1, 1, 1, 8, 8, 8))
                reset_mask_insn(ir_builder, "float16")

        if remain_len > 0:
            reset_mask_insn(ir_builder, src_dtype, bits=remain_len)
            repeat_src_offset = repeat_times*cal_once_len
            repeat_dst_offset = repeat_times*cal_once_len
            first_src_repeat_offset = repeat_src_offset
            second_src_repeat_offset = repeat_src_offset
            if args[2] is not None:
                first_src_repeat_offset = 0
            if args[3] is not None:
                second_src_repeat_offset = 0
            # cmp the input1 and input2
            ir_builder.emit(tvm.call_extern(dst_dtype,
                                            op_cmd,
                                            src_buffers[0].access_ptr(
                                                "r",
                                                offset=first_src_repeat_offset),
                                            src_buffers[1].access_ptr(
                                                "r",
                                                offset=second_src_repeat_offset),
                                            1, 1, 1, 1, 8, 8, 8))
            ir_builder.emit(tvm.call_extern("float16",
                                            "vsel",
                                            fp16_res_buffer.access_ptr(
                                                "rw",
                                                offset=repeat_dst_offset),
                                            ones_buffer.access_ptr("r"),
                                            zeros_buffer.access_ptr("r"),
                                            1, 1, 1, 1, 8, 8, 8))

        reset_mask_insn(ir_builder, "float16")
        vec_cmd_factory(ir_builder, 'vconv_f162s8', [fp16_res_buffer],
                        [dst_buffers[0]],
                        size, 1, [],
                        [1, 1, 4, 8],
                        repeat_cal_dtype='fp16')
    else:
        multiples = cal_bit_len // 8  # 2 for fp16, 4 for fp32
        if remain_len == 0 or repeat_times % multiples == 0:
            # the vcmpv's dst addr must be 32B align, so only when size is
            # 128 multiple(remain_len == 0) or the repeat_times is 2 multipe
            # (the dst addr is 32B align when compute the tail), we can use
            # vcmpv instric. otherwise, use for loop and vcmp and get_cmpmask
            # to compute the mask.
            op_cmd = "vcmpv_" + args[0]
            count = 0
            max_repeat_times = 254
            if repeat_times > max_repeat_times:
                count = repeat_times // max_repeat_times
                with ir_builder.for_range(0, count,
                                          name="cmp_index") as cmp_index:
                    repeat_offset = cal_once_len * cmp_index
                    dst_repeat_offset = cal_once_len // 8 * cmp_index
                    first_src_repeat_offset = repeat_offset
                    second_src_repeat_offset = repeat_offset
                    vcmpv_repeat_stride = (1, 1, 1, 8, 8, 8)
                    if args[2] is not None:
                        first_src_repeat_offset = 0
                        vcmpv_repeat_stride = (1, 1, 1, 8, 0, 8)
                    if args[3] is not None:
                        second_src_repeat_offset = 0
                        vcmpv_repeat_stride = (1, 1, 1, 8, 8, 0)
                    # cmp the input1 and input2
                    ir_builder.emit(tvm.call_extern(dst_dtype,
                                                    op_cmd,
                                                    dst_buffers[0].access_ptr(
                                                        "wr",
                                                        offset=dst_repeat_offset),
                                                    src_buffers[0].access_ptr(
                                                        "r",
                                                        offset=first_src_repeat_offset),
                                                    src_buffers[1].access_ptr(
                                                        "r",
                                                        offset=second_src_repeat_offset),
                                                    max_repeat_times, *vcmpv_repeat_stride))
                repeat_times = repeat_times % max_repeat_times

            if repeat_times > 0:
                repeat_offset = cal_once_len * (count * max_repeat_times)
                dst_repeat_offset = cal_once_len // 8 * (count * max_repeat_times)
                first_src_repeat_offset = repeat_offset
                second_src_repeat_offset = repeat_offset
                vcmpv_repeat_stride = (1, 1, 1, 8, 8, 8)
                if args[2] is not None:
                    first_src_repeat_offset = 0
                    vcmpv_repeat_stride = (1, 1, 1, 8, 0, 8)
                if args[3] is not None:
                    second_src_repeat_offset = 0
                    vcmpv_repeat_stride = (1, 1, 1, 8, 8, 0)
                # cmp the input1 and input2
                ir_builder.emit(tvm.call_extern(dst_dtype,
                                                op_cmd,
                                                dst_buffers[0].access_ptr(
                                                    "wr",
                                                    offset=dst_repeat_offset),
                                                src_buffers[0].access_ptr(
                                                    "r",
                                                    offset=first_src_repeat_offset),
                                                src_buffers[1].access_ptr(
                                                    "r",
                                                    offset=second_src_repeat_offset),
                                                repeat_times, *vcmpv_repeat_stride))

            if remain_len > 0:
                reset_mask_insn(ir_builder, src_dtype, bits=remain_len)
                repeat_src_offset = (count * max_repeat_times + repeat_times) * cal_once_len
                repeat_dst_offset = (count * max_repeat_times + repeat_times) * cal_once_len // 8
                first_src_repeat_offset = repeat_src_offset
                second_src_repeat_offset = repeat_src_offset
                vcmpv_repeat_stride = (1, 1, 1, 8, 8, 8)
                if args[2] is not None:
                    first_src_repeat_offset = 0
                    vcmpv_repeat_stride = (1, 1, 1, 8, 0, 8)
                if args[3] is not None:
                    second_src_repeat_offset = 0
                    vcmpv_repeat_stride = (1, 1, 1, 8, 8, 0)
                # cmp the input1 and input2
                ir_builder.emit(tvm.call_extern(dst_dtype,
                                                op_cmd,
                                                dst_buffers[0].access_ptr(
                                                    "wr",
                                                    offset=repeat_dst_offset),
                                                src_buffers[0].access_ptr(
                                                    "r",
                                                    offset=first_src_repeat_offset),
                                                src_buffers[1].access_ptr(
                                                    "r",
                                                    offset=second_src_repeat_offset),
                                                1, *vcmpv_repeat_stride))

            reset_mask_insn(ir_builder, src_dtype)
        else:
            if repeat_times > 0:
                with ir_builder.for_range(0, repeat_times,
                                          name="cmp_index") as cmp_index:
                    repeat_offset = cal_once_len*cmp_index
                    first_src_repeat_offset = repeat_offset
                    second_src_repeat_offset = repeat_offset
                    if args[2] is not None:
                        first_src_repeat_offset = 0
                    if args[3] is not None:
                        second_src_repeat_offset = 0
                    # cmp the input1 and input2
                    ir_builder.emit(tvm.call_extern(dst_dtype,
                                                    op_cmd,
                                                    src_buffers[0].access_ptr(
                                                        "r",
                                                        offset=first_src_repeat_offset),
                                                    src_buffers[1].access_ptr(
                                                        "r",
                                                        offset=second_src_repeat_offset),
                                                    1, 1, 1, 1, 8, 8, 8))
                    # get 16 uint8 cmp result from cmp_mask
                    ir_builder.emit(tvm.call_extern(
                        dst_buffers[0].dtype, "get_cmpmask",
                        dst_buffers[0].access_ptr(
                            "wr",
                            offset=cal_once_len//8*cmp_index)))

            if remain_len > 0:
                reset_mask_insn(ir_builder, src_dtype, bits=remain_len)
                repeat_src_offset = repeat_times*cal_once_len
                repeat_dst_offset = repeat_times*cal_once_len//8
                first_src_repeat_offset = repeat_src_offset
                second_src_repeat_offset = repeat_src_offset
                if args[2] is not None:
                    first_src_repeat_offset = 0
                if args[3] is not None:
                    second_src_repeat_offset = 0
                # cmp the input1 and input2
                ir_builder.emit(tvm.call_extern(dst_dtype,
                                                op_cmd,
                                                src_buffers[0].access_ptr(
                                                    "r",
                                                    offset=first_src_repeat_offset),
                                                src_buffers[1].access_ptr(
                                                    "r",
                                                    offset=second_src_repeat_offset),
                                                1, 1, 1, 1, 8, 8, 8))
                # get 16 uint8 cmp result from cmp_mask and
                ir_builder.emit(tvm.call_extern(
                    dst_buffers[0].dtype, "get_cmpmask",
                    dst_buffers[0].access_ptr("wr", offset=repeat_dst_offset)))

            reset_mask_insn(ir_builder, src_dtype)

    return ir_builder.get()

# pylint: disable=too-many-locals
def vec_multiple_sel(tensor_op):
    '''
    vec_multiple_sel
    '''
    src_buffers, dst_buffers = cce_util.get_buffer(tensor_op, True)
    if not src_buffers:
        raise RuntimeError("vec_multiple_sel at least has one src")

    if len(dst_buffers) != 1:
        raise RuntimeError("vec_multiple_sel only has one dst")
    # decl_buffer for clear index in loop
    src_buffers = _get_decl_buffer(src_buffers)
    dst_buffers = _get_decl_buffer(dst_buffers)

    size = cce_util.get_op_lenth(tensor_op)
    condition = src_buffers[0]
    ir_builder = tvm.ir_builder.create()

    block_len = 256
    if len(src_buffers) > 1:
        src_dtype = src_buffers[1].dtype
    else:
        src_dtype = "float16"

    repeat_cal_dtype = src_dtype
    cal_bit_len = cce_util.get_bits_of(repeat_cal_dtype)
    cal_once_len = block_len*8//cal_bit_len

    dst_dtype = dst_buffers[0].dtype
    repeat_times = size//cal_once_len
    remain_len = size - repeat_times*cal_once_len

    # tensor_to_tensor tensor_to_scalar scalar_to_tensor scalar_to_scalar
    sel_type, scalar_value = cce_util.get_sel_type(tensor_op)
    if sel_type == 'tensor_to_scalar':
        scalar_buffer = cce_util.apply_for_new_alloc(ir_builder, src_dtype,
                                                     (128,),
                                                     cce.scope_ubuf, "scalar")
        vec_cmd_factory(ir_builder, "vector_dup", [], [scalar_buffer],
                        cal_once_len, 1,
                        [scalar_value[0]],
                        [1, 1, 8, 8])
        src_buffers.append(scalar_buffer)
    elif sel_type == 'scalar_to_tensor':
        scalar_buffer = cce_util.apply_for_new_alloc(ir_builder, src_dtype,
                                                     (128,),
                                                     cce.scope_ubuf, "scalar")
        vec_cmd_factory(ir_builder, "vector_dup", [], [scalar_buffer],
                        cal_once_len, 1,
                        [scalar_value[0]],
                        [1, 1, 8, 8])
        src_buffers.insert(1, scalar_buffer)
    elif sel_type == 'scalar_to_scalar':
        scalar_buffer0 = cce_util.apply_for_new_alloc(ir_builder, src_dtype,
                                                      (128,),
                                                      cce.scope_ubuf,
                                                      "scalar0")
        vec_cmd_factory(ir_builder, "vector_dup", [], [scalar_buffer0],
                        cal_once_len, 1,
                        [scalar_value[0]],
                        [1, 1, 8, 8])
        scalar_buffer1 = cce_util.apply_for_new_alloc(ir_builder, src_dtype,
                                                      (128,),
                                                      cce.scope_ubuf,
                                                      "scalar1")
        vec_cmd_factory(ir_builder, "vector_dup", [], [scalar_buffer1],
                        cal_once_len, 1,
                        [scalar_value[1]],
                        [1, 1, 8, 8])
        src_buffers.append(scalar_buffer0)
        src_buffers.append(scalar_buffer1)
    # bit mode
    if condition.dtype == "uint8":
        if repeat_times > 0:
            with ir_builder.for_range(0, repeat_times,
                                      name="cmp_index") as cmp_index:
                repeat_condition_offset = cal_once_len*cmp_index//8
                repeat_offset = cal_once_len*cmp_index
                first_src_repeat_offset = repeat_offset
                second_src_repeat_offset = repeat_offset
                if sel_type == 'tensor_to_scalar':
                    second_src_repeat_offset = 0
                elif sel_type == 'scalar_to_tensor':
                    first_src_repeat_offset = 0
                elif sel_type == 'scalar_to_scalar':
                    first_src_repeat_offset = 0
                    second_src_repeat_offset = 0
                # cmp the input1 and input2
                ir_builder.emit(tvm.call_extern("uint8",
                                                "set_cmpmask",
                                                condition.access_ptr(
                                                    "r",
                                                    offset=repeat_condition_offset)))
                ir_builder.emit(tvm.call_extern(dst_dtype,
                                                "vsel",
                                                dst_buffers[0].access_ptr(
                                                    "rw",
                                                    offset=repeat_offset),
                                                src_buffers[1].access_ptr(
                                                    "r",
                                                    offset=first_src_repeat_offset),
                                                src_buffers[2].access_ptr(
                                                    "r",
                                                    offset=second_src_repeat_offset),
                                                1, 1, 1, 1, 8, 8, 8))
        if remain_len > 0:
            reset_mask_insn(ir_builder, src_dtype, bits=remain_len)
            repeat_condition_offset = repeat_times*cal_once_len//8
            repeat_src_offset = repeat_times*cal_once_len
            repeat_dst_offset = repeat_times*cal_once_len
            first_src_repeat_offset = repeat_src_offset
            second_src_repeat_offset = repeat_src_offset
            if sel_type == 'tensor_to_scalar':
                second_src_repeat_offset = 0
            elif sel_type == 'scalar_to_tensor':
                first_src_repeat_offset = 0
            elif sel_type == 'scalar_to_scalar':
                first_src_repeat_offset = 0
                second_src_repeat_offset = 0
            ir_builder.emit(tvm.call_extern("uint8",
                                            "set_cmpmask",
                                            condition.access_ptr(
                                                "r",
                                                offset=repeat_condition_offset)))
            ir_builder.emit(tvm.call_extern(dst_dtype,
                                            "vsel",
                                            dst_buffers[0].access_ptr(
                                                "rw",
                                                offset=repeat_dst_offset),
                                            src_buffers[1].access_ptr(
                                                "r",
                                                offset=first_src_repeat_offset),
                                            src_buffers[2].access_ptr(
                                                "r",
                                                offset=second_src_repeat_offset),
                                            1, 1, 1, 1, 8, 8, 8))
    # bool mode
    else:
        # condition is int8, convert to float16
        fp16_cond_buffer = cce_util.apply_for_new_alloc(ir_builder, "float16",
                                                        condition.shape,
                                                        cce.scope_ubuf,
                                                        "fp16_cond_buffer")
        vec_cmd_factory(ir_builder, 'vconv_s82f16', [condition],
                        [fp16_cond_buffer],
                        size, 1, [],
                        [1, 1, 8, 4],
                        repeat_cal_dtype='fp16')

        # allocate zeros buffer
        zeros_buffer = cce_util.apply_for_new_alloc(ir_builder, "float16", (128,),
                                                    cce.scope_ubuf, "zeros")
        vec_cmd_factory(ir_builder, "vector_dup", [], [zeros_buffer], 128, [],
                        [tvm.const(0.0, "float16")],
                        [1, 1, 8, 8])

        if repeat_times > 0:
            with ir_builder.for_range(0, repeat_times,
                                      name="cmp_index") as cmp_index:
                repeat_offset = cal_once_len*cmp_index
                first_src_repeat_offset = repeat_offset
                second_src_repeat_offset = repeat_offset
                if sel_type == 'tensor_to_scalar':
                    second_src_repeat_offset = 0
                elif sel_type == 'scalar_to_tensor':
                    first_src_repeat_offset = 0
                elif sel_type == 'scalar_to_scalar':
                    first_src_repeat_offset = 0
                    second_src_repeat_offset = 0
                # cmp the input1 and input2
                ir_builder.emit(tvm.call_extern(dst_dtype,
                                                "vcmp_gt",
                                                fp16_cond_buffer.access_ptr(
                                                    "r",
                                                    offset=repeat_offset),
                                                zeros_buffer.access_ptr("r"),
                                                1, 1, 1, 1, 8, 8, 8))

                ir_builder.emit(tvm.call_extern(dst_dtype,
                                                "vsel",
                                                dst_buffers[0].access_ptr(
                                                    "rw",
                                                    offset=repeat_offset),
                                                src_buffers[1].access_ptr(
                                                    "r",
                                                    offset=first_src_repeat_offset),
                                                src_buffers[2].access_ptr(
                                                    "r",
                                                    offset=second_src_repeat_offset),
                                                1, 1, 1, 1, 8, 8, 8))

        if remain_len > 0:
            repeat_src_offset = repeat_times*cal_once_len
            repeat_dst_offset = repeat_times*cal_once_len
            first_src_repeat_offset = repeat_src_offset
            second_src_repeat_offset = repeat_src_offset
            if sel_type == 'tensor_to_scalar':
                second_src_repeat_offset = 0
            elif sel_type == 'scalar_to_tensor':
                first_src_repeat_offset = 0
            elif sel_type == 'scalar_to_scalar':
                first_src_repeat_offset = 0
                second_src_repeat_offset = 0
            # cmp the input1 and input2
            reset_mask_insn(ir_builder, dst_dtype, bits=remain_len)
            ir_builder.emit(tvm.call_extern(dst_dtype,
                                            "vcmp_gt",
                                            fp16_cond_buffer.access_ptr(
                                                "r",
                                                offset=repeat_src_offset),
                                            zeros_buffer.access_ptr("r"),
                                            1, 1, 1, 1, 8, 8, 8))

            ir_builder.emit(tvm.call_extern(dst_dtype,
                                            "vsel",
                                            dst_buffers[0].access_ptr(
                                                "rw",
                                                offset=repeat_dst_offset),
                                            src_buffers[1].access_ptr(
                                                "r",
                                                offset=first_src_repeat_offset),
                                            src_buffers[2].access_ptr(
                                                "r",
                                                offset=second_src_repeat_offset),
                                            1, 1, 1, 1, 8, 8, 8))

    reset_mask_insn(ir_builder, dst_dtype)

    return ir_builder.get()

# pylint: disable=too-many-locals
# intrin for pooling2d start
@tvm.register_func("tvm.intrin.cce.pooling2d_process")
def pooling2d_process(op_expr):
    """
    :param op_expr: the stmt of for with pooling2d_process(max/sum)
    :return: stmt
    """

    ins, outs, _, _ = cce_util.get_dma_buffer(op_expr)

    pooling_mode = get_emitinsn_params("pooling_mode")

    c1_value = get_emitinsn_params("c1_value")
    window_h = get_emitinsn_params("window_h")
    window_w = get_emitinsn_params("window_w")
    c_block_size = get_emitinsn_params("c_block_size")

    out_size_h = get_emitinsn_params("out_size_h")
    out_size_w = get_emitinsn_params("out_size_w")
    fmap_img2col_setps_cuth = get_emitinsn_params("fmap_img2col_setps_cuth")
    l1_cut_to_ub_factor = get_emitinsn_params("l1_cut_to_ub_factor")

    res_cut_factor = get_emitinsn_params("res_cut_factor")
    need_cut_c1 = get_emitinsn_params("need_cut_c1")
    cuth_loop = get_emitinsn_params("cuth_loop")
    is_cut_l1_to_ub = get_emitinsn_params("is_cut_l1_to_ub")

    fmap_img2col_cuth = l1_cut_to_ub_factor*16

    ib_expr = tvm.ir_builder.create()

    if len(ins) <= 1:
        stmt = ib_expr.get()
        return stmt

    src = ins[1]
    src_shape = [src.shape[j].value for j in range(len(src.shape))]

    src_buffer = tvm.decl_buffer(src_shape,
                                 src.dtype,
                                 name=src.name,
                                 data=src.data,
                                 offset_factor=src.offset_factor,
                                 data_alignment=src.data_alignment,
                                 scope=cce_util.get_buf_scope(src.name),
                                 elem_offset=0)

    dst_buffer = outs[0]

    if pooling_mode == "MAX":
        pooling_intrin = 'vmax'
        dump_value = tvm.const(-65504.0, dtype="float16")
    else:
        dump_value = tvm.const(0.0, dtype="float16")
        pooling_intrin = 'vadd'

    fmap_img2col_m = l1_cut_to_ub_factor*c_block_size

    c1_range = 1
    loop_extent = []
    def _post_order(stmt_in):
        if isinstance(stmt_in, tvm.stmt.For):
            if stmt_in.loop_var.name == "k":
                loop_extent.append(stmt_in.extent.value)

    _ = tvm.ir_pass.IRTransform(op_expr, None, _post_order, ["For"])
    if loop_extent:
        c1_range = loop_extent[0]

    repeat = (fmap_img2col_m*c1_range*c_block_size+128 - 1)//128

    dst_stride_m0 = 1
    src_stride_m0 = 1
    dst_stride_m1 = 8
    src_stride_m1 = 8

    is_tile = bool(op_expr.extent.value < l1_cut_to_ub_factor)
    if is_tile:
        if is_cut_l1_to_ub:
            remained_tile_size = ((out_size_h*out_size_w + 15)//16*16%
                                  (fmap_img2col_setps_cuth*out_size_w))

            elem_offset = (remained_tile_size//c_block_size//l1_cut_to_ub_factor) \
                         *c_block_size*l1_cut_to_ub_factor*c_block_size*c1_range
            repeat = (remained_tile_size*c_block_size*c1_range - elem_offset)//128
        else:
            burst_len_pre = res_cut_factor*(cuth_loop - 1)
            reamin_out_rows = (out_size_h*out_size_w - burst_len_pre*c_block_size +
                               c_block_size - 1)//c_block_size*c_block_size
            elem_offset = 0
            repeat = reamin_out_rows*c_block_size*c1_range//128

        dst_buffer = tvm.decl_buffer((repeat*128,), dst_buffer.dtype,
                                     name=dst_buffer.name, data=dst_buffer.data,
                                     offset_factor=dst_buffer.offset_factor,
                                     data_alignment=dst_buffer.data_alignment,
                                     scope=cce_util.get_buf_scope(dst_buffer.name),
                                     elem_offset=elem_offset)

    # if not is_tile:
    dst_addr = dst_buffer.access_ptr('w', offset=0)

    if 0 < repeat <= 255:
        ib_expr.emit(tvm.call_extern(dst_buffer.dtype, 'vector_dup', dst_addr,
                                     dump_value, repeat, dst_stride_m0,
                                     src_stride_m0, dst_stride_m1,
                                     src_stride_m1))
    elif repeat > 255:
        repeat_loop_dump = repeat//255
        repeat_loop_count = 255
        repeat_left_count = repeat%255

        for i in range(repeat_loop_dump):
            dst_addr_loop = dst_buffer.access_ptr('w', offset=i*255*128)
            ib_expr.emit(
                tvm.call_extern(dst_buffer.dtype, 'vector_dup', dst_addr_loop,
                                dump_value, repeat_loop_count, dst_stride_m0, src_stride_m0,
                                dst_stride_m1, src_stride_m1))

        dst_addr_left = dst_buffer.access_ptr('w',
                                              offset=repeat_loop_dump*255*128)
        ib_expr.emit(
            tvm.call_extern(dst_buffer.dtype, 'vector_dup', dst_addr_left,
                            dump_value, repeat_left_count, dst_stride_m0, src_stride_m0,
                            dst_stride_m1, src_stride_m1))

    else:
        raise RuntimeError("invalid repeat params, it must be >= 1")

    # params for vmax of pooling2d_max
    fracz_size = c_block_size*c_block_size
    loop_outer_h = (fmap_img2col_cuth+c_block_size - 1)//c_block_size
    # half fracZ size 128 fp16 data vmax with vector_dup each time
    loop_inner_w = window_h*window_w

    with ib_expr.new_scope():
        # loop in W direction of fmap_img2col_ub
        with ib_expr.for_range(0, loop_inner_w,
                               name="loop_inner_w") as _loop_inner_w:
            if loop_inner_w < c_block_size:
                # src1_stride_m1 is uint8_t, so the max value is 255.
                # when loop_inner_w > 15, the src1_stride_m1 value will greater than 255.
                # src1 is address of fmap_img2col_ub
                offset_src1 = fracz_size*_loop_inner_w
                dst_addr = dst_buffer.access_ptr('w')
                src0_addr = dst_buffer.access_ptr('r')
                src1_addr = src_buffer.access_ptr('r', offset=offset_src1)

                if is_tile:
                    repeat = repeat//2
                else:
                    if need_cut_c1:
                        repeat = c1_range*loop_outer_h
                    else:
                        repeat = c1_value*loop_outer_h
                dst_stride_m0 = 1
                src0_stride_m0 = 1
                src1_stride_m0 = 1
                dst_stride_m1 = fracz_size//c_block_size
                src0_stride_m1 = fracz_size//c_block_size
                src1_stride_m1 = fracz_size*loop_inner_w//c_block_size

                # emit vmax/vadd for the first half 128 fp16 data of each fracZ
                ib_expr.emit(
                    tvm.call_extern(dst_buffer.dtype, pooling_intrin, dst_addr,
                                    src0_addr, src1_addr, repeat, dst_stride_m0,
                                    src0_stride_m0, src1_stride_m0, dst_stride_m1,
                                    src0_stride_m1, src1_stride_m1))

                # emit vmax/vadd for the second half 128 data of each fracZ
                dst_addr = dst_buffer.access_ptr('w', offset=fracz_size//2)
                src0_addr = dst_buffer.access_ptr('r', offset=fracz_size//2)
                src1_addr = src_buffer.access_ptr('r',
                                                  offset=offset_src1+fracz_size//2)
                ib_expr.emit(
                    tvm.call_extern(dst_buffer.dtype, pooling_intrin, dst_addr,
                                    src0_addr, src1_addr, repeat, dst_stride_m0,
                                    src0_stride_m0, src1_stride_m0, dst_stride_m1,
                                    src0_stride_m1, src1_stride_m1))
            else:
                dst_stride_m0 = 1
                src0_stride_m0 = 1
                src1_stride_m0 = 1
                dst_stride_m1 = 8
                src0_stride_m1 = 8
                src1_stride_m1 = 8
                # when loop_inner_w > 15, the src1_stride_m1 value will greater than 255.
                # only compute 256 num one time.

                h_factor = l1_cut_to_ub_factor
                if is_tile:
                    h_factor = repeat//2//c1_range

                repeat = 2

                for ci_item in range(c1_range):
                    for hi_item in range(h_factor):
                        offset_src1 = hi_item*c1_range*fracz_size*loop_inner_w \
                                     +ci_item*fracz_size*loop_inner_w \
                                     +fracz_size*_loop_inner_w
                        offset_dst = hi_item*c1_range*fracz_size+ci_item*fracz_size
                        offset_src0 = offset_dst
                        dst_addr = dst_buffer.access_ptr('w', offset=offset_dst)
                        src0_addr = dst_buffer.access_ptr('r',
                                                          offset=offset_src0)
                        src1_addr = src_buffer.access_ptr('r',
                                                          offset=offset_src1)
                        # emit vmax for the first half 128 fp16 data of each fracZ
                        ib_expr.emit(
                            tvm.call_extern(dst_buffer.dtype, pooling_intrin,
                                            dst_addr, src0_addr, src1_addr, repeat,
                                            dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                            dst_stride_m1, src0_stride_m1, src1_stride_m1))

    stmt = ib_expr.get()

    return stmt

# pylint: disable=too-many-locals
@tvm.register_func("tvm.intrin.cce.pooling2d_avg_mul_factor")
def pooling2d_avg_mul_factor(op_expr, times, loop_extent):
    """
    :param op_expr: the stmt of for with pooling2d_avg_mul_factor
    :return: stmt
    """

    if times == 0:
        AvgPoolingParam.factor_vector = None

    is_cut_l1_to_ub = get_emitinsn_params("is_cut_l1_to_ub")
    ins, outs, _, _ = cce_util.get_dma_buffer(op_expr)

    ib_expr = tvm.ir_builder.create()

    if not ins:
        stmt = ib_expr.get()
        return stmt

    pooling_mode = get_emitinsn_params("pooling_mode")
    padding_mode = get_emitinsn_params("padding_mode")

    c_block_size = get_emitinsn_params("c_block_size")

    pad_top = get_emitinsn_params("pad_top")
    pad_bottom = get_emitinsn_params("pad_bottom")
    pad_left = get_emitinsn_params("pad_left")
    pad_right = get_emitinsn_params("pad_right")

    cut_flag = get_emitinsn_params("cut_flag")
    cuth_loop = get_emitinsn_params("cuth_loop")
    res_cut_factor = get_emitinsn_params("res_cut_factor")

    out_size_h = get_emitinsn_params("out_size_h")
    out_size_w = get_emitinsn_params("out_size_w")

    src = ins[0]
    src_buffer = tvm.decl_buffer(src.shape, src.dtype, name=src.name,
                                 data=src.data, offset_factor=src.offset_factor,
                                 data_alignment=src.data_alignment,
                                 scope=cce_util.get_buf_scope(src.name),
                                 elem_offset=0)

    dst = outs[0]
    dst_buffer = tvm.decl_buffer(dst.shape, dst.dtype, name=dst.name,
                                 data=dst.data, offset_factor=dst.offset_factor,
                                 data_alignment=dst.data_alignment,
                                 scope=cce_util.get_buf_scope(dst.name),
                                 elem_offset=0)

    out_var = None
    base_var = 0

    # get out loop var and offset
    if isinstance(op_expr, tvm.stmt.AttrStmt):
        if isinstance(op_expr.value, tvm.expr.Var):
            out_var = op_expr.value
        elif isinstance(op_expr.value, tvm.expr.IntImm):
            base_var = op_expr.value.value
            out_var = op_expr.value.value
        elif isinstance(op_expr.value, tvm.expr.Add):
            if isinstance(op_expr.value.a, tvm.expr.Var):
                out_var = op_expr.value.a
                base_var = op_expr.value.b.value
            elif isinstance(op_expr.value.b, tvm.expr.Var):
                out_var = op_expr.value.b
                base_var = op_expr.value.a.value
        else:
            out_var = 0
    else:
        out_var = 0

    current_offset = base_var*res_cut_factor*c_block_size

    inner_extend = op_expr.body.extent.value

    is_tile = False
    if inner_extend < res_cut_factor or cut_flag == "NO_CUT":
        is_tile = True
    elif (out_size_h*out_size_w*c_block_size - current_offset) \
            < res_cut_factor*c_block_size*c_block_size:
        is_tile = True

    current_length = res_cut_factor*c_block_size

    if is_tile:
        burst_len_pre = res_cut_factor*(cuth_loop - 1)
        reamin_out_rows = out_size_h*out_size_w - burst_len_pre*c_block_size
        current_length = reamin_out_rows

    if AvgPoolingParam.factor_vector is None:
        mean_factor_vector = get_mean_factor_vector()
        if mean_factor_vector:
            AvgPoolingParam.factor_vector = mean_factor_vector
        else:
            raise RuntimeError("area mean factor vector is empty.")

    avg_factor_shape = (
        ((res_cut_factor*c_block_size*c_block_size+128 - 1)//128)*128+128,)
    avg_factor_ub = apply_for_new_alloc(ib_expr, dst.dtype, avg_factor_shape,
                                        scope=cce.scope_ubuf)

    def pooling_ib_emit(ib_expr, index):
        """
        descripe:the function to dump avg factor
        """
        start_index = current_offset+index*current_length
        end_index = start_index+current_length
        # pylint: disable=unsubscriptable-object
        values = AvgPoolingParam.factor_vector[start_index: end_index]
        value, count, offset = get_vector_count_and_offset(values)

        pooling_avg_factor_vector_dump(ib_expr, value, count, offset,
                                       avg_factor_ub, c_block_size)

    def pooling_ib_gen(ib_expr, start_idx, out_var_range):
        """
        descripe:the function to generate IR using ir_builder
        """
        # if only one loop extent,the outer_axis is Null and no need to add if stmt
        # the limit of the nesting depth in the generated CCE,
        # in case nesting depth  > MAX_BRACKET_DEPTH,
        # stack will overflow, so in this case we generated cce with non-nesting
        if (out_var_range - start_idx) > MAX_BRACKET_DEPTH or out_var_range == 1:
            unrecurisive_pooling_ib_gen(ib_expr, out_var_range)
        else:
            recursive_pooling_ib_gen(ib_expr, start_idx, out_var_range)

    def unrecurisive_pooling_ib_gen(ib_expr, out_var_range):
        """
        non_recurisive situation to generate only if statement with none-nesting
        """
        if out_var_range == 1:
            pooling_ib_emit(ib_expr, 0)
        else:
            # change the stmt from recursion to non-recursion
            for index in range(out_var_range):
                with ib_expr.if_scope(out_var == index):
                    pooling_ib_emit(ib_expr, index)

    def recursive_pooling_ib_gen(ib_expr, start_idx, out_var_range):
        """
        recurisive situation to generate embed if-else with nesting,
        to decrease the scalar operation
        """
        # start_idx is the value of the pooling_id the embed if_else has counted to
        with ib_expr.if_scope(out_var == start_idx):
            pooling_ib_emit(ib_expr, start_idx)
            if start_idx == out_var_range - 1:
                return

        recursive_pooling_ib_gen(ib_expr, start_idx+1, out_var_range)

    if isinstance(out_var, tvm.expr.Var):
        if is_cut_l1_to_ub:
            dump_factor_map = {}
            for i in range(loop_extent):
                start_index = current_offset
                end_index = current_offset+current_length
                # pylint: disable=unsubscriptable-object
                values = AvgPoolingParam.factor_vector[start_index: end_index]
                value, count, _ = get_vector_count_and_offset(values)
                key = "value_" + "_".join([str(v) for v in value]) + "_count_" + \
                      "_".join([str(ci_item) for ci_item in count])
                if key not in dump_factor_map:
                    dump_factor_map[key] = []

                dump_factor_map[key].append(i)

            if len(dump_factor_map) == 1:
                # values = AvgPoolingParam.factor_vector[current_offset:
                # current_offset+current_length]
                # value, count, offset = get_vector_count_and_offset(values)
                # pooling_avg_factor_vector_dump(ib_expr, value, count, offset,
                # avg_factor_ub, c_block_size)
                pooling_ib_emit(ib_expr, 0)
            else:
                index_len_0_key_list = []
                index_len_gt_0_key_list = []
                for key in dump_factor_map:
                    index_list = dump_factor_map[key]
                    if len(index_list) == 1:
                        index_len_0_key_list.append(key)
                    else:
                        index_len_gt_0_key_list.append(key)

                for key in index_len_0_key_list:
                    index_list = dump_factor_map[key]
                    with ib_expr.if_scope(out_var == index_list[0]):
                        pooling_ib_emit(ib_expr, index_list[0])

                if index_len_gt_0_key_list:
                    with ib_expr.else_scope():
                        pooling_ib_emit(ib_expr, index_len_gt_0_key_list[0][0])
        else:
            if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
                pooling_ib_emit(ib_expr, 0)
            elif loop_extent > MAX_BRACKET_DEPTH:
                factor_dump_times = loop_extent//MAX_BRACKET_DEPTH
                for i in range(factor_dump_times):
                    pooling_ib_gen(ib_expr, i*MAX_BRACKET_DEPTH,
                                   (i+1)*MAX_BRACKET_DEPTH)

                if loop_extent%MAX_BRACKET_DEPTH != 0:
                    pooling_ib_gen(ib_expr, factor_dump_times*MAX_BRACKET_DEPTH,
                                   loop_extent)
            else:
                pooling_ib_gen(ib_expr, 0, loop_extent)
    else:
        # values = AvgPoolingParam.factor_vector[current_offset :
        # current_offset+current_length]
        # value, count, offset = get_vector_count_and_offset(values)
        # pooling_avg_factor_vector_dump(ib_expr, value, count, offset,
        # avg_factor_ub, c_block_size)
        pooling_ib_emit(ib_expr, 0)

    # params for vmul of pooling2d_avg_mul_factor
    fracz_size = c_block_size*c_block_size
    loop_outer_h = (current_length+c_block_size - 1)//c_block_size

    loop_outer_w = 1
    loop_extent = []

    def _post_order(stmt_in):
        if isinstance(stmt_in, tvm.stmt.For):
            if stmt_in.loop_var.name == "i2":
                loop_extent.append(stmt_in.extent.value)

    _ = tvm.ir_pass.IRTransform(op_expr, None, _post_order, ["For"])

    if loop_extent:
        loop_outer_w = loop_extent[0]

    # SAME MODE : avg_factor is a vector, avg_factor = window_h*window_w - sizeof(pad)
    # half fracZ size 128 fp16 data vmul with vector_dup each time
    if pooling_mode == "AVG" and padding_mode == "SAME":
        # execute for pooling_avg = avg_sum / avg_factor[i] in H direction
        # c1_value avg sum data in zZ format, compute mean value for each fracZ block
        # src0 is address of avg sum, dst address is the same as src0
        with ib_expr.new_scope():
            with ib_expr.for_range(0, loop_outer_h,
                                   name="loop_avg_H") as _loop_avg_h:
                with ib_expr.for_range(0, loop_outer_w,
                                       name="loop_avg_Ci") as _loop_avg_ci:
                    src0_offset = _loop_avg_h*loop_outer_w*fracz_size \
                                 +_loop_avg_ci*fracz_size
                    dst_offset = _loop_avg_h*loop_outer_w*fracz_size \
                                +_loop_avg_ci*fracz_size
                    src0_addr = src_buffer.access_ptr('r', offset=src0_offset)
                    dst_addr = dst_buffer.access_ptr('w', offset=dst_offset)

                    # vmul for the fisrt half 128 fp16 data of each fracZ
                    # src1 is the address of avg_factor_ub
                    src1_offset = _loop_avg_h*fracz_size
                    src1_addr = avg_factor_ub.access_ptr('r',
                                                         offset=src1_offset)
                    repeat = fracz_size//128
                    dst_stride_m0 = 1
                    src0_stride_m0 = 1
                    src1_stride_m0 = 1
                    dst_stride_m1 = 8
                    src0_stride_m1 = 8
                    src1_stride_m1 = 8

                    ib_expr.emit(
                        tvm.call_extern(dst.dtype, 'vmul', dst_addr, src0_addr,
                                        src1_addr, repeat, dst_stride_m0, src0_stride_m0,
                                        src1_stride_m0, dst_stride_m1, src0_stride_m1,
                                        src1_stride_m1))

    stmt = ib_expr.get()
    return stmt


def get_vector_count_and_offset(avg_mean_factor):
    """
    :params:
    :avg_mean_factor: avg mean area factors
    :return:
    :continuous avg factor count and it's offset from the beginning
    """
    metadata = 0
    counter_end = 0
    is_first_time_flag = True

    value = []
    offset = []
    count = []

    # compute value, count, offset list
    for i, _ in enumerate(avg_mean_factor):
        if is_first_time_flag:
            metadata = avg_mean_factor[0]
            is_first_time_flag = False

        current_data = avg_mean_factor[i]

        if current_data == metadata:
            counter_end = counter_end+1

        else:
            value.append(metadata)
            offset.append(i - counter_end)
            count.append(counter_end)

            metadata = current_data
            counter_end = 1

    value.append(metadata)
    offset.append(len(avg_mean_factor) - counter_end)
    count.append(counter_end)

    return value, count, offset


class AvgPoolingParam():
    """
    : pooling avg factor for SAME mode
    """

    def __init__(self):
        pass

    factor_vector = None

# pylint: disable=too-many-locals
def get_mean_factor_vector():
    """
    :avg pooling mean_factor_vector
    :return:
    :avg_mean_factor_vector
    """
    data_mode = get_emitinsn_params("data_mode")

    if data_mode == 0:
        return get_caffe_mean_factor_vector()
    if data_mode == 1:
        return get_tensorflow_mean_factor_vector()
    raise RuntimeError("data mode only support 0:CAFFE or 1:TENSORFLOW.")


# pylint: disable=too-many-locals
def get_tensorflow_mean_factor_vector():
    """
    :avg pooling mean_factor_vector in tensorflow
    :return:
    :avg_mean_factor_vector
    """
    window_h = get_emitinsn_params("window_h")
    window_w = get_emitinsn_params("window_w")

    in_size_h = get_emitinsn_params("in_size_h")
    in_size_w = get_emitinsn_params("in_size_w")

    stride_h = get_emitinsn_params("stride_h")
    stride_w = get_emitinsn_params("stride_w")

    pad_top = get_emitinsn_params("pad_top")
    pad_left = get_emitinsn_params("pad_left")

    out_size_h = get_emitinsn_params("out_size_h")
    out_size_w = get_emitinsn_params("out_size_w")

    avg_mean_factor_vector = []

    for steps_h in range(out_size_h):
        for steps_w in range(out_size_w):
            h_start = steps_h*stride_h - pad_top
            w_start = steps_w*stride_w - pad_left
            h_end = min(h_start+window_h, in_size_h)
            w_end = min(w_start+window_w, in_size_w)
            h_start = max(h_start, 0)
            w_start = max(w_start, 0)
            area = max((h_end - h_start)*(w_end - w_start), 1)
            mean_value = 1 / float(area)
            avg_mean_factor_vector.append(mean_value)

    return avg_mean_factor_vector


# pylint: disable=too-many-locals
def get_caffe_mean_factor_vector():
    """
    :avg pooling mean_factor_vector in caffe
    :return:
    :avg_mean_factor_vector
    """
    window_h = get_emitinsn_params("window_h")
    window_w = get_emitinsn_params("window_w")

    in_size_h = get_emitinsn_params("in_size_h")
    in_size_w = get_emitinsn_params("in_size_w")

    stride_h = get_emitinsn_params("stride_h")
    stride_w = get_emitinsn_params("stride_w")

    pad_top = get_emitinsn_params("pad_top")
    pad_left = get_emitinsn_params("pad_left")

    out_size_h = get_emitinsn_params("out_size_h")
    out_size_w = get_emitinsn_params("out_size_w")

    avg_mean_factor_vector = []

    for steps_h in range(out_size_h):
        for steps_w in range(out_size_w):
            # Like caffe src
            h_start = steps_h * stride_h - pad_top
            w_start = steps_w * stride_w - pad_left
            h_end = min(h_start + window_h, in_size_h + pad_top)
            w_end = min(w_start + window_w, in_size_w + pad_left)
            area = max((h_end - h_start) * (w_end - w_start), 1)

            mean_value = 1 / float(area)
            avg_mean_factor_vector.append(mean_value)

    return avg_mean_factor_vector


def pooling_avg_factor_vector_dump(ib_expr, value, count, offset, avg_factor_ub,
                                   c_block_size):
    """
    :params:
    :ib_expr: ib
    :value: avg factor value
    :count: count of the same continuous avg factor
    :offset: dump offset of avg factor from the beginning
    :avg_factor_ub: area factor in ub
    :c_block_size: c0
    """
    for i, _ in enumerate(value):
        dst_offset = offset[i]*c_block_size
        dst_addr = avg_factor_ub.access_ptr('w', offset=dst_offset)
        avg_scalar = tvm.const(value[i], dtype="float16")
        repeat = (count[i]*c_block_size+128 - 1)//128

        if 0 < repeat <= 255:
            ib_expr.emit(
                tvm.call_extern("float16", 'vector_dup', dst_addr, avg_scalar,
                                repeat, 1, 1, 8, 8))
        elif repeat > 255:
            avg_factor_dump_loop = repeat//255
            repeat_loop = 255
            repeat_res = repeat%255

            for idx in range(avg_factor_dump_loop):
                dst_addr_loop = avg_factor_ub.access_ptr('w',
                                                         offset=dst_offset+idx*255*128)
                ib_expr.emit(
                    tvm.call_extern("float16", 'vector_dup', dst_addr_loop,
                                    avg_scalar, repeat_loop, 1, 1, 8, 8))

            dst_addr_loop_res = avg_factor_ub.access_ptr('w',
                                                         offset=dst_offset \
                                                               +avg_factor_dump_loop*255*128)
            ib_expr.emit(
                tvm.call_extern("float16", 'vector_dup', dst_addr_loop_res,
                                avg_scalar, repeat_res, 1, 1, 8, 8))

        else:
            raise RuntimeError("invalid repeat params, it must be >= 1")


# pylint: disable=too-many-locals
@tvm.register_func("tvm.intrin.cce.pooling2d_global_process")
def pooling2d_global_process(op_expr):
    """
    :param op_expr: the stmt of for with pooling2d_global_process(max/sum)
    :return: stmt
    """

    ins, outs, _, _ = cce_util.get_dma_buffer(op_expr)

    pooling_mode = get_emitinsn_params("pooling_mode")
    c1_value = get_emitinsn_params("c1_value")
    c_block_size = get_emitinsn_params("c_block_size")
    cut_ci_factor = get_emitinsn_params("cut_ci_factor")
    cut_hi_factor = get_emitinsn_params("cut_hi_factor")
    cut_wi_factor = get_emitinsn_params("cut_wi_factor")
    cut_flag = get_emitinsn_params("cut_flag")

    ib_expr = tvm.ir_builder.create()

    # Get vadd vmul vconv ability
    vconv_ability = cce_conf.intrinsic_check_support("Intrinsic_vconv",
                                                     "f162f32")
    vadd_ability = cce_conf.intrinsic_check_support("Intrinsic_vadd",
                                                    "float32")
    vmul_ability = cce_conf.intrinsic_check_support("Intrinsic_vmul",
                                                    "float32")
    fp32_ability = vconv_ability and\
                   vadd_ability and\
                   vmul_ability and \
                   (not cce_conf.get_soc_spec("SOC_VERSION") in ["Ascend310"])
    if ins:
        src = ins[1]
        src_buffer = tvm.decl_buffer(src.shape, src.dtype, name=src.name,
                                     data=src.data,
                                     offset_factor=src.offset_factor,
                                     data_alignment=src.data_alignment,
                                     scope=cce_util.get_buf_scope(src.name),
                                     elem_offset=0)

    dst = outs[0]

    dst_buffer = tvm.decl_buffer(dst.shape, dst.dtype, name=dst.name,
                                 data=dst.data, offset_factor=dst.offset_factor,
                                 data_alignment=dst.data_alignment,
                                 scope=cce_util.get_buf_scope(dst.name),
                                 elem_offset=0)

    is_reg_mov = []

    def _post_order(op_expr):
        if isinstance(op_expr,
                      tvm.stmt.AttrStmt) and op_expr.attr_key == "pragma_emit_insn":
            is_reg_mov.append(op_expr.value)
            return op_expr.body
        return None

    _ = tvm.ir_pass.IRTransform(op_expr, None, _post_order, ["AttrStmt"])

    # get repeat data size of each compute loop
    # get extent value of each compute loop that for rq copy ub to gm
    var_extent_list = []
    var_extent_mul = [1]

    def _post_order_for(op_expr):
        if isinstance(op_expr, tvm.stmt.For):
            var_extent_mul[0] = var_extent_mul[0]*op_expr.extent.value
            var_extent_list.append(op_expr.extent.value)

    op_expr = tvm.ir_pass.IRTransform(op_expr, None, _post_order_for, ["For"])

    # check whether real_cutci_current_loop is 1, get real_cutci_current_loop
    if len(var_extent_list) == 1 or c1_value == 1 or cut_ci_factor == 1:
        real_cutci_current_loop = 1
    else:
        real_cutci_current_loop = var_extent_list[-1]

    if pooling_mode == "GMP":
        pooling_intrin = 'vmax'
        dump_value = tvm.const(-65504.0, dtype="float16")

    if pooling_mode == "GAP" and fp32_ability:
        dump_value = tvm.const(0.0, dtype="float32")
        pooling_intrin = 'vadd'
    elif pooling_mode == "GAP":
        dump_value = tvm.const(0.0, dtype="float16")
        pooling_intrin = 'vadd'

    is_mini_or_lhisi = (cce_conf.get_soc_spec("SOC_VERSION") in \
                        ["Ascend310", "Hi3796CV300ES"])

    if is_reg_mov:
        if isinstance(is_reg_mov[0], tvm.expr.StringImm) and\
            is_reg_mov[0].value == "reg_mov":
            # instead reg_mov to be vector_dup here.
            dst_stride_m0 = 1
            src_stride_m0 = 1
            dst_stride_m1 = 8
            src_stride_m1 = 8

            dst_addr = dst_buffer.access_ptr('w', offset=0)
            # Under float16 mode, vector_dup can process 128 data per repeat
            instruction_max = 128
            if pooling_mode == "GAP" and fp32_ability:
                instruction_max = 64
            data_size = real_cutci_current_loop*c_block_size
            repeat = (data_size+instruction_max - 1)//instruction_max
            tail_dump_size = data_size%instruction_max

            if tail_dump_size != 0:
                dump_bits = tail_dump_size
            else:
                dump_bits = instruction_max

            if repeat <= 0:
                raise RuntimeError("invalid repeat params, it must be >= 1")

            # two cases includ data_size < 128 or = 128, reset mask to dump_bits.
            if repeat == 1:
                reset_mask_insn(ib_expr, dst.dtype, bits=dump_bits)
                ib_expr.emit(
                    tvm.call_extern(dst_buffer.dtype, 'vector_dup', dst_addr,
                                    dump_value, repeat,
                                    dst_stride_m0, src_stride_m0,
                                    dst_stride_m1, src_stride_m1))

            elif 1 < repeat <= 255:
                # no tail, loop repeat times, each repeat is 128 fp16 numbers.
                if tail_dump_size == 0:
                    reset_mask_insn(ib_expr, dst.dtype, bits=instruction_max)
                    ib_expr.emit(tvm.call_extern(dst_buffer.dtype, 'vector_dup',
                                                 dst_addr, dump_value,
                                                 repeat, dst_stride_m0,
                                                 src_stride_m0,
                                                 dst_stride_m1, src_stride_m1))
                else:
                    # main dump size, loop (repeat - 1) times,
                    # each repeat is 128 fp16 numbers.
                    repeat_main = repeat - 1
                    reset_mask_insn(ib_expr, dst.dtype, bits=instruction_max)
                    ib_expr.emit(tvm.call_extern(dst_buffer.dtype, 'vector_dup',
                                                 dst_addr, dump_value,
                                                 repeat_main, dst_stride_m0,
                                                 src_stride_m0,
                                                 dst_stride_m1, src_stride_m1))

                    # tail dump size, only repeat = 1 for tail bits.
                    dst_offset = (repeat - 1)*instruction_max
                    dst_addr = dst_buffer.access_ptr('w', offset=dst_offset)
                    repeat_tail = 1
                    reset_mask_insn(ib_expr, dst.dtype, bits=dump_bits)
                    ib_expr.emit(tvm.call_extern(dst_buffer.dtype, 'vector_dup',
                                                 dst_addr, dump_value,
                                                 repeat_tail, dst_stride_m0,
                                                 src_stride_m0,
                                                 dst_stride_m1, src_stride_m1))

            elif repeat > 255:
                repeat_loop_dump = repeat//255
                repeat_loop_count = 255
                repeat_left_count = repeat%255

                reset_mask_insn(ib_expr, dst.dtype, bits=instruction_max)
                for i in range(repeat_loop_dump):
                    dst_addr_loop = \
                        dst_buffer.access_ptr(
                            'w', offset=i*255*instruction_max)
                    ib_expr.emit(tvm.call_extern(dst_buffer.dtype, 'vector_dup',
                                                 dst_addr_loop, dump_value,
                                                 repeat_loop_count,
                                                 dst_stride_m0,
                                                 src_stride_m0, dst_stride_m1,
                                                 src_stride_m1))

                if tail_dump_size == 0:
                    dst_addr_left = dst_buffer.access_ptr(
                        'w', offset=repeat_loop_dump*255*instruction_max)
                    reset_mask_insn(ib_expr, dst.dtype, bits=instruction_max)
                    ib_expr.emit(tvm.call_extern(dst_buffer.dtype, 'vector_dup',
                                                 dst_addr_left, dump_value,
                                                 repeat_left_count,
                                                 dst_stride_m0,
                                                 src_stride_m0, dst_stride_m1,
                                                 src_stride_m1))
                else:
                    dst_addr_left = dst_buffer.access_ptr(
                        'w', offset=repeat_loop_dump*255*instruction_max)
                    reset_mask_insn(ib_expr, dst.dtype, bits=instruction_max)
                    ib_expr.emit(tvm.call_extern(dst_buffer.dtype, 'vector_dup',
                                                 dst_addr_left, dump_value,
                                                 repeat_left_count - 1,
                                                 dst_stride_m0,
                                                 src_stride_m0, dst_stride_m1,
                                                 src_stride_m1))

                    dst_addr_left = \
                        dst_buffer.access_ptr(
                            'w', offset=(repeat - 1)*instruction_max)
                    repeat_tail = 1
                    reset_mask_insn(ib_expr, dst.dtype, bits=dump_bits)
                    ib_expr.emit(tvm.call_extern(dst_buffer.dtype, 'vector_dup',
                                                 dst_addr_left, dump_value,
                                                 repeat_tail,
                                                 dst_stride_m0,
                                                 src_stride_m0,
                                                 dst_stride_m1, src_stride_m1))
    else:
        with ib_expr.new_scope():
            with ib_expr.for_range(0, real_cutci_current_loop,
                                   name="cut_ci_factor") as _loop_cut_ci:
                # process_tensor is tmp workspace used for vadd or vmax
                if pooling_mode == "GAP" and fp32_ability:
                    num_per_repeat = 64
                    process_shape = (num_per_repeat,)
                else:
                    num_per_repeat = 128
                    process_shape = (num_per_repeat,)
                process_tensor = apply_for_new_alloc(ib_expr, src.dtype,
                                                     process_shape,
                                                     scope=cce.scope_ubuf)

                dst_stride_m0 = 1
                src_stride_m0 = 1
                dst_stride_m1 = 8
                src_stride_m1 = 8

                dst_addr = process_tensor.access_ptr('w', offset=0)
                repeat = 1

                # reset mask for vadd or vmax to -1 and
                # vector_dup 128 fp16 area.
                reset_mask_insn(ib_expr, dst.dtype, bits=num_per_repeat)
                ib_expr.emit(tvm.call_extern(dst_buffer.dtype, 'vector_dup',
                                             dst_addr, dump_value,
                                             repeat, dst_stride_m0,
                                             src_stride_m0,
                                             dst_stride_m1, src_stride_m1))

                # vmax or vadd with process_tensor and tensor_in_ub, result of
                # vmax or vadd save in process_tensor, Ci > 1
                # dst_addr add src1_addr is process_tensor
                # src0_addr is tensor_in_ub and src_buffer
                dst_offset = 0
                dst_addr = process_tensor.access_ptr('w', offset=dst_offset)

                src0_offset = _loop_cut_ci * cut_hi_factor *\
                              cut_wi_factor * c_block_size
                src0_addr = src_buffer.access_ptr('r', offset=src0_offset)

                src1_offset = 0
                src1_addr = process_tensor.access_ptr('r',
                                                      offset=src1_offset)

                if pooling_mode == "GAP" and fp32_ability:
                    num_per_repeat = 64
                    repeat_data_size = cut_hi_factor *\
                                       cut_wi_factor * c_block_size
                    repeat = (repeat_data_size + 64 - 1) // 64
                    tail_repeat_data = repeat_data_size % 64
                else:
                    num_per_repeat = 128
                    repeat_data_size = cut_hi_factor *\
                                       cut_wi_factor * c_block_size
                    repeat = (repeat_data_size + 128 - 1) // 128
                    tail_repeat_data = repeat_data_size % 128

                if tail_repeat_data != 0:
                    process_bits = tail_repeat_data
                else:
                    process_bits = num_per_repeat

                if repeat <= 0:
                    raise RuntimeError("invalid repeat params,it must be >= 1")

                # dst and src1 is the same of every vmax or vadd repeat
                # so dstM1 and src1M1 is 0
                dst_stride_m0 = 1
                src0_stride_m0 = 1
                src1_stride_m0 = 1
                dst_stride_m1 = 0
                src0_stride_m1 = 8
                src1_stride_m1 = 0

                if repeat == 1:
                    # reset mask for vadd or vmax to process_bits,
                    # two scenarios included data_size < 128 or = 128
                    reset_mask_insn(ib_expr, dst.dtype, bits=process_bits)
                    ib_expr.emit(
                        tvm.call_extern(dst_buffer.dtype, pooling_intrin,
                                        dst_addr, src0_addr, src1_addr, repeat,
                                        dst_stride_m0, src0_stride_m0,
                                        src1_stride_m0,
                                        dst_stride_m1, src0_stride_m1,
                                        src1_stride_m1))

                elif 1 < repeat <= 255:
                    if tail_repeat_data == 0:
                        # Overlapping between dst and
                        # next src per repeat in one instruction
                        # is not supported in some cases
                        # Hence, use of for_range to inplace
                        # the vadd instruction is required
                        # Detect Version and use repeat instead of
                        # for_range to enhance performance
                        reset_mask_insn(ib_expr, dst.dtype, bits=num_per_repeat)
                        if is_mini_or_lhisi:
                            with ib_expr.for_range(0, repeat,
                                                   name="vadd_repeat_fix") as \
                                    _block_loop_process_tensor:
                                src0_addr = src_buffer.access_ptr(
                                    'r',
                                    offset=src0_offset +
                                    _block_loop_process_tensor *\
                                    num_per_repeat)
                                ib_expr.emit(
                                    tvm.call_extern(
                                        dst_buffer.dtype,
                                        pooling_intrin,
                                        dst_addr,
                                        src0_addr,
                                        src1_addr,
                                        1,
                                        dst_stride_m0,
                                        src0_stride_m0,
                                        src1_stride_m0,
                                        dst_stride_m1,
                                        src0_stride_m1,
                                        src1_stride_m1))
                        else:
                            ib_expr.emit(
                                tvm.call_extern(
                                    dst_buffer.dtype,
                                    pooling_intrin,
                                    dst_addr, src0_addr,
                                    src1_addr,
                                    repeat,
                                    dst_stride_m0,
                                    src0_stride_m0,
                                    src1_stride_m0,
                                    dst_stride_m1,
                                    src0_stride_m1,
                                    src1_stride_m1))
                    else:
                        # reset mask for vadd or vmax to -1
                        repeat_main = repeat - 1
                        reset_mask_insn(ib_expr, dst.dtype, bits=num_per_repeat)
                        if is_mini_or_lhisi:
                            with ib_expr.for_range(0, repeat_main,
                                                   name="vadd_repeat_fix") as \
                                    _block_loop_process_tensor:
                                src0_addr = src_buffer.access_ptr(
                                    'r',
                                    offset=src0_offset +
                                    _block_loop_process_tensor *\
                                    num_per_repeat)
                                ib_expr.emit(
                                    tvm.call_extern(
                                        dst_buffer.dtype,
                                        pooling_intrin,
                                        dst_addr,
                                        src0_addr,
                                        src1_addr,
                                        1,
                                        dst_stride_m0,
                                        src0_stride_m0,
                                        src1_stride_m0,
                                        dst_stride_m1,
                                        src0_stride_m1,
                                        src1_stride_m1))
                        else:
                            ib_expr.emit(
                                tvm.call_extern(
                                    dst_buffer.dtype,
                                    pooling_intrin,
                                    dst_addr, src0_addr,
                                    src1_addr,
                                    repeat_main,
                                    dst_stride_m0,
                                    src0_stride_m0,
                                    src1_stride_m0,
                                    dst_stride_m1,
                                    src0_stride_m1,
                                    src1_stride_m1))

                        # reset mask for vadd or vmax to process_bits
                        src0_offset = _loop_cut_ci * cut_hi_factor *\
                                      cut_wi_factor * c_block_size \
                                      + (repeat - 1) * num_per_repeat
                        src0_addr = src_buffer.access_ptr('r',
                                                          offset=src0_offset)
                        repeat_tail = 1
                        reset_mask_insn(ib_expr, dst.dtype, bits=process_bits)
                        ib_expr.emit(
                            tvm.call_extern(
                                dst_buffer.dtype,
                                pooling_intrin,
                                dst_addr, src0_addr,
                                src1_addr,
                                repeat_tail,
                                dst_stride_m0,
                                src0_stride_m0,
                                src1_stride_m0,
                                dst_stride_m1,
                                src0_stride_m1,
                                src1_stride_m1))

                elif repeat > 255:
                    repeat_loop_process = repeat // 255
                    repeat_loop_count = 255
                    repeat_left_count = repeat % 255

                    reset_mask_insn(ib_expr, dst.dtype, bits=num_per_repeat)
                    for i in range(repeat_loop_process):
                        if is_mini_or_lhisi:
                            with ib_expr.for_range(0, repeat_loop_count,
                                                   name="vadd_repeat_fix") as \
                                    _block_loop_process_tensor:
                                src0_addr = src_buffer.access_ptr(
                                    'r',
                                    offset=src0_offset +
                                    i * 255 * num_per_repeat +
                                    _block_loop_process_tensor *\
                                    num_per_repeat)
                                ib_expr.emit(
                                    tvm.call_extern(
                                        dst_buffer.dtype,
                                        pooling_intrin,
                                        dst_addr,
                                        src0_addr,
                                        src1_addr,
                                        1,
                                        dst_stride_m0,
                                        src0_stride_m0,
                                        src1_stride_m0,
                                        dst_stride_m1,
                                        src0_stride_m1,
                                        src1_stride_m1))
                        else:
                            src0_addr = src_buffer.access_ptr(
                                'r',
                                offset=src0_offset + i * 255 * num_per_repeat)
                            ib_expr.emit(
                                tvm.call_extern(
                                    dst_buffer.dtype,
                                    pooling_intrin,
                                    dst_addr, src0_addr,
                                    src1_addr,
                                    repeat_loop_count,
                                    dst_stride_m0,
                                    src0_stride_m0,
                                    src1_stride_m0,
                                    dst_stride_m1,
                                    src0_stride_m1,
                                    src1_stride_m1))

                    if tail_repeat_data == 0:
                        reset_mask_insn(ib_expr, dst.dtype, bits=num_per_repeat)
                        if is_mini_or_lhisi:
                            with ib_expr.for_range(
                                    0, repeat_left_count,
                                    name="vadd_repeat_fix_tail_repeat") as \
                                    _block_loop_process_tensor:
                                src0_addr = src_buffer.access_ptr(
                                    'r',
                                    offset=src0_offset +
                                    repeat_loop_process*255*num_per_repeat +
                                    _block_loop_process_tensor*num_per_repeat)
                                ib_expr.emit(
                                    tvm.call_extern(
                                        dst_buffer.dtype,
                                        pooling_intrin,
                                        dst_addr,
                                        src0_addr,
                                        src1_addr,
                                        1,
                                        dst_stride_m0,
                                        src0_stride_m0,
                                        src1_stride_m0,
                                        dst_stride_m1,
                                        src0_stride_m1,
                                        src1_stride_m1))
                        else:
                            src0_addr = src_buffer.access_ptr(
                                'r',
                                offset=src0_offset +
                                repeat_loop_process * 255 * num_per_repeat)
                            ib_expr.emit(
                                tvm.call_extern(
                                    dst_buffer.dtype,
                                    pooling_intrin,
                                    dst_addr, src0_addr,
                                    src1_addr,
                                    repeat_left_count,
                                    dst_stride_m0,
                                    src0_stride_m0,
                                    src1_stride_m0,
                                    dst_stride_m1,
                                    src0_stride_m1,
                                    src1_stride_m1))
                    else:
                        reset_mask_insn(ib_expr, dst.dtype, bits=num_per_repeat)
                        if repeat_left_count > 1:
                            if is_mini_or_lhisi:
                                with ib_expr.for_range(
                                        0, repeat_left_count - 1,
                                        name="vadd_repeat_fix_tail_repeat") \
                                        as _block_loop_process_tensor:
                                    src0_addr = src_buffer.access_ptr(
                                        'r',
                                        offset=src0_offset +
                                        repeat_loop_process * 255 *\
                                        num_per_repeat +
                                        _block_loop_process_tensor *\
                                        num_per_repeat)
                                    ib_expr.emit(
                                        tvm.call_extern(
                                            dst_buffer.dtype,
                                            pooling_intrin,
                                            dst_addr, src0_addr,
                                            src1_addr,
                                            1,
                                            dst_stride_m0,
                                            src0_stride_m0,
                                            src1_stride_m0,
                                            dst_stride_m1,
                                            src0_stride_m1,
                                            src1_stride_m1))
                            else:
                                src0_addr = src_buffer.access_ptr(
                                    'r',
                                    offset=src0_offset +
                                    repeat_loop_process*255*num_per_repeat)
                                ib_expr.emit(
                                    tvm.call_extern(
                                        dst_buffer.dtype,
                                        pooling_intrin,
                                        dst_addr,
                                        src0_addr,
                                        src1_addr,
                                        repeat_left_count - 1,
                                        dst_stride_m0,
                                        src0_stride_m0,
                                        src1_stride_m0,
                                        dst_stride_m1,
                                        src0_stride_m1,
                                        src1_stride_m1))

                        # add tail_repeat_data to process_tensor
                        src0_addr = src_buffer.access_ptr(
                            'r',
                            offset=src0_offset +
                            repeat_loop_process * 255 * num_per_repeat +
                            (repeat_left_count - 1) * num_per_repeat)
                        repeat_tail = 1
                        reset_mask_insn(ib_expr, dst.dtype,
                                        bits=process_bits)
                        ib_expr.emit(
                            tvm.call_extern(
                                dst_buffer.dtype,
                                pooling_intrin,
                                dst_addr, src0_addr,
                                src1_addr,
                                repeat_tail,
                                dst_stride_m0,
                                src0_stride_m0,
                                src1_stride_m0,
                                dst_stride_m1,
                                src0_stride_m1,
                                src1_stride_m1))

                # vadd or vmax the rest 7 blocks of process_tensor
                # to the first block to get 16 fp16 datas.
                # reset_mask_insn to bits = 16
                process_tensor_loop_count = num_per_repeat // c_block_size - 1
                with ib_expr.for_range(0, process_tensor_loop_count,
                                       name="block_loop_process_tensor") as \
                        _block_loop_process_tensor:
                    # get process_tensor vector include 128-fp16 numbers
                    # vadd or vmax for every 16 numbers of this 128-fp16 vector,
                    # then get 16-fp16 vectors at last, that is the result vector.
                    reset_mask_insn(ib_expr, dst.dtype, bits=16)

                    dst_offset = 0
                    dst_addr = process_tensor.access_ptr('w',
                                                         offset=dst_offset)

                    src0_offset = _block_loop_process_tensor *\
                                  c_block_size + c_block_size
                    src0_addr = process_tensor.access_ptr('r',
                                                          offset=src0_offset)

                    src1_offset = 0
                    src1_addr = process_tensor.access_ptr('r',
                                                          offset=src1_offset)

                    dst_stride_m0 = 1
                    src0_stride_m0 = 1
                    if pooling_mode == "GAP" and fp32_ability:
                        src1_stride_m0 = 1
                        dst_stride_m1 = 0
                    else:
                        src1_stride_m0 = 0
                        dst_stride_m1 = 8
                    src0_stride_m1 = 8
                    src1_stride_m1 = 0

                    repeat = 1

                    ib_expr.emit(
                        tvm.call_extern(
                            dst_buffer.dtype,
                            pooling_intrin,
                            dst_addr, src0_addr, src1_addr, repeat,
                            dst_stride_m0, src0_stride_m0,
                            src1_stride_m0,
                            dst_stride_m1, src0_stride_m1,
                            src1_stride_m1))

                # vadd or vmax with process_tensor and res_sum / res_max tensor,
                # dst is res_sum / res_max tensor, src is process_tensor
                # reset_mask_insn to bits = 16
                reset_mask_insn(ib_expr, dst.dtype, bits=16)

                dst_offset = _loop_cut_ci * c_block_size
                dst_addr = dst_buffer.access_ptr('w', offset=dst_offset)

                src0_offset = 0
                src0_addr = process_tensor.access_ptr('r', offset=src0_offset)

                src1_offset = _loop_cut_ci * c_block_size
                src1_addr = dst_buffer.access_ptr('r', offset=src1_offset)

                dst_stride_m0 = 1
                src0_stride_m0 = 1
                if pooling_mode == "GAP" and fp32_ability:
                    src1_stride_m0 = 1
                    dst_stride_m1 = 8
                else:
                    src1_stride_m0 = 8
                    dst_stride_m1 = 1
                src0_stride_m1 = 8
                src1_stride_m1 = 8

                repeat = 1

                ib_expr.emit(
                    tvm.call_extern(dst_buffer.dtype, pooling_intrin,
                                    dst_addr, src0_addr, src1_addr, repeat,
                                    dst_stride_m0, src0_stride_m0,
                                    src1_stride_m0,
                                    dst_stride_m1, src0_stride_m1,
                                    src1_stride_m1))

        # reset mask to -1 when compute finish
        # make sure do not influent other computes
        reset_mask_insn(ib_expr, dst.dtype)

    stmt = ib_expr.get()

    return stmt


OUT_OFFSET_RECORD = {}

# intrin for pooling2d end


@tvm.register_func("tvm.intrin.cce.broadcast_with_transpose")
def broadcast_with_transpose(stmt_op):
    """support broadcast with transpose method"""
    return broadcast_with_transpose_proc(stmt_op)


def broadcast_with_transpose_proc(stmt_op):
    """broadcast with transpose process"""
    # Get input and output buffers
    for_extents = []
    ir_builder = tvm.ir_builder.create()

    def _post_order_for(_stmt):
        if isinstance(_stmt, tvm.stmt.For):
            for_extents.append(int(_stmt.extent.value))

    def new_alloc(ir_builder, dtype, shape, name, scope):
        """
        new_alloc
        """
        buf_var = ir_builder.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name,
                                     scope=scope, data=buf_var)
        return new_buffer

    def _param_valid_check(for_extents, in_buffer):
        if for_extents[0] != 2:
            raise RuntimeError(
                "broadcast transpose only supports last axis is 2")

        if in_buffer.dtype != "float32":
            raise RuntimeError(
                "broadcast transpose only supports float32 date type")

    def _transform_buffer_shape(shape):
        transform_list = list(shape)
        transform_list[-1] = transform_list[-1] * 2
        transform_tuple = tuple(transform_list)
        return transform_tuple

    tvm.ir_pass.IRTransform(stmt_op, None, _post_order_for, ["For"])
    input_size = 1

    for axis_size in for_extents:
        input_size *= axis_size

    ins, outs = cce_util.get_buffer(stmt_op, need_unique=True)
    in_buffer = ins[0]
    out_buffer = outs[0]

    _param_valid_check(for_extents, in_buffer)

    vtranspose_unit_size_fp32 = cce.VECTOR_INST_BLOCK_WIDTH
    repeat_times = input_size * 2 // cce.VECTOR_INST_BLOCK_WIDTH

    # for transpose, date type * size equal 1024, 2 time vtranspose size
    conv_buffer = new_alloc(ir_builder,
                            out_buffer.dtype,
                            (vtranspose_unit_size_fp32,),
                            name="conv_buffer",
                            scope=cce.scope_ubuf)
    # for data reorder
    mid_buffer = new_alloc(ir_builder,
                           out_buffer.dtype,
                           (vtranspose_unit_size_fp32,),
                           name="mid_buffer",
                           scope=cce.scope_ubuf)

    tran_in_buffer_shape = _transform_buffer_shape(in_buffer.shape)
    transpose_in_buf = tvm.decl_buffer(tran_in_buffer_shape,
                                       "uint16_t",
                                       scope=cce.scope_ubuf,
                                       data=in_buffer.data)

    tran_out_buffer_shape = _transform_buffer_shape(out_buffer.shape)
    out_buf = tvm.decl_buffer(tran_out_buffer_shape,
                              "uint16_t",
                              scope=cce.scope_ubuf,
                              data=out_buffer.data)

    tran_mid_buffer_shape = _transform_buffer_shape(mid_buffer.shape)
    transpose_mid_buf = tvm.decl_buffer(tran_mid_buffer_shape,
                                        "uint16_t",
                                        scope=cce.scope_ubuf,
                                        data=mid_buffer.data)

    tran_conv_buffer_shape = _transform_buffer_shape(conv_buffer.shape)
    tranpose_buffer = tvm.decl_buffer(tran_conv_buffer_shape,
                                      "uint16_t",
                                      scope=cce.scope_ubuf,
                                      data=conv_buffer.data)

    with ir_builder.for_range(0, repeat_times, name="vtranspose_index") as vt_index:
        frist_transpose_offset = vt_index * 256
        ir_builder.emit(tvm.call_extern(
            "uint16_t",
            "vtranspose",
            tranpose_buffer.access_ptr("rw", offset=0),
            transpose_in_buf.access_ptr("r", offset=frist_transpose_offset)))

        ir_builder.emit(tvm.call_extern(
            in_buffer.dtype,
            "copy_ubuf_to_ubuf",
            transpose_mid_buf.access_ptr("rw", offset=0),
            tranpose_buffer.access_ptr("r", offset=0),
            0, 8, 2, 0, 2))

        ir_builder.emit(tvm.call_extern(
            in_buffer.dtype,
            "copy_ubuf_to_ubuf",
            transpose_mid_buf.access_ptr("rw", offset=32),
            tranpose_buffer.access_ptr("r", offset=0),
            0, 8, 2, 0, 2))

        ir_builder.emit(tvm.call_extern(
            "uint16_t",
            "vtranspose",
            tranpose_buffer.access_ptr("rw", offset=0),
            transpose_mid_buf.access_ptr("r", offset=0)))

        ir_builder.emit(tvm.call_extern(
            "uint16_t",
            "vtranspose",
            tranpose_buffer.access_ptr("rw", offset=256),
            transpose_mid_buf.access_ptr("r", offset=256)))

        second_reorder_offset = vt_index * 256 * 2
        ir_builder.emit(tvm.call_extern(
            in_buffer.dtype,
            "copy_ubuf_to_ubuf",
            out_buf.access_ptr("rw", offset=second_reorder_offset),
            tranpose_buffer.access_ptr("r", offset=0),
            0, 16, 1, 0, 1))

        third_reorder_offset = vt_index * 256 * 2 + 16
        ir_builder.emit(tvm.call_extern(
            in_buffer.dtype,
            "copy_ubuf_to_ubuf",
            out_buf.access_ptr("rw", offset=third_reorder_offset),
            tranpose_buffer.access_ptr("r", offset=256),
            0, 16, 1, 0, 1))

    return ir_builder.get()

@tvm.register_func("tvm.intrin.cce.broadcast_for_tensor_opt")
def broadcast_for_tensor_opt(tensor_op):
    """
    broadcast_for_tensor_opt
    """
    return broadcast_last_axis_for_tensor(tensor_op)

@tvm.register_func("tvm.intrin.cce.vector_broadcast_transpose")
def vector_broadcast_transpose(tensor_op):
    """
    :param tensor_op: tvm IR stmt in
    :return: tvm IR stmt out
    """
    def _post_order_for(tensor_op):
        if isinstance(tensor_op, tvm.stmt.For):
            for_extent_vals.append(tensor_op.extent.value)
            for_vars.append(tensor_op.loop_var)

    instr_cmd = ["vmax", "vtranspose"]
    dtype = "uint16"
    dtype_ori = "float16"
    ins, outs = cce_util.get_buffer(tensor_op, target_type=dtype)
    dst_buffer = outs[0]
    ins, outs = cce_util.get_buffer(tensor_op)
    dst_buffer_ori = outs[0]
    src_buffer_ori = ins[0]
    tvm_ib = tvm.ir_builder.create()

    for_extent_vals = []
    for_vars = []
    _ = tvm.ir_pass.IRTransform(tensor_op, None, _post_order_for, ["For"])

    error_message = "pragma vector_broadcast_transpose don't support this pattern. \n"\
                    + str(tensor_op)
    if dst_buffer_ori.dtype != dtype_ori or for_extent_vals != [16, 16]:
        raise RuntimeError(error_message)
    tvm_ib.emit(tvm.call_extern(CALL_TYPE, instr_cmd[0],
                                dst_buffer_ori.access_ptr("w", offset=0),
                                src_buffer_ori.access_ptr("r", offset=0),
                                src_buffer_ori.access_ptr("r", offset=0),
                                2, 1, 0, 0, 8, 0, 0))
    tvm_ib.emit(tvm.call_extern(CALL_TYPE, instr_cmd[1],
                                dst_buffer.access_ptr("w", offset=0),
                                dst_buffer.access_ptr("r", offset=0)))
    # print(tvm_ib.get())
    return tvm_ib.get()



def get_broadcast_axis(outs):
    """
    Get broadcast_axis_offset for tensor broadcasting
    :param outs:
    :return: broadcast_axis_offset
    """
    broadcast_max_axis_offset_list = \
        cce_emitinsn_params.cceEmitParamsIns.get_param("broadcast_axis_offset")
    if not broadcast_max_axis_offset_list:
        raise RuntimeError("Broadcast axis offset not found in emit insn params: "
                           + str(broadcast_max_axis_offset_list))
    if isinstance(broadcast_max_axis_offset_list, dict):
        broadcast_max_axis_offset = broadcast_max_axis_offset_list[outs[0].name]
    else:
        broadcast_max_axis_offset = broadcast_max_axis_offset_list
    return broadcast_max_axis_offset

def broadcast_last_axis_for_tensor(tensor_op):
    """
    broadcast_last_axis_for_tensor
    """
    buffer_var = []

    def _post_order_for(stmt_op):
        if isinstance(stmt_op, tvm.stmt.For):
            buffer_var.append(stmt_op.extent.value)

    tvm.ir_pass.IRTransform(tensor_op, None, _post_order_for, ["For"])
    _, outs = cce_util.get_buffer(tensor_op)

    broadcast_max_axis_offset = get_broadcast_axis(outs)
    broadcast_len = 1

    # (16,10,32,2,8,160,32), (16,10,32,1,1,1,1) ub tiling axis 3, broadcast_max_axis_offset 4
    broadcast_len_loop = min(broadcast_max_axis_offset, len(buffer_var))
    for i in range(0, broadcast_len_loop):
        # Broadcast (1,) to broadcast_len
        broadcast_len *= buffer_var[i]

    loop_count = 1
    if len(buffer_var) > broadcast_max_axis_offset:
        for i in range(broadcast_len_loop, len(buffer_var)):
            loop_count *= buffer_var[i]

    align_factor = cce_util.get_data_alignment(outs[0].dtype)
    if loop_count == 1:
        return broadcast_last_axis_align_for_tensor(tensor_op)
    if broadcast_len % align_factor == 0:
        return broadcast_last_axis_align_for_tensor(tensor_op)
    if broadcast_len > cce_util.get_align_factor(outs[0].dtype)[0]:
        return broadcast_last_axis_no_align_for_tensor_enhance(tensor_op)
    return broadcast_last_axis_no_align_for_tensor(tensor_op)


def broadcast_last_axis_align_for_tensor(tensor_op):
    """
    broadcast_last_axis_align_for_tensor
    """
    buffer_var = []

    def _post_order_for(stmt_op):
        if isinstance(stmt_op, tvm.stmt.For):
            buffer_var.append(stmt_op.extent.value)

    tvm.ir_pass.IRTransform(tensor_op, None, _post_order_for, ["For"])
    ins, outs = cce_util.get_buffer(tensor_op)

    broadcast_max_axis_offset = get_broadcast_axis(outs)

    loop_count = 1
    broadcast_len = 1
    broadcast_len_loop = min(broadcast_max_axis_offset, len(buffer_var))
    for i in range(0, broadcast_len_loop):
        broadcast_len *= buffer_var[i]

    if len(buffer_var) > broadcast_max_axis_offset:
        for i in range(broadcast_len_loop, len(buffer_var)):
            loop_count *= buffer_var[i]

    intrinsic_cmd = "vector_dup"
    ir_builder = tvm.ir_builder.create()
    reg = ir_builder.allocate(outs[0].dtype, (1,), name="reg_buf", scope=cce.scope_reg)
    reset_mask = 1
    with ir_builder.for_range(0, loop_count, name="idx") as idx:
        ir_builder.emit(tvm.call_extern(
            outs[0].dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[0]),
            ins[0].access_ptr("rw", offset=idx), ))

        vec_broadcast_opt(ir_builder, intrinsic_cmd, outs, broadcast_len, reset_mask, [reg[0]], idx)

    return ir_builder.get()


def vec_broadcast_opt(ir_builder, op_cmd, dst_buffers, op_length, reset_mask,
                      extern_args, loop_index, args=None):
    """
    vec_broadcast_opt
    """
    if (len(dst_buffers) != 1) or (len(extern_args) != 1):
        raise RuntimeError("vec_elewise broadcast only support one src buffer and one dst buffer")

    if args is None:
        args = [1, 1, 8, 8]

    if isinstance(extern_args[0], int):
        extern_args[0] = tvm.const(extern_args[0], "int32")
    elif isinstance(extern_args[0], float):
        extern_args[0] = tvm.const(extern_args[0], "float32")

    block_len = 256
    cal_bit_len = cce_util.get_bits_of(dst_buffers[0].dtype)
    cal_once_len = block_len*8//cal_bit_len

    local_total_len = op_length
    vec_repeat_elewise_broadcast(ir_builder, op_cmd, [], dst_buffers, local_total_len, cal_once_len,
                                 reset_mask, extern_args, args, loop_index)

# pylint: disable=too-many-locals
def vec_repeat_elewise_broadcast(ir_builder, op_cmd, src_buffers, dst_buffers, local_total_len,
                                 cal_once_len, reset_mask, extern_args, args, loop_index):
    """
    vec_repeat_elewise_broadcast
    """
    dst_dtype = dst_buffers[0].dtype
    repeat_times = local_total_len//cal_once_len
    remain_len = local_total_len - repeat_times*cal_once_len

    if repeat_times > 0:
        local_repeat_times = repeat_times
        src_roffset = 0
        dst_roffset = 0
        while local_repeat_times > 0:
            if local_repeat_times > MAX_CAL_TIMES:
                tmp_repeat_times = MAX_CAL_TIMES
            else:
                tmp_repeat_times = local_repeat_times

            tmp_args = concat_args_broadcast(src_buffers, dst_buffers,
                                             src_roffset, dst_roffset,
                                             tmp_repeat_times, extern_args,
                                             args, loop_index, local_total_len)
            ir_builder.emit(tvm.call_extern(dst_dtype, op_cmd, *tmp_args))

            local_repeat_times -= MAX_CAL_TIMES
            src_roffset += cal_once_len*tmp_repeat_times
            dst_roffset += cal_once_len*tmp_repeat_times

    if remain_len > 0:
        reset_mask_insn(ir_builder, dst_dtype, bits=remain_len)
        repeat_src_offset = repeat_times*cal_once_len*args[1]
        repeat_dst_offset = repeat_times*cal_once_len*args[0]

        tmp_args = concat_args_broadcast(src_buffers, dst_buffers,
                                         repeat_src_offset,
                                         repeat_dst_offset,
                                         1, extern_args,
                                         args, loop_index, local_total_len)
        ir_builder.emit(tvm.call_extern(dst_dtype, op_cmd, *tmp_args))

        if reset_mask is not None:
            reset_mask_insn(ir_builder, dst_dtype)


def concat_args_broadcast(src_buffers, dst_buffers, repeat_src_offset, \
                          repeat_dst_offset, repeat_times, extern_args, \
                          args, idx, local_total_len):
    """
    concat_args_broadcast
    """
    res_args = []
    for i in dst_buffers:
        res_args.append(i.access_ptr("wr", offset=idx*local_total_len+repeat_dst_offset))
    for i in src_buffers:
        res_args.append(i.access_ptr("r", offset=repeat_src_offset))
    if not isinstance(extern_args, type(None)):
        res_args += extern_args
    res_args.append(repeat_times)
    res_args += args
    return res_args

# pylint: disable=too-many-locals
def broadcast_last_axis_no_align_for_tensor(tensor_op):
    """
    broadcast_last_axis_no_align_for_tensor
    """
    buffer_var = []

    def _post_order_for(tensor_op):
        if isinstance(tensor_op, tvm.stmt.For):
            buffer_var.append(tensor_op.extent.value)

    tvm.ir_pass.IRTransform(tensor_op, None, _post_order_for, ["For"])
    ins, outs = cce_util.get_buffer(tensor_op)

    broadcast_max_axis_offset = get_broadcast_axis(outs)

    loop_total_count = 1
    loop_number = 0
    broadcast_len = 1

    broadcast_len_loop = min(broadcast_max_axis_offset, len(buffer_var))
    for i in range(0, broadcast_len_loop):
        broadcast_len *= buffer_var[i]

    if len(buffer_var) > broadcast_max_axis_offset:
        for i in range(broadcast_len_loop, len(buffer_var)):
            loop_total_count *= buffer_var[i]

    loop_number = loop_total_count//8
    if loop_number:
        remain_loop_number = loop_total_count - loop_number*8
    else:
        remain_loop_number = 1

    if loop_total_count < 8:
        reg_num = loop_total_count
    else:
        reg_num = 8

    ir_builder = tvm.ir_builder.create()
    reg = ir_builder.allocate(outs[0].dtype, (reg_num,), name="reg_buf", scope=cce.scope_reg)

    if loop_number:
        with ir_builder.for_range(0, loop_number, name="loop_idx") as loop_idx:
            ir_builder.emit(tvm.call_extern(
                outs[0].dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                ins[0].access_ptr("rw", offset=loop_idx*8+0)))
            ir_builder.emit(tvm.call_extern(
                outs[0].dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[1]),
                ins[0].access_ptr("rw", offset=loop_idx*8+1)))
            ir_builder.emit(tvm.call_extern(
                outs[0].dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[2]),
                ins[0].access_ptr("rw", offset=loop_idx*8+2)))
            ir_builder.emit(tvm.call_extern(
                outs[0].dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[3]),
                ins[0].access_ptr("rw", offset=loop_idx*8+3)))
            ir_builder.emit(tvm.call_extern(
                outs[0].dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[4]),
                ins[0].access_ptr("rw", offset=loop_idx*8+4)))
            ir_builder.emit(tvm.call_extern(
                outs[0].dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[5]),
                ins[0].access_ptr("rw", offset=loop_idx*8+5)))
            ir_builder.emit(tvm.call_extern(
                outs[0].dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[6]),
                ins[0].access_ptr("rw", offset=loop_idx*8+6)))
            ir_builder.emit(tvm.call_extern(
                outs[0].dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[7]),
                ins[0].access_ptr("rw", offset=loop_idx*8+7)))

            with ir_builder.for_range(0, broadcast_len, name="idx") as idx:
                ir_builder.emit(tvm.call_extern(
                    outs[0].dtype, "reg_mov",
                    outs[0].access_ptr("rw", offset=idx+(loop_idx*8+0)*broadcast_len),
                    tvm.call_extern(outs[0].dtype, "reg", reg[0])
                ))
                ir_builder.emit(tvm.call_extern(
                    outs[0].dtype, "reg_mov",
                    outs[0].access_ptr("rw", offset=idx+(loop_idx*8+1)*broadcast_len),
                    tvm.call_extern(outs[0].dtype, "reg", reg[1])
                ))
                ir_builder.emit(tvm.call_extern(
                    outs[0].dtype, "reg_mov",
                    outs[0].access_ptr("rw", offset=idx+(loop_idx*8+2)*broadcast_len),
                    tvm.call_extern(outs[0].dtype, "reg", reg[2])
                ))
                ir_builder.emit(tvm.call_extern(
                    outs[0].dtype, "reg_mov",
                    outs[0].access_ptr("rw", offset=idx+(loop_idx*8+3)*broadcast_len),
                    tvm.call_extern(outs[0].dtype, "reg", reg[3])
                ))
                ir_builder.emit(tvm.call_extern(
                    outs[0].dtype, "reg_mov",
                    outs[0].access_ptr("rw", offset=idx+(loop_idx*8+4)*broadcast_len),
                    tvm.call_extern(outs[0].dtype, "reg", reg[4])
                ))
                ir_builder.emit(tvm.call_extern(
                    outs[0].dtype, "reg_mov",
                    outs[0].access_ptr("rw", offset=idx+(loop_idx*8+5)*broadcast_len),
                    tvm.call_extern(outs[0].dtype, "reg", reg[5])
                ))
                ir_builder.emit(tvm.call_extern(
                    outs[0].dtype, "reg_mov",
                    outs[0].access_ptr("rw", offset=idx+(loop_idx*8+6)*broadcast_len),
                    tvm.call_extern(outs[0].dtype, "reg", reg[6])
                ))
                ir_builder.emit(tvm.call_extern(
                    outs[0].dtype, "reg_mov",
                    outs[0].access_ptr("rw", offset=idx+(loop_idx*8+7)*broadcast_len),
                    tvm.call_extern(outs[0].dtype, "reg", reg[7])
                ))

    source_offset = loop_number*8
    dest_offset = (loop_number*8)*broadcast_len

    if remain_loop_number != 1:
        reamin_reg_num = remain_loop_number
    else:
        reamin_reg_num = reg_num

    if remain_loop_number:
        with ir_builder.for_range(0, 1, name="loop_idx") as loop_idx:
            with ir_builder.for_range(0, reamin_reg_num, name="reg_idx") as reg_idx:
                ir_builder.emit(tvm.call_extern(
                    outs[0].dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[reg_idx]),
                    ins[0].access_ptr("rw", offset=source_offset+loop_idx*8+reg_idx)))
                with ir_builder.for_range(0, broadcast_len, name="idx") as idx:
                    ir_builder.emit(tvm.call_extern(outs[0].dtype, "reg_mov", outs[0].access_ptr(
                        "rw", offset=dest_offset+idx+(loop_idx*8+reg_idx)*broadcast_len),
                                                    tvm.call_extern(outs[0].dtype,
                                                                    "reg", reg[reg_idx])))


    return ir_builder.get()

def broadcast_last_axis_no_align_for_tensor_enhance(tensor_op):
    """
    broadcast_last_axis_no_align_for_tensor
    """
    buffer_var = []

    def _post_order_for(tensor_op):
        if isinstance(tensor_op, tvm.stmt.For):
            buffer_var.append(tensor_op.extent.value)

    # Initialize buffer and ir_builder
    tvm.ir_pass.IRTransform(tensor_op, None, _post_order_for, ["For"])
    ins, outs = cce_util.get_buffer(tensor_op)
    ir_builder = tvm.ir_builder.create()

    # Dtype
    dtype = ins[0].dtype
    # Block unit num
    unit_per_block, _ = cce_util.get_align_factor(dtype)
    # Broadcast 1 to broadcast_len
    broadcast_len = 1
    # Broadcast count, number of float you need to broadcast
    broadcast_count = 1
    # Number of Axis we need to loop, here is alway 2
    broadcast_max_axis_offset = get_broadcast_axis(outs)
    broadcast_len_loop = min(broadcast_max_axis_offset, len(buffer_var))
    # number of elements processed per vector op
    vector_inst_one_repeat_size = cce.VECTOR_INST_BLOCK_WIDTH \
                                  // cce_util.get_align_factor(dtype)[1]

    for i in range(0, broadcast_len_loop):
        broadcast_len *= buffer_var[i]

    if len(buffer_var) > broadcast_max_axis_offset:
        for i in range(broadcast_len_loop, len(buffer_var)):
            broadcast_count *= buffer_var[i]

    # Save regs
    current_reg_buf = []
    current_buf_indexes = []

    def read_nums_to_reg(num_indexes, current_reg_buf, current_buf_indexes):
        if current_reg_buf:
            if tuple(current_buf_indexes) == num_indexes:
                return current_reg_buf[0]
        reg = current_reg_buf[0]
        i = 0
        num_index_dict = {}
        for num_index in num_indexes:
            num_index_dict[num_index] = num_index
        pattern = find_pattern(num_index_dict)
        if pattern is None:
            for num_index in num_indexes:
                ir_builder.emit(tvm.call_extern(
                    ins[0].dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[i]),
                    ins[0].access_ptr("r", offset=num_index)))
                i += 1
        elif len(num_indexes) == 1:
            ir_builder.emit(tvm.call_extern(
                ins[0].dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                ins[0].access_ptr("r", offset=num_indexes[0])))
        else:
            with ir_builder.for_range(0, len(num_indexes), name="loop_regidx") as loop_idx:
                first_reg = num_indexes[0]
                ir_builder.emit(tvm.call_extern(
                    ins[0].dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[loop_idx]),
                    ins[0].access_ptr("r", offset=first_reg + loop_idx * pattern[0])))
        if current_reg_buf:
            current_buf_indexes.clear()
            for i in num_indexes:
                current_buf_indexes.append(i)
        else:
            current_reg_buf.append(reg)
            for i in num_indexes:
                current_buf_indexes.append(i)
        return reg

    # This function is used to find the pattern of address
    def find_pattern(addrs):
        last_addrs = None
        last_index = None
        distance = None
        index_stride = None
        for line_index in addrs.keys():
            if last_addrs is None:
                last_addrs = addrs[line_index]
                last_index = line_index
                continue
            if distance is None:
                distance = addrs[line_index] - last_addrs
                index_stride = line_index - last_index
                last_addrs = addrs[line_index]
                last_index = line_index
            else:
                current_distance = addrs[line_index] - last_addrs
                current_stride = line_index - last_index
                if distance == current_distance and index_stride == current_stride:
                    last_addrs = addrs[line_index]
                    last_index = line_index
                    continue
                else:
                    return None
        return distance, index_stride

    # Generally, each line needs two instr, broadcast me and remain for next
    # Except last line
    def get_instr(line_index, total_line, unit_per_line, align_factor):
        # Start address for each line
        start_addr = line_index * unit_per_line
        remain = start_addr % align_factor
        is_aligned = remain == 0
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
        return this_line_part, this_line_tail_part, next_line_part

    # Collect all instruction with their mask
    mask2insn = {}
    for broadcast_index in range(broadcast_count):
        start, mid, end = get_instr(broadcast_index,
                                    broadcast_count,
                                    broadcast_len,
                                    unit_per_block)
        if start is not None:
            if not start[1] in mask2insn:
                mask2insn[start[1]] = {}
            # line_index points to its address for mask
            mask2insn[start[1]][start[2]] = start[0]
        if mid is not None:
            if not mid[1] in mask2insn:
                mask2insn[mid[1]] = {}
            # line_index points to its address for mask
            mask2insn[mid[1]][mid[2]] = mid[0]
        if end is not None:
            if not end[1] in mask2insn:
                mask2insn[end[1]] = {}
            mask2insn[end[1]][end[2]] = end[0]
    broadcast_last_axis_no_align_enhance_pre_emit_insn(current_buf_indexes,
                                                       current_reg_buf, dtype,
                                                       find_pattern, ir_builder,
                                                       mask2insn, outs,
                                                       read_nums_to_reg,
                                                       unit_per_block,
                                                       vector_inst_one_repeat_size,
                                                       ins)
    reset_mask_insn(ir_builder, dtype)
    return ir_builder.get()


def broadcast_last_axis_no_align_enhance_pre_emit_insn(*args):
    """Preinitialization for broadcast last axis no align"""
    current_buf_indexes, current_reg_buf, dtype, find_pattern, \
        ir_builder, mask2insn, outs, read_nums_to_reg, unit_per_block, \
        vector_inst_one_repeat_size, ins = args
    # Emit insn
    needed_reg = 0
    for _, addrs in mask2insn.items():
        if len(addrs.keys()) > needed_reg:
            needed_reg = len(addrs.keys())
    reg_buffer = ir_builder.allocate(ins[0].dtype,
                                     (needed_reg,),
                                     name="reg_buf",
                                     scope=cce.scope_reg)
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
                                                  mask_insn, outs, pattern, reg, remain, rpt,
                                                  unit_per_block))


def broadcast_for_tensor_unaligned_emit_insn(inputs):
    """Emit Insn for the previous function"""
    addr_index_in_mask, addrs, dtype, ir_builder, \
    mask_insn, outs, pattern, reg, remain, rpt, \
    unit_per_block = inputs
    if pattern is None:
        for _, addr in addrs.items():
            if rpt > 0:
                ir_builder.emit(tvm.call_extern
                                (dtype,
                                 'vector_dup',
                                 outs[0].access_ptr("rw", offset=addr),
                                 reg[addr_index_in_mask], rpt, 1, 1, 8, 8))
            if remain > 0:
                mask_insn(ir_builder, dtype, bits=remain, ref=unit_per_block)
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
            ir_builder.emit(tvm.call_extern
                            (dtype,
                             'vector_dup',
                             outs[0].access_ptr("rw", offset=addr_offset),
                             reg[0], rpt, 1, 1, 8, 8))
        if remain > 0:
            mask_insn(ir_builder, dtype, bits=remain, ref=unit_per_block)
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
                ir_builder.emit(tvm.call_extern
                                (dtype,
                                 'vector_dup',
                                 outs[0].access_ptr("rw", offset=addr_offset),
                                 reg[loop_idx], rpt, 1, 1, 8, 8))
            if remain > 0:
                mask_insn(ir_builder, dtype, bits=remain, ref=unit_per_block)
                ir_builder.emit(tvm.call_extern
                                (dtype,
                                 'vector_dup',
                                 outs[0].access_ptr("rw", offset=addr_offset),
                                 reg[loop_idx], 1, 1, 1, 8, 8))


@tvm.register_func("tvm.intrin.cce.vector_sub_with_broadcast_enhance")
def vector_sub_with_broadcast_enhance(stmt):
    '''
    vector instric replace for sub operation with broadcast
    '''
    instr_cmd = "vsub"
    return vector_instr_with_broadcast_enhance(stmt, instr_cmd)

# pylint: disable=too-many-locals
@tvm.register_func("tvm.intrin.cce.vector_add_with_broadcast_enhance")
def vector_add_with_broadcast_enhance(stmt):
    '''
    vector instric replace for sub operation with broadcast
    '''
    instr_cmd = "vadd"
    return vector_instr_with_broadcast_enhance(stmt, instr_cmd)


@tvm.register_func("tvm.intrin.cce.vector_mul_with_broadcast_enhance")
def vector_mul_with_broadcast_enhance(stmt):
    '''
    vector instric replace for sub operation with broadcast
    '''
    instr_cmd = "vmul"
    return vector_instr_with_broadcast_enhance(stmt, instr_cmd)


@tvm.register_func("tvm.intrin.cce.vector_div_with_broadcast_enhance")
def vector_div_with_broadcast_enhance(stmt):
    '''
    vector instric replace for sub operation with broadcast
    '''
    instr_cmd = "vdiv"
    return vector_instr_with_broadcast_enhance(stmt, instr_cmd)

@tvm.register_func("tvm.intrin.cce.vector_min_with_broadcast_enhance")
def vector_min_with_broadcast_enhance(stmt):
    '''
    vector instric replace for sub operation with broadcast
    '''
    instr_cmd = "vmin"
    return vector_instr_with_broadcast_enhance(stmt, instr_cmd)

@tvm.register_func("tvm.intrin.cce.vector_max_with_broadcast_enhance")
def vector_max_with_broadcast_enhance(stmt):
    '''
    vector instric replace for max operation with broadcast
    '''
    instr_cmd = "vmax"
    return vector_instr_with_broadcast_enhance(stmt, instr_cmd)

def vector_instr_with_broadcast_enhance(stmt, instr_cmd):
    '''
    vector instric for stmt with broadcast vector operation
    '''
    ins, outs = cce_util.get_buffer(stmt)
    ir_builder = tvm.ir_builder.create()

    for_extent_vals = []
    for_vars = []

    def _post_order_for(_stmt):
        if isinstance(_stmt, tvm.stmt.For):
            for_extent_vals.append(_stmt.extent.value)
            for_vars.append(_stmt.loop_var)

    _ = tvm.ir_pass.IRTransform(stmt, None, _post_order_for, ["For"])
    op_size = 1
    _broadcast_buffer_saving_threshold = 16
    _broadcast_buffer_eco = 'ECONOMIC'
    _broadcast_buffer_agg = 'AGGRESSIVE'
    for i in for_extent_vals:
        op_size = op_size * i

    if get_emitinsn_params("broadcast_axis_multiply_flag"):
        last_number = for_extent_vals[0] * for_extent_vals[1]
    else:
        last_number = for_extent_vals[0]

    dtype = ins[0].dtype
    dst_buffer = outs[0]

    small_buf_index = get_op_small_tensor_index(stmt)

    if small_buf_index == len(ins):
        reset_mask_insn(ir_builder, dtype, bits=last_number)
        ir_builder.emit(
            tvm.call_extern(dtype, instr_cmd,
                            dst_buffer.access_ptr("rw", offset=0),
                            ins[0].access_ptr("r", offset=0),
                            ins[1].access_ptr("r", offset=0), 1, 1, 1, 1, 8, 8, 8))
        return ir_builder.get()

    src_buf_bc = ins[small_buf_index]
    src_buffer = ins[(small_buf_index + 1) % 2]
    res_buffer = outs[0]

    def new_alloc(ir_builder, dtype, shape, name, scope):
        buf_var = ir_builder.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name,
                                     scope=scope, data=buf_var)

        return new_buffer

    vector_inst_one_repeat_size = cce.VECTOR_INST_BLOCK_WIDTH\
        // cce_util.get_align_factor(dtype)[1]

    # It is impossible to let last_number be greater than one_repeat size
    tail_in_each_repeat = vector_inst_one_repeat_size % last_number

    # Generate mask needed for (1, a) broadcasting
    def get_spec_broadcast_mask(last_num, loop_index, repeat_index):
        reg_bitsize = 64
        actual_reg_num = 2
        mask_unit_size = last_num
        result = ''
        for repeat_unit in range(vector_inst_one_repeat_size):
            expect_mask_unit_index = (repeat_index * tail_in_each_repeat + repeat_unit) % \
                                     mask_unit_size
            if expect_mask_unit_index == loop_index:
                result = '1' + result
            else:
                result = '0' + result
        final = []
        for reg_num in range(vector_inst_one_repeat_size // reg_bitsize):
            final.append(result[reg_num * reg_bitsize:(reg_num + 1) * reg_bitsize])
        if len(final) < actual_reg_num:
            for reg_num in range(actual_reg_num - len(final)):
                final.insert(0, reg_bitsize * '0')
        return final

    def gen_broadcast_offset_map(last_num, vinst_one_repeat_size):
        _broadcast_unit_size = ALIGNMENT_BYTES // cce_util.get_align_factor(dtype)[1]
        _tail2aggoffset = {}
        _aggoffset2tail = {}
        for n_i in range(0, last_num):
            aggtail = int((n_i + 1) * vinst_one_repeat_size % last_num)
            _tail2aggoffset[aggtail] = n_i
            _aggoffset2tail[n_i] = aggtail
        _tail2ecoindex = {}
        _ecoindex2tail = {}
        n_i = 0
        ecotail = int((vector_inst_one_repeat_size + n_i * _broadcast_unit_size) % last_num)
        while ecotail not in _tail2ecoindex:
            _tail2ecoindex[ecotail] = n_i
            _ecoindex2tail[n_i] = ecotail
            n_i += 1
            ecotail = int((vector_inst_one_repeat_size + n_i * _broadcast_unit_size) % last_num)
        return _tail2aggoffset, _aggoffset2tail, _tail2ecoindex, _ecoindex2tail

    def get_broadcast_offset(_mode, _offset2tail, _tail2ecoindex, _aggoffset):
        # No offset needed for aggressive broadcasting
        if _mode == _broadcast_buffer_agg:
            return 0

        if _mode == _broadcast_buffer_eco:
            _tail = _offset2tail[_aggoffset]
            _ecoindex = _tail2ecoindex[_tail]
            return _ecoindex

        raise RuntimeError("Unknown broadcasting mode")

    # Calculate repeat needed for broadcasting final result
    # 2 -> 1 because 128 % 2 = 0, 3 -> 3 because 128 % 3 = 2
    repeat_for_broadcast = last_number
    if tail_in_each_repeat == 0:
        repeat_for_broadcast = 1
    # For small last axis, use aggressive broadcasting strategy
    # Otherwise, use economic broadcasting strategy
    mode = _broadcast_buffer_agg
    if last_number > _broadcast_buffer_saving_threshold and repeat_for_broadcast != 1:
        mode = _broadcast_buffer_eco
        repeat_for_broadcast = math.ceil(last_number /
                                         (vector_inst_one_repeat_size /
                                          (ALIGNMENT_BYTES /
                                           cce_util.get_align_factor(dtype)[1])))*2
        _, aggoffset2tail, tail2ecoindex, _ = gen_broadcast_offset_map(last_number,
                                                                       vector_inst_one_repeat_size)
    else:
        _, aggoffset2tail, tail2ecoindex, _ = None, None, None, None
    # Distribution of (repeat, dataindex) to masks in order to minimize number of set_mask
    mask2reploop = {}
    reploop2mask = {}
    for l_i in range(last_number):
        k = tuple(get_spec_broadcast_mask(last_number, l_i, 0))
        mask2reploop[tuple(int(mask, 2) for mask in k)] = [(0, l_i)]
        reploop2mask[(0, l_i)] = tuple(int(mask, 2) for mask in k)
    for r_i in range(1, last_number):
        for l_i in range(last_number):
            mapindex = (r_i * (last_number - tail_in_each_repeat) + l_i) % last_number
            mask2reploop[reploop2mask[0, mapindex]].append((r_i, l_i))
    # Allocate buffer needed for broadcasting final result
    tmp_buffer = new_alloc(ir_builder, dtype,
                           (vector_inst_one_repeat_size *
                           (repeat_for_broadcast + 1)),
                           'tmp_buffer', scope=cce.scope_ubuf)

    # Zerorize the buffer
    reset_mask_insn(ir_builder, dtype, bits=vector_inst_one_repeat_size)
    ir_builder.emit(tvm.call_extern
                    (dtype, 'vector_dup', tmp_buffer.access_ptr("rw", offset=0),
                     tvm.const(0, dtype=dtype), repeat_for_broadcast, 1, 1, 8, 8))

    # Allocate reg buffer needed for holding src data
    reg = ir_builder.allocate(outs[0].dtype, (last_number,), name="reg_buf",
                              scope=cce.scope_reg)
    # reg_mov src data
    with ir_builder.for_range(0, last_number, name="reg_idx") as reg_idx:
        ir_builder.emit(tvm.call_extern(
            outs[0].dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_idx]),
            src_buf_bc.access_ptr("rw", offset=reg_idx)))

    # do Broadcast
    # For (1,a) to 128 instead of a * 128 such as 1, 2, 4, 8
    # Exchange their loop and repeat number, turn dynamic offset for tail part off
    loop_count = last_number
    dyn_offset = True
    if repeat_for_broadcast == 1:
        loop_count = 1
        repeat_for_broadcast = last_number
        dyn_offset = False
    all_masks = list(mask2reploop.keys())
    if mode == _broadcast_buffer_eco:
        original_broadcast_num = last_number
    elif mode == _broadcast_buffer_agg:
        original_broadcast_num = repeat_for_broadcast
    else:
        raise RuntimeError('Unknown broadcast mode')
    for r_i in range(original_broadcast_num):
        reset_multi_broaddcast_mask_insn(ir_builder, dtype,
                                         all_masks[r_i][0], all_masks[r_i][1])
        for l_i in range(loop_count):
            reploops = mask2reploop[all_masks[r_i]]
            reploop = reploops[l_i]
            rep = reploop[0]
            if mode == _broadcast_buffer_eco and rep > repeat_for_broadcast:
                continue
            ir_builder.emit(tvm.call_extern(
                dtype, "vector_dup",
                tmp_buffer.access_ptr("rw",
                                      offset=rep * vector_inst_one_repeat_size),
                reg[mask2reploop[all_masks[r_i]][l_i][1]],
                1, 1, 1, 8, 0))
    max_repeat_times = last_number
    repeat_times = op_size // vector_inst_one_repeat_size
    original_loop_count = repeat_times
    remain_size = op_size - repeat_times * vector_inst_one_repeat_size

    reset_mask_insn(ir_builder, dtype, bits=vector_inst_one_repeat_size)

    count = 0
    # Main repeat
    repeat_stride = 8
    if mode == _broadcast_buffer_eco:
        max_repeat_times = repeat_for_broadcast // 2
    if tail_in_each_repeat == 0:
        repeat_stride = 0
        max_repeat_times = cce.VECTOR_INST_MAX_REPEAT_TIMES
    count = repeat_times // max_repeat_times
    if count > 0:
        for loop in range(0, count):
            repeat_times = repeat_times - max_repeat_times
            offset_value = loop * max_repeat_times * vector_inst_one_repeat_size
            aggoffset = loop * max_repeat_times % last_number
            broadcast_unit_size = ALIGNMENT_BYTES // cce_util.get_align_factor(dtype)[1]

            src_0 = tmp_buffer.access_ptr(
                "r", offset=get_broadcast_offset(mode,
                                                 aggoffset2tail,
                                                 tail2ecoindex,
                                                 aggoffset)*broadcast_unit_size)
            src_1 = src_buffer.access_ptr("r", offset=offset_value)

            if small_buf_index == 1:
                ir_builder.emit(
                    tvm.call_extern(dtype, instr_cmd,
                                    dst_buffer.access_ptr("rw",
                                                          offset=offset_value),
                                    src_1, src_0,
                                    max_repeat_times, 1, 1, 1, 8, 8, repeat_stride))
            else:
                ir_builder.emit(
                    tvm.call_extern(dtype, instr_cmd,
                                    dst_buffer.access_ptr("rw",
                                                          offset=offset_value),
                                    src_0, src_1,
                                    max_repeat_times, 1, 1, 1, 8, repeat_stride, 8))

        repeat_times = repeat_times % max_repeat_times
    # Tail repeat
    if repeat_times != 0:
        offset_value = count * max_repeat_times * vector_inst_one_repeat_size
        aggoffset = count * max_repeat_times % last_number
        broadcast_unit_size = ALIGNMENT_BYTES // cce_util.get_align_factor(dtype)[1]
        tailrepoffset = get_broadcast_offset(mode,
                                             aggoffset2tail,
                                             tail2ecoindex,
                                             aggoffset) * broadcast_unit_size

        src_0 = tmp_buffer.access_ptr("r", offset=tailrepoffset)
        src_1 = src_buffer.access_ptr("r", offset=offset_value)
        if small_buf_index == 1:
            ir_builder.emit(
                tvm.call_extern(dtype, instr_cmd,
                                dst_buffer.access_ptr("rw",
                                                      offset=offset_value),
                                src_1, src_0, repeat_times, 1, 1, 1, 8, 8, repeat_stride))
        else:
            ir_builder.emit(
                tvm.call_extern(dtype, instr_cmd,
                                dst_buffer.access_ptr("rw",
                                                      offset=offset_value),
                                src_0, src_1, repeat_times, 1, 1, 1, 8, repeat_stride, 8))
    # Real tail
    broadcast_src_offset = vector_inst_one_repeat_size * (repeat_times % last_number)
    if not dyn_offset:
        broadcast_src_offset = 0
    if remain_size > 0:
        reset_mask_insn(ir_builder, src_buffer.dtype, bits=remain_size)
        offset_value = \
            original_loop_count * vector_inst_one_repeat_size
        aggoffset = count * max_repeat_times % last_number
        broadcast_unit_size = ALIGNMENT_BYTES // cce_util.get_align_factor(dtype)[1]

        src_0 = tmp_buffer.access_ptr(
            "r",
            offset=get_broadcast_offset(mode,
                                        aggoffset2tail,
                                        tail2ecoindex,
                                        aggoffset)*broadcast_unit_size + broadcast_src_offset)
        src_1 = src_buffer.access_ptr("r", offset=offset_value)

        if small_buf_index == 1:
            ir_builder.emit(
                tvm.call_extern(dtype, instr_cmd,
                                dst_buffer.access_ptr("rw",
                                                      offset=offset_value),
                                src_1, src_0, 1, 1, 1, 1, 8, 8, repeat_stride))
        else:
            ir_builder.emit(
                tvm.call_extern(dtype, instr_cmd,
                                dst_buffer.access_ptr("rw",
                                                      offset=offset_value),
                                src_0, src_1, 1, 1, 1, 1, 8, repeat_stride, 8))

    reset_mask_insn(ir_builder, res_buffer.dtype)

    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.vector_add_with_broadcast_non_32align")
def vector_add_with_broadcast_enhance_enhance(stmt):
    '''
    vector_add_with_broadcast_enhance_enhance
    '''
    instr_cmd = "vadd"
    return vector_instr_with_broadcast_non_32align(stmt, instr_cmd)


@tvm.register_func("tvm.intrin.cce.vector_sub_with_broadcast_non_32align")
def vector_sub_with_broadcast_enhance_enhance(stmt):
    '''
    vector_add_with_broadcast_enhance_enhance
    '''
    instr_cmd = "vsub"
    return vector_instr_with_broadcast_non_32align(stmt, instr_cmd)


@tvm.register_func("tvm.intrin.cce.vector_mul_with_broadcast_non_32align")
def vector_mul_with_broadcast_enhance_enhance(stmt):
    '''
    vector_add_with_broadcast_enhance_enhance
    '''
    instr_cmd = "vmul"
    return vector_instr_with_broadcast_non_32align(stmt, instr_cmd)


@tvm.register_func("tvm.intrin.cce.vector_div_with_broadcast_non_32align")
def vector_div_with_broadcast_enhance_enhance(stmt):
    '''
    vector_add_with_broadcast_enhance_enhance
    '''
    instr_cmd = "vdiv"
    return vector_instr_with_broadcast_non_32align(stmt, instr_cmd)


def vector_instr_with_broadcast_non_32align(stmt, instr_cmd):
    '''
    vector_instr_with_broadcast_non_32align operation
    '''
    # pylint: too-many-statements
    # pylint: too-many-branches
    input_shape = []
    cce_util.get_input_shape_from_stmt(input_shape, stmt)

    if input_shape[0] == input_shape[1]:
        if instr_cmd == "vadd":
            return elewise_binary_add(stmt)
        if instr_cmd == "vsub":
            return elewise_binary_sub(stmt)
        if instr_cmd == "vmul":
            return elewise_binary_mul(stmt)
        if instr_cmd == "vdiv":
            return vec_binary_elewise(stmt, instr_cmd)

    ins, outs = cce_util.get_buffer(stmt)
    ir_builder = tvm.ir_builder.create()

    for_extent_vals = []
    for_vars = []

    def _post_order_for(_stmt):
        if isinstance(_stmt, tvm.stmt.For):
            for_extent_vals.append(_stmt.extent.value)
            for_vars.append(_stmt.loop_var)

    _ = tvm.ir_pass.IRTransform(stmt, None, _post_order_for, ["For"])
    op_size = 1
    for i in for_extent_vals:
        op_size = op_size * i

    last_number = for_extent_vals[0]

    dtype = ins[0].dtype

    small_buf_index = get_op_small_tensor_index(stmt)
    src_buf_bc = ins[small_buf_index]
    src_buffer = ins[(small_buf_index + 1) % 2]
    res_buffer = outs[0]

    tensor_var_list = get_op_tensor_var(stmt)

    for_extent_vals.reverse()
    for_vars.reverse()

    if not tensor_var_list:
        raise RuntimeError("Get broadcast stmt failed!")

    loop_var = []
    loop_op_size = []

    def __get_loop_params():
        index = 0
        for var in for_vars:
            if index == len(for_vars) - 1:
                value = for_extent_vals[index]
                loop_op_size.append(value)
                break
            if var in tensor_var_list:
                value = for_extent_vals[index]
                loop_var.append(value)
            else:
                value = for_extent_vals[index]
                loop_op_size.append(value)
            index = index + 1

    __get_loop_params()

    def __shape_mul(shape):
        if not shape:
            return 1
        return reduce(lambda x, y: x * y, shape)

    def __get_loop_cout():
        loop_count = 1
        loop_count_out = 1
        if loop_var:
            for i in range(len(loop_var) - 1, -1, -1):
                loop_count_temp = loop_count
                loop_count = loop_count * loop_var[i]
                if loop_count > 512:
                    loop_count_out = __shape_mul(loop_var[:i+1])
                    loop_count = loop_count_temp
                    break
        return loop_count, loop_count_out

    loop_count, loop_count_out = __get_loop_cout()

    op_size = 1
    for i in loop_op_size:
        op_size = op_size * i

    mid_count = op_size // last_number

    with ir_builder.for_range(0, loop_count_out, name="reg_idx_out_out") as loop_count_out_index:
        reg = ir_builder.allocate(outs[0].dtype, (1,), name="reg",
                                  scope=cce.scope_reg)
        with ir_builder.for_range(0, loop_count, name="reg_idx_out") as reg_idx_out:
            with ir_builder.for_range(0, mid_count, name="reg_idx_mid") as reg_idx_mid:
                with ir_builder.for_range(0, last_number, name="reg_idx_in") as reg_idx_in:
                    reg_in_offset = loop_count_out_index * loop_count * last_number + \
                                    reg_idx_out * last_number + reg_idx_in
                    reg_out_offset = loop_count_out_index * loop_count * op_size + \
                                     reg_idx_out * mid_count * last_number + \
                                     reg_idx_mid * last_number + reg_idx_in
                    ir_builder.emit(tvm.call_extern(
                        outs[0].dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[0]),
                        src_buf_bc.access_ptr("r", offset=reg_in_offset)))
                    ir_builder.emit(tvm.call_extern(
                        outs[0].dtype, "reg_mov",
                        outs[0].access_ptr("rw", offset=reg_out_offset),
                        tvm.call_extern(reg.dtype, "reg", reg[0])))
        block_len = 256
        cal_bit_len = cce_util.get_bits_of(dtype)
        cal_once_len = block_len*8//cal_bit_len

        repeat_times = loop_count * op_size // cal_once_len
        remain_len = loop_count * op_size - repeat_times * cal_once_len

        vector_instr_with_broadcast_non_32align_emit_insn(cal_once_len, dtype,
                                                          ins, instr_cmd,
                                                          ir_builder,
                                                          loop_count,
                                                          loop_count_out_index,
                                                          op_size, remain_len,
                                                          repeat_times,
                                                          res_buffer,
                                                          src_buffer)

    return ir_builder.get()


def vector_instr_with_broadcast_non_32align_emit_insn(*args):
    """
    EmitInsn step for vector_instr_with_broadcast_non_32align operation
    """
    cal_once_len, dtype, ins, instr_cmd, ir_builder, loop_count, \
        loop_count_out_index, op_size, remain_len, repeat_times, \
        res_buffer, src_buffer = args
    count = 0
    max_repeat_times = 254
    if repeat_times > max_repeat_times:
        count = repeat_times // max_repeat_times
        with ir_builder.for_range(0, count, name="add_index") as add_index:
            repeat_offset = \
                loop_count_out_index * loop_count * op_size + \
                cal_once_len * add_index
            if src_buffer == ins[0]:
                ir_builder.emit(tvm.call_extern(dtype,
                                                instr_cmd,
                                                res_buffer.access_ptr(
                                                    "wr", offset=repeat_offset),
                                                src_buffer.access_ptr(
                                                    "r", offset=repeat_offset),
                                                res_buffer.access_ptr(
                                                    "r", offset=repeat_offset),
                                                max_repeat_times, 1, 1, 1, 8, 8, 8))
            else:
                ir_builder.emit(tvm.call_extern(dtype,
                                                instr_cmd,
                                                res_buffer.access_ptr(
                                                    "wr", offset=repeat_offset),
                                                res_buffer.access_ptr(
                                                    "r", offset=repeat_offset),
                                                src_buffer.access_ptr(
                                                    "r", offset=repeat_offset),
                                                max_repeat_times, 1, 1, 1, 8, 8, 8))
        repeat_times = repeat_times % max_repeat_times
    if repeat_times > 0:
        repeat_offset = \
            loop_count_out_index * loop_count * op_size + \
            cal_once_len * (count * max_repeat_times)
        if src_buffer == ins[0]:
            ir_builder.emit(tvm.call_extern(dtype,
                                            instr_cmd,
                                            res_buffer.access_ptr(
                                                "wr", offset=repeat_offset),
                                            src_buffer.access_ptr(
                                                "r", offset=repeat_offset),
                                            res_buffer.access_ptr(
                                                "r", offset=repeat_offset),
                                            repeat_times, 1, 1, 1, 8, 8, 8))
        else:
            ir_builder.emit(tvm.call_extern(dtype,
                                            instr_cmd,
                                            res_buffer.access_ptr(
                                                "wr", offset=repeat_offset),
                                            res_buffer.access_ptr(
                                                "r", offset=repeat_offset),
                                            src_buffer.access_ptr(
                                                "r", offset=repeat_offset),
                                            repeat_times, 1, 1, 1, 8, 8, 8))
    if remain_len > 0:
        reset_mask_insn(ir_builder, dtype, bits=remain_len)
        repeat_offset = \
            loop_count_out_index * loop_count * op_size + \
            (count * max_repeat_times + repeat_times) * cal_once_len
        if src_buffer == ins[0]:
            ir_builder.emit(tvm.call_extern(dtype,
                                            instr_cmd,
                                            res_buffer.access_ptr(
                                                "wr", offset=repeat_offset),
                                            src_buffer.access_ptr(
                                                "r", offset=repeat_offset),
                                            res_buffer.access_ptr(
                                                "r", offset=repeat_offset),
                                            1, 1, 1, 1, 8, 8, 8))
        else:
            ir_builder.emit(tvm.call_extern(dtype,
                                            instr_cmd,
                                            res_buffer.access_ptr(
                                                "wr", offset=repeat_offset),
                                            res_buffer.access_ptr(
                                                "r", offset=repeat_offset),
                                            src_buffer.access_ptr(
                                                "r", offset=repeat_offset),
                                            1, 1, 1, 1, 8, 8, 8))
    reset_mask_insn(ir_builder, res_buffer.dtype)


@tvm.register_func("tvm.intrin.cce.broadcast_for_tensor_opt_mid_le32")
def broadcast_for_tensor_opt_mid_le32(stmt):
    """
    fp32 only, coverage strictly limited
    :param stmt:
    :return: ir_builder.get()
    """
    ins, outs = cce_util.get_buffer(stmt)
    ir_builder = tvm.ir_builder.create()

    def new_alloc(ir_build, dtype, shape, name, scope):
        buf_var = ir_build.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name,
                                     scope=scope, data=buf_var)

        return new_buffer

    # Allocate extra buffer as zeroed_buffer
    zeroed_shape = ins[0].shape[:]
    zeroed_shape[0] = zeroed_shape[0] // 2
    zeroed_buffer = new_alloc(ir_builder,
                              ins[0].dtype,
                              zeroed_shape,
                              "broadcast_zeroed_buf",
                              cce.scope_ubuf)
    ir_builder.emit(
        tvm.call_extern(ins[0].dtype,
                        'vector_dup',
                        zeroed_buffer.access_ptr("rw", offset=0),
                        tvm.const(0,
                                  dtype=ins[0].dtype),
                        ins[0].shape[0]//16, 1, 1, 8, 8))
    # Allocate extra buffer for vadd
    filler_buffer = new_alloc(ir_builder,
                              ins[0].dtype,
                              ins[0].shape[:],
                              "broadcast_filling_buf",
                              cce.scope_ubuf)
    # Set mask for vadd and perform the transformation
    upper_mask = int('0b0000000000000000000000000000000000000000000000000000000000000000', 2)
    lower_mask = int('0b0000111100001111000011110000111100001111000011110000111100001111', 2)
    reset_multi_broaddcast_mask_insn(ir_builder, ins[0].dtype, upper_mask, lower_mask)
    ir_builder.emit(tvm.call_extern(
        ins[0].dtype, "vadd",
        filler_buffer.access_ptr("rw", offset=0),
        ins[0].access_ptr("r", offset=0),
        zeroed_buffer.access_ptr("r", offset=0),
        ins[0].shape[0] // 16, 2, 1, 1, 16, 8, 8))
    lower_mask = int('0b1111000011110000111100001111000011110000111100001111000011110000', 2)
    reset_multi_broaddcast_mask_insn(ir_builder, ins[0].dtype, upper_mask, lower_mask)
    ir_builder.emit(tvm.call_extern(
        ins[0].dtype, "vadd",
        filler_buffer.access_ptr("rw", offset=4 * 2),
        ins[0].access_ptr("r", offset=0),
        zeroed_buffer.access_ptr("r", offset=0),
        ins[0].shape[0] // 16, 2, 1, 1, 16, 8, 8))
    # Tail part
    if int(ins[0].shape[0] % 16) > 0:
        lower_mask_bstr = ''
        for half_block in range(int(ins[0].shape[0] % 16)):
            if half_block % 2 == 0:
                lower_mask_bstr = '1111' + lower_mask_bstr
            else:
                lower_mask_bstr = '0000' + lower_mask_bstr
        lower_mask_bstr = '0b' + lower_mask_bstr.zfill(64)
        lower_mask = int(lower_mask_bstr, 2)
        reset_multi_broaddcast_mask_insn(ir_builder, ins[0].dtype, upper_mask, lower_mask)
        ir_builder.emit(tvm.call_extern(
            ins[0].dtype, "vadd",
            filler_buffer.access_ptr("rw", offset=ins[0].shape[0] // 16 * 128),
            ins[0].access_ptr("r", offset=ins[0].shape[0] // 16 * 64),
            zeroed_buffer.access_ptr("r", offset=0),
            1, 2, 1, 1, 16, 8, 8))
        lower_mask_bstr = ''
        for half_block in range(int(ins[0].shape[0] % 16)):
            if half_block % 2 == 0:
                lower_mask_bstr = '0000' + lower_mask_bstr
            else:
                lower_mask_bstr = '1111' + lower_mask_bstr
        lower_mask_bstr = '0b' + lower_mask_bstr.zfill(64)
        lower_mask = int(lower_mask_bstr, 2)
        reset_multi_broaddcast_mask_insn(ir_builder, ins[0].dtype, upper_mask, lower_mask)
        ir_builder.emit(tvm.call_extern(
            ins[0].dtype, "vadd",
            filler_buffer.access_ptr("rw", offset=ins[0].shape[0] // 16 * 128 + 4 * 2),
            ins[0].access_ptr("r", offset=ins[0].shape[0] // 16 * 64),
            zeroed_buffer.access_ptr("r", offset=0),
            1, 2, 1, 1, 16, 8, 8))
    # Move out to UB
    ir_builder.emit(
        tvm.call_extern(ins[0].dtype, "copy_ubuf_to_gm",
                        outs[0].access_ptr("rw", offset=0), filler_buffer.access_ptr("r", offset=0),
                        0, 1, ins[0].shape[0], 0, 0))
    ir_builder.emit(
        tvm.call_extern(ins[0].dtype, "copy_ubuf_to_gm",
                        outs[0].access_ptr("rw", offset=4), ins[0].access_ptr("r", offset=0),
                        0, ins[0].shape[0] // 2, 1, 0, 1))
    return ir_builder.get()


def reset_multi_broaddcast_mask_insn(ir_builder, type_, mask1, mask2):
    """
    :describe: caculate the mask, and set vector mask
    :param ir_builder: ir builder
    :param type_: the type of mask dst
    """

    ir_builder.emit(tvm.call_extern(
        type_, "set_vector_mask", tvm.const(mask1, dtype="uint64"),
        tvm.const(mask2, dtype="uint64")))

@tvm.register_func("tvm.intrin.cce.vector_add_with_broadcast")
def vector_add_with_broadcast(stmt):
    '''
    vector instric replace for add operation with broadcast
    '''
    instr_cmd = "vadd"
    return vector_instr_with_broadcast(stmt, instr_cmd)


@tvm.register_func("tvm.intrin.cce.vector_mul_with_broadcast")
def vector_mul_with_broadcast(stmt):
    '''
    vector instric replace for mul operation with broadcast
    '''
    instr_cmd = "vmul"
    return vector_instr_with_broadcast(stmt, instr_cmd)


@tvm.register_func("tvm.intrin.cce.vector_sub_with_broadcast")
def vector_sub_with_broadcast(stmt):
    '''
    vector instric replace for sub operation with broadcast
    '''
    instr_cmd = "vsub"
    return vector_instr_with_broadcast(stmt, instr_cmd)

# pylint: disable=too-many-locals
@tvm.register_func("tvm.intrin.cce.vector_div_with_broadcast")
def vector_div_with_broadcast(stmt):
    '''
    vector instric replace for div operation with broadcast
    '''
    instr_cmd = "vdiv"
    return vector_instr_with_broadcast(stmt, instr_cmd)


def get_op_small_tensor_index(stmt_in):
    """
    descripe:get the small buffer index in src buffer
    stmt_in:the emit_insn ir:
        for (i3.c, 0, 768) {
            for (i4.c, 0, 16) {
                mul_9.local.UB[((i3.c*16) + i4.c)] =
                    (div_6.local.UB[i4.c]*cast_0.local.UB[((i3.c*16) + i4.c)])
            }
        }
    return :the small buffer index, in this case, div_6.local.UB is small buffer, so return 0
    """
    for_var = []

    def _post_order_for(stmt):
        if isinstance(stmt, tvm.stmt.For):
            if stmt.extent.value != 1:
                for_var.append(stmt.loop_var)

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order_for, ["For"])

    loads = []

    def _post_order(stmt):
        if isinstance(stmt, tvm.expr.Load):
            loads.append(stmt)

    # get the all load
    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Load"])

    index = 0
    for load in loads:
        var_list = []

        def get_var(stmt):
            # pylint: disable=cell-var-from-loop
            if isinstance(stmt, tvm.expr.Add):
                get_var(stmt.a)
                get_var(stmt.b)
            elif isinstance(stmt, tvm.expr.Mul):
                get_var(stmt.a)
                get_var(stmt.b)
            elif isinstance(stmt, tvm.expr.Var):
                var_list.append(stmt)

        get_var(load.index)

        if len(var_list) != len(for_var):
            break

        for var in for_var:
            if var not in var_list:
                return index

        index += 1

    return index

def get_op_tensor_var(stmt_in):
    """
    descripe:get the small op in stmt
    """
    for_var = []

    def _post_order_for(stmt):
        if isinstance(stmt, tvm.stmt.For):
            if stmt.extent.value != 1:
                for_var.append(stmt.loop_var)

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order_for, ["For"])

    loads = []

    def _post_order(stmt):
        if isinstance(stmt, tvm.expr.Load):
            loads.append(stmt)

    # get the all load
    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Load"])

    tensor_var_list = []
    for load in loads:
        tensor_var_list = []

        def get_var(stmt):
            # pylint: disable=cell-var-from-loop
            if isinstance(stmt, tvm.expr.Add):
                get_var(stmt.a)
                get_var(stmt.b)
            elif isinstance(stmt, tvm.expr.Mul):
                get_var(stmt.a)
                get_var(stmt.b)
            elif isinstance(stmt, tvm.expr.Var):
                tensor_var_list.append(stmt)

        get_var(load.index)

        if len(tensor_var_list) != len(for_var):
            break

        for var in for_var:
            if var not in tensor_var_list:
                return tensor_var_list
    return tensor_var_list

# pylint: disable=too-many-locals
def vector_instr_with_broadcast(stmt, instr_cmd):
    '''
    vector instric for stmt with broadcast vector operation
    '''
    ins, outs = cce_util.get_buffer(stmt)
    ir_builder = tvm.ir_builder.create()

    for_extent_vals = []
    for_vars = []

    def _post_order_for(_stmt):
        if isinstance(_stmt, tvm.stmt.For):
            for_extent_vals.append(_stmt.extent.value)
            for_vars.append(_stmt.loop_var)

    _ = tvm.ir_pass.IRTransform(stmt, None, _post_order_for, ["For"])

    dtype = ins[0].dtype
    dst_buffer = outs[0]
    c0_size = 16

    tensor_var_list = []
    tensor_var_list = get_op_tensor_var(stmt)

    for_extent_vals.reverse()
    for_vars.reverse()

    if not tensor_var_list:
        raise RuntimeError("Get broadcast stmt failed!")

    loop_var = []
    loop_op_size = []
    index = 0
    for var in for_vars:
        if index == len(for_vars) - 1:
            value = for_extent_vals[index]
            loop_op_size.append(value)
            break
        if var in tensor_var_list:
            value = for_extent_vals[index]
            loop_var.append(value)
        else:
            value = for_extent_vals[index]
            loop_op_size.append(value)
        index = index + 1

    loop_count = 1
    for index, element in enumerate(loop_var):
        loop_count = loop_count * element

    op_size = 1
    for i in loop_op_size:
        op_size = op_size * i

    small_buf_index = get_op_small_tensor_index(stmt)

    if small_buf_index == len(ins):
        if op_size > c0_size:
            raise RuntimeError(
                "vector_instr_with_broadcast not supported such emit_insn!")

        src_buf_0 = ins[0]
        src_buf_1 = ins[1]
        res_buf = outs[0]
        reset_mask_insn(ir_builder, dtype, bits=c0_size)

        ir_builder.emit(
            tvm.call_extern(
                dtype, instr_cmd,
                res_buf.access_ptr("rw", offset=0),
                src_buf_0.access_ptr("rw", offset=0),
                src_buf_1.access_ptr("rw", offset=0),
                1, 1, 1, 1, 8, 8, 8))

        return ir_builder.get()

    loop_offset = op_size

    src_buf_bc = ins[small_buf_index]
    src_buffer = ins[(small_buf_index + 1) % 2]
    res_buffer = outs[0]

    def new_alloc(ir_builder, dtype, shape, name, scope):
        buf_var = ir_builder.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name,
                                     scope=scope, data=buf_var)

        return new_buffer

    if dtype == "float16":
        vector_inst_one_repeat_size = 128
    else:
        vector_inst_one_repeat_size = 64

        tmp_buffer = new_alloc(ir_builder, dtype, (vector_inst_one_repeat_size,),
                               'tmp_buffer', scope=cce.scope_ubuf)

    repeat_times = op_size // vector_inst_one_repeat_size
    remain_size = op_size % vector_inst_one_repeat_size

    max_repeat_times = cce.VECTOR_INST_MAX_REPEAT_TIMES
    count = 0

    with ir_builder.for_range(0, loop_count, name="loop_idx") as loop_idx:
        reset_mask_insn(ir_builder, dtype, bits=c0_size)
        ir_builder.emit(tvm.call_extern(
            dtype, "vadds",
            tmp_buffer.access_ptr("rw", offset=0),
            src_buf_bc.access_ptr("r", offset=loop_idx*c0_size),
            tvm.const(0, dtype=dtype),
            vector_inst_one_repeat_size // c0_size, 1, 1, 2, 0))
        reset_mask_insn(ir_builder, dtype, bits=vector_inst_one_repeat_size)
        if repeat_times > max_repeat_times:
            count = repeat_times // max_repeat_times
            with ir_builder.for_range(0, count, name="iter") as loop:
                repeat_times = repeat_times - max_repeat_times
                offset_value = loop*max_repeat_times*vector_inst_one_repeat_size

                src_0 = tmp_buffer.access_ptr("r", offset=0)
                src_1 = src_buffer.access_ptr(
                    "r", offset=offset_value + loop_idx*loop_offset)

                if instr_cmd in ("vdiv", "vsub") and small_buf_index == 1:
                    ir_builder.emit(
                        tvm.call_extern(
                            dtype, instr_cmd,
                            dst_buffer.access_ptr(
                                "rw", offset=offset_value + loop_idx*loop_offset),
                            src_1, src_0, max_repeat_times, 1, 1, 1, 8, 8, 0))
                else:
                    ir_builder.emit(
                        tvm.call_extern(
                            dtype, instr_cmd,
                            dst_buffer.access_ptr(
                                "rw", offset=offset_value + loop_idx*loop_offset),
                            src_0, src_1, max_repeat_times, 1, 1, 1, 8, 0, 8))

            repeat_times = repeat_times % max_repeat_times

        offset_value = count*max_repeat_times*vector_inst_one_repeat_size
        src_0 = tmp_buffer.access_ptr("r", offset=0)
        src_1 = src_buffer.access_ptr(
            "r", offset=offset_value + loop_idx*loop_offset)

        if repeat_times > 0:
            if instr_cmd in ("vdiv", "vsub") and small_buf_index == 1:
                ir_builder.emit(
                    tvm.call_extern(
                        dtype, instr_cmd,
                        dst_buffer.access_ptr(
                            "rw", offset=offset_value + loop_idx*loop_offset),
                        src_1, src_0, repeat_times, 1, 1, 1, 8, 8, 0))
            else:
                ir_builder.emit(
                    tvm.call_extern(
                        dtype, instr_cmd,
                        dst_buffer.access_ptr(
                            "rw", offset=offset_value + loop_idx*loop_offset),
                        src_0, src_1, repeat_times, 1, 1, 1, 8, 0, 8))

        if remain_size > 0:
            reset_mask_insn(ir_builder, src_buffer.dtype, bits=remain_size)
            offset_value = op_size // vector_inst_one_repeat_size*vector_inst_one_repeat_size
            src_0 = tmp_buffer.access_ptr("r", offset=0)
            src_1 = src_buffer.access_ptr(
                "r", offset=offset_value + loop_idx*loop_offset)

            if instr_cmd in ("vdiv", "vsub") and small_buf_index == 1:
                ir_builder.emit(
                    tvm.call_extern(
                        dtype, instr_cmd,
                        dst_buffer.access_ptr(
                            "rw", offset=offset_value + loop_idx*loop_offset),
                        src_1, src_0, 1, 1, 1, 1, 8, 8, 8))
            else:
                ir_builder.emit(
                    tvm.call_extern(
                        dtype, instr_cmd,
                        dst_buffer.access_ptr(
                            "rw", offset=offset_value + loop_idx*loop_offset),
                        src_0, src_1, 1, 1, 1, 1, 8, 8, 8))

    reset_mask_insn(ir_builder, res_buffer.dtype)

    return ir_builder.get()

@tvm.register_func("tvm.intrin.cce.vector_tuple_reduce_sum_for_bn_update_grad")
def vector_tuple_reduce_sum_for_bn_update_grad(tensor_op):
    """
    tuple reduce sum convert to dichotomy add for bn_update_grad
    :param tensor_op: the stmt
    :return: the intric stmt what we want
    """

    def _post_order_for(tensor_op):
        if isinstance(tensor_op, tvm.stmt.For):
            for_extent_vals.append(tensor_op.extent.value)

    tvm_ib = tvm.ir_builder.create()

    # get input size and loop num
    for_extent_vals = []
    _ = tvm.ir_pass.IRTransform(tensor_op, None, _post_order_for, ["For"])
    input_size = 1
    for i in for_extent_vals:
        input_size = input_size * i

    # get in and out buffer
    ins, _ = cce_util.get_buffer(tensor_op, need_unique=True, need_origin_adress=True)
    _, outs = cce_util.get_buffer(tensor_op, need_unique=True)
    in_buffer_1 = ins[1]

    # calculate vector_inst_one_repeat_size
    dtype = in_buffer_1.dtype
    if dtype == "float16":
        vector_inst_one_repeat_size = 128
    elif dtype == "float32":
        vector_inst_one_repeat_size = 64
    else:
        raise RuntimeError("Batch normalization dtype not supported.")

    # get reduce_axis shape
    if len(for_extent_vals) == 1:
        input_reduce_axis_shape = for_extent_vals[0]
        ub_loop_num = 1
    else:
        input_reduce_axis_shape = for_extent_vals[0]
        ub_loop_num = for_extent_vals[1]

    collapse_loop_num = math.log(input_reduce_axis_shape / vector_inst_one_repeat_size, 2)

    # judge reduce_shape is remaining or not after dichotomy add
    remain_flag = False
    collapse_repeat = 0
    if not collapse_loop_num.is_integer():
        collapse_repeat = int(math.pow(2, int(collapse_loop_num)))
        out_of_collapse_repeat = \
            input_reduce_axis_shape / vector_inst_one_repeat_size - collapse_repeat
        if not out_of_collapse_repeat.is_integer():
            raise RuntimeError("Input size is not aligned:", input_reduce_axis_shape)
        remain_flag = True

    # Do Emit Insn
    def collapse(ir_b, buffer_tensor, current_size):
        repeat = current_size // 2 / vector_inst_one_repeat_size
        tail_flag = False
        if not repeat.is_integer():
            tail_flag = True
        repeat = int(repeat)

        ir_b.emit(tvm.call_extern(
            buffer_tensor.dtype,
            "vadd",
            buffer_tensor.access_ptr("rw", offset=0),
            buffer_tensor.access_ptr("r", offset=0),
            buffer_tensor.access_ptr("r", offset=8),
            repeat, 1, 2, 2, 8, 16, 16))

        # solve tail vadd
        if tail_flag:
            tail_mask = (current_size - repeat * 2 * vector_inst_one_repeat_size) // 2
            reset_mask_insn(tvm_ib, dtype, tail_mask)
            ir_b.emit(tvm.call_extern(
                buffer_tensor.dtype,
                "vadd",
                buffer_tensor.access_ptr("rw", offset=repeat * vector_inst_one_repeat_size),
                buffer_tensor.access_ptr("r", offset=repeat * 2 * vector_inst_one_repeat_size),
                buffer_tensor.access_ptr("r", offset=repeat * 2 * vector_inst_one_repeat_size + 8),
                1, 1, 2, 2, 0, 0, 0))
            reset_mask_insn(tvm_ib, dtype)
        return current_size // 2

    in_index = 1
    for out_buffer in outs:
        # emit vadd
        cur_size = input_size
        loop = 0
        while loop < int(collapse_loop_num):
            cur_size = collapse(tvm_ib, ins[in_index], cur_size)
            loop += 1

        if remain_flag:
            # solve remain repeat
            mask_bits = input_reduce_axis_shape / collapse_repeat - vector_inst_one_repeat_size
            add_repeat_stride = int(8 + mask_bits / 8)
            reset_mask_insn(tvm_ib, dtype, mask_bits)
            tvm_ib.emit(tvm.call_extern(
                dtype,
                "vadd",
                ins[in_index].access_ptr("rw", offset=0),
                ins[in_index].access_ptr("r", offset=0),
                ins[in_index].access_ptr("r", offset=vector_inst_one_repeat_size),
                ub_loop_num, 1, 1, 1, add_repeat_stride, add_repeat_stride, add_repeat_stride))

            # emit vcadd for remain
            reset_mask_insn(tvm_ib, dtype)
            tvm_ib.emit(tvm.call_extern(
                dtype,
                "vcadd",
                out_buffer.access_ptr("rw", offset=0),
                ins[in_index].access_ptr("r", offset=0), ub_loop_num, 1, 1, add_repeat_stride))
        else:
            # emit vcadd for no remain
            tvm_ib.emit(tvm.call_extern(
                dtype,
                "vcadd",
                out_buffer.access_ptr("rw", offset=0),
                ins[in_index].access_ptr("r", offset=0), ub_loop_num, 1, 1, 8))

        in_index += 2

    return tvm_ib.get()

@tvm.register_func("tvm.intrin.cce.vector_dichotomy_add_for_bn_reduce")
def vector_dichotomy_add_for_bn_reduce(tensor_op):
    """
    vector dichotomy add for bn_reduce
    :param tensor_op:
    :return:
    """
    def _post_order_for(tensor_op):
        if isinstance(tensor_op, tvm.stmt.For):
            for_extent_vals.append(tensor_op.extent.value)
            for_vars.append(tensor_op.loop_var)

    def new_alloc(tvm_ib, dtype, shape, name, scope):
        buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)
        return new_buffer

    instr_cmd = "vadd"

    ins, outs = cce_util.get_buffer(tensor_op)
    tvm_ib = tvm.ir_builder.create()

    for_extent_vals = []
    for_vars = []

    _ = tvm.ir_pass.IRTransform(tensor_op, None, _post_order_for, ["For"])

    if len(ins) != 4 and len(outs) != 2:
        raise RuntimeError("Batch normalization not support such emit_insn.")

    if not (ins[0].dtype == ins[1].dtype and ins[1].dtype == ins[2].dtype and
            ins[2].dtype == ins[3].dtype):
        raise RuntimeError("Batch normalization not support such emit_insn.")

    none_reduce_var = for_vars[0].name
    if none_reduce_var.find('k') != -1:
        raise RuntimeError("Dichotomy add not support reduce last axis.")

    sum_x_dst_buffer = outs[0]
    sum_x_src_buffer = ins[1]
    square_x_dst_buffer = outs[1]
    square_x_src_buffer = ins[3]

    sum_x_orignal_src_buffer = ins[1]
    square_x_orignal_src_buffer = ins[3]

    dtype = ins[0].dtype
    if dtype == "float16":
        vector_inst_one_repeat_size = 128
        dtype_size = 2
    elif dtype == "float32":
        vector_inst_one_repeat_size = 64
        dtype_size = 4
    else:
        raise RuntimeError("Batch normalization dtype not supported.")

    last_none_reduce_size = 1
    for i, _ in enumerate(for_vars):
        var = for_vars[i].name
        if var.find('k') != -1:
            break
        last_none_reduce_size *= for_extent_vals[i]
    block_size = 32
    if last_none_reduce_size > vector_inst_one_repeat_size or \
            vector_inst_one_repeat_size % last_none_reduce_size != 0 or \
            last_none_reduce_size * dtype_size % block_size != 0:
        raise RuntimeError("Batch normalization not supported such emit_insn.")

    op_size = 1
    for i in for_extent_vals:
        op_size = op_size * i

    total_repeats = op_size // vector_inst_one_repeat_size

    # dichotomy buffer
    sum_x_temp_buffer = new_alloc(tvm_ib, dtype, (op_size // 2,),
                                  'sum_x_temp_buffer', scope=cce.scope_ubuf)
    square_x_temp_buffer = new_alloc(tvm_ib, dtype, (op_size // 2,),
                                     'square_x_temp_buffer', scope=cce.scope_ubuf)

    if total_repeats > 0:
        dichotomy_times = math.ceil((math.log(total_repeats, 2)))
    else:
        dichotomy_times = 0
    repeats = total_repeats
    loop_tail = 4
    reset_mask_insn(tvm_ib, dtype, bits=vector_inst_one_repeat_size)
    while dichotomy_times > loop_tail:
        tvm_ib.emit(tvm.call_extern(dtype, instr_cmd,
                                    sum_x_temp_buffer.access_ptr("rw", offset=0),
                                    sum_x_src_buffer.access_ptr("r", offset=0),
                                    sum_x_src_buffer.access_ptr(
                                        "r", offset=vector_inst_one_repeat_size),
                                    repeats // 2, 1, 1, 1, 8, 16, 16))

        tvm_ib.emit(tvm.call_extern(dtype, instr_cmd,
                                    square_x_temp_buffer.access_ptr("rw", offset=0),
                                    square_x_src_buffer.access_ptr("r", offset=0),
                                    square_x_src_buffer.access_ptr(
                                        "r", offset=vector_inst_one_repeat_size),
                                    repeats // 2, 1, 1, 1, 8, 16, 16))

        if repeats % 2 != 0:
            tvm_ib.emit(tvm.call_extern(dtype, instr_cmd,
                                        sum_x_temp_buffer.access_ptr("rw", offset=0),
                                        sum_x_src_buffer.access_ptr(
                                            "r", offset=(repeats//2)*2*vector_inst_one_repeat_size),
                                        sum_x_temp_buffer.access_ptr("r", offset=0),
                                        repeats % 2, 1, 1, 1, 0, 8, 0))
            tvm_ib.emit(tvm.call_extern(dtype, instr_cmd,
                                        square_x_temp_buffer.access_ptr("rw", offset=0),
                                        square_x_src_buffer.access_ptr(
                                            "r", offset=(repeats//2)*2*vector_inst_one_repeat_size),
                                        square_x_temp_buffer.access_ptr("r", offset=0),
                                        repeats % 2, 1, 1, 1, 0, 8, 0))

        sum_x_temp_buffer, sum_x_src_buffer = sum_x_src_buffer, sum_x_temp_buffer

        square_x_temp_buffer, square_x_src_buffer = square_x_src_buffer, square_x_temp_buffer

        repeats = repeats // 2
        dichotomy_times = dichotomy_times - 1

    if repeats > 1:
        tvm_ib.emit(tvm.call_extern(dtype, instr_cmd,
                                    sum_x_src_buffer.access_ptr("rw", offset=0),
                                    sum_x_src_buffer.access_ptr(
                                        "r", offset=vector_inst_one_repeat_size),
                                    sum_x_src_buffer.access_ptr("r", offset=0),
                                    repeats - 1, 1, 1, 1, 0, 8, 0))
        tvm_ib.emit(tvm.call_extern(dtype, instr_cmd,
                                    square_x_src_buffer.access_ptr("rw", offset=0),
                                    square_x_src_buffer.access_ptr(
                                        "r", offset=vector_inst_one_repeat_size),
                                    square_x_src_buffer.access_ptr("r", offset=0),
                                    repeats - 1, 1, 1, 1, 0, 8, 0))

    remain_size = last_none_reduce_size
    reset_mask_insn(tvm_ib, dtype, bits=remain_size)
    block_size = 32
    block_num = last_none_reduce_size * dtype_size // block_size
    # sum_x
    if total_repeats > 0:
        combine_repeat = vector_inst_one_repeat_size // last_none_reduce_size
        tvm_ib.emit(tvm.call_extern(dtype, instr_cmd,
                                    sum_x_dst_buffer.access_ptr("rw", offset=0),
                                    sum_x_src_buffer.access_ptr("r", offset=0),
                                    sum_x_dst_buffer.access_ptr("r", offset=0),
                                    combine_repeat, 1, 1, 1, 0, block_num, 0))

        # square_x
        tvm_ib.emit(tvm.call_extern(dtype, instr_cmd,
                                    square_x_dst_buffer.access_ptr("rw", offset=0),
                                    square_x_src_buffer.access_ptr("r", offset=0),
                                    square_x_dst_buffer.access_ptr("r", offset=0),
                                    combine_repeat, 1, 1, 1, 0, block_num, 0))

    # tail
    tail_nums = (op_size % vector_inst_one_repeat_size) // last_none_reduce_size
    if tail_nums > 0:
        tvm_ib.emit(tvm.call_extern(dtype, instr_cmd,
                                    sum_x_dst_buffer.access_ptr("rw", offset=0),
                                    sum_x_orignal_src_buffer.access_ptr(
                                        "r", offset=total_repeats*vector_inst_one_repeat_size),
                                    sum_x_dst_buffer.access_ptr("r", offset=0),
                                    tail_nums, 1, 1, 1, 0, block_num, 0))
        tvm_ib.emit(tvm.call_extern(dtype, instr_cmd,
                                    square_x_dst_buffer.access_ptr("rw", offset=0),
                                    square_x_orignal_src_buffer.access_ptr(
                                        "r", offset=total_repeats*vector_inst_one_repeat_size),
                                    square_x_dst_buffer.access_ptr("r", offset=0),
                                    tail_nums, 1, 1, 1, 0, block_num, 0))
    reset_mask_insn(tvm_ib, dtype, bits=128)
    stmt = tvm_ib.get()

    return stmt


@tvm.register_func("tvm.intrin.cce.vector_dichotomy_add")
def vector_dichotomy_add(tensor_op):
    """
    vector dichotomy add
    :param tensor_op:
    :return:
    """
    def _post_order_for(tensor_op):
        if isinstance(tensor_op, tvm.stmt.For):
            for_extent_vals.append(tensor_op.extent.value)
            for_vars.append(tensor_op.loop_var)

    def new_alloc(tvm_ib, dtype, shape, name, scope):
        buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)
        return new_buffer

    instr_cmd = "vadd"

    ins, outs = cce_util.get_buffer(tensor_op)
    tvm_ib = tvm.ir_builder.create()

    for_extent_vals = []
    for_vars = []
    _ = tvm.ir_pass.IRTransform(tensor_op, None, _post_order_for, ["For"])

    if len(ins) != 2 and len(outs) != 1:
        raise RuntimeError("Dichotomy add not support such emit_insn.")
    if not ins[0].dtype == ins[1].dtype:
        raise RuntimeError("Dichotomy add not support such emit_insn.")

    if not for_vars:
        raise RuntimeError("Dichotomy add not support such emit_insn.")

    none_reduce_var = for_vars[0].name
    if none_reduce_var.find('k') != -1:
        raise RuntimeError("Dichotomy add not support reduce last axis.")

    x_dst_buffer = outs[0]
    x_src_buffer = ins[1]

    x_orignal_src_buffer = ins[1]

    dtype = ins[0].dtype
    if dtype == "float16":
        vector_inst_one_repeat_size = 128
        dtype_size = 2
    elif dtype == "float32":
        vector_inst_one_repeat_size = 64
        dtype_size = 4
    else:
        raise RuntimeError("Dichotomy add dtype not supported.")
    last_none_reduce_size = 1
    for i, _ in enumerate(for_vars):
        var = for_vars[i].name
        if var.find('k') != -1:
            break
        last_none_reduce_size *= for_extent_vals[i]
    block_size = 32
    if last_none_reduce_size > vector_inst_one_repeat_size or \
            vector_inst_one_repeat_size % last_none_reduce_size != 0 or \
            last_none_reduce_size * dtype_size % block_size != 0:
        raise RuntimeError("Dichotomy add not supported such emit_insn.")

    op_size = 1
    for i in for_extent_vals:
        op_size = op_size * i

    total_repeats = op_size // vector_inst_one_repeat_size

    # dichotomy buffer
    x_temp_buffer = new_alloc(tvm_ib, dtype, (op_size // 2,), 'x_temp_buffer', scope=cce.scope_ubuf)

    if total_repeats > 0:
        dichotomy_times = math.ceil((math.log(total_repeats, 2)))
    else:
        dichotomy_times = 0
    repeats = total_repeats
    loop_tail = 3
    while dichotomy_times > loop_tail:
        tvm_ib.emit(tvm.call_extern(dtype, instr_cmd,
                                    x_temp_buffer.access_ptr("rw", offset=0),
                                    x_src_buffer.access_ptr("r", offset=0),
                                    x_src_buffer.access_ptr("r",
                                                            offset=vector_inst_one_repeat_size),
                                    repeats // 2, 1, 1, 1, 8, 16, 16))

        if repeats % 2 != 0:
            tvm_ib.emit(tvm.call_extern(dtype, instr_cmd,
                                        x_temp_buffer.access_ptr("rw", offset=0),
                                        x_src_buffer.access_ptr(
                                            "r", offset=(repeats//2)*2*vector_inst_one_repeat_size),
                                        x_temp_buffer.access_ptr("r", offset=0),
                                        repeats % 2, 1, 1, 1, 0, 8, 0))

        x_temp_buffer, x_src_buffer = x_src_buffer, x_temp_buffer

        repeats = repeats // 2
        dichotomy_times = dichotomy_times - 1
    if repeats > 1:
        tvm_ib.emit(tvm.call_extern(dtype, instr_cmd,
                                    x_src_buffer.access_ptr("rw", offset=0),
                                    x_src_buffer.access_ptr("r",
                                                            offset=vector_inst_one_repeat_size),
                                    x_src_buffer.access_ptr("r", offset=0),
                                    repeats - 1, 1, 1, 1, 0, 8, 0))

    remain_size = last_none_reduce_size
    reset_mask_insn(tvm_ib, dtype, bits=remain_size)
    block_size = 32
    block_num = last_none_reduce_size * dtype_size // block_size
    # sum_x
    if total_repeats > 0:
        combine_repeat = vector_inst_one_repeat_size // last_none_reduce_size
        tvm_ib.emit(tvm.call_extern(dtype, instr_cmd,
                                    x_dst_buffer.access_ptr("rw", offset=0),
                                    x_src_buffer.access_ptr("r", offset=0),
                                    x_dst_buffer.access_ptr("r", offset=0),
                                    combine_repeat, 1, 1, 1, 0, block_num, 0))


    # tail
    tail_nums = (op_size % vector_inst_one_repeat_size) // last_none_reduce_size
    if tail_nums > 0:
        tvm_ib.emit(tvm.call_extern(dtype, instr_cmd,
                                    x_dst_buffer.access_ptr("rw", offset=0),
                                    x_orignal_src_buffer.access_ptr(
                                        "r", offset=total_repeats*vector_inst_one_repeat_size),
                                    x_dst_buffer.access_ptr("r", offset=0),
                                    tail_nums, 1, 1, 1, 0, block_num, 0))
    reset_mask_insn(tvm_ib, dtype, bits=128)
    stmt = tvm_ib.get()

    return stmt


@tvm.register_func("tvm.intrin.cce.vector_dichotomy_reduce")
def vector_dichotomy_reduce(tensor_op):
    """
    vector dichotomy reduce
    :param tensor_op:
    :return:
    """
    error_meg_dichotomy_reduce = "dichotomy reduce not supported such emit_insn."

    def get_buffer(size, dtype, buffer_var):
        return tvm.decl_buffer(size, dtype,
                               name=buffer_var.name,
                               data=buffer_var,
                               scope=cce.scope_ubuf,
                               data_alignment=bytes_per_block)

    def new_alloc(tvm_ib, dtype, shape, name, scope):
        # pylint: disable=protected-access
        buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
        return get_buffer(shape, dtype, buf_var._buffer_var)

    def _post_order_reduce_op(tensor_op):
        if isinstance(tensor_op, tvm.stmt.Store):
            if isinstance(tensor_op.value, tvm.expr.FloatImm):
                return
            if isinstance(tensor_op.value, tvm.expr.Add):
                instr_cmd.append("vadd")
                instr_cmd.append("vcgadd")
                instr_cmd.append("vcadd")
            elif isinstance(tensor_op.value, tvm.expr.Max):
                instr_cmd.append("vmax")
                instr_cmd.append("vcgmax")
                instr_cmd.append("vcmax")
                instr_cmd.append("vcpadd")
            elif isinstance(tensor_op.value, tvm.expr.Min):
                instr_cmd.append("vmin")
                instr_cmd.append("vcgmin")
                instr_cmd.append("vcmin")
                instr_cmd.append("vcpadd")
            else:
                raise RuntimeError(error_meg_dichotomy_reduce)

    def _post_order_for(tensor_op):
        if isinstance(tensor_op, tvm.stmt.For):
            for_vars_extent[tensor_op.loop_var] = tensor_op.extent.value

    def _post_order_buffer_and_index(tensor_op):
        if isinstance(tensor_op, tvm.expr.Load):
            load_index[tensor_op.buffer_var] = tensor_op.index
            buf = get_buffer(1, tensor_op.dtype, tensor_op.buffer_var)
            buffers.append(buf)

    def get_stride_info(index_op):
        stride_map = {}
        stride_list = []
        def _post_order_stride(tensor_op):
            if isinstance(tensor_op, tvm.expr.Mul):
                if isinstance(tensor_op.a, tvm.expr.Var) and \
                        isinstance(tensor_op.b, tvm.expr.IntImm):
                    stride_map[tensor_op.a] = tensor_op.b.value
                else:
                    raise RuntimeError(error_meg_dichotomy_reduce)
            if isinstance(tensor_op, tvm.expr.Var):
                if tensor_op not in stride_map.keys():
                    stride_map[tensor_op] = 1
                # check index
                if tensor_op not in for_vars_extent.keys():
                    raise RuntimeError(error_meg_dichotomy_reduce)

        # recursion visit for index
        tvm.ir_pass.PostOrderVisit(index_op, _post_order_stride)
        # handle for loop order by src stride
        index_stride_cmp_function = lambda x, y: y[1] - x[1]
        for i in stride_map:
            stride_list.append([i, stride_map[i]])
        stride_list.sort(key=cmp_to_key(index_stride_cmp_function))
        # stride check
        for i in range(len(stride_list) - 1):
            expect_stride = stride_list[i+1][1] * for_vars_extent[stride_list[i+1][0]]
            if stride_list[i][1] != expect_stride:
                raise RuntimeError(error_meg_dichotomy_reduce)
        return stride_map, stride_list

    def set_vcpadd_mask(nums):
        def get_vcpadd_mask_value(mask_n):
            mask_n //= 2
            value = 0
            while mask_n > 0:
                mask_n -= 1
                value = (value << 2) | 1
            return value
        mask_max_value = 64
        if nums > mask_max_value:
            mask1 = get_vcpadd_mask_value(mask_max_value)
            mask2 = get_vcpadd_mask_value(nums - mask_max_value)
        else:
            mask1 = get_vcpadd_mask_value(nums)
            mask2 = 0
        tvm_ib.emit(tvm.call_extern(CALL_TYPE, "set_vector_mask",
                                    tvm.const(mask2, dtype="uint64"),
                                    tvm.const(mask1, dtype="uint64")))

    tvm_ib = tvm.ir_builder.create()

    # public info
    i_var = 0
    i_ext = 1
    dst = 0
    src = 1
    block_per_repeat = 8
    bytes_per_block = 32
    bytes_per_repeat = 256
    repeat_times_max = [2**8 - 1, 2**16 - 1]
    data_type_wides = {"float16" : 2, "float32" : 4}
    full_mask_nums = 128
    least_cycle_intrin = "vmax"
    least_cycle_rpt = 2**8 - 1

    instr_cmd = []
    for_vars_extent = {}
    buffers = []
    load_index = {}
    tvm.ir_pass.PostOrderVisit(tensor_op, _post_order_reduce_op)
    tvm.ir_pass.PostOrderVisit(tensor_op, _post_order_for)
    tvm.ir_pass.PostOrderVisit(tensor_op, _post_order_buffer_and_index)
    dst_index_stride_map, dst_index_stride_order = get_stride_info(load_index[buffers[dst].data])
    src_index_stride_map, src_index_stride_order = get_stride_info(load_index[buffers[src].data])

    dtype = buffers[dst].dtype
    if dtype not in data_type_wides.keys():
        raise RuntimeError(error_meg_dichotomy_reduce)
    repeat_per_nums = bytes_per_repeat // data_type_wides[dtype]
    block_per_nums = repeat_per_nums // block_per_repeat

    # 5 mode here
    # reduce in front not support
    # reduce in middle not support
    # reduce at tail not support
    # reduce both in front and at tail
    # reduce all, reduce non, not support
    reduce_in_front = src_index_stride_order[0][i_var] not in dst_index_stride_map.keys()
    reduce_at_tail = src_index_stride_order[-1][i_var] not in dst_index_stride_map.keys()
    reduce_all = len(dst_index_stride_order) == 0
    reduce_none = len(dst_index_stride_order) == len(src_index_stride_order)
    # dispatch info
    if reduce_all or reduce_none:
        raise RuntimeError(error_meg_dichotomy_reduce)

    if reduce_in_front and reduce_at_tail:
        pass
    elif reduce_in_front and not reduce_at_tail:
        raise RuntimeError(error_meg_dichotomy_reduce)
    elif not reduce_in_front and reduce_at_tail:
        raise RuntimeError(error_meg_dichotomy_reduce)
    else:
        raise RuntimeError(error_meg_dichotomy_reduce)

    # private info, reduce both in front and at tail
    # get 3 part, reduce_tail, reduce_const, reduce_front
    # case like, (64,32,16) reduce in (0,2)
    # reduce_front is 64, reduce_const is 32, reduce_tail is 16
    reduce_tail = src_index_stride_map[dst_index_stride_order[-1][i_var]]
    temp_size = src_index_stride_order[0][i_ext]
    for s_item in src_index_stride_order:
        if s_item[i_var].same_as(dst_index_stride_order[0][i_var]):
            break
        temp_size = s_item[i_ext]
    reduce_const = temp_size // reduce_tail
    all_size = src_index_stride_order[0][i_ext] * for_vars_extent[src_index_stride_order[0][i_var]]
    reduce_front = all_size // (reduce_tail * reduce_const)

    # last axis length limit
    if reduce_tail > repeat_per_nums or reduce_tail % block_per_nums:
        raise RuntimeError(error_meg_dichotomy_reduce)

    # part 1, (64,32,16) reduce into (1,32,16)
    reset_mask_insn(tvm_ib, CALL_TYPE, bits=repeat_per_nums)
    if reduce_front > 1:
        dichotomy_size = math.ceil(reduce_front / 2) * 2
        dichotomy_per_size = reduce_tail * reduce_const
        buffer_size = dichotomy_per_size * dichotomy_size // 2
        dichotomy_reduce_temp_buffer = new_alloc(tvm_ib, dtype, buffer_size,
                                                 'dichotomy_reduce_temp_buffer',
                                                 scope=cce.scope_ubuf)
        # special for bert Nz
        if dtype == "float32" and len(instr_cmd) == 3 and \
                reduce_tail == 16 and reduce_front == 16:
            tvm_ib.emit(tvm.call_extern(CALL_TYPE, instr_cmd[0],
                                        dichotomy_reduce_temp_buffer.access_ptr("w", offset=0),
                                        buffers[src].access_ptr("r", offset=buffer_size),
                                        buffers[src].access_ptr("r", offset=0),
                                        reduce_const * 2, 1, 1, 1, 8, 8, 8))
            tvm_ib.emit(tvm.call_extern(CALL_TYPE, instr_cmd[0],
                                        dichotomy_reduce_temp_buffer.access_ptr("w", offset=0),
                                        dichotomy_reduce_temp_buffer.access_ptr("r", offset=8),
                                        dichotomy_reduce_temp_buffer.access_ptr("r", offset=0),
                                        reduce_const, 2, 2, 2, 16, 16, 16))
            tvm_ib.emit(tvm.call_extern(CALL_TYPE, instr_cmd[2],
                                        buffers[dst].access_ptr("w", offset=0),
                                        dichotomy_reduce_temp_buffer.access_ptr("r", offset=0),
                                        reduce_const, 1, 2 * reduce_const, 2))
            reset_mask_insn(tvm_ib, CALL_TYPE, bits=full_mask_nums)
            return tvm_ib.get()

        # general case
        while dichotomy_size > 1:
            # case like axis (5) in dichotomy operation
            # dichotomy_size is 6, intact_size is 3, handle_size is 2, reserve_size is 1
            intact_size = math.ceil(dichotomy_size / 2)
            handle_size = dichotomy_size - intact_size
            intact_nums = intact_size * dichotomy_per_size
            emit_insn_in_one_operation([dichotomy_reduce_temp_buffer, buffers[src], buffers[src]],
                                       [0, intact_nums, 0],
                                       instr_cmd[0], handle_size * dichotomy_per_size,
                                       repeat_times_max[0], repeat_per_nums, tvm_ib)
            # for first times, lack part need move to temp buffer
            if dichotomy_size > reduce_front:
                reserve_size = intact_size - handle_size
                init_offset = handle_size * dichotomy_per_size
                emit_insn_in_one_operation([dichotomy_reduce_temp_buffer,
                                            buffers[src], buffers[src]],
                                           [init_offset, intact_nums + init_offset, init_offset],
                                           least_cycle_intrin, reserve_size * dichotomy_per_size,
                                           least_cycle_rpt, repeat_per_nums, tvm_ib)

            buffers[src] = dichotomy_reduce_temp_buffer
            dichotomy_size = intact_size

    # part 2, (m, 128/64) reduce into (m)
    repeat_stride = reduce_tail // block_per_nums
    temp_repeats = reduce_const
    if temp_repeats > repeat_times_max[1]:
        # unsupport yet
        raise RuntimeError(error_meg_dichotomy_reduce)
    # use vcg model
    if dtype == "float16" and reduce_tail == block_per_nums:
        full_part = reduce_const // block_per_repeat
        rest_part = reduce_const % block_per_repeat
        if full_part:
            reset_mask_insn(tvm_ib, CALL_TYPE, bits=repeat_per_nums)
            tvm_ib.emit(tvm.call_extern(dtype, instr_cmd[1],
                                        buffers[dst].access_ptr("w", offset=0),
                                        buffers[src].access_ptr("r", offset=0),
                                        full_part, 1, 1, 8))
        if rest_part:
            reset_mask_insn(tvm_ib, CALL_TYPE, bits=rest_part * block_per_nums)
            tvm_ib.emit(tvm.call_extern(dtype, instr_cmd[1],
                                        buffers[dst].access_ptr(
                                            "w", offset=full_part * block_per_repeat),
                                        buffers[src].access_ptr(
                                            "r", offset=full_part * repeat_per_nums),
                                        1, 1, 1, 8))
    else:
        # handle sum
        if len(instr_cmd) == 3:
            # use vc model
            reset_mask_insn(tvm_ib, CALL_TYPE, bits=reduce_tail)
            tvm_ib.emit(tvm.call_extern(dtype, instr_cmd[2],
                                        buffers[dst].access_ptr("w", offset=0),
                                        buffers[src].access_ptr("r", offset=0),
                                        temp_repeats, 1, 1, repeat_stride))
        # handle max/min
        else:
            # use vc + vcpadd model
            temp_buffer_size = reduce_const * 2
            vcpadd_temp_buffer = new_alloc(tvm_ib, dtype, temp_buffer_size,
                                           'vcpadd_temp_buffer', scope=cce.scope_ubuf)
            reset_mask_insn(tvm_ib, CALL_TYPE, bits=reduce_tail)
            tvm_ib.emit(tvm.call_extern(dtype, instr_cmd[2],
                                        vcpadd_temp_buffer.access_ptr("w", offset=0),
                                        buffers[src].access_ptr("r", offset=0),
                                        temp_repeats, 1, 1, repeat_stride))
            full_part = temp_buffer_size // repeat_per_nums
            rest_part = temp_buffer_size % repeat_per_nums
            if full_part:
                set_vcpadd_mask(repeat_per_nums)
                tvm_ib.emit(tvm.call_extern(dtype, instr_cmd[3],
                                            buffers[dst].access_ptr("w", offset=0),
                                            vcpadd_temp_buffer.access_ptr("r", offset=0),
                                            full_part, 1, 1, 8))
            if rest_part:
                set_vcpadd_mask(rest_part)
                tvm_ib.emit(tvm.call_extern(dtype, instr_cmd[3],
                                            buffers[dst].access_ptr(
                                                "w", offset=full_part * repeat_per_nums // 2),
                                            vcpadd_temp_buffer.access_ptr(
                                                "r", offset=full_part * repeat_per_nums),
                                            1, 1, 1, 8))

    reset_mask_insn(tvm_ib, CALL_TYPE, bits=full_mask_nums)
    return tvm_ib.get()

def emit_insn_in_one_operation(op_buffers, init_offset, op_cmd, op_nums,
                               repeat_limit, repeat_per_nums, ir_b):
    '''
    :param op_buffers: buffers in intrinsic
    :param init_offset: init offset in buffer
    :param op_cmd: intrinsic name
    :param op_nums: handle numbers
    :param repeat_limit: max repeat times
    :param ir_b: ir build
    :return:
    '''
    full_part = op_nums // repeat_per_nums
    rest_part = op_nums % repeat_per_nums
    if full_part:
        reset_mask_insn(ir_b, CALL_TYPE, bits=repeat_per_nums)
        if full_part > repeat_limit:
            full_rpt = full_part // repeat_limit * repeat_limit
            rest_rpt = full_part % repeat_limit
            size_per_full = repeat_limit * repeat_per_nums
            cur_times = 0
            while cur_times < full_rpt:
                ac_offset = size_per_full * cur_times
                ir_b.emit(tvm.call_extern(CALL_TYPE, op_cmd,
                                          op_buffers[0].access_ptr("w", offset=init_offset[0] +
                                                                   ac_offset),
                                          op_buffers[1].access_ptr("r", offset=init_offset[1] +
                                                                   ac_offset),
                                          op_buffers[2].access_ptr("r", offset=init_offset[2] +
                                                                   ac_offset),
                                          repeat_limit, 1, 1, 1, 8, 8, 8))
                cur_times += 1
            if rest_part:
                ac_offset = size_per_full * cur_times
                ir_b.emit(tvm.call_extern(CALL_TYPE, op_cmd,
                                          op_buffers[0].access_ptr("w", offset=init_offset[0] +
                                                                   ac_offset),
                                          op_buffers[1].access_ptr("r", offset=init_offset[1] +
                                                                   ac_offset),
                                          op_buffers[2].access_ptr("r", offset=init_offset[2] +
                                                                   ac_offset),
                                          rest_rpt, 1, 1, 1, 8, 8, 8))
        else:
            ir_b.emit(tvm.call_extern(CALL_TYPE, op_cmd,
                                      op_buffers[0].access_ptr("w", offset=init_offset[0]),
                                      op_buffers[1].access_ptr("r", offset=init_offset[1]),
                                      op_buffers[2].access_ptr("r", offset=init_offset[2]),
                                      full_part, 1, 1, 1, 8, 8, 8))
    if rest_part:
        reset_mask_insn(ir_b, CALL_TYPE, bits=rest_part)
        ac_offset = full_part * repeat_per_nums
        ir_b.emit(tvm.call_extern(CALL_TYPE, op_cmd,
                                  op_buffers[0].access_ptr("w", offset=init_offset[0] + ac_offset),
                                  op_buffers[1].access_ptr("r", offset=init_offset[1] + ac_offset),
                                  op_buffers[2].access_ptr("r", offset=init_offset[2] + ac_offset),
                                  1, 1, 1, 1, 8, 8, 8))

OUT_REDUCE_BUFF_FP32 = 32768 # 32kB
@tvm.register_func("tvm.intrin.cce.reduce_2_3_axis_reduce_sum_optimal")
def reduce_sum_5d_2_3_axis_optimal(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_reduce_2_3_axis(tensor_op, "vadd")

def vec_reduce_2_3_axis(tensor_op, intrinsic_cmd, args=None):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    # op_len = cce_util.get_op_lenth(tensor_op)
    _, tile_size, _ = cce_util.get_opshape_tilesize(tensor_op)

    ins, outs = cce_util.get_buffer(tensor_op, need_origin_adress=True)
    # ins, outs = cce_util.get_dma_origin_buffer(tensor_op)
    dtype = outs[0].dtype
    if dtype not in ('float16', 'float32'):
        raise ValueError("only support float16/float32 while dtype is %s"%dtype)

    ir_builder = tvm.ir_builder.create()
    reset_mask = []

    # 5HD
    if intrinsic_cmd == "vadd":
        reduce_2_3_cmd_factory(ir_builder, intrinsic_cmd, ins, outs, tile_size, reset_mask)
    elif intrinsic_cmd in ("vadd_4d", "vmean_4d"):
        reduce_2_3_cmd_factory_4d(ir_builder, intrinsic_cmd, ins, outs, tile_size, reset_mask)
    return ir_builder.get()

def reduce_2_3_cmd_factory(ir_builder, intrinsic_cmd, ins, outs, tile_size, reset_mask):
    """
    factory function for generate commond , only support reduce_sum and reduce_mean

    ir_builder: instance of ir_builder

    intrinsic_cmd : string
        commond type

    ins : list
        contains source buffers

    outs : list
        contains dst buffers

    tile_size : list
        contain shape info
        e.g. [32*4, 64, 7, 7, 16] -> [32*4, 64, 16]
        tile_size= [16, 7, 7, 64, 7, 7, 16, 64, 16, 64, 4]

    reset_mask : int or bool
        if reset_mask == True:
            means want to add reset mask to 128 at the end of commond if mask is changed,
            and pipeline id follows pipeline_count_list changes
        if reset_mask == False:
            means NOT want to add reset mask to 128 at the end of commond
        if reset_mask is int:
            means want to add reset mask to 128 at the end of commond if mask is changed,
            and pipeline id is reset_mask*8

    extern_args : list
        external args in VS or broadcast commond

    args : list
        commond args like the strides in vector commond
    """
    # tile_size: [16, 7, 7, 64, 7, 7, 16, 64, 16, 64, 4]
    reduce_type = len(tile_size)
    reduce_type_map = {11: "single_reduce_sum_float32",
                       17: "cast_single_reduce_sum"}
    for i in reduce_type_map:
        if reduce_type == i:
            shape_0 = tile_size[-1]
            break
        if reduce_type == i - 1:
            shape_0 = 1
            reduce_type = i

    type_len_map = {"float16": 1,
                    "float32": 2}
    dtype = outs[0].dtype
    shape = [shape_0, tile_size[3], tile_size[2], tile_size[1], tile_size[0]]
    buffer_size = shape[0] * shape[1] * shape[3] * shape[4]
    src_buffer = ins[0]
    for i in outs:
        dst_buffer = i

    def new_alloc(ir_builder, dtype, shape, name, scope):
        """
        new_alloc
        """
        buf_var = ir_builder.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)
        return new_buffer

    if reduce_type_map[reduce_type] == "single_reduce_sum_float32" and dtype == "float32":
        ub_tmp_buffer \
            = new_alloc(ir_builder, dtype, (buffer_size, ), 'ub_tmp_buffer', scope=cce.scope_ubuf)
        ub_ping_buffer \
            = new_alloc(ir_builder, dtype, (buffer_size, ), 'ub_ping_buffer', scope=cce.scope_ubuf)
        ub_pong_buffer \
            = new_alloc(ir_builder, dtype, (buffer_size, ), 'ub_pong_buffer', scope=cce.scope_ubuf)
        reset_mask_insn(ir_builder, dtype)

        # init zero ub_buffer
        repeat_times = (buffer_size * type_len_map[dtype]) // 128
        ub_offset = 0
        while repeat_times > 0:
            ir_builder.emit(
                tvm.call_extern(dtype, "vmuls",
                                ub_tmp_buffer.access_ptr("rw", offset=ub_offset),
                                ub_tmp_buffer.access_ptr("rw", offset=ub_offset),
                                tvm.const(0, dtype),
                                min(255, repeat_times), 1, 1, 8, 8))
            repeat_times -= 255
            ub_offset += 255 * 128 // type_len_map[dtype]

        thread_block = get_emitinsn_params("thread_block")
        dst_offset = thread_block * shape[0] * shape[1] * shape[4]
        src_offset = thread_block * shape[0] * shape[1] * shape[2] * shape[3] * shape[4]

        # copy in + loop1
        dma_offset = 0
        sid = 0
        n_burst = shape[0] * shape[1]
        len_burst = shape[3] * shape[4] * type_len_map[dtype] // 16
        src_stride = (shape[2] - 1) * shape[3] * shape[4] * type_len_map[dtype] // 16
        dst_stride = 0
        for i in range(shape[2]):
            # ping
            if i % 2 == 0:
                ub_pingpong_buffer = ub_ping_buffer
            else:
                ub_pingpong_buffer = ub_pong_buffer
            src_addr = src_buffer.access_ptr('r', offset=(src_offset + dma_offset))
            ir_builder.emit(
                tvm.call_extern(dtype, "copy_gm_to_ubuf",
                                ub_pingpong_buffer.access_ptr("rw", offset=0), src_addr,
                                sid, n_burst, len_burst, src_stride, dst_stride))
            dma_offset += shape[3] * shape[4]

            repeat_times = (buffer_size * type_len_map[dtype]) // 128
            ub_offset = 0
            while repeat_times > 0:
                ir_builder.emit(
                    tvm.call_extern(dtype, "vadd",
                                    ub_tmp_buffer.access_ptr("rw", offset=ub_offset),
                                    ub_tmp_buffer.access_ptr("rw", offset=ub_offset),
                                    ub_pingpong_buffer.access_ptr("r", offset=ub_offset),
                                    min(255, repeat_times), 1, 1, 1, 8, 8, 8))
                repeat_times -= 255
                ub_offset += 255 * 128 // type_len_map[dtype]

        # reshape and copy in ub
        buffer_size //= shape[3]
        dma_offset = 0
        ping_offset = 0
        sid = 0
        n_burst = shape[0] * shape[1]
        len_burst = shape[4] * type_len_map[dtype] // 16
        src_stride = (shape[3] - 1) * shape[4] * type_len_map[dtype] // 16
        dst_stride = 0
        for i in range(shape[3]):
            ir_builder.emit(
                tvm.call_extern(dtype, "copy_ubuf_to_ubuf",
                                ub_ping_buffer.access_ptr("rw", offset=ping_offset),
                                ub_tmp_buffer.access_ptr("rw", offset=dma_offset),
                                sid, n_burst, len_burst, src_stride, dst_stride))
            dma_offset += shape[4]
            ping_offset += buffer_size

        # loop2, all vector compute, not need pingpang
        dma_offset = 0
        for i in range(shape[3] - 1):
            dma_offset += buffer_size
            repeat_times = (buffer_size * type_len_map[dtype]) // 128
            ub_offset = 0
            while repeat_times > 0:
                ir_builder.emit(
                    tvm.call_extern(dtype, "vadd",
                                    ub_ping_buffer.access_ptr("rw", offset=ub_offset),
                                    ub_ping_buffer.access_ptr("rw", offset=ub_offset),
                                    ub_ping_buffer.access_ptr("r", offset=(dma_offset+ub_offset)),
                                    min(255, repeat_times), 1, 1, 1, 8, 8, 8))
                repeat_times -= 255
                ub_offset += 255 * 128 // type_len_map[dtype]

        # copy out
        dst_addr = dst_buffer.access_ptr('w', offset=dst_offset)
        dma_offset = 0
        sid = 0
        n_burst = 1
        len_burst = shape[0] * shape[1] * shape[4] * type_len_map[dtype] // 16
        src_stride = 0
        dst_stride = 0
        ir_builder.emit(
            tvm.call_extern(dtype, "copy_ubuf_to_gm", dst_addr,
                            ub_ping_buffer.access_ptr("rw", offset=0),
                            sid, n_burst, len_burst, src_stride, dst_stride))

        reset_mask_insn(ir_builder, dtype)

@tvm.register_func("tvm.intrin.cce.reduce_2_3_axis_reduce_sum_cast_4D_optimal")
def reduce_sum_cast_4d_2_3_axis_optimal(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_reduce_2_3_axis(tensor_op, "vadd_4d")

def reduce_2_3_cmd_factory_4d(ir_builder, intrinsic_cmd, ins, outs, tile_size, reset_mask):
    """
    factory function for generate commond , only support reduce_sum and reduce_mean

    ir_builder: instance of ir_builder

    intrinsic_cmd : string
        commond type

    ins : list
        contains source buffers

    outs : list
        contains dst buffers

    tile_size : list
        contain shape info

    reset_mask : int or bool
        if reset_mask == True:
            means want to add reset mask to 128 at the end of commond if mask is changed,
            and pipeline id follows pipeline_count_list changes
        if reset_mask == False:
            means NOT want to add reset mask to 128 at the end of commond
        if reset_mask is int:
            means want to add reset mask to 128 at the end of commond if mask is changed,
            and pipeline id is reset_mask*8

    extern_args : list
        external args in VS or broadcast commond

    args : list
        commond args like the strides in vector commond
    """
    # tile_size: [7, 7, 256, 7, 7, 256, 256, 4]
    reduce_type = len(tile_size)
    if intrinsic_cmd == "vmean_4d":
        reduce_type_map = {11: "cast_single_reduce_sum"}
    else:
        reduce_type_map = {8: "single_reduce_sum_float32",
                           12: "cast_single_reduce_sum"}
    type_len_map = {"float16": 1,
                    "float32": 2}
    for i in reduce_type_map:
        if reduce_type == i:
            shape_0 = tile_size[-1]
            break
        if reduce_type == i - 1:
            shape_0 = 1
            reduce_type = i
    dtype = outs[0].dtype
    shape = [shape_0, tile_size[2], tile_size[1], tile_size[0]]
    src_buffer = ins[0]
    for i in outs:
        dst_buffer = i

    thread_block = get_emitinsn_params("thread_block")
    ub_max_enable_buff_size = get_emitinsn_params("ub_max_enable_buff_size")
    out_reduce_buff_fp32 = OUT_REDUCE_BUFF_FP32
    dst_offset = thread_block * shape[0] * shape[1]
    src_offset = thread_block * shape[0] * shape[1] * shape[2] * shape[3]

    def new_alloc(ir_builder, dtype, shape, name, scope):
        """
        new_alloc
        """
        buf_var = ir_builder.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)
        return new_buffer

    # pylint: disable=too-many-nested-blocks
    if reduce_type_map[reduce_type] == "cast_single_reduce_sum" and dtype == "float16":
        out_reduce_max_buff = out_reduce_buff_fp32 + out_reduce_buff_fp32 // 2
        ping_buff = min((ub_max_enable_buff_size - out_reduce_max_buff) // 4 // 2, 16384)
        out_buff = out_reduce_buff_fp32 // 4

        ub_tmp_buffer = \
            new_alloc(ir_builder, "float32", (ping_buff, ), 'ub_tmp_buffer', scope=cce.scope_ubuf)
        ub_ping_buffer \
            = new_alloc(ir_builder, dtype, (ping_buff, ), 'ub_ping_buffer', scope=cce.scope_ubuf)
        ub_pong_buffer \
            = new_alloc(ir_builder, dtype, (ping_buff, ), 'ub_pong_buffer', scope=cce.scope_ubuf)
        ub_out_buffer = new_alloc(ir_builder, dtype, (out_buff, ), 'ub_out', scope=cce.scope_ubuf)
        ub_out_fp32_buffer \
            = new_alloc(ir_builder, "float32", (out_buff, ), 'ub_out_fp32', scope=cce.scope_ubuf)
        ub_out_tmp_buffer \
            = new_alloc(ir_builder, "float32", (out_buff, ), 'ub_out_tmp', scope=cce.scope_ubuf)
        reset_mask_insn(ir_builder, dtype)

        m_stride = 2
        while ((shape[2] * shape[3] * m_stride) % (16 // type_len_map[dtype])) != 0:
            m_stride *= 2

        # calculate for column
        if ((shape[0] * shape[1]) // m_stride) \
            >= (shape[2] * shape[3] * type_len_map[dtype] // 128):
            if (shape[2] * shape[3] * type_len_map[dtype]) % 16 == 0:
                max_len_burst = (shape[2] * shape[3] * type_len_map[dtype]) // 16
            else:
                max_len_burst = (shape[2] * shape[3] * type_len_map[dtype]) // 16 + 1
            max_n_burst = (shape[0] * shape[1]) // m_stride
            dma_out_offset = 0

            while max_n_burst > 0:
                if max_n_burst >= 256:
                    n_burst = 256
                else:
                    n_burst = max_n_burst

                max_len_burst_tmp = max_len_burst
                max_mask = shape[2] * shape[3]
                while max_len_burst_tmp > 0:
                    if max_len_burst_tmp >= 4:
                        len_burst = 4
                    else:
                        len_burst = max_len_burst_tmp
                    src_stride = (shape[2] * shape[3] * m_stride * type_len_map[dtype]) // 16 \
                                - len_burst
                    dst_stride = 4 - len_burst
                    if max_mask >= 64:
                        mask = 64
                    else:
                        mask = max_mask
                    dma_offset = shape[2] * shape[3] \
                                * (shape[0] * shape[1] - max_n_burst * m_stride) \
                                + (shape[2] * shape[3] - max_mask)

                    for i in range(m_stride): #  // 2
                        # ping
                        # copy in: pingpang buffer
                        if i % 2 == 0:
                            ub_pingpang_buffer = ub_pong_buffer #ub_pong_buffer
                        else:
                            ub_pingpang_buffer = ub_ping_buffer
                        sid = 0
                        src_addr = src_buffer.access_ptr('r', offset=(src_offset + dma_offset))
                        ir_builder.emit(
                            tvm.call_extern(dtype, "copy_gm_to_ubuf",
                                            ub_pingpang_buffer.access_ptr("rw", offset=0),
                                            src_addr,
                                            sid, n_burst, len_burst, src_stride, dst_stride))
                        dma_offset += shape[2] * shape[3]
                        # cast to fp32
                        reset_mask_insn(ir_builder, dtype, bits=64)
                        repeat_times = n_burst
                        ub_offset = 0
                        while repeat_times > 0:
                            ir_builder.emit(
                                tvm.call_extern(
                                    "float32", "vconv_f162f32",
                                    ub_tmp_buffer.access_ptr("rw", offset=ub_offset),
                                    ub_pingpang_buffer.access_ptr("r", offset=ub_offset),
                                    min(repeat_times, 255), 1, 1, 8, 4)
                                )
                            repeat_times -= 255
                            ub_offset += 255 * 128 // type_len_map["float32"]
                        # reduce sum
                        repeat_times = n_burst
                        ub_offset = 0
                        ub_offset_scale = i
                        reset_mask_insn(ir_builder, dtype, bits=mask)
                        while repeat_times > 0:
                            ir_builder.emit(tvm.call_extern(
                                "float32", "vcadd",
                                ub_out_tmp_buffer.access_ptr("rw", offset=ub_offset_scale),
                                ub_tmp_buffer.access_ptr("r", offset=ub_offset),
                                min(repeat_times, 255), m_stride, 1, 8))
                            repeat_times -= 255
                            ub_offset += 255 * 128 // type_len_map["float32"]
                            ub_offset_scale += 255 * m_stride

                    reset_mask_insn(ir_builder, dtype, bits=64)
                    if max_len_burst_tmp == max_len_burst:
                        vector_times = n_burst * m_stride * type_len_map["float32"] // 16
                        ir_builder.emit(
                            tvm.call_extern("float32", "copy_ubuf_to_ubuf",
                                            ub_out_fp32_buffer.access_ptr("rw", offset=0),
                                            ub_out_tmp_buffer.access_ptr("rw", offset=0),
                                            0, 1, vector_times, 0, 0))
                    # add in ub_out
                    else:
                        vector_times = n_burst * m_stride * type_len_map["float32"] // 128 + 1
                        ir_builder.emit(
                            tvm.call_extern("float32", "vadd",
                                            ub_out_fp32_buffer.access_ptr("rw", offset=0),
                                            ub_out_fp32_buffer.access_ptr("rw", offset=0),
                                            ub_out_tmp_buffer.access_ptr("r", offset=0),
                                            vector_times, 1, 1, 1, 8, 8, 8))
                    max_len_burst_tmp -= 4
                    max_mask -= 64

                if intrinsic_cmd == "vmean_4d":
                    vector_times = n_burst * m_stride * type_len_map["float32"] // 128 + 1
                    const_mul = get_emitinsn_params("const_mul")
                    ir_builder.emit(
                        tvm.call_extern("float32", "vmuls",
                                        ub_out_fp32_buffer.access_ptr("rw", offset=0),
                                        ub_out_fp32_buffer.access_ptr("rw", offset=0),
                                        tvm.const(const_mul, "float32"),
                                        vector_times, 1, 1, 8, 8))
                # cast to fp16
                vector_times = n_burst * m_stride * type_len_map["float32"] // 128 + 1
                ir_builder.emit(
                    tvm.call_extern(dtype, "vconv_f322f16",
                                    ub_out_buffer.access_ptr("rw", offset=0),
                                    ub_out_fp32_buffer.access_ptr("r", offset=0),
                                    vector_times, 1, 1, 4, 8))
                #copy to out
                dst_addr = dst_buffer.access_ptr('w', offset=(dst_offset + dma_out_offset))
                dma_out_offset += n_burst * m_stride
                ir_builder.emit(
                    tvm.call_extern(dtype, "copy_ubuf_to_gm", dst_addr,
                                    ub_out_buffer.access_ptr("rw", offset=0),
                                    0, 1, n_burst * m_stride * type_len_map[dtype] // 16, 0, 0))
                max_n_burst -= 256
        reset_mask_insn(ir_builder, dtype)

    elif reduce_type_map[reduce_type] == "single_reduce_sum_float32" and dtype == "float32":
        pass


@tvm.register_func("tvm.intrin.cce.reduce_2_3_axis_reduce_mean_cast_4D_optimal")
def reduce_mean_cast_4d_2_3_axis_optimal(tensor_op):
    """
          :param tensor_op: the stmt of for with cast
          :return: the intric stmt what we want
    """
    return vec_reduce_2_3_axis(tensor_op, "vmean_4d")

@tvm.register_func("tvm.intrin.cce.last_axis_reduce_max")
def last_axis_reduce_max(stmt_op):
    """Collapse second input tensor to one repeat and calculate max to output"""
    # Get input and output buffers
    for_extents = []
    ir_builder = tvm.ir_builder.create()

    def _post_order_for(_stmt):
        if isinstance(_stmt, tvm.stmt.For):
            for_extents.append(int(_stmt.extent.value))

    tvm.ir_pass.IRTransform(stmt_op, None, _post_order_for, ["For"])
    if len(for_extents) > 2:
        raise RuntimeError("last_axis_reduce_max only supports 2-dimentional last-axis reduce max")
    if len(for_extents) == 2:
        outer_loop = for_extents[1]
    else:
        outer_loop = 1
    input_size = for_extents[0]
    ins, _ = cce_util.get_buffer(stmt_op, need_unique=True, buffer_shape=for_extents,
                                 need_origin_adress=True)
    _, outs = cce_util.get_buffer(stmt_op, need_unique=True)
    in_buffer = ins[1]
    out_buffer = outs[0]

    # Check if input is aligned for each loop
    element_size = cce_util.get_align_factor(in_buffer.dtype)[1]
    element_per_block = ALIGNMENT_BYTES // element_size
    vector_inst_one_repeat_size = cce.VECTOR_INST_BLOCK_WIDTH // element_size
    repeat_times = input_size // vector_inst_one_repeat_size
    remains = input_size - vector_inst_one_repeat_size * repeat_times
    if outer_loop > 1 and input_size * element_size % ALIGNMENT_BYTES != 0:
        raise RuntimeError("Reduce last axis data is not aligned for ub split factor larger than 1")

    # Do Emit Insn
    def allocate_reduce_max_temp_buffer(ir_b, dtype):
        def new_alloc(tvm_ib, dtype, shape, name, scope):
            buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
            new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)
            return new_buffer
        res = new_alloc(ir_b, dtype, (input_size,),
                        "last_axis_reduce_max_temp", cce.scope_ubuf)
        return res

    buffer = allocate_reduce_max_temp_buffer(ir_builder, in_buffer.dtype)
    with ir_builder.for_range(0, outer_loop, "last_axis_reduce_max_outer_loop") as outer_loop:
        # Initialize temp buffer
        ir_builder.emit(tvm.call_extern(in_buffer.dtype,
                                        'copy_ubuf_to_ubuf',
                                        buffer.access_ptr("rw"),
                                        in_buffer.access_ptr("r", offset=outer_loop * input_size),
                                        0, 1, 8, 0, 0))
        # Main part
        reset_mask_insn(ir_builder, buffer.dtype)
        offset = outer_loop * input_size
        # Because first repeat is moved into dst prior to calculation, repeat offset will be 1.
        MAXIMUM_REPEAT = 255
        REPEAT_OFFSET = 1
        if repeat_times > MAXIMUM_REPEAT * 2 + REPEAT_OFFSET:
            raise RuntimeError("Reduce max axis too large, max 255 * 2 + 1:", repeat_times)
        if repeat_times > MAXIMUM_REPEAT + REPEAT_OFFSET:
            ir_builder.emit(tvm.call_extern(
                buffer.dtype,
                "vmax",
                buffer.access_ptr("rw", offset=0),
                buffer.access_ptr("r", offset=0),
                in_buffer.access_ptr("r", offset=offset + vector_inst_one_repeat_size),
                MAXIMUM_REPEAT, 1, 1, 1, 0, 0, 8))
            ir_builder.emit(tvm.call_extern(
                buffer.dtype,
                "vmax",
                buffer.access_ptr("rw", offset=0),
                buffer.access_ptr("r", offset=0),
                in_buffer.access_ptr("r", offset=offset
                                     + vector_inst_one_repeat_size*(MAXIMUM_REPEAT+REPEAT_OFFSET)),
                repeat_times - MAXIMUM_REPEAT - REPEAT_OFFSET, 1, 1, 1, 0, 0, 8))
        else:
            ir_builder.emit(tvm.call_extern(
                buffer.dtype,
                "vmax",
                buffer.access_ptr("rw", offset=0),
                buffer.access_ptr("r", offset=0),
                in_buffer.access_ptr("r", offset=offset + vector_inst_one_repeat_size),
                repeat_times - 1, 1, 1, 1, 0, 0, 8))
        # Remain part
        if remains > 0:
            reset_mask_insn(ir_builder, buffer.dtype, bits=remains)
            ir_builder.emit(tvm.call_extern(
                buffer.dtype,
                "vmax",
                buffer.access_ptr("rw", offset=0),
                buffer.access_ptr("r", offset=0),
                in_buffer.access_ptr("r", offset=offset
                                     + vector_inst_one_repeat_size * repeat_times),
                1, 1, 1, 1, 0, 0, 8))
        # Collapse
        collapse_loop = 3
        current_size = vector_inst_one_repeat_size
        for i in range(collapse_loop):
            current_size = current_size // 2
            reset_mask_insn(ir_builder, buffer.dtype, bits=current_size)
            ir_builder.emit(tvm.call_extern(
                buffer.dtype,
                "vmax",
                buffer.access_ptr("rw", offset=0),
                buffer.access_ptr("r", offset=0),
                buffer.access_ptr("r", offset=current_size),
                1, 1, 1, 1, 8, 8, 8))
        # Split to block
        for i in range(1, element_per_block):
            ir_builder.emit(tvm.call_extern(
                buffer.dtype,
                "reg_mov",
                buffer.access_ptr("rw", offset=i * element_per_block),
                buffer.access_ptr("r", offset=i)))
        # Final reduce
        reset_mask_insn(ir_builder, buffer.dtype, bits=1)
        ir_builder.emit(tvm.call_extern(
            buffer.dtype,
            "vmax",
            buffer.access_ptr("rw", offset=0),
            buffer.access_ptr("r", offset=0),
            buffer.access_ptr("r", offset=element_per_block),
            element_per_block - 1, 1, 1, 1, 1, 0, 0))
        # Output
        ir_builder.emit(tvm.call_extern(
            buffer.dtype,
            "reg_mov",
            out_buffer.access_ptr("rw", offset=outer_loop),
            buffer.access_ptr("r", offset=0)))
    reset_mask_insn(ir_builder, buffer.dtype)
    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.last_axis_reduce_sum_reuse")
def last_axis_reduce_sum_reuse(stmt_op):
    """Collapse second input tensor to one repeat and calculate sum to output"""
    # Get input and output buffers
    for_extents = []
    ir_builder = tvm.ir_builder.create()

    def _post_order_for(_stmt):
        if isinstance(_stmt, tvm.stmt.For):
            for_extents.append(int(_stmt.extent.value))

    tvm.ir_pass.IRTransform(stmt_op, None, _post_order_for, ["For"])
    if len(for_extents) > 2:
        raise RuntimeError("last_axis_reduce_sum_re only supports 2-dimentional last-axis reduce")
    if len(for_extents) == 2:
        outer_loop = for_extents[1]
    else:
        outer_loop = 1
    input_size = for_extents[0]
    ins, _ = cce_util.get_buffer(stmt_op, need_unique=True, buffer_shape=for_extents,
                                 need_origin_adress=True)
    _, outs = cce_util.get_buffer(stmt_op, need_unique=True)
    in_buffer = ins[1]
    out_buffer = outs[0]

    # Check if input is aligned for each loop
    element_size = cce_util.get_align_factor(in_buffer.dtype)[1]
    vector_inst_one_repeat_size = cce.VECTOR_INST_BLOCK_WIDTH // element_size
    repeat_times = input_size // vector_inst_one_repeat_size
    remains = input_size - vector_inst_one_repeat_size * repeat_times
    collapse_loop_num = int(math.log(input_size / vector_inst_one_repeat_size, 2))
    collapse_repeat = int(math.pow(2, collapse_loop_num))
    out_of_collapse_repeat = repeat_times - collapse_repeat
    if outer_loop > 1 and input_size * element_size % ALIGNMENT_BYTES != 0:
        raise RuntimeError("Reduce last axis data is not aligned for ub split factor larger than 1")

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
    buffer = in_buffer
    with ir_builder.for_range(0, outer_loop, "last_axis_reduce_sum_reuse_outer_loop") as outer_loop:
        # Main part before collapse
        offset = outer_loop * input_size
        if out_of_collapse_repeat > 0:
            reset_mask_insn(ir_builder, buffer.dtype)
            ir_builder.emit(tvm.call_extern(
                buffer.dtype,
                "vadd",
                buffer.access_ptr("rw", offset=offset),
                buffer.access_ptr("r", offset=offset),
                in_buffer.access_ptr("r", offset=offset
                                     + vector_inst_one_repeat_size * collapse_repeat),
                out_of_collapse_repeat, 1, 1, 1, 8, 8, 8))
        # Collapse
        cur_size = collapse_repeat * vector_inst_one_repeat_size
        for _ in range(int(collapse_loop_num)):
            cur_size = collapse(ir_builder, in_buffer, cur_size)
        # Remain part
        if remains > 0:
            reset_mask_insn(ir_builder, buffer.dtype, bits=remains)
            ir_builder.emit(tvm.call_extern(
                buffer.dtype,
                "vadd",
                buffer.access_ptr("rw", offset=offset),
                buffer.access_ptr("r", offset=offset),
                in_buffer.access_ptr("r", offset=offset
                                     + vector_inst_one_repeat_size * repeat_times),
                1, 1, 1, 1, 8, 8, 8))
        # Final reduce
        reset_mask_insn(ir_builder, buffer.dtype)
        ir_builder.emit(tvm.call_extern(
            buffer.dtype,
            "vcadd",
            out_buffer.access_ptr("rw", offset=outer_loop),
            buffer.access_ptr("r", offset=0),
            1, 1, 1, 8))
    reset_mask_insn(ir_builder, buffer.dtype)
    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.elewise_binary_sub_scalar_L1")
def elewise_binary_sub_scalar_L1(stmt_op):
    """A[a][b] - B[a]"""
    # Get input and output buffers
    for_extents = []
    ir_builder = tvm.ir_builder.create()

    def _post_order_for(_stmt):
        if isinstance(_stmt, tvm.stmt.For):
            for_extents.append(int(_stmt.extent.value))

    tvm.ir_pass.IRTransform(stmt_op, None, _post_order_for, ["For"])
    if len(for_extents) > 2:
        raise RuntimeError("elewise_binary_sub_scalar only supports 2-dimentional last-axis scalar")
    if len(for_extents) == 2:
        outer_loop = for_extents[1]
    else:
        outer_loop = 1
    input_size = for_extents[0]
    ins, _ = cce_util.get_buffer(stmt_op, need_unique=True, buffer_shape=for_extents,
                                 need_origin_adress=True)
    _, outs = cce_util.get_buffer(stmt_op, need_unique=True)
    in_buffer = ins[0]
    scalar = ins[1]
    out_buffer = outs[0]

    # Check if input is aligned for each loop
    element_size = cce_util.get_align_factor(in_buffer.dtype)[1]
    vector_inst_one_repeat_size = cce.VECTOR_INST_BLOCK_WIDTH // element_size
    repeat_times = input_size // vector_inst_one_repeat_size
    remains = input_size - vector_inst_one_repeat_size * repeat_times
    if outer_loop > 1 and input_size * element_size % ALIGNMENT_BYTES != 0:
        raise RuntimeError("Reduce last axis data is not aligned for ub split factor larger than 1")

    # Do Emit Insn
    with ir_builder.for_range(0, outer_loop, "elewise_binary_sub_scalar_outer_loop") as outer_loop:
        def new_alloc(tvm_ib, dtype, shape, name, scope):
            buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
            new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)
            return new_buffer
        tmp_buffer = new_alloc(ir_builder,
                               scalar.dtype, (ALIGNMENT_BYTES // element_size,), "sub_tmp_buf",
                               cce.scope_ubuf)
        reg = ir_builder.allocate(scalar.dtype, (1,), "reg_buf_sub", cce.scope_reg)
        ir_builder.emit(tvm.call_extern(
            scalar.dtype,
            "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[0]),
            scalar.access_ptr("r", offset=outer_loop)))
        reset_mask_insn(ir_builder, in_buffer.dtype, bits=ALIGNMENT_BYTES // element_size)
        ir_builder.emit(tvm.call_extern(
            scalar.dtype, 'vector_dup',
            tmp_buffer.access_ptr('rw', offset=0),
            reg[0],
            1,
            1,  # dst stridem0
            1,
            8,  # dst stridem1
            8
        ))
        reset_mask_insn(ir_builder, in_buffer.dtype)
        MAXIMUM_REPEAT = 255
        if repeat_times > 2 * MAXIMUM_REPEAT:
            raise RuntimeError("Repeat time too large:", repeat_times)
        if repeat_times > MAXIMUM_REPEAT:
            ir_builder.emit(tvm.call_extern(
                scalar.dtype,
                "vsub",
                out_buffer.access_ptr("rw", offset=outer_loop * input_size),
                in_buffer.access_ptr("r", offset=outer_loop * input_size),
                tmp_buffer.access_ptr("r", offset=0),
                MAXIMUM_REPEAT, 1, 1, 0, 8, 8, 0))
            offset = outer_loop * input_size + MAXIMUM_REPEAT * vector_inst_one_repeat_size
            ir_builder.emit(tvm.call_extern(
                scalar.dtype,
                "vsub",
                out_buffer.access_ptr("rw", offset=offset),
                in_buffer.access_ptr("r", offset=offset),
                tmp_buffer.access_ptr("r", offset=0),
                repeat_times - MAXIMUM_REPEAT, 1, 1, 0, 8, 8, 0))
        else:
            ir_builder.emit(tvm.call_extern(
                scalar.dtype,
                "vsub",
                out_buffer.access_ptr("rw", offset=outer_loop * input_size),
                in_buffer.access_ptr("r", offset=outer_loop * input_size),
                tmp_buffer.access_ptr("r", offset=0),
                repeat_times, 1, 1, 0, 8, 8, 0))
        # Remain part
        if remains > 0:
            reset_mask_insn(ir_builder, in_buffer.dtype, bits=remains)
            offset = outer_loop * input_size + repeat_times * vector_inst_one_repeat_size
            ir_builder.emit(tvm.call_extern(
                scalar.dtype,
                "vsub",
                out_buffer.access_ptr("rw",
                                      offset=offset),
                in_buffer.access_ptr("r", offset=offset),
                tmp_buffer.access_ptr("r", offset=0),
                1, 1, 1, 0, 8, 8, 0))
        reset_mask_insn(ir_builder, in_buffer.dtype)
        L1_workspace = new_alloc(ir_builder, out_buffer.dtype, out_buffer.shape, "L1_workspace",
                                 cce.scope_cbuf)
        burst_length = int(math.ceil(input_size * element_size / ALIGNMENT_BYTES))
        ir_builder.emit(tvm.call_extern(
            out_buffer.dtype,
            "copy_ubuf_to_cbuf",
            L1_workspace.access_ptr("rw", offset=0),
            out_buffer.access_ptr("r", offset=outer_loop * input_size),
            0, 1, burst_length, 0, 0))
        cce_emitinsn_params.cceEmitParamsIns.clear_param()
        cce_emitinsn_params.cceEmitParamsIns.insert_param("L1_workspace", L1_workspace)
        cce_emitinsn_params.cceEmitParamsIns.insert_param("L1_workspace_burst_length", burst_length)
    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.elewise_get_L1_workspace")
def elewise_get_L1_workspace(stmt_op):
    """Get L1 workspace defined by elewise_binary_sub_scalar_L1"""
    _, outs = cce_util.get_buffer(stmt_op, need_unique=True)
    in_buffer = cce_emitinsn_params.cceEmitParamsIns.get_param("L1_workspace")
    burst_length = cce_emitinsn_params.cceEmitParamsIns.get_param("L1_workspace_burst_length")
    ir_builder = tvm.ir_builder.create()
    ir_builder.emit(tvm.call_extern(
        outs[0].dtype, 'copy_cbuf_to_ubuf',
        outs[0].access_ptr('rw', offset=0),
        in_buffer.access_ptr('rw', offset=0),
        0, 1, burst_length, 0, 0))

    return ir_builder.get()


@tvm.register_func("tvm.intrin.cce.tik_exception_process")
def tik_exception_process(loc, msg):
    """when there is an exception in Tik calling TVM, first print tik error msg

    :param loc: node location including file and column
    :param msg: error message
    :return: None
    """
    if loc is None:
        print("Error: {}\n".format(msg.rstrip("\n")))
        return
    print("\n".join(get_context_msg(loc.file, int(loc.column), msg)))
