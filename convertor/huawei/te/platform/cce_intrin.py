#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

CCE related intrinsics
"""
# pylint: disable=too-many-lines, unused-import
from __future__ import absolute_import as _abs

from functools import reduce as functools_reduce

from te import tvm
from . import cce_conf
from . import cce_runtime
from . import cce_params as param
from . import cce_util

# the limit of the nesting depth in the generated CCE,
# if num_segments > MAX_BRACKET_DEPTH, stackoverflow,
# the number was tested by experiment
MAX_BRACKET_DEPTH = 250

# pylint: disable=too-many-locals
def intrin_gemm_cce(input_m, input_n, input_k, mock=False):
    """Store intrinsics"""
    wgt_lanes = param.WGT_ELEM_BYTES*8 // param.WGT_WIDTH
    if wgt_lanes != param.BLOCK_OUT * param.BLOCK_REDUCE:
        raise RuntimeError(
            "param.WGT_WIDTH not equal to param.BLOCK_OUT*param.BLOCK_REDUCE.")
    wgt_shape = (input_k // param.BLOCK_REDUCE, input_n // param.BLOCK_OUT,
                 param.BLOCK_OUT, param.BLOCK_REDUCE)
    if wgt_shape[2]*wgt_shape[3] != wgt_lanes:
        raise RuntimeError("Shapes not equal.")

    inp_lanes = param.INP_ELEM_BYTES*8 // param.INP_WIDTH
    if inp_lanes != param.BLOCK_IN*param.BLOCK_REDUCE:
        raise RuntimeError("Shapes not equal.")
    inp_shape = (input_m // param.BLOCK_IN, input_k // param.BLOCK_REDUCE,
                 param.BLOCK_IN, param.BLOCK_REDUCE)
    if inp_shape[2]*inp_shape[3] != inp_lanes:
        raise RuntimeError("Shapes not equal.")
    if inp_shape[1] != wgt_shape[0]:
        raise RuntimeError("Shapes not equal.")
    if inp_shape[3] != wgt_shape[3]:
        raise RuntimeError("Shapes not equal.")

    out_lanes = param.OUT_ELEM_BYTES*8 // param.OUT_WIDTH
    if out_lanes != param.BLOCK_OUT*param.BLOCK_IN:
        raise RuntimeError("Shapes not equal.")
    out_shape = (input_n // param.BLOCK_OUT, input_m // param.BLOCK_IN,
                 param.BLOCK_IN, param.BLOCK_OUT)
    if out_shape[2]*out_shape[3] != out_lanes:
        raise RuntimeError("Shapes not equal.")
    if out_shape[0] != wgt_shape[1]:
        raise RuntimeError("Shapes not equal.")
    if out_shape[1] != inp_shape[0]:
        raise RuntimeError("Shapes not equal.")
    wgt = tvm.placeholder(wgt_shape,
                          dtype="float%d" % param.WGT_WIDTH,
                          name=param.scope_cb)
    inp = tvm.placeholder(inp_shape,
                          dtype="float%d" % param.INP_WIDTH,
                          name=param.scope_ca)
    res_k2 = tvm.reduce_axis((0, wgt_shape[0]), name="k2")
    res_k1 = tvm.reduce_axis((0, wgt_shape[3]), name="k1")
    out_dtype = "float%d" % param.OUT_WIDTH
    out = tvm.compute(out_shape,
                      lambda y, x, i, j: tvm.sum(inp[x, res_k2, i, res_k1].astype(out_dtype)*
                                                 wgt[res_k2, y, j, res_k1].astype(out_dtype),
                                                 axis=[res_k2, res_k1]),
                      name="out")
    wgt_layout = tvm.decl_buffer(
        wgt.shape, wgt.dtype, param.scope_cb,
        scope=param.scope_cb, offset_factor=wgt_lanes, data_alignment=wgt_lanes)
    inp_layout = tvm.decl_buffer(
        inp.shape, inp.dtype, param.scope_ca,
        scope=param.scope_ca, offset_factor=inp_lanes, data_alignment=inp_lanes)
    out_layout = tvm.decl_buffer(
        out.shape, out.dtype, param.scope_cc,
        scope=param.scope_cc, offset_factor=out_lanes, data_alignment=out_lanes)

    def intrin_func(ins, outs):
        """
        intrin_func fuction
        """
        dinp, dwgt = ins
        dout = outs[0]

        def instr(index):
            """
            instr fuction
            """
            ib_ins = tvm.ir_builder.create()
            ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope", 3)
            control_c_bit = 1 if index == 0 else 0
            ib_ins.emit(tvm.call_extern(
                dout.dtype, "mad",
                dout.access_ptr("rw"),
                dinp.access_ptr("r"),
                dwgt.access_ptr("r"),
                dout.shape[0]*dout.shape[2],
                dinp.shape[1]*dinp.shape[3],
                dout.shape[1]*dout.shape[3],
                control_c_bit))
            return ib_ins.get()

        # return a triple of normal-set, reset, update
        nop = tvm.make.Evaluate(0)
        if mock:
            return (nop, nop, nop)
        return (instr(0), None, instr(2))

    return tvm.decl_tensor_intrin(out.op, intrin_func,
                                  name="mad",
                                  binds={inp: inp_layout,
                                         wgt: wgt_layout,
                                         out: out_layout})

def flatten_indice(data, indice):
    """
    flatten indice of data
    Parameters
    ----------
    data : tvm.tensor
        input data

    indice : list
        list of indice

    Returns
    -------
    ret : tensor
        example when indice = [1,2,3]:
        return tensor : data[1][2][3].
    """
    tmp = data
    for i in indice:
        tmp = tmp[i]
    return tmp


def produce_res_shape(shape, reduce_axis_index):
    """
    produce_res_shape fuction
    """
    reduce_axis_index = reduce_axis_index if reduce_axis_index >= 0 else len(
        shape) + reduce_axis_index
    return shape[:reduce_axis_index] + shape[reduce_axis_index + 1:]


def produce_res_indice(indice, reduce_axis_index, tvm_reduce_axis):
    """
    produce_res_indice fuction
    """
    reduce_axis_index = reduce_axis_index if reduce_axis_index >= 0 else len(
        indice) + 1 + reduce_axis_index
    return list(indice[:reduce_axis_index]) + [tvm_reduce_axis] + list(
        indice[reduce_axis_index:])


def get_bit_len(dtype):
    """
    calculate bits of dtype of TVM
    Parameters
    ----------
    dtype : string
        dtype of TVM

    Returns
    -------
    ret : int
        bit length of dtype.
    """

    # logic use vmul vadd vsub vcmp vsel, only support float16, so
    # 1. int8 -> float16 2.  mul vadd vsub vcmp vsel 3. float16 ->int8
    # bit len use float16 length
    if dtype == 'bool':
        return 16

    if dtype != 'bool':
        index = 0
        for i in dtype:
            if i.isdigit():
                break
            index += 1
        return int(dtype[index:])
    return None


def get_data_alignment(dtype):
    """
    calculate the unified buffer data alignment
    Parameters
    ----------
    dtype : string
        dtype of TVM

    Returns
    -------
    ret : int
        the num of data alignment.
    """
    return 32 * 8 // get_bit_len(dtype)


def set_mask(length):
    """
    calculate MASK in cce

    Parameters
    ----------
    length : int
        calculate length

    Returns
    -------
    mask : tuple of int
        low and high bit of mask.
    """
    length = int(length)
    mask1 = 2 ** max(length - 64, 0) - 1
    mask2 = 2 ** min(length, 64) - 1
    return mask1, mask2


# pylint: disable=too-many-statements
def intrin_factor(buffer_scope=param.scope_ubuf):
    """
    factory function For tensorize of cce. Using closure tech of Python.
    This function is a API to generate inner explicit functions for tensorize.

    Parameters
    ----------
    None

    Returns
    -------
    ret : factory function of of each type.

    example:
        vec_intrin = intrin_factor()
        intrin_mul = vec_intrin("elewise_binary_intrin_cce")
        intrin_sub = vec_intrin("elewise_binary_intrin_cce")
        s[A].tensorize(s[A].op.axis[0], intrin_mul(((256,)),
        "elewise_binary_mul", "foat16", "foat16"))

    """
    vec_count = {}
    total_pipe_line = 11

    # pylint: disable=too-many-locals
    def vec_intrin(intrin_key):
        # pylint: disable=too-many-statements
        """
        vec_intrin fuction
        """
        def __cal_map_sum(dict_data):
            return sum([dict_data[i] for i in dict_data])

        def __apply_for_new_alloc(ib_instance, dtype, shape, scope=buffer_scope):
            buf_var = ib_instance.allocate(dtype, shape, name="tmp_buf", scope=scope)
            tmp_buffer = tvm.decl_buffer(shape, buf_var.dtype,
                                         name="tmp_buf",
                                         scope=buffer_scope,
                                         data=buf_var)
            return tmp_buffer

        def __concat_args(concat_args):
            src_buffers = concat_args["src_buffers"]
            dst_buffers = concat_args["dst_buffers"]
            repeat_src_offset = concat_args["repeat_src_offset"]
            repeat_dst_offset = concat_args["repeat_dst_offset"]
            repeat_times = concat_args["repeat_times"]
            extern_args = concat_args["extern_args"]
            args = concat_args["args"]
            res_args = []
            for i in dst_buffers:
                res_args.append(i.access_ptr("wr", offset=repeat_dst_offset))
            for i in src_buffers:
                res_args.append(i.access_ptr("r", offset=repeat_src_offset))
            if extern_args is not None:
                res_args += extern_args
            res_args.append(repeat_times)
            res_args += args
            return res_args

        # pylint: disable=too-many-branches, len-as-condition
        def __vec_cmd_factory(vec_cmd_factory_args):
            """
            factory function for generate commond , only support elewise,
            broadcast, vcg, do not support VCOP

            ib: instance of ir_builder

            op_cmd : string
                commond type

            src_buffers : list
                contains source buffers

            dst_buffers : list
                contains dst buffers

            op_length : int
                data length

            reduce_factor : int
                equals to (src data length) / (dst data length)
            base_pipeline : int
                base pipeline number

            pipeline_count_list : list
                begin pipeline number

            reset_mask_pipe_line : int or bool
                if reset_mask_pipe_line == True:
                    means want to add reset mask to 128 at the end of commond if mask is changed,
                    and pipeline id follows pipeline_count_list changes.
                if reset_mask_pipe_line == False:
                    means NOT want to add reset mask to 128 at the end of commond
                if reset_mask_pipe_line is int:
                    means want to add reset mask to 128 at the end of commond if mask is changed,
                    and pipeline id is reset_mask_pipe_line*8.

            extern_args : list
                external args in VS or broadcast commond

            args : list
                commond args like the strides in vector commond
            """
            ib_instance = vec_cmd_factory_args["ib_instance"]
            op_cmd = vec_cmd_factory_args["op_cmd"]
            src_buffers = vec_cmd_factory_args["src_buffers"]
            dst_buffers = vec_cmd_factory_args["dst_buffers"]
            op_length = vec_cmd_factory_args["op_length"]
            reduce_factor = vec_cmd_factory_args["reduce_factor"]
            reset_mask_pipe_line = vec_cmd_factory_args["reset_mask_pipe_line"]
            extern_args = vec_cmd_factory_args["extern_args"]
            args = vec_cmd_factory_args["args"]
            repeat_cal_dtype = vec_cmd_factory_args["repeat_cal_dtype"]
            block_len = 256
            local_total_len = op_length
            if len(src_buffers) == 0:
                src_dtype = extern_args[0].dtype
            else:
                src_dtype = src_buffers[0].dtype
            if repeat_cal_dtype is None:
                repeat_cal_dtype = src_dtype
            dst_dtype = dst_buffers[0].dtype
            cal_bit_len = get_bit_len(repeat_cal_dtype)

            cal_once_len = block_len * 8 // cal_bit_len
            repeat_times = local_total_len // cal_once_len
            remain_len = local_total_len - repeat_times * cal_once_len
            if op_cmd == 'vcond':
                temp_thredhold = tvm.const(float(extern_args[1]), dst_dtype)
                temp_bias = tvm.const(float(extern_args[2]), dst_dtype)
                thred_buf = __apply_for_new_alloc(ib_instance, dst_dtype,
                                                  (cal_once_len,),
                                                  scope=buffer_scope)
                ib_instance.emit(tvm.call_extern(
                    dst_dtype, "vector_dup",
                    thred_buf.access_ptr("rw"),
                    temp_thredhold, 1, 1, 1, 8, 8))

                bias_buf = __apply_for_new_alloc(ib_instance, dst_dtype, (cal_once_len,),
                                                 scope=buffer_scope)
                ib_instance.emit(tvm.call_extern(
                    dst_dtype, "vector_dup",
                    bias_buf.access_ptr("rw"),
                    temp_bias, 1, 1, 1, 8, 8))
            elif op_cmd == 'vlogic':
                dup_repeat_times = op_length // cal_once_len
                new_src_buffer0 = __apply_for_new_alloc(ib_instance, "float16",
                                                        src_buffers[0].shape,
                                                        scope=buffer_scope)
                new_src_buffer1 = __apply_for_new_alloc(ib_instance, "float16",
                                                        src_buffers[0].shape,
                                                        scope=buffer_scope)
                if extern_args[0] == 'or' or extern_args[0] == 'not':
                    thred_buf = __apply_for_new_alloc(ib_instance, "float16",
                                                      src_buffers[0].shape,
                                                      scope=buffer_scope)
                    ib_instance.emit(tvm.call_extern(
                        "float16", "vector_dup",
                        thred_buf.access_ptr("rw"),
                        tvm.const(0, "float16"),
                        dup_repeat_times if (dup_repeat_times != 0) else 1,
                        1, 1, 8, 8))

                    bias_buf = __apply_for_new_alloc(ib_instance, "float16",
                                                     src_buffers[0].shape,
                                                     scope=buffer_scope)
                    ib_instance.emit(tvm.call_extern(
                        "float16", "vector_dup",
                        bias_buf.access_ptr("rw"),
                        tvm.const(1, "float16"),
                        dup_repeat_times if (dup_repeat_times != 0) else 1,
                        1, 1, 8, 8))

                    temp_andor_out = __apply_for_new_alloc(ib_instance, "float16",
                                                           src_buffers[0].shape,
                                                           scope=buffer_scope)
                temp_out = __apply_for_new_alloc(ib_instance, "float16",
                                                 src_buffers[0].shape,
                                                 scope=buffer_scope)
            elif op_cmd.lower().find("vcmp_") != -1:
                dup_repeat_times = op_length // cal_once_len
                scalar_buf = __apply_for_new_alloc(ib_instance,
                                                   dst_dtype,
                                                   src_buffers[0].shape,
                                                   scope=buffer_scope)
                ib_instance.emit(tvm.call_extern(dst_dtype,
                                                 "vector_dup",
                                                 scalar_buf.access_ptr("rw"),
                                                 extern_args[0],
                                                 dup_repeat_times if (
                                                     dup_repeat_times != 0) else 1,
                                                 1, 1, 8, 8))
                bias_buf = __apply_for_new_alloc(ib_instance,
                                                 dst_dtype,
                                                 src_buffers[0].shape,
                                                 scope=buffer_scope)
                ib_instance.emit(tvm.call_extern(dst_dtype,
                                                 "vector_dup",
                                                 bias_buf.access_ptr("rw"),
                                                 tvm.const(0, dtype=dst_dtype),
                                                 dup_repeat_times if (
                                                     dup_repeat_times != 0) else 1,
                                                 1, 1, 8, 8))

            # pylint: disable=too-many-nested-blocks
            if repeat_times > 0:
                local_repeat_times = repeat_times
                repeat_dst_offset = 0
                repeat_src_offset = 0

                if op_cmd.lower().find("vcmp_") != -1:
                    with ib_instance.for_range(0, repeat_times,
                                               name="cmp_index") as cmp_index:
                        repeat_offset = cal_once_len * cmp_index
                        dst_repeat_offset = (
                            cal_once_len // reduce_factor) * cmp_index
                        ib_instance.emit(tvm.call_extern(dst_dtype,
                                                         op_cmd,
                                                         src_buffers[0].access_ptr(
                                                             "r",
                                                             offset=repeat_offset),
                                                         scalar_buf.access_ptr(
                                                             "r",
                                                             offset=repeat_offset),
                                                         1, 1, 1, 1, 8, 8, 8))
                        ib_instance.emit(tvm.call_extern(dst_dtype,
                                                         "vsel",
                                                         dst_buffers[0].access_ptr(
                                                             "w",
                                                             offset=dst_repeat_offset),
                                                         src_buffers[1].access_ptr(
                                                             "r",
                                                             offset=repeat_offset),
                                                         bias_buf.access_ptr("r"),
                                                         1, 1, 1, 1, 8, 8, 8))
                else:
                    while local_repeat_times > 0:
                        if local_repeat_times > 255:
                            tmp_repeat_times = 255
                        else:
                            tmp_repeat_times = local_repeat_times

                        if op_cmd == 'vcond':
                            # vsel only can process 128 numbers
                            with ib_instance.for_range(0, tmp_repeat_times,
                                                       name="cmp_index") as cmp_index:
                                tmp_src_repeat_offset = cal_once_len * cmp_index
                                tmp_dst_repeat_offset = (
                                    cal_once_len // reduce_factor) * cmp_index
                                ib_instance.emit(tvm.call_extern(
                                    dst_dtype, "vcmp_" + extern_args[0],
                                    src_buffers[0].access_ptr("r",
                                                              offset=tmp_src_repeat_offset),
                                    thred_buf.access_ptr("r"), 1, 1, 1, 1, 8, 8,
                                    8))

                                ib_instance.emit(tvm.call_extern(
                                    dst_dtype, "vsel",
                                    dst_buffers[0].access_ptr("rw",
                                                              offset=tmp_dst_repeat_offset),
                                    src_buffers[0].access_ptr("r",
                                                              offset=tmp_src_repeat_offset),
                                    bias_buf.access_ptr("r"),
                                    1, 1, 1, 1, 8, 8, 8))
                        elif op_cmd == 'vcmpsel':
                            # vsel only can process 128 numbers
                            with ib_instance.for_range(0, tmp_repeat_times,
                                                       name="cmp_index") as cmp_index:
                                tmp_src_repeat_offset = cal_once_len * cmp_index
                                tmp_dst_repeat_offset = (
                                    cal_once_len // reduce_factor) * cmp_index
                                ib_instance.emit(tvm.call_extern(
                                    dst_dtype, "vcmp_" + extern_args[0],
                                    src_buffers[0].access_ptr("r",
                                                              offset=tmp_src_repeat_offset),
                                    src_buffers[1].access_ptr("r",
                                                              offset=tmp_src_repeat_offset),
                                    1, 1, 1, 1, 8, 8, 8))

                                ib_instance.emit(tvm.call_extern(
                                    dst_dtype, "vsel",
                                    dst_buffers[0].access_ptr("rw",
                                                              offset=tmp_dst_repeat_offset),
                                    src_buffers[0].access_ptr("r",
                                                              offset=tmp_src_repeat_offset),
                                    src_buffers[1].access_ptr("r",
                                                              offset=tmp_src_repeat_offset),
                                    1, 1, 1, 1, 8, 8, 8))
                        elif op_cmd == 'vlogic':
                            ib_instance.emit(tvm.call_extern(
                                "float16", "vconv_s82f16",
                                new_src_buffer0.access_ptr(
                                    "w",
                                    offset=repeat_src_offset),
                                src_buffers[0].access_ptr("r",
                                                          offset=repeat_src_offset),
                                tmp_repeat_times, 1, 1, 8, 4))
                            if extern_args[0] != 'not':
                                ib_instance.emit(tvm.call_extern(
                                    "float16", "vconv_s82f16",
                                    new_src_buffer1.access_ptr(
                                        "w",
                                        offset=repeat_src_offset),
                                    src_buffers[1].access_ptr("r",
                                                              offset=repeat_src_offset),
                                    tmp_repeat_times, 1, 1, 8, 4))

                            if extern_args[0] == 'and':
                                ib_instance.emit(tvm.call_extern("float16", "vmul",
                                                                 temp_out.access_ptr(
                                                                     "rw",
                                                                     offset=repeat_dst_offset),
                                                                 new_src_buffer0.access_ptr(
                                                                     "r",
                                                                     offset=repeat_src_offset),
                                                                 new_src_buffer1.access_ptr(
                                                                     "r",
                                                                     offset=repeat_src_offset),
                                                                 tmp_repeat_times, 1, 1,
                                                                 1, 8, 8, 8))
                            elif extern_args[0] == 'or' or extern_args[0] == 'not':
                                if extern_args[0] == 'or':
                                    ib_instance.emit(tvm.call_extern("float16", "vadd",
                                                                     temp_andor_out.access_ptr(
                                                                         "rw",
                                                                         offset=repeat_dst_offset),
                                                                     new_src_buffer0.access_ptr(
                                                                         "r",
                                                                         offset=repeat_src_offset),
                                                                     new_src_buffer1.access_ptr(
                                                                         "r",
                                                                         offset=repeat_src_offset),
                                                                     tmp_repeat_times, 1,
                                                                     1, 1, 8, 8, 8))
                                elif extern_args[0] == 'not':
                                    ib_instance.emit(tvm.call_extern("float16", "vsub",
                                                                     temp_andor_out.access_ptr(
                                                                         "rw",
                                                                         offset=repeat_dst_offset),
                                                                     new_src_buffer0.access_ptr(
                                                                         "r",
                                                                         offset=repeat_src_offset),
                                                                     bias_buf.access_ptr(
                                                                         "r"),
                                                                     tmp_repeat_times, 1,
                                                                     1, 1, 8, 8, 8))

                                with ib_instance.for_range(0, tmp_repeat_times,
                                                           name="cmp_index") as cmp_index:
                                    tmp_src_repeat_offset = cal_once_len * cmp_index
                                    tmp_dst_repeat_offset = (
                                        cal_once_len // reduce_factor) * cmp_index
                                    ib_instance.emit(tvm.call_extern(
                                        "float16", "vcmp_eq",
                                        temp_andor_out.access_ptr("r",
                                                                  offset=tmp_src_repeat_offset),
                                        thred_buf.access_ptr("r"),
                                        1, 1, 1, 1, 8, 8, 8))

                                    ib_instance.emit(tvm.call_extern(
                                        "float16", "vsel",
                                        temp_out.access_ptr("rw",
                                                            offset=tmp_dst_repeat_offset),
                                        temp_andor_out.access_ptr("r",
                                                                  offset=tmp_src_repeat_offset),
                                        bias_buf.access_ptr("r"),
                                        1, 1, 1, 1, 8, 8, 8))

                            ib_instance.emit(tvm.call_extern(
                                "float16", "vconv_f162s8",
                                dst_buffers[0].access_ptr("rw",
                                                          offset=repeat_dst_offset),
                                temp_out.access_ptr("r",
                                                    offset=repeat_dst_offset),
                                tmp_repeat_times, 1, 1, 4, 8))
                        else:
                            concat_args = {}
                            concat_args["src_buffers"] = src_buffers
                            concat_args["dst_buffers"] = dst_buffers
                            concat_args["repeat_src_offset"] = repeat_src_offset
                            concat_args["repeat_dst_offset"] = repeat_dst_offset
                            concat_args["repeat_times"] = tmp_repeat_times
                            concat_args["extern_args"] = extern_args
                            concat_args["args"] = args
                            tmp_args = __concat_args(concat_args)
                            ib_instance.emit(tvm.call_extern(
                                dst_dtype, op_cmd, *tmp_args))

                        local_repeat_times -= 255
                        repeat_src_offset += cal_once_len * tmp_repeat_times
                        repeat_dst_offset += (
                            cal_once_len // reduce_factor) * tmp_repeat_times

            if remain_len > 0:
                mask1, mask2 = set_mask(remain_len)
                ib_instance.emit(tvm.call_extern(
                    dst_dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                repeat_src_offset = repeat_times * (
                    cal_once_len // reduce_factor)
                repeat_dst_offset = repeat_times * cal_once_len

                if op_cmd.lower().find("vcmp_") != -1:
                    ib_instance.emit(tvm.call_extern(dst_dtype,
                                                     op_cmd,
                                                     src_buffers[0].access_ptr(
                                                         "r",
                                                         offset=repeat_src_offset),
                                                     scalar_buf.access_ptr("r"),
                                                     1, 1, 1, 1, 8, 8, 8))
                    ib_instance.emit(tvm.call_extern(dst_dtype,
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
                    ib_instance.emit(tvm.call_extern(
                        dst_dtype, "vcmp_" + extern_args[0],
                        src_buffers[0].access_ptr("r",
                                                  offset=repeat_src_offset),
                        thred_buf.access_ptr("r"), 1, 1, 1, 1, 8, 8, 8))
                    ib_instance.emit(tvm.call_extern(
                        dst_dtype, "vsel",
                        dst_buffers[0].access_ptr("rw",
                                                  offset=repeat_dst_offset),
                        src_buffers[0].access_ptr("r",
                                                  offset=repeat_src_offset),
                        bias_buf.access_ptr("r"),
                        1, 1, 1, 1, 8, 8, 8))
                elif op_cmd == 'vcmpsel':
                    ib_instance.emit(tvm.call_extern(
                        dst_dtype, "vcmp_" + extern_args[0],
                        src_buffers[0].access_ptr("r",
                                                  offset=repeat_src_offset),
                        src_buffers[1].access_ptr("r",
                                                  offset=repeat_src_offset), 1,
                        1, 1, 1, 8, 8,
                        8))
                    ib_instance.emit(tvm.call_extern(
                        dst_dtype, "vsel",
                        dst_buffers[0].access_ptr("rw",
                                                  offset=repeat_dst_offset),
                        src_buffers[0].access_ptr("r",
                                                  offset=repeat_src_offset),
                        src_buffers[1].access_ptr("r",
                                                  offset=repeat_src_offset),
                        1, 1, 1, 1, 8, 8, 8))
                elif op_cmd == 'vlogic':
                    ib_instance.emit(tvm.call_extern(
                        "float16", "vconv_s82f16",
                        new_src_buffer0.access_ptr("rw", offset=repeat_src_offset),
                        src_buffers[0].access_ptr("r",
                                                  offset=repeat_src_offset),
                        1, 1, 1, 8, 4))
                    if extern_args[0] != 'not':
                        ib_instance.emit(tvm.call_extern(
                            "float16", "vconv_s82f16",
                            new_src_buffer1.access_ptr("rw", offset=repeat_src_offset),
                            src_buffers[1].access_ptr("r",
                                                      offset=repeat_src_offset),
                            1, 1, 1, 8, 4))

                    if extern_args[0] == 'and':
                        ib_instance.emit(tvm.call_extern("float16", "vmul",
                                                         temp_out.access_ptr(
                                                             "rw",
                                                             offset=repeat_dst_offset),
                                                         new_src_buffer0.access_ptr(
                                                             "r",
                                                             offset=repeat_src_offset),
                                                         new_src_buffer1.access_ptr(
                                                             "r",
                                                             offset=repeat_src_offset),
                                                         1, 1, 1, 1, 8, 8, 8))
                    elif extern_args[0] == 'or' or extern_args[0] == 'not':
                        if extern_args[0] == 'or':
                            ib_instance.emit(tvm.call_extern("float16", "vadd",
                                                             temp_andor_out.access_ptr(
                                                                 "rw",
                                                                 offset=repeat_dst_offset),
                                                             new_src_buffer0.access_ptr(
                                                                 "r",
                                                                 offset=repeat_src_offset),
                                                             new_src_buffer1.access_ptr(
                                                                 "r",
                                                                 offset=repeat_src_offset),
                                                             1, 1, 1, 1, 8, 8, 8))
                        elif extern_args[0] == 'not':
                            ib_instance.emit(tvm.call_extern("float16", "vsub",
                                                             temp_andor_out.access_ptr(
                                                                 "rw",
                                                                 offset=repeat_dst_offset),
                                                             new_src_buffer0.access_ptr(
                                                                 "r",
                                                                 offset=repeat_src_offset),
                                                             bias_buf.access_ptr("r"),
                                                             1, 1, 1, 1, 8, 8, 8))
                        ib_instance.emit(tvm.call_extern(
                            "float16", "vcmp_eq",
                            temp_andor_out.access_ptr("r",
                                                      offset=repeat_src_offset),
                            thred_buf.access_ptr("r"),
                            1, 1, 1, 1, 8, 8, 8))

                        ib_instance.emit(tvm.call_extern(
                            "float16", "vsel",
                            temp_out.access_ptr("rw", offset=repeat_src_offset),
                            temp_andor_out.access_ptr("r",
                                                      offset=repeat_src_offset),
                            bias_buf.access_ptr("r"),
                            1, 1, 1, 1, 8, 8, 8))

                    ib_instance.emit(tvm.call_extern(
                        "float16", "vconv_f162s8",
                        dst_buffers[0].access_ptr("rw",
                                                  offset=repeat_dst_offset),
                        temp_out.access_ptr("r", offset=repeat_src_offset),
                        1, 1, 1, 4, 8))
                else:
                    concat_args = {}
                    concat_args["src_buffers"] = src_buffers
                    concat_args["dst_buffers"] = dst_buffers
                    concat_args["repeat_src_offset"] = repeat_src_offset
                    concat_args["repeat_dst_offset"] = repeat_dst_offset
                    concat_args["repeat_times"] = 1
                    concat_args["extern_args"] = extern_args
                    concat_args["args"] = args
                    tmp_args = __concat_args(concat_args)
                    ib_instance.emit(tvm.call_extern(dst_dtype, op_cmd, *tmp_args))

                if reset_mask_pipe_line is not None:
                    mask1, mask2 = set_mask(128)
                    ib_instance.emit(tvm.call_extern(
                        dst_dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))

        # pylint: disable=too-many-arguments
        def vec_single_elewise(ib_instance, op_cmd, src_buffers, dst_buffers, op_length,
                               base_pipeline,
                               pipeline_count_list, reset_mask_pipe_line,
                               args=None):
            """
            vec_single_elewise fuction
            """
            if (len(src_buffers) != 1) or (len(dst_buffers) != 1):
                raise RuntimeError(
                    "vec_single_elewise only support ONE src buffer and ONE dst buffer ")
            if args is None:
                args = [1, 1, 8, 8]
            vec_cmd_factory_args = {}
            vec_cmd_factory_args["ib_instance"] = ib_instance
            vec_cmd_factory_args["op_cmd"] = op_cmd
            vec_cmd_factory_args["src_buffers"] = src_buffers
            vec_cmd_factory_args["dst_buffers"] = dst_buffers
            vec_cmd_factory_args["op_length"] = op_length
            vec_cmd_factory_args["reduce_factor"] = 1
            vec_cmd_factory_args["base_pipeline"] = base_pipeline
            vec_cmd_factory_args["pipeline_count_list"] = pipeline_count_list
            vec_cmd_factory_args["reset_mask_pipe_line"] = reset_mask_pipe_line
            vec_cmd_factory_args["extern_args"] = []
            vec_cmd_factory_args["args"] = args
            vec_cmd_factory_args["repeat_cal_dtype"] = None
            __vec_cmd_factory(vec_cmd_factory_args)

        # pylint: disable=too-many-arguments
        def vec_cast_elewise(ib_instance, op_cmd, src_buffers, dst_buffers, op_length,
                             base_pipeline,
                             pipeline_count_list, reset_mask_pipe_line, args,
                             repeat_cal_dtype):
            """
            vec_cast_elewise fuction
            """
            if (len(src_buffers) != 1) or (len(dst_buffers) != 1):
                raise RuntimeError(
                    "vec_cast_elewise only support ONE src buffer and ONE dst buffer ")
            if args is None:
                raise RuntimeError("vec_cast_elewise must specify args")
            vec_cmd_factory_args = {}
            vec_cmd_factory_args["ib_instance"] = ib_instance
            vec_cmd_factory_args["op_cmd"] = op_cmd
            vec_cmd_factory_args["src_buffers"] = src_buffers
            vec_cmd_factory_args["dst_buffers"] = dst_buffers
            vec_cmd_factory_args["op_length"] = op_length
            vec_cmd_factory_args["reduce_factor"] = 1
            vec_cmd_factory_args["base_pipeline"] = base_pipeline
            vec_cmd_factory_args["pipeline_count_list"] = pipeline_count_list
            vec_cmd_factory_args["reset_mask_pipe_line"] = reset_mask_pipe_line
            vec_cmd_factory_args["extern_args"] = []
            vec_cmd_factory_args["args"] = args
            vec_cmd_factory_args["repeat_cal_dtype"] = repeat_cal_dtype
            __vec_cmd_factory(vec_cmd_factory_args)

        # pylint: disable=too-many-arguments, len-as-condition
        def vec_vssingle_elewise(ib_instance, op_cmd, src_buffers, dst_buffers,
                                 op_length, base_pipeline,
                                 pipeline_count_list, reset_mask_pipe_line,
                                 extern_args, args=None):
            """
            vec_cast_elewise fuction
            """
            if (len(src_buffers) != 1) or (len(dst_buffers) != 1) or (
                    len(extern_args) < 1):
                raise RuntimeError(
                    "vec_single_elewise only support ONE src buffer and ONE dst buffer ")
            if args is None:
                args = [1, 1, 8, 8]
            vec_cmd_factory_args = {}
            vec_cmd_factory_args["ib_instance"] = ib_instance
            vec_cmd_factory_args["op_cmd"] = op_cmd
            vec_cmd_factory_args["src_buffers"] = src_buffers
            vec_cmd_factory_args["dst_buffers"] = dst_buffers
            vec_cmd_factory_args["op_length"] = op_length
            vec_cmd_factory_args["reduce_factor"] = 1
            vec_cmd_factory_args["base_pipeline"] = base_pipeline
            vec_cmd_factory_args["pipeline_count_list"] = pipeline_count_list
            vec_cmd_factory_args["reset_mask_pipe_line"] = reset_mask_pipe_line
            vec_cmd_factory_args["extern_args"] = extern_args
            vec_cmd_factory_args["args"] = args
            vec_cmd_factory_args["repeat_cal_dtype"] = None
            __vec_cmd_factory(vec_cmd_factory_args)

        # pylint: disable=too-many-arguments
        def vec_binary_elewise(ib_instance, op_cmd, src_buffers, dst_buffers, op_length,
                               base_pipeline,
                               pipeline_count_list, reset_mask_pipe_line,
                               args=None,
                               extern_args=None):
            """
            vec_binary_elewise fuction
            """
            if (len(src_buffers) != 2) or (len(dst_buffers) != 1):
                raise RuntimeError(
                    "vec_binary_elewise only support TWO src buffer and ONE dst buffer ")
            if args is None:
                args = [1, 1, 1, 8, 8, 8]
            vec_cmd_factory_args = {}
            vec_cmd_factory_args["ib_instance"] = ib_instance
            vec_cmd_factory_args["op_cmd"] = op_cmd
            vec_cmd_factory_args["src_buffers"] = src_buffers
            vec_cmd_factory_args["dst_buffers"] = dst_buffers
            vec_cmd_factory_args["op_length"] = op_length
            vec_cmd_factory_args["reduce_factor"] = 1
            vec_cmd_factory_args["base_pipeline"] = base_pipeline
            vec_cmd_factory_args["pipeline_count_list"] = pipeline_count_list
            vec_cmd_factory_args["reset_mask_pipe_line"] = reset_mask_pipe_line
            vec_cmd_factory_args["extern_args"] = extern_args
            vec_cmd_factory_args["args"] = args
            vec_cmd_factory_args["repeat_cal_dtype"] = None
            __vec_cmd_factory(vec_cmd_factory_args)

        # pylint: disable=too-many-arguments
        def vec_binary_elewise_compare(ib_instance, op_cmd, src_buffers, dst_buffers,
                                       op_length, base_pipeline,
                                       pipeline_count_list,
                                       reset_mask_pipe_line,
                                       extern_args, args=None):
            """
            vec_binary_elewise_compare fuction
            """
            if (len(src_buffers) != 2) or (len(dst_buffers) != 1):
                raise RuntimeError(
                    "vec_binary_elewise_compare only support TWO src buffer and ONE dst buffer ")
            if args is None:
                args = [1, 1, 1, 8, 8, 8]
            vec_cmd_factory_args = {}
            vec_cmd_factory_args["ib_instance"] = ib_instance
            vec_cmd_factory_args["op_cmd"] = op_cmd
            vec_cmd_factory_args["src_buffers"] = src_buffers
            vec_cmd_factory_args["dst_buffers"] = dst_buffers
            vec_cmd_factory_args["op_length"] = op_length
            vec_cmd_factory_args["reduce_factor"] = 1
            vec_cmd_factory_args["base_pipeline"] = base_pipeline
            vec_cmd_factory_args["pipeline_count_list"] = pipeline_count_list
            vec_cmd_factory_args["reset_mask_pipe_line"] = reset_mask_pipe_line
            vec_cmd_factory_args["extern_args"] = extern_args
            vec_cmd_factory_args["args"] = args
            vec_cmd_factory_args["repeat_cal_dtype"] = None
            __vec_cmd_factory(vec_cmd_factory_args)

        # pylint: disable=consider-using-in
        def reduce_nist_axis(shape, op_instance, reduce_axis, dtype):
            """
            factory funtion for nist axis reduction operations. For tensorize

            Parameters
            ----------
            data_shape : tuple or list
                The arrays to concatenate

            op_instance : string
                operation type, supports reduce_sum, reduce_min, reduce_max

            reduce_axis : int
                reduce axis num

            dtype : string
                The source data type

            Returns
            -------
            ret : TensorIntrin
                A TensorIntrin for nist axis reduction operations
                that can be used in tensorize schedule.

            Example :
            -------
                tensorize code like :
                    for (k1, 0, 402) {
                        A0.local.UB[k1] = (A0.local.UB[k1] + A0.local.UB[(k1 + 512)])
                    }
            """
            pipe_line_index = __cal_map_sum(vec_count)
            vec_count["reduce_nist_axis"] += 5

            op_dict = {"reduce_sum": (tvm.sum, "add"),
                       "reduce_prod": (tvm.prod, "mul"),
                       "reduce_min": (tvm.min, "min"),
                       "reduce_max": (tvm.max, "max")}
            op_attr = op_dict.get(op_instance)
            if not op_attr:
                raise RuntimeError("op %s not support yet" % op_instance)
            func, cmd = op_attr

            last_reduce_axis = (reduce_axis == -1) or (
                reduce_axis == len(shape) - 1)
            reduce_axis_index = reduce_axis if reduce_axis >= 0 else len(
                shape) + reduce_axis
            if last_reduce_axis:
                raise RuntimeError(
                    "reduce_nist_axis only support nist axis tensorize")
            index_k = tvm.reduce_axis((0, 1), name='index_k')

            res_shape = produce_res_shape(shape, reduce_axis_index)


            tensor_a = tvm.placeholder(shape, dtype, name='A0')

            res_b = tvm.compute(res_shape, lambda *indice: func(
                flatten_indice(tensor_a, produce_res_indice(indice, reduce_axis_index,
                                                            index_k)), axis=index_k),
                                name='B')

            buf_a = tvm.decl_buffer(tensor_a.shape, tensor_a.dtype,
                                    name="A0_buf",
                                    offset_factor=1,
                                    scope=buffer_scope,
                                    strides=[tvm.var() for _ in
                                             range(len(tensor_a.shape))],
                                    data_alignment=get_data_alignment(
                                        tensor_a.dtype))

            buf_b = tvm.decl_buffer(res_b.shape, res_b.dtype,
                                    name="B_buf",
                                    offset_factor=1,
                                    scope=buffer_scope,
                                    strides=[tvm.var() for _ in
                                             range(len(res_b.shape))],
                                    data_alignment=get_data_alignment(res_b.dtype))

            v_cmd = "v" + cmd
            size = functools_reduce(lambda i, j: i * j, shape)
            burst_ele_num = 256 // get_bit_len(dtype)
            dst_stride_m0 = 1
            src0_stride_m0 = 1
            src1_stride_m0 = 1

            dst_stride_m1 = 8
            src0_stride_m1 = 8
            src1_stride_m1 = 8

            def intrin_func(ins, outs):
                """
                intrin_func
                """
                instace_a0 = ins[0]
                out = outs[0]

                def instr(flag):
                    """
                    instr fuction
                    """
                    ib_instance = tvm.ir_builder.create()
                    if flag == 'body':
                        sid = 0
                        burst = (size + burst_ele_num - 1) // burst_ele_num
                        nburst = 1
                        body_func_name = 'copy_ubuf_to_ubuf'
                        ib_instance.scope_attr(param.CCE_AXIS, "coproc_scope",
                                               2 + total_pipe_line * pipe_line_index)
                        ib_instance.emit(tvm.call_extern(
                            out.dtype, body_func_name,
                            out.access_ptr("rw"),
                            instace_a0.access_ptr("r"),
                            sid,
                            nburst,
                            burst,
                            dst_stride_m0,
                            src0_stride_m0
                        ))
                    elif flag == 'update':
                        args = [dst_stride_m0,  # dst stride
                                src0_stride_m0,  # src0 stride
                                src1_stride_m0,
                                dst_stride_m1,
                                src0_stride_m1,
                                src1_stride_m1
                               ]
                        pipeline_count_list = [1]
                        src_buffers = [instace_a0, out]
                        dst_buffers = [out]
                        vec_binary_elewise(ib_instance, v_cmd, src_buffers, dst_buffers,
                                           size,
                                           pipe_line_index, pipeline_count_list,
                                           0, args)
                    return ib_instance.get()

                return instr('body'), None, instr('update')

            return tvm.decl_tensor_intrin(res_b.op, intrin_func,
                                          binds={tensor_a: buf_a, res_b: buf_b})

        def compute_prod_segmentation(ib_instace, total_len, src_a, src_c):
            """
            compute_prod_segmentation fuction
            """
            vec_count["reduce_last_axis"] += 4

            dst_stride_m0 = 1
            src0_stride_m0 = 1
            src1_stride_m0 = 1
            dst_stride_m1 = 0
            src0_stride_m1 = 8
            src1_stride_m1 = 0

            repeat_times = total_len // 128
            remain_len = total_len - repeat_times * 128
            src_offset = 0
            if repeat_times > 0:
                local_repeat_times = repeat_times
                while local_repeat_times > 0:
                    if local_repeat_times > 255:
                        tmp_repeat_times = 255
                    else:
                        tmp_repeat_times = local_repeat_times
                    with ib_instace.new_scope():
                        ib_instace.emit(tvm.call_extern(
                            src_c.dtype, "vmul",
                            src_c.access_ptr("rw"),
                            src_a.access_ptr("r", offset=src_offset),
                            src_c.access_ptr("r"),
                            tmp_repeat_times,
                            dst_stride_m0, src0_stride_m0, src1_stride_m0,
                            dst_stride_m1, src0_stride_m1, src1_stride_m1))
                        src_offset += 128 * tmp_repeat_times
                        local_repeat_times -= 255

            if remain_len > 0:
                with ib_instace.new_scope():
                    mask1, mask2 = set_mask(remain_len)
                    ib_instace.emit(tvm.call_extern(
                        src_c.dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))
                with ib_instace.new_scope():
                    ib_instace.emit(tvm.call_extern(
                        src_c.dtype, "vmul",
                        src_c.access_ptr("rw"),
                        src_a.access_ptr("r", offset=src_offset),
                        src_c.access_ptr("r"),
                        1,
                        dst_stride_m0, src0_stride_m0, src1_stride_m0,
                        dst_stride_m1, src0_stride_m1, src1_stride_m1))
                with ib_instace.new_scope():
                    mask1, mask2 = set_mask(128)
                    ib_instace.emit(tvm.call_extern(
                        src_c.dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))

        def compute_last_128_numbers(ib_instace, input_c):
            """
            compute_last_128_numbers fuction
            """
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
                with ib_instace.new_scope():
                    mask1, mask2 = set_mask(num)
                    ib_instace.emit(tvm.call_extern(
                        input_c.dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))
                with ib_instace.new_scope():
                    ib_instace.emit(tvm.call_extern(
                        input_c.dtype, "vmul",
                        input_c.access_ptr("rw"),
                        input_c.access_ptr("r"),
                        input_c.access_ptr("r", offset=num),
                        1,
                        dst_stride_m0, src0_stride_m0, src1_stride_m0,
                        dst_stride_m1, src0_stride_m1, src1_stride_m1))

            # 64*64
            fold_mul_1(64)
            # 32*32
            fold_mul_1(32)
            # 16*16
            fold_mul_1(16)

            # last 16 numbers
            buf_var = ib_instace.allocate(input_c.dtype, (128,), "d_buf", scope="local.UB")
            d_buf = tvm.decl_buffer((128,), input_c.dtype, "d_buf", scope="local.UB",
                                    data=buf_var)
            reg = ib_instace.allocate(input_c.dtype, (2,), name="reg_buf",
                                      scope=param.scope_reg)

            def fold_mul_2(num):
                '''
                fold mul of last 16 data.
                Because of alignment constraints,the remanent data must split to two parts.
                The front half stay at source address,and the back half move to buffer address.
                Then, the two half data mul by vmul after setting mask.
                '''
                with ib_instace.for_range(0, num, name="ii") as offset_num:
                    ib_instace.emit(tvm.call_extern(input_c.dtype, "reg_mov",
                                                    tvm.call_extern(reg.dtype, "reg",
                                                                    reg[0]),
                                                    input_c.access_ptr("rw",
                                                                       offset=(offset_num + num))))

                    with ib_instace.new_scope():
                        ib_instace.emit(tvm.call_extern(
                            reg.dtype, "reg_mov",
                            d_buf.access_ptr("rw", offset=offset_num),
                            tvm.call_extern(reg.dtype, "reg", reg[0])))

                with ib_instace.new_scope():
                    mask1, mask2 = set_mask(num)
                    ib_instace.emit(tvm.call_extern(
                        input_c.dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))

                with ib_instace.new_scope():
                    ib_instace.emit(tvm.call_extern(
                        input_c.dtype, "vmul",
                        input_c.access_ptr("rw"),
                        input_c.access_ptr("r"),
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

            with ib_instace.new_scope():
                mask1, mask2 = set_mask(128)
                ib_instace.emit(tvm.call_extern(
                    input_c.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))

        # pylint: disable=too-many-branches
        def reduce_prod_last_axis(data_shape, dtype):
            """
            factory funtion for last axis reduction prod operations. For tensorize

            Parameters
            ----------
            data_shape : tuple or list
                The arrays to concatenate

            dtype : string
                The source data type

            Returns
            -------
            ret : TensorIntrin
                A TensorIntrin for last axis reduction operations
                that can be used in tensorize schedule.

            """
		    # pylint: disable=too-many-statements
            pipe_line_index = __cal_map_sum(vec_count)
            vec_count["reduce_last_axis"] += 4

            if dtype != "float16":
                raise ValueError(
                    "reduce_last_axis only support float16 while dtype is %s" % dtype)
            func = tvm.prod

            tensor_a = tvm.placeholder(data_shape, dtype, name='A0')
            if len(data_shape) > 1:
                reduce_axises = []
                for index, value in enumerate(data_shape):
                    reduce_axises.append(
                        tvm.reduce_axis((0, value), name='k' + str(index + 1)))
                res_b = tvm.compute((1,), lambda i: func(tensor_a(*reduce_axises),

                                                         axis=reduce_axises),
                                    name='B')
            else:
                k = tvm.reduce_axis((0, data_shape[-1]), name='k')
                res_b = tvm.compute((1,), lambda i: func(tensor_a[k], axis=[k]), name='B')
            a_buf = tvm.decl_buffer(tensor_a.shape, tensor_a.dtype,
                                    name="A0_buf",
                                    offset_factor=1,
                                    scope=buffer_scope,
                                    strides=[tvm.var() for _ in
                                             range(len(tensor_a.shape))],
                                    data_alignment=get_data_alignment(tensor_a.dtype))

            b_buf = tvm.decl_buffer(res_b.shape, res_b.dtype,
                                    name="B_buf",
                                    offset_factor=1,
                                    scope=buffer_scope,
                                    strides=[tvm.var() for _ in
                                             range(len(res_b.shape))],
                                    data_alignment=get_data_alignment(res_b.dtype))

            dst_stride_m0 = 1
            src0_stride_m0 = 1
            src1_stride_m0 = 1
            dst_stride_m1 = 8
            src0_stride_m1 = 8
            src1_stride_m1 = 8
            stride_inside = 1
            stride_outside = 8

            if len(data_shape) > 1:
                total_len = functools_reduce(lambda x, y: x * y, data_shape[:])
            else:
                total_len = data_shape[-1]

            def intrin_func(ins, outs):
                """
                intrin_func fuction
                """
                ins_a = ins[0]
                out_b = outs[0]

                def instr(flag):
                    """
                    instr fuction
                    """
                    ib_instance = tvm.ir_builder.create()
                    if flag == "update":
                        pipeline_count = 0
                        # __ubuf__ half C_buf[128] vlaue: { 1.0 }
                        buf_var = ib_instance.allocate(dtype, (128,), "c_buf",
                                                       scope="local.UB")
                        c_buf = tvm.decl_buffer((128,), dtype, "c_buf",
                                                scope="local.UB",
                                                data=buf_var)
                        with ib_instance.new_scope():
                            ib_instance.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                   2 + total_pipe_line * (
                                                       pipe_line_index + 2 * pipeline_count))
                            ib_instance.emit(tvm.call_extern(
                                c_buf.dtype, "vector_dup",
                                c_buf.access_ptr("rw", offset=0),
                                tvm.const(1.0, dtype="float16"),
                                1,
                                stride_inside, stride_inside,
                                stride_outside, stride_outside))
                            pipeline_count = 1 - pipeline_count

                        # Compute the product segment
                        compute_prod_segmentation(ib_instance, total_len, ins_a, c_buf)

                        # Compute the product of the last 128 numbers
                        compute_last_128_numbers(ib_instance, c_buf)

                        with ib_instance.new_scope():
                            ib_instance.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                   2 + total_pipe_line * (
                                                       pipe_line_index + 2 * pipeline_count))
                            mask1, mask2 = set_mask(16)
                            ib_instance.emit(tvm.call_extern(
                                out_b.dtype, "set_vector_mask",
                                tvm.const(mask1, dtype="uint64"),
                                tvm.const(mask2, dtype="uint64")))
                            pipeline_count = 1 - pipeline_count

                        with ib_instance.new_scope():
                            ib_instance.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                   2 + total_pipe_line * (
                                                       pipe_line_index + 2 * pipeline_count))
                            ib_instance.emit(tvm.call_extern(
                                out_b.dtype, "vmul",
                                out_b.access_ptr("rw"),
                                out_b.access_ptr("r"),
                                c_buf.access_ptr("r"),
                                1,
                                dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                dst_stride_m1, src0_stride_m1, src1_stride_m1))
                            pipeline_count = 1 - pipeline_count
                        with ib_instance.new_scope():
                            ib_instance.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                   2 + total_pipe_line * (
                                                       pipe_line_index + 1))
                            mask1, mask2 = set_mask(128)
                            ib_instance.emit(tvm.call_extern(
                                out_b.dtype, "set_vector_mask",
                                tvm.const(mask1, dtype="uint64"),
                                tvm.const(mask2, dtype="uint64")))
                    elif flag == "body":
                        pipeline_count = 0
                        # __ubuf__ half C_buf[128] vlaue: { 1.0 }
                        buf_var = ib_instance.allocate(dtype, (128,), "c_buf",
                                                       scope="local.UB")
                        c_buf = tvm.decl_buffer((128,), dtype, "c_buf",
                                                scope="local.UB",
                                                data=buf_var)
                        with ib_instance.new_scope():
                            ib_instance.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                   2 + total_pipe_line * (
                                                       pipe_line_index + 2 * pipeline_count))
                            ib_instance.emit(tvm.call_extern(
                                c_buf.dtype, "vector_dup",
                                c_buf.access_ptr("rw", offset=0),
                                tvm.const(1.0, dtype="float16"),
                                1,
                                stride_inside,
                                stride_inside,
                                stride_outside,
                                stride_outside))
                            pipeline_count = 1 - pipeline_count

                        # Compute the product segment
                        compute_prod_segmentation(ib_instance, total_len, ins_a, c_buf)

                        # Compute the product of the last 128 numbers
                        compute_last_128_numbers(ib_instance, c_buf)

                        with ib_instance.new_scope():
                            ib_instance.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                   2 + total_pipe_line * (
                                                       pipe_line_index + 2 * pipeline_count))
                            ib_instance.emit(tvm.call_extern(
                                out_b.dtype, "copy_ubuf_to_ubuf",
                                out_b.access_ptr("rw"),
                                c_buf.access_ptr("r"),
                                0, 1, 1, 0, 0))
                        with ib_instance.new_scope():
                            ib_instance.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                   2 + total_pipe_line * (
                                                       pipe_line_index + 1))
                            mask1, mask2 = set_mask(128)
                            ib_instance.emit(tvm.call_extern(
                                out_b.dtype, "set_vector_mask",
                                tvm.const(mask1, dtype="uint64"),
                                tvm.const(mask2, dtype="uint64")))
                    return ib_instance.get()

                # return a triple of normal-set, reset, update
                return (instr("body"), None, instr("update"))

            return tvm.decl_tensor_intrin(res_b.op, intrin_func,
                                          binds={tensor_a: a_buf, res_b: b_buf})

        def reduce_last_axis(data_shape, op_instance, dtype):
            """
            factory funtion for last axis reduction operations. For tensorize

            Parameters
            ----------
            data_shape : tuple or list
                The arrays to concatenate

            op_instance : string
                operation type, supports reduce_sum, reduce_min, reduce_max

            dtype : string
                The source data type

            Returns
            -------
            ret : TensorIntrin
                A TensorIntrin for last axis reduction operations
                that can be used in tensorize schedule.

            Example :
            -------
                tensorize code like :
                    for (k1, 0, 402) {
                        A0.local.UB[1024] = (A0.local.UB[1024] + A0.local.UB[(k1 + 512)])
                    }
            """
            # pylint: disable=too-many-statements
            if op_instance == "reduce_prod":
                return reduce_prod_last_axis(data_shape, dtype)
            pipe_line_index = __cal_map_sum(vec_count)
            vec_count["reduce_last_axis"] += 4
            op_dict = {"reduce_sum": (tvm.sum, "add"),
                       "reduce_min": (tvm.min, "min"),
                       "reduce_max": (tvm.max, "max")}
            op_attr = op_dict.get(op_instance)
            if not op_attr:
                raise RuntimeError("op %s not support yet" % op_instance)

            func, cmd = op_attr

            if cmd in ["add", "max", "min"]:
                if dtype not in ["float16", "float32"]:
                    raise ValueError(
                        "reduce_last_axis only support float16 and \
                            float32 while dtype is %s" % dtype)
            else:
                if dtype != "float16":
                    raise ValueError(
                        "reduce_last_axis only support float16 while dtype is %s" % dtype)

            cross_element = 128
            if cmd == "add" and dtype == "float32":
                cross_element = 64

            tensor_a = tvm.placeholder(data_shape, dtype, name='A0')

            if len(data_shape) > 1:
                reduce_axises = []
                for index, value in enumerate(data_shape):
                    reduce_axises.append(
                        tvm.reduce_axis((0, value), name='k' + str(index + 1)))
                tensor_b = tvm.compute((1,), lambda i: func(tensor_a(*reduce_axises),
                                                            axis=reduce_axises),
                                       name='B')
            else:
                k = tvm.reduce_axis((0, data_shape[-1]), name='k')
                tensor_b = tvm.compute((1,), lambda i: func(tensor_a[k], axis=[k]), name='B')

            a_buf = tvm.decl_buffer(tensor_a.shape, tensor_a.dtype,
                                    name="A0_buf",
                                    offset_factor=1,
                                    scope=buffer_scope,
                                    strides=[tvm.var() for _ in
                                             range(len(tensor_a.shape))],
                                    data_alignment=get_data_alignment(tensor_a.dtype))

            b_buf = tvm.decl_buffer(tensor_b.shape, tensor_b.dtype,
                                    name="B_buf",
                                    offset_factor=1,
                                    scope=buffer_scope,
                                    strides=[tvm.var() for _ in
                                             range(len(tensor_b.shape))],
                                    data_alignment=get_data_alignment(tensor_b.dtype))
            vcg_cmd = "vcg" + cmd
            v_cmd = "v" + cmd
            vc_cmd = "vc" + cmd
            src0_stride = 1
            src1_stride = 8
            dst_stride = 1

            dst_stride_m0 = 1  # dst stridem0
            src0_stride_m0 = 1  # src0 stridem0
            src1_stride_m0 = 1  # src1 stridem0
            dst_stride_m1 = 8  # dst stridem1
            src0_stride_m1 = 8  # src0 stridem1
            src1_stride_m1 = 8  # src1 stridem1

            max_cal_times = 254
            if len(data_shape) > 1:
                total_len = functools_reduce(lambda x, y: x * y, data_shape[:])
            else:
                total_len = data_shape[-1]

            def intrin_func(ins, outs):
                """
                intrin_func fuction
                """
                # pylint: disable=too-many-branches, too-many-statements
                ins_a = ins[0]
                out_b = outs[0]

                emit_cmd = vcg_cmd
                res_block_size = 8
                if cmd == "add":
                    emit_cmd = vc_cmd
                    res_block_size = 1

                def __is_need_add_stride_para():
                    is_vcmax = vc_cmd == "vcmax"
                    is_vcmin = vc_cmd == "vcmin"
                    is_vcmax_v200 = out_b.dtype.lower() == "float32" and \
                                    cce_conf.intrinsic_check_support("Intrinsic_vcmax", "float32")
                    is_vcmin_v200 = out_b.dtype.lower() == "float32" and \
                                    cce_conf.intrinsic_check_support("Intrinsic_vcmin", "float32")
                    if (is_vcmax and is_vcmax_v200) or (is_vcmin and is_vcmin_v200):
                        return True
                    return False

                def instr(flag):
                    """
                    instr function
                    """
                    # pylint: disable=too-many-branches, too-many-statements
                    is_need_add_stride_para = __is_need_add_stride_para()
                    ib_ins = tvm.ir_builder.create()
                    if flag == "update":
                        local_total_len = total_len
                        pipeline_count = 0
                        while local_total_len > cross_element:
                            repeat_times = local_total_len // cross_element
                            remain_len = local_total_len - repeat_times * cross_element
                            if repeat_times > 0:
                                local_repeat_times = repeat_times
                                with ib_ins.new_scope():
                                    ib_ins.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope",
                                                      2 + total_pipe_line * (
                                                          pipe_line_index + 2 * pipeline_count))
                                    repeat_dst_offset = 0
                                    repeat_src_offset = 0
                                    while local_repeat_times > 0:
                                        if local_repeat_times > max_cal_times:
                                            tmp_repeat_times = max_cal_times
                                        else:
                                            tmp_repeat_times = local_repeat_times

                                        ib_ins.emit(tvm.call_extern(
                                            out_b.dtype, emit_cmd,
                                            ins_a.access_ptr("rw",
                                                             offset=repeat_dst_offset),
                                            ins_a.access_ptr("r",
                                                             offset=repeat_src_offset),
                                            tmp_repeat_times,
                                            dst_stride,
                                            src0_stride,
                                            src1_stride
                                        ))
                                        local_repeat_times -= max_cal_times
                                        repeat_dst_offset += res_block_size * tmp_repeat_times
                                        repeat_src_offset += cross_element * tmp_repeat_times
                            if remain_len > 0:
                                with ib_ins.new_scope():
                                    pipeline_count = 1 - pipeline_count
                                    ib_ins.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope",
                                                      2 + total_pipe_line * (
                                                          pipe_line_index + 2 * pipeline_count))
                                    mask1, mask2 = set_mask(remain_len)
                                    ib_ins.emit(tvm.call_extern(
                                        out_b.dtype, "set_vector_mask",
                                        tvm.const(mask1, dtype="uint64"),
                                        tvm.const(mask2, dtype="uint64")))
                                with ib_ins.new_scope():
                                    pipeline_count = 1 - pipeline_count
                                    ib_ins.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope",
                                                      2 + total_pipe_line * (
                                                          pipe_line_index + 2 * pipeline_count))

                                    ib_ins.emit(tvm.call_extern(
                                        out_b.dtype, emit_cmd,
                                        ins_a.access_ptr("rw",
                                                         offset=repeat_times * res_block_size),
                                        ins_a.access_ptr("r",
                                                         offset=repeat_times * cross_element),
                                        1,
                                        dst_stride,
                                        src0_stride,
                                        src1_stride
                                    ))
                                with ib_ins.new_scope():
                                    pipeline_count = 1 - pipeline_count
                                    ib_ins.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope",
                                                      2 + total_pipe_line * (
                                                          pipe_line_index + 2 * pipeline_count))
                                    mask1, mask2 = set_mask(128)
                                    ib_ins.emit(tvm.call_extern(
                                        out_b.dtype, "set_vector_mask",
                                        tvm.const(mask1, dtype="uint64"),
                                        tvm.const(mask2, dtype="uint64")))

                            if cmd == "add":
                                if dtype == "float32":
                                    local_total_len = repeat_times + (
                                        remain_len + 63) // 64
                                else:
                                    local_total_len = repeat_times + (
                                        remain_len + 127) // 128
                            else:
                                local_total_len = repeat_times * 8 + (
                                    remain_len + 15) // 16

                            pipeline_count = 1 - pipeline_count

                        if local_total_len > 1:
                            with ib_ins.new_scope():
                                mask1, mask2 = set_mask(local_total_len)
                                ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                  2 + total_pipe_line * (
                                                      pipe_line_index + 2 * pipeline_count))
                                ib_ins.emit(tvm.call_extern(
                                    out_b.dtype, "set_vector_mask",
                                    tvm.const(mask1, dtype="uint64"),
                                    tvm.const(mask2, dtype="uint64")))
                                pipeline_count = 1 - pipeline_count

                            with ib_ins.new_scope():
                                ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                  2 + total_pipe_line * (
                                                      pipe_line_index + 2 * pipeline_count))
                                def __do_ib_ins_emit():
                                    if is_need_add_stride_para:
                                        ib_ins.emit(tvm.call_extern(
                                            out_b.dtype, vc_cmd,
                                            ins_a.access_ptr("rw"),
                                            ins_a.access_ptr("r"),
                                            1,
                                            dst_stride,
                                            src0_stride,
                                            src1_stride,
                                            0,
                                            0,
                                            0
                                        ))
                                    else:
                                        ib_ins.emit(tvm.call_extern(
                                            out_b.dtype, vc_cmd,
                                            ins_a.access_ptr("rw"),
                                            ins_a.access_ptr("r"),
                                            1,
                                            dst_stride,
                                            src0_stride,
                                            src1_stride
                                        ))
                                __do_ib_ins_emit()
                                pipeline_count = 1 - pipeline_count

                        # 1 block save the result, v_cmd to compute the result,
                        # so for float32 dtype, set mask = 8
                        if cmd == "add" and dtype == "float32":
                            v_cmd_mask = 8
                        else:
                            v_cmd_mask = 16

                        with ib_ins.if_scope(out_b.elem_offset > 0):
                            new_buffer_b = __apply_for_new_alloc(ib_ins, out_b.dtype, (1,))
                            ib_ins.emit(tvm.call_extern(
                                out_b.dtype, "reg_mov",
                                new_buffer_b.access_ptr("rw"),
                                out_b.access_ptr("r")))
                            with ib_ins.new_scope():
                                ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                  2 + total_pipe_line * (
                                                      pipe_line_index + 2 * pipeline_count))
                                mask1, mask2 = set_mask(v_cmd_mask)
                                ib_ins.emit(tvm.call_extern(
                                    out_b.dtype, "set_vector_mask",
                                    tvm.const(mask1, dtype="uint64"),
                                    tvm.const(mask2, dtype="uint64")))
                                pipeline_count = 1 - pipeline_count
                            with ib_ins.new_scope():
                                ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                  2 + total_pipe_line * (
                                                      pipe_line_index + 2 * pipeline_count))
                                ib_ins.emit(tvm.call_extern(
                                    out_b.dtype, v_cmd,
                                    new_buffer_b.access_ptr("rw"),
                                    new_buffer_b.access_ptr("r"),
                                    ins_a.access_ptr("r"),
                                    1,
                                    dst_stride_m0,  # dst stridem0
                                    src0_stride_m0,  # src0 stridem0
                                    src1_stride_m0,  # src1 stridem0
                                    dst_stride_m1,  # dst stridem1
                                    src0_stride_m1,  # src0 stridem1
                                    src1_stride_m1  # src1 stridem1
                                ))
                                pipeline_count = 1 - pipeline_count
                            with ib_ins.new_scope():
                                ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                  2 + total_pipe_line * (
                                                      pipe_line_index + 1))
                                mask1, mask2 = set_mask(128)
                                ib_ins.emit(tvm.call_extern(
                                    out_b.dtype, "set_vector_mask",
                                    tvm.const(mask1, dtype="uint64"),
                                    tvm.const(mask2, dtype="uint64")))
                            ib_ins.emit(tvm.call_extern(
                                out_b.dtype, "reg_mov",
                                out_b.access_ptr("rw"),
                                new_buffer_b.access_ptr("r")))
                        with ib_ins.else_scope():
                            with ib_ins.new_scope():
                                ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                  2 + total_pipe_line * (
                                                      pipe_line_index + 2 * pipeline_count))
                                mask1, mask2 = set_mask(v_cmd_mask)
                                ib_ins.emit(tvm.call_extern(
                                    out_b.dtype, "set_vector_mask",
                                    tvm.const(mask1, dtype="uint64"),
                                    tvm.const(mask2, dtype="uint64")))
                                pipeline_count = 1 - pipeline_count
                            with ib_ins.new_scope():
                                ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                  2 + total_pipe_line * (
                                                      pipe_line_index + 2 * pipeline_count))
                                ib_ins.emit(tvm.call_extern(
                                    out_b.dtype, v_cmd,
                                    out_b.access_ptr("rw"),
                                    out_b.access_ptr("r"),
                                    ins_a.access_ptr("r"),
                                    1,
                                    dst_stride_m0,  # dst stridem0
                                    src0_stride_m0,  # src0 stridem0
                                    src1_stride_m0,  # src1 stridem0
                                    dst_stride_m1,  # dst stridem1
                                    src0_stride_m1,  # src0 stridem1
                                    src1_stride_m1  # src1 stridem1
                                ))
                                pipeline_count = 1 - pipeline_count
                            with ib_ins.new_scope():
                                ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                  2 + total_pipe_line * (
                                                      pipe_line_index + 1))
                                mask1, mask2 = set_mask(128)
                                ib_ins.emit(tvm.call_extern(
                                    out_b.dtype, "set_vector_mask",
                                    tvm.const(mask1, dtype="uint64"),
                                    tvm.const(mask2, dtype="uint64")))
                    elif flag == "body":
                        local_total_len = total_len
                        pipeline_count = 0
                        while local_total_len > cross_element:
                            repeat_times = local_total_len // cross_element
                            remain_len = local_total_len - repeat_times * cross_element
                            if repeat_times > 0:
                                local_repeat_times = repeat_times
                                with ib_ins.new_scope():
                                    ib_ins.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope",
                                                      2 + total_pipe_line * (
                                                          pipe_line_index + 2 * pipeline_count))
                                    repeat_dst_offset = 0
                                    repeat_src_offset = 0
                                    while local_repeat_times > 0:
                                        if local_repeat_times > max_cal_times:
                                            tmp_repeat_times = max_cal_times
                                        else:
                                            tmp_repeat_times = local_repeat_times

                                        ib_ins.emit(tvm.call_extern(
                                            out_b.dtype, emit_cmd,
                                            ins_a.access_ptr("rw",
                                                             offset=repeat_dst_offset),
                                            ins_a.access_ptr("r",
                                                             offset=repeat_src_offset),
                                            tmp_repeat_times,
                                            dst_stride,
                                            src0_stride,
                                            src1_stride
                                        ))
                                        local_repeat_times -= max_cal_times
                                        repeat_dst_offset += res_block_size * tmp_repeat_times
                                        repeat_src_offset += cross_element * tmp_repeat_times

                            if remain_len > 0:
                                with ib_ins.new_scope():
                                    pipeline_count = 1 - pipeline_count
                                    ib_ins.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope",
                                                      2 + total_pipe_line * (
                                                          pipe_line_index + 2 * pipeline_count))
                                    mask1, mask2 = set_mask(remain_len)
                                    ib_ins.emit(tvm.call_extern(
                                        out_b.dtype, "set_vector_mask",
                                        tvm.const(mask1, dtype="uint64"),
                                        tvm.const(mask2, dtype="uint64")))
                                with ib_ins.new_scope():
                                    pipeline_count = 1 - pipeline_count
                                    ib_ins.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope",
                                                      2 + total_pipe_line * (
                                                          pipe_line_index + 2 * pipeline_count))

                                    ib_ins.emit(tvm.call_extern(
                                        out_b.dtype, emit_cmd,
                                        ins_a.access_ptr("rw",
                                                         offset=repeat_times * res_block_size),
                                        ins_a.access_ptr("r",
                                                         offset=repeat_times * cross_element),
                                        1,
                                        dst_stride,
                                        src0_stride,
                                        src1_stride
                                    ))
                                with ib_ins.new_scope():
                                    pipeline_count = 1 - pipeline_count
                                    ib_ins.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope",
                                                      2 + total_pipe_line * (
                                                          pipe_line_index + 2 * pipeline_count))
                                    mask1, mask2 = set_mask(128)
                                    ib_ins.emit(tvm.call_extern(
                                        out_b.dtype, "set_vector_mask",
                                        tvm.const(mask1, dtype="uint64"),
                                        tvm.const(mask2, dtype="uint64")))

                            if cmd == "add":
                                if dtype == "float32":
                                    local_total_len = repeat_times + (
                                        remain_len + 63) // 64
                                else:
                                    local_total_len = repeat_times + (
                                        remain_len + 127) // 128
                            else:
                                local_total_len = repeat_times * 8 + (

                                    remain_len + 15) // 16

                            pipeline_count = 1 - pipeline_count

                        if local_total_len > 1:
                            with ib_ins.new_scope():
                                mask1, mask2 = set_mask(local_total_len)
                                ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                  2 + total_pipe_line * (
                                                      pipe_line_index + 2 * pipeline_count))
                                ib_ins.emit(tvm.call_extern(
                                    out_b.dtype, "set_vector_mask",
                                    tvm.const(mask1, dtype="uint64"),
                                    tvm.const(mask2, dtype="uint64")))
                                pipeline_count = 1 - pipeline_count
                            with ib_ins.new_scope():
                                ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope",
                                                  2 + total_pipe_line * (
                                                      pipe_line_index + 2 * pipeline_count))
                                def __do_ib_ins_emit():
                                    if is_need_add_stride_para:
                                        ib_ins.emit(tvm.call_extern(
                                            out_b.dtype, vc_cmd,
                                            ins_a.access_ptr("rw"),
                                            ins_a.access_ptr("r"),
                                            1,
                                            dst_stride,
                                            src0_stride,
                                            src1_stride,
                                            0,
                                            0,
                                            0
                                        ))
                                    else:
                                        ib_ins.emit(tvm.call_extern(
                                            out_b.dtype, vc_cmd,
                                            ins_a.access_ptr("rw"),
                                            ins_a.access_ptr("r"),
                                            1,
                                            dst_stride,
                                            src0_stride,
                                            src1_stride
                                        ))
                                __do_ib_ins_emit()
                                pipeline_count = 1 - pipeline_count
                        with ib_ins.new_scope():
                        #     ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope",
                        #                       2 + total_pipe_line * (
                        #                           pipe_line_index + 2 * pipeline_count))
                            with ib_ins.if_scope(out_b.elem_offset > 0):
                                ib_ins.emit(tvm.call_extern(
                                    out_b.dtype, "reg_mov",
                                    out_b.access_ptr("rw"),
                                    ins_a.access_ptr("r")))
                            with ib_ins.else_scope():
                                ib_ins.emit(tvm.call_extern(
                                    out_b.dtype, "copy_ubuf_to_ubuf",
                                    out_b.access_ptr("rw"),
                                    ins_a.access_ptr("r"),
                                    0,
                                    1,
                                    1,
                                    1,
                                    1
                                ))
                        with ib_ins.new_scope():
                            ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              2 + total_pipe_line * (
                                                  pipe_line_index + 1))
                            mask1, mask2 = set_mask(128)
                            ib_ins.emit(tvm.call_extern(
                                out_b.dtype, "set_vector_mask",
                                tvm.const(mask1, dtype="uint64"),
                                tvm.const(mask2, dtype="uint64")))
                    return ib_ins.get()

                # return a triple of normal-set, reset, update
                return (instr("body"), None, instr("update"))

            return tvm.decl_tensor_intrin(tensor_b.op, intrin_func,
                                          binds={tensor_a: a_buf, tensor_b: b_buf})

        # pylint: disable=too-many-branches, too-many-arguments
        def elewise_binary_intrin_cce(shape, op_ins, dst_dtype, src_dtype,
                                      is_same=False, args=None):
            """factory funtion for elewise operations of binary op. For tensorize

            Parameters
            ----------
            shapes : tuple or list
                The arrays to concatenate

            op_ins : string
                operation type, supports elewise_binary_add,
                elewise_binary_sub, elewise_binary_mul,
                elewise_binary_min, elewise_binary_max

            dst_dtype :
                The destination data type

            src_dtype :
                The source data type

            args : tvm.const
                scalar to broadcast.

            Returns
            -------
            ret : TensorIntrin
                A TensorIntrin for elewise operations of binary op
                that can be used in tensorize schedule.
            """
            # for pylint, reserve argument
            dst_dtype = dst_dtype

            pipe_line_index = __cal_map_sum(vec_count)
            vec_count["elewise_binary_intrin_cce"] += 4

            ten_a = tvm.placeholder(shape, src_dtype, name='A0')
            if is_same:
                ten_a1 = ten_a
            else:
                ten_a1 = tvm.placeholder(shape, src_dtype, name='A1')

            if op_ins.lower() == "elewise_binary_add":
                out = tvm.compute(shape, lambda *indice: flatten_indice(ten_a,
                                                                        indice)
                                  + flatten_indice(ten_a1, indice), name="out")
                intrinsic_cmd = "vadd"
            elif op_ins.lower() == "elewise_binary_sub":
                out = tvm.compute(shape, lambda *indice: flatten_indice(ten_a,
                                                                        indice)
                                  - flatten_indice(ten_a1, indice), name="out")
                intrinsic_cmd = "vsub"
            elif op_ins.lower() == "elewise_binary_mul":
                out = tvm.compute(shape, lambda *indice: flatten_indice(ten_a,
                                                                        indice)
                                  * flatten_indice(ten_a1, indice), name="out")
                intrinsic_cmd = "vmul"
            elif op_ins.lower() == "elewise_binary_min":
                out = tvm.compute(shape, lambda *indice: tvm.min(
                    flatten_indice(ten_a, indice),
                    flatten_indice(ten_a1, indice)),
                                  name="out")
                intrinsic_cmd = "vmin"
            elif op_ins.lower() == "elewise_binary_max":
                out = tvm.compute(shape, lambda *indice: tvm.max(
                    flatten_indice(ten_a, indice),
                    flatten_indice(ten_a1, indice)),
                                  name="out")
                intrinsic_cmd = "vmax"
            elif op_ins.lower() == "elewise_binary_or":
                out = tvm.compute(shape, lambda *indice: flatten_indice(ten_a,
                                                                        indice)
                                  | flatten_indice(ten_a1, indice), name="out")
                intrinsic_cmd = "vor"
            elif op_ins.lower().find("elewise_binary_compare") != -1:
                if args[1] == "lt":
                    out = tvm.compute(shape,
                                      lambda *indice: tvm.select(
                                          flatten_indice(ten_a, indice) < args[
                                              0].astype(src_dtype),
                                          flatten_indice(ten_a1, indice),
                                          tvm.const(0, dtype=dst_dtype)),
                                      name="out")
                    intrinsic_cmd = "vcmp_lt"
                elif args[1] == "gt":
                    out = tvm.compute(shape,
                                      lambda *indice: tvm.select(
                                          flatten_indice(ten_a, indice) > args[
                                              0].astype(src_dtype),
                                          flatten_indice(ten_a1, indice),
                                          tvm.const(0, dtype=dst_dtype)),
                                      name="out")
                    intrinsic_cmd = "vcmp_gt"
            elif op_ins.lower() == "elewise_binary_and":
                out = tvm.compute(shape,
                                  lambda *indice: flatten_indice(ten_a,
                                                                 indice)
                                  & flatten_indice(ten_a1,
                                                   indice),
                                  name="out")
                intrinsic_cmd = "vand"
            elif op_ins.lower() == "elewise_binary_scalar_axpy":
                out = tvm.compute(shape, lambda *indice: args[0].astype(
                    src_dtype) * flatten_indice(
                        ten_a, indice) + flatten_indice(ten_a1, indice), name="out")
                intrinsic_cmd = "vaxpy"
            elif op_ins.lower() == "elewise_binary_cmpsel":
                temp_op = args[0]
                if temp_op == 'lt':
                    lambda_func = lambda *indice: tvm.select(
                        tvm.any(flatten_indice(ten_a, indice) < flatten_indice(ten_a1,
                                                                               indice)),
                        flatten_indice(ten_a, indice), flatten_indice(ten_a1, indice))
                elif temp_op == 'gt':
                    lambda_func = lambda *indice: tvm.select(
                        tvm.any(flatten_indice(ten_a, indice) > flatten_indice(ten_a1,
                                                                               indice)),
                        flatten_indice(ten_a, indice), flatten_indice(ten_a1, indice))
                elif temp_op == 'le':
                    lambda_func = lambda *indice: tvm.select(
                        tvm.any(flatten_indice(ten_a, indice) <= flatten_indice(ten_a1,
                                                                                indice)),
                        flatten_indice(ten_a, indice), flatten_indice(ten_a1, indice))
                elif temp_op == 'ge':
                    lambda_func = lambda *indice: tvm.select(
                        tvm.any(flatten_indice(ten_a, indice) >= flatten_indice(ten_a1,
                                                                                indice)),
                        flatten_indice(ten_a, indice), flatten_indice(ten_a1, indice))
                elif temp_op == 'eq':
                    lambda_func = lambda *indice: tvm.select(
                        tvm.any(flatten_indice(ten_a, indice) == flatten_indice(ten_a1,
                                                                                indice)),
                        flatten_indice(ten_a, indice), flatten_indice(ten_a1, indice))
                elif temp_op == 'ne':
                    lambda_func = lambda *indice: tvm.select(
                        tvm.any(flatten_indice(ten_a, indice) != flatten_indice(ten_a1,
                                                                                indice)),
                        flatten_indice(ten_a, indice), flatten_indice(ten_a1, indice))
                out = tvm.compute(shape, lambda_func, name="out")
                intrinsic_cmd = "vcmpsel"
            elif op_ins.lower() == "elewise_binary_logic":
                temp_op = args[0]
                if temp_op == 'and':
                    lambda_func = lambda *indice: flatten_indice(
                        ten_a, indice) & flatten_indice(
                            ten_a1, indice)
                elif temp_op == 'or':
                    lambda_func = lambda *indice: flatten_indice(
                        ten_a, indice) | flatten_indice(
                            ten_a1, indice)
                elif temp_op == 'not':
                    lambda_func = lambda *indice: ~ flatten_indice(ten_a, indice)
                out = tvm.compute(shape, lambda_func, name="out")
                intrinsic_cmd = "vlogic"
            a0_buf = tvm.decl_buffer(ten_a.shape, ten_a.dtype,
                                     name="A0_buf",
                                     offset_factor=1,
                                     scope=buffer_scope,
                                     strides=[tvm.var() for _ in
                                              range(len(ten_a.shape))],
                                     data_alignment=get_data_alignment(
                                         ten_a.dtype))
            if not is_same:
                a1_buf = tvm.decl_buffer(ten_a1.shape, ten_a1.dtype,
                                         name="A1_buf",
                                         offset_factor=1,
                                         scope=buffer_scope,
                                         strides=[tvm.var() for _ in
                                                  range(len(ten_a1.shape))],
                                         data_alignment=get_data_alignment(
                                             ten_a1.dtype))
            else:
                a1_buf = a0_buf

            out_buf = tvm.decl_buffer(out.shape, out.dtype,
                                      name="OUT_buf",
                                      offset_factor=1,
                                      scope=buffer_scope,
                                      strides=[tvm.var() for _ in
                                               range(len(out.shape))],
                                      data_alignment=get_data_alignment(
                                          out.dtype))
            size = functools_reduce(lambda i, j: i * j, shape)

            def intrin_func(ins, outs):
                """
                intrin_func function
                """
                if is_same:
                    ins_a0 = ins[0]
                    ins_a1 = ins_a0
                else:
                    ins_a0, ins_a1 = ins
                out = outs[0]

                def instr():
                    """
                    instr function
                    """
                    ib_ins = tvm.ir_builder.create()
                    src_buffers = [ins_a0, ins_a1]
                    dst_buffers = [out]
                    if_reset_mask_pipe_line = True
                    pipeline_count_list = [0]
                    if op_ins.lower() == "elewise_binary_scalar_axpy":
                        vec_vssingle_elewise(ib_ins, intrinsic_cmd, [ins_a0], [ins_a1],
                                             size, pipe_line_index,
                                             pipeline_count_list,
                                             if_reset_mask_pipe_line, args)
                        # for storage rewrite reuse, out and A1 is the same buffer
                        ib_ins.emit(tvm.call_extern(out.dtype, "rewrite_inplace",
                                                    out.access_ptr("w"),
                                                    ins_a1.access_ptr("r")))
                    elif op_ins.lower().find("elewise_binary_compare") != -1:
                        vec_binary_elewise_compare(ib_ins,
                                                   intrinsic_cmd,
                                                   src_buffers,
                                                   dst_buffers,
                                                   size,
                                                   pipe_line_index,
                                                   pipeline_count_list,
                                                   if_reset_mask_pipe_line,
                                                   args)
                    else:
                        extern_args = args if intrinsic_cmd in ["vcmpsel", "vlogic"] else []
                        vec_binary_elewise(ib_ins, intrinsic_cmd, src_buffers,
                                           dst_buffers, size,
                                           pipe_line_index,
                                           pipeline_count_list,
                                           if_reset_mask_pipe_line,
                                           extern_args=extern_args)
                    return ib_ins.get()

                # return a triple of normal-set, reset, update
                return (instr(), None, None)

            return tvm.decl_tensor_intrin(out.op, intrin_func,
                                          name=intrinsic_cmd,
                                          binds={ten_a: a0_buf,
                                                 ten_a1: a1_buf,
                                                 out: out_buf})

        # pylint: disable=too-many-branches
        def elewise_single_intrin_cce(shape, op_ins, dst_dtype, src_dtype, args):
            """factory funtion for elewise operations of single op. For tensorize

            Parameters
            ----------
            shapes : tuple or list
                The arrays to concatenate

            op_ins : string
                operation type, supports elewise_single_log, elewise_single_exp,
                elewise_single_rec, elewise_single_abs,
                elewise_single_vs_add,elewise_single_vs_mul,
                elewise_single_round, elewise_single_floor,
                elewise_single_ceil, elewise_single_cast

            dst_dtype :
                The destination data type

            src_dtype :
                The source data type

            args : tvm.const
                scalar to broadcast.

            Returns
            -------
            ret : TensorIntrin
                A TensorIntrin for elewise operations of single op
                that can be used in tensorize schedule.
            """
            # pylint: disable=too-many-statements
            pipe_line_index = __cal_map_sum(vec_count)
            vec_count["elewise_single_intrin_cce"] += 4
            ten_a0 = tvm.placeholder(shape, src_dtype, name='A0')
            repeat_cal_dtype = dst_dtype
            res_args = None
            dst_stride_m1 = 8
            src_stride_m1 = 8
            if op_ins.lower() == "elewise_single_log":
                out = tvm.compute(shape, lambda *indice: tvm.log(
                    flatten_indice(ten_a0, indice)),
                                  name="out")
                intrinsic_cmd = "vln"
            elif op_ins.lower() == "elewise_single_exp":
                out = tvm.compute(shape, lambda *indice: tvm.exp(
                    flatten_indice(ten_a0, indice)),
                                  name="out")
                intrinsic_cmd = "vexp"
            elif op_ins.lower() == "elewise_single_rec":
                out = tvm.compute(shape, lambda *indice: 1 // flatten_indice(ten_a0,
                                                                             indice),
                                  name="out")
                intrinsic_cmd = "vrec"
            elif op_ins.lower() == "elewise_single_abs":
                out = tvm.compute(shape, lambda *indice: tvm.select(
                    flatten_indice(ten_a0, indice) >= 0,
                    flatten_indice(ten_a0, indice),
                    - flatten_indice(ten_a0, indice)),
                                  name="out")
                intrinsic_cmd = "vabs"
            elif op_ins.lower() == "elewise_single_vs_add":
                res_args = args[0]
                out = tvm.compute(shape, lambda *indice: flatten_indice(ten_a0,
                                                                        indice)
                                  + res_args,
                                  name="out")
                intrinsic_cmd = "vadds"
            elif op_ins.lower() == "elewise_single_vs_mul":
                res_args = args[0]
                out = tvm.compute(shape, lambda *indice: flatten_indice(ten_a0,
                                                                        indice)
                                  * res_args,
                                  name="out")
                intrinsic_cmd = "vmuls"
            elif op_ins.lower() == "elewise_single_relu":
                out = tvm.compute(shape, lambda *indice: tvm.select(
                    flatten_indice(ten_a0, indice) >= 0,
                    flatten_indice(ten_a0, indice),
                    tvm.const(0, dtype=dst_dtype)),
                                  name="out")
                intrinsic_cmd = "vrelu"
            elif op_ins.lower() == "elewise_single_not":
                out = tvm.compute(shape,
                                  lambda *indice: ~flatten_indice(ten_a0, indice),
                                  name="out")
                intrinsic_cmd = "vnot"
            elif op_ins.lower() == "elewise_single_sqrt":
                out = tvm.compute(shape, lambda *indice: tvm.sqrt(
                    flatten_indice(ten_a0, indice)),
                                  name="out")
                intrinsic_cmd = "vsqrt"
            elif op_ins.lower() == "elewise_single_round":
                out = tvm.compute(shape,
                                  lambda *indice: tvm.round(
                                      flatten_indice(ten_a0, indice)).astype(
                                          "int32"), name="out")
                intrinsic_cmd = "vconv_f162s32r"
                src_stride_m1 = 4
                if src_dtype == "float32":
                    intrinsic_cmd = "vconv_f322s32r"
                    src_stride_m1 = 8
                repeat_cal_dtype = "int32"
            elif op_ins.lower() == "elewise_single_floor":
                out = tvm.compute(shape,
                                  lambda *indice: tvm.floor(
                                      flatten_indice(ten_a0, indice)).astype(
                                          "int32"), name="out")
                intrinsic_cmd = "vconv_f162s32f"
                src_stride_m1 = 4
                if src_dtype == "float32":
                    intrinsic_cmd = "vconv_f322s32f"
                    src_stride_m1 = 8
                repeat_cal_dtype = "int32"
            elif op_ins.lower() == "elewise_single_ceil":
                out = tvm.compute(shape,
                                  lambda *indice: tvm.ceil(
                                      flatten_indice(ten_a0, indice)).astype(
                                          "int32"), name="out")
                intrinsic_cmd = "vconv_f162s32c"
                src_stride_m1 = 4
                if src_dtype == "float32":
                    intrinsic_cmd = "vconv_f322s32c"
                    src_stride_m1 = 8
                repeat_cal_dtype = "int32"
            elif op_ins.lower() == "elewise_single_trunc":
                out = tvm.compute(shape,
                                  lambda *indice: tvm.trunc(
                                      flatten_indice(ten_a0, indice)).astype(
                                          "int32"), name="out")
                intrinsic_cmd = "vconv_f162s32z"
                src_stride_m1 = 4
                if src_dtype == "float32":
                    intrinsic_cmd = "vconv_f322s32z"
                    src_stride_m1 = 8
                repeat_cal_dtype = "int32"
            elif op_ins.lower() == "elewise_single_cast":
                out = tvm.compute(shape,
                                  lambda *indice: flatten_indice(
                                      ten_a0, indice).astype(dst_dtype),
                                  name="out")
                src_conv_type = cce_util.dtype2ccetype(src_dtype)
                dst_conv_type = cce_util.dtype2ccetype(dst_dtype)
                src_bit_len = get_bit_len(src_dtype)
                dst_bit_len = get_bit_len(dst_dtype)
                if src_bit_len > dst_bit_len:
                    repeat_cal_dtype = src_conv_type
                    dst_stride_m1 = 4
                elif src_bit_len < dst_bit_len:
                    repeat_cal_dtype = dst_conv_type
                    src_stride_m1 = 4
                intrinsic_cmd = "vconv_" + src_conv_type + "2" + dst_conv_type
                if src_conv_type == 's32' and dst_conv_type == 'f16':
                    intrinsic_cmd = "vconv_deq"
            elif op_ins.lower() == "elewise_single_vs_cond":
                res_args = args
                temp_op = args[0]
                temp_thredhold = tvm.const(float(args[1]), dst_dtype)
                temp_bias = tvm.const(float(args[2]), dst_dtype)
                if temp_op == 'lt':
                    out = tvm.compute(shape, lambda *indice: tvm.select(
                        tvm.any(flatten_indice(ten_a0, indice) < temp_thredhold),
                        flatten_indice(ten_a0, indice), temp_bias), name='out')
                elif temp_op == 'gt':
                    out = tvm.compute(shape, lambda *indice: tvm.select(
                        tvm.any(flatten_indice(ten_a0, indice) > temp_thredhold),
                        flatten_indice(ten_a0, indice), temp_bias), name='out')
                elif temp_op == 'le':
                    out = tvm.compute(shape, lambda *indice: tvm.select(
                        tvm.any(flatten_indice(ten_a0, indice) <= temp_thredhold),
                        flatten_indice(ten_a0, indice), temp_bias), name='out')
                elif temp_op == 'ge':
                    out = tvm.compute(shape, lambda *indice: tvm.select(
                        tvm.any(flatten_indice(ten_a0, indice) >= temp_thredhold),
                        flatten_indice(ten_a0, indice), temp_bias), name='out')
                elif temp_op == 'eq':
                    out = tvm.compute(shape, lambda *indice: tvm.select(
                        tvm.any(flatten_indice(ten_a0, indice) == temp_thredhold),
                        flatten_indice(ten_a0, indice), temp_bias), name='out')
                elif temp_op == 'ne':
                    out = tvm.compute(shape, lambda *indice: tvm.select(
                        tvm.any(flatten_indice(ten_a0, indice) != temp_thredhold),
                        flatten_indice(ten_a0, indice), temp_bias), name='out')
                else:
                    raise RuntimeError(
                        "vcond do not support the input op" % temp_op)
                intrinsic_cmd = "vcond"

            a0_buf = tvm.decl_buffer(ten_a0.shape, ten_a0.dtype,
                                     name="A0_buf",
                                     offset_factor=1,
                                     data_alignment=get_data_alignment(
                                         ten_a0.dtype),
                                     scope=buffer_scope,
                                     strides=[tvm.var() for _ in
                                              range(len(ten_a0.shape))])

            out_buf = tvm.decl_buffer(out.shape, out.dtype,
                                      name="OUT_buf",
                                      offset_factor=2,
                                      data_alignment=get_data_alignment(
                                          out.dtype),
                                      scope=buffer_scope,
                                      strides=[tvm.var() for _ in
                                               range(len(out.shape))])
            size = functools_reduce(lambda i, j: i * j, shape)

            def intrin_func(ins, outs):
                """
                intrin_func function
                """
                ins_a0 = ins[0]
                out = outs[0]

                def instr():
                    """
                    instr function
                    """
                    ib_ins = tvm.ir_builder.create()
                    if intrinsic_cmd == "vconv_deq":
                        with ib_ins.new_scope():
                            ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              2 + total_pipe_line * pipe_line_index)
                            ib_ins.emit(tvm.call_extern("float16", "set_deqscale",
                                                        tvm.const(1,
                                                                  dtype="float16")))
                    if_reset_mask_pipe_line = True
                    pipeline_count_list = [0]
                    src_buffers = [ins_a0]
                    dst_buffers = [out]
                    if res_args is not None:
                        extend_args = res_args if isinstance(res_args,
                                                             list) \
                            else [res_args]
                        vec_vssingle_elewise(ib_ins, intrinsic_cmd, src_buffers,
                                             dst_buffers, size,
                                             pipe_line_index,
                                             pipeline_count_list,
                                             if_reset_mask_pipe_line,
                                             extend_args)
                    else:
                        if intrinsic_cmd.find("vconv") != -1:
                            args = [1, 1, dst_stride_m1, src_stride_m1]
                            vec_cast_elewise(ib_ins, intrinsic_cmd, src_buffers,
                                             dst_buffers, size,
                                             pipe_line_index,
                                             pipeline_count_list,
                                             if_reset_mask_pipe_line, args,
                                             repeat_cal_dtype)
                        else:
                            vec_single_elewise(ib_ins, intrinsic_cmd, src_buffers,
                                               dst_buffers, size,
                                               pipe_line_index,
                                               pipeline_count_list,
                                               if_reset_mask_pipe_line)
                    return ib_ins.get()

                # return a triple of normal-set, reset, update
                return (instr(), None, None)

            return tvm.decl_tensor_intrin(out.op, intrin_func,
                                          name=intrinsic_cmd,
                                          binds={ten_a0: a0_buf,
                                                 out: out_buf})

        def elewise_multiple_intrin_cce(shape, op_ins, dst_dtype, src_dtype, args):
            """factory funtion for elewise operations of multiple op. For tensorize

            Parameters
            ----------
            shapes : tuple or list
                The arrays to concatenate

            op_ins : string
                operation type, supports elewise_multiple_mla,
                elewise_multiple_madd, elewise_multiple_maddrelu

            dst_dtype :
                The destination data type

            src_dtype :
                The source data type

            args : tvm.const
                scalar to broadcast.

            Returns
            -------
            ret : TensorIntrin
                A TensorIntrin for elewise operations of multiple op
                that can be used in tensorize schedule.
            """
            # for pylint, reserve argument
            dst_dtype = dst_dtype
            args = args

            pipe_line_index = __cal_map_sum(vec_count)
            vec_count["elewise_multiple_intrin_cce"] += 4

            # elewise_multiple op have three inputs, which may be all the same,
            # partly the same or all different
            ten_a0 = tvm.placeholder(shape, src_dtype, name='A0')
            a0_buf = tvm.decl_buffer(ten_a0.shape, ten_a0.dtype,
                                     name="A0_buf",
                                     offset_factor=1,
                                     scope=buffer_scope,
                                     strides=[tvm.var() for _ in
                                              range(len(ten_a0.shape))],
                                     data_alignment=get_data_alignment(
                                         ten_a0.dtype))
            # input X,Y,Z are same
            if args[0] == 1 and args[1] == 1 and args[2] == 1:
                ins_a1 = ten_a0
                ins_a1 = ten_a0
                a1_buf = a0_buf
                a2_buf = a0_buf
            elif args[0] == 1 and args[1] == 1:  # input X,Y are same
                ins_a1 = ten_a0
                ins_a1 = tvm.placeholder(shape, src_dtype, name='A2')
                a1_buf = a0_buf
                a2_buf = tvm.decl_buffer(ins_a1.shape, ins_a1.dtype,
                                         name="A2_buf",
                                         offset_factor=1,
                                         scope=buffer_scope,
                                         strides=[tvm.var() for _ in
                                                  range(len(ins_a1.shape))],
                                         data_alignment=get_data_alignment(
                                             ins_a1.dtype))
            elif args[0] == 1 and args[2] == 1:  # input X,Z are same
                ins_a1 = tvm.placeholder(shape, src_dtype, name='A1')
                ins_a1 = ten_a0
                a1_buf = tvm.decl_buffer(ins_a1.shape, ins_a1.dtype,
                                         name="A1_buf",
                                         offset_factor=1,
                                         scope=buffer_scope,
                                         strides=[tvm.var() for _ in
                                                  range(len(ins_a1.shape))],
                                         data_alignment=get_data_alignment(
                                             ins_a1.dtype))
                a2_buf = a0_buf
            elif args[1] == 1 and args[2] == 1:  # input Y,Z are same
                ins_a1 = tvm.placeholder(shape, src_dtype, name='A1')
                ins_a1 = ins_a1
                a1_buf = tvm.decl_buffer(ins_a1.shape, ins_a1.dtype,
                                         name="A1_buf",
                                         offset_factor=1,
                                         scope=buffer_scope,
                                         strides=[tvm.var() for _ in
                                                  range(len(ins_a1.shape))],
                                         data_alignment=get_data_alignment(
                                             ins_a1.dtype))
                a2_buf = a1_buf
            else:  # input X,Y,Z are different
                ins_a1 = tvm.placeholder(shape, src_dtype, name='A1')
                ins_a1 = tvm.placeholder(shape, src_dtype, name='A2')
                a1_buf = tvm.decl_buffer(ins_a1.shape, ins_a1.dtype,
                                         name="A1_buf",
                                         offset_factor=1,
                                         scope=buffer_scope,
                                         strides=[tvm.var() for _ in
                                                  range(len(ins_a1.shape))],
                                         data_alignment=get_data_alignment(
                                             ins_a1.dtype))
                a2_buf = tvm.decl_buffer(ins_a1.shape, ins_a1.dtype,
                                         name="A2_buf",
                                         offset_factor=1,
                                         scope=buffer_scope,
                                         strides=[tvm.var() for _ in
                                                  range(len(ins_a1.shape))],
                                         data_alignment=get_data_alignment(
                                             ins_a1.dtype))

            if op_ins.lower() == "elewise_multiple_mla":
                out = tvm.compute(shape, lambda *indice: flatten_indice(ten_a0,
                                                                        indice)
                                  * flatten_indice(ins_a1, indice)
                                  + flatten_indice(ins_a1, indice), name="out")
                intrinsic_cmd = "vmla"
            elif op_ins.lower() == "elewise_multiple_madd":
                out = tvm.compute(shape, lambda *indice: flatten_indice(ten_a0,
                                                                        indice)
                                  * flatten_indice(ins_a1, indice)
                                  + flatten_indice(ins_a1, indice), name="out")
                intrinsic_cmd = "vmadd"
            elif op_ins.lower() == "elewise_multiple_maddrelu":
                out = tvm.compute(shape, lambda *indice: tvm.select(
                    (flatten_indice(ten_a0, indice) *
                     flatten_indice(ins_a1, indice) \
                     + flatten_indice(ins_a1, indice))
                    >= 0,
                    flatten_indice(ten_a0, indice) \
                    * flatten_indice(ins_a1, indice) \
                    + flatten_indice(ins_a1, indice),
                    tvm.const(0, dtype=src_dtype)),
                                  name="out")
                intrinsic_cmd = "vmaddrelu"

            out_buf = tvm.decl_buffer(out.shape, out.dtype,
                                      name="OUT_buf",
                                      offset_factor=1,
                                      scope=buffer_scope,
                                      strides=[tvm.var() for _ in
                                               range(len(out.shape))],
                                      data_alignment=get_data_alignment(
                                          out.dtype))

            size = functools_reduce(lambda i, j: i * j, shape)

            # pylint: disable=inconsistent-return-statements
            def intrin_func(ins, outs):
                """
                intrin_func function
                """
                if args[0] == 1 and args[1] == 1 and args[2] == 1:
                    ins_a0 = ins[0]
                    ins_a1 = ins_a0
                    ins_a2 = ins_a0
                elif args[0] == 1 and args[1] == 1:  # input X,Y are same
                    ins_a0 = ins[0]
                    ins_a1 = ins_a0
                    ins_a2 = ins[1]
                elif args[0] == 1 and args[2] == 1:  # input X,Z are same
                    ins_a0 = ins[0]
                    ins_a1 = ins[1]
                    ins_a2 = ins_a0
                elif args[1] == 1 and args[2] == 1:  # input Y,Z are same
                    ins_a0 = ins[0]
                    ins_a1 = ins[1]
                    ins_a2 = ins_a1
                else:  # input X,Y,Z are different
                    ins_a0, ins_a1, ins_a2 = ins

                out = outs[0]

                def instr():
                    """
                    instr function
                    """
                    ib_ins = tvm.ir_builder.create()
                    # due to the vmla and vmadd is [Xd] = [Xn]*[Xd] + [Xm],
                    # so the src only include the A0, A1, and the dst is A2.
                    src_buffers = [ins_a0, ins_a1]
                    dst_buffers = [ins_a2]
                    if_reset_mask_pipe_line = True
                    pipeline_count_list = [0]
                    vec_binary_elewise(ib_ins, intrinsic_cmd, src_buffers,
                                       dst_buffers, size,
                                       pipe_line_index,
                                       pipeline_count_list,
                                       if_reset_mask_pipe_line)

                    # for storage rewrite reuse,A2 and out is the same
                    # buffer
                    ib_ins.emit(tvm.call_extern(out.dtype, "rewrite_inplace",
                                                out.access_ptr("w"),
                                                ins_a2.access_ptr("r")))
                    return ib_ins.get()

                # return a triple of normal-set, reset, update
                return (instr(), None, None)

            return tvm.decl_tensor_intrin(out.op, intrin_func,
                                          name=intrinsic_cmd,
                                          binds={ten_a0: a0_buf,
                                                 ins_a1: a1_buf,
                                                 ins_a1: a2_buf,
                                                 out: out_buf})

        def concat(shapes, dtype, axis=0):
            """Join a sequence of arrays along an existing axis.

            Parameters
            ----------
            shapes : tuple or list
                The arrays to concatenate

            dtype :
                The data type

            axis : int, optional
                The axis along which the arrays will be joined. Default is 0.

            Returns
            -------
            ret : tvm.Tensor
            """
            # for pylint, reserve argument
            axis = axis

            pipe_line_index = __cal_map_sum(vec_count)
            vec_count["concat"] += 1

            align_factor = 0
            if dtype in ["int8", "uint8"]:
                align_factor = 32
            elif dtype == "float16":
                align_factor = 16
            else:
                align_factor = 8

            # use for the corss intrin
            if len(shapes) == 3 and len(shapes[0]) == 1 and len(
                    shapes[1]) == 1 and len(shapes[2]) == 1:
                tensors = [
                    tvm.placeholder(shapes[i], dtype, name="input" + str(i)) for
                    i in
                    range(len(shapes))]
                out = tvm.compute((len(shapes), shapes[0][0]),
                                  lambda i, j: tvm.select(i == 0, tensors[0][j],
                                                          tvm.select(i == 1,
                                                                     tensors[1][
                                                                         j],
                                                                     tensors[2][
                                                                         j])),
                                  name='out')

                bufs = [tvm.decl_buffer(tensors[i].shape, tensors[i].dtype,
                                        name=str(tensors[i].name) + "_buf",
                                        offset_factor=align_factor,
                                        data_alignment=get_data_alignment(
                                            tensors[i].dtype),
                                        scope=buffer_scope,
                                        strides=[tvm.var() for i in
                                                 range(len(tensors[i].shape))])
                        for i in range(len(tensors))]

                out_buf = tvm.decl_buffer(out.shape, out.dtype,
                                          name="out_buf",
                                          offset_factor=align_factor,
                                          data_alignment=get_data_alignment(
                                              out.dtype),
                                          scope=buffer_scope,
                                          strides=[tvm.var() for i in
                                                   range(len(out.shape))])

                len_burst = (shapes[0][0] + (align_factor - 1)) // align_factor

                def intrin_func(ins, outs):
                    """
                    intrin_func function
                    """
                    ins_a, ins_b, ins_c = ins
                    out = outs[0]

                    def instr(flag):
                        """
                        instr function
                        """
                        ib_ins = tvm.ir_builder.create()
                        ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope",
                                          2 + total_pipe_line * pipe_line_index)
                        if flag == 'body':
                            ib_ins.emit(tvm.call_extern(
                                out.dtype, "copy_ubuf_to_ubuf",
                                out.access_ptr("rw"),
                                ins_a.access_ptr("r"),
                                0,
                                1,
                                len_burst,
                                0,
                                0))
                            ib_ins.emit(tvm.call_extern(
                                out.dtype, "copy_ubuf_to_ubuf",
                                out.access_ptr("rw", offset=(shapes[0][0] + (
                                    align_factor - 1)) // align_factor * align_factor),
                                ins_b.access_ptr("r"),
                                0,
                                1,
                                len_burst,
                                0,
                                0))
                            ib_ins.emit(tvm.call_extern(
                                out.dtype, "copy_ubuf_to_ubuf",
                                out.access_ptr("rw", offset=((shapes[0][0] + (
                                    align_factor - 1)) // align_factor * align_factor) * 2),
                                ins_c.access_ptr("r"),
                                0,
                                1,
                                len_burst,
                                0,
                                0))
                        return ib_ins.get()

                    return instr('body'), None, None

                binds = {}
                tensors_len = len(tensors)
                for i in range(tensors_len):
                    binds[tensors[i]] = bufs[i]
                binds[out] = out_buf
                return tvm.decl_tensor_intrin(out.op, intrin_func, binds=binds)
            return None

        def mov_backup(shape, dtype):
            """This mov is helpful to procss non-aligned data for cross.

            Parameters
            ----------
            shape : tuple or list
                The shape of tensor

            dtype :
                The data type

            Returns
            -------
            ret : tvm.Tensor
            """
            num = functools_reduce(lambda i, j: i * j, shape)

            align_factor = 0
            if dtype in ["int8", "uint8"]:
                align_factor = 32
            elif dtype == "float16":
                align_factor = 16
            else:
                align_factor = 8

            ten_a = tvm.placeholder(shape, dtype, name='a')
            ten_b = tvm.compute(shape, lambda *i: ten_a[i], name='b')

            a_buf = tvm.decl_buffer(ten_a.shape, ten_a.dtype,
                                    name="a_buf",
                                    offset_factor=1,
                                    data_alignment=get_data_alignment(ten_a.dtype),
                                    scope=buffer_scope,
                                    strides=[tvm.var() for _ in
                                             range(len(ten_a.shape))])

            b_buf = tvm.decl_buffer(ten_b.shape, ten_b.dtype,
                                    name="b_buf",
                                    offset_factor=1,
                                    data_alignment=get_data_alignment(ten_b.dtype),
                                    strides=[tvm.var() for _ in
                                             range(len(ten_b.shape))])

            len_burst = (num + (align_factor - 1)) // align_factor

            def intrin_func(ins, outs):
                """
                intrin_func function
                """
                ins = ins[0]
                out = outs[0]

                def instr(flag):
                    """
                    instr function
                    """
                    ib_ins = tvm.ir_builder.create()
                    if flag == 'body':
                        tmp_buffer = __apply_for_new_alloc(ib_ins, ins.dtype,
                                                           (align_factor,))
                        with ib_ins.new_scope():
                            sid = cce_util.get_dma_sid("Sid_copy_gm_to_ubuf")
                            ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                            ib_ins.emit(tvm.call_extern(
                                out.dtype, "copy_gm_to_ubuf",
                                tmp_buffer.access_ptr("w"),
                                out.access_ptr("rw", offset=num),
                                sid,
                                1,
                                1,
                                0,
                                0))
                        with ib_ins.new_scope():
                            ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                            ib_ins.emit(tvm.call_extern(
                                out.dtype, "copy_ubuf_to_gm",
                                out.access_ptr("rw"),
                                ins.access_ptr("r"),
                                0,
                                1,
                                len_burst,
                                0,
                                0))
                        with ib_ins.new_scope():
                            ib_ins.scope_attr(param.CCE_AXIS, "coproc_scope", 14)
                            ib_ins.emit(tvm.call_extern(
                                out.dtype, "copy_ubuf_to_gm",
                                out.access_ptr("rw", offset=num),
                                tmp_buffer.access_ptr("r"),
                                0,
                                1,
                                1,
                                0,
                                0))
                    return ib_ins.get()

                return instr('body'), None, None

            return tvm.decl_tensor_intrin(ten_b.op, intrin_func,
                                          binds={ten_a: a_buf, ten_b: b_buf})

        intrin_map = {"reduce_nist_axis": reduce_nist_axis,
                      "elewise_single_intrin_cce": elewise_single_intrin_cce,
                      "elewise_binary_intrin_cce": elewise_binary_intrin_cce,
                      "elewise_multiple_intrin_cce": elewise_multiple_intrin_cce,
                      "reduce_last_axis": reduce_last_axis,
                      "concat": concat,
                      "mov_backup": mov_backup,

                      # for UT test
                      "vec_single_elewise": vec_single_elewise,
                      "vec_cast_elewise": vec_cast_elewise,
                      "vec_VSsingle_elewise": vec_vssingle_elewise,
                      "vec_binary_elewise": vec_binary_elewise
                     }
        if not intrin_map.get(intrin_key):
            raise RuntimeError("Not implement yet.")
        return intrin_map[intrin_key]

    intrin_list = ["reduce_nist_axis", "elewise_single_intrin_cce",
                   "elewise_binary_intrin_cce",
                   "elewise_multiple_intrin_cce", "reduce_last_axis", "concat",
                   "mov_backup"]

    for i in intrin_list:
        vec_count[i] = 0

    return vec_intrin
