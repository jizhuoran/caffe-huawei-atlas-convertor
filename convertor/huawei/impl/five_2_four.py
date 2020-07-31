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

five_2_four
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce as functools_reduce

from te import platform as cce
from te import tvm
from te.platform.cce_build import build_config
import te.platform.cce_params as cce_params
from topi.cce import util
from impl import slice_d

# available ub size
UB_SIZE_B = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE)
# available number of cores
AICORE_NUM = cce.cce_conf.get_soc_spec(cce.cce_conf.CORE_NUM)


# pylint: disable=locally-disabled,too-many-lines
def _new_alloc(tvm_ib, dtype, shape, name, scope):
    """
    decl new buffer for ir builder make function

    """
    buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name,
                                 scope=scope, data=buf_var)

    return new_buffer


def _ceil_div(value, block):
    """
    integrate the input value by block

    """
    return (value + block - 1) // block


def _ceil_fill(value, block):
    """
    fill the input value by block

    """
    return _ceil_div(value, block)*block


def _func_pipe(tvm_ib):
    pipe_mte2 = tvm.call_pure_intrin("int32", "tvm_cce_string_print",
                                     "PIPE_MTE2")
    pipe_mte3 = tvm.call_pure_intrin("int32", "tvm_cce_string_print",
                                     "PIPE_MTE3")
    event = tvm.call_pure_intrin("int32", "tvm_cce_string_print",
                                 "EVENT_ID3")
    tvm_ib.emit(
        tvm.call_extern("int32", "set_flag", pipe_mte3, pipe_mte2,
                        event))
    tvm_ib.emit(
        tvm.call_extern("int32", "wait_flag", pipe_mte3, pipe_mte2,
                        event))


# pylint: disable=locally-disabled,too-many-locals,too-many-nested-blocks
def _get_param_more_dim(tvm_ib, src_shape, dtype, max_dim, shape_all):
    """
    calculate parameters for more dim ir builder make function

    """
    device_core_num = AICORE_NUM
    ub_bytes = ((UB_SIZE_B - 32) // 2)
    if shape_all > 380000000:
        device_core_num = 1
        ub_bytes = ((UB_SIZE_B // 2 - 32) // 2)
    elif device_core_num == 32 and (max_dim > 800000 or shape_all > 320000000):
        device_core_num = 16
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_ele = ub_bytes // float_size
    dim_ele = functools_reduce(lambda x, y: x*y, src_shape[2:])
    num_dim_in_data = functools_reduce(lambda x, y: x*y, src_shape[0:2])
    num_dim_one_core = ub_ele // dim_ele
    num_dim_one_group = num_dim_one_core*device_core_num
    num_group_index = num_dim_in_data // num_dim_one_group
    num_group_mod = num_dim_in_data % num_dim_one_group

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    param_map = {"num_group_index": num_group_index,
                 "num_group_mod": num_group_mod,
                 "dim_ele": dim_ele,
                 "float_size": float_size,
                 "cp_align_len": cp_align_len,
                 "num_dim_one_core": num_dim_one_core,
                 "num_dim_one_group": num_dim_one_group,
                 "block_index": block_index}

    return param_map


def _ub_to_res(args):
    """
    move from data_ub to data_res to change the C axis for more dim scene

    """
    tvm_ib, data_ub, data_res, reg, num_col_in_dim, ub_offset, \
    ub_res_offset, num_row, c_0 = args

    with tvm_ib.for_range(0, num_col_in_dim, name="num_c") as num_c:
        ele_reg = 8
        r_cycle = num_row // ele_reg
        r_mod = num_row - ele_reg*r_cycle
        reg_zero = 0
        reg_one = 1
        reg_two = 2
        reg_three = 3
        reg_four = 4
        reg_five = 5
        reg_six = 6
        reg_seven = 7

        with tvm_ib.for_range(0, r_cycle, name="num_cr") as num_cr:
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[reg_zero]),
                data_ub.access_ptr('r',
                                   offset=(ub_offset + (
                                       num_cr*ele_reg + reg_zero)*c_0 + num_c))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[reg_one]),
                data_ub.access_ptr('r',
                                   offset=(ub_offset + (
                                       num_cr*ele_reg + reg_one)*c_0 + num_c))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[reg_two]),
                data_ub.access_ptr('r',
                                   offset=(ub_offset + (
                                       num_cr*ele_reg + reg_two)*c_0 + num_c))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[reg_three]),
                data_ub.access_ptr('r',
                                   offset=(ub_offset + (
                                       num_cr*ele_reg + reg_three)*c_0 + num_c))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[reg_four]),
                data_ub.access_ptr('r',
                                   offset=(ub_offset + (
                                       num_cr*ele_reg + reg_four)*c_0 + num_c))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[reg_five]),
                data_ub.access_ptr('r',
                                   offset=(ub_offset + (
                                       num_cr*ele_reg + reg_five)*c_0 + num_c))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[reg_six]),
                data_ub.access_ptr('r',
                                   offset=(ub_offset + (
                                       num_cr*ele_reg + reg_six)*c_0 + num_c))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[reg_seven]),
                data_ub.access_ptr('r',
                                   offset=(ub_offset + (
                                       num_cr*ele_reg + reg_seven)*c_0 + num_c))
            ))

            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=(
                    ub_res_offset + num_c*num_row
                    + (num_cr*ele_reg + reg_zero))),
                tvm.call_extern(reg.dtype, "reg", reg[reg_zero])
            ))
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=(
                    ub_res_offset + num_c*num_row
                    + (num_cr*ele_reg + reg_one))),
                tvm.call_extern(reg.dtype, "reg", reg[reg_one])
            ))
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=(
                    ub_res_offset + num_c*num_row
                    + (num_cr*ele_reg + reg_two))),
                tvm.call_extern(reg.dtype, "reg", reg[reg_two])
            ))
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=(
                    ub_res_offset + num_c*num_row
                    + (num_cr*ele_reg + reg_three))),
                tvm.call_extern(reg.dtype, "reg", reg[reg_three])
            ))
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=(
                    ub_res_offset + num_c*num_row
                    + (num_cr*ele_reg + reg_four))),
                tvm.call_extern(reg.dtype, "reg", reg[reg_four])
            ))
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=(
                    ub_res_offset + num_c*num_row
                    + (num_cr*ele_reg + reg_five))),
                tvm.call_extern(reg.dtype, "reg", reg[reg_five])
            ))
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=(
                    ub_res_offset + num_c*num_row
                    + (num_cr*ele_reg + reg_six))),
                tvm.call_extern(reg.dtype, "reg", reg[reg_six])
            ))
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=(
                    ub_res_offset + num_c*num_row
                    + (num_cr*ele_reg + reg_seven))),
                tvm.call_extern(reg.dtype, "reg", reg[reg_seven])
            ))

        with tvm_ib.if_scope(r_mod > 0):
            with tvm_ib.for_range(0, r_mod, name="num_er") as num_er:
                tvm_ib.emit(tvm.call_extern(
                    data_ub.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_ub.access_ptr('r',
                                       offset=(ub_offset +
                                               (r_cycle*ele_reg + num_er)*c_0
                                               + num_c))
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_res.dtype, "reg_mov",
                    data_res.access_ptr('w', offset=(
                        ub_res_offset + num_c*num_row +
                        (r_cycle*ele_reg + num_er))),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))


def _reg_mov_align_batch(args):
    """
    move the tail of data_res for not 32B align scene in more dim function

    """
    tvm_ib, param, data_res, data_tail, reg, tail_align_dst = args

    ele_reg = 8
    num_row = param.get("cp_align_len")
    r_cycle = num_row // ele_reg
    r_mod = num_row - ele_reg*r_cycle
    reg_zero = 0
    reg_one = 1
    reg_two = 2
    reg_three = 3
    reg_four = 4
    reg_five = 5
    reg_six = 6
    reg_seven = 7

    with tvm_ib.for_range(0, r_cycle, name="num_cr") as num_cr:
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero]),
            data_res.access_ptr('r',
                                offset=(tail_align_dst + num_cr*ele_reg
                                        + reg_zero))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_one]),
            data_res.access_ptr('r',
                                offset=(tail_align_dst + num_cr*ele_reg
                                        + reg_one))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_two]),
            data_res.access_ptr('r',
                                offset=(tail_align_dst + num_cr*ele_reg
                                        + reg_two))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_three]),
            data_res.access_ptr('r',
                                offset=(tail_align_dst + num_cr*ele_reg
                                        + reg_three))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_four]),
            data_res.access_ptr('r',
                                offset=(tail_align_dst + num_cr*ele_reg
                                        + reg_four))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_five]),
            data_res.access_ptr('r',
                                offset=(tail_align_dst + num_cr*ele_reg
                                        + reg_five))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_six]),
            data_res.access_ptr('r',
                                offset=(tail_align_dst + num_cr*ele_reg
                                        + reg_six))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven]),
            data_res.access_ptr('r',
                                offset=(tail_align_dst + num_cr*ele_reg
                                        + reg_seven))
        ))

        tvm_ib.emit(tvm.call_extern(
            data_tail.dtype, "reg_mov",
            data_tail.access_ptr('w', offset=(num_cr*ele_reg + reg_zero)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_tail.dtype, "reg_mov",
            data_tail.access_ptr('w', offset=(num_cr*ele_reg + reg_one)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_one])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_tail.dtype, "reg_mov",
            data_tail.access_ptr('w', offset=(num_cr*ele_reg + reg_two)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_two])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_tail.dtype, "reg_mov",
            data_tail.access_ptr('w', offset=(num_cr*ele_reg + reg_three)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_three])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_tail.dtype, "reg_mov",
            data_tail.access_ptr('w', offset=(num_cr*ele_reg + reg_four)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_four])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_tail.dtype, "reg_mov",
            data_tail.access_ptr('w', offset=(num_cr*ele_reg + reg_five)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_five])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_tail.dtype, "reg_mov",
            data_tail.access_ptr('w', offset=(num_cr*ele_reg + reg_six)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_six])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_tail.dtype, "reg_mov",
            data_tail.access_ptr('w', offset=(num_cr*ele_reg + reg_seven)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven])
        ))

    with tvm_ib.if_scope(r_mod > 0):
        with tvm_ib.for_range(0, r_mod, name="num_er") as num_er:
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_res.access_ptr('r',
                                    offset=(tail_align_dst + r_cycle*ele_reg
                                            + num_er))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_tail.dtype, "reg_mov",
                data_tail.access_ptr('w', offset=(r_cycle*ele_reg + num_er)),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))


def _func_more_dim(args):
    """
    function of moving data for more dim scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, reg_addr, param, \
    num_g, num_dim_block_col, dst_ele_block_col, num_ele_one_block, \
    c_0, num_dim_cur_core = args

    dim_before_core = num_g*param.get("num_dim_one_group") \
                      + param.get("block_index")*param.get("num_dim_one_core")
    data_offset = dim_before_core*param.get('dim_ele')
    burst_len_data = _ceil_div((num_dim_cur_core*param.get('dim_ele')),
                               param.get("cp_align_len"))

    # move from data(GM) to data_ub(UB)
    tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                data_ub.access_ptr("w", offset=0),
                                data.access_ptr('r', offset=data_offset),
                                0, 1, burst_len_data, 0, 0))

    before_block_col_index = dim_before_core // num_dim_block_col
    before_dim_index_block_col = dim_before_core % num_dim_block_col
    before_ele_dst = before_block_col_index*dst_ele_block_col \
                     + before_dim_index_block_col*param.get("dim_ele")
    # move from data_ub to data_res
    with tvm_ib.for_range(0, num_dim_cur_core, name="num_d") as num_d:
        ub_offset = num_d*param.get("dim_ele")
        dim_before = dim_before_core + num_d
        dim_before_block_col_index = dim_before // num_dim_block_col
        dim_before_dim_index_block_col = dim_before % num_dim_block_col
        dim_before_ele_dst = dim_before_block_col_index*dst_ele_block_col \
                             + dim_before_dim_index_block_col\
                             * param.get("dim_ele")
        ub_res_offset = dim_before_ele_dst - before_ele_dst

        with tvm_ib.if_scope(
            dim_before_dim_index_block_col < (num_dim_block_col - 1)):
            num_col_in_dim = c_0
            args = tvm_ib, data_ub, data_res, reg, num_col_in_dim, ub_offset, \
                   ub_res_offset, num_ele_one_block, c_0
            _ub_to_res(args)
        with tvm_ib.else_scope():
            num_col_in_dim = dst.shape[1] - (num_dim_block_col - 1)*c_0
            args = tvm_ib, data_ub, data_res, reg, num_col_in_dim, ub_offset, \
                   ub_res_offset, num_ele_one_block, c_0
            _ub_to_res(args)

    # move from data_res(UB) to res(GM)
    dim_after_core = dim_before_core + num_dim_cur_core
    after_block_col_index = dim_after_core // num_dim_block_col
    after_dim_index_block_col = dim_after_core % num_dim_block_col
    after_ele_dst = after_block_col_index*dst_ele_block_col \
                    + after_dim_index_block_col*param.get("dim_ele")
    cur_core_ele_dst = after_ele_dst - before_ele_dst
    tail_align_dst = cur_core_ele_dst - param.get("cp_align_len")
    with tvm_ib.if_scope(tail_align_dst > 0):
        burst_len_dst_one = tail_align_dst + param.get("cp_align_len") - 1
        reg_addr[1] = burst_len_dst_one
        burst_len_dst = reg_addr[1] // param.get("cp_align_len")
        with tvm_ib.if_scope(burst_len_dst <= 65535):
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=before_ele_dst),
                                        data_res.access_ptr("r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))
        args = tvm_ib, param, data_res, data_tail, reg, tail_align_dst
        _reg_mov_align_batch(args)
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=(before_ele_dst
                                                           + tail_align_dst)),
                                    data_tail.access_ptr("r", offset=0),
                                    0, 1, 1, 0, 0))
    with tvm_ib.if_scope(tail_align_dst <= 0):
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w', offset=before_ele_dst),
                                    data_res.access_ptr("r", offset=0),
                                    0, 1, 1, 0, 0))


def _more_dim_ir(dst, data, max_dim, shape_all):
    """
    function of making ir node builder for more dim scene

    """
    tvm_ib = tvm.ir_builder.create()
    param = _get_param_more_dim(tvm_ib, data.shape, dst.dtype,
                                max_dim, shape_all)

    data_ub = _new_alloc(tvm_ib, dst.dtype,
                         param.get('num_dim_one_core')*param.get("dim_ele"),
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype,
                          param.get('num_dim_one_core')*param.get("dim_ele"),
                          "data_res", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    data_tail = _new_alloc(tvm_ib, dst.dtype, param.get('cp_align_len'),
                           "data_tail", scope=cce.scope_ubuf)
    reg_addr = tvm_ib.allocate("int32", (8,), name='reg_addr',
                               scope=cce.scope_reg)
    c_0 = 16
    num_dim_block_col = _ceil_div(dst.shape[1], c_0)
    dst_ele_block_col = functools_reduce(lambda x, y: x*y, dst.shape[1:])
    num_ele_one_block = functools_reduce(lambda x, y: x*y, dst.shape[2:])

    with tvm_ib.for_range(0, param.get("num_group_index") + 1,
                          name="num_g") as num_g:
        with tvm_ib.if_scope(num_g < param.get("num_group_index")):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   reg_addr, param, num_g, num_dim_block_col, \
                   dst_ele_block_col, num_ele_one_block, \
                   c_0, param.get("num_dim_one_core")
            _func_more_dim(args)
            _func_pipe(tvm_ib)
        with tvm_ib.if_scope(tvm.all(num_g >= param.get("num_group_index"),
                                     param.get("num_group_mod") > 0)):
            num_core = param.get("num_group_mod") // param.get(
                "num_dim_one_core")
            num_dim_mod = param.get("num_group_mod") - param.get(
                "num_dim_one_core")*num_core
            with tvm_ib.if_scope(num_core > 0):
                with tvm_ib.if_scope(param.get("block_index") < num_core):
                    args = tvm_ib, data, dst, data_ub, data_res, data_tail, \
                           reg, reg_addr, param, num_g, num_dim_block_col, \
                           dst_ele_block_col, num_ele_one_block, \
                           c_0, param.get("num_dim_one_core")
                    _func_more_dim(args)
            with tvm_ib.if_scope(num_dim_mod > 0):
                with tvm_ib.if_scope(
                    tvm.all(param.get("block_index") < (num_core + 1),
                            param.get("block_index") > (num_core - 1))):
                    args = tvm_ib, data, dst, data_ub, data_res, data_tail, \
                           reg, reg_addr, param, num_g, num_dim_block_col, \
                           dst_ele_block_col, num_ele_one_block, \
                           c_0, num_dim_mod
                    _func_more_dim(args)
            _func_pipe(tvm_ib)

    return tvm_ib.get()


def _get_param_one_dim(tvm_ib, src_shape, dtype, max_dim, shape_all):
    """
    calculate parameters for one dim ir builder make function

    """
    device_core_num = AICORE_NUM
    if shape_all > 380000000:
        device_core_num = 1
    elif device_core_num == 32 and (max_dim > 800000 or shape_all > 320000000):
        device_core_num = 16
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    dim_ele = functools_reduce(lambda x, y: x*y, src_shape[2:])
    num_dim_in_data = functools_reduce(lambda x, y: x*y, src_shape[0:2])
    num_row_in_dim = functools_reduce(lambda x, y: x*y, src_shape[2:4])
    num_dim_one_core = 1
    num_dim_one_group = num_dim_one_core*device_core_num
    num_group_index = num_dim_in_data // num_dim_one_group
    num_group_mod = num_dim_in_data % num_dim_one_group

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    param_map = {"num_group_index": num_group_index,
                 "num_group_mod": num_group_mod,
                 "num_dim_one_core": num_dim_one_core, "dim_ele": dim_ele,
                 "num_row_in_dim": num_row_in_dim, "cp_align_len": cp_align_len,
                 "num_dim_one_group": num_dim_one_group,
                 "block_index": block_index}

    return param_map


def _reg_mov_one_dim(args):
    """
    move from data_ub to data_res to change the C axis for one dim scene

    """
    tvm_ib, data_ub, data_res, reg, num_row, c_0, num_c = args

    ele_reg = 8
    r_cycle = num_row // ele_reg
    r_mod = num_row - ele_reg*r_cycle
    reg_zero = 0
    reg_one = 1
    reg_two = 2
    reg_three = 3
    reg_four = 4
    reg_five = 5
    reg_six = 6
    reg_seven = 7

    with tvm_ib.for_range(0, r_cycle, name="num_cr") as num_cr:
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero]),
            data_ub.access_ptr('r',
                               offset=(num_c + (
                                   num_cr*ele_reg + reg_zero)*c_0))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_one]),
            data_ub.access_ptr('r',
                               offset=(num_c + (
                                   num_cr*ele_reg + reg_one)*c_0))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_two]),
            data_ub.access_ptr('r',
                               offset=(num_c + (
                                   num_cr*ele_reg + reg_two)*c_0))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_three]),
            data_ub.access_ptr('r',
                               offset=(num_c + (
                                   num_cr*ele_reg + reg_three)*c_0))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_four]),
            data_ub.access_ptr('r',
                               offset=(num_c + (
                                   num_cr*ele_reg + reg_four)*c_0))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_five]),
            data_ub.access_ptr('r',
                               offset=(num_c + (
                                   num_cr*ele_reg + reg_five)*c_0))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_six]),
            data_ub.access_ptr('r',
                               offset=(num_c + (
                                   num_cr*ele_reg + reg_six)*c_0))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven]),
            data_ub.access_ptr('r',
                               offset=(num_c + (
                                   num_cr*ele_reg + reg_seven)*c_0))
        ))

        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_zero)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_one)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_one])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_two)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_two])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_three)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_three])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_four)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_four])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_five)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_five])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_six)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_six])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_seven)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven])
        ))

    with tvm_ib.if_scope(r_mod > 0):
        with tvm_ib.for_range(0, r_mod, name="num_er") as num_er:
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_ub.access_ptr('r',
                                   offset=(num_c + (
                                       r_cycle*ele_reg + num_er)*c_0))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=(r_cycle*ele_reg + num_er)),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))


def _ub_gm_one_dim(args):
    """
    move data from data_ub to data_res to res(GM)

    """
    tvm_ib, param, data_ub, data_res, data_tail, reg, reg_addr, dst, \
    num_col_in_dim, tail_align_dst, block_col_index, \
    num_ele_block_col, block_col_mod, c_0 = args

    tail_align_col_index = tail_align_dst // param.get("num_row_in_dim")
    tail_align_row_index = tail_align_dst - param.get(
        "num_row_in_dim")*tail_align_col_index
    reg_addr[3] = tail_align_row_index
    with tvm_ib.for_range(0, num_col_in_dim, name="num_c") as num_c:
        with tvm_ib.if_scope(num_c < tail_align_col_index):
            # move from data_ub to data_res
            args = tvm_ib, data_ub, data_res, reg, param.get(
                "num_row_in_dim"), c_0, num_c
            _reg_mov_one_dim(args)
            # move from data_res(UB) to res(GM)
            dst_offset = block_col_index*num_ele_block_col \
                         + block_col_mod*param.get("dim_ele")
            burst_len_dst = _ceil_div(param.get("num_row_in_dim"),
                                      param.get("cp_align_len"))
            with tvm_ib.if_scope(burst_len_dst <= 65535):
                tvm_ib.emit(
                    tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=(dst_offset
                                                           + num_c
                                                           * param.get(
                                                               "num_row"
                                                               "_in_dim"))),
                                    data_res.access_ptr("r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))

        with tvm_ib.if_scope(num_c == tail_align_col_index):
            with tvm_ib.if_scope(tail_align_row_index > 0):
                args = tvm_ib, data_ub, data_res, reg, \
                       reg_addr[3], c_0, num_c
                _reg_mov_one_dim(args)
                burst_len_dst_one = tail_align_row_index \
                                    + param.get("cp_align_len") - 1
                reg_addr[4] = burst_len_dst_one
                burst_len_dst = reg_addr[4] // param.get("cp_align_len")
                with tvm_ib.if_scope(burst_len_dst <= 65535):
                    tvm_ib.emit(
                        tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=(dst_offset
                                                               + num_c
                                                               * param.get(
                                                                   "num_row"
                                                                   "_in_"
                                                                   "dim"))),
                                        data_res.access_ptr("r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))

            with tvm_ib.for_range(0, param.get("num_row_in_dim") \
                                - tail_align_row_index, name="num_t") as num_t:
                tvm_ib.emit(tvm.call_extern(
                    data_ub.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_ub.access_ptr('r', offset=(
                        num_c + (num_t + tail_align_row_index)*c_0))
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    data_tail.access_ptr('w', offset=num_t),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))
            dst_offset_tail = dst_offset + tail_align_dst
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset_tail),
                                        data_tail.access_ptr("r", offset=0),
                                        0, 1, 1, 0, 0))


def _func_one_dim(args_one_dim):
    """
    function of moving data for one dim scene

    """
    tvm_ib, param, data, dst, data_ub, data_res, data_tail, reg, reg_addr, \
    num_g, c_0, num_dim_block_col, num_ele_block_col = args_one_dim

    dim_index = num_g*param.get("num_dim_one_group") \
                + param.get("block_index")*param.get("num_dim_one_core")
    # move from data(GM) to data_ub(UB)
    data_offset = dim_index*param.get("dim_ele")
    burst_len_data = _ceil_div(param.get("dim_ele"), param.get("cp_align_len"))
    tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                data_ub.access_ptr("w", offset=0),
                                data.access_ptr('r', offset=data_offset),
                                0, 1, burst_len_data, 0, 0))
    # move from data_ub to data_res to res(GM)
    block_col_index = dim_index // num_dim_block_col
    block_col_mod = dim_index - num_dim_block_col*block_col_index

    with tvm_ib.if_scope(block_col_mod < (num_dim_block_col - 1)):
        num_col_in_dim = c_0
        tail_align_dst = param.get("dim_ele") - param.get("cp_align_len")
        reg_addr[1] = tail_align_dst
        args = tvm_ib, param, data_ub, data_res, data_tail, reg, reg_addr, \
               dst, num_col_in_dim, reg_addr[1], block_col_index, \
               num_ele_block_col, block_col_mod, c_0
        _ub_gm_one_dim(args)

    with tvm_ib.if_scope(block_col_mod >= (num_dim_block_col - 1)):
        num_col_in_dim = dst.shape[1] - ((num_dim_block_col - 1)*c_0)
        tail_align_dst = num_col_in_dim \
                         * functools_reduce(lambda x, y: x*y, dst.shape[2:]) \
                         - param.get("cp_align_len")
        reg_addr[1] = tail_align_dst
        reg_addr[2] = num_col_in_dim
        args = tvm_ib, param, data_ub, data_res, data_tail, reg, reg_addr, \
               dst, reg_addr[2], reg_addr[1], block_col_index, \
               num_ele_block_col, block_col_mod, c_0
        _ub_gm_one_dim(args)


def _one_dim_ir(dst, data, max_dim, shape_all):
    """
    function of making ir node builder for one dim scene

    """
    tvm_ib = tvm.ir_builder.create()
    param = _get_param_one_dim(tvm_ib, data.shape, dst.dtype,
                               max_dim, shape_all)
    data_ub = _new_alloc(tvm_ib, dst.dtype,
                         param.get('num_dim_one_core')*param.get("dim_ele"),
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, param.get('num_row_in_dim'),
                          "data_res", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    data_tail = _new_alloc(tvm_ib, dst.dtype, param.get('cp_align_len'),
                           "data_tail", scope=cce.scope_ubuf)
    reg_addr = tvm_ib.allocate("int32", (8,), name='reg_addr',
                               scope=cce.scope_reg)

    c_0 = 16
    num_dim_block_col = _ceil_div(dst.shape[1], c_0)
    num_ele_block_col = functools_reduce(lambda x, y: x*y, dst.shape[1:])

    with tvm_ib.for_range(0, param.get("num_group_index") + 1,
                          name="num_g") as num_g:
        with tvm_ib.if_scope(num_g < param.get("num_group_index")):
            args_one_dim = tvm_ib, param, data, dst, data_ub, data_res, \
                           data_tail, reg, reg_addr, num_g, c_0, \
                           num_dim_block_col, num_ele_block_col
            _func_one_dim(args_one_dim)
            _func_pipe(tvm_ib)
        with tvm_ib.if_scope(num_g >= param.get("num_group_index")):
            with tvm_ib.if_scope(param.get("num_group_mod") > 0):
                with tvm_ib.if_scope(param.get("block_index") \
                                     < param.get("num_group_mod")):
                    args_one_dim = tvm_ib, param, data, dst, data_ub, data_res,\
                                   data_tail, reg, reg_addr, num_g, c_0, \
                                   num_dim_block_col, num_ele_block_col
                    _func_one_dim(args_one_dim)
                _func_pipe(tvm_ib)

    return tvm_ib.get()


def _get_param_split_dim(tvm_ib, src_shape, dtype, max_dim, shape_all):
    """
    calculate parameters for split dim ir builder make function

    """
    c_0 = 16
    _, _, h_i, w_i, _ = src_shape
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64 - cp_align_len*32*2
    device_core_num = AICORE_NUM
    if shape_all > 380000000:
        device_core_num = 1
        ub_bytes = UB_SIZE_B // 2 - 64 - (cp_align_len*32*2)
    elif device_core_num == 32 and (max_dim > 800000 or shape_all > 320000000):
        device_core_num = 16
    ub_ele = ub_bytes // float_size
    num_row_one_core = ((ub_ele // (c_0 + 1)) // cp_align_len)*cp_align_len
    num_row_one_group = num_row_one_core*device_core_num
    num_row_in_data = _ceil_fill(h_i*w_i, cp_align_len) *\
                      functools_reduce(lambda x, y: x*y, src_shape[0:2])
    num_group_index = num_row_in_data // num_row_one_group
    num_group_mod = num_row_in_data % num_row_one_group

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    param_map = {"num_group_index": num_group_index,
                 "num_group_mod": num_group_mod,
                 "block_index": block_index,
                 "num_row_one_group": num_row_one_group,
                 "num_row_one_core": num_row_one_core,
                 "cp_align_len": cp_align_len}

    return param_map


def _dst_to_data_pos(args):
    """
    calculate position of element in data according to position in res

    """
    pos_dst, num_ele_block_col, dim_ele, \
    num_row_one_dim, num_dim_block_col, c_0 = args

    block_col_index = pos_dst // num_ele_block_col
    block_col_mod = pos_dst % num_ele_block_col
    dim_index = block_col_mod // dim_ele
    dim_mod = block_col_mod % dim_ele
    col = dim_mod // num_row_one_dim
    row = dim_mod % num_row_one_dim
    dim_all = block_col_index*num_dim_block_col + dim_index
    pos_data = dim_all*dim_ele + row*c_0 + col

    return pos_data


def _move_align(args):
    """
    move head and tail of align data to prevent overwriting data in res

    """
    tvm_ib, param, data, data_align, data_res, reg, \
    num_ele_align, pos_data, ub_res_offset = args

    if data.dtype == "float32":
        src_stride = 1
    elif data.dtype == "float16":
        src_stride = 0
    tvm_ib.emit(tvm.call_extern(data_align.dtype, "copy_gm_to_ubuf",
                                data_align.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=pos_data),
                                0, num_ele_align, 1, src_stride, 0))

    with tvm_ib.for_range(0, num_ele_align, name="num_b") as num_b:
        tvm_ib.emit(tvm.call_extern(
            data_align.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[0]),
            data_align.access_ptr('r', offset=num_b*param.get("cp_align_len"))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=ub_res_offset + num_b),
            tvm.call_extern(reg.dtype, "reg", reg[0])
        ))


def _reg_mov_batch(args):
    """
    move from data_ub to data_res to change the C axis for split dim scene

    """
    tvm_ib, data_ub, data_res, reg, ub_res_offset, \
    ub_offset, num_row, c_0, num_c = args

    ele_reg = 8
    r_cycle = num_row // ele_reg
    r_mod = num_row - ele_reg*r_cycle
    reg_zero = 0
    reg_one = 1
    reg_two = 2
    reg_three = 3
    reg_four = 4
    reg_five = 5
    reg_six = 6
    reg_seven = 7

    with tvm_ib.for_range(0, r_cycle, name="num_cr") as num_cr:
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero]),
            data_ub.access_ptr('r',
                               offset=(ub_offset + (
                                   num_cr*ele_reg + reg_zero)*c_0 + num_c))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_one]),
            data_ub.access_ptr('r',
                               offset=(ub_offset + (
                                   num_cr*ele_reg + reg_one)*c_0 + num_c))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_two]),
            data_ub.access_ptr('r',
                               offset=(ub_offset + (
                                   num_cr*ele_reg + reg_two)*c_0 + num_c))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_three]),
            data_ub.access_ptr('r',
                               offset=(ub_offset + (
                                   num_cr*ele_reg + reg_three)*c_0 + num_c))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_four]),
            data_ub.access_ptr('r',
                               offset=(ub_offset + (
                                   num_cr*ele_reg + reg_four)*c_0 + num_c))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_five]),
            data_ub.access_ptr('r',
                               offset=(ub_offset + (
                                   num_cr*ele_reg + reg_five)*c_0 + num_c))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_six]),
            data_ub.access_ptr('r',
                               offset=(ub_offset + (
                                   num_cr*ele_reg + reg_six)*c_0 + num_c))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven]),
            data_ub.access_ptr('r',
                               offset=(ub_offset + (
                                   num_cr*ele_reg + reg_seven)*c_0 + num_c))
        ))

        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(
                ub_res_offset + (num_cr*ele_reg + reg_zero))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(
                ub_res_offset + (num_cr*ele_reg + reg_one))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_one])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(
                ub_res_offset + (num_cr*ele_reg + reg_two))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_two])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(
                ub_res_offset + (num_cr*ele_reg + reg_three))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_three])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(
                ub_res_offset + (num_cr*ele_reg + reg_four))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_four])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(
                ub_res_offset + (num_cr*ele_reg + reg_five))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_five])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(
                ub_res_offset + (num_cr*ele_reg + reg_six))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_six])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(
                ub_res_offset + (num_cr*ele_reg + reg_seven))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven])
        ))

    with tvm_ib.if_scope(r_mod > 0):
        with tvm_ib.for_range(0, r_mod, name="num_er") as num_er:
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_ub.access_ptr('r',
                                   offset=(ub_offset + (
                                       r_cycle*ele_reg + num_er)*c_0 + num_c))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=(
                    ub_res_offset + (r_cycle*ele_reg + num_er))),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))


def _move_align_batch(args):
    """
    batch move head and tail of align data to prevent overwriting data in res

    """
    tvm_ib, param, data, data_align, data_res, reg, \
    num_ele_align, pos_data, ub_res_offset = args

    if data.dtype == "float32":
        src_stride = 1
    elif data.dtype == "float16":
        src_stride = 0
    tvm_ib.emit(tvm.call_extern(data_align.dtype, "copy_gm_to_ubuf",
                                data_align.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=pos_data),
                                0, num_ele_align, 1, src_stride, 0))

    ele_reg = 8
    r_cycle = num_ele_align // ele_reg
    r_mod = num_ele_align - ele_reg*r_cycle
    reg_zero = 0
    reg_one = 1
    reg_two = 2
    reg_three = 3
    reg_four = 4
    reg_five = 5
    reg_six = 6
    reg_seven = 7

    with tvm_ib.for_range(0, r_cycle, name="num_rc") as num_rc:
        tvm_ib.emit(tvm.call_extern(
            data_align.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero]),
            data_align.access_ptr('r',
                                  offset=(num_rc*ele_reg + reg_zero)
                                  * param.get("cp_align_len"))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_align.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_one]),
            data_align.access_ptr('r',
                                  offset=(num_rc*ele_reg + reg_one)
                                  * param.get("cp_align_len"))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_align.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_two]),
            data_align.access_ptr('r',
                                  offset=(num_rc*ele_reg + reg_two)
                                  * param.get("cp_align_len"))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_align.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_three]),
            data_align.access_ptr('r',
                                  offset=(num_rc*ele_reg + reg_three)
                                  * param.get("cp_align_len"))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_align.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_four]),
            data_align.access_ptr('r',
                                  offset=(num_rc*ele_reg + reg_four)
                                  * param.get("cp_align_len"))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_align.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_five]),
            data_align.access_ptr('r',
                                  offset=(num_rc*ele_reg + reg_five)
                                  * param.get("cp_align_len"))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_align.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_six]),
            data_align.access_ptr('r',
                                  offset=(num_rc*ele_reg + reg_six)
                                  * param.get("cp_align_len"))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_align.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven]),
            data_align.access_ptr('r',
                                  offset=(num_rc*ele_reg + reg_seven)
                                  * param.get("cp_align_len"))
        ))

        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=ub_res_offset + (
                num_rc*ele_reg + reg_zero)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=ub_res_offset + (
                num_rc*ele_reg + reg_one)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_one])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=ub_res_offset + (
                num_rc*ele_reg + reg_two)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_two])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=ub_res_offset + (
                num_rc*ele_reg + reg_three)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_three])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=ub_res_offset + (
                num_rc*ele_reg + reg_four)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_four])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=ub_res_offset + (
                num_rc*ele_reg + reg_five)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_five])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=ub_res_offset + (
                num_rc*ele_reg + reg_six)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_six])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=ub_res_offset + (
                num_rc*ele_reg + reg_seven)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven])
        ))
    with tvm_ib.if_scope(r_mod > 0):
        with tvm_ib.for_range(0, r_mod, name="num_er") as num_er:
            tvm_ib.emit(tvm.call_extern(
                data_align.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_align.access_ptr('r',
                                      offset=(r_cycle*ele_reg + num_er)
                                      * param.get("cp_align_len"))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=ub_res_offset + (
                    r_cycle*ele_reg + num_er)),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))


def _ub_gm_split_dim_one(args):
    """
    move from data_ub to data_res to res(GM) for split dim scene

    """
    tvm_ib, param, dst, data_ub, data_res, reg, num_col_in_dim, \
    num_row_one_dim, num_row, c_0, ub_offset, dst_offset = args

    with tvm_ib.for_range(0, num_col_in_dim, name="num_c") as num_c:
        ele_dst_pos = dst_offset + num_c*num_row_one_dim
        # move from data_ub to data_res
        ub_res_offset = 0
        args = tvm_ib, data_ub, data_res, reg, ub_res_offset, \
               ub_offset, num_row, c_0, num_c
        _reg_mov_batch(args)

        burst_len_dst = num_row // param.get("cp_align_len")
        with tvm_ib.if_scope(burst_len_dst <= 65535):
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=ele_dst_pos),
                                        data_res.access_ptr("r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))


def _ub_gm_split_dim_two(args):
    """
    move from data_ub to data_res to res(GM) for split dim scene

    """
    tvm_ib, param, data, dst, data_ub, data_res, data_align, reg, reg_addr, \
    num_col_in_dim, num_ele_block_col, dim_ele, num_row_one_dim, num_row, \
    num_dim_block_col, c_0, ub_offset, dst_offset = args

    with tvm_ib.for_range(0, num_col_in_dim, name="num_c") as num_c:
        ele_dst_pos = dst_offset + num_c*num_row_one_dim
        end_dst_pos = ele_dst_pos + num_row
        num_ele = param.get("cp_align_len")

        # move from data_ub to data_res
        ub_res_offset = 0
        args = tvm_ib, data_ub, data_res, reg, ub_res_offset, \
               ub_offset, num_row, c_0, num_c
        _reg_mov_batch(args)

        args = end_dst_pos, num_ele_block_col, dim_ele, \
               num_row_one_dim, num_dim_block_col, c_0
        end_data_pos = _dst_to_data_pos(args)
        ub_res_offset_align = num_row
        args = tvm_ib, param, data, data_align, data_res, \
               reg, num_ele, end_data_pos, ub_res_offset_align
        _move_align_batch(args)

        # move from data_res(UB) to res(GM)
        burst_len_dst_one = num_row + param.get("cp_align_len") - 1
        reg_addr[2] = burst_len_dst_one
        burst_len_dst = reg_addr[2] // param.get("cp_align_len")
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=ele_dst_pos),
                                    data_res.access_ptr("r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))


# pylint: disable=locally-disabled,too-many-statements
def _func_split_dim(args):
    """
    function of moving data for split dim scene

    """
    tvm_ib, param, data, dst, data_ub, data_res, \
    data_align, reg, reg_addr, c_0, num_g, num_dim_block_col, \
    num_ele_block_col, dim_ele, num_row_cur_core = args

    c_0 = 16
    _, c_i, h_i, w_i = dst.shape
    num_row_one_dim_true = h_i*w_i
    num_row_one_dim_space = _ceil_fill(h_i*w_i, param.get("cp_align_len"))
    num_dim_one_block_col = data.shape[1]
    num_ele_block_col = functools_reduce(lambda x, y: x*y, dst.shape[1:])
    dim_ele = functools_reduce(lambda x, y: x*y, data.shape[2:])

    num_row_before_core = num_g*param.get("num_row_one_group") \
                          + param.get("block_index")*param.get(
                              "num_row_one_core")
    num_dim_before_core = num_row_before_core // num_row_one_dim_space
    num_row_cur_dim_before = num_row_before_core - \
                             num_dim_before_core*num_row_one_dim_space
    num_row_dim_head_true = num_row_one_dim_true - num_row_cur_dim_before
    reg_addr[1] = num_row_dim_head_true
    num_row_dim_head_space = num_row_one_dim_space - num_row_cur_dim_before

    num_row_before_core_ture = _num_row_space_to_true(num_row_before_core,
                                                      num_row_one_dim_true,
                                                      num_row_one_dim_space)

    with tvm_ib.if_scope(num_row_dim_head_space > num_row_cur_core):
        len_burst = num_row_cur_core*2
        with tvm_ib.if_scope(len_burst <= 65535):
            tvm_ib.emit(
                tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                data_ub.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=(num_row_before_core_ture
                                                        * c_0)),
                                0, 1, len_burst, 0, 0))

        dim_index = num_row_before_core_ture // num_row_one_dim_true
        num_block_col = dim_index // num_dim_one_block_col
        cur_dim_index = dim_index - num_dim_one_block_col*num_block_col
        flag_dim = (cur_dim_index + 1)*c_0
        num_row = num_row_cur_core
        ub_offset = 0
        dst_offset = num_block_col*num_ele_block_col \
                     + cur_dim_index*dim_ele \
                     + num_row_cur_dim_before
        with tvm_ib.if_scope(flag_dim <= c_i):
            num_col_in_dim = c_0
            args = tvm_ib, param, dst, data_ub, data_res, reg, num_col_in_dim, \
                   num_row_one_dim_true, num_row, c_0, ub_offset, dst_offset
            _ub_gm_split_dim_one(args)
        with tvm_ib.if_scope(flag_dim > c_i):
            num_col_in_dim = c_i % c_0
            args = tvm_ib, param, dst, data_ub, data_res, reg, num_col_in_dim, \
                   num_row_one_dim_true, num_row, c_0, ub_offset, dst_offset
            _ub_gm_split_dim_one(args)
    with tvm_ib.if_scope(num_row_dim_head_space == num_row_cur_core):
        len_burst = num_row_dim_head_true*2
        with tvm_ib.if_scope(len_burst <= 65535):
            tvm_ib.emit(
                tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                data_ub.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=(num_row_before_core_ture
                                                        * c_0)),
                                0, 1, len_burst, 0, 0))

        dim_index = num_row_before_core_ture // num_row_one_dim_true
        num_block_col = dim_index // num_dim_one_block_col
        cur_dim_index = dim_index - num_dim_one_block_col*num_block_col
        flag_dim = (cur_dim_index + 1)*c_0
        num_row = reg_addr[1]
        ub_offset = 0
        dst_offset = num_block_col*num_ele_block_col \
                     + cur_dim_index*dim_ele \
                     + num_row_cur_dim_before
        with tvm_ib.if_scope(flag_dim <= c_i):
            num_col_in_dim = c_0
            args = tvm_ib, param, data, dst, data_ub, data_res, \
                   data_align, reg, reg_addr, num_col_in_dim, \
                   num_ele_block_col, dim_ele, num_row_one_dim_true, \
                   num_row, num_dim_block_col, c_0, ub_offset, dst_offset
            _ub_gm_split_dim_two(args)
        with tvm_ib.if_scope(flag_dim > c_i):
            num_col_in_dim = c_i % c_0
            args = tvm_ib, param, data, dst, data_ub, data_res, \
                   data_align, reg, reg_addr, num_col_in_dim, \
                   num_ele_block_col, dim_ele, num_row_one_dim_true, \
                   num_row, num_dim_block_col, c_0, ub_offset, dst_offset
            _ub_gm_split_dim_two(args)
    with tvm_ib.if_scope(tvm.all(
        num_row_dim_head_space < num_row_cur_core,
        (num_row_dim_head_space + num_row_one_dim_space)
        > num_row_cur_core)):
        len_burst_head = num_row_dim_head_true*2
        with tvm_ib.if_scope(len_burst_head <= 65535):
            tvm_ib.emit(
                tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                data_ub.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=(num_row_before_core_ture
                                                        * c_0)),
                                0, 1, len_burst_head, 0, 0))

        num_row_before_tail = (num_dim_before_core + 1)*num_row_one_dim_space
        num_row_before_tail_true = _num_row_space_to_true(num_row_before_tail,
                                                          num_row_one_dim_true,
                                                          num_row_one_dim_space)
        num_row_tail = (num_row_cur_core - num_row_dim_head_space)*2
        with tvm_ib.if_scope(num_row_tail <= 65535):
            tvm_ib.emit(
                tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                data_ub.access_ptr("w",
                                                   offset=num_row_dim_head_space
                                                   * c_0),
                                data.access_ptr('r',
                                                offset=(num_row_before_tail_true
                                                        * c_0)),
                                0, 1, num_row_tail, 0, 0))

        dim_index = num_row_before_core_ture // num_row_one_dim_true
        num_block_col = dim_index // num_dim_one_block_col
        cur_dim_index = dim_index - num_dim_one_block_col*num_block_col
        flag_dim = (cur_dim_index + 1)*c_0
        ub_offset = 0
        dst_offset = num_block_col*num_ele_block_col \
                     + cur_dim_index*dim_ele \
                     + num_row_cur_dim_before
        with tvm_ib.if_scope(flag_dim <= c_i):
            num_col_in_dim = c_0
            args = tvm_ib, param, data, dst, data_ub, data_res, \
                   data_align, reg, reg_addr, num_col_in_dim, \
                   num_ele_block_col, dim_ele, num_row_one_dim_true, \
                   reg_addr[1], num_dim_block_col, \
                   c_0, ub_offset, dst_offset
            _ub_gm_split_dim_two(args)
        with tvm_ib.if_scope(flag_dim > c_i):
            num_col_in_dim = c_i % c_0
            args = tvm_ib, param, data, dst, data_ub, data_res, \
                   data_align, reg, reg_addr, num_col_in_dim, \
                   num_ele_block_col, dim_ele, num_row_one_dim_true, \
                   reg_addr[1], num_dim_block_col, \
                   c_0, ub_offset, dst_offset
            _ub_gm_split_dim_two(args)

        dim_index_tail = dim_index + 1
        num_block_col_tail = dim_index_tail // num_dim_one_block_col
        cur_dim_index_tail = dim_index_tail - \
                             num_dim_one_block_col*num_block_col_tail
        flag_dim_tail = (cur_dim_index_tail + 1)*c_0
        num_row = num_row_cur_core - num_row_dim_head_space
        reg_addr[3] = num_row
        dst_offset = num_block_col_tail*num_ele_block_col \
                     + cur_dim_index_tail*dim_ele
        ub_offset = num_row_dim_head_space*c_0
        with tvm_ib.if_scope(flag_dim_tail <= c_i):
            num_col_in_dim = c_0
            args = tvm_ib, param, dst, data_ub, data_res, reg, num_col_in_dim, \
                   num_row_one_dim_true, reg_addr[3], c_0, ub_offset, dst_offset
            _ub_gm_split_dim_one(args)
        with tvm_ib.if_scope(flag_dim_tail > c_i):
            num_col_in_dim = c_i % c_0
            args = tvm_ib, param, dst, data_ub, data_res, reg, num_col_in_dim, \
                   num_row_one_dim_true, reg_addr[3], c_0, ub_offset, dst_offset
            _ub_gm_split_dim_one(args)


def _split_dim_ir(dst, data, max_dim, shape_all):
    """
    function of making ir node builder for split dim scene

    """
    tvm_ib = tvm.ir_builder.create()
    param = _get_param_split_dim(tvm_ib, data.shape, dst.dtype,
                                 max_dim, shape_all)
    c_0 = 16
    data_ub = _new_alloc(tvm_ib, dst.dtype, param.get('num_row_one_core')*c_0,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, param.get('num_row_one_core')
                          + 2*param.get("cp_align_len"),
                          "data_res", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    data_align = _new_alloc(tvm_ib, dst.dtype,
                            param.get('cp_align_len')*param.get(
                                "cp_align_len"),
                            "data_align", scope=cce.scope_ubuf)
    reg_addr = tvm_ib.allocate("int32", (8,), name='reg_addr',
                               scope=cce.scope_reg)
    num_dim_block_col = _ceil_div(dst.shape[1], c_0)
    num_ele_block_col = functools_reduce(lambda x, y: x*y, dst.shape[1:])
    dim_ele = functools_reduce(lambda x, y: x*y, data.shape[2:])

    with tvm_ib.for_range(0, param.get("num_group_index") + 1,
                          name="num_g") as num_g:
        with tvm_ib.if_scope(num_g < param.get("num_group_index")):
            num_row_cur_core = param.get("num_row_one_core")
            args = tvm_ib, param, data, dst, data_ub, data_res, data_align, \
                   reg, reg_addr, c_0, num_g, num_dim_block_col, \
                   num_ele_block_col, dim_ele, num_row_cur_core
            _func_split_dim(args)
            _func_pipe(tvm_ib)
        with tvm_ib.if_scope(num_g >= param.get("num_group_index")):
            with tvm_ib.if_scope(param.get("num_group_mod") > 0):
                num_core = param.get("num_group_mod") // param.get(
                    "num_row_one_core")
                num_row_mod = param.get("num_group_mod") - param.get(
                    "num_row_one_core")*num_core
                with tvm_ib.if_scope(param.get("block_index") < num_core):
                    num_row_cur_core = param.get("num_row_one_core")
                    args = tvm_ib, param, data, dst, data_ub, data_res, \
                           data_align, reg, reg_addr, c_0, num_g, \
                           num_dim_block_col, num_ele_block_col, \
                           dim_ele, num_row_cur_core
                    _func_split_dim(args)
                with tvm_ib.if_scope(num_row_mod > 0):
                    with tvm_ib.if_scope(
                        tvm.all(param.get("block_index") < (num_core + 1),
                                param.get("block_index") > (num_core - 1))):
                        num_row_cur_core = num_row_mod
                        args = tvm_ib, param, data, dst, data_ub, data_res, \
                               data_align, reg, reg_addr, c_0, num_g, \
                               num_dim_block_col, num_ele_block_col, \
                               dim_ele, num_row_cur_core
                        _func_split_dim(args)
                    _func_pipe(tvm_ib)

    return tvm_ib.get()


# pylint: disable = locally-disabled,too-many-arguments
def _get_param_more_dim_fp16(tvm_ib, src_shape, dst_shape, dtype,
                             max_dim, shape_all):
    """
    calculate parameters for float16 more dim ir builder make function

    """
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - (cp_align_len*32)
    device_core_num = AICORE_NUM
    if shape_all > 380000000:
        device_core_num = 1
        ub_bytes = UB_SIZE_B // 2 - (cp_align_len*32)
    elif device_core_num == 32 and (max_dim > 800000 or shape_all > 320000000):
        device_core_num = 16
    ub_half = ub_bytes // 2
    c_0 = 16
    _, _, h_i, w_i = dst_shape
    src_dim_space = _ceil_fill(h_i*w_i, c_0)*c_0
    src_dim_space_bytes = src_dim_space*float_size

    num_dim_in_data = functools_reduce(lambda x, y: x*y, src_shape[0:2])
    num_dim_one_core = ub_half // src_dim_space_bytes
    num_dim_one_group = num_dim_one_core*device_core_num
    num_group_index = num_dim_in_data // num_dim_one_group
    num_group_mod = num_dim_in_data - num_dim_one_group*num_group_index
    src_dim_gm = h_i*w_i*c_0

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    param_map = {"num_group_index": num_group_index,
                 "num_group_mod": num_group_mod,
                 "src_dim_space": src_dim_space,
                 "src_dim_space_bytes": src_dim_space_bytes,
                 "num_dim_one_core": num_dim_one_core,
                 "cp_align_len": cp_align_len,
                 "block_index": block_index,
                 "num_dim_one_group": num_dim_one_group,
                 "src_dim_gm": src_dim_gm, "float_size": float_size}

    return param_map


def _move_align_new(args):
    """
    move head and tail of align data to prevent overwriting data in res

    """
    tvm_ib, data, data_align, data_res, reg, \
    pos_data, ub_res_offset = args

    tvm_ib.emit(tvm.call_extern(data_align.dtype, "copy_gm_to_ubuf",
                                data_align.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=pos_data),
                                0, 1, 1, 0, 0))
    tvm_ib.emit(tvm.call_extern(
        data_align.dtype, "reg_mov",
        tvm.call_extern(reg.dtype, "reg", reg[0]),
        data_align.access_ptr('r', offset=0)
    ))
    tvm_ib.emit(tvm.call_extern(
        data_res.dtype, "reg_mov",
        data_res.access_ptr('w', offset=ub_res_offset),
        tvm.call_extern(reg.dtype, "reg", reg[0])
    ))


def _ub_res_to_gm_more(args):
    """
    move from output_ub to dst(GM) for tail in more dim float16 scene

    """
    tvm_ib, param, data, dst, output_ub, data_align, \
    reg, reg_addr, num_block_one_dim, num_dim_dst_col, num_ele_block_col, \
    dim_ele, dst_offset, out_offset, num_col = args

    c_0 = 16
    len_burst = _ceil_div(num_block_one_dim, param.get("cp_align_len"))
    num_block_one_dim_space = _ceil_fill(num_block_one_dim,
                                         param.get("cp_align_len"))
    with tvm_ib.for_range(0, num_col, name="num_c") as num_c:
        with tvm_ib.if_scope(num_block_one_dim % c_0 != 0):
            num_ele_align = num_block_one_dim_space - num_block_one_dim
            reg_addr[11] = num_ele_align

            with tvm_ib.if_scope(num_ele_align > num_block_one_dim):
                with tvm_ib.for_range(0, reg_addr[11],
                                      name="num_ne") as num_ne:
                    align_dst_pos = dst_offset + num_block_one_dim*(
                        num_c + 1) + num_ne
                    args = align_dst_pos, num_ele_block_col, dim_ele, \
                           num_block_one_dim, num_dim_dst_col, c_0
                    align_data_pos = _dst_to_data_pos(args)
                    ub_res_offset = out_offset + \
                                    num_c*num_block_one_dim_space \
                                    + num_block_one_dim + num_ne

                    args_align = tvm_ib, data, data_align, output_ub, \
                                 reg, align_data_pos, ub_res_offset
                    _move_align_new(args_align)
            with tvm_ib.if_scope(num_ele_align <= num_block_one_dim):
                align_dst_pos = dst_offset + num_block_one_dim*(num_c + 1)
                args = align_dst_pos, num_ele_block_col, dim_ele, \
                       num_block_one_dim, num_dim_dst_col, c_0
                align_data_pos = _dst_to_data_pos(args)
                ub_res_offset = out_offset + num_c*num_block_one_dim_space \
                                + num_block_one_dim
                args_align = tvm_ib, param, data, data_align, output_ub, reg, \
                             reg_addr[11], align_data_pos, ub_res_offset
                _move_align(args_align)

        with tvm_ib.if_scope(len_burst <= 65535):
            tvm_ib.emit(
                tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                dst.access_ptr('w',
                                               offset=dst_offset +
                                               num_block_one_dim*num_c),
                                output_ub.access_ptr("r",
                                                     offset=out_offset + \
                                            (num_block_one_dim_space*num_c)),
                                0, 1, len_burst, 0, 0))


def _move_ub_to_gm(args):
    """
    move from output_ub to dst(GM) for more dim float16 scene

    """
    tvm_ib, data, dst, data_align, output_ub, reg, reg_addr, param, \
    tail_32b_dim_index, num_g, c_0, c_i, num_dim_dst_col, block_col_dim_ele, \
    num_block_one_dim, num_dim_cur_core = args

    with tvm_ib.if_scope(tail_32b_dim_index < 2):
        tvm_ib.emit(tvm.call_extern(
            output_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[0]),
            output_ub.access_ptr('r',
                                 offset=0)
        ))
        tvm_ib.emit(tvm.call_extern(
            data_align.dtype, "reg_mov",
            data_align.access_ptr('w', offset=0),
            tvm.call_extern(reg.dtype, "reg", reg[0])
        ))

    with tvm_ib.for_range(0, (tail_32b_dim_index + 1), name="num_dc") as num_dc:
        dim_index_before = num_g*param.get("num_dim_one_group") \
            + param.get("block_index")*param.get("num_dim_one_core") + num_dc
        divide = dim_index_before // num_dim_dst_col
        mod = dim_index_before - num_dim_dst_col*divide
        flag_dim = (mod + 1)*c_0
        block_col_index = dim_index_before // num_dim_dst_col
        block_col_mod = dim_index_before - num_dim_dst_col*block_col_index
        block_col_ele = functools_reduce(lambda x, y: x*y, dst.shape[1:])
        dst_before_ele = block_col_index*block_col_ele \
                         + block_col_mod*block_col_dim_ele

        with tvm_ib.if_scope(num_dc < tail_32b_dim_index):
            with tvm_ib.if_scope(flag_dim <= c_i):
                with tvm_ib.for_range(0, c_0, name="num_c0") as num_c0:
                    out_ub_offset = num_dc*_ceil_fill(num_block_one_dim,
                                                      c_0)*c_0 \
                                    + num_c0*_ceil_fill(num_block_one_dim,
                                                        c_0)
                    dst_offset = dst_before_ele + num_c0*num_block_one_dim
                    burst_len = _ceil_div(num_block_one_dim,
                                          param.get("cp_align_len"))
                    with tvm_ib.if_scope(burst_len <= 65535):
                        tvm_ib.emit(
                            tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr('w',
                                                           offset=dst_offset),
                                            output_ub.access_ptr(
                                                "r",
                                                offset=out_ub_offset),
                                            0, 1, burst_len, 0, 0))
            with tvm_ib.if_scope(flag_dim > c_i):
                cycle_last_dim = c_i % c_0
                with tvm_ib.for_range(0, cycle_last_dim,
                                      name="num_c0") as num_c0:
                    out_ub_offset = num_dc*_ceil_fill(num_block_one_dim,
                                                      c_0)*c_0 \
                                    + num_c0*_ceil_fill(num_block_one_dim,
                                                        c_0)
                    dst_offset = dst_before_ele + num_c0*num_block_one_dim
                    burst_len = _ceil_div(num_block_one_dim,
                                          param.get("cp_align_len"))
                    with tvm_ib.if_scope(burst_len <= 65535):
                        tvm_ib.emit(
                            tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr('w',
                                                           offset=dst_offset),
                                            output_ub.access_ptr(
                                                "r",
                                                offset=out_ub_offset),
                                            0, 1, burst_len, 0, 0))
        with tvm_ib.if_scope(num_dc >= tail_32b_dim_index):
            num_dim_align = num_dim_cur_core - tail_32b_dim_index
            reg_addr[9] = num_dim_align
            with tvm_ib.for_range(0, reg_addr[9], name="num_a") as num_a:
                dim_index_cur = dim_index_before + num_a
                divide = dim_index_cur // num_dim_dst_col
                mod = dim_index_cur - num_dim_dst_col*divide
                flag_dim = (mod + 1)*c_0

                block_col_index_cur = dim_index_cur // num_dim_dst_col
                block_col_mod_cur = dim_index_cur - \
                                    num_dim_dst_col*block_col_index_cur
                reg_addr[10] = block_col_mod_cur
                dst_before_ele_cur = block_col_index_cur*block_col_ele \
                                     + reg_addr[10]*block_col_dim_ele

                ub_offset_cur = (num_dc + num_a)*_ceil_fill(num_block_one_dim,
                                                            c_0)*c_0

                with tvm_ib.if_scope(flag_dim <= c_i):
                    num_col_cur = c_0
                    args = tvm_ib, param, data, dst, output_ub, data_align, \
                           reg, reg_addr, num_block_one_dim, num_dim_dst_col, \
                           block_col_ele, block_col_dim_ele, \
                           dst_before_ele_cur, ub_offset_cur, num_col_cur
                    _ub_res_to_gm_more(args)
                with tvm_ib.if_scope(flag_dim > c_i):
                    num_col_cur = c_i % c_0
                    args = tvm_ib, param, data, dst, output_ub, data_align, \
                           reg, reg_addr, num_block_one_dim, num_dim_dst_col, \
                           block_col_ele, block_col_dim_ele, \
                           dst_before_ele_cur, ub_offset_cur, num_col_cur
                    _ub_res_to_gm_more(args)


def _func_more_dim_ir_fp16(args):
    """
    function of moving data for more dim float16 scene

    """
    param, tvm_ib, dst, data, data_align, input_ub, output_ub, \
    addr_array, addr_array_buf, reg, reg_addr, num_g, num_dim_cur_core = args

    c_0 = 16
    _, c_i, h_i, w_i = dst.shape
    src0_offset = 8*0
    src1_offset = 8*1
    dst0_offset = 8*2
    dst1_offset = 8*3
    one_block = 32
    num_block_one_dim = h_i*w_i

    len_burst = num_block_one_dim
    with tvm_ib.if_scope(num_dim_cur_core > 1):
        with tvm_ib.if_scope(len_burst <= 65535):
            dst_stride = _ceil_fill(num_block_one_dim, c_0) - num_block_one_dim
            tvm_ib.emit(
                tvm.call_extern(input_ub.dtype, "copy_gm_to_ubuf",
                                input_ub.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=(num_g*param.get(
                                                    "num_dim_one_group")
                                                        * param.get(
                                                            "src_dim_gm")
                                                        + param.get(
                                                            "block_index")
                                                        * param.get(
                                                            "num_dim_"
                                                            "one_core")
                                                        * param.get(
                                                            "src_dim_gm"))),
                                0, num_dim_cur_core, len_burst, 0, dst_stride))
    with tvm_ib.if_scope(num_dim_cur_core <= 1):
        with tvm_ib.if_scope(len_burst <= 65535):
            tvm_ib.emit(
                tvm.call_extern(input_ub.dtype, "copy_gm_to_ubuf",
                                input_ub.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=(num_g*param.get(
                                                    "num_dim_one_group")
                                                        * param.get(
                                                            "src_dim_gm")
                                                        + param.get(
                                                            "block_index")
                                                        * param.get(
                                                            "num_dim_"
                                                            "one_core")
                                                        * param.get(
                                                            "src_dim_gm"))),
                                0, num_dim_cur_core, len_burst, 0, 0))

    output_offset = param.get("num_dim_one_core")*param.get(
        "src_dim_space")*param.get("float_size")
    with tvm_ib.for_range(0, num_dim_cur_core, name="num_d") as num_d:
        dim_offset = num_d*param.get("src_dim_space")*param.get(
            "float_size")
        src_gap = one_block
        dst_gap = _ceil_div(num_block_one_dim, c_0)*one_block
        src_eight_block = 8*src_gap
        dst_eight_block = 8*dst_gap
        with tvm_ib.for_range(0, 8, name="i") as i:
            tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                        tvm.call_extern(addr_array.dtype, "reg",
                                                        addr_array[
                                                            src0_offset + i]),
                                        dim_offset + i*src_gap))
            tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                        tvm.call_extern(addr_array.dtype, "reg",
                                                        addr_array[
                                                            src1_offset + i]),
                                        dim_offset + src_eight_block
                                        + i*src_gap))
            tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                        tvm.call_extern(addr_array.dtype, "reg",
                                                        addr_array[
                                                            dst0_offset + i]),
                                        output_offset + dim_offset + i*dst_gap))
            tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                        tvm.call_extern(addr_array.dtype, "reg",
                                                        addr_array[
                                                            dst1_offset + i]),
                                        output_offset + dim_offset
                                        + dst_eight_block
                                        + i*dst_gap))

        tvm_ib.emit(tvm.call_extern("int32",
                                    "set_va_reg_sb",
                                    "VA0",
                                    addr_array_buf.access_ptr(
                                        "rw", offset=src0_offset)))
        tvm_ib.emit(tvm.call_extern("int32",
                                    "set_va_reg_sb",
                                    "VA1",
                                    addr_array_buf.access_ptr(
                                        "rw", offset=src1_offset)))
        tvm_ib.emit(tvm.call_extern("int32",
                                    "set_va_reg_sb",
                                    "VA2",
                                    addr_array_buf.access_ptr(
                                        "rw", offset=dst0_offset)))
        tvm_ib.emit(tvm.call_extern("int32",
                                    "set_va_reg_sb",
                                    "VA3",
                                    addr_array_buf.access_ptr(
                                        "rw", offset=dst1_offset)))
        repeat_vconv = _ceil_div(num_block_one_dim, c_0)
        with tvm_ib.if_scope(repeat_vconv > 1):
            src_stride_vconv = 16
            dst_stride_vconv = 1
            tvm_ib.emit(tvm.call_extern("int32",
                                        "scatter_vnchwconv_b16",
                                        "VA2",
                                        "VA0",
                                        repeat_vconv,
                                        dst_stride_vconv,
                                        src_stride_vconv))
        with tvm_ib.if_scope(repeat_vconv <= 1):
            tvm_ib.emit(tvm.call_extern("int32",
                                        "scatter_vnchwconv_b16",
                                        "VA2",
                                        "VA0",
                                        repeat_vconv,
                                        0,
                                        0))

    with tvm_ib.if_scope(num_block_one_dim % c_0 > 0):
        block_col_dim_ele = num_block_one_dim*c_0
        dim_cur_core_before = num_g*param.get("num_dim_one_group") \
                              + param.get("block_index")\
                              * param.get("num_dim_one_core")
        num_dim_dst_col = _ceil_div(c_i, c_0)
        block_col_before_index = dim_cur_core_before // num_dim_dst_col
        block_col_before_mod = dim_cur_core_before - \
                               num_dim_dst_col*block_col_before_index

        num_ele_block_col = functools_reduce(lambda x, y: x*y, dst.shape[1:])

        dim_core_before_ele = block_col_before_index*num_ele_block_col \
                              + block_col_before_mod*num_block_one_dim*c_0

        dim_cur_core_after = num_g*param.get("num_dim_one_group") \
                             + param.get("block_index")\
                             * param.get("num_dim_one_core")\
                             + num_dim_cur_core
        block_col_after_index = dim_cur_core_after // num_dim_dst_col
        block_col_after_mod = dim_cur_core_after - \
                              num_dim_dst_col*block_col_after_index
        dim_core_after_ele = block_col_after_index*num_ele_block_col \
                             + block_col_after_mod*num_block_one_dim*c_0
        num_ele_cur_core_dst = dim_core_after_ele - dim_core_before_ele

        with tvm_ib.if_scope(num_ele_cur_core_dst > param.get("cp_align_len")):
            all_tail_32b_core_dst = dim_core_after_ele - param.get(
                "cp_align_len")
            reg_addr[1] = all_tail_32b_core_dst
            block_col_before = reg_addr[1] // num_ele_block_col
            cur_block_col_ele = reg_addr[1] - \
                                block_col_before*functools_reduce(
                                    lambda x, y: x*y, dst.shape[1:])
            reg_addr[2] = cur_block_col_ele
            num_dim_cur_block_col = reg_addr[2] // (num_block_one_dim*c_0)
            after_dim_num = block_col_before*num_dim_dst_col \
                            + num_dim_cur_block_col
            tail_32b_dim_index = after_dim_num - dim_cur_core_before
            reg_addr[3] = tail_32b_dim_index

            args = tvm_ib, data, dst, data_align, output_ub, reg, reg_addr, \
                   param, reg_addr[3], num_g, c_0, c_i, num_dim_dst_col, \
                   block_col_dim_ele, num_block_one_dim, num_dim_cur_core
            _move_ub_to_gm(args)

        with tvm_ib.if_scope(num_ele_cur_core_dst <= param.get("cp_align_len")):
            tail_32b_dim_index = 0
            args = tvm_ib, data, dst, data_align, output_ub, reg, reg_addr, \
                   param, tail_32b_dim_index, num_g,\
                   c_0, c_i, num_dim_dst_col,\
                   block_col_dim_ele, num_block_one_dim, num_dim_cur_core
            _move_ub_to_gm(args)

    with tvm_ib.else_scope():
        dim_cur_core_before = num_g*param.get("num_dim_one_group") \
                        + param.get("block_index")*param.get("num_dim_one_core")
        num_dim_dst_col = _ceil_div(c_i, c_0)
        block_col_index = dim_cur_core_before // num_dim_dst_col
        block_col_mod = dim_cur_core_before % num_dim_dst_col
        dim_cur_core_before_ele = block_col_index*functools_reduce(
            lambda x, y: x*y,
            dst.shape[1:]) \
                                  + block_col_mod*num_block_one_dim*c_0
        with tvm_ib.if_scope(c_i % c_0 > 0):
            div = dim_cur_core_before // num_dim_dst_col
            cur_col_dim_index = dim_cur_core_before - num_dim_dst_col*div
            reg_addr[4] = cur_col_dim_index
            num_dim_head_block_col = num_dim_dst_col - cur_col_dim_index
            reg_addr[5] = num_dim_head_block_col
            # head dim in current core no full
            with tvm_ib.if_scope(num_dim_head_block_col > num_dim_cur_core):
                burst_len_core = (num_dim_cur_core*num_block_one_dim*c_0
                                  * param.get("float_size")) // 32
                with tvm_ib.if_scope(burst_len_core <= 65535):
                    tvm_ib.emit(
                        tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr(
                                            'w',
                                            offset=dim_cur_core_before_ele),
                                        output_ub.access_ptr("r", offset=0),
                                        0, 1, burst_len_core, 0, 0))

            burst_len_one = functools_reduce(lambda x, y: x*y,
                                             dst.shape[1:]) \
                            - (cur_col_dim_index*num_block_one_dim*c_0)
            reg_addr[6] = burst_len_one
            # head dim in current core full
            with tvm_ib.if_scope(num_dim_head_block_col == num_dim_cur_core):
                burst_len_all_head = reg_addr[6]*param.get("float_size") // 32
                tvm_ib.emit(
                    tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr(
                                        'w',
                                        offset=dim_cur_core_before_ele),
                                    output_ub.access_ptr("r", offset=0),
                                    0, 1, burst_len_all_head, 0, 0))
            # head and tail dim in current core not full
            with tvm_ib.if_scope(tvm.all(\
                (num_dim_head_block_col + num_dim_dst_col) > num_dim_cur_core,
                num_dim_head_block_col < num_dim_cur_core)):
                burst_len_all_head = reg_addr[6]*param.get("float_size") // 32
                tvm_ib.emit(
                    tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr(
                                        'w',
                                        offset=dim_cur_core_before_ele),
                                    output_ub.access_ptr("r", offset=0),
                                    0, 1, burst_len_all_head, 0, 0))
                out_ub_offset = (num_dim_dst_col - reg_addr[4]) \
                                * num_block_one_dim*c_0
                dst_offset = dim_cur_core_before_ele + reg_addr[6]
                tail_one = num_dim_dst_col - cur_col_dim_index
                reg_addr[7] = tail_one
                tail_dim = num_dim_cur_core - tail_one
                reg_addr[8] = tail_dim
                tail_burst_len = reg_addr[8]*num_block_one_dim*c_0*param.get(
                    "float_size") // 32
                tvm_ib.emit(
                    tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    output_ub.access_ptr("r",
                                                         offset=out_ub_offset),
                                    0, 1, tail_burst_len, 0, 0))
            # head and tail dim in current core full
            with tvm_ib.if_scope( \
                (num_dim_head_block_col + num_dim_dst_col) == num_dim_cur_core):
                burst_len_all_head = reg_addr[6]*param.get("float_size") // 32
                tvm_ib.emit(
                    tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr(
                                        'w',
                                        offset=dim_cur_core_before_ele),
                                    output_ub.access_ptr("r", offset=0),
                                    0, 1, burst_len_all_head, 0, 0))
                out_ub_offset = (num_dim_dst_col - reg_addr[4]) \
                                * num_block_one_dim*c_0
                dst_offset = dim_cur_core_before_ele + (
                    functools_reduce(lambda x, y: x*y, dst.shape[1:])
                    - reg_addr[4]*num_block_one_dim*c_0)
                tail_burst_len = functools_reduce(lambda x, y: x*y,
                                                  dst.shape[1:])\
                                 * param.get("float_size") // 32
                with tvm_ib.if_scope(tail_burst_len <= 65535):
                    tvm_ib.emit(
                        tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        output_ub.access_ptr(
                                            "r",
                                            offset=out_ub_offset),
                                        0, 1, tail_burst_len, 0, 0))
            # head, middle and tail dim in current core
            with tvm_ib.if_scope(\
                (num_dim_head_block_col + num_dim_dst_col) < num_dim_cur_core):
                burst_len_all_head = reg_addr[6]*param.get("float_size") // 32
                tvm_ib.emit(
                    tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr(
                                        'w',
                                        offset=dim_cur_core_before_ele),
                                    output_ub.access_ptr("r", offset=0),
                                    0, 1, burst_len_all_head, 0, 0))
                num_remain_dim = num_dim_cur_core - reg_addr[5]
                reg_addr[7] = num_remain_dim
                mid_cycle = reg_addr[7] // num_dim_dst_col
                num_tail_dim = reg_addr[7] % num_dim_dst_col
                out_ub_offset = (num_dim_dst_col - cur_col_dim_index) \
                                * num_block_one_dim*c_0
                dst_offset = dim_cur_core_before_ele + burst_len_one
                with tvm_ib.if_scope(mid_cycle > 0):
                    with tvm_ib.for_range(0, mid_cycle,
                                          name="num_mc") as num_mc:
                        num_ele_dst_col = num_dim_dst_col*num_block_one_dim*c_0
                        cur_out_ub_offset = out_ub_offset \
                                            + num_mc*num_ele_dst_col
                        cur_dst_offset = dst_offset + num_mc*functools_reduce(
                            lambda x, y: x*y, dst.shape[1:])
                        len_burst_mid = functools_reduce(lambda x, y: x*y,
                                                         dst.shape[1:])\
                                        * param.get("float_size") // 32
                        with tvm_ib.if_scope(len_burst_mid <= 65535):
                            tvm_ib.emit(
                                tvm.call_extern(
                                    dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=cur_dst_offset),
                                    output_ub.access_ptr(
                                        "r",
                                        offset=cur_out_ub_offset),
                                    0, 1, len_burst_mid, 0, 0))
                with tvm_ib.if_scope(num_tail_dim > 0):
                    num_ele_dst_col = num_dim_dst_col*num_block_one_dim*c_0
                    tail_out_ub_offset = out_ub_offset + \
                                         mid_cycle*num_ele_dst_col
                    tail_dst_offset = dst_offset + mid_cycle*functools_reduce(
                        lambda x, y: x*y, dst.shape[1:])
                    len_burst_tail = num_tail_dim*num_block_one_dim*c_0 \
                                     * param.get("float_size") // 32
                    with tvm_ib.if_scope(len_burst_tail <= 65535):
                        tvm_ib.emit(
                            tvm.call_extern(
                                dst.dtype, "copy_ubuf_to_gm",
                                dst.access_ptr('w',
                                               offset=tail_dst_offset),
                                output_ub.access_ptr("r",
                                                     offset=tail_out_ub_offset),
                                0, 1, len_burst_tail, 0, 0))

        with tvm_ib.else_scope():
            len_burst_core = num_dim_cur_core*num_block_one_dim*c_0*param.get(
                "float_size") // 32
            with tvm_ib.if_scope(len_burst_core <= 65535):
                tvm_ib.emit(
                    tvm.call_extern(
                        dst.dtype, "copy_ubuf_to_gm",
                        dst.access_ptr('w', offset=dim_cur_core_before_ele),
                        output_ub.access_ptr("r", offset=0),
                        0, 1, len_burst_core, 0, 0))


def _more_dim_ir_fp16(dst, data, max_dim, shape_all):
    """
    function of making ir node builder for more dim float16 scene

    """
    tvm_ib = tvm.ir_builder.create()
    param = _get_param_more_dim_fp16(tvm_ib, data.shape, dst.shape,
                                     dst.dtype, max_dim, shape_all)
    input_ub = _new_alloc(tvm_ib, dst.dtype,
                          param.get('num_dim_one_core')*param.get(
                              "src_dim_space"),
                          "input_ub", scope=cce.scope_ubuf)
    output_ub = _new_alloc(tvm_ib, dst.dtype,
                           param.get('num_dim_one_core')*param.get(
                               "src_dim_space"),
                           "output_ub", scope=cce.scope_ubuf)
    data_align = _new_alloc(tvm_ib, dst.dtype,
                            param.get('cp_align_len')*param.get(
                                "cp_align_len"),
                            "data_align", scope=cce.scope_ubuf)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)
    reg = tvm_ib.allocate(dst.dtype, (1,), name='reg', scope=cce.scope_reg)
    reg_addr = tvm_ib.allocate("int32", (16,), name='reg_addr',
                               scope=cce.scope_reg)

    with tvm_ib.for_range(0, param.get("num_group_index") + 1,
                          name="num_g") as num_g:
        with tvm_ib.if_scope(num_g < param.get("num_group_index")):
            args = param, tvm_ib, dst, data, data_align, input_ub, \
                   output_ub, addr_array, addr_array_buf, \
                   reg, reg_addr, num_g, param.get("num_dim_one_core")
            _func_more_dim_ir_fp16(args)
            _func_pipe(tvm_ib)
        with tvm_ib.if_scope(tvm.all(num_g >= param.get("num_group_index"),
                                     param.get("num_group_mod") > 0)):
            num_core = param.get("num_group_mod") // param.get(
                "num_dim_one_core")
            num_dim_mod = param.get("num_group_mod") - param.get(
                "num_dim_one_core")*num_core
            with tvm_ib.if_scope(num_core > 0):
                with tvm_ib.if_scope(param.get("block_index") < num_core):
                    args = param, tvm_ib, dst, data, data_align, input_ub, \
                           output_ub, addr_array, addr_array_buf, \
                           reg, reg_addr, num_g, param.get("num_dim_one_core")
                    _func_more_dim_ir_fp16(args)
            with tvm_ib.if_scope(num_dim_mod > 0):
                with tvm_ib.if_scope(
                    tvm.all(param.get("block_index") < (num_core + 1),
                            param.get("block_index") > (num_core - 1))):
                    args = param, tvm_ib, dst, data, data_align, input_ub, \
                           output_ub, addr_array, addr_array_buf, \
                           reg, reg_addr, num_g, num_dim_mod
                    _func_more_dim_ir_fp16(args)
            _func_pipe(tvm_ib)

    return tvm_ib.get()


def _get_param_split_dim_fp16(tvm_ib, src_shape, dst_shape,
                              dtype, max_dim, shape_all):
    """
    calculate parameters for split dim float16 ir builder make function

    """
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - (cp_align_len*32)
    device_core_num = AICORE_NUM
    if shape_all > 380000000:
        device_core_num = 1
        ub_bytes = UB_SIZE_B // 2 - (cp_align_len*32)
    elif device_core_num == 32 and (max_dim > 800000 or shape_all > 320000000):
        device_core_num = 16
    ub_half = ub_bytes // 2
    c_0 = 16
    _, _, h_i, w_i = dst_shape
    src_dim_space = _ceil_fill(h_i*w_i, c_0)*c_0
    src_dim_space_bytes = src_dim_space*float_size

    row_bytes = c_0*float_size
    num_row_one_core = (ub_half // row_bytes // c_0)*c_0
    num_row_one_group = num_row_one_core*device_core_num
    num_row_in_data = _ceil_fill(h_i*w_i, c_0) \
                      * functools_reduce(lambda x, y: x*y, src_shape[0:2])
    num_group_index = num_row_in_data // num_row_one_group
    num_group_mod = num_row_in_data - num_row_one_group*num_group_index

    src_dim_gm = h_i*w_i*c_0

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    param_map = {"num_group_index": num_group_index,
                 "num_group_mod": num_group_mod,
                 "src_dim_space": src_dim_space,
                 "src_dim_space_bytes": src_dim_space_bytes,
                 "num_row_one_core": num_row_one_core,
                 "cp_align_len": cp_align_len,
                 "block_index": block_index,
                 "num_row_one_group": num_row_one_group,
                 "src_dim_gm": src_dim_gm, "float_size": float_size}

    return param_map


def _func_vnchw(args):
    """
    function of changing NC1HWC0 to NCHW

    """
    tvm_ib, addr_array, addr_array_buf, \
    num_group, dim_offset, output_offset = args

    src0_offset = 8*0
    src1_offset = 8*1
    dst0_offset = 8*2
    dst1_offset = 8*3
    one_block = 32
    src_gap = one_block
    dst_gap = num_group*one_block
    src_eight_block = 8*src_gap
    dst_eight_block = 8*dst_gap

    with tvm_ib.for_range(0, 8, name="i") as i:
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src0_offset + i]),
                                    dim_offset + i*src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src1_offset + i]),
                                    dim_offset + src_eight_block + i*src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst0_offset + i]),
                                    output_offset + dim_offset + i*dst_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst1_offset + i]),
                                    output_offset + dim_offset
                                    + dst_eight_block + i*dst_gap))

    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA0",
                                addr_array_buf.access_ptr("rw",
                                                          offset=src0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA1",
                                addr_array_buf.access_ptr("rw",
                                                          offset=src1_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA2",
                                addr_array_buf.access_ptr("rw",
                                                          offset=dst0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA3",
                                addr_array_buf.access_ptr("rw",
                                                          offset=dst1_offset)))
    repeat_vconv = num_group
    with tvm_ib.if_scope(repeat_vconv > 1):
        src_stride_vconv = 16
        dst_stride_vconv = 1
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    repeat_vconv,
                                    dst_stride_vconv,
                                    src_stride_vconv))
    with tvm_ib.if_scope(repeat_vconv <= 1):
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    repeat_vconv,
                                    0,
                                    0))


def _vnchwconv(args):
    """
    function of changing NC1HWC0 to NCHW for head, middle and tail in core

    """
    tvm_ib, param, addr_array, addr_array_buf, head_num_group, tail_num_group, \
    mid_num_dim, mid_num_group, output_offset = args
    bytes_one_num_group = 16*16*param.get("float_size")

    with tvm_ib.if_scope(head_num_group > 0):
        dim_offset = 0
        args = tvm_ib, addr_array, addr_array_buf, head_num_group, \
               dim_offset, output_offset
        _func_vnchw(args)
    with tvm_ib.for_range(0, mid_num_dim, name="num_mid") as num_mid:
        dim_offset = (head_num_group + num_mid*mid_num_group) \
                     * bytes_one_num_group
        args = tvm_ib, addr_array, addr_array_buf, mid_num_group, \
               dim_offset, output_offset
        _func_vnchw(args)
    with tvm_ib.if_scope(tail_num_group > 0):
        dim_offset = (head_num_group + mid_num_dim*mid_num_group) \
                     * bytes_one_num_group
        args = tvm_ib, addr_array, addr_array_buf, tail_num_group, \
               dim_offset, output_offset
        _func_vnchw(args)


def _num_row_space_to_true(num_row_space, num_row_one_dim_true,
                           num_row_one_dim_space):
    """
    get num row in dim according to num row space

    """
    num_dim = num_row_space // num_row_one_dim_space
    num_row_mod = num_row_space % num_row_one_dim_space
    num_row_true = num_dim*num_row_one_dim_true + num_row_mod

    return num_row_true


def _ub_res_to_gm_one(args):
    """
    move from output_ub to dst(GM) for 32B align scene

    """
    tvm_ib, param, dst, output_ub, num_row_one_dim_true, dst_offset, \
    num_row_cur_core, num_col = args

    len_burst = num_row_cur_core // param.get("cp_align_len")
    with tvm_ib.if_scope(len_burst <= 65535):
        with tvm_ib.for_range(0, num_col, name="num_c") as num_c:
            out_offset = num_c*num_row_cur_core
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset + \
                                                    num_row_one_dim_true*num_c),
                                        output_ub.access_ptr("r",
                                                             offset=out_offset),
                                        0, 1, len_burst, 0, 0))


def _ub_res_to_gm_two(args):
    """
    move from output_ub to dst(GM) for not 32B align scene

    """
    tvm_ib, param, data, dst, output_ub, data_align, reg, reg_addr, \
    num_row_dim_head_space, num_row_one_dim_true, num_row_one_dim_space, \
    num_ele_block_col, dim_ele, num_dim_one_block_col, dst_offset, \
    out_offset, num_row, dst_offset_origin, num_col = args

    c_0 = 16
    len_burst_one = num_row + param.get("cp_align_len") - 1
    reg_addr[4] = len_burst_one
    len_burst = reg_addr[4] // param.get("cp_align_len")
    tvm_ib.emit(tvm.call_extern(
        output_ub.dtype, "reg_mov",
        tvm.call_extern(reg.dtype, "reg", reg[0]),
        output_ub.access_ptr('r', offset=0)
    ))
    tvm_ib.emit(tvm.call_extern(
        data_align.dtype, "reg_mov",
        data_align.access_ptr('w', offset=0),
        tvm.call_extern(reg.dtype, "reg", reg[0])
    ))
    with tvm_ib.for_range(0, num_col, name="num_c") as num_c:
        with tvm_ib.if_scope(num_row_one_dim_true % c_0 != 0):
            align_dst_pos = dst_offset_origin + num_row_one_dim_true*(num_c + 1)
            args = align_dst_pos, num_ele_block_col, dim_ele, \
                   num_row_one_dim_true, num_dim_one_block_col, c_0
            align_data_pos = _dst_to_data_pos(args)
            num_ele_align = num_row_one_dim_space - num_row_one_dim_true
            ub_res_offset = out_offset + num_c*num_row_dim_head_space \
                            + num_row
            args_align = tvm_ib, param, data, data_align, output_ub, reg, \
                         num_ele_align, align_data_pos, ub_res_offset
            _move_align(args_align)

        with tvm_ib.if_scope(len_burst <= 65535):
            tvm_ib.emit(
                tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                dst.access_ptr('w',
                                               offset=dst_offset +
                                               num_row_one_dim_true*num_c),
                                output_ub.access_ptr("r",
                                                     offset=out_offset + \
                                                num_row_dim_head_space*num_c),
                                0, 1, len_burst, 0, 0))


def _ub_res_to_gm_three(args):
    """
    move from output_ub to dst(GM) for tail in core

    """
    tvm_ib, param, dst, output_ub, num_row_one_dim_true, dst_offset, \
    out_offset, num_row, num_col = args

    len_burst = num_row // param.get("cp_align_len")
    with tvm_ib.if_scope(len_burst <= 65535):
        with tvm_ib.for_range(0, num_col, name="num_c") as num_c:
            tvm_ib.emit(
                tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                dst.access_ptr('w',
                                               offset=dst_offset +
                                               num_row_one_dim_true*num_c),
                                output_ub.access_ptr("r",
                                                     offset=out_offset
                                                     + num_c*num_row),
                                0, 1, len_burst, 0, 0))


def _func_split_dim_ir_fp16(args):
    """
    function of moving data for split dim float16 scene

    """
    param, tvm_ib, dst, data, input_ub, output_ub, \
    addr_array, addr_array_buf, reg, reg_addr, data_align, \
    output_offset, num_g, num_row_cur_core = args

    c_0 = 16
    _, c_i, h_i, w_i = dst.shape
    num_row_one_dim_true = h_i*w_i
    num_row_one_dim_space = _ceil_fill(h_i*w_i, c_0)
    num_dim_one_block_col = data.shape[1]
    num_ele_block_col = functools_reduce(lambda x, y: x*y, dst.shape[1:])
    dim_ele = functools_reduce(lambda x, y: x*y, data.shape[2:])

    num_row_before_core = num_g*param.get("num_row_one_group") \
                        + param.get("block_index")*param.get("num_row_one_core")
    num_dim_before_core = num_row_before_core // num_row_one_dim_space
    num_row_cur_dim_before = num_row_before_core - \
                             num_dim_before_core*num_row_one_dim_space
    num_row_dim_head_true = num_row_one_dim_true - num_row_cur_dim_before
    reg_addr[3] = num_row_dim_head_true
    num_row_dim_head_space = num_row_one_dim_space - num_row_cur_dim_before
    reg_addr[1] = num_row_dim_head_space

    num_row_before_core_ture = _num_row_space_to_true(num_row_before_core,
                                                      num_row_one_dim_true,
                                                      num_row_one_dim_space)

    with tvm_ib.if_scope(num_row_dim_head_space > num_row_cur_core):
        len_burst = num_row_cur_core
        with tvm_ib.if_scope(len_burst <= 65535):
            tvm_ib.emit(
                tvm.call_extern(input_ub.dtype, "copy_gm_to_ubuf",
                                input_ub.access_ptr("w", offset=0),
                                data.access_ptr('r', \
                                        offset=(num_row_before_core_ture*c_0)),
                                0, 1, len_burst, 0, 0))

        num_16group_row = num_row_cur_core // 16
        args = tvm_ib, param, addr_array, addr_array_buf, \
               num_16group_row, 0, 0, 0, output_offset
        _vnchwconv(args)

        dim_index = num_row_before_core_ture // num_row_one_dim_true
        num_block_col = dim_index // num_dim_one_block_col
        cur_dim_index = dim_index - num_dim_one_block_col*num_block_col
        flag_dim = (cur_dim_index + 1)*c_0
        dst_offset = num_block_col*num_ele_block_col \
                     + cur_dim_index*dim_ele \
                     + num_row_cur_dim_before

        with tvm_ib.if_scope(flag_dim <= c_i):
            num_col = c_0
            args = tvm_ib, param, dst, output_ub, num_row_one_dim_true, \
                   dst_offset, num_row_cur_core, num_col
            _ub_res_to_gm_one(args)
        with tvm_ib.if_scope(flag_dim > c_i):
            num_col = c_i % c_0
            args = tvm_ib, param, dst, output_ub, num_row_one_dim_true, \
                   dst_offset, num_row_cur_core, num_col
            _ub_res_to_gm_one(args)

    with tvm_ib.if_scope(num_row_dim_head_space == num_row_cur_core):
        len_burst = num_row_dim_head_true
        with tvm_ib.if_scope(len_burst <= 65535):
            tvm_ib.emit(
                tvm.call_extern(input_ub.dtype, "copy_gm_to_ubuf",
                                input_ub.access_ptr("w", offset=0),
                                data.access_ptr('r', \
                                        offset=(num_row_before_core_ture*c_0)),
                                0, 1, len_burst, 0, 0))

        num_16group_row = reg_addr[1] // 16
        args = tvm_ib, param, addr_array, addr_array_buf, \
               num_16group_row, 0, 0, 0, output_offset
        _vnchwconv(args)

        dim_index = num_row_before_core_ture // num_row_one_dim_true
        num_block_col = dim_index // num_dim_one_block_col
        cur_dim_index = dim_index - num_dim_one_block_col*num_block_col
        flag_dim = (cur_dim_index + 1)*c_0
        dst_offset = num_block_col*num_ele_block_col \
                     + cur_dim_index*dim_ele \
                     + num_row_cur_dim_before

        with tvm_ib.if_scope(num_row_dim_head_true < num_row_dim_head_space):
            dst_offset_origin = num_block_col*num_ele_block_col \
                                + cur_dim_index*dim_ele
            dst_offset_head = dst_offset_origin + num_row_cur_dim_before
            out_offset_head = 0
            with tvm_ib.if_scope(flag_dim <= c_i):
                num_col = c_0
                args = tvm_ib, param, data, dst, output_ub, data_align, reg, \
                       reg_addr, reg_addr[1], num_row_one_dim_true, \
                       num_row_one_dim_space, num_ele_block_col, dim_ele, \
                       num_dim_one_block_col, dst_offset_head,\
                       out_offset_head,\
                       reg_addr[3], dst_offset_origin, num_col
                _ub_res_to_gm_two(args)
            with tvm_ib.if_scope(flag_dim > c_i):
                num_col = c_i % c_0
                args = tvm_ib, param, data, dst, output_ub, data_align, reg, \
                       reg_addr, reg_addr[1], num_row_one_dim_true, \
                       num_row_one_dim_space, num_ele_block_col, dim_ele, \
                       num_dim_one_block_col, dst_offset_head,\
                       out_offset_head,\
                       reg_addr[3], dst_offset_origin, num_col
                _ub_res_to_gm_two(args)
        with tvm_ib.if_scope(num_row_dim_head_true >= num_row_dim_head_space):
            with tvm_ib.if_scope(flag_dim <= c_i):
                num_col = c_0
                args = tvm_ib, param, dst, output_ub, num_row_one_dim_true, \
                       dst_offset, num_row_cur_core, num_col
                _ub_res_to_gm_one(args)
            with tvm_ib.if_scope(flag_dim > c_i):
                num_col = c_i % c_0
                args = tvm_ib, param, dst, output_ub, num_row_one_dim_true, \
                       dst_offset, num_row_cur_core, num_col
                _ub_res_to_gm_one(args)

    with tvm_ib.if_scope(tvm.all(
        num_row_dim_head_space < num_row_cur_core, \
        (num_row_dim_head_space + num_row_one_dim_space) > num_row_cur_core)):
        len_burst_head = num_row_dim_head_true
        with tvm_ib.if_scope(len_burst_head <= 65535):
            tvm_ib.emit(
                tvm.call_extern(input_ub.dtype, "copy_gm_to_ubuf",
                                input_ub.access_ptr("w", offset=0),
                                data.access_ptr('r', \
                                        offset=(num_row_before_core_ture*c_0)),
                                0, 1, len_burst_head, 0, 0))

        num_row_before_tail = (num_dim_before_core + 1)*num_row_one_dim_space
        num_row_before_tail_true = _num_row_space_to_true(num_row_before_tail,
                                                          num_row_one_dim_true,
                                                          num_row_one_dim_space)
        num_row_tail = num_row_cur_core - num_row_dim_head_space
        reg_addr[2] = num_row_tail
        with tvm_ib.if_scope(num_row_tail <= 65535):
            tvm_ib.emit(
                tvm.call_extern(
                    input_ub.dtype, "copy_gm_to_ubuf",
                    input_ub.access_ptr("w", offset=num_row_dim_head_space*c_0),
                    data.access_ptr('r',
                                    offset=(num_row_before_tail_true*c_0)),
                    0, 1, num_row_tail, 0, 0))

        head_num_group = reg_addr[1] // 16
        tail_num_group = reg_addr[2] // 16
        args = tvm_ib, param, addr_array, addr_array_buf, head_num_group, \
               tail_num_group, 0, 0, output_offset
        _vnchwconv(args)

        dim_index = num_row_before_core_ture // num_row_one_dim_true
        num_block_col = dim_index // num_dim_one_block_col
        cur_dim_index = dim_index - num_dim_one_block_col*num_block_col
        flag_dim = (cur_dim_index + 1)*c_0
        dst_offset_origin = num_block_col*num_ele_block_col \
                            + cur_dim_index*dim_ele
        dst_offset_head = dst_offset_origin + num_row_cur_dim_before
        out_offset_head = 0
        with tvm_ib.if_scope(flag_dim <= c_i):
            num_col = c_0
            args = tvm_ib, param, data, dst, output_ub, data_align, reg, \
                   reg_addr, reg_addr[1], num_row_one_dim_true, \
                   num_row_one_dim_space, num_ele_block_col, dim_ele, \
                   num_dim_one_block_col, dst_offset_head, out_offset_head, \
                   reg_addr[3], dst_offset_origin, num_col
            _ub_res_to_gm_two(args)
        with tvm_ib.if_scope(flag_dim > c_i):
            num_col = c_i % c_0
            args = tvm_ib, param, data, dst, output_ub, data_align, reg, \
                   reg_addr, reg_addr[1], num_row_one_dim_true, \
                   num_row_one_dim_space, num_ele_block_col, dim_ele, \
                   num_dim_one_block_col, dst_offset_head, out_offset_head, \
                   reg_addr[3], dst_offset_origin, num_col
            _ub_res_to_gm_two(args)

        dim_index_tail = dim_index + 1
        num_block_col_tail = dim_index_tail // num_dim_one_block_col
        cur_dim_index_tail = dim_index_tail - \
                             num_dim_one_block_col*num_block_col_tail
        flag_dim_tail = (cur_dim_index_tail + 1)*c_0
        dst_offset_tail = num_block_col_tail*num_ele_block_col \
                          + cur_dim_index_tail*dim_ele
        out_offset_tail = reg_addr[1]*c_0
        num_row = num_row_cur_core - num_row_dim_head_space
        reg_addr[5] = num_row
        with tvm_ib.if_scope(flag_dim_tail <= c_i):
            num_col = c_0
            args = tvm_ib, param, dst, output_ub, num_row_one_dim_true, \
                   dst_offset_tail, out_offset_tail, reg_addr[5], num_col
            _ub_res_to_gm_three(args)
        with tvm_ib.if_scope(flag_dim_tail > c_i):
            num_col = c_i % c_0
            args = tvm_ib, param, dst, output_ub, num_row_one_dim_true, \
                   dst_offset_tail, out_offset_tail, reg_addr[5], num_col
            _ub_res_to_gm_three(args)


def _split_dim_ir_fp16(dst, data, max_dim, shape_all):
    """
    function of making ir node builder for split dim float16 scene

    """
    tvm_ib = tvm.ir_builder.create()
    c_0 = 16
    param = _get_param_split_dim_fp16(tvm_ib, data.shape, dst.shape,
                                      dst.dtype, max_dim, shape_all)
    input_ub = _new_alloc(tvm_ib, dst.dtype,
                          param.get('num_row_one_core')*c_0,
                          "input_ub", scope=cce.scope_ubuf)
    output_ub = _new_alloc(tvm_ib, dst.dtype,
                           param.get('num_row_one_core')*c_0,
                           "output_ub", scope=cce.scope_ubuf)
    data_align = _new_alloc(tvm_ib, dst.dtype,
                            param.get('cp_align_len')*param.get(
                                "cp_align_len"),
                            "data_align", scope=cce.scope_ubuf)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)
    reg = tvm_ib.allocate(dst.dtype, (1,), name='reg', scope=cce.scope_reg)
    reg_addr = tvm_ib.allocate("int32", (16,), name='reg_addr',
                               scope=cce.scope_reg)

    output_offset = param.get("num_row_one_core")*c_0*param.get(
        "float_size")
    with tvm_ib.for_range(0, param.get("num_group_index") + 1,
                          name="num_g") as num_g:
        with tvm_ib.if_scope(num_g < param.get("num_group_index")):
            args = param, tvm_ib, dst, data, input_ub, output_ub, addr_array, \
                   addr_array_buf, reg, reg_addr, data_align, output_offset, \
                   num_g, param.get("num_row_one_core")
            _func_split_dim_ir_fp16(args)
            _func_pipe(tvm_ib)
        with tvm_ib.if_scope(tvm.all(num_g >= param.get("num_group_index"),
                                     param.get("num_group_mod") > 0)):
            num_core = param.get("num_group_mod") // param.get(
                "num_row_one_core")
            num_row_mod = param.get("num_group_mod") - param.get(
                "num_row_one_core")*num_core
            with tvm_ib.if_scope(num_core > 0):
                with tvm_ib.if_scope(param.get("block_index") < num_core):
                    args = param, tvm_ib, dst, data, input_ub, output_ub, \
                           addr_array, addr_array_buf, reg, reg_addr, \
                           data_align, output_offset, num_g, \
                           param.get("num_row_one_core")
                    _func_split_dim_ir_fp16(args)
            with tvm_ib.if_scope(num_row_mod > 0):
                with tvm_ib.if_scope(
                    tvm.all(param.get("block_index") < (num_core + 1),
                            param.get("block_index") > (num_core - 1))):
                    args = param, tvm_ib, dst, data, input_ub, output_ub, \
                           addr_array, addr_array_buf, reg, reg_addr, \
                           data_align, output_offset, num_g, num_row_mod
                    _func_split_dim_ir_fp16(args)
            _func_pipe(tvm_ib)

    return tvm_ib.get()


def _get_param_more_row_nhwc(tvm_ib, dst_shape, dtype, max_dim, shape_all):
    """
    calculate parameters for more row nhwc ir builder make function

    """
    c_0 = 16
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = (UB_SIZE_B - 64) // 10*4
    if shape_all > 380000000:
        device_core_num = 1
        ub_bytes = (((UB_SIZE_B // 2) - 64)) // 10*4
    elif device_core_num == 32 and (max_dim > 800000 or shape_all > 320000000):
        device_core_num = 16
    n_i, h_i, w_i, c_i = dst_shape
    c_bytes = _ceil_fill(c_i, c_0)*float_size
    num_row_half_core = ub_bytes // c_bytes
    num_row_one_core = num_row_half_core*2
    num_row_one_group = num_row_one_core*device_core_num
    num_row_in_data = n_i*h_i*w_i
    num_group_index = num_row_in_data // num_row_one_group
    num_group_mod = num_row_in_data - (num_row_one_group*num_group_index)

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    param_map = {"num_group_index": num_group_index,
                 "num_group_mod": num_group_mod,
                 "num_row_one_core": num_row_one_core,
                 "cp_align_len": cp_align_len,
                 "block_index": block_index,
                 "num_row_one_group": num_row_one_group,
                 "float_size": float_size,
                 "num_row_half_core": num_row_half_core}

    return param_map


def _ub_to_dst_tail_nhwc(args):
    """
    move from data_ub to dst(GM) for tail in core

    """
    tvm_ib, param, dst, data_ub, data_tail, reg, c_i, c_align, \
    num_row_before_core, num_row_cur_core, tail_32b_row, tail_row_len = args

    with tvm_ib.for_range(0, tail_32b_row + 1, name="num_t") as num_t:
        with tvm_ib.if_scope(num_t < tail_32b_row):
            ub_offset = num_t*c_align
            dst_offset = (num_row_before_core + num_t)*c_i
            burst_len = _ceil_div(c_i, param.get("cp_align_len"))
            with tvm_ib.if_scope(burst_len <= 65535):
                tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr('w',
                                                           offset=dst_offset),
                                            data_ub.access_ptr(
                                                "r",
                                                offset=ub_offset),
                                            0, 1, burst_len, 0, 0))
        with tvm_ib.if_scope(num_t >= tail_32b_row):
            ub_offset = num_t*c_align
            dst_offset = (num_row_before_core + num_t)*c_i
            burst_len = _ceil_div(tail_row_len, param.get("cp_align_len"))
            with tvm_ib.if_scope(tvm.all(burst_len > 0, burst_len <= 65535)):
                tvm_ib.emit(
                    tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w', offset=dst_offset),
                                    data_ub.access_ptr("r", offset=ub_offset),
                                    0, 1, burst_len, 0, 0))
            with tvm_ib.for_range(0, c_i - tail_row_len, name="num_c") as num_c:
                tvm_ib.emit(tvm.call_extern(
                    data_ub.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_ub.access_ptr('r',
                                       offset=ub_offset + tail_row_len + num_c)
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    data_tail.access_ptr('w', offset=num_c),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))
            num_ele_tail_before = c_i - tail_row_len
            with tvm_ib.if_scope(tail_32b_row + 1 < num_row_cur_core):
                with tvm_ib.for_range(0, num_row_cur_core - tail_32b_row - 1,
                                      name="num_rt") as num_rt:
                    with tvm_ib.for_range(0, c_i, name="num_ct") as num_ct:
                        tvm_ib.emit(tvm.call_extern(
                            data_ub.dtype, "reg_mov",
                            tvm.call_extern(reg.dtype, "reg", reg[0]),
                            data_ub.access_ptr('r',
                                               offset=(num_t + num_rt + 1)
                                               * c_align + num_ct)
                        ))
                        tvm_ib.emit(tvm.call_extern(
                            data_tail.dtype, "reg_mov",
                            data_tail.access_ptr('w',
                                                 offset=num_ele_tail_before
                                                 + num_rt*c_i + num_ct),
                            tvm.call_extern(reg.dtype, "reg", reg[0])
                        ))

            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset
                                                       + tail_row_len),
                                        data_tail.access_ptr("r", offset=0),
                                        0, 1, 1, 0, 0))


def _func_ub_to_res_nhwc(args):
    """
    function of moving data from ub to res for more row nhwc scene

    """
    tvm_ib, param, dst, data_ub, data_tail, reg, c_i, c_0, c_align, \
    block_bytes, num_block_col_before, num_ele_dst_block_col, \
    num_row_cur_block_before, num_row_before_core, num_row_cur_core = args

    with tvm_ib.if_scope(c_i % c_0 > 0):
        with tvm_ib.if_scope(c_i % param.get("cp_align_len") > 0):
            with tvm_ib.if_scope(c_i >= param.get("cp_align_len")):
                tail_32b_row = num_row_cur_core - 1
                tail_row_len = c_i - param.get("cp_align_len")
                args = tvm_ib, param, dst, data_ub, data_tail, reg, c_i, \
                       c_align, num_row_before_core, num_row_cur_core, \
                       tail_32b_row, tail_row_len
                _ub_to_dst_tail_nhwc(args)
            with tvm_ib.if_scope(c_i < param.get("cp_align_len")):
                flag_num_row = _ceil_div(param.get("cp_align_len"), c_i)
                with tvm_ib.if_scope(flag_num_row >= num_row_cur_core):
                    tail_32b_row = 0
                    tail_row_len = _ceil_fill(param.get("cp_align_len"),
                                              c_i) - param.get("cp_align_len")
                    args = tvm_ib, param, dst, data_ub, data_tail, reg, c_i, \
                           c_align, num_row_before_core, num_row_cur_core, \
                           tail_32b_row, tail_row_len
                    _ub_to_dst_tail_nhwc(args)
                with tvm_ib.if_scope(flag_num_row < num_row_cur_core):
                    tail_32b_row = num_row_cur_core - flag_num_row
                    tail_row_len = _ceil_fill(param.get("cp_align_len"),
                                              c_i) - param.get("cp_align_len")
                    args = tvm_ib, param, dst, data_ub, data_tail, reg, c_i, \
                           c_align, num_row_before_core, num_row_cur_core, \
                           tail_32b_row, tail_row_len
                    _ub_to_dst_tail_nhwc(args)
        with tvm_ib.else_scope():
            with tvm_ib.for_range(0, num_row_cur_core, name="num_r") as num_r:
                ub_offset = num_r*c_align
                dst_offset = (num_row_before_core + num_r)*c_i
                burst_len = c_i // param.get("cp_align_len")
                with tvm_ib.if_scope(burst_len > 0):
                    with tvm_ib.if_scope(burst_len <= 65535):
                        tvm_ib.emit(
                            tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr('w',
                                                           offset=dst_offset),
                                            data_ub.access_ptr(
                                                "r",
                                                offset=ub_offset),
                                            0, 1, burst_len, 0, 0))
    with tvm_ib.else_scope():
        dst_offset = num_block_col_before*num_ele_dst_block_col \
                     + num_row_cur_block_before*c_i
        burst_len_dst = num_row_cur_core*c_i*param.get(
            "float_size") // block_bytes
        with tvm_ib.if_scope(burst_len_dst <= 65535):
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_ub.access_ptr("r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))


def _func_move_more_nhwc(args):
    """
    function of moving data for more row nhwc scene

    """
    tvm_ib, param, data, dst, data_ub, data_tail, reg, reg_addr, \
    num_dim_block_col, num_ele_data_block_col, num_ele_dim, c_0, block_bytes, \
    num_ele_dst_block_col, c_i, c_align, num_row_one_dim, \
    num_block_col_before, num_row_cur_block_before, num_row_cur_block, \
    num_row_before_core, num_row_cur_half = args

    with tvm_ib.if_scope(num_row_cur_block >= num_row_cur_half):
        with tvm_ib.for_range(0, num_dim_block_col, name="num_d") as num_d:
            data_offset = num_block_col_before*num_ele_data_block_col \
                          + num_row_cur_block_before*c_0 \
                          + num_d*num_ele_dim
            ub_offset = num_d*c_0
            n_burst_data = num_row_cur_half
            burst_len_data = c_0*param.get("float_size") // block_bytes
            dst_stride = (num_dim_block_col - 1)*burst_len_data
            with tvm_ib.if_scope(burst_len_data <= 65535):
                tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                            data_ub.access_ptr(
                                                "w",
                                                offset=ub_offset),
                                            data.access_ptr('r',
                                                            offset=data_offset),
                                            0, n_burst_data, burst_len_data, 0,
                                            dst_stride))

        args = tvm_ib, param, dst, data_ub, data_tail, reg, c_i, c_0, c_align, \
               block_bytes, num_block_col_before, num_ele_dst_block_col, \
               num_row_cur_block_before, num_row_before_core, num_row_cur_half
        _func_ub_to_res_nhwc(args)
    with tvm_ib.if_scope(num_row_cur_block < num_row_cur_half):
        after_num_row = num_row_cur_half - num_row_cur_block
        reg_addr[1] = after_num_row
        num_mid_block_col = reg_addr[1] // num_row_one_dim
        tail_num_row = after_num_row - num_row_one_dim*num_mid_block_col

        # head : move from gm to ub
        with tvm_ib.if_scope(num_row_cur_block > 0):
            with tvm_ib.for_range(0, num_dim_block_col, name="num_d") as num_d:
                data_offset = num_block_col_before*num_ele_data_block_col \
                              + num_row_cur_block_before*c_0 \
                              + num_d*num_ele_dim
                ub_offset = num_d*c_0
                n_burst_data = num_row_cur_block
                burst_len_data = c_0*param.get("float_size") // block_bytes
                dst_stride = (num_dim_block_col - 1)*burst_len_data

                args = tvm_ib, param, data, data_ub, data_offset, ub_offset,\
                       n_burst_data,\
                       burst_len_data, 0, dst_stride
                _func_gm_to_ub(args)
        # middle : move from gm to ub
        with tvm_ib.if_scope(num_mid_block_col > 0):
            with tvm_ib.for_range(0, num_mid_block_col,
                                  name="num_mb") as num_mb:
                with tvm_ib.for_range(0, num_dim_block_col,
                                      name="num_d") as num_d:
                    data_offset = (num_block_col_before + num_mb + 1) \
                                  * num_ele_data_block_col \
                                  + num_d*num_ele_dim
                    ub_offset = num_row_cur_block*c_align \
                                + num_mb*num_ele_data_block_col \
                                + num_d*c_0
                    n_burst_data = num_row_one_dim
                    burst_len_data = c_0*param.get(
                        "float_size") // block_bytes
                    dst_stride = (num_dim_block_col - 1)*burst_len_data
                    args = tvm_ib, param, data, data_ub, data_offset,\
                           ub_offset, n_burst_data, \
                           burst_len_data, 0, dst_stride
                    _func_gm_to_ub(args)

        # tail : move from gm to ub
        with tvm_ib.if_scope(tail_num_row > 0):
            with tvm_ib.for_range(0, num_dim_block_col, name="num_d") as num_d:
                data_offset = (num_block_col_before + num_mid_block_col + 1) \
                              * num_ele_data_block_col + num_d*num_ele_dim
                ub_offset = num_row_cur_block*c_align \
                            + num_mid_block_col*num_ele_data_block_col \
                            + num_d*c_0
                n_burst_data = tail_num_row
                burst_len_data = c_0*param.get("float_size") // block_bytes
                dst_stride = (num_dim_block_col - 1)*burst_len_data

                args = tvm_ib, param, data, data_ub, data_offset, \
                       ub_offset, n_burst_data, burst_len_data, 0, dst_stride
                _func_gm_to_ub(args)

        args = tvm_ib, param, dst, data_ub, data_tail, reg, c_i, c_0, \
               c_align, block_bytes, num_block_col_before, \
               num_ele_dst_block_col, num_row_cur_block_before, \
               num_row_before_core, num_row_cur_half
        _func_ub_to_res_nhwc(args)


def _func_more_row_nhwc(args):
    """
    function of different situations in more row nhwc scene

    """
    tvm_ib, param, data, dst, data_ub_one, data_ub_two, data_tail_one, \
    data_tail_two, reg_one, reg_two, reg_addr_one, reg_addr_two, \
    c_align, num_g, num_row_cur_core = args

    c_0 = 16
    block_bytes = 32
    _, h_i, w_i, c_i = dst.shape
    num_row_one_dim = h_i*w_i
    num_dim_block_col = data.shape[1]
    num_ele_data_block_col = functools_reduce(lambda x, y: x*y,
                                              data.shape[1:])
    num_ele_dim = functools_reduce(lambda x, y: x*y, data.shape[2:])
    num_ele_dst_block_col = functools_reduce(lambda x, y: x*y, dst.shape[1:])
    num_row_before_core = num_g*param.get("num_row_one_group") \
                        + param.get("block_index")*param.get("num_row_one_core")
    num_block_col_before = num_row_before_core // num_row_one_dim
    num_row_cur_block_before = num_row_before_core - \
                               num_row_one_dim*num_block_col_before
    num_row_cur_block = num_row_one_dim - num_row_cur_block_before

    num_row_before_half = num_row_before_core + param.get("num_row_half_core")
    num_block_col_before_half = num_row_before_half // num_row_one_dim
    num_row_cur_block_before_half = num_row_before_half - \
                                    num_row_one_dim*num_block_col_before_half
    num_row_cur_block_half = num_row_one_dim - num_row_cur_block_before_half

    with tvm_ib.if_scope(num_row_cur_core <= param.get("num_row_half_core")):
        num_row_cur_half = num_row_cur_core
        args = tvm_ib, param, data, dst, data_ub_one, data_tail_one, \
               reg_one, reg_addr_one, num_dim_block_col, \
               num_ele_data_block_col, num_ele_dim, c_0, block_bytes, \
               num_ele_dst_block_col, c_i, c_align, num_row_one_dim, \
               num_block_col_before, num_row_cur_block_before, \
               num_row_cur_block, num_row_before_core, num_row_cur_half
        _func_move_more_nhwc(args)
    with tvm_ib.if_scope(num_row_cur_core > param.get("num_row_half_core")):
        num_row_cur_half_one = param.get("num_row_half_core")
        args = tvm_ib, param, data, dst, data_ub_one, data_tail_one, reg_one, \
               reg_addr_one, num_dim_block_col, num_ele_data_block_col, \
               num_ele_dim, c_0, block_bytes, num_ele_dst_block_col, c_i, \
               c_align, num_row_one_dim, num_block_col_before, \
               num_row_cur_block_before, num_row_cur_block, \
               num_row_before_core, num_row_cur_half_one
        _func_move_more_nhwc(args)

        num_row_cur_half_two = num_row_cur_core - param.get("num_row_half_core")
        args = tvm_ib, param, data, dst, data_ub_two, data_tail_two, reg_two, \
               reg_addr_two, num_dim_block_col, num_ele_data_block_col, \
               num_ele_dim, c_0, block_bytes, num_ele_dst_block_col, c_i, \
               c_align, num_row_one_dim, num_block_col_before_half, \
               num_row_cur_block_before_half, num_row_cur_block_half, \
               num_row_before_half, num_row_cur_half_two
        _func_move_more_nhwc(args)


def _more_row_ir_nhwc(dst, data, max_dim, shape_all):
    """
    function of making ir node builder for more row nhwc scene

    """
    tvm_ib = tvm.ir_builder.create()
    param = _get_param_more_row_nhwc(tvm_ib, dst.shape, dst.dtype,
                                     max_dim, shape_all)
    c_i = dst.shape[3]
    c_0 = 16
    c_align = _ceil_fill(c_i, c_0)
    data_ub_one = _new_alloc(tvm_ib, dst.dtype,
                             param.get('num_row_half_core')*c_align,
                             "data_ub_one", scope=cce.scope_ubuf)
    data_ub_two = _new_alloc(tvm_ib, dst.dtype,
                             param.get('num_row_half_core')*c_align,
                             "data_ub_two", scope=cce.scope_ubuf)
    data_tail_one = _new_alloc(tvm_ib, dst.dtype, param.get('cp_align_len'),
                               "data_tail_one", scope=cce.scope_ubuf)
    data_tail_two = _new_alloc(tvm_ib, dst.dtype, param.get('cp_align_len'),
                               "data_tail_two", scope=cce.scope_ubuf)
    reg_one = tvm_ib.allocate(dst.dtype, (1,), name='reg_one',
                              scope=cce.scope_reg)
    reg_two = tvm_ib.allocate(dst.dtype, (1,), name='reg_two',
                              scope=cce.scope_reg)
    reg_addr_one = tvm_ib.allocate("int32", (8,), name='reg_addr_one',
                                   scope=cce.scope_reg)
    reg_addr_two = tvm_ib.allocate("int32", (8,), name='reg_addr_two',
                                   scope=cce.scope_reg)

    with tvm_ib.for_range(0, param.get("num_group_index") + 1,
                          name="num_g") as num_g:
        with tvm_ib.if_scope(num_g < param.get("num_group_index")):
            args = tvm_ib, param, data, dst, data_ub_one, data_ub_two, \
                   data_tail_one, data_tail_two, reg_one, reg_two, \
                   reg_addr_one, reg_addr_two, c_align, \
                   num_g, param.get("num_row_one_core")
            _func_more_row_nhwc(args)
            _func_pipe(tvm_ib)
        with tvm_ib.if_scope(tvm.all(num_g >= param.get("num_group_index"),
                                     param.get("num_group_mod") > 0)):
            num_core = param.get("num_group_mod") // param.get(
                "num_row_one_core")
            num_row_mod = param.get("num_group_mod") - param.get(
                "num_row_one_core")*num_core
            with tvm_ib.if_scope(num_core > 0):
                with tvm_ib.if_scope(param.get("block_index") < num_core):
                    args = tvm_ib, param, data, dst, data_ub_one, data_ub_two, \
                           data_tail_one, data_tail_two, reg_one, reg_two, \
                           reg_addr_one, reg_addr_two, \
                           c_align, num_g, param.get("num_row_one_core")
                    _func_more_row_nhwc(args)
            with tvm_ib.if_scope(num_row_mod > 0):
                with tvm_ib.if_scope(
                    tvm.all(param.get("block_index") < (num_core + 1),
                            param.get("block_index") > (num_core - 1))):
                    args = tvm_ib, param, data, dst, data_ub_one, data_ub_two, \
                           data_tail_one, data_tail_two, reg_one, reg_two, \
                           reg_addr_one, reg_addr_two, \
                           c_align, num_g, num_row_mod
                    _func_more_row_nhwc(args)
            _func_pipe(tvm_ib)

    return tvm_ib.get()


def _get_param_split_row_nhwc(tvm_ib, src_shape, dtype, max_dim, shape_all):
    """
    calculate parameters for split row nhwc ir builder make function

    """
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = ((UB_SIZE_B - 64)) // 10*4
    if shape_all > 380000000:
        device_core_num = 1
        ub_bytes = (((UB_SIZE_B // 2) - 64)) // 10*4
    elif device_core_num == 32 and (max_dim > 800000 or shape_all > 320000000):
        device_core_num = 16
    c_0 = src_shape[4]
    c_0_bytes = c_0*float_size
    num_c_0_half_core = ub_bytes // c_0_bytes
    num_c_0_one_core = num_c_0_half_core*2
    num_c_0_one_group = num_c_0_one_core*device_core_num
    num_c_0_in_data = functools_reduce(lambda x, y: x*y, src_shape[0:4])
    num_group_index = num_c_0_in_data // num_c_0_one_group
    num_group_mod = num_c_0_in_data - num_c_0_one_group*num_group_index

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    param_map = {"num_group_index": num_group_index,
                 "num_group_mod": num_group_mod,
                 "num_c_0_one_core": num_c_0_one_core,
                 "cp_align_len": cp_align_len,
                 "block_index": block_index,
                 "num_c_0_one_group": num_c_0_one_group,
                 "float_size": float_size,
                 "num_c_0_half_core": num_c_0_half_core}

    return param_map


def _move_one_split(args):
    """
    function of moving data for 32B align scene in split row nhwc function

    """
    tvm_ib, param, data, dst, data_ub, data_offset, dst_offset, ub_offset, \
    c_0, block_bytes, num_c_one_block_col, num_c_0_cur_half = args

    burst_len_data = c_0*param.get("float_size") // block_bytes
    src_stride = (num_c_one_block_col - 1)*burst_len_data
    burst_len_dst = num_c_0_cur_half*c_0*param.get(
        "float_size") // block_bytes

    args = tvm_ib, param, data, data_ub, data_offset, ub_offset,\
           num_c_0_cur_half, burst_len_data, src_stride, 0
    _func_gm_to_ub(args)

    with tvm_ib.if_scope(burst_len_dst <= 65535):
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_ub.access_ptr("r", offset=ub_offset),
                                    0, 1, burst_len_dst, 0, 0))


def _move_two_split(args):
    """
    function of moving data for not 32B align scene in split row nhwc function

    """
    tvm_ib, param, data, dst, data_ub, data_tail, reg, data_offset, \
    dst_offset, ub_offset, c_0, c_i, block_bytes, num_c_one_block_col, \
    num_c_0_cur_half, row_len = args

    burst_len_data = c_0*param.get("float_size") // block_bytes
    src_stride = (num_c_one_block_col - 1)*burst_len_data
    args = tvm_ib, param, data, data_ub, data_offset, ub_offset,\
           num_c_0_cur_half, burst_len_data, src_stride, 0
    _func_gm_to_ub(args)

    with tvm_ib.if_scope(c_i % param.get("cp_align_len") > 0):
        row_len_32b_align = row_len - param.get("cp_align_len")
        burst_len_dst = _ceil_div(row_len_32b_align, param.get("cp_align_len"))
        with tvm_ib.if_scope(burst_len_dst <= 65535):
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_ub.access_ptr("r",
                                                           offset=ub_offset),
                                        0, 1, burst_len_dst, 0, 0))

        with tvm_ib.for_range(0, param.get("cp_align_len"),
                              name="num_c") as num_c:
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_ub.access_ptr('r',
                                   offset=ub_offset + row_len_32b_align + num_c)
            ))
            tvm_ib.emit(tvm.call_extern(
                data_tail.dtype, "reg_mov",
                data_tail.access_ptr('w',
                                     offset=num_c),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))

        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset
                                                   + row_len_32b_align),
                                    data_tail.access_ptr("r", offset=0),
                                    0, 1, 1, 0, 0))

    with tvm_ib.else_scope():
        burst_len_dst = row_len // param.get("cp_align_len")
        with tvm_ib.if_scope(burst_len_dst <= 65535):
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_ub.access_ptr("r",
                                                           offset=ub_offset),
                                        0, 1, burst_len_dst, 0, 0))


def _move_three_split(args):
    """
    function of moving data for head and tail scene in split row nhwc function

    """
    tvm_ib, param, data, dst, data_ub, data_tail, reg, c_0, c_i, block_bytes, \
    num_c_one_block_col, num_c_0_head_row, row_len_head, data_offset_head, \
    dst_offset_head, ub_offset_head, data_offset_tail, dst_offset_tail, \
    ub_offset_tail, num_c_0_tail_row, burst_len_dst_tail = args

    # head part in current core
    burst_len_data = c_0*param.get("float_size") // block_bytes
    src_stride = (num_c_one_block_col - 1)*burst_len_data

    args = tvm_ib, param, data, data_ub, data_offset_head, ub_offset_head, \
           num_c_0_head_row, burst_len_data, src_stride, 0
    _func_gm_to_ub(args)

    with tvm_ib.if_scope(c_i % param.get("cp_align_len") > 0):
        row_len_32b_align_head = row_len_head - param.get("cp_align_len")
        burst_len_dst_head = _ceil_div(row_len_32b_align_head,
                                       param.get("cp_align_len"))
        with tvm_ib.if_scope(burst_len_dst_head <= 65535):
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset_head),
                                        data_ub.access_ptr(
                                            "r",
                                            offset=ub_offset_head),
                                        0, 1, burst_len_dst_head, 0, 0))

        with tvm_ib.for_range(0, param.get("cp_align_len"),
                              name="num_c") as num_c:
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_ub.access_ptr('r',
                                   offset=ub_offset_head +
                                   row_len_32b_align_head + num_c)
            ))
            tvm_ib.emit(tvm.call_extern(
                data_tail.dtype, "reg_mov",
                data_tail.access_ptr('w',
                                     offset=num_c),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))

        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset_head
                                                   + row_len_32b_align_head),
                                    data_tail.access_ptr("r", offset=0),
                                    0, 1, 1, 0, 0))
    with tvm_ib.else_scope():
        burst_len_dst_head = row_len_head // param.get("cp_align_len")
        with tvm_ib.if_scope(burst_len_dst_head <= 65535):
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset_head),
                                        data_ub.access_ptr(
                                            "r",
                                            offset=ub_offset_head),
                                        0, 1, burst_len_dst_head, 0, 0))

    # tail part in current core
    burst_len_data = c_0*param.get("float_size") // block_bytes
    src_stride = (num_c_one_block_col - 1)*burst_len_data

    args = tvm_ib, param, data, data_ub, data_offset_tail, ub_offset_tail, \
           num_c_0_tail_row, burst_len_data, src_stride, 0
    _func_gm_to_ub(args)

    with tvm_ib.if_scope(burst_len_dst_tail <= 65535):
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset_tail),
                                    data_ub.access_ptr("r",
                                                       offset=ub_offset_tail),
                                    0, 1, burst_len_dst_tail, 0, 0))


def _func_half_split(args):
    """
    function of moving data for half core in split row nhwc function

    """
    tvm_ib, param, data, dst, data_ub, data_tail,\
    reg, reg_addr, c_1, c_0, c_i,\
    block_bytes, num_c_one_block_col, num_ele_block_col_data, \
    num_ele_block_col_dst, num_ele_one_dim, num_block_col_before, \
    num_c_before_core, num_c_cur_block_col, num_c_0_cur_row_before, \
    num_c_0_cur_half, num_c_0_head_row, ub_offset = args

    with tvm_ib.if_scope(num_c_0_head_row > num_c_0_cur_half):
        data_offset = num_block_col_before*num_ele_block_col_data \
                      + num_c_cur_block_col*c_0 \
                      + num_c_0_cur_row_before*num_ele_one_dim
        dst_offset = num_block_col_before*num_ele_block_col_dst \
                     + num_c_cur_block_col*c_i \
                     + num_c_0_cur_row_before*c_0
        args = tvm_ib, param, data, dst, data_ub, data_offset, dst_offset, \
               ub_offset, c_0, block_bytes, num_c_one_block_col, \
               num_c_0_cur_half
        _move_one_split(args)
    with tvm_ib.if_scope(num_c_0_head_row == num_c_0_cur_half):
        data_offset = num_block_col_before*num_ele_block_col_data \
                      + num_c_cur_block_col*c_0 \
                      + num_c_0_cur_row_before*num_ele_one_dim
        dst_offset = num_block_col_before*num_ele_block_col_dst \
                     + num_c_cur_block_col*c_i \
                     + num_c_0_cur_row_before*c_0
        row_len = c_i - (num_c_0_cur_row_before*c_0)
        reg_addr[3] = row_len
        args = tvm_ib, param, data, dst, data_ub, data_tail, reg,\
               data_offset, dst_offset, ub_offset, c_0, c_i, block_bytes, \
               num_c_one_block_col, num_c_0_cur_half, reg_addr[3]
        _move_two_split(args)
    with tvm_ib.if_scope(tvm.all(num_c_0_head_row < num_c_0_cur_half,
                                 num_c_0_head_row + c_1 > num_c_0_cur_half)):
        row_len_head = c_i - (num_c_0_cur_row_before * c_0)
        with tvm_ib.if_scope(row_len_head >= param.get("cp_align_len")):
            data_offset_head = num_block_col_before * num_ele_block_col_data \
                               + num_c_cur_block_col * c_0 \
                               + num_c_0_cur_row_before * num_ele_one_dim
            dst_offset_head = num_block_col_before * num_ele_block_col_dst \
                              + num_c_cur_block_col * c_i \
                              + num_c_0_cur_row_before * c_0
            row_len_head = c_i - (num_c_0_cur_row_before * c_0)
            reg_addr[4] = row_len_head

            num_c_0_tail_row = num_c_0_cur_half - num_c_0_head_row
            reg_addr[1] = num_c_0_tail_row
            num_c_before_tail = num_c_before_core + 1
            num_block_col_before_tail = num_c_before_tail\
                                        // num_c_one_block_col
            num_c_cur_block_col_tail = num_c_before_tail - \
                                       num_c_one_block_col \
                                       * num_block_col_before_tail

            data_offset_tail = num_block_col_before_tail \
                               * num_ele_block_col_data \
                               + num_c_cur_block_col_tail * c_0
            ub_offset_tail = ub_offset + num_c_0_head_row * c_0
            dst_offset_tail = num_block_col_before_tail \
                              * num_ele_block_col_dst \
                              + num_c_cur_block_col_tail * c_i
            burst_len_dst_tail = (reg_addr[1] * c_0) // param.get(
                "cp_align_len")
            ub_offset_head = ub_offset

            args = tvm_ib, param, data, dst, data_ub, data_tail,\
                   reg, c_0, c_i, \
                   block_bytes, num_c_one_block_col, num_c_0_head_row, \
                   reg_addr[4], data_offset_head, dst_offset_head,\
                   ub_offset_head, \
                   data_offset_tail, dst_offset_tail, ub_offset_tail, \
                   num_c_0_tail_row, burst_len_dst_tail
            _move_three_split(args)
        with tvm_ib.else_scope():
            num_c_0_cur_row_before = num_c_0_cur_row_before - 1
            data_offset_head = num_block_col_before * num_ele_block_col_data \
                               + num_c_cur_block_col * c_0 \
                               + num_c_0_cur_row_before * num_ele_one_dim
            dst_offset_head = num_block_col_before * num_ele_block_col_dst \
                              + num_c_cur_block_col * c_i \
                              + num_c_0_cur_row_before * c_0
            row_len_head = c_i - (num_c_0_cur_row_before * c_0)
            reg_addr[4] = row_len_head

            num_c_0_tail_row = num_c_0_cur_half - num_c_0_head_row
            reg_addr[1] = num_c_0_tail_row
            num_c_before_tail = num_c_before_core + 1
            num_block_col_before_tail = num_c_before_tail\
                                        // num_c_one_block_col
            num_c_cur_block_col_tail = num_c_before_tail - \
                                       num_c_one_block_col\
                                       * num_block_col_before_tail

            data_offset_tail = num_block_col_before_tail\
                               * num_ele_block_col_data \
                               + num_c_cur_block_col_tail * c_0
            ub_offset_tail = ub_offset + num_c_0_head_row * c_0
            dst_offset_tail = num_block_col_before_tail\
                              * num_ele_block_col_dst \
                              + num_c_cur_block_col_tail * c_i
            burst_len_dst_tail = (reg_addr[1] * c_0) // param.get(
                "cp_align_len")
            ub_offset_head = ub_offset

            args = tvm_ib, param, data, dst, data_ub, data_tail,\
                   reg, c_0, c_i, \
                   block_bytes, num_c_one_block_col, (num_c_0_head_row + 1), \
                   reg_addr[4], data_offset_head,\
                   dst_offset_head, ub_offset_head, \
                   data_offset_tail, dst_offset_tail, ub_offset_tail, \
                   num_c_0_tail_row, burst_len_dst_tail
            _move_three_split(args)


def _func_split_row_nhwc(args):
    """
    function of different moving scenes in split row nhwc function

    """
    tvm_ib, param, data, dst, data_ub_one, data_ub_two, data_tail_one, \
    data_tail_two, reg_one, reg_two, reg_addr_one, reg_addr_two, \
    num_g, num_c_0_cur_core = args

    block_bytes = 32
    _, c_1, h_i, w_i, c_0 = data.shape
    num_c_one_block_col = h_i*w_i
    num_ele_one_dim = h_i*w_i*c_0
    c_i = dst.shape[3]
    num_ele_block_col_data = c_1*num_ele_one_dim
    num_ele_block_col_dst = functools_reduce(lambda x, y: x*y, dst.shape[1:])

    num_c_0_before_core = num_g*param.get("num_c_0_one_group") \
                        + param.get("block_index")*param.get("num_c_0_one_core")
    num_c_before_core = num_c_0_before_core // c_1
    num_c_0_cur_row_before = num_c_0_before_core - num_c_before_core*c_1
    num_c_0_head_row = c_1 - num_c_0_cur_row_before
    num_block_col_before = num_c_before_core // num_c_one_block_col
    num_c_cur_block_col = num_c_before_core - \
                          num_c_one_block_col*num_block_col_before

    num_c_0_before_half = num_c_0_before_core + param.get("num_c_0_half_core")
    num_c_before_half = num_c_0_before_half // c_1
    num_c_0_half_cur_row_before = num_c_0_before_half - num_c_before_half*c_1
    num_c_0_head_row_half = c_1 - num_c_0_half_cur_row_before
    num_block_col_before_half = num_c_before_half // num_c_one_block_col
    num_c_cur_block_col_half = num_c_before_half - \
                               num_c_one_block_col*num_block_col_before_half

    with tvm_ib.if_scope(num_c_0_cur_core <= param.get("num_c_0_half_core")):
        num_c_0_cur_half = num_c_0_cur_core
        ub_offset = 0
        args = tvm_ib, param, data, dst, data_ub_one, data_tail_one, reg_one, \
               reg_addr_one, c_1, c_0, c_i, block_bytes, num_c_one_block_col, \
               num_ele_block_col_data, num_ele_block_col_dst, num_ele_one_dim, \
               num_block_col_before, num_c_before_core, num_c_cur_block_col, \
               num_c_0_cur_row_before, num_c_0_cur_half, \
               num_c_0_head_row, ub_offset
        _func_half_split(args)
    with tvm_ib.if_scope(num_c_0_cur_core > param.get("num_c_0_half_core")):
        num_c_0_cur_half_one = param.get("num_c_0_half_core")
        ub_offset = 0
        args = tvm_ib, param, data, dst, data_ub_one, data_tail_one, reg_one, \
               reg_addr_one, c_1, c_0, c_i, block_bytes, num_c_one_block_col, \
               num_ele_block_col_data, num_ele_block_col_dst, num_ele_one_dim, \
               num_block_col_before, num_c_before_core, num_c_cur_block_col, \
               num_c_0_cur_row_before, num_c_0_cur_half_one, \
               num_c_0_head_row, ub_offset
        _func_half_split(args)

        num_c_0_cur_half_two = num_c_0_cur_core - param.get("num_c_0_half_core")
        reg_addr_two[2] = num_c_0_cur_half_two
        ub_offset = 0
        args = tvm_ib, param, data, dst, data_ub_two, data_tail_two, reg_two, \
               reg_addr_two, c_1, c_0, c_i, block_bytes, num_c_one_block_col, \
               num_ele_block_col_data, num_ele_block_col_dst, num_ele_one_dim, \
               num_block_col_before_half, num_c_before_half, \
               num_c_cur_block_col_half, num_c_0_half_cur_row_before, \
               reg_addr_two[2], num_c_0_head_row_half, ub_offset
        _func_half_split(args)


def _split_row_ir_nhwc(dst, data, max_dim, shape_all):
    """
    function of making ir node builder for split row nhwc scene

    """
    tvm_ib = tvm.ir_builder.create()
    param = _get_param_split_row_nhwc(tvm_ib, data.shape, dst.dtype,
                                      max_dim, shape_all)
    c_0 = data.shape[4]
    data_ub_one = _new_alloc(tvm_ib, dst.dtype,
                             param.get('num_c_0_half_core')*c_0,
                             "data_ub_one", scope=cce.scope_ubuf)
    data_ub_two = _new_alloc(tvm_ib, dst.dtype,
                             param.get('num_c_0_half_core')*c_0,
                             "data_ub_two", scope=cce.scope_ubuf)
    data_tail_one = _new_alloc(tvm_ib, dst.dtype, param.get('cp_align_len'),
                               "data_tail_one", scope=cce.scope_ubuf)
    data_tail_two = _new_alloc(tvm_ib, dst.dtype, param.get('cp_align_len'),
                               "data_tail_two", scope=cce.scope_ubuf)
    reg_one = tvm_ib.allocate(dst.dtype, (1,), name='reg_one',
                              scope=cce.scope_reg)
    reg_two = tvm_ib.allocate(dst.dtype, (1,), name='reg_two',
                              scope=cce.scope_reg)
    reg_addr_one = tvm_ib.allocate("int32", (8,), name='reg_addr_one',
                                   scope=cce.scope_reg)
    reg_addr_two = tvm_ib.allocate("int32", (8,), name='reg_addr_two',
                                   scope=cce.scope_reg)

    with tvm_ib.for_range(0, param.get("num_group_index"),
                          name="num_g") as num_g:
        args = tvm_ib, param, data, dst, data_ub_one, data_ub_two, \
               data_tail_one, data_tail_two, reg_one, reg_two, \
               reg_addr_one, reg_addr_two, num_g, \
               param.get("num_c_0_one_core")
        _func_split_row_nhwc(args)
        _func_pipe(tvm_ib)
    with tvm_ib.if_scope(param.get("num_group_mod") > 0):
        num_core = param.get("num_group_mod") // param.get(
            "num_c_0_one_core")
        num_c_0_mod = param.get("num_group_mod") - param.get(
            "num_c_0_one_core") * num_core
        with tvm_ib.if_scope(num_core > 0):
            with tvm_ib.if_scope(param.get("block_index") < num_core):
                args = tvm_ib, param, data, dst, data_ub_one, data_ub_two, \
                       data_tail_one, data_tail_two, reg_one, reg_two, \
                       reg_addr_one, reg_addr_two,\
                       param.get("num_group_index"),\
                       param.get("num_c_0_one_core")
                _func_split_row_nhwc(args)
        with tvm_ib.if_scope(num_c_0_mod > 0):
            with tvm_ib.if_scope(
                tvm.all(param.get("block_index") < (num_core + 1),
                        param.get("block_index") > (num_core - 1))):
                args = tvm_ib, param, data, dst, data_ub_one, data_ub_two, \
                       data_tail_one, data_tail_two, reg_one, reg_two, \
                       reg_addr_one, reg_addr_two,\
                       param.get("num_group_index"), num_c_0_mod
                _func_split_row_nhwc(args)
        _func_pipe(tvm_ib)

    return tvm_ib.get()


# pylint: disable=locally-disabled,invalid-name,unused-variable
# pylint: disable=locally-disabled,superfluous-parens,too-many-return-statements
def _cal_core(total_core_loop_num, core_number, device_core_num):
    """
    calculate the loop number on each core

    input:
    total_core_loop_num : Virtual computing cores
    core_number : Actual cores
    device_core_num : Physical cores

    return:
    split_block_index : Watershed of different core
    list_out[i][0] : [0:i]  core_loop
    list_out[i+1][0]: [i+1,:] core_loop
    """
    list_out = []
    for block_index in range(core_number):
        core_loop = total_core_loop_num // core_number
        sum_core = ((core_loop + 1) * (total_core_loop_num % device_core_num) +
                    core_loop * (block_index - total_core_loop_num %
                                 device_core_num))
        if block_index < total_core_loop_num % device_core_num:
            core_loop = ((total_core_loop_num + core_number - 1) //
                         core_number)
            sum_core = (core_loop * block_index)

        list_in = [core_loop, sum_core]
        list_out.append(list_in)

    split_core_index = 0
    for i in range(len(list_out)-1):
        if list_out[i][0] != list_out[i+1][0]:
            return i, list_out[i][0], list_out[i+1][0]
        split_core_index += 1

    return split_core_index, list_out[0][0], list_out[0][0]


def _deal_case0(tvm_ib, data_x_ub, data_y_ub, psm, psm_out,
                num_bit, core_loop, sum_core,
                input_data, output_data, base_shape):

    with tvm_ib.for_range(0, core_loop, name="num_core_loop") as num_core_loop:

        # ---------------连续搬运base_shape--------------------#
        src_x_index = psm * (sum_core + num_core_loop)
        nburst_gm2ub = 1
        burstlen_gm2ub = psm * num_bit // 32
        srcstride_gm2ub = 0
        dststride_gm2ub = 0

        tvm_ib.emit(
            tvm.call_extern(data_x_ub.dtype, "copy_gm_to_ubuf",
                            data_x_ub.access_ptr("w", offset=0),
                            input_data.access_ptr('r', offset=src_x_index),
                            0,
                            nburst_gm2ub,
                            burstlen_gm2ub,
                            srcstride_gm2ub,
                            dststride_gm2ub))

        # 第一次b16指令
        # move data, fp32要求至少2次b16指令,当基本处理单元为[16,16,16]
        # 当base_shape[0] = 16 * n时，指令数目为2n，repeat=16
        # 这不是最好的策略，但是当前最可行
        # 高级策略是提升repeat次数

        #------------------------b16_first---------------------#
        # move data, psm : index
        src0_offset = 8*0
        src1_offset = 8*1
        dst0_offset = 8*2
        dst1_offset = 8*3
        # move data, psm : 1B
        src_ub_x_gap = 16*16*4
        dst_ub_y_gap = 32

        # malloc reg space
        addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                     scope=cce.scope_reg)
        addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                         scope=cce.scope_reg, data=addr_array)

        addr_array_2 = tvm_ib.allocate("uint64", (32,), name="addr_array_2",
                                       scope=cce.scope_reg)
        addr_array_buf_2 = tvm.decl_buffer((32,), "uint64_t", \
                    "addr_array_buf_2", scope=cce.scope_reg, data=addr_array_2)

        with tvm_ib.for_range(0, base_shape[0]//16, name='i') as i:
            src_ub_x_index = 16 * 16 * 16 * i
            src_ub_x_index_offset = src_ub_x_index * 4

            dst_ub_y_index = psm + 16 * 16 * 16 * i
            dst_ub_y_index_offset = dst_ub_y_index * 4

            # -------------------第一条b16指令---------------------------#
            with tvm_ib.for_range(0, 8, name="j") as j:
                # 0~7 addr
                # psm: 1B
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array.dtype,
                                                            "reg", \
                                                addr_array[src0_offset + j]), \
                                    src_ub_x_index_offset + j * src_ub_x_gap))

                # 8~15 addr
                # psm: 1B
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array.dtype,
                                                            "reg", \
                                                addr_array[src1_offset + j]), \
                                src_ub_x_index_offset + (j+8) * src_ub_x_gap))

                # 0~7 dst_addr
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array.dtype,
                                                            "reg", \
                                                addr_array[dst0_offset + j]), \
                                    dst_ub_y_index_offset + j * dst_ub_y_gap))

                # 8~15 dst_addr
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array.dtype,
                                                            "reg", \
                                                addr_array[dst1_offset + j]), \
                                dst_ub_y_index_offset + (j+8) * dst_ub_y_gap))


            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA0",
                                        addr_array_buf.access_ptr("rw", \
                                                        offset=src0_offset)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA1",
                                        addr_array_buf.access_ptr("rw", \
                                                        offset=src1_offset)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA2",
                                        addr_array_buf.access_ptr("rw", \
                                                        offset=dst0_offset)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA3",
                                        addr_array_buf.access_ptr("rw", \
                                                        offset=dst1_offset)))

            # repeat 16
            # fp32 src distance between 16 * 4 ~ 64
            # fp32 dst distance between 16 * 16 * 4 ~ 32*32
            repeat = 16
            src_stride = 2
            dst_stride = 32
            tvm_ib.emit(tvm.call_extern("int32",
                                        "scatter_vnchwconv_b16",
                                        "VA2",
                                        "VA0",
                                        repeat,
                                        dst_stride,
                                        src_stride))

            # -------------------第2条b16指令---------------------------#
            # 使用新的寄存器，存储第二条b16的block地址
            # src起始位置与第一条指令相差32B
            # dst起始位置与第一条指令相差16*32B

            with tvm_ib.for_range(0, 8, name="j") as j:
                # 0~7 addr
                # psm: 1B
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_2.dtype,
                                                            "reg", \
                                            addr_array_2[src0_offset + j]), \
                                src_ub_x_index_offset + 32 + j * src_ub_x_gap))

                # 8~15 addr
                # psm: 1B
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_2.dtype,
                                                            "reg", \
                                            addr_array_2[src1_offset + j]), \
                            src_ub_x_index_offset + 32 + (j+8) * src_ub_x_gap))

                # 0~7 dst_addr
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_2.dtype,
                                                            "reg", \
                                            addr_array_2[dst0_offset + j]), \
                            dst_ub_y_index_offset + 16*32 + j * dst_ub_y_gap))

                # 8~15 dst_addr
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_2.dtype,
                                                            "reg", \
                                            addr_array_2[dst1_offset + j]), \
                        dst_ub_y_index_offset + 16*32 + (j+8) * dst_ub_y_gap))


            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA0", \
                        addr_array_buf_2.access_ptr("rw", offset=src0_offset)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA1", \
                        addr_array_buf_2.access_ptr("rw", offset=src1_offset)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA2", \
                        addr_array_buf_2.access_ptr("rw", offset=dst0_offset)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA3", \
                        addr_array_buf_2.access_ptr("rw", offset=dst1_offset)))

            repeat = 16
            src_stride = 2
            dst_stride = 32
            tvm_ib.emit(tvm.call_extern("int32",
                                        "scatter_vnchwconv_b16",
                                        "VA2",
                                        "VA0",
                                        repeat,
                                        dst_stride,
                                        src_stride))

        # -------------------ub_2_ub 消0---------------------------#
        # data_y_ub - > data_x_ub
        # 可以将data_y_ub上的形状看做 8 * 32 * base_shape[0]
        # 等同于 16 * 16 * base_shape[0]
        # 假设有13个0，那么只有32-26=6行是有效的
        zero_suppression = 16 - output_data.shape[-1]

        with tvm_ib.if_scope(zero_suppression >= 0):
            # 跳着取，连续放
            src_ub_y_index = 0
            dst_ub_x_index = 0
            nburst_ub2ub = base_shape[0]
            burstlen_ub2ub = 8 * (output_data.shape[-1]*2) * num_bit // 32
            srcstride_ub2ub = 8 * zero_suppression * 2 * num_bit // 32
            dststride_ub2ub = 0

            tvm_ib.emit(
                tvm.call_extern(data_x_ub.dtype, "copy_ubuf_to_ubuf",
                                data_x_ub.access_ptr("w", offset=dst_ub_x_index),
                                data_y_ub.access_ptr('r', offset=src_ub_y_index),
                                0,
                                nburst_ub2ub,
                                burstlen_ub2ub,
                                srcstride_ub2ub,
                                dststride_ub2ub))


        # -------------------b16_second---------------------------#
        # data_x_ub->data_y_ub
        src0_offset_s = 8*0
        src1_offset_s = 8*1
        dst0_offset_s = 8*2
        dst1_offset_s = 8*3
        # psm : 1B
        src_ub_x_gap_s = 32
        dst_ub_y_gap_s = 32 * output_data.shape[-1] * 2

        # malloc reg space
        addr_array_s = tvm_ib.allocate("uint64", (32,), name="addr_array_s",
                                       scope=cce.scope_reg)
        addr_array_buf_s = tvm.decl_buffer((32,), "uint64_t", \
                    "addr_array_buf_s", scope=cce.scope_reg, data=addr_array_s)

        with tvm_ib.for_range(0, base_shape[0]//16, name='ii') as ii:
            src_ub_x_index_s = 8 * output_data.shape[-1] * 2 * 16 * ii
            src_ub_x_index_offset_s = src_ub_x_index_s * 4

            dst_ub_y_index_s = psm + 8 * output_data.shape[-1] * 2 * 16 * ii
            dst_ub_y_index_offset_s = dst_ub_y_index_s * 4

            # -------------------只使用一条b16指令---------------------------#
            with tvm_ib.for_range(0, 8, name="jj") as jj:
                # 0~7 addr
                # psm: 1B
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_s.dtype,
                                                            "reg", \
                                            addr_array_s[src0_offset_s + jj]), \
                                src_ub_x_index_offset_s + jj * src_ub_x_gap_s))

                # 8~15 addr
                # psm: 1B
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_s.dtype,
                                                            "reg", \
                                            addr_array_s[src1_offset_s + jj]), \
                            src_ub_x_index_offset_s + (jj+8) * src_ub_x_gap_s))

                # 0~7 dst_addr
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_s.dtype,
                                                            "reg", \
                                            addr_array_s[dst0_offset_s + jj]), \
                                dst_ub_y_index_offset_s + jj * dst_ub_y_gap_s))

                # 8~15 dst_addr
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_s.dtype,
                                                            "reg", \
                                            addr_array_s[dst1_offset_s + jj]), \
                            dst_ub_y_index_offset_s + (jj+8) * dst_ub_y_gap_s))


            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA0", \
                    addr_array_buf_s.access_ptr("rw", offset=src0_offset_s)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA1", \
                    addr_array_buf_s.access_ptr("rw", offset=src1_offset_s)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA2", \
                    addr_array_buf_s.access_ptr("rw", offset=dst0_offset_s)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA3", \
                    addr_array_buf_s.access_ptr("rw", offset=dst1_offset_s)))

            repeat_s = output_data.shape[-1] * 2
            src_stride_s = 16
            dst_stride_s = 1
            tvm_ib.emit(tvm.call_extern("int32",
                                        "scatter_vnchwconv_b16",
                                        "VA2",
                                        "VA0",
                                        repeat_s,
                                        dst_stride_s,
                                        src_stride_s))

        # -------------------data_out---------------------------#
        # data_y_ub->gm

        dst_gm = psm_out * (sum_core + num_core_loop)
        nburst_ub2gm = 1
        burstlen_ub2gm = psm_out * num_bit // 32
        srcstride_ub2gm = 0
        dststride_ub2gm = 0

        tvm_ib.emit(
            tvm.call_extern(output_data.dtype, "copy_ubuf_to_gm",
                            output_data.access_ptr("w", offset=dst_gm),
                            data_y_ub.access_ptr('r', offset=0),
                            0,
                            nburst_ub2gm,
                            burstlen_ub2gm,
                            srcstride_ub2gm,
                            dststride_ub2gm))


def _deal_case1(tvm_ib, data_x_ub, data_y_ub, psm, psm_out,
                num_bit, core_loop, sum_core, input_data,
                output_data, tail, num_gm2ub):

    with tvm_ib.for_range(0, core_loop, name="num_core_loop") as num_core_loop:
        with tvm_ib.for_range(0, tail, name="num_ub_loop") as num_ub_loop:

            src_x_index_t = psm * (sum_core + num_core_loop)
            src_x_index_temp_t = src_x_index_t
            src_x_index_t = \
                src_x_index_temp_t + num_ub_loop * num_gm2ub * 16 * 16 * 16
            nburst_gm2ub_t = 1
            burstlen_gm2ub_t = num_gm2ub * 16 * 16 * 16 * num_bit // 32
            srcstride_gm2ub_t = 0
            dststride_gm2ub_t = 0

            tvm_ib.emit(
                tvm.call_extern(data_x_ub.dtype, "copy_gm_to_ubuf",
                                data_x_ub.access_ptr("w", offset=0),
                                input_data.access_ptr('r', offset=src_x_index_t),
                                0,
                                nburst_gm2ub_t,
                                burstlen_gm2ub_t,
                                srcstride_gm2ub_t,
                                dststride_gm2ub_t))

            #------------------------b16_first---------------------#
            # move data, psm : index
            src0_offset = 8*0
            src1_offset = 8*1
            dst0_offset = 8*2
            dst1_offset = 8*3
            # psm : 1B
            src_ub_x_gap = 16*16*4
            dst_ub_y_gap = 32

            # malloc reg space
            addr_array_t = tvm_ib.allocate("uint64", (32,),
                                           name="addr_array_t",
                                           scope=cce.scope_reg)
            addr_array_buf_t = tvm.decl_buffer((32,), "uint64_t",
                                               "addr_array_buf_t",
                                               scope=cce.scope_reg,
                                               data=addr_array_t)

            addr_array_2_t = tvm_ib.allocate("uint64", (32,),
                                             name="addr_array_2_t",
                                             scope=cce.scope_reg)
            addr_array_buf_2_t = tvm.decl_buffer((32,), "uint64_t",
                                                 "addr_array_buf_2_t",
                                                 scope=cce.scope_reg,
                                                 data=addr_array_2_t)

            with tvm_ib.for_range(0, num_gm2ub, name='i') as i:
                src_ub_x_index_t = 16 * 16 * 16 * i
                src_ub_x_index_offset_t = src_ub_x_index_t * 4

                dst_ub_y_index_t = 16 * num_gm2ub * 16 * 16 + 16 * 16 * 16 * i
                dst_ub_y_index_offset_t = dst_ub_y_index_t * 4

                # -------------------第一条b16指令---------------------------#
                with tvm_ib.for_range(0, 8, name="j") as j:
                    # 0~7 addr
                    # psm: 1B
                    tvm_ib.emit(tvm.call_extern("uint64", "reg_mov", \
                                tvm.call_extern(addr_array_t.dtype, "reg", \
                                            addr_array_t[src0_offset + j]), \
                                    src_ub_x_index_offset_t + j * src_ub_x_gap))

                    # 8~15 addr
                    # psm: 1B
                    tvm_ib.emit(tvm.call_extern("uint64", "reg_mov", \
                                tvm.call_extern(addr_array_t.dtype, "reg", \
                                            addr_array_t[src1_offset + j]), \
                                src_ub_x_index_offset_t + (j+8) * src_ub_x_gap))

                    # 0~7 dst_addr
                    tvm_ib.emit(tvm.call_extern("uint64", "reg_mov", \
                                tvm.call_extern(addr_array_t.dtype, "reg", \
                                            addr_array_t[dst0_offset + j]), \
                                    dst_ub_y_index_offset_t + j * dst_ub_y_gap))

                    # 8~15 dst_addr
                    tvm_ib.emit(tvm.call_extern("uint64", "reg_mov", \
                                tvm.call_extern(addr_array_t.dtype, "reg", \
                                            addr_array_t[dst1_offset + j]), \
                                dst_ub_y_index_offset_t + (j+8) * dst_ub_y_gap))


                tvm_ib.emit(tvm.call_extern("int32",
                                            "set_va_reg_sb",
                                            "VA0", \
                        addr_array_buf_t.access_ptr("rw", offset=src0_offset)))

                tvm_ib.emit(tvm.call_extern("int32",
                                            "set_va_reg_sb",
                                            "VA1", \
                        addr_array_buf_t.access_ptr("rw", offset=src1_offset)))

                tvm_ib.emit(tvm.call_extern("int32",
                                            "set_va_reg_sb",
                                            "VA2", \
                        addr_array_buf_t.access_ptr("rw", offset=dst0_offset)))

                tvm_ib.emit(tvm.call_extern("int32",
                                            "set_va_reg_sb",
                                            "VA3", \
                        addr_array_buf_t.access_ptr("rw", offset=dst1_offset)))

                repeat = 16
                src_stride = 2
                dst_stride = 32
                tvm_ib.emit(tvm.call_extern("int32",
                                            "scatter_vnchwconv_b16",
                                            "VA2",
                                            "VA0",
                                            repeat,
                                            dst_stride,
                                            src_stride))

                # -------------------第2条b16指令---------------------------#
                # 使用新的寄存器，存储第二条b16的block地址
                # src起始位置与第一条指令相差32B
                # dst起始位置与第一条指令相差16*32B

                with tvm_ib.for_range(0, 8, name="j") as j:
                    # 0~7 addr
                    # psm: 1B
                    tvm_ib.emit(
                        tvm.call_extern("uint64", "reg_mov", \
                                    tvm.call_extern(addr_array_2_t.dtype, \
                                            "reg", \
                                            addr_array_2_t[src0_offset + j]), \
                            src_ub_x_index_offset_t + 32 + j * src_ub_x_gap))

                    # 8~15 addr
                    # psm: 1B
                    tvm_ib.emit(
                        tvm.call_extern("uint64", "reg_mov", \
                                tvm.call_extern(addr_array_2_t.dtype, "reg",\
                                            addr_array_2_t[src1_offset + j]), \
                        src_ub_x_index_offset_t + 32 + (j+8) * src_ub_x_gap))

                    # 0~7 dst_addr
                    tvm_ib.emit(
                        tvm.call_extern("uint64", "reg_mov",
                                        tvm.call_extern(addr_array_2_t.dtype,
                                                        "reg", \
                                        addr_array_2_t[dst0_offset + j]), \
                            dst_ub_y_index_offset_t + 16*32 + j * dst_ub_y_gap))

                    # 8~15 dst_addr
                    tvm_ib.emit(
                        tvm.call_extern("uint64", "reg_mov",
                                        tvm.call_extern(addr_array_2_t.dtype,
                                                        "reg", \
                                        addr_array_2_t[dst1_offset + j]), \
                        dst_ub_y_index_offset_t + 16*32 + (j+8) * dst_ub_y_gap))


                tvm_ib.emit(
                    tvm.call_extern("int32",
                                    "set_va_reg_sb",
                                    "VA0", \
                    addr_array_buf_2_t.access_ptr("rw", offset=src0_offset)))

                tvm_ib.emit(
                    tvm.call_extern("int32",
                                    "set_va_reg_sb",
                                    "VA1", \
                    addr_array_buf_2_t.access_ptr("rw", offset=src1_offset)))

                tvm_ib.emit(
                    tvm.call_extern("int32",
                                    "set_va_reg_sb",
                                    "VA2", \
                    addr_array_buf_2_t.access_ptr("rw", offset=dst0_offset)))

                tvm_ib.emit(
                    tvm.call_extern("int32",
                                    "set_va_reg_sb",
                                    "VA3", \
                    addr_array_buf_2_t.access_ptr("rw", offset=dst1_offset)))

                repeat = 16
                src_stride = 2
                dst_stride = 32
                tvm_ib.emit(tvm.call_extern("int32",
                                            "scatter_vnchwconv_b16",
                                            "VA2",
                                            "VA0",
                                            repeat,
                                            dst_stride,
                                            src_stride))

            # -------------------ub_2_ub 消0---------------------------#
            # data_y_ub
            zero_suppression = 16 - output_data.shape[-1]

            with tvm_ib.if_scope(zero_suppression >= 0):
                # 跳着取，连续放
                src_ub_y_index_t = 0
                dst_ub_x_index_t = 0
                nburst_ub2ub_t = num_gm2ub * 16
                burstlen_ub2ub_t = 8 * (output_data.shape[-1]*2) * num_bit // 32
                srcstride_ub2ub_t = 8 * zero_suppression * 2 * num_bit // 32
                dststride_ub2ub_t = 0

                tvm_ib.emit(
                    tvm.call_extern(data_x_ub.dtype, "copy_ubuf_to_ubuf", \
                            data_x_ub.access_ptr("w", offset=dst_ub_x_index_t),\
                            data_y_ub.access_ptr('r', offset=src_ub_y_index_t),\
                                    0,
                                    nburst_ub2ub_t,
                                    burstlen_ub2ub_t,
                                    srcstride_ub2ub_t,
                                    dststride_ub2ub_t))


            # -------------------b16_second---------------------------#
            # data_x_ub->data_y_ub
            # move data, psm : index
            src0_offset_s = 8*0
            src1_offset_s = 8*1
            dst0_offset_s = 8*2
            dst1_offset_s = 8*3
            # psm : 1B
            src_ub_x_gap_s = 32
            dst_ub_y_gap_s = 32 * output_data.shape[-1] * 2

            # malloc reg space
            addr_array_s_t = tvm_ib.allocate("uint64", (32,),
                                             name="addr_array_s_t",
                                             scope=cce.scope_reg)
            addr_array_buf_s_t = tvm.decl_buffer((32,), "uint64_t",
                                                 "addr_array_buf_s_t",
                                                 scope=cce.scope_reg,
                                                 data=addr_array_s_t)

            with tvm_ib.for_range(0, num_gm2ub, name='ii') as ii:
                src_ub_x_index_s_t = 8 * output_data.shape[-1] * 2 * 16 * ii
                src_ub_x_index_offset_s_t = src_ub_x_index_s_t * 4

                dst_ub_y_index_s_t = 16 * num_gm2ub * 16 * 16 + \
                                     8 * output_data.shape[-1] * 2 * 16 * ii
                dst_ub_y_index_offset_s_t = dst_ub_y_index_s_t * 4

                # -------------------只使用一条b16指令---------------------------#
                with tvm_ib.for_range(0, 8, name="jj") as jj:
                    # 0~7 addr
                    # psm: 1B
                    tvm_ib.emit(
                        tvm.call_extern("uint64", "reg_mov",
                                        tvm.call_extern(addr_array_s_t.dtype,
                                                        "reg", \
                                        addr_array_s_t[src0_offset_s + jj]), \
                            src_ub_x_index_offset_s_t + jj * src_ub_x_gap_s))

                    # 8~15 addr
                    # psm: 1B
                    tvm_ib.emit(
                        tvm.call_extern("uint64", "reg_mov",
                                        tvm.call_extern(addr_array_s_t.dtype,
                                                        "reg", \
                                        addr_array_s_t[src1_offset_s + jj]), \
                        src_ub_x_index_offset_s_t + (jj+8) * src_ub_x_gap_s))

                    # 0~7 dst_addr
                    tvm_ib.emit(
                        tvm.call_extern("uint64", "reg_mov",
                                        tvm.call_extern(addr_array_s_t.dtype,
                                                        "reg", \
                                        addr_array_s_t[dst0_offset_s + jj]), \
                        dst_ub_y_index_offset_s_t + jj * dst_ub_y_gap_s))

                    # 8~15 dst_addr
                    tvm_ib.emit(
                        tvm.call_extern("uint64", "reg_mov",
                                        tvm.call_extern(addr_array_s_t.dtype,
                                                        "reg", \
                                        addr_array_s_t[dst1_offset_s + jj]), \
                    dst_ub_y_index_offset_s_t + (jj+8) * dst_ub_y_gap_s))

                tvm_ib.emit(
                    tvm.call_extern("int32",
                                    "set_va_reg_sb",
                                    "VA0", \
                    addr_array_buf_s_t.access_ptr("rw", offset=src0_offset_s)))

                tvm_ib.emit(
                    tvm.call_extern("int32",
                                    "set_va_reg_sb",
                                    "VA1", \
                    addr_array_buf_s_t.access_ptr("rw", offset=src1_offset_s)))

                tvm_ib.emit(
                    tvm.call_extern("int32",
                                    "set_va_reg_sb",
                                    "VA2", \
                    addr_array_buf_s_t.access_ptr("rw", offset=dst0_offset_s)))

                tvm_ib.emit(
                    tvm.call_extern("int32",
                                    "set_va_reg_sb",
                                    "VA3", \
                    addr_array_buf_s_t.access_ptr("rw", offset=dst1_offset_s)))

                repeat_s = output_data.shape[-1] * 2
                src_stride_s = 16
                dst_stride_s = 1
                tvm_ib.emit(tvm.call_extern("int32",
                                            "scatter_vnchwconv_b16",
                                            "VA2",
                                            "VA0",
                                            repeat_s,
                                            dst_stride_s,
                                            src_stride_s))

            #-------------------data_out---------------------------#
            # data_y_ub->gm
            dst_gm_t = psm_out * (sum_core + num_core_loop) + \
                       num_ub_loop * num_gm2ub * 16 * 16 * output_data.shape[-1]
            nburst_ub2gm_t = 1
            burstlen_ub2gm_t = \
                num_gm2ub * 16 * 16 * output_data.shape[-1] * num_bit // 32

            srcstride_ub2gm_t = 0
            dststride_ub2gm_t = 0

            tvm_ib.emit(
                tvm.call_extern(output_data.dtype, "copy_ubuf_to_gm",
                                output_data.access_ptr("w", offset=dst_gm_t),
                                data_y_ub.access_ptr('r', offset=0),
                                0,
                                nburst_ub2gm_t,
                                burstlen_ub2gm_t,
                                srcstride_ub2gm_t,
                                dststride_ub2gm_t))


def _deal_case2(tvm_ib, data_x_ub, data_y_ub, psm, psm_out,
                num_bit, core_loop, sum_core, input_data,
                output_data, tail, tail_block, num_gm2ub):

    with tvm_ib.for_range(0, core_loop, name="num_core_loop") as num_core_loop:

        src_x_index = psm * (sum_core + num_core_loop)
        src_x_index_temp = src_x_index
        src_x_index = src_x_index_temp + tail * num_gm2ub * 16 * 16 * 16
        nburst_gm2ub = 1
        burstlen_gm2ub = tail_block * 16 * 16 * 16 * num_bit // 32
        srcstride_gm2ub = 0
        dststride_gm2ub = 0

        tvm_ib.emit(
            tvm.call_extern(data_x_ub.dtype, "copy_gm_to_ubuf",
                            data_x_ub.access_ptr("w", offset=0),
                            input_data.access_ptr('r', offset=src_x_index),
                            0,
                            nburst_gm2ub,
                            burstlen_gm2ub,
                            srcstride_gm2ub,
                            dststride_gm2ub))

        #------------------------b16_first---------------------#
        # move data, psm : index
        src0_offset = 8*0
        src1_offset = 8*1
        dst0_offset = 8*2
        dst1_offset = 8*3
        # psm : 1B
        src_ub_x_gap = 16*16*4
        dst_ub_y_gap = 32

        # malloc reg space
        addr_array = tvm_ib.allocate("uint64", (32,),
                                     name="addr_array", scope=cce.scope_reg)
        addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                         scope=cce.scope_reg, data=addr_array)

        addr_array_2 = tvm_ib.allocate("uint64", (32,),
                                       name="addr_array_2", scope=cce.scope_reg)
        addr_array_buf_2 = tvm.decl_buffer((32,), "uint64_t",
                                           "addr_array_buf_2",
                                           scope=cce.scope_reg,
                                           data=addr_array_2)

        with tvm_ib.for_range(0, tail_block, name='i') as i:
            src_ub_x_index = 16 * 16 * 16 * i
            src_ub_x_index_offset = src_ub_x_index * 4

            dst_ub_y_index = 16 * num_gm2ub * 16 * 16 + 16 * 16 * 16 * i
            dst_ub_y_index_offset = dst_ub_y_index * 4

            # -------------------第一条b16指令---------------------------#
            with tvm_ib.for_range(0, 8, name="j") as j:
                # 0~7 addr
                # psm: 1B
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array.dtype,
                                                            "reg", \
                                                addr_array[src0_offset + j]), \
                                    src_ub_x_index_offset + j * src_ub_x_gap))

                # 8~15 addr
                # psm: 1B
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array.dtype,
                                                            "reg", \
                                                addr_array[src1_offset + j]), \
                                src_ub_x_index_offset + (j+8) * src_ub_x_gap))

                # 0~7 dst_addr
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array.dtype,
                                                            "reg", \
                                                addr_array[dst0_offset + j]), \
                                    dst_ub_y_index_offset + j * dst_ub_y_gap))

                # 8~15 dst_addr
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array.dtype,
                                                            "reg", \
                                                addr_array[dst1_offset + j]), \
                                dst_ub_y_index_offset + (j+8) * dst_ub_y_gap))


            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA0", \
                        addr_array_buf.access_ptr("rw", offset=src0_offset)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA1", \
                        addr_array_buf.access_ptr("rw", offset=src1_offset)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA2", \
                        addr_array_buf.access_ptr("rw", offset=dst0_offset)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA3", \
                        addr_array_buf.access_ptr("rw", offset=dst1_offset)))

            repeat = 16
            src_stride = 2
            dst_stride = 32
            tvm_ib.emit(tvm.call_extern("int32",
                                        "scatter_vnchwconv_b16",
                                        "VA2",
                                        "VA0",
                                        repeat,
                                        dst_stride,
                                        src_stride))

            # -------------------第2条b16指令---------------------------#
            # 使用新的寄存器，存储第二条b16的block地址
            # src起始位置与第一条指令相差32B
            # dst起始位置与第一条指令相差16*32B

            with tvm_ib.for_range(0, 8, name="j") as j:
                # 0~7 addr
                # psm: 1B
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_2.dtype,
                                                            "reg", \
                                            addr_array_2[src0_offset + j]), \
                                src_ub_x_index_offset + 32 + j * src_ub_x_gap))

                # 8~15 addr
                # psm: 1B
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_2.dtype,
                                                            "reg", \
                                            addr_array_2[src1_offset + j]), \
                            src_ub_x_index_offset + 32 + (j+8) * src_ub_x_gap))

                # 0~7 dst_addr
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_2.dtype,
                                                            "reg", \
                                            addr_array_2[dst0_offset + j]), \
                            dst_ub_y_index_offset + 16*32 + j * dst_ub_y_gap))

                # 8~15 dst_addr
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_2.dtype,
                                                            "reg", \
                                            addr_array_2[dst1_offset + j]), \
                        dst_ub_y_index_offset + 16*32 + (j+8) * dst_ub_y_gap))


            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA0", \
                        addr_array_buf_2.access_ptr("rw", offset=src0_offset)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA1", \
                        addr_array_buf_2.access_ptr("rw", offset=src1_offset)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA2", \
                        addr_array_buf_2.access_ptr("rw", offset=dst0_offset)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA3", \
                        addr_array_buf_2.access_ptr("rw", offset=dst1_offset)))

            repeat = 16
            src_stride = 2
            dst_stride = 32
            tvm_ib.emit(tvm.call_extern("int32",
                                        "scatter_vnchwconv_b16",
                                        "VA2",
                                        "VA0",
                                        repeat,
                                        dst_stride,
                                        src_stride))


        # -------------------ub_2_ub 消0---------------------------#
        # data_y_ub
        zero_suppression = 16 - output_data.shape[-1]

        with tvm_ib.if_scope(zero_suppression >= 0):
            # 跳着取，连续放
            src_ub_y_index = 0
            dst_ub_x_index = 0
            nburst_ub2ub = tail_block * 16
            burstlen_ub2ub = 8 * (output_data.shape[-1]*2) * num_bit // 32
            srcstride_ub2ub = 8 * zero_suppression * 2 * num_bit // 32
            dststride_ub2ub = 0

            tvm_ib.emit(
                tvm.call_extern(data_x_ub.dtype, "copy_ubuf_to_ubuf",
                                data_x_ub.access_ptr("w", offset=dst_ub_x_index),
                                data_y_ub.access_ptr('r', offset=src_ub_y_index),
                                0,
                                nburst_ub2ub,
                                burstlen_ub2ub,
                                srcstride_ub2ub,
                                dststride_ub2ub))


        # -------------------b16_second---------------------------#
        # move data, psm : index
        src0_offset_s = 8*0
        src1_offset_s = 8*1
        dst0_offset_s = 8*2
        dst1_offset_s = 8*3
        # psm : 1B
        src_ub_x_gap_s = 32
        dst_ub_y_gap_s = 32 * output_data.shape[-1] * 2

        # malloc reg space
        addr_array_s = tvm_ib.allocate("uint64", (32,),
                                       name="addr_array_s", scope=cce.scope_reg)
        addr_array_buf_s = tvm.decl_buffer((32,), "uint64_t",
                                           "addr_array_buf_s",
                                           scope=cce.scope_reg,
                                           data=addr_array_s)

        with tvm_ib.for_range(0, tail_block, name='ii') as ii:
            src_ub_x_index_s = 8 * output_data.shape[-1] * 2 * 16 * ii
            src_ub_x_index_offset_s = src_ub_x_index_s * 4

            dst_ub_y_index_s = 16 * num_gm2ub * 16 * 16 + 8 * \
                               output_data.shape[-1] * 2 * 16 * ii
            dst_ub_y_index_offset_s = dst_ub_y_index_s * 4

            # -------------------只使用一条b16指令---------------------------#
            with tvm_ib.for_range(0, 8, name="jj") as jj:
                # 0~7 addr
                # psm: 1B
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_s.dtype,
                                                            "reg", \
                                            addr_array_s[src0_offset_s + jj]), \
                                src_ub_x_index_offset_s + jj * src_ub_x_gap_s))

                # 8~15 addr
                # psm: 1B
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_s.dtype,
                                                            "reg", \
                                            addr_array_s[src1_offset_s + jj]), \
                            src_ub_x_index_offset_s + (jj+8) * src_ub_x_gap_s))

                # 0~7 dst_addr
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_s.dtype,
                                                            "reg", \
                                            addr_array_s[dst0_offset_s + jj]), \
                                dst_ub_y_index_offset_s + jj * dst_ub_y_gap_s))

                # 8~15 dst_addr
                tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                            tvm.call_extern(addr_array_s.dtype,
                                                            "reg", \
                                            addr_array_s[dst1_offset_s + jj]), \
                            dst_ub_y_index_offset_s + (jj+8) * dst_ub_y_gap_s))


            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA0", \
                    addr_array_buf_s.access_ptr("rw", offset=src0_offset_s)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA1", \
                    addr_array_buf_s.access_ptr("rw", offset=src1_offset_s)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA2", \
                    addr_array_buf_s.access_ptr("rw", offset=dst0_offset_s)))

            tvm_ib.emit(tvm.call_extern("int32",
                                        "set_va_reg_sb",
                                        "VA3", \
                    addr_array_buf_s.access_ptr("rw", offset=dst1_offset_s)))

            repeat_s = output_data.shape[-1] * 2
            src_stride_s = 16
            dst_stride_s = 1
            tvm_ib.emit(tvm.call_extern("int32",
                                        "scatter_vnchwconv_b16",
                                        "VA2",
                                        "VA0",
                                        repeat_s,
                                        dst_stride_s,
                                        src_stride_s))

        # -------------------data_out---------------------------#
        # data_y_ub->gm
        dst_gm = psm_out * (sum_core + num_core_loop)
        dst_gm_temp = dst_gm
        dst_gm = \
            dst_gm_temp + tail * num_gm2ub * 16 * 16 * output_data.shape[-1]
        nburst_ub2gm = 1
        burstlen_ub2gm = \
            int(tail_block * 16 * 16 * output_data.shape[-1] * num_bit // 32)
        srcstride_ub2gm = 0
        dststride_ub2gm = 0

        tvm_ib.emit(
            tvm.call_extern(output_data.dtype, "copy_ubuf_to_gm",
                            output_data.access_ptr("w", offset=dst_gm),
                            data_y_ub.access_ptr('r', offset=0),
                            0,
                            nburst_ub2gm,
                            burstlen_ub2gm,
                            srcstride_ub2gm,
                            dststride_ub2gm))


def _less_ub_ir_nhwc(output_data, input_data, base_shape, core_num):

    tvm_ib = tvm.ir_builder.create()

    # malloc space for ub_x ub_y
    # move data, psm : 单核处理的数据
    # move data, num_bit : float_size
    # move data, data_x_ub [0,psm-1]
    # data_y_ub [psm,2psm]
    data_x_ub = _new_alloc(tvm_ib, input_data.dtype, base_shape,
                           'data_x_ub', scope=cce.scope_ubuf)
    data_y_ub = _new_alloc(tvm_ib, input_data.dtype, base_shape,
                           'data_y_ub', scope=cce.scope_ubuf)
    psm = functools_reduce(lambda x, y: x * y, base_shape[:])
    base_shape_out = base_shape.copy()
    base_shape_out[-1] = output_data.shape[-1]
    psm_out = functools_reduce(lambda x, y: x * y, base_shape_out[:])
    num_bit = 4

    device_core_num = AICORE_NUM
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    # total_num_loop  实际处理次数
    # core_number     maximum core_nums
    # block_index     当前第几个核
    total_core_loop_num = core_num
    if total_core_loop_num >= device_core_num:
        core_number = device_core_num
    else:
        core_number = total_core_loop_num

    split_core_index,\
    core_loop_before,\
    core_loop_after = _cal_core(total_core_loop_num,
                                core_number, device_core_num)

    if core_number < device_core_num:
        with tvm_ib.if_scope(block_index <= core_number-1):
            core_loop = core_loop_before
            sum_core = core_loop_before * block_index
            _deal_case0(tvm_ib, data_x_ub, data_y_ub,
                        psm, psm_out, num_bit,
                        core_loop, sum_core, input_data,
                        output_data, base_shape)
    else:
        if core_loop_before == core_loop_after:
            core_loop = core_loop_before
            sum_core = core_loop * block_index
            _deal_case0(tvm_ib, data_x_ub, data_y_ub,
                        psm, psm_out, num_bit, core_loop,
                        sum_core, input_data, output_data, base_shape)

        else:
            with tvm_ib.if_scope(block_index <= split_core_index):
                core_loop = core_loop_before
                sum_core = core_loop_before * block_index
                _deal_case0(tvm_ib, data_x_ub, data_y_ub,
                            psm, psm_out, num_bit,
                            core_loop, sum_core, input_data,
                            output_data, base_shape)

            with tvm_ib.if_scope(block_index > split_core_index):
                core_loop = core_loop_after
                sum_core = (block_index - split_core_index - 1) * \
                           core_loop + core_loop_before * (split_core_index + 1)
                _deal_case0(tvm_ib, data_x_ub, data_y_ub,
                            psm, psm_out, num_bit, core_loop,
                            sum_core, input_data, output_data, base_shape)

    return tvm_ib.get()


def _more_ub_ir_nhwc(output_data, input_data, base_shape, core_num):
    """
    base_shape too bigger to save in ub
    """
    # in this situation,num_gm2ub < num_gm
    # move data, tail >= 1
    tvm_ib = tvm.ir_builder.create()
    num_bit = 4
    ub_maximum_size = UB_SIZE_B // 2
    ub_maximum_size = ub_maximum_size // num_bit
    num_gm2ub = ub_maximum_size // (16*16*16)
    num_gm = base_shape[0] // 16
    tail = num_gm // num_gm2ub
    tail_block = num_gm % num_gm2ub

    # malloc space for ub_x ub_y
    # move data, psm : 单核处理的数据
    # move data, data_x_ub [0,psm-1]
    # data_y_ub [psm,2psm]
    data_x_ub = _new_alloc(tvm_ib, input_data.dtype, [16 * num_gm2ub, 16, 16],
                           'data_x_ub', scope=cce.scope_ubuf)
    data_y_ub = _new_alloc(tvm_ib, input_data.dtype, [16 * num_gm2ub, 16, 16],
                           'data_y_ub', scope=cce.scope_ubuf)
    psm = functools_reduce(lambda x, y: x * y, base_shape[:])
    base_shape_out = base_shape.copy()
    base_shape_out[-1] = output_data.shape[-1]
    psm_out = functools_reduce(lambda x, y: x * y, base_shape_out[:])

    device_core_num = AICORE_NUM
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    # total_num_loop  实际处理次数
    # core_number     maximum core_nums
    # block_index     当前第几个核
    total_core_loop_num = core_num
    if total_core_loop_num >= device_core_num:
        core_number = device_core_num
    else:
        core_number = total_core_loop_num

    split_core_index,\
    core_loop_before,\
    core_loop_after = _cal_core(total_core_loop_num,
                                core_number, device_core_num)

    if core_number < device_core_num:
        with tvm_ib.if_scope(block_index <= core_number-1):
            core_loop = core_loop_before
            sum_core = core_loop * block_index
            _deal_case1(tvm_ib, data_x_ub, data_y_ub, psm,
                        psm_out, num_bit, core_loop, sum_core,
                        input_data, output_data, tail, num_gm2ub)
            if tail_block != 0:
                _deal_case2(tvm_ib, data_x_ub, data_y_ub, psm, psm_out,
                            num_bit, core_loop, sum_core, input_data,
                            output_data, tail, tail_block, num_gm2ub)
    else:
        if core_loop_before == core_loop_after:
            core_loop = core_loop_before
            sum_core = core_loop * block_index
            _deal_case1(tvm_ib, data_x_ub, data_y_ub, psm,
                        psm_out, num_bit, core_loop, sum_core,
                        input_data, output_data, tail, num_gm2ub)
            if tail_block != 0:
                _deal_case2(tvm_ib, data_x_ub, data_y_ub, psm, psm_out,
                            num_bit, core_loop, sum_core, input_data,
                            output_data, tail, tail_block, num_gm2ub)

        else:
            with tvm_ib.if_scope(block_index <= split_core_index):
                core_loop = core_loop_before
                sum_core = core_loop_before * block_index
                _deal_case1(tvm_ib, data_x_ub, data_y_ub, psm, psm_out,
                            num_bit, core_loop, sum_core,
                            input_data, output_data, tail, num_gm2ub)
                if tail_block != 0:
                    _deal_case2(tvm_ib, data_x_ub, data_y_ub, psm, psm_out,
                                num_bit, core_loop, sum_core, input_data,
                                output_data, tail, tail_block, num_gm2ub)

            with tvm_ib.if_scope(block_index > split_core_index):
                core_loop = core_loop_after
                sum_core = (block_index - split_core_index - 1) * core_loop + \
                           core_loop_before * (split_core_index + 1)
                _deal_case1(tvm_ib, data_x_ub, data_y_ub, psm, psm_out,
                            num_bit, core_loop, sum_core,
                            input_data, output_data, tail, num_gm2ub)
                if tail_block != 0:
                    _deal_case2(tvm_ib, data_x_ub, data_y_ub, psm, psm_out,
                                num_bit, core_loop, sum_core, input_data,
                                output_data, tail, tail_block, num_gm2ub)

    return tvm_ib.get()


def _get_ir_branch_sp_nhwc(src_shape, dtype):
    n = src_shape[0]
    h = src_shape[-3]
    w = src_shape[-2]
    hw = h * w
    split_hw = hw // 256
    split_hw_temp = split_hw
    total_num_loop = int(split_hw * src_shape[0])
    base_shape = [16, 16, 16]
    core_num = total_num_loop
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    ub_bytes = UB_SIZE_B // 2
    mark = ""
    # in this situation
    if hw % 256 == 0 and src_shape[1] == 1 and dtype == "float32":
        if total_num_loop > 32:
            if src_shape[0] < 32:
                while(split_hw % 2 == 0  and total_num_loop > 32):
                    split_hw = split_hw / 2
                    total_num_loop = total_num_loop / 2

                core_num = int(total_num_loop)
                base_shape[0] = int(base_shape[0] * (split_hw_temp/split_hw))
            else:
                if n % 32 == 0:
                    core_num = 32
                    base_shape[0] = int(base_shape[0] * (n // 32 * split_hw))
                else:
                    core_num = n
                    base_shape[0] = int(base_shape[0] * split_hw)

        psm = functools_reduce(lambda x, y: x * y, base_shape[:]) * float_size

        if psm > ub_bytes:
            mark = "more_ub"
        else:
            mark = "less_ub"

    else:
        mark = "inconformity"

    return mark, core_num, base_shape


def _get_param_nhwc_adds_fp16(tvm_ib, dst_shape, dtype, c_0):
    """
    calculate parameters for nhwc adds fp16 ir builder make function

    """
    device_core_num = AICORE_NUM
    ub_bytes = ((UB_SIZE_B - 32) // 2)
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_ele = ub_bytes // float_size
    n_i, h_i, w_i, _ = dst_shape
    num_group_index = n_i // device_core_num
    num_group_mod = n_i % device_core_num
    dim_ele = h_i*w_i*c_0
    num_dim_one_core = ((ub_ele // dim_ele) // 8)*8

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    param_map = {"num_group_index": num_group_index,
                 "num_group_mod": num_group_mod,
                 "float_size": float_size,
                 "cp_align_len": cp_align_len,
                 "block_index": block_index,
                 "ub_ele": ub_ele, "device_core_num": device_core_num,
                 "num_dim_one_core": num_dim_one_core}

    return param_map


def _func_vadds(args):
    """
    function of moving data with vadds function

    """
    tvm_ib, data_ub, data_res, c_0, ub_offset,\
    res_offset, repeat, srcm0, dstm0, srcm1, dstm1 = args
    max_r = 255

    with tvm_ib.if_scope(repeat <= max_r):
        with tvm_ib.if_scope(repeat == 1):
            tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                        data_res.access_ptr("w",
                                                            offset=res_offset),
                                        data_ub.access_ptr('r',
                                                           offset=ub_offset),
                                        0, repeat, dstm0, srcm0, 0, 0))
        with tvm_ib.else_scope():
            tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                        data_res.access_ptr("w",
                                                            offset=res_offset),
                                        data_ub.access_ptr('r',
                                                           offset=ub_offset),
                                        0, repeat, dstm0, srcm0, dstm1, srcm1))
    with tvm_ib.else_scope():
        zu_repeat = repeat // max_r
        mod_repeat = repeat % max_r
        with tvm_ib.for_range(0, zu_repeat, name="num_zr") as num_zr:
            ub_offset_cur = ub_offset + num_zr*max_r*c_0
            res_offset_cur = res_offset + num_zr*max_r*c_0*dstm1
            tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                        data_res.access_ptr("w", \
                                                        offset=res_offset_cur),
                                        data_ub.access_ptr('r', \
                                                        offset=ub_offset_cur),
                                        0, max_r, dstm0, srcm0, dstm1, srcm1))
        with tvm_ib.if_scope(mod_repeat > 0):
            ub_offset_cur = ub_offset + zu_repeat*max_r*c_0
            res_offset_cur = res_offset + zu_repeat*max_r*c_0*dstm1
            with tvm_ib.if_scope(mod_repeat == 1):
                tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                            data_res.access_ptr("w", \
                                                        offset=res_offset_cur),
                                            data_ub.access_ptr('r', \
                                                        offset=ub_offset_cur),
                                            0, mod_repeat, dstm0, srcm0, 0, 0))
            with tvm_ib.else_scope():
                tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                            data_res.access_ptr("w", \
                                                        offset=res_offset_cur),
                                            data_ub.access_ptr('r', \
                                                        offset=ub_offset_cur),
                                            0, mod_repeat, dstm0, srcm0, dstm1,
                                            srcm1))


def _func_nhwc_adds_fp16(args):
    """
    function of moving data for nhwc adds fp16 scene

    """
    tvm_ib, param, data, dst, data_ub, data_res, reg, reg_addr, num_g = args

    c_1 = data.shape[1]
    c_0 = data.shape[4]
    _, h_i, w_i, c_i = dst.shape
    dim_ele = h_i*w_i*c_0
    num_zu = c_1 // param.get("num_dim_one_core")
    dim_mod = c_1 % param.get("num_dim_one_core")
    group_ele = h_i*w_i*c_i*param.get("device_core_num")

    zu_ele = param.get("num_dim_one_core")*dim_ele
    burst_len_data = zu_ele // param.get("cp_align_len")
    with tvm_ib.for_range(0, num_zu, name="num_z") as num_z:
        data_offset = num_g*group_ele \
                      + param.get("block_index")*h_i*w_i*c_i + num_z*zu_ele
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr("w", offset=0),
                                    data.access_ptr('r', offset=data_offset),
                                    0, 1, burst_len_data, 0, 0))

        eight_group = param.get("num_dim_one_core") // 8
        with tvm_ib.for_range(0, eight_group, name="num_eg") as num_eg:
            ub_offset = num_eg*dim_ele*8
            res_offset = num_eg*c_0*8
            repeat = h_i*w_i
            srcm0 = h_i*w_i
            dstm0 = 1
            srcm1 = 1
            dstm1 = param.get("num_dim_one_core")
            args = tvm_ib, data_ub, data_res, c_0, ub_offset, res_offset,\
                   repeat, srcm0, dstm0, srcm1, dstm1
            _func_vadds(args)

        dst_offset = num_g*group_ele + param.get("block_index")*h_i*w_i*c_i\
                     + num_z*param.get("num_dim_one_core")*c_0
        n_burst = h_i*w_i
        burst_len = param.get("num_dim_one_core")
        src_stride = 0
        dst_stride = c_1 - param.get("num_dim_one_core")
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr("r", offset=0),
                                    0, n_burst, burst_len, src_stride,
                                    dst_stride))

    with tvm_ib.if_scope(dim_mod > 0):

        data_offset = num_g*group_ele \
                      + param.get("block_index")*h_i*w_i*c_i + num_zu*zu_ele
        burst_len_data = dim_mod*h_i*w_i
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr("w", offset=0),
                                    data.access_ptr('r',
                                                    offset=data_offset),
                                    0, 1, burst_len_data, 0, 0))

        zu_mod = dim_mod // 8
        with tvm_ib.for_range(0, zu_mod, name="num_zm") as num_zm:
            ub_offset = num_zm*dim_ele*8
            res_offset = num_zm*c_0*8
            repeat = h_i*w_i
            srcm0 = h_i*w_i
            dstm0 = 1
            srcm1 = 1
            dstm1 = dim_mod
            args = tvm_ib, data_ub, data_res, c_0, ub_offset, res_offset,\
                   repeat, srcm0, dstm0, srcm1, dstm1
            _func_vadds(args)

        dst_offset = num_g*group_ele + param.get("block_index")*h_i*w_i*c_i\
                     + num_zu*param.get("num_dim_one_core")*c_0
        n_burst = h_i*w_i
        burst_len = dim_mod
        src_stride = 0
        dst_stride = c_1 - dim_mod
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr("r", offset=0),
                                    0, n_burst, burst_len, src_stride,
                                    dst_stride))


def _nhwc_adds_fp16_ir(dst, data):
    """
    function of making ir node builder for nhwc adds fp16 scene

    """
    tvm_ib = tvm.ir_builder.create()
    c_0 = data.shape[4]
    param = _get_param_nhwc_adds_fp16(tvm_ib, dst.shape, dst.dtype, c_0)

    data_ub = _new_alloc(tvm_ib, dst.dtype,
                         param.get("ub_ele"),
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype,
                          param.get("ub_ele"),
                          "data_res", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    reg_addr = tvm_ib.allocate("int32", (8,), name='reg_addr',
                               scope=cce.scope_reg)

    with tvm_ib.for_range(0, param.get("num_group_index") + 1,
                          name="num_g") as num_g:
        with tvm_ib.if_scope(num_g < param.get("num_group_index")):
            args = tvm_ib, param, data, dst, data_ub,\
                   data_res, reg, reg_addr, num_g
            _func_nhwc_adds_fp16(args)
        with tvm_ib.if_scope(tvm.all(num_g >= param.get("num_group_index"),
                                     param.get("num_group_mod") > 0)):
            num_core = param.get("num_group_mod")
            with tvm_ib.if_scope(num_core > 0):
                with tvm_ib.if_scope(param.get("block_index") < num_core):
                    args = tvm_ib, param, data, dst, data_ub,\
                           data_res, reg, reg_addr, num_g
                    _func_nhwc_adds_fp16(args)

    return tvm_ib.get()


def _func_ci_align_nhwc_fp16_store(args):
    """
    function of moving data for ci align nhwc float16 scene

    """
    tvm_ib, data, dst, data_ub, data_res, device_core_num, block_index,\
    h_i, w_i, c_i, c_1, c_0, col_len_shape, col_len_ub, cp_align_len,\
    ub_loop, ub_mod, num_g = args

    n_index = num_g * device_core_num + block_index

    with tvm_ib.for_range(0, ub_loop, name="num_u") as num_u:
        data_offset = n_index * h_i * w_i * c_i + num_u * col_len_ub * c_0
        ub_offset = 0
        ori_nburst = c_1
        burst_len = col_len_ub
        src_stride = col_len_shape - col_len_ub
        dst_stride = 0
        args = tvm_ib, data, data_ub, data_offset, ub_offset, ori_nburst, \
               burst_len, src_stride, dst_stride, cp_align_len
        _func_gm_to_ub_align(args)

        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            ub_offset = num_c1 * col_len_ub * c_0
            res_offset = num_c1 * c_0
            nburst_ub = col_len_ub
            burst_len_ub = 1
            src_stride_ub = 0
            dst_stride_ub = c_1 - 1
            tvm_ib.emit(tvm.call_extern(data_res.dtype,
                                        "copy_ubuf_to_ubuf",
                                        data_res.access_ptr(
                                            "w", offset=res_offset),
                                        data_ub.access_ptr(
                                            'r', offset=ub_offset),
                                        0, nburst_ub, burst_len_ub,
                                        src_stride_ub, dst_stride_ub))

        dst_offset = n_index * h_i * w_i * c_i + num_u * col_len_ub * c_i
        burst_len_dst = col_len_ub * c_1
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr("r",
                                                        offset=0),
                                    0, 1, burst_len_dst, 0, 0))
    with tvm_ib.if_scope(ub_mod > 0):
        data_offset = n_index * h_i * w_i * c_i + ub_loop * col_len_ub * c_0
        ub_offset = 0
        ori_nburst = c_1
        burst_len = ub_mod
        src_stride = col_len_shape - ub_mod
        dst_stride = 0
        args = tvm_ib, data, data_ub, data_offset, ub_offset, ori_nburst, \
               burst_len, src_stride, dst_stride, cp_align_len
        _func_gm_to_ub_align(args)

        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            ub_offset = num_c1 * ub_mod * c_0
            res_offset = num_c1 * c_0
            nburst_ub = ub_mod
            burst_len_ub = 1
            src_stride_ub = 0
            dst_stride_ub = c_1 - 1
            tvm_ib.emit(tvm.call_extern(data_res.dtype,
                                        "copy_ubuf_to_ubuf",
                                        data_res.access_ptr(
                                            "w", offset=res_offset),
                                        data_ub.access_ptr(
                                            'r', offset=ub_offset),
                                        0, nburst_ub, burst_len_ub,
                                        src_stride_ub, dst_stride_ub))

        dst_offset = n_index * h_i * w_i * c_i + ub_loop * col_len_ub * c_i
        burst_len_dst = ub_mod * c_1
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr("r",
                                                        offset=0),
                                    0, 1, burst_len_dst, 0, 0))


def _func_vadds_for_move(args):
    """
    function of moving data with vadds function

    """
    tvm_ib, data_ub, data_res, c_0, c_1, ub_offset, res_offset, repeat,\
    srcm0, dstm0, srcm1, dstm1, cp_align_len = args
    max_r = 255

    with tvm_ib.if_scope(repeat <= max_r):
        with tvm_ib.if_scope(repeat == 1):
            tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                        data_res.access_ptr("w",
                                                            offset=res_offset),
                                        data_ub.access_ptr('r',
                                                           offset=ub_offset),
                                        0, repeat, dstm0, srcm0, 0, 0))
        with tvm_ib.else_scope():
            tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                        data_res.access_ptr("w",
                                                            offset=res_offset),
                                        data_ub.access_ptr('r',
                                                           offset=ub_offset),
                                        0, repeat, dstm0, srcm0, dstm1, srcm1))
    with tvm_ib.else_scope():
        zu_repeat = repeat // max_r
        mod_repeat = repeat % max_r
        with tvm_ib.for_range(0, zu_repeat, name="num_zr") as num_zr:
            ub_offset_cur = ub_offset + num_zr*max_r*8*cp_align_len
            res_offset_cur = res_offset + num_zr*max_r*8*c_1*cp_align_len
            tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                        data_res.access_ptr(
                                            "w", offset=res_offset_cur),
                                        data_ub.access_ptr(
                                            "r", offset=ub_offset_cur),
                                        0, max_r, dstm0, srcm0, dstm1, srcm1))
        with tvm_ib.if_scope(mod_repeat > 0):
            ub_offset_cur = ub_offset + zu_repeat*max_r*8*cp_align_len
            res_offset_cur = res_offset + zu_repeat*max_r*8*c_1*cp_align_len
            with tvm_ib.if_scope(mod_repeat == 1):
                tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                            data_res.access_ptr(
                                                "w", offset=res_offset_cur),
                                            data_ub.access_ptr(
                                                "r", offset=ub_offset_cur),
                                            0, mod_repeat, dstm0, srcm0, 0, 0))
            with tvm_ib.else_scope():
                tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                            data_res.access_ptr(
                                                "w", offset=res_offset_cur),
                                            data_ub.access_ptr(
                                                "r", offset=ub_offset_cur),
                                            0, mod_repeat, dstm0, srcm0, dstm1,
                                            srcm1))


def _set_mask_slice_fp16(before, after):
    """
    calculate MASK in cce

    """
    before = int(before)
    after = int(after)

    if after >= 64:
        mask1 = 0
        after_new = after - 64
        mask2 = (2**(64 - after_new) - 1) - (2**before - 1)
    elif before >= 64:
        before_new = before - 64
        mask1 = (2**(64 - after) - 1) - (2**before_new - 1)
        mask2 = 0
    else:
        mask1 = (2**(64 - after) - 1)
        mask2 = (2**64 - 1) - (2**before - 1)

    return mask1, mask2


def _func_ci_align_nhwc_fp16(args):
    """
    function of moving data for ci align nhwc float16 scene

    """
    tvm_ib, data, dst, data_ub, data_res, device_core_num, block_index,\
    h_i, w_i, c_i, c_1, c_0, col_len_shape, col_len_ub, cp_align_len,\
    ub_loop, ub_mod, num_g = args

    n_index = num_g * device_core_num + block_index

    with tvm_ib.for_range(0, ub_loop, name="num_u") as num_u:
        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            data_offset = n_index * h_i * w_i * c_i\
                          + num_c1 * col_len_shape * c_0\
                          + num_u * col_len_ub * c_0
            ub_offset = num_c1 * col_len_ub * c_0
            ori_nburst = 1
            burst_len = col_len_ub
            src_stride = 0
            dst_stride = 0
            args = tvm_ib, data, data_ub, data_offset, ub_offset, ori_nburst,\
                   burst_len, src_stride, dst_stride, cp_align_len
            _func_gm_to_ub_align(args)

        vadds_zu = col_len_ub // 8
        vadds_mod = col_len_ub % 8
        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            with tvm_ib.if_scope(vadds_zu > 0):
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(-1, dtype="uint64"),
                    tvm.const(-1, dtype="uint64")))

                ub_offset = num_c1 * col_len_ub * c_0
                res_offset = num_c1 * c_0
                repeat = vadds_zu
                srcm0 = 1
                dstm0 = c_1
                srcm1 = 8
                dstm1 = 8 * c_1
                args = tvm_ib, data_ub, data_res, c_0, c_1, ub_offset,\
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1,\
                       cp_align_len
                _func_vadds_for_move(args)
            with tvm_ib.if_scope(vadds_mod > 0):
                ub_offset = num_c1 * col_len_ub * c_0 + vadds_zu * 8 * c_0
                res_offset = num_c1 * c_0 + vadds_zu * 8 * c_i
                repeat = 1
                srcm0 = 1
                dstm0 = c_1
                srcm1 = 0
                dstm1 = 0
                before = 0
                after = (8 - vadds_mod) * 16
                mask1, mask2 = _set_mask_slice_fp16(before, after)
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                args = tvm_ib, data_ub, data_res, c_0, c_1, ub_offset, \
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1, \
                       cp_align_len
                _func_vadds_for_move(args)

        tvm_ib.emit(tvm.call_extern(
            dst.dtype, "set_vector_mask",
            tvm.const(-1, dtype="uint64"),
            tvm.const(-1, dtype="uint64")))

        dst_offset = n_index * h_i * w_i * c_i + num_u * col_len_ub * c_i
        burst_len_dst = col_len_ub * c_1
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr("r",
                                                        offset=0),
                                    0, 1, burst_len_dst, 0, 0))
    with tvm_ib.if_scope(ub_mod > 0):
        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            data_offset = n_index * h_i * w_i * c_i\
                          + num_c1 * col_len_shape * c_0\
                          + ub_loop * col_len_ub * c_0
            ub_offset = num_c1 * ub_mod * c_0
            ori_nburst = 1
            burst_len = ub_mod
            src_stride = 0
            dst_stride = 0
            args = tvm_ib, data, data_ub, data_offset, ub_offset, ori_nburst,\
                   burst_len, src_stride, dst_stride, cp_align_len
            _func_gm_to_ub_align(args)

        vadds_zu = ub_mod // 8
        vadds_mod = ub_mod % 8
        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            with tvm_ib.if_scope(vadds_zu > 0):
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(-1, dtype="uint64"),
                    tvm.const(-1, dtype="uint64")))

                ub_offset = num_c1 * ub_mod * c_0
                res_offset = num_c1 * c_0
                repeat = vadds_zu
                srcm0 = 1
                dstm0 = c_1
                srcm1 = 8
                dstm1 = 8 * c_1
                args = tvm_ib, data_ub, data_res, c_0, c_1, ub_offset,\
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1,\
                       cp_align_len
                _func_vadds_for_move(args)
            with tvm_ib.if_scope(vadds_mod > 0):
                ub_offset = num_c1 * ub_mod * c_0 + vadds_zu * 8 * c_0
                res_offset = num_c1 * c_0 + vadds_zu * 8 * c_i
                repeat = 1
                srcm0 = 1
                dstm0 = c_1
                srcm1 = 0
                dstm1 = 0
                before = 0
                after = (8 - vadds_mod) * 16
                mask1, mask2 = _set_mask_slice_fp16(before, after)
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                _set_mask_slice_fp16(before, after)
                args = tvm_ib, data_ub, data_res, c_0, c_1, ub_offset, \
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1, \
                       cp_align_len
                _func_vadds_for_move(args)

        tvm_ib.emit(tvm.call_extern(
            dst.dtype, "set_vector_mask",
            tvm.const(-1, dtype="uint64"),
            tvm.const(-1, dtype="uint64")))

        dst_offset = n_index * h_i * w_i * c_i + ub_loop * col_len_ub * c_i
        burst_len_dst = ub_mod * c_1
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr("r",
                                                        offset=0),
                                    0, 1, burst_len_dst, 0, 0))


def _ci_align_nhwc_fp16(dst, data):
    """
    function of making ir node builder for ci align nhwc float16 scene

    """
    tvm_ib = tvm.ir_builder.create()

    n_i, h_i, w_i, c_i = dst.shape
    c_1 = data.shape[1]
    c_0 = data.shape[4]

    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 1024
    ub_ele = (ub_bytes // 2 // float_size // c_i) * c_i

    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    col_len_shape = h_i * w_i
    col_len_ub = ub_ele // c_i
    ub_loop = col_len_shape // col_len_ub
    ub_mod = col_len_shape % col_len_ub

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, device_core_num,\
               block_index, h_i, w_i, c_i, c_1, c_0, col_len_shape,\
               col_len_ub, cp_align_len, ub_loop, ub_mod, num_g
        _func_ci_align_nhwc_fp16(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, device_core_num, \
                   block_index, h_i, w_i, c_i, c_1, c_0, col_len_shape, \
                   col_len_ub, cp_align_len, ub_loop, ub_mod, group_index
            _func_ci_align_nhwc_fp16(args)

    return tvm_ib.get()


def _func_ci_align_nhwc_fp16_one(args):
    """
    function of moving data for ci align nhwc float16 scene

    """
    tvm_ib, data, dst, data_ub, data_res, device_core_num, block_index,\
    h_i, w_i, c_i, c_1, c_0, col_len_shape, col_len_ub, cp_align_len,\
    ub_loop, ub_mod = args

    with tvm_ib.for_range(0, ub_loop, name="num_u") as num_u:
        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            data_offset = block_index * col_len_shape * c_0\
                          + num_c1 * h_i * w_i * c_0\
                          + num_u * col_len_ub * c_0
            ub_offset = num_c1 * col_len_ub * c_0
            ori_nburst = 1
            burst_len = col_len_ub
            src_stride = 0
            dst_stride = 0
            args = tvm_ib, data, data_ub, data_offset, ub_offset, ori_nburst,\
                   burst_len, src_stride, dst_stride, cp_align_len
            _func_gm_to_ub_align(args)

        vadds_zu = col_len_ub // 8
        vadds_mod = col_len_ub % 8
        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            with tvm_ib.if_scope(vadds_zu > 0):
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(-1, dtype="uint64"),
                    tvm.const(-1, dtype="uint64")))

                ub_offset = num_c1 * col_len_ub * c_0
                res_offset = num_c1 * c_0
                repeat = vadds_zu
                srcm0 = 1
                dstm0 = c_1
                srcm1 = 8
                dstm1 = 8 * c_1
                args = tvm_ib, data_ub, data_res, c_0, c_1, ub_offset,\
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1,\
                       cp_align_len
                _func_vadds_for_move(args)
            with tvm_ib.if_scope(vadds_mod > 0):
                ub_offset = num_c1 * col_len_ub * c_0 + vadds_zu * 8 * c_0
                res_offset = num_c1 * c_0 + vadds_zu * 8 * c_i
                repeat = 1
                srcm0 = 1
                dstm0 = c_1
                srcm1 = 0
                dstm1 = 0
                before = 0
                after = (8 - vadds_mod) * 16
                mask1, mask2 = _set_mask_slice_fp16(before, after)
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                args = tvm_ib, data_ub, data_res, c_0, c_1, ub_offset, \
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1, \
                       cp_align_len
                _func_vadds_for_move(args)

        tvm_ib.emit(tvm.call_extern(
            dst.dtype, "set_vector_mask",
            tvm.const(-1, dtype="uint64"),
            tvm.const(-1, dtype="uint64")))

        dst_offset = block_index * col_len_shape * c_i\
                     + num_u * col_len_ub * c_i
        burst_len_dst = col_len_ub * c_1
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr("r",
                                                        offset=0),
                                    0, 1, burst_len_dst, 0, 0))
    with tvm_ib.if_scope(ub_mod > 0):
        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            data_offset = block_index * col_len_shape * c_0 \
                          + num_c1 * h_i * w_i * c_0 \
                          + ub_loop * col_len_ub * c_0
            ub_offset = num_c1 * ub_mod * c_0
            ori_nburst = 1
            burst_len = ub_mod
            src_stride = 0
            dst_stride = 0
            args = tvm_ib, data, data_ub, data_offset, ub_offset, ori_nburst,\
                   burst_len, src_stride, dst_stride, cp_align_len
            _func_gm_to_ub_align(args)

        vadds_zu = ub_mod // 8
        vadds_mod = ub_mod % 8
        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            with tvm_ib.if_scope(vadds_zu > 0):
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(-1, dtype="uint64"),
                    tvm.const(-1, dtype="uint64")))

                ub_offset = num_c1 * ub_mod * c_0
                res_offset = num_c1 * c_0
                repeat = vadds_zu
                srcm0 = 1
                dstm0 = c_1
                srcm1 = 8
                dstm1 = 8 * c_1
                args = tvm_ib, data_ub, data_res, c_0, c_1, ub_offset,\
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1,\
                       cp_align_len
                _func_vadds_for_move(args)
            with tvm_ib.if_scope(vadds_mod > 0):
                ub_offset = num_c1 * ub_mod * c_0 + vadds_zu * 8 * c_0
                res_offset = num_c1 * c_0 + vadds_zu * 8 * c_i
                repeat = 1
                srcm0 = 1
                dstm0 = c_1
                srcm1 = 0
                dstm1 = 0
                before = 0
                after = (8 - vadds_mod) * 16
                mask1, mask2 = _set_mask_slice_fp16(before, after)
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                _set_mask_slice_fp16(before, after)
                args = tvm_ib, data_ub, data_res, c_0, c_1, ub_offset, \
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1, \
                       cp_align_len
                _func_vadds_for_move(args)

        tvm_ib.emit(tvm.call_extern(
            dst.dtype, "set_vector_mask",
            tvm.const(-1, dtype="uint64"),
            tvm.const(-1, dtype="uint64")))

        dst_offset = block_index * col_len_shape * c_i \
                     + ub_loop * col_len_ub * c_i
        burst_len_dst = ub_mod * c_1
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr("r",
                                                        offset=0),
                                    0, 1, burst_len_dst, 0, 0))


def _ci_align_nhwc_fp16_one(dst, data):
    """
    function of making ir node builder for ci align nhwc float16 scene

    """
    tvm_ib = tvm.ir_builder.create()

    n_i, h_i, w_i, c_i = dst.shape
    c_1 = data.shape[1]
    c_0 = data.shape[4]

    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 1024
    ub_ele = (ub_bytes // 2 // float_size // c_i) * c_i

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    col_len_shape = h_i * w_i // device_core_num
    col_len_ub = ub_ele // c_i
    ub_loop = col_len_shape // col_len_ub
    ub_mod = col_len_shape % col_len_ub

    args = tvm_ib, data, dst, data_ub, data_res, device_core_num, \
           block_index, h_i, w_i, c_i, c_1, c_0, col_len_shape, \
           col_len_ub, cp_align_len, ub_loop, ub_mod
    _func_ci_align_nhwc_fp16_one(args)

    return tvm_ib.get()


def _vconv_one_fp16(args):
    """
    function of vnchwconv for _func_vconv_fp16

    """
    tvm_ib, addr_array, addr_array_buf, one_begin, two_begin,\
    repeat_vconv, src_stride_vconv, dst_stride_vconv, col_zu = args

    src0_offset = 8*0
    src1_offset = 8*1
    dst0_offset = 8*2
    dst1_offset = 8*3
    src_gap = 32
    src_eight_gap = src_gap*8
    dst_gap = 32 * col_zu
    dst_eight_gap = dst_gap*8

    with tvm_ib.for_range(0, 8, name="i") as i:
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src0_offset + i]),
                                    one_begin + i * src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src1_offset + i]),
                                    one_begin + src_eight_gap
                                    + i * src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst0_offset + i]),
                                    two_begin + i * dst_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst1_offset + i]),
                                    two_begin + dst_eight_gap
                                    + i * dst_gap))

    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA0",
                                addr_array_buf.access_ptr("rw",
                                                          offset=src0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA1",
                                addr_array_buf.access_ptr("rw",
                                                          offset=src1_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA2",
                                addr_array_buf.access_ptr("rw",
                                                          offset=dst0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA3",
                                addr_array_buf.access_ptr("rw",
                                                          offset=dst1_offset)))

    with tvm_ib.if_scope(repeat_vconv == 1):
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    1,
                                    0,
                                    0))
    with tvm_ib.else_scope():
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    repeat_vconv,
                                    dst_stride_vconv,
                                    src_stride_vconv))


def _func_ci_align_nchw_fp16(args):
    """
    function of moving data for ci align nhwc float16 scene

    """
    tvm_ib, data, dst, data_ub, data_res, addr_array, addr_array_buf,\
    device_core_num, block_index, float_size, h_i, w_i, c_i, c_1, c_0,\
    col_len_shape, col_len_ub, cp_align_len, ub_ele, ub_loop,\
    ub_mod, num_g = args

    n_index = num_g * device_core_num + block_index

    with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
        with tvm_ib.for_range(0, ub_loop, name="num_u") as num_u:
            data_offset = n_index * h_i * w_i * c_i\
                          + num_c1 * h_i * w_i * c_0\
                          + num_u * col_len_ub * c_0
            burst_len = col_len_ub * c_0 // cp_align_len
            tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                        data_ub.access_ptr(
                                            "w", offset=0),
                                        data.access_ptr(
                                            "r", offset=data_offset),
                                        0, 1, burst_len, 0, 0))

            one_begin = 0
            two_begin = ub_ele * float_size
            repeat_vconv = col_len_ub // 16
            src_stride_vconv = 16
            dst_stride_vconv = 1
            col_zu = col_len_ub // 16
            args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin,\
                   repeat_vconv, src_stride_vconv, dst_stride_vconv, col_zu
            _vconv_one_fp16(args)

            dst_offset = n_index * h_i * w_i * c_i\
                         + num_c1 * h_i * w_i * c_0\
                         + num_u * col_len_ub
            n_burst = c_0
            burst_len_dst = col_len_ub // cp_align_len
            dst_stride = (col_len_shape - col_len_ub) // cp_align_len
            args = tvm_ib, dst, data_res, dst_offset, 0, n_burst, \
                   burst_len_dst, 0, dst_stride, cp_align_len
            _func_ub_to_gm_align(args)

        with tvm_ib.if_scope(ub_mod > 0):
            data_offset = n_index * h_i * w_i * c_i \
                          + num_c1 * h_i * w_i * c_0 \
                          + ub_loop * col_len_ub * c_0
            burst_len = ub_mod * c_0 // cp_align_len
            tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                        data_ub.access_ptr(
                                            "w", offset=0),
                                        data.access_ptr(
                                            "r", offset=data_offset),
                                        0, 1, burst_len, 0, 0))

            one_begin = 0
            two_begin = ub_ele * float_size
            repeat_vconv = ub_mod // 16
            src_stride_vconv = 16
            dst_stride_vconv = 1
            col_zu = ub_mod // 16
            args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
                   repeat_vconv, src_stride_vconv, dst_stride_vconv, col_zu
            _vconv_one_fp16(args)

            dst_offset = n_index * h_i * w_i * c_i \
                         + num_c1 * h_i * w_i * c_0 \
                         + ub_loop * col_len_ub
            n_burst = c_0
            burst_len_dst = ub_mod // cp_align_len
            dst_stride = (col_len_shape - ub_mod) // cp_align_len
            args = tvm_ib, dst, data_res, dst_offset, 0, n_burst, \
                   burst_len_dst, 0, dst_stride, cp_align_len
            _func_ub_to_gm_align(args)


def _ci_align_nchw_fp16(dst, data):
    """
    function of making ir node builder for ci align nhwc float16 scene

    """
    tvm_ib = tvm.ir_builder.create()

    n_i, c_i, h_i, w_i = dst.shape
    c_1 = data.shape[1]
    c_0 = data.shape[4]

    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 1024
    ub_ele = (ub_bytes // 2 // float_size // c_i) * c_i

    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    col_len_shape = h_i * w_i
    col_len_ub = (ub_ele // c_0) // 16 * 16
    ub_loop = col_len_shape // col_len_ub
    ub_mod = col_len_shape % col_len_ub

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, addr_array,\
               addr_array_buf, device_core_num,\
               block_index, float_size, h_i, w_i, c_i, c_1, c_0,\
               col_len_shape, col_len_ub, cp_align_len, ub_ele,\
               ub_loop, ub_mod, num_g
        _func_ci_align_nchw_fp16(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, addr_array,\
                   addr_array_buf, device_core_num, \
                   block_index, float_size, h_i, w_i, c_i, c_1, c_0,\
                   col_len_shape, col_len_ub, cp_align_len, ub_ele, ub_loop,\
                   ub_mod, group_index
            _func_ci_align_nchw_fp16(args)

    return tvm_ib.get()


def _move_full(dst, data):
    """
    move for full shape

    """
    tvm_ib = tvm.ir_builder.create()
    dim_ele = functools_reduce(lambda x, y: x * y, dst.shape[1:])
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 32
    ub_ele = (ub_bytes // float_size // cp_align_len)*cp_align_len
    device_core_num = AICORE_NUM

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    n_i = data.shape[0]
    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    ub_loop = dim_ele // ub_ele
    ub_mod = dim_ele % ub_ele

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        with tvm_ib.for_range(0, ub_loop, name="num_uloop") as num_uloop:
            data_offset = num_g*device_core_num*dim_ele + block_index*dim_ele\
                          + num_uloop*ub_ele
            burst_len_data = ub_ele // cp_align_len
            tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                        data_ub.access_ptr("w", offset=0),
                                        data.access_ptr('r',
                                                        offset=data_offset),
                                        0, 1, burst_len_data, 0, 0))

            dst_offset = num_g*device_core_num*dim_ele + block_index*dim_ele\
                         + num_uloop*ub_ele
            burst_len_dst = ub_ele // cp_align_len
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_ub.access_ptr("r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))
        with tvm_ib.if_scope(ub_mod > 0):
            data_offset = num_g * device_core_num * dim_ele\
                          + block_index * dim_ele + ub_loop * ub_ele
            burst_len_data = ub_mod // cp_align_len
            tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                        data_ub.access_ptr("w", offset=0),
                                        data.access_ptr('r',
                                                        offset=data_offset),
                                        0, 1, burst_len_data, 0, 0))

            dst_offset = num_g * device_core_num * dim_ele\
                         + block_index * dim_ele + ub_loop * ub_ele
            burst_len_dst = ub_mod // cp_align_len
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_ub.access_ptr("r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            with tvm_ib.for_range(0, ub_loop, name="num_uloop") as num_uloop:
                data_offset = group_index * device_core_num * dim_ele\
                              + block_index * dim_ele + num_uloop * ub_ele
                burst_len_data = ub_ele // cp_align_len
                tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                            data_ub.access_ptr("w", offset=0),
                                            data.access_ptr('r',
                                                            offset=data_offset),
                                            0, 1, burst_len_data, 0, 0))

                dst_offset = group_index * device_core_num * dim_ele\
                             + block_index * dim_ele + num_uloop * ub_ele
                burst_len_dst = ub_ele // cp_align_len
                tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr('w',
                                                           offset=dst_offset),
                                            data_ub.access_ptr("r", offset=0),
                                            0, 1, burst_len_dst, 0, 0))
            with tvm_ib.if_scope(ub_mod > 0):
                data_offset = group_index * device_core_num * dim_ele\
                              + block_index * dim_ele + ub_loop * ub_ele
                burst_len_data = ub_mod // cp_align_len
                tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                            data_ub.access_ptr("w", offset=0),
                                            data.access_ptr('r',
                                                            offset=data_offset),
                                            0, 1, burst_len_data, 0, 0))

                dst_offset = group_index * device_core_num * dim_ele\
                             + block_index * dim_ele + ub_loop * ub_ele
                burst_len_dst = ub_mod // cp_align_len
                tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr('w',
                                                           offset=dst_offset),
                                            data_ub.access_ptr("r", offset=0),
                                            0, 1, burst_len_dst, 0, 0))

    return tvm_ib.get()


def _vconv_one(args):
    """
    function of vnchwconv for _func_vconv_fp16

    """
    tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
    repeat_vconv, src_stride_vconv, dst_stride_vconv = args

    src0_offset = 8 * 0
    src1_offset = 8 * 1
    dst0_offset = 8 * 2
    dst1_offset = 8 * 3
    src_gap = 32 * 16
    src_eight_gap = src_gap * 8
    dst_gap = 32
    dst_eight_gap = dst_gap * 8

    with tvm_ib.for_range(0, 8, name="i") as i:
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src0_offset + i]),
                                    one_begin + i * src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src1_offset + i]),
                                    one_begin + src_eight_gap
                                    + i * src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst0_offset + i]),
                                    two_begin + i * dst_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst1_offset + i]),
                                    two_begin + dst_eight_gap
                                    + i * dst_gap))

    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA0",
                                addr_array_buf.access_ptr("rw",
                                                          offset=src0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA1",
                                addr_array_buf.access_ptr("rw",
                                                          offset=src1_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA2",
                                addr_array_buf.access_ptr("rw",
                                                          offset=dst0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA3",
                                addr_array_buf.access_ptr("rw",
                                                          offset=dst1_offset)))
    with tvm_ib.if_scope(repeat_vconv == 1):
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    1,
                                    0,
                                    0))
    with tvm_ib.else_scope():
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    repeat_vconv,
                                    dst_stride_vconv,
                                    src_stride_vconv))


def _vconv_two(args):
    """
    function of vnchwconv for _func_vconv_fp16

    """
    tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
    repeat_vconv, src_stride_vconv, dst_stride_vconv, c_i = args

    src0_offset = 8 * 0
    src1_offset = 8 * 1
    dst0_offset = 8 * 2
    dst1_offset = 8 * 3
    src_gap = 32
    src_eight_gap = src_gap * 8
    dst_gap = c_i * 32
    dst_eight_gap = dst_gap * 8

    with tvm_ib.for_range(0, 8, name="i") as i:
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src0_offset + i]),
                                    one_begin + i * src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src1_offset + i]),
                                    one_begin + src_eight_gap
                                    + i * src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst0_offset + i]),
                                    two_begin + i * dst_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst1_offset + i]),
                                    two_begin + dst_eight_gap
                                    + i * dst_gap))

    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA0",
                                addr_array_buf.access_ptr("rw",
                                                          offset=src0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA1",
                                addr_array_buf.access_ptr("rw",
                                                          offset=src1_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA2",
                                addr_array_buf.access_ptr("rw",
                                                          offset=dst0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA3",
                                addr_array_buf.access_ptr("rw",
                                                          offset=dst1_offset)))

    with tvm_ib.if_scope(repeat_vconv == 1):
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    1,
                                    0,
                                    0))
    with tvm_ib.else_scope():
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    repeat_vconv,
                                    dst_stride_vconv,
                                    src_stride_vconv))


def _func_vconv_fp16_new(args):
    """
    function of moving data for vconv float16 scene

    """
    tvm_ib, data, dst, data_one, data_two, \
    addr_array, addr_array_buf, device_core_num, block_index, \
    cp_align_len, float_size, c_i, \
    num_dim_group_one_core, dim_ele_c0, dim_ele_ci, num_g = args

    num_dim_group_cur_block = dst.shape[1]
    ub_loop = num_dim_group_cur_block // num_dim_group_one_core
    dim_group_mod = num_dim_group_cur_block % num_dim_group_one_core

    block_ele_in = functools_reduce(lambda x, y: x * y, data.shape[1:])
    block_ele_out = functools_reduce(lambda x, y: x * y, dst.shape[1:])

    n_index = num_g * device_core_num + block_index

    with tvm_ib.for_range(0, ub_loop, name="num_u") as num_u:
        data_offset = n_index * block_ele_in\
                      + num_u * num_dim_group_one_core * dim_ele_c0
        burst_len_data = _ceil_div(num_dim_group_one_core * dim_ele_c0,
                                   cp_align_len)
        tvm_ib.emit(tvm.call_extern(data_one.dtype, "copy_gm_to_ubuf",
                                    data_one.access_ptr("w", offset=0),
                                    data.access_ptr('r',
                                                    offset=data_offset),
                                    0, 1, burst_len_data, 0, 0))

        tvm_ib.emit(tvm.call_extern(data_two.dtype, "copy_ubuf_to_ubuf",
                                    data_two.access_ptr("w", offset=0),
                                    data_one.access_ptr('r',
                                                        offset=0),
                                    0, 1, 1, 0, 0))

        with tvm_ib.for_range(0, num_dim_group_one_core, name="num_d")\
                as num_d:
            one_begin = (num_d * dim_ele_c0) * float_size
            two_begin = (num_dim_group_one_core * dim_ele_c0
                         + num_d * dim_ele_c0) * float_size
            repeat_vconv = 16
            src_stride_vconv = 1
            dst_stride_vconv = 16
            args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
                   repeat_vconv, src_stride_vconv, dst_stride_vconv
            _vconv_one(args)

        n_burst = num_dim_group_one_core * 16
        burst_len_data = _ceil_div(c_i * 16, cp_align_len)
        src_stride = _ceil_div((16 - c_i) * 16, cp_align_len)
        tvm_ib.emit(tvm.call_extern(data_one.dtype, "copy_ubuf_to_ubuf",
                                    data_one.access_ptr("w", offset=0),
                                    data_two.access_ptr('r', offset=0),
                                    0, n_burst, burst_len_data, src_stride, 0))

        with tvm_ib.for_range(0, num_dim_group_one_core, name="num_d")\
                as num_d:
            three_begin = (num_d * dim_ele_ci) * float_size
            four_begin = (num_dim_group_one_core * dim_ele_c0
                          + num_d * dim_ele_ci) * float_size
            with tvm_ib.if_scope(c_i > 1):
                repeat_vconv = c_i
                src_stride_vconv = 16
                dst_stride_vconv = 1
                args = tvm_ib, addr_array, addr_array_buf, three_begin,\
                       four_begin, repeat_vconv, src_stride_vconv,\
                       dst_stride_vconv, c_i
                _vconv_two(args)
            with tvm_ib.if_scope(c_i == 1):
                repeat_vconv = c_i
                src_stride_vconv = 0
                dst_stride_vconv = 0
                args = tvm_ib, addr_array, addr_array_buf, three_begin,\
                       four_begin, repeat_vconv, src_stride_vconv,\
                       dst_stride_vconv, c_i
                _vconv_two(args)

        dst_offset = n_index * block_ele_out\
                     + num_u * num_dim_group_one_core * dim_ele_ci
        burst_len_dst = _ceil_div(num_dim_group_one_core * dim_ele_ci,
                                  cp_align_len)
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_two.access_ptr("r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))

    with tvm_ib.if_scope(dim_group_mod > 0):
        data_offset = n_index * block_ele_in\
                      + ub_loop * num_dim_group_one_core * dim_ele_c0
        burst_len_data = _ceil_div(dim_group_mod * dim_ele_c0, cp_align_len)
        tvm_ib.emit(tvm.call_extern(data_one.dtype, "copy_gm_to_ubuf",
                                    data_one.access_ptr("w", offset=0),
                                    data.access_ptr('r',
                                                    offset=data_offset),
                                    0, 1, burst_len_data, 0, 0))

        tvm_ib.emit(tvm.call_extern(data_two.dtype, "copy_ubuf_to_ubuf",
                                    data_two.access_ptr("w", offset=0),
                                    data_one.access_ptr('r',
                                                        offset=0),
                                    0, 1, 1, 0, 0))

        with tvm_ib.for_range(0, dim_group_mod, name="num_d") as num_d:
            one_begin = (num_d * dim_ele_c0) * float_size
            two_begin = (num_dim_group_one_core * dim_ele_c0
                         + num_d * dim_ele_c0) * float_size
            repeat_vconv = 16
            src_stride_vconv = 1
            dst_stride_vconv = 16
            args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
                   repeat_vconv, src_stride_vconv, dst_stride_vconv
            _vconv_one(args)

        n_burst = dim_group_mod * 16
        burst_len_data = _ceil_div(c_i * 16, cp_align_len)
        src_stride = _ceil_div((16 - c_i) * 16, cp_align_len)
        tvm_ib.emit(tvm.call_extern(data_one.dtype, "copy_ubuf_to_ubuf",
                                    data_one.access_ptr("w", offset=0),
                                    data_two.access_ptr('r', offset=0),
                                    0, n_burst, burst_len_data, src_stride, 0))

        with tvm_ib.for_range(0, dim_group_mod, name="num_d") as num_d:
            three_begin = (num_d * dim_ele_ci) * float_size
            four_begin = (num_dim_group_one_core * dim_ele_c0
                          + num_d * dim_ele_ci) * float_size
            with tvm_ib.if_scope(c_i > 1):
                repeat_vconv = c_i
                src_stride_vconv = 16
                dst_stride_vconv = 1
                args = tvm_ib, addr_array, addr_array_buf, three_begin,\
                       four_begin, repeat_vconv, src_stride_vconv,\
                       dst_stride_vconv, c_i
                _vconv_two(args)
            with tvm_ib.if_scope(c_i == 1):
                repeat_vconv = c_i
                src_stride_vconv = 0
                dst_stride_vconv = 0
                args = tvm_ib, addr_array, addr_array_buf, three_begin,\
                       four_begin, repeat_vconv, src_stride_vconv,\
                       dst_stride_vconv, c_i
                _vconv_two(args)

        dst_offset = n_index * block_ele_out\
                     + ub_loop * num_dim_group_one_core * dim_ele_ci
        burst_len_dst = _ceil_div(dim_group_mod * dim_ele_ci,
                                  cp_align_len)
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_two.access_ptr("r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))


def _move_vconv_fp16_new(dst, data):
    """
    function of making ir node builder for vconv fp16 scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 32
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_ele = ub_bytes // float_size
    c_i = dst.shape[3]
    dim_group_ele = 16 * 16 * 16 * 2
    num_dim_group_one_core = ub_ele // dim_group_ele
    dim_ele_c0 = 16 * 16 * 16
    dim_ele_ci = 16 * 16 * c_i

    data_one = _new_alloc(tvm_ib, dst.dtype,
                          num_dim_group_one_core * dim_ele_c0,
                          "data_one", scope=cce.scope_ubuf)
    data_two = _new_alloc(tvm_ib, dst.dtype,
                          num_dim_group_one_core * dim_ele_c0,
                          "data_two", scope=cce.scope_ubuf)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)

    n_i = dst.shape[0]
    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_one, data_two,\
               addr_array, addr_array_buf, device_core_num, block_index, \
               cp_align_len, float_size, c_i, \
               num_dim_group_one_core, dim_ele_c0, dim_ele_ci, num_g
        _func_vconv_fp16_new(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_one, data_two,\
                   addr_array, addr_array_buf, device_core_num, block_index, \
                   cp_align_len, float_size, c_i, \
                   num_dim_group_one_core, dim_ele_c0, dim_ele_ci, group_index
            _func_vconv_fp16_new(args)

    return tvm_ib.get()


def _func_vconv_fp16_new_fencore(args):
    """
    function of moving data for vconv float16 new fencore scene

    """
    tvm_ib, data, dst, data_one, data_two, \
    addr_array, addr_array_buf, \
    cp_align_len, float_size, c_i, num_dim_group_one_core, dim_ele_c0,\
    dim_ele_ci, n_index, fen_index, num_dimg_now = args

    block_ele_in = functools_reduce(lambda x, y: x * y, data.shape[1:])
    block_ele_out = functools_reduce(lambda x, y: x * y, dst.shape[1:])

    data_offset = n_index * block_ele_in \
                  + fen_index * num_dim_group_one_core * dim_ele_c0
    burst_len_data = _ceil_div(num_dimg_now * dim_ele_c0, cp_align_len)
    tvm_ib.emit(tvm.call_extern(data_one.dtype, "copy_gm_to_ubuf",
                                data_one.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=data_offset),
                                0, 1, burst_len_data, 0, 0))

    tvm_ib.emit(tvm.call_extern(data_two.dtype, "copy_ubuf_to_ubuf",
                                data_two.access_ptr("w", offset=0),
                                data_one.access_ptr('r',
                                                    offset=0),
                                0, 1, 1, 0, 0))

    with tvm_ib.for_range(0, num_dimg_now, name="num_d") as num_d:
        one_begin = (num_d * dim_ele_c0) * float_size
        two_begin = (num_dim_group_one_core * dim_ele_c0
                     + num_d * dim_ele_c0) * float_size
        repeat_vconv = 16
        src_stride_vconv = 1
        dst_stride_vconv = 16
        args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
               repeat_vconv, src_stride_vconv, dst_stride_vconv
        _vconv_one(args)

    n_burst = num_dimg_now * 16
    burst_len_data = _ceil_div(c_i * 16, cp_align_len)
    src_stride = _ceil_div((16 - c_i) * 16, cp_align_len)
    tvm_ib.emit(tvm.call_extern(data_one.dtype, "copy_ubuf_to_ubuf",
                                data_one.access_ptr("w", offset=0),
                                data_two.access_ptr('r', offset=0),
                                0, n_burst, burst_len_data, src_stride, 0))

    with tvm_ib.for_range(0, num_dimg_now, name="num_d") as num_d:
        three_begin = (num_d * dim_ele_ci) * float_size
        four_begin = (num_dim_group_one_core * dim_ele_c0
                      + num_d * dim_ele_ci) * float_size
        with tvm_ib.if_scope(c_i > 1):
            repeat_vconv = c_i
            src_stride_vconv = 16
            dst_stride_vconv = 1
            args = tvm_ib, addr_array, addr_array_buf, three_begin, \
                   four_begin, repeat_vconv, src_stride_vconv, \
                   dst_stride_vconv, c_i
            _vconv_two(args)
        with tvm_ib.if_scope(c_i == 1):
            repeat_vconv = c_i
            src_stride_vconv = 0
            dst_stride_vconv = 0
            args = tvm_ib, addr_array, addr_array_buf, three_begin, \
                   four_begin, repeat_vconv, src_stride_vconv, \
                   dst_stride_vconv, c_i
            _vconv_two(args)

    dst_offset = n_index * block_ele_out \
                 + fen_index * num_dim_group_one_core * dim_ele_ci
    burst_len_dst = _ceil_div(num_dimg_now * dim_ele_ci,
                              cp_align_len)
    tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                dst.access_ptr('w',
                                               offset=dst_offset),
                                data_two.access_ptr("r", offset=0),
                                0, 1, burst_len_dst, 0, 0))


def _func_vconv_fp16_new_nomod(args):
    """
    function of moving data for vconv float16 new nomod scene

    """
    tvm_ib, data, dst, data_one, data_two, \
    addr_array, addr_array_buf, device_core_num, block_index, \
    cp_align_len, float_size, c_i, num_dim_group_one_core, \
    dim_ele_c0, dim_ele_ci, fen_n, num_g = args

    core_index = num_g * device_core_num + block_index
    n_index = core_index // fen_n
    fen_index = core_index % fen_n

    num_dimg_now = num_dim_group_one_core
    args = tvm_ib, data, dst, data_one, data_two, \
    addr_array, addr_array_buf, \
    cp_align_len, float_size, c_i, num_dim_group_one_core, dim_ele_c0,\
    dim_ele_ci, n_index, fen_index, num_dimg_now
    _func_vconv_fp16_new_fencore(args)


def _func_vconv_fp16_new_mod(args):
    """
    function of moving data for vconv float16 new mod scene

    """
    tvm_ib, data, dst, data_one, data_two, \
    addr_array, addr_array_buf, device_core_num, block_index, \
    cp_align_len, float_size, c_i, num_dim_group_one_core, \
    dim_ele_c0, dim_ele_ci, dim_group_mod, fen_n, num_g = args

    core_index = num_g * device_core_num + block_index
    n_index = core_index // fen_n
    fen_index = core_index % fen_n

    with tvm_ib.if_scope(fen_index < (fen_n - 1)):
        num_dimg_now = num_dim_group_one_core
        args = tvm_ib, data, dst, data_one, data_two, \
               addr_array, addr_array_buf, cp_align_len, float_size, c_i,\
               num_dim_group_one_core, dim_ele_c0,\
               dim_ele_ci, n_index, fen_index, num_dimg_now
        _func_vconv_fp16_new_fencore(args)
    with tvm_ib.else_scope():
        num_dimg_now = dim_group_mod
        args = tvm_ib, data, dst, data_one, data_two,\
               addr_array, addr_array_buf, cp_align_len, float_size, c_i,\
               num_dim_group_one_core, dim_ele_c0, dim_ele_ci,\
               n_index, fen_index, num_dimg_now
        _func_vconv_fp16_new_fencore(args)


def _move_vconv_fp16_new_fencore(dst, data):
    """
    function of making ir node builder for vconv fp16 new fencore scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 32
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_ele = ub_bytes // float_size
    c_i = dst.shape[3]
    dim_group_ele = 16 * 16 * 16 * 2
    num_dim_group_one_core = ub_ele // dim_group_ele
    dim_ele_c0 = 16 * 16 * 16
    dim_ele_ci = 16 * 16 * c_i

    data_one = _new_alloc(tvm_ib, dst.dtype,
                          num_dim_group_one_core * dim_ele_c0,
                          "data_one", scope=cce.scope_ubuf)
    data_two = _new_alloc(tvm_ib, dst.dtype,
                          num_dim_group_one_core * dim_ele_c0,
                          "data_two", scope=cce.scope_ubuf)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)

    n_i = dst.shape[0]
    num_dim_group_cur_block = dst.shape[1]
    ub_loop = num_dim_group_cur_block // num_dim_group_one_core
    dim_group_mod = num_dim_group_cur_block % num_dim_group_one_core
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.if_scope(dim_group_mod > 0):
        fen_n = ub_loop + 1
        all_core = n_i * fen_n
        group_index = all_core // device_core_num
        group_mod = all_core % device_core_num

        with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
            args = tvm_ib, data, dst, data_one, data_two, \
                   addr_array, addr_array_buf, device_core_num, block_index, \
                   cp_align_len, float_size, c_i, num_dim_group_one_core, \
                   dim_ele_c0, dim_ele_ci, dim_group_mod, fen_n, num_g
            _func_vconv_fp16_new_mod(args)
        with tvm_ib.if_scope(group_mod > 0):
            with tvm_ib.if_scope(block_index < group_mod):
                args = tvm_ib, data, dst, data_one, data_two, \
                   addr_array, addr_array_buf, device_core_num, block_index, \
                   cp_align_len, float_size, c_i, num_dim_group_one_core, \
                   dim_ele_c0, dim_ele_ci, dim_group_mod, fen_n, group_index
                _func_vconv_fp16_new_mod(args)

    with tvm_ib.else_scope():
        fen_n = ub_loop
        all_core = n_i * fen_n
        group_index = all_core // device_core_num
        group_mod = all_core % device_core_num

        with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
            args = tvm_ib, data, dst, data_one, data_two, \
                   addr_array, addr_array_buf, device_core_num, block_index, \
                   cp_align_len, float_size, c_i, num_dim_group_one_core, \
                   dim_ele_c0, dim_ele_ci, fen_n, num_g
            _func_vconv_fp16_new_nomod(args)
        with tvm_ib.if_scope(group_mod > 0):
            with tvm_ib.if_scope(block_index < group_mod):
                args = tvm_ib, data, dst, data_one, data_two, \
                       addr_array, addr_array_buf, device_core_num,\
                       block_index, cp_align_len, float_size, c_i,\
                       num_dim_group_one_core, dim_ele_c0, dim_ele_ci,\
                       fen_n, group_index
                _func_vconv_fp16_new_nomod(args)

    return tvm_ib.get()


def _func_move_one_more_batch(args):
    """
    function of moving data for move one for more batch scene

    """
    tvm_ib, data, dst, data_ub, data_tail, reg, device_core_num,\
    block_index, cp_align_len, ub_ele, c_i, ub_loop, ub_mod, num_g = args

    _, c_1, _, _, c_0 = data.shape
    n_index = num_g * device_core_num + block_index

    with tvm_ib.for_range(0, ub_loop, name="num_u") as num_u:
        data_offset = n_index * c_1 * c_0 + num_u * ub_ele
        burst_len_data = ub_ele // cp_align_len
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr("w", offset=0),
                                    data.access_ptr('r',
                                                    offset=data_offset),
                                    0, 1, burst_len_data, 0, 0))

        dst_offset = n_index * c_i + num_u * ub_ele
        burst_len_dst = ub_ele // cp_align_len
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_ub.access_ptr("r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))
    with tvm_ib.if_scope(ub_mod > 0):
        with tvm_ib.if_scope(ub_mod >= cp_align_len):
            data_offset = n_index * c_1 * c_0 + ub_loop * ub_ele
            burst_len_data = _ceil_div(ub_mod, cp_align_len)
            tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                        data_ub.access_ptr("w", offset=0),
                                        data.access_ptr('r',
                                                        offset=data_offset),
                                        0, 1, burst_len_data, 0, 0))
            with tvm_ib.if_scope(ub_mod % cp_align_len > 0):
                ub_mod_align = ub_mod - cp_align_len
                dst_offset = n_index * c_i + ub_loop * ub_ele
                burst_len_dst = _ceil_div(ub_mod_align, cp_align_len)
                tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr('w',
                                                           offset=dst_offset),
                                            data_ub.access_ptr("r",
                                                               offset=0),
                                            0, 1, burst_len_dst, 0, 0))
                with tvm_ib.for_range(0, cp_align_len,
                                      name="num_a") as num_a:
                    tvm_ib.emit(tvm.call_extern(
                        data_ub.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[0]),
                        data_ub.access_ptr(
                            'r', offset=ub_mod_align + num_a)
                    ))
                    tvm_ib.emit(tvm.call_extern(
                        data_tail.dtype, "reg_mov",
                        data_tail.access_ptr(
                            'w', offset=num_a),
                        tvm.call_extern(reg.dtype, "reg", reg[0])
                    ))
                tvm_ib.emit(
                    tvm.call_extern(
                        dst.dtype, "copy_ubuf_to_gm",
                        dst.access_ptr(
                            'w',
                            offset=(dst_offset + ub_mod_align)),
                        data_tail.access_ptr("r", offset=0),
                        0, 1, 1, 0, 0))

            with tvm_ib.else_scope():
                dst_offset = n_index * c_i + ub_loop * ub_ele
                burst_len_dst = ub_mod // cp_align_len
                tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr('w',
                                                           offset=dst_offset),
                                            data_ub.access_ptr("r", offset=0),
                                            0, 1, burst_len_dst, 0, 0))
        with tvm_ib.else_scope():
            data_offset = n_index * c_1 * c_0 + c_i - cp_align_len
            tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                        data_ub.access_ptr("w", offset=0),
                                        data.access_ptr('r',
                                                        offset=data_offset),
                                        0, 1, 1, 0, 0))
            dst_offset = n_index * c_i + c_i - cp_align_len
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_ub.access_ptr("r", offset=0),
                                        0, 1, 1, 0, 0))


def _move_one_more_batch(dst, data, c_i):
    """
    function of making ir node builder for move one for more batch scene

    """
    tvm_ib = tvm.ir_builder.create()

    n_i, c_1, _, _, c_0 = data.shape
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 1024
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    ub_loop = c_i // ub_ele
    ub_mod = c_i % ub_ele

    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_tail, reg, device_core_num,\
               block_index, cp_align_len, ub_ele, c_i, ub_loop, ub_mod, num_g
        _func_move_one_more_batch(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_tail, reg,\
                   device_core_num, block_index, cp_align_len, ub_ele,\
                   c_i, ub_loop, ub_mod, group_index
            _func_move_one_more_batch(args)

    return tvm_ib.get()


def _vconv_two_not_align(args):
    """
    function of vnchwconv for _func_vconv_fp32

    """
    tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
    repeat_vconv, src_stride_vconv, dst_stride_vconv = args

    src0_offset = 8 * 0
    src1_offset = 8 * 1
    dst0_offset = 8 * 2
    dst1_offset = 8 * 3
    src_gap = 32
    src_eight_gap = src_gap * 8
    dst_gap = 32
    dst_eight_gap = dst_gap * 8

    with tvm_ib.for_range(0, 8, name="i") as i:
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src0_offset + i]),
                                    one_begin + i * src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src1_offset + i]),
                                    one_begin + src_eight_gap
                                    + i * src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst0_offset + i]),
                                    two_begin + i * dst_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst1_offset + i]),
                                    two_begin + dst_eight_gap
                                    + i * dst_gap))

    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA0",
                                addr_array_buf.access_ptr("rw",
                                                          offset=src0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA1",
                                addr_array_buf.access_ptr("rw",
                                                          offset=src1_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA2",
                                addr_array_buf.access_ptr("rw",
                                                          offset=dst0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA3",
                                addr_array_buf.access_ptr("rw",
                                                          offset=dst1_offset)))
    with tvm_ib.if_scope(repeat_vconv == 1):
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    1,
                                    0,
                                    0))
    with tvm_ib.else_scope():
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    repeat_vconv,
                                    dst_stride_vconv,
                                    src_stride_vconv))


def _move_for_two_split_col_fp32_fencore(args):
    """
    function of moving data for 16_16_64_64 no mod scene

    """
    tvm_ib, data, dst, data_ub, data_res, addr_array, \
    addr_array_buf, float_size, cp_align_len, ub_ele, n_index, \
    col_len, row_len, col_len_begin, col_len_now = args

    data_offset = n_index * col_len * row_len \
                  + col_len_begin * row_len
    burst_len_data = row_len * col_len_now // cp_align_len
    tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                data_ub.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=data_offset),
                                0, 1, burst_len_data, 0, 0))

    dim_ele = col_len_now * row_len
    dim_ele_two = dim_ele * 2
    dim_ele_two_align = _ceil_fill(dim_ele_two, 16)
    dim_zu = dim_ele_two_align // 16

    one_begin = 0
    two_begin = ub_ele * float_size
    repeat_vconv = dim_zu
    src_stride_vconv = 1
    dst_stride_vconv = 16
    args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
           repeat_vconv, src_stride_vconv, dst_stride_vconv
    _vconv_two_not_align(args)

    with tvm_ib.for_range(0, row_len, name="num_r") as num_r:
        res_offset = num_r * 16
        ub_offset = num_r * 16 * col_len_now
        n_burst = col_len_now
        burst_len = 2
        src_stride = row_len * 2 - 2
        dst_stride = 0
        tvm_ib.emit(
            tvm.call_extern(data_ub.dtype, "copy_ubuf_to_ubuf",
                            data_ub.access_ptr(
                                "w", offset=ub_offset),
                            data_res.access_ptr(
                                "r", offset=res_offset),
                            0, n_burst, burst_len,
                            src_stride, dst_stride))

    one_begin = 0
    two_begin = ub_ele * float_size
    repeat_vconv = dim_zu
    src_stride_vconv = 16
    dst_stride_vconv = 1
    args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
           repeat_vconv, src_stride_vconv, dst_stride_vconv
    _vconv_two_not_align(args)

    with tvm_ib.for_range(0, row_len, name="num_r") as num_r:
        dst_offset = n_index * col_len * row_len \
                     + num_r * col_len + col_len_begin
        res_offset = num_r * col_len_now
        burst_len_dst = col_len_now // cp_align_len
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr(
                                        "r", offset=res_offset),
                                    0, 1, burst_len_dst, 0, 0))


def _func_two_split_col_fp32_fencore_nomod(args):
    """
    function of 16_16_64_64 no mod scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
    addr_array, addr_array_buf, ub_ele, float_size, \
    device_core_num, block_index, cp_align_len, col_len_ub, \
    fen_n, ub_loop, ub_mod, num_g = args

    n_i, col_len, row_len = data.shape
    all_fen_index = num_g * device_core_num + block_index
    n_index = all_fen_index // fen_n
    fen_index = all_fen_index % fen_n

    _clean_ubuf(tvm_ib, data_ub, 0, ub_ele)
    _clean_ubuf(tvm_ib, data_res, 0, ub_ele)

    col_len_now = col_len_ub
    col_len_begin = fen_index * col_len_ub
    args = tvm_ib, data, dst, data_ub, data_res, addr_array, \
           addr_array_buf, float_size, cp_align_len, ub_ele, n_index, \
           col_len, row_len, col_len_begin, col_len_now
    _move_for_two_split_col_fp32_fencore(args)


def _func_two_split_col_fp32_fencore_mod(args):
    """
    function of 16_16_64_64 mod scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
    addr_array, addr_array_buf, ub_ele, float_size, \
    device_core_num, block_index, cp_align_len, col_len_ub, \
    fen_n, ub_loop, ub_mod, num_g = args

    n_i, col_len, row_len = data.shape
    all_fen_index = num_g * device_core_num + block_index
    n_index = all_fen_index // fen_n
    fen_index = all_fen_index % fen_n

    _clean_ubuf(tvm_ib, data_ub, 0, ub_ele)
    _clean_ubuf(tvm_ib, data_res, 0, ub_ele)

    with tvm_ib.if_scope(fen_index < fen_n - 1):
        col_len_now = col_len_ub
        col_len_begin = fen_index * col_len_ub
        args = tvm_ib, data, dst, data_ub, data_res, addr_array, \
               addr_array_buf, float_size, cp_align_len, ub_ele, n_index, \
               col_len, row_len, col_len_begin, col_len_now
        _move_for_two_split_col_fp32_fencore(args)

    with tvm_ib.else_scope():
        ub_mod_align = _ceil_fill(ub_mod, cp_align_len)
        col_len_now = ub_mod_align
        col_len_begin = col_len - col_len_now
        args = tvm_ib, data, dst, data_ub, data_res, addr_array, \
               addr_array_buf, float_size, cp_align_len, ub_ele, n_index, \
               col_len, row_len, col_len_begin, col_len_now
        _move_for_two_split_col_fp32_fencore(args)


def _nchw_sp_16_16_64_64_fp32(dst, data):
    """
    function of making ir node builder for 16_16_64_64 scene

    """
    tvm_ib = tvm.ir_builder.create()

    n_i, col_len, row_len = data.shape
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // 2 // float_size // 128) * 128

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)

    new_space_ele_split_col = row_len * 8 * 2 * 8
    num_col_len_ub = ub_ele // new_space_ele_split_col
    col_len_ub = num_col_len_ub * 8
    ub_loop = col_len // col_len_ub
    ub_mod = col_len % col_len_ub
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.if_scope(ub_mod > 0):
        fen_n = ub_loop + 1
        all_fen = n_i * fen_n
        group_index = all_fen // device_core_num
        group_mod = all_fen % device_core_num
        with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   addr_array, addr_array_buf, ub_ele, float_size, \
                   device_core_num, block_index, cp_align_len, col_len_ub, \
                   fen_n, ub_loop, ub_mod, num_g
            _func_two_split_col_fp32_fencore_mod(args)
        with tvm_ib.if_scope(group_mod > 0):
            with tvm_ib.if_scope(block_index < group_mod):
                args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                       addr_array, addr_array_buf, ub_ele, float_size, \
                       device_core_num, block_index, cp_align_len,\
                       col_len_ub, fen_n, ub_loop, ub_mod, group_index
                _func_two_split_col_fp32_fencore_mod(args)
    with tvm_ib.else_scope():
        fen_n = ub_loop
        all_fen = n_i * fen_n
        group_index = all_fen // device_core_num
        group_mod = all_fen % device_core_num
        with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   addr_array, addr_array_buf, ub_ele, float_size, \
                   device_core_num, block_index, cp_align_len, col_len_ub, \
                   fen_n, ub_loop, ub_mod, num_g
            _func_two_split_col_fp32_fencore_nomod(args)
        with tvm_ib.if_scope(group_mod > 0):
            with tvm_ib.if_scope(block_index < group_mod):
                args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                       addr_array, addr_array_buf, ub_ele, float_size, \
                       device_core_num, block_index, cp_align_len,\
                       col_len_ub, fen_n, ub_loop, ub_mod, group_index
                _func_two_split_col_fp32_fencore_nomod(args)

    return tvm_ib.get()


def _func_nchw_fp32_split_hw(args):
    """
    function of moving data for nchw fp32 split hw scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
    addr_array, addr_array_buf, block_index, device_core_num,\
    ub_ele, float_size, cp_align_len, c_0, c_i, hw_i, hw_ub, hw_mod,\
    ub_loop, num_g = args

    n_index = num_g * device_core_num + block_index

    with tvm_ib.for_range(0, ub_loop, name="num_u") as num_u:
        data_offset = n_index * hw_i * c_0 + num_u * hw_ub * c_0
        burst_len_data = hw_ub * c_0 // cp_align_len
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr("w", offset=0),
                                    data.access_ptr('r',
                                                    offset=data_offset),
                                    0, 1, burst_len_data, 0, 0))

        dim_ele = hw_ub * c_0
        dim_ele_two = dim_ele * 2
        dim_ele_two_align = _ceil_fill(dim_ele_two, 16)
        dim_zu = dim_ele_two_align // 16

        one_begin = 0
        two_begin = ub_ele * float_size
        repeat_vconv = dim_zu
        src_stride_vconv = 1
        dst_stride_vconv = 16
        args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
               repeat_vconv, src_stride_vconv, dst_stride_vconv
        _vconv_two_not_align(args)

        with tvm_ib.for_range(0, c_i, name="num_c") as num_c:
            res_offset = num_c * 2 * 8
            ub_offset = num_c * hw_ub * 2 * 8
            n_burst = hw_ub
            burst_len = 2
            src_stride = c_0 * 2 - 2
            dst_stride = 0
            tvm_ib.emit(
                tvm.call_extern(data_ub.dtype, "copy_ubuf_to_ubuf",
                                data_ub.access_ptr(
                                    "w", offset=ub_offset),
                                data_res.access_ptr(
                                    "r", offset=res_offset),
                                0, n_burst, burst_len,
                                src_stride, dst_stride))

        out_ele = hw_ub * c_i
        out_ele_two = out_ele * 2
        out_ele_two_align = _ceil_fill(out_ele_two, 16)
        out_zu = out_ele_two_align // 16

        one_begin = 0
        two_begin = ub_ele * float_size
        repeat_vconv = out_zu
        src_stride_vconv = 16
        dst_stride_vconv = 1
        args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
               repeat_vconv, src_stride_vconv, dst_stride_vconv
        _vconv_two_not_align(args)

        with tvm_ib.for_range(0, c_i, name="num_ci") as num_ci:
            dst_offset = n_index * c_i * hw_i + num_ci * hw_i + num_u * hw_ub
            res_offset = num_ci * hw_ub
            burst_len_dst = hw_ub // cp_align_len
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr(
                                            "r", offset=res_offset),
                                        0, 1, burst_len_dst, 0, 0))

    with tvm_ib.if_scope(hw_mod > 0):
        hw_mod_align = _ceil_fill(hw_mod, cp_align_len)
        hw_mod_begin = hw_i - hw_mod_align

        data_offset = n_index * hw_i * c_0 + hw_mod_begin * c_0
        burst_len_data = hw_mod_align * c_0 // cp_align_len
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr("w", offset=0),
                                    data.access_ptr('r',
                                                    offset=data_offset),
                                    0, 1, burst_len_data, 0, 0))

        dim_ele = hw_mod_align * c_0
        dim_ele_two = dim_ele * 2
        dim_ele_two_align = _ceil_fill(dim_ele_two, 16)
        dim_zu = dim_ele_two_align // 16

        one_begin = 0
        two_begin = ub_ele * float_size
        repeat_vconv = dim_zu
        src_stride_vconv = 1
        dst_stride_vconv = 16
        args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
               repeat_vconv, src_stride_vconv, dst_stride_vconv
        _vconv_two_not_align(args)

        with tvm_ib.for_range(0, c_i, name="num_c") as num_c:
            res_offset = num_c * 2 * 8
            ub_offset = num_c * hw_mod_align * 2 * 8
            n_burst = hw_mod_align
            burst_len = 2
            src_stride = c_0 * 2 - 2
            dst_stride = 0
            tvm_ib.emit(
                tvm.call_extern(data_ub.dtype, "copy_ubuf_to_ubuf",
                                data_ub.access_ptr(
                                    "w", offset=ub_offset),
                                data_res.access_ptr(
                                    "r", offset=res_offset),
                                0, n_burst, burst_len,
                                src_stride, dst_stride))

        out_ele = hw_mod_align * c_i
        out_ele_two = out_ele * 2
        out_ele_two_align = _ceil_fill(out_ele_two, 16)
        out_zu = out_ele_two_align // 16

        one_begin = 0
        two_begin = ub_ele * float_size
        repeat_vconv = out_zu
        src_stride_vconv = 16
        dst_stride_vconv = 1
        args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
               repeat_vconv, src_stride_vconv, dst_stride_vconv
        _vconv_two_not_align(args)

        with tvm_ib.for_range(0, c_i, name="num_ci") as num_ci:
            dst_offset = n_index * c_i * hw_i + num_ci * hw_i + hw_mod_begin
            res_offset = num_ci * hw_mod_align
            burst_len_dst = hw_mod_align // cp_align_len
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr(
                                            "r", offset=res_offset),
                                        0, 1, burst_len_dst, 0, 0))


def _nchw_fp32_split_hw(dst, data):
    """
    function of making ir node builder for nchw fp32 split hw scene

    """
    tvm_ib = tvm.ir_builder.create()

    n_i, c_i, h_i, w_i = dst.shape
    c_0 = data.shape[4]
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 64
    one_unit_ele = cp_align_len * c_0 * 2 * cp_align_len
    ub_ele = ub_bytes // 2 // float_size

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)

    num_unit_ub = ub_ele // one_unit_ele
    hw_ub = num_unit_ub * cp_align_len
    hw_i = h_i * w_i
    ub_loop = hw_i // hw_ub
    hw_mod = hw_i % hw_ub

    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
               addr_array, addr_array_buf, block_index, device_core_num,\
               ub_ele, float_size, cp_align_len, c_0, c_i, hw_i, hw_ub,\
               hw_mod, ub_loop, num_g
        _func_nchw_fp32_split_hw(args)

    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   addr_array, addr_array_buf, block_index, device_core_num, \
                   ub_ele, float_size, cp_align_len, c_0, c_i, hw_i, hw_ub, \
                   hw_mod, ub_loop, group_index
            _func_nchw_fp32_split_hw(args)

    return tvm_ib.get()


def _move_nchw_fp32_split_hw(args):
    """
    function of nchw fp32 split hw scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
    addr_array, addr_array_buf, cp_align_len, float_size, ub_ele,\
    c_0, c_i, hw_i, n_index, hw_len, hw_before = args

    data_offset = n_index * hw_i * c_0 + hw_before * c_0
    burst_len_data = hw_len * c_0 // cp_align_len
    tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                data_ub.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=data_offset),
                                0, 1, burst_len_data, 0, 0))

    dim_ele = hw_len * c_0
    dim_ele_two = dim_ele * 2
    dim_ele_two_align = _ceil_fill(dim_ele_two, 16)
    dim_zu = dim_ele_two_align // 16

    one_begin = 0
    two_begin = ub_ele * float_size
    repeat_vconv = dim_zu
    src_stride_vconv = 1
    dst_stride_vconv = 16
    args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
           repeat_vconv, src_stride_vconv, dst_stride_vconv
    _vconv_two_not_align(args)

    with tvm_ib.for_range(0, c_i, name="num_c") as num_c:
        res_offset = num_c * 2 * 8
        ub_offset = num_c * hw_len * 2 * 8
        n_burst = hw_len
        burst_len = 2
        src_stride = c_0 * 2 - 2
        dst_stride = 0
        tvm_ib.emit(
            tvm.call_extern(data_ub.dtype, "copy_ubuf_to_ubuf",
                            data_ub.access_ptr(
                                "w", offset=ub_offset),
                            data_res.access_ptr(
                                "r", offset=res_offset),
                            0, n_burst, burst_len,
                            src_stride, dst_stride))

    out_ele = hw_len * c_i
    out_ele_two = out_ele * 2
    out_ele_two_align = _ceil_fill(out_ele_two, 16)
    out_zu = out_ele_two_align // 16

    one_begin = 0
    two_begin = ub_ele * float_size
    repeat_vconv = out_zu
    src_stride_vconv = 16
    dst_stride_vconv = 1
    args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
           repeat_vconv, src_stride_vconv, dst_stride_vconv
    _vconv_two_not_align(args)

    with tvm_ib.for_range(0, c_i, name="num_ci") as num_ci:
        dst_offset = n_index * c_i * hw_i + num_ci * hw_i + hw_before
        res_offset = num_ci * hw_len
        burst_len_dst = hw_len // cp_align_len
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr(
                                        "r", offset=res_offset),
                                    0, 1, burst_len_dst, 0, 0))


def _func_nchw_fp32_split_hw_fencore_nomod(args):
    """
    function of nchw fp32 split hw fencore nomod scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
    addr_array, addr_array_buf, block_index, device_core_num,\
    cp_align_len, float_size, ub_ele, c_0, c_i, fen_n, hw_i,\
    hw_ub, hw_mod, num_g = args

    all_index = num_g * device_core_num + block_index
    n_index = all_index // fen_n
    fen_index = all_index % fen_n

    hw_len = hw_ub
    hw_before = hw_ub * fen_index
    args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
           addr_array, addr_array_buf, cp_align_len, float_size, ub_ele, \
           c_0, c_i, hw_i, n_index, hw_len, hw_before
    _move_nchw_fp32_split_hw(args)


def _func_nchw_fp32_split_hw_fencore_mod(args):
    """
    function of nchw fp32 split hw fencore mod scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
    addr_array, addr_array_buf, block_index, device_core_num,\
    cp_align_len, float_size, ub_ele, c_0, c_i, fen_n, hw_i,\
    hw_ub, hw_mod, num_g = args

    all_index = num_g * device_core_num + block_index
    n_index = all_index // fen_n
    fen_index = all_index % fen_n

    with tvm_ib.if_scope(fen_index < fen_n - 1):
        hw_len = hw_ub
        hw_before = hw_ub * fen_index
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
               addr_array, addr_array_buf, cp_align_len, float_size, ub_ele,\
               c_0, c_i, hw_i, n_index, hw_len, hw_before
        _move_nchw_fp32_split_hw(args)

    with tvm_ib.else_scope():
        hw_mod_align = _ceil_fill(hw_mod, cp_align_len)
        hw_len = hw_mod_align
        hw_before = hw_i - hw_mod_align
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
               addr_array, addr_array_buf, cp_align_len, float_size, ub_ele,\
               c_0, c_i, hw_i, n_index, hw_len, hw_before
        _move_nchw_fp32_split_hw(args)


def _nchw_fp32_split_hw_fencore(dst, data):
    """
    function of making ir node builder for nchw fp32 split hw fencore scene

    """
    tvm_ib = tvm.ir_builder.create()

    n_i, c_i, h_i, w_i = dst.shape
    c_0 = data.shape[4]
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 64
    one_unit_ele = cp_align_len * c_0 * 2 * cp_align_len
    ub_ele = ub_bytes // 2 // float_size

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)

    num_unit_ub = ub_ele // one_unit_ele
    hw_ub = num_unit_ub * cp_align_len
    hw_i = h_i * w_i
    ub_loop = hw_i // hw_ub
    hw_mod = hw_i % hw_ub

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.if_scope(hw_mod > 0):
        fen_n = ub_loop + 1
        all_core = n_i * fen_n
        group_index = all_core // device_core_num
        group_mod = all_core % device_core_num

        with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
                   addr_array, addr_array_buf, block_index, device_core_num,\
                   cp_align_len, float_size, ub_ele, c_0, c_i, fen_n, hw_i,\
                   hw_ub, hw_mod, num_g
            _func_nchw_fp32_split_hw_fencore_mod(args)
        with tvm_ib.if_scope(group_mod > 0):
            with tvm_ib.if_scope(block_index < group_mod):
                args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
                       addr_array, addr_array_buf, block_index,\
                       device_core_num, cp_align_len, float_size, ub_ele,\
                       c_0, c_i, fen_n, hw_i,\
                       hw_ub, hw_mod, group_index
                _func_nchw_fp32_split_hw_fencore_mod(args)
    with tvm_ib.else_scope():
        fen_n = ub_loop
        all_core = n_i * fen_n
        group_index = all_core // device_core_num
        group_mod = all_core % device_core_num

        with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   addr_array, addr_array_buf, block_index, device_core_num, \
                   cp_align_len, float_size, ub_ele, c_0, c_i, fen_n, hw_i, \
                   hw_ub, hw_mod, num_g
            _func_nchw_fp32_split_hw_fencore_nomod(args)
        with tvm_ib.if_scope(group_mod > 0):
            with tvm_ib.if_scope(block_index < group_mod):
                args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                       addr_array, addr_array_buf, block_index, \
                       device_core_num, cp_align_len, float_size, ub_ele, \
                       c_0, c_i, fen_n, hw_i, \
                       hw_ub, hw_mod, group_index
                _func_nchw_fp32_split_hw_fencore_nomod(args)

    return tvm_ib.get()


def _func_nchw_fp32_sp1_small(args):
    """
    function of moving data for nchw fp32 sp1 small scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
    addr_array, addr_array_buf, cp_align_len, float_size,\
    device_core_num, block_index, ub_ele, c_0, c_i, hw_i, num_g = args

    n_index = num_g * device_core_num + block_index

    data_offset = n_index * hw_i * c_0
    burst_len_data = hw_i * c_0 // cp_align_len
    tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                data_ub.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=data_offset),
                                0, 1, burst_len_data, 0, 0))

    dim_ele = hw_i * c_0
    dim_ele_two = dim_ele * 2
    dim_ele_two_align = _ceil_fill(dim_ele_two, 16)
    dim_zu = dim_ele_two_align // 16

    one_begin = 0
    two_begin = ub_ele * float_size
    repeat_vconv = dim_zu
    src_stride_vconv = 1
    dst_stride_vconv = 16
    args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
           repeat_vconv, src_stride_vconv, dst_stride_vconv
    _vconv_two_not_align(args)

    with tvm_ib.for_range(0, c_i, name="num_c") as num_c:
        res_offset = num_c * 2 * 8
        ub_offset = num_c * hw_i * 2 * 8
        n_burst = hw_i
        burst_len = 2
        src_stride = c_0 * 2 - 2
        dst_stride = 0
        tvm_ib.emit(
            tvm.call_extern(data_ub.dtype, "copy_ubuf_to_ubuf",
                            data_ub.access_ptr(
                                "w", offset=ub_offset),
                            data_res.access_ptr(
                                "r", offset=res_offset),
                            0, n_burst, burst_len,
                            src_stride, dst_stride))

    out_ele = hw_i * c_i
    out_ele_two = out_ele * 2
    out_ele_two_align = _ceil_fill(out_ele_two, 16)
    out_zu = out_ele_two_align // 16

    one_begin = 0
    two_begin = ub_ele * float_size
    repeat_vconv = out_zu
    src_stride_vconv = 16
    dst_stride_vconv = 1
    args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
           repeat_vconv, src_stride_vconv, dst_stride_vconv
    _vconv_two_not_align(args)

    with tvm_ib.if_scope(out_ele % cp_align_len > 0):
        sub_ele = out_ele - cp_align_len
        dst_offset = n_index * hw_i * c_i
        burst_len_dst = _ceil_div(sub_ele, cp_align_len)
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr(
                                        "r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))
        with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_res.access_ptr(
                    'r', offset=(sub_ele + num_a))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_tail.dtype, "reg_mov",
                data_tail.access_ptr(
                    'w', offset=num_a),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr(
                                        'w',
                                        offset=(dst_offset + sub_ele)),
                                    data_tail.access_ptr(
                                        "r", offset=0),
                                    0, 1, 1, 0, 0))
    with tvm_ib.else_scope():
        dst_offset = n_index * hw_i * c_i
        burst_len_dst = out_ele // cp_align_len
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr(
                                        "r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))


def _nchw_fp32_sp1_small(dst, data):
    """
    function of making ir node builder for nchw fp32 sp1 small scene

    """
    tvm_ib = tvm.ir_builder.create()

    n_i, c_i, h_i, w_i = dst.shape
    c_0 = data.shape[4]
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 64
    ub_ele = ub_bytes // 2 // float_size

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)

    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)
    hw_i = h_i * w_i

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
               addr_array, addr_array_buf, cp_align_len, float_size,\
               device_core_num, block_index, ub_ele, c_0, c_i,\
               hw_i, num_g
        _func_nchw_fp32_sp1_small(args)

    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   addr_array, addr_array_buf, cp_align_len, float_size, \
                   device_core_num, block_index, ub_ele, c_0, c_i,\
                   hw_i, group_index
            _func_nchw_fp32_sp1_small(args)

    return tvm_ib.get()


def _func_nchw_fp32_sp1_little(args):
    """
    function of moving data for nchw fp32 sp1 little scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
    addr_array, addr_array_buf, device_core_num, block_index,\
    cp_align_len, float_size, ub_ele, num_dim_one_core,\
    num_g, num_dim_cur_core = args

    core_index = num_g * device_core_num + block_index
    num_dim_before = core_index * num_dim_one_core
    n_i, c_i, h_i, w_i = dst.shape
    c_0 = data.shape[4]
    hw_i = h_i * w_i
    in_ele = num_dim_cur_core * hw_i * c_0

    data_offset = num_dim_before * hw_i * c_0
    burst_len_data = _ceil_div(in_ele, cp_align_len)
    tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                data_ub.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=data_offset),
                                0, 1, burst_len_data, 0, 0))

    in_ele_two = in_ele * 2
    in_ele_two_align = _ceil_fill(in_ele_two, 16)
    in_zu = in_ele_two_align // 16

    one_begin = 0
    two_begin = ub_ele * float_size
    repeat_vconv = in_zu
    src_stride_vconv = 1
    dst_stride_vconv = 16
    args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
           repeat_vconv, src_stride_vconv, dst_stride_vconv
    _vconv_two_not_align(args)

    with tvm_ib.for_range(0, num_dim_cur_core, name="num_d") as num_d:
        with tvm_ib.for_range(0, c_i, name="num_c") as num_c:
            res_offset = num_d * hw_i * c_0 * 2 * 8 + num_c * 2 * 8
            ub_offset = num_d * hw_i * c_i * 2 * 8 + num_c * hw_i * 2 * 8
            n_burst = hw_i
            burst_len = 2
            src_stride = c_0 * 2 - 2
            dst_stride = 0
            tvm_ib.emit(
                tvm.call_extern(data_ub.dtype, "copy_ubuf_to_ubuf",
                                data_ub.access_ptr(
                                    "w", offset=ub_offset),
                                data_res.access_ptr(
                                    "r", offset=res_offset),
                                0, n_burst, burst_len,
                                src_stride, dst_stride))

    out_ele = num_dim_cur_core * hw_i * c_i
    out_ele_two = out_ele * 2
    out_ele_two_align = _ceil_fill(out_ele_two, 16)
    out_zu = out_ele_two_align // 16

    one_begin = 0
    two_begin = ub_ele * float_size
    repeat_vconv = out_zu
    src_stride_vconv = 16
    dst_stride_vconv = 1
    args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
           repeat_vconv, src_stride_vconv, dst_stride_vconv
    _vconv_two_not_align(args)

    with tvm_ib.if_scope(out_ele % cp_align_len > 0):
        with tvm_ib.if_scope(out_ele > cp_align_len):
            move_len = out_ele - cp_align_len
            dst_offset = num_dim_before * hw_i * c_i
            burst_len_dst = _ceil_div(move_len, cp_align_len)
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr(
                                            "r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))
            with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                tvm_ib.emit(tvm.call_extern(
                    data_res.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_res.access_ptr(
                        'r', offset=(move_len + num_a))
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    data_tail.access_ptr(
                        'w', offset=num_a),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr(
                                            'w',
                                            offset=(dst_offset + move_len)),
                                        data_tail.access_ptr(
                                            "r", offset=0),
                                        0, 1, 1, 0, 0))
        with tvm_ib.else_scope():
            dst_offset = num_dim_before * hw_i * c_i
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr(
                                            "r", offset=0),
                                        0, 1, 1, 0, 0))
    with tvm_ib.else_scope():
        dst_offset = num_dim_before * hw_i * c_i
        burst_len_dst = out_ele // cp_align_len
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr(
                                        "r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))


def _nchw_fp32_sp1_little(dst, data):
    """
    function of making ir node builder for nchw fp32 sp1 little scene

    """
    tvm_ib = tvm.ir_builder.create()

    n_i, c_i, h_i, w_i = dst.shape
    c_0 = data.shape[4]
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 64
    one_unit_ele = cp_align_len * c_0 * 2 * cp_align_len
    ub_ele = ub_bytes // 2 // float_size

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)

    num_unit_ub = ub_ele // one_unit_ele
    hw_ub = num_unit_ub * cp_align_len
    hw_i = h_i * w_i

    num_dim_one_core = hw_ub // hw_i
    full_core = n_i // num_dim_one_core
    num_dim_mod = n_i % num_dim_one_core

    group_index = full_core // device_core_num
    group_mod = full_core % device_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
               addr_array, addr_array_buf, device_core_num, block_index,\
               cp_align_len, float_size, ub_ele, num_dim_one_core,\
               num_g, num_dim_one_core
        _func_nchw_fp32_sp1_little(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   addr_array, addr_array_buf, device_core_num, block_index, \
                   cp_align_len, float_size, ub_ele, num_dim_one_core, \
                   group_index, num_dim_one_core
            _func_nchw_fp32_sp1_little(args)

    with tvm_ib.if_scope(num_dim_mod > 0):
        core_group = full_core // device_core_num
        core_block = full_core % device_core_num
        with tvm_ib.if_scope(tvm.all(block_index > (core_block - 1),
                                     block_index < (core_block + 1))):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   addr_array, addr_array_buf, device_core_num, block_index, \
                   cp_align_len, float_size, ub_ele, num_dim_one_core, \
                   core_group, num_dim_mod
            _func_nchw_fp32_sp1_little(args)

    return tvm_ib.get()


def _func_nhwc_fp16_sp1_little(args):
    """
    function of moving data for nhwc fp16 sp1 little scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
    addr_array, addr_array_buf, device_core_num, block_index, \
    cp_align_len, float_size, ub_ele, num_dim_ub, \
    num_g, num_dim_cur = args

    n_i, h_i, w_i, c_i = dst.shape
    c_1 = data.shape[1]
    c_0 = data.shape[4]
    hw_i = h_i * w_i

    core_before = num_g * device_core_num + block_index
    num_dim_before = core_before * num_dim_ub

    ele_in = num_dim_cur * c_1 * hw_i * c_0
    ele_in_div = _ceil_div(ele_in, 16)

    data_offset = num_dim_before * c_1 * hw_i * c_0
    burst_len_data = ele_in_div
    tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                data_ub.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=data_offset),
                                0, 1, burst_len_data, 0, 0))

    one_begin = 0
    two_begin = ub_ele * float_size
    repeat_vconv = ele_in_div
    src_stride_vconv = 1
    dst_stride_vconv = 16
    args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
           repeat_vconv, src_stride_vconv, dst_stride_vconv
    _vconv_two_not_align(args)

    with tvm_ib.for_range(0, num_dim_cur, name="num_d") as num_d:
        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            with tvm_ib.if_scope(num_c1 < c_1 - 1):
                res_offset = num_d * c_1 * hw_i * c_0 * 16\
                             + num_c1 * c_0 * hw_i * 16
                ub_offset = num_d * hw_i * c_i * 16\
                            + num_c1 * c_0 * 16
                n_burst = hw_i
                burst_len = c_0
                src_stride = 0
                dst_stride = c_i - c_0
                tvm_ib.emit(
                    tvm.call_extern(data_ub.dtype, "copy_ubuf_to_ubuf",
                                    data_ub.access_ptr(
                                        "w", offset=ub_offset),
                                    data_res.access_ptr(
                                        "r", offset=res_offset),
                                    0, n_burst, burst_len,
                                    src_stride, dst_stride))
            with tvm_ib.else_scope():
                c_mod = c_i - ((c_1 - 1) * c_0)
                res_offset = num_d * c_1 * hw_i * c_0 * 16\
                             + num_c1 * c_0 * hw_i * 16
                ub_offset = num_d * hw_i * c_i * 16 + num_c1 * c_0 * 16
                n_burst = hw_i
                burst_len = c_mod
                src_stride = c_0 - c_mod
                dst_stride = c_i - c_mod
                tvm_ib.emit(
                    tvm.call_extern(data_ub.dtype, "copy_ubuf_to_ubuf",
                                    data_ub.access_ptr(
                                        "w", offset=ub_offset),
                                    data_res.access_ptr(
                                        "r", offset=res_offset),
                                    0, n_burst, burst_len,
                                    src_stride, dst_stride))

    ele_out = num_dim_cur * hw_i * c_i
    ele_out_div = _ceil_div(ele_out, 16)

    one_begin = 0
    two_begin = ub_ele * float_size
    repeat_vconv = ele_out_div
    src_stride_vconv = 16
    dst_stride_vconv = 1
    args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
           repeat_vconv, src_stride_vconv, dst_stride_vconv
    _vconv_two_not_align(args)

    with tvm_ib.if_scope(ele_out % cp_align_len > 0):
        with tvm_ib.if_scope(ele_out > cp_align_len):
            move_len = ele_out - cp_align_len
            dst_offset = num_dim_before * hw_i * c_i
            burst_len_dst = _ceil_div(move_len, cp_align_len)
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr(
                                            "r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))
            with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                tvm_ib.emit(tvm.call_extern(
                    data_res.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_res.access_ptr(
                        'r', offset=(move_len + num_a))
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    data_tail.access_ptr(
                        'w', offset=num_a),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr(
                                            'w',
                                            offset=(dst_offset + move_len)),
                                        data_tail.access_ptr(
                                            "r", offset=0),
                                        0, 1, 1, 0, 0))
        with tvm_ib.else_scope():
            dst_offset = num_dim_before * hw_i * c_i
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr(
                                            "r", offset=0),
                                        0, 1, 1, 0, 0))
    with tvm_ib.else_scope():
        dst_offset = num_dim_before * hw_i * c_i
        burst_len_dst = ele_out // cp_align_len
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr(
                                        "r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))


def _nhwc_fp16_sp1_little(dst, data):
    """
    function of making ir node builder for nhwc fp16 sp1 little scene

    """
    tvm_ib = tvm.ir_builder.create()

    n_i, h_i, w_i, c_i = dst.shape
    c_0 = data.shape[4]
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 64
    ub_ele = ub_bytes // 2 // float_size

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)

    c_align = _ceil_fill(c_i, c_0)
    big_unit_ele = cp_align_len * c_align * cp_align_len
    num_unit_ub = ub_ele // big_unit_ele
    hw_ub = num_unit_ub * cp_align_len
    hw_i = h_i * w_i

    num_dim_ub = hw_ub // hw_i
    full_core = n_i // num_dim_ub
    num_dim_mod = n_i % num_dim_ub

    group_index = full_core // device_core_num
    group_mod = full_core % device_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
               addr_array, addr_array_buf, device_core_num, block_index,\
               cp_align_len, float_size, ub_ele, num_dim_ub,\
               num_g, num_dim_ub
        _func_nhwc_fp16_sp1_little(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   addr_array, addr_array_buf, device_core_num, block_index, \
                   cp_align_len, float_size, ub_ele, num_dim_ub, \
                   group_index, num_dim_ub
            _func_nhwc_fp16_sp1_little(args)

    with tvm_ib.if_scope(num_dim_mod > 0):
        core_group = full_core // device_core_num
        core_block = full_core % device_core_num

        with tvm_ib.if_scope(tvm.all(block_index > (core_block - 1),
                                     block_index < (core_block + 1))):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   addr_array, addr_array_buf, device_core_num, block_index, \
                   cp_align_len, float_size, ub_ele, num_dim_ub, \
                   core_group, num_dim_mod
            _func_nhwc_fp16_sp1_little(args)

    return tvm_ib.get()


def _func_nhwc_fp16_sp1_small(args):
    """
    function of moving data for nhwc fp16 sp1 small scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
    addr_array, addr_array_buf, cp_align_len, float_size, \
    device_core_num, block_index, ub_ele, c_0, c_i, \
    hw_i, num_g = args

    n_index = num_g * device_core_num + block_index
    _, c_1, h_i, w_i, _ = data.shape

    dim_ele_in = c_1 * h_i * w_i * c_0
    dim_ele_in_div = _ceil_div(dim_ele_in, 16)

    data_offset = n_index * dim_ele_in
    burst_len_data = _ceil_div(dim_ele_in, cp_align_len)
    tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                data_ub.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=data_offset),
                                0, 1, burst_len_data, 0, 0))

    one_begin = 0
    two_begin = ub_ele * float_size
    repeat_vconv = dim_ele_in_div
    src_stride_vconv = 1
    dst_stride_vconv = 16
    args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
           repeat_vconv, src_stride_vconv, dst_stride_vconv
    _vconv_two_not_align(args)

    with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
        with tvm_ib.if_scope(num_c1 < c_1 - 1):
            res_offset = num_c1 * c_0 * hw_i * 16
            ub_offset = num_c1 * c_0 * 16
            n_burst = hw_i
            burst_len = c_0
            src_stride = 0
            dst_stride = c_i - c_0
            tvm_ib.emit(
                tvm.call_extern(data_ub.dtype, "copy_ubuf_to_ubuf",
                                data_ub.access_ptr(
                                    "w", offset=ub_offset),
                                data_res.access_ptr(
                                    "r", offset=res_offset),
                                0, n_burst, burst_len,
                                src_stride, dst_stride))
        with tvm_ib.else_scope():
            c_mod = c_i - ((c_1 - 1) * c_0)
            res_offset = num_c1 * c_0 * hw_i * 16
            ub_offset = num_c1 * c_0 * 16
            n_burst = hw_i
            burst_len = c_mod
            src_stride = c_0 - c_mod
            dst_stride = c_i - c_mod
            tvm_ib.emit(
                tvm.call_extern(data_ub.dtype, "copy_ubuf_to_ubuf",
                                data_ub.access_ptr(
                                    "w", offset=ub_offset),
                                data_res.access_ptr(
                                    "r", offset=res_offset),
                                0, n_burst, burst_len,
                                src_stride, dst_stride))

    dim_ele_out = h_i * w_i * c_i
    dim_ele_out_div = _ceil_div(dim_ele_out, 16)
    one_begin = 0
    two_begin = ub_ele * float_size
    repeat_vconv = dim_ele_out_div
    src_stride_vconv = 16
    dst_stride_vconv = 1
    args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
           repeat_vconv, src_stride_vconv, dst_stride_vconv
    _vconv_two_not_align(args)

    with tvm_ib.if_scope(dim_ele_out % cp_align_len > 0):
        move_len = dim_ele_out - cp_align_len
        dst_offset = n_index * hw_i * c_i
        burst_len_dst = _ceil_div(move_len, cp_align_len)
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr(
                                        "r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))
        with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_res.access_ptr(
                    'r', offset=(move_len + num_a))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_tail.dtype, "reg_mov",
                data_tail.access_ptr(
                    'w', offset=num_a),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr(
                                        'w',
                                        offset=(dst_offset + move_len)),
                                    data_tail.access_ptr(
                                        "r", offset=0),
                                    0, 1, 1, 0, 0))
    with tvm_ib.else_scope():
        dst_offset = n_index * hw_i * c_i
        burst_len_dst = dim_ele_out // cp_align_len
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr(
                                        "r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))


def _nhwc_fp16_sp1_small(dst, data):
    """
    function of making ir node builder for nhwc fp16 sp1 small scene

    """
    tvm_ib = tvm.ir_builder.create()

    n_i, h_i, w_i, c_i = dst.shape
    c_0 = data.shape[4]
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 64
    ub_ele = ub_bytes // 2 // float_size

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)

    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)
    hw_i = h_i * w_i

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
               addr_array, addr_array_buf, cp_align_len, float_size,\
               device_core_num, block_index, ub_ele, c_0, c_i,\
               hw_i, num_g
        _func_nhwc_fp16_sp1_small(args)

    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   addr_array, addr_array_buf, cp_align_len, float_size, \
                   device_core_num, block_index, ub_ele, c_0, c_i,\
                   hw_i, group_index
            _func_nhwc_fp16_sp1_small(args)

    return tvm_ib.get()


def _move_nhwc_fp16_split_hw(args):
    """
    function of moving data for nhwc fp16 split hw scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
    addr_array, addr_array_buf, cp_align_len, float_size, ub_ele,\
    n_index, hw_before, hw_len = args

    _, c_1, h_i, w_i, c_0 = data.shape
    c_i = dst.shape[3]

    hw_i = h_i * w_i
    dim_ele_in = c_1 * h_i * w_i * c_0

    cur_ele_in = hw_len * c_0 * c_1
    cur_ele_in_div = _ceil_div(cur_ele_in, 16)

    data_offset = n_index * dim_ele_in + hw_before * c_0
    n_burst = c_1
    burst_len_data = hw_len
    src_stride = hw_i - hw_len
    args = tvm_ib, data, data_ub, data_offset, 0, n_burst,\
    burst_len_data, src_stride, 0, cp_align_len
    _func_gm_to_ub_align(args)

    one_begin = 0
    two_begin = ub_ele * float_size
    repeat_vconv = cur_ele_in_div
    src_stride_vconv = 1
    dst_stride_vconv = 16
    args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
           repeat_vconv, src_stride_vconv, dst_stride_vconv
    _vconv_two_not_align(args)

    with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
        with tvm_ib.if_scope(num_c1 < c_1 - 1):
            res_offset = num_c1 * c_0 * hw_len * 16
            ub_offset = num_c1 * c_0 * 16
            n_burst = hw_len
            burst_len = c_0
            src_stride = 0
            dst_stride = c_i - c_0
            tvm_ib.emit(
                tvm.call_extern(data_ub.dtype, "copy_ubuf_to_ubuf",
                                data_ub.access_ptr(
                                    "w", offset=ub_offset),
                                data_res.access_ptr(
                                    "r", offset=res_offset),
                                0, n_burst, burst_len,
                                src_stride, dst_stride))
        with tvm_ib.else_scope():
            c_mod = c_i - ((c_1 - 1) * c_0)
            res_offset = num_c1 * c_0 * hw_len * 16
            ub_offset = num_c1 * c_0 * 16
            n_burst = hw_len
            burst_len = c_mod
            src_stride = c_0 - c_mod
            dst_stride = c_i - c_mod
            tvm_ib.emit(
                tvm.call_extern(data_ub.dtype, "copy_ubuf_to_ubuf",
                                data_ub.access_ptr(
                                    "w", offset=ub_offset),
                                data_res.access_ptr(
                                    "r", offset=res_offset),
                                0, n_burst, burst_len,
                                src_stride, dst_stride))

    dim_ele_out = hw_len * c_i
    dim_ele_out_div = _ceil_div(dim_ele_out, 16)
    one_begin = 0
    two_begin = ub_ele * float_size
    repeat_vconv = dim_ele_out_div
    src_stride_vconv = 16
    dst_stride_vconv = 1
    args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
           repeat_vconv, src_stride_vconv, dst_stride_vconv
    _vconv_two_not_align(args)

    with tvm_ib.if_scope(dim_ele_out % cp_align_len > 0):
        move_len = dim_ele_out - cp_align_len
        dst_offset = n_index * hw_i * c_i + hw_before * c_i
        burst_len_dst = _ceil_div(move_len, cp_align_len)
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr(
                                        "r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))
        with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_res.access_ptr(
                    'r', offset=(move_len + num_a))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_tail.dtype, "reg_mov",
                data_tail.access_ptr(
                    'w', offset=num_a),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr(
                                        'w',
                                        offset=(dst_offset + move_len)),
                                    data_tail.access_ptr(
                                        "r", offset=0),
                                    0, 1, 1, 0, 0))
    with tvm_ib.else_scope():
        dst_offset = n_index * hw_i * c_i + hw_before * c_i
        burst_len_dst = dim_ele_out // cp_align_len
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr(
                                        "r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))


def _func_nhwc_fp16_split_hw(args):
    """
    function of making ir node builder for nhwc fp16 split hw scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
    addr_array, addr_array_buf, cp_align_len, float_size, \
    device_core_num, block_index, ub_ele, num_g = args

    n_i, h_i, w_i, c_i = dst.shape

    hw_i = h_i * w_i
    c_align = _ceil_fill(c_i, cp_align_len)
    one_unit_ele = cp_align_len * c_align * cp_align_len
    num_unit_ub = ub_ele // one_unit_ele
    hw_ub = num_unit_ub * cp_align_len

    ub_loop = hw_i // hw_ub
    hw_mod = hw_i % hw_ub
    n_index = num_g * device_core_num + block_index

    with tvm_ib.for_range(0, ub_loop, name="num_u") as num_u:
        hw_len = hw_ub
        hw_before = num_u * hw_ub
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
               addr_array, addr_array_buf, cp_align_len, float_size, ub_ele,\
               n_index, hw_before, hw_len
        _move_nhwc_fp16_split_hw(args)
    with tvm_ib.if_scope(hw_mod > 0):
        hw_len = _ceil_fill(hw_mod, cp_align_len)
        hw_before = hw_i - hw_len
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
               addr_array, addr_array_buf, cp_align_len, float_size, ub_ele, \
               n_index, hw_before, hw_len
        _move_nhwc_fp16_split_hw(args)


def _nhwc_fp16_split_hw(dst, data):
    """
    function of making ir node builder for nhwc fp16 split hw scene

    """
    tvm_ib = tvm.ir_builder.create()

    n_i, h_i, w_i, c_i = dst.shape
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 64
    ub_ele = ub_bytes // 2 // float_size

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)

    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
               addr_array, addr_array_buf, cp_align_len, float_size, \
               device_core_num, block_index, ub_ele, num_g
        _func_nhwc_fp16_split_hw(args)

    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   addr_array, addr_array_buf, cp_align_len, float_size, \
                   device_core_num, block_index, ub_ele, group_index
            _func_nhwc_fp16_split_hw(args)

    return tvm_ib.get()


def _func_nhwc_fp16_split_hw_mod(args):
    """
    function of nhwc fp16 split hw mod scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
    addr_array, addr_array_buf, device_core_num, block_index,\
    cp_align_len, float_size, ub_ele, hw_i, hw_ub, hw_mod, fen_n, num_g = args

    core_index = num_g * device_core_num + block_index
    n_index = core_index // fen_n
    fen_index = core_index % fen_n

    with tvm_ib.if_scope(fen_index < (fen_n - 1)):
        hw_len = hw_ub
        hw_before = fen_index * hw_ub
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
               addr_array, addr_array_buf, cp_align_len, float_size, ub_ele, \
               n_index, hw_before, hw_len
        _move_nhwc_fp16_split_hw(args)
    with tvm_ib.else_scope():
        hw_len = _ceil_fill(hw_mod, cp_align_len)
        hw_before = hw_i - hw_len
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
               addr_array, addr_array_buf, cp_align_len, float_size, ub_ele, \
               n_index, hw_before, hw_len
        _move_nhwc_fp16_split_hw(args)


def _func_nhwc_fp16_split_hw_nomod(args):
    """
    function of nhwc fp16 split hw nomod scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
    addr_array, addr_array_buf, device_core_num, block_index, \
    cp_align_len, float_size, ub_ele, hw_i, hw_ub, hw_mod, fen_n, num_g = args

    core_index = num_g * device_core_num + block_index
    n_index = core_index // fen_n
    fen_index = core_index % fen_n

    hw_len = hw_ub
    hw_before = fen_index * hw_ub
    args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
           addr_array, addr_array_buf, cp_align_len, float_size, ub_ele, \
           n_index, hw_before, hw_len
    _move_nhwc_fp16_split_hw(args)


def _nhwc_fp16_split_hw_fencore(dst, data):
    """
    function of nhwc fp16 split hw fencore scene

    """
    tvm_ib = tvm.ir_builder.create()

    n_i, h_i, w_i, c_i = dst.shape
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 64
    ub_ele = ub_bytes // 2 // float_size

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)

    hw_i = h_i * w_i
    c_align = _ceil_fill(c_i, cp_align_len)
    one_unit_ele = cp_align_len * c_align * cp_align_len
    num_unit_ub = ub_ele // one_unit_ele
    hw_ub = num_unit_ub * cp_align_len

    ub_loop = hw_i // hw_ub
    hw_mod = hw_i % hw_ub

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.if_scope(hw_mod > 0):
        fen_n = ub_loop + 1
        all_core = fen_n * n_i
        group_index = all_core // device_core_num
        group_mod = all_core % device_core_num

        with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
                   addr_array, addr_array_buf, device_core_num, block_index,\
                   cp_align_len, float_size, ub_ele, hw_i, hw_ub, hw_mod,\
                   fen_n, num_g
            _func_nhwc_fp16_split_hw_mod(args)
        with tvm_ib.if_scope(group_mod > 0):
            with tvm_ib.if_scope(block_index < group_mod):
                args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                       addr_array, addr_array_buf, device_core_num,\
                       block_index, cp_align_len, float_size, ub_ele,\
                       hw_i, hw_ub, hw_mod, fen_n, group_index
                _func_nhwc_fp16_split_hw_mod(args)

    with tvm_ib.else_scope():
        fen_n = ub_loop
        all_core = fen_n * n_i
        group_index = all_core // device_core_num
        group_mod = all_core % device_core_num

        with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   addr_array, addr_array_buf, device_core_num, block_index, \
                   cp_align_len, float_size, ub_ele, hw_i, hw_ub, hw_mod, \
                   fen_n, num_g
            _func_nhwc_fp16_split_hw_nomod(args)
        with tvm_ib.if_scope(group_mod > 0):
            with tvm_ib.if_scope(block_index < group_mod):
                args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                       addr_array, addr_array_buf, device_core_num,\
                       block_index, cp_align_len, float_size, ub_ele,\
                       hw_i, hw_ub, hw_mod, fen_n, group_index
                _func_nhwc_fp16_split_hw_nomod(args)

    return tvm_ib.get()


def _clean_ubuf(ib_, src, src_offset, dup_len):
    """
    function of cleaning space in UB

    """
    uint64_all_one = tvm.const(2**64 - 1, dtype="uint64")
    uint64_all_zero = tvm.const(0, dtype="uint64")
    dtype_factor = 32 // cce.cce_intrin.get_bit_len(src.dtype)
    dup_value = tvm.const(0.0, dtype=src.dtype)
    batch_cnt = 64

    if dup_len > 0:
        if src.dtype == "float16":
            ib_.emit(
                tvm.call_extern("uint64", 'set_vector_mask', uint64_all_one,
                                uint64_all_one))
        else:
            ib_.emit(
                tvm.call_extern("uint64", 'set_vector_mask', uint64_all_zero,
                                uint64_all_one))

        repeat = dup_len // (batch_cnt * dtype_factor)
        dup_left = dup_len % (batch_cnt * dtype_factor)
        if repeat >= 255:
            repeat_loop = (repeat + 255 - 1) // 255
            with ib_.for_range(0, repeat_loop) as i:
                with ib_.if_scope(i != repeat_loop - 1):
                    ib_.emit(
                        tvm.call_extern(
                            src.dtype, 'vector_dup',
                            src.access_ptr(
                                "rw",
                                offset=(255 * batch_cnt * dtype_factor) * i +
                                src_offset), dup_value, 255, 1, 1, 8, 8))
                with ib_.else_scope():
                    ib_.emit(
                        tvm.call_extern(
                            src.dtype, "vector_dup",
                            src.access_ptr(
                                "rw",
                                offset=(255 * batch_cnt * dtype_factor) * i +
                                src_offset), dup_value, repeat % 255, 1, 1, 8,
                            8))

        else:
            ib_.emit(
                tvm.call_extern(src.dtype, "vector_dup",
                                src.access_ptr("rw", offset=src_offset),
                                dup_value, repeat, 1, 1, 8, 8))

            if dup_left > 0:
                if dup_left > 64:
                    high_mask = tvm.const(2 ** (dup_left % 64) - 1,
                                          dtype="uint64")
                    ib_.emit(
                        tvm.call_extern("uint64", 'set_vector_mask',
                                        high_mask,
                                        uint64_all_one))
                elif 0 < dup_left <= 64:
                    low_mask = tvm.const(2 ** dup_left - 1, dtype="uint64")
                    ib_.emit(
                        tvm.call_extern("uint64", 'set_vector_mask',
                                        uint64_all_zero,
                                        low_mask))
                ib_.emit(
                    tvm.call_extern(src.dtype, "vector_dup",
                                    src.access_ptr(
                                        "rw",
                                        offset=(src_offset
                                                + repeat * batch_cnt
                                                * dtype_factor)),
                                    dup_value, 1, 1, 1, 8, 8))

        ib_.emit(
            tvm.call_extern("uint64", 'set_vector_mask', uint64_all_one,
                            uint64_all_one))


def _func_ub_to_gm_align(args):
    """
    function of moving data from ub to gm

    """
    tvm_ib, dst, data_res, dst_offset, res_offset, ori_nburst,\
    burst_len, src_stride, dst_stride, cp_align_len = args

    with tvm_ib.if_scope(ori_nburst > 0):
        with tvm_ib.if_scope(burst_len > 0):
            with tvm_ib.if_scope(burst_len <= 65535):
                with tvm_ib.if_scope(src_stride >= 0):
                    with tvm_ib.if_scope(src_stride <= 65535):
                        with tvm_ib.if_scope(dst_stride >= 0):
                            with tvm_ib.if_scope(dst_stride <= 65535):
                                with tvm_ib.if_scope(ori_nburst <= 4095):
                                    tvm_ib.emit(
                                        tvm.call_extern(
                                            dst.dtype,
                                            "copy_ubuf_to_gm",
                                            dst.access_ptr(
                                                'w', offset=dst_offset),
                                            data_res.access_ptr(
                                                "r", offset=res_offset),
                                            0, ori_nburst,
                                            burst_len,
                                            src_stride,
                                            dst_stride))
                                with tvm_ib.else_scope():
                                    n_burst = 4095
                                    c_cycle = ori_nburst // n_burst
                                    c_mod = ori_nburst % n_burst
                                    with tvm_ib.for_range(0, c_cycle,
                                                          name="num_cy")\
                                            as num_cy:
                                        dst_cur = dst_offset + (
                                            burst_len + dst_stride) \
                                                  * cp_align_len\
                                                  * n_burst * num_cy
                                        res_cur = res_offset + (
                                            burst_len + src_stride) \
                                                  * cp_align_len\
                                                  * n_burst * num_cy
                                        tvm_ib.emit(
                                            tvm.call_extern(
                                                dst.dtype,
                                                "copy_ubuf_to_gm",
                                                dst.access_ptr(
                                                    'w',
                                                    offset=dst_cur),
                                                data_res.access_ptr(
                                                    "r", offset=res_cur),
                                                0, n_burst,
                                                burst_len,
                                                src_stride,
                                                dst_stride))
                                    with tvm_ib.if_scope(c_mod > 0):
                                        dst_cur = dst_offset + (
                                            burst_len + dst_stride) \
                                                  * cp_align_len\
                                                  * n_burst * c_cycle
                                        res_cur = res_offset + (
                                            burst_len + src_stride) \
                                                  * cp_align_len\
                                                  * n_burst * c_cycle
                                        tvm_ib.emit(
                                            tvm.call_extern(
                                                dst.dtype,
                                                "copy_ubuf_to_gm",
                                                dst.access_ptr(
                                                    'w', offset=dst_cur),
                                                data_res.access_ptr(
                                                    "r", offset=res_cur),
                                                0, c_mod,
                                                burst_len,
                                                src_stride,
                                                dst_stride))
                            with tvm_ib.else_scope():
                                with tvm_ib.for_range(0, ori_nburst,
                                                      name="num_nb") as num_nb:
                                    dst_cur = dst_offset + (
                                        burst_len + dst_stride)\
                                              * cp_align_len * num_nb
                                    res_cur = res_offset + (
                                        burst_len + src_stride)\
                                              * cp_align_len * num_nb
                                    tvm_ib.emit(
                                        tvm.call_extern(
                                            dst.dtype,
                                            "copy_ubuf_to_gm",
                                            dst.access_ptr(
                                                'w', offset=dst_cur),
                                            data_res.access_ptr(
                                                "r", offset=res_cur),
                                            0, 1, burst_len,
                                            0, 0))


def _func_gm_to_ub_align(args):
    """
    function of moving data from data to data_ub

    """
    tvm_ib, data, data_ub, data_offset, ub_offset, ori_nburst,\
    burst_len, src_stride, dst_stride, cp_align_len = args

    with tvm_ib.if_scope(ori_nburst > 0):
        with tvm_ib.if_scope(burst_len > 0):
            with tvm_ib.if_scope(burst_len <= 65535):
                with tvm_ib.if_scope(src_stride >= 0):
                    with tvm_ib.if_scope(dst_stride >= 0):
                        with tvm_ib.if_scope(dst_stride <= 65535):
                            with tvm_ib.if_scope(src_stride <= 65535):
                                with tvm_ib.if_scope(ori_nburst <= 4095):
                                    tvm_ib.emit(
                                        tvm.call_extern(
                                            data_ub.dtype,
                                            "copy_gm_to_ubuf",
                                            data_ub.access_ptr(
                                                "w", offset=ub_offset),
                                            data.access_ptr(
                                                'r', offset=data_offset),
                                            0, ori_nburst,
                                            burst_len,
                                            src_stride, dst_stride))
                                with tvm_ib.else_scope():
                                    n_burst = 4095
                                    c_cycle = ori_nburst // n_burst
                                    c_mod = ori_nburst % n_burst
                                    with tvm_ib.for_range(0, c_cycle,
                                                          name="num_cy")\
                                            as num_cy:
                                        data_cur = data_offset + (
                                            burst_len + src_stride) \
                                                   * cp_align_len\
                                                   * n_burst * num_cy
                                        ub_cur = ub_offset + (
                                            burst_len + dst_stride) \
                                                 * cp_align_len\
                                                 * n_burst * num_cy
                                        tvm_ib.emit(
                                            tvm.call_extern(
                                                data_ub.dtype,
                                                "copy_gm_to_ubuf",
                                                data_ub.access_ptr(
                                                    "w",
                                                    offset=ub_cur),
                                                data.access_ptr(
                                                    'r',
                                                    offset=data_cur),
                                                0, n_burst,
                                                burst_len,
                                                src_stride,
                                                dst_stride))
                                    with tvm_ib.if_scope(c_mod > 0):
                                        data_cur = data_offset + (
                                            burst_len + src_stride) \
                                                   * cp_align_len\
                                                   * n_burst * c_cycle
                                        ub_cur = ub_offset + (
                                            burst_len + dst_stride) \
                                                 * cp_align_len\
                                                 * n_burst * c_cycle
                                        tvm_ib.emit(
                                            tvm.call_extern(
                                                data_ub.dtype,
                                                "copy_gm_to_ubuf",
                                                data_ub.access_ptr(
                                                    "w", offset=ub_cur),
                                                data.access_ptr(
                                                    'r', offset=data_cur),
                                                0, c_mod, burst_len,
                                                src_stride,
                                                dst_stride))
                            with tvm_ib.else_scope():
                                with tvm_ib.for_range(0, ori_nburst,
                                                      name="num_nb") as num_nb:
                                    data_cur = data_offset + (
                                        burst_len + src_stride)\
                                               * cp_align_len\
                                               * num_nb
                                    ub_cur = ub_offset + (
                                        burst_len + dst_stride)\
                                             * cp_align_len\
                                             * num_nb
                                    tvm_ib.emit(
                                        tvm.call_extern(
                                            data_ub.dtype,
                                            "copy_gm_to_ubuf",
                                            data_ub.access_ptr(
                                                "w", offset=ub_cur),
                                            data.access_ptr(
                                                'r', offset=data_cur),
                                            0, 1, burst_len,
                                            0, 0))


def _func_gm_to_ub(args):
    """
    function of moving data from data to data_ub

    """
    tvm_ib, param, data, data_ub, data_offset, ub_offset, ori_nburst,\
    burst_len, src_stride, dst_stride = args

    with tvm_ib.if_scope(ori_nburst > 0):
        with tvm_ib.if_scope(burst_len > 0):
            with tvm_ib.if_scope(burst_len <= 65535):
                with tvm_ib.if_scope(src_stride >= 0):
                    with tvm_ib.if_scope(dst_stride >= 0):
                        with tvm_ib.if_scope(dst_stride <= 65535):
                            with tvm_ib.if_scope(src_stride <= 65535):
                                with tvm_ib.if_scope(ori_nburst <= 4095):
                                    tvm_ib.emit(
                                        tvm.call_extern(data_ub.dtype,
                                                        "copy_gm_to_ubuf",
                                                        data_ub.access_ptr("w",\
                                                            offset=ub_offset),
                                                        data.access_ptr('r', \
                                                            offset=data_offset),
                                                        0, ori_nburst,
                                                        burst_len,
                                                        src_stride, dst_stride))
                                with tvm_ib.else_scope():
                                    n_burst = 4095
                                    c_cycle = ori_nburst // n_burst
                                    c_mod = ori_nburst % n_burst
                                    with tvm_ib.for_range(0, c_cycle, \
                                                    name="num_cy") as num_cy:
                                        data_cur = data_offset + (
                                            burst_len + src_stride) \
                                                   * param.get("cp_align_len")\
                                                   * n_burst * num_cy
                                        ub_cur = ub_offset + (
                                            burst_len + dst_stride) \
                                                 * param.get("cp_align_len")\
                                                 * n_burst * num_cy
                                        tvm_ib.emit(
                                            tvm.call_extern(data_ub.dtype,
                                                            "copy_gm_to_ubuf",
                                                            data_ub.access_ptr(
                                                                "w",
                                                                offset=ub_cur),\
                                        data.access_ptr('r', offset=data_cur), \
                                                            0, n_burst,
                                                            burst_len,
                                                            src_stride,
                                                            dst_stride))
                                    with tvm_ib.if_scope(c_mod > 0):
                                        data_cur = data_offset + (
                                            burst_len + src_stride) \
                                                   * param.get("cp_align_len")\
                                                   * n_burst * c_cycle
                                        ub_cur = ub_offset + (
                                            burst_len + dst_stride) \
                                                 * param.get("cp_align_len")\
                                                 * n_burst * c_cycle
                                        tvm_ib.emit(
                                            tvm.call_extern(data_ub.dtype,
                                                            "copy_gm_to_ubuf",
                                                            data_ub.access_ptr(
                                                                "w",
                                                                offset=ub_cur),\
                                        data.access_ptr('r', offset=data_cur),
                                                            0, c_mod, burst_len,
                                                            src_stride,
                                                            dst_stride))
                            with tvm_ib.else_scope():
                                with tvm_ib.for_range(0, ori_nburst,
                                                      name="num_nb") as num_nb:
                                    data_cur = data_offset + (
                                        burst_len + src_stride)\
                                            * param.get("cp_align_len") * num_nb
                                    ub_cur = ub_offset + (
                                        burst_len + dst_stride)\
                                            * param.get("cp_align_len") * num_nb
                                    tvm_ib.emit(
                                        tvm.call_extern(data_ub.dtype,
                                                        "copy_gm_to_ubuf", \
                                        data_ub.access_ptr("w", offset=ub_cur),\
                                        data.access_ptr('r', offset=data_cur),
                                                        0, 1, burst_len,
                                                        0, 0))


def _get_ir_branch_nhwc(dst_shape, dtype, shape_all):
    """
    judge ir node builder branch for nhwc scene

    """
    c_0 = 16
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    if shape_all > 380000000:
        ub_bytes = (UB_SIZE_B // 2 - 64) // 10*4
    else:
        ub_bytes = (UB_SIZE_B - 64) // 10*4
    c_i = dst_shape[3]
    c_bytes = _ceil_fill(c_i, c_0)*float_size
    if c_bytes <= ub_bytes:
        return "more_row_nhwc"
    return "split_row_nhwc"


def _get_ir_branch_fp16(dst_shape, dtype, shape_all):
    """
    judge ir node builder branch for nchw float16 scene

    """
    c_0 = 16
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    if shape_all > 380000000:
        ub_bytes = UB_SIZE_B // 2 - (cp_align_len*32)
    else:
        ub_bytes = UB_SIZE_B - cp_align_len*32
    ub_half = ub_bytes // 2
    _, _, h_i, w_i = dst_shape
    src_dim_space = _ceil_fill(h_i*w_i, c_0)*c_0
    src_dim_space_bytes = src_dim_space*float_size

    if src_dim_space_bytes <= ub_half:
        return "more_dim_fp16"
    return "split_dim_fp16"


def _get_ir_branch(src_shape, dtype, shape_all):
    """
    judge ir node builder branch for nchw float32 scene

    """
    c_0 = 16
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    if shape_all > 380000000:
        ub_bytes = UB_SIZE_B // 2 - 64 - (cp_align_len*32*2)
    else:
        ub_bytes = UB_SIZE_B - 64 - (cp_align_len*32*2)
    ub_half = ub_bytes // 2
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    dim_bytes = functools_reduce(lambda x, y: x*y, src_shape[2:])*float_size
    dim_plus_bytes = (functools_reduce(lambda x, y: x*y, src_shape[2:4])*(
        c_0 + 1))*float_size

    if dim_bytes <= ub_half:
        return "more_dim"
    if dim_plus_bytes <= ub_bytes:
        return "one_dim"
    return "split_dim"


def _get_factor(ele_zero, ele_cnt, total_ele, no_remainder):
    """
    get split factor for _tilling_one_axis function

    """
    split_factor = 1
    if no_remainder:
        for i in reversed(list(range(1, ele_zero))):
            if ele_zero % i == 0 and i*ele_cnt <= total_ele:
                split_factor = i
                break
    else:
        for i in reversed(list(range(1, ele_zero))):
            if i*ele_cnt <= total_ele:
                split_factor = i
                break

    return split_factor


def _tilling_axis(shape, dtype, no_remainder):
    """
    calculate the split parameters according to different shapes

    """
    ub_size_bytes = UB_SIZE_B - 32
    # 8 bit = 1byte, '8' below for this reason
    dtype_bytes_size = cce.cce_intrin.get_bit_len(dtype) // 8
    total_ele = ub_size_bytes // dtype_bytes_size
    split_axis = 0
    split_factor = 1

    for i, _ in enumerate(shape):
        ele_cnt = functools_reduce(lambda x, y: x*y, shape[i:])
        if ele_cnt <= total_ele:
            split_axis = i - 1
            split_factor = total_ele // ele_cnt
            if no_remainder and i == 1 and shape[0] % split_factor != 0:
                split_factor = _get_factor(shape[0], ele_cnt,
                                           total_ele, no_remainder)
            break
        if i == len(shape) - 1:
            if len(shape) == 1:
                split_axis = 0
                split_factor = _get_factor(shape[0], 1,
                                           total_ele, no_remainder)
            else:
                split_axis = i
                split_factor = total_ele
            break

    if split_axis < 0:
        split_axis = 0
        split_factor = shape[0]

    return split_axis, split_factor


# pylint: disable=locally-disabled,too-many-branches,unnecessary-lambda
def _move_for_one(c_i, dtype):
    """
    move data for n=1, h=1, w=1 scene

    """
    shape = [c_i]
    data = tvm.placeholder(shape, dtype=dtype, name='data')
    data_ub = tvm.compute(shape, lambda *i: data(*i), name='data_ub')
    res = tvm.compute(shape, lambda *i: data_ub(*i), name='res')
    sch = tvm.create_schedule(res.op)
    sch[data_ub].set_scope(cce.scope_ubuf)

    split_axis, split_factor = _tilling_axis(shape, dtype, True)
    if split_axis == 0:
        core_num = shape[split_axis] // split_factor
    else:
        core_num = shape[0]

    if core_num <= 65535:
        axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis],
                                                factor=split_factor)
        if split_axis == 0:
            sch[res].bind(axis_outer, tvm.thread_axis('blockIdx.x'))
        else:
            sch[res].bind(res.op.axis[0], tvm.thread_axis('blockIdx.x'))
    else:
        split_axis, split_factor = _tilling_axis(shape, dtype, False)
        axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis],
                                                factor=split_factor)

    sch[data_ub].compute_at(sch[res], axis_outer)
    sch[data_ub].emit_insn(data_ub.op.axis[split_axis], 'dma_copy')
    sch[res].emit_insn(axis_inner, 'dma_copy')

    tensor_list = [data, res]

    return sch, tensor_list


def _check_parameters(src, dst, src_format, dst_format, kernel_name):
    """
    check the parameters including src_shape, dst_shape,
    src_format, dst_format, dtype and kernel_name

    """
    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype")
    dtype_dst = dst.get("dtype")

    if src_format.lower() != "nc1hwc0":
        raise RuntimeError("src_format must be NC1HWC0 !")

    if dst_format.lower() != "nchw" and dst_format.lower() != "nhwc":
        raise RuntimeError("dst_format must be NCHW or NHWC!")

    util.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype, check_list)
    if dtype != dtype_dst:
        raise RuntimeError("dtype of src and dst are different !")

    util.check_shape_rule(src_shape, 5, 5)
    util.check_shape_rule(dst_shape, 4, 4)
    util.check_tensor_shape_size(src_shape)
    util.check_tensor_shape_size(dst_shape)

    if src_shape[4] != 16:
        raise RuntimeError(
            "the last dimension of src_shape is not 16, c0 must be 16 !")

    if dst_format.lower() == "nchw":
        if src_shape[0] != dst_shape[0] or src_shape[2] != dst_shape[2] or \
                src_shape[3] != dst_shape[3]:
            raise RuntimeError("the shape of src and dst not match, "
                               "the 1st,3rd,4th dimension of shape "
                               "must be the same !")
        c_dst = dst_shape[1]
    else:
        if src_shape[0] != dst_shape[0] or src_shape[2] != dst_shape[1] or \
                src_shape[3] != dst_shape[2]:
            raise RuntimeError("the shape of src and dst not match, "
                               "the 1st,3rd,4th dimension of src_shape and "
                               "the 1st,2nd,3rd dimension of dst_shape "
                               "must be the same !")
        c_dst = dst_shape[3]

    c_1 = src_shape[1]
    c_0 = src_shape[4]
    if not ((c_1 - 1)*c_0 < c_dst <= c_1*c_0):
        raise RuntimeError("c must be less than or equal to c1*c0,"
                           "and greater than ((c1 - 1)*c0 )!")


def _check_divide_sp_fp16(src_shape, dst_shape, dtype):
    """
    judge whether to use divide special fp16 branch

    """
    if dtype != "float16":
        return False
    c_1 = src_shape[1]
    c_0 = src_shape[4]
    c = dst_shape[3]
    c_mul = c_1*c_0
    if c_mul != c:
        return False

    if c_1 % 8 > 0:
        return False

    ub_bytes = ((UB_SIZE_B - 32) // 2)
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    ub_ele = ub_bytes // float_size
    n_i, h_i, w_i, _ = dst_shape
    dim_ele = h_i * w_i * c_0
    num_zu_one_core = (ub_ele // dim_ele) // 8

    if num_zu_one_core < 1:
        return False

    num_dim_one_core = num_zu_one_core*8
    num_zu = c_1 // num_dim_one_core
    dim_mod = c_1 % num_dim_one_core
    device_core_num = AICORE_NUM
    max_ir = 255

    if num_zu > 0:
        if num_dim_one_core > max_ir:
            return False

    if dim_mod > max_ir:
        return False

    if device_core_num == 2:
        if n_i < device_core_num and num_zu > 5:
            return False
    else:
        half_core = device_core_num // 2
        if n_i < half_core and num_zu > 5:
            return False

    return True


def _check_ci_align_nhwc_fp16(src_shape, dst_shape, dst_format, dtype):
    """
    check whether to use ci align nhwc fp16 branch

    """
    if dst_format.lower() != "nhwc":
        return False

    if dtype != "float16":
        return False

    n_i, h_i, w_i, c_i = dst_shape
    c_1 = src_shape[1]
    c_0 = src_shape[4]

    if c_i != c_1 * c_0:
        return False

    if c_i > 256:
        return False

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    ub_bytes = UB_SIZE_B - 1024
    ub_ele_a = ub_bytes // 2 // float_size

    if ub_ele_a < c_i:
        return False

    return True


def _check_ci_align_nchw_fp16(src_shape, dst_shape, dst_format, dtype):
    """
    check whether to use ci align nchw fp16 branch

    """
    if dst_format.lower() != "nchw":
        return False

    if dtype != "float16":
        return False

    n_i, c_i, h_i, w_i = dst_shape
    c_1 = src_shape[1]
    c_0 = src_shape[4]

    device_core_num = AICORE_NUM
    if n_i < device_core_num:
        return False

    if c_i != c_0 * c_1:
        return False

    if c_i > 64:
        return False

    col_len = h_i * w_i
    if col_len % 16 > 0:
        return False

    if col_len < 256 or col_len > 65536:
        return False

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    ub_bytes = UB_SIZE_B - 1024
    ub_ele = (ub_bytes // 2 // float_size // c_i) * c_i

    if ub_ele < 256:
        return False

    return True


def _check_move_full(shape_five, shape_nhwc, dst_format, dtype):
    if dst_format.lower() != "nhwc":
        return False

    if dtype != "float16":
        return False

    n_i, h_i, w_i, c_i = shape_nhwc
    c_0 = shape_five[4]

    if c_i != c_0:
        return False

    return True


def _check_vconv_fp16(shape_five, shape_nhwc, dst_format, dtype):
    """
    check whether vconv float16

    """
    if dst_format.lower() != "nhwc":
        return False

    if dtype != "float16":
        return False

    n_i, c_1, h_i, w_i, c_0 = shape_five

    if c_1 != 1:
        return False

    c_i = shape_nhwc[3]

    if c_i < 1 or c_i >= 16:
        return False

    if (h_i * w_i) % 256 > 0:
        return False

    ub_bytes = UB_SIZE_B - 32
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    ub_ele = ub_bytes // float_size
    dim_group_ele = 16 * 16 * 16 * 2

    if ub_ele < dim_group_ele:
        return False

    return True


def _get_new_shape_vconv_fp16(shape_five, shape_nhwc):
    """
    get new shape for vconv float16

    """
    n_i, c_1, h_i, w_i, c_0 = shape_five
    device_core_num = AICORE_NUM

    c_i = shape_nhwc[3]
    group = n_i*h_i*w_i // 256

    if group % device_core_num == 0:
        two_dim = group // device_core_num
        shape_five_new = [device_core_num, two_dim, 256, c_0]
        shape_nhwc_new = [device_core_num, two_dim, 256, c_i]
    else:
        two_dim = h_i * w_i // 256
        shape_five_new = [n_i, two_dim, 256, c_0]
        shape_nhwc_new = [n_i, two_dim, 256, c_i]

    return shape_five_new, shape_nhwc_new


def _choose_vconv_fp16(shape_nhwc, dtype):
    """
    choose the branch of vconv float16 scene

    """
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 32
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    ub_ele = ub_bytes // float_size
    dim_group_ele = 16 * 16 * 16 * 2
    num_dim_group_one_core = ub_ele // dim_group_ele

    n_i = shape_nhwc[0]
    num_dim_group_cur_block = shape_nhwc[1]

    if num_dim_group_cur_block > num_dim_group_one_core\
            and n_i < device_core_num:
        return "fencore"
    else:
        return "normal"


def _check_move_one_more_batch(dst_shape, dst_format, dtype):
    """
    check whether move one for more batch

    """
    if dst_format.lower() == "nchw":
        _, c_i, h_i, w_i = dst_shape
    else:
        _, h_i, w_i, c_i = dst_shape

    if h_i != 1 or w_i != 1:
        return False

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    if c_i < cp_align_len:
        return False

    return True


def _check_slice_branch(shape_five, shape_nhwc, dst_format, dtype):
    """
    check whether move one for more batch

    """
    if dtype != "float16":
        return False

    if dst_format.lower() != "nhwc":
        return False

    n_i, h_i, w_i, c_i = shape_nhwc
    c_1 = shape_five[1]

    if c_1 != 1:
        return False

    if c_i < 1 or c_i >= 16:
        return False

    hw_i = h_i * w_i
    if hw_i % 256 == 0:
        return False

    if hw_i <= 512:
        return False

    nhw_i = n_i * h_i * w_i
    if nhw_i > 30000000:
        return False

    return True


def _check_16_16_64_64(dst_shape, dst_format, dtype):
    """
    check whether 16_16_64_64 batch

    """
    if dtype != "float32":
        return False

    if dst_format.lower() != "nchw":
        return False

    if dst_shape != [16, 16, 64, 64] and dst_shape != [16, 32, 64, 64]:
        return False

    device_core_num = AICORE_NUM
    if device_core_num != 32:
        return False

    col_len = 64 * 64
    row_len = 16

    ub_bytes = UB_SIZE_B - 64
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_ele = (ub_bytes // 2 // float_size // 128) * 128

    dim_ele = col_len * row_len
    dim_ele_two = col_len * row_len * 2
    dim_ele_two_align = _ceil_fill(dim_ele_two, 16)
    space_ele = dim_ele_two_align * 8
    new_space_split_col_fp32 = row_len * 8 * 2 * 8

    if dim_ele > cp_align_len and space_ele > ub_ele \
            and new_space_split_col_fp32 <= ub_ele:
        return True

    return False


def _check_nchw_fp32_sp1(src_shape, dst_shape, dst_format, dtype):
    """
    check whether fp32 sp1 scene

    """
    if dtype != "float32":
        return False

    if dst_format.lower() != "nchw":
        return False

    c_1 = src_shape[1]
    if c_1 != 1:
        return False

    n_i, c_i, h_i, w_i = dst_shape
    if c_i < 1 or c_i > 16:
        return False

    c_0 = src_shape[4]
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = ub_bytes // 2 // float_size
    one_unit_ele = cp_align_len * c_0 * 2 * cp_align_len

    if one_unit_ele > ub_ele:
        return False

    return True


def _choose_nchw_fp32_sp1(src_shape, dst_shape, dtype):
    """
    choose branch of nchw fp32 sp1 scene

    """
    c_0 = src_shape[4]

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 64
    one_unit_ele = cp_align_len * c_0 * 2 * cp_align_len
    ub_ele = ub_bytes // 2 // float_size

    if one_unit_ele <= ub_ele:
        num_unit_ub = ub_ele // one_unit_ele
        hw_ub = num_unit_ub * cp_align_len
        n_i, c_i, h_i, w_i = dst_shape
        hw_i = h_i * w_i

        if hw_i <= hw_ub:
            if hw_i * c_i < cp_align_len:
                return "little"

            num_dim_ub = hw_ub // hw_i
            num_dim_group = num_dim_ub * device_core_num
            if num_dim_group <= n_i and num_dim_ub > 1:
                return "little"
            else:
                return "small"
        elif hw_i > hw_ub and n_i < device_core_num:
            return "split_hw_fencore"
        elif hw_i > hw_ub:
            return "split_hw"


def _check_nhwc_fp16_sp1(src_shape, dst_shape, dst_format, dtype):
    """
    check whether nhwc fp16 sp1 scene

    """
    if dst_format.lower() != "nhwc":
        return False

    if dtype != "float16":
        return False

    c_0 = src_shape[4]
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    one_unit_ele = cp_align_len * c_0 * cp_align_len
    ub_ele = ub_bytes // 2 // float_size

    if one_unit_ele > ub_ele:
        return False

    c_i = dst_shape[3]
    c_align = _ceil_fill(c_i, 16)
    big_unit_ele = cp_align_len * c_align * cp_align_len

    if big_unit_ele > ub_ele:
        return False

    return True


def _choose_nhwc_fp16_sp1(dst_shape, dtype):
    """
    choose branch of nhwc fp16 sp1 scene

    """
    n_i, h_i, w_i, c_i = dst_shape
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 64
    ub_ele = ub_bytes // 2 // float_size
    c_align = _ceil_fill(c_i, 16)
    big_unit_ele = cp_align_len * c_align * cp_align_len

    if big_unit_ele <= ub_ele:
        num_unit_ub = ub_ele // big_unit_ele
        hw_ub = num_unit_ub * cp_align_len
        hw_i = h_i * w_i

        if hw_i <= hw_ub:
            if hw_i * c_i < cp_align_len:
                return "little"

            num_dim_ub = hw_ub // hw_i
            num_dim_group = num_dim_ub * device_core_num
            if num_dim_group <= n_i and num_dim_ub > 1:
                return "little"
            else:
                return "small"

        elif hw_i > hw_ub and n_i < device_core_num:
            return "split_hw_fencore"
        elif hw_i > hw_ub:
            return "split_hw"


@util.check_input_type(dict, dict, str, str, str)
def five_2_four(src, dst, src_format, dst_format, kernel_name='five_2_four'):
    """
    algorithm: five_2_four
    calculating: change data format from NC1HWC0 to NCHW/NHWC

    Parameters
    ----------
    src: dict
        contains shape and dtype information of input tensor
    dst: dict
        contains shape and dtype information of output tensor
    src_format: str
        represents the format of input tensor, only support "NC1HWC0"
    dst_format: str
        represents the format of output tensor, only support "NCHW/NHWC"
    kernel_name: str
        cce kernel name, default value is "five_2_four"

    Returns
    -------
    None
    """
    _check_parameters(src, dst, src_format, dst_format, kernel_name)
    src_shape = list(src.get("shape"))
    dst_shape = list(dst.get("shape"))
    dtype = src.get("dtype")

    if dst_format.lower() == "nchw":
        n_i, c_i, h_i, w_i = dst_shape
    else:
        n_i, h_i, w_i, c_i = dst_shape

    if _check_slice_branch(src_shape, dst_shape, dst_format, dtype) and False:
        slice_in = src.copy()
        slice_out = dst.copy()
        n_i, _, h_i, w_i, c_0 = src_shape
        x_shape = [n_i, h_i, w_i, c_0]
        slice_in["shape"] = x_shape
        slice_out["shape"] = dst_shape
        begin = [0, 0, 0, 0]
        size = dst_shape
        slice_d(slice_in, slice_out, begin, size, kernel_name)
    else:
        if n_i == 1 and h_i == 1 and w_i == 1:
            sch, tensor_list = _move_for_one(c_i, dtype)
        elif _check_move_one_more_batch(dst_shape, dst_format, dtype):
            data = tvm.placeholder(src_shape, dtype=dtype, name="data")
            res = tvm.extern(dst_shape, [data],
                             lambda ins, outs: _move_one_more_batch(
                                 outs[0], ins[0], c_i),
                             name="res", dtype=dtype)
            tensor_list = [data, res]
            sch = tvm.create_schedule(res.op)
        else:
            max_dim = max(dst_shape)
            shape_all = functools_reduce(lambda x, y: x * y, src_shape[:])

            data = tvm.placeholder(src_shape, dtype=dtype, name="data")
            if dst_format.lower() == "nchw":
                if dtype == "float32":
                    if _check_16_16_64_64(dst_shape, dst_format, dtype):
                        if dst_shape == [16, 16, 64, 64]:
                            shape_new = [16, 64 * 64, 16]
                            shape_res = [16, 16, 64 * 64]
                        elif dst_shape == [16, 32, 64, 64]:
                            shape_new = [32, 64 * 64, 16]
                            shape_res = [32, 16, 64 * 64]

                        data = tvm.placeholder(shape_new, dtype=dtype,
                                               name="data")
                        res = tvm.extern(shape_res, [data],
                                         lambda ins,
                                                outs:
                                         _nchw_sp_16_16_64_64_fp32(
                                             outs[0], ins[0]),
                                         name="res", dtype=dtype)

                    elif _check_nchw_fp32_sp1(src_shape, dst_shape,
                                              dst_format, dtype):
                        data = tvm.placeholder(src_shape, dtype=dtype,
                                               name="data")
                        branch = _choose_nchw_fp32_sp1(src_shape, dst_shape,
                                                       dtype)
                        if branch == "little":
                            res = tvm.extern(dst_shape, [data],
                                             lambda ins,
                                                    outs:
                                             _nchw_fp32_sp1_little(
                                                 outs[0], ins[0]),
                                             name="res", dtype=dtype)
                        elif branch == "small":
                            res = tvm.extern(dst_shape, [data],
                                             lambda ins,
                                                    outs:
                                             _nchw_fp32_sp1_small(
                                                 outs[0], ins[0]),
                                             name="res", dtype=dtype)
                        elif branch == "split_hw":
                            res = tvm.extern(dst_shape, [data],
                                             lambda ins,
                                                    outs:
                                             _nchw_fp32_split_hw(
                                                 outs[0], ins[0]),
                                             name="res", dtype=dtype)
                        elif branch == "split_hw_fencore":
                            res = tvm.extern(dst_shape, [data],
                                             lambda ins,
                                                    outs:
                                             _nchw_fp32_split_hw_fencore(
                                                 outs[0], ins[0]),
                                             name="res", dtype=dtype)

                    else:
                        branch = _get_ir_branch(src_shape, dtype, shape_all)
                        if branch == "more_dim":
                            res = tvm.extern(dst_shape, [data],
                                             lambda ins, outs: _more_dim_ir(
                                                 outs[0], ins[0],
                                                 max_dim, shape_all),
                                             name="res", dtype=dtype)
                        elif branch == "one_dim":
                            res = tvm.extern(dst_shape, [data],
                                             lambda ins, outs: _one_dim_ir(
                                                 outs[0], ins[0],
                                                 max_dim, shape_all),
                                             name="res", dtype=dtype)
                        else:
                            res = tvm.extern(dst_shape, [data],
                                             lambda ins, outs: _split_dim_ir(
                                                 outs[0],
                                                 ins[0],
                                                 max_dim,
                                                 shape_all),
                                             name="res", dtype=dtype)
                else:
                    if _check_ci_align_nchw_fp16(src_shape, dst_shape,
                                                 dst_format,
                                                 dtype):
                        res = tvm.extern(dst_shape, [data],
                                         lambda ins, outs: _ci_align_nchw_fp16(
                                             outs[0],
                                             ins[0]),
                                         name="res", dtype=dtype)
                    else:
                        branch_fp16 = _get_ir_branch_fp16(dst_shape, dtype,
                                                          shape_all)
                        if branch_fp16 == "more_dim_fp16":
                            res = tvm.extern(dst_shape, [data],
                                             lambda ins, outs:
                                             _more_dim_ir_fp16(
                                                 outs[0],
                                                 ins[0],
                                                 max_dim,
                                                 shape_all),
                                             name="res", dtype=dtype)
                        else:
                            res = tvm.extern(dst_shape, [data],
                                             lambda ins,
                                                    outs:
                                             _split_dim_ir_fp16(
                                                 outs[0], ins[0], max_dim,
                                                 shape_all),
                                             name="res", dtype=dtype)
            else:
                mark, core_num, base_shape = _get_ir_branch_sp_nhwc(src_shape,
                                                                    dtype)

                if mark != "inconformity":
                    if mark == "less_ub":
                        res = tvm.extern(dst_shape, [data], \
                                         lambda ins, outs: _less_ub_ir_nhwc(
                                             outs[0], ins[0], \
                                             base_shape, core_num),
                                         name="res", dtype=dtype)
                    else:
                        res = tvm.extern(dst_shape, [data], \
                                         lambda ins, outs: _more_ub_ir_nhwc(
                                             outs[0], ins[0], \
                                             base_shape, core_num),
                                         name="res", dtype=dtype)
                elif _check_divide_sp_fp16(src_shape, dst_shape, dtype):
                    res = tvm.extern(dst_shape, [data],
                                     lambda ins, outs: _nhwc_adds_fp16_ir(
                                         outs[0], ins[0]),
                                     name="res", dtype=dtype)
                elif _check_move_full(src_shape, dst_shape, dst_format, dtype):
                    n_i, h_i, w_i, _ = dst_shape
                    device_core_num = AICORE_NUM
                    if n_i == 1 and ((h_i * w_i) % device_core_num == 0):
                        n_new = device_core_num
                        hw = h_i * w_i // device_core_num
                        h_new = 1
                        w_new = hw
                        dst_shape[0] = n_new
                        dst_shape[1] = h_new
                        dst_shape[2] = w_new
                        src_shape[0] = n_new
                        src_shape[2] = h_new
                        src_shape[3] = w_new
                    data = tvm.placeholder(src_shape, dtype=dtype, name="data")
                    res = tvm.extern(dst_shape, [data],
                                     lambda ins, outs: _move_full(outs[0],
                                                                  ins[0]),
                                     name="res", dtype=dtype)

                elif _check_vconv_fp16(src_shape, dst_shape,
                                       dst_format, dtype):
                    shape_five_new, shape_nhwc_new = _get_new_shape_vconv_fp16(
                        src_shape, dst_shape)
                    data = tvm.placeholder(shape_five_new, dtype=dtype,
                                           name="data")
                    branch = _choose_vconv_fp16(shape_nhwc_new, dtype)
                    if branch == "fencore":
                        res = tvm.extern(shape_nhwc_new, [data],
                                         lambda ins, outs:
                                         _move_vconv_fp16_new_fencore(
                                             outs[0], ins[0]),
                                         name="res", dtype=dtype)
                    else:
                        res = tvm.extern(shape_nhwc_new, [data],
                                         lambda ins, outs:
                                         _move_vconv_fp16_new(
                                             outs[0], ins[0]),
                                         name="res", dtype=dtype)

                elif _check_ci_align_nhwc_fp16(src_shape, dst_shape,
                                               dst_format, dtype):
                    n_i, h_i, w_i, _ = dst_shape
                    device_core_num = AICORE_NUM
                    if n_i == 1 and ((h_i * w_i) % device_core_num == 0):
                        res = tvm.extern(dst_shape, [data],
                                         lambda ins, outs:
                                         _ci_align_nhwc_fp16_one(
                                             outs[0],
                                             ins[0]),
                                         name="res", dtype=dtype)
                    else:
                        res = tvm.extern(dst_shape, [data],
                                         lambda ins, outs: _ci_align_nhwc_fp16(
                                             outs[0],
                                             ins[0]),
                                         name="res", dtype=dtype)

                elif _check_nhwc_fp16_sp1(src_shape, dst_shape,
                                          dst_format, dtype):
                    data = tvm.placeholder(src_shape, dtype=dtype,
                                           name="data")
                    branch = _choose_nhwc_fp16_sp1(dst_shape, dtype)

                    if branch == "little":
                        res = tvm.extern(dst_shape, [data],
                                         lambda ins,
                                                outs:
                                         _nhwc_fp16_sp1_little(
                                             outs[0], ins[0]),
                                         name="res", dtype=dtype)
                    elif branch == "small":
                        res = tvm.extern(dst_shape, [data],
                                         lambda ins,
                                                outs:
                                         _nhwc_fp16_sp1_small(
                                             outs[0], ins[0]),
                                         name="res", dtype=dtype)
                    elif branch == "split_hw_fencore":
                        res = tvm.extern(dst_shape, [data],
                                         lambda ins,
                                                outs:
                                         _nhwc_fp16_split_hw_fencore(
                                             outs[0], ins[0]),
                                         name="res", dtype=dtype)
                    elif branch == "split_hw":
                        res = tvm.extern(dst_shape, [data],
                                         lambda ins,
                                                outs:
                                         _nhwc_fp16_split_hw(
                                             outs[0], ins[0]),
                                         name="res", dtype=dtype)

                else:
                    branch = _get_ir_branch_nhwc(dst_shape, dtype, shape_all)
                    if branch == "more_row_nhwc":
                        res = tvm.extern(dst_shape, [data],
                                         lambda ins, outs: _more_row_ir_nhwc(
                                             outs[0],
                                             ins[0],
                                             max_dim,
                                             shape_all),
                                         name="res", dtype=dtype)
                    else:
                        res = tvm.extern(dst_shape, [data],
                                         lambda ins, outs: _split_row_ir_nhwc(
                                             outs[0], ins[0], max_dim,
                                             shape_all),
                                         name="res", dtype=dtype)

            tensor_list = [data, res]
            sch = tvm.create_schedule(res.op)

        with build_config:
            tvm.build(sch, tensor_list, "cce", name=kernel_name)
