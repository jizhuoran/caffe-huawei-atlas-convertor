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

zn_2_nchw
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from te import platform as cce
from te import tvm
from te.platform.cce_build import build_config
import te.platform.cce_params as cce_params
from topi.cce import util

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


# pylint: disable=locally-disabled,too-many-locals
def _get_param_more_row(tvm_ib, src_shape, dtype):
    """
    calculate parameters for more row ir builder make function

    """
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_ele = ((UB_SIZE_B - 32) // 2) // float_size
    _, n_no, n_ni, c_0 = src_shape
    row_ele = n_no*n_ni*c_0

    num_row_one_core = ub_ele // row_ele
    num_row_one_group = num_row_one_core*device_core_num
    num_row_in_data = src_shape[0]
    num_group_index = num_row_in_data // num_row_one_group
    num_group_mod = num_row_in_data % num_row_one_group

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    param_map = {"num_group_index": num_group_index,
                 "num_group_mod": num_group_mod,
                 "row_ele": row_ele,
                 "float_size": float_size,
                 "cp_align_len": cp_align_len,
                 "num_row_one_core": num_row_one_core,
                 "num_row_one_group": num_row_one_group,
                 "block_index": block_index}

    return param_map


# pylint: disable=locally-disabled,too-many-statements
def _func_more_row(args):
    """
    function of moving data for more row scene

    """
    tvm_ib, param, data, dst, data_ub, data_res, data_tail, reg, reg_addr,\
    num_g, num_row_cur_core, c_0 = args

    _, n_no, n_ni, c_0 = data.shape
    row_ele = n_no*n_ni*c_0
    h_i, w_i, c_i, n_i = dst.shape
    c_1 = _ceil_div(c_i, c_0)
    h_w = h_i*w_i
    num_row_before_core = num_g*param.get("num_row_one_group")\
                          + param.get("block_index")\
                          * param.get("num_row_one_core")
    num_hw_dst_before_core = num_row_before_core // c_1
    num_c0_dst_cur_hw_before = num_row_before_core % c_1
    num_c0_dst_cur_hw = c_1 - num_c0_dst_cur_hw_before
    reg_count = 8

    with tvm_ib.if_scope(num_row_cur_core <= num_c0_dst_cur_hw):
        data_offset = num_c0_dst_cur_hw_before*h_w*row_ele\
                      + num_hw_dst_before_core*row_ele
        n_burst = num_row_cur_core
        burst_len_data = _ceil_div(row_ele, param.get("cp_align_len"))
        src_stride = _ceil_div((h_w - 1)*row_ele, param.get("cp_align_len"))
        args = tvm_ib, param, data, data_ub, data_offset, 0, n_burst,\
               burst_len_data, src_stride, 0
        _func_gm_to_ub(args)

        c_t = tvm.min((num_c0_dst_cur_hw_before + 1) * c_0, c_i)
        c_cur = c_t - (num_c0_dst_cur_hw_before*c_0)
        with tvm_ib.for_range(0, num_row_cur_core, name="num_tr") as num_tr:
            with tvm_ib.for_range(0, c_cur, name="num_c")  as num_c:
                with tvm_ib.for_range(0, n_no, name="num_no") as num_no:
                    n_t = tvm.min((num_no + 1)*n_ni, n_i)
                    n_cur = n_t - num_no*n_ni
                    with tvm_ib.if_scope(n_cur % reg_count == 0):
                        n_cur_times_8 = n_cur // reg_count
                        reg_list = [n for n in range(reg_count)]
                        with tvm_ib.for_range(0, n_cur_times_8,
                                              name="num_nc") as num_nc:
                            for reg_idx in reg_list:
                                tvm_ib.emit(tvm.call_extern(
                                    data_ub.dtype,
                                    "reg_mov",
                                    tvm.call_extern(reg.dtype,
                                                    "reg",
                                                    reg[reg_idx]),
                                    data_ub.access_ptr(
                                        'r',
                                        offset=(num_tr * row_ele +
                                                num_no * n_ni * c_0 +
                                                (reg_idx +
                                                 num_nc * reg_count) * c_0 +
                                                num_c))
                                ))

                            for reg_idx in reg_list:
                                tvm_ib.emit(tvm.call_extern(
                                    data_res.dtype,
                                    "reg_mov",
                                    data_res.access_ptr(
                                        'w',
                                        offset=(num_tr * c_0 * n_i +
                                                num_c * n_i +
                                                num_no * n_ni +
                                                (reg_idx +
                                                 num_nc * reg_count))),
                                    tvm.call_extern(reg.dtype,
                                                    "reg",
                                                    reg[reg_idx])
                                ))
                    with tvm_ib.else_scope():
                        with tvm_ib.for_range(0, n_cur,
                                              name="num_nc") as num_nc:
                            tvm_ib.emit(tvm.call_extern(
                                data_ub.dtype, "reg_mov",
                                tvm.call_extern(reg.dtype, "reg",
                                                reg[0]),
                                data_ub.access_ptr(
                                    'r',
                                    offset=(
                                        num_tr * row_ele +
                                        num_no * n_ni * c_0 +
                                        num_nc * c_0 + num_c))
                            ))

                            tvm_ib.emit(tvm.call_extern(
                                data_res.dtype, "reg_mov",
                                data_res.access_ptr(
                                    'w',
                                    offset=(
                                        num_tr * c_0 * n_i +
                                        num_c * n_i +
                                        num_no * n_ni +
                                        num_nc)),
                                tvm.call_extern(reg.dtype, "reg",
                                                reg[0])
                            ))

        c_t = tvm.min((num_c0_dst_cur_hw_before+num_row_cur_core)*c_0, c_i)
        c_cur = c_t - (num_c0_dst_cur_hw_before * c_0)
        total_len = c_cur * n_i
        reg_addr[5] = total_len
        dst_offset = num_hw_dst_before_core*c_i*n_i\
                     + num_c0_dst_cur_hw_before*c_0*n_i
        args = tvm_ib, param, dst, data_res, data_tail, reg, reg_addr, 0, 0,\
               dst_offset, reg_addr[5]
        _res_to_gm_more_row(args)
    with tvm_ib.if_scope(num_row_cur_core > num_c0_dst_cur_hw):
        num_c0_head = num_c0_dst_cur_hw
        num_row_after = num_row_cur_core - num_c0_head
        reg_addr[2] = num_row_after
        num_g_mid = reg_addr[2] // c_1
        num_c0_tail = reg_addr[2] % c_1

        # gm to ub to ub_res
        with tvm_ib.if_scope(num_c0_head > 0):
            data_offset = num_c0_dst_cur_hw_before * h_w * row_ele\
                          + num_hw_dst_before_core * row_ele
            n_burst = num_c0_head
            burst_len_data = _ceil_div(row_ele, param.get("cp_align_len"))
            src_stride = _ceil_div((h_w - 1) * row_ele,
                                   param.get("cp_align_len"))
            args = tvm_ib, param, data, data_ub, data_offset, 0, n_burst,\
                   burst_len_data, src_stride, 0
            _func_gm_to_ub(args)
            # ub to ub_res
            with tvm_ib.for_range(0, num_c0_head, name="num_c0") as num_c0:
                c_t = tvm.min((num_c0_dst_cur_hw_before + num_c0 + 1) * c_0,
                              c_i)
                c_cur = c_t - (num_c0_dst_cur_hw_before + num_c0)*c_0
                with tvm_ib.for_range(0, c_cur, name="num_cr") as num_cr:
                    with tvm_ib.for_range(0, n_no, name="num_no") as num_no:
                        n_t = tvm.min((num_no + 1) * n_ni, n_i)
                        n_cur = n_t - num_no*n_ni
                        with tvm_ib.for_range(0, n_cur, name="num_nc")\
                                as num_nc:
                            tvm_ib.emit(tvm.call_extern(
                                data_ub.dtype, "reg_mov",
                                tvm.call_extern(reg.dtype, "reg", reg[0]),
                                data_ub.access_ptr('r',
                                                   offset=(num_c0*row_ele
                                                           + num_no*n_ni*c_0
                                                           + num_nc*c_0
                                                           + num_cr))
                            ))
                            tvm_ib.emit(tvm.call_extern(
                                data_res.dtype, "reg_mov",
                                data_res.access_ptr('w', offset=(
                                    num_c0*c_0*n_i + num_cr*n_i
                                    + num_no*n_ni + num_nc)),
                                tvm.call_extern(reg.dtype, "reg", reg[0])
                            ))

        with tvm_ib.if_scope(num_g_mid > 0):
            num_row_before_mid = num_row_before_core + num_c0_head
            reg_addr[3] = num_row_before_mid
            num_hw_dst_before_mid = reg_addr[3] // c_1
            n_burst = c_1
            burst_len_data = _ceil_div(row_ele, param.get("cp_align_len"))
            src_stride = _ceil_div((h_w - 1) * row_ele,
                                   param.get("cp_align_len"))
            ub_offset_mid_begin = num_c0_head*row_ele
            with tvm_ib.for_range(0, num_g_mid, name="num_mg") as num_mg:
                data_offset = (num_hw_dst_before_mid + num_mg) * row_ele
                ub_offset = ub_offset_mid_begin + num_mg*c_1*row_ele
                args = tvm_ib, param, data, data_ub, data_offset, ub_offset,\
                       n_burst, burst_len_data, src_stride, 0
                _func_gm_to_ub(args)

            # ub to ub_res
            res_offset_mid_begin = c_i*n_i - num_c0_dst_cur_hw_before*c_0*n_i
            with tvm_ib.for_range(0, num_g_mid, name="num_mg") as num_mg:
                with tvm_ib.for_range(0, c_i, name="num_ci") as num_ci:
                    c_t = tvm.min((num_ci + 1) * c_0, c_i)
                    c_cur = c_t - num_ci*c_0
                    with tvm_ib.for_range(0, c_cur, name="num_cr") as num_cr:
                        with tvm_ib.for_range(0, n_no, name="num_no") as num_no:
                            n_t = tvm.min((num_no + 1)*n_ni, n_i)
                            n_cur = n_t - num_no*n_ni
                            with tvm_ib.for_range(0, n_cur, name="num_nc")\
                                    as num_nc:
                                tvm_ib.emit(tvm.call_extern(
                                    data_ub.dtype, "reg_mov",
                                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                                    data_ub.access_ptr(
                                        'r',
                                        offset=(ub_offset_mid_begin
                                                + num_mg*c_1*row_ele +
                                                num_ci*row_ele + num_no*n_ni*c_0
                                                + num_nc*c_0 + num_cr))
                                ))
                                tvm_ib.emit(tvm.call_extern(
                                    data_res.dtype, "reg_mov",
                                    data_res.access_ptr(
                                        'w',
                                        offset=(res_offset_mid_begin
                                                + num_mg*c_i*n_i
                                                + num_ci*c_0*n_i
                                                + num_cr*n_i
                                                + num_no*n_ni + num_nc)),
                                    tvm.call_extern(reg.dtype, "reg", reg[0])
                                ))

        with tvm_ib.if_scope(num_c0_tail > 0):
            num_row_before_tail = num_row_before_core + num_c0_head\
                                  + num_g_mid*c_1
            reg_addr[4] = num_row_before_tail
            num_hw_dst_before_tail = reg_addr[4] // c_1

            ub_offset_tail_begin = (num_c0_head + num_g_mid*c_1)*row_ele
            data_offset = num_hw_dst_before_tail*row_ele
            n_burst = num_c0_tail
            burst_len_data = _ceil_div(row_ele, param.get("cp_align_len"))
            src_stride = _ceil_div((h_w - 1) * row_ele,
                                   param.get("cp_align_len"))
            args = tvm_ib, param, data, data_ub, data_offset,\
                   ub_offset_tail_begin, n_burst, burst_len_data, src_stride, 0
            _func_gm_to_ub(args)

            # ub to ub_res
            res_offset_tail_begin = c_i*n_i - num_c0_dst_cur_hw_before*c_0*n_i\
                                    + num_g_mid*n_i*c_i
            with tvm_ib.for_range(0, num_c0_tail, name="num_tc") as num_tc:
                c_t = tvm.min((num_tc + 1) * c_0, c_i)
                c_cur = c_t - num_tc*c_0
                with tvm_ib.for_range(0, c_cur, name="num_cr") as num_cr:
                    with tvm_ib.for_range(0, n_no, name="num_no") as num_no:
                        n_t = tvm.min((num_no + 1)*n_ni, n_i)
                        n_cur = n_t - num_no*n_ni
                        with tvm_ib.for_range(0, n_cur, name="num_nc")\
                                as num_nc:
                            tvm_ib.emit(tvm.call_extern(
                                data_ub.dtype, "reg_mov",
                                tvm.call_extern(reg.dtype, "reg", reg[0]),
                                data_ub.access_ptr(
                                    'r',
                                    offset=(ub_offset_tail_begin
                                            + num_tc*row_ele
                                            + num_no*n_ni*c_0
                                            + num_nc*c_0 + num_cr))
                            ))
                            tvm_ib.emit(tvm.call_extern(
                                data_res.dtype, "reg_mov",
                                data_res.access_ptr('w', offset=(
                                    res_offset_tail_begin +
                                    num_tc*c_0*n_i + num_cr * n_i
                                    + num_no*n_ni + num_nc)),
                                tvm.call_extern(reg.dtype, "reg", reg[0])
                            ))
        # ub_res to dst
        total_len = c_i*n_i - num_c0_dst_cur_hw_before*c_0*n_i\
                    + num_g_mid*n_i*c_i + num_c0_tail*c_0*n_i
        reg_addr[6] = total_len
        dst_offset = num_hw_dst_before_core*c_i*n_i\
                     + num_c0_dst_cur_hw_before*c_0*n_i
        args = tvm_ib, param, dst, data_res, data_tail, reg, reg_addr, 1, 0,\
               dst_offset, reg_addr[6]
        _res_to_gm_more_row(args)


def _res_to_gm_more_row(args):
    """
    function of moving data from data_res(UB) to dst(GM) for more row scene

    """
    tvm_ib, param, dst, data_res, data_tail, reg, reg_addr, index, res_offset,\
    dst_offset, total_len = args
    reg_count = 8

    with tvm_ib.if_scope(total_len % param.get("cp_align_len") > 0):
        with tvm_ib.if_scope(total_len > param.get("cp_align_len")):
            total_len_align = total_len - param.get("cp_align_len")
            reg_addr[index] = total_len_align
            burst_len = _ceil_div(total_len_align, param.get("cp_align_len"))
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w', offset=dst_offset),
                                        data_res.access_ptr("r",
                                                            offset=res_offset),
                                        0, 1, burst_len, 0, 0))
            cp_align_len = param.get("cp_align_len")
            if cp_align_len % reg_count == 0:
                cp_align_len_time_8 = cp_align_len // reg_count
                reg_list = [n for n in range(reg_count)]
                with tvm_ib.for_range(0, cp_align_len_time_8, name="num_a") \
                        as num_a:
                    for reg_idx in reg_list:
                        tvm_ib.emit(tvm.call_extern(
                            data_res.dtype, "reg_mov",
                            tvm.call_extern(reg.dtype, "reg", reg[reg_idx]),
                            data_res.access_ptr(
                                'r',
                                offset=res_offset + total_len_align +
                                (reg_idx + num_a * reg_count))
                        ))

                    for reg_idx in reg_list:
                        tvm_ib.emit(tvm.call_extern(
                            data_tail.dtype, "reg_mov",
                            data_tail.access_ptr('w', offset=(
                                reg_idx + num_a * reg_count)),
                            tvm.call_extern(reg.dtype, "reg", reg[reg_idx])
                        ))
            else:
                with tvm_ib.for_range(0, cp_align_len, name="num_a") \
                        as num_a:
                    tvm_ib.emit(tvm.call_extern(
                        data_res.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[0]),
                        data_res.access_ptr(
                            'r',
                            offset=res_offset + total_len_align + num_a)
                    ))

                    tvm_ib.emit(tvm.call_extern(
                        data_tail.dtype, "reg_mov",
                        data_tail.access_ptr(
                            'w',
                            offset=num_a),
                        tvm.call_extern(reg.dtype, "reg", reg[0])
                    ))

            tvm_ib.emit(
                tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                dst.access_ptr('w', offset=dst_offset
                                               + reg_addr[index]),
                                data_tail.access_ptr("r", offset=0),
                                0, 1, 1, 0, 0))
        with tvm_ib.else_scope():
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w', offset=dst_offset),
                                        data_res.access_ptr("r",
                                                            offset=0),
                                        0, 1, 1, 0, 0))
    with tvm_ib.else_scope():
        burst_len = total_len // param.get("cp_align_len")
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w', offset=dst_offset),
                                    data_res.access_ptr("r", offset=res_offset),
                                    0, 1, burst_len, 0, 0))


def _more_row_ir(dst, data, c_0):
    """
    function of making ir node builder for more row scene

    """
    tvm_ib = tvm.ir_builder.create()
    param = _get_param_more_row(tvm_ib, data.shape, dst.dtype)

    data_ub = _new_alloc(tvm_ib, dst.dtype,
                         param.get('num_row_one_core')*param.get("row_ele"),
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype,
                          param.get('num_row_one_core')*param.get("row_ele"),
                          "data_res", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    data_tail = _new_alloc(tvm_ib, dst.dtype, param.get('cp_align_len'),
                           "data_tail", scope=cce.scope_ubuf)
    reg_addr = tvm_ib.allocate("int32", (8,), name='reg_addr',
                               scope=cce.scope_reg)

    with tvm_ib.for_range(0, param.get("num_group_index") + 1,
                          name="num_g") as num_g:
        with tvm_ib.if_scope(num_g < param.get("num_group_index")):
            args = tvm_ib, param, data, dst, data_ub, data_res, data_tail,\
                   reg, reg_addr, num_g, param.get("num_row_one_core"), c_0
            _func_more_row(args)
        with tvm_ib.if_scope(tvm.all(num_g >= param.get("num_group_index"),
                                     param.get("num_group_mod") > 0)):
            num_core = param.get("num_group_mod") // param.get(
                "num_row_one_core")
            num_row_mod = param.get("num_group_mod") % param.get(
                "num_row_one_core")
            with tvm_ib.if_scope(num_core > 0):
                with tvm_ib.if_scope(param.get("block_index") < num_core):
                    args = tvm_ib, param, data, dst, data_ub, data_res,\
                           data_tail, reg, reg_addr, num_g,\
                           param.get("num_row_one_core"), c_0
                    _func_more_row(args)
            with tvm_ib.if_scope(num_row_mod > 0):
                with tvm_ib.if_scope(
                    tvm.all(param.get("block_index") < (num_core + 1),
                            param.get("block_index") > (num_core - 1))):
                    args = tvm_ib, param, data, dst, data_ub, data_res,\
                           data_tail, reg, reg_addr, num_g, num_row_mod, c_0
                    _func_more_row(args)

    return tvm_ib.get()


def _get_param_split_row(tvm_ib, src_shape, dtype):
    """
    calculate parameters for more dim ir builder make function

    """
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_ele = ((UB_SIZE_B - 32) // 2) // float_size
    n_row, n_no, n_ni, c_0 = src_shape
    num_ele_unit = n_ni*c_0
    num_unit_one_core = ub_ele // num_ele_unit
    num_unit_one_group = num_unit_one_core*device_core_num
    num_unit_in_data = n_row*n_no
    num_group_index = num_unit_in_data // num_unit_one_group
    num_group_mod = num_unit_in_data % num_unit_one_group

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    param_map = {"num_group_index": num_group_index,
                 "num_group_mod": num_group_mod,
                 "num_ele_unit": num_ele_unit,
                 "float_size": float_size,
                 "cp_align_len": cp_align_len,
                 "num_unit_one_core": num_unit_one_core,
                 "num_unit_one_group": num_unit_one_group,
                 "block_index": block_index}

    return param_map


def _func_split_row(args):
    """
    function of moving data for split row scene

    """
    tvm_ib, param, data, dst, data_ub, data_res, data_tail, reg, reg_addr,\
    num_g, num_unit_cur_core = args

    _, n_no, n_ni, c_0 = data.shape
    row_ele = n_no*n_ni*c_0
    num_ele_unit = n_ni*c_0
    h_i, w_i, c_i, n_i = dst.shape
    c_1 = _ceil_div(c_i, c_0)
    h_w = h_i*w_i
    num_unit_before_core = num_g*param.get("num_unit_one_group")\
                           + param.get("block_index")\
                           * param.get("num_unit_one_core")
    num_row_before_core = num_unit_before_core // n_no
    num_hw_before_core = num_row_before_core // c_1
    num_row_cur_hw_before = num_row_before_core % c_1
    num_unit_cur_row_before = num_unit_before_core % n_no
    num_unit_cur_row = n_no - num_unit_cur_row_before

    with tvm_ib.if_scope(num_unit_cur_core <= num_unit_cur_row):
        # gm to ub
        data_offset = num_row_cur_hw_before*h_w*row_ele\
                      + num_hw_before_core*row_ele\
                      + num_unit_cur_row_before*num_ele_unit
        burst_len_data = _ceil_div(num_unit_cur_core*num_ele_unit,
                                   param.get("cp_align_len"))
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr("w", offset=0),
                                    data.access_ptr('r', offset=data_offset),
                                    0, 1, burst_len_data, 0, 0))

        # ub to ub_res
        c_t = tvm.min((num_row_cur_hw_before + 1)*c_0, c_i)
        c_cur = c_t - num_row_cur_hw_before*c_0
        with tvm_ib.for_range(0, c_cur, name="num_c") as num_c:
            with tvm_ib.for_range(0, num_unit_cur_core, name="num_nu")\
                    as num_nu:
                n_now = num_unit_cur_row_before + num_nu
                n_t = tvm.min((n_now + 1)*n_ni, n_i)
                n_cur = n_t - n_now*n_ni
                with tvm_ib.for_range(0, n_cur, name="num_nc") as num_nc:
                    tvm_ib.emit(tvm.call_extern(
                        data_ub.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[0]),
                        data_ub.access_ptr('r',
                                           offset=(num_nu*num_ele_unit
                                                   + num_nc*c_0 + num_c))
                    ))
                    tvm_ib.emit(tvm.call_extern(
                        data_res.dtype, "reg_mov",
                        data_res.access_ptr('w', offset=(num_nu*n_ni + num_nc)),
                        tvm.call_extern(reg.dtype, "reg", reg[0])
                    ))

            num_unit_total = num_unit_cur_core + num_unit_cur_row_before
            n_true = tvm.min(num_unit_total*n_ni, n_i)
            total_len = n_true - num_unit_cur_row_before*n_ni
            dst_offset = num_hw_before_core*c_i*n_i\
                         + num_row_cur_hw_before*c_0*n_i\
                         + num_c*n_i + num_unit_cur_row_before*n_ni
            args = tvm_ib, param, data, dst, data_res, data_tail,\
                   reg, reg_addr, 0, 0, dst_offset, total_len, h_w, row_ele,\
                   num_ele_unit, c_0, c_i, h_i, n_i, n_ni
            _res_to_gm_split_row(args)

    with tvm_ib.if_scope(num_unit_cur_core > num_unit_cur_row):
        num_unit_head = num_unit_cur_row
        num_unit_tail = num_unit_cur_core - num_unit_cur_row

        # head
        # gm to ub
        data_offset = num_row_cur_hw_before*h_w*row_ele\
                      + num_hw_before_core*row_ele\
                      + num_unit_cur_row_before*num_ele_unit
        burst_len_data = _ceil_div(num_unit_head*num_ele_unit,
                                   param.get("cp_align_len"))
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr("w", offset=0),
                                    data.access_ptr('r', offset=data_offset),
                                    0, 1, burst_len_data, 0, 0))

        # ub to ub_res
        c_t = tvm.min((num_row_cur_hw_before + 1)*c_0, c_i)
        c_cur = c_t - num_row_cur_hw_before*c_0
        with tvm_ib.for_range(0, c_cur, name="num_c") as num_c:
            with tvm_ib.for_range(0, num_unit_head, name="num_nu") as num_nu:
                n_now = num_unit_cur_row_before + num_nu
                n_t = tvm.min((n_now + 1)*n_ni, n_i)
                n_cur = n_t - n_now*n_ni
                with tvm_ib.for_range(0, n_cur, name="num_nc") as num_nc:
                    tvm_ib.emit(tvm.call_extern(
                        data_ub.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[0]),
                        data_ub.access_ptr('r',
                                           offset=(num_nu*num_ele_unit
                                                   + num_nc*c_0 + num_c))
                    ))
                    tvm_ib.emit(tvm.call_extern(
                        data_res.dtype, "reg_mov",
                        data_res.access_ptr('w', offset=(num_nu*n_ni + num_nc)),
                        tvm.call_extern(reg.dtype, "reg", reg[0])
                    ))

            num_unit_total = num_unit_head + num_unit_cur_row_before
            n_true = tvm.min(num_unit_total*n_ni, n_i)
            total_len = n_true - num_unit_cur_row_before*n_ni
            dst_offset = num_hw_before_core*c_i*n_i\
                         + num_row_cur_hw_before*c_0*n_i\
                         + num_c*n_i + num_unit_cur_row_before*n_ni
            args = tvm_ib, param, data, dst, data_res, data_tail,\
                   reg, reg_addr, 1, 0, dst_offset, total_len, h_w, row_ele,\
                   num_ele_unit, c_0, c_i, h_i, n_i, n_ni
            _res_to_gm_split_row(args)

        # tail
        num_row_before_core_tail = num_row_before_core + 1
        num_hw_before_core_tail = num_row_before_core_tail // c_1
        num_row_cur_hw_before_tail = num_row_before_core_tail % c_1
        data_offset = num_row_cur_hw_before_tail * h_w * row_ele\
                      + num_hw_before_core_tail * row_ele
        ub_offset_tail = num_unit_head*num_ele_unit
        # gm to ub
        burst_len_data = _ceil_div(num_unit_tail*num_ele_unit,
                                   param.get("cp_align_len"))
        tvm_ib.emit(
            tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                            data_ub.access_ptr("w", offset=ub_offset_tail),
                            data.access_ptr('r', offset=data_offset),
                            0, 1, burst_len_data, 0, 0))

        # ub to ub_res
        num_unit_total = num_unit_head + num_unit_cur_row_before
        n_true = tvm.min(num_unit_total*n_ni, n_i)
        total_len_head = n_true - num_unit_cur_row_before*n_ni
        res_offset_tail = _ceil_fill(total_len_head, param.get("cp_align_len"))

        c_t = tvm.min((num_row_cur_hw_before_tail + 1)*c_0, c_i)
        c_cur = c_t - num_row_cur_hw_before_tail*c_0
        with tvm_ib.for_range(0, c_cur, name="num_c") as num_c:
            with tvm_ib.for_range(0, num_unit_tail, name="num_nu") as num_nu:
                n_cur = n_ni
                with tvm_ib.for_range(0, n_cur, name="num_nc") as num_nc:
                    tvm_ib.emit(tvm.call_extern(
                        data_ub.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[0]),
                        data_ub.access_ptr('r',
                                           offset=(ub_offset_tail
                                                   + num_nu*num_ele_unit
                                                   + num_nc*c_0 + num_c))
                    ))
                    tvm_ib.emit(tvm.call_extern(
                        data_res.dtype, "reg_mov",
                        data_res.access_ptr('w', offset=(res_offset_tail
                                                         + num_nu*n_ni
                                                         + num_nc)),
                        tvm.call_extern(reg.dtype, "reg", reg[0])
                    ))

            dst_offset = num_hw_before_core_tail*c_i*n_i\
                         + num_row_cur_hw_before_tail*c_0*n_i + num_c*n_i
            total_len = num_unit_tail*n_ni
            burst_len = total_len // param.get("cp_align_len")
            tvm_ib.emit(
                tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                dst.access_ptr('w', offset=dst_offset),
                                data_res.access_ptr("r",
                                                    offset=res_offset_tail),
                                0, 1, burst_len, 0, 0))


def _dst_to_data_pos(args):
    """
    function of calculating dst position according to data position

    """
    dst_pos, h_w, row_ele, num_ele_unit, c_0, c_i, h_i, n_i, n_ni = args
    hw_row_ele = h_w*row_ele
    num_c0_cur_hw = dst_pos // hw_row_ele
    num_ele_cur_hw_dst = dst_pos % hw_row_ele
    num_hw = num_ele_cur_hw_dst // row_ele
    num_ele_cur_row = num_ele_cur_hw_dst % row_ele
    num_ni_cur_row = num_ele_cur_row // num_ele_unit
    num_ele_cur_unit = num_ele_cur_row % num_ele_unit
    num_n_cur_ni = num_ele_cur_unit // c_0
    num_c = num_ele_cur_unit % c_0

    data_pos = num_hw*c_i*h_i + num_c0_cur_hw*c_0*n_i + num_c*n_i\
               + num_ni_cur_row*n_ni + num_n_cur_ni
    return data_pos


def _res_to_gm_split_row(args):
    """
    function of moving data from data_res(UB) to dst(GM) for split row scene

    """
    tvm_ib, param, data, dst, data_res, data_tail, reg, reg_addr,\
    index, res_offset, dst_offset, total_len, h_w, row_ele, num_ele_unit,\
    c_0, c_i, h_i, n_i, n_ni = args

    with tvm_ib.if_scope(total_len % param.get("cp_align_len") > 0):
        with tvm_ib.if_scope(total_len > param.get("cp_align_len")):
            total_len_align = total_len - param.get("cp_align_len")
            reg_addr[index] = total_len_align
            burst_len = _ceil_div(total_len_align, param.get("cp_align_len"))
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w', offset=dst_offset),
                                        data_res.access_ptr("r",
                                                            offset=res_offset),
                                        0, 1, burst_len, 0, 0))
            with tvm_ib.for_range(0, param.get("cp_align_len"), name="num_a")\
                    as num_a:
                tvm_ib.emit(tvm.call_extern(
                    data_res.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_res.access_ptr('r',
                                        offset=res_offset + total_len_align
                                        + num_a)
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    data_tail.access_ptr('w', offset=num_a),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))
            tvm_ib.emit(
                tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                dst.access_ptr('w', offset=dst_offset
                                               + reg_addr[index]),
                                data_tail.access_ptr("r", offset=0),
                                0, 1, 1, 0, 0))
        with tvm_ib.else_scope():
            num_ele = param.get("cp_align_len") - total_len
            with tvm_ib.for_range(0, num_ele, name="num_e") as num_e:
                reg_addr[index] = total_len + num_e
                dst_pos = dst_offset + reg_addr[index]
                args = dst_pos, h_w, row_ele, num_ele_unit,\
                       c_0, c_i, h_i, n_i, n_ni
                data_pos = _dst_to_data_pos(args)
                tvm_ib.emit(tvm.call_extern(data_tail.dtype, "copy_gm_to_ubuf",
                                            data_tail.access_ptr("w", offset=0),
                                            data.access_ptr('r',
                                                            offset=data_pos),
                                            0, 1, 1, 0, 0))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_tail.access_ptr('r', offset=0)
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_res.dtype, "reg_mov",
                    data_res.access_ptr('w', offset=total_len + num_e),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w', offset=dst_offset),
                                        data_res.access_ptr("r",
                                                            offset=0),
                                        0, 1, 1, 0, 0))
    with tvm_ib.else_scope():
        burst_len = total_len // param.get("cp_align_len")
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w', offset=dst_offset),
                                    data_res.access_ptr("r", offset=res_offset),
                                    0, 1, burst_len, 0, 0))


def _split_row_ir(dst, data):
    """
    function of making ir node builder for split row scene

    """
    tvm_ib = tvm.ir_builder.create()
    param = _get_param_split_row(tvm_ib, data.shape, dst.dtype)

    data_ub = _new_alloc(tvm_ib, dst.dtype,
                         param.get('num_unit_one_core')
                         * param.get("num_ele_unit"),
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype,
                          param.get('num_unit_one_core')
                          * param.get("num_ele_unit"),
                          "data_res", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    data_tail = _new_alloc(tvm_ib, dst.dtype, param.get('cp_align_len'),
                           "data_tail", scope=cce.scope_ubuf)
    reg_addr = tvm_ib.allocate("int32", (8,), name='reg_addr',
                               scope=cce.scope_reg)

    with tvm_ib.for_range(0, param.get("num_group_index") + 1,
                          name="num_g") as num_g:
        with tvm_ib.if_scope(num_g < param.get("num_group_index")):
            args = tvm_ib, param, data, dst, data_ub, data_res, data_tail,\
                   reg, reg_addr, num_g, param.get("num_unit_one_core")
            _func_split_row(args)
        with tvm_ib.if_scope(tvm.all(num_g >= param.get("num_group_index"),
                                     param.get("num_group_mod") > 0)):
            num_core = param.get("num_group_mod") // param.get(
                "num_unit_one_core")
            num_unit_mod = param.get("num_group_mod") % param.get(
                "num_unit_one_core")
            with tvm_ib.if_scope(num_core > 0):
                with tvm_ib.if_scope(param.get("block_index") < num_core):
                    args = tvm_ib, param, data, dst, data_ub, data_res,\
                           data_tail, reg, reg_addr,\
                           num_g, param.get("num_unit_one_core")
                    _func_split_row(args)
            with tvm_ib.if_scope(num_unit_mod > 0):
                with tvm_ib.if_scope(
                    tvm.all(param.get("block_index") < (num_core + 1),
                            param.get("block_index") > (num_core - 1))):
                    args = tvm_ib, param, data, dst, data_ub, data_res,\
                           data_tail, reg, reg_addr, num_g, num_unit_mod
                    _func_split_row(args)

    return tvm_ib.get()


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
                                                   * param.get("cp_align_len")\
                                                   * n_burst * num_cy
                                        ub_cur = ub_offset + (
                                            burst_len + dst_stride) \
                                                 * param.get("cp_align_len")\
                                                 * n_burst * num_cy
                                        tvm_ib.emit(
                                            tvm.call_extern(
                                                data_ub.dtype,
                                                "copy_gm_to_ubuf",
                                                data_ub.access_ptr(
                                                    "w", offset=ub_cur),
                                                data.access_ptr(
                                                    'r', offset=data_cur),
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
                                               * param.get("cp_align_len")\
                                               * num_nb
                                    ub_cur = ub_offset + (
                                        burst_len + dst_stride)\
                                             * param.get("cp_align_len")\
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


def _check_parameters(src, dst, src_format, dst_format, kernel_name):
    """
    check the parameters including src_shape, dst_shape,
    src_format, dst_format, dtype and kernel_name

    """
    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype")
    dtype_dst = dst.get("dtype")

    if src_format.lower() != "fractal_zn" and src_format.lower() != "fractal_z":
        raise RuntimeError("src_format must be FRACTAL_Zn !")

    if dst_format.lower() != "hwcn":
        raise RuntimeError("dst_format must be HWCN !")

    util.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype, check_list)
    if dtype != dtype_dst:
        raise RuntimeError("dtype of src and dst are different !")

    util.check_shape_rule(src_shape, 4, 4)
    util.check_shape_rule(dst_shape, 4, 4)
    util.check_tensor_shape_size(src_shape)
    util.check_tensor_shape_size(dst_shape)

    if src_shape[2] != 16 or src_shape[3] != 16:
        raise RuntimeError(
            "ni and c0 must be 16 !")

    h_i, w_i, c_i, n_i = dst_shape

    c_0 = 16
    c_1 = _ceil_div(c_i, c_0)
    src_one = c_1*h_i*w_i
    n_ni = 16
    n_no = _ceil_div(n_i, n_ni)

    if list(src_shape) != [src_one, n_no, 16, 16]:
        raise RuntimeError("src_shape is wrong !")


def _get_ir_branch(src_shape, dtype):
    """
    judge ir node builder branch for nchw float32 scene

    """
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    ub_bytes = UB_SIZE_B - 32
    ub_half = ub_bytes // 2
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    n_no = src_shape[1]
    row_bytes = n_no*16*16*float_size
    if row_bytes <= ub_half:
        return "more_row"
    else:
        return "split_row"


@util.check_input_type(dict, dict, str, str, str)
def zn_2_hwcn(src, dst, src_format, dst_format, kernel_name='zn_2_hwcn'):
    """
    algorithm: zn_2_hwcn
    calculating: change data format from Zn to HWCN

    Parameters
    ----------
    src: dict
        contains shape and dtype information of input tensor
    dst: dict
        contains shape and dtype information of output tensor
    src_format: str
        represents the format of input tensor, only support "Zn"
    dst_format: str
        represents the format of output tensor, only support "HWCN"
    kernel_name: str
        cce kernel name, default value is "zn_2_hwcn"

    Returns
    -------
    None
    """
    _check_parameters(src, dst, src_format, dst_format, kernel_name)
    dst_shape = dst.get("shape")
    dtype = src.get("dtype")

    h_i, w_i, c_i, n_i = dst_shape
    c_0 = 16
    if dtype == "int8":
        c_0 = 32
    c_1 = _ceil_div(c_i, c_0)
    n_ni = 16
    n_no = _ceil_div(n_i, n_ni)
    shape_zn = [c_1*h_i*w_i, n_no, n_ni, c_0]

    branch = _get_ir_branch(shape_zn, dtype)
    data = tvm.placeholder(shape_zn, dtype=dtype, name="data")
    if branch == "more_row":
        res = tvm.extern(dst_shape, [data],
                         lambda ins, outs: _more_row_ir(outs[0], ins[0], c_0),
                         name="res", dtype=dtype)
    else:
        res = tvm.extern(dst_shape, [data],
                         lambda ins, outs: _split_row_ir(outs[0], ins[0]),
                         name="res", dtype=dtype)

    tensor_list = [data, res]
    sch = tvm.create_schedule(res.op)
    with build_config:
        tvm.build(sch, tensor_list, "cce", name=kernel_name)
