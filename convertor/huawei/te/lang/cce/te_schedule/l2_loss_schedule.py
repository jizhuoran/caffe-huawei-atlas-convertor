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

l2_loss_schedule
"""

from __future__ import absolute_import
import math
from te import tvm
from te import platform as cce


def _map_apend(input_map, key, value):
    if input_map.get(key):
        if isinstance(value, list):
            for tmp_v in value:
                if not tmp_v in input_map[key]:
                    input_map[key].append(tmp_v)
        else:
            if not value in input_map[key]:
                input_map[key].append(value)
    else:
        if isinstance(value, list):
            input_map[key] = value
        else:
            input_map[key] = [value]


def _gen_reversed_subgraph_list(out_tensor, tensor_list_map,
                                tensor_list_dst_tensor_map):
    """traverse tensors by Depth-First-Search

    Parameters
    ----------
    out_tensor : tensor
        traverse tensors from this tensor,
        traversing its input tensors recursively.

    tensor_list : list
        record tensors in the order of Depth-First-Search.

    """
    if out_tensor is None:
        return
    stack = [out_tensor]
    visited_list = []
    while stack:
        cur_tensor = stack.pop()
        visited_list.append(cur_tensor)
        for in_tensor in cur_tensor.op.input_tensors:
            if in_tensor not in visited_list:
                stack.append(in_tensor)
                tensor_list_map[in_tensor.name] = in_tensor
            _map_apend(tensor_list_dst_tensor_map, in_tensor, cur_tensor)


def get_max_factor(cnt, align_limit_cnt, ub_limit_cnt):
    """
    caculate the max element num loaded in UB buffer
    :return: max element num loaded in UB buffer
    """
    max_factor = int(1)
    sqrt_n = int(math.sqrt(cnt))
    for fac in range(cnt, sqrt_n, -1):
        quotient = cnt // fac
        remainder = cnt - (quotient) * fac
        fac_cnt = fac
        remainder_cnt = remainder
        if align_limit_cnt <= fac_cnt <= ub_limit_cnt \
                and ((align_limit_cnt <= remainder_cnt <= ub_limit_cnt) \
                     or remainder_cnt == 0):
            max_factor = fac
            break
    return max_factor


def get_max_ub_count(datatype):
    """
    caculate the max element num loaded in UB buffer
    :return: max element num loaded in UB buffer
    """
    _total_size = cce.get_soc_spec("UB_SIZE")
    # 8k for reserve
    if datatype == "float32":
        max_ub_count = (_total_size - 8192) // 4
    elif datatype == "float16":
        max_ub_count = (_total_size - 8192) // 2
    else:
        raise RuntimeError("Not supported dtype!")

    return max_ub_count


def ubtiling(one_core_data_cnt, dtype):
    """
    ubtiling for l2loss
    :param one_core_data_cnt: one_core_data_cnt
    :param dtype: data type
    :return: ubtiling factor
    """
    max_ub_count = get_max_ub_count(dtype)
    if one_core_data_cnt <= (max_ub_count // 2):
        # mul_0_local_UB + mul_1_local_UB
        max_factor = one_core_data_cnt
    else:
        # data_input_local_UB + mul_0_local_UB + mul_1_local_UB +
        # data_input_local_UB1
        cut_ub_count = max_ub_count // 4
        max_factor = cut_ub_count

    return max_factor


def get_max_divided_factor(date_cnt, core_num):
    """
    caculate the min devided factor by a max mod litter equal m. eg: factor =
    date_cnt // core_num
    :return:  max mod,  min devided factor
    """
    max_mod = int(1)
    factor = date_cnt
    if date_cnt <= core_num:
        return 1, date_cnt
    for i in range(core_num, 0, -1):
        if date_cnt % i == 0:
            max_mod = i
            factor = date_cnt // i
            break
    return max_mod, factor


# pylint: disable=too-many-locals
def l2_loss_schedule(res, input_tensors):
    '''
    l2_loss schedule for float32 and dim cnt equal to 1
    :param data_features: input tensor 1
    :param res: res tensor
    :return: sch
    '''
    # tensor_list:input -> vmuls -> vmul -> reduce
    reduce_c = res[-1]
    sch = tvm.create_schedule(reduce_c.op)

    tensor_list_map = {}
    tensor_list_dst_tensor_map = {}

    _gen_reversed_subgraph_list(reduce_c, tensor_list_map,
                                tensor_list_dst_tensor_map)

    input_tensor_dst_tensor_map = {}
    mid_tensor_dst_tensor_map = {}
    cache_read_tensor_list = []
    cache_write_tensor_list = []
    for tensor in tensor_list_dst_tensor_map:
        if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
            input_tensor_dst_tensor_map[tensor] = tensor_list_dst_tensor_map[
                tensor]
            cache_read_tensor_list.append(tensor)
        else:
            mid_tensor_dst_tensor_map[tensor] = tensor_list_dst_tensor_map[
                tensor]
            cache_write_tensor_list.append(tensor)

    # ---------cache read/write--------------
    cache_read_buffer_list = []
    for tensor in cache_read_tensor_list:
        cache_read_buffer_list.append(
            sch.cache_read(tensor, cce.scope_ubuf,
                           input_tensor_dst_tensor_map[tensor]))
    cache_write_buffer_list = []
    for tensor in cache_write_tensor_list:
        cache_write_buffer_list.append(sch.cache_write(tensor, cce.scope_ubuf))

    # ---------compute inline----------------
    for tensor in cache_write_tensor_list:
        sch[tensor].compute_inline()

    # ---------add reduce rfactor------------
    shape = input_tensors[0].shape[-1]
    block_dim = cce.get_soc_spec("CORE_NUM")
    # (N,M) N is less equal blockDim, and can be divided by totaldata
    _, shape_n = get_max_divided_factor(int(shape), block_dim)

    res_o, _ = sch[reduce_c].split(reduce_c.op.reduce_axis[0], shape_n)
    reduce_i_ub = sch.rfactor(reduce_c, res_o)
    sch[reduce_i_ub].set_scope(cce.scope_ubuf)
    reduce_o_gm = sch.cache_write(reduce_c, "")

    ub_split_factor = ubtiling(shape_n, reduce_c.dtype)
    reduce_i_ub_o, reduce_i_ub_i = sch[reduce_i_ub].split(
        reduce_i_ub.op.reduce_axis[0],
        factor=ub_split_factor)

    # ----------------reorder------------------
    sch[reduce_o_gm].reorder(reduce_o_gm.op.reduce_axis[0],
                             reduce_o_gm.op.axis[0])

    # ----------------compute at---------------
    for tensor_u in cache_read_buffer_list:
        sch[tensor_u].compute_at(sch[reduce_i_ub], reduce_i_ub_o)
    for tensor_u in cache_write_buffer_list:
        sch[tensor_u].compute_at(sch[reduce_i_ub], reduce_i_ub_o)
    sch[reduce_i_ub].compute_at(sch[reduce_o_gm], reduce_o_gm.op.reduce_axis[0])

    # -----------------emit_insn----------------
    for tensor_u in cache_read_buffer_list:
        sch[tensor_u].emit_insn(tensor_u.op.axis[0], 'dma_copy')
    # pylint: disable=consider-using-enumerate
    for i in range(len(cache_write_buffer_list)):
        emit_insn_pragma = _get_emit_insn_map(cache_write_tensor_list[i])
        sch[cache_write_buffer_list[i]].emit_insn(
            cache_write_buffer_list[i].op.axis[0],
            emit_insn_pragma)
    sch[reduce_i_ub].emit_insn(reduce_i_ub_i, "reduce_last_axis_reduce_sum")
    sch[reduce_o_gm].emit_insn(reduce_o_gm.op.axis[0], "dma_copy")
    sch[reduce_c].emit_insn(sch[reduce_c].op.axis[0], "phony_insn")

    # ------------------bind----------------------
    block = tvm.thread_axis("blockIdx.x")
    sch[reduce_o_gm].bind(reduce_o_gm.op.reduce_axis[0], block)

    # --------------exchange real out--------------
    res.pop()
    res.append(reduce_o_gm)

    return sch


def _get_emit_insn_map(tensor):
    insn_map = {"elewise_binary_mul": "vector_mul",
                "elewise_single_VS_mul": "vector_muls",
                }

    if tensor.op.tag.find("|") != -1:
        str_list = tensor.op.tag.split("|")
        insn = insn_map.get(str_list[0])
    else:
        insn = insn_map.get(tensor.op.tag)
    return insn
