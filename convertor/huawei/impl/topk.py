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

topk
"""

# pylint: disable=C0302
# pylint: disable=E0401
# pylint: disable=W0603
# pylint: disable=R0914
# pylint: disable=W0612
import math
from te import tik
from te import platform as tbe_platform

VMS4_ELEMENT_NUM = 4         # we're doing 4-way merge sorting
VBS_SORT_NUM = 16
REGION_SIZE_INFP16 = 8       # region size in term of fp16
MAX_INPUT_LIST_LENGTH = 4096

UB_SIZE = 128 * 1024          # UB size in byte
REGION_SIZE_INBYTE = 16      # fp16 region size in byte
HALF_UB_REGION_CAPACITY = UB_SIZE // 2 // REGION_SIZE_INBYTE

def check_topk_param(input_args, output_args):
    """
    check topk param

    Parameters
    ----------
    input_args: is a dict, the keys as follow:
        proposal_num: proposal total number
        k: topk
        score_threshold: socore threshold
        region_orig: region proposal tensor
        mem_swap: middle memory tensor

    output_args: is a dict, the keys as follow:
        batch_id: batch id
        regions_sorted: sorted proposal tensor
        proposal_actual_num: proposla autual output number

    Returns
    -------
    NA
    """

    if input_args["proposal_num"] % 16 != 0:
        raise RuntimeError("proposal_num%16 != 0")
    if input_args["score_threshold"] < 0:
        raise RuntimeError("score_threshold < 0")
    if input_args["regions_orig"].shape[2] != 8:
        raise RuntimeError("regions_orig.shape[2] != 8")
    if input_args["mem_swap"].shape[2] != 8:
        raise RuntimeError("mem_swap.shape[2] != 8")

    if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES", "Hi3796CV300CS"):
        if input_args["k"] > 3000:
            raise RuntimeError("k > 3000")
    else:
        if input_args["regions_orig"].dtype == "float32":
            if input_args["k"] > 3000:
                raise RuntimeError("k > 3000")
        else:
            if input_args["k"] > 6000:
                raise RuntimeError("k > 6000")

    if output_args["regions_sorted"].shape[1] < 16:
        raise RuntimeError("regions_sorted.shape[1] < 16")
    if output_args["regions_sorted"].shape[1] < input_args["k"]:
        raise RuntimeError("regions_sorted.shape[1] < k")

    if output_args["regions_sorted"].shape[2] != 8:
        raise RuntimeError("regions_sorted.shape[2] != 8")

def set_ub_size_by_product_type(data_type):
    """
    set topk param about ub

    Parameters
    ----------
    data_type: proposal data type

    Returns
    -------
    NA
    """
    global UB_SIZE
    #UB_SIZE = tik.Dprofile().get_unified_buffer_size()
    UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
    
    global REGION_SIZE_INBYTE
    if data_type == "float16":
        REGION_SIZE_INBYTE = 2 * 8
    else:
        REGION_SIZE_INBYTE = 4 * 8

    global HALF_UB_REGION_CAPACITY
    HALF_UB_REGION_CAPACITY = UB_SIZE // 2 // REGION_SIZE_INBYTE

def tik_topk_min(tik_inst, val_a, val_b, result_):
    """
    get min val

    Parameters
    ----------
    tik_inst: tik instance
    val_a: value a
    val_b: value b
    result_: value result

    Returns
    -------
    NA
    """
    with tik_inst.if_scope(val_a < val_b):
        result_.set_as(val_a)
    with tik_inst.else_scope():
        result_.set_as(val_b)

def tik_topk(tik_inst, topk_in, topk_out):
    """
    topk entry function

    Parameters
    ----------
    tik_inst: tik instance
    topk_in: is a dict, the keys as follow:
        proposal_num: proposal total number
        k: topk
        score_threshold: socore threshold
        region_orig: region proposal tensor
        mem_swap: middle memory tensor

    topk_out: is a dict, the keys as follow:
        batch_id: batch id
        regions_sorted: sorted proposal tensor
        proposal_actual_num: proposla autual output number

    Returns
    -------
    NA
    """
    check_topk_param(topk_in, topk_out)
    data_type = topk_in["regions_orig"].dtype
    set_ub_size_by_product_type(data_type)

    n_required = min(math.ceil(topk_in["k"]/16)*16, topk_in["proposal_num"])
    with tik_inst.new_stmt_scope():
        tensor_shape_ub = (1, UB_SIZE//REGION_SIZE_INBYTE, REGION_SIZE_INFP16)
        mem_ub = tik_inst.Tensor(data_type, tensor_shape_ub, name="UB",
                                 scope=tik.scope_ubuf)
        proposal_store = (topk_out["batch_id"], mem_ub,
                          topk_out["regions_sorted"],
                          topk_in["regions_orig"], topk_in["mem_swap"])
        src_pos = 0
        dest_pos = 0
        task_id = "0"
        split_param = (topk_in["proposal_num"], n_required,
                       src_pos, dest_pos, task_id)
        tik_topk_split_proposal_group(tik_inst, proposal_store, split_param)

        if topk_in["proposal_num"] <= HALF_UB_REGION_CAPACITY:
            tik_inst.data_move(mem_ub[0, 0, 0],
                               topk_in["mem_swap"][topk_out["batch_id"], 0, 0],
                               sid=0,
                               nburst=1,
                               burst=math.ceil(REGION_SIZE_INBYTE * 4 / 32) * (n_required // 4),
                               src_stride=0,
                               dst_stride=0)
            tik_inst.data_move(
                topk_out["regions_sorted"][topk_out["batch_id"], 0, 0],
                mem_ub[0, 0, 0],
                sid=0,
                nburst=1,
                burst=math.ceil(REGION_SIZE_INBYTE * 4 / 32) * (n_required // 4),
                src_stride=0,
                dst_stride=0)

    with tik_inst.new_stmt_scope():
        tik_topk_filter_by_score_threshold(
            tik_inst,
            (n_required, topk_in["k"], topk_in["score_threshold"], data_type),
            topk_out)

# pylint: disable=too-many-statements
def tik_topk_filter_by_score_threshold(tik_inst, filter_input, top_output):
    """
    filter invalid score

    Parameters
    ----------
    tik_inst: tik instance
    filter_input: is a list, the keys as follow:
        n_required: required tensor num
        k: top k
        score_threshold: score threshold
        data_type: tensor data type
    topk_out: is a dict, the keys as follow:
        batch_id: batch id
        regions_sorted: sorted proposal tensor
        proposal_actual_num: proposla autual output number

    Returns
    -------
    NA
    """
    n_required = filter_input[0]
    k = filter_input[1]
    score_threshold = filter_input[2]
    data_type = filter_input[3]

    ceil_num = 64 if data_type == "float32" else 128
    n_required_apply = math.ceil(n_required/ceil_num) * ceil_num
    vsel_score_ub = tik_inst.Tensor(
        data_type, (n_required_apply, 8), name="vsel_score_ub",
        scope=tik.scope_ubuf)

    required_score_ub = tik_inst.Tensor(
        data_type, (n_required_apply, REGION_SIZE_INFP16),
        name="required_score_ub", scope=tik.scope_ubuf)

    tik_inst.data_move(
        required_score_ub[0, 0],
        top_output["regions_sorted"][top_output["batch_id"], 0, 0],
        sid=0,
        nburst=1,
        burst=math.ceil(REGION_SIZE_INBYTE * n_required / 32),
        src_stride=0,
        dst_stride=0)

    init_scalar = tik_inst.Scalar(data_type, "init_scalar", 0)
    with tik_inst.for_range(n_required, n_required_apply) as index:
        required_score_ub[index, 4].set_as(init_scalar)

    if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Ascend310","Ascend910","Hi3796CV300ES", "Hi3796CV300CS"):
        count_scalar = tik_inst.Scalar("int32", "count_scalar", 0)

        ones_ub = tik_inst.Tensor(data_type, (ceil_num,), name="ones_ub",
                                  scope=tik.scope_ubuf)
        tik_inst.vector_dup(ceil_num, ones_ub, 1, 1, 1, 8)
        zeros_ub = tik_inst.Tensor(data_type, (ceil_num,), name="zeros_ub",
                                   scope=tik.scope_ubuf)
        tik_inst.vector_dup(ceil_num, zeros_ub, 0, 1, 1, 8)
        threshold_ub = tik_inst.Tensor(data_type, (16, 8), name="threshold_ub",
                                       scope=tik.scope_ubuf)
        tik_inst.vector_dup([0x1010101010101010, 0x1010101010101010],
                            threshold_ub, score_threshold, 1, 1, 8)

        init_mask_ub = tik_inst.Tensor("uint16", (16,), name="init_mask_ub", scope=tik.scope_ubuf)
        tik_inst.vector_dup(16, init_mask_ub, 0, 1, 1, 8)
        tik_inst.mov_tensor_to_cmpmask(init_mask_ub)
        with tik_inst.for_range(0, n_required_apply // (ceil_num // 8)) as cmp_index:
            cmp_mask = tik_inst.vcmp_gt([0x1010101010101010, 0x1010101010101010],
                                        required_score_ub[cmp_index * (ceil_num // 8), 0],
                                        threshold_ub, 1, 1)
            tik_inst.vsel(ceil_num, 0, vsel_score_ub[cmp_index * (ceil_num // 8), 0], cmp_mask,
                          ones_ub, zeros_ub, 1, 1, 1, 1)

        count_dst_ub = tik_inst.Tensor("int32", (8,), name="count_dst_ub",
                                       scope=tik.scope_ubuf)
        count_src_ub = tik_inst.Tensor("float16", (16,), name="count_src_ub",
                                       scope=tik.scope_ubuf)
        vec_reduce_add_ub = tik_inst.Tensor("float16", (n_required_apply // (ceil_num // 8),),
                                            name="vec_reduce_add_ub",
                                            scope=tik.scope_ubuf)

        tik_inst.vec_reduce_add(ceil_num, count_src_ub, vsel_score_ub,
                                vec_reduce_add_ub, n_required_apply // (ceil_num // 8), 8)

        tik_inst.vconv(1, "round", count_dst_ub, count_src_ub, 1, 1, 1, 8, 4)
        count_scalar.set_as(count_dst_ub[0])
        tik_topk_min(tik_inst, k, count_scalar, top_output["proposal_actual_num"])
    else:
        count_scalar = tik_inst.Scalar("uint32", "count_scalar", 0)
        thres = tik_inst.Scalar(data_type, "thres", score_threshold)

        vcmvs_type = "uint32" if data_type == "float32" else "uint16"
        vcmvs_shape = math.ceil(n_required / 16 * 2 / 32) * 32
        vex_score_ub = tik_inst.Tensor(
            data_type, (n_required_apply, ), name="vex_score_ub",
            scope=tik.scope_ubuf)
        tik_inst.vector_dup(ceil_num, vex_score_ub[0], 0,
                            n_required_apply // ceil_num, 1, 8)
        dst_ub = tik_inst.Tensor(vcmvs_type, (vcmvs_shape,), name="dst_ub",
                                 scope=tik.scope_ubuf)
        if vcmvs_shape // ceil_num > 0:
            tik_inst.vector_dup(ceil_num, dst_ub, 0,
                                vcmvs_shape // ceil_num, 1, 8)
        if vcmvs_shape % ceil_num > 0:
            tik_inst.vector_dup(vcmvs_shape % ceil_num,
                                dst_ub[ceil_num * (vcmvs_shape // ceil_num)],
                                0, 1, 1, 1)

        with tik_inst.for_range(0, n_required_apply // 16 // 255) as index:
            vex_offset = index * 16 * 255
            tik_inst.vextract(vex_score_ub[vex_offset,],
                              required_score_ub[vex_offset, 0], 255, 4)
        with tik_inst.if_scope(n_required_apply // 16 % 255 > 0):
            vex_offset = n_required_apply // 16 // 255 * 16 * 255
            tik_inst.vextract(vex_score_ub[vex_offset,],
                              required_score_ub[vex_offset, 0],
                              n_required_apply // 16 % 255, 4)

        tik_inst.vcmpvs_gt(dst_ub, vex_score_ub, thres,
                           n_required_apply // ceil_num, 1, 8)
        tik_inst.vreduce(n_required, vsel_score_ub, vex_score_ub, dst_ub,
                         1, 1, 8, 1, 0, count_scalar, mask_mode="counter")

        tik_topk_min(tik_inst, k, count_scalar, top_output["proposal_actual_num"])

def tik_topk_split_proposal_group(tik_inst, proposal_store, split_param):
    """
    divide proposal num for sorting in ub

    Parameters
    ----------
    tik_inst: tik instance
    proposal_store: is a list, the keys as follow:
        batch_id: batch id
        mem_ub: ub tensor
        regions_sorted: sorted tensor
        regions_orig: original tersor
        mem_interm: intermediate tensor
    split_param: is a list, the keys as follow:
        n_regions: split tensor num
        n_required: required tensor num
        src_pos: src pos
        dest_pos: dest pos
        task_id: task id

    Returns
    -------
    level_from_leaf: split depth
    """
    batch_id = proposal_store[0]
    mem_ub = proposal_store[1]
    regions_sorted = proposal_store[2]
    regions_orig = proposal_store[3]
    mem_interm = proposal_store[4]

    n_regions = split_param[0]
    n_required = split_param[1]
    src_pos = split_param[2]
    dest_pos = split_param[3]
    task_id = split_param[4]

    level_from_leaf = 0
    if n_regions <= HALF_UB_REGION_CAPACITY:
        dest_pos = src_pos
        tik_topk_merge_sort_internal(
            tik_inst,
            (batch_id, mem_ub, mem_interm, regions_orig),
            ([n_regions], n_required, [src_pos], dest_pos))
        level_from_leaf = 0
    else:
        n_regions_subtsk = math.ceil(n_regions/VMS4_ELEMENT_NUM)
        n_regions_subtsk = math.ceil(n_regions_subtsk / VBS_SORT_NUM) *\
                           VBS_SORT_NUM
        n_required_subtsk = min(n_regions_subtsk, n_required)

        n_remains = n_regions
        subtsk_src_pos = src_pos
        subtsk_dest_pos = dest_pos

        subtsk_dest_pos_list = []
        subtsk_n_required_list = []

        for i in range(VMS4_ELEMENT_NUM):
            subtsk_n_regions = min(n_regions_subtsk, n_remains)
            subtsk_n_required = min(n_required_subtsk, subtsk_n_regions)

            level_from_leaf = tik_topk_split_proposal_group(
                tik_inst,
                (batch_id, mem_ub, regions_sorted, regions_orig, mem_interm),
                (subtsk_n_regions, n_required_subtsk, subtsk_src_pos,
                 subtsk_dest_pos, task_id + (".%d" % i)))

            if level_from_leaf == 0:
                subtsk_dest_pos_list.append(subtsk_src_pos)
            else:
                subtsk_dest_pos_list.append(subtsk_dest_pos)
            subtsk_n_required_list.append(subtsk_n_required)
            subtsk_src_pos += subtsk_n_regions
            subtsk_dest_pos += n_required_subtsk
            n_remains -= subtsk_n_regions
        if n_remains != 0:
            raise RuntimeError("After divison, no regions should be left!")

        tik_topk_merge_subgroup(
            tik_inst,
            (batch_id, mem_ub, regions_orig, regions_sorted, mem_interm),
            (n_required, subtsk_dest_pos_list, subtsk_n_required_list, dest_pos,
             task_id, level_from_leaf))
        level_from_leaf += 1

    return level_from_leaf

def tik_topk_internal_vbs(tik_inst, vbs_src, dest_pos_ub):
    """
    do vbs

    Parameters
    ----------
    tik_inst: tik instance
    vbs_src: is a list, the keys as follow:
        mem_ub: ub tensor
        src_region_num_list: src region num list
        n_total_regions: total regions
    dest_pos_ub: dest_pos_ub

    Returns
    -------
    region_info_list: region info list
    """
    mem_ub = vbs_src[0]
    src_region_num_list = vbs_src[1]
    n_total_regions = vbs_src[2]

    region_info_list = []

    region_info_list.append({"offset": 0, "length":1,
                             "repeat": src_region_num_list[0]})
    if region_info_list[0]["length"] == 1:
        if n_total_regions % 16 != 0:
            raise RuntimeError("n_total_regions should be multiple of 16. {}"\
                .format(n_total_regions))
        region_info_list = region_info_list[1:]

        offset = 0
        n_repeat_total = n_total_regions//16
        while n_repeat_total > 0:
            n_repeat = min(n_repeat_total, 255)
            tik_inst.vrpsort16(dst=mem_ub[0, dest_pos_ub + offset, 0],
                               src=mem_ub[0, offset, 0],
                               repeat_times=int(n_repeat))

            offset += 16*n_repeat
            n_repeat_total -= n_repeat
        if offset != n_total_regions:
            raise RuntimeError("offset != n_total_regions")

        region_info = {"offset":0, "length":16, "repeat":n_total_regions//16}
        region_info_list.append(region_info)
    return region_info_list

def tik_topk_internal_vms4(tik_inst, mem_ub, dest_pos_ub, region_info_list):
    """
    do vms4

    Parameters
    ----------
    tik_inst: tik instance
    mem_ub: vbs src
    dest_pos_ub: dest_pos_ub
    region_info_list: region info list

    Returns
    -------
    dest_pos_ub: dest_pos_ub
    """
    src_pos_ub = 0
    n_vms4 = 0
    while True:
        if len(region_info_list) == 1 and region_info_list[0]["repeat"] == 1:
            break
        src_pos_ub, dest_pos_ub = dest_pos_ub, src_pos_ub

        # Do vms4 with repeat here
        n_vms4 += 1

        new_region_info_list = []
        while region_info_list:
            region_info = region_info_list[0]
            if region_info["repeat"] < 0:
                raise RuntimeError("repeat < 0")

            if region_info_list[0]["repeat"] >= VMS4_ELEMENT_NUM:
                if region_info_list[0]["length"]*VMS4_ELEMENT_NUM < \
                        MAX_INPUT_LIST_LENGTH:
                    region_info_list = tik_topk_merge_sort_same_subgroup(
                        tik_inst, (mem_ub, dest_pos_ub, src_pos_ub),
                        region_info_list, new_region_info_list)
                    continue

                region_info_list = tik_topk_merge_sort_reassign_subgroup(
                    tik_inst, (mem_ub, dest_pos_ub, src_pos_ub),
                    region_info_list, new_region_info_list)
                continue

            region_info_list = tik_topk_merge_sort_tail_subgroup(
                tik_inst, (mem_ub, dest_pos_ub, src_pos_ub),
                region_info_list, new_region_info_list)
        region_info_list = new_region_info_list
    return dest_pos_ub

def tik_topk_merge_sort_internal(tik_inst, inter_sort_store, inter_split_param):
    """
    sort internal

    Parameters
    ----------
    tik_inst: tik instance
    inter_sort_store: is a list, the keys as follow:
        batch_id: batch id
        mem_ub: ub tensor
        regions_sorted: sorted tensor
        region_orig: region proposal tensor

    inter_split_param: is a list, the keys as follow:
        src_region_num_list: src region num list
        n_required: required tensor num
        src_pos_list: src pos list
        dest_pos: dest pos

    Returns
    -------
    NA
    """
    batch_id = inter_sort_store[0]
    mem_ub = inter_sort_store[1]
    regions_sorted = inter_sort_store[2]
    regions_orig = inter_sort_store[3]

    src_region_num_list = inter_split_param[0]
    n_required = inter_split_param[1]
    src_pos_list = inter_split_param[2]
    dest_pos = inter_split_param[3]

    if len(src_region_num_list) != len(src_pos_list):
        raise RuntimeError("len(src_region_num_list) != len(src_pos_list)")
    if not src_region_num_list or (len(src_region_num_list) > 4):
        raise RuntimeError("len(src_region_num_list) < 1 and len(src_region_num_list) > 4)")

    n_input_list = len(src_region_num_list)
    n_total_regions = sum(src_region_num_list)

    # 1. Move data from OUT to UB
    for i in range(n_input_list):
        if src_region_num_list[i] < 4:
            raise RuntimeError("len(src_region_num_list) < 4")
        tik_inst.data_move(
            mem_ub[0, 0, 0], regions_orig[batch_id, src_pos_list[i], 0], sid=0,
            nburst=1,
            burst=math.ceil(REGION_SIZE_INBYTE * 4 / 32) * (src_region_num_list[i] // 4),
            src_stride=0, dst_stride=0)

    # 2. vbs
    dest_pos_ub = HALF_UB_REGION_CAPACITY if n_input_list == 1 else \
        sum(src_region_num_list)
    region_info_list = tik_topk_internal_vbs(
        tik_inst, (mem_ub, src_region_num_list, n_total_regions), dest_pos_ub)
    # 3. vms4
    dest_pos_ub = tik_topk_internal_vms4(tik_inst, mem_ub, dest_pos_ub,
                                         region_info_list)

    # 4. Move Data from UB to OUT
    nburst_val = region_info_list[0]["offset"]
    if region_info_list[0]["offset"] == 0:
        nburst_val = n_required

    tik_inst.data_move(regions_sorted[batch_id, dest_pos, 0],
                       mem_ub[0, dest_pos_ub, 0], sid=0, nburst=1,
                       burst=math.ceil(REGION_SIZE_INBYTE * 4 / 32) * (nburst_val // 4),
                       src_stride=0, dst_stride=0)

def tik_topk_merge_subgroup(tik_inst, subgroup_store, subgroup_split_param):
    """
    merge sub group

    Parameters
    ----------
    tik_inst: tik instance
    subgroup_store: is a list, the keys as follow:
        batch_id: batch id
        mem_ub: ub tensor
        region_orig: region proposal tensor
        regions_sorted: sorted tensor
        mem_interm: intermediate tensor
    subgroup_split_param: is a list, the keys as follow:
        n_required: required tensor
        subtsk_dest_pos_list: subtask dest pos list
        subtsk_n_required_list:  subtask required list
        dest_pos: dest list
        task_id: task id
        level_from_leaf: split depth

    Returns
    -------
    NA
    """
    batch_id = subgroup_store[0]
    mem_ub = subgroup_store[1]
    regions_orig = subgroup_store[2]
    regions_sorted = subgroup_store[3]
    mem_interm = subgroup_store[4]

    n_required = subgroup_split_param[0]
    subtsk_dest_pos_list = subgroup_split_param[1]
    subtsk_n_required_list = subgroup_split_param[2]
    dest_pos = subgroup_split_param[3]
    task_id = subgroup_split_param[4]
    level_from_leaf = subgroup_split_param[5]

    if level_from_leaf % 2 == 0:
        merge_result = (regions_sorted if task_id == "0" else regions_orig)
        regions_orig = mem_interm
    else:
        merge_result = (regions_sorted if task_id == "0" else mem_interm)

    if len(subtsk_dest_pos_list) != len(subtsk_n_required_list):
        raise RuntimeError("POS_LIST.length != N_REGION.length")
    if sum(subtsk_n_required_list) < n_required:
        raise RuntimeError("sum{ orig_region_num } < n_required")

    tik_topk_merge_sort_external(
        tik_inst, (mem_ub, merge_result, regions_orig, batch_id, n_required),
        (subtsk_dest_pos_list, subtsk_n_required_list, dest_pos))

def tik_topk_merge_sort_same_subgroup(tik_inst, ub_info, region_info_list,
                                      new_region_info_list):
    """
    sub sorted group do vms4

    Parameters
    ----------
    tik_inst: tik instance
    ub_info: is a list, the keys as follow:
        mem_ub: ub tensor
        dest_pos_ub: dest pos ub
        src_pos_ub: src pos ub
    region_info_list: region info list
    new_region_info_list: new region info list

    Returns
    -------
    region_info_list: region info list
    """
    mem_ub = ub_info[0]
    dest_pos_ub = ub_info[1]
    src_pos_ub = ub_info[2]

    region_info = region_info_list[0]
    if region_info["repeat"] < 0:
        raise RuntimeError("repeat < 0")

    n_repeat = (region_info["repeat"] // VMS4_ELEMENT_NUM)
    n_remainder = (region_info_list[0]["repeat"] % VMS4_ELEMENT_NUM)

    offset = region_info["offset"]
    dst = mem_ub[0, dest_pos_ub+offset, 0]
    src_list = []
    src_list_lengths = []

    for i in range(VMS4_ELEMENT_NUM):
        src_list.append(mem_ub[0, src_pos_ub+offset, 0])
        src_list_lengths.append(region_info["length"])
        offset += region_info["length"]

    tik_inst.vmrgsort4(dst, src_list, src_list_lengths,
                       if_exhausted_suspension=False,
                       valid_bit="1111", repeat_times=n_repeat)

    new_region_info_list.append(
        {"offset": region_info["offset"],
         "length": region_info["length"]*VMS4_ELEMENT_NUM,
         "repeat": n_repeat})

    if n_remainder > 0:
        region_info_list[0]["offset"] +=\
            region_info["length"] * VMS4_ELEMENT_NUM*n_repeat
        region_info_list[0]["repeat"] = n_remainder
    else:
        region_info_list = region_info_list[1:]

    return region_info_list

def tik_topk_merge_sort_reassign_subgroup(tik_inst, ub_info, region_info_list,
                                          new_region_info_list):
    """
    reassign sub group when proposal's num larger than 4096

    Parameters
    ----------
    tik_inst: tik instance
    ub_info: is a list, the keys as follow:
        mem_ub: ub tensor
        dest_pos_ub: dest pos ub
        src_pos_ub: src pos ub
    region_info_list: region info list
    new_region_info_list: new region info list

    Returns
    -------
    region_info_list: region info list
    """
    mem_ub = ub_info[0]
    dest_pos_ub = ub_info[1]
    src_pos_ub = ub_info[2]

    region_info = region_info_list[0]

    if region_info["repeat"] < 0:
        raise RuntimeError("repeat < 0")
    if region_info["repeat"] < VMS4_ELEMENT_NUM:
        raise RuntimeError("repeat < VMS4_ELEMENT_NUM")
    if region_info["length"] >= MAX_INPUT_LIST_LENGTH:
        raise RuntimeError("length >= MAX_INPUT_LIST_LENGTH")
    if region_info["length"] * VMS4_ELEMENT_NUM < MAX_INPUT_LIST_LENGTH:
        raise RuntimeError("length < * VMS4_ELEMENT_NUM < MAX_INPUT_LIST_LENGTH")

    merge_sort_factor = VMS4_ELEMENT_NUM
    while region_info["length"]*merge_sort_factor >= MAX_INPUT_LIST_LENGTH:
        merge_sort_factor -= 1
    if merge_sort_factor < 2:
        raise RuntimeError("merge_sort_factor < 2")

    offset = region_info["offset"]
    dst = mem_ub[0, dest_pos_ub+offset, 0]
    src_list = [mem_ub[0, 0, 0] for i in range(VMS4_ELEMENT_NUM)]
    src_list_lengths = [0 for i in range(VMS4_ELEMENT_NUM)]
    valid_bit = 0

    for i in range(merge_sort_factor):
        src_list[i] = mem_ub[0, src_pos_ub + offset, 0]
        src_list_lengths[i] = region_info["length"]
        offset += region_info["length"]
        valid_bit += 2**i

    tik_inst.vmrgsort4(dst, src_list, src_list_lengths,
                       if_exhausted_suspension=False, valid_bit=valid_bit,
                       repeat_times=1)
    new_region_info_list.append(
        {"offset": region_info["offset"],
         "length": region_info["length"]*merge_sort_factor,
         "repeat": 1})
    if region_info_list[0]["repeat"] <= merge_sort_factor:
        raise RuntimeError("repeat <= merge_sort_factor")
    region_info_list[0]["offset"] += region_info["length"]*merge_sort_factor
    region_info_list[0]["repeat"] -= merge_sort_factor

    return region_info_list

def tik_topk_merge_sort_tail_subgroup(tik_inst, ub_info, region_info_list,
                                      new_region_info_list):
    """
    sort tail sub group

    Parameters
    ----------
    tik_inst: tik instance
    ub_info: is a list, the keys as follow:
        mem_ub: ub tensor
        dest_pos_ub: dest pos ub
        src_pos_ub: src pos ub
    region_info_list: region info list
    new_region_info_list: new region info list

    Returns
    -------
    region_info_list: region info list
    """
    mem_ub = ub_info[0]
    dest_pos_ub = ub_info[1]
    src_pos_ub = ub_info[2]

    region_info = region_info_list[0]

    if region_info["repeat"] <= 0:
        raise RuntimeError("repeat <= 0")
    if region_info["repeat"] >= VMS4_ELEMENT_NUM:
        raise RuntimeError("repeat >= VMS4_ELEMENT_NUM")
    if region_info["length"] >= MAX_INPUT_LIST_LENGTH:
        raise RuntimeError("length >= MAX_INPUT_LIST_LENGTH")

    new_region_info = {"offset": region_info["offset"],
                       "length": 0,
                       "repeat": 1}

    offset = region_info["offset"]
    dst = mem_ub[0, dest_pos_ub + offset, 0]
    src_list = [mem_ub[0, 0, 0] for i in range(VMS4_ELEMENT_NUM)]
    src_list_lengths = [0 for i in range(VMS4_ELEMENT_NUM)]
    valid_bit = 0
    for i in range(VMS4_ELEMENT_NUM):
        if not region_info_list:
            break

        region_info = region_info_list[0]
        src_list[i] = mem_ub[0, src_pos_ub + region_info["offset"], 0]
        src_list_lengths[i] = region_info["length"]
        valid_bit += 2**i

        new_region_info["length"] += region_info["length"]
        region_info_list[0]["repeat"] -= 1
        region_info_list[0]["offset"] += region_info["length"]
        if region_info_list[0]["repeat"] == 0:
            region_info_list = region_info_list[1:]

    if valid_bit > 1:
        tik_inst.vmrgsort4(dst=dst,
                           src_list=src_list,
                           element_lengths=src_list_lengths,
                           if_exhausted_suspension=False,
                           valid_bit=valid_bit,
                           repeat_times=1)
    else:
        tik_inst.data_move(dst, src_list[0], sid=0,
                           nburst=1,
                           burst=math.ceil(REGION_SIZE_INBYTE * 4 / 32)
                           * (src_list_lengths[0] // 4),
                           src_stride=0, dst_stride=0)
    new_region_info_list.append(new_region_info)
    return region_info_list

def tik_topk_external_out_to_ub(tik_inst, data_src, slot_val):
    """
    data from out to ub when sortting external

    Parameters
    ----------
    tik_inst: tik instance
    data_src: is a list, the keys as follow:
        regions_orig: original tensors
        n_src_list: src list
        src_list_rem_:list remian num
        slot_capacity: slot capacity
        ms_src_list_len_: list length
        src_pos_: src pos
        batch_id: batch id
    slot_val: is a list, the keys as follow:
        ms_valid_bit_: ms_valid_bit_
        list_slot_map_: list_slot_map_
        n_burst_: n_burst_
        ms_src_list: ms_src_list

    Returns
    -------
    ms_valid_bit_: ms_valid_bit_
    list_slot_map_: list slot map
    n_burst_: nburst
    ms_src_list: dst slot data
    """
    regions_orig = data_src[0]
    n_src_list = data_src[1]
    src_list_rem_ = data_src[2]
    slot_capacity = data_src[3]
    ms_src_list_len_ = data_src[4]
    src_pos_ = data_src[5]
    batch_id = data_src[6]

    ms_valid_bit_ = slot_val[0]
    list_slot_map_ = slot_val[1]
    n_burst_ = slot_val[2]
    ms_src_list = slot_val[3]

    ms_valid_bit_.set_as(0)
    variable_temp_ = tik_inst.Scalar("int64", "variable_temp_", 0)

    for list_idx in range(n_src_list):
        with tik_inst.if_scope(src_list_rem_[list_idx] > 0):
            list_slot_map_[list_idx].set_as(variable_temp_)

            for slot_idx in range(n_src_list):

                with tik_inst.if_scope(slot_idx == variable_temp_):
                    tik_topk_min(tik_inst, slot_capacity,
                                 src_list_rem_[list_idx],
                                 ms_src_list_len_[slot_idx])

                    with tik_inst.if_scope((ms_src_list_len_[slot_idx] % 4) > 0):
                        n_burst_.set_as(ms_src_list_len_[slot_idx]//4 + 1)
                    with tik_inst.else_scope():
                        n_burst_.set_as(ms_src_list_len_[slot_idx]//4)

                    ms_valid_bit_.set_as(ms_valid_bit_ + 2**slot_idx)
                    tik_inst.data_move(
                        ms_src_list[slot_idx],
                        regions_orig[batch_id, src_pos_[list_idx], 0],
                        sid=0,
                        nburst=1,
                        burst=math.ceil(REGION_SIZE_INBYTE * 4 / 32) * n_burst_,
                        src_stride=0,
                        dst_stride=0)

            variable_temp_.set_as(variable_temp_ + 1)

    return ms_valid_bit_, list_slot_map_, n_burst_, ms_src_list

def tik_top_external_set_slot_data(tik_inst, src_list_set):
    """
    setting slot data when sortting external

    Parameters
    ----------
    tik_inst: tik instance
    src_list_set: is a list, the keys as follow:
        vms4_sr: vms4_sr
        n_src_list: src list
        num_exhausted_: num_exhausted_
        n_selected_: select num
        ms_valid_bit_: ms_valid_bit_
        list_slot_map_: list map
        src_pos_: src pos
        src_list_rem_: list remian num

    Returns
    -------
    n_selected_: select data
    """
    vms4_sr = src_list_set[0]
    n_src_list = src_list_set[1]
    num_exhausted_ = src_list_set[2]
    n_selected_ = src_list_set[3]
    ms_valid_bit_ = src_list_set[4]
    list_slot_map_ = src_list_set[5]
    src_pos_ = src_list_set[6]
    src_list_rem_ = src_list_set[7]

    num_exhausted = [tik_inst.Scalar('int64') for i in range(VMS4_ELEMENT_NUM)]
    tik_inst.mov_vmrgsort4_sr_to_scalar(num_exhausted, vms4_sr)
    for i in range(n_src_list):
        num_exhausted_[i].set_as(num_exhausted[i])

    n_selected_.set_as(0)
    for slot_idx in range(n_src_list):
        with tik_inst.if_scope(ms_valid_bit_ & (0x01 << slot_idx)):
            n_selected_.set_as(n_selected_ + num_exhausted_[slot_idx])
            for list_idx in range(n_src_list):
                with tik_inst.if_scope(list_slot_map_[list_idx] == slot_idx):
                    src_pos_[list_idx].set_as(src_pos_[list_idx]
                                              + num_exhausted_[slot_idx])
                    src_list_rem_[list_idx].set_as(src_list_rem_[list_idx]
                                                   - num_exhausted_[slot_idx])

    return n_selected_

def tik_top_external_ub_to_out(tik_inst, sorted_data, move_data_param):
    """
    data from ub to out when sortting external

    Parameters
    ----------
    tik_inst: tik instance
    sorted_data: is a list, the keys as follow:
        regions_sorted: sorted tensor
        batch_id: batch id
        dest_pos: dest pis
        ms_dest: ms dest

    move_data_param: is a list, the keys as follow:
        dest_pos_: dest pos
        n_selected_: select num
        n_total_rem_: total remain num
        n_burst_:  n_burst_
        n_total_selected_ = total select num

    Returns
    -------
    NA
    """
    regions_sorted = sorted_data[0]
    batch_id = sorted_data[1]
    dest_pos = sorted_data[2]
    ms_dest = sorted_data[3]

    dest_pos_ = move_data_param[0]
    n_selected_ = move_data_param[1]
    n_total_rem_ = move_data_param[2]
    n_burst_ = move_data_param[3]
    n_total_selected_ = move_data_param[4]

    tik_topk_min(tik_inst, n_selected_, n_total_rem_, n_selected_)

    n_burst_.set_as(n_selected_ // 4)
    with tik_inst.if_scope((n_selected_ & 0x3) > 0):
        n_burst_.set_as(n_burst_ + 1)
    dest_pos_.set_as(dest_pos + n_total_selected_)

    with tik_inst.if_scope(n_burst_ > 0):
        tik_inst.data_move(regions_sorted[batch_id, dest_pos_, 0], ms_dest, 0,
                           nburst=1,
                           burst=math.ceil(REGION_SIZE_INBYTE * 4 / 32) * n_burst_,
                           src_stride=0, dst_stride=0)

    # Step-4: Do post update
    n_total_selected_.set_as(n_total_selected_ + n_selected_)
    with tik_inst.if_scope(n_total_rem_ > n_selected_):
        n_total_rem_.set_as(n_total_rem_ - n_selected_)
    with tik_inst.else_scope():
        n_total_rem_.set_as(0)

def tik_topk_merge_sort_external(tik_inst, data_store, sub_list):
    """
    external sort

    Parameters
    ----------
    tik_inst: tik instance
    data_store: is a list, the keys as follow:
        mem_ub: ub trensor
        regions_sorted: sorted regions
        regions_orig: original regions
        batch_id: batch id
        n_required: required tensor num
    sub_list: is a list, the keys as follow:
        src_pos_list: src pos list
        src_region_num_list: src region num list
        dest_pos: dest pis

    Returns
    -------
    NA
    """
    mem_ub = data_store[0]
    regions_sorted = data_store[1]
    regions_orig = data_store[2]
    batch_id = data_store[3]
    n_required = data_store[4]

    src_pos_list = sub_list[0]
    src_region_num_list = sub_list[1]
    dest_pos = sub_list[2]

    n_src_list = len(src_pos_list)
    slot_capacity = UB_SIZE//(n_src_list*2)//REGION_SIZE_INBYTE

    n_total_selected_ = tik_inst.Scalar("int64", "n_total_selected_", 0)
    n_total_rem_ = tik_inst.Scalar("int64", "n_total_rem_", n_required)
    n_selected_ = tik_inst.Scalar("int64", "n_selected_", 0)
    n_burst_ = tik_inst.Scalar("int64", "n_burst_", 0)
    dest_pos_ = tik_inst.Scalar("int64", "dest_pos_", dest_pos)
    ms_valid_bit_ = tik_inst.Scalar("int64", "ms_valid_bit_", 0)

    src_pos_ = [tik_inst.Scalar(dtype="int64") for i in range(n_src_list)]
    src_list_rem_ = [tik_inst.Scalar(dtype="int64") for i in range(n_src_list)]
    list_slot_map_ = [tik_inst.Scalar(dtype="int64") for i in range(n_src_list)]
    num_exhausted_ = [tik_inst.Scalar(dtype="int64") for i in range(n_src_list)]

    for i in range(n_src_list):
        src_pos_[i].set_as(src_pos_list[i])
        src_list_rem_[i].set_as(src_region_num_list[i])

    ms_dest = mem_ub[0, UB_SIZE//REGION_SIZE_INBYTE//2, 0]
    ms_src_list = [mem_ub[0, 0, 0] for i in range(VMS4_ELEMENT_NUM)]
    ms_src_list_len_ = [tik_inst.Scalar(dtype="int64") for i in
                        range(VMS4_ELEMENT_NUM)]

    for slot_idx in range(n_src_list):
        ms_src_list[slot_idx] = mem_ub[0, slot_capacity *slot_idx, 0]
        ms_src_list_len_[slot_idx].set_as(0)

    min_input_length = min(src_region_num_list)
    max_iteration = math.ceil(n_required /
                              int(min(slot_capacity, min_input_length)))

    with tik_inst.for_range(0, max_iteration):

        with tik_inst.if_scope(n_total_selected_ < n_required):
            # Step-1: Fullfill all the input slots on UB
            ms_valid_bit_, list_slot_map_, n_burst_, ms_src_list = \
                tik_topk_external_out_to_ub(
                    tik_inst,
                    (regions_orig, n_src_list, src_list_rem_, slot_capacity,
                     ms_src_list_len_, src_pos_, batch_id),
                    (ms_valid_bit_, list_slot_map_, n_burst_, ms_src_list))

            with tik_inst.if_scope(ms_valid_bit_ > 0):
                # Step-2: Do sort with exhausted suspend mode enabled
                vms4_sr = tik_inst.vmrgsort4(
                    ms_dest, ms_src_list, element_lengths=ms_src_list_len_,
                    if_exhausted_suspension=True, valid_bit=ms_valid_bit_,
                    repeat_times=1)

                # Step-3: Move result from UB to OUT
                n_selected_ = tik_top_external_set_slot_data(
                    tik_inst,
                    (vms4_sr, n_src_list, num_exhausted_, n_selected_,
                     ms_valid_bit_, list_slot_map_, src_pos_, src_list_rem_))
                tik_top_external_ub_to_out(
                    tik_inst,
                    (regions_sorted, batch_id, dest_pos, ms_dest),
                    (dest_pos_, n_selected_, n_total_rem_, n_burst_,
                     n_total_selected_))
