#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

roi_pooling
"""
from topi.cce import util
from te import tik
from te import platform as tbe_platform

from impl.roi_pooling_128c0 import RoiClass128C0
from impl.roi_pooling_1c0_fm_l1 import RoiClassOneC0FML1
from impl.roi_pooling_onec0 import RoiOneC0Class
from impl.roi_pooling_l1 import RoiClassL1
from impl.roi_pooling_four_c0 import RoiClass4C0

from impl.roi_pooling_base import TYPELEN_DICT
from impl.roi_pooling_base import align


# pylint: disable=C0103
# pylint: disable=C0301
# pylint: disable=C0111
# pylint: disable=C0103
# pylint: disable=unused-argument,no-member
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals,too-many-lines
# pylint: disable=too-many-arguments,attribute-defined-outside-init

NoneType = type(None)


# 8 C0
def get_roi_ub_cost(pooled_h, pooled_w, proposal_num_per_tiling):
    """
    get roi ub cost of 8 C0
    """
    roi_start_h_cost = pooled_h * proposal_num_per_tiling * 4
    roi_start_w_cost = pooled_w * proposal_num_per_tiling * 4
    roi_bin_h_cost = pooled_h * proposal_num_per_tiling * 4
    roi_bin_w_cost = pooled_w * proposal_num_per_tiling * 4
    roi_start_w_from0_cost = pooled_w * proposal_num_per_tiling * 4
    proposals_ub_int32_cost = 5 * proposal_num_per_tiling * 4
    roi_height_cost = proposal_num_per_tiling * 4
    roi_width_cost = proposal_num_per_tiling * 4
    const_value_cost = 64 * 4
    const_zero_cost = 64 * 4
    calced_rois_scalar = 4
    range_end_scalar = 4
    proposal_ub_validnum = 4

    return roi_start_h_cost + roi_start_w_cost + roi_bin_h_cost + \
           roi_bin_w_cost + roi_start_w_from0_cost + \
           proposals_ub_int32_cost + roi_height_cost + roi_width_cost + \
           const_value_cost + const_zero_cost + calced_rois_scalar + \
           range_end_scalar + proposal_ub_validnum


def get_pool_ub_cost_128c0(block_num, fm_h, fm_w, fm_c0, dtype, pooled_h,
                           pooled_w):
    """
    get pooling ub cost of 8 C0
    """
    proposal_fm_data_cost = block_num * fm_h * fm_w * fm_c0 * TYPELEN_DICT[dtype]
    proposals_ub_batchid_scalar = 4

    pooled_h_res_cost = block_num*1*fm_w * fm_c0 * TYPELEN_DICT[dtype]
    pooled_res_cost = block_num*pooled_h*pooled_w*fm_c0*TYPELEN_DICT[dtype]
    scalar_roi_start_w_cost = 4
    scalar_roi_width_cost = 4
    scalar_roi_start_h_cost = 4
    scalar_roi_bin_h_cost = 4
    scalar_roi_start_w_from0_cost = 4
    scalar_roi_bin_w_cost = 4

    return proposal_fm_data_cost + proposals_ub_batchid_scalar + \
           (pooled_h_res_cost + pooled_res_cost + scalar_roi_start_w_cost + \
            scalar_roi_width_cost + scalar_roi_start_h_cost +
            scalar_roi_bin_h_cost + scalar_roi_start_w_from0_cost +
            scalar_roi_bin_w_cost)*2


# 4 C0
def get_4c0_ub_roi_cost(pooled_h, pooled_w):
    """
    get_4c0_ub_roi_cost
    """
    proposal_num_l1_ub_tiling = 128
    roi_start_h_cost = pooled_h * proposal_num_l1_ub_tiling * 4
    roi_start_w_cost = pooled_w * proposal_num_l1_ub_tiling * 4
    roi_bin_h_cost = pooled_h * proposal_num_l1_ub_tiling * 4
    roi_bin_w_cost = pooled_w * proposal_num_l1_ub_tiling * 4
    roi_start_w_from0_cost = pooled_w * proposal_num_l1_ub_tiling * 4
    proposals_ub_int32_cost = 5 * proposal_num_l1_ub_tiling * 4
    roi_width_cost = proposal_num_l1_ub_tiling*4

    return roi_start_h_cost + roi_start_w_cost + roi_bin_h_cost + \
           roi_bin_w_cost + roi_start_w_from0_cost + \
           proposals_ub_int32_cost + roi_width_cost


def get_pool_ub_cost_4c0(fm_h, fm_w, fm_c0, dtype, pooled_h, pooled_w):
    """
    get pooling ub cost of 4 C0
    """
    four_c0 = 4
    align_eight = 8
    proposal_fm_data_cost = four_c0 * fm_h * fm_w * fm_c0 * TYPELEN_DICT[dtype]

    pooled_h_res_cost = four_c0 * 1 * (fm_w + align_eight - 1) // align_eight \
                        * fm_c0 * TYPELEN_DICT[dtype]
    pooled_res_cost = four_c0 * pooled_h * pooled_w * fm_c0 \
                      * TYPELEN_DICT[dtype]

    return proposal_fm_data_cost + (pooled_h_res_cost + pooled_res_cost) * 2 \
           + get_4c0_ub_roi_cost(pooled_h, pooled_w)

def get_bin_one_ub(pooled_h, pooled_w, feature_batch, dtype):
    proposal_num_l1_ub_tiling = 128
    output_offset = feature_batch * 64 * 4 
    roi_actual_num_ub = 8 * 4
    roi_height = proposal_num_l1_ub_tiling * 4
    roi_width = proposal_num_l1_ub_tiling * 4
    const_value = 4 * 64
    const_zero = 4 * 64
    bin_h_fp16 = (pooled_h + 1) * proposal_num_l1_ub_tiling * TYPELEN_DICT[dtype]
    bin_w_fp16 = (pooled_w + 1) * proposal_num_l1_ub_tiling * TYPELEN_DICT[dtype]
    proposals_ub = 5 * proposal_num_l1_ub_tiling * TYPELEN_DICT[dtype]
    res = output_offset+roi_actual_num_ub+roi_height+roi_width+const_value+const_zero+bin_w_fp16+bin_h_fp16+proposals_ub
    return res
# 1 C0
def get_roi_onec0_ub_cost(fm_h, fm_w, fm_c0, dtype, pooled_h, pooled_w, feature_batch):
    fm_w_align = align(fm_w, 8)
    proposal_fm_data_cost = (fm_h + 2) * fm_w_align * fm_c0 * \
                        TYPELEN_DICT[dtype]
    pooled_h_align = align(pooled_h, 8)
    pooled_h_res_cost = pooled_h_align * fm_w_align * fm_c0 * \
                        TYPELEN_DICT[dtype]
    pooled_res_cost = pooled_h_align * pooled_w * fm_c0 * \
                      TYPELEN_DICT[dtype]

    return proposal_fm_data_cost + (pooled_h_res_cost+pooled_res_cost) * 2 + \
           get_4c0_ub_roi_cost(pooled_h, pooled_w)+ get_bin_one_ub(pooled_h, pooled_w, feature_batch, dtype)



# 1 C0 and rois in L1
def get_roi_onec0_posl1_ub_rois_cost(pooled_h, pooled_w):
    """
    get_roi_onec0_posl1_ub_cost
    """
    proposal_num_l1_ub_tiling = 8
    roi_start_h_cost = pooled_h * proposal_num_l1_ub_tiling * 4
    roi_start_w_cost = pooled_w * proposal_num_l1_ub_tiling * 4
    roi_bin_h_cost = pooled_h * proposal_num_l1_ub_tiling * 4
    roi_bin_w_cost = pooled_w * proposal_num_l1_ub_tiling * 4
    roi_start_w_from0_cost = pooled_w * proposal_num_l1_ub_tiling * 4
    proposals_ub_int32_cost = 5 * proposal_num_l1_ub_tiling * 4
    roi_width_cost = proposal_num_l1_ub_tiling * 4

    return roi_start_h_cost + roi_start_w_cost + roi_bin_h_cost + \
           roi_bin_w_cost + roi_start_w_from0_cost + \
           proposals_ub_int32_cost + roi_width_cost


def get_roi_onec0_posl1_ub_fm_cost(fm_h, fm_w, fm_c0, dtype, pooled_h,
                                   pooled_w):
    """
    get_roi_onec0_posl1_ub_fm_cost
    """

    fm_w_align = align(fm_w, 8)
    proposal_fm_data_cost = (fm_h + 2) * fm_w_align * fm_c0 * TYPELEN_DICT[dtype]
    pooled_h_align = align(pooled_h, 8)
    pooled_h_res_cost = pooled_h_align * fm_w_align * fm_c0 * TYPELEN_DICT[dtype]
    pooled_res_cost = pooled_h_align * pooled_w * fm_c0 * TYPELEN_DICT[dtype]

    return proposal_fm_data_cost + (pooled_h_res_cost+pooled_res_cost) * 2


def get_roi_onec0_posl1_ub_cost(fm_h, fm_w, fm_c0, dtype, pooled_h, pooled_w):
    """
    get roi ub cost of one c0 posl1
    """
    return get_roi_onec0_posl1_ub_rois_cost(pooled_h, pooled_w) + \
           get_roi_onec0_posl1_ub_fm_cost(fm_h, fm_w, fm_c0, dtype,
                                          pooled_h, pooled_w)


def get_roi_onec0_posl1_l1_cost(pooled_h, pooled_w, propsal_num_pertiling):
    """
    get_roi_onec0_posl1_l1_cost
    """
    roi_start_h_cost = pooled_h * propsal_num_pertiling * 4
    roi_start_w_cost = pooled_w * propsal_num_pertiling * 4
    roi_bin_h_cost = pooled_h * propsal_num_pertiling * 4
    roi_bin_w_cost = pooled_w * propsal_num_pertiling * 4
    roi_start_w_from0_cost = pooled_w * propsal_num_pertiling * 4
    proposals_ub_int32_cost = 5 * propsal_num_pertiling * 4
    roi_width_cost = propsal_num_pertiling * 4

    return roi_start_h_cost + roi_start_w_cost + roi_bin_h_cost + \
           roi_bin_w_cost + roi_start_w_from0_cost + \
           proposals_ub_int32_cost + roi_width_cost


# 1 C0 and fm in L1
def get_l1_cost_1c0_fm_l1(fm_h, fm_w, fm_c0, dtype):
    """
    get_L1_cost_1C0_FML1
    """
    return fm_h * fm_w * fm_c0 * TYPELEN_DICT[dtype]


def get_pool_ub_cost_1c0_fm_l1(fm_h, fm_w_align, fm_c0, dtype, pooled_h,
                               pooled_w, res_pad):
    """
    get_pool_Ub_cost_1C0_FML1
    """
    proposal_fm_data_cost = (fm_h // pooled_h + 2) * fm_w_align * fm_c0 *\
                            TYPELEN_DICT[dtype]
    proposals_ub_batchid_scalar = 4

    pooled_h_res_cost = (pooled_h + res_pad) * fm_w_align * fm_c0
    pooled_res_cost = (pooled_h + res_pad) * pooled_w * fm_c0 * TYPELEN_DICT[dtype]

    proposals_ub_batchid = 4
    scalar_propoal_width_cost = 4

    scalar_roi_start_h_cost = 4
    scalar_roi_start_w_cost = 4

    scalar_roi_width_cost = 4
    scalar_roi_bin_h_cost = 4

    scalar_roi_start_w_from0_cost = 4
    scalar_roi_bin_w_cost = 4

    return proposals_ub_batchid + (scalar_propoal_width_cost + \
        proposal_fm_data_cost + proposals_ub_batchid_scalar + \
        pooled_h_res_cost + pooled_res_cost + scalar_roi_start_w_cost + \
        scalar_roi_width_cost + scalar_roi_start_h_cost + \
        scalar_roi_bin_h_cost + scalar_roi_start_w_from0_cost + \
        scalar_roi_bin_w_cost) * 2


def get_subroiclass(x_dict, pooled_h, pooled_w):
    """
    get_subroiclass
    """
    feature_batch = x_dict.get("shape")[0]
    fm_h = x_dict.get("shape")[2]
    fm_w = x_dict.get("shape")[3]
    fm_c0 = x_dict.get("shape")[4]
    dtype = x_dict.get("dtype").lower()
    
    proposal_num_per_tiling = 128
    ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
    l1_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L1_SIZE)

    ub_cost_8c0 = get_pool_ub_cost_128c0(8, fm_h, fm_w, fm_c0, dtype,
                                         pooled_h, pooled_w) + \
                  get_roi_ub_cost(pooled_h, pooled_w, proposal_num_per_tiling)

    ub_cost_4c0 = get_pool_ub_cost_4c0(fm_h, fm_w, fm_c0, dtype, pooled_h,
                                       pooled_w)

    ub_cost_onec0 = get_roi_onec0_ub_cost(fm_h, fm_w, fm_c0, dtype, pooled_h,
                                          pooled_w, feature_batch)

    roi_onec0_posl1_ub = get_roi_onec0_posl1_ub_cost(fm_h, fm_w, fm_c0,
                                                     dtype, pooled_h, pooled_w)
    get_roi_onec0_posl1_l1 = get_roi_onec0_posl1_l1_cost(pooled_h, \
            pooled_w, proposal_num_per_tiling)

    res_pad = 0 if((pooled_h % 8) == 0) else (align(pooled_h, 8) - pooled_h)
    fm_w_align = align(fm_w, 8)
    ub_cost_1c0_fm_l1 = get_pool_ub_cost_1c0_fm_l1(fm_h, fm_w_align, fm_c0, \
                                dtype, pooled_h, pooled_w, res_pad) + \
                        get_roi_ub_cost(pooled_h, pooled_w,
                                        proposal_num_per_tiling)

    if ub_size > ub_cost_8c0 and pooled_h <= 8:
        # 8c0
        return RoiClass128C0()
    elif ub_size >= ub_cost_4c0 and pooled_h <= 6:
        # 4c0, pooled_h must be smaller than 6
        return RoiClass4C0()
    elif ub_size >= ub_cost_onec0:
        # 1c0
        return RoiOneC0Class(0)
    elif (ub_size >= roi_onec0_posl1_ub) and \
            (l1_size >= get_roi_onec0_posl1_l1):
        # 1c0PosL1
        return RoiOneC0Class(1)
        
    elif l1_size > get_l1_cost_1c0_fm_l1(fm_h, fm_w, fm_c0, dtype) and \
            ub_size > ub_cost_1c0_fm_l1:
        # 1c0FML1
        return RoiClassOneC0FML1()
    else:
        # L1
        return RoiClassL1()


def safe_check(dicts, kernel_name):
    """
    check if the inputs are legal

    Parameters
    ----------
    dicts: (x_dict, rois_dict, actual_dict, y_dict)
    kernel_name: kernel name

    Returns
    -------
    None
    """
    x_shape = dicts[0].get("shape")
    x_dtype = dicts[0].get("dtype").lower()
    rois_shape = dicts[1].get("shape")
    rois_dtype = dicts[1].get("dtype").lower()

    y_dtype = dicts[3].get("dtype").lower()
    y_shape = dicts[3].get("shape")

    profile = tik.Dprofile()
    tik_name_check = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if tik_name_check in ("Ascend310", "Ascend910", "Hi3796CV300ES", "Hi3796CV300CS"):
        util.check_dtype_rule(x_dtype, ("float16",))
        util.check_dtype_rule(rois_dtype, ("float16",))
    else:
        util.check_dtype_rule(x_dtype, ("float16", "float32"))
        util.check_dtype_rule(rois_dtype, ("float16", "float32"))

    if x_dtype != rois_dtype or x_dtype != y_dtype:
        raise RuntimeError("dtype in x, rois and y must be equal")

    util.check_shape_rule(x_shape, min_dim=5, max_dim=5)
    util.check_tensor_shape_size(x_shape)
    util.check_shape_rule(rois_shape, min_dim=3, max_dim=3)
    util.check_tensor_shape_size(rois_shape)
    util.check_shape_rule(y_shape, min_dim=5, max_dim=5)
    util.check_tensor_shape_size(y_shape)

    roi_max_num = rois_shape[2]
    if roi_max_num > 6000 or roi_max_num % 16 != 0:
        raise ValueError("the dim 2 of rois_shape is illegal")

    util.check_kernel_name(kernel_name)


@util.check_input_type(dict, dict, (dict, NoneType), dict, int, int,
                       float, float, str)
def roi_pooling(x_dict, rois_dict, actual_dict, y_dict, pooled_h, pooled_w,
                spatial_scale_h, spatial_scale_w, kernel_name="roi_pooling"):
    """
    roi pooling interface

    Parameters
    ----------
    x_dict: feature map size and data type
    rois_dict: rois_dictsize and data type
    actual_dict: actual num of rois size and data type
    out_dict: output size and data type
    pooled_h: pooled_h size
    pooled_w: pooled_w size
    spatial_scale_h: spatial scale h
    spatial_scale_w: spatial scale w
    kernel_name: kernel name of roi pooling op

    Returns
    -------
    None
    """
    safe_check((x_dict, rois_dict, actual_dict, y_dict), kernel_name)

    roi_pooling_instance = get_subroiclass(x_dict, pooled_h, pooled_w)
    roi_pooling_instance.init_param((pooled_h, pooled_w),
                                    (x_dict, rois_dict, actual_dict, y_dict),
                                    (spatial_scale_h, spatial_scale_w),
                                    kernel_name)
    roi_pooling_instance.roi_pooling_main()
