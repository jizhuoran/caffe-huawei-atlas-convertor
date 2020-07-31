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

roi_pooling_base
"""
from te import tik
from te import platform as tbe_platform

# pylint: disable=C0103
# pylint: disable=C0330
# pylint: disable=C0301
# pylint: disable=C0325
# pylint: disable=E1136
# pylint: disable=unused-argument,no-member
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals,too-many-lines
# pylint: disable=too-many-arguments,attribute-defined-outside-init


INDEX_N = 0
INDEX_C1 = 1
INDEX_H = 2
INDEX_W = 3
INDEX_C0 = 4
TYPELEN_DICT = {"float16": 2, "float32": 4}


def align(value, factor):
    """
    make value align to factor

    Parameters
    ----------
    value:  input number
    factor: factor

    Returns
    -------
    res:
    """
    return (value + factor - 1) // factor*factor


def ceil_div(value, factor):
    """
    if not divide exactlly  then plus 1

    Parameters
    ----------
    value:  input number
    factor: factor

    Returns
    -------
    res:
    """
    return (value + factor - 1)//factor


class RoiClass():
    """
    class that execute roi_pooling
    """
    def __init__(self):
        """
        constructor of RoiClass

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.pooled_h = None
        self.pooled_w = None

        self.dtype = None
        self.shape = None

        self.rois_dtype = None
        self.rois_shape = None

        self.output_shape = None

        self.spatial_scale_h = None
        self.spatial_scale_w = None

        self.roi_actual_num_effect = None

        self.kernel_name = None

        self.ouput_proposal_offset = None

        self.roi_max_num = None
        self.roi_actual_num_ub = None
        self.proposal_num_per_tiling = None
        self.tiling_num = None

        self.roi_start_h = None
        self.roi_start_w = None
        self.roi_bin_h = None
        self.roi_bin_w = None
        self.roi_start_w_from0 = None
        self.proposals_ub_int32 = None
        self.calced_rois = None
        self.range_end = None
        self.proposal_ub_validnum = None
        self.roi_height = None
        self.roi_width = None
        self.const_value = None
        self.const_zero = None

        self.tik_instance = None
        self.device_core_num = None

        self.roi_actual_num = None

        self.feature_batch = None
        self.fm_c1 = None
        self.fm_h = None
        self.fm_w = None
        self.fm_c0 = None

        self.output = None
        self.x = None
        self.rois = None
        self.y = None

    def init_param(self, pooled_hw, dicts, spatial_scale_list, kernel_name):
        """
        init parameters

        Parameters
        ----------
        pooled_hw: (pooled_h, pooled_w)
        dicts: (x_dict, rois_dict, actual_dict, y_dict)
        spatial_scale_list: (spatial_scale_h, spatial_scale_w)
        kernel_name: kernel name

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.pooled_h = pooled_hw[0]
        self.pooled_w = pooled_hw[1]

        self.dtype = dicts[0].get("dtype").lower()
        self.shape = dicts[0].get("shape")

        self.rois_dtype = dicts[1].get("dtype").lower()
        self.rois_shape = dicts[1].get("shape")

        self.output_shape = dicts[3].get("shape")

        self.spatial_scale_h = spatial_scale_list[0]
        self.spatial_scale_w = spatial_scale_list[1]

        self.roi_actual_num_effect = (dicts[2] != None)

        self.kernel_name = kernel_name

        self.feature_batch = self.shape[0]
        self.fm_c1 = self.shape[1]
        self.fm_h = self.shape[2]
        self.fm_w = self.shape[3]
        self.fm_c0 = self.shape[4]

        self.device_core_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

        self.proposal_num_per_tiling = 128
        self.roi_max_num = self.rois_shape[2]

    def cal_output_offset_with_actualnum(self, batch_id):
        """
        calculate output offset with actual_dict

        Parameters
        ----------
        batch_id : batch id

        Returns
        -------
        None
        """
        self.ouput_proposal_offset = self.tik_instance.Scalar("int32")
        self.roi_actual_num_ub = self.tik_instance.Scalar("int32")
        with self.tik_instance.new_stmt_scope():
            output_offset = \
                self.tik_instance.Tensor(dtype="int32",
                                         shape=(self.feature_batch, 64),
                                         scope=tik.scope_ubuf,
                                         name="output_offset")

            self.tik_instance.vec_dup(128//2,
                                         output_offset[0, 0],
                                         0, 64*4//256, 8)

            for row in range(self.feature_batch-1):
                self.tik_instance.data_move(output_offset[row+1, 0],
                                            self.roi_actual_num[row, 0],
                                            0, 1, 1, 0, 0)

            with self.tik_instance.for_range(1, self.feature_batch) as bid:
                self.tik_instance.vec_add(128//2, output_offset[bid, 0], \
                                       output_offset[bid, 0],
                                       output_offset[bid-1, 0], \
                                       64*4//256, 8, 8, 8)
            self.ouput_proposal_offset.set_as(output_offset[batch_id, 0])

            roi_actual_num_ub_tmp = \
                self.tik_instance.Tensor(dtype="int32", shape=(8,),
                                         name="roi_actual_num_ub",
                                         scope=tik.scope_ubuf)
            self.tik_instance.data_move(roi_actual_num_ub_tmp, \
                            self.roi_actual_num[batch_id, 0], \
                            0, 1, 1, 0, 0, 0)
            self.roi_actual_num_ub.set_as(roi_actual_num_ub_tmp[0])

    def cal_output_offset_without_actualnum(self, batch_id):
        """
        calculate output offset without actual_dict

        Parameters
        ----------
        batch_id : batch id

        Returns
        -------
        None
        """
        self.ouput_proposal_offset = self.tik_instance.Scalar("int32")
        self.ouput_proposal_offset.set_as(batch_id*self.roi_max_num)
        self.roi_actual_num_ub = self.tik_instance.Scalar("int32")
        self.roi_actual_num_ub.set_as(self.roi_max_num)

    def space_alloc(self, batch_id):
        """
        calculate Tensors's size and define them

        Parameters
        ----------
        batch_id : batch id

        Returns
        -------
        None
        """
        if self.roi_actual_num_effect:
            self.cal_output_offset_with_actualnum(batch_id)
        else:
            self.cal_output_offset_without_actualnum(batch_id)

        if self.roi_max_num % self.proposal_num_per_tiling == 0:
            self.tiling_num = self.roi_max_num // self.proposal_num_per_tiling
        else:
            self.tiling_num = self.roi_max_num // self.proposal_num_per_tiling \
                              + 1

        # all flooring pos roi_start_h
        self.roi_start_h = \
            self.tik_instance.Tensor("int32",
                                     [self.pooled_h,
                                      self.proposal_num_per_tiling],
                                     name="roi_start_h",
                                     scope=tik.scope_ubuf)
        self.roi_start_w = \
            self.tik_instance.Tensor("int32",
                                     [self.pooled_w,
                                      self.proposal_num_per_tiling],
                                     name="roi_start_w",
                                     scope=tik.scope_ubuf)

        # all ceiling pos  roi_bin_h
        self.roi_bin_h = \
            self.tik_instance.Tensor("int32", [self.pooled_h,
                                               self.proposal_num_per_tiling],
                                     name="roi_bin_h", scope=tik.scope_ubuf)
        self.roi_bin_w = \
            self.tik_instance.Tensor("int32",
                                     [self.pooled_w,
                                      self.proposal_num_per_tiling],
                                     name="roi_bin_w",
                                     scope=tik.scope_ubuf)
        self.roi_start_w_from0 = \
            self.tik_instance.Tensor("int32",
                                     shape=(self.pooled_w,
                                            self.proposal_num_per_tiling),
                                     scope=tik.scope_ubuf,
                                     name="roi_start_w_from0")
        self.proposals_ub_int32 = \
            self.tik_instance.Tensor("int32",
                                     [5, self.proposal_num_per_tiling],
                                     name="proposals_ub_int32",
                                     scope=tik.scope_ubuf)
        self.roi_height = \
            self.tik_instance.Tensor("int32",
                                     shape=(self.proposal_num_per_tiling,),
                                     scope=tik.scope_ubuf, name="roi_height")
        self.roi_width = \
            self.tik_instance.Tensor("int32",
                                     shape=(self.proposal_num_per_tiling,),
                                     scope=tik.scope_ubuf, name="roi_width")

        self.const_value = self.tik_instance.Tensor("int32", shape=(64,),
                                                    name="const_value",
                                                    scope=tik.scope_ubuf)
        self.const_zero = self.tik_instance.Tensor("int32", (64,),
                                                   name="const_zero",
                                                   scope=tik.scope_ubuf)

        self.calced_rois = self.tik_instance.Scalar("int32")
        self.calced_rois.set_as(0)
        self.range_end = self.tik_instance.Scalar("int32")
        self.range_end.set_as(self.roi_actual_num_ub)
        self.proposal_ub_validnum = self.tik_instance.Scalar("int32")

    def space_alloc_oneC0L1(self, batch_id):
        """
        calculate Tensors's size and define them

        Parameters
        ----------
        batch_id : batch id

        Returns
        -------
        None
        """
        if self.roi_actual_num_effect:
            self.cal_output_offset_with_actualnum(batch_id)
        else:
            self.cal_output_offset_without_actualnum(batch_id)

        if self.roi_max_num % self.proposal_num_per_tiling == 0:
            self.tiling_num = self.roi_max_num // self.proposal_num_per_tiling
        else:
            self.tiling_num = self.roi_max_num // self.proposal_num_per_tiling \
                              + 1

        self.calced_rois = self.tik_instance.Scalar("int32")
        self.calced_rois.set_as(0)
        self.range_end = self.tik_instance.Scalar("int32")
        self.range_end.set_as(self.roi_actual_num_ub)
        self.proposal_ub_validnum = self.tik_instance.Scalar("int32")

    def get_proposal_height_width(self, tiling_index, blockid):
        """
        calculate all proposals's height and width which are loaded to ub

        Parameters
        ----------
        tiling_index: load time, if there too many proposals, we have to load
                        them to ub particaly
        blockid : number of aicore

        Returns
        -------
        None
        """
        with self.tik_instance.new_stmt_scope():
            proposals_ub = \
                self.tik_instance.Tensor(self.dtype, \
                        shape=(5, self.proposal_num_per_tiling), \
                        name="proposals_ub", scope=tik.scope_ubuf)

            if self.tiling_num == 1:
                self.proposal_ub_validnum.set_as(self.roi_max_num - \
                        tiling_index*self.proposal_num_per_tiling)

                self.tik_instance.data_move(
                    proposals_ub[0, 0],
                    self.rois[blockid, 0,
                              tiling_index*self.proposal_num_per_tiling],
                    0,
                    5,
                    ((self.roi_max_num - tiling_index*\
                      self.proposal_num_per_tiling)*\
                      TYPELEN_DICT[self.dtype]) // 32,
                    (self.tiling_num - 1)*self.proposal_num_per_tiling*\
                            TYPELEN_DICT[self.dtype] // 32,
                    (self.proposal_num_per_tiling - self.roi_max_num)*\
                            TYPELEN_DICT[self.dtype] // 32)

            else:
                with self.tik_instance.if_scope(
                        tiling_index == (self.tiling_num - 1)):
                    self.proposal_ub_validnum.set_as(self.roi_max_num - \
                        tiling_index*self.proposal_num_per_tiling)

                    self.tik_instance.data_move(
                        proposals_ub[0, 0],
                        self.rois[blockid, 0,
                                  tiling_index*self.proposal_num_per_tiling],
                        0,
                        5,
                        (self.roi_max_num- tiling_index*\
                            self.proposal_num_per_tiling)*\
                            TYPELEN_DICT[self.dtype] // 32,
                        (self.tiling_num - 1)*self.proposal_num_per_tiling*\
                            TYPELEN_DICT[self.dtype] // 32,
                        (self.proposal_num_per_tiling - (self.roi_max_num - \
                            tiling_index*self.proposal_num_per_tiling))*\
                            TYPELEN_DICT[self.dtype] // 32)

                with self.tik_instance.else_scope():
                    self.proposal_ub_validnum.set_as(
                        self.proposal_num_per_tiling)
                    self.tik_instance.data_move(
                        proposals_ub[0, 0],
                        self.rois[blockid, 0, tiling_index*\
                                  self.proposal_num_per_tiling],
                        0,
                        5,
                        self.proposal_num_per_tiling*\
                            TYPELEN_DICT[self.dtype] // 32,
                        (self.roi_max_num - self.proposal_num_per_tiling)*\
                            TYPELEN_DICT[self.dtype] // 32,
                        0)

            self.tik_instance.vec_muls(
                256 // TYPELEN_DICT[self.dtype], proposals_ub[1, 0],
                proposals_ub[1, 0], self.spatial_scale_w,
                self.proposal_num_per_tiling*TYPELEN_DICT[self.dtype] // 256,
                8, 8)
            self.tik_instance.vec_muls(
                256 // TYPELEN_DICT[self.dtype], proposals_ub[2, 0],
                proposals_ub[2, 0], self.spatial_scale_h,
                self.proposal_num_per_tiling*TYPELEN_DICT[self.dtype] // 256,
                8, 8)
            self.tik_instance.vec_muls(
                256 // TYPELEN_DICT[self.dtype], proposals_ub[3, 0],
                proposals_ub[3, 0], self.spatial_scale_w,
                self.proposal_num_per_tiling*TYPELEN_DICT[self.dtype] // 256,
                8, 8)
            self.tik_instance.vec_muls(
                256 // TYPELEN_DICT[self.dtype], proposals_ub[4, 0],
                proposals_ub[4, 0], self.spatial_scale_h,
                self.proposal_num_per_tiling*TYPELEN_DICT[self.dtype] // 256,
                8, 8)

            self.tik_instance.vec_conv(128 // 2, "round", self.proposals_ub_int32,
                                    proposals_ub,
                                    self.proposal_num_per_tiling*5 // 64,
                                    8, 2*TYPELEN_DICT[self.dtype])

        self.tik_instance.vec_sub(128 // 2, self.roi_height, \
                               self.proposals_ub_int32[4, 0], \
                               self.proposals_ub_int32[2, 0], \
                               self.proposal_num_per_tiling // 64, \
                               8, 8, 8)
        self.tik_instance.vec_sub(128 // 2, self.roi_width, \
                               self.proposals_ub_int32[3, 0], \
                               self.proposals_ub_int32[1, 0], \
                               self.proposal_num_per_tiling // 64, \
                               8, 8, 8)

        self.tik_instance.vec_dup(64, self.const_value, 1, 1, 0)

        self.tik_instance.vec_dup(64, self.const_zero, 0, 1, 0)

        self.tik_instance.vec_add(128 // 2, self.roi_height, self.roi_height,
                               self.const_value,
                               self.proposal_num_per_tiling // 64,
                               8, 8, 0)
        self.tik_instance.vec_add(128 // 2, self.roi_width, self.roi_width,
                               self.const_value,
                               self.proposal_num_per_tiling // 64,
                               8, 8, 0)

        self.tik_instance.vec_max(128 // 2, self.roi_height, self.const_value,
                               self.roi_height,
                               self.proposal_num_per_tiling // 64,
                               8, 0, 8)
        self.tik_instance.vec_max(128 // 2, self.roi_width, self.const_value,
                               self.roi_width,
                               self.proposal_num_per_tiling // 64,
                               8, 0, 8)

    def get_proposal_height_width_param(self, tiling_index, blockid, proposals_ub_int32, roi_height, roi_width, const_value, const_zero):
        """
        calculate all proposals's height and width which are loaded to ub

        Parameters
        ----------
        tiling_index: load time, if there too many proposals, we have to load
                        them to ub particaly
        blockid : number of aicore

        Returns
        -------
        None
        """

        with self.tik_instance.new_stmt_scope():
            proposals_ub = \
                self.tik_instance.Tensor(self.dtype, \
                        shape=(5, self.proposal_num_per_tiling), \
                        name="proposals_ub", scope=tik.scope_ubuf)

            if self.tiling_num == 1:
                self.proposal_ub_validnum.set_as(self.roi_max_num - \
                        tiling_index*self.proposal_num_per_tiling)

                self.tik_instance.data_move(
                    proposals_ub[0, 0],
                    self.rois[blockid, 0,
                              tiling_index*self.proposal_num_per_tiling],
                    0,
                    5,
                    ((self.roi_max_num - tiling_index*\
                      self.proposal_num_per_tiling)*\
                      TYPELEN_DICT[self.dtype]) // 32,
                    (self.tiling_num - 1)*self.proposal_num_per_tiling*\
                            TYPELEN_DICT[self.dtype] // 32,
                    (self.proposal_num_per_tiling - self.roi_max_num)*\
                            TYPELEN_DICT[self.dtype] // 32)

            else:
                with self.tik_instance.if_scope(
                        tiling_index == (self.tiling_num - 1)):
                    self.proposal_ub_validnum.set_as(self.roi_max_num - \
                        tiling_index*self.proposal_num_per_tiling)

                    self.tik_instance.data_move(
                        proposals_ub[0, 0],
                        self.rois[blockid, 0,
                                  tiling_index*self.proposal_num_per_tiling],
                        0,
                        5,
                        (self.roi_max_num- tiling_index*\
                            self.proposal_num_per_tiling)*\
                            TYPELEN_DICT[self.dtype] // 32,
                        (self.tiling_num - 1)*self.proposal_num_per_tiling*\
                            TYPELEN_DICT[self.dtype] // 32,
                        (self.proposal_num_per_tiling - (self.roi_max_num - \
                            tiling_index*self.proposal_num_per_tiling))*\
                            TYPELEN_DICT[self.dtype] // 32)

                with self.tik_instance.else_scope():
                    self.proposal_ub_validnum.set_as(
                        self.proposal_num_per_tiling)
                    self.tik_instance.data_move(
                        proposals_ub[0, 0],
                        self.rois[blockid, 0, tiling_index*\
                                  self.proposal_num_per_tiling],
                        0,
                        5,
                        self.proposal_num_per_tiling*\
                            TYPELEN_DICT[self.dtype] // 32,
                        (self.roi_max_num - self.proposal_num_per_tiling)*\
                            TYPELEN_DICT[self.dtype] // 32,
                        0)

            self.tik_instance.vec_muls(
                256 // TYPELEN_DICT[self.dtype], proposals_ub[1, 0],
                proposals_ub[1, 0], self.spatial_scale_w,
                self.proposal_num_per_tiling*TYPELEN_DICT[self.dtype] // 256,
                8, 8)
            self.tik_instance.vec_muls(
                256 // TYPELEN_DICT[self.dtype], proposals_ub[2, 0],
                proposals_ub[2, 0], self.spatial_scale_h,
                self.proposal_num_per_tiling*TYPELEN_DICT[self.dtype] // 256,
                8, 8)
            self.tik_instance.vec_muls(
                256 // TYPELEN_DICT[self.dtype], proposals_ub[3, 0],
                proposals_ub[3, 0], self.spatial_scale_w,
                self.proposal_num_per_tiling*TYPELEN_DICT[self.dtype] // 256,
                8, 8)
            self.tik_instance.vec_muls(
                256 // TYPELEN_DICT[self.dtype], proposals_ub[4, 0],
                proposals_ub[4, 0], self.spatial_scale_h,
                self.proposal_num_per_tiling*TYPELEN_DICT[self.dtype] // 256,
                8, 8)

            self.tik_instance.vec_conv(128 // 2, "round", proposals_ub_int32,
                                    proposals_ub,
                                    self.proposal_num_per_tiling*5 // 64,
                                    8, 2*TYPELEN_DICT[self.dtype])

        self.tik_instance.vec_sub(128 // 2, roi_height, \
                               proposals_ub_int32[4, 0], \
                               proposals_ub_int32[2, 0], \
                               self.proposal_num_per_tiling // 64, \
                               8, 8, 8)
        self.tik_instance.vec_sub(128 // 2, roi_width, \
                               proposals_ub_int32[3, 0], \
                               proposals_ub_int32[1, 0], \
                               self.proposal_num_per_tiling // 64, \
                               8, 8, 8)

        self.tik_instance.vec_dup(64, const_value, 1, 1, 0)

        self.tik_instance.vec_dup(64, const_zero, 0, 1, 0)

        self.tik_instance.vec_add(128 // 2, roi_height, roi_height,
                               const_value,
                               self.proposal_num_per_tiling // 64,
                               8, 8, 0)
        self.tik_instance.vec_add(128 // 2, roi_width, roi_width,
                               const_value,
                               self.proposal_num_per_tiling // 64,
                               8, 8, 0)

        self.tik_instance.vec_max(128 // 2, roi_height, const_value,
                               roi_height,
                               self.proposal_num_per_tiling // 64,
                               8, 0, 8)
        self.tik_instance.vec_max(128 // 2, roi_width, const_value,
                               roi_width,
                               self.proposal_num_per_tiling // 64,
                               8, 0, 8)

    def init_pooled_proposal_start_hw(self):
        """
        calculate all proposals's h and w coordinate start from zero.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.new_scope():
            bin_h_fp16 = \
                self.tik_instance.Tensor(self.dtype,
                                         [self.pooled_h + 1,
                                          self.proposal_num_per_tiling],
                                         name="bin_h_fp16",
                                         scope=tik.scope_ubuf)
            bin_w_fp16 = \
                self.tik_instance.Tensor(self.dtype,
                                         [self.pooled_w + 1,
                                          self.proposal_num_per_tiling],
                                         name="bin_w_fp16",
                                         scope=tik.scope_ubuf)
            if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300CS","Hi3796CV300ES"):
                src_scalar = self.tik_instance.Scalar(dtype="int32")
                to_dst_scalar = self.tik_instance.Scalar(dtype="float32")
                to_dst_scalar_16 = self.tik_instance.Scalar(dtype="float16")
                with self.tik_instance.for_range(0, \
                        self.proposal_num_per_tiling) as i:
                    src_scalar.set_as(self.roi_height[i])
                    self.tik_instance.scalar_conv('', to_dst_scalar, src_scalar)
                    self.tik_instance.scalar_conv('', to_dst_scalar_16, to_dst_scalar)
                    bin_h_fp16[1, i].set_as(to_dst_scalar_16)
                    src_scalar.set_as(self.roi_width[i])
                    self.tik_instance.scalar_conv('', to_dst_scalar, src_scalar)
                    self.tik_instance.scalar_conv('', to_dst_scalar_16, to_dst_scalar)
                    bin_w_fp16[1, i].set_as(to_dst_scalar_16)
            else:
                if self.dtype == "float16":
                    self.tik_instance.vec_conv(128 // 2, "", bin_h_fp16[1, 0],
                                            self.roi_height,
                                            self.proposal_num_per_tiling // 64,
                                            2*TYPELEN_DICT[self.dtype], 8,
                                            1.0)
                    self.tik_instance.vec_conv(128 // 2,
                                            "", bin_w_fp16[1, 0],
                                            self.roi_width,
                                            self.proposal_num_per_tiling // 64,
                                            2*TYPELEN_DICT[self.dtype], 8,
                                            1.0)
                else:
                    self.tik_instance.vec_conv(128 // 2, "", bin_h_fp16[1, 0],
                                            self.roi_height,
                                            self.proposal_num_per_tiling // 64,
                                            2*TYPELEN_DICT[self.dtype], 8)
                    self.tik_instance.vec_conv(128 // 2,
                                            "", bin_w_fp16[1, 0],
                                            self.roi_width,
                                            self.proposal_num_per_tiling // 64,
                                            2*TYPELEN_DICT[self.dtype], 8)

            self.tik_instance.vec_dup(256 // TYPELEN_DICT[self.dtype],
                                         bin_h_fp16[0, 0], 0,
                                         self.proposal_num_per_tiling*\
                                            TYPELEN_DICT[self.dtype] // 256,
                                         8)
            self.tik_instance.vec_dup(256 // TYPELEN_DICT[self.dtype],
                                         bin_w_fp16[0, 0], 0,
                                         self.proposal_num_per_tiling*\
                                             TYPELEN_DICT[self.dtype] // 256,
                                         8)
            self.tik_instance.vec_muls(256 // TYPELEN_DICT[self.dtype],
                                    bin_h_fp16[1, 0],
                                    bin_h_fp16[1, 0],
                                    1.0 / self.pooled_h,
                                    self.proposal_num_per_tiling*\
                                        TYPELEN_DICT[self.dtype] // 256,
                                    8, 8)
            self.tik_instance.vec_muls(256 // TYPELEN_DICT[self.dtype],
                                    bin_w_fp16[1, 0],
                                    bin_w_fp16[1, 0],
                                    1.0 / self.pooled_w,
                                    self.proposal_num_per_tiling*\
                                        TYPELEN_DICT[self.dtype] // 256,
                                    8, 8)

            with self.tik_instance.for_range(2, self.pooled_h + 1) as i:
                self.tik_instance.vec_add(256 // TYPELEN_DICT[self.dtype],
                                       bin_h_fp16[i, 0],
                                       bin_h_fp16[i - 1, 0], bin_h_fp16[1, 0],
                                       self.proposal_num_per_tiling*\
                                           TYPELEN_DICT[self.dtype] // 256,
                                       8, 8, 8)
            with self.tik_instance.for_range(2, self.pooled_w + 1) as i:
                self.tik_instance.vec_add(256 // TYPELEN_DICT[self.dtype],
                                       bin_w_fp16[i, 0],
                                       bin_w_fp16[i - 1, 0], bin_w_fp16[1, 0],
                                       self.proposal_num_per_tiling*\
                                           TYPELEN_DICT[self.dtype] // 256,
                                       8, 8, 8)

            self.tik_instance.vec_conv(128 // 2, "floor", self.roi_start_h[0, 0],
                                    bin_h_fp16[0, 0],
                                    self.proposal_num_per_tiling *
                                    self.pooled_h // 64,
                                    8, 2*TYPELEN_DICT[self.dtype])
            self.tik_instance.vec_conv(128 // 2, "ceil", self.roi_bin_h[0, 0],
                                    bin_h_fp16[1, 0],
                                    self.proposal_num_per_tiling*\
                                        self.pooled_h // 64,
                                    8, 2*TYPELEN_DICT[self.dtype])
            self.tik_instance.vec_conv(128 // 2, "floor", self.roi_start_w[0, 0],
                                    bin_w_fp16[0, 0],
                                    self.proposal_num_per_tiling*\
                                        self.pooled_w // 64,
                                    8, 2*TYPELEN_DICT[self.dtype])
            self.tik_instance.vec_conv(128 // 2, "ceil", self.roi_bin_w[0, 0],
                                    bin_w_fp16[1, 0],
                                    self.proposal_num_per_tiling*\
                                        self.pooled_w // 64,
                                    8, 2*TYPELEN_DICT[self.dtype])

    def init_pooled_proposal_start_hw_param(self, roi_start_h, roi_start_w, roi_bin_h, roi_bin_w, roi_height, roi_width):
        """
        calculate all proposals's h and w coordinate start from zero.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        with self.tik_instance.new_scope():
            bin_h_fp16 = \
                self.tik_instance.Tensor(self.dtype,
                                         [self.pooled_h + 1,
                                          self.proposal_num_per_tiling],
                                         name="bin_h_fp16",
                                         scope=tik.scope_ubuf)
            bin_w_fp16 = \
                self.tik_instance.Tensor(self.dtype,
                                         [self.pooled_w + 1,
                                          self.proposal_num_per_tiling],
                                         name="bin_w_fp16",
                                         scope=tik.scope_ubuf)
            if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300CS"):
                src_scalar = self.tik_instance.Scalar(dtype="int32")
                to_dst_scalar = self.tik_instance.Scalar(dtype="float32")
                to_dst_scalar_16 = self.tik_instance.Scalar(dtype="float16")
                with self.tik_instance.for_range(0, \
                        self.proposal_num_per_tiling) as i:
                    src_scalar.set_as(roi_height[i])
                    self.tik_instance.scalar_conv('', to_dst_scalar, src_scalar)
                    self.tik_instance.scalar_conv('', to_dst_scalar_16,
                                                  to_dst_scalar)
                    bin_h_fp16[1, i].set_as(to_dst_scalar_16)
                    src_scalar.set_as(roi_width[i])
                    self.tik_instance.scalar_conv('', to_dst_scalar, src_scalar)
                    self.tik_instance.scalar_conv('', to_dst_scalar_16,
                                                  to_dst_scalar)
                    bin_w_fp16[1, i].set_as(to_dst_scalar_16)
            else:
                self.tik_instance.vec_conv(128 // 2, "", bin_h_fp16[1, 0],
                                        roi_height,
                                        self.proposal_num_per_tiling // 64,
                                        2*TYPELEN_DICT[self.dtype], 8,
                                        1.0)
                self.tik_instance.vec_conv(128 // 2,
                                        "", bin_w_fp16[1, 0],
                                        roi_width,
                                        self.proposal_num_per_tiling // 64,
                                        2*TYPELEN_DICT[self.dtype], 8,
                                        1.0)

            self.tik_instance.vec_dup(256 // TYPELEN_DICT[self.dtype],
                                         bin_h_fp16[0, 0], 0,
                                         self.proposal_num_per_tiling*\
                                            TYPELEN_DICT[self.dtype] // 256,
                                         8)
            self.tik_instance.vec_dup(256 // TYPELEN_DICT[self.dtype],
                                         bin_w_fp16[0, 0], 0,
                                         self.proposal_num_per_tiling*\
                                             TYPELEN_DICT[self.dtype] // 256,
                                         8)
            self.tik_instance.vec_muls(256 // TYPELEN_DICT[self.dtype],
                                    bin_h_fp16[1, 0],
                                    bin_h_fp16[1, 0],
                                    1.0 / self.pooled_h,
                                    self.proposal_num_per_tiling*\
                                        TYPELEN_DICT[self.dtype] // 256,
                                    8, 8)
            self.tik_instance.vec_muls(256 // TYPELEN_DICT[self.dtype],
                                    bin_w_fp16[1, 0],
                                    bin_w_fp16[1, 0],
                                    1.0 / self.pooled_w,
                                    self.proposal_num_per_tiling*\
                                        TYPELEN_DICT[self.dtype] // 256,
                                    8, 8)

            with self.tik_instance.for_range(2, self.pooled_h + 1) as i:
                self.tik_instance.vec_add(256 // TYPELEN_DICT[self.dtype],
                                       bin_h_fp16[i, 0],
                                       bin_h_fp16[i - 1, 0], bin_h_fp16[1, 0],
                                       self.proposal_num_per_tiling*\
                                           TYPELEN_DICT[self.dtype] // 256,
                                       8, 8, 8)
            with self.tik_instance.for_range(2, self.pooled_w + 1) as i:
                self.tik_instance.vec_add(256 // TYPELEN_DICT[self.dtype],
                                       bin_w_fp16[i, 0],
                                       bin_w_fp16[i - 1, 0], bin_w_fp16[1, 0],
                                       self.proposal_num_per_tiling*\
                                           TYPELEN_DICT[self.dtype] // 256,
                                       8, 8, 8)

            self.tik_instance.vec_conv(128 // 2, "floor", roi_start_h[0, 0],
                                    bin_h_fp16[0, 0],
                                    self.proposal_num_per_tiling *
                                    self.pooled_h // 64,
                                    8, 2*TYPELEN_DICT[self.dtype])
            self.tik_instance.vec_conv(128 // 2, "ceil", roi_bin_h[0, 0],
                                    bin_h_fp16[1, 0],
                                    self.proposal_num_per_tiling*\
                                        self.pooled_h // 64,
                                    8, 2*TYPELEN_DICT[self.dtype])
            self.tik_instance.vec_conv(128 // 2, "floor", roi_start_w[0, 0],
                                    bin_w_fp16[0, 0],
                                    self.proposal_num_per_tiling*\
                                        self.pooled_w // 64,
                                    8, 2*TYPELEN_DICT[self.dtype])
            self.tik_instance.vec_conv(128 // 2, "ceil", roi_bin_w[0, 0],
                                    bin_w_fp16[1, 0],
                                    self.proposal_num_per_tiling*\
                                        self.pooled_w // 64,
                                    8, 2*TYPELEN_DICT[self.dtype])
    def get_pooled_proposal_start_h(self):
        """
        calculate all proposals's h coordinate start from fm's height.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tik_instance.vector_dup(128 // 2,
                                     self.const_value,
                                     self.fm_h, 1, 0, 0, 0)
        with self.tik_instance.for_range(0, self.pooled_h) as i:
            self.tik_instance.vec_add(128 // 2, self.roi_start_h[i, 0],
                                   self.roi_start_h[i, 0], \
                                   self.proposals_ub_int32[2, 0], \
                                   self.proposal_num_per_tiling // 64, \
                                   8, 8, 8)

        self.tik_instance.vec_max(128 // 2, self.roi_start_h, self.roi_start_h, \
                               self.const_zero, \
                               self.proposal_num_per_tiling*\
                                   self.pooled_h // 64, \
                               8, 8, 0)
        self.tik_instance.vec_min(128//2, self.roi_start_h, self.roi_start_h, \
                               self.const_value, \
                               self.proposal_num_per_tiling*\
                                   self.pooled_h // 64, \
                               8, 8, 0)

    def get_pooled_proposal_start_h_param(self, roi_start_h, proposals_ub_int32, const_value, const_zero):
        """
        calculate all proposals's h coordinate start from fm's height.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.tik_instance.vector_dup(128 // 2,
                                     const_value,
                                     self.fm_h, 1, 0, 0, 0)
        with self.tik_instance.for_range(0, self.pooled_h) as i:
            self.tik_instance.vec_add(128 // 2, roi_start_h[i, 0],
                                   roi_start_h[i, 0], \
                                   proposals_ub_int32[2, 0], \
                                   self.proposal_num_per_tiling // 64, \
                                   8, 8, 8)

        self.tik_instance.vec_max(128 // 2, roi_start_h, roi_start_h, \
                               const_zero, \
                               self.proposal_num_per_tiling*\
                                   self.pooled_h // 64, \
                               8, 8, 0)
        self.tik_instance.vec_min(128//2, roi_start_h, roi_start_h, \
                               const_value, \
                               self.proposal_num_per_tiling*\
                                   self.pooled_h // 64, \
                               8, 8, 0)


    def get_pooled_proposal_bin_h(self):
        """
        calculate all proposals's sub height of all pooled h

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.pooled_h) as i:
            self.tik_instance.vec_add(128 // 2, self.roi_bin_h[i, 0],
                                   self.roi_bin_h[i, 0],
                                   self.proposals_ub_int32[2, 0],
                                   self.proposal_num_per_tiling // 64,
                                   8, 8, 8)

        self.tik_instance.vec_max(128 // 2, self.roi_bin_h, self.roi_bin_h,
                               self.const_zero,
                               self.proposal_num_per_tiling*self.pooled_h // 64,
                               8, 8,
                               0)
        self.tik_instance.vec_min(128 // 2, self.roi_bin_h, self.roi_bin_h,
                               self.const_value,
                               self.proposal_num_per_tiling*self.pooled_h // 64,
                               8, 8,
                               0)

        self.tik_instance.vec_sub(128//2, self.roi_bin_h, self.roi_bin_h,
                               self.roi_start_h,
                               self.proposal_num_per_tiling*self.pooled_h // 64,
                               8, 8,
                               8)

    def get_pooled_proposal_bin_h_param(self, roi_start_h, roi_bin_h, proposals_ub_int32, const_value, const_zero):
        """
        calculate all proposals's sub height of all pooled h

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        with self.tik_instance.for_range(0, self.pooled_h) as i:
            self.tik_instance.vec_add(128 // 2, roi_bin_h[i, 0],
                                   roi_bin_h[i, 0],
                                   proposals_ub_int32[2, 0],
                                   self.proposal_num_per_tiling // 64,
                                   8, 8, 8)

        self.tik_instance.vec_max(128 // 2, roi_bin_h, roi_bin_h,
                               const_zero,
                               self.proposal_num_per_tiling*self.pooled_h // 64,
                               8, 8,
                               0)
        self.tik_instance.vec_min(128 // 2, roi_bin_h, roi_bin_h,
                               const_value,
                               self.proposal_num_per_tiling*self.pooled_h // 64,
                               8, 8,
                               0)

        self.tik_instance.vec_sub(128//2, roi_bin_h, roi_bin_h,
                               roi_start_h,
                               self.proposal_num_per_tiling*self.pooled_h // 64,
                               8, 8,
                               8)

    def get_pooled_proposal_start_w(self):
        """
        calculate all proposals's w coordinate start from fm's width.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tik_instance.vector_dup(128 // 2, self.const_value, self.fm_w,
                                     1, 0, 0)
        with self.tik_instance.for_range(0, self.pooled_w) as i:
            self.tik_instance.vec_add(128 // 2, self.roi_start_w[i, 0],
                                   self.roi_start_w[i, 0],
                                   self.proposals_ub_int32[1, 0],
                                   self.proposal_num_per_tiling // 64,
                                   8, 8, 8)

        self.tik_instance.vec_max(128 // 2, self.roi_start_w, self.roi_start_w,
                               self.const_zero,
                               self.proposal_num_per_tiling*self.pooled_w // 64,
                               8, 8, 0)
        self.tik_instance.vec_min(128 // 2, self.roi_start_w, self.roi_start_w,
                               self.const_value,
                               self.proposal_num_per_tiling*self.pooled_w // 64,
                               8, 8, 0)
        with self.tik_instance.for_range(0, self.pooled_w) as i:
            self.tik_instance.vec_sub(64, self.roi_start_w_from0[i, 0],
                                   self.roi_start_w[i, 0],
                                   self.roi_start_w[0, 0],
                                   self.proposal_num_per_tiling // 64,
                                   8, 8, 8)

    def get_pooled_proposal_start_w_param(self, roi_start_w, roi_start_w_from0, proposals_ub_int32, const_value, const_zero):
        """
        calculate all proposals's w coordinate start from fm's width.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.tik_instance.vector_dup(128 // 2, const_value, self.fm_w,
                                     1, 0, 0)
        with self.tik_instance.for_range(0, self.pooled_w) as i:
            self.tik_instance.vec_add(128 // 2, roi_start_w[i, 0],
                                   roi_start_w[i, 0],
                                   proposals_ub_int32[1, 0],
                                   self.proposal_num_per_tiling // 64,
                                   8, 8, 8)

        self.tik_instance.vec_max(128 // 2, roi_start_w, roi_start_w,
                               const_zero,
                               self.proposal_num_per_tiling*self.pooled_w // 64,
                               8, 8, 0)
        self.tik_instance.vec_min(128 // 2, roi_start_w, roi_start_w,
                               const_value,
                               self.proposal_num_per_tiling*self.pooled_w // 64,
                               8, 8, 0)
        with self.tik_instance.for_range(0, self.pooled_w) as i:
            self.tik_instance.vec_sub(64, roi_start_w_from0[i, 0],
                                   roi_start_w[i, 0],
                                   roi_start_w[0, 0],
                                   self.proposal_num_per_tiling // 64,
                                   8, 8, 8)

    def get_pooled_proposal_bin_w(self):
        """
        calculate all proposals's sub width of all pooled w

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.pooled_w) as i:
            self.tik_instance.vec_add(128 // 2, self.roi_bin_w[i, 0],
                                   self.roi_bin_w[i, 0],
                                   self.proposals_ub_int32[1, 0],
                                   self.proposal_num_per_tiling // 64,
                                   8, 8, 8)
        self.tik_instance.vec_max(128 // 2, self.roi_bin_w, self.roi_bin_w,
                               self.const_zero,
                               self.proposal_num_per_tiling*self.pooled_w // 64,
                               8, 8, 0)
        self.tik_instance.vec_min(128 // 2, self.roi_bin_w, self.roi_bin_w,
                               self.const_value,
                               self.proposal_num_per_tiling*self.pooled_w // 64,
                               8, 8, 0)
        self.tik_instance.vec_sub(128 // 2, self.roi_width,
                               self.roi_bin_w[self.pooled_w - 1, 0],
                               self.roi_start_w[0, 0],
                               self.proposal_num_per_tiling // 64,
                               8, 8, 8)

        self.tik_instance.vec_sub(128 // 2, self.roi_bin_w, self.roi_bin_w,
                               self.roi_start_w,
                               self.proposal_num_per_tiling*self.pooled_w // 64,
                               8, 8, 8)

    def get_pooled_proposal_bin_w_param(self, roi_start_w, roi_bin_w, proposals_ub_int32, roi_width, const_value, const_zero):
        """
        calculate all proposals's sub width of all pooled w

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.pooled_w) as i:
            self.tik_instance.vec_add(128 // 2, roi_bin_w[i, 0],
                                   roi_bin_w[i, 0],
                                   proposals_ub_int32[1, 0],
                                   self.proposal_num_per_tiling // 64,
                                   8, 8, 8)
        self.tik_instance.vec_max(128 // 2, roi_bin_w, roi_bin_w,
                               const_zero,
                               self.proposal_num_per_tiling*self.pooled_w // 64,
                               8, 8, 0)
        self.tik_instance.vec_min(128 // 2, roi_bin_w, roi_bin_w,
                               const_value,
                               self.proposal_num_per_tiling*self.pooled_w // 64,
                               8, 8, 0)
        self.tik_instance.vec_sub(128 // 2, roi_width,
                               roi_bin_w[self.pooled_w - 1, 0],
                               roi_start_w[0, 0],
                               self.proposal_num_per_tiling // 64,
                               8, 8, 8)

        self.tik_instance.vec_sub(128 // 2, roi_bin_w, roi_bin_w,
                               roi_start_w,
                               self.proposal_num_per_tiling*self.pooled_w // 64,
                               8, 8, 8)

    def roi_pooling_main(self):
        """
        main process of roi pooling.
         including calculate the coordinate of pooled
        egde and max poolinf from the h direction
         and the from the w direction

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return

