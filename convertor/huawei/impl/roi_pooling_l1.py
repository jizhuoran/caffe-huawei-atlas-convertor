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

roi_pooling_l1
"""

from te import tik
from te import platform as cce
from te import platform as tbe_platform

from impl.roi_pooling_base import RoiClass
from impl.roi_pooling_base import TYPELEN_DICT
from impl.roi_pooling_base import INDEX_C1
from impl.roi_pooling_base import INDEX_C0

# pylint: disable=too-many-instance-attributes
class RoiClassL1(RoiClass):
    """
    Class that execute roi_pooling using L1
    """

    def __init__(self):
        """
        constructor of RoiClassL1
        Parameters
        -------
        None
        """
        super().__init__()
        self.l1_byte_size = None
        self.c1_loops = None
        self.c1_num_in_l1 = None

        self.fm_in_l1 = None
        self.pooled_fm_for_a_roi = None
        self.window_fm = None

        self.max_bin_h = None
        self.max_bin_w = None


    def init_param(self, roinum_pooledimg, shapedict_list,
                   spatial_scale_list, kernel_name):
        """
        init parameters of RoiClassL1
        Parameters
        -------
        None
        """
        super(RoiClassL1, self).init_param(roinum_pooledimg,
                                           shapedict_list,
                                           spatial_scale_list,
                                           kernel_name)
        self.roi_actual_num_effect = True

        self.l1_byte_size = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L1_SIZE)
        self.c1_loops = self.fm_c1
        self.c1_num_in_l1 = self.l1_byte_size // (self.fm_h *
                                                  self.fm_w *
                                                  self.fm_c0 *
                                                  TYPELEN_DICT[self.dtype])
        if self.fm_c1 <= self.c1_num_in_l1:
            self.c1_num_in_l1 = self.fm_c1
        self.max_bin_h = self.fm_h // self.pooled_h + 2
        self.max_bin_w = self.fm_w // self.pooled_w + 2

    def get_l1_cost(self):
        """
        get L1 cost
        Parameters
        -------
        None
        """
        return self.l1_byte_size

    # pylint: disable=no-self-use
    def get_roi_ub_cost(self):
        """
        get UB cost of roi (only calculate ub of current class)
        Parameters
        -------
        None
        """
        return 32

    def get_pool_ub_cost(self):
        """
        get UB cost of pooling
        Parameters
        -------
        None
        """
        ub_cost = self.pooled_h * self.pooled_w * self.fm_c0 * \
                  TYPELEN_DICT[self.dtype]
        ub_cost = ub_cost + \
                  self.max_bin_h * self.max_bin_w * self.fm_c0 * \
                  TYPELEN_DICT[self.dtype]
        ub_cost = ub_cost + self.fm_c0 * TYPELEN_DICT[self.dtype]
        return ub_cost

    def roi_pooling_main(self):
        """
        main procedure of roi_pooling
        Parameters
        -------
        None
        """
        if self.roi_actual_num_effect:
            self.roi_actual_num = \
                    self.tik_instance.Tensor(dtype="int32",
                                             shape=(self.feature_batch, 8),
                                             name="roi_actual_num",
                                             scope=tik.scope_gm)

        self.x = self.tik_instance.Tensor(self.dtype, self.shape, name="x",
                                          scope=tik.scope_gm)

        self.rois = self.tik_instance.Tensor(self.dtype,
                                             shape=(self.feature_batch,
                                                    5, self.roi_max_num),
                                             name="rois", scope=tik.scope_gm)

        self.y = self.tik_instance.Tensor(self.dtype,
                                          shape=(self.feature_batch * \
                                                 self.roi_max_num,
                                                 self.shape[INDEX_C1],
                                                 self.pooled_h, self.pooled_w,
                                                 self.shape[INDEX_C0]),
                                          name="y",
                                          scope=tik.scope_gm)

        if self.feature_batch == 1:
            self.proposal_pooling_multibatch()
        else:
            self.proposal_pooling_multibatch()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.x,
                                           self.rois,
                                           self.roi_actual_num),
                                   outputs=(self.y),
                                   enable_l2=False)

    def proposal_pooling_multibatch(self):
        """
        multibatch pooling process
        Parameters
        -------
        None
        """
        with self.tik_instance.for_range(0, self.feature_batch,
                                         block_num=self.feature_batch) \
                as block_id:
            self.roi_pooling_each_core(block_id)

    def roi_pooling_each_core(self, block_id):
        """
        pooling process for each core
        Parameters
        -------
        None
        """
        self.space_alloc(block_id)

        self.fm_in_l1 = self.tik_instance.Tensor(dtype=self.dtype,
                                                 shape=(self.c1_num_in_l1,
                                                        self.fm_h,
                                                        self.fm_w,
                                                        self.fm_c0),
                                                 scope=tik.scope_cbuf,
                                                 name="fm_in_l1")

        with self.tik_instance.for_range(0, self.c1_loops) as ci_loop:
            self.calced_rois.set_as(0)

            if self.c1_num_in_l1 == 1:
                self.tik_instance.data_move(
                    self.fm_in_l1[0, 0, 0, 0],
                    self.x[block_id, ci_loop, 0, 0, 0],
                    0,
                    1, (self.c1_num_in_l1 *
                        self.fm_h *
                        self.fm_w *
                        self.fm_c0 *
                        TYPELEN_DICT[self.dtype]) // 32,
                    0, 0)
            else:
                with self.tik_instance.if_scope(
                        ci_loop % self.c1_num_in_l1 == 0):
                    self.tik_instance.data_move(
                        self.fm_in_l1[0, 0, 0, 0],
                        self.x[block_id, ci_loop, 0, 0, 0],
                        0,
                        1, (self.c1_num_in_l1 *
                            self.fm_h *
                            self.fm_w *
                            self.fm_c0 *
                            TYPELEN_DICT[self.dtype]) // 32,
                        0, 0)

            with self.tik_instance.for_range(0, self.tiling_num) \
                    as roi_group_index:
                self.get_proposal_height_width(roi_group_index, block_id)
                self.init_pooled_proposal_start_hw()
                self.get_pooled_proposal_start_h()
                self.get_pooled_proposal_bin_h()
                self.get_pooled_proposal_start_w()
                self.get_pooled_proposal_bin_w()

                with self.tik_instance.for_range(0, self.proposal_ub_validnum)\
                        as roi_index:
                    with self.tik_instance.if_scope(
                            (self.calced_rois + roi_index) < self.range_end):
                        self.pooled_fm_for_a_roi = \
                            self.tik_instance.Tensor(dtype=self.dtype,
                                                     shape=(self.pooled_h,
                                                            self.pooled_w,
                                                            self.fm_c0),
                                                     scope=tik.scope_ubuf,
                                                     name="pooled_fm_for_a_roi")
                        with self.tik_instance.for_range(0, self.pooled_h) \
                                as ph_index:
                            with self.tik_instance.for_range(0, self.pooled_w)\
                                    as pw_index:
                                self.window_fm = self.tik_instance.Tensor(
                                    dtype=self.dtype,
                                    shape=(self.max_bin_h, self.max_bin_w,
                                           self.fm_c0),
                                    scope=tik.scope_ubuf,
                                    name="window_fm")

                                window_start_h = self.tik_instance.Scalar(
                                    "int32")
                                window_start_h.set_as(
                                    self.roi_start_h[ph_index, roi_index])
                                window_start_w = self.tik_instance.Scalar(
                                    "int32")
                                window_start_w.set_as(
                                    self.roi_start_w[pw_index, roi_index])
                                window_w = self.tik_instance.Scalar("int32")
                                window_w.set_as(self.roi_bin_w[pw_index,
                                                               roi_index])
                                window_h = self.tik_instance.Scalar("int32")
                                window_h.set_as(self.roi_bin_h[ph_index,
                                                               roi_index])
                                nburst = self.tik_instance.Scalar("int32")
                                nburst.set_as(self.roi_bin_h[ph_index,
                                                             roi_index])
                                burst_len = self.tik_instance.Scalar("int32")
                                burst_len.set_as((window_w * self.fm_c0 * \
                                    TYPELEN_DICT[self.dtype]) // 32)
                                src_gap = self.tik_instance.Scalar("int32")
                                src_gap.set_as(((self.fm_w - window_w) * \
                                    self.fm_c0 * \
                                    TYPELEN_DICT[self.dtype]) // 32)
                                dst_gap = self.tik_instance.Scalar("int32")
                                dst_gap.set_as((self.max_bin_w - window_w) * \
                                    self.fm_c0 * TYPELEN_DICT[self.dtype] // 32)
                                self.tik_instance.data_move(
                                    self.window_fm[0, 0, 0],
                                    self.fm_in_l1[ci_loop%self.c1_num_in_l1,
                                                  window_start_h,
                                                  window_start_w,
                                                  0],
                                    0,
                                    nburst, burst_len,
                                    src_gap, dst_gap)
                                self.window_pooling(roi_index,
                                                    ph_index, pw_index)

                        self.tik_instance.data_move(
                            self.y[self.ouput_proposal_offset +
                                   roi_group_index * \
                                   self.proposal_num_per_tiling + roi_index,
                                   ci_loop, 0, 0, 0],
                            self.pooled_fm_for_a_roi,
                            0,
                            1, (self.pooled_h * self.pooled_w * self.fm_c0 *
                                TYPELEN_DICT[self.dtype]) // 32,
                            0, 0)
                self.calced_rois.set_as(self.calced_rois +
                                        self.proposal_ub_validnum)

    def window_pooling(self, roi_index, ph_index, pw_index):
        """
        pooling process of a bin in a roi
        Parameters
        -------
        None
        """
        with self.tik_instance.new_stmt_scope():
            max_val = self.tik_instance.Tensor(dtype=self.dtype,
                                               shape=(self.fm_c0,),
                                               scope=tik.scope_ubuf,
                                               name="max_val")
            self.tik_instance.data_move(max_val, self.window_fm[0, 0, 0],
                                        0,
                                        1,
                                        (self.fm_c0 *
                                         TYPELEN_DICT[self.dtype]) // 32,
                                        0, 0)
            bin_h = self.tik_instance.Scalar("int32")
            bin_w = self.tik_instance.Scalar("int32")
            bin_h.set_as(self.roi_bin_h[ph_index, roi_index])
            bin_w.set_as(self.roi_bin_w[pw_index, roi_index])
            with self.tik_instance.for_range(0, bin_h) as b_h_index:
                with self.tik_instance.for_range(0, bin_w) as b_w_index:
                    self.tik_instance.vec_max(self.fm_c0,
                                           max_val,
                                           max_val,
                                           self.window_fm[b_h_index, b_w_index,
                                                          0],
                                           1,
                                           8, 8, 8)

            self.tik_instance.data_move(self.pooled_fm_for_a_roi[ph_index,
                                                                 pw_index, 0],
                                        max_val,
                                        0,
                                        1,
                                        (self.fm_c0 *
                                         TYPELEN_DICT[self.dtype]) // 32,
                                        0, 0)
