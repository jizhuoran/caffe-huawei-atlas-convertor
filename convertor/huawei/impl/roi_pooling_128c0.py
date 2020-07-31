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

roi_pooling_128c0
"""
from te import tik
from impl.roi_pooling_base import RoiClass
from impl.roi_pooling_base import TYPELEN_DICT
from impl.roi_pooling_base import INDEX_C1
from impl.roi_pooling_base import INDEX_C0


# pylint: disable=C0103
# pylint: disable=unused-argument,no-member
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals,too-many-lines
# pylint: disable=too-many-arguments,attribute-defined-outside-init


EIGHT_C0 = 8


class RoiClass128C0(RoiClass):
    """
    class that execute RoiClass128C0
    """
    def __init__(self):
        """
        constructor of RoiClass128C0
        Parameters
        -------
        None
        """

    def init_param(self, roinum_pooledimg, shapedict_list, spatial_scale_list,
                   kernel_name):
        super(RoiClass128C0, self).init_param(roinum_pooledimg, shapedict_list,
                                              spatial_scale_list, kernel_name)

    def proposal_pooling_h(self, proposal_id, fm_c1_index):
        """
        load the pooled_h * fm_width size featuremap to ub. maxpooling
        according to h direction

        Parameters
        ----------
        proposal_id: which proposal is now being processed
        fm_c1_index: c1 index of the feature map

        Returns
        -------
        None
        """
        scalar_roi_start_w = self.tik_instance.Scalar("int32")
        scalar_roi_start_w.set_as(self.roi_start_w[0, proposal_id])
        scalar_roi_width = self.tik_instance.Scalar("int32")
        scalar_roi_width.set_as(self.roi_width[proposal_id])
        scalar_roi_start_h = self.tik_instance.Scalar("int32")
        scalar_roi_bin_h = self.tik_instance.Scalar("int32")

        pooled_h_res = self.tik_instance.Tensor(self.dtype, \
                shape=(EIGHT_C0, 1, self.fm_w, self.fm_c0), \
                scope=tik.scope_ubuf, name="pooled_h_res")

        pooled_res = self.tik_instance.Tensor(self.dtype, \
                shape=(EIGHT_C0, self.pooled_h, self.pooled_w, self.fm_c0), \
                                    scope=tik.scope_ubuf,name="pooled_res")
        with self.tik_instance.for_range(0, self.pooled_w) as w_index:
            self.tik_instance.vector_dup(256 // TYPELEN_DICT[self.dtype], \
                                         pooled_res[0, 0, w_index, 0],
                                         0,
                                         self.pooled_h,
                                         self.pooled_h * self.pooled_w,
                                         self.pooled_w)
        with self.tik_instance.for_range(0, self.pooled_h) as poolh:
            self.tik_instance.vector_dup(256 // TYPELEN_DICT[self.dtype], \
                                         pooled_h_res[0, 0, 0, 0],
                                         0,
                                         self.fm_w,
                                         self.fm_w,
                                         1)
            scalar_roi_start_h.set_as(self.roi_start_h[poolh,proposal_id])
            scalar_roi_bin_h.set_as(self.roi_bin_h[poolh, proposal_id])
            with self.tik_instance.if_scope(tik.all(scalar_roi_bin_h != \
                                                    0, scalar_roi_width != 0)):
                if self.fm_h * self.fm_w < 255:
                    with self.tik_instance.for_range(0, scalar_roi_width) \
                            as w_index:
                        self.tik_instance.vmax(256 // TYPELEN_DICT[self.dtype],
                                               pooled_h_res[0, 0, w_index, 0],
                                               self.proposal_fm_data[0, \
                                                    scalar_roi_start_h, \
                                                scalar_roi_start_w+w_index, 0],
                                               pooled_h_res[0, 0, w_index, 0],
                                               scalar_roi_bin_h,
                                               self.fm_w,
                                               self.fm_h*self.fm_w,
                                               self.fm_w,
                                               0,
                                               self.fm_w,
                                               0)
                else:
                    with self.tik_instance.for_range(0, EIGHT_C0) as c0_i:
                        with self.tik_instance.for_range(0, \
                             scalar_roi_width // 8) as loop_8w_i:
                            self.tik_instance.vec_max(
                                256 // TYPELEN_DICT[self.dtype],
                                pooled_h_res[c0_i, 0, 8*loop_8w_i, 0],
                                self.proposal_fm_data[c0_i,
                                                      scalar_roi_start_h,
                                                      scalar_roi_start_w + \
                                                          8*loop_8w_i, 0],
                                pooled_h_res[c0_i, 0, 8*loop_8w_i, 0],
                                scalar_roi_bin_h,
                                0,
                                self.fm_w*16*2 // 32,
                                0)

                        with self.tik_instance.if_scope(
                                scalar_roi_width % 8 != 0):
                            tmp_w = scalar_roi_width // 8*8
                            self.tik_instance.vmax(
                                (scalar_roi_width - tmp_w)*self.fm_c0,
                                pooled_h_res[c0_i, 0, tmp_w, 0],
                                self.proposal_fm_data[c0_i, \
                                          scalar_roi_start_h, \
                                          scalar_roi_start_w + tmp_w, 0],
                                pooled_h_res[c0_i, 0, tmp_w, 0],
                                scalar_roi_bin_h,
                                1,
                                1,
                                1,
                                0,
                                self.fm_w*16*2 // 32,
                                0)

                if(self.dtype == "float32"):
                    self.proposal_pooling_w_float32(proposal_id)
                else:
                    self.proposal_pooling_w_float16(proposal_id, poolh, \
                                                    pooled_res, pooled_h_res)
                pooled_res = pooled_res
        with self.tik_instance.if_scope(fm_c1_index != self.c1_looptime-1):
            self.tik_instance.data_move(
                self.y[self.ouput_proposal_offset + self.calced_rois
                       + proposal_id, fm_c1_index * EIGHT_C0, 0, 0, 0],
                pooled_res[0, 0, 0, 0],
                0,
                1,
                EIGHT_C0*self.pooled_h*self.pooled_w*\
                    TYPELEN_DICT[self.dtype]*self.fm_c0 // 32,
                0,
                0)

        with self.tik_instance.else_scope():
            self.tik_instance.data_move(
                self.y[self.ouput_proposal_offset + self.calced_rois
                       + proposal_id, fm_c1_index *  EIGHT_C0, 0, 0, 0],
                pooled_res[0, 0, 0, 0],
                0,
                1,
                (self.fm_c1 - fm_c1_index*EIGHT_C0)*self.pooled_h*\
                    self.pooled_w*TYPELEN_DICT[self.dtype]*self.fm_c0 // 32,
                0,
                0)

    def proposal_pooling_w_float16(self, proposal_id, poolh, pooled_res,
                                   pooled_h_res):
        """
        after max pooling from the h direction, then max pooling from the
        w direction
        Parameters
        ----------
        proposal_id: the id of proposal which is being processed
        Returns
        -------
        None
        """
        scalar_roi_start_w_from0 = self.tik_instance.Scalar("int32")
        scalar_roi_bin_w = self.tik_instance.Scalar("int32")
        with self.tik_instance.for_range(0, self.pooled_w) as poolw:
            scalar_roi_start_w_from0.set_as(self.roi_start_w_from0[poolw, \
                                                                   proposal_id])
            scalar_roi_bin_w.set_as(self.roi_bin_w[poolw, proposal_id])
            self.tik_instance.vmax(
                256//TYPELEN_DICT[self.dtype],
                pooled_res[0, poolh, poolw, 0],
                pooled_h_res[0, 0, scalar_roi_start_w_from0, 0],
                pooled_res[0, poolh,poolw, 0],
                scalar_roi_bin_w,
                self.pooled_w*self.pooled_h,
                self.fm_w,
                self.pooled_w*self.pooled_h,
                0,
                1,
                0)

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
        self.x = self.tik_instance.Tensor(self.dtype, self.shape,
                                          name="x", scope=tik.scope_gm)
        if self.roi_actual_num_effect:
            self.roi_actual_num = \
                self.tik_instance.Tensor(dtype="int32",
                                         shape=(self.feature_batch, 8),
                                         name="roi_actual_num",
                                         scope=tik.scope_gm)

        self.rois = self.tik_instance.Tensor(self.dtype,
                                             shape=(self.feature_batch,
                                                    5, self.roi_max_num),
                                             name="rois", scope=tik.scope_gm)
        self.y = \
            self.tik_instance.Tensor(self.dtype,
                                     shape=(self.feature_batch*self.roi_max_num,
                                            self.shape[INDEX_C1],
                                            self.pooled_h, self.pooled_w,
                                            self.shape[INDEX_C0]), name="y",
                                     scope=tik.scope_gm)

        if self.feature_batch == 1:
            self.batch_factor = self.roi_max_num // self.device_core_num
            self.batch_factor_tail = self.roi_max_num - self.batch_factor*\
                                     self.device_core_num

            self.proposal_pooling_onebatch()
        else:
            self.batch_factor = self.feature_batch // self.device_core_num
            self.batch_factor_tail = self.feature_batch - self.batch_factor*\
                                     self.device_core_num

            self.proposal_pooling_multibatch()

        opt_config = {
            "double_buffer_non_reuse": True,
            "out_of_bound_sync_check": True
        }

        if self.roi_actual_num_effect:
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=(self.x,
                                               self.rois, self.roi_actual_num),
                                       outputs=(self.y,), enable_l2=False,
                                       config=opt_config)
        else:
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=(self.x,
                                               self.rois),
                                       outputs=(self.y,), enable_l2=False,
                                       config=opt_config)

    def proposal_pooling_onebatch(self):
        self.proposal_num_per_tiling = 128
        if self.roi_max_num % self.proposal_num_per_tiling == 0:
            self.tiling_num = self.roi_max_num // self.proposal_num_per_tiling
        else:
            self.tiling_num = self.roi_max_num // self.proposal_num_per_tiling \
                              + 1

        with self.tik_instance.for_range(0, self.tiling_num, \
                 block_num=self.tiling_num) as tiling_index:
            self.space_alloc(0)
            self.get_proposal_height_width(tiling_index, 0)
            self.init_pooled_proposal_start_hw()
            self.get_pooled_proposal_start_h()
            self.get_pooled_proposal_bin_h()
            self.get_pooled_proposal_start_w()
            self.get_pooled_proposal_bin_w()

            if self.fm_c1 % EIGHT_C0 == 0:
                self.c1_looptime = self.fm_c1 // EIGHT_C0
            else:
                self.c1_looptime = self.fm_c1 // EIGHT_C0 + 1

            self.proposal_fm_data = self.tik_instance.Tensor(
                self.dtype, (EIGHT_C0, self.fm_h, self.fm_w, self.fm_c0),
                name="proposal_fm_data", scope=tik.scope_ubuf)
            burst_len = self.fm_h*self.fm_w*self.fm_c0*\
                        TYPELEN_DICT[self.dtype] // 32

            with self.tik_instance.for_range(0, self.c1_looptime) as \
                    fm_c1_index:
                with self.tik_instance.if_scope(
                        fm_c1_index != self.c1_looptime - 1):
                    self.tik_instance.data_move(
                        self.proposal_fm_data,
                        self.x[0, fm_c1_index*EIGHT_C0, 0, 0, 0],
                        0,
                        EIGHT_C0,
                        burst_len,
                        0,
                        0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        self.proposal_fm_data,
                        self.x[0, fm_c1_index*EIGHT_C0, 0, 0, 0],
                        0,
                        self.fm_c1 - fm_c1_index*EIGHT_C0,
                        burst_len,
                        0,
                        0)
                self.calced_rois.set_as(self.proposal_num_per_tiling*\
                                        tiling_index)
                with self.tik_instance.for_range(0, self.proposal_ub_validnum,
                                                 thread_num=2) as proposal_id:
                    with self.tik_instance.if_scope(
                            (self.calced_rois + proposal_id) < self.range_end):
                        self.proposal_pooling_h(proposal_id, fm_c1_index)

    def proposal_pooling_multibatch_impl(self, batch_id):
        self.space_alloc(batch_id)
        with self.tik_instance.for_range(0, self.tiling_num) as tiling_index:
            self.get_proposal_height_width(tiling_index, batch_id)
            self.init_pooled_proposal_start_hw()
            self.get_pooled_proposal_start_h()
            self.get_pooled_proposal_bin_h()
            self.get_pooled_proposal_start_w()
            self.get_pooled_proposal_bin_w()

            if self.fm_c1 % EIGHT_C0 == 0:
                self.c1_looptime = self.fm_c1 // EIGHT_C0
            else:
                self.c1_looptime = self.fm_c1 // EIGHT_C0 + 1

            proposals_ub_batchid = self.tik_instance.Scalar("int32")
            self.proposal_fm_data = self.tik_instance.Tensor(
                self.dtype, (EIGHT_C0, self.fm_h, self.fm_w, self.fm_c0),
                name="proposal_fm_data", scope=tik.scope_ubuf)
            burst_len = self.fm_h*self.fm_w*self.fm_c0*\
                        TYPELEN_DICT[self.dtype]//32

            with self.tik_instance.for_range(0, self.c1_looptime)\
                    as fm_c1_index:
                with self.tik_instance.if_scope(
                        fm_c1_index != self.c1_looptime-1):
                    self.tik_instance.data_move(
                        self.proposal_fm_data,
                        self.x[batch_id, fm_c1_index * EIGHT_C0, 0, 0, 0],
                        0,
                        EIGHT_C0,
                        burst_len,
                        0,
                        0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        self.proposal_fm_data,
                        self.x[batch_id, fm_c1_index*EIGHT_C0, 0, 0, 0],
                        0,
                        self.fm_c1-fm_c1_index*EIGHT_C0,
                        burst_len,
                        0,
                        0)
                with self.tik_instance.for_range(0, self.proposal_ub_validnum, \
                                                 thread_num=2) as proposal_id:
                    with self.tik_instance.if_scope(
                            (self.calced_rois + proposal_id)< self.range_end):
                        proposals_ub_batchid.set_as(self.proposals_ub_int32[0,\
                                                                proposal_id])
                        with self.tik_instance.if_scope(
                                batch_id == proposals_ub_batchid):
                            self.proposal_pooling_h(proposal_id, fm_c1_index)
            self.calced_rois.set_as(self.calced_rois +
                                    self.proposal_ub_validnum)

    def proposal_pooling_multibatch(self):
        with self.tik_instance.for_range(0, self.device_core_num, \
                block_num=self.device_core_num) as block_id:
            with self.tik_instance.for_range(0, self.batch_factor)\
                    as batch_index:
                batch_id = block_id*self.batch_factor+batch_index
                self.proposal_pooling_multibatch_impl(batch_id)

            with self.tik_instance.if_scope(block_id < self.batch_factor_tail):
                batch_id = self.batch_factor*self.device_core_num + block_id
                self.proposal_pooling_multibatch_impl(batch_id)
