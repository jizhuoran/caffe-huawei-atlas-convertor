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
from te import tik

from impl.roi_pooling_base import RoiClass
from impl.roi_pooling_base import TYPELEN_DICT
from impl.roi_pooling_base import INDEX_C1
from impl.roi_pooling_base import INDEX_C0

from impl.roi_pooling_base import align
from impl.roi_pooling_base import ceil_div


# pylint: disable=C0103
# pylint: disable=unused-argument,no-member
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals,too-many-lines
# pylint: disable=too-many-arguments,attribute-defined-outside-init


class RoiClassOneC0FML1(RoiClass):
    """
    class that execute roi_pooling
    """
    def __init__(self):
        """
        constructor of RoiClass
        Parameters
        -------
        None
        """
        super().__init__()

        self.res_pad = None
        self.fm_w_align = None
        self.fm_c0_data = None

    def init_param(self, roinum_pooledimg, shapedict_list, spatial_scale_list, \
                   kernel_name):
        super(RoiClassOneC0FML1, self).init_param(roinum_pooledimg, \
                            shapedict_list, spatial_scale_list, kernel_name)
        self.res_pad = 0 if((self.pooled_h%8) == 0) else \
            (align(self.pooled_h, 8)-self.pooled_h)
        self.fm_w_align = align(self.fm_w, 8)

    def proposal_pooling_h(self, block_id, proposal_id, fm_c1_index):
        """
        load the pooled_h * fm_width size featuremap to ub. maxpooling accroing
        to h direction

        Parameters
        ----------
        block_id:  aicore id
        proposal_id: which proposal is now being processed
        fm_c1_index: c1 index of the feature map
        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.pooled_h) as poolh:
            proposal_fm_data = \
                self.tik_instance.Tensor(self.dtype,
                                         (self.fm_h//self.pooled_h+2,
                                          self.fm_w_align, self.fm_c0),
                                         name="proposal_data",
                                         scope=tik.scope_ubuf)

            scalar_roi_start_h = self.tik_instance.Scalar("int32")
            scalar_roi_start_h.set_as(self.roi_start_h[poolh, proposal_id])
            scalar_roi_start_w = self.tik_instance.Scalar("int32")
            scalar_roi_start_w.set_as(self.roi_start_w[0, proposal_id])
            scalar_roi_width = self.tik_instance.Scalar("int32")
            scalar_roi_width.set_as(self.roi_width[proposal_id])
            scalar_roi_bin_h = self.tik_instance.Scalar("int32")
            scalar_roi_bin_h.set_as(self.roi_bin_h[poolh, proposal_id])

            with self.tik_instance.if_scope(tik.all(
                    scalar_roi_bin_h != 0, scalar_roi_width != 0)):
                coeff = self.fm_c0*TYPELEN_DICT[self.dtype]//32
                self.tik_instance.data_move(proposal_fm_data,
                                            self.fm_c0_data[
                                                scalar_roi_start_h,
                                                scalar_roi_start_w,
                                                0],
                                            0,
                                            scalar_roi_bin_h,
                                            scalar_roi_width*coeff,
                                            (self.fm_w-
                                             scalar_roi_width) * coeff,
                                            (self.fm_w_align-
                                             scalar_roi_width) * coeff)

                ceil_loop = 16//TYPELEN_DICT[self.dtype]
                with self.tik_instance.for_range(0,
                                                 ceil_div(scalar_roi_width,
                                                          ceil_loop)) as \
                        loop_w:
                    self.tik_instance.vec_max(256//
                                           TYPELEN_DICT[self.dtype],
                                           self.pooled_h_res[poolh,
                                                             ceil_loop *
                                                             loop_w, 0],
                                           proposal_fm_data[0,
                                                            ceil_loop *
                                                            loop_w, 0],
                                           self.pooled_h_res[poolh,
                                                             ceil_loop *
                                                             loop_w, 0],
                                           scalar_roi_bin_h,
                                           0,
                                           self.fm_w_align*coeff,
                                           0)

    def proposal_pooling_w_float16(self, proposal_id):
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
        with self.tik_instance.for_range(0, self.pooled_w) as poolw:
            scalar_roi_start_w_from0 = self.tik_instance.Scalar("int32")
            scalar_roi_start_w_from0.set_as(
                self.roi_start_w_from0[poolw, proposal_id])

            scalar_roi_bin_w = self.tik_instance.Scalar("int32")
            scalar_roi_bin_w.set_as(self.roi_bin_w[poolw, proposal_id])
            with self.tik_instance.if_scope(scalar_roi_bin_w != 0):
                with self.tik_instance.for_range(0, \
                        ceil_div(self.pooled_h + self.res_pad, 8)) \
                        as loop_h:
                    self.tik_instance.vmax(
                        256//TYPELEN_DICT[self.dtype],
                        self.pooled_res[loop_h*8, poolw, 0],
                        self.pooled_h_res[loop_h*8,
                                          scalar_roi_start_w_from0, 0],
                        self.pooled_res[loop_h*8, poolw, 0],
                        scalar_roi_bin_w, self.pooled_w,
                        self.fm_w_align, self.pooled_w, 0, 1, 0)

    def proposal_pooling_w_float32(self, proposal_id):
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
        self.pooled_h_res = self.pooled_h_res.reshape((self.pooled_h +
                                                       self.res_pad,
                                                       self.fm_w_align, 2,
                                                       self.fm_c0//2))

        self.pooled_res = self.pooled_res.reshape((self.pooled_h+self.res_pad,
                                                   self.pooled_w, 2,
                                                   self.fm_c0//2))

        with self.tik_instance.for_range(0, self.pooled_w) as poolw:
            scalar_roi_start_w_from0 = self.tik_instance.Scalar("int32")
            scalar_roi_start_w_from0.set_as(
                self.roi_start_w_from0[poolw, proposal_id])

            scalar_roi_bin_w = self.tik_instance.Scalar("int32")
            scalar_roi_bin_w.set_as(self.roi_bin_w[poolw, proposal_id])
            with self.tik_instance.for_range(0,
                                             ceil_div(self.pooled_h +
                                                      self.res_pad, 8)) \
                    as loop_h:

                with self.tik_instance.for_range(0, 2) as c0_index:
                    self.tik_instance.vmax(
                        256//TYPELEN_DICT[self.dtype],
                        self.pooled_res[loop_h*8, poolw, c0_index, 0],
                        self.pooled_h_res[loop_h*8,
                                          scalar_roi_start_w_from0,
                                          c0_index, 0],
                        self.pooled_res[loop_h*8, poolw, c0_index, 0],
                        scalar_roi_bin_w, self.pooled_w*2,
                        self.fm_w_align*2, self.pooled_w*2, 0, 2, 0)

        self.pooled_res = self.pooled_res.reshape((self.pooled_h+self.res_pad, \
                                                   self.pooled_w, self.fm_c0))
        self.pooled_h_res = self.pooled_h_res.reshape(
            (self.pooled_h + self.res_pad, self.fm_w_align, self.fm_c0))

    def proposal_pooling(self, block_id, tiling_index):
        """
        roi pooling of proposal frame
        Parameters
        ----------
        block_id: aicore id
        Returns
        -------
        None
        """
        proposals_ub_batchid = self.tik_instance.Scalar("int32")

        with self.tik_instance.for_range(0, self.fm_c1) as \
                fm_c1_index:

            self.fm_c0_data = \
                self.tik_instance.Tensor(self.dtype, (self.fm_h, self.fm_w,
                                                      self.fm_c0),
                                         name="fm_c0_data",
                                         scope=tik.scope_cbuf)
            c0_burst_len = self.fm_h*self.fm_w*self.fm_c0*\
                           TYPELEN_DICT[self.dtype]//32


            self.tik_instance.data_move(self.fm_c0_data,
                                        self.x[block_id, fm_c1_index, 0, 0, 0],
                                        0, 1, c0_burst_len, 0, 0)
            if self.feature_batch == 1:
                self.calced_rois.set_as(self.proposal_num_per_tiling*\
                                        tiling_index)
            with self.tik_instance.for_range(0, self.proposal_ub_validnum,
                                             thread_num=2) as proposal_id:
                with self.tik_instance.if_scope(
                        (self.calced_rois + proposal_id) < self.range_end):
                    proposals_ub_batchid.set_as(
                        self.proposals_ub_int32[0, proposal_id])

                    with self.tik_instance.if_scope(block_id ==
                                                    proposals_ub_batchid):

                        self.pooled_h_res = self.tik_instance.Tensor(
                            self.dtype, shape=(self.pooled_h +
                                               self.res_pad,
                                               self.fm_w_align,
                                               self.fm_c0),
                            scope=tik.scope_ubuf,
                            name="pooled_h_res")
                        scalar_propoal_width = \
                            self.tik_instance.Scalar("int32")
                        scalar_propoal_width.set_as(
                            self.roi_width[proposal_id])

                        ceil_loop = 16//TYPELEN_DICT[self.dtype]
                        coeff = self.fm_c0*TYPELEN_DICT[self.dtype]//32
                        with self.tik_instance.for_range(
                                0, ceil_div(scalar_propoal_width,
                                            ceil_loop)) as loop_w:

                            self.tik_instance.vec_dup(
                                256//TYPELEN_DICT[self.dtype],
                                self.pooled_h_res[0, ceil_loop*loop_w,
                                                  0],
                                0, self.pooled_h+self.res_pad,
                                self.fm_w_align*coeff)
                        self.proposal_pooling_h(block_id,
                                                proposal_id,
                                                fm_c1_index)

                        self.pooled_res = self.tik_instance.Tensor(
                            self.dtype, shape=(self.pooled_h +
                                               self.res_pad,
                                               self.pooled_w,
                                               self.fm_c0),
                            scope=tik.scope_ubuf,
                            name="pooled_res")

                        self.tik_instance.vec_dup(
                            256//TYPELEN_DICT[self.dtype],
                            self.pooled_res[0, 0, 0], 0,
                            ((self.pooled_h + self.res_pad) * \
                                self.pooled_w) * self.fm_c0 * \
                                TYPELEN_DICT[self.dtype]//256,
                            8)
                        with self.tik_instance.if_scope(self.dtype ==
                                                        "float32"):
                            self.proposal_pooling_w_float32(proposal_id)
                        with self.tik_instance.else_scope():
                            self.proposal_pooling_w_float16(proposal_id)

                        self.tik_instance.data_move(self.y
                                                    [self.ouput_proposal_offset
                                                     + self.calced_rois
                                                     + proposal_id, fm_c1_index,
                                                     0, 0, 0],
                                                    self.pooled_res[0, 0, 0],
                                                    0, 1,
                                                    self.
                                                    pooled_h*self.pooled_w *
                                                    TYPELEN_DICT[self.dtype] *
                                                    self.fm_c0//32, 0, 0)
        if self.feature_batch != 1:
            self.calced_rois.set_as(self.calced_rois +
                                    self.proposal_ub_validnum)

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
            self.proposal_pooling_onebatch()
        else:
            self.proposal_pooling_multibatch()

        if self.roi_actual_num_effect:
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=(self.x,
                                               self.rois, self.roi_actual_num),
                                       outputs=(self.y), enable_l2=False)
        else:
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=(self.x,
                                               self.rois),
                                       outputs=(self.y), enable_l2=False)

    def proposal_pooling_multibatch(self):
        """
        multibatch pooling process
        Parameters
        -------
        None
        """
        with self.tik_instance.for_range(0,
                                         self.feature_batch,
                                         block_num=self.feature_batch) \
                as block_id:
            self.space_alloc(block_id)
            with self.tik_instance.for_range(0,
                                             self.tiling_num) as tiling_index:
                self.get_proposal_height_width(tiling_index, block_id)
                self.init_pooled_proposal_start_hw()
                self.get_pooled_proposal_start_h()
                self.get_pooled_proposal_bin_h()
                self.get_pooled_proposal_start_w()
                self.get_pooled_proposal_bin_w()
                self.proposal_pooling(block_id, tiling_index)

    def proposal_pooling_onebatch(self):
        """
        onebatch pooling process
        Parameters
        -------
        None
        """
        self.proposal_num_per_tiling = 128
        if self.roi_max_num%self.proposal_num_per_tiling == 0:
            self.tiling_num = self.roi_max_num//self.proposal_num_per_tiling
        else:
            self.tiling_num = self.roi_max_num//self.proposal_num_per_tiling \
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
            self.proposal_pooling(0, tiling_index)
