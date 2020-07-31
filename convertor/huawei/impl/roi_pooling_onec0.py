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
from te import platform as tbe_platform

from impl.roi_pooling_base import RoiClass
from impl.roi_pooling_base import TYPELEN_DICT
from impl.roi_pooling_base import INDEX_C1
from impl.roi_pooling_base import INDEX_C0

NoneType = type(None)
BLOCKNUM = 8

# pylint: disable=C0103
# pylint: disable=C0301
# pylint: disable=C0326
# pylint: disable=C0330
# pylint: disable=unused-argument,no-member
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals,too-many-lines
# pylint: disable=too-many-arguments,attribute-defined-outside-init

def align(value, factor):
    """
    make value align to factor

    Parameters
    ----------
    value:  input number
    factor:

    Returns
    -------
    res:
    """
    return (value + factor - 1) // factor*factor


def ceil_div(value, factor):
    """
    if not divide exactlly then plus 1

    Parameters
    ----------
    value:  input number
    factor: factor

    Returns
    -------
    res:
    """
    return (value + factor - 1) // factor


class RoiOneC0Class(RoiClass):
    """
    class that execute roi_pooling
    """
    def __init__(self, isOneC0PosL1):
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

        # OneC0PosL1 start
        self.isOneC0PosL1 = isOneC0PosL1
        self.propsal_num_l1_ub_tiling = None
        self.l1_roi_width = None
        self.l1_roi_start_h = None
        self.l1_roi_start_w = None
        self.l1_roi_bin_h = None
        self.l1_roi_bin_w = None
        self.l1_roi_start_w_from0 = None
        self.l1_proposals_ub_int32 = None

        self.sub_roi_width = None
        self.sub_roi_start_h = None
        self.sub_roi_start_w = None
        self.sub_roi_bin_h = None
        self.sub_roi_bin_w = None
        self.sub_roi_start_w_from0 = None
        self.sub_proposals_ub_int32 = None
        # OneC0PosL1 end

    def init_param(self, roinum_pooledimg, shapedict_list, spatial_scale_list,
                   kernel_name):
        super(RoiOneC0Class, self).init_param(roinum_pooledimg, shapedict_list,
                                              spatial_scale_list, kernel_name)
        self.res_pad = 0 if((self.pooled_h%8) == 0) else \
            (align(self.pooled_h, 8)-self.pooled_h)
        self.fm_w_align = align(self.fm_w, 8)

    def proposal_pooling_h(self, block_id, proposal_id, fm_c1_index,
                           roi_start_h, roi_start_w, roi_width, roi_bin_h):
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
        scalar_roi_start_h = self.tik_instance.Scalar("int32")
        scalar_roi_start_w = self.tik_instance.Scalar("int32")
        scalar_roi_start_w.set_as(roi_start_w[0, proposal_id])

        scalar_roi_width = self.tik_instance.Scalar("int32")
        scalar_roi_width.set_as(roi_width[proposal_id])
        scalar_roi_bin_h = self.tik_instance.Scalar("int32")
        coeff = self.fm_c0*TYPELEN_DICT[self.dtype]//32

        ceil_loop = 16//TYPELEN_DICT[self.dtype]
        scalar_loopw = self.tik_instance.Scalar("int32")
        scalar_loopw.set_as(ceil_div(
            scalar_roi_width,
            ceil_loop))
        scalar_loop = self.tik_instance.Scalar("int32")
        with self.tik_instance.for_range(0, self.pooled_h) as poolh:
            scalar_roi_start_h.set_as(roi_start_h[poolh, proposal_id])
            scalar_roi_bin_h.set_as(roi_bin_h[poolh, proposal_id])

            with self.tik_instance.for_range(0, scalar_loopw) as \
                    loop_w:
                #param proposal_fm_data
                scalar_loop.set_as(ceil_loop*loop_w)
                self.tik_instance.vec_max(
                    256//TYPELEN_DICT[self.dtype],
                    self.pooled_h_res[poolh, scalar_loop, 0],
                    self.proposal_fm_data[scalar_roi_start_h, \
                            scalar_roi_start_w + scalar_loop, 0], \
                    self.pooled_h_res[poolh, scalar_loop, 0],
                    scalar_roi_bin_h,
                    0,
                    self.fm_w_align*coeff,
                    0)

    def proposal_pooling_w_float16(self, proposal_id, roi_start_w_from0,
                                   roi_bin_w):
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
        iter_h = ceil_div(self.pooled_h + \
                          self.res_pad, 8)
        with self.tik_instance.for_range(0, self.pooled_w) as poolw:
            scalar_roi_start_w_from0.set_as(roi_start_w_from0[poolw, \
                                                              proposal_id])

            scalar_roi_bin_w.set_as(roi_bin_w[poolw, proposal_id])

            with self.tik_instance.for_range(0, iter_h) as loop_h:
                self.tik_instance.vmax(
                    256 // TYPELEN_DICT[self.dtype],
                    self.pooled_res[loop_h*8, poolw, 0],
                    self.pooled_h_res[loop_h*8,
                                      scalar_roi_start_w_from0, 0],
                    self.pooled_res[loop_h*8, poolw, 0],
                    scalar_roi_bin_w, self.pooled_w,
                    self.fm_w_align, self.pooled_w, 0, 1, 0)

    # OneC0PosL1 start
    def init_l1_rois(self):
        """
        init tensors used in l1
        Parameters
        -------
        None
        """
        self.l1_roi_width = \
            self.tik_instance.Tensor("int32",
                                     shape=(self.proposal_num_per_tiling,),
                                     scope=tik.scope_cbuf, name="l1_roi_width")

        self.l1_roi_start_h = \
            self.tik_instance.Tensor("int32",
                                     [self.pooled_h,
                                      self.proposal_num_per_tiling],
                                     name="l1_roi_start_h",
                                     scope=tik.scope_cbuf)
        self.l1_roi_start_w = \
            self.tik_instance.Tensor("int32",
                                     [self.pooled_w,
                                      self.proposal_num_per_tiling],
                                     name="l1_roi_start_w",
                                     scope=tik.scope_cbuf)

        self.l1_roi_bin_h = \
            self.tik_instance.Tensor("int32", [self.pooled_h,
                                               self.proposal_num_per_tiling],
                                     name="l1_roi_bin_h", scope=tik.scope_cbuf)
        self.l1_roi_bin_w = \
            self.tik_instance.Tensor("int32",
                                     [self.pooled_w,
                                      self.proposal_num_per_tiling],
                                     name="l1_roi_bin_w",
                                     scope=tik.scope_cbuf)
        self.l1_roi_start_w_from0 = \
            self.tik_instance.Tensor("int32",
                                     shape=(self.pooled_w,
                                            self.proposal_num_per_tiling),
                                     scope=tik.scope_cbuf,
                                     name="l1_roi_start_w_from0")
        self.l1_proposals_ub_int32 = \
            self.tik_instance.Tensor("int32",
                                     [5, self.proposal_num_per_tiling],
                                     name="l1_proposals_ub_int32",
                                     scope=tik.scope_cbuf)

    def init_sub_rois(self):
        """
        init sub rois
        Parameters
        -------
        None
        """
        self.propsal_num_l1_ub_tiling = 8
        self.sub_roi_width = \
            self.tik_instance.Tensor("int32",
                                     shape=(self.propsal_num_l1_ub_tiling,),
                                     scope=tik.scope_ubuf, name="sub_roi_width")

        self.sub_roi_start_h = \
            self.tik_instance.Tensor("int32",
                                     [self.pooled_h,
                                      self.propsal_num_l1_ub_tiling],
                                     name="sub_roi_start_h",
                                     scope=tik.scope_ubuf)
        self.sub_roi_start_w = \
            self.tik_instance.Tensor("int32",
                                     [self.pooled_w,
                                      self.propsal_num_l1_ub_tiling],
                                     name="sub_roi_start_w",
                                     scope=tik.scope_ubuf)

        self.sub_roi_bin_h = \
            self.tik_instance.Tensor("int32", [self.pooled_h,
                                               self.propsal_num_l1_ub_tiling],
                                     name="sub_roi_bin_h", scope=tik.scope_ubuf)
        self.sub_roi_bin_w = \
            self.tik_instance.Tensor("int32",
                                     [self.pooled_w,
                                      self.propsal_num_l1_ub_tiling],
                                     name="sub_roi_bin_w",
                                     scope=tik.scope_ubuf)
        self.sub_roi_start_w_from0 = \
            self.tik_instance.Tensor("int32",
                                     shape=(self.pooled_w,
                                            self.propsal_num_l1_ub_tiling),
                                     scope=tik.scope_ubuf,
                                     name="sub_roi_start_w_from0")
        self.sub_proposals_ub_int32 = \
            self.tik_instance.Tensor("int32",
                                     [5, self.propsal_num_l1_ub_tiling],
                                     name="sub_proposals_ub_int32",
                                     scope=tik.scope_ubuf)

    def mov_rois_ub_to_l1(self, roi_width, roi_start_h, roi_start_w, roi_bin_h, roi_bin_w, roi_start_w_from0, proposals_ub_int32):
        """
        move rois
        Parameters
        -------
        None
        """
        self.tik_instance.data_move(self.l1_roi_width,
                                    roi_width,
                                    0,
                                    1,
                                    self.proposal_num_per_tiling*4 // 32,
                                    0,
                                    0)

        self.tik_instance.data_move(self.l1_roi_start_h,
                                    roi_start_h,
                                    0,
                                    1,
                                    self.pooled_h*self.proposal_num_per_tiling\
                                        *4 // 32,
                                    0,
                                    0)

        self.tik_instance.data_move(self.l1_roi_start_w,
                                    roi_start_w,
                                    0,
                                    1,
                                    self.pooled_w*self.proposal_num_per_tiling\
                                        *4 // 32,
                                    0,
                                    0)

        self.tik_instance.data_move(self.l1_roi_bin_h,
                                    roi_bin_h,
                                    0,
                                    1,
                                    self.pooled_h*self.proposal_num_per_tiling\
                                        *4 // 32,
                                    0,
                                    0)

        self.tik_instance.data_move(self.l1_roi_bin_w, roi_bin_w,
                                    0,
                                    1,
                                    self.pooled_w*self.proposal_num_per_tiling\
                                        *4 // 32,
                                    0,
                                    0)

        self.tik_instance.data_move(self.l1_roi_start_w_from0,
                                    roi_start_w_from0,
                                    0,
                                    1,
                                    self.pooled_w*self.proposal_num_per_tiling\
                                        *4 // 32,
                                    0,
                                    0)

        self.tik_instance.data_move(self.l1_proposals_ub_int32,
                                    proposals_ub_int32,
                                    0,
                                    1,
                                    5*self.proposal_num_per_tiling*4 // 32,
                                    0,
                                    0)

    def mov_rois_l1_to_ub(self, outer_proposal_id):
        """
        move rois
        Parameters
        -------
        None
        """
        start_l1_proposal_id = self.propsal_num_l1_ub_tiling*outer_proposal_id
        self.tik_instance.data_move(self.sub_roi_width,
                                    self.l1_roi_width[start_l1_proposal_id],
                                    0,
                                    1,
                                    self.propsal_num_l1_ub_tiling*4 // 32,
                                    0,
                                    0)

        self.tik_instance.data_move(self.sub_roi_start_h,
                                    self.l1_roi_start_h[0,
                                                        start_l1_proposal_id],
                                    0,
                                    self.pooled_h,
                                    self.propsal_num_l1_ub_tiling*4 // 32,
                                    (self.proposal_num_per_tiling -
                                     self.propsal_num_l1_ub_tiling)*4 // 32,
                                    0)

        self.tik_instance.data_move(self.sub_roi_start_w,
                                    self.l1_roi_start_w[0,
                                                        start_l1_proposal_id],
                                    0,
                                    self.pooled_w,
                                    self.propsal_num_l1_ub_tiling*4 // 32,
                                    (self.proposal_num_per_tiling -
                                     self.propsal_num_l1_ub_tiling)*4 // 32,
                                    0)

        self.tik_instance.data_move(self.sub_roi_bin_h,
                                    self.l1_roi_bin_h[0, start_l1_proposal_id],
                                    0,
                                    self.pooled_h,
                                    self.propsal_num_l1_ub_tiling*4 // 32,
                                    (self.proposal_num_per_tiling -
                                     self.propsal_num_l1_ub_tiling)*4 // 32,
                                    0)

        self.tik_instance.data_move(self.sub_roi_bin_w,
                                    self.l1_roi_bin_w[0, start_l1_proposal_id],
                                    0,
                                    self.pooled_w,
                                    self.propsal_num_l1_ub_tiling*4 // 32,
                                    (self.proposal_num_per_tiling -
                                     self.propsal_num_l1_ub_tiling)*4 // 32,
                                    0)

        self.tik_instance.data_move(self.sub_roi_start_w_from0,
                                    self.l1_roi_start_w_from0[0, \
                                        start_l1_proposal_id],
                                    0,
                                    self.pooled_w,
                                    self.propsal_num_l1_ub_tiling*4 // 32,
                                    (self.proposal_num_per_tiling -
                                     self.propsal_num_l1_ub_tiling)*4 // 32,
                                    0)

        self.tik_instance.data_move(self.sub_proposals_ub_int32,
                                    self.l1_proposals_ub_int32[0, \
                                        start_l1_proposal_id],
                                    0,
                                    5,
                                    self.propsal_num_l1_ub_tiling*4 // 32,
                                    (self.proposal_num_per_tiling -
                                     self.propsal_num_l1_ub_tiling)*4 // 32,
                                    0)
    # OneC0PosL1 end


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
        opt_config = {
            "double_buffer_non_reuse": True,
            "out_of_bound_sync_check": True
        }
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300CS","Hi3796CV300ES"):
            if self.roi_actual_num_effect:
                self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                         inputs=(self.x,
                                                 self.rois, self.roi_actual_num),
                                         outputs=(self.y,), enable_l2=False)
            else:
                self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                         inputs=(self.x,
                                                 self.rois),
                                         outputs=(self.y,), enable_l2=False)
        else:
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
        """
        onebatch pooling process
        Parameters
        -------
        None
        """
        self.proposal_num_per_tiling = 128
        if self.roi_max_num%self.proposal_num_per_tiling == 0:
            self.tiling_num = self.roi_max_num // self.proposal_num_per_tiling
        else:
            self.tiling_num = self.roi_max_num // self.proposal_num_per_tiling\
                              + 1

        with self.tik_instance.for_range(0, self.tiling_num, \
                block_num=self.tiling_num) as tiling_index:
            if self.isOneC0PosL1 == 0:
                self.space_alloc(0)
                self.get_proposal_height_width(tiling_index, 0)
                self.init_pooled_proposal_start_hw()
                self.get_pooled_proposal_start_h()
                self.get_pooled_proposal_bin_h()
                self.get_pooled_proposal_start_w()
                self.get_pooled_proposal_bin_w()

            # OneC0PosL1 start
            if self.isOneC0PosL1 != 0:
                self.space_alloc_oneC0L1(0)
                self.init_l1_rois()
                self.init_sub_rois()

                with self.tik_instance.new_stmt_scope():
                    roi_start_h = None
                    roi_start_w = None
                    roi_bin_h = None
                    roi_bin_w = None
                    roi_start_w_from0 = None
                    proposals_ub_int32 = None
                    roi_height = None
                    roi_width = None
                    const_value = None
                    const_zero = None
                    proposals_ub_int32 = self.tik_instance.Tensor("int32",
                                         [5, self.proposal_num_per_tiling],
                                         name="proposals_ub_int32",
                                         scope=tik.scope_ubuf)
                    roi_start_h = \
                        self.tik_instance.Tensor("int32",
                                                 [self.pooled_h,
                                                  self.proposal_num_per_tiling],
                                                 name="roi_start_h",
                                                 scope=tik.scope_ubuf)
                    roi_start_w = \
                        self.tik_instance.Tensor("int32",
                                                 [self.pooled_w,
                                                  self.proposal_num_per_tiling],
                                                 name="roi_start_w",
                                                 scope=tik.scope_ubuf)

                    # all ceiling pos  roi_bin_h
                    roi_bin_h = \
                        self.tik_instance.Tensor("int32", [self.pooled_h,
                                                           self.proposal_num_per_tiling],
                                                 name="roi_bin_h", scope=tik.scope_ubuf)
                    roi_bin_w = \
                        self.tik_instance.Tensor("int32",
                                                 [self.pooled_w,
                                                  self.proposal_num_per_tiling],
                                                 name="roi_bin_w",
                                                 scope=tik.scope_ubuf)
                    roi_start_w_from0 = \
                        self.tik_instance.Tensor("int32",
                                                 shape=(self.pooled_w,
                                                        self.proposal_num_per_tiling),
                                                 scope=tik.scope_ubuf,
                                                 name="roi_start_w_from0")

                    roi_height = \
                        self.tik_instance.Tensor("int32",
                                                 shape=(self.proposal_num_per_tiling,),
                                                 scope=tik.scope_ubuf, name="roi_height")
                    roi_width = \
                        self.tik_instance.Tensor("int32",
                                                 shape=(self.proposal_num_per_tiling,),
                                                 scope=tik.scope_ubuf, name="roi_width")

                    const_value = self.tik_instance.Tensor("int32", shape=(64,),
                                                                name="const_value",
                                                                scope=tik.scope_ubuf)
                    const_zero = self.tik_instance.Tensor("int32", (64,),
                                                               name="const_zero",
                                                               scope=tik.scope_ubuf)

                    self.get_proposal_height_width_param(tiling_index, 0, proposals_ub_int32, roi_height, roi_width, const_value, const_zero)
                    self.init_pooled_proposal_start_hw_param(roi_start_h, roi_start_w, roi_bin_h, roi_bin_w, roi_height, roi_width)
                    self.get_pooled_proposal_start_h_param(roi_start_h, proposals_ub_int32, const_value, const_zero)
                    self.get_pooled_proposal_bin_h_param(roi_start_h, roi_bin_h, proposals_ub_int32, const_value, const_zero)
                    self.get_pooled_proposal_start_w_param(roi_start_w, roi_start_w_from0, proposals_ub_int32, const_value, const_zero)
                    self.get_pooled_proposal_bin_w_param(roi_start_w, roi_bin_w, proposals_ub_int32, roi_width, const_value, const_zero)
                    self.mov_rois_ub_to_l1(roi_width, roi_start_h, roi_start_w, roi_bin_h, roi_bin_w, roi_start_w_from0, proposals_ub_int32)
            # OneC0PosL1 end
            self.proposal_pooling(0, tiling_index)

    def proposal_pooling_w_float32(self, proposal_id, roi_start_w_from0,
                                   roi_bin_w):
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

        scalar_roi_start_w_from0 = self.tik_instance.Scalar("int32")
        scalar_roi_bin_w = self.tik_instance.Scalar("int32")
        iter_h = ceil_div(self.pooled_h + self.res_pad, 8)
        with self.tik_instance.for_range(0, self.pooled_w) as poolw:
            scalar_roi_start_w_from0.set_as(
                roi_start_w_from0[poolw, proposal_id])

            scalar_roi_bin_w.set_as(roi_bin_w[poolw, proposal_id])
            with self.tik_instance.for_range(0, iter_h) as loop_h:
                with self.tik_instance.for_range(0, 2) as c0_index:
                    self.tik_instance.vmax(
                        256 // TYPELEN_DICT[self.dtype],
                        self.pooled_res[loop_h*8, poolw, c0_index, 0],
                        self.pooled_h_res[loop_h*8,
                                          scalar_roi_start_w_from0,
                                          c0_index, 0],
                        self.pooled_res[loop_h*8, poolw, c0_index, 0],
                        scalar_roi_bin_w, self.pooled_w*2,
                        self.fm_w_align*2, self.pooled_w*2, 0, 2, 0)

        self.pooled_res = self.pooled_res.reshape((self.pooled_h+self.res_pad,\
                                                   self.pooled_w, self.fm_c0))
        self.pooled_h_res = self.pooled_h_res.reshape((self.pooled_h + \
                self.res_pad, self.fm_w_align, self.fm_c0))

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
                as batch_id:
            if self.isOneC0PosL1 == 0:
                self.space_alloc(batch_id)
            else:
                self.space_alloc_oneC0L1(batch_id)

            with self.tik_instance.for_range(0, self.tiling_num)\
                    as tiling_index:
                if self.isOneC0PosL1 == 0:
                    self.get_proposal_height_width(tiling_index, batch_id)
                    self.init_pooled_proposal_start_hw()
                    self.get_pooled_proposal_start_h()
                    self.get_pooled_proposal_bin_h()
                    self.get_pooled_proposal_start_w()
                    self.get_pooled_proposal_bin_w()

                # OneC0PosL1 start
                if self.isOneC0PosL1 != 0:
                    self.init_l1_rois()
                    self.init_sub_rois()

                    with self.tik_instance.new_stmt_scope():
                        roi_start_h = None
                        roi_start_w = None
                        roi_bin_h = None
                        roi_bin_w = None
                        roi_start_w_from0 = None
                        proposals_ub_int32 = None
                        roi_height = None
                        roi_width = None
                        const_value = None
                        const_zero = None
                        proposals_ub_int32 = self.tik_instance.Tensor("int32",
                                             [5, self.proposal_num_per_tiling],
                                             name="proposals_ub_int32",
                                             scope=tik.scope_ubuf)
                        roi_start_h = \
                            self.tik_instance.Tensor("int32",
                                                     [self.pooled_h,
                                                      self.proposal_num_per_tiling],
                                                     name="roi_start_h",
                                                     scope=tik.scope_ubuf)
                        roi_start_w = \
                            self.tik_instance.Tensor("int32",
                                                     [self.pooled_w,
                                                      self.proposal_num_per_tiling],
                                                     name="roi_start_w",
                                                     scope=tik.scope_ubuf)

                        # all ceiling pos  roi_bin_h
                        roi_bin_h = \
                            self.tik_instance.Tensor("int32", [self.pooled_h,
                                                               self.proposal_num_per_tiling],
                                                     name="roi_bin_h", scope=tik.scope_ubuf)
                        roi_bin_w = \
                            self.tik_instance.Tensor("int32",
                                                     [self.pooled_w,
                                                      self.proposal_num_per_tiling],
                                                     name="roi_bin_w",
                                                     scope=tik.scope_ubuf)
                        roi_start_w_from0 = \
                            self.tik_instance.Tensor("int32",
                                                     shape=(self.pooled_w,
                                                            self.proposal_num_per_tiling),
                                                     scope=tik.scope_ubuf,
                                                     name="roi_start_w_from0")

                        roi_height = \
                            self.tik_instance.Tensor("int32",
                                                     shape=(self.proposal_num_per_tiling,),
                                                     scope=tik.scope_ubuf, name="roi_height")
                        roi_width = \
                            self.tik_instance.Tensor("int32",
                                                     shape=(self.proposal_num_per_tiling,),
                                                     scope=tik.scope_ubuf, name="roi_width")

                        const_value = self.tik_instance.Tensor("int32", shape=(64,),
                                                                    name="const_value",
                                                                    scope=tik.scope_ubuf)
                        const_zero = self.tik_instance.Tensor("int32", (64,),
                                                                   name="const_zero",
                                                                   scope=tik.scope_ubuf)

                        self.get_proposal_height_width_param(tiling_index, batch_id, proposals_ub_int32, roi_height, roi_width, const_value, const_zero)
                        self.init_pooled_proposal_start_hw_param(roi_start_h, roi_start_w, roi_bin_h, roi_bin_w, roi_height, roi_width)
                        self.get_pooled_proposal_start_h_param(roi_start_h, proposals_ub_int32, const_value, const_zero)
                        self.get_pooled_proposal_bin_h_param(roi_start_h, roi_bin_h, proposals_ub_int32, const_value, const_zero)
                        self.get_pooled_proposal_start_w_param(roi_start_w, roi_start_w_from0, proposals_ub_int32, const_value, const_zero)
                        self.get_pooled_proposal_bin_w_param(roi_start_w, roi_bin_w, proposals_ub_int32, roi_width, const_value, const_zero)
                        self.mov_rois_ub_to_l1(roi_width, roi_start_h, roi_start_w, roi_bin_h, roi_bin_w, roi_start_w_from0, proposals_ub_int32)
                # OneC0PosL1 end
                self.proposal_pooling(batch_id, tiling_index)

    def do_roi_pooling(self, block_id, proposal_id, inner_proposal_id,
                       proposals_ub_int32, fm_c1_index, roi_start_h,
                       roi_start_w_from0, roi_start_w, roi_width,
                       roi_bin_h, roi_bin_w):
        """
        execute roi_pooling
        Parameters
        -------
        None
        """
        with self.tik_instance.if_scope(
                (self.calced_rois + proposal_id) < self.range_end):
            proposals_ub_batchid = self.tik_instance.Scalar("int32")
            proposals_ub_batchid.set_as(proposals_ub_int32[0,
                                                           inner_proposal_id])
            with self.tik_instance.if_scope(block_id == proposals_ub_batchid):
                self.pooled_h_res = self.tik_instance.Tensor(self.dtype, \
                        shape=(self.pooled_h+self.res_pad,
                               self.fm_w_align, self.fm_c0), \
                        scope=tik.scope_ubuf, name="pooled_h_res")

                self.tik_instance.vec_dup(
                    256 // TYPELEN_DICT[self.dtype],
                    self.pooled_h_res[0, 0, 0],
                    0, int((self.pooled_h + self.res_pad)*self.fm_w_align / 8),
                    8)

                self.pooled_res = self.tik_instance.Tensor(
                    self.dtype, shape=(self.pooled_h +
                                       self.res_pad,
                                       self.pooled_w,
                                       self.fm_c0),
                    scope=tik.scope_ubuf,
                    name="pooled_res")
                self.tik_instance.vec_dup(
                    256 // TYPELEN_DICT[self.dtype],
                    self.pooled_res[0, 0, 0], 0,
                    ((self.pooled_h+self.res_pad)*self.pooled_w)*
                    self.fm_c0*TYPELEN_DICT[self.dtype] // 256, 8)
                self.proposal_pooling_h(block_id, inner_proposal_id, \
                        fm_c1_index, roi_start_h, roi_start_w, roi_width, \
                        roi_bin_h)

                if self.dtype == "float32":
                    self.proposal_pooling_w_float32(inner_proposal_id, \
                            roi_start_w_from0, roi_bin_w)
                else:
                    self.proposal_pooling_w_float16(inner_proposal_id, \
                            roi_start_w_from0, roi_bin_w)

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

    def proposal_pooling(self, block_id, tiling_index):
        """
        execute pooling
        Parameters
        -------
        None
        """
        self.proposal_fm_data = \
            self.tik_instance.Tensor(self.dtype,
                                     (self.fm_h + 2,
                                      self.fm_w_align, self.fm_c0),
                                     name="proposal_data",
                                     scope=tik.scope_ubuf)
        coeff = self.fm_c0*TYPELEN_DICT[self.dtype] // 32
        with self.tik_instance.for_range(0, self.fm_c1) as fm_c1_index:
            self.tik_instance.data_move(self.proposal_fm_data,
                                        self.x[block_id, fm_c1_index, 0, 0, 0],
                                        0, self.fm_h, self.fm_w*coeff, 0,
                                        (self.fm_w_align-self.fm_w)*coeff)

            if self.feature_batch == 1:
                self.calced_rois.set_as(self.proposal_num_per_tiling*\
                                        tiling_index)

            if self.isOneC0PosL1 == 0:
                with self.tik_instance.for_range(0, self.proposal_ub_validnum,
                                                 thread_num=2) as proposal_id:
                    roi_start_h = self.roi_start_h
                    roi_start_w = self.roi_start_w
                    roi_start_w_from0 = self.roi_start_w_from0
                    roi_width = self.roi_width
                    roi_bin_h = self.roi_bin_h
                    roi_bin_w = self.roi_bin_w
                    proposals_ub_int32 = self.proposals_ub_int32
                    inner_proposal_id = proposal_id

                    self.do_roi_pooling(block_id, proposal_id, \
                            inner_proposal_id, proposals_ub_int32, \
                            fm_c1_index, roi_start_h, roi_start_w_from0, \
                            roi_start_w, roi_width, roi_bin_h, roi_bin_w)

            else:
                outer_proposal_nums = ceil_div(self.proposal_ub_validnum, 8)
                with self.tik_instance.for_range(0, outer_proposal_nums)\
                        as outer_proposal_id:
                    # at least sub rois first element in valid range
                    with self.tik_instance.if_scope((self.calced_rois + \
                                 outer_proposal_id * 8) < self.range_end):
                        self.mov_rois_l1_to_ub(outer_proposal_id)
                        with self.tik_instance.for_range(0, 8, thread_num=2)\
                                as inner_proposal_id:
                            proposal_id = outer_proposal_id * 8 + \
                                          inner_proposal_id

                            roi_start_h = self.sub_roi_start_h
                            roi_start_w = self.sub_roi_start_w
                            roi_start_w_from0 = self.sub_roi_start_w_from0
                            roi_width = self.sub_roi_width
                            roi_bin_h = self.sub_roi_bin_h
                            roi_bin_w = self.sub_roi_bin_w
                            proposals_ub_int32 = self.sub_proposals_ub_int32
                            self.do_roi_pooling(block_id, proposal_id, \
                                inner_proposal_id, proposals_ub_int32, \
                                fm_c1_index, roi_start_h, roi_start_w_from0, \
                                roi_start_w, roi_width, roi_bin_h, roi_bin_w)

        if self.feature_batch != 1:
            self.calced_rois.set_as(self.calced_rois +
                                    self.proposal_ub_validnum)
