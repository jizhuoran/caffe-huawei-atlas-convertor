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

roi_pooling_four_c0
"""

from te import tik
from impl.roi_pooling_base import RoiClass
from impl.roi_pooling_base import TYPELEN_DICT
from impl.roi_pooling_base import INDEX_C1
from impl.roi_pooling_base import INDEX_C0


# four C0
FOUR_C0 = 4
# one block size takes up 32b
BLOCK_SIZE = 32
# C0 is 16
C0 = 16
# digit 1
DIGIT_1 = 1
# digit 4
DIGIT_4 = 4
# digit 8
DIGIT_8 = 8
# digit 64
DIGIT_64 = 64
# digit 128
DIGIT_128 = 128
# digit 256
DIGIT_256 = 256
# data type of fp16
FP16 = "float16"
# data type of fp32
FP32 = "float32"


def align_value(value, factor):
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


# pylint: disable=too-many-instance-attributes, too-many-locals, invalid-name
# pylint: disable=unused-argument, attribute-defined-outside-init
class RoiClass4C0(RoiClass):
    """
    RoiClass4C0 class that execute roi_pooling
    """
    def __init__(self):
        """
        constructor of RoiClass4C0

        Parameters
        -------
        None
        """
        super(RoiClass4C0, self).__init__()
        self.tail_c0_num = 0
        self.fm_w_align = None
        self.dsize = 0
        self.proposal_fm_data = None
        self.c1_looptime = None
        self.tail_c0_num = None

    def proposal_pooling_w(self, proposal_id, pooled_h_i, pooled_res,
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
        scalar_roi_start_w_from0 = self.tik_instance.Scalar(
            "int32", name="scalar_roi_start_w_from0")
        scalar_roi_bin_w = self.tik_instance.Scalar(
            "int32", name="scalar_roi_bin_w")

        with self.tik_instance.for_range(0, self.pooled_w) as pooled_w_i:
            scalar_roi_start_w_from0.set_as(self.roi_start_w_from0[pooled_w_i,
                                                                   proposal_id])
            scalar_roi_bin_w.set_as(self.roi_bin_w[pooled_w_i, proposal_id])

            # calc element num is 4*16 per repeat
            self.tik_instance.vmax(
                FOUR_C0*self.fm_c0,
                pooled_res[0, pooled_h_i, pooled_w_i, 0],
                pooled_h_res[0, 0, scalar_roi_start_w_from0, 0],
                pooled_res[0, pooled_h_i, pooled_w_i, 0],
                scalar_roi_bin_w,
                self.pooled_w*self.pooled_h*C0*self.dsize // BLOCK_SIZE,
                self.fm_w_align*C0*self.dsize // BLOCK_SIZE,
                self.pooled_w*self.pooled_h*C0*self.dsize // BLOCK_SIZE,
                0, C0*self.dsize // BLOCK_SIZE, 0)

    def proposal_pooling(self, proposal_id, c1_loop_index):
        """
        max pooling from the h direction, then max pooling from the w
        direction, for fp16

        Parameters
        ----------
        proposal_id: which proposal is now being processed
        c1_loop_index: c1 index of the feature map 4C0

        Returns
        -------
        None
        """
        scalar_roi_start_w = self.tik_instance.Scalar("int32",
                                                      name="scalar_roi_start_w")
        scalar_roi_start_w.set_as(self.roi_start_w[0, proposal_id])

        scalar_roi_start_h = self.tik_instance.Scalar("int32",
                                                      name="scalar_roi_start_h")

        scalar_roi_bin_h = self.tik_instance.Scalar("int32",
                                                    name="scalar_roi_bin_h")

        scalar_roi_width = self.tik_instance.Scalar("int32",
                                                    name="scalar_roi_width")
        scalar_roi_width.set_as(self.roi_width[proposal_id])

        scalar_roi_height = self.tik_instance.Scalar("int32",
                                                     name="scalar_roi_height")
        scalar_roi_height.set_as(self.roi_height[proposal_id])

        pooled_res = self.tik_instance.Tensor(self.dtype, \
                shape=(FOUR_C0, self.pooled_h, self.pooled_w, self.fm_c0), \
                scope=tik.scope_ubuf, name="pooled_res")
        res_size = FOUR_C0*self.pooled_h*self.pooled_w*self.fm_c0

        if res_size // DIGIT_128 >= 1:
            self.tik_instance.vec_dup(DIGIT_256 // self.dsize,
                                      pooled_res[0, 0, 0, 0],
                                      0, res_size // DIGIT_128,
                                      DIGIT_8)
        if res_size % DIGIT_128 != 0:  # tail
            self.tik_instance.vec_dup(
                res_size % DIGIT_128,
                pooled_res[res_size // DIGIT_128*DIGIT_128],
                0, DIGIT_1, DIGIT_8)

        pooled_h_res = self.tik_instance.Tensor(self.dtype, \
                shape=(FOUR_C0, 1, self.fm_w_align, self.fm_c0), \
                scope=tik.scope_ubuf, name="pooled_h_res")
        pooled_h_res_size = FOUR_C0*1*self.fm_w_align*self.fm_c0

        with self.tik_instance.for_range(0, self.pooled_h) as pooled_h_i:
            scalar_roi_start_h.set_as(self.roi_start_h[pooled_h_i, proposal_id])
            scalar_roi_bin_h.set_as(self.roi_bin_h[pooled_h_i, proposal_id])

            with self.tik_instance.if_scope(tik.all(scalar_roi_bin_h != 0,
                                                    scalar_roi_width != 0)):
                self.tik_instance.vec_dup(DIGIT_256 // self.dsize,
                                          pooled_h_res[0, 0, 0, 0], 0,
                                          pooled_h_res_size // DIGIT_128,
                                          DIGIT_8)

                if self.fm_h*self.fm_w < DIGIT_256:
                    with self.tik_instance.for_range(0, scalar_roi_width)\
                            as w_index:
                        self.tik_instance.vmax(
                            FOUR_C0*self.fm_c0, pooled_h_res[0, 0, w_index, 0],
                            self.proposal_fm_data[0, scalar_roi_start_h,
                                                  scalar_roi_start_w + w_index,
                                                  0],
                            pooled_h_res[0, 0, w_index, 0],
                            scalar_roi_bin_h,
                            self.fm_w_align*C0*self.dsize // BLOCK_SIZE,
                            self.fm_h*self.fm_w*C0*self.dsize // BLOCK_SIZE,
                            self.fm_w_align*C0*self.dsize // BLOCK_SIZE,
                            0,
                            self.fm_w*C0*self.dsize // BLOCK_SIZE,
                            0)
                else:
                    with self.tik_instance.for_range(0, FOUR_C0) as c0_i:
                        with self.tik_instance.if_scope(
                                scalar_roi_width <= DIGIT_8):
                            self.tik_instance.vec_max(
                                scalar_roi_width*self.fm_c0,
                                pooled_h_res[c0_i, 0, 0, 0],
                                self.proposal_fm_data[c0_i, scalar_roi_start_h,
                                                      scalar_roi_start_w, 0],
                                pooled_h_res[c0_i, 0, 0, 0],
                                scalar_roi_bin_h,
                                0, self.fm_w*C0*self.dsize // BLOCK_SIZE, 0)
                        with self.tik_instance.else_scope():
                            with self.tik_instance.for_range(0, \
                                    scalar_roi_width // DIGIT_8) as loop_8w_i:
                                self.tik_instance.vec_max(
                                    DIGIT_256 // self.dsize,
                                    pooled_h_res[c0_i, 0, DIGIT_8*loop_8w_i, 0],
                                    self.proposal_fm_data[
                                        c0_i, scalar_roi_start_h,
                                        scalar_roi_start_w + DIGIT_8*loop_8w_i,
                                        0],
                                    pooled_h_res[c0_i, 0, DIGIT_8*loop_8w_i, 0],
                                    scalar_roi_bin_h,
                                    0, self.fm_w*C0*self.dsize // BLOCK_SIZE, 0)
                            with self.tik_instance.if_scope(
                                    scalar_roi_width % DIGIT_8 != 0):
                                tmp_w = scalar_roi_width // DIGIT_8*DIGIT_8
                                self.tik_instance.vec_max(
                                    (scalar_roi_width - tmp_w)*self.fm_c0,
                                    pooled_h_res[c0_i, 0, tmp_w, 0],
                                    self.proposal_fm_data[c0_i, \
                                            scalar_roi_start_h, \
                                            scalar_roi_start_w + tmp_w, 0],
                                    pooled_h_res[c0_i, 0, tmp_w, 0],
                                    scalar_roi_bin_h,
                                    0, self.fm_w*C0*self.dsize // BLOCK_SIZE, 0)

                self.proposal_pooling_w(proposal_id, pooled_h_i, pooled_res,
                                        pooled_h_res)

        # move result to out
        with self.tik_instance.if_scope(c1_loop_index != self.c1_looptime - 1):
            self.tik_instance.data_move(
                self.y[self.ouput_proposal_offset + self.calced_rois
                       + proposal_id,
                       c1_loop_index*FOUR_C0, 0, 0, 0],
                pooled_res[0, 0, 0, 0],
                0,
                1,
                FOUR_C0*self.pooled_h*self.pooled_w*C0*self.dsize // BLOCK_SIZE,
                0,
                0)
        with self.tik_instance.else_scope():  # tail
            self.tik_instance.data_move(
                self.y[self.ouput_proposal_offset+self.calced_rois+proposal_id,
                       (self.c1_looptime - 1)*FOUR_C0, 0, 0, 0],
                pooled_res[0, 0, 0, 0],
                0,
                1,
                self.tail_c0_num*self.pooled_h*self.pooled_w\
                    *C0*self.dsize // BLOCK_SIZE,
                0,
                0)

    def proposal_pooling_fp32(self, proposal_id, c1_loop_index):
        """
        max pooling from the h direction, then max pooling from the w
        direction, for fp32

        Parameters
        ----------
        proposal_id: which proposal is now being processed
        c1_loop_index: c1 index of the feature map 4C0

        Returns
        -------
        None
        """
        scalar_roi_start_w = self.tik_instance.Scalar("int32", \
                name="scalar_roi_start_w")
        scalar_roi_start_w.set_as(self.roi_start_w[0, proposal_id])

        scalar_roi_start_h = self.tik_instance.Scalar("int32", \
                name="scalar_roi_start_h")

        scalar_roi_bin_h = self.tik_instance.Scalar("int32",
                                                    name="scalar_roi_bin_h")

        scalar_roi_width = self.tik_instance.Scalar("int32",
                                                    name="scalar_roi_width")
        scalar_roi_width.set_as(self.roi_width[proposal_id])

        scalar_roi_height = self.tik_instance.Scalar("int32",
                                                     name="scalar_roi_height")
        scalar_roi_height.set_as(self.roi_height[proposal_id])

        pooled_res = self.tik_instance.Tensor(FP32, \
              shape=(FOUR_C0, self.pooled_h, self.pooled_w, self.fm_c0), \
              scope=tik.scope_ubuf, name="pooled_res")
        res_size = FOUR_C0*self.pooled_h*self.pooled_w*self.fm_c0

        if res_size // DIGIT_64 >= 1:
            self.tik_instance.vec_dup(DIGIT_256 // self.dsize,
                                      pooled_res[0, 0, 0, 0],
                                      0, res_size // DIGIT_64,
                                      DIGIT_8)
        if res_size % DIGIT_64 != 0:  # tail
            self.tik_instance.vec_dup(
                res_size % DIGIT_64,
                pooled_res[res_size // DIGIT_64*DIGIT_64],
                0, DIGIT_1, DIGIT_8)

        pooled_h_res = self.tik_instance.Tensor(FP32, \
                shape=(FOUR_C0, 1, self.fm_w_align, self.fm_c0), \
                scope=tik.scope_ubuf, name="pooled_h_res")
        pooled_h_res_size = FOUR_C0*1*self.fm_w_align*self.fm_c0

        with self.tik_instance.for_range(0, self.pooled_h) as pooled_h_i:
            scalar_roi_start_h.set_as(self.roi_start_h[pooled_h_i, proposal_id])
            scalar_roi_bin_h.set_as(self.roi_bin_h[pooled_h_i, proposal_id])

            with self.tik_instance.if_scope(tik.all(scalar_roi_bin_h != 0,
                                                    scalar_roi_width != 0)):
                self.tik_instance.vec_dup(DIGIT_256 // self.dsize,
                                          pooled_h_res[0, 0, 0, 0], 0,
                                          pooled_h_res_size // DIGIT_64,
                                          DIGIT_8)

                if self.fm_h*self.fm_w < DIGIT_128:
                    with self.tik_instance.for_range(0, scalar_roi_width) \
                            as w_index:
                        self.tik_instance.vmax(
                            FOUR_C0*self.fm_c0, pooled_h_res[0, 0, w_index, 0],
                            self.proposal_fm_data[0, scalar_roi_start_h,
                                                  scalar_roi_start_w + w_index,
                                                  0],
                            pooled_h_res[0, 0, w_index, 0],
                            scalar_roi_bin_h,
                            self.fm_w_align*C0*self.dsize // BLOCK_SIZE,
                            self.fm_h*self.fm_w*C0*self.dsize // BLOCK_SIZE,
                            self.fm_w_align*C0*self.dsize // BLOCK_SIZE,
                            0,
                            self.fm_w*C0*self.dsize // BLOCK_SIZE,
                            0)
                else:
                    with self.tik_instance.for_range(0, FOUR_C0) as c0_i:
                        with self.tik_instance.if_scope(
                                scalar_roi_width <= DIGIT_4):
                            self.tik_instance.vec_max(
                                scalar_roi_width*self.fm_c0,
                                pooled_h_res[c0_i, 0, 0, 0],
                                self.proposal_fm_data[c0_i, scalar_roi_start_h,
                                                      scalar_roi_start_w, 0],
                                pooled_h_res[c0_i, 0, 0, 0],
                                scalar_roi_bin_h,
                                0, self.fm_w*C0*self.dsize // BLOCK_SIZE, 0)
                        with self.tik_instance.else_scope():
                            with self.tik_instance.for_range(0, \
                                    scalar_roi_width // DIGIT_4) as loop_4w_i:
                                self.tik_instance.vec_max(
                                    DIGIT_256 // self.dsize,
                                    pooled_h_res[c0_i, 0, DIGIT_4*loop_4w_i, 0],
                                    self.proposal_fm_data[
                                        c0_i, scalar_roi_start_h,
                                        scalar_roi_start_w + DIGIT_4*loop_4w_i,
                                        0],
                                    pooled_h_res[c0_i, 0, DIGIT_4*loop_4w_i, 0],
                                    scalar_roi_bin_h,
                                    0, self.fm_w*C0*self.dsize // BLOCK_SIZE, 0)
                            with self.tik_instance.if_scope(
                                    scalar_roi_width % DIGIT_4 != 0):
                                tmp_w = scalar_roi_width // DIGIT_4*DIGIT_4
                                self.tik_instance.vec_max(
                                    (scalar_roi_width - tmp_w)*self.fm_c0,
                                    pooled_h_res[c0_i, 0, tmp_w, 0],
                                    self.proposal_fm_data[c0_i, \
                                          scalar_roi_start_h, \
                                          scalar_roi_start_w + tmp_w, 0],
                                    pooled_h_res[c0_i, 0, tmp_w, 0],
                                    scalar_roi_bin_h,
                                    0, self.fm_w*C0*self.dsize // BLOCK_SIZE, 0)

                self.proposal_pooling_w(proposal_id, pooled_h_i, pooled_res,
                                        pooled_h_res)

        # move result to out
        with self.tik_instance.if_scope(c1_loop_index != self.c1_looptime - 1):
            self.tik_instance.data_move(
                self.y[self.ouput_proposal_offset + self.calced_rois
                       + proposal_id,
                       c1_loop_index*FOUR_C0, 0, 0, 0],
                pooled_res[0, 0, 0, 0],
                0,
                1,
                FOUR_C0*self.pooled_h*self.pooled_w*C0*self.dsize // BLOCK_SIZE,
                0,
                0)
        with self.tik_instance.else_scope():  # tail
            self.tik_instance.data_move(
                self.y[self.ouput_proposal_offset+self.calced_rois+proposal_id,
                       (self.c1_looptime - 1)*FOUR_C0, 0, 0, 0],
                pooled_res[0, 0, 0, 0],
                0,
                1,
                self.tail_c0_num*self.pooled_h*self.pooled_w*\
                    C0*self.dsize // BLOCK_SIZE,
                0,
                0)

    def proposal_pooling_multibatch_impl(self, batch_id):
        """
        calculate max pooling of multi-batch fm

        Parameters
        ----------
        batch_id: batch id

        Returns
        -------
        None
        """
        self.space_alloc(batch_id)

        if self.fm_c1 % FOUR_C0 != 0:
            self.c1_looptime = self.fm_c1 // FOUR_C0 + 1
            self.tail_c0_num = self.fm_c1 - (self.c1_looptime - 1)*FOUR_C0
        else:
            self.c1_looptime = self.fm_c1 // FOUR_C0
            self.tail_c0_num = FOUR_C0

        burst_len = FOUR_C0*self.fm_h*self.fm_w*C0*self.dsize // BLOCK_SIZE
        burst_len_l = self.tail_c0_num*self.fm_h*self.fm_w*C0\
                      *self.dsize // BLOCK_SIZE

        # rois loop, loop step is 128 roi
        with self.tik_instance.for_range(0, self.tiling_num) as tiling_index:
            self.get_proposal_height_width(tiling_index, batch_id)
            self.init_pooled_proposal_start_hw()
            self.get_pooled_proposal_start_h()
            self.get_pooled_proposal_bin_h()
            self.get_pooled_proposal_start_w()
            self.get_pooled_proposal_bin_w()

            self.proposal_fm_data = self.tik_instance.Tensor(self.dtype, \
                    (FOUR_C0, self.fm_h, self.fm_w, self.fm_c0), \
                    name="proposal_fm_data", scope=tik.scope_ubuf)

            with self.tik_instance.for_range(0, self.c1_looptime) as c1_loop_i:
                # move 4 C0 fm to ub
                with self.tik_instance.if_scope(
                        c1_loop_i != self.c1_looptime - 1):
                    self.tik_instance.data_move(
                        self.proposal_fm_data,
                        self.x[batch_id, c1_loop_i*FOUR_C0, 0, 0, 0],
                        0, 1, burst_len, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        self.proposal_fm_data,
                        self.x[batch_id, (self.c1_looptime-1)*FOUR_C0, 0, 0, 0],
                        0, 1, burst_len_l, 0, 0)

                with self.tik_instance.for_range(0, self.proposal_ub_validnum,
                                                 thread_num=2) as proposal_id:
                    with self.tik_instance.if_scope(
                            (self.calced_rois + proposal_id) < self.range_end):
                        proposals_ub_batchid = self.tik_instance.Scalar(
                            "int32", name="proposals_ub_batchid")
                        proposals_ub_batchid.set_as(
                            self.proposals_ub_int32[0, proposal_id])

                        with self.tik_instance.if_scope(
                                batch_id == proposals_ub_batchid):
                            # max pooling
                            if self.dtype == FP16:
                                self.proposal_pooling(proposal_id, c1_loop_i)
                            else:
                                self.proposal_pooling_fp32(proposal_id,
                                                           c1_loop_i)

            self.calced_rois.set_as(self.calced_rois + 
                                    self.proposal_ub_validnum)

    def proposal_pooling_multibatch(self):
        """
        calculate max pooling of multi-batch fm

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.device_core_num, \
                 block_num=self.device_core_num) as block_id:
            # process of one aicore
            with self.tik_instance.for_range(0, self.batch_factor) \
                    as factor_index:
                batch_id = block_id*self.batch_factor + factor_index
                self.proposal_pooling_multibatch_impl(batch_id)

            with self.tik_instance.if_scope(block_id < self.batch_factor_tail):
                batch_id = self.batch_factor*self.device_core_num + block_id
                self.proposal_pooling_multibatch_impl(batch_id)

    def proposal_pooling_onebatch(self):
        """
        calculate max pooling of one batch fm

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.proposal_num_per_tiling = DIGIT_128
        if self.roi_max_num % self.proposal_num_per_tiling == 0:
            self.tiling_num = self.roi_max_num // self.proposal_num_per_tiling
        else:
            self.tiling_num = self.roi_max_num // self.proposal_num_per_tiling\
                              + 1

        if self.fm_c1 % FOUR_C0 != 0:
            self.c1_looptime = self.fm_c1 // FOUR_C0 + 1
            self.tail_c0_num = self.fm_c1 - (self.c1_looptime - 1)*FOUR_C0
        else:
            self.c1_looptime = self.fm_c1 // FOUR_C0
            self.tail_c0_num = FOUR_C0

        burst_len = FOUR_C0*self.fm_h*self.fm_w*C0*self.dsize // BLOCK_SIZE
        burst_len_l = self.tail_c0_num*self.fm_h*self.fm_w*C0\
                      *self.dsize // BLOCK_SIZE

        with self.tik_instance.for_range(0, self.tiling_num, \
                block_num=self.tiling_num) as tiling_index:
            self.space_alloc(0)
            self.get_proposal_height_width(tiling_index, 0)
            self.init_pooled_proposal_start_hw()
            self.get_pooled_proposal_start_h()
            self.get_pooled_proposal_bin_h()
            self.get_pooled_proposal_start_w()
            self.get_pooled_proposal_bin_w()

            self.proposal_fm_data = self.tik_instance.Tensor(self.dtype, \
                    (FOUR_C0, self.fm_h, self.fm_w, self.fm_c0), \
                    name="proposal_fm_data", scope=tik.scope_ubuf)

            with self.tik_instance.for_range(0, self.c1_looptime) as c1_loop_i:
                # move 4 C0 fm to ub
                with self.tik_instance.if_scope(
                        c1_loop_i != self.c1_looptime - 1):
                    self.tik_instance.data_move(
                        self.proposal_fm_data,
                        self.x[0, c1_loop_i*FOUR_C0, 0, 0, 0],
                        0, 1, burst_len, 0, 0)
                # tail
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        self.proposal_fm_data,
                        self.x[0, (self.c1_looptime - 1)*FOUR_C0, 0, 0, 0],
                        0, 1, burst_len_l, 0, 0)

                self.calced_rois.set_as(self.proposal_num_per_tiling*\
                                        tiling_index)

                with self.tik_instance.for_range(0, self.proposal_ub_validnum,
                                                 thread_num=2) as proposal_id:
                    if self.dtype == FP16:
                        self.proposal_pooling(proposal_id, c1_loop_i)
                    else:
                        self.proposal_pooling_fp32(proposal_id, c1_loop_i)

    def roi_pooling_main(self):
        """
        main process of roi pooling.
         including calculate the coordinate of pooled
        edge and max pooling from the h direction
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
            self.roi_actual_num = self.tik_instance.Tensor(
                dtype="int32", shape=(self.feature_batch, DIGIT_8),
                name="roi_actual_num", scope=tik.scope_gm)

        self.rois = self.tik_instance.Tensor(
            self.dtype, shape=(self.feature_batch, 5, self.roi_max_num),
            name="rois", scope=tik.scope_gm)
        self.y = self.tik_instance.Tensor(
            self.dtype,
            shape=(self.feature_batch*self.roi_max_num,
                   self.shape[INDEX_C1],
                   self.pooled_h, self.pooled_w,
                   self.shape[INDEX_C0]),
            name="y", scope=tik.scope_gm)

        if self.dtype == FP16:
            align_factor = DIGIT_8
        else:
            align_factor = DIGIT_4
        self.fm_w_align = align_value(self.fm_w, align_factor)
        self.dsize = TYPELEN_DICT[self.dtype]

        if self.feature_batch == 1:
            self.batch_factor = self.roi_max_num // self.device_core_num
            self.batch_factor_tail = self.roi_max_num - \
                                     self.batch_factor*self.device_core_num

            self.proposal_pooling_onebatch()
        else:
            self.batch_factor = self.feature_batch // self.device_core_num
            self.batch_factor_tail = self.feature_batch - \
                                     self.batch_factor*self.device_core_num

            self.proposal_pooling_multibatch()

        if self.roi_actual_num_effect:
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=(self.x,
                                               self.rois, self.roi_actual_num),
                                       outputs=(self.y,), enable_l2=False)
        else:
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=(self.x, self.rois),
                                       outputs=(self.y,), enable_l2=False)
