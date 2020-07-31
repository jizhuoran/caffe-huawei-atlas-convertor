#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

decoded_bbox
"""

# pylint: disable=C0103
# pylint: disable=R0902
# pylint: disable=R0914
# pylint: disable=R0913

from te import tik

class DecodeBbox:
    """
    decode_bbox
    """

    def __init__(self, tik_instance, input_data):
        self.tik_instance = tik_instance
        N = input_data[0]
        self.N = input_data[0]
        input_dtype = input_data[1]
        self.input_dtype = input_data[1]
        self.batch_id = input_data[2]
        self.min_box_size = input_data[3]
        self.input_shape = input_data[4]

        self.num_anchor = self.input_shape[1]//4
        self.H = self.input_shape[2]
        self.W = self.input_shape[3]

        self.output_region_proposal_ub = \
            tik_instance.Tensor(input_dtype, (N*8*16, 8),
                                name="output_region_proposal_ub",
                                scope=tik.scope_ubuf)

        if self.input_dtype == "float16":
            self.size = 2
            self.mask = 128
            self.ratio = 1
        elif self.input_dtype == "float32":
            self.size = 4
            self.mask = 64
            self.ratio = 2

        tik_instance.vector_dup(self.mask, self.output_region_proposal_ub, 0,
                                N*8*self.ratio, 1, 8)

        self.input_score_ub = tik_instance.Tensor(input_dtype, (1, N*8*16),
                                                  name="input_score_ub",
                                                  scope=tik.scope_ubuf)
        self.input_bbox_ub = tik_instance.Tensor(input_dtype, (4, N*8*16),
                                                 name="input_bbox_ub",
                                                 scope=tik.scope_ubuf)
        self.input_region_proposal_ub = \
            tik_instance.Tensor(input_dtype, (4, N*8*16),
                                name="input_region_proposal_ub",
                                scope=tik.scope_ubuf)

        self.im_info_ub = \
            tik_instance.Tensor(input_dtype, (1, 16//self.ratio),
                                name="im_info_ub",
                                scope=tik.scope_ubuf)

        self.pred_box_left_up_x = tik_instance.Tensor(input_dtype, (1, N*8*16),
                                                      name="pred_box_left_up_x",
                                                      scope=tik.scope_ubuf)
        self.pred_box_left_up_y = tik_instance.Tensor(input_dtype, (1, N*8*16),
                                                      name="pred_box_left_up_y",
                                                      scope=tik.scope_ubuf)
        self.pred_box_right_down_x = \
            tik_instance.Tensor(input_dtype, (1, N*8*16),
                                name="pred_box_right_down_x",
                                scope=tik.scope_ubuf)
        self.pred_box_right_down_y = \
            tik_instance.Tensor(input_dtype, (1, N*8*16),
                                name="pred_box_right_down_y",
                                scope=tik.scope_ubuf)

    def generate_pred_box_left_up(self, input_list, pred_box_left_up):
        """
        :param input_list:
        :param pred_box_left_up:
        :return:
        """
        pred = input_list[0]
        pred_center = input_list[1]
        image_size = input_list[2]
        a_buffer_ub = input_list[3]
        b_buffer_ub = input_list[4]
        c_buffer_ub = input_list[5]

        # 0.5*pred_w
        self.tik_instance.vmuls(self.mask, a_buffer_ub[0], pred[0], 0.5,
                                (self.N)*self.ratio, 1, 1, 8, 8)
        # pred_ctr_x - 0.5 * pred_w
        self.tik_instance.vsub(self.mask, b_buffer_ub[0], pred_center[0],
                               a_buffer_ub[0], (self.N)*self.ratio,
                               1, 1, 1, 8, 8, 8)
        # y : min(pred_ctr_x - 0.5 * pred_w, img_width - 1)
        self.tik_instance.vector_dup(self.mask, c_buffer_ub, image_size,
                                     (self.N)*self.ratio, 1, 8)
        self.tik_instance.vmin(self.mask, a_buffer_ub[0], b_buffer_ub[0],
                               c_buffer_ub[0], (self.N)*self.ratio,
                               1, 1, 1, 8, 8, 8)
        # pred_box_left_up_x : max(min(pred_ctr_x - 0.5 * pred_w, img_width - 1), 0)
        self.tik_instance.vrelu(self.mask, pred_box_left_up[0], a_buffer_ub[0],
                                (self.N)*self.ratio, 1, 1, 8, 8)

    def generate_pred_box_right_down(self, input_list, pred_box_right_down):
        """
        :param input_list:
        :param pred_box_right_down:
        :return:
        """
        pred = input_list[0]
        pred_center = input_list[1]
        image_size = input_list[2]
        a_buffer_ub = input_list[3]
        b_buffer_ub = input_list[4]
        c_buffer_ub = input_list[5]

        tik_instance = self.tik_instance

        # 0.5*pred_w
        tik_instance.vmuls(self.mask, a_buffer_ub[0], pred[0], 0.5,
                           (self.N)*self.ratio, 1, 1, 8, 8)
        # pred_ctr_x + 0.5 * pred_w
        tik_instance.vadd(self.mask, b_buffer_ub[0], pred_center[0],
                          a_buffer_ub[0], (self.N)*self.ratio, 1, 1, 1, 8, 8, 8)
        # y : min(pred_ctr_x + 0.5 * pred_w, img_width - 1)
        tik_instance.vector_dup(self.mask, c_buffer_ub, image_size,
                                (self.N)*self.ratio, 1, 8)
        tik_instance.vmin(self.mask, a_buffer_ub[0], b_buffer_ub[0],
                          c_buffer_ub, (self.N)*self.ratio, 1, 1, 1, 8, 8, 8)

        # pred_box_right_down_x : max(min(pred_ctr_x + 0.5 * pred_w, img_width - 1), 0)
        tik_instance.vrelu(self.mask, pred_box_right_down[0], a_buffer_ub[0],
                           (self.N)*self.ratio, 1, 1, 8, 8)

    def updata_bbox_position(self, a_buffer_ub, b_buffer_ub, c_buffer_ub):
        """
        :param a_buffer_ub:
        :param b_buffer_ub:
        :param c_buffer_ub:
        :return:
        """
        tik_instance = self.tik_instance
        input_dtype = self.input_dtype
        input_region_proposal_ub = self.input_region_proposal_ub
        input_bbox_ub = self.input_bbox_ub
        N = self.N

        with self.tik_instance.if_scope(True):
            widths = tik_instance.Tensor(input_dtype, (1, N*8*16),
                                         name="widths", scope=tik.scope_ubuf)
            heights = tik_instance.Tensor(input_dtype, (1, N*8*16),
                                          name="heights", scope=tik.scope_ubuf)
            ctr_x = tik_instance.Tensor(input_dtype, (1, N*8*16),
                                        name="ctr_x", scope=tik.scope_ubuf)
            ctr_y = tik_instance.Tensor(input_dtype, (1, N*8*16),
                                        name="ctr_y", scope=tik.scope_ubuf)

            pred_center_x = tik_instance.Tensor(input_dtype, (1, N*8*16),
                                                name="pred_center_x",
                                                scope=tik.scope_ubuf)
            pred_center_y = tik_instance.Tensor(input_dtype, (1, N*8*16),
                                                name="pred_center_y",
                                                scope=tik.scope_ubuf)
            pred_w = tik_instance.Tensor(input_dtype, (1, N*8*16),
                                         name="pred_w", scope=tik.scope_ubuf)
            pred_h = tik_instance.Tensor(input_dtype, (1, N*8*16),
                                         name="pred_h", scope=tik.scope_ubuf)

            # widths : boxes[:, 2] - boxes[:, 0] + 1.0
            tik_instance.vsub(self.mask, a_buffer_ub[0],
                              input_region_proposal_ub[2, 0],
                              input_region_proposal_ub[0, 0],
                              N*self.ratio, 1, 1, 1, 8, 8, 8)
            tik_instance.vadds(self.mask, widths[0], a_buffer_ub[0],
                               1, N*self.ratio, 1, 1, 8, 8)

            # heights : boxes[:, 3] - boxes[:, 1] + 1.0
            tik_instance.vsub(self.mask, a_buffer_ub[0],
                              input_region_proposal_ub[3, 0],
                              input_region_proposal_ub[1, 0],
                              N*self.ratio, 1, 1, 1, 8, 8, 8)
            tik_instance.vadds(self.mask, heights[0], a_buffer_ub[0], 1,
                               N*self.ratio, 1, 1, 8, 8)

            # ctr_x : boxes[:, 0] + 0.5 * widths
            tik_instance.vmuls(self.mask, a_buffer_ub[0], widths[0], 0.5,
                               N*self.ratio, 1, 1, 8, 8)
            tik_instance.vadd(self.mask, ctr_x[0],
                              input_region_proposal_ub[0, 0], a_buffer_ub[0],
                              N*self.ratio, 1, 1, 1, 8, 8, 8)

            # ctr_y : boxes[:, 1] + 0.5 * heights
            tik_instance.vmuls(self.mask, a_buffer_ub[0], heights[0], 0.5,
                               N*self.ratio, 1, 1, 8, 8)
            tik_instance.vadd(self.mask, ctr_y[0],
                              input_region_proposal_ub[1, 0], a_buffer_ub[0],
                              N*self.ratio, 1, 1, 1, 8, 8, 8)

            # pred_center_x : ctr_x + widths*dx
            tik_instance.vmul(self.mask, a_buffer_ub[0], widths,
                              input_bbox_ub[0, 0],
                              N*self.ratio, 1, 1, 1, 8, 8, 8)
            tik_instance.vadd(self.mask, pred_center_x, ctr_x, a_buffer_ub,
                              N*self.ratio, 1, 1, 1, 8, 8, 8)

            # pred_center_y : ctr_y + heights * dy
            tik_instance.vmul(self.mask, a_buffer_ub[0], heights,
                              input_bbox_ub[1, 0],
                              N*self.ratio, 1, 1, 1, 8, 8, 8)
            tik_instance.vadd(self.mask, pred_center_y, ctr_y, a_buffer_ub,
                              N*self.ratio, 1, 1, 1, 8, 8, 8)

            # exp(dw)
            tik_instance.vexp(self.mask, a_buffer_ub[0], input_bbox_ub[2, 0],
                              N*self.ratio, 1, 1, 8, 8)
            # pred_w : widths * exp(dw)
            tik_instance.vmul(self.mask, pred_w[0], widths, a_buffer_ub[0],
                              N*self.ratio, 1, 1, 1, 8, 8, 8)

            # exp(dh)
            tik_instance.vexp(self.mask, a_buffer_ub[0], input_bbox_ub[3, 0],
                              N*self.ratio, 1, 1, 8, 8)
            # pred_h : heights * exp(dh)
            tik_instance.vmul(self.mask, pred_h[0], heights, a_buffer_ub[0],
                              N*self.ratio, 1, 1, 1, 8, 8, 8)

            tik_instance.vadds(16 // self.ratio, self.im_info_ub, self.im_info_ub,
                               -1, 1, 1, 1, 8, 8)
            image_info = tik_instance.Scalar(dtype=input_dtype)
            image_info.set_as(self.im_info_ub[0, 1])
            self.generate_pred_box_left_up((pred_w,
                                            pred_center_x,
                                            image_info,
                                            a_buffer_ub,
                                            b_buffer_ub,
                                            c_buffer_ub),
                                           self.pred_box_left_up_x)

            image_info.set_as(self.im_info_ub[0, 0])
            self.generate_pred_box_left_up((pred_h,
                                            pred_center_y,
                                            image_info,
                                            a_buffer_ub,
                                            b_buffer_ub,
                                            c_buffer_ub),
                                           self.pred_box_left_up_y)

            image_info.set_as(self.im_info_ub[0, 1])
            self.generate_pred_box_right_down((pred_w,
                                               pred_center_x,
                                               image_info,
                                               a_buffer_ub,
                                               b_buffer_ub,
                                               c_buffer_ub),
                                              self.pred_box_right_down_x)

            image_info.set_as(self.im_info_ub[0, 0])
            self.generate_pred_box_right_down((pred_h,
                                               pred_center_y,
                                               image_info,
                                               a_buffer_ub,
                                               b_buffer_ub,
                                               c_buffer_ub),
                                              self.pred_box_right_down_y)

    def decode_bbox_sel(self, N, result_tensor, matrix):
        """
        :param N:
        :param result_tensor:
        :param matrix:
        :return:
        """
        tik_instance = self.tik_instance
        with tik_instance.if_scope(True):
            zeros_tensor = tik_instance.Tensor(self.input_dtype, (1, N*8*16),
                                               name="zeros_tensor",
                                               scope=tik.scope_ubuf)
            ones_tensor = tik_instance.Tensor(self.input_dtype, (1, N*8*16),
                                              name="ones_tensor",
                                              scope=tik.scope_ubuf)

            tik_instance.vector_dup(self.mask, zeros_tensor, 0,
                                    N*self.ratio, 1, 8)
            tik_instance.vector_dup(self.mask, ones_tensor, 1,
                                    N*self.ratio, 1, 8)

            if self.input_dtype == "float16":
                with tik_instance.for_range(0, N) as i:
                    dst_cmp_mask = \
                        self.tik_instance.mov_tensor_to_cmpmask(matrix[i*8])
                    tik_instance.vsel(128, 0, result_tensor[0, i*8*16],
                                      dst_cmp_mask,
                                      ones_tensor, zeros_tensor,
                                      1, 1, 1, 1, 8, 8, 8)
            else:
                tik_instance.vsel(64, 2, result_tensor[0, 0], matrix[0],
                                  ones_tensor, zeros_tensor,
                                  2*N, 1, 1, 1, 8, 8, 8)

    def get_decode_bbox(self):
        """
        :return:
        """
        tik_instance = self.tik_instance
        N = self.N
        mask = self.mask

        max_length = (N*8*16)//16

        a_buffer_ub = tik_instance.Tensor(self.input_dtype, (1, N*8*16),
                                          name="a_buffer_ub",
                                          scope=tik.scope_ubuf)
        b_buffer_ub = tik_instance.Tensor(self.input_dtype, (1, N*8*16),
                                          name="b_buffer_ub",
                                          scope=tik.scope_ubuf)
        c_buffer_ub = tik_instance.Tensor("uint16", [max_length],
                                          name="c_buffer_ub",
                                          scope=tik.scope_ubuf)
        d_buffer_ub = tik_instance.Tensor(self.input_dtype, (1, N*8*16),
                                          name="d_buffer_ub",
                                          scope=tik.scope_ubuf)

        self.updata_bbox_position(a_buffer_ub, b_buffer_ub, d_buffer_ub)

        result_x_tensor = tik_instance.Tensor(self.input_dtype, (1, N*8*16),
                                              name="result_x_tensor",
                                              scope=tik.scope_ubuf)
        result_y_tensor = tik_instance.Tensor(self.input_dtype, (1, N*8*16),
                                              name="result_y_tensor",
                                              scope=tik.scope_ubuf)
        score_bitmap = tik_instance.Tensor(self.input_dtype, (1, N*8*16),
                                           name="score_bitmap",
                                           scope=tik.scope_ubuf)

        # x2_cliped-x1_cliped,min_w-1----> 0/1 tensor
        tik_instance.vector_dup(mask, b_buffer_ub[0],
                                self.min_box_size[1] - 1,
                                N*self.ratio, 1, 8)
        tik_instance.vsub(self.mask, a_buffer_ub[0],
                          self.pred_box_right_down_x[0],
                          self.pred_box_left_up_x[0],
                          N*self.ratio, 1, 1, 1, 8, 8, 8)
        tik_instance.vcmpv_ge(c_buffer_ub, a_buffer_ub, b_buffer_ub,
                              N*self.ratio, 1, 1, 8, 8)
        self.decode_bbox_sel(N, result_x_tensor, c_buffer_ub)

        # y2_cliped-y1_cliped,min_h-1----> 0/1 tensor
        tik_instance.vector_dup(mask, b_buffer_ub[0], self.min_box_size[0] - 1,
                                N*self.ratio, 1, 8)
        tik_instance.vsub(mask, a_buffer_ub[0], self.pred_box_right_down_y[0],
                          self.pred_box_left_up_y[0],
                          N*self.ratio, 1, 1, 1, 8, 8, 8)
        tik_instance.vcmpv_ge(c_buffer_ub, a_buffer_ub, b_buffer_ub,
                              N*self.ratio, 1, 1, 8, 8)
        self.decode_bbox_sel(N, result_y_tensor, c_buffer_ub)

        tik_instance.vmul(mask, a_buffer_ub, result_x_tensor, result_y_tensor,
                          N*self.ratio, 1, 1, 1, 8, 8, 8)

        tik_instance.vmul(mask, score_bitmap, a_buffer_ub, self.input_score_ub,
                          N*self.ratio, 1, 1, 1, 8, 8, 8)

        tik_instance.vconcat(self.output_region_proposal_ub,
                             self.pred_box_left_up_x, N*8, 0)
        tik_instance.vconcat(self.output_region_proposal_ub,
                             self.pred_box_left_up_y, N*8, 1)
        tik_instance.vconcat(self.output_region_proposal_ub,
                             self.pred_box_right_down_x, N*8, 2)
        tik_instance.vconcat(self.output_region_proposal_ub,
                             self.pred_box_right_down_y, N*8, 3)
        tik_instance.vconcat(self.output_region_proposal_ub,
                             score_bitmap, N*8, 4)

    def generate_bbox(self, input_list, output_region_proposal, burst):
        """
        :param input_list:
        :param batch_id:
        :param output_region_proposal:
        :return:
        """
        input_bbox_offset = input_list[0]
        input_score_offset = input_list[1]
        output_bbox_offset = input_list[2]

        score = input_list[3]
        bbox = input_list[4]
        region_proposal = input_list[5]
        im_info = input_list[6]

        self.tik_instance.data_move(self.input_score_ub, score[input_score_offset], 0, 1,
                                    burst, 0, 0, 0)

        self.tik_instance.data_move(self.input_bbox_ub[0, 0],
                                    bbox[input_bbox_offset],
                                    0, 1, burst, 0, 0, 0)
        self.tik_instance.data_move(self.input_bbox_ub[1, 0],
                                    bbox[input_bbox_offset+self.H*self.W],
                                    0, 1, burst, 0, 0, 0)
        self.tik_instance.data_move(self.input_bbox_ub[2, 0],
                                    bbox[input_bbox_offset+2*self.H*self.W],
                                    0, 1, burst, 0, 0, 0)
        self.tik_instance.data_move(self.input_bbox_ub[3, 0],
                                    bbox[input_bbox_offset+3*self.H*self.W],
                                    0, 1, burst, 0, 0, 0)

        self.tik_instance.data_move(self.input_region_proposal_ub[0, 0],
                                    region_proposal[input_bbox_offset],
                                    0, 1, burst, 0, 0, 0)
        self.tik_instance.data_move(
            self.input_region_proposal_ub[1, 0],
            region_proposal[input_bbox_offset+self.H*self.W],
            0, 1, burst, 0, 0, 0)
        self.tik_instance.data_move(
            self.input_region_proposal_ub[2, 0],
            region_proposal[input_bbox_offset+2*self.H*self.W],
            0, 1, burst, 0, 0, 0)
        self.tik_instance.data_move(
            self.input_region_proposal_ub[3, 0],
            region_proposal[input_bbox_offset+3*self.H*self.W],
            0, 1, burst, 0, 0, 0)

        self.tik_instance.data_move(self.im_info_ub[0, 0],
                                    im_info[self.batch_id, 0],
                                    0, 1, 1, 0, 0, 0)

        self.get_decode_bbox()

        self.tik_instance.data_move(output_region_proposal[output_bbox_offset],
                                    self.output_region_proposal_ub[0, 0],
                                    0, 1, (self.N*16*8*8)*self.size//32, 0, 0, 0)

class OneCoreProcess:
    """
    OneCoreProcess
    """
    def __init__(self, input_data):
        self.tik_instance = input_data[0]
        self.min_box_size = input_data[1]

        self.input_dtype = input_data[2]
        self.size = input_data[3]
        self.input_shape = input_data[4]

        self.device_core_num = input_data[5]
        self.batch_factor = input_data[6]
        self.ub_size = input_data[7]
        self.num_anchor = self.input_shape[1]//4
        self.H = self.input_shape[2]
        self.W = self.input_shape[3]
        self.reserved_ub_size = 33*8*16*self.size

        self.n = (self.H*self.W)//128

    def get_offset(self, batch_id, i, num):
        """
        :param batch_id:
        :param i:
        :param num:
        :return:
        """
        box_one_batch_num = 4*self.num_anchor*self.H*self.W
        score_one_batch_num = 2*self.num_anchor*self.H*self.W

        input_score_offset = batch_id*score_one_batch_num \
                             + self.num_anchor*self.H*self.W + i*self.H*self.W + num

        input_bbox_offset = batch_id*box_one_batch_num \
                            + 4*i*self.H*self.W + num

        N = (self.num_anchor*self.H*self.W + 127) // 128
        burst = ((N*128 - self.num_anchor*self.H*self.W)*8*self.size + 31)//32
        tail = burst*32//(8*self.size) - (N*128 - self.num_anchor*self.H*self.W)
        output_bbox_offset = batch_id*(N*128+tail)*8 \
                             + i*self.H*self.W*8 + num*8

        return input_bbox_offset, input_score_offset, output_bbox_offset

    def one_core_process_decode_bbox(self, batch_id, score, bbox,
                                     region_proposal, im_info, output_region_proposal):
        """
        :param batch_id:
        :param score:
        :param bbox:
        :param region_proposal:
        :param output_region_proposal:
        :return:
        """

        with self.tik_instance.for_range(0, self.num_anchor) as i:
            ub_size = self.ub_size

            N = (self.H*self.W + 127) // 128
            tiling_n = (ub_size - 32) // self.reserved_ub_size
            tiling_number = self.n // tiling_n

            tail_n = self.n - tiling_number*tiling_n

            tail = self.H*self.W - self.n*16*8

            if tiling_number == 0:
                input_bbox_offset, input_score_offset, output_bbox_offset =\
                self.get_offset(batch_id, i, 0)
                decode_bbox_object = DecodeBbox(self.tik_instance,
                                                (N,
                                                 self.input_dtype,
                                                 batch_id,
                                                 self.min_box_size,
                                                 self.input_shape))
                burst = (self.H*self.W*self.size + 31)//32
                decode_bbox_object.generate_bbox((input_bbox_offset,
                                                  input_score_offset,
                                                  output_bbox_offset,
                                                  score,
                                                  bbox,
                                                  region_proposal,
                                                  im_info),
                                                 output_region_proposal, burst)

            else:
                with self.tik_instance.for_range(0, tiling_number) as j:
                    input_bbox_offset, input_score_offset, \
                    output_bbox_offset = \
                        self.get_offset(batch_id,
                                        i, j*tiling_n*16*8)

                    decode_bbox_object = DecodeBbox(self.tik_instance,
                                                    (tiling_n,
                                                     self.input_dtype,
                                                     batch_id,
                                                     self.min_box_size,
                                                     self.input_shape))

                    burst = (tiling_n*8*16*self.size)//32

                    decode_bbox_object.generate_bbox((input_bbox_offset,
                                                      input_score_offset,
                                                      output_bbox_offset,
                                                      score,
                                                      bbox,
                                                      region_proposal,
                                                      im_info),
                                                     output_region_proposal, burst)
                if tail_n > 0:
                    input_bbox_offset, input_score_offset, \
                    output_bbox_offset = \
                        self.get_offset(batch_id,
                                        i, tiling_number*tiling_n*16*8)

                    decode_bbox_object = DecodeBbox(self.tik_instance,
                                                    (tail_n,
                                                     self.input_dtype,
                                                     batch_id,
                                                     self.min_box_size,
                                                     self.input_shape))

                    burst = (tail_n*8*16*self.size)//32

                    decode_bbox_object.generate_bbox((input_bbox_offset,
                                                      input_score_offset,
                                                      output_bbox_offset,
                                                      score,
                                                      bbox,
                                                      region_proposal,
                                                      im_info),
                                                     output_region_proposal, burst)

                if tail > 0:
                    input_bbox_offset, input_score_offset, \
                    output_bbox_offset = \
                        self.get_offset(batch_id,
                                        i, self.n*16*8)

                    decode_bbox_object = DecodeBbox(self.tik_instance,
                                                    (1,
                                                     self.input_dtype,
                                                     batch_id,
                                                     self.min_box_size,
                                                     self.input_shape))

                    burst = (tail*self.size + 31)//32

                    decode_bbox_object.generate_bbox((input_bbox_offset,
                                                      input_score_offset,
                                                      output_bbox_offset,
                                                      score,
                                                      bbox,
                                                      region_proposal,
                                                      im_info),
                                                     output_region_proposal, burst)
