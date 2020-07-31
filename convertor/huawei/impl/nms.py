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

nms
"""

import math
from te import tik
from te import platform as tbe_platform

# pylint: disable=R0914
# pylint: disable=C0121

class NmsProcess:
    """
    NmsProcess
    """

    def __init__(self, input_data):
        self.tik_instance = input_data[0]
        self.dtype = input_data[1]
        overlap_threshold = input_data[2]
        self.size = input_data[3]
        self.factor = input_data[4]
        self.N = input_data[5]
        self.actual_N = input_data[6]
        self.scale_factor = input_data[7]

        self.thresh = overlap_threshold/(1.0 + overlap_threshold)
        if self.dtype == "float16":
            self.ratio = 1
        elif self.dtype == "float32":
            self.ratio = 2

    def nms(self, i, max_supmatrix_num, input_offset, proposal_box, supvec_ub):
        """
        :param i:
        :param max_supmatrix_num:
        :param input_offset:
        :param proposal_box:
        :param supvec_ub:
        :return:
        """
        proposal_ub = self.tik_instance.Tensor(self.dtype, (self.N, 16, 8),
                                               name="proposal_ub",
                                               scope=tik.scope_ubuf)

        self.tik_instance.data_move(
            proposal_ub, proposal_box[input_offset + i*self.factor*16*8], 0, 1,
            self.actual_N*16*8*self.size // 32, 0, 0)
        self.tik_instance.vmuls(128 // self.ratio, proposal_ub, proposal_ub,
                                self.scale_factor, self.N*self.ratio, 1, 1, 8, 8)

        with self.tik_instance.for_range(0, self.actual_N) as j:
            #calculate area of current j group,one group has 16 proposals
            area_ub = self.tik_instance.Tensor(self.dtype, [16],
                                               name="area_ub",
                                               scope=tik.scope_ubuf)
            self.tik_instance.vrpac(area_ub, proposal_ub[j, 0, 0], 1)

            supmatrix_len = (i*self.factor+j + 1)*16*16 // 16
            max_supmatrix_len = max_supmatrix_num*16 // 16
            tempsupmatrix_ub = self.tik_instance.Tensor("uint16",
                                                        [max_supmatrix_len],
                                                        name="tempsupmatrix_ub",
                                                        scope=tik.scope_ubuf)

            with self.tik_instance.if_scope(i > 0):
                with self.tik_instance.for_range(0, i) as k:
                    if self.factor > 0:
                        tempproposal_ub = self.tik_instance.Tensor(
                            self.dtype, (self.factor, 16, 8),
                            name="tempproposal_ub", scope=tik.scope_ubuf)

                        self.tik_instance.data_move(
                            tempproposal_ub,
                            proposal_box[input_offset + k*self.factor*16*8],
                            0, 1, self.factor*16*8*self.size // 32, 0, 0)
                        self.tik_instance.vmuls(128//self.ratio,
                                                tempproposal_ub,
                                                tempproposal_ub,
                                                self.scale_factor,
                                                self.factor*self.ratio,
                                                1, 1, 8, 8)

                        tempiou_ub = self.tik_instance.Tensor(
                            self.dtype, (self.factor, 16, 16),
                            name="tempiou_ub", scope=tik.scope_ubuf)
                        #calculate iou with j and 0~i group proposal,one group have 16 proposal
                        self.tik_instance.viou(tempiou_ub[0, 0, 0],
                                               tempproposal_ub[0, 0, 0],
                                               proposal_ub[j, 0, 0], self.factor)

                        #calculate totle area
                        temparea_ub = self.tik_instance.Tensor(
                            self.dtype, (self.factor, 16),
                            name="temparea_ub", scope=tik.scope_ubuf)
                        self.tik_instance.vrpac(temparea_ub,
                                                tempproposal_ub[0, 0, 0],
                                                self.factor)
                        tempjoin_ub = self.tik_instance.Tensor(
                            self.dtype, (self.factor, 16, 16),
                            name="tempjoin_ub", scope=tik.scope_ubuf)
                        #calculate totle area with i and 0~i group proposal
                        self.tik_instance.vaadd(tempjoin_ub[0, 0, 0],
                                                temparea_ub[0, 0],
                                                area_ub[0], self.factor)

                        #calculate join*(thresh/(1+thresh))
                        self.tik_instance.vmuls(128 // self.ratio,
                                                tempjoin_ub[0, 0, 0],
                                                tempjoin_ub[0, 0, 0],
                                                self.thresh,
                                                2*self.factor*self.ratio,
                                                1, 1, 8, 8)
                        #compare and generate suppression matrix
                        self.tik_instance.vcmpv_gt(
                            tempsupmatrix_ub[k*self.factor*16],
                            tempiou_ub[0, 0, 0],
                            tempjoin_ub[0, 0, 0],
                            2*self.factor*self.ratio,
                            1, 1, 8, 8)

            tempiou_ub = self.tik_instance.Tensor(self.dtype, (self.N, 16, 16),
                                                  name="tempiou_ub",
                                                  scope=tik.scope_ubuf)
            #calculate iou with j and 0~j group proposal
            self.tik_instance.viou(tempiou_ub[0, 0, 0], proposal_ub[0, 0, 0],
                                   proposal_ub[j, 0, 0], j + 1)

            #calculate totle area
            temparea_ub = self.tik_instance.Tensor(self.dtype,
                                                   (self.N, 16),
                                                   name="temparea_ub",
                                                   scope=tik.scope_ubuf)
            self.tik_instance.vrpac(temparea_ub, proposal_ub[0, 0, 0], j + 1)
            tempjoin_ub = self.tik_instance.Tensor(self.dtype,
                                                   (self.N, 16, 16),
                                                   name="tempjoin_ub",
                                                   scope=tik.scope_ubuf)
            #calculate totle area with j and 0~j group proposal
            self.tik_instance.vaadd(tempjoin_ub[0, 0, 0], temparea_ub[0, 0],
                                    area_ub[0], j + 1)

            #calculate join*(thresh/(1+thresh))
            self.tik_instance.vmuls(128//self.ratio, tempjoin_ub[0, 0, 0],
                                    tempjoin_ub[0, 0, 0],
                                    self.thresh, 2*(j + 1)*self.ratio, 1, 1, 8, 8)
            #compare and generate suppression matrix
            self.tik_instance.vcmpv_gt(tempsupmatrix_ub[(i*self.factor)*16],
                                       tempiou_ub[0, 0, 0],
                                       tempjoin_ub[0, 0, 0],
                                       2*(j + 1)*self.ratio, 1, 1, 8, 8)

            rpn_cor_ir = self.tik_instance.set_rpn_cor_ir(0)
            with self.tik_instance.if_scope(i == 0):
                with self.tik_instance.if_scope(j == 0):
                    rpn_cor_ir = self.tik_instance.rpn_cor(tempsupmatrix_ub[0],
                                                           supvec_ub[0], 1, 1, 1)
                    self.tik_instance.rpn_cor_diag(supvec_ub[0],
                                                   tempsupmatrix_ub[0], rpn_cor_ir)
                with self.tik_instance.else_scope():
                    rpn_cor_ir = self.tik_instance.rpn_cor(tempsupmatrix_ub[0],
                                                           supvec_ub[0], 1, 1,
                                                           i*self.factor + j)
                    self.tik_instance.rpn_cor_diag(
                        supvec_ub[(i*self.factor + j)*16],
                        tempsupmatrix_ub[supmatrix_len - 16], rpn_cor_ir)
            with self.tik_instance.else_scope():
                rpn_cor_ir = self.tik_instance.rpn_cor(tempsupmatrix_ub[0],
                                                       supvec_ub[0],
                                                       1, 1, i*self.factor + j)
                self.tik_instance.rpn_cor_diag(supvec_ub[(i*self.factor + j)*16],
                                               tempsupmatrix_ub[supmatrix_len - 16],
                                               rpn_cor_ir)


def init_select_proposal_ub(tik_instance, dtype, select_proposal_ub, num):
    """
    :param tik_instance:
    :param dtype:
    :param select_proposal_ub:
    :param num:
    :return:
    """
    if dtype == "float16":
        if num > 255:
            cycle = num // 255
            with tik_instance.for_range(0, cycle) as index:
                tik_instance.vector_dup(128, select_proposal_ub[index*255*16, 0], 0, 255, 1, 8)

            if num % 255 > 0:
                tik_instance.vector_dup(128, select_proposal_ub[cycle*255*16, 0],
                                        0, num % 255, 1, 8)
        else:
            tik_instance.vector_dup(128, select_proposal_ub, 0, num, 1, 8)
    else:
        if num > 127:
            cycle = num // 127
            with tik_instance.for_range(0, cycle) as index:
                tik_instance.vector_dup(64, select_proposal_ub[index*127*16, 0], 0, 254, 1, 8)

            if num % 127 > 0:
                tik_instance.vector_dup(64, select_proposal_ub[cycle*127*16, 0], 0,
                                        2*(num % 127), 1, 8)
        else:
            tik_instance.vector_dup(64, select_proposal_ub, 0, num*2, 1, 8)


def nms_no_tiling_select_proposal(input_data, selected_count, input_offset,
                                  proposal_box, supvec_ub, temp_proposal_out):
    """
    :param input_data:
    :param selected_count:
    :param input_offset:
    :param proposal_box:
    :param supvec_ub:
    :param temp_proposal_out:
    :return:
    """
    tik_instance = input_data[0]
    dtype = input_data[1]
    ub_size = input_data[2]
    supvec_ub_size = input_data[3]
    batch_index = input_data[4]
    actual_num = input_data[5]
    post_nms_topn = input_data[6]

    if dtype == "float16":
        size = 2
    elif dtype == "float32":
        size = 4

    #reserved_ub_size of proposal_ub and select_proposal_ub
    reserved_ub_size = (16*8*size)*2
    max_n = ((ub_size - supvec_ub_size - 2 - 2 - 4 - 6) // reserved_ub_size)

    proposal_ub = tik_instance.Tensor(dtype, (max_n*16, 8),
                                      name="proposal_out_ub",
                                      scope=tik.scope_ubuf)
    tik_instance.data_move(proposal_ub, proposal_box[input_offset + 0],
                           0, 1, (actual_num*8*size + 31) // 32, 0, 0)

    select_proposal_ub = tik_instance.Tensor(dtype, (max_n*16, 8),
                                             name="select_proposal_ub",
                                             scope=tik.scope_ubuf)
    init_select_proposal_ub(tik_instance, dtype, select_proposal_ub, max_n)
    temp_select_count = tik_instance.Scalar(dtype="int32")
    temp_select_count.set_as(0)

    index = tik_instance.Scalar(dtype="int32")
    index.set_as(0)

    with tik_instance.for_range(0, actual_num) as i:
        with tik_instance.if_scope(supvec_ub[i] == 0):
            with tik_instance.if_scope(temp_select_count < post_nms_topn):
                with tik_instance.for_range(0, 6) as j:
                    select_proposal_ub[index, j].set_as(proposal_ub[i, j])
                index.set_as(index + 1)
                temp_select_count.set_as(temp_select_count + 1)

    with tik_instance.if_scope(index > 0):
        tik_instance.data_move(temp_proposal_out[batch_index, selected_count, 0],
                               select_proposal_ub[0, 0],
                               0, 1, (index*8*size + 31) // 32, 0, 0)
        selected_count.set_as(selected_count+temp_select_count)


def nms_tiling_select_proposal(input_data, selected_count, input_offset,
                               proposal_box, supvec_ub, temp_proposal_out):
    """
    :param input_data:
    :param selected_count:
    :param input_offset:
    :param proposal_box:
    :param supvec_ub:
    :param temp_proposal_out:
    :return:
    """
    tik_instance = input_data[0]
    dtype = input_data[1]
    size = input_data[2]
    tiling_num = input_data[3]
    tiling = input_data[4]
    tail = input_data[5]
    batch_index = input_data[6]
    post_nms_topn = input_data[7]

    temp_select_count = tik_instance.Scalar(dtype="int32")
    temp_select_count.set_as(0)

    index = tik_instance.Scalar(dtype="int32")
    index.set_as(0)

    with tik_instance.for_range(0, tiling_num) as i:
        proposal_ub = tik_instance.Tensor(dtype, (tiling*16, 8),
                                          name="proposal_out_ub",
                                          scope=tik.scope_ubuf)
        tik_instance.data_move(proposal_ub,
                               proposal_box[input_offset + i*tiling*16*8], 0, 1,
                               (tiling*16*8*size)//32, 0, 0)

        select_proposal_ub = tik_instance.Tensor(dtype, (tiling*16, 8),
                                                 name="select_proposal_ub",
                                                 scope=tik.scope_ubuf)

        init_select_proposal_ub(tik_instance, dtype, select_proposal_ub, tiling)
        index.set_as(0)

        with tik_instance.for_range(0, tiling*16) as j:
            with tik_instance.if_scope(supvec_ub[i*tiling*16 + j] == 0):
                with tik_instance.if_scope(temp_select_count < post_nms_topn):
                    with tik_instance.for_range(0, 6) as k:
                        select_proposal_ub[index, k].set_as(proposal_ub[j, k])
                    index.set_as(index + 1)
                    temp_select_count.set_as(temp_select_count + 1)

        with tik_instance.if_scope(index > 0):
            tik_instance.data_move(
                temp_proposal_out[batch_index, selected_count, 0],
                select_proposal_ub[0, 0], 0, 1, (index*8*size + 31) // 32, 0, 0)
            selected_count.set_as(selected_count+index)

    with tik_instance.if_scope(tail > 0):
        with tik_instance.if_scope(temp_select_count < post_nms_topn):
            proposal_ub = tik_instance.Tensor(dtype, (tiling*16, 8),
                                              name="proposal_out_ub",
                                              scope=tik.scope_ubuf)
            tik_instance.data_move(
                proposal_ub, proposal_box[input_offset + tiling_num*tiling*16*8],
                0, 1, (tail*8*size + 31) // 32, 0, 0)

            select_proposal_ub = tik_instance.Tensor(dtype, (tiling*16, 8),
                                                     name="select_proposal_ub",
                                                     scope=tik.scope_ubuf)
            init_select_proposal_ub(tik_instance, dtype, select_proposal_ub, tiling)

            index.set_as(0)

            with tik_instance.for_range(0, tail) as i:
                with tik_instance.if_scope(supvec_ub[tiling_num*tiling*16 + i] == 0):
                    with tik_instance.if_scope(temp_select_count < post_nms_topn):
                        with tik_instance.for_range(0, 6) as j:
                            select_proposal_ub[index, j].set_as(proposal_ub[i, j])
                        index.set_as(index + 1)
                        temp_select_count.set_as(temp_select_count + 1)

            with tik_instance.if_scope(index > 0):
                tik_instance.data_move(
                    temp_proposal_out[batch_index, selected_count, 0],
                    select_proposal_ub[0, 0], 0, 1, (index*8*size + 31) // 32, 0, 0)
                selected_count.set_as(selected_count + index)


def nms_select_proposal(input_data, selected_count, input_offset, proposal_box,
                        supvec_ub, temp_proposal_out):
    """
    :param input_data:
    :param selected_count:
    :param input_offset:
    :param proposal_box:
    :param supvec_ub:
    :param temp_proposal_out:
    :return:
    """
    tik_instance = input_data[0]
    dtype = input_data[1]
    ub_size = input_data[2]
    supvec_ub_size = input_data[3]
    batch_index = input_data[4]
    actual_num = input_data[5]
    post_nms_topn = input_data[6]

    if dtype == "float16":
        size = 2
    elif dtype == "float32":
        size = 4

    #reserved_ub_size of proposal_ub and select_proposal_ub
    reserved_ub_size = (16*8*size + 16*8*size)*2

    #one 2 is the size of temp_select_count,one 2 is the size of selected_count
    #4 is two scalar size of tiling_num and tail
    tiling = (ub_size-supvec_ub_size-2-2-4)//reserved_ub_size

    tiling_num = tik_instance.Scalar(dtype="uint16")
    tiling_num.set_as(actual_num//16//tiling)

    tail = tik_instance.Scalar(dtype="uint16")
    tail.set_as(actual_num - tiling_num*tiling*16)

    #no tiling
    with tik_instance.if_scope(tiling_num == 0):
        nms_no_tiling_select_proposal(input_data, selected_count, input_offset,
                                      proposal_box, supvec_ub, temp_proposal_out)
    #need tilingn
    with tik_instance.else_scope():
        nms_tiling_select_proposal((tik_instance, dtype, size,
                                    tiling_num, tiling, tail,
                                    batch_index, post_nms_topn),
                                   selected_count, input_offset,
                                   proposal_box, supvec_ub, temp_proposal_out)


def nms_extract(tik_instance, ret_ub, select_proposal_ub, repeat):
    """
    :param tik_instance:
    :param ret_ub:
    :param select_proposal_ub:
    :param repeat:
    :return:
    """
    # tik_instance.vextract(ret_ub[0, 0], select_proposal_ub[0], repeat, 5)
    tik_instance.vextract(ret_ub[1, 0], select_proposal_ub[0], repeat, 0)
    tik_instance.vextract(ret_ub[2, 0], select_proposal_ub[0], repeat, 1)
    tik_instance.vextract(ret_ub[3, 0], select_proposal_ub[0], repeat, 2)
    tik_instance.vextract(ret_ub[4, 0], select_proposal_ub[0], repeat, 3)


def post_proposal_nms_extract(tik_instance, ret_ub, select_proposal_ub, repeat):
    """
    :param tik_instance:
    :param ret_ub:
    :param select_proposal_ub:
    :param repeat:
    :return:
    """
    if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Ascend310",):
        with tik_instance.for_range(0, repeat*16) as offset:
            ret_ub[4, offset].set_as(select_proposal_ub[offset, 4])
    else:
        tik_instance.vextract(ret_ub[4, 0], select_proposal_ub[0], repeat, 4)
    tik_instance.vextract(ret_ub[0, 0], select_proposal_ub[0], repeat, 0)
    tik_instance.vextract(ret_ub[1, 0], select_proposal_ub[0], repeat, 1)
    tik_instance.vextract(ret_ub[2, 0], select_proposal_ub[0], repeat, 2)
    tik_instance.vextract(ret_ub[3, 0], select_proposal_ub[0], repeat, 3)


def get_batch_id(tik_instance, ret_ub, dtype, batch_id, num):
    """
    :param tik_instance:
    :param ret_ub:
    :param dtype:
    :param batch_id:
    :param num:
    :return:
    """

    with tik_instance.if_scope(True):
        #if tik.Dprofile().get_product_name() in ["hisi-cs"]:
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300CS",):
            src_scalar = tik_instance.Scalar(dtype="int32", init_value=batch_id)
            to_dst_scalar = tik_instance.Scalar(dtype="float32")
            dst_scalar = tik_instance.Scalar(dtype="float16")
            tik_instance.scalar_conv('', to_dst_scalar, src_scalar)
            tik_instance.scalar_conv('', dst_scalar, to_dst_scalar)

            tik_instance.vector_dup(16, ret_ub, dst_scalar, num, 1, 1)
        else:
            tmp_buffer_ub = tik_instance.Tensor(dtype, (1, num*16),
                                                name="tmp_buffer_ub",
                                                scope=tik.scope_ubuf)

            batch_id_ub = tik_instance.Tensor("int32", (1, num*16),
                                              name="batch_id_ub",
                                              scope=tik.scope_ubuf)

            tik_instance.vector_dup(16, batch_id_ub[0], batch_id, num, 1, 2)
            if dtype == "float16":
                tik_instance.vconv(16, "", tmp_buffer_ub[0], batch_id_ub[0],
                                   num, 1, 1, 1, 2, 1.0)
                tik_instance.data_move(ret_ub, tmp_buffer_ub,
                                       0, 1, num*16*2 // 32, 0, 0)
            else:
                tik_instance.vconv(16, "", tmp_buffer_ub[0], batch_id_ub[0],
                                   num, 1, 1, 2, 2)
                tik_instance.data_move(ret_ub, tmp_buffer_ub,
                                       0, 1, num*16*4 // 32, 0, 0)


def nms_output_proposal(input_data, post_nms_topn, selected_count, temp_proposal_out, proposal_out):
    """
    :param input_data:
    :param post_nms_topn:
    :param selected_count:
    :param temp_proposal_out:
    :param proposal_out:
    :return:
    """
    tik_instance = input_data[0]
    dtype = input_data[1]
    size = input_data[2]
    ub_size = input_data[3]
    supvec_ub_size = input_data[4]
    batch_index = input_data[5]

    reserved_ub_size = (16*8*size + 16*8*size)*2 + 16*size + 16*4
    tiling = (ub_size-supvec_ub_size - 2 - 2) // reserved_ub_size

    with tik_instance.if_scope(selected_count > post_nms_topn):
        selected_count.set_as(post_nms_topn)

    tiling_num = tik_instance.Scalar(dtype="uint16")
    tiling_num.set_as(selected_count // 16 // tiling)

    tail = tik_instance.Scalar(dtype="uint16")
    tail.set_as(selected_count - tiling_num*tiling*16)

    #no tiling
    with tik_instance.if_scope(tiling_num == 0):
        tmp_post_nms_topn = ((post_nms_topn + 15) // 16)*16

        factor = tmp_post_nms_topn // 16
        if factor > 255:
            factor = 255

        ret_ub = tik_instance.Tensor(dtype, (8, tmp_post_nms_topn), name="ret_ub",
                                     scope=tik.scope_ubuf)
        get_batch_id(tik_instance, ret_ub, dtype, batch_index, factor)

        select_proposal_ub = tik_instance.Tensor(dtype, (factor*16, 8),
                                                 name="select_proposal_ub",
                                                 scope=tik.scope_ubuf)

        tik_instance.data_move(select_proposal_ub,
                               temp_proposal_out[batch_index, 0, 0],
                               0, 1, factor*16*8*size//32, 0, 0)

        nms_extract(tik_instance, ret_ub, select_proposal_ub, factor)
        tik_instance.data_move(proposal_out[batch_index, 0, 0], ret_ub[0, 0],
                               0, 1, tmp_post_nms_topn*5*size//32, 0, 0)
    #need tiling
    with tik_instance.else_scope():
        with tik_instance.for_range(0, tiling_num) as i:
            ret_ub = tik_instance.Tensor(dtype, (8, tiling*16), name="ret_ub",
                                         scope=tik.scope_ubuf)
            get_batch_id(tik_instance, ret_ub, dtype, batch_index, tiling)

            select_proposal_ub = tik_instance.Tensor(dtype, (tiling*16, 8),
                                                     name="select_proposal_ub",
                                                     scope=tik.scope_ubuf)

            tik_instance.data_move(
                select_proposal_ub,
                temp_proposal_out[batch_index, i*tiling*16, 0],
                0, 1, tiling*16*8*size // 32, 0, 0)

            nms_extract(tik_instance, ret_ub, select_proposal_ub, tiling)

            with tik_instance.for_range(0, 5) as j:
                tik_instance.data_move(
                    proposal_out[batch_index, j, i*tiling*16], ret_ub[j, 0],
                    0, 1, tiling*16*size // 32, 0, 0)

        with tik_instance.if_scope(tail > 0):
            tail_num = tik_instance.Scalar(dtype="uint16")
            tail_num.set_as((tail + 15) // 16)
            ret_ub = tik_instance.Tensor(dtype, (8, tiling*16), name="ret_ub",
                                         scope=tik.scope_ubuf)
            get_batch_id(tik_instance, ret_ub, dtype, batch_index, tiling)

            select_proposal_ub = tik_instance.Tensor(dtype, (tiling*16, 8),
                                                     name="select_proposal_ub",
                                                     scope=tik.scope_ubuf)

            tik_instance.data_move(
                select_proposal_ub,
                temp_proposal_out[batch_index, tiling_num*tiling*16, 0],
                0, 1, (tail*size + 31) // 32, 0, 0)

            nms_extract(tik_instance, ret_ub, select_proposal_ub, tail_num)

            with tik_instance.for_range(0, 5) as j:
                tik_instance.data_move(
                    proposal_out[batch_index, j, tiling_num*tiling*16],
                    ret_ub[j, 0], 0, 1, (tail*size + 31) // 32, 0, 0)


def vec_dup(inputs, ub_to_dup, const=0):
    """
    :param inputs:
    :param ub_to_dup:
    :param const:
    :return:
    """
    tik_instance = inputs[0]
    cur_process_num = inputs[1]
    input_dtype = inputs[2]

    if input_dtype == "float16":
        size = 2
        mask = 128
    else:
        size = 4
        mask = 64

    tail = cur_process_num
    tail_n = tail//mask
    tail_tail = tail % mask

    if tail_n != 0:
        tik_instance.vector_dup(mask, ub_to_dup[0], const, tail_n, 1, 8)
    if tail_tail != 0:
        tik_instance.vector_dup(
            tail_tail, ub_to_dup[tail_n*mask], const, 1, 1,
            tail_tail // (16*2 // size))


def vec_conv(tik_instance, src_ub, dst_ub,
             output_dtype, cur_process_num, base):
    """
    :param tik_instance:
    :param src_ub:
    :param dst_ub:
    :param input_dtype:
    :param output_dtype:
    :param cur_process_num:
    :return:
    """
    mask = 64
    tail = cur_process_num
    tail_n = tail//mask
    if output_dtype == "float16":
        size = 2
        if tail_n != 0:
            tik_instance.vconv(mask, "", dst_ub[base, 0],
                               src_ub[0], tail_n, 1,
                               1, mask//(16*2//size), mask//8, 1.0)
        tail_tail = tail % mask
        if tail_tail != 0:
            tik_instance.vconv(tail_tail, "", dst_ub[base, tail_n*mask],
                               src_ub[tail_n*mask], 1, 1, 1,
                               tail_tail//(16*2//size), tail_tail//8, 1.0)

    if output_dtype == "float32":
        size = 4
        if tail_n != 0:
            tik_instance.vconv(mask, "", dst_ub[base, 0],
                               src_ub[0], tail_n, 1,
                               1, mask//(16*2//size), mask//8)
        tail_tail = tail % mask
        if tail_tail != 0:
            tik_instance.vconv(tail_tail, "", dst_ub[base, tail_n*mask],
                               src_ub[tail_n*mask], 1, 1, 1,
                               tail_tail//(16*2//size), tail_tail//8)


def nms_output_tiling_postproposal(tik_instance, inputs, input_index,
                                   temp_proposal_out, proposal_out):
    """
    :param tik_instance:
    :param inputs:
    :param input_index:
    :param temp_proposal_out:
    :param proposal_out:
    :return:
    """
    tiling_num = inputs[0]
    tiling = inputs[1]
    dtype = inputs[2]
    batch_index = input_index[0]
    real_batch_index = input_index[1]
    class_index = input_index[2]
    if dtype == "float16":
        size = 2
        ratio = 1
    elif dtype == "float32":
        size = 4
        ratio = 2
    with tik_instance.if_scope(tiling_num > 1):
        tmp_thread_num = 2
    with tik_instance.else_scope():
        tmp_thread_num = 1

    with tik_instance.for_range(0, tiling_num,
                                thread_num=tmp_thread_num) as i:
        ret_ub = tik_instance.Tensor(dtype, (16, tiling*16), name="ret_ub",
                                     scope=tik.scope_ubuf)
        new_ret_ub = tik_instance.Tensor(dtype, (16, tiling*16),
                                         name="new_ret_ub",
                                         scope=tik.scope_ubuf)

        select_proposal_ub = tik_instance.Tensor(dtype, (tiling*16, 8),
                                                 name="select_proposal_ub",
                                                 scope=tik.scope_ubuf)

        tik_instance.data_move(
            select_proposal_ub,
            temp_proposal_out[batch_index, i*tiling*16, 0], 0, 1,
            tiling*16*8*size//32, 0, 0)
        post_proposal_nms_extract(tik_instance, ret_ub, select_proposal_ub,
                                  tiling)

        # need to do float32
        temp = class_index
        temp_dup = tik_instance.Tensor("int32", (tiling*16,), name="temp_dup",
                                       scope=tik.scope_ubuf)

        vec_dup((tik_instance, tiling*16, "int32"), temp_dup, temp)
        vec_conv(tik_instance, temp_dup, ret_ub, dtype, tiling, 5)

        temp = real_batch_index
        vec_dup((tik_instance, tiling*16, "int32"), temp_dup, temp)
        vec_conv(tik_instance, temp_dup, ret_ub, dtype, tiling, 6)

        tik_instance.data_move(new_ret_ub[0], ret_ub[0, 0], 0, tiling,
                               ratio, 0, 15*ratio)
        tik_instance.data_move(new_ret_ub[16], ret_ub[1, 0], 0, tiling,
                               ratio, 0, 15*ratio)
        for i in range(7):
            tik_instance.data_move(new_ret_ub[i*16], ret_ub[i, 0], 0, tiling,
                                   ratio, 0, 15*ratio)

        if dtype == "float16":
            dst_list = [ret_ub[16*i] for i in range(16)]
            src_list = [new_ret_ub[16*i] for i in range(16)]
            tik_instance.vnchwconv(True, False, dst_list, src_list, tiling,
                                   16, 16)
        else:
            src_list = [new_ret_ub[i*16] for i in range(16)]
            dst_list = [ret_ub[(i // 2)*16 + (i % 2)*8] for i in range(16)]
            tik_instance.vnchwconv(False, False, dst_list, src_list, tiling,
                                   32, 32)
            src_list = [new_ret_ub[i*16 + 8] for i in range(16)]
            dst_list = [ret_ub[(i // 2)*16 + (i % 2)*8 + 8*16] for i in range(16)]
            tik_instance.vnchwconv(False, False, dst_list, src_list, tiling,
                                   32, 32)

        with tik_instance.for_range(0, tiling*16) as loop:
            tik_instance.data_move(proposal_out[class_index,
                                                real_batch_index,
                                                i*tiling*16 + loop, 0],
                                   ret_ub[loop*16], 0, 1, ratio, 0, 0)


def nms_output_tiling_tail_postproposal(tik_instance, inputs, input_index,
                                        temp_proposal_out, proposal_out):
    """
    :param tik_instance:
    :param inputs:
    :param input_index:
    :param temp_proposal_out:
    :param proposal_out:
    :return:
    """
    tiling_num = inputs[0]
    tiling = inputs[1]
    dtype = inputs[2]
    tail = inputs[3]
    batch_index = input_index[0]
    real_batch_index = input_index[1]
    class_index = input_index[2]
    if dtype == "float16":
        size = 2
        ratio = 1
    elif dtype == "float32":
        size = 4
        ratio = 2

    tail_num = tik_instance.Scalar(dtype="uint16")
    tail_num.set_as((tail + 15) // 16)

    ret_ub = tik_instance.Tensor(dtype, (16, tiling*16), name="ret_ub",
                                 scope=tik.scope_ubuf)
    new_ret_ub = tik_instance.Tensor(dtype, (16, tiling*16),
                                     name="new_ret_ub",
                                     scope=tik.scope_ubuf)

    select_proposal_ub = tik_instance.Tensor(dtype, (tiling*16, 8),
                                             name="select_proposal_ub",
                                             scope=tik.scope_ubuf)
    tik_instance.data_move(select_proposal_ub,
                           temp_proposal_out[batch_index,
                                             tiling_num*tiling*16, 0],
                           0, 1, (tail*size + 31) // 32, 0, 0)

    post_proposal_nms_extract(tik_instance, ret_ub, select_proposal_ub,
                              tiling)

    temp = class_index
    temp_dup = tik_instance.Tensor("int32", (tiling*16,),
                                   name="temp_dup", scope=tik.scope_ubuf)
    vec_dup((tik_instance, tiling*16, "int32"), temp_dup, temp)
    vec_conv(tik_instance, temp_dup, ret_ub, dtype, tiling, 5)
    temp = real_batch_index
    vec_dup((tik_instance, tiling*16, "int32"), temp_dup, temp)
    vec_conv(tik_instance, temp_dup, ret_ub, dtype, tiling, 6)

    for i in range(7):
        tik_instance.data_move(new_ret_ub[i*16], ret_ub[i, 0], 0, tiling,
                               ratio, 0, 15*ratio)

    if dtype == "float16":
        dst_list = [ret_ub[16*i] for i in range(16)]
        src_list = [new_ret_ub[16*i] for i in range(16)]
        tik_instance.vnchwconv(True, False, dst_list, src_list, tiling,
                               16, 16)
    else:
        src_list = [new_ret_ub[i*16] for i in range(16)]
        dst_list = [ret_ub[(i // 2)*16 + (i % 2)*8] for i in range(16)]
        tik_instance.vnchwconv(False, False, dst_list, src_list, tiling,
                               32, 32)
        src_list = [new_ret_ub[i*16+8] for i in range(16)]
        dst_list = [ret_ub[(i // 2)*16 + (i % 2)*8 + 8*16] for i in range(16)]
        tik_instance.vnchwconv(False, False, dst_list, src_list, tiling,
                               32, 32)

    with tik_instance.for_range(0, (tail*size+31)//32) as loop:
        tik_instance.data_move(proposal_out[real_batch_index,
                                            class_index,
                                            tiling_num*tiling*16+loop,
                                            0],
                               ret_ub[loop*16], 0, 1, ratio, 0, 0)


def nms_output_postproposal(input_data, post_nms_topn, selected_count,
                            temp_proposal_out, proposal_out):
    """
    :param input_data:
    :param post_nms_topn:
    :param selected_count:
    :param temp_proposal_out:
    :param proposal_out:
    :return:
    """
    tik_instance = input_data[0]
    dtype = input_data[1]
    size = input_data[2]
    ub_size = input_data[3]
    supvec_ub_size = input_data[4]
    batch_index = input_data[5]
    real_batch_index = input_data[6]
    ratio = input_data[7]
    class_index = input_data[8]
    reserved_ub_size = 16*8*size*2 + 16*8*size*3 + (16*16)*2
    tiling = (ub_size-supvec_ub_size - 2 - 2)//reserved_ub_size
    if tiling > 255:
        tiling = 255

    with tik_instance.if_scope(selected_count > post_nms_topn):
        selected_count.set_as(post_nms_topn)

    tiling_num = tik_instance.Scalar(dtype="uint16")
    tiling_num.set_as(selected_count//16//tiling)

    tail = tik_instance.Scalar(dtype="uint16")
    tail.set_as(selected_count - tiling_num*tiling*16)

    with tik_instance.if_scope(tiling_num == 0):
        ret_ub = tik_instance.Tensor(dtype, (16, (post_nms_topn+15)//16*16), name="ret_ub",
                                     scope=tik.scope_ubuf)
        select_proposal_ub = tik_instance.Tensor(
            dtype, ((post_nms_topn+15)//16*16, 8), name="select_proposal_ub",
            scope=tik.scope_ubuf)

        tik_instance.data_move(
            select_proposal_ub, temp_proposal_out[batch_index, 0, 0], 0, 1,
            ((post_nms_topn+15)//16*16*8*size)//32, 0, 0)

        #if tik.Dprofile().get_product_name() in ["hisi-cs"]:
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300CS",):
            src_scalar = tik_instance.Scalar(dtype="int32", init_value=class_index)
            to_dst_scalar = tik_instance.Scalar(dtype="float32")
            dst_scalar = tik_instance.Scalar(dtype="float16")
            tik_instance.scalar_conv('', to_dst_scalar, src_scalar)
            tik_instance.scalar_conv('', dst_scalar, to_dst_scalar)
            with tik_instance.for_range(0, selected_count) as i:
                select_proposal_ub[i, 5].set_as(dst_scalar)
            src_scalar.set_as(real_batch_index)
            tik_instance.scalar_conv('', to_dst_scalar, src_scalar)
            tik_instance.scalar_conv('', dst_scalar, to_dst_scalar)
            with tik_instance.for_range(0, selected_count) as i:
                select_proposal_ub[i, 6].set_as(dst_scalar)
        else:
            temp = class_index
            temp_dup = tik_instance.Tensor("int32", (16,),
                                           name="temp_dup", scope=tik.scope_ubuf)
            vec_dup((tik_instance, 16, "int32"), temp_dup, temp)
            vec_conv(tik_instance, temp_dup, ret_ub, dtype, 16, 0)
            with tik_instance.for_range(0, selected_count) as i:
                select_proposal_ub[i, 5].set_as(ret_ub[0, 0])

            temp = real_batch_index
            vec_dup((tik_instance, 16, "int32"), temp_dup, temp)
            vec_conv(tik_instance, temp_dup, ret_ub, dtype, 16, 0)
            with tik_instance.for_range(0, selected_count) as i:
                select_proposal_ub[i, 6].set_as(ret_ub[0, 0])
        tik_instance.data_move(
            proposal_out[real_batch_index, class_index, 0, 0],
            select_proposal_ub, 0, 1, (post_nms_topn+15)//16*16*ratio//2, 0, 0)

    with tik_instance.else_scope():

        nms_output_tiling_postproposal(
            tik_instance, (tiling_num, tiling, dtype),
            (batch_index, real_batch_index, class_index), temp_proposal_out,
            proposal_out)

        with tik_instance.if_scope(tail > 0):
            nms_output_tiling_tail_postproposal(
                tik_instance, (tiling_num, tiling, dtype, tail),
                (batch_index, real_batch_index, class_index), temp_proposal_out,
                proposal_out)


def get_scale_factor(tik_instance, im_info, batch_id, scale_factor):
    """
    :param tik_instance:
    :param im_info:
    :param batch_id:
    :param scale_factor:
    :return:
    """
    a_ub = tik_instance.Tensor("float16", (1, 16), name="a_ub", scope=tik.scope_ubuf)
    b_ub = tik_instance.Tensor("float16", (1, 16), name="b_ub", scope=tik.scope_ubuf)
    c_ub = tik_instance.Tensor("float16", (1, 16), name="c_ub", scope=tik.scope_ubuf)

    im_info_ub = tik_instance.Tensor("float16", (1, 16), name="im_info_ub", scope=tik.scope_ubuf)
    tik_instance.data_move(im_info_ub[0, 0], im_info[batch_id, 0], 0, 1, 1, 0, 0, 0)

    src_img_height = tik_instance.Scalar(dtype="float16")
    src_img_height.set_as(im_info_ub[0, 0])

    src_img_width = tik_instance.Scalar(dtype="float16")
    src_img_width.set_as(im_info_ub[0, 1])

    tik_instance.vector_dup(16, a_ub, src_img_height, 1, 1, 8)
    tik_instance.vector_dup(16, b_ub, src_img_width, 1, 1, 8)

    # scale_factor : sqrt(65404/(src_h*src_w*2)):sqrt(65404/2)*sqrt(1/(src_h*src_w))
    #             : sqrt(65404/2)*((1/(src_h*src_w))**0.5):sqrt(65404/2)*exp(0.5*ln(1/(src_h*src_w))
    #             : sqrt(65404/2)*exp((-0.5)*(ln(src_h)+ln(src_w))))
    const_factor = math.sqrt(65404/2.0)
    # ln(src_h)
    tik_instance.vln(16, a_ub, a_ub, 1, 1, 1, 8, 8)
    # ln(src_w)
    tik_instance.vln(16, b_ub, b_ub, 1, 1, 1, 8, 8)
    # ln(src_h)+ln(src_w)
    tik_instance.vadd(16, c_ub, a_ub, b_ub, 1, 1, 1, 1, 8, 8, 8)
    # (-0.5)ln(src_h)+ln(src_w)
    tik_instance.vmuls(16, c_ub, c_ub, (-0.5), 1, 1, 1, 8, 8)
    # exp((-0.5)*(ln(src_h)+ln(src_w))))
    tik_instance.vexp(16, c_ub, c_ub, 1, 1, 1, 8, 8)
    # sqrt(65404/2)*exp((-0.5)*(ln(src_h)+ln(src_w))))
    tik_instance.vmuls(16, c_ub, c_ub, const_factor, 1, 1, 1, 8, 8)

    scale_factor.set_as(c_ub[0, 0])


def get_scale_factor_lhisi(tik_instance, im_info, batch_id, scale_factor):
    """
    :param tik_instance:
    :param im_info:
    :param batch_id:
    :param scale_factor:
    :return:
    """
    im_info_ub = tik_instance.Tensor("float16", (1, 16),
                                     name="im_info_ub", scope=tik.scope_ubuf)
    tik_instance.data_move(im_info_ub[0, 0],
                           im_info[batch_id, 0], 0, 1, 1, 0, 0, 0)

    src_h = tik_instance.Scalar(dtype="float16")
    src_w = tik_instance.Scalar(dtype="float16")
    src_h.set_as(im_info_ub[0, 0])
    src_w.set_as(im_info_ub[0, 1])

    src_h_fp32 = tik_instance.Scalar(dtype="float32")
    src_w_fp32 = tik_instance.Scalar(dtype="float32")
    tik_instance.scalar_conv('', src_h_fp32, src_h)
    tik_instance.scalar_conv('', src_w_fp32, src_w)

    tik_instance.scalar_sqrt(src_h_fp32, src_h_fp32)
    tik_instance.scalar_sqrt(src_w_fp32, src_w_fp32)

    const_factor = math.sqrt(65404/2.0)
    scale_factor_fp32 = tik_instance.Scalar(dtype="float32")

    scale_factor_fp32.set_as(const_factor * (1 / (src_h_fp32 * src_w_fp32)))

    tik_instance.scalar_conv('', scale_factor, scale_factor_fp32)


def get_scale_factor_mini(tik_instance, im_info, batch_id, scale_factor):
    """
    :param tik_instance:
    :param im_info:
    :param batch_id:
    :param scale_factor:
    :return:
    """
    im_info_ub = tik_instance.Tensor("float16", (1, 16), name="im_info_ub", scope=tik.scope_ubuf)
    tik_instance.data_move(im_info_ub[0, 0], im_info[batch_id, 0], 0, 1, 1, 0, 0, 0)
    src0_ub = tik_instance.Tensor("float32", (8,), name="src0_ub", scope=tik.scope_ubuf)
    tik_instance.vconv(8, "", src0_ub[0], im_info_ub[0],
                       1, 0, 0, 0, 0)
    src1_ub = tik_instance.Tensor("float32", (8,), name="src1_ub", scope=tik.scope_ubuf)
    src1_ub[0].set_as(src0_ub[1])
    dst_ub = tik_instance.Tensor("float32", (8,), name="dst_ub", scope=tik.scope_ubuf)
    src0_list = [src0_ub[0]]
    src1_list = [src1_ub[0]]
    dst_list = [dst_ub[0]]
    tik_instance.scatter_vmul(8, dst_list, src0_list, src1_list, 1, 0, 0, 0)

    index_reg = tik_instance.Scalar(dtype="float32")
    index_reg.set_as(1/32752.0)
    src0_ub[0].set_as(index_reg)

    src0_list = [src0_ub[0]]
    src1_list = [src1_ub[0]]
    dst_list = [dst_ub[0]]
    tik_instance.scatter_vmul(8, src1_list, dst_list, src0_list, 1, 0, 0, 0)

    # //310 910 610 es
    src_ub = tik_instance.Tensor("float32", (8,), name="src_ub", scope=tik.scope_ubuf)
    work_tensor = tik_instance.Tensor("float32", (4*8,), name="work_tensor", scope=tik.scope_ubuf)
    tik_instance.vec_rsqrt_high_preci(8, src_ub, src1_ub, work_tensor[0:], 1, 0, 0)

    src_ub_fp16 = tik_instance.Tensor("float16", (16,), name="src_ub_fp16", scope=tik.scope_ubuf)
    tik_instance.vconv(16, "", src_ub_fp16[0], src_ub[0], 1, 0, 0, 0, 0)
    scale_factor.set_as(src_ub_fp16[0])


def init_proposal_out(tik_instance, dtype, batch_id, post_nms_topn, proposal_out):
    """
    :param tik_instance:
    :param dtype:
    :param batch_id:
    :param post_nms_topn:
    :param proposal_out:
    :return:
    """

    tmp_post_nms_topn = ((post_nms_topn + 15) // 16)*16
    a_ub = tik_instance.Tensor(dtype, (1, tmp_post_nms_topn), name="a_ub", scope=tik.scope_ubuf)
    number = tmp_post_nms_topn // 16

    if dtype == "float16":
        size = 2
    else:
        size = 4

    with tik_instance.for_range(0, number) as index:
        if dtype == "float16":
            tik_instance.vector_dup(16, a_ub[0, index*16], 0, 1, 1, 8)
        else:
            tik_instance.vector_dup(8, a_ub[0, index*16], 0, 1, 1, 8)

    with tik_instance.for_range(0, 5) as index:
        tik_instance.data_move(proposal_out[batch_id, index, 0], a_ub, 0, 1,
                               tmp_post_nms_topn*size // 32, 0, 0, 0)


def cce_nms(input_data, temp_proposal_out, proposal_box, proposal_actual_num,
            output_actual_bbox_num, proposal_out, used_in_proposal=True, used_in_ssd=False):
    """
    :param input_data:
    :param temp_proposal_out:
    :param proposal_box:
    :param proposal_actual_num:
    :param output_actual_bbox_num:
    :param proposal_out:
    :param used_in_proposal
    :return:
    """
    dtype = input_data[0]
    ub_size = input_data[1]
    overlap_threshold = input_data[2]
    batch_index = input_data[3]
    pre_nms_topn = input_data[4]
    post_nms_topn = input_data[5]
    input_offset = input_data[6]
    im_info = input_data[7]
    tik_instance = input_data[8]
    if used_in_proposal == False:
        class_index = input_data[10]
        real_batch_index = input_data[11]
        if dtype == "float16":
            size = 2
            ratio = 1
        elif dtype == "float32":
            size = 4
            ratio = 2

    with tik_instance.if_scope(proposal_actual_num == 0):
        output_actual_bbox_num_ub = tik_instance.Tensor(
            "int32", [8], name="output_actual_bbox_num", scope=tik.scope_ubuf)
        tik_instance.vector_dup(8, output_actual_bbox_num_ub, 0, 1, 1, 8)
        if used_in_proposal == True:
            tik_instance.data_move(output_actual_bbox_num[batch_index, 0],
                                   output_actual_bbox_num_ub, 0, 1, 1, 0, 0)
            init_proposal_out(tik_instance, dtype, batch_index, post_nms_topn, proposal_out)
        else:
            tik_instance.data_move(output_actual_bbox_num[real_batch_index, class_index, 0],
                                   output_actual_bbox_num_ub, 0, 1, 1, 0, 0)
    with tik_instance.else_scope():
        if dtype == "float16":
            size = 2
            ratio = 1
        elif dtype == "float32":
            size = 4
            ratio = 2

        N = (pre_nms_topn + 15)//16
        if N % 2 != 0:
            N = N + 1

        supvec_ub = tik_instance.Tensor("uint16", [N*16], name="supvec_ub",
                                        scope=tik.scope_ubuf)
        tik_instance.vector_dup(32, supvec_ub[0], 1, N//2, 1, 2)
        scalar_uint16 = tik_instance.Scalar(dtype="uint16")
        scalar_uint16.set_as(0)
        supvec_ub[0].set_as(scalar_uint16)

        scale_factor = tik_instance.Scalar(dtype=dtype)
        scale_factor.set_as(1)
        if not used_in_ssd:
            if dtype == "float16":
                if used_in_proposal == True:
                    with tik_instance.new_stmt_scope():
                        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES",
                                                                                 "Hi3796CV300CS",
                                                                                 "Ascend610",
                                                                                 "Ascend620"):
                            get_scale_factor_lhisi(tik_instance, im_info, batch_index, scale_factor)
                        elif tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Ascend310",):
                            get_scale_factor_mini(tik_instance, im_info, batch_index, scale_factor)
                        else:
                            get_scale_factor(tik_instance, im_info, batch_index, scale_factor)
                else:
                    with tik_instance.new_stmt_scope():
                        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES",
                                                                                 "Hi3796CV300CS",
                                                                                 "Ascend610",
                                                                                 "Ascend620"):
                            get_scale_factor_lhisi(tik_instance, im_info, real_batch_index,
                                                   scale_factor)
                        elif tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Ascend310",):
                            get_scale_factor_mini(tik_instance, im_info, real_batch_index,
                                                  scale_factor)
                        else:
                            get_scale_factor(tik_instance, im_info, real_batch_index, scale_factor)

        reserved_ub_size = (12*16*8*size)*2

        factor = ((ub_size - N*16*2) - N*16*2-4) // reserved_ub_size

        tiling_num = tik_instance.Scalar(dtype="uint16")
        tiling_num.set_as((proposal_actual_num // 16) // factor)

        tail = tik_instance.Scalar(dtype="uint16")
        tail.set_as(proposal_actual_num - tiling_num*factor*16)

        #no tiling
        with tik_instance.if_scope(tiling_num == 0):
            nms_process_object = NmsProcess((tik_instance,
                                             dtype,
                                             overlap_threshold,
                                             size,
                                             0,
                                             factor,
                                             (proposal_actual_num + 15)//16,
                                             scale_factor))

            nms_process_object.nms(0, N*16, input_offset, proposal_box,
                                   supvec_ub)

        #need tiling
        with tik_instance.else_scope():
            with tik_instance.for_range(0, tiling_num) as i:
                nms_process_object = NmsProcess((tik_instance,
                                                 dtype,
                                                 overlap_threshold,
                                                 size,
                                                 factor,
                                                 factor,
                                                 factor,
                                                 scale_factor))
                nms_process_object.nms(i, N*16, input_offset, proposal_box,
                                       supvec_ub)

            with tik_instance.if_scope(tail > 0):
                tail_num = tik_instance.Scalar(dtype="uint16")
                tail_num.set_as((tail+15)//16)
                nms_process_object = NmsProcess((tik_instance,
                                                 dtype,
                                                 overlap_threshold,
                                                 size,
                                                 factor,
                                                 factor,
                                                 tail_num,
                                                 scale_factor))
                nms_process_object.nms(tiling_num, N*16, input_offset,
                                       proposal_box, supvec_ub)

        selected_count = tik_instance.Scalar(dtype="int32")
        selected_count.set_as(0)

        with tik_instance.if_scope(True):
            nms_select_proposal(
                (tik_instance, dtype, ub_size, N*16*2, batch_index,
                 proposal_actual_num, post_nms_topn, pre_nms_topn),
                selected_count, input_offset, proposal_box, supvec_ub,
                temp_proposal_out)

        if used_in_proposal == True:
            nms_output_proposal((tik_instance, dtype, size, ub_size, N*16*2,
                                 batch_index, ratio), post_nms_topn,
                                selected_count, temp_proposal_out, proposal_out)
        else:
            nms_output_postproposal(
                (tik_instance, dtype, size, ub_size, N*16*2, batch_index,
                 real_batch_index, ratio, class_index), post_nms_topn,
                selected_count, temp_proposal_out, proposal_out)

        output_actual_bbox_num_ub = tik_instance.Tensor(
            "int32", [8], name="output_actual_bbox_num", scope=tik.scope_ubuf)
        tik_instance.vector_dup(8, output_actual_bbox_num_ub, 0, 1, 1, 8)
        output_actual_bbox_num_ub[0].set_as(selected_count)
        if used_in_proposal == True:
            tik_instance.data_move(output_actual_bbox_num[batch_index, 0],
                                   output_actual_bbox_num_ub, 0, 1, 1, 0, 0)
        else:
            tik_instance.data_move(output_actual_bbox_num[real_batch_index,
                                                          class_index,
                                                          0],
                                   output_actual_bbox_num_ub, 0, 1, 1, 0, 0)
