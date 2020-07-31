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

fsr_detection_output
"""

from te import tik
from te import platform as cce
from topi.cce import util
from impl import nms
from impl import topk
from impl import constant_util as constant
from te import platform as tbe_platform

# pylint: disable=R0913
# pylint: disable=R0914
# pylint: disable=R0915
# pylint: disable=W0201
# pylint: disable=C0111
# pylint: disable=C0121

NoneType = type(None)
MAX_REPEAT_TIME = 255
FP16_ALIGN_NUM = 16
TO_ALIGN_NUM = 15
FP16_SIZE = 2
FP16_MASK = 128
FP16_RATIO = 1
FP32_SIZE = 4
FP32_MASK = 64
FP32_RATIO = 2
BLOCK_SIZE = 32
VECTOR_BLOCK_SIZE = 256
DATA_EIGHT = 8
DATA_ONE = 1

def get_params(dtype):
    """
    :param dtype:
    :return:
    """
    if dtype == "float16":
        size = FP16_SIZE
        mask = FP16_MASK
        ratio = FP16_RATIO
    elif dtype == "float32":
        size = FP32_SIZE
        mask = FP32_MASK
        ratio = FP32_RATIO
    return size, mask, ratio


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

    size, mask, _ = get_params(input_dtype)

    repeat = (cur_process_num//mask)//MAX_REPEAT_TIME
    with tik_instance.if_scope(cur_process_num//mask > MAX_REPEAT_TIME):
        with tik_instance.for_range(
                0, repeat) as i:
            tik_instance.vector_dup(
                mask, ub_to_dup[MAX_REPEAT_TIME*mask*i], const,
                MAX_REPEAT_TIME, DATA_ONE, DATA_EIGHT)
    tail = cur_process_num % (MAX_REPEAT_TIME*mask)
    tail_n = tail//mask
    if tail_n != 0:
        tik_instance.vector_dup(mask, ub_to_dup[MAX_REPEAT_TIME*mask*repeat],
                                const, tail_n, DATA_ONE,
                                DATA_EIGHT)
    tail_tail = tail % mask
    if tail_tail != 0:
        tik_instance.vector_dup(
            tail_tail, ub_to_dup[MAX_REPEAT_TIME*mask*repeat+tail_n*mask],
            const, DATA_ONE, DATA_ONE,
            tail_tail//(BLOCK_SIZE//size))


def vec_muls(inputs, dst, const, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param const:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src = inputs[1]
    input_dtype = inputs[2]

    size, mask, _ = get_params(input_dtype)

    repeat = (cur_process_num//mask)//MAX_REPEAT_TIME
    if cur_process_num//mask > MAX_REPEAT_TIME:
        for i in range(repeat):
            tik_instance.vmuls(mask, dst[MAX_REPEAT_TIME*mask*i],
                               src[MAX_REPEAT_TIME*mask*i],
                               const, MAX_REPEAT_TIME,
                               DATA_ONE, DATA_ONE,
                               DATA_EIGHT,
                               DATA_EIGHT)
    tail = cur_process_num % (mask*MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vmuls(mask, dst[MAX_REPEAT_TIME*mask*repeat],
                           src[MAX_REPEAT_TIME*mask*repeat],
                           const, tail_n,
                           DATA_ONE, DATA_ONE,
                           DATA_EIGHT,
                           DATA_EIGHT)
    tail_tail = tail%mask
    if tail_tail > 0:
        tik_instance.vmuls(tail_tail,
                           dst[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                           src[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                           const, DATA_ONE, DATA_ONE, DATA_ONE,
                           tail_tail//(BLOCK_SIZE//size),
                           tail_tail//(BLOCK_SIZE//size))


def vec_sub(inputs, dst, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src1 = inputs[1]
    src2 = inputs[2]
    input_dtype = inputs[3]

    size, mask, _ = get_params(input_dtype)

    repeat = (cur_process_num//mask)//MAX_REPEAT_TIME
    if cur_process_num//mask > MAX_REPEAT_TIME:
        for i in range(repeat):
            tik_instance.vsub(mask, dst[MAX_REPEAT_TIME*mask*i],
                              src1[MAX_REPEAT_TIME*mask*i],
                              src2[MAX_REPEAT_TIME*mask*i],
                              MAX_REPEAT_TIME, DATA_ONE,
                              DATA_ONE, DATA_ONE,
                              DATA_EIGHT,
                              DATA_EIGHT,
                              DATA_EIGHT)
    tail = cur_process_num % (mask*MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vsub(mask, dst[MAX_REPEAT_TIME*mask*repeat],
                          src1[MAX_REPEAT_TIME*mask*repeat],
                          src2[MAX_REPEAT_TIME*mask*repeat],
                          tail_n, DATA_ONE,
                          DATA_ONE, DATA_ONE,
                          DATA_EIGHT,
                          DATA_EIGHT,
                          DATA_EIGHT)
    tail_tail = tail % mask
    if tail_tail > 0:
        tik_instance.vsub(tail_tail,
                          dst[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src1[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src2[MAX_REPEAT_TIME*mask*repeat+mask*tail_n], DATA_ONE,
                          DATA_ONE, DATA_ONE,
                          DATA_ONE,
                          tail_tail//(BLOCK_SIZE//size),
                          tail_tail//(BLOCK_SIZE//size),
                          tail_tail//(BLOCK_SIZE//size))


def vec_adds(inputs, dst, const, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param const:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src = inputs[1]
    input_dtype = inputs[2]

    size, mask, _ = get_params(input_dtype)
    repeat = (cur_process_num//mask)//MAX_REPEAT_TIME
    if cur_process_num//mask > MAX_REPEAT_TIME:
        for i in range(repeat):
            tik_instance.vadds(mask, dst[MAX_REPEAT_TIME*mask*i],
                               src[MAX_REPEAT_TIME*mask*i],
                               const, MAX_REPEAT_TIME,
                               DATA_ONE, DATA_ONE,
                               DATA_EIGHT,
                               DATA_EIGHT)
    tail = cur_process_num % (mask*MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vadds(mask, dst[MAX_REPEAT_TIME*mask*repeat],
                           src[MAX_REPEAT_TIME*mask*repeat],
                           const, tail_n,
                           DATA_ONE, DATA_ONE,
                           DATA_EIGHT,
                           DATA_EIGHT)
    tail_tail = tail%mask
    if tail_tail > 0:
        tik_instance.vadds(tail_tail,
                           dst[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                           src[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                           const, DATA_ONE, DATA_ONE, DATA_ONE,
                           tail_tail//(BLOCK_SIZE//size),
                           tail_tail//(BLOCK_SIZE//size))


def vec_add(inputs, dst, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src1 = inputs[1]
    src2 = inputs[2]
    input_dtype = inputs[3]

    size, mask, _ = get_params(input_dtype)
    repeat = (cur_process_num//mask)//MAX_REPEAT_TIME
    if cur_process_num//mask > MAX_REPEAT_TIME:
        for i in range(repeat):
            tik_instance.vadd(mask, dst[MAX_REPEAT_TIME*mask*i],
                              src1[MAX_REPEAT_TIME*mask*i],
                              src2[MAX_REPEAT_TIME*mask*i],
                              MAX_REPEAT_TIME,
                              DATA_ONE, DATA_ONE,
                              DATA_ONE, DATA_EIGHT,
                              DATA_EIGHT,
                              DATA_EIGHT)
    tail = cur_process_num % (mask*MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vadd(mask, dst[MAX_REPEAT_TIME*mask*repeat],
                          src1[MAX_REPEAT_TIME*mask*repeat],
                          src2[MAX_REPEAT_TIME*mask*repeat],
                          tail_n, DATA_ONE,
                          DATA_ONE, DATA_ONE,
                          DATA_EIGHT,
                          DATA_EIGHT,
                          DATA_EIGHT)
    tail_tail = tail%mask
    if tail_tail > 0:
        tik_instance.vadd(tail_tail,
                          dst[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src1[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src2[MAX_REPEAT_TIME*mask*repeat+mask*tail_n], DATA_ONE,
                          DATA_ONE,
                          DATA_ONE,
                          DATA_ONE,
                          tail_tail//(BLOCK_SIZE//size),
                          tail_tail//(BLOCK_SIZE//size),
                          tail_tail//(BLOCK_SIZE//size))


def vec_mla(inputs, dst, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src1 = inputs[1]
    src2 = inputs[2]
    input_dtype = inputs[3]

    size, mask, _ = get_params(input_dtype)
    repeat = (cur_process_num//mask)//MAX_REPEAT_TIME
    if cur_process_num//mask > MAX_REPEAT_TIME:
        for i in range(repeat):
            tik_instance.vmla(mask, dst[MAX_REPEAT_TIME*mask*i],
                              src1[MAX_REPEAT_TIME*mask*i],
                              src2[MAX_REPEAT_TIME*mask*i], MAX_REPEAT_TIME,
                              DATA_ONE, DATA_ONE,
                              DATA_ONE, DATA_EIGHT,
                              DATA_EIGHT,
                              DATA_EIGHT)
    tail = cur_process_num % (mask*MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vmla(mask, dst[MAX_REPEAT_TIME*mask*repeat],
                          src1[MAX_REPEAT_TIME*mask*repeat],
                          src2[MAX_REPEAT_TIME*mask*repeat], tail_n,
                          DATA_ONE, DATA_ONE,
                          DATA_ONE,
                          DATA_EIGHT,
                          DATA_EIGHT,
                          DATA_EIGHT)
    tail_tail = tail % mask
    if tail_tail > 0:
        tik_instance.vmla(tail_tail,
                          dst[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src1[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src2[MAX_REPEAT_TIME*mask*repeat+mask*tail_n], DATA_ONE,
                          DATA_ONE,
                          DATA_ONE, DATA_ONE,
                          tail_tail//(BLOCK_SIZE//size),
                          tail_tail//(BLOCK_SIZE//size),
                          tail_tail//(BLOCK_SIZE//size))


def vec_exp(inputs, dst, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src = inputs[1]
    input_dtype = inputs[2]

    size, mask, _ = get_params(input_dtype)
    repeat = (cur_process_num//mask)//MAX_REPEAT_TIME
    if cur_process_num//mask > MAX_REPEAT_TIME:
        for i in range((cur_process_num//mask)//MAX_REPEAT_TIME):
            tik_instance.vexp(mask, dst[MAX_REPEAT_TIME*mask*i],
                              src[MAX_REPEAT_TIME*mask*i], MAX_REPEAT_TIME,
                              DATA_ONE, DATA_ONE,
                              DATA_EIGHT,
                              DATA_EIGHT)
    tail = cur_process_num % (mask*MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vexp(mask, dst[MAX_REPEAT_TIME*mask*repeat],
                          src[MAX_REPEAT_TIME*mask*repeat],
                          tail_n, DATA_ONE, DATA_ONE,
                          DATA_EIGHT,
                          DATA_EIGHT)
    tail_tail = tail % mask
    if tail_tail > 0:
        tik_instance.vexp(tail_tail,
                          dst[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          DATA_ONE, DATA_ONE, DATA_ONE,
                          tail_tail//(BLOCK_SIZE//size),
                          tail_tail//(BLOCK_SIZE//size))


def vec_mul(inputs, dst, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src1 = inputs[1]
    src2 = inputs[2]
    input_dtype = inputs[3]
    size, mask, _ = get_params(input_dtype)

    repeat = (cur_process_num//mask)//MAX_REPEAT_TIME
    if cur_process_num//mask > MAX_REPEAT_TIME:
        for i in range((cur_process_num//mask)//MAX_REPEAT_TIME):
            tik_instance.vmul(mask, dst[MAX_REPEAT_TIME*mask*i],
                              src1[MAX_REPEAT_TIME*mask*i],
                              src2[MAX_REPEAT_TIME*mask*i], MAX_REPEAT_TIME,
                              DATA_ONE, DATA_ONE,
                              DATA_ONE,
                              DATA_EIGHT,
                              DATA_EIGHT,
                              DATA_EIGHT)
    tail = cur_process_num % (mask*MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vmul(mask, dst[MAX_REPEAT_TIME*mask*repeat],
                          src1[MAX_REPEAT_TIME*mask*repeat],
                          src2[MAX_REPEAT_TIME*mask*repeat], tail_n,
                          DATA_ONE, DATA_ONE,
                          DATA_ONE,
                          DATA_EIGHT,
                          DATA_EIGHT,
                          DATA_EIGHT)
    tail_tail = tail%mask
    if tail_tail > 0:
        tik_instance.vmul(tail_tail,
                          dst[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src1[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src2[MAX_REPEAT_TIME*mask*repeat+mask*tail_n], DATA_ONE,
                          DATA_ONE,
                          DATA_ONE, DATA_ONE,
                          tail_tail//(BLOCK_SIZE//size),
                          tail_tail//(BLOCK_SIZE//size),
                          tail_tail//(BLOCK_SIZE//size))


def vec_min(inputs, dst, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src1 = inputs[1]
    src2 = inputs[2]
    input_dtype = inputs[3]
    size, mask, _ = get_params(input_dtype)

    repeat = (cur_process_num//mask)//MAX_REPEAT_TIME
    if cur_process_num//mask > MAX_REPEAT_TIME:
        for i in range(repeat):
            tik_instance.vmin(mask, dst[MAX_REPEAT_TIME*mask*i],
                              src1[MAX_REPEAT_TIME*mask*i],
                              src2[MAX_REPEAT_TIME*mask*i], MAX_REPEAT_TIME,
                              DATA_ONE, DATA_ONE,
                              DATA_ONE, DATA_EIGHT,
                              DATA_EIGHT,
                              DATA_EIGHT)
    tail = cur_process_num % (mask*MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vmin(mask, dst[MAX_REPEAT_TIME*mask*repeat],
                          src1[MAX_REPEAT_TIME*mask*repeat],
                          src2[MAX_REPEAT_TIME*mask*repeat], tail_n,
                          DATA_ONE, DATA_ONE,
                          DATA_ONE,
                          DATA_EIGHT,
                          DATA_EIGHT,
                          DATA_EIGHT)
    tail_tail = tail%mask
    if tail_tail > 0:
        tik_instance.vmin(tail_tail,
                          dst[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src1[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src2[MAX_REPEAT_TIME*mask*repeat+mask*tail_n], DATA_ONE,
                          DATA_ONE,
                          DATA_ONE, DATA_ONE,
                          tail_tail//(BLOCK_SIZE//size),
                          tail_tail//(BLOCK_SIZE//size),
                          tail_tail//(BLOCK_SIZE//size))


def vec_relu(inputs, dst, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src = inputs[1]
    input_dtype = inputs[2]
    size, mask, _ = get_params(input_dtype)

    repeat = (cur_process_num//mask)//MAX_REPEAT_TIME
    if cur_process_num//mask > MAX_REPEAT_TIME:
        for i in range((cur_process_num//mask)//MAX_REPEAT_TIME):
            tik_instance.vrelu(mask, dst[MAX_REPEAT_TIME*mask*i],
                               src[MAX_REPEAT_TIME*mask*i],
                               MAX_REPEAT_TIME,
                               DATA_ONE, DATA_ONE,
                               DATA_EIGHT,
                               DATA_EIGHT)
    tail = cur_process_num % (mask*MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vrelu(mask, dst[MAX_REPEAT_TIME*mask*repeat],
                           src[MAX_REPEAT_TIME*mask*repeat],
                           tail_n,
                           DATA_ONE, DATA_ONE,
                           DATA_EIGHT,
                           DATA_EIGHT)
    tail_tail = tail%mask
    if tail_tail > 0:
        tik_instance.vrelu(tail_tail,
                           dst[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                           src[MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                           DATA_ONE,
                           DATA_ONE, DATA_ONE,
                           tail_tail//(BLOCK_SIZE//size),
                           tail_tail//(BLOCK_SIZE//size))


def vec_concat(inputs, dst, cur_process_num, const):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :param const:
    :return:
    """
    tik_instance = inputs[0]
    src = inputs[1]
    input_dtype = inputs[2]
    _, mask, _ = get_params(input_dtype)

    repeat = (cur_process_num//FP16_ALIGN_NUM)//MAX_REPEAT_TIME
    if cur_process_num//FP16_ALIGN_NUM > MAX_REPEAT_TIME:
        for i in range((cur_process_num//FP16_ALIGN_NUM)//MAX_REPEAT_TIME):
            tik_instance.vconcat(dst[MAX_REPEAT_TIME*mask*i],
                                 src[MAX_REPEAT_TIME*FP16_ALIGN_NUM*i],
                                 MAX_REPEAT_TIME, const)
    tail = cur_process_num % (FP16_ALIGN_NUM*MAX_REPEAT_TIME)
    if tail > 0:
        tik_instance.vconcat(dst[MAX_REPEAT_TIME*mask*repeat],
                             src[MAX_REPEAT_TIME*FP16_ALIGN_NUM*repeat],
                             tail//FP16_ALIGN_NUM, const)


def get_ub_size():
    """
    :return:
    """
    ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
    return ub_size


def filter_device_core(batch):
    """
    :param batch:
    :return:
    """
    device_core_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
    if batch >= device_core_num:
        batch_factor = batch//device_core_num
        batch_factor_tail = batch - batch_factor*device_core_num
    else:
        batch_factor = batch
        batch_factor_tail = 0
        device_core_num = DATA_ONE

    return device_core_num, batch_factor, batch_factor_tail


def call_topk_sort(tik_instance, input_topk, output):
    """
    :param tik_instance:
    :param input_topk:
    :param output:
    :return:
    """
    max_rois_num = input_topk[0]
    score_threshold = input_topk[1]
    pre_nms_topn = input_topk[2]
    output_box = input_topk[3]
    mem_swap = input_topk[4]

    batch_id = output[0]
    regions_sorted = output[1]
    proposal_actual_num = output[2]

    k = pre_nms_topn

    topk_input = {
        "proposal_num": max_rois_num,
        "k": k,
        "score_threshold": score_threshold,
        "regions_orig": output_box,
        "mem_swap": mem_swap,
        # "batch_offset": batch_offset
    }

    topk_out = {
        "batch_id": batch_id,
        "regions_sorted": regions_sorted,
        "proposal_actual_num": proposal_actual_num,
    }

    topk.tik_topk(tik_instance, topk_input, topk_out)


class DecodeRois:
    """
    Update Decode
    """
    def __init__(self, tik_instance, input_data, tiling_flage):
        """
        :param tik_instance:
        :param input_data:
        """
        self.tik_instance = tik_instance
        self.cur_process_num = input_data[0]
        self.input_dtype = input_data[1]
        self.image_info = input_data[2]
        self.num_class = input_data[3]
        if tiling_flage:
            shape = (self.cur_process_num//FP16_ALIGN_NUM, FP16_ALIGN_NUM)
            self.output_region_proposal_ub = tik_instance.Tensor(
                self.input_dtype, (self.cur_process_num//FP16_ALIGN_NUM,
                                   FP16_ALIGN_NUM, constant.REPEAT_STRIDE_EIGHT),
                name="output_region_proposal_ub", scope=tik.scope_ubuf)
            vec_dup((tik_instance,
                     self.cur_process_num*DATA_EIGHT,
                     self.input_dtype),
                    self.output_region_proposal_ub)
        else:
            shape = (self.cur_process_num//FP16_ALIGN_NUM*self.num_class, FP16_ALIGN_NUM)
            self.output_region_proposal_ub = tik_instance.Tensor(
                self.input_dtype, (self.num_class, self.cur_process_num//FP16_ALIGN_NUM,
                                   FP16_ALIGN_NUM, DATA_EIGHT),
                name="output_region_proposal_ub", scope=tik.scope_ubuf)

            vec_dup((tik_instance,
                     self.cur_process_num*DATA_EIGHT*self.num_class,
                     self.input_dtype),
                    self.output_region_proposal_ub)

        self.size, self.mask, self.ratio = get_params(self.input_dtype)
        self.im_info_ub = tik_instance.Tensor(self.input_dtype,
                                              (FP16_ALIGN_NUM/self.ratio,),
                                              name="im_info_ub", scope=tik.scope_ubuf)

        self.x1_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="x1_ubaddr",
            scope=tik.scope_ubuf)
        self.y1_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="y1_ubaddr",
            scope=tik.scope_ubuf)
        self.x2_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="x2_ubaddr",
            scope=tik.scope_ubuf)
        self.y2_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="y2_ubaddr",
            scope=tik.scope_ubuf)
        self.dx_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="dx_ubaddr",
            scope=tik.scope_ubuf)
        self.dy_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="dy_ubaddr",
            scope=tik.scope_ubuf)
        self.dw_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="dw_ubaddr",
            scope=tik.scope_ubuf)
        self.dh_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="dh_ubaddr",
            scope=tik.scope_ubuf)
        self.ubaddr0 = tik_instance.Tensor(
            self.input_dtype, shape, name="ubaddr0",
            scope=tik.scope_ubuf)
        self.ubaddr1 = tik_instance.Tensor(
            self.input_dtype, shape, name="ubaddr1",
            scope=tik.scope_ubuf)

    def generate_rois(self, input_list, cur_batch_index, output_region_proposal):
        """
        :param input_list:
        :param batchID:
        :param output_region_proposal:
        :return:
        """
        cur_process_num = input_list[0]
        rois_offset = input_list[1]
        prior_offset = input_list[2]
        score_offset = input_list[3]
        score_gm = input_list[4]
        prior_box_gm = input_list[5]
        rois = input_list[6]
        max_rois_num = input_list[7]

        self.tik_instance.data_move(self.im_info_ub,
                                    self.image_info[cur_batch_index, 0],
                                    0, DATA_ONE, DATA_ONE, 0, 0, 0)

        with self.tik_instance.new_scope():
            self.tik_instance.data_move(self.x1_ubaddr[0], rois[rois_offset],
                                        0, cur_process_num//FP16_ALIGN_NUM,
                                        self.ratio, 0, 0)
            self.tik_instance.data_move(self.y1_ubaddr[0],
                                        rois[rois_offset+max_rois_num], 0,
                                        cur_process_num//FP16_ALIGN_NUM,
                                        self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.x2_ubaddr[0], rois[rois_offset+max_rois_num*2], 0,
                cur_process_num//FP16_ALIGN_NUM, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.y2_ubaddr[0], rois[rois_offset+max_rois_num*3], 0,
                cur_process_num//FP16_ALIGN_NUM, self.ratio, 0, 0)

            with self.tik_instance.for_range(1, self.num_class) as class_index:
                self.tik_instance.data_move(
                    self.x1_ubaddr[cur_process_num*class_index], self.x1_ubaddr,
                    0, cur_process_num//FP16_ALIGN_NUM, self.ratio, 0, 0)

                self.tik_instance.data_move(
                    self.y1_ubaddr[cur_process_num*class_index], self.y1_ubaddr,
                    0, cur_process_num//FP16_ALIGN_NUM, self.ratio, 0, 0)

                self.tik_instance.data_move(
                    self.x2_ubaddr[cur_process_num*class_index], self.x2_ubaddr,
                    0, cur_process_num//FP16_ALIGN_NUM, self.ratio, 0, 0)

                self.tik_instance.data_move(
                    self.y2_ubaddr[cur_process_num*class_index], self.y2_ubaddr,
                    0, cur_process_num//FP16_ALIGN_NUM, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.dx_ubaddr[0], prior_box_gm[prior_offset], 0,
                cur_process_num//FP16_ALIGN_NUM*self.num_class, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.dy_ubaddr[0],
                prior_box_gm[prior_offset + max_rois_num*self.num_class],
                0, cur_process_num//FP16_ALIGN_NUM*self.num_class, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.dw_ubaddr[0],
                prior_box_gm[prior_offset + max_rois_num*2*self.num_class],
                0, cur_process_num//FP16_ALIGN_NUM*self.num_class, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.dh_ubaddr[0], prior_box_gm[
                    prior_offset + max_rois_num * 3*self.num_class], 0,
                cur_process_num//FP16_ALIGN_NUM*self.num_class, self.ratio, 0, 0)

            vec_sub((self.tik_instance, self.x2_ubaddr, self.x1_ubaddr,
                     self.input_dtype), self.ubaddr0,
                    cur_process_num*self.num_class)

            vec_sub((self.tik_instance, self.y2_ubaddr, self.y1_ubaddr,
                     self.input_dtype), self.ubaddr1,
                    cur_process_num*self.num_class)

            temp = DATA_ONE
            vec_adds((self.tik_instance, self.ubaddr0, self.input_dtype),
                     self.x2_ubaddr, temp, cur_process_num*self.num_class)

            temp = DATA_ONE
            vec_adds((self.tik_instance, self.ubaddr1, self.input_dtype),
                     self.y2_ubaddr, temp, cur_process_num*self.num_class)

            temp = 0.5
            vec_muls((self.tik_instance, self.x2_ubaddr, self.input_dtype),
                     self.ubaddr0, temp, cur_process_num*self.num_class)

            temp = 0.5
            vec_muls((self.tik_instance, self.y2_ubaddr, self.input_dtype),
                     self.ubaddr1, temp, cur_process_num*self.num_class)

            vec_add((self.tik_instance, self.ubaddr0, self.x1_ubaddr,
                     self.input_dtype), self.x1_ubaddr,
                    cur_process_num*self.num_class)

            vec_add((self.tik_instance, self.ubaddr1, self.y1_ubaddr,
                     self.input_dtype), self.y1_ubaddr,
                    cur_process_num*self.num_class)

            vec_mla((self.tik_instance, self.dx_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.x1_ubaddr,
                    cur_process_num*self.num_class)

            vec_mla((self.tik_instance, self.dy_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.y1_ubaddr,
                    cur_process_num*self.num_class)

            vec_exp((self.tik_instance, self.dw_ubaddr, self.input_dtype),
                    self.dx_ubaddr, cur_process_num*self.num_class)

            vec_exp((self.tik_instance, self.dh_ubaddr, self.input_dtype),
                    self.dy_ubaddr, cur_process_num*self.num_class)

            vec_mul((self.tik_instance, self.dx_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.x2_ubaddr,
                    cur_process_num*self.num_class)

            vec_mul((self.tik_instance, self.dy_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.y2_ubaddr,
                    cur_process_num*self.num_class)

            temp = 0.5
            vec_muls((self.tik_instance, self.x2_ubaddr, self.input_dtype),
                     self.x2_ubaddr, temp, cur_process_num*self.num_class)

            temp = 0.5
            vec_muls((self.tik_instance, self.y2_ubaddr, self.input_dtype),
                     self.y2_ubaddr, temp, cur_process_num*self.num_class)

            vec_sub((self.tik_instance, self.x1_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.dx_ubaddr,
                    cur_process_num*self.num_class)

            vec_sub((self.tik_instance, self.y1_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.dy_ubaddr,
                    cur_process_num*self.num_class)

            vec_add((self.tik_instance, self.x1_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.dw_ubaddr,
                    cur_process_num*self.num_class)

            vec_add((self.tik_instance, self.y1_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.dh_ubaddr,
                    cur_process_num*self.num_class)
            self.tik_instance.vadds(FP16_ALIGN_NUM//self.ratio, self.im_info_ub,
                                    self.im_info_ub, -1, DATA_ONE, DATA_ONE, DATA_ONE, 0, 0)

            #clip

            im_scalar = self.tik_instance.Scalar(dtype=self.input_dtype)
            im_scalar.set_as(self.im_info_ub[1])

            vec_dup((self.tik_instance, cur_process_num*self.num_class,
                     self.input_dtype), self.ubaddr0, im_scalar)
            im_scalar.set_as(self.im_info_ub[0])
            vec_dup((self.tik_instance, cur_process_num*self.num_class,
                     self.input_dtype), self.ubaddr1, im_scalar)

            vec_min((self.tik_instance, self.dx_ubaddr, self.ubaddr0,
                     self.input_dtype), self.dx_ubaddr,
                    cur_process_num*self.num_class)

            vec_min((self.tik_instance, self.dy_ubaddr, self.ubaddr1,
                     self.input_dtype), self.dy_ubaddr,
                    cur_process_num*self.num_class)

            vec_min((self.tik_instance, self.dw_ubaddr, self.ubaddr0,
                     self.input_dtype), self.dw_ubaddr,
                    cur_process_num*self.num_class)

            vec_min((self.tik_instance, self.dh_ubaddr, self.ubaddr1,
                     self.input_dtype), self.dh_ubaddr,
                    cur_process_num*self.num_class)

            vec_relu((self.tik_instance, self.dx_ubaddr, self.input_dtype),
                     self.x1_ubaddr, cur_process_num*self.num_class)

            vec_relu((self.tik_instance, self.dy_ubaddr, self.input_dtype),
                     self.y1_ubaddr, cur_process_num*self.num_class)

            vec_relu((self.tik_instance, self.dw_ubaddr, self.input_dtype),
                     self.x2_ubaddr, cur_process_num*self.num_class)

            vec_relu((self.tik_instance, self.dh_ubaddr, self.input_dtype),
                     self.y2_ubaddr, cur_process_num*self.num_class)
            self.tik_instance.data_move(
                self.ubaddr0, score_gm[score_offset], 0,
                cur_process_num//FP16_ALIGN_NUM*self.num_class, self.ratio, 0, 0)

            vec_concat((self.tik_instance, self.x1_ubaddr, self.input_dtype),
                       self.output_region_proposal_ub,
                       cur_process_num*self.num_class, 0)
            vec_concat((self.tik_instance, self.y1_ubaddr, self.input_dtype),
                       self.output_region_proposal_ub,
                       cur_process_num*self.num_class, 1)
            vec_concat((self.tik_instance, self.x2_ubaddr, self.input_dtype),
                       self.output_region_proposal_ub,
                       cur_process_num*self.num_class, 2)
            vec_concat((self.tik_instance, self.y2_ubaddr, self.input_dtype),
                       self.output_region_proposal_ub,
                       cur_process_num*self.num_class, 3)
            vec_concat((self.tik_instance, self.ubaddr0, self.input_dtype),
                       self.output_region_proposal_ub,
                       cur_process_num*self.num_class, 4)

            self.tik_instance.data_move(
                output_region_proposal[score_offset*DATA_EIGHT],
                self.output_region_proposal_ub, 0,
                cur_process_num*DATA_EIGHT//FP16_ALIGN_NUM*self.num_class,
                self.ratio, 0, 0)

    def tiling_generate_rois(self, input_list, cur_batch_index, output_region_proposal):
        """
        :param input_list:
        :param batchID:
        :param output_region_proposal:
        :return:
        """
        cur_process_num = input_list[0]
        rois_offset = input_list[1]
        prior_offset = input_list[2]
        score_offset = input_list[3]
        score_gm = input_list[4]
        prior_box_gm = input_list[5]
        rois = input_list[6]
        max_rois_num = input_list[7]

        self.tik_instance.data_move(self.im_info_ub,
                                    self.image_info[cur_batch_index, 0],
                                    0, DATA_ONE, DATA_ONE, 0, 0, 0)

        with self.tik_instance.new_scope():
            self.tik_instance.data_move(self.x1_ubaddr[0], rois[rois_offset],
                                        0, cur_process_num//FP16_ALIGN_NUM,
                                        self.ratio, 0, 0)
            self.tik_instance.data_move(self.y1_ubaddr[0],
                                        rois[rois_offset+max_rois_num], 0,
                                        cur_process_num//FP16_ALIGN_NUM,
                                        self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.x2_ubaddr[0], rois[rois_offset+max_rois_num*2], 0,
                cur_process_num//FP16_ALIGN_NUM, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.y2_ubaddr[0], rois[rois_offset+max_rois_num*3], 0,
                cur_process_num//FP16_ALIGN_NUM, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.dx_ubaddr[0], prior_box_gm[prior_offset], 0,
                cur_process_num//FP16_ALIGN_NUM, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.dy_ubaddr[0], prior_box_gm[prior_offset + max_rois_num*self.num_class],
                0, cur_process_num//FP16_ALIGN_NUM, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.dw_ubaddr[0], prior_box_gm[prior_offset + max_rois_num*2*self.num_class],
                0, cur_process_num//FP16_ALIGN_NUM, self.ratio, 0, 0)

            self.tik_instance.data_move(self.dh_ubaddr[0],
                                        prior_box_gm[prior_offset + max_rois_num
                                                     * 3*self.num_class], 0,
                                        cur_process_num//FP16_ALIGN_NUM,
                                        self.ratio, 0, 0)

            vec_sub((self.tik_instance, self.x2_ubaddr, self.x1_ubaddr,
                     self.input_dtype), self.ubaddr0, cur_process_num)

            vec_sub((self.tik_instance, self.y2_ubaddr, self.y1_ubaddr,
                     self.input_dtype), self.ubaddr1, cur_process_num)

            temp = DATA_ONE
            vec_adds((self.tik_instance, self.ubaddr0, self.input_dtype),
                     self.x2_ubaddr, temp, cur_process_num)

            temp = DATA_ONE
            vec_adds((self.tik_instance, self.ubaddr1, self.input_dtype),
                     self.y2_ubaddr, temp, cur_process_num)

            temp = 0.5
            vec_muls((self.tik_instance, self.x2_ubaddr, self.input_dtype),
                     self.ubaddr0, temp, cur_process_num)

            temp = 0.5
            vec_muls((self.tik_instance, self.y2_ubaddr, self.input_dtype),
                     self.ubaddr1, temp, cur_process_num)

            vec_add((self.tik_instance, self.ubaddr0, self.x1_ubaddr,
                     self.input_dtype), self.x1_ubaddr, cur_process_num)

            vec_add((self.tik_instance, self.ubaddr1, self.y1_ubaddr,
                     self.input_dtype), self.y1_ubaddr, cur_process_num)

            vec_mla((self.tik_instance, self.dx_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.x1_ubaddr, cur_process_num)

            vec_mla((self.tik_instance, self.dy_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.y1_ubaddr, cur_process_num)

            vec_exp((self.tik_instance, self.dw_ubaddr, self.input_dtype),
                    self.dx_ubaddr, cur_process_num)

            vec_exp((self.tik_instance, self.dh_ubaddr, self.input_dtype),
                    self.dy_ubaddr, cur_process_num)

            vec_mul((self.tik_instance, self.dx_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.x2_ubaddr, cur_process_num)

            vec_mul((self.tik_instance, self.dy_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.y2_ubaddr, cur_process_num)

            temp = 0.5
            vec_muls((self.tik_instance, self.x2_ubaddr, self.input_dtype),
                     self.x2_ubaddr, temp, cur_process_num)

            temp = 0.5
            vec_muls((self.tik_instance, self.y2_ubaddr, self.input_dtype),
                     self.y2_ubaddr, temp, cur_process_num)

            vec_sub((self.tik_instance, self.x1_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.dx_ubaddr, cur_process_num)

            vec_sub((self.tik_instance, self.y1_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.dy_ubaddr, cur_process_num)

            vec_add((self.tik_instance, self.x1_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.dw_ubaddr, cur_process_num)

            vec_add((self.tik_instance, self.y1_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.dh_ubaddr, cur_process_num)

            self.tik_instance.vadds(16//self.ratio, self.im_info_ub,
                                    self.im_info_ub, -1, DATA_ONE, DATA_ONE, DATA_ONE, 0, 0)

            #clip

            im_scalar = self.tik_instance.Scalar(dtype=self.input_dtype)
            im_scalar.set_as(self.im_info_ub[1])

            vec_dup((self.tik_instance, cur_process_num,
                     self.input_dtype), self.ubaddr0, im_scalar)
            im_scalar.set_as(self.im_info_ub[0])
            vec_dup((self.tik_instance, cur_process_num,
                     self.input_dtype), self.ubaddr1, im_scalar)

            vec_min((self.tik_instance, self.dx_ubaddr, self.ubaddr0,
                     self.input_dtype), self.dx_ubaddr, cur_process_num)

            vec_min((self.tik_instance, self.dy_ubaddr, self.ubaddr1,
                     self.input_dtype), self.dy_ubaddr, cur_process_num)

            vec_min((self.tik_instance, self.dw_ubaddr, self.ubaddr0,
                     self.input_dtype), self.dw_ubaddr, cur_process_num)

            vec_min((self.tik_instance, self.dh_ubaddr, self.ubaddr1,
                     self.input_dtype), self.dh_ubaddr, cur_process_num)

            vec_relu((self.tik_instance, self.dx_ubaddr, self.input_dtype),
                     self.x1_ubaddr, cur_process_num)

            vec_relu((self.tik_instance, self.dy_ubaddr, self.input_dtype),
                     self.y1_ubaddr, cur_process_num)

            vec_relu((self.tik_instance, self.dw_ubaddr, self.input_dtype),
                     self.x2_ubaddr, cur_process_num)

            vec_relu((self.tik_instance, self.dh_ubaddr, self.input_dtype),
                     self.y2_ubaddr, cur_process_num)

            self.tik_instance.data_move(self.ubaddr0, score_gm[score_offset],
                                        0, cur_process_num//FP16_ALIGN_NUM,
                                        self.ratio, 0, 0)

            vec_concat((self.tik_instance, self.x1_ubaddr, self.input_dtype),
                       self.output_region_proposal_ub, cur_process_num, 0)
            vec_concat((self.tik_instance, self.y1_ubaddr, self.input_dtype),
                       self.output_region_proposal_ub, cur_process_num, 1)
            vec_concat((self.tik_instance, self.x2_ubaddr, self.input_dtype),
                       self.output_region_proposal_ub, cur_process_num, 2)
            vec_concat((self.tik_instance, self.y2_ubaddr, self.input_dtype),
                       self.output_region_proposal_ub, cur_process_num, 3)
            vec_concat((self.tik_instance, self.ubaddr0, self.input_dtype),
                       self.output_region_proposal_ub, cur_process_num, 4)
            # print("score_offset:",score_offset)
            self.tik_instance.data_move(
                output_region_proposal[score_offset*constant.DATA_SIZE_EIGHT],
                self.output_region_proposal_ub, 0,
                cur_process_num*constant.DATA_SIZE_EIGHT//FP16_ALIGN_NUM,
                self.ratio, 0, 0)


class OneCoreProcess:
    """
    One Core Process
    """
    def __init__(self, input_data):
        """
        :param input_data:
        """
        self.tik_instance = input_data[0]
        self.input_dtype = input_data[1]
        self.input_shape = input_data[2]
        self.size = input_data[3]
        self.batch_rois = input_data[4]
        self.image_info = input_data[5]
        self.num_classes = input_data[6]
        self.max_rois_num = input_data[7]

        self.total_num = input_data[8]
        self.device_cor_num = input_data[9]
        self.batch_factor = input_data[10]
        self.block_id = input_data[11]
        self.ub_size = input_data[12]
        self.total_num = self.input_shape[0]

        cal_var_num = 18
        self.total_size = cal_var_num*self.max_rois_num*self.size*self.num_classes
        self.one_batch_one_class_size = cal_var_num*self.max_rois_num*self.size
        # one time need space
        # self.reserved_ub_size = cal_var_num*16*self.size*self.num_classes
        self.reserved_ub_size = cal_var_num*FP16_ALIGN_NUM*self.size

    def get_offset(self, batch_index):
        """
        :param input_offset:
        :return:
        """
        rois_offset = batch_index*self.max_rois_num*5+self.max_rois_num
        prior_offset = self.num_classes*batch_index*self.max_rois_num*4
        score_offset = self.num_classes*batch_index*self.max_rois_num
        return rois_offset, prior_offset, score_offset

    def get_tiling_branch1_offset(self, batch_index, class_index):
        """
        :param batch_index:
        :param class_index:
        :return:
        """
        rois_offset = batch_index * self.max_rois_num * 5 + self.max_rois_num
        prior_offset = class_index*self.max_rois_num + \
                       batch_index*self.max_rois_num*4*self.num_classes
        score_offset = class_index*self.max_rois_num + \
                       batch_index*self.max_rois_num*self.num_classes
        return rois_offset, prior_offset, score_offset

    def get_tiling_branch2_offset(self, input_list):
        """
        :param input_list:
        :return:
        """
        batch_index = input_list[0]
        class_index = input_list[1]
        tiling_process = input_list[2]
        tiling_loop = input_list[3]
        rois_offset = batch_index*self.max_rois_num*5 +\
                      self.max_rois_num+tiling_process*FP16_ALIGN_NUM*tiling_loop
        prior_offset = self.num_classes*batch_index * \
                       self.max_rois_num*4+class_index*self.max_rois_num*4 + \
                       tiling_process*FP16_ALIGN_NUM*tiling_loop
        score_offset = self.num_classes*batch_index * \
                       self.max_rois_num+class_index*self.max_rois_num+tiling_process * \
                       FP16_ALIGN_NUM*tiling_loop
        return rois_offset, prior_offset, score_offset

    def get_tiling_tail_offset(self, input_list):
        """
        :param input_list:
        :return:
        """
        batch_index = input_list[0]
        class_index = input_list[1]
        tiling_process = input_list[2]
        tiling_num = input_list[3]
        rois_offset = batch_index*self.max_rois_num*5+self.max_rois_num + \
                      tiling_process*FP16_ALIGN_NUM*tiling_num
        prior_offset = self.num_classes*batch_index*self.max_rois_num*4 + \
                       class_index*self.max_rois_num*4+tiling_process*FP16_ALIGN_NUM*tiling_num
        score_offset = self.num_classes*batch_index*self.max_rois_num + \
                       class_index*self.max_rois_num+tiling_process*FP16_ALIGN_NUM*tiling_num
        return rois_offset, prior_offset, score_offset

    def one_core_process_decode_rois(self, input_list, output_box):
        """
        :param block_id:
        :param class_index:
        :param batch_index:
        :param rois:
        :param actual_rois_num:
        :param prior_box_gm:
        :param score_gm:
        :param output_box:
        :return:
        """

        batch_index = input_list[0]
        rois = input_list[1]
        input_tensor = input_list[2]
        self.actual_rois_num_effect = input_tensor[0]
        prior_box_gm = input_list[3]
        score_gm = input_list[4]
        cur_batch_index = batch_index
        if self.actual_rois_num_effect:
            actual_rois_num = input_tensor[1]
            actual_rois_num_ub = self.tik_instance.Tensor(
                "int32", (self.batch_rois, DATA_EIGHT),
                name="actual_rois_num_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(actual_rois_num_ub, actual_rois_num, 0,
                                        self.batch_rois, DATA_ONE, 0, 0)

            cur_batch_num = self.tik_instance.Scalar("int32")
            cur_batch_num.set_as(
                actual_rois_num_ub[cur_batch_index*DATA_EIGHT])
        else:
            cur_batch_num = self.max_rois_num

        if self.ub_size >= self.total_size:
            rois_offset, prior_offset, score_offset = \
                self.get_offset(batch_index)
            decode_rois_object = DecodeRois(
                self.tik_instance, (self.max_rois_num, self.input_dtype,
                                    self.image_info, self.num_classes), False)

            decode_rois_object.generate_rois(
                (self.max_rois_num, rois_offset, prior_offset,
                 score_offset, score_gm, prior_box_gm,
                 rois, self.max_rois_num), cur_batch_index,
                output_box)

        else:
            with self.tik_instance.for_range(DATA_ONE, self.num_classes) \
                    as class_index:
                if self.ub_size >= self.one_batch_one_class_size:
                    rois_offset, prior_offset, score_offset = \
                        self.get_tiling_branch1_offset(batch_index, class_index)
                    decode_rois_object = DecodeRois(
                        self.tik_instance, (self.max_rois_num, self.input_dtype,
                                            self.image_info, self.num_classes), True)
                    decode_rois_object.tiling_generate_rois(
                        (self.max_rois_num, rois_offset, prior_offset,
                         score_offset, score_gm, prior_box_gm,
                         rois, self.max_rois_num), cur_batch_index,
                        output_box)
                else:
                    reserved_space = self.batch_rois*constant.DATA_SIZE_EIGHT*4
                    tiling_process = \
                        (self.ub_size-reserved_space) // self.reserved_ub_size
                    tiling_num = self.max_rois_num//(tiling_process*FP16_ALIGN_NUM)

                    tiling_tail = self.max_rois_num - \
                                  tiling_num*tiling_process*FP16_ALIGN_NUM

                    with self.tik_instance.for_range(0, tiling_num) as tiling_loop:
                        with self.tik_instance.if_scope(cur_batch_num >= tiling_process
                                                        * FP16_ALIGN_NUM * tiling_loop):

                            rois_offset, prior_offset, score_offset = \
                                self.get_tiling_branch2_offset((batch_index, class_index,
                                                                tiling_process, tiling_loop))
                            decode_rois_object = DecodeRois(
                                self.tik_instance, (tiling_process*FP16_ALIGN_NUM,
                                                    self.input_dtype, self.image_info,
                                                    self.num_classes), True)
                            decode_rois_object.tiling_generate_rois(
                                (tiling_process*FP16_ALIGN_NUM, rois_offset,
                                 prior_offset, score_offset, score_gm,
                                 prior_box_gm, rois, self.max_rois_num),
                                cur_batch_index, output_box)
                    with self.tik_instance.if_scope(tiling_tail > 0):
                        with self.tik_instance.if_scope(cur_batch_num >= tiling_process
                                                        * FP16_ALIGN_NUM * tiling_num):
                            rois_offset, prior_offset, score_offset = \
                                self.get_tiling_tail_offset(
                                    (batch_index, class_index, tiling_process,
                                     tiling_num))
                            decode_rois_object = DecodeRois(
                                self.tik_instance, (tiling_tail, self.input_dtype,
                                                    self.image_info, self.num_classes), True)
                            decode_rois_object.tiling_generate_rois((tiling_tail,
                                                                     rois_offset, prior_offset,
                                                                     score_offset,
                                                                     score_gm, prior_box_gm,
                                                                     rois, self.max_rois_num),
                                                                    cur_batch_index,
                                                                    output_box)


class PreProcess:
    """
    PreProcess
    """
    def __init__(self, input_data, input_tensor):
        """
        :param input_data:
        """
        self.tik_instance = input_data[0]
        self.max_rois_num = input_data[1]
        self.num_classes = input_data[2]
        self.batch_rois = input_data[3]
        self.input_shape = input_data[4]
        self.input_dtype = input_data[5]

        self.size, self.mask, self.ratio = get_params(self.input_dtype)
        self.actual_rois_num_effect = input_tensor[0]
        if self.actual_rois_num_effect:
            actual_rois_num = input_tensor[1]
            self.actual_rois_num_ub = self.tik_instance.Tensor(
                "int32", (self.batch_rois, DATA_EIGHT),
                name="actual_rois_num_ub", scope=tik.scope_ubuf)
            rois_act_dup_times = self.batch_rois//MAX_REPEAT_TIME
            rois_act_tail = self.batch_rois - rois_act_dup_times*MAX_REPEAT_TIME
            with self.tik_instance.for_range(0, rois_act_dup_times) \
                    as rois_act_loop:
                self.tik_instance.vector_dup(
                    DATA_EIGHT,
                    self.actual_rois_num_ub[MAX_REPEAT_TIME *
                                            rois_act_loop*DATA_EIGHT],
                    0, MAX_REPEAT_TIME, 0, DATA_ONE)
            if rois_act_tail != 0:
                self.tik_instance.vector_dup(
                    DATA_EIGHT,
                    self.actual_rois_num_ub[
                        MAX_REPEAT_TIME*DATA_EIGHT *
                        rois_act_dup_times], 0, rois_act_tail, 0, DATA_ONE)

            self.tik_instance.data_move(self.actual_rois_num_ub, actual_rois_num, 0,
                                        self.batch_rois, DATA_ONE, 0, 0)
            self.actual_sum_num = self.tik_instance.Tensor(
                "int32", (self.batch_rois, DATA_EIGHT),
                name="actual_sum_num", scope=tik.scope_ubuf)

            vec_dup((self.tik_instance,
                     self.batch_rois*DATA_EIGHT, "float32"),
                    self.actual_sum_num)

            with self.tik_instance.for_range(DATA_ONE, self.batch_rois) as batch_index:
                self.tik_instance.vadd(BLOCK_SIZE//4,
                                       self.actual_sum_num[
                                           batch_index*DATA_EIGHT],
                                       self.actual_sum_num[
                                           (batch_index-DATA_ONE) *
                                           DATA_EIGHT],
                                       self.actual_rois_num_ub[
                                           (batch_index-DATA_ONE) *
                                           DATA_EIGHT],
                                       DATA_ONE, DATA_ONE, DATA_ONE,
                                       DATA_ONE, DATA_ONE,
                                       DATA_ONE, DATA_ONE,)

    def trans(self, src_ub, dst_ub, length):
        """
        :param src_ub:
        :param dst_ub:
        :param length:
        :return:
        """
        if self.input_dtype == "float16":
            vnch_loop_times = ((length*FP16_ALIGN_NUM) //
                               VECTOR_BLOCK_SIZE)//MAX_REPEAT_TIME
            tail_loop_times = ((length*FP16_ALIGN_NUM) //
                               VECTOR_BLOCK_SIZE) % MAX_REPEAT_TIME
            with self.tik_instance.for_range(0, vnch_loop_times) as vnch_loop:
                src_list = \
                    [src_ub[FP16_ALIGN_NUM*i+vnch_loop *
                            MAX_REPEAT_TIME*VECTOR_BLOCK_SIZE]
                     for i in range(FP16_ALIGN_NUM)]
                dst_list = [dst_ub[FP16_ALIGN_NUM*i+vnch_loop *
                                   MAX_REPEAT_TIME*VECTOR_BLOCK_SIZE]
                            for i in range(FP16_ALIGN_NUM)]
                self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                            MAX_REPEAT_TIME, FP16_ALIGN_NUM,
                                            FP16_ALIGN_NUM)
            src_list = [src_ub[FP16_ALIGN_NUM*i+vnch_loop_times *
                               MAX_REPEAT_TIME*VECTOR_BLOCK_SIZE]
                        for i in range(FP16_ALIGN_NUM)]
            dst_list = [dst_ub[FP16_ALIGN_NUM*i+vnch_loop_times *
                               MAX_REPEAT_TIME*VECTOR_BLOCK_SIZE]
                        for i in range(FP16_ALIGN_NUM)]
            if tail_loop_times == DATA_ONE:
                self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                            tail_loop_times, 0, 0)
            else:
                self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                            tail_loop_times, FP16_ALIGN_NUM,
                                            FP16_ALIGN_NUM)
        elif self.input_dtype == "float32":
            src_list = [src_ub[i*FP16_ALIGN_NUM] for i in range(FP16_ALIGN_NUM)]
            dst_list = [dst_ub[i//2*FP16_ALIGN_NUM +
                               (i % 2) *
                               DATA_EIGHT]
                        for i in range(FP16_ALIGN_NUM)]
            if (length*FP16_ALIGN_NUM) // VECTOR_BLOCK_SIZE == DATA_ONE:
                self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                            (length*FP16_ALIGN_NUM) //
                                            VECTOR_BLOCK_SIZE,
                                            0, 0)
            else:
                self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                            (length*FP16_ALIGN_NUM) //
                                            VECTOR_BLOCK_SIZE,
                                            BLOCK_SIZE,
                                            BLOCK_SIZE)
            src_list = [src_ub[i*FP16_ALIGN_NUM+DATA_EIGHT]
                        for i in range(FP16_ALIGN_NUM)]
            dst_list = [dst_ub[i//2*FP16_ALIGN_NUM +
                               (i % 2) *
                               DATA_EIGHT +
                               DATA_EIGHT*FP16_ALIGN_NUM]
                        for i in range(FP16_ALIGN_NUM)]
            self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                        (length*FP16_ALIGN_NUM) //
                                        VECTOR_BLOCK_SIZE,
                                        BLOCK_SIZE,
                                        BLOCK_SIZE)

    def no_tiling_trans_score(self, input_list, score, score_gm):
        shape = input_list[0]
        sum_addr = input_list[1]
        cur_batch_index = input_list[2]
        cur_batch_num = input_list[3]
        align_num_times = shape[0]
        score_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                            name="score_ub",
                                            scope=tik.scope_ubuf)
        new_score_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                                name="new_score_ub",
                                                scope=tik.scope_ubuf)

        vec_dup((self.tik_instance, align_num_times * FP16_ALIGN_NUM * self.max_rois_num,
                 self.input_dtype), new_score_ub)

        self.tik_instance.data_move(
            new_score_ub[0],
            score[sum_addr*align_num_times*FP16_ALIGN_NUM],
            0, cur_batch_num*align_num_times, self.ratio,
            0, 0)

        vec_dup((self.tik_instance, align_num_times * FP16_ALIGN_NUM * self.max_rois_num,
                 self.input_dtype), score_ub)

        with self.tik_instance.for_range(0, align_num_times) as loop:
            self.tik_instance.data_move(
                score_ub[loop*cur_batch_num*FP16_ALIGN_NUM],
                new_score_ub[loop*FP16_ALIGN_NUM], 0, cur_batch_num, self.ratio,
                (align_num_times - DATA_ONE) * self.ratio, 0)

        self.trans(score_ub, new_score_ub,
                   score_ub.shape[0]*score_ub.shape[2])

        with self.tik_instance.for_range(0, self.num_classes // FP16_ALIGN_NUM) as loop:
            with self.tik_instance.for_range(0, FP16_ALIGN_NUM) as inner_loop:
                self.tik_instance.data_move(
                    score_ub[loop*self.max_rois_num*FP16_ALIGN_NUM+
                             inner_loop*self.max_rois_num],
                    new_score_ub[inner_loop*FP16_ALIGN_NUM+
                                 loop*self.max_rois_num*FP16_ALIGN_NUM], 0,
                    self.max_rois_num//FP16_ALIGN_NUM, self.ratio,
                    15*self.ratio, 0)

        loop_times = self.num_classes%FP16_ALIGN_NUM
        with self.tik_instance.for_range(0, loop_times) as inner_loop:
            self.tik_instance.data_move(
                score_ub[(self.num_classes//FP16_ALIGN_NUM)*self.max_rois_num*FP16_ALIGN_NUM+
                         inner_loop*self.max_rois_num],
                new_score_ub[inner_loop*FP16_ALIGN_NUM+
                             (self.num_classes//FP16_ALIGN_NUM)*self.max_rois_num*FP16_ALIGN_NUM],
                0, self.max_rois_num//FP16_ALIGN_NUM, self.ratio,
                15*self.ratio, 0)
        self.tik_instance.data_move(
            score_gm[cur_batch_index*self.max_rois_num*self.num_classes],
            score_ub, 0,
            self.max_rois_num//FP16_ALIGN_NUM*self.num_classes, self.ratio,
            0, 0)

    def tiling_trans_score_branch1(self, input_list, score, score_gm):
        shape = input_list[0]
        sum_addr = input_list[1]
        class_loop = input_list[2]
        cur_batch_index = input_list[3]
        cur_batch_num = input_list[4]

        score_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                            name="score_ub",
                                            scope=tik.scope_ubuf)
        new_score_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                                name="new_score_ub",
                                                scope=tik.scope_ubuf)

        vec_dup((self.tik_instance, shape[0]*shape[1],
                 self.input_dtype), score_ub)
        vec_dup((self.tik_instance, shape[0]*shape[1],
                 self.input_dtype), new_score_ub)

        self.tik_instance.data_move(
            score_ub[0],
            score[sum_addr*((self.num_classes+TO_ALIGN_NUM) //
                            FP16_ALIGN_NUM)*FP16_ALIGN_NUM +
                  ((class_loop+TO_ALIGN_NUM)//FP16_ALIGN_NUM) *
                  FP16_ALIGN_NUM],
            0, cur_batch_num, self.ratio,
            ((self.num_classes+TO_ALIGN_NUM) //
             FP16_ALIGN_NUM-DATA_ONE)*self.ratio, 0)
        self.trans(score_ub, new_score_ub, shape[0])
        vec_dup((self.tik_instance, shape[0]*shape[1],
                 self.input_dtype), score_ub)
        with self.tik_instance.for_range(0, FP16_ALIGN_NUM) as loop:
            with self.tik_instance.if_scope(
                    (class_loop*FP16_ALIGN_NUM+loop) <
                    self.num_classes):
                self.tik_instance.data_move(
                    score_ub[loop*self.max_rois_num],
                    new_score_ub[loop*FP16_ALIGN_NUM], 0,
                    self.max_rois_num//FP16_ALIGN_NUM, self.ratio,
                    15*self.ratio, 0)
        with self.tik_instance.for_range(0, FP16_ALIGN_NUM) as loop:
            with self.tik_instance.if_scope(
                    class_loop*FP16_ALIGN_NUM+loop < self.num_classes):
                with self.tik_instance.if_scope(
                        (class_loop*FP16_ALIGN_NUM+loop) <
                        self.num_classes):
                    score_gm_offset = \
                        ((class_loop+TO_ALIGN_NUM)//FP16_ALIGN_NUM) * \
                        FP16_ALIGN_NUM * \
                        self.max_rois_num+(class_loop -
                                           (class_loop+TO_ALIGN_NUM) //
                                           FP16_ALIGN_NUM) * \
                        self.max_rois_num + \
                        cur_batch_index*self.max_rois_num*self.num_classes+loop * \
                        self.max_rois_num

                    self.tik_instance.data_move(
                        score_gm[score_gm_offset],
                        score_ub[loop*self.max_rois_num], 0,
                        self.max_rois_num//FP16_ALIGN_NUM, self.ratio,
                        0, 0)

    def tiling_trans_score_branch2(self, input_list, score, score_gm):
        shape = input_list[0]
        sum_addr = input_list[1]
        class_loop = input_list[2]
        cur_batch_index = input_list[3]
        cur_batch_num = input_list[4]
        one_tiling_process = input_list[5]
        score_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                            name="score_ub",
                                            scope=tik.scope_ubuf)
        new_score_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                                name="new_score_ub",
                                                scope=tik.scope_ubuf)
        vec_dup((self.tik_instance, shape[0]*shape[1],
                 self.input_dtype), score_ub)
        vec_dup((self.tik_instance, shape[0]*shape[1],
                 self.input_dtype), new_score_ub)

        one_batch_loop_time = cur_batch_num // \
                              (one_tiling_process//FP16_ALIGN_NUM)
        with self.tik_instance.for_range(0, one_batch_loop_time) \
                as inner_batch_loop:
            with self.tik_instance.if_scope(
                    cur_batch_num >= (inner_batch_loop *
                                      (one_tiling_process // FP16_ALIGN_NUM))):
                vec_dup((self.tik_instance, shape[0]*shape[1],
                         self.input_dtype), score_ub)
                vec_dup((self.tik_instance, shape[0]*shape[1],
                         self.input_dtype), new_score_ub)
                self.tik_instance.data_move(
                    score_ub,
                    score[sum_addr+inner_batch_loop *
                          (one_tiling_process//FP16_ALIGN_NUM),
                          class_loop, 0, 0, 0],
                    0, one_tiling_process//FP16_ALIGN_NUM, self.ratio,
                    ((self.num_classes+TO_ALIGN_NUM)//FP16_ALIGN_NUM-DATA_ONE)*
                    self.ratio, 0)
                self.trans(score_ub, new_score_ub, shape[0])
                vec_dup((self.tik_instance, shape[0]*shape[1],
                         self.input_dtype), score_ub)
                with self.tik_instance.for_range(0, FP16_ALIGN_NUM) as loop:
                    self.tik_instance.data_move(
                        score_ub[loop*(one_tiling_process//FP16_ALIGN_NUM)],
                        new_score_ub[loop*FP16_ALIGN_NUM], 0,
                        (one_tiling_process//FP16_ALIGN_NUM)//FP16_ALIGN_NUM,
                        self.ratio, 15*self.ratio, 0)
                with self.tik_instance.for_range(0, FP16_ALIGN_NUM) as loop:
                    with self.tik_instance.if_scope(
                            class_loop*FP16_ALIGN_NUM+loop < self.num_classes):
                        self.tik_instance.data_move(
                            score_gm[cur_batch_index, class_loop*FP16_ALIGN_NUM +
                                     loop,
                                     inner_batch_loop *
                                     ((one_tiling_process//FP16_ALIGN_NUM) //
                                      FP16_ALIGN_NUM),
                                     0, 0],
                            score_ub[loop*(one_tiling_process//FP16_ALIGN_NUM)],
                            0, (one_tiling_process//FP16_ALIGN_NUM) //
                            FP16_ALIGN_NUM, self.ratio,
                            0, 0)
        with self.tik_instance.if_scope(
                cur_batch_num-one_batch_loop_time*(one_tiling_process //
                                                   FP16_ALIGN_NUM) > 0):
            shape = [one_tiling_process//FP16_ALIGN_NUM, FP16_ALIGN_NUM]
            self.tik_instance.data_move(
                score_ub[0],
                score[sum_addr+one_batch_loop_time *
                      (one_tiling_process//FP16_ALIGN_NUM),
                      class_loop, 0, 0, 0],
                0, cur_batch_num-one_batch_loop_time*(one_tiling_process //
                                                      FP16_ALIGN_NUM),
                self.ratio,
                ((self.num_classes+TO_ALIGN_NUM)//FP16_ALIGN_NUM-DATA_ONE)*self.ratio,
                0)
            self.trans(score_ub, new_score_ub, shape[0])
            vec_dup((self.tik_instance, shape[0]*shape[1],
                     self.input_dtype), score_ub)
            with self.tik_instance.for_range(0, FP16_ALIGN_NUM) as loop:
                self.tik_instance.data_move(
                    score_ub[loop*(one_tiling_process//FP16_ALIGN_NUM)],
                    new_score_ub[loop*FP16_ALIGN_NUM], 0,
                    one_tiling_process//(FP16_ALIGN_NUM*FP16_ALIGN_NUM),
                    self.ratio, 15*self.ratio, 0)
            with self.tik_instance.for_range(0, FP16_ALIGN_NUM) as loop:
                with self.tik_instance.if_scope(
                        class_loop*FP16_ALIGN_NUM+loop < self.num_classes):
                    self.tik_instance.data_move(
                        score_gm[cur_batch_index,
                                 class_loop*FP16_ALIGN_NUM+loop,
                                 one_batch_loop_time *
                                 (one_tiling_process //
                                  (FP16_ALIGN_NUM*FP16_ALIGN_NUM)), 0, 0],
                        score_ub[loop*(one_tiling_process//FP16_ALIGN_NUM)],
                        0, (cur_batch_num-one_batch_loop_time *
                            (one_tiling_process//FP16_ALIGN_NUM) +
                            TO_ALIGN_NUM)//FP16_ALIGN_NUM, self.ratio,
                        0, 0)

    def trans_score(self, score, score_gm, cur_batch_index):
        """
        :param score:
        :param score_gm:
        :param cur_batch_index:
        :return:
        """
        align_num_times = (self.num_classes+15) // FP16_ALIGN_NUM
        ub_size = get_ub_size()

        one_batch_size = self.max_rois_num * FP16_ALIGN_NUM * self.size *\
                         align_num_times * 2 + self.batch_rois * DATA_EIGHT * 4 * 2

        if self.max_rois_num*self.batch_rois != score.shape[0]:
            if self.actual_rois_num_effect == False:
                raise RuntimeError("the input tensor shape does not match!")
            sum_addr = self.tik_instance.Scalar("int32")
            sum_addr.set_as(0)
            cur_batch_num = self.tik_instance.Scalar("int32")
            cur_batch_num.set_as(0)
            sum_addr.set_as(self.actual_sum_num[
                                cur_batch_index*DATA_EIGHT])
            cur_batch_num.set_as(
                self.actual_rois_num_ub[
                    cur_batch_index*DATA_EIGHT])
        else:
            cur_batch_num = self.max_rois_num
            sum_addr = cur_batch_index*self.max_rois_num

        if one_batch_size < ub_size:
            shape = [align_num_times, FP16_ALIGN_NUM, self.max_rois_num]
            self.no_tiling_trans_score((shape, sum_addr,
                                        cur_batch_index, cur_batch_num),
                                       score, score_gm)
        else:
            num_piece_of_space = \
                ((ub_size-self.batch_rois*DATA_EIGHT*4*2) //
                 2) // ((FP16_ALIGN_NUM*FP16_ALIGN_NUM)*self.size)
            one_tiling_process = num_piece_of_space*FP16_ALIGN_NUM*FP16_ALIGN_NUM
            if one_tiling_process//FP16_ALIGN_NUM < self.max_rois_num:
                shape = [one_tiling_process//FP16_ALIGN_NUM, FP16_ALIGN_NUM]
            else:
                shape = [self.max_rois_num, FP16_ALIGN_NUM]
            with self.tik_instance.for_range(0,
                                             (self.num_classes+FP16_ALIGN_NUM) //
                                             FP16_ALIGN_NUM) as class_loop:
                if one_tiling_process >= self.max_rois_num*FP16_ALIGN_NUM:
                    self.tiling_trans_score_branch1((shape, sum_addr, class_loop,
                                                     cur_batch_index, cur_batch_num),
                                                    score, score_gm)
                else:
                    self.tiling_trans_score_branch2((shape, sum_addr, class_loop,
                                                     cur_batch_index, cur_batch_num,
                                                     one_tiling_process), score, score_gm)

    def prior_ub_move(self, loop, length, dst_ub, src_ub):
        """
        :param loop:
        :param length:
        :param dst_ub:
        :param src_ub:
        :return:
        """
        self.tik_instance.data_move(dst_ub[loop*length*4],
                                    src_ub[loop*FP16_ALIGN_NUM*4], 0,
                                    length//FP16_ALIGN_NUM, self.ratio, 15*self.ratio, 0)
        self.tik_instance.data_move(
            dst_ub[loop*length*4+length],
            src_ub[loop*FP16_ALIGN_NUM*4+FP16_ALIGN_NUM], 0,
            length//FP16_ALIGN_NUM, self.ratio, 15*self.ratio, 0)
        self.tik_instance.data_move(
            dst_ub[loop*length*4+length*2],
            src_ub[loop*FP16_ALIGN_NUM*4+FP16_ALIGN_NUM*2],
            0, length//FP16_ALIGN_NUM, self.ratio, 15*self.ratio, 0)
        self.tik_instance.data_move(
            dst_ub[loop*length*4+length*3],
            src_ub[loop*FP16_ALIGN_NUM*4+FP16_ALIGN_NUM*3], 0,
            length//FP16_ALIGN_NUM, self.ratio, 15*self.ratio, 0)

    def prior_gm_move(self, dst_list, src_list, repeat_times):
        """
        :param dst_list:
        :param src_list:
        :param repeat_times:
        :return:
        """
        prior_gm_offset = dst_list[0]
        prior_box_gm = dst_list[1]
        base_offset = dst_list[2]
        prior_ub_offset = src_list[0]
        prior_ub = src_list[1]
        ub_base_offset = src_list[2]
        self.tik_instance.data_move(prior_box_gm[prior_gm_offset],
                                    prior_ub[prior_ub_offset], 0,
                                    repeat_times, self.ratio, 0, 0)
        self.tik_instance.data_move(prior_box_gm[prior_gm_offset+base_offset],
                                    prior_ub[prior_ub_offset+ub_base_offset],
                                    0, repeat_times, self.ratio, 0, 0)
        self.tik_instance.data_move(
            prior_box_gm[prior_gm_offset+base_offset*2],
            prior_ub[prior_ub_offset+ub_base_offset*2],
            0, repeat_times, self.ratio, 0, 0)
        self.tik_instance.data_move(prior_box_gm[prior_gm_offset+base_offset*3],
                                    prior_ub[prior_ub_offset+ub_base_offset*3],
                                    0, repeat_times, self.ratio, 0, 0)

    def no_tiling_trans_prior_box(self, input_list, prior, prior_box_gm):
        shape = input_list[0]
        sum_addr = input_list[1]
        cur_batch_index = input_list[2]
        cur_batch_num = input_list[3]
        prior_box_align = 4
        align_times = (self.num_classes * 4 + 15) // FP16_ALIGN_NUM
        prior_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                            name="prior_ub",
                                            scope=tik.scope_ubuf)
        new_prior_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                                name="prior_ub",
                                                scope=tik.scope_ubuf)
        self.tik_instance.data_move(
            new_prior_ub[0],
            prior[sum_addr*align_times*FP16_ALIGN_NUM], 0,
            cur_batch_num*align_times, self.ratio,
            0, 0)

        with self.tik_instance.for_range(0, align_times) as loop:
            self.tik_instance.data_move(
                prior_ub[loop*cur_batch_num*FP16_ALIGN_NUM],
                new_prior_ub[loop*FP16_ALIGN_NUM], 0, cur_batch_num,
                self.ratio, (align_times - DATA_ONE) * self.ratio, 0)

        self.trans(prior_ub, new_prior_ub,
                   prior_ub.shape[0]*prior_ub.shape[2])
        vec_dup((self.tik_instance, shape[0]*shape[DATA_ONE]*shape[2],
                 self.input_dtype), prior_ub)
        with self.tik_instance.for_range(
                0, self.num_classes*prior_box_align // FP16_ALIGN_NUM) \
                as out_loop:
            with self.tik_instance.for_range(0, prior_box_align) as loop:
                with self.tik_instance.for_range(
                        0, prior_box_align) as inner_loop:
                    self.tik_instance.data_move(
                        prior_ub[
                            loop*self.max_rois_num+inner_loop *
                            self.max_rois_num*self.num_classes+out_loop *
                            self.max_rois_num*prior_box_align],
                        new_prior_ub[out_loop*FP16_ALIGN_NUM *
                                     self.max_rois_num + inner_loop *
                                     FP16_ALIGN_NUM +
                                     (loop*prior_box_align)*FP16_ALIGN_NUM],
                        0, self.max_rois_num//FP16_ALIGN_NUM, self.ratio,
                        15*self.ratio, 0)

        loop_times = self.num_classes % 4
        with self.tik_instance.for_range(0, loop_times) as loop:
            with self.tik_instance.for_range(0,
                                             prior_box_align) as inner_loop:
                self.tik_instance.data_move(
                    prior_ub[
                        loop*self.max_rois_num+inner_loop *
                        self.max_rois_num*self.num_classes +
                        (self.num_classes*prior_box_align) //
                        FP16_ALIGN_NUM *
                        self.max_rois_num*prior_box_align],
                    new_prior_ub[(self.num_classes*prior_box_align) //
                                 FP16_ALIGN_NUM * FP16_ALIGN_NUM *
                                 self.max_rois_num+inner_loop *
                                 FP16_ALIGN_NUM+(loop*prior_box_align) *
                                 FP16_ALIGN_NUM], 0,
                    self.max_rois_num//FP16_ALIGN_NUM, self.ratio,
                    15*self.ratio, 0)

        prior_gm_offset = cur_batch_index*self.max_rois_num * \
                          self.num_classes*prior_box_align

        self.tik_instance.data_move(
            prior_box_gm[prior_gm_offset],
            prior_ub, 0,
            self.max_rois_num//FP16_ALIGN_NUM *
            self.num_classes*prior_box_align,
            self.ratio, 0, 0)

    def tiling_trans_prior_box_branch1(self, input_list, prior, prior_box_gm):
        shape = input_list[0]
        sum_addr = input_list[1]
        class_loop = input_list[2]
        cur_batch_index = input_list[3]
        cur_batch_num = input_list[4]
        prior_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                            name="prior_ub",
                                            scope=tik.scope_ubuf)
        new_prior_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                                name="prior_ub",
                                                scope=tik.scope_ubuf)
        self.tik_instance.data_move(
            prior_ub[0],
            prior[sum_addr*((self.num_classes*4+TO_ALIGN_NUM) //
                            FP16_ALIGN_NUM)*FP16_ALIGN_NUM +
                  ((class_loop*4+TO_ALIGN_NUM)//FP16_ALIGN_NUM) *
                  FP16_ALIGN_NUM], 0, cur_batch_num, self.ratio,
            ((self.num_classes*4+TO_ALIGN_NUM)//FP16_ALIGN_NUM-DATA_ONE
             )*self.ratio, 0)
        self.trans(prior_ub, new_prior_ub, shape[0])
        vec_dup((self.tik_instance, shape[0]*shape[1],
                 self.input_dtype), prior_ub)

        with self.tik_instance.for_range(0, 4) as loop:
            with self.tik_instance.if_scope(class_loop*4+loop <
                                            self.num_classes):
                self.prior_ub_move(loop, shape[0], prior_ub,
                                   new_prior_ub)
        with self.tik_instance.for_range(0, 4) as loop:
            # if class_loop*16+loop < self.num_classes:
            with self.tik_instance.if_scope(
                    class_loop*4+loop < self.num_classes):
                prior_gm_offset = \
                    ((class_loop*4+TO_ALIGN_NUM)//FP16_ALIGN_NUM
                     )*FP16_ALIGN_NUM * \
                    self.max_rois_num+(class_loop*4 -
                                       ((class_loop*4+TO_ALIGN_NUM) //
                                        FP16_ALIGN_NUM) *
                                       FP16_ALIGN_NUM) * \
                    self.max_rois_num + \
                    cur_batch_index*self.max_rois_num*self.num_classes*4+loop * \
                    self.max_rois_num
                self.prior_gm_move((prior_gm_offset, prior_box_gm,
                                    self.max_rois_num*self.num_classes),
                                   (loop*self.max_rois_num*4, prior_ub,
                                    self.max_rois_num),
                                   self.max_rois_num//FP16_ALIGN_NUM)

    def tiling_trans_prior_box_branch2(self, input_list, prior, prior_box_gm):
        shape = input_list[0]
        sum_addr = input_list[1]
        class_loop = input_list[2]
        cur_batch_index = input_list[3]
        cur_batch_num = input_list[4]
        one_tiling_process = input_list[5]
        prior_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                            name="prior_ub",
                                            scope=tik.scope_ubuf)
        new_prior_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                                name="prior_ub",
                                                scope=tik.scope_ubuf)
        with self.tik_instance.for_range \
                    (0, cur_batch_num//(one_tiling_process //
                                        FP16_ALIGN_NUM)) as \
                inner_batch_loop:
            with self.tik_instance.if_scope(
                    cur_batch_num >=
                    (inner_batch_loop*one_tiling_process //
                     FP16_ALIGN_NUM)):
                self.tik_instance.data_move(
                    prior_ub[0],
                    prior[sum_addr+inner_batch_loop *
                          (one_tiling_process//FP16_ALIGN_NUM),
                          class_loop, 0, 0, 0], 0,
                    one_tiling_process//FP16_ALIGN_NUM, self.ratio,
                    ((self.num_classes*4+TO_ALIGN_NUM) //
                     FP16_ALIGN_NUM-DATA_ONE)*self.ratio, 0)

                self.trans(prior_ub, new_prior_ub, shape[0])
                vec_dup((self.tik_instance, shape[0]*shape[1],
                         self.input_dtype), prior_ub)

                with self.tik_instance.for_range(0, 4) as loop:
                    self.prior_ub_move(loop, shape[0], prior_ub,
                                       new_prior_ub)
                with self.tik_instance.for_range(0, 4) as loop:
                    with self.tik_instance.if_scope(
                            class_loop*4+loop < self.num_classes):
                        prior_gm_offset = \
                            (class_loop*4+loop)*self.batch_rois * \
                            self.max_rois_num*4+cur_batch_index * \
                            self.max_rois_num*4 + \
                            (one_tiling_process//FP16_ALIGN_NUM) * \
                            inner_batch_loop
                        self.prior_gm_move(
                            (prior_gm_offset, prior_box_gm,
                             self.max_rois_num),
                            (loop*(one_tiling_process //
                                   FP16_ALIGN_NUM)*4, prior_ub,
                             one_tiling_process//FP16_ALIGN_NUM),
                            (one_tiling_process//FP16_ALIGN_NUM) //
                            FP16_ALIGN_NUM)

        with self.tik_instance.if_scope(
                cur_batch_num -
                (cur_batch_num//(one_tiling_process//FP16_ALIGN_NUM)) *
                (one_tiling_process//FP16_ALIGN_NUM) > 0):
            # shape = [((tail+15)//16)*16, 16]
            self.tik_instance.data_move(
                prior_ub[0],
                prior[sum_addr+cur_batch_num//(one_tiling_process //
                                               FP16_ALIGN_NUM) *
                      (one_tiling_process//FP16_ALIGN_NUM),
                      class_loop, 0, 0, 0], 0,
                cur_batch_num -
                (cur_batch_num//(one_tiling_process//FP16_ALIGN_NUM)) *
                (one_tiling_process//FP16_ALIGN_NUM), self.ratio,
                ((self.num_classes*4+TO_ALIGN_NUM)//FP16_ALIGN_NUM-DATA_ONE) *
                self.ratio, 0)
            self.trans(prior_ub, new_prior_ub, shape[0])
            vec_dup((self.tik_instance, shape[0]*shape[1],
                     self.input_dtype), prior_ub)

            with self.tik_instance.for_range(0, 4) as loop:
                self.prior_ub_move(loop, shape[0], prior_ub,
                                   new_prior_ub)
            with self.tik_instance.for_range(0, 4) as loop:
                with self.tik_instance.if_scope(
                        class_loop*4+loop < self.num_classes):
                    prior_gm_offset = \
                        class_loop*4*self.batch_rois * \
                        self.max_rois_num*4 + \
                        cur_batch_index*self.max_rois_num*4 + \
                        loop*self.batch_rois*self.max_rois_num*4 + \
                        (one_tiling_process//FP16_ALIGN_NUM)*(
                            cur_batch_num //
                            (one_tiling_process//FP16_ALIGN_NUM))

                    self.prior_gm_move(
                        (prior_gm_offset, prior_box_gm,
                         self.max_rois_num), (
                             (loop*one_tiling_process //
                              FP16_ALIGN_NUM)*4,
                             prior_ub,
                             (one_tiling_process//FP16_ALIGN_NUM)),
                        (cur_batch_num -
                         (cur_batch_num //
                          (one_tiling_process//FP16_ALIGN_NUM)) *
                         (one_tiling_process//FP16_ALIGN_NUM) +
                         TO_ALIGN_NUM)//FP16_ALIGN_NUM)

    def trans_prior_box(self, prior, prior_box_gm, cur_batch_index):
        """
        :param prior:
        :param prior_box_gm:
        :param cur_batch_index:
        :return:
        """
        ub_size = get_ub_size()
        align_times = (self.num_classes * 4 + 15) // FP16_ALIGN_NUM
        one_batch_size = self.max_rois_num * FP16_ALIGN_NUM * self.size *\
                         align_times * 2 + self.batch_rois * DATA_EIGHT * 4 * 2

        if self.max_rois_num*self.batch_rois != prior.shape[0]:
            sum_addr = self.tik_instance.Scalar("int32")
            sum_addr.set_as(0)
            cur_batch_num = self.tik_instance.Scalar("int32")
            cur_batch_num.set_as(0)
            sum_addr.set_as(self.actual_sum_num[
                                cur_batch_index*DATA_EIGHT])
            cur_batch_num.set_as(
                self.actual_rois_num_ub[
                    cur_batch_index*DATA_EIGHT])
        else:
            cur_batch_num = self.max_rois_num
            sum_addr = cur_batch_index*self.max_rois_num

        if one_batch_size < ub_size:
            shape = [align_times, FP16_ALIGN_NUM, self.max_rois_num]
            self.no_tiling_trans_prior_box((shape, sum_addr,
                                            cur_batch_index, cur_batch_num),
                                           prior, prior_box_gm)
        else:
            num_piece_of_space = \
                ((ub_size-self.batch_rois*constant.DATA_SIZE_EIGHT*4*2) //
                 constant.DATA_SIZE_TWO)//((FP16_ALIGN_NUM*FP16_ALIGN_NUM) *
                                           self.size)
            one_tiling_process = num_piece_of_space*FP16_ALIGN_NUM*FP16_ALIGN_NUM
            if one_tiling_process//FP16_ALIGN_NUM < self.max_rois_num:
                shape = [one_tiling_process//FP16_ALIGN_NUM, FP16_ALIGN_NUM]
            else:
                shape = [self.max_rois_num, FP16_ALIGN_NUM]
            with self.tik_instance.for_range(
                    0, (self.num_classes*4+TO_ALIGN_NUM)//FP16_ALIGN_NUM) \
                    as class_loop:
                if one_tiling_process >= self.max_rois_num*FP16_ALIGN_NUM:
                    self.tiling_trans_prior_box_branch1((shape, sum_addr,
                                                         class_loop, cur_batch_index,
                                                         cur_batch_num), prior, prior_box_gm)
                else:
                    self.tiling_trans_prior_box_branch2((shape, sum_addr, class_loop,
                                                         cur_batch_index, cur_batch_num,
                                                         one_tiling_process), prior, prior_box_gm)


class FsrProcess:
    """
    Faster Process
    """
    def __init__(self, tik_instance, input_fsr, attr_fsr):
        """
        :param tik_instance:
        :param input_fsr:
        :param attr_fsr:
        """
        rois_dic = input_fsr[0]
        bbox_delta_dic = input_fsr[1]
        score_dic = input_fsr[2]
        im_info_dic = input_fsr[3]
        actual_bbox_num_dic = input_fsr[4]
        box_dic = input_fsr[5]
        if len(input_fsr) == 7:
            actual_rois_num_dic = input_fsr[6]

        self.batch_rois = attr_fsr[0]
        self.num_classes = attr_fsr[1]
        self.score_threshlod = attr_fsr[2]
        self.nms_threshold = attr_fsr[3]

        self.max_rois_num = rois_dic.get("shape")[2]
        if len(input_fsr) == 7:
            self.actual_rois_num_effect = True
        else:
            self.actual_rois_num_effect = False

        self.total_num = bbox_delta_dic.get("shape")[0]

        if self.max_rois_num >= 1024:
            self.post_nms_topn = 1024
        else:
            self.post_nms_topn = self.max_rois_num

        self.input_dtype = rois_dic.get('dtype')
        self.input_shape = bbox_delta_dic.get('shape')

        self.tik_instance = tik_instance

        # self.size = get_dtype_size(self.input_dtype)
        self.size, _, _ = get_params(self.input_dtype)
        self.rois = self.tik_instance.Tensor(
            self.input_dtype, rois_dic.get('shape'),
            name="rois", scope=tik.scope_gm)

        self.bbox_delta = self.tik_instance.Tensor(
            self.input_dtype, self.input_shape, name="bbox_delta", scope=tik.scope_gm)

        self.score = self.tik_instance.Tensor(
            self.input_dtype, score_dic.get('shape'), name="score", scope=tik.scope_gm)

        self.im_info = self.tik_instance.Tensor(self.input_dtype,
                                                im_info_dic.get("shape"),
                                                name="im_info",
                                                scope=tik.scope_gm)

        self.score_gm = self.tik_instance.Tensor(
            self.input_dtype, (self.batch_rois, self.num_classes,
                               self.max_rois_num//FP16_ALIGN_NUM, DATA_ONE, FP16_ALIGN_NUM),
            name="score_gm", is_workspace=True, scope=tik.scope_gm)

        self.prior_box_gm = self.tik_instance.Tensor(
            self.input_dtype, (self.batch_rois, 4, self.num_classes,
                               self.max_rois_num//FP16_ALIGN_NUM, FP16_ALIGN_NUM),
            scope=tik.scope_gm, is_workspace=True, name="prior_box_gm")

        if self.actual_rois_num_effect:
            # print("aaa")
            self.actual_rois_num = self.tik_instance.Tensor(
                "int32", actual_rois_num_dic.get("shape"),
                name="actual_rois_num", scope=tik.scope_gm)

        self.output_box = self.tik_instance.Tensor(
            self.input_dtype, (self.num_classes*self.batch_rois, self.max_rois_num,
                               DATA_EIGHT),
            name="output_box", scope=tik.scope_gm, is_workspace=True)

        self.mem_swap = self.tik_instance.Tensor(
            self.input_dtype, (self.num_classes*self.batch_rois, self.max_rois_num,
                               DATA_EIGHT),
            name="mem_swap", scope=tik.scope_gm, is_workspace=True)

        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES","Hi3796CV300CS"):
            self.pre_nms_topn = 3000
        else:
            if self.input_dtype == "float32":
                self.pre_nms_topn = 3000
            else:
                self.pre_nms_topn = 6000
        if self.max_rois_num < self.pre_nms_topn:
            self.pre_nms_topn = self.max_rois_num

        self.topk_output_proposal = self.tik_instance.Tensor(
            self.input_dtype, (self.num_classes*self.batch_rois,
                               ((self.pre_nms_topn+15)//FP16_ALIGN_NUM) *
                               FP16_ALIGN_NUM+4, DATA_EIGHT),
            name="topk_output_proposal", is_workspace=True, scope=tik.scope_gm)

        self.temp_proposal_out = self.tik_instance.Tensor(
            self.input_dtype, (self.num_classes*self.batch_rois,
                               (self.post_nms_topn+15)//FP16_ALIGN_NUM*FP16_ALIGN_NUM,
                               DATA_EIGHT),
            name="temp_proposal_out",
            is_workspace=True, scope=tik.scope_gm)

        self.box = self.tik_instance.Tensor(
            self.input_dtype, box_dic.get("shape"),
            name="box", scope=tik.scope_gm)

        self.actual_bbox_num = self.tik_instance.Tensor(
            "int32", actual_bbox_num_dic.get("shape"),
            name="actual_bbox_num", scope=tik.scope_gm)

    def cce_fsr(self, kernel_name):
        """
        :param kernel_name:
        :return:
        """
        device_core_num, batch_factor, batch_factor_tail = \
            filter_device_core(self.batch_rois)

        with self.tik_instance.for_range(0, device_core_num,
                                         block_num=device_core_num) as \
                block_id:

            ub_size = get_ub_size()
            one_core_process_object = OneCoreProcess(
                (self.tik_instance, self.input_dtype, self.input_shape,
                 self.size, self.batch_rois, self.im_info,
                 self.num_classes, self.max_rois_num,
                 self.total_num, device_core_num, batch_factor, block_id,
                 ub_size))

            if self.actual_rois_num_effect:
                input_tensor = [self.actual_rois_num_effect, self.actual_rois_num]
            else:
                input_tensor = [self.actual_rois_num_effect]

            with self.tik_instance.for_range(
                    0, batch_factor)  as batch_index:

                cur_batch_index = block_id*batch_factor+batch_index
                with self.tik_instance.new_stmt_scope():
                    pre_object = PreProcess((self.tik_instance,
                                             self.max_rois_num,
                                             self.num_classes,
                                             self.batch_rois,
                                             self.input_shape,
                                             self.input_dtype),
                                            input_tensor)
                    with self.tik_instance.new_stmt_scope():
                        pre_object.trans_score(self.score, self.score_gm,
                                               cur_batch_index)
                    with self.tik_instance.new_stmt_scope():
                        pre_object.trans_prior_box(self.bbox_delta,
                                                   self.prior_box_gm,
                                                   cur_batch_index)
                with self.tik_instance.new_stmt_scope():
                    one_core_process_object.one_core_process_decode_rois(
                        (cur_batch_index, self.rois, input_tensor,
                         self.prior_box_gm, self.score_gm), self.output_box)

                with self.tik_instance.for_range(DATA_ONE, self.num_classes) \
                        as class_index:

                    topk_output_actual_proposal_num = \
                        self.tik_instance.Scalar(dtype="int32")

                    batch_id = cur_batch_index*self.num_classes + class_index

                    with self.tik_instance.new_stmt_scope():
                        call_topk_sort(
                            self.tik_instance, (self.max_rois_num,
                                                self.score_threshlod,
                                                self.pre_nms_topn,
                                                self.output_box, self.mem_swap),
                            (batch_id, self.topk_output_proposal,
                             topk_output_actual_proposal_num))

                    input_offset = \
                        batch_id*(((self.pre_nms_topn+15) //
                                   FP16_ALIGN_NUM)*FP16_ALIGN_NUM+4) * \
                        DATA_EIGHT
                    real_batch_index = cur_batch_index
                    with self.tik_instance.new_stmt_scope():
                        used_in_proposal = False
                        nms.cce_nms(
                            (self.input_dtype, ub_size, self.nms_threshold,
                             batch_id, self.pre_nms_topn, self.post_nms_topn,
                             input_offset, self.im_info, self.tik_instance,
                             self.num_classes, class_index, real_batch_index),
                            self.temp_proposal_out, self.topk_output_proposal,
                            topk_output_actual_proposal_num,
                            self.actual_bbox_num, self.box,
                            used_in_proposal)

                with self.tik_instance.if_scope(block_id
                                                < batch_factor_tail):
                    cur_batch_index = batch_factor*device_core_num+block_id
                    with self.tik_instance.new_stmt_scope():
                        pre_object = PreProcess((self.tik_instance,
                                                 self.max_rois_num,
                                                 self.num_classes,
                                                 self.batch_rois,
                                                 self.input_shape,
                                                 self.input_dtype),
                                                input_tensor)
                        with self.tik_instance.new_stmt_scope():
                            pre_object.trans_score(self.score,
                                                   self.score_gm,
                                                   cur_batch_index)
                        with self.tik_instance.new_stmt_scope():
                            pre_object.trans_prior_box(self.bbox_delta,
                                                       self.prior_box_gm,
                                                       cur_batch_index)
                    with self.tik_instance.new_stmt_scope():
                        one_core_process_object.one_core_process_decode_rois((
                            cur_batch_index, self.rois, input_tensor,
                            self.prior_box_gm, self.score_gm), self.output_box)
                    with self.tik_instance.for_range(DATA_ONE, self.num_classes) \
                            as class_index:
                        topk_output_actual_proposal_num = \
                            self.tik_instance.Scalar(dtype="int32")

                        batch_id = cur_batch_index*self.num_classes+class_index
                        real_batch_index = cur_batch_index
                        with self.tik_instance.new_stmt_scope():
                            call_topk_sort(self.tik_instance,
                                           (self.max_rois_num,
                                            self.score_threshlod,
                                            self.pre_nms_topn, self.output_box,
                                            self.mem_swap),
                                           (batch_id, self.topk_output_proposal,
                                            topk_output_actual_proposal_num))

                        input_offset = \
                            batch_id*(((self.pre_nms_topn+15) //
                                       FP16_ALIGN_NUM)*FP16_ALIGN_NUM+4) * \
                            DATA_EIGHT

                        with self.tik_instance.new_stmt_scope():
                            used_in_proposal = False
                            nms.cce_nms((self.input_dtype, ub_size,
                                         self.nms_threshold, batch_id,
                                         self.pre_nms_topn, self.post_nms_topn,
                                         input_offset, self.im_info,
                                         self.tik_instance,
                                         self.num_classes, class_index,
                                         real_batch_index),
                                        self.temp_proposal_out,
                                        self.topk_output_proposal,
                                        topk_output_actual_proposal_num,
                                        self.actual_bbox_num,
                                        self.box, used_in_proposal)

        if self.actual_rois_num_effect:
            self.tik_instance.BuildCCE(kernel_name,
                                       inputs=[self.rois,
                                               self.bbox_delta, self.score,
                                               self.im_info,
                                               self.actual_rois_num],
                                       outputs=[self.actual_bbox_num,
                                                self.box])
        else:
            self.tik_instance.BuildCCE(kernel_name,
                                       inputs=[self.rois,
                                               self.bbox_delta, self.score,
                                               self.im_info],
                                       outputs=[self.actual_bbox_num,
                                                self.box])

        return self.tik_instance


def check_dtype(tik_name, dtype):
    """
    :param tik_name:
    :param dtype:
    :return:
    """

    if tik_name in ("Ascend310",):
        util.check_dtype_rule(dtype.lower(), ["float16"])
    elif tik_name in ("Ascend910",):
        util.check_dtype_rule(dtype.lower(), ["float16"])
    elif tik_name in ("Hi3796CV300ES",):
        util.check_dtype_rule(dtype.lower(), ["float16"])
    elif tik_name in ("Ascend610","Ascend620"):
        util.check_dtype_rule(dtype.lower(), ["float16", "float32"])


@util.check_input_type(dict, dict, dict, dict, (dict, NoneType),
                       dict, dict, int, float, float, int, str)
def fsr_detection_output(rois_dic, bbox_delta_dic, score_dic, im_info_dic,
                         actual_rois_num_dic, actual_bbox_num_dic, box_dic,
                         num_classes, score_threshold, iou_threshold, batch_rois=1,
                         kernel_name="fsr_detection_output"):
    """
    :param rois_dic:
    :param bbox_delta_dic:
    :param score_dic:
    :param im_info_dic:
    :param actual_rois_num_dic:
    :param actual_bbox_num_dic:
    :param box_dic:
    :param num_classes:
    :param score_threshold:
    :param iou_threshold:
    :param batch_rois:
    :param kernel_name:
    :return:
    """

    tik_instance = tik.Tik(tik.Dprofile())
    tik_name = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    input_dtype = rois_dic.get('dtype')
    check_dtype(tik_name, input_dtype)
    batch_rois = rois_dic.get("shape")[0]
    if num_classes > score_dic.get("shape")[1] * score_dic.get("shape")[4]:
        raise RuntimeError("num_classes is larger than the second tensor need!")
    if num_classes > bbox_delta_dic.get("shape")[1] * bbox_delta_dic.get("shape")[4] // 4:
        raise RuntimeError("num_classes is larger than the third tensor need!")
    if iou_threshold <= 0.0 or iou_threshold >= 1.0:
        raise RuntimeError("nms_thresh should be within (0.0, 1.0)!")
    if score_threshold < 0.0 or score_threshold > 1.0:
        raise RuntimeError("score_thresh should be within [0.0, 1.0]!")
    if num_classes < 1:
        raise RuntimeError("the num_class should be greater than 0!")
    if actual_rois_num_dic:
        input_list = (rois_dic, bbox_delta_dic, score_dic, im_info_dic,
                      actual_bbox_num_dic, box_dic, actual_rois_num_dic)
    else:
        input_list = (rois_dic, bbox_delta_dic, score_dic, im_info_dic,
                      actual_bbox_num_dic, box_dic)
    fsr_result = FsrProcess(tik_instance, input_list,
                            (batch_rois, num_classes, score_threshold,
                             iou_threshold))

    return fsr_result.cce_fsr(kernel_name)
