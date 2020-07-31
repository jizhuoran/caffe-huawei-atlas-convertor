#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

ssd_detection_out
"""
import math
from te import tik
from topi.cce import util

from impl import ssd_decode_bbox
from impl import topk
from impl import nms
from te import platform as tbe_platform

def check_product_info(input_dict):
    """
    check product info

    Parameters
    ----------
    input_dict: input dict

    Returns
    -------
    None
    """
    #tik_name = tik.Dprofile().get_product_name()
    tik_name = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    conf_dtype = input_dict.get("mbox_conf").get("dtype").lower()
    loc_dtype = input_dict.get("mbox_loc").get("dtype").lower()
    priorbox_dtype = input_dict.get("mbox_priorbox").get("dtype").lower()

    if not conf_dtype == loc_dtype and conf_dtype == loc_dtype \
            and loc_dtype == priorbox_dtype:
        raise RuntimeError("input type is error")

    if tik_name in ("Ascend310",):
        util.check_dtype_rule(conf_dtype.lower(), ["float16"])
    elif tik_name in ("Ascend910",):
        util.check_dtype_rule(conf_dtype.lower(), ["float16"])
    elif tik_name in ("Hi3796CV300ES",):# "Hi3796CV300CS"):
        util.check_dtype_rule(conf_dtype.lower(), ["float16"])
    elif tik_name in ("Ascend610","Ascend620"):
        util.check_dtype_rule(conf_dtype.lower(), ["float16", "float32"])

def check_input_attr_value(input_dict):
    """
    check input attr value,

    Parameters
    ----------
    input_dict: input dict

    Returns
    -------
    None
    """
    if not (input_dict.get("num_classes") >= 1 and
            input_dict.get("num_classes") <= 1024):
        raise RuntimeError("num classes val should be [2, 1024]")

    if not input_dict.get("share_location"):
        raise RuntimeError("share_location should be True")

    if not (input_dict.get("background_label_id") >= -1 and
            input_dict.get("background_label_id") <= (input_dict.get("num_classes")-1)):
        raise RuntimeError("background_label_id should be [-1, %d]"%(
            input_dict.get("num_classes")-1))

    if not (input_dict.get("nms_threshold") > 0 and
            input_dict.get("nms_threshold") <= 1):
        raise RuntimeError("nms_threshold should be (0, 1]")

    if not input_dict.get("eta") == 1:
        raise RuntimeError("eta only support 1")

    if not (input_dict.get("code_type") >= 1 and
            input_dict.get("code_type") <= 3):
        raise RuntimeError("code_type should be [1, 3]")

    if not ((input_dict.get("keep_top_k") <= 1024 and input_dict.get("keep_top_k") > 0)
            or input_dict.get("keep_top_k") == -1):
        raise RuntimeError("keep_top_k should be (0, 1024] or == -1")

    if not (input_dict.get("confidence_threshold") >= 0 and
            input_dict.get("confidence_threshold") <= 1):
        raise RuntimeError("confidence_threshold should be [0, 1]")

    check_input_topk_value(input_dict.get("mbox_loc").get("dtype").lower(),
                           input_dict.get("top_k"))

def check_input_topk_value(dtype, topk_value):
    """
    check input topk value,

    Parameters
    ----------
    input_dict: input dict

    Returns
    -------
    None
    """

    if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES", "Hi3796CV300CS"):
        if dtype == "float32":
            if not topk_value <= 1500:
                raise RuntimeError("top_k > 1500")
        else:
            if not topk_value <= 3000:
                raise RuntimeError("top_k > 3000")

        if not topk_value <= 3000:
            raise RuntimeError("top_k > 3000")
    else:
        if dtype == "float32":
            if not topk_value <= 3000:
                raise RuntimeError("top_k > 3000")
        else:
            if not topk_value <= 6000:
                raise RuntimeError("top_k > 6000")

def check_input_data_logical_relationship(input_dict):
    """
    check_input_data_logical_relationship

    Parameters
    ----------
    input_dict: input dict

    Returns
    -------
    None
    """
    conf_shape = input_dict.get("mbox_conf").get("shape")
    loc_shape = input_dict.get("mbox_loc").get("shape")
    priorbox_shape = input_dict.get("mbox_priorbox").get("shape")
    num_classes = input_dict.get("num_classes")

    if not conf_shape[0] == loc_shape[0] and conf_shape[0] == priorbox_shape[0] \
            and loc_shape[0] == priorbox_shape[0]:
        raise RuntimeError("batch num is error")

    if not conf_shape[1] // num_classes == loc_shape[1] // 4:
        raise RuntimeError("input shape is error")

    if not loc_shape[1] // 4 == priorbox_shape[2] // 4:
        raise RuntimeError("input shape is error")

    if not priorbox_shape[1] == 2:
        raise RuntimeError("priorbox shape is error")

# pylint: disable=invalid-name, too-many-arguments, too-many-locals
@util.check_input_type(dict, dict, dict, dict, dict, int, bool, int, float, int,
                       float, bool, int, int, float, str)
def ssd_detection_output(bbox_delta, score, anchors,
                         out_boxnum, y,
                         num_classes,
                         share_location=True,
                         background_label_id=0,
                         iou_threshold=0.45,
                         top_k=400,
                         eta=1.0,
                         variance_encoded_in_target=False,
                         code_type=1,
                         keep_top_k=-1,
                         confidence_threshold=0.0,
                         kernel_name="ssd_detection_output"):

    """
    the entry function of ssd detection output

    Parameters
    ----------
    mbox_conf: dict, the shape of mbox conf
    mbox_loc: dict, the shape of mbox loc
    mbox_priorbox: dict, the shape of mbox priorbox
    out_box_num: dict, the shape of out box number
    y: dict, the shape of out box
    num_classes: class num
    share_location: share location
    background_label_id: background label id
    nms_threshold: nms threshold
    top_k: class top num value
    eta: eta
    variance_encoded_in_target: variance_encoded_in_target
    code_type: code type
    keep_top_k: keep nms num value
    confidence_threshold: topk threshold
    kernel_name: cce kernel name

    Returns
    -------
    tik_instance:
    """
    input_dict = {
        "mbox_loc": bbox_delta,
        "mbox_conf": score,
        "mbox_priorbox": anchors,

        "out_box_num": out_boxnum,
        "out_box": y,

        "num_classes": num_classes,
        "share_location": share_location,
        "background_label_id": background_label_id,
        "nms_threshold": iou_threshold,
        "top_k": top_k,
        "eta": eta,
        "variance_encoded_in_target": variance_encoded_in_target,
        "code_type": code_type,
        "keep_top_k": keep_top_k,
        "confidence_threshold": confidence_threshold,

        "kernel_name": kernel_name
    }

    tik_instance = tik.Tik(tik.Dprofile())
    check_product_info(input_dict)
    check_input_attr_value(input_dict)
    check_input_data_logical_relationship(input_dict)

    decode_bbox_process = ssd_decode_bbox.SSDDecodeBBox(input_dict, tik_instance)

    detection_out_process = SSDDetectionOutput(
        input_dict, tik_instance,
        decode_bbox_process.decode_bbox_out_gm.shape[2]-decode_bbox_process.burnest_len)

    block_num, outer_loop, outer_tail = decode_bbox_process.get_block_param()
    with tik_instance.for_range(0, block_num, block_num=block_num) as block_i:

        batch = tik_instance.Scalar("int32", "batch", 0)
        with tik_instance.for_range(0, outer_loop) as outer_i:
            batch.set_as(block_i * outer_loop + outer_i)
            decode_bbox_process.parser_loc_data(batch)
            decode_bbox_process.parser_priorbox_data(batch)
            decode_bbox_process.parser_conf_data(batch)
            decode_bbox_process.compute_detection_out(batch)
            detection_out_process.get_topk_target_info(
                batch, decode_bbox_process.decode_bbox_out_gm)
        if outer_tail > 0:
            with tik_instance.if_scope(block_i < outer_tail):
                batch.set_as(block_num * outer_loop + block_i)
                decode_bbox_process.parser_loc_data(batch)
                decode_bbox_process.parser_priorbox_data(batch)
                decode_bbox_process.parser_conf_data(batch)
                decode_bbox_process.compute_detection_out(batch)
                detection_out_process.get_topk_target_info(
                    batch, decode_bbox_process.decode_bbox_out_gm)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=(decode_bbox_process.mbox_loc_gm,
                                  decode_bbox_process.mbox_conf_gm,
                                  decode_bbox_process.mbox_prior_gm),
                          outputs=(detection_out_process.out_box_num_gm,
                                   detection_out_process.out_box_gm))

    return tik_instance

# pylint: disable=too-many-instance-attributes
class SSDDetectionOutput(ssd_decode_bbox.SSDDectionParamInit):
    """
    define SSDDetectionOutput class

    """

    def __init__(self, input_dict, tik_instance, topk_src_len):
        """
        SSDDetectionOutput init function

        Parameters
        ----------
        input_dict: dict, inout dict
        tik_instance: tik instance
        topk_src_len: topk num

        Returns
        -------
        None
        """
        self.instance = tik_instance
        super(SSDDetectionOutput, self).__init__(input_dict)

        #paser input args
        self.nms_threshold = input_dict.get("nms_threshold")
        self.top_k = input_dict.get("top_k")
        self.eta = input_dict.get("eta")
        self.keep_top_k = input_dict.get("keep_top_k")
        self.confidence_threshold = input_dict.get("confidence_threshold")

        #define for topk1
        self.topk1_in_gm_len = topk_src_len
        self.topk1_in_gm = self.instance.Tensor(self.dtype,
                                                (self.batch, self.topk1_in_gm_len, 8),
                                                name="topk1_in_gm",
                                                is_workspace=True,
                                                scope=tik.scope_gm)

        self.topk1_swap_gm = self.instance.Tensor(self.dtype,
                                                  (self.batch, self.topk1_in_gm_len, 8),
                                                  name="topk1_swap_gm",
                                                  is_workspace=True,
                                                  scope=tik.scope_gm)

        topk1_out_gm_len = math.ceil(input_dict["top_k"] / 16) * 16
        self.topk1_out_gm = self.instance.Tensor(self.dtype,
                                                 (self.batch, topk1_out_gm_len+4, 8),
                                                 name="topk1_out_gm",
                                                 is_workspace=True,
                                                 scope=tik.scope_gm)
        #define for nms
        self.nms_box_num_gm = self.instance.Tensor(
            "int32",
            (self.num_classes, self.batch, 8),
            name="nms_box_num_gm",
            is_workspace=True,
            scope=tik.scope_gm)

        self.post_nms_topn = math.ceil(self.keep_top_k / 16) * 16
        if self.keep_top_k <= 0 or self.keep_top_k > self.top_k:
            self.post_nms_topn = math.ceil(self.top_k / 16) * 16
        self.nms_swap_gm = self.instance.Tensor(
            self.dtype,
            (self.num_classes * self.batch, self.post_nms_topn, 8),
            name="nms_swap_gm",
            is_workspace=True,
            scope=tik.scope_gm)

        self.nms_box_gm = self.instance.Tensor(
            self.dtype,
            (self.num_classes, self.batch, self.post_nms_topn, 8),
            name="nms_box_gm",
            is_workspace=True,
            scope=tik.scope_gm)

        #define for topk2
        topk2_in_gm_len = self.num_classes * self.post_nms_topn
        self.topk2_in_gm = self.instance.Tensor(self.dtype,
                                                (self.batch, topk2_in_gm_len, 8),
                                                name="topk2_in_gm",
                                                is_workspace=True,
                                                scope=tik.scope_gm)

        self.topk2_swap_gm = self.instance.Tensor(self.dtype,
                                                  (self.batch, topk2_in_gm_len, 8),
                                                  name="topk2_swap_gm",
                                                  is_workspace=True,
                                                  scope=tik.scope_gm)
        self.topk2_num = self.instance.Scalar("int32", "topk2_num", 0)

        #define for outbox
        out_box_len = math.ceil(self.keep_top_k / 128) * 128
        if self.keep_top_k <= 0:
            out_box_len = math.ceil(1024 / 128) * 128
        self.out_box_gm = self.instance.Tensor(self.dtype,
                                               (self.batch, out_box_len, 8),
                                               name="out_box_gm",
                                               scope=tik.scope_gm)

        self.out_box_num_gm = self.instance.Tensor("int32",
                                                   (self.batch, 8),
                                                   name="out_box_num_gm",
                                                   scope=tik.scope_gm)
        #define for topk3
        self.topk3_in_gm = self.instance.Tensor(self.dtype,
                                                (self.batch, topk2_in_gm_len, 8),
                                                name="topk3_in_gm",
                                                is_workspace=True,
                                                scope=tik.scope_gm)

        self.topk3_swap_gm = self.instance.Tensor(self.dtype,
                                                  (self.batch, topk2_in_gm_len, 8),
                                                  name="topk3_swap_gm",
                                                  is_workspace=True,
                                                  scope=tik.scope_gm)
        self.topk3_num = self.instance.Scalar("int32", "topk3_num", 0)
        self.topk3_out_gm = self.instance.Tensor(self.dtype,
                                                 (self.batch, out_box_len, 8),
                                                 name="topk3_out_gm",
                                                 is_workspace=True,
                                                 scope=tik.scope_gm)

    def sort_each_class_prepare(self, batch, class_index, topk_src_data):
        """
        sort each class prepare

        Parameters
        ----------
        batch: batch num
        class_index: class num
        topk_src_data: topk data

        Returns
        -------
        None
        """
        with self.instance.new_stmt_scope():

            topk1_in_data_tmp_ub = self.instance.Tensor(self.dtype,
                                                        (self.ub_capacity, ),
                                                        name="topk1_in_data_tmp_ub",
                                                        scope=tik.scope_ubuf)

            data_move_loop = self.topk1_in_gm_len * 8 // self.ub_capacity
            data_move_tail = self.topk1_in_gm_len * 8 % self.ub_capacity

            with self.instance.for_range(0, data_move_loop) as data_move_index:
                topk1_in_offset = data_move_index * (self.ub_capacity // 8)
                self.instance.data_move(topk1_in_data_tmp_ub,
                                        topk_src_data[batch, class_index, topk1_in_offset, 0],
                                        0, 1,
                                        (self.ub_capacity // self.burnest_len), 0, 0)

                self.instance.data_move(self.topk1_in_gm[batch, topk1_in_offset, 0],
                                        topk1_in_data_tmp_ub,
                                        0, 1,
                                        (self.ub_capacity // self.burnest_len), 0, 0)

            with self.instance.if_scope(data_move_tail > 0):
                topk1_in_offset = data_move_loop * self.ub_capacity // 8
                self.instance.data_move(topk1_in_data_tmp_ub,
                                        topk_src_data[batch, class_index, topk1_in_offset, 0],
                                        0, 1,
                                        (data_move_tail // self.burnest_len), 0, 0)

                self.instance.data_move(self.topk1_in_gm[batch, topk1_in_offset, 0],
                                        topk1_in_data_tmp_ub,
                                        0, 1,
                                        (data_move_tail // self.burnest_len), 0, 0)

    def get_tersor_data_burst_val(self, is_scalar, tersor_num_data, burst_val_tmp_scalar):
        """
        get tersor data burst val

        Parameters
        ----------
        is_scalar: whether tersor_num_data is scalar or not
        tersor_num_data: tersor data num
        burst_val_tmp_scalar: data move burst value

        Returns
        -------
        None
        """
        with self.instance.new_stmt_scope():
            nms_box_num_tmp_scalar = self.instance.Scalar("int32", "nms_box_num_tmp_scalar", 0)

            if is_scalar:
                nms_box_num_tmp_scalar.set_as(tersor_num_data)
            else:
                nms_box_num_tmp_scalar.set_as(tersor_num_data[0])

            with self.instance.if_scope(nms_box_num_tmp_scalar % 2 != 0):
                nms_box_num_tmp_scalar.set_as(nms_box_num_tmp_scalar + 1)


            with self.instance.if_scope(self.dsize == 4):
                burst_val_tmp_scalar.set_as(nms_box_num_tmp_scalar)
            with self.instance.else_scope():
                burst_val_tmp_scalar.set_as(nms_box_num_tmp_scalar >> 1)

    def sort_all_class_prepare(self, batch, class_index, topk_num_ecah_class):
        """
        sort all class prepare

        Parameters
        ----------
        batch: batch num
        class_index: class index
        data_offset: data offset

        Returns
        -------
        None
        """
        with self.instance.new_stmt_scope():
            topk2_in_data_len = math.ceil(self.top_k / 16) * 16
            topk2_in_data_tmp_ub = self.instance.Tensor(
                self.dtype, (topk2_in_data_len, 8),
                name="topk2_in_data_tmp_ub",
                scope=tik.scope_ubuf)

            nms_bbox_num_ub = self.instance.Tensor("int32", (8, ),
                                                   name="nms_bbox_num_ub",
                                                   scope=tik.scope_ubuf)
            self.instance.data_move(nms_bbox_num_ub,
                                    self.nms_box_num_gm[class_index, batch, 0],
                                    0, 1, 1, 0, 0)
            nms_num_scalar = self.instance.Scalar("int32", "nms_num_scalar",
                                                  nms_bbox_num_ub[0])

            burst_val_tmp_scalar = self.instance.Scalar("int32",
                                                        "burst_val_tmp_scalar", 0)
            self.get_tersor_data_burst_val(False, nms_bbox_num_ub,
                                           burst_val_tmp_scalar)

            with self.instance.if_scope(burst_val_tmp_scalar > 0):
                self.instance.data_move(topk2_in_data_tmp_ub,
                                        self.nms_box_gm[class_index, batch, 0, 0],
                                        0, 1,
                                        burst_val_tmp_scalar, 0, 0)
                self.instance.data_move(self.topk2_in_gm[batch, topk_num_ecah_class, 0],
                                        topk2_in_data_tmp_ub,
                                        0, 1,
                                        burst_val_tmp_scalar, 0, 0)

                with self.instance.for_range(0, nms_num_scalar) as num_index:
                    topk2_in_data_tmp_ub[num_index, 0].set_as(topk2_in_data_tmp_ub[num_index, 6])
                    topk2_in_data_tmp_ub[num_index, 1].set_as(topk2_in_data_tmp_ub[num_index, 5])

                self.instance.data_move(self.topk3_in_gm[batch, topk_num_ecah_class, 0],
                                        topk2_in_data_tmp_ub,
                                        0, 1,
                                        burst_val_tmp_scalar, 0, 0)

    def adjust_topk_crood(self, batch, topk_num_ecah_class):
        """
        modify x1 and y1 value

        Parameters
        ----------
        batch: batch num
        topk_num_ecah_class: out box data num

        Returns
        -------
        None
        """
        with self.instance.new_stmt_scope():

            box_data_ub = self.instance.Tensor(self.dtype,
                                               (self.out_box_gm.shape[1], 8),
                                               name="box_data_ub",
                                               scope=tik.scope_ubuf)
            topk3_out_ub = self.instance.Tensor(self.dtype,
                                                (self.out_box_gm.shape[1], 8),
                                                name="topk3_out_ub",
                                                scope=tik.scope_ubuf)
            burst_val_tmp_scalar = self.instance.Scalar("int32",
                                                        "burst_val_tmp_scalar", 0)

            with self.instance.if_scope(tik.all(self.keep_top_k > -1,
                                                topk_num_ecah_class > self.keep_top_k)):
                if self.keep_top_k > 0:
                    self.instance.data_move(box_data_ub, self.out_box_gm[batch, 0, 0],
                                            0, 1,
                                            math.ceil(self.keep_top_k * 8 / self.burnest_len),
                                            0, 0)
                    self.instance.data_move(topk3_out_ub, self.topk3_out_gm[batch, 0, 0],
                                            0, 1,
                                            math.ceil(self.keep_top_k * 8 / self.burnest_len),
                                            0, 0)

            with self.instance.else_scope():
                self.get_tersor_data_burst_val(True, topk_num_ecah_class,
                                               burst_val_tmp_scalar)
                with self.instance.if_scope(burst_val_tmp_scalar > 0):
                    self.instance.data_move(box_data_ub,
                                            self.topk2_in_gm[batch, 0, 0],
                                            0, 1, burst_val_tmp_scalar, 0, 0)
            self.set_crood_data_order(batch, topk_num_ecah_class, box_data_ub, topk3_out_ub)


    def set_crood_data_order(self, batch, topk_num_ecah_class, box_data_ub, topk3_out_ub):
        """
        modify out box data order as same as caffe

        Parameters
        ----------
        batch: batch num
        topk_num_ecah_class: out box data num

        Returns
        -------
        None
        """
        combin_out = self.instance.Tensor(self.dtype, (8, self.out_box_gm.shape[1]),
                                          name="combin_out", scope=tik.scope_ubuf)
        vnchw_src = self.instance.Tensor(self.dtype, (16 * self.out_box_gm.shape[1],),
                                         name="vnchw_src", scope=tik.scope_ubuf)
        vnchw_dst = self.instance.Tensor(self.dtype, (16 * self.out_box_gm.shape[1],),
                                         name="vnchw_dst", scope=tik.scope_ubuf)

        with self.instance.for_range(0, topk_num_ecah_class) as combin_index:

            with self.instance.if_scope(tik.all(self.keep_top_k > -1,
                                                topk_num_ecah_class > self.keep_top_k)):
                with self.instance.if_scope(combin_index <= self.keep_top_k):
                    combin_out[0, combin_index].set_as(topk3_out_ub[combin_index, 0])
                    combin_out[1, combin_index].set_as(topk3_out_ub[combin_index, 1])
                    combin_out[2, combin_index].set_as(box_data_ub[combin_index, 4])
            with self.instance.else_scope():
                combin_out[0, combin_index].set_as(box_data_ub[combin_index, 6])
                combin_out[1, combin_index].set_as(box_data_ub[combin_index, 5])
                combin_out[2, combin_index].set_as(box_data_ub[combin_index, 4])

        self.instance.vextract(combin_out[3, 0], box_data_ub,
                               self.out_box_gm.shape[1]//16, 0)
        self.instance.vextract(combin_out[4, 0], box_data_ub,
                               self.out_box_gm.shape[1]//16, 1)
        self.instance.vadds(self.mask,
                            combin_out[3, 0], combin_out[3, 0],
                            -1.0,
                            self.out_box_gm.shape[1] // self.mask,
                            1, 1, 8, 8)
        self.instance.vadds(self.mask,
                            combin_out[4, 0], combin_out[4, 0],
                            -1.0,
                            self.out_box_gm.shape[1] // self.mask,
                            1, 1, 8, 8)

        self.instance.vextract(combin_out[5, 0], box_data_ub,
                               self.out_box_gm.shape[1]//16, 2)
        self.instance.vextract(combin_out[6, 0], box_data_ub,
                               self.out_box_gm.shape[1]//16, 3)

        self.instance.data_move(vnchw_src[0], combin_out[0, 0], 0,
                                self.out_box_gm.shape[1] // self.burnest_len,
                                1, 0, 15)
        self.instance.data_move(vnchw_src[16], combin_out[1, 0], 0,
                                self.out_box_gm.shape[1] // self.burnest_len,
                                1, 0, 15)
        self.instance.data_move(vnchw_src[32], combin_out[2, 0], 0,
                                self.out_box_gm.shape[1] // self.burnest_len,
                                1, 0, 15)
        self.instance.data_move(vnchw_src[48], combin_out[3, 0], 0,
                                self.out_box_gm.shape[1] // self.burnest_len,
                                1, 0, 15)
        self.instance.data_move(vnchw_src[64], combin_out[4, 0], 0,
                                self.out_box_gm.shape[1] // self.burnest_len,
                                1, 0, 15)
        self.instance.data_move(vnchw_src[80], combin_out[5, 0], 0,
                                self.out_box_gm.shape[1] // self.burnest_len,
                                1, 0, 15)
        self.instance.data_move(vnchw_src[96], combin_out[6, 0], 0,
                                self.out_box_gm.shape[1] // self.burnest_len,
                                1, 0, 15)

        length = self.out_box_gm.shape[1] * 16 // 16
        tail_loop_times = ((length*16)//(16*16)) % 255

        src_list = [vnchw_src[16*i] for i in range(16)]
        dst_list = [vnchw_dst[16*i] for i in range(16)]
        self.instance.vnchwconv(False, False, dst_list, src_list,
                                tail_loop_times, 16, 16)

        with self.instance.if_scope(tik.all(self.keep_top_k > -1,
                                            topk_num_ecah_class > self.keep_top_k)):
            if self.keep_top_k > 0:
                with self.instance.for_range(0, self.keep_top_k) as index:
                    self.instance.data_move(self.out_box_gm[batch, index, 0],
                                            vnchw_dst[index*16], 0, 1, 1, 0, 0)
        with self.instance.else_scope():
            with self.instance.for_range(0, topk_num_ecah_class) as index:
                self.instance.data_move(self.out_box_gm[batch, index, 0],
                                        vnchw_dst[index*16], 0, 1, 1, 0, 0)

    def sort_each_class(self, batch, topk1_data_num, topk1_out_actual_num):
        """
        sort box

        Parameters
        ----------
        batch: batch num
        topk1_data_num: topk data num

        Returns
        -------
        None
        """
        topk_input_data = {
            "proposal_num": topk1_data_num,
            "k": self.top_k,
            "score_threshold": self.confidence_threshold,
            "regions_orig": self.topk1_in_gm,
            "mem_swap": self.topk1_swap_gm,
        }

        topk_out_data = {
            "batch_id": batch,
            "regions_sorted": self.topk1_out_gm,
            "proposal_actual_num": topk1_out_actual_num,
        }

        topk.tik_topk(self.instance, topk_input_data, topk_out_data)

    def nms_each_class(self, batch, class_index, topk1_out_actual_num):
        """
        nms box

        Parameters
        ----------
        batch: batch num
        class_index: class index

        Returns
        -------
        None
        """
        input_offset = batch*(((self.top_k+15)//16)*16 + 4)*8
        image_info = (817.55, 40)
        nms.cce_nms((self.dtype, self.ub_size,
                     self.nms_threshold, batch,
                     self.top_k, self.post_nms_topn,
                     input_offset, image_info,
                     self.instance, self.num_classes, class_index, batch),
                    self.nms_swap_gm,
                    self.topk1_out_gm,
                    topk1_out_actual_num,
                    self.nms_box_num_gm, self.nms_box_gm, False, used_in_ssd=True)

    def get_nms_all_class_result(self, batch, topk_num_ecah_class):
        """
        handle nms result

        Parameters
        ----------
        batch: batch num
        topk_num_ecah_class: nms result

        Returns
        -------
        None
        """
        with self.instance.if_scope(tik.all(self.keep_top_k > -1,
                                            topk_num_ecah_class > self.keep_top_k)):
            with self.instance.new_stmt_scope():
                topk2_tail_init_tmp_ub = self.instance.Tensor(
                    self.dtype, (128, ), name="topk2_in_data_tmp_ub", scope=tik.scope_ubuf)
                self.instance.vector_dup(self.mask, topk2_tail_init_tmp_ub, 0,
                                         128 // self.mask, 1, 8)

                topk2_tail_num = self.instance.Scalar("int32", "topk_num_ecah_class", 16)
                burst_tail_scalar = self.instance.Scalar("int32", "burst_tail_scalar", 0)
                self.get_tersor_data_burst_val(True, topk2_tail_num, burst_tail_scalar)

                self.instance.data_move(self.topk2_in_gm[batch, topk_num_ecah_class, 0],
                                        topk2_tail_init_tmp_ub,
                                        0, 1, burst_tail_scalar, 0, 0)
                self.instance.data_move(self.topk3_in_gm[batch, topk_num_ecah_class, 0],
                                        topk2_tail_init_tmp_ub,
                                        0, 1, burst_tail_scalar, 0, 0)

            self.sort_all_class(batch)
            self.sort_for_get_label(batch)

        with self.instance.new_stmt_scope():
            #set out box num and tensor
            out_box_num_ub = self.instance.Tensor(
                "int32", (8, ), name="out_box_num_ub", scope=tik.scope_ubuf)
            with self.instance.if_scope(tik.all(self.keep_top_k > -1,
                                                topk_num_ecah_class > self.keep_top_k)):
                out_box_num_ub[0].set_as(self.topk2_num)
            with self.instance.else_scope():
                out_box_num_ub[0].set_as(topk_num_ecah_class)
            self.instance.data_move(self.out_box_num_gm[batch, 0], out_box_num_ub,
                                    0, 1, 1, 0, 0)

        self.adjust_topk_crood(batch, topk_num_ecah_class)


    def get_topk_target_info(self, batch, topk_src_data):
        """
        get box result

        Parameters
        ----------
        batch: batch num
        topk_src_data: bbox data

        Returns
        -------
        None
        """
        topk_num_ecah_class = self.instance.Scalar(dtype="int32",
                                                   name="topk_num_ecah_class",
                                                   init_value=0)
        with self.instance.new_stmt_scope():
            topk2_init_tmp_ub = self.instance.Tensor(
                self.dtype, (128, ), name="topk2_init_tmp_ub", scope=tik.scope_ubuf)
            self.instance.vector_dup(self.mask, topk2_init_tmp_ub, 0,
                                     128 // self.mask, 1, 8)

            move_loops = self.topk2_in_gm.shape[1] // 16
            with self.instance.for_range(0, move_loops) as move_index:
                move_offset = 16 * move_index
                self.instance.data_move(self.topk2_in_gm[batch, move_offset, 0],
                                        topk2_init_tmp_ub, 0, 1,
                                        128 // self.burnest_len, 0, 0)
                self.instance.data_move(self.topk3_in_gm[batch, move_offset, 0],
                                        topk2_init_tmp_ub, 0, 1,
                                        128 // self.burnest_len, 0, 0)

        topk1_out_actual_num = self.instance.Scalar("int32",
                                                    "topk1_out_actual_num",
                                                    0)
        with self.instance.for_range(0, self.num_classes) as class_index:

            with self.instance.if_scope(class_index != self.background_label_id):
                self.sort_each_class_prepare(batch, class_index, topk_src_data)
                self.sort_each_class(batch, topk_src_data.shape[2]-self.burnest_len,
                                     topk1_out_actual_num)
                self.nms_each_class(batch, class_index, topk1_out_actual_num)
                self.sort_all_class_prepare(batch, class_index, topk_num_ecah_class)

                with self.instance.new_stmt_scope():
                    nms_bbox_num_ub = self.instance.Tensor("int32", (8, ),
                                                           name="nms_bbox_num_ub",
                                                           scope=tik.scope_ubuf)
                    self.instance.data_move(nms_bbox_num_ub,
                                            self.nms_box_num_gm[class_index, batch, 0],
                                            0, 1, 1, 0, 0)

                    topk_num_ecah_class_ub = self.instance.Tensor(
                        "int32", (8, ),
                        name="topk_num_ecah_class_ub",
                        scope=tik.scope_ubuf)
                    topk_num_ecah_class_ub[0].set_as(topk_num_ecah_class)

                    topk_num_ecah_class_vadd_ub = self.instance.Tensor(
                        "int32", (8, ),
                        name="topk_num_ecah_class_vadd_ub",
                        scope=tik.scope_ubuf)

                    self.instance.vadd(1, topk_num_ecah_class_vadd_ub,
                                       nms_bbox_num_ub, topk_num_ecah_class_ub,
                                       1, 1, 1, 1, 0, 0, 0)

                    topk_num_ecah_class.set_as(topk_num_ecah_class_vadd_ub[0])

        self.get_nms_all_class_result(batch, topk_num_ecah_class)

    def sort_all_class(self, batch):
        """
        sort box

        Parameters
        ----------
        batch: batch num
        Returns
        -------
        None
        """
        topK_in_num = self.keep_top_k
        if self.keep_top_k <= 0:
            topK_in_num = self.top_k
        topk_input_data = {
            "proposal_num": self.topk2_in_gm.shape[1],
            "k": topK_in_num,
            "score_threshold": self.confidence_threshold,
            "regions_orig": self.topk2_in_gm,
            "mem_swap": self.topk2_swap_gm,
        }

        topk_out_data = {
            "batch_id": batch,
            "regions_sorted": self.out_box_gm,
            "proposal_actual_num": self.topk2_num,
        }

        topk.tik_topk(self.instance, topk_input_data, topk_out_data)


    def sort_for_get_label(self, batch):
        """
        sort box

        Parameters
        ----------
        batch: batch num
        Returns
        -------
        None
        """

        topK_in_num = self.keep_top_k
        if self.keep_top_k <= 0:
            topK_in_num = self.top_k

        topk_input_data = {
            "proposal_num": self.topk3_in_gm.shape[1],
            "k": topK_in_num,
            "score_threshold": self.confidence_threshold,
            "regions_orig": self.topk3_in_gm,
            "mem_swap": self.topk3_swap_gm,
        }

        topk_out_data = {
            "batch_id": batch,
            "regions_sorted": self.topk3_out_gm,
            "proposal_actual_num": self.topk3_num,
        }

        topk.tik_topk(self.instance, topk_input_data, topk_out_data)
