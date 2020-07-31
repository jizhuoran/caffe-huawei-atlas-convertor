#!/usr/bin/env python
# -*- coding:utf-8 -*-
# pylint: disable=too-many-lines
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

yolo_v3_detection_output
"""
from te import tik
from topi.cce import util
from impl import common_util as common
from impl import constant_util as constant
from impl import yolo_v3_cls_prob as cls
from te import platform as tbe_platform

PRE_NMS_TOPN = 1024

UB_NUM = 10240


# pylint: disable=invalid-name, too-many-locals, too-many-arguments
# pylint: disable=unused-argument
@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict, dict,
                       dict, dict, dict, dict, dict, dict, dict, dict, dict,
                       (tuple, list), (tuple, list), (tuple, list), int, int,
                       int, bool, (float, int), (float, int), (float, int),
                       (float, int), (float, int), str)
def yolo_v3_detection_output_d(coord_data_low_dic, coord_data_mid_dic,
                               coord_data_high_dic, obj_prob_low_dic,
                               obj_prob_mid_dic,
                               obj_prob_high_dic, classes_prob_low_dic,
                               classes_prob_mid_dic, classes_prob_high_dic,
                               img_info_dic, windex1_dic, windex2_dic,
                               windex3_dic, hindex1_dic, hindex2_dic,
                               hindex3_dic, box_out_dic, box_out_num_dic,
                               biases_low, biases_mid, biases_high, boxes=3,
                               coords=4,
                               classes=80, relative=True, obj_threshold=0.5,
                               post_nms_topn=1024, score_threshold=0.5,
                               iou_threshold=0.45, pre_nms_topn=512,
                               kernel_name="yolo_v3_detection_output_d"):
    """
      yolov3_detection_output

      Parameters
      ----------
      coord_data_low_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      coord_data_mid_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      coord_data_high_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      obj_prob_low_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      obj_prob_mid_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      obj_prob_high_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      classes_prob_low_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      classes_prob_mid_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      classes_prob_high_dic:_dic dict, shape, dtype:fp16,fp32 format:only support NCHW
      img_info_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      windex1_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      windex2_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      windex3_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      hindex1_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      hindex2_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      hindex3_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      box_out_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      box_out_num_dic: dict, shape, dtype:fp16,fp32 format:only support NCHW
      biases_low: box1's biases
      biases_mid: box2's biases
      biases_high: box3's biases
      boxes: number of boxes
      coords: number of coordinates
      classes: number of classes
      relative:
      obj_threshold: threshold of probability of objects
      score_threshold: threshold for each category
      post_nms_topn: after nms, return posttopk boxes
      iou_threshold: nms threshold
      pre_nms_topn: for each category,take the number of pre nms topn
                    before processing, and the maximum is 1024
      kernel_name: kernel_name

      Returns
      -------
      tik_instance: tik_instance
  """
    batch = coord_data_low_dic['shape'][0]
    h1 = windex1_dic['shape'][0]
    w1 = windex1_dic['shape'][1]
    h2 = windex2_dic['shape'][0]
    w2 = windex2_dic['shape'][1]
    h3 = windex3_dic['shape'][0]
    w3 = windex3_dic['shape'][1]
    dtype = windex3_dic['dtype']
    box1_info = {"shape": (batch, boxes * (4 + 1 + classes), h1, w1),
                 "dtype": dtype, "format": "NCHW"}
    box2_info = {"shape": (batch, boxes * (4 + 1 + classes), h2, w2),
                 "dtype": dtype, "format": "NCHW"}
    box3_info = {"shape": (batch, boxes * (4 + 1 + classes), h3, w3),
                 "dtype": dtype, "format": "NCHW"}
    input_dict = {
        "box1_info": box1_info,
        "box2_info": box2_info,
        "box3_info": box3_info,
        "biases1": biases_low,
        "biases2": biases_mid,
        "biases3": biases_high,
        "coords": coords,
        "boxes": boxes,
        "classes": classes,
        "relative": relative,
        "obj_threshold": obj_threshold,
        "classes_threshold": score_threshold,
        "post_top_k": post_nms_topn,
        "nms_threshold": iou_threshold,
        "pre_nms_topn": pre_nms_topn,
        "max_box_number_per_batch": post_nms_topn,
        "kernel_name": kernel_name,
    }

    cls.check_param(input_dict)
    detection_output = DetectionOutput(input_dict)
    tik_instance = detection_output.compute_detection_output(kernel_name)

    return tik_instance


# pylint: disable=too-many-ancestors,too-many-public-methods
class DetectionOutput(cls.ClsProbComputer):
    """
    Function: use to process DetectionOutput
    Modify : 2019-11-06
    """

    def __init__(self, input_dict):
        """
          init the detection output parameters

          Parameters
          ----------
          input_dict: input_dict is a dict, the keys as follow:
                      box1_info,box2_info,box3_info,biases1,biases2,biases3,
                      coords,boxes,classes,relative,obj_threshold,post_top_k,
                      post_top_k,nms_threshold,pre_nms_topn,
                      max_box_number_per_batch,kernel_name, for more details,
                      please check the yolov3_detection_output function

          Returns
          -------
          None
          """
        super(DetectionOutput, self).__init__(input_dict)
        self.max_ub_num = UB_NUM
        if tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) // 1024 < 200 \
                or self.dtype == constant.DATA_TYPE_FP32:
            self.max_ub_num = UB_NUM // 2

        self.obj_num = self.boxes * (self.height1 * self.width1 + \
                                     self.height2 * self.width2 + \
                                     self.height3 * self.width3)
        self.bbox = self.instance.Tensor(self.dtype, (
            self.batch, self.max_box_number_per_batch * 6), \
                                         name="box_out", scope=tik.scope_gm)
        self.bbox_num = self.instance.Tensor("int32", (self.batch, 8), \
                                             name="box_out_num",
                                             scope=tik.scope_gm)
        
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") not in ( \
                "Ascend310","Ascend910", \
                "Hi3796CV300ES") \
                and self.obj_data.size // (8 * self.dsize) > self.max_ub_num:
            each_loop = (8 * self.dsize)
            shape = (self.obj_data.size // each_loop + \
                     (each_loop - 1)) // each_loop * each_loop
            if constant.DATA_TYPE_FP32 == self.dtype:
                self.mask_gm = self.instance.Tensor("uint32",
                                                    (self.batch, shape),
                                                    name="mask_gm",
                                                    is_workspace=True,
                                                    scope=tik.scope_gm)
            else:
                self.mask_gm = self.instance.Tensor("uint16",
                                                    (self.batch, shape),
                                                    name="mask_gm",
                                                    is_workspace=True,
                                                    scope=tik.scope_gm)

    def compute_detection_output(self, kernel_name):
        """
          compute detection output is main function of the detection output

          Parameters
          ----------
           None

          Returns
          -------
          None
          """
        with self.instance.for_range(0, self.block_num,
                                     block_num=self.block_num) as block_i:
            image_ub = self.instance.Tensor(self.dtype,
                                            [constant.BLOCK_SIZE // self.dsize],
                                            scope=tik.scope_ubuf,
                                            name="image_ub")
            batch = self.instance.Scalar("int32")
            with self.instance.for_range(0, self.outer_loop) as outer_i:
                batch.set_as(block_i * self.outer_loop + outer_i)
                param = {}
                self.init_param(batch, param)
                self.instance.data_move(image_ub, self.img_info[batch * 4],
                                        constant.SID, constant.DEFAULT_NBURST,
                                        constant.DEFAULT_BURST_LEN, 0, 0)
                self.correct_box(batch, image_ub)
                self.cls_prob(batch, param)
                self.multi_class(batch, image_ub, param)
            if self.outer_tail > 0:
                with self.instance.if_scope(block_i < self.outer_tail):
                    batch.set_as(self.block_num * self.outer_loop + block_i)
                    self.instance.data_move(image_ub, self.img_info[batch * 4],
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            constant.DEFAULT_BURST_LEN, 0, 0)
                    param = {}
                    self.init_param(batch, param)
                    self.correct_box(batch, image_ub)
                    self.cls_prob(batch, param)
                    self.multi_class(batch, image_ub, param)

        self.instance.BuildCCE(kernel_name=kernel_name,
                               inputs=(self.coord_data1, self.coord_data2,
                                       self.coord_data3, self.obj_data1,
                                       self.obj_data2, self.obj_data3,
                                       self.classes_data1, self.classes_data2,
                                       self.classes_data3, self.img_info,
                                       self.windex1,
                                       self.windex2, self.windex3, self.hindex1,
                                       self.hindex2, self.hindex3),
                               outputs=(self.bbox, self.bbox_num),
                               enable_l2=False)

        return self.instance

    def init_param(self, batch, param):
        """
          init some parameters

          Parameters
          ----------
          batch: the sequence of the photo
          param: param is an empty dict

          Returns
          -------
          None
          """
        index_ub = None
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ( \
            "Ascend310","Ascend910", \
            "Hi3796CV300ES"):
            index_ub = self.instance.Tensor("int32", (PRE_NMS_TOPN,),
                                            name="index_ub",
                                            scope=tik.scope_ubuf)
        index_offset = self.instance.Scalar("int32")
        index_offset.set_as(0)
        count = self.instance.Scalar("int32")
        count.set_as(0)
        param["index_offset"] = index_offset
        param["count"] = count
        param["index_ub"] = index_ub
        obj_gm_offset = self.instance.Scalar("int32")
        obj_gm_offset.set_as(batch * self.obj_num)
        param['obj_gm_offset'] = obj_gm_offset

    def multi_class(self, batch, image_ub, param):
        """
          do multi class

          Parameters
          ----------
          batch: the photo number
          image_ub: is a tensor, which store image info's weight and height
          param: param is a dict, the keys as follow:
                 index_offset: index_offset of objects
                 count: the number of filtered boxes before doing Iou
                 index_ub: is a tensor,which used to store filted obj index

          Returns
          -------
          None
          """
        proposals_ub = self.instance.Tensor(self.dtype, (PRE_NMS_TOPN * 8,),
                                            name="proposals_ub",
                                            scope=tik.scope_ubuf)
        mask = None
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") not in ( \
            "Ascend310","Ascend910", \
            "Hi3796CV300ES"):
            dtype = "uint16"
            if self.dtype == constant.DATA_TYPE_FP32:
                dtype = "uint32"
            mask = self.instance.Tensor(dtype, (self.max_ub_num,),
                                        name="mask", scope=tik.scope_ubuf)

            self.instance.vec_dup(constant.VECTOR_BYTE_SIZE // self.dsize,
                                  mask, constant.INT_DEFAULT_ZERO,
                                  self.max_ub_num * self.dsize //
                                  constant.VECTOR_BYTE_SIZE,
                                  constant.REPEAT_STRIDE_EIGHT)
        obj_total = self.instance.Scalar("int32")
        obj_total.set_as(self.dsize * self.max_ub_num)
        param["obj_total"] = obj_total
        param["batch"] = batch
        use_gm_mask = self.instance.Scalar("int32")
        use_gm_mask.set_as(0)
        param["use_gm_mask"] = use_gm_mask
        with self.instance.new_stmt_scope():
            xyhw_ub = self.instance.Tensor(self.dtype, (4, PRE_NMS_TOPN),
                                           name="xyhw_ub",
                                           scope=tik.scope_ubuf)
            if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") not in ( \
                "Ascend310","Ascend910", \
                "Hi3796CV300ES"):
                self.filter_obj(mask, xyhw_ub, param)
            loop_cycle, ub_num, last_ub_num = self.get_loop_param(
                param["index_offset"])
            param["loop_cycle"] = loop_cycle
            param["ub_num"] = ub_num
            param["last_ub_num"] = last_ub_num
            if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ( \
                "Ascend310","Ascend910", \
                "Hi3796CV300ES"):
                self.get_xyhw_by_index(xyhw_ub, param)
            x1y1x2y2_ub = self.instance.Tensor(self.dtype, (4, PRE_NMS_TOPN),
                                               name="x1y1x2y2_ub",
                                               scope=tik.scope_ubuf)
            with self.instance.if_scope(param["count"] > 0):
                self.get_x1y1x2y2(xyhw_ub, x1y1x2y2_ub, param)
                self.concatx1y1x2y2(x1y1x2y2_ub, proposals_ub, param)
                param["image_ub"] = image_ub
            self.process_each_class(proposals_ub, mask, param)

    def filter_obj(self, mask, xyhw_ub, param):
        """
          filter object

          Parameters
          ----------
          mask: a tensor which used to store filtered obj's mask
          xyhw_ub: is a tensor, which store coordinate of x,y,h,w
          param: param is a dict, the keys as follow:
                 batch: the number of photo
                 obj_total: a scalar

          Returns
          -------
          None
          """
        obj_loop_times, ub_size, obj_last_ub_size = \
            cls.get_loop_param(self.obj_num, self.max_ub_num)
        offset = self.instance.Scalar("int32")
        offset.set_as(param["batch"] * self.obj_num)
        ub_num = self.instance.Scalar("int32")
        ub_num.set_as(ub_size)
        index_offset_x = self.instance.Scalar("int32")
        index_offset_y = self.instance.Scalar("int32")
        index_offset_h = self.instance.Scalar("int32")
        index_offset_w = self.instance.Scalar("int32")
        index_offset_x.set_as(param["batch"] * self.coords * self.obj_num)
        index_offset_y.set_as(index_offset_x + self.obj_num)
        index_offset_h.set_as(index_offset_y + self.obj_num)
        index_offset_w.set_as(index_offset_h + self.obj_num)
        mask_offset = self.instance.Scalar("int32")
        mask_offset.set_as(0)
        mask_cycle = self.instance.Scalar("int32")
        mask_cycle.set_as(0)
        param["obj_total"].set_as(ub_size * self.dsize)
        mask_total = self.obj_num // (8 * self.dsize)
        if self.obj_num % (8 * self.dsize) != 0:
            mask_total = mask_total + 1
        mask_loop, mask_size, mask_last_ub = \
            cls.get_loop_param(mask_total, self.max_ub_num)
        param["mask_size"] = mask_size
        param["mask_loop"] = mask_loop
        param["mask_last_ub"] = mask_last_ub
        param["obj_loop_times"] = obj_loop_times
        with self.instance.for_range(0, obj_loop_times) as cycle:
            obj_ub = self.instance.Tensor(self.dtype, (self.max_ub_num,),
                                          name="obj_ub",
                                          scope=tik.scope_ubuf)
            with self.instance.if_scope(param["count"] < PRE_NMS_TOPN):
                with self.instance.if_scope(cycle == obj_loop_times - 1):
                    param["obj_total"].set_as(obj_last_ub_size * self.dsize)
                    ub_num.set_as(obj_last_ub_size)
                nburst = common.get_datamove_nburst(self.instance,
                                                    param["obj_total"])
                self.instance.data_move(obj_ub, self.obj_data[offset],
                                        constant.SID,
                                        constant.DEFAULT_NBURST, nburst,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
                param["cycle"] = cycle
                param["index_offset_x"] = index_offset_x
                param["index_offset_y"] = index_offset_y
                param["index_offset_h"] = index_offset_h
                param["index_offset_w"] = index_offset_w
                param["mask_offset"] = mask_offset
                param["mask_cycle"] = mask_cycle
                self.get_xyhw(obj_ub, mask, xyhw_ub, param)
                offset.set_as(offset + ub_num)

    # pylint: disable=too-many-statements
    def get_xyhw(self, obj_ub, mask, xyhw_ub, param):
        """
          ge filtered coordinates of x,y,h,w

          Parameters
          ----------
          obj_ub: a tensor which used to store obj
          mask: a tensor which used to store filtered obj's mask
          xyhw_ub: is a tensor, which store coordinate of x,y,h,w
          param: param is a dict, the keys as follow:
                 mask_offset:  a scalar used to store mask_offset
                 mask_cycle: a scalar used to store mask_cycle
                 count: the number of boxes before Iou

          Returns
          -------
          None
          """
        if self.obj_data.size // (8 * self.dsize) > self.max_ub_num:
            mod = param["mask_offset"] % self.max_ub_num
            with self.instance.if_scope(
                    tik.all(mod == 0, param["mask_offset"] != 0)):
                offset = param["mask_cycle"] * self.max_ub_num
                nburst = self.instance.Scalar("int32")
                nburst.set_as(
                    self.max_ub_num * self.dsize // constant.BLOCK_SIZE)
                with self.instance.if_scope(
                        param["mask_loop"] - 1 == param["mask_cycle"]):
                    nburst.set_as(
                        common.get_datamove_nburst(self.instance,
                                                   param[
                                                       "mask_last_ub"] * self.dsize))
                self.instance.data_move(self.mask_gm[param["batch"], offset],
                                        mask, constant.SID,
                                        constant.DEFAULT_NBURST, nburst,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
                param["use_gm_mask"].set_as(1)
                param["mask_cycle"].set_as(param["mask_cycle"] + 1)
                param["mask_offset"].set_as(0)
        repeats = common.get_vector_repeat_times(self.instance,
                                                 param["obj_total"])
        self.instance.vcmpvs_gt(mask[param["mask_offset"]], obj_ub[0],
                                self.obj_threshold,
                                repeats, 1, 8)
        if self.obj_data.size // (8 * self.dsize) > self.max_ub_num:
            with self.instance.if_scope(
                    tik.all(param["obj_loop_times"] - 1 == param["cycle"],
                            param["mask_offset"] != 0)):
                offset = param["mask_cycle"] * self.max_ub_num
                nburst = common.get_datamove_nburst(self.instance,
                                                    param[
                                                        "mask_last_ub"] * self.dsize)
                self.instance.data_move(self.mask_gm[param["batch"], offset],
                                        mask, constant.SID,
                                        constant.DEFAULT_NBURST, nburst,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
        coords_ub_xyhw = self.instance.Tensor(self.dtype, (4, self.max_ub_num),
                                              name="coords_ub_xyhw",
                                              scope=tik.scope_ubuf)
        param["loop_offset"] = param["cycle"] * self.max_ub_num
        self.data_move_xyhw(coords_ub_xyhw, param)
        scalar = self.instance.Scalar("uint32")
        mask_tmp = self.instance.Scalar("int32")
        mask_tmp.set_as(param["obj_total"] // self.dsize)
        with self.instance.if_scope(
                param["count"] * self.dsize % constant.BLOCK_SIZE == 0):
            reduce_xyhw = self.instance.Tensor(self.dtype, (self.max_ub_num,),
                                               name="reduce_xyhw",
                                               scope=tik.scope_ubuf)
            self.instance.vreduce(mask_tmp, reduce_xyhw,
                                  coords_ub_xyhw[0, 0],
                                  mask[param["mask_offset"]],
                                  repeats, 1, 8, 1, 0, scalar,
                                  mask_mode="counter")
            with self.instance.if_scope(param["count"] + scalar > PRE_NMS_TOPN):
                total_size = (PRE_NMS_TOPN - param["count"]) * self.dsize
                nburst = common.get_datamove_nburst(self.instance, total_size)
                self.instance.data_move(xyhw_ub[0, param["count"]], reduce_xyhw,
                                        0, 1,
                                        nburst, 0, 0)
                self.instance.vreduce(mask_tmp, reduce_xyhw,
                                      coords_ub_xyhw[1, 0],
                                      mask[param["mask_offset"]], repeats, 1,
                                      8, 1, 0, scalar, mask_mode="counter")
                self.instance.data_move(xyhw_ub[1, param["count"]], reduce_xyhw,
                                        0, 1,
                                        nburst, 0, 0)
                self.instance.vreduce(mask_tmp, reduce_xyhw,
                                      coords_ub_xyhw[2, 0],
                                      mask[param["mask_offset"]], repeats, 1,
                                      8, 1, 0, scalar, mask_mode="counter")
                self.instance.data_move(xyhw_ub[2, param["count"]], reduce_xyhw,
                                        0, 1,
                                        nburst, 0, 0)
                self.instance.vreduce(mask_tmp, reduce_xyhw,
                                      coords_ub_xyhw[3, 0],
                                      mask[param["mask_offset"]],
                                      repeats, 1, 8, 1, 0, scalar,
                                      mask_mode="counter")
                self.instance.data_move(xyhw_ub[3, param["count"]], reduce_xyhw,
                                        0, 1,
                                        nburst, 0, 0)
                param["count"].set_as(PRE_NMS_TOPN)
            with self.instance.else_scope():
                self.instance.vreduce(mask_tmp, xyhw_ub[0, param["count"]],
                                      coords_ub_xyhw[0, 0],
                                      mask[param["mask_offset"]],
                                      repeats, 1, 8, 1, 0, scalar,
                                      mask_mode="counter")
                self.instance.vreduce(mask_tmp, xyhw_ub[1, param["count"]],
                                      coords_ub_xyhw[1, 0],
                                      mask[param["mask_offset"]],
                                      repeats, 1, 8, 1, 0, scalar,
                                      mask_mode="counter")
                self.instance.vreduce(mask_tmp, xyhw_ub[2, param["count"]],
                                      coords_ub_xyhw[2, 0],
                                      mask[param["mask_offset"]],
                                      repeats, 1, 8, 1, 0, scalar,
                                      mask_mode="counter")
                self.instance.vreduce(mask_tmp, xyhw_ub[3, param["count"]],
                                      coords_ub_xyhw[3, 0],
                                      mask[param["mask_offset"]],
                                      repeats, 1, 8, 1, 0, scalar,
                                      mask_mode="counter")
                param["count"].set_as(param["count"] + scalar)
        with self.instance.else_scope():
            param["repeats"] = repeats
            param["scalar"] = scalar
            param["mask_tmp"] = mask_tmp
            self.get_nonaligned_xyhw(mask, coords_ub_xyhw, xyhw_ub, param)
        param["mask_offset"].set_as(
            param["mask_offset"] + self.max_ub_num // (8 * self.dsize))

    def get_nonaligned_xyhw(self, mask, coords_ub_xyhw, xyhw_ub, param):
        """
          ub's offset is nonaligned, calculate the xyhw

          Parameters
          ----------
          mask: a tensor which used to store filtered obj's mask
          coords_ub_xyhw: is a tensor, which store coordinate of x,y,h,w from gm
          param: param is a dict, the keys as follow:
                 mask_tmp:  a tensor used to store mask
                 mask_offset: a scalar used to store mask_offset
                 repeats: the number of vector repeat time
                 scalar: a scalar used to store the reserved number after vreduce
                 count:the number of boxes before Iou

          Returns
          -------
          None
          """
        reduce_xyhw = self.instance.Tensor(self.dtype, (self.max_ub_num, 1),
                                           name="reduce_xyhw",
                                           scope=tik.scope_ubuf)
        self.instance.vreduce(param["mask_tmp"], reduce_xyhw,
                              coords_ub_xyhw[0, 0],
                              mask[param["mask_offset"]], param["repeats"], 1,
                              8, 1, 0, param["scalar"], mask_mode="counter")
        with self.instance.if_scope(
                param["count"] + param["scalar"] > PRE_NMS_TOPN):
            param["scalar"].set_as(PRE_NMS_TOPN - param["count"])
        with self.instance.for_range(0, param["scalar"]) as index:
            xyhw_ub[0, param["count"] + index].set_as(reduce_xyhw[index])
        scalar_tmp = self.instance.Scalar("uint32")
        self.instance.vreduce(param["mask_tmp"], reduce_xyhw,
                              coords_ub_xyhw[1, 0],
                              mask[param["mask_offset"]], param["repeats"], 1,
                              8, 1, 0, scalar_tmp, mask_mode="counter")
        with self.instance.for_range(0, param["scalar"]) as index:
            xyhw_ub[1, param["count"] + index].set_as(reduce_xyhw[index])

        self.instance.vreduce(param["mask_tmp"], reduce_xyhw,
                              coords_ub_xyhw[2, 0],
                              mask[param["mask_offset"]], param["repeats"], 1,
                              8, 1, 0, scalar_tmp, mask_mode="counter")
        with self.instance.for_range(0, param["scalar"]) as index:
            xyhw_ub[2, param["count"] + index].set_as(reduce_xyhw[index])

        self.instance.vreduce(param["mask_tmp"], reduce_xyhw,
                              coords_ub_xyhw[3, 0],
                              mask[param["mask_offset"]], param["repeats"], 1,
                              8, 1, 0, scalar_tmp, mask_mode="counter")
        with self.instance.for_range(0, param["scalar"]) as index:
            xyhw_ub[3, param["count"] + index].set_as(reduce_xyhw[index])
        param["count"].set_as(param["count"] + param["scalar"])

    def get_xyhw_by_index(self, xyhw_ub, param):
        """
          get x,y,h,w coordinate by index

          Parameters
          ----------
          xyhw_ub: is a tensor, which store filtered coordinate of x,y,h,w
          param: param is a dict, the keys as follow:
                 loop_cycle: loop cycle
                 obj_total: the number of objects
                 ub_num: the number of elements
                 last_ub_num: the number of elements of last loops
                 count:the number of boxes before Iou

          Returns
          -------
          None
          """
        param["obj_total"].set_as(param["ub_num"] * self.dsize)
        count_offset = self.instance.Scalar("int32")
        count_offset.set_as(0)
        index_offset_x = self.instance.Scalar("int32")
        index_offset_x.set_as(param["batch"] * self.coords * self.obj_num)
        index_offset_y = self.instance.Scalar("int32")
        index_offset_h = self.instance.Scalar("int32")
        index_offset_w = self.instance.Scalar("int32")
        index_offset_y.set_as(index_offset_x + self.obj_num)
        index_offset_h.set_as(index_offset_y + self.obj_num)
        index_offset_w.set_as(index_offset_h + self.obj_num)
        loop_offset = self.instance.Scalar("int32")
        loop_offset.set_as(0)
        with self.instance.for_range(0, param["loop_cycle"]) as inner_cycle:
            with self.instance.if_scope(inner_cycle == param["loop_cycle"] - 1):
                param["obj_total"].set_as(param["last_ub_num"] * self.dsize)
                param["ub_num"].set_as(param["last_ub_num"])

            coords_ub_xyhw = self.instance.Tensor(self.dtype,
                                                  (4, self.max_ub_num),
                                                  name="coords_ub_xyhw",
                                                  scope=tik.scope_ubuf)
            param["loop_offset"] = loop_offset
            param["index_offset_x"] = index_offset_x
            param["index_offset_y"] = index_offset_y
            param["index_offset_h"] = index_offset_h
            param["index_offset_w"] = index_offset_w
            self.data_move_xyhw(coords_ub_xyhw, param)
            loop_offset.set_as(loop_offset + param["ub_num"])
            loop_start = self.instance.Scalar("int32")
            loop_start.set_as(count_offset)
            with self.instance.for_range(loop_start, param["count"]) as index:
                offset = self.instance.Scalar("int32")
                offset.set_as(param["index_ub"][index])
                offset.set_as(offset - 1)
                with self.instance.if_scope(offset < loop_offset):
                    with self.instance.if_scope(offset > self.max_ub_num - 1):
                        offset.set_as(offset % self.max_ub_num)
                    data = self.instance.Scalar(self.dtype)
                    data.set_as(coords_ub_xyhw[0, offset])
                    xyhw_ub[0, index].set_as(data)
                    data.set_as(coords_ub_xyhw[1, offset])
                    xyhw_ub[1, index].set_as(data)
                    data.set_as(coords_ub_xyhw[2, offset])
                    xyhw_ub[2, index].set_as(data)
                    data.set_as(coords_ub_xyhw[3, offset])
                    xyhw_ub[3, index].set_as(data)
                    count_offset.set_as(count_offset + 1)

    def data_move_xyhw(self, coords_ub_xyhw, param):
        """
          move x,y,h,w coordinate from gm

          Parameters
          ----------
          coords_ub_xyhw: is a tensor, which store coordinate of x,y,h,w from gm
          param: param is a dict, the keys as follow:
                 index_offset_x: a scalar store index_offset_x
                 index_offset_y: a scalar store index_offset_y
                 index_offset_h: a scalar store index_offset_h
                 index_offset_w: a scalar store index_offset_w
                 obj_total:the number of objects

          Returns
          -------
          None
          """
        nburst = common.get_datamove_nburst(self.instance, param["obj_total"])
        self.instance.data_move(coords_ub_xyhw[0, 0],
                                self.inter_coords[param["index_offset_x"] +
                                                  param["loop_offset"]],
                                constant.SID, constant.DEFAULT_NBURST, nburst,
                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        self.instance.data_move(coords_ub_xyhw[1, 0],
                                self.inter_coords[param["index_offset_y"] +
                                                  param["loop_offset"]],
                                constant.SID, constant.DEFAULT_NBURST, nburst,
                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        self.instance.data_move(coords_ub_xyhw[2, 0],
                                self.inter_coords[param["index_offset_h"] +
                                                  param["loop_offset"]],
                                constant.SID, constant.DEFAULT_NBURST, nburst,
                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        self.instance.data_move(coords_ub_xyhw[3, 0],
                                self.inter_coords[param["index_offset_w"] +
                                                  param["loop_offset"]],
                                constant.SID, constant.DEFAULT_NBURST, nburst,
                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def get_x1y1x2y2(self, xyhw_ub, x1y1x2y2_ub, param):
        """
          calculate x1,y1,x2,,y2 from x,y,h,w coordinates

          Parameters
          ----------
          xyhw_ub: is a tensor, which store filtered coordinate of x,y,h,w
          param: param is a dict, the keys as follow:
                 count: the number of boxes before IOU
          Returns
          -------
          None
          """
        repeats = common.get_vector_repeat_times(self.instance,
                                                 self.dsize * param["count"])
        self.instance.vec_muls(self.mask, xyhw_ub[2, 0], xyhw_ub[2, 0], 0.5,
                               repeats, constant.REPEAT_STRIDE_EIGHT,
                               constant.REPEAT_STRIDE_EIGHT)
        self.instance.vec_muls(self.mask, xyhw_ub[3, 0], xyhw_ub[3, 0], 0.5,
                               repeats, constant.REPEAT_STRIDE_EIGHT,
                               constant.REPEAT_STRIDE_EIGHT)
        self.instance.vec_sub(self.mask, x1y1x2y2_ub[0, 0], xyhw_ub[0, 0],
                              xyhw_ub[3, 0], repeats,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT)
        self.instance.vec_add(self.mask, x1y1x2y2_ub[2, 0], xyhw_ub[0, 0],
                              xyhw_ub[3, 0], repeats,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT)
        self.instance.vec_sub(self.mask, x1y1x2y2_ub[1, 0], xyhw_ub[1, 0],
                              xyhw_ub[2, 0], repeats,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT)
        self.instance.vec_add(self.mask, x1y1x2y2_ub[3, 0], xyhw_ub[1, 0],
                              xyhw_ub[2, 0], repeats,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT)
        self.instance.vec_adds(self.mask, x1y1x2y2_ub[0, 0], x1y1x2y2_ub[0, 0],
                               1.0, repeats, constant.REPEAT_STRIDE_EIGHT,
                               constant.REPEAT_STRIDE_EIGHT)
        self.instance.vec_adds(self.mask, x1y1x2y2_ub[1, 0], x1y1x2y2_ub[1, 0],
                               1.0, repeats, constant.REPEAT_STRIDE_EIGHT,
                               constant.REPEAT_STRIDE_EIGHT)

    def concatx1y1x2y2(self, x1y1x2y2_ub, proposals_ub, param):
        """
          concat x1,y1,x2,,y2 into proposals

          Parameters
          ----------
          x1y1x2y2_ub: is a tensor, which store coordinate of x1,y1,x2,,y2
          proposals_ub: a tensor ,which is for sort
          param: param is a dict, the keys as follow:
                 count: the number of boxes before IOU
          Returns
          -------
          None
          """
        repeats = (param["count"] + 15) // 16
        self.instance.vconcat(proposals_ub, x1y1x2y2_ub[0, 0], repeats, 0)
        self.instance.vconcat(proposals_ub, x1y1x2y2_ub[1, 0], repeats, 1)
        self.instance.vconcat(proposals_ub, x1y1x2y2_ub[2, 0], repeats, 2)
        self.instance.vconcat(proposals_ub, x1y1x2y2_ub[3, 0], repeats, 3)

    def get_loop_param(self, element_num):
        """
          calculate loop parameters

          Parameters
          ----------
          element_num: total number of elements

          Returns
          -------
          loop_cycle: loop cycle
          ub_num: the number of elements of each loop
          last_ub_num: the number of elements of last loop
          """
        loop_cycle = self.instance.Scalar("int32")
        loop_cycle.set_as(element_num // self.max_ub_num)
        last_ub_num = self.instance.Scalar("int32")
        last_ub_num.set_as(element_num % self.max_ub_num)
        ub_num = self.instance.Scalar("int32")
        ub_num.set_as(self.max_ub_num)
        with self.instance.if_scope(loop_cycle == 0):
            ub_num.set_as(element_num)
        with self.instance.if_scope(last_ub_num != 0):
            loop_cycle.set_as(loop_cycle + 1)
        with self.instance.else_scope():
            last_ub_num.set_as(self.max_ub_num)

        return loop_cycle, ub_num, last_ub_num

    def process_each_class(self, proposals_ub, mask, param):
        """
          process each class

          Parameters
          ----------
          proposals_ub: a tensor,which store proposals
          mask: a tensor which store the filtered objects's mask
          param: param is a dict, the keys as follow:
                 index_offset: a scalar which store index_offset
                 batch: the number of photo
          Returns
          -------
          None
          """
        param["index_offset"].set_as(param["batch"] * self.classes \
                                     * self.obj_num)
        selected_count = self.instance.Scalar(dtype="int32")
        selected_count.set_as(0)

        count_offset = self.instance.Scalar(dtype="int32")
        offset = self.instance.Scalar(dtype="int32")
        proposals_selected = self.instance.Tensor(self.dtype, (PRE_NMS_TOPN, 8),
                                                  name="proposals_selected",
                                                  scope=tik.scope_ubuf)

        ret_label_ub = self.instance.Tensor(self.dtype, (6, PRE_NMS_TOPN),
                                            name="ret_label_ub",
                                            scope=tik.scope_ubuf)
        class_ret_ub = self.instance.Tensor(self.dtype, (128,),
                                            name="class_ret_ub",
                                            scope=tik.scope_ubuf)
        self.instance.vec_dup(self.mask, class_ret_ub, -1.0, 1,
                              constant.REPEAT_STRIDE_EIGHT)

        loop_cycle, ub_num, last_ub_num = self.get_loop_param(self.obj_num)
        param["ub_num"] = ub_num
        param["count_offset"] = count_offset
        param["offset"] = offset
        param["loop_cycle"] = loop_cycle
        param["last_ub_num"] = last_ub_num
        param["proposals_ub"] = proposals_ub
        param["mask"] = mask
        param["class_ret_ub"] = class_ret_ub
        param["ret_label_ub"] = ret_label_ub
        mask_total = self.obj_num // (8 * self.dsize)
        if self.obj_num % (8 * self.dsize) != 0:
            mask_total = mask_total + 1
        mask_loop, mask_size, mask_last_ub = self.get_loop_param(mask_total)
        param["mask_size"] = mask_size
        param["mask_loop"] = mask_loop
        param["mask_last_ub"] = mask_last_ub
        with self.instance.if_scope(param["count"] > 0):
            with self.instance.for_range(0, self.classes) as class_cycle:
                with self.instance.if_scope(
                        selected_count < self.max_box_number_per_batch):
                    param["class_cycle"] = class_cycle
                    self.nms_of_each_class(param, selected_count,
                                           proposals_selected)
        box_ub = self.instance.Tensor(constant.DATA_TYPE_INT32, (8,),
                                      name="box_ub",
                                      scope=tik.scope_ubuf)
        box_ub[0].set_as(selected_count)
        self.instance.data_move(self.bbox_num[param["batch"], 0], box_ub,
                                constant.SID, constant.DEFAULT_NBURST,
                                constant.DEFAULT_BURST_LEN,
                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        with self.instance.if_scope(selected_count > 0):
            repeats = selected_count // 16
            with self.instance.if_scope(selected_count % 16 != 0):
                repeats = repeats + 1
            ret_ub = self.instance.Tensor(self.dtype, (5, PRE_NMS_TOPN),
                                          name="ret_ub",
                                          scope=tik.scope_ubuf)
            self.instance.vextract(ret_ub[0, 0], proposals_selected[0], repeats,
                                   0)
            self.instance.vextract(ret_ub[1, 0], proposals_selected[0], repeats,
                                   1)
            self.instance.vextract(ret_ub[2, 0], proposals_selected[0], repeats,
                                   2)
            self.instance.vextract(ret_ub[3, 0], proposals_selected[0], repeats,
                                   3)
            with self.instance.for_range(0, selected_count) as index:
                scalar = self.instance.Scalar(self.dtype)
                scalar.set_as(proposals_selected[8 * index + 4])
                ret_ub[4, index].set_as(scalar)

            self.multi_class_boxes(ret_ub, param["image_ub"], selected_count)
            self.move_result_to_gm(ret_ub, param["ret_label_ub"],
                                   selected_count,
                                   param["batch"])

    def nms_of_each_class(self, param, selected_count, proposals_selected):
        """
          nms of each class

          Parameters
          ----------
          param: param is a dict, the keys as follow:
                 ub_num: a scalar which store ub_num
                 last_ub_num: a scalar which store last_ub_num
                 obj_total: the number of objects
                 count_offset: a scalar which store count_offset
                 loop_cycle: a scalar,which store loop_cycle
          selected_count:  a scalar which store the number of selected after
                           filtered by classthreshold
          proposals_selected: a tensor used to store proposals after filtered
                             by classthreshold
          Returns
          -------
          None
          """
        param["ub_num"].set_as(self.max_ub_num)
        param["obj_total"].set_as(param["ub_num"] * self.dsize)
        param["count_offset"].set_as(0)
        param["offset"].set_as(
            param["index_offset"] + param["class_cycle"] * self.obj_num)
        mask_offset = self.instance.Scalar("int32")
        mask_offset.set_as(0)
        mask_cycle = self.instance.Scalar("int32")
        mask_cycle.set_as(0)
        with self.instance.new_stmt_scope():
            classes_ub_nms = self.instance.Tensor(self.dtype, (PRE_NMS_TOPN,),
                                                  name="classes_ub_nms",
                                                  scope=tik.scope_ubuf)
            param["classes_ub_nms"] = classes_ub_nms
            with self.instance.for_range(0, param["loop_cycle"]) as inner_cycle:
                with self.instance.if_scope(
                        param["count_offset"] < PRE_NMS_TOPN):
                    with self.instance.if_scope(
                            inner_cycle == param["loop_cycle"] - 1):
                        param["obj_total"].set_as(
                            param["last_ub_num"] * self.dsize)
                        param["ub_num"].set_as(param["last_ub_num"])
                    nburst = common.get_datamove_nburst(self.instance,
                                                        param["obj_total"])
                    classes_ub = self.instance.Tensor(self.dtype,
                                                      (self.max_ub_num,),
                                                      name="classes_ub",
                                                      scope=tik.scope_ubuf)
                    self.instance.data_move(classes_ub,
                                            self.inter_classes[param["offset"]],
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            nburst
                                            , constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
                    param["offset"].set_as(param["offset"] + param["ub_num"])
                    param["classes_ub"] = classes_ub
                    param["inner_cycle"] = inner_cycle
                    param["mask_offset"] = mask_offset
                    param["mask_cycle"] = mask_cycle
                    param["index_offset"] = param["offset"] - \
                                            param["index_offset"] - \
                                            param["class_cycle"] * self.obj_num
                    self.set_class_nms(param)
            self.instance.vconcat(param["proposals_ub"],
                                  param["classes_ub_nms"],
                                  PRE_NMS_TOPN // 16, 4)
        iou_count = self.instance.Scalar(dtype="int32")
        iou_count.set_as(0)
        param["selected_count"] = selected_count
        param["iou_count"] = iou_count
        self.nms(param["proposals_ub"], proposals_selected, param)
        self.instance.vec_adds(self.mask, param["class_ret_ub"],
                               param["class_ret_ub"], 1.0,
                               1, constant.REPEAT_STRIDE_EIGHT,
                               constant.REPEAT_STRIDE_EIGHT)
        with self.instance.if_scope(iou_count > 0):
            param["offset"].set_as(selected_count - iou_count)
            scalar = self.instance.Scalar(dtype=self.dtype)
            scalar.set_as(param["class_ret_ub"][0])
            with self.instance.for_range(0, iou_count) as index:
                param["ret_label_ub"][param["offset"] + index].set_as(scalar)

    def multi_class_boxes(self, ret_ub, image_ub, selected_count):
        """
          multi class boxes

          Parameters
          ----------
          ret_ub:  a tensor,which store result
          image_ub: a tensor used to image's height and weight
          selected_count:  a scalar which store the number of selected after
                           filtered by classthreshold
          Returns
          -------
          None
          """
        image_w = self.instance.Scalar(self.dtype)
        image_w.set_as(image_ub[3])
        image_h = self.instance.Scalar(self.dtype)
        image_h.set_as(image_ub[2])
        repeats = common.get_vector_repeat_times(self.instance,
                                                 selected_count * self.dsize)
        self.instance.vec_adds(self.mask, ret_ub[0, 0], ret_ub[0, 0],
                               -1.0, repeats, constant.REPEAT_STRIDE_EIGHT,
                               constant.REPEAT_STRIDE_EIGHT)
        self.instance.vec_adds(self.mask, ret_ub[1, 0], ret_ub[1, 0],
                               -1.0, repeats, constant.REPEAT_STRIDE_EIGHT,
                               constant.REPEAT_STRIDE_EIGHT)

        self.instance.vec_muls(self.mask, ret_ub[0, 0], ret_ub[0, 0],
                               image_w, repeats, constant.REPEAT_STRIDE_EIGHT,
                               constant.REPEAT_STRIDE_EIGHT)
        self.instance.vec_muls(self.mask, ret_ub[1, 0], ret_ub[1, 0],
                               image_h, repeats, constant.REPEAT_STRIDE_EIGHT,
                               constant.REPEAT_STRIDE_EIGHT)
        self.instance.vec_muls(self.mask, ret_ub[2, 0], ret_ub[2, 0],
                               image_w, repeats, constant.REPEAT_STRIDE_EIGHT,
                               constant.REPEAT_STRIDE_EIGHT)
        self.instance.vec_muls(self.mask, ret_ub[3, 0], ret_ub[3, 0],
                               image_h, repeats, constant.REPEAT_STRIDE_EIGHT,
                               constant.REPEAT_STRIDE_EIGHT)
        threshold = self.instance.Tensor(self.dtype, (PRE_NMS_TOPN,),
                                         scope=tik.scope_ubuf,
                                         name="threshold")

        self.instance.vec_dup(self.mask, threshold[0],
                              constant.FLOAT_DEFAULT_ZERO, repeats,
                              constant.REPEAT_STRIDE_EIGHT)

        self.instance.vec_max(self.mask, ret_ub[0, 0], ret_ub[0, 0],
                              threshold[0], repeats,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT)
        self.instance.vec_max(self.mask, ret_ub[1, 0], ret_ub[1, 0],
                              threshold[0], repeats,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT)
        self.instance.vec_dup(self.mask, threshold[0], image_w, repeats,
                              constant.REPEAT_STRIDE_EIGHT)
        self.instance.vec_adds(self.mask, threshold, threshold, -1,
                               repeats, constant.REPEAT_STRIDE_EIGHT,
                               constant.REPEAT_STRIDE_EIGHT)

        self.instance.vec_min(self.mask, ret_ub[2, 0], ret_ub[2, 0],
                              threshold[0], repeats,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT)

        self.instance.vec_dup(self.mask, threshold[0], image_h,
                              repeats, constant.REPEAT_STRIDE_EIGHT)
        self.instance.vec_adds(self.mask, threshold, threshold, -1,
                               repeats, constant.REPEAT_STRIDE_EIGHT,
                               constant.REPEAT_STRIDE_EIGHT)

        self.instance.vec_min(self.mask, ret_ub[3, 0], ret_ub[3, 0],
                              threshold, repeats,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT)

    def set_class_nms(self, param):
        """
          set class nms

          Parameters
          ----------
           param: param is a dict, the keys as follow:
                 mask: a scalar which store ub_num
                 obj_total: the number of objects
                 index_ub: a tensor used to store index of objects
                 count_offset: a scalar which store count_offset
                 index_offset: a scalar,which store index_offset
                 classes_ub: a tensor used to store classes_ub
                 classes_ub_nms: a tensor used to store classes_ub_nms
          Returns
          -------
          None
          """
        repeats = common.get_vector_repeat_times(self.instance,
                                                 param["obj_total"])
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ( \
                "Ascend310","Ascend910", \
                "Hi3796CV300ES"):
            loop_start = self.instance.Scalar("int32")
            loop_start.set_as(param["count_offset"])
            with self.instance.for_range(loop_start, param["count"]) as index:
                offset = self.instance.Scalar("int32")
                offset.set_as(param["index_ub"][index])
                offset.set_as(offset - 1)
                with self.instance.if_scope(offset < param["index_offset"]):
                    with self.instance.if_scope(offset > self.max_ub_num - 1):
                        offset.set_as(offset % self.max_ub_num)

                    param["count_offset"].set_as(param["count_offset"] + 1)
                    data = self.instance.Scalar(self.dtype)
                    data.set_as(param["classes_ub"][offset])
                    param["classes_ub_nms"][index].set_as(data)
        else:
            if self.obj_data.size // (8 * self.dsize) > self.max_ub_num:
                with self.instance.if_scope(param["use_gm_mask"] == 1):
                    with self.instance.if_scope(
                            tik.any(param["mask_offset"] % self.max_ub_num == 0,
                                    param["inner_cycle"] == 0)):
                        nburst = self.instance.Scalar("int32")
                        nburst.set_as(
                            self.max_ub_num * self.dsize // constant.BLOCK_SIZE)
                        with self.instance.if_scope(
                                param["mask_loop"] - 1 == param["mask_cycle"]):
                            nburst.set_as(
                                common.get_datamove_nburst(self.instance, param[
                                    "mask_last_ub"] * self.dsize))
                        self.instance.data_move(param["mask"], self.mask_gm[
                            param["batch"], param[
                                "mask_cycle"] * self.max_ub_num],
                                                constant.SID,
                                                constant.DEFAULT_NBURST,
                                                nburst, constant.STRIDE_ZERO,
                                                constant.STRIDE_ZERO)
                        param["mask_cycle"].set_as(param["mask_cycle"] + 1)
                        param["mask_offset"].set_as(0)

            mod = param["count_offset"] * self.dsize % constant.BLOCK_SIZE
            scalar = self.instance.Scalar("uint32")
            with self.instance.if_scope(mod == 0):
                reduce_xyhw = self.instance.Tensor(self.dtype,
                                                   (self.max_ub_num,),
                                                   name="reduce_xyhw",
                                                   scope=tik.scope_ubuf)
                self.instance.vreduce(param["ub_num"], reduce_xyhw,
                                      param["classes_ub"],
                                      param["mask"][param["mask_offset"]],
                                      repeats, 1, 8, 1, 0, scalar,
                                      mask_mode="counter")
                with self.instance.if_scope(param["count_offset"] + scalar \
                                            > PRE_NMS_TOPN):
                    total_size = (PRE_NMS_TOPN - param["count_offset"]) \
                                 * self.dsize
                    nburst = common.get_datamove_nburst(self.instance,
                                                        total_size)
                    self.instance.data_move(
                        param["classes_ub_nms"][param["count_offset"]],
                        reduce_xyhw, 0, 1, nburst, 0, 0)
                    param["count_offset"].set_as(PRE_NMS_TOPN)
                with self.instance.else_scope():
                    self.instance.vreduce(param["ub_num"],
                                          param["classes_ub_nms"][
                                              param["count_offset"]],
                                          param["classes_ub"],
                                          param["mask"][param["mask_offset"]],
                                          repeats, 1, 8,
                                          1, 0, scalar, mask_mode="counter")
                    param["count_offset"].set_as(param["count_offset"] + scalar)

            with self.instance.else_scope():
                reduce_xyhw = self.instance.Tensor(self.dtype,
                                                   (self.max_ub_num, 1),
                                                   name="reduce_xyhw",
                                                   scope=tik.scope_ubuf)
                self.instance.vreduce(param["ub_num"], reduce_xyhw,
                                      param["classes_ub"],
                                      param["mask"][param["mask_offset"]],
                                      repeats, 1,
                                      8, 1, 0,
                                      scalar, mask_mode="counter")
                with self.instance.if_scope(
                        param["count_offset"] + scalar > PRE_NMS_TOPN):
                    scalar.set_as(PRE_NMS_TOPN - param["count_offset"])
                with self.instance.for_range(0, scalar) as index:
                    offset = param["count_offset"] + index
                    param["classes_ub_nms"][offset].set_as(reduce_xyhw[index])
                param["count_offset"].set_as(param["count_offset"] + scalar)

            param["mask_offset"].set_as(param["mask_offset"] + \
                                        self.max_ub_num // (8 * self.dsize))

    def nms(self, proposals_ub, proposals_selected, param):
        """
          set class nms

          Parameters
          ----------
          proposals_ub: a tensor,used to store proposals
          proposals_selected: the number of boxes after filtered by
                              class threshold
           param: param is a dict, the keys as follow:
                 count_offset: a scalar which store count_offset
          Returns
          -------
          None
          """
        selected_class = self.instance.Scalar(dtype="uint16")
        selected_class.set_as(0)
        selected_tmp = self.instance.Tensor(self.dtype, (PRE_NMS_TOPN, 8),
                                            name="selected_tmp",
                                            scope=tik.scope_ubuf)
        with self.instance.new_stmt_scope():
            self.class_filter(selected_tmp, proposals_ub, selected_class, param)
            self.sort(selected_tmp)

        with self.instance.if_scope(selected_class > 0):
            return self.nms_filter(proposals_selected, selected_tmp,
                                   param["selected_count"], selected_class,
                                   param["iou_count"])

    def class_filter(self, selected_tmp, proposals_ub, selected_class, param):
        """
      filter by class threshold

      Parameters
      ----------
      selected_tmp: a tensor,used to store proposals after filtered by class
                    threshold
      proposals_ub: a tensor,used to store proposals
      selected_class: the number of boxes after class filtered
       param: param is a dict, the keys as follow:
             count_offset: a scalar which store count_offset
      Returns
      -------
      None
      """
        ones_ub = self.instance.Tensor(self.dtype, (128,), name="ones_ub",
                                       scope=tik.scope_ubuf)
        self.instance.vec_dup(self.mask, ones_ub[0], 1, 1,
                              constant.REPEAT_STRIDE_EIGHT)
        zeros_ub = self.instance.Tensor(self.dtype, (128,), name="zeros_ub",
                                        scope=tik.scope_ubuf)
        self.instance.vec_dup(self.mask, zeros_ub[0], 0, 1,
                              constant.REPEAT_STRIDE_EIGHT)
        clsthreshold_ub = self.instance.Tensor(self.dtype, (128,),
                                               name="clsthreshold_ub",
                                               scope=tik.scope_ubuf)
        self.instance.vec_dup(self.mask, clsthreshold_ub[0],
                              self.classes_threshold, 1,
                              constant.REPEAT_STRIDE_EIGHT)
        reduce_mask_ub = self.instance.Tensor(self.dtype, (128,),
                                              name="reduce_mask_ub",
                                              scope=tik.scope_ubuf)

        repeats = common.get_vector_repeat_times(self.instance,
                                                 PRE_NMS_TOPN * self.dsize)
        with self.instance.new_stmt_scope():
            classes_ub_nms = self.instance.Tensor(self.dtype, (PRE_NMS_TOPN,),
                                                  name="classes_ub_nms",
                                                  scope=tik.scope_ubuf)
            self.instance.vec_dup(self.mask, classes_ub_nms,
                                  constant.FLOAT_DEFAULT_ZERO, repeats,
                                  constant.REPEAT_STRIDE_EIGHT)
            self.instance.vec_dup(self.mask, selected_tmp,
                                  constant.FLOAT_DEFAULT_ZERO, repeats * 8,
                                  constant.REPEAT_STRIDE_EIGHT)
            self.instance.vconcat(selected_tmp, classes_ub_nms,
                                  PRE_NMS_TOPN // 16, 4)
        param["count_offset"].set_as(param["count_offset"] * 8)
        loop_cycle = self.instance.Scalar("int32")
        loop_cycle.set_as((param["count_offset"] + 127) // 128)
        mask_scalar = self.instance.Scalar("uint16", name="mask_scalar")
        each_repeat = 128
        cmp = 15360
        if self.dtype == constant.DATA_TYPE_FP32:
            loop_cycle.set_as((param["count_offset"] + 63) // 64)
            each_repeat = 64
            cmp = 1065353216
            mask_scalar = self.instance.Scalar("uint32", name="mask_scalar")
        count_tmp = self.instance.Scalar(dtype="int32")
        count_tmp.set_as(0)
        with self.instance.for_range(0, loop_cycle) as cmp_index:
            offset = cmp_index * each_repeat
            sel = self.instance.Tensor("uint16", (8, ), name="sel",
                                       scope=tik.scope_ubuf)
            self.instance.vec_dup(8, sel, 0, 1, 8)
            self.instance.vec_cmpv_gt(sel, proposals_ub[offset],
                                      clsthreshold_ub[0], 1, 8, 8)
            self.instance.vec_sel(self.mask, 0, reduce_mask_ub[0], sel,
                                  ones_ub[0], zeros_ub[0], 1, 1)
            with self.instance.for_range(0, each_repeat // 8) as mask_index:
                mask_offset = 8 * mask_index
                count_tmp.set_as(offset + mask_offset)
                with self.instance.if_scope(count_tmp < param["count_offset"]):
                    mask_scalar.set_as(reduce_mask_ub[mask_offset + 4])
                    with self.instance.if_scope(mask_scalar == cmp):
                        if self.dtype == constant.DATA_TYPE_FP32:
                            self.instance.data_move(
                                selected_tmp[selected_class, 0],
                                proposals_ub[count_tmp],
                                constant.SID,
                                constant.DEFAULT_NBURST,
                                constant.DEFAULT_BURST_LEN,
                                constant.STRIDE_ZERO,
                                constant.STRIDE_ZERO)
                        else:
                            self.set_selected(proposals_ub, selected_tmp,
                                              count_tmp, selected_class)

                        selected_class.set_as(selected_class + 1)

        zero = self.instance.Scalar(dtype=self.dtype)
        zero.set_as(0)
        selected_tmp[selected_class, 0].set_as(zero)
        selected_tmp[selected_class, 1].set_as(zero)
        selected_tmp[selected_class, 2].set_as(zero)
        selected_tmp[selected_class, 3].set_as(zero)
        selected_tmp[selected_class, 4].set_as(zero)
        selected_tmp[selected_class, 5].set_as(zero)

    def set_selected(self, proposals_ub, selected_tmp, proposals_offset,
                     selected_offset):
        """
          set_selected after sort

          Parameters
          ----------
          proposals_ub: a tensor,used to store proposals
          selected_tmp: a tensor,used to store proposals after filtered by
                       classthreshold
          proposals_offset: a scalar which store proposals_offset
          selected_offset: a scalar which store selected_offset
          Returns
          -------
          None
          """
        data = self.instance.Scalar(self.dtype)
        data.set_as(proposals_ub[proposals_offset])
        selected_tmp[selected_offset, 0].set_as(data)
        data.set_as(proposals_ub[proposals_offset + 1])
        selected_tmp[selected_offset, 1].set_as(data)
        data.set_as(proposals_ub[proposals_offset + 2])
        selected_tmp[selected_offset, 2].set_as(data)
        data.set_as(proposals_ub[proposals_offset + 3])
        selected_tmp[selected_offset, 3].set_as(data)
        data.set_as(proposals_ub[proposals_offset + 4])
        selected_tmp[selected_offset, 4].set_as(data)
        data.set_as(proposals_ub[proposals_offset + 5])
        selected_tmp[selected_offset, 5].set_as(data)

    def nms_filter(self, proposals_selected, selected_tmp, selected_count,
                   selected_class, iou_count):
        """
          filter nms

          Parameters
          ----------
          proposals_selected: a tensor used to store proposals after filtered
                              by IOU
          selected_tmp: a tensor,used to store proposals after filtered by
                       classthreshold
          selected_count: a scalar which store selected_count
          selected_class: a scalar which store selected_class
           iou_count:
          Returns
          -------
          None
          """
        iou_num = PRE_NMS_TOPN
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ( \
                "Hi3796CV300ES") or \
                self.dtype == constant.DATA_TYPE_FP32:
            iou_num = iou_num // 2
            with self.instance.if_scope(selected_class > iou_num):
                selected_class.set_as(iou_num)
        with self.instance.if_scope(selected_class > self.pre_nms_topn):
            selected_class.set_as(self.pre_nms_topn)
        area = self.instance.Tensor(self.dtype, (iou_num,), name="area",
                                    scope=tik.scope_ubuf)
        cycle_roi = self.instance.Scalar(dtype="uint16")
        cycle_roi.set_as((selected_class + 15) // 16)
        self.instance.vrpac(area, selected_tmp, cycle_roi)
        supvec_ub = self.instance.Tensor("uint16", (iou_num,), name="supVec_ub",
                                         scope=tik.scope_ubuf)
        repeat_supvec = common.get_vector_repeat_times(self.instance,
                                                       selected_class * 2)
        self.instance.vec_dup(self.mask, supvec_ub, 1, repeat_supvec,
                              constant.REPEAT_STRIDE_EIGHT)

        tempiou_ub = self.instance.Tensor(self.dtype, (iou_num * 16,),
                                          name="tempiou_ub",
                                          scope=tik.scope_ubuf)
        tempjoin_ub = self.instance.Tensor(self.dtype, (iou_num * 16,),
                                           name="tempjoin_ub",
                                           scope=tik.scope_ubuf)
        scalar_tmp = self.instance.Scalar(dtype="int16")
        scalar_tmp.set_as(0)
        supvec_ub[0].set_as(scalar_tmp)
        tempsupmatrix_ub = self.instance.Tensor("uint16", (iou_num,),
                                                name="tempsupmatrix_ub",
                                                scope=tik.scope_ubuf)
        repeat_times = 2
        if self.dtype == constant.DATA_TYPE_FP32:
            repeat_times = 4

        with self.instance.for_range(0, cycle_roi) as count_cycle:
            offset_tmp = 16 * count_cycle
            repeat_tmp = count_cycle + 1

            self.instance.viou(tempiou_ub[0], selected_tmp[0],
                               selected_tmp[offset_tmp * 8], repeat_tmp)
            self.instance.vaadd(tempjoin_ub[0], area[0], area[offset_tmp],
                                repeat_tmp)

            self.instance.vec_muls(self.mask, tempjoin_ub[0], tempjoin_ub[0],
                                   self.nms_threshold / (1+self.nms_threshold),
                                   repeat_times * repeat_tmp,
                                   constant.REPEAT_STRIDE_EIGHT,
                                   constant.REPEAT_STRIDE_EIGHT)
            self.instance.vcmpv_gt(tempsupmatrix_ub[0], tempiou_ub[0],
                                   tempjoin_ub[0],
                                   repeat_times * repeat_tmp,
                                   constant.STRIDE_ONE,
                                   constant.STRIDE_ONE,
                                   constant.REPEAT_STRIDE_EIGHT,
                                   constant.REPEAT_STRIDE_EIGHT)
            rpn_cor_ir = self.instance.set_rpn_cor_ir(0)
            with self.instance.if_scope(count_cycle == 0):
                rpn_cor_ir = self.instance.rpn_cor(tempsupmatrix_ub[0],
                                                   supvec_ub[0],
                                                   1, 1, 1)
                self.instance.rpn_cor_diag(supvec_ub[0], tempsupmatrix_ub[0],
                                           rpn_cor_ir)
            with self.instance.else_scope():
                length = (repeat_tmp) * 16 * 16 // 16
                rpn_cor_ir = self.instance.rpn_cor(tempsupmatrix_ub[0],
                                                   supvec_ub[0],
                                                   1, 1, count_cycle + 1)
                self.instance.rpn_cor_diag(supvec_ub[offset_tmp],
                                           tempsupmatrix_ub[length - 16],
                                           rpn_cor_ir)
            with self.instance.for_range(0, 16) as j:
                tmp = 16 * count_cycle + j
                with self.instance.if_scope(tik.all(tmp + 1 <= selected_class, \
                                                    supvec_ub[tmp] == 0, \
                                                    selected_count < self.max_box_number_per_batch,
                                                    iou_count < self.post_top_k)):
                    index_tmp = tmp * 8
                    if self.dtype == constant.DATA_TYPE_FP32:
                        offset = selected_count * 8
                        self.instance.data_move(proposals_selected[offset],
                                                selected_tmp[index_tmp],
                                                constant.SID,
                                                constant.DEFAULT_NBURST,
                                                constant.DEFAULT_BURST_LEN,
                                                constant.STRIDE_ZERO,
                                                constant.STRIDE_ZERO)
                    else:
                        self.set_selected(selected_tmp, proposals_selected,
                                          index_tmp, selected_count)
                    iou_count.set_as(iou_count + 1)
                    selected_count.set_as(selected_count + 1)

    def move_result_to_gm(self, ret_ub, ret_label_ub, selected_count, batch):
        """
         move result to gm

          Parameters
          ----------
          ret_ub: a tensor used to store result
          selected_count: a scalar which store selected_count
          batch: the sequence of photo
          Returns
          -------
          None
          """
        nburst = common.get_datamove_nburst(self.instance,
                                            selected_count * self.dsize)
        offset = self.instance.Scalar(dtype="int32")
        offset.set_as(0)
        self.instance.data_move(self.bbox[batch, offset],
                                ret_ub[0, 0], constant.SID,
                                constant.DEFAULT_NBURST,
                                nburst, constant.STRIDE_ZERO,
                                constant.STRIDE_ZERO)
        offset.set_as(offset + selected_count)
        self.instance.data_move(self.bbox[batch, offset],
                                ret_ub[1, 0], constant.SID,
                                constant.DEFAULT_NBURST,
                                nburst, constant.STRIDE_ZERO,
                                constant.STRIDE_ZERO)

        offset.set_as(offset + selected_count)
        self.instance.data_move(self.bbox[batch, offset],
                                ret_ub[2, 0], constant.SID,
                                constant.DEFAULT_NBURST,
                                nburst, constant.STRIDE_ZERO,
                                constant.STRIDE_ZERO)
        offset.set_as(offset + selected_count)
        self.instance.data_move(self.bbox[batch, offset],
                                ret_ub[3, 0], constant.SID,
                                constant.DEFAULT_NBURST,
                                nburst,
                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        offset.set_as(offset + selected_count)
        self.instance.data_move(self.bbox[batch, offset],
                                ret_ub[4, 0], constant.SID,
                                constant.DEFAULT_NBURST,
                                nburst,
                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        offset.set_as(offset + selected_count)
        self.instance.data_move(self.bbox[batch, offset], ret_label_ub[0],
                                constant.SID, constant.DEFAULT_NBURST, nburst,
                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def sort(self, proposals_ub):
        """
         move result to gm

          Parameters
          ----------
          proposals_ub: a tensor used to store proposals

          Returns
          -------
          None
          """
        proposals_ub_tmp = self.instance.Tensor(self.dtype, (PRE_NMS_TOPN * 8,),
                                                name="proposals_ub_tmp",
                                                scope=tik.scope_ubuf)
        each_proposal_num = 8
        repeats = PRE_NMS_TOPN // 16
        self.instance.vrpsort16(proposals_ub_tmp, proposals_ub, repeats)
        repeats = PRE_NMS_TOPN // 64
        offset = 16 * each_proposal_num
        length = 16

        src_list = []
        src_list_lengths = []
        src_list.append(proposals_ub_tmp[0])
        src_list_lengths.append(length)
        src_list.append(proposals_ub_tmp[offset])
        src_list_lengths.append(length)
        src_list.append(proposals_ub_tmp[offset * 2])
        src_list_lengths.append(length)
        src_list.append(proposals_ub_tmp[offset * 3])
        src_list_lengths.append(length)
        self.instance.vmrgsort4(proposals_ub, src_list, src_list_lengths,
                                False,
                                15, repeats)
        repeats = repeats // 4
        offset = offset * 4
        length = length * 4
        src_list = []
        src_list_lengths = []
        src_list.append(proposals_ub[0])
        src_list_lengths.append(length)
        src_list.append(proposals_ub[offset])
        src_list_lengths.append(length)
        src_list.append(proposals_ub[offset * 2])
        src_list_lengths.append(length)
        src_list.append(proposals_ub[offset * 3])
        src_list_lengths.append(length)
        self.instance.vmrgsort4(proposals_ub_tmp, src_list, src_list_lengths,
                                False,
                                15, repeats)
        repeats = repeats // 4
        offset = offset * 4
        length = length * 4
        src_list = []
        src_list_lengths = []
        src_list.append(proposals_ub_tmp[0])
        src_list_lengths.append(length)
        src_list.append(proposals_ub_tmp[offset])
        src_list_lengths.append(length)
        src_list.append(proposals_ub_tmp[offset * 2])
        src_list_lengths.append(length)
        src_list.append(proposals_ub_tmp[offset * 3])
        src_list_lengths.append(length)
        self.instance.vmrgsort4(proposals_ub, src_list, src_list_lengths,
                                False,
                                15, repeats)
