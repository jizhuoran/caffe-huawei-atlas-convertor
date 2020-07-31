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

yolov_3_cls_prob
"""

# pylint: disable=ungrouped-imports,import-error,too-many-branches
from te import tik
from topi.cce import util
from impl.constant_util import BLOCK_SIZE, \
    VECTOR_BYTE_SIZE, STRIDE_ONE
from impl.yolo_v3_correct_region_box import GetCorrectBoxComputer
from impl import constant_util as constant
from impl import common_util as common
from te import platform as tbe_platform

# param for nms compute
PRE_NMS_TOPN = 1024

# one repeat
REPEAT_ONE = 1

# one nburst
NBURST_ONE = 1

# value one
VALUE_ONE = 1

# stride eight for ISA
STRIDE_EIGHT = 8

# stride zero for dma
GAP_ZERO = 0

# sid for dma
SID = 0

# value zero
VALUE_ZERO = 0


# pylint: disable=too-many-ancestors
class ClsProbComputer(GetCorrectBoxComputer):
    """
    Function: store yolov3 ClsProb parameters
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
        super(ClsProbComputer, self).__init__(input_dict)
        self.len_32b = BLOCK_SIZE // self.dsize
        self.len3 = self.boxes * self.height3 * self.width3
        self.tail_len = self.len3 % (self.len_32b)
        shape = self.boxes * (self.height1 * self.width1 + \
                              self.height2 * self.width2)
        
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") not in ( \
                    "Ascend310","Ascend910", \
                    "Hi3796CV300ES"):
            each_burst = constant.BLOCK_SIZE // self.dsize

            shape = ((self.boxes * self.height3 * self.width3 + each_burst - 1) \
                     // each_burst) * each_burst + shape
            shape = (shape + each_burst - 1) // each_burst * each_burst
            self.obj_data = self.instance.Tensor(self.dtype,
                                                 (self.batch, shape),
                                                 name="obj_data",
                                                 is_workspace=True,
                                                 scope=tik.scope_gm)

    def cls_prob(self, batch, param):
        """
          compute cls pro

          Parameters
          ----------
          batch: the photo number
          param: param is a dict, the keys as follow:
                 index_offset: index_offset of objects
                 count: the number of filtered boxes before doing Iou
                 index_ub: is a tensor,which used to store filted obj index
                 obj_gm_offset: a scalar,store obj_gm_offset

          Returns
          -------
          None
          """
        with self.instance.new_stmt_scope():
            self.handle_clsprob_1(batch, param)
        with self.instance.new_stmt_scope():
            self.handle_clsprob_2(batch, param)
        with self.instance.new_stmt_scope():
            self.handle_clsprob_3(batch, param)

    def handle_clsprob_1(self, batch, param):
        """
          compute cls pro of boxinfo1

          Parameters
          ----------
          batch: the photo number
          param: param is a dict, the keys as follow:
                 index_offset: index_offset of objects
                 count: the number of filtered boxes before doing Iou
                 index_ub: is a tensor,which used to store filted obj index
                 obj_gm_offset: a scalar,store obj_gm_offset

          Returns
          -------
          None
          """
        in_param = {}
        in_param["out_offset"] = VALUE_ZERO
        in_param["w"] = self.width1
        in_param["h"] = self.height1
        in_param["obj_data"] = self.obj_data1
        in_param["clz_data"] = self.classes_data1
        in_param["index_ub"] = param['index_ub']
        in_param["count"] = param['count']
        in_param["index_offset"] = param['index_offset']
        in_param["total_len"] = self.boxes * self.width1 * self.height1
        in_param["obj_gm_offset"] = param['obj_gm_offset']
        if self.boxes * self.width1 * self.height1 * self.dsize < self.one_max_size // 2:
            self.small_clsprob(batch, in_param)
        else:
            self.big_clsprob(batch, in_param)

    def set_index_ub(self, param, length):
        """
          set object index after filtered by object threshold

          Parameters
          ----------
          param: param is a dict, the keys as follow:
                 index_offset: index_offset of objects
                 reduce_mask_ub: a tensor store reduce mask
                 index_ub: a tensor, store index
                 index_offset: a scalar,store index_offset
                 count: a scalar,store the number of index
          length: the number of element

          Returns
          -------
          None
          """
        
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Ascend310", "Ascend910", \
                                                                 "Hi3796CV300ES"):
            self.set_index_ub_by_mask(param, length)

    def set_index_ub_by_mask(self, param, length):
        """
          set object index by mask after filtered by object threshold

          Parameters
          ----------
          param: param is a dict, the keys as follow:
                 index_offset: index_offset of objects
                 reduce_mask_ub: a tensor store reduce mask
                 index_ub: a tensor, store index
                 index_offset: a scalar,store index_offset
                 count: a scalar,store the number of index
          length: the number of element

          Returns
          -------
          None
          """
        sum_mask_ub = self.instance.Tensor(self.dtype, (16,),
                                           name="sum_mask_ub",
                                           scope=tik.scope_ubuf)
        work_tensor_ub = self.instance.Tensor(self.dtype, (16,),
                                              name="work_tensor_ub",
                                              scope=tik.scope_ubuf)
        self.instance.vec_reduce_add(128, sum_mask_ub, param['reduce_mask_ub'], work_tensor_ub, 1, 8)

        mask_scalar = self.instance.Scalar("uint16", name="mask_scalar")
        mask_scalar.set_as(sum_mask_ub[0])
        with self.instance.if_scope(mask_scalar != 0):
            with self.instance.if_scope(param['count'] < PRE_NMS_TOPN):
                with self.instance.for_range(0, length) as mask_index:
                    param['index_offset'].set_as(param['index_offset'] + 1)
                    with self.instance.if_scope(param['count'] < PRE_NMS_TOPN):
                        mask_scalar.set_as(param['reduce_mask_ub'][mask_index])

                        # 1 fp16 == 15360 uint16
                        with self.instance.if_scope(mask_scalar == 15360):
                            param['index_ub'][param['count']].set_as(
                                param['index_offset'])
                            param['count'].set_as(param['count'] + 1)
        with self.instance.else_scope():
            param['index_offset'].set_as(param['index_offset'] + length)

    def handle_clsprob_2(self, batch, param):
        """
          compute cls pro of boxinfo2

          Parameters
          ----------
          batch: the photo number
          param: param is a dict, the keys as follow:
                 index_offset: index_offset of objects
                 count: the number of filtered boxes before doing Iou
                 index_ub: is a tensor,which used to store filted obj index
                 obj_gm_offset: a scalar,store obj_gm_offset

          Returns
          -------
          None
          """
        in_param = {}
        in_param["out_offset"] = self.boxes * (self.height1 * self.width1)
        in_param["w"] = self.width2
        in_param["h"] = self.height2
        in_param["obj_data"] = self.obj_data2
        in_param["clz_data"] = self.classes_data2
        in_param["index_ub"] = param['index_ub']
        in_param["count"] = param['count']
        in_param["index_offset"] = param['index_offset']
        in_param["total_len"] = self.boxes * self.width2 * self.height2
        in_param["obj_gm_offset"] = param['obj_gm_offset']

        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ( \
                "Ascend310","Ascend910", \
                "Hi3796CV300ES"):
            in_param['index_offset'].set_as(
                self.boxes * self.width1 * self.height1)
        if self.boxes * self.width2 * self.height2 * self.dsize < \
                self.one_max_size // 2:
            self.small_clsprob(batch, in_param)
        else:
            self.big_clsprob(batch, in_param)

    def handle_clsprob_3(self, batch, param):
        """
          compute cls pro of boxinfo2

          Parameters
          ----------
          batch: the photo number
          param: param is a dict, the keys as follow:
                 index_offset: index_offset of objects
                 count: the number of filtered boxes before doing Iou
                 index_ub: is a tensor,which used to store filted obj index
                 obj_gm_offset: a scalar,store obj_gm_offset

          Returns
          -------
          None
          """
        in_param = {}
        in_param["out_offset"] = self.boxes * (self.height1 * self.width1 + \
                                               self.height2 * self.width2)
        in_param["w"] = self.width3
        in_param["h"] = self.height3
        in_param["obj_data"] = self.obj_data3
        in_param["clz_data"] = self.classes_data3
        in_param["index_ub"] = param['index_ub']
        in_param["count"] = param['count']
        in_param["index_offset"] = param['index_offset']
        in_param["total_len"] = self.boxes * self.width3 * self.height3
        in_param["obj_gm_offset"] = param['obj_gm_offset']

        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ( \
                "Ascend310","Ascend910", \
                "Hi3796CV300ES"):
            in_param['index_offset'].set_as(
                self.boxes * self.width1 * self.height1 + \
                self.boxes * self.width2 * self.height2)
        if self.boxes * self.width3 * self.height3 * self.dsize < \
                self.one_max_size // 2:
            self.small_clsprob(batch, in_param)
        else:
            self.big_clsprob(batch, in_param)

    # pylint: disable=too-many-locals,too-many-statements
    def small_clsprob(self, batch, param):
        """
          compute small cls prob

          Parameters
          ----------
          param: param is a dict, the keys as follow:
                 ub_a: a tensor,store clsprob object data
                 ub_b: a tensor,store clsprob object threshold data
                 zero_tensor: a tensor, init with zero value
                 adj_len: the number of elements of each boxes with 32 alignment
                 burlen: data move nurst
                 num: process the number of elements with each repeat
                 repeat: vector repeat
          Returns
          -------
          None
          """
        self.init_small_clsprob_param(param)
        self.instance.vec_muls(self.mask, param['zero_tensor'], param['zero_tensor'],
                            VALUE_ZERO,
                            REPEAT_ONE, STRIDE_EIGHT,
                            STRIDE_EIGHT)

        self.instance.data_move(param['ub_a'], param['obj_data'][batch, 0], SID,
                                NBURST_ONE, param['burlen'], GAP_ZERO, GAP_ZERO)
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") not in ( \
                                            "Ascend310","Ascend910", \
                                            "Hi3796CV300ES"):
            self.instance.data_move(self.obj_data[param['obj_gm_offset']],
                                    param['ub_a'], SID,
                                    NBURST_ONE, param['burlen'], GAP_ZERO,
                                    GAP_ZERO)
            param['obj_gm_offset'].set_as(param['obj_gm_offset'] + \
                                          self.boxes * param['h'] * param['w'])
        # if obj_data < obj_threshold
        self.instance.vec_dup(self.mask, param['ub_b'], self.obj_threshold,
                                 param['repeat'], STRIDE_EIGHT)

        ones_ub = self.instance.Tensor(self.dtype, (128,), name="ones_ub",
                                       scope=tik.scope_ubuf)
        self.instance.vec_dup(self.mask, ones_ub[0], 1, 1, 8)
        zeros_ub = self.instance.Tensor(self.dtype, (128,), name="zeros_ub",
                                        scope=tik.scope_ubuf)
        self.instance.vec_dup(self.mask, zeros_ub[0], 0, 1, 8)
        reduce_mask_ub = self.instance.Tensor(self.dtype, (128,),
                                              name="reduce_mask_ub",
                                              scope=tik.scope_ubuf)
        index_len = self.instance.Scalar("int32")
        index_len.set_as(constant.VECTOR_BYTE_SIZE // self.dsize)
        last_index_len = self.instance.Scalar("int32")
        last_index_len.set_as(param['total_len'] % index_len)
        with self.instance.if_scope(last_index_len == 0):
            last_index_len.set_as(index_len)

        with self.instance.for_range(VALUE_ZERO, param['repeat']) as cycle:
            sel = self.instance.Tensor("uint16", (8, ),
                                       name="sel",
                                       scope=tik.scope_ubuf)
            self.instance.vec_dup(8, sel, 0, 1, 8)
            self.instance.vec_cmpv_gt(sel,
                                      param['ub_a'][
                                          param['num'] * cycle], 
                                      param['ub_b'][
                                          param['num'] * cycle],
                                      1, 8, 8)

            self.instance.vec_sel(self.mask, VALUE_ZERO,
                               param['ub_a'][param['num'] * cycle],
                               sel,
                               param['ub_a'][param['num'] * cycle],
                               param['zero_tensor'], REPEAT_ONE)
            self.instance.vec_sel(self.mask, 0, reduce_mask_ub[0], sel,
                               ones_ub[0], zeros_ub[0], REPEAT_ONE,
                               STRIDE_ONE)
            param['reduce_mask_ub'] = reduce_mask_ub
            with self.instance.if_scope(cycle == param['repeat'] - 1):
                index_len.set_as(last_index_len)
            self.set_index_ub(param, index_len)
        param['faces_in_loop'], param['last_loop'], param['loop'] = \
            self.get_faces_params(param['adj_len'], self.classes)
        with self.instance.for_range(VALUE_ZERO, param['loop']) as loop_idx:

            ub_c = self.instance.Tensor(self.dtype,
                                        (self.one_max_size // self.dsize,),
                                        scope=tik.scope_ubuf, name="ub_c")
            last_32b = self.instance.Tensor(self.dtype,
                                            (BLOCK_SIZE,),
                                            scope=tik.scope_ubuf,
                                            name="last_32b")
            faces = self.instance.Scalar("int32")
            with self.instance.if_scope(loop_idx != param['loop'] - VALUE_ONE):
                faces.set_as(param['faces_in_loop'])
            with self.instance.else_scope():
                faces.set_as(param['last_loop'])

            param['burlen'].set_as(
                (faces * param['adj_len'] * self.dsize) // BLOCK_SIZE)

            self.instance.data_move(ub_c,
                                    param['clz_data'][
                                        batch, param[
                                            'faces_in_loop'] * loop_idx, 0],
                                    SID,
                                    NBURST_ONE, param['burlen'], GAP_ZERO,
                                    GAP_ZERO)
            # burlen for mov out
            param['burlen'].set_as(
                self.get_burlen(param["h"] * param["w"] * self.boxes))

            # a face = h*w*box
            with self.instance.for_range(VALUE_ZERO, faces,
                                         thread_num=2) as loop:
                param['ub_d'] = self.instance.Tensor(self.dtype,
                                                     (
                                                         self.one_max_size // self.dsize,),
                                                     scope=tik.scope_ubuf,
                                                     name="ub_d")
                start_idx = self.instance.Scalar()
                start_idx.set_as(loop * param['adj_len'])
                co_id = self.instance.Scalar()
                co_id.set_as(param['faces_in_loop'] * loop_idx + loop)

                self.instance.vec_mul(self.mask, param['ub_d'], ub_c[start_idx],
                                   param['ub_a'], param['repeat'],
                                   STRIDE_EIGHT, STRIDE_EIGHT,
                                   STRIDE_EIGHT)

                if self.tail_len != VALUE_ZERO and param['h'] == self.height3 \
                        and param['w'] == self.width3:
                    param['burlen'].set_as(param['burlen'] - VALUE_ONE)
                    with self.instance.if_scope(param['burlen'] > VALUE_ZERO):
                        self.instance.data_move(
                            self.inter_classes[
                                batch, co_id, param['out_offset']],
                            param['ub_d'], SID, NBURST_ONE, param['burlen'],
                            GAP_ZERO, GAP_ZERO)
                    param['burlen'].set_as(param['burlen'] + 1)
                    tail_idx = self.instance.Scalar(name="tail_idx")
                    tail_idx.set_as(self.len3 - self.len_32b)
                    self.instance.data_move(last_32b, self.inter_classes[
                        batch, co_id, param['out_offset'] + tail_idx],
                                            SID, NBURST_ONE, VALUE_ONE,
                                            GAP_ZERO,
                                            GAP_ZERO)
                    with self.instance.for_range(VALUE_ZERO,
                                                 self.tail_len) as cycle:
                        tmp_scalar = self.instance.Scalar(self.dtype)
                        tmp_scalar.set_as(
                            param['ub_d'][self.len3 - self.tail_len + cycle])
                        last_32b[self.len_32b - self.tail_len + cycle].set_as(
                            tmp_scalar)

                    self.instance.data_move(
                        self.inter_classes[batch, co_id, param[
                            'out_offset'] + tail_idx],
                        last_32b, SID, NBURST_ONE, VALUE_ONE,
                        GAP_ZERO, GAP_ZERO)

                else:
                    self.instance.data_move(
                        self.inter_classes[batch, co_id, param['out_offset']],
                        param['ub_d'], SID, NBURST_ONE, param['burlen'],
                        GAP_ZERO, GAP_ZERO)

    def init_small_clsprob_param(self, param):
        """
          init small cls prob parameters

          Parameters
          ----------
          param: param is a dict, the keys as follow:
                 ub_a: a tensor,store clsprob object data
                 ub_b: a tensor,store clsprob object threshold data
                 zero_tensor: a tensor, init with zero value
                 adj_len: the number of elements of each boxes with 32 alignment
                 burlen: data move nurst
                 num: process the number of elements with each repeat
                 repeat: vector repeat

          Returns
          -------
          None
          """
        param['ub_a'] = self.instance.Tensor(self.dtype,
                                             (self.one_max_size // self.dsize,),
                                             scope=tik.scope_ubuf, name="ub_a")
        param['ub_b'] = self.instance.Tensor(self.dtype,
                                             (self.one_max_size // self.dsize,),
                                             scope=tik.scope_ubuf, name="ub_b")
        param['zero_tensor'] = self.instance.Tensor(self.dtype,
                                                    (VECTOR_BYTE_SIZE,),
                                                    scope=tik.scope_ubuf,
                                                    name="zero_tensor")
        param['adj_len'] = self.get_adj_hw(self.boxes * param['h'], param['w'])
        param['burlen'] = self.instance.Scalar()
        param['burlen'].set_as(self.get_burlen(
            self.boxes * param["h"] * param["w"]))
        param['num'] = VECTOR_BYTE_SIZE // self.dsize
        param['repeat'] = self.get_repeat(self.boxes * param["h"] * param["w"])

    def big_clsprob(self, batch, param):
        """
          compute big cls prob

          Parameters
          ----------
          param: param is a dict, the keys as follow:
                 mov_len: each loop process the number of elements
                 mov_loop: data move loop times
                 ub_a: a tensor, store object data
                 ub_b: a tensor, store object threshold data
                 repeat: data move nurst
                 obj_data: process the number of elements with each repeat
                 repeat: vector repeat times
          Returns
          -------
          None
          """
        param['mov_len'], param['mov_loop'], param[
            'last_len'] = self.get_tiling_param(self.boxes * param['h'],
                                                param['w'])
        each_len = self.instance.Scalar("int32")
        each_len.set_as(param['mov_len'])
        with self.instance.for_range(VALUE_ZERO, param['mov_loop']) as loop:
            self.init_bigcls_param(loop, param)
            self.instance.vec_muls(self.mask, param['zero_tensor'],
                                param['zero_tensor'],
                                VALUE_ZERO, REPEAT_ONE,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)
            # move obj data to ub a
            self.instance.data_move(param['ub_a'],
                                    param['obj_data'][
                                        batch, param['mov_len'] * loop],
                                    SID,
                                    NBURST_ONE, param['burlen'], GAP_ZERO,
                                    GAP_ZERO)

            # if obj_data < obj_threshold
            self.instance.vec_dup(self.mask, param['ub_b'], self.obj_threshold,
                                     param['repeat'], STRIDE_EIGHT)

            reduce_mask_ub = self.instance.Tensor(self.dtype, (128,),
                                                  name="reduce_mask_ub",
                                                  scope=tik.scope_ubuf)
            ones_ub = self.instance.Tensor(self.dtype, (128,), name="ones_ub",
                                           scope=tik.scope_ubuf)
            self.instance.vec_dup(self.mask, ones_ub[0], VALUE_ONE, REPEAT_ONE,
                                     STRIDE_EIGHT)
            zeros_ub = self.instance.Tensor(self.dtype, (128,), name="zeros_ub",
                                            scope=tik.scope_ubuf)
            self.instance.vec_dup(self.mask, zeros_ub[0], VALUE_ZERO,
                                     REPEAT_ONE,
                                     STRIDE_EIGHT)

            with self.instance.if_scope(loop == param['mov_loop'] - 1):
                each_len.set_as(param['last_len'])
            index_len = self.instance.Scalar("int32")
            index_len.set_as(param['num'])
            last_index_len = self.instance.Scalar("int32")
            last_index_len.set_as(each_len % index_len)
            with self.instance.if_scope(last_index_len == 0):
                last_index_len.set_as(index_len)

            if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") not in ( \
                "Ascend310","Ascend910", \
                "Hi3796CV300ES"):
                self.instance.data_move(self.obj_data[param['obj_gm_offset']],
                                        param['ub_a'], SID,
                                        NBURST_ONE, param['burlen'], GAP_ZERO,
                                        GAP_ZERO)
                with self.instance.if_scope(loop == param['mov_loop'] - 1):
                    param['obj_gm_offset'].set_as(param['obj_gm_offset'] + \
                                                  param['last_len'])
                with self.instance.else_scope():
                    param['obj_gm_offset'].set_as(param['obj_gm_offset'] + \
                                                  param['mov_len'])

            with self.instance.for_range(VALUE_ZERO, param['repeat']) as cycle:
                sel = self.instance.Tensor("uint16", (8, ),
                                           name="sel",
                                           scope=tik.scope_ubuf)
                self.instance.vec_dup(8, sel, 0, 1, 8)
                self.instance.vec_cmpv_gt(sel,
                                          param['ub_a'][
                                              param['num'] * cycle], 
                                          param['ub_b'][
                                              param['num'] * cycle],
                                          1, 8, 8)
                
                self.instance.vec_sel(self.mask, VALUE_ZERO, param['ub_a'][
                    param['num'] * cycle], sel,
                                   param['ub_a'][
                                       param['num'] * cycle],
                                   param['zero_tensor'], REPEAT_ONE)

                self.instance.vec_sel(self.mask, 0, reduce_mask_ub[0], sel,
                                   ones_ub[0], zeros_ub[0], 1, 1)
                param['reduce_mask_ub'] = reduce_mask_ub
                with self.instance.if_scope(cycle == param['repeat'] - 1):
                    index_len.set_as(last_index_len)
                self.set_index_ub(param, index_len)
            shape = self.one_max_size // self.dsize
            thread_num = 2
            if self.classes == 1:
                thread_num = 1
            with self.instance.for_range(VALUE_ZERO, self.classes,
                                         thread_num=thread_num) as co_id:

                param['ub_c'] = self.instance.Tensor(self.dtype,
                                                     (shape,),
                                                     scope=tik.scope_ubuf,
                                                     name="ub_c")

                # move classes data to ub c
                self.instance.data_move(param['ub_c'], param['clz_data'][
                    batch, co_id, param['mov_len'] * loop], SID, NBURST_ONE,
                                        param['burlen'], GAP_ZERO, GAP_ZERO)

                self.instance.vec_mul(self.mask, param['ub_c'], param['ub_a'],
                                   param['ub_c'], param['repeat'],
                                   STRIDE_EIGHT,
                                   STRIDE_EIGHT,
                                   STRIDE_EIGHT)

                if self.tail_len != VALUE_ZERO and param['h'] == self.height3 \
                        and param['w'] == self.width3:
                    with self.instance.if_scope(
                            loop == param['mov_loop'] - VALUE_ONE):
                        param['burlen'].set_as(param['burlen'] - VALUE_ONE)
                        with self.instance.if_scope(
                                param['burlen'] > VALUE_ZERO):
                            self.instance.data_move(
                                self.inter_classes[batch, co_id,
                                                   param['out_offset'] +
                                                   param['mov_len'] * loop],
                                param['ub_c'], SID, NBURST_ONE, param['burlen'],
                                GAP_ZERO, GAP_ZERO)
                        param['burlen'].set_as(param['burlen'] + VALUE_ONE)
                        tail_idx = self.instance.Scalar(name="tail_idx")
                        tail_idx.set_as(param['last_len'] - self.len_32b)
                        self.instance.data_move(
                            param['last_32b'],
                            self.inter_classes[batch, co_id,
                                               param['out_offset'] +
                                               param['mov_len'] * loop +
                                               tail_idx],
                            SID, NBURST_ONE, VALUE_ONE, GAP_ZERO, GAP_ZERO)
                        with self.instance.for_range(VALUE_ZERO,
                                                     self.tail_len) as cycle:
                            scalar = self.instance.Scalar(self.dtype)
                            scalar.set_as(param['ub_c'][param['last_len'] \
                                                        - self.tail_len + cycle])
                            param['last_32b'][self.len_32b - \
                                              self.tail_len + cycle].set_as( \
                                scalar)
                        offset = param['out_offset'] + param['mov_len'] * loop \
                                 + tail_idx
                        dest = self.inter_classes[batch, co_id, offset]
                        self.instance.data_move(dest, param['last_32b'], SID,
                                                NBURST_ONE,
                                                VALUE_ONE, GAP_ZERO, GAP_ZERO)
                    with self.instance.else_scope():
                        offset = param['out_offset'] + param['mov_len'] * loop
                        self.instance.data_move(
                            self.inter_classes[batch, co_id, offset],
                            param['ub_c'], SID, NBURST_ONE,
                            param['burlen'], GAP_ZERO, GAP_ZERO)
                else:
                    dest = self.inter_classes[
                        batch, co_id, param['out_offset'] + \
                        param['mov_len'] * loop]
                    self.instance.data_move(dest, param['ub_c'],
                                            SID, NBURST_ONE, param['burlen'],
                                            GAP_ZERO, GAP_ZERO)

    def init_bigcls_param(self, loop, param):
        """
          init big cls prob parameters

          Parameters
          ----------
          param: param is a dict, the keys as follow:
                 ub_a: a tensor, store object data
                 ub_b: a tensor, store object threshold data
                 zero_tensor: a tensor, init with zero value
                 last_32b: a tensor, store last32b data
                 burlen: data move nurst
                 repeat: vector repeat times
                 num: process the number of elements with each repeat
          Returns
          -------
          None
          """
        param['ub_a'] = self.instance.Tensor(self.dtype,
                                             (self.one_max_size // self.dsize,),
                                             scope=tik.scope_ubuf, name="ub_a")

        param['ub_b'] = self.instance.Tensor(self.dtype,
                                             (self.one_max_size // self.dsize,),
                                             scope=tik.scope_ubuf, name="ub_b")
        param['zero_tensor'] = self.instance.Tensor(self.dtype,
                                                    (VECTOR_BYTE_SIZE,),
                                                    scope=tik.scope_ubuf,
                                                    name="zero_tensor")
        param['last_32b'] = self.instance.Tensor(self.dtype,
                                                 (BLOCK_SIZE,),
                                                 scope=tik.scope_ubuf,
                                                 name="last_32b")

        param['burlen'] = self.instance.Scalar(name="burlen")
        param['repeat'] = self.instance.Scalar(name="repeat")
        param['num'] = VECTOR_BYTE_SIZE // self.dsize
        with self.instance.if_scope(loop == param['mov_loop'] - VALUE_ONE):
            param['burlen'].set_as(self.get_burlen(param['last_len']))
            param['repeat'].set_as(self.get_repeat(param['last_len']))
        with self.instance.else_scope():
            param['burlen'].set_as(self.get_burlen(param['mov_len']))
            param['repeat'].set_as(self.get_repeat(param['mov_len']))


def get_loop_param(length, max_ub_num):
    """
    get loop parameters

    Parameters
    ----------
    length: total number
    max_ub_num: max of ub num

    Returns
    -------
    loop_cycle: loop cycle
    last_ub_num: the last data needs ub num
    """
    loop_cycle = length // max_ub_num
    last_ub_num = length % max_ub_num
    ub_num = max_ub_num
    if loop_cycle == 0:
        ub_num = length
    if last_ub_num != 0:
        loop_cycle = loop_cycle + 1
    else:
        last_ub_num = max_ub_num

    return loop_cycle, ub_num, last_ub_num


def check_param(input_dict):
    """
      check parameters

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

    pre_nms_topn = input_dict.get("pre_nms_topn")
    if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ( \
            "Ascend310","Ascend910", \
            "Hi3796CV300ES"):
        util.check_dtype_rule(input_dict.get("box1_info").get("dtype"),
                              (constant.DATA_TYPE_FP16))
        util.check_dtype_rule(input_dict.get("box2_info").get("dtype"),
                              (constant.DATA_TYPE_FP16))
        util.check_dtype_rule(input_dict.get("box3_info").get("dtype"),
                              (constant.DATA_TYPE_FP16))
    else:
        util.check_dtype_rule(input_dict.get("box1_info").get("dtype"), (
            constant.DATA_TYPE_FP16, constant.DATA_TYPE_FP32))
        util.check_dtype_rule(input_dict.get("box2_info").get("dtype"), (
            constant.DATA_TYPE_FP16, constant.DATA_TYPE_FP32))
        util.check_dtype_rule(input_dict.get("box3_info").get("dtype"), (
            constant.DATA_TYPE_FP16, constant.DATA_TYPE_FP32))
    util.check_kernel_name(input_dict.get("kernel_name"))
    coords = input_dict.get("coords")
    post_top_k = input_dict.get("post_top_k")
    if coords != 4:
        raise RuntimeError("coords[%d] only support 4" % (coords))
    max_box_number_per_batch = input_dict.get("max_box_number_per_batch")
    dtype = input_dict.get("box1_info").get("dtype")
    if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ( \
            "Hi3796CV300ES") \
            or dtype == constant.DATA_TYPE_FP32:
        if pre_nms_topn > PRE_NMS_TOPN // 2:
            raise RuntimeError(
                "pre_nms_topn[%d] must less than 512" % (pre_nms_topn))
    else:
        if pre_nms_topn > PRE_NMS_TOPN:
            raise RuntimeError(
                "pre_nms_topn[%d] must less than 1024" % (pre_nms_topn))
    if max_box_number_per_batch > PRE_NMS_TOPN or max_box_number_per_batch <= 0:
        raise RuntimeError(
            "max_box_number_per_batch[%d] must less than 1024 and bigger than 0" % (
                max_box_number_per_batch))
    if max_box_number_per_batch % 16 != 0:
        raise RuntimeError(
            "max_box_number_per_batch[%d] must be a multiple of 16" % (
                max_box_number_per_batch))

    if max_box_number_per_batch < pre_nms_topn or pre_nms_topn <= 0:
        raise RuntimeError(
            "pre_nms_topn[%d] must less than max_box_number_per_batch[%d] and bigger than 0" % (
                pre_nms_topn, max_box_number_per_batch))
    if max_box_number_per_batch < post_top_k or post_top_k <= 0:
        raise RuntimeError(
            "post_top_k[%d] must less than max_box_number_per_batch[%d] and bigger than 0" % (
                post_top_k, max_box_number_per_batch))

    dsize = common.get_data_size(input_dict.get("box1_info").get("dtype"))
    height = input_dict.get("box1_info").get("shape")[2]
    width = input_dict.get("box1_info").get("shape")[3]
    if height * width * dsize < constant.BLOCK_SIZE:
        raise RuntimeError(
            "box1_info's height[%d] multi with width[%d]'s size \
            must bigger than 32b" % (height, width))

    height = input_dict.get("box2_info").get("shape")[2]
    width = input_dict.get("box2_info").get("shape")[3]
    if height * width * dsize < constant.BLOCK_SIZE:
        raise RuntimeError(
            "box2_info's height[%d] multi with width[%d]'s size \
            must bigger than 32b" % (height, width))
    height = input_dict.get("box3_info").get("shape")[2]
    width = input_dict.get("box3_info").get("shape")[3]
    if height * width * dsize < constant.BLOCK_SIZE:
        raise RuntimeError(
            "box3_info's height[%d] multi with width[%d]'s size\
             must bigger than 32b" % (height, width))
