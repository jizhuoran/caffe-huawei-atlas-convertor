#!/usr/bin/env python
# -*- coding:utf-8 -*-
# pylint: disable=too-many-lines,import-error,no-self-use
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

yolo_v3_correct_region_box
"""

import math
from te import tik
from te import platform as tbe_platform
from impl.constant_util import BLOCK_SIZE, DATA_SIZE_TWO, DATA_SIZE_FOUR, \
    VECTOR_BYTE_SIZE, STRIDE_ONE, DATA_TYPE_FP16, AIC, CLOUD

# reserve size for ub
RESERVE_SIZE = 16 * 1024

# repeat one
REPEAT_ONE = 1

# one nburst
NBURST_ONE = 1

# value one
VALUE_ONE = 1

# stride eight for dma
STRIDE_EIGHT = 8

# stride zero for dma
GAP_ZERO = 0

# sid for dma
SID = 0

# value zero
VALUE_ZERO = 0

# value two
VALUE_TWO = 2

# value three
VALUE_THREE = 3

# neg two
NEG_TWO = -2

# neg one
NEG_ONE = -1

# value half
VALUE_HALF = 0.5


# pylint: disable=too-many-instance-attributes
class GetCorrectBoxBase():
    """
    Function: store yolov3 ClsProb parameters
    Modify : 2019-11-06
    """

    def __init__(self, input_dict):
        """
      init the CorrectBox parameters

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
        self.instance = tik.Tik(tik.Dprofile())
        self.one_max_size = (tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) -
                             RESERVE_SIZE) // 8
        self.batch = input_dict.get("box1_info").get("shape")[0]
        self.dtype = input_dict.get("box1_info").get("dtype")
        self.classes = input_dict.get("classes")
        self.boxes = input_dict.get("boxes")
        self.relative = input_dict.get("relative")
        self.dsize = DATA_SIZE_FOUR
        if self.dtype == DATA_TYPE_FP16:
            self.dsize = DATA_SIZE_TWO
        self.mask = VECTOR_BYTE_SIZE // self.dsize

    def set_dsize(self, dsize):
        """
          set dsize

          Parameters
          ----------
           dsize: dsize

          Returns
          -------
          None
          """
        self.dsize = dsize

    def get_dtype(self):
        """
          get dtype

          Parameters
          ----------
           None

          Returns
          -------
          dtype:data type
          """
        return self.dtype


class GetCorrectBoxParam(GetCorrectBoxBase):
    """
    Function: store GetCorrectBoxParam parameters
    Modify : 2019-11-06
    """

    def __init__(self, input_dict):
        """
      init the CorrectBox parameters

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
        super(GetCorrectBoxParam, self).__init__(input_dict)

        self.obj_threshold = input_dict.get("obj_threshold")
        self.classes_threshold = input_dict.get("classes_threshold")
        self.post_top_k = input_dict.get("post_top_k")
        self.nms_threshold = input_dict.get("nms_threshold")
        self.pre_nms_topn = input_dict.get("pre_nms_topn")
        self.max_box_number_per_batch = input_dict.get(
            "max_box_number_per_batch")
        self.kernel_name = input_dict.get("kernel_name")

    def set_pre_nms_topn(self, pre_nms_topn):
        """
          set pre_nms_topn

          Parameters
          ----------
           pre_nms_topn: for each category,take the number of pre nms topn
                    before processing, and the maximum is 1024

          Returns
          -------
          None
          """
        self.pre_nms_topn = pre_nms_topn

    def get_shape(self, old_shape, need_low_dim=False):
        """
          compute shape

          Parameters
          ----------
           old_shape: shape before compute
           need_low_dim: whether need low dim,true or false

          Returns
          -------
          None
          """
        old_shape = list(old_shape)
        length = len(old_shape)

        if length == 1:
            old_shape[0] += BLOCK_SIZE
            return tuple(old_shape)

        if not need_low_dim:
            size = self.dsize
            for i in range(0, length):
                size *= old_shape[i]
            unit_rev = self.dsize
            for i in range(1, length):
                unit_rev *= old_shape[i]
        else:
            size = self.dsize * old_shape[length - 1]
            unit_rev = size

        if size % BLOCK_SIZE == 0:
            rev = 0
        else:
            rev = BLOCK_SIZE // unit_rev + 1
        old_shape[0] += rev

        return tuple(old_shape)


class GetCorrectBoxParam2(GetCorrectBoxParam):
    """
    Function: store GetCorrectBoxParam2 parameters
    Modify : 2019-11-06
    """

    def __init__(self, input_dict):
        """
      init the GetCorrectBox2 parameters

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
        super(GetCorrectBoxParam2, self).__init__(input_dict)
        self.biases = input_dict.get("biases")
        self.height1 = input_dict.get("box1_info").get("shape")[2]
        self.width1 = input_dict.get("box1_info").get("shape")[3]
        self.height2 = input_dict.get("box2_info").get("shape")[2]
        self.width2 = input_dict.get("box2_info").get("shape")[3]
        self.height3 = input_dict.get("box3_info").get("shape")[2]
        self.width3 = input_dict.get("box3_info").get("shape")[3]

    def get_adj_hw(self, height, width):
        """
          compute height and weight with 32 alignment

          Parameters
          ----------
           height: box height
           width: box width

          Returns
          -------
          None
          """
        return math.ceil((height * width + 16) / 16) * 16


class GetCorrectBoxTensor(GetCorrectBoxParam2):
    """
    Function: store GetCorrectBoxTensor parameters
    Modify : 2019-11-06
    """

    def __init__(self, input_dict):
        """
      init the GetCorrectBox2 parameters

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
        super(GetCorrectBoxTensor, self).__init__(input_dict)

        adj_hw = self.get_adj_hw(self.height1, self.width1)
        self.coord_data1 = self.instance.Tensor(self.dtype,
                                                (self.batch, self.boxes * 4,
                                                 adj_hw),
                                                scope=tik.scope_gm,
                                                name="coord_data1")
        self.windex1 = self.instance.Tensor(self.dtype, (adj_hw,),
                                            scope=tik.scope_gm,
                                            name="windex1")
        self.hindex1 = self.instance.Tensor(self.dtype, (adj_hw,),
                                            scope=tik.scope_gm,
                                            name="hindex1")
        adj_hw = self.get_adj_hw(self.boxes * self.height1, self.width1)
        self.obj_data1 = self.instance.Tensor(self.dtype, (self.batch, adj_hw),
                                              scope=tik.scope_gm,
                                              name="obj_data1")
        self.classes_data1 = self.instance.Tensor(self.dtype,
                                                  (self.batch, self.classes,
                                                   adj_hw),
                                                  scope=tik.scope_gm,
                                                  name="classes_data1")
        adj_hw = self.get_adj_hw(self.height2, self.width2)
        self.coord_data2 = self.instance.Tensor(self.dtype,
                                                (self.batch, self.boxes * 4,
                                                 adj_hw),
                                                scope=tik.scope_gm,
                                                name="coord_data2")

        self.windex2 = self.instance.Tensor(self.dtype, (adj_hw,),
                                            scope=tik.scope_gm,
                                            name="windex2")


class GetCorrectBoxTensor2(GetCorrectBoxTensor):
    """
    Function: store GetCorrectBoxTensor2 parameters
    Modify : 2019-11-06
    """

    def __init__(self, input_dict):
        """
      init the GetCorrectBox2 parameters

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
        super(GetCorrectBoxTensor2, self).__init__(input_dict)

        adj_hw = self.get_adj_hw(self.height2, self.width2)
        self.hindex2 = self.instance.Tensor(self.dtype, (adj_hw,),
                                            scope=tik.scope_gm,
                                            name="hindex2")
        adj_hw = self.get_adj_hw(self.boxes * self.height2, self.width2)
        self.classes_data2 = self.instance.Tensor(self.dtype,
                                                  (self.batch, self.classes,
                                                   adj_hw),
                                                  scope=tik.scope_gm,
                                                  name="classes_data2")
        self.obj_data2 = self.instance.Tensor(self.dtype, (self.batch, adj_hw),
                                              scope=tik.scope_gm,
                                              name="obj_data2")

        adj_hw = self.get_adj_hw(self.height3, self.width3)
        self.coord_data3 = self.instance.Tensor(self.dtype,
                                                (self.batch, self.boxes * 4,
                                                 adj_hw),
                                                scope=tik.scope_gm,
                                                name="coord_data3")
        self.windex3 = self.instance.Tensor(self.dtype, (adj_hw,),
                                            scope=tik.scope_gm,
                                            name="windex3")
        adj_hw = self.get_adj_hw(self.boxes * self.height3, self.width3)
        self.obj_data3 = self.instance.Tensor(self.dtype, (self.batch, adj_hw),
                                              scope=tik.scope_gm,
                                              name="obj_data3")
        self.classes_data3 = self.instance.Tensor(self.dtype,
                                                  (self.batch, self.classes,
                                                   adj_hw),
                                                  scope=tik.scope_gm,
                                                  name="classes_data3")

    def get_block_param(self):
        """
          compute block parameters

          Parameters
          ----------
           None

          Returns
          -------
          None
          """
        #block_num = tik.Dprofile().get_aicore_num()
        block_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        if block_num > self.batch:
            outer_loop = 1
            block_num = self.batch
            outer_tail = 0
        else:
            outer_loop = self.batch // block_num
            outer_tail = self.batch - block_num * outer_loop

        return block_num, outer_loop, outer_tail


class GetCorrectBoxTensor3(GetCorrectBoxTensor2):
    """
    Function: store GetCorrectBoxTensor3 parameters
    Modify : 2019-11-06
    """

    def __init__(self, input_dict):
        """
      init the GetCorrectBox2 parameters

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
        super(GetCorrectBoxTensor3, self).__init__(input_dict)

        adj_hw = self.get_adj_hw(self.height3, self.width3)
        self.hindex3 = self.instance.Tensor(self.dtype, (adj_hw,),
                                            scope=tik.scope_gm,
                                            name="hindex3")
        self.img_info = self.instance.Tensor(self.dtype,
                                             (self.batch * 4 + BLOCK_SIZE,),
                                             scope=tik.scope_gm,
                                             name="img_info")

        self.totalwh = self.height1 * self.width1 + \
                       self.height2 * self.width2 + self.height3 * self.width3

        # Intermediate Output
        self.inter_coords = self.instance.Tensor(self.dtype, self.get_shape(
            (self.batch, 4, self.boxes * self.totalwh), True),
                                                 scope=tik.scope_gm,
                                                 name="inter_coords",
                                                 is_workspace=True)
        self.inter_classes = self.instance.Tensor(self.dtype, self.get_shape(
            (self.batch, self.classes, self.boxes * self.totalwh), True),
                                                  scope=tik.scope_gm,
                                                  name="inter_classes",
                                                  is_workspace=True)
        self.len_32b = BLOCK_SIZE // self.dsize
        self.hwtail_len = self.height3 * self.width3 % (self.len_32b)

    def t_small_mov_to_gm(self, batch, param):
        """
          data move of small shape

          Parameters
          ----------
           batch: the number of the picture
           param: param is a dict, the keys as follow:
                 tail_idx: a scalar,store tail_idx
                 box_id:  a scalar,store box_id
                 burlen: a scalar,store burlen
                 last_32b: a tensor,store last_32b data
                 co_id: a scalar,store co_id
                 out_offset: a scalar,store out_offset
          Returns
          -------
          None
          """
        tail_idx = self.instance.Scalar(name="tail_idx", init_value=0)
        if (param['h'] * param['w'] * self.dsize) % BLOCK_SIZE != 0:
            with self.instance.if_scope(param['box_id'] == self.boxes - 1):
                tail_idx.set_as(
                    (param['burlen'] - 2) * (BLOCK_SIZE // self.dsize) \
                    + (param['h'] * param['w']) % (BLOCK_SIZE // self.dsize))
                param['burlen'].set_as(param['burlen'] - 1)
                with self.instance.for_range(0,
                                             BLOCK_SIZE // self.dsize) as loop:
                    tmp_scalar = self.instance.Scalar(self.dtype)
                    tmp_scalar.set_as(param['ub_b'][tail_idx + loop])
                    param['last_32b'][loop].set_as(tmp_scalar)
        # move y to gm
        self.instance.data_move(
            self.inter_coords[
                batch, param['co_id'], param['out_offset'] + param['w'] * param[
                    'h'] * param['box_id']],
            param['ub_b'], SID, NBURST_ONE, param['burlen'], GAP_ZERO, GAP_ZERO)
        if (param['h'] * param['w'] * self.dsize) % BLOCK_SIZE != 0:
            with self.instance.if_scope(param['box_id'] == self.boxes - 1):
                self.instance.data_move(self.inter_coords[batch, param['co_id'], \
                                                          param['out_offset'] \
                                                          + param['w'] * param[
                                                              'h'] \
                                                          * param['box_id'] \
                                                          + tail_idx],
                                        param['last_32b'], SID, NBURST_ONE,
                                        VALUE_ONE,
                                        GAP_ZERO, GAP_ZERO)

    def get_faces_params(self, adj_hw, c_num):
        """
          data move of small shape

          Parameters
          ----------
           adj_hw: the lenth of boxes
           c_num: the length of boxes's c dim
          Returns
          -------
          None
          """
        if self.one_max_size // (adj_hw * self.dsize) > c_num:
            loop = 1
            last_loop = c_num
            faces_in_one_loop = 0
        else:
            faces_in_one_loop = self.one_max_size // (adj_hw * self.dsize)
            loop = c_num // faces_in_one_loop
            faces_tail = c_num - faces_in_one_loop * loop
            loop = loop if faces_tail == 0 else loop + 1
            last_loop = faces_in_one_loop if faces_tail == 0 else faces_tail

        return faces_in_one_loop, last_loop, loop

    def newton_div(self, dst, divisor, dividend, repeat):
        """
          use newton_div to improve performance

          Parameters
          ----------
           dst: vdiv's dest tensor
           divisor: vdiv's src0 tensor
           dividend: vdiv's src1 tensor
           repeat: vdiv's needs repeat times
          Returns
          -------
          None
          """
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ( \
                "Ascend610","Ascend620","Ascend910") :
            self.instance.vdiv(self.mask, dst, divisor, dividend, repeat,
                               STRIDE_ONE,
                               STRIDE_ONE, STRIDE_ONE, STRIDE_EIGHT,
                               STRIDE_EIGHT,
                               STRIDE_EIGHT)
        else:
            with self.instance.new_stmt_scope():
                tmp_tensor = self.instance.Tensor(self.dtype, dividend.shape,
                                                  scope=tik.scope_ubuf,
                                                  name="tmp_tensor")
                # 1/dividend
                self.instance.vec_rec(self.mask, tmp_tensor, dividend, repeat,
                                   STRIDE_EIGHT, STRIDE_EIGHT)
                self.instance.vec_mul(self.mask, dividend, dividend, tmp_tensor,
                                   repeat,
                                   STRIDE_EIGHT,
                                   STRIDE_EIGHT, STRIDE_EIGHT)
                self.instance.vec_adds(self.mask, dividend, dividend, NEG_TWO,
                                    repeat,
                                    STRIDE_EIGHT,
                                    STRIDE_EIGHT)
                self.instance.vec_mul(self.mask, dividend, dividend, tmp_tensor,
                                   repeat,
                                   STRIDE_EIGHT,
                                   STRIDE_EIGHT, STRIDE_EIGHT)
                self.instance.vec_muls(self.mask, dividend, dividend, NEG_ONE,
                                    repeat,
                                    STRIDE_EIGHT,
                                    STRIDE_EIGHT)

                # divisor * (1/dividend)
                self.instance.vec_mul(self.mask, dst, divisor, dividend, repeat,
                                   STRIDE_EIGHT,
                                   STRIDE_EIGHT, STRIDE_EIGHT)

    def get_burlen(self, length):
        """
          compute data move nburst

          Parameters
          ----------
           length: the number of elements need data move

          Returns
          -------
          burlen: data move nburst
          """
        if (length * self.dsize) % BLOCK_SIZE == 0:
            return (length * self.dsize) // BLOCK_SIZE

        return (length * self.dsize) // BLOCK_SIZE + 1

    def get_repeat(self, length):
        """
          compute vector instructs repeat times

          Parameters
          ----------
           length: the number of elements need data move
          Returns
          -------
          repeats: vector instructs repeat times
          """
        if (length * self.dsize) % VECTOR_BYTE_SIZE == 0:
            return (length * self.dsize) // VECTOR_BYTE_SIZE

        return (length * self.dsize) // VECTOR_BYTE_SIZE + 1

    def get_x_y_params(self, img_info):
        """
          compute x,y parameters

          Parameters
          ----------
           img_info: a tensor,store image's width and height

          Returns
          -------
          x_vmuls_val: a scalar,store x_vmuls_val
          y_vmuls_val: a scalar,store y_vmuls_val
          x_vadds_val: a scalar,store x_vadds_val
          y_vadds_val: a scalar,store y_vadds_val
          """
        x_vmuls_val = self.instance.Scalar(self.dtype)
        y_vmuls_val = self.instance.Scalar(self.dtype)
        x_vadds_val = self.instance.Scalar(self.dtype)
        y_vadds_val = self.instance.Scalar(self.dtype)

        param = {}

        with self.instance.new_stmt_scope():
            param['ub_d'] = self.instance.Tensor(self.dtype,
                                                 (VECTOR_BYTE_SIZE,),
                                                 scope=tik.scope_ubuf,
                                                 name="ub_d")
            param['ub_e'] = self.instance.Tensor(self.dtype,
                                                 (VECTOR_BYTE_SIZE,),
                                                 scope=tik.scope_ubuf,
                                                 name="ub_e")
            param['ub_f'] = self.instance.Tensor(self.dtype,
                                                 (VECTOR_BYTE_SIZE,),
                                                 scope=tik.scope_ubuf,
                                                 name="ub_f")
            param['ub_g'] = self.instance.Tensor(self.dtype,
                                                 (VECTOR_BYTE_SIZE,),
                                                 scope=tik.scope_ubuf,
                                                 name="ub_g")
            param['lgt_tensor'] = self.instance.Tensor(self.dtype,
                                                       (VECTOR_BYTE_SIZE,),
                                                       scope=tik.scope_ubuf,
                                                       name="lgt_tensor")
            param['ret_tensor'] = self.instance.Tensor(self.dtype,
                                                       (VECTOR_BYTE_SIZE,),
                                                       scope=tik.scope_ubuf,
                                                       name="ret_tensor")

            new_h, new_w = self.get_new_h_w(img_info, param)

            tmp_scalar = self.instance.Scalar(self.dtype)

            # x vmuls param --> netw / new_w
            tmp_scalar.set_as(img_info[1])
            # ub_d netw
            self.instance.vec_dup(self.mask, param['ub_d'], tmp_scalar,
                                     REPEAT_ONE,
                                     STRIDE_EIGHT)
            # ub_e new_w
            self.instance.vec_dup(self.mask, param['ub_e'], new_w, REPEAT_ONE,
                                     STRIDE_EIGHT)
            # netw / new_w

            self.newton_div(param['ub_d'], param['ub_d'], param['ub_e'],
                            REPEAT_ONE)

            x_vmuls_val.set_as(param['ub_d'][0])

            # x vadds param --> ((new_w - netw)/2.0/netw) * (netw / new_w)
            # --> ((-1)*(netw / new_w) + 1)* 0.5
            self.instance.vec_muls(self.mask, param['ub_d'], param['ub_d'], NEG_ONE,
                                REPEAT_ONE,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)
            self.instance.vec_adds(self.mask, param['ub_d'], param['ub_d'],
                                VALUE_ONE,
                                REPEAT_ONE,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)
            self.instance.vec_muls(self.mask, param['ub_d'], param['ub_d'],
                                VALUE_HALF,
                                REPEAT_ONE,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)
            x_vadds_val.set_as(param['ub_d'][0])

            # y vmuls param --> neth / new_h
            tmp_scalar.set_as(img_info[0])
            # ub_d neth
            self.instance.vec_dup(self.mask, param['ub_d'], tmp_scalar,
                                     REPEAT_ONE,
                                     STRIDE_EIGHT)
            # ub_e new_h
            self.instance.vec_dup(self.mask, param['ub_e'], new_h, REPEAT_ONE,
                                     STRIDE_EIGHT)
            # neth / new_h

            self.newton_div(param['ub_d'], param['ub_d'], param['ub_e'],
                            REPEAT_ONE)

            y_vmuls_val.set_as(param['ub_d'][0])

            # y vadds param --> ((-1)*(neth / new_h) + 1)* 0.5
            self.instance.vec_muls(self.mask, param['ub_d'], param['ub_d'], NEG_ONE,
                                REPEAT_ONE,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)
            self.instance.vec_adds(self.mask, param['ub_d'], param['ub_d'],
                                VALUE_ONE,
                                REPEAT_ONE,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)
            self.instance.vec_muls(self.mask, param['ub_d'], param['ub_d'],
                                VALUE_HALF,
                                REPEAT_ONE,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)
            y_vadds_val.set_as(param['ub_d'][0])

        return x_vmuls_val, x_vadds_val, y_vmuls_val, y_vadds_val

    def get_new_h_w(self, img_info, param):
        """
          compute boxes's height and width

          Parameters
          ----------
           img_info: a tensor,store image's width and height
           param: a dict,the keys as follow:
                  ub_d: a middle tensor,used to compute boxes's height and width
                  ub_e:a middle tensor,used to compute boxes's height and width
                  ub_f:a middle tensor,used to compute boxes's height and width
                  ub_g:a middle tensor,used to compute boxes's height and width
                  ret_tensor:a middle tensor,used to compute boxes's
                             height and width
                  lgt_tensor:a middle tensor,used to compute boxes's
                             height and width

          Returns
          -------
          new_h: a scalar,store new_h
          new_w: a scalar,store new_w
          """
        tmp_scalar = self.instance.Scalar(self.dtype)
        new_h = self.instance.Scalar(self.dtype)
        new_w = self.instance.Scalar(self.dtype)
        # if netw/w < neth/h
        # vdup neth/h
        tmp_scalar.set_as(img_info[0])
        self.instance.vec_dup(self.mask, param['ub_d'], tmp_scalar, REPEAT_ONE,
                                 STRIDE_EIGHT)
        tmp_scalar.set_as(img_info[2])
        self.instance.vec_dup(self.mask, param['ub_g'], tmp_scalar, REPEAT_ONE,
                                 STRIDE_EIGHT)

        self.newton_div(param['ub_d'], param['ub_d'], param['ub_g'], REPEAT_ONE)

        # vdup netw/w
        tmp_scalar.set_as(img_info[1])
        self.instance.vec_dup(self.mask, param['ub_e'], tmp_scalar, REPEAT_ONE,
                                 STRIDE_EIGHT)
        tmp_scalar.set_as(img_info[3])
        self.instance.vec_dup(self.mask, param['ub_g'], tmp_scalar, REPEAT_ONE,
                                 STRIDE_EIGHT)
        self.newton_div(param['ub_e'], param['ub_e'], param['ub_g'], REPEAT_ONE)

        sel = self.instance.Tensor("uint16", (8, ),
                                   name="sel",
                                   scope=tik.scope_ubuf)
        self.instance.vec_dup(8, sel, 0, 1, 8)
        self.instance.vec_cmpv_lt(sel, param['ub_e'], param['ub_d'], 1, 8, 8)

        # get new w
        tmp_scalar.set_as(img_info[1])
        param['lgt_tensor'][0].set_as(tmp_scalar)
        tmp_scalar.set_as(img_info[3])
        self.instance.vec_muls(self.mask, param['ub_d'], param['ub_d'], tmp_scalar,
                            REPEAT_ONE, STRIDE_EIGHT,
                            STRIDE_EIGHT)
        self.instance.vec_sel(self.mask, VALUE_ZERO, param['ret_tensor'], sel,
                           param['lgt_tensor'],
                           param['ub_d'], STRIDE_ONE)
        new_w.set_as(param['ret_tensor'][0])
        # get new h
        tmp_scalar.set_as(img_info[2])
        self.instance.vec_muls(self.mask, param['ub_e'], param['ub_e'], tmp_scalar,
                            REPEAT_ONE, STRIDE_EIGHT,
                            STRIDE_EIGHT)
        tmp_scalar.set_as(img_info[0])
        param['lgt_tensor'][0].set_as(tmp_scalar)
        self.instance.vec_sel(self.mask, VALUE_ZERO, param['ret_tensor'], sel,
                           param['ub_e'],
                           param['lgt_tensor'], STRIDE_ONE)
        new_h.set_as(param['ret_tensor'][0])
        return new_h, new_w


class GetCorrectBoxComputer(GetCorrectBoxTensor3):
    """
    Function: store GetCorrectBoxComputer parameters
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
        super(GetCorrectBoxComputer, self).__init__(input_dict)

        self.block_num, self.outer_loop, self.outer_tail = self.get_block_param()
        self.coords = input_dict.get('coords')
        self.input_dict = input_dict

    def convert_biases_data(self, param):
        """
           compute biases data

           Parameters
           ----------
            param: a dict,the keys as follow:
                   box_id: batch
                   biases1:box1's biases
                   biases2:box2's biases
                   biases3:box3's biases
           Returns
           -------
           ub_bias: a tensor,store bias
           """
        ub_bias = self.instance.Tensor(self.dtype, (VECTOR_BYTE_SIZE,),
                                       scope=tik.scope_ubuf,
                                       name="ub_bias")
        t_scalar = self.instance.Scalar(self.dtype)
        if param['box_id'] == 1:
            biases = self.input_dict['biases1']
        elif param['box_id'] == 2:
            biases = self.input_dict['biases2']
        else:
            biases = self.input_dict['biases3']

            # set bias to ub
        for i in range(0, self.boxes):
            t_scalar.set_as(biases[2 * i])
            ub_bias[2 * i].set_as(t_scalar)
            t_scalar.set_as(biases[2 * i + 1])
            ub_bias[2 * i + 1].set_as(t_scalar)
        return ub_bias

    def get_tiling_param(self, height, weight):
        """
           compute tilling param

           Parameters
           ----------
            height: box's height
            weight: box's width
           Returns
           -------
           mov_len: the number of elements of each loop
           mov_loop: loop times
           last_len: the number of elements of last loop
           """
        max_size = self.one_max_size

        mov_len = max_size // self.dsize
        mov_loop = (height * weight) // (max_size // self.dsize)
        mov_tail = height * weight - mov_len * mov_loop

        mov_loop = mov_loop if mov_tail == 0 else mov_loop + 1
        last_len = mov_len if mov_tail == 0 else mov_tail

        return mov_len, mov_loop, last_len

    def correct_box(self, batch, img_ub):
        """
           compute correct_box

           Parameters
           ----------
            batch: the number of picture
            img_ub: a tensor,store image info
           Returns
           -------
           None
           """
        with self.instance.new_stmt_scope():
            self.handle_coords_1(batch, img_ub)
        with self.instance.new_stmt_scope():
            self.handle_coords_2(batch, img_ub)
        with self.instance.new_stmt_scope():
            self.handle_coords_3(batch, img_ub)

    def handle_coords_3(self, batch, img_ub):
        """
           compute box3 coords

           Parameters
           ----------
            batch: the number of picture
            img_ub: a tensor,store image info
           Returns
           -------
           None
           """
        width = self.width3
        height = self.height3
        param = {}
        param['out_offset'] = self.height1 * self.width1 * self.boxes + \
                              self.height2 * self.width2 * self.boxes
        param['w'] = self.width3
        param['h'] = self.height3
        param['in_data'] = self.coord_data3
        param['windex'] = self.windex3
        param['hindex'] = self.hindex3
        param['box_id'] = 3
        param['img_ub'] = img_ub

        if width * height * self.dsize < self.one_max_size // 2:
            self.small_surface_template(batch, param)
        else:
            self.big_surface_template(batch, param)

    def handle_coords_2(self, batch, img_ub):
        """
           compute box2 coords

           Parameters
           ----------
            batch: the number of picture
            img_ub: a tensor,store image info
           Returns
           -------
           None
           """
        width = self.width2
        height = self.height2
        param = {}
        param['out_offset'] = self.height1 * self.width1 * self.boxes
        param['w'] = self.width2
        param['h'] = self.height2
        param['in_data'] = self.coord_data2
        param['windex'] = self.windex2
        param['hindex'] = self.hindex2
        param['box_id'] = 2
        param['img_ub'] = img_ub

        if width * height * self.dsize < self.one_max_size // 2:
            self.small_surface_template(batch, param)
        else:
            self.big_surface_template(batch, param)

    def handle_coords_1(self, batch, img_ub):
        """
           compute box1 coords

           Parameters
           ----------
            batch: the number of picture
            img_ub: a tensor,store image info
           Returns
           -------
           None
           """
        width = self.width1
        height = self.height1
        param = {}
        param['out_offset'] = 0
        param['w'] = self.width1
        param['h'] = self.height1
        param['in_data'] = self.coord_data1
        param['windex'] = self.windex1
        param['hindex'] = self.hindex1
        param['box_id'] = 1
        param['img_ub'] = img_ub

        if width * height * self.dsize < self.one_max_size // 2:
            self.small_surface_template(batch, param)
        else:
            self.big_surface_template(batch, param)

    def big_surface_template(self, batch, param):
        """
           compute big shape

           Parameters
           ----------
            batch: the number of picture
            param: a dict,the keys as fllow:
                   mov_len: the number of elements of each data move
                   mov_loop: data move loop times
                   last_len: the number of elements of last_len data move
                   ub_bias: a tensor,store bias
                   x_vmuls_val: a scalar
                   x_vadds_val: a scalar
                   y_vmuls_val: a scalar
                   ub_a: a tensor,store middle compute data
                   ub_b: a tensor,store middle compute data
                   ub_c: a tensor,store middle compute data
                   last_32b: a tensor,store last_32b data
                   co_id: a scalar,store co_id
                   box_id:  a scalar,store box_id
                   in_data: a tensor
           Returns
           -------
           None
           """
        param['mov_len'], param['mov_loop'], param[
            'last_len'] = self.get_tiling_param(param['h'], param['w'])
        param['ub_bias'] = self.convert_biases_data(param)

        param['x_vmuls_val'], param['x_vadds_val'], param['y_vmuls_val'], param[
            'y_vadds_val'] = self.get_x_y_params(param['img_ub'])
        shape = self.one_max_size // self.dsize
        with self.instance.for_range(0, self.boxes * self.coords) as cycle:
            param['ub_a'] = self.instance.Tensor(self.dtype, (shape,),
                                                 scope=tik.scope_ubuf,
                                                 name="ub_a")
            param['ub_b'] = self.instance.Tensor(self.dtype, (shape,),
                                                 scope=tik.scope_ubuf,
                                                 name="ub_b")
            param['ub_c'] = self.instance.Tensor(self.dtype, (shape,),
                                                 scope=tik.scope_ubuf,
                                                 name="ub_c")
            param['last_32b'] = self.instance.Tensor(self.dtype, (BLOCK_SIZE,),
                                                     scope=tik.scope_ubuf,
                                                     name="last_32b")

            param['co_id'] = self.instance.Scalar()
            param['box_id'] = self.instance.Scalar()
            param['co_id'].set_as(cycle // self.boxes)
            param['box_id'].set_as(cycle % self.boxes)
            with self.instance.for_range(0, param['mov_loop']) as loop:
                param['burlen'] = self.instance.Scalar(name="burlen")
                repeat = self.instance.Scalar(name="repeat")

                with self.instance.if_scope(loop == param['mov_loop'] - 1):
                    param['burlen'].set_as(self.get_burlen(param['last_len']))
                    repeat.set_as(self.get_repeat(param['last_len']))
                with self.instance.else_scope():
                    param['burlen'].set_as(self.get_burlen(param['mov_len']))
                    repeat.set_as(self.get_repeat(param['mov_len']))

                # move coords data to ub a
                self.instance.data_move(param['ub_a'],
                                        param['in_data'][
                                            batch, cycle, param[
                                                'mov_len'] * loop],
                                        SID, NBURST_ONE,
                                        param['burlen'], GAP_ZERO, GAP_ZERO)

                self.compute_big_xy(batch, loop, param, repeat)

                self.compute_big_hw(batch, loop, param, repeat)

    def compute_big_hw(self, batch, loop, param, repeat):
        """
           compute big shape height and weight

           Parameters
           ----------
            batch: the number of picture
            loop: loop times
            param: a dict,the keys as fllow:
                   mov_len: the number of elements of each data move
                   mov_loop: data move loop times
                   last_len: the number of elements of last_len data move
                   ub_bias: a tensor,store bias
                   x_vmuls_val: a scalar
                   x_vadds_val: a scalar
                   y_vmuls_val: a scalar
                   ub_a: a tensor,store middle compute data
                   ub_b: a tensor,store middle compute data
                   ub_c: a tensor,store middle compute data
                   last_32b: a tensor,store last_32b data
                   co_id: a scalar,store co_id
                   box_id:  a scalar,store box_id
                   in_data: a tensor
            repeat: vector repeat times

           Returns
           -------
           None
           """
        tmp_scalar = self.instance.Scalar(self.dtype)
        bias_value = self.instance.Scalar(self.dtype)
        # h
        with self.instance.if_scope(param['co_id'] == VALUE_TWO):
            bias_value.set_as(
                param['ub_bias'][VALUE_TWO * param['box_id'] + VALUE_ONE])
            tmp_scalar.set_as(param['img_ub'][0])

            self.instance.vec_dup(self.mask, param['ub_b'], tmp_scalar, repeat,
                                     STRIDE_EIGHT)

            self.instance.vec_exp(self.mask, param['ub_c'], param['ub_a'], repeat,
                               STRIDE_EIGHT,
                               STRIDE_EIGHT)

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_c'],
                                bias_value,
                                repeat, STRIDE_EIGHT,
                                STRIDE_EIGHT)

            self.newton_div(param['ub_b'], param['ub_c'], param['ub_b'], repeat)

            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                param['y_vmuls_val'], repeat,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][2])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                    tmp_scalar,
                                    repeat,
                                    STRIDE_EIGHT,
                                    STRIDE_EIGHT)
            self.data_mov_out(batch, loop, param)
        # w
        with self.instance.if_scope(param['co_id'] == VALUE_THREE):
            bias_value.set_as(param['ub_bias'][VALUE_TWO * param['box_id']])

            # img ub: neth,netw,scaleh,scalew
            tmp_scalar.set_as(param['img_ub'][1])
            self.instance.vec_dup(self.mask, param['ub_b'], tmp_scalar, repeat,
                                     STRIDE_EIGHT)

            self.instance.vec_exp(self.mask, param['ub_c'], param['ub_a'], repeat,
                               STRIDE_EIGHT,
                               STRIDE_EIGHT)

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_c'],
                                bias_value,
                                repeat, STRIDE_EIGHT,
                                STRIDE_EIGHT)

            self.newton_div(param['ub_b'], param['ub_c'], param['ub_b'], repeat)
            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                param['x_vmuls_val'], repeat,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][3])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                    tmp_scalar,
                                    repeat,
                                    STRIDE_EIGHT,
                                    STRIDE_EIGHT)

            self.data_mov_out(batch, loop, param)

    def compute_big_xy(self, batch, cycle, param, repeat):
        """
           compute big shape of x,y

           Parameters
           ----------
            batch: the number of picture
            loop: loop times
            param: a dict,the keys as fllow:
                   mov_len: the number of elements of each data move
                   mov_loop: data move loop times
                   last_len: the number of elements of last_len data move
                   ub_bias: a tensor,store bias
                   x_vmuls_val: a scalar
                   x_vadds_val: a scalar
                   y_vmuls_val: a scalar
                   ub_a: a tensor,store middle compute data
                   ub_b: a tensor,store middle compute data
                   ub_c: a tensor,store middle compute data
                   last_32b: a tensor,store last_32b data
                   co_id: a scalar,store co_id
                   box_id:  a scalar,store box_id
                   in_data: a tensor
            repeat: vector repeat times

           Returns
           -------
           None
           """
        tmp_scalar = self.instance.Scalar(self.dtype)
        # x
        with self.instance.if_scope(param['co_id'] == VALUE_ZERO):
            # move windex to ub b
            self.instance.data_move(param['ub_b'],
                                    param['windex'][cycle * param['mov_len']],
                                    SID,
                                    NBURST_ONE,
                                    param['burlen'],
                                    GAP_ZERO, GAP_ZERO)

            # a = x + windex
            self.instance.vec_add(self.mask, param['ub_a'], param['ub_a'],
                               param['ub_b'],
                               repeat,
                               STRIDE_EIGHT, STRIDE_EIGHT,
                               STRIDE_EIGHT)
            # a = (x + windex)*(1/lw)
            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_a'],
                                (1.0 / param['w']), repeat,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)

            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                param['x_vmuls_val'], repeat,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)
            self.instance.vec_adds(self.mask, param['ub_b'], param['ub_b'],
                                param['x_vadds_val'], repeat,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][3])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                    tmp_scalar,
                                    repeat,
                                    STRIDE_EIGHT,
                                    STRIDE_EIGHT)

            self.data_mov_out(batch, cycle, param)
        # y
        with self.instance.if_scope(param['co_id'] == 1):
            # move hindex to ub
            self.instance.data_move(param['ub_b'],
                                    param['hindex'][cycle * param['mov_len']],
                                    SID,
                                    NBURST_ONE,
                                    param['burlen'],
                                    GAP_ZERO, GAP_ZERO)

            # a = y + hindex
            self.instance.vec_add(self.mask, param['ub_b'], param['ub_a'],
                               param['ub_b'],
                               repeat,
                               STRIDE_EIGHT, STRIDE_EIGHT,
                               STRIDE_EIGHT)
            # a = (y + hindex)*(1/lh)
            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                (1.0 / param['h']), repeat,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)

            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                param['y_vmuls_val'], repeat,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)
            self.instance.vec_adds(self.mask, param['ub_b'], param['ub_b'],
                                param['y_vadds_val'], repeat,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][2])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                    tmp_scalar,
                                    repeat,
                                    STRIDE_EIGHT,
                                    STRIDE_EIGHT)

            self.data_mov_out(batch, cycle, param)

    def data_mov_out(self, batch, loop, param):
        """
           move result to gm

           Parameters
           ----------
            batch: the number of picture
            loop: loop times
            param: a dict,the keys as fllow:
                   mov_loop: data move loop times
                   burlen: data move nburst
                   h: height
                   w: weight
                   out_offset: a scalar,store out_offset
                   ub_a: a tensor,store middle compute data
                   ub_b: a tensor,store middle compute data
                   ub_c: a tensor,store middle compute data
                   last_32b: a tensor,store last_32b data
                   co_id: a scalar,store co_id
                   box_id:  a scalar,store box_id
                   mov_len: the number of elements of each data move
           Returns
           -------
           None
           """
        if self.hwtail_len != VALUE_ZERO and param['h'] == self.height3 and \
                param['w'] == self.width3:
            with self.instance.if_scope(loop == param['mov_loop'] - VALUE_ONE):
                param['burlen'].set_as(param['burlen'] - VALUE_ONE)
                with self.instance.if_scope(param['burlen'] > VALUE_ZERO):
                    self.instance.data_move(
                        self.inter_coords[batch, param['co_id'] \
                            , param['out_offset'] + param['w'] * param['h'] * \
                                          param['box_id'] + \
                                          param['mov_len'] * loop],
                        param['ub_b'], SID, NBURST_ONE,
                        param['burlen'], GAP_ZERO, GAP_ZERO)

                param['burlen'].set_as(param['burlen'] + VALUE_ONE)
                tail_idx = self.instance.Scalar(name="tail_idx")
                tail_idx.set_as(param['last_len'] - self.len_32b)
                self.instance.data_move(param['last_32b'], self.inter_coords[
                    batch, param['co_id'], param['out_offset'] + param['w'] * \
                    param['h'] * param['box_id'] + param['mov_len'] * loop + \
                    tail_idx],
                                        SID, NBURST_ONE, VALUE_ONE, GAP_ZERO,
                                        GAP_ZERO)
                print("self.hwtail_len ", self.hwtail_len)
                with self.instance.for_range(VALUE_ZERO,
                                             self.hwtail_len) as cycle:
                    tmp_scalar = self.instance.Scalar(self.dtype)
                    tmp_scalar.set_as(param['ub_b'][param['last_len'] - \
                                                    self.hwtail_len + cycle])
                    param['last_32b'][self.len_32b - \
                                      self.hwtail_len + \
                                      cycle].set_as(tmp_scalar)
                self.instance.data_move(self.inter_coords[batch, param['co_id'] \
                    , param['out_offset'] + param['w'] * param['h'] * \
                                                          param['box_id'] + \
                                                          param['mov_len'] * \
                                                          loop + tail_idx],
                                        param['last_32b'], SID, NBURST_ONE,
                                        VALUE_ONE,
                                        GAP_ZERO, GAP_ZERO)
            with self.instance.else_scope():
                dest = self.inter_coords[batch, param['co_id'], \
                                         param['out_offset'] + \
                                         param['w'] * param['h'] * \
                                         param['box_id'] + \
                                         param['mov_len'] * loop]
                self.instance.data_move(dest, param['ub_b'], SID, NBURST_ONE,
                                        param['burlen'], GAP_ZERO, GAP_ZERO)
        else:
            dest = self.inter_coords[batch, param['co_id'], param['out_offset'] \
                                     + param['w'] * param['h'] * param['box_id'] \
                                     + param['mov_len'] * loop]
            self.instance.data_move(dest, param['ub_b'], SID, NBURST_ONE,
                                    param['burlen'],
                                    GAP_ZERO, GAP_ZERO)

    def small_surface_template(self, batch, param):
        """
          compute small shape

           Parameters
           ----------
            batch: the number of picture
            param: a dict,the keys as fllow:
                   mov_loop: data move loop times
                   burlen: data move nburst
                   out_offset: a scalar,store out_offset
                   ub_a: a tensor,store middle compute data
                   ub_b: a tensor,store middle compute data
                   ub_c: a tensor,store middle compute data
                   last_32b: a tensor,store last_32b data
                   co_id: a scalar,store co_id
                   box_id:  a scalar,store box_id
                   mov_len: the number of elements of each data move
                   x_vmuls_val: a scalar,store x_vmuls_val
                   y_vadds_val: a scalar,store y_vadds_val
                   y_vmuls_val: a scalar,store y_vmuls_val
                   adj_hw: a scalar,store adj_hw
           Returns
           -------
           None
           """
        param['ub_bias'] = self.convert_biases_data(param)
        param['x_vmuls_val'], param['x_vadds_val'], param['y_vmuls_val'], param[
            'y_vadds_val'] = self.get_x_y_params(param['img_ub'])

        param['adj_hw'] = self.get_adj_hw(param['h'], param['w'])
        param['faces_one_loop'], param['last_loop'], param[
            'loop'] = self.get_faces_params(param['adj_hw'], 4 * self.boxes)

        with self.instance.for_range(0, param['loop']) as loop_idx:
            param['ub_a'] = self.instance.Tensor(self.dtype,
                                                 (
                                                     self.one_max_size // self.dsize,),
                                                 scope=tik.scope_ubuf,
                                                 name="ub_a")
            param['ub_b'] = self.instance.Tensor(self.dtype,
                                                 (
                                                     self.one_max_size // self.dsize,),
                                                 scope=tik.scope_ubuf,
                                                 name="ub_b")
            param['ub_c'] = self.instance.Tensor(self.dtype,
                                                 (
                                                     self.one_max_size // self.dsize,),
                                                 scope=tik.scope_ubuf,
                                                 name="ub_c")
            param['last_32b'] = self.instance.Tensor(self.dtype,
                                                     (BLOCK_SIZE,),
                                                     scope=tik.scope_ubuf,
                                                     name="last_32b")

            param['faces'] = self.instance.Scalar("int32")
            with self.instance.if_scope(loop_idx != param['loop'] - 1):
                param['faces'].set_as(param['faces_one_loop'])
            with self.instance.else_scope():
                param['faces'].set_as(param['last_loop'])

            param['burlen'] = self.instance.Scalar()
            param['burlen'].set_as(
                (param['faces'] * param['adj_hw'] * self.dsize) // BLOCK_SIZE)

            # move coords gm to ub_a
            self.instance.data_move(param['ub_a'],
                                    param['in_data'][
                                        batch, param[
                                            'faces_one_loop'] * loop_idx, 0],
                                    SID,
                                    NBURST_ONE, param['burlen'], GAP_ZERO,
                                    GAP_ZERO)

            with self.instance.for_range(0, param['faces'], thread_num=2) as cycle:
                # Calculate the cindex.
                start_idx = self.instance.Scalar()
                # start_idx.set_as((faces_in_one_loop * loop_idx + fi) * adj_hw)
                start_idx.set_as(cycle * param['adj_hw'])

                # Indicates the number of the box.
                param['box_id'] = self.instance.Scalar()
                param['box_id'].set_as(
                    (param['faces_one_loop'] * loop_idx + cycle) % self.boxes)

                param['co_id'] = self.instance.Scalar()
                param['co_id'].set_as(
                    (param['faces_one_loop'] * loop_idx + cycle) // self.boxes)

                # burlen and repeat for move out
                param['burlen'].set_as(self.get_burlen(param["h"]*param["w"]))
                repeat = self.get_repeat(param["h"]*param["w"])

                self.compute_small_xy(batch, param, repeat, start_idx)

                self.compute_small_hw(batch, param, repeat, start_idx)

    def compute_small_hw(self, batch, param, repeat, start_idx):
        """
          compute small shape of height and weight

           Parameters
           ----------
            batch: the number of picture
            param: a dict,the keys as fllow:
                   ub_a: a tensor,store middle compute data
                   ub_b: a tensor,store middle compute data
                   ub_c: a tensor,store middle compute data
                   last_32b: a tensor,store last_32b data
                   co_id: a scalar,store co_id
                   box_id: a scalar,store box_id
                   img_ub: a tensor,store img data
                   x_vmuls_val: a scalar,store x_vmuls_val
                   y_vmuls_val: a scalar,store y_vmuls_val
                   ub_bias:  a tensor,store bias data

            repeat: vector repeat times
            start_idx: a scalar,store start_idx

           Returns
           -------
           None
           """
        tmp_scalar = self.instance.Scalar(self.dtype)
        bias_value = self.instance.Scalar(self.dtype)
        with self.instance.if_scope(param['co_id'] == VALUE_TWO):
            bias_value.set_as(
                param['ub_bias'][VALUE_TWO * param['box_id'] + VALUE_ONE])
            tmp_scalar.set_as(param['img_ub'][0])

            self.instance.vec_dup(self.mask, param['ub_b'], tmp_scalar, repeat,
                                     STRIDE_EIGHT)

            self.instance.vec_exp(self.mask, param['ub_c'], param['ub_a'][start_idx],
                               repeat,
                               STRIDE_EIGHT,
                               STRIDE_EIGHT)

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_c'],
                                bias_value,
                                repeat,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)

            self.newton_div(param['ub_b'], param['ub_c'], param['ub_b'],
                            repeat)
            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                param['y_vmuls_val'], repeat,
                                STRIDE_EIGHT, STRIDE_EIGHT)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][2])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                    tmp_scalar,
                                    repeat,
                                    STRIDE_EIGHT,
                                    STRIDE_EIGHT)

            self.t_small_mov_to_gm(batch, param)
        with self.instance.if_scope(param['co_id'] == VALUE_THREE):
            bias_value.set_as(param['ub_bias'][VALUE_TWO * param['box_id']])
            tmp_scalar.set_as(param['img_ub'][1])
            self.instance.vec_dup(self.mask, param['ub_b'], tmp_scalar, repeat,
                                     STRIDE_EIGHT)

            self.instance.vec_exp(self.mask, param['ub_c'], param['ub_a'][start_idx],
                               repeat,
                               STRIDE_EIGHT,
                               STRIDE_EIGHT)

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_c'],
                                bias_value,
                                repeat,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)

            self.newton_div(param['ub_b'], param['ub_c'], param['ub_b'],
                            repeat)
            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                param['x_vmuls_val'], repeat,
                                STRIDE_EIGHT, STRIDE_EIGHT)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][3])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                    tmp_scalar,
                                    repeat,
                                    STRIDE_EIGHT,
                                    STRIDE_EIGHT)

            self.t_small_mov_to_gm(batch, param)

    def compute_small_xy(self, batch, param, repeat, start_idx):
        """
          compute small shape of x,y

           Parameters
           ----------
            batch: the number of picture
            param: a dict,the keys as fllow:
                   ub_a: a tensor,store middle compute data
                   ub_b: a tensor,store middle compute data
                   ub_c: a tensor,store middle compute data
                   last_32b: a tensor,store last_32b data
                   co_id: a scalar,store co_id
                   box_id: a scalar,store box_id
                   img_ub: a tensor,store img data
                   x_vmuls_val: a scalar,store x_vmuls_val
                   y_vmuls_val: a scalar,store y_vmuls_val
                   ub_bias:  a tensor,store bias data

            repeat: vector repeat times
            start_idx: a scalar,store start_idx

           Returns
           -------
           None
           """
        tmp_scalar = self.instance.Scalar(self.dtype)
        with self.instance.if_scope(param['co_id'] == VALUE_ZERO):
            self.instance.data_move(param['ub_b'], param['windex'], SID,
                                    NBURST_ONE,
                                    param['burlen'], GAP_ZERO, GAP_ZERO)

            self.instance.vec_add(self.mask, param['ub_b'], param['ub_a'][start_idx],
                               param['ub_b'], repeat,
                               STRIDE_EIGHT, STRIDE_EIGHT,
                               STRIDE_EIGHT)
            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                (1.0 / param['w']),
                                repeat, STRIDE_EIGHT,
                                STRIDE_EIGHT)

            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                param['x_vmuls_val'], repeat,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)
            self.instance.vec_adds(self.mask, param['ub_b'], param['ub_b'],
                                param['x_vadds_val'], repeat,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][3])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                    tmp_scalar,
                                    repeat,
                                    STRIDE_EIGHT,
                                    STRIDE_EIGHT)

            self.t_small_mov_to_gm(batch, param)

        with self.instance.if_scope(param['co_id'] == VALUE_ONE):
            self.instance.data_move(param['ub_b'], param['hindex'], SID,
                                    NBURST_ONE,
                                    param['burlen'], GAP_ZERO, GAP_ZERO)

            # a = y + hindex
            self.instance.vec_add(self.mask, param['ub_b'], param['ub_a'][start_idx],
                               param['ub_b'], repeat,
                               STRIDE_EIGHT, STRIDE_EIGHT,
                               STRIDE_EIGHT)

            # a = (y + hindex)*(1/lh)
            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                (1.0 / param['h']),
                                repeat, STRIDE_EIGHT,
                                STRIDE_EIGHT)
            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                param['y_vmuls_val'], repeat,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)
            self.instance.vec_adds(self.mask, param['ub_b'], param['ub_b'],
                                param['y_vadds_val'], repeat,
                                STRIDE_EIGHT,
                                STRIDE_EIGHT)
            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][2])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                    tmp_scalar,
                                    repeat,
                                    STRIDE_EIGHT,
                                    STRIDE_EIGHT)

            self.t_small_mov_to_gm(batch, param)
