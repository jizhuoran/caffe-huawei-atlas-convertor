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
# pylint: disable=too-many-lines

import math
from te import tik
from impl import constant_util as constant
from impl import common_util
from te import platform as tbe_platform

# pylint: disable=too-many-instance-attributes,too-few-public-methods
class SSDDectionParamInit():
    """
    define SSDDectionParamInit class

    """
    def __init__(self, input_dict):
        """
        SSDDectionParamInit init

        Parameters
        ----------
        input_dict: input dict

        Returns
        -------
        None
        """
        self.dtype = input_dict.get("mbox_loc").get("dtype").lower()
        self.dsize = common_util.get_data_size(self.dtype)

        #self.ub_size = tik.Dprofile().get_unified_buffer_size()
        self.ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
        ub_capacity = self.ub_size // self.dsize

        self.burnest_len = constant.BLOCK_SIZE // self.dsize
        self.ub_capacity = (ub_capacity // self.burnest_len) * self.burnest_len
        self.mask = 64 if self.dtype == "float32" else 128

        self.batch = input_dict.get("mbox_loc").get("shape")[0]
        self.loc_num = input_dict.get("mbox_loc").get("shape")[1]
        self.num_classes = input_dict.get("num_classes")
        self.loc_coord = input_dict.get("mbox_loc").get("shape")[1] // 4

        self.background_label_id = input_dict.get("background_label_id")
# pylint: disable=too-many-instance-attributes,too-many-public-methods
class SSDDecodeBBox(SSDDectionParamInit):
    """
    define SSDDecodeBBox class

    """
    def __init__(self, input_dict, tik_instance):
        """
        SSDDecodeBBox init

        Parameters
        ----------
        input_dict: input dict
        tik_instance: tik instance

        Returns
        -------
        None
        """
        self.instance = tik_instance
        super(SSDDecodeBBox, self).__init__(input_dict)
        self.init_decode_bbox_args(input_dict)

        # input data
        add_tail_num = 256
        mbox_loc_len = self.get_shape_total_number(
            input_dict.get("mbox_loc").get("shape"))
        self.mbox_loc_gm = self.instance.Tensor(self.dtype, (mbox_loc_len+add_tail_num, ),
                                                name="mbox_loc_gm",
                                                scope=tik.scope_gm)

        mbox_conf_len = self.get_shape_total_number(
            input_dict.get("mbox_conf").get("shape"))
        self.mbox_conf_gm = self.instance.Tensor(self.dtype, (mbox_conf_len+add_tail_num, ),
                                                 name="mbox_conf_gm",
                                                 scope=tik.scope_gm)

        mbox_priorbox_len = self.get_shape_total_number(
            input_dict.get("mbox_priorbox").get("shape"))
        self.mbox_prior_gm = self.instance.Tensor(self.dtype,
                                                  (mbox_priorbox_len+add_tail_num, ),
                                                  name="mbox_prior_gm",
                                                  scope=tik.scope_gm)

        # parser input data
        # loc_coord_num_align = self.loc_coord + self.burnest_len
        loc_coord_num_align = math.ceil(self.loc_coord / 16) * 16 + self.burnest_len
        self.loc_data_parser_gm = self.instance.Tensor(
            self.dtype,
            (self.batch, 4, loc_coord_num_align),
            name="loc_data_parser_gm",
            is_workspace=True,
            scope=tik.scope_gm)

        self.conf_data_parser_gm = self.instance.Tensor(
            self.dtype,
            (self.batch, self.num_classes, loc_coord_num_align),
            name="conf_data_parser_gm",
            is_workspace=True,
            scope=tik.scope_gm)

        self.prior_bbox_parser_gm = self.instance.Tensor(
            self.dtype,
            (self.batch, 4, loc_coord_num_align),
            name="prior_bbox_parser_gm",
            is_workspace=True,
            scope=tik.scope_gm)

        self.prior_variance_parser_gm = self.instance.Tensor(
            self.dtype,
            (self.batch, 4, loc_coord_num_align),
            name="prior_variance_parser_gm",
            is_workspace=True,
            scope=tik.scope_gm)

        # decode bbox out data
        self.decode_bbox_out_gm = self.instance.Tensor(
            self.dtype,
            (self.batch, self.num_classes, loc_coord_num_align, 8),
            name="decode_bbox_out_gm",
            is_workspace=True,
            scope=tik.scope_gm)

    def init_decode_bbox_args(self, input_dict):
        """
        decode bbox args init

        Parameters
        ----------
        input_dict: input dict

        Returns
        -------
        None
        """
        self.variance_encoded_in_target = input_dict.get("variance_encoded_in_target")
        self.code_type = input_dict.get("code_type")

        each_take_ub_size = 10240 * self.dsize
        each_handle_unit_num = 512
        ub_unit_num = self.ub_size // each_take_ub_size

        self.handle_each_src = each_handle_unit_num * ub_unit_num
        self.handle_each_dst = self.handle_each_src // 4

        self.handle_loops_num = self.loc_num // self.handle_each_src
        self.handle_tail_actual_num = self.loc_num - self.handle_loops_num * self.handle_each_src

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

    # pylint: disable=no-self-use
    def get_shape_total_number(self, shape):
        """
        get shape total number

        Parameters
        ----------
        shape: shape info

        Returns
        -------
        None
        """
        total_number = len(shape)
        if total_number == 0:
            return 0
        total_number = 1
        for i in shape:
            total_number = total_number * i

        return total_number

    def compute_detection_out(self, batch):
        """
        compute decection out

        Parameters
        ----------
        batch: batch

        Returns
        -------
        None
        """
        with self.instance.new_stmt_scope():
            data_offset = self.instance.Scalar("int32", "data_offset", 0)

            with self.instance.for_range(0, self.handle_loops_num) as cycle_index:
                data_offset.set_as(cycle_index * self.handle_each_src)
                self.comput_decoced_bbox(batch, data_offset, False)

            if self.handle_tail_actual_num > 0:
                data_tail_offet = self.handle_loops_num * self.handle_each_src
                self.comput_decoced_bbox(batch, data_tail_offet, True)

            tail_tensor_num = math.ceil(self.loc_coord / 16) * 16 - self.loc_coord
            if tail_tensor_num != 0:
                self.set_decode_bbox_output_tail_conf(batch, tail_tensor_num)

    def compute_decode_bbox_coord_corner(self, batch, data_offset, is_tail,
                                         decode_bbox_ori):
        """
        compute decode bbox code type 1

        Parameters
        ----------
        batch: batch
        data_offset: data offset
        is_tail: whether tail
        decode_bbox_ori: decode bbox original data

        Returns
        -------
        None
        """
        # 1. get loc data
        loc_dst_ub = self.instance.Tensor(self.dtype,
                                          (4, self.handle_each_dst),
                                          name="loc_dst_ub",
                                          scope=tik.scope_ubuf)

        self.get_loc_data(batch, data_offset, loc_dst_ub, is_tail)

        # 2. get prior bbox data
        prior_bbox_dest_ub = self.instance.Tensor(
            self.dtype, (4, self.handle_each_dst), name="prior_bbox_dest_ub",
            scope=tik.scope_ubuf)
        prior_var_dest_ub = self.instance.Tensor(
            self.dtype, (4, self.handle_each_dst), name="prior_var_dest_ub",
            scope=tik.scope_ubuf)

        self.get_priorbox_data((batch, data_offset, prior_bbox_dest_ub,
                                prior_var_dest_ub), is_tail)

        # 3. computer bbox
        handle_each_dst_loops = self.handle_each_dst // self.mask
        with self.instance.for_range(0, 4) as computer_index:
            # 3.1 true
            with self.instance.if_scope(self.variance_encoded_in_target):
                self.instance.vadd(self.mask,
                                   decode_bbox_ori[computer_index, 0],
                                   prior_bbox_dest_ub[computer_index, 0],
                                   loc_dst_ub[computer_index, 0],
                                   handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
            # 3.2 false
            with self.instance.else_scope():
                self.instance.vmul(self.mask,
                                   decode_bbox_ori[computer_index, 0],
                                   prior_var_dest_ub[computer_index, 0],
                                   loc_dst_ub[computer_index, 0],
                                   handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
                self.instance.vadd(self.mask,
                                   decode_bbox_ori[computer_index, 0],
                                   prior_bbox_dest_ub[computer_index, 0],
                                   decode_bbox_ori[computer_index, 0],
                                   handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

    def compute_prior_bbox_center_size(self, batch, data_offset, is_tail):
        """
        compute decection out code type 2

        Parameters
        ----------
        batch: batch
        data_offset: data offset
        is_tail: whether tail

        Returns
        -------
        None
        """
        # 1. get bbox
        prior_bbox_dest_ub = self.instance.Tensor(
            self.dtype, (4, self.handle_each_dst), name="prior_bbox_dest_ub",
            scope=tik.scope_ubuf)
        prior_var_dest_ub = self.instance.Tensor(
            self.dtype, (4, self.handle_each_dst), name="prior_var_dest_ub",
            scope=tik.scope_ubuf)

        self.get_priorbox_data((batch, data_offset, prior_bbox_dest_ub,
                                prior_var_dest_ub), is_tail)

        prior_width = self.instance.Tensor(self.dtype, (self.handle_each_dst, ),
                                           name="prior_width", scope=tik.scope_ubuf)
        prior_height = self.instance.Tensor(self.dtype, (self.handle_each_dst, ),
                                            name="prior_height", scope=tik.scope_ubuf)
        prior_center_x = self.instance.Tensor(self.dtype, (self.handle_each_dst, ),
                                              name="prior_center_x", scope=tik.scope_ubuf)
        prior_center_y = self.instance.Tensor(self.dtype, (self.handle_each_dst, ),
                                              name="prior_center_y", scope=tik.scope_ubuf)

        handle_each_dst_loops = self.handle_each_dst // self.mask

        self.instance.vsub(self.mask, prior_width,
                           prior_bbox_dest_ub[2, 0],
                           prior_bbox_dest_ub[0, 0],
                           handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
        self.instance.vsub(self.mask, prior_height,
                           prior_bbox_dest_ub[3, 0],
                           prior_bbox_dest_ub[1, 0],
                           handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

        self.instance.vadd(self.mask, prior_center_x,
                           prior_bbox_dest_ub[2, 0],
                           prior_bbox_dest_ub[0, 0],
                           handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
        self.instance.vmuls(self.mask, prior_center_x,
                            prior_center_x,
                            0.5, handle_each_dst_loops, 1, 1, 8, 8)

        self.instance.vadd(self.mask, prior_center_y,
                           prior_bbox_dest_ub[3, 0],
                           prior_bbox_dest_ub[1, 0],
                           handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
        self.instance.vmuls(self.mask, prior_center_y,
                            prior_center_y,
                            0.5, handle_each_dst_loops, 1, 1, 8, 8)

        return prior_width, prior_height, prior_center_x, prior_center_y, prior_var_dest_ub

    # pylint: disable=too-many-locals
    def compute_decode_bbox_center_size(self, batch, data_offset, prior_data, is_tail):
        """
        compute decection out code type 2

        Parameters
        ----------
        batch: batch
        data_offset: data offset
        is_tail: whether tail

        Returns
        -------
        None
        """
        prior_width = prior_data[0]
        prior_height = prior_data[1]
        prior_center_x = prior_data[2]
        prior_center_y = prior_data[3]
        prior_var_dest_ub = prior_data[4]

        loc_dst_ub = self.instance.Tensor(self.dtype,
                                          (4, self.handle_each_dst),
                                          name="loc_dst_ub",
                                          scope=tik.scope_ubuf)

        self.get_loc_data(batch, data_offset, loc_dst_ub, is_tail)

        #2.2
        decode_bbox_center_x = self.instance.Tensor(
            self.dtype, (self.handle_each_dst, ), name="decode_bbox_center_x",
            scope=tik.scope_ubuf)
        decode_bbox_center_y = self.instance.Tensor(
            self.dtype, (self.handle_each_dst, ), name="decode_bbox_center_y",
            scope=tik.scope_ubuf)
        decode_bbox_width = self.instance.Tensor(
            self.dtype, (self.handle_each_dst, ), name="decode_bbox_width",
            scope=tik.scope_ubuf)
        decode_bbox_height = self.instance.Tensor(
            self.dtype, (self.handle_each_dst, ), name="decode_bbox_height",
            scope=tik.scope_ubuf)
        decode_bbox_vexp = self.instance.Tensor(
            self.dtype, (self.handle_each_dst, ), name="decode_bbox_vexp",
            scope=tik.scope_ubuf)


        handle_each_dst_loops = self.handle_each_dst // self.mask

        with self.instance.if_scope(self.variance_encoded_in_target):
            self.instance.vmul(self.mask,
                               decode_bbox_center_x,
                               loc_dst_ub[0, 0],
                               prior_width,
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
            self.instance.vadd(self.mask,
                               decode_bbox_center_x,
                               prior_center_x,
                               decode_bbox_center_x,
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

            self.instance.vmul(self.mask,
                               decode_bbox_center_y,
                               loc_dst_ub[1, 0],
                               prior_height,
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
            self.instance.vadd(self.mask,
                               decode_bbox_center_y,
                               prior_center_y,
                               decode_bbox_center_y,
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

            self.instance.vexp(self.mask,
                               decode_bbox_vexp,
                               loc_dst_ub[2, 0],
                               handle_each_dst_loops, 1, 1, 8, 8)
            self.instance.vmul(self.mask,
                               decode_bbox_width,
                               decode_bbox_vexp,
                               prior_width,
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

            self.instance.vexp(self.mask,
                               decode_bbox_vexp,
                               loc_dst_ub[3, 0],
                               handle_each_dst_loops, 1, 1, 8, 8)
            self.instance.vmul(self.mask,
                               decode_bbox_height,
                               decode_bbox_vexp,
                               prior_height,
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

        with self.instance.else_scope():
            self.instance.vmul(self.mask,
                               decode_bbox_center_x,
                               prior_var_dest_ub[0, 0],
                               loc_dst_ub[0, 0],
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
            self.instance.vmul(self.mask,
                               decode_bbox_center_x,
                               prior_width,
                               decode_bbox_center_x,
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
            self.instance.vadd(self.mask,
                               decode_bbox_center_x,
                               prior_center_x,
                               decode_bbox_center_x,
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

            self.instance.vmul(self.mask,
                               decode_bbox_center_y,
                               prior_var_dest_ub[1, 0],
                               loc_dst_ub[1, 0],
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
            self.instance.vmul(self.mask,
                               decode_bbox_center_y,
                               prior_height,
                               decode_bbox_center_y,
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
            self.instance.vadd(self.mask,
                               decode_bbox_center_y,
                               prior_center_y,
                               decode_bbox_center_y,
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

            self.instance.vmul(self.mask,
                               decode_bbox_width,
                               prior_var_dest_ub[2, 0],
                               loc_dst_ub[2, 0],
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
            self.instance.vexp(self.mask,
                               decode_bbox_vexp,
                               decode_bbox_width,
                               handle_each_dst_loops, 1, 1, 8, 8)
            self.instance.vmul(self.mask,
                               decode_bbox_width,
                               decode_bbox_vexp,
                               prior_width,
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

            self.instance.vmul(self.mask,
                               decode_bbox_height,
                               prior_var_dest_ub[3, 0],
                               loc_dst_ub[3, 0],
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
            self.instance.vexp(self.mask,
                               decode_bbox_vexp,
                               decode_bbox_height,
                               handle_each_dst_loops, 1, 1, 8, 8)
            self.instance.vmul(self.mask,
                               decode_bbox_height,
                               decode_bbox_vexp,
                               prior_height,
                               handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

        return decode_bbox_center_x, decode_bbox_center_y, decode_bbox_width, \
               decode_bbox_height

    def compute_decode_bbox_coord_center_size(self, decode_bbox_data,
                                              decode_bbox_ori):
        """
        compute decection out code type 2

        Parameters
        ----------
        decode_bbox_data: decode bbox data
        decode_bbox_ori: decode bbox original data

        Returns
        -------
        None
        """

        decode_bbox_center_x = decode_bbox_data[0]
        decode_bbox_center_y = decode_bbox_data[1]
        decode_bbox_width = decode_bbox_data[2]
        decode_bbox_height = decode_bbox_data[3]

        handle_each_dst_loops = self.handle_each_dst // self.mask

        self.instance.vmuls(self.mask, decode_bbox_width,
                            decode_bbox_width,
                            0.5, handle_each_dst_loops, 1, 1, 8, 8)
        self.instance.vsub(self.mask, decode_bbox_ori[0, 0],
                           decode_bbox_center_x,
                           decode_bbox_width,
                           handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

        self.instance.vmuls(self.mask, decode_bbox_height,
                            decode_bbox_height,
                            0.5, handle_each_dst_loops, 1, 1, 8, 8)
        self.instance.vsub(self.mask, decode_bbox_ori[1, 0],
                           decode_bbox_center_y,
                           decode_bbox_height,
                           handle_each_dst_loops, 1, 1, 1, 8, 8, 8)


        self.instance.vadd(self.mask, decode_bbox_ori[2, 0],
                           decode_bbox_center_x,
                           decode_bbox_width,
                           handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

        self.instance.vadd(self.mask, decode_bbox_ori[3, 0],
                           decode_bbox_center_y,
                           decode_bbox_height,
                           handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

    def compute_decode_bbox_coord_corner_size(self, batch, data_offset, is_tail,
                                              decode_bbox_ori):
        """
        compute decode bbox code type 3

        Parameters
        ----------
        batch: batch
        data_offset: data offset
        is_tail: whether tail
        decode_bbox_ori: decode bbox original data

        Returns
        -------
        None
        """
        # 1. get loc data
        loc_dst_ub = self.instance.Tensor(self.dtype,
                                          (4, self.handle_each_dst),
                                          name="loc_dst_ub",
                                          scope=tik.scope_ubuf)

        self.get_loc_data(batch, data_offset, loc_dst_ub, is_tail)

        # 2. get prior bbox data
        prior_bbox_dest_ub = self.instance.Tensor(
            self.dtype, (4, self.handle_each_dst), name="prior_bbox_dest_ub",
            scope=tik.scope_ubuf)
        prior_var_dest_ub = self.instance.Tensor(
            self.dtype, (4, self.handle_each_dst), name="prior_var_dest_ub",
            scope=tik.scope_ubuf)

        self.get_priorbox_data((batch, data_offset, prior_bbox_dest_ub,
                                prior_var_dest_ub), is_tail)

        # 3. prior width and height
        prior_width = self.instance.Tensor(self.dtype, (self.handle_each_dst, ),
                                           name="prior_width", scope=tik.scope_ubuf)
        prior_height = self.instance.Tensor(self.dtype, (self.handle_each_dst, ),
                                            name="prior_height", scope=tik.scope_ubuf)

        handle_each_dst_loops = self.handle_each_dst // self.mask
        self.instance.vsub(self.mask, prior_width,
                           prior_bbox_dest_ub[2, 0],
                           prior_bbox_dest_ub[0, 0],
                           handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
        self.instance.vsub(self.mask, prior_height,
                           prior_bbox_dest_ub[3, 0],
                           prior_bbox_dest_ub[1, 0],
                           handle_each_dst_loops, 1, 1, 1, 8, 8, 8)


        # 4. computer bbox

        # 4.1 true
        with self.instance.if_scope(self.variance_encoded_in_target):

            with self.instance.for_range(0, 4) as index:

                with self.instance.if_scope(index % 2 == 0):

                    self.instance.vmul(self.mask,
                                       decode_bbox_ori[index, 0],
                                       loc_dst_ub[index, 0],
                                       prior_width,
                                       handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

                with self.instance.else_scope():

                    self.instance.vmul(self.mask,
                                       decode_bbox_ori[index, 0],
                                       loc_dst_ub[index, 0],
                                       prior_height,
                                       handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

                self.instance.vadd(self.mask,
                                   decode_bbox_ori[index, 0],
                                   prior_bbox_dest_ub[index, 0],
                                   decode_bbox_ori[index, 0],
                                   handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

        # 4.2 false
        with self.instance.else_scope():

            with self.instance.for_range(0, 4) as index:

                self.instance.vmul(self.mask,
                                   decode_bbox_ori[index, 0],
                                   prior_var_dest_ub[index, 0],
                                   loc_dst_ub[index, 0],
                                   handle_each_dst_loops, 1, 1, 1, 8, 8, 8)

                with self.instance.if_scope(index % 2 == 0):
                    self.instance.vmul(self.mask,
                                       decode_bbox_ori[index, 0],
                                       prior_width,
                                       decode_bbox_ori[index, 0],
                                       handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
                with self.instance.else_scope():
                    self.instance.vmul(self.mask,
                                       decode_bbox_ori[index, 0],
                                       prior_height,
                                       decode_bbox_ori[index, 0],
                                       handle_each_dst_loops, 1, 1, 1, 8, 8, 8)
                self.instance.vadd(self.mask,
                                   decode_bbox_ori[index, 0],
                                   prior_bbox_dest_ub[index, 0],
                                   decode_bbox_ori[index, 0],
                                   handle_each_dst_loops, 1, 1, 1, 8, 8, 8)


    def clip_bbox(self, decode_bbox_ori):
        """
        clip bbox

        Parameters
        ----------
        batch: batch
        data_offset: data offset
        is_tail: whether tail

        Returns
        -------
        None
        """
        decode_bbox_clip = self.instance.Tensor(
            self.dtype, (4, self.handle_each_dst), name="decode_bbox_clip",
            scope=tik.scope_ubuf)
        val_0_ub = self.instance.Tensor(self.dtype, (self.mask,),
                                        name="val_0_ub", scope=tik.scope_ubuf)
        val_1_ub = self.instance.Tensor(self.dtype, (self.mask,),
                                        name="val_1_ub", scope=tik.scope_ubuf)
        self.instance.vector_dup(self.mask, val_0_ub, 0, 1, 1, 8)
        self.instance.vector_dup(self.mask, val_1_ub, 1, 1, 1, 8)

        with self.instance.for_range(0, 4) as clip_index:
            self.instance.vmin(self.mask, decode_bbox_clip[clip_index, 0],
                               val_1_ub,
                               decode_bbox_ori[clip_index, 0],
                               self.handle_each_dst//self.mask, 1, 1, 1, 8, 0, 8)
            self.instance.vmax(self.mask, decode_bbox_clip[clip_index, 0],
                               val_0_ub,
                               decode_bbox_clip[clip_index, 0],
                               self.handle_each_dst//self.mask, 1, 1, 1, 8, 0, 8)

        return decode_bbox_clip

    def adjust_decoded_bbox_coord(self, decode_bbox_clip):
        """
        adjust decection bbox coord

        Parameters
        ----------
        batch: batch
        data_offset: data offset
        is_tail: whether tail

        Returns
        -------
        None
        """
        self.instance.vadds(self.mask, decode_bbox_clip[0, 0],
                            decode_bbox_clip[0, 0], 1.0,
                            self.handle_each_dst // self.mask, 1, 1, 8, 8)
        self.instance.vadds(self.mask, decode_bbox_clip[1, 0],
                            decode_bbox_clip[1, 0], 1.0,
                            self.handle_each_dst // self.mask, 1, 1, 8, 8)

    def comput_decoced_bbox(self, batch, data_offset, is_tail):
        """
        compute decection out

        Parameters
        ----------
        batch: batch
        data_offset: data offset
        is_tail: whether tail

        Returns
        -------
        None
        """
        decode_bbox_ori = self.instance.Tensor(
            self.dtype, (4, self.handle_each_dst), name="decode_bbox_ori",
            scope=tik.scope_ubuf)

        with self.instance.if_scope(self.code_type == 1):
            self.compute_decode_bbox_coord_corner(batch, data_offset, is_tail,
                                                  decode_bbox_ori)

        with self.instance.if_scope(self.code_type == 2):
            prior_width, prior_height, prior_center_x, prior_center_y, prior_var_dest_ub = \
                self.compute_prior_bbox_center_size(batch, data_offset, is_tail)

            decode_bbox_center_x, decode_bbox_center_y, decode_bbox_width, decode_bbox_height = \
                self.compute_decode_bbox_center_size(batch, data_offset,
                                                     (prior_width, prior_height,
                                                      prior_center_x,
                                                      prior_center_y,
                                                      prior_var_dest_ub),
                                                     is_tail)

            self.compute_decode_bbox_coord_center_size(
                (decode_bbox_center_x, decode_bbox_center_y,
                 decode_bbox_width, decode_bbox_height), decode_bbox_ori)

        with self.instance.if_scope(self.code_type == 3):
            self.compute_decode_bbox_coord_corner_size(batch, data_offset,
                                                       is_tail, decode_bbox_ori)
        decode_bbox_clip = self.clip_bbox(decode_bbox_ori)
        self.adjust_decoded_bbox_coord(decode_bbox_clip)

        self.vconcat_decode_bbox_output(batch, decode_bbox_clip, data_offset, is_tail)

    def set_decode_bbox_output_tail_conf(self, batch, tail_tensor_num):
        """
        set decection out tail conf

        Parameters
        ----------
        batch: batch
        tail_tensor_num: tail tensor num

        Returns
        -------
        None
        """
        init_val = self.instance.Scalar(self.dtype, "init_val", 0)
        decode_bbox_output_tail = self.instance.Tensor(
            self.dtype, (16, 8), name="decode_bbox_output_tail", scope=tik.scope_ubuf)
        self.instance.vector_dup(self.mask, decode_bbox_output_tail, 0, 16 * 8 // self.mask, 1, 8)

        with self.instance.for_range(0, self.num_classes) as class_index:

            with self.instance.if_scope(class_index != self.background_label_id):
                self.instance.data_move(decode_bbox_output_tail,
                                        self.decode_bbox_out_gm[batch, class_index,
                                                                self.loc_coord, 0],
                                        0, 1,
                                        math.ceil(tail_tensor_num * 8 / self.burnest_len),
                                        0, 0)

                with self.instance.for_range(0, tail_tensor_num) as tail_index:
                    decode_bbox_output_tail[tail_index, 4].set_as(init_val)

                self.instance.data_move(self.decode_bbox_out_gm[batch, class_index,
                                                                self.loc_coord, 0],
                                        decode_bbox_output_tail,
                                        0, 1,
                                        math.ceil(tail_tensor_num * 8 / self.burnest_len),
                                        0, 0)

    def vconcat_decode_bbox_output(self, batch, decode_bbox_ori, data_offset, is_tail):
        """
        vconcat decode bbox out

        Parameters
        ----------
        batch: batch
        decode_bbox_ori: original decode bbox data
        data_offset: data offset
        is_tail: whether tail

        Returns
        -------
        None
        """
        decode_bbox_out_ub = self.instance.Tensor(
            self.dtype, (self.handle_each_dst*8, ), name="decode_bbox_out_ub",
            scope=tik.scope_ubuf)

        handle_each_vconcat_num = self.handle_each_dst // 16
        burst_val = self.handle_each_dst * 8 // self.burnest_len

        if self.handle_tail_actual_num > 0 and is_tail:
            handle_each_vconcat_num = math.ceil((self.handle_tail_actual_num // 4) / 16)
            burst_val = math.ceil((self.handle_tail_actual_num // 4 * 8) / self.burnest_len)

        vconcat_loops = handle_each_vconcat_num // 255
        vconcat_tail = handle_each_vconcat_num % 255

        with self.instance.for_range(0, vconcat_loops) as index:
            vconcat_offset = index * 255
            self.instance.vconcat(decode_bbox_out_ub[vconcat_offset],
                                  decode_bbox_ori[0, vconcat_offset], 255, 0)
            self.instance.vconcat(decode_bbox_out_ub[vconcat_offset],
                                  decode_bbox_ori[1, vconcat_offset], 255, 1)
            self.instance.vconcat(decode_bbox_out_ub[vconcat_offset],
                                  decode_bbox_ori[2, vconcat_offset], 255, 2)
            self.instance.vconcat(decode_bbox_out_ub[vconcat_offset],
                                  decode_bbox_ori[3, vconcat_offset], 255, 3)

        with self.instance.if_scope(vconcat_tail > 0):
            vconcat_offset = vconcat_loops * 255

            self.instance.vconcat(decode_bbox_out_ub[vconcat_offset],
                                  decode_bbox_ori[0, vconcat_offset],
                                  vconcat_tail, 0)
            self.instance.vconcat(decode_bbox_out_ub[vconcat_offset],
                                  decode_bbox_ori[1, vconcat_offset],
                                  vconcat_tail, 1)
            self.instance.vconcat(decode_bbox_out_ub[vconcat_offset],
                                  decode_bbox_ori[2, vconcat_offset],
                                  vconcat_tail, 2)
            self.instance.vconcat(decode_bbox_out_ub[vconcat_offset],
                                  decode_bbox_ori[3, vconcat_offset],
                                  vconcat_tail, 3)

        # vconcat conf data
        with self.instance.for_range(0, self.num_classes) as class_index:
            with self.instance.if_scope(class_index != self.background_label_id):
                #paser conf
                conf_dst_ub = self.instance.Tensor(
                    self.dtype, (self.handle_each_dst,), name="conf_dst_ub",
                    scope=tik.scope_ubuf)

                self.get_conf_data((batch, class_index, data_offset, conf_dst_ub),
                                   is_tail)

                with self.instance.for_range(0, vconcat_loops) as index:
                    vconcat_offset = index * 255
                    self.instance.vconcat(decode_bbox_out_ub[vconcat_offset],
                                          conf_dst_ub[vconcat_offset], 255, 4)

                with self.instance.if_scope(vconcat_tail > 0):
                    vconcat_offset = vconcat_loops * 255
                    self.instance.vconcat(decode_bbox_out_ub[vconcat_offset],
                                          conf_dst_ub[vconcat_offset],
                                          vconcat_tail, 4)

                self.instance.data_move(
                    self.decode_bbox_out_gm[batch, class_index, data_offset//4, 0],
                    decode_bbox_out_ub, 0, 1, burst_val, 0, 0)

    def handle_loc_data(self, batch, handle_loc_data_info, handle_loc_ub_info):
        """
        handle loc data

        Parameters
        ----------
        batch: batch
        handle_loc_data_info: handle loc data info
        handle_loc_ub_info: handle loc ub info
        Returns
        -------
        None
        """
        handle_actual_num = handle_loc_data_info[0]
        loc_gm_start = handle_loc_data_info[1]
        length = handle_loc_data_info[2]
        loc_gm_offset = handle_loc_data_info[3]
        loc_burst_val = handle_loc_data_info[4]

        loc_ub = handle_loc_ub_info[0]
        loc_vnch_ub = handle_loc_ub_info[1]

        # 1. gm to ub
        with self.instance.for_range(0, handle_actual_num // 4) as move_index:
            ub_start = 16 * move_index
            gm_start = 4 * move_index
            self.instance.data_move(loc_ub[ub_start],
                                    self.mbox_loc_gm[loc_gm_start+gm_start],
                                    0, 1, 1, 0, 0)
        # 2. vnchwconv
        tail_loop_times = ((length*16)//(16*16)) % 255
        dst_rep_stride = 16
        src_rep_stride = 16
        if tail_loop_times == 1:
            dst_rep_stride = 0
            src_rep_stride = 0

        src_list1 = [loc_ub[16*i] for i in range(16)]
        dst_list1 = [loc_vnch_ub[16*i] for i in range(16)]
        self.instance.vnchwconv(False, False, dst_list1, src_list1,
                                tail_loop_times, dst_rep_stride, src_rep_stride)

        # 3. ub to gm

        self.instance.data_move(self.loc_data_parser_gm[batch, 0, loc_gm_offset],
                                loc_vnch_ub[0], 0, loc_burst_val, 1, 15, 0)
        self.instance.data_move(self.loc_data_parser_gm[batch, 1, loc_gm_offset],
                                loc_vnch_ub[16], 0, loc_burst_val, 1, 15, 0)
        self.instance.data_move(self.loc_data_parser_gm[batch, 2, loc_gm_offset],
                                loc_vnch_ub[32], 0, loc_burst_val, 1, 15, 0)
        self.instance.data_move(self.loc_data_parser_gm[batch, 3, loc_gm_offset],
                                loc_vnch_ub[48], 0, loc_burst_val, 1, 15, 0)

    # pylint: disable=too-many-locals
    def parser_loc_data(self, batch):
        """
        parser loc data

        Parameters
        ----------
        batch: batch
        data_offset: data offset
        loc_dst_ub: mbox loc destination ub data
        is_tail: whether tail

        Returns
        -------
        None
        """
        with self.instance.new_stmt_scope():

            handle_unit_num = self.ub_size // (256 * 4 * 2 * self.dsize)
            loc_handle_num = math.ceil(self.loc_num / 256)
            if loc_handle_num > handle_unit_num:
                handle_num = 256 * handle_unit_num
            else:
                handle_num = 256 * loc_handle_num
            loc_handle_loops = self.loc_num // handle_num

            loc_ub = self.instance.Tensor(
                self.dtype, (handle_num * 4,), name="loc_ub", scope=tik.scope_ubuf)
            loc_vnch_ub = self.instance.Tensor(
                self.dtype, (handle_num * 4,), name="loc_vnch_ub", scope=tik.scope_ubuf)

            with self.instance.for_range(0, loc_handle_loops) as loc_handle_index:
                loc_gm_start = batch * self.loc_num + loc_handle_index * handle_num
                length = handle_num * 4 // 16
                loc_gm_offset = loc_handle_index * handle_num // 4
                loc_burst_val = math.ceil(handle_num // 4 / self.burnest_len)
                self.handle_loc_data(batch,
                                     (handle_num, loc_gm_start,
                                      length, loc_gm_offset, loc_burst_val),
                                     (loc_ub, loc_vnch_ub))

            if self.loc_num % handle_num > 0:
                loc_data_tail_num = self.loc_num - loc_handle_loops * handle_num
                loc_gm_start = batch * self.loc_num + loc_handle_loops * handle_num
                length = math.ceil(loc_data_tail_num / 64) * 64 * 4 // 16
                loc_gm_offset = loc_handle_loops * handle_num // 4
                loc_burst_val = math.ceil(loc_data_tail_num // 4 / self.burnest_len)
                self.handle_loc_data(batch,
                                     (loc_data_tail_num, loc_gm_start,
                                      length, loc_gm_offset, loc_burst_val),
                                     (loc_ub, loc_vnch_ub))


    # pylint: disable=too-many-locals
    def get_loc_data(self, batch, data_offset, loc_dst_ub, is_tail):
        """
        get loc data

        Parameters
        ----------
        batch: batch
        data_offset: data offset
        loc_dst_ub: mbox loc destination ub data
        is_tail: whether tail

        Returns
        -------
        None
        """

        loc_burst_val = self.handle_each_dst // self.burnest_len
        if is_tail:
            loc_burst_val = math.ceil(self.handle_tail_actual_num // 4 / self.burnest_len)

        with self.instance.for_range(0, 4) as loc_move_index:
            self.instance.data_move(loc_dst_ub[loc_move_index, 0],
                                    self.loc_data_parser_gm[batch, loc_move_index, data_offset//4],
                                    0, 1, loc_burst_val, 0, 0)


    # pylint: disable=too-many-locals
    def parser_conf_data(self, batch):
        """
        parser conf data

        Parameters
        ----------
        batch: batch

        Returns
        -------
        None
        """
        parser_num = 16 * math.ceil(self.num_classes / 16) * 16
        parser_times = math.ceil(self.loc_coord / 16)
        burst_in_val = (math.ceil(self.num_classes / 16) * 16) // self.burnest_len
        burst_out_val = 16 // self.burnest_len

        with self.instance.new_stmt_scope():
            conf_ub = self.instance.Tensor(self.dtype, (parser_num,),
                                           name="conf_ub", scope=tik.scope_ubuf)
            conf_vnch_ub = self.instance.Tensor(self.dtype, (parser_num,),
                                                name="conf_vnch_ub", scope=tik.scope_ubuf)
            conf_move_loops = self.instance.Scalar("int32", "conf_move_loops", 16)

            with self.instance.for_range(0, parser_times) as parser_index:

                with self.instance.if_scope(parser_index == parser_times - 1):
                    conf_move_loops.set_as(
                        (self.loc_coord * self.num_classes - parser_index * 16 * self.num_classes)
                        // self.num_classes)
                # 1. gm to ub:
                with self.instance.for_range(0, conf_move_loops) as move_index:
                    ub_start = move_index * (math.ceil(self.num_classes / 16) * 16)
                    gm_start = batch * self.loc_coord * self.num_classes + \
                               parser_index * self.num_classes * 16 + \
                               move_index * self.num_classes
                    self.instance.data_move(conf_ub[ub_start],
                                            self.mbox_conf_gm[gm_start], 0, 1,
                                            burst_in_val, 0, 0)

                # 2. vnchwconv:
                length = parser_num // 16
                tail_loop_times = ((length*16)//(16*16)) % 255

                src_list = [conf_ub[math.ceil(self.num_classes / 16) * 16*i] for i in range(16)]
                dst_list = [conf_vnch_ub[16*i] for i in range(16)]

                dst_rep_stride = 0 if self.num_classes <= 16 else 16
                src_rep_stride = 0 if self.num_classes // 16 == 0 else 1

                self.instance.vnchwconv(False, False, dst_list, src_list,
                                        tail_loop_times, dst_rep_stride,
                                        src_rep_stride)

                # 3. ub to gm
                with self.instance.for_range(0, self.num_classes) as class_index:
                    with self.instance.if_scope(class_index != self.background_label_id):
                        self.instance.data_move(
                            self.conf_data_parser_gm[batch, class_index, parser_index * 16],
                            conf_vnch_ub[class_index * 16],
                            0, 1, burst_out_val, 0, 0)

    def get_conf_data(self, conf_info, is_tail):
        """
        get conf data

        Parameters
        ----------
        batch: batch
        class_index: class index
        data_offset: data offset
        conf_dst_ub: mbox conf destination
        is_tail: whether tail

        Returns
        -------
        None
        """
        batch = conf_info[0]
        class_index = conf_info[1]
        data_offset = conf_info[2]
        conf_dst_ub = conf_info[3]

        conf_burst_val = self.handle_each_dst // self.burnest_len
        if is_tail:
            conf_burst_val = math.ceil(self.handle_tail_actual_num // 4 / self.burnest_len)

        self.instance.data_move(conf_dst_ub,
                                self.conf_data_parser_gm[batch, class_index, data_offset//4],
                                0, 1, conf_burst_val, 0, 0)

        init_val = self.instance.Scalar(self.dtype, "init_val", 0)
        with self.instance.if_scope(tik.all(is_tail, self.handle_tail_actual_num > 0)):
            conf_tail_num = self.handle_tail_actual_num // 4 % self.burnest_len
            with self.instance.if_scope(conf_tail_num != 0):
                tail_start = conf_burst_val * 16 - (self.burnest_len - conf_tail_num)
                with self.instance.for_range(0, self.burnest_len - conf_tail_num) as index:
                    conf_dst_ub[tail_start + index].set_as(init_val)



    def handle_priorbox_data(self, batch, handle_priorbox_data_info, handle_priorbox_ub_info):
        """
        handle priorbox data

        Parameters
        ----------
        batch: batch
        handle_priorbox_data_info: handle priorbox data info
        handle_priorbox_ub_info: handle priorbox ub info
        Returns
        -------
        None
        """
        handle_actual_num = handle_priorbox_data_info[0]
        bbox_gm_start = handle_priorbox_data_info[1]
        var_gm_start = handle_priorbox_data_info[2]
        length = handle_priorbox_data_info[3]
        priorbox_gm_offset = handle_priorbox_data_info[4]
        priorbox_burst_val = handle_priorbox_data_info[5]

        prior_bbox_ub = handle_priorbox_ub_info[0]
        prior_var_ub = handle_priorbox_ub_info[1]
        prior_bbox_vnch_ub = handle_priorbox_ub_info[2]
        prior_var_vnch_ub = handle_priorbox_ub_info[3]

        # 1. gm to ub
        with self.instance.for_range(0, handle_actual_num // 4) as move_index:
            ub_start = 16 * move_index
            gm_start = 4 * move_index

            self.instance.data_move(prior_bbox_ub[ub_start],
                                    self.mbox_prior_gm[bbox_gm_start+gm_start],
                                    0, 1, 1, 0, 0)
            self.instance.data_move(prior_var_ub[ub_start],
                                    self.mbox_prior_gm[var_gm_start+gm_start],
                                    0, 1, 1, 0, 0)

        # 2. vnchwconv
        tail_loop_times = ((length*16)//(16*16)) % 255
        dst_rep_stride = 16
        src_rep_stride = 16
        if tail_loop_times == 1:
            dst_rep_stride = 0
            src_rep_stride = 0

        src_list1 = [prior_bbox_ub[16*i] for i in range(16)]
        dst_list1 = [prior_bbox_vnch_ub[16*i] for i in range(16)]
        self.instance.vnchwconv(False, False, dst_list1, src_list1,
                                tail_loop_times, dst_rep_stride, src_rep_stride)

        src_list2 = [prior_var_ub[16*i] for i in range(16)]
        dst_list2 = [prior_var_vnch_ub[16*i] for i in range(16)]
        self.instance.vnchwconv(False, False, dst_list2, src_list2,
                                tail_loop_times, dst_rep_stride, src_rep_stride)
        # 3. ub to gm
        self.instance.data_move(self.prior_bbox_parser_gm[batch, 0, priorbox_gm_offset],
                                prior_bbox_vnch_ub, 0,
                                priorbox_burst_val, 1, 15, 0)
        self.instance.data_move(self.prior_bbox_parser_gm[batch, 1, priorbox_gm_offset],
                                prior_bbox_vnch_ub[16], 0,
                                priorbox_burst_val, 1, 15, 0)
        self.instance.data_move(self.prior_bbox_parser_gm[batch, 2, priorbox_gm_offset],
                                prior_bbox_vnch_ub[32], 0,
                                priorbox_burst_val, 1, 15, 0)
        self.instance.data_move(self.prior_bbox_parser_gm[batch, 3, priorbox_gm_offset],
                                prior_bbox_vnch_ub[48], 0,
                                priorbox_burst_val, 1, 15, 0)

        self.instance.data_move(self.prior_variance_parser_gm[batch, 0, priorbox_gm_offset],
                                prior_var_vnch_ub, 0,
                                priorbox_burst_val, 1, 15, 0)
        self.instance.data_move(self.prior_variance_parser_gm[batch, 1, priorbox_gm_offset],
                                prior_var_vnch_ub[16], 0,
                                priorbox_burst_val, 1, 15, 0)
        self.instance.data_move(self.prior_variance_parser_gm[batch, 2, priorbox_gm_offset],
                                prior_var_vnch_ub[32], 0,
                                priorbox_burst_val, 1, 15, 0)
        self.instance.data_move(self.prior_variance_parser_gm[batch, 3, priorbox_gm_offset],
                                prior_var_vnch_ub[48], 0,
                                priorbox_burst_val, 1, 15, 0)

    # pylint: disable=too-many-locals
    def parser_priorbox_data(self, batch):
        """
        parser priorbox data

        Parameters
        ----------
        batch: batch
        data_offset: data offset
        is_tail: whether tail

        Returns
        -------
        None
        """

        with self.instance.new_stmt_scope():
            handle_unit_num = self.ub_size // (256 * 4 * 4 * self.dsize)
            prior_handle_num = math.ceil(self.loc_num / 256)
            if prior_handle_num > handle_unit_num:
                handle_num = 256 * handle_unit_num
            else:
                handle_num = 256 * prior_handle_num
            priorbox_handle_loops = self.loc_num // handle_num

            prior_bbox_ub = self.instance.Tensor(self.dtype, (handle_num * 4,),
                                                 name="prior_bbox_ub",
                                                 scope=tik.scope_ubuf)
            prior_var_ub = self.instance.Tensor(self.dtype, (handle_num * 4,),
                                                name="prior_var_ub",
                                                scope=tik.scope_ubuf)
            prior_bbox_vnch_ub = self.instance.Tensor(self.dtype, (handle_num * 4,),
                                                      name="prior_bbox_vnch_ub",
                                                      scope=tik.scope_ubuf)
            prior_var_vnch_ub = self.instance.Tensor(self.dtype, (handle_num * 4,),
                                                     name="prior_var_vnch_ub",
                                                     scope=tik.scope_ubuf)

            with self.instance.for_range(0, priorbox_handle_loops) as priorbox_handle_index:

                bbox_gm_start = batch * self.loc_num * 2  + priorbox_handle_index * handle_num
                var_gm_start = batch * self.loc_num * 2 + self.loc_num + \
                               priorbox_handle_index * handle_num
                length = handle_num * 4 // 16
                priorbox_gm_offset = priorbox_handle_index * handle_num // 4
                priorbox_burst_val = math.ceil(handle_num // 4 / self.burnest_len)

                self.handle_priorbox_data(batch,
                                          (handle_num, bbox_gm_start, var_gm_start,
                                           length, priorbox_gm_offset, priorbox_burst_val),
                                          (prior_bbox_ub, prior_var_ub,
                                           prior_bbox_vnch_ub, prior_var_vnch_ub))

            if self.loc_num % handle_num > 0:
                priorbox_data_tail_num = self.loc_num - priorbox_handle_loops * handle_num
                bbox_gm_start = batch * self.loc_num * 2  + priorbox_handle_loops * handle_num
                var_gm_start = batch * self.loc_num * 2 + self.loc_num + \
                               priorbox_handle_loops * handle_num
                length = math.ceil(priorbox_data_tail_num / 64) * 64 * 4 // 16
                priorbox_gm_offset = priorbox_handle_loops * handle_num // 4
                priorbox_burst_val = math.ceil(priorbox_data_tail_num // 4 / self.burnest_len)

                self.handle_priorbox_data(batch,
                                          (priorbox_data_tail_num, bbox_gm_start, var_gm_start,
                                           length, priorbox_gm_offset, priorbox_burst_val),
                                          (prior_bbox_ub, prior_var_ub,
                                           prior_bbox_vnch_ub, prior_var_vnch_ub))

    # pylint: disable=too-many-locals
    def get_priorbox_data(self, prior_bbox_info, is_tail):
        """
        get priorbox data

        Parameters
        ----------
        batch: batch
        data_offset: data offset
        is_tail: whether tail

        Returns
        -------
        None
        """
        batch = prior_bbox_info[0]
        data_offset = prior_bbox_info[1]
        prior_bbox_dest_ub = prior_bbox_info[2]
        prior_var_dest_ub = prior_bbox_info[3]

        priorbox_burst_val = self.handle_each_dst // self.burnest_len
        if is_tail:
            priorbox_burst_val = math.ceil(self.handle_tail_actual_num // 4 / self.burnest_len)

        with self.instance.for_range(0, 4) as priorbox_move_index:

            self.instance.data_move(prior_bbox_dest_ub[priorbox_move_index, 0],
                                    self.prior_bbox_parser_gm[batch,
                                                              priorbox_move_index,
                                                              data_offset//4],
                                    0, 1, priorbox_burst_val, 0, 0)
            self.instance.data_move(prior_var_dest_ub[priorbox_move_index, 0],
                                    self.prior_variance_parser_gm[batch,
                                                                  priorbox_move_index,
                                                                  data_offset//4],
                                    0, 1, priorbox_burst_val, 0, 0)
