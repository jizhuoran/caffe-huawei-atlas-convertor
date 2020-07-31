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

BoundingBoxEncode
"""
from te import tik
from te import platform as tbe_platform
from topi.cce import util


# the number of bits per byte
THREAD_NUM = 2
# the number of data contained in each coordinate box
DEFAULT_NBURST = 1
# The Maximum number of float16 data can store in UB with pingpong (256 * 15)
MAX_UB_ELEMENT_NUMBER_FP16 = 5120
# The Maximum number of float32 data can store in UB with pingpong(128 * 15)
MAX_UB_ELEMENT_NUMBER_FP32 = 2560
# the number of blocks included in each repeat with float16
BLOCK_NUMBER_FP16 = 32
# the number of blocks included in each repeat with float32
BLOCK_NUMBER_FP32 = 64
# one block size takes up 32b
BLOCK_SIZE = 32


# pylint: disable=too-many-instance-attributes
class BoundingBoxEncode():
    """
    Funtion: use to store BoundingBoxEncode base parameters
    """
    # pylint: disable=too-many-arguments
    def __init__(self, anchorbox, ground_truth_box, delta, means, stds,
                 kernel_name):
        self.init_tik_instance()
        self.anchor_box_shape = anchorbox.get("shape")
        self.anchor_box_dtype = anchorbox.get("dtype").lower()
        self.ground_truth_shape = ground_truth_box.get("shape")
        self.ground_truth_dtype = ground_truth_box.get("dtype").lower()
        self.delta_shape = delta.get("shape")
        self.delta_shape = delta.get("dtype").lower()
        self.means = means
        self.stds = stds
        self.kernel_name = kernel_name
        self.core_num = 32
        self.data_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.anchor_box_dtype) // 8
        self.data_num_in_each_block = BLOCK_SIZE // self.data_dtype_bytes_size
        self.each_core_start_addr, self.each_core_calcul_num = \
            self.get_core_param()
        self.ub_max_size = MAX_UB_ELEMENT_NUMBER_FP16
        self.init_gm_tensor()

        if self.anchor_box_dtype == "float32":
            self.ub_max_size = MAX_UB_ELEMENT_NUMBER_FP32
        self.loop_cycle = self.get_loop_cycle()
        self.start_block_addr, self.block_number = self.get_loop_param()
        self.repeat_times = self.get_repeat_cycle()

    def init_tik_instance(self):
        """init_tik_instance

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik()
        self.support_div = tbe_platform.cce_conf.api_check_support(
            "tik.vdiv", "float32")

    def data_move_mte2_function(self, loop_input, block_number):
        """data_move_mte2_function

        Parameters
        ----------
        loop_input : int
            loop index
        block_number: int
            block_number

        Returns
        -------
        result : list
            [anchor_box_ub, ground_truth_in_ub]
        """
        anchor_box_ub = self.tik_instance.Tensor(
            self.anchor_box_dtype, (self.ub_max_size // 4, 4),
            name="anchor_box_ub",
            scope=tik.scope_ubuf)
        self.tik_instance.data_move(anchor_box_ub,
                                    self.anchorbox_in[loop_input], 0,
                                    DEFAULT_NBURST, block_number, 1, 1, 1)
        ground_truth_in_ub = self.tik_instance.Tensor(
            self.ground_truth_dtype, (self.ub_max_size // 4, 4),
            name="ground_truth_in_ub",
            scope=tik.scope_ubuf)
        self.tik_instance.data_move(ground_truth_in_ub,
                                    self.ground_truth_in[loop_input], 0,
                                    DEFAULT_NBURST, block_number, 1, 1, 1)
        return anchor_box_ub, ground_truth_in_ub

    def data_move_mte3_function(self, loop_input, block_num, delta_dst_ub):
        """data_move_mte3_function

        Parameters
        ----------
        loop_input : int
            loop index
        block_num: int
            block_number
        delta_dst_ub : addr
            delta_dst_ub

        Returns
        -------
        None
        """
        self.tik_instance.data_move(self.delta_out[loop_input], delta_dst_ub,
                                    0, DEFAULT_NBURST, block_num, 0, 0)

    def get_repeat_cycle(self):
        """data_move_mte2_function

        Parameters
        ----------
        None

        Returns
        -------
        result : int
            repeat_times
        """
        each_repeat_block_number = BLOCK_NUMBER_FP16
        if self.anchor_box_dtype == "float32":
            each_repeat_block_number = BLOCK_NUMBER_FP32
        if self.block_number < each_repeat_block_number:
            repeat_times = 1
        elif self.block_number % each_repeat_block_number == 0:
            repeat_times = self.block_number // each_repeat_block_number
        else:
            repeat_times = self.block_number // each_repeat_block_number + 1
        return repeat_times

    def get_core_param(self):
        """calculate data in number, each core start address
        """
        data_in_number = self.anchor_box_shape[0] * self.anchor_box_shape[1]
        each_core_start_addr = (data_in_number // (self.core_num * 4)) * 4

        # check input data number can equal divivde to (32 core * 4 point)
        if data_in_number % (self.core_num * 4) == 0:
            # check input data number is equal to block
            if each_core_start_addr % self.data_num_in_each_block == 0:
                each_core_calcul_num = each_core_start_addr
            else:
                each_core_calcul_num = (
                    each_core_start_addr // self.data_num_in_each_block + 1
                ) * self.data_num_in_each_block
        else:
            each_core_calcul_num = data_in_number - each_core_start_addr * (
                self.core_num - 1)
            if each_core_start_addr % self.data_num_in_each_block != 0:
                each_core_calcul_num = (
                    each_core_calcul_num // self.data_num_in_each_block + 1
                ) * self.data_num_in_each_block
        return each_core_start_addr, each_core_calcul_num

    def set_means_stds_scalar(self, means, stds):
        """set_means_stds_scalar"""
        dtype = "float16"
        # set means value [0, 0, 0, 0]
        means_0_scalar = self.tik_instance.Scalar(dtype, name="means_0_scalar")
        means_0_scalar.set_as(means[0])
        means_1_scalar = self.tik_instance.Scalar(dtype, name="means_1_scalar")
        means_1_scalar.set_as(means[1])
        means_2_scalar = self.tik_instance.Scalar(dtype, name="means_2_scalar")
        means_2_scalar.set_as(means[2])
        means_3_scalar = self.tik_instance.Scalar(dtype, name="means_3_scalar")
        means_3_scalar.set_as(means[3])

        # set stds value [1, 1, 1, 1]
        stds_0_scalar = self.tik_instance.Scalar(dtype, name="stds_0_scalar")
        stds_0_scalar.set_as(stds[0])
        stds_1_scalar = self.tik_instance.Scalar(dtype, name="stds_1_scalar")
        stds_1_scalar.set_as(stds[1])
        stds_2_scalar = self.tik_instance.Scalar(dtype, name="stds_2_scalar")
        stds_2_scalar.set_as(stds[2])
        stds_3_scalar = self.tik_instance.Scalar(dtype, name="stds_3_scalar")
        stds_3_scalar.set_as(stds[3])

        return (means_0_scalar, means_1_scalar, means_2_scalar, means_3_scalar,
                stds_0_scalar, stds_1_scalar, stds_2_scalar, stds_3_scalar)

    def tik_instance_function(self):
        """tik_instance_function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(
                0, self.core_num, block_num=self.core_num) as block_id:
            self.calculation_process(block_id)
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.anchorbox_in, self.ground_truth_in],
            outputs=[self.delta_out])

    def init_gm_tensor(self):
        """init_gm_tensor

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        gm_shape_size = self.each_core_start_addr * (
            self.core_num - 1) + self.each_core_calcul_num
        self.anchorbox_in = self.tik_instance.Tensor(
            self.anchor_box_dtype, (gm_shape_size // 4, 4),
            name="anchorbox_in",
            scope=tik.scope_gm)
        self.ground_truth_in = self.tik_instance.Tensor(
            self.anchor_box_dtype, (gm_shape_size // 4, 4),
            name="ground_truth_in",
            scope=tik.scope_gm)
        self.delta_out = self.tik_instance.Tensor(
            self.anchor_box_dtype, (gm_shape_size // 4, 4),
            name="delta_out",
            scope=tik.scope_gm)

    def get_loop_cycle(self):
        """get_loop_cycle

        Parameters
        ----------
        None

        Returns
        -------
        result : int
            loop_cycle
        """
        if self.each_core_calcul_num % self.ub_max_size == 0:
            loop_cycle = int(self.each_core_calcul_num // self.ub_max_size)
        else:
            loop_cycle = int(self.each_core_calcul_num // self.ub_max_size + 1)

        return loop_cycle

    def get_loop_param(self):
        """get_loop_param

        Parameters
        ----------
        None

        Returns
        -------
        result : list
            [start_block_addr, block_number]
        """
        block_number = self.each_core_calcul_num // self.data_num_in_each_block
        if block_number == 0:
            block_number = 1
        start_block_addr = block_number // self.loop_cycle

        if self.loop_cycle > 1:
            if block_number % self.loop_cycle != 0:
                block_number_loop = block_number - start_block_addr * (
                    self.loop_cycle - 1)
                if block_number_loop*16 > MAX_UB_ELEMENT_NUMBER_FP16:
                    self.loop_cycle += 1
                    start_block_addr = block_number // self.loop_cycle
                    block_number_loop = block_number - start_block_addr * (
                        self.loop_cycle - 1)
                block_number = block_number_loop
            else:
                block_number = start_block_addr
        return start_block_addr, block_number

    def calculation_process(self, block_id):
        """get_loop_param

        Parameters
        ----------
        block_id : int
            block_id

        Returns
        -------
        None
        """
        scalar_list = self.set_means_stds_scalar(self.means, self.stds)
        if self.loop_cycle == 1:
            loop_input = block_id * self.each_core_start_addr
            anchorbox_src_ub, groundtruthbox_src_ub = \
                self.data_move_mte2_function(loop_input, self.block_number)
            delta_dst_ub = self.bounding_box_encode_compute(
                scalar_list, self.repeat_times, anchorbox_src_ub,
                groundtruthbox_src_ub)
            self.data_move_mte3_function(loop_input, self.block_number,
                                         delta_dst_ub)
        else:
            loop_input = block_id * self.each_core_start_addr
            with self.tik_instance.for_range(
                    0, self.loop_cycle, thread_num=THREAD_NUM) as cycle:
                loop_input = loop_input + cycle * self.start_block_addr \
                             * self.data_num_in_each_block
                anchorbox_src_ub, groundtruthbox_src_ub = \
                    self.data_move_mte2_function(loop_input, self.block_number)
                delta_dst_ub = self.bounding_box_encode_compute(
                    scalar_list, self.repeat_times, anchorbox_src_ub,
                    groundtruthbox_src_ub)
                self.data_move_mte3_function(loop_input, self.block_number,
                                             delta_dst_ub)

    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    def bounding_box_encode_compute(self, scalar_list, repeat_times,
                                    anchorbox_src_ub, groundtruthbox_src_ub):
        """use tik instruction to calculate result bounding_box_encode_compute

        Parameters
        ----------
        scalar_list : list
            block_id
        repeat_times : int
            repeat_times
        anchorbox_src_ub : TVM tensor
            anchorbox_src_ub
        groundtruthbox_src_ub : TVM tensor
            groundtruthbox_src_ub

        Returns
        -------
        delta_out_ub : TVM tensor
        """
        anchorbox_dst_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="anchorbox_dst_ub",
            scope=tik.scope_ubuf)
        groundtruthbox_dst_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="groundtruthbox_dst_ub",
            scope=tik.scope_ubuf)

        # convert float32 to float16
        anchorbox_vconv_src_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="anchorbox_vconv_src_ub",
            scope=tik.scope_ubuf)
        groundtruthbox_vconv_src_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="groundtruthbox_vconv_src_ub",
            scope=tik.scope_ubuf)

        if self.anchor_box_dtype == "float32":
            self.tik_instance.vconv(64, 'none', anchorbox_vconv_src_ub,
                                    anchorbox_src_ub, repeat_times * 8,
                                    1, 1, 4, 8)
            self.tik_instance.vconv(64, 'none', groundtruthbox_vconv_src_ub,
                                    groundtruthbox_src_ub, repeat_times * 8, 1,
                                    1, 4, 8)
        else:
            anchorbox_vconv_src_ub = anchorbox_src_ub
            groundtruthbox_vconv_src_ub = groundtruthbox_src_ub

        # transverse input data use vnchwconv instruction
        anchorbox_src_list = [anchorbox_vconv_src_ub[16 * i]
                              for i in range(16)]
        anchorbox_dst_list = [anchorbox_dst_ub[16 * i] for i in range(16)]

        groundtruthbox_src_list = [
            groundtruthbox_vconv_src_ub[16 * i] for i in range(16)
        ]
        groundtruthbox_dst_list = [
            groundtruthbox_dst_ub[16 * i] for i in range(16)
        ]

        anchorbox_ptmp_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="anchorbox_ptmp_ub",
            scope=tik.scope_ubuf)
        groundtruthbox_ptmp_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="groundtruthbox_ptmp_ub",
            scope=tik.scope_ubuf)

        # transform anchorbox and groundtruth box
        self.tik_instance.vnchwconv(True, True, anchorbox_dst_list,
                                    anchorbox_src_list, repeat_times * 2, 16,
                                    16)
        self.tik_instance.vnchwconv(True, True, groundtruthbox_dst_list,
                                    groundtruthbox_src_list, repeat_times * 2,
                                    16, 16)

        # Calculate px, py, pw, ph
        anchorbox_dst_ub16 = anchorbox_dst_ub[16]
        anchorbox_dst_ub32 = anchorbox_dst_ub[32]
        anchorbox_dst_ub48 = anchorbox_dst_ub[48]
        anchorbox_ptmp_ub16 = anchorbox_ptmp_ub[16]
        anchorbox_ptmp_ub32 = anchorbox_ptmp_ub[32]
        anchorbox_ptmp_ub48 = anchorbox_ptmp_ub[48]
        groundtruthbox_ptmp_ub16 = groundtruthbox_ptmp_ub[16]
        groundtruthbox_ptmp_ub32 = groundtruthbox_ptmp_ub[32]
        groundtruthbox_ptmp_ub48 = groundtruthbox_ptmp_ub[48]
        groundtruthbox_dst_ub16 = groundtruthbox_dst_ub[16]
        groundtruthbox_dst_ub32 = groundtruthbox_dst_ub[32]
        groundtruthbox_dst_ub48 = groundtruthbox_dst_ub[48]
        self.tik_instance.vadd(128, anchorbox_ptmp_ub,
                               anchorbox_dst_ub,
                               anchorbox_dst_ub32, repeat_times, 4, 4, 4, 32,
                               32, 32)
        self.tik_instance.vmuls(128, anchorbox_ptmp_ub,
                                anchorbox_ptmp_ub,
                                0.5, repeat_times, 4, 4, 32, 32)

        self.tik_instance.vadd(128, anchorbox_ptmp_ub16,
                               anchorbox_dst_ub16,
                               anchorbox_dst_ub48, repeat_times, 4, 4, 4, 32,
                               32, 32)
        self.tik_instance.vmuls(128, anchorbox_ptmp_ub16,
                                anchorbox_ptmp_ub16, 0.5, repeat_times, 4, 4,
                                32, 32)

        self.tik_instance.vsub(128, anchorbox_ptmp_ub32,
                               anchorbox_dst_ub32,
                               anchorbox_dst_ub, repeat_times, 4, 4, 4, 32,
                               32, 32)
        self.tik_instance.vadds(128, anchorbox_ptmp_ub32,
                                anchorbox_ptmp_ub32, 1, repeat_times, 4, 4,
                                32, 32)

        if self.support_div == False:
            rec_1 = groundtruthbox_ptmp_ub32
            rec_2 = groundtruthbox_ptmp_ub48
            self.tik_instance.vrec(128, rec_1, anchorbox_ptmp_ub32,
                                   repeat_times,
                                   4, 4, 32, 32)
            self.tik_instance.vmul(128, rec_2,
                                   rec_1,
                                   anchorbox_ptmp_ub32, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)
            self.tik_instance.vmuls(128, rec_2,
                                    rec_2, -1, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vadds(128, rec_2,
                                    rec_2, 2, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vmul(128, rec_2,
                                   rec_2,
                                   rec_1, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)
            self.tik_instance.vmul(128, rec_1,
                                   rec_2,
                                   anchorbox_ptmp_ub32, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)
            self.tik_instance.vmuls(128, rec_1,
                                    rec_1, -1, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vadds(128, rec_1,
                                    rec_1, 2, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vmul(128, anchorbox_ptmp_ub32,
                                   rec_1,
                                   rec_2, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)

        self.tik_instance.vsub(128, anchorbox_ptmp_ub48,
                               anchorbox_dst_ub48,
                               anchorbox_dst_ub16, repeat_times, 4, 4, 4, 32,
                               32, 32)
        self.tik_instance.vadds(128, anchorbox_ptmp_ub48,
                                anchorbox_ptmp_ub48, 1, repeat_times, 4, 4,
                                32, 32)
        if self.support_div == False:
            rec_1 = groundtruthbox_ptmp_ub32
            rec_2 = groundtruthbox_ptmp_ub48
            self.tik_instance.vrec(128, rec_1, anchorbox_ptmp_ub48,
                                   repeat_times,
                                   4, 4, 32, 32)
            self.tik_instance.vmul(128, rec_2,
                                   rec_1,
                                   anchorbox_ptmp_ub48, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)
            self.tik_instance.vmuls(128, rec_2,
                                    rec_2, -1, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vadds(128, rec_2,
                                    rec_2, 2, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vmul(128, rec_2,
                                   rec_2,
                                   rec_1, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)
            self.tik_instance.vmul(128, rec_1,
                                   rec_2,
                                   anchorbox_ptmp_ub48, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)
            self.tik_instance.vmuls(128, rec_1,
                                    rec_1, -1, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vadds(128, rec_1,
                                    rec_1, 2, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vmul(128, anchorbox_ptmp_ub48,
                                   rec_1,
                                   rec_2, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)

        # Calculate gx, gy, gw, gh
        self.tik_instance.vadd(
            128, groundtruthbox_ptmp_ub, groundtruthbox_dst_ub,
            groundtruthbox_dst_ub32, repeat_times, 4, 4, 4, 32, 32, 32)
        self.tik_instance.vmuls(128, groundtruthbox_ptmp_ub,
                                groundtruthbox_ptmp_ub, 0.5, repeat_times,
                                4, 4, 32, 32)

        self.tik_instance.vadd(
            128, groundtruthbox_ptmp_ub16, groundtruthbox_dst_ub16,
            groundtruthbox_dst_ub48, repeat_times, 4, 4, 4, 32, 32, 32)
        self.tik_instance.vmuls(128, groundtruthbox_ptmp_ub16,
                                groundtruthbox_ptmp_ub16, 0.5, repeat_times,
                                4, 4, 32, 32)

        self.tik_instance.vsub(
            128, groundtruthbox_ptmp_ub32, groundtruthbox_dst_ub32,
            groundtruthbox_dst_ub, repeat_times, 4, 4, 4, 32, 32, 32)
        self.tik_instance.vadds(128, groundtruthbox_ptmp_ub32,
                                groundtruthbox_ptmp_ub32, 1, repeat_times, 4,
                                4, 32, 32)

        self.tik_instance.vsub(
            128, groundtruthbox_ptmp_ub48, groundtruthbox_dst_ub48,
            groundtruthbox_dst_ub16, repeat_times, 4, 4, 4, 32, 32, 32)
        self.tik_instance.vadds(128, groundtruthbox_ptmp_ub48,
                                groundtruthbox_ptmp_ub48, 1, repeat_times, 4,
                                4, 32, 32)

        # Calculate dx, dy, dw, dh
        delta_tmp_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="delta_tmp_ub",
            scope=tik.scope_ubuf)

        self.tik_instance.vsub(128, delta_tmp_ub, groundtruthbox_ptmp_ub,
                               anchorbox_ptmp_ub, repeat_times, 4, 4, 4, 32,
                               32, 32)

        if self.support_div == True:
            self.tik_instance.vdiv(128, delta_tmp_ub, delta_tmp_ub,
                                   anchorbox_ptmp_ub32, repeat_times,
                                   4, 4, 4, 32, 32, 32)
        else:
            self.tik_instance.vmul(128, delta_tmp_ub, delta_tmp_ub,
                                   anchorbox_ptmp_ub32, repeat_times,
                                   4, 4, 4, 32, 32, 32)
        self.tik_instance.vadds(128, delta_tmp_ub, delta_tmp_ub,
                                scalar_list[0], repeat_times, 4, 4, 32, 32)
        self.tik_instance.vmuls(128, delta_tmp_ub, delta_tmp_ub,
                                scalar_list[5], repeat_times, 4, 4, 32, 32)

        # dy = ( (gy - py)/ph + (-means[1]) * (1/stds[1])
        delta_tmp_ub16 = delta_tmp_ub[16]
        self.tik_instance.vsub(
            128, delta_tmp_ub16, groundtruthbox_ptmp_ub16,
            anchorbox_ptmp_ub16, repeat_times, 4, 4, 4, 32, 32, 32)

        if self.support_div == True:
            self.tik_instance.vdiv(128, delta_tmp_ub16, delta_tmp_ub16,
                                   anchorbox_ptmp_ub48, repeat_times,
                                   4, 4, 4, 32, 32, 32)
        else:
            self.tik_instance.vmul(128, delta_tmp_ub16, delta_tmp_ub16,
                                   anchorbox_ptmp_ub48, repeat_times,
                                   4, 4, 4, 32, 32, 32)

        self.tik_instance.vadds(128, delta_tmp_ub16, delta_tmp_ub16,
                                scalar_list[1], repeat_times, 4, 4, 32, 32)
        self.tik_instance.vmuls(128, delta_tmp_ub16, delta_tmp_ub16,
                                scalar_list[5], repeat_times, 4, 4, 32, 32)

        # dw = ( log(gw/pw) + (-means[2]) * (1/stds[2])
        delta_tmp_ub32 = delta_tmp_ub[32]
        if self.support_div == True:
            self.tik_instance.vdiv(
                128, delta_tmp_ub32, groundtruthbox_ptmp_ub32,
                anchorbox_ptmp_ub32, repeat_times, 4, 4, 4, 32, 32, 32)
        else:
            self.tik_instance.vmul(
                128, delta_tmp_ub32, groundtruthbox_ptmp_ub32,
                anchorbox_ptmp_ub32, repeat_times, 4, 4, 4, 32, 32, 32)

        self.tik_instance.vln(128, delta_tmp_ub32, delta_tmp_ub32,
                              repeat_times, 4, 4, 32, 32)
        self.tik_instance.vadds(128, delta_tmp_ub32, delta_tmp_ub32,
                                scalar_list[2], repeat_times, 4, 4, 32, 32)
        self.tik_instance.vmuls(128, delta_tmp_ub32, delta_tmp_ub32,
                                scalar_list[6], repeat_times, 4, 4, 32, 32)

        # dy = ( log(gh/ph) + (-means[3]) * (1/stds[3])
        delta_tmp_ub48 = delta_tmp_ub[48]
        if self.support_div == True:
            self.tik_instance.vdiv(
                128, delta_tmp_ub48, groundtruthbox_ptmp_ub48,
                anchorbox_ptmp_ub48, repeat_times, 4, 4, 4, 32, 32, 32)
        else:
            self.tik_instance.vmul(
                128, delta_tmp_ub48, groundtruthbox_ptmp_ub48,
                anchorbox_ptmp_ub48, repeat_times, 4, 4, 4, 32, 32, 32)

        self.tik_instance.vln(128, delta_tmp_ub48, delta_tmp_ub48,
                              repeat_times, 4, 4, 32, 32)
        self.tik_instance.vadds(128, delta_tmp_ub48, delta_tmp_ub48,
                                scalar_list[3], repeat_times, 4, 4, 32, 32)
        self.tik_instance.vmuls(128, delta_tmp_ub48, delta_tmp_ub48,
                                scalar_list[7], repeat_times, 4, 4, 32, 32)

        # transverse output data back
        delta_out_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="delta_out_ub",
            scope=tik.scope_ubuf)
        delta_out_fp32_ub = self.tik_instance.Tensor(
            "float32", (self.ub_max_size,),
            name="delta_out_fp32_ub",
            scope=tik.scope_ubuf)

        delta_tmp_list = [delta_tmp_ub[16 * i] for i in range(16)]
        delta_out_list = [delta_out_ub[16 * i] for i in range(16)]

        self.tik_instance.vnchwconv(True, True, delta_out_list, delta_tmp_list,
                                    repeat_times * 2, 16, 16)

        if self.anchor_box_dtype == "float32":
            self.tik_instance.vconv(64, 'none', delta_out_fp32_ub,
                                    delta_out_ub, repeat_times * 8, 1, 1, 8, 4)
            delta_out_ub = delta_out_fp32_ub
        return delta_out_ub


# pylint: disable=too-many-arguments
@util.check_input_type(dict, dict, dict, (tuple, list), (tuple, list), str)
def bounding_box_encode(anchorbox_in_dict,
                        ground_truth_in_dict,
                        delta_out_dict,
                        means_attrs=(0, 0, 0, 0),
                        stds_attrs=(1, 1, 1, 1),
                        kernel_name_val="bounding_box_encode"):
    """
    algorithm: bounding_box_encode

    Parameters
    ----------
    anchorbox_in_dict : dict
        shape and dtype of input
    ground_truth_in_dict : dict
        shape and dtype of input
    delta_out_dict : dict
        shape and dtype of output, should be same shape and type as input
    means_attrs : list
        shape and dtype of output, should be same shape and type as input
    stds_attrs : list
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "bounding_box_encode"

    Returns
    -------
    None
    """
    anchor_box_shape = anchorbox_in_dict.get("shape")
    ground_truth_box_shape = ground_truth_in_dict.get("shape")

    util.check_shape_rule(anchor_box_shape)
    util.check_tensor_shape_size(anchor_box_shape)

    util.check_shape_rule(ground_truth_box_shape)
    util.check_tensor_shape_size(ground_truth_box_shape)

    bounding_box_encode_ = BoundingBoxEncode(
        anchorbox_in_dict, ground_truth_in_dict, delta_out_dict, means_attrs,
        stds_attrs, kernel_name_val)

    bounding_box_encode_.tik_instance_function()

    return bounding_box_encode_.tik_instance
