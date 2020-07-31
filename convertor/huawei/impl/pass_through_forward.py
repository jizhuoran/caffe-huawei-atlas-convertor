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

depth_to_space
"""
# pylint: disable=too-many-lines,import-error
from functools import reduce as functools_reduce
from te import tik
from te import platform as tbe_platform

# byte of type int8, uint8
SIZE_ONE_BYTE = 1
# bytes of type int16, uint16, float16
SIZE_TWO_BYTES = 2
# bytes of type int32, uint32, float32
SIZE_FOUR_BYTES = 4
# bytes of type uint64
SIZE_EIGHT_BYTES = 8
# minimum unit of data_move: 32Bytes
DATA_MOVE_MIN_UNIT = 32
# maximum burst number
MAX_N_BURST = 4095


def pass_through_forward_func(feature_dic, stride):
    """
    interface of passth_reverse: depth_to_space

    Parameters
    ----------
    feature_dic: dict
                 dict with keys(shape, dtype) of input
    stride: int
            the size of the spatial block

    Returns
    -------
    tik_instance: tik_instance, input_x_gm, output_y_gm
    """
    input_shape = feature_dic.get("shape")
    input_dtype = feature_dic.get("dtype").lower()

    passth_d2s = DepthToSpace(input_shape, input_dtype, stride)
    return passth_d2s.run_depth_to_space_computer()


def _cal_core(tik_instance, total_core_loop_num, num_core, core_number):
    """
    calculate the loop number on each core
    """
    #max_core_num = tik.Dprofile().get_aicore_num()
    max_core_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
    core_loop = tik_instance.Scalar("uint64")
    sum_core = tik_instance.Scalar("uint64")
    core_loop.set_as(total_core_loop_num//core_number)
    sum_core.set_as((core_loop + 1)*(total_core_loop_num % max_core_num) +
                    core_loop*(num_core - total_core_loop_num % max_core_num))
    with tik_instance.if_scope(num_core < total_core_loop_num % max_core_num):
        core_loop.set_as((total_core_loop_num + core_number - 1) // core_number)
        sum_core.set_as(core_loop*num_core)

    return core_loop, sum_core


# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-many-statements
# pylint: disable=attribute-defined-outside-init
class DepthToSpace:
    """
    Rearranges data from depth into blocks of spatial data

    Functions
    ----------
    __init__:
        initialize some properties
    set_tik_instance:
        set tik_instance
    set_src_dst_tensor:
        set input and output tensor
    set_tiling_axis:
        calculate the tiling axis
    set_core_num:
        set the block_num
    cal_core:
        alculate the loop number on each core
    data_move_gm2ub:
        move data from GM to UB according to rules
        in the case of 32B not-aligned
    data_move_gm2ub_align:
        move data from GM to UB according to rules
        in the case of 32B aligned
    data_move_ub2gm_case_division:
        divide the case of move data from UB to GM
    data_move_ub2gm:
        move data from UB to GM, which can be divided into 32B aligned
        and non-aligned cases
    data_move_ub2gm_padding:
        move data from UB to GM, in the case of 32B not-aligned and
        the data in UB is less than 32B
    data_move_ub2gm_align:
        move data from UB to GM, in the case of 32B aligned
    depth_to_space_axis_zero:
        data move process when tililng_axis = 0
    depth_to_space_axis_one:
        data move process when tililng_axis = 1
    depth_to_space_axis_two:
        data move process when tililng_axis = 2
    depth_to_space_axis_three:
        data move process when tililng_axis = 3
    depth_to_space_axis_four:
        data move process when tililng_axis = 4
    depth_to_space_axis_five:
        data move process when tililng_axis = 5
    depth_to_space_compute:
        the overall data move process
    get_tik_instance:
        obtain tik instance

    Returns
    -------
    None
    """
    def __init__(self, input_shape, dtype, block_size):
        """
        initialize some properties
        """
        (self.input_batch, self.input_height,
         self.input_width, self.input_depth) = input_shape
        self.input_shape = input_shape
        self.block_size = block_size
        self.dtype = dtype
        self.output_batch = self.input_batch
        self.output_height = self.input_height*block_size
        self.output_width = self.input_width*block_size
        self.output_depth = self.input_depth//(block_size*block_size)

        num_bit = SIZE_TWO_BYTES
        if self.dtype in ("int8", "uint8"):
            num_bit = SIZE_ONE_BYTE
        elif self.dtype in ("int32", "uint32", "float32"):
            num_bit = SIZE_FOUR_BYTES
        elif self.dtype in ("uint64", "int64"):
            num_bit = SIZE_EIGHT_BYTES
        # the number of data that can be moved in each data_move
        self.num_data = DATA_MOVE_MIN_UNIT//num_bit

        self.output_shape = (self.output_batch, self.output_height,
                             self.output_width, self.output_depth)

        # the number of data that UB can put in
        total_ub_memory = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
        self.ub_memory = total_ub_memory//num_bit - self.num_data
        # minimum granularity of data_move
        self.min_size = self.block_size*self.output_depth
        # minimum granularity of data_move actually
        self.num_data_one_move = (self.min_size + self.num_data - 1) // \
                                 self.num_data*self.num_data
        # axis for tiling
        self.tiling_shape = (self.input_batch, self.input_height,
                             self.block_size, self.input_width,
                             self.num_data_one_move)
        # 32B align
        self.is_align = ((self.min_size % self.num_data) == 0)

    def set_tik_instance(self):
        """
        set tik_instance
        """
        tik_instance = tik.Tik(tik.Dprofile())
        self.set_src_dst_tensor(tik_instance)

        return tik_instance

    def set_src_dst_tensor(self, tik_instance):
        """
        set input and output tensor
        """
        shape_gm_in = self.input_batch * self.input_height * \
                      self.input_width * self.input_depth + 32
        shape_gm_out = self.output_batch * self.output_height * \
                       self.output_width * self.output_depth + 32
        self.input_x_gm = tik_instance.Tensor(self.dtype,
                                              (shape_gm_in, ),
                                              name="input_x_gm",
                                              scope=tik.scope_gm)
        self.output_y_gm = tik_instance.Tensor(self.dtype,
                                               (shape_gm_out, ),
                                               name="output_y_gm",
                                               scope=tik.scope_gm)

    def set_tiling_axis(self):
        """
        calculate the tiling axis
        """
        for i, _ in enumerate(self.tiling_shape):
            buf_size_needed = functools_reduce(lambda x1, x2: x1*x2,
                                               self.tiling_shape[i:])
            is_buffer_large_enough = self.ub_memory//buf_size_needed
            if is_buffer_large_enough > 0:
                if i == 0:
                    return 1
                else:
                    return i

        return len(self.tiling_shape)

    def set_core_num(self, tiling_index):
        """
        set the block_num
        """
        # if UB cannot put in block_size*output_depth, set block_num according
        # to the product of first 4 axis of tiling_shape and number of times
        # to divide the block_size*output_depth
        #max_core_num = tik.Dprofile().get_aicore_num()
        max_core_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        if tiling_index == 5:
            loop_memory = self.ub_memory - self.ub_memory % self.num_data
            loop_times = (self.num_data_one_move+loop_memory-1)//loop_memory
            loop_number = functools_reduce(lambda x1, x2: x1*x2,
                                           self.tiling_shape[:4])*loop_times
        # set block_num according to the product of first tiling_index axis
        # of tiling_shape
        elif tiling_index == 0:
            return 1
        else:
            loop_number = functools_reduce(lambda x1, x2: x1*x2,
                                           self.tiling_shape[:tiling_index])

        return loop_number if loop_number < max_core_num else max_core_num

    def data_move_gm2ub(self, tik_instance, num_b, num_h, num_block_h,
                        num_w, dst_ub_index):
        """
        move data from GM to UB and consider move output_depth*block_size data
        at a loop in the case of 32B not-aligned
        """
        src_x_index = self.min_size*num_block_h + (num_w + self.input_width *
                                                   (num_h + self.input_height *
                                                    num_b))*self.input_depth
        tik_instance.data_move(self.input_x_ub[dst_ub_index],
                               self.input_x_gm[src_x_index],
                               0, 1, self.num_data_one_move//self.num_data,
                               0, 0)

    def data_move_gm2ub_align(self, tik_instance, num_b, num_h, num_block_h,
                              dst_ub_index):
        """
        move data from GM to UB and consider move output_depth*block_size data
        at a loop in the case of 32B aligned
        """
        src_x_index = self.min_size*num_block_h + self.input_width * \
                      (num_h + self.input_height*num_b)*self.input_depth

        with tik_instance.for_range(0, self.input_width // MAX_N_BURST) \
                as num_burst:
            tik_instance.data_move(self.input_x_ub[dst_ub_index +
                                                   self.num_data_one_move *
                                                   MAX_N_BURST * num_burst],
                                   self.input_x_gm[src_x_index +
                                                   self.input_depth *
                                                   MAX_N_BURST * num_burst],
                                   0, MAX_N_BURST,
                                   self.num_data_one_move // self.num_data,
                                   (self.input_depth - self.min_size) //
                                   self.num_data, 0)
        if self.input_width % MAX_N_BURST != 0:
            tik_instance.data_move(self.input_x_ub[dst_ub_index +
                                                   self.input_width //
                                                   MAX_N_BURST *
                                                   self.num_data_one_move *
                                                   MAX_N_BURST],
                                   self.input_x_gm[src_x_index +
                                                   self.input_width //
                                                   MAX_N_BURST *
                                                   self.input_depth *
                                                   MAX_N_BURST],
                                   0, self.input_width % MAX_N_BURST,
                                   self.num_data_one_move // self.num_data,
                                   (self.input_depth - self.min_size) //
                                   self.num_data, 0)

    def data_move_ub2gm_case_division(self, tik_instance, core_loop,
                                      loop_number, num_core_loop,
                                      dst_y_first_index):
        """
        divide the case of move data from UB to GM
        """
        # 32B not-aligned and the data in UB is less than 32B
        if self.num_data > loop_number*self.min_size:
            padding_status = tik_instance.Scalar("uint64")
            num_padding_loop = tik_instance.Scalar("uint64")
            padding_loop = (self.num_data + loop_number*self.min_size - 1) // \
                           (loop_number*self.min_size)
            with tik_instance.if_scope(num_core_loop >=
                                       (core_loop - padding_loop)):
                num_padding_loop.set_as(num_core_loop - core_loop +
                                        padding_loop)
                padding_status.set_as(1)
                with tik_instance.if_scope(num_core_loop == (core_loop -
                                                             padding_loop)):
                    if self.num_data % (loop_number*self.min_size):
                        padding_status.set_as(2)
                with tik_instance.if_scope(num_core_loop == (core_loop - 1)):
                    padding_status.set_as(3)
            self.data_move_ub2gm_padding(tik_instance, dst_y_first_index,
                                         loop_number, padding_status,
                                         num_padding_loop)
        else:
            # 32B not-aligned and processing the tail block in the last loop
            is_last_loop = tik_instance.Scalar("uint64")
            is_last_loop.set_as(0)
            with tik_instance.if_scope(num_core_loop == (core_loop - 1)):
                is_last_loop.set_as(1)
            self.data_move_ub2gm(tik_instance, dst_y_first_index,
                                 loop_number, is_last_loop)

    def data_move_ub2gm(self, tik_instance, dst_y_first_index, handling_times,
                        is_last_loop=0):
        """
        move data from UB to GM, which can be divided into 32B aligned
        and non-aligned cases
        """
        with tik_instance.if_scope(is_last_loop == 1):
            with tik_instance.for_range(0, handling_times) as output_index:
                # in the case of 32B non-aligned, processing non-tail blocks
                with tik_instance.if_scope(output_index != handling_times - 1):
                    if self.num_data//self.min_size < 2:
                        tik_instance.data_move(self.output_y_gm
                                               [dst_y_first_index +
                                                output_index*self.min_size],
                                               self.input_x_ub
                                               [output_index *
                                                self.num_data_one_move],
                                               0, 1, self.num_data_one_move //
                                               self.num_data, 0, 0)
                    else:
                        with tik_instance.if_scope(output_index <
                                                   (handling_times -
                                                    self.num_data //
                                                    self.min_size)):
                            tik_instance.data_move(self.output_y_gm
                                                   [dst_y_first_index +
                                                    output_index *
                                                    self.min_size],
                                                   self.input_x_ub
                                                   [output_index *
                                                    self.num_data_one_move],
                                                   0, 1,
                                                   self.num_data_one_move //
                                                   self.num_data, 0, 0)
                # in the case of 32B non-aligned, processing tail block
                with tik_instance.else_scope():
                    tmp_ub = tik_instance.Tensor(self.dtype, (self.num_data,),
                                                 name="tmp_ub",
                                                 scope=tik.scope_ubuf)
                    tmp_scalar = tik_instance.Scalar(self.dtype)
                    # block_size*self.output_depth < 32B
                    if self.num_data_one_move == self.num_data:
                        total_num_pad = self.num_data//self.min_size
                        remainder_pad = self.num_data % self.min_size
                        with tik_instance.for_range(0, remainder_pad) \
                                as num_remainder_pad:
                            src_ub_index = self.num_data_one_move * \
                                           (handling_times - 1 -
                                            total_num_pad) + self.min_size - \
                                           remainder_pad
                            tmp_scalar.set_as(self.input_x_ub
                                              [src_ub_index +
                                               num_remainder_pad])
                            tmp_ub[num_remainder_pad] = tmp_scalar
                        with tik_instance.for_range(0, total_num_pad) \
                                as num_pad:
                            src_ub_index = self.num_data_one_move * \
                                           (handling_times - total_num_pad +
                                            num_pad)
                            with tik_instance.for_range(0, self.min_size) \
                                    as num_min_size:
                                tmp_scalar.set_as(self.input_x_ub
                                                  [src_ub_index +
                                                   num_min_size])
                                tmp_ub[remainder_pad + num_pad*self.min_size +
                                       num_min_size] = tmp_scalar
                        tik_instance.data_move(self.output_y_gm
                                               [dst_y_first_index +
                                                (handling_times-total_num_pad) *
                                                self.min_size - remainder_pad],
                                               tmp_ub[0], 0, 1, 1, 0, 0)
                    # block_size*self.output_depth > 32B
                    else:
                        tik_instance.data_move(self.output_y_gm
                                               [dst_y_first_index +
                                                output_index*self.min_size],
                                               self.input_x_ub
                                               [output_index *
                                                self.num_data_one_move],
                                               0, 1, self.num_data_one_move //
                                               self.num_data - 1, 0, 0)
                        with tik_instance.for_range(0, self.num_data) \
                                as num_data_index:
                            src_ub_index = self.num_data_one_move * \
                                           (handling_times - 1) + \
                                           self.min_size - 1 - num_data_index
                            tmp_scalar.set_as(self.input_x_ub[src_ub_index])
                            tmp_ub[self.num_data - 1 - num_data_index] = \
                                tmp_scalar
                        tik_instance.data_move(self.output_y_gm
                                               [dst_y_first_index +
                                                handling_times*self.min_size -
                                                self.num_data],
                                               tmp_ub[0], 0, 1, 1, 0, 0)
        # in the case of 32B aligned, moving data from UB to GM
        with tik_instance.else_scope():
            with tik_instance.for_range(0, handling_times) as output_index:
                tik_instance.data_move(self.output_y_gm[dst_y_first_index +
                                                        output_index *
                                                        self.min_size],
                                       self.input_x_ub[output_index *
                                                       self.num_data_one_move],
                                       0, 1, self.num_data_one_move //
                                       self.num_data, 0, 0)

    def data_move_ub2gm_padding(self, tik_instance, dst_y_first_index,
                                handling_times, padding_status,
                                num_padding_loop):
        """
        move data from UB to GM, in the case of 32B not-aligned and
        the data in UB is less than 32B
        """
        tmp_ub = tik_instance.Tensor(self.dtype, (self.num_data,),
                                     name="tmp_ub",
                                     scope=tik.scope_ubuf)
        tmp_scalar = tik_instance.Scalar(self.dtype)

        padding_remainder = self.num_data % (self.min_size*handling_times)
        if not padding_remainder % self.min_size:
            num_padding_loop += 1
        # moving data from UB to GM
        with tik_instance.if_scope(padding_status == 0):
            with tik_instance.for_range(0, handling_times) as output_index:
                tik_instance.data_move(self.output_y_gm
                                       [dst_y_first_index + output_index *
                                        self.min_size],
                                       self.input_x_ub
                                       [output_index*self.num_data_one_move],
                                       0, 1, self.num_data_one_move //
                                       self.num_data, 0, 0)
        # moving data from UB to tmp_ub
        with tik_instance.if_scope(padding_status == 1):
            with tik_instance.for_range(0, handling_times) as output_index:
                with tik_instance.for_range(0, self.min_size) as num_min_size:
                    tmp_scalar.set_as(self.input_x_ub[output_index *
                                                      self.num_data_one_move +
                                                      num_min_size])
                    tmp_ub[padding_remainder + self.min_size*handling_times *
                           (num_padding_loop - 1) + output_index *
                           self.min_size + num_min_size] = tmp_scalar
        # moving data from UB to tmp_ub and GM
        with tik_instance.if_scope(padding_status == 2):
            with tik_instance.for_range(0, ((self.min_size*handling_times -
                                             padding_remainder) +
                                            self.min_size - 1)//self.min_size) \
                    as output_index:
                tik_instance.data_move(self.output_y_gm[dst_y_first_index +
                                                        output_index *
                                                        self.min_size],
                                       self.input_x_ub[output_index *
                                                       self.num_data_one_move],
                                       0, 1, self.num_data_one_move //
                                       self.num_data, 0, 0)
            with tik_instance.for_range(0, padding_remainder % self.min_size) \
                    as num_padding_first:
                src_ub_index = (self.min_size*handling_times -
                                padding_remainder)//self.min_size * \
                               self.num_data_one_move + self.min_size - \
                               padding_remainder % self.min_size
                tmp_scalar.set_as(self.input_x_ub[src_ub_index +
                                                  num_padding_first])
                tmp_ub[num_padding_first] = tmp_scalar
            with tik_instance.for_range(0, padding_remainder//self.min_size) \
                    as num_padding_second:
                src_ub_index = (self.min_size*handling_times -
                                padding_remainder)//self.min_size * \
                               self.num_data_one_move + self.num_data_one_move
                if not padding_remainder % self.min_size:
                    src_ub_index = src_ub_index - self.min_size
                with tik_instance.for_range(0, self.min_size) as num_min_size:
                    tmp_scalar.set_as(self.input_x_ub
                                      [src_ub_index + num_padding_second *
                                       self.num_data_one_move + num_min_size])
                    tmp_ub[padding_remainder % self.min_size +
                           num_padding_second*self.min_size + num_min_size] = \
                        tmp_scalar
        # moving data from UB to tmp_ub and moving tmp_ub to GM
        with tik_instance.if_scope(padding_status == 3):
            with tik_instance.for_range(0, handling_times) as output_index:
                with tik_instance.for_range(0, self.min_size) as num_min_size:
                    tmp_scalar.set_as(self.input_x_ub[output_index *
                                                      self.num_data_one_move +
                                                      num_min_size])
                    tmp_ub[padding_remainder + self.min_size*handling_times *
                           (num_padding_loop-1) + output_index*self.min_size +
                           num_min_size] = tmp_scalar
                    tik_instance.data_move(self.output_y_gm[dst_y_first_index -
                                                            self.num_data +
                                                            self.min_size *
                                                            handling_times],
                                           tmp_ub[0], 0, 1, 1, 0, 0)

    def data_move_ub2gm_align(self, tik_instance, dst_y_first_index,
                              handling_data):
        """
        move data from UB to GM, in the case of 32B aligned
        """
        tik_instance.data_move(self.output_y_gm[dst_y_first_index],
                               self.input_x_ub[0],
                               0, 1, handling_data, 0, 0)

    def depth_to_space_axis_zero(self, tik_instance):
        """
        UB can put down (input_batch, input_height, block_size, input_width,
                         num_data_one_move)
        """
        dst_y_first_index = 0
        self.input_x_ub = tik_instance.Tensor(self.dtype,
                                              (self.input_batch,
                                               self.input_height,
                                               self.block_size,
                                               self.input_width,
                                               self.num_data_one_move),
                                              name="input_x_ub",
                                              scope=tik.scope_ubuf)
        if self.is_align:
            loop_number = self.input_batch*self.input_height*self.block_size
            with tik_instance.for_range(0, loop_number) as num_index:
                num_block_h = num_index % self.block_size
                div_block_size_h = (num_index - num_block_h)//self.block_size
                num_h = div_block_size_h % self.input_height
                num_b = (div_block_size_h - num_h)//self.input_height
                dst_ub_index = (((num_b*self.input_height + num_h) *
                                 self.block_size + num_block_h) *
                                self.input_width)*self.num_data_one_move
                self.data_move_gm2ub_align(tik_instance, num_b, num_h,
                                           num_block_h, dst_ub_index)

            handling_data = (self.input_batch*self.input_height *
                             self.block_size*self.input_width*self.min_size +
                             self.num_data - 1)//self.num_data
            self.data_move_ub2gm_align(tik_instance, dst_y_first_index,
                                       handling_data)
        else:
            loop_number = self.input_batch*self.input_height*self.block_size * \
                          self.input_width
            with tik_instance.for_range(0, loop_number) as num_index:
                num_w = num_index % self.input_width
                div_input_width = (num_index - num_w)//self.input_width
                num_block_h = div_input_width % self.block_size
                div_block_size_h = (div_input_width - num_block_h) // \
                                   self.block_size
                num_h = div_block_size_h % self.input_height
                num_b = (div_block_size_h - num_h)//self.input_height
                dst_ub_index = (((num_b*self.input_height + num_h) *
                                 self.block_size + num_block_h) *
                                self.input_width+num_w)*self.num_data_one_move
                self.data_move_gm2ub(tik_instance, num_b, num_h, num_block_h,
                                     num_w, dst_ub_index)

            self.data_move_ub2gm(tik_instance, dst_y_first_index, loop_number)

        return tik_instance

    def depth_to_space_axis_one(self, tik_instance):
        """
        UB can put down (input_height, block_size, input_width,
                         num_data_one_move)
        """
        core_number = self.set_core_num(self.tiling_axis)
        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            self.input_x_ub = tik_instance.Tensor(self.dtype,
                                                  (self.input_height,
                                                   self.block_size,
                                                   self.input_width,
                                                   self.num_data_one_move),
                                                  name="input_x_ub",
                                                  scope=tik.scope_ubuf)
            total_core_loop_num = self.input_batch
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            with tik_instance.for_range(0, core_loop) as num_core_loop:
                total_core_loop = sum_core + num_core_loop
                num_b = total_core_loop
                if self.is_align:
                    loop_number = self.input_height*self.block_size
                    total_num_one_loop = loop_number*self.input_width *\
                                         self.min_size
                    align_num = (core_loop*total_num_one_loop +
                                 self.ub_memory - 1)//self.ub_memory
                    align_loop = tik_instance.Scalar("uint64")
                    align_loop.set_as((core_loop + align_num - 1)//align_num)
                    with tik_instance.if_scope((align_loop-1)*core_loop *
                                               total_num_one_loop >
                                               self.ub_memory):
                        align_loop.set_as((core_loop + align_num - 1) //
                                          align_num - 1)
                    remainder = tik_instance.Scalar("uint64")
                    remainder.set_as(core_loop % align_loop)
                    with tik_instance.if_scope(remainder == 0):
                        remainder.set_as(align_loop)
                    with tik_instance.for_range(0, loop_number) as num_index:
                        num_block_h = num_index % self.block_size
                        num_h = (num_index - num_block_h)//self.block_size
                        dst_ub_index = ((num_h*self.block_size + num_block_h) *
                                        self.input_width) * \
                                       self.num_data_one_move + \
                                       (num_core_loop % align_loop) * \
                                       total_num_one_loop
                        self.data_move_gm2ub_align(tik_instance, num_b, num_h,
                                                   num_block_h, dst_ub_index)
                    # move data from ub to gm when ub is full
                    with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                                       align_loop == 0,
                                                       num_core_loop !=
                                                       core_loop - 1)):
                        dst_y_first_index = num_b*self.output_height * \
                                            self.output_width * \
                                            self.output_depth - \
                                            (align_loop - 1)*total_num_one_loop
                        handling_data = (align_loop*total_num_one_loop +
                                         self.num_data - 1)//self.num_data
                        self.data_move_ub2gm_align(tik_instance,
                                                   dst_y_first_index,
                                                   handling_data)
                    # move the remaining data
                    with tik_instance.if_scope(num_core_loop == core_loop - 1):
                        dst_y_first_index = num_b*self.output_height * \
                                            self.output_width * \
                                            self.output_depth - \
                                            (remainder - 1)*total_num_one_loop
                        handling_data = (remainder*total_num_one_loop +
                                         self.num_data - 1)//self.num_data
                        self.data_move_ub2gm_align(tik_instance,
                                                   dst_y_first_index,
                                                   handling_data)
                else:
                    loop_number = self.input_height*self.block_size * \
                                  self.input_width
                    with tik_instance.for_range(0, loop_number) as num_index:
                        num_w = num_index % self.input_width
                        div_input_width = (num_index - num_w)//self.input_width
                        num_block_h = div_input_width % self.block_size
                        num_h = (div_input_width - num_block_h) // \
                                self.block_size
                        dst_ub_index = ((num_h*self.block_size + num_block_h) *
                                        self.input_width + num_w) * \
                                       self.num_data_one_move
                        self.data_move_gm2ub(tik_instance, num_b, num_h,
                                             num_block_h, num_w, dst_ub_index)
                    dst_y_first_index = num_b*self.output_height * \
                                        self.output_width*self.output_depth
                    self.data_move_ub2gm_case_division(tik_instance, core_loop,
                                                       loop_number,
                                                       num_core_loop,
                                                       dst_y_first_index)

        return tik_instance

    def depth_to_space_axis_two(self, tik_instance):
        """
        UB can put down (block_size, input_width, num_data_one_move)
        """
        core_number = self.set_core_num(self.tiling_axis)
        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            self.input_x_ub = tik_instance.Tensor(self.dtype,
                                                  (self.block_size,
                                                   self.input_width,
                                                   self.num_data_one_move),
                                                  name="input_x_ub",
                                                  scope=tik.scope_ubuf)
            total_core_loop_num = self.input_batch*self.input_height
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            with tik_instance.for_range(0, core_loop) as num_core_loop:
                total_core_loop = sum_core + num_core_loop
                num_b = total_core_loop//self.input_height
                num_h = total_core_loop % self.input_height
                if self.is_align:
                    loop_number = self.block_size
                    total_num_one_loop = loop_number*self.input_width * \
                                         self.min_size
                    align_num = (core_loop*total_num_one_loop +
                                 self.ub_memory - 1)//self.ub_memory
                    align_loop = tik_instance.Scalar("uint64")
                    align_loop.set_as((core_loop + align_num - 1)//align_num)
                    with tik_instance.if_scope((align_loop-1)*core_loop *
                                               total_num_one_loop >
                                               self.ub_memory):
                        align_loop.set_as((core_loop + align_num - 1) //
                                          align_num - 1)
                    remainder = tik_instance.Scalar("uint64")
                    remainder.set_as(core_loop % align_loop)
                    with tik_instance.if_scope(remainder == 0):
                        remainder.set_as(align_loop)
                    with tik_instance.for_range(0, loop_number) as num_index:
                        num_block_h = num_index
                        dst_ub_index = num_block_h*self.input_width * \
                                       self.num_data_one_move + \
                                       (num_core_loop % align_loop) * \
                                       total_num_one_loop
                        self.data_move_gm2ub_align(tik_instance, num_b, num_h,
                                                   num_block_h, dst_ub_index)
                    # move data from ub to gm when ub is full
                    with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                                       align_loop == 0,
                                                       num_core_loop !=
                                                       core_loop - 1)):
                        dst_y_first_index = (num_b*self.input_height+num_h) * \
                                            self.block_size*self.output_width *\
                                            self.output_depth - \
                                            (align_loop - 1)*total_num_one_loop
                        handling_data = (align_loop*total_num_one_loop +
                                         self.num_data - 1)//self.num_data
                        self.data_move_ub2gm_align(tik_instance,
                                                   dst_y_first_index,
                                                   handling_data)
                    # move the remaining data
                    with tik_instance.if_scope(num_core_loop == core_loop - 1):
                        dst_y_first_index = (num_b*self.input_height+num_h) * \
                                            self.block_size*self.output_width *\
                                            self.output_depth - \
                                            (remainder - 1)*total_num_one_loop
                        handling_data = (remainder*total_num_one_loop +
                                         self.num_data - 1)//self.num_data
                        self.data_move_ub2gm_align(tik_instance,
                                                   dst_y_first_index,
                                                   handling_data)
                else:
                    loop_number = self.block_size*self.input_width
                    with tik_instance.for_range(0, loop_number) as num_index:
                        num_w = num_index % self.input_width
                        num_block_h = (num_index - num_w)//self.input_width
                        dst_ub_index = (num_block_h*self.input_width +
                                        num_w)*self.num_data_one_move
                        self.data_move_gm2ub(tik_instance, num_b, num_h,
                                             num_block_h, num_w, dst_ub_index)
                    dst_y_first_index = (num_b*self.input_height + num_h) * \
                                        self.block_size*self.output_width * \
                                        self.output_depth
                    self.data_move_ub2gm_case_division(tik_instance, core_loop,
                                                       loop_number,
                                                       num_core_loop,
                                                       dst_y_first_index)

        return tik_instance

    def depth_to_space_axis_three(self, tik_instance):
        """
        UB can put down (input_width, num_data_one_move)
        """
        core_number = self.set_core_num(self.tiling_axis)
        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            self.input_x_ub = tik_instance.Tensor(self.dtype,
                                                  (self.input_width,
                                                   self.num_data_one_move),
                                                  name="input_x_ub",
                                                  scope=tik.scope_ubuf)
            total_core_loop_num = self.input_batch*self.input_height * \
                                  self.block_size
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            with tik_instance.for_range(0, core_loop) as num_core_loop:
                total_core_loop = sum_core + num_core_loop
                num_block_h = total_core_loop % self.block_size
                num_h = (total_core_loop - num_block_h) // \
                        self.block_size % self.input_height
                num_b = (total_core_loop - num_block_h) // \
                        self.block_size//self.input_height
                if self.is_align:
                    total_num_one_loop = self.input_width*self.min_size
                    align_num = (core_loop*total_num_one_loop +
                                 self.ub_memory - 1)//self.ub_memory
                    align_loop = tik_instance.Scalar("uint64")
                    align_loop.set_as((core_loop + align_num - 1)//align_num)
                    with tik_instance.if_scope((align_loop-1)*core_loop *
                                               total_num_one_loop >
                                               self.ub_memory):
                        align_loop.set_as((core_loop + align_num - 1) //
                                          align_num - 1)
                    remainder = tik_instance.Scalar("uint64")
                    remainder.set_as(core_loop % align_loop)
                    with tik_instance.if_scope(remainder == 0):
                        remainder.set_as(align_loop)
                    dst_ub_index = (num_core_loop % align_loop) * \
                                   total_num_one_loop
                    self.data_move_gm2ub_align(tik_instance, num_b, num_h,
                                               num_block_h, dst_ub_index)
                    # move data from ub to gm when ub is full
                    with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                                       align_loop == 0,
                                                       num_core_loop !=
                                                       core_loop - 1)):
                        dst_y_first_index = ((num_b*self.input_height+num_h) *
                                             self.block_size+num_block_h) * \
                                            self.output_width * \
                                            self.output_depth - \
                                            (align_loop - 1)*total_num_one_loop
                        handling_data = (align_loop*total_num_one_loop +
                                         self.num_data - 1)//self.num_data
                        self.data_move_ub2gm_align(tik_instance,
                                                   dst_y_first_index,
                                                   handling_data)
                    # move the remaining data
                    with tik_instance.if_scope(num_core_loop == core_loop - 1):
                        dst_y_first_index = ((num_b*self.input_height+num_h) *
                                             self.block_size + num_block_h) * \
                                            self.output_width * \
                                            self.output_depth - \
                                            (remainder - 1)*total_num_one_loop
                        handling_data = (remainder*total_num_one_loop +
                                         self.num_data - 1)//self.num_data
                        self.data_move_ub2gm_align(tik_instance,
                                                   dst_y_first_index,
                                                   handling_data)
                else:
                    loop_number = self.input_width
                    with tik_instance.for_range(0, loop_number) as num_index:
                        num_w = num_index
                        dst_ub_index = num_w*self.num_data_one_move
                        self.data_move_gm2ub(tik_instance, num_b, num_h,
                                             num_block_h, num_w, dst_ub_index)
                    dst_y_first_index = ((num_b*self.input_height + num_h) *
                                         self.block_size + num_block_h) * \
                                        self.output_width*self.output_depth
                    self.data_move_ub2gm_case_division(tik_instance, core_loop,
                                                       loop_number,
                                                       num_core_loop,
                                                       dst_y_first_index)

        return tik_instance

    def depth_to_space_axis_four(self, tik_instance):
        """
        UB can put down (num_data_one_move)
        """
        core_number = self.set_core_num(self.tiling_axis)
        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            self.input_x_ub = tik_instance.Tensor(self.dtype,
                                                  (self.num_data_one_move,),
                                                  name="input_x_ub",
                                                  scope=tik.scope_ubuf)
            total_core_loop_num = self.input_batch*self.input_height * \
                                  self.block_size*self.input_width
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            with tik_instance.for_range(0, core_loop) as num_core_loop:
                total_core_loop = sum_core + num_core_loop
                num_w = total_core_loop % self.input_width
                div_input_width = (total_core_loop - num_w)//self.input_width
                num_block_h = div_input_width % self.block_size
                div_block_size_h = (div_input_width - num_block_h) // \
                                   self.block_size
                num_h = div_block_size_h % self.input_height
                num_b = (div_block_size_h - num_h)//self.input_height

                if self.is_align:
                    align_num = (core_loop*self.min_size + self.ub_memory -
                                 1)//self.ub_memory
                    align_loop = tik_instance.Scalar("uint64")
                    align_loop.set_as((core_loop + align_num - 1)//align_num)
                    with tik_instance.if_scope((align_loop-1)*core_loop *
                                               self.min_size >
                                               self.ub_memory):
                        align_loop.set_as((core_loop + align_num - 1) //
                                          align_num - 1)
                    remainder = tik_instance.Scalar("uint64")
                    remainder.set_as(core_loop % align_loop)
                    with tik_instance.if_scope(remainder == 0):
                        remainder.set_as(align_loop)
                    dst_ub_index = (num_core_loop % align_loop)*self.min_size
                    self.data_move_gm2ub(tik_instance, num_b, num_h,
                                         num_block_h, num_w, dst_ub_index)
                    # move data from ub to gm when ub is full
                    with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                                       align_loop == 0,
                                                       num_core_loop !=
                                                       core_loop - 1)):
                        dst_y_first_index = (((num_b*self.input_height +
                                               num_h)*self.block_size +
                                              num_block_h)*self.input_width +
                                             num_w)*self.min_size - \
                                            (align_loop - 1)*self.min_size
                        handling_data = (align_loop*self.min_size +
                                         self.num_data - 1)//self.num_data
                        self.data_move_ub2gm_align(tik_instance,
                                                   dst_y_first_index,
                                                   handling_data)
                    # move the remaining data
                    with tik_instance.if_scope(num_core_loop == core_loop - 1):
                        dst_y_first_index = (((num_b*self.input_height +
                                               num_h)*self.block_size +
                                              num_block_h)*self.input_width +
                                             num_w)*self.min_size - \
                                            (remainder - 1)*self.min_size
                        handling_data = (remainder*self.min_size +
                                         self.num_data - 1)//self.num_data
                        self.data_move_ub2gm_align(tik_instance,
                                                   dst_y_first_index,
                                                   handling_data)
                else:
                    dst_ub_index = 0
                    self.data_move_gm2ub(tik_instance, num_b, num_h,
                                         num_block_h, num_w, dst_ub_index)
                    dst_y_first_index = (((num_b*self.input_height + num_h) *
                                          self.block_size + num_block_h) *
                                         self.input_width + num_w) * \
                                        self.block_size*self.output_depth
                    loop_number = 1
                    self.data_move_ub2gm_case_division(tik_instance, core_loop,
                                                       loop_number,
                                                       num_core_loop,
                                                       dst_y_first_index)

        return tik_instance

    def depth_to_space_axis_five(self, tik_instance):
        """
        UB can not put down (num_data_one_move,)
        and need to move multiple times
        """
        loop_memory = self.ub_memory - self.ub_memory % self.num_data
        loop_times = (self.num_data_one_move + loop_memory - 1)//loop_memory
        core_number = self.set_core_num(self.tiling_axis)
        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            self.input_x_ub = tik_instance.Tensor(self.dtype, (loop_memory,),
                                                  name="input_x_ub",
                                                  scope=tik.scope_ubuf)
            total_core_loop_num = self.input_batch*self.input_height * \
                                  self.block_size*self.input_width*loop_times
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            with tik_instance.for_range(0, core_loop) as num_core_loop:
                total_core_loop = sum_core + num_core_loop
                num_loop_time = total_core_loop % loop_times
                div_loop_times = (total_core_loop - num_loop_time)//loop_times
                num_w = div_loop_times % self.input_width
                div_input_width = (div_loop_times - num_w)//self.input_width
                num_block_h = div_input_width % self.block_size
                div_block_size_h = (div_input_width - num_block_h) // \
                                   self.block_size
                num_h = div_block_size_h % self.input_height
                num_b = (div_block_size_h - num_h)//self.input_height
                src_x_index = self.output_depth*self.block_size*num_block_h + \
                              (num_w + self.input_width *
                               (num_h + self.input_height*num_b)) * \
                              self.input_depth + num_loop_time*loop_memory
                dst_ub_index = 0
                with tik_instance.if_scope(num_loop_time == loop_times - 1):
                    if self.min_size % loop_memory == 0:
                        remainder = loop_memory
                    else:
                        remainder = self.min_size % loop_memory
                    handling_times = (remainder + self.num_data - 1) // \
                                     self.num_data
                # handling times of tail blocks in the case of 32B aligned
                # and non-tail blocks in the case of 32B non-aligned
                handling_times_other = tik_instance.Scalar("uint64")
                handling_times_other.set_as(loop_memory//self.num_data)
                with tik_instance.if_scope(num_loop_time == loop_times - 1):
                    handling_times_other.set_as((remainder + self.num_data -
                                                 1)//self.num_data)
                tik_instance.data_move(self.input_x_ub[dst_ub_index],
                                       self.input_x_gm[src_x_index],
                                       0, 1, handling_times_other, 0, 0)
                dst_y_first_index = (((num_b * self.input_height + num_h) *
                                      self.block_size + num_block_h) *
                                     self.input_width + num_w) * \
                                    self.block_size*self.output_depth + \
                                    loop_memory*num_loop_time
                with tik_instance.if_scope(tik.all(num_loop_time ==
                                                   loop_times - 1,
                                                   num_core_loop ==
                                                   (core_loop - 1))):
                    if not self.is_align:
                        tmp_ub = tik_instance.Tensor(self.dtype,
                                                     (self.num_data,),
                                                     name="tmp_ub",
                                                     scope=tik.scope_ubuf)
                        tmp_scalar = tik_instance.Scalar(self.dtype)
                        # the size of tail block < 32B
                        if handling_times == 1:
                            tmp_ub_before = \
                                tik_instance.Tensor(self.dtype,
                                                    (self.num_data,),
                                                    name="tmp_ub_before",
                                                    scope=tik.scope_ubuf)
                            src_x_before_index = src_x_index - self.num_data
                            tik_instance.data_move(tmp_ub_before[0],
                                                   self.input_x_gm
                                                   [src_x_before_index],
                                                   0, 1, 1, 0, 0)
                            with tik_instance.for_range(0, remainder) \
                                    as num_remainder:
                                tmp_scalar.set_as(self.input_x_ub
                                                  [num_remainder])
                                tmp_ub[self.num_data - remainder +
                                       num_remainder] = tmp_scalar
                            with tik_instance.for_range(0,
                                                        self.num_data -
                                                        remainder) \
                                    as num_data_before:
                                tmp_scalar.set_as(tmp_ub_before
                                                  [remainder +
                                                   num_data_before])
                                tmp_ub[num_data_before] = tmp_scalar
                            tik_instance.data_move(self.output_y_gm
                                                   [dst_y_first_index -
                                                    self.num_data +
                                                    remainder], tmp_ub[0],
                                                   0, 1, 1, 0, 0)
                        # # the size of tail block > 32B
                        else:
                            tik_instance.data_move(self.output_y_gm
                                                   [dst_y_first_index],
                                                   self.input_x_ub[0],
                                                   0, 1, handling_times - 1,
                                                   0, 0)
                            with tik_instance.for_range(0, self.num_data) \
                                    as num_data_index:
                                src_ub_index = remainder - self.num_data + \
                                               num_data_index
                                tmp_scalar.set_as(self.input_x_ub
                                                  [src_ub_index])
                                tmp_ub[num_data_index] = tmp_scalar
                            tik_instance.data_move(self.output_y_gm
                                                   [dst_y_first_index +
                                                    remainder - self.num_data],
                                                   tmp_ub[0], 0, 1, 1, 0, 0)
                    else:
                        tik_instance.data_move(self.output_y_gm
                                               [dst_y_first_index],
                                               self.input_x_ub[0], 0, 1,
                                               handling_times_other, 0, 0)
                with tik_instance.else_scope():
                    tik_instance.data_move(self.output_y_gm
                                           [dst_y_first_index],
                                           self.input_x_ub[0], 0, 1,
                                           handling_times_other, 0, 0)

        return tik_instance

    def depth_to_space_compute(self):
        """
        the overall data move process
        """
        self.tiling_axis = self.set_tiling_axis()
        tik_instance = self.set_tik_instance()

        if self.tiling_axis == 0:
            tik_instance = self.depth_to_space_axis_zero(tik_instance)
        elif self.tiling_axis == 1:
            tik_instance = self.depth_to_space_axis_one(tik_instance)
        elif self.tiling_axis == 2:
            tik_instance = self.depth_to_space_axis_two(tik_instance)
        elif self.tiling_axis == 3:
            tik_instance = self.depth_to_space_axis_three(tik_instance)
        elif self.tiling_axis == 4:
            tik_instance = self.depth_to_space_axis_four(tik_instance)
        elif self.tiling_axis == 5:
            tik_instance = self.depth_to_space_axis_five(tik_instance)

        return tik_instance

    def run_depth_to_space_computer(self):
        """
        run depth to space computer
        """
        tik_instance = self.depth_to_space_compute()

        return tik_instance, self.input_x_gm, self.output_y_gm
