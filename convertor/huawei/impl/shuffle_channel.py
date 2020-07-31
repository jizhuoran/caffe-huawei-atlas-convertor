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

shuffle_channel
"""
from te import tik
from topi.cce import util
from impl import constant_util as constant
from impl import common_util
from te import platform as tbe_platform

# reserve size for ub
RESERVE_SIZE = 16 * 1024


# pylint: disable=invalid-name,too-many-locals
@util.check_input_type(dict, dict, int, str)
def shuffle_channel(x, y, group=1, kernel_name="shuffle_channel"):
    """
    the main function of shuffle_channel

    Parameters
    ----------
    x: dict,shape and datatype,datatype supports int8,uint8,int16,uint16,int32,
         uint32,int64,uint64,float16,float32
    y: dict,shape and datatype,datatype supports int8,uint8,int16,uint16,int32,
         uint32,int64,uint64,float16,float32
    group: 1 channel group
    kernel_name: cce kernel name, default value is "shuffle_channel"

    Returns
    -------
    tik_instance: tik_instance
    """
    input_dict = {
        "x": x,
        "y": y,
        "group": group,
        "kernel_name": kernel_name
    }
    check_param(input_dict)
    shuffle_process = ShuffleChannel(input_dict)
    shuffle_process.compute_shuffle_channel()
    shuffle_process.instance.BuildCCE(kernel_name=kernel_name,
                                      inputs=(shuffle_process.x_gm),
                                      outputs=(shuffle_process.y_gm),
                                      enable_l2=False)

    return shuffle_process.instance


class ShuffleChannel:
    """
    Function: store ShuffleChannel parameters  and compute ShuffleChannel
    Modify : 202--02-12
    """

    def __init__(self, input_dict):
        """
      init the ShuffleChannel parameters

      Parameters
      ----------
        input_dict: input_dict is a dict, the keys as follow:
            x: dict,shape and datatype,datatype supports int8,uint8,int16,
              uint16,int32,uint32,int64,uint64,float16,float32
            y: dict,shape and datatype,datatype supports int8,uint8,int16,
              uint16,int32,uint32,int64,uint64,float16,float32
            group: 1 channel group
            kernel_name: cce kernel name, default value is "shuffle_channel"
      Returns
      -------
      None
      """
        self.instance = tik.Tik(tik.Dprofile())
        self.dtype = input_dict.get("x").get("dtype").lower()
        self.dsize = common_util.get_data_size(self.dtype)
        total_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
        ub_size = (total_size - RESERVE_SIZE) // (2 * self.dsize)
        burnest_len = constant.BLOCK_SIZE // self.dsize
        ub_size = ((ub_size + burnest_len - 1) // burnest_len) * burnest_len
        self.one_max_size = ub_size
        x_len = get_shape_total_number(input_dict.get("x").get("shape"))
        x_len = ((x_len + burnest_len - 1) // burnest_len) * burnest_len
        hw = input_dict.get("y").get("shape")[2] * \
             input_dict.get("y").get("shape")[3]
        mod = hw % burnest_len
        if mod != 0:
            x_len = x_len + burnest_len
        self.x_gm = self.instance.Tensor(self.dtype, (x_len,), name="x_gm",
                                         scope=tik.scope_gm)
        self.y_gm = self.instance.Tensor(self.dtype, (x_len,), name="y_gm",
                                         scope=tik.scope_gm)
        self.input_dict = input_dict

    def get_blockdim_and_loop_cycle(self):
        """
      get block dim and loop cycle

      Parameters
      ----------
        None
      Returns
      -------
      None
      """
        #block_num = tik.Dprofile().get_aicore_num()
        block_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        
        shape_y = self.input_dict.get("y").get("shape")
        limit_size_of_each_block = shape_y[2] * shape_y[3]
        total_channel = shape_y[0] * shape_y[1]
        each_block_num = constant.BLOCK_SIZE // self.dsize
        each_block_align = \
            ((each_block_num + limit_size_of_each_block - 1) //
             limit_size_of_each_block) * limit_size_of_each_block
        if limit_size_of_each_block * self.dsize < constant.BLOCK_SIZE:
            all_size = total_channel * limit_size_of_each_block * self.dsize
            if all_size < constant.BLOCK_SIZE:
                block_num = 1
                return block_num, total_channel, 0

            limit_size_of_each_block = each_block_align
        limit_channel_of_each_block = limit_size_of_each_block // \
                                      (shape_y[2] * shape_y[3])
        loop = (total_channel * shape_y[2] * shape_y[3]) // \
               limit_size_of_each_block
        mod_channel = ((total_channel * shape_y[2] * shape_y[3]) % \
                       limit_size_of_each_block) // (shape_y[2] * shape_y[3])
        if loop <= block_num:
            block_num = loop
            inner_loop = limit_channel_of_each_block
            inner_loop_mod = mod_channel
        else:
            inner_loop = (loop // block_num) * limit_channel_of_each_block
            inner_loop_mod = (loop % block_num) * limit_channel_of_each_block \
                             + mod_channel
            if inner_loop_mod > block_num:
                inner_loop = inner_loop + inner_loop_mod // block_num
                inner_loop_mod = inner_loop_mod % block_num

        return block_num, inner_loop, inner_loop_mod

    def compute_shuffle_channel(self):
        """
        compute shuffle_channel

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        block_num, inner_loop, tail = self.get_blockdim_and_loop_cycle()
        shape_out = self.input_dict.get("y").get("shape")
        hw = shape_out[2] * shape_out[3]
        if hw * self.dsize < constant.BLOCK_SIZE:
            if block_num == 1 and inner_loop > 1:
                thread_num = 2
            else:
                thread_num = 1
        else:
            thread_num = 1
            if inner_loop > 1:
                thread_num = 2
        with self.instance.for_range(0, block_num, block_num=block_num) \
                as block_id:
            ub_tmp = self.instance.Tensor(self.dtype, (256,),
                                          name="ub_tmp", scope=tik.scope_ubuf)
            loop = self.instance.Scalar("int32")
            tmp_offset = self.instance.Scalar("int32")
            tmp_offset.set_as(0)
            with self.instance.for_range(0,
                                         inner_loop,
                                         thread_num=thread_num) as inner_cycle:
                x_ub = self.instance.Tensor(self.dtype, (self.one_max_size,),
                                            name="x_ub", scope=tik.scope_ubuf)
                loop.set_as(block_id * inner_loop + inner_cycle)
                with self.instance.if_scope(tail > 0):
                    with self.instance.if_scope(block_id < tail):
                        loop.set_as(block_id * inner_loop + \
                                    inner_cycle + block_id)
                    with self.instance.else_scope():
                        loop.set_as(block_id * inner_loop + inner_cycle + tail)

                src_start, dest_start = self.get_start_address(loop)
                if hw * self.dsize < constant.BLOCK_SIZE and block_num > 1:
                    input_dict = {
                        "x_ub": x_ub,
                        "ub_tmp": ub_tmp,
                        "src_start": src_start,
                        "dest_start": dest_start,
                        "element_num": hw,
                        "each_loop": inner_cycle,
                        "total_loop": inner_loop,
                        "tmp_offset": tmp_offset,
                    }

                    self.move_out_less_than32b(input_dict)

                else:
                    input_dict = {
                        "x_ub": x_ub,
                        "src_start": src_start,
                        "dest_start": dest_start,
                        "element_num": hw,
                        "block_num": block_num,
                    }
                    self.data_move(input_dict)
            if tail > 0:
                with self.instance.if_scope(block_id < tail):
                    x_ub = self.instance.Tensor(self.dtype,
                                                (self.one_max_size,),
                                                name="x_ub",
                                                scope=tik.scope_ubuf)
                    loop.set_as(loop + 1)
                    src_start, dest_start = self.get_start_address(loop)

                    with self.instance.if_scope((hw * self.dsize) >= \
                                                constant.BLOCK_SIZE):
                        input_dict = {
                            "x_ub": x_ub,
                            "src_start": src_start,
                            "dest_start": dest_start,
                            "element_num": hw,
                            "block_num": block_num,
                        }
                        self.data_move(input_dict)

                    with self.instance.else_scope():
                        self.instance.data_move(x_ub,
                                                self.x_gm[src_start],
                                                constant.SID,
                                                constant.DEFAULT_NBURST, 1,
                                                constant.STRIDE_ZERO,
                                                constant.STRIDE_ZERO)
                        input_dict = {
                            "instance": self.instance,
                            "out_ub": x_ub,
                            "out_gm": self.y_gm,
                            "gm_offset": dest_start,
                            "element_num": hw,
                            "dsize": self.dsize,
                        }
                        common_util.move_out_non32_alignment(input_dict)

    def get_start_address(self, loop):
        """
      get the start address of the source and dest tensor

      Parameters
      ----------
        loop: loop times
      Returns
      -------
      None
      """
        shape_out = self.input_dict.get("y").get("shape")
        channel = shape_out[1]
        group = self.input_dict.get("group")
        src_start = self.instance.Scalar("int32")
        group_row = (loop % channel) // group
        group_col = (loop % channel) % group
        index = (loop // channel) * channel + \
                group_col * (channel // group) + group_row
        hw = shape_out[2] * shape_out[3]
        src_start.set_as(index * hw)
        dest_start = self.instance.Scalar("int32")
        dest_start.set_as(loop * hw)
        return src_start, dest_start

    def move_out_less_than32b(self, input_dict):
        """
      move data from ub to gm

      Parameters
      ----------
        input_dict: input_dict is a dict, the keys as follow:
                x_ub: x_ub is a tensor,store data from gm
                ub_tmp: ub_tmp is a tensor,store last loop 32b data from gm
                src_start: src address
                dest_start: dest address
                element_num: each continuous segment
                each_loop: loop times
                total_loop: total loop of each block
                tmp_offset: the offset of ub_tmp
      Returns
      -------
      None
      """
        x_ub = input_dict.get("x_ub")
        ub_tmp = input_dict.get("ub_tmp")
        src_start = input_dict.get("src_start")
        dest_start = input_dict.get("dest_start")
        each_loop = input_dict.get("each_loop")
        element_num = input_dict.get("element_num")
        total_loop = input_dict.get("total_loop")
        tmp_offset = input_dict.get("tmp_offset")
        loop_32b = (constant.BLOCK_SIZE // self.dsize) // element_num
        if (constant.BLOCK_SIZE // self.dsize) % element_num != 0:
            loop_32b = loop_32b + 1

        nburst = common_util.get_datamove_nburst(self.instance,
                                                 element_num * self.dsize)
        self.instance.data_move(x_ub, self.x_gm[src_start], constant.SID,
                                constant.DEFAULT_NBURST, nburst,
                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        scalar = self.instance.Scalar(x_ub.dtype)

        with self.instance.if_scope(each_loop >= total_loop - loop_32b):
            with self.instance.for_range(0, element_num) as time:
                scalar.set_as(x_ub[time])
                ub_tmp[tmp_offset + time].set_as(scalar)
            tmp_offset.set_as(tmp_offset + element_num)
            with self.instance.if_scope(each_loop == total_loop - 1):
                dest_start.set_as(dest_start - (loop_32b - 1) * element_num)
                input_dict = {
                    "instance": self.instance,
                    "out_ub": ub_tmp,
                    "out_gm": self.y_gm,
                    "gm_offset": dest_start,
                    "element_num": element_num * loop_32b,
                    "dsize": self.dsize,
                }
                common_util.move_out_non32_alignment(input_dict)

        with self.instance.else_scope():
            nburst = common_util.get_datamove_nburst(self.instance,
                                                     element_num * self.dsize)
            self.instance.data_move(self.y_gm[dest_start],
                                    x_ub,
                                    constant.SID,
                                    constant.DEFAULT_NBURST, nburst,
                                    constant.STRIDE_ZERO,
                                    constant.STRIDE_ZERO)

    def data_move(self, input_dict):
        """
      move data from ub to gm

      Parameters
      ----------
        input_dict: input_dict is a dict, the keys as follow:
                x_ub: x_ub is a tensor,store data from gm
                src_start: the start address of src tensor
                dest_start: the start address of dest tensor
                element_num: each continuous segment
                block_num: blcok number
      Returns
      -------
      None
      """
        x_ub = input_dict.get("x_ub")
        element_num = input_dict.get("element_num")
        block_num = input_dict.get("block_num")
        loop_num, last_ub_num = get_loop_param(element_num,
                                               self.one_max_size)
        cur_size = self.instance.Scalar("int32")
        cur_size.set_as(self.one_max_size * self.dsize)
        ub_num = self.instance.Scalar("int32")
        ub_num.set_as(self.one_max_size)
        offset_in = self.instance.Scalar("int32")
        offset_in.set_as(input_dict.get("src_start"))
        offset_out = self.instance.Scalar("int32")
        offset_out.set_as(input_dict.get("dest_start"))
        each_burst_num = constant.BLOCK_SIZE // self.dsize
        with self.instance.for_range(0, loop_num) as cycle:
            with self.instance.if_scope(cycle == loop_num - 1):
                cur_size.set_as(last_ub_num * self.dsize)
                ub_num.set_as(last_ub_num)
            n_burst = common_util.get_datamove_nburst(self.instance,
                                                      cur_size)
            mod = cur_size % constant.BLOCK_SIZE
            with self.instance.if_scope(
                    tik.all(cycle == loop_num - 1, mod != 0, block_num > 1)):
                x_ub_tail = self.instance.Tensor(self.dtype, (32,),
                                                 name="x_ub_tail",
                                                 scope=tik.scope_ubuf)
                self.instance.data_move(x_ub_tail,
                                        self.x_gm[offset_in +
                                                  ub_num - each_burst_num],
                                        constant.SID,
                                        constant.DEFAULT_NBURST, 1,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
                self.instance.data_move(self.y_gm[offset_out +
                                                  ub_num - each_burst_num],
                                        x_ub_tail,
                                        constant.SID,
                                        constant.DEFAULT_NBURST, 1,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
                with self.instance.if_scope(cur_size > constant.BLOCK_SIZE):
                    self.instance.data_move(x_ub,
                                            self.x_gm[offset_in],
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            n_burst - 1,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
                    self.instance.data_move(self.y_gm[offset_out],
                                            x_ub,
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            n_burst - 1,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
            with self.instance.else_scope():
                self.instance.data_move(x_ub,
                                        self.x_gm[offset_in],
                                        constant.SID, constant.DEFAULT_NBURST,
                                        n_burst, constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
                self.instance.data_move(self.y_gm[offset_out],
                                        x_ub,
                                        constant.SID,
                                        constant.DEFAULT_NBURST, n_burst,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
            offset_in.set_as(offset_in + ub_num)
            offset_out.set_as(offset_out + ub_num)


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
    if last_ub_num != 0:
        loop_cycle = loop_cycle + 1
    else:
        last_ub_num = max_ub_num

    return loop_cycle, last_ub_num


def check_param(input_dict):
    """
    check the parameters is valid

    Parameters
    ----------
    input_dict: input_dict is a dict, the keys as follow:
                x: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                y: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                group: channel group default 1
                kernel_name: cce kernel name, default value is "shuffle_channel"
    Returns
    -------
    None
    """
    util.check_kernel_name(input_dict.get('kernel_name'))
    x_dtype = input_dict.get("x").get("dtype").lower()
    x_shape = input_dict.get("x").get("shape")
    y_dtype = input_dict.get("y").get("dtype").lower()
    y_shape = input_dict.get("y").get("shape")

    util.check_shape_rule(x_shape)
    util.check_tensor_shape_size(x_shape)
    util.check_dtype_rule(x_dtype,
                          ("int8", "uint8", "int16", "uint16", "int32",
                           "uint32", "int64", "uint64", "float16",
                           "float32"))

    util.check_shape_rule(y_shape)
    util.check_tensor_shape_size(y_shape)
    util.check_dtype_rule(y_dtype,
                          ("int8", "uint8", "int16", "uint16", "int32",
                           "uint32", "int64", "uint64", "float16",
                           "float32"))

    if x_dtype != y_dtype:
        raise RuntimeError("x's data type must be the same as y's data type")

    if len(x_shape) > 4 or len(x_shape) < 2:
        raise RuntimeError(
            "x's dim must between 2 to 4")
    if len(x_shape) == 3:
        x_shape = list((x_shape[0], x_shape[1], x_shape[2], 1))
    if len(x_shape) == 2:
        x_shape = list((x_shape[0], x_shape[1], 1, 1))
    input_dict["x"]["shape"] = x_shape

    if len(y_shape) > 4 or len(y_shape) < 2:
        raise RuntimeError(
            "y's dim must between 2 to 4")
    if len(y_shape) == 3:
        y_shape = list((y_shape[0], y_shape[1], y_shape[2], 1))
    if len(y_shape) == 2:
        y_shape = list((y_shape[0], y_shape[1], 1, 1))
    input_dict["y"]["shape"] = y_shape

    if not check_same_dim(y_shape, x_shape):
        raise RuntimeError(
            "y's shape must be the same as x's shape")

    group = input_dict.get("group")
    if group <= 0:
        raise RuntimeError("group must be greater than 0")

    channel = x_shape[1]
    if channel % group != 0:
        raise RuntimeError(
            "channel must be divisible by group")


def get_shape_total_number(shape):
    """
    get the number of element from the shape

    Parameters
    ----------
    shape: out put shape

    Returns
    -------
    total_number: the number of element of the shape
    """
    total_number = 1
    for i in shape:
        total_number = total_number * i

    return total_number


def check_same_dim(shape_x, shape_y):
    """
    check shape_x is the same shape as shape_y

    Parameters
    ----------
    shape_x: a tuple or list
    shape_y: a tuple or list

    Returns
    -------
    boolean: True has the same shape, False does't has the same shape
    """
    shape_x_len = len(shape_x)
    for k in range(shape_x_len):
        if shape_x[k] != shape_y[k]:
            return False

    return True
