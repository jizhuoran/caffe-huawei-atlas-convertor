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

crop
"""
from te import tik
from te import platform as tbe_platform
from topi.cce import util
from impl import constant_util as constant
from impl import common_util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

# reserve size for ub
RESERVE_SIZE = 16 * 1024


# pylint: disable=invalid-name,too-many-arguments,unused-argument
@util.check_input_type(dict, dict, dict, int, (tuple, list), str)
def crop(x, size, y, axis=2, offsets=(0), kernel_name="crop"):
    """
    the main function of crop

    Parameters
    ----------
    x1: dict,shape and datatype,datatype supports int8,uint8,int16,uint16,
        int32,uint32,int64,uint64,float16,float32
    x2: dict,shape and datatype,datatype supportsint8,uint8,int16,uint16,
        int32,uint32,int64,uint64,float16,float32
    y: dict,shape and datatype,datatype supports int8,uint8,int16,uint16,
        int32,uint32,int64,uint64,float16,float32
    axis: crop start with axis
    offsets: crop start offset of each axis
    kernel_name: cce kernel name, default value is "crop"

    Returns
    -------
    tik_instance: tik_instance
    """
    input_dict = {
        "x1": x,
        "x2": y,
        "y": y,
        "axis": axis,
        "offset": offsets,
        "kernel_name": kernel_name
    }
    check_and_adjust_offset(input_dict)
    crop_process = Crop(input_dict)
    crop_process.compute_crop()
    crop_process.instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(
                                       crop_process.x1_gm, crop_process.x2_gm),
                                   outputs=(crop_process.y_gm), enable_l2=False)

    return crop_process.instance


# pylint: disable=too-many-instance-attributes,too-many-locals
class Crop:
    """
    Function: store Crop parameters  and compute crop
    Modify : 2019-11-06
    """

    def __init__(self, input_dict):
        """
      init the Crop parameters

      Parameters
      ----------
        input_dict: input_dict is a dict, the keys as follow:
                x1: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                x2: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                y: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                axis: crop start with axis
                offsets: crop start offset of each axis
                kernel_name: cce kernel name, default value is "crop"
      Returns
      -------
      None
      """
        self.instance = tik.Tik(tik.Dprofile())
        self.dtype = input_dict.get("x1").get("dtype").lower()
        self.dsize = common_util.get_data_size(self.dtype)
        total_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
        ub_size = (total_size - RESERVE_SIZE) // (2 * self.dsize)
        burnest_len = constant.BLOCK_SIZE // self.dsize
        ub_size = ((ub_size + burnest_len - 1) // burnest_len) * burnest_len
        self.one_max_size = ub_size
        x1_len = get_shape_total_number(input_dict.get("x1").get("shape"))
        x1_len = ((x1_len + burnest_len - 1) // burnest_len) * burnest_len
        mod = input_dict.get("y").get("shape")[-1] % burnest_len
        if mod != 0:
            x1_len = x1_len + burnest_len
        self.x1_gm = self.instance.Tensor(self.dtype, (x1_len,), name="x1_gm",
                                          scope=tik.scope_gm)
        self.x2_gm = self.instance.Tensor(self.dtype, (32,), name="x2_gm",
                                          scope=tik.scope_gm)
        y_len = get_shape_total_number(input_dict.get("y").get("shape"))
        y_len = ((y_len + burnest_len - 1) // burnest_len) * burnest_len
        if mod != 0:
            y_len = y_len + burnest_len
        self.y_gm = self.instance.Tensor(self.dtype, (y_len,), name="y_gm",
                                         scope=tik.scope_gm)
        self.input_dict = input_dict

    def get_element_num(self):
        """
        get the block size
        """
        shape_y = self.input_dict.get("y").get("shape")
        shape_len = len(shape_y)
        element_num = shape_y[-1]
        if "format" in self.input_dict.get("x1"):
            x1_format = self.input_dict.get("x1").get("format")
            if x1_format == "NC1HWC0":
                element_num = shape_y[-1] * shape_y[-2]
                shape_len = len(shape_y) - 1
        return element_num, shape_len

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
        limit_size_of_each_block = self.get_limit_size_of_each_block()
        block_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        shape_y = self.input_dict.get("y").get("shape")
        loop = get_shape_total_number(shape_y) // limit_size_of_each_block
        element_num, _ = self.get_element_num()
        all_num = get_shape_total_number(shape_y) // element_num
        if loop <= block_num:
            block_num = loop
        inner_loop = all_num // block_num
        inner_loop_mod = all_num % block_num
        return block_num, limit_size_of_each_block, inner_loop, inner_loop_mod

    def get_limit_size_of_each_block(self):
        """
        get limit size of each block

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        shape_y = self.input_dict.get("y").get("shape")
        if "format" in self.input_dict.get("x1"):
            x1_format = self.input_dict.get("x1").get("format")
            if x1_format == "NC1HWC0":
                return shape_y[-1] * shape_y[-2]
        each_size = 1
        each_block_num = constant.BLOCK_SIZE // self.dsize
        each_block_align = (each_block_num + shape_y[-1] - 1) // shape_y[-1] \
                           * shape_y[-1]
        for j in range(len(shape_y) - 1, -1, -1):
            each_size = each_size * shape_y[j]
            if each_size * self.dsize >= constant.BLOCK_SIZE:
                each_size = each_block_align
                break
        return each_size

    def get_thread_num(self, block_num, each_loop, element_num):
        """
        get thread num

        Parameters
        ----------
        block_num: the num of block
        each_loop: the loop of each core
        element_num: the block size moved at a time
        Returns
        -------
        thread_num
        """
        if element_num * self.dsize < constant.BLOCK_SIZE:
            if block_num == 1 and each_loop > 1:
                thread_num = 2
            else:
                thread_num = 1
        else:
            thread_num = 1
            if each_loop > 1:
                thread_num = 2
        return thread_num

    def compute_crop(self):
        """
        compute crop

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        block_num, each_block_size, loop, tail = \
            self.get_blockdim_and_loop_cycle()
        shape_out = self.input_dict.get("y").get("shape")
        shape_out_len = get_shape_total_number(shape_out)
        offset_in = self.input_dict.get("offset")
        shape = self.input_dict.get("x1").get("shape")
        element_num, shape_len = self.get_element_num()
        x1_shape_list = get_elem_of_each_dim(shape, len(shape))
        shape = self.input_dict.get("x2").get("shape")
        x2_shape_list = get_elem_of_each_dim(shape, shape_len - 1)
        thread_n = self.get_thread_num(block_num, loop, element_num)

        with self.instance.for_range(0, block_num, block_num=block_num) \
                as block_id:
            ub_tmp = self.instance.Tensor(self.dtype, (256,),
                                          name="ub_tmp", scope=tik.scope_ubuf)
            self.instance.data_move(ub_tmp,
                                    self.x2_gm[0],
                                    constant.SID,
                                    constant.DEFAULT_NBURST, 1,
                                    constant.STRIDE_ZERO,
                                    constant.STRIDE_ZERO)
            count = self.instance.Scalar("int32")
            count.set_as(0)
            each_loop = self.instance.Scalar("int32")
            each_loop.set_as(loop)
            offset = self.instance.Scalar("int32")
            if tail > 0:
                with self.instance.if_scope(block_id < tail):
                    each_loop.set_as(each_loop + 1)
            offset.set_as(block_id * each_loop)
            with self.instance.if_scope(tik.all(block_id >= tail, tail > 0)):
                offset.set_as(block_id * (each_loop + 1) - (block_id - tail))
            out_offset = self.instance.Scalar("int32")
            out_offset.set_as(offset * element_num)
            cycles = shape_out_len // element_num
            tmp_offset = self.instance.Scalar("int32")
            tmp_offset.set_as(0)
            with self.instance.for_range(offset, cycles,
                                         thread_num=thread_n) as times:
                with self.instance.if_scope(count < each_loop):
                    x1_ub = self.instance.Tensor(self.dtype,
                                                 (self.one_max_size,),
                                                 name="x1_ub",
                                                 scope=tik.scope_ubuf)
                    x1_offset = self.instance.Scalar("int32")
                    x1_offset.set_as(0)
                    for q in range(shape_len):
                        mod = times
                        for s in range(q):
                            mod %= x2_shape_list[s]
                        mod = mod // x2_shape_list[q] + offset_in[q]
                        x1_offset.set_as(
                            x1_offset + mod * x1_shape_list[q])
                    if element_num * self.dsize < constant.BLOCK_SIZE \
                            and block_num > 1:
                        input_dict = {
                            "x1_ub": x1_ub,
                            "ub_tmp": ub_tmp,
                            "x1_offset": x1_offset,
                            "out_offset": out_offset,
                            "tmp_offset": tmp_offset,
                            "element_num": element_num,
                            "each_block_size": each_block_size,
                            "count": count,
                            "each_loop": each_loop, }
                        self.move_out_less_than32b(input_dict)
                        out_offset.set_as(out_offset + element_num)
                    else:
                        input_dict = {
                            "x1_ub": x1_ub,
                            "x1_offset": x1_offset,
                            "out_offset": out_offset,
                            "element_num": element_num,
                            "block_num": block_num,
                        }
                        self.data_move(input_dict)
                        out_offset.set_as(out_offset + element_num)
                    count.set_as(count + 1)

    def move_out_less_than32b(self, input_dict):
        """
      move data from ub to gm

      Parameters
      ----------
        input_dict: input_dict is a dict, the keys as follow:
                x1_ub: x1_ub is a tensor,store data from gm
                ub_tmp: ub_tmp is a tensor,store last loop 32b data from gm
                x1_offset: x1 gm data offset
                out_offset: output data offset
                tmp_offset: ub_tmp's offset
                element_num: each continuous segment
                each_block_size: each block process the number of element
                count: loop count
                each_loop: the total loop of each block
      Returns
      -------
      None
      """
        x1_ub = input_dict.get("x1_ub")
        ub_tmp = input_dict.get("ub_tmp")
        x1_offset = input_dict.get("x1_offset")
        out_offset = input_dict.get("out_offset")
        tmp_offset = input_dict.get("tmp_offset")
        element_num = input_dict.get("element_num")
        count = input_dict.get("count")
        each_loop = input_dict.get("each_loop")
        nburst = common_util.get_datamove_nburst(self.instance,
                                                 element_num * self.dsize)
        self.instance.data_move(x1_ub, self.x1_gm[x1_offset], constant.SID,
                                constant.DEFAULT_NBURST, nburst,
                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        loop_32b = (constant.BLOCK_SIZE // self.dsize) // element_num
        if (constant.BLOCK_SIZE // self.dsize) % element_num != 0:
            loop_32b = loop_32b + 1
        scalar = self.instance.Scalar(x1_ub.dtype)
        with self.instance.if_scope(count >= each_loop - loop_32b):
            with self.instance.for_range(0, element_num) as time:
                scalar.set_as(x1_ub[time])
                ub_tmp[tmp_offset + time].set_as(scalar)
            tmp_offset.set_as(tmp_offset + element_num)
            with self.instance.if_scope(count == each_loop - 1):
                out_offset.set_as(out_offset - (loop_32b - 1) * element_num)
                input_dict = {
                    "instance": self.instance,
                    "out_ub": ub_tmp,
                    "out_gm": self.y_gm,
                    "gm_offset": out_offset,
                    "element_num": element_num * loop_32b,
                    "dsize": self.dsize,
                }
                common_util.move_out_non32_alignment(input_dict)

        with self.instance.else_scope():
            nburst = common_util.get_datamove_nburst(self.instance,
                                                     element_num * self.dsize)
            self.instance.data_move(self.y_gm[out_offset],
                                    x1_ub,
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
                x1_ub: x1_ub is a tensor,store data from gm
                x1_offset: x1 gm data offset
                out_offset: output data offset
                element_num: each continuous segment
                block_num: blcok number
      Returns
      -------
      None
      """
        x1_ub = input_dict.get("x1_ub")
        out_offset = input_dict.get("out_offset")
        element_num = input_dict.get("element_num")
        block_num = input_dict.get("block_num")
        loop_cycle, last_ub_num = get_loop_param(element_num,
                                                 self.one_max_size)
        total_size = self.instance.Scalar("int32")
        total_size.set_as(self.one_max_size * self.dsize)
        ub_size = self.instance.Scalar("int32")
        ub_size.set_as(self.one_max_size)
        offset_x1 = self.instance.Scalar("int32")
        offset_x1.set_as(input_dict.get("x1_offset"))
        offset_out = self.instance.Scalar("int32")
        offset_out.set_as(out_offset)
        each_burst_num = constant.BLOCK_SIZE // self.dsize
        with self.instance.for_range(0, loop_cycle) as cycle:
            with self.instance.if_scope(cycle == loop_cycle - 1):
                total_size.set_as(last_ub_num * self.dsize)
                ub_size.set_as(last_ub_num)
            nburst = common_util.get_datamove_nburst(self.instance,
                                                     total_size)
            with self.instance.if_scope(
                    tik.all(cycle == loop_cycle - 1,
                            total_size % constant.BLOCK_SIZE != 0,
                            block_num > 1)):
                x1_ub_tmp = self.instance.Tensor(self.dtype, (32,),
                                                 name="x1_ub_tmp",
                                                 scope=tik.scope_ubuf)
                self.instance.data_move(x1_ub_tmp,
                                        self.x1_gm[offset_x1 +
                                                   ub_size - each_burst_num],
                                        constant.SID,
                                        constant.DEFAULT_NBURST, 1,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
                self.instance.data_move(self.y_gm[offset_out +
                                                  ub_size - each_burst_num],
                                        x1_ub_tmp,
                                        constant.SID,
                                        constant.DEFAULT_NBURST, 1,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
                with self.instance.if_scope(total_size > constant.BLOCK_SIZE):
                    self.instance.data_move(x1_ub,
                                            self.x1_gm[offset_x1],
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            nburst - 1,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
                    self.instance.data_move(self.y_gm[offset_out],
                                            x1_ub,
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            nburst - 1,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
            with self.instance.else_scope():
                self.instance.data_move(x1_ub,
                                        self.x1_gm[offset_x1],
                                        constant.SID,
                                        constant.DEFAULT_NBURST, nburst,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
                self.instance.data_move(self.y_gm[offset_out],
                                        x1_ub,
                                        constant.SID,
                                        constant.DEFAULT_NBURST, nburst,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
            offset_x1.set_as(offset_x1 + ub_size)
            offset_out.set_as(offset_out + ub_size)


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


def op_select_format(x, size, y, axis=2, offsets=(0), kernel_name="crop"):
    """
    select format dynamically
    """
    dtype_base = [
        "float16", "float", "int32", "int8", "int16", "int64", "uint8",
        "uint16", "uint32", "uint64"
    ]
    dtype_lhisi = [
        "float16", "int32", "int8", "int16", "int64", "uint8",
        "uint16", "uint32", "uint64"
    ]

    ori_format = x.get("ori_format").upper()
    ori_shape = x.get("ori_shape")

    dtype_out = dtype_base
    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
        dtype_out = dtype_lhisi

    if axis < 0:
        axis = axis + len(ori_shape)

    format_out = ["ND"] * len(dtype_out)
    if ori_format == "NCHW" and len(ori_shape) == 4 and axis >= 2:
        format_out = format_out + ["NC1HWC0"] * len(dtype_out)
        dtype_out = dtype_out + dtype_out

    dtype_str = ','.join(dtype_out)
    format_str = ','.join(format_out)

    input0 = gen_param(
        classify="input0", name="x", datatype=dtype_str, format=format_str)
    input1 = gen_param(
        classify="input1", name="size", datatype=dtype_str, format=format_str)
    output0 = gen_param(
        classify="output0", name="y", datatype=dtype_str, format=format_str)
    param_list = [input0, input1, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def check_and_adjust_offset(input_dict):
    """
    check the parameters is valid, if one is invalid,then raise error
    adjust offset's length as the same as len(x1_shape)

    Parameters
    ----------
    input_dict: input_dict is a dict, the keys as follow:
                x1: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                x2: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                y: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                axis: crop start with axis
                offsets: crop start offset of each axis
                kernel_name: cce kernel name, default value is "crop"
    Returns
    -------
    None
    """
    util.check_kernel_name(input_dict.get('kernel_name'))
    x1_dtype = input_dict.get("x1").get("dtype").lower()
    x1_shape = input_dict.get("x1").get("shape")
    x2_dtype = input_dict.get("x2").get("dtype").lower()
    x2_shape = input_dict.get("x2").get("shape")
    y_dtype = input_dict.get("y").get("dtype").lower()
    y_shape = input_dict.get("y").get("shape")

    util.check_shape_rule(x1_shape)
    util.check_tensor_shape_size(x1_shape)
    util.check_dtype_rule(x1_dtype,
                          ("int8", "uint8", "int16", "uint16", "int32",
                           "uint32", "int64", "uint64", "float16",
                           "float32"))

    util.check_shape_rule(x2_shape)
    util.check_tensor_shape_size(x2_shape)
    util.check_dtype_rule(x2_dtype,
                          ("int8", "uint8", "int16", "uint16", "int32",
                           "uint32", "int64", "uint64", "float16",
                           "float32"))

    util.check_shape_rule(y_shape)
    util.check_tensor_shape_size(y_shape)
    util.check_dtype_rule(y_dtype, ("int8", "uint8", "int16", "uint16", "int32",
                                    "uint32", "int64", "uint64", "float16",
                                    "float32"))
    if x2_dtype != y_dtype or y_dtype != x1_dtype:
        raise RuntimeError("size's datatype must be the same as \
        y's datatype and x's datatype")

    if not check_same_shape(y_shape, x2_shape):
        raise RuntimeError(
            "y's shape must be the same as size's shape")
    if len(x2_shape) != len(x1_shape):
        raise RuntimeError(
            "x's dim must be the same as size's dim")
    # check his-es check offset
    axis = input_dict.get("axis")
    if axis >= len(x1_shape) or axis < -len(x1_shape):
        raise RuntimeError("axis out of range")
    if axis < 0:
        input_dict["axis"] = axis + len(x1_shape)
        axis = axis + len(x1_shape)
    # the same verify as caffe
    offset = input_dict.get("offset")
    x1_ori_shape = input_dict.get("x1").get("shape")
    if 'ori_shape' in input_dict.get("x1"):
        x1_ori_shape = input_dict.get("x1").get("ori_shape")
    offset_final = [0] * len(x1_shape)
    if len(offset) == 1:
        for i in range(axis, len(x1_ori_shape)):
            offset_final[i] = offset[0]
    elif len(offset) != 0:
        if len(offset) != len(x1_ori_shape) - axis:
            raise RuntimeError(
                "axis+len(offset) must equals input dim(x)")
        offset_final[axis:len(x1_ori_shape)] = offset
    len_offset_final = len(offset_final)
    for i in range(len_offset_final):
        if x1_shape[i] - offset_final[i] < x2_shape[i]:
            raise RuntimeError(
                "size's dimension i[%s]'s size can't be bigger than \
                x's size minus offset" % i)
    input_dict["offset"] = offset_final


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


def check_same_shape(shape_x, shape_y):
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
    shape_y_len = len(shape_y)
    if shape_x_len != shape_y_len:
        return False
    for k in range(shape_x_len):
        if shape_x[k] != shape_y[k]:
            return False

    return True


def get_elem_of_each_dim(shape, element_num):
    """
    get element of each dim

    Parameters
    ----------
    shape: out put shape

    Returns
    -------
    None
    """
    elem_of_each_dim = [1] * len(shape)
    for i in range(element_num):
        j = i + 1
        while j < element_num:
            elem_of_each_dim[i] = elem_of_each_dim[i] * shape[j]
            j = j + 1

    return elem_of_each_dim
