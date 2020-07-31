#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

copy_only
"""

from te.platform.fusion_manager import fusion_manager
from te import tik
from topi.cce import util
from te import platform as tbe_platform

def _apply_mem(tik_instance, dtype, shape, name, scope=tik.scope_ubuf):
    """apply mem fuc

    Parameters
    ----------
    tik_instance: tik_instance
        tik_instance
    dtype: str
        ub dtype
    shape: list
        ub shape
    name: str
        ub name
    scope: scope
        scope_ubuf or scope_gm
    Returns
    -------
    Tensor: Tensor
    """
    return tik_instance.Tensor(dtype, shape, name=name, scope=scope)


def _get_ceil_int(int1, int2):
    _result = int1 // int2
    if int1 % int2 == 0:
        return _result
    return _result + 1


# pylint: disable=locally-disabled,too-many-instance-attributes
class CopyOnly():
    """Function: use to finish CopyOnly main functions
    """
    def __init__(self, src, dst):
        """
        init CopyOnly parameters

        Parameters
        ----------
        src : dict
            shape and dtype of input
        dst: dict
            shape and dtype of output, should be same shape and type as input

        Returns
        -------
        None
        """
        self.src_shape = src.get("shape")
        self.src_dtype = src.get("dtype").lower()
        self.dst_shape = dst.get("shape")
        self.dst_dtype = dst.get("dtype").lower()
        if self.dst_dtype == "bool":
            self.dst_dtype = "int8"
        self.data_size = util.check_tensor_shape_size(list(self.src_shape))

        if len(self.dst_shape) == 0:
            self.data_dst_size = 1
            self.dst_shape = [1]
        else:
            self.data_dst_size = \
                util.check_tensor_shape_size(list(self.dst_shape))

        if self.data_size != self.data_dst_size:
            raise RuntimeError("The size of src and des is not equal,"
                               " can not use fuc(CopyOnly)")
        # get dtype size, float16 size = 2 byte   / float32 size = 4 byte
        self.dtype_size = \
            tbe_platform.cce_intrin.get_bit_len(self.src_dtype) // 8
        # get one block data size, block align len
        # the len in one block = 16 fp16 and = 8 fp32
        self.data_len_one_block = 32 // self.dtype_size
        self.data_len_one_vector = self.data_len_one_block * 8

        self.ub_availble = \
            tbe_platform.cce_conf.get_soc_spec(
                tbe_platform.cce_conf.UB_SIZE) - 8 * 1024
        self.ub_max_data = self.ub_availble // self.dtype_size

        self.tik_instance = tik.Tik()
        self.core_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

        # input and output tensor in gm
        self.src_gm = self.tik_instance.Tensor(
            self.src_dtype,
            self.src_shape,
            name="src_gm",
            scope=tik.scope_gm)
        self.dst_gm = self.tik_instance.Tensor(
            self.dst_dtype,
            self.dst_shape,
            name="dst_gm",
            scope=tik.scope_gm)
        self.data_ub = None

    def copy_only(self):
        """copy all data from src to des
        """
        # core scedule
        core_data = self.data_size // self.core_num
        if core_data == 0:
            core_data = 1
        core_data = \
            _get_ceil_int(core_data, self.data_len_one_block) * \
            self.data_len_one_block
        core_used = _get_ceil_int(self.data_size, core_data)
        core_last = self.data_size - (core_data * (core_used - 1))
        # calcu max copy segment
        copy_segment = self.ub_max_data // 2
        copy_segment = \
            (_get_ceil_int(copy_segment, self.data_len_one_block) - 1) * \
            self.data_len_one_block
        # core process
        with self.tik_instance.for_range(
                0, core_used, block_num=core_used) as core_index:
            with self.tik_instance.if_scope(core_index < (core_used - 1)):
                copy_loop = core_data // copy_segment
                copy_tail = core_data % copy_segment
                thread_num = 2
                if copy_loop < 2:
                    thread_num = 1
                with self.tik_instance.for_range(
                        0, copy_loop, thread_num=thread_num) as loop_index:
                    gm_in_offset = core_index*core_data + \
                                   loop_index*copy_segment
                    gm_out_offset = core_index*core_data + \
                                    loop_index*copy_segment
                    self._copy_in_and_out(copy_segment,
                                          gm_in_offset,
                                          gm_out_offset)
                if copy_tail != 0:
                    gm_in_offset = core_index*core_data + \
                                   copy_loop*copy_segment
                    gm_out_offset = core_index*core_data + \
                                    copy_loop*copy_segment
                    self._copy_in_and_out(copy_tail,
                                          gm_in_offset,
                                          gm_out_offset)
            with self.tik_instance.else_scope():
                copy_loop = core_last // copy_segment
                copy_tail = core_last % copy_segment
                thread_num = 2
                if copy_loop < 2:
                    thread_num = 1
                with self.tik_instance.for_range(
                        0, copy_loop, thread_num=thread_num) as loop_index:
                    gm_in_offset = core_index*core_data + \
                                   loop_index*copy_segment
                    gm_out_offset = core_index*core_data + \
                                    loop_index*copy_segment
                    self._copy_in_and_out(copy_segment,
                                          gm_in_offset, gm_out_offset)
                if copy_tail != 0:
                    gm_in_offset = core_index*core_data + \
                                   copy_loop*copy_segment
                    gm_out_offset = core_index*core_data + \
                                    copy_loop*copy_segment
                    self._copy_in_and_out(copy_tail,
                                          gm_in_offset, gm_out_offset)

    def _copy_in_and_out(self, copy_len, copy_in_offset, copy_out_offset):
        nbust = _get_ceil_int(copy_len, self.data_len_one_block)
        self.data_ub = _apply_mem(self.tik_instance, self.dst_dtype,
                                  [nbust*self.data_len_one_block],
                                  "data_ub")
        self.tik_instance.data_move(self.data_ub[0],
                                    self.src_gm[copy_in_offset],
                                    0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.dst_gm[copy_out_offset],
                                    self.data_ub[0],
                                    0, 1, nbust, 0, 0)

    def run_tik(self, kernel_name):
        """cal tik_instance according to mode
        """
        self.copy_only()
        self.tik_instance.BuildCCE(
            kernel_name=kernel_name,
            inputs=[self.src_gm],
            outputs=[self.dst_gm])
        return self.tik_instance


@fusion_manager.register("copy_only")
def copy_only_compute(src, dst,
                      kernel_name):
    """
    algorithm: copy_only

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name

    Returns
    -------
    instance
    """
    res = CopyOnly(src, dst)

    return res.run_tik(kernel_name)


@util.check_input_type(dict, dict, str)
def copy_only(src, dst, kernel_name):
    """
    algorithm: copy_only

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name

    Returns
    -------
    None
    """
    res = copy_only_compute(src, dst, kernel_name)

    return res
