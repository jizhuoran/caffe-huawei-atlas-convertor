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

split_d
"""
from __future__ import absolute_import

from functools import reduce as functools_reduce
import numpy as np
import te.lang.cce
from te import tvm
from te import tik
from te import platform as tbe_platform
from te.platform.cce_build import build_config
from topi.cce import util
from impl.copy_only import copy_only
from impl.split_last_dim import split_last_dim
from impl.split_last_dim import check_use_last_dim_branch
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

# one block size
BLOCK_SIZE = 32
# vtranspose can deal 16*16
TRANSPOSE_SIZE = 256


# pylint: disable=locally-disabled,unused-argument,too-many-locals
# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-statements
class SplitMov:
    """Function: use to finish SplitMov main functions
    """

    def __init__(self,
                 shape,
                 dtype,
                 split_dim,
                 num_split,
                 size_splits=None,
                 kernel_name="split_d"):
        """init split_d base parameters
        """
        self.tik_instance = tik.Tik()
        self.aicore_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        self.kernel_name = kernel_name
        self.dtype = dtype
        self.dtype_size = tbe_platform.cce_intrin.get_bit_len(self.dtype) // 8
        self.one_block_ele = 32 // self.dtype_size
        self.half_ub_ele = (
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) //
            self.dtype_size // 2 - self.one_block_ele)

        if not size_splits:
            size_splits = [shape[split_dim] // num_split] * num_split
        self.size_splits = []
        self.output_shapes = []
        if split_dim == 0:
            self.input_shape = [int(np.prod(shape))]
            for size in size_splits:
                self.size_splits.append(self.input_shape[0] //
                                        shape[split_dim] * size)
                self.output_shapes.append(
                    [self.input_shape[0] // shape[split_dim] * size])
            self.split_dim = 0
        else:
            out_dim = int(np.prod(shape[:split_dim]))
            if out_dim == 1:
                self.input_shape = [int(np.prod(shape))]
                for size in size_splits:
                    self.size_splits.append(self.input_shape[0] //
                                            shape[split_dim] * size)
                    self.output_shapes.append(
                        [self.input_shape[0] // shape[split_dim] * size])
                self.split_dim = 0
            else:
                self.input_shape = [
                    int(np.prod(shape[:split_dim])),
                    int(np.prod(shape[split_dim:len(shape)]))
                ]
                for size in size_splits:
                    self.size_splits.append(self.input_shape[1] //
                                            shape[split_dim] * size)
                    self.output_shapes.append([
                        self.input_shape[0],
                        self.input_shape[1] // shape[split_dim] * size
                    ])
                self.split_dim = 1

        self.input_tensor, self.output_tensors = self.init_gm_tensor()

    def init_gm_tensor(self):
        """init gm tensor
        """
        input_tensor = self.tik_instance.Tensor(
            self.dtype, self.input_shape, name="gm_input", scope=tik.scope_gm)

        output_tensors = []
        for index, tensor_shape in enumerate(self.output_shapes):
            tensor_name = "gm_output_" + str(index)
            gm_tensor = self.tik_instance.Tensor(
                self.dtype, tensor_shape, name=tensor_name, scope=tik.scope_gm)
            output_tensors.append(gm_tensor)

        return input_tensor, output_tensors

    def get_one_core_ele(self, output_shape):
        """get_one_core_ele
        """
        total_ele = output_shape[self.split_dim]
        last_ele = total_ele % self.aicore_num

        if last_ele == 0:
            one_core_ele = total_ele // self.aicore_num
        else:
            one_core_ele = (
                total_ele // self.aicore_num // self.one_block_ele *
                self.one_block_ele)
            last_ele = total_ele - (self.aicore_num - 1) * one_core_ele

        return one_core_ele, last_ele

    def split_compute_for_tensor(self, move_in_index, output_tensor,
                                 one_core_ele):
        """split_compute_for_tensor
        """
        loop_burst_len = 0
        if one_core_ele < self.half_ub_ele:
            loop_num = 0
            one_loop_ele = 0
            last_ele = one_core_ele
            ub_size = self.half_ub_ele
        else:
            if one_core_ele % self.half_ub_ele < self.one_block_ele:
                ub_size = self.half_ub_ele - self.one_block_ele
            else:
                ub_size = self.half_ub_ele
            loop_num = one_core_ele // ub_size
            one_loop_ele = ub_size
            last_ele = one_core_ele % ub_size
            loop_burst_len = ub_size // self.one_block_ele
        if loop_num > 0:
            if loop_num > 1:
                multi_thread = 2
            else:
                multi_thread = 1
            with self.tik_instance.for_range(
                    0, loop_num, thread_num=multi_thread) as inner_loop:
                ub_tensor = self.tik_instance.Tensor(
                    self.dtype, (ub_size,), name="ub_tmp", scope=tik.scope_ubuf)
                offset = inner_loop * one_loop_ele
                self.tik_instance.data_move(
                    ub_tensor, self.input_tensor[move_in_index][offset], 0, 1,
                    loop_burst_len, 0, 0)
                self.tik_instance.data_move(output_tensor[offset], ub_tensor, 0,
                                            1, loop_burst_len, 0, 0)
        if last_ele > 0:
            with self.tik_instance.for_range(0, 1) as _:
                ub_tensor = self.tik_instance.Tensor(
                    self.dtype, (ub_size,), name="ub_tmp", scope=tik.scope_ubuf)
                offset = loop_num * one_loop_ele
                if last_ele // self.one_block_ele != 0:
                    last_burst_len = last_ele // self.one_block_ele
                    self.tik_instance.data_move(
                        ub_tensor, self.input_tensor[move_in_index][offset], 0,
                        1, last_burst_len, 0, 0)
                    self.tik_instance.data_move(output_tensor[offset],
                                                ub_tensor, 0, 1, last_burst_len,
                                                0, 0)

                if last_ele % self.one_block_ele != 0:
                    ub_last = self.tik_instance.Tensor(
                        self.dtype, (self.one_block_ele,),
                        name="ub_last",
                        scope=tik.scope_ubuf)
                    offset = one_core_ele - self.one_block_ele
                    self.tik_instance.data_move(
                        ub_last, self.input_tensor[move_in_index][offset], 0, 1,
                        1, 0, 0)
                    self.tik_instance.data_move(output_tensor[offset], ub_last,
                                                0, 1, 1, 0, 0)

    def split_compute_first_dim_for_core(self, core_index):
        """split_compute_first_dim_for_core
        """
        out_offset = 0
        for tensor_index, output_shape in enumerate(self.output_shapes):
            one_core_ele, last_ele = self.get_one_core_ele(output_shape)
            if last_ele == 0:
                move_in_index = (out_offset + one_core_ele * core_index)
                move_out_index = (one_core_ele * core_index)
                self.split_compute_for_tensor(
                    move_in_index,
                    self.output_tensors[tensor_index][move_out_index],
                    one_core_ele)
            else:
                with self.tik_instance.if_scope(
                        core_index < self.aicore_num - 1):
                    move_in_index = (out_offset + one_core_ele * core_index)
                    move_out_index = (one_core_ele * core_index)
                    self.split_compute_for_tensor(
                        move_in_index,
                        self.output_tensors[tensor_index][move_out_index],
                        one_core_ele)
                with self.tik_instance.else_scope():
                    move_in_index = (out_offset + one_core_ele * core_index)
                    move_out_index = (one_core_ele * core_index)
                    self.split_compute_for_tensor(
                        move_in_index,
                        self.output_tensors[tensor_index][move_out_index],
                        last_ele)
            out_offset += output_shape[self.split_dim]

    def split_compute_last_dim_for_core(self, core_index):
        """split_compute_last_dim_for_core
        """
        is_div = self.input_shape[0] % self.aicore_num
        if is_div != 0:
            out_offset = 0
            out_loop = self.input_shape[0]
            for tensor_index, output_shape in enumerate(self.output_shapes):
                one_core_ele, last_ele = self.get_one_core_ele(output_shape)
                if last_ele == 0:
                    with self.tik_instance.for_range(
                            0, out_loop, thread_num=2) as loop_index:
                        move_in_index = (
                            out_offset + one_core_ele * core_index +
                            loop_index * self.input_shape[self.split_dim])
                        move_out_index = (
                            one_core_ele * core_index +
                            loop_index * output_shape[self.split_dim])
                        self.split_compute_for_tensor(
                            move_in_index,
                            self.output_tensors[tensor_index][move_out_index],
                            one_core_ele)
                else:
                    with self.tik_instance.if_scope(
                            core_index < self.aicore_num - 1):
                        with self.tik_instance.for_range(
                                0, out_loop, thread_num=2) as loop_index:
                            move_in_index = (
                                out_offset + one_core_ele * core_index +
                                loop_index * self.input_shape[self.split_dim])
                            move_out_index = (
                                one_core_ele * core_index +
                                loop_index * output_shape[self.split_dim])
                            self.split_compute_for_tensor(
                                move_in_index, self.output_tensors[tensor_index]
                                [move_out_index], one_core_ele)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.for_range(
                                0, out_loop, thread_num=2) as loop_index:
                            move_in_index = (
                                out_offset + one_core_ele * core_index +
                                loop_index * self.input_shape[self.split_dim])
                            move_out_index = (
                                one_core_ele * core_index +
                                loop_index * output_shape[self.split_dim])
                            self.split_compute_for_tensor(
                                move_in_index, self.output_tensors[tensor_index]
                                [move_out_index], last_ele)
                out_offset += output_shape[self.split_dim]
        else:
            out_offset = 0
            out_loop = self.input_shape[0] // self.aicore_num
            thread_num = 1
            if out_loop != 1:
                thread_num = 2
            for tensor_index, output_shape in enumerate(self.output_shapes):
                one_core_ele = output_shape[self.split_dim]
                with self.tik_instance.for_range(
                        0, out_loop, thread_num=thread_num) as loop_index:
                    move_in_index = (
                        out_offset +
                        core_index * out_loop * self.input_shape[self.split_dim]
                        + loop_index * self.input_shape[self.split_dim])
                    move_out_index = (
                        core_index * out_loop * output_shape[self.split_dim] +
                        loop_index * output_shape[self.split_dim])
                    self.split_compute_for_tensor(
                        move_in_index,
                        self.output_tensors[tensor_index][move_out_index],
                        one_core_ele)
                out_offset += output_shape[self.split_dim]

    def split_mov_compute(self):
        """split_mov_compute
        """
        if self.split_dim == 0:
            with self.tik_instance.for_range(
                    0, self.aicore_num, block_num=self.aicore_num) as index:
                self.split_compute_first_dim_for_core(index)
        else:
            with self.tik_instance.for_range(
                    0, self.aicore_num, block_num=self.aicore_num) as index:
                self.split_compute_last_dim_for_core(index)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_tensor],
            outputs=self.output_tensors,
            enable_l2=False)

    def check_whether_use_split_mov(self):
        """check if split_d schedule support this shape
        """
        is_supported = True
        for _, output_shape in enumerate(self.output_shapes):
            split_dim_len = output_shape[self.split_dim]
            if self.split_dim == 0 and \
                    split_dim_len // self.aicore_num < 2 * self.one_block_ele:
                is_supported = False
                return is_supported
            if self.split_dim == 1 and \
                    self.input_shape[0] % self.aicore_num == 0 and \
                    split_dim_len < 2 * self.one_block_ele:
                is_supported = False
                return is_supported
            if self.split_dim == 1 and \
                    self.input_shape[0] % self.aicore_num != 0 and \
                    split_dim_len // self.aicore_num < 2 * self.one_block_ele:
                is_supported = False
                return is_supported

        return is_supported


class SplitLastDimVnv:
    """Function: use to finish SplitLastDimVnv main functions
    """

    def __init__(self, shape, dtype, output_shapes, split_dim, num_split,
                 kernel_name):
        """init split_d base parameters
        """
        self.tik_instance = tik.Tik()
        self.aicore_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        self.kernel_name = kernel_name
        self.dtype = dtype
        self.dtype_size = tbe_platform.cce_intrin.get_bit_len(self.dtype) // 8
        self.input_shape = shape
        self.input_size = functools_reduce(lambda x, y: x * y, self.input_shape)
        self.split_dim = split_dim
        self.num_split = num_split
        self.output_shapes = output_shapes
        self.output_size = functools_reduce(lambda x, y: x * y,
                                            self.output_shapes[0])
        self.input_tensor, self.output_tensors = self.init_gm_tensor()

    def init_gm_tensor(self):
        """init gm tensor
        """
        input_tensor = self.tik_instance.Tensor(
            self.dtype, self.input_shape, name="gm_input", scope=tik.scope_gm)

        output_tensors = []
        for index, tensor_shape in enumerate(self.output_shapes):
            tensor_name = "gm_output_" + str(index)
            gm_tensor = self.tik_instance.Tensor(
                self.dtype, tensor_shape, name=tensor_name, scope=tik.scope_gm)
            output_tensors.append(gm_tensor)

        return input_tensor, output_tensors

    def split_last_dim_vnc_compute(self):
        """split_last_dim_vnc_compute
        """
        if self.num_split == 3 and self.input_size >= TRANSPOSE_SIZE * 8 * 3:
            self.split_last_dim_vnc_compute_for_three()
            return
        mov_size = TRANSPOSE_SIZE * self.num_split
        mov_num = (self.input_size + mov_size - 1) // mov_size
        core_loop = mov_num // self.aicore_num
        core_last = mov_num % self.aicore_num
        with self.tik_instance.for_range(
                0, self.aicore_num, block_num=self.aicore_num) as core_idx:
            src_offset_core = core_idx * core_loop * mov_size
            dst_offset_core = core_idx * core_loop * TRANSPOSE_SIZE
            thread = 1
            if core_loop > 1:
                thread = 2
            with self.tik_instance.for_range(
                    0, core_loop, thread_num=thread) as loop_idx:
                ub_x = self.tik_instance.Tensor(self.dtype, (mov_size,),
                                                tik.scope_ubuf, "ub_x")
                ub_y = self.tik_instance.Tensor(self.dtype, (mov_size,),
                                                tik.scope_ubuf, "ub_y")
                ub_m = self.tik_instance.Tensor(self.dtype, (TRANSPOSE_SIZE,),
                                                tik.scope_ubuf, "ub_m")
                ub_n = self.tik_instance.Tensor(self.dtype, (TRANSPOSE_SIZE,),
                                                tik.scope_ubuf, "ub_n")
                src_offset = src_offset_core + loop_idx * mov_size
                dst_offset = dst_offset_core + loop_idx * TRANSPOSE_SIZE

                # copy gm to ub
                self.tik_instance.data_move(
                    ub_x, self.input_tensor[src_offset], 0, 1,
                    mov_size * self.dtype_size // BLOCK_SIZE, 0, 0)

                # vadds & vtranspose
                for num_idx in range(self.num_split):
                    src_offset_ub = num_idx * BLOCK_SIZE // self.dtype_size
                    dst_offset_ub = num_idx * TRANSPOSE_SIZE
                    self.tik_instance.vadds(128, ub_m, ub_x[src_offset_ub], 0,
                                            2, 1, self.num_split, 8,
                                            self.num_split * 8)
                    self.tik_instance.vtranspose(ub_y[dst_offset_ub], ub_m)

                for num_idx in range(self.num_split):
                    src_offset_ub = num_idx * BLOCK_SIZE // self.dtype_size
                    self.tik_instance.vadds(128, ub_m, ub_y[src_offset_ub], 0,
                                            2, 1, self.num_split, 8,
                                            self.num_split * 8)
                    self.tik_instance.vtranspose(ub_n, ub_m)
                    # copy ub to gm
                    self.tik_instance.data_move(
                        self.output_tensors[num_idx][dst_offset], ub_n, 0, 1,
                        TRANSPOSE_SIZE * self.dtype_size // BLOCK_SIZE, 0, 0)
            if core_last != 0:
                src_offset_core_last = (core_loop * self.aicore_num +
                                        core_idx) * mov_size
                dst_offset_core_last = (core_loop * self.aicore_num +
                                        core_idx) * TRANSPOSE_SIZE
                with self.tik_instance.for_range(0, 1):
                    ub_x = self.tik_instance.Tensor(self.dtype, (mov_size,),
                                                    tik.scope_ubuf, "ub_x")
                    ub_y = self.tik_instance.Tensor(self.dtype, (mov_size,),
                                                    tik.scope_ubuf, "ub_y")
                    ub_m = self.tik_instance.Tensor(self.dtype,
                                                    (TRANSPOSE_SIZE,),
                                                    tik.scope_ubuf, "ub_m")
                    ub_n = self.tik_instance.Tensor(self.dtype,
                                                    (TRANSPOSE_SIZE,),
                                                    tik.scope_ubuf, "ub_n")
                    # copy gm to ub
                    with self.tik_instance.if_scope(core_idx < core_last - 1):
                        self.tik_instance.data_move(
                            ub_x, self.input_tensor[src_offset_core_last], 0, 1,
                            mov_size * self.dtype_size // BLOCK_SIZE, 0, 0)
                    with self.tik_instance.if_scope(core_idx == core_last - 1):
                        self.tik_instance.data_move(
                            ub_x, self.input_tensor[self.input_size - mov_size],
                            0, 1, mov_size * self.dtype_size // BLOCK_SIZE, 0,
                            0)

                    # vadds & vtranspose
                    for num_idx in range(self.num_split):
                        src_offset_ub = num_idx * BLOCK_SIZE // self.dtype_size
                        dst_offset_ub = num_idx * TRANSPOSE_SIZE
                        self.tik_instance.vadds(128, ub_m, ub_x[src_offset_ub],
                                                0, 2, 1, self.num_split, 8,
                                                self.num_split * 8)
                        self.tik_instance.vtranspose(ub_y[dst_offset_ub], ub_m)

                    for num_idx in range(self.num_split):
                        src_offset_ub = num_idx * BLOCK_SIZE // self.dtype_size
                        self.tik_instance.vadds(128, ub_m, ub_y[src_offset_ub],
                                                0, 2, 1, self.num_split, 8,
                                                self.num_split * 8)
                        self.tik_instance.vtranspose(ub_n, ub_m)
                        # copy ub to gm
                        with self.tik_instance.if_scope(
                                core_idx < core_last - 1):
                            self.tik_instance.data_move(
                                self.output_tensors[num_idx]
                                [dst_offset_core_last], ub_n, 0, 1,
                                TRANSPOSE_SIZE * self.dtype_size // BLOCK_SIZE,
                                0, 0)
                        with self.tik_instance.if_scope(core_idx == core_last -
                                                        1):
                            self.tik_instance.data_move(
                                self.output_tensors[num_idx][self.output_size -
                                                             TRANSPOSE_SIZE],
                                ub_n, 0, 1,
                                TRANSPOSE_SIZE * self.dtype_size // BLOCK_SIZE,
                                0, 0)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_tensor],
            outputs=self.output_tensors,
            enable_l2=False)
        return self.tik_instance

    def split_last_dim_vnc_compute_for_three(self):
        """split_last_dim_vnc_compute_for_three
        """
        mov_size = TRANSPOSE_SIZE * 8 * 3
        mov_num = (self.input_size + mov_size - 1) // mov_size
        core_loop = mov_num // self.aicore_num
        core_last = mov_num % self.aicore_num

        with self.tik_instance.for_range(
                0, self.aicore_num, block_num=self.aicore_num) as core_idx:
            src_offset_core = core_idx * core_loop * mov_size
            dst_offset_core = core_idx * core_loop * TRANSPOSE_SIZE * 8
            thread = 1
            if core_loop > 1:
                thread = 2
            with self.tik_instance.for_range(
                    0, core_loop, thread_num=thread) as loop_idx:
                ub_x = self.tik_instance.Tensor(self.dtype, (mov_size,),
                                                tik.scope_ubuf, "ub_x")
                ub_y = self.tik_instance.Tensor(self.dtype, (mov_size,),
                                                tik.scope_ubuf, "ub_y")
                ub_m = self.tik_instance.Tensor(self.dtype,
                                                (TRANSPOSE_SIZE * 8,),
                                                tik.scope_ubuf, "ub_m")
                ub_n = self.tik_instance.Tensor(self.dtype,
                                                (TRANSPOSE_SIZE * 8,),
                                                tik.scope_ubuf, "ub_n")
                src_offset = src_offset_core + loop_idx * mov_size
                dst_offset = dst_offset_core + loop_idx * TRANSPOSE_SIZE * 8

                # copy gm to ub
                self.tik_instance.data_move(
                    ub_x, self.input_tensor[src_offset], 0, 1,
                    mov_size * self.dtype_size // BLOCK_SIZE, 0, 0)

                # vadds & vtranspose
                for num_idx in range(3):
                    src_offset_ub = num_idx * BLOCK_SIZE // self.dtype_size
                    dst_offset_ub = num_idx * TRANSPOSE_SIZE
                    self.tik_instance.vadds(128, ub_m, ub_x[src_offset_ub], 0,
                                            2 * 8, 1, 3, 8, 3 * 8)
                    for trans_idx in range(8):
                        src_offset_trans = trans_idx * TRANSPOSE_SIZE
                        dst_offset_trans = dst_offset_ub + 3 * trans_idx * \
                                           TRANSPOSE_SIZE
                        self.tik_instance.vtranspose(ub_y[dst_offset_trans],
                                                     ub_m[src_offset_trans])

                for num_idx in range(self.num_split):
                    src_offset_ub = num_idx * BLOCK_SIZE // self.dtype_size
                    self.tik_instance.vadds(128, ub_m, ub_y[src_offset_ub], 0,
                                            2 * 8, 1, 3, 8, 3 * 8)
                    for trans_idx in range(8):
                        src_offset_trans = trans_idx * TRANSPOSE_SIZE
                        dst_offset_trans = trans_idx * TRANSPOSE_SIZE
                        self.tik_instance.vtranspose(ub_n[dst_offset_trans],
                                                     ub_m[src_offset_trans])
                    # copy ub to gm
                    self.tik_instance.data_move(
                        self.output_tensors[num_idx][dst_offset], ub_n, 0, 1,
                        TRANSPOSE_SIZE * 8 * self.dtype_size // BLOCK_SIZE, 0,
                        0)
            if core_last != 0:
                src_offset_core_last = (core_loop * self.aicore_num +
                                        core_idx) * mov_size
                dst_offset_core_last = (core_loop * self.aicore_num +
                                        core_idx) * TRANSPOSE_SIZE * 8
                with self.tik_instance.for_range(0, 1):
                    ub_x = self.tik_instance.Tensor(self.dtype, (mov_size,),
                                                    tik.scope_ubuf, "ub_x")
                    ub_y = self.tik_instance.Tensor(self.dtype, (mov_size,),
                                                    tik.scope_ubuf, "ub_y")
                    ub_m = self.tik_instance.Tensor(self.dtype,
                                                    (TRANSPOSE_SIZE * 8,),
                                                    tik.scope_ubuf, "ub_m")
                    ub_n = self.tik_instance.Tensor(self.dtype,
                                                    (TRANSPOSE_SIZE * 8,),
                                                    tik.scope_ubuf, "ub_n")
                    # copy gm to ub
                    with self.tik_instance.if_scope(core_idx < core_last - 1):
                        self.tik_instance.data_move(
                            ub_x, self.input_tensor[src_offset_core_last], 0, 1,
                            mov_size * self.dtype_size // BLOCK_SIZE, 0, 0)
                    with self.tik_instance.if_scope(core_idx == core_last - 1):
                        self.tik_instance.data_move(
                            ub_x, self.input_tensor[self.input_size - mov_size],
                            0, 1, mov_size * self.dtype_size // BLOCK_SIZE, 0,
                            0)

                    # vadds & vtranspose
                    for num_idx in range(3):
                        src_offset_ub = num_idx * BLOCK_SIZE // self.dtype_size
                        dst_offset_ub = num_idx * TRANSPOSE_SIZE
                        self.tik_instance.vadds(128, ub_m, ub_x[src_offset_ub],
                                                0, 2 * 8, 1, 3, 8, 3 * 8)
                        for trans_idx in range(8):
                            src_offset_trans = trans_idx * TRANSPOSE_SIZE
                            dst_offset_trans = dst_offset_ub + 3 * trans_idx * \
                                               TRANSPOSE_SIZE
                            self.tik_instance.vtranspose(
                                ub_y[dst_offset_trans], ub_m[src_offset_trans])

                    for num_idx in range(self.num_split):
                        src_offset_ub = num_idx * BLOCK_SIZE // self.dtype_size
                        self.tik_instance.vadds(128, ub_m, ub_y[src_offset_ub],
                                                0, 2 * 8, 1, 3, 8, 3 * 8)
                        for trans_idx in range(8):
                            src_offset_trans = trans_idx * TRANSPOSE_SIZE
                            dst_offset_trans = trans_idx * TRANSPOSE_SIZE
                            self.tik_instance.vtranspose(
                                ub_n[dst_offset_trans], ub_m[src_offset_trans])
                        # copy ub to gm
                        with self.tik_instance.if_scope(
                                core_idx < core_last - 1):
                            self.tik_instance.data_move(
                                self.output_tensors[num_idx]
                                [dst_offset_core_last], ub_n, 0, 1,
                                TRANSPOSE_SIZE * 8 * self.dtype_size //
                                BLOCK_SIZE, 0, 0)
                        with self.tik_instance.if_scope(core_idx == core_last -
                                                        1):
                            self.tik_instance.data_move(
                                self.output_tensors[num_idx][self.output_size -
                                                             TRANSPOSE_SIZE *
                                                             8], ub_n, 0, 1,
                                TRANSPOSE_SIZE * 8 * self.dtype_size //
                                BLOCK_SIZE, 0, 0)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_tensor],
            outputs=self.output_tensors,
            enable_l2=False)
        return self.tik_instance


def split_d_compute(input_value,
                    output_data,
                    split_dim,
                    num_split,
                    kernel_name="split_d"):
    """Split a tensor into `num_split` tensors along one dimension.

    Parameters
    ----------
    input_value: TVM tensor
        input tensor.
    output_data: list or tuple
        the list of output tensor.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        an integer indicating the number of split_d along `split_dim`.
    kernel_name: str
        cce kernel name, default value is "split_d".

    Returns
    -------
    output_shape_list: list
        the list of output shapes.
    output_tensor_list: list
        the list of output tensors, output tensor type is TVM tensor.
    """
    shape = te.lang.cce.util.shape_to_list(input_value.shape)
    size = shape[split_dim] // num_split

    size_splits = [size] * num_split

    output_shape_list, output_tensor_list = te.lang.cce.split_compute_com(
        input_value, split_dim, size_splits)

    return output_shape_list, output_tensor_list


def op_select_format(input_value,
                     output_data,
                     split_dim,
                     num_split,
                     kernel_name="split_d"):
    """Split a tensor into `num_split` tensors along one dimension.

    Parameters
    ----------
    input_value: dict
        the dict of input tensor.
    output_data: list or tuple
        the list of output tensor.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        an integer indicating the number of split_d along `split_dim`.
    kernel_name: str
        cce kernel name.

    Returns
    -------
    None.
    """
    dtype = input_value.get("dtype").lower()
    if dtype == "int8":
        c0_len = 32
    else:
        c0_len = 16
    output_org_shape_list = []
    output_org_format_list = []
    is_support_5hd = True
    support_ori_format = ["NCHW", "NHWC"]
    input_ori_shape = input_value.get("ori_shape")
    input_ori_format = input_value.get("ori_format")
    split_dim = split_dim % len(input_ori_shape)

    for _, output_dict in enumerate(output_data):
        ori_format = output_dict.get("ori_format").upper()
        ori_shape = output_dict.get("ori_shape")
        output_org_shape_list.append(ori_shape)
        output_org_format_list.append(ori_format)

        if ori_format not in support_ori_format or len(ori_shape) != 4:
            is_support_5hd = False
            break

        # when split_d by N,H,W, support NC1HWC0
        if ori_format[split_dim] != "C":
            break

        # when split_d by C, but output size not C0 align donot support NC1HWC0
        if ori_format == "NCHW" and ori_shape[1] % c0_len != 0:
            is_support_5hd = False
            break

        if ori_format == "NHWC" and ori_shape[3] % c0_len != 0:
            is_support_5hd = False
            break

    is_support_nz = False
    if input_ori_format[0] == "N" and split_dim == 0:
        is_support_nz = True

    dtype_base = [
        "float16", "float", "int32", "int8", "int16", "int64", "uint8",
        "uint16", "uint32", "uint64"
    ]
    dtype_5hd = [
        "float16", "float", "int32", "int8", "int16", "uint16", "uint32"
    ]
    dtype_base_out = dtype_base.copy()
    format_base_out = ["ND"] * len(dtype_base)

    if is_support_5hd:
        dtype_base_out = dtype_base_out + dtype_5hd
        format_base_out = format_base_out + ["NC1HWC0"] * len(dtype_5hd)

    if is_support_nz:
        dtype_base_out = dtype_base_out + dtype_base
        format_base_out = format_base_out + ["FRACTAL_NZ"] * len(dtype_base)

    dtype_str = ','.join(dtype_base_out)
    format_str = ','.join(format_base_out)

    input0 = gen_param(
        classify="input0", name="x", datatype=dtype_str, format=format_str)
    output0 = gen_param(
        classify="output0", name="y", datatype=dtype_str, format=format_str)
    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@util.check_input_type(dict, (list, tuple), int, int, str)
def split_d(input_value,
            output_data,
            split_dim,
            num_split,
            kernel_name="split_d"):
    """Split a tensor into `num_split` tensors along one dimension.

    Parameters
    ----------
    input_value: dict
        the dict of input tensor.
    output_data: list or tuple
        the list of output tensor.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        an integer indicating the number of split_d along `split_dim`.
    kernel_name: str
        cce kernel name, default value is "split_d".

    Returns
    -------
    None.
    """
    input_format = input_value.get("format")
    ori_format = input_value.get("ori_format")
    if input_format == "NC1HWC0":
        split_dim = util.axis_transfrom_5d(split_dim, ori_format)

    shape = input_value.get("shape")
    dtype = input_value.get("dtype")
    dtype_lower = dtype.lower()
    check_list = ("int8", "int16", "int32", "int64", "uint8", "uint16",
                  "uint32", "uint64", "float16", "float32")

    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_dtype_rule(dtype_lower, check_list)
    util.check_kernel_name(kernel_name)

    shape_len = len(shape)
    split_dim = util.axis_check(shape_len, split_dim)

    if num_split < 1:
        raise RuntimeError("The num_split (%d) must be greater or equal to %d" %
                           (num_split, 1))
    if shape[split_dim] % num_split != 0:
        raise RuntimeError(
            "The num_split (%d) must be divisible by the length of"
            "split_dim (%d)" % (num_split, shape[split_dim]))

    if num_split == 1:
        copy_only(input_value, input_value, kernel_name)
        return

    split_mov = SplitMov(shape, dtype_lower, split_dim, num_split, None,
                         kernel_name)
    new_shape = split_mov.input_shape
    new_split_dim = split_mov.split_dim
    new_size_splits = split_mov.size_splits
    new_output_shapes = split_mov.output_shapes
    input_size = functools_reduce(lambda x, y: x * y, new_shape)

    if dtype_lower == "float16" and new_split_dim == len(new_shape) - 1 and \
            new_size_splits[0] == 1 and num_split <= 16 \
            and input_size >= TRANSPOSE_SIZE * num_split:
        split_vnc = SplitLastDimVnv(new_shape, dtype_lower, new_output_shapes,
                                    new_split_dim, num_split, kernel_name)
        split_vnc.split_last_dim_vnc_compute()
        return

    if check_use_last_dim_branch(new_shape, dtype_lower, new_split_dim,
                                 num_split, new_size_splits):
        split_last_dim(new_shape, dtype_lower, new_split_dim, num_split,
                       new_size_splits, kernel_name)
        return

    if split_mov.check_whether_use_split_mov():
        split_mov.split_mov_compute()
        return

    data = tvm.placeholder(shape, name="data", dtype=dtype_lower)
    output_shape_list, output_tensor_list = split_d_compute(
        data, output_data, split_dim, num_split, kernel_name)

    sch, build_list = te.lang.cce.split_schedule_com(data, split_dim,
                                                     output_shape_list,
                                                     output_tensor_list)

    with build_config:
        tvm.build(sch, build_list, "cce", name=kernel_name)
