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

scatter_non_aliasing_add
"""
import math
from functools import reduce as functools_reduce
from te import tik
import te.platform.cce_params as cce_params
from te import platform as tbe_platform
from topi.cce import util

#General limitation of the size for input shape
SHAPE_SIZE_LIMIT = 2**30
#block length in number
BLOCK_LENGTH = 32
# since we have inputs, indices, updates,
# then left 0.1*ubsize is for indices or inputs and updates
UB_SIZE_RATIO = 0.9
#max core number
MAX_CORE_NUMBER = 65536


class ScatterNonAliasingAdd(object):
    """
       Function: use to store scatter base parameters
       Modify : 2019-11-21
    """
    def __init__(self, inputs, indices, updates, outputs, kernel_name):
        """
        Init scatter base parameters

        Parameters
        ----------
        inputs: dict
            data of input
            datatype suports float32,float16,int32
        indices: dict
            data of indices
            datatype supports int32,int64
        updates: dict
            data of updates
            datatype supports float32,float16,int32
        outputs: dict
            data of outputs
        kernel_name: str
            the name of the operator
        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik()
        self.device_core_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        self.unused_ub_sizes = 8192  # this size is keep for scalar unit
        self.inputs_shape = inputs.get("shape")
        self.inputs_dtype = inputs.get("dtype").lower()
        self.indices_shape = indices.get("shape")
        self.indices_dtype = indices.get("dtype").lower()
        self.updates_shape = updates.get("shape")
        self.updates_dtype = updates.get("dtype").lower()
        self.inputs_ele_num = functools_reduce(lambda x, y: x*y,
                                               self.inputs_shape)
        self.indices_num = functools_reduce(lambda x, y: x*y,
                                            self.indices_shape)
        self.kernel_name = kernel_name
        self.check_param(outputs)
        if self.indices_shape[-1] == len(self.inputs_shape):
            self.update_data_num = 1
        else:
            self.update_data_num = functools_reduce(
                lambda x, y: x*y, self.inputs_shape[self.indices_shape[-1]:])
        self.max_indice = functools_reduce(
            lambda x, y: x*y, self.inputs_shape[0:self.indices_shape[-1]])
        self.block_number = cce_params.VECTOR_INST_BLOCK_NUM
        self.ub_size_bytes = (tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.UB_SIZE) - self.unused_ub_sizes)
        self.inputs_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.inputs_dtype) // 8
        self.indices_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.indices_dtype) // 8
        self.inputs_data_each_block = BLOCK_LENGTH // \
                                      self.inputs_dtype_bytes_size
        self.indices_data_each_block = BLOCK_LENGTH // \
                                       self.indices_dtype_bytes_size
        self.indices_ub_number = 0
        self.updates_ub_number = 0
        self.index_loop_num = 0

        self.max_num_one_repeat = cce_params.ELEMENTS_VECTOR_OP_FP16
        if self.inputs_dtype in ("float32", "int32"):
            self.max_num_one_repeat = cce_params.ELEMENTS_VECTOR_OP_FP16 // 2

        self.gen_block_num()

        self.inputs_gm = self.tik_instance.Tensor(self.inputs_dtype,
                                                  self.inputs_shape,
                                                  name="inputs_gm",
                                                  scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(self.indices_dtype,
                                                   self.indices_shape,
                                                   name="indices_gm",
                                                   scope=tik.scope_gm)
        self.updates_gm = self.tik_instance.Tensor(self.updates_dtype,
                                                   self.updates_shape,
                                                   name="updates_gm",
                                                   scope=tik.scope_gm)
        self.outputs_gm = self.tik_instance.Tensor(self.inputs_dtype,
                                                   self.inputs_shape,
                                                   name="outputs_gm",
                                                   scope=tik.scope_gm)

    def gen_block_num(self):
        if self.update_data_num < self.inputs_data_each_block or len(
                self.inputs_shape) == 1:
            self.block_num = 1
        else:
            self.indice_step = math.ceil(self.max_indice / BLOCK_LENGTH)
            if self.max_indice % BLOCK_LENGTH != 0:
                self.block_num = self.max_indice
            else:
                self.block_num = math.ceil(self.max_indice / self.indice_step)
        self.update_num_block = 1
        if self.block_num > self.device_core_num:
            for i in range(self.device_core_num):
                block_number_step = self.device_core_num - i
                if self.max_indice % block_number_step == 0:
                    self.update_num_block = block_number_step
                    break
        if self.update_num_block > 1:
            self.block_num = self.update_num_block
        elif self.max_indice > MAX_CORE_NUMBER:
            self.block_num = 1

    def copy_input_to_output(self, indices_loop_index):
        """
        copy inputs to outputs by multiple core

        Parameters
        ----------
        indices_loop_index:
        index of the core.

        Returns
        -------
        None
        """
        ub_size_bytes = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.UB_SIZE) - self.unused_ub_sizes
        input_to_ub_number = ub_size_bytes // self.inputs_dtype_bytes_size
        input_to_ub_number = math.ceil(
            input_to_ub_number /
            self.inputs_data_each_block)*self.inputs_data_each_block
        with self.tik_instance.new_stmt_scope():
            inputs_ub = self.tik_instance.Tensor(self.inputs_dtype,
                                                 (input_to_ub_number, ),
                                                 name="inputs_ub_copy",
                                                 scope=tik.scope_ubuf)

            def _do_copy_input_to_output(inputs_ub, indices_in_index,
                                         indice_num):
                indices_burst_len = math.floor(indice_num /
                                               self.inputs_data_each_block)
                if indices_burst_len == 0:
                    indices_burst_len = 1
                self.tik_instance.data_move(inputs_ub,
                                            self.inputs_gm[indices_in_index],
                                            0, 1, indices_burst_len, 0, 0)
                self.tik_instance.data_move(self.outputs_gm[indices_in_index],
                                            inputs_ub, 0, 1, indices_burst_len,
                                            0, 0)
                if self.inputs_ele_num < self.inputs_data_each_block:
                    tile_ele_num = 0
                else:
                    tile_ele_num = indice_num % self.inputs_data_each_block
                align_offset = 0
                if tile_ele_num != 0:
                    align_ele_num = (indice_num //
                                     self.inputs_data_each_block*
                                     self.inputs_data_each_block)
                    align_offset = (
                            indices_in_index + align_ele_num -
                            (self.inputs_data_each_block - tile_ele_num))
                    self.tik_instance.data_move(inputs_ub,
                                                self.inputs_gm[align_offset],
                                                0, 1, 1, 0, 0)
                    self.tik_instance.data_move(self.outputs_gm[align_offset],
                                                inputs_ub, 0, 1, 1, 0, 0)

            input_loop_num = self.inputs_ele_num // \
                             self.block_num // input_to_ub_number

            if input_loop_num > 0:
                with self.tik_instance.for_range(
                        0, input_loop_num) as input_loop_index:
                    _do_copy_input_to_output(
                        inputs_ub, indices_loop_index*self.inputs_ele_num //
                                   self.block_num + input_loop_index*input_to_ub_number,
                        input_to_ub_number)

            input_last_num = self.inputs_ele_num // self.block_num \
                             % input_to_ub_number
            if input_last_num > 0:
                _do_copy_input_to_output(
                    inputs_ub,
                    indices_loop_index*self.inputs_ele_num // self.block_num
                    + input_loop_num*input_to_ub_number, input_last_num)

    def init_ub_tensor(self):
        """
        Compute the ub size of tensors

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.inputs_read_index = self.tik_instance.Scalar(
            "int32", name="inputs_read_index")
        self.inputs_read_index.set_as(0)

        self.updates_read_index = self.tik_instance.Scalar(
            "int32", name="updates_read_index")
        self.updates_read_index.set_as(0)

        self.indices_loop_index = self.tik_instance.Scalar(
            "int32", name="indices_loop_index")
        self.indices_loop_index.set_as(0)

        self.indices_tmp = self.tik_instance.Scalar("int32",
                                                    name="indices_tmp")
        self.indices_tmp.set_as(0)
        updates_size_bytes = self.inputs_dtype_bytes_size*self.update_data_num
        indices_size_bytes = self.indices_dtype_bytes_size*self.indices_num
        if updates_size_bytes*2 < self.ub_size_bytes*UB_SIZE_RATIO:
            self.updates_ub_number = math.ceil(
                self.update_data_num /
                self.inputs_data_each_block)*self.inputs_data_each_block

            self.indices_ub_number = (self.ub_size_bytes - updates_size_bytes*
                                      2) // self.indices_dtype_bytes_size
            self.indices_ub_number = math.ceil(
                self.indices_ub_number /
                self.indices_data_each_block)*self.indices_data_each_block
        elif indices_size_bytes < self.ub_size_bytes*UB_SIZE_RATIO:
            self.indices_ub_number = math.ceil(
                self.indices_num /
                self.indices_data_each_block)*self.indices_data_each_block

            self.updates_ub_number = (self.ub_size_bytes - indices_size_bytes
                                      ) // 2 // self.inputs_dtype_bytes_size

            self.updates_ub_number = math.ceil(
                self.updates_ub_number /
                self.inputs_data_each_block)*self.inputs_data_each_block
        else:
            self.indices_ub_number = (self.ub_size_bytes //
                                      self.indices_dtype_bytes_size // 2 //
                                      BLOCK_LENGTH*BLOCK_LENGTH)
            self.updates_ub_number = self.indices_ub_number // 2
        self.inputs_ub = self.tik_instance.Tensor(self.inputs_dtype,
                                                  (self.updates_ub_number, ),
                                                  name="inputs_ub",
                                                  scope=tik.scope_ubuf)
        self.updates_ub = self.tik_instance.Tensor(self.updates_dtype,
                                                   (self.updates_ub_number, ),
                                                   name="updates_ub",
                                                   scope=tik.scope_ubuf)
        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype,
                                                   (self.indices_ub_number, ),
                                                   name="indices_ub",
                                                   scope=tik.scope_ubuf)

        self.inputs_tile_ub = self.tik_instance.Tensor(
            self.inputs_dtype, (self.inputs_data_each_block, ),
            name="inputs_tile_ub",
            scope=tik.scope_ubuf)
        self.updates_tile_ub = self.tik_instance.Tensor(
            self.updates_dtype, (self.inputs_data_each_block, ),
            name="updates_tile_ub",
            scope=tik.scope_ubuf)

    def get_inputs_read_index(self, indices_ub_index):
        """
        Calculate the index of the read inputs

        Parameters
        ----------
        indices_ub_index: int32,int64
            the index of the currently traversed indices in UB

        Returns
        -------
        None
        """
        indices_ub_index = indices_ub_index*self.indices_shape[-1]
        self.inputs_read_index.set_as(0)
        for i in range(0, self.indices_shape[-1] - 1):
            self.indices_tmp.set_as(self.indices_ub[indices_ub_index + i])
            self.inputs_read_index.set_as(
                self.inputs_read_index + self.indices_tmp*
                functools_reduce(lambda x, y: x*y, self.inputs_shape[i + 1:])
            )
        self.indices_tmp.set_as(self.indices_ub[indices_ub_index +
                                                self.indices_shape[-1] - 1])
        if self.indices_shape[-1] == len(self.inputs_shape):
            self.inputs_read_index.set_as(self.inputs_read_index +
                                          self.indices_tmp)
        else:
            self.inputs_read_index.set_as(
                self.inputs_read_index + self.indices_tmp*
                functools_reduce(lambda x, y: x*y,
                                 self.inputs_shape[self.indices_shape[-1]:])
            )

    def get_updates_read_index(self, indices_ub_index):
        """
        Calculate the index of the read updates

        Parameters
        ----------
        indices_ub_index:int32
            the index of the currently traversed indices in UB

        Returns
        -------
        None
        """
        self.updates_read_index.set_as(indices_ub_index*self.update_data_num)

    def updates_the_inputs(self, indices_in_index, indice_num):
        """
        Update the update fragment corresponding to the index

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        indices_burst_len = math.ceil(indice_num /
                                      self.indices_data_each_block)
        self.tik_instance.data_move(self.indices_ub,
                                    self.indices_gm[indices_in_index], 0, 1,
                                    indices_burst_len, 0, 0)
        indice_loop_num = indice_num // self.indices_shape[-1]
        with self.tik_instance.for_range(0,
                                         indice_loop_num) as indices_ub_index:
            self.get_inputs_read_index(indices_ub_index)
            if self.block_num > 1:
                ele_every_block = math.ceil(self.inputs_ele_num /
                                            self.block_num)
                with self.tik_instance.if_scope(
                        self.indices_loop_index*
                        ele_every_block <= self.inputs_read_index):
                    with self.tik_instance.if_scope(
                            (self.indices_loop_index + 1)*
                            ele_every_block > self.inputs_read_index):
                        self.get_updates_read_index(indices_ub_index +
                                                    indices_in_index)
                        self.inputs_read_index.set_as(self.inputs_read_index)
                        self.calc_updates()
            else:
                self.get_updates_read_index(indices_ub_index)
                self.inputs_read_index.set_as(self.inputs_read_index)
                self.calc_updates()

    def calc_updates(self):
        """
        Calculate updates fragment

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        updates_loop = self.update_data_num // self.updates_ub_number
        if updates_loop > 0:
            with self.tik_instance.for_range(0, updates_loop) as loop_index:
                self.calc_updates_small(loop_index*self.updates_ub_number,
                                        self.updates_ub_number)

        last_num = self.update_data_num % self.updates_ub_number
        if last_num > 0:
            self.calc_updates_small(updates_loop*self.updates_ub_number,
                                    last_num)

    def calc_updates_small(self, read_index_offset, element_num):
        """
        Transfer update to UB and calculate

        Parameters
        ----------
        read_index_offset: int32
            the offset used to read the updates fragment
        element_num:
            the number of elements in the slice of updates

        Returns
        -------
        None
        """
        updates_burst_len = element_num // self.inputs_data_each_block
        if updates_burst_len == 0:
            updates_burst_len = 1
        self.tik_instance.data_move(
            self.inputs_ub,
            self.outputs_gm[self.inputs_read_index + read_index_offset], 0, 1,
            updates_burst_len, 0, 0)

        self.tik_instance.data_move(
            self.updates_ub,
            self.updates_gm[self.updates_read_index + read_index_offset], 0, 1,
            updates_burst_len, 0, 0)

        if self.update_data_num < self.indices_data_each_block:
            tile_ele_num = 0
        else:
            tile_ele_num = element_num % self.inputs_data_each_block
        align_offset = 0
        if tile_ele_num != 0:
            align_ele_num = (element_num // self.inputs_data_each_block*
                             self.inputs_data_each_block)
            align_offset = (read_index_offset + align_ele_num -
                            (self.inputs_data_each_block - tile_ele_num))
            self.tik_instance.data_move(
                self.inputs_tile_ub,
                self.outputs_gm[self.inputs_read_index + align_offset], 0, 1,
                1, 0, 0)

            self.tik_instance.data_move(
                self.updates_tile_ub,
                self.updates_gm[self.updates_read_index + align_offset], 0, 1,
                1, 0, 0)
        max_repeat_times = cce_params.VECTOR_INST_MAX_REPEAT_TIMES
        compute_loop = element_num // self.max_num_one_repeat // \
                       max_repeat_times

        if compute_loop > 0:
            with self.tik_instance.for_range(0, compute_loop) as index:
                index_offset = index*self.max_num_one_repeat*max_repeat_times
                self.calc_process(self.max_num_one_repeat, index_offset,
                                  index_offset, index_offset, max_repeat_times,
                                  False)
        last_loop = element_num % (self.max_num_one_repeat*
                                   max_repeat_times) // self.max_num_one_repeat

        if last_loop > 0:
            index_offset = compute_loop*self.max_num_one_repeat*max_repeat_times
            self.calc_process(self.max_num_one_repeat, index_offset,
                              index_offset, index_offset, last_loop, False)

        compute_mask = element_num % self.max_num_one_repeat
        if compute_mask > 0:
            index_offset = (element_num // self.max_num_one_repeat*
                            self.max_num_one_repeat)
            if (tile_ele_num == 0
                    or self.update_data_num < self.inputs_data_each_block):
                self.calc_process(compute_mask, index_offset, index_offset,
                                  index_offset, 1, False)

                self.tik_instance.data_move(
                    self.outputs_gm[self.inputs_read_index +
                                    read_index_offset], self.inputs_ub, 0, 1,
                    updates_burst_len, 0, 0)
            else:
                self.calc_process(compute_mask, index_offset, index_offset,
                                  index_offset, 1, False)
                self.calc_process(self.inputs_data_each_block, 0, 0, 0, 1,
                                  True)
                self.tik_instance.data_move(
                    self.outputs_gm[self.inputs_read_index +
                                    read_index_offset], self.inputs_ub, 0,
                    1, updates_burst_len, 0, 0)
                self.tik_instance.data_move(
                    self.outputs_gm[self.inputs_read_index + align_offset],
                    self.inputs_tile_ub, 0, 1, 1, 0, 0)
        else:
            self.tik_instance.data_move(
                self.outputs_gm[self.inputs_read_index + read_index_offset],
                self.inputs_ub, 0, 1, updates_burst_len, 0, 0)

    def calc_process(self, mask, dest_addr, src_addr1, src_addr2, repeat_times,
                     is_tile):
        """
        Execute the corresponding calculation instruction

        Parameters
        ----------
        mask: int
            the mask of instruction
        dest_addr: int
            testination address offset
        src_addr1: int
            src1 address offset
        src_addr2: int
            src2 address offset
        repeat_times: int
            the repeat times of instruction
        is_tile: bool
            determine whether the currently calculated data is the
            tail of inputs and updates

        Returns
        -------
        None
        """

        if is_tile:
            compute_repeat_strid = (self.max_num_one_repeat //
                                    self.inputs_data_each_block)
            src1_ub = self.inputs_tile_ub
            src2_ub = self.updates_tile_ub
            dst_ub = self.inputs_tile_ub
            mask = self.inputs_data_each_block
        else:
            compute_repeat_strid = (self.max_num_one_repeat //
                                    self.inputs_data_each_block)
            src1_ub = self.inputs_ub[src_addr1]
            src2_ub = self.updates_ub[src_addr2]
            dst_ub = self.inputs_ub[dest_addr]

        self.tik_instance.vadd(mask, dst_ub, src1_ub, src2_ub, repeat_times, 1,
                               1, 1, compute_repeat_strid,
                               compute_repeat_strid, compute_repeat_strid)

    def traversing_indices(self):
        """
        Traversing the index in the indices

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        indices_loop_num = self.indices_num // self.indices_ub_number

        if indices_loop_num > 0:
            with self.tik_instance.for_range(
                    0, indices_loop_num) as indices_loop_index:
                self.updates_the_inputs(
                    indices_loop_index*self.indices_ub_number,
                    self.indices_ub_number)

        indices_last_num = self.indices_num % self.indices_ub_number
        if indices_last_num > 0:
            self.updates_the_inputs(indices_loop_num*self.indices_ub_number,
                                    indices_last_num)

    def check_param(self, outputs):
        """
        Check parameter

        Parameters
        ----------
        outputs: dict
            data of input
            datatype suports float32,float16,int32,int8,uint8

        Returns
        -------
        None
        """
        outputs_shape = outputs.get("shape")
        outputs_dtype = outputs.get("dtype").lower()

        util.check_kernel_name(self.kernel_name)
        util.check_shape_rule(self.inputs_shape)
        util.check_shape_rule(self.indices_shape)
        util.check_shape_rule(self.updates_shape)
        util.check_shape_rule(outputs_shape)

        util.check_shape_size(self.inputs_shape, SHAPE_SIZE_LIMIT)
        util.check_shape_size(self.indices_shape, SHAPE_SIZE_LIMIT)
        util.check_shape_size(self.updates_shape, SHAPE_SIZE_LIMIT)
        util.check_shape_size(outputs_shape, SHAPE_SIZE_LIMIT)

        check_list_inputs = ("float16", "float32", "int32")
        check_list_indices = ("int32", "int64")
        util.check_dtype_rule(self.inputs_dtype, check_list_inputs)
        util.check_dtype_rule(self.indices_dtype, check_list_indices)
        util.check_dtype_rule(self.updates_dtype, check_list_inputs)
        util.check_dtype_rule(outputs_dtype, check_list_inputs)
        add_support = tbe_platform.cce_conf.api_check_support(
            "tik.vadd", "float32")
        if self.inputs_dtype == "float32" and not add_support:
            raise RuntimeError(
                "inputs_dtype only support float16 while inputs_dtype is float32")

        if (self.updates_dtype != self.inputs_dtype
                or outputs_dtype != self.inputs_dtype):
            raise RuntimeError(
                "updates's datatype and outputs's datatype must be the"
                "same as inputs's datatype.")

        k = self.indices_shape[-1]
        updates_true_shape = self.indices_shape[:-1] + self.inputs_shape[k:]
        if k > len(self.inputs_shape):
            raise RuntimeError(
                "indices_shape[-1][%d] can not be large than inputs's rank[%d]"
                % (k, len(self.inputs_shape)))

        if self.updates_shape != updates_true_shape:
            raise RuntimeError("updates's shape is not supported.")

    def scatter_operator(self):
        """
        Scatter operation

        Parameters
        ----------
        None

        Returns:
        ----------
        tik_instance: tik instance
        """
        if self.block_num > 1:
            with self.tik_instance.for_range(
                    0, self.block_num,
                    block_num=self.block_num) as indices_loop_index:
                self.copy_input_to_output(indices_loop_index)
                self.init_ub_tensor()
                self.indices_loop_index.set_as(indices_loop_index)

                self.traversing_indices()
        else:
            self.copy_input_to_output(0)
            self.init_ub_tensor()
            self.traversing_indices()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.inputs_gm, self.indices_gm,
                                           self.updates_gm),
                                   outputs=(self.outputs_gm),
                                   enable_l2=False)

        return self.tik_instance


def scatter_non_aliasing_add(inputs,
                             indices,
                             updates,
                             outpus,
                             kernel_name="ScatterNonAliasingAdd"):
    """
    Generate scatter_non_aliasing_add operator use scatter_non_aliasing_add

    Parameters
    ----------n
    input_x: dict
        data of input.
        source data type, support "int32", "float16", "float32"
    output_y: dict
        data of output.
    axis: list
        the axis list for reverse
    kernel_name: str
        kernel name, default value is "ScatterNonAliasingAdd"

    Returns:
    tik instance
    """
    scatterNonAliasingAdd = ScatterNonAliasingAdd(inputs, indices, updates,
                                                  outpus, kernel_name)

    tik_instance = scatterNonAliasingAdd.scatter_operator()

    return tik_instance
