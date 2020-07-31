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

sparse_apply_adagrad
"""
import math
from functools import reduce as functools_reduce

from te import tik
from te import platform as tbe_platform
from te import platform as cce
from topi.cce import util


# pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals
class SparseAdagrad():
    """
       Function: use to store sparse_apply_adagrad base parameters
       Modify : 2019-12-28
    """

    # pylint: disable=too-many-statements
    def __init__(self, var, accum, lr, epsilon, grad, indices, var_out,
                accum_out, update_slots, kernel_name, compute_type):
        """
        Init sparse_apply_adagrad base parameters

        Parameters
        ----------
        var: dict
            data of input
            datatype suports float32
        accum: dict
            data of input
            datatype suports float32
        lr: float
            scalar
        grad: dict
            data of grad
            datatype supports float32
        indices: dict
            data of indices
            datatype supports int32
        var_out: dict
            data of input
        accum_out: dict
            data of input
        kernel_name: str
            the name of the operator
        compute_type: str
            the compute type of sparse_apply_adagrad
        Returns
        -------
        None
        """
        if cce.CceProductParams().cce_product in ("1.1", "1.3"):
            product_name = "mini"
        elif cce.CceProductParams().cce_product == "1.60":
            product_name = "cloud"
        else:
            raise RuntimeError(
                "sparse apply adagrad only support target:cloud_v100/mini_v100")
        self.tik_instance = tik.Tik(tik.Dprofile("v100", product_name), True)
        self.lr = lr
        self.epsilon = epsilon
        self.update_slots = update_slots
        shape_var = var.get("shape")
        shape_var_new = list(shape_var)
        shape_var_new[0] = shape_var_new[0]+1
        shape_var_new = tuple(shape_var_new)
        self.var_shape_new = shape_var_new
        self.var_shape = var.get("shape")
        self.var_dtype = var.get("dtype").lower()
        shape_accum = accum.get("shape")
        shape_accum_new = list(shape_accum)
        shape_accum_new[0] = shape_accum_new[0]+1
        shape_accum_new = tuple(shape_accum_new)
        self.accum_shape_new = shape_accum_new
        self.accum_shape = accum.get("shape")
        self.accum_dtype = accum.get("dtype").lower()
        shape_grad = grad.get("shape")
        shape_grad_new = list(shape_grad)
        shape_grad_new[0] = shape_grad_new[0]+1
        shape_grad_new = tuple(shape_grad_new)
        self.grad_shape_new = shape_grad_new
        self.grad_shape = grad.get("shape")
        self.grad_dtype = grad.get("dtype").lower()
        shape_indices = indices.get("shape")
        shape_indices_new = list(shape_indices)
        shape_indices_new[0] = shape_indices_new[0]+1
        shape_indices_new = tuple(shape_indices_new)
        self.indices_shape_new = shape_indices_new
        self.indices_shape = indices.get("shape")
        self.indices_dtype = indices.get("dtype").lower()
        self.var_ele_num = functools_reduce(lambda x, y: x * y, self.var_shape)
        self.accum_ele_num = functools_reduce(lambda x, y: x * y, self.accum_shape)
        self.indices_num = functools_reduce(lambda x, y: x * y, self.indices_shape)
        self.grad_num = functools_reduce(lambda x, y: x * y, self.grad_shape)
        self.kernel_name = kernel_name
        self.check_param(var_out)
        if len(self.var_shape) > 1:
            self.update_data_num = functools_reduce(lambda x, y: x * y,
                                                    self.var_shape[1:])
        else:
            self.update_data_num = 1
        self.max_indice = self.var_shape[0]

        self.compute_type = compute_type

        self.ub_size_bytes = (
            tbe_platform.cce_conf.get_soc_spec(
                tbe_platform.cce_conf.UB_SIZE) - 8192)
        self.var_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.var_dtype) // 8
        self.indices_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.indices_dtype) // 8
        self.accum_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.accum_dtype) // 8
        self.var_data_each_block = 32 // self.var_dtype_bytes_size
        self.accum_data_each_block = 32 // self.accum_dtype_bytes_size
        self.indices_data_each_block = 32 // self.indices_dtype_bytes_size
        self.indices_ub_number = 0
        self.grad_ub_number = 0

        self.index_loop_num = 0

        self.max_num_one_repeat = 128
        if self.var_dtype in ("float32"):
            self.max_num_one_repeat = 64

        if self.update_data_num < self.var_data_each_block:
            self.block_num = 1
        else:
            ai_core_num = tbe_platform.cce_conf.get_soc_spec(
                tbe_platform.cce_conf.CORE_NUM)
            self.indice_step = math.ceil(self.max_indice / ai_core_num)
            self.block_num = math.ceil(self.max_indice / self.indice_step)

        self.var_gm = self.tik_instance.Tensor(
            self.var_dtype, self.var_shape_new, name="var_gm", scope=tik.scope_gm)
        self.accum_gm = self.tik_instance.Tensor(
            self.accum_dtype, self.accum_shape_new,
            name="accum_gm", scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(
            self.indices_dtype,
            self.indices_shape_new,
            name="indices_gm",
            scope=tik.scope_gm)
        self.grad_gm = self.tik_instance.Tensor(
            self.grad_dtype,
            self.grad_shape_new,
            name="grad_gm",
            scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(
            self.var_dtype, self.var_shape, name="out_gm", scope=tik.scope_gm)
        self.out_accum_gm = self.tik_instance.Tensor(
            self.accum_dtype, self.accum_shape_new,
            name="out_accum_gm", scope=tik.scope_gm)

        self.vconv_dst_dtype = "float16"

        self.init_ub_tensor_para()
        self.var_vconv_ub = None
        self.grad_vconv_ub = None
        self.var_tile_vconv_ub = None
        self.updates_tile_vconv_ub = None

        self.var_ub = None
        self.grad_ub = None
        self.indices_ub = None
        self.var_tile_ub = None
        self.accum_tile_ub = None
        self.updates_tile_ub = None

        self.var_read_index = None
        self.grad_read_index = None
        self.indices_loop_index = None
        self.indices_tmp = None

    def init_ub_tensor_para(self):
        """
        Compute the ub size of tensors

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        updates_size_bytes = self.var_dtype_bytes_size * self.update_data_num
        indices_size_bytes = self.indices_dtype_bytes_size * self.indices_num

        if updates_size_bytes * 4 < self.ub_size_bytes * 0.9:
            self.grad_ub_number = math.ceil(
                self.update_data_num /
                self.var_data_each_block) * self.var_data_each_block
            self.indices_ub_number = (
                self.ub_size_bytes -
                updates_size_bytes * 4) // self.indices_dtype_bytes_size
            self.indices_ub_number = math.ceil(
                self.indices_ub_number /
                self.indices_data_each_block) * self.indices_data_each_block
            if self.indices_num < self.indices_ub_number:
                self.indices_ub_number = math.ceil(
                    self.indices_num /
                    self.indices_data_each_block) * self.indices_data_each_block
        elif indices_size_bytes < self.ub_size_bytes * 0.9:
            self.indices_ub_number = math.ceil(
                self.indices_num /
                self.indices_data_each_block) * self.indices_data_each_block

            self.grad_ub_number = (
                self.ub_size_bytes -
                indices_size_bytes) // 4 // self.var_dtype_bytes_size

            self.grad_ub_number = math.ceil(
                self.grad_ub_number /
                self.var_data_each_block) * self.var_data_each_block
        else:
            self.indices_ub_number = (
                self.ub_size_bytes // self.indices_dtype_bytes_size // 4 //
                self.indices_data_each_block * self.indices_data_each_block)
            self.grad_ub_number = (
                self.indices_ub_number // 4 // self.var_data_each_block *
                self.var_data_each_block)

        last_num = self.update_data_num % self.grad_ub_number
        if (last_num < self.var_data_each_block and
                self.update_data_num > self.var_data_each_block):
            self.grad_ub_number -= self.var_data_each_block

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

        self.var_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.grad_ub_number,),
            name="var_ub",
            scope=tik.scope_ubuf)
        self.accum_ub = self.tik_instance.Tensor(
            self.accum_dtype, (self.grad_ub_number,),
            name="accum_ub",
            scope=tik.scope_ubuf)
        self.grad_ub = self.tik_instance.Tensor(
            self.grad_dtype, (self.grad_ub_number,),
            name="grad_ub",
            scope=tik.scope_ubuf)
        self.temp_ub = self.tik_instance.Tensor(
            self.grad_dtype, (self.grad_ub_number,),
            name="temp_ub",
            scope=tik.scope_ubuf)
        self.indices_ub = self.tik_instance.Tensor(
            self.indices_dtype, (self.indices_ub_number,),
            name="indices_ub",
            scope=tik.scope_ubuf)
        self.lr_ub = self.tik_instance.Scalar(self.var_dtype)
        self.lr_ub.set_as(self.lr)

        self.var_tile_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.var_data_each_block,),
            name="var_tile_ub",
            scope=tik.scope_ubuf)
        self.accum_tile_ub = self.tik_instance.Tensor(
            self.accum_dtype, (self.var_data_each_block,),
            name="accum_tile_ub",
            scope=tik.scope_ubuf)
        self.updates_tile_ub = self.tik_instance.Tensor(
            self.grad_dtype, (self.var_data_each_block,),
            name="updates_tile_ub",
            scope=tik.scope_ubuf)
        self.temp_tile_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.var_data_each_block,),
            name="temp_tile_ub",
            scope=tik.scope_ubuf)

        self.var_read_index = self.tik_instance.Scalar("int32")
        self.var_read_index.set_as(0)

        self.accum_read_index = self.tik_instance.Scalar("int32")
        self.accum_read_index.set_as(0)

        self.grad_read_index = self.tik_instance.Scalar("int32")
        self.grad_read_index.set_as(0)

        self.indices_loop_index = self.tik_instance.Scalar("int32")
        self.indices_loop_index.set_as(0)

        self.indices_tmp = self.tik_instance.Scalar("int32")
        self.indices_tmp.set_as(0)

    def get_var_read_index(self, indices_ub_index):
        """
        Calculate the index of the read var

        Parameters
        ----------
        indices_ub_index: int32
            the index of the currently traversed indices in UB

        Returns
        -------
        None
        """

        self.var_read_index.set_as(self.indices_ub[indices_ub_index])

    def get_grad_read_index(self, indices_ub_index):
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
        self.grad_read_index.set_as(indices_ub_index * self.update_data_num)

    def updates_the_var(self, indices_in_index, indice_num):
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
        indices_burst_len = math.ceil(indice_num / self.indices_data_each_block)
        if self.indices_num == 1:
            self.tik_instance.data_move(self.indices_ub, self.indices_gm, 0, 1,
                                        indices_burst_len, 0, 0)
        else:
            self.tik_instance.data_move(self.indices_ub,
                                        self.indices_gm[indices_in_index], 0, 1,
                                        indices_burst_len, 0, 0)
        indice_loop_num = indice_num

        with self.tik_instance.for_range(0,
                                         indice_loop_num) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)
            if self.block_num > 1:
                with self.tik_instance.if_scope(
                        self.indices_loop_index *
                        self.indice_step <= self.var_read_index):
                    with self.tik_instance.if_scope(
                            (self.indices_loop_index + 1) *
                            self.indice_step > self.var_read_index):
                        self.get_grad_read_index(indices_ub_index +
                                                 indices_in_index)
                        self.var_read_index.set_as(self.var_read_index *
                                                   self.update_data_num)
                        self.calc_updates()
            else:
                self.get_grad_read_index(indices_ub_index + indices_in_index)
                self.var_read_index.set_as(self.var_read_index *
                                           self.update_data_num)
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
        updates_loop = self.update_data_num // self.grad_ub_number
        if updates_loop > 0:
            with self.tik_instance.for_range(0, updates_loop) as loop_index:
                self.calc_updates_small(loop_index * self.grad_ub_number,
                                        self.grad_ub_number)

        last_num = self.update_data_num % self.grad_ub_number
        if last_num > 0:
            self.calc_updates_small(updates_loop * self.grad_ub_number,
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
        updates_burst_len = math.ceil(element_num / self.var_data_each_block)
        self.tik_instance.data_move(
            self.var_ub, self.var_gm[self.var_read_index + read_index_offset],
            0, 1, updates_burst_len, 0, 0)


        self.tik_instance.data_move(
            self.accum_ub, self.accum_gm[self.var_read_index + read_index_offset],
            0, 1, updates_burst_len, 0, 0)

        self.tik_instance.data_move(
            self.grad_ub,
            self.grad_gm[self.grad_read_index + read_index_offset], 0, 1,
            updates_burst_len, 0, 0)

        tile_ele_num = element_num % self.var_data_each_block
        align_offset = 0
        if (tile_ele_num != 0 and
                self.update_data_num > self.var_data_each_block):
            align_ele_num = (
                element_num // self.var_data_each_block *
                self.var_data_each_block)
            align_offset = (
                read_index_offset + align_ele_num -
                (self.var_data_each_block - tile_ele_num))
            self.tik_instance.data_move(
                self.var_tile_ub,
                self.var_gm[self.var_read_index + align_offset], 0, 1, 1, 0, 0)

            self.tik_instance.data_move(
                self.accum_tile_ub,
                self.accum_gm[self.var_read_index + align_offset], 0, 1, 1,
                0, 0)

            self.tik_instance.data_move(
                self.updates_tile_ub,
                self.grad_gm[self.grad_read_index + align_offset], 0, 1,
                1, 0, 0)

        compute_loop = element_num // self.max_num_one_repeat // 255

        if compute_loop > 0:
            with self.tik_instance.for_range(0, compute_loop) as index:
                index_offset = index * self.max_num_one_repeat * 255
                self.calc_process(self.max_num_one_repeat, index_offset,
                                  index_offset, index_offset, 255, False)
        last_loop = element_num % (self.max_num_one_repeat *
                                   255) // self.max_num_one_repeat

        if last_loop > 0:
            index_offset = compute_loop * self.max_num_one_repeat * 255
            self.calc_process(self.max_num_one_repeat, index_offset,
                              index_offset, index_offset, last_loop, False)

        compute_mask = element_num % self.max_num_one_repeat
        if compute_mask > 0:
            index_offset = (
                element_num // self.max_num_one_repeat *
                self.max_num_one_repeat)
            if (tile_ele_num == 0 or
                    self.update_data_num < self.var_data_each_block):
                self.calc_process(compute_mask, index_offset, index_offset,
                                  index_offset, 1, False)

                self.tik_instance.data_move(
                    self.out_gm[self.var_read_index + read_index_offset],
                    self.var_ub, 0, 1, updates_burst_len, 0, 0)
                self.tik_instance.data_move(
                    self.out_accum_gm[self.var_read_index + read_index_offset],
                    self.accum_ub, 0, 1, updates_burst_len, 0, 0)
            else:
                self.calc_process(self.var_data_each_block, 0, 0, 0, 1, True)
                self.tik_instance.data_move(
                    self.out_gm[self.var_read_index + align_offset],
                    self.var_tile_ub, 0, 1, 1, 0, 0)
                self.tik_instance.data_move(
                    self.out_accum_gm[self.var_read_index + align_offset],
                    self.accum_tile_ub, 0, 1, 1, 0, 0)
                self.calc_process(compute_mask, index_offset, index_offset,
                                  index_offset, 1, False)
                self.tik_instance.data_move(
                    self.out_gm[self.var_read_index + read_index_offset],
                    self.var_ub, 0, 1, updates_burst_len - 1, 0, 0)
                self.tik_instance.data_move(
                    self.out_accum_gm[self.var_read_index + read_index_offset],
                    self.accum_ub, 0, 1, updates_burst_len - 1, 0, 0)
        else:
            self.tik_instance.data_move(
                self.out_gm[self.var_read_index + read_index_offset],
                self.var_ub, 0, 1, updates_burst_len, 0, 0)
            self.tik_instance.data_move(
                self.out_accum_gm[self.var_read_index + read_index_offset],
                self.accum_ub, 0, 1, updates_burst_len, 0, 0)


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
            determine whether the currently calculated data is the tail of var
            and updates

        Returns
        -------
        None
        """


        if is_tile:
            compute_repeat_strid = (
                self.max_num_one_repeat // self.var_data_each_block)
            src_accum_ub = self.accum_tile_ub
            src_grad_ub = self.updates_tile_ub
            dst_var_ub = self.var_tile_ub
            mask = self.var_data_each_block
            src_temp_ub = self.temp_tile_ub
        else:
            compute_repeat_strid = (
                self.max_num_one_repeat // self.var_data_each_block)
            src_accum_ub = self.accum_ub[src_addr1]
            src_grad_ub = self.grad_ub[src_addr2]
            dst_var_ub = self.var_ub[dest_addr]
            src_temp_ub = self.temp_ub[dest_addr]

        if self.compute_type == "sparse_apply_adagrad_d":
            if self.update_slots:
                self.tik_instance.vmul(mask, src_temp_ub, src_grad_ub, src_grad_ub,
                                       repeat_times,
                                       1, 1, 1, compute_repeat_strid,
                                       compute_repeat_strid, compute_repeat_strid)
                self.tik_instance.vadd(mask, src_accum_ub, src_accum_ub,
                                       src_temp_ub, repeat_times,
                                       1, 1, 1, compute_repeat_strid,
                                       compute_repeat_strid, compute_repeat_strid)

                self.tik_instance.vsqrt(mask, src_temp_ub, src_accum_ub,
                                        repeat_times, 1, 1, compute_repeat_strid,
                                        compute_repeat_strid)

                if self.epsilon != 0:
                    self.tik_instance.vadds(mask, src_temp_ub, src_temp_ub,
                                            self.epsilon, repeat_times, 1, 1,
                                            compute_repeat_strid,
                                            compute_repeat_strid)

                self.tik_instance.vmuls(mask, src_grad_ub, src_grad_ub, self.lr_ub,
                                        repeat_times, 1, 1, compute_repeat_strid,
                                        compute_repeat_strid)
                self.tik_instance.vdiv(mask, src_temp_ub, src_grad_ub, src_temp_ub,
                                       repeat_times,
                                       1, 1, 1, compute_repeat_strid,
                                       compute_repeat_strid, compute_repeat_strid)
                self.tik_instance.vsub(mask, dst_var_ub, dst_var_ub, src_temp_ub,
                                       repeat_times, 1, 1, 1, compute_repeat_strid,
                                       compute_repeat_strid, compute_repeat_strid)
            else:
                self.tik_instance.vsqrt(mask, src_temp_ub, src_accum_ub,
                                        repeat_times, 1, 1, compute_repeat_strid,
                                        compute_repeat_strid)

                if self.epsilon != 0:
                    self.tik_instance.vadds(mask, src_temp_ub, src_temp_ub,
                                            self.epsilon, repeat_times, 1, 1,
                                            compute_repeat_strid,
                                            compute_repeat_strid)

                self.tik_instance.vmuls(mask, src_grad_ub, src_grad_ub, self.lr_ub,
                                        repeat_times, 1, 1, compute_repeat_strid,
                                        compute_repeat_strid)
                self.tik_instance.vdiv(mask, src_temp_ub, src_grad_ub, src_temp_ub,
                                       repeat_times,
                                       1, 1, 1, compute_repeat_strid,
                                       compute_repeat_strid, compute_repeat_strid)
                self.tik_instance.vsub(mask, dst_var_ub, dst_var_ub, src_temp_ub,
                                       repeat_times, 1, 1, 1, compute_repeat_strid,
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
                self.updates_the_var(
                    indices_loop_index * self.indices_ub_number,
                    self.indices_ub_number)

        indices_last_num = self.indices_num % self.indices_ub_number
        if indices_last_num > 0:
            self.updates_the_var(indices_loop_num * self.indices_ub_number,
                                 indices_last_num)

    def check_param(self, var_out):
        """
        Check parameter

        Parameters
        ----------
        var_out: dict
            data of input
            datatype suports float32

        Returns
        -------
        None
        """
        var_out_shape = var_out.get("shape")
        var_out_dtype = var_out.get("dtype").lower()

        util.check_kernel_name(self.kernel_name)
        util.check_shape_rule(self.var_shape)
        util.check_shape_rule(self.indices_shape)
        util.check_shape_rule(self.grad_shape)
        util.check_shape_rule(var_out_shape)

        util.check_tensor_shape_size(self.var_shape)
        util.check_tensor_shape_size(self.indices_shape)
        util.check_tensor_shape_size(self.grad_shape)
        util.check_tensor_shape_size(var_out_shape)

        check_list_var = ("float32")
        check_list_indices = ("int32")
        util.check_dtype_rule(self.var_dtype, check_list_var)
        util.check_dtype_rule(self.indices_dtype, check_list_indices)
        util.check_dtype_rule(self.grad_dtype, check_list_var)
        util.check_dtype_rule(var_out_dtype, check_list_var)

        if var_out_shape != self.var_shape:
            raise RuntimeError(
                "var_out's shape[%s] must be the same as var's shape[%s]" %
                (var_out_shape, self.var_shape))

        if self.accum_shape != self.var_shape:
            raise RuntimeError(
                "accum's shape[%s] must be the same as var's shape[%s]" %
                (self.accum_shape, self.var_shape))

        if self.grad_shape[1:] != self.var_shape[1:]:
            raise RuntimeError(
                "grad's shape[%s] must be the same as var's shape[%s] except"
                "first dimension" %
                (self.grad_shape, self.var_shape))

        if len(self.indices_shape) != 1:
            raise RuntimeError(
                "indices must be one-dimensioal")

        if (self.grad_dtype != self.var_dtype or
                var_out_dtype != self.var_dtype or
                self.accum_dtype != self.var_dtype):
            raise RuntimeError(
                "var.type,accum.type and grad.type must be same")

        updates_true_shape = self.indices_shape + self.grad_shape[1:]

        if self.grad_shape != updates_true_shape:
            raise RuntimeError("grad must be the same shape as indices in"
                               "first dimension")

    def sparse_apply_adagrad_operator(self):
        """
        SparseAdagrad operation

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
                self.init_ub_tensor()
                self.indices_loop_index.set_as(indices_loop_index)
                self.traversing_indices()
        else:
            self.init_ub_tensor()
            self.traversing_indices()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=(self.var_gm, self.accum_gm, self.grad_gm, self.indices_gm),
            outputs=(self.out_gm, self.out_accum_gm),
            enable_l2=False)

        return self.tik_instance
