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

sparse_apply_ftrl_common
"""

import math
from functools import reduce as functools_reduce
from te import tik
from te import platform as tbe_platform
from topi.cce import util


# pylint: disable=too-many-arguments,unused-argument,locally-disabled
# pylint: disable=too-many-branches,too-many-instance-attributes,too-many-locals
# pylint: disable=attribute-defined-outside-init,invalid-name,,too-many-lines
class SparseApplyFtrl():
    """
       Function: use to store sparse_apply_ftrl base parameters
       Modify : 2020-3-5
    """

    # pylint: disable=too-many-statements
    def __init__(self, var, accum, linear, grad, indices, lr, l1, l2,
                 l2_shrinkage, lr_power, kernel_name, compute_type, version):
        """
        Init sparse_apply_ftrl base parameters

        Parameters
        ----------
        var: dict
            data of input var
            datatype suports float32,float16
        accum: dict
            data of input accum
            datatype suports float32,float16
        linear: dict
            data of input linear
            datatype suports float32,float16
        grad: dict
            data of grad
            datatype supports float32,float16
        indices: dict
            data of indices
            datatype supports int32
        lr: const
            data of lr
            datatype supports float32,float16,int32
        l1: const
            data of l1
            datatype supports float32,float16,int32
        l2: const
            data of l2
            datatype supports float32,float16,int32
        lr_power: const
            data of lr_power
            datatype supports float32,float16,int32
        kernel_name: str
            the name of the operator
        compute_type: str
            the compute type of sparse_apply_ftrl
        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik()

        self.var_shape = var.get("shape")
        self.var_dtype = var.get("dtype").lower()
        self.var_new_shape = list(self.var_shape)
        self.var_new_shape[0] = self.var_new_shape[0] + 1

        self.accum_shape = accum.get("shape")
        self.accum_dtype = accum.get("dtype").lower()
        self.accum_new_shape = list(self.accum_shape)
        self.accum_new_shape[0] = self.accum_new_shape[0] + 1

        self.linear_shape = linear.get("shape")
        self.linear_dtype = linear.get("dtype").lower()
        self.linear_new_shape = list(self.linear_shape)
        self.linear_new_shape[0] = self.linear_new_shape[0] + 1

        self.indices_shape = indices.get("shape")
        self.indices_dtype = indices.get("dtype").lower()
        self.indices_new_shape = list(self.indices_shape)
        self.indices_new_shape[0] = self.indices_new_shape[0] + 1

        self.grad_shape = grad.get("shape")
        self.grad_dtype = grad.get("dtype").lower()
        self.grad_new_shape = list(self.grad_shape)
        self.grad_new_shape[0] = self.grad_new_shape[0] + 1

        self.var_ele_num = functools_reduce(lambda x, y: x * y, self.var_shape)
        self.indices_num = functools_reduce(lambda x, y: x * y,
                                            self.indices_shape)
        self.grad_num = functools_reduce(lambda x, y: x * y,
                                         self.grad_shape)
        self.kernel_name = kernel_name
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        self.l2_shrinkage = l2_shrinkage
        self.lr_power = lr_power
        self.check_param()

        if len(self.var_shape) > 1:
            self.update_data_num = functools_reduce(lambda x, y: x * y,
                                                    self.var_shape[1:])
        else:
            self.update_data_num = 1
        self.max_indice = self.var_shape[0]

        self.compute_type = compute_type
        self.version = version

        self.ub_size_bytes = \
            (tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) - 8192)

        self.var_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.var_dtype) // 8

        self.indices_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.indices_dtype) // 8
        self.var_data_each_block = 32 // self.var_dtype_bytes_size
        self.indices_data_each_block = 32 // self.indices_dtype_bytes_size
        self.indices_ub_number = 0
        self.grad_ub_number = 0

        self.index_loop_num = 0

        self.max_num_one_repeat = 128
        if self.var_dtype in ("float32", "float16"):
            self.max_num_one_repeat = 64

        if self.update_data_num < self.var_data_each_block:
            self.block_num = 1
        else:
            ai_core_num = tbe_platform.cce_conf.get_soc_spec(
                tbe_platform.cce_conf.CORE_NUM)
            self.indice_step = math.ceil(self.max_indice / ai_core_num)
            self.block_num = math.ceil(self.max_indice / self.indice_step)

        self.var_gm = self.tik_instance.Tensor(
            self.var_dtype, self.var_new_shape, name="var_gm",
            scope=tik.scope_gm)
        self.accum_gm = self.tik_instance.Tensor(
            self.var_dtype, self.accum_new_shape, name="accum_gm",
            scope=tik.scope_gm)
        self.linear_gm = self.tik_instance.Tensor(
            self.var_dtype, self.linear_new_shape, name="linear_gm",
            scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(
            self.indices_dtype,
            self.indices_new_shape,
            name="indices_gm",
            scope=tik.scope_gm)
        self.grad_gm = self.tik_instance.Tensor(
            self.grad_dtype,
            self.grad_new_shape,
            name="grad_gm",
            scope=tik.scope_gm)

        self.var_gm_out = self.tik_instance.Tensor(
            self.var_dtype, self.var_new_shape, name="var_gm_out",
            scope=tik.scope_gm)
        self.accum_gm_out = self.tik_instance.Tensor(
            self.accum_dtype, self.accum_new_shape, name="accum_gm_out",
            scope=tik.scope_gm)
        self.linear_gm_out = self.tik_instance.Tensor(
            self.linear_dtype, self.linear_new_shape, name="linear_gm_out",
            scope=tik.scope_gm)

        self.vconv_dst_dtype = "float32"

        self.var_vconv_ub = None
        self.accum_vconv_ub = None
        self.linear_vconv_ub = None
        self.grad_vconv_ub = None
        self.var_tile_vconv_ub = None
        self.accum_tile_vconv_ub = None
        self.linear_tile_vconv_ub = None
        self.grad_tile_vconv_ub = None

        self.var_ub = None
        self.grad_ub = None
        self.accum_ub = None
        self.linear_ub = None
        self.indices_ub = None
        self.var_tile_ub = None
        self.grad_tile_ub = None
        self.accum_tile_ub = None
        self.linear_tile_ub = None
        self.var_read_index = None
        self.grad_read_index = None
        self.indices_loop_index = None
        self.indices_tmp = None
        self.init_ub_tensor_para()

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
        grad_size_bytes = self.var_dtype_bytes_size * self.update_data_num
        indices_size_bytes = self.indices_dtype_bytes_size * self.indices_num

        need_vconv_dtype = ("float16",)
        if self.var_dtype in need_vconv_dtype:
            vconv_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
                self.vconv_dst_dtype)
            vconv_data_each_block = 32 // vconv_dtype_bytes_size
            vconv_size_bytes = \
                (grad_size_bytes //
                 self.var_dtype_bytes_size * vconv_dtype_bytes_size)

            if (grad_size_bytes + vconv_size_bytes) * 6 < (
                    self.ub_size_bytes * 0.9):
                self.grad_ub_number = math.ceil(
                    self.update_data_num /
                    self.var_data_each_block) * self.var_data_each_block

                self.vconv_ub_number = math.ceil(
                    self.update_data_num /
                    vconv_data_each_block) * vconv_data_each_block

                self.indices_ub_number = (self.ub_size_bytes - grad_size_bytes
                                          * 6 - vconv_size_bytes * 6) \
                                         // self.indices_dtype_bytes_size

                self.indices_ub_number = math.ceil(
                    self.indices_ub_number /
                    self.indices_data_each_block) * self.indices_data_each_block

            elif indices_size_bytes < (self.ub_size_bytes * 0.9):
                self.indices_ub_number = math.ceil(
                    self.indices_num /
                    self.indices_data_each_block) * self.indices_data_each_block
                self.grad_ub_number = (self.ub_size_bytes - indices_size_bytes) \
                                      // self.var_dtype_bytes_size // 18

                self.grad_ub_number = math.ceil(
                    self.grad_ub_number /
                    self.var_data_each_block) * self.var_data_each_block

                self.vconv_ub_number = math.ceil(
                    self.grad_ub_number /
                    vconv_data_each_block) * vconv_data_each_block

            else:
                self.grad_ub_number = \
                    (self.ub_size_bytes // 18 //
                     (vconv_dtype_bytes_size + self.var_dtype_bytes_size) //
                     18 // self.var_data_each_block * self.var_data_each_block)

                self.indices_ub_number = \
                    (self.ub_size_bytes // self.indices_dtype_bytes_size //
                     18 // self.var_data_each_block * self.var_data_each_block)

                self.vconv_ub_number = self.grad_ub_number
        else:
            if grad_size_bytes * 6 < self.ub_size_bytes * 0.9:
                self.grad_ub_number = math.ceil(
                    self.update_data_num /
                    self.var_data_each_block) * self.var_data_each_block
                self.indices_ub_number = \
                    (self.ub_size_bytes - grad_size_bytes * 6) // \
                    self.indices_dtype_bytes_size

                self.indices_ub_number = math.ceil(
                    self.indices_ub_number /
                    self.indices_data_each_block) * self.indices_data_each_block
                if self.indices_num < self.indices_ub_number:
                    self.indices_ub_number = math.ceil(
                        self.indices_num / self.indices_data_each_block
                    ) * self.indices_data_each_block

            elif indices_size_bytes < self.ub_size_bytes * 0.9:
                self.indices_ub_number = math.ceil(
                    self.indices_num /
                    self.indices_data_each_block) * self.indices_data_each_block

                self.grad_ub_number = (self.ub_size_bytes -
                                       indices_size_bytes) \
                                      // 6 // self.var_dtype_bytes_size

                self.grad_ub_number = math.ceil(
                    self.grad_ub_number /
                    self.var_data_each_block) * self.var_data_each_block
            else:
                self.indices_ub_number = \
                    (self.ub_size_bytes // self.indices_dtype_bytes_size //
                     6 // self.indices_data_each_block *
                     self.indices_data_each_block)

                self.grad_ub_number = \
                    (self.indices_ub_number // 6 //
                     self.var_data_each_block * self.var_data_each_block)

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
        need_vconv_dtype = ("float16",)
        if self.var_dtype in need_vconv_dtype:
            self.var_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.vconv_ub_number,),
                name="var_vconv_ub",
                scope=tik.scope_ubuf)
            self.accum_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.vconv_ub_number,),
                name="accum_vconv_ub",
                scope=tik.scope_ubuf)
            self.linear_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.vconv_ub_number,),
                name="linear_vconv_ub",
                scope=tik.scope_ubuf)
            self.grad_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.vconv_ub_number,),
                name="grad_vconv_ub",
                scope=tik.scope_ubuf)
            self.temp_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.vconv_ub_number,),
                name="temp_vconv_ub",
                scope=tik.scope_ubuf)
            self.temp2_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.vconv_ub_number,),
                name="temp2_vconv_ub",
                scope=tik.scope_ubuf)
            self.var_tile_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.var_data_each_block,),
                name="var_tile_vconv_ub",
                scope=tik.scope_ubuf)
            self.accum_tile_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.var_data_each_block,),
                name="accum_tile_vconv_ub",
                scope=tik.scope_ubuf)
            self.linear_tile_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.var_data_each_block,),
                name="linear_tile_vconv_ub",
                scope=tik.scope_ubuf)
            self.grad_tile_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.var_data_each_block,),
                name="grad_tile_vconv_ub",
                scope=tik.scope_ubuf)
            self.temp2_tile_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.var_data_each_block,),
                name="temp2_tile_vconv_ub",
                scope=tik.scope_ubuf)
            self.temp_tile_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.var_data_each_block,),
                name="temp_tile_vconv_ub",
                scope=tik.scope_ubuf)

        self.var_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.grad_ub_number,),
            name="var_ub",
            scope=tik.scope_ubuf)
        self.accum_ub = self.tik_instance.Tensor(
            self.accum_dtype, (self.grad_ub_number,),
            name="accum_ub",
            scope=tik.scope_ubuf)
        self.linear_ub = self.tik_instance.Tensor(
            self.linear_dtype, (self.grad_ub_number,),
            name="linear_ub",
            scope=tik.scope_ubuf)
        self.grad_ub = self.tik_instance.Tensor(
            self.grad_dtype, (self.grad_ub_number,),
            name="grad_ub",
            scope=tik.scope_ubuf)
        self.temp_ub = self.tik_instance.Tensor(
            self.linear_dtype, (self.grad_ub_number,),
            name="temp_ub",
            scope=tik.scope_ubuf)
        self.temp2_ub = self.tik_instance.Tensor(
            self.linear_dtype, (self.grad_ub_number,),
            name="temp2_ub",
            scope=tik.scope_ubuf)
        self.var_tile_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.var_data_each_block,),
            name="var_tile_ub",
            scope=tik.scope_ubuf)
        self.accum_tile_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.var_data_each_block,),
            name="accum_tile_ub",
            scope=tik.scope_ubuf)
        self.linear_tile_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.var_data_each_block,),
            name="linear_tile_ub",
            scope=tik.scope_ubuf)
        self.grad_tile_ub = self.tik_instance.Tensor(
            self.grad_dtype, (self.var_data_each_block,),
            name="grad_tile_ub",
            scope=tik.scope_ubuf)
        self.temp_tile_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.var_data_each_block,),
            name="temp_tile_ub",
            scope=tik.scope_ubuf)
        self.temp2_tile_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.var_data_each_block,),
            name="temp2_tile_ub",
            scope=tik.scope_ubuf)

        self.indices_ub = self.tik_instance.Tensor(
            self.indices_dtype, (self.indices_ub_number,),
            name="indices_ub",
            scope=tik.scope_ubuf)

        self.l1_ub = self.tik_instance.Scalar(self.vconv_dst_dtype)
        self.l1_ub.set_as(self.l1)
        self.l2_ub = self.tik_instance.Scalar(self.vconv_dst_dtype)
        self.l2_ub.set_as(self.l2 * 2)
        self.lr_power_ub_1 = self.tik_instance.Scalar(self.vconv_dst_dtype)
        self.lr_power_ub_1.set_as(self.lr_power * -1)
        self.lr_vrec_ub = self.tik_instance.Scalar(self.vconv_dst_dtype)
        self.lr_vrec_ub.set_as(1 / self.lr)
        self.zero_scaler = self.tik_instance.Scalar(self.vconv_dst_dtype)
        self.zero_scaler.set_as(0)
        self.one_scaler = self.tik_instance.Scalar(self.indices_dtype)
        self.one_scaler.set_as(1)
        self.one_scaler_1 = self.tik_instance.Scalar(self.indices_dtype)
        self.one_scaler_1.set_as(0)

        self.var_read_index = self.tik_instance.Scalar("int32")
        self.var_read_index.set_as(0)

        self.indices_value = self.tik_instance.Scalar("int32")
        self.indices_value.set_as(0)

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
        self.indices_value.set_as(self.indices_ub[indices_ub_index])

    def get_grad_read_index(self, indices_ub_index):
        """
        Calculate the index of the read grad

        Parameters
        ----------
        indices_ub_index:int32
            the index of the currently traversed indices in UB

        Returns
        -------
        None
        """
        self.grad_read_index.set_as(indices_ub_index * self.update_data_num)

    def grad_the_var(self, indices_in_index, indice_num):
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
                        need_vconv_dtype = ("float16",)
                        if self.var_dtype in need_vconv_dtype:
                            self.calc_grad(copy_flag=True)
                        else:
                            self.calc_grad(copy_flag=True)
            else:
                self.get_grad_read_index(indices_ub_index +
                                         indices_in_index)
                self.var_read_index.set_as(self.var_read_index *
                                           self.update_data_num)
                self.calc_grad(copy_flag=True)

    def calc_grad(self, copy_flag):
        """
        Calculate grad fragment

        Parameters
        ----------
        copy_flag: bool
            determine whether the currently var linear accum value need to be
            moved to ub

        Returns
        -------
        None
        """
        grad_loop = self.update_data_num // self.grad_ub_number
        if grad_loop > 0:
            with self.tik_instance.for_range(0, grad_loop) as loop_index:
                self.calc_grad_small(loop_index * self.grad_ub_number,
                                     self.grad_ub_number, copy_flag)

        last_num = self.update_data_num % self.grad_ub_number

        if last_num > 0:
            self.calc_grad_small(grad_loop * self.grad_ub_number,
                                 last_num, copy_flag)

    def calc_grad_small(self, read_index_offset, element_num, copy_flag):
        """
        Transfer update to UB and calculate

        Parameters
        ----------
        read_index_offset: int32
            the offset used to read the grad fragment
        element_num:
            the number of elements in the slice of grad
        copy_flag: bool
            determine whether the currently var linear accum value need to be
            moved to ub

        Returns
        -------
        None
        """
        grad_burst_len = math.ceil(element_num / self.var_data_each_block)
        tile_ele_num = element_num % self.var_data_each_block
        if copy_flag:
            self.tik_instance.data_move(
                self.var_ub, self.var_gm[self.var_read_index +
                                         read_index_offset], 0, 1, grad_burst_len, 0, 0)
            self.tik_instance.data_move(
                self.accum_ub, self.accum_gm[self.var_read_index +
                                             read_index_offset], 0, 1, grad_burst_len, 0, 0)
            self.tik_instance.data_move(
                self.linear_ub, self.linear_gm[self.var_read_index +
                                               read_index_offset], 0, 1, grad_burst_len, 0, 0)

        self.tik_instance.data_move(
            self.grad_ub,
            self.grad_gm[self.grad_read_index + read_index_offset], 0, 1,
            grad_burst_len, 0, 0)

        align_offset = 0
        if (tile_ele_num != 0 and
                self.update_data_num > self.var_data_each_block):
            align_ele_num = \
                (element_num // self.var_data_each_block *
                 self.var_data_each_block)

            align_offset = \
                (read_index_offset + align_ele_num -
                 (self.var_data_each_block - tile_ele_num))

            if copy_flag:
                self.tik_instance.data_move(
                    self.var_tile_ub,
                    self.var_gm[self.var_read_index + align_offset],
                    0, 1, 1, 0, 0)
                self.tik_instance.data_move(
                    self.accum_tile_ub,
                    self.accum_gm[self.var_read_index + align_offset],
                    0, 1, 1, 0, 0)
                self.tik_instance.data_move(
                    self.linear_tile_ub,
                    self.linear_gm[self.var_read_index + align_offset],
                    0, 1, 1, 0, 0)
            self.tik_instance.data_move(
                self.grad_tile_ub,
                self.grad_gm[self.grad_read_index + align_offset],
                0, 1, 1, 0, 0)

        compute_loop = element_num // self.max_num_one_repeat // 255

        if compute_loop > 0:
            with self.tik_instance.for_range(0, compute_loop) as index:
                index_offset = index * self.max_num_one_repeat * 255
                self.calc_process(self.max_num_one_repeat, index_offset,
                                  index_offset, index_offset, 255, False,
                                  copy_flag)
        last_loop = element_num % (self.max_num_one_repeat *
                                   255) // self.max_num_one_repeat
        if last_loop > 0:
            index_offset = compute_loop * self.max_num_one_repeat * 255
            self.calc_process(self.max_num_one_repeat, index_offset,
                              index_offset, index_offset, last_loop, False,
                              copy_flag)

        compute_mask = element_num % self.max_num_one_repeat
        if compute_mask > 0:
            index_offset = \
                (element_num // self.max_num_one_repeat *
                 self.max_num_one_repeat)

            if (tile_ele_num == 0 or
                    self.update_data_num < self.var_data_each_block):

                self.calc_process(compute_mask, index_offset, index_offset,
                                  index_offset, 1, False, copy_flag)
                self.tik_instance.data_move(
                    self.var_gm_out[self.var_read_index + read_index_offset],
                    self.var_ub, 0, 1, grad_burst_len, 0, 0)

                self.tik_instance.data_move(
                    self.accum_gm_out[self.var_read_index + read_index_offset],
                    self.accum_ub, 0, 1, grad_burst_len, 0, 0)
                self.tik_instance.data_move(
                    self.linear_gm_out[self.var_read_index + read_index_offset],
                    self.linear_ub, 0, 1, grad_burst_len, 0, 0)
            else:
                self.calc_process(self.var_data_each_block, 0, 0, 0, 1, True,
                                  copy_flag)
                self.tik_instance.data_move(
                    self.var_gm_out[self.var_read_index + align_offset],
                    self.var_tile_ub, 0, 1, 1, 0, 0)
                self.tik_instance.data_move(
                    self.accum_gm_out[self.var_read_index + align_offset],
                    self.accum_tile_ub, 0, 1, 1, 0, 0)
                self.tik_instance.data_move(
                    self.linear_gm_out[self.var_read_index + align_offset],
                    self.linear_tile_ub, 0, 1, 1, 0, 0)
                self.calc_process(compute_mask, index_offset, index_offset,
                                  index_offset, 1, False, copy_flag)
                self.tik_instance.data_move(
                    self.var_gm_out[self.var_read_index + read_index_offset],
                    self.var_ub, 0, 1, grad_burst_len - 1, 0, 0)
                self.tik_instance.data_move(
                    self.accum_gm_out[self.var_read_index + read_index_offset],
                    self.accum_ub, 0, 1, grad_burst_len - 1, 0, 0)
                self.tik_instance.data_move(
                    self.linear_gm_out[self.var_read_index + read_index_offset],
                    self.linear_ub, 0, 1, grad_burst_len - 1, 0, 0)
        else:
            self.tik_instance.data_move(
                self.var_gm_out[self.var_read_index + read_index_offset],
                self.var_ub, 0, 1, grad_burst_len, 0, 0)

            self.tik_instance.data_move(
                self.accum_gm_out[self.var_read_index + read_index_offset],
                self.accum_ub, 0, 1, grad_burst_len, 0, 0)
            self.tik_instance.data_move(
                self.linear_gm_out[self.var_read_index + read_index_offset],
                self.linear_ub, 0, 1, grad_burst_len, 0, 0)

    def calc_process(self, mask, dest_addr, src_addr1, src_addr2, repeat_times,
                     is_tile, copy_flag):
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
            and grad
        copy_flag: bool
            determine whether the currently var linear accum value need to be
            moved to ub

        Returns
        -------
        None
        """
        need_vconv_dtype = ("float16",)
        if self.var_dtype in need_vconv_dtype:
            if is_tile:
                if copy_flag:
                    self.tik_instance.vconv(mask, "",
                                            self.var_tile_vconv_ub[dest_addr],
                                            self.var_tile_ub[src_addr1],
                                            repeat_times, 1, 1, 8, 4)
                    self.tik_instance.vconv(mask, "",
                                            self.accum_tile_vconv_ub[dest_addr],
                                            self.accum_tile_ub[src_addr1],
                                            repeat_times, 1, 1, 8, 4)
                    self.tik_instance.vconv(mask, "",
                                            self.linear_tile_vconv_ub[dest_addr],
                                            self.linear_tile_ub[src_addr1],
                                            repeat_times, 1, 1, 8, 4)
                self.tik_instance.vconv(mask, "",
                                        self.grad_tile_vconv_ub[dest_addr],
                                        self.grad_tile_ub[src_addr1],
                                        repeat_times, 1, 1, 8, 4)
                compute_repeat_strid = 8
                mask = self.var_data_each_block
                src_var_ub = self.var_tile_vconv_ub
                src_accum_ub = self.accum_tile_vconv_ub
                src_linear_ub = self.linear_tile_vconv_ub
                src_grad_ub = self.grad_tile_vconv_ub
                temp_ub = self.temp_tile_vconv_ub
                temp2_ub = self.temp2_tile_vconv_ub
            else:
                if copy_flag:
                    self.tik_instance.vconv(mask, "",
                                            self.var_vconv_ub[dest_addr],
                                            self.var_ub[src_addr1],
                                            repeat_times, 1, 1, 8, 4)
                    self.tik_instance.vconv(mask, "",
                                            self.accum_vconv_ub[dest_addr],
                                            self.accum_ub[src_addr1],
                                            repeat_times, 1, 1, 8, 4)
                    self.tik_instance.vconv(mask, "",
                                            self.linear_vconv_ub[dest_addr],
                                            self.linear_ub[src_addr1],
                                            repeat_times, 1, 1, 8, 4)
                self.tik_instance.vconv(mask, "",
                                        self.grad_vconv_ub[dest_addr],
                                        self.grad_ub[src_addr2],
                                        repeat_times, 1, 1, 8, 4)
                compute_repeat_strid = 8
                src_var_ub = self.var_vconv_ub[src_addr1]
                src_accum_ub = self.accum_vconv_ub[src_addr1]
                src_linear_ub = self.linear_vconv_ub[src_addr1]
                src_grad_ub = self.grad_vconv_ub[src_addr2]
                temp_ub = self.temp_vconv_ub[src_addr1]
                temp2_ub = self.temp2_vconv_ub[src_addr1]
        else:
            if is_tile:
                compute_repeat_strid = \
                    (self.max_num_one_repeat // self.var_data_each_block)

                mask = self.var_data_each_block
                src_var_ub = self.var_tile_ub
                src_accum_ub = self.accum_tile_ub
                src_linear_ub = self.linear_tile_ub
                src_grad_ub = self.grad_tile_ub
                temp_ub = self.temp_tile_ub
                temp2_ub = self.temp2_tile_ub
            else:
                compute_repeat_strid = \
                    (self.max_num_one_repeat // self.var_data_each_block)

                src_var_ub = self.var_ub[src_addr1]
                src_accum_ub = self.accum_ub[src_addr1]
                src_linear_ub = self.linear_ub[src_addr1]
                src_grad_ub = self.grad_ub[src_addr2]
                temp_ub = self.temp_ub[src_addr1]
                temp2_ub = self.temp2_ub[src_addr1]

        if self.compute_type == "apply_ftrl":
            if self.version == "v2":
                # 0. grad_with_shrinkage=grad+2*l2_shrinkage*var
                self.tik_instance.vmuls(mask, temp_ub, src_var_ub,
                                        2 * self.l2_shrinkage, repeat_times, 1,
                                        1,
                                        compute_repeat_strid,
                                        compute_repeat_strid)

                self.tik_instance.vadd(mask, temp2_ub, src_grad_ub, temp_ub,
                                       repeat_times, 1, 1, 1,
                                       compute_repeat_strid,
                                       compute_repeat_strid,
                                       compute_repeat_strid)

            # 1.accum_new = accum + grad^2
            self.tik_instance.vmul(mask, temp_ub, src_grad_ub, src_grad_ub,
                                   repeat_times, 1, 1, 1, compute_repeat_strid,
                                   compute_repeat_strid, compute_repeat_strid)

            if self.version == "v2":
                self.tik_instance.vadd(mask, src_linear_ub, temp2_ub,
                                       src_linear_ub, repeat_times,
                                       1, 1, 1, compute_repeat_strid,
                                       compute_repeat_strid, compute_repeat_strid)
            else:
                self.tik_instance.vadd(mask, src_linear_ub, src_grad_ub,
                                       src_linear_ub, repeat_times,
                                       1, 1, 1, compute_repeat_strid,
                                       compute_repeat_strid, compute_repeat_strid)

            self.tik_instance.vln(mask, src_grad_ub, src_accum_ub, repeat_times,
                                  1, 1, compute_repeat_strid,
                                  compute_repeat_strid)

            self.tik_instance.vadd(mask, src_accum_ub, src_accum_ub, temp_ub,
                                   repeat_times,
                                   1, 1, 1, compute_repeat_strid,
                                   compute_repeat_strid, compute_repeat_strid)

            # accum_new^(-lr_power)    temp_ub
            self.tik_instance.vln(mask, temp_ub, src_accum_ub, repeat_times,
                                  1, 1, compute_repeat_strid,
                                  compute_repeat_strid)

            self.tik_instance.vmuls(mask, temp_ub, temp_ub, self.lr_power_ub_1,
                                    repeat_times, 1, 1, compute_repeat_strid,
                                    compute_repeat_strid)

            self.tik_instance.vexp(mask, temp_ub, temp_ub, repeat_times, 1, 1,
                                   compute_repeat_strid, compute_repeat_strid)

            self.tik_instance.vmuls(mask, temp2_ub, temp_ub, self.lr_vrec_ub,
                                    repeat_times, 1, 1, compute_repeat_strid,
                                    compute_repeat_strid)

            # accum^(-lr_power)    src_grad_ub
            self.tik_instance.vmuls(mask, src_grad_ub, src_grad_ub,
                                    self.lr_power_ub_1, repeat_times,
                                    1, 1, compute_repeat_strid,
                                    compute_repeat_strid)

            self.tik_instance.vexp(mask, src_grad_ub, src_grad_ub, repeat_times,
                                   1, 1, compute_repeat_strid,
                                   compute_repeat_strid)

            # 2.linear += grad - (accum_new^(-lr_power)-
            # accum^(-lr_power))/lr*var
            self.tik_instance.vsub(mask, temp_ub, src_grad_ub, temp_ub,
                                   repeat_times, 1, 1, 1, compute_repeat_strid,
                                   compute_repeat_strid, compute_repeat_strid)

            self.tik_instance.vmuls(mask, temp_ub, temp_ub, self.lr_vrec_ub,
                                    repeat_times, 1, 1, compute_repeat_strid,
                                    compute_repeat_strid)

            self.tik_instance.vmul(mask, temp_ub, temp_ub, src_var_ub,
                                   repeat_times, 1, 1, 1, compute_repeat_strid,
                                   compute_repeat_strid, compute_repeat_strid)
            self.tik_instance.vadd(mask, src_linear_ub, temp_ub, src_linear_ub,
                                   repeat_times, 1, 1, 1, compute_repeat_strid,
                                   compute_repeat_strid, compute_repeat_strid)

            # 3.x_res = l1*linear.sign()-linear     temp_ub
            self.tik_instance.vabs(mask, src_grad_ub, src_linear_ub,
                                   repeat_times,
                                   1, 1, compute_repeat_strid,
                                   compute_repeat_strid)

            self.tik_instance.vdiv(mask, temp_ub, src_linear_ub,
                                   src_grad_ub, repeat_times, 1, 1, 1,
                                   compute_repeat_strid,
                                   compute_repeat_strid,
                                   compute_repeat_strid)

            self.tik_instance.vmuls(mask, temp_ub, temp_ub, self.l1_ub,
                                    repeat_times, 1, 1, compute_repeat_strid,
                                    compute_repeat_strid)
            self.tik_instance.vsub(mask, temp_ub, temp_ub, src_linear_ub,
                                   repeat_times, 1, 1, 1, compute_repeat_strid,
                                   compute_repeat_strid, compute_repeat_strid)

            # 4.y_res = accum_new^(-lr_power)/lr + 2*l2   temp2_ub
            self.tik_instance.vadds(mask, temp2_ub, temp2_ub, self.l2_ub,
                                    repeat_times, 1, 1, compute_repeat_strid,
                                    compute_repeat_strid)

            # 5.var = x_res / y_res if linear.abs > l1, else var = 0   temp_ub
            self.tik_instance.vdiv(mask, temp_ub, temp_ub, temp2_ub,
                                   repeat_times, 1, 1, 1,
                                   compute_repeat_strid,
                                   compute_repeat_strid,
                                   compute_repeat_strid)

            self.tik_instance.vabs(mask, src_grad_ub, src_linear_ub,
                                   repeat_times, 1, 1, compute_repeat_strid,
                                   compute_repeat_strid)

            # 0 tensor
            self.tik_instance.vmuls(mask, temp2_ub, src_var_ub,
                                    self.zero_scaler, repeat_times, 1, 1,
                                    compute_repeat_strid, compute_repeat_strid)

            # l1 tensor
            self.tik_instance.vadds(mask, temp2_ub, temp2_ub, self.l1_ub,
                                    repeat_times, 1, 1, compute_repeat_strid,
                                    compute_repeat_strid)

            with self.tik_instance.for_range(0, repeat_times) as index:
                cmpmask = self.tik_instance.vcmp_gt(mask,
                                                    src_grad_ub[index*self.max_num_one_repeat],
                                                    temp2_ub[index*self.max_num_one_repeat],
                                                    1, 1)
                self.tik_instance.vmuls(mask, temp2_ub[index*self.max_num_one_repeat],
                                        src_var_ub[index*self.max_num_one_repeat],
                                        self.zero_scaler, 1, 1, 1,
                                        compute_repeat_strid, compute_repeat_strid)
                self.tik_instance.vsel(mask, 0, src_var_ub[index*self.max_num_one_repeat],
                                       cmpmask, temp_ub[index*self.max_num_one_repeat],
                                       temp2_ub[index*self.max_num_one_repeat], 1, 1, 1, 1,
                                       compute_repeat_strid, compute_repeat_strid,
                                       compute_repeat_strid)

        else:
            raise RuntimeError("the operater is not supported.")
        if self.var_dtype in need_vconv_dtype:
            if is_tile:
                self.tik_instance.vconv(mask, "", self.var_tile_ub,
                                        self.var_tile_vconv_ub, repeat_times, 1,
                                        1, 4, 8)
                self.tik_instance.vconv(mask, "", self.accum_tile_ub,
                                        self.accum_tile_vconv_ub, repeat_times,
                                        1, 1, 4, 8)
                self.tik_instance.vconv(mask, "", self.linear_tile_ub,
                                        self.linear_tile_vconv_ub, repeat_times,
                                        1, 1, 4, 8)
            else:
                self.tik_instance.vconv(mask, "", self.var_ub[src_addr1],
                                        self.var_vconv_ub[dest_addr],
                                        repeat_times, 1, 1, 4, 8)
                self.tik_instance.vconv(mask, "", self.accum_ub[src_addr1],
                                        self.accum_vconv_ub[dest_addr],
                                        repeat_times, 1, 1, 4, 8)
                self.tik_instance.vconv(mask, "", self.linear_ub[src_addr1],
                                        self.linear_vconv_ub[dest_addr],
                                        repeat_times, 1, 1, 4, 8)

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
                self.grad_the_var(
                    indices_loop_index * self.indices_ub_number,
                    self.indices_ub_number)

        indices_last_num = self.indices_num % self.indices_ub_number
        if indices_last_num > 0:
            self.grad_the_var(indices_loop_num * self.indices_ub_number,
                              indices_last_num)

    def check_param(self):
        """
        Check parameter

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        util.check_kernel_name(self.kernel_name)
        util.check_shape_rule(self.var_shape)
        util.check_shape_rule(self.indices_shape)
        util.check_shape_rule(self.grad_shape)

        util.check_tensor_shape_size(self.var_shape)
        util.check_tensor_shape_size(self.indices_shape)
        util.check_tensor_shape_size(self.grad_shape)

        check_list_var = ("float16", "float32")
        check_list_indices = ("int32")
        util.check_dtype_rule(self.var_dtype, check_list_var)
        util.check_dtype_rule(self.indices_dtype, check_list_indices)
        util.check_dtype_rule(self.grad_dtype, check_list_var)

        if self.accum_shape != self.var_shape:
            raise RuntimeError(
                "var and accum do not have the same shape.")

        if self.linear_shape != self.var_shape:
            raise RuntimeError(
                "var and linear do not have the same shape")

        if len(self.indices_shape) != 1:
            raise RuntimeError(
                "indices must be one-dimensional.")

        if self.lr <= 0:
            raise RuntimeError("lr should be a positive scaler.")

        if self.l1 < 0:
            raise RuntimeError(
                "l1 regularization strength should be a non-negative scalar.")

        if self.l2 < 0:
            raise RuntimeError(
                "l2 regularization strength should be a non-negative scalar.")

        if self.lr_power > 0:
            raise RuntimeError(
                "lr_power is should be a non-positive scalar.")

        if self.grad_shape[0] != self.indices_shape[0]:
            raise RuntimeError(
                "grad must be the same size as indices in the first dimension")

        if self.grad_shape[1:] != self.var_shape[1:]:
            raise RuntimeError(
                "var and grad must match in dimension")

        if (self.grad_dtype != self.accum_dtype or
                self.var_dtype != self.linear_dtype or
                self.var_dtype != self.accum_dtype):
            raise RuntimeError(
                "input dtype of var,accum,linear,grad must match ")

    def sparse_apply_ftrl_operator(self):
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
                self.init_ub_tensor()
                self.indices_loop_index.set_as(indices_loop_index)

                self.traversing_indices()
        else:
            self.init_ub_tensor()
            self.traversing_indices()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=(self.var_gm, self.accum_gm, self.linear_gm,
                    self.grad_gm, self.indices_gm),
            outputs=(self.var_gm_out, self.accum_gm_out, self.linear_gm_out),
            enable_l2=False)

        return self.tik_instance
