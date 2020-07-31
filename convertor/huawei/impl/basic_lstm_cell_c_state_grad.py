#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

basic_lstm_cell_cstate_grad
"""
from functools import reduce as functools_reduce

from te import platform as cce
from te import tik
from topi.cce import util


# pylint: disable=too-many-instance-attributes
class LstmCellGradInput():
    """
    Class: use to store LstmCellGradInput input parameters
    Modify : 2019-12-28
    """

    # pylint: disable=too-many-arguments
    def __init__(self, cell_state, dht_out, dht, dct, inpute_gate, forget_gate,
                 update_gate, output_gate, tanh_ct, kernel_name):
        """
        init LstmCellGradInput base parameters

        Parameters
        ----------
        cell_state: dict
            cell state at the last moment
        dht_out: dict
            output state gradient at time t
        dht: dict
            hidden state gradient at time t
        dct: dict
            cell state gradient at time t

        inpute_gate: dict
            forward it buffer value at time t
        forget_gate: dict
            forward ft buffer value at time t
        update_gate: dict
            forward jt buffer value at time t
        output_gate: dict
            forward ot buffer value at time t
        tanh_ct: dict
            forward tanh_ct buffer value at time t
        kernel_name: str
            op kernel name

        Returns
        -------
        None
        """
        if dht_out is not None:
            self.dht_out_shape = dht_out.get("shape")
            self.dht_out_dtype = dht_out.get("dtype")
        else:
            self.dht_out_shape = None
            self.dht_out_dtype = None
        self.dht_shape = dht.get("shape")
        self.dht_dtype = dht.get("dtype")
        self.dct_shape = dct.get("shape")
        self.dct_dtype = dct.get("dtype")
        self.it_shape = inpute_gate.get("shape")
        self.it_dtype = inpute_gate.get("dtype")
        self.ft_shape = forget_gate.get("shape")
        self.ft_dtype = forget_gate.get("dtype")
        self.jt_shape = update_gate.get("shape")
        self.jt_dtype = update_gate.get("dtype")
        self.ot_shape = output_gate.get("shape")
        self.ot_dtype = output_gate.get("dtype")
        self.tanh_ct_shape = tanh_ct.get("shape")
        self.tanh_ct_dtype = tanh_ct.get("dtype")
        self.c_shape = cell_state.get("shape")
        self.c_dtype = cell_state.get("dtype")

        self.batch_size = self.c_shape[1] * 16
        self.hidden_size = self.c_shape[0] * 16

        self.dgate_shape = (self.c_shape[0] * 4, self.c_shape[1],
                            self.c_shape[2], self.c_shape[3])
        self.dgate_dtype = "float16"

        self.kernel_name = kernel_name

        self.check_input_param()

        product_name = "cloud"
        self.tik_instance = tik.Tik(tik.Dprofile("v100", product_name))
        self.aicore_num = tik.Dprofile("v100", product_name).get_aicore_num()

        if self.c_shape[1] * self.c_shape[0] < self.aicore_num:
            self.aicore_num = self.c_shape[1] * self.c_shape[0]

        self.init_gm_tensor()

    def check_input_param(self):
        """
        Check the input parameter

        Parameters
        ----------
        None

        Returns:
        None
        """
        shape_list = (self.c_shape, self.dht_shape, self.dct_shape,
                      self.it_shape, self.jt_shape, self.ft_shape,
                      self.ot_shape, self.tanh_ct_shape)
        if self.dht_out_shape is not None:
            shape_list += (self.dht_out_shape,)

        for shape in shape_list:
            util.check_shape_rule(shape, min_dim=4, max_dim=4)
            util.check_tensor_shape_size(shape)
            if shape != self.c_shape:
                raise RuntimeError("the input shapes are not same")

        check_list = ("float16", "float32")
        dtype_list = (self.c_dtype, self.dht_dtype, self.dct_dtype,
                      self.it_dtype, self.jt_dtype, self.ft_dtype,
                      self.ot_dtype, self.tanh_ct_dtype)

        if self.dht_out_dtype is not None:
            dtype_list += (self.dht_out_dtype,)

        for dtype in dtype_list:
            util.check_dtype_rule(dtype.lower(), check_list)
            if dtype != self.c_dtype:
                raise RuntimeError("the input dtypes are not same")

        util.check_kernel_name(self.kernel_name)

    def init_gm_tensor(self):
        """
        Declare tensor on gm

        Parameters
        ----------
        None

        Returns:
        None
        """
        if self.dht_out_dtype is not None:
            self.gm_dht_out = self.tik_instance.Tensor(
                self.dht_out_dtype,
                self.dht_out_shape,
                name="gm_dht_out",
                scope=tik.scope_gm)
        self.gm_dht = self.tik_instance.Tensor(
            self.dht_dtype, self.dht_shape, name="gm_dht", scope=tik.scope_gm)
        self.gm_dct = self.tik_instance.Tensor(
            self.dct_dtype, self.dct_shape, name="gm_dct", scope=tik.scope_gm)
        self.gm_it = self.tik_instance.Tensor(
            self.it_dtype, self.it_shape, name="gm_it", scope=tik.scope_gm)
        self.gm_ft = self.tik_instance.Tensor(
            self.ft_dtype, self.ft_shape, name="gm_ft", scope=tik.scope_gm)
        self.gm_jt = self.tik_instance.Tensor(
            self.jt_dtype, self.jt_shape, name="gm_jt", scope=tik.scope_gm)
        self.gm_ot = self.tik_instance.Tensor(
            self.ot_dtype, self.ot_shape, name="gm_ot", scope=tik.scope_gm)
        self.gm_tanh_ct = self.tik_instance.Tensor(
            self.tanh_ct_dtype,
            self.tanh_ct_shape,
            name="gm_tanh_ct",
            scope=tik.scope_gm)
        self.gm_c = self.tik_instance.Tensor(
            self.c_dtype, self.c_shape, name="gm_c", scope=tik.scope_gm)

        # output gm
        self.gm_dct1 = self.tik_instance.Tensor(
            self.c_dtype, self.c_shape, name="gm_dct1", scope=tik.scope_gm)

        # tmp_output
        self.gm_dgate = self.tik_instance.Tensor(
            self.dgate_dtype,
            self.dgate_shape,
            name="gm_dgate",
            scope=tik.scope_gm)


class LstmCellGrad(LstmCellGradInput):
    """
    Class: use to store LstmCellGrad input parameters
    Modify : 2019-12-28
    """

    # pylint: disable=too-many-arguments
    def __init__(self, cell_state, dht_out, dht, dct, inpute_gate, forget_gate,
                 update_gate, output_gate, tanh_ct, kernel_name):
        """
        init LstmCellGradInput base parameters

        Parameters
        ----------
        cell_state: dict
            cell state at the last moment
        dht_out: dict
            output state gradient at time t
        dht: dict
            hidden state gradient at time t
        dct: dict
            cell state gradient at time t

        inpute_gate: dict
            forward it buffer value at time t
        forget_gate: dict
            forward ft buffer value at time t
        update_gate: dict
            forward jt buffer value at time t
        output_gate: dict
            forward ot buffer value at time t
        tanh_ct: dict
            forward tanh_ct buffer value at time t
        kernel_name: str
            op kernel name

        Returns
        -------
        None
        """
        super(LstmCellGrad,
              self).__init__(cell_state, dht_out, dht, dct, inpute_gate,
                             forget_gate, update_gate, output_gate, tanh_ct,
                             kernel_name)
        self.fz_num = functools_reduce(lambda x, y: x * y, self.dht_shape[:2])

        self.ele_each_core = 0
        self.out_loop_ele_num = 0
        self.out_loop_num = 0
        self.inner_loop_ele_num = 0
        self.inner_loop_num = 0
        self.ub_size = 0

        # get vector compute parameters
        dtype_bytes_size = cce.cce_intrin.get_bit_len(self.dht_dtype) // 8
        self.v_mask_max = 128 // (dtype_bytes_size // 2)
        self.v_repeat_max = 255
        self.v_ele_each_block = 32 // dtype_bytes_size

        self.get_loop_params()

        self.ub_dot_conv = None
        self.ub_dit_conv = None
        self.ub_djt_conv = None
        self.ub_dft_conv = None

        self.ub_dht_out = None
        self.ub_dht = None
        self.ub_dht_add = None
        self.ub_ot = None
        self.ub_dot = None
        self.ub_tanh_ct = None
        self.ub_dc = None
        self.ub_dct = None
        self.ub_it = None
        self.ub_jt = None
        self.ub_djt = None
        self.ub_dit = None
        self.ub_dft = None
        self.ub_c = None
        self.ub_ft = None
        self.ub_dct1 = None
        self.tmp_data1 = None

    def get_tik_instance(self):
        """
        Return tik instance for tik debug

        Parameters
        ----------
        None

        Returns:
        tik_instance:
            tik instance
        """
        return self.tik_instance

    def get_loop_params(self):
        """
        Get loop params for vector compute

        Parameters
        ----------
        None

        Returns:
        None
        """

        ub_size_bytes = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE)
        self.ele_each_core = (
            functools_reduce(lambda x, y: x * y, self.dht_shape) //
            self.aicore_num)
        if self.ele_each_core < self.v_ele_each_block * 2:
            self.out_loop_num = 1
        else:
            self.out_loop_num = 2

        self.out_loop_ele_num = self.ele_each_core // self.out_loop_num

        ub_pice_num = 20
        total_num_each_core = self.ele_each_core * ub_pice_num

        dtype_size = cce.cce_intrin.get_bit_len(self.dht_dtype) // 8

        if total_num_each_core * dtype_size < ub_size_bytes:
            self.inner_loop_num = 0
            self.inner_loop_ele_num = 0
            self.last_loop_ele_num = self.out_loop_ele_num
            self.ub_size = self.out_loop_ele_num
        else:
            self.ub_size = (
                ub_size_bytes // dtype_size // ub_pice_num //
                self.out_loop_num // self.v_ele_each_block *
                self.v_ele_each_block)
            self.inner_loop_ele_num = self.ub_size
            self.inner_loop_num = (
                self.out_loop_ele_num // self.inner_loop_ele_num)
            self.last_loop_ele_num = (
                self.out_loop_ele_num % self.inner_loop_ele_num)

    def init_ub(self):
        """
        Declare tensor on UB buffer

        Parameters
        ----------
        None

        Returns:
        None
        """
        self.ub_dht = self.tik_instance.Tensor(
            self.dht_dtype, (self.ub_size,),
            name="ub_dht",
            scope=tik.scope_ubuf)
        if self.dht_out_shape is not None:
            self.ub_dht_out = self.tik_instance.Tensor(
                self.dht_out_dtype, (self.ub_size,),
                name="ub_dht_out",
                scope=tik.scope_ubuf)
            self.ub_dht_add = self.tik_instance.Tensor(
                self.dht_out_dtype, (self.ub_size,),
                name="ub_dht_add",
                scope=tik.scope_ubuf)
        self.ub_ot = self.tik_instance.Tensor(
            self.ot_dtype, (self.ub_size,), name="ub_ot", scope=tik.scope_ubuf)
        self.ub_dot = self.tik_instance.Tensor(
            self.ot_dtype, (self.ub_size,), name="ub_dot", scope=tik.scope_ubuf)
        self.ub_tanh_ct = self.tik_instance.Tensor(
            self.tanh_ct_dtype, (self.ub_size,),
            name="ub_tanh_ct",
            scope=tik.scope_ubuf)

        self.ub_dc = self.tik_instance.Tensor(
            self.dct_dtype, (self.ub_size,), name="ub_dc", scope=tik.scope_ubuf)
        self.ub_dct = self.tik_instance.Tensor(
            self.dct_dtype, (self.ub_size,),
            name="ub_dct",
            scope=tik.scope_ubuf)

        self.ub_it = self.tik_instance.Tensor(
            self.it_dtype, (self.ub_size,), name="ub_it", scope=tik.scope_ubuf)
        self.ub_jt = self.tik_instance.Tensor(
            self.jt_dtype, (self.ub_size,), name="ub_jt", scope=tik.scope_ubuf)
        self.ub_djt = self.tik_instance.Tensor(
            self.jt_dtype, (self.ub_size,), name="ub_djt", scope=tik.scope_ubuf)

        self.ub_dit = self.tik_instance.Tensor(
            self.it_dtype, (self.ub_size,), name="ub_dit", scope=tik.scope_ubuf)

        self.ub_dft = self.tik_instance.Tensor(
            self.ft_dtype, (self.ub_size,), name="ub_dft", scope=tik.scope_ubuf)
        self.ub_c = self.tik_instance.Tensor(
            self.c_dtype, (self.ub_size,), name="ub_c", scope=tik.scope_ubuf)
        self.ub_ft = self.tik_instance.Tensor(
            self.ft_dtype, (self.ub_size,), name="ub_ft", scope=tik.scope_ubuf)

        self.ub_dct1 = self.tik_instance.Tensor(
            self.c_dtype, (self.ub_size,), name="ub_dct1", scope=tik.scope_ubuf)

        self.tmp_data1 = self.tik_instance.Tensor(
            self.c_dtype, (self.ub_size,),
            name="temp_data1",
            scope=tik.scope_ubuf)

        if self.it_dtype == "float32":
            # vconv dot
            self.ub_dot_conv = self.tik_instance.Tensor(
                "float16", (self.ub_size,),
                name="ub_dot_conv",
                scope=tik.scope_ubuf)

            # vconv dit
            self.ub_dit_conv = self.tik_instance.Tensor(
                "float16", (self.ub_size,),
                name="ub_dit_conv",
                scope=tik.scope_ubuf)

            # vconv djt
            self.ub_djt_conv = self.tik_instance.Tensor(
                "float16", (self.ub_size,),
                name="ub_djt_conv",
                scope=tik.scope_ubuf)

            # vconv dft
            self.ub_dft_conv = self.tik_instance.Tensor(
                "float16", (self.ub_size,),
                name="ub_dft_conv",
                scope=tik.scope_ubuf)

    def vector_compute(self, index, mask, repeat):
        """
        Calculate the smallest data shard

        Parameters
        ----------
        src: int
            source address offset
        dst: int
            destination address offset
        mask: int
            vector compute mask
        repeat:
            vector compute repeat times
        Returns:
        None
        """
        # compute process for dot
        if self.dht_out_shape is not None:
            self.tik_instance.vadd(mask, self.ub_dht_add[index],
                                   self.ub_dht_out[index], self.ub_dht[index],
                                   repeat, 1, 1, 1, 8, 8, 8)
        else:
            self.ub_dht_add = self.ub_dht

        self.tik_instance.vmul(mask, self.ub_dot[index], self.ub_tanh_ct[index],
                               self.ub_dht_add[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dot[index], self.ub_dot[index],
                               self.ub_ot[index], repeat, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vmuls(mask, self.tmp_data1[index], self.ub_ot[index],
                                -1.0, repeat, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.tmp_data1[index],
                                self.tmp_data1[index], 1.0, repeat, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dot[index], self.ub_dot[index],
                               self.tmp_data1[index], repeat, 1, 1, 1, 8, 8, 8)

        # compute process for dc
        self.tik_instance.vmul(mask, self.ub_dht_add[index],
                               self.ub_dht_add[index], self.ub_ot[index],
                               repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dc[index], self.ub_tanh_ct[index],
                               self.ub_tanh_ct[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, self.ub_dc[index], self.ub_dc[index],
                                -1.0, repeat, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.ub_dc[index], self.ub_dc[index], 1,
                                repeat, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dc[index], self.ub_dc[index],
                               self.ub_dht_add[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.ub_dc[index], self.ub_dc[index],
                               self.ub_dct[index], repeat, 1, 1, 1, 8, 8, 8)

        # compute process for djt
        self.tik_instance.vmul(mask, self.tmp_data1[index], self.ub_jt[index],
                               self.ub_jt[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, self.tmp_data1[index],
                                self.tmp_data1[index], -1.0, repeat, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.tmp_data1[index],
                                self.tmp_data1[index], 1, repeat, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.ub_djt[index], self.ub_it[index],
                               self.ub_dc[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_djt[index], self.ub_djt[index],
                               self.tmp_data1[index], repeat, 1, 1, 1, 8, 8, 8)

        # compute process for dit
        self.tik_instance.vmuls(mask, self.tmp_data1[index], self.ub_it[index],
                                -1.0, repeat, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.tmp_data1[index],
                                self.tmp_data1[index], 1.0, repeat, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dit[index], self.ub_dc[index],
                               self.ub_jt[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dit[index], self.ub_dit[index],
                               self.ub_it[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dit[index], self.ub_dit[index],
                               self.tmp_data1[index], repeat, 1, 1, 1, 8, 8, 8)

        # compute process for dft
        self.tik_instance.vmuls(mask, self.tmp_data1[index], self.ub_ft[index],
                                -1.0, repeat, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.tmp_data1[index],
                                self.tmp_data1[index], 1, repeat, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dft[index], self.ub_dc[index],
                               self.ub_c[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dft[index], self.ub_dft[index],
                               self.ub_ft[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dft[index], self.ub_dft[index],
                               self.tmp_data1[index], repeat, 1, 1, 1, 8, 8, 8)

        # compute process for dct-1
        self.tik_instance.vmul(mask, self.ub_dct1[index], self.ub_dc[index],
                               self.ub_ft[index], repeat, 1, 1, 1, 8, 8, 8)

        if self.it_dtype == "float32":
            # vconv dot
            self.tik_instance.vconv(mask, "", self.ub_dot_conv[index],
                                    self.ub_dot[index], repeat, 1, 1, 4, 8)
            # vconv dit
            self.tik_instance.vconv(mask, "", self.ub_dit_conv[index],
                                    self.ub_dit[index], repeat, 1, 1, 4, 8)
            # vconv djt
            self.tik_instance.vconv(mask, "", self.ub_djt_conv[index],
                                    self.ub_djt[index], repeat, 1, 1, 4, 8)
            # vconv dft
            self.tik_instance.vconv(mask, "", self.ub_dft_conv[index],
                                    self.ub_dft[index], repeat, 1, 1, 4, 8)

    def compute_each_loop(self, ele_num):
        """
        Calculate each loop

        Parameters
        ----------
        start_index: int
            source address offset

        Returns:
        None
        """
        # vector compute
        loop_num = ele_num // (self.v_mask_max * self.v_repeat_max)
        if loop_num > 0:
            with self.tik_instance.for_range(0, loop_num) as index:
                compute_index = self.v_mask_max * self.v_repeat_max * index
                self.vector_compute(compute_index, self.v_mask_max,
                                    self.v_repeat_max)

        repeat_times = (
            ele_num % (self.v_mask_max * self.v_repeat_max) // self.v_mask_max)
        if repeat_times > 0:
            compute_index = self.v_mask_max * self.v_repeat_max * loop_num
            self.vector_compute(compute_index, self.v_mask_max, repeat_times)

        tile_mask = ele_num % self.v_mask_max
        if tile_mask > 0:
            compute_index = (
                self.v_mask_max * self.v_repeat_max * loop_num +
                repeat_times * self.v_mask_max)
            self.vector_compute(compute_index, tile_mask, 1)

    def input_data_move_in(self, start_index, ele_num):
        """
        Move the input data to ub

        Parameters
        ----------
        start_index: int
            source address offset

        Returns:
        None
        """
        # move in vector data
        v_burst_lens = ele_num // self.v_ele_each_block
        if self.dht_out_shape is not None:
            self.tik_instance.data_move(self.ub_dht_out,
                                        self.gm_dht_out[start_index], 0, 1,
                                        v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_dht, self.gm_dht[start_index], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_ot, self.gm_ot[start_index], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_tanh_ct,
                                    self.gm_tanh_ct[start_index], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_it, self.gm_it[start_index], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_jt, self.gm_jt[start_index], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_c, self.gm_c[start_index], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_ft, self.gm_ft[start_index], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_dct, self.gm_dct[start_index], 0, 1,
                                    v_burst_lens, 0, 0)

    def compute_each_core(self, core_index, out_loop_index):
        """
        Calculate the data on each core

        Parameters
        ----------
        core_index: int
            the index of aicore
        core_index: int
            the index of out loop

        Returns:
        None
        """
        self.init_ub()
        loop_offset = (
            core_index * self.ele_each_core +
            out_loop_index * self.out_loop_ele_num)
        if self.inner_loop_num > 0:
            with self.tik_instance.for_range(0, self.inner_loop_num) as index:
                start_index = loop_offset + index * self.inner_loop_ele_num
                self.input_data_move_in(start_index, self.inner_loop_ele_num)
                self.compute_each_loop(self.inner_loop_ele_num)

                # move vector compute result to l2 and gm
                self.move_vector_data_out(start_index, self.inner_loop_ele_num)

        if self.last_loop_ele_num > 0:
            start_index = (
                loop_offset + self.inner_loop_num * self.inner_loop_ele_num)
            self.input_data_move_in(start_index, self.last_loop_ele_num)
            self.compute_each_loop(self.last_loop_ele_num)

            # move vector compute result to l2 and gm
            self.move_vector_data_out(start_index, self.last_loop_ele_num)

    def move_vector_data_out(self, index, ele_num):
        """
        Move the vector compute result to gm

        Parameters
        ----------
        index: int
            move out index
        ele_num: int
            the element number of result

        Returns:
        None
        """
        burst_len = ele_num // self.v_ele_each_block
        if self.it_dtype == "float32":
            djt_src = self.ub_djt_conv
            dit_src = self.ub_dit_conv
            dot_src = self.ub_dot_conv
            dft_src = self.ub_dft_conv

            dgate_burst_len = burst_len // 2
        else:
            djt_src = self.ub_djt
            dit_src = self.ub_dit
            dot_src = self.ub_dot
            dft_src = self.ub_dft
            dgate_burst_len = burst_len

        offset = self.batch_size * self.hidden_size
        self.tik_instance.data_move(self.gm_dgate[index], dit_src, 0, 1,
                                    dgate_burst_len, 0, 0)
        self.tik_instance.data_move(self.gm_dgate[index + offset], djt_src, 0,
                                    1, dgate_burst_len, 0, 0)
        self.tik_instance.data_move(self.gm_dgate[index + offset * 2], dft_src,
                                    0, 1, dgate_burst_len, 0, 0)
        self.tik_instance.data_move(self.gm_dgate[index + offset * 3], dot_src,
                                    0, 1, dgate_burst_len, 0, 0)

        self.tik_instance.data_move(self.gm_dct1[index], self.ub_dct1, 0, 1,
                                    burst_len, 0, 0)

    def compute(self):
        """
        Calculate the data

        Parameters
        ----------
        None

        Returns:
        None
        """
        with self.tik_instance.for_range(
                0, self.aicore_num, block_num=self.aicore_num) as index0:
            with self.tik_instance.for_range(
                    0, self.out_loop_num,
                    thread_num=self.out_loop_num) as index1:
                self.compute_each_core(index0, index1)

        if self.dht_out_shape is not None:
            input_list = (self.gm_c, self.gm_dht_out, self.gm_dht, self.gm_dct,
                          self.gm_it, self.gm_jt, self.gm_ft, self.gm_ot,
                          self.gm_tanh_ct)
        else:
            input_list = (self.gm_c, self.gm_dht, self.gm_dct, self.gm_it,
                          self.gm_jt, self.gm_ft, self.gm_ot, self.gm_tanh_ct)
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=input_list,
            outputs=(self.gm_dgate, self.gm_dct1),
            enable_l2=False)


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict, dict,
                       dict, float, str, str)
# pylint: disable=unused-argument,too-many-arguments,invalid-name
def basic_lstm_cell_c_state_grad(c,
                                 dht,
                                 dct,
                                 it,
                                 jt,
                                 ft,
                                 ot,
                                 tanhct,
                                 dgate,
                                 dct1,
                                 forget_bias=1,
                                 activation="None",
                                 kernel_name="basic_lstm_cell_cstate_grad"):
    """
    Calculate the gradient of the four gates and the state of c at t-1

    Parameters
    ----------
    c: dict
        cell state at the last moment
    dht: dict
        hidden state gradient at time t
    dct: dict
        cell state gradient at time t
    it: dict
        forward it buffer value at time t
    ft: dict
        forward ft buffer value at time t
    jt: dict
        forward jt buffer value at time t
    ot: dict
        forward ot buffer value at time t
    tanh_ct: dict
        forward tanh_ct buffer value at time t
    forget_bias: int
        the bias of forget gate
    activation: str
        activation method
    kernel_name: str
        op kernel name

    Returns:
    None
    """
    dht_out = None
    lstm_cell_grad = LstmCellGrad(c, dht_out, dht, dct, it, ft, jt, ot, tanhct,
                                  kernel_name)
    lstm_cell_grad.compute()

    return lstm_cell_grad.get_tik_instance()
