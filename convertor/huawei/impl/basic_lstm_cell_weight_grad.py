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

basic_lstm_cell_input_grad
"""
import numpy as np

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
    def __init__(self, input_x, hidden_state, dgate, weight_gradient,
                 bias_gradient, kernel_name):
        """
        init LstmCellGradInput base parameters

        input_x: dict
            input date at time t
        hidden_state: dict
            hidden state at time t-1
        weight_gradient: dict
            four gates gradient
        bias_gradient: dict
            bias gradient
        kernel_name: str
            op kernel name

        Returns
        -------
        None
        """
        self.dgate_shape = dgate.get("shape")
        self.dgate_dtype = dgate.get("dtype")
        self.x_shape = input_x.get("shape")
        self.x_dtype = input_x.get("dtype")
        self.h_shape = hidden_state.get("shape")
        self.h_dtype = hidden_state.get("dtype")

        self.dw_shape = weight_gradient.get("shape")
        self.dw_dtype = weight_gradient.get("dtype")
        self.db_shape = bias_gradient.get("shape")
        self.db_dtype = bias_gradient.get("dtype")

        self.kernel_name = kernel_name

        self.check_input_param()

        product_name = "cloud"
        self.tik_instance = tik.Tik(tik.Dprofile("v100", product_name))
        self.aicore_num = tik.Dprofile("v100", product_name).get_aicore_num()

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
        for check_shape in (self.dgate_shape, self.x_shape, self.h_shape):
            util.check_shape_rule(check_shape, min_dim=4, max_dim=4)
            util.check_tensor_shape_size(check_shape)
            for check_dim in (check_shape[2], check_shape[3]):
                if check_dim != 16:
                    raise RuntimeError("the shape do not match the format!")

        util.check_dtype_rule(self.dgate_dtype.lower(), ("float16",))
        util.check_dtype_rule(self.x_dtype.lower(), ("float16",))
        util.check_dtype_rule(self.h_dtype.lower(), ("float16",))

        util.check_kernel_name(self.kernel_name)

        # check k axis length match
        if self.x_shape[1] != self.dgate_shape[1]:
            raise RuntimeError("k axis length of inputs must match!")

    def init_gm_tensor(self):
        """
        Declare tensor on gm

        Parameters
        ----------
        None

        Returns:
        None
        """
        self.gm_dgate = self.tik_instance.Tensor(
            self.dgate_dtype,
            self.dgate_shape,
            name="gm_dgate",
            scope=tik.scope_gm)
        self.gm_h = self.tik_instance.Tensor(
            self.h_dtype, self.h_shape, name="gm_h", scope=tik.scope_gm)
        self.gm_x = self.tik_instance.Tensor(
            self.x_dtype, self.x_shape, name="gm_x", scope=tik.scope_gm)

        self.gm_dw = self.tik_instance.Tensor(
            self.dw_dtype, self.dw_shape, name="gm_dw", scope=tik.scope_gm)
        self.gm_db = self.tik_instance.Tensor(
            self.db_dtype, self.db_shape, name="gm_db", scope=tik.scope_gm)


class LstmCellGrad(LstmCellGradInput):
    """
    Class: use to store LstmCellGrad input parameters
    Modify : 2019-12-28
    """

    # pylint: disable=too-many-arguments
    def __init__(self, input_x, hidden_state, dgate, weight_gradient,
                 bias_gradient, kernel_name):
        """
        init LstmCellGradInput base parameters

        input_x: dict
            input date at time t
        hidden_state: dict
            hidden state at time t-1
        weight_gradient: dict
            four gates gradient
        bias_gradient: dict
            bias gradient
        kernel_name: str
            op kernel name

        Returns
        -------
        None
        """
        super(LstmCellGrad, self).__init__(input_x, hidden_state, dgate,
                                           weight_gradient, bias_gradient,
                                           kernel_name)

        self.ele_each_core = 0
        self.loop_num_each_core = 0
        self.ele_each_loop = 0

        # get vector compute parameters
        dtype_bytes_size = cce.cce_intrin.get_bit_len(self.x_dtype) // 8
        self.mask_max = 128 // (dtype_bytes_size // 2)
        self.repeat_max = 255
        self.ele_each_block = 32 // dtype_bytes_size

        self.l0a_size = 8192

        self.l1_left = None
        self.l1_right = None
        self.l1_b = None
        self.ub_dw = None
        self.ub_trans = None
        self.ub_db = None
        self.b_matrix = None
        self.ub_conv = None
        self.zero_matrix = None

        self.m_num = self.x_shape[0] + self.h_shape[0]
        self.k_num = self.dgate_shape[1]
        self.n_num = self.dgate_shape[0]

        self.m_last_core = 0
        if self.m_num * self.n_num <= 32:
            self.aicore_num = self.m_num * self.n_num
            self.m_each_core = 1
            self.n_each_core = 1
        else:
            if self.m_num > self.aicore_num:
                if self.m_num % self.aicore_num == 0:
                    self.m_each_core = self.m_num // self.aicore_num
                    self.n_each_core = self.n_num
                else:
                    self.m_each_core = self.m_num // (self.aicore_num - 1)
                    self.m_last_core = (
                        self.m_num - self.m_each_core * (self.aicore_num - 1))
                    self.n_each_core = self.n_num
            else:
                if (self.m_num * self.n_num) % self.aicore_num == 0:
                    self.m_each_core = (
                        self.m_num // np.gcd(self.m_num, self.aicore_num))
                    self.n_each_core = (
                        self.n_num // (self.aicore_num //
                                       np.gcd(self.m_num, self.aicore_num)))
                else:
                    self.aicore_num = self.m_num
                    self.m_each_core = 1
                    self.n_each_core = self.n_num

        self.res_num = 256 * self.m_each_core * self.n_each_core

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

    # pylint: disable=too-many-locals
    def matmul_compute_each_core(self, row_num, column_num, m_loop, n_loop):
        """
        compute the matrix matmul result

        Parameters
        ----------
        core_idx: int
            aicore index

        Returns:
        None
        """
        k_data_size = self.k_num * 256

        with self.tik_instance.for_range(0, m_loop) as idx_0:
            # move left matrix to l1
            if k_data_size + self.l0a_size < 262144:
                l1_left = self.tik_instance.Tensor(
                    self.x_dtype, (k_data_size,),
                    name="l1_left",
                    scope=tik.scope_cbuf)

                burst_lens = k_data_size // self.ele_each_block
                with self.tik_instance.if_scope(
                        row_num + idx_0 < self.x_shape[0]):
                    src_offset = (row_num + idx_0) * k_data_size
                    self.tik_instance.data_move(l1_left, self.gm_x[src_offset],
                                                0, 1, burst_lens, 0, 0)
                with self.tik_instance.else_scope():
                    src_offset = ((row_num + idx_0 - self.x_shape[0]) *
                                  k_data_size)
                    self.tik_instance.data_move(l1_left, self.gm_h[src_offset],
                                                0, 1, burst_lens, 0, 0)

            if n_loop > 1:
                thread = 2
            else:
                thread = 1
            with self.tik_instance.for_range(
                    0, n_loop, thread_num=thread) as idx_1:
                l0a_x = self.tik_instance.Tensor(
                    "float16", (self.l0a_size,),
                    name="l0a_x",
                    scope=tik.scope_ca)
                l0b_y = self.tik_instance.Tensor(
                    "float16", (self.l0a_size,),
                    name="l0b_y",
                    scope=tik.scope_cb)

                if k_data_size * 2 < 262144:
                    l1_right = self.tik_instance.Tensor(
                        self.dgate_dtype, (k_data_size,),
                        name="l1_right",
                        scope=tik.scope_cbuf)
                    # move right matrix to l1 buffer
                    burst_lens = k_data_size // self.ele_each_block
                    src_idx = (column_num + idx_1) * k_data_size
                    self.tik_instance.data_move(l1_right,
                                                self.gm_dgate[src_idx], 0, 1,
                                                burst_lens, 0, 0)

                l0c_dw, l0c_db = self.matmul_compute_k_loop(
                    l0a_x, l0b_y, l1_left, l1_right)

                # move dw result to ub and gm
                ub_dw = self.tik_instance.Tensor(
                    "float16", (256,), name="ub_dw", scope=tik.scope_ubuf)
                self.tik_instance.data_move(ub_dw, l0c_dw, 0, 1, 1, 0, 0, 1)
                self.tik_instance.vtranspose(ub_dw, ub_dw)
                dw_out_offset = ((row_num + idx_0) * self.n_num * 256 +
                                 (column_num + idx_1) * 256)

                if self.dw_dtype == "float32":
                    ub_dw_conv = self.tik_instance.Tensor(
                        "float32", (256,), name="ub_dw", scope=tik.scope_ubuf)
                    self.tik_instance.vconv(64, "", ub_dw_conv, ub_dw, 4, 1, 1,
                                            8, 4)
                    self.tik_instance.data_move(self.gm_dw[dw_out_offset],
                                                ub_dw_conv, 0, 1, 32, 0, 0)
                else:
                    self.tik_instance.data_move(self.gm_dw[dw_out_offset],
                                                ub_dw, 0, 1, 16, 0, 0)

                # move db result to ub and gm
                dst = (column_num + idx_1) * 16
                ub_db = self.tik_instance.Tensor(
                    "float16", (256,), name="ub_db", scope=tik.scope_ubuf)

                if self.db_dtype == "float32":
                    b_len = 2
                    if_cast = 0
                else:
                    b_len = 1
                    if_cast = 1
                self.tik_instance.data_move(ub_db, l0c_db, 0, 1, 1, 0, 0,
                                            if_cast)
                self.tik_instance.data_move(self.gm_db[dst], ub_db, 0, 1, b_len,
                                            0, 0)

    def matmul_compute_k_loop(self, l0a_x, l0b_y, l1_left, l1_right):
        """
        compute the matrix matmul result

        Parameters
        ----------
        l0a_x: tik tensor
            l0a tensor for left matrix
        l0b_y: tik tensor
            l0b tensor for right matrix
        l1_left: tik tensor
            l1 tensor for left matrix
        l1_right: tik tensor
            l1 tensor for right matrix

        Returns:
        l0c_dw: tik tensor
            l0c tensor for dw result
        l0c_db: tik tensor
            l0c tensor for db result
        """
        k_data_size = self.k_num * 256

        loop_num = k_data_size // self.l0a_size
        last_num = k_data_size % self.l0a_size

        l0c_dw = self.tik_instance.Tensor(
            "float32", (256,), name="l0c_dw", scope=tik.scope_cc)
        l0c_db = self.tik_instance.Tensor(
            "float32", (256,), name="l0c_db", scope=tik.scope_cc)
        if last_num > 0:
            offset = loop_num * self.l0a_size
            self.matmul_compute_each_k_loop(l0a_x, l0b_y, l1_left, l1_right,
                                            True, True, last_num, 0, l0c_dw,
                                            l0c_db, offset)
            if loop_num > 0:
                with self.tik_instance.for_range(0, loop_num) as idx:
                    offset = idx * self.l0a_size
                    self.matmul_compute_each_k_loop(l0a_x, l0b_y, l1_left,
                                                    l1_right, True, True,
                                                    self.l0a_size, 1, l0c_dw,
                                                    l0c_db, offset)
        else:
            if loop_num == 1:
                self.matmul_compute_each_k_loop(l0a_x, l0b_y, l1_left, l1_right,
                                                True, True, self.l0a_size, 0,
                                                l0c_dw, l0c_db, 0)
            elif loop_num > 1:
                self.matmul_compute_each_k_loop(l0a_x, l0b_y, l1_left, l1_right,
                                                True, True, self.l0a_size, 0,
                                                l0c_dw, l0c_db, 0)
                with self.tik_instance.for_range(1, loop_num) as idx:
                    offset = idx * self.l0a_size
                    self.matmul_compute_each_k_loop(l0a_x, l0b_y, l1_left,
                                                    l1_right, True, True,
                                                    self.l0a_size, 1, l0c_dw,
                                                    l0c_db, offset)

        return l0c_dw, l0c_db

    # pylint: disable=too-many-locals
    def matmul_compute_each_k_loop(self, l0a_x, l0b_y, l1_left, l1_right,
                                   is_trans_l, is_trans_r, ele_num, is_bias,
                                   l0c_dw, l0c_db, k_offset):
        """
        compute the matrix matmul result

        Parameters
            ----------
        l1_left: tik tensor
            l1 tensor
        l1_right: tik tensor
            l1 tensor
        is_trans_l: bool
            if the left matrix need be transpose
        is_trans_r: bool
            if the right matrix need be transpose
        ele_num: int
            the number of elements to be calculated
        is_bias: int
            whether to accumulate
        l0c_dw: tik tensor
            l0c buffer for matmul result
        l0c_db: tik tensor
            l0c buffer for matmul result
        k_offset: int
            matrix read index offset

        Returns:
        None
        """
        load_len = ele_num // 256

        self.tik_instance.load2dv1(l0a_x, l1_left[k_offset], 0, load_len, 1, 0,
                                   is_trans_l)

        self.tik_instance.load2dv1(l0b_y, l1_right[k_offset], 0, load_len, 1, 0,
                                   is_trans_r)
        self.tik_instance.mmad(l0c_dw, l0a_x, l0b_y, 16, ele_num // 16, 16,
                               is_bias)

        self.tik_instance.mmad(l0c_db, self.zero_matrix, l0b_y, 16,
                               ele_num // 16, 16, is_bias)

    def init_b_matrix(self):
        """
        Init b matrix for db compute

        Parameters
        ----------
        None

        Returns:
        None
        """
        b_size = self.k_num * 256
        if b_size > self.l0a_size:
            b_size = self.l0a_size
        self.b_matrix = self.tik_instance.Tensor(
            self.x_dtype, (b_size,), name="b_matrix", scope=tik.scope_ubuf)
        repeat = b_size // self.mask_max
        self.tik_instance.vector_dup(self.mask_max, self.b_matrix, 0.0, repeat,
                                     1, 8)

        loop_num = self.k_num
        with self.tik_instance.for_range(0, loop_num) as index:
            offset = index * 256
            self.tik_instance.vector_dup(16, self.b_matrix[offset], 1.0, 1, 1,
                                         8)

        self.l1_b = self.tik_instance.Tensor(
            self.dgate_dtype, (b_size,), name="l1_b", scope=tik.scope_cbuf)
        self.tik_instance.data_move(self.l1_b, self.b_matrix, 0, 1,
                                    b_size // 16, 0, 0)

        self.zero_matrix = self.tik_instance.Tensor(
            "float16", (self.l0a_size,), name="zero_matrix", scope=tik.scope_ca)
        self.tik_instance.load2dv1(self.zero_matrix, self.l1_b, 0, self.k_num,
                                   1, 0, False)

    def compute_each_core(self, core_index):
        """
        Calculate the data on each core

        Parameters
        ----------
        core_index: int
            the index of aicore

        Returns:
        None
        """
        self.init_b_matrix()
        row_num = self.tik_instance.Scalar(dtype="int32")
        column_num = self.tik_instance.Scalar(dtype="int32")
        if self.m_last_core == 0:
            row_num.set_as(core_index % (self.m_num // self.m_each_core) *
                           self.m_each_core)
            column_num.set_as(core_index // (self.m_num // self.m_each_core) *
                              self.n_each_core)

            self.matmul_compute_each_core(row_num, column_num, self.m_each_core,
                                          self.n_each_core)
        else:
            row_num.set_as(self.m_each_core * core_index)
            column_num.set_as(0)
            with self.tik_instance.if_scope(core_index < self.aicore_num - 1):
                self.matmul_compute_each_core(row_num, column_num,
                                              self.m_each_core,
                                              self.n_each_core)
            with self.tik_instance.else_scope():
                self.matmul_compute_each_core(row_num, column_num,
                                              self.m_last_core,
                                              self.n_each_core)

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
                0, self.aicore_num, block_num=self.aicore_num) as index:
            self.compute_each_core(index)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=(self.gm_x, self.gm_h, self.gm_dgate),
            outputs=(self.gm_dw, self.gm_db),
            enable_l2=False)


@util.check_input_type(dict, dict, dict, dict, dict, str)
# pylint: disable=unused-argument,too-many-arguments,invalid-name
def basic_lstm_cell_weight_grad(x,
                                h,
                                dgate,
                                dw,
                                db,
                                kernel_name="basic_lstm_cell_weight_grad"):
    """
    Calculate the gradient of input

    Parameters
    ----------
    x: dict
        input date at time t
    h: dict
        hidden state at time t-1
    dgate: dict
        four gates gradient
    dw: dict
        weight gradient
    db: dict
        bias gradient
    kernel_name: str
        op kernel name

    Returns:
    None
    """
    lstm_cell_grad = LstmCellGrad(x, h, dgate, dw, db, kernel_name)
    lstm_cell_grad.compute()

    return lstm_cell_grad.get_tik_instance()
