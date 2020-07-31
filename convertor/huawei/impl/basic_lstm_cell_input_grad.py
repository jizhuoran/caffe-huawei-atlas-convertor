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
    def __init__(self, dgate, input_weight, dropout_mask, dxt, dht, keep_prob,
                 kernel_name):
        """
        init LstmCellGradInput base parameters

        Parameters
        ----------
        dgate: dict
            the gradient of four gate
        input_weight: dict
            weight
        dropout_mask: dict
            the mask of dropout
        keep_prob: dict
            the keep prob
        kernel_name: str
            op kernel name

        Returns
        -------
        None
        """
        self.dgate_shape = dgate.get("shape")
        self.dgate_dtype = dgate.get("dtype")
        self.w_shape = input_weight.get("shape")
        self.w_dtype = input_weight.get("dtype")
        if dropout_mask:
            self.dropout_mask_shape = dropout_mask.get("shape")
            self.dropout_mask_dtype = dropout_mask.get("dtype")
        else:
            self.dropout_mask_shape = None
            self.dropout_mask_dtype = None
        self.dxt_shape = dxt.get("shape")
        self.dxt_dtype = dxt.get("dtype")
        self.dht_shape = dht.get("shape")
        self.dht_dtype = dht.get("dtype")

        self.keep_prob = keep_prob
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
        for check_shape in (self.dgate_shape, self.w_shape):
            util.check_shape_rule(check_shape, min_dim=4, max_dim=4)
            util.check_tensor_shape_size(check_shape)
            for check_dim in (check_shape[2], check_shape[3]):
                if check_dim != 16:
                    raise RuntimeError("the shape do not match the format!")

        # check k axis length match of dgate and w
        if self.dgate_shape[0] != self.w_shape[1]:
            raise RuntimeError("k axis length of dgate and w must match!")

        if self.w_shape[1] // 4 > self.w_shape[0]:
            raise RuntimeError(" the shape of weight is not satisfied!")

        util.check_dtype_rule(self.dgate_dtype.lower(), ("float16",))
        util.check_dtype_rule(self.w_dtype.lower(), ("float16",))

        if self.dropout_mask_dtype:
            util.check_dtype_rule(self.dropout_mask_dtype.lower(), ("uint8",))
            util.check_shape_rule(self.dropout_mask_shape, max_dim=1)
            util.check_tensor_shape_size(self.dropout_mask_shape)

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
        self.gm_dgate = self.tik_instance.Tensor(
            self.dgate_dtype,
            self.dgate_shape,
            name="gm_dgate",
            scope=tik.scope_gm)
        self.gm_w = self.tik_instance.Tensor(
            self.w_dtype, self.w_shape, name="gm_w", scope=tik.scope_gm)
        if self.dropout_mask_dtype:
            self.gm_dropout_mask = self.tik_instance.Tensor(
                self.dropout_mask_dtype,
                self.dropout_mask_shape,
                name="gm_dropout_mask",
                scope=tik.scope_gm)

        hidden_size = self.w_shape[1] // 4
        input_size = self.w_shape[0] - hidden_size
        xt_shape = (input_size, self.dgate_shape[1], 16, 16)
        ht_shape = (hidden_size, self.dgate_shape[1], 16, 16)
        self.gm_dxt = self.tik_instance.Tensor(
            self.dxt_dtype, xt_shape, name="gm_dxt", scope=tik.scope_gm)
        self.gm_dht = self.tik_instance.Tensor(
            self.dht_dtype, ht_shape, name="gm_dht", scope=tik.scope_gm)


class LstmCellGrad(LstmCellGradInput):
    """
    Class: use to store LstmCellGrad input parameters
    Modify : 2019-12-28
    """

    # pylint: disable=too-many-arguments
    def __init__(self, dgate, input_weight, dropout_mask, dxt, dht, keep_prob,
                 kernel_name):
        """
        init LstmCellGrad base parameters

        Parameters
        ----------
        dgate: dict
            the gradient of four gate
        input_weight: dict
            weight
        dropout_mask: dict
            the mask of dropout
        keep_prob: dict
            the keep prob
        kernel_name: str
            op kernel name

        Returns
        -------
        None
        """
        super(LstmCellGrad, self).__init__(dgate, input_weight, dropout_mask,
                                           dxt, dht, keep_prob, kernel_name)

        self.ele_each_core = 0
        self.loop_num_each_core = 0
        self.ele_each_loop = 0

        # get vector compute parameters
        dtype_bytes_size = cce.cce_intrin.get_bit_len(self.dxt_dtype) // 8
        self.mask_max = 128 // (dtype_bytes_size // 2)
        self.repeat_max = 255
        self.ele_each_block = 32 // dtype_bytes_size

        self.l0a_size = 16384

        self.l1_left = None
        self.l1_right = None
        self.ub_res = None
        self.ub_res_conv = None
        self.ub_prob_mask = None

        self.m_num = self.dgate_shape[1]
        self.k_num = self.dgate_shape[0]
        self.n_num = self.w_shape[0]

        if self.m_num * self.n_num <= 32:
            self.aicore_num = self.m_num * self.n_num
            self.m_each_core = 1
            self.n_each_core = 1
        else:
            self.aicore_num = self.n_num
            self.m_each_core = self.m_num
            self.n_each_core = 1

        self.res_num = 256 * self.m_each_core

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

    def init_buffer(self):
        """
        Declare tensor on UB buffer

        Parameters
        ----------
        None

        Returns:
        None
        """
        self.ub_res = self.tik_instance.Tensor(
            self.dxt_dtype, (self.res_num,),
            name="ub_dxt",
            scope=tik.scope_ubuf)

    def matmul_compute_each_core(self, core_idx):
        """
        compute the matrix matmul result

        Parameters
        ----------
        core_idx: int
            aicore index

        Returns:
        None
        """
        # load left matrix to l0a
        data_size = self.k_num * 256

        loop_num = data_size // self.l0a_size
        last_num = data_size % self.l0a_size
        l0c_res = self.tik_instance.Tensor(
            "float32", (self.res_num,), name="l0c_res", scope=tik.scope_cc)
        if last_num > 0:
            l_offset = self.l0a_size * self.m_num * loop_num
            r_offset = loop_num * self.l0a_size
            self.matmul_compute_each_k_loop(False, True, last_num, 0, l0c_res,
                                            l_offset, r_offset, core_idx)
            if loop_num > 0:
                with self.tik_instance.for_range(0, loop_num) as idx:
                    l_offset = idx * self.l0a_size * self.m_num
                    r_offset = idx * self.l0a_size
                    self.matmul_compute_each_k_loop(False, True, self.l0a_size,
                                                    1, l0c_res, l_offset,
                                                    r_offset, core_idx)
        else:
            if loop_num == 1:
                self.matmul_compute_each_k_loop(False, True, self.l0a_size, 0,
                                                l0c_res, 0, 0, core_idx)
            elif loop_num > 1:
                self.matmul_compute_each_k_loop(False, True, self.l0a_size, 0,
                                                l0c_res, 0, 0, core_idx)
                with self.tik_instance.for_range(1, loop_num) as idx:
                    l_offset = idx * self.l0a_size * self.m_num
                    r_offset = idx * self.l0a_size
                    self.matmul_compute_each_k_loop(False, True, self.l0a_size,
                                                    1, l0c_res, l_offset,
                                                    r_offset, core_idx)

        # move the matmul result to ub
        if self.dxt_dtype == "float16":
            self.tik_instance.data_move(self.ub_res, l0c_res, 0, 1,
                                        self.m_each_core, 0, 0, 1)
        else:
            self.tik_instance.data_move(self.ub_res, l0c_res, 0, 1,
                                        self.m_each_core, 0, 0)

    # pylint: disable=too-many-locals
    def matmul_compute_each_k_loop(self, is_trans_l, is_trans_r, ele_num,
                                   is_bias, l0c_res, l_k_offset, r_k_offset,
                                   core_idx):
        """
        compute the matrix matmul result

        Parameters
            ----------
        is_trans_l: bool
            if the left matrix need be transpose
        is_trans_r: bool
            if the right matrix need be transpose
        ele_num: int
            the number of elements to be calculated
        is_bias: int
            whether to accumulate
        l0c_res: tik tensor
            l0c buffer for matmul result
        l_k_offset: int
            left matrix read index offset
        r_k_offset: int
            right matrix read index offset
        core_idx: int
            aicore index

        Returns:
        None
        """
        if self.m_each_core > 1:
            thread = 2
        else:
            thread = 1
        burst_len = ele_num // 256

        with self.tik_instance.for_range(0, self.n_each_core) as idx_0:
            # move right matrix to l0b
            l1_right = self.tik_instance.Tensor(
                self.w_dtype, (ele_num,), name="l1_right", scope=tik.scope_cbuf)
            src_idx = (
                core_idx //
                (self.m_num // self.m_each_core) * self.k_num * 256 +
                idx_0 * self.k_num * 256 + r_k_offset)
            self.tik_instance.data_move(l1_right, self.gm_w[src_idx], 0, 1,
                                        ele_num // 16, 0, 0)
            l0b_y = self.tik_instance.Tensor(
                "float16", (self.l0a_size,), name="l0b_y", scope=tik.scope_cb)
            self.tik_instance.load2dv1(l0b_y, l1_right, 0, burst_len, 1, 0,
                                       is_trans_r)

            with self.tik_instance.for_range(
                    0, self.m_each_core, thread_num=thread) as idx_1:
                # move left matrix to l0a
                l1_left = self.tik_instance.Tensor(
                    self.w_dtype, (ele_num,),
                    name="l1_left",
                    scope=tik.scope_cbuf)
                l0a_x = self.tik_instance.Tensor(
                    "float16", (self.l0a_size,),
                    name="l0a_x",
                    scope=tik.scope_ca)
                core_offset = (
                    core_idx %
                    (self.m_num // self.m_each_core) * self.m_each_core * 256 +
                    l_k_offset)
                with self.tik_instance.for_range(0, ele_num // 256) as j:
                    src_idx = core_offset + idx_1 * 256 + j * self.m_num * 256
                    dst_idx = j * 256
                    self.tik_instance.data_move(l1_left[dst_idx],
                                                self.gm_dgate[src_idx], 0, 1,
                                                16, 0, 0)

                self.tik_instance.load2dv1(l0a_x, l1_left, 0, burst_len, 1, 0,
                                           is_trans_l)

                self.tik_instance.mmad(l0c_res[idx_1 * 256], l0a_x, l0b_y, 16,
                                       ele_num // 16, 16, is_bias)

    def dropout_compute(self, core_index):
        """
        Compute dxt dropout result

        Parameters
        ----------
        core_index: int
            aicore index

        Returns:
        None
        """
        # move dropout_mask to ub
        prob_mask_num = self.res_num // 8
        self.ub_prob_mask = self.tik_instance.Tensor(
            self.dropout_mask_dtype, (prob_mask_num,),
            name="ub_prob_mask",
            scope=tik.scope_ubuf)
        idx = prob_mask_num * core_index
        self.tik_instance.data_move(self.ub_prob_mask,
                                    self.gm_dropout_mask[idx], 0, 1,
                                    prob_mask_num // 32, 0, 0)
        # creat cmp mask
        cmp_mask = self.tik_instance.mov_tensor_to_cmpmask(self.ub_prob_mask)
        ub_keep_prob = self.tik_instance.Tensor(
            self.dxt_dtype, (self.mask_max * self.repeat_max,),
            name="ub_keep_prob",
            scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(self.mask_max, ub_keep_prob,
                                     self.keep_prob, self.repeat_max, 1, 8)
        ub_zero = self.tik_instance.Tensor(
            self.dxt_dtype, (self.mask_max * self.repeat_max,),
            name="ub_keep_prob",
            scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(self.mask_max, ub_zero, 0.0,
                                     self.repeat_max, 1, 8)

        loop_num = self.res_num // (self.mask_max * self.repeat_max)
        if loop_num > 0:
            with self.tik_instance.for_range(0, loop_num) as index:
                idx = index * self.repeat_max * self.mask_max
                self.tik_instance.vdiv(self.mask_max, self.ub_res[idx],
                                       self.ub_res[idx], ub_keep_prob,
                                       self.repeat_max, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vsel(self.mask_max, 0, self.ub_res[idx],
                                       cmp_mask, self.ub_res[idx], ub_zero, 1,
                                       1, 1, 1, 8, 8, 8)

        repeat_time = (
            self.res_num % (self.mask_max * self.repeat_max) // self.mask_max)
        if repeat_time > 0:
            idx = loop_num * self.repeat_max * self.mask_max
            self.tik_instance.vdiv(self.mask_max, self.ub_res[idx],
                                   self.ub_res[idx], ub_keep_prob, repeat_time,
                                   1, 1, 1, 8, 8, 8)
            with self.tik_instance.for_range(0, repeat_time) as idx:
                src_idx = (
                    idx * self.mask_max +
                    loop_num * self.repeat_max * self.mask_max)
                cmp_mask = self.tik_instance.mov_tensor_to_cmpmask(
                    self.ub_prob_mask[src_idx // 8])
                self.tik_instance.vsel(self.mask_max, 0, self.ub_res[src_idx],
                                       cmp_mask, self.ub_res[src_idx], ub_zero,
                                       1, 1, 1, 1, 8, 8, 8)

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
        self.matmul_compute_each_core(core_index)
        hidden_size = self.w_shape[1] // 4
        input_size = self.w_shape[0] - hidden_size
        if self.dropout_mask_dtype:
            with self.tik_instance.if_scope(
                    core_index < (self.m_num // self.m_each_core) * input_size):
                self.dropout_compute(core_index)
        self.data_move_out(core_index)

    def data_move_out(self, core_index):
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
        hidden_size = self.w_shape[1] // 4
        input_size = self.w_shape[0] - hidden_size

        burst_len = self.res_num // self.ele_each_block

        with self.tik_instance.if_scope(
                core_index < self.m_num // self.m_each_core * input_size):
            core_offset = core_index * self.res_num
            # cast result to float32
            self.tik_instance.data_move(self.gm_dxt[core_offset], self.ub_res,
                                        0, 1, burst_len, 0, 0)
        with self.tik_instance.else_scope():
            core_offset = (
                (core_index - self.m_num // self.m_each_core * input_size) *
                self.res_num)
            # cast result to float32
            self.tik_instance.data_move(self.gm_dht[core_offset], self.ub_res,
                                        0, 1, burst_len, 0, 0)

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
            self.init_buffer()
            self.compute_each_core(index)

        if self.dropout_mask_dtype:
            input_list = (self.gm_dgate, self.gm_w, self.gm_dropout_mask)
        else:
            input_list = (self.gm_dgate, self.gm_w)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=input_list,
            outputs=(self.gm_dxt, self.gm_dht),
            enable_l2=False)


@util.check_input_type(dict, dict, (dict, type(None)), dict, dict, float, str)
# pylint: disable=unused-argument,too-many-arguments,invalid-name
def basic_lstm_cell_input_grad(dgate,
                               w,
                               dropout_mask,
                               dxt,
                               dht,
                               keep_prob=1.0,
                               kernel_name="basic_lstm_cell_input_grad"):
    """
    Calculate the gradient of input

    Parameters
    ----------
    dgate: dict
        the gradient of four gate
    w: dict
        weight
    dropout_mask: dict
        the mask of dropout
    dxt: dict
        input x gradient
    dht: dict
        hidden state gradient value at time t
    keep_prob: dict
        the keep prob
    kernel_name: str
        op kernel name

    Returns:
    None
    """
    lstm_cell_grad = LstmCellGrad(dgate, w, dropout_mask, dxt, dht, keep_prob,
                                  kernel_name)
    lstm_cell_grad.compute()

    return lstm_cell_grad.get_tik_instance()
