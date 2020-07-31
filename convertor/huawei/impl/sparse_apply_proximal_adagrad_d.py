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

sparse_apply_proximal_adagrad
"""
# pylint: disable=import-error
from te import tik
from topi.cce import util


# max elememts can be put into ub, foor loop will be used for larger shape
IDX_SHAPE_MAX_INT32 = 1024 # INT32, must be set as x128 FP16
VAR_SHAPE_MAX_FP32 = 3776 # FP32, must be set as x128 FP16

# multi-core task assignment
MAX_CORE_NUM = 65535 # max usable core number
HARD_CORE_NUM = 32


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals,unused-argument,invalid-name
def sparse_apply_proximal_adagrad_d(var, accum, lr, l1, l2,
                                    grad, indices, var_out,
                                    accum_out,
                                    use_locking=False,
                                    kernel_name="sparse_apply_proximal_adagrad_d"):
    """
    the operator's compute:
    for i in range(index_len):
        gid = i
        vid = index[i]

        accum[vid] = accum[vid] + grad[gid]*grad[gid]

        lr1 = lr / np.sqrt(accum[vid])
        prox_var = var[vid] - grad[gid] * lr1
        lr2 = 1.0 / (1.0 + l2 * lr1)

        if l1 > 0:
            var_t1 = np.abs(prox_var) - lr1 * l1
            var[vid] = np.sign(prox_var) * np.maximum(var_t1, 0.0) * lr2
        else:
            var[vid] = prox_var * lr2

    Parameters
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32.
    accum: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    lr: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l1: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l2: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    indices: dict
        input tensor contains shape and dtype attributes.
        only support "int16", "int32"
    var_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    accum_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'accum'.
    use_locking: bool
        default value is "False"
    kernel_name: str
        kernel name, default value is "sparse_apply_proximal_adagrad_d"

    Returns:
        var_out, updated var, accum
    """

    var_shape = var.get("shape")
    var_dtype = var.get("dtype")

    idx_shape = indices.get("shape")
    idx_dtype = indices.get("dtype")
    accum_dtype = accum.get("dtype").lower()
    lr_dtype = lr.get("dtype").lower()
    l1_dtype = l1.get("dtype").lower()
    l2_dtype = l2.get("dtype").lower()
    grad_dtype = grad.get("dtype").lower()
    check_list = ("float32")
    util.check_dtype_rule(accum_dtype, check_list)
    util.check_dtype_rule(lr_dtype, check_list)
    util.check_dtype_rule(l1_dtype, check_list)
    util.check_dtype_rule(l2_dtype, check_list)
    util.check_dtype_rule(grad_dtype, check_list)
    _param_check(var_shape, var_dtype, idx_shape, idx_dtype, kernel_name)

    grad_shape = (idx_shape[0], ) + var_shape[1:]

    op_class = SparseApplyProximalAdagrad(var_shape, var_dtype, grad_shape,
                                          idx_shape, idx_dtype, kernel_name)

    tik_instance = op_class.compute()

    return tik_instance


def _param_check(var_shape, var_dtype, idx_shape, idx_dtype, kernel_name):
    util.check_shape_rule(var_shape, min_dim=1, max_dim=8)
    util.check_shape_rule(idx_shape, min_dim=1, max_dim=1)

    var_dtype_list = ("float16", "float32")
    util.check_dtype_rule(var_dtype.lower(), var_dtype_list)

    index_dtype_list = ("int32", "uint32", "int16", "uint16", "int64", "uint64")
    util.check_dtype_rule(idx_dtype.lower(), index_dtype_list)

    util.check_kernel_name(kernel_name)


class SparseApplyProximalAdagrad:
    '''
    wrap the inner varaiable and compute of the operator
    SparseApplyProximalAdagrad
    '''

    # pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-instance-attributes
    def __init__(self, var_shape, var_dtype, grad_shape,
                 idx_shape, idx_dtype, kernel_name):

        self.var_shape = var_shape
        self.var_dtype = var_dtype
        self.grad_shape = grad_shape
        self.idx_shape = idx_shape
        self.idx_dtype = idx_dtype
        self.kernel_name = kernel_name

        # var elements number in a block (32B)
        self.elem_32b = _block_elem(var_dtype)
        # var elements number in a vector repeat
        self.elem_128fp = _v128_elem(var_dtype)
        # index elements number in a block (32B)
        self.elem_32b_idx = _block_elem(idx_dtype)

        # max elememts can be put into ub of one core
        self.idx_shape_max = IDX_SHAPE_MAX_INT32 * _dtype_size("int32") // \
                            _dtype_size(idx_dtype)
        self.var_shape_max = VAR_SHAPE_MAX_FP32 * _dtype_size("float32") // \
                            _dtype_size(var_dtype)

        # length of index tensor
        self.index_len = idx_shape[0]
        self.var_len = var_shape[0]

        # shape configure for calculation
        self.var_dim0_elem_raw = _get_elem_num(var_shape) // var_shape[0]
        self.core_num, self.elem_per_core = self.multi_core_task_assign(
            self.var_dim0_elem_raw)

        # elem assigned to normal cores and last core
        self.var_elem_core0 = self.elem_per_core
        self.var_elem_corer = self.var_dim0_elem_raw - \
                             (self.core_num - 1) * self.elem_per_core

        self.tik_instance = tik.Tik()

        self._declare_in_out_gm(self.tik_instance)

    def _declare_in_out_gm(self, tik_instance):

        # input parameters
        self.var_gm = tik_instance.Tensor(self.var_dtype, self.var_shape,
                                          name="var_gm", scope=tik.scope_gm)
        self.accum_gm = tik_instance.Tensor(self.var_dtype, self.var_shape,
                                            name="accum_gm", scope=tik.scope_gm)
        # input scalar shape in gm
        in_scalar_shape_gm = (1,)

        self.lr_gm = tik_instance.Tensor(self.var_dtype, in_scalar_shape_gm,
                                         name="lr_gm", scope=tik.scope_gm)
        self.l1_gm = tik_instance.Tensor(self.var_dtype, in_scalar_shape_gm,
                                         name="l1_gm", scope=tik.scope_gm)
        self.l2_gm = tik_instance.Tensor(self.var_dtype, in_scalar_shape_gm,
                                         name="l2_gm", scope=tik.scope_gm)

        self.grad_gm = tik_instance.Tensor(self.var_dtype, self.grad_shape,
                                           name="grad_gm", scope=tik.scope_gm)
        self.index_gm = tik_instance.Tensor(self.idx_dtype, self.idx_shape,
                                            name="index_gm", scope=tik.scope_gm)

        self.var_out_gm = tik_instance.Tensor(self.var_dtype, self.var_shape,
                                              name="var_out_gm",
                                              scope=tik.scope_gm)
        self.accum_out_gm = tik_instance.Tensor(self.var_dtype, self.var_shape,
                                                name="accum_out_gm",
                                                scope=tik.scope_gm)

    def _def_read_scalar(self, tik_instance):

        # input scalar shape in ub
        in_scalar_shape_ub = (self.elem_32b,)

        # read and set scalar, 16 fp16
        lr_ub_t = tik_instance.Tensor(self.var_dtype, in_scalar_shape_ub,
                                      name="lr_ub_t", scope=tik.scope_ubuf)
        l1_ub_t = tik_instance.Tensor(self.var_dtype, in_scalar_shape_ub,
                                      name="l1_ub_t", scope=tik.scope_ubuf)
        l2_ub_t = tik_instance.Tensor(self.var_dtype, in_scalar_shape_ub,
                                      name="l2_ub_t", scope=tik.scope_ubuf)

        # read lr and momentum from gm to ub
        tik_instance.tensor_mov(lr_ub_t, self.lr_gm, '', 1, 1, 0, 0)
        tik_instance.tensor_mov(l1_ub_t, self.l1_gm, '', 1, 1, 0, 0)
        tik_instance.tensor_mov(l2_ub_t, self.l2_gm, '', 1, 1, 0, 0)

        # set scalar
        lr_scalar = tik_instance.Scalar(self.var_dtype, name="lr_scalar")
        l1_scalar = tik_instance.Scalar(self.var_dtype, name="l1_scalar")
        l2_scalar = tik_instance.Scalar(self.var_dtype, name="l2_scalar")

        lr_scalar.set_as(lr_ub_t[0])
        l1_scalar.set_as(l1_ub_t[0])
        l2_scalar.set_as(l2_ub_t[0])

        # pylint: disable=attribute-defined-outside-init
        self.l1_scalar_1 = tik_instance.Scalar("float32", name="l1_scalar_1")
        self.zero_scalar = tik_instance.Scalar("float32", name="zero_scalar")

        self.l1_scalar_1.set_as(l1_scalar)
        self.zero_scalar.set_as(0.0)

        # lr and l1, l2 scalar dump vector for vector calculation
        # 128 fp16 or 64 fp32, shared for all repeat
        scalar_shape_ub = (self.elem_128fp,)

        self.lr_vec_ub = tik_instance.Tensor(self.var_dtype,
                                             scalar_shape_ub,
                                             name="lr_vec_ub",
                                             scope=tik.scope_ubuf)
        self.l1_vec_ub = tik_instance.Tensor(self.var_dtype,
                                             scalar_shape_ub,
                                             name="l1_vec_ub",
                                             scope=tik.scope_ubuf)
        self.l2_vec_ub = tik_instance.Tensor(self.var_dtype,
                                             scalar_shape_ub,
                                             name="l2_vec_ub",
                                             scope=tik.scope_ubuf)

        self.one_vec_ub = tik_instance.Tensor(self.var_dtype,
                                              scalar_shape_ub,
                                              name="one_vec_ub",
                                              scope=tik.scope_ubuf)
        self.zero_vec_ub = tik_instance.Tensor(self.var_dtype,
                                               scalar_shape_ub,
                                               name="zero_vec_ub",
                                               scope=tik.scope_ubuf)
        self.neg1_vec_ub = tik_instance.Tensor(self.var_dtype,
                                               scalar_shape_ub,
                                               name="neg1_vec_ub",
                                               scope=tik.scope_ubuf)

        tik_instance.vector_dup(self.elem_128fp, self.lr_vec_ub,
                                lr_scalar, 1, 1, 8)
        tik_instance.vector_dup(self.elem_128fp, self.l1_vec_ub,
                                l1_scalar, 1, 1, 8)
        tik_instance.vector_dup(self.elem_128fp, self.l2_vec_ub,
                                l2_scalar, 1, 1, 8)

        tik_instance.vector_dup(self.elem_128fp, self.one_vec_ub,
                                1.0, 1, 1, 8)
        tik_instance.vector_dup(self.elem_128fp, self.zero_vec_ub,
                                0.0, 1, 1, 8)
        tik_instance.vector_dup(self.elem_128fp, self.neg1_vec_ub,
                                -1.0, 1, 1, 8)

    def compute(self):
        '''
        read indices tensor and access the dim0 of var, accum and grad
        and then apply the computation
        '''
        self._read_index_and_process(self.tik_instance)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.var_gm, self.accum_gm, self.lr_gm, self.l1_gm,
                    self.l2_gm, self.grad_gm, self.index_gm],
            outputs=[self.var_out_gm, self.accum_out_gm])

        return self.tik_instance

    def multi_core_task_assign(self, var_dim0_elem_raw):
        '''
        assign process of var and accum of each dim0 to multi-core
        '''
        # try to assign self.elem_32b for each core
        # elem_per_core should be x32B
        min_elem_per_core = self.elem_32b
        max_elem_per_core = self.var_shape_max - self.elem_32b

        elem_per_core = _ceil_div(var_dim0_elem_raw, HARD_CORE_NUM)
        elem_per_core = _round_up(elem_per_core, min_elem_per_core)
        core_num = _ceil_div(var_dim0_elem_raw, elem_per_core)

        if var_dim0_elem_raw <= 2 * self.elem_128fp:
            elem_per_core = _round_up(var_dim0_elem_raw, self.elem_32b)
            core_num = 1

        if elem_per_core > max_elem_per_core:

            core_num = _ceil_div(var_dim0_elem_raw, max_elem_per_core)
            elem_per_core = max_elem_per_core

            if core_num > MAX_CORE_NUM:
                core_num = _ceil_div(var_dim0_elem_raw, self.var_shape_max)
                elem_per_core = self.var_shape_max

        # last core must process elem >= elem_32b
        # if not, reduce 1 core and add task to the last core
        last_core_elem = var_dim0_elem_raw - (core_num - 1) * elem_per_core

        if last_core_elem < self.elem_32b:
            core_num -= 1

        if core_num == 0:
            core_num = 1

        return core_num, elem_per_core

    def _read_index_and_process(self, tik_instance):

        if((self.core_num > MAX_CORE_NUM) or \
            (self.var_elem_core0 > self.var_shape_max) or \
            (self.var_elem_corer > self.var_shape_max)):
            raise RuntimeError("var dim0 elem is too large:{}".format(
                self.var_dim0_elem_raw))

        if self.index_len <= self.idx_shape_max:
            self._read_small_index_and_process(self.tik_instance)
        else:
            self._read_long_index_and_process(self.tik_instance)

    def _read_small_index_and_process(self, tik_instance):
        idx_len_ub = _round_up(self.index_len, self.elem_32b_idx)
        idx_shape_ub = (idx_len_ub,)
        idx_read_rep = idx_len_ub // self.elem_32b_idx

        thread_num = _get_thread_num(self.index_len)

        with self.tik_instance.for_range(0, self.core_num,
                                         block_num=self.core_num) as core_id:

            self._def_read_scalar(self.tik_instance)

            # declare and read index tensor ub
            # pylint: disable=attribute-defined-outside-init
            self.index_ub = tik_instance.Tensor(self.idx_dtype, idx_shape_ub,
                                                name="index_ub",
                                                scope=tik.scope_ubuf)

            tik_instance.tensor_mov(self.index_ub, self.index_gm,
                                    '', 1, idx_read_rep, 0, 0)

            with tik_instance.if_scope(core_id < self.core_num - 1):
                self._apply_compute_multi_core(
                    self.tik_instance, 0, self.index_len,
                    thread_num, core_id, self.var_elem_core0)
            with tik_instance.else_scope():
                # last core
                self._apply_compute_multi_core(
                    self.tik_instance, 0, self.index_len,
                    thread_num, core_id, self.var_elem_corer)

            # copy var to var_out after calculation
            with tik_instance.if_scope(core_id < self.core_num - 1):
                self._copy_var_to_var_out(tik_instance, core_id,
                                          self.var_elem_core0)
            with tik_instance.else_scope():
                self._copy_var_to_var_out(tik_instance, core_id,
                                          self.var_elem_corer)

    def _read_long_index_and_process(self, tik_instance):
        idx_copy_num = _ceil_div(self.index_len, self.idx_shape_max)

        index_len_ub = self.idx_shape_max
        index_len_ub = _round_up(index_len_ub, self.elem_32b_idx)
        idx_shape_ub = (index_len_ub,)

        # last for loop read number
        idx_num_resi = self.index_len - (idx_copy_num - 1) * self.idx_shape_max

        idx_read_rep0 = _ceil_div(self.idx_shape_max, self.elem_32b_idx)

        thread_num0 = _get_thread_num(self.idx_shape_max)
        thread_numr = _get_thread_num(idx_num_resi)

        with self.tik_instance.for_range(0, self.core_num,
                                         block_num=self.core_num) as core_id:

            self._def_read_scalar(self.tik_instance)

            # pylint: disable=attribute-defined-outside-init
            self.index_ub = tik_instance.Tensor(self.idx_dtype, idx_shape_ub,
                                                name="index_ub",
                                                scope=tik.scope_ubuf)

            # read index tensor segment
            with tik_instance.for_range(0, idx_copy_num) as idx_cnt:

                idx_addr = tik_instance.Scalar("uint32", name="idx_addr")
                idx_addr.set_as(idx_cnt * self.idx_shape_max)

                # self.idx_shape_max index len
                tik_instance.tensor_mov(self.index_ub,
                                        self.index_gm[idx_addr],
                                        '', 1, idx_read_rep0, 0, 0)

                with tik_instance.if_scope(idx_cnt < idx_copy_num - 1):

                    with tik_instance.if_scope(core_id < self.core_num - 1):
                        self._apply_compute_multi_core(
                            self.tik_instance, idx_addr, self.idx_shape_max,
                            thread_num0, core_id, self.var_elem_core0)
                    with tik_instance.else_scope():
                        # last core
                        self._apply_compute_multi_core(
                            self.tik_instance, idx_addr, self.idx_shape_max,
                            thread_num0, core_id, self.var_elem_corer)

                with tik_instance.else_scope():

                    with tik_instance.if_scope(core_id < self.core_num - 1):
                        self._apply_compute_multi_core(
                            self.tik_instance, idx_addr, idx_num_resi,
                            thread_numr, core_id, self.var_elem_core0)
                    with tik_instance.else_scope():
                        # last core
                        self._apply_compute_multi_core(
                            self.tik_instance, idx_addr, idx_num_resi,
                            thread_numr, core_id, self.var_elem_corer)

            # copy var to var_out after calculation
            with tik_instance.if_scope(core_id < self.core_num - 1):
                self._copy_var_to_var_out(tik_instance, core_id,
                                          self.var_elem_core0)
            with tik_instance.else_scope():
                self._copy_var_to_var_out(tik_instance, core_id,
                                          self.var_elem_corer)

    def _copy_var_to_var_out(self, tik_instance, core_id, elem_cur_core):

        if self.core_num > 1:
            self._copy_var_to_var_out_multi_core(tik_instance,
                                                 core_id, elem_cur_core)
        else:
            self._copy_var_to_var_out_single_core(tik_instance)

    def _copy_var_to_var_out_multi_core(self, tik_instance,
                                        core_id, elem_cur_core):
        # the copy number of each core must >= 32B
        copy_elem_ub = _round_up(elem_cur_core, self.elem_32b)
        copy_shape_ub = (copy_elem_ub, )

        copy_repeat0 = elem_cur_core // self.elem_32b

        # last 32B need special treatment
        x32b_resi_num = elem_cur_core % self.elem_32b

        thread_num = _get_thread_num(self.var_len)

        with tik_instance.for_range(0, self.var_len,
                                    thread_num=thread_num) as cnt:

            copy_ub = tik_instance.Tensor(self.var_dtype, copy_shape_ub,
                                          name="copy_ub", scope=tik.scope_ubuf)
            copy_ub_accum = tik_instance.Tensor(self.var_dtype, copy_shape_ub,
                                                name="copy_ub_accum", scope=tik.scope_ubuf)

            var_addr = tik_instance.Scalar("uint32", name="var_addr")
            var_addr.set_as(cnt * self.var_dim0_elem_raw + \
                            core_id * self.elem_per_core)

            if copy_repeat0 > 0:
                # copy data from var to var_out
                tik_instance.tensor_mov(copy_ub, self.var_gm[var_addr],
                                        '', 1, copy_repeat0, 0, 0)
                tik_instance.tensor_mov(copy_ub_accum, self.accum_gm[var_addr],
                                        '', 1, copy_repeat0, 0, 0)
                # results move out
                tik_instance.tensor_mov(self.var_out_gm[var_addr], copy_ub,
                                        '', 1, copy_repeat0, 0, 0)
                tik_instance.tensor_mov(self.accum_out_gm[var_addr], copy_ub_accum,
                                        '', 1, copy_repeat0, 0, 0)

            if x32b_resi_num > 0:
                # last 32B address, each core process data  >= 32B
                gm_offset = elem_cur_core - self.elem_32b

                var_addr.set_as(var_addr + gm_offset)
                tik_instance.tensor_mov(copy_ub, self.var_gm[var_addr],
                                        '', 1, 1, 0, 0)
                tik_instance.tensor_mov(copy_ub_accum, self.accum_gm[var_addr],
                                        '', 1, 1, 0, 0)
                # results move out
                tik_instance.tensor_mov(self.var_out_gm[var_addr], copy_ub,
                                        '', 1, 1, 0, 0)
                tik_instance.tensor_mov(self.accum_out_gm[var_addr], copy_ub_accum,
                                        '', 1, 1, 0, 0)

    def _copy_var_to_var_out_single_core(self, tik_instance):

        total_elem = _get_elem_num(self.var_shape)
        batch_read_block = 1024 * 2 // _dtype_size(self.var_dtype)
        batch_read_elem = batch_read_block * self.elem_32b

        batch_loop_num = total_elem // batch_read_elem
        resi_elem = total_elem - batch_loop_num * batch_read_elem

        # last 32B need special treatment
        x32b_resi_num = resi_elem % self.elem_32b

        # repeat for loop read
        copy_repeat0 = batch_read_elem // self.elem_32b
        copy_repeatr = _ceil_div(resi_elem, self.elem_32b)

        thread_num = _get_thread_num(batch_loop_num)

        batch_copy_shape_ub = (batch_read_elem,)
        x32b_shape = (self.elem_32b,)

        # copy full batch
        with tik_instance.for_range(0, batch_loop_num,
                                    thread_num=thread_num) as cnt:

            copy_ub = tik_instance.Tensor(self.var_dtype, batch_copy_shape_ub,
                                          name="copy_ub", scope=tik.scope_ubuf)
            copy_ub_accum = tik_instance.Tensor(self.var_dtype, batch_copy_shape_ub,
                                                name="copy_ub_accum", scope=tik.scope_ubuf)

            var_addr = tik_instance.Scalar("uint32", name="var_addr")
            var_addr.set_as(cnt * batch_read_elem)

            # copy data from var to var_out
            tik_instance.tensor_mov(copy_ub, self.var_gm[var_addr],
                                    '', 1, copy_repeat0, 0, 0)
            tik_instance.tensor_mov(self.var_out_gm[var_addr], copy_ub,
                                    '', 1, copy_repeat0, 0, 0)
            tik_instance.tensor_mov(copy_ub_accum, self.accum_gm[var_addr],
                                    '', 1, copy_repeat0, 0, 0)
            tik_instance.tensor_mov(self.accum_out_gm[var_addr], copy_ub_accum,
                                    '', 1, copy_repeat0, 0, 0)

        # copy resi batch
        if resi_elem > 0:
            copy_ub = tik_instance.Tensor(self.var_dtype, batch_copy_shape_ub,
                                          name="copy_ub",
                                          scope=tik.scope_ubuf)
            copy_ub_accum = tik_instance.Tensor(self.var_dtype, batch_copy_shape_ub,
                                                name="copy_ub_accum",
                                                scope=tik.scope_ubuf)
            replace_ub = tik_instance.Tensor(self.var_dtype, x32b_shape,
                                             name="replace_ub",
                                             scope=tik.scope_ubuf)
            replace_ub_accum = tik_instance.Tensor(self.var_dtype, x32b_shape,
                                                   name="replace_ub_accum",
                                                   scope=tik.scope_ubuf)

            resi_addr = batch_loop_num * batch_read_elem

            # load resi loop data
            tik_instance.tensor_mov(copy_ub, self.var_gm[resi_addr],
                                    '', 1, copy_repeatr, 0, 0)
            tik_instance.tensor_mov(copy_ub_accum, self.accum_gm[resi_addr],
                                    '', 1, copy_repeatr, 0, 0)

            if x32b_resi_num > 0:
                # last 32B address
                resi_addr_ub = (resi_elem // self.elem_32b) * self.elem_32b
                resi_addr_gm = (total_elem // self.elem_32b) * self.elem_32b

                tik_instance.tensor_mov(replace_ub,
                                        self.var_out_gm[resi_addr_gm],
                                        '', 1, 1, 0, 0)
                tik_instance.tensor_mov(replace_ub_accum,
                                        self.accum_out_gm[resi_addr_gm],
                                        '', 1, 1, 0, 0)

                # don't influence the excessive copy
                with tik_instance.for_range(x32b_resi_num, self.elem_32b) as i:
                    copy_ub[resi_addr_ub + i].set_as(replace_ub[i])
                    copy_ub_accum[resi_addr_ub + i].set_as(replace_ub_accum[i])

            # results move out
            tik_instance.tensor_mov(self.var_out_gm[resi_addr], copy_ub,
                                    '', 1, copy_repeatr, 0, 0)
            tik_instance.tensor_mov(self.accum_out_gm[resi_addr], copy_ub_accum,
                                    '', 1, copy_repeatr, 0, 0)

    def _apply_compute_multi_core(self, tik_instance, idx_addr, idx_read_len,
                                  thread_num, core_id, elem_cur_core):

        # buffer size in ub, should be x128 fp16 (64 fp32)
        var_dim0_elem_ub = _round_up(elem_cur_core, self.elem_128fp)

        # copy 32B repeat numberfrom gm to ub
        mov_burst_num = _ceil_div(elem_cur_core, self.elem_32b)

        # vector calc 128 fp16 repeat number
        calc_repeat = var_dim0_elem_ub  // self.elem_128fp

        # calculation ub size for var, accum and grad
        var_shape_ub = (var_dim0_elem_ub, )

        x32b_resi_num = elem_cur_core % self.elem_32b

        with tik_instance.for_range(0, idx_read_len,
                                    thread_num=thread_num) as idx_id:

            # ub for var, accum, grad
            var_ub = tik_instance.Tensor(self.var_dtype, var_shape_ub,
                                         name="var_ub", scope=tik.scope_ubuf)
            accum_ub = tik_instance.Tensor(self.var_dtype, var_shape_ub,
                                           name="accum_ub", scope=tik.scope_ubuf)
            grad_ub = tik_instance.Tensor(self.var_dtype, var_shape_ub,
                                          name="grad_ub", scope=tik.scope_ubuf)

            # temporal calculation buffer, lenth is same as var_ub
            temp_ub = tik_instance.Tensor(self.var_dtype, var_shape_ub,
                                          name="temp_ub", scope=tik.scope_ubuf)
            lr1_ub = tik_instance.Tensor(self.var_dtype, var_shape_ub,
                                         name="lr1_ub", scope=tik.scope_ubuf)
            lr2_ub = tik_instance.Tensor(self.var_dtype, var_shape_ub,
                                         name="lr2_ub", scope=tik.scope_ubuf)

            prox_var_ub = tik_instance.Tensor(self.var_dtype, var_shape_ub,
                                              name="prox_var_ub",
                                              scope=tik.scope_ubuf)

            # index of var and accum, grad
            v_id = tik_instance.Scalar(self.idx_dtype, name="v_id")
            g_id = tik_instance.Scalar(self.idx_dtype, name="g_id")
            v_id.set_as(self.index_ub[idx_id])
            g_id.set_as(idx_id + idx_addr)

            # read data and calculation
            var_addr = tik_instance.Scalar("uint32", name="var_addr")
            grad_addr = tik_instance.Scalar("uint32", name="grad_addr")

            var_addr.set_as(v_id * self.var_dim0_elem_raw + \
                            core_id * self.elem_per_core)
            grad_addr.set_as(g_id * self.var_dim0_elem_raw + \
                             core_id * self.elem_per_core)

            # load input data
            tik_instance.tensor_mov(var_ub, self.var_gm[var_addr],
                                    '', 1, mov_burst_num, 0, 0)
            tik_instance.tensor_mov(accum_ub, self.accum_gm[var_addr],
                                    '', 1, mov_burst_num, 0, 0)
            tik_instance.tensor_mov(grad_ub, self.grad_gm[grad_addr],
                                    '', 1, mov_burst_num, 0, 0)

            # compatible with x32B mode
            self._core_compute_x32b(tik_instance, var_ub, accum_ub, grad_ub,
                                    temp_ub, lr1_ub, lr2_ub, prox_var_ub,
                                    calc_repeat)

            if x32b_resi_num > 0:
                mov_burst_num -= 1

            if mov_burst_num > 0:
                # results move out, except last data < 32B
                tik_instance.tensor_mov(self.var_gm[var_addr], var_ub,
                                        '', 1, mov_burst_num, 0, 0)

                tik_instance.tensor_mov(self.accum_gm[var_addr], accum_ub,
                                        '', 1, mov_burst_num, 0, 0)
            if x32b_resi_num > 0:
                if self.core_num > 1:
                    self._copy_resi_data_out_multi_core(
                        tik_instance, var_addr, var_ub, accum_ub,
                        elem_cur_core)
                else:
                    self._copy_resi_data_out_single_core(
                        tik_instance, var_addr, var_ub, accum_ub,
                        elem_cur_core)

    def _copy_resi_data_out_multi_core(self, tik_instance, var_addr,
                                       var_ub, accum_ub, elem_cur_core):
        # copy residual data <32B out
        # each core process >= 32B for multi-core mode
        x32b_resi_num = elem_cur_core % self.elem_32b

        var_ub_resi = tik_instance.Tensor(self.var_dtype, (self.elem_32b,),
                                          name="var_ub_resi",
                                          scope=tik.scope_ubuf)
        accum_ub_resi = tik_instance.Tensor(self.var_dtype, (self.elem_32b,),
                                            name="accum_ub_resi",
                                            scope=tik.scope_ubuf)

        # move last data < 32B out
        gm_offset = elem_cur_core - self.elem_32b

        var_addr.set_as(var_addr + gm_offset)

        tik_instance.tensor_mov(var_ub_resi, self.var_gm[var_addr],
                                '', 1, 1, 0, 0)
        tik_instance.tensor_mov(accum_ub_resi, self.accum_gm[var_addr],
                                '', 1, 1, 0, 0)

        ub_addr = elem_cur_core // self.elem_32b * self.elem_32b
        ub_offset = self.elem_32b - x32b_resi_num
        # don't influence the excessive copy
        with tik_instance.for_range(0, x32b_resi_num) as i:
            var_ub_resi[ub_offset + i].set_as(var_ub[ub_addr + i])
            accum_ub_resi[ub_offset + i].set_as(accum_ub[ub_addr + i])

        # move the residual data out
        tik_instance.tensor_mov(self.var_gm[var_addr], var_ub_resi,
                                '', 1, 1, 0, 0)

        tik_instance.tensor_mov(self.accum_gm[var_addr], accum_ub_resi,
                                '', 1, 1, 0, 0)

    def _copy_resi_data_out_single_core(self, tik_instance, var_addr,
                                        var_ub, accum_ub, elem_cur_core):
        # copy residual data <32B out
        x32b_resi_num = elem_cur_core % self.elem_32b

        var_ub_resi = tik_instance.Tensor(self.var_dtype, (self.elem_32b,),
                                          name="var_ub_resi",
                                          scope=tik.scope_ubuf)
        accum_ub_resi = tik_instance.Tensor(self.var_dtype, (self.elem_32b,),
                                            name="accum_ub_resi",
                                            scope=tik.scope_ubuf)

        # move last data < 32B out
        gm_offset = elem_cur_core // self.elem_32b * self.elem_32b

        var_addr.set_as(var_addr + gm_offset)

        tik_instance.tensor_mov(var_ub_resi, self.var_gm[var_addr],
                                '', 1, 1, 0, 0)
        tik_instance.tensor_mov(accum_ub_resi, self.accum_gm[var_addr],
                                '', 1, 1, 0, 0)

        ub_addr = elem_cur_core // self.elem_32b * self.elem_32b

        # don't influence the excessive copy
        with tik_instance.for_range(0, x32b_resi_num) as i:
            var_ub_resi[i].set_as(var_ub[ub_addr + i])
            accum_ub_resi[i].set_as(accum_ub[ub_addr + i])

        # move the residual data out
        tik_instance.tensor_mov(self.var_gm[var_addr], var_ub_resi,
                                '', 1, 1, 0, 0)

        tik_instance.tensor_mov(self.accum_gm[var_addr], accum_ub_resi,
                                '', 1, 1, 0, 0)

    # core vector element-wise compute of the operator
    # pylint: disable=too-many-arguments
    def _core_compute_x32b(self, tik_instance, var_ub, accum_ub, grad_ub,
                           temp_ub, lr1_ub, lr2_ub, prox_var_ub, rep_num):

        tik_instance.vmla(self.elem_128fp, accum_ub, grad_ub, grad_ub,
                          rep_num, 1, 1, 1, 8, 8, 8)

        # precision is higher but don't support "mini"
        tik_instance.vsqrt(self.elem_128fp, temp_ub, accum_ub,
                           rep_num, 1, 1, 8, 8)
        tik_instance.vdiv(self.elem_128fp, lr1_ub, self.lr_vec_ub, temp_ub,
                          rep_num, 1, 1, 1, 8, 0, 8)

        tik_instance.vmul(self.elem_128fp, temp_ub, lr1_ub, grad_ub,
                          rep_num, 1, 1, 1, 8, 8, 8)
        tik_instance.vsub(self.elem_128fp, prox_var_ub, var_ub, temp_ub,
                          rep_num, 1, 1, 1, 8, 8, 8)

        tik_instance.vmul(self.elem_128fp, lr2_ub, self.l2_vec_ub, lr1_ub,
                          rep_num, 1, 1, 1, 8, 0, 8)
        tik_instance.vadd(self.elem_128fp, temp_ub, self.one_vec_ub, lr2_ub,
                          rep_num, 1, 1, 1, 8, 0, 8)
        tik_instance.vdiv(self.elem_128fp, lr2_ub, self.one_vec_ub, temp_ub,
                          rep_num, 1, 1, 1, 8, 0, 8)

        # if l1_scalar_1 and zero_scalar, compiling error happens
        with tik_instance.if_scope(self.l1_scalar_1 > self.zero_scalar):

            # reuse grad_ub as (lr1 * l1)
            tik_instance.vmul(self.elem_128fp, grad_ub, lr1_ub,
                              self.l1_vec_ub,
                              rep_num, 1, 1, 1, 8, 8, 0)

            # reuse lr1_ub as (abs(prox_var))
            tik_instance.vabs(self.elem_128fp, lr1_ub, prox_var_ub,
                              rep_num, 1, 1, 8, 8)
            tik_instance.vsub(self.elem_128fp, temp_ub, lr1_ub, grad_ub,
                              rep_num, 1, 1, 1, 8, 8, 8)

            # reuse grad_ub as (maximum(var_t1, 0.0))
            tik_instance.vmax(self.elem_128fp, grad_ub, temp_ub,
                              self.zero_vec_ub,
                              rep_num, 1, 1, 1, 8, 8, 0)
            # reuse lr1_ub as maximum(var_t1, 0.0) * lr2
            tik_instance.vmul(self.elem_128fp, lr1_ub, grad_ub, lr2_ub,
                              rep_num, 1, 1, 1, 8, 8, 8)

            with tik_instance.for_range(0, rep_num) as rcnt:
                rep_addr = tik_instance.Scalar("uint32", name="rep_addr")
                rep_addr.set_as(rcnt * self.elem_128fp)

                # no repeat, thus requre a for loop
                # the order of following sentences are not changable
                cmpmask_p = tik_instance.vcmp_gt(self.elem_128fp,
                                                 prox_var_ub[rep_addr],
                                                 self.zero_vec_ub, 1, 1)
                tik_instance.vsel(self.elem_128fp, 0, temp_ub[rep_addr],
                                  cmpmask_p,
                                  self.one_vec_ub, self.zero_vec_ub,
                                  1, 1, 1, 1, 0, 0, 0)

                cmpmask_n = tik_instance.vcmp_lt(self.elem_128fp,
                                                 prox_var_ub[rep_addr],
                                                 self.zero_vec_ub, 1, 1)
                tik_instance.vsel(self.elem_128fp, 0, prox_var_ub[rep_addr],
                                  cmpmask_n,
                                  self.neg1_vec_ub, temp_ub[rep_addr],
                                  1, 1, 1, 1, 0, 0, 0)

            tik_instance.vmul(self.elem_128fp, var_ub, prox_var_ub, lr1_ub,
                              rep_num, 1, 1, 1, 8, 8, 8)

        with tik_instance.else_scope():
            tik_instance.vmul(self.elem_128fp, var_ub, prox_var_ub, lr2_ub,
                              rep_num, 1, 1, 1, 8, 8, 8)


def _get_elem_num(shape):
    total_elem_num = 1
    for elem in shape:
        total_elem_num *= elem
    return total_elem_num


def _dtype_size(dtype):

    mem_size = {
        "float8": 1,
        "float16": 2,
        "float32": 4,

        "int8": 1,
        "int16": 2,
        "int32": 4,
        "int64": 8,

        "uint8": 1,
        "uint16": 2,
        "uint32": 4,
        "uint64": 8,

        "float": 4,
        "int": 4,
    }

    dtype = dtype.lower()
    byte_size = mem_size.get(dtype)
    return byte_size


def _block_elem(dtype):
    dsize = _dtype_size(dtype)
    elem = 32 // dsize # 32B per block
    return elem


def _v128_elem(dtype):
    dsize = _dtype_size(dtype)
    elem = 128 * 2 // dsize # 128 fp16
    return elem


def _round_up(length, round_base):
    return ((length + round_base - 1) // round_base) * round_base


def _ceil_div(divided, to_div):
    return (divided + to_div - 1) // to_div


def _get_thread_num(task_num):
    thread_num = 2
    if task_num < 2:
        thread_num = 1
    return thread_num
