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

sparse_apply_ftrl
"""

from topi.cce import util
from impl.sparse_apply_common import SparseApply

SHAPE_SIZE_LIMIT = (2**29 - 1)

class SparseApplyFtrl(SparseApply):
    """
    Sub class inherited form SparseApply for sparse_apply_ftrl op
    """

    def __init__(self, var, accum, linear, grad, indices,
                 lr, lr_power, l1, l2, l2_shrinkage, kernel_name):
        """
        init sparse_apply_ftrl  base parameters

        Parameters
        ----------
        lr: float
            scalar
        lr_power: float
            scalar
        l1: float
            scalar
        l2: float
            scalar
        l2_shrinkage: float
            scalar

        Returns
        -------
        None
        """
        super().__init__(var, grad, indices, kernel_name)
        self.lr = lr
        self.lr_power = lr_power
        self.l1 = l1
        self.l2 = l2
        self.l2_shrinkage = l2_shrinkage
        self.lr_vrec = 1.0 / self.lr
        self.var_shape = var.get("shape")
        self.var_dtype = var.get("dtype").lower()
        self.accum_shape = accum.get("shape")
        self.accum_dtype = accum.get("dtype").lower()
        self.linear_shape = linear.get("shape")
        self.linear_dtype = linear.get("dtype").lower()
        self.check_param()

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
        util.check_shape_rule(self.var_shape)
        util.check_shape_rule(self.accum_shape)
        util.check_shape_rule(self.linear_shape)
        util.check_shape_size(self.var_shape, SHAPE_SIZE_LIMIT)
        util.check_shape_size(self.accum_shape, SHAPE_SIZE_LIMIT)
        util.check_shape_size(self.linear_shape, SHAPE_SIZE_LIMIT)

        check_list_var_dtype = ("float32")
        util.check_dtype_rule(self.var_dtype, check_list_var_dtype)
        util.check_dtype_rule(self.accum_dtype, check_list_var_dtype)
        util.check_dtype_rule(self.linear_dtype, check_list_var_dtype)

        if self.accum_shape != self.var_shape:
            raise RuntimeError("accum's shape must be the same as var's shape")

        if self.linear_shape != self.var_shape:
            raise RuntimeError(
                "linear's shape must be the same as var's shape")

    def calc(self, repeat, mask, offset):
        """
        calculate data according to the Ftrl-proximal scheme
        will automated called by basic class function

        Parameters
        ----------
        repeat: repeat times of insn
        mask: mask for vector insn
        offset: offset of ub addr

        Returns
        -------
        None
        """
        tmp_ub = self.get_ub("tmp_ub")[offset]
        tmp2_ub = self.get_ub("tmp2_ub")[offset]
        if self.each_row_data_num <= self.cache_threshold_col:
            var_ub = self.get_ub("var_align_ub")[offset]
            accum_ub = self.get_ub("accum_align_ub")[offset]
            linear_ub = self.get_ub("linear_align_ub")[offset]
            grad_ub = self.grad_align_ub[offset]
        else:
            var_ub = self.get_ub("var_ub")[offset]
            accum_ub = self.get_ub("accum_ub")[offset]
            linear_ub = self.get_ub("linear_ub")[offset]
            grad_ub = self.grad_ub[offset]

        if self.l2_shrinkage != 0:
            self.tik_instance.vmuls(mask, tmp_ub, var_ub, 2 * self.l2_shrinkage,
                                    repeat, 1, 1, 8, 8)
            self.tik_instance.vadd(mask, tmp2_ub, grad_ub, tmp_ub,
                                   repeat, 1, 1, 1, 8, 8, 8)

        # tmp =  grad*grad
        self.tik_instance.vmul(mask, tmp_ub, grad_ub, grad_ub,
                               repeat, 1, 1, 1, 8, 8, 8)
        # linear += grad, grad will not used after this operation
        if self.l2_shrinkage != 0:
            self.tik_instance.vadd(mask, linear_ub, tmp2_ub, linear_ub,
                                   repeat, 1, 1, 1, 8, 8, 8)
        else:
            self.tik_instance.vadd(mask, linear_ub, grad_ub, linear_ub,
                                   repeat, 1, 1, 1, 8, 8, 8)
        # a^b = e^(b*lna)
        # grad = ln(accum)
        self.tik_instance.vln(mask, grad_ub, accum_ub,
                              repeat, 1, 1, 8, 8)
        # grad = -lr_power*ln(accum)
        self.tik_instance.vmuls(mask, grad_ub, grad_ub, -self.lr_power,
                                repeat, 1, 1, 8, 8)
        # grad = e^(-lr_power*ln(accum)) = accum ^ (-lr_power)
        self.tik_instance.vexp(mask, grad_ub, grad_ub,
                               repeat, 1, 1, 8, 8)

        # accum += grad*grad
        self.tik_instance.vadd(mask, accum_ub, accum_ub, tmp_ub,
                               repeat, 1, 1, 1, 8, 8, 8)
        # tmp = ln(accum_new)
        self.tik_instance.vln(mask, tmp_ub, accum_ub,
                              repeat, 1, 1, 8, 8)

        # accum_new^(-lr_power)
        self.tik_instance.vmuls(mask, tmp_ub, tmp_ub, -self.lr_power,
                                repeat, 1, 1, 8, 8)

        # tmp = accum_new ^ (-lr_power)
        self.tik_instance.vexp(mask, tmp_ub, tmp_ub,
                               repeat, 1, 1, 8, 8)

        # tmp2 =accum_new^(-lr_power)/lr, used by y_res
        self.tik_instance.vmuls(mask, tmp2_ub, tmp_ub, self.lr_vrec,
                                repeat, 1, 1, 8, 8)

        # tmp = accum^(-lr_power)- accum_new^(-lr_power)
        self.tik_instance.vsub(mask, tmp_ub, grad_ub, tmp_ub,
                               repeat, 1, 1, 1, 8, 8, 8)

        # tmp = tmp / lr
        self.tik_instance.vmuls(mask, tmp_ub, tmp_ub, self.lr_vrec,
                                repeat, 1, 1, 8, 8)

        # tmp = tmp / lr * var
        self.tik_instance.vmul(mask, tmp_ub, tmp_ub, var_ub,
                               repeat, 1, 1, 1, 8, 8, 8)

        # linear += grad +(accum^(-lr_power)-accum_new^(-lr_power))/lr*var
        self.tik_instance.vadd(mask, linear_ub, tmp_ub, linear_ub,
                               repeat, 1, 1, 1, 8, 8, 8)

        # x_res=linear.min(l1).max(-l1)-linear

        self.tik_instance.vector_dup(mask, tmp_ub, self.l1,
                                     repeat, 1, 8)
        self.tik_instance.vmin(mask, grad_ub, linear_ub, tmp_ub,
                               repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vector_dup(mask, tmp_ub, -self.l1,
                                     repeat, 1, 8)
        self.tik_instance.vmax(mask, tmp_ub, grad_ub, tmp_ub,
                               repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, tmp_ub, tmp_ub, linear_ub,
                               repeat, 1, 1, 1, 8, 8, 8)

        # y_res = accum_new^(-lr_power)/lr + 2*l2

        self.tik_instance.vadds(mask, tmp2_ub, tmp2_ub, 2 * self.l2,
                                repeat, 1, 1, 8, 8)

        # var = x_res/y_res
        self.tik_instance.vdiv(mask, var_ub, tmp_ub, tmp2_ub,
                               repeat, 1, 1, 1, 8, 8, 8)


# pylint: disable=too-many-arguments,unused-argument,locally-disabled
# pylint: disable=too-many-branches,too-many-instance-attributes,too-many-locals
# pylint: disable=attribute-defined-outside-init,invalid-name,,too-many-lines
@util.check_input_type(dict, dict, dict, dict, dict, dict, dict,dict,(float, int),
                       (float, int),
                       (float, int), (float, int), (float, int),
                       bool, str)
def sparse_apply_ftrl_v2_d(var,
                           accum,
                           linear,
                           grad,
                           indices,
                           out_var,
                           out_accum,
                           out_linear,
                           lr, l1, l2, l2_shrinkage, lr_power,
                           use_locking=False,
                           kernel_name="sparse_apply_ftrl_v2"):
    """
    Update the variable referenced by resource.

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
    use_locking: bool
        not used in this compute
    kernel_name: str
        kernel name, default value is "sparse_apply_ftrl"

    Returns:
    None
    """
    apply_ftrl = SparseApplyFtrl(var, accum, linear, grad, indices,
                                 lr, lr_power, l1, l2, l2_shrinkage, kernel_name)
    var_shape = var.get('shape')
    apply_ftrl.add_input("var_in_gm", "float32", var_shape)
    apply_ftrl.add_input("accum_in_gm", "float32", var_shape)
    apply_ftrl.add_input("linear_in_gm", "float32", var_shape)
    apply_ftrl.add_output("var_out_gm", "float32", var_shape)
    apply_ftrl.add_output("accum_out_gm", "float32", var_shape)
    apply_ftrl.add_output("linear_out_gm", "float32", var_shape)
    apply_ftrl.reserve_ub("var_ub", "float32", "var_align_ub")
    apply_ftrl.reserve_ub("accum_ub", "float32", "accum_align_ub")
    apply_ftrl.reserve_ub("linear_ub", "float32", "linear_align_ub")
    apply_ftrl.reserve_ub("tmp_ub", "float32")
    apply_ftrl.reserve_ub("tmp2_ub", "float32")
    apply_ftrl.set_var_rows(var_shape[0])
    apply_ftrl.sparse_apply_operator()
