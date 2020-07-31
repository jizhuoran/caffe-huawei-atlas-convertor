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

nll_loss_grad
"""
# pylint: disable=ungrouped-imports,import-error
import math
from te import tik
from topi.cce import util
from te import platform as tbe_platform
from impl.constant_util import MASK64
from impl.constant_util import DATA_SIZE_EIGHT

DIM2 = 2
NUM_EIGHT = 8
NUM_FOUR = 4
NEGATIVE = -1
MAX_REPEAT = 255
ONE_KB = 1024
NUM_SIXTYFOUR = MASK64


# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
# pylint: disable=too-many-arguments
def _shape_and_dtype_check(x, y_grad, target, weight, total_weight, reduction,
                           kernel_name):
    x_shape = x.get("shape")
    x_dtype = x.get("dtype").lower()
    y_grad_shape = y_grad.get("shape")
    y_grad_dtype = y_grad.get("dtype").lower()
    target_shape = target.get("shape")
    target_dtype = target.get("dtype").lower()
    total_weight_shape = total_weight.get("shape")
    total_weight_dtype = total_weight.get("dtype").lower()
    weight_shape = weight.get("shape")
    weight_dtype = weight.get("dtype").lower()
    util.check_tensor_shape_size(weight_shape)
    util.check_shape_rule(weight_shape)

    util.check_shape_rule(x_shape)
    util.check_shape_rule(y_grad_shape)
    util.check_shape_rule(target_shape)
    util.check_tensor_shape_size(y_grad_shape)
    util.check_tensor_shape_size(target_shape)

    util.check_kernel_name(kernel_name)
    util.check_dtype_rule(x_dtype, "float32")
    util.check_dtype_rule(y_grad_dtype, "float32")
    util.check_dtype_rule(target_dtype, "int32")
    util.check_dtype_rule(weight_dtype, "float32")
    util.check_dtype_rule(total_weight_dtype, "float32")

    if reduction in ("mean", "sum") and y_grad_shape[0] != 1:
        raise RuntimeError("The shape of y_grad must be (1,),"
                           " while reduction is mean or sum. ")
    if len(x_shape) == 1 and y_grad_shape[0] != 1:
        raise RuntimeError("The shape of y_grad must be (1,),"
                           " while input x is 1D. ")
    if len(x_shape) > DIM2:
        raise RuntimeError("The dimension of x should be equal to"
                           "or less than two.")
    if len(x_shape) == DIM2 and x_shape[0] != target_shape[0]:
        raise RuntimeError("The first dimension of x and"
                           " target should be equal")
    if x_shape[-1] != weight_shape[0]:
        raise RuntimeError("The last dimension of x and the first dimension"
                           " of weight should be equal")
    if len(y_grad_shape) != 1:
        raise RuntimeError("The dimension of y_grad should be 1D.")
    if len(weight_shape) != 1:
        raise RuntimeError("The dimension of weight should be 1D.")
    if len(target_shape) != 1:
        raise RuntimeError("The dimension of target should be 1D.")
    if total_weight_shape[0] != 1:
        raise RuntimeError("The shape of total_weight must be (1,)")


# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-many-statements
# pylint: disable=attribute-defined-outside-init
class nll_loss_grad_compute:
    """
    NLLLOSSGRAD

    Returns
    -------
    None
    """
    def __init__(self, x, y_grad, target, weight, reduction, kernel_name):
        self.init_tik_instance()
        self.reduction = reduction
        self.kernel_name = kernel_name
        self.x_shape = x.get("shape")
        self.x_dtype = x.get("dtype").lower()
        self.y_grad_shape = y_grad.get("shape")
        self.y_grad_dtype = y_grad.get("dtype").lower()
        self.target_shape = target.get("shape")
        self.target_dtype = target.get("dtype").lower()
        self.weight_shape = weight.get("shape")
        self.weight_dtype = weight.get("dtype").lower()
        self.x_dim = len(self.x_shape)
        self.init_size()
        self.init_gm()

    def init_tik_instance(self):
        """
        init the tik_instance

        Parameters
        ----------

        Returns
        -------
        None
        """
        profile = tik.Dprofile()
        self.tik_instance = tik.Tik(profile)

    def init_size(self):
        """
        init the size of args.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.core_num = self.x_shape[0]
        self.ub_size_bytes = tbe_platform.CceProductParams().getParams(
            "Unified_Buffer") - ONE_KB
        if len(self.x_shape) == DIM2:
            self.output_gm_size = self.x_shape[0] * self.x_shape[1]
        else:
            self.output_gm_size = self.x_shape[0]
        self.c_dim = self.x_shape[-1]
        self.repeat_time = math.ceil(self.c_dim / NUM_SIXTYFOUR)
        self.move_len = math.ceil(self.c_dim / NUM_EIGHT)
        if self.c_dim > NUM_EIGHT:
            self.end_num = self.c_dim % NUM_EIGHT
        self.y_grad_ub_size = math.ceil(self.y_grad_shape[0]/NUM_SIXTYFOUR) * \
                              NUM_SIXTYFOUR
        self.target_ub_size = math.ceil(self.target_shape[0]/NUM_SIXTYFOUR) * \
                              NUM_SIXTYFOUR
        self.total_weight_ub_size = NUM_SIXTYFOUR
        self.weight_ub_size = math.ceil(self.weight_shape[0]/NUM_SIXTYFOUR) * \
                              NUM_SIXTYFOUR
        self.dup_ub_size = math.ceil(self.c_dim/NUM_SIXTYFOUR)*NUM_SIXTYFOUR
        self.y_grad_gm_size = self.y_grad_shape[0]
        self.target_gm_size = self.target_shape[0]
        self.total_weight_gm_size = 1
        self.weight_gm_size = self.weight_shape[0]

    def init_gm(self):
        """
        init the gm of input and output.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.data_x = self.tik_instance.Tensor(self.y_grad_dtype,
                                               [self.y_grad_gm_size],
                                               name="data_x",
                                               scope=tik.scope_gm)
        self.y_grad = self.tik_instance.Tensor(self.y_grad_dtype,
                                               [self.y_grad_gm_size],
                                               name="y_grad",
                                               scope=tik.scope_gm)

        self.data_target = self.tik_instance.Tensor(self.target_dtype,
                                                    [self.target_gm_size],
                                                    name="data_target",
                                                    scope=tik.scope_gm)
        self.total_weight_gm = self.tik_instance.Tensor(
            self.x_dtype, [self.total_weight_gm_size], name="total_weight_gm",
            scope=tik.scope_gm)
        self.data_weight = self.tik_instance.Tensor(self.weight_dtype,
                                                    [self.weight_gm_size],
                                                    name="data_weight",
                                                    scope=tik.scope_gm)
        self.output = self.tik_instance.Tensor(self.x_dtype,
                                               [self.output_gm_size],
                                               name="output",
                                               scope=tik.scope_gm)

    def init_ub(self):
        """
        init the ub of input and output.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.y_grad_ub = self.tik_instance.Tensor("float32",
                                                  [self.y_grad_ub_size],
                                                  name="y_grad_ub",
                                                  scope=tik.scope_ubuf)
        self.target_ub = self.tik_instance.Tensor(self.weight_dtype,
                                                  [self.target_ub_size],
                                                  name="target_ub",
                                                  scope=tik.scope_ubuf)
        self.total_weight_ub = self.tik_instance.Tensor(
            self.x_dtype, [self.total_weight_ub_size], name="total_weight_ub",
            scope=tik.scope_ubuf)
        self.weight_ub = self.tik_instance.Tensor(self.weight_dtype,
                                                  [self.weight_ub_size],
                                                  name="weight_ub",
                                                  scope=tik.scope_ubuf)

        self.dup_ub = self.tik_instance.Tensor(self.x_dtype,
                                               [self.dup_ub_size],
                                               name="dup_ub",
                                               scope=tik.scope_ubuf)
        self.index_x = self.tik_instance.Scalar(dtype="int32")

    def one_dim_compute(self):
        """
        calculate process when input is 1D.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.init_ub()
        self.tik_instance.data_move(self.y_grad_ub, self.y_grad,
                                    1, 1, 1, 0, 0)
        self.tik_instance.data_move(self.target_ub, self.total_weight_gm,
                                    1, 1, 1, 0, 0)
        self.tik_instance.data_move(self.target_ub, self.data_target,
                                    1, 1, 1, 0, 0)
        self.index_x.set_as(self.target_ub(0))
        self.tik_instance.data_move(self.weight_ub,
                                    self.data_weight[self.index_x],
                                    1, 1, 1, 0, 0)
        self.tik_instance.vmul(1, self.weight_ub, self.weight_ub,
                               self.y_grad_ub, 1, 1, 1, 1, 1, 0, 0)
        self.tik_instance.vmuls(1, self.weight_ub, self.weight_ub,
                                NEGATIVE, 1, 1, 1, 0, 0)
        self.tik_instance.vector_dup(MASK64, self.dup_ub, 0,
                                     self.repeat_time, 1, NUM_EIGHT)
        self.dup_ub(self.index_x).set_as(self.weight_ub[0])
        self.tik_instance.data_move(self.output, self.dup_ub,
                                    0, 1, self.move_len, 0, 0)

    def two_dim_compute(self):
        """
        calculate process when x is 2D and the shape of weight
        bigger lower than eight.

        Parameters
        ----------

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.core_num,
                                         block_num=self.core_num) as cycle:
            self.temp_ub = self.tik_instance.Tensor(self.x_dtype, [NUM_EIGHT],
                                                    name="temp_ub",
                                                    scope=tik.scope_ubuf)
            self.init_ub()
            if self.reduction == "none":
                self.tik_instance.data_move(self.y_grad_ub, self.y_grad[cycle],
                                            0, 1, 1, 0, 0)
            else:
                self.tik_instance.data_move(self.y_grad_ub, self.y_grad,
                                            0, 1, 1, 0, 0)
            y_grad = self.tik_instance.Scalar(dtype="float32")
            y_grad.set_as(self.y_grad_ub[0])
            self.tik_instance.data_move(self.target_ub, self.data_target[cycle],
                                        0, 1, 1, 0, 0)
            self.index_x.set_as(self.target_ub[0])
            self.tik_instance.vector_dup(MASK64, self.dup_ub,
                                         0, self.repeat_time, 1, NUM_EIGHT)
            self.tik_instance.data_move(self.weight_ub,
                                        self.data_weight[self.index_x],
                                        0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.total_weight_ub,
                                        self.total_weight_gm,
                                        0, 1, 1, 0, 0)
            if self.reduction == "mean":
                self.tik_instance.vdiv(1, self.weight_ub, self.weight_ub,
                                       self.total_weight_ub, 1, 1, 1, 1,
                                       NUM_EIGHT, NUM_EIGHT, NUM_EIGHT)
            self.dup_ub(self.index_x).set_as(self.weight_ub[0])
            self.tik_instance.vmuls(MASK64, self.dup_ub, self.dup_ub, NEGATIVE,
                                    self.repeat_time, 1, 1,
                                    NUM_EIGHT, NUM_EIGHT)
            self.tik_instance.vmuls(MASK64, self.dup_ub, self.dup_ub, y_grad,
                                    self.repeat_time, 1, 1,
                                    NUM_EIGHT, NUM_EIGHT)

            if self.c_dim > NUM_EIGHT and self.end_num != 0:
                for i in range(0, NUM_EIGHT):
                    self.temp_ub[i].set_as(
                        self.dup_ub[self.c_dim - NUM_EIGHT + i])
                self.tik_instance.data_move(self.output[cycle * self.c_dim],
                                            self.dup_ub, 0, 1,
                                            self.move_len - 1, 0, 0)
                self.tik_instance.data_move(
                    self.output[cycle * self.c_dim + self.c_dim - NUM_EIGHT],
                    self.temp_ub, 0, 1, 1, 0, 0)
            else:
                self.tik_instance.data_move(self.output[cycle * self.c_dim],
                                            self.dup_ub, 0, 1,
                                            self.move_len, 0, 0)

    def weight_no_32b_tiling(self):
        """
        calculate weight lower 32b tiling

        Parameters
        ----------

        Returns
        -------
        None
        """
        shape = self.x_shape
        if self.reduction == "none":
            max_align_32b_ub_size = ((self.ub_size_bytes/(self.x_shape[1]+3))
                                     // 32)*32
        else:
            max_align_32b_ub_size = ((self.ub_size_bytes/(self.x_shape[1]+2))
                                     // 32)*32
        self.max_move_num = int(max_align_32b_ub_size//DATA_SIZE_EIGHT//64)*64
        self.move_times = math.ceil(shape[0]/self.max_move_num)
        self.last_move_num = shape[0] - self.max_move_num*(self.move_times - 1)
        self.max_dup_repeat = math.ceil((self.max_move_num *
                                         self.x_shape[1])/NUM_SIXTYFOUR)
        self.last_dup_repeat = math.ceil((self.last_move_num *
                                          self.x_shape[1])/NUM_SIXTYFOUR)
        self.max_burst_len = int(self.max_move_num//NUM_EIGHT)
        self.max_vmul_repeat = math.ceil(self.max_burst_len/NUM_EIGHT)
        if self.reduction == "none":
            self.max_vmul_repeat = math.ceil(self.x_shape[0]/NUM_SIXTYFOUR)
        if self.last_move_num != 0:
            self.last_move_burst_len = math.ceil(self.last_move_num/NUM_EIGHT)
        else:
            self.last_move_num = self.max_move_num
            self.last_move_burst_len = self.max_burst_len
        self.last_vmul_repeat = math.ceil(self.last_move_burst_len *
                                          self.x_shape[1]/NUM_EIGHT)

    def _calculate_process_none(self, move_num, max_move_num, vmul_repeat,
                                dup_repeat, burst_len, cycle):
        MAX_REPEAT_NUM = MAX_REPEAT*NUM_SIXTYFOUR
        self.tik_instance.data_move(self.total_weight_ub,
                                    self.total_weight_gm, 0, 1, 1, 0, 0)
        total_weight = self.tik_instance.Scalar(dtype="float32")
        total_weight.set_as(self.total_weight_ub[0])
        self.tik_instance.data_move(self.weight_ub, self.data_weight,
                                    0, 1, 1, 0, 0)
        self.temp_ub = self.tik_instance.Tensor(
            self.x_dtype, [math.ceil(move_num*self.x_shape[1]/64)*64],
            name="temp_ub", scope=tik.scope_ubuf)
        self.temp_out_ub = self.tik_instance.Tensor(
            self.x_dtype, [math.ceil(move_num*self.x_shape[1]/64)*64],
            name="temp_out_ub", scope=tik.scope_ubuf)
        repeat_times = math.ceil(dup_repeat/MAX_REPEAT)
        with self.tik_instance.for_range(0, repeat_times-1, thread_num=1) as index:
            self.tik_instance.vector_dup(MASK64,
                                         self.temp_out_ub[index*MAX_REPEAT_NUM],
                                         0, MAX_REPEAT, 1, NUM_EIGHT)
        if dup_repeat <= MAX_REPEAT:
            self.tik_instance.vector_dup(
                MASK64, self.temp_out_ub[(repeat_times-1)*MAX_REPEAT_NUM], 0,
                dup_repeat, 1, NUM_EIGHT)

        self.tik_instance.data_move(self.y_grad_ub,
                                    self.y_grad[cycle*move_num],
                                    0, 1, burst_len, 0, 0)
        self.tik_instance.data_move(self.target_ub,
                                    self.data_target[cycle*move_num],
                                    0, 1, burst_len, 0, 0)
        with self.tik_instance.for_range(0, move_num, thread_num=1) as index:
            self.index_x.set_as(self.target_ub[index])
            self.temp_ub[index].set_as(self.weight_ub[self.index_x])
        vmul_repeat_times = math.ceil(vmul_repeat/MAX_REPEAT)
        for i in range(vmul_repeat_times-1):
            self.tik_instance.vmul(
                MASK64, self.temp_ub[i*MAX_REPEAT_NUM],
                self.temp_ub[i*MAX_REPEAT_NUM],
                self.y_grad_ub[i*MAX_REPEAT_NUM],
                MAX_REPEAT, 1, 1, 1, NUM_EIGHT, NUM_EIGHT, NUM_EIGHT)
        if vmul_repeat <= MAX_REPEAT:
            self.tik_instance.vmul(
                MASK64, self.temp_ub, self.temp_ub, self.y_grad_ub,
                vmul_repeat, 1, 1, 1, NUM_EIGHT, NUM_EIGHT, NUM_EIGHT)
        with self.tik_instance.for_range(0, move_num, thread_num=1) as index:
            self.index_x.set_as(self.target_ub[index])
            self.temp_out_ub[index*self.x_shape[-1]+self.index_x].set_as(
                self.temp_ub[index])
        self.tik_instance.data_move(
            self.output[cycle*max_move_num], self.temp_out_ub,
            0, 1, burst_len*self.x_shape[1], 0, 0)

    def _calculate_process_sum_and_mean(self, move_num, max_move_num,
                                        vmul_repeat, dup_repeat,
                                        burst_len, cycle):
        MAX_REPEAT_NUM = MAX_REPEAT*NUM_SIXTYFOUR
        self.temp_out_ub = self.tik_instance.Tensor(
            self.x_dtype, [math.ceil(move_num*self.x_shape[1]/NUM_SIXTYFOUR) *
                           NUM_SIXTYFOUR],
            name="temp_out_ub", scope=tik.scope_ubuf)
        temp_y_grad = self.tik_instance.Scalar(dtype="float32")
        self.tik_instance.data_move(self.y_grad_ub, self.y_grad, 0, 1, 1, 0, 0)
        self.tik_instance.data_move(self.weight_ub, self.data_weight,
                                    0, 1, 1, 0, 0)

        repeat_times = math.ceil(dup_repeat/MAX_REPEAT)
        for i in range(repeat_times-1):
            self.tik_instance.vector_dup(MASK64,
                                         self.temp_out_ub[i*MAX_REPEAT_NUM],
                                         0, MAX_REPEAT, 1, NUM_EIGHT)
        if dup_repeat <= MAX_REPEAT:
            self.tik_instance.vector_dup(
                MASK64, self.temp_out_ub[(repeat_times-1)*MAX_REPEAT_NUM], 0,
                dup_repeat, 1, NUM_EIGHT)

        temp_y_grad.set_as(self.y_grad_ub[0])
        self.tik_instance.data_move(self.target_ub,
                                    self.data_target[cycle*max_move_num],
                                    0, 1, burst_len, 0, 0)
        with self.tik_instance.for_range(0, move_num, thread_num=1) as index:
            self.index_x.set_as(self.target_ub[index])
            self.temp_out_ub[index*self.x_shape[-1]+self.index_x].set_as(
                self.weight_ub[self.index_x])

        vmul_repeat_times = math.ceil(vmul_repeat/MAX_REPEAT)
        for i in range(vmul_repeat_times-1):
            self.tik_instance.vmuls(MASK64, self.temp_out_ub[i*MAX_REPEAT_NUM],
                                    self.temp_out_ub[i*MAX_REPEAT_NUM],
                                    temp_y_grad, MAX_REPEAT, 1, 1, NUM_EIGHT,
                                    NUM_EIGHT)
            self.tik_instance.vmuls(MASK64, self.temp_out_ub[i*MAX_REPEAT_NUM],
                                    self.temp_out_ub[i*MAX_REPEAT_NUM],
                                    NEGATIVE, MAX_REPEAT, 1, 1,
                                    NUM_EIGHT, NUM_EIGHT)
        if vmul_repeat <= MAX_REPEAT:
            self.tik_instance.vmuls(
                MASK64, self.temp_out_ub[(vmul_repeat_times-1)*MAX_REPEAT_NUM],
                self.temp_out_ub[(vmul_repeat_times-1)*MAX_REPEAT_NUM],
                temp_y_grad, vmul_repeat, 1, 1, NUM_EIGHT, NUM_EIGHT)
            self.tik_instance.vmuls(
                MASK64, self.temp_out_ub[(vmul_repeat_times-1)*MAX_REPEAT_NUM],
                self.temp_out_ub[(vmul_repeat_times-1)*MAX_REPEAT_NUM],
                NEGATIVE, vmul_repeat, 1, 1, NUM_EIGHT, NUM_EIGHT)
        self.tik_instance.data_move(self.total_weight_ub,
                                    self.total_weight_gm, 0, 1, 1, 0, 0)
        total_weight = self.tik_instance.Scalar(dtype="float32")
        total_weight.set_as(self.total_weight_ub[0])
        if self.reduction == "mean":
            temp_vidv_ub = self.tik_instance.Tensor(
                self.x_dtype,
                [math.ceil(move_num*self.x_shape[1]/NUM_SIXTYFOUR) *
                 NUM_SIXTYFOUR], name="temp_vidv_ub", scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(MASK64, temp_vidv_ub, total_weight,
                                         1, 1, NUM_EIGHT)

            for i in range(vmul_repeat_times-1):
                self.tik_instance.vdiv(
                    MASK64, self.temp_out_ub[i*MAX_REPEAT_NUM],
                    self.temp_out_ub[i*MAX_REPEAT_NUM], temp_vidv_ub,
                    MAX_REPEAT, 1, 1, 1, NUM_EIGHT, NUM_EIGHT, 0)
            if vmul_repeat <= MAX_REPEAT:
                self.tik_instance.vdiv(
                    MASK64,
                    self.temp_out_ub[(vmul_repeat_times-1)*MAX_REPEAT_NUM],
                    self.temp_out_ub[(vmul_repeat_times-1)*MAX_REPEAT_NUM],
                    temp_vidv_ub, vmul_repeat, 1, 1, 1, NUM_EIGHT, NUM_EIGHT, 0)
        self.tik_instance.data_move(self.output[cycle*max_move_num],
                                    self.temp_out_ub, 0, 1,
                                    burst_len*self.x_shape[1],
                                    NUM_EIGHT, NUM_EIGHT)

    def two_dim_and_weight_no_32b_compute(self):
        """
        calculate input two dim and weight lower 32b.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.weight_no_32b_tiling()
        with self.tik_instance.for_range(0, self.move_times,
                                         block_num=self.move_times) as cycle:
            self.init_ub()
            if self.reduction == "none":
                with self.tik_instance.if_scope(cycle < self.move_times-1):
                    self._calculate_process_none(
                        self.max_move_num, self.max_move_num,
                        self.max_vmul_repeat, self.max_dup_repeat,
                        self.max_burst_len, cycle)
                with self.tik_instance.if_scope(cycle == self.move_times-1):
                    self._calculate_process_none(
                        self.last_move_num, self.max_move_num,
                        self.last_vmul_repeat, self.last_dup_repeat,
                        self.last_move_burst_len, cycle)
            else:
                with self.tik_instance.if_scope(cycle < self.move_times-1):
                    self._calculate_process_sum_and_mean(
                        self.max_move_num, self.max_move_num,
                        self.max_vmul_repeat, self.max_dup_repeat,
                        self.max_burst_len, cycle)
                with self.tik_instance.if_scope(cycle == self.move_times-1):
                    self._calculate_process_sum_and_mean(
                        self.last_move_num, self.max_move_num,
                        self.last_vmul_repeat, self.last_dup_repeat,
                        self.last_move_burst_len, cycle)

    def nll_loss_compute_start(self):
        """
        Different calculation methods

        Parameters
        ----------

        Returns
        -------
        None
        """
        if self.x_dim == 1:
            self.one_dim_compute()
        elif self.x_dim == DIM2 and self.x_shape[1] >= NUM_EIGHT:
            self.two_dim_compute()
        elif self.x_dim == DIM2 and self.x_shape[1] < NUM_EIGHT:
            self.two_dim_and_weight_no_32b_compute()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.data_x,
                                           self.y_grad,
                                           self.data_target,
                                           self.data_weight,
                                           self.total_weight_gm],
                                   outputs=[self.output])
        return self.tik_instance


@util.check_input_type(dict, dict, dict, dict, dict, dict, str, str)
def nll_loss_grad(x, y_grad, target, weight, total_weight, x_grad,
                  reduction="mean", kernel_name="nll_loss_grad"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input, the length of shape should be two or one.
    y_grad : dict
        shape and dtype of input, the length of shape must be one.
    target : dict
        shape and dtype of input, the length of shape only support one.
    total_weight : dict
        shape and dtype of input, it is a scalar.
    weight : dict or None
        the length of shape only support one when weight is dict.
    x_grad: dict
        It’s a tensor with shape(minibatch, ) when reduction == ‘none’ and
        the input is 2D. Otherwise, the output is a scalar.
    reduction: str
        default value is "mean"
    kernel_name : str
        kernel name, default value is "nll_loss_grad"

    Returns
    -------
    None
    """
    _shape_and_dtype_check(x, y_grad, target, weight, total_weight,
                           reduction, kernel_name)
    nll_loss_function = nll_loss_grad_compute(x, y_grad, target, weight,
                                              reduction, kernel_name)
    return nll_loss_function.nll_loss_compute_start()
