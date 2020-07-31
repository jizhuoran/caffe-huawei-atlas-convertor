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

decode_cornerpoints_target_wrt_center_v1_tik
"""

from te import tik
from topi.cce import util
# the max num of single copy
SINGLE_N_MAX = 640
N_MAX = 65500

ONE = 1
TWO = 2
FOUR = 4
EIGHT = 8
SIXTEEN = 16
BLOCK_DATA = 32
MASK = 128
# the matrix of hardware
MATRIX = 256
MATRIX_DATA = 512

# data order of data_y code
TEN = 10
TWENTYONE = 21
THIRTY = 30


def int_ceil_div(divisor_a, divisor_b):
    """
    round up function

    Paramater:
    :param divisor_a: int.
    :param divisor_b: int.
    :return: int
    """
    if divisor_b == 0:
        raise RuntimeError("division by zero")
    return (divisor_a + divisor_b - ONE) // divisor_b


def check_decode_cornerpoints_target_wrt_center_v1_shape_params(
        keypoints_prediction,
        anchors,
        keypoints_decoded):
    """
    The params check function of decode_cornerpoints_target_wrt_center_v1

    Parameters:
    ----------
    Returns : All transformed params.
    ----------
    """
    shape_x = keypoints_prediction.get("shape")
    shape_x_num = len(shape_x)
    shape_y = anchors.get("shape")
    shape_y_num = len(shape_y)
    shape_z = keypoints_decoded.get("shape")
    shape_z_num = len(shape_z)
    dtype_x = keypoints_prediction.get("dtype").lower()
    dtype_y = anchors.get("dtype").lower()
    dtype_z = keypoints_decoded.get("dtype").lower()

    # Abnormality test
    if dtype_x != dtype_y or dtype_x != dtype_z or dtype_y != dtype_z:
        raise RuntimeError("dtype of inputs and output should be consistent")
    if dtype_x != 'float16':
        raise RuntimeError("dtype of inputs should be float16")
    if shape_x_num != shape_y_num or shape_x_num != shape_z_num or shape_y_num != shape_z_num:
        raise RuntimeError("dimension of inputs should be consistent")
    if shape_x_num != TWO:
        raise RuntimeError("dimension of inputs should be 2")
    check_result(shape_x, shape_y, shape_z)


def check_result(shape_x, shape_y, shape_z):
    """
    The params check function of decode_wheels_target

    Parameters:
    ----------
    Returns : All transformed params.
    ----------
    """
    n_x, m_x = shape_x
    n_y, m_y = shape_y
    n_z, m_z = shape_z
    if not isinstance(n_x, int):
        raise RuntimeError("n dimension of input should be int")
    if n_x != n_y or n_x != n_z or n_y != n_z:
        raise RuntimeError("n dimension of inputs should be consistent")
    if m_x != EIGHT:
        raise RuntimeError("m dimension of boundary_predictions should be 8")
    if m_y != FOUR:
        raise RuntimeError("m dimension of anchors should be 4")
    if m_z != EIGHT:
        raise RuntimeError("m dimension of boundary_encoded should be 8")
    if n_x < ONE or n_x > N_MAX:
        raise RuntimeError("n dimension of inputs should in [1, 65500]")


class Tiling:
    """
    calculating the shape
    Returns
    -------
    None
    """
    def __init__(self, n_x, core_num):
        self.n_x = n_x
        self.core_num = core_num
        self.last_n = self.n_x % SINGLE_N_MAX
        self.last_core = self.n_x // SINGLE_N_MAX % self.core_num
        self.factor = self.n_x // SINGLE_N_MAX // self.core_num

    def set_shape_maxtrix(self, n_x):
        """
        set_input_shape
        :return:
        """
        self.n_x = n_x

    def set_n_maxtrix(self, core_num):
        """
        set_input_shape
        :return:
        """
        self.core_num = core_num


class InitShape:
    """
    calculating the shape
    Returns
    -------
    None
    """
    def __init__(self, n_x):
        self.n_x = n_x
        # number of 16*16
        self.n_maxtrix = int_ceil_div(n_x * EIGHT, MATRIX)
        # shape of calculate
        self.shape_maxtrix = (self.n_maxtrix, SIXTEEN, SIXTEEN)
        # repeat times of rep_stride*block
        self.repeat_whxy = self.n_maxtrix * TWO // EIGHT
        # rep_stride*block of one repeat
        self.rep_stride = self.n_maxtrix * TWO % EIGHT
        # number of instruction
        self.instruction_number = TWO

        if self.repeat_whxy == 0:
            self.repeat_whxy = ONE
            self.instruction_number = ONE

        if self.rep_stride == 0:
            self.instruction_number = ONE
            self.rep_stride = EIGHT

    def set_input_shape(self, repeat_whxy):
        """
        set_input_shape
        :return:
        """
        self.repeat_whxy = repeat_whxy

    def set_output_shape(self, rep_stride):
        """
        set_input_shape
        :return:
        """
        self.rep_stride = rep_stride


class InitTensor:
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, shape_x, shape_y, dtype_x):
        self.data_x = tik_instance.Tensor(
            dtype_x, shape_x, name="data_x", scope=tik.scope_gm)
        self.data_y = tik_instance.Tensor(
            dtype_x, shape_y, name="data_y", scope=tik.scope_gm)
        self.data_z = tik_instance.Tensor(
            dtype_x, shape_x, name="data_z", scope=tik.scope_gm)
        self.dump_0 = tik_instance.Scalar(dtype="float16", init_value=0.0)
        self.dump_half = tik_instance.Scalar(dtype="float16", init_value=0.5)

    def set_data_x(self, data_x):
        """
        data_x_ub
        :return:
        """
        self.data_x = data_x

    def set_data_y(self, data_y):
        """
        data_y_ub
        :return:
        """
        self.data_y = data_y


class InitSecondTensor:
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, init_shape, dtype_x):
        self.data_x_ub = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_x_ub", scope=tik.scope_ubuf)
        self.data_y_ub = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_y_ub", scope=tik.scope_ubuf)
        self.data_x_ub_trs = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_x_ub_trs", scope=tik.scope_ubuf)
        self.data_y_ub_trs = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_y_ub_trs", scope=tik.scope_ubuf)
        self.data_y_ub_trs1 = tik_instance.Tensor(
            dtype_x, (init_shape.n_maxtrix, EIGHT, SIXTEEN),
            name="data_y_ub_trs1", scope=tik.scope_ubuf)

    def set_data_x_ub(self, data_x_ub):
        """
        data_y_ub_trs1
        :return:
        """
        self.data_x_ub = data_x_ub

    def set_data_y_ub(self, data_y_ub):
        """
        data_anchor_wh
        :return:
        """
        self.data_y_ub = data_y_ub


class InitThirdTensor(InitSecondTensor):
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, init_shape, dtype_x):
        super(InitThirdTensor, self).__init__(tik_instance, init_shape, dtype_x)
        self.data_anchor_wh = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_anchor_wh", scope=tik.scope_ubuf)
        self.data_anchor_x0y0 = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_anchor_x0y0", scope=tik.scope_ubuf)
        self.data_anchor_xy = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_anchor_xy", scope=tik.scope_ubuf)
        self.data_z_ub0 = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_z_ub0", scope=tik.scope_ubuf)
        self.data_z_ub1 = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_z_ub1", scope=tik.scope_ubuf)
        self.data_z_ub = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_z_ub", scope=tik.scope_ubuf)

    def set_data_y_ub_trs1(self, data_y_ub_trs1):
        """
        data_y_ub_trs1
        :return:
        """
        self.data_y_ub_trs1 = data_y_ub_trs1

    def set_data_anchor_wh(self, data_anchor_wh):
        """
        data_anchor_wh
        :return:
        """
        self.data_anchor_wh = data_anchor_wh


def box_coordinate_offset_integer_block(tik_instance, shape, mid_tensor, i, j):
    """
    :param tik_instance: tik
    :param shape: class
    :param gm_tensor:class
    :param mid_tensor: class
    :param i,j: int
    :return:
    """
    dump_twice = tik_instance.Scalar(dtype="float16", init_value=2.0)
    with tik_instance.if_scope(tik.any((TEN * i + j == ONE),
                                       (TEN * i + j == THIRTY))):
        # 01 replace by 2*y1 / 30 replace by 2*x1
        tik_instance.vmuls(SIXTEEN * shape.rep_stride,
                           mid_tensor.data_anchor_x0y0[BLOCK_DATA * i +
                                                       SIXTEEN * j],
                           mid_tensor.data_y_ub_trs1[SIXTEEN * j],
                           dump_twice,
                           shape.repeat_whxy,
                           EIGHT,
                           FOUR,
                           EIGHT * shape.rep_stride,
                           FOUR * shape.rep_stride)
    with tik_instance.else_scope():
        with tik_instance.if_scope(tik.any((TEN * i + j == TEN),
                                           (TEN * i + j == TWENTYONE))):
            # 10 replace by 2*x2  /  21 replace by 2*y2
            tik_instance.vmuls(SIXTEEN * shape.rep_stride,
                               mid_tensor.data_anchor_x0y0[BLOCK_DATA * i +
                                                           SIXTEEN * j],
                               mid_tensor.data_y_ub_trs1[SIXTEEN * j +
                                                         BLOCK_DATA],
                               dump_twice,
                               shape.repeat_whxy,
                               EIGHT, FOUR,
                               EIGHT * shape.rep_stride,
                               FOUR * shape.rep_stride)
        with tik_instance.else_scope():
            tik_instance.vadd(SIXTEEN * shape.rep_stride,
                              mid_tensor.data_anchor_x0y0[BLOCK_DATA * i +
                                                          SIXTEEN * j],
                              mid_tensor.data_y_ub_trs1[SIXTEEN * j +
                                                        BLOCK_DATA],
                              mid_tensor.data_y_ub_trs1[SIXTEEN * j],
                              shape.repeat_whxy,
                              EIGHT,
                              FOUR,
                              FOUR,
                              EIGHT * shape.rep_stride,
                              FOUR * shape.rep_stride,
                              FOUR * shape.rep_stride)


def box_coordinate_offset_remainder_block(tik_instance, shape, mid_tensor, i, j):
    """
    :param tik_instance: tik
    :param shape: class
    :param gm_tensor:class
    :param mid_tensor: class
    :param i,j: int
    :return:
    """
    dump_twice = tik_instance.Scalar(dtype="float16", init_value=2.0)
    with tik_instance.if_scope(tik.any((TEN * i + j == ONE),
                                       (TEN * i + j == THIRTY))):
        # 01 replace by 2*y1 / # replace by 2*x1
        tik_instance.vmuls(MASK,
                           mid_tensor.data_anchor_x0y0[BLOCK_DATA * i +
                                                       SIXTEEN * j],
                           mid_tensor.data_y_ub_trs1[SIXTEEN * j],
                           dump_twice,
                           shape.repeat_whxy,
                           EIGHT,
                           FOUR,
                           EIGHT * EIGHT,
                           BLOCK_DATA)
        tik_instance.vmuls(SIXTEEN * shape.rep_stride,
                           mid_tensor.data_anchor_x0y0[BLOCK_DATA * i +
                                                       SIXTEEN * j +
                                                       shape.repeat_whxy *
                                                       FOUR * MATRIX],
                           mid_tensor.data_y_ub_trs1[SIXTEEN * j +
                                                     shape.repeat_whxy *
                                                     MATRIX_DATA],
                           dump_twice,
                           ONE,
                           EIGHT,
                           FOUR,
                           EIGHT * shape.rep_stride,
                           BLOCK_DATA)
    with tik_instance.else_scope():
        with tik_instance.if_scope(tik.any((TEN * i + j == TEN),
                                           (TEN * i + j == TWENTYONE))):
            # 10 replace by 2*x2  /  21 replace by 2*y2
            tik_instance.vmuls(MASK,
                               mid_tensor.data_anchor_x0y0[BLOCK_DATA * i +
                                                           SIXTEEN * j],
                               mid_tensor.data_y_ub_trs1[SIXTEEN * j +
                                                         BLOCK_DATA],
                               dump_twice,
                               shape.repeat_whxy,
                               EIGHT,
                               FOUR,
                               EIGHT * EIGHT,
                               BLOCK_DATA)
            tik_instance.vmuls(SIXTEEN * shape.rep_stride,
                               mid_tensor.data_anchor_x0y0[BLOCK_DATA * i +
                                                           SIXTEEN * j +
                                                           shape. \
                                                           repeat_whxy *
                                                           FOUR * MATRIX],
                               mid_tensor.data_y_ub_trs1[SIXTEEN * j +
                                                         BLOCK_DATA +
                                                         shape. \
                                                         repeat_whxy *
                                                         MATRIX_DATA],
                               dump_twice,
                               ONE,
                               EIGHT,
                               FOUR,
                               EIGHT * shape.rep_stride,
                               BLOCK_DATA)
        with tik_instance.else_scope(): # replace by 2*x0 2*y0
            tik_instance.vadd(MASK,
                              mid_tensor.data_anchor_x0y0[BLOCK_DATA * i +
                                                          SIXTEEN * j],
                              mid_tensor.data_y_ub_trs1[SIXTEEN * j +
                                                        BLOCK_DATA],
                              mid_tensor.data_y_ub_trs1[SIXTEEN * j],
                              shape.repeat_whxy,
                              EIGHT,
                              FOUR,
                              FOUR,
                              EIGHT * EIGHT,
                              BLOCK_DATA,
                              BLOCK_DATA)
            tik_instance.vadd(SIXTEEN * shape.rep_stride,
                              mid_tensor.data_anchor_x0y0[BLOCK_DATA * i +
                                                          SIXTEEN * j +
                                                          shape. \
                                                          repeat_whxy *
                                                          FOUR * MATRIX],
                              mid_tensor.data_y_ub_trs1[SIXTEEN * j +
                                                        BLOCK_DATA +
                                                        shape.repeat_whxy *
                                                        MATRIX_DATA],
                              mid_tensor.data_y_ub_trs1[SIXTEEN * j +
                                                        shape.repeat_whxy *
                                                        MATRIX_DATA],
                              ONE,
                              EIGHT,
                              FOUR,
                              FOUR,
                              EIGHT * shape.rep_stride,
                              BLOCK_DATA,
                              BLOCK_DATA)


def calculate_process(tik_instance, gm_tensor, shape,
                      current_data_x, current_data_y):
    """
    :param tik_instance: tik
    :param shape: class
    :param mid_tensor: class
    :param current_data_x: int
    :param current_data_y: int
    :return:
    """

    mid_tensor = InitThirdTensor(tik_instance, shape, "float16")
    tik_instance.data_move(mid_tensor.data_x_ub,
                           gm_tensor.data_x[current_data_x],
                           0,
                           ONE, int_ceil_div(shape.n_x * EIGHT, SIXTEEN),
                           0, 0)

    tik_instance.data_move(mid_tensor.data_y_ub,
                           gm_tensor.data_y[current_data_y],
                           0,
                           int_ceil_div(shape.n_x * FOUR, SIXTEEN), ONE,
                           0, ONE)
    if shape.n_x not in (ONE, TWO):
        tik_instance.data_move(mid_tensor.data_y_ub[0 + SIXTEEN],
                               gm_tensor.data_y[current_data_y + EIGHT],
                               0, int_ceil_div(shape.n_x * FOUR, SIXTEEN), ONE, 0, ONE)
    with tik_instance.for_range(0, shape.n_maxtrix) as i:
        tik_instance.vtranspose(mid_tensor.data_x_ub_trs[MATRIX*i],
                                mid_tensor.data_x_ub[MATRIX*i])
        tik_instance.vtranspose(mid_tensor.data_y_ub_trs[MATRIX*i],
                                mid_tensor.data_y_ub[MATRIX*i])
    # extract tensor_y_ub_trs:
    tik_instance.vadds(MASK,
                       mid_tensor.data_y_ub_trs1,
                       mid_tensor.data_y_ub_trs,
                       gm_tensor.dump_0,
                       shape.n_maxtrix,
                       ONE, ONE,
                       EIGHT, SIXTEEN)
    # calculate data_anchor_wh and data_anchor_x0y0
    with tik_instance.if_scope(shape.instruction_number == ONE):
        with tik_instance.for_range(0, FOUR) as i:
            with tik_instance.for_range(0, TWO) as j:
                tik_instance.vsub(SIXTEEN * shape.rep_stride,
                                  mid_tensor.data_anchor_wh[BLOCK_DATA * i +
                                                            SIXTEEN * j],
                                  mid_tensor.data_y_ub_trs1[SIXTEEN * j +
                                                            BLOCK_DATA],
                                  mid_tensor.data_y_ub_trs1[SIXTEEN * j],
                                  shape.repeat_whxy,
                                  EIGHT,
                                  FOUR,
                                  FOUR,
                                  EIGHT * shape.rep_stride,
                                  FOUR * shape.rep_stride,
                                  FOUR * shape.rep_stride)
                box_coordinate_offset_integer_block(tik_instance,
                                                    shape,
                                                    mid_tensor,
                                                    i, j)

    with tik_instance.else_scope():
        with tik_instance.for_range(0, FOUR) as i:
            with tik_instance.for_range(0, TWO) as j:
                tik_instance.vsub(MASK,
                                  mid_tensor.data_anchor_wh[BLOCK_DATA * i
                                                            + SIXTEEN * j],
                                  mid_tensor.data_y_ub_trs1[SIXTEEN * j
                                                            + BLOCK_DATA],
                                  mid_tensor.data_y_ub_trs1[SIXTEEN * j],
                                  shape.repeat_whxy,
                                  EIGHT, FOUR, FOUR,
                                  EIGHT * EIGHT, BLOCK_DATA, BLOCK_DATA)

                tik_instance.vsub(SIXTEEN * shape.rep_stride,
                                  mid_tensor.data_anchor_wh[BLOCK_DATA * i +
                                                            SIXTEEN * j +
                                                            shape.repeat_whxy *
                                                            FOUR * MATRIX],
                                  mid_tensor.data_y_ub_trs1[SIXTEEN * j +
                                                            BLOCK_DATA +
                                                            shape.repeat_whxy *
                                                            MATRIX_DATA],
                                  mid_tensor.data_y_ub_trs1[SIXTEEN * j +
                                                            shape.repeat_whxy *
                                                            MATRIX_DATA],
                                  ONE,
                                  EIGHT, FOUR, FOUR,
                                  EIGHT * shape.rep_stride,
                                  FOUR * shape.rep_stride,
                                  FOUR * shape.rep_stride)
                box_coordinate_offset_remainder_block(tik_instance,
                                                      shape,
                                                      mid_tensor,
                                                      i, j)

    # calculate data_anchor_xy
    tik_instance.vmuls(MASK, mid_tensor.data_anchor_xy,
                       mid_tensor.data_anchor_x0y0,
                       gm_tensor.dump_half,
                       shape.n_maxtrix * TWO,
                       ONE, ONE,
                       EIGHT, EIGHT)

    # calculate input * data_anchor_wh
    tik_instance.vmul(MASK,
                      mid_tensor.data_z_ub0,
                      mid_tensor.data_x_ub_trs,
                      mid_tensor.data_anchor_wh,
                      shape.n_maxtrix * TWO,
                      ONE, ONE, ONE,
                      EIGHT, EIGHT, EIGHT)

    # calculate input * data_anchor_wh + data_anchor_xy
    tik_instance.vadd(MASK,
                      mid_tensor.data_z_ub1,
                      mid_tensor.data_z_ub0,
                      mid_tensor.data_anchor_xy,
                      shape.n_maxtrix * TWO,
                      ONE, ONE, ONE,
                      EIGHT, EIGHT, EIGHT)
    # transpose output
    with tik_instance.for_range(0, shape.n_maxtrix) as i:
        tik_instance.vtranspose(mid_tensor.data_z_ub[MATRIX*i],
                                mid_tensor.data_z_ub1[MATRIX*i])
    # copy ub to gm
    tik_instance.data_move(gm_tensor.data_z[current_data_x],
                           mid_tensor.data_z_ub,
                           0,
                           ONE, int_ceil_div(shape.n_x * EIGHT, SIXTEEN),
                           0, 0)


@util.check_input_type(dict, dict, dict, str)
def decode_cornerpoints_target_wrt_center_v1(
        keypoints_prediction,
        anchors,
        keypoints_decoded,
        kernel_name="cce_decode_cornerpoints_target_wrt_center_v1_float16"):
    """
    The params check function of decode_wheels_target

    Parameters:
    ----------
    Returns : All transformed params.
    ----------
    """
    check_decode_cornerpoints_target_wrt_center_v1_shape_params(
        keypoints_prediction,
        anchors,
        keypoints_decoded)
    util.check_kernel_name(kernel_name)
    shape_x = keypoints_prediction.get("shape")

    tik_instance = tik.Tik(tik.Dprofile(), True)
    core_num = tik.Dprofile().get_aicore_num()
    tiling = Tiling(shape_x[0], core_num)
    # gm_tensor init
    gm_tensor = InitTensor(tik_instance, shape_x, [shape_x[0], FOUR], 'float16')
    if tiling.factor > 0:
        thread_num = TWO if tiling.factor != ONE else ONE
        with tik_instance.for_range(0, core_num, block_num=core_num) as current_core:
            with tik_instance.for_range(0, tiling.factor, thread_num=thread_num) as current_factor:
                shape = InitShape(SINGLE_N_MAX)
                current_data_x = EIGHT * SINGLE_N_MAX * (current_core + core_num * current_factor)
                current_data_y = FOUR * SINGLE_N_MAX * (current_core + core_num * current_factor)
                calculate_process(tik_instance, gm_tensor, shape, current_data_x, current_data_y)
    if tiling.last_core > 0:
        thread_num = TWO if tiling.last_core != ONE else ONE
        with tik_instance.for_range(0, tiling.last_core, thread_num=thread_num) as current_core:
            shape = InitShape(SINGLE_N_MAX)
            current_data_x = EIGHT * SINGLE_N_MAX * (core_num * tiling.factor + current_core)
            current_data_y = FOUR * SINGLE_N_MAX * (core_num * tiling.factor + current_core)
            calculate_process(tik_instance, gm_tensor, shape, current_data_x, current_data_y)
    if tiling.last_n > 0:
        shape = InitShape(tiling.last_n)
        current_data_x = EIGHT * SINGLE_N_MAX * (core_num * tiling.factor + tiling.last_core)
        current_data_y = FOUR * SINGLE_N_MAX * (core_num * tiling.factor + tiling.last_core)
        calculate_process(tik_instance, gm_tensor, shape, current_data_x, current_data_y)

    # build_cce
    tik_instance.BuildCCE(
        kernel_name=kernel_name,
        inputs=[gm_tensor.data_x, gm_tensor.data_y],
        outputs=[gm_tensor.data_z])
