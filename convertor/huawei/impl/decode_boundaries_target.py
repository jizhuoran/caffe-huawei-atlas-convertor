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

decode_boundaries_target
"""
from te import tik
from topi.cce import util


NMAX = 128
SHAPE_DIMENSION = 2
SHAPE_ONE = 1
SHAPE_TWO = 4
MAX = 65500
LINE = 16
FOUR = 4
VECTOR = 128
LINE3 = 32
VECTOR_NUM = 8
MATRIX_NUM = 256
MIN = 4


class Input():
    """
    decrease parameters
    """

    def __init__(self, **kwargs):
        self.shape_boundary_predictions = None
        self.shape_anchors = None
        self.dtype_boundary_predictions = None
        self.dtype_anchors = None
        self.n_max = 0
        self.burst_x = 0
        self.burst_y = 0
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        self.__dict__.update(kwargs)

    def set_nmax(self, n_max):
        """
        :param n_max:
        :return:
        """
        self.n_max = n_max


class Output():
    """
    decrease parameters
    """

    def __init__(self, **kwargs):
        self.burst_num = 0
        self.n_vector = None
        self.n_matrix = None
        self.shape_matrix = None
        self.shape_vector = None
        self.rep = 0
        self.overflow = 0
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        self.__dict__.update(kwargs)

    def set_burst_num(self, burst_num):
        """
        :param n_max:
        :return:
        """
        self.burst_num = burst_num


def int_ceil_div(divisor_a, divisor_b):
    """
    round up function

    Paramater:
    :param divisor_a: int.
    :param divisor_b: int.
    :return: int
    """
    return (divisor_a + divisor_b - 1) // divisor_b


def tiling_func(frame, n_max):
    """
    :param frame:
    :param n_max:
    :return:
    """

    total_handling_times = frame // n_max
    last_handling_n = frame % n_max

    return total_handling_times, last_handling_n


def check_input(boundary_predictions, anchors, boundary_encoded, n_max):
    """
    :param boundary_predictions:
    :param anchors:
    :return:
    """
    shape_boundary_predictions = boundary_predictions.get("shape")
    shape_boundary_predictions_num = len(shape_boundary_predictions)
    shape_anchors = anchors.get("shape")
    dtype_boundary_predictions = boundary_predictions.get("dtype").lower()
    dtype_anchors = anchors.get("dtype").lower()

    # Abnormality test
    n_x, m_x = shape_boundary_predictions
    n_y, m_y = shape_anchors

    if dtype_boundary_predictions != dtype_anchors or \
            dtype_boundary_predictions != boundary_encoded.get("dtype").lower():
        raise RuntimeError("dtype of inputs should be consistent")
    if dtype_boundary_predictions != 'float16':
        raise RuntimeError("dtype of inputs should be float16")
    if shape_boundary_predictions_num != len(anchors.get("shape")) or \
            shape_boundary_predictions_num != len(boundary_encoded.get("shape")):
        raise RuntimeError("dimension of inputs should be consistent")
    if shape_boundary_predictions_num != SHAPE_DIMENSION:
        raise RuntimeError("dimension of inputs should be 2")
    if not isinstance(n_x, int):
        raise RuntimeError("n dimension of input should be int")
    if n_x != n_y:
        raise RuntimeError("n dimension of inputs should be consistent")
    if m_x != SHAPE_ONE:
        raise RuntimeError("m dimension of input_x should be 1")
    if m_y != SHAPE_TWO:
        raise RuntimeError("m dimension of input_y should be 4")
    if n_x <= 0 or n_x > MAX:
        raise RuntimeError("n dimension of inputs should in [1, 65500]")

    # tiling
    total_handling_times, last_handling_n = tiling_func(n_x, n_max)

    return total_handling_times, last_handling_n


def tranpose(input_tensor, n_matrix, tik_instance, **kwargs):
    """
    tranpose tensor
    :return:
    """
    tensortype = kwargs['type']
    tensorshape = kwargs['shape']
    tensorname = kwargs['name']
    # transpose input
    tensor_trs = tik_instance.Tensor(
        tensortype, tensorshape, name=tensorname, scope=tik.scope_ubuf)

    with tik_instance.for_range(0, n_matrix) as i:
        tik_instance.vtranspose(tensor_trs[i * MATRIX_NUM],
                                input_tensor[i * MATRIX_NUM])

    return tensor_trs


def get_gm(tik_instance, **kwargs):
    """
    :param tik_instance:
    :param kwargs:
    :return:
    """
    dtype = kwargs['dtype']
    shape1 = kwargs['shape1']
    shape2 = kwargs['shape2']
    name1 = kwargs['name1']
    name2 = kwargs['name2']
    name3 = kwargs['name3']
    scope = kwargs['scope']

    data_boundary_predictions = tik_instance.Tensor(
        dtype,
        shape1,
        name=name1,
        scope=scope)
    data_anchors = tik_instance.Tensor(
        dtype,
        shape2,
        name=name2,
        scope=scope)
    data_z = tik_instance.Tensor(
        dtype,
        shape1,
        name=name3,
        scope=scope)

    return data_boundary_predictions, data_anchors, data_z


def process_calculate(tik_instance, input_info, output_info,
                      current_handling_times, **kwargs):
    """
    calculate
    :return:
    """
    def cal():
        with tik_instance.new_stmt_scope():
            # copy gm to ub
            data_x_ub = tik_instance.Tensor(input_info.dtype_boundary_predictions,
                                            output_info.shape_vector,
                                            name="data_x_ub",
                                            scope=tik.scope_ubuf)
            data_y_ub = tik_instance.Tensor(input_info.dtype_anchors,
                                            output_info.shape_matrix,
                                            name="data_y_ub",
                                            scope=tik.scope_ubuf)

            tik_instance.data_move(data_x_ub,
                                   data_boundary_predictions[current_handling_times *
                                                             input_info.n_max],
                                   0, 1, input_info.burst_x, 0, 0)
            with tik_instance.for_range(0, FOUR) as i:
                tik_instance.data_move(data_y_ub[i * LINE],
                                       data_anchors[current_handling_times * FOUR *
                                                    input_info.n_max + i * FOUR],
                                       0, input_info.burst_y, 1, 0, FOUR - 1)

            data_y_ub_trs = tranpose(
                input_tensor=data_y_ub,
                n_matrix=output_info.n_matrix * FOUR,
                tik_instance=tik_instance,
                type=input_info.dtype_anchors,
                shape=output_info.shape_matrix,
                name="data_y_ub_trs")

            # calculate data_anchor_wh and data_anchor_x0y0
            data_anchor_wh = tik_instance.Tensor(
                input_info.dtype_anchors, output_info.shape_vector,
                name="data_anchor_wh", scope=tik.scope_ubuf)
            data_anchor_x0y0 = tik_instance.Tensor(
                input_info.dtype_anchors, output_info.shape_vector,
                name="data_anchor_x0y0", scope=tik.scope_ubuf)

            tik_instance.vsub(VECTOR,
                              data_anchor_wh,
                              data_y_ub_trs[LINE3],
                              data_y_ub_trs,
                              output_info.rep,
                              1,
                              LINE,
                              LINE,
                              1 * VECTOR_NUM,
                              LINE * VECTOR_NUM,
                              LINE * VECTOR_NUM)
            tik_instance.vadd(VECTOR,
                              data_anchor_x0y0,
                              data_y_ub_trs[LINE3],
                              data_y_ub_trs,
                              output_info.rep,
                              1,
                              LINE,
                              LINE,
                              1 * VECTOR_NUM,
                              LINE * VECTOR_NUM,
                              LINE * VECTOR_NUM)

            # calculate data_anchor_xa = (x1+x2)*0.5
            data_anchor_xaya = tik_instance.Tensor(
                input_info.dtype_anchors, output_info.shape_vector,
                name="data_anchor_xaya", scope=tik.scope_ubuf)

            tik_instance.vmuls(VECTOR, data_anchor_xaya,
                               data_anchor_x0y0,
                               tik_instance.Scalar(
                                   dtype="float16", init_value=0.5),
                               output_info.rep,
                               1, 1,
                               VECTOR_NUM, VECTOR_NUM)

            # calculate input * data_anchor_wh
            data_z_ub0 = tik_instance.Tensor(
                input_info.dtype_anchors, output_info.shape_vector,
                name="data_z_ub0", scope=tik.scope_ubuf)

            tik_instance.vmul(VECTOR,
                              data_z_ub0,
                              data_x_ub,
                              data_anchor_wh,
                              output_info.rep,
                              1, 1, 1,
                              VECTOR_NUM, VECTOR_NUM, VECTOR_NUM)

            # calculate input * data_anchor_wh + data_anchor_xaya
            data_z_ub = tik_instance.Tensor(
                input_info.dtype_boundary_predictions, output_info.shape_vector,
                name="data_z_ub", scope=tik.scope_ubuf)

            tik_instance.vadd(VECTOR,
                              data_z_ub,
                              data_z_ub0,
                              data_anchor_xaya,
                              output_info.rep,
                              1, 1, 1,
                              VECTOR_NUM, VECTOR_NUM, VECTOR_NUM)

            # copy ub to gm
            tik_instance.data_move(data_z[current_handling_times * input_info.n_max],
                                   data_z_ub,
                                   0,
                                   1, input_info.burst_x,
                                   0, 0)

    data_boundary_predictions = kwargs['data_boundary_predictions']
    data_anchors = kwargs['data_anchors']
    data_z = kwargs['data_z']

    cal()


def process_end(tik_instance, input_info, output_info,
                current_handling_times, **kwargs):
    """
    calculate
    :return:
    """
    def cal():
        with tik_instance.new_stmt_scope():
            # copy gm to ub
            data_x_ub = tik_instance.Tensor(input_info.dtype_boundary_predictions,
                                            output_info.shape_vector,
                                            name="data_x_ub",
                                            scope=tik.scope_ubuf)
            data_y_ub = tik_instance.Tensor(input_info.dtype_anchors,
                                            output_info.shape_matrix,
                                            name="data_y_ub",
                                            scope=tik.scope_ubuf)

            tik_instance.data_move(data_x_ub,
                                   data_boundary_predictions[current_handling_times *
                                                             input_info.n_max],
                                   0, 1, input_info.burst_x, 0, 0)
            with tik_instance.for_range(0, FOUR) as i:
                tik_instance.data_move(data_y_ub[i * LINE],
                                       data_anchors[current_handling_times * FOUR *
                                                    input_info.n_max + i * FOUR],
                                       0, input_info.burst_y, 1, 0, FOUR - 1)

            data_y_ub_trs = tranpose(
                input_tensor=data_y_ub,
                n_matrix=output_info.n_matrix * FOUR,
                tik_instance=tik_instance,
                type=input_info.dtype_anchors,
                shape=output_info.shape_matrix,
                name="data_y_ub_trs")

            # calculate data_anchor_wh and data_anchor_x0y0
            data_anchor_wh = tik_instance.Tensor(
                input_info.dtype_anchors, output_info.shape_vector,
                name="data_anchor_wh", scope=tik.scope_ubuf)
            data_anchor_x0y0 = tik_instance.Tensor(
                input_info.dtype_anchors, output_info.shape_vector,
                name="data_anchor_x0y0", scope=tik.scope_ubuf)

            tik_instance.vsub(output_info.overflow,
                              data_anchor_wh[output_info.rep * VECTOR],
                              data_y_ub_trs[LINE3 +
                                            output_info.rep * MATRIX_NUM * VECTOR_NUM],
                              data_y_ub_trs[output_info.rep *
                                            MATRIX_NUM * VECTOR_NUM],
                              1,
                              1,
                              LINE,
                              LINE,
                              1 * VECTOR_NUM,
                              LINE * VECTOR_NUM,
                              LINE * VECTOR_NUM)

            tik_instance.vadd(output_info.overflow,
                              data_anchor_x0y0[output_info.rep * VECTOR],
                              data_y_ub_trs[LINE3 +
                                            output_info.rep * MATRIX_NUM * VECTOR_NUM],
                              data_y_ub_trs[output_info.rep *
                                            MATRIX_NUM * VECTOR_NUM],
                              1,
                              1,
                              LINE,
                              LINE,
                              1 * VECTOR_NUM,
                              LINE * VECTOR_NUM,
                              LINE * VECTOR_NUM)

            # calculate data_anchor_xa = (x1+x2)*0.5
            data_anchor_xaya = tik_instance.Tensor(
                input_info.dtype_anchors, output_info.shape_vector,
                name="data_anchor_xaya", scope=tik.scope_ubuf)

            tik_instance.vmuls(output_info.overflow, data_anchor_xaya[output_info.rep * VECTOR],
                               data_anchor_x0y0[output_info.rep * VECTOR],
                               tik_instance.Scalar(
                                   dtype="float16", init_value=0.5),
                               1,
                               1, 1,
                               VECTOR_NUM, VECTOR_NUM)

            # calculate input * data_anchor_wh
            data_z_ub0 = tik_instance.Tensor(
                input_info.dtype_anchors, output_info.shape_vector,
                name="data_z_ub0", scope=tik.scope_ubuf)

            tik_instance.vmul(output_info.overflow,
                              data_z_ub0[output_info.rep * VECTOR],
                              data_x_ub[output_info.rep * VECTOR],
                              data_anchor_wh[output_info.rep * VECTOR],
                              1,
                              1, 1, 1,
                              VECTOR_NUM, VECTOR_NUM, VECTOR_NUM)

            # calculate input * data_anchor_wh + data_anchor_xaya
            data_z_ub = tik_instance.Tensor(
                input_info.dtype_boundary_predictions, output_info.shape_vector,
                name="data_z_ub", scope=tik.scope_ubuf)

            tik_instance.vadd(output_info.overflow,
                              data_z_ub[output_info.rep * VECTOR],
                              data_z_ub0[output_info.rep * VECTOR],
                              data_anchor_xaya[output_info.rep * VECTOR],
                              1,
                              1, 1, 1,
                              VECTOR_NUM, VECTOR_NUM, VECTOR_NUM)

            # copy ub to gm
            tik_instance.data_move(data_z[current_handling_times * input_info.n_max],
                                   data_z_ub,
                                   0,
                                   1, input_info.burst_x,
                                   0, 0)

    data_boundary_predictions = kwargs['data_boundary_predictions']
    data_anchors = kwargs['data_anchors']
    data_z = kwargs['data_z']

    cal()


@util.check_input_type(dict, dict, dict, str)
def decode_boundaries_target(boundary_predictions, anchors, boundary_encoded,
                             kernel_name="cce_decode_boundaries_target_fpLINE"):
    """
    calculating data

    Parameters
    ----------
    boundary_predictions : dict
        shape and dtype of input
    anchors : dict
        shape and dtype of input
    boundary_encoded : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "decode_boundaries_target"

    Returns
    -------
    None
    """
    util.check_kernel_name(kernel_name)
    input_info = Input(
        shape_boundary_predictions=boundary_predictions.get("shape"),
        shape_anchors=anchors.get("shape"),
        dtype_boundary_predictions=boundary_predictions.get("dtype").lower(),
        dtype_anchors=anchors.get("dtype").lower()
    )
    input_info.set_nmax(n_max=NMAX)
    output_info = Output()

    total_handling_times, last_handling_n = check_input(
        boundary_predictions=boundary_predictions,
        anchors=anchors,
        boundary_encoded=boundary_encoded,
        n_max=input_info.n_max)
    tik_instance = tik.Tik(tik.Dprofile(), True)
    # tensor init
    data_boundary_predictions, data_anchors, \
        data_z = get_gm(tik_instance=tik_instance,
                        dtype=input_info.dtype_anchors,
                        shape1=input_info.shape_boundary_predictions,
                        shape2=input_info.shape_anchors,
                        name1="data_boundary_predictions",
                        name2="data_anchors",
                        name3="data_z",
                        scope=tik.scope_gm)

    if total_handling_times > 0:
        with tik_instance.for_range(0, total_handling_times) as current_handling_times:
            # current_handling_times:
            output_info.set_burst_num(burst_num=input_info.n_max)

            # number of LINE*LINE
            output_info.update(
                n_vector=int_ceil_div(output_info.burst_num, MATRIX_NUM),
                n_matrix=int_ceil_div(output_info.burst_num * FOUR, MATRIX_NUM)
            )
            output_info.update(
                shape_vector=(output_info.n_vector, LINE, LINE),
                shape_matrix=(output_info.n_matrix * FOUR, LINE, LINE)
            )

            # move x_gm to ub times
            # move y_gm to ub times
            input_info.update(
                burst_x=int_ceil_div(output_info.burst_num, LINE),
                burst_y=int_ceil_div(output_info.burst_num * FOUR, LINE)
            )

            output_info.update(
                rep=output_info.burst_num // VECTOR,
                overflow=0
            )

            process_calculate(tik_instance=tik_instance,
                              input_info=input_info,
                              output_info=output_info,
                              current_handling_times=current_handling_times,
                              data_boundary_predictions=data_boundary_predictions,
                              data_anchors=data_anchors,
                              data_z=data_z)

    current_handling_times = total_handling_times
    if last_handling_n > 0:
        output_info.set_burst_num(burst_num=last_handling_n)

        # number of LINE*LINE
        output_info.update(
            n_vector=int_ceil_div(output_info.burst_num, MATRIX_NUM),
            n_matrix=int_ceil_div(output_info.burst_num * FOUR, MATRIX_NUM)
        )
        output_info.update(
            shape_vector=(output_info.n_vector, LINE, LINE),
            shape_matrix=(output_info.n_matrix * FOUR, LINE, LINE)
        )

        # move x_gm to ub times
        # move y_gm to ub times
        input_info.update(
            burst_x=int_ceil_div(output_info.burst_num, LINE),
            burst_y=int_ceil_div(output_info.burst_num * FOUR, LINE)
        )

        output_info.update(
            rep=0,
            overflow=output_info.burst_num - VECTOR *
            (output_info.burst_num // VECTOR)
        )

        process_end(tik_instance=tik_instance,
                    input_info=input_info,
                    output_info=output_info,
                    current_handling_times=current_handling_times,
                    data_boundary_predictions=data_boundary_predictions,
                    data_anchors=data_anchors,
                    data_z=data_z)

    # build_cce
    tik_instance.BuildCCE(
        kernel_name=kernel_name,
        inputs=[data_boundary_predictions, data_anchors],
        outputs=[data_z])

    return tik_instance
