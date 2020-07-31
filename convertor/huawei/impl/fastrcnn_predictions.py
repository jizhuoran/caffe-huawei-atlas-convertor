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

add_tik
"""
# pylint: disable=C0302
# pylint: disable=R0913

from te import tik
from topi.cce import util

ONE = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5
SIX = 6
EIGHT = 8
TEN = 10
SIXTEEN = 16
BLOCK_DATA = 32
SIXTY_FOUR = 64
MASK = 128
MAXTRIX = 256
# DOWN_FACTOR = (65404/(720*1024*2)) ** 0.5
DOWN_FACTOR = 0.05
REC_FACTOR = 1/DOWN_FACTOR
IF_USE_V200 = ("aic", "vec", "hisi-es")


def ceil_div(divisor_a, divisor_b):
    """
    round up function

    Paramater:
    :param divisor_a: int.
    :param divisor_b: int.
    :return: int
    """
    if divisor_b == 0:
        raise RuntimeError("division by zero")
    return (divisor_a + divisor_b - 1) // divisor_b


def check_fastrcnn_predictions_params(rois, score, other_inputs):
    """
    The params check function of fastrcnn_predictions

    Parameters:
    ----------
    Returns : All transformed params.
    ----------
    """
    shape_x = rois.get("shape")
    shape_y = score.get("shape")
    dtype_x = rois.get("dtype").lower()
    dtype_y = score.get("dtype").lower()

    # Abnormality test
    if dtype_x != dtype_y:
        raise RuntimeError("dtype of inputs and output should be consistent")
    if dtype_x != 'float16':
        raise RuntimeError("dtype of rois should be float16")
    if len(shape_x) != TWO:
        raise RuntimeError("dimension of rois should be TWO")
    if len(shape_y) != TWO:
        raise RuntimeError("dimension of score score should be TWO")
    if shape_y[0] not in (16, 32, 96):
        raise RuntimeError("first dimension of score should be 16 or 32 or 96")
    if not 1 <= shape_y[1] - 1 <= 32:
        raise RuntimeError("second dimension of rois should in [1, 32]")
    if shape_x[0] != shape_y[0] * (shape_y[1] - 1):
        raise RuntimeError("first dimension of rois should be consistent to"
                           " first dimension mul second dimension of score")
    if shape_x[1] != FOUR:
        raise RuntimeError("second dimension of rois should be FOUR")
    check_fastrcnn_predictions_params_2(score, other_inputs)


def check_fastrcnn_predictions_params_2(score, other_inputs):
    """
    The params check function of fastrcnn_predictions

    Parameters:
    ----------
    Returns : All transformed params.
    ----------
    """
    shape_y = score.get("shape")
    sorted_rois_check = other_inputs[0].get("shape")
    sorted_scores_check = other_inputs[1].get("shape")
    sorted_classes_check = other_inputs[2].get("shape")
    nms_threshold_check = other_inputs[3]
    score_threshold_check = other_inputs[4]
    k_check = other_inputs[5]

    if not 0 <= nms_threshold_check <= 1:
        raise RuntimeError("the nms_threshold should in [0, 1]")
    if not 0 <= score_threshold_check <= 1:
        raise RuntimeError("the score_threshold should in [0, 1]")
    if sorted_rois_check[0] != k_check or sorted_scores_check[0] != k_check \
            or sorted_classes_check[0] != k_check:
        raise RuntimeError("first dimension of output should be consistent")
    if k_check != shape_y[0]:
        raise RuntimeError("the k should be equle to N")


class InitShape:
    """
    calculating the shape
    Returns
    -------
    None
    """
    def __init__(self, input_dict):
        self.n_x = input_dict[1].get("shape")[0]
        self.classes = input_dict[1].get("shape")[1] - 1

        # number of SIXTEEN * SIXTEEN
        self.n_maxtrix = ceil_div(self.n_x, SIXTEEN)

        self.classes_maxtrix_x = ceil_div(self.classes, FOUR)
        self.classes_maxtrix_y = ceil_div(self.classes, SIXTEEN)

        # shape of calculate
        self.shape_maxtrix_x = (self.classes_maxtrix_x, self.n_maxtrix, SIXTEEN, SIXTEEN)
        self.shape_maxtrix_y = (self.classes_maxtrix_y, self.n_maxtrix, SIXTEEN, SIXTEEN)

    def set_input_shape(self, n_x):
        """
        set_input_shape
        :return:
        """
        self.n_x = n_x

    def set_output_shape(self, classes):
        """
        set_input_shape
        :return:
        """
        self.classes = classes


class InitGmTensor:
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, input_dict):
        shape_x = input_dict[0].get("shape")
        shape_y = input_dict[1].get("shape")
        shape_z0 = input_dict[2].get("shape")
        shape_z1 = input_dict[3].get("shape")
        self.data_x = tik_instance.Tensor(
            "float16", shape_x, name="data_x", scope=tik.scope_gm)
        self.data_y = tik_instance.Tensor(
            "float16", shape_y, name="data_y", scope=tik.scope_gm)
        self.gm_sorted_rois = tik_instance.Tensor(
            "float16", shape_z0, name="gm_sorted_rois", scope=tik.scope_gm)
        self.gm_sorted_scores = tik_instance.Tensor(
            "float16", shape_z1, name="gm_sorted_scores", scope=tik.scope_gm)
        self.gm_sorted_classes = tik_instance.Tensor(
            "float16", shape_z1, name="gm_sorted_classes", scope=tik.scope_gm)

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


class InitMiddleTensor:
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, shape):
        self.data_ub_proposal = tik_instance.Tensor("float16",
                                                    (shape.classes, shape.n_maxtrix,
                                                     SIXTEEN, EIGHT),
                                                    name="data_ub_proposal",
                                                    scope=tik.scope_ubuf)

        self.data_ub_proposal_topk = tik_instance.Tensor("float16",
                                                         (shape.classes, shape.n_maxtrix,
                                                          SIXTEEN, EIGHT),
                                                         name="data_ub_proposal_topk",
                                                         scope=tik.scope_ubuf)
        tik_instance.vector_dup(MASK,
                                self.data_ub_proposal_topk,
                                0,
                                shape.n_maxtrix * shape.classes,
                                ONE,
                                EIGHT)

        tik_instance.vector_dup(MASK,
                                self.data_ub_proposal,
                                0,
                                shape.n_maxtrix * shape.classes,
                                ONE,
                                EIGHT)

        self.data_x_ub_trs2 = tik_instance.Tensor("float16",
                                                  (shape.classes, shape.n_maxtrix, FOUR, SIXTEEN),
                                                  name="data_x_ub_trs2",
                                                  scope=tik.scope_ubuf)
        self.data_x_ub_trs3 = tik_instance.Tensor("float16",
                                                  (shape.classes, FOUR, shape.n_maxtrix, SIXTEEN),
                                                  name="data_x_ub_trs3",
                                                  scope=tik.scope_ubuf)

        self.cal_topk_k = tik_instance.Scalar(dtype="int32", init_value=0)

        self.data_ub_class = tik_instance.Tensor("float16",
                                                 (shape.classes, shape.n_maxtrix, ONE, SIXTEEN),
                                                 name="data_ub_class",
                                                 scope=tik.scope_ubuf)
        self.data_y_ub_trs1 = tik_instance.Tensor("float16",
                                                  (shape.classes, shape.n_maxtrix, 16),
                                                  name="data_y_ub_trs1",
                                                  scope=tik.scope_ubuf)

    def set_data_x_ub_trs2(self, data_x_ub_trs2):
        """
        data_x_ub_trs2
        :return:
        """
        self.data_x_ub_trs2 = data_x_ub_trs2

    def set_data_y_ub_trs1(self, data_y_ub_trs1):
        """
        data_y_ub_trs1
        :return:
        """
        self.data_y_ub_trs1 = data_y_ub_trs1


class InitFirstTensorV100:
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, shape):
        self.data_x_ub = tik_instance.Tensor("float16",
                                             shape.shape_maxtrix_x,
                                             name="data_x_ub",
                                             scope=tik.scope_ubuf)
        tik_instance.vector_dup(MASK,
                                self.data_x_ub,
                                0,
                                TWO * shape.n_maxtrix * shape.classes_maxtrix_x,
                                ONE, EIGHT)

        self.data_x_ub_trs = tik_instance.Tensor("float16",
                                                 shape.shape_maxtrix_x,
                                                 name="data_x_ub_trs",
                                                 scope=tik.scope_ubuf)
        self.data_x_ub_trs1 = tik_instance.Tensor("float16",
                                                  (shape.classes, shape.n_maxtrix, FOUR, SIXTEEN),
                                                  name="data_x_ub_trs1",
                                                  scope=tik.scope_ubuf)

        self.data_y_ub = tik_instance.Tensor("float16",
                                             shape.shape_maxtrix_y,
                                             name="data_y_ub",
                                             scope=tik.scope_ubuf)

        tik_instance.vector_dup(MASK,
                                self.data_y_ub,
                                0,
                                TWO * shape.n_maxtrix * shape.classes_maxtrix_y,
                                ONE,
                                EIGHT)

        self.data_y_ub_trs = tik_instance.Tensor("float16",
                                                 shape.shape_maxtrix_y,
                                                 name="data_y_ub_trs",
                                                 scope=tik.scope_ubuf)

        self.data_ub_class = tik_instance.Tensor("float16",
                                                 (shape.classes, shape.n_maxtrix, ONE, SIXTEEN),
                                                 name="data_ub_class",
                                                 scope=tik.scope_ubuf)

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


class InitFirstTensorV200:
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, shape):
        self.data_x_ub = tik_instance.Tensor("float16",
                                             (shape.n_maxtrix, 16, shape.classes * 4),
                                             name="data_x_ub",
                                             scope=tik.scope_ubuf)
        self.data_y_ub = tik_instance.Tensor("float16",
                                             (shape.n_maxtrix, 16, shape.classes + 1),
                                             name="data_y_ub",
                                             scope=tik.scope_ubuf)
        self.data_x_ub_trs = tik_instance.Tensor("float16",
                                                 (shape.classes * 4, shape.n_maxtrix, 16),
                                                 name="data_x_ub_trs",
                                                 scope=tik.scope_ubuf)
        self.data_x_ub_trs1 = tik_instance.Tensor("float16",
                                                  (shape.classes, 4, shape.n_maxtrix, 16),
                                                  name="data_x_ub_trs1",
                                                  scope=tik.scope_ubuf)
        self.data_y_ub_trs = tik_instance.Tensor("float16",
                                                 (shape.classes + 1, shape.n_maxtrix, 16),
                                                 name="data_y_ub_trs",
                                                 scope=tik.scope_ubuf)

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


class InitClassTensor:
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, shape, score_threshold):
        self.data_one = tik_instance.Tensor("float16",
                                            (shape.n_maxtrix, SIXTEEN),
                                            name="data_one",
                                            scope=tik.scope_ubuf)
        tik_instance.vector_dup(shape.n_x,
                                self.data_one,
                                ONE,
                                ONE,
                                ONE,
                                shape.n_maxtrix)
        self.data_zero = tik_instance.Tensor("float16",
                                             (shape.n_maxtrix, SIXTEEN),
                                             name="data_zero",
                                             scope=tik.scope_ubuf)
        tik_instance.vector_dup(shape.n_x,
                                self.data_zero,
                                0,
                                ONE,
                                ONE,
                                shape.n_maxtrix)
        self.threshold_ub = tik_instance.Tensor("float16",
                                                (shape.n_maxtrix, SIXTEEN),
                                                name="threshold_ub",
                                                scope=tik.scope_ubuf)
        tik_instance.vector_dup(shape.n_x,
                                self.threshold_ub,
                                score_threshold,
                                ONE,
                                ONE, shape.n_maxtrix)

        self.vsel_score_ub = tik_instance.Tensor("float16", (shape.n_maxtrix, SIXTEEN),
                                                 name="vsel_score_ub", scope=tik.scope_ubuf)
        self.num = tik_instance.Scalar(dtype="int32")

        self.vsel_score_ub1 = tik_instance.Tensor("float16", (SIXTEEN,),
                                                  name="vsel_score_ub1", scope=tik.scope_ubuf)
        self.vsel_score_ub2 = tik_instance.Tensor("int32", (EIGHT,),
                                                  name="vsel_score_ub2", scope=tik.scope_ubuf)

    def set_num_0(self, data_zero):
        """
        data_y_ub_trs1
        :return:
        """
        self.data_zero = data_zero

    def set_num(self, num):
        """
        data_anchor_wh
        :return:
        """
        self.num = num


class InitNmsTensor:
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, shape):
        self.join_ub = tik_instance.Tensor("float16",
                                           [shape.n_maxtrix, SIXTEEN, SIXTEEN],
                                           name="join_ub",
                                           scope=tik.scope_ubuf)
        self.join_ub1 = tik_instance.Tensor("float16",
                                            [shape.n_maxtrix, SIXTEEN, SIXTEEN],
                                            name="join_ub1",
                                            scope=tik.scope_ubuf)

        self.iou_ub = tik_instance.Tensor("float16",
                                          [shape.n_maxtrix, SIXTEEN, SIXTEEN],
                                          name="iou_ub",
                                          scope=tik.scope_ubuf)

        self.vrpac_ub = tik_instance.Tensor("float16",
                                            [shape.n_maxtrix, SIXTEEN],
                                            name="vrpac_ub",
                                            scope=tik.scope_ubuf)

        self.sup_matrix_ub = tik_instance.Tensor("uint16",
                                                 [shape.n_maxtrix, SIXTEEN, SIXTEEN],
                                                 name="sup_matrix_ub",
                                                 scope=tik.scope_ubuf)

        self.sup_vec_ub = tik_instance.Tensor("uint16",
                                              [shape.n_maxtrix, SIXTEEN],
                                              name="sup_vec_ub",
                                              scope=tik.scope_ubuf)

    def set_iou_ub(self, iou_ub):
        """
        data_y_ub_trs1
        :return:
        """
        self.iou_ub = iou_ub

    def set_join_ub(self, join_ub):
        """
        data_anchor_wh
        :return:
        """
        self.join_ub = join_ub


class InitNmsV200Tensor:
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, shape):
        self.num_nms = tik_instance.Scalar(dtype="uint32")
        self.nms_tensor_0 = tik_instance.Tensor(dtype="float16",
                                                shape=(6, shape.n_x),
                                                name="nms_tensor_0",
                                                scope=tik.scope_ubuf)
        self.nms_tensor_1 = tik_instance.Tensor(dtype="float16",
                                                shape=(6, shape.n_x),
                                                name="nms_tensor_1",
                                                scope=tik.scope_ubuf)
        tik_instance.vector_dup(shape.n_x,
                                self.nms_tensor_1, 0, 6, ONE, shape.n_maxtrix)
        self.nms_tensor_pattern = tik_instance.Tensor(dtype="uint16",
                                                      shape=(SIXTEEN, ),
                                                      name="nms_tensor_pattern",
                                                      scope=tik.scope_ubuf)

    def set_iou_ub(self, num_nms):
        """
        data_y_ub_trs1
        :return:
        """
        self.num_nms = num_nms

    def set_join_ub(self, nms_tensor_0):
        """
        data_anchor_wh
        :return:
        """
        self.nms_tensor_0 = nms_tensor_0


class InitFinalTensorV100:
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, topk_k):
        self.sorted_scores_ub = tik_instance.Tensor(
            "float16",
            (topk_k, ONE),
            name="sorted_score_ub",
            scope=tik.scope_ubuf)
        tik_instance.vector_dup(SIXTEEN, self.sorted_scores_ub,
                                0, topk_k // SIXTEEN, ONE, ONE)

        self.sorted_classes_ub = tik_instance.Tensor(
            "float16",
            (topk_k, ONE),
            name="sorted_classes_ub",
            scope=tik.scope_ubuf)
        tik_instance.vector_dup(SIXTEEN, self.sorted_classes_ub,
                                0, topk_k // SIXTEEN, ONE, ONE)

        self.sorted_rois_ub = tik_instance.Tensor(
            "float16",
            (SIXTEEN, topk_k),
            name="sorted_rois_ub",
            scope=tik.scope_ubuf)

        tik_instance.vector_dup(MASK, self.sorted_rois_ub, 0,
                                topk_k // SIXTEEN * TWO, ONE, EIGHT)
        tik_instance.vector_dup(BLOCK_DATA, self.sorted_rois_ub, ONE,
                                topk_k // SIXTEEN, ONE, TWO)

        self.sorted_rois_ub1 = tik_instance.Tensor(
            "float16",
            (SIXTEEN, topk_k),
            name="sorted_rois_ub1",
            scope=tik.scope_ubuf)

        self.sorted_rois_ub2 = tik_instance.Tensor(
            "float16",
            (SIXTEEN, topk_k),
            name="sorted_rois_ub2",
            scope=tik.scope_ubuf)

        self.sorted_rois_ub_str = tik_instance.Tensor(
            "float16",
            (topk_k, SIXTEEN),
            name="sorted_rois_ub_str",
            scope=tik.scope_ubuf)

        self.sorted_rois_ub_str1 = tik_instance.Tensor(
            "float16",
            (topk_k, FOUR),
            name="sorted_rois_ub_str1",
            scope=tik.scope_ubuf)

    def set_sorted_scores_ub(self, sorted_scores_ub):
        """
        sorted_scores_ub
        :return:
        """
        self.sorted_scores_ub = sorted_scores_ub

    def set_sorted_classes_ub(self, sorted_classes_ub):
        """
        sorted_classes_ub
        :return:
        """
        self.sorted_classes_ub = sorted_classes_ub


class InitScalar:
    """
    init the input scalar
    """
    def __init__(self, tik_instance):
        self.dump_0 = tik_instance.Scalar(dtype="float16", init_value=0.0)
        self.dump_factor = tik_instance.Scalar(dtype="float16", init_value=DOWN_FACTOR)
        self.dump_1 = tik_instance.Scalar(dtype="float16", init_value=1.0)

    def set_dump_factor(self, dump_0):
        """
        :param dump_0: float
        :return:
        """
        self.dump_0 = dump_0

    def set_rec_dump_factor(self, dump_1):
        """
        :param dump_1:float
        :return:
        """
        self.dump_1 = dump_1


def topk(tik_ins, proposal_list_num, middle_tensor, start):
    """
    :param tik_ins:
    :param proposal_list_num:
    :param middle_tensor:
    :param start:
    :return:
    """
    data_ub_proposal = middle_tensor.data_ub_proposal
    data_ub_proposal_topk = middle_tensor.data_ub_proposal_topk
    tik_ins.vrpsort16(data_ub_proposal_topk[start],
                      data_ub_proposal[start],
                      proposal_list_num)
    tik_ins.vadds(MASK,
                  data_ub_proposal[start],
                  data_ub_proposal_topk[start],
                  0,
                  proposal_list_num, ONE, ONE, EIGHT, EIGHT)
    if proposal_list_num == 36:
        tik_ins.vmrgsort4(data_ub_proposal,
                          (data_ub_proposal_topk,
                           data_ub_proposal_topk[MASK],
                           data_ub_proposal_topk[MASK * TWO],
                           data_ub_proposal_topk[MASK * THREE]),
                          SIXTEEN,
                          False, SIXTEEN - ONE, EIGHT + ONE)
        tik_ins.vmrgsort4(data_ub_proposal_topk,
                          (data_ub_proposal,
                           data_ub_proposal[MASK * FOUR],
                           data_ub_proposal[MASK * FOUR * TWO],
                           data_ub_proposal[MASK * FOUR * THREE]),
                          SIXTY_FOUR,
                          False, SIXTEEN - ONE, TWO)
        tik_ins.vmrgsort4(data_ub_proposal,
                          (data_ub_proposal_topk,
                           data_ub_proposal_topk[MASK * SIXTEEN],
                           data_ub_proposal_topk[MASK * BLOCK_DATA],
                           data_ub_proposal_topk),
                          (MAXTRIX, MAXTRIX, SIXTY_FOUR, 0),
                          False, EIGHT - ONE, ONE)
    elif proposal_list_num == 60:
        tik_ins.vmrgsort4(data_ub_proposal,
                          (data_ub_proposal_topk,
                           data_ub_proposal_topk[MASK],
                           data_ub_proposal_topk[MASK * TWO],
                           data_ub_proposal_topk[MASK * THREE]),
                          SIXTEEN,
                          False, SIXTEEN - ONE, SIXTEEN - ONE)
        tik_ins.vmrgsort4(data_ub_proposal_topk,
                          (data_ub_proposal,
                           data_ub_proposal[MASK * FOUR],
                           data_ub_proposal[MASK * FOUR * TWO],
                           data_ub_proposal[MASK * FOUR * THREE]),
                          SIXTY_FOUR,
                          False, SIXTEEN - ONE, THREE)
        tik_ins.vmrgsort4(data_ub_proposal,
                          (data_ub_proposal_topk,
                           data_ub_proposal_topk[MASK * SIXTEEN],
                           data_ub_proposal_topk[MASK * SIXTEEN * TWO],
                           data_ub_proposal_topk[MASK * SIXTEEN * THREE]),
                          (MAXTRIX, MAXTRIX, MAXTRIX, SIXTY_FOUR),
                          False, SIXTEEN - ONE, ONE)
        tik_ins.vmrgsort4(data_ub_proposal_topk,
                          (data_ub_proposal,
                           data_ub_proposal[MASK * 52],
                           data_ub_proposal[MASK * 56],
                           data_ub_proposal),
                          (52 * SIXTEEN, SIXTY_FOUR, SIXTY_FOUR, 0),
                          False, EIGHT - ONE, ONE)
        tik_ins.vadds(MASK,
                      data_ub_proposal,
                      data_ub_proposal_topk,
                      0,
                      proposal_list_num, ONE, ONE, EIGHT, EIGHT)
    elif proposal_list_num > ONE:
        if proposal_list_num > 7:
            # for topk_index in range((proposal_list_num - TWO) // SIX):
            with tik_ins.for_range(0, (proposal_list_num - TWO) // SIX) as topk_index:
                src_list = ((ONE + SIX * topk_index) * MASK,
                            (TWO + SIX * topk_index) * MASK,
                            (THREE + SIX * topk_index) * MASK)
                element_count_list = ((ONE + SIX * topk_index) * SIXTEEN,
                                      SIXTEEN,
                                      SIXTEEN,
                                      SIXTEEN)
                tik_ins.vmrgsort4(data_ub_proposal[start],
                                  (data_ub_proposal_topk[start],
                                   data_ub_proposal_topk[start + src_list[0]],
                                   data_ub_proposal_topk[start + src_list[1]],
                                   data_ub_proposal_topk[start + src_list[TWO]]),
                                  element_count_list,
                                  False, 15, ONE)

                src_list_2 = ((FOUR + SIX * topk_index) * MASK,
                              (FIVE + SIX * topk_index) * MASK,
                              (SIX + SIX * topk_index) * MASK)
                element_count_list_2 = ((FOUR + THREE * topk_index) * SIXTEEN,
                                        SIXTEEN,
                                        SIXTEEN,
                                        SIXTEEN)
                tik_ins.vmrgsort4(data_ub_proposal_topk[start],
                                  (data_ub_proposal[start],
                                   data_ub_proposal[start + src_list_2[0]],
                                   data_ub_proposal[start + src_list_2[1]],
                                   data_ub_proposal[start + src_list_2[TWO]]),
                                  element_count_list_2,
                                  False, 15, ONE)

        topk_index1 = (proposal_list_num - TWO) // SIX
        topk_index_last = (proposal_list_num - TWO) % SIX + ONE
        topk_index_last_2 = (proposal_list_num - TWO) % THREE + ONE
        if topk_index_last > THREE:
            src_list = ((ONE + SIX * topk_index1) * MASK,
                        (TWO + SIX * topk_index1) * MASK,
                        (THREE + SIX * topk_index1) * MASK)

            element_count_list = ((ONE + SIX * topk_index1) * SIXTEEN,
                                  SIXTEEN,
                                  SIXTEEN,
                                  SIXTEEN)
            tik_ins.vmrgsort4(data_ub_proposal[start],
                              (data_ub_proposal_topk[start],
                               data_ub_proposal_topk[start + src_list[0]],
                               data_ub_proposal_topk[start + src_list[ONE]],
                               data_ub_proposal_topk[start + src_list[TWO]]),
                              element_count_list,
                              False, 15, ONE)

            src_list_2 = ((FOUR + SIX * topk_index1) * MASK,
                          (FOUR + SIX * topk_index1 + topk_index_last_2 // TWO) * MASK,
                          (FOUR + SIX * topk_index1 + topk_index_last_2 // THREE * TWO) * MASK)
            element_count_list_2 = ((FOUR + SIX * topk_index1) * SIXTEEN,
                                    SIXTEEN,
                                    SIXTEEN,
                                    SIXTEEN)
            tik_ins.vmrgsort4(data_ub_proposal_topk[start],
                              (data_ub_proposal[start],
                               data_ub_proposal[start + src_list_2[0]],
                               data_ub_proposal[start + src_list_2[1]],
                               data_ub_proposal[start + src_list_2[TWO]]),
                              element_count_list_2,
                              False, TWO ** (topk_index_last_2 + ONE) - ONE, ONE)
            tik_ins.vadds(MASK,
                          data_ub_proposal[start],
                          data_ub_proposal_topk[start],
                          0,
                          proposal_list_num, ONE, ONE, EIGHT, EIGHT)

        else:
            src_list = ((ONE + SIX * topk_index1) * MASK,
                        (ONE + SIX * topk_index1 + topk_index_last // TWO) * MASK,
                        (ONE + SIX * topk_index1 + topk_index_last // THREE * TWO) * MASK)
            element_count_list = ((ONE + SIX * topk_index1) * SIXTEEN,
                                  SIXTEEN,
                                  SIXTEEN,
                                  SIXTEEN)
            tik_ins.vmrgsort4(data_ub_proposal[start],
                              (data_ub_proposal_topk[start],
                               data_ub_proposal_topk[start + src_list[0]],
                               data_ub_proposal_topk[start + src_list[ONE]],
                               data_ub_proposal_topk[start + src_list[TWO]]),
                              element_count_list,
                              False, TWO ** (topk_index_last_2 + ONE) - ONE, ONE)


def combine_proposals(tik_instance, gm_tensor, shape, scalar, middle_tensor):
    """
    :param tik_instance:
    :param gm_tensor:
    :param shape:
    :param scalar:
    :param middle_tensor:
    :return:
    """

    # init first tensor
    for i in range(shape.classes):
        tik_instance.vector_dup(SIXTEEN,
                                middle_tensor.data_ub_class[shape.n_x * i],
                                i + ONE, shape.n_maxtrix, ONE, ONE)
    #if tik.Dprofile().get_product_name() not in IF_USE_V200:
    if True:
        first_tensor = InitFirstTensorV100(tik_instance, shape)

        # cpoy data_x form gm to ub
        with tik_instance.for_range(0, shape.classes_maxtrix_x) as j:
            with tik_instance.for_range(0, FOUR) as i:
                tik_instance.data_move(first_tensor.data_x_ub[j, 0, i, 0],
                                       gm_tensor.data_x[FOUR * shape.classes * i + j * SIXTEEN],
                                       0,
                                       ceil_div(shape.n_x, FOUR), ONE,
                                       shape.classes - ONE, THREE)

        # cpoy data_y form gm to ub
        with tik_instance.for_range(0, shape.classes_maxtrix_y) as j:
            with tik_instance.for_range(0, SIXTEEN) as i:
                tik_instance.data_move(first_tensor.data_y_ub[j, 0, i, 0],
                                       gm_tensor.data_y[ONE
                                                        + (ONE + shape.classes) * i + j * SIXTEEN],
                                       0,
                                       ceil_div(shape.n_x, SIXTEEN), ONE,
                                       shape.classes, SIXTEEN - ONE)

        # transpose data_x
        with tik_instance.for_range(0, shape.n_maxtrix * shape.classes_maxtrix_x) as i:
            tik_instance.vtranspose(first_tensor.data_x_ub_trs[MAXTRIX * i],
                                    first_tensor.data_x_ub[MAXTRIX * i])

        # transpose data_y
        with tik_instance.for_range(0, shape.n_maxtrix * shape.classes_maxtrix_y) as i:
            tik_instance.vtranspose(first_tensor.data_y_ub_trs[MAXTRIX * i],
                                    first_tensor.data_y_ub[MAXTRIX * i])

        # shrink data_x to prevent overflow fp16 and arrangement data_x
        with tik_instance.for_range(0, shape.classes // FOUR) as c_i:
            with tik_instance.for_range(0, FOUR) as i:
                tik_instance.vmuls(SIXTY_FOUR,
                                   first_tensor.data_x_ub_trs1[(c_i * FOUR + i) * FOUR * shape.n_x],
                                   first_tensor.data_x_ub_trs[i * SIXTY_FOUR
                                                              + c_i * shape.n_x * SIXTEEN],
                                   scalar.dump_factor, shape.n_maxtrix,
                                   ONE, ONE, FOUR, SIXTEEN)

        with tik_instance.for_range(0, shape.classes % FOUR) as j:
            tik_instance.vmuls(SIXTY_FOUR,
                               first_tensor.data_x_ub_trs1[(j + shape.classes // FOUR * FOUR)
                                                           * FOUR * shape.n_x],
                               first_tensor.data_x_ub_trs[j * SIXTY_FOUR + shape.classes // FOUR
                                                          * shape.n_x * SIXTEEN],
                               scalar.dump_factor, shape.n_maxtrix,
                               ONE, ONE, FOUR, SIXTEEN)

        # add 1 for x1 and y1 because nms operate would reduces 1
        with tik_instance.for_range(0, TWO) as x_i:
            tik_instance.vadds(SIXTEEN, middle_tensor.data_x_ub_trs2[SIXTEEN * x_i],
                               first_tensor.data_x_ub_trs1[SIXTEEN * x_i],
                               1,
                               shape.n_maxtrix * shape.classes, ONE, ONE, FOUR, FOUR)

        with tik_instance.for_range(0, TWO) as x_i:
            tik_instance.vadds(SIXTEEN, middle_tensor.data_x_ub_trs2[32 + SIXTEEN * x_i],
                               first_tensor.data_x_ub_trs1[32 + SIXTEEN * x_i],
                               scalar.dump_0,
                               shape.n_maxtrix * shape.classes, ONE, ONE, FOUR, FOUR)

        # arrangement data_y
        with tik_instance.for_range(0, shape.classes // SIXTEEN) as c_i:
            with tik_instance.for_range(0, SIXTEEN) as i:
                tik_instance.vadds(SIXTEEN,
                                   middle_tensor.data_y_ub_trs1[(c_i * SIXTEEN + i) * shape.n_x],
                                   first_tensor.data_y_ub_trs[(i + c_i * shape.n_x) * SIXTEEN],
                                   scalar.dump_0, shape.n_maxtrix, ONE, SIXTEEN, ONE, SIXTEEN)

        with tik_instance.for_range(0, shape.classes % SIXTEEN) as i:
            tik_instance.vadds(SIXTEEN,
                               middle_tensor.data_y_ub_trs1[(i + shape.classes // SIXTEEN * SIXTEEN)
                                                            * shape.n_x],
                               first_tensor.data_y_ub_trs[i * SIXTEEN + shape.classes // SIXTEEN
                                                          * shape.n_x * SIXTEEN],
                               scalar.dump_0,
                               shape.n_maxtrix, ONE, SIXTEEN, ONE, SIXTEEN)

        with tik_instance.for_range(0, shape.classes) as class_i:
            with tik_instance.for_range(0, shape.n_maxtrix) as k:
                tik_instance.vadds(SIXTY_FOUR,
                                   middle_tensor.data_x_ub_trs3[class_i, 0, k, 0],
                                   middle_tensor.data_x_ub_trs2[class_i, k, 0, 0],
                                   scalar.dump_0,
                                   ONE,
                                   shape.n_maxtrix, ONE, FOUR * shape.n_maxtrix, FOUR)

    else:
        first_tensor = InitFirstTensorV200(tik_instance, shape)

        tik_instance.data_move(first_tensor.data_x_ub, gm_tensor.data_x, 0,
                               1, shape.n_maxtrix * shape.classes * 4, 0, 0)

        tik_instance.data_move(first_tensor.data_y_ub, gm_tensor.data_y, 0,
                               1, shape.n_maxtrix * (shape.classes + 1), 0, 0)

        tik_instance.v4dtrans(False, first_tensor.data_x_ub_trs,
                              first_tensor.data_x_ub, shape.n_x, shape.classes * 4)

        tik_instance.v4dtrans(False, first_tensor.data_y_ub_trs,
                              first_tensor.data_y_ub, shape.n_x, shape.classes + 1)

        tik_instance.vmuls(SIXTY_FOUR, first_tensor.data_x_ub_trs1, first_tensor.data_x_ub_trs,
                           DOWN_FACTOR, shape.n_maxtrix * shape.classes, 1, 1, 4, 4)

        tik_instance.vadds(SIXTY_FOUR, middle_tensor.data_x_ub_trs3, first_tensor.data_x_ub_trs1,
                           0, shape.n_maxtrix * shape.classes, 1, 1, 4, 4)
        with tik_instance.for_range(0, shape.classes) as class_i:
            tik_instance.vadds(shape.n_x,
                               middle_tensor.data_x_ub_trs3[class_i, 0, 0, 0],
                               first_tensor.data_x_ub_trs1[class_i, 0, 0, 0],
                               1,
                               2, 1, 1, shape.n_maxtrix, shape.n_maxtrix)

        tik_instance.vadds(16, middle_tensor.data_y_ub_trs1, first_tensor.data_y_ub_trs[1, 0, 0],
                           0, shape.n_maxtrix * shape.classes, 1, 1, 1, 1)


def cal_score_filter_num(tik_instance, shape, class_tensor, middle_tensor, other_parm):
    """
    :param tik_instance:
    :param shape:
    :param class_tensor:
    :param middle_tensor:
    :param other_parm:
    :return:
    """
    class_i = other_parm[0]
    scalar = other_parm[1]

    # form proposal
    tik_instance.vconcat(middle_tensor.data_ub_proposal[class_i, 0, 0, 0],
                         middle_tensor.data_ub_class[class_i, 0, 0, 0],
                         shape.n_maxtrix, FIVE)

    tik_instance.vconcat(middle_tensor.data_ub_proposal[class_i, 0, 0, 0],
                         middle_tensor.data_y_ub_trs1[class_i, 0, 0],
                         shape.n_maxtrix, FOUR)

    # with tik_instance.for_range(0, shape.classes) as class_i:
    with tik_instance.for_range(0, FOUR) as j:
        tik_instance.vconcat(middle_tensor.data_ub_proposal[class_i, 0, 0, 0],
                             middle_tensor.data_x_ub_trs3[class_i, j, 0, 0],
                             shape.n_maxtrix, j)

    cmpmask = tik_instance.vcmp_gt(shape.n_x,
                                   middle_tensor.data_y_ub_trs1[shape.n_x * class_i],
                                   class_tensor.threshold_ub,
                                   ONE,
                                   ONE)
    tik_instance.vsel(shape.n_x,
                      0,
                      class_tensor.vsel_score_ub,
                      cmpmask,
                      class_tensor.data_one,
                      class_tensor.data_zero,
                      ONE, ONE, ONE, ONE,
                      shape.n_maxtrix, shape.n_maxtrix, shape.n_maxtrix)

    tik_instance.vcadd(shape.n_x,
                       class_tensor.vsel_score_ub1,
                       class_tensor.vsel_score_ub,
                       ONE, ONE, ONE, shape.n_maxtrix)

    tik_instance.vconv(EIGHT, "round", class_tensor.vsel_score_ub2,
                       class_tensor.vsel_score_ub1, ONE, ONE, ONE, ONE, ONE)

    class_tensor.num.set_as(class_tensor.vsel_score_ub2[0])
    # topk sort for every class
    topk(tik_instance,
         shape.n_maxtrix,
         middle_tensor,
         class_i * shape.n_x * EIGHT)

    with tik_instance.for_range(class_tensor.num, shape.n_x) as index:
        middle_tensor.data_ub_proposal[class_i * shape.n_x * EIGHT
                                       + index * EIGHT + FOUR].set_as(scalar.dump_0)


def nms(tik_instance, shape, scalar, middle_tensor, other_parm):
    """
    :param tik_instance:
    :param other_parm:
    :param shape:
    :param scalar:
    :param middle_tensor:
    :return:
    """
    nms_thres = other_parm[0]
    class_i = other_parm[1]
    nms_tensor = other_parm[2]
    class_tensor = other_parm[3]
    nms_v200 = other_parm[4]
    tik_instance.vrpac(nms_tensor.vrpac_ub, middle_tensor.data_ub_proposal[class_i, 0, 0, 0],
                       shape.n_maxtrix)

    with tik_instance.for_range(0, shape.n_maxtrix) as j:

        # calculate iou area
        tik_instance.viou(nms_tensor.iou_ub, middle_tensor.data_ub_proposal[class_i, 0, 0, 0],
                          middle_tensor.data_ub_proposal[class_i, j, 0, 0],
                          j + ONE)

        # calculate aadd area
        tik_instance.vaadd(nms_tensor.join_ub,
                           nms_tensor.vrpac_ub,
                           nms_tensor.vrpac_ub[j, 0],
                           j + ONE)

        # aadd area muls nms_threshold_new
        tik_instance.vmuls(MASK, nms_tensor.join_ub1,
                           nms_tensor.join_ub, nms_thres,
                           TWO * (j + ONE), ONE, ONE, EIGHT, EIGHT)

        # compare and generate suppression matrix
        tik_instance.vcmpv_gt(nms_tensor.sup_matrix_ub,
                              nms_tensor.iou_ub, nms_tensor.join_ub1,
                              TWO * (j + ONE), ONE, ONE, EIGHT, EIGHT)

        # generate rpn_cor_ir
        rpn_cor_ir = tik_instance.set_rpn_cor_ir(0)
        # non-diagonal
        rpn_cor_ir = tik_instance.rpn_cor(nms_tensor.sup_matrix_ub,
                                          nms_tensor.sup_vec_ub, ONE, ONE, j)
        with tik_instance.if_scope(j > 0):
            rpn_cor_ir = tik_instance.rpn_cor(
                nms_tensor.sup_matrix_ub[j * SIXTEEN], nms_tensor.sup_vec_ub, ONE, ONE, ONE)
        # get final sup_vec
        tik_instance.rpn_cor_diag(nms_tensor.sup_vec_ub[j * SIXTEEN],
                                  nms_tensor.sup_matrix_ub[j * SIXTEEN], rpn_cor_ir)

    #if tik.Dprofile().get_product_name() not in IF_USE_V200:
    if True:
        middle_tensor.cal_topk_k.set_as(middle_tensor.cal_topk_k + class_tensor.num)
        # if rpn_cor_ir == ONE ; make scores 0
        with tik_instance.for_range(0, class_tensor.num) as k:
            with tik_instance.if_scope(nms_tensor.sup_vec_ub[k] != 0):
                middle_tensor.cal_topk_k.set_as(middle_tensor.cal_topk_k - ONE)
                middle_tensor.data_ub_proposal[class_i * shape.n_x * EIGHT + k
                                               * EIGHT + FOUR].set_as(scalar.dump_0)

    else:
        with tik_instance.for_range(0, 6) as i:
            tik_instance.vextract(
                nms_v200.nms_tensor_0[i, 0],
                middle_tensor.data_ub_proposal[class_i * shape.n_x * EIGHT],
                shape.n_maxtrix,
                i)

        temp_tensor = nms_tensor.sup_vec_ub.reinterpret_cast_to("float16")
        cmpmask = tik_instance.vcmp_eq(class_tensor.num, temp_tensor,
                                       class_tensor.data_zero, ONE, ONE)

        tik_instance.mov_cmpmask_to_tensor(
            nms_v200.nms_tensor_pattern.reinterpret_cast_to("uint64"),
            cmpmask)

        with tik_instance.for_range(0, 6) as i:
            tik_instance.vreduce(class_tensor.num,
                                 nms_v200.nms_tensor_1[i, 0],
                                 nms_v200.nms_tensor_0[i, 0],
                                 nms_v200.nms_tensor_pattern,
                                 ONE,
                                 ONE,
                                 shape.n_maxtrix,
                                 0,
                                 0,
                                 nms_v200.num_nms,
                                 "counter")
        with tik_instance.for_range(0, 6) as i:
            tik_instance.vconcat(middle_tensor.data_ub_proposal[class_i * shape.n_x * EIGHT],
                                 nms_v200.nms_tensor_1[i, 0],
                                 shape.n_maxtrix,
                                 i)
        middle_tensor.cal_topk_k.set_as(middle_tensor.cal_topk_k + nms_v200.num_nms)


def postprocessing(tik_instance, gm_tensor, shape, middle_tensor):
    """
    :param tik_instance:
    :param gm_tensor:
    :param shape:
    :param middle_tensor:
    :return:
    """
    topk_k = shape.n_x
    final_tensor = InitFinalTensorV100(tik_instance, topk_k)

    # extract scores and classes from sorted proposals
    if tik.Dprofile().get_product_name() == "mini":
        with tik_instance.for_range(0, middle_tensor.cal_topk_k) as i:
            final_tensor.sorted_scores_ub[i].set_as(middle_tensor.data_ub_proposal[i * EIGHT
                                                                                   + FOUR])
            final_tensor.sorted_classes_ub[i].set_as(middle_tensor.data_ub_proposal[i * EIGHT
                                                                                    + FIVE])
    else:
        tik_instance.vextract(
            final_tensor.sorted_scores_ub,
            middle_tensor.data_ub_proposal,
            ceil_div(middle_tensor.cal_topk_k, SIXTEEN),
            FOUR)
        tik_instance.vextract(
            final_tensor.sorted_classes_ub,
            middle_tensor.data_ub_proposal,
            middle_tensor.cal_topk_k // SIXTEEN,
            FIVE)

        with tik_instance.for_range(middle_tensor.cal_topk_k // SIXTEEN * SIXTEEN,
                                    middle_tensor.cal_topk_k) as i:
            final_tensor.sorted_classes_ub[i].set_as(middle_tensor.data_ub_proposal[i * EIGHT
                                                                                    + FIVE])

    # move ub_data to gm
    tik_instance.data_move(gm_tensor.gm_sorted_scores,
                           final_tensor.sorted_scores_ub,
                           0,
                           ONE, topk_k // SIXTEEN,
                           0, 0)

    tik_instance.data_move(gm_tensor.gm_sorted_classes,
                           final_tensor.sorted_classes_ub,
                           0,
                           ONE, topk_k // SIXTEEN,
                           0, 0)

    # extract rois from sorted_proposals
    with tik_instance.for_range(0, FOUR) as box_i:
        tik_instance.vextract(
            final_tensor.sorted_rois_ub[box_i * topk_k],
            middle_tensor.data_ub_proposal,
            ceil_div(middle_tensor.cal_topk_k, SIXTEEN),
            box_i)

    # x1,y1 reduce ONE
    tik_instance.vadds(BLOCK_DATA,
                       final_tensor.sorted_rois_ub1,
                       final_tensor.sorted_rois_ub,
                       -1,
                       topk_k // SIXTEEN, ONE, ONE, TWO, TWO)
    tik_instance.vadds(BLOCK_DATA,
                       final_tensor.sorted_rois_ub1[topk_k * TWO],
                       final_tensor.sorted_rois_ub[topk_k * TWO],
                       0,
                       topk_k // SIXTEEN, ONE, ONE, TWO, TWO)

    tik_instance.vector_dup(MASK, final_tensor.sorted_rois_ub2, 0, shape.n_maxtrix*TWO, ONE, EIGHT)

    # x1,y1,x2,y2 multipy rec_factor
    with tik_instance.for_range(0, shape.n_maxtrix) as i:
        tik_instance.vmuls(SIXTY_FOUR,
                           final_tensor.sorted_rois_ub2[256 * i],
                           final_tensor.sorted_rois_ub1[SIXTEEN * i],
                           REC_FACTOR, ONE, ONE, shape.n_maxtrix, FOUR, FOUR * shape.n_maxtrix)

    # SIXTEEN * k(vaild shape is FOUR * k) transpose k * SIXTEEN(vaild shape is k * FOUR)
    with tik_instance.for_range(0, shape.n_maxtrix) as i:
        tik_instance.vtranspose(final_tensor.sorted_rois_ub_str[256 * i],
                                final_tensor.sorted_rois_ub2[256 * i])

    tik_instance.vmuls(SIXTEEN,
                       final_tensor.sorted_rois_ub_str[middle_tensor.cal_topk_k * SIXTEEN],
                       final_tensor.sorted_rois_ub_str[middle_tensor.cal_topk_k * SIXTEEN],
                       0,
                       topk_k - middle_tensor.cal_topk_k,
                       ONE, ONE, ONE, ONE)

    # cut out topk * FOUR from topk * SIXTEEN
    tik_instance.vmrgch(
        final_tensor.sorted_rois_ub_str1,
        final_tensor.sorted_rois_ub_str,
        TWO * topk_k // SIXTEEN)

    # copy ub to gm
    tik_instance.data_move(gm_tensor.gm_sorted_rois,
                           final_tensor.sorted_rois_ub_str1,
                           # data_ub_proposal[shape.n_x * FOUR],
                           0,
                           ONE, topk_k * FOUR // SIXTEEN,
                           0, 0)


def fastrcnn_predictions_compute(input_dict, input_param, kernel_name):
    """
    :param input_dict: rois, score, sorted_rois, sorted_scores, sorted_classes
    :param input_param: nms_threshold_new, score_threshold, k
    :param kernel_name:str
        default value is "fastrcnn_predictions"
    :return:
    """

    nms_threshold_new, score_threshold, k = input_param

    # initial the tik container
    tik_instance = tik.Tik(tik.Dprofile(), True)

    # init shape param
    shape = InitShape(input_dict)

    # init gm_tensor
    gm_tensor = InitGmTensor(tik_instance, input_dict)

    # scalr init
    scalar = InitScalar(tik_instance)

    # init middle tensor
    middle_tensor = InitMiddleTensor(tik_instance, shape)

    class_tensor = InitClassTensor(tik_instance, shape, score_threshold)

    with tik_instance.new_stmt_scope():
        # combine the rois, scores and classes into proposals
        combine_proposals(tik_instance, gm_tensor, shape, scalar, middle_tensor)

    nms_tensor = InitNmsTensor(tik_instance, shape)

    nms_v200 = InitNmsV200Tensor(tik_instance, shape)

    # calu for every class
    with tik_instance.for_range(0, shape.classes) as class_i:
        # calculate scorenum
        cal_score_filter_num(tik_instance, shape, class_tensor, middle_tensor,
                             (class_i, scalar))

        nms(tik_instance, shape, scalar, middle_tensor,
            (nms_threshold_new, class_i, nms_tensor, class_tensor, nms_v200))

    with tik_instance.if_scope(middle_tensor.cal_topk_k > k):
        middle_tensor.cal_topk_k.set_as(k)

    # final topk sort
    topk(tik_instance, shape.classes * shape.n_maxtrix, middle_tensor, 0)

    # postprocess
    postprocessing(tik_instance, gm_tensor, shape, middle_tensor)

    # build_cce
    tik_instance.BuildCCE(
        kernel_name=kernel_name,
        inputs=[gm_tensor.data_x, gm_tensor.data_y],
        outputs=[gm_tensor.gm_sorted_rois,
                 gm_tensor.gm_sorted_scores,
                 gm_tensor.gm_sorted_classes])
    return tik_instance


@util.check_input_type(dict, dict, dict, dict, dict, float, float, int, str)
def fastrcnn_predictions(rois, score,
                         sorted_rois, sorted_scores, sorted_classes,
                         nms_threshold, score_threshold, k,
                         kernel_name="fastrcnn_predictions"):

    """
    :param rois: input
        dict
        shape and dtype of box;
        shape = (N, classes, 4), dtype = "float16"
        N [SIXTEEN, 32, 96]
        1 <= classes  <= 32
    :param score: input
        dict
        shape and dtype of score
        shape = (N, classes + 1), dtype = "float16"
    :param sorted_rois: outout
        dict
        shape and dtype of sorted_box
        shape = (k, 4), format = "ND", dtype = "float16"
    :param sorted_scores: outout
        dict
        shape and dtype of sorted_score
        shape = (k, 1), format = "ND", dtype = "float16"
    :param sorted_classes: outout
        dict
        shape and dtype of sorted_classes
        shape = (k, 1), format = "ND", dtype = "float16"
    :param nms_threshold:  attr
        float
         filtering threshold of nms
         (0, 1)
    :param score_threshold: attr
        float
         filtering threshold of score
         (0, 1)
    :param k: attr
        int
        Number of boxes for final output
        k == N
    :param kernel_name: str
        default value is "fastrcnn_predictions"
    :return: tik_instance
    """

    check_fastrcnn_predictions_params(
        rois, score,
        (sorted_rois, sorted_scores, sorted_classes, nms_threshold, score_threshold, k))
    util.check_kernel_name(kernel_name)

    nms_threshold_new = nms_threshold / (1.0 + nms_threshold)
    tik_instance = fastrcnn_predictions_compute(
        (rois, score, sorted_rois, sorted_scores, sorted_classes),
        (nms_threshold_new, score_threshold, k),
        kernel_name)

    return tik_instance
