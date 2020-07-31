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

score_filter_pre_sort
"""

# pylint: disable=C0302
# pylint: disable=R0902
# pylint: disable=R0915
# pylint: disable=R0914
# pylint: disable=R0913
# pylint: disable=R0912


from te import tik

from topi.cce import util


SHAPE_SIZE_LIMIT = 709920
MAX_HEIGHT = 2000
MAX_WIDTH = 3000
MAX_TOPK = 6000
CONFIG_ONE_NEG = -1
CONFIG_ONE = 1
CONFIG_TWO = 2
CONFIG_THREE = 3
CONFIG_FOUR = 4
CONFIG_FIVE = 5
CONFIG_SIX = 6
CONFIG_SEVEN = 7
CONFIG_EIGHT = 8
CONFIG_NINE = 9
CONFIG_TEN = 10
CONFIG_TWELEV = 12
CONFIG_SIXTEEN = 16
CONFIG_DATA_ALIGN = 32
CONFIG_DATA_TRANS = 64
CONFIG_MASK = 128
CONFIG_UNIT = 1024
MATRIX = 256
MAX_REP_TIME = 255
CONFIG_TOPK = 4096
CONFIG_TOPK2 = 6144
CONFIG_LEN = 1536
# 1MB can save 1024*1024/16 = 65536 proposals
L1_MAX_NUM = 46080
CONFIG_FP16 = 65404
IF_USE_V200 = ("aic", "vec")


def ceil_div(num_a, num_bulk):
    """
    calculate number of bulk needed
    num_a: the num of  input boxes
    num_bulk : the num of elements each bulk
    return  the num of bulk at least needed
    """

    return (num_a + num_bulk - CONFIG_ONE) // num_bulk


class InitGmTensor:
    """
    define some tensors in the DDR
    used for v200
    """
    def __init__(self, tik_instance, input_para):
        shape_in = input_para[0]
        num_one_core = input_para[CONFIG_ONE]
        num_topk = input_para[CONFIG_TWO]
        core_num = input_para[CONFIG_THREE]
        data_type = input_para[CONFIG_FOUR]

        # input tensor
        self.rois_gm = tik_instance.Tensor(data_type,
                                           (shape_in, CONFIG_FOUR),
                                           name="rois_gm",
                                           scope=tik.scope_gm)
        self.prob_gm = tik_instance.Tensor(data_type,
                                           (shape_in, CONFIG_ONE),
                                           name="prob_gm",
                                           scope=tik.scope_gm)

        # output tensors in DDR
        self.out_gm = tik_instance.Tensor(data_type,
                                          (core_num * (num_topk + CONFIG_TWO), CONFIG_EIGHT),
                                          name="out_gm",
                                          scope=tik.scope_gm)
        self.proposal_num = tik_instance.Tensor("uint32",
                                                (core_num, CONFIG_EIGHT),
                                                name="proposal_num",
                                                scope=tik.scope_gm)

        # large tensor for cache
        self.proposal_gm = tik_instance.Tensor(data_type,
                                               (core_num * num_one_core, CONFIG_EIGHT),
                                               name="proposal_gm",
                                               scope=tik.scope_gm,
                                               is_workspace=True)

    def set_proposal_num(self, num_in):
        """
        set the num_post_score
        param num_in:
        return:
        """

        # self.proposal_num[index, 0].set_as(num_in)
        self.proposal_num = num_in

    def set_out_gm(self, num_in):
        """
        set num_total_cb
        param num_in:
        return:
        """
        self.out_gm = num_in


class InitGlobalTensor:
    """
    define some tensors in the L1
    also some scalars for each core
    used for v200
    """
    def __init__(self, tik_instance, input_para):
        num_topk = input_para[0]
        data_type = input_para[CONFIG_ONE]

        # Tensors after the score filter
        self.proposal_cb = tik_instance.Tensor(data_type,
                                               (L1_MAX_NUM + CONFIG_FOUR, CONFIG_EIGHT),
                                               name="proposal_cb",
                                               scope=tik.scope_cbuf)

        self.mem_swap = tik_instance.Tensor(data_type,
                                            (num_topk * CONFIG_TWO + CONFIG_FOUR, CONFIG_EIGHT),
                                            name="men_swap",
                                            scope=tik.scope_cbuf)

        self.mem_swap2 = tik_instance.Tensor(data_type,
                                             (num_topk + CONFIG_FOUR, CONFIG_EIGHT),
                                             name="mem_swap2",
                                             scope=tik.scope_cbuf)
        # some scalars
        self.num_post_score = tik_instance.Scalar("uint32",
                                                  name="num_post_score",
                                                  init_value=0)

        self.num_total_cb = tik_instance.Scalar("uint32",
                                                name="num_total_cb",
                                                init_value=0)

        self.num_total_gm = tik_instance.Scalar("uint32",
                                                name="num_total_gm",
                                                init_value=0)

        self.flag_cb = tik_instance.Scalar("uint8",
                                           name="flag_cb",
                                           init_value=CONFIG_ONE)

    def set_num_post_score(self, num_in):
        """
        set the num_post_score
        param num_in:
        return:
        """
        self.num_post_score = num_in

    def set_num_total_cb(self, num_in):
        """
        set num_total_cb
        param num_in:
        return:
        """
        self.num_total_cb = num_in


class InitMiddleTensor:
    """
    define some tensors used in the score filter
    in v200 branch
    """
    def __init__(self, tik_instance, const_one_core, data_type):
        self.rois_ub = tik_instance.Tensor(data_type,
                                           (const_one_core, CONFIG_FOUR),
                                           name="rois_ub",
                                           scope=tik.scope_ubuf)
        self.prob_ub = tik_instance.Tensor(data_type,
                                           (const_one_core, CONFIG_ONE),
                                           name="prob_ub",
                                           scope=tik.scope_ubuf)

        self.rois_ub_tran = tik_instance.Tensor(data_type,
                                                (CONFIG_FOUR, const_one_core),
                                                name="rois_ub_tran",
                                                scope=tik.scope_ubuf)

        self.mask_ub = tik_instance.Tensor("uint16",
                                           (ceil_div(const_one_core // CONFIG_MASK, CONFIG_TWO),
                                            CONFIG_SIXTEEN),
                                           name="mask_ub",
                                           scope=tik.scope_ubuf)

        self.rois_ub_select = tik_instance.Tensor(data_type,
                                                  (CONFIG_FOUR, const_one_core),
                                                  name="rois_ub_select",
                                                  scope=tik.scope_ubuf)
        self.prob_ub_select = tik_instance.Tensor(data_type,
                                                  (const_one_core, CONFIG_ONE),
                                                  name="prob_ub_select",
                                                  scope=tik.scope_ubuf)

        self.num_scorefilter = tik_instance.Scalar("uint32",
                                                   name="num_scorefilter",
                                                   init_value=0)

        self.proposal_ub = tik_instance.Tensor(data_type,
                                               (const_one_core, CONFIG_EIGHT),
                                               name="proposal_ub",
                                               scope=tik.scope_ubuf)

    def set_num_scorefilter(self, num_in):
        """
        set num_scorefilter
        param num_in:
        return:
        """
        self.num_scorefilter = num_in

    def set_prob_ub_select(self, prob_tensor):
        """
        set prob_ub_select
        param prob_tensor:
        return:
        """
        self.prob_ub_select = prob_tensor


def score_filter_one(tik_instance, data_gm, data_cb, data_ub, input_param):
    """
    perform the score filter one whole part
    param tik_instance:
    param data_gm:
    param data_cb:
    param data_ub:
    param input_param:
        0  score_threshold  : the score threshold
        1  num_offset       :  the offset
        2  const_one_core   :  the num processing one time
        3  num_real         :  real num
        4  score_filter     : whether score_filter is needed
        5  core_offset

    return: None
    """

    score_threshold = input_param[0]
    # the current offset of the src
    num_offset = input_param[CONFIG_ONE]
    const_one_core = input_param[CONFIG_TWO]
    num_real = input_param[CONFIG_THREE]
    score_filter = input_param[CONFIG_FOUR]
    # the original address offset the src  for proposal saving
    core_offset = input_param[CONFIG_FIVE]

    # move data from DDR to UB
    tik_instance.data_move(data_ub.rois_ub,
                           data_gm.rois_gm[num_offset, 0],
                           0,
                           CONFIG_ONE,
                           ceil_div(num_real, CONFIG_FOUR),
                           0,
                           0)
    # tail
    if num_real < const_one_core:
        tik_instance.vector_dup(CONFIG_MASK,
                                data_ub.prob_ub,
                                score_threshold,
                                const_one_core // CONFIG_MASK,
                                CONFIG_ONE,
                                CONFIG_EIGHT)
    tik_instance.data_move(data_ub.prob_ub,
                           data_gm.prob_gm[num_offset],
                           0,
                           CONFIG_ONE,
                           ceil_div(num_real, CONFIG_SIXTEEN),
                           0,
                           0)
    # ==================================>
    # transpose the boxes from N*4 to 4*N
    # =====>  using the vreduce
    # x1,   3,  0001000100010001
    tik_instance.vreduce(CONFIG_MASK,
                         data_ub.rois_ub_tran,
                         data_ub.rois_ub,
                         CONFIG_THREE,
                         ceil_div(num_real, CONFIG_DATA_ALIGN),
                         CONFIG_ONE,
                         CONFIG_EIGHT,
                         0,
                         0,
                         None,
                         "normal")
    # y1,   4,  0010001000100010
    tik_instance.vreduce(CONFIG_MASK,
                         data_ub.rois_ub_tran[CONFIG_ONE, 0],
                         data_ub.rois_ub,
                         CONFIG_FOUR,
                         ceil_div(num_real, CONFIG_DATA_ALIGN),
                         CONFIG_ONE,
                         CONFIG_EIGHT,
                         0,
                         0,
                         None,
                         "normal")
    # x2,   5,  0100010001000100
    tik_instance.vreduce(CONFIG_MASK,
                         data_ub.rois_ub_tran[CONFIG_TWO, 0],
                         data_ub.rois_ub,
                         CONFIG_FIVE,
                         ceil_div(num_real, CONFIG_DATA_ALIGN),
                         CONFIG_ONE,
                         CONFIG_EIGHT,
                         0,
                         0,
                         None,
                         "normal")
    # y2,  6,  1000100010001000
    tik_instance.vreduce(CONFIG_MASK,
                         data_ub.rois_ub_tran[CONFIG_THREE, 0],
                         data_ub.rois_ub,
                         CONFIG_SIX,
                         ceil_div(num_real, CONFIG_DATA_ALIGN),
                         CONFIG_ONE,
                         CONFIG_EIGHT,
                         0,
                         0,
                         None,
                         "normal")

    if score_filter:
        # =================================>
        # perform the score filter
        # obtain the mask
        tik_instance.vcmpvs_gt(data_ub.mask_ub,
                               data_ub.prob_ub,
                               score_threshold,
                               const_one_core // CONFIG_MASK,
                               CONFIG_ONE,
                               CONFIG_EIGHT)
        # score
        tik_instance.vector_dup(CONFIG_MASK,
                                data_ub.prob_ub_select,
                                score_threshold,
                                const_one_core // CONFIG_MASK,
                                CONFIG_ONE,
                                CONFIG_EIGHT)
        tik_instance.vreduce(CONFIG_MASK,
                             data_ub.prob_ub_select,
                             data_ub.prob_ub,
                             data_ub.mask_ub,
                             ceil_div(const_one_core, CONFIG_MASK),
                             CONFIG_ONE,
                             CONFIG_EIGHT,
                             CONFIG_EIGHT,
                             0,
                             data_ub.num_scorefilter,
                             "normal")

        with tik_instance.for_range(0, CONFIG_FOUR) as t_cyc:
            tik_instance.vreduce(CONFIG_MASK,
                                 data_ub.rois_ub_select[t_cyc, 0],
                                 data_ub.rois_ub_tran[t_cyc, 0],
                                 data_ub.mask_ub,
                                 ceil_div(const_one_core, CONFIG_MASK),
                                 CONFIG_ONE,
                                 CONFIG_EIGHT,
                                 CONFIG_EIGHT,
                                 0,
                                 None,
                                 "normal")

        # combine into proposals
        # three are proposals left
        with tik_instance.if_scope(data_ub.num_scorefilter > 0):
            with tik_instance.for_range(0, CONFIG_FOUR) as t_cyc:
                tik_instance.vconcat(data_ub.proposal_ub,
                                     data_ub.rois_ub_select[t_cyc, 0],
                                     ceil_div(data_ub.num_scorefilter, CONFIG_SIXTEEN),
                                     t_cyc)
            tik_instance.vconcat(data_ub.proposal_ub,
                                 data_ub.prob_ub_select,
                                 ceil_div(data_ub.num_scorefilter, CONFIG_SIXTEEN),
                                 CONFIG_FOUR)

            # update num of proposals left
            data_cb.num_post_score.set_as(data_cb.num_post_score + data_ub.num_scorefilter)

            # three is still mem in L1 buffer
            with tik_instance.if_scope(data_cb.flag_cb == CONFIG_ONE):
                num_here = ceil_div(data_ub.num_scorefilter, CONFIG_TWO) * CONFIG_TWO + \
                           data_cb.num_total_cb
                with tik_instance.if_scope(num_here <= L1_MAX_NUM):
                    tik_instance.data_move(data_cb.proposal_cb[data_cb.num_total_cb, 0],
                                           data_ub.proposal_ub,
                                           0,
                                           CONFIG_ONE,
                                           ceil_div(data_ub.num_scorefilter, CONFIG_TWO),
                                           0,
                                           0)
                    data_cb.num_total_cb.set_as(num_here)
                # move data into DDR
                with tik_instance.else_scope():
                    num_here = ceil_div(data_ub.num_scorefilter, CONFIG_TWO) * CONFIG_TWO + \
                               data_cb.num_total_gm
                    data_cb.flag_cb.set_as(0)
                    tik_instance.data_move(data_gm.proposal_gm[core_offset + data_cb.num_total_gm,
                                                               0],
                                           data_ub.proposal_ub,
                                           0,
                                           CONFIG_ONE,
                                           ceil_div(data_ub.num_scorefilter, CONFIG_TWO),
                                           0,
                                           0)
                    data_cb.num_total_gm.set_as(num_here)
            # move data into DDR
            with tik_instance.else_scope():
                num_here = ceil_div(data_ub.num_scorefilter, CONFIG_TWO) * CONFIG_TWO + \
                           data_cb.num_total_gm
                tik_instance.data_move(data_gm.proposal_gm[core_offset + data_cb.num_total_gm, 0],
                                       data_ub.proposal_ub,
                                       0,
                                       CONFIG_ONE,
                                       ceil_div(data_ub.num_scorefilter, CONFIG_TWO),
                                       0,
                                       0)
                data_cb.num_total_gm.set_as(num_here)

    else:
        with tik_instance.for_range(0, CONFIG_FOUR) as t_cyc:
            tik_instance.vconcat(data_ub.proposal_ub,
                                 data_ub.rois_ub_tran[t_cyc, 0],
                                 ceil_div(num_real, CONFIG_SIXTEEN),
                                 t_cyc)
        tik_instance.vconcat(data_ub.proposal_ub,
                             data_ub.prob_ub,
                             ceil_div(num_real, CONFIG_SIXTEEN),
                             CONFIG_FOUR)

        # update num of proposals left
        data_cb.num_post_score.set_as(data_cb.num_post_score + num_real)

        tik_instance.data_move(data_gm.proposal_gm[num_offset, 0],
                               data_ub.proposal_ub,
                               0,
                               CONFIG_ONE,
                               ceil_div(num_real, CONFIG_TWO),
                               0,
                               0)
        data_cb.num_total_gm.set_as(data_cb.num_post_score)


def score_filter_processing(tik_instance, data_gm, data_cb, input_param):
    """
    perform the score filter
    param tik_instance:
    param data_gm:
    param data_cb: the tensors and scalars in the L1 buffer

    input_param: a list of parameter
        0  score_threshold
        1  core_offset
        2  shape_in:  box to be processing by this core
        3  score_filter
        4  dtype

    return: None
    """

    score_threshold = input_param[0]
    core_offset = input_param[CONFIG_ONE]
    shape_in = input_param[CONFIG_TWO]
    score_filter = input_param[CONFIG_THREE]

    # tiling
    # num of box to processing one time, -128 to avoid rep_time problems
    const_one_core = CONFIG_UNIT * CONFIG_FOUR - CONFIG_MASK
    num_cycle = shape_in // const_one_core
    num_left = shape_in % const_one_core

    with tik_instance.new_stmt_scope():
        data_ub = InitMiddleTensor(tik_instance, const_one_core, input_param[CONFIG_FOUR])
        with tik_instance.for_range(0, num_cycle, thread_num=CONFIG_ONE) as t_thread:
            num_offset = core_offset + const_one_core * t_thread
            score_filter_one(tik_instance, data_gm, data_cb, data_ub,
                             (score_threshold, num_offset, const_one_core,
                              const_one_core, score_filter, core_offset))

        # =====> the tail
        if num_left > 0:
            num_offset = core_offset + const_one_core * num_cycle
            score_filter_one(tik_instance, data_gm, data_cb, data_ub,
                             (score_threshold, num_offset, const_one_core,
                              num_left, score_filter, core_offset))


def tik_scalar_min(tik_instance, val_a, val_b, result_):
    """
    get min val

    Parameters
    ----------
    tik_inst: tik instance
    val_a: value a
    val_b: value b
    result_: value result

    Returns
    -------
    NA
    """

    with tik_instance.if_scope(val_a < val_b):
        result_.set_as(val_a)
    with tik_instance.else_scope():
        result_.set_as(val_b)


def tik_topk_6114(tik_instance, data_ub, actual_num, score_threshold):
    """
    the input data in data_ub_a
    the output results in data_ub_a
    param tik_instance:
    param data_tensor:
    return: None
    """

    data_ub_a = data_ub[0]
    data_ub_b = data_ub[CONFIG_ONE]
    num_real = tik_instance.Scalar("uint32", name="num_real", init_value=actual_num)

    # init the non_used area
    with tik_instance.if_scope(num_real < CONFIG_TOPK2):
        rep_time = ceil_div((CONFIG_TOPK2 - num_real), CONFIG_SIXTEEN)
        offset = CONFIG_TOPK2 - rep_time * CONFIG_SIXTEEN

        with tik_instance.if_scope(rep_time <= MAX_REP_TIME):
            tik_instance.vector_dup(CONFIG_MASK,
                                    data_ub_b[offset, 0],
                                    score_threshold,
                                    rep_time,
                                    CONFIG_ONE,
                                    CONFIG_EIGHT)
        with tik_instance.else_scope():
            tik_instance.vector_dup(CONFIG_MASK,
                                    data_ub_b[offset, 0],
                                    score_threshold,
                                    MAX_REP_TIME,
                                    CONFIG_ONE,
                                    CONFIG_EIGHT)
            rep_time = rep_time - MAX_REP_TIME
            offset = offset + MAX_REP_TIME * CONFIG_SIXTEEN
            tik_instance.vector_dup(CONFIG_MASK,
                                    data_ub_b[offset, 0],
                                    score_threshold,
                                    rep_time,
                                    CONFIG_ONE,
                                    CONFIG_EIGHT)

    # sort each 16 proposals
    rep_time = ceil_div(num_real, CONFIG_SIXTEEN)
    with tik_instance.if_scope(rep_time <= MAX_REP_TIME):
        tik_instance.vrpsort16(data_ub_b,
                               data_ub_a,
                               rep_time)
    with tik_instance.else_scope():
        tik_instance.vrpsort16(data_ub_b,
                               data_ub_a,
                               MAX_REP_TIME)
        rep_time = rep_time - MAX_REP_TIME
        tik_instance.vrpsort16(data_ub_b[MAX_REP_TIME * CONFIG_SIXTEEN, 0],
                               data_ub_a[MAX_REP_TIME * CONFIG_SIXTEEN, 0],
                               rep_time)

    # 16 --> 64
    rep_str = CONFIG_SIXTEEN
    tik_instance.vmrgsort4(data_ub_a,
                           (data_ub_b[0, 0],
                            data_ub_b[rep_str * CONFIG_ONE, 0],
                            data_ub_b[rep_str * CONFIG_TWO, 0],
                            data_ub_b[rep_str * CONFIG_THREE, 0]),
                           (rep_str, rep_str, rep_str, rep_str),
                           False,
                           CONFIG_SIXTEEN - CONFIG_ONE,
                           ceil_div(num_real, rep_str * CONFIG_FOUR))

    # 64 --> 256
    rep_str = rep_str * CONFIG_FOUR
    tik_instance.vmrgsort4(data_ub_b,
                           (data_ub_a[0, 0],
                            data_ub_a[rep_str * CONFIG_ONE, 0],
                            data_ub_a[rep_str * CONFIG_TWO, 0],
                            data_ub_a[rep_str * CONFIG_THREE, 0]),
                           (rep_str, rep_str, rep_str, rep_str),
                           False,
                           CONFIG_SIXTEEN - CONFIG_ONE,
                           ceil_div(num_real, rep_str * CONFIG_FOUR))

    # 256 --> 1024
    rep_str = rep_str * CONFIG_FOUR
    tik_instance.vmrgsort4(data_ub_a,
                           (data_ub_b[0, 0],
                            data_ub_b[rep_str * CONFIG_ONE, 0],
                            data_ub_b[rep_str * CONFIG_TWO, 0],
                            data_ub_b[rep_str * CONFIG_THREE, 0]),
                           (rep_str, rep_str, rep_str, rep_str),
                           False,
                           CONFIG_SIXTEEN - CONFIG_ONE,
                           ceil_div(num_real, rep_str * CONFIG_FOUR))

    # 1024 * 3 --> 3072
    rep_str = rep_str * CONFIG_FOUR
    with tik_instance.if_scope(tik.all(num_real > rep_str, num_real <= rep_str * CONFIG_THREE)):
        tik_instance.vmrgsort4(data_ub_b,
                               (data_ub_a[0, 0],
                                data_ub_a[rep_str * CONFIG_ONE, 0],
                                data_ub_a[rep_str * CONFIG_TWO, 0],
                                data_ub_a[rep_str * CONFIG_TWO, 0]),
                               (rep_str, rep_str, rep_str, rep_str),
                               False,
                               CONFIG_EIGHT - CONFIG_ONE,
                               CONFIG_ONE)
        tik_instance.data_move(data_ub_a,
                               data_ub_b,
                               0,
                               CONFIG_ONE,
                               ceil_div(num_real, CONFIG_TWO),
                               0, 0)

    with tik_instance.if_scope(num_real > rep_str * CONFIG_THREE):
        tik_instance.vmrgsort4(data_ub_b,
                               (data_ub_a[0, 0],
                                data_ub_a[rep_str * CONFIG_ONE, 0],
                                data_ub_a[rep_str * CONFIG_TWO, 0],
                                data_ub_a[rep_str * CONFIG_TWO, 0]),
                               (rep_str, rep_str, rep_str, rep_str),
                               False,
                               CONFIG_EIGHT - CONFIG_ONE,
                               CONFIG_ONE)
        rep_time = ceil_div(num_real, rep_str) - CONFIG_THREE
        # rep_time = rep_time * rep_time - rep_time + CONFIG_ONE
        valid_bit_scalar = tik_instance.Scalar("uint32", name="valid_bit_scalar", init_value=0)
        valid_bit_scalar.set_as(rep_time * rep_time - rep_time + CONFIG_ONE)
        tik_instance.vmrgsort4(data_ub_b[rep_str * CONFIG_THREE, 0],
                               (data_ub_a[rep_str * CONFIG_THREE, 0],
                                data_ub_a[rep_str * CONFIG_FOUR, 0],
                                data_ub_a[rep_str * CONFIG_FIVE, 0],
                                data_ub_a[rep_str * CONFIG_FIVE, 0]),
                               (rep_str, rep_str, rep_str, rep_str),
                               False,
                               valid_bit_scalar,
                               CONFIG_ONE)
        # 3072 * 2 --> 6144
        rep_str = rep_str * CONFIG_THREE
        tik_instance.vmrgsort4(data_ub_a,
                               (data_ub_b[0, 0],
                                data_ub_b[rep_str * CONFIG_ONE, 0],
                                data_ub_b[rep_str * CONFIG_ONE, 0],
                                data_ub_b[rep_str * CONFIG_ONE, 0]),
                               (rep_str, num_real - rep_str, rep_str, rep_str),
                               False,
                               CONFIG_THREE,
                               CONFIG_ONE)


def tik_topk_internal_sort(tik_instance, data_gm, param_list, core_offset):
    """
    the proposals can be moved in at one time
    param tik_instance:
    param data_gm: a list
            src_tensor = data_gm[0] : the tensor store the original proposals
            dst_tensor = data_gm[1] : the tensor to store the results in DDR / L1
    param_list: a list
        score_threshold:  for dump
        num_actual: actual num of the input proposals
        topk_k : the max num needed after topk

    core_offset : the offset of the input tensor address
    return: None
    """

    score_threshold = param_list[0]
    num_actual = param_list[CONFIG_ONE]
    topk_k = param_list[CONFIG_TWO]
    data_type = "float16"

    src_tensor = data_gm[0]
    dst_tensor = data_gm[1]

    n_required = tik_instance.Scalar("uint32",
                                     name="n_required")
    tik_scalar_min(tik_instance, num_actual, topk_k, n_required)

    with tik_instance.new_stmt_scope():
        data_ub_a = tik_instance.Tensor(data_type,
                                        (CONFIG_TOPK2, CONFIG_EIGHT),
                                        name="data_ub_a",
                                        scope=tik.scope_ubuf)
        data_ub_b = tik_instance.Tensor(data_type,
                                        (CONFIG_TOPK2, CONFIG_EIGHT),
                                        name="data_ub_b",
                                        scope=tik.scope_ubuf)

        # move data from DDR to UB
        with tik_instance.if_scope(num_actual < CONFIG_TOPK2):
            rep_time = ceil_div((CONFIG_TOPK2 - num_actual), CONFIG_SIXTEEN)
            offset = CONFIG_TOPK2 - rep_time * CONFIG_SIXTEEN
            with tik_instance.if_scope(rep_time <= MAX_REP_TIME):
                tik_instance.vector_dup(CONFIG_MASK,
                                        data_ub_a[offset, 0],
                                        score_threshold,
                                        rep_time,
                                        CONFIG_ONE,
                                        CONFIG_EIGHT)
            with tik_instance.else_scope():
                tik_instance.vector_dup(CONFIG_MASK,
                                        data_ub_a[offset, 0],
                                        score_threshold,
                                        MAX_REP_TIME,
                                        CONFIG_ONE,
                                        CONFIG_EIGHT)
                rep_time = rep_time - MAX_REP_TIME
                offset = offset + MAX_REP_TIME * CONFIG_SIXTEEN
                tik_instance.vector_dup(CONFIG_MASK,
                                        data_ub_a[offset, 0],
                                        score_threshold,
                                        rep_time,
                                        CONFIG_ONE,
                                        CONFIG_EIGHT)

        tik_instance.data_move(data_ub_a,
                               src_tensor[core_offset[0], 0],
                               0,
                               CONFIG_ONE,
                               ceil_div(num_actual, CONFIG_TWO),
                               0, 0)
        # perform topk
        tik_topk_6114(tik_instance, (data_ub_a, data_ub_b), num_actual, score_threshold)

        # move data out to DDR
        tik_instance.data_move(dst_tensor[core_offset[CONFIG_ONE], 0],
                               data_ub_a,
                               0,
                               CONFIG_ONE,
                               ceil_div(n_required, CONFIG_TWO),
                               0, 0)


def tik_topk_vms4(tik_instance, dst_tensor, dest_pos, src_tensor, src_pos, count_list,
                  valid_bit, topk_k, score_threshold):
    """
    combine the sorted list into one list
    param tik_instance:

    param dst_tensor: the tensor to sort the results
        param dest_pos: the offset of the dst_tensor

    param src_tensor: the tensor of sorted lists
        param src_pos:the offset of each list, a list or tuple of four
        param count_list: the proposal num of each list, a list or tuple of four

    param valid_bit: same to the  vmrgsort4, [3, 7, 15]
    param topk_k:
    param score_threshold:
    return: None
    """

    max_iteration = ceil_div(topk_k, CONFIG_LEN)

    with tik_instance.new_stmt_scope():
        src_ub = tik_instance.Tensor("float16",
                                     (CONFIG_FOUR, CONFIG_LEN, CONFIG_EIGHT),
                                     name="src_ub",
                                     scope=tik.scope_ubuf)
        dst_ub = tik_instance.Tensor("float16",
                                     (CONFIG_TOPK2, CONFIG_EIGHT),
                                     name="dst_ub",
                                     scope=tik.scope_ubuf)
        dest_pos_ = tik_instance.Scalar("uint32", "dest_pos_", dest_pos)
        n_total_selected_ = tik_instance.Scalar("uint32", "n_total_selected_", 0)
        n_selected_ = tik_instance.Scalar("uint32", "n_selected_", 0)

        num_exhausted = [tik_instance.Scalar(dtype="uint32") for i in range(CONFIG_FOUR)]
        src_exhausted = [tik_instance.Scalar(dtype="uint32") for i in range(CONFIG_FOUR)]
        src_pos_ = [tik_instance.Scalar(dtype="uint32") for i in range(CONFIG_FOUR)]

        for i in range(CONFIG_FOUR):
            src_exhausted[i].set_as(0)
            num_exhausted[i].set_as(0)
            src_pos_[i].set_as(src_pos[i])

        # first time data in
        # four src
        if valid_bit == (CONFIG_SIXTEEN - CONFIG_ONE):
            for t_cyc in range(CONFIG_FOUR):
                with tik_instance.if_scope(src_exhausted[t_cyc] + CONFIG_LEN <= count_list[t_cyc]):
                    tik_instance.data_move(src_ub[t_cyc, 0, 0],
                                           src_tensor[t_cyc][src_pos_[t_cyc], 0],
                                           0,
                                           CONFIG_ONE,
                                           CONFIG_LEN // CONFIG_TWO,
                                           0, 0)
                with tik_instance.else_scope():
                    tik_instance.vector_dup(CONFIG_MASK,
                                            src_ub[t_cyc, 0, 0],
                                            score_threshold,
                                            CONFIG_LEN // CONFIG_SIXTEEN,
                                            CONFIG_ONE,
                                            CONFIG_EIGHT)
                    with tik_instance.if_scope(src_exhausted[t_cyc] < count_list[t_cyc]):
                        tik_instance.data_move(src_ub[t_cyc, 0, 0],
                                               src_tensor[t_cyc][src_pos_[t_cyc], 0],
                                               0,
                                               CONFIG_ONE,
                                               ceil_div(count_list[t_cyc] - src_exhausted[t_cyc],
                                                        CONFIG_TWO),
                                               0, 0)
        # three src
        elif valid_bit == (CONFIG_EIGHT - CONFIG_ONE):
            for t_cyc in range(CONFIG_THREE):
                with tik_instance.if_scope(src_exhausted[t_cyc] + CONFIG_LEN <= count_list[t_cyc]):
                    tik_instance.data_move(src_ub[t_cyc, 0, 0],
                                           src_tensor[t_cyc][src_pos_[t_cyc], 0],
                                           0,
                                           CONFIG_ONE,
                                           CONFIG_LEN // CONFIG_TWO,
                                           0, 0)
                with tik_instance.else_scope():
                    tik_instance.vector_dup(CONFIG_MASK,
                                            src_ub[t_cyc, 0, 0],
                                            score_threshold,
                                            CONFIG_LEN // CONFIG_SIXTEEN,
                                            CONFIG_ONE,
                                            CONFIG_EIGHT)
                    with tik_instance.if_scope(src_exhausted[t_cyc] < count_list[t_cyc]):
                        tik_instance.data_move(src_ub[t_cyc, 0, 0],
                                               src_tensor[t_cyc][src_pos_[t_cyc], 0],
                                               0,
                                               CONFIG_ONE,
                                               ceil_div(count_list[t_cyc] - src_exhausted[t_cyc],
                                                        CONFIG_TWO),
                                               0, 0)
        # two src
        elif valid_bit == CONFIG_THREE:
            for t_cyc in range(CONFIG_TWO):
                with tik_instance.if_scope(src_exhausted[t_cyc] + CONFIG_LEN <= count_list[t_cyc]):
                    tik_instance.data_move(src_ub[t_cyc, 0, 0],
                                           src_tensor[t_cyc][src_pos_[t_cyc], 0],
                                           0,
                                           CONFIG_ONE,
                                           CONFIG_LEN // CONFIG_TWO,
                                           0, 0)
                with tik_instance.else_scope():
                    tik_instance.vector_dup(CONFIG_MASK,
                                            src_ub[t_cyc, 0, 0],
                                            score_threshold,
                                            CONFIG_LEN // CONFIG_SIXTEEN,
                                            CONFIG_ONE,
                                            CONFIG_EIGHT)
                    with tik_instance.if_scope(src_exhausted[t_cyc] < count_list[t_cyc]):
                        tik_instance.data_move(src_ub[t_cyc, 0, 0],
                                               src_tensor[t_cyc][src_pos_[t_cyc], 0],
                                               0,
                                               CONFIG_ONE,
                                               ceil_div(count_list[t_cyc] - src_exhausted[t_cyc],
                                                        CONFIG_TWO),
                                               0, 0)
        # not support
        else:
            raise RuntimeError("valid_bit only support 3, 7, 15!")

        with tik_instance.for_range(0, max_iteration):
            with tik_instance.if_scope(n_total_selected_ < topk_k):
                # Step-1: move data from DDR to UB
                # four src
                if valid_bit == (CONFIG_SIXTEEN - CONFIG_ONE):
                    for t_cyc in range(CONFIG_FOUR):
                        with tik_instance.if_scope(num_exhausted[t_cyc] > 0):
                            src_exhausted[t_cyc].set_as(src_exhausted[t_cyc] + num_exhausted[t_cyc])
                            src_pos_[t_cyc].set_as(src_pos_[t_cyc] + num_exhausted[t_cyc])
                            with tik_instance.if_scope(src_exhausted[t_cyc] + CONFIG_LEN <=
                                                       count_list[t_cyc]):
                                tik_instance.data_move(src_ub[t_cyc, 0, 0],
                                                       src_tensor[t_cyc][src_pos_[t_cyc], 0],
                                                       0,
                                                       CONFIG_ONE,
                                                       CONFIG_LEN // CONFIG_TWO,
                                                       0, 0)
                            with tik_instance.else_scope():
                                tik_instance.vector_dup(CONFIG_MASK,
                                                        src_ub[t_cyc, 0, 0],
                                                        score_threshold,
                                                        CONFIG_LEN // CONFIG_SIXTEEN,
                                                        CONFIG_ONE,
                                                        CONFIG_EIGHT)
                                with tik_instance.if_scope(src_exhausted[t_cyc] <
                                                           count_list[t_cyc]):
                                    tik_instance.data_move(src_ub[t_cyc, 0, 0],
                                                           src_tensor[t_cyc][src_pos_[t_cyc], 0],
                                                           0,
                                                           CONFIG_ONE,
                                                           ceil_div(count_list[t_cyc] -
                                                                    src_exhausted[t_cyc],
                                                                    CONFIG_TWO),
                                                           0, 0)

                # three src
                elif valid_bit == (CONFIG_EIGHT - CONFIG_ONE):
                    for t_cyc in range(CONFIG_THREE):
                        with tik_instance.if_scope(num_exhausted[t_cyc] > 0):
                            src_exhausted[t_cyc].set_as(src_exhausted[t_cyc] + num_exhausted[t_cyc])
                            src_pos_[t_cyc].set_as(src_pos_[t_cyc] + num_exhausted[t_cyc])
                            with tik_instance.if_scope(src_exhausted[t_cyc] + CONFIG_LEN <=
                                                       count_list[t_cyc]):
                                tik_instance.data_move(src_ub[t_cyc, 0, 0],
                                                       src_tensor[t_cyc][src_pos_[t_cyc], 0],
                                                       0,
                                                       CONFIG_ONE,
                                                       CONFIG_LEN // CONFIG_TWO,
                                                       0, 0)
                            with tik_instance.else_scope():
                                tik_instance.vector_dup(CONFIG_MASK,
                                                        src_ub[t_cyc, 0, 0],
                                                        score_threshold,
                                                        CONFIG_LEN // CONFIG_SIXTEEN,
                                                        CONFIG_ONE,
                                                        CONFIG_EIGHT)
                                with tik_instance.if_scope(src_exhausted[t_cyc] <
                                                           count_list[t_cyc]):
                                    tik_instance.data_move(src_ub[t_cyc, 0, 0],
                                                           src_tensor[t_cyc][src_pos_[t_cyc], 0],
                                                           0,
                                                           CONFIG_ONE,
                                                           ceil_div(count_list[t_cyc] -
                                                                    src_exhausted[t_cyc],
                                                                    CONFIG_TWO),
                                                           0, 0)

                # two src
                elif valid_bit == CONFIG_THREE:
                    for t_cyc in range(CONFIG_TWO):
                        with tik_instance.if_scope(num_exhausted[t_cyc] > 0):
                            src_exhausted[t_cyc].set_as(src_exhausted[t_cyc] + num_exhausted[t_cyc])
                            src_pos_[t_cyc].set_as(src_pos_[t_cyc] + num_exhausted[t_cyc])
                            with tik_instance.if_scope(src_exhausted[t_cyc] + CONFIG_LEN <=
                                                       count_list[t_cyc]):
                                tik_instance.data_move(src_ub[t_cyc, 0, 0],
                                                       src_tensor[t_cyc][src_pos_[t_cyc], 0],
                                                       0,
                                                       CONFIG_ONE,
                                                       CONFIG_LEN // CONFIG_TWO,
                                                       0, 0)
                            with tik_instance.else_scope():
                                tik_instance.vector_dup(CONFIG_MASK,
                                                        src_ub[t_cyc, 0, 0],
                                                        score_threshold,
                                                        CONFIG_LEN // CONFIG_SIXTEEN,
                                                        CONFIG_ONE,
                                                        CONFIG_EIGHT)
                                with tik_instance.if_scope(src_exhausted[t_cyc] <
                                                           count_list[t_cyc]):
                                    tik_instance.data_move(src_ub[t_cyc, 0, 0],
                                                           src_tensor[t_cyc][src_pos_[t_cyc], 0],
                                                           0,
                                                           CONFIG_ONE,
                                                           ceil_div(count_list[t_cyc] -
                                                                    src_exhausted[t_cyc],
                                                                    CONFIG_TWO),
                                                           0, 0)

                # Step-2: Perform the sort with exhausted suspend mode enabled
                tik_instance.vmrgsort4(dst_ub,
                                       (src_ub[0, 0, 0], src_ub[CONFIG_ONE, 0, 0],
                                        src_ub[CONFIG_TWO, 0, 0], src_ub[CONFIG_THREE, 0, 0]),
                                       (CONFIG_LEN, CONFIG_LEN, CONFIG_LEN, CONFIG_LEN),
                                       True,
                                       valid_bit,
                                       CONFIG_ONE,
                                       num_exhausted)

                # Step-3: Move result from UB to OUT
                n_selected_.set_as(0)
                for i in range(CONFIG_FOUR):
                    n_selected_.set_as(n_selected_ + num_exhausted[i])

                with tik_instance.if_scope(n_total_selected_ + n_selected_ <= topk_k):
                    tik_instance.data_move(dst_tensor[dest_pos_, 0],
                                           dst_ub,
                                           0,
                                           CONFIG_ONE,
                                           ceil_div(n_selected_, CONFIG_TWO),
                                           0, 0)
                    dest_pos_.set_as(dest_pos_ + n_selected_)

                with tik_instance.else_scope():
                    tik_instance.data_move(dst_tensor[dest_pos_, 0],
                                           dst_ub,
                                           0,
                                           CONFIG_ONE,
                                           ceil_div(topk_k - n_total_selected_, CONFIG_TWO),
                                           0, 0)

                n_total_selected_.set_as(n_total_selected_ + n_selected_)


def tik_topk_external_sort(tik_instance, data_gm, input_param, core_offset):
    """
    the proposals can not be moved in at one time
    param tik_instance:
    param data_gm:
        0  src_tensor
        1  mem_swap
        2  mem_swap_2
        3  dst_tensor
    input_param
        0  score_threshold:
        1  num_actual:  actual num
        2  topk_k:
    core_offset : the offset of the input tensor address
    return: None
    """

    score_threshold = input_param[0]
    num_actual = input_param[CONFIG_ONE]
    topk_k = input_param[CONFIG_TWO]
    data_type = "float16"

    src_tensor = data_gm[0]
    mem_swap = data_gm[CONFIG_ONE]
    mem_swap_2 = data_gm[CONFIG_TWO]
    dst_tensor = data_gm[CONFIG_THREE]

    n_required = min(topk_k, CONFIG_TOPK2)

    num_cycle = num_actual // CONFIG_TOPK2
    num_left = num_actual % CONFIG_TOPK2

    # sort each 6144 proposals
    with tik_instance.new_stmt_scope():
        data_ub_a = tik_instance.Tensor(data_type,
                                        (CONFIG_TOPK2, CONFIG_EIGHT),
                                        name="data_ub_a",
                                        scope=tik.scope_ubuf)
        data_ub_b = tik_instance.Tensor(data_type,
                                        (CONFIG_TOPK2, CONFIG_EIGHT),
                                        name="data_ub_b",
                                        scope=tik.scope_ubuf)
        with tik_instance.for_range(0, num_cycle) as t_cyc:
            offset = core_offset[0] + CONFIG_TOPK2 * t_cyc
            # move data from DDR to UB
            tik_instance.data_move(data_ub_a,
                                   src_tensor[offset, 0],
                                   0,
                                   CONFIG_ONE,
                                   ceil_div(CONFIG_TOPK2, CONFIG_TWO),
                                   0, 0)
            # perform topk
            tik_topk_6114(tik_instance, (data_ub_a, data_ub_b), CONFIG_TOPK2, score_threshold)

            offset = core_offset[CONFIG_ONE] + CONFIG_TOPK2 * t_cyc
            # move data out to DDR
            tik_instance.data_move(mem_swap[offset, 0],
                                   data_ub_a,
                                   0,
                                   CONFIG_ONE,
                                   ceil_div(CONFIG_TOPK2, CONFIG_TWO),
                                   0, 0)

        # the tail
        with tik_instance.if_scope(num_left > 0):
            # move data from DDR to UB
            offset = core_offset[0] + CONFIG_TOPK2 * num_cycle
            rep_time = ceil_div((CONFIG_TOPK2 - num_left), CONFIG_SIXTEEN)
            offset_ub = CONFIG_TOPK2 - rep_time * CONFIG_SIXTEEN
            with tik_instance.if_scope(rep_time <= MAX_REP_TIME):
                tik_instance.vector_dup(CONFIG_MASK,
                                        data_ub_a[offset_ub, 0],
                                        score_threshold,
                                        rep_time,
                                        CONFIG_ONE,
                                        CONFIG_EIGHT)
            with tik_instance.else_scope():
                with tik_instance.for_range(0, rep_time // MAX_REP_TIME) as t_cyc:
                    offset_ub = offset_ub + MAX_REP_TIME * CONFIG_SIXTEEN * t_cyc
                    tik_instance.vector_dup(CONFIG_MASK,
                                            data_ub_a[offset_ub, 0],
                                            score_threshold,
                                            MAX_REP_TIME,
                                            CONFIG_ONE,
                                            CONFIG_EIGHT)

                rep_time = rep_time % MAX_REP_TIME
                offset_ub = CONFIG_TOPK2 - rep_time * CONFIG_SIXTEEN
                tik_instance.vector_dup(CONFIG_MASK,
                                        data_ub_a[offset_ub, 0],
                                        score_threshold,
                                        rep_time,
                                        CONFIG_ONE,
                                        CONFIG_EIGHT)

            tik_instance.data_move(data_ub_a,
                                   src_tensor[offset, 0],
                                   0,
                                   CONFIG_ONE,
                                   ceil_div(num_left, CONFIG_TWO),
                                   0, 0)

            # perform topk
            tik_topk_6114(tik_instance, (data_ub_a, data_ub_b), num_left, score_threshold)

            # move data out to DDR
            offset = core_offset[CONFIG_ONE] + CONFIG_TOPK2 * num_cycle
            tik_instance.data_move(mem_swap[offset, 0],
                                   data_ub_a,
                                   0,
                                   CONFIG_ONE,
                                   ceil_div(num_left, CONFIG_TWO),
                                   0, 0)

    src_pos_ = [tik_instance.Scalar(dtype="uint32") for i in range(CONFIG_FOUR)]
    for t_cyc in range(CONFIG_FOUR):
        src_pos_[t_cyc].set_as(core_offset[CONFIG_ONE] + CONFIG_TOPK2 * t_cyc)

    # combine the sorted lists
    with tik_instance.if_scope(num_cycle < CONFIG_FOUR):
        with tik_instance.new_stmt_scope():
            with tik_instance.if_scope(num_cycle == CONFIG_THREE):
                tik_topk_vms4(tik_instance,
                              dst_tensor,
                              core_offset[CONFIG_THREE],
                              (mem_swap, mem_swap, mem_swap, mem_swap),
                              (src_pos_[0], src_pos_[CONFIG_ONE],
                               src_pos_[CONFIG_TWO], src_pos_[CONFIG_THREE]),
                              (CONFIG_TOPK2, CONFIG_TOPK2, CONFIG_TOPK2, num_left),
                              CONFIG_SIXTEEN - CONFIG_ONE,
                              topk_k,
                              score_threshold)

            with tik_instance.if_scope(num_cycle == CONFIG_TWO):
                src_pos_[CONFIG_TWO].set_as(CONFIG_TOPK2 * num_cycle)
                tik_topk_vms4(tik_instance,
                              dst_tensor,
                              core_offset[CONFIG_THREE],
                              (mem_swap, mem_swap, mem_swap, None),
                              (src_pos_[0], src_pos_[CONFIG_ONE],
                               src_pos_[CONFIG_TWO], src_pos_[CONFIG_TWO]),
                              (CONFIG_TOPK2, CONFIG_TOPK2, num_left, 0),
                              CONFIG_EIGHT - CONFIG_ONE,
                              topk_k,
                              score_threshold)

            with tik_instance.if_scope(num_cycle == CONFIG_ONE):
                src_pos_[CONFIG_ONE].set_as(CONFIG_TOPK2 * num_cycle)
                tik_topk_vms4(tik_instance,
                              dst_tensor,
                              core_offset[CONFIG_THREE],
                              (mem_swap, mem_swap, None, None),
                              (src_pos_[0], src_pos_[CONFIG_ONE],
                               src_pos_[CONFIG_ONE], src_pos_[CONFIG_ONE]),
                              (CONFIG_TOPK2, num_left, 0, 0),
                              CONFIG_THREE,
                              topk_k,
                              score_threshold)

    with tik_instance.else_scope():
        # first combine four list
        with tik_instance.new_stmt_scope():
            tik_topk_vms4(tik_instance,
                          mem_swap_2,
                          core_offset[CONFIG_TWO],
                          (mem_swap, mem_swap, mem_swap, mem_swap),
                          (src_pos_[0], src_pos_[CONFIG_ONE],
                           src_pos_[CONFIG_TWO], src_pos_[CONFIG_THREE]),
                          (n_required, n_required, n_required, n_required),
                          CONFIG_SIXTEEN - CONFIG_ONE,
                          topk_k,
                          score_threshold)
        # whether rep is needed
        max_rep = ceil_div(num_cycle, CONFIG_THREE) - CONFIG_TWO
        left_rep = num_cycle % CONFIG_THREE
        # need repeat
        with tik_instance.if_scope(max_rep > 0):
            with tik_instance.for_range(0, max_rep) as t_cyc:
                src_pos_[CONFIG_ONE].set_as(src_pos_[CONFIG_ONE] + CONFIG_TOPK2 * CONFIG_THREE)
                src_pos_[CONFIG_TWO].set_as(src_pos_[CONFIG_TWO] + CONFIG_TOPK2 * CONFIG_THREE)
                src_pos_[CONFIG_THREE].set_as(src_pos_[CONFIG_THREE] + CONFIG_TOPK2 * CONFIG_THREE)

                dst_offset1 = (t_cyc % CONFIG_TWO) * CONFIG_TOPK2 + core_offset[CONFIG_TWO]
                dst_offset2 = ((t_cyc + CONFIG_ONE) % CONFIG_TWO) * CONFIG_TOPK2 +\
                              core_offset[CONFIG_TWO]
                with tik_instance.new_stmt_scope():
                    tik_topk_vms4(tik_instance,
                                  mem_swap_2,
                                  dst_offset2,
                                  (mem_swap_2, mem_swap, mem_swap, mem_swap),
                                  (dst_offset1, src_pos_[CONFIG_ONE],
                                   src_pos_[CONFIG_TWO], src_pos_[CONFIG_THREE]),
                                  (topk_k, topk_k, topk_k, topk_k),
                                  CONFIG_SIXTEEN - CONFIG_ONE,
                                  topk_k,
                                  score_threshold)

        dst_offset1 = CONFIG_TOPK2 * (max_rep % CONFIG_TWO) + core_offset[CONFIG_TWO]
        src_pos_[CONFIG_ONE].set_as(src_pos_[CONFIG_ONE] + CONFIG_TOPK2 * CONFIG_THREE)
        src_pos_[CONFIG_TWO].set_as(src_pos_[CONFIG_TWO] + CONFIG_TOPK2 * CONFIG_THREE)
        src_pos_[CONFIG_THREE].set_as(src_pos_[CONFIG_THREE] + CONFIG_TOPK2 * CONFIG_THREE)
        with tik_instance.if_scope(left_rep == 0):
            with tik_instance.new_stmt_scope():
                tik_topk_vms4(tik_instance,
                              dst_tensor,
                              core_offset[CONFIG_THREE],
                              (mem_swap_2, mem_swap, mem_swap, mem_swap),
                              (dst_offset1, src_pos_[CONFIG_ONE],
                               src_pos_[CONFIG_TWO], src_pos_[CONFIG_THREE]),
                              (topk_k, topk_k, topk_k, topk_k),
                              CONFIG_SIXTEEN - CONFIG_ONE,
                              topk_k,
                              score_threshold)

        with tik_instance.if_scope(left_rep == CONFIG_TWO):
            with tik_instance.new_stmt_scope():
                tik_topk_vms4(tik_instance,
                              dst_tensor,
                              core_offset[CONFIG_THREE],
                              (mem_swap_2, mem_swap, mem_swap, None),
                              (dst_offset1,
                               src_pos_[CONFIG_ONE],
                               src_pos_[CONFIG_TWO],
                               src_pos_[CONFIG_THREE]),
                              (topk_k, topk_k, topk_k, 0),
                              CONFIG_EIGHT - CONFIG_ONE,
                              topk_k,
                              score_threshold)

        with tik_instance.if_scope(left_rep == CONFIG_ONE):
            with tik_instance.new_stmt_scope():
                tik_topk_vms4(tik_instance,
                              dst_tensor,
                              core_offset[CONFIG_THREE],
                              (mem_swap_2, mem_swap, None, None),
                              (dst_offset1,
                               src_pos_[CONFIG_ONE],
                               src_pos_[CONFIG_TWO],
                               src_pos_[CONFIG_THREE]),
                              (topk_k, topk_k, 0, 0),
                              CONFIG_THREE,
                              topk_k,
                              score_threshold)


def tik_topk(tik_instance, data_tensor, input_param, core_offset):
    """
    do the topk
    param tik_instance:
    param data_tensor: a list
        data_in = data_tensor[0]  : proposals to be sorted
        data_out = data_tensor[1]  : the tensor to store the results
        mem_swap = data_tensor[2]  :  the middle tensor to store the sorted list 6144,
                                     the size of this tensor should be same as the data_in
        mem_swap_2 = data_tensor[3] : the middle tensor for list combining, as least [k*2, 8]

    input_param
         0  proposal_num: actual num of proposals
         1  k:  max num of proposals output
         2  score_threshold: used for dump


    core_offset: the offset of the input tensor address

    return: None
    """
    data_in = data_tensor[0]
    data_out = data_tensor[CONFIG_ONE]

    proposal_num = input_param[0]
    k = input_param[CONFIG_ONE]
    score_threshold = input_param[CONFIG_TWO]

    with tik_instance.if_scope(proposal_num <= CONFIG_TOPK2):
        with tik_instance.new_stmt_scope():
            tik_topk_internal_sort(tik_instance,
                                   (data_in, data_out),
                                   (score_threshold, proposal_num, k),
                                   (core_offset[0], core_offset[CONFIG_ONE]))

    with tik_instance.else_scope():
        mem_swap = data_tensor[CONFIG_TWO]
        mem_swap_2 = data_tensor[CONFIG_THREE]
        tik_topk_external_sort(tik_instance,
                               (data_in, mem_swap, mem_swap_2, data_out),
                               (score_threshold, proposal_num, k),
                               (core_offset[0], core_offset[CONFIG_TWO],
                                core_offset[CONFIG_THREE], core_offset[CONFIG_ONE]))


def call_topk_sort(tik_instance, input_tensor, proposal_num, input_param, core_offset):
    """
    call the topk function to perform the topk
    param tik_instance:
    param input_tensor: data_gm
           0 proposal_post_topk : the tensor to store the results
           1 proposal_cb:  proposals in L1, num_total_cb
           2 mem_swap:  the cache to storing the proposals in L1
           3 proposal_gm:   proposals in DDR, num_total_gm
           4 mem_swap_gm:  the tensor to cache the proposals results of L1

    param proposal_num:  data_tensor
          0 flag_cb: whether the DDR tensor is used for the proposals storing
                       0 the DDR is used; otherwise, not
          1 num_total_cb
          2 num_total_gm

    param input_param:
          0  k:
          1  score_threshold:
          2  data_type
          3  score_filter
    core_offset: the offset of the tensor address
    return:  None
    """

    score_flag = input_param[CONFIG_THREE]
    if score_flag:
        # proposals stored in L1
        # L1 Buffer, the results stored in the dst_tensor
        with tik_instance.if_scope(proposal_num[CONFIG_ONE] > 0):
            tik_topk(tik_instance,
                     (input_tensor[CONFIG_ONE], input_tensor[0],
                      input_tensor[CONFIG_ONE], input_tensor[CONFIG_TWO]),
                     (proposal_num[CONFIG_ONE],
                      input_param[0],
                      input_param[CONFIG_ONE]),
                     (core_offset[CONFIG_ONE], core_offset[0],
                      core_offset[CONFIG_ONE], core_offset[CONFIG_TWO]))

        # not only in L1 buffer
        with tik_instance.if_scope(tik.all(proposal_num[0] == 0, proposal_num[CONFIG_TWO] > 0)):
            # both L1 buffer and DDR
            with tik_instance.if_scope(proposal_num[CONFIG_TWO] <= L1_MAX_NUM):
                # useing the L1 Buffer as swap_tensor, the results stored in the DDR
                tik_topk(tik_instance,
                         (input_tensor[CONFIG_THREE], input_tensor[CONFIG_FOUR],
                          input_tensor[CONFIG_ONE], input_tensor[CONFIG_TWO]),
                         (proposal_num[CONFIG_TWO],
                          input_param[0],
                          input_param[CONFIG_ONE]),
                         (core_offset[CONFIG_THREE], core_offset[CONFIG_FOUR],
                          core_offset[CONFIG_ONE], core_offset[CONFIG_TWO]))
            with tik_instance.else_scope():
                tik_topk(tik_instance,
                         (input_tensor[CONFIG_THREE], input_tensor[CONFIG_FOUR],
                          input_tensor[CONFIG_THREE], input_tensor[CONFIG_TWO]),
                         (proposal_num[CONFIG_TWO],
                          input_param[0],
                          input_param[CONFIG_ONE]),
                         (core_offset[CONFIG_THREE], core_offset[CONFIG_FOUR],
                          core_offset[CONFIG_THREE], core_offset[CONFIG_TWO]))

            # move the results to the mem_swap
            num_in_gm = tik_instance.Scalar("uint32", name="num_in_gm")
            tik_scalar_min(tik_instance,
                           input_param[0],
                           proposal_num[CONFIG_TWO],
                           num_in_gm)
            with tik_instance.new_stmt_scope():
                temp_ub = tik_instance.Tensor(input_param[CONFIG_TWO],
                                              (input_param[0] + CONFIG_FOUR, CONFIG_EIGHT),
                                              name="temp_ub",
                                              scope=tik.scope_ubuf)
                # L1 buffer results, storing in dst_tensor
                tik_instance.data_move(temp_ub,
                                       input_tensor[0][core_offset[0], 0],
                                       0,
                                       CONFIG_ONE,
                                       ceil_div(input_param[0], CONFIG_TWO),
                                       0, 0)
                tik_instance.data_move(input_tensor[CONFIG_TWO][core_offset[CONFIG_TWO], 0],
                                       temp_ub,
                                       0,
                                       CONFIG_ONE,
                                       ceil_div(input_param[0], CONFIG_TWO),
                                       0, 0)
                # DDR buffer results, storing in input_tensor[CONFIG_FOUR]-->mem_swap_gm
                tik_instance.data_move(temp_ub,
                                       input_tensor[CONFIG_FOUR][core_offset[CONFIG_FOUR], 0],
                                       0,
                                       CONFIG_ONE,
                                       ceil_div(num_in_gm, CONFIG_TWO),
                                       0, 0)
                tik_instance.data_move(input_tensor[CONFIG_TWO][core_offset[CONFIG_TWO] +
                                                                CONFIG_TOPK2, 0],
                                       temp_ub,
                                       0,
                                       CONFIG_ONE,
                                       ceil_div(num_in_gm, CONFIG_TWO),
                                       0, 0)

            tik_topk_vms4(tik_instance,
                          input_tensor[0],
                          core_offset[0],
                          (input_tensor[CONFIG_TWO], input_tensor[CONFIG_TWO],
                           None, None),
                          (core_offset[CONFIG_TWO], core_offset[CONFIG_TWO] + CONFIG_TOPK2, 0, 0),
                          (input_param[0], num_in_gm, 0, 0),
                          CONFIG_THREE,
                          input_param[0],
                          input_param[CONFIG_ONE])

    else:
        with tik_instance.if_scope(proposal_num[CONFIG_TWO] < L1_MAX_NUM):
            # useing the L1 Buffer as swap_tensor, the results stored in the DDR
            tik_topk(tik_instance,
                     (input_tensor[CONFIG_THREE], input_tensor[0],
                      input_tensor[CONFIG_ONE], input_tensor[CONFIG_TWO]),
                     (proposal_num[CONFIG_TWO],
                      input_param[0],
                      input_param[CONFIG_ONE]),
                     (core_offset[CONFIG_THREE], core_offset[0],
                      core_offset[CONFIG_ONE], core_offset[CONFIG_TWO]))
        with tik_instance.else_scope():
            tik_topk(tik_instance,
                     (input_tensor[CONFIG_THREE], input_tensor[0],
                      input_tensor[CONFIG_THREE], input_tensor[CONFIG_TWO]),
                     (proposal_num[CONFIG_TWO],
                      input_param[0],
                      input_param[CONFIG_ONE]),
                     (core_offset[CONFIG_THREE], core_offset[0],
                      core_offset[CONFIG_THREE], core_offset[CONFIG_TWO]))


def check_input_param(input_para, kernel_name):
    """
    check other input of generate_rpn_proposals()
    0  score_threshold (-inf, inf)
    1  k: [0, 6000]
    2  core_max_num
    3  aicore_num  available aircore num
    return: None
    """

    k = input_para[CONFIG_ONE]
    core_max_num = input_para[CONFIG_TWO]
    aicore_num = input_para[CONFIG_THREE]

    def _check_range_of_input(input_x, min_limit, max_limit, input_name):
        """
        internal function
        check whether min_limit<=input_para<=max_limit
        """
        if input_x < min_limit or input_x > max_limit:
            raise RuntimeError("The %s should be in [%d, %d]!" % (input_name, min_limit, max_limit))

    _check_range_of_input(core_max_num, CONFIG_ONE, CONFIG_DATA_ALIGN, "min_size")
    _check_range_of_input(k, 0, MAX_TOPK, "k")

    if k % CONFIG_SIXTEEN:
        raise RuntimeError("K should be times of 16!")

    util.check_kernel_name(kernel_name)


def check_input_dict(dict_list, param_list):
    """
    check the input dict of score_filter_pre_sort()
    Parameters
    ----------
    dict_list: a list of input dict
         0 rois : dict
            shape and dtype of input boxes
         1 cls_bg_prob : dict
            shape and dtype of input probobilities
         2 sorted_proposal: dict
            shape and dtype of output sorted proposal
         3 proposal_num: actual num of proposal
            num of proposals after scorefilter
    param_list: a list of param
         0 score_threshold
         1 k
         2 core_max_num
    Returns
    -------
    None
    """

    def _check_input_type_dict(input_dict, input_key, input_name):
        """
        internal function, check the key of dict
        whether the necessary key is included
        """
        for key in input_key:
            if key not in input_dict.keys():
                raise RuntimeError(
                    "the input parameter %s must have arrt <%s>" %
                    (input_name, key))

    def _check_shape_size_limit(input_shape, input_name, shape_para=4, output_flag=True):
        """
        internal function, check the size of the shape
        two different modes
        """
        n_x = CONFIG_ONE
        if len(input_shape) > CONFIG_ONE:
            if input_shape[-1] != shape_para:
                raise RuntimeError("The last dimension of %s should be %d!" %
                                   (input_name, shape_para))

            for loop_i in range(len(input_shape) - CONFIG_ONE):
                n_x = n_x * input_shape[loop_i]
        else:
            n_x = input_shape[0]

        if output_flag:
            if n_x % CONFIG_SIXTEEN != 0:
                raise RuntimeError("N of input %s should be times of 16!" % input_name)

    # check whether both "shape" and "dtype" is included and following the rule
    input_key = ("shape", "dtype")
    _check_input_type_dict(dict_list[0], input_key, "rois")
    _check_input_type_dict(dict_list[CONFIG_ONE], input_key, "cls_bg_prob")
    _check_input_type_dict(dict_list[CONFIG_TWO], input_key, "sorted_proposal")
    _check_input_type_dict(dict_list[CONFIG_THREE], input_key, "proposal_num")

    # check the dtype
    util.check_dtype_rule(dict_list[0].get("dtype"), ("float16", ))
    util.check_dtype_rule(dict_list[CONFIG_ONE].get("dtype"), ("float16", ))
    util.check_dtype_rule(dict_list[CONFIG_TWO].get("dtype"), ("float16", ))
    util.check_dtype_rule(dict_list[CONFIG_THREE].get("dtype"), ("uint32", ))

    # get the parameters from dicts
    input_rois_shape = dict_list[0].get("shape")
    input_prob_shape = dict_list[CONFIG_ONE].get("shape")
    output_proposal_shape = dict_list[CONFIG_TWO].get("shape")
    output_proposal_num_shape = dict_list[CONFIG_THREE].get("shape")
    # check the shape
    util.check_shape_rule(input_rois_shape,
                          min_dim=CONFIG_TWO,
                          max_dim=CONFIG_TWO,
                          max_shape_num=SHAPE_SIZE_LIMIT)
    util.check_shape_rule(input_prob_shape,
                          min_dim=CONFIG_ONE,
                          max_dim=CONFIG_TWO,
                          max_shape_num=SHAPE_SIZE_LIMIT)
    util.check_shape_rule(output_proposal_shape,
                          min_dim=CONFIG_TWO,
                          max_dim=CONFIG_TWO)
    util.check_shape_rule(output_proposal_num_shape,
                          min_dim=CONFIG_TWO,
                          max_dim=CONFIG_TWO)

    # Check the size of the input/output shape
    _check_shape_size_limit(input_rois_shape,
                            "rois",
                            shape_para=CONFIG_FOUR,
                            output_flag=True)
    _check_shape_size_limit(input_prob_shape,
                            "cls_bg_prob",
                            shape_para=CONFIG_ONE,
                            output_flag=True)
    _check_shape_size_limit(output_proposal_shape,
                            "sorted_propsoal",
                            shape_para=CONFIG_EIGHT,
                            output_flag=True)
    _check_shape_size_limit(output_proposal_num_shape,
                            "propsoal_num",
                            shape_para=CONFIG_EIGHT,
                            output_flag=False)

    if input_rois_shape[0] != input_prob_shape[0]:
        raise RuntimeError("n dimension of inputs rois and cls_bg_prob should be consistent")

    if output_proposal_num_shape[0] != param_list[CONFIG_TWO]:
        raise RuntimeError("n dimension of outputs proposal_num should"
                           " be consistent with max_core_num")

    if output_proposal_shape[0] != (param_list[CONFIG_ONE] + CONFIG_TWO) * param_list[CONFIG_TWO]:
        raise RuntimeError("sorted_proposal shape should be consistent with"
                           " k and core_max_num!")


def score_filter_pre_sort_compute(dict_list, param_list, kernel_name):
    """
    check the input dict of score_filter_pre_sort()
    Parameters
    ----------
    dict_list: a list of input dict
         0 rois : dict
            shape and dtype of input boxes
         1 cls_bg_prob : dict
            shape and dtype of input probobilities
         2 sorted_proposal: dict
            shape and dtype of output sorted proposal
         3 proposal_num: actual num of proposal
            num of proposals after scorefilter
    param_list: a list of param
         0 score_threshold
         1 k
         2 score_filter
         3 core_max_num
         4 aiccore_num
         5 dtype

    kernel_name:

    Returns
    -------
    None
    """

    # calculate the multi-core param
    num_tot = dict_list[0].get("shape")[0]
    aicore_num = param_list[CONFIG_FOUR]

    # num of box to be processed in one core
    num_one_core = ceil_div(num_tot, CONFIG_SIXTEEN * aicore_num) * CONFIG_SIXTEEN
    # the tail
    num_tail = num_tot - num_one_core * (aicore_num - CONFIG_ONE)

    tik_instance = tik.Tik(tik.Dprofile(), disable_debug=True)
    data_gm = InitGmTensor(tik_instance,
                           (num_tot, num_one_core, param_list[CONFIG_ONE],
                            param_list[CONFIG_THREE], param_list[CONFIG_FIVE]))
    # no tail
    if num_tot % num_one_core == 0:
        with tik_instance.for_range(0, aicore_num, block_num=aicore_num) as core_index:
            data_cb = InitGlobalTensor(tik_instance,
                                       (param_list[CONFIG_ONE], param_list[CONFIG_FIVE]))

            # the src address deviation of each core
            core_offset = num_one_core * core_index
            # the dst address deviation of each core
            core_offset0 = (param_list[CONFIG_ONE] + CONFIG_TWO) * core_index

            # score_filter
            score_filter_processing(tik_instance, data_gm, data_cb,
                                    (param_list[0], core_offset, num_one_core,
                                     param_list[CONFIG_TWO], param_list[CONFIG_FIVE]))
            # topk
            call_topk_sort(tik_instance,
                           (data_gm.out_gm, data_cb.proposal_cb, data_cb.mem_swap,
                            data_gm.proposal_gm, data_cb.mem_swap2),
                           (data_cb.flag_cb, data_cb.num_total_cb, data_cb.num_total_gm),
                           (param_list[CONFIG_ONE], param_list[0],
                            param_list[CONFIG_FIVE], param_list[CONFIG_TWO]),
                           (core_offset0, 0, 0, core_offset, 0))
            num_tensor = tik_instance.Tensor("uint32",
                                             (CONFIG_ONE, CONFIG_EIGHT),
                                             name="num_tensor",
                                             scope=tik.scope_ubuf)

            tik_scalar_min(tik_instance, param_list[CONFIG_ONE],
                           data_cb.num_post_score, data_cb.num_post_score)
            tik_instance.vector_dup(CONFIG_EIGHT,
                                    num_tensor,
                                    data_cb.num_post_score,
                                    CONFIG_ONE,
                                    CONFIG_ONE,
                                    CONFIG_EIGHT)
            tik_instance.data_move(data_gm.proposal_num[core_index, 0],
                                   num_tensor,
                                   0,
                                   CONFIG_ONE,
                                   CONFIG_ONE,
                                   0, 0)
    else:
        with tik_instance.for_range(0, aicore_num, block_num=aicore_num) as core_index:
            data_cb = InitGlobalTensor(tik_instance,
                                       (param_list[CONFIG_ONE], param_list[CONFIG_FIVE]))
            core_offset = num_one_core * core_index
            core_offset0 = (param_list[CONFIG_ONE] + CONFIG_TWO) * core_index
            with tik_instance.if_scope(core_index < aicore_num - CONFIG_ONE):
                # score_filter
                score_filter_processing(tik_instance, data_gm, data_cb,
                                        (param_list[0], core_offset, num_one_core,
                                         param_list[CONFIG_TWO], param_list[CONFIG_FIVE]))
                # topk
                call_topk_sort(tik_instance,
                               (data_gm.out_gm, data_cb.proposal_cb, data_cb.mem_swap,
                                data_gm.proposal_gm, data_cb.mem_swap2),
                               (data_cb.flag_cb, data_cb.num_total_cb, data_cb.num_total_gm),
                               (param_list[CONFIG_ONE], param_list[0],
                                param_list[CONFIG_FIVE], param_list[CONFIG_TWO]),
                               (core_offset0, 0, 0, core_offset, 0))
                num_tensor = tik_instance.Tensor("uint32",
                                                 (CONFIG_ONE, CONFIG_EIGHT),
                                                 name="num_tensor",
                                                 scope=tik.scope_ubuf)

                tik_scalar_min(tik_instance, param_list[CONFIG_ONE],
                               data_cb.num_post_score, data_cb.num_post_score)
                tik_instance.vector_dup(CONFIG_EIGHT,
                                        num_tensor,
                                        data_cb.num_post_score,
                                        CONFIG_ONE,
                                        CONFIG_ONE,
                                        CONFIG_EIGHT)
                tik_instance.data_move(data_gm.proposal_num[core_index, 0],
                                       num_tensor,
                                       0,
                                       CONFIG_ONE,
                                       CONFIG_ONE,
                                       0, 0)

            # the tail
            with tik_instance.else_scope():
                # score_filter
                score_filter_processing(tik_instance, data_gm, data_cb,
                                        (param_list[0], core_offset, num_tail,
                                         param_list[CONFIG_TWO], param_list[CONFIG_FIVE]))
                # topk
                call_topk_sort(tik_instance,
                               (data_gm.out_gm, data_cb.proposal_cb, data_cb.mem_swap,
                                data_gm.proposal_gm, data_cb.mem_swap2),
                               (data_cb.flag_cb, data_cb.num_total_cb, data_cb.num_total_gm),
                               (param_list[CONFIG_ONE], param_list[0],
                                param_list[CONFIG_FIVE], param_list[CONFIG_TWO]),
                               (core_offset0, 0, 0, core_offset, 0))

                num_tensor = tik_instance.Tensor("uint32",
                                                 (CONFIG_ONE, CONFIG_EIGHT),
                                                 name="num_tensor",
                                                 scope=tik.scope_ubuf)

                tik_scalar_min(tik_instance, param_list[CONFIG_ONE],
                               data_cb.num_post_score, data_cb.num_post_score)

                tik_instance.vector_dup(CONFIG_EIGHT,
                                        num_tensor,
                                        data_cb.num_post_score,
                                        CONFIG_ONE,
                                        CONFIG_ONE,
                                        CONFIG_EIGHT)
                tik_instance.data_move(data_gm.proposal_num[core_index, 0],
                                       num_tensor,
                                       0,
                                       CONFIG_ONE,
                                       CONFIG_ONE,
                                       0, 0)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[data_gm.rois_gm, data_gm.prob_gm],
                          outputs=[data_gm.out_gm, data_gm.proposal_num])
    print("============> IR line num >======== {}".format(tik_instance.get_ir_num()))

    return tik_instance


@util.check_input_type(dict, dict, dict, dict,
                       (float, int), int,
                       bool, int, str)
def score_filter_pre_sort(rois, cls_bg_prob, sorted_proposal, proposal_num,
                          score_threshold, k,
                          score_filter=True, core_max_num=CONFIG_EIGHT,
                          kernel_name="score_filter_pre_sort"):
    """
    the entry function of score_filter_pre_sort
    Parameters
    ----------
    rois : dict
        shape and dtype of input boxes
    cls_bg_prob : dict
        shape and dtype of input probobilities
    sorted_proposal: : dict
        shape and dtype of output sorted proposal
    proposal_num: dict
        shape and dtype of actual_output_proposal_num

    score_threshold : float, init=0,   score filter threshold
    k: the topk, init 6000

    score_filter: bool,  True
    core_max_num:  max num of core aviliable
    kernel_name : str
        kernel name, default value is "generate_rpn_proposals"
    Returns
    -------
    None
    """

    aicore_num = tik.Dprofile().get_aicore_num()
    if aicore_num > core_max_num:
        aicore_num = core_max_num

    check_input_dict((rois, cls_bg_prob, sorted_proposal, proposal_num),
                     (score_threshold, k, core_max_num))
    check_input_param((score_threshold, k, core_max_num, aicore_num), kernel_name)
    if k == 0:
        if rois.get("shape")[0] > MAX_TOPK:
            k = MAX_TOPK
        else:
            k = rois.get("shape")[0]

    # the main calculate processing
    tik_instance = score_filter_pre_sort_compute((rois, cls_bg_prob,
                                                  sorted_proposal, proposal_num),
                                                 (score_threshold, k, score_filter,
                                                  core_max_num, aicore_num,
                                                  rois.get("dtype")),
                                                 kernel_name)
    return tik_instance
