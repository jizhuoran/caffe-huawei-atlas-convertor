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

rpn_proposals_d_MDC_V200
"""

# pylint: disable=C0302
# pylint: disable=R0902
# pylint: disable=R0915
# pylint: disable=R0914
# pylint: disable=R0913
# pylint: disable=R0912


from te import tik


SHAPE_SIZE_LIMIT = 709920
MAX_HEIGHT = 2000
MAX_WIDTH = 3000
MAX_TOPK = 6000
CONFIG_ONE_NEG = -1
CONFIG_HALF = 0.5
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
CONFIG_TOPK = 4096
CONFIG_TOPK2 = 6144
CONFIG_LEN = 1536
MATRIX = 256
MAX_REP_TIME = 255
CONFIG_SCORE_THRESHOLD = 0
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


def get_ub_size():
    """
    get the size of UB menmory
    return: ub_size  in Byte
    """

    dprofile = tik.Dprofile()
    ub_size = dprofile.get_unified_buffer_size()
    return ub_size


class InitGmTensorV200:
    """
    define some tensors in the DDR
    used for v200
    """
    def __init__(self, tik_instance, input_para):
        shape_in = input_para[0]
        shape_out = input_para[1]
        num_topk = input_para[2]
        data_type = input_para[3]

        # input tensor
        self.rois_gm = tik_instance.Tensor(data_type,
                                           (shape_in, CONFIG_FOUR),
                                           name="rois_gm",
                                           scope=tik.scope_gm)
        self.prob_gm = tik_instance.Tensor(data_type,
                                           (shape_in, CONFIG_ONE),
                                           name="prob_gm",
                                           scope=tik.scope_gm)

        # Final output tensors in DDR
        self.box_gm = tik_instance.Tensor(data_type,
                                          (shape_out, CONFIG_FOUR),
                                          name="box_gm",
                                          scope=tik.scope_gm)

        # Tensors after the score filter
        self.proposal_cb = tik_instance.Tensor(data_type,
                                               (L1_MAX_NUM + CONFIG_FOUR, CONFIG_EIGHT),
                                               name="proposal_cb",
                                               scope=tik.scope_cbuf)

        self.mem_swap = tik_instance.Tensor(data_type,
                                            (num_topk * CONFIG_TWO + CONFIG_FOUR, CONFIG_EIGHT),
                                            name="men_swap",
                                            scope=tik.scope_cbuf)

        # large tensor for cache
        self.proposal_gm = tik_instance.Tensor(data_type,
                                               (shape_in, CONFIG_EIGHT),
                                               name="proposal_gm",
                                               scope=tik.scope_gm,
                                               is_workspace=True)
        self.mem_swap_gm = tik_instance.Tensor(data_type,
                                               (num_topk + CONFIG_FOUR, CONFIG_EIGHT),
                                               name="mem_swap_gm",
                                               scope=tik.scope_gm,
                                               is_workspace=True)

        # Tensors after the first topk
        self.proposal_post_topk = tik_instance.Tensor(data_type,
                                                      (num_topk + CONFIG_FOUR,
                                                       CONFIG_EIGHT),
                                                      name="proposal_post_topk",
                                                      scope=tik.scope_cbuf)
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


class MiddleTensorV200:
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


def score_filter_processing(tik_instance, data_gm, shape_in, score_threshold):
    """
    perform the score filter
    param tik_instance:
    param data_gm:
    param data_ub:
    return:
    """
    # tiling
    # num of box to processing one time, -128 to avoid rep_time problems
    const_one_core = CONFIG_UNIT * CONFIG_FOUR - CONFIG_MASK
    num_cycle = shape_in // const_one_core
    num_left = shape_in % const_one_core

    with tik_instance.new_stmt_scope():
        data_ub = MiddleTensorV200(tik_instance, const_one_core, "float16")
        with tik_instance.for_range(0, num_cycle, thread_num=CONFIG_ONE) as t_thread:
            num_offset = t_thread * const_one_core

            # move data from DDR to UB
            tik_instance.data_move(data_ub.rois_ub,
                                   data_gm.rois_gm[num_offset, 0],
                                   0,
                                   CONFIG_ONE,
                                   const_one_core // CONFIG_FOUR,
                                   0,
                                   0)
            tik_instance.data_move(data_ub.prob_ub,
                                   data_gm.prob_gm[num_offset],
                                   0,
                                   CONFIG_ONE,
                                   const_one_core // CONFIG_SIXTEEN,
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
                                 const_one_core // CONFIG_DATA_ALIGN,
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
                                 const_one_core // CONFIG_DATA_ALIGN,
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
                                 const_one_core // CONFIG_DATA_ALIGN,
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
                                 const_one_core // CONFIG_DATA_ALIGN,
                                 CONFIG_ONE,
                                 CONFIG_EIGHT,
                                 0,
                                 0,
                                 None,
                                 "normal")

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
                                 const_one_core // CONFIG_MASK,
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
                                     const_one_core // CONFIG_MASK,
                                     CONFIG_ONE,
                                     CONFIG_EIGHT,
                                     CONFIG_EIGHT,
                                     0,
                                     None,
                                     "normal")

            # combine to proposals
            # init the proposal tensor
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

                data_gm.num_post_score.set_as(data_gm.num_post_score + data_ub.num_scorefilter)

                with tik_instance.if_scope(data_gm.flag_cb == CONFIG_ONE):
                    num_here = ceil_div(data_ub.num_scorefilter, CONFIG_TWO) * CONFIG_TWO + \
                               data_gm.num_total_cb
                    with tik_instance.if_scope(num_here < L1_MAX_NUM):
                        tik_instance.data_move(data_gm.proposal_cb[data_gm.num_total_cb, 0],
                                               data_ub.proposal_ub,
                                               0,
                                               CONFIG_ONE,
                                               ceil_div(data_ub.num_scorefilter, CONFIG_TWO),
                                               0,
                                               0)
                        data_gm.num_total_cb.set_as(num_here)
                    with tik_instance.else_scope():
                        num_here = ceil_div(data_ub.num_scorefilter, CONFIG_TWO) * CONFIG_TWO + \
                                   data_gm.num_total_gm
                        data_gm.flag_cb.set_as(0)
                        tik_instance.data_move(data_gm.proposal_gm[data_gm.num_total_gm, 0],
                                               data_ub.proposal_ub,
                                               0,
                                               CONFIG_ONE,
                                               ceil_div(data_ub.num_scorefilter, CONFIG_TWO),
                                               0,
                                               0)
                        data_gm.num_total_gm.set_as(num_here)
                with tik_instance.else_scope():
                    num_here = ceil_div(data_ub.num_scorefilter, CONFIG_TWO) * CONFIG_TWO + \
                               data_gm.num_total_gm
                    tik_instance.data_move(data_gm.proposal_gm[data_gm.num_total_gm, 0],
                                           data_ub.proposal_ub,
                                           0,
                                           CONFIG_ONE,
                                           ceil_div(data_ub.num_scorefilter, CONFIG_TWO),
                                           0,
                                           0)
                    data_gm.num_total_gm.set_as(num_here)

    # =====> the tail
    num_offset = num_cycle * const_one_core
    with tik_instance.new_stmt_scope():
        data_ub = MiddleTensorV200(tik_instance, const_one_core, "float16")

        # move data from DDR to UB
        tik_instance.data_move(data_ub.rois_ub,
                               data_gm.rois_gm[num_offset, 0],
                               0,
                               CONFIG_ONE,
                               ceil_div(num_left, CONFIG_FOUR),
                               0,
                               0)

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
                               ceil_div(num_left, CONFIG_SIXTEEN),
                               0,
                               0)
        # ==================================>
        # transpose the boxes from N*4 to 4*N
        # x1,   3,   0001000100010001
        tik_instance.vreduce(CONFIG_MASK,
                             data_ub.rois_ub_tran,
                             data_ub.rois_ub,
                             CONFIG_THREE,
                             ceil_div(num_left, CONFIG_DATA_ALIGN),
                             CONFIG_ONE,
                             CONFIG_EIGHT,
                             0,
                             0,
                             None,
                             "normal")
        # y1,   4,   0010001000100010
        tik_instance.vreduce(CONFIG_MASK,
                             data_ub.rois_ub_tran[CONFIG_ONE, 0],
                             data_ub.rois_ub,
                             CONFIG_FOUR,
                             ceil_div(num_left, CONFIG_DATA_ALIGN),
                             CONFIG_ONE,
                             CONFIG_EIGHT,
                             0,
                             0,
                             None,
                             "normal")
        # x2,   5,   0100010001000100
        tik_instance.vreduce(CONFIG_MASK,
                             data_ub.rois_ub_tran[CONFIG_TWO, 0],
                             data_ub.rois_ub,
                             CONFIG_FIVE,
                             ceil_div(num_left, CONFIG_DATA_ALIGN),
                             CONFIG_ONE,
                             CONFIG_EIGHT,
                             0,
                             0,
                             None,
                             "normal")
        # y2,   6,   1000100010001000
        tik_instance.vreduce(CONFIG_MASK,
                             data_ub.rois_ub_tran[CONFIG_THREE, 0],
                             data_ub.rois_ub,
                             CONFIG_SIX,
                             ceil_div(num_left, CONFIG_DATA_ALIGN),
                             CONFIG_ONE,
                             CONFIG_EIGHT,
                             0,
                             0,
                             None,
                             "normal")

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
                             ceil_div(num_left, CONFIG_MASK),
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
                                 ceil_div(num_left, CONFIG_MASK),
                                 CONFIG_ONE,
                                 CONFIG_EIGHT,
                                 CONFIG_EIGHT,
                                 0,
                                 None,
                                 "normal")

        # combine to proposals
        # init the proposal tensor
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

            data_gm.num_post_score.set_as(data_gm.num_post_score + data_ub.num_scorefilter)

            with tik_instance.if_scope(data_gm.flag_cb == CONFIG_ONE):
                num_here = ceil_div(data_ub.num_scorefilter, CONFIG_TWO) * CONFIG_TWO + \
                           data_gm.num_total_cb
                with tik_instance.if_scope(num_here < L1_MAX_NUM):
                    tik_instance.data_move(data_gm.proposal_cb[data_gm.num_total_cb, 0],
                                           data_ub.proposal_ub,
                                           0,
                                           CONFIG_ONE,
                                           ceil_div(data_ub.num_scorefilter, CONFIG_TWO),
                                           0,
                                           0)
                    data_gm.num_total_cb.set_as(num_here)
                with tik_instance.else_scope():
                    num_here = ceil_div(data_ub.num_scorefilter, CONFIG_TWO) * CONFIG_TWO + \
                               data_gm.num_total_gm
                    data_gm.flag_cb.set_as(0)
                    tik_instance.data_move(data_gm.proposal_gm[data_gm.num_total_gm, 0],
                                           data_ub.proposal_ub,
                                           0,
                                           CONFIG_ONE,
                                           ceil_div(data_ub.num_scorefilter, CONFIG_TWO),
                                           0,
                                           0)
                    data_gm.num_total_gm.set_as(num_here)
            with tik_instance.else_scope():
                num_here = ceil_div(data_ub.num_scorefilter, CONFIG_TWO) * CONFIG_TWO + \
                           data_gm.num_total_gm
                tik_instance.data_move(data_gm.proposal_gm[data_gm.num_total_gm, 0],
                                       data_ub.proposal_ub,
                                       0,
                                       CONFIG_ONE,
                                       ceil_div(data_ub.num_scorefilter, CONFIG_TWO),
                                       0,
                                       0)
                data_gm.num_total_gm.set_as(num_here)


# ====================>  topk local
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


def tik_topk_256(tik_instance, data_ub):
    """
    the input data in data_ub_a
    the output results in data_ub_b
    param tik_instance:
    param data_tensor:
    return: None
    """

    data_ub_a = data_ub[0]
    data_ub_b = data_ub[CONFIG_ONE]

    tik_instance.vrpsort16(data_ub_b,
                           data_ub_a,
                           MATRIX // CONFIG_SIXTEEN)

    # 16 ---> 64
    rep_str = CONFIG_SIXTEEN
    tik_instance.vmrgsort4(data_ub_a,
                           (data_ub_b[0, 0],
                            data_ub_b[rep_str * CONFIG_ONE, 0],
                            data_ub_b[rep_str * CONFIG_TWO, 0],
                            data_ub_b[rep_str * CONFIG_THREE, 0]),
                           (rep_str, rep_str, rep_str, rep_str),
                           False,
                           CONFIG_SIXTEEN - CONFIG_ONE,
                           CONFIG_DATA_TRANS // rep_str)

    # 64 ---> 256
    rep_str = rep_str * CONFIG_FOUR
    tik_instance.vmrgsort4(data_ub_b,
                           (data_ub_a[0, 0],
                            data_ub_a[rep_str * CONFIG_ONE, 0],
                            data_ub_a[rep_str * CONFIG_TWO, 0],
                            data_ub_a[rep_str * CONFIG_THREE, 0]),
                           (rep_str, rep_str, rep_str, rep_str),
                           False,
                           CONFIG_SIXTEEN - CONFIG_ONE,
                           CONFIG_DATA_TRANS // rep_str)


def tik_topk_1024(tik_instance, data_ub):
    """
    the input data in data_ub_a
    the output results in data_ub_a
    param tik_instance:
    param data_tensor:
    return: None
    """

    data_ub_a = data_ub[0]
    data_ub_b = data_ub[CONFIG_ONE]

    tik_instance.vrpsort16(data_ub_b,
                           data_ub_a,
                           CONFIG_UNIT // CONFIG_SIXTEEN)

    # 16 ---> 64
    rep_str = CONFIG_SIXTEEN
    tik_instance.vmrgsort4(data_ub_a,
                           (data_ub_b[0, 0],
                            data_ub_b[rep_str * CONFIG_ONE, 0],
                            data_ub_b[rep_str * CONFIG_TWO, 0],
                            data_ub_b[rep_str * CONFIG_THREE, 0]),
                           (rep_str, rep_str, rep_str, rep_str),
                           False,
                           CONFIG_SIXTEEN - CONFIG_ONE,
                           MATRIX // rep_str)

    # 64 ---> 256
    rep_str = rep_str * CONFIG_FOUR
    tik_instance.vmrgsort4(data_ub_b,
                           (data_ub_a[0, 0],
                            data_ub_a[rep_str * CONFIG_ONE, 0],
                            data_ub_a[rep_str * CONFIG_TWO, 0],
                            data_ub_a[rep_str * CONFIG_THREE, 0]),
                           (rep_str, rep_str, rep_str, rep_str),
                           False,
                           CONFIG_SIXTEEN - CONFIG_ONE,
                           MATRIX // rep_str)

    # 256 ---> 1024
    rep_str = rep_str * CONFIG_FOUR
    tik_instance.vmrgsort4(data_ub_a,
                           (data_ub_b[0, 0],
                            data_ub_b[rep_str * CONFIG_ONE, 0],
                            data_ub_b[rep_str * CONFIG_TWO, 0],
                            data_ub_b[rep_str * CONFIG_THREE, 0]),
                           (rep_str, rep_str, rep_str, rep_str),
                           False,
                           CONFIG_SIXTEEN - CONFIG_ONE,
                           MATRIX // rep_str)


def tik_topk_4096(tik_instance, data_ub):
    """
    the input data in data_ub_a
    the output results in data_ub_b
    param tik_instance:
    param data_tensor:
    return: None
    """

    data_ub_a = data_ub[0]
    data_ub_b = data_ub[CONFIG_ONE]

    tik_instance.vrpsort16(data_ub_b,
                           data_ub_a,
                           CONFIG_UNIT // CONFIG_EIGHT)
    tik_instance.vrpsort16(data_ub_b[CONFIG_TOPK // CONFIG_TWO, 0],
                           data_ub_a[CONFIG_TOPK // CONFIG_TWO, 0],
                           CONFIG_UNIT // CONFIG_EIGHT)

    # rep_str = CONFIG_FOUR
    for t_rep in range(0, CONFIG_TWO):
        rep_str = CONFIG_SIXTEEN ** (t_rep + CONFIG_ONE)
        tik_instance.vmrgsort4(data_ub_a,
                               (data_ub_b[0, 0],
                                data_ub_b[rep_str * CONFIG_ONE, 0],
                                data_ub_b[rep_str * CONFIG_TWO, 0],
                                data_ub_b[rep_str * CONFIG_THREE, 0]),
                               (rep_str, rep_str, rep_str, rep_str),
                               False,
                               CONFIG_SIXTEEN - CONFIG_ONE,
                               CONFIG_UNIT // rep_str)

        rep_str = rep_str * CONFIG_FOUR
        tik_instance.vmrgsort4(data_ub_b,
                               (data_ub_a[0, 0],
                                data_ub_a[rep_str * CONFIG_ONE, 0],
                                data_ub_a[rep_str * CONFIG_TWO, 0],
                                data_ub_a[rep_str * CONFIG_THREE, 0]),
                               (rep_str, rep_str, rep_str, rep_str),
                               False,
                               CONFIG_SIXTEEN - CONFIG_ONE,
                               CONFIG_UNIT // rep_str)


def tik_topk_6114(tik_instance, data_ub):
    """
    the input data in data_ub_a
    the output results in data_ub_a
    param tik_instance:
    param data_tensor:
    return: None
    """

    data_ub_a = data_ub[0]
    data_ub_b = data_ub[CONFIG_ONE]

    tik_instance.vrpsort16(data_ub_b,
                           data_ub_a,
                           CONFIG_TOPK2 // CONFIG_DATA_ALIGN)
    tik_instance.vrpsort16(data_ub_b[CONFIG_TOPK2 // CONFIG_TWO, 0],
                           data_ub_a[CONFIG_TOPK2 // CONFIG_TWO, 0],
                           CONFIG_TOPK2 // CONFIG_DATA_ALIGN)

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
                           CONFIG_TOPK2 // (rep_str * CONFIG_FOUR))

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
                           CONFIG_TOPK2 // (rep_str * CONFIG_FOUR))

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
                           CONFIG_TOPK2 // (rep_str * CONFIG_FOUR))

    # 1024 * 3 --> 3072
    rep_str = rep_str * CONFIG_FOUR
    tik_instance.vmrgsort4(data_ub_b,
                           (data_ub_a[0, 0],
                            data_ub_a[rep_str * CONFIG_ONE, 0],
                            data_ub_a[rep_str * CONFIG_TWO, 0],
                            data_ub_a[rep_str * CONFIG_TWO, 0]),
                           (rep_str, rep_str, rep_str, rep_str),
                           False,
                           CONFIG_EIGHT - CONFIG_ONE,
                           CONFIG_ONE)
    tik_instance.vmrgsort4(data_ub_b[rep_str * CONFIG_THREE, 0],
                           (data_ub_a[rep_str * CONFIG_THREE, 0],
                            data_ub_a[rep_str * CONFIG_FOUR, 0],
                            data_ub_a[rep_str * CONFIG_FIVE, 0],
                            data_ub_a[rep_str * CONFIG_FIVE, 0]),
                           (rep_str, rep_str, rep_str, rep_str),
                           False,
                           CONFIG_EIGHT - CONFIG_ONE,
                           CONFIG_ONE)
    # 3072 * 2 --> 6144
    rep_str = rep_str * CONFIG_THREE
    tik_instance.vmrgsort4(data_ub_a,
                           (data_ub_b[0, 0],
                            data_ub_b[rep_str * CONFIG_ONE, 0],
                            data_ub_b[rep_str * CONFIG_ONE, 0],
                            data_ub_b[rep_str * CONFIG_ONE, 0]),
                           (rep_str, rep_str, rep_str, rep_str),
                           False,
                           CONFIG_FOUR - CONFIG_ONE,
                           CONFIG_ONE)


def tik_topk_internal_sort(tik_instance, data_gm, score_threshold, num_actual, topk_k):
    """
    the proposals can be moved in at one time
    param tik_instance:
    param data_gm: a list
            src_tensor = data_gm[0] : the tensor store the original proposals
            dst_tensor = data_gm[1] : the tensor to store the results in DDR / L1
    score_threshold:  for dump
    param num_actual: actual num of the input proposals
    topk_k : the max num needed after topk
    return: None
    """

    src_tensor = data_gm[0]
    dst_tensor = data_gm[1]
    data_type = "float16"
    n_required = tik_instance.Scalar("uint32",
                                     name="n_required")
    tik_scalar_min(tik_instance, num_actual, topk_k, n_required)

    tik_instance.tikdb.debug_print('"proposal_actual_num:"+str(num_actual)')

    with tik_instance.new_stmt_scope():
        with tik_instance.if_scope(num_actual > CONFIG_TOPK):
            # 6144
            data_ub_a = tik_instance.Tensor(data_type,
                                            (CONFIG_TOPK2, CONFIG_EIGHT),
                                            name="data_ub_a",
                                            scope=tik.scope_ubuf)
            data_ub_b = tik_instance.Tensor(data_type,
                                            (CONFIG_TOPK2, CONFIG_EIGHT),
                                            name="data_ub_b",
                                            scope=tik.scope_ubuf)

            # move data from DDR to UB
            with tik_instance.if_scope(num_actual <= CONFIG_TOPK2):
                rep_time = ceil_div((CONFIG_TOPK2 - num_actual), CONFIG_SIXTEEN)
                offset = CONFIG_TOPK2 - rep_time * CONFIG_SIXTEEN
                tik_instance.vector_dup(CONFIG_MASK,
                                        data_ub_a[offset, 0],
                                        score_threshold,
                                        rep_time,
                                        CONFIG_ONE,
                                        CONFIG_EIGHT)
            tik_instance.data_move(data_ub_a,
                                   src_tensor,
                                   0,
                                   CONFIG_ONE,
                                   ceil_div(num_actual, CONFIG_TWO),
                                   0, 0)
            # perform topk
            tik_topk_6114(tik_instance, (data_ub_a, data_ub_b))

            # move data out to DDR
            tik_instance.data_move(dst_tensor,
                                   data_ub_a,
                                   0,
                                   CONFIG_ONE,
                                   ceil_div(n_required, CONFIG_TWO),
                                   0, 0)
        with tik_instance.else_scope():
            with tik_instance.if_scope(num_actual > CONFIG_UNIT):
                # 4096
                data_ub_a = tik_instance.Tensor(data_type,
                                                (CONFIG_TOPK, CONFIG_EIGHT),
                                                name="data_ub_a",
                                                scope=tik.scope_ubuf)
                data_ub_b = tik_instance.Tensor(data_type,
                                                (CONFIG_TOPK, CONFIG_EIGHT),
                                                name="data_ub_b",
                                                scope=tik.scope_ubuf)

                # move data from DDR to UB
                with tik_instance.if_scope(num_actual < CONFIG_TOPK):
                    rep_time = ceil_div((CONFIG_TOPK - num_actual), CONFIG_SIXTEEN)
                    offset = CONFIG_TOPK - rep_time * CONFIG_SIXTEEN
                    tik_instance.vector_dup(CONFIG_MASK,
                                            data_ub_a[offset, 0],
                                            score_threshold,
                                            rep_time,
                                            CONFIG_ONE,
                                            CONFIG_EIGHT)
                tik_instance.data_move(data_ub_a,
                                       src_tensor,
                                       0,
                                       CONFIG_ONE,
                                       ceil_div(num_actual, CONFIG_TWO),
                                       0, 0)
                # perform topk
                tik_topk_4096(tik_instance, (data_ub_a, data_ub_b))

                # move data out to DDR
                tik_instance.data_move(dst_tensor,
                                       data_ub_b,
                                       0,
                                       CONFIG_ONE,
                                       ceil_div(n_required, CONFIG_TWO),
                                       0, 0)
            with tik_instance.else_scope():
                with tik_instance.if_scope(num_actual > MATRIX):
                    # 1024
                    data_ub_a = tik_instance.Tensor(data_type,
                                                    (CONFIG_UNIT, CONFIG_EIGHT),
                                                    name="data_ub_a",
                                                    scope=tik.scope_ubuf)
                    data_ub_b = tik_instance.Tensor(data_type,
                                                    (CONFIG_UNIT, CONFIG_EIGHT),
                                                    name="data_ub_b",
                                                    scope=tik.scope_ubuf)

                    # move data from DDR to UB
                    with tik_instance.if_scope(num_actual < CONFIG_UNIT):
                        rep_time = ceil_div((CONFIG_UNIT - num_actual), CONFIG_SIXTEEN)
                        offset = CONFIG_UNIT - rep_time * CONFIG_SIXTEEN
                        tik_instance.vector_dup(CONFIG_MASK,
                                                data_ub_a[offset, 0],
                                                score_threshold,
                                                rep_time,
                                                CONFIG_ONE,
                                                CONFIG_EIGHT)
                    tik_instance.data_move(data_ub_a,
                                           src_tensor,
                                           0,
                                           CONFIG_ONE,
                                           ceil_div(num_actual, CONFIG_TWO),
                                           0, 0)
                    # perform topk
                    tik_topk_1024(tik_instance, (data_ub_a, data_ub_b))

                    # move data out to DDR
                    tik_instance.data_move(dst_tensor,
                                           data_ub_a,
                                           0,
                                           CONFIG_ONE,
                                           ceil_div(n_required, CONFIG_TWO),
                                           0, 0)
                with tik_instance.else_scope():
                    # 256
                    data_ub_a = tik_instance.Tensor(data_type,
                                                    (MATRIX, CONFIG_EIGHT),
                                                    name="data_ub_a",
                                                    scope=tik.scope_ubuf)
                    data_ub_b = tik_instance.Tensor(data_type,
                                                    (MATRIX, CONFIG_EIGHT),
                                                    name="data_ub_b",
                                                    scope=tik.scope_ubuf)

                    # move data from DDR to UB
                    tik_instance.vector_dup(CONFIG_MASK,
                                            data_ub_a,
                                            score_threshold,
                                            MATRIX // CONFIG_SIXTEEN,
                                            CONFIG_ONE,
                                            CONFIG_EIGHT)
                    tik_instance.data_move(data_ub_a,
                                           src_tensor,
                                           0,
                                           CONFIG_ONE,
                                           ceil_div(num_actual, CONFIG_TWO),
                                           0, 0)
                    # perform topk
                    tik_topk_256(tik_instance, (data_ub_a, data_ub_b))

                    # move data out to DDR
                    tik_instance.data_move(dst_tensor,
                                           data_ub_b,
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
                                     (6144, CONFIG_EIGHT),
                                     name="dst_ub",
                                     scope=tik.scope_ubuf)
        dest_pos_ = tik_instance.Scalar("uint32", "dest_pos_", dest_pos)
        n_total_selected_ = tik_instance.Scalar("uint32", "n_total_selected_", 0)
        n_selected_ = tik_instance.Scalar("uint32", "n_selected_", 0)

        num_exhausted = [tik_instance.Scalar("uint32") for i in range(CONFIG_FOUR)]
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


def tik_topk_external_sort(tik_instance, data_gm, score_threshold, num_actual, topk_k):
    """

    :param tik_instance:
    :param data_gm:
    :param score_threshold:
    :param num_actual:
    :param topk_k:
    :return:
    """

    src_tensor = data_gm[0]
    mem_swap = data_gm[1]
    mem_swap_2 = data_gm[2]
    dst_tensor = data_gm[3]
    data_type = "float16"
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
            offset = CONFIG_TOPK2 * t_cyc
            # move data from DDR to UB
            tik_instance.data_move(data_ub_a,
                                   src_tensor[offset, 0],
                                   0,
                                   CONFIG_ONE,
                                   ceil_div(CONFIG_TOPK2, CONFIG_TWO),
                                   0, 0)
            # perform topk
            tik_topk_6114(tik_instance, (data_ub_a, data_ub_b))

            # move data out to DDR
            tik_instance.data_move(mem_swap[offset, 0],
                                   data_ub_a,
                                   0,
                                   CONFIG_ONE,
                                   ceil_div(CONFIG_TOPK2, CONFIG_TWO),
                                   0, 0)

        # the tail
        offset = CONFIG_TOPK2 * num_cycle
        with tik_instance.if_scope(num_left > 0):
            # move data from DDR to UB
            rep_time = ceil_div((CONFIG_TOPK2 - num_left), CONFIG_SIXTEEN)
            offset_ub = CONFIG_TOPK2 - rep_time * CONFIG_SIXTEEN
            with tik_instance.if_scope(rep_time > MAX_REP_TIME):
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
            with tik_instance.else_scope():
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
            with tik_instance.if_scope(num_left > CONFIG_TOPK):
                # perform topk
                tik_topk_6114(tik_instance, (data_ub_a, data_ub_b))
                # move data out to DDR
                tik_instance.data_move(mem_swap[offset, 0],
                                       data_ub_a,
                                       0,
                                       CONFIG_ONE,
                                       ceil_div(num_left, CONFIG_TWO),
                                       0, 0)
            with tik_instance.else_scope():
                with tik_instance.if_scope(num_left > CONFIG_UNIT):
                    # perform topk
                    tik_topk_4096(tik_instance, (data_ub_a, data_ub_b))
                    # move data out to DDR
                    tik_instance.data_move(mem_swap[offset, 0],
                                           data_ub_b,
                                           0,
                                           CONFIG_ONE,
                                           ceil_div(num_left, CONFIG_TWO),
                                           0, 0)
                with tik_instance.else_scope():
                    # perform topk
                    tik_topk_1024(tik_instance, (data_ub_a, data_ub_b))
                    # move data out to DDR
                    tik_instance.data_move(mem_swap[offset, 0],
                                           data_ub_a,
                                           0,
                                           CONFIG_ONE,
                                           ceil_div(num_left, CONFIG_TWO),
                                           0, 0)

    src_pos_ = [tik_instance.Scalar(dtype="uint32") for i in range(CONFIG_FOUR)]
    for t_cyc in range(CONFIG_FOUR):
        src_pos_[t_cyc].set_as(CONFIG_TOPK2 * t_cyc)

    # combine the sorted lists
    with tik_instance.if_scope(num_cycle < CONFIG_FOUR):
        with tik_instance.new_stmt_scope():
            with tik_instance.if_scope(num_cycle == CONFIG_THREE):
                tik_topk_vms4(tik_instance,
                              dst_tensor,
                              0,
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
                              0,
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
                              0,
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
                          0,
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

                dst_offset1 = (t_cyc % CONFIG_TWO) * CONFIG_TOPK2
                dst_offset2 = ((t_cyc + CONFIG_ONE) % CONFIG_TWO) * CONFIG_TOPK2
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

        dst_offset1 = CONFIG_TOPK2 * (max_rep % CONFIG_TWO)
        src_pos_[CONFIG_ONE].set_as(src_pos_[CONFIG_ONE] + CONFIG_TOPK2 * CONFIG_THREE)
        src_pos_[CONFIG_TWO].set_as(src_pos_[CONFIG_TWO] + CONFIG_TOPK2 * CONFIG_THREE)
        src_pos_[CONFIG_THREE].set_as(src_pos_[CONFIG_THREE] + CONFIG_TOPK2 * CONFIG_THREE)
        with tik_instance.if_scope(left_rep == 0):
            with tik_instance.new_stmt_scope():
                tik_topk_vms4(tik_instance,
                              dst_tensor,
                              0,
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
                              0,
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
                              0,
                              (mem_swap_2, mem_swap, None, None),
                              (dst_offset1,
                               src_pos_[CONFIG_ONE],
                               src_pos_[CONFIG_TWO],
                               src_pos_[CONFIG_THREE]),
                              (topk_k, topk_k, 0, 0),
                              CONFIG_THREE,
                              topk_k,
                              score_threshold)


def tik_topk(tik_instance, data_tensor, proposal_num, k, score_threshold):
    """
    do the topk
    param tik_instance:
    param data_tensor: a list
        data_in = data_tensor[0]  : proposals to be sorted
        data_out = data_tensor[1]  : the tensor to store the results
        mem_swap = data_tensor[2]  :  the middle tensor to store the sorted list 6144,
                                     the size of this tensor should be same as the data_in
        mem_swap_2 = data_tensor[3] : the middle tensor for list combining, as least [k*2, 8]

    param score_threshold: used for dump
    param k:  max num of proposals output
    param proposal_num: actual num of proposals
    return: None
    """
    data_in = data_tensor[0]
    data_out = data_tensor[1]

    with tik_instance.if_scope(tik.all(proposal_num <= CONFIG_TOPK2, proposal_num > 0)):
        with tik_instance.new_stmt_scope():
            tik_topk_internal_sort(tik_instance,
                                   (data_in, data_out),
                                   score_threshold,
                                   proposal_num,
                                   k)

    with tik_instance.else_scope():
        mem_swap = data_tensor[2]
        mem_swap_2 = data_tensor[3]
        tik_topk_external_sort(tik_instance,
                               (data_in, mem_swap, mem_swap_2, data_out),
                               score_threshold,
                               proposal_num,
                               k)


def call_topk_sort_v200(tik_instance, input_tensor, proposal_num, input_param):
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
    return:  None
    """

    # proposals stored in L1
    # L1 Buffer, the results stored in the dst_tensor
    tik_topk(tik_instance,
             (input_tensor[CONFIG_ONE], input_tensor[0],
              input_tensor[CONFIG_ONE], input_tensor[CONFIG_TWO]),
             proposal_num[CONFIG_ONE],
             input_param[0],
             input_param[CONFIG_ONE])

    # not only in L1 buffer
    with tik_instance.if_scope(tik.all(proposal_num[0] == 0, proposal_num[CONFIG_TWO] > 0)):
        # both L1 buffer and DDR
        with tik_instance.if_scope(proposal_num[CONFIG_TWO] < L1_MAX_NUM):
            # useing the L1 Buffer as swap_tensor, the results stored in the DDR
            tik_topk(tik_instance,
                     (input_tensor[CONFIG_THREE], input_tensor[CONFIG_FOUR],
                      input_tensor[CONFIG_ONE], input_tensor[CONFIG_TWO]),
                     proposal_num[CONFIG_TWO],
                     input_param[0],
                     input_param[CONFIG_ONE])
        with tik_instance.else_scope():
            tik_topk(tik_instance,
                     (input_tensor[CONFIG_THREE], input_tensor[CONFIG_FOUR],
                      input_tensor[CONFIG_THREE], input_tensor[CONFIG_TWO]),
                     proposal_num[CONFIG_TWO],
                     input_param[0],
                     input_param[CONFIG_ONE])

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
                                   input_tensor[0],
                                   0,
                                   CONFIG_ONE,
                                   ceil_div(input_param[0], CONFIG_TWO),
                                   0, 0)
            tik_instance.data_move(input_tensor[CONFIG_TWO],
                                   temp_ub,
                                   0,
                                   CONFIG_ONE,
                                   ceil_div(input_param[0], CONFIG_TWO),
                                   0, 0)
            # DDR buffer results, storing in input_tensor[CONFIG_FOUR]-->mem_swap_gm
            tik_instance.data_move(temp_ub,
                                   input_tensor[CONFIG_FOUR],
                                   0,
                                   CONFIG_ONE,
                                   ceil_div(num_in_gm, CONFIG_TWO),
                                   0, 0)
            tik_instance.data_move(input_tensor[CONFIG_TWO][CONFIG_TOPK2, 0],
                                   temp_ub,
                                   0,
                                   CONFIG_ONE,
                                   ceil_div(num_in_gm, CONFIG_TWO),
                                   0, 0)

        tik_topk_vms4(tik_instance,
                      input_tensor[0],
                      0,
                      (input_tensor[CONFIG_TWO], input_tensor[CONFIG_TWO],
                       None, None),
                      (0, CONFIG_TOPK2, 0, 0),
                      (input_param[0], num_in_gm, 0, 0),
                      CONFIG_THREE,
                      input_param[0],
                      input_param[CONFIG_ONE])


def clip_size_filter(tik_instance, input_tensor, proposal_num, input_param):
    """
    perform the clip boxes and size filter
    param tik_instance:
    param input_tensor:
        0  src_proposals
        1  dst_proposals
    param proposal_num:
        0  k:
        1  actual proposal_num, a scalar
    param input_param:
            0   img_size: [h, w]
            1   min_size: the threshold
            2  data_type
            3  score_filter
    return: None
    """

    proposal_ub = tik_instance.Tensor(input_param[CONFIG_TWO],
                                      (proposal_num[0], CONFIG_EIGHT),
                                      name="proposal_ub",
                                      scope=tik.scope_ubuf)
    num = ceil_div(proposal_num[0], CONFIG_MASK) * CONFIG_MASK # 6016
    box_ub = tik_instance.Tensor(input_param[CONFIG_TWO],
                                 (CONFIG_TEN, num),
                                 name="box_ub",
                                 scope=tik.scope_ubuf)
    num = num // CONFIG_MASK  # 47
    num = ceil_div(num, CONFIG_SIXTEEN) * CONFIG_SIXTEEN  # 48
    mask_ub = tik_instance.Tensor("uint16",
                                  (CONFIG_THREE, num * CONFIG_EIGHT),
                                  name="mask_ub",
                                  scope=tik.scope_ubuf)
    num_post_nms = tik_instance.Scalar("uint32", name="num_post_nms", init_value=0)

    # move data into UB
    # the actual proposals num 16 times (8 block)
    num = proposal_num[CONFIG_ONE] // CONFIG_MASK * CONFIG_MASK
    tik_instance.vector_dup(CONFIG_MASK,
                            proposal_ub[num, 0],
                            0,
                            CONFIG_ONE,
                            CONFIG_ONE,
                            CONFIG_EIGHT)
    tik_instance.data_move(proposal_ub,
                           input_tensor[0],
                           0,
                           CONFIG_ONE,
                           ceil_div(proposal_num[CONFIG_ONE], CONFIG_TWO),
                           0, 0)

    # ======================>
    # extract the proposals into box
    tik_instance.vector_dup(CONFIG_MASK,
                            box_ub,
                            0,
                            ceil_div(proposal_num[0], CONFIG_MASK) * CONFIG_FIVE,
                            CONFIG_ONE,
                            CONFIG_EIGHT)
    num = ceil_div(proposal_num[CONFIG_ONE], CONFIG_SIXTEEN)
    with tik_instance.if_scope(num > MAX_REP_TIME):
        with tik_instance.for_range(0, CONFIG_FIVE) as t_cyc:
            tik_instance.vextract(box_ub[t_cyc, 0],
                                  proposal_ub,
                                  MAX_REP_TIME,
                                  t_cyc)
        num = num - MAX_REP_TIME
        with tik_instance.for_range(0, CONFIG_FIVE) as t_cyc:
            tik_instance.vextract(box_ub[t_cyc, MAX_REP_TIME * CONFIG_SIXTEEN],
                                  proposal_ub[MAX_REP_TIME * CONFIG_SIXTEEN, 0],
                                  num,
                                  t_cyc)
    with tik_instance.else_scope():
        with tik_instance.for_range(0, CONFIG_FIVE) as t_cyc:
            tik_instance.vextract(box_ub[t_cyc, 0],
                                  proposal_ub,
                                  num,
                                  t_cyc)

    # ======================>
    # perform the clip box
    num = ceil_div(proposal_num[0], CONFIG_MASK) * CONFIG_FOUR  # 47*4
    tik_instance.vrelu(CONFIG_MASK,
                       box_ub[CONFIG_FIVE, 0],
                       box_ub,
                       num,
                       CONFIG_ONE,
                       CONFIG_ONE,
                       CONFIG_EIGHT,
                       CONFIG_EIGHT)
    num = num // CONFIG_FOUR
    # x1
    tik_instance.vmins(CONFIG_MASK,
                       box_ub,
                       box_ub[CONFIG_FIVE, 0],
                       input_param[0][CONFIG_ONE],
                       num,
                       CONFIG_ONE,
                       CONFIG_ONE,
                       CONFIG_EIGHT,
                       CONFIG_EIGHT)
    # x2
    tik_instance.vmins(CONFIG_MASK,
                       box_ub[CONFIG_TWO, 0],
                       box_ub[CONFIG_SEVEN, 0],
                       input_param[0][CONFIG_ONE],
                       num,
                       CONFIG_ONE,
                       CONFIG_ONE,
                       CONFIG_EIGHT,
                       CONFIG_EIGHT)
    # y1
    tik_instance.vmins(CONFIG_MASK,
                       box_ub[CONFIG_ONE, 0],
                       box_ub[CONFIG_SIX, 0],
                       input_param[0][0],
                       num,
                       CONFIG_ONE,
                       CONFIG_ONE,
                       CONFIG_EIGHT,
                       CONFIG_EIGHT)
    # y2
    tik_instance.vmins(CONFIG_MASK,
                       box_ub[CONFIG_THREE, 0],
                       box_ub[CONFIG_EIGHT, 0],
                       input_param[0][0],
                       num,
                       CONFIG_ONE,
                       CONFIG_ONE,
                       CONFIG_EIGHT,
                       CONFIG_EIGHT)

    if input_param[CONFIG_THREE]:
        # ======================>
        # perform size filter
        # x2 - x1
        tik_instance.vsub(CONFIG_MASK,
                          box_ub[CONFIG_FIVE, 0],
                          box_ub[CONFIG_TWO, 0],
                          box_ub,
                          num,
                          CONFIG_ONE,
                          CONFIG_ONE,
                          CONFIG_ONE,
                          CONFIG_EIGHT,
                          CONFIG_EIGHT,
                          CONFIG_EIGHT)
        # y2 - y1
        tik_instance.vsub(CONFIG_MASK,
                          box_ub[CONFIG_SIX, 0],
                          box_ub[CONFIG_THREE, 0],
                          box_ub[CONFIG_ONE, 0],
                          num,
                          CONFIG_ONE,
                          CONFIG_ONE,
                          CONFIG_ONE,
                          CONFIG_EIGHT,
                          CONFIG_EIGHT,
                          CONFIG_EIGHT)

        # perform the compare
        rep_time = ceil_div(ceil_div(proposal_num[0], CONFIG_MASK), CONFIG_SIXTEEN)  # 3
        tik_instance.vector_dup(CONFIG_MASK,
                                mask_ub,
                                0,
                                rep_time * CONFIG_THREE,
                                CONFIG_ONE,
                                CONFIG_EIGHT)
        # width
        tik_instance.vcmpvs_gt(mask_ub,
                               box_ub[CONFIG_FIVE, 0],
                               input_param[CONFIG_ONE],
                               num,
                               CONFIG_ONE,
                               CONFIG_EIGHT)
        # height
        tik_instance.vcmpvs_gt(mask_ub[CONFIG_ONE, 0],
                               box_ub[CONFIG_SIX, 0],
                               input_param[CONFIG_ONE],
                               num,
                               CONFIG_ONE,
                               CONFIG_EIGHT)

        tik_instance.vand(CONFIG_MASK,
                          mask_ub[CONFIG_TWO, 0],
                          mask_ub,
                          mask_ub[CONFIG_ONE, 0],
                          rep_time,
                          CONFIG_ONE,
                          CONFIG_ONE,
                          CONFIG_ONE,
                          CONFIG_EIGHT,
                          CONFIG_EIGHT,
                          CONFIG_EIGHT)

        # vreduce
        tik_instance.vector_dup(CONFIG_MASK,
                                box_ub[CONFIG_FIVE, 0],
                                0,
                                ceil_div(proposal_num[0], CONFIG_MASK) * CONFIG_FIVE,
                                CONFIG_ONE,
                                CONFIG_EIGHT)

        rep_time = ceil_div(proposal_num[0], CONFIG_MASK)
        with tik_instance.for_range(0, CONFIG_FIVE) as t_cyc:
            tik_instance.vreduce(CONFIG_MASK,
                                 box_ub[CONFIG_FIVE + t_cyc, 0],
                                 box_ub[t_cyc, 0],
                                 mask_ub[CONFIG_TWO, 0],
                                 rep_time,
                                 CONFIG_ONE,
                                 CONFIG_EIGHT,
                                 CONFIG_EIGHT,
                                 0,
                                 num_post_nms,
                                 "normal")

        proposal_num[CONFIG_ONE].set_as(num_post_nms)

        # concat into proposal
        with tik_instance.if_scope(num_post_nms % CONFIG_TWO == CONFIG_ONE):
            tik_instance.vector_dup(CONFIG_SIXTEEN,
                                    proposal_ub[num_post_nms - CONFIG_ONE, 0],
                                    0,
                                    CONFIG_ONE,
                                    CONFIG_ONE,
                                    CONFIG_EIGHT)

        num = ceil_div(num_post_nms, CONFIG_SIXTEEN)
        with tik_instance.if_scope(num > MAX_REP_TIME):
            with tik_instance.for_range(0, CONFIG_FIVE) as t_cyc:
                tik_instance.vconcat(proposal_ub,
                                     box_ub[CONFIG_FIVE + t_cyc, 0],
                                     MAX_REP_TIME,
                                     t_cyc)
            num = num - MAX_REP_TIME
            with tik_instance.for_range(0, CONFIG_FIVE) as t_cyc:
                tik_instance.vconcat(proposal_ub[MAX_REP_TIME * CONFIG_SIXTEEN, 0],
                                     box_ub[CONFIG_FIVE + t_cyc, MAX_REP_TIME * CONFIG_SIXTEEN],
                                     num,
                                     t_cyc)
        with tik_instance.else_scope():
            with tik_instance.for_range(0, CONFIG_FIVE) as t_cyc:
                tik_instance.vconcat(proposal_ub,
                                     box_ub[CONFIG_FIVE + t_cyc, 0],
                                     num,
                                     t_cyc)
    else:
        num_post_nms.set_as(proposal_num[CONFIG_ONE])
        num = ceil_div(num_post_nms, CONFIG_SIXTEEN)
        with tik_instance.if_scope(num > MAX_REP_TIME):
            with tik_instance.for_range(0, CONFIG_FIVE) as t_cyc:
                tik_instance.vconcat(proposal_ub,
                                     box_ub[t_cyc, 0],
                                     MAX_REP_TIME,
                                     t_cyc)
            num = num - MAX_REP_TIME
            with tik_instance.for_range(0, CONFIG_FIVE) as t_cyc:
                tik_instance.vconcat(proposal_ub[MAX_REP_TIME * CONFIG_SIXTEEN, 0],
                                     box_ub[t_cyc, MAX_REP_TIME * CONFIG_SIXTEEN],
                                     num,
                                     t_cyc)
        with tik_instance.else_scope():
            with tik_instance.for_range(0, CONFIG_FIVE) as t_cyc:
                tik_instance.vconcat(proposal_ub,
                                     box_ub[t_cyc, 0],
                                     num,
                                     t_cyc)

    # move data from UB to DDR/L1
    with tik_instance.if_scope(num_post_nms > 0):
        tik_instance.data_move(input_tensor[CONFIG_ONE],
                               proposal_ub,
                               0,
                               CONFIG_ONE,
                               ceil_div(num_post_nms, CONFIG_TWO),
                               0, 0)


# =========>  NMS  here
def nms_local(tik_instance, data_tensor, input_param):
    """
    perform the nms
    :param tik_instance:
    :param data_tensor:
    :param input_param:
    :return:
    """

    actual_input_nms_num = input_param[0]
    nms_threshold = input_param[CONFIG_ONE]
    img_size = input_param[CONFIG_TWO]
    post_nms_num = input_param[CONFIG_THREE]

    data_x_gm = data_tensor[0]
    data_y_gm = data_tensor[CONFIG_ONE]

    # init some param
    nms_threshold_new = nms_threshold / (CONFIG_ONE + nms_threshold)
    # scalar factor
    down_factor = (CONFIG_FP16 / (img_size[0] * img_size[CONFIG_ONE] * CONFIG_TWO)) ** CONFIG_HALF

    # init tensor every 128 proposals
    data_x_ub_burst = tik_instance.Tensor("float16",
                                          (CONFIG_MASK, CONFIG_EIGHT),
                                          name="data_x_ub_burst",
                                          scope=tik.scope_ubuf)
    data_x_ub_burst_reduce = tik_instance.Tensor("float16",
                                                 (CONFIG_MASK, CONFIG_EIGHT),
                                                 name="data_x_ub_burst_reduce",
                                                 scope=tik.scope_ubuf)
    tik_instance.vector_dup(CONFIG_MASK,
                            data_x_ub_burst_reduce,
                            0,
                            CONFIG_EIGHT,
                            CONFIG_ONE,
                            CONFIG_EIGHT)

    # init nms tensor
    temp_area_ub = tik_instance.Tensor("float16",
                                       (CONFIG_SIXTEEN, CONFIG_SIXTEEN),
                                       name="temp_area_ub",
                                       scope=tik.scope_ubuf)
    iou_ub = tik_instance.Tensor("float16",
                                 (CONFIG_SIXTEEN, CONFIG_SIXTEEN, CONFIG_SIXTEEN),
                                 name="iou_ub",
                                 scope=tik.scope_ubuf)
    join_ub = tik_instance.Tensor("float16",
                                  (CONFIG_SIXTEEN, CONFIG_SIXTEEN, CONFIG_SIXTEEN),
                                  name="join_ub",
                                  scope=tik.scope_ubuf)
    join_ub1 = tik_instance.Tensor("float16",
                                   (CONFIG_SIXTEEN, CONFIG_SIXTEEN, CONFIG_SIXTEEN),
                                   name="join_ub1",
                                   scope=tik.scope_ubuf)
    sup_matrix_ub = tik_instance.Tensor("uint16",
                                        (CONFIG_SIXTEEN, CONFIG_SIXTEEN, CONFIG_SIXTEEN),
                                        name="sup_matrix_ub",
                                        scope=tik.scope_ubuf)
    temp_sup_vec_ub = tik_instance.Tensor("uint16",
                                          (CONFIG_EIGHT, CONFIG_SIXTEEN),
                                          name="temp_sup_vec_ub",
                                          scope=tik.scope_ubuf)

    # init final select coord
    selected_reduced_coord_ub = tik_instance.Tensor("float16",
                                                    (MATRIX - CONFIG_DATA_ALIGN, CONFIG_FOUR),
                                                    name="selected_reduced_coord_ub",
                                                    scope=tik.scope_ubuf)
    tik_instance.vector_dup(CONFIG_DATA_TRANS,
                            selected_reduced_coord_ub,
                            0,
                            CONFIG_SIXTEEN - CONFIG_TWO,
                            CONFIG_ONE,
                            CONFIG_FOUR)

    # init middle selected proposals
    temp_reduced_proposals_ub = tik_instance.Tensor("float16",
                                                    (CONFIG_MASK, CONFIG_EIGHT),
                                                    name="temp_reduced_proposals_ub",
                                                    scope=tik.scope_ubuf)
    tik_instance.vector_dup(CONFIG_MASK,
                            temp_reduced_proposals_ub,
                            0,
                            CONFIG_EIGHT,
                            CONFIG_ONE,
                            CONFIG_EIGHT)

    # init middle selected area
    selected_area_ub = tik_instance.Tensor("float16",
                                           (CONFIG_SIXTEEN, CONFIG_SIXTEEN),
                                           name="selected_area_ub",
                                           scope=tik.scope_ubuf)

    # init middle sup_vec
    sup_vec_ub = tik_instance.Tensor("uint16",
                                     (CONFIG_SIXTEEN, CONFIG_SIXTEEN),
                                     name="sup_vec_ub",
                                     scope=tik.scope_ubuf)

    scalar_zero = tik_instance.Scalar(dtype="uint16")
    scalar_zero.set_as(0)
    sup_vec_ub[0].set_as(scalar_zero)

    # init zero tensor
    data_zero = tik_instance.Tensor("float16",
                                    (CONFIG_EIGHT, CONFIG_SIXTEEN),
                                    name="data_zero",
                                    scope=tik.scope_ubuf)
    tik_instance.vector_dup(CONFIG_MASK,
                            data_zero,
                            0,
                            CONFIG_ONE,
                            CONFIG_ONE,
                            CONFIG_EIGHT)

    # init selected number of proposals
    output_nms_num = tik_instance.Scalar(dtype="int32", init_value=0)

    # init v200 reduce param
    nms_tensor_pattern = tik_instance.Tensor(dtype="uint16",
                                             shape=(CONFIG_EIGHT, ),
                                             name="nms_tensor_pattern",
                                             scope=tik.scope_ubuf)

    # init reduce num
    num_nms = tik_instance.Scalar(dtype="uint32")

    # init middle selected coord
    ori_coord_reduce = tik_instance.Tensor("float16",
                                           (CONFIG_FOUR, MATRIX - CONFIG_DATA_ALIGN),
                                           name="ori_coord",
                                           scope=tik.scope_ubuf)
    tik_instance.vector_dup(CONFIG_MASK,
                            ori_coord_reduce,
                            0,
                            CONFIG_SEVEN,
                            CONFIG_ONE,
                            CONFIG_EIGHT)

    # init ori coord
    ori_coord = tik_instance.Tensor("float16",
                                    (CONFIG_FOUR, CONFIG_MASK),
                                    name="ori_coord",
                                    scope=tik.scope_ubuf)

    # init zoom coord
    zoom_coord = tik_instance.Tensor("float16",
                                     (CONFIG_EIGHT, CONFIG_MASK),
                                     name="zoom_coord",
                                     scope=tik.scope_ubuf)
    zoom_coord1 = tik_instance.Tensor("float16",
                                      (CONFIG_FOUR, CONFIG_MASK),
                                      name="zoom_coord1",
                                      scope=tik.scope_ubuf)

    # init reduce zoom coord
    zoom_coord_reduce = tik_instance.Tensor("float16",
                                            (CONFIG_FOUR, CONFIG_MASK),
                                            name="zoom_coord",
                                            scope=tik.scope_ubuf)

    # handle 128 proposlas every time
    with tik_instance.for_range(0, ceil_div(actual_input_nms_num, CONFIG_MASK)) as burst_index:
        # if selected num larger than except num; break
        with tik_instance.if_scope(output_nms_num < post_nms_num):
            # ********************** 1 zoom for the original data **********************
            tik_instance.data_move(data_x_ub_burst,
                                   data_x_gm[burst_index * CONFIG_MASK * CONFIG_EIGHT],
                                   0,
                                   CONFIG_ONE,
                                   CONFIG_DATA_TRANS,
                                   0, 0)

            # Extract original coordinates
            with tik_instance.for_range(0, CONFIG_FOUR) as i:
                tik_instance.vextract(ori_coord[CONFIG_MASK * i], data_x_ub_burst, CONFIG_EIGHT, i)

            # Coordinate multiplied by down_factor to prevent out of range
            tik_instance.vmuls(CONFIG_MASK,
                               zoom_coord,
                               ori_coord,
                               down_factor,
                               CONFIG_FOUR,
                               CONFIG_ONE,
                               CONFIG_ONE,
                               CONFIG_EIGHT,
                               CONFIG_EIGHT)

            # add 1 for x1 and y1 because nms operate would reduces 1
            tik_instance.vadds(CONFIG_MASK,
                               zoom_coord1,
                               zoom_coord,
                               CONFIG_ONE,
                               CONFIG_TWO,
                               CONFIG_ONE,
                               CONFIG_ONE,
                               CONFIG_EIGHT,
                               CONFIG_EIGHT)
            tik_instance.vadds(CONFIG_MASK,
                               zoom_coord1[CONFIG_TWO, 0],
                               zoom_coord[CONFIG_TWO, 0],
                               0,
                               CONFIG_TWO,
                               CONFIG_ONE,
                               CONFIG_ONE,
                               CONFIG_EIGHT,
                               CONFIG_EIGHT)

            # Compose new proposals
            with tik_instance.for_range(0, CONFIG_FOUR) as i:
                tik_instance.vconcat(data_x_ub_burst_reduce,
                                     zoom_coord1[CONFIG_MASK * i],
                                     CONFIG_EIGHT, i)

            # ********************** 2 start to nms operate **********************

            # calculate the area of reduced-proposal
            tik_instance.vrpac(temp_area_ub, data_x_ub_burst_reduce, CONFIG_EIGHT)

            length = tik_instance.Scalar(dtype="uint16")
            length.set_as(ceil_div(output_nms_num, CONFIG_SIXTEEN) * CONFIG_SIXTEEN)

            # init the sup_vec
            tik_instance.vector_dup(CONFIG_MASK,
                                    temp_sup_vec_ub[0],
                                    CONFIG_ONE,
                                    CONFIG_ONE,
                                    CONFIG_ONE,
                                    CONFIG_EIGHT)

            with tik_instance.for_range(0, CONFIG_EIGHT) as i:
                length.set_as(length + CONFIG_SIXTEEN)
                # calculate iou area
                tik_instance.viou(iou_ub,
                                  temp_reduced_proposals_ub,
                                  data_x_ub_burst_reduce[i * CONFIG_MASK],
                                  ceil_div(output_nms_num, CONFIG_SIXTEEN))

                tik_instance.viou(iou_ub[ceil_div(output_nms_num, CONFIG_SIXTEEN), 0, 0],
                                  data_x_ub_burst_reduce,
                                  data_x_ub_burst_reduce[i * CONFIG_MASK],
                                  i + CONFIG_ONE)

                # calculate aadd area
                tik_instance.vaadd(join_ub,
                                   selected_area_ub,
                                   temp_area_ub[i, 0],
                                   ceil_div(output_nms_num, CONFIG_SIXTEEN))

                tik_instance.vaadd(join_ub[ceil_div(output_nms_num, CONFIG_SIXTEEN), 0, 0],
                                   temp_area_ub,
                                   temp_area_ub[i, 0],
                                   i + CONFIG_ONE)

                # aadd area muls nms_threshold_new
                tik_instance.vmuls(CONFIG_MASK, join_ub1,
                                   join_ub, nms_threshold_new,
                                   ceil_div(length, CONFIG_EIGHT),
                                   CONFIG_ONE, CONFIG_ONE, CONFIG_EIGHT, CONFIG_EIGHT)

                # compare and generate suppression matrix
                tik_instance.vcmpv_gt(sup_matrix_ub,
                                      iou_ub, join_ub1,
                                      ceil_div(length, CONFIG_EIGHT),
                                      CONFIG_ONE, CONFIG_ONE, CONFIG_EIGHT, CONFIG_EIGHT)

                # generate rpn_cor_ir
                rpn_cor_ir = tik_instance.set_rpn_cor_ir(0)

                # non-diagonal
                rpn_cor_ir = tik_instance.rpn_cor(sup_matrix_ub,
                                                  sup_vec_ub,
                                                  CONFIG_ONE,
                                                  CONFIG_ONE,
                                                  ceil_div(output_nms_num, CONFIG_SIXTEEN))
                with tik_instance.if_scope(i > 0):
                    rpn_cor_ir = tik_instance.rpn_cor(
                        sup_matrix_ub[ceil_div(output_nms_num, CONFIG_SIXTEEN) * CONFIG_SIXTEEN],
                        temp_sup_vec_ub,
                        CONFIG_ONE,
                        CONFIG_ONE, i)
                # get final sup_vec
                tik_instance.rpn_cor_diag(temp_sup_vec_ub[i * CONFIG_SIXTEEN],
                                          sup_matrix_ub[length - CONFIG_SIXTEEN], rpn_cor_ir)

            # v100 branch
            if tik.Dprofile().get_product_name() not in IF_USE_V200:
                # 每次对128 proposla 得到的抑制矩阵 0表示不过滤 1表示需要过滤
                with tik_instance.for_range(0, CONFIG_MASK) as i:
                    with tik_instance.if_scope(temp_sup_vec_ub[i] == 0):
                        # 如果抑制矩阵为0, 就进行以下处理
                        for j in range(CONFIG_FOUR):
                            # 把这个proposal对应的原坐标保存起来
                            selected_reduced_coord_ub[
                                output_nms_num * CONFIG_FOUR + j].set_as(ori_coord[i +
                                                                                   CONFIG_MASK * j])
                            # 把这个proposal保存起来供下次nms使用
                            temp_reduced_proposals_ub[
                                output_nms_num * CONFIG_EIGHT + j].set_as(
                                    data_x_ub_burst_reduce[i * CONFIG_EIGHT + j])
                        # 把这个proposal对应的面积保存起来供下次nms使用
                        selected_area_ub[output_nms_num].set_as(temp_area_ub[i])
                        # sup_vec_ub set as 0
                        sup_vec_ub[output_nms_num].set_as(scalar_zero)
                        # nms_num add 1
                        output_nms_num.set_as(output_nms_num + CONFIG_ONE)
            # v200 branch
            else:
                # get the mask tensor of temp_sup_vec_ub
                temp_tensor = temp_sup_vec_ub.reinterpret_cast_to("float16")
                cmpmask = tik_instance.vcmp_eq(CONFIG_MASK,
                                               temp_tensor,
                                               data_zero,
                                               CONFIG_ONE,
                                               CONFIG_ONE)

                tik_instance.mov_cmpmask_to_tensor(
                    nms_tensor_pattern.reinterpret_cast_to("uint64"),
                    cmpmask)

                # 把这个proposal对应的原坐标保存起来
                with tik_instance.for_range(0, CONFIG_FOUR) as i:
                    tik_instance.vreduce(CONFIG_MASK,
                                         ori_coord_reduce[i, output_nms_num],
                                         ori_coord[i, 0],
                                         nms_tensor_pattern,
                                         CONFIG_ONE,
                                         CONFIG_ONE,
                                         CONFIG_EIGHT,
                                         0,
                                         0,
                                         num_nms,
                                         "counter")
                # sup_vec_ub set as 0
                tik_instance.vector_dup(CONFIG_SIXTEEN,
                                        sup_vec_ub[output_nms_num],
                                        0, ceil_div(num_nms, CONFIG_SIXTEEN),
                                        CONFIG_ONE, CONFIG_ONE)
                with tik_instance.for_range(ceil_div(num_nms, CONFIG_SIXTEEN), num_nms) as j:
                    sup_vec_ub[output_nms_num + j].set_as(scalar_zero)
                # 把这个proposal对应的面积保存起来供下次nms使用
                tik_instance.vreduce(CONFIG_MASK,
                                     selected_area_ub[output_nms_num],
                                     temp_area_ub,
                                     nms_tensor_pattern,
                                     CONFIG_ONE,
                                     CONFIG_ONE,
                                     CONFIG_EIGHT,
                                     0,
                                     0,
                                     None,
                                     "counter")
                output_nms_num.set_as(output_nms_num + num_nms)

                with tik_instance.if_scope(output_nms_num < post_nms_num):
                    # 把这个proposal保存起来供下次nms使用
                    tik_instance.vector_dup(CONFIG_MASK,
                                            zoom_coord_reduce,
                                            0,
                                            CONFIG_FOUR,
                                            CONFIG_ONE,
                                            CONFIG_EIGHT)
                    with tik_instance.for_range(0, CONFIG_FOUR) as i:
                        tik_instance.vreduce(CONFIG_MASK,
                                             zoom_coord_reduce[i, 0],
                                             zoom_coord1[i, 0],
                                             nms_tensor_pattern,
                                             CONFIG_ONE,
                                             CONFIG_ONE,
                                             CONFIG_EIGHT,
                                             0,
                                             0,
                                             None,
                                             "counter")

                    with tik_instance.for_range(0, CONFIG_FOUR) as i:
                        tik_instance.vconcat(temp_reduced_proposals_ub,
                                             zoom_coord_reduce[i, 0],
                                             CONFIG_EIGHT,
                                             i)

    if tik.Dprofile().get_product_name() in IF_USE_V200:
        data_y_last = tik_instance.Tensor("float16",
                                          [post_nms_num, CONFIG_FOUR],
                                          name="data_y_last",
                                          scope=tik.scope_ubuf)
        # extract front 4*post_nms_num from 4*224
        tik_instance.vadds(post_nms_num,
                           data_y_last,
                           ori_coord_reduce, 0, CONFIG_FOUR, CONFIG_ONE, CONFIG_ONE,
                           post_nms_num//CONFIG_SIXTEEN,
                           CONFIG_SIXTEEN - CONFIG_TWO)
        # transpose 4*post_nms_num to post_nms_num*4
        tik_instance.v4dtrans(True,
                              selected_reduced_coord_ub,
                              data_y_last,
                              post_nms_num,
                              CONFIG_FOUR)
    # copy data to gm
    tik_instance.data_move(data_y_gm,
                           selected_reduced_coord_ub,
                           0,
                           CONFIG_ONE,
                           post_nms_num * CONFIG_FOUR // CONFIG_SIXTEEN,
                           0, 0)


def call_nms_v200(tik_instance, data_tensor, input_param):
    """
    call the nms

    param tik_instance:

    param data_tensor:
    data_x_gm = data_tensor[0]
    data_y_gm = data_tensor[1]

    param input_param:
    actual_input_nms_num = input_param[0]
    nms_threshold = input_param[1]
    img_size = input_param[2]
    post_nms_num = input_param[3]
    """

    with tik_instance.new_stmt_scope():
        nms_local(tik_instance, data_tensor, input_param)


def rpn_proposals_d_compute_v200(input_dict,
                                 input_param,
                                 kernel_name):
    """
    calculating data

    Parameters
    ----------
    input_dict : a list of input dict
      rois, cls_bg_prob, sorted_box

    input_param : a list of attr
    img_size,  score_threshold,  k,  min_size,   nms_threshold,
    score_filter,     box_filter,     score_sigmoid,

    kernel_name : str
        kernel name, default value is "generate_rpn_proposals"

    Returns
    -------
    tik_instance
    """

    tik_instance = tik.Tik(tik.Dprofile(), disable_debug=True)

    data_type = input_dict[0].get("dtype")
    shape_in = input_dict[0].get("shape")[0]
    shape_out = input_dict[CONFIG_TWO].get("shape")[0]

    # init the GM tensors
    data_gm = InitGmTensorV200(tik_instance,
                               (shape_in, shape_out, input_param[CONFIG_TWO], data_type))

    # perform the score filter, results saved in data_gm(proposal_cb, proposal_gm, num_post_score,
    #                              num_total_gm, num_total_cb)
    with tik_instance.new_stmt_scope():
        score_filter_processing(tik_instance, data_gm, shape_in, input_param[CONFIG_ONE])

    # perform the topk, results save in data_gm(proposal_post_topk, num_post_score)
    with tik_instance.new_stmt_scope():
        call_topk_sort_v200(tik_instance,
                            (data_gm.proposal_post_topk, data_gm.proposal_cb,
                             data_gm.mem_swap, data_gm.proposal_gm, data_gm.mem_swap_gm),
                            (data_gm.flag_cb, data_gm.num_total_cb, data_gm.num_total_gm),
                            (input_param[CONFIG_TWO], input_param[CONFIG_ONE], data_type))
    tik_scalar_min(tik_instance, data_gm.num_post_score,
                   input_param[CONFIG_TWO], data_gm.num_post_score)

    # call clip_box and size filter, results save in data.gm(proposal_post_topk, num_post_score)
    with tik_instance.if_scope(data_gm.num_post_score > 0):
        with tik_instance.new_stmt_scope():
            clip_size_filter(tik_instance,
                             (data_gm.proposal_post_topk, data_gm.proposal_post_topk),
                             (input_param[CONFIG_TWO], data_gm.num_post_score),
                             (input_param[0], input_param[CONFIG_THREE],
                              data_type, input_param[CONFIG_SIX]))

        # perform the NMS
        call_nms_v200(tik_instance,
                      (data_gm.proposal_post_topk, data_gm.box_gm),
                      (data_gm.num_post_score, input_param[CONFIG_FOUR], input_param[0], shape_out))
    with tik_instance.else_scope():
        temp_box = tik_instance.Tensor("float16",
                                       (ceil_div(shape_out, CONFIG_DATA_ALIGN), CONFIG_MASK),
                                       name="temp_box",
                                       scope=tik.scope_ubuf)
        if ceil_div(shape_out, CONFIG_DATA_ALIGN) > MAX_REP_TIME:
            tik_instance.vector_dup(CONFIG_MASK,
                                    temp_box,
                                    0,
                                    ceil_div(shape_out, CONFIG_DATA_ALIGN),
                                    CONFIG_ONE,
                                    CONFIG_EIGHT)
        else:
            tik_instance.vector_dup(CONFIG_MASK,
                                    temp_box,
                                    0,
                                    ceil_div(shape_out, CONFIG_DATA_ALIGN),
                                    CONFIG_ONE,
                                    CONFIG_EIGHT)
        tik_instance.data_move(data_gm.box_gm,
                               temp_box,
                               0,
                               CONFIG_ONE,
                               ceil_div(shape_out, CONFIG_FOUR),
                               0, 0)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[data_gm.rois_gm, data_gm.prob_gm],
                          outputs=[data_gm.box_gm])

    return tik_instance
