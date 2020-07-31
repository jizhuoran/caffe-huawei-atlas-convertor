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


rpn_proposals_post_processing
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
        num_topk = input_para[0]
        core_num = input_para[CONFIG_ONE]
        data_type = input_para[CONFIG_TWO]
        shape_out = input_para[CONFIG_THREE]

        # input tensor
        self.proposal_gm = tik_instance.Tensor(data_type,
                                               (core_num * (num_topk + CONFIG_TWO), CONFIG_EIGHT),
                                               name="proposal_gm",
                                               scope=tik.scope_gm)
        self.proposal_num = tik_instance.Tensor("uint32",
                                                (core_num, CONFIG_EIGHT),
                                                name="proposal_num",
                                                scope=tik.scope_gm)

        # output tensors in DDR
        self.box_gm = tik_instance.Tensor(data_type,
                                          (shape_out, CONFIG_FOUR),
                                          name="box_gm",
                                          scope=tik.scope_gm)

        # cache memory in L1 buffer
        # for topk
        self.mem_swap = tik_instance.Tensor(data_type,
                                            (num_topk * CONFIG_TWO + CONFIG_FOUR, CONFIG_EIGHT),
                                            name="men_swap",
                                            scope=tik.scope_cbuf)
        # Tensors after the first topk
        self.proposal_post_topk = tik_instance.Tensor(data_type,
                                                      (num_topk + CONFIG_FOUR,
                                                       CONFIG_EIGHT),
                                                      name="proposal_post_topk",
                                                      scope=tik.scope_cbuf)
        self.actual_proposal = tik_instance.Scalar("uint32", name="actual_proposal", init_value=0)

    def set_proposal_num(self, tik_instance, num_in, index):
        """
        set the num_post_score
        param num_in:
        return:
        """

        # self.proposal_num[index, 0].set_as(num_in)
        tik_instance.vector_dup(CONFIG_EIGHT,
                                self.proposal_num[index, 0],
                                num_in,
                                CONFIG_ONE,
                                CONFIG_ONE,
                                CONFIG_EIGHT)

    def set_proposal_gm(self, num_in):
        """
        set num_total_cb
        param num_in:
        return:
        """
        self.proposal_gm = num_in


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


# =========> topk  here
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

def topk_sort(tik_instance, data_gm, topk_k, num_list, score_threshold, aicore_num):
    """

    :param tik_instance:
    :param data_gm:
    :param topk_k:
    :param num_list:
    :param score_threshold:
    :param aicore_num:
    :return:
    """

    n_required = tik_instance.Scalar(dtype="uint32", name="n_required", init_value=0)
    num_onelist = tik_instance.Scalar(dtype="uint32", name="num_onelist", init_value=0)

    src_pos_ = [tik_instance.Scalar(dtype="uint32") for i in range(CONFIG_FOUR)]
    for t_cyc in range(CONFIG_FOUR):
        src_pos_[t_cyc].set_as((topk_k + CONFIG_TWO) * t_cyc)

    # combine the sorted lists
    if aicore_num > CONFIG_FOUR:
        # first combine four list
        for t_cyc in range(CONFIG_FOUR):
            n_required.set_as(n_required + num_list[t_cyc])
        tik_scalar_min(tik_instance, topk_k, n_required, n_required)

        with tik_instance.new_stmt_scope():
            tik_topk_vms4(tik_instance,
                          data_gm.mem_swap,
                          0,
                          (data_gm.proposal_gm, data_gm.proposal_gm,
                           data_gm.proposal_gm, data_gm.proposal_gm),
                          (src_pos_[0], src_pos_[CONFIG_ONE],
                           src_pos_[CONFIG_TWO], src_pos_[CONFIG_THREE]),
                          (num_list[0], num_list[CONFIG_ONE],
                           num_list[CONFIG_TWO], num_list[CONFIG_THREE]),
                          CONFIG_SIXTEEN - CONFIG_ONE,
                          n_required,
                          score_threshold)
            num_onelist.set_as(n_required)

        # whether rep is needed
        max_rep = ceil_div(aicore_num - CONFIG_FOUR, CONFIG_THREE) - CONFIG_ONE
        left_rep = (aicore_num - CONFIG_FOUR) % CONFIG_THREE
        # need repeat
        if max_rep > 0:
            for t_cyc in range(max_rep):
                src_pos_[CONFIG_ONE].set_as(src_pos_[CONFIG_ONE] +
                                            (topk_k + CONFIG_TWO) * CONFIG_THREE)
                src_pos_[CONFIG_TWO].set_as(src_pos_[CONFIG_TWO] +
                                            (topk_k + CONFIG_TWO) * CONFIG_THREE)
                src_pos_[CONFIG_THREE].set_as(src_pos_[CONFIG_THREE] +
                                              (topk_k + CONFIG_TWO) * CONFIG_THREE)

                dst_offset1 = (t_cyc % CONFIG_TWO) * topk_k
                dst_offset2 = ((t_cyc + CONFIG_ONE) % CONFIG_TWO) * topk_k

                index_scalar = t_cyc * CONFIG_THREE + CONFIG_FOUR
                n_required.set_as(n_required + num_list[index_scalar])
                index_scalar = t_cyc * CONFIG_THREE + CONFIG_FIVE
                n_required.set_as(n_required + num_list[index_scalar])
                index_scalar = t_cyc * CONFIG_THREE + CONFIG_SIX
                n_required.set_as(n_required + num_list[index_scalar])

                tik_scalar_min(tik_instance, topk_k, n_required, n_required)

                with tik_instance.new_stmt_scope():
                    tik_topk_vms4(tik_instance,
                                  data_gm.mem_swap,
                                  dst_offset2,
                                  (data_gm.mem_swap, data_gm.proposal_gm,
                                   data_gm.proposal_gm, data_gm.proposal_gm),
                                  (dst_offset1, src_pos_[CONFIG_ONE],
                                   src_pos_[CONFIG_TWO], src_pos_[CONFIG_THREE]),
                                  (num_onelist, num_list[t_cyc * CONFIG_THREE + CONFIG_FOUR],
                                   num_list[t_cyc * CONFIG_THREE + CONFIG_FIVE],
                                   num_list[t_cyc * CONFIG_THREE + CONFIG_SIX]),
                                  CONFIG_SIXTEEN - CONFIG_ONE,
                                  n_required,
                                  score_threshold)
                    num_onelist.set_as(n_required)

        dst_offset1 = topk_k * (max_rep % CONFIG_TWO)
        src_pos_[CONFIG_ONE].set_as(src_pos_[CONFIG_ONE] +
                                    (topk_k + CONFIG_TWO) * CONFIG_THREE)
        src_pos_[CONFIG_TWO].set_as(src_pos_[CONFIG_TWO] +
                                    (topk_k + CONFIG_TWO) * CONFIG_THREE)
        src_pos_[CONFIG_THREE].set_as(src_pos_[CONFIG_THREE] +
                                      (topk_k + CONFIG_TWO) * CONFIG_THREE)

        if left_rep == CONFIG_ONE:
            index_scalar = t_cyc * CONFIG_THREE + CONFIG_FOUR
            n_required.set_as(n_required + num_list[index_scalar])

            tik_scalar_min(tik_instance, topk_k, n_required, n_required)
            with tik_instance.new_stmt_scope():
                tik_topk_vms4(tik_instance,
                              data_gm.proposal_post_topk,
                              0,
                              (data_gm.mem_swap, data_gm.proposal_gm, None, None),
                              (dst_offset1,
                               src_pos_[CONFIG_ONE],
                               src_pos_[CONFIG_TWO],
                               src_pos_[CONFIG_THREE]),
                              (num_onelist, num_list[max_rep * CONFIG_THREE + CONFIG_FOUR],
                               0, 0),
                              CONFIG_THREE,
                              n_required,
                              score_threshold)
                data_gm.actual_proposal.set_as(n_required)

        if left_rep == 0:
            index_scalar = t_cyc * CONFIG_THREE + CONFIG_FOUR
            n_required.set_as(n_required + num_list[index_scalar])
            index_scalar = t_cyc * CONFIG_THREE + CONFIG_FIVE
            n_required.set_as(n_required + num_list[index_scalar])
            index_scalar = t_cyc * CONFIG_THREE + CONFIG_SIX
            n_required.set_as(n_required + num_list[index_scalar])

            tik_scalar_min(tik_instance, topk_k, n_required, n_required)
            with tik_instance.new_stmt_scope():
                tik_topk_vms4(tik_instance,
                              data_gm.proposal_post_topk,
                              0,
                              (data_gm.mem_swap, data_gm.proposal_gm,
                               data_gm.proposal_gm, data_gm.proposal_gm),
                              (dst_offset1, src_pos_[CONFIG_ONE],
                               src_pos_[CONFIG_TWO], src_pos_[CONFIG_THREE]),
                              (num_onelist, num_list[max_rep * CONFIG_THREE + CONFIG_FOUR],
                               num_list[max_rep * CONFIG_THREE + CONFIG_FIVE],
                               num_list[max_rep * CONFIG_THREE + CONFIG_SIX]),
                              CONFIG_SIXTEEN - CONFIG_ONE,
                              n_required,
                              score_threshold)
                data_gm.actual_proposal.set_as(n_required)

        if left_rep == CONFIG_TWO:
            index_scalar = t_cyc * CONFIG_THREE + CONFIG_FOUR
            n_required.set_as(n_required + num_list[index_scalar])
            index_scalar = t_cyc * CONFIG_THREE + CONFIG_FIVE
            n_required.set_as(n_required + num_list[index_scalar])

            tik_scalar_min(tik_instance, topk_k, n_required, n_required)
            with tik_instance.new_stmt_scope():
                tik_topk_vms4(tik_instance,
                              data_gm.proposal_post_topk,
                              0,
                              (data_gm.mem_swap, data_gm.proposal_gm, data_gm.proposal_gm, None),
                              (dst_offset1,
                               src_pos_[CONFIG_ONE],
                               src_pos_[CONFIG_TWO],
                               src_pos_[CONFIG_THREE]),
                              (num_onelist, num_list[max_rep * CONFIG_THREE + CONFIG_FOUR],
                               num_list[max_rep * CONFIG_THREE + CONFIG_FIVE], 0),
                              CONFIG_EIGHT - CONFIG_ONE,
                              n_required,
                              score_threshold)
                data_gm.actual_proposal.set_as(n_required)

    else:
        with tik_instance.new_stmt_scope():
            if aicore_num == CONFIG_FOUR:
                for t_cyc in range(CONFIG_FOUR):
                    n_required.set_as(n_required + num_list[t_cyc])
                tik_scalar_min(tik_instance, topk_k, n_required, n_required)
                tik_topk_vms4(tik_instance,
                              data_gm.proposal_post_topk,
                              0,
                              (data_gm.proposal_gm, data_gm.proposal_gm,
                               data_gm.proposal_gm, data_gm.proposal_gm),
                              (src_pos_[0], src_pos_[CONFIG_ONE],
                               src_pos_[CONFIG_TWO], src_pos_[CONFIG_THREE]),
                              (num_list[0], num_list[CONFIG_ONE],
                               num_list[CONFIG_TWO], num_list[CONFIG_THREE]),
                              CONFIG_SIXTEEN - CONFIG_ONE,
                              n_required,
                              score_threshold)
                data_gm.actual_proposal.set_as(n_required)

            if aicore_num == CONFIG_THREE:
                for t_cyc in range(CONFIG_THREE):
                    n_required.set_as(n_required + num_list[t_cyc])
                tik_scalar_min(tik_instance, topk_k, n_required, n_required)
                tik_topk_vms4(tik_instance,
                              data_gm.proposal_post_topk,
                              0,
                              (data_gm.proposal_gm, data_gm.proposal_gm, data_gm.proposal_gm, None),
                              (src_pos_[0], src_pos_[CONFIG_ONE],
                               src_pos_[CONFIG_TWO], src_pos_[CONFIG_TWO]),
                              (num_list[0], num_list[CONFIG_ONE], num_list[CONFIG_TWO], 0),
                              CONFIG_EIGHT - CONFIG_ONE,
                              n_required,
                              score_threshold)
                data_gm.actual_proposal.set_as(n_required)

            if aicore_num == CONFIG_TWO:
                for t_cyc in range(CONFIG_TWO):
                    n_required.set_as(n_required + num_list[t_cyc])
                tik_scalar_min(tik_instance, topk_k, n_required, n_required)
                tik_topk_vms4(tik_instance,
                              data_gm.proposal_post_topk,
                              0,
                              (data_gm.proposal_gm, data_gm.proposal_gm, None, None),
                              (src_pos_[0], src_pos_[CONFIG_ONE],
                               src_pos_[CONFIG_ONE], src_pos_[CONFIG_ONE]),
                              (num_list[0], num_list[CONFIG_ONE], 0, 0),
                              CONFIG_THREE,
                              n_required,
                              score_threshold)
                data_gm.actual_proposal.set_as(n_required)

            if aicore_num == CONFIG_ONE:
                data_gm.actual_proposal.set_as(num_list[0])
                with tik_instance.new_stmt_scope():
                    temp_proposal = tik_instance.Tensor("float16",
                                                        (topk_k, CONFIG_EIGHT),
                                                        name="temp_proposal",
                                                        scope=tik.scope_ubuf)
                    with tik_instance.if_scope(num_list[0] > 0):
                        tik_instance.data_move(temp_proposal,
                                               data_gm.proposal_gm,
                                               0,
                                               CONFIG_ONE,
                                               ceil_div(num_list[0], CONFIG_TWO),
                                               0, 0)
                        tik_instance.data_move(data_gm.proposal_post_topk,
                                               temp_proposal,
                                               0,
                                               CONFIG_ONE,
                                               ceil_div(num_list[0], CONFIG_TWO),
                                               0, 0)


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
                            temp_reduced_proposals_ub[output_nms_num * CONFIG_EIGHT + j].set_as(
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


def rpn_proposal_calculate(param_list, kernel_name):
    """
    the main compute function
    param param_list:
    0	img_size: listfloat, size of image, [h, w]
    1	score_threshold : float, init=0,   score filter threshold
    2	k: the topk, init 6000
    3	min_size: parameter for size filter, 0
    4	nms_threshold: float,  nms threshold
    5	post_nms_num: num of proposals output after NMS

    6	score_filter: bool,  True
    7	core_max_num : actual core num.
    8   dtype

    param kernel_name:
    return:
    """

    score_threshold = param_list[CONFIG_ONE]
    num_topk = param_list[CONFIG_TWO]
    core_num = param_list[CONFIG_SEVEN]
    data_type = param_list[CONFIG_EIGHT]
    shape_out = param_list[CONFIG_FIVE]

    tik_instance = tik.Tik(tik.Dprofile(), disable_debug=True)

    data_gm = InitGmTensor(tik_instance, (num_topk, core_num, data_type, shape_out))
    num_list = [tik_instance.Scalar(dtype="uint32") for i in range(core_num)]

    with tik_instance.new_stmt_scope():
        proposal_num_ub = tik_instance.Tensor("uint32",
                                              (core_num, CONFIG_EIGHT),
                                              name="proposal_num_ub",
                                              scope=tik.scope_ubuf)
        tik_instance.data_move(proposal_num_ub,
                               data_gm.proposal_num,
                               0,
                               CONFIG_ONE,
                               core_num,
                               0, 0)
        for t_cyc in range(core_num):
            num_list[t_cyc].set_as(proposal_num_ub[t_cyc, 0])

    # topk sort
    topk_sort(tik_instance, data_gm, num_topk,
              num_list, score_threshold, core_num)

    # call clip_box and size filter, results save in data.gm(proposal_post_topk, num_post_score)
    with tik_instance.if_scope(data_gm.actual_proposal > 0):
        with tik_instance.new_stmt_scope():
            clip_size_filter(tik_instance,
                             (data_gm.proposal_post_topk, data_gm.proposal_post_topk),
                             (num_topk, data_gm.actual_proposal),
                             (param_list[0], param_list[CONFIG_THREE],
                              data_type, param_list[CONFIG_SIX]))

        # perform the NMS
        call_nms_v200(tik_instance,
                      (data_gm.proposal_post_topk, data_gm.box_gm),
                      (data_gm.actual_proposal, param_list[CONFIG_FOUR], param_list[0], shape_out))
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
                          inputs=[data_gm.proposal_gm, data_gm.proposal_num],
                          outputs=[data_gm.box_gm])
    print("============> IR line num >======== {}".format(tik_instance.get_ir_num()))

    return tik_instance


def check_input_param(input_para, kernel_name):
    """
    check other input of generate_rpn_proposals()
    img_size,  [h, w]
        img_height: [0, 2000]
        img_width: [0, 3000]
    score_threshold (-inf, inf)
    k: [0, 6000]
    min_size [0, 2000]
    nms_threshold: [0, 1]

    return: None
    """

    if len(input_para[0]) != CONFIG_TWO:
        raise RuntimeError("The length of img_size should be 2!")

    img_height, img_width = input_para[0]

    k = input_para[CONFIG_TWO]
    min_size = input_para[CONFIG_THREE]
    nms_threshold = input_para[CONFIG_FOUR]

    def _check_range_of_input(input_x, min_limit, max_limit, input_name):
        """
        internal function
        check whether min_limit<=input_para<=max_limit
        """
        if input_x < min_limit or input_x > max_limit:
            raise RuntimeError("The %s should be in [%d, %d]!" % (input_name, min_limit, max_limit))

    def _check_input_dtype(input_x, input_name):
        """
        internal function, check whether the dtype is right
        param input_x:
        param check_list:
        param input_name:
        return: None
        """
        if not isinstance(input_x, int):
            raise RuntimeError("%s should be type of float or int!" % input_name)

    _check_input_dtype(img_height, "img_height")
    _check_input_dtype(img_width, "img_width")

    _check_range_of_input(nms_threshold, 0, CONFIG_ONE, "nms_threshold")
    _check_range_of_input(img_height, 0, MAX_HEIGHT, "img_height")
    _check_range_of_input(img_width, 0, MAX_WIDTH, "img_width")
    _check_range_of_input(min_size, 0, MAX_HEIGHT, "min_size")
    _check_range_of_input(k, 0, MAX_TOPK, "k")

    if k % CONFIG_SIXTEEN:
        raise RuntimeError("K should be times of 16!")

    if min_size > min(img_height, img_width):
        raise RuntimeError("min_size should be less than min(img_height, img_width)!")

    util.check_kernel_name(kernel_name)


def check_input_dict(dict_list, param_list):
    """
    check the input dict of generate_rpn_proposals()
    Parameters
    ----------
    dict_list:
      sorted_propsoal : dict
        shape and dtype of input boxes
      proposal_num : dict
        shape and dtype of input probobilities
      sorted_box: : dict
        shape and dtype of output sorted boxes

    param_list: a list of param
         0 score_threshold
         1 k
         2 core_max_num
         3 post_nms_num: Int      num of proposals after NMS
     Returns: None
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
    _check_input_type_dict(dict_list[0], input_key, "sorted_proposal")
    _check_input_type_dict(dict_list[CONFIG_ONE], input_key, "proposal_num")
    _check_input_type_dict(dict_list[CONFIG_TWO], input_key, "sorted_box")

    # check the dtype
    util.check_dtype_rule(dict_list[0].get("dtype"), ("float16", ))
    util.check_dtype_rule(dict_list[CONFIG_ONE].get("dtype"), ("uint32", ))
    util.check_dtype_rule(dict_list[CONFIG_TWO].get("dtype"), ("float16", ))

    # get the parameters from dicts
    input_proposal_shape = dict_list[0].get("shape")
    input_proposal_num_shape = dict_list[CONFIG_ONE].get("shape")
    output_box_shape = dict_list[CONFIG_TWO].get("shape")

    # check the shape
    util.check_shape_rule(input_proposal_shape,
                          min_dim=CONFIG_TWO,
                          max_dim=CONFIG_TWO)
    util.check_shape_rule(input_proposal_num_shape,
                          min_dim=CONFIG_ONE,
                          max_dim=CONFIG_TWO)
    util.check_shape_rule(output_box_shape,
                          min_dim=CONFIG_TWO,
                          max_dim=CONFIG_TWO)

    # Check the size of the input/output shape
    _check_shape_size_limit(input_proposal_shape,
                            "sorted_propsoal",
                            shape_para=CONFIG_EIGHT,
                            output_flag=True)
    _check_shape_size_limit(input_proposal_num_shape,
                            "proposal_num",
                            shape_para=CONFIG_EIGHT,
                            output_flag=False)
    _check_shape_size_limit(output_box_shape,
                            "sorted_box",
                            shape_para=CONFIG_FOUR,
                            output_flag=True)

    if input_proposal_shape[0] != (param_list[CONFIG_ONE] + CONFIG_TWO) * param_list[CONFIG_TWO]:
        raise RuntimeError("sorted_proposal shape should be consistent with"
                           " k and core_max_num!")

    if input_proposal_num_shape[0] != param_list[CONFIG_TWO]:
        raise RuntimeError("n dimension of inputs proposal_num should"
                           " be consistent with max_core_num")

    if param_list[CONFIG_THREE] != output_box_shape[0]:
        raise RuntimeError("post_nms_num should be consistent with"
                           " n dimension of inputs sorted_box")


@util.check_input_type(dict, dict, dict,
                       (tuple, list), (float, int), int,
                       (float, int), (float, int), int,
                       bool, int, str)
def rpn_proposal_post_processing(sorted_proposal, proposal_num, sorted_box,
                                 img_size, score_threshold, k, min_size,
                                 nms_threshold, post_nms_num,
                                 box_filter=True, core_max_num=CONFIG_EIGHT,
                                 kernel_name="rpn_proposals_post_processing"):
    """
    the entry function of rpn_proposals_post_processing
    Parameters
    ----------
    sorted_proposal : dict
        shape and dtype of input boxes
    proposal_num : dict
        shape and dtype of input probobilities
    sorted_box: : dict
        shape and dtype of output sorted boxes

    img_size: listfloat, size of image, [h, w]
    score_threshold : float, init=0,   score filter threshold
    k: the topk, init 6000
    min_size: parameter for size filter, 0
    nms_threshold: float,  nms threshold
    post_nms_num: num of proposals output after NMS

    score_filter: bool,  True
    core_max_num : max core num.
    kernel_name : str
        kernel name, default value is "generate_rpn_proposals"
    Returns
    -------
    None
    """

    check_input_dict((sorted_proposal, proposal_num, sorted_box),
                     (score_threshold, k, core_max_num, post_nms_num))

    check_input_param((img_size, score_threshold, k, min_size,
                       nms_threshold), kernel_name)

    aicore_num = tik.Dprofile().get_aicore_num()
    if aicore_num > core_max_num:
        aicore_num = core_max_num

    tik_instance = rpn_proposal_calculate((img_size, score_threshold, k, min_size,
                                           nms_threshold, post_nms_num,
                                           box_filter, aicore_num, sorted_proposal.get("dtype")),
                                          kernel_name)

    return tik_instance
