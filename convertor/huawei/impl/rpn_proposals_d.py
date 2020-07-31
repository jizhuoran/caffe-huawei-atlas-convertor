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

rpn_proposals
"""

# pylint: disable=C0302
# pylint: disable=R0913
# pylint: disable=R0914


from te import tik

from topi.cce import util

from impl import topk
from impl import nms
from impl import rpn_proposals_d_mdc_v200 as rpn_v200


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
CONFIG_SCORE_THRESHOLD = 0
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


class InitScalar:
    """
    the global scalar used
    the actual num after topk1, topk2, nms
    input_param : a list of attr
    img_size,
    score_threshold,
    k,
    min_size,
    nms_threshold,
    score_filter,
    box_filter,
    score_sigmoid,
    """

    def __init__(self, tik_instance, input_dict, input_param):
        data_type = input_dict[0].get("dtype")
        shape_n = input_dict[0].get("shape")[0]
        self.actual_num_psot_topk1 = tik_instance.Scalar("int32", init_value=CONFIG_SIXTEEN)
        self.actual_num_psot_topk2 = tik_instance.Scalar("int32", init_value=CONFIG_SIXTEEN)
        self.actual_num_psot_nms = tik_instance.Scalar("int32", init_value=CONFIG_SIXTEEN)

        # Middle tensors store the proposals  before the first topk
        self.proposal_gm = tik_instance.Tensor(data_type,
                                               (CONFIG_ONE, shape_n, CONFIG_EIGHT),
                                               name="proposal_gm",
                                               scope=tik.scope_gm,
                                               is_workspace=True)
        self.mem_swap = tik_instance.Tensor(data_type,
                                            (CONFIG_ONE, shape_n, CONFIG_EIGHT),
                                            name="men_swap",
                                            scope=tik.scope_gm,
                                            is_workspace=True)

        # real num of proposals after topk
        self.actual_proposal_num = tik_instance.Tensor("int32",
                                                       (CONFIG_ONE, CONFIG_EIGHT),
                                                       name="actual_proposal_num",
                                                       scope=tik.scope_cbuf)
        # Tensor after the size filter
        self.proposal_post_size_filter = tik_instance.Tensor(data_type,
                                                             (CONFIG_ONE,
                                                              input_param[CONFIG_TWO] +
                                                              CONFIG_FOUR,
                                                              CONFIG_EIGHT),
                                                             name="proposal_post_size_filter",
                                                             scope=tik.scope_cbuf)

    def set_actual_num_post_topk1(self, num):
        """
        set the num of proposals after first topk
        num: the num to set
        return: None
        """
        self.actual_num_psot_topk1.set_as(num)

    def set_actual_num_post_topk2(self, num):
        """
        set the num of proposals after first topk
        num: the num to set
        return: None
        """
        self.actual_num_psot_topk2.set_as(num)


class InitGmTensor(InitScalar):
    """
    initial the input and output tensor
    also middle tensor in DDR
    input_param : a list of attr
    img_size,
    score_threshold,
    k,
    min_size,
    nms_threshold,
    score_filter,
    box_filter,
    score_sigmoid,
    """

    def __init__(self, tik_instance, input_dict, input_param):
        super(InitGmTensor, self).__init__(tik_instance, input_dict, input_param)
        data_type = input_dict[0].get("dtype")
        shape_n = input_dict[0].get("shape")[0]
        shape_out = input_dict[CONFIG_TWO].get("shape")[0]

        # input tensor
        self.rois_gm = tik_instance.Tensor(data_type,
                                           (shape_n + CONFIG_FOUR, CONFIG_FOUR),
                                           name="rois_gm",
                                           scope=tik.scope_gm)
        self.prob_gm = tik_instance.Tensor(data_type,
                                           (shape_n + CONFIG_SIXTEEN, CONFIG_ONE),
                                           name="prob_gm",
                                           scope=tik.scope_gm)

        # Final output tensors in DDR
        self.box_gm = tik_instance.Tensor(data_type,
                                          (shape_out, CONFIG_FOUR),
                                          name="box_gm",
                                          scope=tik.scope_gm)

        # Tensors after the first topk
        self.proposal_post_topk = tik_instance.Tensor(data_type,
                                                      (CONFIG_ONE,
                                                       input_param[CONFIG_TWO] + CONFIG_FOUR,
                                                       CONFIG_EIGHT),
                                                      name="proposal_post_topk",
                                                      scope=tik.scope_cbuf)

        self.mem_swap_second = tik_instance.Tensor(data_type,
                                                   (CONFIG_ONE,
                                                    input_param[CONFIG_TWO] + CONFIG_FOUR,
                                                    CONFIG_EIGHT),
                                                   name="men_swap_second",
                                                   scope=tik.scope_cbuf)

        # tensors after second topk and befor nms
        self.proposal_pre_nms = tik_instance.Tensor(data_type,
                                                    (CONFIG_ONE,
                                                     ceil_div(input_param[CONFIG_TWO], CONFIG_MASK)*
                                                     CONFIG_MASK,
                                                     CONFIG_EIGHT),
                                                    name="proposal_pre_nms",
                                                    scope=tik.scope_cbuf)

    def set_actual_num_post_nms(self, tik_instance):
        """
        set the actual proposal num
        param num_tensor:
        return:
        """
        with tik_instance.new_stmt_scope():
            num_tensor = tik_instance.Tensor("int32",
                                             (CONFIG_ONE, CONFIG_EIGHT),
                                             name="mun_tensor",
                                             scope=tik.scope_ubuf)
            tik_instance.data_move(num_tensor,
                                   self.actual_proposal_num,
                                   0,
                                   CONFIG_ONE,
                                   CONFIG_ONE,
                                   0, 0)
            self.actual_num_psot_nms.set_as(num_tensor[0])

    def set_mem_swap(self, temp_tensor):
        """
        set the mem_swap tensor for topk
         temp_tensor: the middle tensor in GM
        return: None
        """
        self.mem_swap = temp_tensor


class InitNmsTensor:
    """
    Init the GM tensor neede by NMS
        param input_param:   nms_threshold, img_height, img_width,
                              score_filter=True, score_threshold=0,
                              k=6000,
                              box_filter=True, min_height=0, min_width=0,
                             score_sigmoid=True,
        param input_dict:  rois[N,4], cls_bg_prob[N,1], sorted_box[M,4], sorted_scores[M,1]
    """

    def __init__(self, tik_instance, input_dict):
        self.sorted_scores = input_dict[CONFIG_TWO]
        self.dtype = self.sorted_scores.get("dtype")
        self.post_nms_topn = input_dict[CONFIG_TWO].get("shape")[0]

        # nms out put gm
        self.nms_proposal_gm = tik_instance.Tensor(self.sorted_scores.get("dtype"),
                                                   (CONFIG_ONE, CONFIG_ONE,
                                                    self.post_nms_topn + CONFIG_FOUR, CONFIG_EIGHT),
                                                   name="nms_proposal_gm",
                                                   scope=tik.scope_gm,
                                                   is_workspace=True)
        # temp nms gm
        self.nms_temp_proposal_out = tik_instance.Tensor(self.sorted_scores.get("dtype"),
                                                         (CONFIG_ONE,
                                                          self.post_nms_topn,
                                                          CONFIG_EIGHT),
                                                         name="nms_temp_proposal_out",
                                                         scope=tik.scope_cbuf)
        # nms out put actual num
        self.nms_actual_rois_num = tik_instance.Tensor("int32",
                                                       (CONFIG_ONE, CONFIG_ONE, CONFIG_EIGHT),
                                                       name="nms_actual_rois_num",
                                                       scope=tik.scope_cbuf)

    def set_sorted_scores(self, dict_in):
        """
        set the sorted score dict
         dict_in: dict in
        return: None
        """
        self.sorted_scores = dict_in

    def set_post_nms_topn(self, num):
        """
        set the num of proposals after nms
         num: num of post nms proposals
        return: None
        """
        self.post_nms_topn = num


class InitMiddleTensor:
    """
    init the middle tensoe in the UB
    for each core
    """

    def __init__(self, tik_instance, tiling_para):

        self.rois_ub = tik_instance.Tensor(tiling_para.dtype,
                                           (tiling_para.one_core_num, CONFIG_FOUR),
                                           name="rois_ub",
                                           scope=tik.scope_ubuf)
        self.prob_ub = tik_instance.Tensor(tiling_para.dtype,
                                           (tiling_para.one_core_num * CONFIG_FOUR, CONFIG_ONE),
                                           name="prob_ub",
                                           scope=tik.scope_ubuf)
        self.rois_ub_trans = tik_instance.Tensor(tiling_para.dtype,
                                                 (CONFIG_FOUR, tiling_para.one_core_num),
                                                 name="rois_ub_trans",
                                                 scope=tik.scope_ubuf)
        self.prob_ub_trans = tik_instance.Tensor(tiling_para.dtype,
                                                 (CONFIG_ONE,
                                                  tiling_para.one_core_num * CONFIG_FOUR),
                                                 name="prob_ub_trans",
                                                 scope=tik.scope_ubuf)
        self.proposal = tik_instance.Tensor(tiling_para.dtype,
                                            (tiling_para.one_core_num, CONFIG_EIGHT),
                                            name="proposal",
                                            scope=tik.scope_ubuf)

    def set_rois_ub(self, rois_ub):
        """
        set the rois_ub
        param rois_ub: input
        return: None
        """
        self.rois_ub = rois_ub

    def set_prob_ub(self, prob_ub):
        """
        set the prob_ub
        param prob_ub: input
        return: None
        """
        self.prob_ub = prob_ub


class InitTialTensor:
    """
    init the middle tensoe in the UB
    for each core
    """

    def __init__(self, tik_instance, tiling_para):
        """
        param tik_instance:
        param tail_tiling:
        one_core_num, cycle_times, left_num, tiling_para.dtype, mode_flag
        """

        self.rois_ub = tik_instance.Tensor(tiling_para.dtype,
                                           (tiling_para.one_core_num, CONFIG_SIXTEEN),
                                           name="rois_ub",
                                           scope=tik.scope_ubuf)
        self.prob_ub = tik_instance.Tensor(tiling_para.dtype,
                                           (tiling_para.one_core_num, CONFIG_ONE),
                                           name="prob_ub",
                                           scope=tik.scope_ubuf)
        self.rois_ub_trans = tik_instance.Tensor(tiling_para.dtype,
                                                 (CONFIG_SIXTEEN, tiling_para.one_core_num),
                                                 name="rois_ub_trans",
                                                 scope=tik.scope_ubuf)
        self.prob_ub_trans = tik_instance.Tensor(tiling_para.dtype,
                                                 (CONFIG_ONE, tiling_para.one_core_num),
                                                 name="prob_ub_trans",
                                                 scope=tik.scope_ubuf)
        self.proposal = tik_instance.Tensor(tiling_para.dtype,
                                            (tiling_para.one_core_num, CONFIG_EIGHT),
                                            name="proposal",
                                            scope=tik.scope_ubuf)

    def set_rois_ub_tail(self, rois_ub):
        """
        set the rois_ub
        param rois_ub: input
        return: None
        """
        self.rois_ub = rois_ub

    def set_prob_ub_tail(self, prob_ub):
        """
        set the prob_ub
        param prob_ub: input
        return: None
        """
        self.prob_ub = prob_ub


class InitFilterConst:
    """
    init some const for the size filter using
    according to the input and output shape
    input_param : a list of attr
            img_size, score_threshold, k,  min_size,
            nms_threshold,
            score_filter,  box_filter,   score_sigmoid
    """

    def __init__(self, input_param, input_dict):
        self.sorted_scores = input_dict[2]
        self.dtype = self.sorted_scores.get("dtype")

        self.top_k = input_param[CONFIG_TWO]

        self.filter_ub_size = \
            ceil_div(self.top_k // CONFIG_TWO, CONFIG_DATA_TRANS) * CONFIG_DATA_TRANS
        self.mask_size = \
            ceil_div(self.filter_ub_size // CONFIG_EIGHT, CONFIG_MASK) * CONFIG_MASK

    def set_dtype(self, dtype):
        """
        set the dtype
         dtype: input
        return: None
        """
        self.dtype = dtype

    def set_top_k(self, num):
        """
        set the topk num
        param num: input
        return: None
        """
        self.top_k = num


class InitFilterTensor(InitFilterConst):
    """
    init tensor in UB
    theses tensors are used for boxes clip and size filter
    """

    def __init__(self, tik_instance, input_param, input_dict):
        """
        tik_instance: tik container
        input_param : a list of attr
            img_size, score_threshold, k,  min_size,
            nms_threshold,
            score_filter,  box_filter,   score_sigmoid,

        input_dict: a list
           rois[N,4], cls_bg_prob[N,1], sorted_box[M,4], sorted_scores[M,1]
        """
        super(InitFilterTensor, self).__init__(input_param, input_dict)

        self.filter_ub = tik_instance.Tensor(self.dtype,
                                             (self.filter_ub_size, CONFIG_SIXTEEN),
                                             name="filter_ub",
                                             scope=tik.scope_ubuf)

        self.filter_trans_ub = tik_instance.Tensor(self.dtype,
                                                   (self.filter_ub_size, CONFIG_SIXTEEN),
                                                   name="filter_trans_ub",
                                                   scope=tik.scope_ubuf)
        self.filter_weight_ub = tik_instance.Tensor(self.dtype,
                                                    (CONFIG_MASK,),
                                                    name="filter_weight_ub",
                                                    scope=tik.scope_ubuf)
        self.filter_height_ub = tik_instance.Tensor(self.dtype,
                                                    (CONFIG_MASK,),
                                                    name="filter_height_ub",
                                                    scope=tik.scope_ubuf)
        self.mask_x_tensor = tik_instance.Tensor("uint16",
                                                 (self.mask_size,),
                                                 name="mask_x_tensor",
                                                 scope=tik.scope_ubuf)
        self.mask_y_tensor = tik_instance.Tensor("uint16",
                                                 (self.mask_size,),
                                                 name="mask_y_tensor",
                                                 scope=tik.scope_ubuf)
        self.mask_tensor = tik_instance.Tensor("uint16",
                                               (self.mask_size,),
                                               name="mask_tensor",
                                               scope=tik.scope_ubuf)

    def set_filter_ub(self, filter_ub):
        """
        set the filter_ub
        param filter_ub: input
        return: None
        """
        self.filter_ub = filter_ub

    def set_filter_trans_ub(self, filter_trans_ub):
        """
        set the filter_trans_ub
        param filter_trans_ub:  input
        return: None
        """
        self.filter_trans_ub = filter_trans_ub


class InitSigmoidTensor:
    """
    init tensors for sigmoid
    param input_param:   nms_threshold, img_height, img_width,
                          score_filter=True, score_threshold=0,
                          k=6000,
                          box_filter=True, min_height=0, min_width=0,
                         score_sigmoid=True,
    param input_dict:  rois[N,4], cls_bg_prob[N,1], sorted_box[M,4], sorted_scores[M,1]
    """

    def __init__(self, tik_instance, input_dict):
        self.sorted_box = input_dict[CONFIG_TWO]
        self.sigmoid_shape = self.sorted_box.get("shape")[0]
        self.sigmoid_box_ub = tik_instance.Tensor(self.sorted_box.get("dtype"),
                                                  (self.sigmoid_shape, CONFIG_SIXTEEN),
                                                  name="sigmoid_box_ub",
                                                  scope=tik.scope_ubuf)
        self.merged_box_ub = tik_instance.Tensor(self.sorted_box.get("dtype"),
                                                 (self.sigmoid_shape, CONFIG_FOUR),
                                                 name="merged_box_ub",
                                                 scope=tik.scope_ubuf)

    def set_sigmoid_ub(self, sigmoid_box_ub):
        """
        set the sigmoid_ub
        param sigmoid_ub: input
        return: None
        """
        self.sigmoid_box_ub = sigmoid_box_ub

    def set_sigmoid_extract_ub(self, merged_box_ub):
        """
        set the sigmoid_extract_ub
        param sigmoid_extract_ub: input
        return: None
        """
        self.merged_box_ub = merged_box_ub


class MultiCoreParam:
    """
    set the multi-core parameters
    """

    def __init__(self):
        # self.core_num = CONFIG_ONE
        self.thread_num = CONFIG_ONE
        self.thread_cycles = CONFIG_ONE
        self.left_cycles = 0
        self.clc_flag = True

    def set_left_cycles(self, num):
        """
        set the cycle(each part difined by the cycle_param) times
        num: num of cycles to be calculated
        """
        self.left_cycles = num

    def set_multicore_pingpang(self):
        """
        canot use the  multi-core, only use the pingpang

        core_num: the core aviliable, not used
        """
        # multi-core cannot be used
        # connot use multi-core, using pingpang
        if self.left_cycles >= CONFIG_TWO:
            # self.core_num = CONFIG_ONE
            self.thread_num = CONFIG_TWO
            self.thread_cycles = self.left_cycles
            self.left_cycles = 0
            self.clc_flag = True

        elif self.left_cycles > 0:
            # self.core_num = CONFIG_ONE
            self.thread_num = CONFIG_ONE
            self.thread_cycles = CONFIG_ONE
            self.left_cycles = 0
            self.clc_flag = True

        else:
            self.clc_flag = False


class TilingParam(MultiCoreParam):
    """
    the tiling parameters for proposal combining
    each core processing 1024 proposals each time
    """

    def __init__(self, input_dict):
        super(TilingParam, self).__init__()
        # obtain the input shape and dtype
        rois = input_dict[0]
        self.proposal_num = rois.get("shape")[0]
        self.dtype = rois.get("dtype")

        # obtain the UB size  Bytes
        size_of_ub = tik.Dprofile().get_unified_buffer_size()
        if CONFIG_UNIT * CONFIG_SIXTEEN * CONFIG_THREE > size_of_ub:
            raise RuntimeError("The proposals numbers is too lager,"
                               " please reset one according to the UB size!")

        self.one_core_num = CONFIG_UNIT
        self.cycle_times = self.proposal_num // CONFIG_UNIT
        self.left_num = self.proposal_num % CONFIG_UNIT
        self.off_set = 0

    def set_one_cycle_param(self, one_core_n, left_n):
        """
        reset the one cycle parameters for tail part

        one_core_n: proposals num processed onr time
        left_n: proposals to be deal with
        """
        self.one_core_num = one_core_n
        self.cycle_times = left_n // one_core_n
        self.left_num = left_n % one_core_n

    def set_offset(self, offset_n):
        """
        set the offset info

        offset_n: offset num of proposals
        """
        self.off_set = offset_n


def one_core_process(tik_instance, data_tensor, middle_tensor, tiling_para):
    """
    Read in the data in DDR
    Combine them into proposal
    And move them out in the DDR

    Input:
    ---------------------
    tik_instance: the tik container
    data_tensor: tensors in the DDR
       rois_gm, prob_gm, box_gm, scores_gm,
        proposal_gm, mem_swap, proposal_post_topk,
         topk_output_actual_proposal_num

    midle_tensor: tensors in UB
    rois_ub, prob_ub, rois_ub_trans, prob_ub_trans, proposal

    tiling_para:
    one_core_num, cycle_times, left_num,

    off_set:
    the offset of the DDR adress, unit CONFIG_UNIT

    """

    num_offset = tiling_para.off_set
    # move date from gm to ub
    tik_instance.data_move(middle_tensor.rois_ub[0],
                           data_tensor.rois_gm[num_offset * CONFIG_FOUR],
                           0,
                           CONFIG_ONE,
                           tiling_para.one_core_num//CONFIG_FOUR,
                           0, 0)

    tik_instance.data_move(middle_tensor.prob_ub[0],
                           data_tensor.prob_gm[num_offset],
                           0,
                           tiling_para.one_core_num//CONFIG_SIXTEEN,
                           CONFIG_ONE,
                           0,
                           CONFIG_THREE)
    tik_instance.data_move(middle_tensor.prob_ub[CONFIG_SIXTEEN],
                           data_tensor.prob_gm[num_offset + CONFIG_FOUR],
                           0,
                           tiling_para.one_core_num//CONFIG_SIXTEEN,
                           CONFIG_ONE,
                           0,
                           CONFIG_THREE)
    tik_instance.data_move(middle_tensor.prob_ub[CONFIG_SIXTEEN *CONFIG_TWO],
                           data_tensor.prob_gm[num_offset + CONFIG_EIGHT],
                           0,
                           tiling_para.one_core_num//CONFIG_SIXTEEN,
                           CONFIG_ONE,
                           0,
                           CONFIG_THREE)
    tik_instance.data_move(middle_tensor.prob_ub[CONFIG_SIXTEEN * CONFIG_THREE],
                           data_tensor.prob_gm[num_offset + CONFIG_TWELEV],
                           0,
                           tiling_para.one_core_num//CONFIG_SIXTEEN,
                           CONFIG_ONE,
                           0,
                           CONFIG_THREE)

    # data transform from N*a to a*N
    dst_list = [middle_tensor.rois_ub_trans[CONFIG_SIXTEEN * i]
                for i in range(CONFIG_SIXTEEN)]
    src_list = [middle_tensor.rois_ub[CONFIG_SIXTEEN * i]
                for i in range(CONFIG_SIXTEEN)]
    tik_instance.vnchwconv(True, False, dst_list, src_list,
                           tiling_para.one_core_num//CONFIG_DATA_TRANS,
                           CONFIG_SIXTEEN,
                           CONFIG_SIXTEEN)

    dst_list = [middle_tensor.prob_ub_trans[CONFIG_SIXTEEN * i]
                for i in range(CONFIG_SIXTEEN)]
    src_list = [middle_tensor.prob_ub[CONFIG_SIXTEEN * i]
                for i in range(CONFIG_SIXTEEN)]
    tik_instance.vnchwconv(True, False, dst_list, src_list,
                           tiling_para.one_core_num//CONFIG_DATA_TRANS,
                           CONFIG_SIXTEEN,
                           CONFIG_SIXTEEN)

    # move the x1,y1,x2,y2 and prob to continue area
    tik_instance.vadds(CONFIG_MASK,
                       middle_tensor.rois_ub,
                       middle_tensor.rois_ub_trans,
                       0,
                       tiling_para.one_core_num//CONFIG_MASK,
                       CONFIG_ONE,
                       CONFIG_FOUR,
                       CONFIG_EIGHT,
                       CONFIG_DATA_ALIGN)
    tik_instance.vadds(CONFIG_MASK,
                       middle_tensor.rois_ub[tiling_para.one_core_num],
                       middle_tensor.rois_ub_trans[CONFIG_SIXTEEN],
                       0,
                       tiling_para.one_core_num//CONFIG_MASK,
                       CONFIG_ONE,
                       CONFIG_FOUR,
                       CONFIG_EIGHT,
                       CONFIG_DATA_ALIGN)
    tik_instance.vadds(CONFIG_MASK,
                       middle_tensor.rois_ub[tiling_para.one_core_num * CONFIG_TWO],
                       middle_tensor.rois_ub_trans[CONFIG_DATA_ALIGN],
                       0,
                       tiling_para.one_core_num//CONFIG_MASK,
                       CONFIG_ONE,
                       CONFIG_FOUR,
                       CONFIG_EIGHT,
                       CONFIG_DATA_ALIGN)
    tik_instance.vadds(CONFIG_MASK,
                       middle_tensor.rois_ub[tiling_para.one_core_num * CONFIG_THREE],
                       middle_tensor.rois_ub_trans[CONFIG_DATA_ALIGN + CONFIG_SIXTEEN],
                       0,
                       tiling_para.one_core_num//CONFIG_MASK,
                       CONFIG_ONE,
                       CONFIG_FOUR,
                       CONFIG_EIGHT,
                       CONFIG_DATA_ALIGN)

    tik_instance.vadds(CONFIG_MASK,
                       middle_tensor.prob_ub,
                       middle_tensor.prob_ub_trans,
                       0,
                       tiling_para.one_core_num//CONFIG_DATA_TRANS,
                       CONFIG_ONE,
                       CONFIG_ONE,
                       CONFIG_FOUR,
                       CONFIG_SIXTEEN)

    # for debug only
    tik_instance.vector_dup(CONFIG_MASK,
                            middle_tensor.proposal,
                            0,
                            tiling_para.one_core_num//CONFIG_SIXTEEN,
                            CONFIG_ONE,
                            CONFIG_EIGHT)

    # combine into proposals
    tik_instance.vconcat(middle_tensor.proposal,
                         middle_tensor.rois_ub,
                         tiling_para.one_core_num//CONFIG_SIXTEEN,
                         0)
    tik_instance.vconcat(middle_tensor.proposal,
                         middle_tensor.rois_ub[tiling_para.one_core_num],
                         tiling_para.one_core_num//CONFIG_SIXTEEN,
                         CONFIG_ONE)
    tik_instance.vconcat(middle_tensor.proposal,
                         middle_tensor.rois_ub[tiling_para.one_core_num * CONFIG_TWO],
                         tiling_para.one_core_num//CONFIG_SIXTEEN,
                         CONFIG_TWO)
    tik_instance.vconcat(middle_tensor.proposal,
                         middle_tensor.rois_ub[tiling_para.one_core_num * CONFIG_THREE],
                         tiling_para.one_core_num//CONFIG_SIXTEEN,
                         CONFIG_THREE)
    tik_instance.vconcat(middle_tensor.proposal,
                         middle_tensor.prob_ub,
                         tiling_para.one_core_num//CONFIG_SIXTEEN,
                         CONFIG_FOUR)

    # move out to DDR
    tik_instance.data_move(data_tensor.proposal_gm[num_offset * CONFIG_EIGHT],
                           middle_tensor.proposal,
                           0,
                           CONFIG_ONE,
                           tiling_para.one_core_num//CONFIG_TWO,
                           0, 0)


def one_core_process_tial(tik_instance, data_tensor, middle_tensor, tiling_para):
    """
    Read in the data in DDR
    Combine them into proposal
    And move them out in the DDR
    deal with the tail part

    Input:
    ---------------------
    tik_instance: the tik container
    data_tensor: tensors in the DDR
        rois_gm, prob_gm, box_gm, scores_gm,
        proposal_gm, mem_swap, proposal_post_topk,
         topk_output_actual_proposal_num

    midle_tensor: tensors in UB
    rois_ub, prob_ub, rois_ub_trans, prob_ub_trans, proposal

    tiling_para:
    one_core_num, cycle_times, left_num,

    """

    num_offset = tiling_para.off_set

    # real num of block of data left
    num_block = tiling_para.left_num // CONFIG_SIXTEEN

    # move date from gm to ub
    tik_instance.data_move(middle_tensor.rois_ub[0],
                           data_tensor.rois_gm[num_offset * CONFIG_FOUR],
                           0,
                           num_block * CONFIG_FOUR,
                           CONFIG_ONE,
                           0,
                           CONFIG_THREE)
    tik_instance.data_move(middle_tensor.rois_ub[CONFIG_SIXTEEN],
                           data_tensor.rois_gm[num_offset * CONFIG_FOUR + CONFIG_FOUR],
                           0,
                           num_block * CONFIG_FOUR,
                           CONFIG_ONE,
                           0,
                           CONFIG_THREE)
    tik_instance.data_move(middle_tensor.rois_ub[CONFIG_SIXTEEN * CONFIG_TWO],
                           data_tensor.rois_gm[num_offset * CONFIG_FOUR + CONFIG_EIGHT],
                           0,
                           num_block * CONFIG_FOUR,
                           CONFIG_ONE,
                           0,
                           CONFIG_THREE)
    tik_instance.data_move(middle_tensor.rois_ub[CONFIG_SIXTEEN * CONFIG_THREE],
                           data_tensor.rois_gm[num_offset * CONFIG_FOUR + CONFIG_TWELEV],
                           0,
                           num_block * CONFIG_FOUR,
                           CONFIG_ONE,
                           0,
                           CONFIG_THREE)

    tik_instance.data_move(middle_tensor.prob_ub[0],
                           data_tensor.prob_gm[num_offset],
                           0,
                           CONFIG_ONE,
                           num_block,
                           0,
                           0)

    # data transform from N*a to a*N
    dst_list = [middle_tensor.rois_ub_trans[CONFIG_MASK * i]
                for i in range(CONFIG_SIXTEEN)]
    src_list = [middle_tensor.rois_ub[CONFIG_SIXTEEN * i]
                for i in range(CONFIG_SIXTEEN)]
    tik_instance.vnchwconv(True, False, dst_list, src_list,
                           tiling_para.one_core_num//CONFIG_SIXTEEN,
                           CONFIG_ONE,
                           CONFIG_SIXTEEN)

    # for debug only
    tik_instance.vector_dup(CONFIG_MASK,
                            middle_tensor.proposal,
                            0,
                            tiling_para.one_core_num//CONFIG_SIXTEEN,
                            CONFIG_ONE,
                            CONFIG_EIGHT)


    # combine into proposals
    tik_instance.vconcat(middle_tensor.proposal,
                         middle_tensor.rois_ub_trans,
                         num_block,
                         0)
    tik_instance.vconcat(middle_tensor.proposal,
                         middle_tensor.rois_ub_trans[tiling_para.one_core_num],
                         num_block,
                         CONFIG_ONE)
    tik_instance.vconcat(middle_tensor.proposal,
                         middle_tensor.rois_ub_trans[tiling_para.one_core_num * CONFIG_TWO],
                         num_block,
                         CONFIG_TWO)
    tik_instance.vconcat(middle_tensor.proposal,
                         middle_tensor.rois_ub_trans[tiling_para.one_core_num * CONFIG_THREE],
                         num_block,
                         CONFIG_THREE)
    tik_instance.vconcat(middle_tensor.proposal,
                         middle_tensor.prob_ub,
                         num_block,
                         CONFIG_FOUR)


    # move out to DDR
    tik_instance.data_move(data_tensor.proposal_gm[num_offset * CONFIG_EIGHT],
                           middle_tensor.proposal,
                           0,
                           CONFIG_ONE,
                           tiling_para.left_num//CONFIG_TWO,
                           0, 0)


def combine_proposals(tik_instance, data_tensor, tiling_para):
    """
    combine the boxes and prob into proposals

    tik_instance: the tik container
    data_tensor: the tensors stored in gm
    tiling_para:  tiling parameters

    return: None
    """

    tiling_para.set_left_cycles(tiling_para.cycle_times)

    tiling_para.set_multicore_pingpang()
    offset_num = 0

    # combine to proposals and store in DDR
    if tiling_para.clc_flag:
        with tik_instance.new_stmt_scope():
            with tik_instance.for_range(0, tiling_para.thread_cycles,
                                        thread_num=tiling_para.thread_num) as t_thread:
                # initial the middle tensor in UB for proposal combining
                middle_tensor = InitMiddleTensor(tik_instance, tiling_para)
                tiling_para.set_offset(offset_num + t_thread * tiling_para.one_core_num)
                # each cycle processing CONFIG_UNIT proposals
                one_core_process(tik_instance, data_tensor, middle_tensor, tiling_para)

        offset_num += tiling_para.one_core_num * tiling_para.thread_cycles
        tiling_para.set_offset(offset_num)
        tiling_para.set_multicore_pingpang()

    offset_num = tiling_para.one_core_num * tiling_para.cycle_times
    tiling_para.set_offset(offset_num)

    # the big tail data processing
    if tiling_para.left_num > 0:
        tiling_para.set_one_cycle_param(CONFIG_MASK, tiling_para.left_num)

        tiling_para.set_left_cycles(tiling_para.cycle_times)
        tiling_para.set_multicore_pingpang()

        if tiling_para.clc_flag:
            with tik_instance.new_stmt_scope():
                # initial the middle tensor in UB for proposal combining
                middle_tensor = InitMiddleTensor(tik_instance, tiling_para)
                with tik_instance.for_range(0, tiling_para.thread_cycles,
                                            thread_num=tiling_para.thread_num) as t_thread:
                    tiling_para.set_offset(offset_num + t_thread * tiling_para.one_core_num)
                    one_core_process(tik_instance, data_tensor, middle_tensor, tiling_para)

        offset_num += tiling_para.one_core_num * tiling_para.cycle_times
        tiling_para.set_offset(offset_num)

    # little tail data processing. use one core only
    if tiling_para.left_num > 0:
        with tik_instance.new_stmt_scope():
            # initial the middle tensor in UB for proposal combining
            middle_tensor = InitTialTensor(tik_instance, tiling_para)
            one_core_process_tial(tik_instance, data_tensor, middle_tensor, tiling_para)


def call_topk_sort(tik_instance, input_tensor, input_param, proposal_num):
    """
    call the topk function

    tik_instance: tik container

    data_tensor: tensors in the DDR
      proposal_gm, proposal_post_topk, mem_swap, actual_proposal_num
    input_tensor:
       [0]: Tensors in DDR before Topk
       [1]: Tensors in DDR used as temp
       [2]: Tensors in DDR after Topk

    input_param : a list of attr
    img_size, score_threshold, k,  min_size,   nms_threshold,
    score_filter,  box_filter,  score_sigmoid,

    proposal_num:
       num of proposals
    output:
    :return:
    """

    batch_id = 0

    # whether the score filter is needed, if not reset the threshold
    score_threshold = input_param[CONFIG_ONE]

    topk_input = {
        "proposal_num": proposal_num,
        "k": input_param[CONFIG_TWO],
        "score_threshold": score_threshold,
        "regions_orig": input_tensor[0],
        "mem_swap": input_tensor[CONFIG_ONE],
    }

    topk_out = {
        "batch_id": batch_id,
        "regions_sorted": input_tensor[CONFIG_TWO],
        "proposal_actual_num": input_tensor[CONFIG_THREE],
    }

    with tik_instance.new_stmt_scope():
        topk.tik_topk(tik_instance, topk_input, topk_out)


def filter_with_height_weight(tik_instance, data_tensor, filter_tensor, input_param):
    """
    do the clipboxes and size filter
        tik_instance:  tik container
        data_tensor : data tensor in the DDR
        filter_tensor: UB tensors needed here inited by InitFilterTensor
        input_param : a list of attr
            img_size, score_threshold, k,  min_size,  nms_threshold,
            score_filter,  box_filter,   score_sigmoid,

        input_dict:  rois[N,4], cls_bg_prob[N,1], sorted_box[M,4], sorted_scores[M,1]
    return: None
    """

    score_threshold = input_param[CONFIG_ONE]
    box_filter = input_param[CONFIG_SIX]
    # the size of the image [h, w]
    image_size = input_param[0]
    # the size filter, only one parameter
    image_threshold = input_param[CONFIG_THREE]

    filter_repeat_data = ceil_div(input_param[CONFIG_TWO] * CONFIG_EIGHT, CONFIG_SIXTEEN)

    # THE SIZE(topk_proposal_gm) = (1,1,6000+4,8) ,SIZE(filter_ub)= (6000 // 2 + 63) // 64 * 64
    tik_instance.data_move(filter_tensor.filter_ub,
                           data_tensor.proposal_post_topk,
                           0,
                           CONFIG_ONE,
                           filter_repeat_data,
                           0, 0)

    fliter_repeat_trans = filter_tensor.filter_ub_size // CONFIG_EIGHT

    # trans the data filter_trans_ub[0,1,2,3,4,5,6,7]
    with tik_instance.for_range(0, fliter_repeat_trans // CONFIG_TWO) as i:
        tik_instance.vtranspose(filter_tensor.filter_trans_ub[MATRIX * i],
                                filter_tensor.filter_ub[MATRIX * i])

    # dump the clip box vector
    tik_instance.vector_dup(CONFIG_MASK,
                            filter_tensor.filter_height_ub,
                            image_size[0],
                            CONFIG_ONE,
                            CONFIG_ONE,
                            CONFIG_EIGHT)
    tik_instance.vector_dup(CONFIG_MASK,
                            filter_tensor.filter_weight_ub,
                            image_size[CONFIG_ONE],
                            CONFIG_ONE,
                            CONFIG_ONE,
                            CONFIG_EIGHT)

    # clip box
    # tik_instance.tikdb.debug_print("fliter_repeat_trans")
    if fliter_repeat_trans > MAX_REP_TIME:
        tik_instance.vrelu(CONFIG_DATA_TRANS,
                           filter_tensor.filter_trans_ub,
                           filter_tensor.filter_trans_ub,
                           MATRIX - CONFIG_ONE,
                           CONFIG_ONE,
                           CONFIG_ONE,
                           CONFIG_EIGHT,
                           CONFIG_EIGHT)
        tik_instance.vrelu(CONFIG_DATA_TRANS,
                           filter_tensor.filter_trans_ub[MAX_REP_TIME * CONFIG_EIGHT, 0],
                           filter_tensor.filter_trans_ub[MAX_REP_TIME * CONFIG_EIGHT, 0],
                           (fliter_repeat_trans - MAX_REP_TIME),
                           CONFIG_ONE,
                           CONFIG_ONE,
                           CONFIG_EIGHT,
                           CONFIG_EIGHT)
    else:
        tik_instance.vrelu(CONFIG_DATA_TRANS,
                           filter_tensor.filter_trans_ub[0, 0],
                           filter_tensor.filter_trans_ub[0, 0],
                           fliter_repeat_trans,
                           CONFIG_ONE,
                           CONFIG_ONE,
                           CONFIG_EIGHT,
                           CONFIG_EIGHT)

    tik_instance.vmin(CONFIG_MASK,
                      filter_tensor.filter_trans_ub,
                      filter_tensor.filter_trans_ub,
                      filter_tensor.filter_weight_ub,
                      fliter_repeat_trans // CONFIG_EIGHT,
                      CONFIG_EIGHT,
                      CONFIG_EIGHT,
                      CONFIG_ONE,
                      CONFIG_DATA_TRANS,
                      CONFIG_DATA_TRANS,
                      0)

    tik_instance.vmin(CONFIG_MASK,
                      filter_tensor.filter_trans_ub[CONFIG_TWO, 0],
                      filter_tensor.filter_trans_ub[CONFIG_TWO, 0],
                      filter_tensor.filter_weight_ub,
                      fliter_repeat_trans // CONFIG_EIGHT,
                      CONFIG_EIGHT,
                      CONFIG_EIGHT,
                      CONFIG_ONE,
                      CONFIG_DATA_TRANS,
                      CONFIG_DATA_TRANS,
                      0)
    tik_instance.vmin(CONFIG_MASK,
                      filter_tensor.filter_trans_ub[CONFIG_ONE, 0],
                      filter_tensor.filter_trans_ub[CONFIG_ONE, 0],
                      filter_tensor.filter_height_ub,
                      fliter_repeat_trans // CONFIG_EIGHT,
                      CONFIG_EIGHT,
                      CONFIG_EIGHT,
                      CONFIG_ONE,
                      CONFIG_DATA_TRANS,
                      CONFIG_DATA_TRANS,
                      0)
    tik_instance.vmin(CONFIG_MASK,
                      filter_tensor.filter_trans_ub[CONFIG_THREE, 0],
                      filter_tensor.filter_trans_ub[CONFIG_THREE, 0],
                      filter_tensor.filter_height_ub,
                      fliter_repeat_trans // CONFIG_EIGHT,
                      CONFIG_EIGHT,
                      CONFIG_EIGHT,
                      CONFIG_ONE,
                      CONFIG_DATA_TRANS,
                      CONFIG_DATA_TRANS,
                      0)

    if box_filter:
        # calculate the x2 - x1 & y2 - y1
        tik_instance.vsub(CONFIG_MASK,
                          filter_tensor.filter_trans_ub[CONFIG_FIVE, 0],
                          filter_tensor.filter_trans_ub[CONFIG_TWO, 0],
                          filter_tensor.filter_trans_ub,
                          fliter_repeat_trans // CONFIG_EIGHT,
                          CONFIG_EIGHT,
                          CONFIG_EIGHT,
                          CONFIG_EIGHT,
                          CONFIG_DATA_TRANS,
                          CONFIG_DATA_TRANS,
                          CONFIG_DATA_TRANS)
        tik_instance.vsub(CONFIG_MASK,
                          filter_tensor.filter_trans_ub[CONFIG_SIX, 0],
                          filter_tensor.filter_trans_ub[CONFIG_THREE, 0],
                          filter_tensor.filter_trans_ub[CONFIG_ONE, 0],
                          fliter_repeat_trans // CONFIG_EIGHT,
                          CONFIG_EIGHT,
                          CONFIG_EIGHT,
                          CONFIG_EIGHT,
                          CONFIG_DATA_TRANS,
                          CONFIG_DATA_TRANS,
                          CONFIG_DATA_TRANS)

        tik_instance.vector_dup(CONFIG_MASK,
                                filter_tensor.filter_trans_ub[CONFIG_SEVEN, 0],
                                image_threshold,
                                fliter_repeat_trans // CONFIG_EIGHT,
                                CONFIG_EIGHT,
                                CONFIG_DATA_TRANS)
        tik_instance.vcmpv_gt(filter_tensor.mask_x_tensor,
                              filter_tensor.filter_trans_ub[CONFIG_SIX, 0],
                              filter_tensor.filter_trans_ub[CONFIG_SEVEN, 0],
                              fliter_repeat_trans // CONFIG_EIGHT,
                              CONFIG_EIGHT,
                              CONFIG_EIGHT,
                              CONFIG_DATA_TRANS,
                              CONFIG_DATA_TRANS)
        tik_instance.vcmpv_gt(filter_tensor.mask_y_tensor,
                              filter_tensor.filter_trans_ub[CONFIG_FIVE, 0],
                              filter_tensor.filter_trans_ub[CONFIG_SEVEN, 0],
                              fliter_repeat_trans // CONFIG_EIGHT,
                              CONFIG_EIGHT,
                              CONFIG_EIGHT,
                              CONFIG_DATA_TRANS,
                              CONFIG_DATA_TRANS)

        tik_instance.vector_dup(CONFIG_MASK,
                                filter_tensor.mask_tensor,
                                0,
                                filter_tensor.mask_size // CONFIG_MASK,
                                CONFIG_ONE,
                                CONFIG_EIGHT)
        tik_instance.vand([0, CONFIG_TWO ** CONFIG_DATA_TRANS - CONFIG_ONE],
                          filter_tensor.mask_tensor,
                          filter_tensor.mask_y_tensor,
                          filter_tensor.mask_x_tensor,
                          filter_tensor.mask_size // CONFIG_MASK * CONFIG_TWO,
                          CONFIG_ONE,
                          CONFIG_ONE,
                          CONFIG_ONE,
                          CONFIG_FOUR,
                          CONFIG_FOUR,
                          CONFIG_FOUR)

        # mask_tensor.size = 382 uint16
        tik_instance.vector_dup(CONFIG_MASK,
                                filter_tensor.filter_height_ub,
                                score_threshold,
                                CONFIG_ONE,
                                CONFIG_ONE,
                                CONFIG_EIGHT)

        with tik_instance.for_range(0, (filter_tensor.mask_size // CONFIG_EIGHT - CONFIG_ONE)) as i:
            cmpmask = \
                tik_instance.mov_tensor_to_cmpmask(filter_tensor.mask_tensor[CONFIG_EIGHT * i])

            tik_instance.vsel(CONFIG_MASK,
                              0,
                              filter_tensor.filter_trans_ub[
                                  CONFIG_FOUR + (i * CONFIG_DATA_TRANS), 0],
                              cmpmask,
                              filter_tensor.filter_trans_ub[
                                  CONFIG_FOUR + (i * CONFIG_DATA_TRANS), 0],
                              filter_tensor.filter_height_ub,
                              CONFIG_ONE,
                              CONFIG_EIGHT,
                              CONFIG_EIGHT,
                              CONFIG_ONE,
                              CONFIG_DATA_TRANS,
                              CONFIG_DATA_TRANS,
                              CONFIG_EIGHT)

    # trans the vector
    with tik_instance.for_range(0, filter_tensor.filter_ub_size // CONFIG_SIXTEEN) as i:
        tik_instance.vtranspose(filter_tensor.filter_ub[MATRIX * i],
                                filter_tensor.filter_trans_ub[MATRIX * i])

    tik_instance.data_move(data_tensor.proposal_post_size_filter,
                           filter_tensor.filter_ub,
                           0,
                           CONFIG_ONE,
                           filter_repeat_data,
                           0, 0)


def call_nms_processing(tik_instance, data_tensor, nms_tensor, input_param, input_dict):
    """
    perform the NMS

    tik_instance:
    data_tensor:  data_tensor.proposal_gm[6000]
    nms_tensor: tensors in DDR for NMS results
    input_param : a list of attr
    img_size,  score_threshold,  k,  min_size,   nms_threshold,
    score_filter,     box_filter,     score_sigmoid,

    input_dict:  rois[N,4], cls_bg_prob[N,1], sorted_box[M,4], sorted_scores[M,1]

    the call of nms

    input_dtype : the data type, such as "float16"
    ub_size: available ub size
    nms_threshold:  the threshold
    batch_id: here is 0
    pre_nms_topn:
    post_nms_topn: the proposal num to left post nms
    input_offset: here is 0
    image_info: height and width of image
    tik_instance:
    None : not used
    class_index: here is 0
    real_batch_index: here is 0

    temp_proposal_out: nms_tensor.nms_temp_proposal_out
    topk_output_proposal:  data_tensor.proposal_pre_nms
    topk_output_actual_proposal_num: data_tensor.actual_num_psot_topk2
    nms_actual_rois_num: nms_tensor.nms_actual_rois_num
    rois_tensor: nms_tensor.nms_proposal_gm
    used_in_proposal

    nms.cce_nms((input_dtype, ub_size,
                     nms_threshold, batch_id,
                     pre_nms_topn, post_nms_topn,
                     input_offset, image_info,
                     tik_instance, None, class_index,
                     real_batch_index),
                    temp_proposal_out,
                    topk_output_proposal,
                    topk_output_actual_proposal_num,
                    nms_actual_rois_num, rois_tensor, False)
    return:
    """

    ub_size = get_ub_size()
    nms_threshold = input_param[CONFIG_FOUR]

    pre_nms_topn = input_param[CONFIG_TWO]
    post_nms_topn = input_dict[CONFIG_TWO].get("shape")[0]

    image_info = input_param[0]

    with tik_instance.if_scope(data_tensor.actual_num_psot_topk2 > 0):
        # real rois_num
        nms_actual_rois_num = nms_tensor.nms_actual_rois_num
        rois_tensor = nms_tensor.nms_proposal_gm
        nms.cce_nms((nms_tensor.dtype, ub_size,
                     nms_threshold, 0,
                     pre_nms_topn, post_nms_topn,
                     0, image_info,
                     tik_instance, None, 0,
                     0),
                    nms_tensor.nms_temp_proposal_out,
                    data_tensor.proposal_pre_nms,
                    data_tensor.actual_num_psot_topk2,
                    nms_actual_rois_num, rois_tensor, False)
    with tik_instance.else_scope():
        with tik_instance.new_stmt_scope():
            temp_vector = tik_instance.Tensor("int32",
                                              (CONFIG_ONE, CONFIG_EIGHT),
                                              name="temp_vector",
                                              scope=tik.scope_ubuf)
            tik_instance.vector_dup(CONFIG_EIGHT,
                                    temp_vector,
                                    0,
                                    CONFIG_ONE,
                                    0, 0)
            tik_instance.data_move(nms_tensor.nms_actual_rois_num,
                                   temp_vector,
                                   0,
                                   CONFIG_ONE,
                                   CONFIG_ONE,
                                   0, 0)

        data_tensor.actual_num_psot_nms.set_as(0)
        with tik_instance.new_stmt_scope():
            temp_tensor = tik_instance.Tensor(nms_tensor.dtype,
                                              (post_nms_topn + CONFIG_FOUR, CONFIG_EIGHT),
                                              name="temp_tensor",
                                              scope=tik.scope_ubuf)
            tik_instance.data_move(temp_tensor,
                                   data_tensor.proposal_pre_nms,
                                   0,
                                   CONFIG_ONE,
                                   ceil_div(post_nms_topn, CONFIG_TWO),
                                   0,
                                   0)
            tik_instance.data_move(nms_tensor.nms_proposal_gm,
                                   temp_tensor,
                                   0,
                                   CONFIG_ONE,
                                   ceil_div(post_nms_topn, CONFIG_TWO),
                                   0,
                                   0)


def score_sigmod(tik_instance, data_tensor, nms_tensor, sigmoid_tensor):
    """
    :param tik_instance:
    :param nms_tensor:
    :param input_param:
    :param input_dict:
    :return:
    """

    # block of data to move in
    # box --> block num
    repeat_box_data = ceil_div(sigmoid_tensor.sigmoid_shape * CONFIG_FOUR, CONFIG_SIXTEEN)

    # trans times extract times
    repeat_time = ceil_div(sigmoid_tensor.sigmoid_shape, CONFIG_SIXTEEN)

    tik_instance.vector_dup(CONFIG_MASK,
                            sigmoid_tensor.sigmoid_box_ub,
                            0,
                            ceil_div(sigmoid_tensor.sigmoid_shape, CONFIG_EIGHT),
                            CONFIG_ONE,
                            CONFIG_EIGHT)

    with tik_instance.if_scope(data_tensor.actual_num_psot_nms > 0):
        tik_instance.data_move(sigmoid_tensor.sigmoid_box_ub,
                               nms_tensor.nms_proposal_gm,
                               0,
                               ceil_div(data_tensor.actual_num_psot_nms, CONFIG_TWO),
                               CONFIG_ONE,
                               0,
                               CONFIG_ONE)
    with tik_instance.if_scope(data_tensor.actual_num_psot_nms > CONFIG_ONE):
        tik_instance.data_move(sigmoid_tensor.sigmoid_box_ub[CONFIG_SIXTEEN],
                               nms_tensor.nms_proposal_gm[CONFIG_EIGHT],
                               0,
                               data_tensor.actual_num_psot_nms // CONFIG_TWO,
                               CONFIG_ONE,
                               0,
                               CONFIG_ONE)

    tik_instance.vmrgch(sigmoid_tensor.merged_box_ub,
                        sigmoid_tensor.sigmoid_box_ub,
                        repeat_time * CONFIG_TWO)

    tik_instance.data_move(data_tensor.box_gm,
                           sigmoid_tensor.merged_box_ub,
                           0,
                           CONFIG_ONE,
                           repeat_box_data,
                           0, 0)


def rpn_proposals_d_compute(input_dict,
                            input_param,
                            kernel_name):
    """
    calculating data

    Parameters
    ----------
    input_dict : a list of input dict
      rois, cls_bg_prob, sorted_box, sorted_scores

    input_param : a list of attr
    img_size,  score_threshold,  k,  min_size,   nms_threshold,
    score_filter,  box_filter,     score_sigmoid, post_nms_num

    kernel_name : str
        kernel name, default value is "generate_rpn_proposals"

    Returns
    -------
    tik_instance
    """

    # initial the tik container
    tik_instance = tik.Tik(tik.Dprofile(), disable_debug=True)
    # tik_instance = tik.Tik(tik.Dprofile(), disable_debug=False)

    #  tiling parameters for data move
    tiling_para = TilingParam(input_dict)

    # initial the tensors in DDR
    data_tensor = InitGmTensor(tik_instance, input_dict, input_param)

    # combine the boxes and prob into proposals
    combine_proposals(tik_instance, data_tensor, tiling_para)

    # perform the topk
    with tik_instance.new_stmt_scope():
        call_topk_sort(tik_instance,
                       (data_tensor.proposal_gm,
                        data_tensor.mem_swap,
                        data_tensor.proposal_post_topk,
                        data_tensor.actual_num_psot_topk1),
                       input_param,
                       tiling_para.proposal_num)

    # perform the ClipBox  and size filter
    with tik_instance.new_stmt_scope():
        filter_tensor = InitFilterTensor(tik_instance, input_param, input_dict)
        filter_with_height_weight(tik_instance,
                                  data_tensor,
                                  filter_tensor,
                                  input_param)

    # if only clipBoxes, no need to topk here
    if input_param[CONFIG_SIX]:
        with tik_instance.new_stmt_scope():
            call_topk_sort(tik_instance,
                           (data_tensor.proposal_post_size_filter,
                            data_tensor.mem_swap_second,
                            data_tensor.proposal_pre_nms,
                            data_tensor.actual_num_psot_topk2),
                           input_param,
                           input_param[CONFIG_TWO])

    # perform the nms
    nms_tensor = InitNmsTensor(tik_instance, input_dict)
    with tik_instance.new_stmt_scope():
        call_nms_processing(tik_instance,
                            data_tensor,
                            nms_tensor,
                            input_param,
                            input_dict)

    with tik_instance.new_stmt_scope():
        temp_vector = tik_instance.Tensor("int32",
                                          (CONFIG_EIGHT, CONFIG_ONE),
                                          name="temp_vector",
                                          scope=tik.scope_ubuf)
        tik_instance.data_move(temp_vector,
                               nms_tensor.nms_actual_rois_num,
                               0,
                               CONFIG_ONE,
                               CONFIG_ONE,
                               0, 0)

        data_tensor.actual_num_psot_nms.set_as(temp_vector[0])

    # score sigmoid
    with tik_instance.new_stmt_scope():
        sigmoid_tensor = InitSigmoidTensor(tik_instance, input_dict)
        score_sigmod(tik_instance, data_tensor, nms_tensor, sigmoid_tensor)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[data_tensor.rois_gm, data_tensor.prob_gm],
                          outputs=[data_tensor.box_gm])

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


def check_input_dict(rois, cls_bg_prob, sorted_box, post_nms_num):
    """
    check the input dict of generate_rpn_proposals()
    Parameters
    ----------
    rois : dict
        shape and dtype of input boxes
    cls_bg_prob : dict
        shape and dtype of input probobilities
    sorted_box: : dict
        shape and dtype of output sorted boxes
    post_nms_num: Int
        num of proposals after NMS
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

    def _check_dtype_rule_local(dtype, check_list):
        """
        The common check rule for tensor dtype
        """
        if dtype is None:
            raise RuntimeError("dtype is None")

        if dtype.lower() not in check_list:
            raise RuntimeError("Dtype only supports %s" % check_list)

    # check whether both "shape" and "dtype" is included and following the rule
    input_key = ("shape", "dtype")
    _check_input_type_dict(rois, input_key, "rois")
    _check_input_type_dict(cls_bg_prob, input_key, "cls_bg_prob")
    _check_input_type_dict(sorted_box, input_key, "sorted_box")

    # get the parameters from dicts
    input_rois_shape = rois.get("shape")
    input_rois_dtype = rois.get("dtype")
    input_prob_shape = cls_bg_prob.get("shape")
    input_prob_dtype = cls_bg_prob.get("dtype")

    output_box_shape = sorted_box.get("shape")
    output_box_dtype = sorted_box.get("dtype")

    # check the dtype
    _check_dtype_rule_local(input_rois_dtype, "float16")
    _check_dtype_rule_local(input_prob_dtype, "float16")
    _check_dtype_rule_local(output_box_dtype, "float16")

    # check the shape
    util.check_shape_rule(input_rois_shape,
                          min_dim=CONFIG_TWO,
                          max_dim=CONFIG_TWO,
                          max_shape_num=SHAPE_SIZE_LIMIT)
    util.check_shape_rule(input_prob_shape,
                          min_dim=CONFIG_ONE,
                          max_dim=CONFIG_TWO,
                          max_shape_num=SHAPE_SIZE_LIMIT)
    util.check_shape_rule(output_box_shape,
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
    _check_shape_size_limit(output_box_shape,
                            "sorted_box",
                            shape_para=CONFIG_FOUR,
                            output_flag=True)

    if input_rois_shape[0] != input_prob_shape[0]:
        raise RuntimeError("n dimension of inputs rois and cls_bg_prob should be consistent")

    if post_nms_num != output_box_shape[0]:
        raise RuntimeError("post_nms_num should be consistent with"
                           " n dimension of inputs sorted_box")


@util.check_input_type(dict, dict, dict,
                       (tuple, list), (float, int), int,
                       (float, int), (float, int), int,
                       bool, bool, bool, str)
def rpn_proposals_d(rois, cls_bg_prob, sorted_box,
                    img_size, score_threshold, k, min_size,
                    nms_threshold, post_nms_num,
                    score_filter=True, box_filter=True,
                    score_sigmoid=False,
                    kernel_name="generate_rpn_proposals"):
    """
    the entry function of generate_rpn_proposals
    Parameters
    ----------
    rois : dict
        shape and dtype of input boxes
    cls_bg_prob : dict
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
    box_filter: bool,   True
    score_sigmoid: bool, False
    kernel_name : str
        kernel name, default value is "generate_rpn_proposals"
    Returns
    -------
    None
    """
    # input check
    check_input_dict(rois, cls_bg_prob, sorted_box, post_nms_num)
    check_input_param((img_size, score_threshold, k, min_size,
                       nms_threshold), kernel_name)

    if score_threshold < 0:
        raise RuntimeError("score_threshold should be large than 0!")

    if k == 0:
        if rois.get("shape")[0] > MAX_TOPK:
            k = MAX_TOPK
        else:
            k = rois.get("shape")[0]

    if tik.Dprofile().get_product_name() not in IF_USE_V200:
        tik_instance = rpn_proposals_d_compute((rois, cls_bg_prob, sorted_box),
                                               (img_size, score_threshold, k, min_size,
                                                nms_threshold, score_filter, box_filter,
                                                score_sigmoid, post_nms_num),
                                               kernel_name)
    else:
        tik_instance = rpn_v200.rpn_proposals_d_compute_v200((rois, cls_bg_prob, sorted_box),
                                                             (img_size, score_threshold,
                                                              k, min_size,
                                                              nms_threshold, score_filter,
                                                              box_filter, score_sigmoid),
                                                             kernel_name)


    return tik_instance
