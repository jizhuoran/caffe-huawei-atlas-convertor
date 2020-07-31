"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     common_util.py
DESC:     common util file for tik api and debug
CREATED:  2020-01-10 19:02:50
MODIFIED: 2020-01-10 19:02:50
"""
# disabling:
# C0302: too-many-lines (this file is full of data operation instructions)
# R0913: too-many-arguments
# E1101: no-member
# R0914: too-many-locals
# R0904: too-many-public-methods

import numpy as np  # pylint: disable=C0302

from te.platform.cce_params import HI3796CV300ESAIC
from te.platform.cce_params import AIC
from te.tik.tik_lib.tik_basic_data import BasicData
from te.tik.tik_lib.tik_expr import Expr, BasicExpr
from te import tvm
from .util import get_bit_len, TikUtil, reduce_mul, ceil_div, \
    check_integer_in_range, get_mask_len, DTYPE_SIZE, is_basic_expr, \
    check_scalar_dtype, is_immediate_number, check_imme_mask_full_mode
from ..tik_lib.tik_params import ONE_REP_BYTE_SIZE, BLK_NUM_PER_REP, \
    VDEQ_BLK_STRIDE, VDEQ_REP_STRIDE, ONE_BYTE_BIT_LEN, \
    MASK_LEN_CONTINOUS_MODE, MASK_HIGH_IDX, MIN_CHANNEL_SIZE, MAX_CHANNEL_SIZE,\
    PADDING_LEFT_IDX, PADDING_RIGHT_IDX, PADDING_TOP_IDX, PADDING_BOT_IDX, \
    ELE_PER_FRACTAL_EDGE, ONE_BLK_SIZE, MAX_REP_STRIDE_DOUBLE_BYTE, \
    MAX_REP_STRIDE_SINGLE_BYTE, MASK_VALUE_ZERO, BIT_LEN_32, BIT_LEN_16, \
    BIT_LEN_8, MAX_BLK_STRIDE_DOUBLE_BYTE, MAX_BLK_STRIDE_SINGLE_BYTE, \
    MIN_INDEX, MAX_INDEX, CONST_MASK_VALUE, MASK_LEN_128, MASK_LOW_IDX, \
    VALUE_4, VALUE_3, VALUE_1, VALUE_4096, VALUE_256, VALUE_128, MASK_LEN_64, \
    BLK_16_LIST, BLK_32_LIST, AIPP_INPUT_TYPE_SWAP_ALIGN, CROP_BIT, SWAP_BIT, \
    CSC_BIT, DTC_BIT, AREA_PAD_BIT, CPAD_BIT, PRE_CLIP_BIT, SCF_BIT, \
    POST_CLIP_BIT, FLIP_BIT, STRETCH, RAW_BIT, REAL_SCALE_MIN, REAL_SCALE_MAX, \
    MIN_VNCHWTRANS_STRIDE, MAX_VNCHWTRANS_STRIDE, MAX_VNCHWTRANS_REPEAT_TIMES, \
    MIN_REPEAT_TIMES, PER_TRANSPOSE_DATA_SIZE, MAX_REPEAT_TIMES, \
    MAX_VREDUCE_REPEAT_TIMES, MIN_L1_H_W_C, MAX_L1_H_W, MAX_L1_C, \
    MAX_DST_GAP_WINO, MAX_EXTENSION, MAX_K_WINO, MAX_COL_INDIC, MAX_START_PT, \
    MIN_MASK, MAX_MASK, MAX_MASK_64, MASK_LEN_FULL_MODE
from ..tik_lib.tik_check_util import TikCheckUtil
from ..tik_lib.tik_api_constants import WINO_PAD_MAP


_BYTE_PER_C0 = 32


def check_vconv_deqs162b8_overflow(src, dst, deqscale, mask, repeat_times,
                                   ldst_high_half, blk_stride_list,
                                   rep_stride_list, stride_unit):
    """for vconv deqs162b8 and vdeqs162b8 mode, check tensor overflow

    Parameters
    ----------
    src : source operation
    dst : destination operator
    deqscale : default None
    mask : Effective operation on element
    repeat_times : Repeated iterations times
    ldst_high_half : default false
    blk_stride_list : list of dst_blk_stride and src_blk_stride
    rep_stride_list : list of dst_rep_stride and src_rep_stride
    stride_unit : address and offset unit both affect it. default = 0

    Returns
    -------
    None
    """
    # pylint: disable=R0913, R0914
    # function's input params is too much, so disable them
    dst_blk_stride, src_blk_stride = blk_stride_list
    dst_rep_stride, src_rep_stride = rep_stride_list
    src_block_len = ONE_REP_BYTE_SIZE // get_bit_len(src.dtype)
    # src's bit_len is twice as much as dst's bit_len
    src_dst_len_ratio = get_bit_len(src.dtype) // get_bit_len(dst.dtype)
    if isinstance(deqscale, BasicData) and deqscale.is_tensor() is True:
        vector_tensor_overflow_check(
            deqscale, mask, BLK_NUM_PER_REP, src_block_len, repeat_times,
            VDEQ_BLK_STRIDE, VDEQ_REP_STRIDE, "deqscale tensor overflow")
    vector_tensor_overflow_check(
        src, mask, BLK_NUM_PER_REP, src_block_len, repeat_times, src_blk_stride,
        src_rep_stride, "src tensor overflow", stride_unit)
    # ldst_high_half is False, dst is in the first half for each block
    if not ldst_high_half:
        vector_tensor_overflow_check(
            dst, mask, BLK_NUM_PER_REP, src_block_len, repeat_times,
            dst_blk_stride*src_dst_len_ratio, dst_rep_stride*src_dst_len_ratio,
            "dst tensor overflow", stride_unit)
    # ldst_high_half is True, dst is in the second half for each block
    else:
        if is_basic_expr(TikUtil.to_list(mask)) or \
                is_basic_expr([repeat_times]):
            return
        if repeat_times == 0:
            return
        offset = dst.offset + src_block_len
        total_size = reduce_mul(dst.indice.origin_shape)
        extend_offset = vector_max_offset_cal(
            mask, dst.dtype, src_block_len, repeat_times,
            dst_blk_stride*src_dst_len_ratio, dst_rep_stride*src_dst_len_ratio,
            stride_unit)
        if Expr(extend_offset + offset).eval_value() is not None:
            TikCheckUtil.check_le(
                Expr(extend_offset + offset).eval_value(), total_size,
                "dst tensor overflow, instruction need {} but only {}"
                .format(Expr(extend_offset + offset).eval_value(), total_size))


def check_vconv_deqs162b8_overlap(  # pylint: disable=R0913
        src, dst, deqscale, mask, repeat_times,
        ldst_high_half, blk_stride_list, rep_stride_list, stride_unit):
    """vconv deqs162b8 and vdeqs162b8 mode, check tensor address overlap"""
    dst_blk_stride, src_blk_stride = blk_stride_list
    dst_rep_stride, src_rep_stride = rep_stride_list
    # src's bit_len is twice as much as dst's bit_len
    src_dst_len_ratio = get_bit_len(src.dtype) // get_bit_len(dst.dtype)
    if all(isinstance(value, int) for \
           value in (repeat_times, dst_blk_stride, dst_rep_stride,
                     src_blk_stride, src_rep_stride, stride_unit)):
        if not ldst_high_half:
            # check address overlap
            _check_dst_deqscale_src_overlap(
                src, dst, deqscale, mask, repeat_times,
                Expr(dst.offset).eval_value(), src_dst_len_ratio,
                ONE_REP_BYTE_SIZE // get_bit_len(src.dtype),
                dst_blk_stride, src_blk_stride,
                dst_rep_stride, src_rep_stride, stride_unit)
        else:
            dst_offset = Expr(dst.offset).eval_value()
            if dst_offset is not None:
                dst_offset += ONE_REP_BYTE_SIZE // get_bit_len(src.dtype)
                _check_dst_deqscale_src_overlap(
                    src, dst, deqscale, mask, repeat_times,
                    dst_offset, src_dst_len_ratio,
                    ONE_REP_BYTE_SIZE // get_bit_len(src.dtype),
                    dst_blk_stride, src_blk_stride,
                    dst_rep_stride, src_rep_stride, stride_unit)


def _check_dst_deqscale_src_overlap(  # pylint: disable=R0913
        src, dst, deqscale, mask, repeat_times, dst_offset,
        src_dst_len_ratio, src_block_len, dst_blk_stride, src_blk_stride,
        dst_rep_stride, src_rep_stride, stride_unit):
    """check vconv dst address overlap among src and deqscale"""
    # check address overlap
    if isinstance(deqscale, BasicData) and deqscale.is_tensor() is True:
        if deqscale.buffer == dst.buffer:
            check_address_overlapping(
                "vconv", mask, dst, deqscale, BLK_NUM_PER_REP,
                src_block_len, src_block_len, src_block_len,
                repeat_times, dst_blk_stride*src_dst_len_ratio,
                VDEQ_BLK_STRIDE, dst_rep_stride*src_dst_len_ratio,
                VDEQ_REP_STRIDE, dst_offset,
                Expr(deqscale.offset).eval_value(),
                stride_unit, msg="dst and deqscale")

    if src.buffer == dst.buffer:
        check_address_overlapping(
            "vconv", mask, dst, src, BLK_NUM_PER_REP,
            src_block_len, src_block_len, src_block_len,
            repeat_times, dst_blk_stride*src_dst_len_ratio,
            src_blk_stride, dst_rep_stride*src_dst_len_ratio,
            src_rep_stride, dst_offset,
            Expr(src.offset).eval_value(), stride_unit)


def check_sel_overflow(dst, src0, sel, mask, repeat_times):
    """vsel instruction, check sel tensor overflow when mode is 1 or 2

    Parameters
    ----------
    dst : destination operator
    src0 : source operator
    sel : sel operator, when mode is 1 or 2, should be Tensor
    mask : Effective operation on element
    repeat_times : Repeated iterations times

    Returns
    -------
    None
    """
    sel_bit_len = get_bit_len(sel.dtype)
    parallelism = ONE_REP_BYTE_SIZE*ONE_BYTE_BIT_LEN // \
                  max(get_bit_len(dst.dtype), get_bit_len(src0.dtype))
    offset = sel.offset
    total_size = reduce_mul(sel.indice.origin_shape)
    sel_num_each_repeat = parallelism // sel_bit_len
    mask_list = TikUtil.to_list(mask)
    if not is_basic_expr(mask_list):
        if len(mask_list) == MASK_LEN_CONTINOUS_MODE:
            mask_len = mask_list[MASK_HIGH_IDX]
        else:
            mask_len, _ = get_mask_len(mask_list)
        extend_offset = (repeat_times - 1)*sel_num_each_repeat + \
                        ceil_div(mask_len, sel_bit_len)
        if Expr(extend_offset + offset).eval_value() is not None:
            TikCheckUtil.check_le(
                Expr(extend_offset + offset).eval_value(),
                total_size,
                "sel tensor overflow, expected elements: {}, actual elements: "
                "{}".format(Expr(extend_offset + offset).eval_value(),
                            total_size))


def _check_single_it_sel_dst(  # pylint: disable=R0913
        dst, sel, mask, mask_list, dst_blk_stride,
        dst_rep_stride, dst_offset, sel_offset):
    """check single iteration address overlap"""
    repeat_times = 1
    if len(mask_list) == 1:
        mask_len = mask_list[0]
    else:
        mask_len, _ = get_mask_len(mask_list)
    sel_extend = ceil_div(mask_len, get_bit_len(sel.dtype))
    dst_extend = vector_max_offset_cal(
        mask, dst.dtype, ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
        repeat_times, dst_blk_stride, dst_rep_stride)
    sel_need_offset = Expr(sel_offset + sel_extend).eval_value()
    dst_need_offset = Expr(dst_offset + dst_extend).eval_value()
    if sel_offset and dst_need_offset is not None:
        if dst_offset == sel_offset and dst_blk_stride == 1:
            pass
        elif sel_offset <= sel_need_offset <= dst_offset or \
                dst_offset <= dst_need_offset <= sel_offset:
            pass
        else:
            TikCheckUtil.raise_error(
                "when repeat_times=1, vsel dst and sel"
                " not support partially address overlapping.")


def check_sel_dst_overlap(dst, src0, sel, mask,  # pylint: disable=R0913
                          dst_offset, sel_offset,
                          repeat_times, dst_blk_stride, dst_rep_stride):
    """check vsel dst and sel address overlap"""
    # function's input params is too much, so disable them
    if repeat_times <= 0:
        return
    if dst_offset is None or sel_offset is None:
        return
    parallelism = ONE_REP_BYTE_SIZE*ONE_BYTE_BIT_LEN // \
                  max(get_bit_len(dst.dtype), get_bit_len(src0.dtype))
    sel_num_each_repeat = parallelism // get_bit_len(sel.dtype)
    mask_list = TikUtil.to_list(mask)
    if not is_basic_expr(mask_list):
        if repeat_times == 1:
            _check_single_it_sel_dst(
                dst, sel, mask, mask_list, dst_blk_stride,
                dst_rep_stride, dst_offset, sel_offset)
        else:
            dst_block_len = ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype)
            for time in range(repeat_times - 1):
                _check_addr_overlap(
                    dst, mask_list, dst_blk_stride,
                    dst_rep_stride, dst_block_len,
                    sel_num_each_repeat, dst_offset, sel_offset,
                    time, sel.dtype)


def _check_addr_overlap(dst, mask_list, dst_blk_stride,  # pylint: disable=R0913
                        dst_rep_stride, dst_block_len,
                        sel_num_each_repeat, dst_offset, sel_offset,
                        time, sel_dtype):
    """check sel dst overlap"""
    dst_blk_valid_list = get_blk_valid_list(
        mask_list, dst.dtype, dst_block_len)
    dst_extend_interval = get_extend_interval(
        dst_blk_valid_list, dst_block_len, time,
        dst_blk_stride, dst_rep_stride,
        stride_unit=0, offset=dst_offset, dtype=dst.dtype)
    sel_begin = (time + 1)*sel_num_each_repeat + sel_offset
    sel_end = sel_begin + sel_num_each_repeat
    interval_sel = [sel_begin*DTYPE_SIZE[sel_dtype],
                    sel_end*DTYPE_SIZE[sel_dtype]]
    # check
    for interval_dst in dst_extend_interval:
        if max(interval_sel[0], interval_dst[0]) < \
                min(interval_sel[1], interval_dst[1]):
            TikCheckUtil.raise_error(
                "When repeat_times>1, vsel dst and sel address"
                " overlapping error. It is not support "
                "iteration N's destination is the source"
                " of next iteration")


def check_load3dv2_channel_size(channel_size, src_dtype):
    """ check channel_size

    Parameters
    ----------
    channel_size: input param channel_size of instruction load3dv2
    src_dtype: dtype of src, int8/uint8/float16

    Returns
    -------
    None
    """
    check_integer_in_range(
        channel_size, range(MIN_CHANNEL_SIZE, MAX_CHANNEL_SIZE),
        "channel_size should be in the range of [1, 63], "
        "input channel_size: {}".format(channel_size))
    if Expr(channel_size).eval_value() is not None:
        # ISA v7.9.3
        # channel_size_f16 mod 16 should be 0, 4 or 8
        # channel_size_b8 mod 32 should be 0, 4, 8, 16
        if src_dtype == "float16":
            TikCheckUtil.check_in_range(
                channel_size % (_BYTE_PER_C0 // DTYPE_SIZE[src_dtype]),
                (0, 4, 8),
                "for f16 src tensor, channel_size mod 16 should be 0, 4, or"
                " 8, input channel_size: {}".format(channel_size))
        else:
            TikCheckUtil.check_in_range(
                channel_size % (_BYTE_PER_C0 // DTYPE_SIZE[src_dtype]),
                (0, 4, 8, 16),
                "for b8 src tensor, channel_size mod 32 should be 0, 4, 8 "
                "or 16, input channel_size: {}".format(channel_size))


def check_load3dv2_m_extension(filter_w, filter_h, dilation_filter_w,
                               dilation_filter_h, pad_list, m_extension, l1_w,
                               l1_h, stride_w, stride_h, m_start_pt):
    """ check load3dv2 m_extension, m_start_pt

    Parameters
    ----------
    filter_w: width of filter
    filter_h: height of filter
    dilation_filter_w: dilation size of filter in w dimension
    dilation_filter_h: dilation size of filter in h dimension
    pad_list: [left, right, top, bottom]
    m_extension: m direction extension steps from the start position
    l1_w: width of src tensor
    l1_h: height of src tensor
    stride_w: filter stride size in w dimension
    stride_h: filter stride size in h dimension
    m_start_pt: m direction start position of the feature matrix

    Returns
    -------
    None
    """
    # pylint: disable=R0913, R0914
    # function's input params is too much, so disable them
    dlt_w = (filter_w - 1)*dilation_filter_w + 1
    dlt_h = (filter_h - 1)*dilation_filter_h + 1

    w_o = (l1_w + pad_list[PADDING_LEFT_IDX] +
           pad_list[PADDING_RIGHT_IDX] - dlt_w) // stride_w + 1
    h_o = (l1_h + pad_list[PADDING_TOP_IDX] +
           pad_list[PADDING_BOT_IDX] - dlt_h) // stride_h + 1

    # expected m should be multiple of 16
    max_m = w_o*h_o
    expected_m = ceil_div(m_extension + m_start_pt, ELE_PER_FRACTAL_EDGE)*\
                 ELE_PER_FRACTAL_EDGE
    max_m = Expr(max_m).eval_value()
    expected_m = Expr(expected_m).eval_value()
    if max_m is not None and expected_m is not None:
        TikCheckUtil.check_ge(
            max_m, expected_m,
            "m_extension({}) + m_start_pt({}) round up to multiple of 16 is "
            "larger than expected maximum m extension: {}"
            .format(m_extension, m_start_pt, max_m))


def check_load3dv2_k_extension(channel_size, k_extension, filter_h, filter_w,
                               k_start_pt, src_dtype):
    """ check load3dv2 k_extension, k_start_pt

    Parameters
    ----------
    channel_size: input param k_extension of instruction load3dv2
    k_extension: k direction extension steps from the start position
    filter_h: height of filter
    filter_w: width of filter
    k_start_pt: k direction start position of the feature matrix
    src_dtype: dtype of src, float16/int8/uint8

    Returns
    -------
    None
    """
    # pylint: disable=R0913
    # function's input params is too much, so disable them
    max_k = channel_size*filter_h*filter_w
    # expected k should be multiple of 32B
    expected_k = ceil_div((k_extension + k_start_pt)*DTYPE_SIZE[src_dtype],
                          ONE_BLK_SIZE)*ONE_BLK_SIZE // DTYPE_SIZE[src_dtype]
    max_k = Expr(max_k).eval_value()
    expected_k = Expr(expected_k).eval_value()
    if max_k is not None and expected_k is not None:
        TikCheckUtil.check_ge(
            max_k, expected_k,
            "k_extension({}) + k_start_pt({}) in Byte round up to multiple of "
            "32B is larger than expected maximum k extension: {}, src dtype: {}"
            .format(k_extension, k_start_pt, max_k, src_dtype))


def check_dilation_filter_size(filter_len, dlt_filter_coef, src_len, pad_0,
                               pad_1, dimension):
    """check dilation filter size, should be smaller than l1 tensor size after
    padding

    Parameters
    ----------
    filter_len: filter_h/w
    dlt_filter_coef: dilation_filter_h/w
    src_len: l1_h/W
    pad_0: pad_top for H dimension, pad_left for W dimension
    pad_1: pad_bottom for H dimension, pad_right for W dimension
    dimension: H/W dimension, dtype:str

    Returns
    -------
    None
    """
    # pylint: disable=R0913
    # function's input params is too much, so disable them
    dlt_filter_len = (filter_len - 1)*dlt_filter_coef + 1
    pad_src_len = src_len + pad_0 + pad_1
    dlt_filter_len = Expr(dlt_filter_len).eval_value()
    pad_src_len = Expr(pad_src_len).eval_value()
    if dlt_filter_len is not None and pad_src_len is not None:
        TikCheckUtil.check_le(
            dlt_filter_len, pad_src_len,
            "in {} dimension, feature map size(after_padding): {} should be "
            "larger than filter size(after dilation): {}"
            .format(dimension, pad_src_len, dlt_filter_len))


def check_vreduce_src1_overflow(mask, mask_mode, src1_rep_stride, src1_pattern,
                                src1_actual_ele, repeat_times):
    """check vreduce src1_pattern overflow

    Parameters
    ----------
    mask :
    mask_mode : normal/counter
    src1_rep_stride : decide whether src1 is read continuously
    src1_pattern : src1 tensor
    src1_actual_ele : src1 tensor elements
    repeat_times: iterating times

    Returns
    -------
    None
    """
    # pylint: disable=R0913
    # function's input params is too much, so disable them
    if Expr(src1_rep_stride).eval_value() is None:
        return
    if mask_mode == "normal":
        if src1_rep_stride == 0:
            src1_expected_ele = ONE_REP_BYTE_SIZE // \
                                DTYPE_SIZE[src1_pattern.dtype] // \
                                get_bit_len(src1_pattern.dtype) + \
                                src1_pattern.offset
        else:
            if Expr(repeat_times).eval_value() is None:
                return
            src1_expected_ele = repeat_times*ONE_REP_BYTE_SIZE // \
                                DTYPE_SIZE[src1_pattern.dtype] // \
                                get_bit_len(src1_pattern.dtype) + \
                                src1_pattern.offset
    else:
        if Expr(mask).eval_value() is None:
            return
        if src1_rep_stride == 0:
            src1_expected_ele = min(
                ceil_div(mask, get_bit_len(src1_pattern.dtype)),
                ONE_REP_BYTE_SIZE // DTYPE_SIZE[src1_pattern.dtype] //
                get_bit_len(src1_pattern.dtype)) + src1_pattern.offset
        else:
            src1_expected_ele = ceil_div(mask, get_bit_len(src1_pattern.dtype))\
                                + src1_pattern.offset

    src1_expected_ele = Expr(src1_expected_ele).eval_value()
    TikCheckUtil.check_ge(
        src1_actual_ele, src1_expected_ele,
        "src1_pattern tensor overflow, expected shape: {}, actual "
        "shape: {}".format(src1_expected_ele, src1_actual_ele))


def get_dst_offset_vreduce(mask_mode, src1_pattern, repeat_times, dst, mask):
    """get dst need offset"""
    if mask_mode == "normal":
        # dst
        if not (isinstance(src1_pattern, BasicData) and
                src1_pattern.is_tensor()) and \
                all(Expr(value).eval_value() is not None
                    for value in (src1_pattern, repeat_times)):
            # src1_pattern is 1~2, extract odd/even element of src0
            if src1_pattern < 3:
                dst_expected_size = repeat_times*ONE_REP_BYTE_SIZE // \
                                    DTYPE_SIZE[dst.dtype] // 2
            # src1_pattern is 3~6, extract 1/4 elements of src0
            else:
                dst_expected_size = repeat_times*ONE_REP_BYTE_SIZE // \
                                    DTYPE_SIZE[dst.dtype] // 4
            return dst_expected_size
    else:
        # dst
        if not (isinstance(src1_pattern, BasicData) and
                src1_pattern.is_tensor()) and \
                all(Expr(value).eval_value() is not None
                    for value in (src1_pattern, mask, dst.offset)):
            # src1_pattern is 1~2, extract odd/even element of src0
            if src1_pattern < 3:
                dst_expected_size = Expr(ceil_div(mask, 2) + dst.offset) \
                    .eval_value()
            # src1_pattern is 3~6, extract 1/4 elements of src0
            else:
                dst_expected_size = Expr(ceil_div(mask, 4) + dst.offset) \
                    .eval_value()
            return dst_expected_size
    return None


def get_src0_offset_vreduce(tensor, mask, nblock,  # pylint: disable=R0913
                            block_len, repeat, stride1, stride2,
                            stride_unit=0, mask_mode="normal"):
    """ get src0 need offset"""
    if mask_mode == "counter":
        TikCheckUtil.check_type_match(
            mask, int, "mask here should be int, input type of "
                       "mask: {}".format(type(mask)))
        repeat = ceil_div(mask, nblock*block_len)
        mask = mask % (nblock*block_len)
        if mask == MASK_VALUE_ZERO:
            mask = nblock*block_len
    offset = tensor.offset
    extend_offset = vector_max_offset_cal(
        mask, tensor.dtype, block_len, repeat, stride1, stride2, stride_unit)
    need_offset = Expr(extend_offset + offset).eval_value()
    return need_offset


def _check_many_it_vreduce(  # pylint: disable=R0913
        dst, src, src1_pattern, mask,
        repeat, block_len, src_blk_stride,
        src_rep_stride, src_offset, dst_offset, stride_unit):
    """check N(>1) iteration address overlap for vreduce"""
    if not isinstance(mask, (list, tuple)):
        mask = [mask]
    src_blk_valid_list = get_blk_valid_list(mask, src.dtype, block_len)
    for time in range(repeat - 1):
        src_extend_interval = get_extend_interval(
            src_blk_valid_list, block_len, time + 1,
            src_blk_stride, src_rep_stride, stride_unit,
            src_offset, src.dtype)
        # check
        _check_vreduce_overlap(dst, src1_pattern,
                               dst_offset, src_extend_interval, time)


def _check_vreduce_overlap(dst, src1_pattern,
                           dst_offset, src_extend_interval, time):
    """check many iteration for vreduce"""
    # get dst_extend_interval
    # src1_pattern is 1~2, extract odd/even element of src0
    if src1_pattern < 3:
        begin = time*ONE_REP_BYTE_SIZE // \
                DTYPE_SIZE[dst.dtype] // 2 + dst_offset
        end = begin + ONE_REP_BYTE_SIZE // DTYPE_SIZE[dst.dtype] // 2
    # src1_pattern is 3~6, extract 1/4 elements of src0
    else:
        begin = time * ONE_REP_BYTE_SIZE // \
                DTYPE_SIZE[dst.dtype] // 4 + dst_offset
        end = begin + ONE_REP_BYTE_SIZE // DTYPE_SIZE[dst.dtype] // 4

    interval_dst = [begin*DTYPE_SIZE[dst.dtype], end*DTYPE_SIZE[dst.dtype]]
    # check
    for interval_src in src_extend_interval:
        if max(interval_src[0], interval_dst[0]) < \
                min(interval_src[1], interval_dst[1]):
            TikCheckUtil.raise_error(
                "When repeat_times>1, vreduce dst and src0 address"
                " overlapping error. It is not support "
                "iteration N's destination is the source"
                " of next iteration")


def _check_overlap_param(mask, repeat, dst_offset, src_offset):
    """get a flag of address overlap check"""
    is_check = True
    if repeat <= 0:
        is_check = False
    if dst_offset is None or src_offset is None:
        is_check = False
    if is_basic_expr(TikUtil.to_list(mask)):
        is_check = False
    return is_check


def check_address_overlap_vreduce(dst, src,  # pylint: disable=R0913, R0914
                                  src1_pattern, mask, nblock,
                                  block_len, repeat, src_blk_stride,
                                  src_rep_stride, dst_offset, src_offset,
                                  stride_unit=0, mask_mode="normal"):
    """check address overlapping for vreduce"""
    if not _check_overlap_param(mask, repeat, dst_offset, src_offset):
        return
    src_extend = get_src0_offset_vreduce(
        src, mask, nblock, block_len, repeat,
        src_blk_stride, src_rep_stride, stride_unit, mask_mode)
    dst_extend = get_dst_offset_vreduce(
        mask_mode, src1_pattern, repeat, dst, mask)
    src_need_offset = Expr(src_offset + src_extend).eval_value()
    dst_need_offset = Expr(dst_offset + dst_extend).eval_value()
    if mask_mode == "counter":
        repeat = ceil_div(mask, nblock*block_len)
        mask = mask % (nblock*block_len)
        if mask == MASK_VALUE_ZERO:
            mask = nblock*block_len
    if repeat == 1:
        if (src_offset == dst_offset and src_blk_stride == 1) or \
                src_offset <= src_need_offset <= dst_offset or \
                dst_offset <= dst_need_offset <= src_offset:
            pass
        else:
            TikCheckUtil.raise_error(
                "vreduce address overlapping error. "
                "when repeat_times=1, dst and src0"
                " should be 100% the same.")
    else:
        _check_many_it_vreduce(
            dst, src, src1_pattern, mask,
            repeat, block_len, src_blk_stride,
            src_rep_stride, src_offset, dst_offset, stride_unit)


def check_addr_overlap_v4dtrans(dst, src, m_len,  # pylint: disable=R0913
                                channels, dst_offset, src_offset):
    """check address overlap for v4dtrans"""
    if dst_offset is None or src_offset is None:
        return
    if src.buffer == dst.buffer:
        need_ele = m_len*channels
        total_repeat_size = need_ele*get_bit_len(src.dtype) // \
                            ONE_BYTE_BIT_LEN
        src_need_offset = Expr(src_offset + need_ele).eval_value()
        dst_need_offset = Expr(dst_offset + need_ele).eval_value()
        if total_repeat_size > ONE_REP_BYTE_SIZE:
            if src_offset <= src_need_offset <= dst_offset or \
                    dst_offset <= dst_need_offset <= src_offset:
                pass
            else:
                TikCheckUtil.raise_error(
                    "when m_len*channels*dtype*size>256B, v4dtrans"
                    " dst, src not support address overlapping.")
        else:
            if src_offset <= src_need_offset <= dst_offset or \
                    dst_offset <= dst_need_offset <= src_offset or \
                    src_offset == dst_offset:
                pass
            else:
                TikCheckUtil.raise_error(
                    "when m_len*channels*dtype*size<=256B, "
                    "v4dtrans dst, src address overlap error,"
                    " only support 100% same.")


def check_vector_stride(blk_stride_list, rep_stride_list, blk_stride_range,
                        rep_stride_range, name, is_scatter=False):
    """check blk_stride and rep_stride params of vector instructions

    Parameters
    ----------
    blk_stride_list : list of dst_blk_stride and src_blk_stride
    rep_stride_list : list of dst_rep_stride and src_rep_stride
    blk_stride_range : upper bound of blk_stride
    rep_stride_range : upper bound of rep_stride
    name : list of tensor name

    Returns
    -------
    None
    """
    # pylint: disable=R0913
    # function's input params is too much, so disable them
    if rep_stride_list is not None:
        TikCheckUtil.check_type_match(rep_stride_list, (list, tuple),
                                      "rep_stride_list should be list or tuple")
        TikCheckUtil.check_equality(len(rep_stride_list), len(name),
                                    "rep_stride_list and name should contains "
                                    "same number of elements")
        TikCheckUtil.check_in_range(rep_stride_range,
                                    (MAX_REP_STRIDE_DOUBLE_BYTE,
                                     MAX_REP_STRIDE_SINGLE_BYTE),
                                    "rep_stride_range not support!")
        for i, rep_stride in enumerate(rep_stride_list):
            TikCheckUtil.check_type_match(rep_stride, (int, BasicExpr),
                                          "{}_rep_stride should be int, "
                                          "Expr or Scalar, input "
                                          "type is {}".format(name[i],
                                                              type(rep_stride)))
            check_scalar_dtype(rep_stride,
                               "scalar_{}_rep_stride should be "
                               "a scalar of int/uint".format(name[i]))
            check_integer_in_range(
                rep_stride, range(rep_stride_range),
                "{}_rep_stride should be in the range of [0, {}], "
                "input value is {}".format(name[i],
                                           rep_stride_range - 1, rep_stride))
    if not is_scatter and blk_stride_list is not None:
        TikCheckUtil.check_type_match(blk_stride_list, (list, tuple),
                                      "blk_stride_list should be list  or tuple")
        TikCheckUtil.check_equality(len(blk_stride_list), len(name),
                                    "blk_stride_list and name should contains "
                                    "same number of elements")
        TikCheckUtil.check_in_range(blk_stride_range,
                                    (MAX_BLK_STRIDE_DOUBLE_BYTE,
                                     MAX_BLK_STRIDE_SINGLE_BYTE),
                                    "blk_stride_range not support!")
        for i, blk_stride in enumerate(blk_stride_list):
            TikCheckUtil.check_type_match(
                blk_stride, (int, BasicExpr),
                "{}_blk_stride should be int, Expr or Scalar, input type is "
                "{}".format(name[i], type(blk_stride)))
            check_scalar_dtype(blk_stride,
                               "scalar_{}_blk_stride should be "
                               "a scalar of int/uint".format(name[i]))
            check_integer_in_range(
                blk_stride, range(blk_stride_range),
                "{}_blk_stride should be in the range of [0, {}], input value"
                " is {}".format(name[i], blk_stride_range - 1, blk_stride))


def check_tensor_overflow(tensor_list, mask, repeat_times, blk_stride_list,
                          rep_stride_list, tensor_name_list, stride_unit=0,
                          mask_mode="normal"):
    """check tensor overflow

    Parameters
    ----------
    tensor_list
    mask : Effective operation on element,
           divided into two model: Continuous and bit by bit
    repeat_times : Repeated iterations times
    blk_stride_list : offset of src operator between different block in
                     one repeat
    rep_stride_list : offset of src operator in the same block between
                     two repeats

    Returns
    -------
    None
    """
    # pylint: disable=R0913
    # function's input params is too much, so disable them
    bit_len = []
    for tensor_in in tensor_list:
        tensor_bit_len = get_bit_len(tensor_in.dtype)
        bit_len.append(tensor_bit_len)
    parallelism = ONE_REP_BYTE_SIZE*ONE_BYTE_BIT_LEN // max(bit_len)
    for tensor_in, blk_stride, rep_stride, tensor_bit_len, name in \
            zip(tensor_list, blk_stride_list, rep_stride_list, bit_len,
                    tensor_name_list):
        vector_tensor_overflow_check(
            tensor_in, mask,
            parallelism // (ONE_REP_BYTE_SIZE // tensor_bit_len),
            ONE_REP_BYTE_SIZE // tensor_bit_len, repeat_times, blk_stride,
            rep_stride, "{} tensor overflow".format(name), stride_unit,
            mask_mode)


def vector_tensor_overflow_check(tensor, mask,  # pylint: disable=R0914
                                 nblock, block_len, repeat,
                                 stride1, stride2, msg="tensor overflow",
                                 stride_unit=0, mask_mode="normal",
                                 ori_offset=0):
    """check overflow vector tensor
    :param tensor: tensor object
    :param mask: instr's mask
    :param nblock: effective block nums in one repeat
    :param block_len: number of elements in one block
    :param repeat: the times of instr run
    :param stride1: block stride
    :param stride2: repeat stride
    :param msg: error msg
    """
    # pylint: disable=R0913
    if is_basic_expr(TikUtil.to_list(mask)) or is_basic_expr([repeat]):
        return
    if repeat == 0:
        return
    if not isinstance(nblock, int):
        return
    if mask_mode == "counter":
        TikCheckUtil.check_type_match(
            mask, int, "mask here should be int, "
                       "input type of mask: {}".format(type(mask)))
        repeat = ceil_div(mask, nblock*block_len)
        mask = mask % (nblock*block_len)
        if mask == MASK_VALUE_ZERO:
            mask = nblock*block_len
    offset = tensor.offset
    if isinstance(offset, (tvm.expr.IntImm, tvm.expr.FloatImm,
                           tvm.expr.StringImm, tvm.expr.UIntImm)):
        offset = offset.value
    total_size = reduce_mul(tensor.indice.origin_shape)
    extend_offset = vector_max_offset_cal(
        mask, tensor.dtype, block_len, repeat, stride1, stride2, stride_unit)
    # offset means offset away from tensor head address, it's 16 for tensor[16]
    # entend_offset means valid data offset
    if isinstance(ori_offset, int):
        need_offset = Expr(extend_offset + offset + ori_offset).eval_value()
    else:
        ori_offset_value = Expr(ori_offset).eval_value()
        if ori_offset_value is not None:
            need_offset = Expr(extend_offset + offset + ori_offset).eval_value()
        else:
            need_offset = None

    if need_offset is not None:
        TikCheckUtil.check_le(
            need_offset, total_size, "{}, instruction need {} but only "
                                     "{}".format(msg, need_offset, total_size))


def get_blk_valid_list(mask, dtype, block_len):
    """get block id list"""
    bit_len = get_bit_len(dtype)
    blk_valid_list = []
    if bit_len == BIT_LEN_32:
        if len(mask) == MASK_LEN_CONTINOUS_MODE:
            mask_len = mask[MASK_HIGH_IDX]
            blk_num = ceil_div(mask_len, block_len)
            blk_valid_list = range(blk_num)
        else:
            for blk_id in range(BLK_NUM_PER_REP):
                if not mask[MASK_LOW_IDX] & BLK_32_LIST[blk_id] == 0:
                    blk_valid_list.append(blk_id)
    else:
        if len(mask) == MASK_LEN_CONTINOUS_MODE:
            mask_len = mask[MASK_HIGH_IDX]
            blk_num = ceil_div(mask_len, block_len)
            blk_valid_list = range(blk_num)
        else:
            for blk_id in range(BLK_NUM_PER_REP):
                if blk_id < 4:
                    if not mask[MASK_LOW_IDX] & BLK_16_LIST[blk_id] == 0:
                        blk_valid_list.append(blk_id)
                else:
                    if not mask[MASK_HIGH_IDX] & BLK_16_LIST[blk_id - 4] == 0:
                        blk_valid_list.append(blk_id)
    return blk_valid_list


def get_extend_interval(blk_valid_list, block_len,  # pylint: disable=R0913
                        time, blk_stride, rep_stride, stride_unit,
                        offset, dtype):
    """get tensor each blk interval"""
    extend_interval = []
    for blk_id in blk_valid_list:
        # blk_stride, rep_stride, unit: 32B
        if stride_unit == 0:
            begin = blk_id*blk_stride*block_len + \
                    time*rep_stride*block_len + offset
        # blk_stride, rep_gap, unit: 32B
        elif stride_unit == 1:
            begin = blk_id*blk_stride*block_len + \
                    time*(((BLK_NUM_PER_REP - 1)*blk_stride + 1)*block_len +
                          rep_stride*block_len) + offset
        # stride1: blk_gap, stride2: rep_stride, unit: elements
        elif stride_unit == 2:
            begin = blk_id*(blk_stride + block_len) + time*rep_stride + offset
        # blk_gap, rep_gap, unit: elements
        else:
            begin = blk_id*(block_len + blk_stride) + \
                    time*(((BLK_NUM_PER_REP - 1) * blk_stride + 1)*block_len +
                          rep_stride * block_len) + offset

        end = begin + block_len
        extend_interval.append(
            [begin*DTYPE_SIZE[dtype], end*DTYPE_SIZE[dtype]])
    return extend_interval


def _get_src_dst_blk_list(mask, dst, src,  # pylint: disable=R0913
                          dst_block_len, src_block_len, src_mask):
    """get blk valid list for check overlap"""
    if not isinstance(mask, (list, tuple)):
        mask = [mask]
    dst_blk_valid_list = get_blk_valid_list(mask, dst.dtype, dst_block_len)
    if src_mask is not None:
        if not isinstance(src_mask, (list, tuple)):
            mask = [src_mask]
        else:
            mask = src_mask
    src_blk_valid_list = get_blk_valid_list(mask, src.dtype, src_block_len)
    return dst_blk_valid_list, src_blk_valid_list


def check_overlap_single(  # pylint: disable=R0913, R0914
        name, msg, dst, src, mask, dst_block_len,
        src_block_len, repeat_times, dst_blk_stride, src_blk_stride,
        dst_rep_stride, src_rep_stride, dst_offset,
        src_offset, stride_unit, src_mask):
    """check overlap for single repeat"""
    dst_blk_valid_list, src_blk_valid_list = \
        _get_src_dst_blk_list(mask, dst, src,
                              dst_block_len, src_block_len, src_mask)
    single_dst_extend_interval = get_extend_interval(
        dst_blk_valid_list, dst_block_len, repeat_times-1,
        dst_blk_stride, dst_rep_stride, stride_unit, dst_offset, dst.dtype)

    single_src_extend_interval = get_extend_interval(
        src_blk_valid_list, src_block_len, repeat_times-1,
        src_blk_stride, src_rep_stride, stride_unit, src_offset, src.dtype)

    # check for single
    for interval_dst in single_dst_extend_interval:
        for interval_src in single_src_extend_interval:
            if max(interval_src[0], interval_dst[0]) < \
                    min(interval_src[1], interval_dst[1]):
                TikCheckUtil.raise_error(
                    "{} address overlapping error. "
                    "when repeat_times=1, only support {} 100% same,"
                    " and stride must be same too.".format(name, msg))


def check_overlap_many(  # pylint: disable=R0913, R0914
        name, msg, dst, src, mask, dst_block_len,
        src_block_len, repeat, dst_blk_stride, src_blk_stride,
        dst_rep_stride, src_rep_stride, dst_offset,
        src_offset, stride_unit=0, src_mask=None):
    """
    check overlap for not scatter instr
    """
    dst_blk_valid_list, src_blk_valid_list = \
        _get_src_dst_blk_list(mask, dst, src,
                              dst_block_len, src_block_len, src_mask)

    for time in range(repeat - 1):
        dst_extend_interval = get_extend_interval(
            dst_blk_valid_list, dst_block_len, time,
            dst_blk_stride, dst_rep_stride, stride_unit, dst_offset, dst.dtype)

        src_extend_interval = get_extend_interval(
            src_blk_valid_list, src_block_len, time + 1,
            src_blk_stride, src_rep_stride, stride_unit, src_offset, src.dtype)

        # check
        for interval_dst in dst_extend_interval:
            for interval_src in src_extend_interval:
                if max(interval_src[0], interval_dst[0]) < \
                        min(interval_src[1], interval_dst[1]):
                    TikCheckUtil.raise_error(
                        "When repeat_times>1, {} {} address "
                        "overlapping error. It is not support "
                        "iteration N's destination is the source"
                        " of next iteration".format(name, msg))


def check_address_overlapping(  # pylint: disable=R0913, R0914
        name, mask, dst, src, nblock, block_len, dst_block_len,
        src_block_len, repeat_times, dst_blk_stride, src_blk_stride,
        dst_rep_stride, src_rep_stride, dst_offset, src_offset,
        stride_unit=0, mask_mode="normal", msg="dst and src", src_mask=None):
    """check address overlap"""
    if not _check_overlap_param(mask, repeat_times, dst_offset, src_offset):
        return
    if mask_mode == "counter":
        repeat_times = ceil_div(mask, nblock*block_len)
        mask = mask % (nblock*block_len)
        if mask == MASK_VALUE_ZERO:
            mask = nblock*block_len
    # check address overlap
    if repeat_times == 1:
        if src_offset == dst_offset and \
                src_blk_stride == dst_blk_stride:
            # 100 same, support
            pass
        else:
            check_overlap_single(
                name, msg, dst, src, mask, dst_block_len,
                src_block_len, repeat_times, dst_blk_stride, src_blk_stride,
                dst_rep_stride, src_rep_stride, dst_offset,
                src_offset, stride_unit, src_mask)
    else:
        check_overlap_many(
            name, msg, dst, src, mask, dst_block_len, src_block_len,
            repeat_times, dst_blk_stride, src_blk_stride,
            dst_rep_stride, src_rep_stride, dst_offset,
            src_offset, stride_unit=stride_unit, src_mask=src_mask)


def vector_max_offset_cal(mask, dtype, block_len, repeat, stride1,
                          stride2, stride_unit=0):
    """
    get max offset of calculate vector
    block_len: elements in one repeat\
    stride1: blk_stride
    stride2: rep_stride
    """
    # pylint: disable=R0913
    if not isinstance(mask, (list, tuple)):
        mask = [mask]
    # mask_len, the last effective element
    mask_len = 0
    bit_len = get_bit_len(dtype)
    if bit_len == BIT_LEN_32:
        mask_len = _get_32bit_dtype_mask_len(mask)
    elif bit_len == BIT_LEN_8 or BIT_LEN_16:
        mask_len = _get_8or16bit_dtype_mask_len(mask)
    if mask_len % block_len == 0:
        # blk_stride, rep_stride, unit: 32B
        if stride_unit == 0:
            max_offset = (repeat - 1)*stride2*block_len + \
                         (mask_len // block_len - 1)*stride1*block_len + \
                         block_len
        # blk_stride, rep_gap, unit: 32B
        elif stride_unit == 1:
            max_offset = \
                (repeat - 1)*(((BLK_NUM_PER_REP - 1)*stride1 + 1)*block_len +
                              stride2*block_len) + (mask_len // block_len - 1)*\
                stride1*block_len + block_len
        # stride1: blk_gap, stride2: rep_stride, unit: elements
        elif stride_unit == 2:
            max_offset = (repeat - 1)*stride2 + \
                         (mask_len // block_len - 1)*(block_len + stride1) + \
                         block_len
        # blk_gap, rep_gap, unit: elements
        else:
            max_offset = \
                (repeat - 1)*((BLK_NUM_PER_REP - 1)*(block_len + stride1) +
                              block_len + stride2) + \
                (mask_len // block_len - 1)*(block_len + stride1) + block_len
    else:
        # blk_stride, rep_stride, unit: 32B
        if stride_unit == 0:
            max_offset = (repeat - 1)*stride2*block_len + \
                         (mask_len // block_len)*stride1*block_len + \
                         mask_len % block_len
        # blk_stride, rep_gap, unit: 32B
        elif stride_unit == 1:
            max_offset = \
                (repeat - 1)*(((BLK_NUM_PER_REP - 1)*stride1 + 1)*block_len +
                              stride2*block_len) + (mask_len // block_len)*\
                stride1*block_len + mask_len % block_len
        # stride1: blk_gap, stride2: rep_stride, unit: elements
        elif stride_unit == 2:
            max_offset = (repeat - 1)*stride2 + \
                         (mask_len // block_len)*(block_len + stride1) + \
                         mask_len % block_len
        # blk_gap, rep_gap, unit: elements
        else:
            max_offset = \
                (repeat - 1)*((BLK_NUM_PER_REP - 1)*(block_len + stride1) +
                              block_len + stride2) + (mask_len // block_len)*\
                (block_len + stride1) + mask_len % block_len

    return max_offset


def _get_32bit_dtype_mask_len(mask):
    """
    get mask len when dtype bit length is 32
    :param mask: mask
    :return: mask length
    """
    mask_len = 0
    if is_basic_expr(mask):
        mask_len = MASK_LEN_64
    elif len(mask) == MASK_LEN_CONTINOUS_MODE:
        mask_len = mask[MASK_HIGH_IDX]
        TikCheckUtil.check_le(
            mask_len, MASK_LEN_64,
            "mask should not be more than 64 when dtype bit length is 32")
    else:
        TikCheckUtil.check_equality(
            mask[MASK_HIGH_IDX], 0,
            "mask list should be [0, xxx] when dtype bit length is 32")
        for index in range(MIN_INDEX, MAX_INDEX):
            if not mask[MASK_LOW_IDX] & (CONST_MASK_VALUE >> index) == 0:
                mask_len = MAX_INDEX - index
                break
    return mask_len


def _get_8or16bit_dtype_mask_len(mask):
    """
    get mask len when dtype bit length is 8 or 16
    :param mask: mask
    :return: mask length
    """
    if is_basic_expr(mask):
        mask_len = MASK_LEN_128
    elif len(mask) == MASK_LEN_CONTINOUS_MODE:
        mask_len = mask[MASK_HIGH_IDX]
    else:
        mask_len, _ = get_mask_len(mask)
    return mask_len


def check_depthwise_conv_params(src_fm, pad_mode, l1_h, l1_w, feature_offset):
    """ check depthwise_conv params range

    Parameters
    ----------
    src_fm: source tensor_left
    pad_mode:   0 - no padding
                1 - two colume on right side
                2 - two colume on left side
                3 - one colume on right&left side
    l1_h: height of src_fm
    l1_w: width of src_fm
    feature_offset: the feature map matrix offset, dtype is same as src_fm.
                    If no offset is needed, set to 8'b0. Only works for
                    src_fm dtype is b8.

    Returns
    -------
    None
    """
    check_integer_in_range(
        pad_mode, range(VALUE_4),
        "pad_mode should be in the range of [0, 3], input pad_mode: {}"
        .format(pad_mode))
    check_integer_in_range(l1_h, range(VALUE_3, VALUE_4096),
                           "l1_h should be in the range of [3, 4095], input"
                           " l1_h : {}".format(l1_h))
    check_integer_in_range(l1_w, range(VALUE_1, VALUE_4096),
                           "l1_w should be in the range of [1, 4095], input"
                           " l1_w : {}".format(l1_w))
    if src_fm.dtype == "uint8":
        check_integer_in_range(
            feature_offset, range(VALUE_256),
            "when src_fm dtype is uint8, feature_offset should be in the range "
            "of [0, 255], input feature_offset : {}".format(feature_offset))
    elif src_fm.dtype == "int8":
        check_integer_in_range(
            feature_offset, range(-VALUE_128, VALUE_128),
            "when src_fm dtype is int8, feature_offset should be in the range "
            "of [-128, 127], input feature_offset : {}".format(feature_offset))
    else: # float16
        check_integer_in_range(
            feature_offset, range(VALUE_1),
            "when src_fm dtype is float16, feature_offset is invalid so it "
            "should be 0, input feature_offset : {}".format(feature_offset))


def check_depthwise_conv_l1_w(pad_mode, l1_w):
    """ check depthwise_conv l1_w

    Parameters
    ----------
    pad_mode:   0 - no padding
                1 - two colume on right side
                2 - two colume on left side
                3 - one colume on right&left side
    l1_h: height of src_fm

    Returns
    -------
    None
    """
    pad_mode = Expr(pad_mode).eval_value()
    l1_w = Expr(l1_w).eval_value()
    if pad_mode is None or l1_w is None:
        return
    if pad_mode == 0:
        if l1_w // 16 == 0 or l1_w % 16 != 2:
            TikCheckUtil.raise_error(
                "In depthwise_conv, when pad_mode is 0, l1_w should be "
                "16*i + 2(i is a positive integer), input l1_w: {}"
                .format(l1_w))


def float16format2uint16(numbers):
    """
    transeform data from float16 2 binary uint16
    :param numbers: input
    :return: uint16 result
    """
    if isinstance(numbers, list):
        result = []
        for one in numbers:
            one_np = np.float16(one)
            one_str = one_np.view(np.uint16)  # pylint: disable=E1121
            result.append(np.uint16(one_str))

        return result
    one_np = np.float16(numbers)
    one_str = one_np.view(np.uint16)  # pylint: disable=E1121

    return np.uint16(one_str)


def check_dict_and_not_none(input_dict, name):
    """check input dict"""
    if input_dict is None:
        TikCheckUtil.raise_error(name + " not support None")
    TikCheckUtil.check_type_match(input_dict, dict, name + " must be dict")


def check_not_none(input_n, name):
    """check input name"""
    if input_n is None:
        TikCheckUtil.raise_error(name + " not support None")


def check_list_type_and_range(
        input_list, length, list_type, input_range=None, name=''):
    """check input list type and range"""
    if input_list is None or length is None or list_type is None:
        TikCheckUtil.raise_error(name + "list input error")

    if len(input_list) != length:
        TikCheckUtil.raise_error(name + "input_list length error")

    for i in range(length):
        TikCheckUtil.check_type_match(
            input_list[i], list_type, name + "list[i] type error")
        if input_range:
            if Expr(input_list[i]).eval_value() is not None:
                TikCheckUtil.check_in_range(
                    input_list[i], input_range, name + "input out of range")


def check_aipp_one_src_overflow(src0, input_format,
                                src_horizontal_size, src_vertical_size):
    """check aipp one src tensor overflow"""
    src_size = src_vertical_size*src_horizontal_size*\
               AIPP_INPUT_TYPE_SWAP_ALIGN.get(input_format).get("size_channel")
    TikCheckUtil.check_ge(src0.buffer_size, src_size,
                          "src0 buffer_size less than "
                          "picture size, src0 overflow, "
                          "src_size: {}".format(src_size))


def check_aipp_two_src_overflow(src0, src1, input_format,
                                src_horizontal_size, src_vertical_size):
    """check aipp two src tensor overflow"""
    src0_size = src_horizontal_size*src_vertical_size*\
                AIPP_INPUT_TYPE_SWAP_ALIGN.get(
                    input_format).get("src0_size_channel")
    TikCheckUtil.check_ge(src0.buffer_size, src0_size,
                          "src0 buffer_size less than "
                          "picture size, src0 overflow. "
                          "src0_size: {}".format(src0_size))
    src1_size = src_horizontal_size*src_vertical_size*\
                AIPP_INPUT_TYPE_SWAP_ALIGN.get(
                    input_format).get("src1_size_channel")
    TikCheckUtil.check_ge(src1.buffer_size, src1_size,
                          "src1 buffer_size less than "
                          "picture size, src1 overflow, "
                          "input: {}".format(src1_size))


def aipp_get_enable_bit(arch_version, function_switch):
    """get enable bit"""
    if function_switch is None:
        TikCheckUtil.raise_error('function_switch not support None')

    TikCheckUtil.check_type_match(function_switch, (int),
                                  "function_switch should be int")

    crop_enbale = function_switch & CROP_BIT
    swap_enable = (function_switch // SWAP_BIT) & 1
    csc_enable = (function_switch // CSC_BIT) & 1
    dtc_enable = (function_switch // DTC_BIT) & 1
    area_pad_enable = (function_switch // AREA_PAD_BIT) & 1
    channel_pad_enable = (function_switch // CPAD_BIT) & 1

    if arch_version == HI3796CV300ESAIC:
        pre_clip_enable = (function_switch // PRE_CLIP_BIT) & 1
        scf_enable = (function_switch // SCF_BIT) & 1
        post_clip_enable = (function_switch // POST_CLIP_BIT) & 1
        flip_enable = (function_switch // FLIP_BIT) & 1
        stretch_enable = (function_switch // STRETCH) & 1
    else:
        pre_clip_enable = 0
        scf_enable = 0
        post_clip_enable = 0
        flip_enable = 0
        stretch_enable = 0

    if arch_version == AIC:
        raw_enable = (function_switch // RAW_BIT) & 1
    else:
        raw_enable = 0

    return crop_enbale, swap_enable, csc_enable, \
           dtc_enable, area_pad_enable, channel_pad_enable, \
           pre_clip_enable, scf_enable, post_clip_enable, \
           flip_enable, stretch_enable, raw_enable


def check_scale_range(real_hori_scaling, real_vert_scaling):
    """check scale range"""
    if REAL_SCALE_MIN >= real_hori_scaling >= REAL_SCALE_MAX:
        TikCheckUtil.raise_error("hori_scale out of range,"
                                 " hori_scale = {}".format(real_hori_scaling/4))

    if REAL_SCALE_MIN >= real_vert_scaling >= REAL_SCALE_MAX:
        TikCheckUtil.raise_error("vert_scale out of range, "
                                 "vert_scale = {}".format(real_vert_scaling/4))


def check_vec_trans_params_range(repeat_times, dst_rep_stride, src_rep_stride):
    """used to check the range of params of vec_trans

    Parameters
    ----------
    repeat_times : Repeated iterations times
    dst_rep_stride : offset of dst operator between adjacent iterations,
                     offset unit is 512B
    src_rep_stride : offset of src operator between adjacent iterations,
                     offset unit is 512B

    Returns
    -------
    Nones
    """
    check_integer_in_range(
        repeat_times, range(MIN_REPEAT_TIMES, MAX_VNCHWTRANS_REPEAT_TIMES),
        "repeat_times should be in the range of [{}, {}], "
        "input repeat_times: {}".format(MIN_REPEAT_TIMES,
                                        MAX_VNCHWTRANS_REPEAT_TIMES - 1,
                                        repeat_times))
    check_integer_in_range(
        dst_rep_stride, range(MIN_VNCHWTRANS_STRIDE, MAX_VNCHWTRANS_STRIDE),
        "dst_rep_stride should be in the range of [{}, {}], "
        "input dst_rep_stride: {}".format(MIN_VNCHWTRANS_STRIDE,
                                          MAX_VNCHWTRANS_STRIDE - 1,
                                          dst_rep_stride))
    check_integer_in_range(
        src_rep_stride, range(MIN_VNCHWTRANS_STRIDE, MAX_VNCHWTRANS_STRIDE),
        "src_rep_stride should be in the range of [{}, {}], "
        "input src_rep_stride: {}".format(MIN_VNCHWTRANS_STRIDE,
                                          MAX_VNCHWTRANS_STRIDE - 1,
                                          src_rep_stride))


def check_vec_trans_overflow(dst_shape, src_shape,  # pylint: disable=R0913
                             dst_offset, src_offset,
                             repeat_times, dst_rep_stride, src_rep_stride):
    """used to check the tensor size of params of vec_trans

    Parameters
    ----------
    dst_shape: dst tensor shape
    src_shape: src tensor shape
    dst_offset: dst tensor offset
    src_offset: src tensor offset
    repeat_times : Repeated iterations times
    dst_rep_stride : offset of dst operator between adjacent iterations,
                     offset unit is 512B
    src_rep_stride : offset of src operator between adjacent iterations,
                     offset unit is 512B

    Returns
    -------
    Nones
    """
    if dst_offset is not None:
        dst_elt_count = reduce_mul(dst_shape)
        required_dst_elt_count = \
            PER_TRANSPOSE_DATA_SIZE + dst_rep_stride * \
            PER_TRANSPOSE_DATA_SIZE * (repeat_times - 1) + dst_offset
        TikCheckUtil.check_ge(dst_elt_count, required_dst_elt_count,
                              "elements of dst should be more than %d" %
                              required_dst_elt_count)
    if src_offset is not None:
        src_elt_count = reduce_mul(src_shape)
        required_src_elt_count = \
            PER_TRANSPOSE_DATA_SIZE + src_rep_stride * \
            PER_TRANSPOSE_DATA_SIZE * (repeat_times - 1) + src_offset
        TikCheckUtil.check_ge(src_elt_count, required_src_elt_count,
                              "elements of src should be more then %d" %
                              required_src_elt_count)


def align_start_pos(start_pos, dtype_size):
    """
    Align the src address in the per iteration of vreduce

    Parameters
    ----------
    start_pos: int/Scalar, the start position in work_tensor
    dtype_size: int, the size of element in src. unit is Byte
    align_offset_map: dict, record the position

    Returns
    -------
    align start position
    """
    return ceil_div(start_pos*dtype_size, 32)*32//dtype_size


def check_space_overflow(mask, dst, work_tensor,  # pylint: disable=R0913, R0914
                         src, dst_offset,
                         work_tensor_offset, src_offset, repeat_times,
                         src_rep_stride, cal_index=False, rep_is_scalar=False):
    """
    check work_tensor's elements if is enough

    Parameters
    ----------
    src: Tensor, the source tensor
    work_tensor: Tensor, temporary memory, for internal calculation
    repeat_times: int/Scalar, the times of instruction run

    Returns
    -------
    None
    """
    dtype_size = DTYPE_SIZE[src.dtype]

    # check dst tensor size
    if dst_offset is not None:
        # if need save index, need 2 element, else only need 1 elememt
        if not cal_index:
            required_dst_size = 1
        else:
            required_dst_size = 2

        dst_size = reduce_mul(dst.indice.origin_shape) - dst_offset
        TikCheckUtil.check_ge(dst_size, required_dst_size,
                              "dst's element should be more than {}"
                              " but input {}".format(required_dst_size,
                                                     dst_size))

    # check src tensor size
    if src_offset is not None:
        # check tensor overflow
        # cause tensor has at least one element, dst does not need to check.
        src_blk_stride = 1
        check_tensor_overflow((src,), mask, repeat_times, (src_blk_stride,),
                              (src_rep_stride,), ("src",))

    per_rep_output = 2
    if work_tensor_offset is not None:
        work_tensor_elements = reduce_mul(work_tensor.indice.origin_shape) - \
                               work_tensor_offset
        it1_output_count = per_rep_output*repeat_times

        if not cal_index:
            TikCheckUtil.check_ge(work_tensor_elements, it1_output_count,
                                  "work_tensor's element should be "
                                  "more than {} but get {}".
                                  format(it1_output_count,
                                         work_tensor_elements))
            return

        def _vreduce_body_cal(pre_data_count, cur_start_pos):
            body_rep_times = pre_data_count // element_num_per_rep
            body_output_count = per_rep_output * body_rep_times
            has_tail = (pre_data_count % element_num_per_rep) != 0
            tail_output_count = per_rep_output if has_tail else 0
            output_count = body_output_count + tail_output_count
            next_start_pos = align_start_pos(cur_start_pos + output_count,
                                             dtype_size)
            return output_count, next_start_pos
        it2_align_start = align_start_pos(it1_output_count, dtype_size)
        element_num_per_rep = ONE_REP_BYTE_SIZE // DTYPE_SIZE[src.dtype]

        it2_output_count, it3_start_pos = _vreduce_body_cal(
            it1_output_count, it2_align_start)
        if it2_output_count == per_rep_output and not rep_is_scalar:
            it3_start_pos = it2_align_start
            res_index = it3_start_pos + 1
        else:
            res_index = it3_start_pos + 1
            if it2_output_count > element_num_per_rep or rep_is_scalar:
                _, it4_start_pos = _vreduce_body_cal(it2_output_count,
                                                     it3_start_pos)
                res_index = it4_start_pos + 1

        need_size = res_index + 1

        TikCheckUtil.check_ge(work_tensor_elements, need_size,
                              "work_tensor's element should not be less than {}"
                              " but get {}".format(need_size,
                                                   work_tensor_elements))


def vreduce_create_mask(data_len):
    """
    get mask in the "0101010101" format

    Parameters
    ----------
    data_len: int, src_element // 2, src_element is the element count of
                vreduce instruction will cover

    Returns
    -------
    the new mask
    """
    # mask_len record the valid data num of low mask
    # B16: max data num of low mask is 64
    # B32: max data num of low mask is 64, and all data can select by low mask
    # out data saved as: Data1Index1Data2Index2....DatanIndexn
    # so valid data is half of max data num
    mask_len = 32
    high_mask_bit = data_len - mask_len
    high_mask = 0
    low_mask = 0
    for _ in range(0, high_mask_bit):
        high_mask = high_mask << 2
        high_mask = high_mask | 1

    if data_len >= mask_len:
        low_mask_bit = mask_len
    else:
        low_mask_bit = data_len
    for _ in range(0, low_mask_bit):
        low_mask = low_mask << 2
        low_mask = low_mask | 1
    return [high_mask, low_mask]


def check_vreduce_repeat_times(repeat_times, cal_index, dtype):
    """
    check repeat_times of vec_reduce_max/vec_reduce_min instruction

    Parameters
    ----------
    repeat_times: int/Scalar, the times of instrction run
    cal_index: bool, if calculate index
    dtype: s16/f16/f32, tensor's dtype

    Returns
    -------
    None
    """
    if cal_index:
        if dtype == "int16":
            check_integer_in_range(repeat_times, range(1, MAX_REPEAT_TIMES),
                                   "int16 data type repeat_times should be in"
                                   " [1, 255] but get {}".format(repeat_times))
        elif dtype == "float16":
            max_repeat_times = 512
            check_integer_in_range(repeat_times, range(1, max_repeat_times),
                                   "float16 data type repeat_times should be in"
                                   " [1, 511] but get {}".format(repeat_times))
        else:
            check_integer_in_range(repeat_times,
                                   range(1, MAX_VREDUCE_REPEAT_TIMES),
                                   "float32 data type repeat_times should be in"
                                   " [1, 4096] but get {}".format(repeat_times))
    else:
        check_integer_in_range(repeat_times,
                               range(1, MAX_VREDUCE_REPEAT_TIMES),
                               "repeat_times should be in [1, 4096] but get"
                               " {}".format(repeat_times))


def cal_extent_stride_unit_mask(mask, repeat_times, tensor, stride_unit,
                                blk_stride, rep_stride, mask_mode):
    """calculate extent, based on mask, stride_unit

    Parameters
    ----------
    mask: Effective operation on element
    repeat_times: Repeated iterations times
    tensor: dst/src tensor
    stride_unit: address and offset unit both affect it. default = 0
    blk_stride: offset of dst/src operator between different block in one
                iteration
    rep_stride: offset of dst/src operator in the same block between adjacent
                iterations
    mask_mode: "normal" - mask normal mode
               "counter" - mask counter mode

    Returns
    -------
    extent
    """
    # pylint: disable=R0913
    # function's input params is too much, so disable them
    if mask_mode == 'counter':
        repeat = ceil_div(mask, ONE_REP_BYTE_SIZE // DTYPE_SIZE[tensor.dtype])
    else:  # normal
        repeat = repeat_times
    # blk_stride: stride, rep_stride: stride, unit: 32B
    if stride_unit == 0:
        extent = ((repeat - 1)*rep_stride + (BLK_NUM_PER_REP - 1)*blk_stride +
                  1)*ONE_BLK_SIZE
    # blk_stride: stride, rep_stride: gap, unit: 32B
    elif stride_unit == 1:
        extent = ((repeat - 1)*rep_stride +
                  repeat*((BLK_NUM_PER_REP - 1)*blk_stride + 1))*ONE_BLK_SIZE
    # blk_stride: gap, rep_stride: stride, unit: elements
    elif stride_unit == 2:
        extent = ((repeat - 1)*rep_stride + (BLK_NUM_PER_REP - 1)*blk_stride +
                  ONE_REP_BYTE_SIZE // DTYPE_SIZE[tensor.dtype])*\
                 DTYPE_SIZE[tensor.dtype]
    # blk_stride: gap, rep_stride: gap, unit: elements
    else:
        extent = ((repeat - 1)*rep_stride + repeat*(
            ONE_REP_BYTE_SIZE // DTYPE_SIZE[tensor.dtype] +
            (BLK_NUM_PER_REP - 1)*blk_stride))*DTYPE_SIZE[tensor.dtype]
    return Expr(extent).get()


def check_dtype_overflow(mask, repeat_times, src, rep_stride):
    """check overflow"""
    element_byte = cal_extent_stride_unit_mask(mask, repeat_times, src, 0, 1,
                                               rep_stride, "normal")
    element_byte = Expr(element_byte).eval_value()
    if element_byte is not None:
        bit_len = get_bit_len(src.dtype)
        byte_len = bit_len // 8
        element = element_byte // byte_len
        if src.dtype == "int16":
            limit = 2**15 - 1
        elif src.dtype == "float16":
            limit = 65504
        else:
            limit = 2**32 - 1

        TikCheckUtil.check_le(element, limit,
                              "Elements number should be less than the limit of"
                              " unsigned int type corresponding to {} which is "
                              "{} but get {}".format(src.dtype,
                                                     limit + 1, element))


def check_vshl_vshr_scalar(src_dtype, scalar):
    """check vshl/vshr input scalar value

    Parameters
    ----------
    name: instr name
    src: src tensor
    scalar: scalar_value

    Returns
    -------
    None
    """
    _max_value = get_bit_len(src_dtype)
    check_integer_in_range(
        scalar, range(_max_value + 1),
        "src_scalar should be in the range of [0, %d], input value: %s"
        % (_max_value, str(scalar)))


def check_wino_ft_params(l1_h, l1_w, l1_c, # pylint: disable=R0913, R0914
                         dst_stride, pad_left, pad_right,
                         pad_top, pad_bottom, m_extension, m_start_pt,
                         k_extension, k_start_pt, column_indicator, src):
    """check winograd_weight_transform params

    Parameters
    ----------
    mask: Effective operation on element
    repeat_times: Repeated iterations times
    tensor: dst/src tensor
    stride_unit: address and offset unit both affect it. default = 0
    blk_stride: offset of dst/src operator between different block in one
                iteration
    rep_stride: offset of dst/src operator in the same block between adjacent
                iterations
    mask_mode: "normal" - mask normal mode
               "counter" - mask counter mode

    Returns
    -------
    None
    """
    # check feature map size
    check_integer_in_range(l1_h, range(MIN_L1_H_W_C, MAX_L1_H_W),
                           "l1_h should be in the range of [1, 65535], "
                           "input l1_h: {}".format(l1_h))
    check_integer_in_range(l1_w, range(MIN_L1_H_W_C, MAX_L1_H_W),
                           "l1_w should be in the range of [1, 65535], "
                           "input l1_w: {}".format(l1_w))
    check_integer_in_range(l1_c, range(MIN_L1_H_W_C, MAX_L1_C),
                           "l1_c should be in the range of [1, 4095], "
                           "input l1_c: {}".format(l1_c))
    # check dst_stride
    check_integer_in_range(
        dst_stride, range(MAX_DST_GAP_WINO),
        "dst_stride should be in the range of [0, 63], input dst_stride: {}"
        .format(dst_stride))
    # check padding
    pad_left = Expr(pad_left).eval_value()
    pad_right = Expr(pad_right).eval_value()
    if pad_left is not None and pad_right is not None:
        TikCheckUtil.check_in_range(
            (pad_left, pad_right), WINO_PAD_MAP.keys(),
            "(pad_left, pad_right) should be (0, 0), (0, 1), (0, 2), "
            "(1, 0), (1, 1) or (1,2), input pad_left: {}, pad_right: {}"
            .format(pad_left, pad_right))
    pad_top = Expr(pad_top).eval_value()
    pad_bottom = Expr(pad_bottom).eval_value()
    if pad_top is not None and pad_bottom is not None:
        TikCheckUtil.check_in_range(
            (pad_top, pad_bottom), WINO_PAD_MAP.keys(),
            "(pad_top, pad_bottom) should be (0, 0), (0, 1), (0, 2), "
            "(1, 0), (1, 1) or (1,2), input pad_top: {}, pad_bottom: {}"
            .format(pad_top, pad_bottom))
    check_integer_in_range(
        m_extension, range(MAX_EXTENSION),
        "m_extension should be in the range of [0, 65535], "
        "input m_extension: {}".format(m_extension))
    check_integer_in_range(
        m_start_pt, range(MAX_START_PT),
        "m_start_pt should be in the range of [0, 65535], "
        "input m_start_pt: {}".format(m_start_pt))
    m_start_pt = Expr(m_start_pt).eval_value()
    if m_start_pt is not None:
        m_start_pt_ele_align = 16
        if m_start_pt % m_start_pt_ele_align != 0:
            TikCheckUtil.raise_error(
                "m_start_ptshould be multiple of 16, input m_start_pt: {}"
                .format(m_start_pt))
    check_integer_in_range(
        k_start_pt, range(MAX_K_WINO),
        "k_start_pt should be in the range of [0, 4096], "
        "input k_start_pt: {}".format(k_start_pt))
    k_start_pt = Expr(k_start_pt).eval_value()
    if k_start_pt is not None:
        k_start_pt_byte_align = 32
        if k_start_pt * DTYPE_SIZE[src.dtype] % k_start_pt_byte_align != 0:
            TikCheckUtil.raise_error(
                "k_start_pt in Byte should be multiple of 32B, input "
                "k_start_pt: {}, input src dtype: {}"
                .format(k_start_pt, src.dtype))
    check_integer_in_range(
        k_extension, range(MAX_K_WINO),
        "k_extension should be in the range of [0, 4096], "
        "input k_extension: {}".format(k_extension))
    k_extension = Expr(k_extension).eval_value()
    if k_extension is not None:
        k_extension_byte_align = 32
        if k_extension * DTYPE_SIZE[src.dtype] % k_extension_byte_align != 0:
            TikCheckUtil.raise_error(
                "k_extension in Byte should be multiple of 32B, input "
                "k_extension: {}, input src dtype: {}"
                .format(k_extension, src.dtype))
    # check column_indicator
    check_integer_in_range(
        column_indicator, range(MAX_COL_INDIC),
        "column_indicator should be in the range of [0, 3], input "
        "column_indicator: {}".format(column_indicator))


def is_scalar(obj):
    """check whether input object is scalar

    Parameters
    ----------
    obj: input object

    Returns
    -------
    bool: whether obj is scalar
    """
    if isinstance(obj, BasicData) and obj.is_scalar():
        return True
    return False

def is_tensor(obj):
    """check whether input object is tensor

    Parameters
    ----------
    obj: input object

    Returns
    -------
    bool: whether obj is tensor
    """
    if isinstance(obj, BasicData) and obj.is_tensor():
        return True
    return False


def check_vms4_repeat_times(repeat_times, element_count_list, valid_bit,
                            if_exhausted_suspension):
    """check vms repeat_times range, type

    Parameters
    ----------
    repeat_times: repeat_times: times of invoke this instrction
    element_count_list : length of the proposal list
    valid_bit : 0001 one lines are valid
                0011 two lines are valid
                0111 three lines are valid
                1111 four lines are valid
    if_exhausted_suspension : 0 not stop, 1 stop

    Returns
    -------
    None
    """
    length_list_same_flag = True
    if all(isinstance(value, int) for value in element_count_list):
        if len(set(element_count_list)) != 1:
            length_list_same_flag = False
    valid_bit_15_flag = True
    if Expr(valid_bit).eval_value() is not None:
        if Expr(valid_bit).eval_value() != 15:
            valid_bit_15_flag = False
    if if_exhausted_suspension or length_list_same_flag is False or \
            valid_bit_15_flag is False:
        if isinstance(repeat_times, int):
            TikCheckUtil.check_equality(
                repeat_times, MIN_REPEAT_TIMES,
                "When input params cannot meet repeat criterions as follows: 1."
                " if_exhuasted_suspension == False, 2. valid_bit in decimal is "
                "15, 3. elements of element_count_list are equal to each other,"
                " repeat_times should be 1, but input repeat_times: {}"
                .format(repeat_times))


def get_vbi_src_common_need_size(dtype, mask_len, total_repeat_times):
    """for vbi instruction, get src0_offset/src1 need size(could be an expr),
     unit is elements. when mask is scalar-list, mask is considered as 128

    Parameters
    ----------
    dtype: dst/src1/src0 dtype, float16
    mask_len: length between lowest digit and top effective digit of mask
    total_repeat_times: horizontal_repeat_times multiplied by
                        vertical_repeat_times
    Returns
    -------
    need_src_block: need elements
    """
    if mask_len is None:
        need_src_block = BLK_NUM_PER_REP*total_repeat_times
    else:
        valid_block_len = ceil_div(
            mask_len, ONE_BLK_SIZE // DTYPE_SIZE[dtype])
        need_src_block = (total_repeat_times - 1)*\
                          BLK_NUM_PER_REP + valid_block_len
    return need_src_block


def get_vbi_src0_offset_need_size(dtype, mask_len, total_repeat_times):
    """for vbi instruction, get src0_offset need size(could be an expr),
     unit is elements. when mask is scalar-list, mask is considered as 128

    Parameters
    ----------
    dtype: dst/src1/src0 dtype, float16
    mask_len: length between lowest digit and top effective digit of mask
    total_repeat_times: horizontal_repeat_times multiplied by
                        vertical_repeat_times
    Returns
    -------
    need_src0_block: src0_offset need elements
    """
    return get_vbi_src_common_need_size(dtype, mask_len, total_repeat_times)


def get_vbi_src1_tensor_need_size(dtype, mask_len, repeat_mode,
                                  total_repeat_times):
    """for vbi instruction, get src1 need size(could be an expr),
     unit is elements, when mask is scalar-list, mask is considered as 128

    Parameters
    ----------
    dtype: dst/src1/src0 dtype, float16
    mask_len: length between lowest digit and top effective digit of mask
    repeat_mode:indicate how many elements at src1 are consumed in one iteration
    total_repeat_times: horizontal_repeat_times multiplied by
                        vertical_repeat_times
    Returns
    -------
    src1 need size
    """
    if repeat_mode == 0:
        src1_num_per_repeat = 1
        need_size = src1_num_per_repeat*total_repeat_times
    else:
        need_size = get_vbi_src_common_need_size(dtype, mask_len,
                                                 total_repeat_times)
    return need_size


def check_vbi_src1_tensor_overflow(src1, repeat_mode, total_repeat_times,
                                   mask_len, src1_offset):
    """for vbi instruction, check whether src1 is overflow

    Parameters
    ----------
    src1: operation tensor
    repeat_mode:indicate how many elements at src1 are consumed in one iteration
    total_repeat_times: horizontal_repeat_times multiplied by
                        vertical_repeat_times
    mask_len: length between lowest digit and top effective digit of mask
    src1_offset: src1 tensor's offset,
                 for example: src1=tensor[16], then src1_offset=16
    Returns
    -------
    None
    """
    if mask_len is None:
        return
    # first check src1
    total_size = reduce_mul(src1.indice.origin_shape)
    need_size = get_vbi_src1_tensor_need_size(
        src1.dtype, mask_len, repeat_mode, total_repeat_times)
    need_size = Expr(need_size + src1_offset).eval_value()
    if need_size is not None:
        TikCheckUtil.check_le(need_size, total_size,
                              "vbi src1 tensor overflow, instruction need %s "
                              "but only %s" % (need_size, total_size))


def check_vbi_overlap(src0_offset, src1, repeat_mode,  # pylint: disable=R0913
                      total_repeat_times, mask_len,
                      src0_offset_offset, src1_offset):
    """for vbi instruction, check whether src0_offset and src1 is overlap

    Parameters
    ----------
    src0_offset: operator tensor
    src1: operator tensor
    repeat_mode:indicate how many elements at src1 are consumed in one iteration
    total_repeat_times: horizontal_repeat_times multiplied by
                        vertical_repeat_times
    mask_len: length between lowest digit and top effective digit of mask
    src0_offset_offset: src0_offset tensor's offset, for example:
                        src0_offset=tensor[16], then src0_offset_offset=16
    src1_offset: src1 tensor's offset,
                 for example: src1=tensor[16], then src1_offset=16
    Returns
    -------
    None
    """
    if mask_len is None or is_basic_expr(TikUtil.to_list(mask_len)):
        return
    if is_basic_expr([total_repeat_times]):
        return
    if src0_offset.buffer == src1.buffer:
        src0_offset_start = Expr(src0_offset_offset*\
                                 DTYPE_SIZE[src0_offset.dtype]).eval_value()
        src0_offset_stop = Expr(src0_offset_start +
                                get_vbi_src0_offset_need_size(
                                    src1.dtype, mask_len, total_repeat_times)*\
                                DTYPE_SIZE[src0_offset.dtype]
                                ).eval_value()
        src1_start = Expr(src1_offset*DTYPE_SIZE[src1.dtype]).eval_value()
        src1_stop = Expr(src1_start +
                         get_vbi_src1_tensor_need_size(src1.dtype, mask_len,
                                                       repeat_mode,
                                                       total_repeat_times)*
                         DTYPE_SIZE[src1.dtype]).eval_value()
        if all((value is not None) for value in
               (src0_offset_start, src0_offset_stop, src1_start, src1_stop)):
            if max(src0_offset_start, src1_start) < \
                    min(src0_offset_stop, src1_stop):
                TikCheckUtil.raise_error("vbi src0_offset and src1 "
                                         "address overlapping error.")


def check_mask_valid(mask, tensor_bit_len):
    """func of check mask"""
    mask_list = TikUtil.to_list(mask)
    if len(mask_list) == MASK_LEN_CONTINOUS_MODE and is_basic_expr(mask_list):
        for msk in mask_list:
            check_scalar_dtype(msk,
                               "scalar_mask should be a scalar's dtype"
                               " of int/uint, input dtype %s" % type(msk))
    elif len(mask_list) == MASK_LEN_CONTINOUS_MODE and \
            is_immediate_number(mask_list):
        for msk in mask_list:
            TikCheckUtil.check_type_match(
                msk, int, "mask should be int type, input type %s" % type(msk))
        # for immediate mask, value should  be in range of [1,128], b16
        if tensor_bit_len == BIT_LEN_16:
            TikCheckUtil.check_in_range(
                mask_list[0], range(MIN_MASK, MAX_MASK),
                "mask value should be in the range of [1, 128] for b16 "
                "tensor, input mask %s" % mask_list[0])
        # b32 case
        else:
            TikCheckUtil.check_in_range(
                mask_list[0], range(MIN_MASK, MAX_MASK_64),
                "mask value should be in the range of [1, 64] for b32 "
                "tensor, input mask %s" % mask_list[0])
    elif len(mask_list) == MASK_LEN_FULL_MODE and is_basic_expr(mask_list):
        for msk in mask_list:
            check_scalar_dtype(msk,
                               "scalar_mask should be a scalar's dtype"
                               " of int/uint, input dtype %s" % type(msk))
    elif len(mask_list) == MASK_LEN_FULL_MODE and \
            is_immediate_number(mask_list):
        check_imme_mask_full_mode(mask_list, tensor_bit_len)
    else:
        TikCheckUtil.raise_error("not support this type of mask now")


def check_work_tensor_overflow(mask, src,  # pylint: disable=R0913
                               work_tensor, repeat_times,
                               src_rep_stride, multi_factor):
    """check work tensor space"""
    if is_basic_expr(TikUtil.to_list(mask)):
        return
    src_bit_len = get_bit_len(src.dtype)
    src_extend = vector_max_offset_cal(
        mask, src.dtype, ONE_REP_BYTE_SIZE // src_bit_len,
        repeat_times, 1, src_rep_stride)
    src_extend = ceil_div(src_extend, ONE_REP_BYTE_SIZE // src_bit_len)*\
                 ONE_REP_BYTE_SIZE // src_bit_len
    work_need_ele = Expr(multi_factor*src_extend).eval_value()
    work_tensor_size = Expr(work_tensor.size).eval_value()

    # check work_tensor overflow
    if work_need_ele is not None and work_tensor_size is not None:
        # warning: origin_shape is Scalar
        TikCheckUtil.check_le(
            work_need_ele, work_tensor_size,
            "work_tensor overflow, needed %s but only %s." %
            (work_need_ele, work_tensor_size))


def check_high_preci_overlap(mask, src, dst,  # pylint: disable=R0913
                             work_tensor, dst_extend,
                             src_extend, dst_offset, src_offset,
                             work_tensor_offset, multi_factor, name):
    """check high preci instr overlap"""

    if is_basic_expr(TikUtil.to_list(mask)):
        return
    if any(offset is None for offset in
           (dst_offset, src_offset, work_tensor_offset)):
        return
    src_bit_len = get_bit_len(src.dtype)
    tensor_size = ceil_div(src_extend, ONE_REP_BYTE_SIZE // src_bit_len) * \
                  ONE_REP_BYTE_SIZE // src_bit_len
    if dst.buffer == src.buffer:
        src_need = Expr(src_extend + src_offset).eval_value()
        dst_need = Expr(dst_extend + dst_offset).eval_value()
        if src_offset <= src_need <= dst_offset or \
                dst_offset <= dst_need <= src_offset:
            pass
        else:
            TikCheckUtil.raise_error(
                "%s doesn't "
                "support dst and src address "
                "overlapping fully or partially." % name)
    if src.buffer == work_tensor.buffer:
        src_need = Expr(src_extend + src_offset).eval_value()
        work_need_ele = Expr(
            multi_factor*tensor_size + work_tensor_offset).eval_value()
        if src_offset <= src_need <= work_tensor_offset or \
                work_tensor_offset <= work_need_ele <= src_offset:
            pass
        else:
            TikCheckUtil.raise_error(
                "%s doesn't "
                "support src and work_tensor address "
                "overlapping fully or partially." % name)
    if dst.buffer == work_tensor.buffer:
        dst_need = Expr(dst_extend + dst_offset).eval_value()
        work_need_ele = Expr(
            multi_factor*tensor_size + work_tensor_offset).eval_value()
        if dst_offset <= dst_need <= work_tensor_offset or \
                work_tensor_offset <= work_need_ele <= dst_offset:
            pass
        else:
            TikCheckUtil.raise_error(
                "%s doesn't "
                "support dst and work_tensor address "
                "overlapping fully or partially." % name)


def check_over_high_preci(  # pylint: disable=R0913
        mask, dst, src, work_tensor,
        repeat_times, dst_rep_stride, src_rep_stride,
        dst_offset, src_offset, work_tensor_offset, multi_factor, name):
    """check overflow and overlap for high preci instr"""
    if all(isinstance(value, int) for \
           value in (repeat_times, dst_rep_stride, src_rep_stride)):
        # check tensor overflow
        check_tensor_overflow((dst, src), mask, repeat_times,
                              (1, 1), (dst_rep_stride, src_rep_stride),
                              ("dst", "src"))

        # check work_tensor overflow by !!!core version and dtype!!!
        check_work_tensor_overflow(mask, src,
                                   work_tensor, repeat_times,
                                   src_rep_stride, multi_factor)
        # check overlap
        src_bit_len = get_bit_len(src.dtype)
        dst_bit_len = get_bit_len(dst.dtype)
        dst_extend = vector_max_offset_cal(
            mask, dst.dtype, ONE_REP_BYTE_SIZE // dst_bit_len,
            repeat_times, 1, dst_rep_stride)
        src_extend = vector_max_offset_cal(
            mask, src.dtype, ONE_REP_BYTE_SIZE // src_bit_len,
            repeat_times, 1, src_rep_stride)
        check_high_preci_overlap(mask, src, dst, work_tensor,
                                 dst_extend, src_extend, dst_offset, src_offset,
                                 work_tensor_offset, multi_factor, name)
