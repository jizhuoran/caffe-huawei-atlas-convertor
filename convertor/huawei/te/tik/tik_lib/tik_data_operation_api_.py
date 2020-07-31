"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_data_operation_api_.py
DESC:     provide data padding, data conversion, data move related instructions
CREATED:  2019-08-12 18:53:42
MODIFIED: 2019-08-12 21:29:18
"""
# disabling:
# C0302: too-many-lines (this file is full of data operation instructions)
# R0913: too-many-arguments
# E1101: no-member
# R0914: too-many-locals
# R0904: too-many-public-methods
# W0613: unused-argument

from copy import deepcopy  # pylint: disable=C0302
import numpy as np

from te import tvm
from te.platform import cce_params
from te.platform.cce_conf import intrinsic_check_support
from te.platform.cce_conf import api_check_support
from te.platform.cce_params import scope_cc
from te.platform.cce_params import scope_cbuf
from te.platform.cce_params import scope_cb
from te.platform.cce_params import scope_smask
from te.platform.cce_params import scope_ca
from te.platform.cce_params import scope_ubuf
from te.platform.cce_params import scope_gm
from te.platform.cce_params import VEC
from te.platform.cce_params import HI3796CV300ES
from te.platform.cce_params import AIC
from te.platform.cce_params import ASCEND_910
from te.platform.cce_params import ASCEND_310AIC
from te.platform.cce_params import ASCEND_910AIC
from te.platform.cce_params import HI3796CV300ESAIC
from te.platform.cce_params import HI3796CV300CSAIC
from te.platform.cce_params import ASCEND_610
from te.platform.cce_params import ASCEND_620
from te.tik.common.util import TikUtil, DTYPE_SIZE, ceil_div, get_bit_len,\
    check_integer_in_range, reduce_mul, check_vnchwconv_overflow,\
    is_immediate_number, is_basic_expr, check_scalar_dtype, \
    check_scalar_dtype_float
from .tik_expr import Expr
from .. import debug
from ..api.tik_ir_builder import TikIRBuilder
from .tik_util import type_convert, concat_params, change_dtype_str, \
    emit_scatter_instr, set_tensor_addr_to_scalar
from ..api.tik_scalar import mask_concat, Scalar
from ..api.tik_tensor import get_addr_list, Tensor
from .tik_params import VA_REG, PAD_LENGTH,\
    MAX_PADDING, MAX_TENSOR_WIDTH, MAX_TENSOR_HEIGHT,\
    MIN_TENSOR_WIDTH, MIN_TENSOR_HEIGHT, MAX_FETCH_POS,\
    MAX_START_POINT, MIN_START_POINT, MAX_STRIDE, MIN_REPEAT_TIMES,\
    MIN_STRIDE, MAX_FILTER_WIDTH, MIN_FILTER_WIDTH,\
    MAX_FILTER_HEIGHT, MIN_FILTER_HEIGHT, MAX_DILATION,\
    MIN_DILATION, MAX_VA_ADDR_NUM, MAX_BLK_STRIDE_DOUBLE_BYTE,\
    PIPE_V, MAX_REP_STRIDE_DOUBLE_BYTE, MAX_REP_STRIDE_SINGLE_BYTE,\
    MAX_REPEAT_TIMES, MAX_SID, MAX_DST_GAP_DOUBLE_BYTE,\
    MAX_START_INDEX, PIPE_MTE1, MAX_REPEAT_MODE, VA0_INDEX, VA2_INDEX, PIPE_S,\
    DST_TYPE_LEN, MIN_NBURST, MAX_NBURST_SINGLE_BYTE, MAX_DST_GAP_SINGLE_BYTE,\
    MAX_SRC_GAP, MIN_BURST_REPEAT, MAX_BURST_REPEAT, ONE_BYTE_BIT_LEN,\
    ONE_REP_BYTE_SIZE, PADDING_LEFT_IDX, PADDING_RIGHT_IDX, PADDING_TOP_IDX,\
    PADDING_BOT_IDX, FMATRIX_OFFSET_LIST, FMATRIX_SEGMENT_LIST,\
    LOAD3DV1_REG_XM_OFFSET_LIST, LOAD3DV1_REG_XM_SEGMENT_LIST,\
    LOAD3DV1_REG_XT_OFFSET_LIST, LOAD3DV1_REG_XT_SEGMENT_LIST,\
    COL2IMG_REG_XT_OFFSET_LIST, COL2IMG_REG_XT_SEGMENT_LIST, BLK_NUM_PER_REP,\
    REG_FCOL2IMG_OFFSET_LIST, REG_FCOL2IMG_SEGMENT_LIST, MASK_VALUE_128,\
    COL2IMG_REG_XM_OFFSET_LIST, COL2IMG_REG_XM_SEGMENT_LIST,\
    CONV_RELU_VECTOR_QUANT, CONV_RELU_QUANT, BYTE_SIZE, SCALE_ADDR_BIT_POS,\
    ONE_BLK_SIZE, MIN_M_LEN, MAX_M_LEN, MIN_CHANNELS, MAX_CHANNELS,\
    V4DTRANS_OFFSET_LIST, V4DTRANS_SEGMENT_LIST, MAX_PAD_MODE,\
    MAX_STRIDE_UNIT, VPADDING_OFFSET_LIST, VPADDING_SEGMENT_LIST,\
    MIN_EXTENSION, MAX_EXTENSION, MAX_START_PT, LOAD3DV2_REG_XM_OFFSET_LIST,\
    LOAD3DV2_REG_XM_SEGMENT_LIST, LOAD3DV2_REG_XT_OFFSET_LIST,\
    LOAD3DV2_REG_XT_SEGMENT_LIST, ONE_IR, TWO_IR, MASK_VALUE_64,\
    VNCHWCONV_LIST_LEN,\
    ELE_PER_FRACTAL_EDGE, MAX_C_SIZE, PIPE_MTE3, PIPE_MTE2, BYTE_PER_FRACTAL,\
    MAX_BLK_STRIDE_SINGLE_BYTE, VALUE_128, VALUE_127, LOAD_SMASK_OFFSET_LIST,\
    LOAD_SMASK_SEGMENT_LIST, MAX_NBURST_DOUBLE_BYTE, MIN_BURST_LEN,\
    MAX_BURST_LEN_DOUBLE_BYTE, MAX_PADMODE, PADMODE_NO_PADDING, INC_MODE,\
    DEC_MODE, HAS_PARAM_CONCAT, NEED_PARAM_CONCAT,\
    PADDING_ONE_BYTE_OFFSET_LIST, PADDING_ONE_BYTE_SEGMENT_LIST,\
    PADDING_TWO_BYTE_OFFSET_LIST, PADDING_TWO_BYTE_SEGMENT_LIST, MAX_C1_INDEX,\
    MAX_JUMP_OFFSET, MIN_JUMP_OFFSET, MAX_BURST_LEN_SINGLE_BYTE, STRIDES_LEN,\
    DST_BLK_STRIDE_IDX, SRC_BLK_STRIDE_IDX, DEQSCALE_SHIFT_POS,\
    SCALE_SHIFT_POS, TENSOR_PADDING_OFFSET_LIST, TENSOR_PADDING_SEGMENT_LIST,\
    THREE_IR, VSCATTER_VGATHER_XT_OFFSET_LIST, \
    VSCATTER_VGATHER_XT_SEGMENT_LIST, UINT_MIN, UINT8_MAX, \
    INT8_MIN, INT8_MAX, AIPP0_OFFSET_LIST, \
    AIPP0_SEGMENT_LIST, AIPP1_OFFSET_LIST, AIPP1_SEGMENT_LIST, \
    AIPP2_OFFSET_LIST, AIPP2_SEGMENT_LIST, AIPP3_OFFSET_LIST, \
    AIPP3_SEGMENT_LIST, AIPP4_OFFSET_LIST, AIPP4_SEGMENT_LIST, \
    AIPP5_OFFSET_LIST, AIPP5_SEGMENT_LIST, AIPP6_OFFSET_LIST, \
    AIPP6_SEGMENT_LIST, AIPP7_OFFSET_LIST, AIPP7_SEGMENT_LIST, \
    AIPP8_OFFSET_LIST, AIPP8_SEGMENT_LIST, AIPP9_OFFSET_LIST, \
    AIPP9_SEGMENT_LIST, AIPP10_OFFSET_LIST, AIPP10_SEGMENT_LIST, \
    AIPP11_OFFSET_LIST, AIPP11_SEGMENT_LIST, AIPP12_OFFSET_LIST, \
    AIPP12_SEGMENT_LIST, \
    AIPP13_OFFSET_LIST, AIPP13_SEGMENT_LIST, AIPP15_OFFSET_LIST, \
    AIPP15_SEGMENT_LIST, AIPP16_OFFSET_LIST, AIPP16_SEGMENT_LIST, \
    AIPP17_OFFSET_LIST, AIPP17_SEGMENT_LIST, BIT_16, YUV420, XRGB8888, \
    NC1HWC0DI_INT8, NC1HWC0DI_FP16, RGB888, ARGB8888, YUYV, YUV422, AYUV444, \
    YUV400, RAW10, RAW12, RAW16, RAW24, AIPP_INPUT_VERSON_AND_FUNCTION, \
    SWAP, CSC, DTC, AERA_PADDING, PRE_CLIP, SCF, POST_CLIP, FLIP, STRETCH, \
    RAW, AIPP_INPUT_TYPE_SWAP_ALIGN, AIPP_FORMAT_CONVERT, AIPP_INIT_VALUE, \
    AIPP_ENABLE, AIPP_DISABLE, AIPP_INIT_FLOAT_VALUE_ZERO, \
    AIPP_INIT_FLOAT_VALUE_ONE, SCALE_COF, RAW_TO_16_N, \
    INSTR_DTYPE_SUPPORT_STATEMENT
from .tik_params import WINO_WGT_OFFSET_LIST
from .tik_params import WINO_WGT_SEGMENT_LIST
from .tik_params import WINO_FM_XT_OFFSET_LIST
from .tik_params import WINO_FM_XT_SEGMENT_LIST
from .tik_params import WINO_FM_XM_OFFSET_LIST
from .tik_params import WINO_FM_XM_SEGMENT_LIST
from .tik_params import MAX_REP_DIR
from .tik_params import MIN_ONTHEFLY_MODE
from .tik_params import MAX_ONTHEFLY_MODE
from .tik_params import SHIFT_BIT_POS_8
from .tik_params import SHIFT_BIT_POS_2
from .tik_params import VALUE_3
from .tik_params import VALUE_1
from .tik_params import MAX_COL_INDIC
from .tik_params import MASK_LEN_CONTINOUS_MODE
from .tik_params import MASK_HIGH_IDX
from .tik_api_constants import DTYPE_MAP
from .tik_api_constants import SCOPE_MAP
from .tik_api_constants import LOAD3DV2_FUNC_MAP
from .tik_api_constants import VNCHWCONV_INSTR_APPENDIX_MAP
from .tik_api_constants import LOAD2D_DMA_LIST
from .tik_api_constants import CR_MODE_MAP
from .tik_api_constants import ARCHVERSION_ONTHEFLY
from .tik_api_constants import WINO_PAD_MAP
from .tik_api_util import check_repeat_times, set_ctrl_counter_mask, \
    reset_ctrl_counter_mask, check_stride_unit, do_load3d_padding, \
    check_weight_offset
from .tik_vector_scatter_api_ import check_scatter_address_overlap
from ..common.common_util import check_vector_stride, \
    check_load3dv2_channel_size, check_load3dv2_m_extension, \
    check_dilation_filter_size, check_load3dv2_k_extension, \
    vector_tensor_overflow_check, check_tensor_overflow, \
    check_address_overlapping, check_addr_overlap_v4dtrans, \
    float16format2uint16, check_dict_and_not_none, check_aipp_one_src_overflow,\
    check_aipp_two_src_overflow, aipp_get_enable_bit, \
    cal_extent_stride_unit_mask, check_wino_ft_params, get_mask_len, \
    get_vbi_src0_offset_need_size, check_vbi_overlap, \
    check_vbi_src1_tensor_overflow, get_vbi_src1_tensor_need_size
from ..common.tik_get_soc_name import get_soc_name
from ..common.tik_get_soc_name import get_soc_core_type
from .tik_check_util import TikCheckUtil
from .tik_source_info import source_info_decorator


_ADDR_MODE_BIT_INCREASE = 1
_ADDR_MODE_BIT_DECREASE = 0
_SRC_BURST_LEN_SIZE_ELE = 16
_DST_BURST_LEN_SIZE_ELE = 256
_SRC_GAP_SIZE_BYTE = 32
_DEFAULT_STRIDE = 0
_STRIDE_UNIT_ZERO = 0
_STRIDE_UNIT_ONE = 1


def _regen_tensor_mov_scope(src, dst, block_mode):
    """ regenerate src_scope and dst_scope in tensor_mov

    Parameters
    ----------
    src: src operator
    dst: dst operator
    block_mode

    Returns
    -------
    src_scope
    dst_scope
    """
    src_scope = SCOPE_MAP[src.scope]
    dst_scope = SCOPE_MAP[dst.scope]
    if src_scope == "cc":
        src_scope = src_scope + "_" + block_mode
    if dst_scope == "cc":
        dst_scope = dst_scope + "_" + block_mode
    return src_scope, dst_scope


def _calculate_winograd_ft_extent(repeat_times, src_rep_stride, dst_rep_stride,
                                  dst_blk_stride):
    """ calculate winograd_weight_transform src/dst extent

    Parameters
    ----------
    repeat_times: the number of iterations this instruction would be
                  executed
    src_rep_stride: source repeat stride between the base source addresses
                    of 2 successive iterations
    dst_rep_stride: destination repeat stride between the desitination
                    addresses of 2 successive interations
    dst_blk_stride: inner destination stride between the 4 weight matrixes
                    to be written into L0B in one single iteration in unit
                    of fractal matrix

    Returns
    -------
    src_extent
    dst_extent
    """
    # one instr reads 9 fractals from L1
    src_frac_num = 9
    src_extent = ((repeat_times - 1) * src_rep_stride + src_frac_num) * \
                 BYTE_PER_FRACTAL
    src_extent = Expr(src_extent).get()
    # one instr writes 4 fractals to L0B
    dst_frac_num = 4
    dst_extent = ((repeat_times - 1) * dst_rep_stride + (dst_frac_num - 1) *
                  dst_blk_stride + 1) * BYTE_PER_FRACTAL
    dst_extent = Expr(dst_extent).get()
    return src_extent, dst_extent


def cal_vbi_extent(mask_len, dst, src1,  # pylint: disable=R0913
                   src0_offset, horizontal_repeat_times, repeat_mode,
                   dst_blk_stride, vertical_repeat_offset,
                   vertical_repeat_times):
    """calculate dst/src0_offset/src1 extent for vbi instrction"""
    # cal extent
    dst_extent = Expr(get_vbi_dst_need_size(
        mask_len, ONE_BLK_SIZE // DTYPE_SIZE[dst.dtype],
        vertical_repeat_times, vertical_repeat_offset, dst_blk_stride)*\
                      DTYPE_SIZE[dst.dtype]).get()
    src0_offset_extent = Expr(get_vbi_src0_offset_need_size(
        src1.dtype, mask_len,
        vertical_repeat_times*horizontal_repeat_times)*\
                              DTYPE_SIZE[src0_offset.dtype]).get()
    src1_extent = Expr(get_vbi_src1_tensor_need_size(
        src1.dtype, mask_len, repeat_mode,
        vertical_repeat_times*horizontal_repeat_times)*\
                       DTYPE_SIZE[src1.dtype]).get()
    return dst_extent, src0_offset_extent, src1_extent


def get_vbi_mask_len(mask):
    """get vbi mask_len corresponding to mask

    Parameters
    ----------
    mask: Effective operation on element
    Returns
    -------
    mask_len: mask_len is None(mask is scalar-list) or
              imme(mask is imme/imme-list) or scalar(mask is scalar)
    """
    if not isinstance(mask, (list, tuple)):
        mask = [mask]
    if len(mask) == MASK_LEN_CONTINOUS_MODE:
        mask_len = mask[MASK_HIGH_IDX]
    else:
        if is_basic_expr(TikUtil.to_list(mask)):
            return None
        mask_len, _ = get_mask_len(mask)
    return mask_len


def get_vbi_dst_need_size(mask_len, block_len, vertical_repeat_times,
                          dst_repeat_offset, dst_blk_stride):
    """for vbi instruction, get dst operator need size, unit is element
    when mask is scalar/scalar-list, mask is considered as 128

    Parameters
    ----------
    mask_len: length between lowest digit and top effective digit of mask
    block_len: elements num per block(32B)
    vertical_repeat_times: repeat_times in vertical direction
    dst_repeat_offset: vertical repeat offset between dst address of
                       iterations in the vertical direction
    dst_blk_stride: destination block stride
    Returns
    -------
    max_offset
    """
    # for cal dst_extent when mask is scalar/scalar-list
    if mask_len is None or is_basic_expr(TikUtil.to_list(mask_len)):
        max_offset = (vertical_repeat_times - 1)*dst_repeat_offset + \
                     (BLK_NUM_PER_REP - 1)*dst_blk_stride*block_len + \
                     block_len
        return max_offset
    if mask_len % block_len == 0:
        max_offset = (vertical_repeat_times - 1)*dst_repeat_offset + \
                     (mask_len // block_len - 1)*dst_blk_stride*block_len + \
                     block_len
    else:
        max_offset = (vertical_repeat_times - 1)*dst_repeat_offset + \
                     (mask_len // block_len)*dst_blk_stride*block_len + \
                     mask_len % block_len
    return max_offset


def check_vbi_dst_offset_overflow(dst, src0_offset,  #pylint: disable=R0913
                                  mask_len, horizontal_repeat_times,
                                  vertical_repeat_times, dst_blk_stride,
                                  dst_repeat_offset):
    """for vbi instruction, check whether dst and src0_offset is overflow

    Parameters
    ----------
    dst: operator dst
    src0_offset: operator src0_offset
    mask_len: length between lowest digit and top effective digit of mask
    horizontal_repeat_times: repeat_times in horizontal direction
    vertical_repeat_times: repeat_times in vertical direction
    dst_blk_stride: destination block stride
    dst_repeat_offset: vertical repeat offset between dst address of
                       iterations in the vertical direction
    Returns
    -------
    None
    """
    if mask_len is None or is_basic_expr(TikUtil.to_list(mask_len)):
        return
    if is_basic_expr([horizontal_repeat_times, vertical_repeat_times]):
        return
    total_repeat_times = Expr(horizontal_repeat_times*\
                              vertical_repeat_times).eval_value()
    if total_repeat_times is not None and total_repeat_times == 0:
        return
    block_len = ONE_BLK_SIZE // DTYPE_SIZE[dst.dtype]
    # check src0_offset tensor overflow
    valid_block_len = ceil_div(mask_len, block_len)
    check_tensor_overflow((src0_offset,), valid_block_len,
                          horizontal_repeat_times*vertical_repeat_times,
                          (MIN_STRIDE,), (MIN_STRIDE,),
                          ("src0_offset",))
    # check dst tensor overflow
    max_offset = get_vbi_dst_need_size(mask_len, block_len,
                                       vertical_repeat_times, dst_repeat_offset,
                                       dst_blk_stride)
    offset = dst.offset
    total_size = reduce_mul(dst.indice.origin_shape)
    need_offset = Expr(max_offset + offset).eval_value()
    if need_offset is not None:
        TikCheckUtil.check_le(need_offset, total_size,
                              "dst tensor overflow, instruction need {} but "
                              "only {}".format(need_offset, total_size))


def _check_dst_overflow_load3dv2(k_start_pt, m_start_pt, k_extension,
                                 m_extension, dst):
    """calculate src_extent and dst_extent of instruction load2dv1 and load2dv2

    Parameters
    ----------
    k_start_pt: k direction start position of the feature matrix
    m_start_pt: m direction start position of the feature matrix
    k_extension: k direction extension steps from the start position
    m_extension: m direction extension steps from the start position
    dst: destination operator

    Returns
    -------
    None
    """
    if all(Expr(value).eval_value() is not None for value in
           [k_start_pt, m_start_pt, k_extension, m_extension, dst.offset]):
        k_ext = ceil_div(k_extension*DTYPE_SIZE[dst.dtype], ONE_BLK_SIZE)*\
                ONE_BLK_SIZE // DTYPE_SIZE[dst.dtype]
        m_ext = ceil_div(m_extension, ELE_PER_FRACTAL_EDGE)*ELE_PER_FRACTAL_EDGE
        dst_expected_ele = Expr(k_ext*m_ext + dst.offset).eval_value()
        dst_actual_ele = reduce_mul(dst.indice.origin_shape)
        TikCheckUtil.check_ge(
            dst_actual_ele, dst_expected_ele,
            "dst tensor overflow, expected dst shape: {}, actual dst shape: {}"
            .format(dst_expected_ele, dst_actual_ele))


def _calculate_extent_load2d(start_index, repeat_times, src_stride, dst_gap):
    """calculate src_extent and dst_extent of instruction load2dv1 and load2dv2

    Parameters
    ----------
    start_index: start fractal index
    repeat_times: Repeated iterations times
    src_stride: gap of dst tensor between adjacent data segment
    dst_gap: stride of src tensor between adjacent data segment

    Returns
    -------
    src_extent
    dst_extent
    """
    src_extent = Expr((start_index + (repeat_times - 1) * src_stride + 1) *
                      BYTE_PER_FRACTAL).get()
    # repeat_times*512
    if dst_gap is None:
        dst_gap = 0
    dst_extent = Expr(repeat_times * BYTE_PER_FRACTAL + (repeat_times - 1) *
                      dst_gap * BYTE_PER_FRACTAL).get()
    return src_extent, dst_extent


def _load3dv2_check(k_extension, m_extension, m_start_pt, pad_value,
                    channel_size):
    # check extension
    TikCheckUtil.check_type_match(
        k_extension, (int, Scalar, Expr),
        "k_extension should be int, Scalar, Expr, input type of k_extension: "
        "{}".format(type(k_extension)))
    check_scalar_dtype(k_extension,
                       "scalar_k_extension should be a scalar of int/uint")
    check_integer_in_range(
        k_extension, range(MIN_EXTENSION, MAX_EXTENSION),
        "k_extension should be in the range of [1, 65535], "
        "input k_extension: {}".format(k_extension))
    TikCheckUtil.check_type_match(
        m_extension, (int, Scalar, Expr),
        "m_extension should be int, Scalar, Expr, input type of m_extension: "
        "{}".format(type(m_extension)))
    check_scalar_dtype(m_extension,
                       "scalar_m_extension should be a scalar of int/uint")
    check_integer_in_range(
        m_extension, range(MIN_EXTENSION, MAX_EXTENSION),
        "m_extension should be in the range of [1, 65535], "
        "input m_extension: {}".format(m_extension))
    TikCheckUtil.check_type_match(
        m_start_pt, (int, Scalar, Expr),
        "m_start_pt should be int, Scalar, Expr, input type of m_start_pt: "
        "{}".format(type(m_start_pt)))
    check_scalar_dtype(m_start_pt,
                       "scalar_m_start_pt should be a scalar of int/uint")
    check_integer_in_range(
        m_start_pt, range(MAX_START_PT),
        "m_start_pt should be in the range of [0, 65535], "
        "input m_start_pt: {}".format(m_start_pt))
    if isinstance(m_start_pt, int):
        m_start_pt_ele_align = 16
        if m_start_pt % m_start_pt_ele_align != 0:
            TikCheckUtil.raise_error(
                "m_start_ptshould be multiple of 16, input "
                "m_start_pt: {}".format(m_start_pt))
    # check pad_value
    if pad_value is not None:
        TikCheckUtil.check_type_match(
            pad_value, (int, float),
            "pad_value should be python int or float, input type of pad_value: "
            "{}".format(type(pad_value)))
    # check channel_size
    TikCheckUtil.check_type_match(
        channel_size, (int, Scalar, Expr),
        "channel_size should be int, Scalar or Expr, input type of "
        "channel_size:{}".format(type(channel_size)))
    check_scalar_dtype(channel_size,
                       "scalar_channel_size should be a scalar of int/uint")


def _calculate_dst_extent_load3dv1(dst, repeat_mode, repeat_time, jump_offset):
    """
    Calculate load3dv's dst extent.
    :param dst:  dst tensor
    :param repeat_mode: 0 or 1
    :param repeat_time: run times.
    :param jump_offset: offset that instruction jump
    :return: dst extent
    """
    if repeat_mode == 0:
        # dst_extent: repeat_time*512
        dst_extent = Expr(repeat_time*BYTE_PER_FRACTAL).get()
    elif repeat_mode == 1:
        # dst_extent: (repeat_time - 1)*jump_offset*512 + 512
        dst_extent = Expr((Expr(repeat_time) - 1)*Expr(jump_offset)*
                          BYTE_PER_FRACTAL + BYTE_PER_FRACTAL).get()
    else:
        dst_extent = Expr(reduce_mul(dst.indice.origin_shape)*
                          DTYPE_SIZE[dst.dtype]).get()
    return dst_extent


def _calculate_extent_load3dv1(dst, repeat_mode, repeat_time, jump_offset):
    """
    Calculate dst load3dv's extent by different type
    :param dst:  dst tensor
    :param repeat_mode: 0 or 1
    :param repeat_time: run times.
    :param jump_offset: offset that instruction jump
    :return: dst extent
    """
    # cal dst_extent, not support repeat_mode SCALAR
    repeat_mode = Expr(repeat_mode).eval_value()
    if repeat_mode is not None:
        dst_extent = _calculate_dst_extent_load3dv1(
            dst, repeat_mode, repeat_time, jump_offset)
    else:
        dst_extent = (reduce_mul(dst.indice.origin_shape) - dst.offset)*\
                     DTYPE_SIZE[dst.dtype]
        dst_extent = Expr(dst_extent).get()
    return dst_extent


def _get_scope_str(scope_map, block_mode, src, dst):
    """
    get scope in string format
    :param scope_map: the map store scope
    :param block_mode: data move's mode, like "m" "v" ...
    :param src: src tensor
    :param dst: dst tensor
    :return: scope string
    """
    src_scope = scope_map[src.scope]
    dst_scope = scope_map[dst.scope]
    if src_scope == "cc":
        src_scope = src_scope + "_" + block_mode + str(
            get_bit_len(src.dtype))
    if dst_scope == "cc":
        dst_scope = dst_scope + "_" + block_mode + str(
            get_bit_len(dst.dtype))
    return src_scope + '2' + dst_scope


def _calculate_extent(block_mode, src, dst, params, # pylint: disable=R0913
                      is_src, en_onthefly=False):
    """
    calculate data move instruction's extent
    :param block_mode: data move's mode, like "m" "v" ...
    :param src: src tensor
    :param dst: dst tensor
    :param params: args
    :param is_src: if calculate src or calculate dst
    :param en_onthefly: if enable on-the-fly
    :return: src_extent or dst_extent
    """
    # params: [nburst, burst_length, gap_block]
    burst_length_size = _cal_burst_len_size(block_mode, src, dst)
    if is_src:
        gap_block_size = _cal_src_gap_block_size(block_mode, src, dst)
    else:
        gap_block_size = _cal_dst_gap_block_size(block_mode, src, dst)
        if en_onthefly:
            # if enable on-the-fly, only low 8 bits store dst_stride
            params[2] = params[2] & 0x00FF

    extent = Expr(params[0]*params[1]*burst_length_size + (params[0] - 1)*
                  params[2]*gap_block_size)
    return extent.get()


def _calculate_extent_broadcast_ub_to_l0c(dst, src, nburst, burst_len,
                                          strides):
    """
    calculate extent of broadcast ub to l0c
    :param dst: dst tensor
    :param src: src tensor
    :param nburst: number of burst
    :param burst_len: burst length
    :param strides: src_stride and dst_stride
    :return: dst_extent and src_extent
    """
    # strides: [dst_stride, src_stride]
    # Byte
    src_extent = Expr((nburst*burst_len*_SRC_BURST_LEN_SIZE_ELE
                       *DTYPE_SIZE[src.dtype]) +
                      ((nburst - 1)*strides[1]*ONE_BLK_SIZE)).get()
    # Byte
    dst_extent = Expr((nburst*burst_len*_DST_BURST_LEN_SIZE_ELE
                       *DTYPE_SIZE[dst.dtype]) +
                      ((nburst - 1)*strides[0]*_DST_BURST_LEN_SIZE_ELE
                       *DTYPE_SIZE[dst.dtype])).get()
    return [dst_extent, src_extent]


def _cal_burst_len_size(block_mode, src, dst):
    """
    get the space of per burst length
    :param block_mode: data move's mode, like "m" "v" ...
    :param src: src tensor
    :param dst: dst tensor
    :return: burst len size. Byte
    """
    scope_str = _get_scope_str(SCOPE_MAP, block_mode, src, dst)
    # Byte. according to ISA DMA table
    burst_len_size_map = {
        "cc_m322ubuf": 1024,
        "cc_m162ubuf": 512,
        "cc_v322ubuf": 64,
        "cc_v162ubuf": 32,
        "cc_sc32ubuf": 256,
        "cc_dp16ubuf": 256,
        "cc_dp32ubuf": 512,
        "ubuf2cc_m32": 1024,
        "ubuf2cc_m16": 512,
        "ubuf2cc_v32": 64,
        "ubuf2cc_v16": 32,
        "ubuf2cc_sc32": 256,
    }
    if scope_str == "cc_dp32ubuf":
        if src.dtype == "f32":
            return 512
        if src.dtype == "s32":
            return 1024
    else:
        if scope_str in burst_len_size_map:
            return burst_len_size_map.get(scope_str)
    return 32


def _cal_src_gap_block_size(block_mode, src, dst):
    """
    get the space of per src gap block
    :param block_mode: data move's mode, like "m" "v" ...
    :param src: src tensor
    :param dst: dst tensor
    :return: src block size. Byte
    """
    scope_str = _get_scope_str(SCOPE_MAP, block_mode, src, dst)
    src_gap_size_map = {
        "cc_m322ubuf": 1024,
        "cc_m162ubuf": 512,
        "cc_v322ubuf": 1024,
        "cc_v162ubuf": 512,
        "cc_sc32ubuf": 256,
        "cc_dp16ubuf": 256,
        "cc_dp32ubuf": 512,
    }
    if scope_str == "cc_dp32ubuf":
        if src.dtype == "f32":
            return 512
        if src.dtype == "s32":
            return 1024
    else:
        if scope_str in src_gap_size_map:
            return src_gap_size_map.get(scope_str)
    return 32


def _cal_dst_gap_block_size(block_mode, src, dst):
    """
    get the space of per dst gap block
    :param block_mode: data move's mode, like "m" "v" ...
    :param src: src tensor
    :param dst: dst tensor
    :return: dst block size. Byte
    """
    scope_str = _get_scope_str(SCOPE_MAP, block_mode, src, dst)
    dst_gap_size_map = {
        "ubuf2cc_m32": 1024,
        "ubuf2cc_m16": 512,
        "ubuf2cc_v32": 1024,
        "ubuf2cc_v16": 512,
        "ubuf2cc_sc32": 256,
    }
    if scope_str in dst_gap_size_map:
        return dst_gap_size_map.get(scope_str)
    return 32


def _check_src_overflow_brc(src, nburst, burst_len, src_gap):
    """
    check src tensor if overflow
    :param src: src tensor
    :param nburst: number of burst
    :param burst_len: burst length
    :param src_gap: src gap
    :return: None
    """
    offset = src.offset
    total_size = reduce_mul(src.indice.origin_shape)
    byte_len = DTYPE_SIZE[src.dtype]
    extend_offset = nburst*(burst_len*_SRC_BURST_LEN_SIZE_ELE +
                            src_gap*_SRC_GAP_SIZE_BYTE // byte_len) -\
                    src_gap*_SRC_GAP_SIZE_BYTE // byte_len
    if Expr(extend_offset + offset).eval_value() is not None:
        TikCheckUtil.check_le(Expr(extend_offset + offset).eval_value(),
                              total_size, "src tensor overflow")


def _check_dst_overflow_brc(dst, nburst, burst_len, dst_gap):
    """
    check dst tensor if overflow
    :param dst: dst tensor
    :param nburst: number of burst
    :param burst_len: burst length
    :param dst_gap: dst gap
    :return: None
    """
    offset = dst.offset
    total_size = reduce_mul(dst.indice.origin_shape)
    extend_offset = nburst*(burst_len + dst_gap)*_DST_BURST_LEN_SIZE_ELE -\
                    dst_gap*_DST_BURST_LEN_SIZE_ELE
    if Expr(extend_offset + offset).eval_value() is not None:
        TikCheckUtil.check_le(Expr(extend_offset + offset).eval_value(),
                              total_size, "dst tensor overflow")


def _dtype_convert(value, dtype):
    """Get target's scope

    Parameters
    ----------
    name : str, The scope name

    Returns
    -------
    str : the key of scope
    """
    valuet = type_convert(value)
    return valuet.astype(dtype)


def _load3dv1_load3dv2_col2img_check(pad, l1_w, l1_h,  # pylint: disable=R0913
                                     stride_w, stride_h, filter_w, filter_h,
                                     dilation_filter_w, dilation_filter_h):
    """check load3dv2's and col2img's params"""
    # check pad
    TikCheckUtil.check_type_match(
        pad, (list, tuple), "pad_list should be list or tuple, please specify "
                            "padding: [left, right, top, bottom].")
    TikCheckUtil.check_equality(len(pad), PAD_LENGTH,
                                "pad length should be 4, input pad length: "
                                "{}".format(len(pad)))
    TikCheckUtil.check_type_match(pad[PADDING_LEFT_IDX], (int, Scalar, Expr),
                                  "pad[0] should be int, Scalar or Expr")
    check_scalar_dtype(pad[PADDING_LEFT_IDX],
                       "scalar_pad[0] should be a scalar of int/uint")
    check_integer_in_range(pad[PADDING_LEFT_IDX], range(MAX_PADDING),
                           "pad[0] should be in the range of [0, 255], "
                           "input pad[0]: {}".format(pad[PADDING_LEFT_IDX]))
    TikCheckUtil.check_type_match(pad[PADDING_RIGHT_IDX], (int, Scalar, Expr),
                                  "pad[1] should be int, Scalar or Expr")
    check_scalar_dtype(pad[PADDING_RIGHT_IDX],
                       "scalar_pad[1] should be a scalar of int/uint")
    check_integer_in_range(pad[PADDING_RIGHT_IDX], range(MAX_PADDING),
                           "pad[1] should be in the range of [0, 255],"
                           "input pad[1]: {}".format(pad[PADDING_RIGHT_IDX]))
    TikCheckUtil.check_type_match(pad[PADDING_TOP_IDX], (int, Scalar, Expr),
                                  "pad[2] should be int, Scalar or Expr")
    check_scalar_dtype(pad[PADDING_TOP_IDX],
                       "scalar_pad[2] should be a scalar of int/uint")
    check_integer_in_range(pad[PADDING_TOP_IDX], range(MAX_PADDING),
                           "pad[2] should be in the range of [0, 255], "
                           "input pad[2]: {}".format(pad[PADDING_TOP_IDX]))
    TikCheckUtil.check_type_match(pad[PADDING_BOT_IDX], (int, Scalar, Expr),
                                  "pad[3] should be int, Scalar or Expr")
    check_scalar_dtype(pad[PADDING_BOT_IDX],
                       "scalar_pad[3] should be a scalar of int/uint")
    check_integer_in_range(pad[PADDING_BOT_IDX], range(MAX_PADDING),
                           "pad[3] should be in the range of [0, 255], "
                           "input pad[3]: {}".format(pad[PADDING_BOT_IDX]))
    # check feature map
    TikCheckUtil.check_type_match(l1_w, (int, Scalar, Expr),
                                  "l1_w should be int, Scalar or Expr")
    check_scalar_dtype(l1_w,
                       "scalar_l1_w should be a scalar of int/uint")
    check_integer_in_range(l1_w, range(MIN_TENSOR_WIDTH, MAX_TENSOR_WIDTH),
                           "l1_w should be in the range of [1, 32767], "
                           "input value is %s" % str(l1_w))
    TikCheckUtil.check_type_match(l1_h, (int, Scalar, Expr),
                                  "l1_h should be int, Scalar or Expr")
    check_scalar_dtype(l1_h,
                       "scalar_l1_h should be a scalar of int/uint")
    check_integer_in_range(l1_h, range(MIN_TENSOR_HEIGHT, MAX_TENSOR_HEIGHT),
                           "l1_h should be in the range of [1, 32767]")
    # check stride
    TikCheckUtil.check_type_match(stride_w, (int, Scalar, Expr),
                                  "stride_w should be int, Scalar or Expr")
    check_scalar_dtype(stride_w,
                       "scalar_stride_w should be a scalar of int/uint")
    check_integer_in_range(stride_w, range(MIN_STRIDE, MAX_STRIDE),
                           "stride_w should be in the range of [1, 63], "
                           "input stride_w: {}".format(stride_w))
    TikCheckUtil.check_type_match(stride_h, (int, Scalar, Expr),
                                  "stride_h should be int, Scalar or Expr")
    check_scalar_dtype(stride_h,
                       "scalar_stride_h should be a scalar of int/uint")
    check_integer_in_range(stride_h, range(MIN_STRIDE, MAX_STRIDE),
                           "stride_h should be in the range of [1, 63], "
                           "input stride_h: {}".format(stride_h))
    # check filter
    TikCheckUtil.check_type_match(filter_w, (int, Scalar, Expr),
                                  "filter_w should be int, Scalar or Expr")
    check_scalar_dtype(filter_w,
                       "scalar_filter_w should be a scalar of int/uint")
    check_integer_in_range(filter_w, range(MIN_FILTER_WIDTH, MAX_FILTER_WIDTH),
                           "filter_w should be in the range of [1, 255], "
                           "input filter_w: {}".format(filter_w))
    TikCheckUtil.check_type_match(filter_h, (int, Scalar, Expr),
                                  "filter_h should be int, Scalar or Expr")
    check_scalar_dtype(filter_h,
                       "scalar_filter_h should be a scalar of int/uint")
    check_integer_in_range(
        filter_h, range(MIN_FILTER_HEIGHT, MAX_FILTER_HEIGHT),
        "filter_h should be in the range of [1, 255], "
        "input filter_h: {}".format(filter_h))
    # check dilation
    TikCheckUtil.check_type_match(
        dilation_filter_w, (int, Scalar, Expr),
        "dilation_filter_w should be int, Scalar or Expr")
    check_scalar_dtype(dilation_filter_w,
                       "scalar_dilation_filter_w "
                       "should be a scalar of int/uint")
    check_integer_in_range(
        dilation_filter_w, range(MIN_DILATION, MAX_DILATION),
        "dilation_filter_w should be in the range of [1, 255], "
        "input dilation_filter_w: {}".format(dilation_filter_w))
    TikCheckUtil.check_type_match(
        dilation_filter_h, (int, Scalar, Expr),
        "dilation_filter_h should be int, Scalar or Expr")
    check_scalar_dtype(dilation_filter_h,
                       "scalar_dilation_filter_h should be a scalar of"
                       " int/uint")
    check_integer_in_range(
        dilation_filter_h, range(MIN_DILATION, MAX_DILATION),
        "dilation_filter_h should be in the range of [1, 255], "
        "input dilation_filter_h: {}".format(dilation_filter_h))


def _load3dv1_col2img_check(fetch_filter_w, fetch_filter_h, left_top_w,
                            left_top_h):
    """check load3dv1's and col2img's params"""
    # check fetch pos in filter
    TikCheckUtil.check_type_match(
        fetch_filter_w, (int, Scalar, Expr),
        "fetch_filter_w should be int, Scalar or Expr")
    check_scalar_dtype(fetch_filter_w,
                       "scalar_fetch_filter_w should be a scalar of int/uint")
    check_integer_in_range(fetch_filter_w, range(MAX_FETCH_POS),
                           "fetch_filter_w should be in the range of [0, 255], "
                           "input fetch_filter_w: {}".format(fetch_filter_w))
    TikCheckUtil.check_type_match(
        fetch_filter_h, (int, Scalar, Expr),
        "fetch_filter_h should be int, Scalar or Expr")
    check_scalar_dtype(fetch_filter_h,
                       "scalar_fetch_filter_h should be a scalar of int/uint")
    check_integer_in_range(fetch_filter_h, range(MAX_FETCH_POS),
                           "fetch_filter_h should be in the range of [0, 255], "
                           "input fetch_filter_h: {}".format(fetch_filter_h))
    # check start-point
    TikCheckUtil.check_type_match(left_top_h, (int, Scalar, Expr),
                                  "left_top_h should be int, Scalar or Expr")
    check_scalar_dtype(left_top_h,
                       "scalar_left_top_h should be a scalar of int/uint")
    check_integer_in_range(left_top_h, range(MIN_START_POINT, MAX_START_POINT),
                           "left_top_h should be in the range of [-255, 32767],"
                           " input left_top_h: {}".format(left_top_h))
    TikCheckUtil.check_type_match(left_top_w, (int, Scalar, Expr),
                                  "left_top_w should be int, Scalar or Expr")
    check_scalar_dtype(left_top_w,
                       "scalar_left_top_w should be a scalar of int/uint")
    check_integer_in_range(left_top_w, range(MIN_START_POINT, MAX_START_POINT),
                           "left_top_w should be in the range of [-255, 32767],"
                           " input left_top_w: {}".format(left_top_w))


def _get_s32f16_deq_mode(deqscale):
    """
    get deq mode when dtype is s32f16
    :param deqscale: deqscale
    :return: deq mode
    """
    if not isinstance(deqscale, (float, int)):
        if not (isinstance(deqscale, (Scalar, Tensor))
                and (deqscale.dtype in ('uint64', 'float16'))):
            TikCheckUtil.raise_error("deqscale type error.")
    if isinstance(deqscale, float) \
            or (isinstance(deqscale, Scalar)
                    and (deqscale.dtype in ('float16',))):  # deq
        deq_mode = 'deq'
    elif isinstance(deqscale, Tensor)\
            and (deqscale.dtype in ('float16',)):  # vdeq
        deq_mode = 'vdeq'
    elif isinstance(deqscale, int)\
            or (isinstance(deqscale, Scalar)
                    and (deqscale.dtype in ('uint64',))):  # deq16
        deq_mode = 'deq16'
    else:
        deq_mode = 'vdeq16'
    return deq_mode


def _get_f16f16_deq_mode(deqscale):
    """
    get deq mode when dtype is f16f16
    :param deqscale: deqscale
    :return: deq mode
    """
    if deqscale is None:
        deq_mode = ''
    else:
        if not isinstance(deqscale, float):
            if not (isinstance(deqscale, Scalar) and
                    (deqscale.dtype in ('float16',))):
                TikCheckUtil.raise_error("deqscale type error.")
        deq_mode = 'deq'
    return deq_mode


def _get_s32s8u8_deq_mode(deqscale):
    """
    get deq mode when dtype is s32s8 or s32u8
    :param deqscale: deqscale
    :return: deq mode
    """
    if not isinstance(deqscale, int):
        if not (isinstance(deqscale, (Scalar, Tensor))
                and deqscale.dtype in 'uint64'):
            TikCheckUtil.raise_error("deqscale type error.")
    if isinstance(deqscale, (int, Scalar)):
        deq_mode = 'deq8'
    else:
        deq_mode = 'vdeq8'
    return deq_mode


def _get_s32s16_deq_mode(deqscale):
    """
    get deq mode when dtype is s32s16
    :param deqscale: deqscale
    :return: deq mode
    """
    if not isinstance(deqscale, int):
        if not (isinstance(deqscale, (Scalar, Tensor))
                and deqscale.dtype in 'uint64'):
            TikCheckUtil.raise_error("deqscale type error.")
    if isinstance(deqscale, (int, Scalar)):
        deq_mode = 'deqs16'
    else:
        deq_mode = 'vdeqs16'
    return deq_mode


def _make_deq_mode(dtype_str, deqscale):
    """get deq_mode"""
    # deq/vdeq/deq16/vdeq16
    if dtype_str in ("s32f16",):
        deq_mode = _get_s32f16_deq_mode(deqscale)
    # deq
    elif dtype_str in ("f16f16",):
        deq_mode = _get_f16f16_deq_mode(deqscale)
    # deq8/vdeq8
    elif dtype_str in ("s32s8", "s32u8"):
        deq_mode = _get_s32s8u8_deq_mode(deqscale)
    # deqs16/vdeqs16
    elif dtype_str in ("s32s16",):
        deq_mode = _get_s32s16_deq_mode(deqscale)
    else:
        deq_mode = ''
    return deq_mode


def _get_addr_list(dst_list, src_list, extents):
    """get addr list"""
    dst_addr_list0 = []
    dst_addr_list1 = []
    for index in range(MAX_VA_ADDR_NUM):
        get_addr_list(dst_addr_list0, dst_list[index], "w",
                      extent=extents[0])  # dst_extent
        get_addr_list(dst_addr_list1, dst_list[index + MAX_VA_ADDR_NUM], "w",
                      extent=extents[0])  # dst_extent

    src_addr_list0 = []
    src_addr_list1 = []
    for index in range(MAX_VA_ADDR_NUM):
        get_addr_list(src_addr_list0, src_list[index], "r",
                      extent=extents[1])  # src_extent
        get_addr_list(src_addr_list1, src_list[index + MAX_VA_ADDR_NUM], "r",
                      extent=extents[1])  # src_extent
    return dst_addr_list0, dst_addr_list1, src_addr_list0, src_addr_list1


def _get_load2d_dtype_str(src_scope, args, transpose_bit,  # pylint: disable=R0913
                          arch_version_str, addr_mode_bit, dtype_str, dst):
    """ get dtype str for load2d instruciton"""
    if src_scope in ('cbuf',):
        args.append(transpose_bit)
    if not arch_version_str in (ASCEND_310AIC,):
        args.append(addr_mode_bit)
    if dtype_str in ("s4s4", "u4u4"):
        dtype_str = "int4"
    elif dtype_str in ("s16s16", "u16u16"):
        dtype_str = "float16"
    else:
        dtype_str = dst.dtype
    return dtype_str


def _extend_args(param_key, args, argv):
    """extend args

    Parameters
    ----------
    param_key : param name
    args : to extend args
    argv : to extend args

    Returns
    -------
    the extended args
    """
    if args:
        if argv:
            TikCheckUtil.raise_error("argv should be None")
        TikCheckUtil.check_equality(len(args), 1, "args length should be 1")
        return args
    if argv:
        if not ((len(argv.key()) == 1) and (argv.get(param_key))):
            TikCheckUtil.raise_error("argv value error")
        TikCheckUtil.check_type_match(
            argv.get(param_key), (int, bool),
            "argv value of %s should be int or bool" % (param_key))
        return [argv.get(param_key)]
    return [0]


def check_dma_instr_params(dst, src, nburst, burst_len,  # pylint: disable=R0913
                           src_stride, dst_stride, en_onthefly=False,
                           src_onthefly_stride=0):
    """check params of data_move data_move_quant tensor_move"""
    TikCheckUtil.check_type_match(dst, Tensor, "dst should be tensor")
    TikCheckUtil.check_type_match(src, Tensor, "src should be tensor")
    TikCheckUtil.check_type_match(nburst, (int, Scalar, Expr),
                                  "nburst should be int, Scalar or Expr")
    check_scalar_dtype(nburst,
                       "scalar_nburst should be a scalar of int/uint")
    TikCheckUtil.check_type_match(burst_len, (int, Scalar, Expr),
                                  "burst_len should be int, Scalar or Expr")
    check_scalar_dtype(burst_len,
                       "scalar_burst_len should be a scalar of int/uint")
    TikCheckUtil.check_type_match(src_stride, (int, Scalar, Expr),
                                  "src_stride should be int, Scalar or Expr")
    check_scalar_dtype(src_stride,
                       "scalar_src_stride should be a scalar of int/uint")
    TikCheckUtil.check_type_match(dst_stride, (int, Scalar, Expr),
                                  "dst_stride should be int, Scalar or Expr")
    check_scalar_dtype(dst_stride,
                       "scalar_dst_stride should be a scalar of int/uint")
    check_integer_in_range(nburst, range(MIN_NBURST, MAX_NBURST_DOUBLE_BYTE),
                           "nburst should be in the range of [1, 4095], input "
                           "value is: {}".format(nburst))
    check_integer_in_range(
        burst_len, range(MIN_BURST_LEN, MAX_BURST_LEN_DOUBLE_BYTE),
        "burst_len should be in the range of [1, 65535], input value is: "
        "{}".format(burst_len))
    check_integer_in_range(src_stride, range(MAX_BLK_STRIDE_DOUBLE_BYTE),
                           "src_stride should be in the range of [0, 65535], "
                           "input value is: {}".format(src_stride))
    if en_onthefly:
        check_integer_in_range(
            dst_stride, range(MAX_BLK_STRIDE_SINGLE_BYTE),
            "dst_stride should be in the range of [0, 255], input value is: "
            "{}".format(dst_stride))
        check_integer_in_range(
            src_onthefly_stride, range(VALUE_1),
            "tensor_mov onthefly doesn't support src_onthefly_stride which is "
            "greater than 0, input src_onthefly_stride: %s" %
            str(src_onthefly_stride))
        check_integer_in_range(
            src_onthefly_stride, range(MAX_BLK_STRIDE_SINGLE_BYTE),
            "src_onthefly_stride should be in the range of [0, 255], input "
            "value is: %s" % (str(src_onthefly_stride)))
    else:
        check_integer_in_range(
            dst_stride, range(MAX_BLK_STRIDE_DOUBLE_BYTE),
            "dst_stride should be in the range of [0, 65535], input value is: "
            "{}".format(dst_stride))


def _aipp_check_arch_version(arch_version):
    """check arch version"""
    if arch_version not in [ASCEND_310AIC, ASCEND_910AIC,
                            HI3796CV300ESAIC, AIC]:
        TikCheckUtil.raise_error(
            "arch_version not support aipp, "
            "arch_version: {}".format(arch_version))


def _aipp_check_dst(input_format, dst):
    """check dst tensor"""
    TikCheckUtil.check_type_match(dst, Tensor,
                                  "dst should be tensor, "
                                  "input: {}".format(type(dst)))
    dst_scope = SCOPE_MAP[dst.scope]
    TikCheckUtil.check_equality(dst_scope, "cbuf",
                                "dst scope must be cubf, "
                                "input: {}".format(dst_scope))

    TikCheckUtil.check_in_range(dst.dtype,
                                ["uint8", "int8", "float16"],
                                "dst type error, "
                                "input type: {}".format(dst.dtype))
    imm_input_format = Expr(input_format).eval_value()
    if imm_input_format is not None:
        if imm_input_format == NC1HWC0DI_INT8:
            TikCheckUtil.check_equality(dst.dtype, "int8",
                                        "NC1HWC0DI_INT8 dst type must int8, "
                                        "input type: {}".format(dst.dtype))
        elif imm_input_format == NC1HWC0DI_FP16:
            TikCheckUtil.check_equality(dst.dtype, "float16",
                                        "NC1HWC0DI_FP16 dst type must float16, "
                                        "input type: {}".format(dst.dtype))


def _aipp_check_src(input_format, src0, src1, src_horizontal_size,
                    src_vertical_size):
    """check src tensor"""
    TikCheckUtil.check_type_match(src0, Tensor,
                                  "src0 should be Tensor, "
                                  "input: {}".format(type(src0)))
    src0_scope = SCOPE_MAP[src0.scope]
    TikCheckUtil.check_equality(src0_scope, "gm",
                                "src0 scope must be gm, "
                                "input scope: {}".format(src0_scope))

    imm_input_format = Expr(input_format).eval_value()
    if imm_input_format is not None:
        TikCheckUtil.check_in_range(
            src0.dtype,
            AIPP_INPUT_TYPE_SWAP_ALIGN[imm_input_format]["type"],
            "src0 type error, input type: {}".format(src0.dtype))

        imm_src_horizontal_size = Expr(src_horizontal_size).eval_value()
        imm_src_vertical_size = Expr(src_vertical_size).eval_value()
        if src1 is None:
            # check src extend
            if imm_src_horizontal_size is not None and \
                    imm_src_vertical_size is not None:
                check_aipp_one_src_overflow(src0, imm_input_format,
                                            imm_src_horizontal_size,
                                            imm_src_vertical_size)
        else:

            TikCheckUtil.check_type_match(src1, Tensor,
                                          "src1 should be Tensor, "
                                          "input: {}".format(type(src1)))
            src1_scope = SCOPE_MAP[src1.scope]
            TikCheckUtil.check_equality(src1_scope, "gm",
                                        "src1 scope must be gm, "
                                        "input scope: {}".format(src1_scope))

            # two input
            if imm_input_format in [YUV420, NC1HWC0DI_INT8,
                                    NC1HWC0DI_FP16, YUV422]:
                TikCheckUtil.check_in_range(
                    src1.dtype,
                    AIPP_INPUT_TYPE_SWAP_ALIGN[imm_input_format]["type"],
                    "src1 type error, input type: {}".format(src1.dtype))

                # check src extend
                if imm_src_horizontal_size is not None and \
                        imm_src_vertical_size is not None:
                    check_aipp_two_src_overflow(src0, src1, imm_input_format,
                                                imm_src_horizontal_size,
                                                imm_src_vertical_size)

            else:
                TikCheckUtil.raise_error("input_format not support two src")


def _aipp_check_input_format(arch_version, input_format):
    """check input format"""
    imm_input_format = Expr(input_format).eval_value()
    if imm_input_format is not None:
        format_ = AIPP_INPUT_VERSON_AND_FUNCTION.get(arch_version).get(
            imm_input_format)

        if format_ is None:
            TikCheckUtil.raise_error(
                "arch_version not support "
                "input_format: {}".format(imm_input_format))


def _aipp_check_function_switch(arch_version, input_format,  # pylint: disable=R0913
                                swap_enable, csc_enable, dtc_enable,
                                area_pad_enable, pre_clip_enable, scf_enable,
                                post_clip_enable, flip_enable, stretch_enable,
                                raw_enable):
    """check aipp function switch"""
    # function's input params is too much, so disable them
    imm_input_format = Expr(input_format).eval_value()
    if imm_input_format is not None:
        functions = AIPP_INPUT_VERSON_AND_FUNCTION.get(arch_version).get(
            imm_input_format)

        if pre_clip_enable == 1:
            TikCheckUtil.check_in_range(PRE_CLIP, functions,
                                        "pre clip not support, "
                                        "input: {}".format(functions))

        if swap_enable == 1:
            TikCheckUtil.check_in_range(SWAP, functions,
                                        "swap not support, "
                                        "input: {}".format(functions))

        if csc_enable == 1:
            TikCheckUtil.check_in_range(CSC, functions,
                                        "csc not support, "
                                        "input: {}".format(functions))

        if scf_enable == 1:
            TikCheckUtil.check_in_range(SCF, functions,
                                        "scf not support, "
                                        "input: {}".format(functions))

        if post_clip_enable == 1:
            TikCheckUtil.check_in_range(POST_CLIP, functions,
                                        "post clip not support, "
                                        "input: {}".format(functions))

        if dtc_enable == 1:
            TikCheckUtil.check_in_range(DTC, functions,
                                        "dtc not support, "
                                        "input: {}".format(functions))

        if area_pad_enable == 1:
            TikCheckUtil.check_in_range(AERA_PADDING, functions,
                                        "area_pad not support, "
                                        "input: {}".format(functions))

        if stretch_enable == 1:
            TikCheckUtil.check_in_range(STRETCH, functions,
                                        "stretch not support, "
                                        "input: {}".format(functions))

        if raw_enable == 1:
            TikCheckUtil.check_in_range(RAW, functions,
                                        "raw channel raw not support, "
                                        "input: {}".format(functions))

        if flip_enable == 1:
            TikCheckUtil.check_in_range(FLIP, functions,
                                        "flip not support, "
                                        "input: {}".format(functions))


def _aipp_check_src_info(arch_version, input_format, src_horizontal_size,
                         src_vertical_size):
    """check src tensor info"""

    TikCheckUtil.check_type_match(
        src_horizontal_size, (int, Scalar, Expr),
        "src_horizontal_size type error, "
        "input type {}".format(type(src_horizontal_size)))
    check_scalar_dtype(src_horizontal_size,
                       "src_horizontal_size should be a scalar of int/uint")
    TikCheckUtil.check_type_match(
        src_vertical_size, (int, Scalar, Expr),
        "src_vertical_size type error, "
        "input type: {}".format(src_vertical_size))
    check_scalar_dtype(src_vertical_size,
                       "src_vertical_size should be a scalar of int/uint")

    imm_src_horizontal_size = Expr(src_horizontal_size).eval_value()
    if imm_src_horizontal_size is not None:
        TikCheckUtil.check_in_range(
            imm_src_horizontal_size, range(8, 4097),
            "horizontal_resolution should in "
            "[8, 4096], input: {}".format(imm_src_horizontal_size))

        imm_input_format = Expr(input_format).eval_value()
        if imm_input_format is not None:
            if imm_input_format in [YUV420, YUYV, YUV422]:
                TikCheckUtil.check_equality(
                    imm_src_horizontal_size % 2, 0,
                    "src_horizontal_size should be even, "
                    "input: {}".format(imm_src_horizontal_size))

            if arch_version == HI3796CV300ESAIC:
                if imm_input_format in [YUV420, YUV422,
                                        YUYV, YUV400, RAW10, RAW12,
                                        RAW16, RGB888]:
                    TikCheckUtil.check_equality(
                        imm_src_horizontal_size % 16, 0,
                        "src_horizontal_size should be multiple of 16,"
                        " input: {}".format(imm_src_horizontal_size))
                elif imm_input_format in [AYUV444, ARGB8888, XRGB8888]:
                    TikCheckUtil.check_equality(
                        imm_src_horizontal_size % 4, 0,
                        "src_horizontal_size should be multiple of 4,"
                        " input: {}".format(imm_src_horizontal_size))

    imm_src_vertical_size = Expr(src_vertical_size).eval_value()
    if imm_src_vertical_size is not None:
        TikCheckUtil.check_ge(imm_src_vertical_size, 1,
                              "src_vertical_size should more than 1, "
                              "input: {}".format(imm_src_vertical_size))


def _aipp_check_crop_info(input_format, single_line_mode,  # pylint: disable=R0913
                          horizontal_size, vertical_size,
                          horizontal_start, vertical_start):
    """check crop info"""
    # function's input params is too much, so disable them
    TikCheckUtil.check_type_match(
        single_line_mode, (int, Scalar, Expr),
        "single_line_mode type error, "
        "input type: {}".format(type(single_line_mode)))
    check_scalar_dtype(single_line_mode,
                       "single_line_mode should be a scalar of int/uint")
    TikCheckUtil.check_type_match(
        horizontal_size, (int, Scalar, Expr),
        "horizontal_size type error, "
        "input type: {}".format(type(horizontal_size)))
    check_scalar_dtype(horizontal_size,
                       "horizontal_size should be a scalar of int/uint")
    TikCheckUtil.check_type_match(
        vertical_size, (int, Scalar, Expr),
        "vertical_size type error, "
        "input type: {}".format(type(vertical_size)))
    check_scalar_dtype(vertical_size,
                       "vertical_size should be a scalar of int/uint")
    TikCheckUtil.check_type_match(
        horizontal_start, (int, Scalar, Expr),
        "horizontal_start type error, "
        "input type: {}".format(type(horizontal_start)))
    check_scalar_dtype(horizontal_start,
                       "horizontal_start should be a scalar of int/uint")
    TikCheckUtil.check_type_match(
        vertical_start, (int, Scalar, Expr),
        "vertical_start type error, "
        "input type: {}".format(type(vertical_start)))
    check_scalar_dtype(vertical_start,
                       "vertical_start should be a scalar of int/uint")

    imm_input_format = Expr(input_format).eval_value()
    imm_single_line_mode = Expr(single_line_mode).eval_value()
    if imm_input_format is not None:
        imm_horizontal_size = Expr(horizontal_size).eval_value()
        if imm_horizontal_size is not None:
            # even YUV420/YUV422 semi-plannar and YUYV packed
            if imm_input_format in [YUV420, YUYV, YUV422]:
                TikCheckUtil.check_equality(
                    imm_horizontal_size % 2, 0,
                    "horizontal_size should be even, "
                    "input: {}".format(imm_horizontal_size))
            # range
            TikCheckUtil.check_in_range(
                imm_horizontal_size, range(8, 4097),
                "horizontal_size should in [8, 4096], "
                "input: {}".format(imm_horizontal_size))

        imm_vertical_size = Expr(vertical_size).eval_value()
        if imm_vertical_size is not None:

            if imm_single_line_mode is not None and imm_single_line_mode == 0:
                # even YUV420
                if imm_input_format == YUV420:
                    TikCheckUtil.check_equality(
                        imm_vertical_size % 2, 0,
                        "vertical_size should be even, "
                        "input: {}".format(imm_vertical_size))

                # range
                TikCheckUtil.check_in_range(
                    imm_vertical_size, range(8, 4097),
                    "vertical_size should in [8, 4096], "
                    "input: {}".format(imm_vertical_size))

        imm_horizontal_start = Expr(horizontal_start).eval_value()
        if imm_horizontal_start is not None:
            # even YUV420/YUV422 semi-plannar and YUYV packed
            if imm_input_format in [YUV420, YUYV, YUV422, XRGB8888,
                                    NC1HWC0DI_INT8, NC1HWC0DI_FP16,
                                    RGB888, YUV400]:
                TikCheckUtil.check_equality(
                    imm_horizontal_start % 2, 0,
                    'horizontal_start should be even, '
                    'input: {}'.format(imm_horizontal_start))

            # range
            TikCheckUtil.check_in_range(
                imm_horizontal_start, range(0, 4096),
                'horizontal_start should in [0, 4095], '
                'input: {}'.format(imm_horizontal_start))

        imm_vertical_start = Expr(vertical_start).eval_value()
        if imm_vertical_start is not None and imm_single_line_mode is not None:
            # even YUV420
            if imm_input_format in [YUV420, XRGB8888, NC1HWC0DI_INT8,
                                    NC1HWC0DI_FP16, RGB888,
                                    YUV400] and imm_single_line_mode == 0:
                TikCheckUtil.check_equality(
                    imm_vertical_start % 2, 0,
                    'vertical_start should be even, '
                    'input: {}'.format(imm_vertical_start))

            # range
            TikCheckUtil.check_in_range(
                imm_vertical_start, range(0, 4096),
                'vertical_start should in [0, 4095], '
                'input: {}'.format(imm_vertical_start))


def _aipp_check_crop_in_picture(src_horizontal_size,  # pylint: disable=R0913
                                src_vertical_size, horizontal_size,
                                vertical_size, horizontal_start,
                                vertical_start):
    """check crop in picture"""
    # function's input params is too much, so disable them
    imm_src_horizontal_size = Expr(src_horizontal_size).eval_value()
    imm_horizontal_start = Expr(horizontal_start).eval_value()
    if imm_src_horizontal_size is not None and imm_horizontal_start is not None:
        # range
        TikCheckUtil.check_in_range(
            imm_horizontal_start, range(0, imm_src_horizontal_size + 1),
            'horizontal_start should in src_horizontal_size, '
            'input: {}'.format(imm_horizontal_start))

        imm_horizontal_size = Expr(horizontal_size).eval_value()
        if imm_horizontal_size is not None:
            # range
            TikCheckUtil.check_in_range(
                imm_horizontal_start + imm_horizontal_size,
                range(0, imm_src_horizontal_size + 1),
                'horizontal_start+horizontal_size should in '
                'src_horizontal_size, '
                'input: {}'.format(imm_horizontal_start + imm_horizontal_size))

    imm_src_vertical_size = Expr(src_vertical_size).eval_value()
    imm_vertical_start = Expr(vertical_start).eval_value()
    if imm_src_vertical_size is not None and imm_vertical_start is not None:
        # range
        TikCheckUtil.check_in_range(
            imm_vertical_start, range(0, imm_src_vertical_size + 1),
            'imm_vertical_start should in src_vertical_size, '
            'imm_vertical_start: {}'.format(imm_vertical_start))

        imm_vertical_size = Expr(vertical_size).eval_value()
        if imm_vertical_size is not None:
            # range
            TikCheckUtil.check_in_range(
                imm_vertical_start + imm_vertical_size,
                range(0, imm_src_vertical_size + 1),
                "vertical_start + vertical_size should in src_vertical_size,"
                " vertical_start + vertical_size: {}".format(vertical_start +
                                                             vertical_size))


def _aipp_check_crop_single_line_mode(arch_version, single_line_mode):
    """check crop single line mode"""
    imm_single_line_mode = Expr(single_line_mode).eval_value()
    if imm_single_line_mode is not None:
        if arch_version in [ASCEND_910AIC, HI3796CV300ESAIC]:
            TikCheckUtil.check_equality(
                imm_single_line_mode, 0,
                'single_line_mode value error, '
                'input: {}'.format(imm_single_line_mode))
        else:
            TikCheckUtil.check_in_range(
                imm_single_line_mode, range(0, 2),
                'single_line_mode should in [0, 1], '
                'input: {}'.format(imm_single_line_mode))


def _check_crop_vertical_size_by_single_line(arch_version, crop_vertical_size,
                                             single_line_mode):
    if arch_version in [ASCEND_310AIC, AIC]:
        imm_single_line_mode = Expr(single_line_mode).eval_value()
        if imm_single_line_mode is not None and imm_single_line_mode == 1:
            imm_crop_vertical_size = Expr(crop_vertical_size).eval_value()
            if imm_crop_vertical_size is not None:
                TikCheckUtil.check_equality(
                    imm_crop_vertical_size, 1,
                    'crop_vertical_size should '
                    'be 1 when single_line_mode enable')


def _aipp_check_format_convert(arch_version, format_convert):
    """check format convert"""
    TikCheckUtil.check_type_match(
        format_convert, (int),
        "format_convert should be int, input: {}".format(type(format_convert)))

    if arch_version == HI3796CV300ESAIC:
        # range
        TikCheckUtil.check_in_range(
            format_convert,
            [0, 10, 11, 12, 13, 14, 15, 16, 17],
            'format_convert should be 0  or in [10, 17] for v200hisi-es, '
            'input: {}'.format(format_convert))
    else:
        TikCheckUtil.check_in_range(
            format_convert, range(0, 10),
            'format_convert should in [0, 9] '
            'for arch_version not hs, '
            'input: {} {}'.format(arch_version, format_convert))


def _check_matrix_type_and_range(matrix, shape, in_type, input_range=None):
    """check matrix type and range"""
    # matrix
    if len(matrix) != shape[0] or len(matrix[0]) != shape[1]:
        TikCheckUtil.raise_error("csc_matrix shape error,should 3*3")

    for i in range(shape[0]):
        for j in range(shape[1]):
            TikCheckUtil.check_type_match(
                matrix[i][j], in_type,
                "csc_matrix type error, input: {}".format(matrix[i][j]))
            check_scalar_dtype(matrix[i][j],
                               "matrix[" + str(i) + "][" + str(j) +
                               "] should be a scalar of int/uint")
            if input_range:
                imm_matrix = Expr(matrix[i][j]).eval_value()
                if imm_matrix is not None:
                    TikCheckUtil.check_in_range(
                        imm_matrix, input_range,
                        "matrix[" + str(i) + "][" + str(j) +
                        "] number out of range")


def _check_list_type_and_range(input_list, length, in_type, input_range=None,
                               name=''):
    """check list type and range"""
    if input_list is None:
        TikCheckUtil.raise_error(name + " is None")

    if len(input_list) != length:
        TikCheckUtil.raise_error(name + "input_list length error, "
                                        "input: {}".format(len(input_list)))

    for i in range(length):
        TikCheckUtil.check_type_match(
            input_list[i], in_type, name + '[' + str(i) + "] type error")

        if int in in_type:
            check_scalar_dtype(input_list[i], name + '[' +
                               str(i) + "] should be a scalar of int/uint")
            if input_range:
                imm_input = Expr(input_list[i]).eval_value()
                if imm_input is not None:
                    TikCheckUtil.check_in_range(
                        imm_input, input_range,
                        name + '[' + str(i) + "] out of range")
        else:
            check_scalar_dtype_float(input_list[i], name + '[' + str(i) +
                                     "] should be a scalar of float")


def _aipp_check_csc_info(csc_matrix, csc_out_bias, csc_in_bias):
    """check csc info"""
    _check_matrix_type_and_range(
        csc_matrix, [3, 3], (int, Scalar, Expr), range(-32768, 32768))
    _check_list_type_and_range(
        csc_out_bias, 3, (int, Scalar, Expr), range(0, 255), 'csc_out_bias')
    _check_list_type_and_range(
        csc_in_bias, 3, (int, Scalar, Expr), range(0, 255), 'csc_in_bias')


# csc info
def _get_csc_parameter(format_convert, csc_info):
    """get csc parameter"""
    if format_convert == 0:
        csc_matrix = csc_info.get('csc_matrix')
        csc_out_bias = csc_info.get('csc_out_bias')
        csc_in_bias = csc_info.get('csc_in_bias')

        _aipp_check_csc_info(csc_matrix, csc_out_bias, csc_in_bias)

    else:
        csc_para = AIPP_FORMAT_CONVERT.get(format_convert)
        csc_matrix = csc_para.get('csc_matrix')
        csc_out_bias = csc_para.get('csc_out_bias')
        csc_in_bias = csc_para.get('csc_in_bias')

    return csc_matrix, csc_out_bias, csc_in_bias


def _aipp_check_swap(input_format, rb_swap, uv_swap, ax_swap):
    imm_input_format = Expr(input_format).eval_value()
    if imm_input_format is not None:
        imm_rb_swap = Expr(rb_swap).eval_value()
        if imm_rb_swap is not None:
            TikCheckUtil.check_in_range(
                imm_rb_swap,
                AIPP_INPUT_TYPE_SWAP_ALIGN.get(imm_input_format).get('swap')[0],
                'rb_swap not support, '
                'input_format: {}'.format(imm_input_format))

        imm_uv_swap = Expr(uv_swap).eval_value()
        if imm_uv_swap is not None:
            TikCheckUtil.check_in_range(
                imm_uv_swap,
                AIPP_INPUT_TYPE_SWAP_ALIGN.get(imm_input_format).get('swap')[1],
                'uv_swap not support, '
                'input_format: {}'.format(imm_input_format))

        imm_ax_swap = Expr(ax_swap).eval_value()
        if imm_ax_swap is not None:
            TikCheckUtil.check_in_range(
                imm_ax_swap,
                AIPP_INPUT_TYPE_SWAP_ALIGN.get(imm_input_format).get('swap')[2],
                'ax_swap not support, '
                'input_format: {}'.format(imm_input_format))


def _aipp_check_pre_clip(pre_top_clip_number, pre_botton_clip_number,
                         crop_vertical_size):
    """check pre clip"""
    TikCheckUtil.check_type_match(
        pre_top_clip_number, (int, Scalar, Expr),
        "pre_top_clip_number should be int, Scalar, Expr, "
        "input: {}".format(type(pre_top_clip_number)))
    check_scalar_dtype(pre_top_clip_number,
                       "pre_top_clip_number should be a scalar of int/uint")
    TikCheckUtil.check_type_match(
        pre_botton_clip_number, (int, Scalar, Expr),
        "pre_botton_clip_number should be int, Scalar, Expr, "
        "input: {}".format(type(pre_botton_clip_number)))
    check_scalar_dtype(pre_botton_clip_number,
                       "pre_top_clip_number should be a scalar of int/uint")

    imm_pre_top_clip_number = Expr(pre_top_clip_number).eval_value()
    if imm_pre_top_clip_number is not None:
        TikCheckUtil.check_in_range(
            imm_pre_top_clip_number, range(0, 2),
            'pre_top_clip_number should in [0, 1], '
            'input: {}'.format(imm_pre_top_clip_number))

    imm_pre_botton_clip_number = Expr(pre_botton_clip_number).eval_value()
    if imm_pre_botton_clip_number is not None:
        TikCheckUtil.check_in_range(
            imm_pre_botton_clip_number, range(0, 2),
            'pre_botton_clip_number should in [0, 1], '
            'input: {}'.format(imm_pre_botton_clip_number))

    imm_crop_vertical_size = Expr(crop_vertical_size).eval_value()
    if imm_pre_top_clip_number is not None and imm_pre_botton_clip_number \
            is not None and imm_crop_vertical_size is not None:
        TikCheckUtil.check_ge(imm_crop_vertical_size,
                              imm_pre_top_clip_number +
                              imm_pre_botton_clip_number + 1,
                              'crop_vertical_size should more than 0, '
                              'after preclip, crop_vertical_size:'
                              '{}'.format(crop_vertical_size))


def _aipp_check_scf(scf_horizontal_size, scf_vertical_size,
                    scf_horizontal_start, scf_vertical_start, scaling_mode):
    TikCheckUtil.check_type_match(
        scf_horizontal_size, (int, Scalar, Expr),
        "scf_horizontal_size should be int, Scalar, Expr, "
        "input: {}".format(type(scf_horizontal_size)))
    check_scalar_dtype(scf_horizontal_size,
                       "src_horizontal_size should be a scalar of int/uint")

    imm_scf_horizontal_size = Expr(scf_horizontal_size).eval_value()
    if imm_scf_horizontal_size is not None:
        TikCheckUtil.check_in_range(
            imm_scf_horizontal_size, range(16, 1921),
            'scf_horizontal_size out of range, '
            'input:{}'.format(imm_scf_horizontal_size))

    TikCheckUtil.check_type_match(
        scf_vertical_size, (int, Scalar, Expr),
        "scf_vertical_size should be int, Scalar, Expr, "
        "input: {}".format(type(scf_vertical_size)))
    check_scalar_dtype(scf_vertical_size,
                       "scf_vertical_size should be a scalar of int/uint")

    imm_scf_vertical_size = Expr(scf_vertical_size).eval_value()
    if imm_scf_vertical_size is not None:
        TikCheckUtil.check_in_range(
            imm_scf_vertical_size, range(16, 1081),
            'scf_vertical_size out of range, '
            'input: {}'.format(imm_scf_vertical_size))

    TikCheckUtil.check_type_match(
        scf_horizontal_start, (int, Scalar, Expr),
        "scf_horizontal_start should be int, Scalar, Expr, "
        "input: {}".format(type(scf_horizontal_start)))
    check_scalar_dtype(scf_horizontal_start,
                       "scf_horizontal_start should be a scalar of int/uint")
    TikCheckUtil.check_type_match(
        scf_vertical_start, (int, Scalar, Expr),
        "scf_vertical_start should be int, Scalar, Expr, "
        "input: {}".format(scf_vertical_start))
    check_scalar_dtype(scf_vertical_start,
                       "scf_vertical_start should be a scalar of int/uint")
    TikCheckUtil.check_type_match(
        scaling_mode, (int, Scalar, Expr),
        "scaling_mode should be int, Scalar, Expr, "
        "input: {}".format(type(scaling_mode)))
    check_scalar_dtype(scaling_mode,
                       "scaling_mode should be a scalar of int/uint")

    imm_scaling_mode = Expr(scaling_mode).eval_value()
    if imm_scaling_mode is not None:
        TikCheckUtil.check_in_range(imm_scaling_mode, range(0, 2),
                                    'scaling_mode out of range, '
                                    'input: {}'.format(type(imm_scaling_mode)))


def _aipp_check_post_clip(post_botton_clip_number, post_top_clip_number,
                          post_right_clip_number, post_left_clip_number):
    """check post clip"""
    TikCheckUtil.check_type_match(
        post_botton_clip_number, (int, Scalar, Expr),
        "post_botton_clip_number should be int, Scalar, Expr, "
        "input: {}".format(type(post_botton_clip_number)))
    check_scalar_dtype(post_botton_clip_number,
                       "post_botton_clip_number should be a scalar of int/uint")
    imm_post_botton_clip_number = Expr(post_botton_clip_number).eval_value()
    if imm_post_botton_clip_number is not None:
        TikCheckUtil.check_in_range(
            imm_post_botton_clip_number, range(0, 64),
            'post_botton_clip_number out of range, '
            'input: {}'.format(imm_post_botton_clip_number))

    TikCheckUtil.check_type_match(
        post_top_clip_number, (int, Scalar, Expr),
        "post_top_clip_number should be int, Scalar, Expr, "
        "input: {}".format(type(post_top_clip_number)))
    check_scalar_dtype(post_top_clip_number,
                       "post_top_clip_number should be a scalar of int/uint")
    imm_post_top_clip_number = Expr(post_top_clip_number).eval_value()
    if imm_post_top_clip_number is not None:
        TikCheckUtil.check_in_range(
            imm_post_top_clip_number, range(0, 64),
            'post_top_clip_number out of range, '
            'input: {}'.format(imm_post_top_clip_number))

    TikCheckUtil.check_type_match(
        post_right_clip_number, (int, Scalar, Expr),
        "post_right_clip_number should be int, Scalar, Expr, "
        "input: {}".format(type(post_right_clip_number)))
    check_scalar_dtype(post_right_clip_number,
                       "post_right_clip_number should be a scalar of int/uint")
    imm_post_right_clip_number = Expr(post_right_clip_number).eval_value()
    if imm_post_right_clip_number is not None:
        TikCheckUtil.check_in_range(
            imm_post_right_clip_number, range(0, 64),
            'post_right_clip_number out of range, '
            'input: {}'.format(post_right_clip_number))

    TikCheckUtil.check_type_match(
        post_left_clip_number, (int, Scalar, Expr),
        "post_left_clip_number should be int, Scalar, Expr, "
        "input: {}".format(type(post_left_clip_number)))
    check_scalar_dtype(post_left_clip_number,
                       "post_left_clip_number should be a scalar of int/uint")
    imm_post_left_clip_number = Expr(post_left_clip_number).eval_value()
    if imm_post_left_clip_number is not None:
        TikCheckUtil.check_in_range(
            imm_post_left_clip_number, range(0, 64),
            'post_left_clip_number out of range, '
            'input: {}'.format(post_left_clip_number))


def _aipp_check_dtc_mean(dtc_mean_type, dtc_mean):
    """check dtc mean"""
    TikCheckUtil.check_type_match(
        dtc_mean_type, (int, Scalar, Expr),
        "dtc_mean_type type error, input: {}".format(type(dtc_mean_type)))
    check_scalar_dtype(dtc_mean_type,
                       "dtc_mean_type should be a scalar of int/uint")

    # dtc mean uint8/int8/b24
    imm_dtc_mean_type = Expr(dtc_mean_type).eval_value()
    if imm_dtc_mean_type is not None:
        TikCheckUtil.check_in_range(
            imm_dtc_mean_type, [0, 1],
            'dtc_mean_type out of range, input: {}'.format(imm_dtc_mean_type))

        if imm_dtc_mean_type == 0:
            _check_list_type_and_range(
                dtc_mean, 4, (int, Scalar, Expr), None, 'dtc_mean')
        else:
            _check_list_type_and_range(
                dtc_mean, 4, (float, Scalar, Expr), None, 'dtc_mean')


def _cal_dtc_mean_by_type(dtc_mean):
    """calculate dtc mean"""
    if Expr(dtc_mean).eval_value() is None:
        if dtc_mean.dtype == 'float16':
            dtc_mean_uint32 = dtc_mean.reinterpret_cast_to(
                'uint16')
        else:
            dtc_mean_uint32 = dtc_mean
    else:
        if isinstance(dtc_mean, float):
            dtc_mean_uint32 = float16format2uint16(dtc_mean)
        else:
            dtc_mean_uint32 = dtc_mean

    return dtc_mean_uint32


def _aipp_get_dtc_mean(dtc_mean):
    """get dtc mean"""
    dtc_mean0_uint32 = _cal_dtc_mean_by_type(dtc_mean[0])
    dtc_mean1_uint32 = _cal_dtc_mean_by_type(dtc_mean[1])
    dtc_mean2_uint32 = _cal_dtc_mean_by_type(dtc_mean[2])
    dtc_mean3_uint32 = _cal_dtc_mean_by_type(dtc_mean[3])

    return dtc_mean0_uint32, dtc_mean1_uint32, \
           dtc_mean2_uint32, dtc_mean3_uint32


def _aipp_get_dtc_min_value(dtc_min):
    """check get dtc min value"""
    imm_dtc_min0 = Expr(dtc_min[0]).eval_value()
    if imm_dtc_min0 is None:
        dtc_min0_uint32 = dtc_min[0].reinterpret_cast_to('uint16')
    else:
        dtc_min0_uint32_tmp = float16format2uint16(imm_dtc_min0)
        dtc_min0_uint32 = dtc_min0_uint32_tmp

    imm_dtc_min1 = Expr(dtc_min[1]).eval_value()
    if imm_dtc_min1 is None:
        dtc_min1_uint32 = dtc_min[1].reinterpret_cast_to('uint16')
    else:
        dtc_min1_uint32_tmp = float16format2uint16(imm_dtc_min1)
        dtc_min1_uint32 = dtc_min1_uint32_tmp

    imm_dtc_min2 = Expr(dtc_min[1]).eval_value()
    if imm_dtc_min1 is None:
        dtc_min2_uint32 = dtc_min[2].reinterpret_cast_to('uint16')
    else:
        dtc_min2_uint32_tmp = float16format2uint16(imm_dtc_min2)
        dtc_min2_uint32 = dtc_min2_uint32_tmp

    imm_dtc_min3 = Expr(dtc_min[1]).eval_value()
    if imm_dtc_min3 is None:
        dtc_min3_uint32 = dtc_min[3].reinterpret_cast_to('uint16')
    else:
        dtc_min3_uint32_tmp = float16format2uint16(imm_dtc_min3)
        dtc_min3_uint32 = dtc_min3_uint32_tmp

    return dtc_min0_uint32, dtc_min1_uint32, dtc_min2_uint32, dtc_min3_uint32


def _aipp_get_dtc_var_value(dtc_var):
    """get dtc var value"""
    imm_dtc_var0 = Expr(dtc_var[0]).eval_value()
    if imm_dtc_var0 is None:
        dtc_var0_uint32 = dtc_var[0].reinterpret_cast_to('uint16')
    else:
        dtc_var0_uint32_tmp = float16format2uint16(imm_dtc_var0)
        dtc_var0_uint32 = dtc_var0_uint32_tmp

    imm_dtc_var1 = Expr(dtc_var[0]).eval_value()
    if imm_dtc_var1 is None:
        dtc_var1_uint32 = dtc_var[1].reinterpret_cast_to('uint16')
    else:
        dtc_var1_uint32_tmp = float16format2uint16(imm_dtc_var1)
        dtc_var1_uint32 = dtc_var1_uint32_tmp

    imm_dtc_var2 = Expr(dtc_var[0]).eval_value()
    if imm_dtc_var2 is None:
        dtc_var2_uint32 = dtc_var[2].reinterpret_cast_to('uint16')
    else:
        dtc_var2_uint32_tmp = float16format2uint16(imm_dtc_var2)
        dtc_var2_uint32 = dtc_var2_uint32_tmp

    imm_dtc_var3 = Expr(dtc_var[0]).eval_value()
    if imm_dtc_var3 is None:
        dtc_var3_uint32 = dtc_var[3].reinterpret_cast_to('uint16')
    else:
        dtc_var3_uint32_tmp = float16format2uint16(imm_dtc_var3)
        dtc_var3_uint32 = dtc_var3_uint32_tmp

    return dtc_var0_uint32, dtc_var1_uint32, dtc_var2_uint32, dtc_var3_uint32


def _aipp_get_channel_pad_value(dst_type, channel0_pad_value,
                                channel1_pad_value, channel2_pad_value,
                                channel3_pad_value):
    """get channel pad value"""
    if dst_type == 'float16':
        imm_channel0_pad_value = Expr(channel0_pad_value).eval_value()
        if imm_channel0_pad_value is None:
            channel0_pad_value_uint32 = channel0_pad_value.reinterpret_cast_to(
                'uint16')
        else:
            channel0_pad_value_uint32 = float16format2uint16(
                imm_channel0_pad_value)

        imm_channel1_pad_value = Expr(channel0_pad_value).eval_value()
        if imm_channel1_pad_value is None:
            channel1_pad_value_uint32 = channel1_pad_value.reinterpret_cast_to(
                'uint16')
        else:
            channel1_pad_value_uint32 = float16format2uint16(
                imm_channel1_pad_value)

        imm_channel2_pad_value = Expr(channel0_pad_value).eval_value()
        if imm_channel2_pad_value is None:
            channel2_pad_value_uint32 = channel2_pad_value.reinterpret_cast_to(
                'uint16')
        else:
            channel2_pad_value_uint32 = float16format2uint16(
                imm_channel2_pad_value)

        imm_channel3_pad_value = Expr(channel0_pad_value).eval_value()
        if imm_channel3_pad_value is None:
            channel3_pad_value_uint32 = channel3_pad_value.reinterpret_cast_to(
                'uint16')
        else:
            channel3_pad_value_uint32 = float16format2uint16(
                imm_channel3_pad_value)
    else:
        channel0_pad_value_uint32 = channel0_pad_value
        channel1_pad_value_uint32 = channel1_pad_value
        channel2_pad_value_uint32 = channel2_pad_value
        channel3_pad_value_uint32 = channel3_pad_value

    return channel0_pad_value_uint32, channel1_pad_value_uint32, \
           channel2_pad_value_uint32, channel3_pad_value_uint32


def _aipp_check_dtc_raw_info(raw_to_f16_n):
    """check dtc raw info"""

    TikCheckUtil.check_type_match(
        raw_to_f16_n, (int, Scalar, Expr),
        "raw_to_f16_n type error, input: {}".format(type(raw_to_f16_n)))
    check_scalar_dtype(raw_to_f16_n,
                       "raw_to_f16_n should be a scalar of int/uint")
    imm_raw_to_f16_n = Expr(raw_to_f16_n).eval_value()
    if imm_raw_to_f16_n is not None:
        TikCheckUtil.check_in_range(
            imm_raw_to_f16_n, RAW_TO_16_N,
            'raw_to_f16_n value out range, input: {}'.format(imm_raw_to_f16_n))


def _aipp_check_flip_dict(arch_version_str, flip_mode):
    """
    check flip mode
    :param arch_version_str: arch_version_str
    :param flip_mode: 0-3
    :return: None
    """
    if arch_version_str in [HI3796CV300ESAIC]:
        TikCheckUtil.check_type_match(
            flip_mode, (int, Scalar, Expr),
            "flip_mode should be int, Scalar, Expr, "
            "input: {}".format(type(flip_mode)))
        check_scalar_dtype(flip_mode,
                           "flip_mode should be a scalar of int/uint")
        imm_flip_mode = Expr(flip_mode).eval_value()
        if imm_flip_mode is not None:
            TikCheckUtil.check_in_range(
                imm_flip_mode, range(4), 'flip_mode value out range, '
                                         'input: {}'.format(imm_flip_mode))


def _aipp_check_cpad(arch_version_str, dst, sfr_cpadding, cpadding_mode):
    """check cpadding"""

    TikCheckUtil.check_type_match(
        cpadding_mode, (int, Scalar, Expr),
        'cpadding_mode type error, input: {}'.format(type(cpadding_mode)))
    check_scalar_dtype(cpadding_mode,
                       "cpadding_mode should be a scalar of int/uint")

    imm_cpadding_mode = Expr(cpadding_mode).eval_value()
    if imm_cpadding_mode is not None:
        # C PADDING
        TikCheckUtil.check_in_range(
            imm_cpadding_mode, range(4), 'cpadding_mode out of range, '
                                         'input: {}'.format(imm_cpadding_mode))
        if imm_cpadding_mode == 1:
            TikCheckUtil.check_in_range(
                arch_version_str, [HI3796CV300ESAIC],
                'only v200hisi support no padding, '
                'now: {}'.format(arch_version_str))
        else:
            if dst.dtype == 'float16':
                TikCheckUtil.check_type_match(
                    sfr_cpadding, (float, Scalar, Expr),
                    "sfr_cpadding type error, "
                    "input: {}".format(type(sfr_cpadding)))
                check_scalar_dtype_float(sfr_cpadding,
                                         "sfr_cpadding should be"
                                         " a scalar of int/uint")
            elif dst.dtype == 'int8':
                TikCheckUtil.check_type_match(
                    sfr_cpadding, (int, Scalar, Expr),
                    "sfr_cpadding type error, "
                    "input: {}".format(type(sfr_cpadding)))
                check_scalar_dtype(sfr_cpadding,
                                   "sfr_cpadding should "
                                   "be a scalar of int/uint")
                imm_sfr_cpadding = Expr(sfr_cpadding).eval_value()
                if imm_sfr_cpadding is not None:
                    TikCheckUtil.check_in_range(
                        imm_sfr_cpadding,
                        range(INT8_MIN, INT8_MAX + 1),
                        "sfr_cpadding out range, "
                        "input: {}".format(imm_sfr_cpadding))
            else:
                TikCheckUtil.check_type_match(
                    sfr_cpadding, (int, Scalar, Expr),
                    "sfr_cpadding type error, "
                    "input: {}".format(type(sfr_cpadding)))
                check_scalar_dtype(sfr_cpadding,
                                   "sfr_cpadding should "
                                   "be a scalar of int/uint")
                imm_sfr_cpadding = Expr(sfr_cpadding).eval_value()
                if imm_sfr_cpadding is not None:
                    TikCheckUtil.check_in_range(
                        imm_sfr_cpadding,
                        range(UINT_MIN, UINT8_MAX + 1),
                        "sfr_cpadding out range, "
                        "input: {}".format(imm_sfr_cpadding))


def _aipp_top_area_pad_check(arch_version, imm_padding_mode, imm_top_pad_size):
    """check top area padding"""
    if arch_version in [ASCEND_310AIC, ASCEND_910AIC, AIC]:
        if imm_padding_mode is not None:
            if imm_padding_mode == 0:
                TikCheckUtil.check_equality(
                    imm_top_pad_size, 0,
                    arch_version + ' do not support top_pad')


def _aipp_botton_area_pad_check(arch_version, imm_padding_mode,
                                imm_botton_pad_size):
    """check botton area padding"""
    if arch_version in [ASCEND_310AIC, ASCEND_910AIC, AIC]:
        if imm_padding_mode is not None:
            if imm_padding_mode == 0:
                TikCheckUtil.check_equality(
                    imm_botton_pad_size, 0,
                    arch_version + ' do not support botton_pad')


def _aipp_check_config_pad_value(imm_padding_mode, input_format, dst,
                                 filling_hblank):
    """check padding config value"""
    # check config value padding
    if imm_padding_mode == 0:
        imm_input_format = Expr(input_format).eval_value()
        if imm_input_format is not None:
            if imm_input_format not in [NC1HWC0DI_FP16, NC1HWC0DI_INT8, RAW24,
                                        RAW16,
                                        RAW12, RAW10]:
                if dst.dtype == 'float16':
                    _check_list_type_and_range(filling_hblank, 4,
                                               (float, Scalar, Expr), None,
                                               'filling_hblank')
                elif dst.dtype == 'int8':
                    _check_list_type_and_range(filling_hblank, 4,
                                               (int, Scalar, Expr),
                                               range(INT8_MIN,
                                                     INT8_MAX + 1),
                                               'filling_hblank')
                else:
                    _check_list_type_and_range(filling_hblank, 4,
                                               (int, Scalar, Expr),
                                               range(UINT_MIN,
                                                     UINT8_MAX + 1),
                                               'filling_hblank')


def _aipp_check_area_pad(input_format, dst,  # pylint: disable=R0913
                         padding_mode, top_pad_size, botton_pad_size,
                         left_pad_size, right_pad_size, filling_hblank,
                         arch_version='v200hisi'):
    """check area padding"""
    # function's input params is too much, so disable them

    TikCheckUtil.check_type_match(
        padding_mode, (int, Scalar, Expr),
        'padding_mode type error, input: {}'.format(type(padding_mode)))
    check_scalar_dtype(padding_mode,
                       "padding_mode should be a scalar of int/uint")
    imm_padding_mode = Expr(padding_mode).eval_value()
    if imm_padding_mode is not None:
        TikCheckUtil.check_in_range(imm_padding_mode, range(4),
                                    'padding_mode out of range, '
                                    'input: {}'.format(imm_padding_mode))

    TikCheckUtil.check_type_match(
        top_pad_size, (int, Scalar, Expr),
        "top_pad_size should be int, Scalar, Expr, "
        "input: {}".format(type(top_pad_size)))
    check_scalar_dtype(top_pad_size,
                       "top_pad_size should be a scalar of int/uint")
    imm_top_pad_size = Expr(top_pad_size).eval_value()
    if imm_top_pad_size is not None:
        TikCheckUtil.check_in_range(
            imm_top_pad_size, range(33),
            'top_pad_size out of range, input: {}'.format(imm_top_pad_size))
        _aipp_top_area_pad_check(arch_version, imm_padding_mode,
                                 imm_top_pad_size)

    TikCheckUtil.check_type_match(
        botton_pad_size, (int, Scalar, Expr),
        "botton_pad_size should be int, Scalar, Expr, "
        "input: {}".format(type(botton_pad_size)))
    check_scalar_dtype(botton_pad_size,
                       "botton_pad_size should be a scalar of int/uint")
    imm_botton_pad_size = Expr(botton_pad_size).eval_value()
    if imm_botton_pad_size is not None:
        TikCheckUtil.check_in_range(
            imm_botton_pad_size, range(33),
            'botton_pad_size should in [0, 32]')
        _aipp_botton_area_pad_check(arch_version, imm_padding_mode,
                                    imm_botton_pad_size)

    if imm_padding_mode is not None:
        TikCheckUtil.check_type_match(
            left_pad_size, (int, Scalar, Expr),
            "left_pad_size should be int, Scalar, Expr, "
            "input: {}".format(type(left_pad_size)))
        check_scalar_dtype(left_pad_size,
                           "left_pad_size should be a scalar of int/uint")
        imm_left_pad_size = Expr(left_pad_size).eval_value()
        if imm_left_pad_size is not None:
            TikCheckUtil.check_in_range(
                imm_left_pad_size, range(33),
                'left_pad_size out of range, '
                'input: {}'.format(imm_left_pad_size))
        TikCheckUtil.check_type_match(
            right_pad_size, (int, Scalar, Expr),
            "right_pad_size should be int, Scalar, Expr, "
            "input: {}".format(type(right_pad_size)))
        check_scalar_dtype(right_pad_size,
                           "right_pad_size should be a scalar of int/uint")
        imm_right_pad_size = Expr(right_pad_size).eval_value()
        if imm_right_pad_size is not None:
            TikCheckUtil.check_in_range(
                imm_right_pad_size, range(33),
                'right_pad_size out of range, '
                'input: {}'.format(imm_right_pad_size))

        # check config value padding
        _aipp_check_config_pad_value(imm_padding_mode, input_format, dst,
                                     filling_hblank)


def _aipp_check_raw_info(raw_image_channel, raw_start_channel):
    """check raw info"""

    TikCheckUtil.check_type_match(
        raw_image_channel, (int, Scalar, Expr),
        "raw_image_channel should be (int, Scalar, Expr), "
        "input: {}".format(type(raw_image_channel)))
    check_scalar_dtype(raw_image_channel,
                       "raw_image_channel should be a scalar of int/uint")
    imm_raw_image_channel = Expr(raw_image_channel).eval_value()
    if imm_raw_image_channel is not None:
        TikCheckUtil.check_in_range(
            imm_raw_image_channel, range(4),
            'raw_image_channel value out range, '
            'input: {}'.format(imm_raw_image_channel))

    TikCheckUtil.check_type_match(
        raw_start_channel, (int, Scalar, Expr),
        "raw_start_channel should be (int, Scalar, Expr), "
        "input: {}".format(type(raw_start_channel)))
    check_scalar_dtype(raw_start_channel,
                       "raw_start_channel should be a scalar of int/uint")
    imm_raw_start_channel = Expr(raw_start_channel).eval_value()
    if imm_raw_start_channel is not None:
        TikCheckUtil.check_in_range(
            imm_raw_start_channel, range(4),
            'raw_start_channel value out range, '
            'input: {}'.format(imm_raw_start_channel))


def _aipp_check_stretch(dst_stride_pixel):
    """check stretch"""

    TikCheckUtil.check_type_match(
        dst_stride_pixel, (int, Scalar, Expr),
        "dst_stride_pixel should be int, Scalar, Expr, "
        "input: {}".format(type(dst_stride_pixel)))
    check_scalar_dtype(dst_stride_pixel,
                       "dst_stride_pixel should be a scalar of int/uint")
    imm_dst_stride_pixel = Expr(dst_stride_pixel).eval_value()
    if imm_dst_stride_pixel is not None:
        TikCheckUtil.check_in_range(
            imm_dst_stride_pixel, range(65536),
            'dst_stride_pixel out of range, '
            'input: {}'.format(imm_dst_stride_pixel))


def _aipp_check_sid(arch_version, sid):
    """check sid"""

    TikCheckUtil.check_type_match(
        sid, (int, Scalar, Expr), "sid should be int, Scalar, Expr, "
                                  "input: {}".format(type(sid)))
    check_scalar_dtype(sid,
                       "sid should be a scalar of int/uint")
    if arch_version in [HI3796CV300ESAIC]:
        imm_sid = Expr(sid).eval_value()
        if imm_sid is not None:
            TikCheckUtil.check_in_range(
                imm_sid, range(11),
                'sid value out range, input: {}'.format(imm_sid))


def _check_vscatter_vgather_operator_scope(src, dst, offset, offset_name):
    """check scope for vscatter and vgather

    Parameters
    ----------
    src: src operator
    dst: dst operator
    offset: addr offset tensor
    offset_name: offset tensor name

    Returns
    -------
    None
    """
    TikCheckUtil.check_equality(src.scope,
                                cce_params.scope_ubuf,
                                "src's scope must be UB. "
                                "input scope: {}".format(src.scope))
    TikCheckUtil.check_equality(dst.scope,
                                cce_params.scope_ubuf,
                                "dst's scope must be UB. "
                                "input scope: {}".format(dst.scope))
    TikCheckUtil.check_equality(offset.scope,
                                cce_params.scope_ubuf,
                                "{}'s scope must be UB. "
                                "input scope: {}".format(offset_name,
                                                         offset.scope))


class TikDataOpApi(TikIRBuilder):  # pylint: disable=R0904
    """
    Data convert, Data fill, Data move Api
    """
    def __init__(self):
        super(TikDataOpApi, self).__init__()

    @source_info_decorator()
    @debug.vtranspose_decorator
    def vtranspose(self, dst, src):
        """Transpose a continuous 16*16 two-dimensional matrix data block

        Parameters
        ----------
        src : destination operator
        dst : destination operator

        Returns
        -------
        None
        """
        TikCheckUtil.check_type_match(dst, Tensor, "dst should be tensor")
        TikCheckUtil.check_type_match(src, Tensor, "src should be tensor")
        src_elements_count = reduce_mul(src.indice.origin_shape)
        dst_elements_count = reduce_mul(dst.indice.origin_shape)
        required_elements_count = 256
        TikCheckUtil.check_ge(
            src_elements_count, required_elements_count,
            "elements of src should be more than 256")
        TikCheckUtil.check_ge(
            dst_elements_count, required_elements_count,
            "elements of dst should be more than 256")
        # all arch-version support
        dst_src_map = ["u16u16", "s16s16", "f16f16"]

        # check dtype
        dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src.dtype]
        if dtype_str not in dst_src_map:
            TikCheckUtil.raise_error(
                "dtype of dst and src should be u16u16, s16s16 or f16f16")

        # check address overlapping
        src_offset = Expr(src.offset).eval_value()
        dst_offset = Expr(dst.offset).eval_value()
        if all(isinstance(value, int) for value in (src_offset, dst_offset)):
            if src.buffer == dst.buffer:
                if src_offset == dst_offset or \
                        src_offset + required_elements_count <= \
                        dst_offset or dst_offset + \
                        required_elements_count <= src_offset:
                    pass
                else:
                    TikCheckUtil.raise_error(
                        "vtranspose not support partially address overlapping")
        # gen
        with self.new_scope():
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            # 2 is size of b16, 2 Bytes
            extent = Expr(required_elements_count*2)
            # one ir is call_extern
            self.emit(
                tvm.call_extern(
                    dst.dtype, "vtranspose",
                    dst.reinterpret_cast_to("uint16").access_ptr(
                        "w", extent=extent.get()),
                    src.reinterpret_cast_to("uint16").access_ptr(
                        "r", extent=extent.get())),
                ONE_IR)

    # VA mode - vnchwconv
    @source_info_decorator()
    @debug.vnchwconv_decorator
    def vnchwconv(self, dst_high_half, src_high_half,  # pylint: disable=R0913
                  dst_list, src_list, repeat_times, dst_rep_stride,
                  src_rep_stride, name=None):
        """used for NCHW to NHWC

        Parameters
        ----------
        dst_high_half : bool
        src_high_half : bool
        src_list : the src operation list
        dst_list: the des operation list
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations

        Returns
        -------
        Nones
        """
        if name is None:
            name = "vnchwconv"
        # check dst_high_half, src_high_half
        TikCheckUtil.check_type_match(
            dst_high_half, bool, "dst_high_half should be bool, input type: {}"
            .format(type(dst_high_half)))
        TikCheckUtil.check_type_match(
            src_high_half, bool, "src_high_half should be bool, input type: {}"
            .format(type(src_high_half)))
        check_repeat_times(repeat_times)
        # check strides
        check_vector_stride(None, [dst_rep_stride, src_rep_stride],
                            None, MAX_REP_STRIDE_DOUBLE_BYTE, ["dst", "src"])
        # check tensor list number
        TikCheckUtil.check_type_match(dst_list, (tuple, list),
                                      "dst_list should be tuple or list")
        TikCheckUtil.check_type_match(src_list, (tuple, list),
                                      "src_list should be tuple or list")
        TikCheckUtil.check_equality(len(dst_list), VNCHWCONV_LIST_LEN,
                                    "there should be 16 addresses in dst_list")
        TikCheckUtil.check_equality(len(src_list), VNCHWCONV_LIST_LEN,
                                    "there should be 16 addresses in src_list")
        for src in src_list:
            TikCheckUtil.check_type_match(src, Tensor, "src should be tensor")
        for dst in dst_list:
            TikCheckUtil.check_type_match(dst, Tensor, "dst should be tensor")

        # check tensor list dtype
        dtype_str = self._get_dtype_str(src_list, dst_list, name)

        # check address overlap
        if VNCHWCONV_INSTR_APPENDIX_MAP[dtype_str] == "b8":
            mask_len = MASK_VALUE_128
        else:
            mask_len = ONE_BLK_SIZE*VNCHWCONV_LIST_LEN // \
                       DTYPE_SIZE[dst_list[VA0_INDEX].dtype]
        if all(isinstance(value, int) for value
               in (dst_rep_stride, src_rep_stride)):
            check_scatter_address_overlap(
                mask_len, dst_list, src_list, repeat_times,
                dst_rep_stride, src_rep_stride,
                store_high_half=dst_high_half,
                src_store_high_half=src_high_half,
                name=name, msg="dst_list and src_list")

        # check tensor overflow(static)
        check_vnchwconv_overflow(
            [src_list, dst_list], ["src_list", "dst_list"], repeat_times,
            [src_rep_stride, dst_rep_stride],
            [src_high_half, dst_high_half],
            VNCHWCONV_INSTR_APPENDIX_MAP[dtype_str])
        # code gen
        config = [_dtype_convert(repeat_times, "int64"), dst_rep_stride,
                  src_rep_stride]
        if VNCHWCONV_INSTR_APPENDIX_MAP[dtype_str] == "b8":
            config.append(int(dst_high_half))
            config.append(int(src_high_half))
        self._config_vas([dst_list, src_list], dtype_str, config,
                         # [dst_extent, src_extent]
                         [Expr(((repeat_times - 1)*dst_rep_stride + 1)
                               *ONE_BLK_SIZE).get(),
                          Expr(((repeat_times - 1)*src_rep_stride + 1)
                               *ONE_BLK_SIZE).get()])

    def _get_dtype_str(self, src_list, dst_list, name):
        dtype_str = ""
        for dst, src in zip(dst_list, src_list):
            dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src.dtype]
            TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                        "Intrinsic {}'s src's dtype "
                                        "should be equal to dst's dtype".
                                        format(name))
            TikCheckUtil.check_equality(api_check_support("tik."
                                                          + name,
                                                          dst.dtype), True,
                                        INSTR_DTYPE_SUPPORT_STATEMENT.
                                        format(dst.dtype, name))
        return dtype_str

    def _config_vas(self, dst_list_src_list, dtype_str, config, extents=None):
        # can't find the function in library, so disable it
        # config VAs
        with self.new_scope():
            intrin = tvm.call_extern("uint64", "scatter_vnchwconv_" +
                                     VNCHWCONV_INSTR_APPENDIX_MAP[dtype_str],
                                     VA_REG[VA0_INDEX], VA_REG[VA2_INDEX],
                                     *type_convert(config))
            addr_list_tuple = _get_addr_list(dst_list_src_list[0],  # dst_list
                                             dst_list_src_list[1],  # src_list
                                             extents)
            intrin_block = tvm.make.Evaluate(0)
            self.source_info.set_node_loc(intrin_block)
            total_ir_num = ONE_IR
            for index, addr_list in enumerate(addr_list_tuple):
                intrin_setva = tvm.call_extern("uint64", "VA_reg_set",
                                               VA_REG[index], *addr_list)
                tmp_instr = tvm.make.Evaluate(intrin_setva)
                self.source_info.set_node_loc(tmp_instr)
                intrin_block = tvm.make.Block(intrin_block, tmp_instr)
                self.source_info.set_node_loc(intrin_block)
                total_ir_num += ONE_IR

            tmp_instr = tvm.make.Evaluate(intrin)
            self.source_info.set_node_loc(tmp_instr)
            intrin_block = tvm.make.Block(intrin_block, tmp_instr)
            self.source_info.set_node_loc(intrin_block)
            emit_scatter_instr(self, total_ir_num, intrin_block)

    @source_info_decorator()
    @debug.load2dv1_decorator
    def load2dv1(self,  # pylint: disable=R0913
                 dst,
                 src,
                 index,
                 repeat_times,
                 src_stride,
                 sid,
                 if_transpose=False,
                 addr_mode=None):
        """Pass the offline processed convolution right
        matrix (davinci format) from gm to ca/cb/cbuf or from cbuf to ca/cb

        Parameters
        ----------
        dst : destination tensor
        src : source tensor
        index : [0, 65535] data index
        repeat_times : [1, 255]
        sid: default 0
        src_stride : offset of src tensor between adjacent data segment
        if_transpose : if transport. True/False

        Returns
        -------
        None
        """
        return self.load2d(dst, src, index, repeat_times, None, src_stride,
                           sid, if_transpose, addr_mode)

    @source_info_decorator()
    @debug.load2dv2_decorator
    def load2dv2(self, dst, src, start_index,  # pylint: disable=R0913
                 repeat_times, dst_gap, src_stride,
                 sid, if_transpose=False, addr_mode=None):
        """Pass the offline processed convolution right
        matrix (davinci format) to scope_ca/scope_cb

        Parameters
        ----------
        dst : destination tensor
        src : source tensor
        start_index : [0, 65535] data index
        repeat_times : [1, 255]
        dst_gap: gap of dst tensor between adjacent data segment
        src_stride : stride of src tensor between adjacent data segment
        sid: default 0
        if_transpose : if transport. True/False
        addr_mode: address mode, default is None

        Returns
        -------
        None
        """
        # too many arguments, so disable R0913
        return self.load2d(dst, src, start_index, repeat_times, dst_gap,
                           src_stride, sid, if_transpose, addr_mode)

    def load2d(self, dst, src, start_index,  # pylint: disable=R0913, R0914
               repeat_times, dst_gap, src_stride, sid,
               en_transpose=False, addr_mode=None):
        """Pass the offline processed convolution right
        matrix (davinci format) to scope_ca/scope_cb
        Note: dst_gap is tail-to-head, src_stride is head-to-head.

        Parameters
        ----------
        dst : destination tensor
        src : source tensor
        start_index : [0, 65535] data index
        repeat_times : [1, 255]
        dst_gap: gap of dst tensor between adjacent data segment
        src_stride : stride of src tensor between adjacent data segment
        sid: default 0
        en_transpose : enable transport. True/False
        addr_mode

        Returns
        -------
        None
        """
        # too many arguments, so disable R0914
        # check instruction
        arch_version_str = get_soc_name() + get_soc_core_type()
        # check scope
        src_scope = SCOPE_MAP[src.scope]
        dst_scope = SCOPE_MAP[dst.scope]
        TikCheckUtil.check_in_range(
            (src_scope, dst_scope), LOAD2D_DMA_LIST,
            "load2d not support from %s to %s" % (src_scope, dst_scope))

        pipe_line, intrin_name = LOAD2D_DMA_LIST[(src_scope, dst_scope)]
        # check dtype
        dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src.dtype]
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "Intrinsic {}'s src's dtype should"
                                    " be equal to dst's dtype".
                                    format("load2d"))
        TikCheckUtil.check_equality(api_check_support("tik." +
                                                      "load2dv1",
                                                      dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, "load2d"))
        # check addr_mode
        # not support online config yet
        TikCheckUtil.check_in_range(
            addr_mode, ('inc', INC_MODE, 'dec', DEC_MODE, None),
            "addr_mode should be 'inc', 'dec', 0, 1 or None")

        if addr_mode in ("dec", DEC_MODE):
            TikCheckUtil.check_equality((get_soc_name() in
                                         (ASCEND_910, HI3796CV300ES)
                                         or get_soc_name() +
                                         get_soc_core_type() == AIC), True,
                                        "current soc not support "
                                        "addr_dec_mode")
            # 1 increase
            addr_mode_bit = _ADDR_MODE_BIT_INCREASE
        else:
            # 0 decrease
            addr_mode_bit = _ADDR_MODE_BIT_DECREASE
        # check en_transpose
        # not support online config yet
        TikCheckUtil.check_type_match(en_transpose, bool,
                                      "en_transpose should be bool")

        if en_transpose:
            TikCheckUtil.check_in_range(
                src_scope, ('cbuf', ),
                "src_scope should be cbuf if enabling transpose")
            TikCheckUtil.check_in_range(
                dst_scope, ('ca', 'cb'),
                "dst_scope should be ca or cb if enabling transpose")
            # 1 enable transpose
            transpose_bit = 1
        else:
            # 0 disable transpose
            transpose_bit = 0
        # check repeat_times
        check_repeat_times(repeat_times)
        # gen start_index
        TikCheckUtil.check_type_match(
            start_index, (int, Scalar, Expr),
            "start_index should be int, Scalar or Expr")
        check_scalar_dtype(start_index,
                           "scalar_start_index should be a scalar of int/uint")
        check_integer_in_range(
            start_index, range(MAX_START_INDEX),
            "start_index should be in the range of [0, 65535], input value is: "
            "{}".format(start_index))
        # check dst_gap
        if dst_gap is not None:
            TikCheckUtil.check_equality((get_soc_name() == HI3796CV300ES
                                         or get_soc_name() +
                                         get_soc_core_type() == AIC), True,
                                        "current soc not support dst_gap")
            TikCheckUtil.check_type_match(dst_gap, (int, Scalar, Expr),
                                          "dst_gap should be int, Scalar, Expr")
            check_scalar_dtype(dst_gap,
                               "scalar_dst_gap should be a scalar of int/uint")
            check_integer_in_range(
                dst_gap, range(MAX_DST_GAP_DOUBLE_BYTE),
                "dst_gap should be in the range of [0, 65535], input value is: "
                "{}".format(dst_gap))
        # check src_stride
        TikCheckUtil.check_type_match(
            src_stride, (int, Scalar, Expr),
            "src_stride should be int, Scalar or Expr")
        check_scalar_dtype(src_stride,
                           "scalar_src_stride should be a scalar of int/uint")
        check_integer_in_range(
            src_stride, range(MAX_BLK_STRIDE_DOUBLE_BYTE),
            "src_stride should be in the range of [0, 65535], input value is: "
            "{}".format(src_stride))
        # check sid
        check_integer_in_range(sid, range(MAX_SID),
                               "sid should be in the range of [0, 15]")
        # gen
        if dst_gap is None:
            config = [start_index, repeat_times, src_stride, sid]
        else:
            config = [start_index, repeat_times, src_stride, dst_gap, sid]
        args = config

        dtype_str = _get_load2d_dtype_str(src_scope, args, transpose_bit,
                                          arch_version_str, addr_mode_bit,
                                          dtype_str, dst)
        # calculate extent
        src_extent, dst_extent = _calculate_extent_load2d(
            start_index, repeat_times, src_stride, dst_gap)

        with self.new_scope():
            instr = tvm.call_extern(dst.dtype,
                                    intrin_name,
                                    dst.reinterpret_cast_to(dtype_str)
                                    .access_ptr("w", extent=dst_extent),
                                    src.reinterpret_cast_to(dtype_str)
                                    .access_ptr("r", extent=src_extent),
                                    *(type_convert(args)))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", pipe_line)
            # one ir is call_extern
            self.emit(instr, ONE_IR)

    def assign(self, dst, src, dst_offset=0, src_offset=None):
        """assign src to dst

        Parameters
        ----------
        dst : destination tensor
        src : source tensor
        dst_offset: dst tensor offset
        src_offset: src tensor offset

        Returns
        -------
        None
        """
        type_list = (Tensor, Scalar, Expr)
        TikCheckUtil.check_type_match(dst, type_list,
                                      "assign only support load or "
                                      "store data between UB and REG")
        TikCheckUtil.check_type_match(src, type_list,
                                      "assign only support load or "
                                      "store data between UB and REG")
        with self.new_scope():
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_S)
            if isinstance(dst, (Scalar, Expr)):
                dst_side = tvm.call_extern(dst.dtype, "reg", dst.get())
            else:
                if is_immediate_number(dst_offset):
                    dst_side = dst.access_ptr(
                        "w", extent=Expr(DTYPE_SIZE[dst.dtype]).get(),
                        offset=dst_offset)
                else:
                    dst_side = dst.access_ptr(
                        "w", extent=Expr(DTYPE_SIZE[dst.dtype]).get(),
                        offset=Expr(dst_offset).get())
            if isinstance(src, (Scalar, Expr)):
                src_side = tvm.call_extern(src.dtype, "reg", src.get())
            else:
                if src_offset is None:
                    src_side = src.access_ptr(
                        "r", extent=Expr(DTYPE_SIZE[src.dtype]).get())
                elif is_immediate_number(src_offset):
                    src_side = src.access_ptr(
                        "r", extent=Expr(DTYPE_SIZE[src.dtype]).get(),
                        offset=src_offset)
                else:
                    src_side = src.access_ptr(
                        "r", extent=Expr(DTYPE_SIZE[src.dtype]).get(),
                        offset=Expr(src_offset).get())

            # one ir is reg_mov
            self.emit(
                tvm.call_extern(
                    dst.dtype,
                    "reg_mov",
                    dst_side,
                    src_side,
                ), ONE_IR)

    def _set_fmatrix(self, *value):
        if len(value) == HAS_PARAM_CONCAT:
            fmatrix_value = value[0]
        elif len(value) == NEED_PARAM_CONCAT:
            pad, l1_h, l1_w = value
            TikCheckUtil.check_type_match(pad, list, "pad should be list")
            TikCheckUtil.check_type_match(l1_h, int, "l1_h should be int")
            TikCheckUtil.check_type_match(l1_w, int, "l1_w should be int")
            params = [l1_w, l1_h, pad[PADDING_LEFT_IDX],
                      pad[PADDING_RIGHT_IDX], pad[PADDING_TOP_IDX],
                      pad[PADDING_BOT_IDX]]
            offset_list = FMATRIX_OFFSET_LIST
            segment_list = FMATRIX_SEGMENT_LIST
            fmatrix_value = concat_params(params, offset_list, segment_list)
        # one ir is call_extern
        self.emit(tvm.call_extern("int64", "set_fmatrix", fmatrix_value),
                  ONE_IR)

    def _set_padding(self, value, dtype):
        if not is_basic_expr(value):
            TikCheckUtil.check_type_match(
                value, (int, float),
                "set value should be float16, uint8 or int8")
        if dtype in ("uint8", "int8"):
            params = [value, value]
            offset_list = PADDING_ONE_BYTE_OFFSET_LIST
            segment_list = PADDING_ONE_BYTE_SEGMENT_LIST
        else:
            params = [value]
            offset_list = PADDING_TWO_BYTE_OFFSET_LIST
            segment_list = PADDING_TWO_BYTE_SEGMENT_LIST
        padding = concat_params(params, offset_list, segment_list)
        with self.new_scope():
            # one ir is call_extern
            self.emit(tvm.call_extern("uint64", "set_padding", padding),
                      ONE_IR)

    @source_info_decorator()
    @debug.set_l0_set_value_decorator
    def set_l0_set_value(self, value, dtype):
        """for tensor padding with matrix

        Parameters
        ----------
        value : input
        dtype : input's type

        Returns
        -------
        None
        """
        TikCheckUtil.check_type_match(value, (int, float, Scalar),
                                      "value should be int, float or Scalar")
        TikCheckUtil.check_in_range(dtype, ("float16"),
                                    "dtype only support float16")
        if isinstance(value, Scalar):
            TikCheckUtil.check_equality(value.dtype, "float16",
                                        "scalar_value should be float16")
        if not isinstance(value, Scalar):
            if dtype == "int16":
                l0_set_2d_value = np.int16(value)
            elif dtype == "uint16":
                l0_set_2d_value = np.uint16(value)
            else:
                l0_set_2d_value = np.float16(value)
            l0_set_2d_value = l0_set_2d_value.view(np.float16)
            l0_set_2d_value = float(l0_set_2d_value)
            l0_set_2d_temp = _dtype_convert(l0_set_2d_value, dtype)
        else:
            l0_set_2d_temp = _dtype_convert(value, dtype)
        with self.new_scope():
            # one ir is call_extern
            self.emit(
                tvm.call_extern("float16", "set_l0_set_value",
                                l0_set_2d_temp), ONE_IR)

    def _do_load3d_fmatrix(self, reg_fmatrix):
        if "fmatrix" in self.global_dict:  # pylint: disable=E1101
            fmatrix = self.global_dict["fmatrix"]  # pylint: disable=E1101
        else:
            fmatrix = self.global_scalar(dtype="int64")  # pylint: disable=E1101
            self.global_dict["fmatrix"] = fmatrix  # pylint: disable=E1101
        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                t_fmatrix = self.Scalar_(dtype="int64")  # pylint: disable=E1101
                t_fmatrix.set_as(reg_fmatrix)
                self.scope_attr(cce_params.CCE_AXIS, "if_protect", PIPE_MTE1)
                with self.if_scope_(fmatrix != t_fmatrix):
                    fmatrix.set_as(t_fmatrix)
                    # one ir is call_extern
                    self.emit(
                        tvm.call_extern("int64", "set_fmatrix", fmatrix.get()),
                        ONE_IR)

    @source_info_decorator()
    @debug.load3dv1_decorator
    def load3dv1(self, dst, src, pad, l1_h, l1_w,  # pylint: disable=R0913, R0914
                 c1_index, fetch_filter_w, fetch_filter_h, left_top_w,
                 left_top_h, stride_w, stride_h, filter_w, filter_h,
                 dilation_filter_w, dilation_filter_h, jump_offset, repeat_mode,
                 repeat_time, _csize=0, pad_value=None):
        """image to colomn, only support L1 to L0A/L0B/UB

        Parameters
        ----------
        dst: destination operator
        src: source operator
        pad_list: [left, right, top, bottom]
        l1_h: height of src tensor
        l1_w: width of src tensor
        c1_index: C channel position/16 for f16, C channel position/32 for b8
        fetch_filter_w: fetch position in filter w dimension
        fetch_filter_h: fetch position in filter h dimension
        left_top_w: the start left top corner coordinate of windown in feature
                    map(1st window position in w dimension)
        left_top_h: the start left top corner coordinate of windown in feature
                    map(1st window position in h dimension)
        stride_w: filter stride size in w dimension
        stride_h: filter stride size in h dimension
        filter_w: width of filter
        filter_h: height of filter
        dilation_filter_w: dilation size of filter in w dimension
        dilation_filter_h: dilation size of filter in h dimension
        jump_offset: jump offset size of destination
        repeat_mode:
        repeat_time:
        _csize:
        pad_value: value for padding, default = None

        Returns
        -------
        None
        """
        # too many arguments, so disable R0914
        dst_src_dtype_list = ["u8u8", "s8s8", "f16f16"]
        scope_map = {'l0a': 'ca', 'l0b': 'cb', 'ub': 'ub'}
        # check core_arch
        TikCheckUtil.check_not_equality(get_soc_name() +
                                        get_soc_core_type(), VEC,
                                        "current soc does't support load3dv1")
        # check tensor scope
        dst_scope = dst.scope.split(".")[-1].lower()
        src_scope = src.scope.split(".")[-1].lower()
        TikCheckUtil.check_in_range(dst_scope, ('l0a', 'l0b', 'ub'),
                                    "dst_scope %s is not supported for "
                                    "load3dv1." % dst.scope)
        TikCheckUtil.check_in_range(src_scope, ('l1',),
                                    "src_scope %s is not supported for "
                                    "load3dv1." % src.scope)
        # check tensor dtype
        dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src.dtype]
        TikCheckUtil.check_in_range(
            dtype_str, dst_src_dtype_list,
            "dtype of dst should be u8u8, s8s8, or f16f16.")
        _load3dv1_col2img_check(fetch_filter_w, fetch_filter_h, left_top_w,
                                left_top_h)
        _load3dv1_load3dv2_col2img_check(pad, l1_w, l1_h, stride_w, stride_h,
                                         filter_w, filter_h, dilation_filter_w,
                                         dilation_filter_h)

        # check index
        TikCheckUtil.check_type_match(c1_index, (int, Scalar, Expr),
                                      "c1_index should be int, Scalar or Expr")
        check_scalar_dtype(c1_index,
                           "scalar_c1_index should be a scalar of int/uint")
        check_integer_in_range(c1_index, range(MAX_C1_INDEX),
                               "c1_index should be in the range of [0, 4095], "
                               "input value is: {}".format(c1_index))

        # check jumpOffset
        TikCheckUtil.check_type_match(
            jump_offset, (int, Scalar, Expr),
            "jump_offset should be int, Scalar or Expr")
        check_scalar_dtype(jump_offset,
                           "scalar_jump_offset should be a scalar of int/uint")
        check_integer_in_range(
            jump_offset, range(MIN_JUMP_OFFSET, MAX_JUMP_OFFSET),
            "jump_offset should be in the range of [1, 127], input value is: "
            "{}".format(jump_offset))
        # check repeat_time
        check_repeat_times(repeat_time)
        # check repeatMode
        TikCheckUtil.check_type_match(
            repeat_mode, (int, Scalar, Expr),
            "repeat_mode should be int, Scalar or Expr")
        check_scalar_dtype(repeat_mode,
                           "scalar_repeat_mode should be a scalar of int/uint")
        check_integer_in_range(repeat_mode, range(MAX_REPEAT_MODE),
                               "repeat_mode should be 0 or 1, input value is: "
                               "{}".format(repeat_mode))
        # check pad_value
        TikCheckUtil.check_type_match(
            pad_value, (int, float),
            "pad_value should be python int or float, "
            "input type is: {}".format(type(pad_value)))
        # check _csize
        TikCheckUtil.check_type_match(_csize, (int, Expr),
                                      "_csize should be int or Expr, input type"
                                      " of_csize: {}".format(_csize))
        check_integer_in_range(_csize, range(MAX_C_SIZE))
        # FMATRIX
        orig_params = []
        params = [l1_w, l1_h, pad[PADDING_LEFT_IDX], pad[PADDING_RIGHT_IDX],
                  pad[PADDING_TOP_IDX], pad[PADDING_BOT_IDX]]
        reg_fmatrix = concat_params(params, FMATRIX_OFFSET_LIST,
                                    FMATRIX_SEGMENT_LIST)
        orig_params += params[:]

        self._do_load3d_fmatrix(reg_fmatrix)
        # padding
        do_load3d_padding(self, src, pad_value)
        # code gen
        params = [
            c1_index, fetch_filter_w, fetch_filter_h, left_top_w, left_top_h
        ]
        offset_list = LOAD3DV1_REG_XM_OFFSET_LIST
        segment_list = LOAD3DV1_REG_XM_SEGMENT_LIST
        reg_xm = concat_params(params, offset_list, segment_list)
        orig_params += params[:]

        params = [
            stride_w, stride_h, filter_w, filter_h, dilation_filter_w,
            dilation_filter_h, jump_offset, repeat_mode, repeat_time
        ]
        offset_list = LOAD3DV1_REG_XT_OFFSET_LIST
        segment_list = LOAD3DV1_REG_XT_SEGMENT_LIST
        reg_xt = concat_params(params, offset_list, segment_list)
        orig_params += params[:]

        # cal extent
        dst_extent = _calculate_extent_load3dv1(
            dst, repeat_mode, repeat_time, jump_offset)

        with self.new_scope():
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_MTE1)
            if get_bit_len(dst.dtype) == DST_TYPE_LEN:
                instr = tvm.call_extern(
                    dst.dtype, "img2col_cbuf_to_" + scope_map[dst_scope],
                    dst.reinterpret_cast_to("float16").access_ptr(
                        "w", extent=dst_extent),
                    src.reinterpret_cast_to("float16").access_ptr(
                        "r"), reg_xm, reg_xt, _csize)
            else:
                instr = tvm.call_extern(
                    dst.dtype, "img2col_cbuf_to_" + scope_map[dst_scope],
                    dst.access_ptr("w", extent=dst_extent),
                    src.access_ptr("r"), reg_xm, reg_xt, _csize)
            # one ir is call_extern
            self.emit(instr, ONE_IR)

    @source_info_decorator()
    @debug.col2img_decorator
    def col2img(self, dst, src, pad, l1_h,  # pylint: disable=R0913, R0914
                l1_w, fetch_filter_w, fetch_filter_h, left_top_w, left_top_h,
                stride_w, stride_h, filter_w, filter_h, dilation_filter_w,
                dilation_filter_h, repeat_time):
        """
        only support L1 to L0A/L0B/UB
        pad : [left, right, top, bottom]
        no Csize<=4, C0=16
        """
        # subclass has the member but parent class call it, so disable E1101
        # too many arguments, so disable R0914
        # check tensor scope
        # last_string
        dst_scope = dst.scope.split(".")[-1].lower()
        src_scope = src.scope.split(".")[-1].lower()
        TikCheckUtil.check_equality(dst_scope, "ub", "dst scope should be ub.")
        TikCheckUtil.check_equality(src_scope, "ub", "src scope should be ub.")
        # check tensor dtype
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "Intrinsic {}'s src's dtype should"
                                    " be equal to dst's dtype".
                                    format("col2img"))
        TikCheckUtil.check_equality(api_check_support("tik." + "col2img",
                                                      dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, "col2img"))
        _load3dv1_col2img_check(fetch_filter_w, fetch_filter_h, left_top_w,
                                left_top_h)
        _load3dv1_load3dv2_col2img_check(pad, l1_w, l1_h, stride_w, stride_h,
                                         filter_w, filter_h, dilation_filter_w,
                                         dilation_filter_h)
        # check repeatMode
        repeat_mode = 1
        orig_params = []
        # gen
        params = [l1_w, l1_h, pad[PADDING_LEFT_IDX], pad[PADDING_RIGHT_IDX],
                  pad[PADDING_TOP_IDX], pad[PADDING_BOT_IDX]]
        offset_list = REG_FCOL2IMG_OFFSET_LIST
        segment_list = REG_FCOL2IMG_SEGMENT_LIST
        reg_fcol2_img = concat_params(params, offset_list, segment_list)
        orig_params += params[:]

        params = [fetch_filter_w, fetch_filter_h, left_top_w, left_top_h]
        offset_list = COL2IMG_REG_XM_OFFSET_LIST
        segment_list = COL2IMG_REG_XM_SEGMENT_LIST
        reg_xm = concat_params(params, offset_list, segment_list)
        orig_params += params[:]

        params = [
            stride_w, stride_h, filter_w, filter_h, dilation_filter_w,
            dilation_filter_h, repeat_mode, repeat_time
        ]
        offset_list = COL2IMG_REG_XT_OFFSET_LIST
        segment_list = COL2IMG_REG_XT_SEGMENT_LIST
        reg_xt = concat_params(params, offset_list, segment_list)
        orig_params += params[:]

        with self.context.freeze():  # pylint: disable=E1101
            if "fcol2img" in self.global_dict:  # pylint: disable=E1101
                fcol2img = self.global_dict["fcol2img"]  # pylint: disable=E1101
            else:
                fcol2img = self.global_scalar(dtype="int64")  # pylint: disable=E1101
                self.global_dict["fcol2img"] = fcol2img  # pylint: disable=E1101
            TikCheckUtil.check_type_match(fcol2img, Scalar,
                                          "fcol2img should be Scalar")
            with self.new_scope():
                temp_scalar = self.Scalar_(dtype="int64")  # pylint: disable=E1101
                temp_scalar.set_as(reg_fcol2_img)
                self.scope_attr(cce_params.CCE_AXIS, "if_protect", PIPE_MTE1)
                with self.if_scope_(fcol2img != temp_scalar):
                    fcol2img.set_as(temp_scalar)
                    # one ir is call_extern
                    self.emit(
                        tvm.call_extern("int64", "set_fcol2img",
                                        fcol2img.get()), ONE_IR)
            with self.new_scope():
                self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
                instr = tvm.call_extern(dst.dtype, "col2img",
                                        dst.access_ptr("w"),
                                        src.access_ptr("r"),
                                        reg_xm, reg_xt)
                # one ir is call_extern
                self.emit(instr, ONE_IR)

    @source_info_decorator()
    @debug.broadcast_ub_to_l0c_decorator
    def broadcast_ub_to_l0c(self, dst, src, nburst, burst_len, *strides):
        """copy the data from tik.ubuf to tik.cc's tensor

        Parameters
        ----------
        dst : destination operator
        src : source operation
        nburst : [1, 255] continuous data segment for transfer instruction
        burst_len: nburst's length [1, 255]
        *strides: [src_gap, dst_gap]

        Returns
        -------
        None
        """
        # check nburst
        TikCheckUtil.check_type_match(nburst, (int, Scalar, Expr),
                                      "nburst should be int, Scalar or Expr")
        check_scalar_dtype(nburst, "scalar_nburst should be a scalar of int")
        check_integer_in_range(
            nburst, range(MIN_NBURST, MAX_NBURST_SINGLE_BYTE),
            "nburst should be in the range of [1, 255], input value is {}"
            .format(nburst))
        # check burst_len
        TikCheckUtil.check_type_match(burst_len, (int, Scalar, Expr),
                                      "burst_len should be int, Scalar or Expr")
        check_scalar_dtype(burst_len,
                           "scalar_burst_len should be a scalar of int")
        check_integer_in_range(
            burst_len, range(MIN_BURST_LEN, MAX_BURST_LEN_SINGLE_BYTE),
            "burst_len should be in the range of [1, 255], input value is {}"
            .format(burst_len))
        # check stride
        TikCheckUtil.check_type_match(
            strides[SRC_BLK_STRIDE_IDX], (int, Scalar, Expr),
            "src_blk_stride should be int, Scalar or Expr")
        TikCheckUtil.check_type_match(
            strides[DST_BLK_STRIDE_IDX], (int, Scalar, Expr),
            "dst_blk_stride should be int, Scalar or Expr")
        check_scalar_dtype(strides[SRC_BLK_STRIDE_IDX],
                           "scalar_src_blk_stride should be a scalar of int")
        check_scalar_dtype(strides[DST_BLK_STRIDE_IDX],
                           "scalar_dst_blk_stride should be a scalar of int")
        TikCheckUtil.check_equality(len(strides), STRIDES_LEN,
                                    "length of strides should be 2")

        if is_immediate_number(strides):
            check_integer_in_range(
                strides[DST_BLK_STRIDE_IDX], range(MAX_BLK_STRIDE_SINGLE_BYTE),
                "dst_blk_stride should be in the range of [0, 255], input value"
                " is {}".format(strides[DST_BLK_STRIDE_IDX]))
            check_integer_in_range(
                strides[SRC_BLK_STRIDE_IDX], range(MAX_BLK_STRIDE_SINGLE_BYTE),
                "src_blk_stride should be in the range of [0, 255], input value"
                " is {}".format(strides[SRC_BLK_STRIDE_IDX]))
        # check tensor dtype
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "Intrinsic {}'s src's dtype should be "
                                    "equal to dst's dtype".
                                    format("broadcast_ub_to_l0c"))
        TikCheckUtil.check_equality(
            intrinsic_check_support("Intrinsic_" + "broadcast_ub_to_cc",
                                    dst.dtype), True,
            INSTR_DTYPE_SUPPORT_STATEMENT.format(dst.dtype,
                                                 "broadcast_ub_to_l0c"))
        # check tensor overflow
        _check_src_overflow_brc(src, nburst, burst_len,
                                strides[SRC_BLK_STRIDE_IDX])
        _check_dst_overflow_brc(dst, nburst, burst_len,
                                strides[DST_BLK_STRIDE_IDX])
        # gen
        params = [nburst, burst_len, strides[SRC_BLK_STRIDE_IDX],
                  strides[DST_BLK_STRIDE_IDX]]
        # src burst_len: 16 element
        # src gap: 32 Byte
        # dst burst_len: 256 element
        # dst gap: 256 element
        # function _calculate_extent_broadcast_ub_to_l0c returns a list, idx 0
        # represents dst_extent and 1 represents src_extent
        extents = _calculate_extent_broadcast_ub_to_l0c(
            dst, src, nburst, burst_len, [strides[DST_BLK_STRIDE_IDX],
                                          strides[SRC_BLK_STRIDE_IDX]])
        self._gen_brc_code(params, dst, src, extents)

    def _gen_brc_code(self, params, dst, src, extents):
        """generate IR for broadcast_ub_to_l0c

        Parameters
        ----------
        params: params list
        dst : destination operator
        src : source operation
        extents: dst_extent, src_extent

        Returns
        -------
        None
        """
        with self.new_scope():
            instr = tvm.call_extern(
                dst.dtype, "broadcast_ub_to_cc",
                dst.reinterpret_cast_to(dst.dtype).access_ptr(
                    "w", extent=extents[0]),
                src.reinterpret_cast_to(src.dtype).access_ptr(
                    "r", extent=extents[1]),
                *type_convert(params))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            # one ir is call_extern
            self.emit(instr, ONE_IR)

    def _gen_mmad_broadcast_code(self, params, scope_map, src, dst):
        new_scope_map = deepcopy(scope_map)
        new_scope_map[scope_cbuf] = "cbuf"
        args = type_convert(params)
        with self.new_scope():
            instr = tvm.call_extern(
                dst.dtype, "broadcast_" + new_scope_map[src.scope] +
                "_to_" + new_scope_map[dst.scope], dst.access_ptr("w"),
                src.access_ptr("r"), *args)
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            # one ir is call_extern
            self.emit(instr, ONE_IR)

    @source_info_decorator()
    @debug.mmad_brc_decorator
    def mmad_broadcast(self, dst, src, repeat_mode,  # pylint: disable=R0913
                       nburst, burst_repeat, dst_gap, src_gap):
        """
        Note: busrt on src side is given in term of 1*16 elements,
              busrt on dst side is given in term of 16*16 elements,
              gap on src side is given in term of 32B,
              gap on dst side is given in term of 16*16 elements.
        repeatMode=0: burst is effective, each 1*16 is broadcast
                      to 16*16 fractal(repeat on N-dim)
        repeatMode=1: burst is restricted to 1, each busrt(1*16) is
              broadcast to repeat*(16*16) fractals(repeat on M-dim);
        """
        arch_version_dst_src_scope_map = {
            ASCEND_310AIC: ['l0cub'],
            ASCEND_910AIC: ['l0cub'],
            HI3796CV300ESAIC: ['l0cub', 'l0cl1'],
            HI3796CV300CSAIC: ['l0cub', 'l0cl1'],
            AIC: ['l0cub', 'l0cl1']
        }
        new_scope_map = deepcopy(SCOPE_MAP)
        new_scope_map[scope_ubuf] = "ub"
        # check scope
        dst_scope = dst.scope.split('.')[-1].lower()
        src_scope = src.scope.split('.')[-1].lower()
        TikCheckUtil.check_in_range(
            dst_scope + src_scope,
            arch_version_dst_src_scope_map[get_soc_name() +
                                           get_soc_core_type()],
            "%s Instruction mmad_broadcast doesn't support "
            "broadcast %s to %s" %
            (get_soc_name() + get_soc_core_type(), src_scope, dst_scope))
        # check repeatMode
        TikCheckUtil.check_type_match(repeat_mode, (int, Scalar),
                                      "repeat_mode should be int or Scalar")
        check_integer_in_range(repeat_mode, range(MAX_REPEAT_MODE),
                               "repeat_mode should be 0 or 1")
        # check nburst
        check_integer_in_range(
            nburst, range(MIN_NBURST, MAX_NBURST_SINGLE_BYTE),
            "nburst should be in the range of [1, 255]")
        # check burst_repeat
        check_integer_in_range(
            burst_repeat, range(MIN_BURST_REPEAT, MAX_BURST_REPEAT),
            "burst_repeat should be in the range of [1, 255]")
        # check gap
        check_integer_in_range(dst_gap, range(MAX_DST_GAP_SINGLE_BYTE),
                               "dst_gap should be in the range of [0, 255]")
        check_integer_in_range(src_gap, range(MAX_SRC_GAP),
                               "src_gap should be in the range of [0, 255]")
        # check tensor dtype
        dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src.dtype]
        TikCheckUtil.check_equality(api_check_support("tik." +
                                                      "mmad_broadcast",
                                                      dtype_str), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dtype_str, "mmad_broadcast"))
        # code gen
        params = [nburst, burst_repeat, src_gap, dst_gap, repeat_mode]
        self._gen_mmad_broadcast_code(params, new_scope_map, src, dst)

    def _check_padding_scope_dtype(self, dst, value):
        """check tensor padding input scope and dtype"""
        if get_soc_name() in (ASCEND_610, ASCEND_620, HI3796CV300ES):
            TikCheckUtil.check_in_range(
                dst.scope, [scope_ca, scope_cb, scope_cbuf],
                "dst scope should be L0A, L0B or L1, input dst scope: %s."
                % dst.scope)
        else:
            TikCheckUtil.check_in_range(
                dst.scope, [scope_ca, scope_cb],
                "dst scope should be L0A or L0B, input dst scope: %s."
                % dst.scope)
        # check dst dtype
        if isinstance(value, Scalar):
            dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[value.dtype]
        else:
            # dtype_str: dst_dtype add dst_dtype
            dtype_str = DTYPE_MAP[dst.dtype]*2
        TikCheckUtil.check_in_range(
            dtype_str, ["u16u16", "s16s16", "f16f16"],
            "dtype of dst and src should be u16u16, s16s16 or f16f16")

    @source_info_decorator()
    @debug.set_2d_decorator
    def tensor_padding_with_matrix(self, dst, repeat_times, value=None):
        """Move value to dst tensor

        Parameters
        ----------
        dst : destination tensor
        value : the value
        repeat_times : [1, 255] the invoke times

        Returns
        -------
        None
        """
        # subclass has the member but parent class call it, so disable E1101
        scope_map = {
            scope_ca: "l0a",
            scope_cb: "l0b",
            scope_cc: "l0c",
            scope_cbuf: "l1",
            scope_ubuf: "ub",
            scope_gm: "out"
        }
        # check repeat_times
        check_integer_in_range(
            repeat_times, range(MIN_REPEAT_TIMES, MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [1, 255], input value is "
            "{}".format(repeat_times))
        # check dst scope
        TikCheckUtil.check_not_equality(get_soc_name() + get_soc_core_type(),
                                        VEC, "%s doesn't support "
                                        "tensor_padding_with_matrix."
                                        % (get_soc_name() +
                                           get_soc_core_type()))
        self._check_padding_scope_dtype(dst, value)
        # padding value
        if value is not None:
            TikCheckUtil.check_type_match(
                value, (int, float, Scalar),
                "value should be int or float or Scalar")
            with self.context.freeze():  # pylint: disable=E1101
                if not isinstance(value, Scalar):  # immediate
                    if "l0_set_2d" in self.global_dict:  # pylint: disable=E1101
                        l0_set_2d = self.global_dict["l0_set_2d"]  # pylint: disable=E1101
                    else:
                        l0_set_2d = self.global_scalar(dtype="int16")  # pylint: disable=E1101
                        self.global_dict["l0_set_2d"] = l0_set_2d  # pylint: disable=E1101
                    t_l0_set_2d = self.Scalar_(dtype="int16")  # pylint: disable=E1101
                    if dst.dtype == "float16":
                        l0_set_2d_value = np.float16(value)
                    elif dst.dtype == "int16":
                        l0_set_2d_value = np.int16(value)
                    else:
                        l0_set_2d_value = np.uint16(value)
                    l0_set_2d_value = l0_set_2d_value.view(np.int16)
                    l0_set_2d_value = int(l0_set_2d_value)
                    t_l0_set_2d.set_as(l0_set_2d_value)
                    self.scope_attr(cce_params.CCE_AXIS, "if_protect",
                                    PIPE_MTE1)
                    with self.if_scope_(l0_set_2d != t_l0_set_2d):
                        l0_set_2d.set_as(t_l0_set_2d)
                        with self.new_scope():
                            # one ir is call_extern
                            self.emit(
                                tvm.call_extern(
                                    "float16", "set_l0_set_value",
                                    tvm.call_extern("float16",
                                                    "reinterpret_cast",
                                                    l0_set_2d.get())), ONE_IR)
                else:  # scalar
                    with self.new_scope():
                        # one ir is call_extern
                        self.emit(
                            tvm.call_extern(
                                "float16", "set_l0_set_value",
                                tvm.call_extern("float16", "reinterpret_cast",
                                                value.get())), ONE_IR)
        # code gen
        args = concat_params([repeat_times],
                             TENSOR_PADDING_OFFSET_LIST,
                             TENSOR_PADDING_SEGMENT_LIST)
        with self.new_scope():
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            instr = tvm.call_extern(dst.dtype,
                                    "set_" + scope_map[dst.scope] + "_2d",
                                    dst.access_ptr("w"), type_convert(args))
            # one ir is call_extern
            self.emit(instr, ONE_IR)

    def _gen_vector_scalar_code(self, scalar, dst, name,  # pylint: disable=R0913
                                config, mask_o, mask_mode, dst_extent):
        scalar_tmp = _dtype_convert(scalar, dst.dtype)
        with self.new_scope():
            if mask_mode == "counter":
                # save orig_ctrl
                orig_ctrl = set_ctrl_counter_mask(self)

            instr = tvm.call_extern(dst.dtype, name,
                                    dst.access_ptr("w", extent=dst_extent),
                                    scalar_tmp, *type_convert(config))
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            # one ir is call_extern, one ir is set_vector_mask
            self.emit(instr, TWO_IR)

            # reset CTRL SPR as orig_ctrl
            if mask_mode == "counter":
                reset_ctrl_counter_mask(self, orig_ctrl)

    def _check_vector_scalar_operator_and_get_dst_name(self, name, dst, scalar,
                                                       print_name):
        """check operator for vector_scalar_elewise_func and
        get special dst name for different instructions
        """
        # check instruction
        if name == "vci":
            TikCheckUtil.check_equality(
                get_soc_name() + get_soc_core_type(), VEC,
                "only {} support instruction vci".format(VEC))
            dst_name = "dst_index"
            scalar_name = "start_point"
        else:
            dst_name = "dst"
            scalar_name = "scalar"
        # check dst
        TikCheckUtil.check_type_match(dst, Tensor,
                                      "{} should be tensor, input type is"
                                      " {}".format(dst_name, type(dst)))
        TikCheckUtil.check_equality(dst.scope, "local.UB",
                                    "{}'s scope must be UB, not support "
                                    "scope: {}".format(dst_name, dst.scope))
        # check scalar
        TikCheckUtil.check_type_match(scalar, (int, float, Expr, Scalar),
                                      "{} should be int, float, Expr or Scalar,"
                                      " input type is {}".format(scalar_name,
                                                                 type(scalar)))
        # check dtype
        if isinstance(scalar, Scalar):
            TikCheckUtil.check_equality(dst.dtype, scalar.dtype,
                                        "Intrinsic {}'s scalar's "
                                        "dtype should be"
                                        " equal to dst's dtype".
                                        format(print_name))
        TikCheckUtil.check_equality(api_check_support("tik." +
                                                      name, dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, print_name))
        if "int" in dst.dtype:
            TikCheckUtil.check_not_equality(
                type(scalar), float,
                "{} should not be float when {}.dtype is {}".format(
                    scalar_name, dst_name, dst.dtype))
        return dst_name

    def _check_vector_scalar_params_and_get_mask_and_extent(  # pylint: disable=R0913
            self, repeat_times, mask_mode, dst_blk_stride,
            dst_rep_stride, stride_unit, mask, dst, dst_name, mask_o):
        """check params for vector_scalar_elewise_func and
        get mask_o and dst extent
        """
        # check repeat
        check_repeat_times(repeat_times)
        # check mask_mode
        TikCheckUtil.check_in_range(
            mask_mode, ("normal", "counter"),
            "mask_mode should be 'normal' or 'counter'.")
        # check strides
        check_vector_stride([dst_blk_stride], [dst_rep_stride],
                            MAX_BLK_STRIDE_DOUBLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE, [dst_name])
        # check stride_unit
        check_stride_unit(stride_unit)
        # check mask and get mask_o
        if mask_o is None:
            mask_o = mask_concat(self, mask, mask_mode=mask_mode,
                                 tensor_bit_len=get_bit_len(dst.dtype))
        # check tensor overflow(static)
        if is_immediate_number(dst_blk_stride) and \
                stride_unit in (_STRIDE_UNIT_ZERO, _STRIDE_UNIT_ONE) \
                and dst_blk_stride == _DEFAULT_STRIDE:
            new_dst_bs = 1
        else:
            new_dst_bs = dst_blk_stride
        check_tensor_overflow((dst,), mask, repeat_times, (new_dst_bs,),
                              (dst_rep_stride,), (dst_name,),
                              stride_unit=stride_unit, mask_mode=mask_mode)
        # calculate dst_extent Byte
        dst_extent = cal_extent_stride_unit_mask(mask, repeat_times, dst,
                                                 stride_unit, new_dst_bs,
                                                 dst_rep_stride,
                                                 mask_mode=mask_mode)
        if stride_unit in (_STRIDE_UNIT_ZERO, _STRIDE_UNIT_ONE) and \
                is_basic_expr(dst_blk_stride):
            dst_extent = dst_extent + ONE_REP_BYTE_SIZE
        return mask_o, dst_extent

    @source_info_decorator(depth=2)
    @debug.vec_scalar_elewise_func_dec
    def _vector_scalar_elewise_func(self, name,  # pylint: disable=R0913, R0914
                                    mask, dst, scalar, repeat_times,
                                    dst_blk_stride, dst_rep_stride, stride_unit,
                                    mask_mode="normal",
                                    print_name=None, mask_o=None):
        """copy scalar to vector

        Parameters
        ----------
        dst : destination operator
        mask : Effective operation on element, divided into two model:
                Continuous and bit by bit.
        dst : destination operator
        scalar : the copied scalar
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
                         in one iteration
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # check dst and scalar
        if print_name is None:
            print_name = name
        dst_name = self._check_vector_scalar_operator_and_get_dst_name(
            name, dst, scalar, print_name)
        # check params and get mask_o, dst_extent
        mask_o, dst_extent = \
            self._check_vector_scalar_params_and_get_mask_and_extent(
                repeat_times, mask_mode, dst_blk_stride, dst_rep_stride,
                stride_unit, mask, dst, dst_name, mask_o)

        if name == "vci":
            config = [repeat_times, dst_blk_stride, dst_rep_stride,
                      stride_unit & 0b01, (stride_unit & 0b10) >> 1]
        else:
            config = [repeat_times, dst_blk_stride, _DEFAULT_STRIDE,
                      dst_rep_stride, _DEFAULT_STRIDE]
        # code gen
        self._gen_vector_scalar_code(scalar, dst, name, config, mask_o,
                                     mask_mode, dst_extent)

    def vector_dup(self,  # pylint: disable=R0913
                   mask,
                   dst,
                   scalar,
                   repeat_times,
                   dst_blk_stride,
                   dst_rep_stride,
                   stride_unit=0):
        """copy scalar to vector

        Parameters
        ----------
        dst : destination operator
        mask : Effective operation on element, divided into two model:
               Continuous and bit by bit.
        scalar : the copied scalar
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
                         in one iteration
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        return self._vector_scalar_elewise_func('vector_dup', mask, dst,
                                                scalar, repeat_times,
                                                dst_blk_stride, dst_rep_stride,
                                                stride_unit)

    def vci(self, mask, dst_index, start_point,  # pylint: disable=R0913
            repeat_times, dst_blk_stride, dst_rep_stride, stride_unit=0,
            mask_mode="normal"):
        """creat vector indexes from a start point

        mask : Effective operation on element, divided into two model:
               Continuous and bit by bit.
        dst_index : destination operator
        start_point : the start point
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
                         in one iteration
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        return self._vector_scalar_elewise_func('vci', mask, dst_index,
                                                start_point, repeat_times,
                                                dst_blk_stride, dst_rep_stride,
                                                stride_unit,
                                                mask_mode=mask_mode)

    def _gen_data_move_code(self, src, dst,  # pylint: disable=R0913, R0914
                            dma_list, type_args, args, argv, name="data_move"):
        src_key_str = TikUtil.get_storage_scope(src.scope)
        dst_key_str = TikUtil.get_storage_scope(dst.scope)
        key = src_key_str + " " + dst_key_str
        if key == "OUT L1":
            e_args = _extend_args("PadMode", args, argv)
        elif key in ["UB L0C", "L0C UB"]:
            e_args = _extend_args("ConvReluMode", args, argv)
        else:
            e_args = []
        TikCheckUtil.check_not_is(dma_list.get(key), None,
                                  "%s doesn't support %s to %s" %
                                  (name, src_key_str, dst_key_str))
        pipe_line, intrin_name = dma_list[src_key_str + " " + dst_key_str]
        with self.new_scope():
            # type_args : sid, nburst, burst, src_stride, dst_stride
            if src.dtype != "int32" or dst.dtype != "float16":
                self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", pipe_line)
                instr = tvm.call_extern(
                    dst.dtype, intrin_name, dst.access_ptr(
                        "w", extent=_calculate_extent("", src, dst,
                                                      [type_args[1],
                                                       type_args[2],
                                                       type_args[4]], False)),
                    src.access_ptr(
                        "r", extent=_calculate_extent("", src, dst,
                                                      [type_args[1],
                                                       type_args[2],
                                                       type_args[3]], True)),
                    *(type_convert(type_args + list(e_args))))
                # one ir is call_extern
                self.emit(instr, ONE_IR)

    def _dma_quant_set_deqscale_tensor(self, quant_param, relu_flag):
        # subclass has the member but parent class call it, so disable E1101
        cr_mode = CONV_RELU_VECTOR_QUANT  # 7
        with self.context.freeze():  # pylint: disable=E1101
            scale_addr = self.Scalar_(dtype="int64")  # pylint: disable=E1101
            # lsb: 32B
            # one ir is call_extern
            self.emit(tvm.call_extern(
                scale_addr.dtype, "reg_set", scale_addr.get(),
                tvm.expr.Cast("int64", quant_param.access_ptr("r")) //
                tvm.const(BYTE_SIZE, "int64")), ONE_IR)
            if relu_flag:
                scale_addr.set_as(scale_addr | (1 <<
                                                SCALE_ADDR_BIT_POS - 1))
            with self.new_scope():
                self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
                # one ir is call_extern
                self.emit(
                    tvm.call_extern(
                        "float16", "set_deqscale",
                        tvm.call_extern("float16", "reinterpret_cast",
                                        scale_addr.get())), ONE_IR)
        return cr_mode

    @source_info_decorator()
    @debug.dma_dquant_decorator
    def data_move_quant(self,  # pylint: disable=R0913
                        dst,
                        src,
                        sid,
                        nburst,
                        burst,
                        src_stride,
                        dst_stride,
                        quant_param,
                        relu_flag=False):
        """Move tensor from tik.cbuf to tik.ubuf

        Parameters
        ----------
        dst : destination operator
        src : source operation
        sid: 0, float16
        nburst : [1, 4095] continuous data segment for transfer instruction
        burst: nburst's length [1, 65535]
        dst_stride : offset of dst tensor between adjacent data segment
        src_stride : offset of src tensor between adjacent data segment
        quant_param : Anti-quantization parameter tensor start element
        relu_flag : True/False

        Returns
        -------
        None
        """
        check_integer_in_range(sid, range(MAX_SID),
                               "sid should be in the range of [0, 15]")
        check_dma_instr_params(dst, src, nburst, burst, src_stride, dst_stride)
        if isinstance(quant_param, Tensor):
            cr_mode = self._dma_quant_set_deqscale_tensor(quant_param,
                                                          relu_flag)
        else:
            cr_mode = CONV_RELU_QUANT # 3
            with self.new_scope():
                self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
                # one ir is call_extern
                self.emit(tvm.call_extern("float16", "set_deqscale",
                                          quant_param), ONE_IR)
        with self.new_scope():
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            args = type_convert(
                [sid, nburst, burst, src_stride, dst_stride, cr_mode])

            instr = tvm.call_extern(
                dst.dtype, "copy_matrix_cc_to_ubuf", dst.access_ptr(
                    "w", extent=_calculate_extent("", src, dst,
                                                  [args[1], args[2], args[4]],
                                                  False)),
                src.access_ptr(
                    "r", extent=_calculate_extent("", src, dst,
                                                  [args[1], args[2], args[3]],
                                                  True)),
                *args)
            # one ir is call extern
            self.emit(instr, ONE_IR)

    def _gen_tensor_mov_code(self, config,  # pylint: disable=R0913, R0914
                             src_scope, dst_scope, dma_list, pad_mode, src, dst,
                             dtype_str, block_mode, deqscale, src_onthefly,
                             en_onthefly):
        # arguments too many, so disabled.
        # instruction issue
        TikCheckUtil.check_in_range((src_scope, dst_scope), dma_list,
                                    "tensor_move doesn't support %s to %s" %
                                    (src_scope, dst_scope))
        # config: sid_store_mode, nburst, burst_len, src_stride, dst_stride
        src_extent = _calculate_extent(block_mode, src, dst,
                                       [config[1], config[2], config[3]], True)
        dst_extent = _calculate_extent(
            block_mode, src, dst, [config[1], config[2], config[4]], False,
            en_onthefly)
        if (src_scope, dst_scope) == ("gm", "cbuf"):
            config.append(pad_mode)
        elif (src_scope, dst_scope) in [("ubuf", "cc_m"), ("ubuf", "cc_v"),
                                        ("ubuf", "cc_sc"), ("cc_m", "ubuf"),
                                        ("cc_v", "ubuf"), ("cc_sc", "ubuf"),
                                        ("cc_dp", "ubuf")]:
            config.append(CR_MODE_MAP[dtype_str])
        pipe_line, intrin_name = dma_list[(src_scope, dst_scope)]
        with self.new_scope():
            if en_onthefly:
                self.scope_attr(
                    cce_params.CCE_AXIS, "critical_bank_conflict",
                    tvm.call_extern(
                        dst.dtype, "tvm_tuple", dst.access_ptr("w"),
                        deqscale.access_ptr("r"),
                        src_onthefly.access_ptr("r")))
            if intrin_name == "copy_matrix_cc_to_ubuf"\
                    and DTYPE_MAP[dst.dtype] in ('s8', 'u8'):
                instr = tvm.call_extern("int8", intrin_name + "_s8",
                                        dst.reinterpret_cast_to("int8")
                                        .access_ptr("w", extent=dst_extent),
                                        src.access_ptr("r", extent=src_extent),
                                        *type_convert(config))
            else:
                instr = tvm.call_extern(dst.dtype, intrin_name,
                                        dst.access_ptr("w", extent=dst_extent),
                                        src.access_ptr("r", extent=src_extent),
                                        *type_convert(config))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", pipe_line)
            # one ir is call_extern
            self.emit(instr, ONE_IR)

    def _check_tensor_mov_scope(self, scope_map,  # pylint: disable=R0913
                                src, dst, block_mode, archversion_scope):
        arch_version_str = get_soc_name() + get_soc_core_type()
        src_scope = scope_map[src.scope]
        dst_scope = scope_map[dst.scope]
        scope_str = _get_scope_str(scope_map, block_mode, src, dst)
        TikCheckUtil.check_in_range(
            scope_str, archversion_scope[arch_version_str],
            "%s Instruction tensor_mov doesn't support %s to %s." %
            (arch_version_str, src_scope, dst_scope))
        return arch_version_str

    @source_info_decorator()
    @debug.tensor_move_decorator
    def tensor_mov(self, dst, src, block_mode, # pylint: disable=R0913, R0914
                   nburst, burst_len, dst_stride, src_stride, deqscale=None,
                   sid_store_mode=0, relu=False, pad_mode=None,
                   pad_value=None, # pylint: disable=W0613
                   onthefly_mode=0, src_onthefly=None, src_onthefly_stride=0):
        # function's unused params(pad_value) is used in decorators, so disable
        # them
        """
        pad_value is not support yet, please use _set_padding().
        Leaky-Relu is not supported yet.
        block_mode: '' for 32B
                    'm' for 16*16 matrix(only cc)
                    'v' for 1*16 matrix(only cc)
                    'sc' for 16*4 matrix(only cc)
                    'dp' for 16*8 matrix(only cc)
        deqscale:   None for normal-mode
                    (float, Scalar.float) for deq-mode;
                    (int, Scalar.uint) for deq8/deq16/deqs16-mode;
                    (tensor.float) for vdeq-mode;
                    (tensor.uint) for vdeq8/vdeq16/vdeqs16-mode;
        sid_storeMode:  sid for gm
                        0 for store-high-16B(only cc)
                        1 for store-low-b6B(only cc)
                        2 for store-compact(only cc)
        Note: if you don't attempt to use/modify onthefly_mode/padMode,
            please config it as None or let it alone,otherwise, the
            perfermance will shrink.
        """
        # too many arguments, so disalbe R0914
        # arch_version_str+scope & scope+dtype_convrelu_onthefly &
        # arch_version_str+lrelu
        archversion_scope = {
            ASCEND_310AIC: [
                'gm2cbuf', 'gm2ubuf', 'cbuf2ubuf', 'cc_m162ubuf',
                'cc_m322ubuf', 'cc_v162ubuf', 'cc_v322ubuf', 'ubuf2gm',
                'ubuf2cbuf', 'ubuf2ubuf', 'ubuf2cc_m16', 'ubuf2cc_m32',
                'ubuf2cc_v16', 'ubuf2cc_v32'
            ],
            ASCEND_910AIC: [
                'gm2cbuf', 'gm2ubuf', 'cbuf2ubuf', 'cc_m162ubuf',
                'cc_m322ubuf', 'cc_v162ubuf', 'cc_v322ubuf', 'ubuf2gm',
                'ubuf2cbuf', 'ubuf2ubuf', 'ubuf2cc_m16', 'ubuf2cc_m32',
                'ubuf2cc_v16', 'ubuf2cc_v32'
            ],
            HI3796CV300ESAIC: [
                'gm2cbuf', 'gm2ubuf', 'cbuf2ubuf', 'cc_m162ubuf',
                'cc_m322ubuf', 'cc_v162ubuf', 'cc_v322ubuf', 'cc_dp162ubuf',
                'ubuf2gm', 'ubuf2cbuf', 'ubuf2ubuf', 'ubuf2cc_m16',
                'ubuf2cc_m32', 'ubuf2cc_v16', 'ubuf2cc_v32'
            ],
            HI3796CV300CSAIC: [
                'gm2cbuf', 'gm2ubuf', 'cbuf2ubuf', 'cbuf2cc_m16',
                'cbuf2cc_m32', 'cbuf2cc_v16', 'cbuf2cc_v32', 'cbuf2cc_sc32',
                'cc_m162ubuf', 'cc_m322ubuf', 'cc_v162ubuf', 'cc_v322ubuf',
                'cc_dp162ubuf', 'ubuf2gm', 'ubuf2cbuf', 'ubuf2ubuf',
                'ubuf2cc_m16', 'ubuf2cc_m32', 'ubuf2cc_v16', 'ubuf2cc_v32'
            ],
            AIC: [
                'gm2cbuf', 'gm2ubuf', 'cbuf2ubuf', 'cbuf2cc_m16',
                'cbuf2cc_m32', 'cbuf2cc_v16', 'cbuf2cc_v32', 'cbuf2cc_sc32',
                'cc_m162ubuf', 'cc_m322ubuf', 'cc_v162ubuf', 'cc_v322ubuf',
                'cc_dp162ubuf', 'cc_dp322ubuf', 'ubuf2gm', 'ubuf2cbuf',
                'ubuf2ubuf', 'ubuf2cc_m16', 'ubuf2cc_m32', 'ubuf2cc_v16',
                'ubuf2cc_v32'
            ],
            VEC: ['gm2ubuf', 'ubuf2gm']
        }
        archversion_convrelu = {
            ASCEND_310AIC: [0, 1, 2, 3, 4, 5, 6, 7],
            ASCEND_910AIC: [0, 1, 2, 3, 4, 5, 6, 7],
            HI3796CV300ESAIC: [0, 3, 5, 6, 7, 8, 9],
            HI3796CV300CSAIC: [0, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            AIC: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        }
        src_dst_scope_dtype_map = {
            "gm": {
                "cbuf": [
                    "u8u8", "s8s8", "f16f16", "u16u16", "s16s16", "f32f32",
                    "s32s32", "u32u32", "f64f64", "u64u64", "s64s64"
                ],
                "ubuf": [
                    "u8u8", "s8s8", "f16f16", "u16u16", "s16s16", "f32f32",
                    "s32s32", "u32u32", "f64f64", "u64u64", "s64s64"
                ]
            },
            "cbuf": {
                "ubuf": [
                    "u8u8", "s8s8", "f16f16", "u16u16", "s16s16", "f32f32",
                    "s32s32", "u32u32", "f64f64", "u64u64", "s64s64"
                ],
                "cc_m":
                    ["f16f16", "u16u16", "s16s16", "f32f32", "s32s32",
                     "u32u32"],
                "cc_v": ["f32f32", "s32s32", "u32u32"],
                "cc_sc": ["f32f32", "s32s32", "u32u32"]
            },
            "cc_m": {
                "ubuf": [
                    "f16f16", "f32f32", "s32s32", "f32f16", "f32f16relu",
                    "s32f16deq", "f16f16relu", "f32f32relu", "s32s32relu",
                    "f16f16deq", "s32f16vdeq", "s32f16vdeqrelu", "s32s8vdeq8",
                    "s32u8vdeq8", "s32s8vdeq8relu", "s32u8vdeq8relu",
                    "s32s8deq8", "s32u8deq8", "s32s8deq8relu", "s32u8deq8relu",
                    "s32f16vdeq16", "s32f16vdeq16relu", "s32f16deq16",
                    "s32f16deq16relu", "s32s16vdeqs16", "s32s16vdeqs16relu",
                    "s32s16deqs16", "s32s16deqs16relu", "u32u32"
                ]
            },
            "cc_v": {
                "ubuf": [
                    "f16f16", "f32f32", "s32s32", "f32f16", "f32f16relu",
                    "s32f16deq", "f16f16relu", "f32f32relu", "s32s32relu",
                    "f16f16deq", "s32f16vdeq", "s32f16vdeqrelu", "s32s8vdeq8",
                    "s32u8vdeq8", "s32s8vdeq8relu", "s32u8vdeq8relu",
                    "s32s8deq8", "s32u8deq8", "s32s8deq8relu", "s32u8deq8relu",
                    "s32f16vdeq16", "s32f16vdeq16relu", "s32f16deq16",
                    "s32f16deq16relu", "s32s16vdeqs16", "s32s16vdeqs16relu",
                    "s32s16deqs16", "s32s16deqs16relu"
                ]
            },
            "cc_sc": {
                "ubuf": [
                    "f32f32", "s32s32", "s32f16deq", "f32f32relu",
                    "s32s32relu", "s32f16vdeq", "s32f16vdeqrelu", "s32s8vdeq8",
                    "s32u8vdeq8", "s32s8vdeq8relu", "s32u8vdeq8relu",
                    "s32s8deq8", "s32u8deq8", "s32s8deq8relu", "s32u8deq8relu",
                    "s32f16vdeq16", "s32f16vdeq16relu", "s32f16deq16",
                    "s32f16deq16relu", "s32s16vdeqs16", "s32s16vdeqs16relu",
                    "s32s16deqs16", "s32s16deqs16relu"
                ]
            },
            "cc_dp": {
                "ubuf": [
                    "f16f16", "f32f32", "s32s32", "f32f16", "f32f16relu",
                    "f16f16relu", "f16f16deq"
                ]
            },
            "ubuf": {
                "gm": [
                    "u8u8", "s8s8", "f16f16", "u16u16", "s16s16", "f32f32",
                    "s32s32", "u32u32", "f64f64", "u64u64", "s64s64"
                ],
                "cbuf": [
                    "u8u8", "s8s8", "f16f16", "u16u16", "s16s16", "f32f32",
                    "s32s32", "u32u32", "f64f64", "u64u64", "s64s64"
                ],
                "ubuf": [
                    "u8u8", "s8s8", "f16f16", "u16u16", "s16s16", "f32f32",
                    "s32s32", "u32u32", "f64f64", "u64u64", "s64s64"
                ],
                "cc_m": ["f16f16", "f32f32", "s32s32", "f32f16", "u32u32"],
                "cc_v": ["f16f16", "f32f32", "s32s32", "f32f16", "u32u32"],
                "cc_sc": ["f32f32", "s32s32", "f32f16"]
            }
        }
        block_mode_appendix_map = {
            "": "",
            "m": "_matrix",
            "v": "_vector",
            "sc": "_small_matrix",
            "dp": "_depthwise"
        }
        dma_list = {
            # l0c16/l0c32-ub
            ("cc_m", "ubuf"): (2, 'copy_matrix_cc_to_ubuf'),
            # ub-l0c16/l0c32
            ("ubuf", "cc_m"): (2, 'copy_matrix_ubuf_to_cc'),
            # l0c16v/l0c32v-ub
            ("cc_v", "ubuf"): (2, 'copy_vector_cc_to_ubuf'),
            # ub-l0c16v/l0c32v
            ("ubuf", "cc_v"): (2, 'copy_vector_ubuf_to_cc'),
            # l0c32sc-ub
            ("cc_sc", "ubuf"): (2, 'copy_small_matrix_cc_to_ubuf'),
            # ub-l0c32sc
            ("ubuf", "cc_sc"): (2, 'copy_small_matrix_ubuf_to_cc'),
            # l0cdpf16/l0cdpf32-ub
            ("cc_dp", "ubuf"): (2, 'copy_depthwise_cc_to_ubuf'),
            ("ubuf", "ubuf"): (2, 'copy_ubuf_to_ubuf'),
            ("cbuf", "ubuf"): (4, 'copy_cbuf_to_ubuf'),
            ("cbuf", "cc_m"): (4, 'copy_cbuf_to_matrix_cc'),
            ("cbuf", "cc_v"): (4, 'copy_cbuf_to_vector_cc'),
            ("cbuf", "cc_sc"): (4, 'copy_cbuf_to_small_matrix_cc'),
            # LSU2
            ("gm", "cbuf"): (5, 'copy_gm_to_cbuf'),
            ("gm", "ubuf"): (5, 'copy_gm_to_ubuf'),
            # LSU3
            ("ubuf", "gm"): (6, 'copy_ubuf_to_gm'),
            ("ubuf", "cbuf"): (6, 'copy_ubuf_to_cbuf')
        }
        # check block_mode
        if block_mode == "":
            block_mode = "m"
        TikCheckUtil.check_in_range(
            block_mode, block_mode_appendix_map,
            "Please specify block_mode: ''/'m'/'v'/'sc'.")
        # check onthefly_mode
        TikCheckUtil.check_type_match(
            onthefly_mode, (int), "onthefly_mode should be int, input type: {}"
            .format(type(onthefly_mode)))
        # 0 is onthefly_mode default value
        arch_version_str = get_soc_name() + get_soc_core_type()
        if onthefly_mode != 0:
            TikCheckUtil.check_in_range(
                onthefly_mode, range(MIN_ONTHEFLY_MODE, MAX_ONTHEFLY_MODE),
                "Please specify onthefly_mode 1-add, 2-sub, input "
                "onthefly_mode: {}".format(onthefly_mode))
            en_onthefly = True
            TikCheckUtil.check_in_range(
                arch_version_str, ARCHVERSION_ONTHEFLY.keys(),
                "%s doesn't support onthefly." % (arch_version_str))
            TikCheckUtil.check_type_match(
                src_onthefly, (Tensor),
                "when onthefly_mode is not 0, src_onthefly should be Tensor, "
                "input type: {}".format(type(src_onthefly)))
            TikCheckUtil.check_in_range(
                DTYPE_MAP[src_onthefly.dtype],
                ARCHVERSION_ONTHEFLY[arch_version_str],
                "src_onthefly dtype should be in {}, input dtype: {}"
                .format(ARCHVERSION_ONTHEFLY[arch_version_str],
                        DTYPE_MAP[src_onthefly.dtype]))
            TikCheckUtil.check_not_equality(
                src_onthefly_stride, None,
                "when onthefly_mode is not 0, src_onthefly_stride should not "
                "be None")
        else:
            en_onthefly = False
        # check params
        check_dma_instr_params(dst, src, nburst, burst_len, src_stride,
                               dst_stride, en_onthefly, src_onthefly_stride)
        # check scope
        arch_version_str = self._check_tensor_mov_scope(SCOPE_MAP, src, dst,
                                                        block_mode,
                                                        archversion_scope)
        # check deqscale
        dtype_str = DTYPE_MAP[src.dtype] + DTYPE_MAP[dst.dtype]
        deq_mode = _make_deq_mode(dtype_str, deqscale)

        # regen scope
        src_scope, dst_scope = _regen_tensor_mov_scope(src, dst, block_mode)

        # check dtype
        dtype_str = dtype_str + deq_mode
        if relu:
            dtype_str = dtype_str + "relu"
            # check convrelu
            TikCheckUtil.check_in_range(
                arch_version_str, archversion_convrelu,
                "%s doesn't support convrelu feature." %
                (get_soc_name() + get_soc_core_type()))
            TikCheckUtil.check_in_range(
                CR_MODE_MAP[dtype_str], archversion_convrelu[arch_version_str],
                "%s doesn't support this convrelu mode: %s." %
                (get_soc_name() + get_soc_core_type(), CR_MODE_MAP[dtype_str]))
        TikCheckUtil.check_in_range(
            dtype_str, src_dst_scope_dtype_map[src_scope][dst_scope],
            "%s Instruction tensor_mov doesn't support %s in %s "
            "to %s in %s." % (get_soc_name() + get_soc_core_type(), src.dtype,
                              src_scope, dst.dtype, dst_scope))
        # check padMode
        if pad_mode is not None:
            TikCheckUtil.check_type_match(
                pad_mode, int,
                "padMode doesn't support online config or expr.")
            TikCheckUtil.check_in_range(pad_mode, range(MAX_PADMODE),
                                        "padMode should in range [0, 5].")
            TikCheckUtil.check_equality((src_scope, dst_scope), ("gm", "cbuf"),
                                        "PadMode only support OUT to L1.")
        else:
            pad_mode = PADMODE_NO_PADDING
        # check deqscale
        if isinstance(deqscale, Tensor):
            TikCheckUtil.check_equality(SCOPE_MAP[deqscale.scope], "ubuf",
                                        "deqscale should in UB.")
        # check sid
        check_integer_in_range(sid_store_mode, range(VALUE_3),
                               "sid_store_mode should be in the range of "
                               "[0, 2], input value: %s" % str(sid_store_mode))
        # set deqscale
        self._set_deqscale(deq_mode, deqscale, relu, en_onthefly, src_onthefly)
        # xt gen
        # low 2 bits: sid_store_mode   high 2 bits: onthefly_mode
        sid_store_mode = sid_store_mode + (onthefly_mode << SHIFT_BIT_POS_2)
        if en_onthefly:
            # low 8 bits: dst_stride  high 8 bits: src_onthefly_stride
            dst_onthefly_stride = dst_stride + \
                                  src_onthefly_stride << SHIFT_BIT_POS_8
        else:
            dst_onthefly_stride = dst_stride
        config = [sid_store_mode, nburst, burst_len, src_stride,
                  dst_onthefly_stride]
        self._gen_tensor_mov_code(config, src_scope, dst_scope, dma_list,
                                  pad_mode, src, dst, dtype_str, block_mode,
                                  deqscale, src_onthefly, en_onthefly)

    def _set_deqscale(self, deq_mode, deqscale, relu, # pylint: disable=R0913
                      en_onthefly, src_onthefly):
        # subclass has the member but parent class call it, so disable E1101
        if deq_mode in ("deq",):
            if en_onthefly:
                with self.context.freeze():  # pylint: disable=E1101
                    scale_addr = self.Scalar_("int64")  # pylint: disable=E1101
                    set_tensor_addr_to_scalar(self, scale_addr, src_onthefly)
                    scale_addr.set_as(
                        (scale_addr << DEQSCALE_SHIFT_POS) | deqscale)
                    with self.new_scope():
                        self.scope_attr(cce_params.CCE_AXIS, "coproc_scope",
                                        PIPE_V)
                        self.emit(tvm.call_extern("float16", "set_deqscale",
                                                  scale_addr.get()), ONE_IR)
            else:
                with self.new_scope():
                    self.scope_attr(cce_params.CCE_AXIS,
                                    "coproc_scope", PIPE_V)
                    # one ir is call_extern
                    self.emit(
                        tvm.call_extern("float16", "set_deqscale",
                                        _dtype_convert(deqscale, "float16")),
                        ONE_IR)
        elif deq_mode in ("deq8", "deq16", "deqs16"):
            with self.context.freeze():  # pylint: disable=E1101
                scale_addr = self.Scalar_("int64")  # pylint: disable=E1101
                if en_onthefly:
                    set_tensor_addr_to_scalar(self, scale_addr, src_onthefly)
                    scale_addr.set_as(
                        (scale_addr << DEQSCALE_SHIFT_POS) | deqscale)
                else:
                    scale_addr.set_as(deqscale)
                scale_addr.set_as(scale_addr | (int(relu) << SCALE_SHIFT_POS))
                with self.new_scope():
                    self.scope_attr(cce_params.CCE_AXIS, "coproc_scope",
                                    PIPE_V)
                    # one ir is call_extern
                    self.emit(
                        tvm.call_extern("int64", "set_deqscale",
                                        scale_addr.get()), ONE_IR)
        else:
            self._set_deqscale_expansion(deqscale, relu, en_onthefly,
                                         src_onthefly, deq_mode)

    def _set_deqscale_expansion(self, deqscale, relu,  # pylint: disable=R0913
                                en_onthefly, src_onthefly, deq_mode):
        """deq_mode in ("vdeq", "vdeq8", "vdeqs16") or else scope"""
        if deq_mode in ("vdeq", "vdeq8", "vdeqs16"):
            with self.context.freeze():  # pylint: disable=E1101
                scale_addr = self.Scalar_("int64")  # pylint: disable=E1101
                # lsb: 32B
                set_tensor_addr_to_scalar(self, scale_addr, deqscale)
                if deq_mode == "vdeq":
                    scale_addr.set_as(scale_addr |
                                      (int(relu) << SCALE_ADDR_BIT_POS))
                else:
                    scale_addr.set_as(scale_addr |
                                      (int(relu) << SCALE_SHIFT_POS))
                if en_onthefly:
                    onthefly_addr = self.Scalar_(
                        "int64") # pylint: disable=E1101
                    set_tensor_addr_to_scalar(self, onthefly_addr,
                                              src_onthefly)
                    scale_addr.set_as(scale_addr |
                                      (onthefly_addr << DEQSCALE_SHIFT_POS))
                with self.new_scope():
                    self.scope_attr(cce_params.CCE_AXIS, "coproc_scope",
                                    PIPE_V)
                    # one ir is call_extern
                    self.emit(tvm.call_extern("int64", "set_deqscale",
                                              scale_addr.get()), ONE_IR)
        else:
            if en_onthefly:
                with self.context.freeze():  # pylint: disable=E1101
                    scale_addr = self.Scalar("int64")  # pylint: disable=E1101
                    set_tensor_addr_to_scalar(self, scale_addr, src_onthefly)
                    scale_addr.set_as(scale_addr << DEQSCALE_SHIFT_POS)
                    with self.new_scope():
                        self.scope_attr(cce_params.CCE_AXIS, "coproc_scope",
                                        PIPE_V)
                        self.emit(tvm.call_extern("int64", "set_deqscale",
                                                  scale_addr.get()))

    @source_info_decorator()
    @debug.v4dtrans_decorator
    def v4dtrans(self, chw2hwc, dst, src,  # pylint: disable=R0913
                 m_len, channels):
        """ transform data between chw and hwc

        Parameters
        ----------
        chw2hwc : bool, True - chw->hwc; False - hwc->chw
        dst : destination operator
        src : source operation
        m_len : H*W direction dimension
        channels: size of C

        Returns
        -------
        None
        """
        # check tensor
        TikCheckUtil.check_type_match(src, Tensor,
                                      "src's type should be tensor, input "
                                      "type: %s" % type(src))
        TikCheckUtil.check_type_match(dst, Tensor,
                                      "dst's type should be tensor, input "
                                      "type: %s" % type(dst))
        # check scope
        TikCheckUtil.check_equality(src.scope, scope_ubuf,
                                    "src's scope must be UB, "
                                    "input scope is: %s" % src.scope)
        TikCheckUtil.check_equality(dst.scope, scope_ubuf,
                                    "dst's scope must be UB, "
                                    "input scope is: %s" % dst.scope)
        # check dtype
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "Intrinsic {}'s src's dtype should "
                                    "be equal to dst's dtype".
                                    format("v4dtrans"))
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" +
                                                            "v4dtrans",
                                                            dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, "v4dtrans"))
        # check mode
        TikCheckUtil.check_type_match(chw2hwc, bool, "chw2hwc should be bool.")
        if chw2hwc:
            dir_v4dtrans = 0
        else:
            dir_v4dtrans = 1
        # check m_len
        if isinstance(m_len, (int, float)):
            TikCheckUtil.check_in_range(
                m_len, range(MIN_M_LEN, MAX_M_LEN),
                "m_len should be in the range of [1,4095], "
                "input m_len: %s" % str(m_len))
            image_size = m_len*get_bit_len(src.dtype) // ONE_BYTE_BIT_LEN
            TikCheckUtil.check_equality(
                image_size % ONE_BLK_SIZE, 0,
                "H*W*dtype_size should be 32 Byte aligned, "
                "input size is %s" % str(image_size))
        # check channels
        if isinstance(channels, (int, float)):
            TikCheckUtil.check_in_range(
                channels, range(MIN_CHANNELS, MAX_CHANNELS),
                "channels should be in the range of [1,4095], "
                "input channels: %s" % str(channels))
        # change dtype_str
        dtype_str = change_dtype_str(dst)
        # code gen
        config = [m_len, channels, dir_v4dtrans]
        args = concat_params(config, V4DTRANS_OFFSET_LIST,
                             V4DTRANS_SEGMENT_LIST)
        # check tensor overflow, only for immediate
        if all(Expr(value).eval_value() is not None
               for value in (m_len, channels, src.offset, dst.offset)):
            # check address overlap
            check_addr_overlap_v4dtrans(dst, src, m_len, channels,
                                        Expr(dst.offset).eval_value(),
                                        Expr(src.offset).eval_value())

            TikCheckUtil.check_le(
                m_len*channels, reduce_mul(src.indice.origin_shape) -
                                Expr(src.offset).eval_value(),
                "src tensor overflow, m_len*channels is too big")
            TikCheckUtil.check_le(
                m_len*channels, reduce_mul(dst.indice.origin_shape) -
                                Expr(dst.offset).eval_value(),
                "dst tensor overflow, m_len*channels is too big")
        # cal extent
        src_extent = Expr(m_len*channels*DTYPE_SIZE[src.dtype]).get()
        dst_extent = Expr(m_len*channels*DTYPE_SIZE[dst.dtype]).get()
        # issue instruction
        with self.new_scope():
            instr = tvm.call_extern(
                dst.dtype, "v4dtrans",
                dst.reinterpret_cast_to(dtype_str).access_ptr(
                    "w", extent=dst_extent),
                src.reinterpret_cast_to(dtype_str).access_ptr(
                    "r", extent=src_extent), args)
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr)

    @source_info_decorator()
    @debug.vpadding_decorator
    def vpadding(self, mask, pad_mode,  # pylint: disable=R0913, R0914
                 pad_side, dst, src, repeat_times, dst_blk_stride,
                 src_blk_stride, dst_rep_stride,
                 src_rep_stride, stride_unit=0, mask_mode="normal"):
        """ padding src tensor

        Parameters
        ----------
        mask:
        pad_mode: 0 -> nearest-padding(aaa|a)
                  1 -> symmetric_padding0(abc|cba)
                  2 -> symmetric_padding1(ab|cba)
        pad_side: 'left'/'right'.
        dst: dst operator
        src: src operator
        repeat_times: Repeated iterations times
        dst_blk_stride: offset of dst operator between different block
                         in one iteration
        src_blk_stride: offset of src operator between different block
                         in one iteration
        dst_rep_stride: offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride: offset of src operator in the same block
                         between adjacent iterations
        stride_unit: address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # check pad_mode
        TikCheckUtil.check_type_match(
            pad_mode, int,
            "pad_mode should be int, input pad_mode: {}".format(type(pad_mode)))
        check_integer_in_range(
            pad_mode, range(MAX_PAD_MODE),
            "pad_mode should be in the range of [0, 2], input pad_mode: {}"
            .format(pad_mode))
        # check pad_side
        TikCheckUtil.check_in_range(pad_side, ("left", "right"),
                                    "pad_side should be 'left' or 'right', "
                                    "input pad_side: {}".format(pad_side))
        if pad_side == "left":
            pad_side_t = 0
        else:
            pad_side_t = 1
        # check mask_mode
        TikCheckUtil.check_in_range(
            mask_mode, ("normal", "counter"),
            "mask_mode should be 'normal' or 'counter'.")
        # check repeat
        check_repeat_times(repeat_times)
        # check strides
        check_vector_stride(
            [dst_blk_stride, src_blk_stride], [dst_rep_stride, src_rep_stride],
            MAX_BLK_STRIDE_DOUBLE_BYTE, MAX_REP_STRIDE_SINGLE_BYTE,
            ["dst", "src"])
        # check stride_unit
        TikCheckUtil.check_type_match(stride_unit, int,
                                      "stride_unit shoule be int, input stride_"
                                      "unit: {}".format(type(stride_unit)))
        check_integer_in_range(
            stride_unit, range(MAX_STRIDE_UNIT),
            "stride_unit should be in the range of [0, 3], "
            "input stride_unit: {}".format(stride_unit))
        # check tensor dtype
        TikCheckUtil.check_type_match(src, Tensor, "src should be Tensor.")
        TikCheckUtil.check_type_match(dst, Tensor, "dst should be Tensor.")
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "Intrinsic {}'s src's dtype should be "
                                    "equal to dst's dtype".format("vpadding"))
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" +
                                                            "vpadding",
                                                            dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, "vpadding"))
        # check tensor scope
        TikCheckUtil.check_equality(src.scope, scope_ubuf,
                                    "src's scope must be UB")
        TikCheckUtil.check_equality(dst.scope, scope_ubuf,
                                    "dst's scope must be UB")
        # mask
        mask_o = mask_concat(self, mask, mask_mode,
                             tensor_bit_len=max(get_bit_len(dst.dtype),
                                                get_bit_len(src.dtype)))
        # check tensor overflow(static)
        if mask_mode == "normal":
            # all elements in src are read even their mask bits are invalid
            if get_bit_len(src.dtype) == 32:
                mask_len = MASK_VALUE_64
            else:
                mask_len = MASK_VALUE_128
        else:
            mask_len = mask

        # check address overlapping
        if src.buffer == dst.buffer:
            if all(isinstance(value, int) for \
                   value in (repeat_times, dst_blk_stride, src_blk_stride,
                             dst_rep_stride, src_rep_stride)):
                check_address_overlapping(
                    "vpadding", mask,
                    dst, src, BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE // max(
                        get_bit_len(dst.dtype), get_bit_len(src.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(src.dtype),
                    repeat_times, dst_blk_stride, src_blk_stride,
                    dst_rep_stride, src_rep_stride,
                    Expr(dst.offset).eval_value(),
                    Expr(src.offset).eval_value(),
                    stride_unit, mask_mode, src_mask=mask_len)

        vector_tensor_overflow_check(src, mask_len, BLK_NUM_PER_REP,
                                     ONE_REP_BYTE_SIZE // get_bit_len(
                                         src.dtype),
                                     repeat_times, src_blk_stride,
                                     src_rep_stride, "src tensor overflow",
                                     stride_unit, mask_mode)
        vector_tensor_overflow_check(dst, mask, BLK_NUM_PER_REP,
                                     ONE_REP_BYTE_SIZE // get_bit_len(
                                         dst.dtype),
                                     repeat_times, dst_blk_stride,
                                     dst_rep_stride, "dst tensor overflow",
                                     stride_unit, mask_mode)
        # change dtype_str
        dtype_str = change_dtype_str(dst)
        # cal extent
        src_extent = cal_extent_stride_unit_mask(
            mask, repeat_times, src, stride_unit, src_blk_stride,
            src_rep_stride, mask_mode)
        dst_extent = cal_extent_stride_unit_mask(
            mask, repeat_times, dst, stride_unit, src_blk_stride,
            src_rep_stride, mask_mode)
        # code gen
        config = [dst_blk_stride, src_blk_stride, dst_rep_stride,
                  src_rep_stride, pad_mode, pad_side_t, stride_unit,
                  repeat_times]
        args = concat_params(config, VPADDING_OFFSET_LIST,
                             VPADDING_SEGMENT_LIST)
        with self.new_scope():
            if mask_mode == "counter":
                # save orig_ctrl
                orig_ctrl = set_ctrl_counter_mask(self)

            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            instr = tvm.call_extern(
                dst.dtype, "vpadding",
                dst.reinterpret_cast_to(dtype_str).access_ptr(
                    "w", extent=dst_extent),
                src.reinterpret_cast_to(dtype_str).access_ptr(
                    "r", extent=src_extent), args)

            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr)

            # reset CTRL SPR as orig_ctrl
            if mask_mode == "counter":
                reset_ctrl_counter_mask(self, orig_ctrl)

    @source_info_decorator()
    @debug.vscatter_decorator
    def vscatter(self, mask, dst, src, dst_offset,  # pylint: disable=R0913
                 repeat_times,
                 src_rep_stride, base_addr=0,
                 stride_unit=0, mask_mode="normal"):
        """
        Scatter elements from src to dst according to offsets in dst_offset,
            programmer should ensure all elements in dst space.

        Parameters
        ----------
        mask:
        dst: dst operator
        src: src operator
        dst_offset: addr offset tensor
        repeat_times: Repeated iterations times
        src_rep_stride: offset of src operator in the same block
                         between adjacent iterations
        stride_unit: address and offset unit both affect it. default = 0
        base_addr: init offset for dst,  default = 0
        mask_mode: mode of mask, counter or normal. default = normal

        Returns
        -------
        None
        """
        # check tensor dtype
        TikCheckUtil.check_type_match(src, Tensor,
                                      "src should be Tensor."
                                      " input type: {}".format(type(src)))
        TikCheckUtil.check_type_match(dst, Tensor,
                                      "dst should be Tensor."
                                      " input type: {}".format(type(dst)))
        TikCheckUtil.check_type_match(dst_offset, Tensor,
                                      "dst_offset should be Tensor. "
                                      "input type: {}".format(type(dst_offset)))
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "Intrinsic {}'s src's dtype should be "
                                    "equal to dst's dtype".format("vscatter"))
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" +
                                                            "vscatter",
                                                            dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, "vscatter"))
        # check offset tensor dtype
        TikCheckUtil.check_equality(dst_offset.dtype, "int32",
                                    "dtype of dst_offset should be 'int32', "
                                    "but input dtype is "
                                    "'{}'".format(dst_offset.dtype))
        # check tensor scope
        _check_vscatter_vgather_operator_scope(src, dst, dst_offset,
                                               "dst_offset")

        # check mask_mode
        TikCheckUtil.check_in_range(
            mask_mode, ("normal", "counter"),
            "mask_mode should be a str of 'normal' or 'counter'."
            " input mask_mode: {}".format(mask_mode))
        # mask
        mask_o = mask_concat(self, mask, mask_mode,
                             tensor_bit_len=max(get_bit_len(src.dtype),
                                                get_bit_len(dst.dtype)))
        # check repeat
        check_repeat_times(repeat_times)
        # check strides
        check_vector_stride(
            None, [src_rep_stride],
            None, MAX_REP_STRIDE_SINGLE_BYTE,
            ["src"])
        # check base_addr
        self._vscatter_check_base_addr(base_addr, dst)
        # check stride_unit
        TikCheckUtil.check_type_match(
            stride_unit, int,
            "stride_unit shoule be int, input stride_unit:"
            " {}".format(type(stride_unit)))
        check_integer_in_range(
            stride_unit, range(MAX_STRIDE_UNIT),
            "stride_unit should be in the range of [0, 3], "
            "input stride_unit: {}".format(stride_unit))

        # check tensor overflow(static)
        self._vscatter_check_overflow(mask, src, dst_offset, repeat_times,
                                      src_rep_stride, stride_unit, mask_mode)
        # code gen
        if isinstance(base_addr, Scalar):
            with self.context.freeze():  # pylint: disable=E1101
                with self.new_scope():
                    self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_S)
                    base_addr.set_as(base_addr +
                                     tvm.expr.Cast("int64",
                                                   tvm.call_extern(
                                                       "handle", "",
                                                       dst.access_ptr("rw"))))
                    self.total_ir_lines += TWO_IR
        else:
            base_addr = base_addr + tvm.expr.Cast(
                "int64", tvm.call_extern("handle", "", dst.access_ptr("rw")))
        self._vscatter_code_gen(dst, src, dst_offset, mask_o,
                                repeat_times, src_rep_stride,
                                base_addr, stride_unit, mask_mode)

    def _vscatter_code_gen(self, dst, src,  # pylint: disable=R0913, R0914
                           dst_offset, mask_o, repeat_times, src_rep_stride,
                           base_addr, stride_unit, mask_mode):
        """vscatter code gen part"""
        default_stride = 0
        config = [base_addr, default_stride,
                  src_rep_stride, stride_unit, repeat_times]
        args = concat_params(config,
                             VSCATTER_VGATHER_XT_OFFSET_LIST,
                             VSCATTER_VGATHER_XT_SEGMENT_LIST)
        if get_bit_len(src.dtype) == 16:
            dtype_str = "uint16"
        else:
            dtype_str = "uint32"
        with self.new_scope():
            if mask_mode == "counter":
                # save orig_ctrl
                orig_ctrl = set_ctrl_counter_mask(self)

            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))

            instr = tvm.call_extern(
                dst_offset.dtype, "vscatter",
                dst_offset.reinterpret_cast_to('uint32').access_ptr("r"),
                src.reinterpret_cast_to(dtype_str).access_ptr("r"), args)
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.scope_attr(cce_params.CCE_AXIS, "append_mem",
                            tvm.call_extern("handle", "mem_vector",
                                            dst.access_ptr("rw")))
            self.emit(instr, THREE_IR)

            # reset CTRL SPR as orig_ctrl
            if mask_mode == "counter":
                reset_ctrl_counter_mask(self, orig_ctrl)

    @staticmethod
    def _vscatter_check_base_addr(base_addr, dst):
        """check vscatter base_addr param"""
        TikCheckUtil.check_type_match(
            base_addr, (int, Scalar),
            "base_addr's type should be int or Scalar,"
            " input type is {}".format(type(base_addr)))
        if isinstance(base_addr, int):
            # 2**31 - 1 is max base_addr, 31 bit len
            TikCheckUtil.check_in_range(
                base_addr, range(2 ** 31),
                "base_addr should be in range of [0, 2**31 - 1]."
                " input base_addr: {}".format(base_addr))

            # check valid
            dst_scope_size = reduce_mul(dst.indice.origin_shape) * \
                             get_bit_len(dst.dtype) // ONE_BYTE_BIT_LEN
            TikCheckUtil.check_le(base_addr, dst_scope_size,
                                  "base_addr should be less equal than"
                                  " dst tensor's buffer size: {},"
                                  " input base_add: {}".format(
                                      dst_scope_size, base_addr))
        elif isinstance(base_addr, Scalar):
            TikCheckUtil.check_equality(base_addr.dtype, "uint32",
                                        "Scalar base_addr "
                                        "should be dtype of uint32")

    @staticmethod
    def _vscatter_check_overflow(mask, src, dst_offset,  # pylint: disable=R0913
                                 repeat_times, src_rep_stride,
                                 stride_unit, mask_mode):
        """vscatter instr check overflow"""
        if mask_mode == "normal":
            # all elements in src are read even their mask bits are invalid
            if get_bit_len(src.dtype) == 32:
                mask_len = MASK_VALUE_64
            else:
                mask_len = MASK_VALUE_128
        else:
            mask_len = mask

        default_blk_stride = 1
        default_rep_stride = 8
        # check dst_offset imm read mem.
        # change offset dtype temporarily, for check overflow
        dst_offset.dtype = src.dtype
        vector_tensor_overflow_check(dst_offset, mask_len,
                                     BLK_NUM_PER_REP,
                                     ONE_REP_BYTE_SIZE // get_bit_len(
                                         src.dtype),
                                     repeat_times, default_blk_stride,
                                     default_rep_stride,
                                     "dst_offset tensor overflow",
                                     mask_mode=mask_mode)
        # reset dst_offset dtype: int32
        dst_offset.dtype = "int32"

        if stride_unit in (2, 3):
            # stide_unit:2,3 for gap, unit is element
            default_blk_stride = 0
        vector_tensor_overflow_check(src, mask_len, BLK_NUM_PER_REP,
                                     ONE_REP_BYTE_SIZE // get_bit_len(
                                         src.dtype),
                                     repeat_times, default_blk_stride,
                                     src_rep_stride, "src tensor overflow",
                                     stride_unit, mask_mode)

    @source_info_decorator()
    @debug.vgather_decorator
    def vgather(self, mask, dst, src, src_offset,  # pylint: disable=R0913
                repeat_times, dst_rep_stride,
                base_addr=0, stride_unit=0, mask_mode="normal"):
        """
        Gather elements from src to dst according to address
            in src_offset, programmer should ensure all elements in src.

        Parameters
        ----------
        mask:
        dst: dst operator
        src: src operator
        src_offset: addr offset tensor
        repeat_times: Repeated iterations times
        dst_rep_stride: offset of dst operator in the same block
                         between adjacent iterations
        stride_unit: address and offset unit both affect it. default = 0
        base_addr: init offset for dst,  default = 0
        mask_mode: mode of mask, counter or normal. default = normal

        Returns
        -------
        None
        """
        # check tensor dtype
        TikCheckUtil.check_type_match(src, Tensor,
                                      "src should be Tensor."
                                      " input type: {}".format(type(src)))
        TikCheckUtil.check_type_match(dst, Tensor,
                                      "dst should be Tensor."
                                      " input type: {}".format(type(dst)))
        TikCheckUtil.check_type_match(src_offset, Tensor,
                                      "src_offset should be Tensor. input "
                                      "type: {}".format(type(src_offset)))
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "Intrinsic {}'s src's dtype should"
                                    " be equal to dst's dtype".
                                    format("vgather"))
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_"
                                                            + "vgather",
                                                            dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, "vgather"))
        # check offset tensor dtype
        TikCheckUtil.check_equality(
            src_offset.dtype, "int32",
            "dtype of src_offset should be 'int32', but input dtype is "
            "'{}'".format(src_offset.dtype))
        # check tensor scope
        _check_vscatter_vgather_operator_scope(src, dst, src_offset,
                                               "src_offset")

        # check mask_mode
        TikCheckUtil.check_in_range(
            mask_mode, ("normal", "counter"),
            "mask_mode should be a str of 'normal' or 'counter'."
            " input mask_mode: {}".format(mask_mode))
        # mask
        mask_o = mask_concat(self, mask, mask_mode,
                             tensor_bit_len=max(get_bit_len(src.dtype),
                                                get_bit_len(dst.dtype)))
        # check repeat
        check_repeat_times(repeat_times)
        # check strides
        check_vector_stride(
            None, [dst_rep_stride],
            None, MAX_REP_STRIDE_SINGLE_BYTE,
            ["dst"])
        # check base_addr
        self._vgather_check_base_addr(base_addr, src)
        # check stride_unit
        TikCheckUtil.check_type_match(
            stride_unit, int,
            "stride_unit shoule be int, input stride_unit: {}".format(
                type(stride_unit)))
        check_integer_in_range(
            stride_unit, range(MAX_STRIDE_UNIT),
            "stride_unit should be in the range of [0, 3], "
            "input stride_unit: {}".format(stride_unit))
        # check tensor overflow(static)
        self._vgather_check_overflow(mask, dst, src, src_offset, repeat_times,
                                     dst_rep_stride, stride_unit, mask_mode)
        # code gen
        if isinstance(base_addr, Scalar):
            with self.context.freeze():  # pylint: disable=E1101
                with self.new_scope():
                    self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_S)
                    base_addr.set_as(base_addr +
                                     tvm.expr.Cast("int64", tvm.call_extern(
                                         "handle", "", src.access_ptr("r"))))
                    self.total_ir_lines += TWO_IR
        else:
            base_addr = base_addr + tvm.expr.Cast("int64",
                                                  tvm.call_extern(
                                                      "handle", "",
                                                      src.access_ptr("r")))

        self._vgather_code_gen(dst, src, src_offset,
                               mask_o, repeat_times, dst_rep_stride,
                               base_addr, stride_unit, mask_mode)

    def _vgather_code_gen(self, dst, src,  # pylint: disable=R0913, R0914
                          src_offset, mask_o, repeat_times, dst_rep_stride,
                          base_addr, stride_unit, mask_mode):
        """vgather code gen part"""
        default_stride = 0
        config = [base_addr, dst_rep_stride,
                  default_stride, stride_unit, repeat_times]
        args = concat_params(config,
                             VSCATTER_VGATHER_XT_OFFSET_LIST,
                             VSCATTER_VGATHER_XT_SEGMENT_LIST)
        if get_bit_len(src.dtype) == 16:
            dtype_str = "uint16"
        else:
            dtype_str = "uint32"
        with self.new_scope():
            if mask_mode == "counter":
                # save orig_ctrl
                orig_ctrl = set_ctrl_counter_mask(self)

            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            instr = tvm.call_extern(
                src_offset.dtype, "vgather",
                dst.reinterpret_cast_to(dtype_str).access_ptr("w"),
                src_offset.reinterpret_cast_to("uint32").access_ptr("r"), args)
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.scope_attr(cce_params.CCE_AXIS, "append_mem",
                            tvm.call_extern("handle",
                                            "mem_vector", src.access_ptr("r")))
            self.emit(instr, THREE_IR)

            # reset CTRL SPR as orig_ctrl
            if mask_mode == "counter":
                reset_ctrl_counter_mask(self, orig_ctrl)

    @staticmethod
    def _vgather_check_base_addr(base_addr, src):
        """check vgather base_addr param"""
        TikCheckUtil.check_type_match(
            base_addr, (int, Scalar),
            "base_addr's type should be int or Scalar,"
            " input type is {}".format(type(base_addr)))
        if isinstance(base_addr, int):
            # 2**31 - 1 is max base_addr, for 31 bit len.
            TikCheckUtil.check_in_range(
                base_addr, range(2**31),
                "base_addr should in range [0, 2**31-1].")

            # check valid
            src_scope_size = reduce_mul(src.indice.origin_shape) * \
                             get_bit_len(src.dtype) // ONE_BYTE_BIT_LEN
            TikCheckUtil.check_le(
                base_addr, src_scope_size,
                "base_addr should be less equal than src tensor's buffer "
                "size: {}, input base_add: {}".format(
                    src_scope_size, base_addr))
        elif isinstance(base_addr, Scalar):
            TikCheckUtil.check_equality(
                base_addr.dtype, "uint32",
                "Scalar base_addr should be type of uint32")

    @staticmethod
    def _vgather_check_overflow(mask, dst, src,  # pylint: disable=R0913
                                src_offset, repeat_times, dst_rep_stride,
                                stride_unit, mask_mode):
        """vgather instr check tensor overflow"""
        if mask_mode == "normal":
            # all elements in src are read even their mask bits are invalid
            if get_bit_len(src.dtype) == 32:
                # b32: 64
                mask_len = MASK_VALUE_64
            else:
                mask_len = MASK_VALUE_128
        else:
            mask_len = mask

        # default stride_unit: 0, 1 for stride, unit is 32B
        default_blk_stride = 1
        default_rep_stride = 8
        # check src_offset imm read mem.
        # change offset dtype temporarily, for check overflow
        src_offset.dtype = src.dtype
        vector_tensor_overflow_check(src_offset, mask_len, BLK_NUM_PER_REP,
                                     ONE_REP_BYTE_SIZE // get_bit_len(
                                         src.dtype),
                                     repeat_times, default_blk_stride,
                                     default_rep_stride,
                                     "src_offset tensor overflow",
                                     mask_mode=mask_mode)
        # reset src_offset dtype: int32
        src_offset.dtype = "int32"

        # stride_unit: 2, 3 is for gap, unit is element.
        if stride_unit in (2, 3):
            default_blk_stride = 0

        # check dst imm write mem.
        vector_tensor_overflow_check(dst, mask, BLK_NUM_PER_REP,
                                     ONE_REP_BYTE_SIZE // get_bit_len(
                                         dst.dtype),
                                     repeat_times, default_blk_stride,
                                     dst_rep_stride, "dst tensor overflow",
                                     stride_unit, mask_mode)

    @source_info_decorator()
    @debug.load3dv2_decorator
    def load3dv2(self, dst, src, pad_list,  # pylint: disable=R0913, R0914
                 l1_h, l1_w, channel_size, k_extension, m_extension, k_start_pt,
                 m_start_pt, stride_w, stride_h, filter_w, filter_h,
                 dilation_filter_w, dilation_filter_h,
                 en_transpose=False, en_small_k=False, pad_value=None):
        """image to colomn, only support v200, only support L1 to L0A/L0B/UB

        Parameters
        ----------
        dst: destination operator
        src: source operator
        pad_list: [left, right, top, bottom]
        l1_h: height of src tensor
        l1_w: width of src tensor
        channel_size: number of src tensor's channels
        k_extension: k direction extension steps from the start position
        m_extension: m direction extension steps from the start position
        k_start_pt: k direction start position of the feature matrix
        m_start_pt: m direction start position of the feature matrix
        stride_w: filter stride size in w dimension
        stride_h: filter stride size in h dimension
        filter_w: width of filter
        filter_h: height of filter
        dilation_filter_w: dilation size of filter in w dimension
        dilation_filter_h: dilation size of filter in h dimension
        en_transpose: enable transpose, default = None
        en_small_k: enable small_k, default = None
        pad_value: value for padding, default = None

        Returns
        -------
        None
        """
        scope_map = {scope_ca: 'ca', scope_cb: 'cb', scope_ubuf: 'ub',
                     scope_cbuf: 'cbuf'}
        # check dst type
        TikCheckUtil.check_type_match(dst, Tensor,
                                      "dst should be tensor, input type"
                                      " of dst: {}".format(type(dst)))
        # check src type
        TikCheckUtil.check_type_match(src, Tensor,
                                      "src should be tensor, input type"
                                      " of src: {}".format(type(src)))
        # check tensor scope
        TikCheckUtil.check_in_range(
            scope_map[dst.scope], ('ca', 'cb', 'ub'),
            "dst_scope should be l0a, l0b or ub for load3dv2, "
            "input dst_scope: {}".format(dst.scope))
        TikCheckUtil.check_in_range(scope_map[src.scope], ('cbuf'),
                                    "src_scope should be l1 for load3dv2, "
                                    "input src_scope: {}".format(src.scope))
        # check tensor dtype
        dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src.dtype]
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "Intrinsic {}'s src's dtype should be "
                                    "equal to dst's dtype".format("load3dv2"))
        TikCheckUtil.check_equality(api_check_support("tik." + "load3dv2",
                                                      dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, "load3dv2"))
        # check
        _load3dv1_load3dv2_col2img_check(pad_list, l1_w, l1_h, stride_w,
                                         stride_h, filter_w, filter_h,
                                         dilation_filter_w, dilation_filter_h)
        _load3dv2_check(k_extension, m_extension, m_start_pt, pad_value,
                        channel_size)
        # check channel_size
        check_load3dv2_channel_size(channel_size, src.dtype)
        # check start-point
        TikCheckUtil.check_type_match(
            k_start_pt, (int, Scalar, Expr),
            "k_start_pt should be int, Scalar, Expr, input type of "
            "k_start_pt: {}".format(type(k_start_pt)))
        check_scalar_dtype(k_start_pt,
                           "scalar_k_start_pt should be a scalar of int/uint")
        check_integer_in_range(
            k_start_pt, range(MAX_START_PT),
            "k_start_pt should be in the range of [0, 65535], "
            "input k_start_pt: {}".format(k_start_pt))
        if isinstance(k_start_pt, int):
            k_start_pt_byte_align = 32
            if k_start_pt*DTYPE_SIZE[src.dtype] % k_start_pt_byte_align != 0:
                TikCheckUtil.raise_error(
                    "k_start_pt in Byte should be multiple of 32B, input "
                    "k_start_pt: {}, input src dtype: {}"
                    .format(k_start_pt, src.dtype))
        # check en_small_k en_and transpose
        sk_tp_bit = self._check_sk_tp(en_small_k, en_transpose)
        # check dilation filter size and l1_h_w size
        check_dilation_filter_size(
            filter_w, dilation_filter_w, l1_w, pad_list[PADDING_LEFT_IDX],
            pad_list[PADDING_RIGHT_IDX], "W")
        check_dilation_filter_size(
            filter_h, dilation_filter_h, l1_h, pad_list[PADDING_TOP_IDX],
            pad_list[PADDING_BOT_IDX], "H")

        # check m_extension and k_extension
        check_load3dv2_m_extension(
            filter_w, filter_h, dilation_filter_w, dilation_filter_h, pad_list,
            m_extension, l1_w, l1_h, stride_w, stride_h, m_start_pt)
        check_load3dv2_k_extension(channel_size, k_extension, filter_h,
                                   filter_w, k_start_pt, src.dtype)

        # check dst tensor overflow
        _check_dst_overflow_load3dv2(k_start_pt, m_start_pt, k_extension,
                                     m_extension, dst)
        # FMATRIX
        orig_params = []
        params = [l1_w, l1_h, pad_list[PADDING_LEFT_IDX],
                  pad_list[PADDING_RIGHT_IDX], pad_list[PADDING_TOP_IDX],
                  pad_list[PADDING_BOT_IDX]]
        orig_params += params[:]
        reg_fmatrix = concat_params(params, FMATRIX_OFFSET_LIST,
                                    FMATRIX_SEGMENT_LIST)
        self._do_load3d_fmatrix(reg_fmatrix)
        # padding
        do_load3d_padding(self, src, pad_value)
        # cal extent
        dst_extent = Expr((m_extension + 1)*(k_extension + 1)*
                          DTYPE_SIZE[dst.dtype]).get()
        # code gen
        params = [k_extension, m_extension, k_start_pt, m_start_pt]
        reg_xm = concat_params(params, LOAD3DV2_REG_XM_OFFSET_LIST,
                               LOAD3DV2_REG_XM_SEGMENT_LIST)
        orig_params += params[:]

        params = [stride_w, stride_h, filter_w, filter_h,
                  dilation_filter_w, dilation_filter_h,
                  sk_tp_bit, channel_size]
        reg_xt = concat_params(params, LOAD3DV2_REG_XT_OFFSET_LIST,
                               LOAD3DV2_REG_XT_SEGMENT_LIST)
        orig_params += params[:]

        if dtype_str in ("s4s4", "u4u4"):
            dtype_str = "int4"
        else:
            dtype_str = dst.dtype
        with self.new_scope():
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_MTE1)
            instr = tvm.call_extern(
                dst.dtype, "img2colv2_cbuf_to_" + scope_map[dst.scope],
                dst.reinterpret_cast_to(dtype_str).access_ptr(
                    "w", extent=dst_extent),
                src.reinterpret_cast_to(dtype_str).access_ptr("r"), reg_xm,
                reg_xt)
            self.emit(instr)

    def _check_sk_tp(self, en_small_k, en_transpose):
        arch_version_str = get_soc_name() + get_soc_core_type()
        TikCheckUtil.check_type_match(
            en_small_k, bool, "en_small_k should be bool, input type of "
                              "en_small_k: {}".format(type(en_small_k)))
        if en_small_k:
            TikCheckUtil.check_in_range(
                arch_version_str, LOAD3DV2_FUNC_MAP["sk"],
                "{} doesn't support small_k"
                .format(arch_version_str))
            sk_tp_bit = 1
        else:
            sk_tp_bit = 0
        # check transpose
        TikCheckUtil.check_type_match(
            en_transpose, bool, "en_transpose should be bool, input type of "
                                "en_transpose: {}".format(type(en_transpose)))
        if en_transpose:
            TikCheckUtil.check_in_range(
                arch_version_str, (HI3796CV300CSAIC, AIC),
                "{} doesn't support transpose"
                .format(arch_version_str))
            sk_tp_bit = 1
        else:
            sk_tp_bit = 0

        return sk_tp_bit

    @source_info_decorator()
    @debug.load_smask_decorator
    def load_smask(self, dst, src, load_size, sid=0):
        """load src to smask

        Parameters
        ----------
        dst: destination operator
        src: source operator
        load_size: load size, unit: 2B
        sid: SID for OUTSMMU

        Returns
        -------
        None
        """
        instr_map = {
            scope_ubuf: ["load_smask_table_from_ub", PIPE_MTE3],
            scope_gm: ["load_smask_table_from_gm", PIPE_MTE2]
        }
        TikCheckUtil.check_type_match(
            dst, Tensor, "dst should be Tensor, input type: {}"
            .format(type(dst)))
        TikCheckUtil.check_equality(
            dst.scope, scope_smask,
            "dst scope should be SMASK, input scope: {}".format(dst.scope))
        TikCheckUtil.check_type_match(
            src, Tensor,
            "dst should be Tensor, input type: {}".format(type(src)))
        TikCheckUtil.check_in_range(
            src.scope, (scope_ubuf, scope_gm),
            "src scope should be gm or ub, input scope: {}".format(src.scope))
        TikCheckUtil.check_type_match(
            load_size, int,
            "load_size should be int, input type: {}".format(type(load_size)))
        check_integer_in_range(
            load_size, range(VALUE_128),
            "load_size should be in the range of [0, 127], input load_size: {}"
            .format(load_size))
        instr_name, pipe_line = instr_map[src.scope]
        len_1 = 0
        len_7 = load_size & VALUE_127
        params = [len_1, len_7, sid]
        args = [concat_params(params, LOAD_SMASK_OFFSET_LIST,
                              LOAD_SMASK_SEGMENT_LIST)]
        # load_size, unit:2B
        smask_extent = load_size*2
        with self.new_scope():
            instr = tvm.call_extern("uint16", instr_name,
                                    dst.access_ptr("w", extent=smask_extent),
                                    src.access_ptr("r", extent=smask_extent),
                                    *args)
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", pipe_line)
            self.emit(instr, ONE_IR)

    @staticmethod
    def _get_channels(dst, arch_version, channel_pad_mode, input_format):
        """get channels"""
        channels = 32 // DTYPE_SIZE[dst.dtype]

        if (channel_pad_mode == 1) and arch_version == HI3796CV300ESAIC:
            channels = AIPP_INPUT_TYPE_SWAP_ALIGN.get(input_format).get(
                'channels')

        if (channel_pad_mode == 2) and \
                arch_version in (HI3796CV300ESAIC, AIC):
            channels = 4
        return channels

    def calculate_dst_extent_aipp(self, arch_version,  # pylint: disable=R0913, R0914
                                  input_format, dst, crop_horizontal_size,
                                  crop_vertical_size, channel_pad_mode,
                                  top_pad_rows, botton_pad_rows, left_pad_cols,
                                  right_pad_cols, scf_horizontal_size,
                                  scf_vertical_size, post_botton_clip_number,
                                  post_top_clip_number, post_right_clip_number,
                                  post_left_clip_number, dst_stride_pixel,
                                  raw_image_channel, raw_enable):
        """calculate dst extent"""
        # function's input params is too much, so disable them
        imm_input_format = Expr(input_format).eval_value()
        imm_channel_pad_mode = Expr(channel_pad_mode).eval_value()
        if any(value is None for value in [imm_input_format,
                                           imm_channel_pad_mode]):
            return dst.buffer_size

        channels = self._get_channels(dst, arch_version,
                                      imm_channel_pad_mode, imm_input_format)

        imm_top_pad_rows = Expr(top_pad_rows).eval_value()
        imm_botton_pad_rows = Expr(botton_pad_rows).eval_value()
        imm_left_pad_cols = Expr(left_pad_cols).eval_value()
        imm_right_pad_cols = Expr(right_pad_cols).eval_value()

        if any(value is None for value in [imm_top_pad_rows,
                                           imm_botton_pad_rows,
                                           imm_left_pad_cols,
                                           imm_right_pad_cols]):
            return dst.buffer_size

        if arch_version == HI3796CV300ESAIC:
            imm_scf_horizontal_size = Expr(scf_horizontal_size).eval_value()
            imm_scf_vertical_size = Expr(scf_vertical_size).eval_value()
            imm_post_right_clip_number = Expr(
                post_right_clip_number).eval_value()
            imm_post_left_clip_number = Expr(post_left_clip_number).eval_value()
            imm_post_botton_clip_number = Expr(
                post_botton_clip_number).eval_value()
            imm_post_top_clip_number = Expr(post_top_clip_number).eval_value()
            imm_dst_stride_pixel = Expr(dst_stride_pixel).eval_value()
            if any(value is None for value in [imm_scf_horizontal_size,
                                               imm_scf_vertical_size,
                                               imm_post_right_clip_number,
                                               imm_post_left_clip_number,
                                               imm_post_botton_clip_number,
                                               imm_post_top_clip_number,
                                               imm_dst_stride_pixel]):
                return dst.buffer_size
            if imm_dst_stride_pixel == 0:
                extent = (imm_scf_horizontal_size -
                          imm_post_right_clip_number -
                          imm_post_left_clip_number +
                          imm_left_pad_cols + imm_right_pad_cols)*\
                         (imm_scf_vertical_size - imm_post_botton_clip_number -
                          imm_post_top_clip_number + imm_top_pad_rows +
                          imm_botton_pad_rows)*channels
            else:
                extent = imm_dst_stride_pixel*(imm_scf_vertical_size -
                                               imm_post_botton_clip_number -
                                               imm_post_top_clip_number +
                                               imm_top_pad_rows +
                                               imm_botton_pad_rows)*channels
        else:
            imm_crop_horizontal_size = Expr(crop_horizontal_size).eval_value()
            imm_crop_vertical_size = Expr(crop_vertical_size).eval_value()
            if any(value is None for value in [imm_crop_horizontal_size,
                                               imm_crop_vertical_size]):
                return dst.buffer_size
            extent = (imm_crop_horizontal_size +
                      imm_left_pad_cols + imm_right_pad_cols)*(
                          imm_crop_vertical_size + imm_top_pad_rows +
                          imm_botton_pad_rows)*channels
            if arch_version == AIC and \
                    imm_input_format in [RAW16, RAW24] and \
                    raw_enable == 1:
                imm_raw_image_channel = Expr(raw_image_channel).eval_value()
                if imm_raw_image_channel is not None:
                    if imm_raw_image_channel == 0:
                        extent = extent // 4
                else:
                    return dst.buffer_size

        extent = extent*DTYPE_SIZE[dst.dtype]
        return extent

    @staticmethod
    def _handel_crop_info(arch_version, input_format,  # pylint: disable=R0913
                          crop_enbale, src_horizontal_size, src_vertical_size,
                          crop_info):
        # crop info
        if crop_enbale == AIPP_DISABLE:
            crop_horizontal_size = src_horizontal_size
            crop_vertical_size = src_vertical_size
            crop_horizontal_start = AIPP_INIT_VALUE
            crop_vertical_start = AIPP_INIT_VALUE
            single_line_mode = AIPP_INIT_VALUE
        else:
            check_dict_and_not_none(crop_info, 'crop_info')
            crop_horizontal_size = crop_info.get('dst_horizontal_size')
            crop_vertical_size = crop_info.get('dst_vertical_size')
            crop_horizontal_start = crop_info.get('crop_horizontal_start')
            crop_vertical_start = crop_info.get('crop_vertical_start')
            single_line_mode = crop_info.get('single_line_enable')

            _aipp_check_crop_info(input_format, single_line_mode,
                                  crop_horizontal_size, crop_vertical_size,
                                  crop_horizontal_start, crop_vertical_start)
            _aipp_check_crop_in_picture(src_horizontal_size, src_vertical_size,
                                        crop_horizontal_size,
                                        crop_vertical_size,
                                        crop_horizontal_start,
                                        crop_vertical_start)
            _aipp_check_crop_single_line_mode(arch_version, single_line_mode)
            _check_crop_vertical_size_by_single_line(arch_version,
                                                     crop_vertical_size,
                                                     single_line_mode)

        return crop_horizontal_size, crop_vertical_size, \
               crop_horizontal_start, crop_vertical_start, single_line_mode

    @staticmethod
    def _handle_csc_info(arch_version, csc_enable, csc_info):
        if csc_enable == AIPP_DISABLE:
            csc_matrix, csc_out_bias, csc_in_bias = \
                [[AIPP_INIT_VALUE, AIPP_INIT_VALUE, AIPP_INIT_VALUE],
                 [AIPP_INIT_VALUE, AIPP_INIT_VALUE, AIPP_INIT_VALUE],
                 [AIPP_INIT_VALUE, AIPP_INIT_VALUE, AIPP_INIT_VALUE]],\
                [AIPP_INIT_VALUE, AIPP_INIT_VALUE, AIPP_INIT_VALUE],\
                [AIPP_INIT_VALUE, AIPP_INIT_VALUE, AIPP_INIT_VALUE]
        else:
            check_dict_and_not_none(csc_info, 'csc_info')
            format_convert = csc_info.get('format_convert')
            _aipp_check_format_convert(arch_version, format_convert)
            csc_matrix, csc_out_bias, csc_in_bias = _get_csc_parameter(
                format_convert, csc_info)

        return csc_matrix, csc_out_bias, csc_in_bias

    @staticmethod
    def _handle_swap_info(input_format, swap_enable, swap_list):
        if swap_enable == AIPP_DISABLE:
            rb_swap, uv_swap, ax_swap = AIPP_INIT_VALUE, \
                                        AIPP_INIT_VALUE, AIPP_INIT_VALUE
        else:
            _check_list_type_and_range(swap_list, 3, (int, Scalar, Expr),
                                       range(0, 2), 'swap')
            rb_swap = swap_list[0]
            uv_swap = swap_list[1]
            ax_swap = swap_list[2]

            _aipp_check_swap(input_format, rb_swap, uv_swap, ax_swap)

        return rb_swap, uv_swap, ax_swap

    @staticmethod
    def _handle_pre_clip_info(pre_clip_enable, pre_clip_info,
                              crop_vertical_size):
        if pre_clip_enable == AIPP_DISABLE:
            pre_top_clip_number = AIPP_INIT_VALUE
            pre_botton_clip_number = AIPP_INIT_VALUE
        else:
            check_dict_and_not_none(pre_clip_info, 'pre_clip_info')
            pre_top_clip_number = pre_clip_info.get('pre_top_clip_number')
            pre_botton_clip_number = pre_clip_info.get(
                'pre_botton_clip_number')

            _aipp_check_pre_clip(pre_top_clip_number,
                                 pre_botton_clip_number, crop_vertical_size)
        return pre_top_clip_number, pre_botton_clip_number

    def _handle_scf_info(self, scf_enable,  # pylint: disable=R0913, R0914, R0915
                         scf_info, crop_horizontal_size, crop_vertical_size,
                         pre_top_clip_number, pre_botton_clip_number):
        if scf_enable == AIPP_DISABLE:
            scf_horizontal_size = crop_horizontal_size
            scf_vertical_size = crop_vertical_size - \
                                pre_top_clip_number - pre_botton_clip_number
            alpha_hori_scaling_mode = AIPP_INIT_VALUE
            hori_scaling_mode = AIPP_INIT_VALUE
            alpha_vert_scaling_mode = AIPP_INIT_VALUE
            vert_scaling_mode = AIPP_INIT_VALUE
            order_hori_vert_filter = AIPP_INIT_VALUE
            vertical_scaling_enable = AIPP_INIT_VALUE
            hori_scaling_enable = AIPP_INIT_VALUE
            vert_scaling = AIPP_INIT_VALUE
            hori_scaling = AIPP_INIT_VALUE
            init_vert_phase = AIPP_INIT_VALUE
            init_hori_phase = AIPP_INIT_VALUE

        else:
            check_dict_and_not_none(scf_info, 'scf_info')
            # scf_info
            scf_horizontal_size = scf_info.get('scf_horizontal_size')
            scf_vertical_size = scf_info.get('scf_vertical_size')
            scf_horizontal_start = scf_info.get('scf_horizontal_start')
            scf_vertical_start = scf_info.get('scf_vertical_start')
            scaling_mode = scf_info.get('scaling_mode')

            _aipp_check_scf(scf_horizontal_size, scf_vertical_size,
                            scf_horizontal_start, scf_vertical_start,
                            scaling_mode)

            # SPR12
            pre_scf_horizontal_size = crop_horizontal_size
            pre_scf_vertical_size = crop_vertical_size - \
                                    pre_botton_clip_number - \
                                    pre_top_clip_number

            # spr13
            alpha_hori_scaling_mode = scaling_mode
            hori_scaling_mode = scaling_mode
            alpha_vert_scaling_mode = scaling_mode
            vert_scaling_mode = scaling_mode

            vertical_scaling_enable = AIPP_ENABLE
            hori_scaling_enable = AIPP_ENABLE

            imm_scf_horizontal_size = Expr(scf_horizontal_size).eval_value()
            imm_pre_scf_horizontal_size = Expr(
                pre_scf_horizontal_size).eval_value()
            if imm_scf_horizontal_size is not None and \
                    imm_pre_scf_horizontal_size is not None:
                if imm_scf_horizontal_size > imm_pre_scf_horizontal_size:
                    order_hori_vert_filter = AIPP_ENABLE
                else:
                    order_hori_vert_filter = AIPP_DISABLE
            else:
                with self.context.freeze():  # pylint: disable=E1101
                    order_hori_vert_filter = self.Scalar_(  # pylint: disable=E1101
                        'uint16', 'order_hori_vert_filter', AIPP_ENABLE)

                    with self.if_scope(
                            scf_horizontal_size < pre_scf_horizontal_size):
                        order_hori_vert_filter.set_as(AIPP_DISABLE)

            # SPR16
            hori_scaling = scf_info.get(
                'scf_horizontal_scale',
                (pre_scf_horizontal_size - 1)*SCALE_COF//
                (scf_horizontal_size - 1)//4*4)
            TikCheckUtil.check_type_match(hori_scaling, (int, Scalar, Expr),
                                          'scf_horizontal_scale type error, '
                                          'input: '
                                          '{}'.format(type(hori_scaling)))
            check_scalar_dtype(hori_scaling,
                               "scf_horizontal_scale should be"
                               " a scalar of int/uint")

            vert_scaling = scf_info.get(
                'scf_vertical_scale',
                (pre_scf_vertical_size - 1)*SCALE_COF //
                (scf_vertical_size - 1) // 4*4)
            TikCheckUtil.check_type_match(vert_scaling, (int, Scalar, Expr),
                                          'scf_vertical_scale type error, '
                                          'input:'
                                          ' {}'.format(type(vert_scaling)))
            check_scalar_dtype(vert_scaling,
                               "scf_vertical_scale should be"
                               " a scalar of int/uint")

            init_vert_phase = scf_vertical_start
            init_hori_phase = scf_horizontal_start

        return scf_horizontal_size, scf_vertical_size, \
               alpha_hori_scaling_mode, hori_scaling_mode, \
               alpha_vert_scaling_mode, vert_scaling_mode, \
               order_hori_vert_filter, vertical_scaling_enable, \
               hori_scaling_enable, vert_scaling, hori_scaling, \
               init_vert_phase, init_hori_phase

    @staticmethod
    def _handle_post_clip(post_clip_enable, post_clip_info):
        if post_clip_enable == AIPP_DISABLE:
            post_botton_clip_number = AIPP_INIT_VALUE
            post_top_clip_number = AIPP_INIT_VALUE
            post_right_clip_number = AIPP_INIT_VALUE
            post_left_clip_number = AIPP_INIT_VALUE
        else:
            check_dict_and_not_none(post_clip_info, 'post_clip_info')
            post_botton_clip_number = post_clip_info.get(
                'post_botton_clip_number')
            post_top_clip_number = post_clip_info.get(
                'post_top_clip_number')
            post_right_clip_number = post_clip_info.get(
                'post_right_clip_number')
            post_left_clip_number = post_clip_info.get(
                'post_left_clip_number')

            _aipp_check_post_clip(post_botton_clip_number,
                                  post_top_clip_number,
                                  post_right_clip_number,
                                  post_left_clip_number)

        return post_botton_clip_number, post_top_clip_number, \
               post_right_clip_number, post_left_clip_number

    @staticmethod
    def _handle_dtc_info(dtc_enable, dtc_info):  # pylint: disable=R0914
        if dtc_enable == AIPP_DISABLE:
            dtc_mean_type = AIPP_INIT_VALUE
            dtc_mean0_uint32 = AIPP_INIT_VALUE
            dtc_mean1_uint32 = AIPP_INIT_VALUE
            dtc_mean2_uint32 = AIPP_INIT_VALUE
            dtc_mean3_uint32 = AIPP_INIT_VALUE
            dtc_min0_uint32 = float16format2uint16(AIPP_INIT_FLOAT_VALUE_ZERO)
            dtc_min1_uint32 = float16format2uint16(AIPP_INIT_FLOAT_VALUE_ZERO)
            dtc_min2_uint32 = float16format2uint16(AIPP_INIT_FLOAT_VALUE_ZERO)
            dtc_min3_uint32 = float16format2uint16(AIPP_INIT_FLOAT_VALUE_ZERO)
            dtc_var0_uint32 = float16format2uint16(AIPP_INIT_FLOAT_VALUE_ONE)
            dtc_var1_uint32 = float16format2uint16(AIPP_INIT_FLOAT_VALUE_ONE)
            dtc_var2_uint32 = float16format2uint16(AIPP_INIT_FLOAT_VALUE_ONE)
            dtc_var3_uint32 = float16format2uint16(AIPP_INIT_FLOAT_VALUE_ONE)
            raw_to_f16_n = AIPP_INIT_VALUE
        else:
            check_dict_and_not_none(dtc_info, 'dtc_info')
            dtc_mean_type = dtc_info.get('dtc_mean_type')
            dtc_mean = dtc_info.get('dtc_mean')
            _aipp_check_dtc_mean(dtc_mean_type, dtc_mean)
            dtc_mean0_uint32, dtc_mean1_uint32, \
            dtc_mean2_uint32, dtc_mean3_uint32 = _aipp_get_dtc_mean(dtc_mean)

            dtc_min = dtc_info.get('dtc_min')
            _check_list_type_and_range(dtc_min, 4, (float, Scalar, Expr), None,
                                       'dtc_min')

            dtc_min0_uint32, dtc_min1_uint32, \
            dtc_min2_uint32, dtc_min3_uint32 = _aipp_get_dtc_min_value(dtc_min)

            dtc_var = dtc_info.get('dtc_var')
            _check_list_type_and_range(dtc_var, 4, (float, Scalar, Expr), None,
                                       'dtc_var')

            dtc_var0_uint32, dtc_var1_uint32, \
            dtc_var2_uint32, dtc_var3_uint32 = _aipp_get_dtc_var_value(dtc_var)

            raw_to_f16_n = dtc_info.get('raw_to_f16_n')
            _aipp_check_dtc_raw_info(raw_to_f16_n)

        return dtc_mean_type, dtc_mean0_uint32, dtc_mean1_uint32,\
               dtc_mean2_uint32, dtc_mean3_uint32, dtc_min0_uint32,\
               dtc_min1_uint32, dtc_min2_uint32, dtc_min3_uint32,\
               dtc_var0_uint32, dtc_var1_uint32, dtc_var2_uint32,\
               dtc_var3_uint32, raw_to_f16_n

    @staticmethod
    def _handle_flip_mode(arch_version, flip_enable, flip_mode):
        if flip_enable == AIPP_DISABLE:
            flip_mode = 0
        else:
            _aipp_check_flip_dict(arch_version, flip_mode)
        return flip_mode

    @staticmethod
    def _handle_channel_pad_info(arch_version, channel_pad_enable,
                                 channel_pad_info, dst):
        if channel_pad_enable == AIPP_DISABLE:
            channel_pad_mode = 0
            channel_pad_value_uint32 = 0
        else:
            check_dict_and_not_none(channel_pad_info, 'channel_pad_info')
            channel_pad_mode = channel_pad_info.get('channel_pad_mode')
            channel_pad_value = channel_pad_info.get('channel_pad_value')
            _aipp_check_cpad(arch_version, dst, channel_pad_value,
                             channel_pad_mode)

            if dst.dtype == 'float16':
                if Expr(channel_pad_value).eval_value() is None:
                    channel_pad_value_uint32 = \
                        channel_pad_value.reinterpret_cast_to('uint16')
                else:
                    channel_pad_value_uint32 = float16format2uint16(
                        channel_pad_value)
            else:
                channel_pad_value_uint32 = channel_pad_value

        return channel_pad_mode, channel_pad_value_uint32

    @staticmethod
    def _handle_area_pad_info(arch_version, input_format,  # pylint: disable=R0914
                              area_pad_enable, area_pad_info, dst):
        if area_pad_enable == AIPP_DISABLE:
            area_pad_mode = AIPP_INIT_VALUE
            top_pad_rows = AIPP_INIT_VALUE
            left_pad_cols = AIPP_INIT_VALUE
            right_pad_cols = AIPP_INIT_VALUE
            botton_pad_rows = AIPP_INIT_VALUE
            channel0_pad_value_uint32 = AIPP_INIT_VALUE
            channel1_pad_value_uint32 = AIPP_INIT_VALUE
            channel2_pad_value_uint32 = AIPP_INIT_VALUE
            channel3_pad_value_uint32 = AIPP_INIT_VALUE
        else:

            check_dict_and_not_none(area_pad_info, 'area_pad_info')
            area_pad_mode = area_pad_info.get('area_pad_mode')

            # area pad value
            top_pad_rows = area_pad_info.get('top_pad_rows')
            botton_pad_rows = area_pad_info.get('botton_pad_rows')
            left_pad_cols = area_pad_info.get('left_pad_cols')
            right_pad_cols = area_pad_info.get('right_pad_cols')

            # channel pad value
            channel0_pad_value = area_pad_info.get('channel0_pad_value')
            channel1_pad_value = area_pad_info.get('channel1_pad_value')
            channel2_pad_value = area_pad_info.get('channel2_pad_value')
            channel3_pad_value = area_pad_info.get('channel3_pad_value')

            _aipp_check_area_pad(input_format, dst, area_pad_mode, top_pad_rows,
                                 botton_pad_rows, left_pad_cols, right_pad_cols,
                                 [channel0_pad_value, channel1_pad_value,
                                  channel2_pad_value, channel3_pad_value],
                                 arch_version)

            channel0_pad_value_uint32, channel1_pad_value_uint32, \
            channel2_pad_value_uint32, channel3_pad_value_uint32 = \
                _aipp_get_channel_pad_value(
                    dst.dtype, channel0_pad_value, channel1_pad_value,
                    channel2_pad_value, channel3_pad_value)

        return area_pad_mode, top_pad_rows, left_pad_cols, right_pad_cols, \
               botton_pad_rows, channel0_pad_value_uint32, \
               channel1_pad_value_uint32, channel2_pad_value_uint32, \
               channel3_pad_value_uint32

    @staticmethod
    def _handle_stretch_info(stretch_enable, stretch_info):
        if stretch_enable == AIPP_DISABLE:
            dst_stride_pixel = AIPP_INIT_VALUE
        else:
            check_dict_and_not_none(stretch_info, 'stretch_info')
            dst_stride_pixel = stretch_info.get('dst_stride_pixel')
            _aipp_check_stretch(dst_stride_pixel)
        return dst_stride_pixel

    @staticmethod
    def _handle_raw_info(raw_enable, raw_info):
        if raw_enable == AIPP_DISABLE:
            raw_image_channel = AIPP_INIT_VALUE
            raw_start_channel = AIPP_INIT_VALUE
        else:
            check_dict_and_not_none(raw_info, 'raw_info')
            raw_image_channel = raw_info.get('raw_image_channel')
            raw_start_channel = raw_info.get('raw_start_channel')
            _aipp_check_raw_info(raw_image_channel, raw_start_channel)

        return raw_image_channel, raw_start_channel

    @source_info_decorator()
    @debug.load_image_decorator
    def load_image(self, dst, src0, src1,  # pylint: disable=R0913, R0914, R0915
                   input_format, function_switch, src_info, crop_info,
                   pre_clip_info, swap_list, csc_info, scf_info, post_clip_info,
                   dtc_info, flip_mode, channel_pad_info, area_pad_info,
                   stretch_info, raw_info, sid):
        """load image api"""
        # function's input params is too much, so disable them
        arch_version = get_soc_name() + get_soc_core_type()
        # check arch_version
        _aipp_check_arch_version(arch_version)

        # check input format
        _aipp_check_input_format(arch_version, input_format)

        # check dst format by input format
        _aipp_check_dst(input_format, dst)

        # src_info
        check_dict_and_not_none(src_info, 'src_info')
        src_horizontal_size = src_info.get('src_horizontal_size')
        src_vertical_size = src_info.get('src_vertical_size')
        _aipp_check_src_info(arch_version, input_format, src_horizontal_size,
                             src_vertical_size)
        _aipp_check_src(input_format, src0, src1, src_horizontal_size,
                        src_vertical_size)

        # enable switch
        crop_enbale, swap_enable, csc_enable, dtc_enable, area_pad_enable, \
        channel_pad_enable, pre_clip_enable, scf_enable, post_clip_enable, \
        flip_enable, stretch_enable, raw_enable = aipp_get_enable_bit(
            arch_version, function_switch)
        _aipp_check_function_switch(arch_version, input_format,
                                    swap_enable, csc_enable, dtc_enable,
                                    area_pad_enable, pre_clip_enable,
                                    scf_enable, post_clip_enable, flip_enable,
                                    stretch_enable, raw_enable)

        # crop info check
        crop_horizontal_size, crop_vertical_size, crop_horizontal_start,\
        crop_vertical_start, single_line_mode = self._handel_crop_info(
            arch_version, input_format, crop_enbale,
            src_horizontal_size, src_vertical_size, crop_info)

        # csc info
        csc_matrix, csc_out_bias, csc_in_bias = self._handle_csc_info(
            arch_version, csc_enable, csc_info)

        # swap list check
        rb_swap, uv_swap, ax_swap = self._handle_swap_info(
            input_format, swap_enable, swap_list)

        if arch_version == HI3796CV300ESAIC:

            # pre clip info
            pre_top_clip_number, pre_botton_clip_number =\
                self._handle_pre_clip_info(pre_clip_enable, pre_clip_info,
                                           crop_vertical_size)

            # scf info
            scf_horizontal_size, scf_vertical_size, alpha_hori_scaling_mode,\
            hori_scaling_mode, alpha_vert_scaling_mode, vert_scaling_mode,\
            order_hori_vert_filter, vertical_scaling_enable,\
            hori_scaling_enable, vert_scaling, hori_scaling, init_vert_phase,\
            init_hori_phase = self._handle_scf_info(
                scf_enable, scf_info, crop_horizontal_size, crop_vertical_size,
                pre_top_clip_number, pre_botton_clip_number)

            # post_clip_info
            post_botton_clip_number, post_top_clip_number,\
            post_right_clip_number, post_left_clip_number =\
                self._handle_post_clip(post_clip_enable, post_clip_info)

        else:
            pre_top_clip_number = AIPP_INIT_VALUE
            pre_botton_clip_number = AIPP_INIT_VALUE
            scf_horizontal_size = AIPP_INIT_VALUE
            scf_vertical_size = AIPP_INIT_VALUE
            alpha_hori_scaling_mode = AIPP_INIT_VALUE
            hori_scaling_mode = AIPP_INIT_VALUE
            alpha_vert_scaling_mode = AIPP_INIT_VALUE
            vert_scaling_mode = AIPP_INIT_VALUE
            order_hori_vert_filter = AIPP_INIT_VALUE
            vertical_scaling_enable = AIPP_INIT_VALUE
            hori_scaling_enable = AIPP_INIT_VALUE
            init_vert_phase = AIPP_INIT_VALUE
            init_hori_phase = AIPP_INIT_VALUE
            vert_scaling = AIPP_INIT_VALUE
            hori_scaling = AIPP_INIT_VALUE
            post_botton_clip_number = AIPP_INIT_VALUE
            post_top_clip_number = AIPP_INIT_VALUE
            post_right_clip_number = AIPP_INIT_VALUE
            post_left_clip_number = AIPP_INIT_VALUE

        # dtc_info
        dtc_mean_type, dtc_mean0_uint32, dtc_mean1_uint32, dtc_mean2_uint32,\
        dtc_mean3_uint32, dtc_min0_uint32, dtc_min1_uint32, dtc_min2_uint32,\
        dtc_min3_uint32, dtc_var0_uint32, dtc_var1_uint32, dtc_var2_uint32,\
        dtc_var3_uint32, raw_to_f16_n = self._handle_dtc_info(
            dtc_enable, dtc_info)

        # flip_mode
        flip_mode = self._handle_flip_mode(arch_version, flip_enable, flip_mode)

        # channel_pad_info
        channel_pad_mode, channel_pad_value_uint32 =\
            self._handle_channel_pad_info(arch_version, channel_pad_enable,
                                          channel_pad_info, dst)

        # area_pad_info
        area_pad_mode, top_pad_rows, left_pad_cols, right_pad_cols, \
        botton_pad_rows, channel0_pad_value_uint32, \
        channel1_pad_value_uint32, channel2_pad_value_uint32, \
        channel3_pad_value_uint32 = self._handle_area_pad_info(
            arch_version, input_format, area_pad_enable, area_pad_info, dst)

        # stretch_info
        dst_stride_pixel = self._handle_stretch_info(stretch_enable,
                                                     stretch_info)

        # raw handle
        raw_image_channel, raw_start_channel = self._handle_raw_info(raw_enable,
                                                                     raw_info)

        # sid
        _aipp_check_sid(arch_version, sid)

        dst_extent = self.calculate_dst_extent_aipp(arch_version, input_format,
                                                    dst, crop_horizontal_size,
                                                    crop_vertical_size,
                                                    channel_pad_mode,
                                                    top_pad_rows,
                                                    botton_pad_rows,
                                                    left_pad_cols,
                                                    right_pad_cols,
                                                    scf_horizontal_size,
                                                    scf_vertical_size,
                                                    post_botton_clip_number,
                                                    post_top_clip_number,
                                                    post_right_clip_number,
                                                    post_left_clip_number,
                                                    dst_stride_pixel,
                                                    raw_image_channel,
                                                    raw_enable)

        imm_dst_extent = Expr(dst_extent).eval_value()
        if imm_dst_extent is not None:
            TikCheckUtil.check_le(
                imm_dst_extent, dst.buffer_size,
                "output out of dst size, input: {}".format(imm_dst_extent))
            TikCheckUtil.check_equality(
                imm_dst_extent % 32, 0,
                "output should be 32 align, input: {}".format(imm_dst_extent))

        # handle
        self._aipp(
            dst, src0, src1, input_format, src_horizontal_size,
            src_vertical_size, crop_horizontal_size, crop_vertical_size,
            crop_horizontal_start, crop_vertical_start, single_line_mode,
            pre_top_clip_number, pre_botton_clip_number, csc_matrix,
            csc_out_bias, csc_in_bias, rb_swap, uv_swap, ax_swap,
            scf_horizontal_size, scf_vertical_size,
            alpha_hori_scaling_mode, hori_scaling_mode,
            alpha_vert_scaling_mode, vert_scaling_mode,
            order_hori_vert_filter,
            vertical_scaling_enable, hori_scaling_enable, init_vert_phase,
            init_hori_phase, vert_scaling, hori_scaling,
            post_botton_clip_number, post_top_clip_number,
            post_right_clip_number, post_left_clip_number,
            dtc_mean_type, dtc_mean0_uint32, dtc_mean1_uint32,
            dtc_mean2_uint32, dtc_mean3_uint32, dtc_min0_uint32,
            dtc_min1_uint32, dtc_min2_uint32, dtc_min3_uint32,
            dtc_var0_uint32, dtc_var1_uint32, dtc_var2_uint32,
            dtc_var3_uint32, raw_to_f16_n, flip_mode, channel_pad_mode,
            channel_pad_value_uint32, area_pad_mode,
            top_pad_rows, botton_pad_rows, left_pad_cols, right_pad_cols,
            channel0_pad_value_uint32, channel1_pad_value_uint32,
            channel2_pad_value_uint32, channel3_pad_value_uint32,
            dst_stride_pixel, raw_image_channel, raw_start_channel, sid,
            arch_version, csc_enable, post_clip_enable, dst_extent)

    def _set_aipp_spr0(self, arch_version,
                       dtc_mean0_uint32, dtc_mean1_uint32, src0):
        """set spr0"""
        if arch_version == AIC:
            sfr_dtc_pixel_mean_ch0 = Expr(dtc_mean0_uint32 // BIT_16)
            sfr_dtc_pixel_mean_ch1 = Expr(dtc_mean1_uint32 // BIT_16)
        else:
            sfr_dtc_pixel_mean_ch0 = AIPP_INIT_VALUE
            sfr_dtc_pixel_mean_ch1 = AIPP_INIT_VALUE

        with self.context.freeze():  # pylint: disable=E1101
            scalar_addr_y = self.Scalar_(dtype='uint64',  # pylint: disable=E1101
                                         name='scalar_addr_y', init_value=0)

            scalar_addr_y.set_as(tvm.expr.Cast("uint64",
                                               tvm.call_extern("handle", "",
                                                               src0.access_ptr(
                                                                   "r"))))
            aipp0_config = [scalar_addr_y, sfr_dtc_pixel_mean_ch0,
                            sfr_dtc_pixel_mean_ch1]

            aipp0_register = concat_params(aipp0_config, AIPP0_OFFSET_LIST,
                                           AIPP0_SEGMENT_LIST, dtype="uint64")

            with self.new_scope():
                aipp_spr0 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr0.set_as(aipp0_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_0",
                                          aipp_spr0.get()))
        return scalar_addr_y

    def _set_aipp_spr1(self, input_format, csc_enable, src1,  #pylint: disable=R0913
                       scalar_addr_y, src_horizontal_size, src_vertical_size):
        """set spr1"""
        with self.context.freeze():  # pylint: disable=E1101
            scalar_addr_uv = self.Scalar_(dtype='uint64',  # pylint: disable=E1101
                                          name='scalar_addr_uv', init_value=0)

            if src1 is not None:
                scalar_addr_uv.set_as(
                    tvm.expr.Cast("uint64",
                                  tvm.call_extern(
                                      "handle", "", src1.access_ptr("r"))))
            else:
                imm_input_format = Expr(input_format).eval_value()
                if imm_input_format is None:
                    src_channel_scalar = self.Scalar_('uint32',  # pylint: disable=E1101
                                                      'src_channel_scalar', 0)
                    with self.if_scope(input_format == 0):
                        src_channel_scalar.set_as(
                            1*src_horizontal_size*src_vertical_size)
                        scalar_addr_uv.set_as(
                            scalar_addr_y + src_channel_scalar)
                    with self.if_scope(input_format == 2):
                        src_channel_scalar.set_as(
                            8*src_horizontal_size*src_vertical_size)
                        scalar_addr_uv.set_as(
                            scalar_addr_y + src_channel_scalar)
                    with self.if_scope(input_format == 3):
                        src_channel_scalar.set_as(
                            4*src_horizontal_size*src_vertical_size)
                        scalar_addr_uv.set_as(
                            scalar_addr_y + src_channel_scalar)
                    with self.if_scope(input_format == 7):
                        src_channel_scalar.set_as(
                            1*src_horizontal_size*src_vertical_size)
                        scalar_addr_uv.set_as(
                            scalar_addr_y + src_channel_scalar)

                else:
                    if input_format in (0, 2, 3, 7):
                        src_channel = AIPP_INPUT_TYPE_SWAP_ALIGN.get(
                            input_format).get('src0_size_bytes')
                        scalar_addr_uv.set_as(
                            scalar_addr_y +
                            src_horizontal_size*src_vertical_size*src_channel)

            aipp1_config = [scalar_addr_uv, csc_enable]

            aipp1_register = concat_params(aipp1_config, AIPP1_OFFSET_LIST,
                                           AIPP1_SEGMENT_LIST, dtype="uint64")
            with self.new_scope():
                aipp_spr1 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr1.set_as(aipp1_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_1",
                                          aipp_spr1.get()))

    def _set_aipp_spr2(self, csc_matrix):
        """set spr2"""
        aipp2_config = [csc_matrix[0][0], csc_matrix[0][1], csc_matrix[0][2],
                        csc_matrix[1][0]]

        aipp2_register = concat_params(aipp2_config, AIPP2_OFFSET_LIST,
                                       AIPP2_SEGMENT_LIST, dtype="uint64")

        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                aipp_spr2 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr2.set_as(aipp2_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_2",
                                          aipp_spr2.get()))

    def _set_aipp_spr3(self, csc_matrix):
        """set spr3"""
        aipp3_config = [csc_matrix[1][1], csc_matrix[1][2], csc_matrix[2][0],
                        csc_matrix[2][1]]

        aipp3_register = concat_params(aipp3_config, AIPP3_OFFSET_LIST,
                                       AIPP3_SEGMENT_LIST, dtype="uint64")

        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                aipp_spr3 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr3.set_as(aipp3_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_3",
                                          aipp_spr3.get()))

    def _set_aipp_spr4(self, csc_matrix, csc_out_bias, csc_in_bias):
        """set spr4"""
        aipp4_config = [csc_matrix[2][2], csc_out_bias[0], csc_out_bias[1],
                        csc_out_bias[2], csc_in_bias[0], csc_in_bias[1],
                        csc_in_bias[2]]

        aipp4_register = concat_params(aipp4_config, AIPP4_OFFSET_LIST,
                                       AIPP4_SEGMENT_LIST, dtype="uint64")

        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                aipp_spr4 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr4.set_as(aipp4_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_4",
                                          aipp_spr4.get()))

    def _set_aipp_spr5(self, dtc_mean0_uint32, dtc_mean1_uint32,
                       dtc_mean2_uint32, dtc_mean3_uint32):
        """set spr5"""
        aipp5_config = [dtc_mean0_uint32 % BIT_16, dtc_mean1_uint32 % BIT_16,
                        dtc_mean2_uint32 % BIT_16, dtc_mean3_uint32 % BIT_16]

        aipp5_register = concat_params(aipp5_config, AIPP5_OFFSET_LIST,
                                       AIPP5_SEGMENT_LIST, dtype="uint64")

        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                aipp_spr5 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr5.set_as(aipp5_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_5",
                                          aipp_spr5.get()))

    def _set_aipp_spr6(self, dtc_min0_uint32, dtc_min1_uint32, dtc_min2_uint32,
                       dtc_min3_uint32):
        """set spr6"""
        aipp6_config = [dtc_min0_uint32, dtc_min1_uint32, dtc_min2_uint32,
                        dtc_min3_uint32]
        aipp6_register = concat_params(aipp6_config, AIPP6_OFFSET_LIST,
                                       AIPP6_SEGMENT_LIST, dtype="uint64")

        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                aipp_spr6 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr6.set_as(aipp6_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_6",
                                          aipp_spr6.get()))

    def _set_aipp_spr7(self, dtc_var0_uint32, dtc_var1_uint32, dtc_var2_uint32,
                       dtc_var3_uint32):
        """set spr7"""
        aipp7_config = [dtc_var0_uint32, dtc_var1_uint32, dtc_var2_uint32,
                        dtc_var3_uint32]

        aipp7_register = concat_params(aipp7_config, AIPP7_OFFSET_LIST,
                                       AIPP7_SEGMENT_LIST, dtype="uint64")

        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                aipp_spr7 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr7.set_as(aipp7_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_7",
                                          aipp_spr7.get()))

    def _set_aipp_spr8(self, channel0_pad_value_uint32,
                       channel1_pad_value_uint32, channel2_pad_value_uint32,
                       channel3_pad_value_uint32):
        """set spr8"""
        aipp8_config = [channel0_pad_value_uint32, channel1_pad_value_uint32,
                        channel2_pad_value_uint32, channel3_pad_value_uint32]

        aipp8_register = concat_params(aipp8_config, AIPP8_OFFSET_LIST,
                                       AIPP8_SEGMENT_LIST, dtype="uint64")

        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                aipp_spr8 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr8.set_as(aipp8_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_8",
                                          aipp_spr8.get()))

    def _set_aipp_spr9(self, arch_version,  # pylint: disable=R0913, R0914
                       dtc_mean2_uint32, dtc_mean3_uint32,
                       channel_pad_mode, flip_mode, channel_pad_value_uint32,
                       rb_swap, uv_swap, ax_swap, input_format,
                       single_line_mode, area_pad_mode, raw_to_f16_n,
                       dtc_mean_type, raw_image_channel, raw_start_channel):
        """set spr9"""
        if arch_version == AIC:
            sfr_dtc_pixel_mean_ch2 = dtc_mean2_uint32 // BIT_16
            sfr_dtc_pixel_mean_ch3 = dtc_mean3_uint32 // BIT_16
        else:
            sfr_dtc_pixel_mean_ch2 = 0
            sfr_dtc_pixel_mean_ch3 = 0
        no_padding = channel_pad_mode & 1
        padd_4channels = channel_pad_mode // 2
        horizontal_flip_enable = flip_mode & 1
        vertical_flip_enable = flip_mode // 2
        aipp9_config = [channel_pad_value_uint32, rb_swap, uv_swap, ax_swap,
                        input_format, single_line_mode, horizontal_flip_enable,
                        vertical_flip_enable, area_pad_mode, no_padding,
                        raw_to_f16_n, dtc_mean_type, raw_image_channel,
                        raw_start_channel, padd_4channels,
                        sfr_dtc_pixel_mean_ch2, sfr_dtc_pixel_mean_ch3]

        aipp9_register = concat_params(aipp9_config, AIPP9_OFFSET_LIST,
                                       AIPP9_SEGMENT_LIST, dtype="uint64")

        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                aipp_spr9 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr9.set_as(aipp9_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_9",
                                          aipp_spr9.get()))

    def _set_aipp_spr10(self, dst_stride_pixel):
        """set spr10"""
        aipp10_config = [dst_stride_pixel]
        aipp10_register = concat_params(aipp10_config, AIPP10_OFFSET_LIST,
                                        AIPP10_SEGMENT_LIST, dtype="uint64")
        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                aipp_spr10 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr10.set_as(aipp10_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_10",
                                          aipp_spr10.get()))

    def _set_aipp_spr11(self, pre_botton_clip_number, pre_top_clip_number):
        """set spr11"""
        aipp11_config = [pre_botton_clip_number, pre_top_clip_number]
        aipp11_register = concat_params(aipp11_config, AIPP11_OFFSET_LIST,
                                        AIPP11_SEGMENT_LIST, dtype="uint64")
        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                aipp_spr11 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr11.set_as(aipp11_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_11",
                                          aipp_spr11.get()))

    def _set_aipp_spr12(self, scf_vertical_size, scf_horizontal_size):
        """set spr12"""
        aipp12_config = [scf_vertical_size - 1, scf_horizontal_size - 1]
        aipp12_register = concat_params(aipp12_config, AIPP12_OFFSET_LIST,
                                        AIPP12_SEGMENT_LIST, dtype="uint64")

        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                aipp_spr12 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr12.set_as(aipp12_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_12",
                                          aipp_spr12.get()))

    def _set_aipp_spr13(self, hori_scaling_enable,  # pylint: disable=R0913
                        vertical_scaling_enable, order_hori_vert_filter,
                        vert_scaling_mode, alpha_vert_scaling_mode,
                        hori_scaling_mode, alpha_hori_scaling_mode):
        """set spr13"""
        aipp13_config = [hori_scaling_enable, vertical_scaling_enable,
                         order_hori_vert_filter, vert_scaling_mode,
                         alpha_vert_scaling_mode, hori_scaling_mode,
                         alpha_hori_scaling_mode]

        aipp13_register = concat_params(aipp13_config, AIPP13_OFFSET_LIST,
                                        AIPP13_SEGMENT_LIST, dtype="uint64")

        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                aipp_spr13 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr13.set_as(aipp13_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_13",
                                          aipp_spr13.get()))

    def _set_aipp_spr15(self, init_vert_phase, init_hori_phase):
        """set spr15"""
        aipp15_config = [init_vert_phase, init_hori_phase]

        aipp15_register = concat_params(aipp15_config, AIPP15_OFFSET_LIST,
                                        AIPP15_SEGMENT_LIST, dtype="uint64")

        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                aipp_spr15 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr15.set_as(aipp15_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_15",
                                          aipp_spr15.get()))

    def _set_aipp_spr16(self, vert_scaling, hori_scaling):
        """set spr16"""
        aipp16_config = [vert_scaling, hori_scaling]

        aipp16_register = concat_params(aipp16_config, AIPP16_OFFSET_LIST,
                                        AIPP16_SEGMENT_LIST, dtype="uint64")

        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                aipp_spr16 = self.Scalar_(dtype="uint64") # pylint: disable=E1101
                aipp_spr16.set_as(aipp16_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_16",
                                          aipp_spr16.get()))

    def _set_aipp_spr17(self, post_botton_clip_number,  # pylint: disable=R0913
                        post_top_clip_number, post_right_clip_number,
                        post_left_clip_number, post_clip_enable):
        """set spr17"""
        aipp17_config = [post_botton_clip_number, post_top_clip_number,
                         post_right_clip_number, post_left_clip_number,
                         post_clip_enable]

        aipp17_register = concat_params(aipp17_config, AIPP17_OFFSET_LIST,
                                        AIPP17_SEGMENT_LIST, dtype="uint64")

        with self.context.freeze():  # pylint: disable=E1101
            with self.new_scope():
                aipp_spr17 = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
                aipp_spr17.set_as(aipp17_register)
                self.emit(tvm.call_extern("uint64", "set_aipp_spr_17",
                                          aipp_spr17.get()))

    def _aipp(self, dst, src0, src1,  # pylint: disable=R0913, R0914
              input_format, src_horizontal_size, src_vertical_size,
              horizontal_size, vertical_size, horizontal_start,
              vertical_start, single_line_mode, pre_top_clip_number,
              pre_botton_clip_number, csc_matrix, csc_out_bias,
              csc_in_bias, rb_swap, uv_swap, ax_swap,
              scf_horizontal_size, scf_vertical_size,
              alpha_hori_scaling_mode, hori_scaling_mode,
              alpha_vert_scaling_mode, vert_scaling_mode,
              order_hori_vert_filter,
              vertical_scaling_enable, hori_scaling_enable, init_vert_phase,
              init_hori_phase, vert_scaling, hori_scaling,
              post_botton_clip_number, post_top_clip_number,
              post_right_clip_number, post_left_clip_number,
              dtc_mean_type, dtc_mean0_uint32, dtc_mean1_uint32,
              dtc_mean2_uint32, dtc_mean3_uint32, dtc_min0_uint32,
              dtc_min1_uint32, dtc_min2_uint32, dtc_min3_uint32,
              dtc_var0_uint32, dtc_var1_uint32, dtc_var2_uint32,
              dtc_var3_uint32, raw_to_f16_n, flip_mode, channel_pad_mode,
              channel_pad_value_uint32, area_pad_mode,
              top_pad_rows, botton_pad_rows, left_pad_cols, right_pad_cols,
              channel0_pad_value_uint32, channel1_pad_value_uint32,
              channel2_pad_value_uint32, channel3_pad_value_uint32,
              dst_stride_pixel, raw_image_channel, raw_start_channel, sid,
              arch_version, csc_enable, post_clip_enable, dst_extent):
        """aipp function"""
        # function's input params is too much, so disable them

        # set aipp spr
        # aipp SPR0
        scalar_addr_y = self._set_aipp_spr0(arch_version, dtc_mean0_uint32,
                                            dtc_mean1_uint32, src0)

        # aipp SPR1
        self._set_aipp_spr1(input_format, csc_enable, src1, scalar_addr_y,
                            src_horizontal_size, src_vertical_size)

        # aipp SPR2
        self._set_aipp_spr2(csc_matrix)

        # aipp SPR3
        self._set_aipp_spr3(csc_matrix)

        # aipp SPR4
        self._set_aipp_spr4(csc_matrix, csc_out_bias, csc_in_bias)

        # aipp SPR5
        self._set_aipp_spr5(dtc_mean0_uint32, dtc_mean1_uint32,
                            dtc_mean2_uint32, dtc_mean3_uint32)

        # aipp SPR6
        self._set_aipp_spr6(dtc_min0_uint32, dtc_min1_uint32, dtc_min2_uint32,
                            dtc_min3_uint32)

        # aipp SPR7
        self._set_aipp_spr7(dtc_var0_uint32, dtc_var1_uint32, dtc_var2_uint32,
                            dtc_var3_uint32)

        # padding_mode:0 set
        # aipp SPR8 (padding_mode)
        self._set_aipp_spr8(channel0_pad_value_uint32,
                            channel1_pad_value_uint32,
                            channel2_pad_value_uint32,
                            channel3_pad_value_uint32)

        # aipp SPR9
        self._set_aipp_spr9(arch_version, dtc_mean2_uint32, dtc_mean3_uint32,
                            channel_pad_mode, flip_mode,
                            channel_pad_value_uint32, rb_swap, uv_swap, ax_swap,
                            input_format, single_line_mode, area_pad_mode,
                            raw_to_f16_n, dtc_mean_type, raw_image_channel,
                            raw_start_channel)

        if arch_version == HI3796CV300ESAIC:
            # aipp SPR10
            self._set_aipp_spr10(dst_stride_pixel)

            # aipp SPR11
            self._set_aipp_spr11(pre_botton_clip_number, pre_top_clip_number)

            # aipp SPR12
            self._set_aipp_spr12(scf_vertical_size, scf_horizontal_size)

            # aipp SPR13
            self._set_aipp_spr13(hori_scaling_enable, vertical_scaling_enable,
                                 order_hori_vert_filter, vert_scaling_mode,
                                 alpha_vert_scaling_mode, hori_scaling_mode,
                                 alpha_hori_scaling_mode)

            # aipp SPR15
            self._set_aipp_spr15(init_vert_phase, init_hori_phase)

            # aipp SPR16
            self._set_aipp_spr16(vert_scaling, hori_scaling)

            # aipp SPR17
            self._set_aipp_spr17(post_botton_clip_number, post_top_clip_number,
                                 post_right_clip_number, post_left_clip_number,
                                 post_clip_enable)

        # set aipp info
        # xs info
        xs_config = [horizontal_size - 1, vertical_size - 1, horizontal_start,
                     vertical_start]

        # xt info
        xt_config = [src_horizontal_size - 1, top_pad_rows, botton_pad_rows,
                     left_pad_cols, right_pad_cols, sid]

        # config
        load_image_config = xs_config + xt_config

        with self.new_scope():
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_MTE2)

            if src1 is not None:
                self.scope_attr(cce_params.CCE_AXIS, "append_mem",
                                tvm.call_extern("", "mem_vector",
                                                src0.access_ptr("r"),
                                                src1.access_ptr("r")))
            else:
                self.scope_attr(cce_params.CCE_AXIS, "append_mem",
                                tvm.call_extern("", "mem_vector",
                                                src0.access_ptr("r")))

            instr = tvm.call_extern(dst.dtype, "load_image_to_cbuf",
                                    dst.access_ptr("w", extent=dst_extent),
                                    *type_convert(load_image_config))
            self.emit(instr)

    @source_info_decorator()
    @debug.winograd_fm_transform_decorator
    def winograd_feature_map_transform( # pylint: disable=R0913, R0914
            self, dst, src, l1_h, l1_w, l1_c, pad_left, pad_right, pad_top,
            pad_bottom, m_extension, m_start_pt, k_extension, k_start_pt,
            column_indicator, dst_stride):
        """ load input feature map from L1 to L0A and do partial winograd
        transform on-the-fly

        Parameters
        ----------
        dst: destination operator, scope_cbuf
        src: src operator, scope_ca
        l1_h: height of input feature_map
        l1_w: width of input feature_map
        l1_c: channels of input feature_map
        pad_left: col nums of padding left
        pad_right: col nums of padding left
        pad_top: row nums of padding top
        pad_bottom: row nums of padding bottom
        m_extension: m direction extension steps from the start position
        m_start_pt: m direction start position of the feature matrix
        k_extension: k direction extension steps from the start position
        k_start_pt: k direction start position of the feature matrix
        column_indicator: partial weight matrix indicator
                          0: the 1st column
                          1: the 2nd column
                          2: the 3rd column
                          3: the 4th column
        dst_stride: inner destination gap between 4 generated expansion feature
                    maps in terms of fractal matrix(512B)

        Returns
        -------
        None
        """
        arch_version_dst_src_map = {
            HI3796CV300ESAIC: ['s8s8'],
            HI3796CV300CSAIC: ['s8s8', 'u8u8'],
            AIC: ['s8s8', 'u8u8', 'f16f16']
        }
        arch_version = get_soc_name() + get_soc_core_type()
        # check instruction
        TikCheckUtil.check_in_range(
            arch_version, arch_version_dst_src_map.keys(),
            "input core_arch: {} doesn't support "
            "winograd_feature_map_transform.".format(arch_version))
        # check scope
        TikCheckUtil.check_equality(
            dst.scope, "local.L0A",
            "dst scope should be l0a, input scope: {}".format(dst.scope))
        TikCheckUtil.check_equality(src.scope, "local.L1",
                                    "src scope should be l1, input scope: {}"
                                    .format(src.scope))
        # check dtype
        dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src.dtype]
        TikCheckUtil.check_in_range(
            dtype_str, arch_version_dst_src_map[arch_version],
            "{} winograd_feature_map_transform doesn't support from {} to {}"
            .format(arch_version, src.dtype, dst.dtype))
        # check m & k
        TikCheckUtil.check_type_match(
            m_extension, (int, Scalar, Expr),
            "m_extension should be int, Scalar, Expr, input type of m_extension"
            ": {}".format(type(m_extension)))
        check_scalar_dtype(m_extension,
                           "scalar_m_extension should be a scalar of int/uint")
        TikCheckUtil.check_type_match(
            m_start_pt, (int, Scalar, Expr),
            "m_start_pt should be int, Scalar, Expr, input type of m_start_pt: "
            "{}".format(type(m_start_pt)))
        check_scalar_dtype(m_start_pt,
                           "scalar_m_start_pt should be a scalar of int/uint")
        TikCheckUtil.check_type_match(
            k_start_pt, (int, Scalar, Expr),
            "k_start_pt should be int, Scalar, Expr, input type of "
            "k_start_pt: {}".format(type(k_start_pt)))
        check_scalar_dtype(k_start_pt,
                           "scalar_k_start_pt should be a scalar of int/uint")
        TikCheckUtil.check_type_match(
            k_extension, (int, Scalar, Expr),
            "k_extension should be int, Scalar, Expr, input type of k_extension"
            ": {}".format(type(k_extension)))
        check_scalar_dtype(k_extension,
                           "scalar_k_extension should be a scalar of int/uint")
        check_wino_ft_params(l1_h, l1_w, l1_c, dst_stride, pad_left, pad_right,
                             pad_top, pad_bottom, m_extension, m_start_pt,
                             k_extension, k_start_pt, column_indicator, src)
        # calculate extent
        src_extent = Expr(l1_w*l1_h*l1_c*DTYPE_SIZE[src.dtype]).get()
        # one instr writes 4 expand feature maps
        expand_fm_num = 4
        dst_extent = m_extension*k_extension*expand_fm_num*\
                     DTYPE_SIZE[src.dtype] + \
                     dst_stride*BYTE_PER_FRACTAL*(expand_fm_num - 1)
        dst_extent = Expr(dst_extent).get()
        # code gen
        config = [l1_w, l1_h, l1_c, dst_stride, column_indicator,
                  WINO_PAD_MAP[(pad_left, pad_right)],
                  WINO_PAD_MAP[(pad_top, pad_bottom)]]
        reg_xm = concat_params(config, WINO_FM_XM_OFFSET_LIST,
                               WINO_FM_XM_SEGMENT_LIST)
        config = [k_extension, k_start_pt, m_extension, m_start_pt]
        reg_xt = concat_params(config, WINO_FM_XT_OFFSET_LIST,
                               WINO_FM_XT_SEGMENT_LIST)
        with self.new_scope():
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_MTE1)
            instr = tvm.call_extern(
                dst.dtype, "load_cbuf_to_ca_winograd",
                dst.access_ptr("w", extent=dst_extent),
                src.access_ptr("r", extent=src_extent), reg_xm, reg_xt)
            self.emit(instr, ONE_IR)

    @source_info_decorator()
    @debug.winograd_weight_trans_decorator
    def winograd_weight_transform( # pylint: disable=R0913, R0914
            self, dst, src, column_indicator, repeat_dir, repeat_times,
            dst_blk_stride, dst_rep_stride, src_rep_stride,
            en_weight_offset=False, smask=None):
        """ reads 9 fractal matrixes from L1, performs partial winograd
        transform and writes 4 transformed fractal matrix into L0B

        Parameters
        ----------
        dst: destination operator, scope_cbuf
        src: src operator, scope_cb
        column_indicator: partial weight indicator
                          0: the 1st column
                          1: the 2nd column
                          2: the 3rd column
                          3: the 4th column
        repeat_dir: repeating direction indicator which is used to indicate on
                    which direction this instruction is repeating
                    0: vertical
                    1: horizontal
        repeat_times: the number of iterations this instruction would be
                      executed
        dst_blk_stride: inner destination stride between the 4 weight matrixes
                        to be written into L0B in one single iteration in unit
                        of fractal matrix
        dst_rep_stride: destination repeat stride between the desitination
                        addresses of 2 successive interations
        src_rep_stride: source repeat stride between the base source addresses
                        of 2 successive iterations
        en_weight_offset: not support yet.
        smask: not support yet.

        Returns
        -------
        None
        """
        arch_version_dst_src_map = {
            'v200': {'hisi-es': ['s8s8'], 'hisi-cs': ['s8s8', 'u8u8'],
                     'aic': ['s8s8', 'u8u8', 'f16f16']}}
        woff_arch_version_map = ['v200hisi-cs']
        # check instruction
        TikCheckUtil.check_in_range(
            self.core_arch, arch_version_dst_src_map.keys(),
            "input core_arch: {} doesn't support "
            "winograd_weight_transform.".format(self.core_arch))
        TikCheckUtil.check_in_range(
            self.core_version,
            arch_version_dst_src_map[self.core_arch].keys(),
            "{}-{} doesn't support winograd_weight_transform."
            .format(self.core_arch, self.core_version))
        # check scope
        TikCheckUtil.check_equality(
            dst.scope, "local.L0B",
            "dst scope should be l0b, input scope: {}".format(dst.scope))
        TikCheckUtil.check_equality(src.scope, "local.L1",
                                    "src scope should be l1, input scope: {}"
                                    .format(src.scope))
        # check dtype
        dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src.dtype]
        TikCheckUtil.check_in_range(
            dtype_str, arch_version_dst_src_map[self.core_arch][
                self.core_version],
            "{}-{} winograd_weight_transform doesn't support from {} to {}"
            .format(self.core_arch, self.core_version, src.dtype, dst.dtype))
        # check column_indicator
        check_integer_in_range(
            column_indicator, range(MAX_COL_INDIC),
            "column_indicator should be in the range of [0, 3], input "
            "column_indicator: {}".format(column_indicator))
        # check repeat_dir
        check_integer_in_range(
            repeat_dir, range(MAX_REP_DIR),
            "repeat_dir should be in the range of [0, 1], input repeat_dir: {}"
            .format(repeat_dir))
        # check repeat_times
        check_repeat_times(repeat_times)
        # check stride
        check_integer_in_range(
            dst_blk_stride, range(MAX_BLK_STRIDE_SINGLE_BYTE),
            "dst_blk_stride should be in the range of [0, 255], input "
            "dst_blk_stride: {}".format(dst_blk_stride))
        check_integer_in_range(
            dst_rep_stride, range(MAX_REP_STRIDE_SINGLE_BYTE),
            "dst_rep_stride should be in the range of [0, 255], input "
            "dst_rep_stride: {}".format(dst_rep_stride))
        check_integer_in_range(
            src_rep_stride, range(MAX_REP_STRIDE_DOUBLE_BYTE),
            "src_rep_stride should be in the range of [0, 65535], input "
            "src_rep_stride: {}".format(src_rep_stride))
        # check en_weight_offset
        arch_version_str = self.core_arch + self.core_version
        if en_weight_offset:
            TikCheckUtil.check_in_range(
                arch_version_str, woff_arch_version_map,
                "{}-{} doesn't support weight_offset."
                .format(self.core_arch, self.core_version))
            check_weight_offset(smask, "winograd_weight_transform", "smask")
            # not support smask yet
            smask_addr = smask.accss_ptr("r")
            woff_bit = 1
        else:
            smask_addr = 0
            woff_bit = 0
        # calculate extent
        src_extent, dst_extent = _calculate_winograd_ft_extent(
            repeat_times, src_rep_stride, dst_rep_stride, dst_blk_stride)
        # code gen
        # smask cannot support config mode
        config = [dst_blk_stride, src_rep_stride, dst_rep_stride, smask_addr,
                  column_indicator, repeat_dir, woff_bit, repeat_times]
        args = concat_params(config, WINO_WGT_OFFSET_LIST,
                             WINO_WGT_SEGMENT_LIST)
        with self.new_scope():
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_MTE1)
            instr = tvm.call_extern(
                dst.dtype, "load_cbuf_to_cb_winograd",
                dst.access_ptr("w", extent=dst_extent),
                src.access_ptr("r", extent=src_extent), args)
            self.emit(instr, ONE_IR)

    def check_vbi_param(self, dst, src0, src1,  # pylint: disable=R0913
                        src0_offset, dst_blk_stride,
                        vertical_repeat_times, horizontal_repeat_times,
                        repeat_mode, vertical_repeat_offset):
        """check param for vbi instruction"""
        # check operator
        TikCheckUtil.check_type_match(dst, Tensor, "dst should be tensor but "
                                                   "get %s" % type(dst))
        TikCheckUtil.check_type_match(src0, Tensor, "src0 should be tensor but "
                                                    "get %s" % type(src0))
        TikCheckUtil.check_type_match(src1, Tensor, "src1 should be tensor but "
                                                    "get %s" % type(src1))
        TikCheckUtil.check_type_match(src0_offset, Tensor,
                                      "src0_offset should be tensor but "
                                      "get %s" % type(src0_offset))
        TikCheckUtil.check_equality(src0.scope, scope_ubuf,
                                    "src0's scope must be UB")
        TikCheckUtil.check_equality(src1.scope, scope_ubuf,
                                    "src1's scope must be UB")
        TikCheckUtil.check_equality(src0_offset.scope, scope_ubuf,
                                    "src0_offset's scope must be UB")
        TikCheckUtil.check_equality(dst.scope, scope_ubuf,
                                    "dst's scope must be UB")
        # check operator dtype
        TikCheckUtil.check_equality(dst.dtype, src0.dtype,
                                    "Intrinsic {}'s src0's dtype should be "
                                    "equal to dst's dtype".
                                    format("vbi"))
        TikCheckUtil.check_equality(dst.dtype, src1.dtype,
                                    "Intrinsic {}'s src1's dtype should be "
                                    "equal to dst's dtype".
                                    format("vbi"))
        TikCheckUtil.check_equality(
            intrinsic_check_support("Intrinsic_" + "vbi",
                                    dst.dtype), True,
            INSTR_DTYPE_SUPPORT_STATEMENT.format(dst.dtype,
                                                 "vbi"))
        # check dst_blk_stride
        TikCheckUtil.check_type_match(
            dst_blk_stride, (int, Scalar, Expr),
            "dst_blk_stride should be int, Expr or Scalar, input type is"
            " %s" % type(dst_blk_stride))
        check_scalar_dtype(dst_blk_stride, "scalar_dst_blk_stride should "
                                           "be a scalar of int/uint")
        check_integer_in_range(
            dst_blk_stride, range(MAX_BLK_STRIDE_DOUBLE_BYTE),
            "dst_blk_stride should be in the range of [0, %s], input value"
            " is %s" % (MAX_BLK_STRIDE_DOUBLE_BYTE - 1, dst_blk_stride))
        # check vertical_repeat_times
        check_repeat_times(vertical_repeat_times)
        # check horizontal_repeat_times
        check_repeat_times(horizontal_repeat_times)
        # check repeat_mode
        TikCheckUtil.check_equality(
            type(repeat_mode), int, "repeat_mode should be int, input type "
                                    "is %s" % type(repeat_mode))
        TikCheckUtil.check_in_range(
            repeat_mode, (0, 1),
            "repeat_mode only support 0 and 1, input value is %s" % repeat_mode)
        # check vertical_repeat_offset
        TikCheckUtil.check_type_match(
            vertical_repeat_offset, (int, Scalar, Expr),
            "vertical_repeat_offset should be int, Expr and Scalar, "
            "input type is %s" % type(vertical_repeat_offset))
        check_scalar_dtype(vertical_repeat_offset,
                           "scalar_vertical_repeat_offset should "
                           "be a scalar of int/uint")
        check_integer_in_range(
            vertical_repeat_offset, range(MAX_REP_STRIDE_DOUBLE_BYTE),
            "vertical_repeat_offset should be in the range of [0, %s], "
            "input value is %s" % (MAX_REP_STRIDE_DOUBLE_BYTE - 1,
                                   vertical_repeat_offset))

    @source_info_decorator()
    @debug.vbi_decorator
    def vbi(self, mask, dst, src0, src1, src0_offset,  # pylint: disable=R0913
            dst_blk_stride, vertical_repeat_times, horizontal_repeat_times,
            repeat_mode, vertical_repeat_offset):
        """vbi instruction, used for bilinear interpolation in ROI alignment

        Parameters
        ----------
        mask: Effective operation on element
        dst: destination tensor
        src0: src0 tensor
        src1: src1 tensor
        src0_offset: src0_offset tensor
        dst_blk_stride: offset of dst operator between different block
                        in one iteration
        vertical_repeat_times: repeat_times in vertical direction
        horizontal_repeat_times: repeat_times in horizontal direction
        repeat_mode: indicate how many elements at src1 are consumed
                     in one iteration
        vertical_repeat_offset: vertical repeat offset between dst address of
                                iterations in the vertical direction
        Returns
        -------
        None
        """
        self.check_vbi_param(dst, src0, src1, src0_offset, dst_blk_stride,
                             vertical_repeat_times, horizontal_repeat_times,
                             repeat_mode, vertical_repeat_offset)

        # check mask and get mask_o
        mask_o = mask_concat(self, mask, tensor_bit_len=get_bit_len(dst.dtype))
        # check tensor overflow, including src1, dst and src_offset
        mask_len = get_vbi_mask_len(mask)
        check_vbi_dst_offset_overflow(dst, src0_offset, mask_len,
                                      horizontal_repeat_times,
                                      vertical_repeat_times,
                                      dst_blk_stride, vertical_repeat_offset)
        check_vbi_src1_tensor_overflow(src1, repeat_mode,
                                       vertical_repeat_times*\
                                       horizontal_repeat_times, mask_len,
                                       src1.offset)
        # check src0_offset and src1 overlap
        check_vbi_overlap(src0_offset, src1, repeat_mode,
                          vertical_repeat_times*horizontal_repeat_times,
                          mask_len, src0_offset.offset, src1.offset)
        # gen code
        self._gen_vbi_code(mask_o, mask_len, dst, src0,
                           src1, src0_offset, dst_blk_stride,
                           vertical_repeat_times, horizontal_repeat_times,
                           repeat_mode, vertical_repeat_offset)

    def _gen_vbi_code_vadds_part(self, mask_len,  # pylint: disable=R0913
                                 dst, src0, src0_offset, total_repeat_times):
        """for vbi instruction, generate vadds part code"""
        with self.context.freeze():  # pylint: disable=E1101
            base_addr = self.Scalar_("int32")  # pylint: disable=E1101
            base_addr.set_as(tvm.expr.Cast("int64", tvm.call_extern(
                "handle", "", src0.access_ptr("r"))).astype("int32"))
            need_src0_block = get_vbi_src0_offset_need_size(
                dst.dtype, mask_len, total_repeat_times)
            self.vadds(need_src0_block, src0_offset,  # pylint: disable=E1101
                       src0_offset, base_addr, MIN_REPEAT_TIMES,
                       MIN_STRIDE, MIN_STRIDE,
                       BLK_NUM_PER_REP, BLK_NUM_PER_REP, mask_mode="counter")

    def _gen_vbi_code_vbi_part(self, mask_o, dst, src1,  # pylint: disable=R0913
                               src0_offset, dst_extent, src0_offset_extent,
                               src1_extent, config):
        """for vbi instruction, generate vbi part code"""
        with self.new_scope():
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o),
                      ONE_IR)
            instr = tvm.call_extern(
                dst.dtype, "vbi",
                dst.access_ptr("w", extent=dst_extent),
                src0_offset.reinterpret_cast_to("uint16").access_ptr(
                    "r", extent=src0_offset_extent),
                src1.access_ptr("r", extent=src1_extent), *type_convert(config))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr, ONE_IR)

    def _gen_vbi_code(self, mask_o, mask_len, dst,  # pylint: disable=R0913
                      src0, src1, src0_offset, dst_blk_stride,
                      vertical_repeat_times, horizontal_repeat_times,
                      repeat_mode, vertical_repeat_offset):
        """generate vbi code"""
        self._gen_vbi_code_vadds_part(
            mask_len, dst, src0, src0_offset,
            horizontal_repeat_times*vertical_repeat_times)
        self._gen_vbi_code_vbi_part(
            mask_o, dst, src1, src0_offset,
            *cal_vbi_extent(mask_len, dst, src1, src0_offset,
                            horizontal_repeat_times, repeat_mode,
                            dst_blk_stride, vertical_repeat_offset,
                            vertical_repeat_times),
            [horizontal_repeat_times, repeat_mode, dst_blk_stride,
             vertical_repeat_offset, vertical_repeat_times])
