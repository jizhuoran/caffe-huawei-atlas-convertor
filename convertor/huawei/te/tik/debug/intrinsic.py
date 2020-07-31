"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     intrinsic.py
DESC:     debug intrinsic
CREATED:  2019-7-04 20:12:13
MODIFIED: 2019-7-24 14:04:45
"""
# disabling:
# R0902: too-many-instance-attributes
# R0913: too-many-arguments
# R0914: too-many-locals
# C0302: too-many-lines
# E1101: no-member
# pylint: disable=C0302

import sys
import math
import numpy as np

from te.platform.cce_params import scope_ubuf
from te.platform.cce_params import scope_gm
from te.platform.cce_params import scope_cc
from te.platform.cce_params import HI3796CV300ESAIC
from te.platform.cce_params import ASCEND_310AIC
from te.platform.cce_params import ASCEND_310
from te.platform.cce_params import ASCEND_910
from te.platform.cce_params import AIC
from te.platform.cce_params import HI3796CV300ES
from te.platform.cce_params import ASCEND_610
from te.platform.cce_params import ASCEND_620
from te.platform.cce_params import ASCEND_910AIC
from te.tik.common.util import TikUtil, DTYPE_SIZE, check_integer_in_range
from te.tik.common.common_util import check_load3dv2_channel_size, \
    check_load3dv2_k_extension, check_load3dv2_m_extension,\
    check_dilation_filter_size, check_depthwise_conv_params, \
    check_depthwise_conv_l1_w, float16format2uint16, check_dict_and_not_none, \
    aipp_get_enable_bit, check_aipp_one_src_overflow, \
    check_aipp_two_src_overflow, is_scalar, is_tensor
from .statement import STMT
from .util import copy_tensor_to_model, cvt_float_to_uint,\
    get_dtype_size, get_dtype_bit_width, reinterpret_type
from .sim.util import TempEnv
from ..tik_lib.tik_params import INC_MODE, DEC_MODE, CONV_F162F32_NO_RELU, \
    CONV_F322F16_IS_RELU, CONV_F322F16_NO_RELU, BYTE_SIZE, \
    CONV_S322F16_QUANT, CONV_S322F16_VECTOR_QUANT, NO_CONV_IS_RELU, \
    SPR_CONFIG_BIT_LEN, ALIGN_TENSOR, ALIGN_DST, ALIGN_SRC, \
    REPEAT_MODE_SHIFT_BIT, MMAD_MATRIX_K_POS, MMAD_MATRIX_N_POS, \
    MMAD_EN_WINOGRAD_A_POS, MMAD_EN_WINOGRAD_B_POS, MAX_ADDR, \
    MMAD_EN_WEIGHT_OFFSET_POS, MMAD_EN_SSPARSE_POS, MMAD_L0C_BIT_POS, \
    ALIGNED_ADDR, REPEAT_SHIFT_POS, UINT64_BIT, MAX_MATRIX, \
    MAX_REPEAT_TIMES, MAX_BLK_STRIDE_DOUBLE_BYTE, MAX_START_INDEX, \
    MAX_DST_GAP_DOUBLE_BYTE, MAX_FETCH_POS, MIN_START_POINT, MAX_START_POINT, \
    MAX_C1_INDEX, MIN_STRIDE, MAX_STRIDE, MIN_FILTER_WIDTH, MAX_FILTER_WIDTH, \
    MIN_FILTER_HEIGHT, MAX_FILTER_HEIGHT, MIN_DILATION, MAX_DILATION, \
    MIN_JUMP_OFFSET, MAX_JUMP_OFFSET, MIN_NBURST, MAX_NBURST_SINGLE_BYTE, \
    MIN_BURST_LEN, MAX_BURST_LEN_SINGLE_BYTE, MAX_BLK_STRIDE_SINGLE_BYTE, \
    MAX_SID, MAX_NBURST_DOUBLE_BYTE, MAX_BURST_LEN_DOUBLE_BYTE, \
    MIN_EXTENSION, MAX_EXTENSION, MAX_START_PT, MAX_TENSOR_WIDTH, \
    MAX_TENSOR_HEIGHT, MIN_TENSOR_WIDTH, MIN_TENSOR_HEIGHT, \
    MAX_PADDING, MAX_REPEAT_MODE, PADDING_BOT_IDX, PADDING_TOP_IDX, \
    SHIFT_BIT_POS_7, SHIFT_BIT_POS_11, \
    PADDING_RIGHT_IDX, PADDING_LEFT_IDX, YUV420, XRGB8888, \
    NC1HWC0DI_INT8, NC1HWC0DI_FP16, RGB888, ARGB8888, YUYV, YUV422, AYUV444, \
    YUV400, RAW10, RAW12, RAW16, RAW24, AIPP_INPUT_VERSON_AND_FUNCTION, \
    SWAP, CSC, DTC, AERA_PADDING, PRE_CLIP, SCF, POST_CLIP, FLIP, STRETCH, \
    RAW, AIPP_INPUT_TYPE_SWAP_ALIGN, AIPP_FORMAT_CONVERT, ALIGN_SRC_EVEN, \
    AIPP0_OFFSET_LIST, AIPP1_OFFSET_LIST, AIPP2_OFFSET_LIST,  \
    AIPP3_OFFSET_LIST, AIPP4_OFFSET_LIST, AIPP5_OFFSET_LIST,\
    AIPP6_OFFSET_LIST, AIPP7_OFFSET_LIST, \
    AIPP8_OFFSET_LIST, AIPP9_OFFSET_LIST, AIPP10_OFFSET_LIST, \
    AIPP11_OFFSET_LIST, AIPP12_OFFSET_LIST, \
    AIPP13_OFFSET_LIST, AIPP15_OFFSET_LIST, AIPP16_OFFSET_LIST, \
    AIPP17_OFFSET_LIST, AIPP_XS_OFFSET_LIST, \
    AIPP_XT_OFFSET_LIST, BIT_16, AIPP_INIT_FLOAT_VALUE_ONE, RAW_TO_16_N, \
    SCALE_COF, SHIFT_BIT_POS_32, SHIFT_BIT_POS_59, SHIFT_BIT_POS_56, \
    SHIFT_BIT_POS_55, SHIFT_BIT_POS_54, SHIFT_BIT_POS_52, SHIFT_BIT_POS_48,\
    SHIFT_BIT_POS_20, SHIFT_BIT_POS_16, SHIFT_BIT_POS_8, ONE_BLK_SIZE, \
    CONV_L0C16_DEQ, VALUE_BI_1001, SHIFT_BIT_POS_47, VALUE_BI_1010, \
    VALUE_BI_1011, VALUE_BI_1100, VALUE_BI_1101, SHIFT_BIT_POS_2, \
    CONV_S322B8_DEQ, SHIFT_BIT_POS_63, SHIFT_BIT_POS_24
from ..common.tik_get_soc_name import get_soc_core_type
from ..common.tik_get_soc_name import get_soc_name
from ..tik_lib.tik_check_util import TikCheckUtil
from ..tik_lib.tik_api_constants import WINO_PAD_MAP

_BRC_DST_ALIGN = {
    'L0C16': 512,
    'L0C32': 1024
}
_BRC_SRC_ID = {
    'UB': 0,
    'L1': 1,
}
_BRC_DST_ID = {
    'L0C16': 0,
    'L0C32': 1
}

_WINO_TYPE_ID = {
    'int8': 0,
    'uint8': 1,
    "float16": 2
}

_LOAD2D_SRC_ID = {'L1': 0, 'OUT': 1}

_LOAD2D_DST_ID = {'L0A': 0, 'L0B': 1, 'L1': 2}

_LOAD2D_DTYPE_ID = {'int8': 1, 'uint8': 0, 'float16': 1}

_LOAD2D_DTYPE_ID_L0B = {'int8': 1, 'uint8': 1, 'float16': 0}

_DMA_SRC_BIN_ID = {
    'L0C16': '0b0000',
    'UB': '0b0001',
    'OUT': '0b0010',
    'L0C32': '0b0011',
    'L1': '0b0100',
    'L0C16V': '0b0101',
    'L0C32V': '0b0110',
    'L0CSC16': '0b0111',
    'L0CSC32': '0b1000',
    'L0CDPF16': '0b1001',
    'L0CDPF32': '0b1010'
}

_DMA_DST_BIN_ID = {
    'L0C16': '0b0000',
    'UB': '0b0001',
    'L1': '0b010',
    'OUT': '0b0011',
    'L0C32': '0b0100',
    'L0C16V': '0b0101',
    'L0C32V': '0b0110',
    'L0CSC32': '0b1001'
}

_L0C_SRC_16B_SCOPE_NAME = {
    'v': 'L0C16V',
    'm': 'L0C16',
    'dp': 'L0CDPF16',
}

_L0C_SRC_32B_SCOPE_NAME = {
    'v': 'L0C32V',
    'm': 'L0C32',
    'dp': 'L0CDPF32',
    'sc': 'L0CSC32'
}

_L0C_DST_16B_SCOPE_NAME = {
    'v': 'L0C16V',
    'm': 'L0C16',
}

_L0C_DST_32B_SCOPE_NAME = {
    'v': 'L0C32V',
    'm': 'L0C32',
    'sc': 'L0CSC32'
}

_LOAD3D_DST_ID = {'L0A': 0, 'L0B': 1, 'UB': 2}

_BRC_DST_TYPE_ID = {'float16': 0, 'float32': 1, 'int32': 1}

_MMAD_TYPE_BITS = {
    ('uint8', 'uint8', 'uint32'): 0b000,
    ('uint8', 'uint8', 'int32'): 0b000,
    ('int8', 'int8', 'int32'): 0b001,
    ('float16', 'float16', 'float16'): 0b010,
    ('float16', 'float16', 'float32'): 0b011,
    ('uint8', 'int8', 'int32'): 0b101
}

_SCALAR_CONV_ID = {
    ('float32', 'int32', 'away-zero'): 0b00000,
    ('float32', 'int32', 'floor'): 0b00001,
    ('float32', 'int32', 'ceil'): 0b00010,
    ('float32', 'int32', 'to-zero'): 0b00011,
    ('float32', 'int32', 'round'): 0b00100,
    ('int32', 'float32'): 0b00101,
    ('float32', 'float16'): 0b00110,
    ('float16', 'float32'): 0b00111,
    ('float32', 'float16', 'odd'): 0b01000,
}

_DMA_V100_ALIGN = {
    'L1': 32,
    'UB': 32,
    'OUT': 1,
    'L0C16': 512,
    'L0C32': 1024
}

_SET_2D_DST_ID = {
    'L0A': 0b00,
    'L0B': 0b01,
    'L1': 0b10
}

_LOAD3DV2_DST_ID = {
    'L0A': 0b00,
    'L0B': 0b01,
    'UB': 0b10
}

_LOAD3DV2_TYPE_ID = {
    '8': 0b00,
    '16': 0b01,
    '4': 0b10
}

_MAX_L0C16_DTYPE_SIZE = 2

_NBURST_SHIFT_BIT_POS = 4
_BURST_SHIFT_BIT_POS = 16
_SRC_STRIDE_SHIFT_BIT_POS = 32
_DST_STRIDE_SHIFT_BIT_POS = 48

_REPEAT_SHIFT_BIT_POS = 16
_LOAD2D_SRC_STRIDE_SHIFT_POS = 24
_PAD_VALUE_SHIFT_BIT_POS = 8

_CTRL_SHIFT_BIT_POS = 63
_JUMP_OFFSET_SHIFT_BIT_POS = 44
_SHIFT_BIT_POS = 52

_FETCH_W_SHIFT_BIT_POS = 16
_FETCH_H_SHIFT_BIT_POS = 24
_LEFT_TOP_W_SHIFT_BIT_POS = 32
_LEFT_TOP_H_SHIFT_BIT_POS = 48

_STRIDE_H_SHIFT_BIT_POS = 6
_FILTER_W_SHIFT_BIT_POS = 12
_FILTER_H_SHIFT_BIT_POS = 20
_DILATION_W_SHIFT_BIT_POS = 28
_DILATION_H_SHIFT_BIT_POS = 36

_L1_H_SHIFT_BIT_POS = 16
_FM_H_SHIFT_BIT_POS = 16
_PAD_L_SHIFT_BIT_POS = 32
_PAD_R_SHIFT_BIT_POS = 40
_PAD_TOP_SHIFT_BIT_POS = 48
_PAD_BOT_SHIFT_BIT_POS = 56

_BURST_LEN_SHIFT_BIT_POS = 8
_SRC_GAP_SHIFT_BIT_POS = 16
_DST_GAP_SHIFT_BIT_POS = 24

_DEPTH_L1_H_SHIFT_POS = 12
_FEATURE_OFFSET_SHIFT_POS = 24
_PAD_MODE_SHIFT_POS = 62

_M_EXTENSION_SHIFT_BIT_POS = 16
_K_START_POINT_SHIFT_BIT_POS = 32
_M_START_POINT_SHIFT_POS = 48
_EN_TRANSPOSE_SMALL_K_SHIFT_POS = 44
_CHANNEL_SIZE_SHIFT_BIT_POS = 48

_L1_C_SHIFT_BIT_POS = 32
_C_IDX_SHIFT_BIT_POS = 54
_PAD_H_SHIFT_BIT_POS = 56
_PAD_W_SHIFT_BIT_POS = 59
_K_EXT_SHIFT_BIT_POS = 8
_K_ST_PT_SHIFT_BIT_POS = 20
_M_EXT_SHIFT_BIT_POS = 32


def _check_pad_list(pad_list):
    """ check range of elements in pad_list

    Parameters
    ----------
    pad_list: parameters of instruction load3dv1, load3dv2, col2img

    Returns
    -------
    None
    """
    for index, pad_ele in enumerate(pad_list):
        check_integer_in_range(
            pad_ele, range(MAX_PADDING),
            "pad_list[{}] should be in the range of [0, 255], input "
            "value: {}".format(index, pad_ele))

def _get_src_dst_mem_id(src, dst, block_mode='m'):
    """get dst_mem_id and src_mem_id

    Parameters
    ----------
    src : src scope

    dst : dst scope

    Returns
    -------
    src_mem_id, dst_mem_id
    """
    src_scope = TikUtil.get_storage_scope(src.scope)
    dst_scope = TikUtil.get_storage_scope(dst.scope)

    src_dtype_size = get_dtype_size(src.dtype)
    dst_dtype_size = get_dtype_size(dst.dtype)

    src_scope_name = src_scope
    if src_scope_name == 'L0C':
        if src_dtype_size <= _MAX_L0C16_DTYPE_SIZE:
            src_scope_name = _L0C_DST_16B_SCOPE_NAME[block_mode]
        else:
            src_scope_name = _L0C_SRC_32B_SCOPE_NAME[block_mode]

    dst_scope_name = dst_scope
    if dst_scope_name == 'L0C':
        if dst_dtype_size <= _MAX_L0C16_DTYPE_SIZE:
            dst_scope_name = _L0C_DST_16B_SCOPE_NAME[block_mode]
        else:
            dst_scope_name = _L0C_DST_32B_SCOPE_NAME[block_mode]

    # binary dtype -> int dtype
    src_mem_id = int(_DMA_SRC_BIN_ID[src_scope_name], 2)
    dst_mem_id = int(_DMA_DST_BIN_ID[dst_scope_name], 2)

    return src_mem_id, dst_mem_id, src_scope_name


def _load3dv1_col2img_param_check(fetch_filter_h,
                                  fetch_filter_w, left_top_h, left_top_w):
    """check params of load3dv1 or col2img"""
    check_integer_in_range(fetch_filter_h, range(MAX_FETCH_POS),
                           "fetch_filter_h should be in the range of [0, 255],"
                           " input value is %s" % str(fetch_filter_h))
    check_integer_in_range(fetch_filter_w, range(MAX_FETCH_POS),
                           "fetch_filter_w should be in the range of [0, 255],"
                           " input value is %s" % str(fetch_filter_w))
    check_integer_in_range(left_top_h, range(MIN_START_POINT, MAX_START_POINT),
                           "left_top_h should be in the range of [-255, 32767],"
                           " input value is %s" % str(left_top_h))
    check_integer_in_range(left_top_w, range(MIN_START_POINT, MAX_START_POINT),
                           "left_top_w should be in the range of [-255, 32767],"
                           " input value is %s" % str(left_top_w))


def _load3d_v1_v2_col2img_param_check(stride_w, stride_h, filter_w,
                                      filter_h, dilation_filter_w,
                                      dilation_filter_h):
    """check params of load3dv1, load3dv2 or col2img"""
    # pylint: disable=R0913
    check_integer_in_range(stride_w, range(MIN_STRIDE, MAX_STRIDE),
                           "stride_w should be in the range of [1, 63], "
                           "input stride_w: {}".format(stride_w))
    check_integer_in_range(stride_h, range(MIN_STRIDE, MAX_STRIDE),
                           "stride_h should be in the range of [1, 63], "
                           "input stride_h: {}".format(stride_h))
    check_integer_in_range(filter_w, range(MIN_FILTER_WIDTH, MAX_FILTER_WIDTH),
                           "filter_w should be in the range of [1, 255], "
                           "input filter_w: {}".format(filter_w))
    check_integer_in_range(
        filter_h, range(MIN_FILTER_HEIGHT, MAX_FILTER_HEIGHT),
        "filter_h should be in the range of [1, 255], "
        "input filter_h: {}".format(filter_h))
    check_integer_in_range(
        dilation_filter_w, range(MIN_DILATION, MAX_DILATION),
        "dilation_filter_w should be in the range of [1, 255], "
        "input dilation_filter_w: {}".format(dilation_filter_w))
    check_integer_in_range(
        dilation_filter_h, range(MIN_DILATION, MAX_DILATION),
        "dilation_filter_h should be in the range of [1, 255], "
        "input dilation_filter_h: {}".format(dilation_filter_h))


def _check_params_dma_instr(sid, nburst, burst, src_stride, dst_stride):
    """check params of dma instr"""
    check_integer_in_range(sid, range(MAX_SID),
                           "sid should be in the range of [0, 15],"
                           " input value is %s" % str(sid))
    check_integer_in_range(nburst,
                           range(MIN_NBURST, MAX_NBURST_DOUBLE_BYTE),
                           "nburst should be in the range of [1, 4095],"
                           " input value is %s" % str(nburst))
    check_integer_in_range(
        burst, range(MIN_BURST_LEN, MAX_BURST_LEN_DOUBLE_BYTE),
        "burst_len should be in the range of [1, 65535],"
        " input value is %s" % str(burst))
    check_integer_in_range(src_stride,
                           range(MAX_BLK_STRIDE_DOUBLE_BYTE),
                           "src_stride should be in the range of [0, 65535],"
                           " input value is %s" % str(src_stride))
    check_integer_in_range(dst_stride,
                           range(MAX_BLK_STRIDE_DOUBLE_BYTE),
                           "dst_stride should be in the range of [0, 65535],"
                           " input value is %s" % str(dst_stride))


def _load3d_v2_params_check(k_extension, m_extension, k_start_pt, m_start_pt, dtype):
    """check params of load3dv2"""
    check_integer_in_range(
        k_extension, range(MIN_EXTENSION, MAX_EXTENSION),
        "k_extension should be in the range of [1, 65535], "
        "input k_extension: {}".format(k_extension))
    check_integer_in_range(
        m_extension, range(MIN_EXTENSION, MAX_EXTENSION),
        "m_extension should be in the range of [1, 65535], "
        "input m_extension: {}".format(m_extension))
    check_integer_in_range(
        k_start_pt, range(MAX_START_PT),
        "k_start_pt should be in the range of [0, 65535], "
        "input k_start_pt: {}".format(k_start_pt))
    k_start_pt_byte_align = 32
    if k_start_pt*DTYPE_SIZE[dtype] % k_start_pt_byte_align != 0:
        TikCheckUtil.raise_error(
            "k_start_pt in Byte should be multiple of 32B, input "
            "k_start_pt: {},"
            " input src dtype: {}".format(k_start_pt, dtype))
    check_integer_in_range(
        m_start_pt, range(MAX_START_PT),
        "m_start_pt should be in the range of [0, 65535], "
        "input m_start_pt: {}".format(m_start_pt))
    if m_start_pt % 16 != 0:
        TikCheckUtil.raise_error(
            "m_start_ptshould be multiple of 16, input "
            "m_start_pt: {}".format(m_start_pt))


class Load2D(STMT):
    """Load2D instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, dst, src, index, repeat_time, dst_gap,
                 src_stride, sid, transpose=False, addr_mode=None):
        # pylint: disable=R0913
        super(Load2D, self).__init__(source_info)
        self.dst = dst
        self.sid = sid
        self.dtype = dst.dtype
        self.src = src
        self.index = index
        self.repeat_time = repeat_time
        self.dst_gap = dst_gap
        self.src_stride = src_stride
        self.transpose = transpose
        self.addr_mode = addr_mode

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        temp_env = TempEnv()

        src_align, dst_align = self.align_fn(context)

        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, src_align, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, dst_align, access_mode='w')

        param = context.encoder.new_param()
        param.src_mem_id = _LOAD2D_SRC_ID[
            TikUtil.get_storage_scope(self.src.scope)]
        param.dst_mem_id = _LOAD2D_DST_ID[
            TikUtil.get_storage_scope(self.dst.scope)]
        # 1 enable transpose; 0 disable transpose
        param.trans = 1 if self.transpose else 0
        param.type = _LOAD2D_DTYPE_ID[self.src.dtype]
        param.addrmode = 0
        if self.addr_mode is not None:
            if self.addr_mode == 'inc':
                param.addrmode = INC_MODE
            elif self.addr_mode == 'dec':
                param.addrmode = DEC_MODE
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = self.create_gpr_x_m(context, temp_env)

        instr = context.encoder.gen_dma_ld_2d(param)

        context.model.step(instr)

        temp_env.check_mem_access(context.model)

        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

    def align_fn(self, context):
        """get the src and dst align

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        src_align, dst_align
        """
        dst_scope = TikUtil.get_storage_scope(self.dst.scope)
        src_align = 512
        dst_align = 512

        if get_soc_name() in (HI3796CV300ES, ASCEND_610, ASCEND_620):
            src_align = get_dtype_size(self.src.dtype)
            if dst_scope in ('L0A', 'L0B'):
                dst_align = 512
            else:
                dst_align = get_dtype_size(self.dst.dtype)
        return src_align, dst_align

    def create_gpr_x_m(self, context, temp_env):
        """create register x_m

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xm_idx
        """
        repeat = context.evaluate_expr(self.repeat_time)
        # check repeat
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times: {}".format(repeat))
        src_stride = context.evaluate_expr(self.src_stride)
        # check src_stride
        check_integer_in_range(
            src_stride, range(MAX_BLK_STRIDE_DOUBLE_BYTE),
            "src_stride should be in the range of [0, 65535],"
            " input value is %s" % str(src_stride))
        xm_idx = temp_env.alloc_register()
        index_ = context.evaluate_expr(self.index)
        # check index
        check_integer_in_range(
            index_, range(MAX_START_INDEX),
            "start_index should be in the range of [0, 65535],"
            " input value is %s" % str(index_))
        x_m = 0
        x_m |= index_
        x_m |= repeat << _REPEAT_SHIFT_BIT_POS
        x_m |= src_stride << _LOAD2D_SRC_STRIDE_SHIFT_POS
        if self.dst_gap is not None:
            # check dst_gap
            check_integer_in_range(
                context.evaluate_expr(self.dst_gap),
                range(MAX_DST_GAP_DOUBLE_BYTE),
                "dst_gap should be in the range of [0, 65535],"
                " input value is %s" % str(
                    context.evaluate_expr(self.dst_gap)))
            x_m |= context.evaluate_expr(self.dst_gap) << 44
        x_m |= context.evaluate_expr(self.sid) << 40

        context.model.write_gpr(xm_idx, x_m)

        return xm_idx


def dma_align_fn(src, dst):
    """get the src and dst align"""
    src_scope = TikUtil.get_storage_scope(src.scope)
    dst_scope = TikUtil.get_storage_scope(dst.scope)

    src_dtype_size = get_dtype_size(src.dtype)
    dst_dtype_size = get_dtype_size(dst.dtype)

    src_scope_name = src_scope
    if src_scope_name == 'L0C':
        if src_dtype_size <= 2:
            src_scope_name = 'L0C16'
        else:
            src_scope_name = 'L0C32'

    dst_scope_name = dst_scope
    if dst_scope_name == 'L0C':
        if dst_dtype_size <= 2:
            dst_scope_name = 'L0C16'
        else:
            dst_scope_name = 'L0C32'

    if get_soc_name() in (ASCEND_310, ASCEND_910):
        src_align = _DMA_V100_ALIGN[src_scope_name]
        dst_align = _DMA_V100_ALIGN[dst_scope_name]

    else:
        def v200_align(scope, dtype):
            align = 1
            if scope == 'UB':
                align = 32
            elif scope == 'L1':
                align = DTYPE_SIZE[dtype]
            elif scope == 'OUT':
                align = DTYPE_SIZE[dtype]
            elif scope == 'L0C16':
                align = 512
            elif scope == 'L0C32':
                align = 1024
            return align
        src_align = v200_align(src_scope_name, src.dtype)
        dst_align = v200_align(dst_scope_name, dst.dtype)
        if (src_scope, dst_scope) in [('UB', 'L0C'), ('L0C', 'UB')]:
            if src_scope == 'UB':
                src_align = src_dtype_size
            elif dst_scope == 'UB':
                dst_align = dst_dtype_size

    return src_align, dst_align


class DataMove(STMT):
    """DataMove instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, dst, src, sid, nburst, burst, src_stride,
                 dst_stride, *args, **argv):
        # pylint: disable=R0913
        super(DataMove, self).__init__(source_info)
        self.dst = dst
        self.src = src
        self.sid = sid
        self.nburst = nburst
        self.burst = burst
        self.src_stride = src_stride
        self.dst_stride = dst_stride
        self.args = args
        self.kwargs = argv

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        src_align, dst_align = dma_align_fn(self.src, self.dst)

        temp_env = TempEnv()

        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, src_align, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, dst_align, access_mode='w')

        encoder = context.encoder
        param = encoder.new_param()
        param.conv_relu = 0
        param.pad = 0
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = self.create_gpr_x_m(context, temp_env)

        param.src_mem_id, param.dst_mem_id, _ = \
            _get_src_dst_mem_id(self.src, self.dst)

        instr = encoder.gen_dma_mov(param)
        context.model.step(instr)

        temp_env.check_mem_access(context.model)

        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

    def create_gpr_x_m(self, context, temp_env):
        """create register x_m

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xm_idx
        """
        xm_idx = temp_env.alloc_register()
        # check params
        _check_params_dma_instr(context.evaluate_expr(self.sid),
                                context.evaluate_expr(self.nburst),
                                context.evaluate_expr(self.burst),
                                context.evaluate_expr(self.src_stride),
                                context.evaluate_expr(self.dst_stride))

        xm_v = 0
        xm_v |= context.evaluate_expr(self.sid)
        xm_v |= context.evaluate_expr(self.nburst) << _NBURST_SHIFT_BIT_POS
        xm_v |= context.evaluate_expr(self.burst) << _BURST_SHIFT_BIT_POS
        xm_v |= context.evaluate_expr(self.src_stride) << _SRC_STRIDE_SHIFT_BIT_POS
        xm_v |= context.evaluate_expr(self.dst_stride) << _DST_STRIDE_SHIFT_BIT_POS

        context.model.write_gpr(xm_idx, xm_v)

        return xm_idx


class DataMoveDeQuant(STMT):
    """DataMoveDeQuant instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, dst, src, sid, nburst, burst, src_stride,
                 dst_stride, quant_param, relu_flag):
        # pylint: disable=R0913
        super(DataMoveDeQuant, self).__init__(source_info)
        self.dst = dst
        self.src = src
        self.sid = sid
        self.nburst = nburst
        self.burst = burst
        self.src_stride = src_stride
        self.dst_stride = dst_stride
        self.quant_param = quant_param
        self.relu_flag = relu_flag

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        align = 512

        temp_env = TempEnv()

        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, align, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, align, access_mode='w')

        encoder = context.encoder
        param = encoder.new_param()
        param.conv_relu = self.get_conv_relu_param(context, temp_env)
        param.pad = 0
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = self.create_gpr_x_m(context, temp_env)

        param.src_mem_id, param.dst_mem_id, _ = \
            _get_src_dst_mem_id(self.src, self.dst)

        instr = encoder.gen_dma_mov(param)
        context.model.step(instr)
        temp_env.check_mem_access(context.model)

        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

    def create_gpr_x_m(self, context, temp_env):
        """create register x_m

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xm_idx
        """
        xm_idx = temp_env.alloc_register()
        _check_params_dma_instr(context.evaluate_expr(self.sid),
                                context.evaluate_expr(self.nburst),
                                context.evaluate_expr(self.burst),
                                context.evaluate_expr(self.src_stride),
                                context.evaluate_expr(self.dst_stride))

        xm_v = 0
        xm_v |= context.evaluate_expr(self.sid)
        xm_v |= context.evaluate_expr(self.nburst) << _NBURST_SHIFT_BIT_POS
        xm_v |= context.evaluate_expr(self.burst) << _BURST_SHIFT_BIT_POS
        xm_v |= context.evaluate_expr(self.src_stride) << _SRC_STRIDE_SHIFT_BIT_POS
        xm_v |= context.evaluate_expr(self.dst_stride) << _DST_STRIDE_SHIFT_BIT_POS

        context.model.write_gpr(xm_idx, xm_v)

        return xm_idx

    def get_conv_relu_param(self, context, temp_env):
        """get conv_relu_param

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        conv_relu_param
        """
        conv_relu_param = 0

        if (self.src.dtype, self.dst.dtype) == ('float32', 'float16'):
            if self.relu_flag:
                conv_relu_param = CONV_F322F16_IS_RELU
            else:
                conv_relu_param = CONV_F322F16_NO_RELU

        elif (self.src.dtype, self.dst.dtype) == ('int32', 'float16'):
            if is_tensor(self.quant_param) is True:
                conv_relu_param = CONV_S322F16_VECTOR_QUANT

                _, deq_addr, _, _ = copy_tensor_to_model(
                    context, temp_env, self.quant_param,
                    ALIGNED_ADDR, access_mode='r')

                # 1  -> 32byte
                deq_spr = deq_addr
                deq_spr = deq_spr // BYTE_SIZE

                if self.relu_flag:
                    deq_spr |= 1 << SPR_CONFIG_BIT_LEN

                context.model.write_spr('DEQSCALE', deq_spr)

            else:
                conv_relu_param = CONV_S322F16_QUANT
                quant_param = context.evaluate_expr(self.quant_param)
                binary_value = cvt_float_to_uint(self.dst.dtype, quant_param)
                context.model.write_spr('DEQSCALE', binary_value)
        elif (self.src.dtype, self.dst.dtype) == ('float16', 'float32'):
            conv_relu_param = CONV_F162F32_NO_RELU
        elif self.src.dtype == self.dst.dtype and self.relu_flag:
            conv_relu_param = NO_CONV_IS_RELU

        return conv_relu_param


class TensorMove(STMT):
    """TensorMove instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, tm_dst, # pylint: disable=R0913, R0914
                 tm_src, tm_block_mode, tm_nburst,
                 tm_burst_len, tm_dst_stride, tm_src_stride, deqscale, sid,
                 relu, pad_mode, pad_value, onthefly_mode, src_onthefly,
                 src_onthefly_stride):
        super(TensorMove, self).__init__(source_info)
        self.dst = tm_dst
        self.src = tm_src
        self.block_mode = tm_block_mode
        if self.block_mode == '':
            self.block_mode = 'm'
        self.nburst = tm_nburst
        self.burst_len = tm_burst_len
        self.dst_stride = tm_dst_stride
        self.src_stride = tm_src_stride
        self.deqscale = deqscale
        self.sid = sid
        self.relu = relu
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        if self.pad_mode is None:
            self.pad_mode = 0
        if self.pad_value is None:
            self.pad_value = 0
        self.onthefly_mode = onthefly_mode
        self.src_onthefly = src_onthefly
        self.src_onthefly_stride = src_onthefly_stride

    def eval_(self, context): # pylint: disable=R0914
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        src_align, dst_align = dma_align_fn(self.src, self.dst)

        temp_env = TempEnv()

        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, src_align, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, dst_align, access_mode='w')

        encoder = context.encoder
        param = encoder.new_param()

        param.xm, deq_scale_val = self.create_gpr_x_m(context, temp_env)

        param.src_mem_id, param.dst_mem_id, src_scope_name = \
            _get_src_dst_mem_id(self.src, self.dst, self.block_mode)

        param.conv_relu = self.get_conv_relu_param(
            context, temp_env, src_scope_name, deq_scale_val)

        param.pad = self.pad_mode
        param.xd = xd_idx
        param.xn = xn_idx

        self.set_spr_padding(context)

        instr = encoder.gen_dma_mov(param)
        context.model.step(instr)
        temp_env.check_mem_access(context.model)

        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

    def set_spr_padding(self, context):
        """set spr PADDING

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        pad_value = context.evaluate_expr(self.pad_value)
        if self.pad_mode == 1:
            pad_value |= pad_value << _PAD_VALUE_SHIFT_BIT_POS
        context.model.write_spr('PADDING', pad_value)

    def write_deqscale_buffer(self, context, temp_env, deq_scale_val):
        """write deqscale buffer

        Parameters
        ----------
        context : the stack context
        temp_env : the temp environment
        deq_scale_val: SPR DEQSCALE value

        Returns
        -------
        deq_scale_val
        """
        _, deq_addr, _, _ = copy_tensor_to_model(
            context, temp_env, self.deqscale, 32, True)
        deq_addr += context.evaluate_expr(self.deqscale.offset) * \
                    DTYPE_SIZE[self.deqscale.dtype]

        # unit: 32B
        deq_spr = deq_addr // ONE_BLK_SIZE

        deq_spr |= int(self.relu) << SPR_CONFIG_BIT_LEN

        deq_scale_val |= deq_spr
        return deq_scale_val

    def get_conv_relu_param(self, context, temp_env, src_scope_name,
                            deq_scale_val):
        """get conv_relu_param

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        conv_relu_param
        """
        conv_relu_param = 0

        if (self.src.dtype, self.dst.dtype) == ('float32', 'float16'):
            if self.relu:
                conv_relu_param = CONV_F322F16_IS_RELU
            else:
                conv_relu_param = CONV_F322F16_NO_RELU

        elif (self.src.dtype, self.dst.dtype) == ('int32', 'float16'):
            if is_tensor(self.deqscale):
                conv_relu_param = CONV_S322F16_VECTOR_QUANT
                deq_scale_val = self.write_deqscale_buffer(context, temp_env,
                                                           deq_scale_val)
            else:
                conv_relu_param = CONV_S322F16_QUANT
                quant_param = context.evaluate_expr(self.deqscale)
                binary_value = cvt_float_to_uint(self.dst.dtype, quant_param)
                deq_scale_val |= binary_value
        elif (self.src.dtype, self.dst.dtype) == ('float16', 'float32'):
            conv_relu_param = CONV_F162F32_NO_RELU
        else:
            conv_relu_param, deq_scale_val = self.get_crd_expansion(
                context, temp_env, deq_scale_val, src_scope_name)



        context.model.write_spr('DEQSCALE', deq_scale_val)

        return conv_relu_param

    def get_crd_expansion(self, context, temp_env, deq_scale_val,
                          src_scope_name):
        """get conv_relu_param and deq_scale_val based on different src/dst type

        Parameters
        ----------
        context : the stack context
        temp_env : the temp environment
        deq_scale_val: DEQSCALE value
        src_scope_name: tranformed src scope name

        Returns
        -------
        conv_relu_param
        deq_scale_val
        """
        conv_relu_param = 0
        if self.src.dtype == self.dst.dtype and self.relu:
            conv_relu_param = NO_CONV_IS_RELU
        elif src_scope_name in ('L0C16', 'L0C16V') and \
                self.deqscale is not None:
            conv_relu_param = CONV_L0C16_DEQ
            quant_param = context.evaluate_expr(self.deqscale)
            binary_value = cvt_float_to_uint(self.dst.dtype, quant_param)
            deq_scale_val |= binary_value
        elif self.src.dtype == 'int32' and self.dst.dtype in ('int8', 'uint8'):
            conv_relu_param, deq_scale_val = self.get_crd_param(
                context, temp_env, deq_scale_val, CONV_S322B8_DEQ,
                VALUE_BI_1001)
        elif (self.src.dtype, self.dst.dtype) == ('int32', 'float16'):
            conv_relu_param, deq_scale_val = self.get_crd_param(
                context, temp_env, deq_scale_val, VALUE_BI_1010,
                VALUE_BI_1011)
        elif (self.src.dtype, self.dst.dtype) == ('int32', 'int16'):
            conv_relu_param, deq_scale_val = self.get_crd_param(
                context, temp_env, deq_scale_val, VALUE_BI_1100,
                VALUE_BI_1101)

        return conv_relu_param, deq_scale_val

    def get_crd_param(self, context, temp_env,   # pylint: disable=R0913
                      deq_scale_val, cr_t, cr_s):
        """get conv_relu_param and deq_scale_val

        Parameters
        ----------
        context : the stack context
        temp_env : the temp environment
        deq_scale_val: DEQSCALE value
        cr_t: conv relu param tensor
        cr_s: conv relu param scalar/int

        Returns
        -------
        conv_relu_param
        deq_scale_val
        """
        if is_tensor(self.deqscale):
            conv_relu_param = cr_t
            deq_scale_val = self.write_deqscale_buffer(context, temp_env,
                                                       deq_scale_val)
        elif (isinstance(self.deqscale, int) or is_scalar(self.deqscale)):
            conv_relu_param = cr_s
            deq_scale_val = context.evaluate_expr(self.deqscale)
            deq_scale_val |= int(self.relu) << SHIFT_BIT_POS_47

        return conv_relu_param, deq_scale_val

    def create_gpr_x_m(self, context, temp_env):
        """get param.xm

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xm_idx
        """
        xm_idx = temp_env.alloc_register()
        _check_params_dma_instr(context.evaluate_expr(self.sid),
                                context.evaluate_expr(self.nburst),
                                context.evaluate_expr(self.burst_len),
                                context.evaluate_expr(self.src_stride),
                                context.evaluate_expr(self.dst_stride))

        xm_v = 0
        xm_v |= context.evaluate_expr(self.sid)
        if self.src.scope == scope_cc and self.dst.scope == scope_ubuf:
            xm_v |= self.onthefly_mode << SHIFT_BIT_POS_2
        xm_v |= context.evaluate_expr(self.nburst) << _NBURST_SHIFT_BIT_POS
        xm_v |= context.evaluate_expr(self.burst_len) << _BURST_SHIFT_BIT_POS
        xm_v |= context.evaluate_expr(self.src_stride) << \
                _SRC_STRIDE_SHIFT_BIT_POS
        xm_v |= context.evaluate_expr(self.dst_stride) << \
                _DST_STRIDE_SHIFT_BIT_POS

        deq_scale_val = 0
        if self.src.scope == scope_cc and self.dst.scope == scope_ubuf \
                and self.onthefly_mode > 0:
            xm_v |= self.src_onthefly_stride << SHIFT_BIT_POS_56
            # align is 65536 for dst and src_onthefly cannot in the same 64KB
            _, onthefly_addr, _, _ = copy_tensor_to_model(
                context, temp_env, self.src_onthefly, 65536, False,
                access_mode='r')
            onthefly_addr += context.evaluate_expr(self.src_onthefly.offset)\
                             * DTYPE_SIZE[self.src_onthefly.dtype]
            onthefly_addr = onthefly_addr // ONE_BLK_SIZE
            deq_scale_val |= onthefly_addr << SHIFT_BIT_POS_48

        context.model.write_gpr(xm_idx, xm_v)

        return xm_idx, deq_scale_val


class Load3DV1(STMT):
    """Load3DV1 instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, dst, src, padding, fm_h, fm_w, c_index,
                 fetch_w, fetch_h, left_top_w, left_top_h, stride_w, stride_h,
                 filter_w, filter_h, dilation_filter_w, dilation_filter_h,
                 jump_offset, repeat_mode, repeat_time, _csize, pad_value):
        # pylint: disable=R0913, R0914
        super(Load3DV1, self).__init__(source_info)
        self.dst = dst
        self.src = src
        self.padding = padding
        self.fm_h = fm_h
        self.fm_w = fm_w
        self.c_index = c_index
        self.fetch_w = fetch_w
        self.fetch_h = fetch_h
        self.left_top_w = left_top_w
        self.left_top_h = left_top_h
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.filter_w = filter_w
        self.filter_h = filter_h
        self.dilation_filter_w = dilation_filter_w
        self.dilation_filter_h = dilation_filter_h
        self.jump_offset = jump_offset
        self.repeat_mode = repeat_mode
        self.repeat_time = repeat_time
        self._csize = _csize
        self.pad_value = pad_value

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        temp_env = TempEnv()

        dst_scope = TikUtil.get_storage_scope(self.dst.scope)

        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, ALIGNED_ADDR, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGNED_ADDR, access_mode='w')

        self.set_spr_padding(context)

        self.set_spr_fmatrix(context)

        if self._csize:
            ctrl = context.model.read_spr('CTRL')
            ctrl |= 1 << _CTRL_SHIFT_BIT_POS
            context.model.write_spr('CTRL', ctrl)

        param = context.encoder.new_param()
        if dst_scope == 'L0B':
            param.type = _LOAD2D_DTYPE_ID_L0B[self.src.dtype]
        else:
            param.type = _LOAD2D_DTYPE_ID[self.src.dtype]
        param.dst_mem_id = _LOAD3D_DST_ID[dst_scope]
        param.csize = self._csize
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = self.create_gpr_x_m(context, temp_env)
        param.xt = self.create_gpr_x_t(context, temp_env)

        instr = context.encoder.gen_dma_ld_3d(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model)

        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

    def create_gpr_x_m(self, context, temp_env):
        """create register x_m

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xm_idx
        """
        c_idx = context.evaluate_expr(self.c_index)
        fetch_h = context.evaluate_expr(self.fetch_h)
        fetch_w = context.evaluate_expr(self.fetch_w)
        left_top_w = context.evaluate_expr(self.left_top_w)
        left_top_h = context.evaluate_expr(self.left_top_h)
        # check params
        check_integer_in_range(c_idx, range(MAX_C1_INDEX),
                               "c1_index should be in the range of [0, 4095],"
                               " input value is %s" % str(c_idx))
        _load3dv1_col2img_param_check(fetch_h, fetch_w, left_top_h, left_top_w)

        xm_idx = temp_env.alloc_register()

        x_m = c_idx
        x_m |= fetch_w << _FETCH_W_SHIFT_BIT_POS
        x_m |= fetch_h << _FETCH_H_SHIFT_BIT_POS
        x_m |= (left_top_w & MAX_ADDR) << _LEFT_TOP_W_SHIFT_BIT_POS
        x_m |= (left_top_h & MAX_ADDR) << _LEFT_TOP_H_SHIFT_BIT_POS

        context.model.write_gpr(xm_idx, x_m)

        return xm_idx

    def create_gpr_x_t(self, context, temp_env):
        """create register x_t

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xt_idx
        """
        stride_w = context.evaluate_expr(self.stride_w)
        stride_h = context.evaluate_expr(self.stride_h)
        dilation_w = context.evaluate_expr(self.dilation_filter_w)
        dilation_h = context.evaluate_expr(self.dilation_filter_h)
        k_w = context.evaluate_expr(self.filter_w)
        k_h = context.evaluate_expr(self.filter_h)
        repeat = context.evaluate_expr(self.repeat_time)
        jump_offset = context.evaluate_expr(self.jump_offset)
        repeat_mode = context.evaluate_expr(self.repeat_mode)
        # check params
        _load3d_v1_v2_col2img_param_check(stride_w, stride_h,
                                          k_w, k_h, dilation_w, dilation_h)
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_time should be in the range of [0, 255],"
            " input repeat_time is %s" % str(repeat))
        check_integer_in_range(
            jump_offset, range(MIN_JUMP_OFFSET, MAX_JUMP_OFFSET),
            "jump_offset should be in the range of [1, 127],"
            " input jump_offset is %s" % str(jump_offset))
        check_integer_in_range(
            repeat_mode, range(MAX_REPEAT_MODE),
            "repeat_mode should be 0 or 1, input repeat_mode: %d" %
            self.repeat_mode)

        xt_idx = temp_env.alloc_register()

        x_t = stride_w
        x_t |= stride_h << _STRIDE_H_SHIFT_BIT_POS
        x_t |= k_w << _FILTER_W_SHIFT_BIT_POS
        x_t |= k_h << _FILTER_H_SHIFT_BIT_POS
        x_t |= dilation_w << _DILATION_W_SHIFT_BIT_POS
        x_t |= dilation_h << _DILATION_H_SHIFT_BIT_POS
        x_t |= jump_offset << _JUMP_OFFSET_SHIFT_BIT_POS
        x_t |= repeat_mode << REPEAT_MODE_SHIFT_BIT
        x_t |= repeat << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx

    def set_spr_fmatrix(self, context):
        """set special register FMATRIX

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        padding = [context.evaluate_expr(pad) for pad in self.padding]
        pad_l, pad_r, pad_top, pad_bot = padding
        _check_pad_list(padding)

        fm_h = context.evaluate_expr(self.fm_h)
        fm_w = context.evaluate_expr(self.fm_w)
        check_integer_in_range(fm_w, range(MIN_TENSOR_WIDTH, MAX_TENSOR_WIDTH),
                               "l1_w should be in the range of [1, 32767], "
                               "input value is %s" % str(fm_w))
        check_integer_in_range(fm_h, range(MIN_TENSOR_HEIGHT, MAX_TENSOR_HEIGHT),
                               "l1_h should be in the range of [1, 32767],"
                               "input value is %s" % str(fm_h))

        fmatrix = fm_w
        fmatrix |= fm_h << _FM_H_SHIFT_BIT_POS
        fmatrix |= pad_l << _PAD_L_SHIFT_BIT_POS
        fmatrix |= pad_r << _PAD_R_SHIFT_BIT_POS
        fmatrix |= pad_top << _PAD_TOP_SHIFT_BIT_POS
        fmatrix |= pad_bot << _PAD_BOT_SHIFT_BIT_POS

        context.model.write_spr('FMATRIX', fmatrix)

    def set_spr_padding(self, context):
        """set special register PADDING

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        pad_value = context.evaluate_expr(self.pad_value)

        spr_pad_value = 0
        if self.dst.dtype == 'float16':
            spr_pad_value = cvt_float_to_uint('float16', pad_value)
        else:
            spr_pad_value = pad_value
            spr_pad_value |= pad_value << _PAD_VALUE_SHIFT_BIT_POS
        context.model.write_spr('PADDING', spr_pad_value)


class Col2Img(STMT):
    """Col2Img instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, dst, src, pad, l1_h, l1_w, fetch_filter_w,
                 fetch_filter_h, left_top_w, left_top_h, stride_w, stride_h,
                 filter_w, filter_h, dilation_filter_w, dilation_filter_h,
                 repeat_time):
        # pylint: disable=R0913, R0914
        super(Col2Img, self).__init__(source_info)
        self.dst = dst
        self.src = src
        self.pad = pad
        self.l1_h = l1_h
        self.l1_w = l1_w
        self.fetch_filter_w = fetch_filter_w
        self.fetch_filter_h = fetch_filter_h
        self.left_top_w = left_top_w
        self.left_top_h = left_top_h
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.filter_w = filter_w
        self.filter_h = filter_h
        self.dilation_filter_w = dilation_filter_w
        self.dilation_filter_h = dilation_filter_h
        self.repeat_time = repeat_time

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        temp_env = TempEnv()

        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, ALIGNED_ADDR, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGNED_ADDR, access_mode='rw')

        self.set_spr_fcol2img(context)

        param = context.encoder.new_param()
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = self.create_gpr_x_m(context, temp_env)
        param.xt = self.create_gpr_x_t(context, temp_env)
        param.type = 1
        if self.dst.dtype == 'float32':
            param.type = 0

        instr = context.encoder.gen_dma_col2_img(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model)
        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

    def set_spr_fcol2img(self, context):
        """set special register fcol2img

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        padding = [context.evaluate_expr(pad) for pad in self.pad]
        pad_l, pad_r, pad_top, pad_bot = padding
        _check_pad_list(padding)
        l1_h = context.evaluate_expr(self.l1_h)
        l1_w = context.evaluate_expr(self.l1_w)

        fcol2img = l1_w
        fcol2img |= l1_h << _L1_H_SHIFT_BIT_POS
        fcol2img |= pad_l << _PAD_L_SHIFT_BIT_POS
        fcol2img |= pad_r << _PAD_R_SHIFT_BIT_POS
        fcol2img |= pad_top << _PAD_TOP_SHIFT_BIT_POS
        fcol2img |= pad_bot << _PAD_BOT_SHIFT_BIT_POS

        context.model.write_spr('FCOL2IMG', fcol2img)

    def create_gpr_x_m(self, context, temp_env):
        """create general purpose register x_m

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xm_idx
        """
        fetch_w = context.evaluate_expr(self.fetch_filter_w)
        fetch_h = context.evaluate_expr(self.fetch_filter_h)
        left_top_w = context.evaluate_expr(self.left_top_w)
        left_top_h = context.evaluate_expr(self.left_top_h)
        _load3dv1_col2img_param_check(fetch_h, fetch_w, left_top_w, left_top_h)

        xm_idx = temp_env.alloc_register()
        x_m = 0
        x_m |= fetch_w << _FETCH_W_SHIFT_BIT_POS
        x_m |= fetch_h << _FETCH_H_SHIFT_BIT_POS
        x_m |= left_top_w << _LEFT_TOP_W_SHIFT_BIT_POS
        x_m |= left_top_h << _LEFT_TOP_H_SHIFT_BIT_POS

        context.model.write_gpr(xm_idx, x_m)

        return xm_idx

    def create_gpr_x_t(self, context, temp_env):
        """create general purpose register x_t

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xt_idx
        """
        stride_w = context.evaluate_expr(self.stride_w)
        stride_h = context.evaluate_expr(self.stride_h)
        filter_w = context.evaluate_expr(self.filter_w)
        filter_h = context.evaluate_expr(self.filter_h)
        dilation_filter_w = context.evaluate_expr(self.dilation_filter_w)
        dilation_filter_h = context.evaluate_expr(self.dilation_filter_h)
        repeats = context.evaluate_expr(self.repeat_time)
        # check repeat
        check_integer_in_range(
            repeats, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255],"
            " input value is %s" % str(repeats))
        _load3d_v1_v2_col2img_param_check(stride_w, stride_h,
                                          filter_w, filter_h,
                                          dilation_filter_w, dilation_filter_h)

        xt_idx = temp_env.alloc_register()
        x_t = stride_w
        x_t |= stride_h << _STRIDE_H_SHIFT_BIT_POS
        x_t |= filter_w << _FILTER_W_SHIFT_BIT_POS
        x_t |= filter_h << _FILTER_H_SHIFT_BIT_POS
        x_t |= dilation_filter_w << _DILATION_W_SHIFT_BIT_POS
        x_t |= dilation_filter_h << _DILATION_H_SHIFT_BIT_POS
        x_t |= 1 << _SHIFT_BIT_POS
        x_t |= repeats << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class BroadcastUB(STMT):
    """BroadcastUB instruction"""
    def __init__(self, source_info, dst, src, *strides):
        super(BroadcastUB, self).__init__(source_info)
        self.dst = dst
        self.src = src
        self.strides = strides

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        src_dtype = self.src.dtype
        dst_dtype = self.dst.dtype

        temp_env = TempEnv()

        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, ALIGN_SRC, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGN_DST, access_mode='w')

        conv = 0
        if _BRC_DST_TYPE_ID[dst_dtype] and src_dtype != dst_dtype:
            conv = 1

        param = context.encoder.new_param()
        param.dst_mem_id = _BRC_DST_TYPE_ID[dst_dtype]
        param.src_mem_id = 0
        param.conv = conv
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = self.create_gpr_x_m(context, temp_env)

        instr = context.encoder.gen_dma_brc(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model)
        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

    def create_gpr_x_m(self, context, temp_env):
        """create general purpose register x_m

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xm_idx
        """
        strides = [context.evaluate_expr(s) for s in self.strides]
        # all param 32B unit
        nburst, burst_len, src_gap, dst_gap = strides
        # check params
        check_integer_in_range(
            nburst, range(MIN_NBURST, MAX_NBURST_SINGLE_BYTE),
            "nburst should be in the range of [1, 255],"
            " input value is %s" % str(nburst))
        check_integer_in_range(
            burst_len, range(MIN_BURST_LEN, MAX_BURST_LEN_SINGLE_BYTE),
            "burst_len should be in the range of [1, 255],"
            " input value is %s" % str(burst_len))
        check_integer_in_range(
            dst_gap, range(MAX_BLK_STRIDE_SINGLE_BYTE),
            "dst_blk_stride should be in the range of [0, 255],"
            " input value is %s" % str(dst_gap))
        check_integer_in_range(
            src_gap, range(MAX_BLK_STRIDE_SINGLE_BYTE),
            "src_blk_stride should be in the range of [0, 255],"
            " input value is %s" % str(src_gap))

        xm_idx = temp_env.alloc_register()

        x_m = nburst
        x_m |= burst_len << _BURST_LEN_SHIFT_BIT_POS
        x_m |= src_gap << _SRC_GAP_SHIFT_BIT_POS
        x_m |= dst_gap << _DST_GAP_SHIFT_BIT_POS

        context.model.write_gpr(xm_idx, x_m)

        return xm_idx


class MMAD(STMT):
    """MMAD instruction"""
    # pylint: disable=R0902
    def __init__(self,
                 source_info,
                 dst,
                 tensor_a,
                 tensor_b,
                 matrix_m,
                 matrix_k,
                 matrix_n,
                 initial_by_L0C_matrix,
                 mmad_fm_offset=0,
                 mmad_en_weight_offset=False,
                 mmad_smask=None,
                 mmad_en_small_channel=False,
                 mmad_en_small_k=False,
                 mmad_en_ssparse=False,
                 mmad_en_winograd_a=False,
                 mmad_en_winograd_b=False):
        # pylint: disable=R0913, R0914
        super(MMAD, self).__init__(source_info)
        self.dst = dst
        self.tensor_a = tensor_a
        self.tensor_b = tensor_b
        self.fm_offset = mmad_fm_offset
        self.en_weight_offset = int(mmad_en_weight_offset)
        self.smask = 0 if (mmad_smask is None) else mmad_smask
        self.en_small_channel = int(mmad_en_small_channel)
        self.en_small_k = int(mmad_en_small_k)
        self.en_ssparse = int(mmad_en_ssparse)
        self.en_winograd_a = int(mmad_en_winograd_a)
        self.en_winograd_b = int(mmad_en_winograd_b)

        self.matrix_m = matrix_m
        self.matrix_k = matrix_k
        self.matrix_n = matrix_n

        self.init_by_l0c = initial_by_L0C_matrix

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        dst_type = self.dst.dtype
        src_type = self.tensor_a.dtype

        temp_env = TempEnv()

        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.tensor_a,
            ALIGN_TENSOR, access_mode='r')
        xm_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.tensor_b,
            ALIGN_TENSOR, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGN_DST, access_mode='rw')

        param = context.encoder.new_param()
        param.type = _MMAD_TYPE_BITS[(src_type, self.tensor_b.dtype, dst_type)]
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = xm_idx
        param.xt = self.create_gpr_x_t(context, temp_env)

        instr = context.encoder.gen_mmad(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model)

        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

    def create_gpr_x_t(self, context, temp_env):
        """create general purpose register x_t

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xt_idx
        """
        matrix_m = context.evaluate_expr(self.matrix_m)
        matrix_k = context.evaluate_expr(self.matrix_k)
        matrix_n = context.evaluate_expr(self.matrix_n)
        # check matrix_* param
        check_integer_in_range(matrix_m, range(MAX_MATRIX),
                               "matrix_m should be in range of [0, 4095],"
                               " input value is %s" % str(matrix_m))
        check_integer_in_range(matrix_n, range(MAX_MATRIX),
                               "matrix_n should be in range of [0, 4095],"
                               " input value is %s" % str(matrix_n))
        check_integer_in_range(matrix_k, range(MAX_MATRIX),
                               "matrix_k should be in range of [0, 4095],"
                               " input value is %s" % str(matrix_k))
        check_integer_in_range(context.evaluate_expr(self.init_by_l0c),
                               range(2),
                               "is_bias should be 0 or 1")

        xt_idx = temp_env.alloc_register()

        if context.evaluate_expr(self.init_by_l0c):
            l0c_bit = 0
        else:
            l0c_bit = 1

        x_t = matrix_m
        x_t |= matrix_k << MMAD_MATRIX_K_POS
        x_t |= matrix_n << MMAD_MATRIX_N_POS

        # fm offset [43:36]
        bit_pos, bit_len = 36, 8
        TikCheckUtil.check_in_range(
            self.fm_offset, range(int(math.pow(2, bit_len) - 1)))
        x_t |= (self.fm_offset << bit_pos)

        # smask [50:44]
        bit_pos, bit_len = 44, 7
        TikCheckUtil.check_in_range(
            self.smask, range(int(math.pow(2, bit_len) - 1)))
        if self.smask is not None:
            x_t |= (self.smask << bit_pos)

        # en_winograd_a
        x_t |= (self.en_winograd_a << MMAD_EN_WINOGRAD_A_POS)
        # en_winograd_b
        x_t |= (self.en_winograd_b << MMAD_EN_WINOGRAD_B_POS)
        # en_weight_matrix_offset
        x_t |= (self.en_weight_offset << MMAD_EN_WEIGHT_OFFSET_POS)
        # en_ssparse
        x_t |= (self.en_ssparse << MMAD_EN_SSPARSE_POS)
        # C inital value control
        x_t |= l0c_bit << MMAD_L0C_BIT_POS
        if self.en_small_channel != 0 or self.en_small_k != 0:
            sys.stderr.write(
                "[INFO]: small-channel & small-k are not supported "
                "in debug flow yet!\n"
            )

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class DepthwiseConv(STMT):
    """DepthwiseConv instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, dst, src0, src1, pad_mode, l1_h, l1_w,
                 store_high_half, feature_offset, weight_offset, pad_value):
        # pylint: disable=R0913
        super(DepthwiseConv, self).__init__(source_info)
        self.dst = dst
        self.feature_map = src0
        self.weight = src1
        self.pad_mode = pad_mode
        self.l1_h = l1_h
        self.l1_w = l1_w
        self.store_high_half = store_high_half
        self.feature_offset = feature_offset
        self.weight_offset = weight_offset
        self.pad_value = pad_value

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        # check params
        check_depthwise_conv_params(
            self.feature_map, context.evaluate_expr(self.pad_mode),
            context.evaluate_expr(self.l1_h), context.evaluate_expr(self.l1_w),
            context.evaluate_expr(self.feature_offset))
        # check l1_w
        check_depthwise_conv_l1_w(context.evaluate_expr(self.pad_mode),
                                  context.evaluate_expr(self.l1_w))

        temp_env = TempEnv()

        dst_align = 1024

        if self.dst.dtype == 'float16':
            dst_align = 512

        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.feature_map, 16, True, access_mode='r')
        xm_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.weight, 512, True, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, dst_align, True, access_mode='w')

        param = context.encoder.new_param()
        param.type = _MMAD_TYPE_BITS[(self.feature_map.dtype, self.weight.dtype,
                                      self.dst.dtype)]
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = xm_idx
        param.xt = self.create_gpr_x_t(context, temp_env)

        instr = context.encoder.gen_dp(param)
        # manually add #h param since encoder is not the latest ISA version

        instr |= self.store_high_half

        self.set_spr_padding(context)

        context.model.step(instr)
        temp_env.check_mem_access(context.model)

        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

    def set_spr_padding(self, context):
        """set spr PADDING

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        if self.pad_value is not None:
            spr_pad_value = 0
            pad_value = context.evaluate_expr(self.pad_value)

            if context.evaluate_expr(self.pad_mode) != 0:
                if self.feature_map.dtype == 'float16':
                    spr_pad_value = cvt_float_to_uint('float16', pad_value)
                else:
                    spr_pad_value = pad_value
                    spr_pad_value |= pad_value << _PAD_VALUE_SHIFT_BIT_POS
                context.model.write_spr('PADDING', spr_pad_value)

    def create_gpr_x_t(self, context, temp_env):
        """create general purpose register x_t

        Parameters
        ----------
        context : the stack context
        temp_env : the temp environment

        Returns
        -------
        xt_idx
        """
        l1_h = context.evaluate_expr(self.l1_h)
        l1_w = context.evaluate_expr(self.l1_w)
        feature_offset = context.evaluate_expr(self.feature_offset)
        if feature_offset < 0:
            feature_offset = cvt_float_to_uint("int8", feature_offset)
        pad_mode = context.evaluate_expr(self.pad_mode)

        xt_idx = temp_env.alloc_register()
        x_t = l1_w
        x_t |= l1_h << _DEPTH_L1_H_SHIFT_POS
        x_t |= feature_offset << _FEATURE_OFFSET_SHIFT_POS
        x_t |= pad_mode << _PAD_MODE_SHIFT_POS



        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class WriteSPR(STMT):
    """WriteSPR instruction"""
    def __init__(self, source_info, spr_name, value):
        super(WriteSPR, self).__init__(source_info)
        self.spr_name = spr_name
        self.value = value

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        value = context.evaluate_expr(self.value)
        np_value = np.asarray(value)
        value = cvt_float_to_uint(str(np_value.dtype), np_value)
        context.model.write_spr(self.spr_name, value)


def _ones_cnt(value, dtype):
    """count bit number of one

    Parameters
    ----------
    value : count value
    dtype: value dtype

    Returns
    -------
    nums_ones: number of bit one
    """
    # pylint: disable=E1101
    # pylint cannot recognize np member, so disable it.
    tmp_arr = np.asarray([value], dtype=dtype)
    tmp_arr = tmp_arr.view('uint8')
    bits_arr = np.unpackbits(tmp_arr)
    num_ones = bits_arr.sum().item()
    return num_ones


class ScalarSingleOp(STMT):
    """ScalarSingleOp instruction"""
    def __init__(self, source_info, name, dst, src):
        super(ScalarSingleOp, self).__init__(source_info)
        self.name = name
        self.dst = dst
        self.src = src

    def debug_bcnt0(self, src_value):
        """debug for bitcount0 instruction

        Parameters
        ----------
        src_value : the value of src

        Returns
        -------
        dst_value: the value of dst
        """
        if isinstance(self.src, (int, float)):
            dst_value = UINT64_BIT - _ones_cnt(int(src_value), 'uint64')
        else:
            dst_value = UINT64_BIT - _ones_cnt(src_value, self.src.dtype)
        return dst_value

    def debug_bcnt1(self, src_value):
        """debug for bitcount0 instruction

        Parameters
        ----------
        src_value : the value of src

        Returns
        -------
        dst_value: the value of dst
        """
        if isinstance(self.src, (int, float)):
            dst_value = _ones_cnt(int(src_value), 'uint64')
        else:
            dst_value = _ones_cnt(src_value, self.src.dtype)
        return dst_value

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        src_value = context.evaluate_expr(self.src)
        try:
            if self.name == 'sqrt':
                dst_value = np.sqrt(np.abs(src_value)).item()
            elif self.name == 'abs':
                dst_value = np.abs(src_value).item()
            elif self.name == 'bcnt0':
                dst_value = self.debug_bcnt0(src_value)
            elif self.name == 'bcnt1':
                dst_value = self.debug_bcnt1(src_value)
            elif self.name == 'clz':
                if isinstance(self.src, (int, float)):
                    dst_value = format(int(src_value), '064b').find('1')
                else:
                    dst_value = format(src_value, '064b').find('1')
                if dst_value <= -1:
                    # '1' is not found, thus all 64 bits are 0
                    dst_value = UINT64_BIT
            else:
                TikCheckUtil.raise_error('unsupported op {}'.format(self.name))
        except FloatingPointError:
            TikCheckUtil.raise_error(
                'got float point error in %s with value %s' % (
                    self.name, str(src_value)))

        context.update_var(self.dst.debug_var, dst_value)


class ScalarBinaryOp(STMT):
    """ScalarBinary instruction for debug"""
    def __init__(self, source_info, name, dst, src0, src1):
        # pylint: disable=R0913
        super(ScalarBinaryOp, self).__init__(source_info)
        self.name = name
        self.dst = dst
        self.src0 = src0
        self.src1 = src1

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        src0_value = context.evaluate_expr(self.src0)
        src1_value = context.evaluate_expr(self.src1)
        try:
            if self.name == 'max':
                dst_value = np.max([src0_value, src1_value]).item()
            elif self.name == 'min':
                dst_value = np.min([src0_value, src1_value]).item()
            else:
                TikCheckUtil.raise_error('unsupported op {}'.format(self.name))
        except FloatingPointError:
            TikCheckUtil.raise_error(
                'got float point error in {} with value {}, {}'.format(
                    self.name, src0_value, src1_value))

        context.update_var(self.dst.debug_var, dst_value)


class ScalarConv(STMT):
    """ScalarConv instruction"""
    def __init__(self, source_info, round_mode, dst, src):
        super(ScalarConv, self).__init__(source_info)
        if round_mode == 'ceiling':
            round_mode = 'ceil'
        self.round_mode = round_mode
        self.dst = dst
        self.src = src

        if self.round_mode in ('', 'none'):
            self.conv_id = _SCALAR_CONV_ID[self.src.dtype, self.dst.dtype]
        else:
            self.conv_id = _SCALAR_CONV_ID[self.src.dtype, self.dst.dtype, self.round_mode]

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        src_value = context.evaluate_expr(self.src)
        temp_env = TempEnv()
        xn_idx = temp_env.alloc_register()
        # conv src to uint values
        src_bit_width = get_dtype_bit_width(self.src.dtype)
        uint_value = getattr(np, self.src.dtype)(src_value).view('uint' + src_bit_width).item()
        context.model.write_gpr(xn_idx, uint_value)

        xd_idx = temp_env.alloc_register()

        param = context.encoder.new_param()
        param.type = self.conv_id
        param.xd = xd_idx
        param.xn = xn_idx

        instr = context.encoder.gen_conv(param)

        context.model.step(instr)

        dst_value_uint = context.model.read_gpr(xd_idx)
        dst_bit_width = get_dtype_bit_width(self.dst.dtype)
        # it is safe to cast fp16 in python, since fp16 to fp32 is round trip safe
        dst_value = getattr(np, 'uint' + dst_bit_width)(dst_value_uint).view(self.dst.dtype).item()

        context.update_var(self.dst.debug_var, dst_value)


class VMS4SR2Scalar(STMT):
    """VMS4SR2Scalar instruction"""
    def __init__(self, source_info, scalar_arr):
        super(VMS4SR2Scalar, self).__init__(source_info)
        self.scalar_arr = scalar_arr

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        sr_value = context.model.read_spr('VMS4_SR')
        # vms4 bit shift
        value = 2**13 - 1
        # bit shift unit for vms4 is 13
        bit_len = 13
        for i, scalar in enumerate(self.scalar_arr):
            temp_value = (sr_value >> (i*bit_len)) & value
            context.update_var(scalar.debug_var, temp_value)


class SetL0SetValue(STMT):
    """SetL0SetValue instruction"""
    def __init__(self, source_info, value, dtype):
        super(SetL0SetValue, self).__init__(source_info)
        self.value = value
        self.dtype = dtype

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        value = context.evaluate_expr(self.value)
        bin_value = reinterpret_type(self.dtype, 'uint16', value)
        context.model.write_spr('L0_SET_VALUE', bin_value)


class SetCtrlSPR(STMT):
    """SetCtrlSPR instruction"""
    def __init__(self, source_info, value, mask, begin):
        super(SetCtrlSPR, self).__init__(source_info)
        self.value = value
        if isinstance(self.value, bool):
            self.value = int(self.value)
        self.mask = mask
        self.begin = begin

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        value = context.evaluate_expr(self.value)
        spr_value = context.model.read_spr('CTRL')
        # clear the bits first
        spr_value &= self.mask
        spr_value |= (value << self.begin)
        context.model.write_spr('CTRL', spr_value)


class GetCtrlSPR(STMT):
    """GetCtrlSPR instruction"""
    def __init__(self, source_info, scalar_arr, mask, begin):
        super(GetCtrlSPR, self).__init__(source_info)
        self.scalar_arr = scalar_arr
        self.mask = mask
        self.begin = begin

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        spr_value = context.model.read_spr('CTRL')
        # clear the bits first
        spr_value &= self.mask
        spr_value = spr_value >> self.begin
        context.update_var(self.scalar_arr.debug_var, spr_value)


class Set2D(STMT):
    """Set2D instruction"""
    def __init__(self, source_info, dst, repeat_times, value=None):
        super(Set2D, self).__init__(source_info)
        self.dst = dst
        self.repeat_time = repeat_times
        self.value = value

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        scope_name = TikUtil.get_storage_scope(self.dst.scope)
        dtype_size_ = get_dtype_size(self.dst.dtype)

        temp_env = TempEnv()

        dst_align = 512
        if scope_name == 'L1':
            dst_align = dtype_size_

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, dst_align, access_mode='rw')

        if self.value is not None:
            value = context.evaluate_expr(self.value)
            bin_value = reinterpret_type(self.dst.dtype, 'uint16', value)
            context.model.write_spr('L0_SET_VALUE', bin_value)

        param = context.encoder.new_param()
        param.dst_mem_id = _SET_2D_DST_ID[scope_name]
        param.xd = xd_idx
        param.xm = self.create_gpr_x_t(context, temp_env)

        instr = context.encoder.gen_dma_set_2d(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model)

        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)

    def create_gpr_x_t(self, context, temp_env):
        """create general purpose register x_t

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xm_idx
        """
        repeat = context.evaluate_expr(self.repeat_time)
        # check repeat
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255],"
            " input value is %s" % str(repeat))

        xm_idx = temp_env.alloc_register()
        x_m = repeat
        context.model.write_gpr(xm_idx, x_m)

        return xm_idx


class Load3DV2(STMT):
    """Load3DV2 instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, dst, src,
                 pad_list, l1_height, l1_width, channel_size,
                 k_extension, m_extension, k_start_pt,
                 m_start_pt, stride_w, stride_h, filter_w,
                 filter_h, dilation_filter_w, dilation_filter_h,
                 en_transpose=False, en_small_k=False, pad_value=None):
        # pylint: disable=R0913, R0914
        super(Load3DV2, self).__init__(source_info)
        self.dst = dst
        self.src = src
        self.pad_list = pad_list
        self.fm_h = l1_height
        self.fm_w = l1_width
        self.channel_size = channel_size
        self.k_ext = k_extension
        self.m_ext = m_extension
        self.k_start = k_start_pt
        self.m_start = m_start_pt
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.filter_w = filter_w
        self.filter_h = filter_h
        self.dilation_filter_w = dilation_filter_w
        self.dilation_filter_h = dilation_filter_h
        self.en_transpose = en_transpose
        self.en_small_k = en_small_k
        self.pad_value = pad_value

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        temp_env = TempEnv()

        # default dst addr align bit equals to 512
        dst_align = 512

        dst_scope = TikUtil.get_storage_scope(self.dst.scope)

        if dst_scope == 'UB':
            dst_align = 32

        # check channel size
        check_load3dv2_channel_size(context.evaluate_expr(self.channel_size),
                                    self.src.dtype)

        src_align = min(32, context.evaluate_expr(self.channel_size)*
                        DTYPE_SIZE[self.src.dtype])
        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, src_align, True, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, dst_align, True, access_mode='w')

        # write fmatrix to spr
        context.model.write_spr('FMATRIX', self.get_spr_fm(context))

        # write pad value to spr
        context.model.write_spr('PADDING', self.get_spr_pad_value(context))

        param = context.encoder.new_param()
        param.dst_mem_id = _LOAD3DV2_DST_ID[dst_scope]
        param.type = _LOAD3DV2_TYPE_ID[get_dtype_bit_width(self.src.dtype)]
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = self.create_gpr_x_m(context, temp_env)
        param.xt = self.create_gpr_x_t(context, temp_env)

        # check dilation filter size and l1_h_w size
        check_dilation_filter_size(
            context.evaluate_expr(self.filter_w),
            context.evaluate_expr(self.dilation_filter_w),
            context.evaluate_expr(self.fm_w),
            context.evaluate_expr(self.pad_list[PADDING_LEFT_IDX]),
            context.evaluate_expr(self.pad_list[PADDING_RIGHT_IDX]), "W")
        check_dilation_filter_size(
            context.evaluate_expr(self.filter_h),
            context.evaluate_expr(self.dilation_filter_h),
            context.evaluate_expr(self.fm_h),
            context.evaluate_expr(self.pad_list[PADDING_TOP_IDX]),
            context.evaluate_expr(self.pad_list[PADDING_BOT_IDX]), "H")
        # check m_extention and k_extension
        check_load3dv2_m_extension(
            context.evaluate_expr(self.filter_w),
            context.evaluate_expr(self.filter_h),
            context.evaluate_expr(self.dilation_filter_w),
            context.evaluate_expr(self.dilation_filter_h),
            [context.evaluate_expr(value) for value in self.pad_list],
            context.evaluate_expr(self.m_ext), context.evaluate_expr(self.fm_w),
            context.evaluate_expr(self.fm_h),
            context.evaluate_expr(self.stride_w),
            context.evaluate_expr(self.stride_h),
            context.evaluate_expr(self.m_start))
        check_load3dv2_k_extension(context.evaluate_expr(self.channel_size),
                                   context.evaluate_expr(self.k_ext),
                                   context.evaluate_expr(self.filter_h),
                                   context.evaluate_expr(self.filter_w),
                                   context.evaluate_expr(self.k_start),
                                   self.src.dtype)

        instr = context.encoder.gen_dma_ld_3dv2(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model)

        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)

    def get_spr_fm(self, context):
        """get special purpose register fm

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        fmatrix : spr
        """
        padding = [context.evaluate_expr(pad) for pad in self.pad_list]
        pad_l, pad_r, pad_top, pad_bot = padding
        _check_pad_list(padding)

        fm_h = context.evaluate_expr(self.fm_h)
        fm_w = context.evaluate_expr(self.fm_w)
        check_integer_in_range(fm_w, range(MIN_TENSOR_WIDTH, MAX_TENSOR_WIDTH),
                               "l1_w should be in the range of [1, 32767], "
                               "input value is %s" % str(fm_w))
        check_integer_in_range(fm_h, range(MIN_TENSOR_HEIGHT, MAX_TENSOR_HEIGHT),
                               "l1_h should be in the range of [1, 32767],"
                               "input value is %s" % str(fm_h))

        fmatrix = fm_w
        fmatrix |= fm_h << _FM_H_SHIFT_BIT_POS
        fmatrix |= pad_l << _PAD_L_SHIFT_BIT_POS
        fmatrix |= pad_r << _PAD_R_SHIFT_BIT_POS
        fmatrix |= pad_top << _PAD_TOP_SHIFT_BIT_POS
        fmatrix |= pad_bot << _PAD_BOT_SHIFT_BIT_POS

        return fmatrix

    def get_spr_pad_value(self, context):
        """get special purpose register pad value

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        spr_pad_value : spr pad value
        """
        spr_pad_value = 0
        if self.pad_value is not None:
            pad_value = context.evaluate_expr(self.pad_value)

            if self.dst.dtype == 'float16':
                spr_pad_value = cvt_float_to_uint('float16', pad_value)
            else:
                spr_pad_value = pad_value
                spr_pad_value |= pad_value << _PAD_VALUE_SHIFT_BIT_POS

        return spr_pad_value

    def create_gpr_x_m(self, context, temp_env):
        """create general purpose register x_t

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xm_idx
        """
        k_ext = context.evaluate_expr(self.k_ext)
        m_ext = context.evaluate_expr(self.m_ext)
        k_start = context.evaluate_expr(self.k_start)
        m_start = context.evaluate_expr(self.m_start)
        # check params
        _load3d_v2_params_check(k_ext, m_ext, k_start, m_start, self.src.dtype)

        xm_idx = temp_env.alloc_register()
        x_m = k_ext
        x_m |= m_ext << _M_EXTENSION_SHIFT_BIT_POS
        x_m |= k_start << _K_START_POINT_SHIFT_BIT_POS
        x_m |= m_start << _M_START_POINT_SHIFT_POS

        context.model.write_gpr(xm_idx, x_m)

        return xm_idx

    def create_gpr_x_t(self, context, temp_env):
        """create general purpose register x_t

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xt_idx
        """
        channel_size = context.evaluate_expr(self.channel_size)

        stride_w = context.evaluate_expr(self.stride_w)
        stride_h = context.evaluate_expr(self.stride_h)
        filter_w = context.evaluate_expr(self.filter_w)
        filter_h = context.evaluate_expr(self.filter_h)
        dilation_filter_w = context.evaluate_expr(self.dilation_filter_w)
        dilation_filter_h = context.evaluate_expr(self.dilation_filter_h)
        # check params
        _load3d_v1_v2_col2img_param_check(stride_w, stride_h,
                                          filter_w, filter_h,
                                          dilation_filter_w, dilation_filter_h)

        xt_idx = temp_env.alloc_register()
        x_t = stride_w
        x_t |= stride_h << _STRIDE_H_SHIFT_BIT_POS
        x_t |= filter_w << _FILTER_W_SHIFT_BIT_POS
        x_t |= filter_h << _FILTER_H_SHIFT_BIT_POS
        x_t |= dilation_filter_w << _DILATION_W_SHIFT_BIT_POS
        x_t |= dilation_filter_h << _DILATION_W_SHIFT_BIT_POS
        if self.en_transpose or self.en_small_k:
            x_t |= 1 << _EN_TRANSPOSE_SMALL_K_SHIFT_POS
        x_t |= channel_size << _CHANNEL_SIZE_SHIFT_BIT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class LoadSmask(STMT):
    """winograd_weight_transform instruction"""
    def __init__(self, source_info, dst, src, load_size, sid):
        # pylint: disable=R0913
        super(LoadSmask, self).__init__(source_info)
        self.dst = dst
        self.src = src
        self.load_size = load_size
        self.sid = sid

    def eval_(self, context):
        # pylint: disable=R0914
        src_align = 32
        dst_align = 32

        temp_env = TempEnv()
        xn_idx, _, _, _ = copy_tensor_to_model(context, temp_env, self.src,
                                               src_align, True, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, dst_align, True, access_mode='w')

        x_t_11 = 0
        x_t_0 = context.evaluate_expr(self.load_size) & 127
        xt_idx = temp_env.alloc_register()
        x_t = x_t_0
        x_t |= context.evaluate_expr(self.sid) << SHIFT_BIT_POS_7
        x_t |= x_t_11 << SHIFT_BIT_POS_11
        context.model.write_gpr(xt_idx, x_t)

        ld_smask_src_mem_id = {
            scope_ubuf: 1,
            scope_gm: 0
        }

        param = context.encoder.new_param()
        param.xd = xd_idx
        param.xn = xn_idx
        param.xt = xt_idx
        param.src_mem_id = ld_smask_src_mem_id[self.src.scope]

        instr = context.encoder.gen_dma_ld_smask(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model)
        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)


class LoadImage(STMT):
    """LoadImage instruction"""
    # pylint: disable=R0902, R0904
    def __init__(self, source_info, dst,  # pylint: disable=R0915
                 src0, src1, input_format, function_switch,
                 src_info, crop_info, pre_clip_info, swap_list,
                 csc_info, scf_info, post_clip_info, dtc_info, flip_mode,
                 channel_pad_info, area_pad_info,
                 stretch_info, raw_info, sid):
        # pylint: disable=R0913, R0914
        super(LoadImage, self).__init__(source_info)
        self.dst = dst
        self.src0 = src0
        if src0 is not None:
            self.src0_start = src0.offset
        self.src1 = src1
        if src1 is not None:
            self.src1_start = src1.offset
        else:
            self.src1_start = 0

        self.dtype = dst.dtype
        self.input_format = input_format
        self.function_switch = function_switch

        # src_info
        self.src_info = src_info
        check_dict_and_not_none(self.src_info, 'src_info')
        self.src_horizontal_size = src_info.get('src_horizontal_size')
        self.src_vertical_size = src_info.get('src_vertical_size')

        # crop_info
        self.crop_info = crop_info

        # pre_clip_info
        self.pre_clip_info = pre_clip_info

        # swap
        self.swap_list = swap_list

        # csc_info
        self.csc_info = csc_info

        # scf_info
        self.scf_info = scf_info

        # post_clip_info
        self.post_clip_info = post_clip_info

        # dtc_info
        self.dtc_info = dtc_info

        # flip_mode
        self.flip_mode = flip_mode

        # channel_pad_info
        self.channel_pad_info = channel_pad_info

        # area_pad_info
        self.area_pad_info = area_pad_info

        # stretch_info
        self.stretch_info = stretch_info

        # raw info
        self.raw_info = raw_info

        self.sid = sid
        self.arch_version = None

        self.crop_enbale = 0
        self.swap_enable = 0
        self.csc_enable = 0
        self.dtc_enable = 0
        self.area_pad_enable = 0
        self.channel_pad_enable = 0
        self.pre_clip_enable = 0
        self.scf_enable = 0
        self.post_clip_enable = 0
        self.flip_enable = 0
        self.stretch_enable = 0
        self.raw_enable = 0

        self.imm_input_format = 0
        self.imm_src_horizontal_size = 0
        self.imm_src_vertical_size = 0

        self.imm_dst_horizontal_size = 0
        self.imm_dst_vertical_size = 0
        self.imm_crop_horizontal_start = 0
        self.imm_crop_vertical_start = 0
        self.imm_single_line_mode = 0

        self.imm_rb_swap = 0
        self.imm_uv_swap = 0
        self.imm_ax_swap = 0

        self.csc_matrix_r0_c0 = 0
        self.csc_matrix_r0_c1 = 0
        self.csc_matrix_r0_c2 = 0
        self.csc_matrix_r1_c0 = 0
        self.csc_matrix_r1_c1 = 0
        self.csc_matrix_r1_c2 = 0
        self.csc_matrix_r2_c0 = 0
        self.csc_matrix_r2_c1 = 0
        self.csc_matrix_r2_c2 = 0
        self.csc_out_bias_0 = 0
        self.csc_out_bias_1 = 0
        self.csc_out_bias_2 = 0
        self.csc_in_bias_0 = 0
        self.csc_in_bias_1 = 0
        self.csc_in_bias_2 = 0

        self.imm_dtc_mean_type = 0
        self.imm_raw_to_f16_n = 0
        self.dtc_pixel_mean_ch0 = 0
        self.dtc_pixel_mean_ch1 = 0
        self.dtc_pixel_mean_ch2 = 0
        self.dtc_pixel_mean_ch3 = 0
        self.dtc_pixel_min_ch0 = 0
        self.dtc_pixel_min_ch1 = 0
        self.dtc_pixel_min_ch2 = 0
        self.dtc_pixel_min_ch3 = 0
        self.sfr_dtc_pixel_mean_ch0 = 0
        self.sfr_dtc_pixel_mean_ch1 = 0
        self.sfr_dtc_pixel_mean_ch2 = 0
        self.sfr_dtc_pixel_mean_ch3 = 0

        self.dtc_pixel_variance_ch0 = float16format2uint16(
            AIPP_INIT_FLOAT_VALUE_ONE)
        self.dtc_pixel_variance_ch1 = float16format2uint16(
            AIPP_INIT_FLOAT_VALUE_ONE)
        self.dtc_pixel_variance_ch2 = float16format2uint16(
            AIPP_INIT_FLOAT_VALUE_ONE)
        self.dtc_pixel_variance_ch3 = float16format2uint16(
            AIPP_INIT_FLOAT_VALUE_ONE)

        self.filling_hblank_ch0 = 0
        self.filling_hblank_ch1 = 0
        self.filling_hblank_ch2 = 0
        self.filling_hblank_ch3 = 0
        self.imm_area_pad_mode = 0
        self.imm_top_pad_size = 0
        self.imm_botton_pad_size = 0
        self.imm_left_pad_size = 0
        self.imm_right_pad_size = 0

        self.imm_channel_pad_mode = 0
        self.no_padding = 0
        self.padd_4channels = 0
        self.cpadding_spr = 0

        self.imm_pre_botton_clip_number = 0
        self.imm_pre_top_clip_number = 0

        self.imm_post_botton_clip_number = 0
        self.imm_post_top_clip_number = 0
        self.imm_post_right_clip_number = 0
        self.imm_post_left_clip_number = 0

        self.imm_scf_horizontal_start = 0
        self.imm_scf_vertical_start = 0
        self.imm_scf_horizontal_size = 0
        self.imm_scf_vertical_size = 0
        self.imm_scaling_mode = 0
        self.spr_scf_vertical_size = 0
        self.spr_scf_horizontal_size = 0
        self.imm_filter_order = 0
        self.imm_init_vert_phase = 0
        self.imm_init_hori_phase = 0
        self.imm_hori_scaling = 0
        self.imm_vert_scaling = 0

        self.imm_dst_stride_pixel = 0

        self.horizontal_flip_enable = 0
        self.vertical_flip_enable = 0

        self.raw_enable = 0
        self.imm_raw_image_channel = 0
        self.imm_start_channel_number = 0
        self.dtc_min = 0
        self.csc_in_bias = 0
        self.channel3_pad_value = 0
        self.botton_pad_rows = 0
        self.channel_pad_mode = 0
        self.raw_start_channel = 0
        self.imm_format_convert = 0
        self.crop_vertical_start = 0
        self.right_pad_cols = 0
        self.right_pad_cols = 0
        self.csc_out_bias = 0
        self.channel1_pad_value = 0
        self.channel_pad_value = 0
        self.single_line_enable = 0
        self.dst_horizontal_size = 0
        self.dtc_mean_type = 0
        self.raw_to_f16_n = 0
        self.area_pad_mode = 0
        self.crop_horizontal_start = 0
        self.dtc_mean = 0
        self.left_pad_cols = 0
        self.format_convert = 0
        self.channel0_pad_value = 0
        self.dst_vertical_size = 0
        self.raw_image_channel = 0
        self.top_pad_rows = 0
        self.channel2_pad_value = 0
        self.csc_matrix = 0
        self.imm_sid = 0
        self.dtc_var = 0

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        temp_env = TempEnv()

        copy_tensor_to_model(context, temp_env, self.src0, ALIGN_SRC_EVEN)
        if self.src1 is not None:
            copy_tensor_to_model(context, temp_env, self.src1, ALIGN_SRC_EVEN)
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGNED_ADDR)

        self.arch_version = get_soc_name() + get_soc_core_type()
        self._check_input_format_and_function(context)
        self.set_spr_aipp0(context)
        self.set_spr_aipp1(context)
        self.set_spr_aipp2(context)
        self.set_spr_aipp3(context)
        self.set_spr_aipp4(context)
        self.set_spr_aipp5(context)
        self.set_spr_aipp6(context)
        self.set_spr_aipp7(context)
        self.set_spr_aipp8(context)
        self.set_spr_aipp9(context)

        if self.arch_version == HI3796CV300ESAIC:
            self.set_spr_aipp10(context)
            self.set_spr_aipp11(context)
            self.set_spr_aipp12(context)
            self.set_spr_aipp13(context)
            self.set_spr_aipp15(context)
            self.set_spr_aipp16(context)
            self.set_spr_aipp17(context)

        param = context.encoder.new_param()

        self.check_dst_buffer()

        param.xd = xd_idx
        param.xs = self.set_gpr_x_s(context, temp_env)
        param.xt = self.set_gpr_x_t(context, temp_env)
        _type_map = {
            "int8": 0b00,
            "uint8": 0b01,
            "float16": 0b10
        }
        param.type = _type_map[self.dtype]
        instr = context.encoder.gen_dma_ld_img(param)

        context.model.step(instr)

        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

    def _cal_crop_info(self, context):
        """crop info"""
        if self.crop_enbale == 1:

            check_dict_and_not_none(self.crop_info, 'crop_info')
            self.dst_horizontal_size = self.crop_info.get('dst_horizontal_size')
            self.dst_vertical_size = self.crop_info.get('dst_vertical_size')
            self.crop_horizontal_start = self.crop_info.get(
                'crop_horizontal_start')
            self.crop_vertical_start = self.crop_info.get('crop_vertical_start')
            self.single_line_enable = self.crop_info.get('single_line_enable')

            self.imm_dst_horizontal_size = context.evaluate_expr(
                self.dst_horizontal_size)
            self.imm_dst_vertical_size = context.evaluate_expr(
                self.dst_vertical_size)
            self.imm_crop_horizontal_start = context.evaluate_expr(
                self.crop_horizontal_start)
            self.imm_crop_vertical_start = context.evaluate_expr(
                self.crop_vertical_start)
            self.imm_single_line_mode = context.evaluate_expr(
                self.single_line_enable)
            if self.arch_version in [ASCEND_310AIC, AIC]:
                if self.imm_single_line_mode == 1:
                    TikCheckUtil.check_equality(self.imm_dst_vertical_size, 1,
                                                'dst_vertical_size should be 1')
            self.check_crop_info()
        else:
            self.imm_dst_horizontal_size = self.imm_src_horizontal_size
            self.imm_dst_vertical_size = self.imm_src_vertical_size

    def _cal_swap_info(self, context, functions):
        if self.swap_enable == 1:
            if SWAP not in functions:
                TikCheckUtil.raise_error('swap not support')

            self.imm_rb_swap = context.evaluate_expr(self.swap_list[0])
            self.imm_uv_swap = context.evaluate_expr(self.swap_list[1])
            self.imm_ax_swap = context.evaluate_expr(self.swap_list[2])

            TikCheckUtil.check_in_range(
                self.imm_rb_swap,
                AIPP_INPUT_TYPE_SWAP_ALIGN.get(
                    self.imm_input_format).get('swap')[0],
                'swap rb out of range, input: {}'.format(self.imm_rb_swap))
            TikCheckUtil.check_in_range(
                self.imm_uv_swap,
                AIPP_INPUT_TYPE_SWAP_ALIGN.get(
                    self.imm_input_format).get('swap')[1],
                'swap uv out of range, input: {}'.format(self.imm_uv_swap))
            TikCheckUtil.check_in_range(
                self.imm_ax_swap,
                AIPP_INPUT_TYPE_SWAP_ALIGN.get(
                    self.imm_input_format).get('swap')[2],
                'swap ax out of range, input: {}'.format(self.imm_ax_swap))

    def _cal_csc_info(self, context, functions):
        """csc info"""
        if self.csc_enable == 1:
            if CSC not in functions:
                TikCheckUtil.raise_error('csc not support')

            check_dict_and_not_none(self.csc_info, 'csc_info')
            self.format_convert = self.csc_info.get('format_convert')
            self.csc_matrix = self.csc_info.get('csc_matrix')
            self.csc_out_bias = self.csc_info.get('csc_out_bias')
            self.csc_in_bias = self.csc_info.get('csc_in_bias')

            # check
            self._check_csc_format_convert(context)

            if self.imm_format_convert == 0:
                self.csc_matrix_r0_c0 = context.evaluate_expr(
                    self.csc_matrix[0][0])
                self.csc_matrix_r0_c1 = context.evaluate_expr(
                    self.csc_matrix[0][1])
                self.csc_matrix_r0_c2 = context.evaluate_expr(
                    self.csc_matrix[0][2])
                self.csc_matrix_r1_c0 = context.evaluate_expr(
                    self.csc_matrix[1][0])
                self.csc_matrix_r1_c1 = context.evaluate_expr(
                    self.csc_matrix[1][1])
                self.csc_matrix_r1_c2 = context.evaluate_expr(
                    self.csc_matrix[1][2])
                self.csc_matrix_r2_c0 = context.evaluate_expr(
                    self.csc_matrix[2][0])
                self.csc_matrix_r2_c1 = context.evaluate_expr(
                    self.csc_matrix[2][1])
                self.csc_matrix_r2_c2 = context.evaluate_expr(
                    self.csc_matrix[2][2])
                self.csc_out_bias_0 = context.evaluate_expr(
                    self.csc_out_bias[0])
                self.csc_out_bias_1 = context.evaluate_expr(
                    self.csc_out_bias[1])
                self.csc_out_bias_2 = context.evaluate_expr(
                    self.csc_out_bias[2])
                self.csc_in_bias_0 = context.evaluate_expr(self.csc_in_bias[0])
                self.csc_in_bias_1 = context.evaluate_expr(self.csc_in_bias[1])
                self.csc_in_bias_2 = context.evaluate_expr(self.csc_in_bias[2])
            else:
                csc_matrix_param = AIPP_FORMAT_CONVERT.get(
                    self.imm_format_convert).get('csc_matrix')
                self.csc_matrix_r0_c0 = csc_matrix_param[0][0]
                self.csc_matrix_r0_c1 = csc_matrix_param[0][1]
                self.csc_matrix_r0_c2 = csc_matrix_param[0][2]
                self.csc_matrix_r1_c0 = csc_matrix_param[1][0]
                self.csc_matrix_r1_c1 = csc_matrix_param[1][1]
                self.csc_matrix_r1_c2 = csc_matrix_param[1][2]
                self.csc_matrix_r2_c0 = csc_matrix_param[2][0]
                self.csc_matrix_r2_c1 = csc_matrix_param[2][1]
                self.csc_matrix_r2_c2 = csc_matrix_param[2][2]
                csc_out_bias_param = AIPP_FORMAT_CONVERT.get(
                    self.imm_format_convert).get('csc_out_bias')
                self.csc_out_bias_0 = csc_out_bias_param[0]
                self.csc_out_bias_1 = csc_out_bias_param[1]
                self.csc_out_bias_2 = csc_out_bias_param[2]
                csc_in_bias_param = AIPP_FORMAT_CONVERT.get(
                    self.imm_format_convert).get('csc_in_bias')
                self.csc_in_bias_0 = csc_in_bias_param[0]
                self.csc_in_bias_1 = csc_in_bias_param[1]
                self.csc_in_bias_2 = csc_in_bias_param[2]

    def _cal_dtc_mean_by_type(self, context):
        """calculate dtc mean"""
        if self.imm_dtc_mean_type == 0:
            self.dtc_pixel_mean_ch0 = context.evaluate_expr(
                self.dtc_mean[0])
            self.dtc_pixel_mean_ch1 = context.evaluate_expr(
                self.dtc_mean[1])
            self.dtc_pixel_mean_ch2 = context.evaluate_expr(
                self.dtc_mean[2])
            self.dtc_pixel_mean_ch3 = context.evaluate_expr(
                self.dtc_mean[3])
            self.sfr_dtc_pixel_mean_ch0 = self.dtc_pixel_mean_ch0 // BIT_16
            self.sfr_dtc_pixel_mean_ch1 = self.dtc_pixel_mean_ch1 // BIT_16
            self.sfr_dtc_pixel_mean_ch2 = self.dtc_pixel_mean_ch2 // BIT_16
            self.sfr_dtc_pixel_mean_ch3 = self.dtc_pixel_mean_ch3 // BIT_16
            self.dtc_pixel_mean_ch0 = self.dtc_pixel_mean_ch0 % BIT_16
            self.dtc_pixel_mean_ch1 = self.dtc_pixel_mean_ch1 % BIT_16
            self.dtc_pixel_mean_ch2 = self.dtc_pixel_mean_ch2 % BIT_16
            self.dtc_pixel_mean_ch3 = self.dtc_pixel_mean_ch3 % BIT_16
        else:
            dtc_pixel_mean_ch0_fp16 = context.evaluate_expr(
                self.dtc_mean[0])
            dtc_pixel_mean_ch1_fp16 = context.evaluate_expr(
                self.dtc_mean[1])
            dtc_pixel_mean_ch2_fp16 = context.evaluate_expr(
                self.dtc_mean[2])
            dtc_pixel_mean_ch3_fp16 = context.evaluate_expr(
                self.dtc_mean[3])
            self.dtc_pixel_mean_ch0 = float16format2uint16(
                dtc_pixel_mean_ch0_fp16)
            self.dtc_pixel_mean_ch1 = float16format2uint16(
                dtc_pixel_mean_ch1_fp16)
            self.dtc_pixel_mean_ch2 = float16format2uint16(
                dtc_pixel_mean_ch2_fp16)
            self.dtc_pixel_mean_ch3 = float16format2uint16(
                dtc_pixel_mean_ch3_fp16)
            self.sfr_dtc_pixel_mean_ch0 = 0
            self.sfr_dtc_pixel_mean_ch1 = 0
            self.sfr_dtc_pixel_mean_ch2 = 0
            self.sfr_dtc_pixel_mean_ch3 = 0

    def _cal_dtc_info(self, context, functions):
        """dtc info"""
        if self.dtc_enable == 1:
            if DTC not in functions:
                TikCheckUtil.raise_error('dtc not support')

            check_dict_and_not_none(self.dtc_info, 'dtc_info')
            self.dtc_mean_type = self.dtc_info.get('dtc_mean_type')
            self.raw_to_f16_n = self.dtc_info.get('raw_to_f16_n')
            self.dtc_mean = self.dtc_info.get('dtc_mean')
            self.dtc_min = self.dtc_info.get('dtc_min')
            self.dtc_var = self.dtc_info.get('dtc_var')

            self.imm_dtc_mean_type = context.evaluate_expr(self.dtc_mean_type)
            TikCheckUtil.check_in_range(
                self.imm_dtc_mean_type, [0, 1],
                'mean_type_ctr value error, '
                'input: {}'.format(self.imm_dtc_mean_type))

            self.imm_raw_to_f16_n = context.evaluate_expr(self.raw_to_f16_n)
            TikCheckUtil.check_in_range(
                self.imm_raw_to_f16_n,
                RAW_TO_16_N,
                'raw_to_f16_n value error, '
                'input: {}'.format(self.imm_raw_to_f16_n))

            self._cal_dtc_mean_by_type(context)

            dtc_pixel_min_ch0_fp16 = context.evaluate_expr(self.dtc_min[0])
            dtc_pixel_min_ch1_fp16 = context.evaluate_expr(self.dtc_min[1])
            dtc_pixel_min_ch2_fp16 = context.evaluate_expr(self.dtc_min[2])
            dtc_pixel_min_ch3_fp16 = context.evaluate_expr(self.dtc_min[3])

            self.dtc_pixel_min_ch0 = float16format2uint16(
                dtc_pixel_min_ch0_fp16)
            self.dtc_pixel_min_ch1 = float16format2uint16(
                dtc_pixel_min_ch1_fp16)
            self.dtc_pixel_min_ch2 = float16format2uint16(
                dtc_pixel_min_ch2_fp16)
            self.dtc_pixel_min_ch3 = float16format2uint16(
                dtc_pixel_min_ch3_fp16)

            dtc_pixel_variance_ch0_fp16 = context.evaluate_expr(self.dtc_var[0])
            dtc_pixel_variance_ch1_fp16 = context.evaluate_expr(self.dtc_var[1])
            dtc_pixel_variance_ch2_fp16 = context.evaluate_expr(self.dtc_var[2])
            dtc_pixel_variance_ch3_fp16 = context.evaluate_expr(self.dtc_var[3])

            self.dtc_pixel_variance_ch0 = float16format2uint16(
                dtc_pixel_variance_ch0_fp16)
            self.dtc_pixel_variance_ch1 = float16format2uint16(
                dtc_pixel_variance_ch1_fp16)
            self.dtc_pixel_variance_ch2 = float16format2uint16(
                dtc_pixel_variance_ch2_fp16)
            self.dtc_pixel_variance_ch3 = float16format2uint16(
                dtc_pixel_variance_ch3_fp16)

    def _cal_area_pad_info(self, context, functions):
        """area padding info"""
        if self.area_pad_enable == 1:
            if AERA_PADDING not in functions:
                TikCheckUtil.raise_error('area_pad not support')

            check_dict_and_not_none(self.area_pad_info, 'area_pad_info')
            self.area_pad_mode = self.area_pad_info.get('area_pad_mode')
            self.top_pad_rows = self.area_pad_info.get('top_pad_rows')
            self.botton_pad_rows = self.area_pad_info.get('botton_pad_rows')
            self.left_pad_cols = self.area_pad_info.get('left_pad_cols')
            self.right_pad_cols = self.area_pad_info.get('right_pad_cols')
            self.channel0_pad_value = self.area_pad_info.get(
                'channel0_pad_value')
            self.channel1_pad_value = self.area_pad_info.get(
                'channel1_pad_value')
            self.channel2_pad_value = self.area_pad_info.get(
                'channel2_pad_value')
            self.channel3_pad_value = self.area_pad_info.get(
                'channel3_pad_value')

            if self.dtype == 'float16':
                filling_hblank_ch0_fp16 = context.evaluate_expr(
                    self.channel0_pad_value)
                filling_hblank_ch1_fp16 = context.evaluate_expr(
                    self.channel1_pad_value)
                filling_hblank_ch2_fp16 = context.evaluate_expr(
                    self.channel2_pad_value)
                filling_hblank_ch3_fp16 = context.evaluate_expr(
                    self.channel3_pad_value)
                self.filling_hblank_ch0 = float16format2uint16(
                    filling_hblank_ch0_fp16)
                self.filling_hblank_ch1 = float16format2uint16(
                    filling_hblank_ch1_fp16)
                self.filling_hblank_ch2 = float16format2uint16(
                    filling_hblank_ch2_fp16)
                self.filling_hblank_ch3 = float16format2uint16(
                    filling_hblank_ch3_fp16)
            else:
                self.filling_hblank_ch0 = context.evaluate_expr(
                    self.channel0_pad_value)
                self.filling_hblank_ch1 = context.evaluate_expr(
                    self.channel1_pad_value)
                self.filling_hblank_ch2 = context.evaluate_expr(
                    self.channel2_pad_value)
                self.filling_hblank_ch3 = context.evaluate_expr(
                    self.channel3_pad_value)

                if self.dtype == 'int8':
                    self.check_list_range(context, [self.filling_hblank_ch0,
                                                    self.filling_hblank_ch1,
                                                    self.filling_hblank_ch2,
                                                    self.filling_hblank_ch3],
                                          4, range(-128, 128), 'filling_hblank')
                else:
                    self.check_list_range(context, [self.filling_hblank_ch0,
                                                    self.filling_hblank_ch1,
                                                    self.filling_hblank_ch2,
                                                    self.filling_hblank_ch3],
                                          4, range(0, 256), 'filling_hblank')

            self.imm_area_pad_mode = context.evaluate_expr(self.area_pad_mode)
            self.imm_top_pad_size = context.evaluate_expr(self.top_pad_rows)
            self.imm_botton_pad_size = context.evaluate_expr(
                self.botton_pad_rows)
            self.imm_left_pad_size = context.evaluate_expr(self.left_pad_cols)
            self.imm_right_pad_size = context.evaluate_expr(self.right_pad_cols)

            TikCheckUtil.check_in_range(
                self.imm_area_pad_mode, range(0, 4),
                'area_pad_mode value error, [0, 3], '
                'input: {}'.format(self.imm_area_pad_mode))

            if self.arch_version == HI3796CV300ESAIC:
                TikCheckUtil.check_in_range(
                    self.imm_top_pad_size, range(0, 33),
                    'top_pad_size value error, [0, 32], '
                    'input: {}'.format(self.imm_top_pad_size))
            elif self.imm_area_pad_mode == 0:
                TikCheckUtil.check_equality(
                    self.imm_top_pad_size, 0,
                    'top_pad_size value should be 0')

            if self.arch_version == HI3796CV300ESAIC:
                TikCheckUtil.check_in_range(
                    self.imm_botton_pad_size, range(0, 33),
                    'botton_pad_size value error, [0, 32], '
                    'input: {}'.format(self.imm_botton_pad_size))
            elif self.imm_area_pad_mode == 0:
                TikCheckUtil.check_equality(
                    self.imm_botton_pad_size, 0,
                    'botton_pad_size value should be 0')

            TikCheckUtil.check_in_range(
                self.imm_left_pad_size, range(0, 33),
                'left_pad_size value error, [0, 32], '
                'input: {}'.format(self.imm_left_pad_size))
            TikCheckUtil.check_in_range(
                self.imm_right_pad_size, range(0, 33),
                'right_pad_size value error, [0, 32], '
                'input: {}'.format(self.imm_right_pad_size))

    def _cal_cpad_info(self, context):
        """cpadding info"""
        if self.channel_pad_enable == 1:

            check_dict_and_not_none(self.channel_pad_info, 'channel_pad_info')
            self.channel_pad_mode = self.channel_pad_info.get(
                'channel_pad_mode')
            self.channel_pad_value = self.channel_pad_info.get(
                'channel_pad_value')

            self.imm_channel_pad_mode = context.evaluate_expr(
                self.channel_pad_mode)
            TikCheckUtil.check_in_range(
                self.imm_channel_pad_mode, range(0, 3),
                'imm_channel_pad_mode value error, [0, 2], '
                'input: {}'.format(self.imm_channel_pad_mode))
            self.no_padding = self.imm_channel_pad_mode % 2
            self.padd_4channels = self.imm_channel_pad_mode // 2

            cpadding = context.evaluate_expr(self.channel_pad_value)
            if self.dtype == 'float16':
                self.cpadding_spr = float16format2uint16(cpadding)
            else:
                self.cpadding_spr = cpadding

                if self.dtype == 'uint8':
                    TikCheckUtil.check_in_range(
                        self.cpadding_spr, range(0, 256),
                        'cpadding_value out of range, [0, 255], '
                        'input: {}'.format(self.cpadding_spr))
                else:
                    TikCheckUtil.check_in_range(
                        self.cpadding_spr, range(-128, 128),
                        'cpadding_value out of range, [-128, 127], '
                        'input: {}'.format(self.cpadding_spr))

    def _cal_pre_clip_info(self, context, functions):
        if self.pre_clip_enable == 1:
            if PRE_CLIP not in functions:
                TikCheckUtil.raise_error('pre_clip not support')

            check_dict_and_not_none(self.pre_clip_info, 'pre_clip_info')
            pre_top_clip_number = self.pre_clip_info.get(
                'pre_top_clip_number')
            pre_botton_clip_number = self.pre_clip_info.get(
                'pre_botton_clip_number')

            self.imm_pre_botton_clip_number = context.evaluate_expr(
                pre_botton_clip_number)
            self.imm_pre_top_clip_number = context.evaluate_expr(
                pre_top_clip_number)
            TikCheckUtil.check_in_range(
                self.imm_pre_botton_clip_number, range(0, 2),
                "pre_botton_clip_number out of range, [0, 1], "
                "input: {}".format(self.imm_pre_botton_clip_number))
            TikCheckUtil.check_in_range(
                self.imm_pre_top_clip_number, range(0, 2),
                "pre_top_clip_number out of range, [0, 1], "
                "input: {}".format(self.imm_pre_top_clip_number))
            TikCheckUtil.check_ge(
                self.imm_dst_vertical_size,
                self.imm_pre_top_clip_number +
                self.imm_pre_botton_clip_number + 1,
                'pre_botton_clip_number + pre_top_clip_number'
                ' bigger than vertical_size, '
                'input: {}'.format(self.imm_dst_vertical_size))

    def _cal_scf_info(self, context, functions):
        if self.scf_enable == 1:
            if SCF not in functions:
                TikCheckUtil.raise_error('scf not support')
            check_dict_and_not_none(self.scf_info, 'scf_info')
            scf_horizontal_size = self.scf_info.get(
                'scf_horizontal_size')
            scf_vertical_size = self.scf_info.get('scf_vertical_size')
            scf_horizontal_start = self.scf_info.get(
                'scf_horizontal_start')
            scf_vertical_start = self.scf_info.get(
                'scf_vertical_start')
            scaling_mode = self.scf_info.get('scaling_mode')

            self.imm_scf_horizontal_start = context.evaluate_expr(
                scf_horizontal_start)
            self.imm_scf_vertical_start = context.evaluate_expr(
                scf_vertical_start)
            self.imm_scf_horizontal_size = context.evaluate_expr(
                scf_horizontal_size)
            self.imm_scf_vertical_size = context.evaluate_expr(
                scf_vertical_size)
            TikCheckUtil.check_in_range(
                self.imm_scf_horizontal_size, range(16, 1921),
                'scf_horizontal_size out of range, '
                'input: {}'.format(self.imm_scf_horizontal_size))
            TikCheckUtil.check_in_range(
                self.imm_scf_vertical_size, range(16, 1081),
                'scf_vertical_size out of range, '
                'input: {}'.format(self.imm_scf_vertical_size))

            self.imm_scaling_mode = context.evaluate_expr(scaling_mode)
            TikCheckUtil.check_in_range(
                self.imm_scaling_mode, [0, 1],
                'scaling_mode out of range, [0, 1], '
                'input: {}'.format(self.imm_scaling_mode))

            # spr 12
            self.spr_scf_vertical_size = self.imm_scf_vertical_size - 1
            self.spr_scf_horizontal_size = self.imm_scf_horizontal_size - 1

            if self.imm_scf_horizontal_size > self.imm_dst_horizontal_size:
                self.imm_filter_order = 1
            else:
                self.imm_filter_order = 0

            # spr 14
            pre_scf_horizontal_size = self.imm_dst_horizontal_size
            pre_scf_vertical_size = self.imm_dst_vertical_size - \
                                    self.imm_pre_botton_clip_number - \
                                    self.imm_pre_top_clip_number

            cal_hori_scaling = ((pre_scf_horizontal_size - 1)*SCALE_COF//(
                self.imm_scf_horizontal_size - 1))//4*4
            cal_vert_scaling = (pre_scf_vertical_size - 1)*SCALE_COF//(
                self.imm_scf_vertical_size - 1)//4*4

            hori_scaling = self.scf_info.get(
                'scf_horizontal_scale', cal_hori_scaling)
            vert_scaling = self.scf_info.get(
                'scf_vertical_scale', cal_vert_scaling)

            self.imm_hori_scaling = context.evaluate_expr(hori_scaling)
            self.imm_vert_scaling = context.evaluate_expr(vert_scaling)

            self.imm_init_vert_phase = self.imm_scf_vertical_start
            self.imm_init_hori_phase = self.imm_scf_horizontal_start

        else:
            self.imm_scf_vertical_size = self.imm_dst_vertical_size - \
                                         self.imm_pre_top_clip_number - \
                                         self.imm_pre_botton_clip_number
            self.imm_scf_horizontal_size = self.imm_dst_horizontal_size
            self.spr_scf_vertical_size = self.imm_scf_vertical_size - 1
            self.spr_scf_horizontal_size = self.imm_dst_horizontal_size - 1

    def _cal_post_clip_info(self, context, functions):
        """post clip info"""
        if self.post_clip_enable == 1:
            if POST_CLIP not in functions:
                TikCheckUtil.raise_error('post clip not support')

            check_dict_and_not_none(self.post_clip_info, 'post_clip_info')
            post_botton_clip_number = self.post_clip_info.get(
                'post_botton_clip_number')
            post_top_clip_number = self.post_clip_info.get(
                'post_top_clip_number')
            post_right_clip_number = self.post_clip_info.get(
                'post_right_clip_number')
            post_left_clip_number = self.post_clip_info.get(
                'post_left_clip_number')

            self.imm_post_botton_clip_number = context.evaluate_expr(
                post_botton_clip_number)
            self.imm_post_top_clip_number = context.evaluate_expr(
                post_top_clip_number)
            self.imm_post_right_clip_number = context.evaluate_expr(
                post_right_clip_number)
            self.imm_post_left_clip_number = context.evaluate_expr(
                post_left_clip_number)
            TikCheckUtil.check_in_range(
                self.imm_post_botton_clip_number, range(0, 64),
                "post_botton_clip_number out of range, [0, 63], "
                "input: {}".format(self.imm_post_botton_clip_number))
            TikCheckUtil.check_in_range(
                self.imm_post_top_clip_number, range(0, 64),
                "post_top_clip_number out of range, [0, 63], "
                "input: {}".format(self.imm_post_top_clip_number))
            TikCheckUtil.check_in_range(
                self.imm_post_right_clip_number, range(0, 64),
                "post_botton_clip_number out of range, [0, 63], "
                "input: {}".format(self.imm_post_right_clip_number))
            TikCheckUtil.check_in_range(
                self.imm_post_left_clip_number, range(0, 64),
                "post_top_clip_number out of range, [0, 63], "
                "input: {}".format(self.imm_post_left_clip_number))

    def _cal_stretch_info(self, context, functions):
        """stretch info"""
        if self.stretch_enable == 1:
            if STRETCH not in functions:
                TikCheckUtil.raise_error('stretch not support')
            check_dict_and_not_none(self.stretch_info, 'stretch_info')
            dst_stride_pixel = self.stretch_info.get(
                'dst_stride_pixel')
            self.imm_dst_stride_pixel = context.evaluate_expr(
                dst_stride_pixel)
            TikCheckUtil.check_in_range(
                self.imm_dst_stride_pixel, range(0, 65536),
                "dst_stride_pixel out of range, "
                "input: {}".format(self.imm_dst_stride_pixel))

    def _cal_flip_info(self, context, functions):
        if self.flip_enable == 1:
            if FLIP not in functions:
                TikCheckUtil.raise_error('flip not support')
            imm_flip_mode = context.evaluate_expr(self.flip_mode)
            TikCheckUtil.check_in_range(
                imm_flip_mode, range(0, 4),
                'flip_mode value out of range, '
                'input: {}'.format(imm_flip_mode))
            self.horizontal_flip_enable = imm_flip_mode % 2
            self.vertical_flip_enable = imm_flip_mode // 2

    def _cal_raw_info(self, context, functions):
        """raw info"""
        if self.raw_enable == 1:
            if RAW not in functions:
                TikCheckUtil.raise_error('raw not support')
            check_dict_and_not_none(self.raw_info, 'raw_info')
            self.raw_start_channel = self.raw_info.get('raw_start_channel')
            self.raw_image_channel = self.raw_info.get('raw_image_channel')

            self.imm_raw_image_channel = context.evaluate_expr(
                self.raw_image_channel)
            TikCheckUtil.check_in_range(
                self.imm_raw_image_channel, range(0, 4),
                'raw_image_channel value error, [0, 3], '
                'input: {}'.format(self.imm_raw_image_channel))
            self.imm_start_channel_number = context.evaluate_expr(
                self.raw_start_channel)
            TikCheckUtil.check_in_range(
                self.imm_start_channel_number, range(0, 4),
                'start_channel_number value error, [0, 3], '
                'input: {}'.format(self.imm_start_channel_number))

    def _check_input_format_and_function(self, context):
        """check input format"""
        self.crop_enbale, self.swap_enable, self.csc_enable,\
        self.dtc_enable, self.area_pad_enable, \
        self.channel_pad_enable, self.pre_clip_enable, \
        self.scf_enable, self.post_clip_enable, self.flip_enable, \
        self.stretch_enable, self.raw_enable = \
            aipp_get_enable_bit(self.arch_version, self.function_switch)

        self.imm_input_format = context.evaluate_expr(self.input_format)

        self.imm_src_horizontal_size = context.evaluate_expr(
            self.src_horizontal_size)
        self.imm_src_vertical_size = context.evaluate_expr(
            self.src_vertical_size)

        self._check_src_info()

        functions = AIPP_INPUT_VERSON_AND_FUNCTION.get(self.arch_version).get(
            self.imm_input_format)

        if functions is None:
            TikCheckUtil.raise_error(
                "arch_version not support "
                "input_format: {}".format(self.imm_input_format))

        self._cal_crop_info(context)

        self._check_src_overflow()

        self._cal_swap_info(context, functions)

        self._cal_csc_info(context, functions)

        self._cal_dtc_info(context, functions)

        self._cal_area_pad_info(context, functions)

        self._cal_cpad_info(context)

        if self.arch_version == HI3796CV300ESAIC:

            self._cal_pre_clip_info(context, functions)

            self._cal_scf_info(context, functions)

            self._cal_post_clip_info(context, functions)

            self._cal_stretch_info(context, functions)

            self._cal_flip_info(context, functions)

        if self.arch_version == AIC:
            self._cal_raw_info(context, functions)
        self.imm_sid = context.evaluate_expr(self.sid)
        if self.arch_version == HI3796CV300ESAIC:
            TikCheckUtil.check_in_range(self.imm_sid, range(0, 11),
                                        'sid value error, [0, 10], '
                                        'input: {}'.format(self.imm_sid))

    def _check_src_info(self):
        """check src info"""
        TikCheckUtil.check_in_range(
            self.imm_src_horizontal_size, range(8, 4097),
            'src_horizontal_resolution should in [8, 4096], '
            'input: {}'.format(self.imm_src_horizontal_size))

        if self.imm_input_format in [YUV420, YUYV, YUV422]:
            TikCheckUtil.check_equality(
                self.imm_src_horizontal_size % 2, 0,
                'src_horizontal_size should be even, '
                'input: {}'.format(self.imm_src_horizontal_size))

        if self.arch_version == HI3796CV300ESAIC:
            if self.imm_input_format in [YUV420, YUV422, YUYV, YUV400, RAW10,
                                         RAW12, RAW16, RGB888]:
                TikCheckUtil.check_equality(
                    self.imm_src_horizontal_size % 16, 0,
                    'src_horizontal_size should be 16*n, '
                    'input: {}'.format(self.imm_src_horizontal_size))
            elif self.imm_input_format in [AYUV444, ARGB8888, XRGB8888]:
                TikCheckUtil.check_equality(
                    self.imm_src_horizontal_size % 4, 0,
                    'src_horizontal_size should be 4*n, '
                    'input: {}'.format(self.imm_src_horizontal_size))

        TikCheckUtil.check_ge(
            self.imm_src_vertical_size, 1,
            'src_vertical_size should more than 0')

    def check_crop_info(self):
        """check crop info"""
        if self.arch_version in [ASCEND_910AIC, HI3796CV300ESAIC]:
            TikCheckUtil.check_equality(
                self.imm_single_line_mode, 0,
                '200hisi-es single_line_mode must 0')
        else:
            TikCheckUtil.check_in_range(
                self.imm_single_line_mode, range(0, 2),
                'single_line_mode should in [0, 1], '
                'input: {}'.format(self.imm_single_line_mode))
            if self.imm_single_line_mode == 1:
                TikCheckUtil.check_equality(
                    self.imm_dst_vertical_size, 1,
                    'dst_vertical_size should be 1')

        # even YUV420/YUV422 semi-plannar and YUYV packed
        if self.imm_input_format in [YUV420, YUYV, YUV422]:
            TikCheckUtil.check_equality(
                self.imm_dst_horizontal_size % 2, 0,
                'horizontal_size should be even, '
                'input: {}'.format(self.imm_dst_horizontal_size))
        TikCheckUtil.check_in_range(
            self.imm_dst_horizontal_size, range(8, 4097),
            'horizontal_size should in [8, 4096], '
            'input: {}'.format(self.imm_dst_horizontal_size))

        # even YUV420
        if self.imm_input_format == YUV420 and self.imm_single_line_mode == 0:
            TikCheckUtil.check_equality(
                self.imm_dst_vertical_size % 2, 0,
                'vertical_size should be even, '
                'input: {}'.format(self.imm_dst_vertical_size))
            TikCheckUtil.check_in_range(
                self.imm_dst_vertical_size, range(8, 4097),
                'vertical_size should in [8, 4096], '
                'input: {}'.format(self.imm_dst_vertical_size))

        # even YUV420/YUV422 semi-plannar and YUYV packed
        if self.imm_input_format in [YUV420, YUYV, YUV422, XRGB8888,
                                     NC1HWC0DI_INT8, NC1HWC0DI_FP16, RGB888,
                                     YUV400]:
            TikCheckUtil.check_equality(
                self.imm_crop_horizontal_start % 2, 0,
                'horizontal_start should be even, '
                'input: {}'.format(self.imm_crop_horizontal_start))
        TikCheckUtil.check_in_range(
            self.imm_crop_horizontal_start, range(0, 4096),
            'horizontal_start should in [0, 4095], '
            'input: {}'.format(self.imm_crop_horizontal_start))

        # even YUV420
        if self.imm_input_format in [YUV420, XRGB8888, NC1HWC0DI_INT8,
                                     NC1HWC0DI_FP16, RGB888,
                                     YUV400] and self.imm_single_line_mode == 0:
            TikCheckUtil.check_equality(
                self.imm_crop_vertical_start % 2, 0,
                'vertical_start should be even, '
                'input: {}'.format(self.imm_crop_vertical_start))
        TikCheckUtil.check_in_range(
            self.imm_crop_vertical_start, range(0, 4096),
            'vertical_start should in [0, 4095], '
            'input: {}'.format(self.imm_crop_vertical_start))

        # range
        TikCheckUtil.check_in_range(
            self.imm_crop_horizontal_start,
            range(0, self.imm_src_horizontal_size + 1),
            'horizontal_start should in src_horizontal_size, '
            'input: {}'.format(self.imm_crop_horizontal_start))
        TikCheckUtil.check_in_range(
            self.imm_crop_horizontal_start + self.imm_dst_horizontal_size,
            range(0, self.imm_src_horizontal_size + 1),
            'horizontal_start+horizontal_size should in src_horizontal_size, '
            'input: {}'.format(self.imm_crop_horizontal_start))

        # range
        TikCheckUtil.check_in_range(
            self.imm_crop_vertical_start,
            range(0, self.imm_src_vertical_size + 1),
            'vertical_start should in src_vertical_size, '
            'input: {}'.format(self.imm_crop_vertical_start))
        TikCheckUtil.check_in_range(
            self.imm_crop_vertical_start + self.imm_dst_vertical_size,
            range(0, self.imm_src_vertical_size + 1),
            'vertical_start + vertical_size should in src_vertical_size, '
            'input: {}'.format(self.imm_crop_vertical_start +
                               self.imm_dst_vertical_size))

    def _check_src_overflow(self):

        if self.src1_start == 0:
            if self.imm_input_format in [YUV420, YUV422]:
                self.src1_start = self.src0_start + \
                                  self.imm_src_horizontal_size*\
                                  self.imm_src_vertical_size
            elif self.imm_input_format == NC1HWC0DI_INT8:
                self.src1_start = self.src0_start + \
                                  self.imm_src_horizontal_size*\
                                  self.imm_src_vertical_size*4
            elif self.imm_input_format == NC1HWC0DI_FP16:
                self.src1_start = self.src0_start + \
                                  self.imm_src_horizontal_size*\
                                  self.imm_src_vertical_size*8

        if self.src1 is None:
            check_aipp_one_src_overflow(self.src0, self.imm_input_format,
                                        self.imm_src_horizontal_size,
                                        self.imm_src_vertical_size)
        else:
            if self.imm_input_format in [YUV420, NC1HWC0DI_INT8, NC1HWC0DI_FP16,
                                         YUV422]:
                check_aipp_two_src_overflow(self.src0, self.src1,
                                            self.imm_input_format,
                                            self.imm_src_horizontal_size,
                                            self.imm_src_vertical_size)

            else:
                TikCheckUtil.check_equality(
                    self.src1, None, "src1 should be None")

    def _check_csc_format_convert(self, context):
        """check csc format convert"""
        self.imm_format_convert = context.evaluate_expr(self.format_convert)
        if self.imm_format_convert == 0:
            self.check_matrix_range(context, self.csc_matrix, [3, 3],
                                    input_range=range(-32768, 32768))
            self.check_list_range(context, self.csc_in_bias, 3,
                                  range(0, 256), 'csc_in_bias')
            self.check_list_range(context, self.csc_out_bias, 3,
                                  range(0, 256), 'csc_out_bias')
        else:
            if self.arch_version == HI3796CV300ESAIC:
                TikCheckUtil.check_in_range(
                    self.imm_format_convert, range(10, 18),
                    'format_convert out of range, '
                    'input: {}'.format(self.imm_format_convert,))
            else:
                TikCheckUtil.check_in_range(
                    self.imm_format_convert, range(1, 10),
                    'format_convert out of range, '
                    'input: {}'.format(self.imm_format_convert))

    @staticmethod
    def check_matrix_range(context, matrix, shape, input_range=None):
        """check matrix type and range"""
        if matrix is None or shape is None:
            TikCheckUtil.raise_error(
                "check_int_scalar_matrix input error")

        # matrix
        if len(matrix) != shape[0] or len(matrix[0]) != shape[1]:
            TikCheckUtil.raise_error("matrix shape error")

        for i in range(shape[0]):
            for j in range(shape[1]):
                one_matrix = context.evaluate_expr(matrix[i][j])
                if input_range:
                    TikCheckUtil.check_in_range(
                        one_matrix, input_range,
                        'matrix out of range, input: {}'.format(one_matrix))

    @staticmethod
    def check_list_range(context, input_list, length,
                         input_range=None, name=''):
        """check input type and range"""
        if input_list is None:
            TikCheckUtil.raise_error(name + "_input is None")

        if len(input_list) != length:
            TikCheckUtil.raise_error(name + "_list length error, "
                                            "input: {}".format(len(input_list)))

        for i in range(length):
            one_input = context.evaluate_expr(input_list[i])
            if input_range:
                TikCheckUtil.check_in_range(one_input, input_range, name +
                                            ' out of range, '
                                            'input: {}'.format(one_input))

    def check_dst_buffer(self):
        """check dst buffer"""
        channels = 32 // DTYPE_SIZE[self.dst.dtype]

        if (self.imm_channel_pad_mode == 1) and \
                self.arch_version == HI3796CV300ESAIC:
            channels = AIPP_INPUT_TYPE_SWAP_ALIGN.get(
                self.imm_input_format).get('channels')

        if (self.imm_channel_pad_mode == 2) and (
                self.arch_version == HI3796CV300ESAIC or
                self.arch_version == AIC):
            channels = 4

        if self.arch_version == HI3796CV300ESAIC:
            if self.imm_dst_stride_pixel == 0:
                extent = (self.imm_scf_horizontal_size -
                          self.imm_post_right_clip_number -
                          self.imm_post_left_clip_number +
                          self.imm_left_pad_size + self.imm_right_pad_size)*(
                              self.imm_scf_vertical_size -
                              self.imm_post_botton_clip_number -
                              self.imm_post_top_clip_number +
                              self.imm_top_pad_size +
                              self.imm_botton_pad_size)*channels
            else:
                extent = self.imm_dst_stride_pixel*(
                    self.imm_scf_vertical_size -
                    self.imm_post_botton_clip_number -
                    self.imm_post_top_clip_number +
                    self.imm_top_pad_size +
                    self.imm_botton_pad_size)*channels
        else:
            extent = (self.imm_dst_horizontal_size +
                      self.imm_left_pad_size + self.imm_right_pad_size)*(
                          self.imm_dst_vertical_size +
                          self.imm_top_pad_size +
                          self.imm_botton_pad_size)*channels
            if self.arch_version == AIC and \
                    self.imm_input_format in [RAW16, RAW24] and \
                    self.raw_enable == 1:
                if self.imm_raw_image_channel == 0:
                    extent = extent // 4

        extent = extent*DTYPE_SIZE[self.dst.dtype]
        TikCheckUtil.check_ge(self.dst.buffer_size, extent,
                              'dst is less than result, '
                              'extent: {}'.format(extent))
        TikCheckUtil.check_equality(extent % 32, 0,
                                    "output should be 32 align, "
                                    "extent: {}".format(extent))

    def set_gpr_x_s(self, context, temp_env):
        """set gpr xs"""
        src_horizontal_size = self.imm_dst_horizontal_size - 1
        src_vertical_size = self.imm_dst_vertical_size - 1
        src_horizontal_start = self.imm_crop_horizontal_start
        src_vertical_start = self.imm_crop_vertical_start
        xs_idx = temp_env.alloc_register()
        x_s = src_horizontal_size
        x_s |= src_vertical_size << AIPP_XS_OFFSET_LIST[1]
        x_s |= src_horizontal_start << AIPP_XS_OFFSET_LIST[2]
        x_s |= src_vertical_start << AIPP_XS_OFFSET_LIST[3]
        context.model.write_gpr(xs_idx, x_s)
        return xs_idx

    def set_gpr_x_t(self, context, temp_env):
        """set xt gpr"""
        src_horizontal_resolution = self.imm_src_horizontal_size - 1
        xt_idx = temp_env.alloc_register()
        x_t = src_horizontal_resolution
        x_t |= self.imm_top_pad_size << AIPP_XT_OFFSET_LIST[1]
        x_t |= self.imm_botton_pad_size << AIPP_XT_OFFSET_LIST[2]
        x_t |= self.imm_left_pad_size << AIPP_XT_OFFSET_LIST[3]
        x_t |= self.imm_right_pad_size << AIPP_XT_OFFSET_LIST[4]
        x_t |= self.imm_sid << AIPP_XT_OFFSET_LIST[5]
        context.model.write_gpr(xt_idx, x_t)
        return xt_idx

    def set_spr_aipp0(self, context):
        """set aipp0 spr"""
        src_image_src_y_saddr = context.evaluate_expr(self.src0_start)

        # check over
        aipp0_register = src_image_src_y_saddr
        aipp0_register |= self.sfr_dtc_pixel_mean_ch0 << AIPP0_OFFSET_LIST[1]
        aipp0_register |= self.sfr_dtc_pixel_mean_ch1 << AIPP0_OFFSET_LIST[2]
        context.model.write_spr('AIPP_SPR_0', aipp0_register)

    def set_spr_aipp1(self, context):
        """set aipp1 spr"""
        src1_start = context.evaluate_expr(self.src1_start)

        aipp1_register = src1_start
        aipp1_register |= self.csc_enable << AIPP1_OFFSET_LIST[1]
        context.model.write_spr('AIPP_SPR_1', aipp1_register)

    def set_spr_aipp2(self, context):
        """set aipp2 spr"""
        if self.csc_enable == 1:
            aipp2_register = np.uint16(self.csc_matrix_r0_c0)
            aipp2_register |= np.uint64(
                np.uint16(self.csc_matrix_r0_c1) << AIPP2_OFFSET_LIST[1])
            aipp2_register |= np.uint64(
                np.uint16(self.csc_matrix_r0_c2) << AIPP2_OFFSET_LIST[2])
            aipp2_register |= np.uint64(
                np.uint16(self.csc_matrix_r1_c0) << AIPP2_OFFSET_LIST[3])
        else:
            aipp2_register = 0

        context.model.write_spr('AIPP_SPR_2', aipp2_register)

    def set_spr_aipp3(self, context):
        """set aipp3 spr"""
        if self.csc_enable == 1:
            aipp3_register = np.uint16(self.csc_matrix_r1_c1)
            aipp3_register |= np.uint64(
                np.uint16(self.csc_matrix_r1_c2) << AIPP3_OFFSET_LIST[1])
            aipp3_register |= np.uint64(
                np.uint16(self.csc_matrix_r2_c0) << AIPP3_OFFSET_LIST[2])
            aipp3_register |= np.uint64(
                np.uint16(self.csc_matrix_r2_c1) << AIPP3_OFFSET_LIST[3])
        else:
            aipp3_register = 0

        context.model.write_spr('AIPP_SPR_3', aipp3_register)

    def set_spr_aipp4(self, context):
        """set aipp4 spr"""
        if self.csc_enable == 1:
            aipp4_register = np.uint16(self.csc_matrix_r2_c2)
            aipp4_register |= np.uint64(
                np.uint16(self.csc_out_bias_0) << AIPP4_OFFSET_LIST[1])
            aipp4_register |= np.uint64(
                np.uint16(self.csc_out_bias_1) << AIPP4_OFFSET_LIST[2])
            aipp4_register |= np.uint64(
                np.uint16(self.csc_out_bias_2) << AIPP4_OFFSET_LIST[3])
            aipp4_register |= np.uint64(
                np.uint16(self.csc_in_bias_0) << AIPP4_OFFSET_LIST[4])
            aipp4_register |= np.uint64(
                np.uint16(self.csc_in_bias_1) << AIPP4_OFFSET_LIST[5])
            aipp4_register |= np.uint64(
                np.uint16(self.csc_in_bias_2) << AIPP4_OFFSET_LIST[6])
        else:
            aipp4_register = 0

        context.model.write_spr('AIPP_SPR_4', aipp4_register)

    def set_spr_aipp5(self, context):
        """set aipp5 spr"""
        aipp5_register = self.dtc_pixel_mean_ch0
        aipp5_register |= self.dtc_pixel_mean_ch1 << AIPP5_OFFSET_LIST[1]
        aipp5_register |= self.dtc_pixel_mean_ch2 << AIPP5_OFFSET_LIST[2]
        aipp5_register |= self.dtc_pixel_mean_ch3 << AIPP5_OFFSET_LIST[3]

        context.model.write_spr('AIPP_SPR_5', aipp5_register)

    def set_spr_aipp6(self, context):
        """set aipp6 spr"""

        aipp6_register = self.dtc_pixel_min_ch0
        aipp6_register |= self.dtc_pixel_min_ch1 << AIPP6_OFFSET_LIST[1]
        aipp6_register |= self.dtc_pixel_min_ch2 << AIPP6_OFFSET_LIST[2]
        aipp6_register |= self.dtc_pixel_min_ch3 << AIPP6_OFFSET_LIST[3]

        context.model.write_spr('AIPP_SPR_6', aipp6_register)

    def set_spr_aipp7(self, context):
        """set aipp7 spr"""
        aipp7_register = self.dtc_pixel_variance_ch0
        aipp7_register |= self.dtc_pixel_variance_ch1 << AIPP7_OFFSET_LIST[1]
        aipp7_register |= self.dtc_pixel_variance_ch2 << AIPP7_OFFSET_LIST[2]
        aipp7_register |= self.dtc_pixel_variance_ch3 << AIPP7_OFFSET_LIST[3]

        context.model.write_spr('AIPP_SPR_7', aipp7_register)

    def set_spr_aipp8(self, context):
        """set aipp8 spr"""

        aipp8_register = self.filling_hblank_ch0
        aipp8_register |= self.filling_hblank_ch1 << AIPP8_OFFSET_LIST[1]
        aipp8_register |= self.filling_hblank_ch2 << AIPP8_OFFSET_LIST[2]
        aipp8_register |= self.filling_hblank_ch3 << AIPP8_OFFSET_LIST[3]

        context.model.write_spr('AIPP_SPR_8', aipp8_register)

    def set_spr_aipp9(self, context):
        """set aipp9 spr"""
        aipp9_register = self.cpadding_spr
        aipp9_register |= self.imm_rb_swap << AIPP9_OFFSET_LIST[1]
        aipp9_register |= self.imm_uv_swap << AIPP9_OFFSET_LIST[2]
        aipp9_register |= self.imm_ax_swap << AIPP9_OFFSET_LIST[3]
        aipp9_register |= self.imm_input_format << AIPP9_OFFSET_LIST[4]
        aipp9_register |= self.imm_single_line_mode << AIPP9_OFFSET_LIST[5]
        aipp9_register |= self.horizontal_flip_enable << AIPP9_OFFSET_LIST[6]
        aipp9_register |= self.vertical_flip_enable << AIPP9_OFFSET_LIST[7]
        aipp9_register |= self.imm_area_pad_mode << AIPP9_OFFSET_LIST[8]
        aipp9_register |= self.no_padding << AIPP9_OFFSET_LIST[9]
        aipp9_register |= self.imm_raw_to_f16_n << AIPP9_OFFSET_LIST[10]
        aipp9_register |= self.imm_dtc_mean_type << AIPP9_OFFSET_LIST[11]
        aipp9_register |= self.imm_raw_image_channel << AIPP9_OFFSET_LIST[12]
        aipp9_register |= self.imm_start_channel_number << AIPP9_OFFSET_LIST[13]
        aipp9_register |= self.padd_4channels << AIPP9_OFFSET_LIST[14]
        aipp9_register |= self.sfr_dtc_pixel_mean_ch2 << AIPP9_OFFSET_LIST[15]
        aipp9_register |= self.sfr_dtc_pixel_mean_ch3 << AIPP9_OFFSET_LIST[16]

        context.model.write_spr('AIPP_SPR_9', aipp9_register)

    def set_spr_aipp10(self, context):
        """set aipp10 spr"""
        aipp10_register = self.imm_dst_stride_pixel << AIPP10_OFFSET_LIST[0]

        context.model.write_spr('AIPP_SPR_10', aipp10_register)

    def set_spr_aipp11(self, context):
        """set aipp11 spr"""

        aipp11_register = self.imm_pre_botton_clip_number << \
                          AIPP11_OFFSET_LIST[0]
        aipp11_register |= self.imm_pre_top_clip_number << \
                           AIPP11_OFFSET_LIST[1]

        context.model.write_spr('AIPP_SPR_11', aipp11_register)

    def set_spr_aipp12(self, context):
        """set aipp12 spr"""
        aipp12_register = self.spr_scf_vertical_size << AIPP12_OFFSET_LIST[0]
        aipp12_register |= self.spr_scf_horizontal_size << AIPP12_OFFSET_LIST[1]

        context.model.write_spr('AIPP_SPR_12', aipp12_register)

    def set_spr_aipp13(self, context):
        """set aipp13 spr"""
        aipp13_register = self.scf_enable << AIPP13_OFFSET_LIST[0]
        aipp13_register |= self.scf_enable << AIPP13_OFFSET_LIST[1]
        aipp13_register |= self.imm_filter_order << AIPP13_OFFSET_LIST[2]
        aipp13_register |= self.imm_scaling_mode << AIPP13_OFFSET_LIST[3]
        aipp13_register |= self.imm_scaling_mode << AIPP13_OFFSET_LIST[4]
        aipp13_register |= self.imm_scaling_mode << AIPP13_OFFSET_LIST[5]
        aipp13_register |= self.imm_scaling_mode << AIPP13_OFFSET_LIST[6]

        context.model.write_spr('AIPP_SPR_13', aipp13_register)

    def set_spr_aipp15(self, context):
        """set aippp15 spr"""

        aipp15_register = self.imm_init_vert_phase << AIPP15_OFFSET_LIST[0]
        aipp15_register |= self.imm_init_hori_phase << AIPP15_OFFSET_LIST[1]

        context.model.write_spr('AIPP_SPR_15', aipp15_register)

    def set_spr_aipp16(self, context):
        """set aipp16 spr"""
        aipp16_register = self.imm_vert_scaling << AIPP16_OFFSET_LIST[0]
        aipp16_register |= self.imm_hori_scaling << AIPP16_OFFSET_LIST[1]

        context.model.write_spr('AIPP_SPR_16', aipp16_register)

    def set_spr_aipp17(self, context):
        """set aipp17 spr"""
        aipp17_register = self.imm_post_botton_clip_number << \
                          AIPP17_OFFSET_LIST[0]
        aipp17_register |= self.imm_post_top_clip_number << \
                           AIPP17_OFFSET_LIST[1]
        aipp17_register |= self.imm_post_right_clip_number << \
                           AIPP17_OFFSET_LIST[2]
        aipp17_register |= self.imm_post_left_clip_number << \
                           AIPP17_OFFSET_LIST[3]
        aipp17_register |= self.post_clip_enable << AIPP17_OFFSET_LIST[4]

        context.model.write_spr('AIPP_SPR_17', aipp17_register)

class LoadL1ToL0AWinograd(STMT):
    """winograd_feature_map_transform instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, dst, src, # pylint: disable=R0913, R0914
                 l1_h, l1_w, l1_c, pad_left, pad_right, pad_top, pad_bottom,
                 m_extension, m_start_pt, k_extension, k_start_pt,
                 column_indicator, dst_gap):
        super(LoadL1ToL0AWinograd, self).__init__(source_info)
        self.dst = dst
        self.src = src
        self.l1_h = l1_h
        self.l1_w = l1_w
        self.l1_c = l1_c
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.m_ext = m_extension
        self.m_start = m_start_pt
        self.k_ext = k_extension
        self.k_start = k_start_pt
        self.c_idx = column_indicator
        self.dst_gap = dst_gap

    def eval_(self, context):
        temp_env = TempEnv()
        xn_idx, _, _, _ = copy_tensor_to_model(context, temp_env, self.src,
                                               ALIGN_SRC, True, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGN_TENSOR, True, access_mode='w')

        param = context.encoder.new_param()
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = self.create_gpr_x_m(context, temp_env)
        param.xt = self.create_gpr_x_t(context, temp_env)
        param.type = _WINO_TYPE_ID[self.src.dtype]

        instr = context.encoder.gen_dma_winograd_l0a(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model)
        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)

    def create_gpr_x_m(self, context, temp_env):
        """create register x_m

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xm_idx
        """
        pad_left = context.evaluate_expr(self.pad_left)
        pad_right = context.evaluate_expr(self.pad_right)
        pad_top = context.evaluate_expr(self.pad_top)
        pad_bottom = context.evaluate_expr(self.pad_bottom)

        xm_idx = temp_env.alloc_register()
        x_m = context.evaluate_expr(self.l1_w)
        x_m |= context.evaluate_expr(self.l1_h) << SHIFT_BIT_POS_16
        x_m |= context.evaluate_expr(self.l1_c) << SHIFT_BIT_POS_32
        x_m |= context.evaluate_expr(self.dst_gap) << SHIFT_BIT_POS_48
        x_m |= context.evaluate_expr(self.c_idx) << SHIFT_BIT_POS_54
        x_m |= WINO_PAD_MAP[pad_left, pad_right] << SHIFT_BIT_POS_56
        x_m |= WINO_PAD_MAP[pad_top, pad_bottom] << SHIFT_BIT_POS_59
        context.model.write_gpr(xm_idx, x_m)

        return xm_idx

    def create_gpr_x_t(self, context, temp_env):
        """create register x_t

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xt_idx
        """
        xt_idx = temp_env.alloc_register()
        x_t = 0
        x_t |= context.evaluate_expr(self.k_ext) << SHIFT_BIT_POS_8
        x_t |= context.evaluate_expr(self.k_start) << SHIFT_BIT_POS_20
        x_t |= context.evaluate_expr(self.m_ext) << SHIFT_BIT_POS_32
        x_t |= context.evaluate_expr(self.m_start) << SHIFT_BIT_POS_48
        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class LoadL1ToL0BWinograd(STMT):
    """winograd_weight_transform instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, dst, src,  # pylint: disable=R0913
                 column_indicator, repeat_dir, repeat_times, dst_blk_stride,
                 dst_rep_stride, src_rep_stride, en_weight_offset, smask):
        super(LoadL1ToL0BWinograd, self).__init__(source_info)
        self.dst = dst
        self.src = src
        self.c_idx = column_indicator
        self.repeat_dir = repeat_dir
        self.repeat = repeat_times
        self.dst_blk_stride = dst_blk_stride
        self.dst_rep_stride = dst_rep_stride
        self.src_rep_stride = src_rep_stride
        self.en_weight_offset = int(en_weight_offset)
        self.smask = smask

    def eval_(self, context):
        temp_env = TempEnv()
        xn_idx, _, _, _ = copy_tensor_to_model(context, temp_env, self.src,
                                               ALIGN_SRC, True, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGN_TENSOR, True, access_mode='w')

        param = context.encoder.new_param()
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = self.create_gpr_x_m(context, temp_env)
        param.type = _WINO_TYPE_ID[self.src.dtype]

        instr = context.encoder.gen_dma_winograd_l0b(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model)
        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)

    def create_gpr_x_m(self, context, temp_env):
        """create register x_m

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xm_idx
        """
        xm_idx = temp_env.alloc_register()
        x_m = 0
        x_m |= context.evaluate_expr(self.dst_blk_stride) << SHIFT_BIT_POS_8
        x_m |= context.evaluate_expr(self.src_rep_stride) << SHIFT_BIT_POS_16
        x_m |= context.evaluate_expr(self.dst_rep_stride) << SHIFT_BIT_POS_32
        # TIK doesn't support SMASK now
        x_m |= context.evaluate_expr(self.c_idx) << SHIFT_BIT_POS_52
        x_m |= context.evaluate_expr(self.repeat_dir) << SHIFT_BIT_POS_54
        x_m |= context.evaluate_expr(self.en_weight_offset) << SHIFT_BIT_POS_55
        x_m |= context.evaluate_expr(self.repeat) << SHIFT_BIT_POS_56

        context.model.write_gpr(xm_idx, x_m)

        return xm_idx


class MmadBrc(STMT):
    """mmad_broadcast instruction"""
    def __init__(self, source_info, dst, src, # pylint: disable=R0913
                 repeat_mode, nburst, burst_repeat, dst_gap, src_gap):
        super(MmadBrc, self).__init__(source_info)
        self.dst = dst
        self.src = src
        self.repeat_mode = repeat_mode
        self.nburst = nburst
        self.burst_repeat = burst_repeat
        self.dst_gap = dst_gap
        self.src_gap = src_gap

    def eval_(self, context):
        src_scope = TikUtil.get_storage_scope(self.src.scope)
        dst_scope = TikUtil.get_storage_scope(self.dst.scope) + \
                    get_dtype_bit_width(self.src.dtype)

        src_align = 32
        if context.dprofile.arch == 'v200':
            src_align = get_dtype_size(self.src.dtype)

        temp_env = TempEnv()

        xn_idx, _, _, _ = copy_tensor_to_model(context, temp_env, self.src,
                                               src_align)

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, _BRC_DST_ALIGN[dst_scope],
            access_mode='w')

        param = context.encoder.new_param()
        param.dst_mem_id = _BRC_DST_ID[dst_scope]
        param.src_mem_id = _BRC_SRC_ID[src_scope]
        param.conv = 0
        if _BRC_DST_ID[dst_scope] and self.src.dtype != self.dst.dtype:
            param.conv = 1
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = self.create_gpr_x_m(context, temp_env)

        instr = context.encoder.gen_dma_brc(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model)
        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)

    def create_gpr_x_m(self, context, temp_env):
        """create register x_m

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xm_idx
        """
        dst_gap = context.evaluate_expr(self.dst_gap)
        src_gap = context.evaluate_expr(self.src_gap)
        nburst = context.evaluate_expr(self.nburst)
        nburst = max(nburst, 1)
        burst_len = context.evaluate_expr(self.burst_repeat)
        burst_len = max(burst_len, 1)
        repeat_mode = context.evaluate_expr(self.repeat_mode)

        xm_idx = temp_env.alloc_register()

        x_m = nburst
        x_m |= burst_len << SHIFT_BIT_POS_8
        x_m |= src_gap << SHIFT_BIT_POS_16
        x_m |= dst_gap << SHIFT_BIT_POS_24
        x_m |= repeat_mode << SHIFT_BIT_POS_63

        context.model.write_gpr(xm_idx, x_m)

        return xm_idx
