"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     simd.py
DESC:     simd instrction
CREATED:  2019-7-04 20:12:13
MODIFIED: 2019-7-24 15:09:45
"""
# disabling:
# C0302: too-many-lines
# R0902: too-many-instance-attributes
# R0913: too-many-arguments
# pylint: disable=C0302

import sys

import numpy as np
from te.platform.cce_params import ASCEND_310
from te.platform.cce_params import ASCEND_910
from te.platform.cce_params import HI3796CV300ES
from te.platform.cce_params import ASCEND_610
from te.platform.cce_params import ASCEND_620
from te.tik.common.util import TikUtil, DTYPE_SIZE, reduce_mul, get_bit_len, \
    check_integer_in_range, check_scatter_vector_overflow, ceil_div, \
    is_basic_expr, get_mask_len
from te.tik.tik_lib.tik_check_util import TikCheckUtil
from te.tik.common.common_util import check_vector_stride, \
    check_vreduce_src1_overflow, check_sel_overflow, \
    check_vconv_deqs162b8_overflow, check_tensor_overflow, \
    check_address_overlap_vreduce, check_sel_dst_overlap, \
    check_address_overlapping, check_vconv_deqs162b8_overlap, \
    check_vec_trans_params_range, check_vec_trans_overflow, \
    check_addr_overlap_v4dtrans, align_start_pos, check_space_overflow, \
    vreduce_create_mask, check_vreduce_repeat_times, check_dtype_overflow, \
    check_vshl_vshr_scalar, is_scalar, is_tensor, \
    check_vbi_src1_tensor_overflow, check_vbi_overlap, check_over_high_preci,\
    vector_max_offset_cal
from .statement import STMT
from .statement import MoveTensor2CMPMASK
from .statement import MoveCMPMASK2Tensor
from .util import get_dtype_bit_width, copy_tensor_to_model, set_vector_mask, \
    cvt_float_to_uint, reinterpret_type, _VEC_DATA_TYPE_ENCODING, \
    copy_tensor_to_model_get_addr, get_dtype_size, get_flatten_idx, \
    debug_check_scatter_overlap, check_db_vsel_dst_sel_overlap
from .sim.util import TempEnv, check_read_mem_out_of_bounds
from .sim.instr_encoder import Encoder
from ..tik_lib.tik_params import SRC_BLOCK_STRIDE_SHIFT_POS, REPEAT_SHIFT_POS, \
    STRIDE_UNIT_SHIFT_POS, DST_REPEAT_STRIDE_SHIFT_POS, ALIGNED_ADDR, \
    SRC_REPEAT_STRIDE_SHIFT_POS, VA0_INDEX, MASK_MODE_MASK, \
    MASK_COUNTER_MODE_ENABLE_SHIFT_POS, MAXMIN_CNT_INDEX_LEN_1,\
    MAXMIN_CNT_INDEX_LEN_3, FOUR_BYTE_VALUE, TWO_BYTE_VALUE, CNT_SHIFT_POS,\
    INDEX_SHIFT_POS, DEQSCALE_46BIT_MASK, DEQSCALE_46BIT_SHIFT_POS, \
    MIN_M_LEN, MAX_M_LEN, MIN_CHANNELS, MAX_CHANNELS, MASK_VALUE_128, \
    MASK_VALUE_64, MAX_BLK_STRIDE_DOUBLE_BYTE, MAX_BLK_STRIDE_SINGLE_BYTE, \
    MAX_REPEAT_TIMES, MAX_REP_STRIDE_SINGLE_BYTE, MIN_REPEAT_TIMES, \
    ONE_REP_BYTE_SIZE, MAX_REP_STRIDE_DOUBLE_BYTE, ONE_BLK_SIZE, \
    MAX_STRIDE_UNIT, MIN_SRC1_PATTERN, MAX_SRC1_PATTERN, BLK_NUM_PER_REP, \
    VNCHWCONV_LIST_LEN, PER_TRANSPOSE_DATA_SIZE, ONE_BYTE_BIT_LEN, \
    MASK_VALUE_ZERO, MAX_VREDUCE_REPEAT_TIMES, MIN_STRIDE_UNIT, \
    VTRANSPOSE_REQUIRED_ELEMENT, VREDUCE_PER_REP_OUTPUT,\
    VREDUCE_MIN_REPEAT_TIMES, VREDUCE_DEFAULT_DST_REP_STRIDE, \
    VREDUCE_DEFAULT_SRC_BLK_STRIDE, VREDUCE_DEFAULT_SRC_REP_STRIDE,\
    MASK_LOW_IDX, MASK_HIGH_IDX
from ..common.tik_get_soc_name import get_soc_name
from ..tik_lib.tik_params import CONST_FIVE_THREE
from ..tik_lib.tik_params import CONST_ONE_THREE
from ..tik_lib.tik_params import CONST_NEG_FOUR_THREE
from ..tik_lib.tik_params import LOG_FOUR_THREE
from ..tik_lib.tik_params import CONST_NEG_ONE
from ..tik_lib.tik_params import CONST_ONE
from ..tik_lib.tik_params import CONST_HALF
from ..tik_lib.tik_params import CONST_THREE_FOUR
from ..tik_lib.tik_params import CONST_ONE_FIVE
from ..tik_lib.tik_params import CONST_NEG_ONE_FOUR
from ..tik_lib.tik_params import CONST_NEG_HALF
from ..tik_lib.tik_params import CONST_ONE_NINE
from ..tik_lib.tik_params import CONST_NEG_ONE_EIGHT
from ..tik_lib.tik_params import CONST_ONE_SEVEN
from ..tik_lib.tik_params import CONST_NEG_ONE_SIX

_ENCODER = Encoder()

_VCSPLIT_N_ENCODING = {32: 0b00, 16: 0b01, 8: 0b10}

_VCSPLIT_TYPE_ENCODING = {'8': 0b0, '16': 0b1}

_LIST_LIST_ELT = {
    'scatter_vmulva': _ENCODER.gen_vmulva,
    'scatter_vaddva': _ENCODER.gen_vaddva
}

_VSHR_DTYPE_ENCODING_V200 = {
    "uint16": 0b100,
    "uint32": 0b110,
    "int16": 0b101,
    "int32": 0b111
}

_VECTOR_SCALAR_FN_ENCODER = {
    'vmuls': _ENCODER.gen_vmulsx,
    'vadds': _ENCODER.gen_vaddsx,
    'vaxpy': _ENCODER.gen_vaxpyx,
    'vmaxs': _ENCODER.gen_vmaxsx,
    'vmins': _ENCODER.gen_vminsx,
    'vshl': _ENCODER.gen_vshlx,
    'vshr': _ENCODER.gen_vshrx,
}

_ROUNDING = {
    'round': np.round,
    'floor': np.floor,
    'ceil': np.ceil,
    'ceiling': np.ceil
}

_VCONV_BLOCK_SIZE = {('float16', 'int32'): 8, ('float16', 'uint8'): 16}

_UINT8_INFO = np.iinfo(np.uint8)

_VCONV_TYPE_ENCODING = {
    ('float32', 'float16'): 0b110000,
    ('float16', 'float32'): 0b110001,
    ('float16', 'int8'): 0b110010,
    ('float16', 'uint8'): 0b110011,
    ('DEQ', ): 0b110100,
    ('float16', 'int32', 'floor'): 0b110101,
    ('float16', 'int32', 'ceil'): 0b110110,
    ('float16', 'int32', 'round'): 0b110111,
    ('uint8', 'float16'): 0b111000,
    ('int8', 'float16'): 0b111001,
    ('float16', 'int32', 'away-zero'): 0b111010,
    ('float16', 'int32', 'to-zero'): 0b111011,
    ('float16', 'int8', 'away-zero'): 0b111100,
    ('float16', 'int8', 'floor'): 0b111101,
    ('float16', 'int8', 'ceil'): 0b111110,
    ('float16', 'int8', 'to-zero'): 0b111111,
    ('float32', 'int32', 'away-zero'): 0b100000,
    ('float32', 'int32', 'floor'): 0b100001,
    ('float32', 'int32', 'ceil'): 0b100010,
    ('float32', 'int32', 'to-zero'): 0b100011,
    ('float32', 'int32', 'round'): 0b100100,
    ('float16', 'uint8', 'away-zero'): 0b100101,
    ('float16', 'uint8', 'floor'): 0b100110,
    ('float16', 'uint8', 'ceil'): 0b100111,
    ('float16', 'uint8', 'to-zero'): 0b101000,
    ('int32', 'float32'): 0b101001,
    ('float16', 'int4'): 0b101011,
    # DEQs162b8
    ('DEQs162b8',): 0b101100,
    ('float16', 'int16', 'round'): 0b101101,
    ('int16', 'float16'): 0b101110,
    ('float32', 'int16', 'round'): 0b101111,
    ('float32', 'int16', 'to-zero'): 0b010000,
    ('int16', 'float32'): 0b010001,
    # VDEQs162b8
    ('VDEQs162b8',): 0b010010,
    ('float32', 'float16', 'odd'): 0b101010
}

_VECTOR_VECTOR_FN_ENCODER = {
    'vadd': _ENCODER.gen_vaddx,
    'vsub': _ENCODER.gen_vsubx,
    'vmul': _ENCODER.gen_vmulx,
    'vdiv': _ENCODER.gen_vdivx,
    'vmax': _ENCODER.gen_vmaxx,
    'vmin': _ENCODER.gen_vminx,
    'vand': _ENCODER.gen_vandx,
    'vor': _ENCODER.gen_vorx,
    'vmla': _ENCODER.gen_vmlax,
    'vmadd': _ENCODER.gen_vmaddx,
    'vmaddrelu': _ENCODER.gen_vmaddrelux,
    'vmulconv': _ENCODER.gen_vmulconvx,
    'vadddeqrelu': _ENCODER.gen_vadd_deq_relux,
    'vaddrelu': _ENCODER.gen_vaddrelux,
    'vsubrelu': _ENCODER.gen_vsubrelux
}

_VECTOR_DTYPE_FN_ENCODER = {
    'uint8': 0b000,
    'uint16': 0b001,
    'uint32': 0b010,
    'int8': 0b011,
    'int32': 0b100,
    'float16': 0b101,
    'int16': 0b110,
    'fmix': 0b110,
    'float32': 0b111,
}

_VEC_DATA_TYPE_ENCODING_V200 = {
    'int32': 0b00, 'float16': 0b01,
    'int16': 0b10, 'float32': 0b11
}

_VEC_RELUCONV_TYPE_ENCODING = {
    ('int16', 'int8'): 0b00,
    ('float16', 'int8'): 0b01,
    ('float32', 'float16'): 0b10
}

_B16_B32_DTYPE_CODE = {
    16: 0b01,
    32: 0b11
}

_VSHL_DTYPE_CODE = {
    "int16": 0b010,
    "uint16": 0b010,
    "int32": 0b011,
    "uint32": 0b011
}

_SPECIAL_DTYPE_INSTR = {
    'vabs': 1,
    'vcmax': 1,
    'vcmin': 1,
    'vadd': 1,
    'vsub': 1,
    'vmax': 1,
    'vmin': 1,
    'vmul': 1,
    'vdiv': 1,
    'vmadd': 1,
    'vlrelu': 0,
    'vaxpy': 1,
    'vaddrelu': 0,
    'vsubrelu': 0,
    'vadds': 1,
    'scatter_vadds': 1,
    'vmuls': 1,
    'scatter_vmuls': 1,
    'vmaxs': 0,
    'scatter_vmaxs': 0,
    'vmins': 0,
    'scatter_vmins': 0,
    'vcmp': 1,
    'vor': 0,
    'vand': 0
}

_VEC_DATA_TYPE_ENCODING_0_ = {
    'int32': 0b00,
    'uint16': 0b01,
    'int16': 0b10,
    'float16': 0b01,
    'float32': 0b11,
}

_VEC_CMP_OP_ENCODER = {
    'lt': 0b010,
    'gt': 0b011,
    'ge': 0b100,
    'eq': 0b000,
    'ne': 0b001,
    'le': 0b101
}

_VCMPV_TYPE_ENCODER = {'float16': 0, 'float32': 1}

_VCI_DTYPE = {
    'int32': 0b000,
    'float16': 0b001,
    'int16': 0b010,
    'float32': 0b011
}

_VEC_REDUCE_V100_DST_ALIGN = {
    'vcmax': 4,
    'vcmin': 4,
    'vcgadd': 16,
    'vcgmax': 16,
    'vcgmin': 16,
    'vcpadd': 32
}

_VEC_WHOLE_REDUCE_ENCODER = {
    'vcadd': _ENCODER.gen_vcadd,
    'vcmax': _ENCODER.gen_vcmax,
    'vcmin': _ENCODER.gen_vcmin,
    'vcgadd': _ENCODER.gen_vcgadd,
    'vcgmax': _ENCODER.gen_vcgmax,
    'vcgmin': _ENCODER.gen_vcgmin,
    'vcpadd': _ENCODER.gen_vcpadd
}

_SCATER_VECTOR_VECTOR_FN_ENCODER = {
    'scatter_vadd': _ENCODER.gen_vaddv,
    'scatter_vsub': _ENCODER.gen_vsubv,
    'scatter_vmul': _ENCODER.gen_vmulv,
    'scatter_vdiv': _ENCODER.gen_vdivv,
    'scatter_vmax': _ENCODER.gen_vmaxv,
    'scatter_vmin': _ENCODER.gen_vminv,
    'scatter_vmadd': _ENCODER.gen_vmaddv,
    'scatter_vmaddrelu': _ENCODER.gen_vmaddreluv,
    'scatter_vmla': _ENCODER.gen_vmlav,
}

_SCATTER_VECTOR_FOR_SUPPORT_S16 = ['scatter_vadd', 'scatter_vsub',
                                   'scatter_vmul', 'scatter_vmax',
                                   'scatter_vmin', 'scatter_vabs']

_SCATTER_VECTOR_SINGLE_FN_ENCODER = {
    'scatter_vector_mov': _ENCODER.gen_vmovv,
    'scatter_vabs': _ENCODER.gen_vabsv,
    'scatter_vexp': _ENCODER.gen_vexpv,
    'scatter_vrelu': _ENCODER.gen_vreluv,
    'scatter_vrec': _ENCODER.gen_vrecv,
    'scatter_vln': _ENCODER.gen_vlnv,
    'scatter_vrsqrt': _ENCODER.gen_vrsqrtv,
    'scatter_vsqrt': _ENCODER.gen_vsqrtv,
}

_SCATTER_VECTOR_SCALAR = {
    'scatter_vaxpy': _ENCODER.gen_vaxpyv,
    'scatter_vadds': _ENCODER.gen_vaddsv,
    'scatter_vmuls': _ENCODER.gen_vmulsv,
    'scatter_vmaxs': _ENCODER.gen_vmaxsv,
    'scatter_vmins': _ENCODER.gen_vminsv
}


_VECTOR_ONLY_FN_ENCODER = {
    'vrelu': _ENCODER.gen_vrelux,
    'vexp': _ENCODER.gen_vexpx,
    'vln': _ENCODER.gen_vlnx,
    'vabs': _ENCODER.gen_vabsx,
    'vrec': _ENCODER.gen_vrecx,
    'vrsqrt': _ENCODER.gen_vrsqrtx,
    'vnot': _ENCODER.gen_vnotx,
    'vsqrt': _ENCODER.gen_vsqrtx
}

_VMULCONV_DTYPE_ENCODING = {'int8': 0b01, 'uint8': 0b10}

_SRC_BLK_STRIDE_SHIFT_POS = 8
_SRC1_BLK_STRIDE_SHIFT_POS = 16
_DST_REPEAT_STRIDE_SHIFT_POS = 24
_SRC_REPEAT_STRIDE_SHIFT_POS = 32
_SRC1_REPEAT_STRIDE_SHIFT_POS = 40
_MODE_SHIFT_POS = 48

_INSTR_SHIFT_POS = 25
_DST_DTYPE_SHIFT_POS = 22
_PARAM_XD_SHIFT_POS = 17
_PARAM_XN_SHIFT_POS = 12
_PARAM_XT_SHIFT_POS = 2
_INSTR_OR_VALUE = 1
_MAX_PARAM_TYPE = 3


_CHANNELS_SHIFT_BIT_POS = 12
_TRANS_DIR_SHIFT_BIT_POS = 63
_PATTERN_SHIFT_POS = 16

_VPADD_DST_REP_SHIFT_POS = 32
_PAD_SIDE_SHIFT_BIT_POS = 50

_DEFAULT_STRIDE = 1
_DEFAULT_BLOCK_LEN = 1
_ROUND_TO_NEAREST_ENABLE = 0


def vadds_actual_eval_(context, temp_env, mask,  # pylint: disable=R0913
                       mask_mode, dst, src, scalar):
    """vadds eval function for vbi instruction

    Parameters
    ----------
    context : the stack context
    temp_env: the temp environment
    mask: Effective operation on element, divided into two model:
          Continuous and bit by bit.
    mask_mode: mode of mask, counter or normal.
    dst: destination tensor
    src: src operator
    scalar: scalar operator
    Returns
    -------
    None
    """

    if mask_mode == "counter":
        orig_ctrl_value = _set_mask_counter_mode(context)

    set_vector_mask(mask, context, mask_mode,
                    tensor_bit_len=max(get_bit_len(src.dtype),
                                       get_bit_len(dst.dtype)))

    param, dst_addr, dst_ptr, dst_alloc_size = vadds_gen_param(
        context, temp_env, dst, src, scalar)

    context.model.step(_ENCODER.gen_vaddsx(param))
    temp_env.check_mem_access(context.model, True)

    # mask: counter_mode, reset CTRL as orig_ctrl_value
    if mask_mode == "counter":
        context.model.write_spr('CTRL', orig_ctrl_value)

    context.model.read_memory(dst_addr, dst.scope, dst_ptr,
                              dst_alloc_size)


def vadds_gen_param(context, temp_env, dst, src, scalar):
    """generate vadds param for function vadds_actual_eval_

    Parameters
    ----------
    context : the stack context
    temp_env: the temp environment
    dst: destination tensor
    src: src operator
    scalar: scalar operator
    Returns
    -------
    param, dst_addr, dst_ptr, dst_alloc_size
    """
    align = _vec_template_align(context, src.dtype)
    xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
        context, temp_env, dst, align, True, access_mode='w')
    xn_idx, _, _, _ = copy_tensor_to_model(
        context, temp_env, src, align, True, access_mode='r')

    param = _ENCODER.new_param()
    param.type = vadds_gen_param_type(src, dst)
    param.xd = xd_idx
    param.xn = xn_idx
    param.xm = vadds_create_gpr_x_m(context, temp_env, scalar, src)
    param.xt = vadds_create_gpr_x_t(context, temp_env)
    param.out = _ROUND_TO_NEAREST_ENABLE
    return param, dst_addr, dst_ptr, dst_alloc_size


def vadds_gen_param_type(src, dst):
    """genarate type encoding param for function vadds_gen_param

    Parameters
    ----------
    src: src operator
    dst: destination tensor

    Returns
    -------
    param_type : the type encoding
    """
    dtype = src.dtype
    if dtype != dst.dtype:
        dtype = 'fmix'
    param_type = _VEC_DATA_TYPE_ENCODING_V200[dtype]
    param_type |= _SPECIAL_DTYPE_INSTR["vadds"] << 2
    return param_type


def vadds_create_gpr_x_m(context, temp_env, scalar, src):
    """create general purpose register x_m for function vadds_gen_param

    Parameters
    ----------
    context : the stack context
    temp_env : the temp environment
    scalar: scalar operator
    src: src operator

    Returns
    -------
    xm_idx
    """
    scalar = context.evaluate_expr(scalar)
    xm_idx = temp_env.alloc_register()
    x_m = cvt_float_to_uint(src.dtype, scalar)

    context.model.write_gpr(xm_idx, x_m)
    return xm_idx


def vadds_create_gpr_x_t(context, temp_env):
    """create general purpose register x_t for function vadds_gen_param

    Parameters
    ----------
    context : the stack context
    temp_env : the temp environment

    Returns
    -------
    xt_idx
    """
    repeat = 1
    dst_block_stride = 1
    src_block_stride = 1
    dst_repeat_stride = 8
    src_repeat_stride = 8
    stride_unit = 0

    xt_idx = temp_env.alloc_register()
    x_t = dst_block_stride
    x_t |= src_block_stride << SRC_BLOCK_STRIDE_SHIFT_POS
    x_t |= stride_unit << STRIDE_UNIT_SHIFT_POS
    x_t |= repeat << REPEAT_SHIFT_POS
    x_t |= dst_repeat_stride << DST_REPEAT_STRIDE_SHIFT_POS
    x_t |= src_repeat_stride << SRC_REPEAT_STRIDE_SHIFT_POS

    context.model.write_gpr(xt_idx, x_t)

    return xt_idx


def _vcadd_eval(context, mask, mask_mode,  # pylint: disable=R0913, R0914
                dst, src, repeat_times,
                src_rep_stride, dst_offset=0, src_offset=0):
    """vcadd part eval function"""
    if repeat_times == 0:
        return
    temp_env = TempEnv()
    if mask_mode == "counter":
        orig_ctrl_value = _set_mask_counter_mode(context)
    set_vector_mask(mask, context, mask_mode=mask_mode,
                    tensor_bit_len=get_bit_len(src.dtype))

    src_align = _vec_template_align(context, src.dtype)
    xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
        context, temp_env, dst, get_dtype_size(src.dtype),
        True, access_mode='w', offset=dst_offset)
    xn_idx, _, src_alloc_size, _ = copy_tensor_to_model(
        context, temp_env, src, src_align, True, access_mode='r',
        offset=src_offset)

    param = _ENCODER.new_param()
    param.type = _VEC_DATA_TYPE_ENCODING[src.dtype]
    param.xt = _vcadd_create_gpr_x_t(context, temp_env, mask, mask_mode,
                                     dst, src, repeat_times,
                                     src_rep_stride)
    param.xd = xd_idx
    param.xn = xn_idx

    # check read mem out of bounds
    check_read_mem_out_of_bounds(context, src_alloc_size, mask,
                                 src, repeat_times, 1, src_rep_stride,
                                 mask_mode=mask_mode, ori_offset=src_offset)

    context.model.step(_ENCODER.gen_vcadd(param))
    temp_env.check_mem_access(context.model, True)

    # mask: counter_mode, reset CTRL as orig_ctrl_value
    if mask_mode == "counter":
        context.model.write_spr('CTRL', orig_ctrl_value)

    context.model.read_memory(dst_addr, dst.scope, dst_ptr,
                              dst_alloc_size)


def _vcadd_create_gpr_x_t(context, temp_env,  # pylint: disable=R0913
                          mask, mask_mode,
                          dst, src, repeat_times, src_rep_stride):
    """create general purpose register x_t for vcadd function"""
    # check params
    check_integer_in_range(
        repeat_times, range(MAX_REPEAT_TIMES),
        "repeat_times should be in the range of [0, 255], "
        "input value is %s" % str(repeat_times))

    # check address overlap
    if repeat_times == 1:
        default_stride = _DEFAULT_STRIDE
    else:
        default_stride = 0
    mask_value = _eval_mask(mask, context)
    if src.buffer == dst.buffer:
        check_address_overlapping(
            "vcadd", mask_value, dst, src,
            BLK_NUM_PER_REP,
            ONE_REP_BYTE_SIZE //
            max(get_bit_len(dst.dtype),
                get_bit_len(src.dtype)),
            _DEFAULT_BLOCK_LEN, ONE_REP_BYTE_SIZE // get_bit_len(src.dtype),
            repeat_times, default_stride, _DEFAULT_STRIDE,
            _DEFAULT_STRIDE, src_rep_stride,
            context.evaluate_expr(dst.offset),
            context.evaluate_expr(src.offset),
            mask_mode=mask_mode)

    xt_idx = temp_env.alloc_register()
    x_t = _DEFAULT_STRIDE
    x_t |= _DEFAULT_STRIDE << SRC_BLOCK_STRIDE_SHIFT_POS
    x_t |= MIN_STRIDE_UNIT << STRIDE_UNIT_SHIFT_POS
    x_t |= repeat_times << REPEAT_SHIFT_POS
    x_t |= src_rep_stride << _SRC_REPEAT_STRIDE_SHIFT_POS

    context.model.write_gpr(xt_idx, x_t)

    return xt_idx


def _vec_reduce_add_first_add(context, mask, dst, src,  # pylint: disable=R0913
                              repeat_times, src_rep_stride):
    if get_soc_name() in (ASCEND_310, ASCEND_910):
        elements_per_block = ONE_BLK_SIZE // DTYPE_SIZE[src.dtype]
        # in order to align the first address of dst, 32B align
        max_repeat_times = MAX_REPEAT_TIMES - elements_per_block
    else:
        max_repeat_times = MAX_REPEAT_TIMES - 1
    for_times = repeat_times // max_repeat_times
    left_repeat_times = repeat_times % max_repeat_times

    for index in range(for_times):
        src_start_index = index*max_repeat_times*\
                          ((src_rep_stride*ONE_BLK_SIZE) //
                           DTYPE_SIZE[src.dtype])
        _vcadd_eval(
            context, mask, "normal", dst,
            src, max_repeat_times, src_rep_stride,
            dst_offset=index*max_repeat_times, src_offset=src_start_index)
    if left_repeat_times > 0:
        src_index = for_times*max_repeat_times * \
                    ((src_rep_stride*ONE_BLK_SIZE) //
                     DTYPE_SIZE[src.dtype])
        _vcadd_eval(
            context, mask, "normal", dst, src,
            left_repeat_times, src_rep_stride,
            dst_offset=for_times*max_repeat_times, src_offset=src_index)


def _eval_mask(mask, context):
    """evaluate mask

    Parameters
    ----------
    mask: Effective operation on element
    context : the stack context

    Returns
    -------
    mask_value
    """
    if isinstance(mask, (list, tuple)):
        mask_value = [context.evaluate_expr(value) for value in mask]
    else:
        mask_value = context.evaluate_expr(mask)

    return mask_value


def _set_mask_counter_mode(context):
    # save orig_ctrl_value
    orig_ctrl_value = context.model.read_spr('CTRL')
    # mask: counter_mode, set CTRL[56] as 1
    ctrl_value = orig_ctrl_value & MASK_MODE_MASK
    ctrl_value = ctrl_value | (1 << MASK_COUNTER_MODE_ENABLE_SHIFT_POS)
    context.model.write_spr('CTRL', ctrl_value)
    return orig_ctrl_value


def _create_va_reg(context, temp_env, addr_list):
    """create va register, the relationships \
    between va register and addr_list are as follows:
    vad: dst_addr_list
    van: src0_addr_list
    vam: src1_addr_list

    Parameters
    ----------
    context : the stack context

    temp_env : the temp environment

    Returns
    -------
    src_addr_list
    """
    if len(addr_list) == 16:
        # 16 need two successive VA register
        va_id, va_id1 = temp_env.alloc_va_register(2)
        context.model.write_va(va_id, addr_list[0:8])
        context.model.write_va(va_id1, addr_list[8:16])
    else:
        va_id = temp_env.alloc_va_register()
        context.model.write_va(va_id, addr_list)

    return va_id


def _get_src_addr_list(context, temp_env, src_list):
    """get src_addr_list

    Parameters
    ----------
    context : the stack context

    temp_env : the temp environment

    Returns
    -------
    src_addr_list
    """
    src_addr_list = []

    for src in src_list:
        copy_tensor_to_model(
            context, temp_env, src, ALIGNED_ADDR,
            require_xt=False, access_mode='r')
        src_addr = temp_env.get_tensor_addr(context, src, access_mode='r')
        src_addr_list.append(src_addr)

    return src_addr_list


def _get_dst_addr_set_list(context, temp_env, dst_list, spec_instr=False):
    """get dst_addr_set and dst_addr_list

    Parameters
    ----------
    context : the stack context

    temp_env : the temp environment

    dst_list

    Returns
    -------
    dst_addr_set

    dst_addr_list
    """
    dst_addr_set = set()
    dst_addr_list = []

    for dst in dst_list:
        if spec_instr and (get_soc_name() in (ASCEND_310, ASCEND_910)):
            # special instr
            # for v100 pvmodel, dst is rw mode;
            # for v200 pvmodel, dst is only w mode
            _, dst_buffer_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
                context, temp_env, dst, ALIGNED_ADDR,
                require_xt=False, access_mode='rw')
        else:
            _, dst_buffer_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
                context, temp_env, dst, ALIGNED_ADDR,
                require_xt=False, access_mode='w')
        dst_addr_set.add(
            (dst.scope, dst_buffer_addr, dst_alloc_size, dst_ptr))
        if spec_instr and (get_soc_name() in (ASCEND_310, ASCEND_910)):
            # special instr
            dst_addr = temp_env.get_tensor_addr(context, dst, access_mode='rw')
        else:
            dst_addr = temp_env.get_tensor_addr(context, dst, access_mode='w')
        dst_addr_list.append(dst_addr)

    return dst_addr_set, dst_addr_list


def _vec_template_align(ctx, dtype):
    """vector instr align

    Parameters
    ----------
    ctx: the stack context
    dtype: vector data type

    """
    if get_soc_name() in (ASCEND_310, ASCEND_910):
        return ALIGNED_ADDR
    if get_soc_name() in (HI3796CV300ES, ASCEND_610, ASCEND_620):
        return get_dtype_size(dtype)
    return TikCheckUtil.raise_error(
        'core_arch is not exist, it should be v100 or v200')


def _eval_mask_high(mask, context):
    if isinstance(mask, (list, tuple)):
        if isinstance(mask[0], int):
            mask_value = _eval_mask(mask, context)
        else:
            mask_value = np.array(
                [context.evaluate_expr(mask[MASK_HIGH_IDX]),
                 context.evaluate_expr(mask[MASK_LOW_IDX])]).astype(np.uint64)
    else:
        mask_value = _eval_mask(mask, context)
    return mask_value


def _fp162fp32_func_compute(source_info,  # pylint: disable=R0913, R0914
                            context, func, mask, dst, src, work_tensor,
                            repeat_times, dst_rep_stride, src_rep_stride,
                            fp32_rep_stride, tmp_tensor_size, name):
    if work_tensor.dtype == "float16":
        tmp_src_tensor = \
            work_tensor[0:tmp_tensor_size].reinterpret_cast_to("float32")
        tmp_dst_tensor = \
            work_tensor[tmp_tensor_size:
                        tmp_tensor_size*2].reinterpret_cast_to("float32")
        tmp_work_tensor = \
            work_tensor[tmp_tensor_size*2:].reinterpret_cast_to("float32")
        tmp_tensor_size = tmp_tensor_size*DTYPE_SIZE[work_tensor.dtype] //\
            DTYPE_SIZE["float32"]
    else:
        tmp_src_tensor = work_tensor[0:tmp_tensor_size]
        tmp_dst_tensor = work_tensor[tmp_tensor_size:tmp_tensor_size*2]
        tmp_work_tensor = work_tensor[tmp_tensor_size*2:]

    vconv_stmt_src = Vconv(
        source_info, mask, "none",
        tmp_src_tensor, src, repeat_times,
        1, 1, fp32_rep_stride, src_rep_stride, print_name=name)
    vconv_stmt_src.eval_(context)

    func(context, mask, tmp_dst_tensor, tmp_src_tensor, tmp_work_tensor,
         repeat_times, fp32_rep_stride, fp32_rep_stride, tmp_tensor_size)

    vconv_stmt_src = Vconv(
        source_info, mask, "none",
        dst, tmp_dst_tensor, repeat_times,
        1, 1, dst_rep_stride, fp32_rep_stride, print_name=name)
    vconv_stmt_src.eval_(context)


def _get_wk_tensor_stride(src_rep_stride):
    if src_rep_stride <= 4:
        fp32_rep_stride_1 = src_rep_stride * 2
    else:
        fp32_rep_stride_1 = 8
    return fp32_rep_stride_1


def _get_src_extend(mask, repeat_times,    # pylint: disable=R0913, R0914
                    src_rep_stride, work_tensor, size_factor):
    default_dtype = "float16"
    block_len = 16
    src_extend = vector_max_offset_cal(
        mask, default_dtype, block_len, repeat_times, 1, src_rep_stride)
    src_extend = ceil_div(src_extend, block_len)*block_len
    if work_tensor.dtype == "float16":
        tmp_tensor_size = src_extend*2
    else:
        tmp_tensor_size = src_extend
    needed_tensor_size = tmp_tensor_size*size_factor
    work_tensor_size = work_tensor.size
    # check if int
    TikCheckUtil.check_ge(work_tensor_size, needed_tensor_size,
                          "Input work tensor size(%d) must be more "
                          "than needed size(%d)" %
                          (work_tensor_size, needed_tensor_size))
    return tmp_tensor_size


def _fp162fp32_func_mask_list(source_info,    # pylint: disable=R0913, R0914
                              context, func, mask, dst, src,
                              work_tensor, repeat_times, dst_rep_stride,
                              src_rep_stride, fp32_rep_stride,
                              size_factor, name):
    default_start_offset = 64
    low_mask = mask[MASK_LOW_IDX]
    high_mask = mask[MASK_HIGH_IDX]
    if high_mask > 0:
        tmp_tensor_size = _get_src_extend(
            [0, high_mask], repeat_times,
            src_rep_stride, work_tensor, size_factor)
        _fp162fp32_func_compute(
            source_info, context, func, [0, high_mask],
            dst[default_start_offset:], src[default_start_offset:],
            work_tensor, repeat_times, dst_rep_stride, src_rep_stride,
            fp32_rep_stride, tmp_tensor_size, name)

    tmp_tensor_size = _get_src_extend(
        [0, low_mask], repeat_times,
        src_rep_stride, work_tensor, size_factor)
    _fp162fp32_func_compute(
        source_info, context, func, [0, low_mask], dst, src,
        work_tensor, repeat_times, dst_rep_stride, src_rep_stride,
        fp32_rep_stride, tmp_tensor_size, name)


def _fp162fp32_func_mask_imm(source_info,    # pylint: disable=R0913, R0914
                             context, func, mask, dst, src,
                             work_tensor, repeat_times, dst_rep_stride,
                             src_rep_stride, fp32_rep_stride,
                             size_factor, name):
    default_start_offset = 64
    if mask <= 64:
        tmp_tensor_size = _get_src_extend(
            mask, repeat_times, src_rep_stride, work_tensor, size_factor)
        _fp162fp32_func_compute(
            source_info, context, func, mask, dst, src, work_tensor,
            repeat_times, dst_rep_stride, src_rep_stride,
            fp32_rep_stride, tmp_tensor_size, name)
    else:
        low_mask = 64
        high_mask = mask - 64
        tmp_tensor_size = _get_src_extend(
            high_mask, repeat_times, src_rep_stride, work_tensor, size_factor)
        _fp162fp32_func_compute(
            source_info, context, func, high_mask, dst[default_start_offset:],
            src[default_start_offset:], work_tensor,
            repeat_times, dst_rep_stride, src_rep_stride,
            fp32_rep_stride, tmp_tensor_size, name)
        tmp_tensor_size = _get_src_extend(
            low_mask, repeat_times, src_rep_stride, work_tensor, size_factor)
        _fp162fp32_func_compute(
            source_info, context, func, low_mask, dst, src, work_tensor,
            repeat_times, dst_rep_stride, src_rep_stride,
            fp32_rep_stride, tmp_tensor_size, name)


def _fp162fp32_high_preci_func(source_info,    # pylint: disable=R0913
                               context, func, mask, dst, src,
                               work_tensor, repeat_times, dst_rep_stride,
                               src_rep_stride, size_factor, name):
    fp32_rep_stride = _get_wk_tensor_stride(src_rep_stride)
    if isinstance(mask, (list, tuple)):
        _fp162fp32_func_mask_list(source_info, context, func, mask, dst, src,
                                  work_tensor, repeat_times, dst_rep_stride,
                                  src_rep_stride, fp32_rep_stride,
                                  size_factor, name)
    else:
        _fp162fp32_func_mask_imm(source_info, context, func, mask, dst, src,
                                 work_tensor, repeat_times, dst_rep_stride,
                                 src_rep_stride, fp32_rep_stride,
                                 size_factor, name)


class VectorOnlyTemplate(STMT):
    """this template only have vector"""
    # pylint: disable=R0902
    def __init__(self, source_info, name, mask, dst, src, repeat_times,
                 dst_block_stride, src_block_stride, dst_repeat_stride,
                 src_repeat_stride, stride_unit, print_name=None):
        # pylint: disable=R0913
        super(VectorOnlyTemplate, self).__init__(source_info)
        self.fn_name = name
        self.mask = mask
        self.dst = dst
        self.src = src
        self.repeat_times = repeat_times
        self.dst_block_stride = dst_block_stride
        self.src_block_stride = src_block_stride
        self.dst_repeat_stride = dst_repeat_stride
        self.src_repeat_stride = src_repeat_stride
        self.stride_unit = stride_unit
        self.print_name = print_name
        if self.print_name is None:
            self.print_name = name

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        mask = self.mask

        set_vector_mask(mask, context,
                        tensor_bit_len=max(get_bit_len(self.src.dtype),
                                           get_bit_len(self.dst.dtype)))
        align = _vec_template_align(context, self.src.dtype)
        temp_env = TempEnv()

        xn_idx, _, src_alloc_size, _ = copy_tensor_to_model(
            context, temp_env, self.src, align, True, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, align, True, access_mode='w')

        param = _ENCODER.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[self.src.dtype]
        if self.fn_name in _SPECIAL_DTYPE_INSTR:
            param.type = _VEC_DATA_TYPE_ENCODING_V200[self.src.dtype]
            param.type |= _SPECIAL_DTYPE_INSTR[self.fn_name] << 2
        param.xd = xd_idx
        param.xn = xn_idx
        param.xt = self.create_gpr_x_t(context, temp_env)

        check_read_mem_out_of_bounds(context, src_alloc_size, self.mask,
                                     self.src, self.repeat_times,
                                     self.src_block_stride,
                                     self.src_repeat_stride)

        instr = _VECTOR_ONLY_FN_ENCODER[self.fn_name](param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

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
        repeat = context.evaluate_expr(self.repeat_times)
        # check params
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times is {}".format(repeat))
        check_vector_stride([context.evaluate_expr(self.dst_block_stride),
                             context.evaluate_expr(self.src_block_stride)],
                            [context.evaluate_expr(self.dst_repeat_stride),
                             context.evaluate_expr(self.src_repeat_stride)],
                            MAX_BLK_STRIDE_DOUBLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE, ["dst", "src"])
        check_integer_in_range(context.evaluate_expr(self.stride_unit),
                               range(MAX_STRIDE_UNIT),
                               "stride_unit should be in the range of [0, 3]")

        # check address overlap
        mask_value = _eval_mask(self.mask, context)
        if self.src.buffer == self.dst.buffer:
            if repeat == 0:
                pass
            elif context.evaluate_expr(self.dst_block_stride) == 0:
                check_address_overlapping(
                    self.print_name, mask_value, self.dst, self.src,
                    BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE //
                    max(get_bit_len(self.dst.dtype), get_bit_len(self.src.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype),
                    repeat, 1, context.evaluate_expr(self.src_block_stride),
                    context.evaluate_expr(self.dst_repeat_stride),
                    context.evaluate_expr(self.src_repeat_stride),
                    context.evaluate_expr(self.dst.offset),
                    context.evaluate_expr(self.src.offset),
                    context.evaluate_expr(self.stride_unit))
            else:
                check_address_overlapping(
                    self.print_name, mask_value, self.dst, self.src,
                    BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE //
                    max(get_bit_len(self.dst.dtype), get_bit_len(self.src.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype),
                    repeat, context.evaluate_expr(self.dst_block_stride),
                    context.evaluate_expr(self.src_block_stride),
                    context.evaluate_expr(self.dst_repeat_stride),
                    context.evaluate_expr(self.src_repeat_stride),
                    context.evaluate_expr(self.dst.offset),
                    context.evaluate_expr(self.src.offset),
                    context.evaluate_expr(self.stride_unit))

        xt_idx = temp_env.alloc_register()
        x_t = context.evaluate_expr(self.dst_block_stride)
        x_t |= context.evaluate_expr(self.src_block_stride) << \
               SRC_BLOCK_STRIDE_SHIFT_POS

        stride_unit = context.evaluate_expr(self.stride_unit)
        x_t |= stride_unit << STRIDE_UNIT_SHIFT_POS
        x_t |= repeat << REPEAT_SHIFT_POS
        x_t |= context.evaluate_expr(self.dst_repeat_stride) << \
               DST_REPEAT_STRIDE_SHIFT_POS
        x_t |= context.evaluate_expr(self.src_repeat_stride) << \
               SRC_REPEAT_STRIDE_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class VrsqrtHighPreci(STMT):    # pylint: disable=R0902
    """this stmt for vrsqrt high precision"""
    def __init__(self, source_info, mask, dst, src,    # pylint: disable=R0913
                 work_tensor, repeat_times, dst_rep_stride, src_rep_stride):
        super(VrsqrtHighPreci, self).__init__(source_info)
        self.name = "vec_rsqrt_high_preci"
        self.mask = mask
        self.dst = dst
        self.src = src
        self.work_tensor = work_tensor
        self.repeat_times = repeat_times
        self.dst_rep_stride = dst_rep_stride
        self.src_rep_stride = src_rep_stride

    def eval_(self, context):
        """exce instr"""
        set_vector_mask(self.mask, context,
                        tensor_bit_len=max(get_bit_len(self.src.dtype),
                                           get_bit_len(self.dst.dtype)))
        mask = _eval_mask(self.mask, context)
        multi_factor = 3
        if get_soc_name() == ASCEND_310:
            multi_factor = 4
        if self.src.dtype == "float16":
            # 4B of fp32, need keep 32B algin
            multi_factor += 2
        # step for check params, overflow, overlap
        check_integer_in_range(
            context.evaluate_expr(self.repeat_times), range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times is {}".format(
                context.evaluate_expr(self.repeat_times)))
        check_vector_stride(None,
                            [context.evaluate_expr(self.dst_rep_stride),
                             context.evaluate_expr(self.src_rep_stride)],
                            None, MAX_REP_STRIDE_SINGLE_BYTE, ["dst", "src"])
        check_over_high_preci(mask, self.dst, self.src, self.work_tensor,
                              context.evaluate_expr(self.repeat_times),
                              context.evaluate_expr(self.dst_rep_stride),
                              context.evaluate_expr(self.src_rep_stride),
                              context.evaluate_expr(self.dst.offset),
                              context.evaluate_expr(self.src.offset),
                              context.evaluate_expr(self.work_tensor.offset),
                              multi_factor, name="vec_rsqrt_high_preci")
        if self.src.dtype == "float16":
            # vconv: fp16->fp32
            if get_soc_name() == ASCEND_310:
                # debug mini
                # new mask, stride and repeat!!!!!!!!!!!
                _fp162fp32_high_preci_func(
                    self.source_info, context, self._vrsqrt_hmini,
                    mask, self.dst, self.src, self.work_tensor,
                    context.evaluate_expr(self.repeat_times),
                    context.evaluate_expr(self.dst_rep_stride),
                    context.evaluate_expr(self.src_rep_stride),
                    multi_factor, self.name)
            else:
                # debug cloud
                # new mask, stride and repeat!!!!!!!!!!!
                _fp162fp32_high_preci_func(
                    self.source_info, context, self._vrsqrt_hcloud,
                    mask, self.dst, self.src, self.work_tensor,
                    context.evaluate_expr(self.repeat_times),
                    context.evaluate_expr(self.dst_rep_stride),
                    context.evaluate_expr(self.src_rep_stride),
                    multi_factor, self.name)
            return

        block_len = ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype)
        src_extend = vector_max_offset_cal(
            mask, self.src.dtype, block_len,
            context.evaluate_expr(self.repeat_times), 1,
            context.evaluate_expr(self.src_rep_stride))
        tensor_split_size = ceil_div(src_extend, block_len)*block_len
        # cloud
        if get_soc_name() == ASCEND_910:
            self._vrsqrt_hcloud(context, mask,
                                self.dst, self.src, self.work_tensor,
                                context.evaluate_expr(self.repeat_times),
                                context.evaluate_expr(self.dst_rep_stride),
                                context.evaluate_expr(self.src_rep_stride),
                                tensor_split_size)
        else:
            # mini
            self._vrsqrt_hmini(context, mask,
                               self.dst, self.src, self.work_tensor,
                               context.evaluate_expr(self.repeat_times),
                               context.evaluate_expr(self.dst_rep_stride),
                               context.evaluate_expr(self.src_rep_stride),
                               tensor_split_size)

    def _vrsqrt_hcloud(self, context, mask,    # pylint: disable=R0913, R0914
                       dst, src, work_tensor, repeat_times,
                       dst_rep_stride, src_rep_stride, tensor_split_size):
        """high preci for cloud"""
        default_stride = 1
        stride_uint = 0

        # cloud
        work_tensor0 = work_tensor[0:tensor_split_size]
        work_tensor1 = work_tensor[tensor_split_size:]
        vsqrt_stmt = VectorOnlyTemplate(
            self.source_info, "vsqrt", mask,
            work_tensor0, src, repeat_times,
            default_stride, default_stride, src_rep_stride,
            src_rep_stride, stride_uint, print_name=self.name)
        vsqrt_stmt.eval_(context)

        # vrec newton debug
        hvrec_stmt_ = VrecHighPreci(
            self.source_info, mask, dst, work_tensor0,
            work_tensor1, repeat_times, dst_rep_stride,
            src_rep_stride)
        hvrec_stmt_.eval_(context)

    def _vrsqrt_hmini(self, context, mask,    # pylint: disable=R0913, R0914
                      dst, src, work_tensor, repeat_times,
                      dst_rep_stride, src_rep_stride, tensor_split_size):
        """high preci for mini"""
        default_stride = 1
        stride_uint = 0
        # mini debug
        work_tensor1 = work_tensor[0:tensor_split_size]
        work_tensor2 = work_tensor[tensor_split_size:2*tensor_split_size]
        work_tensor3 = work_tensor[2*tensor_split_size:]
        vrsqrt_stmt = VectorOnlyTemplate(
            self.source_info, "vrsqrt", mask,
            dst, src, repeat_times,
            default_stride, default_stride, dst_rep_stride,
            src_rep_stride, stride_uint, print_name=self.name)
        vrsqrt_stmt.eval_(context)
        iteration_times = 2
        for _ in range(iteration_times):
            vmul_stmt_ = VectorVectorTemplate(
                self.source_info, "vmul", mask, work_tensor1,
                src, dst, repeat_times,
                default_stride, default_stride, default_stride,
                src_rep_stride, src_rep_stride, dst_rep_stride,
                stride_uint, print_name=self.name)
            vmul_stmt_.eval_(context)
            hvrec_stmt_ = VrecHighPreci(
                self.source_info, mask, work_tensor2,
                work_tensor1, work_tensor3, repeat_times,
                src_rep_stride, src_rep_stride)
            hvrec_stmt_.eval_(context)
            vadd_stmt_ = VectorVectorTemplate(
                self.source_info, "vadd", mask, work_tensor1,
                dst, work_tensor2, repeat_times,
                default_stride, default_stride, default_stride,
                src_rep_stride, dst_rep_stride, src_rep_stride,
                stride_uint, print_name=self.name)
            vadd_stmt_.eval_(context)
            vmuls_stmt_ = VectorScalarTemplate(
                self.source_info, "vmuls", mask,
                dst, work_tensor1, 0.5, repeat_times,
                default_stride, default_stride, dst_rep_stride,
                src_rep_stride, stride_uint, 0, print_name=self.name)
            vmuls_stmt_.eval_(context)


class VlnHighPreci(STMT):  # pylint: disable=R0902
    """this stmt for vln high precision"""
    def __init__(self, source_info, mask, dst, src,    # pylint: disable=R0913
                 work_tensor, repeat_times, dst_rep_stride, src_rep_stride):
        super(VlnHighPreci, self).__init__(source_info)
        self.name = "vec_ln_high_preci"
        self.mask = mask
        self.dst = dst
        self.src = src
        self.work_tensor = work_tensor
        self.repeat_times = repeat_times
        self.dst_rep_stride = dst_rep_stride
        self.src_rep_stride = src_rep_stride
        self.default_blk_stride = 1
        self.stride_unit = 0
        self.blk_data_nums = 16

    def _vadds_vmul(self, context, mask,    # pylint: disable=R0913
                    dst_data, src_data,
                    tmp_work_tensor, scalar_value,
                    repeat_times, src_rep_stride):
        vadds_stmt_ = VectorScalarTemplate(
            self.source_info, "vadds", mask, tmp_work_tensor, dst_data,
            scalar_value, repeat_times, self.default_blk_stride,
            self.default_blk_stride, src_rep_stride, src_rep_stride,
            self.stride_unit, 0, print_name=self.name)
        vadds_stmt_.eval_(context)
        vmul_stmt0_ = VectorVectorTemplate(
            self.source_info, "vmul", mask, dst_data, src_data,
            tmp_work_tensor, repeat_times, self.default_blk_stride,
            self.default_blk_stride, self.default_blk_stride,
            src_rep_stride, src_rep_stride, src_rep_stride,
            self.stride_unit, print_name=self.name)
        vmul_stmt0_.eval_(context)

    def _taylor_compute_five(self, context,    # pylint: disable=R0913
                             mask, dst_data,
                             src_data, tmp_work_tensor,
                             repeat_times, src_rep_stride):
        vmuls_stmt_ = VectorScalarTemplate(
            self.source_info, "vmuls", mask, dst_data, src_data,
            CONST_ONE_FIVE, repeat_times, self.default_blk_stride,
            self.default_blk_stride, src_rep_stride, src_rep_stride,
            self.stride_unit, 0, print_name=self.name)
        vmuls_stmt_.eval_(context)

        self._vadds_vmul(context, mask, dst_data, src_data,
                         tmp_work_tensor, CONST_NEG_ONE_FOUR,
                         repeat_times, src_rep_stride)
        self._vadds_vmul(context, mask, dst_data, src_data,
                         tmp_work_tensor, CONST_ONE_THREE,
                         repeat_times, src_rep_stride)
        self._vadds_vmul(context, mask, dst_data, src_data,
                         tmp_work_tensor, CONST_NEG_HALF,
                         repeat_times, src_rep_stride)
        self._vadds_vmul(context, mask, dst_data, src_data,
                         tmp_work_tensor, CONST_ONE,
                         repeat_times, src_rep_stride)

    def _taylor_compute_nine(self, context,    # pylint: disable=R0913
                             mask, dst_data,
                             src_data, tmp_work_tensor,
                             repeat_times, src_rep_stride):
        vmuls_stmt_ = VectorScalarTemplate(
            self.source_info, "vmuls", mask, dst_data, src_data,
            CONST_ONE_NINE, repeat_times, self.default_blk_stride,
            self.default_blk_stride, src_rep_stride, src_rep_stride,
            self.stride_unit, 0, print_name=self.name)
        vmuls_stmt_.eval_(context)

        self._vadds_vmul(context, mask, dst_data, src_data, tmp_work_tensor,
                         CONST_NEG_ONE_EIGHT, repeat_times, src_rep_stride)
        self._vadds_vmul(context, mask, dst_data, src_data,
                         tmp_work_tensor, CONST_ONE_SEVEN,
                         repeat_times, src_rep_stride)
        self._vadds_vmul(context, mask, dst_data, src_data,
                         tmp_work_tensor, CONST_NEG_ONE_SIX,
                         repeat_times, src_rep_stride)
        self._vadds_vmul(context, mask, dst_data, src_data,
                         tmp_work_tensor, CONST_ONE_FIVE,
                         repeat_times, src_rep_stride)
        self._vadds_vmul(context, mask, dst_data, src_data,
                         tmp_work_tensor, CONST_NEG_ONE_FOUR,
                         repeat_times, src_rep_stride)
        self._vadds_vmul(context, mask, dst_data, src_data,
                         tmp_work_tensor, CONST_ONE_THREE,
                         repeat_times, src_rep_stride)
        self._vadds_vmul(context, mask, dst_data, src_data,
                         tmp_work_tensor, CONST_NEG_HALF,
                         repeat_times, src_rep_stride)
        self._vadds_vmul(context, mask, dst_data, src_data,
                         tmp_work_tensor, CONST_ONE,
                         repeat_times, src_rep_stride)

    def _ln_compute_block_lt_5_3_gt_1(self,    # pylint: disable=R0913, R0914
                                      context,
                                      mask, dst_data, src_data,
                                      tmp_work_tensor, repeat_times,
                                      src_rep_stride, src_data_size,
                                      src_offset):
        # const_neg_one + const_neg_one * const_log_threshold_2
        vadds_stmt = VectorScalarTemplate(
            self.source_info, "vadds", mask, tmp_work_tensor[0:src_data_size],
            src_data, CONST_NEG_FOUR_THREE, repeat_times,
            self.default_blk_stride, self.default_blk_stride, src_rep_stride,
            src_rep_stride, self.stride_unit, 0, print_name=self.name)
        vadds_stmt.eval_(context)
        vmuls_stmt = VectorScalarTemplate(
            self.source_info, "vmuls", mask,
            tmp_work_tensor[src_data_size:src_data_size * 2],
            tmp_work_tensor[0:src_data_size],
            CONST_THREE_FOUR, repeat_times, self.default_blk_stride,
            self.default_blk_stride, src_rep_stride, src_rep_stride,
            self.stride_unit, 0, print_name=self.name)
        vmuls_stmt.eval_(context)
        vector_dup_stmt = VectorScalarEltwise(
            self.source_info, "vector_dup", mask,
            tmp_work_tensor[src_data_size*2:src_data_size*3],
            CONST_ONE_THREE, repeat_times, self.default_blk_stride,
            src_rep_stride, self.stride_unit, mask_mode="normal")
        vector_dup_stmt.eval_(context)

        tmp_tensor = tmp_work_tensor[src_data_size*3:
                                     src_data_size*3 +
                                     repeat_times*8].\
            reinterpret_cast_to("uint64")

        vadds_stmt1 = VectorScalarTemplate(
            self.source_info, "vadds", mask, tmp_work_tensor, src_data,
            CONST_NEG_ONE, repeat_times, self.default_blk_stride,
            self.default_blk_stride, src_rep_stride, src_rep_stride,
            self.stride_unit, 0, print_name=self.name)
        vadds_stmt1.eval_(context)

        for index in range(repeat_times):
            vcmd_stmt = VCMP(
                self.source_info, "vmcp_ge", mask,
                tmp_work_tensor[index * src_offset],
                tmp_work_tensor[src_data_size * 2 + index * src_offset],
                self.default_blk_stride, self.default_blk_stride)
            vcmd_stmt.eval_(context)
            cmpmask_stmt = MoveCMPMASK2Tensor(self.source_info,
                                              tmp_tensor[index * 2])
            cmpmask_stmt.eval_(context)
            vsel_stmt = VSEL(
                self.source_info, mask, 0,
                dst_data[index * src_offset], vcmd_stmt,
                tmp_work_tensor[src_data_size + index * src_offset],
                tmp_work_tensor[index * src_offset], 1,
                self.default_blk_stride,
                self.default_blk_stride, self.default_blk_stride,
                src_rep_stride, src_rep_stride, src_rep_stride,
                name=self.name)
            vsel_stmt.eval_(context)

        self._taylor_compute_five(
            context, mask, tmp_work_tensor[0:src_data_size], dst_data,
            tmp_work_tensor[src_data_size*2:src_data_size*3],
            repeat_times, src_rep_stride)

        # phase3: add log(4/3)
        vadds_stmt_ = VectorScalarTemplate(
            self.source_info, "vadds", mask,
            tmp_work_tensor[src_data_size:src_data_size * 2],
            tmp_work_tensor[0:src_data_size], LOG_FOUR_THREE,
            repeat_times, self.default_blk_stride,
            self.default_blk_stride, src_rep_stride, src_rep_stride,
            self.stride_unit, 0, print_name=self.name)
        vadds_stmt_.eval_(context)

        for index in range(repeat_times):
            cmpmask_stmt = MoveTensor2CMPMASK(self.source_info,
                                              tmp_tensor[index * 2])
            cmpmask_stmt.eval_(context)
            vsel_stmt = VSEL(
                self.source_info, mask, 0,
                dst_data[index*src_offset], cmpmask_stmt,
                tmp_work_tensor[src_data_size + index * src_offset],
                tmp_work_tensor[index*src_offset], 1,
                self.default_blk_stride,
                self.default_blk_stride, self.default_blk_stride,
                src_rep_stride, src_rep_stride, src_rep_stride,
                name='vec_ln_high_preci')
            vsel_stmt.eval_(context)

    def _ln_compute_block_gt_5_3(self,    # pylint: disable=R0913, R0914
                                 context, mask, dst_data, tmp_dst_data,
                                 src_data, tmp_work_tensor, repeat_times,
                                 src_rep_stride, src_data_size, src_offset):
        """
        when src_data > 5/3, use vlog directly
        Parameters
        ----------
        src_data: input tensor that we want to calculate log

        Returns
        -------
        res : return of log

        """
        # if src_data > 5/3, use vlog
        vln_stmt = VectorOnlyTemplate(
            self.source_info, "vln", mask,
            tmp_work_tensor[src_data_size:src_data_size * 2],
            src_data, repeat_times, self.default_blk_stride,
            self.default_blk_stride, src_rep_stride, src_rep_stride,
            0, print_name=self.name)
        vln_stmt.eval_(context)
        vector_dup_stmt = VectorScalarEltwise(
            self.source_info, "vector_dup", mask,
            tmp_work_tensor[0:src_data_size], CONST_FIVE_THREE,
            repeat_times, self.default_blk_stride,
            src_rep_stride, stride_unit=0, mask_mode="normal")
        vector_dup_stmt.eval_(context)

        for index in range(repeat_times):
            vcmd_stmt = VCMP(
                self.source_info, "vmcp_ge", mask,
                src_data[index * src_offset],
                tmp_work_tensor[index * src_offset],
                self.default_blk_stride, self.default_blk_stride)
            vcmd_stmt.eval_(context)
            vsel_stmt = VSEL(
                self.source_info, mask, 0,
                dst_data[index*src_offset], vcmd_stmt,
                tmp_work_tensor[src_data_size + index*src_offset],
                tmp_dst_data[index * src_offset], 1,
                self.default_blk_stride,
                self.default_blk_stride, self.default_blk_stride,
                src_rep_stride, src_rep_stride, src_rep_stride,
                name='vec_ln_high_preci')
            vsel_stmt.eval_(context)

    def _ln_compute_block_gt_half_lt_1(self,  # pylint: disable=R0913, R0914
                                       context, mask, dst_data,
                                       tmp_dst_data, src_data, tmp_work_tensor,
                                       repeat_times, src_rep_stride,
                                       src_data_size, src_offset):
        vadds_stmt_ = VectorScalarTemplate(
            self.source_info, "vadds", mask, tmp_work_tensor, src_data,
            CONST_NEG_ONE, repeat_times, self.default_blk_stride,
            self.default_blk_stride, src_rep_stride, src_rep_stride,
            self.stride_unit, 0, print_name=self.name)
        vadds_stmt_.eval_(context)
        self._taylor_compute_nine(
            context, mask, tmp_work_tensor[src_data_size:src_data_size*2],
            tmp_work_tensor,
            tmp_work_tensor[src_data_size*2:src_data_size*3],
            repeat_times, src_rep_stride)

        vector_dup_stmt = VectorScalarEltwise(
            self.source_info, "vector_dup", mask,
            tmp_work_tensor, CONST_ONE,
            repeat_times, self.default_blk_stride, src_rep_stride,
            stride_unit=0, mask_mode="normal")
        vector_dup_stmt.eval_(context)

        for index in range(repeat_times):
            vcmd_stmt = VCMP(
                self.source_info, "vmcp_le", mask,
                src_data[index * src_offset],
                tmp_work_tensor[index * src_offset],
                self.default_blk_stride, self.default_blk_stride)
            vcmd_stmt.eval_(context)

            vsel_stmt = VSEL(
                self.source_info, mask, 0,
                dst_data[index*src_offset], vcmd_stmt,
                tmp_work_tensor[src_data_size + index * src_offset],
                tmp_dst_data[index * src_offset], 1,
                self.default_blk_stride,
                self.default_blk_stride, self.default_blk_stride,
                src_rep_stride, src_rep_stride, src_rep_stride,
                name='vec_ln_high_preci')
            vsel_stmt.eval_(context)

    def _call_vrec_high_preci(self,  # pylint: disable=R0913
                              context, mask, dst, src, work_tensor,
                              repeat_times, dst_rep_stride, src_rep_stride,
                              tmp_tensor_size):  # pylint: disable=W0613
        hvrec_stmt_ = VrecHighPreci(
            self.source_info, mask, dst, src, work_tensor,
            repeat_times, dst_rep_stride, src_rep_stride)
        hvrec_stmt_.eval_(context)

    def _ln_compute_block_lt_half(self,  # pylint: disable=R0913, R0914
                                  context, mask, dst_data, tmp_dst_data,
                                  src_data, tmp_work_tensor,
                                  repeat_times, src_rep_stride,
                                  src_data_size, src_offset):
        _fp162fp32_high_preci_func(
            self.source_info, context, self._call_vrec_high_preci, mask,
            tmp_work_tensor[0:src_data_size], src_data,
            tmp_work_tensor[src_data_size:], repeat_times,
            src_rep_stride, src_rep_stride, 4, self.name)

        self._ln_compute_block_gt_5_3(
            context, mask, tmp_work_tensor[src_data_size*3:src_data_size*4],
            tmp_dst_data,
            tmp_work_tensor[0:src_data_size],
            tmp_work_tensor[src_data_size:src_data_size*3], repeat_times,
            src_rep_stride, src_data_size, src_offset)
        vmuls_stmt_ = VectorScalarTemplate(
            self.source_info, "vmuls", mask,
            tmp_work_tensor[src_data_size:src_data_size*2],
            tmp_work_tensor[src_data_size*3:src_data_size*4],
            CONST_NEG_ONE, repeat_times, self.default_blk_stride,
            self.default_blk_stride, src_rep_stride, src_rep_stride,
            self.stride_unit, 0, print_name=self.name)
        vmuls_stmt_.eval_(context)

        vector_dup_stmt = VectorScalarEltwise(
            self.source_info, "vector_dup", mask,
            tmp_work_tensor, CONST_HALF,
            repeat_times, self.default_blk_stride,
            src_rep_stride, stride_unit=0, mask_mode="normal")
        vector_dup_stmt.eval_(context)

        dst_offset = context.evaluate_expr(self.dst_rep_stride) * \
                     self.blk_data_nums
        for index in range(repeat_times):
            vcmd_stmt = VCMP(
                self.source_info, "vmcp_le", mask,
                src_data[index * src_offset],
                tmp_work_tensor[index * src_offset],
                self.default_blk_stride, self.default_blk_stride)
            vcmd_stmt.eval_(context)
            vsel_stmt = VSEL(
                self.source_info, mask, 0,
                dst_data[index*dst_offset], vcmd_stmt,
                tmp_work_tensor[src_data_size + index * src_offset],
                tmp_dst_data[index*src_offset], 1,
                self.default_blk_stride,
                self.default_blk_stride, self.default_blk_stride,
                src_rep_stride, src_rep_stride, src_rep_stride,
                name=self.name)
            vsel_stmt.eval_(context)

    def eval_(self, context):
        """instr eval"""
        repeat_times = context.evaluate_expr(self.repeat_times)
        src_rep_stride = context.evaluate_expr(self.src_rep_stride)
        dst_rep_stride = context.evaluate_expr(self.dst_rep_stride)
        set_vector_mask(self.mask, context,
                        tensor_bit_len=max(get_bit_len(self.src.dtype),
                                           get_bit_len(self.dst.dtype)))
        mask = _eval_mask(self.mask, context)

        # check params
        check_integer_in_range(
            repeat_times, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, %d], "
            "but input repeat_times: %d" %
            (MAX_REPEAT_TIMES - 1, repeat_times))
        check_integer_in_range(
            src_rep_stride, range(MAX_REP_STRIDE_SINGLE_BYTE),
            "src_rep_stride should be in the range of [%d, %d], "
            "but input src_rep_stride: %d" % (0,
                                              MAX_REP_STRIDE_SINGLE_BYTE - 1,
                                              src_rep_stride))
        check_integer_in_range(
            dst_rep_stride, range(MAX_REP_STRIDE_SINGLE_BYTE),
            "dst_rep_stride should be in the range of [%d, %d], "
            "but input dst_rep_stride: %d" % (0,
                                              MAX_REP_STRIDE_SINGLE_BYTE - 1,
                                              dst_rep_stride))

        src_offset = src_rep_stride * self.blk_data_nums
        src_data_size = ceil_div(vector_max_offset_cal(
            mask, self.src.dtype,
            ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype),
            repeat_times, 1,
            src_rep_stride), self.blk_data_nums) * self.blk_data_nums

        multi_factor = 10
        check_over_high_preci(mask, self.dst, self.src, self.work_tensor,
                              repeat_times, dst_rep_stride, src_rep_stride,
                              context.evaluate_expr(self.dst.offset),
                              context.evaluate_expr(self.src.offset),
                              context.evaluate_expr(self.work_tensor.offset),
                              multi_factor, name=self.name)

        self._ln_compute_block_lt_5_3_gt_1(
            context, mask, self.work_tensor[:src_data_size], self.src,
            self.work_tensor[src_data_size:src_data_size*5], repeat_times,
            src_rep_stride, src_data_size, src_offset)
        self._ln_compute_block_gt_5_3(
            context, mask, self.work_tensor[src_data_size*4:src_data_size*5],
            self.work_tensor[:src_data_size], self.src,
            self.work_tensor[src_data_size:src_data_size*3], repeat_times,
            src_rep_stride, src_data_size, src_offset)
        self._ln_compute_block_gt_half_lt_1(
            context, mask, self.work_tensor[0:src_data_size],
            self.work_tensor[src_data_size*4:src_data_size*5], self.src,
            self.work_tensor[src_data_size:src_data_size*4], repeat_times,
            src_rep_stride, src_data_size, src_offset)
        self._ln_compute_block_lt_half(
            context, mask, self.dst,
            self.work_tensor[:src_data_size], self.src,
            self.work_tensor[src_data_size:], repeat_times,
            src_rep_stride, src_data_size, src_offset)


class VrecHighPreci(STMT):  # pylint: disable=R0902
    """this stmt for vrec high precision"""
    def __init__(self, source_info, mask,  # pylint: disable=R0913
                 dst, src, work_tensor, repeat_times,
                 dst_rep_stride, src_rep_stride):
        super(VrecHighPreci, self).__init__(source_info)
        self.name = "vec_rec_high_preci"
        self.mask = mask
        self.dst = dst
        self.src = src
        self.work_tensor = work_tensor
        self.repeat_times = repeat_times
        self.dst_rep_stride = dst_rep_stride
        self.src_rep_stride = src_rep_stride

    def _vrec_high_preci(self, context,  # pylint: disable=R0913, R0914
                         mask, dst, src, work_tensor,
                         repeat_times, dst_rep_stride, src_rep_stride,
                         tensor_split_size):
        """high precision for vrec"""
        stride_unit = 0

        work_tensor0 = work_tensor[0:tensor_split_size]
        work_tensor1 = work_tensor[tensor_split_size:]

        vrec_stmt = VectorOnlyTemplate(self.source_info, "vrec", mask,
                                       dst, src, repeat_times,
                                       _DEFAULT_STRIDE, _DEFAULT_STRIDE,
                                       dst_rep_stride, src_rep_stride,
                                       stride_unit, print_name=self.name)
        vrec_stmt.eval_(context)
        # iteration 1
        vmul_stmt0_ = VectorVectorTemplate(self.source_info, "vmul", mask,
                                           work_tensor0, src, dst,
                                           repeat_times, _DEFAULT_STRIDE,
                                           _DEFAULT_STRIDE, _DEFAULT_STRIDE,
                                           src_rep_stride, src_rep_stride,
                                           dst_rep_stride, stride_unit,
                                           print_name=self.name)
        vmul_stmt0_.eval_(context)

        vmuls_stmt_ = VectorScalarTemplate(self.source_info, "vmuls", mask,
                                           work_tensor1, work_tensor0, -1,
                                           repeat_times, _DEFAULT_STRIDE,
                                           _DEFAULT_STRIDE, src_rep_stride,
                                           src_rep_stride, stride_unit, 0,
                                           print_name=self.name)
        vmuls_stmt_.eval_(context)

        vadds_stmt_ = VectorScalarTemplate(self.source_info, "vadds", mask,
                                           work_tensor0, work_tensor1, 2,
                                           repeat_times, _DEFAULT_STRIDE,
                                           _DEFAULT_STRIDE, src_rep_stride,
                                           src_rep_stride, stride_unit, 0,
                                           print_name=self.name)
        vadds_stmt_.eval_(context)

        vmul_stmt1_ = VectorVectorTemplate(self.source_info, "vmul", mask,
                                           work_tensor1, work_tensor0, dst,
                                           repeat_times, _DEFAULT_STRIDE,
                                           _DEFAULT_STRIDE, _DEFAULT_STRIDE,
                                           src_rep_stride, src_rep_stride,
                                           dst_rep_stride, stride_unit,
                                           print_name=self.name)
        vmul_stmt1_.eval_(context)

        # iteration 2
        vmul_stmt0_ = VectorVectorTemplate(self.source_info, "vmul", mask,
                                           work_tensor0, src, work_tensor1,
                                           repeat_times, _DEFAULT_STRIDE,
                                           _DEFAULT_STRIDE, _DEFAULT_STRIDE,
                                           src_rep_stride, src_rep_stride,
                                           src_rep_stride, stride_unit,
                                           print_name=self.name)
        vmul_stmt0_.eval_(context)

        vmuls_stmt_ = VectorScalarTemplate(self.source_info, "vmuls", mask,
                                           dst, work_tensor0, -1,
                                           repeat_times, _DEFAULT_STRIDE,
                                           _DEFAULT_STRIDE, dst_rep_stride,
                                           src_rep_stride, stride_unit, 0,
                                           print_name=self.name)
        vmuls_stmt_.eval_(context)

        vadds_stmt_ = VectorScalarTemplate(self.source_info, "vadds", mask,
                                           work_tensor0, dst, 2,
                                           repeat_times, _DEFAULT_STRIDE,
                                           _DEFAULT_STRIDE, src_rep_stride,
                                           dst_rep_stride, stride_unit, 0,
                                           print_name=self.name)
        vadds_stmt_.eval_(context)

        vmul_stmt1_ = VectorVectorTemplate(self.source_info, "vmul", mask,
                                           dst, work_tensor0,
                                           work_tensor1, repeat_times,
                                           _DEFAULT_STRIDE, _DEFAULT_STRIDE,
                                           _DEFAULT_STRIDE, dst_rep_stride,
                                           src_rep_stride, src_rep_stride,
                                           stride_unit,
                                           print_name=self.name)
        vmul_stmt1_.eval_(context)

    def eval_(self, context):  # pylint: disable=R0913, R0914
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        repeat_times = context.evaluate_expr(self.repeat_times)
        src_rep_stride = context.evaluate_expr(self.src_rep_stride)
        dst_rep_stride = context.evaluate_expr(self.dst_rep_stride)
        dst_offset = context.evaluate_expr(self.dst.offset)
        src_offset = context.evaluate_expr(self.src.offset)
        work_tensor_offset = context.evaluate_expr(self.work_tensor.offset)
        set_vector_mask(self.mask, context,
                        tensor_bit_len=max(get_bit_len(self.src.dtype),
                                           get_bit_len(self.dst.dtype)))
        mask = _eval_mask(self.mask, context)

        multi_factor = 2
        if self.src.dtype == "float16":
            # 4B of fp32, need keep 32B algin
            multi_factor += 2
        # step for check params, overflow, overlap
        check_integer_in_range(
            repeat_times, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times is {}".format(repeat_times))
        check_vector_stride(None,
                            [dst_rep_stride, src_rep_stride],
                            None, MAX_REP_STRIDE_SINGLE_BYTE, ["dst", "src"])
        check_over_high_preci(mask, self.dst, self.src, self.work_tensor,
                              repeat_times, dst_rep_stride, src_rep_stride,
                              dst_offset, src_offset, work_tensor_offset,
                              multi_factor, name="vec_rec_high_preci")

        if self.src.dtype == "float16":
            _fp162fp32_high_preci_func(self.source_info, context,
                                       self._vrec_high_preci, mask,
                                       self.dst, self.src,
                                       self.work_tensor,
                                       repeat_times,
                                       dst_rep_stride,
                                       src_rep_stride,
                                       multi_factor, name=self.name)
            return

        block_len = ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype)
        src_extend = vector_max_offset_cal(mask, self.src.dtype, block_len,
                                           repeat_times, 1, src_rep_stride)
        tensor_split_size = ceil_div(src_extend, block_len) * block_len
        if self.src.dtype == "float32":
            self._vrec_high_preci(context, mask, self.dst, self.src,
                                  self.work_tensor, self.repeat_times,
                                  self.dst_rep_stride, self.src_rep_stride,
                                  tensor_split_size)


class Vexpm1HighPreci(STMT):  # pylint: disable=R0902
    """this stmt for vexpm1 high precision"""
    def __init__(self, source_info,  # pylint: disable=R0913
                 mask, dst, src, work_tensor, repeat_times,
                 dst_rep_stride, src_rep_stride):
        super(Vexpm1HighPreci, self).__init__(source_info)
        self.name = "vec_expm1_high_preci"
        self.mask = mask
        self.dst = dst
        self.src = src
        self.work_tensor = work_tensor
        self.repeat_times = repeat_times
        self.dst_rep_stride = dst_rep_stride
        self.src_rep_stride = src_rep_stride
        self.extent = 0

    def _gen_comparator(self, context):
        cmp_repeat_times = self.repeat_times
        and_rep_stride = 8
        if context.evaluate_expr(self.src_rep_stride) == 0:
            cmp_repeat_times = 1
            and_rep_stride = 0
        extent = self.extent
        v_dup_stmt = VectorScalarEltwise(
            self.source_info, "vector_dup", self.mask,
            self.work_tensor[:extent], 1.7, self.repeat_times,
            1, self.src_rep_stride, 0, "normal")
        v_dup_stmt.eval_(context)

        lt_tensor = self.work_tensor[extent:
                                     2*extent].reinterpret_cast_to("uint16")
        v_lt_stmt = VCMPV(
            self.source_info, "vcmpv_lt", lt_tensor, self.src,
            self.work_tensor[:extent], cmp_repeat_times, 1, 1,
            self.src_rep_stride, self.src_rep_stride)
        v_lt_stmt.eval_(context)

        v_dup_stmt = VectorScalarEltwise(
            self.source_info, "vector_dup", self.mask,
            self.work_tensor[:extent], -0.7, self.repeat_times, 1,
            self.src_rep_stride, 0, "normal")
        v_dup_stmt.eval_(context)
        gt_tensor = self.work_tensor[2*extent:3*extent].reinterpret_cast_to(
            "uint16")

        v_gt_stmt = VCMPV(
            self.source_info, "vcmpv_gt", gt_tensor, self.src,
            self.work_tensor[:extent], cmp_repeat_times, 1, 1,
            self.src_rep_stride, self.src_rep_stride)
        v_gt_stmt.eval_(context)

        and_tensor = self.work_tensor[:extent].reinterpret_cast_to("uint16")
        v_and_stmt = VectorVectorTemplate(
            self.source_info, "vand", 128, and_tensor, lt_tensor,
            gt_tensor, self.repeat_times, 1, 1, 1, and_rep_stride,
            and_rep_stride, and_rep_stride, 0)
        v_and_stmt.eval_(context)
        return and_tensor

    def _expm1(self, context,  # pylint: disable=R0913, R0914
               dst, work_tensor, src, dst_rep_stride,
               wk_rep_stride, src_rep_stride):
        v_exp_stmt = VectorOnlyTemplate(
            self.source_info, "vexp", self.mask, work_tensor,
            src, self.repeat_times, 1, 1, wk_rep_stride, src_rep_stride, 0)
        v_exp_stmt.eval_(context)
        v_adds_stmt = VectorScalarTemplate(
            self.source_info, "vadds", self.mask, dst, work_tensor, -1,
            self.repeat_times, 1, 1, dst_rep_stride, src_rep_stride, 0, 0)
        v_adds_stmt.eval_(context)

    def _expm1_taylor(self, context,  # pylint: disable=R0913, R0914
                      mask, dst, src, work_tensor, repeat_times,
                      dst_rep_stride, src_rep_stride, tmp_tensor_size):
        wk_tensor1 = work_tensor[:tmp_tensor_size]
        wk_tensor2 = work_tensor[tmp_tensor_size:2*tmp_tensor_size]

        v_adds_stmt = VectorScalarTemplate(
            self.source_info, "vadds", mask, dst, src, 0,
            repeat_times, 1, 1, dst_rep_stride, src_rep_stride, 0, 0)
        v_adds_stmt.eval_(context)

        taylor_param = 1 / 2
        v_adds_stmt = VectorScalarTemplate(
            self.source_info, "vadds", mask, wk_tensor1, src, 0,
            repeat_times, 1, 1, src_rep_stride, src_rep_stride, 0, 0)
        v_adds_stmt.eval_(context)
        v_mul_stmt = VectorVectorTemplate(
            self.source_info, "vmul", mask, wk_tensor2, wk_tensor1, src,
            repeat_times, 1, 1, 1, src_rep_stride, src_rep_stride,
            src_rep_stride, 0)
        v_mul_stmt.eval_(context)
        v_muls_stmt = VectorScalarTemplate(
            self.source_info, "vmuls", mask, wk_tensor1, wk_tensor2,
            taylor_param, repeat_times, 1, 1, src_rep_stride,
            src_rep_stride, 0, 0)
        v_muls_stmt.eval_(context)
        v_add_stmt = VectorVectorTemplate(
            self.source_info, "vadd", mask, wk_tensor2, wk_tensor1, dst,
            repeat_times, 1, 1, 1, src_rep_stride, src_rep_stride,
            dst_rep_stride, 0)
        v_add_stmt.eval_(context)

        for index in range(3, 8):
            taylor_param = 1 / index
            if index % 2 == 1:
                v_mul_stmt = VectorVectorTemplate(
                    self.source_info, "vmul", mask, dst, wk_tensor1, src,
                    repeat_times, 1, 1, 1, dst_rep_stride, src_rep_stride,
                    src_rep_stride, 0)
                v_mul_stmt.eval_(context)
                v_muls_stmt = VectorScalarTemplate(
                    self.source_info, "vmuls", mask, wk_tensor1, dst,
                    taylor_param, repeat_times, 1, 1, src_rep_stride,
                    dst_rep_stride, 0, 0)
                v_muls_stmt.eval_(context)
                v_add_stmt = VectorVectorTemplate(
                    self.source_info, "vadd", mask,
                    dst, wk_tensor2, wk_tensor1,
                    repeat_times, 1, 1, 1, dst_rep_stride, src_rep_stride,
                    src_rep_stride, 0)
                v_add_stmt.eval_(context)
            else:
                v_mul_stmt = VectorVectorTemplate(
                    self.source_info, "vmul",
                    mask, wk_tensor2, wk_tensor1, src,
                    repeat_times, 1, 1, 1, src_rep_stride, src_rep_stride,
                    src_rep_stride, 0)
                v_mul_stmt.eval_(context)
                v_muls_stmt = VectorScalarTemplate(
                    self.source_info, "vmuls", mask, wk_tensor1, wk_tensor2,
                    taylor_param, repeat_times, 1, 1, src_rep_stride,
                    src_rep_stride, 0, 0)
                v_muls_stmt.eval_(context)
                v_add_stmt = VectorVectorTemplate(
                    self.source_info, "vadd",
                    mask, wk_tensor2, dst, wk_tensor1,
                    repeat_times, 1, 1, 1, src_rep_stride, dst_rep_stride,
                    src_rep_stride, 0)
                v_add_stmt.eval_(context)

    def _do_select(self, context, cmp_sel, extent, repeat_times):
        src_offset = self.src_rep_stride*32 // DTYPE_SIZE[self.src.dtype]
        dst_offset = self.dst_rep_stride*32 // DTYPE_SIZE[self.dst.dtype]
        if context.evaluate_expr(self.src_rep_stride) == 0:
            sel_offset = 0
        else:
            sel_offset = 8
        for index in range(repeat_times):
            cmpmask_stmt = MoveTensor2CMPMASK(self.source_info,
                                              cmp_sel[index*sel_offset])
            cmpmask_stmt.eval_(context)
            v_sel_stmt = VSEL(self.source_info, self.mask, 0,
                              self.dst[index*dst_offset:], cmpmask_stmt,
                              self.work_tensor[2*extent + index*src_offset:],
                              self.work_tensor[extent + index*src_offset:], 1,
                              1, 1, 1, self.dst_rep_stride,
                              self.src_rep_stride, self.src_rep_stride)
            v_sel_stmt.eval_(context)

    def eval_(self, context):
        repeat_times = context.evaluate_expr(self.repeat_times)
        src_rep_stride = context.evaluate_expr(self.src_rep_stride)
        dst_rep_stride = context.evaluate_expr(self.dst_rep_stride)
        set_vector_mask(self.mask, context,
                        tensor_bit_len=max(get_bit_len(self.src.dtype),
                                           get_bit_len(self.dst.dtype)))

        mask = _eval_mask(self.mask, context)

        # check params
        check_integer_in_range(
            repeat_times, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, %d], "
            "but input repeat_times: %d" %
            (MAX_REPEAT_TIMES - 1, repeat_times))
        check_integer_in_range(
            src_rep_stride, range(MAX_REP_STRIDE_SINGLE_BYTE),
            "src_rep_stride should be in the range of [%d, %d], "
            "but input src_rep_stride: %d" %
            (0, MAX_REP_STRIDE_SINGLE_BYTE - 1, src_rep_stride))
        check_integer_in_range(
            dst_rep_stride, range(MAX_REP_STRIDE_SINGLE_BYTE),
            "dst_rep_stride should be in the range of [%d, %d], "
            "but input dst_rep_stride: %d" %
            (0, MAX_REP_STRIDE_SINGLE_BYTE - 1, dst_rep_stride))
        block_len = ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype)
        default_mask = 128
        default_rep_stride = 8 if 0 < src_rep_stride <= 8 else src_rep_stride
        src_extend = vector_max_offset_cal(
            default_mask, self.src.dtype, block_len,
            repeat_times, 1, default_rep_stride)
        extent = ceil_div(src_extend, block_len)*block_len
        self.extent = extent

        cmp_sel = self._gen_comparator(context)

        self._expm1(context, self.work_tensor[extent:2*extent],
                    self.work_tensor[2*extent:3*extent], self.src,
                    src_rep_stride, src_rep_stride, src_rep_stride)
        _fp162fp32_high_preci_func(
            self.source_info, context, self._expm1_taylor, mask,
            self.work_tensor[2*extent:3*extent], self.src,
            self.work_tensor[3*extent:], repeat_times, src_rep_stride,
            src_rep_stride, 4, self.name)

        self._do_select(context, cmp_sel, extent, repeat_times)


class VectorScalarTemplate(STMT):
    """this template both have vector and scalar"""
    # pylint: disable=R0902
    def __init__(self, source_info, name, mask, dst, src, scalar, repeat_times,
                 dst_blk_stride, src_blk_stride, dst_rep_stride, src_rep_stride,
                 stride_unit, round_en, mask_mode="normal", print_name=None):
        # pylint: disable=R0913, R0914
        super(VectorScalarTemplate, self).__init__(source_info)
        self.fn_name = name
        self.mask = mask
        self.dst = dst
        self.src = src
        self.scalar = scalar
        self.repeat_times = repeat_times
        self.strides = (dst_blk_stride, src_blk_stride, dst_rep_stride,
                        src_rep_stride, stride_unit, round_en)
        self.mask_mode = mask_mode
        self.print_name = print_name
        if self.print_name is None:
            self.print_name = name

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        _, _, _, _, _, shr_round_en = self.strides

        align = _vec_template_align(context, self.src.dtype)
        temp_env = TempEnv()
        if self.mask_mode == "counter":
            orig_ctrl_value = _set_mask_counter_mode(context)

        set_vector_mask(self.mask, context, self.mask_mode,
                        tensor_bit_len=max(get_bit_len(self.src.dtype),
                                           get_bit_len(self.dst.dtype)))

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, align, True, access_mode='w')

        param = _ENCODER.new_param()
        param.type = self.gen_param_type(context)
        param.xd = xd_idx
        param.xn = self.get_xn_idx(context, temp_env, align)
        param.xm = self.create_gpr_x_m(context, temp_env)
        param.xt = self.create_gpr_x_t(context, temp_env)
        param.out = shr_round_en

        instr = _VECTOR_SCALAR_FN_ENCODER[self.fn_name](param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        # mask: counter_mode, reset CTRL as orig_ctrl_value
        if self.mask_mode == "counter":
            context.model.write_spr('CTRL', orig_ctrl_value)

        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

    def get_xn_idx(self, context, temp_env, align):
        """check tensor overflow

        Parameters
        ----------
        context : the stack context
        temp_env : the temp environment
        align : the align address

        Returns
        -------
        xn_idx
        """
        _, src_blk_stride, _, src_rep_stride, stride_unit, _ = \
            self.strides
        xn_idx, _, src_buffer_size, _ = copy_tensor_to_model(
            context, temp_env, self.src, align, True, access_mode='r')
        check_read_mem_out_of_bounds(context, src_buffer_size, self.mask,
                                     self.src, self.repeat_times,
                                     src_blk_stride, src_rep_stride,
                                     stride_unit, self.mask_mode)
        return xn_idx

    def gen_param_type(self, context):
        """genarate type encoding param

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        param_type : the type encoding
        """
        dtype = self.src.dtype
        if dtype != self.dst.dtype:
            dtype = 'fmix'

        if get_soc_name() in (ASCEND_310, ASCEND_910):
            param_type = _VEC_DATA_TYPE_ENCODING[dtype]
        else:
            # vadds vmuls vmaxs vmins
            if self.fn_name in _SPECIAL_DTYPE_INSTR:
                param_type = _VEC_DATA_TYPE_ENCODING_V200[dtype]
                param_type |= _SPECIAL_DTYPE_INSTR[self.fn_name] << 2
            elif self.fn_name == 'vaxpy':
                param_type = _VECTOR_DTYPE_FN_ENCODER[dtype]
            elif self.fn_name == 'vshl':
                param_type = _VSHL_DTYPE_CODE[dtype]
            elif self.fn_name == 'vshr':
                param_type = _VSHR_DTYPE_ENCODING_V200[dtype]
        return param_type

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
        scalar = context.evaluate_expr(self.scalar)
        if self.fn_name in ("vshl", "vshr"):
            check_vshl_vshr_scalar(self.src.dtype, scalar)

        xm_idx = temp_env.alloc_register()
        x_m = cvt_float_to_uint(self.src.dtype, scalar)

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
        repeat = context.evaluate_expr(self.repeat_times)
        # check repeat
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times: {}".format(repeat))

        dst_block_stride, src_block_stride, dst_repeat_stride, \
            src_repeat_stride, stride_unit, _ = self.strides
        # check strides
        check_vector_stride([context.evaluate_expr(dst_block_stride),
                             context.evaluate_expr(src_block_stride)],
                            [context.evaluate_expr(dst_repeat_stride),
                             context.evaluate_expr(src_repeat_stride)],
                            MAX_BLK_STRIDE_DOUBLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE, ["dst", "src"])
        check_integer_in_range(
            context.evaluate_expr(stride_unit), range(MAX_STRIDE_UNIT),
            "stride_unit should be in the range of [0, 3], "
            "input stride_unit: {}".format(context.evaluate_expr(stride_unit)))

        # check address overlap
        mask_value = _eval_mask(self.mask, context)
        if self.src.buffer == self.dst.buffer:
            check_address_overlapping(
                self.print_name, mask_value, self.dst, self.src,
                BLK_NUM_PER_REP,
                ONE_REP_BYTE_SIZE //
                max(get_bit_len(self.dst.dtype), get_bit_len(self.src.dtype)),
                ONE_REP_BYTE_SIZE // get_bit_len(self.dst.dtype),
                ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype),
                repeat, context.evaluate_expr(dst_block_stride),
                context.evaluate_expr(src_block_stride),
                context.evaluate_expr(dst_repeat_stride),
                context.evaluate_expr(src_repeat_stride),
                context.evaluate_expr(self.dst.offset),
                context.evaluate_expr(self.src.offset),
                context.evaluate_expr(stride_unit), mask_mode=self.mask_mode)

        xt_idx = temp_env.alloc_register()
        x_t = context.evaluate_expr(dst_block_stride)
        x_t |= context.evaluate_expr(src_block_stride) << \
               SRC_BLOCK_STRIDE_SHIFT_POS
        x_t |= context.evaluate_expr(stride_unit) << STRIDE_UNIT_SHIFT_POS
        x_t |= repeat << REPEAT_SHIFT_POS
        x_t |= context.evaluate_expr(dst_repeat_stride) << \
               DST_REPEAT_STRIDE_SHIFT_POS
        x_t |= context.evaluate_expr(src_repeat_stride) << \
               SRC_REPEAT_STRIDE_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class VectorVectorTemplate(STMT):
    """this template both have vector and vecotr"""
    # pylint: disable=R0902
    def __init__(self, source_info, name, mask, dst, src, src1, repeat_times,
                 dst_blk_stride, src0_blk_stride, src1_blk_stride,
                 dst_rep_stride, src0_rep_stride, src1_rep_stride,
                 stride_unit, print_name=None, **kwargs):
        # pylint: disable=R0913, R0914
        super(VectorVectorTemplate, self).__init__(source_info)
        self.fn_name = name
        self.mask = mask
        self.dst = dst
        self.src = src
        self.src1 = src1
        self.repeat_times = repeat_times
        self.strides = (dst_blk_stride, src0_blk_stride, src1_blk_stride,
                        dst_rep_stride, src0_rep_stride, src1_rep_stride,
                        stride_unit)
        self.print_name = print_name
        if self.print_name is None:
            self.print_name = name
        self.kwargs = kwargs

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        # pylint: disable=R0914
        if self.fn_name not in _VECTOR_VECTOR_FN_ENCODER:
            sys.stderr.write(
                "[WARN]: Instruction '%s' not supported in debug system yet!\n"
                % self.fn_name)
            return

        mask = self.mask

        align = _vec_template_align(context, self.src.dtype)
        temp_env = TempEnv()
        set_vector_mask(mask, context,
                        tensor_bit_len=max(get_bit_len(self.src.dtype),
                                           get_bit_len(self.dst.dtype)))

        xn_idx, _, src0_buffer_size, _ = copy_tensor_to_model(
            context, temp_env, self.src, align, True, access_mode='r')

        xm_idx, _, src1_buffer_size, _ = copy_tensor_to_model(
            context, temp_env, self.src1, align, True, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, align, True, access_mode='w')

        param = _ENCODER.new_param()
        param.type = self.gen_param_type()
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = xm_idx
        param.xt = self.create_gpr_x_t(context, temp_env)

        # check param before check overflow
        self.check_read_overflow(context, src0_buffer_size, src1_buffer_size)

        if self.fn_name == 'vadddeqrelu':
            deqscale = context.evaluate_expr(self.kwargs['deqscale'])
            bin_deqscale = cvt_float_to_uint('float16', deqscale)
            context.model.write_spr('DEQSCALE', bin_deqscale)

        if self.fn_name in ('vaddreluconv', 'vsubreluconv'):
            if is_tensor(self.kwargs.get('deqscale')):
                deqscale_tensor = self.kwargs['deqscale']
                deq_addr = copy_tensor_to_model_get_addr(context, temp_env,
                                                         deqscale_tensor,
                                                         align,
                                                         access_mode='w')
                deq_addr = deq_addr // align
                context.model.write_spr('DEQSCALE', deq_addr)
                param.type = 0b11
            else:
                param.type = _VEC_RELUCONV_TYPE_ENCODING[
                    (self.src.dtype, self.dst.dtype)]

            store_h = self.kwargs['storeMode']
            if store_h is not None:
                param.h = context.evaluate_expr(store_h)

        instr = _VECTOR_VECTOR_FN_ENCODER[self.fn_name](param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)

    def gen_param_type(self):
        """genarate type encoding param

        Returns
        -------
        param_type : the type encoding
        """
        # for vmla instr fmix mode
        if self.src.dtype != self.dst.dtype and self.fn_name == 'vmla':
            data_type = 'fmix'
        else:
            data_type = self.src.dtype

        param_type = _VECTOR_DTYPE_FN_ENCODER[data_type]
        if self.fn_name in ('vand', 'vor'):
            bit_width = int(get_dtype_bit_width(data_type))
            param_type = _B16_B32_DTYPE_CODE[bit_width]
            param_type |= _SPECIAL_DTYPE_INSTR[self.fn_name] << 2
        elif self.fn_name in _SPECIAL_DTYPE_INSTR:
            param_type = _VEC_DATA_TYPE_ENCODING_0_[data_type]
            param_type |= _SPECIAL_DTYPE_INSTR[self.fn_name] << 2
        elif self.fn_name == 'vmulconv':
            param_type = _VMULCONV_DTYPE_ENCODING[self.dst.dtype]
        return param_type

    def check_read_overflow(self, context, src0_buffer_size, src1_buffer_size):
        """check tensor overflow

        Parameters
        ----------
        context : the stack context
        src0_buffer_size : src0 operation buffer size
        src1_buffer_size : src1 operation buffer size

        Returns
        -------
        None
        """
        _, src0_blk_stride, src1_blk_stride, _, src0_rep_stride,\
            src1_rep_stride, _ = self.strides
        check_read_mem_out_of_bounds(context, src0_buffer_size, self.mask,
                                     self.src, self.repeat_times,
                                     src0_blk_stride, src0_rep_stride)

        check_read_mem_out_of_bounds(context, src1_buffer_size, self.mask,
                                     self.src1, self.repeat_times,
                                     src1_blk_stride, src1_rep_stride)

    def check_address_overlap(self, context, repeat,  # pylint: disable=R0913
                              dst_block_stride,
                              src_block_stride, src1_block_stride,
                              dst_repeat_stride, src_repeat_stride,
                              src1_repeat_stride, stride_unit):
        """check same buffer address overlapping"""
        mask_value = _eval_mask(self.mask, context)
        if self.src.buffer == self.dst.buffer:
            check_address_overlapping(
                self.print_name, mask_value, self.dst, self.src,
                BLK_NUM_PER_REP,
                ONE_REP_BYTE_SIZE //
                max(get_bit_len(self.dst.dtype), get_bit_len(self.src.dtype)),
                ONE_REP_BYTE_SIZE // get_bit_len(self.dst.dtype),
                ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype),
                repeat, context.evaluate_expr(dst_block_stride),
                context.evaluate_expr(src_block_stride),
                context.evaluate_expr(dst_repeat_stride),
                context.evaluate_expr(src_repeat_stride),
                context.evaluate_expr(self.dst.offset),
                context.evaluate_expr(self.src.offset),
                context.evaluate_expr(stride_unit), msg="dst and src0")

        can_ovelap_instr_name = ["vadd", "vsub",
                                 "vmul", "vmax", "vmin", "vor", "vand"]
        # check address overlap
        if self.src1.buffer == self.dst.buffer:
            if self.fn_name not in can_ovelap_instr_name or \
                    self.dst.dtype not in ("float16", "float32", "int32") or \
                    repeat == 1:
                check_address_overlapping(
                    self.print_name, mask_value, self.dst, self.src1,
                    BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE //
                    max(get_bit_len(self.dst.dtype), get_bit_len(self.src1.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.src1.dtype),
                    repeat, context.evaluate_expr(dst_block_stride),
                    context.evaluate_expr(src1_block_stride),
                    context.evaluate_expr(dst_repeat_stride),
                    context.evaluate_expr(src1_repeat_stride),
                    context.evaluate_expr(self.dst.offset),
                    context.evaluate_expr(self.src1.offset),
                    context.evaluate_expr(stride_unit), msg="dst and src0")
            else:
                self.check_dst_src1_overlap_other(
                    context, repeat, dst_block_stride, src_block_stride,
                    src1_block_stride, dst_repeat_stride, src_repeat_stride,
                    src1_repeat_stride, stride_unit)

    def check_dst_src1_overlap_other(self, context,  # pylint: disable=R0913
                                     repeat, dst_block_stride,
                                     src_block_stride, src1_block_stride,
                                     dst_repeat_stride, src_repeat_stride,
                                     src1_repeat_stride, stride_unit):
        """check dst src1 overlap other case"""
        dst_offset = context.evaluate_expr(self.dst.offset)
        src1_offset = context.evaluate_expr(self.src1.offset)
        mask_value = _eval_mask(self.mask, context)
        if dst_offset is not None and src1_offset is not None:
            if dst_offset != src1_offset or \
                    context.evaluate_expr(dst_block_stride) != \
                    context.evaluate_expr(src1_block_stride) or \
                    context.evaluate_expr(dst_repeat_stride) != 0 or \
                    context.evaluate_expr(src1_repeat_stride) != 0:
                check_address_overlapping(
                    self.print_name, mask_value, self.dst, self.src1,
                    BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE //
                    max(get_bit_len(self.dst.dtype),
                        get_bit_len(self.src1.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.src1.dtype),
                    repeat, context.evaluate_expr(dst_block_stride),
                    context.evaluate_expr(src1_block_stride),
                    context.evaluate_expr(dst_repeat_stride),
                    context.evaluate_expr(src1_repeat_stride),
                    dst_offset, src1_offset,
                    context.evaluate_expr(stride_unit),
                    msg="when dst and src1 are not 100% same, dst and src1")

                if self.src.buffer == self.src1.buffer:
                    try:
                        check_address_overlapping(
                            self.print_name, mask_value, self.src, self.src1,
                            BLK_NUM_PER_REP,
                            ONE_REP_BYTE_SIZE //
                            max(get_bit_len(self.src.dtype),
                                get_bit_len(self.src1.dtype)),
                            ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype),
                            ONE_REP_BYTE_SIZE // get_bit_len(self.src1.dtype),
                            repeat, context.evaluate_expr(src_block_stride),
                            context.evaluate_expr(src1_block_stride),
                            context.evaluate_expr(src_repeat_stride),
                            context.evaluate_expr(src1_repeat_stride),
                            context.evaluate_expr(self.src.offset),
                            src1_offset,
                            context.evaluate_expr(stride_unit),
                            msg="src0 and src1")

                        check_address_overlapping(
                            self.print_name, mask_value, self.src1, self.src,
                            BLK_NUM_PER_REP,
                            ONE_REP_BYTE_SIZE //
                            max(get_bit_len(self.src.dtype),
                                get_bit_len(self.src1.dtype)),
                            ONE_REP_BYTE_SIZE // get_bit_len(self.src1.dtype),
                            ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype),
                            repeat, context.evaluate_expr(src1_block_stride),
                            context.evaluate_expr(src_block_stride),
                            context.evaluate_expr(src1_repeat_stride),
                            context.evaluate_expr(src_repeat_stride),
                            src1_offset,
                            context.evaluate_expr(self.src.offset),
                            context.evaluate_expr(stride_unit),
                            msg="src1 and src0")
                    except (RuntimeError, SystemExit):
                        TikCheckUtil.raise_error(
                            "when repeat_times>1, "
                            "{} dst and src1 address overlap is"
                            " not support address overlapping"
                            " between src0 and src1.".format(
                                self.print_name))

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
        repeat = context.evaluate_expr(self.repeat_times)
        # check repeat_times
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times is {}".format(repeat))

        dst_block_stride, src_block_stride, src1_block_stride, \
            dst_repeat_stride, src_repeat_stride, src1_repeat_stride, \
            stride_unit = self.strides
        # check strides
        check_vector_stride([context.evaluate_expr(dst_block_stride),
                             context.evaluate_expr(src_block_stride),
                             context.evaluate_expr(src1_block_stride)],
                            [context.evaluate_expr(dst_repeat_stride),
                             context.evaluate_expr(src_repeat_stride),
                             context.evaluate_expr(src1_repeat_stride)],
                            MAX_BLK_STRIDE_SINGLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE,
                            ["dst", "src0", "src1"])

        # check address overlap
        self.check_address_overlap(context, repeat, dst_block_stride,
                                   src_block_stride, src1_block_stride,
                                   dst_repeat_stride, src_repeat_stride,
                                   src1_repeat_stride, stride_unit)

        xt_idx = temp_env.alloc_register()
        x_t = context.evaluate_expr(dst_block_stride)
        x_t |= context.evaluate_expr(src_block_stride) << \
               _SRC_BLK_STRIDE_SHIFT_POS
        x_t |= context.evaluate_expr(src1_block_stride) << \
               _SRC1_BLK_STRIDE_SHIFT_POS
        x_t |= context.evaluate_expr(stride_unit) << STRIDE_UNIT_SHIFT_POS
        x_t |= repeat << REPEAT_SHIFT_POS
        x_t |= context.evaluate_expr(dst_repeat_stride) << \
               _DST_REPEAT_STRIDE_SHIFT_POS
        x_t |= context.evaluate_expr(src_repeat_stride) << \
               _SRC_REPEAT_STRIDE_SHIFT_POS
        x_t |= context.evaluate_expr(src1_repeat_stride) << \
               _SRC1_REPEAT_STRIDE_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class Vconv(STMT):
    """Vconv instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, mask,  # pylint: disable=R0914
                 name, dst, src, repeat_times,
                 dst_block_stride, src_block_stride, dst_repeat_stride,
                 src_repeat_stride, deq_scale=None, ldst_high_half=False,
                 stride_unit=0, print_name=None):
        # pylint: disable=R0913
        super(Vconv, self).__init__(source_info)
        self.round_method = name
        if name == 'none':
            self.round_method = ''
        if name == 'ceiling':
            self.round_method = 'ceil'
        self.mask = mask
        self.dst = dst
        self.src = src
        self.repeat_times = repeat_times
        self.dst_block_stride = dst_block_stride
        self.src_block_stride = src_block_stride
        self.dst_repeat_stride = dst_repeat_stride
        self.src_repeat_stride = src_repeat_stride
        self.deq_scale = deq_scale
        self.ldst_high_half = ldst_high_half
        self.stride_unit = stride_unit
        self.name = print_name

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

        mask = self.mask
        set_vector_mask(mask, context,
                        tensor_bit_len=max(get_bit_len(self.dst.dtype),
                                           get_bit_len(self.src.dtype)))
        deq_mode = self.set_spr_deqscale(context, temp_env)

        align = _vec_template_align(context, self.src.dtype)
        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, align, True, access_mode='r')
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, align, True, access_mode='w')

        param = context.encoder.new_param()
        param.conv_type = self.get_conv_type(deq_mode)
        # v200
        if self.ldst_high_half is not None:
            param.h = context.evaluate_expr(self.ldst_high_half)
        param.xd = xd_idx
        param.xn = xn_idx
        param.xt = self.create_gpr_x_t(context, temp_env)

        instr = context.encoder.gen_vconvx(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

    def set_spr_deqscale(self, context, temp_env):
        """set special purpose register DEQSCALE

        Parameters
        ----------
        context : the stack context
        temp_env: temp environment

        Returns
        -------
        deq_mode
        """
        deq_mode = ''
        if self.src.dtype == 'int32' and self.dst.dtype == 'float16':
            if self.deq_scale is None:
                TikCheckUtil.raise_error('vconv deq scale is None!')
            deq_scale = context.evaluate_expr(self.deq_scale)
            deq_mode = 'DEQ'
            bin_deq_scale = cvt_float_to_uint(self.dst.dtype, deq_scale)
            context.model.write_spr('DEQSCALE', bin_deq_scale)

        if self.src.dtype == 'int16' and self.dst.dtype in ('int8', 'uint8'):
            if is_tensor(self.deq_scale):
                deq_mode = 'VDEQs162b8'
                deq_addr = copy_tensor_to_model_get_addr(context, temp_env,
                                                         self.deq_scale,
                                                         ALIGNED_ADDR,
                                                         access_mode='r')
                deq_addr = deq_addr // ALIGNED_ADDR
                context.model.write_spr('DEQSCALE', deq_addr)
            elif is_scalar(self.deq_scale):
                deq_mode = 'DEQs162b8'
                self.check_deqscale(context)
                context.model.write_spr('DEQSCALE',
                                        context.evaluate_expr(self.deq_scale))
            else:
                deq_mode = 'DEQs162b8'
                if not isinstance(self.deq_scale, int):
                    TikCheckUtil.raise_error(
                        'invalid type for deqscale'
                        ' {}'.format(type(self.deq_scale)))
                self.check_deqscale(context)
                context.model.write_spr('DEQSCALE', self.deq_scale)

        return deq_mode

    def check_deqscale(self, context):
        """make deqscale[46] consistent with dst.dtype, for deqs162b8 mode

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        deq_46 = (context.evaluate_expr(self.deq_scale) &
                  DEQSCALE_46BIT_MASK) >> DEQSCALE_46BIT_SHIFT_POS
        if self.dst.dtype == "int8":
            # check deqscale[46]=1 which indicate the result is signed
            TikCheckUtil.check_equality(
                deq_46, 1,
                "deqscale[46] bit should be 1 when converting int16 to int8")
        elif self.dst.dtype == "uint8":
            # check deqscale[46]=0 which indicate the result is unsigned
            TikCheckUtil.check_equality(
                deq_46, 0,
                "deqscale[46] bit should be 0 when converting int16 to uint8")

    def get_conv_type(self, deq_mode):
        """get conv_type

        Parameters
        ----------
        deq_mode

        Returns
        -------
        conv_type
        """
        conv_type = 0
        if deq_mode != '':
            conv_type = _VCONV_TYPE_ENCODING[(deq_mode,)]
        elif self.round_method not in ('', None):
            conv_type = _VCONV_TYPE_ENCODING[self.src.dtype, self.dst.
                                             dtype, self.round_method]
        else:
            conv_type = _VCONV_TYPE_ENCODING[self.src.dtype, self.dst.dtype]

        return conv_type

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
        repeat = context.evaluate_expr(self.repeat_times)
        dst_block_stride = context.evaluate_expr(self.dst_block_stride)
        src_block_stride = context.evaluate_expr(self.src_block_stride)
        dst_repeat_stride = context.evaluate_expr(self.dst_repeat_stride)
        src_repeat_stride = context.evaluate_expr(self.src_repeat_stride)
        stride_unit = context.evaluate_expr(self.stride_unit)

        # check repeat_times
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times: {}".format(repeat))
        # check strides
        check_vector_stride([dst_block_stride, src_block_stride],
                            [dst_repeat_stride, src_repeat_stride],
                            MAX_BLK_STRIDE_DOUBLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE, ["dst", "src"])

        mask_value = _eval_mask(self.mask, context)
        if self.src.dtype == 'int16' and self.dst.dtype in ('int8', 'uint8'):
            # check address overlap
            check_vconv_deqs162b8_overlap(
                self.src, self.dst, self.deq_scale, mask_value, repeat,
                self.ldst_high_half, [dst_block_stride, src_block_stride],
                [dst_repeat_stride, src_repeat_stride], stride_unit)

            # check tensor overflow
            check_vconv_deqs162b8_overflow(
                self.src, self.dst, self.deq_scale, mask_value, repeat,
                self.ldst_high_half, [dst_block_stride, src_block_stride],
                [dst_repeat_stride, src_repeat_stride], stride_unit)
        else:
            # check address overlap
            if self.src.buffer == self.dst.buffer:
                check_address_overlapping(
                    self.name, mask_value, self.dst, self.src,
                    BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE //
                    max(get_bit_len(self.dst.dtype),
                        get_bit_len(self.src.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype),
                    repeat, dst_block_stride, src_block_stride,
                    dst_repeat_stride, src_repeat_stride,
                    context.evaluate_expr(self.dst.offset),
                    context.evaluate_expr(self.src.offset), stride_unit)

            check_tensor_overflow(
                (self.dst, self.src), mask_value, repeat,
                (dst_block_stride, src_block_stride),
                (dst_repeat_stride, src_repeat_stride), ("dst", "src"),
                stride_unit=stride_unit)

        xt_idx = temp_env.alloc_register()
        x_t = dst_block_stride
        x_t |= src_block_stride << SRC_BLOCK_STRIDE_SHIFT_POS
        x_t |= stride_unit << STRIDE_UNIT_SHIFT_POS
        x_t |= repeat << REPEAT_SHIFT_POS
        x_t |= dst_repeat_stride << DST_REPEAT_STRIDE_SHIFT_POS
        x_t |= src_repeat_stride << SRC_REPEAT_STRIDE_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class VCMPV(STMT):
    """VCMPV instruction"""
    def __init__(self, source_info, name, dst, src0, src1, repeat_times,
                 src0_blk_stride, src1_blk_stride, src0_rep_stride,
                 src1_rep_stride, print_name=None):
        # pylint: disable=R0913
        super(VCMPV, self).__init__(source_info)
        self.name = name
        self.dst = dst
        self.src = src0
        self.src1 = src1
        self.repeat_times = repeat_times
        self.strides = (src0_blk_stride, src1_blk_stride, src0_rep_stride,
                        src1_rep_stride)
        self.print_name = print_name
        if self.print_name is None:
            self.print_name = name

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

        align = _vec_template_align(context, self.src.dtype)

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, align, True, access_mode='w')

        param = context.encoder.new_param()
        param.cond_op = _VEC_CMP_OP_ENCODER[self.name.split('_')[1]]
        param.type = _VCMPV_TYPE_ENCODER[self.src.dtype]
        param.xd = xd_idx
        param.xt = self.create_gpr_x_t(context, temp_env)
        param.xn, param.xm = self.get_src_idx(context, temp_env, align)

        instr = context.encoder.gen_vcmpvx(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

    def get_src_idx(self, context, temp_env, align):
        """get src xn_idx and xm_idx and check tensor overflow

        Parameters
        ----------
        context : the stack context
        temp_env : the temp environment
        align : the align address

        Returns
        -------
        xn_idx, xm_idx
        """
        xn_idx, _, src_buffer_size, _ = copy_tensor_to_model(
            context, temp_env, self.src, align, True, access_mode='r')

        xm_idx, _, src1_buffer_size, _ = copy_tensor_to_model(
            context, temp_env, self.src1, align, True, access_mode='r')

        src_block_stride, src1_block_stride, \
            src_repeat_stride, src1_repeat_stride = self.strides

        # for vcmpv instr, mask is related to type
        default_mask = ONE_REP_BYTE_SIZE // DTYPE_SIZE[self.src.dtype]
        check_read_mem_out_of_bounds(context, src_buffer_size, default_mask,
                                     self.src, self.repeat_times,
                                     src_block_stride, src_repeat_stride)

        check_read_mem_out_of_bounds(context, src1_buffer_size, default_mask,
                                     self.src1, self.repeat_times,
                                     src1_block_stride, src1_repeat_stride)
        return xn_idx, xm_idx

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
        src_block_stride, src1_block_stride,\
            src_repeat_stride, src1_repeat_stride = self.strides
        repeat = context.evaluate_expr(self.repeat_times)
        # check repeat
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input value is %s" % str(repeat))
        # check stride
        check_vector_stride([context.evaluate_expr(src_block_stride),
                             context.evaluate_expr(src1_block_stride)],
                            [context.evaluate_expr(src_repeat_stride),
                             context.evaluate_expr(src1_repeat_stride)],
                            MAX_BLK_STRIDE_SINGLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE, ["src0", "src1"])

        # check address overlap
        default_blk_stride = 1
        default_rep_stride = 8
        mask = ONE_REP_BYTE_SIZE // \
               max(DTYPE_SIZE[self.dst.dtype],
                   DTYPE_SIZE[self.src.dtype], DTYPE_SIZE[self.src1.dtype])
        if self.src.buffer == self.dst.buffer:
            if self.src.buffer == self.dst.buffer:
                check_address_overlapping(
                    self.print_name, mask, self.dst, self.src,
                    BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE //
                    max(get_bit_len(self.dst.dtype),
                        get_bit_len(self.src.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype),
                    repeat, default_blk_stride,
                    context.evaluate_expr(src_block_stride),
                    default_rep_stride,
                    context.evaluate_expr(src_repeat_stride),
                    context.evaluate_expr(self.dst.offset),
                    context.evaluate_expr(self.src.offset), msg="dst and src0")

        if self.src1.buffer == self.dst.buffer:
            if self.src.buffer == self.dst.buffer:
                check_address_overlapping(
                    self.print_name, mask, self.dst, self.src1,
                    BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE //
                    max(get_bit_len(self.dst.dtype),
                        get_bit_len(self.src1.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.src1.dtype),
                    repeat, default_blk_stride,
                    context.evaluate_expr(src1_block_stride),
                    default_rep_stride,
                    context.evaluate_expr(src1_repeat_stride),
                    context.evaluate_expr(self.dst.offset),
                    context.evaluate_expr(self.src1.offset), msg="dst and src1")

        xt_idx = temp_env.alloc_register()
        x_t = 0
        x_t |= context.evaluate_expr(src_block_stride) << \
               _SRC_BLK_STRIDE_SHIFT_POS
        x_t |= context.evaluate_expr(src1_block_stride) << \
               _SRC1_BLK_STRIDE_SHIFT_POS
        x_t |= repeat << REPEAT_SHIFT_POS
        x_t |= context.evaluate_expr(src_repeat_stride) << \
               _SRC_REPEAT_STRIDE_SHIFT_POS
        x_t |= context.evaluate_expr(src1_repeat_stride) << \
               _SRC1_REPEAT_STRIDE_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class VTranspose(STMT):
    """VTranspose instruction"""
    def __init__(self, source_info, dst, src):
        super(VTranspose, self).__init__(source_info)
        self.dst = dst
        self.src = src

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
        align = _vec_template_align(context, self.src.dtype)
        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, align, True, access_mode='r')
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, align, True, access_mode='w')

        param = context.encoder.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[self.src.dtype]
        param.xd = xd_idx
        param.xn = xn_idx

        # check address overlapping
        src_offset = context.evaluate_expr(self.src.offset)
        dst_offset = context.evaluate_expr(self.dst.offset)
        if all(isinstance(value, int) for value in (src_offset, dst_offset)):
            if self.src.buffer == self.dst.buffer:
                if src_offset == dst_offset or \
                        src_offset + VTRANSPOSE_REQUIRED_ELEMENT <= \
                        dst_offset or dst_offset + \
                        VTRANSPOSE_REQUIRED_ELEMENT <= src_offset:
                    pass
                else:
                    TikCheckUtil.raise_error(
                        "vtranspose not support partially address overlapping")

        instr = context.encoder.gen_vtranspose(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model)

        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)


class VnchwTrans(STMT):
    """VnchwTrans instruction"""
    def __init__(self, source_info, dst, src,  # pylint: disable=R0913
                 repeat_times, dst_repeat_stride, src_repeat_stride):
        super(VnchwTrans, self).__init__(source_info)
        self.dst = dst
        self.src = src
        self.repeat_times = repeat_times
        self.dst_repeat_stride = dst_repeat_stride
        self.src_repeat_stride = src_repeat_stride

    def eval_(self,    # pylint: disable=R0914
              context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        repeat_times = context.evaluate_expr(self.repeat_times)
        dst_rep_stride = context.evaluate_expr(self.dst_repeat_stride)
        src_rep_stride = context.evaluate_expr(self.src_repeat_stride)
        check_vec_trans_params_range(repeat_times, dst_rep_stride,
                                     src_rep_stride)

        # check tensor overflow
        src_offset = context.evaluate_expr(self.src.indice.offset)
        dst_offset = context.evaluate_expr(self.dst.indice.offset)
        check_vec_trans_overflow(self.dst.indice.origin_shape,
                                 self.src.indice.origin_shape,
                                 dst_offset, src_offset, repeat_times,
                                 dst_rep_stride, src_rep_stride)

        for i in range(repeat_times):
            temp_env = TempEnv()
            src_start_addr = src_rep_stride*PER_TRANSPOSE_DATA_SIZE*i
            dst_start_addr = dst_rep_stride*PER_TRANSPOSE_DATA_SIZE*i

            # align is 32B
            align = _vec_template_align(context, self.dst.dtype)
            xn_idx, _, _, _ = copy_tensor_to_model(
                context, temp_env, self.src, align=align, check_align=True,
                access_mode='r', offset=src_start_addr)

            xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
                context, temp_env, self.dst, align=align, check_align=True,
                access_mode='w', offset=dst_start_addr)

            param = context.encoder.new_param()
            param.type = _VEC_DATA_TYPE_ENCODING[self.src.dtype]
            param.xd = xd_idx
            param.xn = xn_idx

            instr = context.encoder.gen_vtranspose(param)
            context.model.step(instr)
            temp_env.check_mem_access(context.model)
            context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                      dst_alloc_size)


class VectorScalarEltwise(STMT):
    """VectorScalarEltwise instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, name, mask, dst, scalar, repeat_times,
                 dst_blk_stride, dst_rep_stride, stride_unit, mask_mode):
        # pylint: disable=R0913
        super(VectorScalarEltwise, self).__init__(source_info)
        self.name = name
        self.mask = mask
        self.dst = dst
        self.scalar = scalar
        self.repeat_times = repeat_times
        self.dst_blk_stride = dst_blk_stride
        self.dst_rep_stride = dst_rep_stride
        self.stride_unit = stride_unit
        self.mask_mode = mask_mode

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        mask = self.mask
        if self.mask_mode == "counter":
            orig_ctrl_value = _set_mask_counter_mode(context)

        set_vector_mask(mask, context, mask_mode=self.mask_mode,
                        tensor_bit_len=get_bit_len(self.dst.dtype))

        temp_env = TempEnv()

        align = _vec_template_align(context, self.dst.dtype)
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, align, True, access_mode='w')

        bit_width = get_dtype_bit_width(self.dst.dtype)
        dtype = 'uint' + bit_width

        param = context.encoder.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[dtype]
        param.xd = xd_idx
        param.xn = self.create_gpr_x_n(context, temp_env)
        param.xt = self.create_gpr_x_t(context, temp_env)

        instr = context.encoder.gen_move_vx(param)
        if self.name == 'vci':
            # init instr pos
            instr = 65
            instr = instr << _INSTR_SHIFT_POS
            instr |= _VCI_DTYPE[self.dst.dtype] << _DST_DTYPE_SHIFT_POS
            instr |= param.xd << _PARAM_XD_SHIFT_POS
            instr |= param.xn << _PARAM_XN_SHIFT_POS
            instr |= param.xt << _PARAM_XT_SHIFT_POS
            instr |= _INSTR_OR_VALUE

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        # mask: counter_mode, reset CTRL as orig_ctrl_value
        if self.mask_mode == "counter":
            context.model.write_spr('CTRL', orig_ctrl_value)

        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

    def create_gpr_x_n(self, context, temp_env):
        """create general purpose register x_n

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xn_idx
        """
        scalar = context.evaluate_expr(self.scalar)

        xn_idx = temp_env.alloc_register()
        x_n = cvt_float_to_uint(self.dst.dtype, scalar)

        context.model.write_gpr(xn_idx, x_n)

        return xn_idx

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
        repeat = context.evaluate_expr(self.repeat_times)
        # check repeat
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255],"
            " input value is %s" % str(repeat))

        dst_block_stride = context.evaluate_expr(self.dst_blk_stride)
        dst_repeat_stride = context.evaluate_expr(self.dst_rep_stride)
        # check strides
        check_vector_stride([dst_block_stride], [dst_repeat_stride],
                            MAX_BLK_STRIDE_DOUBLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE, ["dst"])

        stride_unit = context.evaluate_expr(self.stride_unit)

        xt_idx = temp_env.alloc_register()
        x_t = dst_block_stride
        x_t |= stride_unit << STRIDE_UNIT_SHIFT_POS
        x_t |= repeat << REPEAT_SHIFT_POS
        x_t |= dst_repeat_stride << DST_REPEAT_STRIDE_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class VecReduce(STMT):
    """
    vector all reduce, group reduce, pair reduce has \
    no difference to the debugger
    """
    # pylint: disable=R0902
    def __init__(self, source_info, name, mask, dst, src, repeat_times,
                 dst_rep_stride, src_blk_stride, src_rep_stride, stride_unit,
                 order, maxmin_cnt_index, print_name=None):
        # pylint: disable=R0913
        super(VecReduce, self).__init__(source_info)

        self.name = name
        self.mask = mask
        self.dst = dst
        self.src = src
        self.repeat_time = repeat_times
        self.dst_rep_stride = dst_rep_stride
        self.src_blk_stride = src_blk_stride
        self.src_rep_stride = src_rep_stride
        self.stride_unit = stride_unit
        self.order = order
        self.maxmin_cnt_index = maxmin_cnt_index
        self.print_name = print_name
        if self.print_name is None:
            self.print_name = name

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
        set_vector_mask(self.mask, context,
                        tensor_bit_len=max(get_bit_len(self.dst.dtype),
                                           get_bit_len(self.src.dtype)))

        dst_align = get_dtype_size(self.src.dtype)
        src_align = _vec_template_align(context, self.src.dtype)
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, dst_align, True, access_mode='w')
        xn_idx, _, src_alloc_size, _ = copy_tensor_to_model(
            context, temp_env, self.src, src_align, True, access_mode='r')

        param = _ENCODER.new_param()
        if "add" in self.name:
            param.type = _VEC_DATA_TYPE_ENCODING[self.src.dtype]
        else:
            param.type = _VEC_DATA_TYPE_ENCODING_V200[self.src.dtype]
        param.xt = self.create_gpr_x_t(context, temp_env)
        param.xd = xd_idx
        param.xn = xn_idx

        check_read_mem_out_of_bounds(context, src_alloc_size, self.mask,
                                     self.src, self.repeat_time,
                                     self.src_blk_stride, self.src_rep_stride)
        if self.order is not None:
            param.order = self.order

        instr = _VEC_WHOLE_REDUCE_ENCODER[self.name](param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

        self.update_var_maxmin_cnt(context)

    def update_var_maxmin_cnt(self, context):
        """update the var in var table with value recorded in \
        special purpose register 'MAX_MIN_CNT'

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        if self.maxmin_cnt_index is not None:
            index_num = len(self.maxmin_cnt_index)
            TikCheckUtil.check_in_range(
                index_num, (MAXMIN_CNT_INDEX_LEN_1, MAXMIN_CNT_INDEX_LEN_3),
                "maxmin_cnt_index length must be 1 or 3")

            sr_value = context.model.read_spr('MAX_MIN_CNT')
            # maxmin_cnt_index contains maxmin, cnt and index
            # var0 means maxmin
            var0 = self.maxmin_cnt_index[0].debug_var
            val_bit_width = get_dtype_bit_width(var0.dtype)
            context.update_var(
                var0, reinterpret_type('uint' + val_bit_width,
                                       var0.dtype,
                                       sr_value & FOUR_BYTE_VALUE))
            if index_num == MAXMIN_CNT_INDEX_LEN_3:
                # var1 means cnt
                var1 = self.maxmin_cnt_index[1].debug_var
                # var2 means index
                var2 = self.maxmin_cnt_index[2].debug_var
                context.update_var(var1, (sr_value >> CNT_SHIFT_POS) &
                                   TWO_BYTE_VALUE)
                context.update_var(var2, (sr_value >> INDEX_SHIFT_POS) &
                                   TWO_BYTE_VALUE)

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
        dst_repeat_stride = context.evaluate_expr(self.dst_rep_stride)
        src_blk_stride = context.evaluate_expr(self.src_blk_stride)
        src_repeat_stride = context.evaluate_expr(self.src_rep_stride)
        repeat = context.evaluate_expr(self.repeat_time)
        stride_unit = context.evaluate_expr(self.stride_unit)
        # check params
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input value is %s" % str(repeat))
        # check stride
        TikCheckUtil.check_in_range(
            src_repeat_stride, range(MAX_REP_STRIDE_DOUBLE_BYTE),
            "src_rep_stride should be in the range of [0,65535], "
            "input value is %s" % str(src_repeat_stride))
        TikCheckUtil.check_in_range(
            dst_repeat_stride, range(MAX_REP_STRIDE_DOUBLE_BYTE),
            "dst_rep_stride should be in the range of [0,65535], "
            "input value is %s" % str(dst_repeat_stride))
        check_integer_in_range(
            src_blk_stride, range(MAX_BLK_STRIDE_DOUBLE_BYTE),
            "src_blk_stride should be in the range of [0, 65535], "
            "input value is %s" % str(src_blk_stride))
        check_integer_in_range(stride_unit, range(MAX_STRIDE_UNIT),
                               "stride_unit should be in the range of [0, 3], "
                               "input value is %s" % str(stride_unit))

        # check address overlap
        default_nblock = 1
        default_stride = 1
        mask_value = _eval_mask(self.mask, context)
        if self.name in ("vcmax", "vcmin"):
            block_len = 2
        elif self.name == "vcadd":
            block_len = 1
        elif self.name == "vcpadd":
            block_len = 64
        else:
            block_len = 8
        if self.src.buffer == self.dst.buffer:
            check_address_overlapping(
                self.print_name, mask_value, self.dst, self.src,
                default_nblock,
                ONE_REP_BYTE_SIZE //
                max(get_bit_len(self.dst.dtype),
                    get_bit_len(self.src.dtype)),
                block_len, ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype),
                repeat, default_stride, src_blk_stride,
                dst_repeat_stride, src_repeat_stride,
                context.evaluate_expr(self.dst.offset),
                context.evaluate_expr(self.src.offset), stride_unit)

        xt_idx = temp_env.alloc_register()
        x_t = dst_repeat_stride
        x_t |= src_blk_stride << SRC_BLOCK_STRIDE_SHIFT_POS
        x_t |= stride_unit << STRIDE_UNIT_SHIFT_POS
        x_t |= repeat << REPEAT_SHIFT_POS
        x_t |= src_repeat_stride << _SRC_REPEAT_STRIDE_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class VCMP(STMT):
    """VCMP instruction"""
    def __init__(self, source_info, name, mask, src0, src1, src0_stride,
                 src1_stride):
        # pylint: disable=R0913
        super(VCMP, self).__init__(source_info)
        self.name = name
        self.mask = mask
        self.src0 = src0
        self.src1 = src1
        self.src0_stride = src0_stride
        self.src1_stride = src1_stride

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
        set_vector_mask(self.mask, context,
                        tensor_bit_len=max(get_bit_len(self.src1.dtype),
                                           get_bit_len(self.src0.dtype)))
        align = _vec_template_align(context, self.src0.dtype)

        xn_idx, _, src0_alloc_size, _ = copy_tensor_to_model(
            context, temp_env, self.src0, align, True, access_mode='r')

        # because repeat_time is 1, so we don't need care rep_stride
        default_repeat_time = 1
        default_rep_stride = 0
        check_read_mem_out_of_bounds(context, src0_alloc_size, self.mask,
                                     self.src0, default_repeat_time,
                                     self.src0_stride, default_rep_stride)

        xm_idx, _, src1_alloc_size, _ = copy_tensor_to_model(
            context, temp_env, self.src1, align, True, access_mode='r')

        check_read_mem_out_of_bounds(context, src1_alloc_size, self.mask,
                                     self.src1, default_repeat_time,
                                     self.src1_stride, default_rep_stride)

        param = context.encoder.new_param()
        param.cond_op = _VEC_CMP_OP_ENCODER[self.name.split('_')[1]]
        param.type = _VEC_DATA_TYPE_ENCODING_V200[self.src0.dtype]
        param.xn = xn_idx
        param.xm = xm_idx
        param.xt = self.create_gpr_x_t(context, temp_env)

        instr = context.encoder.gen_vcmpx(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

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
        src0_stride = context.evaluate_expr(self.src0_stride)
        src1_stride = context.evaluate_expr(self.src1_stride)

        # check blk_stride
        check_integer_in_range(
            src0_stride,
            range(MAX_BLK_STRIDE_DOUBLE_BYTE),
            "src0_blk_stride should be in the range of [0, 65535], "
            "input value is %s" % str(src0_stride))
        check_integer_in_range(
            src1_stride,
            range(MAX_BLK_STRIDE_DOUBLE_BYTE),
            "src1_blk_stride should be in the range of [0, 65535], "
            "input value is %s" % str(src1_stride))

        xt_idx = temp_env.alloc_register()
        x_t = 0
        x_t |= src0_stride << _SRC_BLK_STRIDE_SHIFT_POS
        x_t |= src1_stride << _SRC1_BLK_STRIDE_SHIFT_POS
        x_t |= 1 << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)
        return xt_idx


class VSEL(STMT):
    """VSEL instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, mask, mode, dst, sel, src0, src1, repeat_times,
                 dst_block_stride, src_block_stride, src1_block_stride,
                 dst_rep_stride, src0_rep_stride, src1_rep_stride, name="vsel"):
        # pylint: disable=R0913, R0914
        super(VSEL, self).__init__(source_info)
        self.mask = mask
        self.mode = mode
        self.dst = dst
        self.sel = sel
        self.src0 = src0
        self.src1 = src1
        self.repeat_times = repeat_times
        self.dst_blk_stride = dst_block_stride
        self.src0_blk_stride = src_block_stride
        self.src1_blk_stride = src1_block_stride
        self.dst_rep_stride = dst_rep_stride
        self.src0_rep_stride = src0_rep_stride
        self.src1_rep_stride = src1_rep_stride
        self.name = name

    def diff_mode_idx(self, context, temp_env, mode):
        """different mode for xm_idx

        Parameters
        ----------
        context : the stack context
        temp_env: temp environment
        mode: vsel mode

        Returns
        -------
        xm_idx
        """
        mask_align = 32
        align = _vec_template_align(context, self.src0.dtype)
        xm_idx = -1
        if mode == 0:
            xm_idx, _, src1_alloc_size, _ = copy_tensor_to_model(
                context, temp_env, self.src1, align, True, access_mode='r')

            check_read_mem_out_of_bounds(context, src1_alloc_size, self.mask,
                                         self.src1, self.repeat_times,
                                         self.src1_blk_stride,
                                         self.src1_rep_stride)
        elif mode == 1:
            xm_idx, _, _, _ = copy_tensor_to_model(
                context, temp_env, self.sel, mask_align, True, access_mode='r')

            cmp_constant = context.evaluate_expr(self.src1)
            context.model.write_spr('CMPMASK0',
                                    cvt_float_to_uint(self.src0.dtype,
                                                      cmp_constant))
        elif mode == 2:
            xm_idx, _, src1_alloc_size, _ = copy_tensor_to_model(
                context, temp_env, self.src1, align, True, access_mode='r')

            check_read_mem_out_of_bounds(context, src1_alloc_size, self.mask,
                                         self.src1, self.repeat_times,
                                         self.src1_blk_stride,
                                         self.src1_rep_stride)
            cmp_idx, _, _, _ = copy_tensor_to_model(
                context, temp_env, self.sel, mask_align, True, access_mode='r')

            cmp_addr = context.model.read_gpr(cmp_idx)
            context.model.write_spr('CMPMASK0', cmp_addr)
        else:
            TikCheckUtil.raise_error(
                '{} invalid mode {}'.format(self.name, mode))

        return xm_idx

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        set_vector_mask(self.mask, context,
                        tensor_bit_len=max(get_bit_len(self.src0.dtype),
                                           get_bit_len(self.dst.dtype)))

        mode = context.evaluate_expr(self.mode)
        temp_env = TempEnv()

        align = _vec_template_align(context, self.src0.dtype)
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, align, True, access_mode='w')
        xn_idx, _, src0_alloc_size, _ = copy_tensor_to_model(
            context, temp_env, self.src0, align, True, access_mode='r')

        param = context.encoder.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[self.src0.dtype]
        param.xd = xd_idx
        param.xn = xn_idx
        param.xt = self.create_gpr_x_t(context, temp_env)

        check_read_mem_out_of_bounds(context, src0_alloc_size, self.mask,
                                     self.src0, self.repeat_times,
                                     self.src0_blk_stride, self.src0_rep_stride)

        param.xm = self.diff_mode_idx(context, temp_env, mode)

        instr = context.encoder.gen_vselx(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

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
        repeat = context.evaluate_expr(self.repeat_times)
        mode = context.evaluate_expr(self.mode)
        dst_blk_stride = context.evaluate_expr(self.dst_blk_stride)
        src0_blk_stride = context.evaluate_expr(self.src0_blk_stride)
        src1_blk_stride = context.evaluate_expr(self.src1_blk_stride)
        dst_rep_stride = context.evaluate_expr(self.dst_rep_stride)
        src0_rep_stride = context.evaluate_expr(self.src0_rep_stride)
        src1_rep_stride = context.evaluate_expr(self.src1_rep_stride)

        # check repeat_times
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times: {}".format(repeat))
        # check strides
        check_vector_stride([dst_blk_stride, src0_blk_stride, src1_blk_stride],
                            [dst_rep_stride, src0_rep_stride, src1_rep_stride],
                            MAX_BLK_STRIDE_SINGLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE, ["dst", "src0", "src1"])

        # check address overlap
        mask_value = _eval_mask(self.mask, context)
        if self.src0.buffer == self.dst.buffer:
            check_address_overlapping(
                "vsel", mask_value, self.dst, self.src0,
                BLK_NUM_PER_REP,
                ONE_REP_BYTE_SIZE //
                max(get_bit_len(self.dst.dtype),
                    get_bit_len(self.src0.dtype)),
                ONE_REP_BYTE_SIZE // get_bit_len(self.dst.dtype),
                ONE_REP_BYTE_SIZE // get_bit_len(self.src0.dtype),
                repeat, dst_blk_stride, src0_blk_stride,
                dst_rep_stride, src0_rep_stride,
                context.evaluate_expr(self.dst.offset),
                context.evaluate_expr(self.src0.offset), msg="dst and src0")

        if mode in (0, 2):
            if self.src1.buffer == self.dst.buffer:
                check_address_overlapping(
                    self.name, mask_value, self.dst, self.src1,
                    BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE //
                    max(get_bit_len(self.dst.dtype),
                        get_bit_len(self.src1.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(self.src1.dtype),
                    repeat, dst_blk_stride, src1_blk_stride,
                    dst_rep_stride, src1_rep_stride,
                    context.evaluate_expr(self.dst.offset),
                    context.evaluate_expr(self.src1.offset), msg="dst and src1")

        # check sel overflow
        if mode in (1, 2):
            if isinstance(self.mask, (list, tuple)):
                mask_value = [context.evaluate_expr(value) for value in
                              self.mask]
            else:
                mask_value = context.evaluate_expr(self.mask)
            # check dst sel address overlap
            if self.dst.buffer == self.sel.buffer:
                check_sel_dst_overlap(
                    self.dst, self.src0, self.sel, mask_value,
                    context.evaluate_expr(self.dst.offset),
                    context.evaluate_expr(self.sel.offset),
                    repeat, dst_blk_stride, dst_rep_stride)

            check_sel_overflow(self.dst, self.src0, self.sel, mask_value,
                               repeat)

        xt_idx = temp_env.alloc_register()
        x_t = dst_blk_stride
        x_t |= src0_blk_stride << _SRC_BLK_STRIDE_SHIFT_POS
        x_t |= src1_blk_stride << _SRC1_BLK_STRIDE_SHIFT_POS
        x_t |= dst_rep_stride << _DST_REPEAT_STRIDE_SHIFT_POS
        x_t |= src0_rep_stride << _SRC_REPEAT_STRIDE_SHIFT_POS
        x_t |= src1_rep_stride << _SRC1_REPEAT_STRIDE_SHIFT_POS
        x_t |= mode << _MODE_SHIFT_POS
        x_t |= repeat << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class ScatterVectorBinary(STMT):
    """ScatterVectorBinary instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, name, mask, dst_list, src0_list, src1_list,
                 repeat_times, dst_rep_stride, src0_rep_stride,
                 src1_rep_stride, *strides):
        # pylint: disable=R0913
        super(ScatterVectorBinary, self).__init__(source_info)
        self.name = name
        self.mask = mask
        self.dst_list = dst_list
        self.src0_list = src0_list
        self.src1_list = src1_list
        self.repeat_times = repeat_times
        self.strides = (dst_rep_stride, src0_rep_stride,
                        src1_rep_stride, strides)
        self.list_len = len(TikUtil.to_list(dst_list))

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
        set_vector_mask(self.mask, context,
                        tensor_bit_len=max(
                            get_bit_len(self.dst_list[VA0_INDEX].dtype),
                            get_bit_len(self.src0_list[VA0_INDEX].dtype),
                            get_bit_len(self.src1_list[VA0_INDEX].dtype)))

        dst_addr_set, dst_addr_list = _get_dst_addr_set_list(context, temp_env,
                                                             self.dst_list)

        src0_addr_list = _get_src_addr_list(context, temp_env, self.src0_list)
        src1_addr_list = _get_src_addr_list(context, temp_env, self.src1_list)

        param = _ENCODER.new_param()
        vec_dtype = self.src0_list[VA0_INDEX].dtype
        if vec_dtype != self.dst_list[VA0_INDEX].dtype:
            vec_dtype = "fmix"
        if self.name in _SCATTER_VECTOR_FOR_SUPPORT_S16:
            param.type = _VEC_DATA_TYPE_ENCODING_V200[vec_dtype]
        else:
            param.type = _VEC_DATA_TYPE_ENCODING[vec_dtype]
        param.vad = _create_va_reg(context, temp_env, dst_addr_list)
        param.van = _create_va_reg(context, temp_env, src0_addr_list)
        param.vam = _create_va_reg(context, temp_env, src1_addr_list)
        param.xt = self.create_gpr_x_t(context, temp_env)

        instr = _SCATER_VECTOR_VECTOR_FN_ENCODER[self.name](param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        for scope, buffer_addr, alloc_size, ptr in dst_addr_set:
            context.model.read_memory(buffer_addr, scope, ptr, alloc_size)

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
        dst_rep_stride, src0_rep_stride, src1_rep_stride, _ = self.strides
        repeat = context.evaluate_expr(self.repeat_times)
        dst_rep_stride = context.evaluate_expr(dst_rep_stride)
        src0_rep_stride = context.evaluate_expr(src0_rep_stride)
        src1_rep_stride = context.evaluate_expr(src1_rep_stride)
        # check repeat
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times: {}".format(repeat))
        # check strides
        check_vector_stride(None,
                            [dst_rep_stride, src0_rep_stride, src1_rep_stride],
                            None, MAX_REP_STRIDE_DOUBLE_BYTE,
                            ["dst", "src0", "src1"], is_scatter=True)

        # check tensor overflow
        if isinstance(self.mask, (list, tuple)):
            mask_value = [context.evaluate_expr(value) for value in self.mask]
        else:
            mask_value = context.evaluate_expr(self.mask)
        # check address overlapping
        debug_check_scatter_overlap(
            context, mask_value, self.dst_list, self.src0_list,
            repeat, dst_rep_stride, src0_rep_stride, name=self.name,
            msg="dst_list and src0_list")
        debug_check_scatter_overlap(
            context, mask_value, self.dst_list, self.src1_list,
            repeat, dst_rep_stride, src1_rep_stride, name=self.name,
            msg="dst_list and src1_list")

        check_scatter_vector_overflow([self.dst_list[:self.list_len],
                                       self.src0_list[:self.list_len],
                                       self.src1_list[:self.list_len]],
                                      ["dst_list", "src0_list", "src1_list"],
                                      mask_value, repeat,
                                      [dst_rep_stride, src0_rep_stride,
                                       src1_rep_stride])
        src0_rep_stride_shift_pos = 16
        src1_rep_stride_shift_pos = 32

        xt_idx = temp_env.alloc_register()
        x_t = dst_rep_stride
        x_t |= src0_rep_stride << src0_rep_stride_shift_pos
        x_t |= src1_rep_stride << src1_rep_stride_shift_pos
        x_t |= repeat << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class ScatterSingleVector(STMT):
    """ScatterSingleVector instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, name, mask, dst_list, src_list,
                 repeat_times, dst_rep_stride, src_rep_stride):
        # pylint: disable=R0913
        super(ScatterSingleVector, self).__init__(source_info)
        self.name = name
        self.mask = mask
        self.dst_list = dst_list
        self.src_list = src_list
        self.repeat_times = repeat_times
        self.dst_rep_stride = dst_rep_stride
        self.src_rep_stride = src_rep_stride
        self.list_len = len(TikUtil.to_list(dst_list))

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
        set_vector_mask(self.mask, context,
                        tensor_bit_len=max(
                            get_bit_len(self.dst_list[VA0_INDEX].dtype),
                            get_bit_len(self.src_list[VA0_INDEX].dtype)))

        dst_addr_set, dst_addr_list = _get_dst_addr_set_list(context, temp_env,
                                                             self.dst_list)

        src_addr_list = _get_src_addr_list(context, temp_env, self.src_list)

        param = _ENCODER.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[self.src_list[VA0_INDEX].dtype]
        if self.name in _SCATTER_VECTOR_FOR_SUPPORT_S16:
            param.type = _VEC_DATA_TYPE_ENCODING_0_[self.src_list[VA0_INDEX].dtype]
            param.type |= 1 << 2
        param.vad = _create_va_reg(context, temp_env, dst_addr_list)
        param.van = _create_va_reg(context, temp_env, src_addr_list)
        param.xt = self.create_gpr_x_t(context, temp_env)

        instr = _SCATTER_VECTOR_SINGLE_FN_ENCODER[self.name](param)
        context.model.step(instr)

        temp_env.check_mem_access(context.model, True)

        for scope, buffer_addr, alloc_size, ptr in dst_addr_set:
            context.model.read_memory(buffer_addr, scope, ptr, alloc_size)

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
        repeat = context.evaluate_expr(self.repeat_times)
        dst_rep_stride = context.evaluate_expr(self.dst_rep_stride)
        src_rep_stride = context.evaluate_expr(self.src_rep_stride)
        # check params
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times: {}".format(repeat))
        check_vector_stride(None, [dst_rep_stride, src_rep_stride],
                            None, MAX_REP_STRIDE_DOUBLE_BYTE,
                            ["dst", "src"], is_scatter=True)

        # check tensor overflow
        if isinstance(self.mask, (list, tuple)):
            mask_value = [context.evaluate_expr(value) for value in self.mask]
        else:
            mask_value = context.evaluate_expr(self.mask)
        # check address overlapping
        debug_check_scatter_overlap(
            context, mask_value, self.dst_list, self.src_list,
            repeat, dst_rep_stride, src_rep_stride, name=self.name)

        check_scatter_vector_overflow([self.dst_list[:self.list_len],
                                       self.src_list[:self.list_len]],
                                      ["dst_list", "src_list"],
                                      mask_value, repeat,
                                      [dst_rep_stride, src_rep_stride])
        src_rep_stride_shift_pos = 16

        xt_idx = temp_env.alloc_register()
        x_t = dst_rep_stride
        x_t |= src_rep_stride << src_rep_stride_shift_pos
        x_t |= repeat << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class ScatterVCMP(STMT):
    """ScatterVCMP instruction"""
    def __init__(self, source_info, name, mask, src0_list, src1_list):
        # pylint: disable=R0913
        super(ScatterVCMP, self).__init__(source_info)
        self.name = name
        self.mask = mask
        self.src0_list = src0_list
        self.src1_list = src1_list
        self.list_len = len(TikUtil.to_list(src0_list))

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
        set_vector_mask(self.mask, context,
                        tensor_bit_len=max(
                            get_bit_len(self.src0_list[VA0_INDEX].dtype),
                            get_bit_len(self.src1_list[VA0_INDEX].dtype)))

        # check tensor overflow
        if isinstance(self.mask, (list, tuple)):
            mask_value = [context.evaluate_expr(value) for value in self.mask]
        else:
            mask_value = context.evaluate_expr(self.mask)
        # vcmp have default_rep_times as 1, default_rep_stride as 0
        check_scatter_vector_overflow([self.src0_list[:self.list_len],
                                       self.src1_list[:self.list_len]],
                                      ["src0_list", "src1_list"],
                                      mask_value, MIN_REPEAT_TIMES, [0, 0])

        src0_addr_list = _get_src_addr_list(context, temp_env, self.src0_list)
        src1_addr_list = _get_src_addr_list(context, temp_env, self.src1_list)

        param = context.encoder.new_param()
        intrin_tokens = self.name.split('_')
        if intrin_tokens[0] == "scatter":
            cond_op = intrin_tokens[2]
        else:
            cond_op = intrin_tokens[1]
        param.cond_op = _VEC_CMP_OP_ENCODER[cond_op]
        param.type = _VEC_DATA_TYPE_ENCODING_V200[
            self.src0_list[VA0_INDEX].dtype]
        param.van = _create_va_reg(context, temp_env, src0_addr_list)
        param.vam = _create_va_reg(context, temp_env, src1_addr_list)
        param.xt = ScatterVCMP.create_gpr_x_t(context, temp_env)

        instr = context.encoder.gen_vcmpv(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

    @staticmethod
    def create_gpr_x_t(context, temp_env):
        """create general purpose register x_t

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
        # this instrution only repeats once
        x_t |= 1 << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class ScatterVectorScalar(STMT):
    """ScatterVectorScalar instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, name, mask, store_high_half, dst_list,
                 src0_list, src1, repeat_times, dst_stride, src_stride,
                 mask_mode="normal"):
        # pylint: disable=R0913
        super(ScatterVectorScalar, self).__init__(source_info)
        self.name = name
        self.mask = mask
        self.store_high_half = store_high_half
        self.dst_list = dst_list
        self.src0_list = src0_list
        self.src1 = src1
        self.repeat_times = repeat_times
        self.strides = (dst_stride, src_stride)
        self.mask_mode = mask_mode
        self.list_len = len(TikUtil.to_list(dst_list))

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

        set_vector_mask(self.mask, context, self.mask_mode,
                        tensor_bit_len=max(
                            get_bit_len(self.src0_list[VA0_INDEX].dtype),
                            get_bit_len(self.dst_list[VA0_INDEX].dtype)))

        dst_addr_set, dst_addr_list = _get_dst_addr_set_list(context, temp_env,
                                                             self.dst_list)

        src0_addr_list = _get_src_addr_list(context, temp_env, self.src0_list)

        param = _ENCODER.new_param()
        # fmix
        param.type = self.gen_param_type(context)
        param.vad = _create_va_reg(context, temp_env, dst_addr_list)
        param.van = _create_va_reg(context, temp_env, src0_addr_list)
        param.xm = self.create_gpr_x_m(context, temp_env)
        param.xt = self.create_gpr_x_t(context, temp_env)

        if self.name == 'scatter_vaxpy':
            param.h = int(bool(self.store_high_half))

        instr = _SCATTER_VECTOR_SCALAR[self.name](param)

        if self.mask_mode == "counter":
            orig_ctrl_value = _set_mask_counter_mode(context)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        # mask: counter_mode, reset CTRL as orig_ctrl_value
        if self.mask_mode == "counter":
            context.model.write_spr('CTRL', orig_ctrl_value)

        for scope, buffer_addr, alloc_size, ptr in dst_addr_set:
            context.model.read_memory(buffer_addr, scope, ptr, alloc_size)

    def gen_param_type(self, context):
        """genarate type encoding param

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        param_type : the type encoding
        """
        dtype = self.dst_list[VA0_INDEX].dtype
        if dtype != self.src0_list[VA0_INDEX].dtype:
            dtype = 'fmix'

        if get_soc_name() in (ASCEND_310, ASCEND_910):
            param_type = _VEC_DATA_TYPE_ENCODING[dtype]
        else:
            # scatter_vadds/vmuls/vmaxs/vmins
            if self.name in _SPECIAL_DTYPE_INSTR:
                param_type = _VEC_DATA_TYPE_ENCODING_V200[dtype]
                param_type |= _SPECIAL_DTYPE_INSTR[self.name] << 2
            elif self.name == 'scatter_vaxpy':
                param_type = _VECTOR_DTYPE_FN_ENCODER[dtype]
        return param_type

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
        scalar = context.evaluate_expr(self.src1)

        xm_idx = temp_env.alloc_register()
        x_m = cvt_float_to_uint(self.src0_list[VA0_INDEX].dtype, scalar)
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
        repeat = context.evaluate_expr(self.repeat_times)
        dst_stride, src_stride = self.strides
        dst_stride = context.evaluate_expr(dst_stride)
        src_stride = context.evaluate_expr(src_stride)
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times: {}".format(repeat))
        # check strides
        check_vector_stride(None, [dst_stride, src_stride],
                            None, MAX_REP_STRIDE_DOUBLE_BYTE,
                            ["dst", "src"], is_scatter=True)

        # check tensor overflow
        mask_value = _eval_mask(self.mask, context)
        # check address overlapping
        debug_check_scatter_overlap(
            context, mask_value, self.dst_list, self.src0_list,
            repeat, dst_stride, src_stride,
            mask_mode=self.mask_mode, store_high_half=self.store_high_half,
            name=self.name, msg="dst_list and src0_list")

        check_scatter_vector_overflow([self.src0_list[:self.list_len],
                                       self.dst_list[:self.list_len]],
                                      ["src_list", "dst_list"], mask_value,
                                      repeat,
                                      [src_stride, dst_stride],
                                      store_high_half=self.store_high_half,
                                      mask_mode=self.mask_mode)

        src_stride_shift_pos = 16

        xt_idx = temp_env.alloc_register()
        x_t = dst_stride
        x_t |= src_stride << src_stride_shift_pos
        x_t |= repeat << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class ScatterVmulconv(STMT):
    """ScatterVmulconv instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, mask, store_high_half, dst_list, src0_list,
                 src1_list, repeat_times, dst_rep_stride, src0_rep_stride,
                 src1_rep_stride):
        # pylint: disable=R0913
        super(ScatterVmulconv, self).__init__(source_info)
        self.mask = mask
        self.store_high_half = store_high_half
        self.dst_list = dst_list
        self.src0_list = src0_list
        self.src1_list = src1_list
        self.repeat_times = repeat_times
        self.stride = (dst_rep_stride, src0_rep_stride, src1_rep_stride)
        self.list_len = len(TikUtil.to_list(dst_list))

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
        set_vector_mask(self.mask, context,
                        tensor_bit_len=max(
                            get_bit_len(self.dst_list[VA0_INDEX].dtype),
                            get_bit_len(self.src0_list[VA0_INDEX].dtype),
                            get_bit_len(self.src1_list[VA0_INDEX].dtype)))

        dst_addr_set, dst_addr_list = _get_dst_addr_set_list(context, temp_env,
                                                             self.dst_list)

        src0_addr_list = _get_src_addr_list(context, temp_env, self.src0_list)

        src1_addr_list = _get_src_addr_list(context, temp_env, self.src1_list)

        param = _ENCODER.new_param()
        # 1 is store high half; 0 not store high half
        param.h = 1 if self.store_high_half else 0
        param.type = _VMULCONV_DTYPE_ENCODING[self.dst_list[VA0_INDEX].dtype]
        param.vad = _create_va_reg(context, temp_env, dst_addr_list)
        param.van = _create_va_reg(context, temp_env, src0_addr_list)
        param.vam = _create_va_reg(context, temp_env, src1_addr_list)
        param.xt = self.create_gpr_x_t(context, temp_env)

        instr = _ENCODER.gen_vmulconvv(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        for scope, buffer_addr, alloc_size, ptr in dst_addr_set:
            context.model.read_memory(buffer_addr, scope, ptr, alloc_size)

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
        repeat = context.evaluate_expr(self.repeat_times)
        dst_rep_stride, src0_rep_stride, src1_rep_stride = self.stride
        dst_rep_stride = context.evaluate_expr(dst_rep_stride)
        src0_rep_stride = context.evaluate_expr(src0_rep_stride)
        src1_rep_stride = context.evaluate_expr(src1_rep_stride)

        mask_value = _eval_mask(self.mask, context)
        # check address overlapping
        debug_check_scatter_overlap(
            context, mask_value, self.dst_list, self.src0_list,
            repeat, dst_rep_stride, src0_rep_stride,
            store_high_half=self.store_high_half,
            name="scatter_vmulconv", msg="dst_list and src0_list")
        debug_check_scatter_overlap(
            context, mask_value, self.dst_list, self.src1_list,
            repeat, dst_rep_stride, src1_rep_stride,
            store_high_half=self.store_high_half,
            name="scatter_vmulconv", msg="dst_list and src1_list")

        check_scatter_vector_overflow(
            [self.src0_list[:self.list_len], self.src1_list[:self.list_len]],
            ["src0_list", "src1_list"],
            mask_value, repeat, [src0_rep_stride, src1_rep_stride])

        src0_rep_stride_shift_pos = 16
        src1_rep_stride_shift_pos = 32

        xt_idx = temp_env.alloc_register()
        x_t = dst_rep_stride
        x_t |= src0_rep_stride << src0_rep_stride_shift_pos
        x_t |= src1_rep_stride << src1_rep_stride_shift_pos
        x_t |= repeat << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class ScatterVconv(STMT):
    """ScatterVconv instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, mask, round_mode, dst_list, src_list,
                 repeat_times, dst_rep_stride, src_rep_stride, deqscale,
                 ldst_high_half):
        # pylint: disable=R0913
        super(ScatterVconv, self).__init__(source_info)
        self.mask = mask
        self.round_mode = round_mode
        if self.round_mode == 'none':
            self.round_mode = ''
        elif self.round_mode == 'ceiling':
            self.round_mode = 'ceil'
        self.store_high_half = ldst_high_half
        self.dst_list = dst_list
        self.src_list = src_list
        self.repeat_times = repeat_times
        self.stride = (dst_rep_stride, src_rep_stride)
        self.deq_scale = deqscale
        self.list_len = len(TikUtil.to_list(dst_list))

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

        set_vector_mask(self.mask, context,
                        tensor_bit_len=max(
                            get_bit_len(self.src_list[VA0_INDEX].dtype),
                            get_bit_len(self.dst_list[VA0_INDEX].dtype)))

        dst_addr_set, dst_addr_list = _get_dst_addr_set_list(context, temp_env,
                                                             self.dst_list)

        src_addr_list = _get_src_addr_list(context, temp_env, self.src_list)

        deq_mode = self.write_deqscale_spr(context, temp_env)
        conv_type = 0

        if deq_mode != '':
            conv_type = _VCONV_TYPE_ENCODING[(deq_mode,)]
        elif self.round_mode not in ('', None):
            conv_type = _VCONV_TYPE_ENCODING[self.src_list[VA0_INDEX].dtype,
                                             self.dst_list[VA0_INDEX].dtype,
                                             self.round_mode]
        else:
            conv_type = _VCONV_TYPE_ENCODING[self.src_list[VA0_INDEX].dtype,
                                             self.dst_list[VA0_INDEX].dtype]

        param = _ENCODER.new_param()
        # 1 is store high half; 0 not store high half
        param.h = 1 if self.store_high_half else 0
        param.conv_type = conv_type
        param.vad = _create_va_reg(context, temp_env, dst_addr_list)
        param.van = _create_va_reg(context, temp_env, src_addr_list)
        param.xt = self.create_gpr_x_t(context, temp_env)

        instr = _ENCODER.gen_vconvv(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        for scope, buffer_addr, alloc_size, ptr in dst_addr_set:
            context.model.read_memory(buffer_addr, scope, ptr, alloc_size)

    def write_deqscale_spr(self, context, temp_env):
        """write special purpose register DEQSCALE

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        None
        """
        src_type = self.src_list[VA0_INDEX].dtype
        dst_type = self.dst_list[VA0_INDEX].dtype

        deq_mode = ''

        if src_type == 'int32' and dst_type == 'float16':
            if self.deq_scale is None:
                TikCheckUtil.raise_error('vconv deq scale is None!')
            deq_scale = context.evaluate_expr(self.deq_scale)
            deq_mode = 'DEQ'
            bin_deq_scale = cvt_float_to_uint(dst_type, deq_scale)
            context.model.write_spr('DEQSCALE', bin_deq_scale)
        elif src_type == 'int16' and dst_type in ('int8', 'uint8'):
            if is_tensor(self.deq_scale):
                deq_mode = 'VDEQs162b8'
                deq_addr = copy_tensor_to_model_get_addr(
                    context, temp_env, self.deq_scale,
                    32, access_mode='r')
                deq_addr = deq_addr // 32
                context.model.write_spr('DEQSCALE', deq_addr)
            elif is_scalar(self.deq_scale):
                deq_mode = 'DEQs162b8'
                context.model.write_spr('DEQSCALE',
                                        context.evaluate_expr(self.deq_scale))
            else:
                deq_mode = 'DEQs162b8'
                if not isinstance(self.deq_scale, int):
                    TikCheckUtil.raise_error(
                        'invalid type for deqscale {}'.format(
                            type(self.deq_scale)))
                context.model.write_spr('DEQSCALE', self.deq_scale)
        return deq_mode

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
        repeat = context.evaluate_expr(self.repeat_times)
        dst_rep_stride, src_rep_stride = self.stride
        dst_rep_stride = context.evaluate_expr(dst_rep_stride)
        src_rep_stride = context.evaluate_expr(src_rep_stride)

        # check repeat
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times: {}".format(repeat))
        # check strides
        check_vector_stride(None, [dst_rep_stride, src_rep_stride],
                            None, MAX_REP_STRIDE_DOUBLE_BYTE,
                            ["dst", "src"], is_scatter=True)

        # check overflow
        if isinstance(self.mask, (list, tuple)):
            mask_value = [context.evaluate_expr(value) for value in self.mask]
        else:
            mask_value = context.evaluate_expr(self.mask)

        # check address overlapping
        debug_check_scatter_overlap(
            context, mask_value, self.dst_list, self.src_list,
            repeat, dst_rep_stride, src_rep_stride,
            store_high_half=self.store_high_half, name="scatter_vconv")

        check_scatter_vector_overflow([self.dst_list[:self.list_len],
                                       self.src_list[:self.list_len]],
                                      ["dst_list", "src_list"],
                                      mask_value, repeat,
                                      [dst_rep_stride, src_rep_stride],
                                      store_high_half=self.store_high_half)

        src_rep_stride_shift_pos = 16

        xt_idx = temp_env.alloc_register()
        x_t = dst_rep_stride
        x_t |= src_rep_stride << src_rep_stride_shift_pos
        x_t |= repeat << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class ScatterVsel(STMT):
    """ScatterVsel instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, mask, mode, dst_list, sel, src0_list,
                 src1, repeat_times, dst_rep_stride, src0_rep_stride,
                 src1_sel_rep_stride):
        # pylint: disable=R0913
        super(ScatterVsel, self).__init__(source_info)
        self.mask = mask
        self.dst_list = dst_list
        self.src0_list = src0_list
        self.src1 = src1
        self.sel = sel
        self.mode = mode
        self.params = (repeat_times, dst_rep_stride, src0_rep_stride,
                       src1_sel_rep_stride)
        self.list_len = len(TikUtil.to_list(dst_list))

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
        set_vector_mask(
            self.mask, context,
            tensor_bit_len=get_bit_len(self.dst_list[VA0_INDEX].dtype))

        dst_addr_set, dst_addr_list = _get_dst_addr_set_list(context, temp_env,
                                                             self.dst_list)

        src0_addr_list = _get_src_addr_list(context, temp_env, self.src0_list)

        mode = context.evaluate_expr(self.mode)

        param = _ENCODER.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[self.dst_list[VA0_INDEX].dtype]
        param.vad = _create_va_reg(context, temp_env, dst_addr_list)
        param.van = _create_va_reg(context, temp_env, src0_addr_list)
        param.vam = self.create_vam(context, temp_env, mode)
        param.xt = self.create_gpr_x_t(context, temp_env, mode)

        if mode == 2:
            cmp_addr = copy_tensor_to_model_get_addr(
                context, temp_env, self.sel, ALIGNED_ADDR, access_mode='r')
            context.model.write_spr('CMPMASK0', cmp_addr)

        instr = _ENCODER.gen_vselv(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)
        for scope, buffer_addr, alloc_size, ptr in dst_addr_set:
            context.model.read_memory(buffer_addr, scope, ptr, alloc_size)

    def create_gpr_x_t(self, context, temp_env, mode):
        """create general purpose register x_t

        Parameters
        ----------
        context : the stack context
        temp_env : the temp environment
        mode : the instruction mode

        Returns
        -------
        xt_idx
        """
        repeat, dst_rep_stride, src0_rep_stride, src1_sel_rep_stride = \
            self.params
        repeat = context.evaluate_expr(repeat)
        dst_rep_stride = context.evaluate_expr(dst_rep_stride)
        src0_rep_stride = context.evaluate_expr(src0_rep_stride)
        src1_sel_rep_stride = context.evaluate_expr(src1_sel_rep_stride)

        # check repeat_times
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], input "
            "repeat_times is {}".format(repeat))
        if mode == 0:
            TikCheckUtil.check_equality(
                repeat, 1, "repeat_times should be 1 when mode is 0, "
                           "input value is {}".format(repeat))
        # check strides
        check_vector_stride(None, [dst_rep_stride, src0_rep_stride,
                                   src1_sel_rep_stride], None,
                            MAX_REP_STRIDE_DOUBLE_BYTE,
                            ["dst", "src0", "src1"], is_scatter=True)

        if isinstance(self.mask, (list, tuple)):
            mask_value = [context.evaluate_expr(value) for value in self.mask]
        else:
            mask_value = context.evaluate_expr(self.mask)

        # check address overlapping
        debug_check_scatter_overlap(
            context, mask_value, self.dst_list, self.src0_list,
            repeat, dst_rep_stride, src0_rep_stride,
            name="scatter_vsel", msg="dst_list and src0_list")

        if mode in (0, 2):
            # check address overlapping
            debug_check_scatter_overlap(
                context, mask_value, self.dst_list, self.src1,
                repeat, dst_rep_stride, src1_sel_rep_stride,
                name="scatter_vsel", msg="dst_list and src1")

        if mode in (1, 2):
            check_db_vsel_dst_sel_overlap(
                context, self.dst_list, self.sel,
                self.src0_list, mask_value, mode, repeat, dst_rep_stride)

        if mode == 1:
            # check sel overflow
            check_sel_overflow(self.dst_list[VA0_INDEX],
                               self.src0_list[VA0_INDEX],
                               self.sel[VA0_INDEX], mask_value, repeat)
        elif mode == 2:
            # check sel overflow
            check_sel_overflow(self.dst_list[VA0_INDEX],
                               self.src0_list[VA0_INDEX],
                               self.sel, mask_value, repeat)
        # check dst_list and src0_list overflow
        check_scatter_vector_overflow([self.dst_list[:self.list_len],
                                       self.src0_list[:self.list_len]],
                                      ["dst_list", "src0_list"],
                                      mask_value, repeat,
                                      [dst_rep_stride, src0_rep_stride])
        # check src1 overflow
        if mode in (0, 2):
            check_scatter_vector_overflow([self.src1[:self.list_len]],
                                          ["src1"], mask_value,
                                          repeat, [src1_sel_rep_stride])

        xt_idx = temp_env.alloc_register()
        src_rep_stride_shift_pos = 16
        src1_sel_rep_stride_shift_pos = 32
        mode_shift_pos = 48
        x_t = dst_rep_stride
        x_t |= src0_rep_stride << src_rep_stride_shift_pos
        x_t |= src1_sel_rep_stride << src1_sel_rep_stride_shift_pos
        x_t |= repeat << REPEAT_SHIFT_POS
        x_t |= mode << mode_shift_pos
        context.model.write_gpr(xt_idx, x_t)

        return xt_idx

    def create_vam(self, context, temp_env, mode):
        """create va reg

        Parameters
        ----------
        context : the stack context
        temp_env : the temp environment
        mode : the instruction mode

        Returns
        -------
        vam_id
        """
        if mode == 1:
            const_value = context.evaluate_expr(self.src1)
            sel_addr_list = _get_src_addr_list(context, temp_env, self.sel)
            vam_id = _create_va_reg(context, temp_env, sel_addr_list)
            context.model.write_spr('CMPMASK0',
                                    cvt_float_to_uint(self.src0_list[VA0_INDEX].
                                                      dtype,
                                                      const_value))
        else:
            src1_addr_list = _get_src_addr_list(context, temp_env, self.src1)
            vam_id = _create_va_reg(context, temp_env, src1_addr_list)
        return vam_id


class Vnchwconv(STMT):
    """Vnchwconv instruction"""
    def __init__(self, source_info, dst_high_half, src_high_half, dst_list,
                 src_list, repeat_times, dst_rep_stride, src_rep_stride,
                 name=None):
        # pylint: disable=R0913
        super(Vnchwconv, self).__init__(source_info)
        self.dst_high_half = dst_high_half
        self.src_high_half = src_high_half
        self.dst_list = dst_list
        self.src_list = src_list
        self.repeat_times = repeat_times
        self.strides = (dst_rep_stride, src_rep_stride)
        self.name = name
        if self.name is None:
            self.name = "vnchwconv"

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

        # vnchwconv is a special instr
        # for v100 pvmodel, dst is rw mode; v200 dst is only w mode
        dst_addr_set, dst_addr_list = _get_dst_addr_set_list(
            context, temp_env, self.dst_list, spec_instr=True)

        src_addr_list = _get_src_addr_list(context, temp_env, self.src_list)

        # we treat b16 as uint16 ...
        bit_width = get_dtype_bit_width(self.src_list[VA0_INDEX].dtype)
        dtype = 'uint' + bit_width

        param = _ENCODER.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[dtype]
        TikCheckUtil.check_in_range(param.type, range(_MAX_PARAM_TYPE))
        param.vad = _create_va_reg(context, temp_env, dst_addr_list)
        param.van = _create_va_reg(context, temp_env, src_addr_list)
        param.xt = self.create_gpr_x_t(context, temp_env)
        param.srcH = self.src_high_half
        param.dstH = self.dst_high_half

        instr = _ENCODER.gen_vnchwconv(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model)
        for scope, buffer_addr, alloc_size, ptr in dst_addr_set:
            context.model.read_memory(buffer_addr, scope, ptr, alloc_size)

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
        repeat = context.evaluate_expr(self.repeat_times)
        # check repeat
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255],"
            " input value is %s" % str(repeat))
        dst_rep_stride, src_rep_stride = self.strides
        # check strides
        check_vector_stride(None, [context.evaluate_expr(dst_rep_stride),
                                   context.evaluate_expr(src_rep_stride)],
                            None, MAX_REP_STRIDE_DOUBLE_BYTE, ["dst", "src"])

        # check address overlapping
        if self.dst_list[VA0_INDEX].dtype in ("uint8", "int8"):
            mask_len = MASK_VALUE_128
        else:
            mask_len = ONE_BLK_SIZE*VNCHWCONV_LIST_LEN // \
                       DTYPE_SIZE[self.dst_list[VA0_INDEX].dtype]
        debug_check_scatter_overlap(
            context, mask_len, self.dst_list, self.src_list,
            repeat, context.evaluate_expr(dst_rep_stride),
            context.evaluate_expr(src_rep_stride),
            store_high_half=self.dst_high_half,
            src_store_high_half=self.src_high_half, name=self.name)

        src_rep_stride_shift_pos = 16

        xt_idx = temp_env.alloc_register()
        x_t = context.evaluate_expr(dst_rep_stride)
        x_t |= context.evaluate_expr(src_rep_stride) << \
               src_rep_stride_shift_pos
        x_t |= repeat << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class ListListEltwise(STMT):
    """ListListEltwise instruction"""
    def __init__(self, source_info, name, dst_list, src0_list, src1_list):
        # pylint: disable=R0913
        super(ListListEltwise, self).__init__(source_info)
        self.name = name
        self.dst_list = dst_list
        self.src0_list = src0_list
        self.src1_list = src1_list

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

        dst_addr_set, dst_addr_list = _get_dst_addr_set_list(context, temp_env,
                                                             self.dst_list)
        src0_addr_list = _get_src_addr_list(context, temp_env, self.src0_list)
        src1_addr_list = _get_src_addr_list(context, temp_env, self.src1_list)

        param = _ENCODER.new_param()
        param.vad = _create_va_reg(context, temp_env, dst_addr_list)
        param.van = _create_va_reg(context, temp_env, src0_addr_list)
        param.vam = _create_va_reg(context, temp_env, src1_addr_list)

        instr = _LIST_LIST_ELT[self.name](param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        for scope, buffer_addr, alloc_size, ptr in dst_addr_set:
            context.model.read_memory(buffer_addr, scope, ptr, alloc_size)


class VecCMPScalar(STMT):
    """VecCMPScalar instruction"""
    def __init__(self, source_info, name, dst, src, scalar, repeat_times,
                 src_blk_stride, src_rep_stride):
        # pylint: disable=R0913
        super(VecCMPScalar, self).__init__(source_info)
        self.name = name
        self.dst = dst
        self.src = src
        self.scalar = scalar
        self.repeat_times = repeat_times
        self.src_blk_stride = src_blk_stride
        self.src_rep_stride = src_rep_stride

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        align = _vec_template_align(context, self.src.dtype)
        temp_env = TempEnv()

        xn_idx, _, src_buffer_size, _ = copy_tensor_to_model(
            context, temp_env, self.src, align, True, access_mode='r')

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, align, True, access_mode='w')

        param = context.encoder.new_param()
        param.cond_op = _VEC_CMP_OP_ENCODER[self.name.split('_')[1]]
        param.type = _VCMPV_TYPE_ENCODER[self.src.dtype]
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = self.create_gpr_x_m(context, temp_env)
        param.xt = self.create_gpr_x_t(context, temp_env)

        # for vcmpvs instr, mask is always f16: 128, f32:64
        default_mask = ONE_REP_BYTE_SIZE // DTYPE_SIZE[self.src.dtype]
        check_read_mem_out_of_bounds(context, src_buffer_size, default_mask,
                                     self.src, self.repeat_times,
                                     self.src_blk_stride, self.src_rep_stride)

        instr = context.encoder.gen_vcmpvsx(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)

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
        scalar = context.evaluate_expr(self.scalar)
        xm_idx = temp_env.alloc_register()
        x_m = cvt_float_to_uint(self.src.dtype, scalar)

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
        repeat = context.evaluate_expr(self.repeat_times)
        # check repeat
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat times should be in the range of [0, 255], "
            "input repeat_times: {}".format(repeat))
        src_blk_stride = context.evaluate_expr(self.src_blk_stride)
        src_repeat_stride = context.evaluate_expr(self.src_rep_stride)
        # check stride
        check_integer_in_range(
            src_blk_stride, range(MAX_BLK_STRIDE_DOUBLE_BYTE),
            "src_blk_stride should be in the range of [0, 65535], "
            "input src_blk_stride: {}".format(src_blk_stride))
        check_integer_in_range(
            src_repeat_stride, range(MAX_REP_STRIDE_SINGLE_BYTE),
            "src_rep_stride should be in the range of [0, 255], "
            "input src_rep_stride: {}".format(src_repeat_stride))

        # check address_overlapping
        default_blk_stride = 1
        default_rep_stride = 8
        if self.src.buffer == self.dst.buffer:
            check_address_overlapping(
                self.name, ONE_REP_BYTE_SIZE //
                max(DTYPE_SIZE[self.dst.dtype], DTYPE_SIZE[self.src.dtype]),
                self.dst, self.src, BLK_NUM_PER_REP,
                ONE_REP_BYTE_SIZE // max(get_bit_len(self.dst.dtype),
                                         get_bit_len(self.src.dtype)),
                ONE_REP_BYTE_SIZE // get_bit_len(self.dst.dtype),
                ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype),
                repeat, default_blk_stride, src_blk_stride,
                default_rep_stride, src_repeat_stride,
                context.evaluate_expr(self.dst.offset),
                context.evaluate_expr(self.src.offset))

        xt_idx = temp_env.alloc_register()
        x_t = 0
        x_t |= src_blk_stride << _SRC1_BLK_STRIDE_SHIFT_POS
        x_t |= repeat << REPEAT_SHIFT_POS
        x_t |= src_repeat_stride << _SRC1_REPEAT_STRIDE_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class V4DTRANS(STMT):
    """V4DTRANS instruction"""
    def __init__(self, source_info, chw2hwc, dst, src, m_len, channels):
        # pylint: disable=R0913
        super(V4DTRANS, self).__init__(source_info)
        self.chw2hwc = chw2hwc
        self.dst = dst
        self.src = src
        self.m_len = m_len
        self.channels = channels

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
            context, temp_env, self.src, ALIGNED_ADDR, True, access_mode='r')
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGNED_ADDR, True, access_mode='w')

        # we treat b16 as uint16 ...
        bit_width = get_dtype_bit_width(self.src.dtype)
        dtype = 'uint' + bit_width

        param = context.encoder.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[dtype]
        param.xd = xd_idx
        param.xn = xn_idx
        param.xt = self.create_gpr_x_t(context, temp_env)

        instr = context.encoder.gen_v4dtrans(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

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
        xt_idx
        """
        if self.chw2hwc:
            trans_dir = 0
        else:
            trans_dir = 1

        m_len = context.evaluate_expr(self.m_len)
        channels = context.evaluate_expr(self.channels)

        # check m_len and channels in range
        TikCheckUtil.check_in_range(
            m_len, range(MIN_M_LEN, MAX_M_LEN),
            "m_len should be in the range of [1,4095], "
            "input m_len: %s" % str(m_len))

        # check  image size must be 32B align
        image_size = m_len*get_dtype_size(self.src.dtype)
        TikCheckUtil.check_equality(
            image_size % ONE_BLK_SIZE, 0,
            "H*W*dtype_size should be 32 Byte aligned, "
            "input size is %s" % str(image_size))

        TikCheckUtil.check_in_range(
            channels, range(MIN_CHANNELS, MAX_CHANNELS),
            "channels should be in the range of "
            "[1,4095], input channels: %s" % str(channels))
        # check tensor overflow
        src_offset = context.evaluate_expr(self.src.offset)
        dst_offset = context.evaluate_expr(self.dst.offset)
        # check address overlap
        check_addr_overlap_v4dtrans(
            self.dst, self.src, m_len, channels, dst_offset, src_offset)

        TikCheckUtil.check_le(
            m_len * channels,
            reduce_mul(self.src.indice.origin_shape) - src_offset,
            "src tensor overflow, m_len*channels should be less equal src size")
        TikCheckUtil.check_le(
            m_len * channels,
            reduce_mul(self.dst.indice.origin_shape) - dst_offset,
            "dst tensor overflow, m_len*channels should be less equal dst size")

        xt_idx = temp_env.alloc_register()
        x_t = m_len
        x_t |= (channels << _CHANNELS_SHIFT_BIT_POS)
        x_t |= trans_dir << _TRANS_DIR_SHIFT_BIT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


class VReduce(STMT):
    """VReduce instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, mask, dst, src0, src1_pattern, repeat_times,
                 src0_blk_stride, src0_rep_stride, src1_rep_stride, stride_unit,
                 rsvd_scalar, mask_mode):
        # pylint: disable=R0913
        super(VReduce, self).__init__(source_info)
        self.mask = mask
        self.dst = dst
        self.src0 = src0
        self.src1_pattern = src1_pattern
        self.repeat_times = repeat_times
        self.src0_blk_stride = src0_blk_stride
        self.src0_rep_stride = src0_rep_stride
        self.src1_rep_stride = src1_rep_stride
        self.stride_unit = stride_unit
        self.rsvd_scalar = rsvd_scalar
        self.mask_mode = mask_mode

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        if self.mask_mode == "counter":
            orig_ctrl_value = _set_mask_counter_mode(context)

        set_vector_mask(self.mask, context, self.mask_mode,
                        tensor_bit_len=max(get_bit_len(self.src0.dtype),
                                           get_bit_len(self.dst.dtype)))
        align = _vec_template_align(context, self.src0.dtype)
        temp_env = TempEnv()
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, align, True, access_mode='w')

        param = context.encoder.new_param()
        param.type = self.get_type_encoding()
        param.xd = xd_idx
        param.xn = self.create_gpr_x_n(context, temp_env, align)
        param.xm, pattern = self.create_gpr_x_m(context, temp_env, align)
        param.xt = self.create_gpr_x_t(context, temp_env, pattern)

        instr = context.encoder.gen_vreduce(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        # mask: counter_mode, reset CTRL as orig_ctrl_value
        if self.mask_mode == "counter":
            context.model.write_spr('CTRL', orig_ctrl_value)

        if self.rsvd_scalar is not None:
            context.update_var(self.rsvd_scalar,
                               context.model.read_spr('RSVD_CNT'))

        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)

    def create_gpr_x_m(self, context, temp_env, align):
        """create general purpose register x_m

        Parameters
        ----------
        context : the stack context
        temp_env : the temp environment
        align : align address

        Returns
        -------
        xm_idx, pattern
        """
        if is_tensor(self.src1_pattern):
            pattern = 0
            xm_idx, _, src1_alloc_size, _ = copy_tensor_to_model(
                context, temp_env, self.src1_pattern, align, True,
                access_mode='r')

            mask_value = _eval_mask(self.mask, context)
            check_vreduce_src1_overflow(
                mask_value, self.mask_mode,
                context.evaluate_expr(self.src1_rep_stride), self.src1_pattern,
                src1_alloc_size // DTYPE_SIZE[self.src1_pattern.dtype],
                context.evaluate_expr(self.repeat_times))
        else:
            pattern = self.src1_pattern
            # check int pattern
            check_integer_in_range(
                context.evaluate_expr(pattern),
                range(MIN_SRC1_PATTERN, MAX_SRC1_PATTERN),
                "src1_pattern should be in the range of [1, 6], "
                "input src1_pattern: {}".format(
                    context.evaluate_expr(pattern)))
            xm_idx = temp_env.alloc_register()
        return xm_idx, pattern

    def create_gpr_x_n(self, context, temp_env, align):
        """create general purpose register x_n

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xt_idx
        """
        xn_idx, _, src0_alloc_size, _ = copy_tensor_to_model(
            context, temp_env, self.src0, align, True, access_mode='r')
        if self.mask_mode == "normal":
            # in normal mode, mask parameter is b16: 128, b32: 64
            check_read_mem_out_of_bounds(
                context, src0_alloc_size,
                ONE_REP_BYTE_SIZE // DTYPE_SIZE[self.src0.dtype], self.src0,
                self.repeat_times, self.src0_blk_stride, self.src0_rep_stride,
                self.stride_unit, self.mask_mode)
        else:
            check_read_mem_out_of_bounds(
                context, src0_alloc_size, self.mask, self.src0,
                self.repeat_times, self.src0_blk_stride, self.src0_rep_stride,
                self.stride_unit, self.mask_mode)
        return xn_idx

    def get_type_encoding(self):
        """get type encoding code"""
        bit_width = get_dtype_bit_width(self.src0.dtype)
        if bit_width == '16':
            type_encoding = 0
        else:
            type_encoding = 1

        return type_encoding

    def create_gpr_x_t(self, context, temp_env, pattern):
        """create general purpose register x_t

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        pattern: the instr's pattern

        Returns
        -------
        xt_idx
        """
        src0_blk_stride = context.evaluate_expr(self.src0_blk_stride)
        src0_rep_stride = context.evaluate_expr(self.src0_rep_stride)
        src1_rep_stride = context.evaluate_expr(self.src1_rep_stride)
        repeat = context.evaluate_expr(self.repeat_times)
        pattern = context.evaluate_expr(pattern)
        # check repeat_times
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times: {}".format(repeat))
        # check strides
        check_integer_in_range(
            src0_blk_stride, range(MAX_BLK_STRIDE_SINGLE_BYTE),
            "src0_blk_stride should be in the range of [0, 255], "
            "input src0_blk_stride: {}".format(src0_blk_stride))
        check_integer_in_range(
            src0_rep_stride, range(MAX_REP_STRIDE_SINGLE_BYTE),
            "src0_rep_stride should be in the range of [0, 255], "
            "input src0_rep_stride: {}".format(src0_rep_stride))
        check_integer_in_range(
            src1_rep_stride, range(MAX_REP_STRIDE_SINGLE_BYTE),
            "src1_rep_stride should be in the range of [0, 255], "
            "input src1_rep_stride: {}".format(src1_rep_stride))

        # check vreduce address overlap
        if self.src0.buffer == self.dst.buffer:
            check_address_overlap_vreduce(
                self.dst, self.src0, pattern,
                _eval_mask(self.mask, context), BLK_NUM_PER_REP,
                ONE_REP_BYTE_SIZE // get_bit_len(self.dst.dtype),
                repeat, src0_blk_stride, src0_rep_stride,
                context.evaluate_expr(self.dst.offset),
                context.evaluate_expr(self.src0.offset),
                self.stride_unit, self.mask_mode)

        xt_idx = temp_env.alloc_register()
        x_t = 0
        x_t |= context.evaluate_expr(self.stride_unit) << STRIDE_UNIT_SHIFT_POS
        x_t |= repeat << REPEAT_SHIFT_POS
        x_t |= src0_rep_stride << _SRC_REPEAT_STRIDE_SHIFT_POS
        x_t |= src0_blk_stride << _SRC_BLK_STRIDE_SHIFT_POS
        x_t |= src1_rep_stride << _SRC1_REPEAT_STRIDE_SHIFT_POS
        x_t |= pattern << _PATTERN_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)
        return xt_idx


class VPadding(STMT):
    """Vpadding instruction"""
    # pylint: disable=R0902
    def __init__(self, source_info, mask, pad_mode, pad_side, dst, src,
                 repeat_times, dst_blk_stride, src_blk_stride, dst_rep_stride,
                 src_rep_stride, stride_unit, mask_mode):
        # pylint: disable=R0913
        super(VPadding, self).__init__(source_info)
        self.mask = mask
        self.pad_mode = pad_mode
        self.pad_side = pad_side
        self.dst = dst
        self.src = src
        self.repeat_times = repeat_times
        self.dst_blk_stride = dst_blk_stride
        self.src_blk_stride = src_blk_stride
        self.dst_rep_stride = dst_rep_stride
        self.src_rep_stride = src_rep_stride
        self.stride_unit = stride_unit
        self.mask_mode = mask_mode

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        mask = self.mask
        if self.mask_mode == "counter":
            orig_ctrl_value = _set_mask_counter_mode(context)

        set_vector_mask(mask, context, self.mask_mode,
                        tensor_bit_len=max(get_bit_len(self.dst.dtype),
                                           get_bit_len(self.src.dtype)))

        align = _vec_template_align(context, self.src.dtype)
        temp_env = TempEnv()

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, align, True, access_mode='w')

        bit_width = get_dtype_bit_width(self.src.dtype)
        if bit_width == '16':
            type_encoding = 0
        else:
            type_encoding = 1

        param = context.encoder.new_param()
        param.type = type_encoding
        param.xd = xd_idx
        param.xt = self.create_gpr_x_t(context, temp_env)
        param.xn = self.create_gpr_x_n(context, temp_env, align, bit_width)

        instr = context.encoder.gen_vpadding(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        # mask: counter_mode, reset CTRL as orig_ctrl_value
        if self.mask_mode == "counter":
            context.model.write_spr('CTRL', orig_ctrl_value)

        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)

    def create_gpr_x_n(self, context, temp_env, align, bit_width):
        """create general purpose register x_n

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xn_idx
        """
        xn_idx, _, src_buffer_size, _ = copy_tensor_to_model(
            context, temp_env, self.src, align, True, access_mode='r')
        if self.mask_mode == "normal":
            # all elements in src are read even their mask bits are invalid
            if bit_width == '32':
                mask_len = MASK_VALUE_64
            else:
                mask_len = MASK_VALUE_128
        else:
            mask_len = self.mask
        check_read_mem_out_of_bounds(context, src_buffer_size, mask_len,
                                     self.src, self.repeat_times,
                                     self.src_blk_stride,
                                     self.src_rep_stride, self.stride_unit,
                                     self.mask_mode)
        return xn_idx

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
        repeat = context.evaluate_expr(self.repeat_times)
        dst_block_stride = context.evaluate_expr(self.dst_blk_stride)
        src_block_stride = context.evaluate_expr(self.src_blk_stride)
        dst_repeat_stride = context.evaluate_expr(self.dst_rep_stride)
        src_repeat_stride = context.evaluate_expr(self.src_rep_stride)

        # check repeat_times
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times: {}".format(repeat))
        # check strides
        check_vector_stride([dst_block_stride, src_block_stride],
                            [dst_repeat_stride, src_repeat_stride],
                            MAX_BLK_STRIDE_DOUBLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE, ["dst", "src"])

        if self.mask_mode == "normal":
            # all elements in src are read even their mask bits are invalid
            if get_dtype_bit_width(self.src.dtype) == '32':
                mask_len = MASK_VALUE_64
            else:
                mask_len = MASK_VALUE_128
        else:
            mask_len = _eval_mask(self.mask, context)

        if self.src.buffer == self.dst.buffer:
            check_address_overlapping(
                "vpadding", _eval_mask(self.mask, context), self.dst, self.src,
                BLK_NUM_PER_REP, ONE_REP_BYTE_SIZE // max(
                    get_bit_len(self.dst.dtype), get_bit_len(self.src.dtype)),
                ONE_REP_BYTE_SIZE // get_bit_len(self.dst.dtype),
                ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype),
                repeat, dst_block_stride, src_block_stride,
                dst_repeat_stride, src_repeat_stride,
                context.evaluate_expr(self.dst.offset),
                context.evaluate_expr(self.src.offset),
                self.stride_unit, self.mask_mode, src_mask=mask_len)

        xt_idx = temp_env.alloc_register()
        x_t = dst_block_stride
        x_t |= src_block_stride << _SRC1_BLK_STRIDE_SHIFT_POS
        x_t |= dst_repeat_stride << _VPADD_DST_REP_SHIFT_POS
        x_t |= src_repeat_stride << _SRC1_REPEAT_STRIDE_SHIFT_POS
        x_t |= self.pad_mode << _MODE_SHIFT_POS
        x_t |= (0 if self.pad_side == 'left' else 1) << _PAD_SIDE_SHIFT_BIT_POS
        x_t |= context.evaluate_expr(self.stride_unit) << \
               STRIDE_UNIT_SHIFT_POS
        x_t |= repeat << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        return xt_idx


_VSCATTER_TYPE = {
    2: 0b0,
    4: 0b1
}


class VScatter(STMT):
    """ VScatter instruction """
    # pylint: disable=R0902
    def __init__(self, source_info, mask, dst, src,
                 dst_offset, repeat_times,
                 src_rep_stride, base_addr, stride_unit, mask_mode):
        # pylint: disable=R0913
        super(VScatter, self).__init__(source_info)
        self.mask = mask
        self.dst = dst
        self.src = src
        self.dst_offset = dst_offset
        self.repeat_times = repeat_times
        self.src_rep_stride = src_rep_stride
        self.base_addr = base_addr
        self.stride_unit = stride_unit
        self.mask_mode = mask_mode

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        src_dtype_size = get_dtype_size(self.src.dtype)
        src_align = src_dtype_size
        temp_env = TempEnv()

        _, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst,
            ALIGNED_ADDR, True, access_mode="w")

        xd_idx, _, dst_offset_size, _ = copy_tensor_to_model(
            context, temp_env, self.dst_offset,
            src_align, True, access_mode="r")

        if self.mask_mode == "counter":
            orig_ctrl_value = _set_mask_counter_mode(context)

        set_vector_mask(self.mask, context, self.mask_mode,
                        tensor_bit_len=max(get_bit_len(self.dst.dtype),
                                           get_bit_len(self.src.dtype)))

        param = context.encoder.new_param()
        param.xd = xd_idx
        param.xt = self.create_gpr_x_t(context, temp_env, dst_addr)
        param.xn = self.create_gpr_x_n(context, temp_env,
                                       src_align, dst_offset_size)
        param.type = _VSCATTER_TYPE[src_dtype_size]

        instr = context.encoder.gen_vscatter(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        # mask: counter_mode, reset CTRL as orig_ctrl_value
        if self.mask_mode == "counter":
            context.model.write_spr('CTRL', orig_ctrl_value)

        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)

    def create_gpr_x_n(self, context, temp_env, src_align, dst_offset_size):
        """create general purpose register x_n

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        src_align : src tensor addr align byte

        dst_offset_size: dst_offset buffer size

        Returns
        -------
        xt_idx
        """
        xn_idx, _, src_buffer_size, _ = copy_tensor_to_model(
            context, temp_env, self.src,
            src_align, True, access_mode="r")

        if self.mask_mode == "normal":
            # all elements in src are read even their mask bits are invalid
            if get_dtype_bit_width(self.src.dtype) == '32':
                mask_len = MASK_VALUE_64
            else:
                mask_len = MASK_VALUE_128
        else:
            mask_len = self.mask
        stride_unit = context.evaluate_expr(self.stride_unit)
        # stride_unit: 0,1 for stride, unit is 32B
        default_blk_stride = 1
        default_rep_stride = 8
        # check dst_offset read mem.
        # change offset dtype temporarily, for check overflow
        self.dst_offset.dtype = self.src.dtype
        check_read_mem_out_of_bounds(context, dst_offset_size, mask_len,
                                     self.dst_offset, self.repeat_times,
                                     default_blk_stride, default_rep_stride,
                                     mask_mode=self.mask_mode)
        # reset dst_offset dtype: int32
        self.dst_offset.dtype = "int32"

        if stride_unit in (2, 3):
            # stide_unit:2,3 for gap, unit is element
            default_blk_stride = 0
        check_read_mem_out_of_bounds(context, src_buffer_size, mask_len,
                                     self.src, self.repeat_times,
                                     default_blk_stride,
                                     self.src_rep_stride, self.stride_unit,
                                     self.mask_mode)
        return xn_idx

    def create_gpr_x_t(self, context, temp_env, dst_addr):
        """create general purpose register x_t

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        dst_addr : dst tensor addr

        Returns
        -------
        xt_idx
        """
        repeat = context.evaluate_expr(self.repeat_times)

        base_addr = context.evaluate_expr(self.base_addr)
        base_addr_offset = base_addr + dst_addr

        # default dst_rep_stride: 0
        dst_rep_stride = 0
        src_rep_stride = context.evaluate_expr(self.src_rep_stride)

        # check repeat_times
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times: {}".format(repeat))
        # check strides
        check_vector_stride(None,
                            [src_rep_stride],
                            None,
                            MAX_REP_STRIDE_SINGLE_BYTE, ["src"])
        # check base_addr
        # 2**31 - 1 is max base_addr, 31 bit len
        check_integer_in_range(base_addr, range(2**31),
                               "base_addr should be in"
                               " range of [0, 2**31 - 1]. "
                               "input base_addr: {}".format(base_addr))
        # check valid
        dst_scope_size = reduce_mul(
            self.dst.indice.origin_shape)*get_dtype_size(self.dst.dtype)
        TikCheckUtil.check_le(
            base_addr, context.evaluate_expr(dst_scope_size),
            "base_addr should be less equal than dst tensor's buffer size: {},"
            " input base_add: {}".format(dst_scope_size, base_addr))

        xt_idx = temp_env.alloc_register()
        x_t = int(base_addr_offset)
        x_t |= dst_rep_stride << DST_REPEAT_STRIDE_SHIFT_POS
        x_t |= src_rep_stride << SRC_REPEAT_STRIDE_SHIFT_POS
        x_t |= context.evaluate_expr(self.stride_unit) << STRIDE_UNIT_SHIFT_POS
        x_t |= repeat << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)
        return xt_idx


class VGather(STMT):
    """ VGather instruction """
    # pylint: disable=R0902
    def __init__(self, source_info, mask, dst, src, src_offset, repeat_times,
                 dst_rep_stride, base_addr, stride_unit, mask_mode):
        # pylint: disable=R0913
        super(VGather, self).__init__(source_info)
        self.mask = mask
        self.dst = dst
        self.src = src
        self.src_offset = src_offset
        self.repeat_times = repeat_times
        self.dst_rep_stride = dst_rep_stride
        self.base_addr = base_addr
        self.stride_unit = stride_unit
        self.mask_mode = mask_mode

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        dst_align = get_dtype_size(self.dst.dtype)
        temp_env = TempEnv()

        _, src_addr, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, ALIGNED_ADDR, True, access_mode="r")

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, dst_align, True, access_mode="w")

        if self.mask_mode == "counter":
            orig_ctrl_value = _set_mask_counter_mode(context)

        set_vector_mask(self.mask, context, self.mask_mode,
                        tensor_bit_len=max(get_bit_len(self.src.dtype),
                                           get_bit_len(self.dst.dtype)))

        param = context.encoder.new_param()
        param.xd = xd_idx
        param.xt = self.create_gpr_x_t(context, temp_env, src_addr)
        param.xn = self.create_gpr_x_n(context, temp_env)
        param.type = _VSCATTER_TYPE[get_dtype_size(self.dst.dtype)]

        instr = context.encoder.gen_vgather(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        # mask: counter_mode, reset CTRL as orig_ctrl_value
        if self.mask_mode == "counter":
            context.model.write_spr('CTRL', orig_ctrl_value)

        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)

    def create_gpr_x_n(self, context, temp_env):
        """create general purpose register x_t

        Parameters
        ----------
        context : the stack context
        temp_env : the temp environment

        Returns
        -------
        xn_idx
        """
        xn_idx, _, src_offset_buffer_size, _ = copy_tensor_to_model(
            context, temp_env, self.src_offset,
            ALIGNED_ADDR, True, access_mode="r")

        # stride_unit: 0,1 for stride, unit is 32B.
        default_blk_stride = 1
        default_rep_stride = 8
        if self.mask_mode == "normal":
            # all elements in src are read even their mask bits are invalid
            if get_dtype_bit_width(self.src.dtype) == '32':
                mask_len = MASK_VALUE_64
            else:
                mask_len = MASK_VALUE_128
        else:
            mask_len = self.mask

        # check src_offset read mem.
        # change offset dtype temporarily, for check overflow
        self.src_offset.dtype = self.src.dtype
        check_read_mem_out_of_bounds(context, src_offset_buffer_size,
                                     mask_len,
                                     self.src_offset, self.repeat_times,
                                     default_blk_stride, default_rep_stride,
                                     mask_mode=self.mask_mode)
        # reset src_offset dtype: int32
        self.src_offset.dtype = "int32"

        return xn_idx

    def create_gpr_x_t(self, context, temp_env, src_addr):
        """create general purpose register x_t

        Parameters
        ----------
        context : the stack context
        temp_env : the temp environment
        src_addr : src tensor addr

        Returns
        -------
        xt_idx
        """
        base_addr = context.evaluate_expr(self.base_addr)
        base_addr_offset = base_addr + src_addr
        repeat = context.evaluate_expr(self.repeat_times)
        dst_rep_stride = context.evaluate_expr(self.dst_rep_stride)
        # default src_rep_stride: 0
        src_rep_stride = 0

        # check repeat_times
        check_integer_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255], "
            "input repeat_times: {}".format(repeat))
        # check strides
        check_vector_stride(None,
                            [dst_rep_stride],
                            None,
                            MAX_REP_STRIDE_SINGLE_BYTE, ["dst"])

        # check base_addr
        # 2**31 - 1 is max base_addr, 31 bit len
        check_integer_in_range(base_addr, range(2 ** 31),
                               "base_addr should be in"
                               " range of [0, 2**31 - 1]. "
                               "input base_addr: {}".format(base_addr))
        # check valid
        src_scope_size = reduce_mul(
            self.src.indice.origin_shape) * get_dtype_size(self.src.dtype)
        TikCheckUtil.check_le(
            base_addr, context.evaluate_expr(src_scope_size),
            "base_addr should be less equal than src tensor's buffer size: {},"
            " input base_add: {}".format(src_scope_size, base_addr))

        xt_idx = temp_env.alloc_register()
        x_t = int(base_addr_offset)
        x_t |= dst_rep_stride << DST_REPEAT_STRIDE_SHIFT_POS
        x_t |= src_rep_stride << SRC_REPEAT_STRIDE_SHIFT_POS
        x_t |= context.evaluate_expr(self.stride_unit) << STRIDE_UNIT_SHIFT_POS
        x_t |= repeat << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)
        return xt_idx


class VReduceAdd(STMT):
    """ VReduceAdd instruction """

    def __init__(self, source_info, mask, dst,  # pylint: disable=R0913
                 src, work_tensor, repeat_times, src_rep_stride):
        super(VReduceAdd, self).__init__(source_info)
        self.mask = mask
        self.dst = dst
        self.src = src
        self.work_tensor = work_tensor
        self.repeat_times = repeat_times
        self.src_rep_stride = src_rep_stride

    def eval_(self, context):
        src_rep_stride = context.evaluate_expr(self.src_rep_stride)
        cur_repeat_times = context.evaluate_expr(self.repeat_times)
        # check params
        check_integer_in_range(
            cur_repeat_times,
            range(MIN_REPEAT_TIMES, MAX_VREDUCE_REPEAT_TIMES),
            "repeat_times should be in the range of [1, {}], "
            "input value is {}".format(
                MAX_VREDUCE_REPEAT_TIMES - 1, cur_repeat_times))
        check_integer_in_range(
            src_rep_stride, range(MAX_REP_STRIDE_DOUBLE_BYTE),
            "src_rep_stride should be in the range of [0, 65535], "
            "input value is %s" % str(src_rep_stride))
        # check address align
        if get_soc_name() in (ASCEND_310, ASCEND_910):
            for tensor, name in zip((self.work_tensor, self.dst, self.src),
                                    ("work_tensor", "dst", "src")):
                tensor_start = context.evaluate_expr(tensor.offset)
                if tensor_start*DTYPE_SIZE[tensor.dtype] % ONE_BLK_SIZE != 0:
                    TikCheckUtil.raise_error(
                        "Address align error, {} is not 32 align".format(name))
        # check overlap, cause other overlaps will be checked in vcadd_eval,
        # the check will not be repeated here
        if self.work_tensor.buffer == self.dst.buffer:
            work_tensor_start = context.evaluate_expr(self.work_tensor.offset)
            work_tensor_stop = context.evaluate_expr(
                cur_repeat_times + self.work_tensor.offset)
            dst_start = context.evaluate_expr(self.dst.offset)
            if max(work_tensor_start, dst_start) < \
                    min(work_tensor_stop, dst_start + 1):
                TikCheckUtil.raise_error("vec_reduce_add work_tensor and dst "
                                         "address overlapping error.")

        if cur_repeat_times == 1:
            _vcadd_eval(context, self.mask, "normal", self.dst, self.src,
                        cur_repeat_times, src_rep_stride)
            return
        _vec_reduce_add_first_add(context, self.mask, self.work_tensor,
                                  self.src, cur_repeat_times, src_rep_stride)

        operator_byte_size = get_bit_len(self.src.dtype) // ONE_BYTE_BIT_LEN
        elements_per_repeat = ONE_REP_BYTE_SIZE // operator_byte_size
        while cur_repeat_times > 1:
            pre_elements = cur_repeat_times
            cur_repeat_times = ceil_div(pre_elements, elements_per_repeat)
            if cur_repeat_times == 1:
                _vcadd_eval(context, pre_elements, "normal", self.dst,
                            self.work_tensor, cur_repeat_times,
                            BLK_NUM_PER_REP)
            else:
                if get_soc_name() in (HI3796CV300ES, ASCEND_610, ASCEND_620):
                    _vcadd_eval(context, pre_elements, "counter",
                                self.work_tensor,
                                self.work_tensor, cur_repeat_times,
                                BLK_NUM_PER_REP)
                else:
                    tmp_mask = pre_elements % elements_per_repeat
                    if tmp_mask == MASK_VALUE_ZERO:
                        _vcadd_eval(context, elements_per_repeat, "normal",
                                    self.work_tensor,
                                    self.work_tensor, cur_repeat_times,
                                    BLK_NUM_PER_REP)
                    else:
                        _vcadd_eval(context, elements_per_repeat, "normal",
                                    self.work_tensor,
                                    self.work_tensor, cur_repeat_times - 1,
                                    BLK_NUM_PER_REP)
                        _vcadd_eval(context, tmp_mask, "normal",
                                    self.work_tensor, self.work_tensor,
                                    1, BLK_NUM_PER_REP,
                                    dst_offset=cur_repeat_times - 1,
                                    src_offset=elements_per_repeat*
                                    (cur_repeat_times - 1))


class VecAllReduce(STMT):  # pylint: disable=R0902
    """
    vector all reduce, group reduce, pair reduce has \
    no difference to the debugger
    """
    def __init__(self, source_info, min_max_func,  # pylint: disable=R0913
                 mask, dst, src, work_tensor,
                 repeat_times, src_rep_stride, cal_index):
        super(VecAllReduce, self).__init__(source_info)
        self.min_max_func = min_max_func
        self.mask = mask
        self.dst = dst
        self.src = src
        self.work_tensor = work_tensor
        self.repeat_times = repeat_times
        self.src_rep_stride = src_rep_stride
        self.cal_index = cal_index

    def _create_gpr_x_t(self, context, temp_env, repeat_times, src_rep_stride):
        """create general purpose register x_t

        Parameters
        ----------
        context : the stack context

        temp_env : the temp environment

        Returns
        -------
        xt_idx
        """
        # check address overlap
        default_nblock = 1
        stride_unit = 0
        mask_value = _eval_mask(self.mask, context)

        if self.src.buffer == self.work_tensor.buffer:
            name = "vec_reduce_min" if self.min_max_func == "vcmin" \
                else "vec_reduce_max"
            check_address_overlapping(
                name, mask_value, self.dst, self.src,
                default_nblock,
                ONE_REP_BYTE_SIZE //
                max(get_bit_len(self.dst.dtype),
                    get_bit_len(self.src.dtype)),
                VREDUCE_PER_REP_OUTPUT,
                ONE_REP_BYTE_SIZE // get_bit_len(self.src.dtype),
                repeat_times, VREDUCE_DEFAULT_SRC_BLK_STRIDE,
                VREDUCE_DEFAULT_SRC_BLK_STRIDE,
                VREDUCE_DEFAULT_DST_REP_STRIDE, src_rep_stride,
                context.evaluate_expr(self.dst.offset),
                context.evaluate_expr(self.src.offset), stride_unit)

        xt_idx = temp_env.alloc_register()
        x_t = VREDUCE_DEFAULT_DST_REP_STRIDE
        x_t |= VREDUCE_DEFAULT_SRC_BLK_STRIDE << SRC_BLOCK_STRIDE_SHIFT_POS
        x_t |= stride_unit << STRIDE_UNIT_SHIFT_POS
        x_t |= repeat_times << REPEAT_SHIFT_POS
        x_t |= src_rep_stride << _SRC_REPEAT_STRIDE_SHIFT_POS
        context.model.write_gpr(xt_idx, x_t)
        return xt_idx

    def _vreduce_eval(self, context, mask,  # pylint: disable=R0913, R0914
                      dst, src, repeat_times, src_rep_stride,
                      dst_offset=0, src_offset=0):
        temp_env = TempEnv()
        set_vector_mask(mask, context,
                        tensor_bit_len=max(get_bit_len(dst.dtype),
                                           get_bit_len(src.dtype)))

        dst_align = get_dtype_size(src.dtype)
        src_align = _vec_template_align(context, src.dtype)
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, dst, dst_align, True, access_mode='w',
            offset=dst_offset)
        xn_idx, _, src_alloc_size, _ = copy_tensor_to_model(
            context, temp_env, src, src_align, True, access_mode='r',
            offset=src_offset)

        param = _ENCODER.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING_V200[src.dtype]
        param.xt = self._create_gpr_x_t(context, temp_env, repeat_times,
                                        src_rep_stride)
        param.xd = xd_idx
        param.xn = xn_idx

        check_read_mem_out_of_bounds(context, src_alloc_size, mask,
                                     src, repeat_times,
                                     VREDUCE_DEFAULT_DST_REP_STRIDE,
                                     src_rep_stride, ori_offset=src_offset)
        instr = _VEC_WHOLE_REDUCE_ENCODER[self.min_max_func](param)
        context.model.step(instr)
        temp_env.check_mem_access(context.model, True)

        context.model.read_memory(dst_addr, dst.scope, dst_ptr,
                                  dst_alloc_size)

    def _first_step(self, context, repeat_times, src_rep_stride):
        # first step
        max_repeat_times = context.evaluate_expr(MAX_REPEAT_TIMES - 1)
        for_range_times = repeat_times // max_repeat_times
        dtype_len = DTYPE_SIZE[self.src.dtype]
        for index in range(0, for_range_times):
            dst_offset = index * max_repeat_times * VREDUCE_PER_REP_OUTPUT
            src_offset = index * max_repeat_times * src_rep_stride *\
                         ONE_BLK_SIZE // dtype_len
            self._vreduce_eval(context, self.mask, self.work_tensor, self.src,
                               max_repeat_times, src_rep_stride, dst_offset,
                               src_offset)

        left_repeat_times = repeat_times % max_repeat_times
        if left_repeat_times > 0:
            dst_offset = for_range_times * max_repeat_times * \
                         VREDUCE_PER_REP_OUTPUT
            src_offset = for_range_times * max_repeat_times * src_rep_stride *\
                         ONE_BLK_SIZE // dtype_len
            self._vreduce_eval(context, self.mask, self.work_tensor, self.src,
                               left_repeat_times, src_rep_stride, dst_offset,
                               src_offset)

    def _second_step(self, context, dst, src, cur_data):
        # second step
        dtype_len = DTYPE_SIZE[src.dtype]
        new_repeat_times = cur_data * dtype_len // ONE_REP_BYTE_SIZE
        if new_repeat_times >= 1:
            new_mask = vreduce_create_mask(ONE_REP_BYTE_SIZE // dtype_len //
                                           VREDUCE_PER_REP_OUTPUT)
            self._vreduce_eval(context, new_mask, dst,
                               src, new_repeat_times,
                               VREDUCE_DEFAULT_SRC_REP_STRIDE)

        left_data = cur_data % (ONE_REP_BYTE_SIZE // dtype_len)

        if left_data > 0:
            new_mask = vreduce_create_mask(left_data // VREDUCE_PER_REP_OUTPUT)
            self._vreduce_eval(context, new_mask, dst, src,
                               VREDUCE_MIN_REPEAT_TIMES,
                               VREDUCE_DEFAULT_SRC_REP_STRIDE,
                               new_repeat_times*VREDUCE_PER_REP_OUTPUT,
                               new_repeat_times*ONE_REP_BYTE_SIZE//dtype_len)
            # have tail, new_repeat_times used to calculate output data num
            new_repeat_times += 1
        return new_repeat_times * VREDUCE_PER_REP_OUTPUT

    def _vreduce_body_cal(self, context,  # pylint: disable=R0913, R0914
                          pre_data_count, element_num_per_rep,
                          pre_start_pos, cur_start_pos, dtype_size):
        ex_body_repeat_times = pre_data_count // element_num_per_rep
        ex_tail_num_count = pre_data_count % element_num_per_rep
        ex_has_tail = ex_tail_num_count != 0
        ex_body_output_count = 0
        if ex_body_repeat_times != 0:
            if get_bit_len(self.src.dtype) == 16:
                body_mask = [int("01" * 32, 2), int("01" * 32, 2)]
            else:
                body_mask = [0, int("01" * 32, 2)]

            self._vreduce_eval(context, body_mask, self.work_tensor,
                               self.work_tensor, ex_body_repeat_times,
                               VREDUCE_DEFAULT_SRC_REP_STRIDE, cur_start_pos,
                               pre_start_pos)
            ex_body_output_count = VREDUCE_PER_REP_OUTPUT * ex_body_repeat_times
        ex_tail_output_count = 0
        if ex_has_tail:
            tail_mask = vreduce_create_mask(ex_tail_num_count //
                                            VREDUCE_PER_REP_OUTPUT)
            self._vreduce_eval(context, tail_mask, self.work_tensor,
                               self.work_tensor, VREDUCE_MIN_REPEAT_TIMES,
                               VREDUCE_DEFAULT_SRC_REP_STRIDE,
                               cur_start_pos + ex_body_output_count,
                               pre_start_pos + element_num_per_rep *
                               ex_body_repeat_times)
            ex_tail_output_count = VREDUCE_PER_REP_OUTPUT
        output_count = ex_body_output_count + ex_tail_output_count
        next_start_pos = align_start_pos(cur_start_pos +
                                         output_count, dtype_size)
        return output_count, next_start_pos

    def _vreduce_cal_index(self, context,  # pylint: disable=R0914
                           repeat_times, rep_stride):
        it1_output_count = VREDUCE_PER_REP_OUTPUT*repeat_times
        dtype_size = DTYPE_SIZE[self.src.dtype]
        element_num_per_rep = ONE_REP_BYTE_SIZE // dtype_size

        # iteration2
        it2_start_pos = align_start_pos(it1_output_count, dtype_size)
        it2_output_count, it3_start_pos = \
            self._vreduce_body_cal(context, it1_output_count,
                                   element_num_per_rep, 0, it2_start_pos,
                                   dtype_size)

        if get_bit_len(self.src.dtype) == 16:
            index_type = "uint16"
        else:
            index_type = "uint32"

        # iteration3
        offset_num_per_rep = ONE_BLK_SIZE // dtype_size*rep_stride

        np_work_tensor = context.get_value(self.work_tensor).buffer
        np_work_tensor = np_work_tensor.reshape(-1)
        work_tensor_offset = context.evaluate_expr(
            self.work_tensor.indice.offset)

        if it2_output_count == VREDUCE_PER_REP_OUTPUT:
            res_value = np_work_tensor[work_tensor_offset + it2_start_pos]
            it2_index = np_work_tensor[work_tensor_offset + it2_start_pos + 1]\
                .view(index_type)
            it1_index = np_work_tensor[work_tensor_offset + it2_index + 1].\
                view(index_type)
            pre_num = offset_num_per_rep*(it2_index // VREDUCE_PER_REP_OUTPUT)
            res_index = np.array([pre_num + it1_index]).view(self.dst.dtype)[0]
        else:
            if it2_output_count > element_num_per_rep:
                it3_output_count, it4_start_pos = \
                    self._vreduce_body_cal(context, it2_output_count,
                                           element_num_per_rep, it2_start_pos,
                                           it3_start_pos, dtype_size)

                # iteration4
                it4_mask = vreduce_create_mask(it3_output_count //
                                               VREDUCE_PER_REP_OUTPUT)

                self._vreduce_eval(context, it4_mask, self.work_tensor,
                                   self.work_tensor, VREDUCE_MIN_REPEAT_TIMES,
                                   VREDUCE_DEFAULT_SRC_REP_STRIDE,
                                   it4_start_pos, it3_start_pos)
                it4_index = np_work_tensor[work_tensor_offset + it4_start_pos +
                                           1].view(index_type)
                res_value = np_work_tensor[work_tensor_offset + it4_start_pos]
                it3_index = np_work_tensor[work_tensor_offset + it3_start_pos +
                                           it4_index + 1].view(index_type)
                it2_index = np_work_tensor[work_tensor_offset + it2_start_pos +
                                           element_num_per_rep *
                                           (it4_index // VREDUCE_PER_REP_OUTPUT)
                                           + it3_index + 1].view(index_type)
                it1_index = np_work_tensor[work_tensor_offset +
                                           element_num_per_rep *
                                           (element_num_per_rep *
                                            (it4_index // VREDUCE_PER_REP_OUTPUT
                                             ) + it3_index) //
                                           VREDUCE_PER_REP_OUTPUT +
                                           it2_index + 1].view(index_type)
                pre_num = offset_num_per_rep * \
                          (element_num_per_rep *
                           (element_num_per_rep *
                            (it4_index // VREDUCE_PER_REP_OUTPUT) +
                            it3_index) // VREDUCE_PER_REP_OUTPUT +
                           it2_index) // VREDUCE_PER_REP_OUTPUT
                res_index = np.array([pre_num + it1_index]).view(
                    self.dst.dtype)[0]
            else:
                it3_mask = vreduce_create_mask(it2_output_count //
                                               VREDUCE_PER_REP_OUTPUT)
                self._vreduce_eval(context, it3_mask, self.work_tensor,
                                   self.work_tensor, VREDUCE_MIN_REPEAT_TIMES,
                                   VREDUCE_DEFAULT_SRC_REP_STRIDE,
                                   it3_start_pos, it2_start_pos)
                it3_index = np_work_tensor[work_tensor_offset + it3_start_pos +
                                           1].view(index_type)
                res_value = np_work_tensor[work_tensor_offset + it3_start_pos]
                it2_index = np_work_tensor[work_tensor_offset + it2_start_pos +
                                           it3_index + 1].view(index_type)
                it1_index = np_work_tensor[work_tensor_offset +
                                           element_num_per_rep *
                                           (it3_index // VREDUCE_PER_REP_OUTPUT)
                                           + it2_index + 1].view(index_type)
                pre_num = offset_num_per_rep*\
                          (element_num_per_rep*
                           (it3_index // VREDUCE_PER_REP_OUTPUT) + it2_index) // \
                          VREDUCE_PER_REP_OUTPUT
                res_index = np.array([pre_num + it1_index]
                                     ).view(self.dst.dtype)[0]

        dst_tensor = context.get_value(self.dst).buffer.reshape(-1)
        dst_offset = context.evaluate_expr(self.dst.indice.offset)
        dst_tensor[dst_offset] = res_value
        dst_tensor[dst_offset + 1] = res_index

    def _third_step(self, context, dst,  # pylint: disable=R0913
                    work_tensor, src, cur_data):
        # third step
        if cur_data > 0:
            new_mask = vreduce_create_mask(cur_data // VREDUCE_PER_REP_OUTPUT)
            self._vreduce_eval(context, new_mask, work_tensor,
                               src, VREDUCE_MIN_REPEAT_TIMES,
                               VREDUCE_DEFAULT_SRC_REP_STRIDE)
            # save result to dst
            np_work_tensor = context.get_value(work_tensor).\
                buffer.reshape(-1)
            np_work_tensor_offset = context.evaluate_expr(work_tensor.
                                                          indice.offset)
            dst_tensor = context.get_value(dst).buffer.reshape(-1)
            dst_offset = context.evaluate_expr(dst.indice.offset)
            dst_tensor[dst_offset] = np_work_tensor[np_work_tensor_offset]

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        repeat_times = context.evaluate_expr(self.repeat_times)
        src_rep_stride = context.evaluate_expr(self.src_rep_stride)
        # check params
        check_vreduce_repeat_times(repeat_times, self.cal_index, self.src.dtype)
        check_integer_in_range(
            src_rep_stride, range(MAX_REP_STRIDE_DOUBLE_BYTE),
            "src_rep_stride should be in the range of [{}, {}], "
            "input src_rep_stride: {}".format(0,
                                              MAX_REP_STRIDE_DOUBLE_BYTE - 1,
                                              src_rep_stride))
        mask_value = _eval_mask(self.mask, context)
        if self.cal_index:
            check_dtype_overflow(self.mask, repeat_times, self.src,
                                 src_rep_stride)

        src_offset = context.evaluate_expr(self.src.offset)
        rep_is_scalar = is_basic_expr(self.repeat_times)
        check_space_overflow(mask_value, self.dst, self.work_tensor, self.src,
                             context.evaluate_expr(self.dst.offset),
                             context.evaluate_expr(self.work_tensor.offset),
                             src_offset, repeat_times, src_rep_stride,
                             self.cal_index, rep_is_scalar)

        self._first_step(context, repeat_times, src_rep_stride)

        if self.cal_index:
            self._vreduce_cal_index(context, repeat_times, src_rep_stride)
        else:
            dtype_len = DTYPE_SIZE[self.src.dtype]
            # second step
            cur_data = self._second_step(
                context, self.work_tensor, self.work_tensor,
                repeat_times*VREDUCE_PER_REP_OUTPUT)
            one_rep_data = ONE_REP_BYTE_SIZE // dtype_len
            # third step
            if cur_data <= one_rep_data:
                self._third_step(context, self.dst, self.work_tensor,
                                 self.work_tensor, cur_data)
                return

            # second second step
            cur_data = self._second_step(context, self.work_tensor,
                                         self.work_tensor, cur_data)
            new_repeat_times = cur_data * dtype_len // ONE_REP_BYTE_SIZE
            # second third step
            if new_repeat_times <= 1:
                self._third_step(context, self.dst, self.work_tensor,
                                 self.work_tensor, cur_data)


class VBI(STMT):  # pylint: disable=R0902
    """
    VBI instruction
    """
    def __init__(self, source_info, mask, dst,  # pylint: disable=R0913
                 src0, src1, src0_offset, dst_blk_stride,
                 vertical_repeat_times, horizontal_repeat_times,
                 repeat_mode, vertical_repeat_offset):
        super(VBI, self).__init__(source_info)
        self.mask = mask
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src0_offset = src0_offset
        self.dst_blk_stride = dst_blk_stride
        self.vertical_repeat_times = vertical_repeat_times
        self.horizontal_repeat_times = horizontal_repeat_times
        self.repeat_mode = repeat_mode
        self.vertical_repeat_offset = vertical_repeat_offset

    def get_vadds_mask(self, context):
        """get mask_len corresponding to self.mask and
        need elements for vadds part as vadds counter mode mask"""
        mask_value = _eval_mask(self.mask, context)
        vertical_repeat_times_value = context.evaluate_expr(
            self.vertical_repeat_times)
        horizontal_repeat_times_value = context.evaluate_expr(
            self.horizontal_repeat_times)
        if isinstance(mask_value, int):
            mask_len = mask_value
        else:
            mask_len, _ = get_mask_len(mask_value)
        valid_block_len = ceil_div(
            mask_len, ONE_BLK_SIZE // DTYPE_SIZE[self.dst.dtype])
        need_src0_block = (vertical_repeat_times_value *
                           horizontal_repeat_times_value - 1) * \
                          BLK_NUM_PER_REP + valid_block_len
        return mask_len, need_src0_block

    def cal_src0_really_address(self, context, temp_env, align):
        """for vbi, need calculate src0 block address according to
        src0 start address and src0_offset

        Parameters
        ----------
        context : the stack context
        temp_env: the temp environment
        align: the address align size
        Returns
        -------
        mask_len
        """
        _, src0_addr, _, _ = copy_tensor_to_model(
            context, temp_env, self.src0, align, True, access_mode='r')
        src0_flatten_idx = get_flatten_idx(self.src0.indice, context)
        src0_start_addr = context.evaluate_expr(
            src0_addr + src0_flatten_idx * DTYPE_SIZE[self.src0.dtype])
        mask_len, valid_src0_offset_num = self.get_vadds_mask(context)
        vadds_actual_eval_(context, temp_env, valid_src0_offset_num,
                           "counter", self.src0_offset,
                           self.src0_offset, src0_start_addr)
        return mask_len

    def check_vbi_overflow_overlap(self, context, mask_len, src0_buffer_size):
        """check vbi instruction whether src0_offset and src1 is overflow,
         check whether src0_offset and src1 is overlap

        Parameters
        ----------
        context: the stack context
        mask_len: length between lowest digit and top effective digit of mask
        src0_buffer_size: src0_offset tensor buffer size
        Returns
        -------
        None
        """
        # check read mem out of bounds
        total_repeat_times = context.evaluate_expr(self.vertical_repeat_times)*\
                             context.evaluate_expr(self.horizontal_repeat_times)
        if isinstance(self.mask, (list, tuple)) and (is_basic_expr(self.mask)):
            src0_offset_valid_mask = BLK_NUM_PER_REP
        else:
            src0_offset_valid_mask = ceil_div(
                mask_len, ONE_BLK_SIZE // DTYPE_SIZE[self.src0.dtype])
        check_read_mem_out_of_bounds(context, src0_buffer_size,
                                     src0_offset_valid_mask, self.src0_offset,
                                     total_repeat_times, 1, 1,
                                     mask_mode="normal")
        # check src1 overflow
        check_vbi_src1_tensor_overflow(self.src1, self.repeat_mode,
                                       total_repeat_times, mask_len,
                                       context.evaluate_expr(self.src1.offset))
        # check overlapping between src0_offset and src1
        check_vbi_overlap(
            self.src0_offset, self.src1, self.repeat_mode, total_repeat_times,
            mask_len, context.evaluate_expr(self.src0_offset.offset),
            context.evaluate_expr(self.src1.offset))

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
        align = _vec_template_align(context, self.dst.dtype)
        mask_len = self.cal_src0_really_address(context, temp_env, align)

        set_vector_mask(self.mask, context, mask_mode="normal",
                        tensor_bit_len=get_bit_len(self.dst.dtype))
        xn_idx, _, src0_buffer_size, _ = copy_tensor_to_model(
            context, temp_env, self.src0_offset, align, True, access_mode='r')
        xm_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src1, align, True, access_mode='r')
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, align, True, access_mode='w')

        param = _ENCODER.new_param()
        param.type = self.gen_param_type()
        param.xt = self.create_gpr_x_t(context, temp_env)
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = xm_idx

        self.check_vbi_overflow_overlap(context, mask_len, src0_buffer_size)

        context.model.step(_ENCODER.gen_vbi(param))
        temp_env.check_mem_access(context.model, True)

        context.model.read_memory(dst_addr, self.dst.scope, dst_ptr,
                                  dst_alloc_size)

    def gen_param_type(self):
        """generate param type encoding"""
        if self.dst.dtype == "float16":
            type_encoding = 0
        else:
            TikCheckUtil.raise_error(
                "instruction vbi not support this dtype of dst")
        return type_encoding

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
        dst_blk_stride = context.evaluate_expr(self.dst_blk_stride)
        vertical_repeat_times = context.evaluate_expr(
            self.vertical_repeat_times)
        horizontal_repeat_times = context.evaluate_expr(
            self.horizontal_repeat_times)
        vertical_repeat_offset = context.evaluate_expr\
            (self.vertical_repeat_offset)
        # check repeat_times
        check_integer_in_range(
            vertical_repeat_times, range(MIN_REPEAT_TIMES, MAX_REPEAT_TIMES),
            "vertical_repeat_times should be in the range of [1, 255], "
            "input vertical_repeat_times: {}".format(vertical_repeat_times))
        check_integer_in_range(
            horizontal_repeat_times, range(MIN_REPEAT_TIMES, MAX_REPEAT_TIMES),
            "horizontal_repeat_times should be in the range of [1, 255], "
            "input horizontal_repeat_times: {}".format(horizontal_repeat_times))
        # check strides
        check_integer_in_range(
            dst_blk_stride, range(MAX_BLK_STRIDE_DOUBLE_BYTE),
            "dst_blk_stride should be in the range of [0, 65535], "
            "input dst_blk_stride: {}".format(dst_blk_stride))
        check_integer_in_range(
            vertical_repeat_offset, range(MAX_BLK_STRIDE_DOUBLE_BYTE),
            "vertical_repeat_offset should be in the range of [0, 65535], "
            "input vertical_repeat_offset: {}".format(vertical_repeat_offset))

        xt_idx = temp_env.alloc_register()
        x_t = 0
        x_t |= horizontal_repeat_times << 56
        x_t |= vertical_repeat_times << 48
        x_t |= vertical_repeat_offset << 32
        x_t |= dst_blk_stride << 16
        x_t += self.repeat_mode

        context.model.write_gpr(xt_idx, x_t)
        return xt_idx
