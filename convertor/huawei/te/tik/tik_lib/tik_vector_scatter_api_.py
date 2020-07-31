"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_vecotr_scatter_api.py
DESC:     provide scatter vector instructions
CREATED:  2019-08-12 18:53:42
MODIFIED: 2019-08-12 21:29:18
"""
# disabling:
# W0511: fix-me
# C0302: too-many-lines
# R0913: too-many-arguments
# E1101: no-member
# R0904: too-many-public-methods
# disable it because this file consist of vector scatter instruction
from te import tvm    # pylint: disable=C0302
from te.platform import cce_params
from te.platform.cce_conf import intrinsic_check_support
from te.platform.cce_params import scope_ubuf
from te.platform.cce_params import ASCEND_910
from te.platform.cce_params import ASCEND_310
from te.platform.cce_params import ASCEND_310AIC
from te.platform.cce_params import ASCEND_910AIC
from .. import debug
from ..api.tik_ir_builder import TikIRBuilder
from .tik_util import type_convert, is_immediate_number, dtype_convert, \
    insert_set_deqscale_attr, emit_scatter_instr
from .tik_api_util import set_vsel_cmpmask, check_repeat_times, \
    set_ctrl_counter_mask, reset_ctrl_counter_mask
from ..api.tik_scalar import mask_concat, Scalar
from ..api.tik_tensor import get_addr_list, Tensor
from .tik_expr import Expr
from .tik_params import VA_REG, CMPMASK_VAR, BIT_LEN_16, BIT_LEN_8, \
    MAX_VA_ADDR_NUM, VA0_INDEX, VA1_INDEX, VA2_INDEX, VA3_INDEX, VA4_INDEX, \
    VA5_INDEX, VA6_INDEX, VA7_INDEX, MASK_LEN_CONTINOUS_MODE, \
    MAX_REP_STRIDE_DOUBLE_BYTE, PIPE_V, MAX_VSEL_MODE, VSEL_MODE_TENSOR_SCALAR,\
    VSEL_MODE_DOUBLE_TENSOR_MANY_IT, VSEL_MODE_DOUBLE_TENSOR_ONE_IT, \
    ONE_BLK_SIZE, MIN_REPEAT_TIMES, FOUR_IR, ONE_IR, ONE_REP_BYTE_SIZE, \
    ONE_BYTE_BIT_LEN, BLK_NUM_PER_REP, MASK_VALUE_128, VSEL_BLK_PARALLEL_BIT,\
    INSTR_DTYPE_SUPPORT_STATEMENT
from ..common.util import TikUtil
from ..common.util import DTYPE_SIZE
from ..common.util import check_integer_in_range
from ..common.util import get_mask_len
from ..common.util import get_bit_len
from ..common.util import ceil_div
from ..common.util import reduce_mul
from ..common.util import check_scatter_vector_overflow
from ..common.util import is_basic_expr
from ..common.util import check_scatter_dict_for_overlap
from ..common.common_util import check_vector_stride
from .tik_api_constants import ROUND_MODE_MAP
from .tik_api_constants import VTYPE_T_MAP
from .tik_api_constants import DTYPE_MAP
from .tik_check_util import TikCheckUtil
from .tik_source_info import source_info_decorator
from ..common.tik_get_soc_name import get_soc_name
from ..common.tik_get_soc_name import get_soc_core_type

def _calculate_extent_scatter_vs_func(mask_mode, src0_dtype, mask, args):
    """calculate extent

    Parameters
    ----------
    mask_mode
    mask
    args: repeat_times, dst_rep_stride, src_rep_stride

    Returns
    -------
    extent
    """
    repeat_times, dst_rep_stride, src_rep_stride = args[:3]
    if mask_mode == "normal":
        rep_nums = repeat_times
    else:
        rep_nums = ceil_div(mask * DTYPE_SIZE[src0_dtype],
                            ONE_REP_BYTE_SIZE)
    dst_extent = _calculate_extent(rep_nums, dst_rep_stride)
    src_extent = _calculate_extent(rep_nums, src_rep_stride)

    return [dst_extent, src_extent]


def _calculate_extent(repeat_times, rep_stride):
    """calculate extent

    Parameters
    ----------
    repeat_times : Repeated iterations times
    rep_stride : offset of tensor in the same block between adjacent iterations

    Returns
    -------
    extent
    """
    extent = Expr((Expr(repeat_times - 1)*Expr(rep_stride) + 1)*ONE_BLK_SIZE)
    return extent.get()


def _check_list_number(bit_len, dst_list, mask_len):
    """check whether list number match mask

    Parameters
    ----------
    bit_len: bit len of tensor.dtype
    dst_list: dst tensor list
    mask_len: mask length

    Returns
    -------
    extent
    """
    if bit_len == BIT_LEN_16:
        TikCheckUtil.check_ge(
            len(dst_list)*BIT_LEN_16, mask_len, "list number not match mask")
    else:
        TikCheckUtil.check_ge(
            len(dst_list)*BIT_LEN_8, mask_len, "list number not match mask")


def _check_mask_len(dst_list, mask):
    """transform original mask to length,
       and check whether list number match mask

    Parameters
    ----------
    dst_list: dst tensor list
    mask: Effective operation on element,
          divided into two model: Continuous and bit by bit.

    Returns
    -------
    None
    """
    bit_len = get_bit_len(dst_list[VA0_INDEX].dtype)
    if not isinstance(mask, (tuple, list)):
        mask_list = [mask]
    else:
        mask_list = mask
    # online scalar, don't check
    if not is_basic_expr(mask_list):
        # continous
        if len(mask_list) == MASK_LEN_CONTINOUS_MODE:
            mask_len = mask_list[0]
        # decrete
        else:
            mask_len, _ = get_mask_len(mask_list)
        _check_list_number(bit_len, dst_list, mask_len)


def _check_scatter_src_dst_list(va_list, va1_list, va2_list=None):
    """check type and scope of src tensor and dst tensor

    Parameters
    ----------
    va_list: dst tensor list
    va1_list: src0 tensor list
    va2_list: src1 tensor list

    Returns
    -------
    None
    """
    list_num = len(va_list)
    for index in range(list_num):
        TikCheckUtil.check_type_match(
            va1_list[index], Tensor,
            "src0 list member should be Tensor, input type of src0_list[{}]: "
            "{}".format(index, type(va1_list[index])))
        TikCheckUtil.check_type_match(
            va_list[index], Tensor,
            "dst list member should be Tensor, input type of dst_list[{}]: "
            "{}".format(index, type(va_list[index])))
        TikCheckUtil.check_equality(
            va1_list[index].scope, scope_ubuf,
            "scope should be UB, input src0_list[{}] scope: "
            "{}".format(index, va1_list[index].scope))
        TikCheckUtil.check_equality(
            va_list[index].scope, scope_ubuf,
            "scope should be UB, input dst_list[{}] scope: "
            "{}".format(index, va_list[index].scope))
        TikCheckUtil.check_equality(
            va1_list[index].dtype, va1_list[VA0_INDEX].dtype,
            "src0_list dtype should be same, input src0_list[0].dtype: {}, "
            "input src0_list[{}].dtype:{}".format(va1_list[0].dtype, index,
                                                  va1_list[index].dtype))
        TikCheckUtil.check_equality(
            va_list[index].dtype, va_list[VA0_INDEX].dtype,
            "dst_list dtype should be same, input dst_list[0].dtype: {}, "
            "input dst_list[{}].dtype:{}".format(va_list[0].dtype, index,
                                                 va_list[index].dtype))
    if va2_list is not None:
        for index in range(list_num):
            TikCheckUtil.check_type_match(
                va2_list[index], Tensor,
                "src1 list member should be Tensor, input type of src1_list[{}]"
                ": {}".format(index, type(va2_list[index])))
            TikCheckUtil.check_equality(
                va2_list[index].scope, scope_ubuf,
                "scope should be UB, input src1_list[{}] scope: {}"
                .format(index, va2_list[index].scope))
            TikCheckUtil.check_equality(
                va2_list[index].dtype, va2_list[VA0_INDEX].dtype,
                "src1_list dtype should be same, input src1_list[0].dtype: {}, "
                "input src1_list[{}].dtype:{}".format(va2_list[0].dtype, index,
                                                      va2_list[index].dtype))


def _get_ternary_addr_list(dst_list, src0_list,  # pylint: disable=R0913
                           src1_list, dst_mode,
                           src0_mode, src1_mode, extents=None):
    """for ternary operation instruction, get addr list

    Parameters
    ----------
    dst_list: dst tensor list
    src0_list: src0 tensor list
    src1_list: src1 tensor list
    dst_mode: dst mode , w/r/rw
    src0_mode: src0_mode, w/r/rw
    src1_mode: src1_mode, w/r/rw
    extents: extents list: [dst_extent, src0_extent, src1_extent]

    Returns
    -------
    dst_addr_list: dst addr list
    src0_addr_list: src0 addr list
    src1_addr_list: src1 addr list
    """
    # function's input params is too much, so disable them
    if extents is None:
        dst_addr_list = []
        for index in dst_list:
            get_addr_list(dst_addr_list, index, dst_mode)
        src0_addr_list = []
        for index in src0_list:
            get_addr_list(src0_addr_list, index, src0_mode)
        src1_addr_list = []
        for index in src1_list:
            get_addr_list(src1_addr_list, index, src1_mode)
    else:
        # extents: dst_extent, src0_extent, src1_extent
        dst_addr_list = []
        for index in dst_list:
            get_addr_list(dst_addr_list, index, dst_mode, extent=extents[0])
        src0_addr_list = []
        for index in src0_list:
            get_addr_list(src0_addr_list, index, src0_mode, extent=extents[1])
        src1_addr_list = []
        for index in src1_list:
            get_addr_list(src1_addr_list, index, src1_mode, extent=extents[2])
    return dst_addr_list, src0_addr_list, src1_addr_list


def _get_bin_addr_list(dst_list, src_list, dst_mode, src_mode, extents):
    """for two operation instruction, get addr list

    Parameters
    ----------
    dst_list: dst tensor list
    src_list: src tensor list
    dst_mode: dst mode , w/r/rw
    src_mode: src_mode, w/r/rw
    extents: extents list: [dst_extent, src_extent]

    Returns
    -------
    dst_addr_list: dst addr list
    src_addr_list: src addr list
    """
    if extents is None:
        dst_addr_list = []
        for index in dst_list:
            get_addr_list(dst_addr_list, index, dst_mode)
        src_addr_list = []
        for index in src_list:
            get_addr_list(src_addr_list, index, src_mode)
    else:
        #  extents: dst_extent, src0_extent
        dst_addr_list = []
        for index in dst_list:
            get_addr_list(dst_addr_list, index, dst_mode, extent=extents[0])
        src_addr_list = []
        for index in src_list:
            get_addr_list(src_addr_list, index, src_mode, extent=extents[1])
    return dst_addr_list, src_addr_list


def _get_vec_bin_dst_addr_mode(name):
    """for scatter_vector_binary_ternary_elewise_func, get dst_addr_mode

    Parameters
    ----------
    name: name of instruction

    Returns
    -------
    dst_acc: dst_addr_mode, rw/w
    """
    if name in ["scatter_vmadd", "scatter_vmaddrelu", "scatter_vmla"]:
        dst_acc = "rw"
    else:
        dst_acc = "w"
    return dst_acc


def _check_vcon_mask(dst_list, src_list, mask):
    """transform original mask to length,
       and check whether list number match mask

    Parameters
    ----------
    dst_list: dst tensor list
    src_list: src tensor list
    mask: Effective operation on element,
          divided into two model: Continuous and bit by bit.

    Returns
    -------
    None
    """
    bit_len = max(get_bit_len(dst_list[VA0_INDEX].dtype),
                  get_bit_len(src_list[VA0_INDEX].dtype))
    mask_list = TikUtil.to_list(mask)
    # online scalar, don't check
    if not is_basic_expr(mask_list):
        # continous
        if len(mask_list) == MASK_LEN_CONTINOUS_MODE:
            mask_len = mask_list[0]
        # decrete
        else:
            mask_len, _ = get_mask_len(mask_list)
        _check_list_number(bit_len, dst_list, mask_len)


def _check_vconv_mode_deqscale(round_mode_map, round_mode, deqscale):
    """check round_mode and deqscale of vconv

    Parameters
    ----------
    round_mode_map: map of round mode
    round_mode: 'none', 'round', 'floor', 'ceil'/'ceilling', 'away-zero',
                'to-zero', 'odd'
    deqscale: dequant scale, default none

    Returns
    -------
    None
    """
    round_mode_key = [k for k, _ in round_mode_map.items()]
    TikCheckUtil.check_in_range(round_mode, round_mode_key,
                                "round_mode %s is not support" % (round_mode))
    if deqscale is not None:
        TikCheckUtil.check_type_match(
            deqscale, (int, float, Scalar, Expr),
            "deqscale should be int, float, Scalar or Expr")
        if not is_immediate_number(deqscale):
            TikCheckUtil.check_equality(
                deqscale.dtype, "float16", "deqscale's dtype should be float16")


def _get_vsel_bit_len(mode, src1, dst_list, src0_list):
    """get max bit length of src tensor and dst tensor

    Parameters
    ----------
    mode: scatter_vsel mode
    src1: src1 tensor_list/scalar/Imm
    dst_list: dst tensor list
    src0_list: src0 tensor list

    Returns
    -------
    bit_len: max bit length
    """
    if mode == VSEL_MODE_TENSOR_SCALAR:
        TikCheckUtil.check_type_match(
            src1, (int, float, Scalar), "src1 should be int, float or Scalar, "
                                        "input type is {}".format(type(src1)))
        bit_len = max(get_bit_len(dst_list[VA0_INDEX].dtype),
                      get_bit_len(src0_list[VA0_INDEX].dtype))
    else:
        TikCheckUtil.check_type_match(
            src1, (list, tuple), "src1 should be list or tuple,"
                                 " input type is {}".format(type(src1)))

        bit_len = max(get_bit_len(dst_list[VA0_INDEX].dtype),
                      get_bit_len(src0_list[VA0_INDEX].dtype),
                      get_bit_len(src1[VA0_INDEX].dtype))
    return bit_len


def _gen_vsel_mask_len(mode, mask, repeat_times,  # pylint: disable=R0913
                       sel, src1, dst_list, src0_list):
    """transform original mask to mask length and check it

    Parameters
    ----------
    mode: scatter_vsel mode
    sel: select mask
    src1: src1 tensor_list/scalar/Imm
    dst_list: dst tensor list
    src0_list: src0 tensor list

    Returns
    -------
    None
    """
    # function's input params is too much, so disable them
    bit_len = _get_vsel_bit_len(mode, src1, dst_list, src0_list)
    mask_list = TikUtil.to_list(mask)
    # online scalar, don't check
    if is_immediate_number(mask_list):
        # continous
        if len(mask_list) == MASK_LEN_CONTINOUS_MODE:
            mask_len = mask_list[0]
        # decrete
        else:
            mask_len, _ = get_mask_len(mask_list)
        TikCheckUtil.check_ge(
            len(dst_list)*(VSEL_BLK_PARALLEL_BIT // bit_len), mask_len,
            "Please check number of dst_list and src0_list for current mask.")
        if mode == VSEL_MODE_TENSOR_SCALAR:
            sel_tmp = sel[VA0_INDEX]
        elif mode == VSEL_MODE_DOUBLE_TENSOR_MANY_IT:
            sel_tmp = sel
        else:
            return
        if is_basic_expr(repeat_times):
            return
        sel_bit_len = get_bit_len(sel_tmp.dtype)
        extend_offset = (repeat_times - 1)*(VSEL_BLK_PARALLEL_BIT //
                                            bit_len // sel_bit_len) + \
                        ceil_div(mask_len, sel_bit_len)
        if Expr(extend_offset + sel_tmp.offset).eval_value() is not None:
            TikCheckUtil.check_le(
                Expr(extend_offset + sel_tmp.offset).eval_value(),
                reduce_mul(sel_tmp.indice.origin_shape),
                "sel tensor overflow, expected elements: {}, "
                "actual elements: {}".format(
                    Expr(extend_offset + sel_tmp.offset).eval_value(),
                    reduce_mul(sel_tmp.indice.origin_shape)))


def _check_vsel_sel(mode, sel):
    # check sel
    if mode == VSEL_MODE_TENSOR_SCALAR:
        TikCheckUtil.check_type_match(
            sel, (list, tuple),
            "sel should be list or tuple, input type is {}".format(type(sel)))
        for index in sel:
            TikCheckUtil.check_type_match(
                index, Tensor, "sel list member should be Tensor, "
                               "input type is {}".format(type(index)))
            TikCheckUtil.check_equality(
                index.scope, scope_ubuf, "sel list member's scope should be UB,"
                                         " not support {}".format(index.scope))
            TikCheckUtil.check_in_range(
                index.dtype, ("uint8", "uint16", "uint32", "uint64"),
                "sel must be unsigned, not support {}"
                " type.".format(index.dtype))
    elif mode == VSEL_MODE_DOUBLE_TENSOR_MANY_IT:
        TikCheckUtil.check_type_match(
            sel, Tensor,
            "sel should be Tensor, input type is {}".format(type(sel)))
        TikCheckUtil.check_equality(
            sel.scope, scope_ubuf, "sel's scope should be UB")
        TikCheckUtil.check_in_range(
            sel.dtype, ("uint8", "uint16", "uint32", "uint64"),
            "sel must be unsigned, not support {} type.".format(sel.dtype))
    else:
        TikCheckUtil.check_is(sel, CMPMASK_VAR, "Please assure sel is cmpmask.")


def _check_vconv_mask_len(dst_list, mask_list):
    """transform original mask to mask length and check it

    Parameters
    ----------
    dst_list: dst tensor list
    mask_list: mask list

    Returns
    -------
    None
    """
    from .tik_params import VCONV_BLK_PARALLEL
    mask_list = TikUtil.to_list(mask_list)
    if not is_basic_expr(mask_list):
        if len(mask_list) == MASK_LEN_CONTINOUS_MODE:
            mask_len = mask_list[0]
        else:
            mask_len, _ = get_mask_len(mask_list)
        TikCheckUtil.check_ge(
            len(dst_list)*VCONV_BLK_PARALLEL, mask_len,
            "list number not match mask")


def _scatter_vsel_extend_list(dst_list, src0_list, mode, src1, sel):
    """extend to 8 elements for scatter_vsel

    Parameters
    ----------
    dst_list: dst tensor list
    src0_list: src0 tensor list
    mode: scatter_vsel mode
    src1: src1 tensor_list/scalar/Imm

    Returns
    -------
    None
    """
    # extend to 8 elements
    for _ in range(MAX_VA_ADDR_NUM - len(dst_list)):
        dst_list.append(dst_list[VA0_INDEX])
        src0_list.append(src0_list[VA0_INDEX])
        if mode in (VSEL_MODE_DOUBLE_TENSOR_ONE_IT,
                    VSEL_MODE_DOUBLE_TENSOR_MANY_IT):
            src1.append(src1[VA0_INDEX])
    if mode == VSEL_MODE_TENSOR_SCALAR:
        for _ in range(MAX_VA_ADDR_NUM - len(sel)):
            sel.append(sel[VA0_INDEX])


def _scatter_vec_single_extend_list(dst_list, src_list):
    """extend to 8 elements, for scatter vector single src/dst instruction

    Parameters
    ----------
    dst_list: dst tensor list
    src_list: src tensor list

    Returns
    -------
    None
    """
    for _ in range(MAX_VA_ADDR_NUM - len(dst_list)):
        dst_list.append(dst_list[VA0_INDEX])
        src_list.append(src_list[VA0_INDEX])


def _scatter_vconv_va_reg_set(tik_instance, op_addr_list):
    """set VA_REG_SET for scatter_vconv

    Parameters
    ----------
    op_addr_list : operation address list

    Returns
    -------
    intrin_block, total_ir_num
    """
    intrin_block = tvm.make.Evaluate(0)
    tik_instance.source_info.set_node_loc(intrin_block)
    total_ir_num = 1
    for i, addr_list in enumerate(op_addr_list):
        stmt_setva = tvm.make.Evaluate(
            tvm.call_extern("uint64", "VA_reg_set",
                            VA_REG[i + VA6_INDEX], *addr_list))
        tik_instance.source_info.set_node_loc(stmt_setva)
        intrin_block = tvm.make.Block(intrin_block, stmt_setva)
        tik_instance.source_info.set_node_loc(intrin_block)
        total_ir_num += 1
    return intrin_block, total_ir_num


def _set_va_reg_set(tik_instance, op_addr_list):
    """set VA_REG_SET

    Parameters
    ----------
    op_addr_list : operation address list

    Returns
    -------
    intrin_block, total_ir_num
    """
    intrin_block = tvm.make.Evaluate(0)
    tik_instance.source_info.set_node_loc(intrin_block)
    total_ir_num = ONE_IR
    for i, addr_list in enumerate(op_addr_list):
        stmt_setva = tvm.make.Evaluate(
            tvm.call_extern("uint64", "VA_reg_set", VA_REG[i], *addr_list))
        tik_instance.source_info.set_node_loc(stmt_setva)
        intrin_block = tvm.make.Block(intrin_block, stmt_setva)
        tik_instance.source_info.set_node_loc(intrin_block)
        total_ir_num += ONE_IR
    return intrin_block, total_ir_num


def _check_scatter_mask_len(bit_len, mask, mask_mode="normal"):
    """check mask len for scatter address overlap"""
    if not isinstance(mask, (list, tuple)):
        mask = [mask]
    for value in mask:
        if Expr(value).eval_value() is None:
            return False
    if len(mask) == 1 or mask_mode == "counter":
        mask_len = mask[0]
    else:
        mask_len, _ = get_mask_len(mask)

    mask_len = Expr(mask_len).eval_value()
    if isinstance(mask_len, int):
        if mask_len % (ONE_BYTE_BIT_LEN*ONE_REP_BYTE_SIZE // bit_len) == 0:
            return True
        if bit_len == BIT_LEN_8 and mask_len % MASK_VALUE_128 == 0:
            return True
    return False


def _get_block_begin_end(tensor, blk_len,  # pylint: disable=R0913
                         valid_num_per_block,
                         rep_stride, time, store_high_half=False):
    """get tensor a block's begin and end of each repeat"""
    # function's input params is too much, so disable them
    if blk_len != valid_num_per_block and store_high_half:
        begin = Expr((tensor.offset + time*rep_stride *
                      blk_len)*DTYPE_SIZE[tensor.dtype]).eval_value()
        end = Expr(begin + blk_len*DTYPE_SIZE[tensor.dtype]).eval_value()
    else:
        begin = Expr((tensor.offset + time*rep_stride *
                      blk_len)*DTYPE_SIZE[tensor.dtype]).eval_value()
        end = Expr(begin +
                   valid_num_per_block*DTYPE_SIZE[tensor.dtype]).eval_value()
    return begin, end


def _get_src_dst_buffer_dict(dst_list,  # pylint: disable=R0913
                             src_list, dst_rep_stride,
                             src_rep_stride, dst_dict, src_dict,
                             valid_num_per_block, time=1,
                             repeat_times=1, store_high_half=False,
                             src_store_high_half=None):
    """get src_list, dst_list buffer dict info"""
    # function's input params is too much, so disable them
    # elements of 1 blk
    for dst in dst_list:
        begin, end = _get_block_begin_end(
            dst, ONE_BLK_SIZE // DTYPE_SIZE[dst_list[VA0_INDEX].dtype],
            valid_num_per_block, dst_rep_stride, time, store_high_half)
        if isinstance(end, int):
            if dst.buffer not in dst_dict.keys():
                dst_dict[dst.buffer] = []
                dst_dict[dst.buffer].append([begin, end])
            else:
                dst_dict[dst.buffer].append([begin, end])
    if repeat_times > 1:
        time += 1
    for src in src_list:
        if src_store_high_half is not None:
            begin, end = _get_block_begin_end(
                src, ONE_BLK_SIZE // DTYPE_SIZE[src_list[VA0_INDEX].dtype],
                valid_num_per_block, src_rep_stride, time, src_store_high_half)
        else:
            begin, end = _get_block_begin_end(
                src, ONE_BLK_SIZE // DTYPE_SIZE[src_list[VA0_INDEX].dtype],
                valid_num_per_block, src_rep_stride, time, store_high_half)
        if isinstance(end, int):
            if src.buffer not in src_dict.keys():
                src_dict[src.buffer] = []
                src_dict[src.buffer].append([begin, end])
            else:
                src_dict[src.buffer].append([begin, end])
    return dst_dict, src_dict


def _get_same_bffer_num(dst_list, src_list):
    """get number of same buffer"""
    same_buffer_num = 0
    for src in src_list:
        if Expr(src.offset).eval_value() is not None:
            for dst in dst_list:
                if Expr(dst.offset).eval_value() is not None:
                    if src.buffer == dst.buffer and \
                            Expr(src.offset).eval_value() == \
                            Expr(dst.offset).eval_value():
                        same_buffer_num += 1
    return same_buffer_num


def _check_tensor_offset(dst_list, src_list):
    """check Expr(offset) not None"""
    for dst, src in zip(dst_list, src_list):
        dst_offset = Expr(dst.offset).eval_value()
        src_offset = Expr(src.offset).eval_value()
        if dst_offset is None or src_offset is None:
            return False
    return True


def check_scatter_address_overlap(  # pylint: disable=R0913, R0914
        mask, dst_list, src_list, repeat_times, dst_rep_stride,
        src_rep_stride, mask_mode="normal", store_high_half=False,
        src_store_high_half=None,
        name="scatter instr", msg="dst_list and src_list"):
    """check scatter instr address overlapping"""
    # function's input params is too much, so disable them
    if not _check_scatter_mask_len(max(get_bit_len(dst_list[VA0_INDEX].dtype),
                                       get_bit_len(src_list[VA0_INDEX].dtype)),
                                   mask, mask_mode):
        return
    valid_num_per_block = min(ONE_REP_BYTE_SIZE //
                              get_bit_len(dst_list[VA0_INDEX].dtype),
                              ONE_REP_BYTE_SIZE //
                              get_bit_len(src_list[VA0_INDEX].dtype))
    if dst_list[VA0_INDEX].dtype == src_list[VA0_INDEX].dtype and \
            dst_list[VA0_INDEX].dtype in ("uint8", "int8"):
        valid_num_per_block = BIT_LEN_16
    if mask_mode == "counter":
        repeat_times = ceil_div(mask, valid_num_per_block*BLK_NUM_PER_REP)

    if Expr(repeat_times).eval_value() is None or \
            Expr(repeat_times).eval_value() <= 0:
        return
    # check offset
    if not _check_tensor_offset(dst_list, src_list):
        return

    dst_dict = {}
    src_dict = {}
    if repeat_times == 1:
        if dst_rep_stride == src_rep_stride:
            # check 100% same
            same_buffer_num = _get_same_bffer_num(dst_list, src_list)
            if same_buffer_num == len(src_list)*len(dst_list):
                # each VA block same, allow overlap, check end. return
                return
        dst_dict, src_dict = _get_src_dst_buffer_dict(
            dst_list, src_list, dst_rep_stride,
            src_rep_stride, dst_dict, src_dict, valid_num_per_block,
            store_high_half=store_high_half,
            src_store_high_half=src_store_high_half)
        # check dict for overlap
        check_scatter_dict_for_overlap(src_dict, dst_dict, name, msg)
    else:
        for time in range(repeat_times - 1):
            dst_dict = {}
            src_dict = {}
            dst_dict, src_dict = _get_src_dst_buffer_dict(
                dst_list, src_list, dst_rep_stride,
                src_rep_stride, dst_dict, src_dict, valid_num_per_block,
                time, repeat_times, store_high_half=store_high_half,
                src_store_high_half=src_store_high_half)

            # check dict for overlap
            check_scatter_dict_for_overlap(src_dict, dst_dict, name, msg)


def _get_vsel_sel_extent(sel, mode, time, bit_len):
    """get  vsel sel tensor extent"""
    if mode == VSEL_MODE_TENSOR_SCALAR:
        sel_tmp = sel[VA0_INDEX]
    else:
        sel_tmp = sel
    sel_bit_len = get_bit_len(sel_tmp.dtype)
    sel_num_each_time = VSEL_BLK_PARALLEL_BIT // bit_len // sel_bit_len
    begin = (time + 1) * sel_num_each_time + sel_tmp.offset
    end = begin + sel_num_each_time
    return begin, end, sel_tmp


def _get_vsel_dst_extent_list(dst_list, dst_rep_stride,
                              valid_num_per_block, repeat_times, time=0):
    """get  vsel dst_list extent"""
    # function's input params is too much, so disable them
    # elements of 1 blk
    dst_dict = {}
    dst_blk_len = ONE_BLK_SIZE // DTYPE_SIZE[dst_list[VA0_INDEX].dtype]
    if repeat_times == 1:
        for dst in dst_list:
            begin, end = _get_block_begin_end(
                dst, dst_blk_len, valid_num_per_block,
                dst_rep_stride, repeat_times)
            if isinstance(end, int):
                if dst.buffer not in dst_dict.keys():
                    dst_dict[dst.buffer] = []
                    dst_dict[dst.buffer].append([begin, end])
                else:
                    dst_dict[dst.buffer].append([begin, end])
    else:
        for dst in dst_list:
            begin, end = _get_block_begin_end(
                dst, dst_blk_len, valid_num_per_block,
                dst_rep_stride, time)
            if isinstance(end, int):
                if dst.buffer not in dst_dict.keys():
                    dst_dict[dst.buffer] = []
                    dst_dict[dst.buffer].append([begin, end])
                else:
                    dst_dict[dst.buffer].append([begin, end])
    return dst_dict


def _check_sel_dst_begin_end(dst_dict, sel_begin, sel_end, sel_tmp):
    """check sel dst begin end overlap"""
    sel_begin = Expr(sel_begin).eval_value()
    sel_end = Expr(sel_end).eval_value()
    if isinstance(sel_end, int):
        for buffer in dst_dict.keys():
            if buffer == sel_tmp.buffer:
                for interval_dst in dst_dict[buffer]:
                    if max(sel_begin, interval_dst[0]) < \
                            min(sel_end, interval_dst[1]):
                        TikCheckUtil.raise_error(
                            "scater_vsel dst_list and sel"
                            " not support address overlapping.")


def _check_vsel_dst_sel_overlap(dst_list,  # pylint: disable=R0913
                                sel, src0_list, src1,
                                mask, mode, repeat_times, dst_rep_stride):
    """check vsel dst_list and sel address overlap"""
    # function's input params is too much, so disable them
    bit_len = _get_vsel_bit_len(mode, src1, dst_list, src0_list)
    mask_list = TikUtil.to_list(mask)
    # online scalar, don't check
    if not isinstance(Expr(repeat_times).eval_value(), int):
        return
    if is_immediate_number(mask_list):
        # continous
        if len(mask_list) == MASK_LEN_CONTINOUS_MODE:
            mask_len = mask_list[0]
        # decrete
        else:
            mask_len, _ = get_mask_len(mask_list)

        if isinstance(mask_len, int):
            if mask_len % (ONE_BYTE_BIT_LEN*ONE_REP_BYTE_SIZE // bit_len) != 0:
                return
        # check step
        _vsel_check_overlap(dst_list, sel, mode,
                            repeat_times, dst_rep_stride, bit_len)


def _vsel_check_overlap(dst_list, sel, mode,  # pylint: disable=R0913
                        repeat_times, dst_rep_stride, bit_len):
    """check address overlap"""
    if repeat_times == 1:
        sel_begin, sel_end, sel_tmp = _get_vsel_sel_extent(
            sel, mode, repeat_times-1, bit_len)
        dst_dict = _get_vsel_dst_extent_list(
            dst_list, dst_rep_stride,
            ONE_REP_BYTE_SIZE // bit_len, repeat_times)
        _check_sel_dst_begin_end(
            dst_dict, sel_begin, sel_end, sel_tmp)
    for time in range(repeat_times - 1):
        sel_begin, sel_end, sel_tmp = _get_vsel_sel_extent(
            sel, mode, time, bit_len)
        dst_dict = _get_vsel_dst_extent_list(
            dst_list, dst_rep_stride,
            ONE_REP_BYTE_SIZE // bit_len, repeat_times, time)
        _check_sel_dst_begin_end(dst_dict, sel_begin, sel_end, sel_tmp)


class TikVecScatterApi(TikIRBuilder):  # pylint: disable=R0904
    """Provide scatter instruction"""
    # disable it because here are all scatter api
    def __init__(self):
        super(TikVecScatterApi, self).__init__()

    def _check_three_tensor_list(self, dst_list,  # pylint: disable=R0913
                                 src0_list, src1_list, name):
        """for scatter_vector_binary_ternary_elewise_func, check three tensor
           list

        Parameters
        ----------
        dst_list: dst tensor list
        src0_list: src0 tensor list
        src1_list: src1 tensor list
        instr_map: map of core_arch, core_version and instruction
        name: name of instruction

        Returns
        -------
        dtype_str
        """
        # function's input params is too much, so disable them
        dtype_str = ""
        first_dtype_str = ""
        for dst, src0, src1 in zip(dst_list, src0_list, src1_list):
            dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src0.dtype] \
                        + DTYPE_MAP[src1.dtype]
            if first_dtype_str == "":
                first_dtype_str = dtype_str
            TikCheckUtil.check_equality(
                first_dtype_str, dtype_str, "dtype in list are not same!")
            if name != "scatter_vmla":
                # check dtype
                TikCheckUtil.check_equality(dst.dtype, src0.dtype,
                                            "Intrinsic {}'s src0's dtype"
                                            " should be equal to dst's dtype".
                                            format(name))
                TikCheckUtil.check_equality(dst.dtype, src1.dtype,
                                            "Intrinsic {}'s src1's dtype "
                                            "should be equal to dst's dtype".
                                            format(name))
        if name == "scatter_vmla":
            TikCheckUtil.check_equality(
                intrinsic_check_support("Intrinsic_scatter_vmla", dtype_str),
                True, INSTR_DTYPE_SUPPORT_STATEMENT.format(dtype_str,
                                                           "scatter_vmla"))
        else:
            TikCheckUtil.check_equality(
                intrinsic_check_support("Intrinsic_" + name,
                                        dst_list[0].dtype), True,
                INSTR_DTYPE_SUPPORT_STATEMENT.format(dst_list[0].dtype, name))
        return dtype_str

    def _check_two_tensor_list(self, src0_list,  # pylint: disable=R0913
                               src1_list, name, extra_str=""):
        """check two tensor list

        Parameters
        ----------
        src0_list: src0 tensor list
        src1_list: src1 tensor list
        name: name of instruction
        extra_str

        Returns
        -------
        dtype_str
        """
        # function's input params is too much, so disable them
        dtype_str = ""
        for src0, src1 in zip(src0_list, src1_list):
            dtype_str = DTYPE_MAP[src0.dtype] + extra_str + DTYPE_MAP[
                src1.dtype]
            if name != "vcmp":
                # scatter_vconv
                total_dtype_str = dtype_str + name
                TikCheckUtil.check_equality(
                    intrinsic_check_support(
                        "Intrinsic_" + "scatter_vconv", total_dtype_str),
                    True,
                    INSTR_DTYPE_SUPPORT_STATEMENT.format(total_dtype_str,
                                                         "scatter_vconv"))
            else:
                TikCheckUtil.check_equality(src0.dtype, src1.dtype,
                                            "Intrinsic {}'s src0's "
                                            "dtype should be equal to"
                                            " src1's dtype".
                                            format("scatter_vcmp"))
                TikCheckUtil.check_equality(
                    intrinsic_check_support("Intrinsic_scatter_vcmp",
                                            src0.dtype), True,
                    INSTR_DTYPE_SUPPORT_STATEMENT.format(src0.dtype,
                                                         "scatter_vcmp"))
        return dtype_str

    def scatter_vector_mov(self, mask,  # pylint: disable=R0913
                           dst_list, src_list,
                           repeat_times, dst_rep_stride, src_rep_stride):
        """Data copy in UB

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._scatter_vector_single_elewise_func('scatter_vector_mov', mask,
                                                 dst_list, src_list,
                                                 repeat_times, dst_rep_stride,
                                                 src_rep_stride)

    def scatter_vabs(self, mask,  # pylint: disable=R0913
                     dst_list, src_list, repeat_times,
                     dst_rep_stride, src_rep_stride):
        """Get absolute value by single element.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._scatter_vector_single_elewise_func('scatter_vabs', mask,
                                                 dst_list, src_list,
                                                 repeat_times, dst_rep_stride,
                                                 src_rep_stride)

    def scatter_vexp(self, mask,  # pylint: disable=R0913
                     dst_list, src_list, repeat_times,
                     dst_rep_stride, src_rep_stride):
        """Get natural index by single element.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._scatter_vector_single_elewise_func('scatter_vexp', mask,
                                                 dst_list, src_list,
                                                 repeat_times, dst_rep_stride,
                                                 src_rep_stride)

    def scatter_vrelu(self, mask,  # pylint: disable=R0913
                      dst_list, src_list, repeat_times,
                      dst_rep_stride, src_rep_stride):
        """Do linear rectification by single element.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._scatter_vector_single_elewise_func('scatter_vrelu', mask,
                                                 dst_list, src_list,
                                                 repeat_times, dst_rep_stride,
                                                 src_rep_stride)

    def scatter_vrec(self, mask,  # pylint: disable=R0913
                     dst_list, src_list, repeat_times,
                     dst_rep_stride, src_rep_stride):
        """Get reciprocal by single element.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._scatter_vector_single_elewise_func('scatter_vrec', mask,
                                                 dst_list, src_list,
                                                 repeat_times, dst_rep_stride,
                                                 src_rep_stride)

    def scatter_vln(self, mask,  # pylint: disable=R0913
                    dst_list, src_list, repeat_times,
                    dst_rep_stride, src_rep_stride):
        """Get natural logarithm by single element.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._scatter_vector_single_elewise_func('scatter_vln', mask, dst_list,
                                                 src_list, repeat_times,
                                                 dst_rep_stride,
                                                 src_rep_stride)

    def scatter_vrsqrt(self, mask,  # pylint: disable=R0913
                       dst_list, src_list, repeat_times,
                       dst_rep_stride, src_rep_stride):
        """vsqrt + vrec

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._scatter_vector_single_elewise_func('scatter_vrsqrt', mask,
                                                 dst_list, src_list,
                                                 repeat_times, dst_rep_stride,
                                                 src_rep_stride)

    def scatter_vsqrt(self, mask,  # pylint: disable=R0913
                      dst_list, src_list, repeat_times,
                      dst_rep_stride, src_rep_stride):
        """sqrt by single element.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._scatter_vector_single_elewise_func('scatter_vsqrt', mask,
                                                 dst_list, src_list,
                                                 repeat_times, dst_rep_stride,
                                                 src_rep_stride)

    # VA mode - 1src1dst
    @source_info_decorator(depth=2)
    @debug.scat_vec_single_elewise_dec
    def _scatter_vector_single_elewise_func(self,  # pylint: disable=R0913
                                            name,
                                            mask,
                                            dst_list,
                                            src_list,
                                            repeat_times,
                                            dst_rep_stride,
                                            src_rep_stride):
        """scatter vector single elewise function

        Parameters
        ----------
        name: name of instruction
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        # check repeat
        check_repeat_times(repeat_times)
        # check strides
        check_vector_stride(None, [dst_rep_stride, src_rep_stride],
                            None, MAX_REP_STRIDE_DOUBLE_BYTE,
                            ["dst", "src"], is_scatter=True)
        # check tensor list number
        TikCheckUtil.check_type_match(
            src_list, (tuple, list), "src_list should be tuple or list")
        TikCheckUtil.check_type_match(
            dst_list, (tuple, list), "dst_list should be tuple or list")
        TikCheckUtil.check_equality(
            len(dst_list), len(src_list),
            "dst_list and src_list should contain the same number of tensors")
        TikCheckUtil.check_le(
            len(dst_list), 8, "src_list/dst_list contains 8 tensor at most")
        # check tensor and scope
        _check_scatter_src_dst_list(dst_list, src_list)
        mask_o = mask_concat(self, mask, "normal",
                             tensor_bit_len=max(
                                 get_bit_len(src_list[VA0_INDEX].dtype),
                                 get_bit_len(dst_list[VA0_INDEX].dtype)))
        _check_mask_len(dst_list, mask)
        # check tensor list dtype
        for dst, src in zip(dst_list, src_list):
            dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src.dtype]
            TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                        "Intrinsic {}'s src's dtype"
                                        " should be equal to dst's dtype".
                                        format(name))
            TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_"
                                                                + name,
                                                                dst.dtype),
                                        True,
                                        INSTR_DTYPE_SUPPORT_STATEMENT.
                                        format(dst.dtype, name))
        # check address overlapping, mask: 128 or 64
        if all(isinstance(value, int) for value
               in (dst_rep_stride, src_rep_stride)):
            check_scatter_address_overlap(
                mask, dst_list, src_list, repeat_times,
                dst_rep_stride, src_rep_stride, name=name)
        # check tensor overflow(static)
        check_scatter_vector_overflow([dst_list, src_list],
                                      ["dst_list", "src_list"],
                                      mask, repeat_times,
                                      [dst_rep_stride, src_rep_stride])
        # code gen
        # extend to 8 elements
        for _ in range(MAX_VA_ADDR_NUM - len(dst_list)):
            dst_list.append(dst_list[VA0_INDEX])
            src_list.append(src_list[VA0_INDEX])

        # config VAs
        self._scatter_vector_config_vas(dst_list, src_list,
                                        dtype_str, name, mask_o,
                                        [repeat_times,
                                         dst_rep_stride, src_rep_stride])

    def _scatter_vector_intrin_block(self,  # pylint: disable=R0913
                                     dst_addr_list, src_addr_list,
                                     name, config):
        """get intrin block for scatter vector single elewise function"""
        stmt_setva_0 = tvm.make.Evaluate(
            tvm.call_extern("uint64", "VA_reg_set", VA_REG[VA3_INDEX],
                            *dst_addr_list))
        self.source_info.set_node_loc(stmt_setva_0)
        stmt_setva_1 = tvm.make.Evaluate(
            tvm.call_extern("uint64", "VA_reg_set", VA_REG[VA4_INDEX],
                            *src_addr_list))
        self.source_info.set_node_loc(stmt_setva_1)

        instr = tvm.call_extern("uint64", name, VA_REG[VA3_INDEX],
                                VA_REG[VA4_INDEX], *type_convert(config))

        intrin_block = tvm.make.Block(stmt_setva_0, stmt_setva_1)
        self.source_info.set_node_loc(intrin_block)
        tmp_instr = tvm.make.Evaluate(instr)
        self.source_info.set_node_loc(tmp_instr)
        intrin_block = tvm.make.Block(intrin_block, tmp_instr)
        self.source_info.set_node_loc(intrin_block)
        return intrin_block

    def _scatter_vector_config_vas(self, dst_list,  # pylint: disable=R0913
                                   src_list, dtype_str, name, mask_o, config):
        """config VAs

        Parameters
        ----------
        dst_list : destination tensor list
        src_list : source tensor list
        dtype_str
        name: name of instrunction
        mask: Effective operation on element,
              divided into two model: Continuous and bit by bit.
        config

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        # config offset_list segment_list
        # config: repeat_times, dst_extent, src_extent
        dst_addr_list = []
        for index in dst_list:
            get_addr_list(dst_addr_list, index, "w",
                          extent=_calculate_extent(config[0], config[1]))
        src_addr_list = []
        for index in src_list:
            get_addr_list(src_addr_list, index, "r",
                          extent=_calculate_extent(config[0], config[2]))
        with self.new_scope():
            name = name + "_" + VTYPE_T_MAP[dtype_str]

            intrin_block = self._scatter_vector_intrin_block(
                dst_addr_list, src_addr_list, name, config)

            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))

            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.scope_attr(cce_params.CCE_AXIS, "mem_access_scope",
                            tvm.call_extern("int64", "__dummy__",
                                            "VA_reg_set"))
            self.emit(intrin_block, FOUR_IR)

    def scatter_vadd(self, mask, dst_list, src0_list,  # pylint: disable=R0913
                     src1_list, repeat_times,
                     dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """Do add by single element.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src0_list : source operation
        src1_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src0_rep_stride : offset of src operator in the same block
                          between adjacent iterations
        src1_rep_stride : offset of src operator in the same block
                          between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._scatter_vector_binary_ternary_elewise_func(
            'scatter_vadd', mask, dst_list, src0_list, src1_list, repeat_times,
            dst_rep_stride, src0_rep_stride, src1_rep_stride)

    def scatter_vsub(self, mask, dst_list, src0_list,  # pylint: disable=R0913
                     src1_list, repeat_times,
                     dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """Do minus by single elements.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src0_list : source operation
        src1_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src0_rep_stride : offset of src operator in the same block
                          between adjacent iterations
        src1_rep_stride : offset of src operator in the same block
                          between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._scatter_vector_binary_ternary_elewise_func(
            'scatter_vsub', mask, dst_list, src0_list, src1_list, repeat_times,
            dst_rep_stride, src0_rep_stride, src1_rep_stride)

    def scatter_vmul(self, mask, dst_list, src0_list,  # pylint: disable=R0913
                     src1_list, repeat_times,
                     dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """Get product by single element.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src0_list : source operation
        src1_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src0_rep_stride : offset of src operator in the same block
                          between adjacent iterations
        src1_rep_stride : offset of src operator in the same block
                          between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._scatter_vector_binary_ternary_elewise_func(
            'scatter_vmul', mask, dst_list, src0_list, src1_list, repeat_times,
            dst_rep_stride, src0_rep_stride, src1_rep_stride)

    def scatter_vdiv(self, mask, dst_list, src0_list,  # pylint: disable=R0913
                     src1_list, repeat_times,
                     dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """Get division by single element.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src0_list : source operation
        src1_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src0_rep_stride : offset of src operator in the same block
                          between adjacent iterations
        src1_rep_stride : offset of src operator in the same block
                          between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._scatter_vector_binary_ternary_elewise_func(
            'scatter_vdiv', mask, dst_list, src0_list, src1_list, repeat_times,
            dst_rep_stride, src0_rep_stride, src1_rep_stride)

    def scatter_vmax(self, mask, dst_list, src0_list,  # pylint: disable=R0913
                     src1_list, repeat_times,
                     dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """Get the max value in all elements.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src0_list : source operation
        src1_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src0_rep_stride : offset of src operator in the same block
                          between adjacent iterations
        src1_rep_stride : offset of src operator in the same block
                          between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._scatter_vector_binary_ternary_elewise_func(
            'scatter_vmax', mask, dst_list, src0_list, src1_list, repeat_times,
            dst_rep_stride, src0_rep_stride, src1_rep_stride)

    def scatter_vmin(self, mask, dst_list, src0_list,  # pylint: disable=R0913
                     src1_list, repeat_times,
                     dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """Get the min value in all elements.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src0_list : source operation
        src1_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src0_rep_stride : offset of src operator in the same block
                          between adjacent iterations
        src1_rep_stride : offset of src operator in the same block
                          between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._scatter_vector_binary_ternary_elewise_func(
            'scatter_vmin', mask, dst_list, src0_list, src1_list, repeat_times,
            dst_rep_stride, src0_rep_stride, src1_rep_stride)

    def scatter_vmadd(self, mask, dst_list, src0_list,  # pylint: disable=R0913
                      src1_list, repeat_times,
                      dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """Multiply by element and accumulate. dst = src0*dst + src1

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src0_list : source operation
        src1_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src0_rep_stride : offset of src operator in the same block
                          between adjacent iterations
        src1_rep_stride : offset of src operator in the same block
                          between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._scatter_vector_binary_ternary_elewise_func(
            'scatter_vmadd', mask, dst_list, src0_list, src1_list,
            repeat_times, dst_rep_stride, src0_rep_stride, src1_rep_stride)

    def scatter_vmaddrelu(self, mask, dst_list,  # pylint: disable=R0913
                          src0_list, src1_list,
                          repeat_times, dst_rep_stride, src0_rep_stride,
                          src1_rep_stride):
        """Multiply by element and accumulate and then Linear rectification

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src0_list : source operation
        src1_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src0_rep_stride : offset of src operator in the same block
                          between adjacent iterations
        src1_rep_stride : offset of src operator in the same block
                          between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._scatter_vector_binary_ternary_elewise_func(
            'scatter_vmaddrelu', mask, dst_list, src0_list, src1_list,
            repeat_times, dst_rep_stride, src0_rep_stride, src1_rep_stride)

    def scatter_vmla(self, mask, dst_list,  # pylint: disable=R0913
                     src0_list, src1_list, repeat_times,
                     dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """Multiply by element and accumulate. dst = src0*src1 + dst

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator
        src0_list : source operation
        src1_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src0_rep_stride : offset of src operator in the same block
                          between adjacent iterations
        src1_rep_stride : offset of src operator in the same block
                          between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._scatter_vector_binary_ternary_elewise_func(
            'scatter_vmla', mask, dst_list, src0_list, src1_list, repeat_times,
            dst_rep_stride, src0_rep_stride, src1_rep_stride)

    def _vmcv_check_tensor_list(self, dst_list,  # pylint: disable=R0913
                                src0_list, src1_list):
        """check tensor list of vmcv

        Parameters
        ----------
        dst_list : destination operator
        src0_list : source operation
        src1_list : source operation

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        for dst, src0, src1 in zip(dst_list, src0_list, src1_list):
            dtype_str = DTYPE_MAP[dst.dtype] + \
                        DTYPE_MAP[src0.dtype] + DTYPE_MAP[src1.dtype]
            TikCheckUtil.check_equality(
                intrinsic_check_support("Intrinsic_" + "scatter_vmulconv",
                                        dtype_str), True,
                INSTR_DTYPE_SUPPORT_STATEMENT.format(dtype_str,
                                                     "scatter_vmulconv"))

    def _make_va_vmcv_intrin_block(self, op_addr_list, dst_list, config):
        """for scatter_vmulconv"""
        # no-member is for back-end c++
        intrin_block = tvm.make.Evaluate(0)
        self.source_info.set_node_loc(intrin_block)
        total_ir_num = ONE_IR
        for i, addr_list in enumerate(op_addr_list):
            stmt_setva = tvm.make.Evaluate(tvm.call_extern("uint64",
                                                           "VA_reg_set",
                                                           VA_REG[i],
                                                           *addr_list))
            self.source_info.set_node_loc(stmt_setva)
            intrin_block = tvm.make.Block(intrin_block, stmt_setva)
            self.source_info.set_node_loc(intrin_block)
            total_ir_num += ONE_IR
        instr = tvm.call_extern(dst_list[VA0_INDEX].dtype,
                                "scatter_vmulconv_f162" +
                                DTYPE_MAP[dst_list[VA0_INDEX].dtype],
                                VA_REG[VA0_INDEX], VA_REG[VA1_INDEX],
                                VA_REG[VA2_INDEX],
                                *type_convert(config))
        tmp_instr = tvm.make.Evaluate(instr)
        self.source_info.set_node_loc(tmp_instr)
        intrin_block = tvm.make.Block(intrin_block, tmp_instr)
        self.source_info.set_node_loc(intrin_block)
        return intrin_block, total_ir_num

    def _gen_vmcv_code(self, config, dst_list,  # pylint: disable=R0913
                       src0_list, src1_list, mask_o):
        """
        Parameters
        ----------
        dst_list : destination operator
        src0_list : source operation
        src1_list : source operation
        mask_o: concat mask

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        # extend to 8 elements
        for _ in range(MAX_VA_ADDR_NUM - len(dst_list)):
            dst_list.append(dst_list[VA0_INDEX])
            src0_list.append(src0_list[VA0_INDEX])
            src1_list.append(src0_list[VA0_INDEX])
        # config: repeat_times, dst_extent, src0_extent, src1_extent
        op_addr_list = _get_ternary_addr_list(dst_list, src0_list, src1_list,
                                              "w", "r", "r",
                                              [_calculate_extent(config[0],
                                                                 config[1]),
                                               _calculate_extent(config[0],
                                                                 config[2]),
                                               _calculate_extent(config[0],
                                                                 config[3])])

        # config VAs
        with self.new_scope():
            intrin_block, total_ir_num = self._make_va_vmcv_intrin_block(
                op_addr_list, dst_list, config)
            total_ir_num += ONE_IR
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            emit_scatter_instr(self, total_ir_num, intrin_block)

    @source_info_decorator()
    @debug.scatter_vmulconv_decorator
    def scatter_vmulconv(self, mask, store_high_half,  # pylint: disable=R0913
                         dst_list, src0_list,
                         src1_list, repeat_times, dst_rep_stride,
                         src0_rep_stride, src1_rep_stride):
        """Multiply by element and then convert precision

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        store_high_half: true or false
        dst_list : destination operator
        src0_list : source operation
        src1_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src0_rep_stride : offset of src operator in the same block
                          between adjacent iterations
        src1_rep_stride : offset of src operator in the same block
                          between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        # check repeat times
        check_repeat_times(repeat_times)
        # check rep_stride
        check_vector_stride(None,
                            [dst_rep_stride, src0_rep_stride, src1_rep_stride],
                            None, MAX_REP_STRIDE_DOUBLE_BYTE,
                            ["dst", "src0", "src1"], is_scatter=True)
        TikCheckUtil.check_type_match(
            store_high_half, bool, "store_high_half should be bool type")
        # check tensor list number
        TikCheckUtil.check_type_match(
            dst_list, (tuple, list), "dst_list should be tuple or list")
        TikCheckUtil.check_type_match(
            src0_list, (tuple, list), "src0_list should be tuple or list")
        TikCheckUtil.check_type_match(
            src1_list, (tuple, list), "src1_list should be tuple or list")
        TikCheckUtil.check_equality(
            len(dst_list), len(src0_list),
            "dst_list and src0_list should contain the same number of tensors")
        TikCheckUtil.check_equality(
            len(dst_list), len(src1_list),
            "dst_list and src1_list should contain the same number of tensors")
        TikCheckUtil.check_le(
            len(dst_list), 8,
            "src_list/dst_list should contain 8 tensor at most")
        # check scope
        _check_scatter_src_dst_list(dst_list, src0_list, src1_list)
        _check_vconv_mask_len(dst_list, mask)
        # check tensor list dtype
        self._vmcv_check_tensor_list(dst_list, src0_list, src1_list)
        mask_o = mask_concat(self, mask,
                             tensor_bit_len=max(
                                 get_bit_len(dst_list[VA0_INDEX].dtype),
                                 get_bit_len(src0_list[VA0_INDEX].dtype),
                                 get_bit_len(src1_list[VA0_INDEX].dtype)))
        # check address overlap
        if all(isinstance(value, int) for value
               in (dst_rep_stride, src0_rep_stride)):
            check_scatter_address_overlap(
                mask, dst_list, src0_list, repeat_times,
                dst_rep_stride, src0_rep_stride,
                store_high_half=store_high_half,
                name="scatter_vmulconv", msg="dst_list and src0_list")
        if all(isinstance(value, int) for value
               in (dst_rep_stride, src1_rep_stride)):
            check_scatter_address_overlap(
                mask, dst_list, src1_list, repeat_times,
                dst_rep_stride, src1_rep_stride,
                store_high_half=store_high_half,
                name="scatter_vmulconv", msg="dst_list and src1_list")

        # check tensor overflow(static)
        check_scatter_vector_overflow([src0_list, src1_list, dst_list],
                                      ["src0_list", "src1_list", "dst_list"],
                                      mask, repeat_times,
                                      [src0_rep_stride, src1_rep_stride,
                                       dst_rep_stride],
                                      store_high_half=store_high_half)
        # code gen
        config = [
            repeat_times, dst_rep_stride, src0_rep_stride, src1_rep_stride,
            TikUtil.to_int(store_high_half)]
        self._gen_vmcv_code(config, dst_list, src0_list, src1_list, mask_o)

    def _gen_vec_bin_code(self, name, config,  # pylint: disable=R0913
                          dst_list, src0_list, src1_list,
                          mask_o, dtype_str):
        """for scatter_vector_binary_ternary_elewise_func

        Parameters
        ----------
        name: name of instrunctions
        config
        dst_list : destination operator
        src0_list : source operation
        src1_list : source operation
        mask_o: concat mask
        dtype_str

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        for _ in range(MAX_VA_ADDR_NUM - len(dst_list)):  # extend to 8 elements
            dst_list.append(dst_list[VA0_INDEX])
            src0_list.append(src0_list[VA0_INDEX])
            src1_list.append(src1_list[VA0_INDEX])
        # config: repeat_times, dst_extent, src0_extent, src1_extent
        op_addr_list = _get_ternary_addr_list(dst_list, src0_list, src1_list,
                                              _get_vec_bin_dst_addr_mode(name),
                                              "r", "r",
                                              [_calculate_extent(config[0],
                                                                 config[1]),
                                               _calculate_extent(config[0],
                                                                 config[2]),
                                               _calculate_extent(config[0],
                                                                 config[3])])
        name = name + '_' + VTYPE_T_MAP[dtype_str]
        # config VAs
        with self.new_scope():
            intrin_block, total_ir_num = _set_va_reg_set(self, op_addr_list)
            instr = tvm.call_extern("uint64", name,
                                    VA_REG[VA0_INDEX], VA_REG[VA1_INDEX],
                                    VA_REG[VA2_INDEX],
                                    *type_convert(config))

            tmp_instr = tvm.make.Evaluate(instr)
            self.source_info.set_node_loc(tmp_instr)
            intrin_block = tvm.make.Block(intrin_block, tmp_instr)
            self.source_info.set_node_loc(intrin_block)
            total_ir_num += ONE_IR
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            emit_scatter_instr(self, total_ir_num, intrin_block)

    # VA mode - 2src1dst
    @source_info_decorator(depth=2)
    @debug.scat_vec_bin_ternary_ele_dec
    def _scatter_vector_binary_ternary_elewise_func(  # pylint: disable=R0913
            self, name, mask, dst_list, src0_list, src1_list, repeat_times,
            dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """scatter vector binary ternary elewise function

        Parameters
        ----------
        name: name of instructions
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        store_high_half: true or false
        dst_list : destination tensor list
        src0_list : source tensor list
        src1_list : source tensor list
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src0_rep_stride : offset of src operator in the same block
                          between adjacent iterations
        src1_rep_stride : offset of src operator in the same block
                          between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        # check param
        check_repeat_times(repeat_times)
        # check strides
        check_vector_stride(None, [dst_rep_stride, src0_rep_stride,
                                   src1_rep_stride], None,
                            MAX_REP_STRIDE_DOUBLE_BYTE,
                            ["dst", "src0", "src1"], is_scatter=True)

        # check tensor list number
        TikCheckUtil.check_type_match(
            dst_list, (tuple, list), "dst_list should be tuple or list")
        TikCheckUtil.check_type_match(
            src0_list, (tuple, list), "src0_list should be tuple or list")
        TikCheckUtil.check_type_match(
            src1_list, (tuple, list), "src1_list should be tuple or list")
        TikCheckUtil.check_equality(
            len(dst_list), len(src0_list),
            "dst_list and src_list should contain the same number of tensors")
        TikCheckUtil.check_equality(
            len(dst_list), len(src1_list),
            "dst_list and src_list should contain the same number of tensors")
        TikCheckUtil.check_le(
            len(dst_list), 8, "src_list/dst_list contains 8 tensor at most")
        # check scope and tensor
        _check_scatter_src_dst_list(dst_list, src0_list, src1_list)
        TikCheckUtil.check_type_match(
            mask, (int, Scalar, Expr, list),
            "mask should be int, Scalar, Expr or list")
        _check_mask_len(dst_list, mask)
        # check tensor list dtype
        dtype_str = self._check_three_tensor_list(dst_list, src0_list,
                                                  src1_list, name)
        # mask
        mask_o = mask_concat(self, mask,
                             tensor_bit_len=max(
                                 get_bit_len(dst_list[VA0_INDEX].dtype),
                                 get_bit_len(src0_list[VA0_INDEX].dtype),
                                 get_bit_len(src1_list[VA0_INDEX].dtype)))
        # check address overlap
        if all(isinstance(value, int) for value
               in (dst_rep_stride, src0_rep_stride)):
            check_scatter_address_overlap(
                mask, dst_list, src0_list, repeat_times,
                dst_rep_stride, src0_rep_stride, name=name,
                msg="dst_list and src0_list")
        if all(isinstance(value, int) for value
               in (dst_rep_stride, src1_rep_stride)):
            check_scatter_address_overlap(
                mask, dst_list, src1_list, repeat_times,
                dst_rep_stride, src1_rep_stride, name=name,
                msg="dst_list and src1_list")

        # check tensor overflow(static)
        check_scatter_vector_overflow([dst_list, src0_list, src1_list],
                                      ["dst_list", "src0_list", "src1_list"],
                                      mask, repeat_times,
                                      [dst_rep_stride, src0_rep_stride,
                                       src1_rep_stride])
        # code gen
        config = [
            repeat_times, dst_rep_stride, src0_rep_stride, src1_rep_stride
        ]
        self._gen_vec_bin_code(name, config, dst_list, src0_list, src1_list,
                               mask_o, dtype_str)

    def _check_vec_scalar_tensor_list(self, dst_list,  # pylint: disable=R0913
                                      src_list, scalar, name):
        """check vec scalar tensor list

        Parameters
        ----------
        dst_list : destination tensor list
        src_list : source tensor list
        scalar: source scalar
        name: name of instructions
        scatter_vec_instr_map

        Returns
        -------
        dtype_str
        """
        # function's input params is too much, so disable them
        dtype_str = ""
        for dst, src in zip(dst_list, src_list):
            # online scalar
            if isinstance(scalar, Scalar):
                TikCheckUtil.check_equality(scalar.dtype, src.dtype,
                                            "Intrinsic {}'s src's dtype "
                                            "should be equal to "
                                            "scalar's dtype".format(name))
                dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[
                    src.dtype] + DTYPE_MAP[scalar.dtype]
            else:
                dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src.dtype]*2

            if name == "scatter_vaxpy":
                TikCheckUtil.check_equality(
                    intrinsic_check_support("Intrinsic_" + name,
                                            dtype_str), True,
                    INSTR_DTYPE_SUPPORT_STATEMENT.format(dtype_str, name))
            else:
                TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                            "Intrinsic {}'s src's "
                                            "dtype should be equal to "
                                            "dst's dtype".format(name))
                TikCheckUtil.check_equality(
                    intrinsic_check_support("Intrinsic_" + name, dst.dtype),
                    True, INSTR_DTYPE_SUPPORT_STATEMENT.format(dst.dtype,
                                                               name))
        return dtype_str

    def _make_vector_scalar_func_intrin_block(self,  # pylint: disable=R0913
                                              op_addr_list,
                                              name, scalar, args):
        """make intrin block for vector scalar funtion"""
        intrin_block = tvm.make.Evaluate(0)
        self.source_info.set_node_loc(intrin_block)
        total_ir_num = ONE_IR
        for i, addr_list in enumerate(op_addr_list):
            stmt_setva = tvm.make.Evaluate(
                tvm.call_extern("uint64", "VA_reg_set",
                                VA_REG[i + VA5_INDEX], *addr_list))
            self.source_info.set_node_loc(stmt_setva)
            intrin_block = tvm.make.Block(intrin_block, stmt_setva)
            self.source_info.set_node_loc(intrin_block)
            total_ir_num += ONE_IR
        instr = tvm.call_extern("uint64", name,
                                VA_REG[VA5_INDEX], VA_REG[VA6_INDEX],
                                scalar, *type_convert(args))
        tmp_instr = tvm.make.Evaluate(instr)
        self.source_info.set_node_loc(tmp_instr)
        intrin_block = tvm.make.Block(intrin_block, tmp_instr)
        self.source_info.set_node_loc(intrin_block)
        return intrin_block, total_ir_num

    def _gen_vec_scalar_code(self, op_addr_list,  # pylint: disable=R0913
                             scalar, name, args, mask_o, mask_mode):
        """generate vector scalar funtion code

        Parameters
        ----------
        op_addr_list
        scalar: source scalar
        name: name of instructions
        args: arguments
        mask_o: concat mask

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        # config VAs
        with self.new_scope():
            intrin_block, total_ir_num = \
                self._make_vector_scalar_func_intrin_block(op_addr_list,
                                                           name, scalar, args)

            if mask_mode == "counter":
                orig_ctrl = set_ctrl_counter_mask(self)

            total_ir_num += ONE_IR
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            emit_scatter_instr(self, total_ir_num, intrin_block)

            # reset CTRL SPR as orig_ctrl
            if mask_mode == "counter":
                reset_ctrl_counter_mask(self, orig_ctrl)

    # VA mode - 2src1dst
    @debug.scatter_vector_scalar_decorator
    def _scatter_vector_scalar_elewise_func(  # pylint: disable=R0913
            self, name, mask, store_high_half,
            dst_list, src_list, scalar,
            repeat_times, dst_rep_stride, src_rep_stride, mask_mode="normal"):
        """scatter vector scalar elewise function

        Parameters
        ----------
        name: name of instructions
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        store_high_half :  true or false
        dst_list : destination operator list
        src0_list : source operation list
        scalar : source scalar
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        # check repeat
        check_repeat_times(repeat_times)
        # check strides
        check_vector_stride(None, [dst_rep_stride, src_rep_stride],
                            None, MAX_REP_STRIDE_DOUBLE_BYTE,
                            ["dst", "src"], is_scatter=True)
        TikCheckUtil.check_type_match(
            mask, (int, Scalar, Expr, list),
            "mask should be int, Scalar, Expr or list")
        # check scalar
        TikCheckUtil.check_type_match(
            scalar, (int, float, Expr, Scalar),
            "scalar should be int, float, Expr of Scalar, input type of scalar:"
            " {}".format(type(scalar)))
        # check tensor list number
        TikCheckUtil.check_type_match(
            dst_list, (tuple, list),
            "dst_list should be tuple or list, input type of dst_list:"
            " {}".format(type(dst_list)))
        TikCheckUtil.check_type_match(
            src_list, (tuple, list),
            "src_list should be tuple or list, input type of src_list:"
            " {}".format(type(src_list)))
        TikCheckUtil.check_equality(
            len(dst_list), len(src_list),
            "dst_list and src_list should contain the same number of tensors, "
            "input length of dst_list: {}, input length of src_list: "
            "{}".format(len(dst_list), len(src_list)))
        # check mask mode
        TikCheckUtil.check_type_match(
            mask_mode, str,
            "mask_mode should be str, input type of mask_mode: {}"
            .format(type(mask_mode)))
        TikCheckUtil.check_in_range(
            mask_mode, ("normal", "counter"),
            "mask_mode should be 'normal' or 'counter', input mask_mode: "
            "{}".format(mask_mode))
        if mask_mode == "normal":
            TikCheckUtil.check_in_range(
                len(dst_list), range(1, 9),
                "In mask normal mode, length of src_list/dst_list should be in "
                "the range of [1, 8], input length: {}".format(len(dst_list)))
            _check_mask_len(dst_list, mask)
        else:
            TikCheckUtil.check_equality(
                len(dst_list), 8,
                "In mask counter mode, src_list/dst_list should contain 8"
                " tensor, but input length of tensor_list: "
                "{}".format(len(dst_list)))
        # check scope
        _check_scatter_src_dst_list(dst_list, src_list)
        # check tensor list dtype
        dtype_str = self._check_vec_scalar_tensor_list(dst_list,
                                                       src_list, scalar, name)

        # check address overlap
        if all(isinstance(value, int) for value
               in (dst_rep_stride, src_rep_stride)):
            check_scatter_address_overlap(
                mask, dst_list, src_list, repeat_times,
                dst_rep_stride, src_rep_stride,
                mask_mode=mask_mode,
                store_high_half=store_high_half,
                name=name, msg="dst_list and src0_list")

        # check tensor overflow(static)
        check_scatter_vector_overflow([src_list, dst_list],
                                      ["src_list", "dst_list"], mask,
                                      repeat_times,
                                      [src_rep_stride, dst_rep_stride],
                                      store_high_half=store_high_half,
                                      mask_mode=mask_mode)
        # code gen
        args = [dtype_convert(repeat_times, "int64"), dst_rep_stride,
                src_rep_stride]
        if name == "scatter_vaxpy":
            args.append(store_high_half)

        # extend to 8 elements
        _scatter_vec_single_extend_list(dst_list, src_list)
        # mask
        mask_o = mask_concat(self, mask, mask_mode,
                             tensor_bit_len=max(
                                 get_bit_len(src_list[VA0_INDEX].dtype),
                                 get_bit_len(dst_list[VA0_INDEX].dtype)))
        # args: repeat_times, dst_rep_stride, src_rep_stride
        self._gen_vec_scalar_code(
            _get_bin_addr_list(
                dst_list, src_list, "w", "r",
                _calculate_extent_scatter_vs_func(
                    mask_mode, src_list[VA0_INDEX].dtype, mask, args)),
            dtype_convert(scalar, src_list[VA0_INDEX].dtype),
            name + '_' + VTYPE_T_MAP[dtype_str], args, mask_o, mask_mode)

    @source_info_decorator()
    def scatter_vaxpy(self, mask,  # pylint: disable=R0913
                      store_high_half, dst_list, src0_list, src1,
                      repeat_times, dst_rep_stride, src_rep_stride):
        """multiple each element with a scalar in a vector and then acculate

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        store_high_half :  true or false
        dst_list : destination operator list
        src0_list : source operation list
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        # check store_high_half
        TikCheckUtil.check_type_match(
            store_high_half, bool, "store_high_half should be bool type")
        tmp = int(store_high_half)
        self._scatter_vector_scalar_elewise_func('scatter_vaxpy', mask, tmp,
                                                 dst_list, src0_list, src1,
                                                 repeat_times, dst_rep_stride,
                                                 src_rep_stride)

    @source_info_decorator()
    def scatter_vadds(self, mask,  # pylint: disable=R0913
                      dst_list, src0_list, src1, repeat_times,
                      dst_rep_stride, src_rep_stride, mask_mode="normal"):
        """add each element with a scalar in a vector

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator list
        src0_list : source operation list
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._scatter_vector_scalar_elewise_func('scatter_vadds', mask, False,
                                                 dst_list, src0_list, src1,
                                                 repeat_times, dst_rep_stride,
                                                 src_rep_stride, mask_mode)

    @source_info_decorator()
    def scatter_vmuls(self, mask,  # pylint: disable=R0913
                      dst_list, src0_list, src1, repeat_times,
                      dst_rep_stride, src_rep_stride, mask_mode="normal"):
        """Multiple each element with a scalar in a vector

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst_list : destination operator list
        src0_list : source operation list
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._scatter_vector_scalar_elewise_func('scatter_vmuls', mask, False,
                                                 dst_list, src0_list, src1,
                                                 repeat_times, dst_rep_stride,
                                                 src_rep_stride, mask_mode)

    def _check_vsel_dtype(self, dst_list, src0_list, src1):
        dtype_str = ""
        for idx, (dst, src0) in enumerate(zip(dst_list, src0_list)):
            dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src0.dtype]
            if isinstance(src1, Scalar):
                dtype_str += DTYPE_MAP[src1.dtype]
                TikCheckUtil.check_equality(dst.dtype, src1.dtype,
                                            "Intrinsic {}'s src1's dtype "
                                            "should be equal to dst's dtype".
                                            format("scatter_vsel"))
            elif isinstance(src1, (list, tuple)):
                dtype_str += DTYPE_MAP[src1[idx].dtype]
                TikCheckUtil.check_equality(dst.dtype, src1[idx].dtype,
                                            "Intrinsic {}'s src1's dtype"
                                            " should"
                                            " be equal to dst's dtype"
                                            .format("scatter_vsel"))
            else:
                dtype_str += DTYPE_MAP[src0.dtype]
            TikCheckUtil.check_equality(dst.dtype, src0.dtype,
                                        "Intrinsic {}'s src0's dtype should "
                                        "be"
                                        " equal to dst's dtype".
                                        format("scatter_vsel"))
            TikCheckUtil.check_equality(
                intrinsic_check_support("Intrinsic_" + "scatter_vsel",
                                        dst.dtype), True,
                INSTR_DTYPE_SUPPORT_STATEMENT.format(dst.dtype,
                                                     "scatter_vsel"))
        return dtype_str

    def _gen_vsel_code(self, config,  # pylint: disable=R0913
                       mode, dst_list, src0_list, src1, sel,
                       mask_o, dtype_str):
        """generate vsel code

        Parameters
        ----------
        mode    |   dst |   sel           | src0    |   src1                |
        -------------------------------------------------------------------
        0       |   dst |   CMPMASK_VAR   | src0    |   src1                |
        1       |   dst |   Tensor        | src0    |   src1(Scalar/Imme)   |
        2       |   dst |   Tensor        | src0    |   src1                |

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        # set cmpmask
        set_vsel_cmpmask(self, mode, src0_list[VA0_INDEX], src1, sel)

        # set VAs
        _scatter_vsel_extend_list(dst_list, src0_list, mode, src1, sel)

        # mode == 0 ISA 6.4
        if mode == VSEL_MODE_DOUBLE_TENSOR_ONE_IT:
            # config: repeat_times, dst_extent, src0_extent, src1_extent
            op_addr_list = _get_ternary_addr_list(
                dst_list, src0_list, src1, "w", "r", "r",
                [_calculate_extent(config[0], config[1]),
                 _calculate_extent(config[0], config[2]),
                 _calculate_extent(config[0], config[3])])
        # mode == 2 ISA 7.3
        elif mode == VSEL_MODE_DOUBLE_TENSOR_MANY_IT:
            op_addr_list = _get_ternary_addr_list(dst_list, src0_list, src1,
                                                  "w", "r", "r")
        # mode == 1 ISA 7.3
        else:
            op_addr_list = _get_ternary_addr_list(dst_list, src0_list, sel, "w",
                                                  "r", "r")
        # config VAs
        with self.new_scope():
            intrin_block, total_ir_num = _set_va_reg_set(self, op_addr_list)

            instr = tvm.call_extern(
                "uint64", "scatter_vsel_" + VTYPE_T_MAP[dtype_str],
                VA_REG[VA0_INDEX],
                VA_REG[VA1_INDEX], VA_REG[VA2_INDEX], *type_convert(config))
            tmp_instr = tvm.make.Evaluate(instr)
            self.source_info.set_node_loc(tmp_instr)
            intrin_block = tvm.make.Block(intrin_block, tmp_instr)
            self.source_info.set_node_loc(intrin_block)
            total_ir_num += ONE_IR
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            emit_scatter_instr(self, total_ir_num, intrin_block)

    # VA mode - vsel
    @source_info_decorator()
    @debug.scatter_vsel_decorator
    def scatter_vsel(self,  # pylint: disable=R0913
                     mask,
                     mode,
                     dst_list,
                     sel,
                     src0_list,
                     src1,
                     repeat_times,
                     dst_rep_stride=0,
                     src0_rep_stride=0,
                     src1_sel_rep_stride=0):
        """
        mode    |   dst |   sel           | src0    |   src1                |
        -------------------------------------------------------------------
        0       |   dst |   CMPMASK_VAR   | src0    |   src1                |
        1       |   dst |   Tensor        | src0    |   src1(Scalar/Imme)   |
        2       |   dst |   Tensor        | src0    |   src1                |
        """
        # function's input params is too much, so disable them
        # check mode
        TikCheckUtil.check_type_match(mode, int, "mode should be int type")
        check_integer_in_range(mode, range(MAX_VSEL_MODE),
                               "mode only support 0, 1 and 2")
        if get_soc_name() in (ASCEND_310, ASCEND_910):
            TikCheckUtil.check_equality(
                mode, 0, "v100 doesn't support mode {}.".format(mode))
        # check dst/src0
        TikCheckUtil.check_type_match(
            dst_list, (list, tuple), "dst_list should be list or tuple")
        TikCheckUtil.check_type_match(
            src0_list, (list, tuple), "src0_list should be list or tuple")
        TikCheckUtil.check_equality(
            len(dst_list), len(src0_list),
            "dst_list and src0_list should contain the same number of tensors")
        TikCheckUtil.check_le(
            len(dst_list), 8,
            "src0_list and dst_list should contain 8 tensor at most")
        # check tensor list and scope
        if mode in (VSEL_MODE_DOUBLE_TENSOR_ONE_IT,
                    VSEL_MODE_DOUBLE_TENSOR_MANY_IT):
            TikCheckUtil.check_type_match(
                src1, (list, tuple), "src1 should be list or tuple")
            TikCheckUtil.check_equality(
                len(dst_list), len(src1),
                "dst_list and src1 should contain the same number of tensors")
            TikCheckUtil.check_le(
                len(src1), 8, "src1 should contain 8 tensor at most")
            _check_scatter_src_dst_list(dst_list, src0_list, src1)
        else:
            _check_scatter_src_dst_list(dst_list, src0_list)
        # check sel
        _check_vsel_sel(mode, sel)
        # check dtype
        dtype_str = self._check_vsel_dtype(dst_list, src0_list, src1)
        # check repeat_times
        check_repeat_times(repeat_times)
        if mode == VSEL_MODE_DOUBLE_TENSOR_ONE_IT and \
                isinstance(repeat_times, int):
            TikCheckUtil.check_equality(
                repeat_times, 1, "repeat_times should be 1 when mode is 0, "
                                 "input value is {}".format(repeat_times))
        # check strides
        check_vector_stride(None, [dst_rep_stride, src0_rep_stride,
                                   src1_sel_rep_stride], None,
                            MAX_REP_STRIDE_DOUBLE_BYTE,
                            ["dst", "src0", "src1"], is_scatter=True)
        # gen mask and check
        mask_o = mask_concat(self, mask, "normal",
                             get_bit_len(dst_list[VA0_INDEX].dtype))
        # gen mask_len, check src1 type and check sel overflow
        _gen_vsel_mask_len(mode, mask, repeat_times,
                           sel, src1, dst_list, src0_list)

        # check address overlap
        if all(isinstance(value, int) for value
               in (dst_rep_stride, src0_rep_stride)):
            check_scatter_address_overlap(
                mask, dst_list, src0_list, repeat_times,
                dst_rep_stride, src0_rep_stride, name="scatter_vsel",
                msg="dst_list and src0_list")
        if mode in(VSEL_MODE_TENSOR_SCALAR,
                   VSEL_MODE_DOUBLE_TENSOR_MANY_IT):
            if isinstance(dst_rep_stride, int):
                _check_vsel_dst_sel_overlap(
                    dst_list, sel, src0_list, src1,
                    mask, mode, repeat_times, dst_rep_stride)
        # check tensor overflow(static)
        check_scatter_vector_overflow([dst_list, src0_list],
                                      ["dst_list", "src0_list"],
                                      mask, repeat_times,
                                      [dst_rep_stride, src0_rep_stride])
        if mode in (VSEL_MODE_DOUBLE_TENSOR_ONE_IT,
                    VSEL_MODE_DOUBLE_TENSOR_MANY_IT):
            # check address overlap
            if all(isinstance(value, int) for value
                   in (dst_rep_stride, src1_sel_rep_stride)):
                check_scatter_address_overlap(
                    mask, dst_list, src1, repeat_times,
                    dst_rep_stride, src1_sel_rep_stride, name="scatter_vsel",
                    msg="dst_list and src1")

            check_scatter_vector_overflow([src1], ["src1"], mask, repeat_times,
                                          [src1_sel_rep_stride])
        # gen
        self._gen_vsel_code([repeat_times, dst_rep_stride, src0_rep_stride,
                             src1_sel_rep_stride, mode],
                            mode, dst_list, src0_list, src1,
                            sel, mask_o, dtype_str)

    def _gen_vcon_code(self, src_list,  # pylint: disable=R0913
                       dst_list, dtype_str, config,
                       round_mode_map, round_mode, mask_o):
        """generate vcon funtion code

        Parameters
        ----------
        src_list: src tensor list
        dst_list: dst tensor list
        dtype_str
        config
        round_mode_map: map of round mode
        round_mode:'none', 'round', 'floor', 'ceil'/'ceilling', 'away-zero',
                   'to-zero', 'odd'
        mask_o: concat mask

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        # extend to 8 elements
        _scatter_vec_single_extend_list(dst_list, src_list)
        # config: repeat_times, dst_extent, src_extent
        op_addr_list = _get_bin_addr_list(
            dst_list, src_list, "w", "r",
            [_calculate_extent(config[0], config[1]),
             _calculate_extent(config[0], config[2])])
        # config VAs
        with self.new_scope():
            intrin_block, total_ir_num = _scatter_vconv_va_reg_set(self,
                                                                   op_addr_list)
            instr = tvm.call_extern("uint64", "scatter_vconv_" + dtype_str +
                                    round_mode_map[round_mode],
                                    VA_REG[VA6_INDEX], VA_REG[VA7_INDEX],
                                    *type_convert(config))

            tmp_instr = tvm.make.Evaluate(instr)
            self.source_info.set_node_loc(tmp_instr)
            intrin_block = tvm.make.Block(intrin_block, tmp_instr)
            self.source_info.set_node_loc(intrin_block)
            total_ir_num += 1
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.scope_attr(cce_params.CCE_AXIS, "mem_access_scope",
                            tvm.call_extern("int64", "__dummy__",
                                            "VA_reg_set"))
            total_ir_num += 1
            self.emit(intrin_block, total_ir_num)

    # VA mode - 2src1dst
    @source_info_decorator()
    @debug.scatter_vconv_decorator
    def scatter_vconv(self,  # pylint: disable=R0913
                      mask,
                      round_mode,
                      dst_list,
                      src_list,
                      repeat_times,
                      dst_rep_stride,
                      src_rep_stride,
                      deqscale=None,
                      ldst_high_half=False):
        """Accurate numerical conversion between
           integers and floating point numbers.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        round_mode : 'none', 'round', 'floor', 'ceil'/'ceilling', 'away-zero',
                     'to-zero', 'odd'
        dst_list : destination operator
        src_list : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations
        deqscale : default None
        ldst_high_half : default false

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        # check param
        check_repeat_times(repeat_times)
        # check strides
        check_vector_stride(None, [dst_rep_stride, src_rep_stride],
                            None, MAX_REP_STRIDE_DOUBLE_BYTE,
                            ["dst", "src"], is_scatter=True)
        TikCheckUtil.check_type_match(
            mask, (int, Scalar, Expr, list),
            "mask should be int, Expr, Scalar or list")
        TikCheckUtil.check_type_match(
            ldst_high_half, bool, "ldst_high_half should be bool type")
        _check_vconv_mode_deqscale(ROUND_MODE_MAP, round_mode, deqscale)
        # check tensor list number
        TikCheckUtil.check_type_match(
            src_list, (tuple, list), "src_list should be tuple or list")
        TikCheckUtil.check_type_match(
            dst_list, (tuple, list), "dst_list should be tuple or list")
        TikCheckUtil.check_equality(
            len(dst_list), len(src_list),
            "dst_list and src_list should contain the same number of tensors")
        TikCheckUtil.check_le(
            len(dst_list), 8, "src_list/dst_list should "
                              "contains 8 tensor at most")
        # check scope
        _check_scatter_src_dst_list(dst_list, src_list)
        _check_vcon_mask(dst_list, src_list, mask)
        # check tensor dtype
        dtype_str = self._check_two_tensor_list(src_list, dst_list,
                                                ROUND_MODE_MAP[round_mode],
                                                "2")
        if dtype_str == "s322f16":
            dtype_str = "deq"
            TikCheckUtil.check_type_match(
                deqscale, (float, Scalar), "deqscale should be float or Scalar")

        # check address overlap
        if all(isinstance(value, int) for value
               in (dst_rep_stride, src_rep_stride)):
            check_scatter_address_overlap(
                mask, dst_list, src_list, repeat_times,
                dst_rep_stride, src_rep_stride,
                store_high_half=ldst_high_half,
                name="scatter_vconv", msg="dst_list and src_list")

        # check tensor overflow(static)
        check_scatter_vector_overflow([dst_list, src_list],
                                      ["dst_list", "src_list"],
                                      mask, repeat_times,
                                      [dst_rep_stride, src_rep_stride],
                                      store_high_half=ldst_high_half)
        # deqscale
        if dtype_str == "deq":
            insert_set_deqscale_attr(self, deqscale, dtype_str,
                                     dst_list[VA0_INDEX].dtype)

        # deqscale
        config = [repeat_times, dst_rep_stride, src_rep_stride,
                  TikUtil.to_int(ldst_high_half)]
        # code gen
        self._gen_vcon_code(src_list, dst_list, dtype_str, config,
                            ROUND_MODE_MAP, round_mode,
                            mask_concat(self, mask, tensor_bit_len=max(
                                get_bit_len(dst_list[VA0_INDEX].dtype),
                                get_bit_len(src_list[VA0_INDEX].dtype))))

    @source_info_decorator(depth=2)
    @debug.list_list_elewise_decorator
    def _list_list_elewise_func(self, name, dst_list, src0_list, src1_list):
        TikCheckUtil.\
            check_in_range(get_soc_name() + get_soc_core_type(),
                           (ASCEND_310AIC, ASCEND_910AIC),
                           "%s not support %s." % (get_soc_name() +
                                                   get_soc_core_type(), name))
        # extend to 8 elements
        for index in range(MAX_VA_ADDR_NUM - len(dst_list)):
            dst_list.append(dst_list[VA0_INDEX])
            src0_list.append(src0_list[VA0_INDEX])
            src1_list.append(src1_list[VA0_INDEX])
        dst_addr_list = []
        for index in dst_list:
            if isinstance(index, Tensor):
                dst_addr_list.append(
                    tvm.expr.Cast("uint64", index.access_ptr("w"))*
                    tvm.const(1, "uint64"))
            else:
                dst_addr_list.append(index)
        src0_addr_list = []
        for index in src0_list:
            if isinstance(index, Tensor):
                src0_addr_list.append(
                    tvm.expr.Cast("uint64", index.access_ptr("r"))*
                    tvm.const(1, "uint64"))
            else:
                src0_addr_list.append(index)
        src1_addr_list = []
        for index in src1_list:
            if isinstance(index, Tensor):
                src1_addr_list.append(
                    tvm.expr.Cast("uint64", index.access_ptr("r"))*
                    tvm.const(1, "uint64"))
            else:
                src1_addr_list.append(index)
        self._gen_list_list_fn_code(dst_addr_list, src0_addr_list,
                                    src1_addr_list, name)

    def _gen_list_list_fn_code(self,  # pylint: disable=R0913
                               dst_addr_list, src0_addr_list,
                               src1_addr_list, name):
        # config VAs
        with self.new_scope():
            stmt_setva_0 = tvm.make.Evaluate(
                tvm.call_extern("uint64", "VA_reg_set", VA_REG[VA0_INDEX],
                                *dst_addr_list))
            self.source_info.set_node_loc(stmt_setva_0)
            stmt_setva_1 = tvm.make.Evaluate(
                tvm.call_extern("uint64", "VA_reg_set", VA_REG[VA1_INDEX],
                                *src0_addr_list))
            self.source_info.set_node_loc(stmt_setva_1)
            stmt_setva_2 = tvm.make.Evaluate(
                tvm.call_extern("uint64", "VA_reg_set", VA_REG[VA2_INDEX],
                                *src1_addr_list))
            self.source_info.set_node_loc(stmt_setva_2)
            instr = tvm.call_extern("uint64", name,
                                    VA_REG[VA0_INDEX], VA_REG[VA1_INDEX],
                                    VA_REG[VA2_INDEX])

            intrin_block = tvm.make.Block(stmt_setva_0, stmt_setva_1)
            self.source_info.set_node_loc(intrin_block)
            intrin_block = tvm.make.Block(intrin_block, stmt_setva_2)
            self.source_info.set_node_loc(intrin_block)
            tmp_instr = tvm.make.Evaluate(instr)
            self.source_info.set_node_loc(tmp_instr)
            intrin_block = tvm.make.Block(intrin_block, tmp_instr)
            self.source_info.set_node_loc(intrin_block)
            total_ir_num = FOUR_IR
            emit_scatter_instr(self, total_ir_num, intrin_block)

    def scatter_vmulva(self, dst_list, src0_list, src1_list):
        """multiple two tensor's base address ,dst = src0*src1

        Parameters
        ----------
        dst_list : destination tensor list
        src0_list : source tensor list
        src1_list : source tensor list

        Returns
        -------
        None
        """
        return self._list_list_elewise_func('scatter_vmulva', dst_list,
                                            src0_list, src1_list)

    def scatter_vaddva(self, dst_list, src0_list, src1_list):
        """add two tensor's base address ,dst = src0*src1

        Parameters
        ----------
        dst_list : destination tensor list
        src0_list : source tensor list
        src1_list : source tensor list

        Returns
        -------
        None
        """
        return self._list_list_elewise_func('scatter_vaddva', dst_list,
                                            src0_list, src1_list)

    @source_info_decorator(depth=2)
    @debug.scat_vcmp_elewise_func_dec
    def _scatter_vcmp_elewise_func(self, name, mask, src0_list, src1_list):
        # function's input params is too much, so disable them
        # check strides
        # check tensor list number
        TikCheckUtil.check_type_match(
            src0_list, (tuple, list), "src0_list should be tuple or list")
        TikCheckUtil.check_type_match(
            src1_list, (tuple, list), "src1_list should be tuple or list")
        TikCheckUtil.check_equality(
            len(src0_list), len(src1_list),
            "src0_list and src1_list should contain the same number of tensors")
        TikCheckUtil.check_le(
            len(src0_list), 8, "src_list should contain 8 tensor at most")
        # check scope
        _check_scatter_src_dst_list(src0_list, src1_list)
        # mask
        mask_o = mask_concat(self, mask,
                             tensor_bit_len=max(
                                 get_bit_len(src0_list[VA0_INDEX].dtype),
                                 get_bit_len(src1_list[VA0_INDEX].dtype)))
        _check_mask_len(src0_list, mask)
        # check tensor list dtype
        dtype_str = self._check_two_tensor_list(src0_list, src1_list, "vcmp")
        # check tensor overflow(static)
        # vcmp have default_rep_times as 1, default_rep_stride as 0
        check_scatter_vector_overflow([src0_list, src1_list],
                                      ["src0_list", "src1_list"],
                                      mask, MIN_REPEAT_TIMES, [0, 0])
        # code gen
        # extend to 8 elements
        for _ in range(MAX_VA_ADDR_NUM - len(src0_list)):
            src0_list.append(src0_list[VA0_INDEX])
            src1_list.append(src0_list[VA0_INDEX])

        op_addr_list = _get_bin_addr_list(src0_list, src1_list, "r", "r",
                                          [Expr(32).get(), Expr(32).get()])
        self._gen_scatter_vcmp_code(op_addr_list,
                                    name + "_" + VTYPE_T_MAP[dtype_str],
                                    mask_o)

    def _gen_scatter_vcmp_code(self, op_addr_list, name, mask_o):
        """generate scatter_vcmp code

        Parameters
        ----------
        op_addr_list : operation address list
        name : instruction name
        mask_o : mask value

        Returns
        -------
        None
        """
        # config VAs
        with self.new_scope():
            intrin_block = tvm.make.Evaluate(0)
            self.source_info.set_node_loc(intrin_block)
            total_ir_num = ONE_IR
            for i, addr_list in enumerate(op_addr_list):
                stmt_setva = tvm.make.Evaluate(
                    tvm.call_extern("uint64", "VA_reg_set", VA_REG[i],
                                    *addr_list))
                self.source_info.set_node_loc(stmt_setva)
                intrin_block = tvm.make.Block(intrin_block, stmt_setva)
                self.source_info.set_node_loc(intrin_block)
                total_ir_num += ONE_IR
            # repeat time default 1 for vcmp
            repeat_time = 1
            instr = tvm.call_extern(
                "uint64", name, VA_REG[VA0_INDEX],
                VA_REG[VA1_INDEX], type_convert(repeat_time))

            tmp_instr = tvm.make.Evaluate(instr)
            self.source_info.set_node_loc(tmp_instr)
            intrin_block = tvm.make.Block(intrin_block, tmp_instr)
            self.source_info.set_node_loc(intrin_block)
            total_ir_num += ONE_IR
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            emit_scatter_instr(self, total_ir_num, intrin_block)

    def scatter_vcmp_eq(self, mask, src0_list, src1_list):
        """src0 = src1

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        src0_list : source operation list
        src1_list : source operation list

        Returns
        -------
        CMPMASK
        """
        self._scatter_vcmp_elewise_func('scatter_vcmp_eq', mask, src0_list,
                                        src1_list)
        return CMPMASK_VAR

    def scatter_vcmp_ne(self, mask, src0_list, src1_list):
        """src0 != src1

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        src0_list : source operation list
        src1_list : source operation list

        Returns
        -------
        CMPMASK
        """
        self._scatter_vcmp_elewise_func('scatter_vcmp_ne', mask, src0_list,
                                        src1_list)
        return CMPMASK_VAR

    def scatter_vcmp_lt(self, mask, src0_list, src1_list):
        """src0 < src1

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        src0_list : source operation list
        src1_list : source operation list

        Returns
        -------
        CMPMASK
        """
        self._scatter_vcmp_elewise_func('scatter_vcmp_lt', mask, src0_list,
                                        src1_list)
        return CMPMASK_VAR

    def scatter_vcmp_le(self, mask, src0_list, src1_list):
        """src0 <= src1

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        src0_list : source operation list
        src1_list : source operation list

        Returns
        -------
        CMPMASK
        """
        self._scatter_vcmp_elewise_func('scatter_vcmp_le', mask, src0_list,
                                        src1_list)
        return CMPMASK_VAR

    def scatter_vcmp_gt(self, mask, src0_list, src1_list):
        """src0 > src1

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        src0_list : source operation list
        src1_list : source operation list

        Returns
        -------
        CMPMASK
        """
        self._scatter_vcmp_elewise_func('scatter_vcmp_gt', mask, src0_list,
                                        src1_list)
        return CMPMASK_VAR

    def scatter_vcmp_ge(self, mask, src0_list, src1_list):
        """src0 >= src1

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        src0_list : source operation list
        src1_list : source operation list

        Returns
        -------
        CMPMASK
        """
        self._scatter_vcmp_elewise_func('scatter_vcmp_ge', mask, src0_list,
                                        src1_list)
        return CMPMASK_VAR

    @source_info_decorator()
    def scatter_vmaxs(self, mask,  # pylint: disable=R0913
                      dst_list, src0_list, scalar, repeat_times,
                      dst_rep_stride, src_rep_stride, mask_mode="normal"):
        """scatter vector scalar elewise max function

        Parameters
        ----------
        mask : normal_mode-Effective operation on element,
                           divided into two model: Continuous and bit by bit.
               counter_mode-effective element number
        dst_list : destination operator list
        src0_list : source operation list
        scalar : source scalar
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations
        mask_mode: "normal" - mask normal mode
                   "counter" - mask counter mode

        Returns
        -------
        None
        """
        self._scatter_vector_scalar_elewise_func(
            'scatter_vmaxs', mask, False, dst_list, src0_list, scalar,
            repeat_times, dst_rep_stride, src_rep_stride, mask_mode)

    @source_info_decorator()
    def scatter_vmins(self, mask, dst_list, src0_list, scalar,  # pylint: disable=R0913
                      repeat_times, dst_rep_stride, src_rep_stride, mask_mode="normal"):
        """scatter vector scalar elewise min function

        Parameters
        ----------
        mask : normal_mode-Effective operation on element,
                           divided into two model: Continuous and bit by bit.
               counter_mode-effective element number
        dst_list : destination operator list
        src0_list : source operation list
        scalar : source scalar
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations
        mask_mode: "normal" - mask normal mode
                   "counter" - mask counter mode

        Returns
        -------
        None
        """
        self._scatter_vector_scalar_elewise_func(
            'scatter_vmins', mask, False, dst_list, src0_list, scalar,
            repeat_times, dst_rep_stride, src_rep_stride, mask_mode)
