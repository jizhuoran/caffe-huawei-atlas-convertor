"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_api_util.py
DESC:     util file for tik_api(need import Tensor/Scalar)
CREATED:  2020-01-10 19:02:50
MODIFIED: 2020-01-10 19:02:50
"""
import numpy as np

from te import tvm
from te.platform.cce_params import scope_ubuf
from te.platform.cce_params import scope_smask
from te.platform.cce_params import ASCEND_310
from te.platform.cce_params import ASCEND_910
from te.platform import cce_params
from ..api.tik_tensor import Tensor
from .tik_expr import Expr
from ..api.tik_scalar import Scalar
from ..common.util import check_integer_in_range, check_scalar_dtype, DTYPE_SIZE
from ..common.tik_get_soc_name import get_soc_name
from ..common.common_util import check_vector_stride
from .tik_params import MIN_REPEAT_TIMES, MAX_REPEAT_TIMES, ONE_IR, \
    MASK_MODE_MASK, MASK_COUNTER_MODE_ENABLE_SHIFT_POS, PIPE_S, PIPE_MTE1, \
    VALUE_65504, VALUE_128, VALUE_256, PAD_MASK, PADDING_SHIFT_POS, \
    VSEL_MODE_TENSOR_SCALAR, VSEL_MODE_DOUBLE_TENSOR_MANY_IT, \
    MAX_STRIDE_UNIT, MIN_STRIDE_UNIT, ONE_BLK_SIZE
from .tik_params import MAX_REP_STRIDE_SINGLE_BYTE
from .tik_check_util import TikCheckUtil

def check_tensor_list(tensor, name_list):
    """ check tensor_list type(Tensor), scope(ub), dtype(should be same)

    Parameters
    ----------
    tensor: list of tensor_list
    name_list: list of tensor_list name

    Returns
    -------
    None
    """
    TikCheckUtil.check_equality(
        len(tensor), len(name_list),
        "for fuction check_tensor_list, input list length should be same, input"
        " length of tensor.{}, input length of name_list:{}"
        .format(len(tensor), len(name_list)))
    for tensor_list, name in zip(tensor, name_list):
        for index, tensor_member in enumerate(tensor_list):
            TikCheckUtil.check_type_match(
                tensor_member, Tensor,
                "{}[{}] should be tensor, input type: {}"
                .format(name, index, type(tensor_member)))
            TikCheckUtil.check_equality(
                tensor_member.scope, scope_ubuf,
                "{}[{}] scope should be ub, input scope: {}"
                .format(name, index, tensor_member.scope))
            # dtype should be the same as tensor_list[0]
            TikCheckUtil.check_equality(
                tensor_member.dtype, tensor_list[0].dtype,
                "input tensor_list dtype should be the same, input "
                "%s[0] dtype: %s, input %s[%d] dtype: %s" %
                (name, tensor_list[0].dtype, name, index, tensor_member.dtype))


def check_repeat_times(repeat_times):
    """check repeat_times dtype, range

    Parameters
    ----------
    repeat_times: numbers of iterations of this instruction

    Returns
    -------
    None
    """
    TikCheckUtil.check_type_match(
        repeat_times, (int, Scalar, Expr),
        "repeat_times should be int, Scalar or Expr, input type of repeat_times"
        ": {}".format(type(repeat_times)))
    check_scalar_dtype(repeat_times,
                       "scalar_repeat_time should be a scalar of int/uint")
    check_integer_in_range(
        repeat_times, range(MIN_REPEAT_TIMES, MAX_REPEAT_TIMES),
        "repeat_times should be in the range of [1, 255], "
        "input repeat_times: {}".format(repeat_times))


def set_ctrl_counter_mask(tik_instance):
    """set CTRL[56] as

    Parameters
    ----------
    tik_instance: tik object

    Returns
    -------
    orig_ctrl: original value of CTRL
    """
    # save orig_ctrl
    orig_ctrl = tik_instance.Scalar_(dtype="uint64", name="orig_ctrl")
    tik_instance.emit(
        tvm.call_extern(orig_ctrl.dtype, "reg_set", orig_ctrl.get(),
                        tvm.call_extern(orig_ctrl.dtype, "get_ctrl")),
        ONE_IR)
    # set CTRL[56] as 1, mask mode trans from normal to counter
    with tik_instance.context.freeze():
        ctrl = tik_instance.Scalar_(dtype="uint64", name="ctrl")
        ctrl.set_as(orig_ctrl & MASK_MODE_MASK)
        ctrl.set_as(ctrl | (1 << MASK_COUNTER_MODE_ENABLE_SHIFT_POS))
        tik_instance.emit(
            tvm.call_extern("uint64", "set_ctrl", ctrl.get()), ONE_IR)
    return orig_ctrl


def reset_ctrl_counter_mask(tik_instance, orig_ctrl):
    """check repeat_times dtype, range

    Parameters
    ----------
    tik_instance: tik object

    Returns
    -------
    orig_ctrl: original value of CTRL
    """
    tik_instance.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_S)
    tik_instance.emit(
        tvm.call_extern("uint64", "set_ctrl", orig_ctrl.get()), ONE_IR)


def set_vsel_cmpmask(tik_instance, mode, src0, src1, sel):
    """
    set the mask of vsel compare
    :param tik_instance: tik object
    :param mode: 1 or 2
    :param src0:
    :param src1:
    :param sel:
    :return: None
    """
    with tik_instance.context.freeze():
        if mode == VSEL_MODE_TENSOR_SCALAR:
            temp_tensor_shape = [8]
            temp_buf = tik_instance.Tensor_(src0.dtype, temp_tensor_shape,
                                            name="temp_buf",
                                            scope=cce_params.scope_ubuf,
                                            enable_buffer_reuse=True,
                                            no_reuse_list=[None])
            if not isinstance(src1, Scalar):
                temp_scalar = tik_instance.Scalar_(src0.dtype)
                temp_scalar.set_as(src1)
                temp_buf.set_as(temp_scalar)
            else:
                temp_buf.set_as(src1)
            tik_instance.mov_tensor_to_cmpmask_(temp_buf)
        elif mode == VSEL_MODE_DOUBLE_TENSOR_MANY_IT:
            temp_tensor_shape = [2]
            temp_buf = tik_instance.Tensor_("int64", temp_tensor_shape,
                                            name="temp_buf",
                                            scope=cce_params.scope_ubuf)
            with tik_instance.new_scope():
                tik_instance.scope_attr(cce_params.CCE_AXIS, "coproc_scope",
                                        PIPE_S)
                sel_addr = tik_instance.Scalar_("int64")
                sel_addr.set_as(tvm.expr.Cast(
                    "int64", tvm.call_extern("handle", "",
                                             sel.access_ptr("r"))))
                temp_buf.set_as(sel_addr)
            tik_instance.mov_tensor_to_cmpmask_(temp_buf)


def do_load3d_padding(tik_instance, src, pad_value):
    """do padding for load3dv1, load3dv2, depthwise_conv

    Parameters
    ----------
    tik_instance: tik object
    src: src tensor for load3dv1, load3dv2, depthwise_conv
    pad_value:

    Returns
    -------
    None
    """
    if pad_value is not None:
        if "padding" in tik_instance.global_dict:
            padding = tik_instance.global_dict["padding"]
        else:
            padding = tik_instance.global_scalar(dtype="uint16")
            tik_instance.global_dict["padding"] = padding
        with tik_instance.context.freeze():
            with tik_instance.new_scope():
                if src.dtype == "float16":
                    pad_value = np.float16(pad_value)
                    # pad_value belongs to numpy but pylint can't check it
                    pad_value = pad_value.view(np.uint16)  # pylint: disable=E1121
                    pad_value = int(pad_value)
                    t_padding = tik_instance.Scalar_(dtype="uint16")
                    t_padding.set_as(pad_value)
                else:
                    if src.dtype == "int8":
                        pad_value = np.int8(pad_value)
                    else:
                        pad_value = np.uint8(pad_value)
                    pad_value = int(pad_value)
                    t_padding = tik_instance.Scalar_(dtype="uint16")
                    t_padding.set_as(pad_value & PAD_MASK)
                    t_padding.set_as(
                        t_padding << PADDING_SHIFT_POS | t_padding)
                tik_instance.scope_attr(cce_params.CCE_AXIS, "if_protect",
                                        PIPE_MTE1)
                with tik_instance.if_scope_(padding != t_padding):
                    padding.set_as(t_padding)
                    # one ir is call_extern
                    tik_instance.emit(
                        tvm.call_extern("int64", "set_padding",
                                        padding.get()), ONE_IR)


def check_stride_unit(stride_unit):
    """check param stride_unit

    Parameters
    ----------
    stride_unit : address and offset unit both affect it. default = 0
    core_arch : ai_core architecture

    Returns
    -------
    None
    """
    TikCheckUtil.check_type_match(
        stride_unit, int, "stride_unit should be int, input stride_unit:"
                          " {}".format(type(stride_unit)))
    if get_soc_name() in (ASCEND_310, ASCEND_910):
        TikCheckUtil.check_equality(
            stride_unit, MIN_STRIDE_UNIT,
            "{} only support stride_unit=0, input value "
            "is {}".format(get_soc_name(), stride_unit))
    check_integer_in_range(
        stride_unit, range(MAX_STRIDE_UNIT),
        "stride_unit should be in the range of [0, 3], "
        "input stride_unit: {}".format(stride_unit))


def check_pad_value(src, pad_value):
    """check_pad_value for load3dv1, load3dv2, depthwise_conv

    Parameters
    ----------
    src: src tensor for load3dv1, load3dv2, depthwise_conv
    pad_value:

    Returns
    -------
    None
    """
    if src.dtype == "float16":
        if pad_value < -VALUE_65504 or pad_value > VALUE_65504:
            TikCheckUtil.raise_error(
                "when src_fm dtype is f16, pad_value should be in the range"
                " of [-65504, 65504], input value: {}".format(pad_value))
    else:
        TikCheckUtil.check_type_match(
            pad_value, int,
            "when src_fm dtype is {}, pad_value should be int, input type: {}"
            .format(src.dtype, type(pad_value)))
        if src.dtype == "int8":
            TikCheckUtil.check_in_range(
                pad_value, range(-VALUE_128, VALUE_128),
                "when src_fm dtype is s8, pad_value should be in the range of "
                "[-128, 127], input value: {}".format(pad_value))
        else:
            TikCheckUtil.check_in_range(
                pad_value, range(VALUE_256),
                "when src_fm dtype is u8, pad_value should be in the range of "
                "[0, 255], input value: {}".format(pad_value))


def check_address_align(tensor_list, name_list, align_size=ONE_BLK_SIZE):
    """check tensor start address if aligned to given align_size

    Parameters
    ----------
    tensor_list: tensor list
    name_list: tensor name list
    align_size: align address

    Returns
    -------
    None
    """
    for tensor, name in zip(tensor_list, name_list):
        tensor_start = Expr(tensor.offset).eval_value()
        if tensor_start is not None and \
                tensor_start * DTYPE_SIZE[tensor.dtype] % align_size != 0:
            TikCheckUtil.raise_error(
                "Address align error, {} is not {} Byte align".format(
                    name, align_size))


def check_weight_offset(smask, instr_name, tensor_name):
    """if enable weight offset, check smask tensor

    Parameters
    ----------
    smask: smask tensor
    instr_name: name of instrunction
    tensor_name: name of tensor

    Returns
    -------
    None
    """
    TikCheckUtil.raise_error(
        "%s doesn't support enabling weight_offset yet." % instr_name)
    TikCheckUtil.check_type_match(
        smask, Tensor,
        "When weight_offset if enabled, %s should be Tensor, input "
        "type of smask: %s" % (tensor_name, str(type(smask))))
    TikCheckUtil.check_equality(
        smask.scope, scope_smask,
        "%s scope should be SMASK, input scope: %s" %
        (tensor_name, smask.scope))
    TikCheckUtil.check_equality(
        smask.dtype, "uint16",
        "%s should be uint16, input type: %s" % (tensor_name, smask.dtype))


def check_high_preci_param(dst,
                           src,
                           work_tensor,
                           repeat_times,
                           dst_rep_stride,
                           src_rep_stride):
    """check input params of the high precision version

    Parameters
    ----------
    mask : Effective operation on element, divided into two model:
        Continuous and bit by bit.
    dst : destination operator
    src : source operation
    work_tensor : temporary operation
    repeat_times : Repeated iterations times
    dst_blk_stride : offset of dst operator between different block in
                    one repeat
    src_blk_stride : offset of src operator between different block in
                    one repeat
    dst_rep_stride : offset of dst operator in the same block between
                    two repeats
    src_rep_stride : offset of src operator in the same block between
                    two repeats

    -------
    Returns
    -------
    None
    """
    # check repeat
    check_repeat_times(repeat_times)
    # check strides
    check_vector_stride(None, [dst_rep_stride, src_rep_stride],
                        None, MAX_REP_STRIDE_SINGLE_BYTE, ["dst", "src"])
    # check tensor
    TikCheckUtil.check_type_match(src, Tensor, "src should be tensor")
    TikCheckUtil.check_type_match(dst, Tensor, "dst should be tensor")
    TikCheckUtil.check_type_match(work_tensor, Tensor,
                                  "work_tensor should be tensor")
    TikCheckUtil.check_equality(src.dtype, dst.dtype,
                                "src's dtype must be same with dst's dtype")
    # check scope
    TikCheckUtil.check_equality(src.scope, scope_ubuf,
                                "src's scope must be UB")
    TikCheckUtil.check_equality(dst.scope, scope_ubuf,
                                "dst's scope must be UB")
    TikCheckUtil.check_equality(work_tensor.scope, scope_ubuf,
                                "work_tensor's scope must be UB")
    check_address_align((work_tensor, dst, src),
                        ("work_tensor", "dst", "src"))
