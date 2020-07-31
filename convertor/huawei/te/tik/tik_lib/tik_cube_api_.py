"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_cube_api_.py
DESC:     provide cube calculation related instructions
CREATED:  2019-08-12 18:53:42
MODIFIED: 2019-08-12 21:29:18
"""
# disabling:
# R0913: too-many-arguments
# R0914: too-many-locals
# W0613: unused-argument

from te import tvm
from te.platform import cce_params
from te.platform.cce_params import scope_cc
from te.platform.cce_params import scope_cbuf
from te.platform.cce_params import scope_cb
from te.platform.cce_conf import intrinsic_check_support
from te.platform.cce_params import AIC
from te.platform.cce_params import HI3796CV300ESAIC
from te.platform.cce_params import ASCEND_310
from te.platform.cce_params import ASCEND_910
from ..api.tik_tensor import Tensor
from ..api.tik_scalar import Scalar
from .tik_expr import Expr
from .. import debug
from ..api.tik_ir_builder import TikIRBuilder
from .tik_util import type_convert
from ..common.util import DTYPE_SIZE, check_integer_in_range, \
    check_scalar_dtype, reduce_mul, ceil_div
from .tik_api_constants import DTYPE_MAP
from .tik_api_util import do_load3d_padding
from .tik_api_util import check_pad_value
from .tik_api_util import check_weight_offset
from .tik_params import MAX_MATRIX
from .tik_params import PIPE_M
from .tik_params import ONE_IR
from .tik_params import BYTE_PER_FRACTAL
from .tik_params import ELE_PER_FRACTAL_EDGE
from .tik_params import PIPE_MTE1
from .tik_params import INSTR_DTYPE_SUPPORT_STATEMENT
from .tik_params import ONE_BLK_SIZE
from ..common.common_util import check_depthwise_conv_params, \
    check_depthwise_conv_l1_w
from ..common.tik_get_soc_name import get_soc_name
from ..common.tik_get_soc_name import get_soc_core_type
from .tik_check_util import TikCheckUtil
from .tik_source_info import source_info_decorator


_ENABLE_BIAS = 1
_MAX_IS_BIAS = 2
# elements per fractal's edge, 16
_ELE_PER_FRACTAL_EDGE = 16
# bits per Byte, 8
_BIT_PER_BYTE = 8


_DEPTHWISE_CONV_C0_MAP = {
    "uint8": 16,
    "int8": 16,
    "float16": 8
}


def _calculate_mmad_extent(matrix_m, matrix_k, # pylint: disable=R0913
                           matrix_n, dst_fm, src_fm, src_filter):
    """calculate mmad extent

    Parameters
    ----------
    matrix_m: row of left_matrix
    matrix_k: col of left_matrix, row of right_matrix
    matrix_n: col of right_matrix
    dst_fm: destination tensor
    src_fm: source tensor_left
    src_filter: source tensor_right

    Returns
    -------
    dst_fm_extent
    src_fm_extent
    src_filter_extent
    """
    # m and n need to be 16 elements aligned
    # k*dtype_size need to be 32 Bytes aligned
    # all variables are element numbers
    m_round = ceil_div(matrix_m, _ELE_PER_FRACTAL_EDGE) * \
              _ELE_PER_FRACTAL_EDGE
    k_round = ceil_div(matrix_k * DTYPE_SIZE[src_filter.dtype],
                       ONE_BLK_SIZE) * ONE_BLK_SIZE // \
              DTYPE_SIZE[src_filter.dtype]
    n_round = ceil_div(matrix_n, _ELE_PER_FRACTAL_EDGE) * \
              _ELE_PER_FRACTAL_EDGE

    # dst_fm_extent is m_round*n_round*dtype_size
    dst_fm_extent = Expr(m_round * n_round * DTYPE_SIZE[dst_fm.dtype]).get()
    # src_fm_extent is m_round*k_round*dtype_size
    src_fm_extent = Expr(m_round * k_round * DTYPE_SIZE[src_fm.dtype]).get()
    # src_filter_extent is k_round*n_round*dtype_size
    src_filter_extent = Expr(k_round * n_round *
                             DTYPE_SIZE[src_filter.dtype]).get()

    return dst_fm_extent, src_fm_extent, src_filter_extent


def _cal_extent_depthwise_conv(dst_fm, src_fm, pad_mode, l1_h, l1_w):
    """calculate depthwise_conv dst_fm/src_fm/src_filter extent

    Parameters
    ----------
    dst_fm: destination tensor
    src_fm: source tensor_left
    pad_mode:   0 - no padding
            1 - two colume on right side
            2 - two colume on left side
            3 - one colume on right&left side
    l1_h: height of src_fm
    l1_w: width of src_fm

    Returns
    -------
    None
    """
    pad_mode = Expr(pad_mode).eval_value()
    if pad_mode is None:
        dst_fm_extent = (reduce_mul(dst_fm.indice.origin_shape) -
                         dst_fm.offset)*DTYPE_SIZE[dst_fm.dtype]
    else:
        # kernel_size is 3*3, dst_fm height h_o should be l1_h minus 2
        h_o = l1_h - 2
        if pad_mode == 0:
            # pad_mode 0 means no padding, dst_fm width w_o should be l1_w minus
            # 2
            w_o = l1_w - 2
        else:
            # pad_mode 1/2/3, w_o should be l1_w up to 16 aligned
            w_o = ceil_div(l1_w, _ELE_PER_FRACTAL_EDGE)
        dst_fm_extent = h_o*w_o*_ELE_PER_FRACTAL_EDGE*DTYPE_SIZE[dst_fm.dtype]
    dst_fm_extent = Expr(dst_fm_extent).get()
    src_fm_extent = l1_h*l1_w*_DEPTHWISE_CONV_C0_MAP[src_fm.dtype]*\
                    DTYPE_SIZE[src_fm.dtype]
    src_fm_extent = Expr(src_fm_extent).get()
    src_ft_extent = BYTE_PER_FRACTAL

    return dst_fm_extent, src_fm_extent, src_ft_extent


def _check_dc_fm_overflow(l1_h, l1_w, src_fm, pad_mode, dst_fm):
    """check whether depthwise_conv src_fm/dst_fm tensor overflow.

    Parameters
    ----------
    l1_h: height of src_fm
    l1_w: width of src_fm
    src_fm: source tensor_left
    pad_mode:   0 - no padding
            1 - two colume on right side
            2 - two colume on left side
            3 - one colume on right&left side
    dst_fm: destination tensor

    src_filter: source tensor_right

    Returns
    -------
    None
    """
    # check src_fm overflow
    src_fm_expected = l1_h*l1_w*_DEPTHWISE_CONV_C0_MAP[src_fm.dtype] + src_fm.offset
    src_fm_actual = reduce_mul(src_fm.indice.origin_shape)
    src_fm_expected = Expr(src_fm_expected).eval_value()
    src_fm_actual = Expr(src_fm_actual).eval_value()
    if src_fm_expected is not None and src_fm_actual is not None:
        TikCheckUtil.check_ge(
            src_fm_actual, src_fm_expected,
            "In depthwise_conv, src_fm tensor overflow, expected elements: {}, "
            "actual elements: {}".format(src_fm_expected, src_fm_actual))
    # check dst_fm overflow
    if Expr(pad_mode).eval_value() is not None:
        if Expr(pad_mode).eval_value() == 0:
            w_o = l1_w - 2
        else:
            w_o = ceil_div(l1_w, ELE_PER_FRACTAL_EDGE)*ELE_PER_FRACTAL_EDGE
        dst_fm_expected = (l1_h - 2)*w_o*ELE_PER_FRACTAL_EDGE + dst_fm.offset
        dst_fm_actual = reduce_mul(dst_fm.indice.origin_shape)
        dst_fm_expected = Expr(dst_fm_expected).eval_value()
        dst_fm_actual = Expr(dst_fm_actual).eval_value()
        if dst_fm_actual is not None and dst_fm_expected is not None:
            TikCheckUtil.check_ge(
                dst_fm_actual, dst_fm_expected,
                "In depthwise_conv, dst_fm tensor overflow, expected elements: "
                "{}, actual elements: {}"
                .format(dst_fm_expected, dst_fm_actual))


def _check_dc_ft_overflow(src_filter):
    """check whether depthwise_conv src_filter tensor overflow.

    Parameters
    ----------
    src_filter: source tensor_right

    Returns
    -------
    None
    """
    src_ft_expected = BYTE_PER_FRACTAL // DTYPE_SIZE[src_filter.dtype] + \
                      src_filter.offset
    src_ft_actual = reduce_mul(src_filter.indice.origin_shape)
    src_ft_expected = Expr(src_ft_expected).eval_value()
    src_ft_actual = Expr(src_ft_actual).eval_value()
    if src_ft_actual is not None and src_ft_expected is not None:
        TikCheckUtil.check_ge(
            src_ft_actual, src_ft_expected,
            "In depthwise_conv, src_filter tensor overflow, expected "
            "elements: {}, actual elements: {}"
            .format(src_ft_expected, src_ft_actual))


def _check_fm_scope(target_fm, target_type, target_fm_name):
    """check whether the input tensor matches target scope.

    Parameters
    ----------
    target_fm: input tensor
    target_type: target scope
    target_fm_name: input tensor name

    Returns
    -------
    None
    """
    target_fm_scope = target_fm.scope.split('.')[-1].lower()
    TikCheckUtil.check_type_match(target_fm, Tensor,
                                  "%s should be tensor" % (target_fm_name))
    TikCheckUtil.check_equality(
        target_fm_scope, target_type,
        "scope of %s should be %s" % (target_fm_name, target_type))


def _check_dc_weight_offset_overflow(weight_offset):
    """check whether depthwise_conv weight_offset tensor overflow.

    Parameters
    ----------
    weight_offset: source tensor which contains weight offset data

    Returns
    -------
    None
    """
    # in 1 instruction, weight_offset only need 16B
    weight_offset_expected = 16 // DTYPE_SIZE[weight_offset.dtype] + \
                             weight_offset.offset
    weight_offset_actual = reduce_mul(weight_offset.indice.origin_shape)
    if Expr(weight_offset_actual).eval_value() is not None \
            and Expr(weight_offset_expected).eval_value() is not None:
        TikCheckUtil.check_ge(
            weight_offset_actual, weight_offset_expected,
            "In depthwise_conv, weight_offset tensor overflow, expected"
            " elements: {}, actual elements: {}"
            .format(weight_offset_expected, weight_offset_actual))


class TikCubeApi(TikIRBuilder):
    """
    Cube Operation Api
    """
    def __init__(self):
        super(TikCubeApi, self).__init__()

    @source_info_decorator()
    @debug.mmad_decorator
    def mmad(self, dst_fm, src_fm, # pylint: disable=R0913, R0914
             src_filter, matrix_m, matrix_k, matrix_n, is_bias, fm_offset=0,
             en_weight_offset=False, smask=None, en_small_channel=False,
             en_small_k=False, en_ssparse=False,
             en_winograd_a=False, en_winograd_b=False):
        """Matrix multiply-add operation.

        Parameters
        ----------
        dst_fm: destination tensor
        src_fm: source tensor_left
        src_filter: source tensor_right
        matrix_m: row of left_matrix
        matrix_k: col of left_matrix, row of right_matrix
        matrix_n: col of right_matrix
        is_bias: switch for mul_add function, 1 is mul_add, 0 is mul
        fm_offset: not support v100
        en_weight_offset: not support v100
        smask: not support v100
        en_small_channel: switch for small_channel funtion
        en_small_k: not support v100
        en_ssparse: not support v100
        en_winograd_a: not support v100
        en_winograd_b: not support v100

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        # function's unused params is used in decorators, so disable them
        # check whether the input tensor matches the scope
        _check_fm_scope(dst_fm, "l0c", "dst_fm")
        _check_fm_scope(src_fm, "l0a", "src_fm")
        _check_fm_scope(src_filter, "l0b", "src_filter")

        # check dtype
        dtype_str = DTYPE_MAP[dst_fm.dtype] + DTYPE_MAP[src_fm.dtype] + \
                    DTYPE_MAP[src_filter.dtype]
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_mmad",
                                                            dtype_str), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dtype_str, "mmad"))
        # check m k n if both in [0, 4095]
        TikCheckUtil.check_type_match(matrix_m, (int, Expr, Scalar),
                                      "matrix_m should be int, Expr or Scalar")
        check_scalar_dtype(matrix_m, "matrix_m should be a scalar of int/uint")
        check_integer_in_range(matrix_m, range(MAX_MATRIX),
                               "matrix_m should be in range of [0, 4095],"
                               " input value is %s" % str(matrix_m))
        TikCheckUtil.check_type_match(matrix_n, (int, Expr, Scalar),
                                      "matrix_n should be int, Expr or Scalar")
        check_scalar_dtype(matrix_n, "matrix_n should be a scalar of int/uint")
        check_integer_in_range(matrix_n, range(MAX_MATRIX),
                               "matrix_n should be in range of [0, 4095],"
                               " input value is %s" % str(matrix_n))
        TikCheckUtil.check_type_match(matrix_k, (int, Expr, Scalar),
                                      "matrix_k should be int, Expr or Scalar")
        check_scalar_dtype(matrix_k, "matrix_k should be a scalar of int/uint")
        check_integer_in_range(matrix_k, range(MAX_MATRIX),
                               "matrix_k should be in range of [0, 4095],"
                               " input value is %s" % str(matrix_k))

        # check is_bias in [0, 1]
        TikCheckUtil.check_type_match(is_bias, (int, Expr),
                                      "is_bias should be int or Expr")
        check_scalar_dtype(is_bias, "is_bias should be a scalar of int/uint")
        check_integer_in_range(is_bias, range(_MAX_IS_BIAS),
                               "is_bias should be 0 or 1")
        if is_bias == _ENABLE_BIAS:
            # 0 bias disable
            bias_bit = 0
            dst_acc = "rw"
        else:
            # 1 bias enable
            bias_bit = 1
            dst_acc = "w"

        # check feature
        arch_version_str = get_soc_name() + get_soc_core_type()
        # small_k
        # winograd
        # main in smallhs or aic
        if en_winograd_a or en_winograd_b:
            TikCheckUtil.check_in_range(
                arch_version_str, (HI3796CV300ESAIC, AIC),
                "%s not support winograd" % (arch_version_str))
        # smask
        if en_weight_offset:
            check_weight_offset(smask, "mmad", "smask")
            smask_idx = smask.access_ptr('r')
        else:
            smask_idx = 0
        # code gen
        if get_soc_name() in (ASCEND_310, ASCEND_910):
            params = [
                matrix_m, matrix_k, matrix_n, bias_bit
            ]
        else:
            params = [
                matrix_m, matrix_k, matrix_n, fm_offset, smask_idx,
                en_winograd_a, en_winograd_b, en_weight_offset, en_ssparse,
                bias_bit
            ]
        args = type_convert(params)

        dst_fm_extent, src_fm_extent, src_filter_extent = \
            _calculate_mmad_extent(matrix_m, matrix_k, matrix_n, dst_fm,
                                   src_fm, src_filter)

        with self.new_scope():
            instr = tvm.call_extern(dst_fm.dtype, "mad",
                                    dst_fm.access_ptr(dst_acc,
                                                      extent=dst_fm_extent),
                                    src_fm.access_ptr("r",
                                                      extent=src_fm_extent),
                                    src_filter.access_ptr(
                                        "r", extent=src_filter_extent), *args)
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_M)
            # 1 ir is call_extern
            self.emit(instr, ONE_IR)

    @source_info_decorator()
    @debug.depthwise_conv_decorator
    def depthwise_conv(self, dst_fm,  # pylint: disable=R0913, R0914
                       src_fm, src_filter, pad_mode, l1_h, l1_w,
                       store_high_half=False, feature_offset=0,
                       weight_offset=None, pad_value=None):
        """
        depthwise convolution operation.

        Parameters
        ----------
        dst_fm: destination tensor
        src_fm: source tensor_left
        src_filter: source tensor_right
        pad_mode:   0 - no padding
                    1 - two colume on right side
                    2 - two colume on left side
                    3 - one colume on right&left side
        l1_h: height of src_fm
        l1_w: width of src_fm
        store_high_half: the high/low channels indicator, only works for type:
                    {f16f16f16, f32f16f16}. Only support bool.
                    True: Cin/Cout is the lower 8 channels out of 16 channels.
                    False: Cin/Cout is the higher 8 channels out of 16 channels.
        feature_offset: the feature map matrix offset, dtype is same as src_fm.
                        If no offset is needed, set to 8'b0. Only works for
                        src_fm dtype is b8.
        weight_offset: the weight matrix offset, not support yet.
        pad_value: value for padding, default = None

        Returns
        -------
        None
        """
        # function's input params is too much, so disable R0913, R0914
        # check dst/src
        TikCheckUtil.check_type_match(
            dst_fm, Tensor,
            "depthwise_conv dst_fm should be Tensor, input type: {}"
            .format(type(dst_fm)))
        TikCheckUtil.check_equality(
            dst_fm.scope, scope_cc,
            "depthwise_conv dst_fm scope should be L0C, input scope: {}"
            .format(dst_fm.scope))
        TikCheckUtil.check_type_match(
            src_fm, Tensor,
            "depthwise_conv src_fm should be Tensor, input type: {}"
            .format(type(src_fm)))
        TikCheckUtil.check_equality(
            src_fm.scope, scope_cbuf,
            "depthwise_conv src_fm scope should be L1, input scope: {}"
            .format(src_fm.scope))
        TikCheckUtil.check_type_match(
            src_filter, Tensor,
            "depthwise_conv src_filter should be Tensor, input type: {}"
            .format(type(src_filter)))
        TikCheckUtil.check_equality(
            src_filter.scope, scope_cb,
            "depthwise_conv src_filter scope should be L0B, input scope: {}"
            .format(src_filter.scope))
        # check dtype
        dtype_str = DTYPE_MAP[dst_fm.dtype] + DTYPE_MAP[src_fm.dtype] + \
                    DTYPE_MAP[src_filter.dtype]
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" +
                                                            "depthwise_conv",
                                                            dtype_str), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dtype_str, "depthwise_conv"))
        # check store_high_half
        TikCheckUtil.check_type_match(
            store_high_half, bool,
            "store_high_half should be bool, input type of store_high_half: {}"
            .format(type(store_high_half)))
        # check pad_mode
        TikCheckUtil.check_type_match(
            pad_mode, (int, Scalar, Expr),
            "depthwise_conv pad_mode should be int, Scalar or Expr, input type:"
            " {}".format(type(pad_mode)))
        check_scalar_dtype(pad_mode, "pad_mode should be a scalar of int/uint")
        # check W/H
        TikCheckUtil.check_type_match(
            l1_h, (int, Scalar, Expr),
            "l1_h should be int, Scalar or Expr, input "
            "type of l1_h: {}".format(type(l1_h)))
        check_scalar_dtype(l1_h, "l1_h should be a scalar of int/uint")
        TikCheckUtil.check_type_match(
            l1_w, (int, Scalar, Expr),
            "l1_w should be int, Scalar or Expr, input "
            "type of l1_w: {}".format(type(l1_w)))
        check_scalar_dtype(l1_w, "l1_w should be a scalar of int/uint")
        # check feature_offset
        TikCheckUtil.check_type_match(
            feature_offset, (int, Scalar, Expr),
            "feature_offset should be int, Scalar or Expr, input type of "
            "feature_offset: {}".format(type(feature_offset)))
        check_scalar_dtype(feature_offset, "feature_offset should be a scalar "
                                           "of int/uint")
        # check params range
        check_depthwise_conv_params(src_fm, pad_mode, l1_h, l1_w,
                                    feature_offset)
        # check l1_w/pad_mode
        check_depthwise_conv_l1_w(pad_mode, l1_w)
        # check src_fm dst_fm overflow
        _check_dc_fm_overflow(l1_h, l1_w, src_fm, pad_mode, dst_fm)
        # check src_ft overflow
        _check_dc_ft_overflow(src_filter)
        # cal extent
        dst_fm_extent, src_fm_extent, src_ft_extent = \
            _cal_extent_depthwise_conv(dst_fm, src_fm, pad_mode, l1_h, l1_w)
        # do padding
        if pad_value is not None:
            TikCheckUtil.check_type_match(
                pad_value, (int, float),
                "pad_value should be python int or float, input type: {}"
                .format(type(pad_value)))
            check_pad_value(src_fm, pad_value)
        do_load3d_padding(self, src_fm, pad_value)

        if weight_offset is not None:
            weight_offset_en = 1
            check_weight_offset(weight_offset, "depthwise_conv",
                                "weight_offset")
            # check overflow
            _check_dc_weight_offset_overflow(weight_offset)
            # cal extent, actual 16B
            weight_offset_extent = _ELE_PER_FRACTAL_EDGE*\
                                   DTYPE_SIZE[src_filter.dtype]
            args = [l1_w, l1_h, feature_offset,
                    weight_offset.access_ptr("r", extent=weight_offset_extent),
                    weight_offset_en, pad_mode, int(store_high_half)]
        else:
            weight_offset_en = 0
            args = [l1_w, l1_h, feature_offset, 0, weight_offset_en, pad_mode,
                    int(store_high_half)]
        with self.new_scope():
            # for depthwise_conv includes PIPE_MTE1 and PIPE_M, so emit
            # nop_instr for injecting pipe, temporary plan
            nop_instr = tvm.call_extern(
                src_fm.dtype, "dummy_intrin",
                src_fm.access_ptr("rw", extent=src_fm_extent),
                src_filter.access_ptr("rw", extent=src_ft_extent),)
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_MTE1)
            self.emit(nop_instr, ONE_IR)
        with self.new_scope():
            instr = tvm.call_extern(
                dst_fm.dtype, "depthwise_conv",
                dst_fm.access_ptr("w", extent=dst_fm_extent),
                src_fm.access_ptr("r", extent=src_fm_extent),
                src_filter.access_ptr("r", extent=src_ft_extent),
                *type_convert(args))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_M)
            self.emit(instr, ONE_IR)
