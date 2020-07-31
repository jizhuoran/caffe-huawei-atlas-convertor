"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_cube_api_.py
DESC:     provide cube calculation related instructions
CREATED:  2019-08-12 18:53:42
MODIFIED: 2019-08-12 21:29:18
"""
import math  # pylint: disable=C0302

from te.tik.api.tik_ir_builder import TikIRBuilder
from te.tik.tik_lib.tik_check_util import TikCheckUtil
from te.platform.cce_conf import api_check_support
from te.tik.api.tik_tensor import Tensor
from te.tik.common.util import DTYPE_SIZE
from te.tik.common.util import reduce_mul
from te.tik.common.util import ceil_div
from te.tik.tik_lib.tik_params import ONE_BLK_SIZE
from te.tik.tik_lib.tik_api_constants import DTYPE_MAP
from te.tik.api.tik_scalar import Scalar
from te.tik.tik_lib.tik_expr import Expr
from te.platform.cce_params import scope_ca
from te.platform.cce_params import scope_cbuf
from te.platform.cce_params import scope_cb
from te.platform.cce_params import scope_ubuf
from te.platform.cce_params import scope_gm
from te.platform.cce_params import scope_cc
from te.tik.tik_lib.tik_source_info import source_info_decorator
from te.tik.tik_lib.tik_params import INSTR_DTYPE_SUPPORT_STATEMENT
from te.tik.tik_lib.tik_params import MAX_REPEAT_TIMES
from .tiling_engine import FMDesc
from .tiling_engine import FilterDesc
from .tiling_engine import gen_best_tiling
from .tiling_engine import gen_fixpipe_tiling
from .reindex import ReIndexProxy

from .matmul import MatMulImpl
from ...debug.decorators import high_level_api_debug_decorator
from ...tik_lib.tik_util import all  # pylint: disable=W0622


def _get_for_loop_param(param, mode, outer_thread_num=1):
    if mode == "cout":
        iter_num = param.cout_iter_num
        thread_num = max(param.cout_thread_num, outer_thread_num)
        has_tail = param.cout_has_tail
    elif mode == "cin":
        iter_num = param.cin_iter_num
        thread_num = max(param.cin_thread_num, outer_thread_num)
        has_tail = param.cin_has_tail
    elif mode == 'hw':
        iter_num = param.hw_iter_num
        thread_num = max(param.hw_thread_num, outer_thread_num)
        has_tail = param.hw_has_tail
    else:
        raise RuntimeError("Input mode(%s) not support" % mode)

    if has_tail:
        iter_num -= 1
        if iter_num < thread_num:
            thread_num = 1
    elif iter_num < thread_num:
        thread_num = 1
    return iter_num, thread_num


def _checkout_bias_tensor_dim(tensor):
    """
    check dim of bias tensor
    Parameters
    ----------
    tensor: Tensor

    Returns
    ----------
    None
    """
    TikCheckUtil.check_type_match(tensor, Tensor, "input must be Tensor")
    TikCheckUtil.check_equality(len(tensor.shape), 4,
                                "tensor shape must be 4 dimension")
    TikCheckUtil.check_equality(
        tensor.shape[-1], 16,
        "the bias input Tensor 3th dimension must be 16")


def _check_conv2d_tensor_overflow(need_shape, tensor, tensor_name):
    """
    check tensor overflow
    Parameters
    ----------
    need_shape: shape
    tensor: Tensor
    tensor_name: name of tensor

    Returns
    ----------
    """
    need_elements = Expr(reduce_mul(need_shape) +
                         tensor.offset).eval_value()
    total_size = Expr(reduce_mul(
        tensor.indice.origin_shape)).eval_value()
    if need_elements is not None and total_size is not None:
        TikCheckUtil.check_ge(
            total_size, need_elements,
            "%s tensor overflow, instruction need %s "
            "but only %s" % (tensor_name, need_elements, total_size))


def _check_conv2d_tensor_overlap(feature_map, weight, fm_shape, kernel_shape):
    if feature_map.buffer == weight.buffer:
        feature_map_start = Expr(feature_map.offset).eval_value()
        feature_map_end = Expr(feature_map_start +
                               reduce_mul(fm_shape)).eval_value()
        weight_start = Expr(weight.offset).eval_value()
        weight_end = Expr(weight_start + reduce_mul(kernel_shape)).eval_value()
        if all(value is not None for value in
               [feature_map_start, feature_map_end,
                weight_start, weight_end]):
            if max(feature_map_start, weight_start) < \
                    min(feature_map_end, weight_end):
                TikCheckUtil.raise_error(
                    "feature_map and weight tensor address overlapping error.")


def _check_conv2d_params(dst,  # pylint: disable=R0913, R0915
                         feature_map, weight, fm_shape,
                         kernel_shape, stride, pad, dilation, pad_value,
                         init_l1out):
    # check operator
    TikCheckUtil.check_type_match(dst, Tensor,
                                  "dst should be tensor, input "
                                  "type: %s" % type(dst))
    TikCheckUtil.check_type_match(feature_map, Tensor,
                                  "feature_map should be tensor, input "
                                  "type: %s" % type(feature_map))
    TikCheckUtil.check_type_match(weight, Tensor,
                                  "weight should be tensor, input "
                                  "type: %s" % type(weight))
    # check operator scope, waiting for confirmation of naming !!!!
    TikCheckUtil.check_equality(dst.scope, scope_cc,
                                "dst's scope must be L1_out, "
                                "input scope is: %s" % dst.scope)
    TikCheckUtil.check_equality(feature_map.scope, scope_cbuf,
                                "feature_map's scope must be L1, "
                                "input scope is: %s" % feature_map.scope)
    TikCheckUtil.check_equality(weight.scope, scope_cbuf,
                                "weight's scope must be L1, "
                                "input scope is: %s" % weight.scope)
    # check dtype
    dtype_str = DTYPE_MAP[feature_map.dtype] + DTYPE_MAP[weight.dtype] + \
                DTYPE_MAP[dst.dtype]
    TikCheckUtil.check_equality(api_check_support("tik.conv2d",
                                                  dtype_str), True,
                                INSTR_DTYPE_SUPPORT_STATEMENT.
                                format(dtype_str, "conv2d"))
    # check init_l1out
    TikCheckUtil.check_type_match(
        init_l1out, bool, "init_l1out should be bool type.")
    # check fm_shape
    TikCheckUtil.check_type_match(fm_shape, (list, tuple),
                                  "fm_shape should be list or tuple")
    TikCheckUtil.check_equality(len(fm_shape), 4,
                                "fm_shape should be four-dimensional")
    for num in fm_shape:
        TikCheckUtil.check_type_match(num, int,
                                      "fm_shape should be a list of int")
    TikCheckUtil.check_in_range(
        fm_shape[0], range(1, 257),
        "fm_shape C1 dimension should be in range of [1, 256]")
    if feature_map.dtype == "float16":
        TikCheckUtil.check_equality(
            fm_shape[3], 16, "fm_shape C0 dimension should be 16 "
                             "when feature_map is float16")
        TikCheckUtil.check_in_range(
            fm_shape[0] * fm_shape[3], range(16, 4097),
            "fm_shape channel(C1*C0) should be in range of [16, 4096]")
    else:
        TikCheckUtil.check_equality(
            fm_shape[3], 32, "fm_shape C0 dimension should be 32 "
                             "when feature_map is int8 or uint8")
        TikCheckUtil.check_in_range(
            fm_shape[0] * fm_shape[3], range(32, 4097),
            "fm_shape channel(C1*C0) should be in range of [32, 4096]")
    TikCheckUtil.check_in_range(
        fm_shape[1], range(1, 4097),
        "fm_shape H dimension should be in range of [1, 4096]")
    TikCheckUtil.check_in_range(
        fm_shape[2], range(1, 4097),
        "fm_shape W dimension should be in range of [1, 4096]")
    # check kernel_shape
    TikCheckUtil.check_type_match(kernel_shape, (list, tuple),
                                  "kernel_shape should be list or tuple")
    TikCheckUtil.check_equality(len(kernel_shape), 5,
                                "kernel_shape should be five-dimensional")
    for num in kernel_shape:
        TikCheckUtil.check_type_match(
            num, int, "kernel_shape should be a list of int")
    TikCheckUtil.check_in_range(
        kernel_shape[0], range(1, 257),
        "kernel_shape C1 dimension should be in range of [1, 256]")
    if feature_map.dtype == "float16":
        TikCheckUtil.check_equality(
            kernel_shape[4], 16, "kernel_shape C0 dimension should be 16 "
                                 "when feature_map is float16")
        TikCheckUtil.check_in_range(
            kernel_shape[0] * kernel_shape[4], range(16, 4097),
            "kernel_shape channel(C1*C0) should be in range of [16, 4096]")
    else:
        TikCheckUtil.check_equality(
            kernel_shape[4], 32, "kernel_shape C0 dimension should be 32 "
                                 "when feature_map is int8 or uint8")
        TikCheckUtil.check_in_range(
            kernel_shape[0] * kernel_shape[4], range(32, 4097),
            "kernel_shape channel(C1*C0) should be in range of [32, 4096]")
    TikCheckUtil.check_in_range(
        kernel_shape[3], range(16, 4097),
        "kernel_shape Cout dimension should be in range of [16, 4096]")
    TikCheckUtil.check_equality(
        kernel_shape[3] % 16, 0,
        "kernel_shape Cout dimension should be multiple of 16")
    TikCheckUtil.check_in_range(
        kernel_shape[1], range(1, 256),
        "kernel_shape Kh dimension should be in range of [1, 255]")
    TikCheckUtil.check_in_range(
        kernel_shape[2], range(1, 256),
        "kernel_shape Kw dimension should be in range of [1, 255]")
    TikCheckUtil.check_equality(
        kernel_shape[0] * kernel_shape[4], fm_shape[0] * fm_shape[3],
        "kernel_shape channel(C1*C0) should be equal with "
        "fm_shape channel(C1*C0)")
    # check stride
    TikCheckUtil.check_type_match(stride, (list, tuple),
                                  "stride should be list or tuple")
    TikCheckUtil.check_equality(len(stride), 2,
                                "stride should be two-dimensional")
    for num in stride:
        TikCheckUtil.check_type_match(num, int,
                                      "stride should be a list of int")
        TikCheckUtil.check_in_range(
            num, range(1, 64),
            "stride_h/stride_w should be in range of [1, 63]")
    # check pad
    TikCheckUtil.check_type_match(pad, (list, tuple),
                                  "pad should be list or tuple")
    TikCheckUtil.check_equality(len(pad), 4,
                                "pad should be four-dimensional")
    for num in pad:
        TikCheckUtil.check_type_match(num, int,
                                      "pad should be a list of int")
        TikCheckUtil.check_in_range(
            num, range(256), "pad_left/pad_right/pad_top/pad_bottom should "
                             "be in range of [0, 255]")
    # check dilation
    TikCheckUtil.check_type_match(dilation, (list, tuple),
                                  "dilation should be list or tuple")
    TikCheckUtil.check_equality(len(dilation), 2,
                                "dilation should be two-dimensional")
    for num in dilation:
        TikCheckUtil.check_type_match(num, int,
                                      "dilation should be a list of int")
        TikCheckUtil.check_in_range(
            num, range(1, 256),
            "dilation_h/dilation_w should be in range of [1, 255]")
    # check pad_value
    if feature_map.dtype == "float16":
        TikCheckUtil.check_type_match(
            pad_value, (int, float),
            "pad_value should be python int or float, "
            "input type is: %s" % type(pad_value))
    else:
        TikCheckUtil.check_type_match(
            pad_value, int,
            "pad_value should be python int, "
            "input type is: %s" % type(pad_value))
        if feature_map.dtype == "uint8":
            TikCheckUtil.check_in_range(
                pad_value, range(256),
                "pad_value should be in range of [0, 255]")
        elif feature_map.dtype == "int8":
            TikCheckUtil.check_in_range(
                pad_value, range(-128, 128),
                "pad_value should be in range of [-128, 127]")
    # check feature_map tensor overflow
    _check_conv2d_tensor_overflow(fm_shape, feature_map, "feature_map")
    # check weight tensor overflow
    _check_conv2d_tensor_overflow(kernel_shape, weight, "weight")
    # check feature_map and weight overlap
    _check_conv2d_tensor_overlap(feature_map, weight, fm_shape,
                                 kernel_shape)


def _check_fixpipe_deq_params(dtype_str, extend_params, cburst_num,
                              fixpipe_config):
    quantize_params = extend_params.get("quantize_params")
    if quantize_params is None:
        return
    mode_dtype_map = {"int322fp16": "s32f16", "fp322fp16": "f32f16"}
    if dtype_str in ("s32s32", "f32f32"):
        TikCheckUtil.check_is(
            quantize_params, None,
            "extend_params['quantize_params'] should be None when "
            "src and dst dtype is %s" % dtype_str)
        return
    TikCheckUtil.check_type_match(
        quantize_params, dict,
        "extend_params['quantize_params'] should be dict, input type is %s" %
        type(quantize_params))
    if not ("mode" in quantize_params and "mode_param" in quantize_params):
        TikCheckUtil.raise_error("extend_params['quantize_params'] dict must "
                                 "contains 'mode' and 'mode_param'")
    # check mode
    TikCheckUtil.check_in_range(
        quantize_params.get("mode"), mode_dtype_map,
        "Instruction fixpipe doesn't support with quantize_params 'mode' "
        "%s." % quantize_params.get("mode"))
    TikCheckUtil.check_equality(
        dtype_str, mode_dtype_map[quantize_params.get("mode")],
        "src.dtype and dst.dtype mismatch with quantize_params 'mode' "
        "%s." % quantize_params.get("mode"))
    # check cout_blk
    if dtype_str in ("s32u8", "s32s8"):
        TikCheckUtil.check_equality(
            cburst_num % 2, 0, "cburst_num should be multiple of 2 "
                               "when quantize from int32 to int8/uint8")
    # check deqscale
    if dtype_str not in ("s32u8", "s32s8", "s32f16"):
        TikCheckUtil.check_is(
            quantize_params.get("mode_param"), None,
            "quantize_params 'mode_param' should be None when "
            "src and dst dtype is %s" % dtype_str)
    else:
        if isinstance(quantize_params.get("mode_param"), Scalar):
            TikCheckUtil.check_equality(
                quantize_params.get("mode_param").dtype, "float16",
                "quantize_params 'mode_param' should be "
                "a scalar of float16.")
        if fixpipe_config.has_bias or fixpipe_config.has_ele_wise_bias:
            TikCheckUtil.check_type_match(
                quantize_params.get("mode_param"), (float, Scalar),
                "Please specify your quantize_params 'mode_param': "
                "immediate/Scalar(float16) for deq-mode.")
        else:
            TikCheckUtil.check_type_match(
                quantize_params.get("mode_param"), (float, Scalar, Tensor),
                "Please specify your quantize_params 'mode_param': "
                "immediate/Scalar(float16)/Tensor(float16) for deq-mode.")
            if isinstance(quantize_params.get("mode_param"), Tensor):
                TikCheckUtil.check_equality(
                    quantize_params.get("mode_param").dtype, "float16",
                    "quantize_params 'mode_param' should be tensor of "
                    "float16.")
                TikCheckUtil.check_equality(
                    quantize_params.get("mode_param").scope, scope_cbuf,
                    "quantize_params 'mode_param' tensor's scope should be "
                    "L1.")
                actual_ele = Expr(reduce_mul(
                    quantize_params.get("mode_param").indice.
                    origin_shape)).eval_value()
                expected_ele = Expr(quantize_params.get("mode_param").offset +
                                    16).eval_value()
                if actual_ele is not None and expected_ele is not None:
                    TikCheckUtil.check_ge(
                        actual_ele, expected_ele,
                        "deqscale tensor overflow, expected elements: %d, "
                        "actual elements: %d." % (expected_ele, actual_ele))


def _check_fixpipe_bias_params(extend_params, src_dtype, cout):
    bias = extend_params.get("bias")
    if bias is None:
        return
    TikCheckUtil.check_type_match(
        bias, Tensor, "extend_params 'bias' should be Tensor, "
                      "input type is %s" % type(bias))
    TikCheckUtil.check_equality(bias.scope, scope_cbuf,
                                "extend_params bias's scope should be L1")
    TikCheckUtil.check_equality(
        src_dtype, bias.dtype,
        "extend_params 'bias' should have the same dtype with src")
    total_ele = Expr(reduce_mul(bias.indice.origin_shape)).eval_value()
    need_ele = Expr(bias.offset + cout).eval_value()
    if total_ele is not None and need_ele is not None:
        TikCheckUtil.check_ge(
            total_ele, need_ele,
            "extend_params 'bias' tensor overflow, expected elements: %s, "
            "actual elements: %s" % (need_ele, total_ele))


def _check_fixpipe_ele_bias_params(extend_params, src, cburst_num,
                                   burst_len):
    ele_bias = extend_params.get("element-wise-add")
    if ele_bias is None:
        return
    TikCheckUtil.check_type_match(
        ele_bias, Tensor, "extend_params 'element-wise-add' should be "
                          "Tensor, input type is %s" % type(ele_bias))
    TikCheckUtil.check_equality(ele_bias.scope, scope_cbuf,
                                "extend_params 'element-wise-add' scope "
                                "should be L1")
    TikCheckUtil.check_equality(
        ele_bias.dtype, src.dtype, "extend_params 'element-wise-add' should "
                                   "have the same dtype with src")
    # check overflow
    frac_len = 16
    howo = burst_len * ONE_BLK_SIZE // DTYPE_SIZE[src.dtype] // frac_len
    round_howo = ceil_div(howo, frac_len) * frac_len
    extent_ele = cburst_num * round_howo * frac_len
    total_ele = Expr(reduce_mul(ele_bias.indice.origin_shape)).eval_value()
    need_ele = Expr(ele_bias.offset + extent_ele).eval_value()
    if total_ele is not None and need_ele is not None:
        TikCheckUtil.check_ge(
            total_ele, need_ele,
            "extend_params 'element-wise-add' tensor overflow, expected "
            "elements: %s, actual elements: %s" % (need_ele, total_ele))


def _check_fixpipe_relu_params(extend_params, fixpipe_config):
    relu = extend_params.get("relu")
    if "relu" not in extend_params:
        return
    TikCheckUtil.check_type_match(
        relu, bool,
        "extend_params 'relu' should be bool type.")
    if fixpipe_config.has_ele_wise_bias or fixpipe_config.has_bias:
        TikCheckUtil.check_is(
            relu, False,
            "Intrinsic fixpipe doesn't support relu "
            "when enable element-wise-add or bias, "
            "extend_params 'relu' should be False")
    if fixpipe_config.has_deq and isinstance(
            extend_params["quantize_params"].get("mode_param"),
            (float, Scalar)):
        TikCheckUtil.check_is(
            relu, False,
            "Intrinsic fixpipe doesn't support relu when quantize "
            "int322fp16 and mode_param is scalar or float, "
            "extend_params 'relu' should be False ")


def _check_fixpipe_tensor_overflow(dst, src, cburst_num, burst_len):
    frac_len = 16
    howo = burst_len * ONE_BLK_SIZE // DTYPE_SIZE[src.dtype] // frac_len
    round_howo = ceil_div(howo, frac_len) * frac_len
    # check src overflow
    extent_ele = cburst_num * round_howo * frac_len
    total_ele = Expr(reduce_mul(src.indice.origin_shape)).eval_value()
    need_ele = Expr(src.offset + extent_ele).eval_value()
    if total_ele is not None and need_ele is not None:
        TikCheckUtil.check_ge(
            total_ele, need_ele,
            "src tensor overflow, expected elements: %s, "
            "actual elements: %s" % (need_ele, total_ele))
    # check dst overflow
    extent_ele = cburst_num * howo * frac_len
    total_ele = Expr(reduce_mul(dst.indice.origin_shape)).eval_value()
    need_ele = Expr(dst.offset + extent_ele).eval_value()
    if total_ele is not None and need_ele is not None:
        TikCheckUtil.check_ge(
            total_ele, need_ele,
            "dst tensor overflow, expected elements: %s, "
            "actual elements: %s" % (need_ele, total_ele))


class TikCubeOpenApi(TikIRBuilder):
    """Cube Api"""

    # @cond
    def __init__(self):
        super(TikCubeOpenApi, self).__init__()
        self.core_arch = None
        self.core_version = None

    # @endcond

    def _make_load_l0a_code(self,  # pylint: disable=W0613, R0913, R0914
                            cin_actual,
                            cin_i, hw_actual, hw_i, param):
        # no use the hw_actual var, fix because pylint
        hw_actual = hw_actual
        data_l0a = self.Tensor(  # pylint: disable=E1101
            param.fm_desc.input_dtype, (param.hw_tile_block,
                                        param.cin_tile_block,
                                        param.block_size, param.c_0),
            name=param.feature_map.tensor.buffer.name + "L0A",
            scope=scope_ca)

        cin_pos = cin_i * param.cin_tile_block
        if param.tiling.l0a_mode == 1:
            with self.for_range(0, cin_actual) as k:
                ho_wo_pos = hw_i * param.hw_tile_block * param.block_size
                ho_idx = ho_wo_pos // param.fm_desc.w_o
                wo_idx = ho_wo_pos % param.fm_desc.w_o

                hi_idx = ho_idx * param.fm_desc.stride_h
                wi_idx = wo_idx * param.fm_desc.stride_w

                c1_idx = (cin_pos + k) // (param.k_h * param.k_w)
                k_hw_idx = (cin_pos + k) % (param.k_h * param.k_w)
                k_i = k_hw_idx // param.k_w  # h_idx
                k_j = k_hw_idx % param.k_w  # w_idx
                l0a_idx = k * param.block_size * param.c_0
                disable_c1 = 0  # cin_i * Cin_tiling + k
                c1_offset = c1_idx * param.c_0 * param.fm_desc.h_i * \
                            param.fm_desc.w_i
                self.load3dv1(  # pylint: disable=E1101
                    data_l0a.flatten()[l0a_idx],
                    param.feature_map.flat_access(c1_offset),
                    param.fm_desc.pad_list,
                    param.fm_desc.h_i, param.fm_desc.w_i,
                    disable_c1, k_j, k_i,
                    wi_idx - param.fm_desc.pad_left,
                    hi_idx - param.fm_desc.pad_top,
                    param.fm_desc.stride_w, param.fm_desc.stride_h,
                    param.k_w, param.k_h, param.dilation_w, param.dilation_h,
                    cin_actual, 1, param.hw_tile_block,
                    pad_value=param.pad_value)
        elif param.tiling.l0a_mode == 0:
            with self.for_range(0, param.hw_tile_block) as k:
                ho_wo_pos = (hw_i * param.hw_tile_block + k) * param.block_size
                ho_idx = ho_wo_pos // param.fm_desc.w_o
                wo_idx = ho_wo_pos % param.fm_desc.w_o

                hi_idx = ho_idx * param.fm_desc.stride_h
                wi_idx = wo_idx * param.fm_desc.stride_w

                # we load the whole row in 1 load3d
                c1_idx = cin_pos // (param.k_h * param.k_w)
                k_hw_idx = cin_pos % (param.k_h * param.k_w)
                k_i = k_hw_idx // param.k_w  # h_idx
                k_j = k_hw_idx % param.k_w  # w_idx
                l0a_idx = k * cin_actual * param.block_size * param.c_0
                disable_c1 = 0  # Cin_i * Cin_tiling + k
                c1_offset = c1_idx * param.c_0 * param.fm_desc.h_i * \
                            param.fm_desc.w_i
                self.load3dv1(  # pylint: disable=E1101
                    data_l0a.flatten()[l0a_idx],
                    param.feature_map.flat_access(c1_offset),
                    param.fm_desc.pad_list,
                    param.fm_desc.h_i, param.fm_desc.w_i, disable_c1, k_j, k_i,
                    wi_idx - param.fm_desc.pad_left,
                    hi_idx - param.fm_desc.pad_top,
                    param.fm_desc.stride_w, param.fm_desc.stride_h,
                    param.k_w, param.k_h,
                    param.dilation_w, param.dilation_h,
                    1, 0, cin_actual, pad_value=param.pad_value)
        return data_l0a

    def _make_load_l0b_code(self,  # pylint: disable=R0913, R0914
                            cin_actual, cout_actual, cout_i, cin_i, param):
        data_l0b = self.Tensor(  # pylint: disable=E1101
            param.weight.dtype, (param.cout_tile_block *
                                 param.cin_tile_block *
                                 param.block_size * param.c_0,),
            name=param.weight.tensor.buffer.name + "L0B", scope=scope_cb)
        if param.tiling.l0b_mode == 0:
            if param.cout_iter_num == 1:
                # load one column at once
                w_idx = (cin_i * param.cin_tile_block) * \
                        param.cout_block_num*param.block_size*param.c_0 + \
                        cout_i * param.cout_tile_block * \
                        param.block_size * param.c_0
                l0b_idx = 0
                self.load2dv1(  # pylint: disable=E1101
                    data_l0b.flatten()[l0b_idx],
                    param.weight.flat_access(w_idx), 0,
                    cout_actual*cin_actual, 1, 0)
            else:
                # load row by row
                with self.for_range(0, cin_actual) as sub_cin_idx:
                    w_idx = (cin_i * param.cin_tile_block + sub_cin_idx) * \
                            param.cout_block_num*param.block_size*param.c_0 +\
                            cout_i * param.cout_tile_block *\
                            param.block_size * param.c_0
                    l0b_idx = sub_cin_idx * cout_actual *\
                              param.block_size * param.c_0
                    self.load2dv1(  # pylint: disable=E1101
                        data_l0b.flatten()[l0b_idx],
                        param.weight.flat_access(w_idx), 0, cout_actual, 1, 0)
        elif param.tiling.l0b_mode == 1:
            # load one column at once
            w_idx = cin_i * param.cin_tile_block * param.cout_block_num * \
                    param.block_size * param.c_0 + \
                    cout_i * param.cout_tile_block * \
                    param.block_size * param.c_0
            self.load2dv1(data_l0b.flatten()[0],  # pylint: disable=E1101
                          param.weight.flat_access(w_idx), 0,
                          cin_actual, param.cout_block_num, 0)
        return data_l0b

    def _make_mmad_code(self, data_l0a,  # pylint: disable=R0913, R0914
                        data_l0b, cin_actual, hw_actual,
                        cout_actual, cin_i, hw_i, cout_i, param):
        # we only care data in K dim
        dst_flatten_idx = cout_i * param.round_woho * param.block_size *\
                          param.cout_tile_block + hw_i * param.hw_tile_block *\
                          param.block_size * param.block_size
        from functools import partial
        partial_mmad = partial(
            self.mmad,  # pylint: disable=E1101
            param.dst.flatten()[dst_flatten_idx],
            data_l0a, data_l0b, hw_actual*param.block_size,
            cin_actual * param.c_0, cout_actual*param.block_size)
        with self.if_scope(all(cin_i == 0, param.init_l1out)):
            partial_mmad(0)
        with self.else_scope():
            partial_mmad(1)

    def _do_nk_tiling(self, param):
        def _nk_cin_loop(cout_actual, cout_i):
            cin_iter_num, cin_thread_num = _get_for_loop_param(param, "cin")
            if cin_iter_num > 2 and cin_iter_num != ceil_div(cin_iter_num,
                                                             2) * 2:
                with self.for_range(0, cin_iter_num - 1,
                                    thread_num=cin_thread_num) as cin_i:
                    data_l0a = self._make_load_l0a_code(
                        param.cin_tile_block, cin_i,
                        param.hw_tile_block, 0, param)
                    data_l0b = self._make_load_l0b_code(
                        param.cin_tile_block, cout_actual,
                        cout_i, cin_i, param)
                    self._make_mmad_code(
                        data_l0a, data_l0b, param.cin_tile_block,
                        param.hw_tile_block, cout_actual, cin_i,
                        0, cout_i, param)

                with self.new_stmt_scope():
                    data_l0a = self._make_load_l0a_code(
                        param.cin_tile_block, cin_iter_num - 1,
                        param.hw_tile_block, 0, param)
                    data_l0b = self._make_load_l0b_code(
                        param.cin_tile_block, cout_actual,
                        cout_i, cin_iter_num - 1, param)
                    self._make_mmad_code(
                        data_l0a, data_l0b, param.cin_tile_block,
                        param.hw_tile_block, cout_actual, cin_iter_num - 1,
                        0, cout_i, param)

            else:
                with self.for_range(0, cin_iter_num,
                                    thread_num=cin_thread_num) as cin_i:
                    data_l0a = self._make_load_l0a_code(
                        param.cin_tile_block, cin_i,
                        param.hw_tile_block, 0, param)
                    data_l0b = self._make_load_l0b_code(
                        param.cin_tile_block, cout_actual,
                        cout_i, cin_i, param)
                    self._make_mmad_code(
                        data_l0a, data_l0b, param.cin_tile_block,
                        param.hw_tile_block, cout_actual, cin_i,
                        0, cout_i, param)

            if param.cin_has_tail:
                data_l0a = self._make_load_l0a_code(
                    param.cin_tail_block, cin_iter_num,
                    param.hw_tile_block, 0, param)
                data_l0b = self._make_load_l0b_code(
                    param.cin_tail_block, cout_actual,
                    cout_i, cin_iter_num, param)
                self._make_mmad_code(
                    data_l0a, data_l0b, param.cin_tail_block,
                    param.hw_tile_block,
                    cout_actual, cin_iter_num, 0, cout_i, param)

        cout_iter_num, cout_thread_num = _get_for_loop_param(param, "cout")
        with self.for_range(0, cout_iter_num,
                            thread_num=cout_thread_num) as cout_i:
            _nk_cin_loop(param.cout_tile_block, cout_i)
        if param.cout_has_tail:
            _nk_cin_loop(param.cout_tail_block, cout_iter_num)

    def _do_kn_tiling(self, param):
        def _kn_cout_loop(cin_actual, cin_i, outer_thread_num=1):
            data_l0a = self._make_load_l0a_code(
                cin_actual, cin_i, param.hw_tile_block, 0, param)
            cout_iter_num, cout_thread_num = _get_for_loop_param(
                param, "cout", outer_thread_num)
            if cout_iter_num > 2 and cout_iter_num != \
                    ceil_div(cout_iter_num, 2) * 2:
                with self.for_range(0, cout_iter_num - 1,
                                    thread_num=cout_thread_num) as cout_i:
                    data_l0b = self._make_load_l0b_code(
                        cin_actual, param.cout_tile_block,
                        cout_i, cin_i, param)
                    self._make_mmad_code(
                        data_l0a, data_l0b, cin_actual, param.hw_tile_block,
                        param.cout_tile_block, cin_i, 0, cout_i, param)
                with self.new_stmt_scope():
                    data_l0b = self._make_load_l0b_code(
                        cin_actual, param.cout_tile_block,
                        cout_iter_num - 1, cin_i, param)
                    self._make_mmad_code(
                        data_l0a, data_l0b, cin_actual, param.hw_tile_block,
                        param.cout_tile_block, cin_i, 0, cout_iter_num - 1,
                        param)
            else:
                with self.for_range(0, cout_iter_num,
                                    thread_num=cout_thread_num) as cout_i:
                    data_l0b = self._make_load_l0b_code(
                        cin_actual, param.cout_tile_block,
                        cout_i, cin_i, param)
                    self._make_mmad_code(
                        data_l0a, data_l0b, cin_actual, param.hw_tile_block,
                        param.cout_tile_block, cin_i, 0, cout_i, param)

            if param.cout_has_tail:
                data_l0b = self._make_load_l0b_code(
                    cin_actual, param.cout_tail_block,
                    cout_iter_num, cin_i, param)
                self._make_mmad_code(
                    data_l0a, data_l0b, cin_actual,
                    param.hw_tile_block, param.cout_tail_block,
                    cin_i, 0, cout_iter_num, param)

        cin_iter_num, cin_thread_num = _get_for_loop_param(param, "cin")
        if cin_iter_num > 2 and cin_iter_num != ceil_div(cin_iter_num, 2)*2:
            with self.for_range(0, cin_iter_num - 1,
                                thread_num=cin_thread_num) as cin_i:
                _kn_cout_loop(param.cin_tile_block, cin_i)
            with self.new_stmt_scope():
                _kn_cout_loop(param.cin_tile_block, cin_iter_num - 1,
                              cin_thread_num)
        else:
            with self.for_range(0, cin_iter_num,
                                thread_num=cin_thread_num) as cin_i:
                _kn_cout_loop(param.cin_tile_block, cin_i)

        if param.cin_has_tail:
            _kn_cout_loop(param.cin_tail_block, cin_iter_num)

    def _do_mn_tiling(self, param):
        def _mn_cout_loop(cin_actual, cin_i,  # pylint: disable=R0913
                          hw_actual, hw_i, data_l0a, outer_thread_num=1):
            cout_iter_num, cout_thread_num = _get_for_loop_param(
                param, "cout", outer_thread_num)
            if cout_iter_num > 2 and cout_iter_num != \
                    ceil_div(cout_iter_num, 2)*2:
                with self.for_range(0, cout_iter_num - 1,
                                    thread_num=cout_thread_num) as cout_i:
                    data_l0b = self._make_load_l0b_code(
                        cin_actual, param.cout_tile_block,
                        cout_i, cin_i, param)
                    self._make_mmad_code(
                        data_l0a, data_l0b, cin_actual, hw_actual,
                        param.cout_tile_block, cin_i, hw_i, cout_i, param)
                with self.new_stmt_scope():
                    data_l0b = self._make_load_l0b_code(
                        cin_actual, param.cout_tile_block,
                        cout_iter_num - 1, cin_i, param)
                    self._make_mmad_code(
                        data_l0a, data_l0b, cin_actual, hw_actual,
                        param.cout_tile_block, cin_i, hw_i,
                        cout_iter_num - 1, param)
            else:
                with self.for_range(0, cout_iter_num,
                                    thread_num=cout_thread_num) as cout_i:
                    data_l0b = self._make_load_l0b_code(
                        cin_actual, param.cout_tile_block,
                        cout_i, cin_i, param)
                    self._make_mmad_code(
                        data_l0a, data_l0b, cin_actual, hw_actual,
                        param.cout_tile_block, cin_i, hw_i, cout_i, param)
            if param.cout_has_tail:
                data_l0b = self._make_load_l0b_code(
                    cin_actual, param.cout_tail_block,
                    cout_iter_num, cin_i, param)
                self._make_mmad_code(
                    data_l0a, data_l0b, cin_actual, hw_actual,
                    param.cout_tail_block, cin_i, hw_i, cout_iter_num, param)

        def _mn_hw_loop(cin_actual, cin_i, outer_thread_num=1):
            hw_iter_num, hw_thread_num = \
                _get_for_loop_param(param, "hw", outer_thread_num)
            if hw_iter_num > 2 and hw_iter_num != ceil_div(hw_iter_num, 2)*2:
                with self.for_range(0, hw_iter_num - 1,
                                    thread_num=hw_thread_num) as hw_i:
                    data_l0a = self._make_load_l0a_code(
                        cin_actual, cin_i, param.hw_tile_block, hw_i, param)
                    _mn_cout_loop(cin_actual, cin_i, param.hw_tile_block,
                                  hw_i, data_l0a)
                with self.new_stmt_scope():
                    data_l0a = self._make_load_l0a_code(
                        cin_actual, cin_i, param.hw_tile_block,
                        hw_iter_num - 1, param)
                    _mn_cout_loop(cin_actual, cin_i, param.hw_tile_block,
                                  hw_iter_num - 1, data_l0a,
                                  max(hw_thread_num, outer_thread_num))
            else:
                with self.for_range(0, hw_iter_num,
                                    thread_num=hw_thread_num) as hw_i:
                    data_l0a = self._make_load_l0a_code(
                        cin_actual, cin_i, param.hw_tile_block, hw_i, param)
                    _mn_cout_loop(cin_actual, cin_i, param.hw_tile_block,
                                  hw_i, data_l0a)
            if param.hw_has_tail:
                data_l0a = self._make_load_l0a_code(
                    cin_actual, cin_i, param.hw_tile_block, hw_iter_num, param)
                _mn_cout_loop(cin_actual, cin_i, param.hw_tail_block,
                              hw_iter_num, data_l0a)

        cin_iter_num, cin_thread_num = _get_for_loop_param(param, "cin")
        if cin_iter_num > 2 and cin_iter_num != ceil_div(cin_iter_num,
                                                         2) * 2:
            with self.for_range(0, cin_iter_num - 1,
                                thread_num=cin_thread_num) as cin_i:
                _mn_hw_loop(param.cin_tile_block, cin_i)
            with self.new_stmt_scope():
                _mn_hw_loop(param.cin_tile_block, cin_iter_num - 1,
                            cin_thread_num)
        else:
            with self.for_range(0, cin_iter_num,
                                thread_num=cin_thread_num) as cin_i:
                _mn_hw_loop(param.cin_tile_block, cin_i)
        if param.cin_has_tail:
            _mn_hw_loop(param.cin_tail_block, cin_iter_num)

    def _do_nm_tiling(self, param):
        def _nm_hw_loop(cin_actual, cin_i,  # pylint: disable=R0913
                        cout_actual, cout_i, data_l0b, outer_thread_num=1):
            hw_iter_num, hw_thread_num = _get_for_loop_param(
                param, "hw", outer_thread_num)
            if hw_iter_num > 2 and hw_iter_num != ceil_div(hw_iter_num, 2)*2:
                with self.for_range(0, hw_iter_num - 1,
                                    thread_num=hw_thread_num) as hw_i:
                    data_l0a = self._make_load_l0a_code(
                        cin_actual, cin_i, param.hw_tile_block, hw_i, param)
                    self._make_mmad_code(
                        data_l0a, data_l0b, cin_actual, param.hw_tile_block,
                        cout_actual, cin_i, hw_i, cout_i, param)
                with self.new_stmt_scope():
                    data_l0a = self._make_load_l0a_code(
                        cin_actual, cin_i, param.hw_tile_block,
                        hw_iter_num-1, param)
                    self._make_mmad_code(
                        data_l0a, data_l0b, cin_actual, param.hw_tile_block,
                        cout_actual, cin_i, hw_iter_num - 1, cout_i, param)
            else:
                with self.for_range(0, hw_iter_num,
                                    thread_num=hw_thread_num) as hw_i:
                    data_l0a = self._make_load_l0a_code(
                        cin_actual, cin_i, param.hw_tile_block, hw_i, param)
                    self._make_mmad_code(
                        data_l0a, data_l0b, cin_actual, param.hw_tile_block,
                        cout_actual, cin_i, hw_i, cout_i, param)
            if param.hw_has_tail:
                data_l0a = self._make_load_l0a_code(
                    cin_actual, cin_i, param.hw_tile_block,
                    hw_iter_num, param)
                self._make_mmad_code(
                    data_l0a, data_l0b,
                    cin_actual, param.hw_tail_block,
                    cout_actual, cin_i, hw_iter_num, cout_i, param)

        def _nm_cout_loop(cin_actual, cin_i, outer_thread_num=1):
            cout_iter_num, cout_thread_num = _get_for_loop_param(
                param, "cout", outer_thread_num)
            if cout_iter_num > 2 and cout_iter_num != ceil_div(cout_iter_num,
                                                               2) * 2:
                with self.for_range(0, cout_iter_num - 1,
                                    thread_num=cout_thread_num) as cout_i:
                    data_l0b = self._make_load_l0b_code(
                        cin_actual, param.cout_tile_block,
                        cout_i, cin_i, param)
                    _nm_hw_loop(cin_actual, cin_i, param.cout_tile_block,
                                cout_i, data_l0b)
                with self.new_stmt_scope():
                    data_l0b = self._make_load_l0b_code(
                        cin_actual, param.cout_tile_block,
                        cout_iter_num - 1, cin_i, param)
                    _nm_hw_loop(cin_actual, cin_i, param.cout_tile_block,
                                cout_iter_num - 1, data_l0b,
                                max(cout_thread_num, outer_thread_num))
            else:
                with self.for_range(0, cout_iter_num,
                                    thread_num=cout_thread_num) as cout_i:
                    data_l0b = self._make_load_l0b_code(
                        cin_actual, param.cout_tile_block,
                        cout_i, cin_i, param)
                    _nm_hw_loop(cin_actual, cin_i, param.cout_tile_block,
                                cout_i, data_l0b)
            if param.cout_has_tail:
                data_l0b = self._make_load_l0b_code(
                    cin_actual, param.cout_tail_block,
                    cout_iter_num, cin_i, param)
                _nm_hw_loop(cin_actual, cin_i, param.cout_tail_block,
                            cout_iter_num, data_l0b)

        cin_iter_num, cin_thread_num = _get_for_loop_param(param, "cin")
        if cin_iter_num > 2 and cin_iter_num != ceil_div(cin_iter_num,
                                                         2) * 2:
            with self.for_range(0, cin_iter_num - 1,
                                thread_num=cin_thread_num) as cin_i:
                _nm_cout_loop(param.cin_tile_block, cin_i)
            with self.new_stmt_scope():
                _nm_cout_loop(param.cin_tile_block, cin_iter_num - 1,
                              cin_thread_num)
        else:
            with self.for_range(0, cin_iter_num,
                                thread_num=cin_thread_num) as cin_i:
                _nm_cout_loop(param.cin_tile_block, cin_i)
        if param.cin_has_tail:
            _nm_cout_loop(param.cin_tail_block, cin_iter_num)

    @source_info_decorator()
    @high_level_api_debug_decorator
    def conv2d(self,  # pylint: disable=R0913, R0914, R0915
               dst, feature_map, weight, fm_shape, kernel_shape, stride,
               pad, dilation, pad_value=0, init_l1out=True):
        """
        Performs 2D convolution on an input tensor ...
        Description:
          Performs 2D convolution on an input tensor and a weight tensor and
          outputs a result tensor.
          The following data types are supported (feature_map:weight:dst):
            - uint8:int8:int32
            - int8:int8:int32
            - float16:float16:float32
        Args:
          dst: Start element of the destination operand. For details about data
           type restrictions, see Table Data type
          combination of feature_map, weight, and dst. The scope is L1OUT.
          Has format [Cout/16, Ho, Wo, 16], and size Cout * Ho * Wo,where, Ho
          and Wo can be calculated as follows:
            - Ho = floor((H + pad_top + pad_bottom - dilation_h * (Kh - 1) - 1)
             / stride_h + 1)
            - Wo = floor((W + pad_left + pad_right - dilation_w * (Kw - 1) - 1)
             / stride_w + 1)
          The hardware requires that Ho * Wo be a multiple of 16. When defining
           the dst tensor, the shape should be
          rounded up to the nearest multiple of 16 pixels. The actual shape
          size should be Cout * round_howo:
            - round_howo = ceil(Ho * Wo/6) * 16
          The invalid data introduced due to round-up will be removed in the
          subsequent fixpipe operation.
          feature_map: Start element of the input tensor operand. For details
          about data type restrictions, see Table
          Data type combination of feature_map, weight, and dst. The scope
          is L1.
          weight: Start element of the weight tensor operand. For details about
           the data type restrictions, see Table
          Data type combination of feature_map, weight, and dst. The scope
          is L1.
          fm_shape: Shape of the input tensor, in the format of
          [C1, H, W, C0].C1 * C0 indicates the number of input channels.
            - If feature_map is of type float16, C0 is 16, of type int.
            - If feature_map is of type int8 or uint8, C0 is 32, of type int.
            - C1 is an immediate within the range [1, 256]. The number of input
             channels is within the range [16 or 32, 4096].
          H is an immediate of type int, specifying the height. The value range
           is [1, 4096].
          W is an immediate of type int, specifying the width. The value range
          is [1, 4096].
          kernel_shape: Shape of each convolution kernel tensor, in the
          format of [C1, Kh, Kw, Cout, C0].
          C1 * C0 indicates the number of input channels.
            - If feature_map is of type float16, C0 is 16, of type int.
            - If feature_map is of type int8 or uint8, C0 is 32, of type int.
            - C1 is an immediate within the range [1, 256]. The number of input
             channels is within the range [16 or 32, 4096].
            - Has the same number of input channels as fm_shape.
          Cout is an int specifying the number of convolution kernels.
          The value is a multiple of 16 within the range [16, 4096].
          Kh is an int specifying the height of each convolution kernel.
          The value range is [1, 255].
          Kw is an int specifying the width of each convolution kernel.
          The value range is [1, 255].
          stride: Convolution stride, in the format of [stride_h, stride_w].
            - stride_h: an int specifying the height stride.
            The value range is [1, 63].
            - stride_w: an int specifying the width stride.
            The value range is [1, 63].
          pad: Padding factors, in the format of [pad_left, pad_right,
          pad_top, pad_bottom].
            - pad_left: an int specifying the number of columns to be padded
            to the left of the feature_map. The value range is [0, 255].
            - pad_right: an int specifying the number of columns to be padded
            to the right of the feature_map. The value range is [0, 255].
            - pad_top: an int specifying the number of rows to be padded to
            the top of the feature_map. The value range is [0, 255].
            - pad_bottom: an int specifying the number of rows to be padded to
            the bottom of the feature_map. The value range is [0, 255].
          dilation: Convolution dilation factors, in the format of [
          dilation_h, dilation_w]
            - dilation_h: an int specifying the height dilation factor.
            The value range is [1, 255].
            - dilation_w: an int specifying the width dilation factor.
            The value range is [1, 255].
          The width and height of the dilated convolution kernel is
          calculated as follows: dilation_w*(Kw-1)+1;dilation_h* (Kh - 1) + 1
          pad_value: Padding value, an immediate of int or float Defaults
          to 0. Value range:
            - If feature_map is of type uint8, pad_value is within the
            range [0, 255].
            - If feature_map is of type int8, pad_value is within the
            range [-128, +127].
            - If feature_map is of type uint8 or int8, pad_value is an
            immediate of type int.
            - If feature_map is of type float16, pad_value is within
            the range [-65504, +65504].
          init_l1out: A bool specifying whether to initialize dst . Defaults
          to True.
            - True: The dst initial matrix will be overwritten by the
            computation result.
            - False: The dst initial matrix stores the previous conv2d result
            and will be accumulated with the new conv2d result.

        Table:
          Data type combination of feature_map, weight, and dst
        |feature_map.dtype  |weight.dtype  |dst.dtype  |
        |----             |----        |----     |
        |int8             |int8        |int32    |
        |uint8            |int8        |int32    |
        |float16          |float16     |float32  |

        Restrictions:
          - It takes a long time to perform step-by-step debugging. Therefore,
          step-by-step debugging is not recommended.
          - This instruction must not be used together with the vectoring
          instructions.
          - This instruction should be used together with the fixpipe
          instruction.

        Returns:
            None

        Examples:
          #Example 1: feature_map:weight:dst of type uint8:int8:int32

            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            # Define the tensors.
            feature_map_gm = tik_instance.Tensor("uint8", [1, 4, 4, 32],
                                name='feature_map_gm', scope=tik.scope_gm)
            weight_gm = tik_instance.Tensor("int8", [1, 2, 2, 32, 32],
                                name='weight_gm', scope=tik.scope_gm)
            dst_gm = tik_instance.Tensor("int32", [2, 9, 16], name='dst_gm',
                                scope=tik.scope_gm)
            feature_map = tik_instance.Tensor("uint8", [1, 4, 4, 32],
                                name='feature_map', scope=tik.scope_cbuf)
            weight = tik_instance.Tensor("int8", [1, 2, 2, 32, 32],
                                name='weight', scope=tik.scope_cbuf)
            # dst  has shape [2, 16, 16], where, cout = 32. cout_blocks = 2,
            # ho = 3, wo = 3, howo = 9. Therefore, round_howo = 16.
            dst = tik_instance.Tensor("int32", [2, 16, 16], name='dst',
                                scope=tik.scope_cbuf_out)
            # Move data from the GM to the source operand tensor.
            tik_instance.data_move(feature_map, feature_map_gm, 0, 1, 16, 0, 0)
            tik_instance.data_move(weight, weight_gm, 0, 1, 128, 0, 0)
            # Perform convolution.
            tik_instance.conv2d(dst, feature_map, weight, [1, 4, 4, 32],
                            [1, 2, 2, 32, 32], [1, 1], [0, 0, 0, 0], [1, 1], 0)
            # Move dst from L1OUT to the GM by co-working with the fixpipe
            #instruction.
            # cout_blocks = 2, cburst_num = 2, burst_len = howo * 16 *
            #src_dtype_size/32 = 9 * 16 * 4/32 = 18
            tik_instance.fixpipe(dst_gm, dst, 2, 18, 0, 0, extend_params=None)
            tik_instance.BuildCCE(kernel_name="conv2d", inputs=[feature_map_gm,
                                    weight_gm], outputs=[dst_gm])


          #Inputs:
          #feature_map_gm:
            [[[[2, 4, 2, 3, 2, ..., 3, 3, 0]]]]
          #weight_gm:
            [[[[[-3, -5, -4, ..., -2, -4, -2]]]]]
          #Returns:
          #dst_gm:
            [[[-230,  -11,  -83, -103, -123, ..., -174, -255]]]

        #Example 2: feature_map:weight:dst of type float16:float16:float32

          from te import tik
          tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
          # Define the tensors.
          feature_map_gm = tik_instance.Tensor("float16", [2, 4, 4, 16],
                                    name='feature_map_gm', scope=tik.scope_gm)
          weight_gm = tik_instance.Tensor("float16", [2, 2, 2, 16, 16],
                                    name='weight_gm', scope=tik.scope_gm)
          dst_gm = tik_instance.Tensor("float32", [1, 4, 16], name='dst_gm',
                                    scope=tik.scope_gm)
          feature_map = tik_instance.Tensor("float16", [2, 4, 4, 16],
                                    name='feature_map', scope=tik.scope_cbuf)
          weight = tik_instance.Tensor("float16", [2, 2, 2, 16, 16],
                                    name='weight', scope=tik.scope_cbuf)
          # dst  has shape [1, 16, 16], where, cout = 16, cout_blocks = 1,
          # ho = 2, wo = 2, howo = 4. Therefore, round_howo = 16.
          dst = tik_instance.Tensor("float32", [1, 16, 16], name='dst',
                                    scope=tik.scope_cbuf_out)
          # Move data from the GM to the source operand tensor.
          tik_instance.data_move(feature_map, feature_map_gm, 0, 1, 32, 0, 0)
          tik_instance.data_move(weight, weight_gm, 0, 1, 128, 0, 0)
          # Perform convolution.
          tik_instance.conv2d(dst, feature_map, weight, [2, 4, 4, 16],
                    [2, 2, 2, 16, 16], [1, 1], [0, 0, 0, 0], [2, 2], 0)
          # Move dst from L1OUT to the GM by co-working with the fixpipe
          # instruction.
          # cout_blocks = 1, cburst_num = 1, burst_len = howo * 16 *
          # src_dtype_size/32 = 4 * 16 * 4/32 = 8
          tik_instance.fixpipe(dst_gm, dst, 1, 8, 0, 0, extend_params=None)
          tik_instance.BuildCCE(kernel_name="conv2d", inputs=[feature_map_gm,
                                    weight_gm], outputs=[dst_gm])

          #Inputs:
          #feature_map_gm:
          [[[[0.0, 0.01, 0.02, 0.03, 0.04, ..., 5.09, 5.1, 5.11]]]]
          #weight_gm:
          [[[[[0.0, 0.01, 0.02, 0.03, 0.04, ..., 20.46, 20.47]]]]]
          #Returns:
          #    dst_gm:
          #    [[[3568.7373, 3612.8433, 3657.0618, 3701.162 , 3745.287 ,
          #    3789.4834, 3833.6282, 3877.876 , 3921.9812, 3966.0745,
          #   4010.311 , 4054.4119, 4098.5713, 4142.702 , 4186.8457,
          #    4231.0312],
          #    [3753.9888, 3801.3733, 3848.8735, 3896.2534, 3943.6558,
          #    3991.1353, 4038.5586, 4086.0913, 4133.4736, 4180.8457,
          #    4465.4844],
          #    [4309.196 , 4366.4077, 4423.745 , 4480.9565, 4538.1816,
          #    4595.5054, 4652.755 , 4710.135 , 4767.34  , 4824.5405,
          #    4881.897 , 4939.1104, 4996.374 , 5053.6226, 5110.871 ,
          #    5168.179 ],
          #    [4494.4526, 4554.944 , 4615.564 , 4676.0557, 4736.5586,
          #    4797.166 , 4857.695 , 4918.3604, 4978.8433, 5039.323 ,
          #    5099.9624, 5160.456 , 5220.999 , 5281.5293, 5342.0566,
          #    5402.6475]]]
          @endcode
        """
        _check_conv2d_params(dst, feature_map, weight, fm_shape,
                             kernel_shape, stride, pad, dilation, pad_value,
                             init_l1out)

        feature_map = ReIndexProxy(feature_map, fm_shape)
        weight = ReIndexProxy(weight, kernel_shape)

        fm_c1, fm_h, fm_w, fm_c0 = fm_shape
        k_c1, k_h, k_w, k_cout, k_c0 = kernel_shape

        fm_desc = FMDesc(fm_h, fm_w, k_h, k_w, fm_c1 * fm_c0,
                         feature_map.dtype,
                         dst.dtype, stride[0], stride[1], dilation[0],
                         dilation[1], pad)
        filter_desc = FilterDesc(k_c1, k_h, k_w, k_cout, k_c0, weight.dtype)

        tiling = gen_best_tiling(fm_desc, filter_desc,
                                 self.is_double_buffer_for_loop, self)

        hw_tile_block = tiling.howo_tile_block
        hw_thread_num = tiling.howo_tile_thread
        cin_tile_block = tiling.k_tile_block
        cin_thread_num = tiling.k_tile_thread
        cout_tile_block = tiling.cout_tile_block
        cout_thread_num = tiling.cout_tile_thread

        c_0 = 16 if feature_map.dtype == "float16" else 32
        block_size = 16

        cout = filter_desc.cout

        k_h = filter_desc.height
        k_w = filter_desc.width

        dilation_h, dilation_w = dilation

        hw_iter_num = math.ceil(fm_desc.h_o * fm_desc.w_o /
                                (hw_tile_block * block_size))
        cin_block_num = math.ceil(filter_desc.cin * k_h * k_w / c_0)
        cout_block_num = math.ceil(filter_desc.cout / block_size)
        cin_iter_num = math.ceil(
            filter_desc.cin * k_h * k_w / (cin_tile_block * c_0))
        cout_iter_num = math.ceil(filter_desc.cout /
                                  (cout_tile_block * block_size))

        ho_wo = fm_desc.h_o * fm_desc.w_o
        round_woho = math.ceil(ho_wo / block_size) * block_size

        # check dst tensor overflow
        _check_conv2d_tensor_overflow((round_woho, cout), dst, "dst")

        hw_block_num = math.ceil(ho_wo / block_size)
        round_woho = hw_block_num * block_size
        hw_has_tail = hw_block_num < hw_iter_num * hw_tile_block
        cin_has_tail = cin_block_num < cin_iter_num * cin_tile_block
        cout_has_tail = cout_block_num < cout_iter_num * cout_tile_block

        hw_tail_block = hw_block_num - (hw_iter_num - 1)*hw_tile_block
        cin_tail_block = cin_block_num - (cin_iter_num - 1)*cin_tile_block
        cout_tail_block = cout_block_num - (cout_iter_num - 1)*cout_tile_block

        from collections import namedtuple
        conv2d_param_class = namedtuple(
            "Conv2dParam",
            ["cin_iter_num", "cin_has_tail",
             "cout_has_tail", "cin_tail_block",
             "cout_tail_block", "cout_iter_num",
             "cin_tile_block", "cout_tile_block",
             "cout_block_num", "cin_thread_num",
             "cout_thread_num", "hw_iter_num",
             "hw_thread_num", "hw_tile_block",
             "hw_has_tail", "hw_tail_block",
             "weight", "block_size",
             "tiling", "c_0",
             "fm_desc", "feature_map", "k_w", "k_h",
             "dilation_w", "dilation_h",
             "pad_value",
             "round_woho", "init_l1out", "dst"])

        param = conv2d_param_class(
            cin_iter_num, cin_has_tail, cout_has_tail,
            cin_tail_block, cout_tail_block, cout_iter_num, cin_tile_block,
            cout_tile_block, cout_block_num, cin_thread_num, cout_thread_num,
            hw_iter_num, hw_thread_num, hw_tile_block, hw_has_tail,
            hw_tail_block, weight, block_size, tiling, c_0,
            fm_desc, feature_map, k_w, k_h, dilation_w,
            dilation_h, pad_value, round_woho, init_l1out, dst)

        if tiling.loop_mode == "nk":
            self._do_nk_tiling(param)
        elif tiling.loop_mode == "kn":
            self._do_kn_tiling(param)
        elif tiling.loop_mode == "mn":
            self._do_mn_tiling(param)
        elif tiling.loop_mode == "nm":
            self._do_nm_tiling(param)

    @source_info_decorator()
    @high_level_api_debug_decorator
    def fixpipe(self,  # pylint: disable=R0912, R0913, R0914, R0915
                dst, src, cburst_num, burst_len, dst_stride,
                src_stride, extend_params=None):
        """
        Processes the matrix computation result
        Description:
          Processes the matrix computation result, for example, adds an offset
          to and quantizes the computation result,
          and move the data from the L1OUT to the GM.
        Args:
          dst: A tensor of type float16, float32, or int32, for the start
          element of the destination operand. For
          details about data type restrictions, see Table Data type
          combination of src and dst. The scope is GM.
          After fixpipe processing, the extra data allocated during matrix
          computation is deleted in addition to the
          offset and quantization operations.
            - If this API is used to process the conv2d result, the format
            is [cout_blocks, howo, 16].
            - If this API is used to process the matmul result, the format
            is [N1, m, N0].
            - Note: For the meanings of cout_blocks and howo, see the
            parameter description of conv2d in Parameters.
          For the meanings of N1, m, and N0, see parameter description of
          matmul in Parameters.
          src: A tensor of type float32 or int32, for the start element of
          the source operand. For details about data
          type restrictions, see Table Data type combination of src and dst.
          The scope is L1OUT.
          The source operand is the result of matrix computation.
            - If this API is used to process the conv2d result, the format
            is [cout_blocks, round_howo, 16].
            - If this API is used to process the matmul result, the format
            is [N1, M, N0].
            - Note: For the meanings of cout_blocks and round_howo, see the
            parameter description of conv2d in Parameters.
          For the meanings of N1, M, and N0, see parameter description of
          matmul. in Parameters
          cburst_num: An immediate of type int specifying the number of
          bursts. The value range is [1, 4095].
            - If this API is used to process the conv2d result, the format
            is [cout_blocks, round_howo, 16], where,
            cburst_num is set to cout_blocks.
            - If this API is used to process the matmul result, the format
            is [N1, M, N0], where, cburst_num is set to N1.
            - Note: For the meanings of cout_blocks and round_howo, see the
            parameter description of conv2d in Parameters.
          For the meanings of N1, M, and N0, see parameter description of
          matmul in Parameters.
          burst_len: Burst length, in the unit of 32 bytes. The value is an
          even number within the range [2, 65535].The
          argument is an immediate of type int.For src, the valid data
          segment length of each burst is as follows:
            - If this API is used to process the conv2d result, the size is
            calculated as follows:
            howo * 16 * src_dtype_size/32 (unit: 32 bytes)
            - If this API is used to process the matmul result, the size is
            calculated as follows:
            m * N0 * src_dtype_size/32 (unit: 32 bytes)
          dst_stride: Tail-to-header stride between adjacent bursts of the dst
          operand tensor, in the unit of 32 bytes.
          The value range is [0, 65535]. The argument is an immediate
          of type int.
          src_stride: Tail-to-header stride between adjacent bursts of the
          dst operand tensor, in the unit of 256 elements.
          The value range is [0, 65535]. The argument is an immediate
          of type int.This parameter is reserved.
          To ensure data accuracy, pass 0.
          extend_params: A dictionary of extended parameters. Defaults to None.
           Four keys are supported: bias, quantize_params,
          element-wise-add, and relu.
            - key "bias"
              - value: Defaults to None, indicating bias disabled.
              To enable bias, specify the value as the start element of the
              bias operand. Has the same data type as
              src (a tensor of type int32 or float32). Has shape [Cout, ].
              - Cout: number of convolution kernels if src is the output of
              conv2d; or the length in the N dimension
              if src is the output of matmul.
              - The scope is L1.
              - Note: "bias" and "element-wise-add" are mutually exclusive.
            - key "quantize_params"
              - value: Defaults to None, indicating quantization disabled.
              If enabled, the value is a dictionary of two keys: "mode"
              and "mode_param".
              The mode argument is a string, for the quantization mode:
                - "int322fp16": int32 to float16 quantization
                - "fp322fp16": float32 to float16 quantization
              - mode_param has the following meanings:
                - A scalar of type float16 or an immediate of type float,
                for a single scale factor (supported only when
                mode is "int322fp16").
                - A tensor of type float16, with shape is [16] and scope L1.
                Applies to the 16 channels of cout (supported
                only when the bias and element-wise-add functions are disabled
                and the mode is "int322fp16").
                - If mode is set to "fp322fp16", pass None.
            - key "element-wise-add":
              - value: Defaults to None, indicating element-wise addition
              disabled. If enabled, the argument is a tensor
              of int32 or float32, for the start element of the bias operand.
              Has the same data type and shape as src.
              - The scope is L1.
              - Note: "bias" and "element-wise-add" are mutually exclusive.
            - key "relu":
              - value: A bool specifying whether to enable ReLU. Defaults
              to False.
              - Notes:
                - ReLU is supported only when the bias and element-wise-add
                functions are disabled.
                - ReLU is not supported when quantization is enabled, mode
                is set to "int322fp16", and the mode_param
                argument is a scalar of type float16 or an immediate of
                type float.

        Table:
          Data type combination of src and dst
        |src.dtype  |dst.dtype  |extend_params["quantize_params"]  |
        |----             |----        |----            |
        |float32          |float16     |"fp322fp16"     |
        |int32            |float16     |"int322fp16"    |
        |float32          |float32     |None            |
        |int32            |int32       |None            |

        Restrictions:
          - It takes a long time to perform step-by-step debugging. Therefore,
          step-by-step debugging is not recommended.
          - The functions enabled in extend_params is executed in the following
           sequence:


    \f$bias\rightarrow element-wise-add\rightarrow quantize\rightarrow relu\f$
          - This instruction must not be used together with the vectoring
          instructions.

        Returns:
            None

        Examples:
            #Example 1: src is of type int32 and dst is of type float16, bias
            #is disabled, and mode_param is a tensor argument.

              from te import tik
              tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
              # Define the tensors.
              feature_map_gm = tik_instance.Tensor("uint8", [1, 4, 4, 32],
                                name='feature_map_gm', scope=tik.scope_gm)
              weight_gm = tik_instance.Tensor("int8", [1, 2, 2, 32, 32],
                                name='weight_gm', scope=tik.scope_gm)
              deqscale_gm = tik_instance.Tensor("float16", [16],
                                name='deqscale_gm', scope=tik.scope_gm)
              dst_gm = tik_instance.Tensor("float16", [2, 9, 16],
                                name='dst_gm', scope=tik.scope_gm)
              feature_map = tik_instance.Tensor("uint8", [1, 4, 4, 32],
                                name='feature_map', scope=tik.scope_cbuf)
              weight = tik_instance.Tensor("int8", [1, 2, 2, 32, 32],
                                name='weight', scope=tik.scope_cbuf)
              deqscale = tik_instance.Tensor("float16", [16], name='deqscale',
                                scope=tik.scope_cbuf)
              dst_l1out = tik_instance.Tensor("int32", [2, 16, 16],
                                name='dst_l1out', scope=tik.scope_cbuf_out)
              # Move data from the GM to the source operand tensor.
              tik_instance.data_move(feature_map, feature_map_gm, 0, 1, 16
                                    , 0, 0)
              tik_instance.data_move(weight, weight_gm, 0, 1, 128, 0, 0)
              tik_instance.data_move(deqscale, deqscale_gm, 0, 1, 1, 0, 0)
              # Perform convolution.
              tik_instance.conv2d(dst_l1out, feature_map, weight, [1, 4, 4
              , 32], [1, 2, 2, 32, 32], [1, 1], [0, 0, 0, 0], [1, 1], 0)
              # Perform quantization using fixpipe.
              tik_instance.fixpipe(dst_gm, dst_l1out, 2, 18, 0, 0
              , extend_params={"bias": None, "quantize_params": {"mode":
              "int322fp16", "mode_param": deqscale}})
              tik_instance.BuildCCE(kernel_name="conv2d",
              inputs=[feature_map_gm, weight_gm, deqscale_gm],outputs=[dst_gm])

            #Inputs:
            #feature_map_gm:
            [[[[3, 2, 4, 2, ..., 4, 3]]]]
            #weight_gm:
            [[[[[0, -5, -3, ..., -4, -2]]]]]
            #deqscale_gm:
            [ 0.1214, -0.2238, ..., 0.4883, 0.2788]
            #Returns:
            #dst_gm:
            [[[-13.48, 39.38, -114.8, 30.38, ..., 9.766, -24.81]]]

        #Example 2: src is of type float32 and dst is of type float16, bias is
        #enabled, and mode_param is None.

          from te import tik
          tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
          # Define the tensors.
          feature_map_gm = tik_instance.Tensor("float16", [2, 4, 4, 16],
          name='feature_map_gm', scope=tik.scope_gm)
          weight_gm = tik_instance.Tensor("float16", [2, 2, 2, 16, 16],
          name='weight_gm', scope=tik.scope_gm)
          bias_gm = tik_instance.Tensor("float32", (16,), name='bias_gm',
          scope=tik.scope_gm)
          dst_gm = tik_instance.Tensor("float16", [1, 4, 16], name='dst_gm',
          scope=tik.scope_gm)
          feature_map = tik_instance.Tensor("float16", [2, 4, 4, 16],
          name='feature_map', scope=tik.scope_cbuf)
          weight = tik_instance.Tensor("float16", [2, 2, 2, 16, 16],
          name='weight', scope=tik.scope_cbuf)
          bias = tik_instance.Tensor("float32", (16,), name='bias',
          scope=tik.scope_cbuf)
          dst_l1out = tik_instance.Tensor("float32", [1, 16, 16],
          name='dst_l1out', scope=tik.scope_cbuf_out)
          # Move data from the GM to the source operand tensor.
          tik_instance.data_move(feature_map, feature_map_gm, 0, 1, 32, 0, 0)
          tik_instance.data_move(weight, weight_gm, 0, 1, 128, 0, 0)
          tik_instance.data_move(bias, bias_gm, 0, 1, 2, 0, 0)
          # Perform convolution.
          tik_instance.conv2d(dst_l1out, feature_map, weight, [2, 4, 4, 16]
          , [2, 2, 2, 16, 16], [1, 1], [0, 0, 0, 0], [2, 2], 0)
          # Perform bias and quantization using fixpipe.
          tik_instance.fixpipe(dst_gm, dst_l1out, 1, 8, 0, 0
          , extend_params={"bias": bias, "quantize_params": {"mode":
          "fp322fp16", "mode_param": None}})
          tik_instance.BuildCCE(kernel_name="conv2d", inputs=[feature_map_gm
          , weight_gm, bias_gm], outputs=[dst_gm])

          #Inputs:
          #feature_map_gm:
          [[[[0.0, 0.01, 0.02, 0.03, 0.04, ..., 5.09, 5.1, 5.11]]]]
          #weight_gm:
          [[[[[0.0, 0.01, 0.02, 0.03, 0.04, ..., 20.46, 20.47]]]]]
          #bias_gm:
          [0.0, 1.0, 2.0, 3.0, ..., 14.0, 15.0]
          #Returns:
          #    dst_gm:
          #    [[[3568., 3614.,3660., 3704., 3750., 3794., 3840., 3884., 3930.,
          #    3976., 4020., 4066., 4110., 4156., 4200., 4250.],
          #    [3754., 3802., 3850., 3900., 3948., 3996., 4044., 4094., 4140.,
          #     4188., 4240., 4290., 4336., 4384., 4430., 4480.],
          #    [4308., 4370., 4424., 4484., 4544., 4600., 4660., 4716., 4776.,
          #    4830., 4892., 4950., 5010., 5068., 5124., 5184.],
          #    [4496., 4556., 4616., 4680., 4740., 4804., 4864., 4924., 4988.,
          #    5050., 5108., 5172., 5230., 5296., 5356., 5416.]]]
        """
        # check operator
        TikCheckUtil.check_type_match(dst, Tensor,
                                      "dst should be Tensor, input"
                                      " type is %s" % type(dst))
        TikCheckUtil.check_type_match(src, Tensor,
                                      "src should be Tensor, input"
                                      " type is %s" % type(src))
        # check operator scope, waiting for confirmation of naming !!!!
        TikCheckUtil.check_equality(dst.scope, scope_gm,
                                    "dst's scope must be scope_gm, "
                                    "input scope is: %s" % dst.scope)
        TikCheckUtil.check_equality(src.scope, scope_cc,
                                    "src's scope must be L1_out, "
                                    "input scope is: %s" % src.scope)
        # check dtype
        dtype_str = DTYPE_MAP[src.dtype] + DTYPE_MAP[dst.dtype]
        TikCheckUtil.check_equality(api_check_support("tik.fixpipe",
                                                      dtype_str), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dtype_str, "fixpipe"))
        # check nburst
        TikCheckUtil.check_type_match(
            cburst_num, int, "cburst_num should be python int, input type "
                             "is %s" % type(cburst_num))
        TikCheckUtil.check_in_range(
            cburst_num, range(1, 4096),
            "cburst_num should be in range of [1, 4095], "
            "input value is %s" % cburst_num)
        # check burst_len
        TikCheckUtil.check_type_match(
            burst_len, int, "burst_len should be python int, input type "
                            "is %s" % type(burst_len))
        TikCheckUtil.check_in_range(
            burst_len, range(2, 65536),
            "burst_len should be in range of "
            "[2, 65535], input value is %s" % burst_len)
        TikCheckUtil.check_equality(
            burst_len % 2, 0, "burst_len should be even number, input "
                              "burst_len is %s" % burst_len)
        # check dst_stride
        TikCheckUtil.check_type_match(
            dst_stride, int, "dst_stride should be python int, input type "
                             "is %s" % type(dst_stride))
        TikCheckUtil.check_in_range(
            dst_stride, range(65536),
            "dst_stride should be in range of "
            "[0, 65535], input value is %s" % dst_stride)
        # check src_stride
        TikCheckUtil.check_type_match(
            src_stride, int, "src_stride should be python int, input type "
                             "is %s" % type(src_stride))
        TikCheckUtil.check_in_range(
            src_stride, range(65536),
            "src_stride should be in range of "
            "[0, 65535], input value is %s" % src_stride)
        # check extend_params
        if extend_params is not None:
            TikCheckUtil.check_type_match(
                extend_params, dict, "extend_params should be dict, input "
                                     "type is %s" % type(extend_params))
            if not set(list(extend_params.keys())).issubset(
                    {"bias", "quantize_params", "relu", "element-wise-add"}):
                TikCheckUtil.raise_error("input extend_params dict contains "
                                         "invalid key, please check!")

        fixpipe_config = FixpipeInfo(
            dst, src.dtype, cburst_num, burst_len, dst_stride,
            src_stride, extend_params)
        # check extend_params
        if dtype_str in ("s32f16", "f32f16"):
            TikCheckUtil.check_not_is(
                extend_params, None, "extend_params should not be None when "
                                     "src and dst dtype is %s" % dtype_str)
            TikCheckUtil.check_equality(
                fixpipe_config.has_deq, True,
                "extend_params 'quantize_params' should not be None when "
                "src and dst dtype is %s" % dtype_str)
        # bias cannot be used at the same time
        if fixpipe_config.has_ele_wise_bias and fixpipe_config.has_bias:
            TikCheckUtil.raise_error(
                "fixpipe doesn't support enable extend_params 'bias' and "
                "extend_params 'element-wise-add' at the same time")
        if extend_params is not None:
            _check_fixpipe_deq_params(
                dtype_str, extend_params, cburst_num,
                fixpipe_config)
            _check_fixpipe_bias_params(extend_params, src.dtype,
                                       cburst_num * fixpipe_config.frac_len)
            _check_fixpipe_ele_bias_params(extend_params, src,
                                           cburst_num, burst_len)
            _check_fixpipe_relu_params(extend_params, fixpipe_config)

        # check tensor overflow
        _check_fixpipe_tensor_overflow(dst, src, cburst_num, burst_len)

        # set deq_value
        if fixpipe_config.has_deq:
            deqscale = fixpipe_config.extend_params["quantize_params"].get(
                "mode_param")
            fixpipe_config.deq_value = deqscale
            if isinstance(deqscale, Tensor):
                deqscale_l1 = ReIndexProxy(deqscale,
                                           [fixpipe_config.frac_len, ])
                deqscale_value = self.Tensor(  # pylint: disable=E1101
                    "float16", [fixpipe_config.frac_len, ], name="deq_tmp",
                    scope=scope_ubuf)
                self.tensor_mov(  # pylint: disable=E1101
                    deqscale_value, deqscale_l1.flat_access(0), '', 1, 1, 0, 0)
                fixpipe_config.deq_value = deqscale_value

        bias_size = 0
        if fixpipe_config.has_bias:
            bias_size = cburst_num * fixpipe_config.frac_len * DTYPE_SIZE[
                src.dtype]
            bias_l1 = ReIndexProxy(extend_params['bias'],
                                   [1, cburst_num * fixpipe_config.frac_len])
            bias_ub = self.Tensor(  # pylint: disable=E1101
                src.dtype, [1, cburst_num * fixpipe_config.frac_len],
                name="bias_tmp", scope=scope_ubuf)
            self.tensor_mov(  # pylint: disable=E1101
                bias_ub, bias_l1.flat_access(0), '', cburst_num,
                fixpipe_config.frac_len * DTYPE_SIZE[
                    bias_l1.dtype] // ONE_BLK_SIZE, 0, 0)
            fixpipe_config.bias_value = ReIndexProxy(
                bias_ub, [1, cburst_num * fixpipe_config.frac_len])

        tiling = gen_fixpipe_tiling(cburst_num, fixpipe_config.howo_blocks,
                                    src.dtype, dst.dtype,
                                    fixpipe_config.has_bias, bias_size,
                                    fixpipe_config.has_ele_wise_bias, self,
                                    fixpipe_config.deq_value)

        def _make_fixpipe_m_code(l1out_i, howo_i, is_n_tail, is_m_tail):
            src_reindex = ReIndexProxy(src, (cburst_num,
                                             fixpipe_config.round_howo,
                                             fixpipe_config.frac_len))
            if dst.dtype in ("int8", "uint8"):
                self._s32_to_u8s8(src_reindex, fixpipe_config, tiling,
                                  l1out_i, howo_i, is_n_tail, is_m_tail)
            elif dst.dtype in ("int32", "float32"):
                self._s32_to_s32(src_reindex, fixpipe_config, tiling,
                                 l1out_i, howo_i, is_n_tail, is_m_tail)
            elif (src.dtype, dst.dtype) == ("float32", "float16"):
                self._float32_to_float16(src_reindex, fixpipe_config, tiling,
                                         l1out_i, howo_i, is_n_tail, is_m_tail)
            elif (src.dtype, dst.dtype) == ("int32", "float16"):
                self._s32_to_float16(src_reindex, fixpipe_config, tiling,
                                     l1out_i, howo_i, is_n_tail, is_m_tail)
            # f16->f16
            else:
                self._float16_to_float16(src_reindex, fixpipe_config, tiling,
                                         l1out_i, howo_i, is_n_tail, is_m_tail)

        def _make_fixpipe_n_code(l1out_i, is_n_tail=False):
            howo_iter_num = tiling.howo_iter_num
            howo_thread_num = tiling.howo_thread_num
            if fixpipe_config.howo_has_round or tiling.howo_has_tail:
                howo_iter_num -= 1
                if howo_iter_num < howo_thread_num:
                    howo_thread_num = 1
            with self.for_range(0, howo_iter_num,
                                thread_num=howo_thread_num) as howo_i:
                _make_fixpipe_m_code(l1out_i, howo_i, is_n_tail,
                                     is_m_tail=False)

            if fixpipe_config.howo_has_round or tiling.howo_has_tail:
                _make_fixpipe_m_code(l1out_i, howo_iter_num, is_n_tail,
                                     is_m_tail=True)

        l1out_thread_num = tiling.l1out_thread_num
        l1out_iter_num = tiling.l1out_iter_num
        if tiling.l1out_has_tail:
            l1out_iter_num -= 1
            if l1out_iter_num < l1out_thread_num:
                l1out_thread_num = 1
        with self.for_range(0, l1out_iter_num,
                            thread_num=l1out_thread_num) as l1out_i:
            _make_fixpipe_n_code(l1out_i, is_n_tail=False)
        if tiling.l1out_has_tail:
            _make_fixpipe_n_code(l1out_iter_num, is_n_tail=True)

    def _fixpipe_l1out_to_ub(  # pylint: disable=R0913, R0914
            self, input_tensor, fixpipe_config, tiling, l1out_i, howo_i,
            is_n_tail, is_m_tail, avoid_bankconflict=False, deq=False):
        '''

        :param input_tensor: l1out tensor
        :param fixpipe_config: fixpipe params
        :param tiling: fixpipe tiling info
        :param l1out_i: idx of cout tiling for loop
        :param howo_i: idx of howo tiling for loop
        :param avoid_bankconflict: flag indicating whether add 16 elements for
                                   each cout blk
        :param deq: flag indicating whether deq on the fly
        :return: ub_tmp tensor
        '''
        if deq:
            ub_tmp_dtype = "float16"
        else:
            ub_tmp_dtype = input_tensor.dtype

        en_relu = fixpipe_config.has_relu and \
                  not fixpipe_config.has_bias and \
                  not fixpipe_config.has_ele_wise_bias

        l1out_blocks_actual = tiling.l1out_tile_blocks
        howo_blocks_actual = tiling.howo_tile_blocks
        if is_n_tail:
            l1out_blocks_actual = tiling.l1out_tail_blk
        if is_m_tail:
            howo_blocks_actual = tiling.howo_tail_blk

        l1out_offset = \
            l1out_i * tiling.l1out_tile_blocks * \
            (fixpipe_config.howo_blocks + fixpipe_config.src_stride) * \
            fixpipe_config.frac_len * fixpipe_config.frac_len + \
            howo_i * tiling.howo_tile_blocks * fixpipe_config.frac_len * \
            fixpipe_config.frac_len
        l1out_ub_src_stride = \
            fixpipe_config.howo_blocks - howo_blocks_actual + \
            fixpipe_config.src_stride
        l1out_ub_dst_stride = \
            int(avoid_bankconflict) * fixpipe_config.frac_len * \
            DTYPE_SIZE[ub_tmp_dtype] // ONE_BLK_SIZE

        # shape: cout_blks, round_howo + avoid_bc_pad_1, 16
        l0c_dim = l1out_blocks_actual * \
                  (howo_blocks_actual * fixpipe_config.frac_len +
                   int(avoid_bankconflict)) * \
                  fixpipe_config.frac_len

        # round to 256B for following elementwise vector instr
        if input_tensor.dtype in ("float32", "int32"):
            l0c_dim = ceil_div(l0c_dim, 64) * 64

        fixpipe_config.l0c_shape = [l0c_dim, ]
        ub_tmp = self.Tensor(ub_tmp_dtype, (l0c_dim,),  # pylint: disable=E1101
                             name="fixpipe_ub_tmp", scope=scope_ubuf)
        self.tensor_mov(  # pylint: disable=E1101
            ub_tmp, input_tensor.flat_access(l1out_offset), 'm',
            l1out_blocks_actual, howo_blocks_actual, l1out_ub_dst_stride,
            l1out_ub_src_stride, deqscale=fixpipe_config.deq_value,
            relu=en_relu)

        return ub_tmp

    def _fixpipe_bias(  # pylint: disable=R0913, R0914
            self, input_tensor, fixpipe_config, l1out_iter_num, tiling,
            is_n_tail, is_m_tail, avoid_bankconflict=False):
        '''

        :param input_tensor: l1out-> ub tensor
        :param fixpipe_config: fixpipe params
        :param tiling: fixpipe tiling info
        :param l1out_i: idx of cout tiling for loop
        :return: ub_bias_result
        '''
        l1out_tile_blocks = tiling.l1out_tile_blocks
        if is_n_tail:
            l1out_tile_blocks = tiling.l1out_tail_blk

        howo_tile_blocks = tiling.howo_tile_blocks
        if is_m_tail:
            howo_tile_blocks = tiling.howo_tail_blk

        howo_data_index = howo_tile_blocks * fixpipe_config.frac_len + \
                          int(avoid_bankconflict)

        bias_cal_blocks = howo_tile_blocks * 16
        bias_cal_repeat_times = bias_cal_blocks // 8
        bias_cal_repeat_nums = bias_cal_repeat_times // 255
        bias_cal_tail_repeat_times = bias_cal_repeat_times % 255
        bias_cal_tail_blocks = bias_cal_blocks % 8

        ub_bias_result = self.Tensor(  # pylint: disable=E1101
            input_tensor.dtype, input_tensor.shape, name="fixpipe_bias_result",
            scope=scope_ubuf)

        if l1out_tile_blocks:
            with self.for_range(0, l1out_tile_blocks) as cout_idx:
                bias_index = l1out_iter_num * tiling.l1out_tile_blocks * \
                             fixpipe_config.frac_len + cout_idx * \
                             fixpipe_config.frac_len
                if fixpipe_config.src_dtype == "float16":
                    if bias_cal_repeat_nums:
                        with self.for_range(0, bias_cal_repeat_times) as i:
                            data_index = \
                                cout_idx * howo_data_index * \
                                fixpipe_config.frac_len + i * 255 * 128
                            self.vadd(  # pylint: disable=E1101
                                128, ub_bias_result[data_index],
                                fixpipe_config.bias_value.flat_access(
                                    bias_index), input_tensor[data_index],
                                255, 1, 0, 1, 8, 0, 8)

                    if bias_cal_tail_repeat_times:
                        data_index = cout_idx * howo_data_index * \
                                     fixpipe_config.frac_len + \
                                     bias_cal_repeat_nums * 255 * 128
                        self.vadd(  # pylint: disable=E1101
                            128, ub_bias_result[data_index],
                            fixpipe_config.bias_value.flat_access(bias_index),
                            input_tensor[data_index],
                            bias_cal_tail_repeat_times, 1, 0, 1, 8, 0, 8)

                    if bias_cal_tail_blocks:
                        data_index = cout_idx * howo_data_index * \
                                     fixpipe_config.frac_len * \
                                     bias_cal_repeat_nums * 255 * 128 + \
                                     bias_cal_tail_repeat_times * 128
                        mask = bias_cal_tail_blocks * 16
                        self.vadd(  # pylint: disable=E1101
                            mask,
                            ub_bias_result[data_index],
                            fixpipe_config.bias_value.flat_access(bias_index),
                            input_tensor[data_index],
                            1, 1, 0, 1, 8, 0, 8)
                else:
                    bias_low_index = bias_index
                    bias_high_index = bias_index + fixpipe_config.frac_len // 2

                    half_index = fixpipe_config.frac_len // 2
                    if bias_cal_repeat_nums:
                        with self.for_range(0, bias_cal_repeat_nums) as i:
                            data_index = cout_idx * howo_data_index * \
                                         fixpipe_config.frac_len + \
                                         i * 255 * 128
                            self.vadd(  # pylint: disable=E1101
                                64,
                                ub_bias_result[data_index],
                                fixpipe_config.bias_value.flat_access(
                                    bias_low_index),
                                input_tensor[data_index],
                                255, 2, 0, 2, 16, 0, 16)
                            self.vadd(  # pylint: disable=E1101
                                64,
                                ub_bias_result[data_index + half_index],
                                fixpipe_config.bias_value.flat_access(
                                    bias_high_index),
                                input_tensor[data_index + half_index],
                                255, 2, 0, 2, 16, 0, 16)

                    if bias_cal_tail_repeat_times:
                        data_index = \
                            cout_idx * howo_data_index * \
                            fixpipe_config.frac_len + bias_cal_repeat_nums * \
                            255 * 128
                        self.vadd(  # pylint: disable=E1101
                            64,
                            ub_bias_result[data_index],
                            fixpipe_config.bias_value.flat_access(
                                bias_low_index),
                            input_tensor[data_index],
                            bias_cal_tail_repeat_times,
                            2, 0, 2, 16, 0, 16)

                        self.vadd(  # pylint: disable=E1101
                            64,
                            ub_bias_result[data_index + half_index],
                            fixpipe_config.bias_value.flat_access(
                                bias_high_index),
                            input_tensor[data_index + half_index],
                            bias_cal_tail_repeat_times,
                            2, 0, 2, 16, 0, 16)

                    if bias_cal_tail_blocks:
                        data_index = cout_idx * howo_data_index * \
                                     fixpipe_config.frac_len + \
                                     bias_cal_repeat_nums * 255 * 128 + \
                                     bias_cal_tail_repeat_times * 128
                        mask = bias_cal_tail_blocks * 8
                        self.vadd(  # pylint: disable=E1101
                            mask,
                            ub_bias_result[data_index],
                            fixpipe_config.bias_value.flat_access(
                                bias_low_index),
                            input_tensor[data_index],
                            1, 2, 0, 2, 16, 0, 16)
                        self.vadd(  # pylint: disable=E1101
                            mask,
                            ub_bias_result[data_index + half_index],
                            fixpipe_config.bias_value.flat_access(
                                bias_high_index),
                            input_tensor[data_index + half_index],
                            1, 2, 0, 2, 16, 0, 16)

        return ub_bias_result

    def _fixpipe_ele_bias(self,  # pylint: disable=R0913, R0914
                          input_tensor, fixpipe_config, tiling, l1out_i,
                          howo_i, is_n_tail, is_m_tail,
                          avoid_bankconflict=False):
        bias_data_l1 = ReIndexProxy(
            fixpipe_config.extend_params['element-wise-add'],
            (fixpipe_config.cburst_num, fixpipe_config.round_howo,
             fixpipe_config.frac_len))
        dtype = input_tensor.dtype
        l1out_blocks_actual = tiling.l1out_tile_blocks
        howo_blocks_actual = tiling.howo_tile_blocks
        if is_n_tail:
            l1out_blocks_actual = tiling.l1out_tail_blk
        if is_m_tail:
            howo_blocks_actual = tiling.howo_tail_blk

        l1out_offset = \
            l1out_i * tiling.l1out_tile_blocks * \
            (fixpipe_config.howo_blocks + fixpipe_config.src_stride) * \
            fixpipe_config.frac_len * fixpipe_config.frac_len + \
            howo_i * tiling.howo_tile_blocks * fixpipe_config.frac_len * \
            fixpipe_config.frac_len
        l1out_ub_src_stride = (fixpipe_config.howo_blocks -
                               howo_blocks_actual)*fixpipe_config.frac_len * \
                              fixpipe_config.frac_len*DTYPE_SIZE[dtype] // \
                              ONE_BLK_SIZE
        l1out_ub_dst_stride = \
            int(avoid_bankconflict) * fixpipe_config.frac_len * \
            DTYPE_SIZE[dtype] // ONE_BLK_SIZE

        # shape: cout_blks, round_howo + avoid_bc_pad_1, 16
        l0c_dim = l1out_blocks_actual * \
                  (howo_blocks_actual * fixpipe_config.frac_len +
                   int(avoid_bankconflict)) * \
                  fixpipe_config.frac_len

        # round to 256B for following elementwise vector instr
        if input_tensor.dtype in ("float32", "int32"):
            l0c_dim = ceil_div(l0c_dim, 64) * 64

        ub_tmp = self.Tensor(dtype, (l0c_dim,),  # pylint: disable=E1101
                             name="fixpipe_ub_tmp", scope=scope_ubuf)
        self.tensor_mov(  # pylint: disable=E1101
            ub_tmp, bias_data_l1.flat_access(l1out_offset), '',
            l1out_blocks_actual, howo_blocks_actual*fixpipe_config.frac_len *
            16*DTYPE_SIZE[dtype] // ONE_BLK_SIZE, l1out_ub_dst_stride,
            l1out_ub_src_stride)
        res_tensor = self.Tensor( # pylint: disable=E1101
            input_tensor.dtype, input_tensor.shape,
            name="fixpipe_ele_bias_result", scope=scope_ubuf)
        total_elements = reduce_mul(input_tensor.shape)
        elements_per_rep = 64
        max_repeat_times = MAX_REPEAT_TIMES - 1
        total_repeats = total_elements // elements_per_rep
        vadd_repeat_batch = total_repeats // max_repeat_times
        vadd_repeat_tail = total_repeats % max_repeat_times
        with self.for_range(0, vadd_repeat_batch) as vadd_sub_i:
            offset = vadd_sub_i * max_repeat_times * elements_per_rep
            self.vadd(elements_per_rep,  # pylint: disable=E1101
                      res_tensor[offset:], input_tensor[offset:],
                      ub_tmp[offset:], max_repeat_times, 1, 1, 1, 8, 8, 8)
        if vadd_repeat_tail > 0:
            offset = vadd_repeat_batch * max_repeat_times * elements_per_rep
            self.vadd(elements_per_rep,  # pylint: disable=E1101
                      res_tensor[offset:], input_tensor[offset:],
                      ub_tmp[offset:], vadd_repeat_tail, 1, 1, 1, 8, 8, 8)
        return res_tensor

    def _fixpipe_relu(self, input_tensor):
        """This function is temporarily not available for reason:
            vrelu(v100) not support float32 and int32 dtype
        """
        ub_relu_result = self.Tensor(  # pylint: disable=E1101
            input_tensor.dtype, input_tensor.shape,
            name="fixpipe_relu_result", scope=scope_ubuf)
        total_elements = reduce_mul(input_tensor.shape)
        elements_per_rep = 64
        total_repeats = total_elements // elements_per_rep
        relu_repeat_batch = total_repeats // 255
        relu_repeat_tail = total_repeats % 255
        blk_stride = 1
        rep_stride = 8
        with self.for_range(0, relu_repeat_batch) as relu_sub_i:
            offset = relu_sub_i * 255 * elements_per_rep
            self.vrelu(  # pylint: disable=E1101
                elements_per_rep, "", ub_relu_result[offset],
                input_tensor[offset], 255, blk_stride,
                blk_stride, rep_stride, rep_stride)

        offset = relu_repeat_batch * 255 * elements_per_rep
        self.vrelu(  # pylint: disable=E1101
            elements_per_rep, "", ub_relu_result[offset],
            input_tensor[offset], relu_repeat_tail, blk_stride, blk_stride,
            rep_stride, rep_stride)
        return ub_relu_result

    def _fixpipe_deq_b32_to_f16(self, ub_bias_result, fixpipe_config, tiling):
        """realize fixpipe conv from s32/f32 to f16

        Parameters
        ----------
        ub_bias_result : tensor, scope is ub, if have bias, should after bias,
                         tensor size should be multiple of 256B
        fixpipe_config : FixpipeInfo
        tiling : fixpipe tiling

        Returns
        -------
        ub_deq_fp16_result : Tensor, have same shape as ub_bias_result
        """
        ub_deq_fp16_result = self.Tensor(  # pylint: disable=E1101
            "float16", fixpipe_config.l0c_shape,
            name="fixpipe_deq_fp16_result", scope=scope_ubuf)
        total_elements = reduce_mul(fixpipe_config.l0c_shape)
        elements_per_rep = 64
        total_repeats = total_elements // elements_per_rep
        deq_repeat_batch = total_repeats // tiling.max_repeat
        deq_repeat_tail = total_repeats % tiling.max_repeat
        blk_stride = 1
        with self.for_range(0, deq_repeat_batch) as deq_sub_i:
            offset = deq_sub_i * tiling.max_repeat * elements_per_rep
            self.vconv(  # pylint: disable=E1101
                elements_per_rep, "", ub_deq_fp16_result[offset],
                ub_bias_result[offset], tiling.max_repeat, blk_stride,
                blk_stride, 4, 8, deqscale=fixpipe_config.extend_params[
                    'quantize_params'].get("mode_param"))
        if deq_repeat_tail > 0:
            offset = deq_repeat_batch * tiling.max_repeat * elements_per_rep
            self.vconv(  # pylint: disable=E1101
                elements_per_rep, "", ub_deq_fp16_result[offset],
                ub_bias_result[offset], deq_repeat_tail, blk_stride,
                blk_stride,
                4, 8, deqscale=fixpipe_config.extend_params[
                    'quantize_params'].get("mode_param"))
        return ub_deq_fp16_result

    def _fixpipe_deq_fp16_to_u8s8(  # pylint: disable=R0913, R0914
            self, ub_deq_fp16_result, fixpipe_config, tiling, is_n_tail,
            is_m_tail):
        """realize fixpipe conv from f16 to u8/s8

        Parameters
        ----------
        ub_deq_fp16_result: tensor, shape is [cout_blk, howo, 16]
        fixpipe_config: fixpipe config info, FixpipeInfo
        tiling: fixpipe tiling info
        is_n_tail: mark whether it is n-direction tail block
        is_m_tail: mark whether it is m-direction tail block

        Returns
        -------
        ub_deq_result: Tensor, shape is [cout_blk//2, howo, 32]
        """
        l1out_blocks_actual = tiling.l1out_tile_blocks
        howo_blocks_actual = tiling.howo_tile_blocks
        if is_n_tail:
            l1out_blocks_actual = tiling.l1out_tail_blk
        if is_m_tail:
            howo_blocks_actual = tiling.howo_tail_blk

        frac_len = fixpipe_config.frac_len
        _32b_dst_gap = 1
        ub_deq_result = self.Tensor(  # pylint: disable=E1101
            fixpipe_config.dst_dtype, fixpipe_config.l0c_shape,
            name="fixpipe_deq_result", scope=scope_ubuf)
        batch_num = l1out_blocks_actual // 8
        batch_tail = l1out_blocks_actual % 8
        batch_loop = (howo_blocks_actual * frac_len + 1) // \
                     tiling.max_repeat
        batch_loop_tail = (howo_blocks_actual * frac_len + 1) % \
                          tiling.max_repeat
        blk_stride = howo_blocks_actual * frac_len + _32b_dst_gap
        with self.for_range(0, batch_num) as batch_i:
            offset = batch_i * 8 * (howo_blocks_actual * frac_len +
                                    _32b_dst_gap) * frac_len
            with self.for_range(0, batch_loop) as batch_sub_i:
                sub_block_offset = tiling.max_repeat * batch_sub_i
                self.vconv(128, "",  # pylint: disable=E1101
                           ub_deq_result[offset + sub_block_offset * 32],
                           ub_deq_fp16_result[offset + sub_block_offset * 16],
                           tiling.max_repeat, blk_stride, blk_stride, 1, 1)
            if batch_loop_tail > 0:
                sub_block_offset = tiling.max_repeat * batch_loop
                self.vconv(  # pylint: disable=E1101
                    128, "", ub_deq_result[offset + sub_block_offset * 32],
                    ub_deq_fp16_result[offset + sub_block_offset * 16],
                    batch_loop_tail, blk_stride, blk_stride, 1, 1)

        if batch_tail > 0:
            partial_mask = batch_tail * 16
            offset = batch_num * 8 * (howo_blocks_actual * frac_len +
                                      _32b_dst_gap) * frac_len
            with self.for_range(0, batch_loop) as batch_sub_i:
                sub_block_offset = tiling.max_repeat * batch_sub_i
                self.vconv(partial_mask, "",  # pylint: disable=E1101
                           ub_deq_result[offset + sub_block_offset * 32],
                           ub_deq_fp16_result[offset + sub_block_offset * 16],
                           tiling.max_repeat, blk_stride, blk_stride, 1, 1)
            if batch_loop_tail > 0:
                sub_block_offset = tiling.max_repeat * batch_loop
                self.vconv(partial_mask, "",  # pylint: disable=E1101
                           ub_deq_result[offset + sub_block_offset * 32],
                           ub_deq_fp16_result[offset + sub_block_offset * 16],
                           batch_loop_tail, blk_stride, blk_stride, 1, 1)
        return ub_deq_result

    def _fixpipe_ub_to_out(  # pylint: disable=R0913, R0914
            self, input_tensor, fixpipe_config, tiling, l1out_i, howo_i,
            is_n_tail, is_m_tail, avoid_bankconflict=False):
        dst_stride = fixpipe_config.dst_stride

        if fixpipe_config.dst.dtype in ("uint8", "int8"):
            extra_len = dst_stride
            dst = ReIndexProxy(
                fixpipe_config.dst, [fixpipe_config.cburst_num // 2,
                                     fixpipe_config.howo + extra_len, 32])
        else:
            extra_len = dst_stride * ONE_BLK_SIZE // \
                        DTYPE_SIZE[fixpipe_config.dst.dtype] // \
                        fixpipe_config.frac_len
            dst = ReIndexProxy(
                fixpipe_config.dst,
                [fixpipe_config.cburst_num, fixpipe_config.howo + extra_len,
                 16])

        l1out_blocks_actual = tiling.l1out_tile_blocks
        howo_blocks_actual = tiling.howo_tile_blocks
        if is_n_tail:
            l1out_blocks_actual = tiling.l1out_tail_blk
        if is_m_tail:
            howo_blocks_actual = tiling.howo_tail_blk

        c_0 = fixpipe_config.c_0
        frac_len = fixpipe_config.frac_len
        output_tile_burst = l1out_blocks_actual
        if tiling.vconv_merge_channel:
            output_tile_burst = output_tile_burst // 2
        dst_offset = \
            l1out_i * tiling.l1out_tile_blocks * \
            (fixpipe_config.howo + extra_len) * frac_len + howo_i * \
            tiling.howo_tile_blocks * frac_len * c_0
        real_dst_stride = \
            dst_stride + (fixpipe_config.howo - howo_blocks_actual * frac_len)\
            * c_0 * DTYPE_SIZE[fixpipe_config.dst_dtype]// ONE_BLK_SIZE
        real_dst_stride = max(real_dst_stride, 0)  # bypass compile error
        tail_dst_stride = \
            dst_stride + (fixpipe_config.round_howo - howo_blocks_actual *
                          frac_len) * c_0 * \
            DTYPE_SIZE[fixpipe_config.dst_dtype] // ONE_BLK_SIZE
        bc_pad_dst_gap = int(avoid_bankconflict) * c_0 * \
                         DTYPE_SIZE[fixpipe_config.dst_dtype] // ONE_BLK_SIZE
        real_src_stride = bc_pad_dst_gap
        tail_src_stride = \
            bc_pad_dst_gap + (fixpipe_config.round_howo - fixpipe_config.howo)\
            * c_0 * DTYPE_SIZE[fixpipe_config.dst_dtype] // ONE_BLK_SIZE

        real_output_burst_len = \
            howo_blocks_actual * frac_len * c_0 * \
            DTYPE_SIZE[fixpipe_config.dst_dtype] // ONE_BLK_SIZE
        tail_output_burst_len = \
            (howo_blocks_actual * frac_len -
             (fixpipe_config.round_howo - fixpipe_config.howo)) * c_0 * \
            DTYPE_SIZE[fixpipe_config.dst_dtype] // ONE_BLK_SIZE

        if is_m_tail:
            self.tensor_mov(  # pylint: disable=E1101
                dst.flat_access(dst_offset), input_tensor, "",
                output_tile_burst, tail_output_burst_len, tail_dst_stride,
                tail_src_stride)
        else:
            self.tensor_mov(  # pylint: disable=E1101
                dst.flat_access(dst_offset), input_tensor, "",
                output_tile_burst, real_output_burst_len, real_dst_stride,
                real_src_stride)

    def _s32_to_s32(  # pylint: disable=R0913
            self, input_tensor, fixpipe_config, tiling, l1out_i, howo_i,
            is_n_tail, is_m_tail):
        ub_data = self._fixpipe_l1out_to_ub(input_tensor, fixpipe_config,
                                            tiling, l1out_i, howo_i, is_n_tail,
                                            is_m_tail)
        if fixpipe_config.has_bias:
            ub_data = self._fixpipe_bias(ub_data, fixpipe_config, l1out_i,
                                         tiling, is_n_tail, is_m_tail)
        elif fixpipe_config.has_ele_wise_bias:
            ub_data = self._fixpipe_ele_bias(
                ub_data, fixpipe_config, tiling, l1out_i, howo_i,
                is_n_tail, is_m_tail)
        self._fixpipe_ub_to_out(ub_data, fixpipe_config, tiling, l1out_i,
                                howo_i, is_n_tail, is_m_tail)

    def _float16_to_float16(  # pylint: disable=R0913
            self, input_tensor, fixpipe_config, tiling, l1out_i, howo_i,
            is_n_tail, is_m_tail):
        ub_data = self._fixpipe_l1out_to_ub(input_tensor, fixpipe_config,
                                            tiling, l1out_i, howo_i, is_n_tail,
                                            is_m_tail)
        if fixpipe_config.has_bias:
            ub_data = self._fixpipe_bias(ub_data, fixpipe_config, l1out_i,
                                         tiling, is_n_tail, is_m_tail)
        elif fixpipe_config.has_ele_wise_bias:
            ub_data = self._fixpipe_ele_bias(
                ub_data, fixpipe_config, tiling, l1out_i, howo_i,
                is_n_tail, is_m_tail)
        self._fixpipe_ub_to_out(ub_data, fixpipe_config, tiling, l1out_i,
                                howo_i, is_n_tail, is_m_tail)

    def _float32_to_float16(  # pylint: disable=R0913
            self, input_tensor, fixpipe_config, tiling, l1out_i, howo_i,
            is_n_tail, is_m_tail):
        if fixpipe_config.has_bias or fixpipe_config.has_ele_wise_bias:
            ub_data = self._fixpipe_l1out_to_ub(
                input_tensor, fixpipe_config, tiling, l1out_i, howo_i,
                is_n_tail, is_m_tail)
            if fixpipe_config.has_bias:
                ub_data = self._fixpipe_bias(ub_data, fixpipe_config, l1out_i,
                                             tiling, is_n_tail, is_m_tail)
            elif fixpipe_config.has_ele_wise_bias:
                ub_data = self._fixpipe_ele_bias(
                    ub_data, fixpipe_config, tiling, l1out_i, howo_i,
                    is_n_tail, is_m_tail)
            ub_data = self._fixpipe_deq_b32_to_f16(ub_data, fixpipe_config,
                                                   tiling)
            self._fixpipe_ub_to_out(ub_data, fixpipe_config, tiling,
                                    l1out_i, howo_i, is_n_tail, is_m_tail)
        else:
            # deq on the fly
            ub_data = self._fixpipe_l1out_to_ub(input_tensor, fixpipe_config,
                                                tiling, l1out_i, howo_i,
                                                is_n_tail, is_m_tail, deq=True)
            self._fixpipe_ub_to_out(ub_data, fixpipe_config, tiling, l1out_i,
                                    howo_i, is_n_tail, is_m_tail)

    def _s32_to_float16(  # pylint: disable=R0913
            self, input_tensor, fixpipe_config, tiling, l1out_i, howo_i,
            is_n_tail, is_m_tail):
        if fixpipe_config.has_bias or fixpipe_config.has_ele_wise_bias:
            ub_data = self._fixpipe_l1out_to_ub(
                input_tensor, fixpipe_config, tiling, l1out_i, howo_i,
                is_n_tail, is_m_tail)
            if fixpipe_config.has_bias:
                ub_data = self._fixpipe_bias(ub_data, fixpipe_config, l1out_i,
                                             tiling, is_n_tail, is_m_tail)
            elif fixpipe_config.has_ele_wise_bias:
                ub_data = self._fixpipe_ele_bias(
                    ub_data, fixpipe_config, tiling, l1out_i, howo_i,
                    is_n_tail, is_m_tail)
            ub_data = self._fixpipe_deq_b32_to_f16(ub_data, fixpipe_config,
                                                   tiling)
            self._fixpipe_ub_to_out(ub_data, fixpipe_config, tiling,
                                    l1out_i, howo_i, is_n_tail, is_m_tail)
        else:
            # deq on the fly
            ub_data = self._fixpipe_l1out_to_ub(input_tensor, fixpipe_config,
                                                tiling, l1out_i, howo_i,
                                                is_n_tail, is_m_tail, deq=True)
            self._fixpipe_ub_to_out(ub_data, fixpipe_config, tiling, l1out_i,
                                    howo_i, is_n_tail, is_m_tail)

    def _s32_to_u8s8(  # pylint: disable=R0913
            self, input_tensor, fixpipe_config, tiling, l1out_i, howo_i,
            is_n_tail, is_m_tail):
        if fixpipe_config.has_bias or fixpipe_config.has_ele_wise_bias:
            ub_data = self._fixpipe_l1out_to_ub(
                input_tensor, fixpipe_config, tiling, l1out_i, howo_i,
                is_n_tail, is_m_tail, avoid_bankconflict=True)
            if fixpipe_config.has_bias:
                ub_data = self._fixpipe_bias(ub_data, fixpipe_config, l1out_i,
                                             tiling, is_n_tail, is_m_tail,
                                             avoid_bankconflict=True)
            elif fixpipe_config.has_ele_wise_bias:
                ub_data = self._fixpipe_ele_bias(
                    ub_data, fixpipe_config, tiling, l1out_i, howo_i,
                    is_n_tail, is_m_tail, avoid_bankconflict=True)
            ub_data = self._fixpipe_deq_b32_to_f16(ub_data, fixpipe_config,
                                                   tiling)
            ub_data = self._fixpipe_deq_fp16_to_u8s8(
                ub_data, fixpipe_config, tiling, is_n_tail, is_m_tail)
            self._fixpipe_ub_to_out(ub_data, fixpipe_config, tiling,
                                    l1out_i, howo_i, is_n_tail, is_m_tail,
                                    avoid_bankconflict=True)
        else:
            # deq on the fly
            ub_data = self._fixpipe_l1out_to_ub(input_tensor, fixpipe_config,
                                                tiling, l1out_i, howo_i,
                                                is_n_tail, is_m_tail,
                                                avoid_bankconflict=True,
                                                deq=True)
            ub_data = self._fixpipe_deq_fp16_to_u8s8(ub_data, fixpipe_config,
                                                     tiling, is_n_tail,
                                                     is_m_tail)
            self._fixpipe_ub_to_out(ub_data, fixpipe_config, tiling, l1out_i,
                                    howo_i, is_n_tail, is_m_tail,
                                    avoid_bankconflict=True)

    def _fixpipe_ub_to_l1out(  # pylint: disable=R0913
            self, input_tensor, ub_data, fixpipe_config, tiling, l1out_i,
            howo_i, is_n_tail, is_m_tail):
        l1out_blocks_actual = tiling.l1out_tile_blocks
        howo_blocks_actual = tiling.howo_tile_blocks
        if is_n_tail:
            l1out_blocks_actual = tiling.l1out_tail_blk
        if is_m_tail:
            howo_blocks_actual = tiling.howo_tail_blk

        l1out_offset = \
            l1out_i * tiling.l1out_tile_blocks * \
            (fixpipe_config.howo_blocks + fixpipe_config.src_stride) * \
            fixpipe_config.frac_len * fixpipe_config.frac_len + \
            howo_i * tiling.howo_tile_blocks * fixpipe_config.frac_len * \
            fixpipe_config.frac_len

        ub_l1out_dst_stride = fixpipe_config.howo_blocks - howo_blocks_actual \
                              + fixpipe_config.src_stride
        ub_l1out_src_stride = 0

        self.tensor_mov(  # pylint: disable=E1101
            input_tensor.flat_access(l1out_offset), ub_data, 'm',
            l1out_blocks_actual, howo_blocks_actual, ub_l1out_dst_stride,
            ub_l1out_src_stride)

    @source_info_decorator()
    @high_level_api_debug_decorator
    def matmul(self,  # pylint: disable=R0913, C0103
               dst, a, b, m, k, n, init_l1out=True):
        """
        Multiplies matrix a by matrix b and outputs a result tensor.
        Description:
          For details about the data type restrictions, see Table Data type
          combination of a, b, and dst.
        Args:
          dst: Start element of the destination operand. For details about the
          data type restrictions, see Table Data
          type combination of a, b, and dst. The scope is L1OUT.
          A tensor in the format of [N1, M, N0], where, N = N1 * N0
            - N1 = Ceiling(n/N0). Ceiling indicates the round-up operation.
            - M = Ceiling(m/M0) * M0, where, M0 = 16.
            - N0 =16
          a: Source operand, matrix tensor a. For details about the data type
          restrictions, see Table Data type
          combination of a, b, and dst. The scope is L1.
          A tensor in the format of [K1, M, K0], where, K = K1 * K0
            - K1 = Ceiling(k/K0). Ceiling indicates the round-up operation.
            - M = Ceiling(m/M0) * M0, where, M0 = 16.
            - K0: If matrix a is of type float16, K0 = 16 (an immediate of type
             int); if matrix a is of type int8/uint8,
            K0 = 32 (an immediate of type int).
          b: Source operand, matrix tensor b. For details about the data type
          restrictions, see Table Data type
          combination of a, b, and dst. The scope is L1.
          A tensor in the format of [K1, N, K0], where, K = K1 * K0
            - K1 = Ceiling(k/K0). Ceiling indicates the round-up operation.
            - K0: If matrix b is of type float16, K0 = 16 (an immediate of type
             int); if matrix b is of type int8,
            K0 = 32 (an immediate of type int).
            - N = N1* N0, where, N1 = Ceiling(n/N0), N0 = 16.
          m: An immediate of type int specifying the valid height of matrix a.
          The value range is [1, 4096].Note: The m
          argument does not need to be rounded up to a multiple of16.
          k: An immediate of type int specifying the valid width of matrix a
          and the valid height of matrix b.
            - If matrix a is of type float16, the value range is [1, 16384],
            - If matrix a is of type int8/uint8, the value range is [1, 32768],
            - Note: The k argument does not need to be rounded up to
            a multiple of16.
          n: An immediate of type int specifying the valid width of matrix b.
          The value range is [1, 4096].
            - Note: The n argument does not need to be rounded up to
            a multiple of16.
          init_l1out: A bool specifying whether to initialize
          dst . Defaults to True.
            - True: The dst initial matrix will be overwritten by the
            computation result.
            - False: The dst initial matrix stores the previous matmul
            result and will be accumulated with the new matmul result.

        Table:
          Data type combination of src and dst
        |a.dtype          |b.dtype     |dst.dtype       |
        |----             |----        |----            |
        |int8             |int8        |int32           |
        |uint8            |int8        |int32           |
        |float16          |float16     |float32         |

        Restrictions:
          - It takes a long time to perform step-by-step debugging. Therefore,
          step-by-step debugging is not recommended.
          - The tensor[immediate] or tensor[scalar] format indicates a
          1-element tensor. To specify the computation start
          (with offset),use the tensor [immediate:] or tensor [scalar:] format.
          - For Ascend 310 AI Processor, the start addresses of the source
          operands a and b of the instruction must be
          512-byte aligned. For example, when tensor slices are input and
          the source operand is of type float16,
          tensor[256:] can be used. However, tensor[2:] does not meet the
          alignment requirement, and an unknown error
          may occur.
          - For Ascend 910 AI Processor, the start addresses of the source
          operands a and b of the instruction must be
          512-byte aligned. For example, when tensor slices are input and
          the source operand is of type float16,
          tensor[256:] can be used. However, tensor[2:] does not meet the
          alignment requirement, and an unknown error
          may occur.
          - For Ascend 610 AI Processor (AI Core), the start addresses of the
          source operands a and b of the instruction
          must be 32-byte aligned. For example, when tensor slices are input
          and the source operand is of type float16,
          tensor[16:] can be used. However, tensor[2:] does not meet the
          alignment requirement, and an unknown error
          may occur.
          - The start address of the destination operand dst must be 1024-byte
          aligned. For example, when tensor slices
          are input and the destination operand is of type float32,
          tensor[256:] can be used. However, tensor[2:] does
          not meet the alignment requirement, and an unknown error may occur.
          - This instruction must not be used together with the vectoring
          instructions.
          - The m, k, and n arguments do not need to be rounded up to multiples
           of 16 pixels. However, due to hardware
          restrictions, the shape of operands dst, a, and b must meet the
          following alignment requirements. The m and n
          arguments must be rounded up to multiples of 16 pixels, and the k
          argument must be rounded up to multiples of
          16 or 32 pixels, depending on the operand data type.
          - When n is not a multiple of 16, the invalid data in the n dimension
           of dst needs to be processed by the user.
          When m is not a multiple of 16, the invalid data in the m dimension
          of dst can be deleted in the fixpipe
          instruction. The following figure shows the implementation diagram
          of the matmul API. The rightmost data block
          is the output result after dst is processed by the fixpipe API.
          - This instruction should be used together with the
          fixpipe instruction.

        Returns:
            None

        Examples:
            #Example 1: Matrix a and matrix b are of type int8, dst is
            #of type int32, and ReLU is implemented using fixpipe.

            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            # Define the tensors.
            a_gm = tik_instance.Tensor("int8", [2, 32, 32], name='a_gm',
                                        scope=tik.scope_gm)
            b_gm = tik_instance.Tensor("int8", [2, 160, 32], name='b_gm',
                                        scope=tik.scope_gm)
            # For matmul, m = 30. The fixpipe instruction deletes invalid data
            #from dst_l1out. Therefore, set the m dimension of dst_gm to 30.
            dst_gm = tik_instance.Tensor("int32", [10, 30, 16], name='dst_gm',
                                        scope=tik.scope_gm)
            a_l1 = tik_instance.Tensor("int8", [2, 32, 32], name='a_l1',
                                        scope=tik.scope_cbuf)
            b_l1 = tik_instance.Tensor("int8", [2, 160, 32], name='b_l1',
                                        scope=tik.scope_cbuf)
            dst_l1out = tik_instance.Tensor("int32", [10, 32, 16],
                                name='dst_l1out', scope=tik.scope_cbuf_out)
            # Move data to the source operand.
            tik_instance.data_move(a_l1, a_gm, 0, 1, 64, 0, 0)
            tik_instance.data_move(b_l1, b_gm, 0, 1, 320, 0, 0)
            # Perform matmul. The m, k, and n arguments are 30, 64, and 160,
            #respectively. The m dimension of dst_l1out is rounded up to 32.
            tik_instance.matmul(dst_l1out, a_l1, b_l1, 30, 64, 160)
            # Move data to dst_gm, where, burst_len = 30 * 16 *
            #dst_l1out_dtype_size//32 = 60.
            tik_instance.fixpipe(dst_gm, dst_l1out, 10, 60, 0, 0,
                                extend_params={"relu": True})
            tik_instance.BuildCCE(kernel_name="matmul", inputs=[a_gm, b_gm],
                                outputs=[dst_gm])

            #Inputs:
            #a_l1 = [[[-1, -1, -1, ..., -1, -1, -1]
            #         ...
            #         [-1, -1, -1, ..., -1, -1, -1]]
            #       [[-1, -1, -1, ..., -1, -1, -1]
            #         ...
            #       [-1, -1, -1, ..., -1, -1, -1]]]
            #b_l1 = [[[1, 1, 1, ..., 1, 1, 1]
            #       ...
            #       [1, 1, 1, ..., 1, 1, 1]]
            #       [[1, 1, 1, ..., 1, 1, 1]
            #       ...
            #       [1, 1, 1, ..., 1, 1, 1]]]
            #Returns:
            #dst_gm = [[[0, 0, 0, ..., 0, 0, 0]
            #            ...
            #         [0, 0, 0, ..., 0, 0, 0]]
            #         ...
            #         [[0, 0, 0, ..., 0, 0, 0]
            #         ...
            #         [0, 0, 0, ..., 0, 0, 0]]]


            #Example 2: Matrix a and matrix b are of type float16, dst is of
            #type float32, and element-wise addition is implemented using
            #fixpipe.

            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            # Define the tensors.
            a_gm = tik_instance.Tensor("float16", [4, 32, 16], name='a_gm',
                                        scope=tik.scope_gm)
            b_gm = tik_instance.Tensor("float16", [4, 160, 16], name='b_gm',
                                        scope=tik.scope_gm)
            element_wise_add_gm = tik_instance.Tensor("float32", [10, 32, 16],
                                name='element_wise_add_gm', scope=tik.scope_gm)
            # For matmul, m = 30. The fixpipe instruction deletes invalid data
            #from dst_l1out. Therefore, set the m dimension of dst_gm to 30.
            dst_gm = tik_instance.Tensor("float32", [10, 30, 16], name='dst_gm'
                                        , scope=tik.scope_gm)
            a_l1 = tik_instance.Tensor("float16", [4, 32, 16], name='a_l1',
                                        scope=tik.scope_cbuf)
            b_l1 = tik_instance.Tensor("float16", [4, 160, 16], name='b_l1',
                                        scope=tik.scope_cbuf)
            element_wise_add = tik_instance.Tensor("float32", [10, 32, 16],
                            name='element_wise_add', scope=tik.scope_cbuf)
            dst_l1out = tik_instance.Tensor("float32", [10, 32, 16],
                                name='dst_l1out', scope=tik.scope_cbuf_out)
            # Move data to the source operand.
            tik_instance.data_move(a_l1, a_gm, 0, 1, 128, 0, 0)
            tik_instance.data_move(b_l1, b_gm, 0, 1, 640, 0, 0)
            tik_instance.data_move(element_wise_add, element_wise_add_gm, 0,
                                    1, 640, 0, 0)
            # Perform matmul. The m, k, and n arguments are 30, 64, and 160,
            #respectively. The m dimension of dst_l1out is rounded up to 32.
            tik_instance.matmul(dst_l1out, a_l1, b_l1, 30, 64, 160)
            # Move data to dst_gm, where, burst_len = 30 * 16 *
            # dst_l1out_dtype_size//32 = 60.
            tik_instance.fixpipe(dst_gm, dst_l1out, 10, 60, 0, 0,
                        extend_params={"element-wise-add": element_wise_add})
            tik_instance.BuildCCE(kernel_name="matmul", inputs=[a_gm, b_gm,
                                    element_wise_add_gm], outputs=[dst_gm])

            #Inputs:
            #a_l1 = [[[1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]
            #        ...
            #        [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]]
            #        ...
            #        [[1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]
            #        ...
            #        [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]]]
            #b_l1 = [[[1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]
            #        ...
            #        [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]]
            #        ...
            #        [[1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]
            #        ...
            #        [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]]]
            #element_wise_add = [[[1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]
            #                    ...
            #                    [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]]
            #                    ...
            #                    [[1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]
            #                    ...
            #                    [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]]]
            #Returns:
            #dst_gm = [[[65.0, 65.0, 65.0, ..., 65.0, 65.0, 65.0]
            #            ...
            #            [65.0, 65.0, 65.0, ..., 65.0, 65.0, 65.0]]
            #            ...
            #            [[65.0, 65.0, 65.0, ..., 65.0, 65.0, 65.0]
            #            ...
            #            [65.0, 65.0, 65.0, ..., 65.0, 65.0, 65.0]]]
        """
        matmul_op = MatMulImpl(self, dst, init_l1out)
        matmul_op.execute(a, b, m, k, n)


# @cond
class FixpipeInfo():  # pylint: disable=R0903, R0902
    """fixpipe api info"""

    def __init__(  # pylint: disable=R0913
            self, dst, src_dtype, cburst_num, burst_len, dst_stride,
            src_stride,
            extend_params, ):
        self.cburst_num = cburst_num
        self.burst_len = burst_len
        self.dst_stride = dst_stride
        self.src_stride = src_stride
        self.extend_params = extend_params
        self.howo = burst_len * ONE_BLK_SIZE // DTYPE_SIZE[src_dtype] // 16
        self.round_howo = ceil_div(self.howo, 16) * 16
        self.howo_has_round = self.howo != self.round_howo
        self.howo_blocks = self.round_howo // 16
        self.frac_len = 16
        self.has_bias = extend_params is not None and \
                        'bias' in extend_params and \
                        extend_params['bias'] is not None
        self.dst_dtype = dst.dtype
        self.src_dtype = src_dtype
        self.has_deq = extend_params is not None and \
                       'quantize_params' in extend_params and \
                       extend_params['quantize_params'] is not None
        self.has_ele_wise_bias = extend_params is not None and \
                                 'element-wise-add' in extend_params and \
                                 extend_params['element-wise-add'] is not None
        self.has_relu = extend_params is not None and \
                        'relu' in extend_params and \
                        extend_params['relu'] is True
        self.deq_value = None
        self.l0c_shape = []
        self.dst = dst
        self.c_0 = 16
        if dst.dtype in ("int8", "uint8"):
            self.c_0 = 32
        self.bias_value = None
        self.ele_bias_value = None
# @endcond
