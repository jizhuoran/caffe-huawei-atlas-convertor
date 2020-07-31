#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

max_pool_grad_grad_with_argmax op,
computes second-order gradients of the maxpooling function.
"""
import math

from te import platform as tbe_platform
from te import tvm
from te.lang.cce.te_compute import common
from te.platform.cce_build import build_config
from te.platform.cce_build import build_config_update
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
from te.platform import insn_cmd

SHAPE_SIZE_LIMIT = 2**30  # shape limit
BLOCK_SIZE = tbe_platform.cce_params.BLOCK_REDUCE  # 16, 32B = 16 * sizeof(float16)


def _ceil_to(value, ceil_value):
    """
    Return the least multiple of ceil_value integer number(output > 0)
    which is greater than or equal to x.
    """
    if ceil_value <= 0:
        return value
    return ((value + ceil_value - 1) // ceil_value) * ceil_value


# pylint: disable=locally-disabled, too-many-arguments, invalid-name
# pylint: disable=locally-disabled, too-many-locals, too-many-statements
def get_load3d_tiling(fmap_shape, ksize, strides, padding, max_l1_valid_size,
                      max_next_valid_size, dtype):
    """
    get load3d tiling in davinci.
    ----------
        fmap_shape:
             The shape before load3d, should be (n, c1, hi, wi, c0).
        ksize:
             kernel sizes of load3d, should be (kernel_h, kernel_w).
        strides:
             strides of load3d, should be (stride_h, stride_w).
        padding:
             "SAME" or "VALID"
        max_l1_valid_size:
            The max buffer size which can used before load3d.
        max_next_valid_size:
            The max buffer size which can used after load3d.
        dtype:
            "float16" or others.
    Returns
    -------
        is_tiling_valid:
            True or False.
        shape_in_l1:
            (n, c1, hi, wi, c0).
        is_l1_double_buffer:
            True or False or None.
        shape_after_load3d:
            (n, howo, c1, khkw, c0), howo is a multiple of c0.
        is_l0_ub_double_buffer:
            True or False or None
    """
    data_size = tbe_platform.cce_intrin.get_bit_len(dtype.lower()) // 8  # 8bits = 1bytes
    max_l1_valid_num = max_l1_valid_size // data_size
    max_next_valid_num = max_next_valid_size // data_size

    fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0 = fmap_shape
    fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0 = \
        fmap_n.value, fmap_c1.value, fmap_h.value, fmap_w.value, fmap_c0.value
    kernel_h, kernel_w = ksize
    stride_h, stride_w = strides
    output_h, _, _ = common.tf_get_windowed_output_size_verbose(
        fmap_h, kernel_h, stride_h, padding.upper())
    output_w, _, _ = common.tf_get_windowed_output_size_verbose(
        fmap_w, kernel_w, stride_w, padding.upper())

    l1_n = 1  # init param
    l1_c1 = 1  # init param
    l1_hi = fmap_h
    l1_wi = fmap_w
    l1_c0 = fmap_c0

    max_hiwi_l1 = max_l1_valid_num // fmap_c0
    # hi' = ho' * stride_h + filter_h - stride_h
    max_ho_l1 = (max_hiwi_l1 // fmap_w - (kernel_h - stride_h)) // stride_h

    # The memory space of l1 is not enough.
    if max_hiwi_l1 < 1 or max_ho_l1 < 1:
        return False, None, None, None, None
    if max_ho_l1 < 1:
        # not supported tiling wi in l1 now! must repeat in vertical.
        return False, None, None, None, None

    # see if we can do double buffer in l1 and turn on H axis multicore
    DOUBLE_BUFFER = 2
    MULTI_CORE = 1
    if fmap_n * fmap_c1 <= 16:
        MULTI_CORE = 32 // fmap_n // fmap_c1 // 2

    l1_double_buffer = False
    min_com_multi = output_w * fmap_c0 // math.gcd(output_w, fmap_c0)
    if max_ho_l1 >= min_com_multi // output_w * DOUBLE_BUFFER:
        max_ho_l1 = max_ho_l1 // DOUBLE_BUFFER
        l1_double_buffer = True
        if max_ho_l1 >= min_com_multi // output_w * MULTI_CORE:
            max_ho_l1 = max_ho_l1 // MULTI_CORE


    # l1 memory is enough to put the whole feature map.
    if max_ho_l1 >= output_h:
        max_ho_l1 = output_h
        l1_hi = fmap_h
    else:  # not enough to put the whole feature map
        wo_gcd_c0 = math.gcd(output_w, fmap_c0)
        ho_gcd_c0 = fmap_c0 // wo_gcd_c0
        if max_ho_l1 < ho_gcd_c0:
            return False, None, None, None, None
        max_ho_l1 = max_ho_l1 // ho_gcd_c0 * ho_gcd_c0
        l1_hi = max_ho_l1 * stride_h + kernel_h - stride_h

    howo_pad = _ceil_to(output_h * output_w, fmap_c0)
    howo_block = howo_pad // fmap_c0
    l0ub_n = 1
    l0ub_c1 = 1
    # The value of l0ub_howo must be multiplied by c0 later.
    l0ub_howo = howo_block
    l0ub_khkw = kernel_h * kernel_w
    l0ub_c0 = fmap_c0
    l0_double_buffer = False

    max_howokhkw_l0ub = max_next_valid_num // fmap_c0 // fmap_c0
    # The memory space of l0/ub is not enough.
    if max_howokhkw_l0ub < 1:
        return False, None, None, None, None
    # see if we can do double buffer in l0/ub.
    if max_howokhkw_l0ub >= DOUBLE_BUFFER:
        max_howokhkw_l0ub = max_howokhkw_l0ub // DOUBLE_BUFFER
        l0_double_buffer = True

    # l0/ub memory is enough to put the whole col.
    if max_howokhkw_l0ub >= howo_block * kernel_h * kernel_w:
        pass
    # not enough to put whole kernel
    elif max_howokhkw_l0ub < kernel_h * kernel_w:
        l0ub_howo = 1
        l0ub_khkw = max_howokhkw_l0ub
    # enough to put a whole kernel, but not enough for howo
    else:
        l0ub_howo = max_howokhkw_l0ub // (kernel_h * kernel_w)
        if l0ub_howo == 0:
            l0ub_howo = 1
    l0ub_howo *= fmap_c0  # multiplied by c0
    # get min howo in l1 and l0/ub
    l0ub_howo = min(l0ub_howo, max_ho_l1 * output_w)

    return True, (l1_n, l1_c1, l1_hi, l1_wi,
                  l1_c0), l1_double_buffer, (l0ub_n, l0ub_howo, l0ub_c1,
                                             l0ub_khkw,
                                             l0ub_c0), l0_double_buffer


# pylint: disable=locally-disabled, too-many-branches
def check_shape_and_format_vailded(x, grad, argmax, y, ksize, strides, padding,
                                   kernel_name):
    """
    check whether the input param valid or not
    """
    ori_format_x = x.get("ori_format")

    shape_x = x.get("shape")
    shape_grad = grad.get("shape")
    shape_argmax = argmax.get("shape")
    shape_y = y.get("shape")

    if ori_format_x == "NCHW":
        _, _, kernel_h, kernel_w = ksize
        _, _, stride_h, stride_w = strides
    elif ori_format_x == "NHWC":
        _, kernel_h, kernel_w, _ = ksize
        _, stride_h, stride_w, _ = strides
    else:
        raise RuntimeError("ori_format only supports 'NHWC' or 'NCHW'.")

    if padding not in ("SAME", "VALID"):
        raise RuntimeError("Padding must be SAME or VALID.")

    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_argmax)
    util.check_shape_rule(shape_grad)
    util.check_shape_rule(shape_y)

    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_argmax, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_grad, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_y, SHAPE_SIZE_LIMIT)

    dtype_x = x.get("dtype").lower()
    dtype_argmax = argmax.get("dtype").lower()
    dtype_grad = grad.get("dtype").lower()
    dtype_y = y.get("dtype").lower()

    util.check_dtype_rule(dtype_x, ("float16", ))
    util.check_dtype_rule(dtype_argmax, ("uint16", ))
    util.check_dtype_rule(dtype_grad, ("float16", ))
    util.check_dtype_rule(dtype_y, ("float16", ))

    util.check_kernel_name(kernel_name)
    fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0 = shape_x
    _, _, grad_h, grad_w, _ = shape_grad

    shape_max_pool_h, pad_top, pad_bottom = \
        common.tf_get_windowed_output_size_verbose(fmap_h, kernel_h, stride_h,
                                                   padding)

    shape_max_pool_w, pad_left, pad_right = \
        common.tf_get_windowed_output_size_verbose(fmap_w, kernel_w, stride_w,
                                                   padding)

    # check whether grad_h and grad_w are valid
    dilation_rate = 1
    for input_size, kernel_size, stride in ((grad_h, kernel_h, stride_h),
                                            (grad_w, kernel_w, stride_w)):
        effective_kernel_size = (kernel_size - 1) * dilation_rate + 1
        if padding == "SAME":
            output_size = (input_size + stride - 1) // stride
            padding_needed = (output_size - 1) * stride + \
                             effective_kernel_size - input_size
            if padding_needed < 0:
                raise RuntimeError("Not supported shape.")

    # cloud out_size_h = 1 or out_size_w = 1, img2col does not act normally
    if util.get_product_version() == util.VERSION_CLOUD and \
            (shape_max_pool_h != 1 or shape_max_pool_w != 1):
        if fmap_w + pad_left + pad_right - kernel_w < stride_w:
            raise RuntimeError("Invalid params in the platform of cloud, "
                               "it must be fmap_w + pad_l + pad_r - kernel_w "
                               ">= stride_w")

    shape_max_pool = (fmap_n, fmap_c1, shape_max_pool_h, shape_max_pool_w,
                      fmap_c0)

    pad_list = (pad_top, pad_bottom, pad_left, pad_right)

    expect_shape_argmax_1 = \
        (fmap_n, fmap_c1, kernel_h * kernel_w,
         (shape_max_pool[2] * shape_max_pool[3] + 31) // 16, 16)

    expect_shape_argmax_2 = \
        (fmap_n, fmap_c1, kernel_h * kernel_w,
         (shape_max_pool[2] * shape_max_pool[3] + 31) // 16 * 16, 1)

    if list(shape_x) != list(shape_grad):
        raise RuntimeError("shape_x and shape_grad must be equal.")

    if list(shape_y) != list(shape_max_pool):
        raise RuntimeError("shape_y and shape_max_pool must be equal.")

    if list(expect_shape_argmax_1) != list(shape_argmax) and \
            list(expect_shape_argmax_2) != list(shape_argmax):
        raise RuntimeError("shape_argmax not support.")

    if kernel_h <= 0 or kernel_w <= 0 or stride_h <= 0 or stride_w <= 0:
        raise RuntimeError("kernel_h <= 0 or kernel_w <= 0 "
                           "or stride_h <= 0 or stride_w <= 0")

    if len(shape_x) != 5 or len(shape_grad) != 5 or len(shape_y) != 5 \
            or len(shape_argmax) != 5:
        raise RuntimeError("the length of shape must be 5!")

    if shape_x[4] != 16 or shape_y[4] != 16 or shape_grad[4] != 16 or \
            (shape_argmax[4] != 16 and shape_argmax[4] != 1):
        raise RuntimeError("c0 of x, grad, y must be 16,"
                           "c0 of argmax must be 1 or 16")

    return shape_max_pool, pad_list


# pylint: disable=locally-disabled, too-many-arguments, unused-argument,
# pylint: disable=locally-disabled, unnecessary-lambda, too-many-locals
@fusion_manager.register("max_pool_grad_grad_with_argmax")
def _max_pool_grad_grad_with_argmax_compute(
        placeholders,
        x,
        argmax,
        grad,
        y,
        ksize,
        strides,
        padding="VALID",
        ori_format_x="NCHW",
        kernel_name="cce_max_pool_grad_grad"
                    " with_argmax"):
    """
    Computes second-order gradients of the maxpooling function.

    Parameters
    ----------
        x: dict
             Include info about ori_input,
             format, ori_format, shape, ori_shape, dtype.
        grad: dict
             Include info about grad of ori_input,
             format, ori_format, shape, ori_shape, dtype.
        argmax: dict
             Include info about ori_input,
             format, ori_format, shape, ori_shape, dtype.
        y: dict
             Include info about result of function,
             format, ori_format, shape, ori_shape, dtype.
        ksize: list or tuple
            The size of the window for each dimension of the input tensor.
        strides: list or tuple
            The stride of the sliding window of the input tensor.
        padding: str
            The type of padding algorithm to use.
            Only support "VALID" or "SAME"
        kernel_name: str
            Cce kernel name,
            default value is "cce_max_pool_grad_grad_with_argmax"
    Returns
    -------
        grad_in_l1:
            process of movement of grad from gm to l1.
        grad_im2col:
            process of vm tensor of grad on l1.
        grad_fractal:
            process of fractal of grad from l1 to ub.
        grad_fractal_transp:
            process of transposition of grad.
        argmax_ub:
            process of movement of argmax from gm to ub.
        tensor_zero_ub:
            process of movement of zero tensor from gm to ub.
        grad_grad_col:
            tensor after selection.
        grad_grad:
            tensor after reduce_sum.
        output_res:
            output of the calculation.
    """
    argmax_tensor = placeholders[1]
    grad_tensor = placeholders[2]

    (grad_n, grad_c1, grad_h, grad_w, grad_c0) = grad.get("shape")
    if ori_format_x == "NHWC":
        _, kernel_h, kernel_w, _ = ksize
        _, stride_h, stride_w, _ = strides
    else:
        _, _, kernel_h, kernel_w = ksize
        _, _, stride_h, stride_w = strides

    shape_max_pool_h, pad_top, pad_bottom = \
        common.tf_get_windowed_output_size_verbose(
            grad_h, kernel_h, stride_h, padding)

    shape_max_pool_w, pad_left, pad_right = \
        common.tf_get_windowed_output_size_verbose(
            grad_w, kernel_w, stride_w, padding)

    pad_list = (pad_top, pad_bottom, pad_left, pad_right)
    stride = (stride_h, stride_w)

    # howo must be multiple of 16
    howo = _ceil_to(shape_max_pool_h * shape_max_pool_w, BLOCK_SIZE)

    # copy argmax from ub to gm
    shape_argmax_ub = (grad_n, grad_c1 * kernel_h * kernel_w,
                       howo // BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    argmax_ub = tvm.compute(shape_argmax_ub,
                            lambda *i: argmax_tensor(*i),
                            name='argmax_ub')

    # load3d compute
    shape_grad = (grad_n, grad_c1, grad_h, grad_w, grad_c0)
    grad_in_l1 = tvm.compute(shape_grad,
                             lambda *i: grad_tensor[i],
                             name="grad_in_l1")
    # n howo c1 kh kw c0
    shape_grad_vm = (grad_n, shape_max_pool_h * shape_max_pool_w, grad_c1,
                     kernel_h, kernel_w, grad_c0)
    grad_im2col = common.img2col(
        grad_in_l1,
        shape_grad_vm,
        kernel_h,
        kernel_w,
        pad_list,
        stride,
    )
    # n hw c1 kh kw c0  ->  n c1 kh kw hw c0
    shape_fractal = (grad_n, howo // BLOCK_SIZE, grad_c1 * kernel_h * kernel_w,
                     BLOCK_SIZE, BLOCK_SIZE)
    grad_fractal = common.im2col_fractal(shape_fractal,
                                         grad_im2col,
                                         "ca",
                                         tag='')

    # (n, howo/16, c1khkw, 16, c0) -> (n, c1khkw, howo/16, 16, c0)
    shape_grad_fratical_transp = (grad_n, grad_c1 * kernel_h * kernel_w,
                                  howo // BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    grad_fractal_transp = tvm.compute(
        shape_grad_fratical_transp,
        lambda i, j, k, l, m: grad_fractal[i, k, j, l, m],
        name='grad_fractal_transp')

    # declare a zero tensor, and move to ub for vsel
    dtype_tensor_zero = grad_tensor.dtype
    shape_tensor_zero = (BLOCK_SIZE, )
    tensor_zero_ub = tvm.compute(
        shape_tensor_zero,
        lambda *i: tvm.const(0, dtype=dtype_tensor_zero),
        name='tensor_zero_ub')

    # vsel compute
    shape_grad_grad_col = (grad_n, grad_c1 * kernel_h * kernel_w,
                           howo // BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    grad_grad_col = tvm.compute(
        shape_grad_grad_col,
        lambda i, j, k, l, m: tvm.select(argmax_ub[i, j, k, l, m],
                                         grad_fractal_transp[i, j, k, l, m],
                                         tensor_zero_ub[m]),
        name='grad_grad_col')

    # reduce_sum
    # (n, c1khkw, howo/16, 16, c0) -> (n, c1, howo/16, 16, c0)
    m = tvm.reduce_axis((0, kernel_h * kernel_w), "m")
    shape_grad_grad = (grad_n, grad_c1, howo // BLOCK_SIZE, BLOCK_SIZE,
                       BLOCK_SIZE)
    grad_grad = tvm.compute(
        shape_grad_grad,
        lambda i, j, n, p, q: tvm.sum(
            grad_grad_col[i, j * kernel_h * kernel_w + m, n, p, q], axis=[m]),
        name="grad_grad")

    extract_params = {}
    extract_params["padding_mode"] = padding
    extract_params["shape_max_pool_h"] = shape_max_pool_h
    extract_params["shape_max_pool_w"] = shape_max_pool_w
    extract_params["fmap_shape"] = shape_grad
    extract_params["ksizes"] = ksize
    extract_params["strides"] = strides
    extract_params["pad"] = pad_list
    extract_params["fmap_vm_shape"] = shape_grad_vm
    extract_params["fractal_shape"] = shape_fractal
    extract_params["HoWo"] = howo

    setfmatrix_dict = {
        "conv_kernel_h": kernel_h,
        "conv_kernel_w": kernel_w,
        "conv_padding_top": pad_top,
        "conv_padding_bottom": pad_bottom,
        "conv_padding_left": pad_left,
        "conv_padding_right": pad_right,
        "conv_stride_h": stride_h,
        "conv_stride_w": stride_w,
        "conv_fm_c": grad_c1 * grad_c0,
        "conv_fm_h": grad_h,
        "conv_fm_w": grad_w,
    }

    # UB to OUT
    output_res = tvm.compute(
        (grad_n, grad_c1, shape_max_pool_h * shape_max_pool_w, BLOCK_SIZE),
        lambda i, j, l, m: grad_grad[i, j, l // 16, l % 16, m],
        name="ub_to_out",
        attrs={
            'extract_params': extract_params,
            'setfmatrix_dict': setfmatrix_dict
        })

    return grad_in_l1, grad_im2col, grad_fractal, grad_fractal_transp, \
           argmax_ub, tensor_zero_ub, grad_grad_col, grad_grad, output_res


# pylint: disable=locally-disabled, too-many-statements
# pylint: disable=locally-disabled, too-many-locals
def _max_pool_grad_grad_with_argmax_schedule(compute_list, sch_list):
    """
    Computes second-order gradients of the maxpooling function.

    Parameters
    ----------
        compute_list: list
             All of the result of the maxpooling computation
             Include grad_in_l1, grad_im2col, grad_fractal, grad_fractal_transp,
              argmax_ub, tensor_zero_ub, grad_grad_col, grad_grad, res.
        sch_list: list
             sch of the maxpooling, include sch.
    Returns
    -------
    None
    """
    sch = sch_list[0]
    res = compute_list[-1]
    grad_in_l1 = compute_list[0]
    grad_im2col = compute_list[1]
    grad_fractal = compute_list[2]
    grad_fractal_transp = compute_list[3]
    argmax_ub = compute_list[4]
    tensor_zero_ub = compute_list[5]
    grad_grad_col = compute_list[6]
    grad_grad = compute_list[7]

    setfmatrix_map = res.op.attrs['setfmatrix_dict']
    setfmatrix_dict = {}
    for key, value in setfmatrix_map.items():
        if hasattr(value, "value"):
            setfmatrix_dict[key] = value.value
        else:
            setfmatrix_dict[key] = value

    extract_map = res.op.attrs['extract_params']
    extract_params = {}
    for key, value in extract_map.items():
        if hasattr(value, "value"):
            extract_params[key] = value.value
        else:
            extract_params[key] = value

    padding = extract_params['padding_mode']
    fmap_shape = extract_params['fmap_shape']
    shape_max_pool_h = extract_params['shape_max_pool_h']
    shape_max_pool_w = extract_params['shape_max_pool_w']
    stride_h = setfmatrix_dict["conv_stride_h"]
    stride_w = setfmatrix_dict["conv_stride_w"]
    kernel_h = setfmatrix_dict["conv_kernel_h"]
    kernel_w = setfmatrix_dict["conv_kernel_w"]

    # These calculations are on CB
    sch[grad_in_l1].set_scope(tbe_platform.scope_cbuf)
    sch[grad_im2col].set_scope(tbe_platform.scope_cbuf)

    # These calculations are on UB
    sch[grad_fractal].set_scope(tbe_platform.scope_ubuf)
    sch[argmax_ub].set_scope(tbe_platform.scope_ubuf)
    sch[tensor_zero_ub].set_scope(tbe_platform.scope_ubuf)
    sch[grad_grad_col].set_scope(tbe_platform.scope_ubuf)
    sch[grad_grad].set_scope(tbe_platform.scope_ubuf)

    # compute inline
    sch[grad_fractal_transp].compute_inline()

    # Last axis of grad_im2col instr has to be an integer multiple of 16
    sch[grad_grad].buffer_align((1, 1), (1, 1), (1, 1), (1, BLOCK_SIZE),
                                (1, BLOCK_SIZE), (1, 1))
    sch[grad_im2col].buffer_align((1, 1), (1, shape_max_pool_w), (1, 1),
                                  (1, 1), (1, 1), (1, BLOCK_SIZE))

    # get tiling shape value
    max_l1_valid_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L1_SIZE)
    max_ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
    max_next_valid_size = max_ub_size * 16 * kernel_h * kernel_w // \
                          (49 * kernel_h * kernel_w + 16)

    is_tiling_valid, shape_in_l1, is_l1_double_buffer, \
    shape_after_load3d, is_l0_ub_double_buffer = \
        get_load3d_tiling(fmap_shape, (kernel_h, kernel_w),
                          (stride_h, stride_w), padding, max_l1_valid_size,
                          max_next_valid_size, "float16")

    if (is_tiling_valid, shape_in_l1, is_l1_double_buffer, shape_after_load3d,
            is_l0_ub_double_buffer) == \
            (False, None, None, None, None):
        raise RuntimeError(
            "Not supported fmap shape = (%u, %u, %u, %u, %u),"
            " kernel = (1, %u, %u, 1),"
            " stride = (1, %u, %u, 1)" %
            (fmap_shape[0], fmap_shape[1], fmap_shape[2], fmap_shape[3],
             fmap_shape[4], kernel_h, kernel_w, stride_h, stride_w))

    _, _, l1_hi, l1_wi, _ = shape_in_l1

    def _get_output_length(l1_hi, l1_wi, stride, kernel_size):
        if fmap_shape[2].value == l1_hi:
            tile_l1_ho = shape_max_pool_h
        else:
            tile_l1_ho = (l1_hi + stride[0] - kernel_size[0]) // stride[0]
        if fmap_shape[3].value == l1_wi:
            tile_l1_wo = shape_max_pool_w
        else:
            tile_l1_wo = (l1_wi + stride[1] - kernel_size[1]) // stride[1]
        return tile_l1_ho, tile_l1_wo

    tile_l1_ho, tile_l1_wo, = _get_output_length(l1_hi, l1_wi,
                                                 (stride_h, stride_w),
                                                 (kernel_h, kernel_w))
    (_, ub_howo, _, ub_khkw, _) = shape_after_load3d

    # tiling
    split_factor_howo = ub_howo

    # cut grad_grad
    grad_grad_n_outer, grad_grad_n_inner = sch[grad_grad].split(
        grad_grad.op.axis[0], factor=1)
    grad_grad_c1_outer, grad_grad_c1_inner = sch[grad_grad].split(
        grad_grad.op.axis[1], factor=1)
    grad_grad_howo_outer, grad_grad_howo_inner = sch[grad_grad].split(
        grad_grad.op.axis[2], factor=(split_factor_howo + 15) // 16)
    grad_grad_k_outer, grad_grad_k_inner = sch[grad_grad].split(
        grad_grad.op.reduce_axis[0], factor=ub_khkw)
    sch[grad_grad].reorder(grad_grad_n_outer, grad_grad_c1_outer,
                           grad_grad_howo_outer, grad_grad_k_outer,
                           grad_grad_n_inner, grad_grad_c1_inner,
                           grad_grad_howo_inner, grad_grad.op.axis[3],
                           grad_grad_k_inner, grad_grad.op.axis[4])

    # cut res
    res_n_outer, res_n_inner = sch[res].split(res.op.axis[0], factor=1)
    res_c1_outer, res_c1_inner = sch[res].split(res.op.axis[1], factor=1)
    # gm->l1
    res_howo_outer, res_howo_inner = \
        sch[res].split(res.op.axis[2], factor=(tile_l1_ho * tile_l1_wo))
    # l1->ub
    res_mwo_outer, res_mwo_inner = sch[res].split(res_howo_inner,
                                                  factor=split_factor_howo)

    sch[res].reorder(res_n_outer, res_c1_outer, res_howo_outer, res_mwo_outer,
                     res_n_inner, res_c1_inner, res_mwo_inner, res.op.axis[3])

    res_fused_n_c1_howo_outer = sch[res].fuse(res_n_outer,
                                              res_c1_outer,
                                              res_howo_outer)
    core_number = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

    sch[grad_in_l1].compute_at(sch[res], res_fused_n_c1_howo_outer)
    sch[grad_im2col].compute_at(sch[res], res_fused_n_c1_howo_outer)
    sch[tensor_zero_ub].compute_at(sch[res], res_fused_n_c1_howo_outer)
    sch[grad_fractal].compute_at(sch[grad_grad], grad_grad_k_outer)
    sch[argmax_ub].compute_at(sch[grad_grad], grad_grad_k_outer)
    sch[grad_grad_col].compute_at(sch[grad_grad], grad_grad_k_outer)
    sch[grad_grad].compute_at(sch[res], res_mwo_outer)


    sch[grad_in_l1].emit_insn(grad_in_l1.op.axis[0], insn_cmd.DMA_COPY)
    sch[grad_im2col].emit_insn(grad_im2col.op.axis[0], 'set_fmatrix',
                               setfmatrix_dict)
    sch[grad_fractal].emit_insn(grad_fractal.op.axis[0], insn_cmd.IM2COL)
    sch[argmax_ub].emit_insn(argmax_ub.op.axis[0], insn_cmd.DMA_COPY)
    sch[tensor_zero_ub].emit_insn(tensor_zero_ub.op.axis[0], insn_cmd.DUP)
    sch[grad_grad_col].emit_insn(grad_grad_col.op.axis[0],
                                 insn_cmd.SELECT)
    sch[grad_grad].emit_insn(grad_grad_n_inner, insn_cmd.REDUCE_SUM)
    sch[res].emit_insn(res_n_inner, insn_cmd.DMA_COPY)

    # for double buffer
    if is_l0_ub_double_buffer:
        sch[grad_fractal].double_buffer()
        sch[argmax_ub].double_buffer()
        sch[grad_grad_col].double_buffer()
        sch[grad_grad].double_buffer()
        sch[grad_im2col].double_buffer()
    if is_l1_double_buffer:
        sch[grad_in_l1].double_buffer()

    # for multi cores
    block = tvm.thread_axis("blockIdx.x")
    sch[res].bind(res_fused_n_c1_howo_outer, block)

# pylint: disable=locally-disabled, too-many-arguments
# pylint: disable=locally-disabled, too-many-locals, too-many-statements
@util.check_input_type(dict, dict, dict, dict, (list, tuple), (list, tuple),
                       str, str)
def max_pool_grad_grad_with_argmax(x,
                                   grad,
                                   argmax,
                                   y,
                                   ksize,
                                   strides,
                                   padding="VALID",
                                   kernel_name="cce_max_pool_grad_grad"
                                               "_with_argmax"):
    """
    Computes second-order gradients of the maxpooling function.

    Parameters
    ----------
        x: dict
             Include info about ori_input,
             format, ori_format, shape, ori_shape, dtype.
        grad: dict
             Include info about grad of ori_input,
             format, ori_format, shape, ori_shape, dtype.
        argmax: dict
             Include info about ori_input,
             format, ori_format, shape, ori_shape, dtype.
        y: dict
             Include info about result of function,
             format, ori_format, shape, ori_shape, dtype.
        ksize: list or tuple
            The size of the window for each dimension of the input tensor.
        strides: list or tuple
            The stride of the sliding window of the input tensor.
        padding: str
            The type of padding algorithm to use.
            Only support "VALID" or "SAME"
        kernel_name: str
            Cce kernel name,
            default value is "cce_max_pool_grad_grad_with_argmax"
    Returns
    -------
    None
    """
    check_shape_and_format_vailded(x, grad, argmax, y, ksize, strides, padding,
                                   kernel_name)
    shape_x = x.get("shape")
    shape_grad = grad.get("shape")
    shape_argmax = argmax.get("shape")
    shape_argmax = (shape_argmax[0], shape_argmax[1], shape_argmax[2],
                    shape_argmax[3] * shape_argmax[4], 1)

    dtype_x = x.get("dtype").lower()
    dtype_grad = grad.get("dtype").lower()

    ori_format_x = x.get("ori_format")

    x_tensor = tvm.placeholder(shape_x, dtype=dtype_x, name="input_x")

    # argmax is continuous bool, real type is uint16
    _, _, _, howo, _ = shape_argmax
    shape_argmax_boolean = (shape_argmax[0], shape_argmax[1] * shape_argmax[2],
                            howo // 16, 16, shape_argmax[4])
    shape_argmax_boolean = list(shape_argmax_boolean[:-1]) + list(
        [shape_argmax_boolean[-1] * 16])
    argmax_tensor = tvm.placeholder(shape_argmax_boolean,
                                    dtype="bool",
                                    name="argmax")

    grad_tensor = tvm.placeholder(shape_grad,
                                  dtype=dtype_grad,
                                  name="input_grad")

    compute_list = _max_pool_grad_grad_with_argmax_compute(
        [x_tensor, argmax_tensor, grad_tensor], x, argmax, grad, y, ksize,
        strides, padding, ori_format_x, kernel_name)

    res = compute_list[-1]
    sch = tvm.create_schedule(res.op)
    _max_pool_grad_grad_with_argmax_schedule(compute_list, [sch])

    tensor_list = [x_tensor, grad_tensor, argmax_tensor, res]
    new_config = build_config_update(build_config, "dummy_placeholder", True)
    with new_config:
        tvm.build(sch, tensor_list, "cce", name=kernel_name)
