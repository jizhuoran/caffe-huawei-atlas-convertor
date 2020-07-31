#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import te.lang.cce
from topi import generic
from te import tvm
import te.platform.cce_params as cce_params
from te import platform as tbe_platform
from topi.cce import util
from te.lang.cce.te_compute import common
from te.platform import insn_cmd

BLOCK_SIZE = cce_params.BLOCK_REDUCE
SHAPE_SIZE_LIMIT = 100000000
SHAPE_SIZE = 4
NONETYPE = type(None)

# shape's dim of input must be 5
INPUT_DIM = 5

# shape's dim of filter must be 6
FILTER_DIM = 6

# shape's dim of output must be 5
OUTPUT_DIM = 5

# shape's dim of strides must be 2
STRIDES_DIM = 2


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# pylint: disable=unused-variable,too-many-locals,missing-docstring
# pylint: disable=too-many-statements,unnecessary-lambda
def _ceil(x):
    """
    Return the least multiple of 16 integer number
    """
    return ((x + BLOCK_SIZE - 1) // BLOCK_SIZE)*BLOCK_SIZE


def parameter_check(shape_in,
                    shape_k,
                    shape_out,
                    dtype,
                    strides,
                    padding,
                    kernel_name):
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_in, INPUT_DIM, INPUT_DIM)
    util.check_shape_rule(shape_k, FILTER_DIM, FILTER_DIM)
    util.check_shape_rule(shape_out, OUTPUT_DIM, OUTPUT_DIM)
    util.check_shape_rule(strides, STRIDES_DIM, STRIDES_DIM)

    # stride's shape is (stride_h, stride_w)
    # shape_in and shape_out is "NCHW"
    # shape_k is "HWC1"
    # (0, 1, 2, 3) corresponds to (N, C, H, W)in shape_in.
    DIM_S_H, DIM_S_W = 0, 1
    DIM_N, DIM_C1, _, _, DIM_C0 = 0, 1, 2, 3, 4
    DIM_W_C1, DIM_W_H, DIM_W_W, _, DIM_W_Co, DIM_W_C0 = 0, 1, 2, 3, 4, 5

    if shape_in[DIM_N] != shape_out[DIM_N]:
        raise RuntimeError(
            "input N-dim must be equal to out N-dim.")
    if shape_in[DIM_C1] != shape_k[DIM_W_C1]:
        raise RuntimeError(
            "input C-dim must be equal to kernel C-dim..")
    if shape_k[DIM_W_H] > 255 or shape_k[DIM_W_W] > 255:
        raise RuntimeError(
            "chip ISA limit kernel_h or kernel_w must less than 255")
    if shape_in[DIM_C1] != shape_out[DIM_C1]:
        raise RuntimeError(
            "input C-dim must be equal to out C-dim.")

    check_list = ["float16"]
    inp_dtype = dtype.lower()
    util.check_dtype_rule(inp_dtype, check_list)

    if strides[DIM_S_H] != strides[DIM_S_W]:
        raise RuntimeError(
            "only supports equal length strides in W and H dim.")

    pad_is_valided = (padding.lower() == "same") or (padding.lower() == "valid")
    if not pad_is_valided:
        raise RuntimeError(
            "only supported padding mode of "
            "\"same\" and \"valid\".")

    _, _, hi, wi, _ = shape_in
    _, hk, wk, _, _, _ = shape_k
    _, _, ho, wo, _ = shape_out
    stride_h, stride_w = strides
    l1_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L1_SIZE)
    l0a_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L0A_SIZE)
    ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
    data_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
    dilated_w = wo*strides[1] - (strides[1] - 1)
    max_dh_in_l1 = (l1_size//2 - hk*wk*BLOCK_SIZE*BLOCK_SIZE*data_size) // (
            data_size*dilated_w*BLOCK_SIZE)
    if max_dh_in_l1 < BLOCK_SIZE:
        raise RuntimeError("L1's memory space is not enough to support "
                           "dilated_h tiling with 16!")
    # limiting eque get_tiling, but tile_m get max(1024)
    # 3*max_h_in_ub * out_w  + (max_h_in_ub*stride - (stride - 1)) * dila_w
    # < (ub_size/2 - tile_m * BLOCK_SIZE)/BLOCK_SIZE
    #   max_h_in_ub * out_w + (max_h_in_ub*stride - (stride - 1)) * dila_w  < X
    #   max_h_in_ub  < (X + (stride - 1) * dila_w )/ (out_w   + stride*dila_w )
    # 3*max_h_in_ub * out_w which is out matrix + meanmatrix + mulmatrix size
    max_dh_in_ub = ((ub_size - l0a_size // 2) // (data_size*BLOCK_SIZE)
                    +(strides[DIM_S_H] - 1)*dilated_w) // \
                   (3*wo + strides[DIM_S_H]*dilated_w)
    if strides[DIM_S_H] > 1 and max_dh_in_ub < 1:
        raise RuntimeError("UB's memory space is not enough to support "
                           "dilated_h tiling with 1!")

    out_h, _, _ = common.tf_get_windowed_output_size_verbose(
        hi, hk, stride_h, padding)
    out_w, _, _ = common.tf_get_windowed_output_size_verbose(
        wi, wk, stride_w, padding)
    if out_h != ho:
        raise RuntimeError(
            "ho and hi relation is wrong by formula!"
        )
    if out_w != wo:
        raise RuntimeError(
            "wo and wi relation is wrong by formula!!"
        )

    return


def calculation_dilation(input_shape,
                         weight_sizes,
                         strides,
                         padding="SAME"):

    input_n, input_cg, input_ci1, input_h, input_w, input_block = input_shape

    weight_height, weight_width = weight_sizes

    stride_h, stride_w = strides
    out_h, pad_top, _ = common.tf_get_windowed_output_size_verbose(
        input_h, weight_height, stride_h, padding)
    out_w, pad_left, _ = common.tf_get_windowed_output_size_verbose(
        input_w, weight_width, stride_w, padding)

    # get the dialted shape, padding and strides of out_backprop
    dilated_padded_h = input_h + weight_height - 1
    dilated_padded_w = input_w + weight_width - 1

    dilated_h = out_h*stride_h - (stride_h - 1)
    dilated_w = out_w*stride_w - (stride_w - 1)

    dilated_shape = (input_n, input_cg, input_ci1, dilated_h,
                     dilated_w, input_block)

    dilated_pad_top = weight_height - 1 - pad_top
    dilated_pad_bottom = dilated_padded_h - dilated_pad_top - dilated_h
    dilated_pad_left = weight_width - 1 - pad_left
    dilated_pad_right = dilated_padded_w - dilated_pad_left - dilated_w

    dilated_pad = (dilated_pad_top, dilated_pad_bottom, dilated_pad_left,
                   dilated_pad_right)

    return dilated_shape, dilated_pad


def avg_pool_grad_compute(input_shape,
                          weight,
                          out,
                          vealuemean,
                          k_sizes,
                          strides,
                          padding):
    """
    Computes the gradients of avg pool, insert input.

    Parameters
    ----------
    input_shape: a list or tuple representing the shape of input,
                6D format [N, C1, 1, H, W, C0]

    weight: a tensor, 5D with shape [C1, Hf*Wf, 1, C0, C0]

    out: a tensor, 6D format [N, Co1, 1, Ho, Wo, C0]

    weight_sizes: a list or tuple of two ints,[H, W]

    strides: a list or tuple of two ints,[H, W]

    padding: only support "SAME" yet, the type of padding algorithm to use

    Returns
    -------
    dx_res: compute of the gradients of avg pool grad
    """
    out_type = out.dtype
    _, _, _, input_h, input_w, _ = input_shape
    k_height, k_width = k_sizes
    out_shape = (int(i.value) for i in out.shape)
    out_n, out_cgroup, out_c1, out_h, out_w, out_c0 = out_shape
    out_mul_shape = out_n, out_cgroup, out_c1, out_h, out_w, out_c0
    out_mul = tvm.compute(out_mul_shape, lambda *i:
            out(*i) * vealuemean(*i), name='out_mul')

    dilated_shape, dilated_pad = calculation_dilation(input_shape,
                                                      k_sizes,
                                                      strides,
                                                      padding)
    dilated_strides = (1, 1)

    # compute of out_backprop dilation
    out_dilated = tvm.compute(
        dilated_shape,
        lambda n, cg, c1, h, w, c0: tvm.select(
            tvm.all(h % strides[0] == 0, w % strides[1] == 0), out_mul[
                n, cg, c1, h // strides[0], w // strides[1], c0],
            tvm.const(0, out.dtype)),
        attrs={'strides': strides},
        name='out_dilated')

    # image to column of dilated out_backprop
    out_im2col_row_major_shape = (out_n, out_cgroup, input_h*input_w,
                                  out_c1, k_height, k_width,
                                  BLOCK_SIZE)
    out_col = common.im2col_6d(out_dilated, out_im2col_row_major_shape,
                               k_height, k_width, dilated_pad,
                               dilated_strides)
    hiwi_mad = (input_h*input_w + BLOCK_SIZE - 1) // BLOCK_SIZE*BLOCK_SIZE

    dout_im2col_fractal_shape = (out_n, out_cgroup, hiwi_mad // BLOCK_SIZE,
                                 out_c1*k_height*k_width,
                                 BLOCK_SIZE, BLOCK_SIZE)
    dout_col_pad = common.im2col_fractal_6d(dout_im2col_fractal_shape,
                                            out_col)
    # unuse , waiting for delect
    weight_unuse = tvm.compute(
        weight.shape,
        lambda *index: weight(*index),
        name='weight_rotated')

    res_dtype = "float32"

    # matrix multiplication of dilated out_backprop and rotated weight
    mad_shape = (out_n, out_cgroup, out_c1, hiwi_mad, out_c0)
    mad_res = common.mad(mad_shape, dout_col_pad, weight_unuse, res_dtype)

    # cast dX from float32 to float16
    dx_cast = tvm.compute(
        mad_res.shape,
        lambda *index: mad_res(*index).astype(out_type),
        name='dx_cast')

    # remove the padding of dX
    res_shape = (out_n, out_cgroup, out_c1, input_h*input_w, out_c0)
    dx_res = tvm.compute(
        res_shape,
        lambda *index: dx_cast(*index).astype(out_type),
        name='dx_res',
        attrs={
            'weight_height': k_height,
            'weight_width': k_width,
            'dilated_pad': dilated_pad,
            'dilated_strides': dilated_strides
        })
    return dx_res


def avg_pool_grad_tiling(input_w, input_h,
                         kernel_shape, out_shape, res, stride):
    """
    tiling plan, cut of batch and ci;
                 cut of output height and weight;
                 cut of m , k, n; L0
    dst_h: fmap_h
    dst_w: fmap_w
    filter_shape: C1, Hf*Wf, 1, C0, C0
    dout_shape: N, Co1, 1, Ho, Wo, C0
    stride: strideH, strideW
    """
    # float16
    data_size = 2
    l0a_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L0A_SIZE)
    ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)

    hk_wk = kernel_shape[1]
    out_h = out_shape[3]
    out_w = out_shape[4]
    # compute dilation shape
    dila_h = out_h*stride - (stride - 1)
    dila_w = out_w*stride - (stride - 1)

    # 2 is for double buffer
    max_l0a_m = l0a_size // (data_size*BLOCK_SIZE*2)
    dilated_pad_top = res.op.attrs['dilated_pad'][0].value
    dilated_pad_bottom = res.op.attrs['dilated_pad'][1].value
    k_height = res.op.attrs['weight_height'].value

    # tiling in UB
    # out : out_n, out_cgroup, out_c1, out_h, out_w, dout_c0
    #            max_h_in_ub * out_w * BLOCK_SIZE * 2
    # tiling in UB
    # out MATRIX: out_n, out_cgroup, out_c1, out_h, out_w, dout_c0
    #            max_h_in_ub * out_w * BLOCK_SIZE * 2
    # dilate : input_shape[0], input_shape[1], input_shape[2],
    #          dilated_h, dilated_w, input_shape[5]
    #          max_h_in_ub * out_w * BLOCK_SIZE * 2
    #          (max_h_in_ub*stride - (stride - 1)) * dila_w * BLOCK_SIZE * 2
    # cast : out_n, out_cgroup, out_c1, input_h*input_w, out_c0
    #        tile_m * BLOCK_SIZE * 2
    # 3*max_h_in_ub * out_w  + (max_h_in_ub*stride - (stride - 1)) * dila_w
    # < (ub_size/2 - tile_m * BLOCK_SIZE)/BLOCK_SIZE
    #   max_h_in_ub * out_w + (max_h_in_ub*stride - (stride - 1)) * dila_w  < X
    #   max_h_in_ub  < (X + (stride - 1) * dila_w )/ (out_w   + stride*dila_w )
    # becasue tile_m depend on LoC of tiling m, so m set max value max_l0a_m

    # tiling in L1
    # max_h_in_l1 = (l1_size - hk_wk*BLOCK_SIZE*BLOCK_SIZE*data_size) // (
    #        data_size*dila_w*BLOCK_SIZE)
    # It is certain that max_h_in_l1 is grater
    # than max_h_in_ub, so max_h_in_ub one time
    # into L1. L1 SIZE = 1M, UB SIZE = 256K;

    max_h_in_ub = ((ub_size // data_size - (max_l0a_m*BLOCK_SIZE))
                   // BLOCK_SIZE + (stride - 1)*dila_w) \
                  // (3*out_w + stride*dila_w)
    tile_dile_h_ub = max_h_in_ub*stride - (stride - 1)
    tile_hd = tile_dile_h_ub
    tile_input_h = tile_hd + dilated_pad_top + dilated_pad_bottom - k_height + 1
    # if tile_input_h > input_h, input_h no tiling
    if tile_input_h >= input_h:
        tile_input_h = input_h
        tile_hd = tile_input_h -1 + k_height - \
                  dilated_pad_top - dilated_pad_bottom
        tile_dile_h_ub = tile_hd
    # tiling in L0;
    tile_m = min(max_l0a_m, _ceil(tile_input_h*input_w))
    tile_k = 1
    tile_n = 1
    res_l1 = tile_input_h*input_w

    return res_l1, tile_input_h, tile_dile_h_ub, tile_m, tile_k, tile_n


# pylint: disable=locally-disabled,too-many-locals
def avg_pool_grad_schedule(res):
    """
    the tiling avg pool grad schedule
    """
    s = tvm.create_schedule(res.op)

    mad_cast = res.op.input_tensors[0]
    mad_res = mad_cast.op.input_tensors[0]
    dout_col_pad = mad_res.op.input_tensors[0]
    weight_rotated = mad_res.op.input_tensors[1]
    weight = weight_rotated.op.input_tensors[0]
    dout_col = dout_col_pad.op.input_tensors[0]
    dout_dilated = dout_col.op.input_tensors[0]
    dout_mul = dout_dilated.op.input_tensors[0]
    dout = dout_mul.op.input_tensors[0]
    dvealuemean = dout_mul.op.input_tensors[1]

    dout_ubuf = s.cache_read(dout, tbe_platform.scope_ubuf, [dout_mul])
    dvealuemean_ubuf = s.cache_read(dvealuemean, tbe_platform.scope_ubuf, [dout_mul])

    dout_mul_ubuf = s.cache_write(dout_mul, tbe_platform.scope_ubuf)
    dout_cbuf_nc1hwc0 = s.cache_write(dout_dilated, tbe_platform.scope_cbuf)
    dout_dilated_ubuf = s.cache_write(dout_cbuf_nc1hwc0, tbe_platform.scope_ubuf)
    dout_cbuf_row_major = s.cache_write(dout_col, tbe_platform.scope_cbuf)
    dout_ca = s.cache_write(dout_col_pad, tbe_platform.scope_ca)
    s[dout_mul].compute_inline()
    s[dout_dilated].compute_inline()
    s[dout_col].compute_inline()
    s[dout_col_pad].compute_inline()

    weight_cbuf = s.cache_read(weight, tbe_platform.scope_cbuf, [weight_rotated])
    weight_cb = s.cache_write(weight_rotated, tbe_platform.scope_cb)
    s[weight_rotated].compute_inline()

    mad_cc = s.cache_write(mad_res, tbe_platform.scope_cc)
    mad_ubuf = s.cache_write(mad_cast, tbe_platform.scope_ubuf)
    s[mad_res].compute_inline()
    s[mad_cast].compute_inline()

    # get shape value
    dilated_pad_top = res.op.attrs['dilated_pad'][0].value
    dilated_pad_bottom = res.op.attrs['dilated_pad'][1].value
    dilated_pad_left = res.op.attrs['dilated_pad'][2].value
    dilated_pad_right = res.op.attrs['dilated_pad'][3].value
    k_height = res.op.attrs['weight_height'].value
    k_width = res.op.attrs['weight_width'].value
    block_size = dout.op.shape[len(dout.op.shape) - 1].value
    _, _, _, dout_dilated_h, dout_dilated_w, _ = dout_dilated.shape
    input_w = dout_dilated_w.value + dilated_pad_left \
              + dilated_pad_right - k_width + 1
    input_h = dout_dilated_h.value + dilated_pad_top \
              + dilated_pad_bottom - k_height + 1
    stride = dout_dilated.op.attrs["strides"][0].value
    weight_shape = [int(i.value) for i in weight.shape]
    dout_shape = [int(i.value) for i in dout.shape]
    dout_dilated_shape = [int(i.value) for i in dout_dilated.shape]
    mad_cc_axis_n, mad_cc_axis_cg, mad_cc_axis_co1, mad_cc_axis_howomad, \
    mad_cc_axis_co0 = mad_cc.op.axis
    mad_ubuf_axis_n, mad_ubuf_axis_cg, mad_ubuf_axis_co1, \
    mad_ubuf_axis_howomad, mad_ubuf_axis_co0 = mad_ubuf.op.axis
    mad_res_shape = [int(i.value) for i in mad_res.shape]
    res_block_n, res_block_cgroup, _, _, _ = mad_res_shape
    #tiling
    res_l1, tile_input_h, tile_dile_h_ub, tile_m, \
    tile_k, tile_n = avg_pool_grad_tiling(
        input_w, input_h, weight_shape, dout_shape, res, stride)

    mad_cc_Ncut_o, mad_cc_Ncut_i = s[mad_cc].split(mad_cc_axis_n, factor=1)
    mad_cc_mcut_o, mad_cc_mcut_i = s[mad_cc].split(
        mad_cc_axis_howomad, factor=tile_m)
    mad_cc_kcut_o, mad_cc_kcut_i = s[mad_cc].split(
        mad_cc.op.reduce_axis[0], factor=tile_k)
    mad_cc_ncut_o, mad_cc_ncut_i = s[mad_cc].split(
        mad_cc_axis_co1, factor=tile_n)
    s[mad_cc].reorder(mad_cc_Ncut_o, mad_cc_axis_cg, mad_cc_ncut_o,
                      mad_cc_mcut_o, mad_cc_kcut_o, mad_cc_Ncut_i,
                      mad_cc_ncut_i, mad_cc_mcut_i, mad_cc_axis_co0,
                      mad_cc_kcut_i, mad_cc.op.reduce_axis[1])
    s[dout_ca].compute_at(s[mad_cc], mad_cc_kcut_o)
    s[weight_cb].compute_at(s[mad_cc], mad_cc_kcut_o)

    mad_ubuf_Ncut_o, mad_ubuf_Ncut_i = s[mad_ubuf].split(
        mad_ubuf_axis_n, factor=1)
    mad_ubuf_mcut_o, mad_ubuf_mcut_i = s[mad_ubuf].split(
        mad_ubuf_axis_howomad, factor=tile_m)
    mad_ubuf_ncut_o, mad_ubuf_ncut_i = s[mad_ubuf].split(
        mad_ubuf_axis_co1, factor=tile_n)
    s[mad_ubuf].reorder(mad_ubuf_Ncut_o, mad_ubuf_axis_cg, mad_ubuf_ncut_o,
                        mad_ubuf_mcut_o, mad_ubuf_Ncut_i, mad_ubuf_ncut_i,
                        mad_ubuf_mcut_i, mad_ubuf_axis_co0)
    s[mad_cc].compute_at(s[mad_ubuf], mad_ubuf_mcut_o)

    conv_Ncut_o, conv_Ncut_i = s[res].split(res.op.axis[0], factor=1)
    conv_hcut_o, conv_hcut_i = s[res].split(
        res.op.axis[3], factor=(res_l1))
    conv_mcut_o, conv_mcut_i = s[res].split(conv_hcut_i, factor=tile_m)
    s[res].reorder(conv_Ncut_o, res.op.axis[1], conv_hcut_o, conv_mcut_o,
                   conv_Ncut_i, res.op.axis[2], conv_mcut_i,
                   res.op.axis[4])
    s[mad_ubuf].buffer_align((1, 1), (1, 1), (1, 1), (1, block_size),
                             (1, block_size))
    s[mad_ubuf].compute_at(s[res], conv_mcut_o)
    s[dout_cbuf_row_major].buffer_align((1, 1), (1, 1),
                                        (input_w, input_w), (1, 1),
                                        (1, 1), (1, 1), (1, block_size))
    s[dout_cbuf_row_major].compute_at(s[res], conv_hcut_o)
    s[dout_cbuf_nc1hwc0].compute_at(s[res], conv_hcut_o)
    s[weight_cbuf].compute_at(s[res], conv_hcut_o)



    dout_dilated_w = dout_dilated_shape[4]
    ub_l1hcut_o, ub_l1hcut_i = s[dout_cbuf_nc1hwc0].split(
        dout_cbuf_nc1hwc0.op.axis[3], factor=tile_dile_h_ub)

    if stride > 1:
        dila_o_h, dila_i_h = s[dout_dilated_ubuf].split(
            dout_dilated_ubuf.op.axis[3], factor=stride)
        dila_o_w, dila_i_w = s[dout_dilated_ubuf].split(
            dout_dilated_ubuf.op.axis[4], factor=stride)
        s[dout_dilated_ubuf].reorder(dila_i_h, dila_i_w, dila_o_h, dila_o_w)
        s[dout_dilated_ubuf].unroll(dila_i_h)
        s[dout_dilated_ubuf].unroll(dila_i_w)
        s[dout_dilated_ubuf].compute_at(s[dout_cbuf_nc1hwc0], ub_l1hcut_o)
        s[dout_dilated_ubuf].emit_insn(dout_dilated_ubuf.op.axis[0],
                                       insn_cmd.DMA_PADDING)
    else:
        s[dout_dilated_ubuf].compute_inline()

    s[dout_mul_ubuf].compute_at(s[dout_cbuf_nc1hwc0], ub_l1hcut_o)
    s[dout_ubuf].compute_at(s[dout_cbuf_nc1hwc0], ub_l1hcut_o)
    s[dvealuemean_ubuf].compute_at(s[dout_cbuf_nc1hwc0], ub_l1hcut_o)
    s[dout_ubuf].emit_insn(dout_ubuf.op.axis[0], insn_cmd.DMA_COPY)
    s[dvealuemean_ubuf].emit_insn(dvealuemean_ubuf.op.axis[0],
                                  insn_cmd.DMA_COPY)
    s[dout_mul_ubuf].emit_insn(dout_mul_ubuf.op.axis[0],
                               insn_cmd.MUL)
    s[dout_cbuf_nc1hwc0].emit_insn(ub_l1hcut_i,
                                   insn_cmd.DMA_COPY)

    # emit convolution params.
    setfmatrix_dict = {
        "conv_kernel_h": res.op.attrs['weight_height'],
        "conv_kernel_w": res.op.attrs['weight_width'],
        "conv_padding_top": res.op.attrs['dilated_pad'][0],
        "conv_padding_bottom": res.op.attrs['dilated_pad'][1],
        "conv_padding_left": res.op.attrs['dilated_pad'][2],
        "conv_padding_right": res.op.attrs['dilated_pad'][3],
        "conv_stride_h": res.op.attrs['dilated_strides'][0],
        "conv_stride_w": res.op.attrs['dilated_strides'][1],
        "conv_fm_c": dout_dilated.shape[2]*dout_dilated.shape[5],
        "conv_fm_h": dout_dilated.shape[3],
        "conv_fm_w": dout_dilated.shape[4]
    }

    s[dout_cbuf_row_major].emit_insn(dout_cbuf_row_major.op.axis[1],
                                     insn_cmd.SET_FMATRIX, setfmatrix_dict)
    s[dout_ca].emit_insn(dout_ca.op.axis[1], insn_cmd.IM2COL)
    s[weight_cbuf].emit_insn(weight_cbuf.op.axis[0], insn_cmd.DMA_COPY)
    s[weight_cb].emit_insn(weight_cb.op.axis[3], insn_cmd.DMA_COPY)
    s[mad_ubuf].emit_insn(mad_ubuf_Ncut_i, insn_cmd.DMA_COPY)
    mad_dict = {
        "mad_pattern": tbe_platform.cce_params.CONV_MODE,
        "k_outer": mad_cc_kcut_o
    }
    s[mad_cc].emit_insn(mad_cc_Ncut_i, insn_cmd.MAD, mad_dict)
    s[res].emit_insn(conv_Ncut_i, insn_cmd.DMA_COPY)

    s[dout_ca].double_buffer()
    s[weight_cb].double_buffer()
    s[mad_cc].double_buffer()
    # for multi cores
    if res_block_n < 16:
        res_NNCut_o, res_NNCut_i = s[res].split(conv_Ncut_o,
                                                nparts=res_block_n)
        res_ccCut_o, res_ccCut_i = s[res].split(res.op.axis[1],
                                            nparts=res_block_cgroup)
        s[res].reorder(res_NNCut_o, res_ccCut_o, res_NNCut_i, res_ccCut_i)
        out_fused = s[res].fuse(res_NNCut_o, res_ccCut_o)
        out_fused_out, _ = s[res].split(out_fused,
                                    nparts=res_block_n*res_block_cgroup)
        bind_out, _ = s[res].split(out_fused_out, 1)
        blockidx = tvm.thread_axis("blockIdx.x")
        s[res].bind(bind_out, blockidx)
    else:
        block = tvm.thread_axis("blockIdx.x")
        s[res].bind(conv_Ncut_o, block)

    return s


@util.check_input_type(dict, (NONETYPE, dict), (NONETYPE, dict),
                       dict, (list, tuple), (list, tuple),
                       (list, tuple), str, str, (NONETYPE, str))

def avg_pool_grad_d(
        input_grad,
        mean_matrix,
        kernel_matrix,
        out_grad,
        orig_input_shape,
        ksize,
        strides,
        padding,
        data_format='NHWC',
        kernel_name="cce_avg_pool_grad_dilation"):

    """
    computes average pooling backwards gradients.

    Parameters:
    ----------

    input_grad: a dict, global model support 'NHWC' or 'NCHW'
                and padding valid, common model support 'NHWC'
                and float16

    mean_matrix: a dict or nonetype, global model support 'NHWC' or 'NCHW'
                and padding valid, common model support 'NHWC'
                and float16

    kernel_matrix: a dict or nonetype, global model support 'NHWC' or 'NCHW'
                and padding valid, common model support 'NHWC'
                and float16

    out_grad: a dict, global model support 'NHWC' or 'NCHW'
                and padding valid, common model support 'NHWC'
                and float16

    orig_input_shape: orward input shape, 4-D list, global model
                     support 'NHWC' or 'NCHW' and padding valid,
                     common model support 'NHWC'

    ksize: filter window size, int or 4-D list, support 'NHWC'

    strides: strides over h and w axis, int or 4-D list,
             support 'NHWC' or 'NCHW'

    padding:global model support 'NHWC' or 'NCHW' and padding valid

    data_format: support 'NHWC' or 'NCHW'

    kernel_name : cce kernel name, default value is "cce_avg_pool_grad_dilation"

    Returns
    -------
    None
    """

    input_grad_ori_format = input_grad.get('ori_format')
    if input_grad_ori_format == "NHWC":
        kernel_h = ksize[1]
        kernel_w = ksize[2]
        stride_h = strides[1]
        stride_w = strides[2]
        # transfer 4D to 5D orig_input_shape
        ON, OHH, OWW, OC = orig_input_shape
    elif input_grad_ori_format == "NCHW":
        kernel_h = ksize[2]
        kernel_w = ksize[3]
        stride_h = strides[2]
        stride_w = strides[3]
        # transfer 4D to 5D orig_input_shape
        ON, OC, OHH, OWW = orig_input_shape
    OC1 = _ceil(OC) // BLOCK_SIZE
    OC0 = BLOCK_SIZE
    orig_input_shape = ON, OC1, OHH, OWW, OC0
    input_grad_shape = input_grad.get("shape")
    out_grad_shape = out_grad.get("shape")
    dtype = input_grad.get("dtype").lower()

    util.check_shape_rule(input_grad_shape, INPUT_DIM, INPUT_DIM)
    util.check_shape_rule(orig_input_shape, INPUT_DIM, INPUT_DIM)
    util.check_shape_rule(strides, SHAPE_SIZE, SHAPE_SIZE)
    util.check_shape_rule(ksize, SHAPE_SIZE, SHAPE_SIZE)
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_grad_shape)
    util.check_shape_rule(orig_input_shape)
    util.check_shape_size(strides, SHAPE_SIZE_LIMIT)
    util.check_shape_size(ksize, SHAPE_SIZE_LIMIT)

    if out_grad_shape != orig_input_shape:
        raise RuntimeError("out_grad_shape must equal input_grad_shape")
    if stride_h < 1 or stride_w < 1:
        raise RuntimeError("stride should >= 1")

    data_dtype = dtype.lower()
    check_list = ["float16"]
    util.check_dtype_rule(data_dtype, check_list)

    _, _, HH, WW, _ = orig_input_shape

    if (HH == kernel_h and WW == kernel_w and input_grad_shape[2] == 1
            and input_grad_shape[3] == 1 and padding == 'VALID'):
        # for mobileV2 net, only support VALID padding.
        if padding != 'VALID':
            raise RuntimeError(
                "gobla model ,padding only support VALID ")
        else:
            pad_top, pad_left, pad_bottom, pad_right = 0, 0, 0, 0

        input_grad = tvm.placeholder(
            input_grad_shape, name="input_grad", dtype=data_dtype)

        # input_grad is overlapped result
        filter_num_h = (HH - kernel_h + pad_top + pad_bottom) // stride_h + 1
        filter_num_w = (WW - kernel_w + pad_left + pad_right) // stride_w + 1

        # global_avgpool, input FMAP size equals kernel size, kernel number=1
        if not (filter_num_h == 1 and filter_num_w == 1):
            raise RuntimeError(
                "global average pooling, input_grad_h"
                "and input_grad_w must equel 1")

        kernel_size_reciprocal = 1.0 / (kernel_h*kernel_w)

        with tvm.target.cce():
            input_grad_fp32 = te.lang.cce.cast_to(input_grad, "float32")
            grad_tmp = te.lang.cce.vmuls(input_grad_fp32,
                                         kernel_size_reciprocal)
            if data_dtype == "float16":
                grad_tmp = te.lang.cce.cast_to(grad_tmp, "float16")
            res = te.lang.cce.broadcast(grad_tmp, orig_input_shape)
            sch = generic.auto_schedule(res)
        config = {
            "name": kernel_name,
            "tensor_list": [input_grad, res]
        }
        te.lang.cce.cce_build_code(sch, config)
    else:
        shape_in = orig_input_shape
        shape_in_n, shape_in_c1, shape_in_h, shape_in_w, shape_in_c0 = shape_in
        shape_k = (shape_in_c1, kernel_h, kernel_w, 1, BLOCK_SIZE, BLOCK_SIZE)
        shape_out = input_grad_shape
        shape_out_n, shape_out_c1, shape_out_h, shape_out_w, \
        shape_out_c0 = shape_out
        # strides dim is two
        strides = stride_h, stride_w

        parameter_check(shape_in, shape_k, shape_out, dtype, strides,
                        padding, kernel_name)

        shape_in = shape_in_n, shape_in_c1, 1, \
                   shape_in_h, shape_in_w, shape_in_c0
        shape_k = (shape_out_c1, kernel_h*kernel_w, 1, BLOCK_SIZE, BLOCK_SIZE)
        shape_out = shape_out_n, shape_out_c1, 1, \
                    shape_out_h, shape_out_w, shape_out_c0
        kernel_placeholder = tvm.placeholder(shape_k,
                                             dtype=dtype, name='kernel')
        dout_placeholder = tvm.placeholder(shape_out, dtype=dtype, name='dout')

        vealuemean_placeholder = tvm.placeholder(shape_out,
                                                 dtype=dtype,
                                                 name='dvealuemean')
        res = avg_pool_grad_compute(
            shape_in, kernel_placeholder, dout_placeholder,
            vealuemean_placeholder, [kernel_h, kernel_w], strides,
            padding)

        s = avg_pool_grad_schedule(res)

        with tbe_platform.build_config:
            tvm.build(
                s, [dout_placeholder, vealuemean_placeholder,
                    kernel_placeholder, res],
                "cce",
                name=kernel_name)
