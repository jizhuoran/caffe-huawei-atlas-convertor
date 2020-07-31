#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

extract_image_patches
"""

import json
import os
import re
import stat
import math
from functools import reduce as functools_reduce

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.lang.cce.te_compute import common
from te import platform as tbe_platform
from te.platform import cce_params
from te.platform import insn_cmd
from te.platform.cce_build import build_config

SHAPE_SIZE_LIMIT = 2**30
BLOCK_SIZE = 16
BLOCK_SIZE_ALIGN = 16
BLOCK_SIZE_FP16 = 16
BLOCK_SIZE_INT8 = 32

DOUBLE_BUFFER = 2
FP16_SIZE = 2
INT8_SIZE = 1
NEED_UB_SPACE_NUM = 2
L1_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L1_SIZE)
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
LOAD3D_REPEAT_TIME_LIMIT = 255


def ub_split_c1(ub_split_c1_shape, A, ksize):
    def _ub_split_c1_indices(indices, A):
        n, howo, co1, khw, howo0, co0 = indices
        n_index = n
        hw_index = howo
        hw0_index = howo0
        c1_index = co1 * ksize + khw
        c0_index = co0
        return A(n_index, hw_index, c1_index, hw0_index, c0_index)

    return tvm.compute(ub_split_c1_shape,
                       lambda *indices: _ub_split_c1_indices(indices, A),
                       name='ub_split_c1')


def ub_transpose(ub_transpose_shape, A):
    def _ub_transpose_indices(indices, A):
        n, howo, howo0, khw, co1, co0 = indices
        n_index = n
        hw_index = howo
        c1_index = co1
        khw_index = khw
        hw0_index = howo0
        c0_index = co0

        return A(n_index, hw_index, c1_index, khw_index, hw0_index, c0_index)

    return tvm.compute(ub_transpose_shape,
                       lambda *indices: _ub_transpose_indices(indices, A),
                       name='ub_transpose')


def ub_merge_hw(ub_merge_shape, A):
    def _ub_merge_hw_indices(indices, A):
        in_n, in_hw, in_hw0, in_khw, in_c1, in_c0 = A.shape
        n, howo, khw, co1, co0 = indices
        n_index = n
        hw_index = howo // in_hw0
        hw0_index = howo % in_hw0
        c1_index = co1
        khw_index = khw
        c0_index = co0
        return A(n_index, hw_index, hw0_index, khw_index, c1_index, c0_index)

    return tvm.compute(ub_merge_shape,
                       lambda *indices: _ub_merge_hw_indices(indices, A),
                       name='ub_merge_hw')


def ub_merge_co(ub_merge_co_shape, A):
    def _ub_merge_co_indices(indices, A):
        in_n, in_hw, in_khw, in_c1, in_c0 = A.shape
        n, howo, khw, co = indices
        n_index = n
        hw_index = howo
        khw_index = khw
        c1_index = co // in_c0
        c0_index = co % in_c0
        return A(n_index, hw_index, khw_index, c1_index, c0_index)

    return tvm.compute(ub_merge_co_shape,
                       lambda *indices: _ub_merge_co_indices(indices, A),
                       name='ub_merge_co')


def im2col_row_major_v2(A, A_im2col_VM_shape, kernel_h, kernel_w, padding,
                        stride, dilate, compute_dtype):
    """
    calculate im2col_row_major tensor
    Parameters
    ----------
    A : feature map

    A_im2col_VM_shape : shape of A_im2col_row_major

    kernel_h: the kernel value in  h

    kernel_w: the kernel value in  w

    padding: the padding shape

    stride: the stride value

    dilate: the dilation value

    compute_dtype: dtype of compute result
    -------
    Returns : A_im2col_row_major tensor
    """
    def _im2col_row_major_indices(indices, A, kernel_h, kernel_w, padding,
                                  stride, dilate):
        """
        calculate im2col_row_major tvm lambda function
        Parameters
        ----------
        indices : indices in lambda function

        A : feature map

        kernel_h: the kernel value in  h

        kernel_w: the kernel value in  w

        padding: the padding shape

        stride: the stride value

        -------
        Returns  im2col_row_major tvm lambda function
        """
        in_n, in_c1, inH, in_w, in_c0 = A.shape

        n, hw, c1, kh, kw, c0 = indices
        stride_h, stride_w = stride
        dilate_h, dilate_w = dilate
        padding_top, padding_bottom, padding_left, padding_right = padding

        kernel_dilate_w = (kernel_w - 1) * dilate[1] + 1

        width_out = (in_w.value + padding_left + padding_right -
                     kernel_dilate_w) // (stride_w) + 1

        n_index = n
        c1_index = c1
        h_index = (hw // width_out) * stride_h + (kh * dilate_h)
        w_index = (hw % width_out) * stride_w + (kw * dilate_w)
        c0_index = c0
        return tvm.select(
            tvm.any(h_index < padding_top,
                    h_index > inH.value + padding_top - 1,
                    w_index < padding_left,
                    w_index > in_w.value + padding_left - 1),
            tvm.const(0.0, compute_dtype),
            A(n_index, c1_index, h_index - padding_top, w_index - padding_left,
              c0_index))

    return tvm.compute(
        A_im2col_VM_shape,
        lambda *indices: _im2col_row_major_indices(
            indices, A, kernel_h, kernel_w, padding, stride, dilate),
        name='im2col_row_major',
        tag='im2col_row_major')


def im2col_fractal_v2(A_im2col_shape, A, config, compute_dtype):
    """
    calculate im2col_fractal tensor
    Parameters
    ----------
    A_im2col_shape : shape of A_im2col

    A : feature map

    config: the config of cube

    compute_dtype: dtype of compute result
    -------
    Returns : A_im2col_fractal tensor
    """
    def _im2col_fractal_indices(indices, A):
        """
        calculate im2col_fractal tvm lambda function
        Parameters
        ----------
        indices : indices in lambda function

        A : feature map

        -------
        Returns : im2col_fractal tvm lambda function
        """
        block_size = config['mac'][1]
        block_size_M = config['mac'][0]
        n, hw, c1, kernel_h, kernel_w, c0 = A.shape
        batch_size, i1, j1, i0, j0 = indices
        n_index = batch_size

        hw_index = i1 * block_size_M + i0

        c1_index = (((j1 * block_size + j0) // c0.value) //
                    kernel_w.value) // kernel_h.value

        kh_index = (((j1 * block_size + j0) // c0.value) //
                    kernel_w.value) % kernel_h.value

        kw_index = ((j1 * block_size + j0) // c0.value) % kernel_w.value

        c0_index = (j1 * block_size + j0) % c0.value

        dtype = compute_dtype
        return tvm.select(
            tvm.any(hw_index < 0, hw_index > hw.value - 1),
            tvm.const(0.0, dtype),
            A(n_index, hw_index, c1_index, kh_index, kw_index, c0_index))

    return tvm.compute(A_im2col_shape,
                       lambda *indices: _im2col_fractal_indices(indices, A),
                       name='im2col_fractal',
                       tag='im2col_fractal')


@fusion_manager.register("extract_image_patches")
def extract_image_patches_compute(fmap,
                                  c_in_real,
                                  ksizes,
                                  strides,
                                  dilates,
                                  padding,
                                  kernel_name="extract_image_patches"):
    """
    ops compute

    Parameters
    ----------
    fmap : TVM tensor
        the placeholder of input_x
    c_in_real : real c size of input
    ksizes: input attr
    strides: input attr
    dilates: input attr
    padding: input attr
    kernel_name : str
        kernel name, default value is "extract_image_patches"

    Returns
    -------
    compute results
    """
    # fmap's format is NC1HWC0
    fmap_shape = fmap.shape
    dtype_input = fmap.dtype
    if dtype_input == "int8" or dtype_input == "uint8":
        BLOCK_SIZE_ALIGN = BLOCK_SIZE_INT8
        type_size = INT8_SIZE
    else:
        BLOCK_SIZE_ALIGN = BLOCK_SIZE
        type_size = FP16_SIZE
    fmap_batch, fmap_c1, fmap_h, fmap_w, fmap_c0 = fmap_shape
    # out to L1
    fmap_in_l1 = tvm.compute(fmap_shape, lambda *i: fmap[i], name="fmap_in_l1")

    _, filter_h, filter_w, _ = ksizes
    _, stride_h, stride_w, _ = strides
    _, dilate_h, dilate_w, _ = dilates

    out_h, padding_h_before, padding_h_after = \
        common.tf_get_windowed_output_size_verbose_v2(fmap_h.value,
                                                      filter_h, dilate_h,
                                                      stride_h, padding)
    out_w, padding_w_before, padding_w_after = \
        common.tf_get_windowed_output_size_verbose_v2(fmap_w.value, filter_w,
                                                      dilate_w, stride_w,
                                                      padding)

    pad = (padding_h_before, padding_h_after, padding_w_before,
           padding_w_after)
    stride = (stride_h, stride_w)
    dilate = (dilate_h, dilate_w)

    fmap_vm_shape = (fmap_batch, out_h * out_w, fmap_c1, filter_h, filter_w,
                     fmap_c0)

    fmap_im2col = im2col_row_major_v2(fmap_in_l1, fmap_vm_shape, filter_h,
                                      filter_w, pad, stride, dilate,
                                      dtype_input)

    howo = ((out_h * out_w + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    fractal_shape = (fmap_batch, howo // BLOCK_SIZE,
                     fmap_c1 * filter_h * filter_w, BLOCK_SIZE,
                     BLOCK_SIZE_ALIGN)

    config = {"mac": [16, BLOCK_SIZE_ALIGN, 16]}

    fmap_fractal = im2col_fractal_v2(fractal_shape, fmap_im2col, config,
                                     dtype_input)

    extract_params = {}
    extract_params["padding_mode"] = padding
    extract_params["out_h"] = out_h
    extract_params["out_w"] = out_w
    extract_params["fmap_shape"] = fmap_shape
    extract_params["ksizes"] = ksizes
    extract_params["strides"] = strides
    extract_params["pad"] = pad
    extract_params["fmap_vm_shape"] = fmap_vm_shape
    extract_params["fractal_shape"] = fractal_shape
    extract_params["howo"] = howo
    extract_params["c_in_real"] = c_in_real
    setfmatrix_dict = {
        "conv_kernel_h": filter_h,
        "conv_kernel_w": filter_w,
        "conv_padding_top": padding_h_before,
        "conv_padding_bottom": padding_h_after,
        "conv_padding_left": padding_w_before,
        "conv_padding_right": padding_w_after,
        "conv_stride_h": stride_h,
        "conv_stride_w": stride_w,
        "conv_dilation_h": dilate_h,
        "conv_dilation_w": dilate_w,
        "conv_fm_c": fmap_c1 * fmap_c0,
        "conv_fm_h": fmap_h,
        "conv_fm_w": fmap_w,
    }

    ub_split_c1_shape = (fmap_batch, howo // BLOCK_SIZE, fmap_c1,
                         filter_h * filter_w, BLOCK_SIZE, BLOCK_SIZE_ALIGN)
    ub_split_c1_res = ub_split_c1(ub_split_c1_shape, fmap_fractal,
                                  filter_h * filter_w)
    ub_transpose_shape = (fmap_batch, howo // BLOCK_SIZE, BLOCK_SIZE,
                          filter_h * filter_w, fmap_c1, BLOCK_SIZE_ALIGN)
    ub_transpose_res = ub_transpose(ub_transpose_shape, ub_split_c1_res)

    ub_merge_hw_shape = (fmap_batch, howo, filter_h * filter_w, fmap_c1,
                         BLOCK_SIZE_ALIGN)
    ub_merge_hw_res = ub_merge_hw(ub_merge_hw_shape, ub_transpose_res)
    ub_merge_co_shape = (fmap_batch, howo, filter_h * filter_w,
                         fmap_c1 * BLOCK_SIZE_ALIGN)
    ub_merge_co_res = ub_merge_co(ub_merge_co_shape, ub_merge_hw_res)
    workspace_shape = (fmap_batch, out_h * out_w, filter_h * filter_w,
                       fmap_c1 * BLOCK_SIZE_ALIGN)
    workspace_res = tvm.compute(workspace_shape,
                                lambda *i: ub_merge_co_res[i],
                                name="workspace_res")

    ub_res_shape = (fmap_batch, out_h * out_w, filter_h * filter_w,
                    fmap_c1 * BLOCK_SIZE_ALIGN)
    ub_res = tvm.compute(ub_res_shape,
                         lambda *i: workspace_res[i],
                         name="ub_res")

    out_shape = (fmap_batch, out_h * out_w, filter_h * filter_w, c_in_real)
    output_res = tvm.compute(out_shape,
                             lambda *i: ub_res[i],
                             name="res",
                             attrs={
                                 'extract_params': extract_params,
                                 'setfmatrix_dict': setfmatrix_dict
                             })

    return output_res, workspace_res, workspace_shape

def extract_image_patches_schedule(res, sch_list):
    """
    :param res: the multi-results in the operator
    :param sch: schedule list
    """
    sch = sch_list[0]
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
    out_h = extract_params['out_h']
    out_w = extract_params['out_w']
    filter_h = setfmatrix_map['conv_kernel_h'].value
    filter_w = setfmatrix_map['conv_kernel_w'].value
    fmap_shape = extract_params['fmap_shape']
    c_in_real = extract_params["c_in_real"]
    batch = fmap_shape[0].value
    dilate_h = setfmatrix_dict['conv_dilation_h']
    stride_h = setfmatrix_dict['conv_stride_h']

    ub_res = res.op.input_tensors[0]
    workspace_res = ub_res.op.input_tensors[0]
    ub_merge_co = workspace_res.op.input_tensors[0]
    ub_merge_hw = ub_merge_co.op.input_tensors[0]
    ub_transpose = ub_merge_hw.op.input_tensors[0]
    ub_split_c1 = ub_transpose.op.input_tensors[0]
    fmap_fractal = ub_split_c1.op.input_tensors[0]
    fmap_im2col = fmap_fractal.op.input_tensors[0]
    fmap_in_l1 = fmap_im2col.op.input_tensors[0]
    fmap = fmap_in_l1.op.input_tensors[0]

    sch[fmap_in_l1].set_scope(tbe_platform.scope_cbuf)
    sch[fmap_im2col].set_scope(tbe_platform.scope_cbuf)
    sch[fmap_fractal].set_scope(tbe_platform.scope_ubuf)
    sch[ub_split_c1].set_scope(tbe_platform.scope_ubuf)
    sch[ub_transpose].set_scope(tbe_platform.scope_ubuf)
    sch[ub_merge_hw].set_scope(tbe_platform.scope_ubuf)
    sch[ub_merge_co].set_scope(tbe_platform.scope_ubuf)
    sch[workspace_res].set_scope(tbe_platform.scope_gm)
    sch[ub_res].set_scope(tbe_platform.scope_ubuf)

    dtype_input = ub_res.dtype
    if dtype_input == "int8" or dtype_input == "uint8":
        BLOCK_SIZE_ALIGN = BLOCK_SIZE_INT8
        type_size = INT8_SIZE
    else:
        BLOCK_SIZE_ALIGN = BLOCK_SIZE
        type_size = FP16_SIZE
    lcm_out_w = BLOCK_SIZE // math.gcd(out_w, BLOCK_SIZE) * out_w
    out_hw_up16 = ((out_h * out_w - 1) // BLOCK_SIZE + 1) * BLOCK_SIZE
    if lcm_out_w > out_hw_up16:
        lcm_out_w = out_hw_up16
    sch[ub_res].buffer_align((1, 1), (1, 1), (1, 1),
                             (1, BLOCK_SIZE_ALIGN))
    sch[fmap_im2col].buffer_align((1, 1), (lcm_out_w, lcm_out_w), (1, 1), (1, 1),
                                  (1, 1), (1, BLOCK_SIZE_ALIGN))

    sch[fmap_fractal].buffer_align((1, 1), (1, 1), (1, 1), (1, BLOCK_SIZE),
                                   (1, BLOCK_SIZE_ALIGN))

    fractal_shape = extract_params["fractal_shape"]
    C = fractal_shape[2] * fractal_shape[4] // filter_h // filter_w

    split_ub_size = int(UB_SIZE / NEED_UB_SPACE_NUM / type_size /
                        DOUBLE_BUFFER)

    if c_in_real % BLOCK_SIZE_ALIGN == 0:
        n_factor = batch
        howo_factor = out_h * out_w
        khkw_factor = filter_h * filter_w
        c_factor = c_in_real
        if (((lcm_out_w - 1) // BLOCK_SIZE + 1) * fractal_shape[2] * \
            fractal_shape[3] * fractal_shape[4] // filter_h // \
            filter_w >= split_ub_size).value \
            or (fractal_shape[2] > LOAD3D_REPEAT_TIME_LIMIT).value:
            n_factor = 1
            howo_factor = lcm_out_w
            khkw_factor = 1
            c_factor = BLOCK_SIZE_ALIGN
        elif (fractal_shape[1] * fractal_shape[2] * fractal_shape[3] *
              fractal_shape[4] <= split_ub_size).value:
            n_factor = 1
        elif (lcm_out_w * fractal_shape[2] * fractal_shape[3] *
              fractal_shape[4] <= split_ub_size).value:
            n_factor = 1
            howo_factor = lcm_out_w
        else:
            n_factor = 1
            howo_factor = lcm_out_w
            khkw_factor = 1

        res_n_outer, res_n_inner = sch[res].split(res.op.axis[0],
                                                  factor=n_factor)
        res_howo_outer, res_howo_inner = sch[res].split(res.op.axis[1],
                                                        factor=howo_factor)
        res_khkw_outer, res_khkw_inner = sch[res].split(res.op.axis[2],
                                                        factor=khkw_factor)
        res_c_outer, res_c_inner = sch[res].split(res.op.axis[3],
                                                  factor=c_factor)
        sch[res].reorder(res_n_outer, res_howo_outer, res_khkw_outer,
                         res_c_outer, res_n_inner, res_howo_inner,
                         res_khkw_inner, res_c_inner)

        sch[fmap_im2col].compute_at(sch[res], res_howo_outer)
        sch[fmap_in_l1].compute_at(sch[res], res_howo_outer)

        sch[workspace_res].compute_at(sch[res], res_c_outer)
        sch[ub_res].compute_at(sch[res], res_c_outer)
        sch[ub_merge_co].compute_at(sch[res], res_c_outer)
        sch[ub_merge_hw].compute_at(sch[res], res_c_outer)
        sch[ub_transpose].compute_at(sch[res], res_c_outer)
        sch[ub_split_c1].compute_at(sch[res], res_c_outer)
        sch[fmap_fractal].compute_at(sch[res], res_c_outer)

        sch[workspace_res].compute_inline()
        sch[ub_res].compute_inline()
        sch[ub_merge_co].compute_inline()
        sch[ub_merge_hw].compute_inline()
        sch[ub_split_c1].compute_inline()

        block = tvm.thread_axis("blockIdx.x")
        sch[res].bind(res_n_outer, block)


        sch[fmap_in_l1].emit_insn(fmap_in_l1.op.axis[0], insn_cmd.DMA_COPY)
        sch[fmap_im2col].emit_insn(fmap_im2col.op.axis[0],
                                   insn_cmd.SET_FMATRIX,
                                   setfmatrix_dict)
        sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0], insn_cmd.IM2COL)

        sch[ub_split_c1].emit_insn(ub_split_c1.op.axis[0], insn_cmd.DMA_COPY)
        sch[ub_transpose].emit_insn(ub_transpose.op.axis[0], insn_cmd.DMA_COPY)
        sch[ub_merge_hw].emit_insn(ub_merge_hw.op.axis[0], insn_cmd.DMA_COPY)
        sch[ub_merge_co].emit_insn(ub_merge_co.op.axis[0], insn_cmd.DMA_COPY)
        sch[ub_res].emit_insn(ub_res.op.axis[2], insn_cmd.DMA_COPY)
        sch[workspace_res].emit_insn(workspace_res.op.axis[2],
                                     insn_cmd.DMA_COPY)
        sch[res].emit_insn(res_n_inner, insn_cmd.DMA_COPY)

    else:
        c1_factor = BLOCK_SIZE_ALIGN
        res_n_outer, res_n_inner = sch[res].split(res.op.axis[0], factor=1)
        res_c1_outer, res_c1_inner = sch[res].split(res.op.axis[3],
                                                    factor=c_in_real)
        sch[ub_res].compute_at(sch[res], res_c1_outer)

        workspace_res_n_outer, workspace_res_n_inner = sch[
            workspace_res].split(workspace_res.op.axis[0], factor=1)
        workspace_res_howo_outer, workspace_res_howo_inner = sch[
            workspace_res].split(workspace_res.op.axis[1], factor=lcm_out_w)
        workspace_res_khkw_outer, workspace_res_khkw_inner = sch[
            workspace_res].split(workspace_res.op.axis[2], factor=1)

        workspace_res_c1_outer, workspace_res_c1_inner = sch[
            workspace_res].split(workspace_res.op.axis[3],
                                 factor=c1_factor)
        sch[workspace_res].reorder(
            workspace_res_n_outer, workspace_res_howo_outer,
            workspace_res_khkw_outer, workspace_res_c1_outer,
            workspace_res_n_inner, workspace_res_howo_inner,
            workspace_res_khkw_inner, workspace_res_c1_inner)

        sch[ub_merge_co].compute_at(sch[workspace_res],
                                    workspace_res_c1_outer)
        sch[ub_merge_hw].compute_at(sch[workspace_res],
                                    workspace_res_c1_outer)
        sch[ub_transpose].compute_at(sch[workspace_res],
                                     workspace_res_c1_outer)
        sch[ub_split_c1].compute_at(sch[workspace_res],
                                    workspace_res_c1_outer)

        sch[fmap_fractal].compute_at(sch[workspace_res],
                                     workspace_res_c1_outer)
        sch[fmap_im2col].compute_at(sch[workspace_res],
                                    workspace_res_howo_outer)
        sch[fmap_in_l1].compute_at(sch[workspace_res],
                                   workspace_res_howo_outer)

        if c_in_real > BLOCK_SIZE_ALIGN:
            sch[workspace_res].compute_at(sch[res], res_n_outer)
            block = tvm.thread_axis("blockIdx.x")
            sch[res].bind(res_n_outer, block)

        sch[ub_split_c1].compute_inline()
        sch[ub_transpose].compute_inline()
        sch[ub_merge_co].compute_inline()

        sch[fmap_in_l1].emit_insn(fmap_in_l1.op.axis[0], insn_cmd.DMA_COPY)
        sch[fmap_im2col].emit_insn(fmap_im2col.op.axis[0],
                                   insn_cmd.SET_FMATRIX,
                                   setfmatrix_dict)
        sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0], insn_cmd.IM2COL)

        sch[ub_split_c1].emit_insn(ub_split_c1.op.axis[0], insn_cmd.DMA_COPY)
        sch[ub_transpose].emit_insn(ub_transpose.op.axis[0], insn_cmd.DMA_COPY)
        sch[ub_merge_hw].emit_insn(ub_merge_hw.op.axis[0], insn_cmd.DMA_COPY)
        sch[ub_merge_co].emit_insn(ub_merge_co.op.axis[0], insn_cmd.DMA_COPY)
        sch[workspace_res].emit_insn(workspace_res_c1_inner, insn_cmd.DMA_COPY)
        sch[ub_res].emit_insn(ub_res.op.axis[3], insn_cmd.DMA_COPY)
        sch[res].emit_insn(res_c1_inner, insn_cmd.DMA_COPY, {"no_overlap": 1})

    sch[fmap_in_l1].double_buffer()
    sch[fmap_im2col].double_buffer()
    sch[fmap_fractal].double_buffer()
    sch[ub_split_c1].double_buffer()
    sch[ub_transpose].double_buffer()
    sch[ub_merge_hw].double_buffer()
    sch[ub_merge_co].double_buffer()
    sch[ub_res].double_buffer()


@util.check_input_type(dict, dict, (list, tuple), (list, tuple), (list, tuple),
                       str, str)
def extract_image_patches(images,
                          y,
                          ksizes,
                          strides,
                          dilates,
                          padding,
                          kernel_name="extract_image_patches"):
    """
    calculating data

    Parameters
    ----------
    images : dict
        shape and dtype of input, only support float16
    y : dict
        shape and dtype of output, should be same shape and type as input
    ksizes: input attr
    strides: input attr
    dilates: input attr
    padding: input attr
    kernel_name : str
        kernel name, default value is "extract_image_patches"

    Returns
    -------
    None
    """
    shape_input_4d = images.get("ori_shape")
    dtype_input = images.get("dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_input_4d)
    util.check_shape_size(shape_input_4d, SHAPE_SIZE_LIMIT)

    check_list = ("float16", "int8", "uint8")
    util.check_dtype_rule(dtype_input, check_list)
    dtype_input = dtype_input.lower()

    if dtype_input == "int8" or dtype_input == "uint8":
        BLOCK_SIZE_ALIGN = BLOCK_SIZE_INT8
        type_size = INT8_SIZE
    else:
        BLOCK_SIZE_ALIGN = BLOCK_SIZE
        type_size = FP16_SIZE
    fmap_batch, fmap_h, fmap_w, fmap_c = shape_input_4d
    fmap_c1 = (fmap_c + BLOCK_SIZE_ALIGN - 1) // BLOCK_SIZE_ALIGN
    fmap_c0 = BLOCK_SIZE_ALIGN
    shape_input = (fmap_batch, fmap_c1, fmap_h, fmap_w, fmap_c0)

    _, filter_h, filter_w, _ = ksizes
    _, stride_h, stride_w, _ = strides
    _, dilate_h, dilate_w, _ = dilates
    out_h, padding_h_before, padding_h_after = \
        common.tf_get_windowed_output_size_verbose_v2(
            fmap_h, filter_h, dilate_h, stride_h, padding)
    out_w, padding_w_before, padding_w_after = \
        common.tf_get_windowed_output_size_verbose_v2(
            fmap_w, filter_w, dilate_w, stride_w, padding)

    if (out_h <= 0) or (out_w <= 0):
        raise RuntimeError(
            "out_h and out_w can not <= 0, out_h:%d, out_w:%d"
            % (out_h, out_w))

    if (fmap_w + padding_w_before + padding_w_after) <= \
            (filter_w - 1)*dilate_w + 1 + stride_w:
        raise RuntimeError(
            "the size of fmap_w(after pad) <= filter_w(after dilation) + "
            "stride_w is forbidden")

    if (filter_h % 2 == 0 and dilate_h > fmap_h):
        raise RuntimeError(
            "get all data from padding is forbidden")

    # min cut_h
    cut_h = BLOCK_SIZE // math.gcd(out_w, BLOCK_SIZE) * stride_h + filter_h
    if cut_h > fmap_h:
        cut_h = fmap_h
    if (cut_h * fmap_w * fmap_c1 * fmap_c0 * type_size * DOUBLE_BUFFER >
            L1_SIZE):
        raise RuntimeError(
            "Input size is too large load to L1, while cut h, need size: %d" %
            (cut_h * fmap_w * fmap_c1 * fmap_c0 * type_size * DOUBLE_BUFFER))

    data_input = tvm.placeholder(shape_input, name="data", dtype=dtype_input)
    output_res, workspace_res, workspace_shape = extract_image_patches_compute(
        data_input, fmap_c, ksizes, strides, dilates, padding, kernel_name)
    sch = tvm.create_schedule(output_res.op)
    extract_image_patches_schedule(output_res, [sch])

    def _write_workspace_info(workspace_list, kernel_name):
        def write_code(wkspace_dict, fname):
            fname = os.path.realpath(fname)
            if fname.startswith(os.getcwd()):
                if os.path.exists(fname):
                    with open(fname, "r") as f:
                        load_dict = json.load(f)
                    load_dict.update(wkspace_dict)
                    with open(fname, "w") as f:
                        json.dump(load_dict,
                                  f,
                                  sort_keys=True,
                                  indent=4,
                                  separators=(',', ':'))

        def shape_to_list(shape):
            """
            translate tvm.shape to list type in python
            """
            tmp = []
            for i in shape:
                tmp.append(i.value)
            return tmp

        def get_data_width(dtype):
            m = re.search(r'\d+', dtype)
            if m:
                return int(m.group(0)) // 8
            return 0

        num = len(workspace_list)
        if num:
            shape_list = [shape_to_list(i.shape) for i in workspace_list]
            total_size = [
                functools_reduce(lambda x, y: x * y, list_i)
                for list_i in shape_list
            ]

            total_size = [i * get_data_width(j.dtype)
                          for i, j in zip(total_size, workspace_list)]
            if not os.path.exists("kernel_meta"):
                os.mkdir("kernel_meta")
                os.chmod("kernel_meta",
                         stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP)
            wkspace_dict = {"workspace": {"num": num, "size": total_size}}
            write_code(wkspace_dict, "kernel_meta/" + kernel_name + ".json")

    with build_config:
        tvm.build(sch, [data_input, output_res, workspace_res],
                  "cce",
                  name=kernel_name)
        _write_workspace_info([workspace_res], kernel_name)


