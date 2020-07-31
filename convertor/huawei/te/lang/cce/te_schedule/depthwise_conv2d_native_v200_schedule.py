#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Schedule of depthwise conv2d.
"""

import te.platform.cce_params as cce_params
from te import platform as cce
from te import tvm
from te.domain.tiling.tiling_query import tiling_query

BLOCK_SIZE = cce_params.BLOCK_REDUCE

#kernel_h * kernel_w value
KERNEL_SIZE_VALUE = 9

#kernel_h
KERNEL_H = 3

#kernel_w
KERNEL_W = 3

#co value in float16 type
C0_16 = 16

#co value in uint8 or int8 type
C0_32 = 32

#the num of axis
AXIS_NUM = 4


def get_tiling_dict(tiling_new):
    """"get_tiling_dict_first"""
    tiling = {}
    tiling["AL0_matrix"] = tiling_new["AL0_matrix"][0:6]
    tiling["CL0_matrix"] = tiling_new["CL0_matrix"][0:6]
    tiling["CUB_matrix"] = tiling_new["CUB_matrix"][0:6]
    tiling["BL0_matrix"] = tiling_new["BL0_matrix"][0:6]
    tiling["manual_pingpong_buffer"] = tiling_new["manual_pingpong_buffer"]
    tiling["n_bef_batch_flag"] = tiling_new["n_bef_batch_flag"]
    tiling["AUB_shape"] = tiling_new["AUB_shape"][0:4]
    tiling["AL1_shape"] = tiling_new["AL1_shape"][0:4]
    tiling["BL1_shape"] = tiling_new["BL1_shape"][0:5]
    tiling["block_dim"] = tiling_new["block_dim"][0:4]

    tiling["scale_drq_split_flag"] = False
    tiling["bias_split_flag"] = False
    tiling["n_bef_batch_flag"] = tiling_new["n_bef_batch_flag"]
    tiling["n_bef_group_flag"] = tiling_new["n_bef_group_flag"]
    tiling["batch_bef_group_flag"] = tiling_new["batch_bef_group_flag"]
    tiling["A_overhead_opt_flag"] = tiling_new["A_overhead_opt_flag"]
    tiling["B_overhead_opt_flag"] = tiling_new["B_overhead_opt_flag"]
    return tiling


def tiling_fetch(fmap_shape, shape_w, group_num, attr):
    """tiling_fetch"""
    padding = attr[0]
    stride = attr[1]
    mad_dtype = attr[2]
    fmap_shape = fmap_shape.shape[0], fmap_shape.shape[1], \
                         fmap_shape.shape[2], fmap_shape.shape[3], \
                         fmap_shape.shape[4]

    shape_w = 1, shape_w.shape[1], KERNEL_H, KERNEL_W, shape_w.shape[3]

    dtype = "float16"
    tiling = tiling_query(a_shape=fmap_shape,
                          b_shape=shape_w,
                          a_dtype=dtype,
                          b_dtype=dtype,
                          c_dtype=mad_dtype,
                          mad_dtype=mad_dtype,
                          group=group_num,
                          padl=padding[2],
                          padr=padding[3],
                          padu=padding[0],
                          padd=padding[1],
                          strideh=stride[0],
                          stridew=stride[1],
                          fused_double_operand_num=0,
                          op_tag="depthwise_conv2d_native_v200")
    return tiling

def depthwise_conv2d_native_v200_schedule(depthwise_res):
    """"depthwise_conv2d_native_schedule"""
    sch = tvm.create_schedule(depthwise_res.op)

    # Prepare tensors.
    depthwise_cast = depthwise_res.op.input_tensors[0]
    mad_res = depthwise_cast.op.input_tensors[0]
    matmul = mad_res.op.input_tensors[0]
    fmap_pad_hw = matmul.op.input_tensors[0]
    fmap_new = fmap_pad_hw.op.input_tensors[0]
    fmap_low = fmap_new.op.input_tensors[0]
    fmap_high = fmap_new.op.input_tensors[1]
    fmap_pad = fmap_low.op.input_tensors[0]
    fmap = fmap_pad.op.input_tensors[0]
    weight = matmul.op.input_tensors[1]

    sch[fmap_pad_hw].compute_inline()

    # set data flow
    sch[fmap_pad].set_scope(cce.scope_ubuf)
    fmap_ub = fmap_pad
    sch[fmap_low].set_scope(cce.scope_ubuf)
    sch[fmap_new].set_scope(cce.scope_cbuf)

    weight_cb = sch.cache_read(weight, cce.scope_cb, [matmul, mad_res])

    sch[matmul].set_scope(cce.scope_cc)
    sch[mad_res].set_scope(cce.scope_cc)
    sch[depthwise_cast].set_scope(cce.scope_ubuf)

    axis = AXIS_NUM

    # get tiling params
    fmap_shape = [int(i.value) for i in fmap.shape]
    _, _, fmap_h, _, _ = fmap_shape
    pad_top = depthwise_res.op.attrs['padding'][0]
    pad_bottom = depthwise_res.op.attrs['padding'][1]
    kernel_kh = depthwise_res.op.attrs['kernel_h']

    stride_h = depthwise_res.op.attrs['stride'][0]
    _ho = (fmap_h + pad_top + pad_bottom - kernel_kh) // stride_h + 1

    # get tiling params

    def _get_tiling_factor():
        tiling = {
            'AUB_shape': [],
            'AL1_shape': [],
            'BL1_shape': [],
            'CL0_matrix': [],
            'CUB_matrix': []
        }
        aub_tiling = tiling["AUB_shape"]
        cl0_tiling = tiling["CL0_matrix"]
        al1_tiling = tiling["AL1_shape"]
        cub_tiling = tiling["CUB_matrix"]
        depthwise_res_shape = [int(i.value) for i in depthwise_res.shape]
        if aub_tiling == []:
            aub_h_factor = _ho
            aub_c1_factor = depthwise_res_shape[1]
        else:
            aub_h_factor = aub_tiling[1]
            aub_c1_factor = aub_tiling[0] // KERNEL_SIZE_VALUE // C0_16
        if al1_tiling != [] and cl0_tiling != []:
            al1_h_factor = al1_tiling[1] * cl0_tiling[1] * C0_16
            al1_c1_factor = al1_tiling[0] // C0_16 // KERNEL_SIZE_VALUE
        else:
            al1_h_factor = _ho
            al1_c1_factor = aub_c1_factor
        if cub_tiling == []:
            cub_h_factor = _ho
        else:
            cub_h_factor = cub_tiling[1]
        # gen fix factor
        aub_h_factor = 1
        aub_c1_factor = 1
        al1_h_factor = 1
        al1_c1_factor = 1
        cub_h_factor = 1
        tiling_res = [aub_h_factor, aub_c1_factor, al1_h_factor,
                      al1_c1_factor, cub_h_factor]
        return tiling_res
    aub_h_factor, aub_c1_factor, al1_h_factor,\
                  al1_c1_factor, cub_h_factor = _get_tiling_factor()
    res_axis_n, res_axis_c1, res_axis_c2, res_axis_c3,\
        res_axis_k, res_axis_h, res_axis_w, res_axis_c0 = depthwise_res.op.axis

    # C1
    res_c1cut_o, res_c1cut_i = sch[depthwise_res].split(res_axis_c1,
                                                        factor=aub_c1_factor)
    res_c1cut_io, res_c1cut_ii = sch[depthwise_res].split(res_c1cut_i,
                                                          factor=al1_c1_factor)

    # H
    res_hcut_o, res_hcut_i = sch[depthwise_res].split(res_axis_h,
                                                      factor=aub_h_factor)
    res_hcut_io, res_hcut_ii = sch[depthwise_res].split(res_hcut_i,
                                                        factor=al1_h_factor)
    res_hcut_iio, res_hcut_iii = sch[depthwise_res].split(res_hcut_ii,
                                                          factor=cub_h_factor)

    #W
    res_wcut_o, res_wcut_i = sch[depthwise_res].split(res_axis_w, factor=C0_16)
    res_wcut_io, res_wcut_ii = sch[depthwise_res].split(res_wcut_i,
                                                        factor=C0_16)
    res_wcut_iio, res_wcut_iii = sch[depthwise_res].split(res_wcut_ii,
                                                          factor=C0_16)

    sch[depthwise_res].reorder(res_c1cut_o, res_c1cut_io, res_c1cut_ii,
                               res_axis_n, res_axis_k, res_hcut_o, res_wcut_o,
                               res_hcut_io, res_wcut_io, res_hcut_iio,
                               res_wcut_iio, res_hcut_iii, res_wcut_iii,
                               res_axis_c2, res_axis_c3, res_axis_c0)

    # compute_at
    sch[fmap_ub].compute_at(sch[depthwise_res], res_wcut_o)
    sch[fmap_low].compute_at(sch[depthwise_res], res_wcut_o)
    sch[fmap_high].compute_at(sch[depthwise_res], res_wcut_o)

    sch[fmap_new].compute_at(sch[depthwise_res], res_wcut_io)
    sch[matmul].compute_at(sch[depthwise_res], res_wcut_io)
    sch[mad_res].compute_at(sch[depthwise_res], res_wcut_io)

    sch[depthwise_cast].compute_at(sch[depthwise_res], res_wcut_iio)

    sch[weight_cb].compute_at(sch[depthwise_res], res_c1cut_ii)

    weight_reduce = C0_16 if weight.dtype == "float16" else C0_32
    cube_block = BLOCK_SIZE * BLOCK_SIZE
    sch[weight_cb].buffer_align((1, 1), (1, 1), (weight_reduce, weight_reduce),
                                (weight_reduce, weight_reduce))
    sch[fmap_pad].storage_align(sch[fmap_pad].op.axis[-2], weight_reduce, 0)
    sch[fmap_low].storage_align(sch[fmap_low].op.axis[1], weight_reduce, 0)
    sch[fmap_new].storage_align(sch[fmap_new].op.axis[-2],
                                fmap_new.shape[-1].value, 0)

    sch[fmap_new].storage_align(sch[fmap_new].op.axis[2],
                                fmap_new.shape[-1].value * 2, 0)
    sch[mad_res].storage_align(sch[mad_res].op.axis[axis], cube_block, 0)
    sch[matmul].storage_align(sch[matmul].op.axis[axis], cube_block, 0)
    sch[depthwise_cast].storage_align(sch[depthwise_cast].op.axis[axis],
                                      cube_block, 0)
    sch[matmul].reused_by(mad_res)

    # emit insn
    sch[fmap_ub].emit_insn(fmap_ub.op.axis[0], 'dma_padding')
    sch[fmap_low].emit_insn(fmap_low.op.axis[0], 'vscsplit')
    sch[fmap_new].emit_insn(fmap_new.op.axis[0], 'dma_copy')

    sch[weight_cb].emit_insn(weight_cb.op.axis[0], 'dma_copy')

    sch[matmul].emit_insn(matmul.op.axis[0], 'depthwise_conv')
    sch[mad_res].emit_insn(sch[mad_res].op.axis[0], 'phony_insn')
    sch[depthwise_cast].emit_insn(depthwise_cast.op.axis[0], 'dma_copy')
    sch[depthwise_res].emit_insn(res_hcut_iii, 'dma_copy')

    return sch
