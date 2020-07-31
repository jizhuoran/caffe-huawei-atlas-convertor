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

max_pool_grad_grad
"""

import math
from enum import Enum

import te.lang.cce
from te import tvm
from te import platform as tbe_platform
import te.lang.cce.te_compute.common as common
import te.platform.cce_params as cce_params
from topi.cce import util

SHAPE_DIM_SIZE = 5
BLOCK_SIZE = 16
SHAPE_SIZE_LIMIT = 2**30


def _check_dict_key(input_dict, input_key, input_name):
    for key in input_key:
        if key not in input_dict.keys():
            raise RuntimeError("Input parameter must have attr.")


def _check_dtype(orig_x_dict, orig_y_dict, grads_dict, output_dict):
    orig_input_type = orig_x_dict.get('dtype').lower()
    orig_output_type = orig_y_dict.get('dtype').lower()
    grads_type = grads_dict.get('dtype').lower()
    output_type = output_dict.get('dtype').lower()

    if orig_input_type != orig_output_type or \
            orig_input_type != grads_type or \
            orig_input_type != output_type or \
            orig_input_type != "float16":
        raise RuntimeError("Type must be same in orig_input/orig_output/"
                           "grads/outputm, and only supported float16.")


def _check_shape(orig_x_dict, orig_y_dict, grads_dict, output_dict):
    orig_input_shape = list(orig_x_dict.get('shape'))
    orig_output_shape = list(orig_y_dict.get('shape'))
    grads_shape = list(grads_dict.get('shape'))
    output_shape = list(output_dict.get('shape'))

    if orig_input_shape != grads_shape or \
            len(grads_shape) != SHAPE_DIM_SIZE or \
            grads_shape[-1] != BLOCK_SIZE:
        raise RuntimeError("The shape of orig_input and grads must be same, "
                           "and shape must be 5HD.")
    if orig_output_shape != output_shape or \
            len(orig_output_shape) != SHAPE_DIM_SIZE or \
            orig_output_shape[-1] != BLOCK_SIZE:
        raise RuntimeError("The shape of orig_output and output must be same, "
                           "and shape must be 5HD.")


def _check_format(orig_x_dict, orig_y_dict, grads_dict, output_dict):
    orig_x_format = orig_x_dict.get('format')
    orig_y_format = orig_y_dict.get('format')
    grads_format = grads_dict.get('format')
    output_format = output_dict.get('format')
    if orig_x_format not in ("NC1HWC0", "5HD") or \
            orig_y_format not in ("NC1HWC0", "5HD") or \
            grads_format not in ("NC1HWC0", "5HD") or \
            output_format not in ("NC1HWC0", "5HD"):
        raise RuntimeError("The format must be NC1HWC0 in orig_input/"
                           "orig_output/grads/output.")


def _ceil_to(value, ceil_value):
    if ceil_value <= 0:
        return value
    return ((value + ceil_value - 1) // ceil_value)*ceil_value


class BlockTilingType(Enum):
    """
    The type of block tiling.
    invalid: tiling param is invalid.
    DIVISIBLE: Block tilting that can be exactly divided.
    FUSED: Uneven block tiling is split. Therefore, the block tiling is merged
            into one axis tiling.
    """
    INVALID = 0
    DIVISIBLE = 1
    FUSED = 2


class L1TilingType(Enum):
    """
    The type of tiling in l1.
    INVALID: tiling param is invalid.
    NO_ENOUGH_MEMORY: The memory space is insufficient for splitting.
    NOT_SUPPORTED: This tilig is not supported now.
    CUT_H: cut h in fmap.
    CUT_W: cut w in fmap.
    """
    INVALID = 0
    NO_ENOUGH_MEMORY = 1
    NOT_SUPPORTED = 2
    CUT_H = 3
    CUT_W = 4


class L0ubTilingType(Enum):
    """
    The type of tiling in l0 or ub.
    INVALID: tiling param is invalid.
    NO_ENOUGH_MEMORY: The memory space is insufficient for splitting.
    NOT_SUPPORTED: This tilig is not supported now.
    CUT_KHKW: cut KhKw.
    CUT_HOWO: cut HoWo.
    """
    INVALID = 0
    NO_ENOUGH_MEMORY = 1
    NOT_SUPPORTED = 2
    CUT_KHKW = 3
    CUT_HOWO = 4


# pylint:disable=too-many-arguments,too-many-locals
def get_load3d_tiling(fmap_shape,
                      ksize,
                      strides,
                      padding,
                      max_l1_valid_size,
                      max_next_valid_size,
                      dtype,
                      dilation=(1, 1)):
    """
    get load3d tiling in davinci.

    Parameters:
    ----------
        fmap_shape: The shpae before load3d, should be (n, c1, hi, wi, c0).

        ksize: kernel sizes of load3d, should be (kernel_h, kernel_w).

        strides: strides of load3d, should be (stride_h, stride_w).

        padding: "SAME" or "VALID".

        max_l1_valid_size: The max buffer size which can used before load3d.

        max_next_valid_size: The max buffer size which can used after load3d.

        dtype: "float16" or others.

        dilation: dilation for load3d, should be (dilation_h, dilation_w).

    Returns:
    -------
        The result of tiling.
    """
    # pylint: disable=invalid-name
    data_size = tbe_platform.cce_intrin.get_bit_len(dtype.lower()) // 8
    BLOCK_SIZE_C0 = 32 // data_size
    DOUBLE_BUFFER = 2
    LOAD3D_MAX_REPEAT = 255

    fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0 = fmap_shape
    stride_h, stride_w = strides
    dilation_h, dilation_w = dilation
    kernel_h, kernel_w = ksize
    kernel_h = (kernel_h - 1)*dilation_h + 1
    kernel_w = (kernel_w - 1)*dilation_w + 1
    output_h, _, _ = common.tf_get_windowed_output_size_verbose(
        fmap_h, kernel_h, stride_h, padding.upper())
    output_w, _, _ = common.tf_get_windowed_output_size_verbose(
        fmap_w, kernel_w, stride_w, padding.upper())

    tiling = {
        "result": False,
        "block_tiling": {},
        "l1_tiling": {},
        "l0ub_tiling": {}
    }

    def _ceil_to(value, ceil_value):
        if ceil_value <= 0:
            return value
        return ((value + ceil_value - 1) // ceil_value)*ceil_value

    def _get_block_tiling(block_axis_value):
        device_core_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        tiling = {}
        tiling["shape"] = {}

        all_block_value = 1
        for value in block_axis_value:
            all_block_value *= value
        tiling["fuse"] = all_block_value
        if all_block_value % device_core_num == 0:
            unblock_core_num = device_core_num
            cur_device_core_num = device_core_num // unblock_core_num
            block_tiling = []
            for value in block_axis_value:
                cur_axis_core_num = math.gcd(unblock_core_num, value)
                cur_device_core_num *= cur_axis_core_num
                unblock_core_num = device_core_num // cur_device_core_num
                block_tiling.append(value // cur_axis_core_num)
            tiling["shape"]["n"] = block_tiling[0]
            tiling["shape"]["c1"] = block_tiling[1]
            tiling["block_dim"] = device_core_num
            tiling["type"] = BlockTilingType.DIVISIBLE
        else:
            tiling["fuse_factor"] = _ceil_to(
                all_block_value, device_core_num) // device_core_num
            tiling["block_dim"] = _ceil_to(
                all_block_value,
                tiling["fuse_factor"]) // tiling["fuse_factor"]
            tiling["type"] = BlockTilingType.FUSED
        tiling["result"] = True
        return tiling

    block_tiling = _get_block_tiling((fmap_n, fmap_c1))
    if block_tiling["result"] is False:
        return tiling

    def _get_input_length(output_length, kernel_size, stride, out_value,
                          fmap_value):
        # calculate the input length form output length
        # hi' = ho' * stride_h + filter_h - stride_h
        if output_length >= out_value:
            return fmap_value
        return min(fmap_value, output_length*stride + kernel_size - stride)

    def _get_l1_tiling():
        # get the tiling params in l1
        tiling = {}
        tiling["shape"] = {}
        tiling["shape"]["n"] = 1
        tiling["shape"]["c1"] = 1
        tiling["shape"]["c0"] = fmap_c0

        max_l1_valid_num = max_l1_valid_size // data_size
        max_hiwi_l1 = max_l1_valid_num // BLOCK_SIZE_C0
        # The memory space of l1 is not enough.
        if max_hiwi_l1 < _get_input_length(BLOCK_SIZE, kernel_w, stride_w,
                                           output_w, fmap_w):
            tiling["result"] = False
            tiling["type"] = L1TilingType.NO_ENOUGH_MEMORY
            return tiling
        # Not supported
        if max_hiwi_l1 < kernel_h*_get_input_length(
                BLOCK_SIZE, kernel_w, stride_w, output_w, fmap_w):
            tiling["result"] = False
            tiling["type"] = L1TilingType.NOT_SUPPORTED
            return tiling

        max_hi_l1 = max_hiwi_l1 // fmap_w
        max_ho_l1 = (max_hi_l1 + stride_h - kernel_h) // stride_h
        # align for BLOCK_SIZE with multiply output_w
        wo_gcd_block = math.gcd(output_w, BLOCK_SIZE)
        ho_gcd_block = BLOCK_SIZE // wo_gcd_block
        if max_ho_l1 >= DOUBLE_BUFFER*ho_gcd_block:
            tiling["buffer"] = DOUBLE_BUFFER
            max_ho_l1 = max_ho_l1 // DOUBLE_BUFFER
            max_ho_l1 = (max_ho_l1 // ho_gcd_block*
                         ho_gcd_block) if max_ho_l1 < output_h else output_h
            tiling["shape"]["ho"] = max_ho_l1
            tiling["shape"]["hi"] = _get_input_length(
                tiling["shape"]["ho"], kernel_h, stride_h, output_h, fmap_h)
            tiling["shape"]["wo"] = output_w
            tiling["shape"]["wi"] = fmap_w
            tiling["result"] = True
            tiling["type"] = L1TilingType.CUT_H
            return tiling

        # cut w
        l1_wi = max_hiwi_l1 // kernel_h
        l1_wo = (l1_wi + stride_w - kernel_w) // stride_w
        if l1_wo >= DOUBLE_BUFFER*BLOCK_SIZE:
            l1_wo = l1_wo // DOUBLE_BUFFER
            if l1_wo >= output_w:
                l1_wo = output_w
            else:
                l1_wo = (l1_wo // BLOCK_SIZE)*BLOCK_SIZE
            tiling["buffer"] = DOUBLE_BUFFER
        else:
            l1_wo = (l1_wo // BLOCK_SIZE)*BLOCK_SIZE
            tiling["buffer"] = 1
        tiling["shape"]["wo"] = l1_wo
        tiling["shape"]["wi"] = _get_input_length(
            tiling["shape"]["wo"], kernel_w, stride_w, output_w, fmap_w)
        tiling["shape"]["ho"] = 1
        tiling["shape"]["hi"] = _get_input_length(
            tiling["shape"]["ho"], kernel_h, stride_h, output_h, fmap_h)
        tiling["result"] = True
        tiling["type"] = L1TilingType.CUT_W
        return tiling

    l1_tiling = _get_l1_tiling()
    if l1_tiling["result"] is False:
        return tiling

    def _get_l0ub_tiling():
        # get the tiling params in l0 or ub
        tiling = {}
        tiling["shape"] = {}
        tiling["shape"]["n"] = 1
        tiling["shape"]["c1"] = 1
        tiling["shape"]["c0"] = fmap_c0

        max_next_valid_num = max_next_valid_size // data_size
        max_howokhkw_l0ub = max_next_valid_num // BLOCK_SIZE_C0 // BLOCK_SIZE
        # The memory space of l0/ub is not enough.
        if max_howokhkw_l0ub < 1:
            tiling["result"] = False
            tiling["type"] = L0ubTilingType.NO_ENOUGH_MEMORY
            return tiling

        # not enough to put whole kernel
        if max_howokhkw_l0ub < kernel_h*kernel_w:
            # double buffer or not
            tiling["buffer"] = 1
            if max_howokhkw_l0ub >= DOUBLE_BUFFER:
                max_howokhkw_l0ub = max_howokhkw_l0ub // DOUBLE_BUFFER
                tiling["buffer"] = DOUBLE_BUFFER

            tiling["shape"]["howo"] = 1
            tiling["shape"]["khkw"] = max_howokhkw_l0ub
            tiling["type"] = L0ubTilingType.CUT_KHKW
        else:
            howo_pad = _ceil_to(output_h*output_w, BLOCK_SIZE_C0)
            howo_block = howo_pad // BLOCK_SIZE_C0

            # double buffer or not
            tiling["buffer"] = 1
            if max_howokhkw_l0ub >= DOUBLE_BUFFER*kernel_h*kernel_w:
                max_howokhkw_l0ub = max_howokhkw_l0ub // DOUBLE_BUFFER
                tiling["buffer"] = DOUBLE_BUFFER

            tiling["shape"]["howo"] = min(
                howo_block, max_howokhkw_l0ub // (kernel_h*kernel_w))
            tiling["shape"]["howo"] = min(tiling["shape"]["howo"],
                                          LOAD3D_MAX_REPEAT)
            tiling["shape"]["khkw"] = kernel_h*kernel_w
            tiling["type"] = L0ubTilingType.CUT_HOWO
        tiling["shape"]["howo"] *= BLOCK_SIZE
        tiling["result"] = True
        return tiling

    l0ub_tiling = _get_l0ub_tiling()
    if l0ub_tiling["result"] is False:
        return tiling

    # get min howo in l1 and l0/ub
    l0ub_tiling["shape"]["howo"] = min(
        l0ub_tiling["shape"]["howo"],
        _ceil_to(l1_tiling["shape"]["ho"]*l1_tiling["shape"]["wo"],
                 BLOCK_SIZE))
    tiling["result"] = True
    tiling["block_tiling"] = block_tiling
    tiling["l1_tiling"] = l1_tiling
    tiling["l0ub_tiling"] = l0ub_tiling
    return tiling


# pylint:disable=too-many-arguments,too-many-locals,unused-argument
# pylint:disable=too-many-statements
def max_pool_grad_grad_ir_builder(ins,
                                  outs,
                                  ksize,
                                  strides,
                                  padding="SAME",
                                  kernel_name="cce_max_pool_grad_grad"):
    """
    Calculation of maxpoolgradgrad with ir_build.

    Parameters:
    ----------
        ins: input tensors

        outs: output tensors

        ksize: kernel size of pooling, should be (kernel_h, kernel_w)

        strides: strides of pooling, should be (stride_h, stride_w)

        padding: "SAME" or "VALID"

        kernel_name: kernel_name

    Returns:
    -------
        ir_build
    """
    # pylint:disable=redefined-outer-name
    orig_x, orig_y, grads = ins
    output = outs[0]
    tvm_ir = tvm.ir_builder.create()

    # pylint: disable=invalid-name
    FP16_MIN_VALUE = 64511
    SCALAR_DTYPE = "int32"
    FMATRIX_DTYPE = "uint64"
    MASK_DTYPE = "uint16"
    CMPV_DTYPE = "uint8"
    data_size = tbe_platform.cce_intrin.get_bit_len(orig_x.dtype.lower()) // 8
    VECTOR_INST_BLOCK_SIZE = cce_params.VECTOR_INST_BLOCK_WIDTH // data_size
    VECTOR_INST_BLOCK_NUM = cce_params.VECTOR_INST_BLOCK_NUM
    DOUBLE_BUFFER = 2
    LOAD3D_MAX_REPEAT = 255

    shape_in = (int(i.value) for i in orig_x.shape)
    fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0 = shape_in
    shape_out = (int(i.value) for i in orig_y.shape)
    _, out_c1, out_h, out_w, out_c0 = shape_out

    h_pos = 0
    w_pos = 1
    kernel_h, kernel_w = ksize[h_pos], ksize[w_pos]
    stride_h, stride_w = strides[h_pos], strides[w_pos]
    output_h, pad_t, pad_b = common.tf_get_windowed_output_size_verbose(
        fmap_h, kernel_h, stride_h, padding)
    output_w, pad_l, pad_r = common.tf_get_windowed_output_size_verbose(
        fmap_w, kernel_w, stride_w, padding)
    if output_h != out_h or output_w != out_w:
        raise RuntimeError("The shape of the ori_input does not match "
                           "the shape of the ori_output.")
    # cloud out_size_h = 1 or out_size_w = 1, img2col does not act normally
    if util.get_product_version() == util.VERSION_CLOUD and \
            (out_h != 1 or out_w != 1):
        if fmap_w + pad_l + pad_r - kernel_w < stride_w:
            raise RuntimeError("Invalid params in the platform of cloud, "
                               "it must be fmap_w + pad_l + pad_r - kernel_w "
                               ">= stride_w")

    BUFFER_SIZE_IN_SEL_GRAD = 4
    BUFFER_SIZE_OF_INPUT = 4
    max_valid_ub_for_load3d = (
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) -
        ((VECTOR_INST_BLOCK_SIZE + kernel_h*kernel_w*fmap_c0*DOUBLE_BUFFER) +
         BUFFER_SIZE_IN_SEL_GRAD*(fmap_c0*fmap_c0*DOUBLE_BUFFER))*data_size
    )*kernel_h*kernel_w // (
        kernel_h*kernel_w + BUFFER_SIZE_IN_SEL_GRAD + BUFFER_SIZE_OF_INPUT)
    tiling = get_load3d_tiling(
        (fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0), (kernel_h, kernel_w),
        (stride_h, stride_w), padding,
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L1_SIZE) // 2,
        max_valid_ub_for_load3d, "float16")
    if not tiling["result"]:
        raise RuntimeError("Calculate tiling failed.")
    block_tiling = tiling["block_tiling"]
    l1_tiling = tiling["l1_tiling"]
    ub_tiling = tiling["l0ub_tiling"]
    if block_tiling["type"] != BlockTilingType.DIVISIBLE and block_tiling[
            "type"] != BlockTilingType.FUSED:
        raise RuntimeError("Block tiling is invalid.")
    if l1_tiling["type"] != L1TilingType.CUT_H and l1_tiling[
            "type"] != L1TilingType.CUT_W:
        raise RuntimeError("L1 tiling is not supported.")
    if ub_tiling["type"] != L0ubTilingType.CUT_HOWO:
        raise RuntimeError("UB tiling is not supported.")

    l1_hi = l1_tiling["shape"]["hi"]
    tiling_l1_ho_i = l1_tiling["shape"]["ho"]
    tiling_l1_wo_i = l1_tiling["shape"]["wo"]
    tiling_ub_howo_i = (ub_tiling["shape"]["howo"] + fmap_c0 - 1) // fmap_c0

    def _new_alloc(tvm_ir, dtype, shape, name, scope, double_buffer=1):
        """decl new buffer

        Parameters
        ----------
        tvm_ir : tvm.ir_builder
            Developer API of IR node builder make function.

        dtype : string
            buffer date type.

        shape : list of int
            buffer shape.

        name : string
            buffer name.

        scope : string
            buffer memory scope.

        double_buffer :
            whether need double buffer

        Returns
        -------
        buffer : tvm.schedule.Buffer
            Symbolic data buffer.

        """
        buf_var = tvm_ir.allocate(dtype, shape, name=name, scope=scope)
        if double_buffer == DOUBLE_BUFFER:
            tvm_ir.scope_attr(buf_var.asnode(), "double_buffer_scope", 1)
        new_buffer = tvm.decl_buffer(
            shape, buf_var.dtype, name=name, scope=scope, data=buf_var)

        return new_buffer

    def _re_decl_ub_buffer(ub_buffer, decl_dtype):
        return tvm.decl_buffer(
            ub_buffer.shape,
            decl_dtype,
            name=ub_buffer.name,
            data=ub_buffer.data,
            scope=tbe_platform.scope_ubuf)

    def _dup_const_0():
        const_0_shape = (VECTOR_INST_BLOCK_SIZE)
        const_0_ub = _new_alloc(
            tvm_ir,
            grads.dtype,
            const_0_shape,
            "const_0_ub",
            scope=tbe_platform.scope_ubuf)
        tvm_ir.emit(
            tvm.call_extern(grads.dtype, "vector_dup",
                            const_0_ub.access_ptr("w"),
                            tvm.const(0.0, grads.dtype), 1, 1, 1, 8, 8))
        return const_0_ub

    def _img_to_cubf(ho_o, wo_o, actual_pad_top_value, actual_pad_left_value):
        tiling_l1_hi_length = tiling_l1_ho_i*stride_h + kernel_h - stride_h
        tiling_l1_wi_length = tiling_l1_wo_i*stride_w + kernel_w - stride_w
        tiling_l1_hi_i = min(tiling_l1_hi_length,
                             fmap_h) if wo_o is None else kernel_h
        tiling_l1_wi_i = min(tiling_l1_wi_length,
                             fmap_w) if wo_o is not None else fmap_w
        img_l1_shape = (tiling_l1_hi_i, tiling_l1_wi_i, fmap_c0)
        orig_x_l1 = _new_alloc(tvm_ir, orig_x.dtype, img_l1_shape, "orig_x_l1",
                               tbe_platform.scope_cbuf, l1_tiling["buffer"])
        grads_l1 = _new_alloc(tvm_ir, grads.dtype, img_l1_shape, "grads_l1",
                              tbe_platform.scope_cbuf, l1_tiling["buffer"])
        # need scalar handle
        if wo_o is None:
            hi_min = tvm_ir.allocate(
                SCALAR_DTYPE, (1), name='hi_min', scope=tbe_platform.scope_reg)
            hi_max = tvm_ir.allocate(
                SCALAR_DTYPE, (1), name='hi_max', scope=tbe_platform.scope_reg)
            hi_min[0] = ho_o*tvm.const(tiling_l1_ho_i*stride_h,
                                       SCALAR_DTYPE) - pad_t
            hi_max[0] = hi_min[0] + tiling_l1_hi_length
            with tvm_ir.if_scope(hi_min[0] < 0):
                hi_min[0] = tvm.const(0, SCALAR_DTYPE)
            with tvm_ir.if_scope(hi_max[0] > fmap_h):
                hi_max[0] = tvm.const(fmap_h, SCALAR_DTYPE)
            burst_length = (hi_max[0] - hi_min[0])*fmap_w
            offset = (((n*fmap_c1 + c1)*fmap_h + hi_min[0])*fmap_w)*fmap_c0
            repeat_time = 1
            src_repeat_burst_length = 0

        else:
            wi_min = tvm_ir.allocate(
                SCALAR_DTYPE, (1), name='wi_min', scope=tbe_platform.scope_reg)
            wi_max = tvm_ir.allocate(
                SCALAR_DTYPE, (1), name='wi_max', scope=tbe_platform.scope_reg)
            wi_min[0] = wo_o*tvm.const(tiling_l1_wo_i*stride_w,
                                       SCALAR_DTYPE) - pad_l
            wi_max[0] = wi_min[0] + tiling_l1_wi_length
            with tvm_ir.if_scope(wi_min[0] < 0):
                wi_min[0] = tvm.const(0, SCALAR_DTYPE)
            with tvm_ir.if_scope(wi_max[0] > fmap_w):
                wi_max[0] = tvm.const(fmap_w, SCALAR_DTYPE)
            burst_length = (wi_max[0] - wi_min[0])
            offset = (((n*fmap_c1 + c1)*fmap_h + ho_o*stride_h)*fmap_w +
                      wi_min[0])*fmap_c0
            repeat_time = tvm.min(kernel_h, fmap_h - ho_o*stride_h)
            src_repeat_burst_length = fmap_w - burst_length
            _set_cut_w_fmatrix(repeat_time, burst_length, actual_pad_top_value,
                               pad_b, actual_pad_left_value, pad_r)

        gm_buffer = [orig_x, grads]
        l1_buffer = [orig_x_l1, grads_l1]
        pad_mode_call = tvm.call_pure_intrin("int32", "tvm_cce_string_print",
                                             'PAD_NONE')
        for i, gm in enumerate(gm_buffer):
            tvm_ir.emit(
                tvm.call_extern(l1_buffer[i].dtype, "copy_gm_to_cbuf",
                                l1_buffer[i].access_ptr("w"),
                                gm.access_ptr("r", offset=offset), 0,
                                repeat_time, burst_length,
                                src_repeat_burst_length, 0, pad_mode_call))
        return orig_x_l1, grads_l1

    def _get_ori_y_offset(ho_o, wo_o, howo_o):
        offset_in_wo = wo_o*tiling_l1_wo_i if wo_o is not None else 0
        orig_y_offset = (((n*out_c1 + c1)*out_h + ho_o*tiling_l1_ho_i)*out_w +
                         offset_in_wo + howo_o*tiling_ub_howo_i*out_c0)*out_c0
        return orig_y_offset

    def _get_ori_y_burst_length(actual_tiling_l1_wo_i, ho_o, wo_o, howo_o):
        length_in_ho = tvm.min(
            tiling_l1_ho_i, out_h - ho_o*tiling_l1_ho_i) if wo_o is None else 0
        length_in_wo = tvm.min(actual_tiling_l1_wo_i, out_w -
                               wo_o*tiling_l1_wo_i) if wo_o is not None else 0
        orig_y_burst_length = tvm.min(
            tiling_ub_howo_i*out_c0,
            length_in_ho*out_w + length_in_wo - howo_o*tiling_ub_howo_i*out_c0)
        return orig_y_burst_length

    def _mov_orig_y(actual_tiling_l1_wo_i, ho_o, wo_o, howo_o):
        img_shape = (tiling_ub_howo_i, fmap_c0, fmap_c0)
        orig_y_ub = _new_alloc(tvm_ir, orig_y.dtype, img_shape, "orig_y_ub",
                               tbe_platform.scope_ubuf, ub_tiling["buffer"])
        orig_y_burstlength = _get_ori_y_burst_length(actual_tiling_l1_wo_i,
                                                     ho_o, wo_o, howo_o)
        orig_y_offset = _get_ori_y_offset(ho_o, wo_o, howo_o)
        tvm_ir.emit(
            tvm.call_extern(orig_y.dtype, "copy_gm_to_ubuf",
                            orig_y_ub.access_ptr("w"),
                            orig_y.access_ptr("r", offset=orig_y_offset), 0, 1,
                            orig_y_burstlength, 0, 0))
        return orig_y_ub

    def _set_cut_w_fmatrix(actual_fmap_h, actual_fmap_w, actual_pad_top, pad_b,
                           pad_l, pad_r):
        config = actual_fmap_w | actual_fmap_h << 16 | tvm.const(
            pad_l, FMATRIX_DTYPE) << 32 | tvm.const(
                pad_r, FMATRIX_DTYPE) << 40 | actual_pad_top << 48 | tvm.const(
                    pad_b, FMATRIX_DTYPE) << 56
        tvm_ir.emit(tvm.call_extern(orig_x.dtype, "set_fmatrix", config))
        return actual_pad_top, pad_l

    def _set_const_fmatrix(fmap_h, pad_t, pad_b, pad_l, pad_r):
        config = fmap_w | fmap_h << 16 | pad_l << 32 | pad_r << 40 |\
                        pad_t << 48 | pad_b << 56
        tvm_ir.emit(
            tvm.call_extern(orig_x.dtype, "set_fmatrix",
                            tvm.const(config, dtype=FMATRIX_DTYPE)))
        return pad_t, pad_l

    def _set_padding_value(padding_value):
        tvm_ir.emit(tvm.call_extern("uint16", "set_padding", padding_value))

    def _get_first_lefttop(howo_o, howo_i, actual_pad_top_value,
                           actual_pad_left_value):
        howo_offset = (howo_o*tiling_ub_howo_i + howo_i)*fmap_c0
        first_ho = howo_offset // out_w
        first_wo = howo_offset - out_w*first_ho
        first_w = first_wo*stride_w - actual_pad_left_value
        first_h = first_ho*stride_h - actual_pad_top_value
        return first_w, first_h

    def _img_to_col_horizontal(l1_buffer, ub_buffer, actual_tiling_ub_howo_i,
                               howo_o, howo_i, actual_pad_top_value,
                               actual_pad_left_value):
        conv_format_shape = (kernel_h, kernel_w, fmap_c0, fmap_c0)
        csize_call = tvm.call_pure_intrin("int32", "tvm_cce_string_print",
                                          'CSIZE0')
        first_w, first_h = _get_first_lefttop(
            howo_o, howo_i, actual_pad_top_value, actual_pad_left_value)

        def _load3d(ub_buffer):
            cnt = (
                kernel_w*kernel_h + LOAD3D_MAX_REPEAT - 1) // LOAD3D_MAX_REPEAT
            with tvm_ir.for_range(
                    0, cnt - 1, name="l_i", dtype=SCALAR_DTYPE) as l_i:
                cur_khkw = l_i*LOAD3D_MAX_REPEAT
                offset = cur_khkw*BLOCK_SIZE*BLOCK_SIZE
                first_kh = cur_khkw // kernel_w
                first_kw = cur_khkw - first_kh*kernel_w
                tvm_ir.emit(
                    tvm.call_extern(ub_buffer.dtype, "img2col_cbuf_to_ub",
                                    ub_buffer.access_ptr("w", offset=offset),
                                    l1_buffer.access_ptr("r"), first_kw,
                                    first_kh, first_w, first_h, 0, stride_w,
                                    stride_h, kernel_w, kernel_h, 1, 1, 1, 0,
                                    LOAD3D_MAX_REPEAT, csize_call))
            cur_khkw = (cnt - 1)*LOAD3D_MAX_REPEAT
            offset = cur_khkw*BLOCK_SIZE*BLOCK_SIZE
            first_kh = cur_khkw // kernel_w
            first_kw = cur_khkw - first_kh*kernel_w
            last_repeat = kernel_w*kernel_h - cur_khkw
            tvm_ir.emit(
                tvm.call_extern(ub_buffer.dtype, "img2col_cbuf_to_ub",
                                ub_buffer.access_ptr("w", offset=offset),
                                l1_buffer.access_ptr("r"), first_kw, first_kh,
                                first_w, first_h, 0, stride_w, stride_h,
                                kernel_w, kernel_h, 1, 1, 1, 0, last_repeat,
                                csize_call))

        if actual_tiling_ub_howo_i == 1:
            _load3d(ub_buffer)
        else:
            conv_format = _new_alloc(
                tvm_ir, ub_buffer.dtype, conv_format_shape, "conv_format",
                tbe_platform.scope_ubuf, ub_tiling["buffer"])
            _load3d(conv_format)
            ub_offset = howo_i*fmap_c0*fmap_c0
            for i in range(fmap_c0*fmap_c0 // VECTOR_INST_BLOCK_SIZE):
                tvm_ir.emit(
                    tvm.call_extern(
                        ub_buffer.dtype, "vadds",
                        ub_buffer.access_ptr(
                            "w", offset=ub_offset + i*VECTOR_INST_BLOCK_SIZE),
                        conv_format.access_ptr(
                            "r", offset=i*VECTOR_INST_BLOCK_SIZE),
                        tvm.const(0.0, ub_buffer.dtype), kernel_h*kernel_w, 1,
                        1, BLOCK_SIZE*actual_tiling_ub_howo_i, BLOCK_SIZE))

    #pylint: disable=invalid-name
    def _img_to_col_vertical(l1_buffer,
                             ub_buffer,
                             actual_tiling_ub_howo_i,
                             howo_o,
                             kh,
                             kw,
                             actual_pad_top_value,
                             actual_pad_left_value,
                             is_ub_offset=False):
        csize_call = tvm.call_pure_intrin("float16", "tvm_cce_string_print",
                                          'CSIZE0')
        ub_offset = 0 if not is_ub_offset else actual_tiling_ub_howo_i*fmap_c0*\
                                               fmap_c0*(kh*kernel_w + kw)
        first_w, first_h = _get_first_lefttop(howo_o, 0, actual_pad_top_value,
                                              actual_pad_left_value)
        tvm_ir.emit(
            tvm.call_extern(ub_buffer.dtype, "img2col_cbuf_to_ub",
                            ub_buffer.access_ptr("w", offset=ub_offset),
                            l1_buffer.access_ptr("r"), kw, kh, first_w,
                            first_h, 0, stride_w, stride_h, kernel_w, kernel_h,
                            1, 1, 1, 1, actual_tiling_ub_howo_i, csize_call))

    def _orig_x_to_col(orig_x_l1, actual_tiling_ub_howo_i,
                       actual_pad_top_value, actual_pad_left_value, howo_o):
        orig_x_col_shape = (tiling_ub_howo_i, kernel_h*kernel_w, fmap_c0,
                            fmap_c0)
        orig_x_ub = _new_alloc(tvm_ir, orig_x_l1.dtype, orig_x_col_shape,
                               "orig_x_ub", tbe_platform.scope_ubuf,
                               ub_tiling["buffer"])
        if actual_tiling_ub_howo_i < kernel_h*kernel_w:
            _set_padding_value(FP16_MIN_VALUE)
            with tvm_ir.for_range(
                    0, actual_tiling_ub_howo_i, name="howo_i") as howo_i:
                _img_to_col_horizontal(
                    orig_x_l1, orig_x_ub, actual_tiling_ub_howo_i, howo_o,
                    howo_i, actual_pad_top_value, actual_pad_left_value)
        else:
            _set_padding_value(FP16_MIN_VALUE)
            with tvm_ir.for_range(0, kernel_h, name="kh") as kh:
                with tvm_ir.for_range(0, kernel_w, name="kw") as kw:
                    _img_to_col_vertical(orig_x_l1, orig_x_ub,
                                         actual_tiling_ub_howo_i, howo_o, kh,
                                         kw, actual_pad_top_value,
                                         actual_pad_left_value, True)

        return orig_x_ub

    def _init_output_in_ub(mask_repeat, grads_ub_dtype):
        img_shape = (tiling_ub_howo_i, fmap_c0, fmap_c0)
        output_ub = _new_alloc(tvm_ir, grads_ub_dtype, img_shape, "output_ub",
                               tbe_platform.scope_ubuf, ub_tiling["buffer"])
        tvm_ir.emit(
            tvm.call_extern(output_ub.dtype, "vector_dup",
                            output_ub.access_ptr("w"),
                            tvm.const(0.0, grads_ub_dtype), mask_repeat, 1, 1,
                            8, 8))
        return output_ub

    def _img_to_col_grads(grads_l1, howo_o, kh, kw, actual_pad_top_value,
                          actual_pad_left_value):
        grads_col_shape = (tiling_ub_howo_i, fmap_c0, fmap_c0)
        grads_ub = _new_alloc(tvm_ir, grads_l1.dtype, grads_col_shape,
                              "grads_ub", tbe_platform.scope_ubuf,
                              ub_tiling["buffer"])
        _img_to_col_vertical(grads_l1, grads_ub, tiling_ub_howo_i, howo_o, kh,
                             kw, actual_pad_top_value, actual_pad_left_value)
        return grads_ub

    def _sel_grad_col(mask_ub, grads_ub):
        img_shape = (tiling_ub_howo_i, fmap_c0, fmap_c0)
        grads_sel_ub = _new_alloc(tvm_ir, grads_ub.dtype, img_shape,
                                  "grads_sel_ub", tbe_platform.scope_ubuf,
                                  ub_tiling["buffer"])
        # vsel
        with tvm_ir.for_range(0, tiling_ub_howo_i, name="mask_r") as mask_r:
            fractal_repeat = fmap_c0*fmap_c0 // VECTOR_INST_BLOCK_SIZE
            with tvm_ir.for_range(
                    0, fractal_repeat, name="fractal_r") as fractal_r:
                mask_type_bit_size = tbe_platform.cce_intrin.get_bit_len(
                    MASK_DTYPE.lower())
                mask_offset = (mask_r*fractal_repeat + fractal_r)*\
                              VECTOR_INST_BLOCK_SIZE // mask_type_bit_size
                tvm_ir.emit(
                    tvm.call_extern(
                        grads_ub.dtype, "set_cmpmask",
                        mask_ub.access_ptr("w", offset=mask_offset)))
                grads_ub_offset = (
                    (mask_r)*fractal_repeat + fractal_r)*VECTOR_INST_BLOCK_SIZE
                grads_sel_ub_offset = (
                    mask_r*fractal_repeat + fractal_r)*VECTOR_INST_BLOCK_SIZE
                tvm_ir.emit(
                    tvm.call_extern(
                        grads.dtype, "vsel",
                        grads_sel_ub.access_ptr(
                            "w", offset=grads_sel_ub_offset),
                        grads_ub.access_ptr("r", offset=grads_ub_offset),
                        const_0_ub.access_ptr("r"), 1, 1, 1, 1, 8, 8, 8))
        return grads_sel_ub

    def _sel_grads(grads_l1, orig_x_ub, orig_y_ub, actual_tiling_ub_howo_i,
                   actual_pad_top_value, actual_pad_left_value, howo_o):
        # shapes in ub are (howo, c0, c0) and (khkw, howo, c0, c0)
        mask_shape = (_ceil_to(tiling_ub_howo_i, VECTOR_INST_BLOCK_NUM),
                      fmap_c0)
        mask_repeat = actual_tiling_ub_howo_i*fmap_c0*fmap_c0 //\
                      VECTOR_INST_BLOCK_SIZE
        b16_repeat = (actual_tiling_ub_howo_i*fmap_c0 + VECTOR_INST_BLOCK_SIZE
                      - 1) // VECTOR_INST_BLOCK_SIZE
        _set_padding_value(0)
        if kernel_h*kernel_w == 1:
            # mask_ub = mask_ori
            grads_ub = _img_to_col_grads(grads_l1, howo_o, 0, 0,
                                         actual_pad_top_value,
                                         actual_pad_left_value)
            mask_ori = _new_alloc(tvm_ir, MASK_DTYPE, mask_shape, "mask_ori",
                                  tbe_platform.scope_ubuf, ub_tiling["buffer"])
            mask_orig_cmp = _re_decl_ub_buffer(mask_ori, CMPV_DTYPE)
            tvm_ir.emit(
                tvm.call_extern(CMPV_DTYPE, "vcmpv_eq",
                                mask_orig_cmp.access_ptr("w"),
                                orig_x_ub.access_ptr("r"),
                                orig_y_ub.access_ptr("r"), mask_repeat, 1, 1,
                                1, 8, 8, 8))
            output_ub = _sel_grad_col(mask_ori, grads_ub)
            return output_ub
        else:
            output_ub = _init_output_in_ub(mask_repeat, grads_l1.dtype)
            mask_not = _new_alloc(tvm_ir, MASK_DTYPE, mask_shape, "mask_not",
                                  tbe_platform.scope_ubuf, ub_tiling["buffer"])
            mask_or = _new_alloc(tvm_ir, MASK_DTYPE, mask_shape, "mask_or",
                                 tbe_platform.scope_ubuf, ub_tiling["buffer"])
            with tvm_ir.for_range(0, kernel_h, name="kh") as kh:
                with tvm_ir.for_range(0, kernel_w, name="kw") as kw:
                    grads_ub = _img_to_col_grads(grads_l1, howo_o, kh, kw,
                                                 actual_pad_top_value,
                                                 actual_pad_left_value)
                    mask_ori = _new_alloc(tvm_ir, MASK_DTYPE, mask_shape,
                                          "mask_ori", tbe_platform.scope_ubuf,
                                          ub_tiling["buffer"])
                    mask_ub = _new_alloc(tvm_ir, MASK_DTYPE, mask_shape,
                                         "mask_ub", tbe_platform.scope_ubuf,
                                         ub_tiling["buffer"])
                    orig_x_ub_offset = (
                        kh*kernel_w + kw)*tiling_ub_howo_i*fmap_c0*fmap_c0
                    # orig_x_ub_offset = 0
                    with tvm_ir.if_scope(tvm.all(kh == 0, kw == 0)):
                        # mask_ori = vcmpv(orig_x, orig_y)
                        # mask_ub = mask_ori
                        # mask_or = mask_ub
                        # mask_not = not mask_or
                        mask_ub_cmp = _re_decl_ub_buffer(mask_ub, CMPV_DTYPE)
                        tvm_ir.emit(
                            tvm.call_extern(
                                CMPV_DTYPE, "vcmpv_eq",
                                mask_ub_cmp.access_ptr("w"),
                                orig_x_ub.access_ptr(
                                    "r", offset=orig_x_ub_offset),
                                orig_y_ub.access_ptr("r"), mask_repeat, 1, 1,
                                1, 8, 8, 8))

                        tvm_ir.emit(
                            tvm.call_extern(mask_ub.dtype, "copy_ubuf_to_ubuf",
                                            mask_or.access_ptr("w"),
                                            mask_ub.access_ptr("r"),
                                            0, 1, b16_repeat * 8, 0, 0))
                        tvm_ir.emit(
                            tvm.call_extern(mask_ub.dtype, "vnot",
                                            mask_not.access_ptr("w"),
                                            mask_ub.access_ptr("r"),
                                            b16_repeat, 1, 1, 8, 8))
                    with tvm_ir.else_scope():
                        # mask_ori = vcmpv(orig_x, orig_y)
                        # mask_ub = mask_not and mask_ori
                        # if kh != kernel_h - 1 or kw != kernel_w - 1:
                        #     mask_or = mask_or or mask_ori
                        #     mask_not = not mask_or
                        mask_orig_cmp = _re_decl_ub_buffer(
                            mask_ori, CMPV_DTYPE)
                        tvm_ir.emit(
                            tvm.call_extern(
                                CMPV_DTYPE, "vcmpv_eq",
                                mask_orig_cmp.access_ptr("w"),
                                orig_x_ub.access_ptr(
                                    "r", offset=orig_x_ub_offset),
                                orig_y_ub.access_ptr("r"), mask_repeat, 1, 1,
                                1, 8, 8, 8))
                        tvm_ir.emit(
                            tvm.call_extern(mask_ub.dtype, "vand",
                                            mask_ub.access_ptr("w"),
                                            mask_not.access_ptr("r"),
                                            mask_ori.access_ptr("r"),
                                            b16_repeat, 1, 1, 1, 8, 8, 8))
                        with tvm_ir.if_scope(
                                tvm.any((kh != kernel_h - 1),
                                        (kw != kernel_w - 1))):
                            tvm_ir.emit(
                                tvm.call_extern(mask_ub.dtype, "vor",
                                                mask_or.access_ptr("w"),
                                                mask_or.access_ptr("r"),
                                                mask_ub.access_ptr("r"),
                                                b16_repeat, 1, 1, 1, 8, 8, 8))
                            tvm_ir.emit(
                                tvm.call_extern(mask_ub.dtype, "vnot",
                                                mask_not.access_ptr("w"),
                                                mask_or.access_ptr("r"),
                                                b16_repeat, 1, 1, 8, 8))
                    grads_sel_ub = _sel_grad_col(mask_ub, grads_ub)
                    tvm_ir.emit(
                        tvm.call_extern(output_ub.dtype, "vadd",
                                        output_ub.access_ptr("w"),
                                        output_ub.access_ptr("r"),
                                        grads_sel_ub.access_ptr("r"),
                                        mask_repeat, 1, 1, 1, 8, 8, 8))
                    return output_ub

    def _mov_output_ub(actual_tiling_l1_wo_i, output_ub, ho_o, wo_o, howo_o):
        output_burst_length = _get_ori_y_burst_length(actual_tiling_l1_wo_i,
                                                      ho_o, wo_o, howo_o)
        output_offset = _get_ori_y_offset(ho_o, wo_o, howo_o)
        tvm_ir.emit(
            tvm.call_extern(output_ub.dtype, "copy_ubuf_to_gm",
                            output.access_ptr("w", offset=output_offset),
                            output_ub.access_ptr("r"), 0, 1,
                            output_burst_length, 0, 0))

    def _calculation(actual_tiling_l1_ho_i, actual_tiling_l1_wo_i, ho_o, wo_o,
                     actual_pad_top_value, actual_pad_left_value):
        tiling_ub_howo_o = (
            (actual_tiling_l1_ho_i*out_w + actual_tiling_l1_wo_i + fmap_c0 - 1)
            // fmap_c0 + tiling_ub_howo_i - 1) // tiling_ub_howo_i
        orig_x_l1, grads_l1 = _img_to_cubf(ho_o, wo_o, actual_pad_top_value,
                                           actual_pad_left_value)
        l1_howo_pad = (actual_tiling_l1_ho_i*out_w + actual_tiling_l1_wo_i +
                       fmap_c0 - 1) // fmap_c0
        actual_tiling_ub_howo_o = tiling_ub_howo_o
        if l1_howo_pad % tiling_ub_howo_i != 0:
            actual_tiling_ub_howo_o -= 1

        with tvm_ir.for_range(
                0, actual_tiling_ub_howo_o, name="howo_o",
                dtype=SCALAR_DTYPE) as howo_o:
            orig_y_ub = _mov_orig_y(actual_tiling_l1_wo_i, ho_o, wo_o, howo_o)
            orig_x_ub = _orig_x_to_col(orig_x_l1, tiling_ub_howo_i,
                                       actual_pad_top_value,
                                       actual_pad_left_value, howo_o)
            output_ub = _sel_grads(grads_l1, orig_x_ub, orig_y_ub,
                                   tiling_ub_howo_i, actual_pad_top_value,
                                   actual_pad_left_value, howo_o)
            _mov_output_ub(actual_tiling_l1_wo_i, output_ub, ho_o, wo_o,
                           howo_o)

        if actual_tiling_ub_howo_o != tiling_ub_howo_o:
            howo_o = actual_tiling_ub_howo_o
            orig_y_ub = _mov_orig_y(actual_tiling_l1_wo_i, ho_o, wo_o, howo_o)
            orig_x_ub = _orig_x_to_col(orig_x_l1, tiling_ub_howo_i,
                                       actual_pad_top_value,
                                       actual_pad_left_value, howo_o)
            output_ub = _sel_grads(grads_l1, orig_x_ub, orig_y_ub,
                                   tiling_ub_howo_i, actual_pad_top_value,
                                   actual_pad_left_value, howo_o)
            _mov_output_ub(actual_tiling_l1_wo_i, output_ub, ho_o, wo_o,
                           howo_o)

    def _calculation_in_each_block():
        cut_w_flag = True if l1_tiling["type"] == L1TilingType.CUT_W else False
        if cut_w_flag:
            cur_pad_top = tvm_ir.allocate(
                FMATRIX_DTYPE, (1),
                name='pad_top',
                scope=tbe_platform.scope_reg)
            cur_pad_top[0] = tvm.const(pad_t, FMATRIX_DTYPE)
            actual_pad_top_value, actual_pad_left_value = cur_pad_top[0], pad_l
            with tvm_ir.for_range(
                    0, out_h, name="ho_o", dtype=SCALAR_DTYPE) as ho_o:
                tiling_l1_wo_o = (out_w + tiling_l1_wo_i - 1) // tiling_l1_wo_i
                actual_tiling_l1_wo_o = tiling_l1_wo_o
                actual_tiling_l1_wo_i = tiling_l1_wo_i
                if out_w % tiling_l1_wo_i != 0:
                    actual_tiling_l1_wo_o -= 1
                with tvm_ir.for_range(
                        0, actual_tiling_l1_wo_o, name="wo_o",
                        dtype=SCALAR_DTYPE) as wo_o:
                    _calculation(0, actual_tiling_l1_wo_i, ho_o, wo_o,
                                 actual_pad_top_value, actual_pad_left_value)
                    if actual_tiling_l1_wo_o > 1:
                        actual_pad_left_value = 0
                if actual_tiling_l1_wo_o != tiling_l1_wo_o:
                    wo_o = actual_tiling_l1_wo_o
                    actual_tiling_l1_wo_i = out_w - wo_o*tiling_l1_wo_i
                    actual_pad_left_value = 0
                    _calculation(0, actual_tiling_l1_wo_i, ho_o, wo_o,
                                 actual_pad_top_value, actual_pad_left_value)
                if out_h > 1:
                    cur_pad_top[0] = tvm.const(0, FMATRIX_DTYPE)
                    actual_pad_top_value, actual_pad_left_value = cur_pad_top[
                        0], pad_l
        else:
            tiling_l1_ho_o = (out_h + tiling_l1_ho_i - 1) // tiling_l1_ho_i
            actual_tiling_l1_ho_o = tiling_l1_ho_o
            actual_tiling_l1_ho_i = tiling_l1_ho_i
            if out_h % tiling_l1_ho_i != 0:
                actual_tiling_l1_ho_o -= 1
            actual_pad_top_value, actual_pad_left_value = _set_const_fmatrix(
                l1_hi, pad_t, pad_b, pad_l, pad_r)
            with tvm_ir.for_range(
                    0, actual_tiling_l1_ho_o, name="ho_o",
                    dtype=SCALAR_DTYPE) as ho_o:
                _calculation(actual_tiling_l1_ho_i, 0, ho_o, None,
                             actual_pad_top_value, actual_pad_left_value)
                if actual_tiling_l1_ho_o > 1:
                    actual_pad_top_value, actual_pad_left_value = \
                        _set_const_fmatrix(l1_hi, 0, 0, pad_l, pad_r)
            if actual_tiling_l1_ho_o != tiling_l1_ho_o:
                ho_o = actual_tiling_l1_ho_o
                actual_tiling_l1_ho_i = out_h - ho_o*tiling_l1_ho_i
                last_hi = fmap_h + pad_t - ho_o*tiling_l1_ho_i*stride_h
                actual_pad_top_value, actual_pad_left_value = \
                    _set_const_fmatrix(last_hi, 0, pad_b, pad_l, pad_r)
                _calculation(actual_tiling_l1_ho_i, 0, ho_o, None,
                             actual_pad_top_value, actual_pad_left_value)

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ir.scope_attr(block_index, "thread_extent", block_tiling["block_dim"])

    const_0_ub = _dup_const_0()
    if block_tiling["type"] == BlockTilingType.DIVISIBLE:
        with tvm_ir.for_range(
                0, block_tiling["shape"]["n"], name="n_i",
                dtype=SCALAR_DTYPE) as n_i:  # tiling for batch
            with tvm_ir.for_range(
                    0,
                    block_tiling["shape"]["c1"],
                    name="c1_i",
                    dtype=SCALAR_DTYPE) as c1_i:  # tiling for c1
                n_o = block_index // (fmap_c1 // block_tiling["shape"]["c1"])
                n = n_o*block_tiling["shape"]["n"] + n_i
                c1_o = block_index - n_o*(
                    fmap_c1 // block_tiling["shape"]["c1"])
                c1 = c1_o*block_tiling["shape"]["c1"] + c1_i
                _calculation_in_each_block()
    else:
        with tvm_ir.for_range(
                0,
                block_tiling["fuse_factor"],
                name="fuse_factor",
                dtype=SCALAR_DTYPE) as fuse_factor:
            nc1 = block_index*block_tiling["fuse_factor"] + fuse_factor
            with tvm_ir.if_scope(nc1 < block_tiling["fuse"]):
                n = nc1 // fmap_c1
                c1 = nc1 % fmap_c1
                _calculation_in_each_block()

    return tvm_ir.get()


@util.check_input_type(dict, dict, dict, dict, (list, tuple), (list, tuple),
                       str, str, str)
def max_pool_grad_grad(orig_x_dict,
                       orig_y_dict,
                       grads_dict,
                       output_dict,
                       ksize,
                       strides,
                       padding="SAME",
                       data_format="NHWC",
                       kernel_name="max_pool_grad_grad"):
    """
    Computes second-order gradients of the maxpooling function.

    Parameters:
    ----------
    orig_x_dict : the dict of orig_input. support data type: float16.
                      The original input tensor.

    orig_y_dict : the dict of orig_output. support data type: float16.
                       The original output tensor.

    grads_dict : the dict of grad. support data type: float16.
                Gradients of gradients w.r.t. the input of max_pool

    output_dict : the dict of output. support data type: float16.

    ksize : The size of the window for each dimension of the input tensor.

    strides : The stride of the sliding window for each dimension of
              the input tensor.

    padding : The type of padding algorithm to use. Default is "SAME".

    data_format: Specify the data format of the input and output data.

    kernel_name : cce kernel name, default value is "max_pool_grad_grad".

    Returns:
    -------
    None
    """
    dict_key_list = ("shape", "dtype", "ori_format")
    _check_dict_key(orig_x_dict, dict_key_list, "orig_input")
    _check_dict_key(orig_y_dict, dict_key_list, "orig_output")
    _check_dict_key(grads_dict, dict_key_list, "grad")
    _check_dict_key(output_dict, dict_key_list, "output")

    _check_dtype(orig_x_dict, orig_y_dict, grads_dict, output_dict)
    _check_shape(orig_x_dict, orig_y_dict, grads_dict, output_dict)
    _check_format(orig_x_dict, orig_y_dict, grads_dict, output_dict)

    padding = padding.upper()
    if padding not in ("SAME", "VALID"):
        raise RuntimeError("Padding must be SAME or VALID.")
    data_format = data_format.upper()
    if data_format not in ("NHWC", "NCHW"):
        raise RuntimeError("The orig_format in x must be NCHW or NHWC.")

    n_pos = data_format.find("N")
    c_pos = data_format.find("C")
    h_pos = data_format.find("H")
    w_pos = data_format.find("W")
    if len(ksize) != 4 or ksize[n_pos] != 1 or ksize[c_pos] != 1:
        raise RuntimeError("The shape of ksize must be (4,),"
                           " N-dim and C-dim must be 1.")
    if len(strides) != 4 or strides[n_pos] != 1 or strides[c_pos] != 1:
        raise RuntimeError("The shape of strides must be (4,),"
                           " N-dim and C-dim must be 1.")

    util.check_kernel_name(kernel_name)

    shape_in = orig_x_dict.get('shape')
    util.check_shape_rule(shape_in, SHAPE_DIM_SIZE, SHAPE_DIM_SIZE,
                          SHAPE_SIZE_LIMIT)
    shape_out = output_dict.get('shape')
    util.check_shape_rule(shape_out, SHAPE_DIM_SIZE, SHAPE_DIM_SIZE,
                          SHAPE_SIZE_LIMIT)

    if strides[h_pos] > 63 or strides[h_pos] < 1 or \
            strides[w_pos] > 63 or strides[w_pos] < 1:
        raise RuntimeError("Invalid strides params, "
                           "stride size must be [1, 63].")
    if ksize[h_pos] > 255 or ksize[h_pos] < 1 or \
            ksize[w_pos] > 255 or ksize[w_pos] < 1:
        raise RuntimeError("Invalid ksize params, "
                           "ksize size must be [1, 255].")

    out_n, out_c1, out_ho, out_wo, out_c0 = shape_out
    howo_pad = ((out_ho*out_wo + out_c0 - 1) // out_c0)*out_c0
    shape_load3d = (out_n, howo_pad, out_c1, ksize[h_pos]*ksize[n_pos], out_c0)
    util.check_shape_size(shape_load3d, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_in, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_out, SHAPE_SIZE_LIMIT)

    in_dtype = orig_x_dict.get('dtype').lower()
    orig_x = tvm.placeholder(shape_in, dtype=in_dtype, name='orig_x')
    grads = tvm.placeholder(shape_in, dtype=in_dtype, name='grads')
    orig_y = tvm.placeholder(shape_out, dtype=in_dtype, name='orig_y')

    ir_fun = lambda ins, outs: max_pool_grad_grad_ir_builder(
        ins, outs, (ksize[h_pos], ksize[w_pos]), (strides[h_pos], strides[
            w_pos]), padding, kernel_name)
    res = tvm.extern([shape_in, shape_out, shape_in], [orig_x, orig_y, grads],
                     ir_fun,
                     name="res",
                     dtype=in_dtype)

    sch = tvm.create_schedule(res.op)
    with tbe_platform.build_config:
        tvm.build(sch, [orig_x, orig_y, grads, res], "cce", name=kernel_name)
