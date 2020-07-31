#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

four_2_five
"""
from __future__ import absolute_import

from functools import reduce as functools_reduce

from te import platform as cce
from te.platform import cce_intrin as intrin
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_build import build_config
import te.platform.cce_params as cce_params
import te.lang.cce
from te import tvm
from topi import generic
from topi.cce import util

# available ub size
UB_SIZE_B = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE)
# available number of cores
AICORE_NUM = cce.cce_conf.get_soc_spec(cce.cce_conf.CORE_NUM)


# pylint: disable=locally-disabled, unused-argument, too-many-lines
@fusion_manager.register("four_2_five")
def four_2_five_compute(src, dst, src_format, dst_format,
                        kernel_name="four_2_five"):
    """
    algorithm: four_2_five
    doing four_2_five for various data format, such as from NHWC to NC1HWC0

    Parameters
    ----------
    src : TVM tensor
              data of input
    dst: dict
              shape and dtype of output, should be same shape and type as input
    src_format: str
              source data format, can be NHWC etc.
    dst_format: str
               target data format, can be NC1HWC0 etc.
    kernel_name: str
              kernel name, default value is "four_2_five"

    Returns
    -------
    res : TVM tensor
          the compute result
    """
    shape = te.lang.cce.util.shape_to_list(src.shape)
    res = te.lang.cce.compute_four_2_five(src, dst, shape, src_format,
                                          dst_format)

    return res


def _new_alloc(tvm_ib, dtype, shape, name, scope):
    """
    decl new buffer for ir builder make function

    """
    buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name,
                                 scope=scope, data=buf_var)

    return new_buffer


# pylint: disable=locally-disabled, too-many-locals, too-many-statements
def _move_full(dst, data):
    """
    move for full shape

    """
    tvm_ib = tvm.ir_builder.create()
    dim_ele = functools_reduce(lambda x, y: x * y, data.shape[1:])
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 32
    ub_ele = (ub_bytes // float_size // cp_align_len)*cp_align_len
    device_core_num = AICORE_NUM

    data_ub = _new_alloc(tvm_ib, data.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    n_i = data.shape[0]
    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    ub_loop = dim_ele // ub_ele
    ub_mod = dim_ele % ub_ele

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        with tvm_ib.for_range(0, ub_loop, name="num_uloop") as num_uloop:
            data_offset = num_g*device_core_num*dim_ele + block_index*dim_ele\
                          + num_uloop*ub_ele
            burst_len_data = ub_ele // cp_align_len
            tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                        data_ub.access_ptr("w", offset=0),
                                        data.access_ptr('r',
                                                        offset=data_offset),
                                        0, 1, burst_len_data, 0, 0))

            dst_offset = num_g*device_core_num*dim_ele + block_index*dim_ele\
                         + num_uloop*ub_ele
            burst_len_dst = ub_ele // cp_align_len
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_ub.access_ptr("r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))
        with tvm_ib.if_scope(ub_mod > 0):
            data_offset = num_g * device_core_num * dim_ele\
                          + block_index * dim_ele + ub_loop * ub_ele
            burst_len_data = ub_mod // cp_align_len
            tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                        data_ub.access_ptr("w", offset=0),
                                        data.access_ptr('r',
                                                        offset=data_offset),
                                        0, 1, burst_len_data, 0, 0))

            dst_offset = num_g * device_core_num * dim_ele\
                         + block_index * dim_ele + ub_loop * ub_ele
            burst_len_dst = ub_mod // cp_align_len
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_ub.access_ptr("r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            with tvm_ib.for_range(0, ub_loop, name="num_uloop") as num_uloop:
                data_offset = group_index * device_core_num * dim_ele\
                              + block_index * dim_ele + num_uloop * ub_ele
                burst_len_data = ub_ele // cp_align_len
                tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                            data_ub.access_ptr("w", offset=0),
                                            data.access_ptr('r',
                                                            offset=data_offset),
                                            0, 1, burst_len_data, 0, 0))

                dst_offset = group_index * device_core_num * dim_ele\
                             + block_index * dim_ele + num_uloop * ub_ele
                burst_len_dst = ub_ele // cp_align_len
                tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr('w',
                                                           offset=dst_offset),
                                            data_ub.access_ptr("r", offset=0),
                                            0, 1, burst_len_dst, 0, 0))
            with tvm_ib.if_scope(ub_mod > 0):
                data_offset = group_index * device_core_num * dim_ele\
                              + block_index * dim_ele + ub_loop * ub_ele
                burst_len_data = ub_mod // cp_align_len
                tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                            data_ub.access_ptr("w", offset=0),
                                            data.access_ptr('r',
                                                            offset=data_offset),
                                            0, 1, burst_len_data, 0, 0))

                dst_offset = group_index * device_core_num * dim_ele\
                             + block_index * dim_ele + ub_loop * ub_ele
                burst_len_dst = ub_mod // cp_align_len
                tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr('w',
                                                           offset=dst_offset),
                                            data_ub.access_ptr("r", offset=0),
                                            0, 1, burst_len_dst, 0, 0))

    return tvm_ib.get()


def _func_vadds_for_move(args):
    """
    function of moving data with vadds function

    """
    tvm_ib, data_ub, data_res, c_1, ub_offset, res_offset, repeat,\
    srcm0, dstm0, srcm1, dstm1, cp_align_len = args
    max_r = 255

    with tvm_ib.if_scope(repeat <= max_r):
        with tvm_ib.if_scope(repeat == 1):
            tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                        data_res.access_ptr("w",
                                                            offset=res_offset),
                                        data_ub.access_ptr('r',
                                                           offset=ub_offset),
                                        0, repeat, dstm0, srcm0, 0, 0))
        with tvm_ib.else_scope():
            tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                        data_res.access_ptr("w",
                                                            offset=res_offset),
                                        data_ub.access_ptr('r',
                                                           offset=ub_offset),
                                        0, repeat, dstm0, srcm0, dstm1, srcm1))
    with tvm_ib.else_scope():
        zu_repeat = repeat // max_r
        mod_repeat = repeat % max_r
        with tvm_ib.for_range(0, zu_repeat, name="num_zr") as num_zr:
            ub_offset_cur = ub_offset + num_zr*max_r*8*c_1*cp_align_len
            res_offset_cur = res_offset + num_zr*max_r*8*cp_align_len
            tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                        data_res.access_ptr(
                                            "w", offset=res_offset_cur),
                                        data_ub.access_ptr(
                                            "r", offset=ub_offset_cur),
                                        0, max_r, dstm0, srcm0, dstm1, srcm1))
        with tvm_ib.if_scope(mod_repeat > 0):
            ub_offset_cur = ub_offset + zu_repeat*max_r*8*c_1*cp_align_len
            res_offset_cur = res_offset + zu_repeat*max_r*8*cp_align_len
            with tvm_ib.if_scope(mod_repeat == 1):
                tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                            data_res.access_ptr(
                                                "w", offset=res_offset_cur),
                                            data_ub.access_ptr(
                                                "r", offset=ub_offset_cur),
                                            0, mod_repeat, dstm0, srcm0, 0, 0))
            with tvm_ib.else_scope():
                tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                            data_res.access_ptr(
                                                "w", offset=res_offset_cur),
                                            data_ub.access_ptr(
                                                "r", offset=ub_offset_cur),
                                            0, mod_repeat, dstm0, srcm0, dstm1,
                                            srcm1))


def _set_mask_slice_fp16(before, after):
    """
    calculate MASK in cce

    """
    before = int(before)
    after = int(after)

    if after >= 64:
        mask1 = 0
        after_new = after - 64
        mask2 = (2**(64 - after_new) - 1) - (2**before - 1)
    elif before >= 64:
        before_new = before - 64
        mask1 = (2**(64 - after) - 1) - (2**before_new - 1)
        mask2 = 0
    else:
        mask1 = (2**(64 - after) - 1)
        mask2 = (2**64 - 1) - (2**before - 1)

    return mask1, mask2


def _func_ci_align_nhwc_fp16(args):
    """
    function of moving data for ci align nhwc float16 scene

    """
    tvm_ib, data, dst, data_ub, data_res, device_core_num, block_index,\
    h_i, w_i, c_i, c_1, c_0, col_len_shape, col_len_ub, cp_align_len,\
    ub_loop, ub_mod, num_g = args

    n_index = num_g * device_core_num + block_index

    with tvm_ib.for_range(0, ub_loop, name="num_u") as num_u:
        data_offset = n_index * h_i * w_i * c_i \
                      + num_u * col_len_ub * c_i
        burst_len = col_len_ub * c_1
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr(
                                        "w", offset=0),
                                    data.access_ptr(
                                        "r", offset=data_offset),
                                    0, 1, burst_len, 0, 0))

        vadds_zu = col_len_ub // 8
        vadds_mod = col_len_ub % 8
        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            with tvm_ib.if_scope(vadds_zu > 0):
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(-1, dtype="uint64"),
                    tvm.const(-1, dtype="uint64")))

                ub_offset = num_c1 * c_0
                res_offset = num_c1 * col_len_ub * c_0
                repeat = vadds_zu
                srcm0 = c_1
                dstm0 = 1
                srcm1 = 8 * c_1
                dstm1 = 8
                args = tvm_ib, data_ub, data_res, c_1, ub_offset, \
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1, \
                       cp_align_len
                _func_vadds_for_move(args)
            with tvm_ib.if_scope(vadds_mod > 0):
                ub_offset = num_c1 * c_0 + vadds_zu * 8 * c_i
                res_offset = num_c1 * col_len_ub * c_0 + vadds_zu * 8 * c_0
                repeat = 1
                srcm0 = c_1
                dstm0 = 1
                srcm1 = 0
                dstm1 = 0
                before = 0
                after = (8 - vadds_mod) * 16
                mask1, mask2 = _set_mask_slice_fp16(before, after)
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                args = tvm_ib, data_ub, data_res, c_1, ub_offset, \
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1, \
                       cp_align_len
                _func_vadds_for_move(args)

        tvm_ib.emit(tvm.call_extern(
            dst.dtype, "set_vector_mask",
            tvm.const(-1, dtype="uint64"),
            tvm.const(-1, dtype="uint64")))

        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            dst_offset = n_index * h_i * w_i * c_i \
                         + num_c1 * col_len_shape * c_0 \
                         + num_u * col_len_ub * c_0
            res_offset = num_c1 * col_len_ub * c_0
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr("r",
                                                            offset=res_offset),
                                        0, 1, col_len_ub, 0, 0))
    with tvm_ib.if_scope(ub_mod > 0):
        data_offset = n_index * h_i * w_i * c_i \
                      + ub_loop * col_len_ub * c_i
        burst_len = ub_mod * c_1
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr(
                                        "w", offset=0),
                                    data.access_ptr(
                                        "r", offset=data_offset),
                                    0, 1, burst_len, 0, 0))

        vadds_zu = ub_mod // 8
        vadds_mod = ub_mod % 8
        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            with tvm_ib.if_scope(vadds_zu > 0):
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(-1, dtype="uint64"),
                    tvm.const(-1, dtype="uint64")))

                ub_offset = num_c1 * c_0
                res_offset = num_c1 * ub_mod * c_0
                repeat = vadds_zu
                srcm0 = c_1
                dstm0 = 1
                srcm1 = 8 * c_1
                dstm1 = 8
                args = tvm_ib, data_ub, data_res, c_1, ub_offset, \
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1, \
                       cp_align_len
                _func_vadds_for_move(args)
            with tvm_ib.if_scope(vadds_mod > 0):
                ub_offset = num_c1 * c_0 + vadds_zu * 8 * c_i
                res_offset = num_c1 * ub_mod * c_0 + vadds_zu * 8 * c_0
                repeat = 1
                srcm0 = c_1
                dstm0 = 1
                srcm1 = 0
                dstm1 = 0
                before = 0
                after = (8 - vadds_mod) * 16
                mask1, mask2 = _set_mask_slice_fp16(before, after)
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                args = tvm_ib, data_ub, data_res, c_1, ub_offset, \
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1, \
                       cp_align_len
                _func_vadds_for_move(args)

        tvm_ib.emit(tvm.call_extern(
            dst.dtype, "set_vector_mask",
            tvm.const(-1, dtype="uint64"),
            tvm.const(-1, dtype="uint64")))

        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            dst_offset = n_index * h_i * w_i * c_i \
                         + num_c1 * col_len_shape * c_0 \
                         + ub_loop * col_len_ub * c_0
            res_offset = num_c1 * ub_mod * c_0
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr("r",
                                                            offset=res_offset),
                                        0, 1, ub_mod, 0, 0))


def _ci_align_nhwc_fp16(dst, data):
    """
    function of making ir node builder for ci align nhwc float16 scene

    """
    tvm_ib = tvm.ir_builder.create()

    n_i, h_i, w_i, c_i = data.shape
    c_0 = 16
    c_1 = c_i // c_0

    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 1024
    ub_ele = (ub_bytes // 2 // float_size // c_i) * c_i

    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    col_len_shape = h_i * w_i
    col_len_ub = ub_ele // c_i
    ub_loop = col_len_shape // col_len_ub
    ub_mod = col_len_shape % col_len_ub

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, device_core_num,\
               block_index, h_i, w_i, c_i, c_1, c_0, col_len_shape,\
               col_len_ub, cp_align_len, ub_loop, ub_mod, num_g
        _func_ci_align_nhwc_fp16(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, device_core_num, \
                   block_index, h_i, w_i, c_i, c_1, c_0, col_len_shape, \
                   col_len_ub, cp_align_len, ub_loop, ub_mod, group_index
            _func_ci_align_nhwc_fp16(args)

    return tvm_ib.get()


def _func_ci_align_nhwc_fp16_one(args):
    """
    function of moving data for ci align nhwc float16 scene

    """
    tvm_ib, data, dst, data_ub, data_res, block_index,\
    h_i, w_i, c_i, c_1, c_0, col_len_shape, col_len_ub, cp_align_len,\
    ub_loop, ub_mod = args

    with tvm_ib.for_range(0, ub_loop, name="num_u") as num_u:
        data_offset = block_index * col_len_shape * c_i \
                      + num_u * col_len_ub * c_i
        burst_len = col_len_ub * c_1
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr(
                                        "w", offset=0),
                                    data.access_ptr(
                                        "r", offset=data_offset),
                                    0, 1, burst_len, 0, 0))

        vadds_zu = col_len_ub // 8
        vadds_mod = col_len_ub % 8
        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            with tvm_ib.if_scope(vadds_zu > 0):
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(-1, dtype="uint64"),
                    tvm.const(-1, dtype="uint64")))

                ub_offset = num_c1 * c_0
                res_offset = num_c1 * col_len_ub * c_0
                repeat = vadds_zu
                srcm0 = c_1
                dstm0 = 1
                srcm1 = 8 * c_1
                dstm1 = 8
                args = tvm_ib, data_ub, data_res, c_1, ub_offset, \
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1, \
                       cp_align_len
                _func_vadds_for_move(args)
            with tvm_ib.if_scope(vadds_mod > 0):
                ub_offset = num_c1 * c_0 + vadds_zu * 8 * c_i
                res_offset = num_c1 * col_len_ub * c_0 + vadds_zu * 8 * c_0
                repeat = 1
                srcm0 = c_1
                dstm0 = 1
                srcm1 = 0
                dstm1 = 0
                before = 0
                after = (8 - vadds_mod) * 16
                mask1, mask2 = _set_mask_slice_fp16(before, after)
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                args = tvm_ib, data_ub, data_res, c_1, ub_offset, \
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1, \
                       cp_align_len
                _func_vadds_for_move(args)

        tvm_ib.emit(tvm.call_extern(
            dst.dtype, "set_vector_mask",
            tvm.const(-1, dtype="uint64"),
            tvm.const(-1, dtype="uint64")))

        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            dst_offset = block_index * col_len_shape * c_0 \
                         + num_c1 * h_i * w_i * c_0 \
                         + num_u * col_len_ub * c_0
            res_offset = num_c1 * col_len_ub * c_0
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr("r",
                                                            offset=res_offset),
                                        0, 1, col_len_ub, 0, 0))
    with tvm_ib.if_scope(ub_mod > 0):
        data_offset = block_index * col_len_shape * c_i \
                      + ub_loop * col_len_ub * c_i
        burst_len = ub_mod * c_1
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr(
                                        "w", offset=0),
                                    data.access_ptr(
                                        "r", offset=data_offset),
                                    0, 1, burst_len, 0, 0))

        vadds_zu = ub_mod // 8
        vadds_mod = ub_mod % 8
        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            with tvm_ib.if_scope(vadds_zu > 0):
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(-1, dtype="uint64"),
                    tvm.const(-1, dtype="uint64")))

                ub_offset = num_c1 * c_0
                res_offset = num_c1 * ub_mod * c_0
                repeat = vadds_zu
                srcm0 = c_1
                dstm0 = 1
                srcm1 = 8 * c_1
                dstm1 = 8
                args = tvm_ib, data_ub, data_res, c_1, ub_offset, \
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1, \
                       cp_align_len
                _func_vadds_for_move(args)
            with tvm_ib.if_scope(vadds_mod > 0):
                ub_offset = num_c1 * c_0 + vadds_zu * 8 * c_i
                res_offset = num_c1 * ub_mod * c_0 + vadds_zu * 8 * c_0
                repeat = 1
                srcm0 = c_1
                dstm0 = 1
                srcm1 = 0
                dstm1 = 0
                before = 0
                after = (8 - vadds_mod) * 16
                mask1, mask2 = _set_mask_slice_fp16(before, after)
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                args = tvm_ib, data_ub, data_res, c_1, ub_offset, \
                       res_offset, repeat, srcm0, dstm0, srcm1, dstm1, \
                       cp_align_len
                _func_vadds_for_move(args)

        tvm_ib.emit(tvm.call_extern(
            dst.dtype, "set_vector_mask",
            tvm.const(-1, dtype="uint64"),
            tvm.const(-1, dtype="uint64")))

        with tvm_ib.for_range(0, c_1, name="num_c1") as num_c1:
            dst_offset = block_index * col_len_shape * c_0 \
                         + num_c1 * h_i * w_i * c_0 \
                         + ub_loop * col_len_ub * c_0
            res_offset = num_c1 * ub_mod * c_0
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr("r",
                                                            offset=res_offset),
                                        0, 1, ub_mod, 0, 0))


def _ci_align_nhwc_fp16_one(dst, data):
    """
    function of making ir node builder for ci align nhwc float16 scene

    """
    tvm_ib = tvm.ir_builder.create()

    _, h_i, w_i, c_i = data.shape
    c_0 = 16
    c_1 = c_i // c_0

    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 1024
    ub_ele = (ub_bytes // 2 // float_size // c_i) * c_i

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    col_len_shape = h_i * w_i // device_core_num
    col_len_ub = ub_ele // c_i
    ub_loop = col_len_shape // col_len_ub
    ub_mod = col_len_shape % col_len_ub

    args = tvm_ib, data, dst, data_ub, data_res, \
           block_index, h_i, w_i, c_i, c_1, c_0, col_len_shape, \
           col_len_ub, cp_align_len, ub_loop, ub_mod
    _func_ci_align_nhwc_fp16_one(args)

    return tvm_ib.get()


def _set_mask(length):
    """
    calculate MASK in cce

    """
    length = int(length)
    mask1 = 2**max(length - 64, 0) - 1  # high 64bits
    mask2 = 2**min(length, 64) - 1  # low 64bits
    return mask1, mask2


def _clean_ubuf(ib_, src, src_offset, dup_len):
    """
    clean the ubuf
    """
    uint64_all_one = tvm.const(2**64 - 1, dtype="uint64")
    uint64_all_zero = tvm.const(0, dtype="uint64")
    dtype_factor = 32 // intrin.get_bit_len(src.dtype)
    dup_value = tvm.const(0.0, dtype=src.dtype)
    batch_cnt = 64  # one repeate can process 64 elements of float32

    if dup_len > 0:
        if src.dtype == "float16":
            ib_.emit(
                tvm.call_extern("uint64", 'set_vector_mask', uint64_all_one,
                                uint64_all_one))
        else:
            ib_.emit(
                tvm.call_extern("uint64", 'set_vector_mask', uint64_all_zero,
                                uint64_all_one))

        repeat = dup_len // (batch_cnt * dtype_factor)
        dup_left = dup_len % (batch_cnt * dtype_factor)
        if repeat >= 255:  # vector_dup only can support max repeat 255
            repeat_loop = (repeat + 255 - 1) // 255
            with ib_.for_range(0, repeat_loop) as i:
                with ib_.if_scope(i != repeat_loop - 1):
                    ib_.emit(
                        tvm.call_extern(
                            src.dtype, 'vector_dup',
                            src.access_ptr(
                                "rw",
                                offset=(255 * batch_cnt * dtype_factor) * i +
                                src_offset), dup_value, 255, 1, 1, 8, 8))
                with ib_.else_scope():
                    ib_.emit(
                        tvm.call_extern(
                            src.dtype, "vector_dup",
                            src.access_ptr(
                                "rw",
                                offset=(255 * batch_cnt * dtype_factor) * i +
                                src_offset), dup_value, repeat % 255, 1, 1, 8,
                            8))

        else:
            ib_.emit(
                tvm.call_extern(src.dtype, "vector_dup",
                                src.access_ptr("rw", offset=src_offset),
                                dup_value, repeat, 1, 1, 8, 8))

            if dup_left > 0:
                if dup_left > 64:
                    high_mask = tvm.const(2 ** (dup_left % 64) - 1,
                                          dtype="uint64")
                    ib_.emit(
                        tvm.call_extern("uint64", 'set_vector_mask',
                                        high_mask,
                                        uint64_all_one))
                elif 0 < dup_left <= 64:
                    low_mask = tvm.const(2 ** dup_left - 1, dtype="uint64")
                    ib_.emit(
                        tvm.call_extern("uint64", 'set_vector_mask',
                                        uint64_all_zero,
                                        low_mask))
                ib_.emit(
                    tvm.call_extern(src.dtype, "vector_dup",
                                    src.access_ptr(
                                        "rw",
                                        offset=
                                        src_offset +
                                        repeat * batch_cnt * dtype_factor),
                                    dup_value, 1, 1, 1, 8, 8))

        ib_.emit(
            tvm.call_extern("uint64", 'set_vector_mask', uint64_all_one,
                            uint64_all_one))


# pylint: disable=locally-disabled,too-many-return-statements
def _func_vadds_for_vconv(args):
    """
    function of moving data with vadds function

    """
    tvm_ib, data_ub, data_res, c_0, ub_offset, res_offset, repeat,\
    srcm0, dstm0, srcm1, dstm1 = args
    max_r = 255

    with tvm_ib.if_scope(repeat <= max_r):
        with tvm_ib.if_scope(repeat == 1):
            tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                        data_res.access_ptr("w",
                                                            offset=res_offset),
                                        data_ub.access_ptr('r',
                                                           offset=ub_offset),
                                        0, repeat, dstm0, srcm0, 0, 0))
        with tvm_ib.else_scope():
            tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                        data_res.access_ptr("w",
                                                            offset=res_offset),
                                        data_ub.access_ptr('r',
                                                           offset=ub_offset),
                                        0, repeat, dstm0, srcm0, dstm1, srcm1))
    with tvm_ib.else_scope():
        zu_repeat = repeat // max_r
        mod_repeat = repeat % max_r
        with tvm_ib.for_range(0, zu_repeat, name="num_zr") as num_zr:
            ub_offset_cur = ub_offset + num_zr*max_r*c_0*8
            res_offset_cur = res_offset + num_zr*max_r*256*8
            tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                        data_res.access_ptr(
                                            "w", offset=res_offset_cur),
                                        data_ub.access_ptr(
                                            'r', offset=ub_offset_cur),
                                        0, max_r, dstm0, srcm0, dstm1, srcm1))
        with tvm_ib.if_scope(mod_repeat > 0):
            ub_offset_cur = ub_offset + zu_repeat*max_r*c_0*8
            res_offset_cur = res_offset + zu_repeat*max_r*256*8
            with tvm_ib.if_scope(mod_repeat == 1):
                tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                            data_res.access_ptr(
                                                "w", offset=res_offset_cur),
                                            data_ub.access_ptr(
                                                'r', offset=ub_offset_cur),
                                            0, mod_repeat, dstm0, srcm0, 0, 0))
            with tvm_ib.else_scope():
                tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                            data_res.access_ptr(
                                                "w", offset=res_offset_cur),
                                            data_ub.access_ptr(
                                                'r', offset=ub_offset_cur),
                                            0, mod_repeat, dstm0, srcm0, dstm1,
                                            srcm1))


def _vconv_one(args):
    """
    function of vnchwconv for func_move_vconv_one_diff

    """
    tvm_ib, addr_array, addr_array_buf, one_begin, two_begin,\
    repeat_vconv, src_stride_vconv, dst_stride_vconv = args

    src0_offset = 8*0
    src1_offset = 8*1
    dst0_offset = 8*2
    dst1_offset = 8*3
    src_gap = 32
    src_eight_gap = src_gap*8

    with tvm_ib.for_range(0, 8, name="i") as i:
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src0_offset + i]),
                                    one_begin + i * src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src1_offset + i]),
                                    one_begin + src_eight_gap
                                    + i * src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst0_offset + i]),
                                    two_begin + i * src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst1_offset + i]),
                                    two_begin + src_eight_gap
                                    + i * src_gap))

    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA0",
                                addr_array_buf.access_ptr("rw",
                                                          offset=
                                                          src0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA1",
                                addr_array_buf.access_ptr("rw",
                                                          offset=
                                                          src1_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA2",
                                addr_array_buf.access_ptr("rw",
                                                          offset=
                                                          dst0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA3",
                                addr_array_buf.access_ptr("rw",
                                                          offset=
                                                          dst1_offset)))

    with tvm_ib.if_scope(repeat_vconv == 1):
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    1,
                                    0,
                                    0))
    with tvm_ib.else_scope():
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    repeat_vconv,
                                    dst_stride_vconv,
                                    src_stride_vconv))


def _func_move_vconv_one_diff(args):
    """
    function of moving data for vconv one diff scene

    """
    tvm_ib, data, dst, data_one, data_two, data_res, addr_array,\
    addr_array_buf, device_core_num, block_index, cp_align_len,\
    shape_5hd, shape_nhwc, ub_ele_one, ub_ele, float_size, num_g = args

    _, h_i, w_i, _ = shape_nhwc
    c_0 = shape_5hd[4]

    dim_ele = h_i*w_i*c_0
    ub_loop = dim_ele // ub_ele_one
    ele_mod = dim_ele % ub_ele_one

    n_index = num_g * device_core_num + block_index

    _clean_ubuf(tvm_ib, data_one, 0, ub_ele)
    _clean_ubuf(tvm_ib, data_two, 0, ub_ele_one)
    _clean_ubuf(tvm_ib, data_res, 0, ub_ele_one)

    with tvm_ib.for_range(0, ub_loop, name="num_u") as num_u:
        data_offset = n_index*h_i*w_i + num_u*ub_ele
        burst_len_data = ub_ele // cp_align_len
        tvm_ib.emit(tvm.call_extern(data_one.dtype, "copy_gm_to_ubuf",
                                    data_one.access_ptr("w", offset=0),
                                    data.access_ptr('r',
                                                    offset=data_offset),
                                    0, 1, burst_len_data, 0, 0))

        block_num = ub_ele // 16
        repeat_index = block_num // 8
        block_mod = block_num % 8
        with tvm_ib.if_scope(repeat_index > 0):
            one_offset = 0
            two_offset = 0
            srcm0 = 1
            dstm0 = 16
            srcm1 = 8
            dstm1 = 16*8
            args = tvm_ib, data_one, data_two, c_0, one_offset, two_offset,\
                   repeat_index, srcm0, dstm0, srcm1, dstm1
            _func_vadds_for_vconv(args)
        with tvm_ib.if_scope(block_mod > 0):
            mask1, mask2 = _set_mask(block_mod * c_0)
            tvm_ib.emit(tvm.call_extern(
                "float16", "set_vector_mask",
                tvm.const(mask1, dtype="uint64"),
                tvm.const(mask2, dtype="uint64")))
            one_offset = repeat_index*16*8
            two_offset = repeat_index*256*8
            srcm0 = 1
            dstm0 = 16
            tvm_ib.emit(tvm.call_extern(data_two.dtype, "vadds",
                                        data_two.access_ptr("w",
                                                            offset=two_offset),
                                        data_one.access_ptr('r',
                                                            offset=one_offset),
                                        0, 1, dstm0, srcm0, 0, 0))
            tvm_ib.emit(tvm.call_extern(
                "float16", "set_vector_mask",
                tvm.const(-1, dtype="uint64"),
                tvm.const(-1, dtype="uint64")))

        two_begin = ub_ele*float_size
        res_begin = (ub_ele + ub_ele_one)*float_size
        repeat_vconv = ub_ele // 16
        src_stride_vconv = 16
        dst_stride_vconv = 16
        args = tvm_ib, addr_array, addr_array_buf, two_begin, res_begin,\
               repeat_vconv, src_stride_vconv, dst_stride_vconv
        _vconv_one(args)

        dst_offset = n_index*dim_ele + num_u*ub_ele_one
        burst_len = ub_ele_one // cp_align_len
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr("r",
                                                        offset=0),
                                    0, 1, burst_len, 0, 0))

    with tvm_ib.if_scope(ele_mod > 0):
        data_offset = n_index*h_i*w_i + ub_loop*ub_ele
        burst_len_data = ele_mod // 16 // cp_align_len
        tvm_ib.emit(tvm.call_extern(data_one.dtype, "copy_gm_to_ubuf",
                                    data_one.access_ptr("w", offset=0),
                                    data.access_ptr('r',
                                                    offset=data_offset),
                                    0, 1, burst_len_data, 0, 0))

        block_num = ele_mod // 256
        repeat_index = block_num // 8
        block_mod = block_num % 8
        with tvm_ib.if_scope(repeat_index > 0):
            one_offset = 0
            two_offset = 0
            srcm0 = 1
            dstm0 = 16
            srcm1 = 8
            dstm1 = 16*8
            args = tvm_ib, data_one, data_two, c_0, one_offset, two_offset,\
                   repeat_index, srcm0, dstm0, srcm1, dstm1
            _func_vadds_for_vconv(args)
        with tvm_ib.if_scope(block_mod > 0):
            mask1, mask2 = _set_mask(block_mod * c_0)
            tvm_ib.emit(tvm.call_extern(
                "float16", "set_vector_mask",
                tvm.const(mask1, dtype="uint64"),
                tvm.const(mask2, dtype="uint64")))
            one_offset = repeat_index*16*8
            two_offset = repeat_index*256*8
            srcm0 = 1
            dstm0 = 16
            tvm_ib.emit(tvm.call_extern(data_two.dtype, "vadds",
                                        data_two.access_ptr("w",
                                                            offset=two_offset),
                                        data_one.access_ptr('r',
                                                            offset=one_offset),
                                        0, 1, dstm0, srcm0, 0, 0))
            tvm_ib.emit(tvm.call_extern(
                "float16", "set_vector_mask",
                tvm.const(-1, dtype="uint64"),
                tvm.const(-1, dtype="uint64")))

        two_begin = ub_ele*float_size
        res_begin = (ub_ele + ub_ele_one)*float_size
        repeat_vconv = ele_mod // 256
        src_stride_vconv = 16
        dst_stride_vconv = 16
        args = tvm_ib, addr_array, addr_array_buf, two_begin, res_begin,\
               repeat_vconv, src_stride_vconv, dst_stride_vconv
        _vconv_one(args)

        dst_offset = n_index*dim_ele + ub_loop*ub_ele_one
        burst_len = ele_mod // cp_align_len
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_res.access_ptr("r",
                                                        offset=0),
                                    0, 1, burst_len, 0, 0))


def _move_vconv_one_diff(dst, data):
    """
    function of making ir node builder for vconv one diff scene

    """
    tvm_ib = tvm.ir_builder.create()

    n_a = data.shape[0]
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 32
    ub_ele = (ub_bytes // float_size // 33 // cp_align_len)*cp_align_len
    ub_ele_one = ub_ele*16
    device_core_num = AICORE_NUM

    data_one = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_one", scope=cce.scope_ubuf)
    data_two = _new_alloc(tvm_ib, dst.dtype, ub_ele_one,
                          "data_two", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele_one,
                          "data_res", scope=cce.scope_ubuf)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    group_index = n_a // device_core_num
    group_mod = n_a % device_core_num

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_one, data_two, data_res,\
               addr_array, addr_array_buf, device_core_num, block_index, \
               cp_align_len, dst.shape, data.shape,\
               ub_ele_one, ub_ele, float_size, num_g
        _func_move_vconv_one_diff(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_one, data_two, data_res, \
                   addr_array, addr_array_buf, device_core_num, block_index, \
                   cp_align_len, dst.shape, data.shape, \
                   ub_ele_one, ub_ele, float_size, group_index
            _func_move_vconv_one_diff(args)

    return tvm_ib.get()


def _check_parameters(src, dst, src_format, dst_format):
    """
    check the parameters including src_shape, dst_shape,
    src_format, dst_format, dtype and kernel_name

    """
    src_shape = src.get("shape")
    dtype = src.get("dtype")
    dtype_dst = dst.get("dtype")

    if src_format.lower() != "nchw" and src_format.lower() != "nhwc":
        raise RuntimeError("src_format must be NCHW or NHWC!")

    if dst_format.lower() != "nc1hwc0":
        raise RuntimeError("dst_format must be NC1HWC0 !")

    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype, check_list)
    if dtype != dtype_dst:
        raise RuntimeError("dtype of src and dst are different !")

    util.check_shape_rule(src_shape, 4, 4)
    util.check_tensor_shape_size(src_shape)


def _check_move_full(shape_nhwc, src_format, dtype):
    if src_format.lower() != "nhwc":
        return False

    if dtype != "float16":
        return False

    n_i, h_i, w_i, c_i = shape_nhwc
    c_0 = 16

    if c_i != c_0:
        return False

    device_core_num = AICORE_NUM
    if n_i == 1:
        if h_i*w_i % device_core_num > 0:
            return False
    elif n_i < device_core_num:
        return False

    return True


def _check_ci_align_nhwc_fp16(shape_nhwc, src_format, dtype):
    """
    check whether to use ci align nhwc fp16 branch

    """
    if src_format.lower() != "nhwc":
        return False

    if dtype != "float16":
        return False

    n_i, h_i, w_i, c_i = shape_nhwc
    c_0 = 16

    if c_i % c_0 > 0:
        return False

    if c_i > 256:
        return False

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    ub_bytes = UB_SIZE_B - 1024
    ub_ele_a = ub_bytes // 2 // float_size

    if ub_ele_a < c_i:
        return False

    device_core_num = AICORE_NUM
    if n_i == 1:
        if h_i*w_i % device_core_num > 0:
            return False
    elif n_i < device_core_num:
        return False

    return True


def _check_vconv_one(shape_nhwc, src_format, dtype):
    """
    check whether vconv function

    """
    if src_format.lower() != "nhwc":
        return False

    if dtype != "float16":
        return False

    n_i, h_i, w_i, c_i = shape_nhwc

    if c_i != 1:
        return False

    if h_i*w_i % 256 > 0:
        return False

    device_core_num = AICORE_NUM
    if n_i == 1:
        zu_g = h_i*w_i // 256
        if zu_g % device_core_num > 0:
            return False
    elif n_i < device_core_num:
        return False

    return True


@util.check_input_type(dict, dict, str, str, str)
def four_2_five(src, dst, src_format, dst_format, kernel_name="four_2_five"):
    """
    algorithm: four_2_five
    doing four_2_five for various data format, such as from NHWC to NC1HWC0

    Parameters
    ----------
    src : TVM tensor
              data of input
    dst: dict
              shape and dtype of output, should be same shape and type as input
    src_format: str
              source data format, can be NHWC etc.
    dst_format: str
               target data format, can be NC1HWC0 etc.
    kernel_name: str
              kernel name, default value is "four_2_five"

    Returns
    -------
    None
    """
    shape_input = list(src.get("shape"))
    dtype_input = src.get("dtype").lower()
    shape_output = list(dst.get("shape"))
    dtype_output = dst.get("dtype").lower()

    util.check_shape_rule(shape_input)
    util.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype_input, check_list)
    util.check_tensor_shape_size(shape_input)

    _check_parameters(src, dst, src_format, dst_format)
    dtype = src.get("dtype")
    if _check_move_full(shape_input, src_format, dtype):
        n_i = shape_input[0]
        if n_i == 1:
            h_i = shape_input[1]
            w_i = shape_input[2]
            device_core_num = AICORE_NUM
            n_new = device_core_num
            hw_i = h_i * w_i // device_core_num
            h_new = 1
            w_new = hw_i
            shape_input[0] = n_new
            shape_input[1] = h_new
            shape_input[2] = w_new
            shape_output[0] = n_new
            shape_output[1] = 1
            shape_output[2] = h_new
            shape_output[3] = w_new
            shape_output[4] = 16
        data = tvm.placeholder(shape_input, dtype=dtype, name="data")
        res = tvm.extern(shape_output, [data],
                         lambda ins, outs: _move_full(outs[0],
                                                      ins[0]),
                         name="res", dtype=dtype)
        tensor_list = [data, res]
        sch = tvm.create_schedule(res.op)

        with build_config:
            tvm.build(sch, tensor_list, "cce", name=kernel_name)
    elif _check_ci_align_nhwc_fp16(shape_input, src_format, dtype):
        n_i = shape_input[0]
        data = tvm.placeholder(shape_input, dtype=dtype, name="data")
        if n_i == 1:
            res = tvm.extern(shape_output, [data],
                             lambda ins, outs: _ci_align_nhwc_fp16_one(outs[0],
                                                                       ins[0]),
                             name="res", dtype=dtype)
        else:
            res = tvm.extern(shape_output, [data],
                             lambda ins, outs: _ci_align_nhwc_fp16(outs[0],
                                                                   ins[0]),
                             name="res", dtype=dtype)
        tensor_list = [data, res]
        sch = tvm.create_schedule(res.op)

        with build_config:
            tvm.build(sch, tensor_list, "cce", name=kernel_name)
    elif _check_vconv_one(shape_input, src_format, dtype_input):
        n_i, h_i, w_i, c_i = shape_input
        c_0 = 16
        if n_i == 1:
            device_core_num = AICORE_NUM
            n_new = device_core_num
            h_new = h_i*w_i // 256 // device_core_num
            w_new = 256
            shape_input_new = []
            shape_input_new.append(n_new)
            shape_input_new.append(h_new)
            shape_input_new.append(w_new)
            shape_input_new.append(c_i)
            shape_output_new = []
            shape_output_new.append(n_new)
            shape_output_new.append(1)
            shape_output_new.append(h_new)
            shape_output_new.append(w_new)
            shape_output_new.append(c_0)
        else:
            shape_input_new = shape_input
            shape_output_new = [n_i, 1, h_i, w_i, c_0]

        data = tvm.placeholder(shape_input_new, dtype=dtype_input, name="data")
        res = tvm.extern(shape_output_new, [data],
                         lambda ins, outs: _move_vconv_one_diff(outs[0],
                                                                ins[0]),
                         name="res", dtype=dtype_input)
        tensor_list = [data, res]
        sch = tvm.create_schedule(res.op)

        with build_config:
            tvm.build(sch, tensor_list, "cce", name=kernel_name)

    else:
        data_input = tvm.placeholder(shape_input, name="data_input",
                                     dtype=dtype_input)
        data_output = tvm.placeholder(shape_output, name="data_output",
                                      dtype=dtype_output)
        res = four_2_five_compute(data_input, data_output,
                                  src_format, dst_format,
                                  kernel_name)

        with tvm.target.cce():
            sch = generic.auto_schedule(res)

        config = {"name": kernel_name,
                  "tensor_list": [data_input, res]}
        te.lang.cce.cce_build_code(sch, config)
