#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR list_a PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

resize_bilinear
"""
from functools import reduce as reduce_func
import te
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_build import build_config
from te.platform.cce_conf import CceProductParams
from te import platform as tbe_platform
from te.platform import cce_params as param
from topi.cce import util
# pylint: disable=ungrouped-imports
from te.lang.cce.te_compute import irbuilder_api as kernel_api

# H, W size threshold 256
HW_SIZE_256 = 256
# H, W size threshold 512
HW_SIZE_512 = 512
# H, W size threshold 1024
HW_SIZE_1024 = 1024
# H, W size threshold 2048
HW_SIZE_2048 = 2048
# the block count vbi processed one time
VBI_BLOCK_COUNT = 8


# pylint: disable=invalid-name,too-many-locals,too-many-lines


def apply_store_buffer(ib,
                       dtype,
                       shape,
                       name="store_buf",
                       scope=param.scope_ubuf):
    """
        apply storage space
    """
    buf_var = ib.allocate(dtype, shape, name=name, scope=scope)
    return tvm.decl_buffer(shape,
                           buf_var.dtype,
                           name=name,
                           scope=scope,
                           data=buf_var)


def apply_reg_buffer(ib, dtype, shape, name="reg_buf"):
    """
        apply register space
    """
    return ib.allocate(dtype, shape, name=name, scope=param.scope_reg)


# pylint: disable=unused-argument,unused-variable,too-many-arguments
def check_supported(images,
                    y,
                    size,
                    align_corners=False,
                    half_pixel_centers=False,
                    kernel_name="resize_bilinear_v2"):
    """
    To check whether the AICORE operator can support the length of w/h or not
    """
    image_shape = images.get("shape")
    image_format = images.get("format")

    if image_format == "NHWC":
        h_in = image_shape[1]
        w_in = image_shape[2]
    elif image_format in ("NCHW", "NC1HWC0"):
        h_in = image_shape[2]
        w_in = image_shape[3]
    else:
        raise RuntimeError("The input format is not supported,"
                           " which is, ", image_format)

    try:

        if w_in > HW_SIZE_2048 or h_in > HW_SIZE_2048 or \
                size[0] > HW_SIZE_2048 or size[1] > HW_SIZE_2048:
            return False

    except RuntimeError as e:
        return False

    return True


# pylint: disable=too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions
def _resize_bilinear_ir(inputs, outputs, align_corners, half_pixel_centers):
    """
        ir build part
        -------------
        Process:
            1. input H/W = output H/W
            2. input H/W = (1,1)
            3. output H/W = (1,1)
            4. normal input/output H/W
    """
    # get device version
    core_counts = tbe_platform.cce_conf.get_soc_spec("CORE_NUM")
    if tbe_platform.cce_conf.intrinsic_check_support(
            "Intrinsic_vconv", "f322s32f"):
        devices = "cloud"
    else:
        devices = "mini"

    ib = tvm.ir_builder.create()
    # use multi-core resource
    block = tvm.thread_axis("blockIdx.x")
    # get dma sid
    sid = int(CceProductParams().getParams("Sid_copy_gm_to_cbuf"))
    pad_mode = tvm.call_pure_intrin("int32", "tvm_cce_string_print",
                                    "PAD_NONE")

    # get input dtype
    dtype = inputs.dtype.lower()

    # get input\output shape
    size_in = []
    for i in inputs.shape:
        size_in.append(i.value)
    size_out = []
    for i in outputs.shape:
        size_out.append(i.value)
    h_in = size_in[-3]
    w_in = size_in[-2]
    h_out = size_out[-3]
    w_out = size_out[-2]
    x_up = w_in
    x_down = w_out
    y_up = h_in
    y_down = h_out

    # calculate scale
    if align_corners and x_down > 1:
        x_up -= 1
        x_down -= 1
    if align_corners and y_down > 1:
        y_up -= 1
        y_down -= 1
    scale_w = float(x_up) / float(x_down)
    scale_h = float(y_up) / float(y_down)
    # c0 declare
    f32_c0 = 8
    f16_c0 = 16
    c0 = 8 if dtype == "float32" else 16

    # No.1 situation : input H/W = output H/W
    if h_in == h_out and w_in == w_out:
        # Note:
        #   1. all data move in and out directly whatever the dtype is
        #   2. brust length is certainty in limit range
        ib.scope_attr(block, "thread_extent", core_counts)
        # max ubuf block count
        ub_block = 62 * HW_SIZE_1024 // 32
        data_block = reduce_func(lambda x, y: x * y, size_in) // c0
        move_loop = data_block // ub_block
        core_loop = move_loop // core_counts
        core_loop_tail = move_loop % core_counts
        tail_counts = data_block % ub_block
        trans_station = apply_store_buffer(ib,
                                           dtype, [ub_block * c0],
                                           name="trans_station")
        if dtype.lower() == "float16":
            trans_station_fp32 = apply_store_buffer(ib,
                                                    "float32", [ub_block * c0],
                                                    name="trans_station_fp32")

        def _inner_run(core_lp_idx):
            with ib.new_scope():
                ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                ib.emit(
                    tvm.call_extern(
                        dtype, "copy_gm_to_ubuf",
                        trans_station.access_ptr('w'),
                        inputs.access_ptr(
                            'r',
                            offset=block.var * ub_block * c0 +
                            core_lp_idx * core_counts * ub_block * c0),
                        0, 1,
                        ub_block, 0, 0))

            if dtype.lower() == "float32":
                with ib.new_scope():
                    ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                    ib.emit(
                        tvm.call_extern(
                            dtype, "copy_ubuf_to_gm",
                            outputs.access_ptr(
                                'w',
                                offset=block.var * ub_block * c0 +
                                core_lp_idx * core_counts * ub_block * c0),
                            trans_station.access_ptr('r'), 0, 1, ub_block, 0,
                            0))
            else:
                with ib.new_scope():
                    ip_addr = [[trans_station_fp32, 0], [trans_station, 0]]
                    kernel_api.kernel_cast_to_fuc(ib, ip_addr,
                                                  [ub_block * 16, 8 * 8],
                                                  "vconv_f162f32")
                    ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                    ib.emit(
                        tvm.call_extern(
                            "float32", "copy_ubuf_to_gm",
                            outputs.access_ptr(
                                'w',
                                offset=block.var * ub_block * c0 +
                                core_lp_idx * core_counts * ub_block * c0),
                            trans_station_fp32.access_ptr('r'), 0, 1,
                            ub_block * 2, 0, 0))

        with ib.for_range(0, core_loop, name="core_loop") as core_lp_idx:
            _inner_run(core_lp_idx)
        if core_loop_tail > 0:
            with ib.if_scope(block.var < core_loop_tail):
                _inner_run(core_loop)

        if tail_counts > 0:
            with ib.if_scope(block.var < 1):
                tail_start = move_loop * ub_block
                with ib.new_scope():
                    ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                    ib.emit(
                        tvm.call_extern(
                            dtype, "copy_gm_to_ubuf",
                            trans_station.access_ptr('w'),
                            inputs.access_ptr('r', offset=tail_start * c0), 0,
                            1, tail_counts, 0, 0))

                if dtype.lower() == "float32":
                    with ib.new_scope():
                        ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                        ib.emit(
                            tvm.call_extern(
                                dtype, "copy_ubuf_to_gm",
                                outputs.access_ptr('w',
                                                   offset=tail_start * c0),
                                trans_station.access_ptr('r'), 0, 1,
                                tail_counts, 0, 0))
                else:
                    with ib.new_scope():
                        ip_addr = [[trans_station_fp32, 0], [trans_station, 0]]
                        kernel_api.kernel_cast_to_fuc(
                            ib, ip_addr, [tail_counts * 16, 8 * 8],
                            "vconv_f162f32")
                        ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                        ib.emit(
                            tvm.call_extern(
                                "float32", "copy_ubuf_to_gm",
                                outputs.access_ptr('w',
                                                   offset=tail_start * c0),
                                trans_station_fp32.access_ptr('r'), 0, 1,
                                tail_counts * 2, 0, 0))

    # No.2 situation : input H/W = (1,1)
    elif h_in == 1 and w_in == 1:
        # Note:
        #   1. input f16 dtype convert to f32 then expand
        #   2. input f32 dtype expand directly

        # No.2 situation : 1. input f16 dtype
        if dtype == "float16":
            expand_loop = reduce_func(lambda x, y: x * y, size_in) // f16_c0
            actual_loop = min(expand_loop, HW_SIZE_512)
            free_space = HW_SIZE_512 * 12
            free_space_fp32 = HW_SIZE_512 * f16_c0
            expand_size = h_out * w_out
            ib.scope_attr(block, "thread_extent", core_counts)

            f16_in = apply_store_buffer(ib,
                                        "float16", [actual_loop * f16_c0],
                                        name="f16_in")
            f16_out = apply_store_buffer(ib,
                                         "float16", [free_space * f16_c0],
                                         name="f16_out")
            f32_out = apply_store_buffer(ib,
                                         "float32", [free_space_fp32],
                                         name="f32_out")

            if expand_size > HW_SIZE_512:
                f16_8 = apply_store_buffer(ib,
                                           "float16", [8 * f16_c0],
                                           name="f16_8")
            elif expand_size > 32:
                f16_64 = apply_store_buffer(ib,
                                            "float16", [64 * f16_c0],
                                            name="f16_64")
            # Note:
            #   1. input data larger than HW_SIZE_512 should use L1 optimize
            #   2. input small data do not need L1

            if expand_size > 32:
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const((1 << 64) - 1, dtype="uint64"),
                                    tvm.const((1 << 64) - 1, dtype="uint64")))
            # No.2 situation : 1. input f16 dtype : large data
            if expand_loop > HW_SIZE_512:
                l1_half = HW_SIZE_512 * 32
                L1_in = apply_store_buffer(ib,
                                           "float16", [l1_half * f16_c0],
                                           name="L1_in",
                                           scope=param.scope_cbuf)
                L1_out = apply_store_buffer(ib,
                                            "float16", [l1_half * f16_c0],
                                            name="L1_out",
                                            scope=param.scope_cbuf)

                loop_level1 = expand_loop // l1_half
                tail_level1 = expand_loop % l1_half

                # No.2 situation : 1. input f16 dtype : large data : loop_level1 > 0
                if loop_level1 > 0:
                    # level_1 loop
                    with ib.for_range(0, loop_level1) as i1:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                            ib.emit(
                                tvm.call_extern(
                                    "float16", "copy_gm_to_cbuf",
                                    L1_in.access_ptr('w'),
                                    inputs.access_ptr(
                                        'r',
                                        offset=i1 * l1_half * f16_c0),
                                    sid, 1,
                                    l1_half,
                                    0, 0, pad_mode))
                        # level_2 loop
                        loop_level2 = 32
                        with ib.for_range(0, loop_level2) as i2:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              4)
                                ib.emit(
                                    tvm.call_extern(
                                        "float16", "copy_cbuf_to_ubuf",
                                        f16_in.access_ptr('w'),
                                        L1_in.access_ptr(
                                            'r',
                                            offset=i2 * HW_SIZE_512 * f16_c0),
                                        0, 1,
                                        HW_SIZE_512, 0,
                                        0))
                            # Note:
                            #   1. H * W > HW_SIZE_512 is common
                            #   2. HW_SIZE_512 >= H * W > 32
                            #   3. H * W <= 32 this happen really rare

                            # No.2 situation : 1. input f16 dtype : large data : 1. H * W > 512
                            if expand_size > HW_SIZE_512:
                                # level_3 loop
                                loop_level3 = 8
                                with ib.for_range(0, loop_level3) as i3:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_ubuf_to_cbuf",
                                                L1_out.access_ptr(
                                                    'w',
                                                    offset=i3 * f16_c0),
                                                f16_in.access_ptr('r'), 0, HW_SIZE_512,
                                                1, 0, 8 - 1))
                                # level_3 loop
                                loop_level3 = HW_SIZE_512
                                core_lp3 = loop_level3 // core_counts
                                core_lp3_tail = loop_level3 % core_counts

                                def _inner_run_level3(core_lp3_idx):
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_cbuf_to_ubuf",
                                                f16_8.access_ptr('w'),
                                                L1_out.access_ptr(
                                                    'r',
                                                    offset=block.var * 8 *
                                                    f16_c0 + core_lp3_idx *
                                                    core_counts * 8 * f16_c0),
                                                0, 1, 8, 0, 0))
                                    repeat_64 = HW_SIZE_512 // 8
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 2)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "vadds",
                                                f16_out.access_ptr('w'),
                                                f16_8.access_ptr('r'),
                                                tvm.const(0.0,
                                                          dtype="float16"),
                                                repeat_64, 1, 1, 8, 0))
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 2)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_ubuf_to_ubuf",
                                                f16_out.access_ptr(
                                                    'w',
                                                    offset=HW_SIZE_512 * f16_c0),
                                                f16_out.access_ptr('r'), 0, 1,
                                                HW_SIZE_512, 0, 0))
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 2)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_ubuf_to_ubuf",
                                                f16_out.access_ptr(
                                                    'w', offset=HW_SIZE_1024 * f16_c0),
                                                f16_out.access_ptr('r'), 0, 1,
                                                HW_SIZE_1024, 0, 0))
                                    loop_level4 = expand_size // HW_SIZE_2048
                                    tail_level4 = expand_size % HW_SIZE_2048
                                    offset_4 = expand_size * f16_c0
                                    if loop_level4 > 0:
                                        with ib.for_range(0,
                                                          loop_level4) as i5:
                                            with ib.new_scope():
                                                _inner_loop = \
                                                    HW_SIZE_2048 * f16_c0 // free_space_fp32
                                                _inner_tail = \
                                                    HW_SIZE_2048 * f16_c0 % free_space_fp32
                                                with ib.for_range(
                                                        0, _inner_loop
                                                ) as inner_idx:
                                                    ip_addr = [
                                                        [f32_out, 0],
                                                        [
                                                            f16_out,
                                                            inner_idx *
                                                            free_space_fp32
                                                        ]
                                                    ]
                                                    kernel_api.kernel_cast_to_fuc(
                                                        ib, ip_addr, [
                                                            free_space_fp32,
                                                            8 * 8
                                                        ], "vconv_f162f32")
                                                    ib.scope_attr(
                                                        param.CCE_AXIS,
                                                        "coproc_scope", 6)
                                                    ib.emit(tvm.call_extern(
                                                        "float32",
                                                        "copy_ubuf_to_gm",
                                                        outputs.access_ptr(
                                                            'w',
                                                            offset=((i1 * 32 + i2) * HW_SIZE_512 +
                                                                    block.var) *
                                                            offset_4 + i5 * HW_SIZE_2048 * f16_c0 +
                                                            inner_idx * free_space_fp32 +
                                                            core_lp3_idx * core_counts * offset_4),
                                                        f32_out.access_ptr('r'),
                                                        0, 1,
                                                        free_space_fp32 // 8, 0,
                                                        0))

                                    if tail_level4 > 0:
                                        with ib.new_scope():
                                            _inner_loop = tail_level4 * f16_c0 // free_space_fp32
                                            _inner_tail = tail_level4 * f16_c0 % free_space_fp32
                                            with ib.for_range(
                                                    0,
                                                    _inner_loop) as inner_idx:
                                                ip_addr = [[f32_out, 0],
                                                           [
                                                               f16_out,
                                                               inner_idx *
                                                               free_space_fp32
                                                           ]]
                                                kernel_api.kernel_cast_to_fuc(
                                                    ib, ip_addr,
                                                    [free_space_fp32, 8 * 8],
                                                    "vconv_f162f32")
                                                ib.scope_attr(
                                                    param.CCE_AXIS,
                                                    "coproc_scope", 6)
                                                ib.emit(tvm.call_extern(
                                                    "float32",
                                                    "copy_ubuf_to_gm",
                                                    outputs.access_ptr(
                                                        'w',
                                                        offset=((i1 * 32 +\
                                                                 i2)\
                                                         * HW_SIZE_512 \
                                                         + block.var) *
                                                        offset_4 \
                                                        + loop_level4 *
                                                        HW_SIZE_2048 \
                                                        * f16_c0 +
                                                        inner_idx *
                                                        free_space_fp32 +
                                                        core_lp3_idx *
                                                        core_counts *
                                                        offset_4),
                                                    f32_out.access_ptr('r'),
                                                    0, 1, free_space_fp32 // 8,
                                                    0,
                                                    0))
                                            if _inner_tail > 1:
                                                ip_addr = [[f32_out, 0],
                                                           [
                                                               f16_out,
                                                               _inner_loop *
                                                               free_space_fp32
                                                           ]]
                                                kernel_api.kernel_cast_to_fuc(
                                                    ib, ip_addr,
                                                    [_inner_tail, 8 * 8],
                                                    "vconv_f162f32")
                                                ib.scope_attr(
                                                    param.CCE_AXIS,
                                                    "coproc_scope", 6)
                                                ib.emit(tvm.call_extern(
                                                    "float32",
                                                    "copy_ubuf_to_gm",
                                                    outputs.access_ptr(
                                                        'w',
                                                        offset=((i1 * 32 + i2) * HW_SIZE_512 +
                                                                block.var) *
                                                        offset_4 + loop_level4 * HW_SIZE_2048 *
                                                        f16_c0 + _inner_loop * free_space_fp32 +
                                                        core_lp3_idx * core_counts * offset_4),
                                                    f32_out.access_ptr('r'),
                                                    0, 1,
                                                    (_inner_tail + 7) // 8, 0,
                                                    0))

                                with ib.for_range(
                                        0, core_lp3,
                                        name="core_lp3") as core_lp3_idx:
                                    _inner_run_level3(core_lp3_idx)
                                if core_lp3_tail > 0:
                                    with ib.if_scope(
                                            block.var < core_lp3_tail):
                                        _inner_run_level3(core_lp3)

                            elif expand_size > 32:
                                # level_3 loop
                                loop_level3 = 8
                                with ib.for_range(0, loop_level3) as i3:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_ubuf_to_cbuf",
                                                L1_out.access_ptr(
                                                    'w',
                                                    offset=i3 * f16_c0),
                                                f16_in.access_ptr('r'), 0, HW_SIZE_512,
                                                1, 0, 8 - 1))
                                # level_3 loop
                                loop_level3 = HW_SIZE_512 // 8
                                with ib.for_range(0, loop_level3) as i4:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_cbuf_to_ubuf",
                                                f16_64.access_ptr('w'),
                                                L1_out.access_ptr(
                                                    'r',
                                                    offset=i4 * 64 * f16_c0),
                                                0, 1, 64, 0, 0))
                                    # level_4 loop
                                    loop_level4 = 8
                                    offset_l4 = expand_size * f16_c0
                                    repeat_l4 = expand_size // 8 + \
                                                (
                                                    1 if expand_size % 8 > 0 else 0)
                                    with ib.for_range(0, loop_level4) as i5:
                                        with ib.new_scope():
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                2)
                                            ib.emit(
                                                tvm.call_extern(
                                                    "float16", "vadds",
                                                    f16_out.access_ptr(
                                                        'w',
                                                        offset=i5 * offset_l4),
                                                    f16_64.access_ptr(
                                                        'r',
                                                        offset=i5 * 8 * f16_c0),
                                                    tvm.const(0.0,
                                                              dtype="float16"),
                                                    repeat_l4, 1, 1, 8, 0))
                                    offset_4 = expand_size * f16_c0
                                    expand_4 = 8 * expand_size
                                    with ib.new_scope():
                                        _inner_loop = expand_4 * f16_c0 // free_space_fp32
                                        _inner_tail = expand_4 * f16_c0 % free_space_fp32
                                        with ib.for_range(
                                                0, _inner_loop) as inner_idx:
                                            ip_addr = [[f32_out, 0],
                                                       [f16_out,
                                                        inner_idx * free_space_fp32
                                                       ]]
                                            kernel_api.kernel_cast_to_fuc(
                                                ib, ip_addr,
                                                [free_space_fp32, 8 * 8],
                                                "vconv_f162f32")
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(
                                                tvm.call_extern(
                                                    "float32",
                                                    "copy_ubuf_to_gm",
                                                    outputs.access_ptr(
                                                        'w',
                                                        offset=((i1 * 32 + i2) * HW_SIZE_512 \
                                                        + i4 * 8) * offset_4 +
                                                        inner_idx * free_space_fp32),
                                                    f32_out.access_ptr('r'),
                                                    0, 1,
                                                    free_space_fp32 // 8,
                                                    0, 0))
                                        if _inner_tail > 0:
                                            ip_addr = [[f32_out, 0],
                                                       [
                                                           f16_out,
                                                           _inner_loop *
                                                           free_space_fp32
                                                       ]]
                                            kernel_api.kernel_cast_to_fuc(
                                                ib, ip_addr,
                                                [_inner_tail, 8 * 8],
                                                "vconv_f162f32")
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(
                                                tvm.call_extern(
                                                    "float32",
                                                    "copy_ubuf_to_gm",
                                                    outputs.access_ptr(
                                                        'w',
                                                        offset=((i1 * 32 + i2) * HW_SIZE_512 \
                                                        + i4 * 8) * offset_4 +
                                                        _inner_loop * free_space_fp32),
                                                    f32_out.access_ptr('r'),
                                                    0, 1,
                                                    (_inner_tail + 7) // 8,
                                                    0, 0))

                            else:
                                with ib.for_range(0, expand_size) as i3:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_ubuf_to_cbuf",
                                                L1_out.access_ptr(
                                                    'w',
                                                    offset=i3 * f16_c0),
                                                f16_in.access_ptr('r'), 0, HW_SIZE_512,
                                                1, 0, expand_size - 1))
                                loop_level3 = expand_size // 12
                                tail_level3 = expand_size % 12
                                if loop_level3 > 0:
                                    with ib.for_range(0, loop_level3) as i4:
                                        with ib.new_scope():
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                4)
                                            ib.emit(
                                                tvm.call_extern(
                                                    "float16",
                                                    "copy_cbuf_to_ubuf",
                                                    f16_out.access_ptr('w'),
                                                    L1_out.access_ptr(
                                                        'r',
                                                        offset=i4 * HW_SIZE_512 * 12 *
                                                        f16_c0),
                                                    0, 1,
                                                    HW_SIZE_512 * 12, 0, 0))
                                        with ib.new_scope():
                                            _inner_loop = \
                                                HW_SIZE_512 * 12 * f16_c0 // free_space_fp32
                                            _inner_tail = \
                                                HW_SIZE_512 * 12 * f16_c0 % free_space_fp32
                                            with ib.for_range(
                                                    0,
                                                    _inner_loop) as inner_idx:
                                                ip_addr = [[f32_out, 0],
                                                           [
                                                               f16_out,
                                                               inner_idx *
                                                               free_space_fp32
                                                           ]]
                                                kernel_api.kernel_cast_to_fuc(
                                                    ib, ip_addr,
                                                    [free_space_fp32, 8 * 8],
                                                    "vconv_f162f32")
                                                ib.scope_attr(
                                                    param.CCE_AXIS,
                                                    "coproc_scope", 6)
                                                ib.emit(tvm.call_extern(
                                                    "float32",
                                                    "copy_ubuf_to_gm",
                                                    outputs.access_ptr(
                                                        'w',
                                                        offset=(((i1 * 32 + i2) * \
                                                          expand_size) + i4 * 12) \
                                                        * HW_SIZE_512 * f16_c0 +
                                                        inner_idx * free_space_fp32),
                                                    f32_out.access_ptr('r'),
                                                    0, 1, free_space_fp32 // 8,
                                                    0, 0))
                                            if _inner_tail > 0:
                                                ip_addr = [[f32_out, 0],
                                                           [
                                                               f16_out,
                                                               _inner_loop *
                                                               free_space_fp32
                                                           ]]
                                                kernel_api.kernel_cast_to_fuc(
                                                    ib, ip_addr,
                                                    [_inner_tail, 8 * 8],
                                                    "vconv_f162f32")
                                                ib.scope_attr(
                                                    param.CCE_AXIS,
                                                    "coproc_scope", 6)
                                                ib.emit(tvm.call_extern(
                                                    "float32",
                                                    "copy_ubuf_to_gm",
                                                    outputs.access_ptr(
                                                        'w',
                                                        offset=(((i1 * 32 + i2) * \
                                                          expand_size) + i4 * 12) \
                                                        * HW_SIZE_512 * f16_c0 +
                                                        _inner_loop * free_space_fp32),
                                                    f32_out.access_ptr('r'),
                                                    0, 1,
                                                    (_inner_tail + 7) // 8, 0,
                                                    0))

                                if tail_level3 > 0:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(tvm.call_extern(
                                            "float16",
                                            "copy_cbuf_to_ubuf",
                                            f16_out.access_ptr('w'),
                                            L1_out.access_ptr(
                                                'r',
                                                offset=loop_level3 * HW_SIZE_512 \
                                                       * 12 * f16_c0),
                                            0, 1, HW_SIZE_512 * tail_level3, 0, 0))
                                    with ib.new_scope():
                                        _inner_loop = \
                                            HW_SIZE_512 * tail_level3 * f16_c0 // free_space_fp32
                                        _inner_tail = \
                                            HW_SIZE_512 * tail_level3 * f16_c0 % free_space_fp32
                                        with ib.for_range(
                                                0, _inner_loop) as inner_idx:
                                            ip_addr = [[f32_out, 0],
                                                       [f16_out,
                                                        inner_idx * free_space_fp32
                                                       ]]
                                            kernel_api.kernel_cast_to_fuc(
                                                ib, ip_addr,
                                                [free_space_fp32, 8 * 8],
                                                "vconv_f162f32")
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=(((i1 * 32 + i2) * expand_size) \
                                                     + loop_level3 * 12) \
                                                    * HW_SIZE_512 * f16_c0 +
                                                    inner_idx * free_space_fp32),
                                                f32_out.access_ptr('r'),
                                                0, 1, free_space_fp32 // 8, 0,
                                                0))
                                        if _inner_tail > 0:
                                            ip_addr = [[f32_out, 0],
                                                       [
                                                           f16_out,
                                                           _inner_loop *
                                                           free_space_fp32
                                                       ]]
                                            kernel_api.kernel_cast_to_fuc(
                                                ib, ip_addr,
                                                [_inner_tail, 8 * 8],
                                                "vconv_f162f32")
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=(((i1 * 32 + i2) * expand_size) \
                                                     + loop_level3 * 12) \
                                                    * HW_SIZE_512 * f16_c0 +
                                                    _inner_loop * free_space_fp32),
                                                f32_out.access_ptr('r'),
                                                0, 1, (_inner_tail + 7) // 8, 0,
                                                0))

                # No.2 situation : 1. input f16 dtype : large data : tail_level1 > 0
                if tail_level1 > 0:
                    # Note:
                    #   1. tail_level1 larger then HW_SIZE_512 need move to L1 first
                    #   2. tail_level1 less then HW_SIZE_512 move to ubuf directly

                    if tail_level1 > HW_SIZE_512:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                            ib.emit(
                                tvm.call_extern(
                                    "float16", "copy_gm_to_cbuf",
                                    L1_in.access_ptr('w'),
                                    inputs.access_ptr(
                                        'r',
                                        offset=loop_level1 *  l1_half * f16_c0),
                                    sid,
                                    1, tail_level1, 0, 0, pad_mode))
                    else:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                            ib.emit(
                                tvm.call_extern(
                                    "float16", "copy_gm_to_ubuf",
                                    f16_in.access_ptr('w'),
                                    inputs.access_ptr(
                                        'r',
                                        offset=loop_level1 * l1_half * f16_c0),
                                    0, 1,
                                    tail_level1, 0, 0))
                    # level_1 loop
                    loop_l1 = tail_level1 // HW_SIZE_512
                    tail_l1 = tail_level1 % HW_SIZE_512
                    # No.2 situation : 1. input f16 dtype : large data : tail_level1>0 : loop_l1>0
                    if loop_l1 > 0:
                        with ib.for_range(0, loop_l1) as i1:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              4)
                                ib.emit(
                                    tvm.call_extern(
                                        "float16", "copy_cbuf_to_ubuf",
                                        f16_in.access_ptr('w'),
                                        L1_in.access_ptr(
                                            'r',
                                            offset=i1 * HW_SIZE_512 * f16_c0), 0, 1,
                                        HW_SIZE_512, 0,
                                        0))
                            # Note:
                            #   1. H * W >= HW_SIZE_512 is common
                            #   2. HW_SIZE_512 > H * W > 32
                            #   3. H * W <= 32 this happen really rare

                            # No.2 situation : 1. input f16 dtype : large data : 1. H * W >= 512
                            if expand_size > HW_SIZE_512:
                                # level_2 loop
                                loop_l2 = 8
                                with ib.for_range(0, loop_l2) as i2:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_ubuf_to_cbuf",
                                                L1_out.access_ptr(
                                                    'w',
                                                    offset=i2 * f16_c0),
                                                f16_in.access_ptr('r'), 0, HW_SIZE_512,
                                                1, 0, 8 - 1))
                                # level_2 loop
                                loop_l2 = HW_SIZE_512
                                core_lp2 = loop_l2 // core_counts
                                core_lp2_tail = loop_l2 % core_counts

                                def _inner_run_lp2(core_lp2_idx):
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_cbuf_to_ubuf",
                                                f16_8.access_ptr('w'),
                                                L1_out.access_ptr(
                                                    'r',
                                                    offset=block.var * 8 *
                                                    f16_c0 + core_lp2_idx *
                                                    core_counts * 8 * f16_c0),
                                                0, 1, 8, 0, 0))
                                    repeat_64 = HW_SIZE_512 // 8
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 2)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "vadds",
                                                f16_out.access_ptr('w'),
                                                f16_8.access_ptr('r'),
                                                tvm.const(0.0,
                                                          dtype="float16"),
                                                repeat_64, 1, 1, 8, 0))
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 2)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_ubuf_to_ubuf",
                                                f16_out.access_ptr(
                                                    'w',
                                                    offset=HW_SIZE_512 * f16_c0),
                                                f16_out.access_ptr('r'), 0, 1,
                                                HW_SIZE_512, 0, 0))
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 2)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_ubuf_to_ubuf",
                                                f16_out.access_ptr(
                                                    'w', offset=HW_SIZE_1024 * f16_c0),
                                                f16_out.access_ptr('r'), 0, 1,
                                                HW_SIZE_1024, 0, 0))
                                    loop_l3 = expand_size // HW_SIZE_2048
                                    tail_l3 = expand_size % HW_SIZE_2048
                                    offset_3 = expand_size * f16_c0
                                    if loop_l3 > 0:
                                        with ib.for_range(0, loop_l3) as i4:
                                            with ib.new_scope():
                                                _inner_loop = \
                                                    HW_SIZE_2048 * f16_c0 // free_space_fp32
                                                with ib.for_range(
                                                        0, _inner_loop
                                                ) as inner_idx:
                                                    ip_addr = [
                                                        [f32_out, 0],
                                                        [
                                                            f16_out,
                                                            inner_idx *
                                                            free_space_fp32
                                                        ]
                                                    ]
                                                    kernel_api.kernel_cast_to_fuc(
                                                        ib, ip_addr, [
                                                            free_space_fp32,
                                                            8 * 8
                                                        ], "vconv_f162f32")
                                                    ib.scope_attr(
                                                        param.CCE_AXIS,
                                                        "coproc_scope", 6)
                                                    ib.emit(tvm.call_extern(
                                                        "float32",
                                                        "copy_ubuf_to_gm",
                                                        outputs.access_ptr(
                                                            'w',
                                                            offset=((loop_level1 \
                                                              * 32 + i1) * HW_SIZE_512 \
                                                             + block.var) * offset_3 \
                                                            + i4 * HW_SIZE_2048 \
                                                            * f16_c0 +
                                                            inner_idx * free_space_fp32 +
                                                            core_lp2_idx *
                                                            core_counts * offset_3),
                                                        f32_out.access_ptr('r'),
                                                        0, 1,
                                                        free_space_fp32 // 8,
                                                        0, 0))
                                    if tail_l3 > 0:
                                        with ib.new_scope():
                                            _inner_loop = \
                                                tail_l3 * f16_c0 // free_space_fp32
                                            _inner_tail = \
                                                tail_l3 * f16_c0 % free_space_fp32
                                            with ib.for_range(
                                                    0,
                                                    _inner_loop) as inner_idx:
                                                ip_addr = [[f32_out, 0],
                                                           [
                                                               f16_out,
                                                               inner_idx *
                                                               free_space_fp32
                                                           ]]
                                                kernel_api.kernel_cast_to_fuc(
                                                    ib, ip_addr,
                                                    [free_space_fp32, 8 * 8],
                                                    "vconv_f162f32")
                                                ib.scope_attr(
                                                    param.CCE_AXIS,
                                                    "coproc_scope", 6)
                                                ib.emit(tvm.call_extern(
                                                    "float32",
                                                    "copy_ubuf_to_gm",
                                                    outputs.access_ptr(
                                                        'w',
                                                        offset=((loop_level1 * \
                                                        32  + i1) * \
                                                        HW_SIZE_512 + block.var)
                                                        * offset_3 + loop_l3 *
                                                        HW_SIZE_2048 * f16_c0 +
                                                        inner_idx *
                                                        free_space_fp32 +
                                                        core_lp2_idx *
                                                        core_counts * offset_3),
                                                    f32_out.access_ptr('r'),
                                                    0, 1, free_space_fp32 // 8,
                                                    0, 0))
                                            if _inner_tail > 0:
                                                ip_addr = [[f32_out, 0],
                                                           [
                                                               f16_out,
                                                               _inner_loop *
                                                               free_space_fp32
                                                           ]]
                                                kernel_api.kernel_cast_to_fuc(
                                                    ib, ip_addr,
                                                    [_inner_tail, 8 * 8],
                                                    "vconv_f162f32")
                                                ib.scope_attr(
                                                    param.CCE_AXIS,
                                                    "coproc_scope", 6)
                                                ib.emit(tvm.call_extern(
                                                    "float32",
                                                    "copy_ubuf_to_gm",
                                                    outputs.access_ptr(
                                                        'w',
                                                        offset=((loop_level1 * 32 \
                                                          + i1) * HW_SIZE_512 + block.var) \
                                                        * offset_3 \
                                                        + loop_l3 * HW_SIZE_2048 \
                                                        * f16_c0 +
                                                        _inner_loop * free_space_fp32 +
                                                        core_lp2_idx *
                                                        core_counts * offset_3),
                                                    f32_out.access_ptr('r'),
                                                    0, 1,
                                                    (_inner_tail + 7) // 8, 0,
                                                    0))

                                with ib.for_range(
                                        0, core_lp2,
                                        name="core_lp2") as core_lp2_idx:
                                    _inner_run_lp2(core_lp2_idx)
                                if core_lp2_tail > 0:
                                    with ib.if_scope(
                                            block.var < core_lp2_tail):
                                        _inner_run_lp2(core_lp2)

                            elif expand_size > 32:
                                # No.2 situation : 1. input f16 dtype : large data : 2. 512>=H*W>32
                                # level_2 loop
                                loop_l2 = 8
                                with ib.for_range(0, loop_l2) as i2:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_ubuf_to_cbuf",
                                                L1_out.access_ptr(
                                                    'w',
                                                    offset=i2 * f16_c0),
                                                f16_in.access_ptr('r'), 0, HW_SIZE_512,
                                                1, 0, 8 - 1))
                                # level_2 loop
                                loop_l2 = HW_SIZE_512 // 8
                                with ib.for_range(0, loop_l2) as i3:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_cbuf_to_ubuf",
                                                f16_64.access_ptr('w'),
                                                L1_out.access_ptr(
                                                    'r',
                                                    offset=i3 * 64 * f16_c0),
                                                0, 1, 64, 0, 0))
                                    # level_3 loop
                                    loop_l3 = 8
                                    offset_l3 = expand_size * f16_c0
                                    repeat_l3 = expand_size // 8 + \
                                                (
                                                    1 if expand_size % 8 > 0 else 0)
                                    with ib.for_range(0, loop_l3) as i4:
                                        with ib.new_scope():
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                2)
                                            ib.emit(
                                                tvm.call_extern(
                                                    "float16", "vadds",
                                                    f16_out.access_ptr(
                                                        'w',
                                                        offset=i4 * offset_l3),
                                                    f16_64.access_ptr(
                                                        'r',
                                                        offset=i4 * 8 * f16_c0),
                                                    tvm.const(0.0,
                                                              dtype="float16"),
                                                    repeat_l3, 1, 1, 8, 0))
                                    offset_4 = expand_size * f16_c0
                                    expand_4 = 8 * expand_size
                                    with ib.new_scope():
                                        _inner_loop = \
                                            expand_4 * f16_c0 // free_space_fp32
                                        _inner_tail = \
                                            expand_4 * f16_c0 % free_space_fp32
                                        with ib.for_range(
                                                0, _inner_loop) as inner_idx:
                                            ip_addr = [[f32_out, 0],
                                                       [f16_out,
                                                        inner_idx * free_space_fp32
                                                       ]]
                                            kernel_api.kernel_cast_to_fuc(
                                                ib, ip_addr,
                                                [free_space_fp32, 8 * 8],
                                                "vconv_f162f32")
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=((loop_level1 * 32 + i1) \
                                                    * HW_SIZE_512 + i3 * 8) * offset_4 +
                                                    inner_idx * free_space_fp32),
                                                f32_out.access_ptr('r'),
                                                0, 1, free_space_fp32 // 8, 0,
                                                0))
                                        if _inner_tail > 0:
                                            ip_addr = [[f32_out, 0],
                                                       [
                                                           f16_out,
                                                           _inner_loop *
                                                           free_space_fp32
                                                       ]]
                                            kernel_api.kernel_cast_to_fuc(
                                                ib, ip_addr,
                                                [_inner_tail, 8 * 8],
                                                "vconv_f162f32")
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=((loop_level1 * 32 + i1) \
                                                    * HW_SIZE_512 + i3 * 8) * offset_4 +
                                                    _inner_loop * free_space_fp32),
                                                f32_out.access_ptr('r'),
                                                0, 1, (_inner_tail + 7) // 8, 0,
                                                0))

                            else:
                                # No.2 situation : 1. input f16 dtype : large data  3. H * W <= 32
                                with ib.for_range(0, expand_size) as i2:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_ubuf_to_cbuf",
                                                L1_out.access_ptr(
                                                    'w',
                                                    offset=i2 * f16_c0),
                                                f16_in.access_ptr('r'), 0, HW_SIZE_512,
                                                1, 0, expand_size - 1))
                                loop_l2 = expand_size // 12
                                tail_l2 = expand_size % 12
                                if loop_l2 > 0:
                                    with ib.for_range(0, loop_l2) as i3:
                                        with ib.new_scope():
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                4)
                                            ib.emit(
                                                tvm.call_extern(
                                                    "float16",
                                                    "copy_cbuf_to_ubuf",
                                                    f16_out.access_ptr('w'),
                                                    L1_out.access_ptr(
                                                        'r',
                                                        offset=i3 * HW_SIZE_512 * 12 *
                                                        f16_c0),
                                                    0, 1,
                                                    HW_SIZE_512 * 12, 0, 0))
                                        with ib.new_scope():
                                            _inner_loop = \
                                                HW_SIZE_512 * 12 * f16_c0 // free_space_fp32
                                            with ib.for_range(
                                                    0,
                                                    _inner_loop) as inner_idx:
                                                ip_addr = [[f32_out, 0],
                                                           [
                                                               f16_out,
                                                               inner_idx *
                                                               free_space_fp32
                                                           ]]
                                                kernel_api.kernel_cast_to_fuc(
                                                    ib, ip_addr,
                                                    [free_space_fp32, 8 * 8],
                                                    "vconv_f162f32")
                                                ib.scope_attr(
                                                    param.CCE_AXIS,
                                                    "coproc_scope", 6)
                                                ib.emit(tvm.call_extern(
                                                    "float32",
                                                    "copy_ubuf_to_gm",
                                                    outputs.access_ptr(
                                                        'w',
                                                        offset=(((loop_level1 * 32 \
                                                        + i1) * expand_size) \
                                                        + i3 * 12) \
                                                        * HW_SIZE_512 * f16_c0 +
                                                        inner_idx * free_space_fp32),
                                                    f32_out.access_ptr('r'),
                                                    0, 1, free_space_fp32 // 8,
                                                    0, 0))

                                if tail_l2 > 0:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(tvm.call_extern(
                                            "float16",
                                            "copy_cbuf_to_ubuf",
                                            f16_out.access_ptr('w'),
                                            L1_out.access_ptr(
                                                'r',
                                                offset=loop_l2 * HW_SIZE_512 \
                                                       * 12 * f16_c0),
                                            0, 1, HW_SIZE_512 * tail_l2, 0, 0))
                                    with ib.new_scope():
                                        _inner_loop = \
                                            HW_SIZE_512 * tail_l2 * f16_c0 // free_space_fp32
                                        _inner_tail = \
                                            HW_SIZE_512 * tail_l2 * f16_c0 % free_space_fp32
                                        with ib.for_range(
                                                0, _inner_loop) as inner_idx:
                                            ip_addr = [[f32_out, 0],
                                                       [f16_out,
                                                        inner_idx * free_space_fp32
                                                       ]]
                                            kernel_api.kernel_cast_to_fuc(
                                                ib, ip_addr,
                                                [free_space_fp32, 8 * 8],
                                                "vconv_f162f32")
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=(((loop_level1 * 32 \
                                                    + i1) * expand_size) \
                                                    + loop_l2 * 12) \
                                                    * HW_SIZE_512 * f16_c0 +
                                                    inner_idx * free_space_fp32),
                                                f32_out.access_ptr('r'),
                                                0, 1, free_space_fp32 // 8, 0,
                                                0))
                                        if _inner_tail > 0:
                                            ip_addr = [[f32_out, 0],
                                                       [
                                                           f16_out,
                                                           _inner_loop *
                                                           free_space_fp32
                                                       ]]
                                            kernel_api.kernel_cast_to_fuc(
                                                ib, ip_addr,
                                                [_inner_tail, 8 * 8],
                                                "vconv_f162f32")
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=(((loop_level1 * 32 \
                                                    + i1) * expand_size) \
                                                    + loop_l2 * 12) \
                                                    * HW_SIZE_512 * f16_c0 +
                                                    _inner_loop * free_space_fp32),
                                                f32_out.access_ptr('r'),
                                                0, 1, (_inner_tail + 7) // 8, 0,
                                                0))

                    # No.2 situation : 1. input f16 dtype : large data : tail_level1>0 : tail_l1>0
                    if tail_l1 > 0:
                        if tail_level1 > HW_SIZE_512:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              4)
                                ib.emit(
                                    tvm.call_extern(
                                        "float16", "copy_cbuf_to_ubuf",
                                        f16_in.access_ptr('w'),
                                        L1_in.access_ptr(
                                            'r',
                                            offset=loop_l1 * HW_SIZE_512 * f16_c0),
                                        0, 1,
                                        tail_l1, 0, 0))
                        # Note:
                        #   1. H * W >= HW_SIZE_512 is common
                        #   2. HW_SIZE_512 > H * W > 32
                        #   3. H * W <= 32 this happen really rare

                        # No.2 situation : 1. input f16 dtype : large data : 1. H * W >= HW_SIZE_512
                        if expand_size > HW_SIZE_512:
                            # level_1 loop
                            loop_le1 = 8
                            with ib.for_range(0, loop_le1) as i1:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 6)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float16", "copy_ubuf_to_cbuf",
                                            L1_out.access_ptr(
                                                'w',
                                                offset=i1 * f16_c0),
                                            f16_in.access_ptr('r'), 0, tail_l1,
                                            1, 0, 8 - 1))
                            # level_1 loop
                            loop_le1 = tail_l1
                            with ib.for_range(0, loop_le1) as i2:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 4)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float16", "copy_cbuf_to_ubuf",
                                            f16_8.access_ptr('w'),
                                            L1_out.access_ptr(
                                                'r',
                                                offset=i2 * 8 * f16_c0),
                                            0, 1, 8,
                                            0, 0))
                                repeat_64 = HW_SIZE_512 // 8
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 2)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float16", "vadds",
                                            f16_out.access_ptr('w'),
                                            f16_8.access_ptr('r'),
                                            tvm.const(0.0, dtype="float16"),
                                            repeat_64, 1, 1, 8, 0))
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 2)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float16", "copy_ubuf_to_ubuf",
                                            f16_out.access_ptr(
                                                'w',
                                                offset=HW_SIZE_512 * f16_c0),
                                            f16_out.access_ptr('r'), 0, 1, HW_SIZE_512,
                                            0, 0))
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 2)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float16", "copy_ubuf_to_ubuf",
                                            f16_out.access_ptr(
                                                'w',
                                                offset=HW_SIZE_1024 * f16_c0),
                                            f16_out.access_ptr('r'), 0, 1,
                                            HW_SIZE_1024, 0, 0))
                                loop_le2 = expand_size // HW_SIZE_2048
                                tail_le2 = expand_size % HW_SIZE_2048
                                offset_2 = expand_size * f16_c0
                                if loop_le2 > 0:
                                    with ib.for_range(0, loop_le2) as i3:
                                        with ib.new_scope():
                                            _inner_loop = \
                                                HW_SIZE_2048 * f16_c0 // free_space_fp32
                                            with ib.for_range(
                                                    0,
                                                    _inner_loop) as inner_idx:
                                                ip_addr = [[f32_out, 0],
                                                           [
                                                               f16_out,
                                                               inner_idx *
                                                               free_space_fp32
                                                           ]]
                                                kernel_api.kernel_cast_to_fuc(
                                                    ib, ip_addr,
                                                    [free_space_fp32, 8 * 8],
                                                    "vconv_f162f32")
                                                ib.scope_attr(
                                                    param.CCE_AXIS,
                                                    "coproc_scope", 6)
                                                ib.emit(tvm.call_extern(
                                                    "float32",
                                                    "copy_ubuf_to_gm",
                                                    outputs.access_ptr(
                                                        'w',
                                                        offset=((loop_level1 * 32 \
                                                        + loop_l1) * HW_SIZE_512 \
                                                        + i2) * offset_2 \
                                                        + i3 * HW_SIZE_2048 * f16_c0 +
                                                        inner_idx * free_space_fp32),
                                                    f32_out.access_ptr('r'),
                                                    0, 1, free_space_fp32 // 8,
                                                    0, 0))
                                if tail_le2 > 0:
                                    with ib.new_scope():
                                        _inner_loop = \
                                            tail_le2 * f16_c0 // free_space_fp32
                                        _inner_tail = \
                                            tail_le2 * f16_c0 % free_space_fp32
                                        with ib.for_range(
                                                0, _inner_loop) as inner_idx:
                                            ip_addr = [[f32_out, 0],
                                                       [f16_out,
                                                        inner_idx * free_space_fp32
                                                       ]]
                                            kernel_api.kernel_cast_to_fuc(
                                                ib, ip_addr,
                                                [free_space_fp32, 8 * 8],
                                                "vconv_f162f32")
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=((loop_level1 * 32 \
                                                    + loop_l1) * HW_SIZE_512 \
                                                    + i2) * offset_2 \
                                                    + loop_le2 * HW_SIZE_2048 \
                                                    * f16_c0 +
                                                    inner_idx * free_space_fp32),
                                                f32_out.access_ptr('r'),
                                                0, 1, free_space_fp32 // 8, 0,
                                                0))
                                        if _inner_tail > 0:
                                            ip_addr = [[f32_out, 0],
                                                       [
                                                           f16_out,
                                                           _inner_loop *
                                                           free_space_fp32
                                                       ]]
                                            kernel_api.kernel_cast_to_fuc(
                                                ib, ip_addr,
                                                [_inner_tail, 8 * 8],
                                                "vconv_f162f32")
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=((loop_level1 * 32 \
                                                    + loop_l1) * HW_SIZE_512 \
                                                    + i2) * offset_2 \
                                                    + loop_le2 * HW_SIZE_2048 \
                                                    * f16_c0 +
                                                    _inner_loop * free_space_fp32),
                                                f32_out.access_ptr('r'),
                                                0, 1, (_inner_tail + 7) // 8, 0,
                                                0))

                        elif expand_size > 32:
                            # No.2 situation : 1. input f16 dtype : large data : 2. 512>=H*W>32
                            # level_1 loop
                            loop_le1 = 8
                            with ib.for_range(0, loop_le1) as i1:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 6)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float16", "copy_ubuf_to_cbuf",
                                            L1_out.access_ptr(
                                                'w',
                                                offset=i1 * f16_c0),
                                            f16_in.access_ptr('r'), 0, tail_l1,
                                            1, 0, 8 - 1))
                            # level_1 loop
                            loop_le1 = tail_l1 // 8
                            tail_le1 = tail_l1 % 8
                            if loop_le1 > 0:
                                with ib.for_range(0, loop_le1) as i2:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_cbuf_to_ubuf",
                                                f16_64.access_ptr('w'),
                                                L1_out.access_ptr(
                                                    'r',
                                                    offset=i2 * 64 * f16_c0),
                                                0, 1, 64, 0, 0))
                                    # level_2 loop
                                    loop_le2 = 8
                                    offset_le2 = expand_size * f16_c0
                                    repeat_le2 = expand_size // 8 + \
                                                 (
                                                     1 if expand_size % 8 > 0 else 0)
                                    with ib.for_range(0, loop_le2) as i3:
                                        with ib.new_scope():
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                2)
                                            ib.emit(
                                                tvm.call_extern(
                                                    "float16", "vadds",
                                                    f16_out.access_ptr(
                                                        'w',
                                                        offset=i3 * offset_le2),
                                                    f16_64.access_ptr(
                                                        'r',
                                                        offset=i3 * 8 * f16_c0),
                                                    tvm.const(0.0,
                                                              dtype="float16"),
                                                    repeat_le2, 1, 1, 8, 0))
                                    offset_4 = expand_size * f16_c0
                                    expand_4 = 8 * expand_size
                                    with ib.new_scope():
                                        _inner_loop = \
                                            expand_4 * f16_c0 // free_space_fp32
                                        _inner_tail = \
                                            expand_4 * f16_c0 % free_space_fp32
                                        with ib.for_range(
                                                0, _inner_loop) as inner_idx:
                                            ip_addr = [[f32_out, 0],
                                                       [f16_out,
                                                        inner_idx * free_space_fp32
                                                       ]]
                                            kernel_api.kernel_cast_to_fuc(
                                                ib, ip_addr,
                                                [free_space_fp32, 8 * 8],
                                                "vconv_f162f32")
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=((loop_level1 * 32 \
                                                    + loop_l1) * HW_SIZE_512 \
                                                    + i2 * 8) * offset_4 +
                                                    inner_idx * free_space_fp32),
                                                f32_out.access_ptr('r'),
                                                0, 1, free_space_fp32 // 8, 0,
                                                0))
                                        if _inner_tail > 0:
                                            ip_addr = [[f32_out, 0],
                                                       [
                                                           f16_out,
                                                           _inner_loop *
                                                           free_space_fp32
                                                       ]]
                                            kernel_api.kernel_cast_to_fuc(
                                                ib, ip_addr,
                                                [_inner_tail, 8 * 8],
                                                "vconv_f162f32")
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=((loop_level1 * 32 \
                                                    + loop_l1) * HW_SIZE_512 \
                                                    + i2 * 8) * offset_4 +
                                                    _inner_loop * free_space_fp32),
                                                f32_out.access_ptr('r'),
                                                0, 1, (_inner_tail + 7) // 8, 0,
                                                0))
                            if tail_le1 > 0:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 4)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float16", "copy_cbuf_to_ubuf",
                                            f16_64.access_ptr('w'),
                                            L1_out.access_ptr(
                                                'r',
                                                offset=loop_le1 * 64 * f16_c0),
                                            0,
                                            1, tail_le1 * 8, 0, 0))
                                # level_2 loop
                                loop_le2 = tail_le1
                                offset_le2 = expand_size * f16_c0
                                repeat_le2 = expand_size // 8 + \
                                             (1 if expand_size % 8 > 0 else 0)
                                with ib.for_range(0, loop_le2) as i4:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 2)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "vadds",
                                                f16_out.access_ptr(
                                                    'w',
                                                    offset=i4 * offset_le2),
                                                f16_64.access_ptr(
                                                    'r',
                                                    offset=i4 * 8 * f16_c0),
                                                tvm.const(0.0,
                                                          dtype="float16"),
                                                repeat_le2, 1, 1, 8, 0))
                                offset_4 = expand_size * f16_c0
                                expand_4 = tail_le1 * expand_size
                                with ib.new_scope():
                                    _inner_loop = \
                                        expand_4 * f16_c0 // free_space_fp32
                                    _inner_tail = \
                                        expand_4 * f16_c0 % free_space_fp32
                                    with ib.for_range(
                                            0, _inner_loop) as inner_idx:
                                        ip_addr = [[f32_out, 0],
                                                   [f16_out,
                                                    inner_idx * free_space_fp32
                                                   ]]
                                        kernel_api.kernel_cast_to_fuc(
                                            ib, ip_addr,
                                            [free_space_fp32, 8 * 8],
                                            "vconv_f162f32")
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=((loop_level1 * 32 \
                                                + loop_l1) * HW_SIZE_512 \
                                                + loop_le1 * 8) * offset_4 +
                                                inner_idx * free_space_fp32),
                                            f32_out.access_ptr('r'),
                                            0, 1, free_space_fp32 // 8, 0, 0))
                                    if _inner_tail > 0:
                                        ip_addr = [[f32_out, 0],
                                                   [f16_out,
                                                    _inner_loop * free_space_fp32
                                                   ]]
                                        kernel_api.kernel_cast_to_fuc(
                                            ib, ip_addr, [_inner_tail, 8 * 8],
                                            "vconv_f162f32")
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=((loop_level1 * 32 \
                                                + loop_l1) * HW_SIZE_512 \
                                                + loop_le1 * 8) * offset_4 +
                                                _inner_loop * free_space_fp32),
                                            f32_out.access_ptr('r'),
                                            0, 1, (_inner_tail + 7) // 8, 0, 0))
                        else:
                            # No.2 situation : 1. input f16 dtype : large data  3. H * W <= 32
                            with ib.for_range(0, expand_size) as i1:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 6)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float16", "copy_ubuf_to_cbuf",
                                            L1_out.access_ptr(
                                                'w',
                                                offset=i1 * f16_c0),
                                            f16_in.access_ptr('r'), 0, tail_l1,
                                            1, 0, expand_size - 1))
                            loop_le2 = tail_l1 * expand_size // (12 * HW_SIZE_512)
                            tail_le2 = tail_l1 * expand_size % (12 * HW_SIZE_512)
                            if loop_le2 > 0:
                                with ib.for_range(0, loop_le2) as i2:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float16", "copy_cbuf_to_ubuf",
                                                f16_out.access_ptr('w'),
                                                L1_out.access_ptr(
                                                    'r',
                                                    offset=i2 * HW_SIZE_512 *
                                                    12 * f16_c0),
                                                0, 1,
                                                HW_SIZE_512 * 12, 0,
                                                0))
                                    with ib.new_scope():
                                        _inner_loop = \
                                            HW_SIZE_512 * 12 * f16_c0 // free_space_fp32
                                        with ib.for_range(
                                                0, _inner_loop) as inner_idx:
                                            ip_addr = [[f32_out, 0],
                                                       [f16_out,
                                                        inner_idx * free_space_fp32
                                                       ]]
                                            kernel_api.kernel_cast_to_fuc(
                                                ib, ip_addr,
                                                [free_space_fp32, 8 * 8],
                                                "vconv_f162f32")
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=(((loop_level1 * 32 \
                                                    + loop_l1) \
                                                    * expand_size) \
                                                    + i2 * 12) \
                                                    * HW_SIZE_512 * f16_c0 +
                                                    inner_idx * free_space_fp32),
                                                f32_out.access_ptr('r'),
                                                0, 1, free_space_fp32 // 8, 0,
                                                0))
                            if tail_le2 > 0:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 4)
                                    ib.emit(tvm.call_extern(
                                        "float16",
                                        "copy_cbuf_to_ubuf",
                                        f16_out.access_ptr('w'),
                                        L1_out.access_ptr(
                                            'r',
                                            offset=loop_le2 * HW_SIZE_512 \
                                                   * 12 * f16_c0),
                                        0, 1, tail_le2, 0, 0))
                                with ib.new_scope():
                                    _inner_loop = \
                                        tail_le2 * f16_c0 // free_space_fp32
                                    _inner_tail = \
                                        tail_le2 * f16_c0 % free_space_fp32
                                    with ib.for_range(
                                            0, _inner_loop) as inner_idx:
                                        ip_addr = [[f32_out, 0],
                                                   [f16_out,
                                                    inner_idx * free_space_fp32
                                                   ]]
                                        kernel_api.kernel_cast_to_fuc(
                                            ib, ip_addr,
                                            [free_space_fp32, 8 * 8],
                                            "vconv_f162f32")
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=(((loop_level1 * 32 \
                                                + loop_l1) \
                                                * expand_size) \
                                                + loop_le2 * 12) \
                                                * HW_SIZE_512 * f16_c0 +
                                                inner_idx * free_space_fp32),
                                            f32_out.access_ptr('r'),
                                            0, 1, free_space_fp32 // 8, 0, 0))
                                    if _inner_tail > 0:
                                        ip_addr = [[f32_out, 0],
                                                   [f16_out,
                                                    _inner_loop * free_space_fp32
                                                   ]]
                                        kernel_api.kernel_cast_to_fuc(
                                            ib, ip_addr, [_inner_tail, 8 * 8],
                                            "vconv_f162f32")
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=(((loop_level1 * 32 \
                                                + loop_l1) \
                                                * expand_size) \
                                                + loop_le2 * 12) \
                                                * HW_SIZE_512 * f16_c0 +
                                                _inner_loop * free_space_fp32),
                                            f32_out.access_ptr('r'),
                                            0, 1, (_inner_tail + 7) // 8, 0, 0))
            else:
                # No.2 situation : 1. input f16 dtype : small data
                l1_half = HW_SIZE_512 * 32
                L1_out = apply_store_buffer(ib,
                                            "float16", [l1_half * f16_c0],
                                            name="L1_out",
                                            scope=param.scope_cbuf)
                with ib.new_scope():
                    ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                    ib.emit(
                        tvm.call_extern("float16", "copy_gm_to_ubuf",
                                        f16_in.access_ptr('w'),
                                        inputs.access_ptr('r'), 0, 1,
                                        expand_loop, 0, 0))
                # Note:
                #   1. H * W >= HW_SIZE_512 is common
                #   2. HW_SIZE_512 > H * W > 32
                #   3. H * W <= 32 this happen really rare

                # No.2 situation : 1. input f16 dtype : small data : 1. H * W >= HW_SIZE_512
                if expand_size > HW_SIZE_512:
                    # level_1 loop
                    loop_lev1 = 8
                    with ib.for_range(0, loop_lev1) as i1:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                            ib.emit(
                                tvm.call_extern(
                                    "float16", "copy_ubuf_to_cbuf",
                                    L1_out.access_ptr('w', offset=i1 * f16_c0),
                                    f16_in.access_ptr('r'), 0, expand_loop, 1,
                                    0, 8 - 1))
                    # level_1 loop
                    loop_lev1 = expand_loop
                    core_lp1 = loop_lev1 // core_counts
                    core_lp1_tail = loop_lev1 % core_counts

                    def _inner_run_lev1(core_lp1_idx):
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 4)
                            ib.emit(
                                tvm.call_extern(
                                    "float16", "copy_cbuf_to_ubuf",
                                    f16_8.access_ptr('w'),
                                    L1_out.access_ptr(
                                        'r',
                                        offset=block.var * 8 * f16_c0 +
                                        core_lp1_idx * core_counts * 8 * f16_c0),
                                    0, 1, 8, 0, 0))
                        repeat_64 = HW_SIZE_512 // 8
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 2)
                            ib.emit(
                                tvm.call_extern(
                                    "float16", "vadds",
                                    f16_out.access_ptr('w'),
                                    f16_8.access_ptr('r'),
                                    tvm.const(0.0, dtype="float16"), repeat_64,
                                    1, 1, 8, 0))
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 2)
                            ib.emit(
                                tvm.call_extern(
                                    "float16", "copy_ubuf_to_ubuf",
                                    f16_out.access_ptr('w',
                                                       offset=HW_SIZE_512 * f16_c0),
                                    f16_out.access_ptr('r'), 0, 1, HW_SIZE_512, 0, 0))
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 2)
                            ib.emit(
                                tvm.call_extern(
                                    "float16", "copy_ubuf_to_ubuf",
                                    f16_out.access_ptr('w',
                                                       offset=HW_SIZE_1024 * f16_c0),
                                    f16_out.access_ptr('r'), 0, 1, HW_SIZE_1024, 0, 0))
                        loop_lev2 = expand_size // HW_SIZE_2048
                        tail_lev2 = expand_size % HW_SIZE_2048
                        offset_2 = expand_size * f16_c0
                        if loop_lev2 > 0:
                            with ib.for_range(0, loop_lev2) as i3:
                                with ib.new_scope():
                                    _inner_loop = \
                                        HW_SIZE_2048 * f16_c0 // free_space_fp32
                                    _inner_tail = \
                                        HW_SIZE_2048 * f16_c0 % free_space_fp32
                                    with ib.for_range(
                                            0, _inner_loop) as inner_idx:
                                        ip_addr = [[f32_out, 0],
                                                   [f16_out,
                                                    inner_idx * free_space_fp32
                                                   ]]
                                        kernel_api.kernel_cast_to_fuc(
                                            ib, ip_addr,
                                            [free_space_fp32, 8 * 8],
                                            "vconv_f162f32")
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=block.var * offset_2 \
                                                + i3 * HW_SIZE_2048 * f16_c0 +
                                                inner_idx * free_space_fp32 +
                                                core_lp1_idx * core_counts *
                                                offset_2),
                                            f32_out.access_ptr('r'),
                                            0, 1, free_space_fp32 // 8, 0, 0))

                        if tail_lev2 > 0:
                            with ib.new_scope():
                                _inner_loop = \
                                    tail_lev2 * f16_c0 // free_space_fp32
                                _inner_tail = \
                                    tail_lev2 * f16_c0 % free_space_fp32
                                with ib.for_range(0, _inner_loop) as inner_idx:
                                    ip_addr = [[f32_out, 0],
                                               [
                                                   f16_out,
                                                   inner_idx * free_space_fp32
                                               ]]
                                    kernel_api.kernel_cast_to_fuc(
                                        ib, ip_addr, [free_space_fp32, 8 * 8],
                                        "vconv_f162f32")
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 6)
                                    ib.emit(tvm.call_extern(
                                        "float32", "copy_ubuf_to_gm",
                                        outputs.access_ptr(
                                            'w',
                                            offset=block.var * offset_2 \
                                            + loop_lev2 * HW_SIZE_2048 * f16_c0 +
                                            inner_idx * free_space_fp32 +
                                            core_lp1_idx * core_counts * offset_2),
                                        f32_out.access_ptr('r'),
                                        0, 1, free_space_fp32 // 8, 0, 0))

                                if _inner_tail > 0:
                                    ip_addr = [[f32_out, 0],
                                               [f16_out,
                                                _inner_loop * free_space_fp32
                                               ]]
                                    kernel_api.kernel_cast_to_fuc(
                                        ib, ip_addr, [_inner_tail, 8 * 8],
                                        "vconv_f162f32")
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 6)
                                    ib.emit(tvm.call_extern(
                                        "float32", "copy_ubuf_to_gm",
                                        outputs.access_ptr(
                                            'w',
                                            offset=block.var * offset_2 \
                                            + loop_lev2 * HW_SIZE_2048 * f16_c0 +
                                            _inner_loop * free_space_fp32 +
                                            core_lp1_idx * core_counts * offset_2),
                                        f32_out.access_ptr('r'),
                                        0, 1, (_inner_tail + 7) // 8, 0, 0))

                    # level_1 loop
                    with ib.for_range(0, core_lp1,
                                      name="core_lp1") as core_lp1_idx:
                        _inner_run_lev1(core_lp1_idx)
                    if core_lp1_tail > 0:
                        with ib.if_scope(block.var < core_lp1_tail):
                            _inner_run_lev1(core_lp1)

                elif expand_size > 32:
                    # No.2 situation : 1. input f16 dtype : small data : 2. HW_SIZE_512>=H*W>32
                    # level_1 loop
                    loop_lev1 = 8
                    with ib.for_range(0, loop_lev1) as i1:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                            ib.emit(
                                tvm.call_extern(
                                    "float16", "copy_ubuf_to_cbuf",
                                    L1_out.access_ptr('w', offset=i1 * f16_c0),
                                    f16_in.access_ptr('r'), 0, expand_loop, 1,
                                    0, 8 - 1))
                    # level_1 loop
                    loop_lev1 = expand_loop // 8
                    tail_lev1 = expand_loop % 8
                    if loop_lev1 > 0:
                        with ib.for_range(0, loop_lev1) as i2:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              4)
                                ib.emit(
                                    tvm.call_extern(
                                        "float16", "copy_cbuf_to_ubuf",
                                        f16_64.access_ptr('w'),
                                        L1_out.access_ptr(
                                            'r',
                                            offset=i2 * 64 * f16_c0),
                                        0, 1,
                                        64, 0,
                                        0))
                            # level_2 loop
                            loop_lev2 = 8
                            offset_lev2 = expand_size * f16_c0
                            repeat_lev2 = expand_size // 8 + \
                                          (1 if expand_size % 8 > 0 else 0)
                            with ib.for_range(0, loop_lev2) as i3:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 2)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float16", "vadds",
                                            f16_out.access_ptr(
                                                'w',
                                                offset=i3 * offset_lev2),
                                            f16_64.access_ptr(
                                                'r',
                                                offset=i3 * 8 * f16_c0),
                                            tvm.const(0.0, dtype="float16"),
                                            repeat_lev2, 1, 1, 8, 0))
                            offset_4 = expand_size * f16_c0
                            expand_4 = 8 * expand_size
                            with ib.new_scope():
                                _inner_loop = expand_4 * f16_c0 // free_space_fp32
                                _inner_tail = expand_4 * f16_c0 % free_space_fp32
                                with ib.for_range(0, _inner_loop) as inner_idx:
                                    ip_addr = [[f32_out, 0],
                                               [
                                                   f16_out,
                                                   inner_idx * free_space_fp32
                                               ]]
                                    kernel_api.kernel_cast_to_fuc(
                                        ib, ip_addr, [free_space_fp32, 8 * 8],
                                        "vconv_f162f32")
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 6)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=i2 * 8 * offset_4 +
                                                inner_idx * free_space_fp32),
                                            f32_out.access_ptr('r'), 0, 1,
                                            free_space_fp32 // 8, 0, 0))
                                if _inner_tail > 0:
                                    ip_addr = [[f32_out, 0],
                                               [f16_out,
                                                _inner_loop * free_space_fp32
                                               ]]
                                    kernel_api.kernel_cast_to_fuc(
                                        ib, ip_addr, [_inner_tail, 8 * 8],
                                        "vconv_f162f32")
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 6)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=i2 * 8 * offset_4 +
                                                _inner_loop * free_space_fp32),
                                            f32_out.access_ptr('r'), 0, 1,
                                            (_inner_tail + 7) // 8, 0, 0))
                    if tail_lev1 > 0:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 4)
                            ib.emit(
                                tvm.call_extern(
                                    "float16", "copy_cbuf_to_ubuf",
                                    f16_64.access_ptr('w'),
                                    L1_out.access_ptr(
                                        'r',
                                        offset=loop_lev1 * 64 * f16_c0),
                                    0, 1,
                                    tail_lev1 * 8, 0, 0))
                        # level_2 loop
                        loop_lev2 = tail_lev1
                        offset_lev2 = expand_size * f16_c0
                        repeat_lev2 = expand_size // 8 + \
                                      (1 if expand_size % 8 > 0 else 0)
                        with ib.for_range(0, loop_lev2) as i4:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              2)
                                ib.emit(
                                    tvm.call_extern(
                                        "float16", "vadds",
                                        f16_out.access_ptr(
                                            'w',
                                            offset=i4 * offset_lev2),
                                        f16_64.access_ptr(
                                            'r',
                                            offset=i4 * 8 * f16_c0),
                                        tvm.const(0.0, dtype="float16"),
                                        repeat_lev2, 1, 1, 8, 0))
                        offset_4 = expand_size * f16_c0
                        expand_4 = tail_lev1 * expand_size
                        with ib.new_scope():
                            _inner_loop = expand_4 * f16_c0 // free_space_fp32
                            _inner_tail = expand_4 * f16_c0 % free_space_fp32
                            with ib.for_range(0, _inner_loop) as inner_idx:
                                ip_addr = [[f32_out, 0],
                                           [
                                               f16_out,
                                               inner_idx * free_space_fp32
                                           ]]
                                kernel_api.kernel_cast_to_fuc(
                                    ib, ip_addr, [free_space_fp32, 8 * 8],
                                    "vconv_f162f32")
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              6)
                                ib.emit(
                                    tvm.call_extern(
                                        "float32", "copy_ubuf_to_gm",
                                        outputs.access_ptr(
                                            'w',
                                            offset=loop_lev1 * 8 * offset_4 +
                                            inner_idx * free_space_fp32),
                                        f32_out.access_ptr('r'), 0, 1,
                                        free_space_fp32 // 8, 0, 0))
                            if _inner_tail > 0:
                                ip_addr = [[f32_out, 0],
                                           [
                                               f16_out,
                                               _inner_loop * free_space_fp32
                                           ]]
                                kernel_api.kernel_cast_to_fuc(
                                    ib, ip_addr, [_inner_tail, 8 * 8],
                                    "vconv_f162f32")
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              6)
                                ib.emit(
                                    tvm.call_extern(
                                        "float32", "copy_ubuf_to_gm",
                                        outputs.access_ptr(
                                            'w',
                                            offset=loop_lev1 * 8 * offset_4 +
                                            _inner_loop * free_space_fp32),
                                        f32_out.access_ptr('r'), 0, 1,
                                        (_inner_tail + 7) // 8, 0, 0))

                else:
                    # No.2 situation : 1. input f16 dtype : small data  3. H * W <= 16
                    with ib.for_range(0, expand_size) as i1:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                            ib.emit(
                                tvm.call_extern(
                                    "float16", "copy_ubuf_to_cbuf",
                                    L1_out.access_ptr('w', offset=i1 * f16_c0),
                                    f16_in.access_ptr('r'), 0, expand_loop, 1,
                                    0, expand_size - 1))
                    loop_lev2 = expand_loop * expand_size // (12 * HW_SIZE_512)
                    tail_lev2 = expand_loop * expand_size % (12 * HW_SIZE_512)
                    if loop_lev2 > 0:
                        with ib.for_range(0, loop_lev2) as i2:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              4)
                                ib.emit(
                                    tvm.call_extern(
                                        "float16", "copy_cbuf_to_ubuf",
                                        f16_out.access_ptr('w'),
                                        L1_out.access_ptr(
                                            'r',
                                            offset=i2 * HW_SIZE_512 * 12 * f16_c0),
                                        0, 1,
                                        HW_SIZE_512 * 12, 0, 0))
                            with ib.new_scope():
                                _inner_loop = HW_SIZE_512 * 12 * f16_c0 // free_space_fp32
                                with ib.for_range(0, _inner_loop) as inner_idx:
                                    ip_addr = [[f32_out, 0],
                                               [
                                                   f16_out,
                                                   inner_idx * free_space_fp32
                                               ]]
                                    kernel_api.kernel_cast_to_fuc(
                                        ib, ip_addr, [free_space_fp32, 8 * 8],
                                        "vconv_f162f32")
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 6)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=i2 * 12 * HW_SIZE_512 *
                                                f16_c0 +
                                                inner_idx * free_space_fp32),
                                            f32_out.access_ptr('r'), 0, 1,
                                            free_space_fp32 // 8, 0, 0))
                    if tail_lev2 > 0:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 4)
                            ib.emit(
                                tvm.call_extern(
                                    "float16", "copy_cbuf_to_ubuf",
                                    f16_out.access_ptr('w'),
                                    L1_out.access_ptr(
                                        'r',
                                        offset=loop_lev2 * HW_SIZE_512 * 12 * f16_c0),
                                    0, 1,
                                    tail_lev2, 0, 0))
                        with ib.new_scope():
                            _inner_loop = tail_lev2 * f16_c0 // free_space_fp32
                            _inner_tail = tail_lev2 * f16_c0 % free_space_fp32
                            with ib.for_range(0, _inner_loop) as inner_idx:
                                ip_addr = [[f32_out, 0],
                                           [
                                               f16_out,
                                               inner_idx * free_space_fp32
                                           ]]
                                kernel_api.kernel_cast_to_fuc(
                                    ib, ip_addr, [free_space_fp32, 8 * 8],
                                    "vconv_f162f32")
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              6)
                                ib.emit(
                                    tvm.call_extern(
                                        "float32", "copy_ubuf_to_gm",
                                        outputs.access_ptr(
                                            'w',
                                            offset=loop_lev2 * 12 *
                                            HW_SIZE_512 * f16_c0 +
                                            inner_idx * free_space_fp32),
                                        f32_out.access_ptr('r'), 0, 1,
                                        free_space_fp32, 0, 0))
                            if _inner_tail > 0:
                                ip_addr = [[f32_out, 0],
                                           [
                                               f16_out,
                                               _inner_loop * free_space_fp32
                                           ]]
                                kernel_api.kernel_cast_to_fuc(
                                    ib, ip_addr, [_inner_tail, 8 * 8],
                                    "vconv_f162f32")
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              6)
                                ib.emit(
                                    tvm.call_extern(
                                        "float32", "copy_ubuf_to_gm",
                                        outputs.access_ptr(
                                            'w',
                                            offset=loop_lev2 * 12 *
                                            HW_SIZE_512 * f16_c0 +
                                            _inner_loop * free_space_fp32),
                                        f32_out.access_ptr('r'), 0, 1,
                                        (_inner_tail + 7) // 8, 0, 0))

        else:
            # No.2 situation : 2. input f32 dtype
            expand_loop = reduce_func(lambda x, y: x * y, size_in) // f16_c0
            actual_loop = min(expand_loop, HW_SIZE_512)
            free_space = HW_SIZE_512 * 12
            expand_size = h_out * w_out
            ib.scope_attr(block, "thread_extent", core_counts)

            f32_in = apply_store_buffer(ib,
                                        "float32", [HW_SIZE_1024 * f32_c0],
                                        name="f32_in")
            f32_out = apply_store_buffer(ib,
                                         "float32", [free_space * f32_c0],
                                         name="f32_out")

            if expand_size > HW_SIZE_512:
                f32_8 = apply_store_buffer(ib,
                                           "float32", [8 * f32_c0],
                                           name="f32_8")
            elif expand_size > 16:
                f32_32 = apply_store_buffer(ib,
                                            "float32", [32 * f32_c0],
                                            name="f32_32")
            # Note:
            #   1. input data larger than HW_SIZE_512 should use L1 optimize
            #   2. input small data do not need L1

            if expand_size > 16:
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const((1 << 64) - 1, dtype="uint64")))
            # No.2 situation : 1. input f32 dtype : large data
            if expand_loop > HW_SIZE_512:
                l1_half = HW_SIZE_512 * 32
                in_L1 = apply_store_buffer(ib,
                                           "float32", [l1_half * f32_c0],
                                           name="in_L1",
                                           scope=param.scope_cbuf)
                out_L1 = apply_store_buffer(ib,
                                            "float32", [l1_half * f32_c0],
                                            name="out_L1",
                                            scope=param.scope_cbuf)
                f32_half = l1_half // 2
                loop_level1 = expand_loop // f32_half
                tail_level1 = expand_loop % f32_half

                # No.2 situation : 1. input f32 dtype : large data : loop_level1 > 0
                if loop_level1 > 0:
                    # level_1 loop
                    with ib.for_range(0, loop_level1) as i1:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                            ib.emit(
                                tvm.call_extern(
                                    "float32", "copy_gm_to_cbuf",
                                    in_L1.access_ptr('w'),
                                    inputs.access_ptr(
                                        'r',
                                        offset=i1 * f32_half * f16_c0), sid, 1,
                                    l1_half,
                                    0, 0, pad_mode))
                        # level_2 loop
                        loop_level2 = 16
                        with ib.for_range(0, loop_level2) as i2:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              4)
                                ib.emit(
                                    tvm.call_extern(
                                        "float32", "copy_cbuf_to_ubuf",
                                        f32_in.access_ptr('w'),
                                        in_L1.access_ptr(
                                            'r',
                                            offset=i2 * HW_SIZE_512 * f16_c0),
                                        0, 1, HW_SIZE_1024, 0, 0))
                            # Note:
                            #   1. H * W > HW_SIZE_512 is common
                            #   2. HW_SIZE_512 >= H * W > 16
                            #   3. H * W <= 16 this happen really rare

                            # No.2 situation : 1. input f32 dtype : large data : 1. H * W > 512
                            if expand_size > HW_SIZE_512:
                                # level_3 loop
                                loop_level3 = 4
                                with ib.for_range(0, loop_level3) as i3:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_ubuf_to_cbuf",
                                                out_L1.access_ptr(
                                                    'w',
                                                    offset=i3 * f16_c0),
                                                f32_in.access_ptr('r'), 0, HW_SIZE_512,
                                                2, 0, (4 - 1) * 2))
                                # level_3 loop
                                loop_level3 = HW_SIZE_512
                                core_lp3 = loop_level3 // core_counts
                                core_lp3_tail = loop_level3 % core_counts

                                def _inner_run_lev3(core_lp3_idx):
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_cbuf_to_ubuf",
                                                f32_8.access_ptr('w'),
                                                out_L1.access_ptr(
                                                    'r',
                                                    offset=block.var * 4 *
                                                    f16_c0 + core_lp3_idx *
                                                    core_counts * 4 * f16_c0),
                                                0, 1, 8, 0, 0))
                                    repeat_128 = HW_SIZE_512 // 4
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 2)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "vadds",
                                                f32_out.access_ptr('w'),
                                                f32_8.access_ptr('r'),
                                                tvm.const(0.0,
                                                          dtype="float32"),
                                                repeat_128, 1, 1, 8, 0))
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 2)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_ubuf_to_ubuf",
                                                f32_out.access_ptr(
                                                    'w',
                                                    offset=HW_SIZE_512 * f16_c0),
                                                f32_out.access_ptr('r'), 0, 1,
                                                HW_SIZE_1024, 0, 0))
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 2)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_ubuf_to_ubuf",
                                                f32_out.access_ptr(
                                                    'w', offset=HW_SIZE_1024 * f16_c0),
                                                f32_out.access_ptr('r'), 0, 1,
                                                HW_SIZE_2048, 0, 0))
                                    loop_level4 = expand_size // HW_SIZE_2048
                                    tail_level4 = expand_size % HW_SIZE_2048
                                    offset_4 = expand_size * f16_c0
                                    if loop_level4 > 0:
                                        with ib.for_range(0,
                                                          loop_level4) as i5:
                                            with ib.new_scope():
                                                ib.scope_attr(
                                                    param.CCE_AXIS,
                                                    "coproc_scope", 6)
                                                ib.emit(tvm.call_extern(
                                                    "float32",
                                                    "copy_ubuf_to_gm",
                                                    outputs.access_ptr(
                                                        'w',
                                                        offset=((i1 * 16 + i2) * \
                                                        HW_SIZE_512 + block.var)
                                                        * offset_4 + i5 *
                                                        HW_SIZE_2048 * f16_c0 +
                                                        core_lp3_idx *
                                                        core_counts * offset_4),
                                                    f32_out.access_ptr('r'),
                                                    0, 1, HW_SIZE_2048 * 2,
                                                    0, 0))
                                    if tail_level4 > 0:
                                        with ib.new_scope():
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=((i1 * 16 + i2) * \
                                                    HW_SIZE_512 + block.var) *
                                                    offset_4 +
                                                    loop_level4 * HW_SIZE_2048
                                                    * f16_c0 + core_lp3_idx *
                                                    core_counts * offset_4),
                                                f32_out.access_ptr('r'),
                                                0, 1, tail_level4 * 2, 0, 0))

                                # level_3 loop
                                with ib.for_range(
                                        0, core_lp3,
                                        name="core_lp3") as core_lp3_idx:
                                    _inner_run_lev3(core_lp3_idx)
                                if core_lp3_tail > 0:
                                    with ib.if_scope(
                                            block.var < core_lp3_tail):
                                        _inner_run_lev3(core_lp3)

                            elif expand_size > 16:
                                # No.2 situation : 1. input f32 dtype : large data : 512>=H*W>16
                                # level_3 loop
                                loop_level3 = 4
                                with ib.for_range(0, loop_level3) as i3:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_ubuf_to_cbuf",
                                                out_L1.access_ptr(
                                                    'w',
                                                    offset=i3 * f16_c0),
                                                f32_in.access_ptr('r'), 0, HW_SIZE_512,
                                                2, 0, (4 - 1) * 2))
                                # level_3 loop
                                loop_level3 = HW_SIZE_512 // 4
                                core_lp3 = loop_level3 // core_counts
                                core_lp3_tail = loop_level3 % core_counts

                                def _inner_run_lev3_1(core_lp3_idx):
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_cbuf_to_ubuf",
                                                f32_32.access_ptr('w'),
                                                out_L1.access_ptr(
                                                    'r',
                                                    offset=block.var * 32 *
                                                    f32_c0 + core_lp3_idx *
                                                    core_counts * 32 * f32_c0),
                                                0, 1, 32, 0, 0))
                                    # level_4 loop
                                    loop_level4 = 4
                                    offset_l4 = expand_size * f16_c0
                                    repeat_l4 = \
                                        expand_size // 4 + \
                                        (1 if expand_size % 4 > 0 else 0)
                                    with ib.for_range(0, loop_level4) as i5:
                                        with ib.new_scope():
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                2)
                                            ib.emit(
                                                tvm.call_extern(
                                                    "float32", "vadds",
                                                    f32_out.access_ptr(
                                                        'w',
                                                        offset=i5 * offset_l4),
                                                    f32_32.access_ptr(
                                                        'r',
                                                        offset=i5 * 4 * f16_c0),
                                                    tvm.const(0.0,
                                                              dtype="float32"),
                                                    repeat_l4, 1, 1, 8, 0))
                                    offset_4 = expand_size * f16_c0
                                    expand_4 = 4 * expand_size * 2
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=((i1 * 16 + i2) * \
                                                HW_SIZE_512 \
                                                 + block.var * 4) * offset_4 +
                                                core_lp3_idx * core_counts *
                                                4 * offset_4),
                                            f32_out.access_ptr('r'),
                                            0, 1, expand_4, 0, 0))

                                # level_3 loop
                                with ib.for_range(
                                        0, core_lp3,
                                        name="core_lp3") as core_lp3_idx:
                                    _inner_run_lev3_1(core_lp3_idx)
                                if core_lp3_tail > 0:
                                    with ib.if_scope(
                                            block.var < core_lp3_tail):
                                        _inner_run_lev3_1(core_lp3)

                            else:
                                # No.2 situation : 1. input f32 dtype : large data  3. H * W <= 16
                                with ib.for_range(0, expand_size) as i3:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_ubuf_to_cbuf",
                                                out_L1.access_ptr(
                                                    'w',
                                                    offset=i3 * f16_c0),
                                                f32_in.access_ptr('r'), 0, HW_SIZE_512,
                                                2, 0, (expand_size - 1) * 2))
                                loop_level3 = expand_size * 2 // 12
                                tail_level3 = expand_size * 2 % 12
                                if loop_level3 > 0:
                                    with ib.for_range(0, loop_level3) as i4:
                                        with ib.new_scope():
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                4)
                                            ib.emit(
                                                tvm.call_extern(
                                                    "float32",
                                                    "copy_cbuf_to_ubuf",
                                                    f32_out.access_ptr('w'),
                                                    out_L1.access_ptr(
                                                        'r',
                                                        offset=i4 * HW_SIZE_512
                                                        * 6 * f16_c0),
                                                    0, 1,
                                                    HW_SIZE_512 * 12, 0, 0))
                                        with ib.new_scope():
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=(((i1 * 16 + i2) \
                                                             * expand_size) \
                                                            + i4 * 6) \
                                                           * HW_SIZE_512 * f16_c0),
                                                f32_out.access_ptr('r'),
                                                0, 1, HW_SIZE_512 * 12, 0, 0))
                                if tail_level3 > 0:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(tvm.call_extern(
                                            "float32",
                                            "copy_cbuf_to_ubuf",
                                            f32_out.access_ptr('w'),
                                            out_L1.access_ptr(
                                                'r',
                                                offset=loop_level3 * HW_SIZE_512 \
                                                       * 6 * f16_c0),
                                            0, 1, HW_SIZE_512 * tail_level3, 0, 0))
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=(((i1 * 16 + i2) \
                                                         * expand_size) \
                                                        + loop_level3 * 6) \
                                                       * HW_SIZE_512 * f16_c0),
                                            f32_out.access_ptr('r'),
                                            0, 1, HW_SIZE_512 * tail_level3, 0, 0))
                # No.2 situation : 1. input f32 dtype : large data : tail_level1 > 0
                if tail_level1 > 0:
                    if tail_level1 > HW_SIZE_512:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                            ib.emit(
                                tvm.call_extern(
                                    "float32", "copy_gm_to_cbuf",
                                    in_L1.access_ptr('w'),
                                    inputs.access_ptr(
                                        'r',
                                        offset=loop_level1 * f32_half * f16_c0),
                                    sid,
                                    1, tail_level1 * 2, 0, 0, pad_mode))
                    else:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                            ib.emit(
                                tvm.call_extern(
                                    "float32", "copy_gm_to_ubuf",
                                    f32_in.access_ptr('w'),
                                    inputs.access_ptr(
                                        'r',
                                        offset=loop_level1 * f32_half * f16_c0),
                                    0, 1,
                                    tail_level1 * 2, 0, 0))
                    # level_1 loop
                    loop_l1 = tail_level1 // HW_SIZE_512
                    tail_l1 = tail_level1 % HW_SIZE_512
                    # No.2 situation : 1. input f32 dtype : large data : tail_level1>0 : loop_l1>0
                    if loop_l1 > 0:
                        with ib.for_range(0, loop_l1) as i1:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              4)
                                ib.emit(
                                    tvm.call_extern(
                                        "float32", "copy_cbuf_to_ubuf",
                                        f32_in.access_ptr('w'),
                                        in_L1.access_ptr(
                                            'r',
                                            offset=i1 * HW_SIZE_512 * f16_c0), 0, 1,
                                        HW_SIZE_1024,
                                        0, 0))
                            # Note:
                            #   1. H * W >= HW_SIZE_512 is common
                            #   2. HW_SIZE_512 > H * W > 16
                            #   3. H * W <= 16 this happen really rare

                            # No.2 situation : 1. input f32 dtype : large data : 1. H * W >= 512
                            if expand_size > HW_SIZE_512:
                                # level_2 loop
                                loop_l2 = 4
                                with ib.for_range(0, loop_l2) as i2:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_ubuf_to_cbuf",
                                                out_L1.access_ptr(
                                                    'w',
                                                    offset=i2 * f16_c0),
                                                f32_in.access_ptr('r'), 0, HW_SIZE_512,
                                                2, 0, (4 - 1) * 2))
                                # level_2 loop
                                loop_l2 = HW_SIZE_512
                                core_lp2 = loop_l2 // core_counts
                                core_lp2_tail = loop_l2 % core_counts

                                def _inner_run_lp2(core_lp2_idx):
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_cbuf_to_ubuf",
                                                f32_8.access_ptr('w'),
                                                out_L1.access_ptr(
                                                    'r',
                                                    offset=block.var * 4 *
                                                    f16_c0 + core_lp2_idx *
                                                    core_counts * 4 * f16_c0),
                                                0, 1, 8, 0, 0))
                                    repeat_128 = HW_SIZE_512 // 4
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 2)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "vadds",
                                                f32_out.access_ptr('w'),
                                                f32_8.access_ptr('r'),
                                                tvm.const(0.0,
                                                          dtype="float32"),
                                                repeat_128, 1, 1, 8, 0))
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 2)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_ubuf_to_ubuf",
                                                f32_out.access_ptr(
                                                    'w',
                                                    offset=HW_SIZE_512 * f16_c0),
                                                f32_out.access_ptr('r'), 0, 1,
                                                HW_SIZE_1024, 0, 0))
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 2)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_ubuf_to_ubuf",
                                                f32_out.access_ptr(
                                                    'w', offset=HW_SIZE_1024 * f16_c0),
                                                f32_out.access_ptr('r'), 0, 1,
                                                HW_SIZE_2048, 0, 0))
                                    loop_l3 = expand_size // HW_SIZE_2048
                                    tail_l3 = expand_size % HW_SIZE_2048
                                    offset_3 = expand_size * f16_c0
                                    if loop_l3 > 0:
                                        with ib.for_range(0, loop_l3) as i4:
                                            with ib.new_scope():
                                                ib.scope_attr(
                                                    param.CCE_AXIS,
                                                    "coproc_scope", 6)
                                                ib.emit(tvm.call_extern(
                                                    "float32",
                                                    "copy_ubuf_to_gm",
                                                    outputs.access_ptr(
                                                        'w',
                                                        offset=((loop_level1 \
                                                          * 16 + i1) * HW_SIZE_512 \
                                                         + block.var) * offset_3 \
                                                        + i4 * HW_SIZE_2048 * f16_c0 +
                                                        core_lp2_idx *
                                                        core_counts * offset_3),
                                                    f32_out.access_ptr('r'),
                                                    0, 1, HW_SIZE_2048 * 2, 0, 0))
                                    if tail_l3 > 0:
                                        with ib.new_scope():
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=((loop_level1 * 16 \
                                                      + i1) * HW_SIZE_512 + \
                                                    block.var) * offset_3 \
                                                    + loop_l3 * HW_SIZE_2048 *
                                                    f16_c0 + core_lp2_idx *
                                                    core_counts * offset_3),
                                                f32_out.access_ptr('r'),
                                                0, 1, tail_l3 * 2, 0, 0))

                                with ib.for_range(
                                        0, core_lp2,
                                        name="core_lp2") as core_lp2_idx:
                                    _inner_run_lp2(core_lp2_idx)
                                if core_lp2_tail > 0:
                                    with ib.if_scope(
                                            block.var < core_lp2_tail):
                                        _inner_run_lp2(core_lp2)

                            elif expand_size > 16:
                                # No.2 situation : 1. input f32 dtype : large data : 2. 512>=H*W>16
                                # level_2 loop
                                loop_l2 = 4
                                with ib.for_range(0, loop_l2) as i2:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_ubuf_to_cbuf",
                                                out_L1.access_ptr(
                                                    'w',
                                                    offset=i2 * f16_c0),
                                                f32_in.access_ptr('r'), 0, HW_SIZE_512,
                                                2, 0, (4 - 1) * 2))
                                # level_2 loop
                                loop_l2 = HW_SIZE_512 // 4
                                with ib.for_range(0, loop_l2) as i3:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_cbuf_to_ubuf",
                                                f32_32.access_ptr('w'),
                                                out_L1.access_ptr(
                                                    'r',
                                                    offset=i3 * 32 * f32_c0),
                                                0, 1, 32, 0, 0))
                                    # level_3 loop
                                    loop_l3 = 4
                                    offset_l3 = expand_size * f16_c0
                                    repeat_l3 = expand_size // 4 + \
                                                (
                                                    1 if expand_size % 4 > 0 else 0)
                                    with ib.for_range(0, loop_l3) as i4:
                                        with ib.new_scope():
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                2)
                                            ib.emit(
                                                tvm.call_extern(
                                                    "float32", "vadds",
                                                    f32_out.access_ptr(
                                                        'w',
                                                        offset=i4 * offset_l3),
                                                    f32_32.access_ptr(
                                                        'r',
                                                        offset=i4 * 4 * f16_c0),
                                                    tvm.const(0.0,
                                                              dtype="float32"),
                                                    repeat_l3, 1, 1, 8, 0))
                                    offset_4 = expand_size * f16_c0
                                    expand_4 = 4 * expand_size * 2
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=((loop_level1 * 16 + i1) \
                                                        * HW_SIZE_512 + i3 * 4) * offset_4),
                                            f32_out.access_ptr('r'),
                                            0, 1, expand_4, 0, 0))

                            else:
                                # No.2 situation : 1. input f32 dtype : large data  3. H * W <= 16
                                with ib.for_range(0, expand_size) as i2:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_ubuf_to_cbuf",
                                                out_L1.access_ptr(
                                                    'w',
                                                    offset=i2 * f16_c0),
                                                f32_in.access_ptr('r'), 0, HW_SIZE_512,
                                                2, 0, (expand_size - 1) * 2))
                                loop_l2 = expand_size * 2 // 12
                                tail_l2 = expand_size * 2 % 12
                                if loop_l2 > 0:
                                    with ib.for_range(0, loop_l2) as i3:
                                        with ib.new_scope():
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                4)
                                            ib.emit(
                                                tvm.call_extern(
                                                    "float32",
                                                    "copy_cbuf_to_ubuf",
                                                    f32_out.access_ptr('w'),
                                                    out_L1.access_ptr(
                                                        'r',
                                                        offset=i3 * HW_SIZE_512
                                                        * 6 * f16_c0),
                                                    0, 1,
                                                    HW_SIZE_512 * 12, 0, 0))
                                        with ib.new_scope():
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=(((loop_level1 * 16 \
                                                              + i1) * expand_size) \
                                                            + i3 * 6) \
                                                           * HW_SIZE_512 * f16_c0),
                                                f32_out.access_ptr('r'),
                                                0, 1, HW_SIZE_512 * 12, 0, 0))
                                if tail_l2 > 0:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(tvm.call_extern(
                                            "float32",
                                            "copy_cbuf_to_ubuf",
                                            f32_out.access_ptr('w'),
                                            out_L1.access_ptr(
                                                'r',
                                                offset=loop_l2 * HW_SIZE_512 \
                                                       * 6 * f16_c0),
                                            0, 1, HW_SIZE_512 * tail_l2, 0, 0))
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=(((loop_level1 * 16 \
                                                          + i1) * expand_size) \
                                                        + loop_l2 * 6) \
                                                       * HW_SIZE_512 * f16_c0),
                                            f32_out.access_ptr('r'),
                                            0, 1, HW_SIZE_512 * tail_l2, 0, 0))
                    # No.2 situation : 1. input f32 dtype : large data : tail_level1>0 : tail_l1>0
                    if tail_l1 > 0:
                        if tail_level1 > HW_SIZE_512:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              4)
                                ib.emit(
                                    tvm.call_extern(
                                        "float32", "copy_cbuf_to_ubuf",
                                        f32_in.access_ptr('w'),
                                        in_L1.access_ptr(
                                            'r',
                                            offset=loop_l1 * HW_SIZE_512 * f16_c0),
                                        0, 1,
                                        tail_l1 * 2, 0, 0))
                        # Note:
                        #   1. H * W >= HW_SIZE_512 is common
                        #   2. HW_SIZE_512 > H * W > 16
                        #   3. H * W <= 16 this happen really rare

                        # No.2 situation : 1. input f32 dtype : large data : 1. H * W >= HW_SIZE_512
                        if expand_size > HW_SIZE_512:
                            # level_1 loop
                            loop_le1 = 4
                            with ib.for_range(0, loop_le1) as i1:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 6)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float32", "copy_ubuf_to_cbuf",
                                            out_L1.access_ptr(
                                                'w',
                                                offset=i1 * f16_c0),
                                            f32_in.access_ptr('r'), 0, tail_l1,
                                            2, 0, (4 - 1) * 2))
                            # level_1 loop
                            loop_le1 = tail_l1
                            with ib.for_range(0, loop_le1) as i2:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 4)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float32", "copy_cbuf_to_ubuf",
                                            f32_8.access_ptr('w'),
                                            out_L1.access_ptr(
                                                'r',
                                                offset=i2 * 4 * f16_c0), 0,
                                            1, 8,
                                            0, 0))
                                repeat_128 = HW_SIZE_512 // 4
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 2)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float32", "vadds",
                                            f32_out.access_ptr('w'),
                                            f32_8.access_ptr('r'),
                                            tvm.const(0.0, dtype="float32"),
                                            repeat_128, 1, 1, 8, 0))
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 2)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float32", "copy_ubuf_to_ubuf",
                                            f32_out.access_ptr(
                                                'w',
                                                offset=HW_SIZE_512 * f16_c0),
                                            f32_out.access_ptr('r'), 0, 1,
                                            HW_SIZE_1024, 0, 0))
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 2)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float32", "copy_ubuf_to_ubuf",
                                            f32_out.access_ptr(
                                                'w',
                                                offset=HW_SIZE_1024 * f16_c0),
                                            f32_out.access_ptr('r'), 0, 1,
                                            HW_SIZE_2048, 0, 0))
                                loop_le2 = expand_size // HW_SIZE_2048
                                tail_le2 = expand_size % HW_SIZE_2048
                                offset_2 = expand_size * f16_c0
                                if loop_le2 > 0:
                                    with ib.for_range(0, loop_le2) as i3:
                                        with ib.new_scope():
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                6)
                                            ib.emit(tvm.call_extern(
                                                "float32", "copy_ubuf_to_gm",
                                                outputs.access_ptr(
                                                    'w',
                                                    offset=((loop_level1 * 16 \
                                                             + loop_l1) * HW_SIZE_512 \
                                                            + i2) * offset_2 \
                                                           + i3 * HW_SIZE_2048 * f16_c0),
                                                f32_out.access_ptr('r'),
                                                0, 1, HW_SIZE_2048 * 2, 0, 0))
                                if tail_le2 > 0:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=((loop_level1 * 16 \
                                                         + loop_l1) * HW_SIZE_512 \
                                                        + i2) * offset_2 \
                                                       + loop_le2 * HW_SIZE_2048 \
                                                       * f16_c0),
                                            f32_out.access_ptr('r'),
                                            0, 1, tail_le2 * 2, 0, 0))

                        elif expand_size > 16:
                            # No.2 situation : 1. input f32 dtype : large data : 2. 512>=H*W>16
                            # level_1 loop
                            loop_le1 = 4
                            with ib.for_range(0, loop_le1) as i1:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 6)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float32", "copy_ubuf_to_cbuf",
                                            out_L1.access_ptr(
                                                'w',
                                                offset=i1 * f16_c0),
                                            f32_in.access_ptr('r'), 0, tail_l1,
                                            2, 0, (4 - 1) * 2))
                            # level_1 loop
                            loop_le1 = tail_l1 // 4
                            tail_le1 = tail_l1 % 4
                            if loop_le1 > 0:
                                with ib.for_range(0, loop_le1) as i2:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_cbuf_to_ubuf",
                                                f32_32.access_ptr('w'),
                                                out_L1.access_ptr(
                                                    'r',
                                                    offset=i2 * 32 * f32_c0),
                                                0, 1, 32, 0, 0))
                                    # level_2 loop
                                    loop_le2 = 4
                                    offset_le2 = expand_size * f16_c0
                                    repeat_le2 = expand_size // 4 + \
                                                 (
                                                     1 if expand_size % 4 > 0 else 0)
                                    with ib.for_range(0, loop_le2) as i3:
                                        with ib.new_scope():
                                            ib.scope_attr(
                                                param.CCE_AXIS, "coproc_scope",
                                                2)
                                            ib.emit(
                                                tvm.call_extern(
                                                    "float32", "vadds",
                                                    f32_out.access_ptr(
                                                        'w',
                                                        offset=i3 * offset_le2),
                                                    f32_32.access_ptr(
                                                        'r',
                                                        offset=i3 * 4 * f16_c0),
                                                    tvm.const(0.0,
                                                              dtype="float32"),
                                                    repeat_le2, 1, 1, 8, 0))
                                    offset_4 = expand_size * f16_c0
                                    expand_4 = 4 * expand_size * 2
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=((loop_level1 * 16 \
                                                         + loop_l1) * HW_SIZE_512 \
                                                        + i2 * 4) * offset_4),
                                            f32_out.access_ptr('r'),
                                            0, 1, expand_4, 0, 0))
                            if tail_le1 > 0:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 4)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float32", "copy_cbuf_to_ubuf",
                                            f32_32.access_ptr('w'),
                                            out_L1.access_ptr(
                                                'r',
                                                offset=loop_le1 * 32 * f32_c0),
                                            0,
                                            1, tail_le1 * 8, 0, 0))
                                # level_2 loop
                                loop_le2 = tail_le1
                                offset_le2 = expand_size * f16_c0
                                repeat_le2 = expand_size // 4 + \
                                             (1 if expand_size % 4 > 0 else 0)
                                with ib.for_range(0, loop_le2) as i4:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 2)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "vadds",
                                                f32_out.access_ptr(
                                                    'w',
                                                    offset=i4 * offset_le2),
                                                f32_32.access_ptr(
                                                    'r',
                                                    offset=i4 * 4 * f16_c0),
                                                tvm.const(0.0,
                                                          dtype="float32"),
                                                repeat_le2, 1, 1, 8, 0))
                                offset_4 = expand_size * f16_c0
                                expand_4 = tail_le1 * expand_size * 2
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 6)
                                    ib.emit(tvm.call_extern(
                                        "float32", "copy_ubuf_to_gm",
                                        outputs.access_ptr(
                                            'w',
                                            offset=((loop_level1 * 16 \
                                                     + loop_l1) * HW_SIZE_512 \
                                                    + loop_le1 * 4) * offset_4),
                                        f32_out.access_ptr('r'),
                                        0, 1, expand_4, 0, 0))
                        else:
                            # No.2 situation : 1. input f32 dtype : large data  3. H * W <= 16
                            with ib.for_range(0, expand_size) as i1:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 6)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float32", "copy_ubuf_to_cbuf",
                                            out_L1.access_ptr(
                                                'w',
                                                offset=i1 * f16_c0),
                                            f32_in.access_ptr('r'), 0, tail_l1,
                                            2, 0, (expand_size - 1) * 2))
                            loop_le2 = tail_l1 * expand_size * 2 // (12 * HW_SIZE_512)
                            tail_le2 = tail_l1 * expand_size * 2 % (12 * HW_SIZE_512)
                            if loop_le2 > 0:
                                with ib.for_range(0, loop_le2) as i2:
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 4)
                                        ib.emit(
                                            tvm.call_extern(
                                                "float32", "copy_cbuf_to_ubuf",
                                                f32_out.access_ptr('w'),
                                                out_L1.access_ptr(
                                                    'r',
                                                    offset=i2 * HW_SIZE_512 * 6 * f16_c0),
                                                0, 1,
                                                HW_SIZE_512 * 12, 0,
                                                0))
                                    with ib.new_scope():
                                        ib.scope_attr(param.CCE_AXIS,
                                                      "coproc_scope", 6)
                                        ib.emit(tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=(((loop_level1 * 16 \
                                                          + loop_l1) \
                                                         * expand_size) \
                                                        + i2 * 6) \
                                                       * HW_SIZE_512 * f16_c0),
                                            f32_out.access_ptr('r'),
                                            0, 1, HW_SIZE_512 * 12, 0, 0))
                            if tail_le2 > 0:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 4)
                                    ib.emit(tvm.call_extern(
                                        "float32",
                                        "copy_cbuf_to_ubuf",
                                        f32_out.access_ptr('w'),
                                        out_L1.access_ptr(
                                            'r',
                                            offset=loop_le2 * HW_SIZE_512 \
                                                   * 6 * f16_c0),
                                        0, 1, tail_le2, 0, 0))
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 6)
                                    ib.emit(tvm.call_extern(
                                        "float32", "copy_ubuf_to_gm",
                                        outputs.access_ptr(
                                            'w',
                                            offset=(((loop_level1 * 16 \
                                                      + loop_l1) \
                                                     * expand_size) \
                                                    + loop_le2 * 6) \
                                                   * HW_SIZE_512 * f16_c0),
                                        f32_out.access_ptr('r'),
                                        0, 1, tail_le2, 0, 0))
            else:
                # No.2 situation : 1. input f32 dtype : small data
                l1_half = HW_SIZE_512 * 32
                out_L1 = apply_store_buffer(ib,
                                            "float32", [l1_half * f32_c0],
                                            name="out_L1",
                                            scope=param.scope_cbuf)
                with ib.new_scope():
                    ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                    ib.emit(
                        tvm.call_extern("float32", "copy_gm_to_ubuf",
                                        f32_in.access_ptr('w'),
                                        inputs.access_ptr('r'), 0, 1,
                                        expand_loop * 2, 0, 0))
                # Note:
                #   1. H * W >= HW_SIZE_512 is common
                #   2. HW_SIZE_512 > H * W > 16
                #   3. H * W <= 16 this happen really rare

                # No.2 situation : 1. input f32 dtype : large data : 1. H * W >= HW_SIZE_512
                if expand_size > HW_SIZE_512:
                    # level_1 loop
                    loop_lev1 = 4
                    with ib.for_range(0, loop_lev1) as i1:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                            ib.emit(
                                tvm.call_extern(
                                    "float32", "copy_ubuf_to_cbuf",
                                    out_L1.access_ptr('w', offset=i1 * f16_c0),
                                    f32_in.access_ptr('r'), 0, expand_loop, 2,
                                    0, (4 - 1) * 2))
                    # level_1 loop
                    core_lp1 = expand_loop // core_counts
                    core_lp1_tail = expand_loop % core_counts

                    def _inner_run_lev1(core_lp1_idx):
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 4)
                            ib.emit(
                                tvm.call_extern(
                                    "float32", "copy_cbuf_to_ubuf",
                                    f32_8.access_ptr('w'),
                                    out_L1.access_ptr(
                                        'r',
                                        offset=block.var * 4 * f16_c0 +
                                        core_lp1_idx * core_counts * 4 * f16_c0),
                                    0, 1, 8, 0, 0))
                        repeat_128 = HW_SIZE_512 // 4
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 2)
                            ib.emit(
                                tvm.call_extern(
                                    "float32", "vadds",
                                    f32_out.access_ptr('w'),
                                    f32_8.access_ptr('r'),
                                    tvm.const(0.0, dtype="float32"),
                                    repeat_128, 1, 1, 8, 0))
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 2)
                            ib.emit(
                                tvm.call_extern(
                                    "float32", "copy_ubuf_to_ubuf",
                                    f32_out.access_ptr('w',
                                                       offset=HW_SIZE_512 * f16_c0),
                                    f32_out.access_ptr('r'), 0, 1, HW_SIZE_1024, 0, 0))
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 2)
                            ib.emit(
                                tvm.call_extern(
                                    "float32", "copy_ubuf_to_ubuf",
                                    f32_out.access_ptr('w',
                                                       offset=HW_SIZE_1024 * f16_c0),
                                    f32_out.access_ptr('r'), 0, 1, HW_SIZE_2048, 0, 0))
                        loop_lev2 = expand_size // HW_SIZE_2048
                        tail_lev2 = expand_size % HW_SIZE_2048
                        offset_2 = expand_size * f16_c0
                        if loop_lev2 > 0:
                            with ib.for_range(0, loop_lev2) as i3:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 6)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float32", "copy_ubuf_to_gm",
                                            outputs.access_ptr(
                                                'w',
                                                offset=block.var * offset_2 +
                                                i3 * HW_SIZE_2048 * f16_c0 +
                                                core_lp1_idx * core_counts *
                                                offset_2),
                                            f32_out.access_ptr('r'), 0, 1,
                                            HW_SIZE_2048 * 2, 0, 0))
                        if tail_lev2 > 0:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              6)
                                ib.emit(
                                    tvm.call_extern(
                                        "float32", "copy_ubuf_to_gm",
                                        outputs.access_ptr(
                                            'w',
                                            offset=block.var * offset_2 +
                                            loop_lev2 * HW_SIZE_2048 * f16_c0 +
                                            core_lp1_idx *
                                            core_counts * offset_2),
                                        f32_out.access_ptr('r'), 0, 1,
                                        tail_lev2 * 2, 0, 0))

                    # level_1 loop
                    with ib.for_range(0, core_lp1,
                                      name="core_lp1") as core_lp1_idx:
                        _inner_run_lev1(core_lp1_idx)
                    if core_lp1_tail > 0:
                        with ib.if_scope(block.var < core_lp1_tail):
                            _inner_run_lev1(core_lp1)

                elif expand_size > 16:
                    # No.2 situation : 1. input f32 dtype : large data : 2. HW_SIZE_512>=H*W>16
                    # level_1 loop
                    loop_lev1 = 4
                    with ib.for_range(0, loop_lev1) as i1:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                            ib.emit(
                                tvm.call_extern(
                                    "float32", "copy_ubuf_to_cbuf",
                                    out_L1.access_ptr('w', offset=i1 * f16_c0),
                                    f32_in.access_ptr('r'), 0, expand_loop, 2,
                                    0, (4 - 1) * 2))
                    # level_1 loop
                    loop_lev1 = expand_loop // 4
                    tail_lev1 = expand_loop % 4
                    if loop_lev1 > 0:
                        with ib.for_range(0, loop_lev1) as i2:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              4)
                                ib.emit(
                                    tvm.call_extern(
                                        "float32", "copy_cbuf_to_ubuf",
                                        f32_32.access_ptr('w'),
                                        out_L1.access_ptr(
                                            'r',
                                            offset=i2 * 32 * f32_c0), 0, 1,
                                        32, 0,
                                        0))
                            # level_2 loop
                            loop_lev2 = 4
                            offset_lev2 = expand_size * f16_c0
                            repeat_lev2 = expand_size // 4 + \
                                          (1 if expand_size % 4 > 0 else 0)
                            with ib.for_range(0, loop_lev2) as i3:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 2)
                                    ib.emit(
                                        tvm.call_extern(
                                            "float32", "vadds",
                                            f32_out.access_ptr(
                                                'w',
                                                offset=i3 * offset_lev2),
                                            f32_32.access_ptr(
                                                'r',
                                                offset=i3 * 4 * f16_c0),
                                            tvm.const(0.0, dtype="float32"),
                                            repeat_lev2, 1, 1, 8, 0))
                            offset_4 = expand_size * f16_c0
                            expand_4 = 4 * expand_size * 2
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              6)
                                ib.emit(
                                    tvm.call_extern(
                                        dtype, "copy_ubuf_to_gm",
                                        outputs.access_ptr(
                                            'w',
                                            offset=i2 * 4 * offset_4),
                                        f32_out.access_ptr('r'), 0, 1,
                                        expand_4, 0, 0))
                    if tail_lev1 > 0:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 4)
                            ib.emit(
                                tvm.call_extern(
                                    "float32", "copy_cbuf_to_ubuf",
                                    f32_32.access_ptr('w'),
                                    out_L1.access_ptr(
                                        'r',
                                        offset=loop_lev1 * 32 * f32_c0),
                                    0, 1,
                                    tail_lev1 * 8, 0, 0))
                        # level_2 loop
                        loop_lev2 = tail_lev1
                        offset_lev2 = expand_size * f16_c0
                        repeat_lev2 = expand_size // 4 + \
                                      (1 if expand_size % 4 > 0 else 0)
                        with ib.for_range(0, loop_lev2) as i4:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              2)
                                ib.emit(
                                    tvm.call_extern(
                                        "float32", "vadds",
                                        f32_out.access_ptr(
                                            'w',
                                            offset=i4 * offset_lev2),
                                        f32_32.access_ptr(
                                            'r',
                                            offset=i4 * 4 * f16_c0),
                                        tvm.const(0.0, dtype="float32"),
                                        repeat_lev2, 1, 1, 8, 0))
                        offset_4 = expand_size * f16_c0
                        expand_4 = tail_lev1 * expand_size * 2
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                            ib.emit(
                                tvm.call_extern(
                                    dtype, "copy_ubuf_to_gm",
                                    outputs.access_ptr(
                                        'w',
                                        offset=loop_lev1 * 4 * offset_4),
                                    f32_out.access_ptr('r'), 0, 1, expand_4, 0,
                                    0))

                else:
                    # No.2 situation : 1. input f32 dtype : large data  3. H * W <= 16
                    with ib.for_range(0, expand_size) as i1:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                            ib.emit(
                                tvm.call_extern(
                                    "float32", "copy_ubuf_to_cbuf",
                                    out_L1.access_ptr('w', offset=i1 * f16_c0),
                                    f32_in.access_ptr('r'), 0, expand_loop, 2,
                                    0, (expand_size - 1) * 2))
                    loop_lev2 = expand_loop * expand_size * 2 // (12 * HW_SIZE_512)
                    tail_lev2 = expand_loop * expand_size * 2 % (12 * HW_SIZE_512)
                    if loop_lev2 > 0:
                        with ib.for_range(0, loop_lev2) as i2:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              4)
                                ib.emit(
                                    tvm.call_extern(
                                        "float32", "copy_cbuf_to_ubuf",
                                        f32_out.access_ptr('w'),
                                        out_L1.access_ptr(
                                            'r',
                                            offset=i2 * HW_SIZE_512 * 6 * f16_c0), 0, 1,
                                        HW_SIZE_512 * 12, 0, 0))
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              6)
                                ib.emit(
                                    tvm.call_extern(
                                        "float32", "copy_ubuf_to_gm",
                                        outputs.access_ptr(
                                            'w',
                                            offset=i2 * 6 * HW_SIZE_512 * f16_c0),
                                        f32_out.access_ptr('r'), 0, 1,
                                        HW_SIZE_512 * 12, 0, 0))
                    if tail_lev2 > 0:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 4)
                            ib.emit(
                                tvm.call_extern(
                                    "float32", "copy_cbuf_to_ubuf",
                                    f32_out.access_ptr('w'),
                                    out_L1.access_ptr(
                                        'r',
                                        offset=loop_lev2 * HW_SIZE_512 * 6 * f16_c0),
                                    0, 1,
                                    tail_lev2, 0, 0))
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                            ib.emit(
                                tvm.call_extern(
                                    "float32", "copy_ubuf_to_gm",
                                    outputs.access_ptr(
                                        'w',
                                        offset=loop_lev2 * 6 * HW_SIZE_512 * f16_c0),
                                    f32_out.access_ptr('r'), 0, 1, tail_lev2,
                                    0, 0))

    # No.3 situation : output H/W = (1,1)
    elif h_out == 1 and w_out == 1 and \
            ((half_pixel_centers and h_in % 2 == 1 and w_in % 2 == 1)
             or (half_pixel_centers is False)):
        # Note:
        #   1. pick data from out directly
        #   2. large output data use L1 optimize
        # No.3 situation : output H/W = (1,1) : input f16 dtype
        if half_pixel_centers:
            in_offset = int(h_in * 0.5 - 0.5) * w_in * 16 + \
                        int(w_in * 0.5 - 0.5) * 16
        else:
            in_offset = 0

        if dtype == "float16":
            expand_loop = reduce_func(lambda x, y: x * y, size_out) // f16_c0
            gap_limit = (1 << 16) - 1
            f16_stride = w_in * h_in - 1
            l1_half = HW_SIZE_512 * 32
            f16_size = HW_SIZE_256 * 8
            reduce_size = w_in * h_in
            burst_limit = (1 << 12) - 1
            out_ub_f32 = apply_store_buffer(ib,
                                            "float32", [f16_size * f16_c0],
                                            name="out_f32")
            f16_out = apply_store_buffer(ib,
                                         "float16", [f16_size * f16_c0],
                                         name="f16_out")
            in_L1 = apply_store_buffer(ib,
                                       "float16", [l1_half * f16_c0],
                                       name="in_L1",
                                       scope=param.scope_cbuf)
            # Note:
            #   1. output data larger than HW_SIZE_512 * 8 should use L1 optimize
            #   2. output small data do not need L1

            # No.3 situation : output H/W = (1,1) : input f16 dtype : large data
            if expand_loop > f16_size:

                loop_level1 = expand_loop // l1_half
                tail_level1 = expand_loop % l1_half

                # No.3 situation : output H/W = (1,1): input f16 dtype: large data: loop_level1 > 0
                if loop_level1 > 0:
                    # level_1 loop
                    with ib.for_range(0, loop_level1) as i1:
                        # Note:
                        #   1. pick gap out of range need copy many times
                        #   2. pick gap in range need only multi bursts

                        if f16_stride > gap_limit:
                            # level_2 loop
                            loop_level2 = l1_half
                            offset_2 = l1_half * reduce_size * f16_c0
                            with ib.for_range(0, loop_level2) as i3:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 5)
                                    ib.emit(tvm.call_extern(
                                        "float16", "copy_gm_to_cbuf",
                                        in_L1.access_ptr(
                                            'w',
                                            offset=i3 * f16_c0),
                                        inputs.access_ptr(
                                            'r',
                                            offset=i1 * offset_2 \
                                                   + i3 * reduce_size * \
                                                   f16_c0 + in_offset),
                                        sid, 1, 1, 0, 0, pad_mode))
                        else:
                            # level_2 loop
                            loop_level2 = 4
                            offset_2 = l1_half * reduce_size * f16_c0
                            offset_3 = burst_limit * reduce_size * f16_c0
                            with ib.for_range(0, loop_level2) as i3:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 5)
                                    ib.emit(tvm.call_extern(
                                        "float16", "copy_gm_to_cbuf",
                                        in_L1.access_ptr(
                                            'w',
                                            offset=i3 * burst_limit * f16_c0),
                                        inputs.access_ptr(
                                            'r',
                                            offset=i1 * offset_2 \
                                                   + i3 * offset_3 + \
                                                   in_offset),
                                        sid, burst_limit, 1, f16_stride,
                                        0, pad_mode))
                            offset_4 = burst_limit * f16_c0
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              5)
                                ib.emit(tvm.call_extern(
                                    "float16", "copy_gm_to_cbuf",
                                    in_L1.access_ptr(
                                        'w',
                                        offset=loop_level2 * offset_4),
                                    inputs.access_ptr(
                                        'r',
                                        offset=i1 * offset_2 \
                                               + 4 * offset_3 + \
                                               in_offset),
                                    sid, 4, 1, f16_stride, 0, pad_mode))

                        # level_2 loop
                        loop_level2 = 4 * 2
                        with ib.for_range(0, loop_level2) as i2:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              4)
                                ib.emit(
                                    tvm.call_extern(
                                        "float16", "copy_cbuf_to_ubuf",
                                        f16_out.access_ptr('w'),
                                        in_L1.access_ptr(
                                            'r',
                                            offset=i2 * f16_size * f16_c0),
                                        0, 1,
                                        f16_size, 0, 0))

                            offset_4 = l1_half * f16_c0
                            with ib.new_scope():
                                ip_addr = [[out_ub_f32, 0], [f16_out, 0]]
                                kernel_api.kernel_cast_to_fuc(
                                    ib, ip_addr, [f16_size * 16, 8 * 8],
                                    "vconv_f162f32")
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              6)
                                ib.emit(tvm.call_extern(
                                    "float32", "copy_ubuf_to_gm",
                                    outputs.access_ptr(
                                        'w',
                                        offset=i1 * offset_4 \
                                               + i2 * f16_size * f16_c0),
                                    out_ub_f32.access_ptr('r'),
                                    0, 1, f16_size * 2, 0, 0))
                # No.3 situation : output H/W = (1,1): input f16 dtype:
                # large data: tail_level1 > 0
                if tail_level1 > 0:
                    # Note:
                    #   1. pick gap out of range need copy many times
                    #   2. pick gap in range need only one bursts

                    if f16_stride > gap_limit:
                        # level_1 loop
                        offset_1 = l1_half * reduce_size * f16_c0
                        with ib.for_range(0, tail_level1) as i1:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              5)
                                ib.emit(tvm.call_extern(
                                    "float16", "copy_gm_to_cbuf",
                                    in_L1.access_ptr(
                                        'w',
                                        offset=i1 * f16_c0),
                                    inputs.access_ptr(
                                        'r',
                                        offset=loop_level1 * offset_1 \
                                               + i1 * reduce_size * \
                                               f16_c0 + in_offset),
                                    sid, 1, 1, 0, 0, pad_mode))
                    else:
                        # level_1 loop
                        loop_l1 = tail_level1 // burst_limit
                        tail_l1 = tail_level1 % burst_limit
                        offset_2 = l1_half * reduce_size * f16_c0
                        offset_3 = burst_limit * reduce_size * f16_c0
                        if loop_l1 > 0:
                            with ib.for_range(0, loop_l1) as i1:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 5)
                                    ib.emit(tvm.call_extern(
                                        "float16", "copy_gm_to_cbuf",
                                        in_L1.access_ptr(
                                            'w',
                                            offset=i1 * burst_limit * f16_c0),
                                        inputs.access_ptr(
                                            'r',
                                            offset=loop_level1 * offset_2 \
                                                   + i1 * offset_3 + \
                                                   in_offset),
                                        sid, burst_limit, 1, f16_stride,
                                        0, pad_mode))
                        offset_4 = burst_limit * f16_c0
                        if tail_l1 > 0:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              5)
                                ib.emit(tvm.call_extern(
                                    "float16", "copy_gm_to_cbuf",
                                    in_L1.access_ptr(
                                        'w',
                                        offset=loop_l1 * offset_4),
                                    inputs.access_ptr(
                                        'r',
                                        offset=loop_level1 * offset_2 \
                                               + loop_l1 * offset_3 + \
                                               in_offset),
                                    sid, tail_l1, 1, f16_stride, 0, pad_mode))
                    # level_1 loop
                    loop_l1 = tail_level1 // f16_size
                    tail_l1 = tail_level1 % f16_size
                    # No.3 situation : output H/W=(1,1): input f16 dtype:
                    # tail_level1>0: loop_l1>0
                    if loop_l1 > 0:
                        with ib.for_range(0, loop_l1) as i2:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              4)
                                ib.emit(
                                    tvm.call_extern(
                                        "float16", "copy_cbuf_to_ubuf",
                                        f16_out.access_ptr('w'),
                                        in_L1.access_ptr(
                                            'r',
                                            offset=i2 * f16_size * f16_c0),
                                        0, 1,
                                        f16_size, 0, 0))
                            offset_3 = l1_half * f16_c0
                            with ib.new_scope():
                                ip_addr = [[out_ub_f32, 0], [f16_out, 0]]
                                kernel_api.kernel_cast_to_fuc(
                                    ib, ip_addr, [f16_size * 16, 8 * 8],
                                    "vconv_f162f32")
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              6)
                                ib.emit(tvm.call_extern(
                                    "float32", "copy_ubuf_to_gm",
                                    outputs.access_ptr(
                                        'w',
                                        offset=loop_level1 * offset_3 \
                                               + i2 * f16_size * f16_c0),
                                    out_ub_f32.access_ptr('r'),
                                    0, 1, f16_size * 2, 0, 0))

                    # No.3 situation : output H/W=(1,1): input f16 dtype: tail_level1>0: tail_l1>0
                    if tail_l1 > 0:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 4)
                            ib.emit(
                                tvm.call_extern(
                                    "float16", "copy_cbuf_to_ubuf",
                                    f16_out.access_ptr('w'),
                                    in_L1.access_ptr(
                                        'r',
                                        offset=loop_l1 * f16_size * f16_c0),
                                    0, 1,
                                    tail_l1, 0, 0))
                        offset_4 = l1_half * f16_c0
                        with ib.new_scope():
                            ip_addr = [[out_ub_f32, 0], [f16_out, 0]]
                            kernel_api.kernel_cast_to_fuc(
                                ib, ip_addr, [tail_l1 * 16, 8 * 8],
                                "vconv_f162f32")
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                            ib.emit(tvm.call_extern(
                                "float32", "copy_ubuf_to_gm",
                                outputs.access_ptr(
                                    'w',
                                    offset=loop_level1 * offset_4 \
                                           + loop_l1 * f16_size * f16_c0),
                                out_ub_f32.access_ptr('r'),
                                0, 1, tail_l1 * 2, 0, 0))
            else:
                # No.3 situation : output H/W = (1,1) : input f16 dtype : small data

                # Note:
                #   1. pick gap out of range need copy many times
                #   2. pick gap in range need only one bursts

                if f16_stride > gap_limit:
                    # level_1 loop
                    with ib.for_range(0, expand_loop) as i1:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                            ib.emit(
                                tvm.call_extern(
                                    "float16", "copy_gm_to_ubuf",
                                    f16_out.access_ptr('w',
                                                       offset=i1 * f16_c0),
                                    inputs.access_ptr(
                                        'r',
                                        offset=i1 * reduce_size * \
                                               f16_c0 + in_offset),
                                    0, 1, 1,
                                    0, 0))
                else:
                    with ib.new_scope():
                        ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                        ib.emit(
                            tvm.call_extern("float16", "copy_gm_to_ubuf",
                                            f16_out.access_ptr('w'),
                                            inputs.access_ptr(
                                                'r',
                                                offset=in_offset),
                                            0, expand_loop, 1, f16_stride, 0))
                with ib.new_scope():
                    ip_addr = [[out_ub_f32, 0], [f16_out, 0]]
                    kernel_api.kernel_cast_to_fuc(ib, ip_addr,
                                                  [expand_loop * 16, 8 * 8],
                                                  "vconv_f162f32")
                    ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                    ib.emit(
                        tvm.call_extern("float32", "copy_ubuf_to_gm",
                                        outputs.access_ptr('w'),
                                        out_ub_f32.access_ptr('r'), 0, 1,
                                        expand_loop * 2, 0, 0))

        else:
            # No.3 situation : output H/W = (1,1) : input f32 dtype
            expand_loop = reduce_func(lambda x, y: x * y, size_out) // f16_c0
            gap_limit = (1 << 16) - 1
            f32_stride = (w_in * h_in - 1) * 2
            l1_half = HW_SIZE_512 * 32
            f32_size = HW_SIZE_512 * 8
            reduce_size = w_in * h_in
            burst_limit = (1 << 12) - 1

            f32_out = apply_store_buffer(ib,
                                         "float32", [f32_size * f32_c0],
                                         name="f32_out")
            in_L1 = apply_store_buffer(ib,
                                       "float32", [l1_half * f16_c0],
                                       name="in_L1",
                                       scope=param.scope_cbuf)
            f32_in = apply_store_buffer(ib,
                                        "float32", [HW_SIZE_1024 * f32_c0],
                                        name="f32_in")
            # Note:
            #   1. output data larger than HW_SIZE_512 * 8 should use L1 optimize
            #   2. output small data do not need L1

            # No.3 situation : output H/W = (1,1) : input f32 dtype : large data
            if expand_loop > f32_size:

                f32_half = l1_half // 2
                loop_level1 = expand_loop // f32_half
                tail_level1 = expand_loop % f32_half

                # No.3 situation : output H/W = (1,1): input f32 dtype:
                # large data: loop_level1 > 0
                if loop_level1 > 0:
                    # level_1 loop
                    with ib.for_range(0, loop_level1) as i1:
                        # Note:
                        #   1. pick gap out of range need copy many times
                        #   2. pick gap in range need only one bursts

                        if f32_stride > gap_limit:
                            # level_2 loop
                            loop_level2 = f32_half
                            offset_2 = f32_half * reduce_size * f16_c0
                            with ib.for_range(0, loop_level2) as i3:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 5)
                                    ib.emit(tvm.call_extern(
                                        "float32", "copy_gm_to_cbuf",
                                        in_L1.access_ptr(
                                            'w',
                                            offset=i3 * f16_c0),
                                        inputs.access_ptr(
                                            'r',
                                            offset=i1 * offset_2 \
                                                   + i3 * reduce_size * \
                                                   f16_c0 + in_offset),
                                        sid, 1, 2, 0, 0, pad_mode))
                        else:
                            # level_2 loop
                            loop_level2 = 2
                            offset_2 = f32_half * reduce_size * f16_c0
                            offset_3 = burst_limit * reduce_size * f16_c0
                            with ib.for_range(0, loop_level2) as i3:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 5)
                                    ib.emit(tvm.call_extern(
                                        "float32", "copy_gm_to_cbuf",
                                        in_L1.access_ptr(
                                            'w',
                                            offset=i3 * burst_limit * f16_c0),
                                        inputs.access_ptr(
                                            'r',
                                            offset=i1 * offset_2 \
                                                   + i3 * offset_3 \
                                                   + in_offset),
                                        sid, burst_limit, 2, f32_stride,
                                        0, pad_mode))
                            offset_4 = burst_limit * f16_c0
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              5)
                                ib.emit(tvm.call_extern(
                                    "float32", "copy_gm_to_cbuf",
                                    in_L1.access_ptr(
                                        'w',
                                        offset=loop_level2 * offset_4),
                                    inputs.access_ptr(
                                        'r',
                                        offset=i1 * offset_2 \
                                               + 2 * offset_3 + in_offset),
                                    sid, 2, 2, f32_stride, 0, pad_mode))

                        # level_2 loop
                        loop_level2 = 4
                        with ib.for_range(0, loop_level2) as i2:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              4)
                                ib.emit(
                                    tvm.call_extern(
                                        "float32", "copy_cbuf_to_ubuf",
                                        f32_in.access_ptr('w'),
                                        in_L1.access_ptr(
                                            'r',
                                            offset=i2 * f32_size * f32_c0),
                                        0, 1,
                                        f32_size, 0, 0))
                            offset_4 = f32_half * f16_c0
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              6)
                                ib.emit(tvm.call_extern(
                                    "float32", "copy_ubuf_to_gm",
                                    outputs.access_ptr(
                                        'w',
                                        offset=i1 * offset_4 \
                                               + i2 * f32_size * f32_c0),
                                    f32_in.access_ptr('r'),
                                    0, 1, f32_size, 0, 0))
                # No.3 situation : output H/W = (1,1): input f32 dtype:
                # large data: tail_level1 > 0
                if tail_level1 > 0:
                    # Note:
                    #   1. pick gap out of range need copy many times
                    #   2. pick gap in range need only one bursts

                    if f32_stride > gap_limit:
                        # level_1 loop
                        offset_1 = f32_half * reduce_size * f16_c0
                        with ib.for_range(0, tail_level1) as i1:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              5)
                                ib.emit(tvm.call_extern(
                                    "float32", "copy_gm_to_cbuf",
                                    in_L1.access_ptr(
                                        'w',
                                        offset=i1 * f16_c0),
                                    inputs.access_ptr(
                                        'r',
                                        offset=loop_level1 * offset_1 \
                                               + i1 * reduce_size * \
                                               f16_c0 + in_offset),
                                    sid, 1, 2, 0, 0, pad_mode))
                    else:
                        # level_1 loop
                        loop_l1 = tail_level1 // burst_limit
                        tail_l1 = tail_level1 % burst_limit
                        offset_2 = f32_half * reduce_size * f16_c0
                        offset_3 = burst_limit * reduce_size * f16_c0
                        if loop_l1 > 0:
                            with ib.for_range(0, loop_l1) as i1:
                                with ib.new_scope():
                                    ib.scope_attr(param.CCE_AXIS,
                                                  "coproc_scope", 5)
                                    ib.emit(tvm.call_extern(
                                        "float32", "copy_gm_to_cbuf",
                                        in_L1.access_ptr(
                                            'w',
                                            offset=i1 * burst_limit * f16_c0),
                                        inputs.access_ptr(
                                            'r',
                                            offset=loop_level1 * offset_2 \
                                                   + i1 * offset_3 + \
                                                   in_offset),
                                        sid, burst_limit, 2, f32_stride,
                                        0, pad_mode))
                        if tail_l1 > 0:
                            offset_4 = burst_limit * f16_c0
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              5)
                                ib.emit(tvm.call_extern(
                                    "float32", "copy_gm_to_cbuf",
                                    in_L1.access_ptr(
                                        'w',
                                        offset=loop_l1 * offset_4),
                                    inputs.access_ptr(
                                        'r',
                                        offset=loop_level1 * offset_2 \
                                               + loop_l1 * offset_3 + \
                                               in_offset),
                                    sid, tail_l1, 2, f32_stride, 0, pad_mode))
                    # level_1 loop
                    loop_l1 = tail_level1 * 2 // f32_size
                    tail_l1 = tail_level1 * 2 % f32_size
                    # No.3 situation : output H/W=(1,1): input f32 dtype: tail_level1>0: loop_l1>0
                    if loop_l1 > 0:
                        ib.emit(
                            tvm.call_extern(
                                "uint64", "set_vector_mask",
                                tvm.const(0, dtype="uint64"),
                                tvm.const((1 << 64) - 1, dtype="uint64")))
                        with ib.for_range(0, loop_l1) as i2:
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              4)
                                ib.emit(
                                    tvm.call_extern(
                                        "float32", "copy_cbuf_to_ubuf",
                                        f32_out.access_ptr('w'),
                                        in_L1.access_ptr(
                                            'r',
                                            offset=i2 * f32_size * f32_c0),
                                        0, 1, f32_size, 0, 0))
                            offset_3 = f32_half * f16_c0
                            with ib.new_scope():
                                ib.scope_attr(param.CCE_AXIS, "coproc_scope",
                                              6)
                                ib.emit(tvm.call_extern(
                                    "float32", "copy_ubuf_to_gm",
                                    outputs.access_ptr(
                                        'w',
                                        offset=loop_level1 * offset_3 \
                                               + i2 * f32_size * f32_c0),
                                    f32_out.access_ptr('r'),
                                    0, 1, f32_size, 0, 0))

                    # No.3 situation : output H/W=(1,1): input f32 dtype: tail_level1>0: tail_l1>0
                    if tail_l1 > 0:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 4)
                            ib.emit(
                                tvm.call_extern(
                                    "float32", "copy_cbuf_to_ubuf",
                                    f32_out.access_ptr('w'),
                                    in_L1.access_ptr(
                                        'r',
                                        offset=loop_l1 * f32_size * f32_c0),
                                    0,
                                    1, tail_l1, 0, 0))
                        offset_4 = f32_half * f16_c0
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                            ib.emit(tvm.call_extern(
                                "float32", "copy_ubuf_to_gm",
                                outputs.access_ptr(
                                    'w',
                                    offset=loop_level1 * offset_4 \
                                           + loop_l1 * f32_size * f32_c0),
                                f32_out.access_ptr('r'),
                                0, 1, tail_l1, 0, 0))
            else:
                # No.3 situation : output H/W = (1,1) :
                # input f32 dtype : small data

                # Note:
                #   1. pick gap out of range need copy many times
                #   2. pick gap in range need only one bursts

                if f32_stride > gap_limit:
                    # level_1 loop
                    with ib.for_range(0, expand_loop) as i1:
                        with ib.new_scope():
                            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                            ib.emit(
                                tvm.call_extern(
                                    "float32", "copy_gm_to_ubuf",
                                    f32_out.access_ptr('w',
                                                       offset=i1 * f16_c0),
                                    inputs.access_ptr(
                                        'r',
                                        offset=i1 * reduce_size * \
                                               f16_c0 + in_offset),
                                    0, 1, 2,
                                    0, 0))
                else:
                    with ib.new_scope():
                        ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                        ib.emit(
                            tvm.call_extern("float32", "copy_gm_to_ubuf",
                                            f32_out.access_ptr('w'),
                                            inputs.access_ptr(
                                                'r', offset=in_offset),
                                            0, expand_loop, 2, f32_stride, 0))
                with ib.new_scope():
                    ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                    ib.emit(
                        tvm.call_extern("float32", "copy_ubuf_to_gm",
                                        outputs.access_ptr('w'),
                                        f32_out.access_ptr('r'), 0, 1,
                                        expand_loop * 2, 0, 0))

    # No.4 situation: for mini_1951
    elif tbe_platform.cce_conf.intrinsic_check_support(
            "Intrinsic_vbi", "float16") and dtype == "float16":
        # use multi-core resource
        ub_size = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.UB_SIZE)
        l1_size = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.L1_SIZE)
        h_per_core = h_out // core_counts + (1
                                             if h_out % core_counts > 0 else 0)
        is_same_percore = 0 if h_out % core_counts == 0 else 1
        core_num = h_out // h_per_core + is_same_percore
        ib.scope_attr(block, "thread_extent", core_num)
        # use multi-core resource
        core_idx = block.var
        ub_info = {}
        l1_info = {}
        vbi_ub_size = (w_out + VBI_BLOCK_COUNT - 1) // VBI_BLOCK_COUNT * \
                      VBI_BLOCK_COUNT
        vbi_ub_size_h = (h_out + VBI_BLOCK_COUNT - 1) // VBI_BLOCK_COUNT * \
                      VBI_BLOCK_COUNT
        alloc_ub_size = (vbi_ub_size * 12 + VBI_BLOCK_COUNT) * 4 + \
                        (vbi_ub_size_h * 3) * 4 + \
                        (vbi_ub_size * 4) * 2 + \
                        ((w_in + 1) * f16_c0 * 2) * 2 + \
                        (w_out * f16_c0) * 6 + \
                        (f32_c0 * 8) * 4
        # check if needed ub size larger than total ub size
        is_bigger_ub_size = False
        wo_loop_count = 0
        wo_loop_left = 0
        wo_threshold = 640
        if alloc_ub_size > ub_size:
            is_bigger_ub_size = True
            wo_loop_count = w_out // wo_threshold
            wo_loop_left = w_out % wo_threshold
        # check if needed l1 size smaller than total l1 size
        is_smaller_l1_size = False
        hw_scalar_l1_size = vbi_ub_size * 4 * h_out
        if (hw_scalar_l1_size * 2 <= l1_size and not is_bigger_ub_size) or \
            (hw_scalar_l1_size * 2 + vbi_ub_size * 4 * 2 * 4 <= l1_size and
             is_bigger_ub_size):
            is_smaller_l1_size = True

        def _apply_ub_1951():
            """
            apply ubuf for 1951

            """
            if not is_bigger_ub_size:
                # apply ub for w_out position, size is ceil(w_out // 8) * 8 * 4
                ub_info["float32_temp_wo_ub"] = apply_store_buffer(
                    ib, "float32",
                    [vbi_ub_size * 4],
                    name="float32_temp_wo_ub")
                # apply ub for w_out position, size is ceil(w_out // 8) * 8 * 4
                ub_info["float32_w_pos_ub"] = apply_store_buffer(
                    ib, "float32",
                    [vbi_ub_size * 4],
                    name="float32_w_pos_ub")
                # apply ub for w_out position, size is ceil(w_out // 8) * 8 * 4
                ub_info["int32_w_pos_ub"] = apply_store_buffer(
                    ib, "int32",
                    [vbi_ub_size * 4],
                    name="int32_w_pos_ub")
                # apply ub for w scalar, size is ceil(w_out // 8) * 8 * 4
                ub_info["float16_w_scalar_ub"] = apply_store_buffer(
                    ib, "float16",
                    [vbi_ub_size * 4],
                    name="float16_w_scalar_ub")
                # apply ub for output line, size is w_out * c0
                ub_info["out_ub"] = apply_store_buffer(
                    ib, "float16",
                    [w_out * f16_c0],
                    name="out_ub")
                # apply ub for output line, size is w_out * c0
                ub_info["float32_out_ub"] = apply_store_buffer(
                    ib, "float32",
                    [w_out * f16_c0],
                    name="float32_out_ub")
            else:
                # apply ub for w_out position, size is wo_threshold * 4
                ub_info["float32_temp_wo_ub"] = apply_store_buffer(
                    ib, "float32",
                    [wo_threshold * 4],
                    name="float32_temp_wo_ub")
                # apply ub for w_out position, size is vbi_ub_size * 4
                l1_info["float32_temp_wo_l1"] = apply_store_buffer(
                    ib, "float32",
                    [vbi_ub_size * 4],
                    name="float32_temp_wo_l1", scope=param.scope_cbuf)
                # apply ub for w_out position, size is wo_threshold * 4
                ub_info["float32_w_pos_ub"] = apply_store_buffer(
                    ib, "float32",
                    [wo_threshold * 4],
                    name="float32_w_pos_ub")
                # apply ub for w_out position, size is wo_threshold * 4
                ub_info["int32_w_pos_ub"] = apply_store_buffer(
                    ib, "int32",
                    [wo_threshold * 4],
                    name="int32_w_pos_ub")
                # apply ub for w_out position, size is vbi_ub_size * 4
                l1_info["int32_w_pos_l1"] = apply_store_buffer(
                    ib, "int32",
                    [vbi_ub_size * 4],
                    name="int32_w_pos_l1", scope=param.scope_cbuf)
                # apply ub for w scalar, size is wo_threshold * 4
                ub_info["float16_w_scalar_ub"] = apply_store_buffer(
                    ib, "float16",
                    [wo_threshold * 4],
                    name="float16_w_scalar_ub")
                # apply ub for output line, size is wo_threshold * c0
                ub_info["out_ub"] = apply_store_buffer(
                    ib, "float16",
                    [wo_threshold * f16_c0],
                    name="out_ub")
                # apply ub for output line, size is wo_threshold * c0
                ub_info["float32_out_ub"] = apply_store_buffer(
                    ib, "float32",
                    [wo_threshold * f16_c0],
                    name="float32_out_ub")

            # apply ub for const 0, size is 16
            ub_info["float32_const_0"] = apply_store_buffer(
                ib, "float32",
                [VBI_BLOCK_COUNT],
                name="float32_const_0")

            # apply ub for h_out position, size is h_out
            ub_info["int32_h_pos_ub"] = apply_store_buffer(
                ib, "int32",
                [vbi_ub_size_h],
                name="int32_h_pos_ub")

            # apply ub for w_out position, size is h_out * 2
            ub_info["float32_h_scalar_ub"] = apply_store_buffer(
                ib, "float32",
                [vbi_ub_size_h * 2],
                name="float32_h_scalar_ub")

            # apply ub for top_left_right line, size is (w_in + 1) * f16_c0 * 2
            ub_info["w0_w1_in_ub"] = apply_store_buffer(
                ib, "float16",
                [(w_in + 1) * f16_c0 * 2],
                name="w0_w1_in_ub")

            # apply ub for half_pixel_centers, size is 64
            if half_pixel_centers:
                ub_info["float32_const_1"] = apply_store_buffer(
                    ib, "float32",
                    [f32_c0 * 8],
                    name="float32_const_1")

            # apply l1 for hw scalar, size is vbi_ub_size *4 * h_out
            if is_smaller_l1_size:
                l1_info["float16_hw_scalar_l1"] = apply_store_buffer(
                    ib, "float16",
                    [hw_scalar_l1_size],
                    name="float16_hw_scalar_l1",
                    scope=param.scope_cbuf)

        _apply_ub_1951()
        # apply reg for using to get pos
        pos_reg = ib.allocate("int32", (8,),
                              name="pos_reg",
                              scope=param.scope_reg)
        pos_reg_float32 = ib.allocate("float32", (2,),
                                      name="pos_reg_float32",
                                      scope=param.scope_reg)

        def _vector_dup_fuc(ub_var, data_len, dtype, value):
            if dtype == "float32":
                loop_cnt = data_len // 64
                left_cnt = data_len % 64
                repeat_len = 64
            else:
                loop_cnt = data_len // 128
                left_cnt = data_len % 128
                repeat_len = 128
            if loop_cnt > 0:
                ib.emit(
                    tvm.call_extern(dtype, "vector_dup",
                                    ub_var.access_ptr('w'),
                                    tvm.const(value, dtype=dtype),
                                    loop_cnt, 1, 0, 8, 0))
            if left_cnt > 0:
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const((1 << left_cnt) - 1,
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern(dtype, "vector_dup",
                                    ub_var.access_ptr(
                                        'w',
                                        offset=loop_cnt * repeat_len),
                                    tvm.const(1.0, dtype=dtype),
                                    1, 1, 0, 8, 0))
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(-1, dtype="uint64"),
                                    tvm.const(-1, dtype="uint64")))

        _vector_dup_fuc(ub_info["float32_const_0"], f32_c0, "float32", 1.0)
        if half_pixel_centers:
            _vector_dup_fuc(ub_info["float32_const_1"],
                            f32_c0 * 8, "float32", 0.0)

        def _gen_seq_data():
            """
            gen (0-7) seq num

            """
            # fill 0-7 to one block
            with ib.for_range(0, VBI_BLOCK_COUNT) as reg_idx:
                ib.emit(
                    tvm.call_extern(
                        "int64", "reg_mov",
                        tvm.call_extern("int64",
                                        "reg",
                                        pos_reg[reg_idx]),
                        reg_idx))
                ib.emit(
                    tvm.call_extern(
                        "int32", "reg_mov",
                        ub_info["int32_w_pos_ub"].access_ptr(
                            "w", offset=reg_idx),
                        tvm.call_extern("int32",
                                        "reg",
                                        pos_reg[reg_idx])))

        def _gen_w_seq_data():
            """
            gen (0-7) * 4 seq num

            """
            _gen_seq_data()

            ib.emit(
                tvm.call_extern("uint64", "set_vector_mask",
                                tvm.const(-1, dtype="uint64"),
                                tvm.const(-1, dtype="uint64")))

            ib.emit(tvm.call_extern("int32", "copy_ubuf_to_ubuf",
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        "w", offset=f32_c0),
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        "r", offset=0),
                                    0, 1, 1, 0, 0))
            ib.emit(tvm.call_extern("int32", "copy_ubuf_to_ubuf",
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        "w", offset=f32_c0 * 2),
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        "r", offset=0),
                                    0, 1, 2, 0, 0))

        def _calc_h_out_pos_scalar():
            """
            get w_out pos, the order is:
            top_left, bottom_left, top_right, bottom_right

            """
            _gen_seq_data()
            # cast to int32 to get top_left pos
            _addr_info = [[ub_info["float32_h_scalar_ub"], 0],
                          [ub_info["int32_w_pos_ub"], 0]]
            _data_info = [VBI_BLOCK_COUNT, f32_c0 * 8]
            kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info,
                                          "vconv_s322f32")

            def _inner_h_copy_data(vbi_cnt, data_len, add_value, offset):
                """
                copy data for h_out

                """
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const((1 << data_len) - 1,
                                              dtype="uint64")))
                with ib.for_range(0, vbi_cnt - 1) as idx:
                    ib.emit(
                        tvm.call_extern(
                            "float32", "vadds",
                            ub_info["float32_h_scalar_ub"].access_ptr(
                                'w',
                                offset=data_len * (idx + 1) + offset),
                            ub_info["float32_h_scalar_ub"].access_ptr(
                                'r',
                                offset=data_len * idx + offset),
                            tvm.const(add_value, dtype="float32"),
                            1, 1, 1, 8, 8))

                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(-1, dtype="uint64"),
                                    tvm.const(-1, dtype="uint64")))

            h_out_vbi_count = (h_out + VBI_BLOCK_COUNT - 1) // \
                              VBI_BLOCK_COUNT
            if h_out_vbi_count < VBI_BLOCK_COUNT * 2:
                _inner_h_copy_data(h_out_vbi_count, f32_c0, f32_c0, 0)

            else:
                loop_cnt = h_out_vbi_count // VBI_BLOCK_COUNT
                loop_left = h_out_vbi_count % VBI_BLOCK_COUNT

                _inner_h_copy_data(VBI_BLOCK_COUNT, f32_c0, f32_c0, 0)
                _inner_h_copy_data(loop_cnt, f32_c0 * 8, f32_c0 * 8, 0)

                if loop_left > 0:
                    _inner_h_copy_data(loop_left + 1, f32_c0, f32_c0,
                                       loop_cnt * f32_c0 * 8 - f32_c0)

            # muls scalar_w to get top_left pos and scalar
            loop_cnt_64 = h_out // 64
            loop_left_64 = h_out % 64
            if loop_cnt_64 > 0:
                ib.emit(
                    tvm.call_extern("float32", "vadds",
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'w'),
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'r'),
                                    core_idx * tvm.const(h_per_core,
                                                         dtype="float32"),
                                    loop_cnt_64, 1, 1, 8, 8))
                if half_pixel_centers:
                    ib.emit(
                        tvm.call_extern(
                            "float32", "vadds",
                            ub_info["float32_h_scalar_ub"].access_ptr('w'),
                            ub_info["float32_h_scalar_ub"].access_ptr('r'),
                            tvm.const(0.5, dtype="float32"),
                            loop_cnt_64, 1, 1, 8, 8))
                ib.emit(
                    tvm.call_extern("float32", "vmuls",
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'w'),
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'r'),
                                    tvm.const(scale_h, dtype="float32"),
                                    loop_cnt_64, 1, 1, 8, 8))
                if half_pixel_centers:
                    ib.emit(
                        tvm.call_extern(
                            "float32", "vadds",
                            ub_info["float32_h_scalar_ub"].access_ptr('w'),
                            ub_info["float32_h_scalar_ub"].access_ptr('r'),
                            tvm.const(-0.5, dtype="float32"),
                            loop_cnt_64, 1, 1, 8, 8))
                    ib.emit(
                        tvm.call_extern(
                            "float32", "vmax",
                            ub_info["float32_h_scalar_ub"].access_ptr('w'),
                            ub_info["float32_h_scalar_ub"].access_ptr('r'),
                            ub_info["float32_const_1"].access_ptr('r'),
                            loop_cnt_64, 1, 1, 1, 8, 8, 0))
            if loop_left_64 > 0:
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const((1 << loop_left_64) - 1,
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern("float32", "vadds",
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'w', offset=loop_cnt_64 * 64),
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'r', offset=loop_cnt_64 * 64),
                                    core_idx * tvm.const(h_per_core,
                                                         dtype="float32"),
                                    1, 1, 1, 8, 8))
                if half_pixel_centers:
                    ib.emit(
                        tvm.call_extern(
                            "float32", "vadds",
                            ub_info["float32_h_scalar_ub"].access_ptr(
                                'w', offset=loop_cnt_64 * 64),
                            ub_info["float32_h_scalar_ub"].access_ptr(
                                'r', offset=loop_cnt_64 * 64),
                            tvm.const(0.5, dtype="float32"),
                            1, 1, 1, 8, 8))
                ib.emit(
                    tvm.call_extern("float32", "vmuls",
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'w', offset=loop_cnt_64 * 64),
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'r', offset=loop_cnt_64 * 64),
                                    tvm.const(scale_h, dtype="float32"),
                                    1, 1, 1, 8, 8))
                if half_pixel_centers:
                    ib.emit(
                        tvm.call_extern(
                            "float32", "vadds",
                            ub_info["float32_h_scalar_ub"].access_ptr(
                                'w', offset=loop_cnt_64 * 64),
                            ub_info["float32_h_scalar_ub"].access_ptr(
                                'r', offset=loop_cnt_64 * 64),
                            tvm.const(-0.5, dtype="float32"),
                            1, 1, 1, 8, 8))
                    ib.emit(
                        tvm.call_extern(
                            "float32", "vmax",
                            ub_info["float32_h_scalar_ub"].access_ptr(
                                'w', offset=loop_cnt_64 * 64),
                            ub_info["float32_h_scalar_ub"].access_ptr(
                                'r', offset=loop_cnt_64 * 64),
                            ub_info["float32_const_1"].access_ptr('r'),
                            1, 1, 1, 1, 8, 8, 0))
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(-1, dtype="uint64"),
                                    tvm.const(-1, dtype="uint64")))

            # cast to int32 to get h_out pos
            _addr_info = [[ub_info["int32_h_pos_ub"], 0],
                          [ub_info["float32_h_scalar_ub"], 0]]
            _data_info = [h_out, f32_c0 * 8]
            kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info,
                                          "vconv_f322s32f")
            # to get the h_out scalar, the order is scalar, 1-scalar
            _addr_info = [[ub_info["float32_h_scalar_ub"], vbi_ub_size_h],
                          [ub_info["int32_h_pos_ub"], 0]]
            _data_info = [h_out, f32_c0 * 8]
            kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info,
                                          "vconv_s322f32")

            if loop_cnt_64 > 0:
                ib.emit(
                    tvm.call_extern("float32", "vsub",
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'w'),
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'r'),
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'r', offset=vbi_ub_size_h),
                                    loop_cnt_64, 1, 1, 1, 8, 8, 8))
                ib.emit(
                    tvm.call_extern("float32", "vsub",
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'w', offset=vbi_ub_size_h),
                                    ub_info["float32_const_0"].access_ptr(
                                        'r'),
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'r', offset=0),
                                    loop_cnt_64, 1, 0, 1, 8, 0, 8))
            if loop_left_64 > 0:
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const((1 << loop_left_64) - 1,
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern("float32", "vsub",
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'w', offset=loop_cnt_64 * 64),
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'r', offset=loop_cnt_64 * 64),
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'r',
                                        offset=vbi_ub_size_h +
                                        loop_cnt_64 * 64),
                                    1, 1, 1, 1, 8, 8, 8))
                ib.emit(
                    tvm.call_extern("float32", "vsub",
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'w',
                                        offset=vbi_ub_size_h +
                                        loop_cnt_64 * 64),
                                    ub_info["float32_const_0"].access_ptr(
                                        'r'),
                                    ub_info["float32_h_scalar_ub"].access_ptr(
                                        'r', offset=loop_cnt_64 * 64),
                                    1, 1, 0, 1, 8, 0, 8))
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(-1, dtype="uint64"),
                                    tvm.const(-1, dtype="uint64")))

        def _calc_w_out_pos(wo_len, prev_wo_cnt):
            """
            get w_out pos, the order is:
            top_left, bottom_left, top_right, bottom_right

            """
            _gen_w_seq_data()
            # add the processed w_out length
            with ib.if_scope(prev_wo_cnt > 0):
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const((1 << 32) - 1,
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern(
                        "int32", "vadds",
                        ub_info["int32_w_pos_ub"].access_ptr('w'),
                        ub_info["int32_w_pos_ub"].access_ptr('r'),
                        prev_wo_cnt,
                        1, 1, 1, 8, 8))
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(-1, dtype="uint64"),
                                    tvm.const(-1, dtype="uint64")))
            # cast to int32 to float32 pos
            _addr_info = [[ub_info["float32_temp_wo_ub"], 0],
                          [ub_info["int32_w_pos_ub"], 0]]
            _data_info = [VBI_BLOCK_COUNT * 4, f32_c0 * 8]
            kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info,
                                          "vconv_s322f32")
            # get the vbi repeat count
            w_out_vbi_count = (wo_len + VBI_BLOCK_COUNT - 1) // VBI_BLOCK_COUNT

            def _inner_copy_data(vbi_cnt, data_len, add_value, offset):
                """
                copy data for w_out

                """
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const((1 << data_len) - 1,
                                              dtype="uint64")))
                with ib.for_range(0, vbi_cnt - 1) as idx:
                    ib.emit(
                        tvm.call_extern(
                            "float32", "vadds",
                            ub_info["float32_temp_wo_ub"].access_ptr(
                                'w',
                                offset=data_len * (idx + 1) + offset),
                            ub_info["float32_temp_wo_ub"].access_ptr(
                                'r',
                                offset=data_len * idx + offset),
                            tvm.const(add_value, dtype="float32"),
                            1, 1, 1, 8, 8))

                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(-1, dtype="uint64"),
                                    tvm.const(-1, dtype="uint64")))
            if w_out_vbi_count < 4:
                _inner_copy_data(w_out_vbi_count, f32_c0 * 4, f32_c0, 0)
            else:
                loop_cnt = w_out_vbi_count // 2
                loop_left = w_out_vbi_count % 2
                _inner_copy_data(2, f32_c0 * 4, f32_c0, 0)
                _inner_copy_data(loop_cnt, f32_c0 * 8, f32_c0 * 2, 0)

                if loop_left > 0:
                    _inner_copy_data(2, f32_c0 * 4, f32_c0,
                                     loop_cnt * f32_c0 * 8 - f32_c0 * 4)

            # muls scalar_w to get top_left pos and scalar
            loop_cnt_64 = w_out_vbi_count * f32_c0 * 4 // 64
            loop_left_64 = w_out_vbi_count * f32_c0 * 4 % 64
            if loop_cnt_64 > 0:
                if half_pixel_centers:
                    ib.emit(
                        tvm.call_extern(
                            "float32", "vadds",
                            ub_info["float32_temp_wo_ub"].access_ptr('w'),
                            ub_info["float32_temp_wo_ub"].access_ptr('r'),
                            tvm.const(0.5, dtype="float32"),
                            loop_cnt_64, 1, 1, 8, 8))
                ib.emit(
                    tvm.call_extern(
                        "float32", "vmuls",
                        ub_info["float32_temp_wo_ub"].access_ptr('w'),
                        ub_info["float32_temp_wo_ub"].access_ptr('r'),
                        tvm.const(scale_w, dtype="float32"),
                        loop_cnt_64, 1, 1, 8, 8))
                if half_pixel_centers:
                    ib.emit(
                        tvm.call_extern(
                            "float32", "vadds",
                            ub_info["float32_temp_wo_ub"].access_ptr('w'),
                            ub_info["float32_temp_wo_ub"].access_ptr('r'),
                            tvm.const(-0.5, dtype="float32"),
                            loop_cnt_64, 1, 1, 8, 8))
                    ib.emit(
                        tvm.call_extern(
                            "float32", "vmax",
                            ub_info["float32_temp_wo_ub"].access_ptr('w'),
                            ub_info["float32_temp_wo_ub"].access_ptr('r'),
                            ub_info["float32_const_1"].access_ptr('r'),
                            loop_cnt_64, 1, 1, 1, 8, 8, 0))
            if loop_left_64 > 0:
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const((1 << loop_left_64) - 1,
                                              dtype="uint64")))
                if half_pixel_centers:
                    ib.emit(
                        tvm.call_extern(
                            "float32", "vadds",
                            ub_info["float32_temp_wo_ub"].access_ptr(
                                'w', offset=loop_cnt_64 * 64),
                            ub_info["float32_temp_wo_ub"].access_ptr(
                                'r', offset=loop_cnt_64 * 64),
                            tvm.const(0.5, dtype="float32"),
                            1, 1, 1, 8, 8))
                ib.emit(
                    tvm.call_extern("float32", "vmuls",
                                    ub_info["float32_temp_wo_ub"].access_ptr(
                                        'w', offset=loop_cnt_64 * 64),
                                    ub_info["float32_temp_wo_ub"].access_ptr(
                                        'r', offset=loop_cnt_64 * 64),
                                    tvm.const(scale_w, dtype="float32"),
                                    1, 1, 1, 8, 8))
                if half_pixel_centers:
                    ib.emit(
                        tvm.call_extern(
                            "float32", "vadds",
                            ub_info["float32_temp_wo_ub"].access_ptr(
                                'w', offset=loop_cnt_64 * 64),
                            ub_info["float32_temp_wo_ub"].access_ptr(
                                'r', offset=loop_cnt_64 * 64),
                            tvm.const(-0.5, dtype="float32"),
                            1, 1, 1, 8, 8))
                    ib.emit(
                        tvm.call_extern(
                            "float32", "vmax",
                            ub_info["float32_temp_wo_ub"].access_ptr(
                                'w', offset=loop_cnt_64 * 64),
                            ub_info["float32_temp_wo_ub"].access_ptr(
                                'r', offset=loop_cnt_64 * 64),
                            ub_info["float32_const_1"].access_ptr('r'),
                            1, 1, 1, 1, 8, 8, 0))
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(-1, dtype="uint64"),
                                    tvm.const(-1, dtype="uint64")))

            # cast to int to get pos
            _addr_info = [[ub_info["int32_w_pos_ub"], 0],
                          [ub_info["float32_temp_wo_ub"], 0]]
            _data_info = [w_out_vbi_count * f32_c0 * 4, f32_c0 * 8]
            kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info,
                                          "vconv_f322s32f")

            # cast to int to get pos
            _addr_info = [[ub_info["float32_w_pos_ub"], 0],
                          [ub_info["int32_w_pos_ub"], 0]]
            _data_info = [w_out_vbi_count * f32_c0 * 4, f32_c0 * 8]
            kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info,
                                          "vconv_s322f32")

            pos_reg[2] = tvm.expr.Cast(
                "int64",
                tvm.call_extern(
                    "handle", "",
                    ub_info["w0_w1_in_ub"].access_ptr("r"))).astype("int32")
            # get the top_left pos
            if loop_cnt_64 > 0:
                ib.emit(
                    tvm.call_extern("int32", "vmuls",
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'w'),
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'r'),
                                    tvm.const(f16_c0 * 2, dtype="int32"),
                                    loop_cnt_64, 1, 1, 8, 8))
                # to get the absolution address of left_top pos
                ib.emit(
                    tvm.call_extern("int32", "vadds",
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'w'),
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'r'),
                                    pos_reg[2],
                                    loop_cnt_64, 1, 1, 8, 8))
                # to get the absolution address of right_top pos
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const(((1 << 64) - 1) -
                                              ((1 << 48) - 1) +
                                              ((1 << 32)-1) -
                                              ((1 << 16) - 1),
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern("int32", "vadds",
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'w'),
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'r'),
                                    tvm.const(f16_c0 * 2, dtype="int32"),
                                    loop_cnt_64, 1, 1, 8, 8))
                # to get the absolution address of left_right_bottom pos
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const(((1 << 64) - 1) -
                                              ((1 << 56) - 1) +
                                              ((1 << 48) - 1) -
                                              ((1 << 40) - 1) +
                                              ((1 << 32) - 1) -
                                              ((1 << 24) - 1) +
                                              ((1 << 16) - 1) -
                                              ((1 << 8) - 1),
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern("int32", "vadds",
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'w'),
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'r'),
                                    tvm.const((w_in + 1) * f16_c0 * 2,
                                              dtype="int32"),
                                    loop_cnt_64, 1, 1, 8, 8))
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(-1, dtype="uint64"),
                                    tvm.const(-1, dtype="uint64")))

            # the left len should be f32_c0 * 4
            if loop_left_64 > 0:
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const((1 << loop_left_64) - 1,
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern("int32", "vmuls",
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'w', offset=loop_cnt_64 * 64),
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'r', offset=loop_cnt_64 * 64),
                                    tvm.const(f16_c0 * 2, dtype="int32"),
                                    1, 1, 1, 8, 8))
                # to get the absolution address of left_top pos
                ib.emit(
                    tvm.call_extern("int32", "vadds",
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'w', offset=loop_cnt_64 * 64),
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'r', offset=loop_cnt_64 * 64),
                                    pos_reg[2],
                                    1, 1, 1, 8, 8))
                # to get the absolution address of right_top pos
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const(((1 << 32) - 1) -
                                              ((1 << 16) - 1),
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern("int32", "vadds",
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'w', offset=loop_cnt_64 * 64),
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'r', offset=loop_cnt_64 * 64),
                                    tvm.const(f16_c0 * 2, dtype="int32"),
                                    1, 1, 1, 8, 8))
                # to get the absolution address of left_right_bottom pos
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const(((1 << 32) - 1) -
                                              ((1 << 24) - 1) +
                                              ((1 << 16) - 1) -
                                              ((1 << 8) - 1),
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern("int32", "vadds",
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'w', offset=loop_cnt_64 * 64),
                                    ub_info["int32_w_pos_ub"].access_ptr(
                                        'r', offset=loop_cnt_64 * 64),
                                    tvm.const((w_in + 1) * f16_c0 * 2,
                                              dtype="int32"),
                                    1, 1, 1, 8, 8))

                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(-1, dtype="uint64"),
                                    tvm.const(-1, dtype="uint64")))

            # to get the w_scalar and 1 - w_scalar
            if loop_cnt_64 > 0:
                ib.emit(
                    tvm.call_extern("float32", "vsub",
                                    ub_info["float32_temp_wo_ub"].access_ptr(
                                        'w'),
                                    ub_info["float32_temp_wo_ub"].access_ptr(
                                        'r'),
                                    ub_info["float32_w_pos_ub"].access_ptr(
                                        'r'),
                                    loop_cnt_64, 1, 1, 1, 8, 8, 8))
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const(((1 << 48) - 1) -
                                              ((1 << 32) - 1) +
                                              ((1 << 16) - 1),
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern("float32", "vsub",
                                    ub_info["float32_temp_wo_ub"].access_ptr(
                                        'w', offset=0),
                                    ub_info["float32_const_0"].access_ptr(
                                        'r'),
                                    ub_info["float32_temp_wo_ub"].access_ptr(
                                        'r'),
                                    loop_cnt_64, 1, 0, 1, 8, 0, 8))
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(-1, dtype="uint64"),
                                    tvm.const(-1, dtype="uint64")))

            if loop_left_64 > 0:
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const((1 << loop_left_64) - 1,
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern("float32", "vsub",
                                    ub_info["float32_temp_wo_ub"].access_ptr(
                                        'w', offset=loop_cnt_64 * 64),
                                    ub_info["float32_temp_wo_ub"].access_ptr(
                                        'r', offset=loop_cnt_64 * 64),
                                    ub_info["float32_w_pos_ub"].access_ptr(
                                        'r', offset=loop_cnt_64 * 64),
                                    1, 1, 1, 1, 8, 8, 8))
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const(((1 << 16) - 1),
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern("float32", "vsub",
                                    ub_info["float32_temp_wo_ub"].access_ptr(
                                        'w', offset=loop_cnt_64 * 64),
                                    ub_info["float32_const_0"].access_ptr(
                                        'r'),
                                    ub_info["float32_temp_wo_ub"].access_ptr(
                                        'r', offset=loop_cnt_64 * 64),
                                    1, 1, 0, 1, 8, 0, 8))
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(-1, dtype="uint64"),
                                    tvm.const(-1, dtype="uint64")))

            if is_bigger_ub_size:
                ib.emit(tvm.call_extern(
                    "int32", "copy_ubuf_to_cbuf",
                    l1_info["int32_w_pos_l1"].access_ptr(
                        "w", offset=prev_wo_cnt * 4),
                    ub_info["int32_w_pos_ub"].access_ptr(
                        "r", offset=0),
                    0, 1, w_out_vbi_count * 4, 0, 0))
                ib.emit(tvm.call_extern(
                    "float32", "copy_ubuf_to_cbuf",
                    l1_info["float32_temp_wo_l1"].access_ptr(
                        "w", offset=prev_wo_cnt * 4),
                    ub_info["float32_temp_wo_ub"].access_ptr(
                        "r", offset=0),
                    0, 1, w_out_vbi_count * 4, 0, 0))

        def _calc_w_out_scalar(temp_ub, h_scalar, neg_h_scalar, wo_len):
            """
            get w_out scalar, the order is:
            top_left, bottom_left, top_right, bottom_right

            """
            vbi_wo_cnt = (wo_len + VBI_BLOCK_COUNT - 1) // VBI_BLOCK_COUNT
            loop_cnt_64 = vbi_wo_cnt * f32_c0 * 4 // 64
            loop_left_64 = vbi_wo_cnt * f32_c0 * 4 % 64

            if loop_cnt_64 > 0:
                # to get the absolution address of right_top pos
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const(((1 << 56) - 1) -
                                              ((1 << 48) - 1) +
                                              ((1 << 40) - 1) -
                                              ((1 << 32) - 1) +
                                              ((1 << 24) - 1) -
                                              ((1 << 16) - 1) +
                                              ((1 << 8) - 1),
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern("float32", "vmuls",
                                    temp_ub.access_ptr(
                                        'w'),
                                    ub_info["float32_temp_wo_ub"].access_ptr(
                                        'r'),
                                    neg_h_scalar,
                                    loop_cnt_64, 1, 1, 8, 8))
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const(((1 << 64) - 1) -
                                              ((1 << 56) - 1) +
                                              ((1 << 48) - 1) -
                                              ((1 << 40) - 1) +
                                              ((1 << 32) - 1) -
                                              ((1 << 24) - 1) +
                                              ((1 << 16) - 1) -
                                              ((1 << 8) - 1),
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern("float32", "vmuls",
                                    temp_ub.access_ptr(
                                        'w'),
                                    ub_info["float32_temp_wo_ub"].access_ptr(
                                        'r'),
                                    h_scalar,
                                    loop_cnt_64, 1, 1, 8, 8))
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(-1, dtype="uint64"),
                                    tvm.const(-1, dtype="uint64")))

            if loop_left_64 > 0:
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const(((1 << 24) - 1) -
                                              ((1 << 16) - 1) +
                                              ((1 << 8) - 1),
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern("float32", "vmuls",
                                    temp_ub.access_ptr(
                                        'w', offset=loop_cnt_64 * 64),
                                    ub_info["float32_temp_wo_ub"].access_ptr(
                                        'r', offset=loop_cnt_64 * 64),
                                    neg_h_scalar,
                                    1, 1, 1, 8, 8))
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(0, dtype="uint64"),
                                    tvm.const(((1 << 32) - 1) -
                                              ((1 << 24) - 1) +
                                              ((1 << 16) - 1) -
                                              ((1 << 8) - 1),
                                              dtype="uint64")))
                ib.emit(
                    tvm.call_extern("float32", "vmuls",
                                    temp_ub.access_ptr(
                                        'w', offset=loop_cnt_64 * 64),
                                    ub_info["float32_temp_wo_ub"].access_ptr(
                                        'r', offset=loop_cnt_64 * 64),
                                    h_scalar,
                                    1, 1, 1, 8, 8))
                ib.emit(
                    tvm.call_extern("uint64", "set_vector_mask",
                                    tvm.const(-1, dtype="uint64"),
                                    tvm.const(-1, dtype="uint64")))

            # cast float32 scalar to float16
            _addr_info = [[ub_info["float16_w_scalar_ub"], 0],
                          [temp_ub, 0]]
            _data_info = [vbi_wo_cnt * f32_c0 * 4, f32_c0 * 8]
            kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info,
                                          "vconv_f322f16")

        _calc_h_out_pos_scalar()
        if wo_loop_count > 0:
            with ib.for_range(0, wo_loop_count) as wo_lp_idx:
                _calc_w_out_pos(wo_threshold, wo_lp_idx * wo_threshold)
            if wo_loop_left > 0:
                _calc_w_out_pos(wo_loop_left, wo_loop_count * wo_threshold)
        else:
            _calc_w_out_pos(w_out, 0)

        def _mv_gm_to_ub(ib, nc1_idx, hi_idx):
            ib.emit(tvm.call_extern("float16", "copy_gm_to_ubuf",
                                    ub_info["w0_w1_in_ub"].access_ptr(
                                        "rw", offset=0),
                                    inputs.access_ptr(
                                        "r",
                                        offset=(hi_idx + nc1_idx * h_in) *
                                        w_in * f16_c0),
                                    0, 1, w_in, 0, 0))
            ib.emit(tvm.call_extern("float16", "copy_gm_to_ubuf",
                                    ub_info["w0_w1_in_ub"].access_ptr(
                                        "rw", offset=w_in * f16_c0),
                                    inputs.access_ptr(
                                        "r",
                                        offset=(w_in - 1) * f16_c0 +
                                        (hi_idx + nc1_idx * h_in) *
                                        w_in * f16_c0),
                                    0, 1, 1, 0, 0))
            with ib.if_scope(hi_idx != h_in - 1):
                ib.emit(tvm.call_extern("float16", "copy_gm_to_ubuf",
                                        ub_info["w0_w1_in_ub"].access_ptr(
                                            "rw",
                                            offset=(w_in + 1) * f16_c0),
                                        inputs.access_ptr(
                                            "r",
                                            offset=(hi_idx + 1 +
                                                    nc1_idx * h_in) *
                                            w_in * f16_c0),
                                        0, 1, w_in, 0, 0))
                ib.emit(tvm.call_extern("float16", "copy_gm_to_ubuf",
                                        ub_info["w0_w1_in_ub"].access_ptr(
                                            "rw",
                                            offset=(2 * w_in + 1) * f16_c0),
                                        inputs.access_ptr(
                                            "r",
                                            offset=(w_in - 1) * f16_c0 +
                                            (hi_idx + 1 + nc1_idx * h_in) *
                                            w_in * f16_c0),
                                        0, 1, 1, 0, 0))
            with ib.else_scope():
                ib.emit(tvm.call_extern("float16", "copy_gm_to_ubuf",
                                        ub_info["w0_w1_in_ub"].access_ptr(
                                            "rw",
                                            offset=(w_in + 1) * f16_c0),
                                        inputs.access_ptr(
                                            "r",
                                            offset=(h_in - 1 + nc1_idx *
                                                    h_in) *
                                            w_in * f16_c0),
                                        0, 1, w_in, 0, 0))
                ib.emit(tvm.call_extern("float16", "copy_gm_to_ubuf",
                                        ub_info["w0_w1_in_ub"].access_ptr(
                                            "rw",
                                            offset=(2 * w_in + 1) * f16_c0),
                                        inputs.access_ptr(
                                            "r",
                                            offset=(w_in - 1) * f16_c0 +
                                            (h_in - 1 + nc1_idx * h_in) *
                                            w_in * f16_c0),
                                        0, 1, 1, 0, 0))

        pipe_all_str = tvm.call_pure_intrin("int32",
                                            "tvm_cce_string_print",
                                            "PIPE_ALL")
        nc1 = size_in[0] * size_in[1]
        # change the tensor type from int32 to uint16
        uint16_w_pos = tvm.decl_buffer(
            ub_info["int32_w_pos_ub"].shape,
            "uint16_t",
            name="uint16_w_pos",
            scope=param.scope_ubuf,
            data=ub_info["int32_w_pos_ub"].data)

        with ib.for_range(0, nc1) as nc1_idx:
            # set the init value to -1
            pos_reg[1] = tvm.const(-1, dtype="int32")
            with ib.for_range(0, h_per_core) as h_loop_idx:
                with ib.if_scope(
                        core_idx * h_per_core + h_loop_idx < h_out):
                    # get h_out pos
                    ib.emit(
                        tvm.call_extern(
                            "int32", "reg_mov",
                            tvm.call_extern("int32",
                                            "reg",
                                            pos_reg[0]),
                            ub_info["int32_h_pos_ub"].access_ptr(
                                "r",
                                offset=h_loop_idx)))
                    # get h_out scalar and 1-scalar
                    with ib.if_scope(pos_reg[0] < h_in):
                        ib.emit(
                            tvm.call_extern(
                                "float32", "reg_mov",
                                tvm.call_extern("float32",
                                                "reg",
                                                pos_reg_float32[0]),
                                ub_info["float32_h_scalar_ub"].access_ptr(
                                    "r",
                                    offset=h_loop_idx)))
                        ib.emit(
                            tvm.call_extern(
                                "float32", "reg_mov",
                                tvm.call_extern("float32",
                                                "reg",
                                                pos_reg_float32[1]),
                                ub_info["float32_h_scalar_ub"].access_ptr(
                                    "r",
                                    offset=h_loop_idx + vbi_ub_size_h)))

                    # to make sure read scalar is done
                    ib.emit(tvm.call_extern('int32',
                                            'pipe_barrier',
                                            pipe_all_str))

                    with ib.if_scope(pos_reg[0] < h_in):
                        with ib.if_scope(pos_reg[0] != pos_reg[1]):
                            # copy input to ub
                            _mv_gm_to_ub(ib, nc1_idx, pos_reg[0])
                            pos_reg[1] = pos_reg[0]
                            # to make sure mte2 is done
                            ib.emit(tvm.call_extern('int32',
                                                    'pipe_barrier',
                                                    pipe_all_str))

                    def _inner_vbi_process(data_len, elem_offset):
                        """
                        make vbi compute
                        """
                        repeat_vbi = data_len // VBI_BLOCK_COUNT
                        repeat_vbi_left = data_len % VBI_BLOCK_COUNT * f16_c0

                        def _copy_hw_scalar_to_l1(data_len, elem_offset):
                            """
                            copy hw scalar to l1 for temp store
                            """
                            data_len = (data_len + f32_c0 - 1) // f32_c0 \
                                            * f32_c0 * 4
                            elem_offset = (elem_offset + f32_c0 - 1) // \
                                          f32_c0 * f32_c0 * 4

                            ib.emit(tvm.call_extern(
                                "float16", "copy_ubuf_to_cbuf",
                                l1_info["float16_hw_scalar_l1"].access_ptr(
                                    "w", offset=elem_offset),
                                ub_info["float16_w_scalar_ub"].access_ptr(
                                    "r", offset=0),
                                0, 1, data_len // f16_c0, 0, 0))

                        def _copy_hw_scalar_to_ub(data_len, elem_offset):
                            """
                            copy hw scalar to ub for compute
                            """
                            data_len = (data_len + f32_c0 - 1) // f32_c0 \
                                       * f32_c0 * 4
                            elem_offset = (elem_offset + f32_c0 - 1) // \
                                          f32_c0 * f32_c0 * 4

                            ib.emit(tvm.call_extern(
                                "float16", "copy_cbuf_to_ubuf",
                                ub_info["float16_w_scalar_ub"].access_ptr(
                                    "w", offset=0),
                                l1_info["float16_hw_scalar_l1"].access_ptr(
                                    "r", offset=elem_offset),
                                0, 1, data_len // f16_c0, 0, 0))

                        # get h_scalar * w_scalar
                        if is_smaller_l1_size and nc1 > 1:
                            with ib.if_scope(nc1_idx == 0):
                                _calc_w_out_scalar(ub_info["float32_w_pos_ub"],
                                                   pos_reg_float32[0],
                                                   pos_reg_float32[1],
                                                   data_len)
                                _copy_hw_scalar_to_l1(data_len,
                                                      elem_offset +
                                                      h_loop_idx * vbi_ub_size)
                            with ib.else_scope():
                                _copy_hw_scalar_to_ub(data_len,
                                                      elem_offset +
                                                      h_loop_idx * vbi_ub_size)

                        else:
                            _calc_w_out_scalar(ub_info["float32_w_pos_ub"],
                                               pos_reg_float32[0],
                                               pos_reg_float32[1],
                                               data_len)

                        # make bilinear compute
                        if repeat_vbi > 0:
                            vbi_repeat_threshold = 255
                            threshold_left = repeat_vbi % vbi_repeat_threshold
                            if repeat_vbi <= vbi_repeat_threshold:
                                ib.emit(
                                    tvm.call_extern(
                                        "float16",
                                        "vbi",
                                        ub_info["out_ub"].access_ptr(
                                            "w", offset=0),
                                        uint16_w_pos.access_ptr(
                                            "r", offset=0),
                                        ub_info["float16_w_scalar_ub"\
                                        ].access_ptr(
                                            "r", offset=0),
                                        4, 1, 1, 128, repeat_vbi))
                            else:
                                ib.emit(
                                    tvm.call_extern(
                                        "float16",
                                        "vbi",
                                        ub_info["out_ub"].access_ptr(
                                            "w", offset=0),
                                        uint16_w_pos.access_ptr(
                                            "r", offset=0),
                                        ub_info["float16_w_scalar_ub"\
                                            ].access_ptr(\
                                            "r", offset=0),
                                        4, 1, 1, 128, vbi_repeat_threshold))
                                ib.emit(
                                    tvm.call_extern(
                                        "float16",
                                        "vbi",
                                        ub_info["out_ub"].access_ptr(
                                            "w",
                                            offset=vbi_repeat_threshold * 128),
                                        uint16_w_pos.access_ptr(
                                            "r",
                                            offset=vbi_repeat_threshold * 64),
                                        ub_info["float16_w_scalar_ub"\
                                            ].access_ptr(\
                                            "r",
                                            offset=vbi_repeat_threshold * 32),
                                        4, 1, 1, 128, threshold_left))

                        if repeat_vbi_left > 0:
                            if repeat_vbi_left <= 64:
                                ib.emit(
                                    tvm.call_extern(
                                        "uint64", "set_vector_mask",
                                        tvm.const(0, dtype="uint64"),
                                        tvm.const(
                                            (1 << repeat_vbi_left) - 1,
                                            dtype="uint64")))
                            else:
                                left_cnt = repeat_vbi_left % 64
                                ib.emit(
                                    tvm.call_extern(
                                        "uint64", "set_vector_mask",
                                        tvm.const((1 << left_cnt) - 1,
                                                  dtype="uint64"),
                                        tvm.const((1 << 64) - 1,
                                                  dtype="uint64")))
                            ib.emit(
                                tvm.call_extern(
                                    "float16",
                                    "vbi",
                                    ub_info["out_ub"].access_ptr(
                                        "w",
                                        offset=repeat_vbi * 128),
                                    uint16_w_pos.access_ptr(
                                        "r",
                                        offset=repeat_vbi * 64),
                                    ub_info["float16_w_scalar_ub"\
                                    ].access_ptr(
                                        "r",
                                        offset=repeat_vbi * 32),
                                    4, 1, 1, 128, 1))
                            ib.emit(
                                tvm.call_extern(
                                    "uint64", "set_vector_mask",
                                    tvm.const(-1, dtype="uint64"),
                                    tvm.const(-1, dtype="uint64")))

                            # cast to float16 to float32

                        _addr_info = [
                            [ub_info["float32_out_ub"], 0],
                            [ub_info["out_ub"], 0]]
                        _data_info = [data_len * f16_c0, f32_c0 * 8]
                        kernel_api.kernel_cast_to_fuc(
                            ib, _addr_info,
                            _data_info,
                            "vconv_f162f32")

                        # output data to gm
                        ib.emit(
                            tvm.call_extern(
                                "float32",
                                "copy_ubuf_to_gm",
                                outputs.access_ptr(
                                    "rw",
                                    offset=(core_idx * h_per_core +
                                            h_loop_idx +
                                            nc1_idx * h_out) * \
                                           w_out * f16_c0 + \
                                           elem_offset * f16_c0),
                                ub_info["float32_out_ub"].access_ptr("r"),
                                0, 1, data_len * 2, 0, 0))

                    if wo_loop_count > 0:

                        def _copy_l1_to_ub(data_len, elem_offset):
                            """
                            copy pos and scalar of w_out to ub
                            """
                            ib.emit(tvm.call_extern(
                                "int32", "copy_cbuf_to_ubuf",
                                ub_info["int32_w_pos_ub"].access_ptr(
                                    "w", offset=0),
                                l1_info["int32_w_pos_l1"].access_ptr(
                                    "r", offset=elem_offset * 4),
                                0, 1, data_len * 4, 0, 0))
                            ib.emit(tvm.call_extern(
                                "float32", "copy_cbuf_to_ubuf",
                                ub_info["float32_temp_wo_ub"].access_ptr(
                                    "w", offset=0),
                                l1_info["float32_temp_wo_l1"].access_ptr(
                                    "r", offset=elem_offset * 4),
                                0, 1, data_len * 4, 0, 0))

                        with ib.for_range(0, wo_loop_count) as wo_lp_idx:
                            _copy_l1_to_ub(wo_threshold // f32_c0,
                                           wo_lp_idx * wo_threshold)
                            _inner_vbi_process(wo_threshold,
                                               wo_lp_idx * wo_threshold)

                        if wo_loop_left > 0:
                            left_block = (wo_loop_left + f32_c0 - 1) // f32_c0
                            _copy_l1_to_ub(
                                left_block,
                                wo_loop_count * wo_threshold)
                            _inner_vbi_process(
                                wo_loop_left,
                                wo_loop_count * wo_threshold)
                    else:
                        _inner_vbi_process(w_out, 0)

        return ib.get()

    # No.5 normal input/output H/W
    else:
        args_str_v = tvm.call_pure_intrin("int32", "tvm_cce_string_print",
                                          "PIPE_V")
        args_str_v_all = tvm.call_pure_intrin("int32", "tvm_cce_string_print",
                                              "PIPE_ALL")
        # use multi-core resource
        h_per_core = h_out // core_counts + (1
                                             if h_out % core_counts > 0 else 0)
        is_same_percore = 0 if h_out % core_counts == 0 else 1
        core_num = h_out // h_per_core + is_same_percore
        ib.scope_attr(block, "thread_extent", core_num)
        # use multi-core resource
        idx = block.var
        # Note:
        #   1. all scale calculate in ubuf and store in L1, dtype is f32
        #   2. 16 or 8 lines data move in L1 first, each line split by
        #       HW_SIZE_1024 block at most
        #   3. 8 lines data in ubuf move out, each line split by HW_SIZE_512
        #      block at most
        ##########
        # apply ub for x HW_SIZE_512
        ub_info = {}
        l1_info = {}

        def apply_ub():
            # apply ub for x(h,w) HW_SIZE_512  x1(h,w+1) HW_SIZE_512
            ub_info["x0_512_x1_512_ub"] = apply_store_buffer(
                ib, "float32", [HW_SIZE_512 * 2 * f32_c0], name="x_512_x1_512")
            # apply ub for x(h+1,w) HW_SIZE_512  x1(h+1,w+1) HW_SIZE_512
            ub_info["x2_512_x3_512_ub"] = apply_store_buffer(
                ib, "float32", [HW_SIZE_512 * 2 * f32_c0], name="x2_512_x3_512")
            ub_info["int32_512_ub"] = apply_store_buffer(ib,
                                                         "int32",
                                                         [HW_SIZE_256 * f32_c0],
                                                         name="int32_512")
            ub_info["int32_512_ub_y"] = apply_store_buffer(
                ib, "int32", [8 * f32_c0], name="int32_512_ub_y")
            ub_info["const_0"] = apply_store_buffer(ib,
                                                    "float32", [8 * f32_c0],
                                                    name="const_0")

            ub_info["x0_scale_x1_scale_512_ub"] = apply_store_buffer(
                ib, "float32", [HW_SIZE_512 * f32_c0],
                name="x0_scale_x1_scale_512_ub")
            ub_info["y0_scale_ub"] = apply_store_buffer(ib,
                                                        "float32",
                                                        [2 * f32_c0],
                                                        name="y0_scale_ub")

            ub_info["out_f32"] = apply_store_buffer(ib,
                                                    "float32",
                                                    [HW_SIZE_512 * f32_c0],
                                                    name="out_f32")
            l1_info["l1_ypos"] = apply_store_buffer(ib,
                                                    "int32",
                                                    [HW_SIZE_512 * 3 * f32_c0],
                                                    name="l1_ypos",
                                                    scope=param.scope_cbuf)
            l1_info["l1_xpos"] = apply_store_buffer(
                ib,
                "int32",
                [HW_SIZE_512 * 4 * f32_c0],  # to support w_out HW_SIZE_2048
                name="l1_xpos",
                scope=param.scope_cbuf)
            l1_info["l1_xscale"] = apply_store_buffer(
                ib,
                "float32",
                [2 * HW_SIZE_512 * 4 * f32_c0],  # to support w_out HW_SIZE_2048
                name="l1_xscale",
                scope=param.scope_cbuf)
            l1_info["l1_yscale"] = apply_store_buffer(
                ib,
                "float32", [2 * HW_SIZE_512 * 3 * f32_c0],
                name="l1_yscale",
                scope=param.scope_cbuf)
            l1_info["l1_input"] = apply_store_buffer(
                ib,
                "float32", [2 * (w_in + 1) * 16],
                name="l1_input",
                scope=param.scope_cbuf)
            ub_info["ub_input"] = apply_store_buffer(
                ib, "float32", [2 * (w_in + 1) * 16], name="ub_input")
            ub_info["f16_512"] = apply_store_buffer(
                ib, "float16", [HW_SIZE_512 * 8], name="f16_512")
            if devices == "cloud":
                ub_info["f16_512_1"] = apply_store_buffer(
                    ib, "float16", [HW_SIZE_256 * 8], name="f16_512_1")
            else:
                ub_info["f16_512_1"] = apply_store_buffer(
                    ib, "float16", [HW_SIZE_512 * 8], name="f16_512_1")
            # apply ub for x(h,w) HW_SIZE_512  x1(h,w+1) HW_SIZE_512
            ub_info["x0_512_x1_512_ub_1"] = apply_store_buffer(
                ib, "float32", [HW_SIZE_512 * 2 * f32_c0],
                name="x_512_x1_512_1")
            # apply ub for x(h+1,w) HW_SIZE_512  x1(h+1,w+1) HW_SIZE_512
            ub_info["x2_512_x3_512_ub_1"] = apply_store_buffer(
                ib, "float32", [HW_SIZE_512 * 2 * f32_c0],
                name="x2_512_x3_512_1")
            # for unfold
            ub_info["x0_512_x1_512_ub_unfold"] = apply_store_buffer(
                ib, "float32", [HW_SIZE_512 * 2 * f32_c0],
                name="x0_512_x1_512_unfold")
            ub_info["x2_512_x3_512_ub_unfold"] = apply_store_buffer(
                ib, "float32", [HW_SIZE_512 * 2 * f32_c0],
                name="x2_512_x3_512_unfold")

        apply_ub()
        # pylint: disable=simplifiable-if-statement
        if w_in <= HW_SIZE_512 and dtype == "float32":
            is_input_in_ub = True
        elif w_in <= HW_SIZE_256 and dtype == "float16" and devices == "mini":
            is_input_in_ub = True
        elif w_in <= HW_SIZE_512 and dtype == "float16" and devices == "cloud":
            is_input_in_ub = True
        else:
            is_input_in_ub = False

        # No.4 normal input/output H/W : range HW_SIZE_512 create
        range_512 = apply_store_buffer(ib,
                                       "float32", [HW_SIZE_512 * f32_c0],
                                       name="range_512")

        # check whether scale is int or 1/int
        # create HW_SIZE_512 blocks from 0-511 start
        loop_level1 = 8
        c0 = 16
        ib.emit(
            tvm.call_extern("uint64", "set_vector_mask",
                            tvm.const(0, dtype="uint64"),
                            tvm.const((1 << f32_c0) - 1, dtype="uint64")))
        ib.emit(
            tvm.call_extern("float32", "vector_dup",
                            range_512.access_ptr('w', offset=8 * 0),
                            tvm.const(0.0, dtype="float32"), 1, 1, 0, 8, 0))

        with ib.for_range(0, loop_level1 - 1) as i:
            ib.emit(tvm.call_extern('int32', 'pipe_barrier', args_str_v))
            ib.emit(
                tvm.call_extern("float32", "vadds",
                                range_512.access_ptr('w', offset=8 * (i + 1)),
                                range_512.access_ptr('r', offset=8 * i),
                                tvm.const(1.0, dtype="float32"), 1, 1, 1, 1,
                                1))
        ib.emit(
            tvm.call_extern("uint64", "set_vector_mask",
                            tvm.const(-1, dtype="uint64"),
                            tvm.const(-1, dtype="uint64")))
        loop_level1 = HW_SIZE_256 // 8
        with ib.for_range(0, loop_level1 - 1) as i:
            ib.emit(tvm.call_extern('int32', 'pipe_barrier', args_str_v))
            ib.emit(
                tvm.call_extern(
                    "float32", "vadds",
                    range_512.access_ptr('w', offset=8 * 8 * (i + 1)),
                    range_512.access_ptr('r', offset=8 * 8 * i),
                    tvm.const(8.0, dtype="float32"), 1, 1, 1, 8, 8))

        # create HW_SIZE_512 blocks from 0-511 end

        # Note:
        #   1. w direction scale HW_SIZE_512 left and right join together
        #   2. h direction scale HW_SIZE_512 top and bottom store speartely
        #      to double lines
        # Note:
        #   1. devices mini f32 <-> s32 need convert to f16 first
        #   2. devices mini f32 <-> s32 convert directly

        # No.4 normal input/output H/W: scale calculate:
        # input H\W == output: devices mini
        # level_1 loop
        loop_levelx = w_out // HW_SIZE_256 + \
                      (1 if w_out % HW_SIZE_256 > 0 else 0)
        loop_levely = h_per_core // HW_SIZE_256 + \
                      (1 if h_per_core % HW_SIZE_256 > 0 else 0)

        # pylint: disable=too-many-statements
        def calu_x_y_out_pos(_ib, _loop_info, _scale_info, _ub_info, _l1_info):
            _x_loop = _loop_info[0]
            _y_loop = _loop_info[1]
            x_index_ub = _ub_info["x0_scale_x1_scale_512_ub"]
            y_index_ub = _ub_info["x2_512_x3_512_ub"]
            one_ub = ub_info["const_0"]
            int32_ub = _ub_info["int32_512_ub"]
            _scale_w, _scale_h = _scale_info
            min_loop = min(_x_loop, _y_loop)

            def _cast_f32_to_int32(_data_info, f32_ub_info, int32_ub_info,
                                   dst_f32_ub_info):
                _ub_int32_info = int32_ub_info
                _ub_fp32_info = f32_ub_info
                _ub_des_fp32 = dst_f32_ub_info
                _addr_info = [_ub_int32_info, _ub_fp32_info]
                if devices == "cloud":
                    _addr_info = [_ub_int32_info, _ub_fp32_info]
                    kernel_api.kernel_cast_to_fuc(_ib, _addr_info, _data_info,
                                                  "vconv_f322s32f")
                    _addr_info = [_ub_des_fp32, _ub_int32_info]
                    kernel_api.kernel_cast_to_fuc(_ib, _addr_info, _data_info,
                                                  "vconv_s322f32")
                else:
                    _ub_fp16_512 = _ub_info["f16_512_1"]
                    _addr_info = [[_ub_fp16_512, 0], _ub_fp32_info]
                    kernel_api.kernel_cast_to_fuc(_ib, _addr_info, _data_info,
                                                  "vconv_f322f16")
                    _addr_info = [_ub_int32_info, [_ub_fp16_512, 0]]
                    kernel_api.kernel_cast_to_fuc(_ib, _addr_info, _data_info,
                                                  "vconv_f162s32f")
                    ib.emit(
                        tvm.call_extern("float16", "set_deqscale",
                                        tvm.const(1.0, dtype="float16")))
                    _addr_info = [[_ub_fp16_512, 0], _ub_int32_info]
                    kernel_api.kernel_cast_to_fuc(_ib, _addr_info, _data_info,
                                                  "vconv_deq")
                    _addr_info = [_ub_des_fp32, [_ub_fp16_512, 0]]
                    kernel_api.kernel_cast_to_fuc(_ib, _addr_info, _data_info,
                                                  "vconv_f162f32")

            def _process_pos_to_l1(src_ub_info, mid_ub_info, des_l1_info,
                                   para_list, loop_index):
                _input_fp32_ub = src_ub_info[0]
                _input_fp32_512_num = src_ub_info[1]
                _mid_ub_int32 = mid_ub_info[0]
                _mid_one_ub_fp32 = mid_ub_info[1]
                _out_l1_int = des_l1_info[0]
                _out_l1_fp32 = des_l1_info[1]
                _scale, _core_id, _is_h, _h_per_core = para_list
                if _is_h:
                    if devices == "cloud":
                        _ib.emit(
                            tvm.call_extern(
                                _input_fp32_ub.dtype, "vadds",
                                _input_fp32_ub.access_ptr('w'),
                                _input_fp32_512_num.access_ptr('r'),
                                _core_id * tvm.const(_h_per_core,
                                                     dtype="float32"),
                                32, 1, 1, 8, 8))
                        _input_fp32_512_num = _input_fp32_ub
                    else:
                        int_reg = _ib.allocate("int32", (1,),
                                               name="int_data",
                                               scope=param.scope_reg)
                        int_reg[0] = _core_id
                        int32_ub_8 = apply_store_buffer(
                            ib, "int32", [8], name="int32_ub_8")
                        float32_ub_8 = apply_store_buffer(
                            ib, "float32", [64], name="float32_ub_8")
                        kernel_api.kernel_vector_dup_fuc(_ib, [int32_ub_8, 0],
                                                         int_reg[0],
                                                         [1, 64])
                        ib.emit(
                            tvm.call_extern("float16", "set_deqscale",
                                            tvm.const(1.0, dtype="float16")))
                        float16_ub_8 = apply_store_buffer(ib,
                                                          "float16", [128],
                                                          name="float16_ub_8")
                        _addr_info = [[float16_ub_8, 0], [int32_ub_8, 0]]
                        kernel_api.kernel_cast_to_fuc(_ib, _addr_info, [1, 64],
                                                      "vconv_deq")
                        _addr_info = [[float32_ub_8, 0], [float16_ub_8, 0]]
                        kernel_api.kernel_cast_to_fuc(_ib, _addr_info, [1, 64],
                                                      "vconv_f162f32")
                        kernel_api.kernel_scalar_to_one_fuc(
                            _ib,
                            [[float32_ub_8, 0],
                             [float32_ub_8, 0]],
                            [1, 64],
                            ["vmuls", _h_per_core])
                        core_reg = _ib.allocate("float32", (1,),
                                                name="core_reg",
                                                scope=param.scope_reg)
                        _ib.emit(tvm.call_extern('int32',
                                                 'pipe_barrier',
                                                 args_str_v_all))
                        core_reg[0] = float32_ub_8.vload(0, "float32")
                        _ib.emit(
                            tvm.call_extern(
                                _input_fp32_ub.dtype, "vadds",
                                _input_fp32_ub.access_ptr('w'),
                                _input_fp32_512_num.access_ptr('r'),
                                core_reg[0], 32, 1, 1, 8, 8))
                        _input_fp32_512_num = _input_fp32_ub
                if half_pixel_centers:
                    _ib.emit(
                        tvm.call_extern(_input_fp32_512_num.dtype, "vadds",
                                        _input_fp32_512_num.access_ptr('w'),
                                        _input_fp32_512_num.access_ptr('r'),
                                        tvm.const(0.5, dtype="float32"), 32, 1,
                                        1, 8, 8))
                _ib.emit(
                    tvm.call_extern(_input_fp32_ub.dtype, "vmuls",
                                    _input_fp32_ub.access_ptr('w'),
                                    _input_fp32_512_num.access_ptr('r'),
                                    tvm.const(_scale, dtype="float32"), 32, 1,
                                    1, 8, 8))
                if half_pixel_centers:
                    _ib.emit(
                        tvm.call_extern(_input_fp32_ub.dtype, "vadds",
                                        _input_fp32_ub.access_ptr('w'),
                                        _input_fp32_ub.access_ptr('r'),
                                        tvm.const(-0.5, dtype="float32"),
                                        32, 1, 1, 8, 8))
                    ib.emit(
                        tvm.call_extern("float32", "vector_dup",
                                        ub_info["const_0"].access_ptr('w'),
                                        tvm.const(0.0, dtype="float32"),
                                        1, 1, 0, 8, 0))

                    _ib.emit(
                        tvm.call_extern(_input_fp32_ub.dtype, "vmax",
                                        _input_fp32_ub.access_ptr('w'),
                                        _input_fp32_ub.access_ptr('r'),
                                        ub_info["const_0"].access_ptr('r'),
                                        32, 1, 1, 1, 8, 8, 0))
                    if _input_fp32_512_num == range_512:
                        _ib.emit(
                            tvm.call_extern(
                                _input_fp32_512_num.dtype, "vadds",
                                _input_fp32_512_num.access_ptr('w'),
                                _input_fp32_512_num.access_ptr('r'),
                                tvm.const(-0.5, dtype="float32"),
                                32, 1, 1, 8, 8))

                ib.emit(
                    tvm.call_extern("float32", "vector_dup",
                                    ub_info["const_0"].access_ptr('w'),
                                    tvm.const(1.0, dtype="float32"),
                                    1, 1, 0, 8, 0))

                data_info = [32 * 64, 64]
                # process x pos
                _cast_f32_to_int32(data_info, [_input_fp32_ub, 0],
                                   [_mid_ub_int32, 0],
                                   [_input_fp32_ub, HW_SIZE_256 * 8])
                _addr_info = [[_out_l1_int, loop_index * HW_SIZE_256 * 8],
                              [_mid_ub_int32, 0]]
                kernel_api.kernel_cp_fuc(_ib, _addr_info, [HW_SIZE_256 * 8, 8],
                                         "copy_ubuf_to_cbuf")
                # x_float - x_int
                _addr_info = [[_input_fp32_ub, HW_SIZE_256 * 8],
                              [_input_fp32_ub, 0],
                              [_input_fp32_ub, HW_SIZE_256 * 8]]
                kernel_api.kernel_two_to_one_common_fuc(
                    _ib, _addr_info, data_info, "vsub")

                _ib.emit(
                    tvm.call_extern(
                        _input_fp32_ub.dtype, "vsub",
                        _input_fp32_ub.access_ptr('w', offset=0),
                        _mid_one_ub_fp32.access_ptr('r'),
                        _input_fp32_ub.access_ptr('r', offset=HW_SIZE_256 * 8),
                        32, 1, 1, 1, 8, 0, 8))
                # copy ub tp l1
                _addr_info = [[_out_l1_fp32, loop_index * HW_SIZE_256 * 2 * 8],
                              [_input_fp32_ub, 0]]
                kernel_api.kernel_cp_fuc(_ib, _addr_info,
                                         [HW_SIZE_256 * 2 * 8, 8],
                                         "copy_ubuf_to_cbuf")

            with _ib.for_range(0, min_loop) as _loop:
                _process_pos_to_l1(
                    [x_index_ub, range_512], [int32_ub, one_ub],
                    [_l1_info["l1_xpos"], _l1_info["l1_xscale"]],
                    [_scale_w, idx, False, h_per_core], _loop)
                _process_pos_to_l1(
                    [y_index_ub, range_512], [int32_ub, one_ub],
                    [_l1_info["l1_ypos"], _l1_info["l1_yscale"]],
                    [_scale_h, idx, True, h_per_core], _loop)
                kernel_api.kernel_scalar_to_one_fuc(
                    _ib, [[range_512, 0], [range_512, 0]], [HW_SIZE_256 * 8, 64],
                    ["vadds", 256.0])
            if _x_loop != _y_loop:
                tail_loop = (_x_loop -
                             min_loop) if _y_loop == min_loop else (_y_loop -
                                                                    min_loop)
                index_ub = x_index_ub if _y_loop == min_loop else y_index_ub
                index_l1_int = _l1_info[
                    "l1_xpos"] if _y_loop == min_loop else _l1_info["l1_ypos"]
                index_l1_fp32 = _l1_info["l1_xscale"] \
                    if _y_loop == min_loop else _l1_info["l1_yscale"]
                tail_scale = _scale_w if _y_loop == min_loop else _scale_h
                # pylint: disable=simplifiable-if-expression
                is_h = False if _y_loop == min_loop else True
                with _ib.for_range(0, tail_loop) as _loop:
                    _process_pos_to_l1([index_ub, range_512],
                                       [int32_ub, one_ub],
                                       [index_l1_int, index_l1_fp32],
                                       [tail_scale, idx, is_h, h_per_core],
                                       _loop + min_loop)
                    kernel_api.kernel_scalar_to_one_fuc(
                        _ib, [[range_512, 0], [range_512, 0]], [HW_SIZE_256 * 8, 64],
                        ["vadds", 256.0])

        calu_x_y_out_pos(ib, [loop_levelx, loop_levely], [scale_w, scale_h],
                         ub_info, l1_info)
        pos_reg = apply_reg_buffer(ib, "int32", [10], name="pos_reg")
        reg_count = 8

        # calu y edge
        nc1_input_total = size_in[0] * size_in[1]
        nc1_input_offset = size_in[2] * size_in[3] * size_in[4]
        nc1_output_offset = size_out[2] * size_out[3] * size_out[4]
        y_index_reg = pos_reg[0]
        x_index_reg = pos_reg[2]
        x_segment_loop = w_out // HW_SIZE_256
        x_segment_tail = w_out % HW_SIZE_256

        # pylint: disable=too-many-arguments
        def _copy_input_to_l1(src_gm_ub, src_offset, dst_l1_ub, dat_len,
                              mid_ub, y_index):
            dst = dst_l1_ub
            src = src_gm_ub
            _dtype = src.dtype
            cp_segment = dat_len // 2
            if _dtype == "float32":
                if is_input_in_ub is not True:
                    _addr = [[dst, 0], [src, src_offset]]
                    kernel_api.kernel_cp_fuc(ib, _addr, [cp_segment, 8],
                                             "copy_gm_to_cbuf")
                    _addr = [[dst, cp_segment],
                             [src, src_offset + cp_segment - 16]]
                    kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                             "copy_gm_to_cbuf")
                    with ib.if_scope(y_index == size_in[2] - 1):
                        _addr = [[dst, cp_segment + 16], [src, src_offset]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [cp_segment, 8],
                                                 "copy_gm_to_cbuf")
                        _addr = [[dst, cp_segment * 2 + 16],
                                 [src, src_offset + cp_segment - 16]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                 "copy_gm_to_cbuf")
                    with ib.else_scope():
                        _addr = [[dst, cp_segment + 16],
                                 [src, src_offset + cp_segment]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [cp_segment, 8],
                                                 "copy_gm_to_cbuf")
                        _addr = [[dst, cp_segment * 2 + 16],
                                 [src, src_offset + cp_segment * 2 - 16]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                 "copy_gm_to_cbuf")
                else:
                    dst = ub_info["ub_input"]
                    _addr = [[dst, 0], [src, src_offset]]
                    kernel_api.kernel_cp_fuc(ib, _addr, [cp_segment, 8],
                                             "copy_gm_to_ubuf")
                    _addr = [[dst, cp_segment],
                             [src, src_offset + cp_segment - 16]]
                    kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                             "copy_gm_to_ubuf")
                    with ib.if_scope(y_index == size_in[2] - 1):
                        _addr = [[dst, cp_segment + 16], [src, src_offset]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [cp_segment, 8],
                                                 "copy_gm_to_ubuf")
                        _addr = [[dst, cp_segment * 2 + 16],
                                 [src, src_offset + cp_segment - 16]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                 "copy_gm_to_ubuf")
                    with ib.else_scope():
                        _addr = [[dst, cp_segment + 16],
                                 [src, src_offset + cp_segment]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [cp_segment, 8],
                                                 "copy_gm_to_ubuf")
                        _addr = [[dst, cp_segment * 2 + 16],
                                 [src, src_offset + cp_segment * 2 - 16]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                 "copy_gm_to_ubuf")
            else:
                # copy to ub
                # ub do conv fp16 tp f32
                # copy tp l1
                vconv_loop = cp_segment // (HW_SIZE_256 * 8)
                vconv_tail = cp_segment % (HW_SIZE_256 * 8)
                fp32_mid = mid_ub[0]
                fp32_mid_1 = mid_ub[1]
                fp16_mid = mid_ub[2]
                fp16_mid_1 = mid_ub[3]
                fp32_dst = ub_info["ub_input"]
                with ib.for_range(0, vconv_loop) as vconv_loop_i:
                    _addr = [[fp16_mid, 0],
                             [src, src_offset + vconv_loop_i * HW_SIZE_256 * 8]]
                    kernel_api.kernel_cp_fuc(ib, _addr, [HW_SIZE_256 * 8, 16],
                                             "copy_gm_to_ubuf")
                    _addr = [[fp32_mid, 0], [fp16_mid, 0]]
                    kernel_api.kernel_cast_to_fuc(ib, _addr, [HW_SIZE_256 * 8, 8 * 8],
                                                  "vconv_f162f32")
                    if is_input_in_ub is not True:
                        _addr = [[dst, vconv_loop_i * HW_SIZE_256 * 8], [fp32_mid, 0]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [HW_SIZE_256 * 8, 8],
                                                 "copy_ubuf_to_cbuf")
                    else:
                        _addr = [[fp32_dst, vconv_loop_i * HW_SIZE_256 * 8],
                                 [fp32_mid, 0]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [HW_SIZE_256 * 8, 8],
                                                 "copy_ubuf_to_ubuf")
                if vconv_tail != 0:
                    _addr = [[fp16_mid, 0],
                             [src, src_offset + vconv_loop * HW_SIZE_256 * 8]]
                    kernel_api.kernel_cp_fuc(ib, _addr, [vconv_tail, 16],
                                             "copy_gm_to_ubuf")
                    _addr = [[fp32_mid, 0], [fp16_mid, 0]]
                    kernel_api.kernel_cast_to_fuc(ib, _addr,
                                                  [vconv_tail, 8 * 8],
                                                  "vconv_f162f32")
                    if is_input_in_ub is not True:
                        _addr = [[dst, vconv_loop * HW_SIZE_256 * 8], [fp32_mid, 0]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [vconv_tail, 8],
                                                 "copy_ubuf_to_cbuf")
                        _addr = [[dst, cp_segment], [fp32_mid, vconv_tail - 16]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                 "copy_ubuf_to_cbuf")
                    else:
                        _addr = [[fp32_dst, vconv_loop * HW_SIZE_256 * 8],
                                 [fp32_mid, 0]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [vconv_tail, 8],
                                                 "copy_ubuf_to_ubuf")
                        _addr = [[fp32_dst, cp_segment],
                                 [fp32_mid, vconv_tail - 16]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                 "copy_ubuf_to_ubuf")
                else:
                    if is_input_in_ub is not True:
                        _addr = [[dst, cp_segment], [fp32_mid, HW_SIZE_256 * 8 - 16]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                 "copy_ubuf_to_cbuf")
                    else:
                        _addr = [[fp32_dst, cp_segment], [fp32_mid, HW_SIZE_256 * 8 - 16]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                 "copy_ubuf_to_ubuf")
                src_offset += cp_segment
                des_offset = cp_segment + 16
                with ib.if_scope(y_index != size_in[2] - 1):
                    with ib.for_range(0, vconv_loop) as vconv_loop_i:
                        _addr = [[fp16_mid_1, 0],
                                 [src, src_offset + vconv_loop_i * HW_SIZE_256 * 8]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [HW_SIZE_256 * 8, 16],
                                                 "copy_gm_to_ubuf")
                        _addr = [[fp32_mid_1, 0], [fp16_mid_1, 0]]
                        kernel_api.kernel_cast_to_fuc(ib, _addr,
                                                      [HW_SIZE_256 * 8, 8 * 8],
                                                      "vconv_f162f32")
                        if is_input_in_ub is not True:
                            _addr = [[dst, des_offset + vconv_loop_i * HW_SIZE_256 * 8],
                                     [fp32_mid_1, 0]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [HW_SIZE_256 * 8, 8],
                                                     "copy_ubuf_to_cbuf")
                        else:
                            _addr = [
                                [fp32_dst, des_offset + vconv_loop_i * HW_SIZE_256 * 8],
                                [fp32_mid_1, 0]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [HW_SIZE_256 * 8, 8],
                                                     "copy_ubuf_to_ubuf")
                    if vconv_tail != 0:
                        _addr = [[fp16_mid_1, 0],
                                 [src, src_offset + vconv_loop * HW_SIZE_256 * 8]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [vconv_tail, 16],
                                                 "copy_gm_to_ubuf")
                        _addr = [[fp32_mid_1, 0], [fp16_mid_1, 0]]
                        kernel_api.kernel_cast_to_fuc(ib, _addr,
                                                      [vconv_tail, 8 * 8],
                                                      "vconv_f162f32")
                        if is_input_in_ub is not True:
                            _addr = [[dst, des_offset + vconv_loop * HW_SIZE_256 * 8],
                                     [fp32_mid_1, 0]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [vconv_tail, 8],
                                                     "copy_ubuf_to_cbuf")
                            _addr = [[dst, 2 * cp_segment + 16],
                                     [fp32_mid_1, vconv_tail - 16]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                     "copy_ubuf_to_cbuf")
                        else:
                            _addr = [
                                [fp32_dst, des_offset + vconv_loop * HW_SIZE_256 * 8],
                                [fp32_mid_1, 0]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [vconv_tail, 8],
                                                     "copy_ubuf_to_ubuf")
                            _addr = [[fp32_dst, 2 * cp_segment + 16],
                                     [fp32_mid_1, vconv_tail - 16]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                     "copy_ubuf_to_ubuf")
                    else:
                        if is_input_in_ub is not True:
                            _addr = [[dst, 2 * cp_segment + 16],
                                     [fp32_mid_1, HW_SIZE_256 * 8 - 16]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                     "copy_ubuf_to_cbuf")
                        else:
                            _addr = [[fp32_dst, 2 * cp_segment + 16],
                                     [fp32_mid_1, HW_SIZE_256 * 8 - 16]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                     "copy_ubuf_to_ubuf")
                with ib.else_scope():
                    src_offset -= cp_segment
                    with ib.for_range(0, vconv_loop) as vconv_loop_i:
                        _addr = [[fp16_mid_1, 0],
                                 [src, src_offset + vconv_loop_i * HW_SIZE_256 * 8]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [HW_SIZE_256 * 8, 16],
                                                 "copy_gm_to_ubuf")
                        _addr = [[fp32_mid_1, 0], [fp16_mid_1, 0]]
                        kernel_api.kernel_cast_to_fuc(ib, _addr,
                                                      [HW_SIZE_256 * 8, 8 * 8],
                                                      "vconv_f162f32")
                        if is_input_in_ub is not True:
                            _addr = [[dst, des_offset + vconv_loop_i * HW_SIZE_256 * 8],
                                     [fp32_mid_1, 0]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [HW_SIZE_256 * 8, 8],
                                                     "copy_ubuf_to_cbuf")
                        else:
                            _addr = [
                                [fp32_dst, des_offset + vconv_loop_i * HW_SIZE_256 * 8],
                                [fp32_mid_1, 0]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [HW_SIZE_256 * 8, 8],
                                                     "copy_ubuf_to_ubuf")
                    if vconv_tail != 0:
                        _addr = [[fp16_mid_1, 0],
                                 [src, src_offset + vconv_loop * HW_SIZE_256 * 8]]
                        kernel_api.kernel_cp_fuc(ib, _addr, [vconv_tail, 16],
                                                 "copy_gm_to_ubuf")
                        _addr = [[fp32_mid_1, 0], [fp16_mid_1, 0]]
                        kernel_api.kernel_cast_to_fuc(ib, _addr,
                                                      [vconv_tail, 8 * 8],
                                                      "vconv_f162f32")
                        if is_input_in_ub is not True:
                            _addr = [[dst, des_offset + vconv_loop * HW_SIZE_256 * 8],
                                     [fp32_mid_1, 0]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [vconv_tail, 8],
                                                     "copy_ubuf_to_cbuf")
                            _addr = [[dst, 2 * cp_segment + 16],
                                     [fp32_mid_1, vconv_tail - 16]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                     "copy_ubuf_to_cbuf")
                        else:
                            _addr = [
                                [fp32_dst, des_offset + vconv_loop * HW_SIZE_256 * 8],
                                [fp32_mid_1, 0]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [vconv_tail, 8],
                                                     "copy_ubuf_to_ubuf")
                            _addr = [[fp32_dst, 2 * cp_segment + 16],
                                     [fp32_mid_1, vconv_tail - 16]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                     "copy_ubuf_to_ubuf")
                    else:
                        if is_input_in_ub is not True:
                            _addr = [[dst, 2 * cp_segment + 16],
                                     [fp32_mid_1, HW_SIZE_256 * 8 - 16]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                     "copy_ubuf_to_cbuf")
                        else:
                            _addr = [[fp32_dst, 2 * cp_segment + 16],
                                     [fp32_mid_1, HW_SIZE_256 * 8 - 16]]
                            kernel_api.kernel_cp_fuc(ib, _addr, [16, 8],
                                                     "copy_ubuf_to_ubuf")

        # pylint: disable=unused-variable
        def _data_unfold(part_index, part_data_len):
            if part_index == 0:
                x0x1_ub = ub_info["x0_512_x1_512_ub_unfold"]
                x2x3_ub = ub_info["x2_512_x3_512_ub_unfold"]
                pos_int_ub = ub_info["int32_512_ub"]
                index_align_index_num = part_data_len
                scale_offset = 0
            else:
                x0x1_ub = ub_info["x0_512_x1_512_ub_1"]
                x2x3_ub = ub_info["x2_512_x3_512_ub_1"]
                pos_int_ub = ub_info["int32_512_ub"]
                index_align_index_num = part_data_len
                scale_offset = 0
            if is_input_in_ub is True:
                src_addr = ub_info["ub_input"]
                copy_cmd = "copy_ubuf_to_ubuf"
            else:
                src_addr = l1_info["l1_input"]
                copy_cmd = "copy_cbuf_to_ubuf"

            index_align_loop = index_align_index_num // reg_count
            index_align_left = index_align_index_num % reg_count
            reg_list = [0, 1, 2, 3, 4, 5, 6, 7]

            if index_align_loop > 0:
                with ib.for_range(0, index_align_loop) as _index:
                    for reg_index in reg_list:
                        ib.emit(
                            tvm.call_extern(
                                "int32", "reg_mov",
                                tvm.call_extern(x_index_reg.dtype,
                                                "reg",
                                                pos_reg[reg_index + 2]),
                                pos_int_ub.access_ptr(
                                    "r",
                                    offset=(scale_offset + \
                                     (reg_index + _index * reg_count)) * 8),
                                0))
                    # copy HW_SIZE_512 to block segment
                    with ib.for_range(0, 8, name="reg_index") as reg_index:
                        ib.emit(
                            tvm.call_extern(
                                x0x1_ub.dtype, copy_cmd,
                                x0x1_ub.access_ptr(
                                    'w',
                                    offset=(reg_index + _index * reg_count)
                                    * 16),
                                src_addr.access_ptr(
                                    'r',
                                    offset=pos_reg[reg_index + 2] * c0),
                                0, 2, 2, 0, 510))
                    with ib.for_range(0, 8, name="reg_index") as reg_index:
                        ib.emit(
                            tvm.call_extern(
                                x2x3_ub.dtype, copy_cmd,
                                x2x3_ub.access_ptr(
                                    'w',
                                    offset=(reg_index + _index * reg_count)
                                    * 16),
                                src_addr.access_ptr(
                                    'r',
                                    offset=(pos_reg[reg_index + 2] + w_in + 1)
                                    * c0),
                                0, 2, 2, 0, 510))

            if index_align_left > 0:
                # pylint: disable=unnecessary-comprehension
                reg_list_left = [_x for _x in range(index_align_left)]
                for reg_index in reg_list_left:
                    ib.emit(
                        tvm.call_extern(
                            "int32", "reg_mov",
                            tvm.call_extern(x_index_reg.dtype,
                                            "reg",
                                            pos_reg[reg_index + 2]),
                            pos_int_ub.access_ptr(
                                "r",
                                offset=(reg_index + \
                                index_align_loop * reg_count) * 8),
                            0))
                    # copy HW_SIZE_512 to block segment
                with ib.for_range(0, index_align_left,
                                  name="reg_index") as reg_index:
                    ib.emit(
                        tvm.call_extern(
                            x0x1_ub.dtype, copy_cmd,
                            x0x1_ub.access_ptr(
                                'w',
                                offset=(reg_index + \
                                index_align_loop * reg_count) * 16),
                            src_addr.access_ptr(
                                'r',
                                offset=pos_reg[reg_index + 2] * c0),
                            0, 2, 2, 0, 510))
                with ib.for_range(0, index_align_left,
                                  name="reg_index") as reg_index:
                    ib.emit(
                        tvm.call_extern(
                            x2x3_ub.dtype, copy_cmd,
                            x2x3_ub.access_ptr(
                                'w',
                                offset=(reg_index + \
                                index_align_loop * reg_count) * 16),
                            src_addr.access_ptr(
                                'r',
                                offset=(pos_reg[reg_index + 2] + w_in + 1)
                                * c0),
                            0, 2, 2, 0, 510))

        def _process(part_index, index_block, y_index, x_index, output_offset):
            index_align_blcok_1 = index_block
            index_align_blcok_2 = index_block
            if part_index == 0:
                x0x1_ub_unfold = ub_info["x0_512_x1_512_ub_unfold"]
                x2x3_ub_unfold = ub_info["x2_512_x3_512_ub_unfold"]
                x0x1_ub = ub_info["x0_512_x1_512_ub"]
                x2x3_ub = ub_info["x2_512_x3_512_ub"]
                x_scale_512_ub = ub_info["x0_scale_x1_scale_512_ub"]
                y_scale_2_ub = ub_info["y0_scale_ub"]
                fp16_ub = ub_info["f16_512_1"]
                index_align_index_num = index_align_blcok_1
                index_align_blcok = index_align_blcok_1
                scale_offset = 0
            else:
                x0x1_ub_unfold = ub_info["x0_512_x1_512_ub_unfold"]
                x2x3_ub_unfold = ub_info["x2_512_x3_512_ub_unfold"]
                x0x1_ub = ub_info["x0_512_x1_512_ub_1"]
                x2x3_ub = ub_info["x2_512_x3_512_ub_1"]
                x_scale_512_ub = ub_info["x0_scale_x1_scale_512_ub"]
                y_scale_2_ub = ub_info["y0_scale_ub"]
                fp16_ub = ub_info["f16_512_1"]
                index_align_index_num = index_block - index_align_blcok_1
                index_align_blcok = index_align_blcok_2
                scale_offset = index_align_blcok_1

            _repeat = (index_align_blcok // 8 + 1) if index_align_blcok % 8 != 0 \
                else (index_align_blcok // 8 + 0)
            if index_block <= HW_SIZE_256:
                # x0*(1-Yv)  x1*(1-Yv)
                ib.emit(
                    tvm.call_extern(x0x1_ub.dtype, "vmul",
                                    x0x1_ub.access_ptr('w'),
                                    y_scale_2_ub.access_ptr('r', offset=0),
                                    x0x1_ub_unfold.access_ptr('r'),
                                    _repeat * 2, 1, 0, 1, 8, 0, 8))

                # x2*Yv  x3*Yv
                ib.emit(
                    tvm.call_extern(x2x3_ub.dtype, "vmul",
                                    x2x3_ub.access_ptr('w'),
                                    y_scale_2_ub.access_ptr('r', offset=8),
                                    x2x3_ub_unfold.access_ptr('r'),
                                    _repeat * 2, 1, 0, 1, 8, 0, 8))
                # x0*(1-Yv)  x1*(1-Yv)
                ib.emit(
                    tvm.call_extern(
                        x0x1_ub.dtype, "vmul",
                        x0x1_ub.access_ptr('w', offset=HW_SIZE_512 * 8),
                        y_scale_2_ub.access_ptr('r', offset=0),
                        x0x1_ub_unfold.access_ptr('r', offset=HW_SIZE_512 * 8),
                        _repeat * 2, 1, 0, 1, 8, 0, 8))
                # x2*Yv  x3*Yv
                ib.emit(
                    tvm.call_extern(
                        x2x3_ub.dtype, "vmul",
                        x2x3_ub.access_ptr('w', offset=HW_SIZE_512 * 8),
                        y_scale_2_ub.access_ptr('r', offset=8),
                        x2x3_ub_unfold.access_ptr('r', offset=HW_SIZE_512 * 8),
                        _repeat * 2, 1, 0, 1, 8, 0, 8))

                # x0 = x0*(1-Yv)*(1-Xu)     offset = 0
                ib.emit(
                    tvm.call_extern(
                        x0x1_ub.dtype, "vmul", x0x1_ub.access_ptr('w'),
                        x_scale_512_ub.access_ptr('r',
                                                  offset=scale_offset * 8),
                        x0x1_ub.access_ptr('r'), _repeat, 2, 1, 2, 16, 8, 16))

                # x2 = x2*Yv*(1-Xu)    offset = 0
                ib.emit(
                    tvm.call_extern(
                        x2x3_ub.dtype, "vmul", x2x3_ub.access_ptr('w'),
                        x_scale_512_ub.access_ptr('r',
                                                  offset=scale_offset * 8),
                        x2x3_ub.access_ptr('r'), _repeat, 2, 1, 2, 16, 8, 16))
                # x0 = x0*(1-Yv)*(1-Xu)  offset = 8
                ib.emit(
                    tvm.call_extern(
                        x0x1_ub.dtype, "vmul", x0x1_ub.access_ptr('w',
                                                                  offset=8),
                        x_scale_512_ub.access_ptr('r',
                                                  offset=scale_offset * 8),
                        x0x1_ub.access_ptr('r', offset=8), _repeat, 2, 1, 2,
                        16, 8, 16))

                ib.emit(
                    tvm.call_extern(
                        x2x3_ub.dtype, "vmul", x2x3_ub.access_ptr('w',
                                                                  offset=8),
                        x_scale_512_ub.access_ptr('r',
                                                  offset=scale_offset * 8),
                        x2x3_ub.access_ptr('r', offset=8), _repeat, 2, 1, 2,
                        16, 8, 16))

                ib.emit(
                    tvm.call_extern(
                        x0x1_ub.dtype, "vmul",
                        x0x1_ub.access_ptr('w', offset=HW_SIZE_512 * 8),
                        x_scale_512_ub.access_ptr(
                            'r',
                            offset=HW_SIZE_256 * 8 + scale_offset * 8),
                        x0x1_ub.access_ptr('r', offset=HW_SIZE_512 * 8), _repeat, 2, 1,
                        2, 16, 8, 16))

                ib.emit(
                    tvm.call_extern(
                        x2x3_ub.dtype, "vmul",
                        x2x3_ub.access_ptr('w', offset=HW_SIZE_512 * 8),
                        x_scale_512_ub.access_ptr(
                            'r',
                            offset=HW_SIZE_256 * 8 + scale_offset * 8),
                        x2x3_ub.access_ptr('r', offset=HW_SIZE_512 * 8), _repeat, 2, 1,
                        2, 16, 8, 16))

                ib.emit(
                    tvm.call_extern(
                        x0x1_ub.dtype, "vmul",
                        x0x1_ub.access_ptr('w', offset=HW_SIZE_512 * 8 + 8),
                        x_scale_512_ub.access_ptr(
                            'r',
                            offset=HW_SIZE_256 * 8 + scale_offset * 8),
                        x0x1_ub.access_ptr('r', offset=HW_SIZE_512 * 8 + 8), _repeat,
                        2, 1, 2, 16, 8, 16))

                ib.emit(
                    tvm.call_extern(
                        x2x3_ub.dtype, "vmul",
                        x2x3_ub.access_ptr('w', offset=HW_SIZE_512 * 8 + 8),
                        x_scale_512_ub.access_ptr(
                            'r',
                            offset=HW_SIZE_256 * 8 + scale_offset * 8),
                        x2x3_ub.access_ptr('r', offset=HW_SIZE_512 * 8 + 8), _repeat,
                        2, 1, 2, 16, 8, 16))
                # vadd x0+x1
                _addr_info = [[x0x1_ub, 0], [x0x1_ub, 0], [x0x1_ub, HW_SIZE_512 * 8]]
                kernel_api.kernel_two_to_one_common_fuc(
                    ib, _addr_info, [_repeat * 2 * 8 * 8, 8 * 8], "vadd")
                # vadd x2+x3
                _addr_info = [[x2x3_ub, 0], [x2x3_ub, 0], [x2x3_ub, HW_SIZE_512 * 8]]
                kernel_api.kernel_two_to_one_common_fuc(
                    ib, _addr_info, [_repeat * 2 * 8 * 8, 8 * 8], "vadd")

            # vadd x0+x2+x1+x3
            _addr_info = [[ub_info["out_f32"], 0], [x0x1_ub, 0], [x2x3_ub, 0]]
            kernel_api.kernel_two_to_one_common_fuc(
                ib, _addr_info, [_repeat * 2 * 8 * 8, 8 * 8], "vadd")
            out_ub = ub_info["out_f32"]
            out_offset = \
                (idx * h_per_core + y_index) * w_out * c0 + \
                x_index * HW_SIZE_256 * c0 + scale_offset * 16 + output_offset
            if dtype == "float161":  # do not change "float161", avoid to do fp32 to fp16
                # vconv out to fp16
                _addr_info = [[fp16_ub, 0], [out_ub, 0]]
                kernel_api.kernel_cast_to_fuc(ib, _addr_info,
                                              [_repeat * 2 * 8 * 8, 8 * 8],
                                              "vconv_f322f16")
                out_ub = fp16_ub
                # copy HW_SIZE_512 out to gm
                ib.emit(
                    tvm.call_extern("float16", "copy_ubuf_to_gm",
                                    outputs.access_ptr('w', offset=out_offset),
                                    out_ub.access_ptr('r'), 0, 1,
                                    index_align_index_num, 0, 0))
            else:
                ib.emit(
                    tvm.call_extern(outputs.dtype, "copy_ubuf_to_gm",
                                    outputs.access_ptr('w', offset=out_offset),
                                    out_ub.access_ptr('r'), 0, 1,
                                    index_align_index_num * 2, 0, 0))
                if dtype == "float16":
                    ib.emit(
                        tvm.call_extern('int32', 'pipe_barrier',
                                        args_str_v_all))

        # nc1_input_total
        with ib.for_range(0, nc1_input_total) as nc1_loop:
            src_nc1_input_offset = nc1_loop * nc1_input_offset
            src_nc1_output_offset = nc1_loop * nc1_output_offset
            with ib.for_range(0, x_segment_loop) as x_loop:
                pos_reg[1] = tvm.const(-1.0, dtype="int32")
                # pre copy fisrt line of input
                _addr_list = [[ub_info["int32_512_ub_y"], 0],
                              [l1_info["l1_ypos"], 0 * 8]]
                kernel_api.kernel_cp_fuc(ib, _addr_list, [8, 8],
                                         "copy_cbuf_to_ubuf")
                ib.emit(
                    tvm.call_extern(
                        "int32", "reg_mov",
                        tvm.call_extern(y_index_reg.dtype, "reg", y_index_reg),
                        ub_info["int32_512_ub_y"].access_ptr("r"), 0))
                images_offset = src_nc1_input_offset + y_index_reg * w_in * c0
                copy_data_len = w_in * c0 * 2
                with ib.if_scope(y_index_reg < h_in):
                    _copy_input_to_l1(
                        inputs, images_offset, l1_info["l1_input"],
                        copy_data_len,
                        [
                            range_512, range_512, ub_info["f16_512_1"],
                            ub_info["f16_512_1"]
                        ], y_index_reg)
                # copy x pos int from l1 to ub
                _addr_list = [[ub_info["int32_512_ub"], 0],
                              [l1_info["l1_xpos"], x_loop * HW_SIZE_256 * 8]]
                kernel_api.kernel_cp_fuc(ib, _addr_list, [HW_SIZE_256 * 8, 8],
                                         "copy_cbuf_to_ubuf")
                # copy x pos float from l1 to ub
                _addr_list = [[ub_info["x0_scale_x1_scale_512_ub"], 0],
                              [l1_info["l1_xscale"], x_loop * HW_SIZE_512 * 8]]
                kernel_api.kernel_cp_fuc(ib, _addr_list, [HW_SIZE_512 * 8, 8],
                                         "copy_cbuf_to_ubuf")
                with ib.for_range(0, h_per_core) as y_loop:
                    with ib.if_scope(idx * h_per_core + y_loop < h_out):
                        # copy y pos int from l1 to ub
                        # copy y sclace float from l1 to ub
                        ib.emit(
                            tvm.call_extern(
                                ub_info["y0_scale_ub"].dtype,
                                "copy_cbuf_to_ubuf",
                                ub_info["y0_scale_ub"].access_ptr('w',
                                                                  offset=0),
                                l1_info["l1_yscale"].access_ptr(
                                    'r', offset=y_loop * 8), 0, 2, 1, 255, 0))
                        # if y_before_index_reg != y_index_reg
                        # copy from gm to ub or l1 and open data to x1 x2 x3 x4
                        with ib.if_scope(pos_reg[1] != y_index_reg):
                            pos_reg[1] = y_index_reg
                            _data_unfold(0, HW_SIZE_256)

                        _addr_list = [[ub_info["int32_512_ub_y"], 0],
                                      [l1_info["l1_ypos"], y_loop * 8]]
                        kernel_api.kernel_cp_fuc(ib, _addr_list, [16, 8],
                                                 "copy_cbuf_to_ubuf")
                        # copy next unflod data if next y != cu_y
                        ib.emit(
                            tvm.call_extern(
                                "int32", "reg_mov",
                                tvm.call_extern(y_index_reg.dtype, "reg",
                                                y_index_reg),
                                ub_info["int32_512_ub_y"].access_ptr("r",
                                                                     offset=8),
                                0))
                        with ib.if_scope(
                                tvm.all(y_index_reg != pos_reg[1],
                                        y_loop < (h_per_core - 1),
                                        y_index_reg < h_in)):
                            images_offset = src_nc1_input_offset + y_index_reg * w_in * c0
                            _copy_input_to_l1(
                                inputs, images_offset, l1_info["l1_input"],
                                copy_data_len, [
                                    range_512, range_512, ub_info["f16_512_1"],
                                    ub_info["f16_512_1"]
                                ], y_index_reg)
                        _process(0, HW_SIZE_256, y_loop, x_loop, src_nc1_output_offset)

            if x_segment_tail != 0:
                pos_reg[1] = tvm.const(-1.0, dtype="int32")
                # pre copy fisrt line of input
                _addr_list = [[ub_info["int32_512_ub_y"], 0],
                              [l1_info["l1_ypos"], 0 * 8]]
                kernel_api.kernel_cp_fuc(ib, _addr_list, [8, 8],
                                         "copy_cbuf_to_ubuf")
                ib.emit(
                    tvm.call_extern(
                        "int32", "reg_mov",
                        tvm.call_extern(y_index_reg.dtype, "reg", y_index_reg),
                        ub_info["int32_512_ub_y"].access_ptr("r"), 0))
                images_offset = src_nc1_input_offset + y_index_reg * w_in * c0
                copy_data_len = w_in * c0 * 2
                with ib.if_scope(y_index_reg < h_in):
                    _copy_input_to_l1(
                        inputs, images_offset, l1_info["l1_input"],
                        copy_data_len,
                        [
                            range_512, range_512, ub_info["f16_512_1"],
                            ub_info["f16_512_1"]
                        ], y_index_reg)
                _addr_list = [[ub_info["int32_512_ub"], 0],
                              [l1_info["l1_xpos"], x_segment_loop * HW_SIZE_256 * 8]]
                kernel_api.kernel_cp_fuc(ib, _addr_list,
                                         [x_segment_tail * 8, 8],
                                         "copy_cbuf_to_ubuf")
                # copy x pos float from l1 to ub
                _addr_list = [[ub_info["x0_scale_x1_scale_512_ub"], 0],
                              [l1_info["l1_xscale"], x_segment_loop * HW_SIZE_512 * 8]]
                kernel_api.kernel_cp_fuc(ib, _addr_list, [HW_SIZE_512 * 8, 8],
                                         "copy_cbuf_to_ubuf")
                with ib.for_range(0, h_per_core) as y_loop_tail:
                    with ib.if_scope(idx * h_per_core + y_loop_tail < h_out):
                        # copy y pos int from l1 to ub
                        # copy y sclace float from l1 to ub
                        ib.emit(
                            tvm.call_extern(
                                ub_info["y0_scale_ub"].dtype,
                                "copy_cbuf_to_ubuf",
                                ub_info["y0_scale_ub"].access_ptr('w',
                                                                  offset=0),
                                l1_info["l1_yscale"].access_ptr(
                                    'r',
                                    offset=y_loop_tail * 8), 0, 2, 1, 255, 0))
                        # if y_before_index_reg != y_index_reg
                        # copy from gm to ub or l1 and open data to x1 x2 x3 x4
                        with ib.if_scope(pos_reg[1] != y_index_reg):
                            pos_reg[1] = y_index_reg
                            _data_unfold(0, x_segment_tail)

                        _addr_list = [[ub_info["int32_512_ub_y"], 0],
                                      [l1_info["l1_ypos"], y_loop_tail * 8]]
                        kernel_api.kernel_cp_fuc(ib, _addr_list, [16, 8],
                                                 "copy_cbuf_to_ubuf")
                        # copy next unflod data if next y != cu_y
                        ib.emit(
                            tvm.call_extern(
                                "int32", "reg_mov",
                                tvm.call_extern(y_index_reg.dtype, "reg",
                                                y_index_reg),
                                ub_info["int32_512_ub_y"].access_ptr("r",
                                                                     offset=8),
                                0))
                        with ib.if_scope(
                                tvm.all(y_index_reg != pos_reg[1],
                                        y_loop_tail < (h_per_core - 1),
                                        y_index_reg < h_in)):
                            images_offset = src_nc1_input_offset + y_index_reg * w_in * c0
                            _copy_input_to_l1(
                                inputs, images_offset, l1_info["l1_input"],
                                copy_data_len, [
                                    range_512, range_512, ub_info["f16_512_1"],
                                    ub_info["f16_512_1"]
                                ], y_index_reg)

                        # caculate data
                        _process(0, x_segment_tail, y_loop_tail,
                                 x_segment_loop, src_nc1_output_offset)

    return ib.get()


@fusion_manager.register("resize_bilinear_d")
# pylint: disable=unused-argument
def resize_bilinear_v2_d_compute(images,
                                 y,
                                 size,
                                 align_corners=False,
                                 half_pixel_centers=False,
                                 kernel_name="resize_bilinear_v2"):
    """
    resize_bilinear schedule compute part

    Parameters
    ----------
    images: TVM tensor
        the placeholders of images value
    y: dict
        dict info of output value
    size: list
        the shape of output about 'new_height, new_width'
    align_corners: bool
        If true, the centers of the 4 corner pixels
        of the input and output tensors are aligned
    half_pixel_centers: bool
        whether half_pixel_centers
    kernel_name: str
        cce kernel name

    returns
    -------
    res
    """
    images_shape = te.lang.cce.util.shape_to_list(images.shape)
    shape_out = list(images_shape)
    shape_out[-2] = size[-1]
    shape_out[-3] = size[-2]
    util.check_tensor_shape_size(shape_out)

    res = tvm.extern(
        tuple(shape_out), [images],
        lambda ins, outs: _resize_bilinear_ir(ins[0], outs[0], align_corners,
                                              half_pixel_centers),
        name="res",
        dtype="float32")
    return res


# pylint: disable=too-many-arguments
@util.check_input_type(dict, dict, (tuple, list), bool, bool, str)
def resize_bilinear_v2_d(images,
                         y,
                         size,
                         align_corners=False,
                         half_pixel_centers=False,
                         kernel_name="resize_bilinear_v2"):
    """
    resize_bilinear v2 schedule main part

    Parameters
    ----------
    images: dict
        dict info of images value, must include the keys(shape and dtype).
        and shape will be 5HD
    y: dict
        dict info of output value
    size: list
        the shape of output about 'new_height, new_width'
    align_corners: bool
        If true, the centers of the 4 corner pixels
        of the input and output tensors are aligned
    half_pixel_centers: bool
        whether half_pixel_centers
    kernel_name: str
        cce kernel name, default value is "resize_bilinear_v2"

    returns
    -------
    None
    """
    util.check_kernel_name(kernel_name)
    image_dtype = images.get("dtype")
    image_shape = images.get("shape")

    util.check_shape_rule(image_shape)
    util.check_shape_rule(size)
    check_list = ["float16", "float32"]
    if image_dtype not in check_list:
        raise RuntimeError("only support %s while dtype is %s" %
                           (str(check_list), image_dtype))
    if len(image_shape) != 5:
        raise RuntimeError("The ndim of input iamge shape must be 5!")
    if len(size) != 2:
        raise RuntimeError("The ndim of size must be 2!")
    if (image_shape[2] > HW_SIZE_2048 or image_shape[3] > HW_SIZE_2048) or \
            (size[0] > HW_SIZE_2048 or size[1] > HW_SIZE_2048):
        raise RuntimeError("in or out h/w size should be less than 2048!")
    if align_corners is True and half_pixel_centers is True:
        raise RuntimeError("If half_pixel_centers is True, "
                           "align_corners must be False.")

    util.check_tensor_shape_size(image_shape)

    image_data = tvm.placeholder(image_shape,
                                 dtype=image_dtype,
                                 name="image_data")
    res = resize_bilinear_v2_d_compute(image_data, y, size, align_corners,
                                       half_pixel_centers,
                                       kernel_name)

    # build & output
    sch = tvm.create_schedule(res.op)
    with build_config:
        tvm.build(sch, [image_data, res], "cce", name=kernel_name)
