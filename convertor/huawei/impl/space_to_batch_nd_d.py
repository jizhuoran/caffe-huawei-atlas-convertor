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

space_to_batch_nd
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from functools import reduce as functools_reduce
import te.lang.cce
from te import tvm
from te import platform as tbe_platform
from te.platform.cce_build import build_config
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_intrin_md import reset_mask_insn
from topi.cce import util


# pylint: disable=too-many-instance-attributes,useless-object-inheritance
class _SpaceToBatch(object):
    """init parameters for space_to_batch."""

    def __init__(self, shape, dtype, block_shape, paddings):
        self.shape = shape
        self.dtype = dtype
        self.batch = self.shape[0]
        self.channel1 = self.shape[1]
        self.input_height = self.shape[2]
        self.input_width = self.shape[3]
        self.channel0 = self.shape[4]

        self.pad_top = paddings[0][0]
        self.pad_bottom = paddings[0][1]
        self.pad_left = paddings[1][0]
        self.pad_right = paddings[1][1]
        self.pad_height = self.pad_top + self.pad_bottom
        self.pad_width = self.pad_left + self.pad_right
        self.padded_height = self.input_height + self.pad_height
        self.padded_width = self.input_width + self.pad_width

        self.block_height = block_shape[0]
        self.block_width = block_shape[1]
        self.block_size = self.block_height * self.block_width

        self.output_height = self.padded_height // self.block_height
        self.output_width = self.padded_width // self.block_width

        self.padded_shape = [
            self.batch, self.channel1, self.padded_height, self.padded_width,
            self.channel0
        ]

        self.output_shape = [
            self.batch * self.block_size, self.channel1, self.output_height,
            self.output_width, self.channel0
        ]

        self.tile_shape = [
            self.block_height, self.output_height, self.block_width,
            self.output_width, self.channel0
        ]

    def tile_axis(self):
        """calculate tile axis."""
        ub_size = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.UB_SIZE)
        dtype_size = tbe_platform.cce_intrin.get_bit_len(self.dtype) // 8
        total_cnt = ub_size // dtype_size // 2

        tile_axis = 1
        for i, _ in enumerate(self.tile_shape):
            if i > 0:
                ele_cnt = functools_reduce(lambda x, y: x * y,
                                           self.tile_shape[i:])
                if total_cnt // ele_cnt > 0:
                    tile_axis = i
                    break

        return tile_axis

    def new_alloc(self, i_b, shape, name, scope):
        """decl new buffer."""
        buf_var = i_b.allocate(self.dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(
            shape, buf_var.dtype, name=name, scope=scope, data=buf_var)

        return new_buffer

    def vector_dup(self, i_b, buf, size):
        """set buffer to all zero."""
        one_cnt = 128
        if self.dtype == "float32":
            one_cnt = 64
        repeat = size // one_cnt
        remainder = size % one_cnt
        loop_repeat = repeat // 255
        loop_remainder = repeat % 255

        with i_b.if_scope(loop_repeat > 0):
            with i_b.for_range(0, loop_repeat) as i:
                mask = one_cnt
                offset_repeat = i * one_cnt * 255
                reset_mask_insn(i_b, self.dtype, bits=mask)
                i_b.emit(
                    tvm.call_extern(self.dtype, "vector_dup",
                                    buf.access_ptr("w", offset=offset_repeat),
                                    tvm.const(0, dtype=self.dtype), 255, 1, 1,
                                    8, 8))

        offset_remainder = loop_repeat * one_cnt * 255
        with i_b.if_scope(loop_remainder > 0):
            mask = one_cnt
            reset_mask_insn(i_b, self.dtype, bits=mask)
            i_b.emit(
                tvm.call_extern(self.dtype, "vector_dup",
                                buf.access_ptr("w", offset=offset_remainder),
                                tvm.const(0, dtype=self.dtype), loop_remainder,
                                1, 1, 8, 8))

        offset = one_cnt * loop_remainder + loop_repeat * one_cnt * 255
        with i_b.if_scope(remainder > 0):
            mask = remainder
            reset_mask_insn(i_b, self.dtype, bits=mask)
            i_b.emit(
                tvm.call_extern(self.dtype, "vector_dup",
                                buf.access_ptr("w", offset=offset),
                                tvm.const(0, dtype=self.dtype), 1, 1, 1, 8, 8))


# pylint: disable=unused-argument,too-many-statements,too-many-locals
def _kernel_ir(dst, src, space):
    """space_to_batch kernel."""
    i_b = tvm.ir_builder.create()
    input_data = src[0]
    output_data = dst[0]
    dtype = space.dtype
    pad_t = space.pad_top
    pad_l = space.pad_left
    block_h = space.block_height
    block_w = space.block_width
    pad_w = space.pad_width
    input_h = space.input_height
    input_w = space.input_width
    padded_w = space.padded_width
    output_h = space.output_height
    output_w = space.output_width
    c_0 = space.channel0
    num = space.batch * space.channel1
    num_outer = num
    num_inner = 1
    if num > 65535:
        for i in reversed(list(range(1, 65535))):
            if num % i == 0:
                num_outer = i
                num_inner = num // i
                break
    block_idx = tvm.thread_axis("blockIdx.x")
    i_b.scope_attr(block_idx, "thread_extent", num_outer)
    var = block_idx.var
    burst = 1
    if dtype == "float32":
        burst = 2

    tile_axis = space.tile_axis()
    if tile_axis == 1 and input_w * (
            block_h - 1) * burst <= 65535 and pad_w * burst <= 65535 and (
                block_w - 1) * burst <= 65535:
        size = output_h * padded_w * c_0
        pad_data = space.new_alloc(
            i_b, [size], name="pad_data", scope=tbe_platform.scope_ubuf)
        pad_data2 = space.new_alloc(
            i_b, [size], name="pad_data2", scope=tbe_platform.scope_ubuf)

        divisor = block_h // 2
        remainder = block_h % 2
        with i_b.for_range(0, num_inner) as n_i:
            with i_b.for_range(0, divisor) as b_h:
                # set buffer to all zero
                space.vector_dup(i_b, pad_data, size)

                # move data from GM to UB
                start = (pad_t - b_h + block_h - 1) // block_h
                end = (pad_t + input_h - b_h + block_h - 1) // block_h
                offset_base = (var * num_inner + n_i) * input_h * input_w * c_0
                offset_src = (b_h + start * block_h -
                              pad_t) * input_w * c_0 + offset_base
                offset_dst = start * padded_w * c_0 + pad_l * c_0
                with i_b.if_scope(end - start > 0):
                    i_b.emit(
                        tvm.call_extern(
                            dtype, "copy_gm_to_ubuf",
                            pad_data.access_ptr("w", offset=offset_dst),
                            input_data.access_ptr("r", offset=offset_src), 0,
                            end - start, input_w * burst,
                            input_w * (block_h - 1) * burst, pad_w * burst))

                # move data from UB to GM
                with i_b.for_range(0, block_w) as b_w:
                    offset_out = (
                        (b_h * block_w + b_w) * num +
                        (var * num_inner + n_i)) * output_h * output_w * c_0
                    i_b.emit(
                        tvm.call_extern(
                            dtype, "copy_ubuf_to_gm",
                            output_data.access_ptr("w", offset=offset_out),
                            pad_data.access_ptr("r", offset=b_w * c_0), 0,
                            output_h * output_w, burst, (block_w - 1) * burst,
                            0))

                # set buffer to all zero
                space.vector_dup(i_b, pad_data2, size)

                # move data from GM to UB
                start = (pad_t - (b_h + divisor) + block_h - 1) // block_h
                end = (pad_t + input_h -
                       (b_h + divisor) + block_h - 1) // block_h
                offset_base = (var * num_inner + n_i) * input_h * input_w * c_0
                offset_src = ((b_h + divisor) + start * block_h -
                              pad_t) * input_w * c_0 + offset_base
                offset_dst = start * padded_w * c_0 + pad_l * c_0
                with i_b.if_scope(end - start > 0):
                    i_b.emit(
                        tvm.call_extern(
                            dtype, "copy_gm_to_ubuf",
                            pad_data2.access_ptr("w", offset=offset_dst),
                            input_data.access_ptr("r", offset=offset_src), 0,
                            end - start, input_w * burst,
                            input_w * (block_h - 1) * burst, pad_w * burst))

                # move data from UB to GM
                with i_b.for_range(0, block_w) as b_w:
                    offset_out = (
                        ((b_h + divisor) * block_w + b_w) * num +
                        (var * num_inner + n_i)) * output_h * output_w * c_0
                    i_b.emit(
                        tvm.call_extern(
                            dtype, "copy_ubuf_to_gm",
                            output_data.access_ptr("w", offset=offset_out),
                            pad_data2.access_ptr("r", offset=b_w * c_0), 0,
                            output_h * output_w, burst, (block_w - 1) * burst,
                            0))

            if remainder != 0:
                # set buffer to all zero
                space.vector_dup(i_b, pad_data, size)

                # move data from GM to UB
                start = pad_t // block_h
                end = (pad_t + input_h) // block_h
                offset_base = (var * num_inner + n_i) * input_h * input_w * c_0
                offset_src = ((block_h - 1) + start * block_h -
                              pad_t) * input_w * c_0 + offset_base
                offset_dst = start * padded_w * c_0 + pad_l * c_0
                with i_b.if_scope(end - start > 0):
                    i_b.emit(
                        tvm.call_extern(
                            dtype, "copy_gm_to_ubuf",
                            pad_data.access_ptr("w", offset=offset_dst),
                            input_data.access_ptr("r", offset=offset_src), 0,
                            end - start, input_w * burst,
                            input_w * (block_h - 1) * burst, pad_w * burst))

                # move data from UB to GM
                with i_b.for_range(0, block_w) as b_w:
                    offset_out = (
                        ((block_h - 1) * block_w + b_w) * num +
                        (var * num_inner + n_i)) * output_h * output_w * c_0
                    i_b.emit(
                        tvm.call_extern(
                            dtype, "copy_ubuf_to_gm",
                            output_data.access_ptr("w", offset=offset_out),
                            pad_data.access_ptr("r", offset=b_w * c_0), 0,
                            output_h * output_w, burst, (block_w - 1) * burst,
                            0))
    elif tile_axis in (1, 2) and (block_w - 1) * burst <= 65535:
        size = padded_w * c_0
        pad_data = space.new_alloc(
            i_b, [size], name="pad_data", scope=tbe_platform.scope_ubuf)

        with i_b.for_range(0, num_inner) as n_i:
            with i_b.for_range(0, block_h) as b_h:
                with i_b.for_range(0, output_h) as o_h:
                    # set buffer to all zero
                    space.vector_dup(i_b, pad_data, size)

                    # move data from GM to UB
                    with i_b.if_scope(
                            tvm.all(o_h * block_h + b_h >= pad_t,
                                    o_h * block_h + b_h < input_h + pad_t)):
                        offset_base = (var * num_inner +
                                       n_i) * input_h * input_w * c_0
                        offset_src = (o_h * block_h + b_h -
                                      pad_t) * input_w * c_0 + offset_base
                        i_b.emit(
                            tvm.call_extern(
                                dtype, "copy_gm_to_ubuf",
                                pad_data.access_ptr("w", offset=pad_l * c_0),
                                input_data.access_ptr("r", offset=offset_src),
                                0, 1, input_w * burst, 0, 0))

                    # move data from UB to GM
                    with i_b.for_range(0, block_w) as b_w:
                        offset_out = (((b_h * block_w + b_w) * num +
                                       (var * num_inner + n_i)) * output_h +
                                      o_h) * output_w * c_0
                        i_b.emit(
                            tvm.call_extern(
                                dtype, "copy_ubuf_to_gm",
                                output_data.access_ptr("w", offset=offset_out),
                                pad_data.access_ptr("r", offset=b_w * c_0), 0,
                                output_w, burst, (block_w - 1) * burst, 0))
    elif tile_axis == 3 and (block_w - 1) * burst <= 65535:
        size = output_w * c_0
        pad_data = space.new_alloc(
            i_b, [size], name="pad_data", scope=tbe_platform.scope_ubuf)

        with i_b.for_range(0, num_inner) as n_i:
            with i_b.for_range(0, block_h) as b_h:
                with i_b.for_range(0, output_h) as o_h:
                    with i_b.for_range(0, block_w) as b_w:
                        # set buffer to all zero
                        space.vector_dup(i_b, pad_data, size)

                        # move data from GM to UB
                        with i_b.if_scope(
                                tvm.all(o_h * block_h + b_h >= pad_t,
                                        o_h * block_h + b_h < input_h + pad_t)):
                            start = (pad_l - b_w + block_w - 1) // block_w
                            end = (pad_l + input_w - b_w + block_w -
                                   1) // block_w
                            offset_base = (var * num_inner +
                                           n_i) * input_h * input_w * c_0
                            offset_src = (o_h * block_h + b_h -
                                          pad_t) * input_w * c_0 + (
                                              b_w + start * block_w -
                                              pad_l) * c_0 + offset_base
                            with i_b.if_scope(end - start > 0):
                                i_b.emit(
                                    tvm.call_extern(
                                        dtype, "copy_gm_to_ubuf",
                                        pad_data.access_ptr(
                                            "w", offset=start * c_0),
                                        input_data.access_ptr(
                                            "r",
                                            offset=offset_src), 0, end - start,
                                        burst, (block_w - 1) * burst, 0))

                        # move data from UB to GM
                        offset_out = (((b_h * block_w + b_w) * num +
                                       (var * num_inner + n_i)) * output_h +
                                      o_h) * output_w * c_0
                        i_b.emit(
                            tvm.call_extern(
                                dtype, "copy_ubuf_to_gm",
                                output_data.access_ptr("w", offset=offset_out),
                                pad_data.access_ptr("r", offset=0), 0, output_w,
                                burst, 0, 0))
    else:
        size = c_0
        pad_data = space.new_alloc(
            i_b, [size], name="pad_data", scope=tbe_platform.scope_ubuf)

        with i_b.for_range(0, num_inner) as n_i:
            with i_b.for_range(0, block_h) as b_h:
                with i_b.for_range(0, output_h) as o_h:
                    with i_b.for_range(0, block_w) as b_w:
                        with i_b.for_range(0, output_w) as o_w:
                            # set buffer to all zero
                            space.vector_dup(i_b, pad_data, size)

                            # move data from GM to UB
                            with i_b.if_scope(
                                    tvm.all(
                                        o_h * block_h + b_h >= pad_t,
                                        o_h * block_h + b_h < input_h + pad_t,
                                        o_w * block_w + b_w >= pad_l,
                                        o_w * block_w + b_w < input_w + pad_l)):
                                offset_base = (var * num_inner +
                                               n_i) * input_h * input_w * c_0
                                offset_src = (o_h * block_h + b_h -
                                              pad_t) * input_w * c_0 + (
                                                  o_w * block_w + b_w -
                                                  pad_l) * c_0 + offset_base
                                i_b.emit(
                                    tvm.call_extern(
                                        dtype, "copy_gm_to_ubuf",
                                        pad_data.access_ptr("w", offset=0),
                                        input_data.access_ptr(
                                            "r", offset=offset_src), 0, 1,
                                        burst, 0, 0))

                            # move data from UB to GM
                            offset_out = ((
                                ((b_h * block_w + b_w) * num +
                                 (var * num_inner + n_i)) * output_h + o_h) *
                                          output_w + o_w) * c_0
                            i_b.emit(
                                tvm.call_extern(
                                    dtype, "copy_ubuf_to_gm",
                                    output_data.access_ptr(
                                        "w", offset=offset_out),
                                    pad_data.access_ptr("r", offset=0), 0, 1,
                                    burst, 0, 0))

    return i_b.get()


def _check_parms(shape, dtype, block_shape, paddings, kernel_name):
    """check the parameters including shape, dtype, block_shape, paddings
    and kernel_name.
    """
    dtype_list = ("float16", "float32")

    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_dtype_rule(dtype, dtype_list)
    util.check_kernel_name(kernel_name)

    if len(shape) != 5:
        raise RuntimeError("the shape of image_input should be 5, "
                           "but got: %d" % len(shape))

    if len(block_shape) != 2:
        raise RuntimeError("the shape of block_shape should be 2, "
                           "but got: %d" % len(block_shape))

    if len(paddings) != 2 or len(paddings[0]) != 2 or len(paddings[1]) != 2:
        raise RuntimeError("the shape of paddings should be 2x2")

    if not (isinstance(block_shape[0], int) and isinstance(block_shape[1], int)
            and block_shape[0] > 0 and block_shape[1] > 0):
        raise RuntimeError(
            "the value of block_shape should be integer and be greater to 0")

    if not (isinstance(paddings[0][0], int) and paddings[0][0] >= 0 and
            isinstance(paddings[0][1], int) and paddings[0][1] >= 0 and
            isinstance(paddings[1][0], int) and paddings[1][0] >= 0 and
            isinstance(paddings[1][1], int) and paddings[1][1] >= 0):
        raise RuntimeError("the value of paddings should be integer and "
                           "be greater or equal to 0")

    if (shape[2] + paddings[0][0] + paddings[0][1]) % block_shape[0] != 0:
        raise RuntimeError(
            "paddings height should be exactly divisible by block height")
    if (shape[3] + paddings[1][0] + paddings[1][1]) % block_shape[1] != 0:
        raise RuntimeError(
            "paddings width should be exactly divisible by block width")


# pylint: disable=invalid-name
@fusion_manager.register("space_to_batch_nd_d")
def space_to_batch_nd_d_compute(x,
                                y,
                                block_shape,
                                paddings,
                                kernel_name="space_to_batch_nd_d"):
    """space_to_batch for tensor.

    Parameters
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    block_shape: list or tuple
        1-D with shape [2].
    crops: list or tuple
        2-D with shape [2, 2], paddings[i] = [pad_start, pad_end]
    kernel_name: str
        cce kernel name, default value is "space_to_batch_nd_d".

    Returns
    -------
    res: TVM tensor
        output tensor.
    """
    shape = te.lang.cce.util.shape_to_list(x.shape)
    dtype = x.dtype

    if len(paddings) == 4:
        paddings = [[paddings[0], paddings[1]], [paddings[2], paddings[3]]]

    space = _SpaceToBatch(shape, dtype, block_shape, paddings)

    res = tvm.extern([space.output_shape], [x],
                     lambda ins, outs: _kernel_ir(outs, ins, space),
                     dtype=dtype,
                     name="res")

    return res


@util.check_input_type(dict, dict, (list, tuple), (list, tuple), str)
def space_to_batch_nd_d(x,
                        y,
                        block_shape,
                        paddings,
                        kernel_name="space_to_batch_nd_d"):
    """space_to_batch for tensor.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    block_shape: list or tuple
        1-D with shape [2].
    paddings: list or tuple
        2-D with shape [2, 2], paddings[i] = [pad_start, pad_end]
    kernel_name: str
        cce kernel name, default value is "space_to_batch_nd_d".

    Returns
    -------
    None.
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()

    if len(paddings) == 4:
        paddings = [[paddings[0], paddings[1]], [paddings[2], paddings[3]]]

    _check_parms(shape, dtype, block_shape, paddings, kernel_name)

    data = tvm.placeholder(shape, name="data", dtype=dtype)

    res = space_to_batch_nd_d_compute(data, y, block_shape, paddings,
                                      kernel_name)

    sch = tvm.create_schedule(res.op)

    with build_config:
        tvm.build(sch, [data, res], "cce", name=kernel_name)
