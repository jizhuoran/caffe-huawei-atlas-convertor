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

batch_to_space_nd
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
from topi.cce import util


# pylint: disable=too-many-instance-attributes,useless-object-inheritance
class _BatchToSpace(object):
    """init parameters for batch_to_space."""

    def __init__(self, shape, dtype, block_shape, crops):
        self.shape = shape
        self.dtype = dtype
        self.batch = self.shape[0]
        self.channel1 = self.shape[1]
        self.input_height = self.shape[2]
        self.input_width = self.shape[3]
        self.channel0 = self.shape[4]

        self.crop_top = crops[0][0]
        self.crop_bottom = crops[0][1]
        self.crop_left = crops[1][0]
        self.crop_right = crops[1][1]
        self.crop_height = self.crop_top + self.crop_bottom
        self.crop_width = self.crop_left + self.crop_right

        self.block_height = block_shape[0]
        self.block_width = block_shape[1]
        self.block_size = self.block_height * self.block_width
        self.padded_height = self.input_height * self.block_height
        self.padded_width = self.input_width * self.block_width
        self.ibatch = self.batch // self.block_height // self.block_width

        self.output_height = self.padded_height - self.crop_height
        self.output_width = self.padded_width - self.crop_width

        self.permute_shape = [
            self.ibatch, self.channel1, self.padded_height, self.padded_width,
            self.channel0
        ]
        self.output_shape = [
            self.ibatch, self.channel1, self.output_height, self.output_width,
            self.channel0
        ]
        self.tile_shape = [
            self.block_height, self.input_height, self.block_width,
            self.input_width, self.channel0
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


# pylint: disable=unused-argument,too-many-statements,too-many-locals
def _kernel_ir(dst, src, batch):
    """batch_to_space kernel."""
    i_b = tvm.ir_builder.create()
    input_data = src[0]
    output_data = dst[0]
    dtype = batch.dtype
    crop_t = batch.crop_top
    crop_l = batch.crop_left
    block_h = batch.block_height
    block_w = batch.block_width
    crop_w = batch.crop_width
    input_h = batch.input_height
    input_w = batch.input_width
    padded_w = batch.padded_width
    output_h = batch.output_height
    output_w = batch.output_width
    c_0 = batch.channel0
    num = batch.ibatch * batch.channel1
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

    tile_axis = batch.tile_axis()
    if tile_axis == 1 and output_w * (block_h - 1) * burst <= 65535 and (
            block_w - 1) * burst <= 65535 and crop_w * burst <= 65535:
        size = input_h * padded_w * c_0
        padded_data = batch.new_alloc(
            i_b, [size], name="padded_data", scope=tbe_platform.scope_ubuf)

        padded_data2 = batch.new_alloc(
            i_b, [size], name="padded_data2", scope=tbe_platform.scope_ubuf)

        divisor = block_h // 2
        remainder = block_h % 2
        with i_b.for_range(0, num_inner) as n_i:
            with i_b.for_range(0, divisor) as b_h:
                # move data from GM to UB
                with i_b.for_range(0, block_w) as b_w:
                    offset_src = (
                        (b_h * block_w + b_w) * num +
                        (var * num_inner + n_i)) * input_h * input_w * c_0
                    offset_dst = b_w * c_0
                    i_b.emit(
                        tvm.call_extern(
                            dtype, "copy_gm_to_ubuf",
                            padded_data.access_ptr('w', offset=offset_dst),
                            input_data.access_ptr('r', offset=offset_src), 0,
                            input_h * input_w, burst, 0, (block_w - 1) * burst))

                # move data from UB to GM
                start = (crop_t - b_h + block_h - 1) // block_h
                end = (crop_t + output_h - b_h + block_h - 1) // block_h
                offset_base = (var * num_inner +
                               n_i) * output_h * output_w * c_0
                offset_out = (b_h + start * block_h -
                              crop_t) * output_w * c_0 + offset_base
                offset_pad = start * padded_w * c_0 + crop_l * c_0
                with i_b.if_scope(end - start > 0):
                    i_b.emit(
                        tvm.call_extern(
                            dtype, "copy_ubuf_to_gm",
                            output_data.access_ptr('w', offset=offset_out),
                            padded_data.access_ptr('r', offset=offset_pad), 0,
                            end - start, output_w * burst, crop_w * burst,
                            output_w * (block_h - 1) * burst))

                # move data from GM to UB
                with i_b.for_range(0, block_w) as b_w:
                    offset_src = (
                        ((b_h + divisor) * block_w + b_w) * num +
                        (var * num_inner + n_i)) * input_h * input_w * c_0
                    offset_dst = b_w * c_0
                    i_b.emit(
                        tvm.call_extern(
                            dtype, "copy_gm_to_ubuf",
                            padded_data2.access_ptr('w', offset=offset_dst),
                            input_data.access_ptr('r', offset=offset_src), 0,
                            input_h * input_w, burst, 0, (block_w - 1) * burst))

                # move data from UB to GM
                start = (crop_t - (b_h + divisor) + block_h - 1) // block_h
                end = (crop_t + output_h -
                       (b_h + divisor) + block_h - 1) // block_h
                offset_base = (var * num_inner +
                               n_i) * output_h * output_w * c_0
                offset_out = ((b_h + divisor) + start * block_h -
                              crop_t) * output_w * c_0 + offset_base
                offset_pad = start * padded_w * c_0 + crop_l * c_0
                with i_b.if_scope(end - start > 0):
                    i_b.emit(
                        tvm.call_extern(
                            dtype, "copy_ubuf_to_gm",
                            output_data.access_ptr('w', offset=offset_out),
                            padded_data2.access_ptr('r', offset=offset_pad), 0,
                            end - start, output_w * burst, crop_w * burst,
                            output_w * (block_h - 1) * burst))
            if remainder != 0:
                # move data from GM to UB
                with i_b.for_range(0, block_w) as b_w:
                    offset_src = (
                        ((block_h - 1) * block_w + b_w) * num +
                        (var * num_inner + n_i)) * input_h * input_w * c_0
                    offset_dst = b_w * c_0
                    i_b.emit(
                        tvm.call_extern(
                            dtype, "copy_gm_to_ubuf",
                            padded_data.access_ptr('w', offset=offset_dst),
                            input_data.access_ptr('r', offset=offset_src), 0,
                            input_h * input_w, burst, 0, (block_w - 1) * burst))

                # move data from UB to GM
                start = (crop_t - (block_h - 1) + block_h - 1) // block_h
                end = (crop_t + output_h -
                       (block_h - 1) + block_h - 1) // block_h
                offset_base = (var * num_inner +
                               n_i) * output_h * output_w * c_0
                offset_out = ((block_h - 1) + start * block_h -
                              crop_t) * output_w * c_0 + offset_base
                offset_pad = start * padded_w * c_0 + crop_l * c_0
                with i_b.if_scope(end - start > 0):
                    i_b.emit(
                        tvm.call_extern(
                            dtype, "copy_ubuf_to_gm",
                            output_data.access_ptr('w', offset=offset_out),
                            padded_data.access_ptr('r', offset=offset_pad), 0,
                            end - start, output_w * burst, crop_w * burst,
                            output_w * (block_h - 1) * burst))

    elif tile_axis in (1, 2) and (block_w - 1) * burst <= 65535:
        size = padded_w * c_0
        padded_data = batch.new_alloc(
            i_b, [size], name="padded_data", scope=tbe_platform.scope_ubuf)

        with i_b.for_range(0, num_inner) as n_i:
            with i_b.for_range(0, block_h) as b_h:
                with i_b.for_range(0, input_h) as i_h:
                    # move data from GM to UB
                    with i_b.for_range(0, block_w) as b_w:
                        offset_src = (((b_h * block_w + b_w) * num +
                                       (var * num_inner + n_i)) * input_h +
                                      i_h) * input_w * c_0
                        i_b.emit(
                            tvm.call_extern(
                                dtype, "copy_gm_to_ubuf",
                                padded_data.access_ptr("w", offset=b_w * c_0),
                                input_data.access_ptr("r", offset=offset_src),
                                0, input_w, burst, 0, (block_w - 1) * burst))

                    # move data from UB to GM
                    with i_b.if_scope(
                            tvm.all(i_h * block_h + b_h >= crop_t,
                                    i_h * block_h + b_h < output_h + crop_t)):
                        offset_base = (var * num_inner +
                                       n_i) * output_h * output_w * c_0
                        offset_out = (i_h * block_h + b_h -
                                      crop_t) * output_w * c_0 + offset_base
                        i_b.emit(
                            tvm.call_extern(
                                dtype, "copy_ubuf_to_gm",
                                output_data.access_ptr("w", offset=offset_out),
                                padded_data.access_ptr(
                                    "r", offset=crop_l * c_0), 0, 1,
                                output_w * burst, 0, 0))

    elif tile_axis == 3 and (block_w - 1) * burst <= 65535:
        size = input_w * c_0
        padded_data = batch.new_alloc(
            i_b, [size], name="padded_data", scope=tbe_platform.scope_ubuf)

        with i_b.for_range(0, num_inner) as n_i:
            with i_b.for_range(0, block_h) as b_h:
                with i_b.for_range(0, input_h) as i_h:
                    with i_b.for_range(0, block_w) as b_w:
                        # move data from GM to UB
                        offset_src = (((b_h * block_w + b_w) * num +
                                       (var * num_inner + n_i)) * input_h +
                                      i_h) * input_w * c_0
                        i_b.emit(
                            tvm.call_extern(
                                dtype, "copy_gm_to_ubuf",
                                padded_data.access_ptr("w", offset=0),
                                input_data.access_ptr("r", offset=offset_src),
                                0, input_w, burst, 0, 0))

                        # move data from UB to GM
                        with i_b.if_scope(
                                tvm.all(
                                    i_h * block_h + b_h >= crop_t,
                                    i_h * block_h + b_h < output_h + crop_t)):
                            start = (crop_l - b_w + block_w - 1) // block_w
                            end = (crop_l + output_w - b_w + block_w -
                                   1) // block_w
                            offset_base = (var * num_inner +
                                           n_i) * output_h * output_w * c_0
                            offset_dst = (i_h * block_h + b_h -
                                          crop_t) * output_w * c_0 + (
                                              b_w + start * block_w -
                                              crop_l) * c_0 + offset_base
                            with i_b.if_scope(end - start > 0):
                                i_b.emit(
                                    tvm.call_extern(
                                        dtype, "copy_ubuf_to_gm",
                                        output_data.access_ptr(
                                            "w", offset=offset_dst),
                                        padded_data.access_ptr(
                                            "r",
                                            offset=start * c_0), 0, end - start,
                                        burst, 0, (block_w - 1) * burst))

    else:
        size = c_0
        padded_data = batch.new_alloc(
            i_b, [size], name="padded_data", scope=tbe_platform.scope_ubuf)

        with i_b.for_range(0, num_inner) as n_i:
            with i_b.for_range(0, block_h) as b_h:
                with i_b.for_range(0, input_h) as i_h:
                    with i_b.for_range(0, block_w) as b_w:
                        with i_b.for_range(0, input_w) as i_w:
                            # move data from GM to UB
                            offset_src = ((
                                ((b_h * block_w + b_w) * num +
                                 (var * num_inner + n_i)) * input_h + i_h) *
                                          input_w + i_w) * c_0
                            i_b.emit(
                                tvm.call_extern(
                                    dtype, "copy_gm_to_ubuf",
                                    padded_data.access_ptr("w", offset=0),
                                    input_data.access_ptr(
                                        "r", offset=offset_src), 0, 1, burst, 0,
                                    0))

                            # move data from UB to GM
                            with i_b.if_scope(
                                    tvm.all(
                                        i_h * block_h + b_h >= crop_t,
                                        i_h * block_h + b_h < output_h + crop_t,
                                        i_w * block_w + b_w >= crop_l,
                                        i_w * block_w + b_w <
                                        output_w + crop_l)):
                                offset_base = (var * num_inner +
                                               n_i) * output_h * output_w * c_0
                                offset_out = (i_h * block_h + b_h -
                                              crop_t) * output_w * c_0 + (
                                                  i_w * block_w + b_w -
                                                  crop_l) * c_0 + offset_base
                                i_b.emit(
                                    tvm.call_extern(
                                        dtype, "copy_ubuf_to_gm",
                                        output_data.access_ptr(
                                            "w", offset=offset_out),
                                        padded_data.access_ptr("r", offset=0),
                                        0, 1, burst, 0, 0))

    return i_b.get()


def _check_parms(shape, dtype, block_shape, crops, kernel_name):
    """check the parameters including shape, dtype, block_shape, crops
    and kernel_name.
    """
    dtype_list = ("float16", "float32")
    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_dtype_rule(dtype, dtype_list)
    util.check_kernel_name(kernel_name)

    if len(shape) != 5:
        raise RuntimeError(
            "the shape of input tensor should be 5, but got: %d" % len(shape))

    if len(block_shape) != 2:
        raise RuntimeError("the shape of block_shape should be 2, "
                           "but got: %d" % len(block_shape))

    if len(crops) != 2 or len(crops[0]) != 2 or len(crops[1]) != 2:
        raise RuntimeError("the shape of crops should be 2x2")

    if not (isinstance(block_shape[0], int) and isinstance(block_shape[1], int)
            and block_shape[0] > 0 and block_shape[1] > 0):
        raise RuntimeError(
            "the value of block_shape should be integer and be greater to 0")

    if not (isinstance(crops[0][0], int) and crops[0][0] >= 0 and
            isinstance(crops[0][1], int) and crops[0][1] >= 0 and
            isinstance(crops[1][0], int) and crops[1][0] >= 0 and
            isinstance(crops[1][1], int) and crops[1][1] >= 0):
        raise RuntimeError("the value of crops should be integer "
                           "and be greater or equal to 0")

    if crops[0][0] + crops[0][1] >= shape[2] * block_shape[0]:
        raise RuntimeError("crops in height dimension should less than "
                           "(input height)*(block height)")

    if crops[1][0] + crops[1][1] >= shape[3] * block_shape[1]:
        raise RuntimeError("crops in width dimension should less than "
                           "(input width)*(block width)")

    if shape[0] % (block_shape[0] * block_shape[1]) != 0:
        raise RuntimeError(
            "batch size/(block height*block width) should be integer")


# pylint: disable=invalid-name
@fusion_manager.register("batch_to_space_nd_d")
def batch_to_space_nd_d_compute(x,
                                y,
                                block_shape,
                                crops,
                                kernel_name="batch_to_space_nd_d"):
    """batch_to_space for tensor.

    Parameters
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    block_shape: list or tuple
        1-D with shape [2].
    crops: list or tuple
        2-D with shape [2, 2], crops[i] = [crop_start, crop_end].
    kernel_name: str
        cce kernel name, default value is "batch_to_space_nd_d".

    Returns
    -------
    res: TVM tensor
        output tensor.
    """
    shape = te.lang.cce.util.shape_to_list(x.shape)
    dtype = x.dtype

    if len(crops) == 4:
        crops = [[crops[0], crops[1]], [crops[2], crops[3]]]

    batch = _BatchToSpace(shape, dtype, block_shape, crops)

    res = tvm.extern([batch.output_shape], [x],
                     lambda ins, outs: _kernel_ir(outs, ins, batch),
                     dtype=dtype,
                     name="res")

    return res


@util.check_input_type(dict, dict, (list, tuple), (list, tuple), str)
def batch_to_space_nd_d(x,
                        y,
                        block_shape,
                        crops,
                        kernel_name="batch_to_space_nd_d"):
    """batch_to_space for tensor.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    block_shape: list or tuple
        1-D with shape [2].
    crops: list or tuple
        2-D with shape [2, 2], crops[i] = [crop_start, crop_end].
    kernel_name: str
        cce kernel name, default value is "batch_to_space_nd_d".

    Returns
    -------
    None.
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()

    if len(crops) == 4:
        crops = [[crops[0], crops[1]], [crops[2], crops[3]]]

    _check_parms(shape, dtype, block_shape, crops, kernel_name)

    data = tvm.placeholder(shape, name="data", dtype=dtype)

    res = batch_to_space_nd_d_compute(data, y, block_shape, crops, kernel_name)

    sch = tvm.create_schedule(res.op)

    with build_config:
        tvm.build(sch, [data, res], "cce", name=kernel_name)
