#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distrir_builderuted under the License is distrir_builderuted on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

resize_nearest_neighbor_v2_d
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import te.lang.cce
from te import tvm
from te import tik
from te import platform as tbe_platform
from te.platform.cce_build import build_config
from te.platform.cce_intrin_md import reset_mask_insn
from topi.cce import util
from impl.copy_only import copy_only

# one block
BLOCK_SIZE = 32


# pylint: disable=invalid-name,too-many-lines,too-many-arguments
def _apply_store_buffer(ib,
                        dtype,
                        shape,
                        scope=tbe_platform.scope_ubuf,
                        name='store_buf',
                        double_buffer=False):
    """allocate the the specific scope buffer
    """
    buf_var = ib.allocate(dtype, shape, name=name, scope=scope)
    if double_buffer is True:
        ib.scope_attr(buf_var.asnode(), "double_buffer_scope", 1)
    tmp_buffer = tvm.decl_buffer(
        shape, buf_var.dtype, name=name, scope=scope, data=buf_var)

    return tmp_buffer


# pylint: disable=too-many-locals,too-many-statements,unused-argument
def _apply_reg_buffer(ib, dtype, shape, name="reg_buf"):
    """allocate the reg scope buffer
    """
    return ib.allocate(dtype, shape, name=name, scope=tbe_platform.scope_reg)


# pylint: disable=too-many-instance-attributes
class ResizeNearestNeighbor:
    """ResizeNearestNeighbor main functions
    """

    def __init__(self,
                 input_shape,
                 output_shape,
                 dtype,
                 align_corners=False,
                 half_pixel_centers=False,
                 kernel_name="resize_nearest_neighbor"):
        """init ResizeNearestNeighbor base parameters
        """
        self.dtype = dtype
        self.tik_instance = tik.Tik()
        self.core_number = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        self.ub_size = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.UB_SIZE)
        self.l1_size = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.L1_SIZE)
        self.dtype_size = tbe_platform.cce_intrin.get_bit_len(self.dtype) // 8
        self.batch = input_shape[0]
        self.c1 = input_shape[1]
        self.input_h = input_shape[2]
        self.input_w = input_shape[3]
        self.c0 = input_shape[4]
        self.output_h = output_shape[2]
        self.output_w = output_shape[3]
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers
        self.kernel_name = kernel_name
        if dtype == "float16":
            self.c0_block = 1
        else:
            self.c0_block = 2
        self.input_data = self.tik_instance.Tensor(
            self.dtype, input_shape, name="input_data", scope=tik.scope_gm)
        self.output_data = self.tik_instance.Tensor(
            self.dtype, output_shape, name="output_data", scope=tik.scope_gm)
        self.half_ub_block = self.ub_size // BLOCK_SIZE // 2 - 16
        self.half_l1_block = self.l1_size // BLOCK_SIZE // 2 - 16
        self.scale_h, self.scale_w = self.calculate_scale()
        self.one_six_ub_ele = (self.ub_size - 1024) // self.dtype_size // 6

    def calculate_scale(self):
        """calculate scale
        """
        if self.align_corners is True and self.output_h > 1:
            scale_h = float(self.input_h - 1) / float(self.output_h - 1)
        else:
            scale_h = float(self.input_h) / float(self.output_h)

        if self.align_corners is True and self.output_w > 1:
            scale_w = float(self.input_w - 1) / float(self.output_w - 1)
        else:
            scale_w = float(self.input_w) / float(self.output_w)

        return scale_h, scale_w

    def calculate_index(self, ib, scale, ub_int32, ub_fp16, ub_fp32, src_reg,
                        i):
        """ calculate w_location
        """
        flag = tbe_platform.intrinsic_check_support("Intrinsic_vconv",
                                                    "s322f32")
        with ib.if_scope(flag):
            mask = 8
            reset_mask_insn(ib, "float32", bits=mask)
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "vconv_s322f32",
                                    ub_fp32.access_ptr("w", offset=0),
                                    ub_int32.access_ptr("r", offset=0), 1, 1, 1,
                                    8, 8))
        with ib.else_scope():
            mask = 16
            reset_mask_insn(ib, "float16", bits=mask)
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float16", "set_deqscale",
                                    tvm.const(1.0, dtype="float16")))
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float16", "vconv_deq",
                                    ub_fp16.access_ptr("w", offset=0),
                                    ub_int32.access_ptr("r", offset=0), 1, 1, 1,
                                    8, 8))
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "vconv_f162f32",
                                    ub_fp32.access_ptr("w", offset=0),
                                    ub_fp16.access_ptr("r", offset=0), 1, 1, 1,
                                    8, 8))
        with ib.if_scope(not self.align_corners and self.half_pixel_centers):
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "vadds",
                                    ub_fp32.access_ptr("w", offset=0),
                                    ub_fp32.access_ptr("r", offset=0),
                                    tvm.const(0.5).astype("float32"), 1, 1, 1,
                                    8, 8))
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "vmuls",
                                    ub_fp32.access_ptr("w", offset=0),
                                    ub_fp32.access_ptr("r", offset=0),
                                    tvm.const(scale).astype("float32"), 1, 1, 1,
                                    8, 8))
            with ib.if_scope(flag):
                with ib.new_scope():
                    ib.emit(
                        tvm.call_extern("int32", "vconv_f322s32f",
                                        ub_int32.access_ptr("w", offset=0),
                                        ub_fp32.access_ptr("r", offset=0), 1, 1,
                                        1, 8, 8))
            with ib.else_scope():
                with ib.new_scope():
                    ib.emit(
                        tvm.call_extern("float16", "vconv_f322f16",
                                        ub_fp16.access_ptr("w", offset=0),
                                        ub_fp32.access_ptr("r", offset=0), 1, 1,
                                        1, 8, 8))
                with ib.new_scope():
                    ib.emit(
                        tvm.call_extern("int32", "vconv_f162s32f",
                                        ub_int32.access_ptr("w", offset=0),
                                        ub_fp16.access_ptr("r", offset=0), 1, 1,
                                        1, 8, 8))
        with ib.if_scope(not self.align_corners and
                         not self.half_pixel_centers):
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "vmuls",
                                    ub_fp32.access_ptr("w", offset=0),
                                    ub_fp32.access_ptr("r", offset=0),
                                    tvm.const(scale).astype("float32"), 1, 1, 1,
                                    8, 8))
            with ib.if_scope(flag):
                with ib.new_scope():
                    ib.emit(
                        tvm.call_extern("int32", "vconv_f322s32f",
                                        ub_int32.access_ptr("w", offset=0),
                                        ub_fp32.access_ptr("r", offset=0), 1, 1,
                                        1, 8, 8))
            with ib.else_scope():
                with ib.new_scope():
                    ib.emit(
                        tvm.call_extern("float16", "vconv_f322f16",
                                        ub_fp16.access_ptr("w", offset=0),
                                        ub_fp32.access_ptr("r", offset=0), 1, 1,
                                        1, 8, 8))
                with ib.new_scope():
                    ib.emit(
                        tvm.call_extern("int32", "vconv_f162s32f",
                                        ub_int32.access_ptr("w", offset=0),
                                        ub_fp16.access_ptr("r", offset=0), 1, 1,
                                        1, 8, 8))
        with ib.if_scope(self.align_corners and not self.half_pixel_centers):
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "vmuls",
                                    ub_fp32.access_ptr("w", offset=0),
                                    ub_fp32.access_ptr("r", offset=0),
                                    tvm.const(scale).astype("float32"), 1, 1, 1,
                                    8, 8))
                with ib.if_scope(flag):
                    with ib.new_scope():
                        ib.emit(
                            tvm.call_extern("int32", "vconv_f322s32r",
                                            ub_int32.access_ptr("w", offset=0),
                                            ub_fp32.access_ptr("r", offset=0),
                                            1, 1, 1, 8, 8))
                with ib.else_scope():
                    with ib.new_scope():
                        ib.emit(
                            tvm.call_extern("float16", "vconv_f322f16",
                                            ub_fp16.access_ptr("w", offset=0),
                                            ub_fp32.access_ptr("r", offset=0),
                                            1, 1, 1, 8, 8))
                    with ib.new_scope():
                        ib.emit(
                            tvm.call_extern("int32", "vconv_f162s32r",
                                            ub_int32.access_ptr("w", offset=0),
                                            ub_fp16.access_ptr("r", offset=0),
                                            1, 1, 1, 8, 8))
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("int32", "reg_mov",
                                tvm.call_extern("int32", "reg", src_reg[i]),
                                ub_int32.access_ptr("r", offset=0), 0))

    def tile_b_c1(self):
        """tile b*c1
        """
        b_c1 = self.batch * self.c1
        if self.half_ub_block % self.c0_block == 0:
            num = self.half_ub_block // self.c0_block
        else:
            num = (self.half_ub_block - 1) // self.c0_block
        one_loop_num = min(b_c1, num)
        if self.input_h * self.input_w * self.c0_block >= 65536:
            src_loop_num = 1
        else:
            src_loop_num = one_loop_num
        if self.output_h * self.output_w * self.c0_block >= 65536:
            dst_loop_num = 1
        else:
            dst_loop_num = one_loop_num
        src_loop = (b_c1 + src_loop_num - 1) // src_loop_num
        dst_loop = (b_c1 + dst_loop_num - 1) // dst_loop_num

        return src_loop, src_loop_num, dst_loop, dst_loop_num

    def calculate_w_b(self):
        """ calculate w and b for UB to GM
        """
        w_out = 80
        ub_flag = False
        l1_flag = False
        if self.output_w <= w_out:
            w_out = self.output_w
        b_out = self.half_ub_block // (w_out * self.c1 * self.c0_block)
        if self.align_corners and not self.half_pixel_centers:
            w_in = int(
                min(round(float(w_out) * self.scale_w), self.input_w - 1))
        elif not self.align_corners and self.half_pixel_centers:
            w_in = int(
                min(
                    math.floor((float(w_out) + 0.5) * self.scale_w),
                    self.input_w - 1))
        else:
            w_in = int(
                min(math.floor(float(w_out) * self.scale_w), self.input_w - 1))

        # if ub can't put one batch, ub_flag is True
        if b_out == 0:
            b_out = 1
            ub_flag = True
        elif b_out > self.batch:
            b_out = self.batch

        # if l1 can't put the selected data, l1_flag is True
        if w_in * b_out * self.c1 * self.c0_block > self.half_l1_block:
            flag = False
            if b_out == 1:
                l1_flag = True
            else:
                while b_out > 1:
                    b_out = b_out - 1
                    if w_in * b_out * \
                            self.c1 * self.c0_block <= self.half_l1_block:
                        flag = True
                        break
                if not flag:
                    b_out = 1
                    l1_flag = True

        return w_out, w_in, b_out, ub_flag, l1_flag

    def copy_src_to_dst_ub(self, ib, input_data, output_data, src_offset_h,
                           src_offset_w, dst_offset_h, dst_offset_w, loop,
                           loop_num):
        """gm -> ub ->gm
        """
        b_c1 = self.batch * self.c1

        src_loop, src_loop_num, dst_loop, dst_loop_num = self.tile_b_c1()

        if self.input_h * self.input_w * \
                self.c0_block >= 65536 or loop_num == 1:
            src_stride = 0
        else:
            src_stride = self.input_h * self.input_w * \
                         self.c0_block - self.c0_block
        if self.output_h * self.output_w * \
                self.c0_block >= 65536 or loop_num == 1:
            dst_stride = 0
        else:
            dst_stride = self.output_h * self.output_w * \
                         self.c0_block - self.c0_block

        with ib.for_range(0, loop) as loop_idx:
            ub_data = _apply_store_buffer(
                ib,
                self.dtype, [loop_num, self.c0],
                name="ub_data",
                scope=tbe_platform.scope_ubuf,
                double_buffer=True)
            start = loop_idx * loop_num
            end = tvm.min((loop_idx + 1) * loop_num, b_c1)
            one_loop_ele = end - start
            src_offset_loop = loop_idx * loop_num * self.input_h * \
                              self.input_w * self.c0
            dst_offset_loop = loop_idx * loop_num * self.output_h * \
                              self.output_w * self.c0
            src_offset = src_offset_h + src_offset_w + src_offset_loop
            dst_offset = dst_offset_h + dst_offset_w + dst_offset_loop
            flag = src_loop == dst_loop and src_loop_num == dst_loop_num
            with ib.if_scope(flag):
                with ib.if_scope(one_loop_ele > 0):
                    # move gm to ub
                    with ib.new_scope():
                        ib.emit(
                            tvm.call_extern(
                                self.dtype, "copy_gm_to_ubuf",
                                ub_data.access_ptr("w", offset=0),
                                input_data.access_ptr("r", offset=src_offset),
                                0, one_loop_ele, self.c0_block, src_stride, 0))
                    # move ub to gm
                    with ib.new_scope():
                        ib.emit(
                            tvm.call_extern(
                                self.dtype, "copy_ubuf_to_gm",
                                output_data.access_ptr("w", offset=dst_offset),
                                ub_data.access_ptr("r", offset=0), 0,
                                one_loop_ele, self.c0_block, 0, dst_stride))

            with ib.else_scope():
                with ib.if_scope(loop == src_loop):
                    with ib.if_scope(one_loop_ele > 0):
                        # move gm to ub
                        with ib.new_scope():
                            ib.emit(
                                tvm.call_extern(
                                    self.dtype, "copy_gm_to_ubuf",
                                    ub_data.access_ptr("w", offset=0),
                                    input_data.access_ptr(
                                        "r", offset=src_offset), 0, end - start,
                                    self.c0_block, src_stride, 0))
                        # move ub to gm
                        with ib.for_range(0, one_loop_ele) as one_idx:
                            ub_offset_one = one_idx * self.c0
                            dst_offset_one = dst_offset + \
                                             one_idx * self.output_h * \
                                             self.output_w * self.c0
                            with ib.new_scope():
                                ib.emit(
                                    tvm.call_extern(
                                        self.dtype, "copy_ubuf_to_gm",
                                        output_data.access_ptr(
                                            "w", offset=dst_offset_one),
                                        ub_data.access_ptr(
                                            "r", offset=ub_offset_one), 0, 1,
                                        self.c0_block, 0, 0))
                with ib.else_scope():
                    with ib.if_scope(one_loop_ele > 0):
                        # move gm to ub
                        with ib.for_range(0, one_loop_ele) as one_idx:
                            src_offset_one = src_offset + \
                                             one_idx * self.input_h * \
                                             self.input_w * self.c0
                            ub_offset_one = one_idx * self.c0
                            with ib.new_scope():
                                ib.emit(
                                    tvm.call_extern(
                                        self.dtype, "copy_gm_to_ubuf",
                                        ub_data.access_ptr(
                                            "w", offset=ub_offset_one),
                                        input_data.access_ptr(
                                            "r", offset=src_offset_one), 0, 1,
                                        self.c0_block, 0, 0))
                        # move ub to gm
                        with ib.new_scope():
                            ib.emit(
                                tvm.call_extern(
                                    self.dtype, "copy_ubuf_to_gm",
                                    output_data.access_ptr(
                                        "w", offset=dst_offset),
                                    ub_data.access_ptr("r", offset=0), 0,
                                    one_loop_ele, self.c0_block, 0, dst_stride))

    def copy_src_to_dst_l1(self, ib, core_idx, core_number, input_data,
                           output_data, loop_idx):
        """gm -> l1 -> ub ->gm
        """
        w_out, w_in, b_out, _, _ = self.calculate_w_b()
        b_loop = (self.batch + b_out - 1) // b_out
        w_loop = (self.output_w + w_out - 1) // w_out

        src_reg = _apply_reg_buffer(ib, "int32", [4], name="src_reg")
        ub_int32_h = _apply_store_buffer(
            ib,
            "int32", [self.c0],
            name="ub_int32_h",
            scope=tbe_platform.scope_ubuf)
        ub_int32_w = _apply_store_buffer(
            ib,
            "int32", [self.c0],
            name="ub_int32_w",
            scope=tbe_platform.scope_ubuf)
        ub_fp16_h = _apply_store_buffer(
            ib,
            "float16", [self.c0],
            name="ub_fp16_h",
            scope=tbe_platform.scope_ubuf)
        ub_fp16_w = _apply_store_buffer(
            ib,
            "float16", [self.c0],
            name="ub_fp16_w",
            scope=tbe_platform.scope_ubuf)
        ub_fp32_h = _apply_store_buffer(
            ib,
            "float32", [self.c0],
            name="ub_fp32_h",
            scope=tbe_platform.scope_ubuf)
        ub_fp32_w = _apply_store_buffer(
            ib,
            "float32", [self.c0],
            name="ub_fp32_w",
            scope=tbe_platform.scope_ubuf)

        with ib.for_range(0, b_loop) as b_loop_idx:
            start = b_loop_idx * b_out
            end = tvm.min((b_loop_idx + 1) * b_out, self.batch)
            one_loop_b = end - start
            # calculate src_h
            with ib.new_scope():
                src_reg[0] = core_idx + loop_idx * core_number
                ib.emit(
                    tvm.call_extern("int32", "reg_mov",
                                    ub_int32_h.access_ptr("w", offset=0),
                                    tvm.call_extern("int32", "reg",
                                                    src_reg[0])))
                self.calculate_index(ib, self.scale_h, ub_int32_h, ub_fp16_h,
                                     ub_fp32_h, src_reg, 0)
                src_h = tvm.min(src_reg[0], self.input_h - 1)

            with ib.for_range(0, w_loop) as w_loop_idx:
                if w_out < self.output_w:
                    # calc w_in begin
                    with ib.new_scope():
                        src_reg[2] = w_loop_idx * w_out
                        ib.emit(
                            tvm.call_extern(
                                "int32", "reg_mov",
                                ub_int32_w.access_ptr("w", offset=0),
                                tvm.call_extern("int32", "reg", src_reg[2])))
                        self.calculate_index(ib, self.scale_w, ub_int32_w,
                                             ub_fp16_w, ub_fp32_w, src_reg, 2)
                        w_in_begin = tvm.min(src_reg[2], self.input_w - 1)

                    # calc w_in end
                    with ib.new_scope():
                        src_reg[3] = tvm.min((w_loop_idx + 1) * w_out - 1,
                                             self.output_w - 1)
                        ib.emit(
                            tvm.call_extern(
                                "int32", "reg_mov",
                                ub_int32_w.access_ptr("w", offset=0),
                                tvm.call_extern("int32", "reg", src_reg[3])))
                        self.calculate_index(ib, self.scale_w, ub_int32_w,
                                             ub_fp16_w, ub_fp32_w, src_reg, 3)
                        w_in_end = tvm.min(src_reg[3], self.input_w - 1)
                else:
                    w_in_begin = 0
                    w_in_end = self.input_w - 1
                    w_in = self.input_w

                # step1: copy data from GM to L1
                l1_data = _apply_store_buffer(
                    ib,
                    self.dtype, [b_out, self.c1, w_in + 2, self.c0],
                    name="l1_data",
                    scope=tbe_platform.scope_cbuf,
                    double_buffer=True)

                # copy b_out*c1*w_in*c0 to L1
                if self.input_h * self.input_w * self.c0_block <= 65535:
                    gl1_dst_offset = 0
                    gl1_src_offset = src_h * self.input_w * self.c0 + \
                                     w_in_begin * self.c0 + \
                                     b_out * b_loop_idx * self.c1 * \
                                     self.input_h * self.input_w * self.c0
                    ib.emit(
                        tvm.call_extern(
                            self.dtype, "copy_gm_to_cbuf",
                            l1_data.access_ptr("w", offset=gl1_dst_offset),
                            input_data.access_ptr("r", offset=gl1_src_offset),
                            0, one_loop_b * self.c1,
                            (w_in_end - w_in_begin + 1) * self.c0_block,
                            self.input_h * self.input_w * self.c0_block -
                            w_in_end * self.c0_block +
                            w_in_begin * self.c0_block - self.c0_block, 0, 0))
                else:
                    with ib.for_range(0, one_loop_b * self.c1) as lot:
                        gl1_dst_offset = lot * (w_in_end - w_in_begin +
                                                1) * self.c0
                        gl1_src_offset = src_h * self.input_w * self.c0 + \
                                         w_in_begin * self.c0 + \
                                         (b_out * b_loop_idx * self.c1 + lot) * \
                                         self.input_h * self.input_w * self.c0
                        ib.emit(
                            tvm.call_extern(
                                self.dtype, "copy_gm_to_cbuf",
                                l1_data.access_ptr("w", offset=gl1_dst_offset),
                                input_data.access_ptr(
                                    "r", offset=gl1_src_offset), 0, 1,
                                (w_in_end - w_in_begin + 1) * self.c0_block, 0,
                                0, 0))

                # step2: copy data from L1 to UB
                ub_data = _apply_store_buffer(
                    ib,
                    self.dtype, [b_out, self.c1, w_out, self.c0],
                    name="ub_data",
                    scope=tbe_platform.scope_ubuf,
                    double_buffer=True)

                # from L1-UB b_out*c1;
                # from UB-GM, n_burst=b_out*c1 burst_len=w_out
                with ib.for_range(
                        0, tvm.min(w_out, self.output_w -
                                   w_loop_idx * w_out)) as w_out_idx:
                    src_reg[1] = w_loop_idx * w_out + w_out_idx
                    ib.emit(
                        tvm.call_extern(
                            "int32", "reg_mov",
                            ub_int32_w.access_ptr("w", offset=0),
                            tvm.call_extern("int32", "reg", src_reg[1])))
                    self.calculate_index(ib, self.scale_w, ub_int32_w,
                                         ub_fp16_w, ub_fp32_w, src_reg, 1)
                    src_w = tvm.min(src_reg[1], self.input_w - 1) - w_in_begin

                    l1u_src_offset = src_w * self.c0
                    l1u_dst_offset = w_out_idx * self.c0

                    ib.emit(
                        tvm.call_extern(
                            self.dtype, "copy_cbuf_to_ubuf",
                            ub_data.access_ptr("w", offset=l1u_dst_offset),
                            l1_data.access_ptr("r", offset=l1u_src_offset), 0,
                            one_loop_b * self.c1, self.c0_block,
                            (w_in_end - w_in_begin) * self.c0_block,
                            tvm.min(w_out - 1, self.output_w -
                                    w_loop_idx * w_out - 1) * self.c0_block))

                # step3: copy UB to GM
                if self.output_h * self.output_w * self.c0_block <= 65535:
                    ug_src_offset = 0
                    ug_dst_offset = (core_idx + loop_idx * core_number) * \
                                    self.output_w * self.c0 + b_out * \
                                    b_loop_idx * self.c1 * self.output_h * \
                                    self.output_w * self.c0 + w_loop_idx * \
                                    w_out * self.c0
                    ib.emit(
                        tvm.call_extern(
                            self.dtype, "copy_ubuf_to_gm",
                            output_data.access_ptr("w", offset=ug_dst_offset),
                            ub_data.access_ptr("r", offset=ug_src_offset), 0,
                            one_loop_b * self.c1,
                            tvm.min(w_out, self.output_w - w_loop_idx * w_out) *
                            self.c0_block, 0,
                            self.output_w * self.output_h * self.c0_block -
                            tvm.min(w_out, self.output_w - w_loop_idx * w_out) *
                            self.c0_block))
                else:
                    with ib.for_range(0, one_loop_b * self.c1) as lot:
                        ug_src_offset = lot * \
                                        tvm.min(w_out, self.output_w -
                                                w_loop_idx * w_out) * self.c0
                        ug_dst_offset = (core_idx + loop_idx * core_number) * \
                                        self.output_w * self.c0 + b_out * \
                                        b_loop_idx * self.c1 * self.output_h * \
                                        self.output_w * self.c0 + lot * \
                                        self.output_h * self.output_w \
                                        * self.c0 + w_loop_idx * w_out * self.c0
                        ib.emit(
                            tvm.call_extern(
                                self.dtype, "copy_ubuf_to_gm",
                                output_data.access_ptr(
                                    "w", offset=ug_dst_offset),
                                ub_data.access_ptr("r",
                                                   offset=ug_src_offset), 0, 1,
                                tvm.min(w_out, self.output_w -
                                        w_loop_idx * w_out) * self.c0_block, 0,
                                0))

    def copy_src_h_w_to_dst_nh_nw(self):
        """when src shape is [h,w]; dst shape is [n*h,n*w];
           gm -> ub -> ub -> gm
        """
        scale_h = self.output_h // self.input_h
        scale_w = self.output_w // self.input_w
        b_c1_h = self.batch * self.c1 * self.input_h
        core_num = min(b_c1_h, self.core_number)
        core_loop = b_c1_h // core_num
        core_last = b_c1_h % core_num

        def _inner_run(src_offset_core, dst_offset_core, core_loop):
            num = min(core_loop, (2 * self.one_six_ub_ele) //
                      (scale_h * self.output_w * self.c0))
            loop = core_loop // num
            last = core_loop % num

            thread = 1
            if loop > 1:
                thread = 2
            with self.tik_instance.for_range(
                    0, loop, thread_num=thread) as loop_idx:
                # define ub data
                ub_data_1 = self.tik_instance.Tensor(
                    self.dtype, (self.one_six_ub_ele,),
                    name="ub_data_1",
                    scope=tik.scope_ubuf)
                ub_data_2 = self.tik_instance.Tensor(
                    self.dtype, (2 * self.one_six_ub_ele,),
                    name="ub_data_2",
                    scope=tik.scope_ubuf)
                # move gm to ub (input_data -> ub_data_1)
                src_offset = src_offset_core + loop_idx * num * \
                             self.input_w * self.c0
                self.tik_instance.data_move(ub_data_1,
                                            self.input_data[src_offset], 0, 1,
                                            num * self.input_w * self.c0_block,
                                            0, 0)
                # move ub to ub (ub_data_1 -> ub_data_2)
                with self.tik_instance.for_range(0, num) as idx:
                    with self.tik_instance.for_range(0, scale_w) as w_idx:
                        ub_data_1_offset = idx * self.input_w * self.c0
                        ub_data_2_offset = idx * scale_h * self.output_w * \
                                           self.c0 + w_idx * self.c0
                        self.tik_instance.data_move(
                            ub_data_2[ub_data_2_offset],
                            ub_data_1[ub_data_1_offset], 0, self.input_w,
                            self.c0_block, 0, (scale_w - 1) * self.c0_block)
                    with self.tik_instance.for_range(0, scale_h) as h_idx:
                        ub_data_2_offset_src = idx * scale_h * \
                                               self.output_w * self.c0
                        ub_data_2_offset_dst = idx * scale_h * \
                                               self.output_w * self.c0 + \
                                               h_idx * self.output_w * self.c0
                        self.tik_instance.data_move(
                            ub_data_2[ub_data_2_offset_dst],
                            ub_data_2[ub_data_2_offset_src], 0, 1,
                            self.output_w * self.c0_block, 0, 0)
                # move ub to gm (ub_data_2 -> input_data)
                dst_offset = dst_offset_core + loop_idx * num * scale_h * \
                             self.output_w * self.c0
                self.tik_instance.data_move(
                    self.output_data[dst_offset], ub_data_2, 0, 1,
                    num * scale_h * self.output_w * self.c0_block, 0, 0)
            if last != 0:
                with self.tik_instance.for_range(0, 1):
                    # define ub data
                    ub_data_1 = self.tik_instance.Tensor(
                        self.dtype, (self.one_six_ub_ele,),
                        name="ub_data_1",
                        scope=tik.scope_ubuf)
                    ub_data_2 = self.tik_instance.Tensor(
                        self.dtype, (2 * self.one_six_ub_ele,),
                        name="ub_data_2",
                        scope=tik.scope_ubuf)
                    # move gm to ub (input_data -> ub_data_1)
                    src_offset = src_offset_core + loop * num * \
                                 self.input_w * self.c0
                    self.tik_instance.data_move(
                        ub_data_1, self.input_data[src_offset], 0, 1,
                        last * self.input_w * self.c0_block, 0, 0)
                    # move ub to ub (ub_data_1 -> ub_data_2)
                    with self.tik_instance.for_range(0, last) as idx:
                        with self.tik_instance.for_range(0, scale_w) as w_idx:
                            ub_data_1_offset = idx * self.input_w * self.c0
                            ub_data_2_offset = idx * scale_h * self.output_w * \
                                               self.c0 + w_idx * self.c0
                            self.tik_instance.data_move(
                                ub_data_2[ub_data_2_offset],
                                ub_data_1[ub_data_1_offset], 0, self.input_w,
                                self.c0_block, 0, (scale_w - 1) * self.c0_block)
                        with self.tik_instance.for_range(0, scale_h) as h_idx:
                            ub_data_2_offset_src = idx * scale_h * \
                                                   self.output_w * self.c0
                            ub_data_2_offset_dst = idx * scale_h * \
                                                   self.output_w * self.c0 + \
                                                   h_idx * self.output_w * \
                                                   self.c0
                            self.tik_instance.data_move(
                                ub_data_2[ub_data_2_offset_dst],
                                ub_data_2[ub_data_2_offset_src], 0, 1,
                                self.output_w * self.c0_block, 0, 0)
                    # move ub to gm (ub_data_2 -> input_data)
                    dst_offset = dst_offset_core + loop * num * scale_h * \
                                 self.output_w * self.c0
                    self.tik_instance.data_move(
                        self.output_data[dst_offset], ub_data_2, 0, 1,
                        last * scale_h * self.output_w * self.c0_block, 0, 0)

        with self.tik_instance.for_range(
                0, core_num, block_num=core_num) as core_idx:
            src_offset_core = core_idx * core_loop * \
                              self.input_w * self.c0
            dst_offset_core = core_idx * core_loop * scale_h * \
                              self.output_w * self.c0
            _inner_run(src_offset_core, dst_offset_core, core_loop)
            if core_last != 0:
                src_offset_core = (core_loop * core_num) * \
                                  self.input_w * self.c0
                dst_offset_core = (core_loop * core_num) * \
                                  scale_h * self.output_w * self.c0
                _inner_run(src_offset_core, dst_offset_core, core_last)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_data],
            outputs=[self.output_data],
            enable_l2=False)


def check_supported(images,
                    y,
                    size,
                    align_corners=False,
                    half_pixel_centers=False,
                    kernel_name="resize_nearest_neighbor"):
    """check if need go to aicore
    """
    images_shape = images.get("shape")
    images_format = images.get("format")
    if images_format == "NHWC":
        in_size_h = images_shape[1]
        in_size_w = images_shape[2]
    elif images_format in ("NCHW", "NC1HWC0"):
        in_size_h = images_shape[2]
        in_size_w = images_shape[3]
    else:
        raise RuntimeError("The format of images is not supported")

    try:
        if in_size_h > 7680 or in_size_w > 4320:
            return False

        if size[0] > 7680 or size[1] > 4320:
            return False
    except RuntimeError:
        return False

    return True


# pylint: disable=too-many-branches
def resize_nearest_neighbor_v2_d_compute(images,
                                         y,
                                         size,
                                         align_corners=False,
                                         half_pixel_centers=False,
                                         kernel_name="resize_nearest_neighbor"):
    """Resize `images` to `size` using nearest neighbor interpolation.

    Parameters
    ----------
    images: TVM tensor
         the tensor of image in
    y: TVM tensor
        the tensor of image out
    size: list or tuple
        the height and width of output tensor
    align_corners: bool
        whether align_corners
    half_pixel_centers: bool
        whether half_pixel_centers
    kernel_name: str
        cce kernel name, default value is resize_nearest_neighbor

    Returns
    -------
    stmt
    """
    dtype = images.dtype
    input_shape = te.lang.cce.util.shape_to_list(images.shape)
    output_shape = te.lang.cce.util.shape_to_list(y.shape)
    resize = ResizeNearestNeighbor(input_shape, output_shape, dtype,
                                   align_corners, half_pixel_centers,
                                   kernel_name)
    ib = tvm.ir_builder.create()
    block = tvm.thread_axis("blockIdx.x")

    # for No.2 situation
    _, _, _, ub_flag, l1_flag = resize.calculate_w_b()

    # for No.3 situation
    src_loop, src_loop_num, dst_loop, dst_loop_num = resize.tile_b_c1()
    if src_loop == dst_loop and src_loop_num == dst_loop_num:
        loop = src_loop
        loop_num = src_loop_num
    else:
        if src_loop < dst_loop and src_loop_num > dst_loop_num:
            loop = src_loop
            loop_num = src_loop_num
        else:
            loop = dst_loop
            loop_num = dst_loop_num

    if (not ub_flag) and (not l1_flag):
        core_number = min(resize.output_h, resize.core_number)
        ib.scope_attr(block, "thread_extent", core_number)
        if resize.output_h % core_number == 0:
            one_core_loop = resize.output_h // core_number
        else:
            one_core_loop = (resize.output_h + core_number - 1) // core_number

        with ib.for_range(0, one_core_loop) as loop_idx:
            if resize.output_h % core_number == 0:
                resize.copy_src_to_dst_l1(ib, block.var, core_number, images, y,
                                          loop_idx)
            else:
                with ib.if_scope(loop_idx < one_core_loop - 1):
                    resize.copy_src_to_dst_l1(ib, block.var, core_number,
                                              images, y, loop_idx)

                with ib.else_scope():
                    with ib.if_scope(block.var < resize.output_h -
                                     (one_core_loop - 1) * core_number):
                        resize.copy_src_to_dst_l1(ib, block.var, core_number,
                                                  images, y, loop_idx)

    else:
        core_number = min(resize.output_h, resize.core_number)
        ib.scope_attr(block, "thread_extent", core_number)
        if resize.output_h % core_number == 0:
            one_core_loop = resize.output_h // core_number
        else:
            one_core_loop = (resize.output_h + core_number - 1) // core_number

        with ib.for_range(0, one_core_loop) as loop_idx:
            src_reg = _apply_reg_buffer(ib, "int32", [2], name="src_reg")
            ub_int32_h = _apply_store_buffer(
                ib,
                "int32", [resize.c0],
                name="ub_int32_h",
                scope=tbe_platform.scope_ubuf)
            ub_int32_w = _apply_store_buffer(
                ib,
                "int32", [resize.c0],
                name="ub_int32_w",
                scope=tbe_platform.scope_ubuf)
            ub_fp16_h = _apply_store_buffer(
                ib,
                "float16", [resize.c0],
                name="ub_fp16_h",
                scope=tbe_platform.scope_ubuf)
            ub_fp16_w = _apply_store_buffer(
                ib,
                "float16", [resize.c0],
                name="ub_fp16_w",
                scope=tbe_platform.scope_ubuf)
            ub_fp32_h = _apply_store_buffer(
                ib,
                "float32", [resize.c0],
                name="ub_fp32_h",
                scope=tbe_platform.scope_ubuf)
            ub_fp32_w = _apply_store_buffer(
                ib,
                "float32", [resize.c0],
                name="ub_fp32_w",
                scope=tbe_platform.scope_ubuf)
            with ib.if_scope(loop_idx < one_core_loop - 1):
                dst_offset_h_loop = loop_idx * core_number * \
                                    resize.output_w * resize.c0
                dst_offset_h = tvm.min(
                    block.var * resize.output_w * resize.c0 + dst_offset_h_loop,
                    resize.output_h * resize.output_w * resize.c0)
                with ib.new_scope():
                    src_reg[0] = block.var + loop_idx * core_number
                    ib.emit(
                        tvm.call_extern(
                            "int32", "reg_mov",
                            ub_int32_h.access_ptr("w", offset=0),
                            tvm.call_extern("int32", "reg", src_reg[0])))

                # calculate h_location
                resize.calculate_index(ib, resize.scale_h, ub_int32_h,
                                       ub_fp16_h, ub_fp32_h, src_reg, 0)
                src_h = tvm.min(src_reg[0], resize.input_h - 1)
                src_offset_h = src_h * resize.input_w * resize.c0
                with ib.for_range(0, resize.output_w) as w_loop:
                    dst_offset_w = w_loop * resize.c0
                    with ib.new_scope():
                        src_reg[1] = w_loop
                        ib.emit(
                            tvm.call_extern(
                                "int32", "reg_mov",
                                ub_int32_w.access_ptr("w", offset=0),
                                tvm.call_extern("int32", "reg", src_reg[1])))

                    resize.calculate_index(ib, resize.scale_w, ub_int32_w,
                                           ub_fp16_w, ub_fp32_w, src_reg, 1)
                    src_w = tvm.min(src_reg[1], resize.input_w - 1)
                    src_offset_w = src_w * resize.c0

                    resize.copy_src_to_dst_ub(ib, images, y, src_offset_h,
                                              src_offset_w, dst_offset_h,
                                              dst_offset_w, loop, loop_num)
            with ib.else_scope():
                with ib.if_scope(block.var < resize.output_h -
                                 (one_core_loop - 1) * core_number):
                    dst_offset_h_loop = loop_idx * core_number * \
                                        resize.output_w * resize.c0
                    dst_offset_h = tvm.min(
                        block.var * resize.output_w * resize.c0 +
                        dst_offset_h_loop,
                        resize.output_h * resize.output_w * resize.c0)

                    with ib.new_scope():
                        src_reg[0] = block.var + loop_idx * core_number
                        ib.emit(
                            tvm.call_extern(
                                "int32", "reg_mov",
                                ub_int32_h.access_ptr("w", offset=0),
                                tvm.call_extern("int32", "reg", src_reg[0])))

                    # calculate h_location
                    resize.calculate_index(ib, resize.scale_h, ub_int32_h,
                                           ub_fp16_h, ub_fp32_h, src_reg, 0)
                    src_h = tvm.min(src_reg[0], resize.input_h - 1)
                    src_offset_h = src_h * resize.input_w * resize.c0
                    with ib.for_range(0, resize.output_w) as w_loop:
                        dst_offset_w = w_loop * resize.c0

                        with ib.new_scope():
                            src_reg[1] = w_loop
                            ib.emit(
                                tvm.call_extern(
                                    "int32", "reg_mov",
                                    ub_int32_w.access_ptr("w", offset=0),
                                    tvm.call_extern("int32", "reg",
                                                    src_reg[1])))

                        # calculate w_location
                        resize.calculate_index(ib, resize.scale_w, ub_int32_w,
                                               ub_fp16_w, ub_fp32_w, src_reg, 1)
                        src_w = tvm.min(src_reg[1], resize.input_w - 1)
                        src_offset_w = src_w * resize.c0

                        resize.copy_src_to_dst_ub(ib, images, y, src_offset_h,
                                                  src_offset_w, dst_offset_h,
                                                  dst_offset_w, loop, loop_num)

    stmt = ib.get()
    return stmt


@util.check_input_type(dict, dict, (list, tuple), bool, bool, str)
def resize_nearest_neighbor_v2_d(images,
                                 y,
                                 size,
                                 align_corners=False,
                                 half_pixel_centers=False,
                                 kernel_name="resize_nearest_neighbor"):
    """Resize `images` to `size` using nearest neighbor interpolation.

    Parameters
    ----------
    images: dict
        the dict of input, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float16', 'float32'
    y: dict
        the dict of output, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float16', 'float32'
    size: list or tuple
        the height and width of output tensor
    align_corners: bool
        whether align_corners
    half_pixel_centers: bool
        whether half_pixel_centers
    kernel_name: str
        cce kernel name, default value is `resize_nearest_neighbor`

    Returns
    -------
    None
    """
    image_shape = images.get("shape")
    image_dtype = images.get("dtype").lower()
    util.check_shape_rule(image_shape)
    util.check_shape_rule(size)
    util.check_tensor_shape_size(image_shape)
    check_list = ("float16", "float32")
    util.check_dtype_rule(image_dtype, check_list)
    util.check_kernel_name(kernel_name)

    if len(image_shape) != 5:
        raise RuntimeError("the length of image shape must be 5")

    if len(size) != 2:
        raise RuntimeError("the length of size must be 2")

    if align_corners and half_pixel_centers:
        raise RuntimeError(
            "if half_pixel_centers is True, align_corners must be False")

    output_shape = (image_shape[0], image_shape[1], size[0], size[1],
                    image_shape[4])
    util.check_tensor_shape_size(output_shape)

    image_data = tvm.placeholder(
        image_shape, dtype=image_dtype, name="image_data")

    if image_shape[2] == size[0] and image_shape[3] == size[1]:
        copy_only(images, images, kernel_name)
        return

    if size[0] % image_shape[2] == 0 and \
            size[1] % image_shape[3] == 0 and not align_corners \
            and not half_pixel_centers:
        resize = ResizeNearestNeighbor(image_shape, output_shape, image_dtype,
                                       align_corners, half_pixel_centers,
                                       kernel_name)
        if 2 * resize.one_six_ub_ele // (size[0] // image_shape[2] * size[1] *
                                         resize.c0) >= 1:
            resize.copy_src_h_w_to_dst_nh_nw()
            return

    res = tvm.extern([output_shape], [image_data],
                     lambda ins, outs: resize_nearest_neighbor_v2_d_compute(
                         ins[0], outs[0], size, align_corners,
                         half_pixel_centers, kernel_name),
                     dtype=[image_dtype],
                     name="res")

    sch = tvm.create_schedule(res.op)

    with build_config:
        tvm.build(sch, [image_data, res], "cce", name=kernel_name)
