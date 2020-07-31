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

unsorted_segment_sum_d
"""
# pylint: disable=locally-disabled,too-many-lines
import operator
from functools import reduce as functools_reduce

import json
import os
import te.platform.cce_params as cce_params
from te import platform as cce
from te import tvm
from te.platform.cce_build import build_config
from topi.cce import util

from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

MAX_REPEAT_NUM = 255
BLOCK_BYTE = 32
BIT_NUM_ONE_BYTE = 8
MAX_CORE_NUM = 32
REPEAT_BLOCK_NUM = 8


# pylint: disable=locally-disabled,invalid-name
def get_cce_product_version():
    """
    get the current product version

    Parameters
    ----------
    None

    Returns
    -------
    product version
    """
    return cce.cce_conf.get_soc_spec("SOC_VERSION")


def op_select_format(x, segment_ids, y, num_segments,
                     kernel_name="unsorted_segment_sum_d"):
    """
    select format dynamically
    """
    input0_dtype = "float,float"
    input0_format = "NC1HWC0,ND"
    input1_dtype = "int32,int32"
    input1_format = "ND,ND"
    input0_ori_dtype = "float16,float16,float,float,int8,int8,uint8,uint8," \
                       "int32,int32"
    input0_ori_format = "NC1HWC0,ND,NC1HWC0,ND,NC1HWC0,ND,NC1HWC0,ND,NC1HWC0,ND"
    input1_ori_dtype = "int32,int32,int32,int32,int32,int32,int32,int32," \
                       "int32,int32"
    input1_ori_format = "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND"
    ori_dtype = x.get("dtype").lower()
    ori_shape = list(x.get("shape"))
    cce_product = get_cce_product_version()
    if ori_dtype == "float16" and cce_product in ("Ascend910",) and \
            ori_shape == [12288, 1024] and num_segments == 36548:
        input0 = gen_param(classify="input0", name="x",
                           datatype=input0_dtype,
                           format=input0_format)
        input1 = gen_param(classify="input1", name="segment_ids",
                           datatype=input1_dtype,
                           format=input1_format)
        output0 = gen_param(classify="output0", name="y",
                            datatype=input0_dtype,
                            format=input0_format)
    else:
        input0 = gen_param(classify="input0", name="x",
                           datatype=input0_ori_dtype,
                           format=input0_ori_format)
        input1 = gen_param(classify="input1", name="segment_ids",
                           datatype=input1_ori_dtype,
                           format=input1_ori_format)
        output0 = gen_param(classify="output0", name="y",
                            datatype=input0_ori_dtype,
                            format=input0_ori_format)

    param_list = [input0, input1, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _ceil_div(val, block):
    """
    return: (val + block - 1) // block
    """
    return (val + block - 1) // block


def _ceil_fill(val, block):
    """
    return: ((val + block - 1) // block)*block
    """
    return _ceil_div(val, block) * block


def _prod(values):
    """
    return: prod of values
    """
    res = 1
    for value in values:
        res *= value

    return res


def _apply_for_new_alloc(ib_,
                         dtype,
                         buf_len,
                         align_size,
                         scope=cce_params.scope_ubuf):
    """
    ib_: ir builder
    dtype : the data type
    buf_len : apply buffer length
    align_size : for offset align
    scope : cce scope
    return: new buffer
    """
    shape_x = (_ceil_fill(buf_len, align_size), )
    buf_var = ib_.allocate(dtype, shape_x, name="tmp_buf", scope=scope)
    tmp_buffer = tvm.decl_buffer(shape_x,
                                 buf_var.dtype,
                                 name="tmp_buf",
                                 scope=cce_params.scope_ubuf,
                                 data=buf_var)

    return tmp_buffer


# pylint: disable=locally-disabled,too-many-instance-attributes
class SegmentParams():
    """
    parameters for Segment
    """
    def __init__(self, ib_, src_dtype, in_shape, buffers):
        self.ib_ = ib_
        self.src_dtype = src_dtype
        self.element_len = _prod(in_shape[1:])
        self.ids_len = in_shape[0]
        self.in_shape = in_shape
        self.type_size_ids = cce.cce_intrin.get_bit_len('int32') // 8
        self.cp_align_len_ids = \
            cce_params.BLOCK_REDUCE_INT8 // self.type_size_ids
        self.unified_buffer_len_ids = \
            cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE) // \
            self.type_size_ids
        self.vec_align_len_ids = \
            cce_params.VECTOR_INST_BLOCK_WIDTH // self.type_size_ids
        self.uint8_max_value = 255
        # the max value int 64bit, 2**64 - 1
        self.uint64_all_one = \
            tvm.const(18446744073709551615 // 2,
                      "uint64") * tvm.const(2, "uint64") + \
            tvm.const(1, "uint64")
        self.mask_all_one = tvm.const(-1, "uint64")
        self.mask = ib_.allocate("uint64", (2, ),
                                 name="mask",
                                 scope=cce_params.scope_reg)
        self.mask_cast = ib_.allocate("uint64", (1, ),
                                      name="mask",
                                      scope=cce_params.scope_reg)

        self.align_offset = ib_.allocate("int32", (2, ),
                                         name="align_offset",
                                         scope=cce_params.scope_reg)

        self.device_core_num = cce.cce_conf.get_soc_spec(cce.cce_conf.CORE_NUM)
        self.block = tvm.thread_axis("blockIdx.x")
        self.ib_.scope_attr(self.block, "thread_extent", self.device_core_num)
        self.in_multi_core_mode = False
        self.segment_ids_ub = 0
        self.input_ub = 0
        self.output_ub = 0
        self.output_buf, self.segment_ids, self.gm_align = buffers
        self.half_mask = 64
        self.cast_ub = 0
        self.dst_ub = 0
        self.segment_ids_ub = 0
        self.dtype = src_dtype
        self.type_size = cce.cce_intrin.get_bit_len(src_dtype) // 8
        self.cp_align_len = cce_params.BLOCK_REDUCE_INT8 // self.type_size
        self.cp_align_len_int8 = cce_params.BLOCK_REDUCE_INT8 // 1
        self.type_size_ratio = _ceil_div(self.type_size_ids, self.type_size)
        self.vec_align_len = \
            cce_params.VECTOR_INST_BLOCK_WIDTH // self.type_size
        self.vec_align_len_int8 = cce_params.VECTOR_INST_BLOCK_WIDTH // 1
        self.unified_buffer_len = \
            cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE) // self.type_size

    def refresh_dtype(self, dtype):
        """
        update parameters for Segment
        """
        self.dtype = dtype
        self.type_size = cce.cce_intrin.get_bit_len(dtype) // 8
        self.cp_align_len = cce_params.BLOCK_REDUCE_INT8 // self.type_size
        self.type_size_ratio = _ceil_div(self.type_size_ids, self.type_size)
        self.unified_buffer_len = \
            cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE) // self.type_size
        self.vec_align_len = \
            cce_params.VECTOR_INST_BLOCK_WIDTH // self.type_size

    def get_align(self, offset):
        """
        offset: offset value to be convert
        return: converted mask
        """
        if isinstance(offset, int):
            return offset % self.cp_align_len

        # avoid canonical.cc bug
        self.align_offset[0] = 1
        self.align_offset[0] = \
            offset % (self.cp_align_len * self.align_offset[0])

        return self.align_offset[0]

    def get_mask_head(self, align_offset):
        """
        offset: offset value to be convert
        return: converted mask
        """
        return [self.mask_all_one, self.uint64_all_one - \
                tvm.const((2**align_offset - 1), dtype='uint64')]

    def get_mask_tail(self, data_len):
        """
        offset: offset value to be convert
        return: converted mask
        """

        if data_len < self.half_mask:
            return [
                tvm.const(0, dtype='uint64'),
                tvm.const(2**data_len - 1, dtype='uint64')
            ]
        return [
            tvm.const(2**(data_len - self.half_mask) - 1, dtype='uint64'),
            self.mask_all_one
        ]

    def get_mask(self, align, data_len):
        """
        offset: offset value to be convert
        return: converted mask
        """
        if isinstance(align, int) and isinstance(data_len, int):
            if data_len < self.half_mask:
                return [
                    tvm.const(0, dtype='uint64'),
                    tvm.const((2**data_len - 1) * (2**align), dtype='uint64')
                ]
            return [
                tvm.const(2**(data_len - self.half_mask) - 1, dtype='uint64'),
                self.uint64_all_one - tvm.const((2**align - 1), dtype='uint64')
            ]

        with self.ib_.if_scope(data_len <= self.half_mask):
            self.mask[0] = tvm.const(0, "uint64")
            with self.ib_.for_range(0, data_len, for_type="serial", name="i"):
                self.mask[0] = self.mask[0] * tvm.const(2, "uint64") + \
                               tvm.const(1, "uint64")
            with self.ib_.for_range(0, align, for_type="serial", name="i"):
                self.mask[0] = self.mask[0] * tvm.const(2, "uint64")

            self.mask[1] = tvm.const(0, "uint64")

        with self.ib_.else_scope():
            self.mask[0] = self.uint64_all_one
            with self.ib_.for_range(0, align, for_type="serial", name="i"):
                self.mask[0] = self.mask[0] * tvm.const(2, "uint64")

            self.mask[1] = tvm.const(0, "uint64")
            with self.ib_.for_range(0,
                                    data_len - self.half_mask,
                                    for_type="serial"):
                self.mask[1] = self.mask[1] * tvm.const(2, "uint64") + \
                               tvm.const(1, "uint64")

        return [self.mask[1], self.mask[0]]


def _is_32_byte_align(in_shape, dtype):
    byte_each_num = cce.cce_intrin.get_bit_len(dtype) // 8
    byte_all = int(in_shape[1] * byte_each_num)
    if (byte_all % 32) == 0:
        return True

    return False


def _get_vec_align_len(dtype):
    """
    dtype: dtype
    return: vec_align_len
    """
    type_size = cce.cce_intrin.get_bit_len(dtype) // 8

    return cce_params.VECTOR_INST_BLOCK_WIDTH // type_size


def _do_vector_dup(ubuf, dup_len, dtype, params, val=0):
    """
    ubuf: target ubuf
    dup_len : length to be dup
    dtype: target ubuf offset
    params : parameters
    """
    buf, buf_offset = ubuf

    def _dump(data_len, cycle_offset):
        """
        data_len : length to dup
        cycle_offset : cycle_offset
        """
        params.ib_.emit(
            tvm.call_extern(
                dtype, 'vector_dup',
                buf.access_ptr("rw", offset=buf_offset + cycle_offset),
                tvm.const(val, dtype),
                _ceil_div(data_len, _get_vec_align_len(dtype)), 1, 1, 8, 8))

    vec_buffer_max_len = params.uint8_max_value * params.vec_align_len
    num_cycle = dup_len // vec_buffer_max_len
    with params.ib_.for_range(0, num_cycle, for_type="serial", name="i") as i:
        _dump(vec_buffer_max_len, i * vec_buffer_max_len)
    tail_len = dup_len % vec_buffer_max_len
    with params.ib_.if_scope(tail_len > 0):
        _dump(tail_len, num_cycle * vec_buffer_max_len)


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
# pylint: disable=locally-disabled,too-many-statements
def _do_cast(params, src_ubuf, dst_ubuf, data_len, dtype_src, dtype_dst):
    # vconv count of elements every time
    ele_cnt = 128
    vconv_group = 255
    # vconv_ele = vconv_group * ele_cnt  #32640
    vconv_ele = 32640
    if dtype_src == "int8" and dtype_dst == "float16":
        vconv_insn = "vconv_s82f16"
        src_stride = 4
        dst_stride = 8
    elif dtype_src == "uint8" and dtype_dst == "float16":
        vconv_insn = "vconv_u82f16"
        src_stride = 4
        dst_stride = 8
    elif dtype_src == "float16" and dtype_dst == "int8":
        vconv_insn = "vconv_f162s8"
        src_stride = 8
        dst_stride = 4
    else:
        vconv_insn = "vconv_f162u8"
        src_stride = 8
        dst_stride = 4
    ub_vconv_cycle_index = data_len // vconv_ele
    ub_vconv_cycle1 = ub_vconv_cycle_index + tvm.const(1, "uint64")

    ub_vconv_cycle_mod = data_len % vconv_ele
    with params.ib_.if_scope(ub_vconv_cycle_mod > 0):
        with params.ib_.for_range(0, ub_vconv_cycle1, name="k") as k:
            with params.ib_.if_scope(k < ub_vconv_cycle_index):
                params.ib_.emit(
                    tvm.call_extern("uint64", 'set_vector_mask',
                                    params.uint64_all_one,
                                    params.uint64_all_one))
                params.ib_.emit(
                    tvm.call_extern(
                        dtype_dst, vconv_insn,
                        dst_ubuf.access_ptr("w", offset=k * vconv_ele),
                        src_ubuf.access_ptr('r', offset=k * vconv_ele),
                        vconv_group, 1, 1, dst_stride, src_stride))
            with params.ib_.else_scope():
                ub_vconv_mod = data_len - ub_vconv_cycle_index * vconv_ele
                ub_vconv_mod_repeat_more = ub_vconv_mod // ele_cnt
                ub_vconv_mod_repeat_one = ub_vconv_mod - (
                        ub_vconv_mod_repeat_more * ele_cnt)
                with params.ib_.if_scope(ub_vconv_mod_repeat_more > 1):
                    params.ib_.emit(
                        tvm.call_extern("uint64", 'set_vector_mask',
                                        params.uint64_all_one,
                                        params.uint64_all_one))
                    params.ib_.emit(
                        tvm.call_extern(
                            dtype_dst, vconv_insn,
                            dst_ubuf.access_ptr("w",
                                                offset=ub_vconv_cycle_index *
                                                       vconv_ele),
                            src_ubuf.access_ptr('r',
                                                offset=ub_vconv_cycle_index *
                                                       vconv_ele),
                            ub_vconv_mod_repeat_more, 1, 1, dst_stride,
                            src_stride))
                with params.ib_.if_scope(ub_vconv_mod_repeat_more == 1):
                    params.ib_.emit(
                        tvm.call_extern("uint64", 'set_vector_mask',
                                        params.uint64_all_one,
                                        params.uint64_all_one))

                    params.ib_.emit(
                        tvm.call_extern(
                            dtype_dst, vconv_insn,
                            dst_ubuf.access_ptr("w",
                                                offset=ub_vconv_cycle_index * \
                                                       vconv_ele),
                            src_ubuf.access_ptr(
                                'r', offset=ub_vconv_cycle_index * vconv_ele),
                            1, 1, 1, 0, 0))
                with params.ib_.if_scope(ub_vconv_mod_repeat_one > 0):
                    with params.ib_.if_scope(ub_vconv_mod_repeat_one == 1):
                        params.ib_.emit(
                            tvm.call_extern("uint64", 'set_vector_mask', 0, 1))
                        params.ib_.emit(
                            tvm.call_extern(
                                dtype_dst, vconv_insn,
                                dst_ubuf.access_ptr(
                                    "w",
                                    offset=ub_vconv_cycle_index *
                                           vconv_ele +
                                           ub_vconv_mod_repeat_more * ele_cnt),
                                src_ubuf.access_ptr(
                                    'r',
                                    offset=ub_vconv_cycle_index * vconv_ele +
                                           ub_vconv_mod_repeat_more * ele_cnt),
                                1, 0, 0, 0, 0))
                        params.ib_.emit(
                            tvm.call_extern("uint64", 'set_vector_mask',
                                            params.uint64_all_one,
                                            params.uint64_all_one))
                    with params.ib_.if_scope(ub_vconv_mod_repeat_one > 1):
                        with params.ib_.if_scope(
                                ub_vconv_mod_repeat_one <= 64):
                            params.mask_cast[0] = tvm.const(0, "uint64")
                            with params.ib_.for_range(0,
                                                      ub_vconv_mod_repeat_one,
                                                      for_type="serial",
                                                      name="i"):
                                params.mask_cast[
                                    0] = params.mask_cast[0] * tvm.const(
                                    2, "uint64") + tvm.const(1, "uint64")
                            params.ib_.emit(
                                tvm.call_extern("uint64", 'set_vector_mask', 0,
                                                params.mask_cast[0]))

                            params.ib_.emit(
                                tvm.call_extern(
                                    dtype_dst, vconv_insn,
                                    dst_ubuf.access_ptr(
                                        "w",
                                        offset=ub_vconv_cycle_index * \
                                               vconv_ele + \
                                               ub_vconv_mod_repeat_more * \
                                               ele_cnt),
                                    src_ubuf.access_ptr(
                                        'r',
                                        offset=ub_vconv_cycle_index * \
                                               vconv_ele + \
                                               ub_vconv_mod_repeat_more * \
                                               ele_cnt),
                                    1, 1, 1, 0, 0))
                            params.ib_.emit(
                                tvm.call_extern("uint64", 'set_vector_mask',
                                                params.uint64_all_one,
                                                params.uint64_all_one))
                        with params.ib_.if_scope(ub_vconv_mod_repeat_one > 64):
                            offset = ub_vconv_mod_repeat_one - 64
                            params.mask_cast[0] = tvm.const(0, "uint64")
                            with params.ib_.for_range(0,
                                                      offset,
                                                      for_type="serial",
                                                      name="i"):
                                params.mask_cast[
                                    0] = params.mask_cast[0] * tvm.const(
                                    2, "uint64") + tvm.const(1, "uint64")
                            params.ib_.emit(
                                tvm.call_extern("uint64", 'set_vector_mask',
                                                params.mask_cast[0],
                                                params.uint64_all_one))

                            params.ib_.emit(
                                tvm.call_extern(
                                    dtype_dst, vconv_insn,
                                    dst_ubuf.access_ptr(
                                        "w",
                                        offset=ub_vconv_cycle_index * \
                                               vconv_ele + \
                                               ub_vconv_mod_repeat_more * \
                                               ele_cnt),
                                    src_ubuf.access_ptr(
                                        'r',
                                        offset=ub_vconv_cycle_index * \
                                               vconv_ele + \
                                               ub_vconv_mod_repeat_more * \
                                               ele_cnt),
                                    1, 1, 1, 0, 0))
                            params.ib_.emit(
                                tvm.call_extern("uint64", 'set_vector_mask',
                                                params.uint64_all_one,
                                                params.uint64_all_one))

    with params.ib_.if_scope(ub_vconv_cycle_mod == 0):

        with params.ib_.for_range(0, ub_vconv_cycle_index, name="k") as k:
            params.ib_.emit(
                tvm.call_extern("uint64", 'set_vector_mask',
                                params.uint64_all_one, params.uint64_all_one))
            params.ib_.emit(
                tvm.call_extern(dtype_dst, vconv_insn,
                                dst_ubuf.access_ptr("w", offset=k * vconv_ele),
                                src_ubuf.access_ptr('r', offset=k * vconv_ele),
                                vconv_group, 1, 1, dst_stride, src_stride))


def _do_vadd(data_len, output_offset, input_offset, masks, params):
    """
    data_len : length to be add
    output_offset: output offset
    input_offset: input offset
    masks: masks
    params : parameters
    """
    def _vadd(add_len, cycle_offset):
        """
        add_len : length to dup
        cycle_offset : cycle_offset
        """
        params.ib_.emit(
            tvm.call_extern(
                params.dtype, 'vadd',
                params.output_ub.access_ptr('w',
                                            offset=output_offset +
                                                   cycle_offset),
                params.output_ub.access_ptr('r',
                                            offset=output_offset +
                                                   cycle_offset),
                params.input_ub.access_ptr("r",
                                           offset=input_offset + cycle_offset),
                _ceil_div(add_len, params.vec_align_len), 1, 1, 1, 8, 8, 8))

    with params.ib_.if_scope(data_len <= params.vec_align_len):
        if list(masks) == [params.mask_all_one, params.mask_all_one]:
            _vadd(data_len, 0)
        else:
            params.ib_.emit(
                tvm.call_extern("uint64", 'set_vector_mask', masks[0],
                                masks[1]))
            _vadd(data_len, 0)
            params.ib_.emit(
                tvm.call_extern("uint64", 'set_vector_mask',
                                params.mask_all_one, params.mask_all_one))
    with params.ib_.else_scope():
        if list(masks) == [params.mask_all_one, params.mask_all_one]:
            res_len = data_len
            first_offset = 0
        else:
            params.ib_.emit(
                tvm.call_extern("uint64", 'set_vector_mask', masks[0],
                                masks[1]))
            _vadd(params.vec_align_len, 0)
            params.ib_.emit(
                tvm.call_extern("uint64", 'set_vector_mask',
                                params.mask_all_one, params.mask_all_one))

            res_len = data_len - params.vec_align_len
            first_offset = params.vec_align_len

        vec_buffer_max_len = params.uint8_max_value * params.vec_align_len
        num_cycle = res_len // vec_buffer_max_len
        with params.ib_.for_range(0, num_cycle, for_type="serial",
                                  name="i") as i:
            _vadd(vec_buffer_max_len, i * vec_buffer_max_len + first_offset)
        tail_len = res_len % vec_buffer_max_len
        with params.ib_.if_scope(tail_len > 0):
            _vadd(tail_len, num_cycle * vec_buffer_max_len + first_offset)


def _do_cp_input_buf(input_buf, data_len, offset, params, align_offset=0):
    """
    input_buf: gm input buf
    data_len : length to be add
    offset: gm offset
    params : parameters
    """

    with params.ib_.if_scope(offset < 0):
        params.ib_.emit(
            tvm.call_extern(
                params.dtype,
                'copy_gm_to_ubuf',
                params.input_ub.access_ptr("rw", offset=0),
                input_buf.access_ptr("r", offset=(offset + align_offset)),
                0,
                1,
                _ceil_div(data_len, params.cp_align_len),
                # clear warning, 0 < align_offset < params.cp_align_len
                0,
                0))
        params.ib_.emit(
            tvm.call_extern(
                params.dtype,
                'copy_ubuf_to_gm',
                params.gm_align.access_ptr("rw", offset=align_offset),
                params.input_ub.access_ptr("r", offset=0),
                0,
                1,
                _ceil_div(data_len, params.cp_align_len),
                # clear warning, 0 < align_offset < params.cp_align_len
                0,
                0))

        params.ib_.emit(
            tvm.call_extern(
                params.dtype,
                'copy_gm_to_ubuf',
                params.input_ub.access_ptr("rw", offset=0),
                params.gm_align.access_ptr("r", offset=0),
                0,
                1,
                _ceil_div(data_len, params.cp_align_len),
                # clear warning, 0 < align_offset < params.cp_align_len
                0,
                0))
    with params.ib_.else_scope():
        params.ib_.emit(
            tvm.call_extern(params.dtype, 'copy_gm_to_ubuf',
                            params.input_ub.access_ptr("rw", offset=0),
                            input_buf.access_ptr("r", offset=offset), 0, 1,
                            _ceil_div(data_len, params.cp_align_len), 0, 0))


def _do_cp_input_buf_cast(input_buf, data_len, offset, params, align_offset=0):
    """
    input_buf: gm input buf
    data_len : length to be add
    offset: gm offset
    params : parameters
    """

    with params.ib_.if_scope(offset < 0):
        params.ib_.emit(
            tvm.call_extern(
                params.dtype,
                'copy_gm_to_ubuf',
                params.input_ub.access_ptr("rw", offset=0),
                input_buf.access_ptr("r", offset=offset + align_offset),
                0,
                1,
                _ceil_div(data_len, params.cp_align_len_int8),
                # clear warning, 0 < align_offset < params.cp_align_len
                0,
                0))
        params.ib_.emit(
            tvm.call_extern(
                params.dtype,
                'copy_ubuf_to_gm',
                params.gm_align.access_ptr("rw", offset=align_offset),
                params.input_ub.access_ptr("r", offset=0),
                0,
                1,
                _ceil_div(data_len, params.cp_align_len_int8),
                # clear warning, 0 < align_offset < params.cp_align_len
                0,
                0))

        params.ib_.emit(
            tvm.call_extern(
                params.dtype,
                'copy_gm_to_ubuf',
                params.input_ub.access_ptr("rw", offset=0),
                params.gm_align.access_ptr("r", offset=0),
                0,
                1,
                _ceil_div(data_len, params.cp_align_len_int8),
                # clear warning, 0 < align_offset < params.cp_align_len
                0,
                0))
    with params.ib_.else_scope():
        params.ib_.emit(
            tvm.call_extern(params.dtype, 'copy_gm_to_ubuf',
                            params.cast_ub.access_ptr("rw", offset=0),
                            input_buf.access_ptr("r", offset=offset), 0, 1,
                            _ceil_div(data_len, params.cp_align_len_int8), 0,
                            0))

    _do_cast(params, params.cast_ub, params.input_ub, data_len,
             params.src_dtype, "float16")

    params.refresh_dtype("float16")


def _align_offset_func(align_offset, input_offset, output_offset, _main_cp_fun,
                       params):
    """
    align_offset: align offset
    input_offset : input offset
    output_offset: output offset
    _main_cp_fun: _main_cp_fun
    params : parameters
    """
    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        params.refresh_dtype("float16")
    element_len = params.element_len
    num_cycle = (element_len + align_offset) // params.vec_align_len
    with params.ib_.if_scope((element_len < params.vec_align_len)
                             and num_cycle == 0):
        _main_cp_fun(element_len + align_offset, output_offset - align_offset,
                     input_offset - align_offset,
                     params.get_mask(align_offset, element_len))
    with params.ib_.else_scope():
        _main_cp_fun(params.vec_align_len, output_offset - align_offset,
                     input_offset - align_offset,
                     params.get_mask_head(align_offset))
        with params.ib_.if_scope(num_cycle > 1):
            block_offset = params.vec_align_len
            _main_cp_fun((num_cycle - 1) * params.vec_align_len,
                         block_offset + output_offset - align_offset,
                         block_offset + input_offset - align_offset,
                         (params.mask_all_one, params.mask_all_one))

        tail_len = (element_len + align_offset) % params.vec_align_len
        with params.ib_.if_scope(tail_len > 0):
            block_offset = num_cycle * params.vec_align_len
            _main_cp_fun(tail_len, block_offset + output_offset - align_offset,
                         block_offset + input_offset - align_offset,
                         params.get_mask_tail(tail_len))


def _multi_core_tail_align(input_offset, output_offset, _main_cp_fun, params):
    """
    input_offset : input offset
    output_offset: output offset
    _main_cp_fun: _main_cp_fun
    params : parameters
    """
    align_offset = params.get_align(output_offset)
    with params.ib_.if_scope(
            params.element_len <= params.cp_align_len - align_offset):

        _main_cp_fun(1, output_offset - align_offset,
                     input_offset - align_offset,
                     params.get_mask(align_offset,
                                     params.element_len), align_offset)
    with params.ib_.else_scope():
        _main_cp_fun(
            1, output_offset - align_offset, input_offset - align_offset,
            params.get_mask(align_offset, params.cp_align_len - align_offset),
            align_offset)


def _do_cp_ids(segment_ids_ub, segment_ids, gm_offset, data_len, params):
    """
    segment_ids_ub : segment_ids_ub
    segment_ids: segment_ids
    gm_offset: gm offset
    data_len : length to be copy
    params : parameters
    """
    params.ib_.emit(
        tvm.call_extern('int32', 'copy_gm_to_ubuf',
                        segment_ids_ub.access_ptr("rw", offset=0),
                        segment_ids.access_ptr("r", offset=gm_offset), 0, 1,
                        _ceil_div(data_len, params.cp_align_len_ids), 0, 0))
    args_str = tvm.call_pure_intrin("int32", "tvm_cce_string_print",
                                    "PIPE_ALL")
    params.ib_.emit(tvm.call_extern('int32', 'pipe_barrier', args_str))


def _apply_bufs_cast(ids_len, input_len, output_len, params):
    """
    ids_len : id length
    input_len : input length
    output_len: output length
    params : parameters
    """

    params.refresh_dtype("float16")

    if input_len > output_len:
        total_buf_len = _ceil_fill(ids_len * params.type_size_ratio,
                                   params.vec_align_len) + \
                        _ceil_fill(input_len, params.vec_align_len) + \
                        _ceil_fill(output_len, params.vec_align_len) + \
                        _ceil_fill(input_len, params.vec_align_len_int8)
    else:
        total_buf_len = _ceil_fill(ids_len * params.type_size_ratio,
                                   params.vec_align_len) + \
                        _ceil_fill(input_len, params.vec_align_len) + \
                        _ceil_fill(output_len, params.vec_align_len) + \
                        _ceil_fill(output_len, params.vec_align_len)

    if total_buf_len > params.unified_buffer_len:
        return False

    params.segment_ids_ub = _apply_for_new_alloc(params.ib_, 'int32', ids_len,
                                                 params.vec_align_len,
                                                 cce_params.scope_ubuf)
    params.input_ub = _apply_for_new_alloc(params.ib_, 'float16', input_len,
                                           params.vec_align_len,
                                           cce_params.scope_ubuf)
    params.output_ub = _apply_for_new_alloc(params.ib_, 'float16', output_len,
                                            params.vec_align_len,
                                            cce_params.scope_ubuf)

    if input_len > output_len:
        params.cast_ub = _apply_for_new_alloc(params.ib_, params.src_dtype,
                                              input_len,
                                              params.vec_align_len_int8,
                                              cce_params.scope_ubuf)
    else:
        params.cast_ub = _apply_for_new_alloc(params.ib_, params.src_dtype,
                                              output_len, params.vec_align_len,
                                              cce_params.scope_ubuf)

    return params.segment_ids_ub, params.input_ub, params.output_ub, \
           params.cast_ub


def _apply_bufs(ids_len, input_len, output_len, params):
    """
    ids_len : id length
    input_len : input length
    output_len: output length
    params : parameters
    """
    total_buf_len = _ceil_fill(ids_len * params.type_size_ratio,
                               params.vec_align_len) + \
                    _ceil_fill(input_len, params.vec_align_len) + \
                    _ceil_fill(output_len, params.vec_align_len)

    if total_buf_len > params.unified_buffer_len:
        return False

    params.segment_ids_ub = _apply_for_new_alloc(params.ib_, 'int32', ids_len,
                                                 params.vec_align_len,
                                                 cce_params.scope_ubuf)
    params.input_ub = _apply_for_new_alloc(params.ib_, params.dtype, input_len,
                                           params.vec_align_len,
                                           cce_params.scope_ubuf)
    params.output_ub = _apply_for_new_alloc(params.ib_, params.dtype,
                                            output_len, params.vec_align_len,
                                            cce_params.scope_ubuf)

    return params.segment_ids_ub, params.input_ub, params.output_ub


def _multi_core(_core_func, num_segments, params):
    """
     _core_func : _core_func
     num_segments : num_segments
     params : parameters
    """
    element_num_of_core = _ceil_div(num_segments, params.device_core_num)

    out_begin = params.block.var * element_num_of_core
    out_end = params.block.var * element_num_of_core + element_num_of_core

    with params.ib_.if_scope(out_end < num_segments):
        _core_func(out_begin, out_end)
    with params.ib_.else_scope():
        with params.ib_.if_scope(out_begin < num_segments):
            _core_func(out_begin, num_segments)


def _all_in_fun(num_segments, input_buf, segment_ids, params):
    """
     num_segments : num_segments
     input_buf : input buf
     segment_ids : segment_ids
     params : parameters
    """
    ids_len = params.in_shape[0]
    input_len = _prod(params.in_shape)
    element_len = params.element_len
    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        params.refresh_dtype(params.src_dtype)
        if element_len % params.cp_align_len_int8 != 0:
            return False

    else:
        if element_len % params.cp_align_len != 0:
            return False

    output_len = _ceil_div(num_segments, params.device_core_num) * element_len
    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        if _apply_bufs_cast(ids_len, input_len, output_len, params) is False:
            return False
    else:
        if _apply_bufs(ids_len, input_len, output_len, params) is False:
            return False

    _do_cp_ids(params.segment_ids_ub, segment_ids, 0, ids_len, params)
    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        params.refresh_dtype(params.src_dtype)
        _do_cp_input_buf_cast(input_buf, input_len, 0, params)

        params.refresh_dtype("float16")
        _do_vector_dup((params.output_ub, 0), output_len, "float16", params)
    else:
        _do_cp_input_buf(input_buf, input_len, 0, params)
        _do_vector_dup((params.output_ub, 0), output_len, params.dtype, params)

    def _main_cp_fun(data_len, output_offset, input_offset, masks):
        """
         data_len : data_len
         output_offset : output_offset
         input_offset: input_offset
         mask1 : mask1
         massk2 : massk2
        """

        _do_vadd(data_len, output_offset, input_offset, masks, params)

    def _core_func(out_begin, out_end):
        """
         out_begin: core out index begin
         out_end : core out index end
        """
        if params.src_dtype == "int8" or params.src_dtype == "uint8":
            params.refresh_dtype("float16")
        with params.ib_.for_range(0, ids_len, for_type="serial",
                                  name="index") as index:
            segment_id = params.segment_ids_ub.vload(index, 'int32')
            with params.ib_.if_scope(
                    tvm.all(out_begin <= segment_id, segment_id < out_end)):
                input_offset = index * element_len
                output_offset = (segment_id - out_begin) * element_len
                _align_offset_func(0, input_offset, output_offset,
                                   _main_cp_fun, params)

        block_num = params.ib_.allocate("int32", (1, ),
                                        name="data_len",
                                        scope=cce_params.scope_reg)
        block_num[0] = out_end - out_begin

        data_len = block_num[0] * element_len

        num_cp = data_len // params.cp_align_len

        if params.src_dtype == "int8" or params.src_dtype == "uint8":

            params.refresh_dtype(params.src_dtype)

            _do_cast(params, params.output_ub, params.cast_ub, data_len,
                     "float16", params.src_dtype)

            num_cp = data_len // params.cp_align_len_int8

            params.ib_.emit(
                tvm.call_extern(
                    params.dtype, 'copy_ubuf_to_gm',
                    params.output_buf.access_ptr("rw",
                                                 offset=out_begin *
                                                        element_len),
                    params.cast_ub.access_ptr("r", offset=0), 0, 1, num_cp, 0,
                    0))
        else:

            params.ib_.emit(
                tvm.call_extern(
                    params.dtype, 'copy_ubuf_to_gm',
                    params.output_buf.access_ptr("rw",
                                                 offset=out_begin *
                                                        element_len),
                    params.output_ub.access_ptr("r", offset=0), 0, 1, num_cp,
                    0, 0))

    _multi_core(_core_func, num_segments, params)

    return True


def _one_in_multi_out_fun(num_segments, input_buf, segment_ids, params):
    """
     num_segments : num_segments
     input_buf : input buf
     segment_ids : segment_ids
     params : parameters
    """
    ids_len = params.in_shape[0]
    input_len = _prod(params.in_shape)
    element_len = params.element_len
    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        params.refresh_dtype("float16")
        if element_len % params.cp_align_len_int8 != 0:
            return False
    else:
        if element_len % params.cp_align_len != 0:
            return False
    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        if _apply_bufs_cast(ids_len, input_len, element_len, params) is False:
            return False
    else:
        if _apply_bufs(ids_len, input_len, element_len, params) is False:
            return False

    _do_cp_ids(params.segment_ids_ub, segment_ids, 0, ids_len, params)
    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        params.refresh_dtype(params.src_dtype)
        _do_cp_input_buf_cast(input_buf, input_len, 0, params)
    else:
        _do_cp_input_buf(input_buf, input_len, 0, params)

    def _main_cp_fun(data_len, output_offset, input_offset, masks):
        """
         data_len : data_len
         output_offset : output_offset
         input_offset: input_offset
         mask1 : mask1
         massk2 : massk2
        """
        if params.src_dtype == "int8" or params.src_dtype == "uint8":
            params.refresh_dtype("float16")
        _do_vadd(data_len, output_offset, input_offset, masks, params)

    # pylint: disable=locally-disabled,unused-argument
    def _ub_cp(_main_cp_fun, data_len, index, params, block_offset):
        """
         _main_cp_fun : _main_cp_fun
         data_len : data_len
         index : index
         params: parameters
         block_offset : block_offset
        """
        with params.ib_.for_range(0,
                                  params.ids_len,
                                  for_type="serial",
                                  name="j") as j:
            segment_id = params.segment_ids_ub.vload(j)
            with params.ib_.if_scope(segment_id == index):
                _align_offset_func(0, j * params.element_len, 0, _main_cp_fun,
                                   params)

    def _core_func(out_begin, out_end):
        """
         out_begin: core out index begin
         out_end : core out index end
        """
        with params.ib_.for_range(0,
                                  out_end - out_begin,
                                  for_type="serial",
                                  name="index") as index:
            if params.src_dtype == "int8" or params.src_dtype == "uint8":
                _part_cp_cast((_main_cp_fun, _ub_cp), element_len,
                              index + out_begin, params)
            else:
                _part_cp((_main_cp_fun, _ub_cp), element_len,
                         index + out_begin, params)

    _multi_core(_core_func, num_segments, params)

    return True


def _ub_cp_fun(_main_cp_fun, data_len, index, params, block_offset=0):
    """
     _main_cp_fun : _main_cp_fun
     data_len : data_len
     index : index
     params: parameters
     block_offset : block_offset
    """
    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        params.refresh_dtype("float16")
    with params.ib_.for_range(0,
                              params.ids_len,
                              for_type="serial",
                              name="id_index") as id_index:
        segment_id = params.segment_ids_ub.vload(id_index)
        with params.ib_.if_scope(segment_id == index):
            _main_cp_fun(data_len, 0,
                         id_index * params.element_len + block_offset,
                         (params.mask_all_one, params.mask_all_one))


def _ub_cp_part_ids(ids_block, data_block, index, _main_cp_fun, params):
    """
     ids_block : ids block parameters
     data_block : data block parameters
     index : index
     _main_cp_fun : _main_cp_fun
     params : parameters
    """
    ids_block_len, ids_block_offset = ids_block
    data_len, data_block_offset = data_block
    _do_cp_ids(params.segment_ids_ub, params.segment_ids, ids_block_offset,
               ids_block_len, params)
    with params.ib_.for_range(0, ids_block_len, for_type="serial",
                              name="j") as j:
        segment_id = params.segment_ids_ub.vload(j)
        with params.ib_.if_scope(segment_id == index):
            _main_cp_fun(data_len, 0,
                         (ids_block_offset + j) * params.element_len +
                         data_block_offset,
                         (params.mask_all_one, params.mask_all_one))


def _ub_cp_part_ids_tail(ids_block, data_block, index, _main_cp_fun, params):
    """
     ids_block : ids block parameters
     data_block : data block parameters
     index : index
     _main_cp_fun : _main_cp_fun
     params : parameters
    """
    ids_block_len, ids_block_offset = ids_block
    data_len, data_block_offset = data_block
    _do_cp_ids(params.segment_ids_ub, params.segment_ids, ids_block_offset,
               ids_block_len, params)
    with params.ib_.for_range(0,
                              ids_block_len,
                              for_type="serial",
                              name="id_index") as id_index:
        segment_id = params.segment_ids_ub.vload(id_index)
        with params.ib_.if_scope(segment_id == index):
            body_data_len = (data_len //
                             params.vec_align_len) * params.vec_align_len
            with params.ib_.if_scope(data_len >= params.vec_align_len):
                _main_cp_fun(
                    body_data_len, 0,
                    (ids_block_offset + id_index) * params.element_len +
                    data_block_offset,
                    (params.mask_all_one, params.mask_all_one))
            tail_len = data_len % params.vec_align_len
            with params.ib_.if_scope(tail_len > 0):
                _main_cp_fun(
                    tail_len, body_data_len,
                    (ids_block_offset + id_index) * params.element_len +
                    data_block_offset + body_data_len,
                    params.get_mask_tail(tail_len))
        # fix multi core over write
        over_write_num = _ceil_div(
            (params.cp_align_len - params.get_align(data_len)),
            params.element_len)

        with params.ib_.for_range(0,
                                  over_write_num,
                                  for_type="serial",
                                  name="i") as i:
            with params.ib_.if_scope(segment_id == index + 1 + i):
                _multi_core_tail_align(
                    (ids_block_offset + id_index) * params.element_len +
                    data_block_offset,
                    i * params.element_len + data_len,
                    _main_cp_fun,
                    params,
                    )


def _ub_cp_fun_tail(_main_cp_fun, data_len, index, params, block_offset=0):
    """
     _main_cp_fun : _main_cp_fun
     data_len : data_len
     index : index
     params: parameters
     block_offset : block_offset
    """
    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        params.refresh_dtype("float16")
    with params.ib_.for_range(0,
                              params.ids_len,
                              for_type="serial",
                              name="id_index") as id_index:
        segment_id = params.segment_ids_ub.vload(id_index)
        with params.ib_.if_scope(segment_id == index):
            num_cycle = data_len // params.vec_align_len
            body_data_len = num_cycle * params.vec_align_len
            with params.ib_.if_scope(num_cycle > 0):
                _main_cp_fun(body_data_len, 0,
                             id_index * params.element_len + block_offset,
                             (params.mask_all_one, params.mask_all_one))
            tail_len = data_len % params.vec_align_len
            with params.ib_.if_scope(tail_len > 0):
                _main_cp_fun(
                    tail_len, body_data_len, id_index * params.element_len + \
                                             block_offset + body_data_len,
                    params.get_mask_tail(tail_len))
        # fix multi core over write

        align_offset = params.get_align(data_len)
        over_write_num = _ceil_div((params.cp_align_len - align_offset),
                                   params.element_len)

        with params.ib_.for_range(0,
                                  over_write_num,
                                  for_type="serial",
                                  name="i") as i:
            with params.ib_.if_scope(segment_id == index + 1 + i):
                _multi_core_tail_align((id_index * params.element_len),
                                       (i * params.element_len + data_len),
                                       _main_cp_fun, params)


def _part_cp(funs, data_len, index, params, block_offset=0):
    """
     funs : funs
     data_len : data_len
     index : index
     params: parameters
     block_offset : block_offset
    """
    _main_cp_fun, ub_cp_func = funs
    _do_vector_dup((params.output_ub, 0), data_len, params.dtype, params)
    ub_cp_func(_main_cp_fun, data_len, index, params, block_offset)
    params.ib_.emit(
        tvm.call_extern(
            params.dtype, 'copy_ubuf_to_gm',
            params.output_buf.access_ptr("rw",
                                         offset=index * params.element_len +
                                                block_offset),
            params.output_ub.access_ptr("r", offset=0), 0, 1,
            _ceil_div(data_len, params.cp_align_len), 0, 0))


def _part_cp_cast(funs, data_len, index, params, block_offset=0):
    """
     funs : funs
     data_len : data_len
     index : index
     params: parameters
     block_offset : block_offset
    """
    params.refresh_dtype("float16")
    _main_cp_fun, ub_cp_func = funs
    _do_vector_dup((params.output_ub, 0), data_len, params.dtype, params)

    ub_cp_func(_main_cp_fun, data_len, index, params, block_offset)
    _do_cast(params, params.output_ub, params.cast_ub, data_len, "float16",
             params.src_dtype)
    cyle = data_len // params.cp_align_len_int8
    tail = data_len % params.cp_align_len_int8
    tail_num = 32 - tail
    if data_len % params.cp_align_len_int8 != 0 and cyle != 0:
        params.dst_ub = _apply_for_new_alloc(params.ib_, params.src_dtype, 32,
                                             params.vec_align_len_int8,
                                             cce_params.scope_ubuf)

        params.refresh_dtype(params.src_dtype)
        params.ib_.emit(
            tvm.call_extern(
                params.dtype, 'copy_ubuf_to_gm',
                params.output_buf.access_ptr(
                    "rw", offset=index * params.element_len + block_offset),
                params.cast_ub.access_ptr("r", offset=0), 0, 1, cyle, 0, 0))

        reg = params.ib_.allocate(params.src_dtype, (32, ),
                                  name="reg",
                                  scope=cce_params.scope_reg)
        for i in range(32):
            params.ib_.emit(
                tvm.call_extern(
                    params.cast_ub.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[i]),
                    params.cast_ub.access_ptr("r",
                                              offset=cyle * 32 - tail_num +
                                                     i)))

            params.ib_.emit(
                tvm.call_extern("int8", "reg_mov",
                                params.dst_ub.access_ptr("w", offset=i),
                                tvm.call_extern(reg.dtype, "reg", reg[i])))

        params.ib_.emit(
            tvm.call_extern(
                params.dtype, 'copy_ubuf_to_gm',
                params.output_buf.access_ptr(
                    "rw",
                    offset=index * params.element_len + block_offset +
                           cyle * 32 - tail_num),
                params.dst_ub.access_ptr("r", offset=0), 0, 1, 1, 0, 0))
    else:

        params.refresh_dtype(params.src_dtype)
        params.ib_.emit(
            tvm.call_extern(
                params.dtype, 'copy_ubuf_to_gm',
                params.output_buf.access_ptr(
                    "rw", offset=index * params.element_len + block_offset),
                params.cast_ub.access_ptr("r", offset=0), 0, 1,
                _ceil_div(data_len, params.cp_align_len_int8), 0, 0))
    params.refresh_dtype("float16")


def _multi_in_multi_out_fun(num_segments, input_buf, segment_ids, params):
    """
     num_segments : num_segments
     input_buf : input buf
     segment_ids : segment_ids
     params : parameters
    """
    ids_len = params.in_shape[0]
    input_len = _prod(params.in_shape[1:])

    element_len = params.element_len
    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        if element_len < 32:
            params.device_core_num = 1
        if _apply_bufs_cast(ids_len, input_len, element_len, params) is False:
            return False

    else:
        if _apply_bufs(ids_len, input_len, element_len, params) is False:
            return False

    _do_cp_ids(params.segment_ids_ub, segment_ids, 0, ids_len, params)

    def _main_cp_fun(data_len,
                     output_offset,
                     input_offset,
                     masks,
                     align_offset=0):
        """
         data_len : data_len
         output_offset : output_offset
         input_offset: input_offset
         masks : mask
        """

        if params.src_dtype == "int8" or params.src_dtype == "uint8":
            params.refresh_dtype(params.src_dtype)

            _do_cp_input_buf_cast(input_buf, data_len, input_offset, params,
                                  align_offset)

            _do_vadd(data_len, output_offset, 0, masks, params)

        else:
            _do_cp_input_buf(input_buf, data_len, input_offset, params,
                             align_offset)
            _do_vadd(data_len, output_offset, 0, masks, params)

    def _core_func(out_begin, out_end):
        """
         out_begin: core out index begin
         out_end : core out index end
        """
        if params.src_dtype == "int8" or params.src_dtype == "uint8":
            params.refresh_dtype("float16")
            with params.ib_.for_range(0,
                                      out_end - 1 - out_begin,
                                      for_type="serial",
                                      name="index") as index:

                _part_cp_cast((_main_cp_fun, _ub_cp_fun), element_len,
                              index + out_begin, params)
            _part_cp_cast((_main_cp_fun, _ub_cp_fun_tail), element_len,
                          out_end - 1, params)
        else:
            with params.ib_.for_range(0,
                                      out_end - 1 - out_begin,
                                      for_type="serial",
                                      name="index") as index:
                _part_cp((_main_cp_fun, _ub_cp_fun), element_len,
                         index + out_begin, params)
            _part_cp((_main_cp_fun, _ub_cp_fun_tail), element_len, out_end - 1,
                     params)

    _multi_core(_core_func, num_segments, params)

    return True


def _multi_in_multi_out_fun_large(num_segments, input_buf, segment_ids,
                                  params):
    """
     num_segments : num_segments
     input_buf : input buf
     segment_ids : segment_ids
     params : parameters
    """

    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        params.refresh_dtype("float16")
    ids_len = params.ids_len

    ids_buf_len = _ceil_fill(ids_len * params.type_size_ratio,
                             params.vec_align_len)
    one_time_len = (params.unified_buffer_len - ids_buf_len) // 2

    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        one_time_len = (params.unified_buffer_len - ids_buf_len) // 3

    one_time_len = (one_time_len //
                    params.vec_align_len) * params.vec_align_len

    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        if ids_buf_len > params.unified_buffer_len // 3 or one_time_len <= 0:
            return False
        if params.element_len % one_time_len < 32:
            params.device_core_num = 1

    else:
        if ids_buf_len > params.unified_buffer_len // 2 or one_time_len <= 0:
            return False
    element_len = params.element_len
    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        _apply_bufs_cast(ids_len, one_time_len, one_time_len, params)

    else:
        _apply_bufs(ids_len, one_time_len, one_time_len, params)

    _do_cp_ids(params.segment_ids_ub, segment_ids, 0, ids_len, params)

    def _main_cp_fun(data_len,
                     output_offset,
                     input_offset,
                     masks,
                     align_offset=0):
        """
        data_len : data_len
        output_offset : output_offset
        input_offset: input_offset
        mask1 : mask1
        massk2 : massk2
        """
        if params.src_dtype == "int8" or params.src_dtype == "uint8":
            params.refresh_dtype(params.src_dtype)
            _do_cp_input_buf_cast(input_buf, data_len, input_offset, params,
                                  align_offset)
            params.refresh_dtype("float16")
            _do_vadd(data_len, output_offset, 0, masks, params)
        else:
            _do_cp_input_buf(input_buf, data_len, input_offset, params,
                             align_offset)
            _do_vadd(data_len, output_offset, 0, masks, params)

    def _core_func(out_begin, out_end):
        """
        out_begin: core out index begin
        out_end : core out index end
        """

        if params.src_dtype == "int8" or params.src_dtype == "uint8":
            params.refresh_dtype("float16")

        with params.ib_.for_range(0,
                                  out_end - 1 - out_begin,
                                  for_type="serial",
                                  name="index") as index:
            if params.src_dtype == "int8" or params.src_dtype == "uint8":
                params.refresh_dtype("float16")
            i = index + out_begin
            num_cycle = element_len // one_time_len

            with params.ib_.for_range(0,
                                      num_cycle,
                                      for_type="serial",
                                      name="j") as j:
                if params.src_dtype == "int8" or params.src_dtype == "uint8":
                    params.refresh_dtype("float16")
                    _part_cp_cast((_main_cp_fun, _ub_cp_fun),
                                  one_time_len,
                                  i,
                                  params,
                                  block_offset=one_time_len * j)
                else:
                    _part_cp((_main_cp_fun, _ub_cp_fun),
                             one_time_len,
                             i,
                             params,
                             block_offset=one_time_len * j)

            tail_len = element_len % one_time_len
            with params.ib_.if_scope(tail_len > 0):
                if params.src_dtype == "int8" or params.src_dtype == "uint8":
                    _part_cp_cast((_main_cp_fun, _ub_cp_fun),
                                  tail_len,
                                  i,
                                  params,
                                  block_offset=one_time_len * num_cycle)
                else:
                    _part_cp((_main_cp_fun, _ub_cp_fun),
                             tail_len,
                             i,
                             params,
                             block_offset=one_time_len * num_cycle)

        num_cycle = element_len // one_time_len
        if params.src_dtype == "int8" or params.src_dtype == "uint8":
            with params.ib_.for_range(0,
                                      num_cycle,
                                      for_type="serial",
                                      name="k") as k:
                _part_cp_cast((_main_cp_fun, _ub_cp_fun),
                              one_time_len,
                              out_end - 1,
                              params,
                              block_offset=one_time_len * k)
            tail_len = element_len % one_time_len
            with params.ib_.if_scope(tail_len > 0):
                _part_cp_cast((_main_cp_fun, _ub_cp_fun_tail),
                              tail_len,
                              out_end - 1,
                              params,
                              block_offset=one_time_len * num_cycle)
        else:
            with params.ib_.for_range(0,
                                      num_cycle,
                                      for_type="serial",
                                      name="k") as k:
                _part_cp((_main_cp_fun, _ub_cp_fun),
                         one_time_len,
                         out_end - 1,
                         params,
                         block_offset=one_time_len * k)
            tail_len = element_len % one_time_len
            with params.ib_.if_scope(tail_len > 0):
                _part_cp((_main_cp_fun, _ub_cp_fun_tail),
                         tail_len,
                         out_end - 1,
                         params,
                         block_offset=one_time_len * num_cycle)

    _multi_core(_core_func, num_segments, params)

    return True


def _multi_in_multi_out_fun_ids(num_segments, input_buf, segment_ids, params):
    """
    num_segments : num_segments
    input_buf : input buf
    segment_ids : segment_ids
    params : parameters
    """

    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        params.refresh_dtype("float16")
    max_ids_len = params.unified_buffer_len_ids // 2
    max_ids_len = (max_ids_len //
                   params.vec_align_len_ids) * params.vec_align_len_ids
    ids_buf_len = _ceil_fill(max_ids_len * params.type_size_ratio,
                             params.vec_align_len)
    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        one_time_len = (params.unified_buffer_len - ids_buf_len) // 3
    else:
        one_time_len = (params.unified_buffer_len - ids_buf_len) // 2
    one_time_len = (one_time_len //
                    params.vec_align_len) * params.vec_align_len
    element_len = params.element_len
    if params.src_dtype == "int8" or params.src_dtype == "uint8":
        _apply_bufs_cast(max_ids_len, one_time_len, one_time_len, params)
        if params.element_len % one_time_len < 32 or one_time_len < 32:
            params.device_core_num = 1

    else:
        _apply_bufs(max_ids_len, one_time_len, one_time_len, params)
    _do_cp_ids(params.segment_ids_ub, segment_ids, 0, 1, params)

    def _main_cp_fun(data_len,
                     output_offset,
                     input_offset,
                     masks,
                     align_offset=0):
        """
        data_len : data_len
        output_offset : output_offset
        input_offset: input_offset
        masks : mask
        """

        if params.src_dtype == "int8" or params.src_dtype == "uint8":
            params.refresh_dtype(params.src_dtype)
            _do_cp_input_buf_cast(input_buf, data_len, input_offset, params,
                                  align_offset)
            _do_vadd(data_len, output_offset, 0, masks, params)

        else:
            _do_cp_input_buf(input_buf, data_len, input_offset, params,
                             align_offset)
            _do_vadd(data_len, output_offset, 0, masks, params)

    def _ub_cp(_main_cp_fun, data_len, index, params, block_offset):
        """
        _main_cp_fun : _main_cp_fun
        data_len : data_len
        index : index
        params: parameters
        block_offset : block_offset
        """
        num_cycle = params.ids_len // max_ids_len
        with params.ib_.for_range(0, num_cycle, for_type="serial",
                                  name="i") as i:
            _ub_cp_part_ids((max_ids_len, max_ids_len * i),
                            (data_len, block_offset), index, _main_cp_fun,
                            params)

        tail_len = params.ids_len % max_ids_len
        with params.ib_.if_scope(tail_len > 0):
            _ub_cp_part_ids((tail_len, max_ids_len * num_cycle),
                            (data_len, block_offset), index, _main_cp_fun,
                            params)

    def _ub_cp_tail(_main_cp_fun, data_len, index, params, block_offset):
        """
        _main_cp_fun : _main_cp_fun
        data_len : data_len
        index : index
        params: parameters
        block_offset : block_offset
        """
        num_cycle = params.ids_len // max_ids_len
        with params.ib_.for_range(0, num_cycle, for_type="serial",
                                  name="i") as i:
            _ub_cp_part_ids_tail((max_ids_len, max_ids_len * i),
                                 (data_len, block_offset), index, _main_cp_fun,
                                 params)

        tail_len = params.ids_len % max_ids_len
        with params.ib_.if_scope(tail_len > 0):
            _ub_cp_part_ids_tail((tail_len, max_ids_len * num_cycle),
                                 (data_len, block_offset), index, _main_cp_fun,
                                 params)

    def _core_func(out_begin, out_end):
        """
        out_begin: core out index begin
        out_end : core out index end
        """

        if params.src_dtype == "int8" or params.src_dtype == "uint8":
            params.refresh_dtype("float16")
            with params.ib_.for_range(0,
                                      out_end - 1 - out_begin,
                                      for_type="serial",
                                      name="index") as index:
                i = index + out_begin
                num_cycle = element_len // one_time_len
                with params.ib_.for_range(0,
                                          num_cycle,
                                          for_type="serial",
                                          name="j") as j:
                    _part_cp_cast((_main_cp_fun, _ub_cp), one_time_len, i,
                                  params, one_time_len * j)

                tail_len = element_len % one_time_len
                with params.ib_.if_scope(tail_len > 0):
                    _part_cp_cast((_main_cp_fun, _ub_cp), tail_len, i, params,
                                  one_time_len * num_cycle)

            num_cycle = element_len // one_time_len
            with params.ib_.for_range(0,
                                      num_cycle,
                                      for_type="serial",
                                      name="k") as k:
                _part_cp_cast((_main_cp_fun, _ub_cp), one_time_len,
                              out_end - 1, params, one_time_len * k)
            tail_len = element_len % one_time_len
            with params.ib_.if_scope(tail_len > 0):
                _part_cp_cast((_main_cp_fun, _ub_cp_tail), tail_len,
                              out_end - 1, params, one_time_len * num_cycle)
        else:
            with params.ib_.for_range(0,
                                      out_end - 1 - out_begin,
                                      for_type="serial",
                                      name="index") as index:
                i = index + out_begin
                num_cycle = element_len // one_time_len
                with params.ib_.for_range(0,
                                          num_cycle,
                                          for_type="serial",
                                          name="j") as j:
                    _part_cp((_main_cp_fun, _ub_cp), one_time_len, i, params,
                             one_time_len * j)

                tail_len = element_len % one_time_len
                with params.ib_.if_scope(tail_len > 0):
                    _part_cp((_main_cp_fun, _ub_cp), tail_len, i, params,
                             one_time_len * num_cycle)

            num_cycle = element_len // one_time_len
            with params.ib_.for_range(0,
                                      num_cycle,
                                      for_type="serial",
                                      name="k") as k:
                _part_cp((_main_cp_fun, _ub_cp), one_time_len, out_end - 1,
                         params, one_time_len * k)
            tail_len = element_len % one_time_len
            with params.ib_.if_scope(tail_len > 0):
                _part_cp((_main_cp_fun, _ub_cp_tail), tail_len, out_end - 1,
                         params, one_time_len * num_cycle)

    _multi_core(_core_func, num_segments, params)

    return True


def _write_code(wkspace_dict, fname):
    """
    write workspaces to json file

    """
    fname = os.path.realpath(fname)
    if fname.startswith(os.getcwd()):
        if os.path.exists(fname):
            with open(fname, "r") as f_var:
                load_dict = json.load(f_var)
            load_dict.update(wkspace_dict)
            with open(fname, "w") as f_var:
                json.dump(load_dict,
                          f_var,
                          sort_keys=True,
                          indent=4,
                          separators=(',', ':'))


def _new_alloc(tvm_ib, dtype, shape, name, scope, double_buffer=False):
    buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
    if double_buffer is True:
        tvm_ib.scope_attr(buf_var.asnode(), "double_buffer_scope", 1)

    new_buffer = tvm.decl_buffer(shape,
                                 buf_var.dtype,
                                 name=name,
                                 scope=scope,
                                 data=buf_var)

    return new_buffer


def _get_ub_space():
    ub_space_all = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE)
    # open double buffer
    ub_space_all = ub_space_all // 2

    return ub_space_all


def _get_data_type_byte_num(input_dtype):
    return cce.cce_intrin.get_bit_len(input_dtype) // BIT_NUM_ONE_BYTE


def _get_align_size(orig_size, input_dtype):
    data_size = _get_data_type_byte_num(input_dtype)
    one_block_ele_num_input = BLOCK_BYTE // data_size

    if orig_size % one_block_ele_num_input == 0:
        align_size = orig_size
    else:
        align_size = \
            (orig_size // one_block_ele_num_input + 1)*one_block_ele_num_input

    return align_size


def _get_repeat_num(output_shape, output_dtype):
    if output_dtype in ("float32", "int32"):
        each_repeat_num = 64
    else:
        each_repeat_num = 128

    output_size = functools_reduce(lambda i, j: i * j, output_shape)
    dup_repeat_num = ((output_size - 1) // each_repeat_num) + 1

    return int(dup_repeat_num), each_repeat_num


def _check_repeat_valid(target_shape, target_dtype):
    repeat_num, _ = _get_repeat_num(target_shape, target_dtype)

    if repeat_num < MAX_REPEAT_NUM:
        return True

    return False


def _get_copy_block_num(input_shape, input_dtype, output_shape, output_dtype):
    input_size = int(functools_reduce(lambda i, j: i * j, input_shape))
    output_size = int(functools_reduce(lambda i, j: i * j, output_shape))

    if input_dtype == "float16":
        each_block_num = 16
    elif input_dtype in ("int8", "uint8"):
        each_block_num = 32
    else:
        each_block_num = 8

    input_block_num = input_size // each_block_num
    if (input_size - input_block_num * each_block_num) > 0:
        input_block_num = input_block_num + 1

    if output_dtype == "float16":
        each_block_num = 16
    elif output_dtype in ("int8", "uint8"):
        each_block_num = 32
    else:
        each_block_num = 8

    output_block_num = output_size // each_block_num
    if (output_size - output_block_num * each_block_num) > 0:
        output_block_num = output_block_num + 1

    return input_block_num, output_block_num


def _get_target_core_num(ele_num_segments_ids):
    cloud_core_num = 32
    target_core_num = cloud_core_num
    for i in reversed(list(range(1, cloud_core_num + 1))):
        if int(ele_num_segments_ids % i) == 0:
            target_core_num = i
            break

    return target_core_num


# pylint: disable=locally-disabled,unused-argument
def _compute_atomic_once_in_once_out(in_shape, num_segments, dtype, ins,
                                     output_buf, gm_align):
    input_buf = ins[0]
    segment_ids = ins[1]
    shape_segment_ids = segment_ids.shape
    dtype_segment_ids = segment_ids.dtype
    ele_num_segments_ids = shape_segment_ids[0]
    tvm_ib = tvm.ir_builder.create()

    target_core_num = _get_target_core_num(ele_num_segments_ids)
    ele_ids_each_core = ele_num_segments_ids // target_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", target_core_num)

    # allocate reg buffer for id
    reg_for_idx = tvm_ib.allocate(dtype_segment_ids, (1, ),
                                  name="reg_buf_for_idx",
                                  scope=cce.scope_reg)
    # allocate buffer for segment_ids
    segment_ids_ub = _new_alloc(tvm_ib,
                                dtype_segment_ids, (ele_ids_each_core, ),
                                "segment_ids_ub",
                                scope=cce.scope_ubuf)
    # allocate buffer for input
    input_ub = _new_alloc(tvm_ib,
                          dtype, [in_shape[0] // target_core_num, in_shape[1]],
                          "input_ub",
                          scope=cce.scope_ubuf)
    # allocate buffer for output
    output_ub = _new_alloc(tvm_ib,
                           output_buf.dtype,
                           output_buf.shape,
                           "output_ub",
                           scope=cce.scope_ubuf)

    input_segment_ids_block_num, _ = _get_copy_block_num(
        (ele_ids_each_core, ), dtype_segment_ids, (in_shape[0], ), dtype)
    input_ub_block_num, _ = _get_copy_block_num(
        [in_shape[0] // target_core_num, in_shape[1]], dtype, (in_shape[0], ),
        dtype)

    # copy segment_ids to ub
    tvm_ib.emit(
        tvm.call_extern(
            dtype_segment_ids, "copy_gm_to_ubuf",
            segment_ids_ub.access_ptr("w"),
            segment_ids.access_ptr("r",
                                   offset=block_index * ele_ids_each_core), 0,
            1, input_segment_ids_block_num, 0, 0))

    # init the output_ub to zero
    dup_repeat_num, _ = _get_repeat_num(output_ub.shape, output_ub.dtype)
    tvm_ib.emit(
        tvm.call_extern(dtype, "vector_dup", output_ub.access_ptr("w"),
                        tvm.const(0, dtype), dup_repeat_num, 1, 1, 8, 8))

    # copy input once
    tvm_ib.emit(
        tvm.call_extern(
            dtype, "copy_gm_to_ubuf", input_ub.access_ptr("w"),
            input_buf.access_ptr("r",
                                 offset=(block_index * ele_ids_each_core) *
                                        in_shape[1]), 0, 1,
            input_ub_block_num, 0, 0))

    _, output_block_num = _get_copy_block_num((in_shape[1], ), dtype,
                                              output_buf.shape, dtype)
    vadd_repeat_num, _ = _get_repeat_num((in_shape[1], ), dtype)

    with tvm_ib.for_range(0, ele_ids_each_core) as i:
        # get current segment_id according to idx
        tvm_ib.emit(
            tvm.call_extern(
                reg_for_idx.dtype, "reg_mov",
                tvm.call_extern(reg_for_idx.dtype, "reg", reg_for_idx[0]),
                segment_ids_ub.access_ptr("r", offset=i)))

        id_segment_i = reg_for_idx[0]
        # do the add operation in UB
        tvm_ib.emit(
            tvm.call_extern(
                dtype, 'vadd',
                output_ub.access_ptr('w', offset=id_segment_i * in_shape[1]),
                output_ub.access_ptr('r', offset=id_segment_i * in_shape[1]),
                input_ub.access_ptr("r", offset=(i * in_shape[1])),
                vadd_repeat_num, 1, 1, 1, 8, 8, 8))

    # set atomic add
    reg_conf = tvm_ib.allocate("uint64", (2, ),
                               scope=cce.scope_reg,
                               name="reg_conf")
    with tvm_ib.new_scope():
        reg_conf[0] = tvm.call_extern("uint64", "get_ctrl")

    reg_conf[1] = (reg_conf[0] & (~(tvm.const(3, dtype="uint64") << 60))) | (
            tvm.const(1, dtype="uint64") << 60)
    with tvm_ib.new_scope():
        tvm_ib.emit(tvm.call_extern("uint64", "set_ctrl", reg_conf[1]))

    # copy output_ub to gm (atomic)
    with tvm_ib.new_scope():
        tvm_ib.emit(
            tvm.call_extern(dtype, "copy_ubuf_to_gm",
                            output_buf.access_ptr("w"),
                            output_ub.access_ptr("r"), 0, 1, output_block_num,
                            0, 0))

    with tvm_ib.new_scope():
        tvm_ib.emit(tvm.call_extern("uint64", "set_ctrl", reg_conf[0]))

    return tvm_ib.get()


def _get_mte2_args(dtype, row_num_current_core, in_shape, dtype_segment_ids):
    byte_num_id = _get_data_type_byte_num(dtype_segment_ids)
    byte_num_in_out = _get_data_type_byte_num(dtype)

    ub_space_all = _get_ub_space()
    if _get_align_size(in_shape[1], dtype) != in_shape[1]:
        if in_shape[1] == 1:
            ub_space_all = _get_ub_space() - REPEAT_BLOCK_NUM * BLOCK_BYTE
        ub_space_all = _get_ub_space() - \
                       _get_align_size(in_shape[1], dtype) * byte_num_in_out

    byte_num_one_row = byte_num_id + in_shape[1]*byte_num_in_out
    max_row_num_once = ub_space_all // byte_num_one_row

    row_num_each_mte2 = max_row_num_once
    for i in range(max_row_num_once):
        row_num_each_mte2 = max_row_num_once - i
        ub_ids_align_byte = \
            _get_align_size(row_num_each_mte2, dtype_segment_ids) * byte_num_id
        ub_input_align_byte = \
            _get_align_size(row_num_each_mte2*in_shape[1], dtype) * \
            byte_num_in_out
        if (ub_ids_align_byte + ub_input_align_byte) < ub_space_all:
            break

    if row_num_each_mte2 >= row_num_current_core:
        row_num_each_mte2 = row_num_current_core
        row_num_last_mte2 = row_num_each_mte2
        mte2_loop_num = 1
        return int(row_num_each_mte2), int(row_num_last_mte2), \
               int(mte2_loop_num)

    mte2_loop_num = row_num_current_core // row_num_each_mte2
    row_num_last_mte2 = row_num_each_mte2
    if row_num_current_core % row_num_each_mte2 != 0:
        row_num_last_mte2 = \
            row_num_current_core - mte2_loop_num*row_num_each_mte2
        mte2_loop_num = mte2_loop_num + 1

    return int(row_num_each_mte2), int(row_num_last_mte2), int(mte2_loop_num)


def _one_core_in_multi_ir_atomic(tvm_ib, in_shape, dtype, ins, output_buf,
                                 row_num_pre_core, row_num_current_core,
                                 block_index):
    input_buf = ins[0]
    segment_ids = ins[1]
    dtype_segment_ids = segment_ids.dtype

    # copy part of input and part of segment_id each time
    row_num_each_mte2, row_num_last_mte2, mte2_loop_num = \
        _get_mte2_args(dtype, row_num_current_core, in_shape,
                       dtype_segment_ids)
    ids_block_num_each, ids_block_num_last = \
        _get_copy_block_num((row_num_each_mte2, ), dtype_segment_ids,
                            (row_num_last_mte2, ), dtype_segment_ids)
    input_block_num_each, input_block_num_last = \
        _get_copy_block_num((row_num_each_mte2, in_shape[1]), dtype,
                            (row_num_last_mte2, in_shape[1]), dtype)
    output_block_num_each, _ = \
        _get_copy_block_num((1, in_shape[1]), dtype, (1, in_shape[1]), dtype)

    reg_for_idx = tvm_ib.allocate(dtype_segment_ids, (1, ),
                                  name="reg_buf_for_idx", scope=cce.scope_reg)
    reg_conf = tvm_ib.allocate("uint64", (2, ), scope=cce.scope_reg,
                               name="reg_conf")

    # set atomic add
    with tvm_ib.new_scope():
        reg_conf[0] = tvm.call_extern("uint64", "get_ctrl")

    reg_conf[1] = (reg_conf[0] & (~(tvm.const(3, dtype="uint64") << 60))) | \
                  (tvm.const(1, dtype="uint64") << 60)

    with tvm_ib.new_scope():
        tvm_ib.emit(tvm.call_extern("uint64", "set_ctrl", reg_conf[1]))

    if mte2_loop_num > 1:
        with tvm_ib.for_range(0, mte2_loop_num - 1) as mte2_loop_index:
            # allocate ub buffer
            segment_ids_ub = _new_alloc(tvm_ib,
                                        dtype_segment_ids,
                                        (row_num_each_mte2,),
                                        "segment_ids_ub",
                                        scope=cce.scope_ubuf,
                                        double_buffer=True)
            input_ub = _new_alloc(tvm_ib,
                                  dtype, (row_num_each_mte2, in_shape[1]),
                                  "input_ub",
                                  scope=cce.scope_ubuf, double_buffer=True)
            # copy data to ub
            id_offset = block_index * row_num_pre_core + \
                        mte2_loop_index * row_num_each_mte2
            tvm_ib.emit(
                tvm.call_extern(
                    dtype_segment_ids, "copy_gm_to_ubuf",
                    segment_ids_ub.access_ptr("w"),
                    segment_ids.access_ptr("r", offset=id_offset), 0,
                    1, ids_block_num_each, 0, 0))
            tvm_ib.emit(
                tvm.call_extern(
                    dtype, "copy_gm_to_ubuf",
                    input_ub.access_ptr("w"),
                    input_buf.access_ptr("r", offset=id_offset * in_shape[1]),
                    0, 1, input_block_num_each, 0, 0))
            # iterate over each id
            with tvm_ib.new_scope():
                cp = tvm.thread_axis((0, 1), "cce")
                tvm_ib.scope_attr(cp, "group_coproc_scope", 6)
                with tvm_ib.for_range(0, row_num_each_mte2) as id_index:
                    # get id value from segment_ids_ub
                    tvm_ib.emit(
                        tvm.call_extern(
                            reg_for_idx.dtype, "reg_mov",
                            tvm.call_extern(reg_for_idx.dtype, "reg",
                                            reg_for_idx[0]),
                            segment_ids_ub.access_ptr("r", offset=id_index),
                        ))
                    id_value = reg_for_idx[0]
                    # copy data to output_gm
                    with tvm_ib.new_scope():
                        tvm_ib.emit(
                            tvm.call_extern(
                                dtype, "copy_ubuf_to_gm",
                                output_buf.access_ptr(
                                    "w", offset=id_value*in_shape[1]),
                                input_ub.access_ptr(
                                    "r", offset=id_index*in_shape[1]),
                                0, 1, output_block_num_each, 0, 0))
    # do the last loop
    # allocate ub buffer
    segment_ids_ub = _new_alloc(tvm_ib,
                                dtype_segment_ids, (row_num_last_mte2, ),
                                "segment_ids_ub",
                                scope=cce.scope_ubuf, double_buffer=True)
    input_ub = _new_alloc(tvm_ib,
                          dtype, (row_num_last_mte2, in_shape[1]),
                          "input_ub",
                          scope=cce.scope_ubuf, double_buffer=True)
    # copy data to ub
    id_offset = block_index * row_num_pre_core + \
                (mte2_loop_num - 1) * row_num_each_mte2
    tvm_ib.emit(
        tvm.call_extern(
            dtype_segment_ids, "copy_gm_to_ubuf",
            segment_ids_ub.access_ptr("w"),
            segment_ids.access_ptr("r", offset=id_offset), 0,
            1, ids_block_num_last, 0, 0))
    tvm_ib.emit(
        tvm.call_extern(
            dtype, "copy_gm_to_ubuf",
            input_ub.access_ptr("w"),
            input_buf.access_ptr("r", offset=id_offset * in_shape[1]),
            0, 1, input_block_num_last, 0, 0))

    # iterate over each id
    with tvm_ib.new_scope():
        cp = tvm.thread_axis((0, 1), "cce")
        tvm_ib.scope_attr(cp, "group_coproc_scope", 6)
        with tvm_ib.for_range(0, row_num_last_mte2) as id_index:
            # get id value from segment_ids_ub
            tvm_ib.emit(
                tvm.call_extern(
                    reg_for_idx.dtype, "reg_mov",
                    tvm.call_extern(reg_for_idx.dtype, "reg", reg_for_idx[0]),
                    segment_ids_ub.access_ptr("r", offset=id_index),
                ))
            id_value = reg_for_idx[0]
            # copy data to output_gm
            with tvm_ib.new_scope():
                tvm_ib.emit(
                    tvm.call_extern(
                        dtype, "copy_ubuf_to_gm",
                        output_buf.access_ptr(
                            "w", offset=id_value * in_shape[1]),
                        input_ub.access_ptr(
                            "r", offset=id_index * in_shape[1]),
                        0, 1, output_block_num_each, 0, 0))

    with tvm_ib.new_scope():
        tvm_ib.emit(tvm.call_extern("uint64", "set_ctrl", reg_conf[0]))


def _one_core_in_multi_ir_atomic_col_one(tvm_ib, in_shape, dtype, ins,
                                         output_buf, row_num_pre_core,
                                         row_num_current_core, block_index):
    input_buf = ins[0]
    segment_ids = ins[1]
    dtype_segment_ids = segment_ids.dtype
    one_repeat_ele = 64
    if dtype == "float16":
        one_repeat_ele = 128

    # copy part of input and part of segment_id each time
    row_num_each_mte2, row_num_last_mte2, mte2_loop_num = \
        _get_mte2_args(dtype, row_num_current_core, in_shape,
                       dtype_segment_ids)
    ids_block_num_each, ids_block_num_last = \
        _get_copy_block_num((row_num_each_mte2, ), dtype_segment_ids,
                            (row_num_last_mte2, ), dtype_segment_ids)
    input_block_num_each, input_block_num_last = \
        _get_copy_block_num((row_num_each_mte2, in_shape[1]), dtype,
                            (row_num_last_mte2, in_shape[1]), dtype)
    output_block_num_each = 1

    reg_for_idx = tvm_ib.allocate(dtype_segment_ids, (1, ),
                                  name="reg_buf_for_idx", scope=cce.scope_reg)
    reg_conf = tvm_ib.allocate("uint64", (2, ), scope=cce.scope_reg,
                               name="reg_conf")

    # allocate output_ub
    output_ub = _new_alloc(tvm_ib,
                           dtype, (one_repeat_ele, ),
                           "output_ub",
                           scope=cce.scope_ubuf)

    # dump output_ub to zero
    tvm_ib.emit(
        tvm.call_extern(dtype, "vector_dup", output_ub.access_ptr("w"),
                        tvm.const(0, dtype), 1, 1, 1, 8, 8))

    # set atomic add
    with tvm_ib.new_scope():
        reg_conf[0] = tvm.call_extern("uint64", "get_ctrl")

    reg_conf[1] = (reg_conf[0] & (~(tvm.const(3, dtype="uint64") << 60))) | \
                  (tvm.const(1, dtype="uint64") << 60)

    with tvm_ib.new_scope():
        tvm_ib.emit(tvm.call_extern("uint64", "set_ctrl", reg_conf[1]))

    if mte2_loop_num > 1:
        with tvm_ib.for_range(0, mte2_loop_num - 1) as mte2_loop_index:
            # allocate ub buffer
            segment_ids_ub = _new_alloc(tvm_ib,
                                        dtype_segment_ids,
                                        (row_num_each_mte2, ),
                                        "segment_ids_ub",
                                        scope=cce.scope_ubuf,
                                        double_buffer=True)
            input_ub = _new_alloc(tvm_ib,
                                  dtype, (row_num_each_mte2, in_shape[1]),
                                  "input_ub",
                                  scope=cce.scope_ubuf, double_buffer=True)
            # copy data to ub
            id_offset = block_index * row_num_pre_core + \
                        mte2_loop_index * row_num_each_mte2
            tvm_ib.emit(
                tvm.call_extern(
                    dtype_segment_ids, "copy_gm_to_ubuf",
                    segment_ids_ub.access_ptr("w"),
                    segment_ids.access_ptr("r", offset=id_offset), 0,
                    1, ids_block_num_each, 0, 0))
            tvm_ib.emit(
                tvm.call_extern(
                    dtype, "copy_gm_to_ubuf",
                    input_ub.access_ptr("w"),
                    input_buf.access_ptr("r", offset=id_offset * in_shape[1]),
                    0, 1, input_block_num_each, 0, 0))
            # iterate over each id
            with tvm_ib.for_range(0, row_num_each_mte2) as id_index:
                # reg mov input[i] to output_ub[0]
                tvm_ib.emit(
                    tvm.call_extern(
                        dtype, "reg_mov",
                        output_ub.access_ptr("rw"),
                        input_ub.access_ptr("r", offset=id_index)
                    ))
                # get id value from segment_ids_ub
                tvm_ib.emit(
                    tvm.call_extern(
                        reg_for_idx.dtype, "reg_mov",
                        tvm.call_extern(reg_for_idx.dtype, "reg",
                                        reg_for_idx[0]),
                        segment_ids_ub.access_ptr("r", offset=id_index),
                    ))
                id_value = reg_for_idx[0]
                with tvm_ib.new_scope():
                    tvm_ib.emit(
                        tvm.call_extern(
                            dtype, "copy_ubuf_to_gm",
                            output_buf.access_ptr("w",
                                                  offset=id_value*in_shape[1]),
                            output_ub.access_ptr("r"), 0, 1,
                            output_block_num_each, 0, 0))

    # do the last loop
    # allocate ub buffer
    segment_ids_ub = _new_alloc(tvm_ib,
                                dtype_segment_ids, (row_num_last_mte2, ),
                                "segment_ids_ub",
                                scope=cce.scope_ubuf, double_buffer=True)
    input_ub = _new_alloc(tvm_ib,
                          dtype, (row_num_last_mte2, in_shape[1]),
                          "input_ub",
                          scope=cce.scope_ubuf, double_buffer=True)
    # copy data to ub
    id_offset = block_index * row_num_pre_core + \
                (mte2_loop_num - 1) * row_num_each_mte2
    tvm_ib.emit(
        tvm.call_extern(
            dtype_segment_ids, "copy_gm_to_ubuf",
            segment_ids_ub.access_ptr("w"),
            segment_ids.access_ptr("r", offset=id_offset), 0,
            1, ids_block_num_last, 0, 0))
    tvm_ib.emit(
        tvm.call_extern(
            dtype, "copy_gm_to_ubuf",
            input_ub.access_ptr("w"),
            input_buf.access_ptr("r", offset=id_offset * in_shape[1]),
            0, 1, input_block_num_last, 0, 0))
    # iterate over each id
    with tvm_ib.for_range(0, row_num_last_mte2) as id_index:
        # reg mov input[i] to output_ub[0]
        tvm_ib.emit(
            tvm.call_extern(
                dtype, "reg_mov",
                output_ub.access_ptr("rw"),
                input_ub.access_ptr("r", offset=id_index)
            ))
        # get id value from segment_ids_ub
        tvm_ib.emit(
            tvm.call_extern(
                reg_for_idx.dtype, "reg_mov",
                tvm.call_extern(reg_for_idx.dtype, "reg", reg_for_idx[0]),
                segment_ids_ub.access_ptr("r", offset=id_index),
            ))
        id_value = reg_for_idx[0]
        with tvm_ib.new_scope():
            tvm_ib.emit(
                tvm.call_extern(
                    dtype, "copy_ubuf_to_gm",
                    output_buf.access_ptr("w", offset=id_value * in_shape[1]),
                    output_ub.access_ptr("r"), 0, 1, output_block_num_each,
                    0, 0))
    with tvm_ib.new_scope():
        tvm_ib.emit(tvm.call_extern("uint64", "set_ctrl", reg_conf[0]))


def _get_target_core_num_atomic(ins):
    segment_ids = ins[1]
    ele_num_segments_ids = int(segment_ids.shape[0])
    if ele_num_segments_ids < MAX_CORE_NUM:
        return ele_num_segments_ids

    return MAX_CORE_NUM


def _compute_for_atomic(in_shape, dtype, ins, output_buf, func_name):
    tvm_ib = tvm.ir_builder.create()
    target_core_num = _get_target_core_num_atomic(ins)
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", target_core_num)

    if target_core_num < MAX_CORE_NUM:
        row_num_front_core = 1
        row_num_last_core = 1
    else:
        row_num_front_core = in_shape[0] // MAX_CORE_NUM
        row_num_last_core = \
            in_shape[0] - (MAX_CORE_NUM - 1) * row_num_front_core

    row_num_pre_core = row_num_front_core
    with tvm_ib.if_scope(block_index.var == (target_core_num - 1)):
        row_num_current_core = row_num_last_core
        func_name(tvm_ib, in_shape, dtype, ins, output_buf, row_num_pre_core,
                  row_num_current_core, block_index)
    with tvm_ib.else_scope():
        row_num_current_core = row_num_front_core
        func_name(tvm_ib, in_shape, dtype, ins, output_buf, row_num_pre_core,
                  row_num_current_core, block_index)

    return tvm_ib.get()


def _compute_optimization_for_fp16(in_shape, num_segments, dtype, ins,
                                   output_buf, gm_align):
    input_buf = ins[0]
    segment_ids = ins[1]
    shape_segment_ids = segment_ids.shape
    dtype_segment_ids = segment_ids.dtype
    ele_num_segments_ids = shape_segment_ids[0]
    tvm_ib = tvm.ir_builder.create()

    # 1)id_i  2)id_j 3)valid_flag, and 1 is invalid
    reg_for_idx = tvm_ib.allocate(dtype_segment_ids, (3, ),
                                  name="reg_buf_for_idx",
                                  scope=cce.scope_reg)
    target_core_num = _get_target_core_num(ele_num_segments_ids)

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", target_core_num)

    # allocate buffer for segment_ids
    segment_ids_ub = _new_alloc(tvm_ib,
                                dtype_segment_ids,
                                shape_segment_ids,
                                "segment_ids_ub",
                                scope=cce.scope_ubuf)
    # allocate buffer for input one row
    input_ub = _new_alloc(tvm_ib,
                          dtype, (1, in_shape[1]),
                          "input_ub",
                          scope=cce.scope_ubuf)
    # allocate buffer for ouput one row
    output_ub = _new_alloc(tvm_ib,
                           dtype, (1, in_shape[1]),
                           "output_ub",
                           scope=cce.scope_ubuf)

    input_segment_ids_block_num, _ = _get_copy_block_num(
        shape_segment_ids, dtype_segment_ids, (in_shape[0], ), dtype)

    # copy segment_ids to ub
    tvm_ib.emit(
        tvm.call_extern(dtype_segment_ids, "copy_gm_to_ubuf",
                        segment_ids_ub.access_ptr("w"),
                        segment_ids.access_ptr("r"), 0, 1,
                        input_segment_ids_block_num, 0, 0))

    dup_repeat_num, _ = _get_repeat_num((in_shape[1], ), dtype)
    vadd_repeat_num, _ = _get_repeat_num((in_shape[1], ), dtype)
    input_block_num, output_block_num = _get_copy_block_num(
        (in_shape[1], ), dtype, (in_shape[1], ), dtype)

    # do the segment sum
    index_num_each_core = shape_segment_ids[0] // target_core_num
    with tvm_ib.for_range(0, index_num_each_core) as i:
        # init valid flag, default is 0
        reg_for_idx[2] = tvm.const(0, dtype_segment_ids)

        # refresh output_ub
        tvm_ib.emit(
            tvm.call_extern(dtype, "vector_dup", output_ub.access_ptr("w"),
                            tvm.const(0, dtype), dup_repeat_num, 1, 1, 8, 8))

        # get current segment_id according to idx
        tvm_ib.emit(
            tvm.call_extern(
                reg_for_idx.dtype, "reg_mov",
                tvm.call_extern(reg_for_idx.dtype, "reg", reg_for_idx[0]),
                segment_ids_ub.access_ptr("r",
                                          offset=(block_index +
                                                  target_core_num * i))))

        id_segment_i = reg_for_idx[0]
        with tvm_ib.for_range(0, shape_segment_ids[0]) as j:
            tvm_ib.emit(
                tvm.call_extern(
                    reg_for_idx.dtype, "reg_mov",
                    tvm.call_extern(reg_for_idx.dtype, "reg", reg_for_idx[1]),
                    segment_ids_ub.access_ptr("r", offset=j)))

            id_segment_j = reg_for_idx[1]
            with tvm_ib.if_scope(id_segment_i == id_segment_j):
                with tvm_ib.if_scope(j < (block_index + target_core_num * i)):
                    # in this case, no need copy output to gm
                    reg_for_idx[2] = tvm.const(1, dtype_segment_ids)

                with tvm_ib.if_scope(reg_for_idx[2] != 1):
                    # copy the j th row to UB
                    tvm_ib.emit(
                        tvm.call_extern(
                            dtype, "copy_gm_to_ubuf", input_ub.access_ptr("w"),
                            input_buf.access_ptr("r", offset=j * in_shape[1]),
                            0, 1, input_block_num, 0, 0))

                    # sum the rows with same ids
                    tvm_ib.emit(
                        tvm.call_extern(dtype, 'vadd',
                                        output_ub.access_ptr('w'),
                                        output_ub.access_ptr('r'),
                                        input_ub.access_ptr("r"),
                                        vadd_repeat_num, 1, 1, 1, 8, 8, 8))

        # check if copy the sum result to gm
        with tvm_ib.if_scope(reg_for_idx[2] != 1):
            tvm_ib.emit(
                tvm.call_extern(
                    dtype, "copy_ubuf_to_gm",
                    output_buf.access_ptr("w",
                                          offset=id_segment_i * in_shape[1]),
                    output_ub.access_ptr("r"), 0, 1, output_block_num, 0, 0))

    return tvm_ib.get()


def _is_all_ids_in(in_shape, num_segments, dtype, segment_ids):
    shape_segment_ids = segment_ids.shape
    dtype_segment_ids = segment_ids.dtype
    ele_num_segments_ids = shape_segment_ids[0]
    byte_num_segments_id = cce.cce_intrin.get_bit_len(dtype_segment_ids) // 8
    byte_num_in_out = cce.cce_intrin.get_bit_len(dtype) // 8

    ub_space_all_ids = ele_num_segments_ids * byte_num_segments_id
    if (ub_space_all_ids % 32) != 0:
        ub_space_all_ids = ((ub_space_all_ids // 32) + 1) * 32

    ub_space_one_row_in = in_shape[1] * byte_num_in_out
    if (ub_space_one_row_in % 32) != 0:
        ub_space_one_row_in = ((ub_space_one_row_in // 32) + 1) * 32

    ub_space_one_row_out = ub_space_one_row_in
    ub_space_all = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE)
    if int(ub_space_all_ids + ub_space_one_row_in +
           ub_space_one_row_out) < int(ub_space_all):
        return True

    return False


def _is_input_too_large(in_shape, dtype):
    ub_space_all = _get_ub_space()
    byte_num_in_out = _get_data_type_byte_num(dtype)
    ele_num_one_row = in_shape[1]

    align_flag = _is_32_byte_align(in_shape, dtype)
    large_flag = False
    if align_flag:
        ub_space_need = BLOCK_BYTE + ele_num_one_row * byte_num_in_out
    else:
        byte_num_one_row_aligned = \
            _get_align_size(ele_num_one_row, dtype) * byte_num_in_out
        ub_space_need = BLOCK_BYTE + 2 * byte_num_one_row_aligned

    if ub_space_need >= ub_space_all:
        large_flag = True

    return large_flag, align_flag


def _check_atomic_once_in_once_out(in_shape, num_segments):
    if (num_segments == 33) and (in_shape[0] == 16384) and (in_shape[1] == 64):
        return True

    return False


def _get_prod(shape_to_prod):
    return int(functools_reduce(lambda i, j: i * j, shape_to_prod))


def _intrin_factor(in_shape, num_segments, dtype, ins, output_buf, gm_align):
    """
    in_shape: input shape
    num_segments: num_segments
    dtype : the data type
    ins: input tensor
    outs: output tensor
    """
    large_flag, align_flag = _is_input_too_large(in_shape, dtype)
    if (not large_flag) and (dtype == "float32"):
        if align_flag:
            if _check_atomic_once_in_once_out(in_shape, num_segments):
                return _compute_atomic_once_in_once_out(in_shape, num_segments,
                                                        dtype, ins, output_buf,
                                                        gm_align)

            return _compute_for_atomic(in_shape, dtype, ins, output_buf,
                                       _one_core_in_multi_ir_atomic)
        else:
            if in_shape[-1] == 1:
                return _compute_for_atomic(in_shape, dtype, ins, output_buf,
                                           _one_core_in_multi_ir_atomic_col_one)

    if (dtype == "float16") and align_flag and \
            _is_all_ids_in(in_shape, num_segments, dtype, ins[1]) and \
            (in_shape[1] % 128 == 0) and _check_repeat_valid((in_shape[1], ),
                                                             dtype):
        return _compute_optimization_for_fp16(in_shape, num_segments, dtype,
                                              ins, output_buf, gm_align)

    input_buf = ins[0]
    segment_ids = ins[1]
    ib_ = tvm.ir_builder.create()
    params = SegmentParams(ib_, dtype, in_shape,
                           (output_buf, segment_ids, gm_align))
    params.refresh_dtype(dtype)

    with ib_.if_scope(params.block.var < params.device_core_num):
        if _all_in_fun(num_segments, input_buf, segment_ids, params):
            pass
        elif _one_in_multi_out_fun(num_segments, input_buf, segment_ids,
                                   params):
            pass
        elif _multi_in_multi_out_fun(num_segments, input_buf, segment_ids,
                                     params):
            pass
        elif _multi_in_multi_out_fun_large(num_segments, input_buf,
                                           segment_ids, params):
            pass
        else:
            _multi_in_multi_out_fun_ids(num_segments, input_buf, segment_ids,
                                        params)

    return ib_.get()


# pylint: disable=locally-disabled,too-many-arguments,invalid-name
@util.check_input_type(dict, dict, dict, int, str)
def unsorted_segment_sum_d(x,
                           segment_ids,
                           y,
                           num_segments,
                           kernel_name="unsorted_segment_sum_d"):
    """Operation and Schedule for unsorted_segment_sum_d.

    Parameters
    ----------
    x : dict
        shape and dtype of input
    segment_ids : dict
        shape and dtype of segment_ids
    y: dict
        shape and dtype of output
    num_segments: int
        should equal the number of distinct segment IDs
    kernel_name: str
        cce kernel name, default value is "unsorted_segment_sum_d"

    Returns
    -------
        None
    """
    shape_x = x.get("shape")
    shape_y = segment_ids.get("shape")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    if num_segments <= 0:
        raise RuntimeError("num_segments must be more than 0.")
    check_list = ("float16", "float32", "int32", "int8", "uint8")
    dtype = x.get("dtype").lower()
    dtype_segment_ids = segment_ids.get("dtype").lower()
    util.check_dtype_rule(dtype, check_list)
    util.check_dtype_rule(dtype_segment_ids, check_list)
    if len(shape_y) == 1:
        if shape_x[0] != shape_y[0]:
            raise RuntimeError(
                "shape_x and shape_y are not the same length at axis 0.")

        if len(shape_x) == 1:
            shape_x = (shape_x[0], 1)

        shape_x = (shape_x[0],
                   int(functools_reduce(lambda i, j: i * j, shape_x[1:])))

        data_a = tvm.placeholder(shape_x, name="data_a", dtype=dtype)
        segment_ids = tvm.placeholder([
            shape_x[0],
        ],
            name="segment_ids",
            dtype='int32')
        output_shape = [num_segments] + list(shape_x[1:])

    else:
        len_shape_x = len(shape_x)
        len_shape_y = len(shape_y)
        if len_shape_x < len_shape_y:
            raise RuntimeError(
                "len of input should be larger or equal than len of"
                "segment_ids"
            )

        if not operator.eq(shape_x[0:len_shape_y], shape_y):
            raise RuntimeError(
                "the front of input shape should be same with the segment_ids"
                " shape")

        if len_shape_x == len_shape_y:
            shape_x = (_get_prod(shape_x), 1)
        else:
            shape_x = (_get_prod(shape_x[0:len_shape_y]),
                       _get_prod(shape_x[len_shape_y:]))

        shape_y = (_get_prod(shape_y), )
        data_a = tvm.placeholder(shape_x, name="data_a", dtype=dtype)
        segment_ids = tvm.placeholder(shape_y,
                                      name="segment_ids",
                                      dtype='int32')
        output_shape = list((num_segments, shape_x[1]))

    res = tvm.extern([output_shape, [1]], [data_a, segment_ids],
                     lambda ins, outs: _intrin_factor(
                         shape_x, num_segments, dtype, ins, outs[0], outs[1]),
                     name="res",
                     dtype=[dtype, dtype])

    dtype_size = cce.cce_intrin.get_bit_len(dtype) // 8
    if dtype in ("int8", "uint8"):
        dtype_size = 2

    sch = tvm.create_schedule(res[0].op)
    build_list = [data_a, segment_ids, res[0], res[1]]
    with build_config:
        tvm.build(sch, build_list, "cce", name=kernel_name)

    large_flag, align_flag = _is_input_too_large(shape_x, dtype)
    if (not large_flag) and (dtype == "float32") and \
            (align_flag or shape_x[-1] == 1):
        parameter_dict = {"parameters": [0, 0, 1]}
        _write_code(parameter_dict, "kernel_meta/" + kernel_name + ".json")
    elif (dtype == "float16") and (shape_x[1] % 128 == 0) and \
            _is_all_ids_in(shape_x, num_segments, dtype, segment_ids) and \
            _check_repeat_valid((shape_x[1], ), dtype):
        parameter_dict = {"parameters": [0, 0, 1]}
        _write_code(parameter_dict, "kernel_meta/" + kernel_name + ".json")
    else:
        size_align = 1 * dtype_size + 32
        total_size = [size_align]
        num_workspace = 1
        workspace_dict = \
            {"workspace": {"num": num_workspace, "size": total_size}}
        _write_code(workspace_dict, "kernel_meta/" + kernel_name + ".json")

