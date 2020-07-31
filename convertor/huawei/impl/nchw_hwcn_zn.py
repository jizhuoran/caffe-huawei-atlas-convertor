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

nchw_hwcn_fz
"""

from __future__ import absolute_import
import te.lang.cce
from te import tvm
from te import platform as tvm_cce
from te.platform import cce_params as param
from te.platform import cce_intrin as intrin
from te.platform import cce_util
from topi import generic
from topi.cce import util

# pylint: disable=locally-disabled,too-many-lines,too-many-statements
# pylint: disable=locally-disabled,too-many-locals,too-many-arguments
# pylint: disable=locally-disabled,too-many-branches,too-few-public-methods
# pylint: disable=locally-disabled,unnecessary-comprehension

# parameter naming allocated UB
OUTPUT_NAME_SUFFIX = [0]
UB_NAME_SUFFIX = [0]
# available ub size
UB_SIZE_B = tvm_cce.cce_conf.get_soc_spec(tvm_cce.cce_conf.UB_SIZE)
# available number of cores
AICORE_NUM = tvm_cce.cce_conf.get_soc_spec(tvm_cce.cce_conf.CORE_NUM)


def _ceil_div(value, block):
    """
    integrate the input value by block

    """
    return (value + block - 1) // block


def _ceil_fill(value, block):
    """
    fill the input value by block

    """
    return _ceil_div(value, block)*block


def _new_alloc(tvm_ib, dtype, shape, name, scope):
    """
    decl new buffer for ir builder make function

    """
    buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name,
                                 scope=scope, data=buf_var)

    return new_buffer


class FormatTransferParams():
    """
    :param object
        desc: base class object
    :return:
        None
    """

    def __init__(self, ib_):
        self.ib_ = ib_
        self.device_core_num = AICORE_NUM
        self.block = tvm.thread_axis("blockIdx.x")
        self.ib_.scope_attr(self.block, "thread_extent", self.device_core_num)


def nchw_hwcn_fz(input_tensor, output_tensor, raw_shape_4d, src_format,
                 dst_format):
    '''
    :param input_tensor:
        tyep: Tensor
        desc: input data
    :param output_tensor:
        tyep: Tensor
        desc: output data
    :param raw_shape_4d:
        type: tuple or list
        desc: the raw shape of tensor
    :param src_format:
        format: NCHW or NCHW
        type: str
        desc: the format of input tensor
    :param dst_format:
        format: FRACTAL_Z
        type: str
        desc: the format of input tensor
    :return:
        type: Tensor
    '''

    if len(raw_shape_4d) != 4:
        raise RuntimeError("The length of raw_shape_4d must be 4!")

    if output_tensor.dtype != input_tensor.dtype:
        raise RuntimeError("The data type of input and output must be same!")

    format_list = ("NCHW", "HWCN")
    if (src_format.upper() not in format_list) or not (
            dst_format.upper() == "FRACTAL_Z" or
            dst_format.upper() == "FRACTAL_ZN" or
            dst_format.upper() == "FRACTAL_ZN_LSTM"):
        raise RuntimeError("The source format must be HWCN/NCHW and the \
        target format must be FRACTAL_Z !")

    if src_format.upper() == "NCHW" and \
            (dst_format.upper() == "FRACTAL_Z"
             or dst_format.upper() == "FRACTAL_ZN"):
        axis_n, axis_c, axis_h, axis_w = raw_shape_4d
        axis_c1 = (axis_c + 16 - 1) // 16
        axis_n0 = (axis_n + 16 - 1) // 16
        OUTPUT_NAME_SUFFIX[0] += 1
        ir_schedule = tvm.extern(
            [(axis_c1, axis_h, axis_w, axis_n0, 16, 16)], [input_tensor],
            lambda ins, outs:
            _c1hwn0nic0_ir(ins[0], raw_shape_4d, outs[0]),
            dtype=[output_tensor.dtype],
            name="output_" + hex(OUTPUT_NAME_SUFFIX[0]))

    elif src_format.upper() == "HWCN" and \
            (dst_format.upper() == "FRACTAL_Z"
             or dst_format.upper() == "FRACTAL_ZN"):
        axis_h, axis_w, axis_c, axis_n = raw_shape_4d
        axis_c1 = (axis_c + 16 - 1) // 16
        axis_n0 = (axis_n + 16 - 1) // 16
        OUTPUT_NAME_SUFFIX[0] += 1
        ir_schedule = tvm.extern(
            [(axis_c1, axis_h, axis_w, axis_n0, 16, 16)], [input_tensor],
            lambda ins, outs:
            _c1hwn0nic0_hwcn_ir(ins[0], raw_shape_4d, outs[0]),
            dtype=[output_tensor.dtype],
            name="output_" + hex(OUTPUT_NAME_SUFFIX[0]))

    elif src_format.upper() == "HWCN" and \
            dst_format.upper() == "FRACTAL_ZN_LSTM":
        axis_h, axis_w, axis_c, axis_n = raw_shape_4d
        if axis_n % 4 != 0:
            raise RuntimeError("axis_n must be 4n !")
        if axis_c <= axis_n // 4:
            raise RuntimeError("axis_c can't be shorter than axis_n//4 !")
        axis_n0 = (axis_n // 4 + 16 - 1) // 16
        axis_c1 = (axis_c - axis_n // 4 + 16 - 1) // 16
        OUTPUT_NAME_SUFFIX[0] += 1
        ir_schedule = tvm.extern(
            [(axis_c1+axis_n0, axis_h, axis_w, axis_n0*4, 16, 16)],
            [input_tensor],
            lambda ins, outs:
            _c1hwn0nic0_hwcn_lstm_ir(ins[0], raw_shape_4d, outs[0]),
            dtype=[output_tensor.dtype],
            name="output_" + hex(OUTPUT_NAME_SUFFIX[0]))

    return ir_schedule


def _allocate_ub(ib_, dtype, size, name):
    '''
    :param ib_:
        desc: instance of ir_builder
    :param dtype:
        type: str
        desc: support data type: float16, float32
    :param size:
        type: int
        desc: the size to allocate
    :param name:
        type: str
        desc: the name of allocated ub
    :return:
        desc: ub buffer
    '''
    name = name + ".local.UB" + hex(UB_NAME_SUFFIX[0])
    buf_var = ib_.allocate(dtype, (size,), name, scope=param.scope_ubuf)
    return tvm.decl_buffer((size,),
                           dtype,
                           name,
                           scope=param.scope_ubuf,
                           data=buf_var)


def _emit_copy_gm_ubuf(ib_, cmd, dtype, size, dst, dst_offset,
                       src, src_offset):
    '''
    :param ib_:
        desc: instance of ir_builder
    :param cmd:
        type: str
        desc: commond type, copy_gm_to_ubuf or copy_ubuf_to_gm
    :param dtype:
        type: str
        desc: support data type: float16
    :param size:
        type: int
        desc: the size of copy data
    :param dst:
        desc: the addr copy to
    :param dst_offset:
        desc: the offset of dst addr
    :param src:
        desc: the addr copy form
    :param src_offset:
        desc: the offset of src addr
    :return:
        None
    '''
    # the size of UB in bit, 1024*256*8
    n_burst = size // (2**16 * 32)
    tail = size % (2**16 * 32)
    burst_ele_num = 256 // intrin.get_bit_len(dtype)

    if tail > 0:
        dst_offset = dst_offset + n_burst * (2**16 * 32)
        src_offset = src_offset + n_burst * (2**16 * 32)
        len_burst = (tail + burst_ele_num - 1) // burst_ele_num
        sid = 0
        if cmd == "copy_gm_to_ubuf":
            sid = cce_util.get_dma_sid("Sid_copy_gm_to_ubuf")
        ib_.emit(
            tvm.call_extern(dtype, cmd, dst.access_ptr("w", offset=dst_offset),
                            src.access_ptr("rw", offset=src_offset), sid, 1,
                            len_burst, 0, 0))


def _set_array_config(ib_, addr_array, addr_array_buf, src_array_stride,
                      dst_array_stride, output_offset, input_offset):
    '''
    :param ib_:
        desc: instance of ir_builder
    :param addr_array:
        desc: the array of addr
    :param addr_array_buf:
        desc: the array of buf addr
    :param src_array_stride:
        type: int
        desc: src stride of scatter_vnchwconv_b16
    :param dst_array_stride:
        type: int
        desc: dst stride of scatter_vnchwconv_b16
    :param output_offset:
        tyep: int
        desc: the offset of output(ub)
    :param input_offset:
        tyep: int
        desc: the offset of input(ub)
    :return:
        None
    '''
    src0_offset = 8 * 0
    src1_offset = 8 * 1
    dst0_offset = 8 * 2
    dst1_offset = 8 * 3
    dtype_len = 2

    with ib_.for_range(0, 8, name="i") as i:
        ib_.emit(
            tvm.call_extern(
                "uint64", "reg_mov",
                tvm.call_extern(addr_array.dtype, "reg",
                                addr_array[src0_offset + i]),
                dtype_len * (input_offset + src_array_stride * i)))

        ib_.emit(
            tvm.call_extern(
                "uint64", "reg_mov",
                tvm.call_extern(addr_array.dtype, "reg",
                                addr_array[src1_offset + i]),
                dtype_len * (input_offset + src_array_stride * (i + 8))))

        ib_.emit(
            tvm.call_extern(
                "uint64", "reg_mov",
                tvm.call_extern(addr_array.dtype, "reg",
                                addr_array[dst0_offset + i]),
                dtype_len * (output_offset + dst_array_stride * i)))

        ib_.emit(
            tvm.call_extern(
                "uint64", "reg_mov",
                tvm.call_extern(addr_array.dtype, "reg",
                                addr_array[dst1_offset + i]),
                dtype_len * (output_offset + dst_array_stride * (i + 8))))

    ib_.emit(
        tvm.call_extern("int32", "set_va_reg_sb", "VA0",
                        addr_array_buf.access_ptr("rw", offset=src0_offset)))

    ib_.emit(
        tvm.call_extern("int32", "set_va_reg_sb", "VA1",
                        addr_array_buf.access_ptr("rw", offset=src1_offset)))

    ib_.emit(
        tvm.call_extern("int32", "set_va_reg_sb", "VA2",
                        addr_array_buf.access_ptr("rw", offset=dst0_offset)))

    ib_.emit(
        tvm.call_extern("int32", "set_va_reg_sb", "VA3",
                        addr_array_buf.access_ptr("rw", offset=dst1_offset)))


def _get_vnchwconv_cube_buf_max_hp(actual_col_size, dtype):
    '''
    :param actual_col_size:
        type: int
        desc: the size of actual colume (h*w)
    :param dtype:
        type: str
        desc: tensor dtype
    :return:
        type: int
        desc: cube_size
    '''
    if dtype.lower() == "float16":
        byte_len = 2
    elif dtype.lower() == "float32":
        byte_len = 4

    actual_cube_buf_size = (
        (actual_col_size - 1) // 16 + 1) * 16 * 16 * byte_len  # Byte
    ub_cut_upper_limit = UB_SIZE_B // 2  # Byte
    ub_half_248k_size = 248 * 1024 // 2
    if ub_cut_upper_limit > ub_half_248k_size:
        ub_cut_upper_limit = ub_half_248k_size

    if actual_cube_buf_size > ub_cut_upper_limit:
        buf_size = (ub_cut_upper_limit // byte_len) // 16 * 16
    else:
        buf_size = (actual_cube_buf_size // byte_len)

    return buf_size


def _padding_zero(ib_, ubuf, ubuf_offset, dup_len):
    """
    :param ib_:
        type: tvm.ir_builer
        desc: the instance of ir builer
    :param ubuf:
        type: Tensor
        desc: input tensor
    :param ubuf_offset:
        type: int
        desc: address offset of ubuf
    :param dup_len:
        type: int
        desc: data length need to save
    :return:
        None
    """
    uint64_all_one = tvm.const(18446744073709551615, dtype="uint64")
    uint64_all_zero = tvm.const(0, dtype="uint64")
    mask = tvm.const(2**16 - 1 - (2**dup_len - 1), dtype="uint64")
    constant_values = tvm.const(0.0, dtype=ubuf.dtype)

    if dup_len > 0:
        ib_.emit(
            tvm.call_extern("uint64", 'set_vector_mask',
                            uint64_all_zero, mask))

        ib_.emit(
            tvm.call_extern(ubuf.dtype, 'vector_dup',
                            ubuf.access_ptr("rw", offset=ubuf_offset),
                            constant_values, 1, 1, 1, 8, 8))

        ib_.emit(
            tvm.call_extern("uint64", 'set_vector_mask', uint64_all_one,
                            uint64_all_one))


def _clean_ubuf(ib_, src, src_offset, dup_len):
    """
    :param ib_:
        type: tvm.ir_builer
        desc: the instance of ir builer
    :param src:
        type: Tensor
        desc: input tensor
    :param src_offset:
        type: int
        desc: address offset of src
    :param dup_len:
        type: int
        desc: data length need to clean
    :return:
        None
    """
    uint64_all_one = tvm.const(2**64 - 1, dtype="uint64")
    uint64_all_zero = tvm.const(0, dtype="uint64")
    dtype_factor = 32 // intrin.get_bit_len(src.dtype)
    dup_value = tvm.const(0.0, dtype=src.dtype)
    batch_cnt = 64  # one repeat can process 64 elements of float32

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
        left_elem = dup_len % (batch_cnt * dtype_factor)

        def _left_elem_process():
            """
            process the left elements
            """
            if src.dtype == "float16":
                if left_elem > batch_cnt:
                    ib_.emit(
                        tvm.call_extern(
                            "uint64", 'set_vector_mask',
                            tvm.const((1 << (left_elem % batch_cnt))-1,
                                      dtype="uint64"),
                            uint64_all_one))
                else:
                    ib_.emit(
                        tvm.call_extern("uint64", 'set_vector_mask',
                                        uint64_all_zero,
                                        tvm.const((1 << left_elem)-1,
                                                  dtype="uint64")))
            else:
                ib_.emit(
                    tvm.call_extern("uint64", 'set_vector_mask',
                                    uint64_all_zero,
                                    tvm.const((1 << left_elem)-1,
                                              dtype="uint64")))
            ib_.emit(
                tvm.call_extern(
                    src.dtype, "vector_dup",
                    src.access_ptr(
                        "rw",
                        offset=repeat * batch_cnt * dtype_factor + src_offset),
                    dup_value, 1, 1, 1, 8, 8))

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
            if left_elem > 0:
                _left_elem_process()
        else:
            if repeat > 0:
                ib_.emit(
                    tvm.call_extern(src.dtype, "vector_dup",
                                    src.access_ptr("rw", offset=src_offset),
                                    dup_value, repeat, 1, 1, 8, 8))
            if left_elem > 0:
                _left_elem_process()

        ib_.emit(
            tvm.call_extern("uint64", 'set_vector_mask', uint64_all_one,
                            uint64_all_one))


def _mov_data_p2p(args):
    """
    :param args:
        type: list or tuple
        desc: parameters to control data moving point to point
    :return:
        None
    """
    ib_, src_ub, dst_ub, src_offset, dst_offset, num_hw, c0_len, reg = args
    reg_count = 8
    c0_loop = (c0_len + reg_count - 1) // reg_count
    reg_left = c0_len % reg_count

    def _c0_less_eight():
        c0_factor = reg_count // c0_len
        reg_c0_count = c0_factor * c0_len
        hw_loop = num_hw // c0_factor
        hw_loop_left = num_hw % c0_factor

        with ib_.for_range(0, hw_loop, name="hw_loop") as hw_loop_i:
            reg_list = [n for n in range(reg_c0_count)]

            for reg_r_index in reg_list:
                ib_.emit(
                    tvm.call_extern(
                        src_ub.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[reg_r_index]),
                        src_ub.access_ptr(
                            "r",
                            offset=(src_offset +
                                    reg_r_index // c0_len +
                                    (reg_count // c0_len) * hw_loop_i +
                                    (reg_r_index % c0_len) * num_hw))))

            for reg_w_index in reg_list:
                ib_.emit(
                    tvm.call_extern(
                        dst_ub.dtype, "reg_mov",
                        dst_ub.access_ptr(
                            "w",
                            offset=(dst_offset +
                                    (reg_w_index // c0_len) * 16 +
                                    (reg_count // c0_len) * 16 * hw_loop_i +
                                    (reg_w_index % c0_len))),
                        tvm.call_extern(reg.dtype, "reg",
                                        reg[reg_w_index])))

        if hw_loop_left > 0:
            hw_left = num_hw - hw_loop * c0_factor
            with ib_.for_range(0,
                               hw_left, name="left_loop") as left_loop:
                reg_list = [n for n in range(c0_len)]

                for reg_r_index in reg_list:
                    ib_.emit(
                        tvm.call_extern(
                            src_ub.dtype, "reg_mov",
                            tvm.call_extern(reg.dtype, "reg", reg[reg_r_index]),
                            src_ub.access_ptr(
                                "r", offset=(src_offset +
                                             left_loop + hw_loop * c0_factor +
                                             (reg_r_index) * num_hw))))

                for reg_w_index in reg_list:
                    ib_.emit(
                        tvm.call_extern(
                            dst_ub.dtype, "reg_mov",
                            dst_ub.access_ptr(
                                "w",
                                offset=(dst_offset + \
                                        (left_loop + hw_loop * c0_factor) * 16
                                        + (reg_w_index))),
                            tvm.call_extern(reg.dtype, "reg",
                                            reg[reg_w_index])))

    # the load data block is H*W*C0, NC0HW -> NHWC0
    def _c0_larger_eight():
        with ib_.for_range(0, num_hw, name="num_hw") as hw_i:
            # emit 8 statments once in order to improve performance
            with ib_.for_range(0, c0_loop, name="c_0") as c0_loop_i:
                def _p2p_intrinc(c0_count):
                    reg_list = [n for n in range(c0_count)]
                    for reg_r_index in reg_list:
                        ib_.emit(
                            tvm.call_extern(
                                src_ub.dtype, "reg_mov",
                                tvm.call_extern(reg.dtype, "reg",
                                                reg[reg_r_index]),
                                src_ub.access_ptr(
                                    "r",
                                    offset=(src_offset + \
                                            hw_i +
                                            (reg_r_index +
                                             c0_loop_i * reg_count)
                                            * num_hw))))

                    for reg_w_index in reg_list:
                        ib_.emit(
                            tvm.call_extern(
                                dst_ub.dtype, "reg_mov",
                                dst_ub.access_ptr(
                                    "w",
                                    offset=(dst_offset + \
                                            hw_i * 16 + (
                                                reg_w_index + c0_loop_i
                                                * reg_count))),
                                tvm.call_extern(reg.dtype, "reg",
                                                reg[reg_w_index])))

                if reg_left == 0 and c0_len > 0:
                    _p2p_intrinc(reg_count)
                else:
                    with ib_.if_scope(c0_loop_i == c0_loop - 1):
                        _p2p_intrinc(reg_left)
                    with ib_.else_scope():
                        _p2p_intrinc(reg_count)

    if c0_len >= reg_count:
        _c0_larger_eight()
    else:
        _c0_less_eight()


def _mov_data_p2p_lstm(args):
    """
    :param args:
        type: list or tuple
        desc: parameters to control data moving point to point
    :return:
        None
    """
    ib_, src_ub, dst_ub, src_offset, dst_offset, num_hw, \
    c0_len, reg, line_len = args
    reg_count = 8
    c0_loop = (c0_len + reg_count - 1) // reg_count
    reg_left = c0_len % reg_count

    def _c0_less_eight():
        c0_factor = reg_count // c0_len
        reg_c0_count = c0_factor * c0_len
        hw_loop = num_hw // c0_factor
        hw_loop_left = num_hw % c0_factor

        with ib_.for_range(0, hw_loop, name="hw_loop") as hw_loop_i:
            reg_list = [n for n in range(reg_c0_count)]

            for reg_r_index in reg_list:
                ib_.emit(
                    tvm.call_extern(
                        src_ub.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[reg_r_index]),
                        src_ub.access_ptr(
                            "r",
                            offset=(src_offset +
                                    reg_r_index // c0_len +
                                    (reg_count // c0_len) * hw_loop_i +
                                    (reg_r_index % c0_len) * line_len))))

            for reg_w_index in reg_list:
                ib_.emit(
                    tvm.call_extern(
                        dst_ub.dtype, "reg_mov",
                        dst_ub.access_ptr(
                            "w",
                            offset=(dst_offset +
                                    (reg_w_index // c0_len) * 16 +
                                    (reg_count // c0_len) * 16 * hw_loop_i +
                                    (reg_w_index % c0_len))),
                        tvm.call_extern(reg.dtype, "reg",
                                        reg[reg_w_index])))

        if hw_loop_left > 0:
            hw_left = num_hw - hw_loop * c0_factor
            with ib_.for_range(0,
                               hw_left, name="left_loop") as left_loop:
                reg_list = [n for n in range(c0_len)]

                for reg_r_index in reg_list:
                    ib_.emit(
                        tvm.call_extern(
                            src_ub.dtype, "reg_mov",
                            tvm.call_extern(reg.dtype, "reg",
                                            reg[reg_r_index]),
                            src_ub.access_ptr(
                                "r", offset=(src_offset +
                                             left_loop + hw_loop * c0_factor +
                                             (reg_r_index) * line_len))))

                for reg_w_index in reg_list:
                    ib_.emit(
                        tvm.call_extern(
                            dst_ub.dtype, "reg_mov",
                            dst_ub.access_ptr(
                                "w",
                                offset=(dst_offset +
                                        (left_loop + hw_loop * c0_factor)
                                        * 16 + (reg_w_index))),
                            tvm.call_extern(reg.dtype, "reg",
                                            reg[reg_w_index])))

    # the load data block is H*W*C0, NC0HW -> NHWC0
    def _c0_larger_eight():
        with ib_.for_range(0, num_hw, name="num_hw") as hw_i:
            # emit 8 statments once in order to improve performance
            with ib_.for_range(0, c0_loop, name="c_0") as c0_loop_i:
                def _p2p_intrinc(c0_count):
                    reg_list = [n for n in range(c0_count)]
                    for reg_r_index in reg_list:
                        ib_.emit(
                            tvm.call_extern(
                                src_ub.dtype, "reg_mov",
                                tvm.call_extern(reg.dtype, "reg",
                                                reg[reg_r_index]),
                                src_ub.access_ptr(
                                    "r",
                                    offset=(src_offset +
                                            hw_i + (reg_r_index + c0_loop_i
                                                    * reg_count)
                                            * line_len))))

                    for reg_w_index in reg_list:
                        ib_.emit(
                            tvm.call_extern(
                                dst_ub.dtype, "reg_mov",
                                dst_ub.access_ptr(
                                    "w",
                                    offset=(dst_offset +
                                            hw_i * 16 + (reg_w_index
                                                         + c0_loop_i
                                                         * reg_count))),
                                tvm.call_extern(reg.dtype, "reg",
                                                reg[reg_w_index])))

                if reg_left == 0 and c0_len > 0:
                    _p2p_intrinc(reg_count)
                else:
                    with ib_.if_scope(c0_loop_i == c0_loop - 1):
                        _p2p_intrinc(reg_left)
                    with ib_.else_scope():
                        _p2p_intrinc(reg_count)

    if c0_len >= reg_count:
        _c0_larger_eight()
    else:
        _c0_less_eight()


def _c1hwn0nic0_ir(input_tensor, shape_4d, output):
    '''
    :param input_tensor:
         type: Tensor
         desc: input Tensor
    :param shape_4d:
        type: list
        desc: the raw shape fo Tensor (axis_n, axis_c, axis_h, axis_w)
    :param output:
        type: Tensor
        desc: output Tensor
    :return:
        the result statement.
    '''
    axis_n, axis_c, axis_h, axis_w = shape_4d
    axis_c0 = 16
    actual_col_size = axis_h * axis_w
    vnchwconv_cube_buf_max = _get_vnchwconv_cube_buf_max_hp(
        actual_col_size, input_tensor.dtype)
    vnchwconv_cube_col_size = vnchwconv_cube_buf_max // axis_c0

    ib_ = tvm.ir_builder.create()
    params = FormatTransferParams(ib_)

    UB_NAME_SUFFIX[0] += 1
    input_ub = _allocate_ub(ib_, input_tensor.dtype, vnchwconv_cube_buf_max,
                            "input_ub")
    output_ub = _allocate_ub(ib_, output.dtype, vnchwconv_cube_buf_max,
                             "output_ub")
    if input_tensor.dtype.lower() == "float16":
        addr_array = ib_.allocate(
            "uint64", (32,), name="addr_array", scope=param.scope_reg)
        addr_array_buf = tvm.decl_buffer((32,),
                                         "uint64_t",
                                         "addr_array_buf",
                                         scope=param.scope_reg,
                                         data=addr_array)
        src_array_stride = vnchwconv_cube_col_size
        dst_array_stride = axis_c0
        output_offset = vnchwconv_cube_buf_max
        input_offset = 0
        _set_array_config(ib_, addr_array, addr_array_buf, src_array_stride,
                          dst_array_stride, output_offset, input_offset)
    else:
        # emit 8 statments once to improve performance
        reg_array = ib_.allocate(
            output.dtype, (8,), name="addr_array", scope=param.scope_reg)

    # split axis_h*axis_w by core count, to keep the target address align
    # with 32B
    n_per_core_count = axis_n // params.device_core_num
    n_per_core_tail_count = axis_n % params.device_core_num

    c1_range = (axis_c + axis_c0 - 1) // axis_c0
    c1_loop = c1_range
    c1_tail = axis_c % axis_c0  # C1 tail
    if c1_tail != 0:
        c1_loop = c1_range - 1  # remove tail

    repeat_factor = intrin.get_bit_len(output.dtype) // 16
    n0_range = (axis_n + axis_c0 - 1) // axis_c0 * axis_c0

    def _c1hwn0nic0_intrin(ib_, core_loop, c0_loop, c1_begin, c1_end):
        '''
        :param ib_:
            type: tvm.ir_builder
            desc: tvm.ir_builder
        :param core_loop:
            type: int
            desc: loop index of multiple core
        :param c0_loop:
            type: int
            desc: loop number of axis_c0 axis
        :param c1_begin:
            type: int
            desc: start number of C1 axis
        :param c1_end:
            type: int
            desc: end number of C1 axis
        '''
        with ib_.for_range(0, c1_end - c1_begin, name="c1") as c1_index:
            buf_cube_range = (axis_h*axis_w + vnchwconv_cube_col_size - 1) \
                             // vnchwconv_cube_col_size
            with ib_.for_range(
                    0, buf_cube_range, name="buf_cube_index") as cube_i:
                if buf_cube_range > 1:
                    # Standard processing:
                    # Copy gm to ub in 16 copies --> Update transpose
                    # config --> Transposed data --> Copy ub to gm at
                    # one-time

                    with ib_.if_scope(cube_i != buf_cube_range - 1):
                        with ib_.for_range(0, c0_loop, name="col") as col:
                            _emit_copy_gm_ubuf(
                                ib_, "copy_gm_to_ubuf", input_tensor.dtype,
                                vnchwconv_cube_col_size, input_ub,
                                col * vnchwconv_cube_col_size,
                                input_tensor,
                                params.block.var * axis_c * axis_h * axis_w +
                                ((c1_index + c1_begin) * axis_c0 + col) *
                                actual_col_size +
                                cube_i * vnchwconv_cube_col_size +
                                core_loop * params.device_core_num *
                                axis_c * axis_h * axis_w)

                        if input_tensor.dtype.lower() == "float16":
                            repeat = vnchwconv_cube_col_size // 16
                            src_stride = 0 if repeat == 1 else 1
                            dst_stride = 0 if repeat == 1 else 16
                            ib_.emit(
                                tvm.call_extern("int32",
                                                "scatter_vnchwconv_b16",
                                                "VA2", "VA0", repeat,
                                                dst_stride, src_stride))
                        else:
                            args = (ib_, input_ub, output_ub, 0, 0,
                                    vnchwconv_cube_col_size, c0_loop,
                                    reg_array)
                            _mov_data_p2p(args)

                        ib_.emit(
                            tvm.call_extern(
                                output.dtype.lower(),
                                "copy_ubuf_to_gm",
                                output.access_ptr(
                                    "w",
                                    offset=(c1_index+c1_begin) * \
                                    axis_c0*axis_h*axis_w * n0_range +
                                    params.block.var * axis_c0 +
                                    cube_i * axis_c0 *
                                    vnchwconv_cube_col_size*n0_range +
                                    core_loop*params.device_core_num*axis_c0),
                                output_ub.access_ptr("rw", offset=0),
                                0,
                                vnchwconv_cube_col_size,
                                1 * repeat_factor,
                                0,
                                (n0_range-1) * repeat_factor))

                    with ib_.else_scope():
                        # Tail processing:
                        # Copy gm to ub in 16 copies --> Update transpose
                        # config --> Transposed data --> Copy ub to gm at
                        # one-time

                        tail_c0_col_size = \
                            axis_h*axis_w % vnchwconv_cube_col_size
                        if input_tensor.dtype.lower() == "float16":
                            with ib_.for_range(
                                    0, c0_loop, name="col") as col:
                                _emit_copy_gm_ubuf(
                                    ib_, "copy_gm_to_ubuf",
                                    input_tensor.dtype, tail_c0_col_size,
                                    input_ub,
                                    col * vnchwconv_cube_col_size,
                                    input_tensor,
                                    params.block.var*axis_c*axis_h*axis_w +
                                    ((c1_index + c1_begin) *
                                     axis_c0 + col) *
                                    actual_col_size +
                                    cube_i * vnchwconv_cube_col_size +
                                    core_loop * params.device_core_num *
                                    axis_c*axis_h*axis_w)

                            repeat = (tail_c0_col_size + 15) // 16
                            src_stride = 0 if repeat == 1 else 1
                            dst_stride = 0 if repeat == 1 else 16
                            ib_.emit(
                                tvm.call_extern("int32",
                                                "scatter_vnchwconv_b16",
                                                "VA2", "VA0", repeat,
                                                dst_stride, src_stride))
                        else:
                            col_size = (tail_c0_col_size + 15) // 16 * 16
                            with ib_.for_range(
                                    0, c0_loop, name="col") as col:
                                _emit_copy_gm_ubuf(
                                    ib_, "copy_gm_to_ubuf",
                                    input_tensor.dtype, tail_c0_col_size,
                                    input_ub, col * col_size,
                                    input_tensor,
                                    params.block.var*axis_c*axis_h*axis_w +
                                    ((c1_index + c1_begin) * axis_c0 + col) *
                                    actual_col_size +
                                    cube_i * vnchwconv_cube_col_size +
                                    core_loop * params.device_core_num *
                                    axis_c*axis_h*axis_w)

                            args = (ib_, input_ub, output_ub, 0, 0,
                                    col_size, c0_loop, reg_array)
                            _mov_data_p2p(args)

                        ib_.emit(
                            tvm.call_extern(
                                output.dtype.lower(),
                                "copy_ubuf_to_gm",
                                output.access_ptr(
                                    "w",
                                    offset=(c1_index + c1_begin) * axis_c0 * \
                                    axis_h * axis_w * n0_range +
                                    params.block.var * axis_c0 +
                                    cube_i * axis_c0 *
                                    vnchwconv_cube_col_size * n0_range +
                                    core_loop * params.device_core_num *
                                    axis_c0),
                                output_ub.access_ptr("rw", offset=0),
                                0,
                                tail_c0_col_size,
                                1 * repeat_factor,
                                0,
                                (n0_range-1) * repeat_factor))
                else:
                    # Standard processing:
                    # Copy gm to ub in 16 copies --> Update transpose
                    # config -->
                    # Transposed data --> Copy ub to gm at one-time

                    if input_tensor.dtype.lower() == "float16":
                        with ib_.for_range(0, c0_loop, name="col") as col:
                            _emit_copy_gm_ubuf(
                                ib_, "copy_gm_to_ubuf", input_tensor.dtype,
                                axis_h*axis_w, input_ub,
                                col * vnchwconv_cube_col_size,
                                input_tensor,
                                params.block.var * axis_c * axis_h * axis_w +
                                ((c1_index + c1_begin) * axis_c0 + col) *
                                actual_col_size +
                                core_loop * params.device_core_num *
                                axis_c * axis_h * axis_w)

                        repeat = (axis_h*axis_w + 15) // 16
                        src_stride = 0 if repeat == 1 else 1
                        dst_stride = 0 if repeat == 1 else 16
                        ib_.emit(
                            tvm.call_extern("int32",
                                            "scatter_vnchwconv_b16",
                                            "VA2",
                                            "VA0",
                                            repeat,
                                            dst_stride,
                                            src_stride))
                    else:
                        col_size = (axis_h*axis_w + 15) // 16 * 16
                        with ib_.for_range(0, c0_loop, name="col") as col:
                            _emit_copy_gm_ubuf(
                                ib_, "copy_gm_to_ubuf", input_tensor.dtype,
                                axis_h*axis_w, input_ub,
                                col * col_size,
                                input_tensor,
                                params.block.var * axis_c * axis_h * axis_w +
                                ((c1_index + c1_begin) * axis_c0 + col) *
                                actual_col_size +
                                core_loop * params.device_core_num *
                                axis_c * axis_h * axis_w)

                        args = (ib_, input_ub, output_ub, 0, 0,
                                col_size, c0_loop, reg_array)
                        _mov_data_p2p(args)

                    ib_.emit(
                        tvm.call_extern(
                            output.dtype.lower(),
                            "copy_ubuf_to_gm",
                            output.access_ptr(
                                "w",
                                offset=(c1_index + c1_begin) * axis_c0 * \
                                axis_h * axis_w * n0_range +
                                params.block.var * axis_c0 +
                                core_loop * params.device_core_num * axis_c0),
                            output_ub.access_ptr("rw", offset=0),
                            0,
                            axis_h*axis_w,
                            1 * repeat_factor,
                            0,
                            (n0_range-1) * repeat_factor))

    if n_per_core_count >= 0 and c1_loop > 0:
        # clean ubuf to avoid dirty data
        if input_tensor.dtype.lower() == "float16":
            _clean_ubuf(ib_, input_ub, 0, vnchwconv_cube_buf_max)
        else:
            _clean_ubuf(ib_, output_ub, 0, vnchwconv_cube_buf_max)
        with ib_.for_range(0, n_per_core_count, name="n_per_core") as loop_i:
            _c1hwn0nic0_intrin(params.ib_, loop_i, axis_c0, 0, c1_loop)
        if n_per_core_tail_count > 0:
            with ib_.if_scope(params.block.var < n_per_core_tail_count):
                _c1hwn0nic0_intrin(
                    params.ib_, n_per_core_count, axis_c0, 0, c1_loop)
    elif c1_loop > 0:
        # clean ubuf to avoid dirty data
        if input_tensor.dtype.lower() == "float16":
            _clean_ubuf(ib_, input_ub, 0, vnchwconv_cube_buf_max)
        else:
            _clean_ubuf(ib_, output_ub, 0, vnchwconv_cube_buf_max)
        with ib_.if_scope(params.block.var < n_per_core_tail_count):
            _c1hwn0nic0_intrin(
                params.ib_, 0, axis_c0, 0, c1_loop)

    if c1_tail != 0:  # c1 tail processing
        # clean ubuf to avoid dirty data
        if input_tensor.dtype.lower() == "float16":
            _clean_ubuf(ib_, input_ub, 0, vnchwconv_cube_buf_max)
        else:
            _clean_ubuf(ib_, output_ub, 0, vnchwconv_cube_buf_max)

        if n_per_core_count > 0:
            with ib_.for_range(0, n_per_core_count,
                               name="n_per_core") as loop_i:
                _c1hwn0nic0_intrin(params.ib_, loop_i, c1_tail, c1_loop,
                                   c1_range)
            if n_per_core_tail_count > 0:
                with ib_.if_scope(params.block.var < n_per_core_tail_count):
                    _c1hwn0nic0_intrin(
                        params.ib_, n_per_core_count,
                        c1_tail, c1_loop, c1_range)
        else:
            with ib_.if_scope(params.block.var < n_per_core_tail_count):
                _c1hwn0nic0_intrin(params.ib_, 0, c1_tail, c1_loop, c1_range)

    return ib_.get()


def _c1hwn0nic0_hwcn_ir(input_tensor, shape_4d, output):
    '''
    :param input_tensor:
         type: Tensor
         desc: input Tensor
    :param shape_4d:
        type: list
        desc: the raw shape fo Tensor (axis_h, axis_w, axis_c, axis_n)
    :param output:
        type: Tensor
        desc: output Tensor
    :return:
        the result statement.
    '''
    axis_h, axis_w, axis_c, axis_n = shape_4d
    axis_c0 = 16
    actual_col_size = axis_n
    vnchwconv_cube_buf_max = _get_vnchwconv_cube_buf_max_hp(
        actual_col_size, input_tensor.dtype)
    vnchwconv_cube_col_size = vnchwconv_cube_buf_max // axis_c0

    ib_ = tvm.ir_builder.create()
    params = FormatTransferParams(ib_)

    UB_NAME_SUFFIX[0] += 1
    input_ub = _allocate_ub(ib_, input_tensor.dtype, vnchwconv_cube_buf_max,
                            "input_ub")
    output_ub = _allocate_ub(ib_, output.dtype, vnchwconv_cube_buf_max,
                             "output_ub")

    # emit 8 statments once to improve performance
    reg_array = ib_.allocate(
        output.dtype, (8,), name="addr_array", scope=param.scope_reg)

    # each core process axis_c0 * axis_n
    core_loop_count = (axis_c // axis_c0) // params.device_core_num
    # left axis_c0 * axis_n
    core_loop_tail = (axis_c // axis_c0) % params.device_core_num
    # left axis_c * axis_n
    axis_c_left = axis_c % axis_c0
    # used to calc block count moving out
    dtype_factor = 32 // intrin.get_bit_len(input_tensor.dtype)

    def _c1hwn0nic0_hwcn_intrin(ib_, hw_val,
                                core_loop_val, tail_val, c0_len):
        '''
        :param ib_:
            type: tvm.ir_builder
            desc: tvm.ir_builder
        :param hw_val:
            type: int
            desc: index of axis h*w
        :param core_loop_val:
            type: int
            desc: index of multiple core loop
        :param tail_val:
            type: int
            desc: tail of axis c which is not enough for one loop
        :param c0_len:
            type: int
            desc: lines of axis c for each process
        '''
        # move 16 * axis_n each time
        mv_out_len = axis_c0 * ((axis_n + axis_c0 - 1) // axis_c0 * axis_c0)
        # check whether axis_n is larger than allocated ubuf or not
        is_n_larger_col_size = axis_n // vnchwconv_cube_col_size
        is_left_zero = axis_n % vnchwconv_cube_col_size

        if is_n_larger_col_size > 1 or \
                (is_n_larger_col_size == 1 and is_left_zero > 0):
            buf_cube_range = \
                ((axis_n + axis_c0 - 1) // axis_c0 * axis_c0) // \
                vnchwconv_cube_col_size
            n_left = \
                ((axis_n+axis_c0-1)//axis_c0*axis_c0) % vnchwconv_cube_col_size

            with ib_.for_range(0, buf_cube_range, name="n_loop") as n_loop:
                with ib_.for_range(0, c0_len, name="c0_line") as c0_line:
                    _emit_copy_gm_ubuf(
                        ib_, "copy_gm_to_ubuf", input_tensor.dtype,
                        vnchwconv_cube_col_size,
                        input_ub,
                        vnchwconv_cube_col_size * c0_line,
                        input_tensor,
                        params.block.var * c0_len * axis_n +
                        tail_val * axis_c0 * axis_n +
                        core_loop_val * params.device_core_num
                        * axis_c0 * axis_n +
                        hw_val * axis_c * axis_n +
                        n_loop * vnchwconv_cube_col_size + c0_line * axis_n
                    )
                args = (ib_, input_ub, output_ub, 0, 0,
                        vnchwconv_cube_col_size, c0_len, reg_array)
                _mov_data_p2p(args)

                ib_.emit(
                    tvm.call_extern(
                        output.dtype.lower(),
                        "copy_ubuf_to_gm",
                        output.access_ptr(
                            "w",
                            offset=params.block.var * axis_h * axis_w * mv_out_len + \
                            tail_val * axis_h * axis_w * mv_out_len +
                            core_loop_val * params.device_core_num *
                            axis_h * axis_w * mv_out_len +
                            hw_idx * mv_out_len +
                            n_loop * axis_c0 * vnchwconv_cube_col_size
                        ),
                        output_ub.access_ptr("rw", offset=0),
                        0,
                        1,
                        (axis_c0*vnchwconv_cube_col_size) // (dtype_factor*8),
                        0,
                        0))

            if n_left > 0:
                with ib_.for_range(0, c0_len, name="c0_line") as c0_line:
                    _emit_copy_gm_ubuf(
                        ib_, "copy_gm_to_ubuf", input_tensor.dtype,
                        n_left,
                        input_ub,
                        n_left * c0_line,
                        input_tensor,
                        params.block.var * c0_len * axis_n +
                        tail_val * axis_c0 * axis_n +
                        core_loop_val * params.device_core_num
                        * axis_c0 * axis_n +
                        hw_val * axis_c * axis_n +
                        buf_cube_range * vnchwconv_cube_col_size +
                        c0_line * axis_n
                    )
                    pad_n_len = (axis_n % vnchwconv_cube_col_size) % axis_c0
                    pad_offset_len = \
                        (axis_n % vnchwconv_cube_col_size // axis_c0) * axis_c0
                    if pad_n_len > 0:
                        _padding_zero(ib_, input_ub,
                                      n_left*c0_line + pad_offset_len,
                                      pad_n_len)
                args = (ib_, input_ub, output_ub, 0, 0,
                        n_left, c0_len, reg_array)
                _mov_data_p2p(args)

                ib_.emit(
                    tvm.call_extern(
                        output.dtype.lower(),
                        "copy_ubuf_to_gm",
                        output.access_ptr(
                            "w",
                            offset=params.block.var * axis_h * axis_w * mv_out_len + \
                            tail_val * axis_h * axis_w * mv_out_len +
                            core_loop_val * params.device_core_num *
                            axis_h * axis_w * mv_out_len +
                            hw_idx * mv_out_len +
                            buf_cube_range * (axis_c0 * vnchwconv_cube_col_size)
                        ),
                        output_ub.access_ptr("rw", offset=0),
                        0,
                        1,
                        (axis_c0*n_left) // (dtype_factor * 8),
                        0,
                        0))

        else:
            # echo core process c0_len*axis_n one time
            _emit_copy_gm_ubuf(
                ib_, "copy_gm_to_ubuf", input_tensor.dtype,
                c0_len*axis_n, input_ub,
                0,
                input_tensor,
                params.block.var*c0_len*axis_n +
                tail_val*axis_c0*axis_n +
                core_loop_val*params.device_core_num*axis_c0*axis_n +
                hw_val*axis_c*axis_n
                )
            args = (ib_, input_ub, output_ub, 0, 0, axis_n, c0_len, reg_array)
            _mov_data_p2p(args)

            ib_.emit(
                tvm.call_extern(
                    output.dtype.lower(),
                    "copy_ubuf_to_gm",
                    output.access_ptr(
                        "w",
                        offset=params.block.var*axis_h*axis_w*mv_out_len + \
                        tail_val*axis_h*axis_w*mv_out_len +
                        core_loop_val * params.device_core_num *
                        axis_h*axis_w*mv_out_len +
                        hw_idx*mv_out_len),
                    output_ub.access_ptr("rw", offset=0),
                    0,
                    1,
                    mv_out_len // (dtype_factor*8),
                    0,
                    0))

    with ib_.for_range(0, axis_h*axis_w, name="axis_hw") as hw_idx:
        if core_loop_count > 0:
            with ib_.if_scope(hw_idx == 0):
                _clean_ubuf(ib_, output_ub, 0, vnchwconv_cube_buf_max)

            with ib_.for_range(
                    0, core_loop_count,
                    name="core_loop_count") as core_loop_idx:
                _c1hwn0nic0_hwcn_intrin(ib_, hw_idx,
                                        core_loop_idx, 0, axis_c0)

        if core_loop_tail > 0:
            with ib_.if_scope(core_loop_count == 0 and hw_idx == 0):
                _clean_ubuf(ib_, output_ub, 0, vnchwconv_cube_buf_max)

            with ib_.if_scope(params.block.var < core_loop_tail):
                _c1hwn0nic0_hwcn_intrin(ib_, hw_idx,
                                        core_loop_count, 0, axis_c0)

        if axis_c_left > 0:
            with ib_.if_scope(params.block.var < 1):
                if core_loop_count > 0 or core_loop_tail > 0:
                    _clean_ubuf(ib_, output_ub, 0, vnchwconv_cube_buf_max)
                else:
                    with ib_.if_scope(hw_idx == 0):
                        _clean_ubuf(ib_, output_ub, 0, vnchwconv_cube_buf_max)

                _c1hwn0nic0_hwcn_intrin(ib_, hw_idx,
                                        core_loop_count,
                                        core_loop_tail, axis_c_left)

    return ib_.get()


def _c1hwn0nic0_hwcn_lstm_ir(input_tensor, shape_4d, output):
    '''
    :param input_tensor:
         type: Tensor
         desc: input Tensor
    :param shape_4d:
        type: list
        desc: the raw shape fo Tensor (axis_h, axis_w, axis_c, axis_n)
    :param output:
        type: Tensor
        desc: output Tensor
    :return:
        the result statement.
    '''
    axis_h, axis_w, axis_c, axis_n = shape_4d
    axis_c0 = 16
    actual_col_size = axis_n // 4
    vnchwconv_cube_buf_max = _get_vnchwconv_cube_buf_max_hp(
        actual_col_size, input_tensor.dtype)
    vnchwconv_cube_col_size = vnchwconv_cube_buf_max // axis_c0

    ib_ = tvm.ir_builder.create()
    params = FormatTransferParams(ib_)

    UB_NAME_SUFFIX[0] += 1
    input_ub = _allocate_ub(ib_, input_tensor.dtype, vnchwconv_cube_buf_max,
                            "input_ub")
    output_ub = _allocate_ub(ib_, output.dtype, vnchwconv_cube_buf_max,
                             "output_ub")
    if input_tensor.dtype.lower() == "float16":
        addr_array = ib_.allocate(
            "uint64", (32,), name="addr_array", scope=param.scope_reg)
        addr_array_buf = tvm.decl_buffer((32,),
                                         "uint64_t",
                                         "addr_array_buf",
                                         scope=param.scope_reg,
                                         data=addr_array)
        src_array_stride = vnchwconv_cube_col_size
        dst_array_stride = axis_c0
        output_offset = vnchwconv_cube_buf_max
        input_offset = 0
        _set_array_config(ib_, addr_array, addr_array_buf, src_array_stride,
                          dst_array_stride, output_offset, input_offset)
    elif tvm_cce.cce_conf.intrinsic_check_support(
            "Intrinsic_vbi", "float16") and \
            input_tensor.dtype.lower() == "float32":
        addr_array = ib_.allocate(
            "uint64", (32,), name="addr_array", scope=param.scope_reg)
        addr_array_buf = tvm.decl_buffer((32,),
                                         "uint64_t",
                                         "addr_array_buf",
                                         scope=param.scope_reg,
                                         data=addr_array)
        factor_value = 2
        src_array_stride = vnchwconv_cube_col_size * factor_value
        dst_array_stride = axis_c0 * factor_value
        output_offset = vnchwconv_cube_buf_max * factor_value
        input_offset = 0
        _set_array_config(ib_, addr_array, addr_array_buf, src_array_stride,
                          dst_array_stride, output_offset, input_offset)
    else:
        # emit 8 statments once to improve performance
        reg_array = ib_.allocate(
            output.dtype, (8,), name="addr_array", scope=param.scope_reg)
    c_part_list = (axis_c-actual_col_size, actual_col_size)
    # used to calc block count moving out
    dtype_factor = 32 // intrin.get_bit_len(input_tensor.dtype)

    def _c1hwn0nic0_hwcn_lstm_intrin(ib_, hw_val,
                                     core_loop_val, tail_val, c0_len,
                                     c_part_idx, n_part_idx):
        '''
        :param ib_:
            type: tvm.ir_builder
            desc: tvm.ir_builder
        :param hw_val:
            type: int
            desc: index of axis h*w
        :param core_loop_val:
            type: int
            desc: index of multiple core loop
        :param tail_val:
            type: int
            desc: tail of axis c which is not enough for one loop
        :param c0_len:
            type: int
            desc: lines of axis c for each process
        '''
        # move 16 * axis_n each time
        mv_out_len = \
            axis_c0 * ((actual_col_size + axis_c0 - 1) // axis_c0
                       * axis_c0) * 4
        # check whether axis_n is larger than allocated ubuf or not
        is_n_larger_col_size = actual_col_size // vnchwconv_cube_col_size
        is_left_zero = actual_col_size % vnchwconv_cube_col_size

        if is_n_larger_col_size > 1 or \
                (is_n_larger_col_size == 1 and is_left_zero > 0):
            buf_cube_range = \
                ((actual_col_size + axis_c0 - 1) // axis_c0 * axis_c0) // \
                vnchwconv_cube_col_size
            n_left = \
                ((actual_col_size + axis_c0 - 1) // axis_c0 * axis_c0) % \
                vnchwconv_cube_col_size

            with ib_.for_range(0, buf_cube_range, name="n_loop") as n_loop:
                with ib_.for_range(0, c0_len, name="c0_line") as c0_line:
                    _emit_copy_gm_ubuf(
                        ib_, "copy_gm_to_ubuf", input_tensor.dtype,
                        vnchwconv_cube_col_size,
                        input_ub,
                        vnchwconv_cube_col_size * c0_line,
                        input_tensor,
                        params.block.var * c0_len * axis_n +
                        tail_val * axis_c0 * axis_n +
                        core_loop_val * params.device_core_num
                        * axis_c0 * axis_n +
                        hw_val * axis_c * axis_n +
                        n_loop * vnchwconv_cube_col_size + c0_line * axis_n +
                        n_part_idx * actual_col_size +
                        c_part_idx * (c_part_list[0] * axis_n)
                    )
                if input_tensor.dtype.lower() == "float16":
                    repeat = vnchwconv_cube_col_size // 16
                    src_stride = 0 if repeat == 1 else 1
                    dst_stride = 0 if repeat == 1 else 16
                    ib_.emit(
                        tvm.call_extern("int32",
                                        "scatter_vnchwconv_b16",
                                        "VA2", "VA0", repeat,
                                        dst_stride, src_stride))
                elif tvm_cce.cce_conf.intrinsic_check_support(
                        "Intrinsic_vbi", "float16") and \
                    input_tensor.dtype.lower() == "float32":
                    repeat = vnchwconv_cube_col_size // 16
                    src_stride = 0 if repeat == 1 else 2
                    dst_stride = 0 if repeat == 1 else 32
                    ib_.emit(
                        tvm.call_extern("int32",
                                        "scatter_vnchwconv_b32",
                                        "VA2", "VA0", repeat,
                                        dst_stride, src_stride))
                else:
                    args = (ib_, input_ub, output_ub, 0, 0,
                            vnchwconv_cube_col_size, c0_len, reg_array)
                    _mov_data_p2p(args)

                ib_.emit(
                    tvm.call_extern(
                        output.dtype.lower(),
                        "copy_ubuf_to_gm",
                        output.access_ptr(
                            "w",
                            offset=params.block.var * axis_h * axis_w
                            * mv_out_len + tail_val * axis_h * axis_w
                            * mv_out_len + core_loop_val
                            * params.device_core_num * axis_h * axis_w
                            * mv_out_len + hw_idx * mv_out_len +
                            n_loop * axis_c0 * vnchwconv_cube_col_size +
                            n_part_idx * (mv_out_len // 4) +
                            c_part_idx
                            * (((c_part_list[0]+15)//axis_c0*axis_c0)
                               * ((actual_col_size+15)//16*16*4)) *
                            axis_h * axis_w
                        ),
                        output_ub.access_ptr("rw", offset=0),
                        0,
                        1,
                        (axis_c0 * vnchwconv_cube_col_size) // (
                            dtype_factor * 8),
                        0,
                        0))

            if n_left > 0:
                with ib_.for_range(0, c0_len, name="c0_line") as c0_line:
                    _emit_copy_gm_ubuf(
                        ib_, "copy_gm_to_ubuf", input_tensor.dtype,
                        n_left,
                        input_ub,
                        n_left * c0_line,
                        input_tensor,
                        params.block.var * c0_len * axis_n +
                        tail_val * axis_c0 * axis_n +
                        core_loop_val * params.device_core_num
                        * axis_c0 * axis_n +
                        hw_val * axis_c * axis_n +
                        buf_cube_range * vnchwconv_cube_col_size +
                        c0_line * axis_n +
                        n_part_idx * actual_col_size +
                        c_part_idx * (c_part_list[0] * axis_n)
                    )
                    col_left = actual_col_size % vnchwconv_cube_col_size
                    pad_n_len = col_left % axis_c0
                    pad_offset_len = (col_left // axis_c0) * axis_c0
                    if pad_n_len > 0:
                        _padding_zero(ib_, input_ub,
                                      n_left * c0_line + pad_offset_len,
                                      pad_n_len)
                if input_tensor.dtype.lower() == "float16":
                    repeat = (n_left + 15) // 16
                    src_stride = 0 if repeat == 1 else 1
                    dst_stride = 0 if repeat == 1 else 16
                    ib_.emit(
                        tvm.call_extern("int32",
                                        "scatter_vnchwconv_b16",
                                        "VA2", "VA0", repeat,
                                        dst_stride, src_stride))
                    _clean_ubuf(ib_, output_ub, n_left * axis_c0,
                                (axis_c0 - n_left) * axis_c0)
                elif tvm_cce.cce_conf.intrinsic_check_support(
                        "Intrinsic_vbi", "float16") and \
                        input_tensor.dtype.lower() == "float32":
                    repeat = (n_left + 15) // 16
                    src_stride = 0 if repeat == 1 else 2
                    dst_stride = 0 if repeat == 1 else 32
                    ib_.emit(
                        tvm.call_extern("int32",
                                        "scatter_vnchwconv_b32",
                                        "VA2", "VA0", repeat,
                                        dst_stride, src_stride))
                    _clean_ubuf(ib_, output_ub, n_left * axis_c0,
                                (axis_c0 - n_left) * axis_c0)
                else:
                    args = (ib_, input_ub, output_ub, 0, 0,
                            n_left, c0_len, reg_array)
                    _mov_data_p2p(args)

                ib_.emit(
                    tvm.call_extern(
                        output.dtype.lower(),
                        "copy_ubuf_to_gm",
                        output.access_ptr(
                            "w",
                            offset=params.block.var * axis_h * axis_w
                            * mv_out_len + tail_val * axis_h * axis_w
                            * mv_out_len + core_loop_val
                            * params.device_core_num *
                            axis_h * axis_w * mv_out_len +
                            hw_idx * mv_out_len +
                            buf_cube_range * (
                                axis_c0 * vnchwconv_cube_col_size) +
                            n_part_idx * (mv_out_len // 4) +
                            c_part_idx
                            * (((c_part_list[0] + 15) // axis_c0 * axis_c0)
                               * ((actual_col_size+15)//16*16*4)) *
                            axis_h * axis_w
                        ),
                        output_ub.access_ptr("rw", offset=0),
                        0,
                        1,
                        (axis_c0 * n_left) // (dtype_factor * 8),
                        0,
                        0))

        else:
            # echo core process c0_len*axis_n one time
            with ib_.for_range(0, c0_len, name="c0_line") as c0_line:
                _emit_copy_gm_ubuf(
                    ib_, "copy_gm_to_ubuf", input_tensor.dtype,
                    actual_col_size,
                    input_ub,
                    vnchwconv_cube_col_size * c0_line,
                    input_tensor,
                    params.block.var * c0_len * axis_n +
                    tail_val * axis_c0 * axis_n +
                    core_loop_val * params.device_core_num * axis_c0 * axis_n +
                    hw_val * axis_c * axis_n +
                    c0_line * axis_n +
                    n_part_idx * actual_col_size +
                    c_part_idx * (c_part_list[0] * axis_n)
                )
            if input_tensor.dtype.lower() == "float16":
                repeat = (actual_col_size + 15) // 16
                src_stride = 0 if repeat == 1 else 1
                dst_stride = 0 if repeat == 1 else 16
                ib_.emit(
                    tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2", "VA0", repeat,
                                    dst_stride, src_stride))
                if actual_col_size % axis_c0:
                    _clean_ubuf(ib_, output_ub, actual_col_size * axis_c0,
                                (axis_c0 - actual_col_size % axis_c0)
                                * axis_c0)
            elif tvm_cce.cce_conf.intrinsic_check_support(
                    "Intrinsic_vbi", "float16") and \
                    input_tensor.dtype.lower() == "float32":
                repeat = (actual_col_size + 15) // 16
                src_stride = 0 if repeat == 1 else 2
                dst_stride = 0 if repeat == 1 else 32
                ib_.emit(
                    tvm.call_extern("int32",
                                    "scatter_vnchwconv_b32",
                                    "VA2", "VA0", repeat,
                                    dst_stride, src_stride))
                if actual_col_size % axis_c0:
                    _clean_ubuf(ib_, output_ub, actual_col_size * axis_c0,
                                (axis_c0 - actual_col_size % axis_c0)
                                * axis_c0)
            else:
                args = (ib_, input_ub, output_ub, 0, 0, axis_n//4,
                        c0_len, reg_array, vnchwconv_cube_col_size)
                _mov_data_p2p_lstm(args)

            ib_.emit(
                tvm.call_extern(
                    output.dtype.lower(),
                    "copy_ubuf_to_gm",
                    output.access_ptr(
                        "w",
                        offset=params.block.var * axis_h * axis_w
                        * mv_out_len +  tail_val * axis_h * axis_w
                        * mv_out_len + core_loop_val
                        * params.device_core_num *
                        axis_h * axis_w * mv_out_len +
                        hw_idx * mv_out_len +
                        n_part_idx * (mv_out_len // 4) +
                        c_part_idx * (
                            ((c_part_list[0] + 15) // axis_c0 * axis_c0)
                            * ((actual_col_size + 15) // 16 * 16 * 4)) *
                        axis_h * axis_w
                    ),
                    output_ub.access_ptr("rw", offset=0),
                    0,
                    1,
                    (mv_out_len // 4) // (dtype_factor * 8),
                    0,
                    0))

    for c_part_idx in range(2):
        # each core process axis_c0 * axis_n
        core_loop_count = \
            (c_part_list[c_part_idx] // axis_c0) // params.device_core_num
        # left axis_c0 * axis_n
        core_loop_tail = \
            (c_part_list[c_part_idx] // axis_c0) % params.device_core_num
        # left axis_c * axis_n
        axis_c_left = c_part_list[c_part_idx] % axis_c0

        with ib_.for_range(0, axis_h * axis_w, name="axis_hw") as hw_idx:
            with ib_.for_range(0, 4, name="n_part") as n_part_idx:
                if core_loop_count > 0:
                    with ib_.if_scope(hw_idx == 0):
                        if input_tensor.dtype.lower() == "float16":
                            _clean_ubuf(ib_, input_ub, 0,
                                        vnchwconv_cube_buf_max)
                        else:
                            _clean_ubuf(ib_, output_ub, 0,
                                        vnchwconv_cube_buf_max)

                    with ib_.for_range(
                            0, core_loop_count,
                            name="core_loop_count") as core_loop_idx:
                        _c1hwn0nic0_hwcn_lstm_intrin(ib_, hw_idx,
                                                     core_loop_idx, 0, axis_c0,
                                                     c_part_idx, n_part_idx)

                if core_loop_tail > 0:
                    with ib_.if_scope(core_loop_count == 0 and hw_idx == 0):
                        if input_tensor.dtype.lower() == "float16":
                            _clean_ubuf(ib_, input_ub, 0,
                                        vnchwconv_cube_buf_max)
                        else:
                            _clean_ubuf(ib_, output_ub, 0,
                                        vnchwconv_cube_buf_max)

                    with ib_.if_scope(params.block.var < core_loop_tail):
                        _c1hwn0nic0_hwcn_lstm_intrin(ib_, hw_idx,
                                                     core_loop_count, 0,
                                                     axis_c0,
                                                     c_part_idx, n_part_idx)

                if axis_c_left > 0:
                    with ib_.if_scope(params.block.var < 1):
                        if core_loop_count > 0 or core_loop_tail > 0:
                            if input_tensor.dtype.lower() == "float16":
                                _clean_ubuf(ib_, input_ub, 0,
                                            vnchwconv_cube_buf_max)
                            else:
                                _clean_ubuf(ib_, output_ub, 0,
                                            vnchwconv_cube_buf_max)
                        else:
                            with ib_.if_scope(hw_idx == 0):
                                if input_tensor.dtype.lower() == "float16":
                                    _clean_ubuf(ib_, input_ub, 0,
                                                vnchwconv_cube_buf_max)
                                else:
                                    _clean_ubuf(ib_, output_ub, 0,
                                                vnchwconv_cube_buf_max)

                        _c1hwn0nic0_hwcn_lstm_intrin(ib_, hw_idx,
                                                     core_loop_count,
                                                     core_loop_tail,
                                                     axis_c_left,
                                                     c_part_idx, n_part_idx)

    return ib_.get()


def _special_ir(dst, data):
    tvm_ib = tvm.ir_builder.create()
    float_size = tvm_cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = param.BLOCK_REDUCE_INT8 // float_size

    n_i, c_i, _, _ = data.shape
    c_0 = 16
    n_true = _ceil_fill(n_i, c_0)
    ub_max = 3968*16

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_max,
                         "data_ub", scope=tvm_cce.scope_ubuf)

    _clean_ubuf(tvm_ib, data_ub, 0, ub_max)

    loop = c_i // c_0

    with tvm_ib.for_range(0, loop, name="n_loop") as n_loop:
        data_offset = n_loop*c_0
        burst_len_data = c_0 // cp_align_len
        src_stride = (c_i - c_0) // cp_align_len
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr("w", offset=0),
                                    data.access_ptr('r', offset=data_offset),
                                    0, n_i, burst_len_data, src_stride, 0))
        dst_offset = n_loop*c_0*n_true
        burst_len = n_i*c_0 // cp_align_len
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w', offset=dst_offset),
                                    data_ub.access_ptr("r", offset=0),
                                    0, 1, burst_len, 0, 0))

    return tvm_ib.get()


def _check_parameters(src, dst, src_format, dst_format, kernel_name):
    """
    check the parameters including src_shape, dst_shape,
    src_format, dst_format, dtype and kernel_name

    """
    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype")
    dtype_dst = dst.get("dtype")

    if src_format.lower() != "nchw" and src_format.lower() != "hwcn":
        raise RuntimeError("src_format must be NCHW or HWCN !")

    if dst_format.lower() != "fractal_zn"\
            and dst_format.lower() != "fractal_z" \
            and dst_format.lower() != "fractal_zn_lstm":
        raise RuntimeError("dst_format must be FRACTAL_Zn !")

    util.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype, check_list)
    if dtype != dtype_dst:
        raise RuntimeError("dtype of src and dst are different !")

    util.check_shape_rule(src_shape, 4, 4)
    util.check_shape_rule(dst_shape, 4, 4)
    util.check_tensor_shape_size(src_shape)
    util.check_tensor_shape_size(dst_shape)

    if dst_shape[2] != 16 or dst_shape[3] != 16:
        raise RuntimeError(
            "ni and c0 must be 16 !")

    if src_format.lower() == "nchw":
        n_i, c_i, h_i, w_i = src_shape

        c_0 = 16
        c_1 = _ceil_div(c_i, c_0)
        dst_one = c_1 * h_i * w_i
        dst_two = _ceil_div(n_i, 16)

        if list(dst_shape) != [dst_one, dst_two, 16, 16]:
            raise RuntimeError("dst_shape is wrong !")
    elif src_format.lower() == "hwcn" and (dst_format.lower() == "fractal_zn"):
        h_i, w_i, c_i, n_i = src_shape

        c_0 = 16
        c_1 = _ceil_div(c_i, c_0)
        src_one = c_1 * h_i * w_i
        n_ni = 16
        n_no = _ceil_div(n_i, n_ni)

        if list(dst_shape) != [src_one, n_no, 16, 16]:
            raise RuntimeError("src_shape is wrong !")
    else:
        pass


def _check_parameters_special(src, dst, src_format, dst_format, kernel_name):
    """
    check the parameters including src_shape, dst_shape,
    src_format, dst_format, dtype and kernel_name

    """
    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype")
    dtype_dst = dst.get("dtype")

    if src_format.lower() != "nchw" and src_format.lower() != "hwcn":
        raise RuntimeError("src_format must be NCHW or HWCN !")

    if dst_format.lower() != "fractal_zn"\
            and dst_format.lower() != "fractal_z" \
            and dst_format.lower() != "fractal_zn_lstm":
        raise RuntimeError("dst_format must be FRACTAL_Zn !")

    util.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype, check_list)
    if dtype != dtype_dst:
        raise RuntimeError("dtype of src and dst are different !")

    util.check_shape_rule(src_shape, 4, 4)
    util.check_shape_rule(dst_shape, 4, 4)
    util.check_tensor_shape_size(src_shape)
    util.check_tensor_shape_size(dst_shape)

    if dst_shape[2] != 16 or dst_shape[3] != 16:
        raise RuntimeError(
            "ni and c0 must be 16 !")

    if src_format.lower() == "nchw":
        n_i, c_i, h_i, w_i = src_shape

        c_0 = 16
        c_1 = _ceil_div(c_i, c_0)
        dst_one = c_1 * h_i * w_i
        dst_two = _ceil_div(n_i, 16)

        if list(dst_shape) != [dst_one, dst_two, 16, 16]:
            raise RuntimeError("dst_shape is wrong !")
    else:
        h_i, w_i, c_i, n_i = src_shape

        c_0 = 16
        c_1 = _ceil_div(c_i, c_0)
        src_one = c_1 * h_i * w_i
        n_ni = 16
        n_no = _ceil_div(n_i, n_ni)

        if list(dst_shape) != [src_one, n_no, 16, 16]:
            raise RuntimeError("src_shape is wrong !")


# pylint: disable=locally-disabled, unused-argument
def nchw_hwcn_fz_compute(src, dst, src_format, dst_format,
                         kernel_name="nchw_hwcn_zn"):
    """
    algorithm: nchw_hwcn_zn
    doing nchw_hwcn_zn for various data format,
    such as from NCHW/HWCN to Zn

    Parameters
    ----------
    src : TVM tensor
              data of input
    dst: dict
              shape and dtype of output, should be same shape and type as input
    src_format: str
              source data format, can be NHWC/HWCN etc.
    dst_format: str
               target data format, can be FRACTAL_Zn etc.
    kernel_name: str
              kernel name, default value is "nchw_hwcn_zn"
    Returns
    -------
    res : TVM tensor
          the compute result
    """
    shape = te.lang.cce.util.shape_to_list(src.shape)
    res = nchw_hwcn_fz(src, dst, shape, src_format,
                       dst_format)

    return res


@util.check_input_type(dict, dict, str, str, str)
def nchw_hwcn_zn(src, dst, src_format, dst_format, kernel_name="nchw_hwcn_zn"):
    """
    algorithm: nchw_hwcn_zn
    doing nchw_hwcn_zn for various data format, such as from NCHW/HWCN to Zn

    Parameters
    ----------
    src : TVM tensor
              data of input
    dst: dict
              shape and dtype of output, should be same shape and type as input
    src_format: str
              source data format, can be NCHW,HWCN etc.
    dst_format: str
               target data format, can be Zn etc.
    kernel_name: str
              kernel name, default value is "nchw_hwcn_zn"

    Returns
    -------
    None
    """
    shape_input = src.get("shape")
    dtype_input = src.get("dtype").lower()
    shape_output = dst.get("shape")
    dtype_output = dst.get("dtype").lower()

    _check_parameters(src, dst, src_format, dst_format, kernel_name)

    util.check_shape_rule(shape_input)
    util.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype_input, check_list)
    util.check_tensor_shape_size(shape_input)

    data_input = tvm.placeholder(shape_input, name="data_input",
                                 dtype=dtype_input)
    data_output = tvm.placeholder(shape_output, name="data_output",
                                  dtype=dtype_output)
    res = nchw_hwcn_fz_compute(data_input, data_output, src_format,
                               dst_format,
                               kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
