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

four_2_five computer
"""

from __future__ import absolute_import
from te import tvm
from te import platform as tbe_platform
from te.platform import cce_params as param
from te.platform import cce_intrin as intrin
from te.platform import cce_util


#pylint: disable=too-many-lines,too-many-statements,too-few-public-methods
#pylint: disable=too-many-locals,too-many-arguments,superfluous-parens
#pylint: disable=too-many-branches

# parameter naming allocated UB
OUTPUT_NAME_SUFFIX = [0]
UB_NAME_SUFFIX = [0]


class FormatTransferParams():
    """
    :param object
        desc: base class object
    :return:
        None
    """

    def __init__(self, ib_):
        self.ib_ = ib_
        self.device_core_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        self.block = tvm.thread_axis("blockIdx.x")
        self.ib_.scope_attr(self.block, "thread_extent", self.device_core_num)

# pylint: disable=too-many-boolean-expressions
def compute_four_2_five(input_tensor, output_tensor, raw_shape_4d, src_format,
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
        format: NHWC or NCHW
        type: str
        desc: the format of input tensor
    :param dst_format:
        format: NC1HWC0
        type: str
        desc: the format of input tensor
    :return:
        type: Tensor
    '''

    core_num = tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.CORE_NUM)
    if len(raw_shape_4d) != 4:
        raise RuntimeError("The length of raw_shape_4d must be 4!")

    if output_tensor.dtype != input_tensor.dtype:
        raise RuntimeError("The data type of input and output must be same!")

    format_list = ("NCHW", "NHWC")
    if not (src_format.upper() in format_list) or not (
            dst_format.upper() == "NC1HWC0"):
        raise RuntimeError("The source format must be NHWC/NCHW and the \
        target format must be NC1HWC0!")

    if (src_format.upper() == "NHWC" and
            raw_shape_4d[0] < core_num) and \
        ((input_tensor.dtype.lower() == "float32" and
          raw_shape_4d[3] > 992) or
         (input_tensor.dtype.lower() == "float16" and
          raw_shape_4d[3] > 1984)):
        axis_n, axis_h, axis_w, axis_c = raw_shape_4d
        axis_c1 = (axis_c + 16 - 1) // 16
        OUTPUT_NAME_SUFFIX[0] += 1
        ir_schedule = tvm.extern(
            [(axis_n, axis_c1, axis_h, axis_w, 16)], [input_tensor],
            lambda ins, outs:
            _four2five_ir_nhwc(ins[0], raw_shape_4d, outs[0]),
            dtype=[output_tensor.dtype],
            name="output_" + hex(OUTPUT_NAME_SUFFIX[0]))

    elif src_format.upper() == "NHWC":
        axis_n, axis_h, axis_w, axis_c = raw_shape_4d
        axis_c1 = (axis_c + 16 - 1) // 16
        OUTPUT_NAME_SUFFIX[0] += 1
        ir_schedule = tvm.extern(
            [(axis_n, axis_c1, axis_h, axis_w, 16)], [input_tensor],
            lambda ins, outs:
            _four2five_ir_nhwc_hp(ins[0], raw_shape_4d, outs[0]),
            dtype=[output_tensor.dtype],
            name="output_" + hex(OUTPUT_NAME_SUFFIX[0]))

    elif src_format.upper() == "NCHW":
        axis_n, axis_c, axis_h, axis_w = raw_shape_4d
        axis_c1 = (axis_c + 16 - 1) // 16
        OUTPUT_NAME_SUFFIX[0] += 1
        ir_schedule = tvm.extern(
            [(axis_n, axis_c1, axis_h, axis_w, 16)], [input_tensor],
            lambda ins, outs:
            _four2five_ir_nchw(ins[0], raw_shape_4d, outs[0]),
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


def _get_vnhwcconv_buf_max_hp(actual_col_size, c_axis, dtype):
    '''
    :param actual_col_size:
        type: int
        desc: the size of actual colume (h*w)
    :param c_axis:
        type: int
        desc: the size of channel (c)
    :param dtype:
        type: str
        desc: tensor dtype
    :return:
        type: int
        desc: cube_size
    '''
    if dtype.lower() == "float16":
        dtype_factor = 2
    elif dtype.lower() == "float32":
        dtype_factor = 4

    # Byte, a split is axis_h*axis_w*axis_c round up by dividing 16, for fp16
    # costs 2 Bytes, so buf size is axis_h*axis_w*axis_c*dtype_factor
    actual_cube_buf_size = (((actual_col_size - 1) // 16 + 1) * 16) * (
        (c_axis + 15) // 16 * 16) * dtype_factor
    ub_cut_upper_limit = (tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.UB_SIZE)) // 4  # Byte, allocate 4 parts
    if ub_cut_upper_limit > (248 * 1024 // 4):
        ub_cut_upper_limit = 248 * 1024 // 4

    if actual_cube_buf_size > ub_cut_upper_limit:
        buf_size = (ub_cut_upper_limit // dtype_factor)
    else:
        buf_size = (actual_cube_buf_size // dtype_factor)

    return buf_size


def _get_vnhwcconv_cube_buf_max(actual_col_size, c_axis, dtype):
    '''
    :param actual_col_size:
        type: int
        desc: the size of actual colume (h*w)
    :param c_axis:
        type: int
        desc: the size of channel (c)
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

    actual_cube_buf_size = ((actual_col_size + 15) // 16 * 16) * (
        (c_axis + 15) // 16 * 16) * byte_len  # Byte
    ub_cut_upper_limit =\
        (tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE))
    if ub_cut_upper_limit > (248 * 1024):
        ub_cut_upper_limit = 248 * 1024

    if actual_cube_buf_size > ub_cut_upper_limit:
        buf_size = (ub_cut_upper_limit // byte_len)
    else:
        buf_size = (actual_cube_buf_size // byte_len)

    return buf_size


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
    ub_cut_upper_limit = (
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)) // 2
    if ub_cut_upper_limit > (248 * 1024 // 2):
        ub_cut_upper_limit = 248 * 1024 // 2

    if actual_cube_buf_size > ub_cut_upper_limit:
        buf_size = (ub_cut_upper_limit // byte_len)
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
    interv_len = (num_hw + 15) // 16 * 16

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
                                    (reg_r_index % c0_len) * interv_len))))

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
                                             (reg_r_index) * interv_len))))

                for reg_w_index in reg_list:
                    ib_.emit(
                        tvm.call_extern(
                            dst_ub.dtype, "reg_mov",
                            dst_ub.access_ptr(
                                "w",
                                offset=
                                (dst_offset +
                                 (left_loop + hw_loop * c0_factor) * 16 +
                                 (reg_w_index))),
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
                                    offset=
                                    (src_offset +
                                     hw_i +
                                     (reg_r_index +
                                      c0_loop_i * reg_count) * interv_len))))

                    for reg_w_index in reg_list:
                        ib_.emit(
                            tvm.call_extern(
                                dst_ub.dtype, "reg_mov",
                                dst_ub.access_ptr(
                                    "w",
                                    offset=
                                    (dst_offset +
                                     hw_i * 16 + (
                                         reg_w_index + c0_loop_i * reg_count))),
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


def _four2five_ir_nhwc(input_tensor, shape_4d, output):
    '''
    :param input_tensor:
         type: Tensor
         desc: input Tensor
    :param shape_4d:
        type: list
        desc: the raw shape fo Tensor (axis_n, axis_h, axis_w, axis_c)
    :param output:
        type: Tensor
        desc: output Tensor
    :return:
        the result statement.
    '''
    axis_n, axis_h, axis_w, axis_c = shape_4d
    axis_c0 = 16  # the length of axis_c0 axis
    vnhwcconv_cube_buf_max = _get_vnhwcconv_cube_buf_max(
        axis_h * axis_w, axis_c, output.dtype)  # calculate needed buf
    ib_ = tvm.ir_builder.create()
    params = FormatTransferParams(ib_)

    # get dtype factor based one block
    dtype_factor = intrin.get_bit_len(output.dtype) // 16
    # allocate buf
    UB_NAME_SUFFIX[0] += 1
    input_ub = _allocate_ub(params.ib_, input_tensor.dtype,
                            vnhwcconv_cube_buf_max, "input_ub")
    # calculate the c count for one core
    c_count_per_core = axis_h * axis_w // params.device_core_num
    # get the left batches
    c_count_tail = axis_h * axis_w % params.device_core_num
    # calculate how many axis_c0 in axis_c
    c1_range = (axis_c + axis_c0 - 1) // axis_c0
    # get the left length of axis_c
    c1_tail = axis_c % axis_c0  # C1 tail

    def _four2five_intrin(ib_):
        '''
        :param ib_:
            type: tvm.ir_builder
            desc: tvm.ir_builder
        '''

        def _inner_intrin(n_index, c_per_core, in_offset, out_offset):
            virtual_col_size = (
                axis_c + 15) // 16 * 16  # the real buf needed for one split
            buf_cube_range = (virtual_col_size + vnhwcconv_cube_buf_max -
                              1) // vnhwcconv_cube_buf_max
            # axis_c > buf size or axis_c == buf size, do not to do
            # ubuf_to_ubuf
            if (buf_cube_range > 1 or
                    vnhwcconv_cube_buf_max //
                    ((axis_c + 15) // 16 * 16) - 1 == 0):
                with ib_.for_range(
                        0, c_per_core, name="buf_cube_index") as c_core_loop:
                    # calculate how many axis_c0 the buf can have
                    c0_count = vnhwcconv_cube_buf_max // axis_c0
                    # calculate the loop count of one axis_c
                    c_loop_count =\
                        (axis_c + c0_count * axis_c0 - 1) //\
                        (c0_count * axis_c0)
                    # get the left axis_c
                    c_left_len = axis_c % (c0_count * axis_c0)

                    with ib_.for_range(
                            0, c_loop_count, name="c_loop_count") as cube_i:
                        # Standard processing:
                        # Copy gm to ub --> To pad the axis_c to n*axis_c0 by
                        # ub to ub -> Copy ub to gm

                        input_offset = n_index * (axis_h * axis_w * axis_c) + \
                                       cube_i * (c0_count * axis_c0) + \
                                       c_core_loop * axis_c + \
                                       params.block.var *\
                                       c_per_core * axis_c +\
                                       in_offset
                        if c_left_len > 0:
                            with ib_.if_scope(cube_i != c_loop_count - 1):
                                _emit_copy_gm_ubuf(ib_, "copy_gm_to_ubuf",
                                                   input_tensor.dtype,
                                                   c0_count * axis_c0,
                                                   input_ub, 0,
                                                   input_tensor, input_offset)

                                # Copy ub to gm by skipping n*axis_c0
                                with ib_.for_range(
                                        0, c0_count, name="c_count") as c0_i:
                                    output_offset = \
                                        n_index * ((c1_range * axis_c0) *
                                                   axis_h * axis_w) + \
                                        c0_i * (axis_c0 * axis_h * axis_w) +\
                                        cube_i * \
                                        (c0_count *
                                         axis_c0 * axis_h * axis_w) + \
                                        c_core_loop * axis_c0 + \
                                        params.block.var *\
                                        (c_per_core * axis_c0) + \
                                        out_offset

                                    ib_.emit(
                                        tvm.call_extern(
                                            output.dtype, "copy_ubuf_to_gm",
                                            output.access_ptr(
                                                "w", offset=output_offset),
                                            input_ub.access_ptr(
                                                "rw", offset=c0_i * axis_c0),
                                            0, 1,
                                            1 * dtype_factor, 0, 0))
                            with ib_.else_scope():
                                _emit_copy_gm_ubuf(ib_, "copy_gm_to_ubuf",
                                                   input_tensor.dtype,
                                                   c_left_len, input_ub, 0,
                                                   input_tensor, input_offset)
                                _padding_zero(
                                    ib_,
                                    input_ub,
                                    (c_left_len // axis_c0) * axis_c0,
                                    c1_tail)

                                # Copy ub to gm by skipping n*axis_c0
                                with ib_.for_range(
                                        0,
                                        c1_range -
                                        (c_loop_count - 1) * c0_count,
                                        name="c_count") as c0_i:
                                    output_offset = \
                                        n_index * ((c1_range * axis_c0) *
                                                   axis_h * axis_w) + \
                                        c0_i * (axis_c0 * axis_h * axis_w) +\
                                        cube_i * \
                                        (c0_count *
                                         axis_c0 * axis_h * axis_w) + \
                                        c_core_loop * axis_c0 + \
                                        params.block.var *\
                                        (c_per_core * axis_c0) + \
                                        out_offset

                                    ib_.emit(
                                        tvm.call_extern(
                                            output.dtype, "copy_ubuf_to_gm",
                                            output.access_ptr(
                                                "w", offset=output_offset),
                                            input_ub.access_ptr(
                                                "rw", offset=c0_i * axis_c0),
                                            0, 1,
                                            1 * dtype_factor, 0, 0))
                        else:
                            _emit_copy_gm_ubuf(ib_, "copy_gm_to_ubuf",
                                               input_tensor.dtype,
                                               c0_count * axis_c0, input_ub, 0,
                                               input_tensor, input_offset)

                            # Copy ub to gm by skipping n*axis_c0
                            with ib_.for_range(
                                    0, c0_count, name="c_count") as c0_i:
                                output_offset = \
                                    n_index * ((c1_range * axis_c0) *
                                               axis_h * axis_w) + \
                                    c0_i * (axis_c0 * axis_h * axis_w) +\
                                    cube_i * \
                                    (c0_count*axis_c0 * axis_h * axis_w) + \
                                    c_core_loop * axis_c0 + \
                                    params.block.var *\
                                    (c_per_core * axis_c0) + \
                                    out_offset

                                ib_.emit(
                                    tvm.call_extern(
                                        output.dtype, "copy_ubuf_to_gm",
                                        output.access_ptr(
                                            "w", offset=output_offset),
                                        input_ub.access_ptr(
                                            "rw", offset=c0_i * axis_c0), 0, 1,
                                        1 * dtype_factor, 0, 0))
            # axis_c+1 < buf size, do ubuf to ubuf to reduce time of ubuf_to_gm
            elif (vnhwcconv_cube_buf_max // ((axis_c + 15) // 16 * 16) - 1 >
                  0):
                # Standard processing:
                # Copy gm to ub --> To pad the axis_c to n*axis_c0 by ub
                # to ub -> Copy ub to gm

                # calculate how many axis_c the buf can hold
                max_c_count = vnhwcconv_cube_buf_max // (
                    (axis_c + 15) // 16 * 16) - 1
                copy_loop_count = (c_per_core + max_c_count - 1) // max_c_count
                copy_c_left = c_per_core % max_c_count

                with ib_.for_range(
                        0, copy_loop_count, name="copy_loop_count") as copy_i:

                    def _inner_copy(loop_count, i_offset, o_offset):
                        with ib_.for_range(
                                0, loop_count,
                                name="c_per_core") as c_core_loop:
                            input_offset = \
                                n_index * (axis_h * axis_w * axis_c) + \
                                c_core_loop * axis_c + \
                                i_offset + \
                                params.block.var * c_per_core * axis_c + \
                                in_offset

                            _emit_copy_gm_ubuf(ib_, "copy_gm_to_ubuf",
                                               input_tensor.dtype,
                                               axis_c, input_ub,
                                               0, input_tensor, input_offset)

                            # padding to n*axis_c0
                            _padding_zero(
                                ib_, input_ub,
                                (axis_c // axis_c0) * axis_c0, c1_tail)
                            if loop_count > 1:
                                _emit_copy_gm_ubuf(
                                    ib_, "copy_ubuf_to_ubuf",
                                    input_tensor.dtype,
                                    c1_range * axis_c0, input_ub,
                                    (c_core_loop + 1) * (c1_range * axis_c0),
                                    input_ub,
                                    0)

                        # Copy ub to gm by skipping n*axis_c0
                        with ib_.for_range(
                                0, c1_range, name="c1_range") as c0_i:
                            output_offset =\
                                n_index * ((c1_range * axis_c0) *
                                           axis_h * axis_w) + \
                                c0_i * (axis_c0 * axis_h * axis_w) + \
                                o_offset + \
                                params.block.var * \
                                (c_per_core * axis_c0) + \
                                out_offset
                            if loop_count > 1:
                                ib_.emit(
                                    tvm.call_extern(
                                        output.dtype, "copy_ubuf_to_gm",
                                        output.access_ptr(
                                            "w",
                                            offset=output_offset),
                                        input_ub.access_ptr(
                                            "rw",
                                            offset=c0_i * axis_c0 +
                                            c1_range * axis_c0),
                                        0,
                                        loop_count,
                                        1 * dtype_factor,
                                        (c1_range - 1) * dtype_factor,
                                        0))
                            else:
                                ib_.emit(
                                    tvm.call_extern(
                                        output.dtype, "copy_ubuf_to_gm",
                                        output.access_ptr(
                                            "w",
                                            offset=output_offset),
                                        input_ub.access_ptr(
                                            "rw",
                                            offset=c0_i * axis_c0),
                                        0,
                                        loop_count,
                                        1 * dtype_factor,
                                        0,
                                        0))

                    if copy_c_left > 0:
                        with ib_.if_scope(copy_i != copy_loop_count - 1):
                            _inner_copy(
                                max_c_count, copy_i * max_c_count * axis_c,
                                copy_i * max_c_count * axis_c0)
                        with ib_.else_scope():
                            _inner_copy(
                                copy_c_left, copy_i * max_c_count * axis_c,
                                copy_i * max_c_count * axis_c0)
                    else:
                        _inner_copy(max_c_count, copy_i * max_c_count * axis_c,
                                    copy_i * max_c_count * axis_c0)

        with ib_.for_range(0, axis_n, name="axis_n") as n_index:
            if c_count_per_core > 0:
                _inner_intrin(n_index, c_count_per_core, 0, 0)
            if c_count_tail > 0:
                with ib_.if_scope(params.block.var < c_count_tail):
                    _inner_intrin(
                        n_index, 1,
                        c_count_per_core * axis_c * params.device_core_num,
                        c_count_per_core * axis_c0 * params.device_core_num)

    _four2five_intrin(params.ib_)

    return ib_.get()


def _four2five_ir_nhwc_hp(input_tensor, shape_4d, output):
    '''
    :param input_tensor:
         type: Tensor
         desc: input Tensor
    :param shape_4d:
        type: list
        desc: the raw shape fo Tensor (axis_n, axis_h, axis_w, axis_c)
    :param output:
        type: Tensor
        desc: output Tensor
    :return:
        the result statement.
    '''
    axis_n, axis_h, axis_w, axis_c = shape_4d
    axis_c0 = 16  # size of axis_c0 axis
    actual_col_size = axis_h * axis_w  # size of axis_h*axis_w for one batch
    # get buf for one split
    vnchwconv_cube_buf_max = _get_vnhwcconv_buf_max_hp(actual_col_size, axis_c,
                                                       input_tensor.dtype)
    vnchwconv_cube_col_size = vnchwconv_cube_buf_max // axis_c0

    ib_ = tvm.ir_builder.create()
    params = FormatTransferParams(ib_)

    # calculate the c count for one core
    c_count_per_core = axis_h * axis_w // params.device_core_num
    # get the left batches
    c_count_tail = axis_h * axis_w % params.device_core_num
    # calculate the c count for one core
    n_count_per_core = axis_n // params.device_core_num
    # get the left batches
    n_count_tail = axis_n % params.device_core_num
    # get vnconv_b16 repeat factor
    repeat_factor = intrin.get_bit_len(output.dtype) // 16

    # allocate reg buf for VAx
    addr_array = ib_.allocate(
        "uint64", (32,), name="addr_array", scope=param.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,),
                                     "uint64_t",
                                     "addr_array_buf",
                                     scope=param.scope_reg,
                                     data=addr_array)
    UB_NAME_SUFFIX[0] += 1
    src0_array_stride = vnchwconv_cube_col_size * repeat_factor
    dst0_array_stride = axis_c0
    src1_array_stride = axis_c0
    input_ub = _allocate_ub(ib_, input_tensor.dtype,
                            3 * vnchwconv_cube_buf_max,
                            "input_ub")
    output_ub = _allocate_ub(ib_, output.dtype, vnchwconv_cube_buf_max,
                             "output_ub")
    src0_input_offset = 0
    dst0_output_offset = vnchwconv_cube_buf_max * repeat_factor
    src1_input_offset = 2 * vnchwconv_cube_buf_max * repeat_factor
    dst1_output_offset = 3 * vnchwconv_cube_buf_max * repeat_factor

    # set all the buf's value to 0.0 which will be used by 2nd conv
    _clean_ubuf(ib_, input_ub, 2*vnchwconv_cube_buf_max,
                vnchwconv_cube_buf_max)

    # get the value of axis C1
    c1_range = (axis_c + axis_c0 - 1) // axis_c0

    def _four2five_intrin(ib_):
        '''
        :param ib_:
            type: tvm.ir_builder
            desc: tvm.ir_builder
        '''

        def _inner_intrin(n_index, c_per_core, in_offset, out_offset):
            # checking if the real need buf is larger than the allocated buf
            virtual_buf =\
                ((c_per_core + 15) // 16) * 16 * ((axis_c + 15) // 16)
            buf_cube_range = (virtual_buf + vnchwconv_cube_col_size -
                              1) // vnchwconv_cube_col_size
            # the data cannot be loaded one time
            if buf_cube_range > 1:
                # get the axis_c count for each line, there are 16 lines
                c_count = vnchwconv_cube_col_size // ((axis_c + 15) // 16 * 16)
                # get the element count for each line, there are 16 lines
                actual_head_col_size = c_count * axis_c
                # get the loop count for one split
                loop_range = (c_per_core * axis_c +
                              (actual_head_col_size * 16 - 1)) // \
                             (actual_head_col_size * 16)
                core_c_left = (c_per_core * axis_c) % \
                              (actual_head_col_size * 16)
                with ib_.for_range(
                        0, loop_range, name="buf_cube_index") as cube_i:
                    # Standard processing:
                    # Copy gm to ub in 16 copies for each split --> Update
                    # transpose config -->
                    # Transposed data first time --> Copy ub to ub at one-time
                    # to padding axis_c to n*axis_c0 -->
                    # Transposed data second time --> Copy ub to gm

                    # the nchwconv uses 16 lines
                    def _inner_copy_to_ubuf(lp_cnt):
                        with ib_.for_range(0, lp_cnt, name="col") as col:
                            _emit_copy_gm_ubuf(
                                ib_,
                                "copy_gm_to_ubuf",
                                input_tensor.dtype,
                                actual_head_col_size,
                                input_ub,
                                col * vnchwconv_cube_col_size, input_tensor,
                                n_index * axis_c * axis_h * axis_w +
                                actual_head_col_size * col +
                                cube_i * actual_head_col_size * 16 +
                                params.block.var * c_per_core * axis_c +
                                in_offset)

                    if core_c_left == 0:
                        _inner_copy_to_ubuf(16)
                    else:
                        with ib_.if_scope(cube_i != (loop_range - 1)):
                            _inner_copy_to_ubuf(16)
                        with ib_.else_scope():
                            _inner_lp_cnt = core_c_left // actual_head_col_size
                            _inner_left_cnt = core_c_left % actual_head_col_size

                            if _inner_lp_cnt > 0:
                                _inner_copy_to_ubuf(_inner_lp_cnt)
                            if _inner_left_cnt > 0:
                                _emit_copy_gm_ubuf(
                                    ib_,
                                    "copy_gm_to_ubuf",
                                    input_tensor.dtype,
                                    _inner_left_cnt,
                                    input_ub,
                                    _inner_lp_cnt * vnchwconv_cube_col_size,
                                    input_tensor,
                                    n_index * axis_c * axis_h * axis_w +
                                    actual_head_col_size * _inner_lp_cnt +
                                    cube_i * actual_head_col_size * 16 +
                                    params.block.var * c_per_core * axis_c +
                                    in_offset)

                    def _inner_first_conv():
                        # set VA address for conv1, the address should be
                        # aligned with 32Byte
                        _set_array_config(ib_, addr_array, addr_array_buf,
                                          src0_array_stride, dst0_array_stride,
                                          dst0_output_offset, src0_input_offset)
                        # do the conv1
                        repeat_conv1 = (actual_head_col_size + 15) // 16
                        src_stride_conv1 = 0 \
                            if repeat_conv1 * repeat_factor == 1 else 1
                        dst_stride_conv1 = 0 \
                            if repeat_conv1 * repeat_factor == 1 else 16
                        ib_.emit(
                            tvm.call_extern("int32",
                                            "scatter_vnchwconv_b16",
                                            "VA2",
                                            "VA0",
                                            repeat_conv1 * repeat_factor,
                                            dst_stride_conv1,
                                            src_stride_conv1))
                    _inner_first_conv()

                    if axis_c <= axis_c0:
                        # to padding the column axis_c to axis_c0
                        # caution! the option maybe cover memory larger than
                        # one vnchwconv_cube_buf_max
                        # so it's better to split the input data by 16 lines
                        repeat_conv2 = \
                            (actual_head_col_size + axis_c - 1) // axis_c
                        def _inner_vnconv_c_less_c0():
                            ib_.emit(
                                tvm.call_extern(
                                    output.dtype.lower(),
                                    "copy_ubuf_to_ubuf",
                                    input_ub.access_ptr(
                                        "w",
                                        offset=2 * vnchwconv_cube_buf_max),
                                    input_ub.access_ptr(
                                        "rw",
                                        offset=vnchwconv_cube_buf_max),
                                    0,
                                    repeat_conv2, axis_c * repeat_factor,
                                    0,
                                    (16 - axis_c) * repeat_factor))

                            # do the conv2
                            # change the dst_array_stride in order to make
                            # the data in output_ub is continuous
                            _set_array_config(ib_, addr_array, addr_array_buf,
                                              src1_array_stride,
                                              repeat_conv2 * 16 * repeat_factor,
                                              dst1_output_offset,
                                              src1_input_offset)
                            src_stride_conv2 = 0 \
                                if repeat_conv2 * repeat_factor == 1 else 16
                            dst_stride_conv2 = 0 \
                                if repeat_conv2 * repeat_factor == 1 else 1
                            ib_.emit(
                                tvm.call_extern("int32",
                                                "scatter_vnchwconv_b16",
                                                "VA2", "VA0",
                                                repeat_conv2 * repeat_factor,
                                                dst_stride_conv2,
                                                src_stride_conv2))

                        _inner_vnconv_c_less_c0()

                        # output the result
                        def output_result(offset):
                            with ib_.if_scope(cube_i != loop_range - 1):
                                _emit_copy_gm_ubuf(
                                    ib_, 'copy_ubuf_to_gm',
                                    output.dtype,
                                    repeat_conv2 * axis_c0 *16,
                                    output,
                                    n_index * (c1_range * axis_c0) *
                                    axis_h * axis_w + cube_i *
                                    (repeat_conv2 * axis_c0 *16) +
                                    params.block.var * (c_per_core *axis_c0) +
                                    out_offset,
                                    output_ub, offset)
                            with ib_.else_scope():
                                _emit_copy_gm_ubuf(
                                    ib_, 'copy_ubuf_to_gm',
                                    output.dtype,
                                    c_per_core *
                                    axis_c0 - repeat_conv2 * axis_c0 *16 *
                                    (loop_range - 1),
                                    output,
                                    n_index * (c1_range * axis_c0) * axis_h *
                                    axis_w + cube_i *
                                    (repeat_conv2 * axis_c0 *16) +
                                    params.block.var * (c_per_core * axis_c0) +
                                    out_offset,
                                    output_ub, offset)

                        output_result(0)

                    else:
                        # to padding the column axis_c to axis_c0
                        # caution! the option maybe cover memory larger than
                        # one vnchwconv_cube_buf_max
                        # so it's better to split the input data by 16 lines
                        repeat_conv2 =\
                            (actual_head_col_size + axis_c - 1) // axis_c
                        ib_.emit(
                            tvm.call_extern(
                                output.dtype.lower(),
                                "copy_ubuf_to_ubuf",
                                input_ub.access_ptr(
                                    "w",
                                    offset=2 * vnchwconv_cube_buf_max),
                                input_ub.access_ptr(
                                    "rw",
                                    offset=vnchwconv_cube_buf_max),
                                0,
                                repeat_conv2,
                                axis_c * repeat_factor,
                                0,
                                (c1_range * axis_c0 - axis_c) * repeat_factor))

                        # do the conv2
                        src_stride_conv2 = 0 if repeat_conv2 == 1 else \
                            c1_range * 16
                        dst_stride_conv2 = 0 if repeat_conv2 == 1 else 1
                        # repeat the vnchwconv by the (axis_c+16-1)//axis_c0
                        with ib_.for_range(0, c1_range, name="o") as c0_i:
                            # change the dst_array_stride in order to make the
                            # data in output_ub is continuous
                            # change the src_input_offset in order to get data
                            # by channel axis_c0
                            if output.dtype.lower() == "float16":
                                _set_array_config(
                                    ib_, addr_array,
                                    addr_array_buf,
                                    src1_array_stride,
                                    repeat_conv2 * 16,
                                    dst1_output_offset,
                                    src1_input_offset + c0_i * 256)
                                ib_.emit(
                                    tvm.call_extern("int32",
                                                    "scatter_vnchwconv_b16",
                                                    "VA2",
                                                    "VA0",
                                                    repeat_conv2,
                                                    dst_stride_conv2,
                                                    src_stride_conv2))
                            elif output.dtype.lower() == "float32":
                                with ib_.for_range(
                                        0, repeat_conv2, name="r") as repeat_i:
                                    _set_array_config(
                                        ib_,
                                        addr_array,
                                        addr_array_buf,
                                        src1_array_stride,
                                        repeat_conv2 * 16 * repeat_factor,
                                        dst1_output_offset +
                                        repeat_i * 16 * repeat_factor,
                                        src1_input_offset +
                                        c0_i * 512 +
                                        repeat_i * c1_range * 256 *
                                        repeat_factor)
                                    ib_.emit(
                                        tvm.call_extern(
                                            "int32",
                                            "scatter_vnchwconv_b16",
                                            "VA2",
                                            "VA0",
                                            2,
                                            1,
                                            16))

                            # output the result
                            # split the output address into c1_range parts
                            def output_result_multi_c0(offset):
                                with ib_.if_scope(cube_i != loop_range - 1):
                                    _emit_copy_gm_ubuf(
                                        ib_, 'copy_ubuf_to_gm',
                                        output.dtype,
                                        repeat_conv2 * axis_c0 *16,
                                        output,
                                        n_index * (c1_range * axis_c0) *
                                        axis_h * axis_w + cube_i *
                                        (repeat_conv2 * axis_c0 *16) +
                                        params.block.var *
                                        (c_per_core*axis_c0) +
                                        c0_i * (axis_c0 * axis_h * axis_w) +
                                        out_offset,
                                        output_ub, offset)
                                with ib_.else_scope():
                                    tail_size =\
                                        c_per_core * (c1_range * axis_c0) - \
                                        repeat_conv2 * c1_range * 16 * \
                                        axis_c0 * (loop_range - 1)
                                    _emit_copy_gm_ubuf(
                                        ib_, 'copy_ubuf_to_gm',
                                        output.dtype,
                                        tail_size // c1_range,
                                        output,
                                        n_index * (c1_range * axis_c0) *
                                        axis_h * axis_w + cube_i * \
                                        (repeat_conv2 * axis_c0 *16) +
                                        params.block.var *
                                        (c_per_core*axis_c0) + \
                                        c0_i * (axis_c0 * axis_h * axis_w) +
                                        out_offset,
                                        output_ub, offset)

                            output_result_multi_c0(0)

            else:
                # Standard processing:
                # Copy gm to ub in 16 copies for each split --> Update
                # transpose config --> Transposed data first time -->
                # Copy ub to ub at one-time to padding axis_c to n*axis_c0 -->
                # Transposed data second time --> Copy ub to gm

                # the nchwconv uses 16 lines
                hw_cnt = (c_per_core + 15) // 16
                loop_cnt = c_per_core // hw_cnt
                loop_left = c_per_core % hw_cnt
                with ib_.for_range(0, loop_cnt, name="col") as col:
                    _emit_copy_gm_ubuf(ib_, "copy_gm_to_ubuf",
                                       input_tensor.dtype,
                                       (c_per_core+15) // 16 * axis_c,
                                       input_ub, col * vnchwconv_cube_col_size,
                                       input_tensor,
                                       n_index * axis_c * axis_h * axis_w +
                                       ((c_per_core+15) // 16 * axis_c) * col +
                                       params.block.var*c_per_core*axis_c +
                                       in_offset)
                _emit_copy_gm_ubuf(ib_, "copy_gm_to_ubuf",
                                   input_tensor.dtype,
                                   loop_left * axis_c,
                                   input_ub,
                                   loop_cnt * vnchwconv_cube_col_size,
                                   input_tensor,
                                   n_index * axis_c * axis_h * axis_w +
                                   ((c_per_core+15) // 16 * axis_c) *
                                   loop_cnt +
                                   params.block.var*c_per_core*axis_c +
                                   in_offset)

                # set VA address for conv1, the address should be aligned with
                # 32Byte
                _set_array_config(ib_, addr_array, addr_array_buf,
                                  src0_array_stride, dst0_array_stride,
                                  dst0_output_offset, src0_input_offset)
                # do the conv1
                repeat_conv1 = ((c_per_core + 15) // 16 * axis_c + 15) // 16
                src_stride_conv1 = 0\
                    if repeat_conv1 * repeat_factor == 1 else 1
                dst_stride_conv1 = 0\
                    if repeat_conv1 * repeat_factor == 1 else 16
                ib_.emit(
                    tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    repeat_conv1 * repeat_factor,
                                    dst_stride_conv1,
                                    src_stride_conv1))

                if axis_c <= axis_c0:
                    # to padding the column axis_c to axis_c0
                    # caution! the option maybe cover memory larger than one
                    # vnchwconv_cube_buf_max
                    # so it's better to split the input data by 16 lines
                    repeat_conv2 =\
                        ((c_per_core + 15) // 16 * axis_c +
                         axis_c - 1) // axis_c
                    ib_.emit(
                        tvm.call_extern(
                            "float16",
                            "copy_ubuf_to_ubuf",
                            input_ub.access_ptr(
                                "w",
                                offset=2 * vnchwconv_cube_buf_max),
                            input_ub.access_ptr(
                                "rw",
                                offset=vnchwconv_cube_buf_max),
                            0,
                            repeat_conv2,
                            axis_c * repeat_factor,
                            0,
                            (16 - axis_c) * repeat_factor))

                    # do the conv2
                    _set_array_config(ib_, addr_array,
                                      addr_array_buf,
                                      src1_array_stride,
                                      repeat_conv2 * 16 * repeat_factor,
                                      dst1_output_offset,
                                      src1_input_offset)
                    src_stride_conv2 = 0\
                        if repeat_conv2 * repeat_factor == 1 else 16
                    dst_stride_conv2 = 0\
                        if repeat_conv2 * repeat_factor == 1 else 1
                    ib_.emit(
                        tvm.call_extern("int32",
                                        "scatter_vnchwconv_b16",
                                        "VA2",
                                        "VA0",
                                        repeat_conv2 * repeat_factor,
                                        dst_stride_conv2,
                                        src_stride_conv2))

                    # output the result
                    _emit_copy_gm_ubuf(
                        ib_, 'copy_ubuf_to_gm',
                        output.dtype,
                        c_per_core * axis_c0,
                        output,
                        n_index * (c1_range * axis_c0) * axis_h * axis_w + \
                        params.block.var * (c_per_core * axis_c0) +
                        out_offset,
                        output_ub, 0)

                else:
                    # to padding the column axis_c to axis_c0
                    # caution! the option maybe cover memory larger than one
                    # vnchwconv_cube_buf_max
                    # so it's better to split the input data by 16 lines
                    repeat_conv2 =\
                        ((c_per_core + 15) // 16 * axis_c +
                         axis_c - 1) // axis_c
                    ib_.emit(
                        tvm.call_extern(
                            "float16", "copy_ubuf_to_ubuf",
                            input_ub.access_ptr(
                                "w",
                                offset=2 * vnchwconv_cube_buf_max),
                            input_ub.access_ptr(
                                "rw",
                                offset=vnchwconv_cube_buf_max), 0,
                            repeat_conv2,
                            axis_c * repeat_factor,
                            0,
                            (c1_range * axis_c0 - axis_c) * repeat_factor))

                    # do the conv2
                    src_stride_conv2 = 0 if repeat_conv2 == 1 else c1_range*16
                    dst_stride_conv2 = 0 if repeat_conv2 == 1 else 1
                    # repeat the vnchwconv by the (axis_c+16-1)//axis_c0
                    with ib_.for_range(0, c1_range, name="o") as c0_i:
                        if output.dtype.lower() == "float16":
                            _set_array_config(
                                ib_, addr_array,
                                addr_array_buf,
                                src1_array_stride,
                                repeat_conv2 * 16,
                                dst1_output_offset,
                                src1_input_offset + c0_i * 256)
                            ib_.emit(
                                tvm.call_extern(
                                    "int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    repeat_conv2,
                                    dst_stride_conv2,
                                    src_stride_conv2))
                        elif output.dtype.lower() == "float32":
                            with ib_.for_range(
                                    0, repeat_conv2, name="r") as repeat_i:
                                _set_array_config(
                                    ib_,
                                    addr_array,
                                    addr_array_buf,
                                    src1_array_stride,
                                    repeat_conv2 * 16 * repeat_factor,
                                    dst1_output_offset +
                                    repeat_i * 16 * repeat_factor,
                                    src1_input_offset + c0_i * 512 +
                                    repeat_i*c1_range * 256 * repeat_factor)
                                ib_.emit(tvm.call_extern(
                                    "int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    2,
                                    1,
                                    16))

                        # output the result
                        _emit_copy_gm_ubuf(
                            ib_, 'copy_ubuf_to_gm',
                            output.dtype,
                            axis_c0 * c_per_core,
                            output,
                            n_index * (c1_range * axis_c0) * axis_h * axis_w +
                            params.block.var * (c_per_core * axis_c0) +
                            out_offset + c0_i * axis_c0 * axis_h * axis_w,
                            output_ub, 0)

        with ib_.for_range(0, axis_n, name="axis_n") as n_index:
            if c_count_per_core > 0:
                _inner_intrin(n_index, c_count_per_core, 0, 0)
            if c_count_tail > 0:
                with ib_.if_scope(params.block.var < 1):
                    _inner_intrin(
                        n_index, c_count_tail,
                        c_count_per_core * axis_c * params.device_core_num,
                        c_count_per_core * axis_c0 * params.device_core_num)

    def _four2five_intrin_axis_n(ib_):
        '''
        :param ib_:
            type: tvm.ir_builder
            desc: tvm.ir_builder
        '''

        def _inner_intrin_axis_n(n_loop_index, in_offset, out_offset):
            # checking if the real need buf is larger than the allocated buf
            virtual_buf =\
                ((axis_h * axis_w + 15) // 16) * 16 * ((axis_c + 15) // 16)
            buf_cube_range = (virtual_buf + vnchwconv_cube_col_size -
                              1) // vnchwconv_cube_col_size

            # checking if axis c is larger than the vnchconv_cube_col_size
            is_c_large_col = axis_c > vnchwconv_cube_col_size
            # the data cannot be loaded one time
            if buf_cube_range > 1 and is_c_large_col is False:
                # get the axis_c count for each line, there are 16 lines
                c_count = vnchwconv_cube_col_size // ((axis_c + 15) // 16 * 16)
                # get the element count for each line, there are 16 lines
                actual_head_col_size = c_count * axis_c
                # get the loop count for one split
                loop_range = (axis_h * axis_w * axis_c +
                              (actual_head_col_size * 16 - 1)) //\
                             (actual_head_col_size * 16)
                hwc_left = (axis_h * axis_w * axis_c) % \
                           (actual_head_col_size * 16)
                with ib_.for_range(
                        0, loop_range, name="buf_cube_index") as cube_i:
                    # Standard processing:
                    # Copy gm to ub in 16 copies for each split --> Update
                    # transpose config -->
                    # Transposed data first time --> Copy ub to ub at one-time
                    # to padding axis_c to n*axis_c0 -->
                    # Transposed data second time --> Copy ub to gm

                    # the nchwconv uses 16 lines
                    def _inner_copy_to_ubuf_axis_n(lp_cnt):
                        with ib_.for_range(0, lp_cnt, name="col") as col:
                            _emit_copy_gm_ubuf(
                                ib_,
                                "copy_gm_to_ubuf",
                                input_tensor.dtype,
                                actual_head_col_size,
                                input_ub,
                                col * vnchwconv_cube_col_size, input_tensor,
                                (params.block.var +
                                 n_loop_index * params.device_core_num) *
                                axis_c * axis_h * axis_w +
                                actual_head_col_size * col +
                                cube_i * actual_head_col_size * 16 +
                                in_offset)

                    if hwc_left == 0:
                        _inner_copy_to_ubuf_axis_n(16)
                    else:
                        with ib_.if_scope(cube_i != (loop_range - 1)):
                            _inner_copy_to_ubuf_axis_n(16)
                        with ib_.else_scope():
                            _inner_lp_cnt = hwc_left // actual_head_col_size
                            _inner_left_cnt = hwc_left % actual_head_col_size

                            if _inner_lp_cnt > 0:
                                _inner_copy_to_ubuf_axis_n(_inner_lp_cnt)
                            if _inner_left_cnt > 0:
                                _emit_copy_gm_ubuf(
                                    ib_,
                                    "copy_gm_to_ubuf",
                                    input_tensor.dtype,
                                    _inner_left_cnt,
                                    input_ub,
                                    _inner_lp_cnt * vnchwconv_cube_col_size,
                                    input_tensor,
                                    (params.block.var +
                                     n_loop_index * params.device_core_num) *
                                    axis_c * axis_h * axis_w +
                                    actual_head_col_size * _inner_lp_cnt +
                                    cube_i * actual_head_col_size * 16 +
                                    in_offset)

                    def _inner_first_conv_n():
                        # set VA address for conv1, the address should be
                        # aligned with 32Byte
                        _set_array_config(ib_, addr_array, addr_array_buf,
                                          src0_array_stride, dst0_array_stride,
                                          dst0_output_offset, src0_input_offset)
                        # do the conv1
                        repeat_conv1 = (actual_head_col_size + 15) // 16
                        src_stride_conv1 = 0 \
                            if repeat_conv1 * repeat_factor == 1 else 1
                        dst_stride_conv1 = 0 \
                            if repeat_conv1 * repeat_factor == 1 else 16
                        ib_.emit(
                            tvm.call_extern("int32",
                                            "scatter_vnchwconv_b16",
                                            "VA2",
                                            "VA0",
                                            repeat_conv1 * repeat_factor,
                                            dst_stride_conv1,
                                            src_stride_conv1))
                    _inner_first_conv_n()

                    if axis_c <= axis_c0:
                        repeat_conv2 = \
                            (actual_head_col_size + axis_c - 1) // axis_c
                        def _inner_vnconv_c_less_c0_n():
                            # to padding the column axis_c to axis_c0
                            # caution! the option maybe cover memory larger than
                            # one vnchwconv_cube_buf_max
                            # so it's better to split the input data by 16 lines
                            ib_.emit(
                                tvm.call_extern(
                                    output.dtype.lower(),
                                    "copy_ubuf_to_ubuf",
                                    input_ub.access_ptr(
                                        "w",
                                        offset=2 * vnchwconv_cube_buf_max),
                                    input_ub.access_ptr(
                                        "rw",
                                        offset=vnchwconv_cube_buf_max),
                                    0,
                                    repeat_conv2, axis_c * repeat_factor,
                                    0,
                                    (16 - axis_c) * repeat_factor))

                            # do the conv2
                            # change the dst_array_stride in order to make the data
                            # in output_ub is continuous
                            _set_array_config(ib_, addr_array, addr_array_buf,
                                              src1_array_stride,
                                              repeat_conv2 * 16 * repeat_factor,
                                              dst1_output_offset,
                                              src1_input_offset)
                            src_stride_conv2 = 0 \
                                if repeat_conv2 * repeat_factor == 1 else 16
                            dst_stride_conv2 = 0 \
                                if repeat_conv2 * repeat_factor == 1 else 1
                            ib_.emit(
                                tvm.call_extern("int32",
                                                "scatter_vnchwconv_b16",
                                                "VA2", "VA0",
                                                repeat_conv2 * repeat_factor,
                                                dst_stride_conv2,
                                                src_stride_conv2))
                        _inner_vnconv_c_less_c0_n()

                        # output the result
                        def output_result(offset):
                            with ib_.if_scope(cube_i != loop_range - 1):
                                _emit_copy_gm_ubuf(
                                    ib_, 'copy_ubuf_to_gm',
                                    output.dtype,
                                    repeat_conv2 * axis_c0 *16,
                                    output,
                                    (params.block.var +
                                     n_loop_index * params.device_core_num) *
                                    (c1_range * axis_c0) *
                                    axis_h * axis_w + cube_i *
                                    (repeat_conv2 * axis_c0 *16) +
                                    out_offset,
                                    output_ub, offset)
                            with ib_.else_scope():
                                _emit_copy_gm_ubuf(
                                    ib_, 'copy_ubuf_to_gm',
                                    output.dtype,
                                    axis_h * axis_w * axis_c0 -
                                    repeat_conv2 * axis_c0 *16 *
                                    (loop_range - 1),
                                    output,
                                    (params.block.var +
                                     n_loop_index * params.device_core_num) *
                                    (c1_range * axis_c0) * axis_h *
                                    axis_w + cube_i *
                                    (repeat_conv2 * axis_c0 *16) +
                                    out_offset,
                                    output_ub, offset)
                        output_result(0)

                    else:
                        # to padding the column axis_c to axis_c0
                        # caution! the option maybe cover memory larger than
                        # one vnchwconv_cube_buf_max
                        # so it's better to split the input data by 16 lines
                        repeat_conv2 =\
                            (actual_head_col_size + axis_c - 1) // axis_c
                        ib_.emit(
                            tvm.call_extern(
                                output.dtype.lower(),
                                "copy_ubuf_to_ubuf",
                                input_ub.access_ptr(
                                    "w",
                                    offset=2 * vnchwconv_cube_buf_max),
                                input_ub.access_ptr(
                                    "rw",
                                    offset=vnchwconv_cube_buf_max),
                                0,
                                repeat_conv2,
                                axis_c * repeat_factor,
                                0,
                                (c1_range * axis_c0 - axis_c) * repeat_factor))

                        # do the conv2
                        src_stride_conv2 = 0 if repeat_conv2 == 1 else \
                            c1_range * 16
                        dst_stride_conv2 = 0 if repeat_conv2 == 1 else 1
                        # repeat the vnchwconv by the (axis_c+16-1)//axis_c0
                        with ib_.for_range(0, c1_range, name="o") as c0_i:
                            # change the dst_array_stride in order to make the
                            # data in output_ub is continuous
                            # change the src_input_offset in order to get data
                            # by channel axis_c0
                            if output.dtype.lower() == "float16":
                                _set_array_config(
                                    ib_, addr_array,
                                    addr_array_buf,
                                    src1_array_stride,
                                    repeat_conv2 * 16,
                                    dst1_output_offset,
                                    src1_input_offset + c0_i * 256)
                                ib_.emit(
                                    tvm.call_extern("int32",
                                                    "scatter_vnchwconv_b16",
                                                    "VA2",
                                                    "VA0",
                                                    repeat_conv2,
                                                    dst_stride_conv2,
                                                    src_stride_conv2))
                            elif output.dtype.lower() == "float32":
                                with ib_.for_range(
                                        0, repeat_conv2, name="r") as repeat_i:
                                    _set_array_config(
                                        ib_,
                                        addr_array,
                                        addr_array_buf,
                                        src1_array_stride,
                                        repeat_conv2 * 16 * repeat_factor,
                                        dst1_output_offset +
                                        repeat_i * 16 * repeat_factor,
                                        src1_input_offset +
                                        c0_i * 512 +
                                        repeat_i * c1_range * 256 *
                                        repeat_factor)
                                    ib_.emit(
                                        tvm.call_extern(
                                            "int32",
                                            "scatter_vnchwconv_b16",
                                            "VA2",
                                            "VA0",
                                            2,
                                            1,
                                            16))

                            # output the result
                            # split the output address into c1_range parts
                            def output_result_multi_c0(offset):
                                with ib_.if_scope(cube_i != loop_range - 1):
                                    _emit_copy_gm_ubuf(
                                        ib_, 'copy_ubuf_to_gm',
                                        output.dtype,
                                        repeat_conv2 * axis_c0 *16,
                                        output,
                                        (params.block.var +
                                         n_loop_index *
                                         params.device_core_num) *
                                        (c1_range * axis_c0) *
                                        axis_h * axis_w + cube_i *
                                        (repeat_conv2 * axis_c0 *16) +
                                        c0_i * (axis_c0 * axis_h * axis_w) +
                                        out_offset,
                                        output_ub, offset)
                                with ib_.else_scope():
                                    tail_size =\
                                        axis_h * axis_w * axis_c0 - \
                                        repeat_conv2 * 16 * axis_c0 *\
                                        (loop_range - 1)
                                    _emit_copy_gm_ubuf(
                                        ib_, 'copy_ubuf_to_gm',
                                        output.dtype,
                                        tail_size,
                                        output,
                                        (params.block.var +
                                         n_loop_index *
                                         params.device_core_num) *
                                        (c1_range * axis_c0) *
                                        axis_h * axis_w + cube_i * \
                                        (repeat_conv2 * axis_c0 *16) +
                                        c0_i * (axis_c0 * axis_h * axis_w) +
                                        out_offset,
                                        output_ub, offset)

                            output_result_multi_c0(0)
            elif is_c_large_col:
                # get the loop count and left of axis c
                c_lp_cnt = axis_c // vnchwconv_cube_col_size
                c_lp_left = axis_c % vnchwconv_cube_col_size
                # get the loop count and left of axis h*w
                loop_range = axis_h * axis_w // 16
                loop_range_left = axis_h * axis_w % 16

                def _c_large_col_intrin(hw_lp_i, c_lp_i, col_cnt):
                    with ib_.for_range(0, col_cnt, name="col") as col:
                        _emit_copy_gm_ubuf(
                            ib_,
                            "copy_gm_to_ubuf",
                            input_tensor.dtype,
                            vnchwconv_cube_col_size,
                            input_ub,
                            col * vnchwconv_cube_col_size,
                            input_tensor,
                            (params.block.var +
                             n_loop_index * params.device_core_num) *
                            axis_c * axis_h * axis_w +
                            axis_c * col +
                            c_lp_i * vnchwconv_cube_col_size +
                            hw_lp_i * axis_c * 16 +
                            in_offset)

                    # set VA address for conv1, the address should
                    # be aligned with 32Byte
                    # src0_array_stride = vnchwconv_cube_col_size
                    # src0_input_offset = 0
                    # dst0_array_stride = axis_c0
                    # dst0_output_offset = vnchwconv_cube_buf_max
                    _set_array_config(ib_, addr_array,
                                      addr_array_buf,
                                      src0_array_stride,
                                      dst0_array_stride,
                                      dst0_output_offset,
                                      src0_input_offset)
                    # do the conv1
                    repeat_conv1 = \
                        (vnchwconv_cube_col_size + 15) // 16
                    src_stride_conv1 = 0 \
                        if repeat_conv1 * repeat_factor == 1 else 1
                    dst_stride_conv1 = 0 \
                        if repeat_conv1 * repeat_factor == 1 else 16
                    ib_.emit(
                        tvm.call_extern("int32",
                                        "scatter_vnchwconv_b16",
                                        "VA2",
                                        "VA0",
                                        repeat_conv1 * repeat_factor,
                                        dst_stride_conv1,
                                        src_stride_conv1))

                    # do the conv2
                    repeat_conv2 = repeat_conv1
                    # do the vnchwconv
                    if output.dtype.lower() == "float16":
                        src_stride_conv2 = 0 if repeat_conv2 == 1 else 16
                        dst_stride_conv2 = 0 if repeat_conv2 == 1 else 16
                        _set_array_config(
                            ib_, addr_array,
                            addr_array_buf,
                            src1_array_stride,
                            dst0_array_stride,
                            src1_input_offset,
                            dst0_output_offset)
                        ib_.emit(
                            tvm.call_extern("int32",
                                            "scatter_vnchwconv_b16",
                                            "VA2",
                                            "VA0",
                                            repeat_conv2,
                                            dst_stride_conv2,
                                            src_stride_conv2))
                    elif output.dtype.lower() == "float32":
                        src_stride_conv2 = 0 if repeat_conv2 == 1 else 32
                        dst_stride_conv2 = 0 if repeat_conv2 == 1 else 32
                        with ib_.for_range(0, 2) as vn_i:
                            _set_array_config(
                                ib_, addr_array,
                                addr_array_buf,
                                src1_array_stride,
                                dst0_array_stride * 2,
                                src1_input_offset + vn_i * 16,
                                dst0_output_offset + vn_i * 256)
                            ib_.emit(
                                tvm.call_extern("int32",
                                                "scatter_vnchwconv_b16",
                                                "VA2",
                                                "VA0",
                                                repeat_conv2,
                                                dst_stride_conv2,
                                                src_stride_conv2))

                    # output the result
                    # split the output address into c1_range parts
                    def output_result_multi_c0(offset):
                        with ib_.for_range(
                                0, repeat_conv1,
                                name="repeat") as repeat_i:
                            _emit_copy_gm_ubuf(
                                ib_, 'copy_ubuf_to_gm',
                                output.dtype,
                                axis_c0 * col_cnt,
                                output,
                                (params.block.var +
                                 n_loop_index *
                                 params.device_core_num) *
                                (c1_range * axis_c0) *
                                axis_h * axis_w +
                                hw_lp_i * (axis_c0 * 16) +
                                (repeat_i + c_lp_i * repeat_conv1) *
                                (axis_h * axis_w * axis_c0) +
                                out_offset,
                                input_ub,
                                repeat_i * (axis_c0 * 16) +
                                2 * vnchwconv_cube_buf_max +
                                offset)

                    output_result_multi_c0(0)

                def _c_large_col_tail_intrin(hw_lp_i, c_lp_i, col_cnt):

                    with ib_.for_range(0, col_cnt, name="col") as col:
                        _emit_copy_gm_ubuf(
                            ib_,
                            "copy_gm_to_ubuf",
                            input_tensor.dtype,
                            c_lp_left,
                            input_ub,
                            col * vnchwconv_cube_col_size,
                            input_tensor,
                            (params.block.var +
                             n_loop_index * params.device_core_num) *
                            axis_c * axis_h * axis_w +
                            axis_c * col +
                            c_lp_i * vnchwconv_cube_col_size +
                            hw_lp_i * axis_c * 16 +
                            in_offset)

                    # set VA address for conv1, the address should
                    # be aligned with 32Byte
                    # src0_array_stride = vnchwconv_cube_col_size
                    # src0_input_offset = 0
                    # dst0_array_stride = axis_c0
                    # dst0_output_offset = vnchwconv_cube_buf_max
                    _set_array_config(ib_, addr_array,
                                      addr_array_buf,
                                      src0_array_stride,
                                      dst0_array_stride,
                                      dst0_output_offset,
                                      src0_input_offset)
                    # do the conv1
                    repeat_conv1 = \
                        (c_lp_left + 15) // 16
                    src_stride_conv1 = 0 \
                        if repeat_conv1 * repeat_factor == 1 else 1
                    dst_stride_conv1 = 0 \
                        if repeat_conv1 * repeat_factor == 1 else 16
                    ib_.emit(
                        tvm.call_extern("int32",
                                        "scatter_vnchwconv_b16",
                                        "VA2",
                                        "VA0",
                                        repeat_conv1 * repeat_factor,
                                        dst_stride_conv1,
                                        src_stride_conv1))

                    # clean ub used by 1st conv
                    _clean_ubuf(ib_, input_ub,
                                vnchwconv_cube_buf_max + c_lp_left * 16,
                                (16 - c_lp_left % 16) * 16)

                    # do the conv2
                    repeat_conv2 = repeat_conv1
                    # do the vnchwconv
                    if output.dtype.lower() == "float16":
                        src_stride_conv2 = 0 if repeat_conv2 == 1 else 16
                        dst_stride_conv2 = 0 if repeat_conv2 == 1 else 16
                        _set_array_config(
                            ib_, addr_array,
                            addr_array_buf,
                            src1_array_stride,
                            dst0_array_stride,
                            src1_input_offset,
                            dst0_output_offset)
                        ib_.emit(
                            tvm.call_extern("int32",
                                            "scatter_vnchwconv_b16",
                                            "VA2",
                                            "VA0",
                                            repeat_conv2,
                                            dst_stride_conv2,
                                            src_stride_conv2))
                    elif output.dtype.lower() == "float32":
                        src_stride_conv2 = 0 if repeat_conv2 == 1 else 32
                        dst_stride_conv2 = 0 if repeat_conv2 == 1 else 32
                        with ib_.for_range(0, 2) as vn_i:
                            _set_array_config(
                                ib_, addr_array,
                                addr_array_buf,
                                src1_array_stride,
                                dst0_array_stride * 2,
                                src1_input_offset + vn_i * 16,
                                dst0_output_offset + vn_i * 256)
                            ib_.emit(
                                tvm.call_extern("int32",
                                                "scatter_vnchwconv_b16",
                                                "VA2",
                                                "VA0",
                                                repeat_conv2,
                                                dst_stride_conv2,
                                                src_stride_conv2))

                    # output the result
                    # split the output address into c1_range parts
                    def output_result_multi_c0(offset):
                        with ib_.for_range(
                                0, repeat_conv1,
                                name="repeat") as repeat_i:
                            _emit_copy_gm_ubuf(
                                ib_, 'copy_ubuf_to_gm',
                                output.dtype,
                                axis_c0 * col_cnt,
                                output,
                                (params.block.var +
                                 n_loop_index *
                                 params.device_core_num) *
                                (c1_range * axis_c0) *
                                axis_h * axis_w +
                                hw_lp_i * (axis_c0 * 16) +
                                (repeat_i +
                                 c_lp_i * (vnchwconv_cube_col_size//axis_c0)) *
                                (axis_h * axis_w * axis_c0) +
                                out_offset,
                                input_ub,
                                repeat_i * (axis_c0 * 16) +
                                2 * vnchwconv_cube_buf_max +
                                offset)

                    output_result_multi_c0(0)

                if loop_range > 0:
                    with ib_.for_range(
                            0, loop_range, name="hw_lp_idx") as hw_lp_i:
                        # the nchwconv uses 16 lines
                        with ib_.for_range(
                                0, c_lp_cnt, name="c_lp_idx") as c_lp_i:
                            _c_large_col_intrin(hw_lp_i, c_lp_i, 16)

                        if c_lp_left > 0:
                            _c_large_col_tail_intrin(hw_lp_i, c_lp_cnt, 16)

                if loop_range_left > 0:
                    # the nchwconv uses 16 lines
                    with ib_.for_range(
                            0, c_lp_cnt, name="c_lp_idx") as c_lp_i:
                        _c_large_col_intrin(
                            loop_range, c_lp_i, loop_range_left)

                    if c_lp_left > 0:
                        _c_large_col_tail_intrin(
                            loop_range, c_lp_cnt, loop_range_left)
            else:
                # Standard processing:
                # Copy gm to ub in 16 copies for each split --> Update
                # transpose config --> Transposed data first time -->
                # Copy ub to ub at one-time to padding axis_c to n*axis_c0 -->
                # Transposed data second time --> Copy ub to gm

                # the nchwconv uses 16 lines
                hw_cnt = (axis_h * axis_w + 15) // 16
                loop_cnt = axis_h * axis_w // hw_cnt
                loop_left = axis_h * axis_w % hw_cnt
                with ib_.for_range(0, loop_cnt, name="col") as col:
                    _emit_copy_gm_ubuf(ib_, "copy_gm_to_ubuf",
                                       input_tensor.dtype,
                                       (axis_h*axis_w+15) // 16 * axis_c,
                                       input_ub, col * vnchwconv_cube_col_size,
                                       input_tensor,
                                       (params.block.var +
                                        n_loop_index *
                                        params.device_core_num) *
                                       axis_c * axis_h * axis_w +
                                       ((axis_h*axis_w+15)//16*axis_c) * col +
                                       in_offset)
                _emit_copy_gm_ubuf(ib_, "copy_gm_to_ubuf",
                                   input_tensor.dtype,
                                   loop_left * axis_c,
                                   input_ub,
                                   loop_cnt * vnchwconv_cube_col_size,
                                   input_tensor,
                                   (params.block.var +
                                    n_loop_index *
                                    params.device_core_num) *
                                   axis_c * axis_h * axis_w +
                                   ((axis_h*axis_w+15)//16*axis_c) * loop_cnt +
                                   in_offset)

                # set VA address for conv1, the address should be aligned with
                # 32Byte
                _set_array_config(ib_, addr_array, addr_array_buf,
                                  src0_array_stride, dst0_array_stride,
                                  dst0_output_offset, src0_input_offset)
                # do the conv1
                repeat_conv1 = ((axis_h * axis_w + 15) // 16 * axis_c + 15) // 16
                src_stride_conv1 = 0\
                    if repeat_conv1 * repeat_factor == 1 else 1
                dst_stride_conv1 = 0\
                    if repeat_conv1 * repeat_factor == 1 else 16
                ib_.emit(
                    tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    repeat_conv1 * repeat_factor,
                                    dst_stride_conv1,
                                    src_stride_conv1))

                if axis_c <= axis_c0:
                    # to padding the column axis_c to axis_c0
                    # caution! the option maybe cover memory larger than one
                    # vnchwconv_cube_buf_max
                    # so it's better to split the input data by 16 lines
                    repeat_conv2 =\
                        ((axis_h * axis_w + 15) // 16 * axis_c +
                         axis_c - 1) // axis_c
                    ib_.emit(
                        tvm.call_extern(
                            "float16",
                            "copy_ubuf_to_ubuf",
                            input_ub.access_ptr(
                                "w",
                                offset=2 * vnchwconv_cube_buf_max),
                            input_ub.access_ptr(
                                "rw",
                                offset=vnchwconv_cube_buf_max),
                            0,
                            repeat_conv2,
                            axis_c * repeat_factor,
                            0,
                            (16 - axis_c) * repeat_factor))

                    # do the conv2
                    _set_array_config(ib_, addr_array,
                                      addr_array_buf,
                                      src1_array_stride,
                                      repeat_conv2 * 16 * repeat_factor,
                                      dst1_output_offset,
                                      src1_input_offset)
                    src_stride_conv2 = 0\
                        if repeat_conv2 * repeat_factor == 1 else 16
                    dst_stride_conv2 = 0\
                        if repeat_conv2 * repeat_factor == 1 else 1
                    ib_.emit(
                        tvm.call_extern("int32",
                                        "scatter_vnchwconv_b16",
                                        "VA2",
                                        "VA0",
                                        repeat_conv2 * repeat_factor,
                                        dst_stride_conv2,
                                        src_stride_conv2))

                    # output the result
                    _emit_copy_gm_ubuf(
                        ib_, 'copy_ubuf_to_gm',
                        output.dtype,
                        axis_h * axis_w * axis_c0,
                        output,
                        (params.block.var +
                         n_loop_index * params.device_core_num) *
                        (c1_range * axis_c0) * axis_h * axis_w +
                        out_offset,
                        output_ub, 0)

                else:
                    # to padding the column axis_c to axis_c0
                    # caution! the option maybe cover memory larger than one
                    # vnchwconv_cube_buf_max
                    # so it's better to split the input data by 16 lines
                    repeat_conv2 =\
                        ((axis_h * axis_w + 15) // 16 * axis_c +
                         axis_c - 1) // axis_c
                    ib_.emit(
                        tvm.call_extern(
                            "float16", "copy_ubuf_to_ubuf",
                            input_ub.access_ptr(
                                "w",
                                offset=2 * vnchwconv_cube_buf_max),
                            input_ub.access_ptr(
                                "rw",
                                offset=vnchwconv_cube_buf_max), 0,
                            repeat_conv2,
                            axis_c * repeat_factor,
                            0,
                            (c1_range * axis_c0 - axis_c) * repeat_factor))

                    # do the conv2
                    src_stride_conv2 = 0 if repeat_conv2 == 1 else c1_range*16
                    dst_stride_conv2 = 0 if repeat_conv2 == 1 else 1
                    # repeat the vnchwconv by the (axis_c+16-1)//axis_c0
                    with ib_.for_range(0, c1_range, name="o") as c0_i:
                        if output.dtype.lower() == "float16":
                            _set_array_config(
                                ib_, addr_array,
                                addr_array_buf,
                                src1_array_stride,
                                repeat_conv2 * 16,
                                dst1_output_offset,
                                src1_input_offset + c0_i * 256)
                            ib_.emit(
                                tvm.call_extern(
                                    "int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    repeat_conv2,
                                    dst_stride_conv2,
                                    src_stride_conv2))
                        elif output.dtype.lower() == "float32":
                            with ib_.for_range(
                                    0, repeat_conv2, name="r") as repeat_i:
                                _set_array_config(
                                    ib_,
                                    addr_array,
                                    addr_array_buf,
                                    src1_array_stride,
                                    repeat_conv2 * 16 * repeat_factor,
                                    dst1_output_offset +
                                    repeat_i * 16 * repeat_factor,
                                    src1_input_offset + c0_i * 512 +
                                    repeat_i*c1_range * 256 * repeat_factor)
                                ib_.emit(tvm.call_extern(
                                    "int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    2,
                                    1,
                                    16))

                        # output the result
                        _emit_copy_gm_ubuf(
                            ib_, 'copy_ubuf_to_gm',
                            output.dtype,
                            axis_c0 * axis_h * axis_w,
                            output,
                            (params.block.var +
                             n_loop_index * params.device_core_num) *
                            (c1_range * axis_c0) * axis_h * axis_w +
                            out_offset + c0_i * axis_c0 * axis_h * axis_w,
                            output_ub, 0)

        if n_count_per_core > 0:
            with ib_.for_range(
                    0, n_count_per_core, name="n_loop_index") as n_loop_index:
                _inner_intrin_axis_n(n_loop_index, 0, 0)
        if n_count_tail > 0:
            with ib_.if_scope(params.block.var < n_count_tail):
                _inner_intrin_axis_n(n_count_per_core, 0, 0)

    if n_count_per_core < 1 < c_count_per_core and \
        ((input_tensor.dtype.lower() == "float16" and axis_c <= 1984) or
         (input_tensor.dtype.lower() == "float32" and axis_c <= 992)):
        _four2five_intrin(params.ib_)
    else:
        _four2five_intrin_axis_n(params.ib_)

    return ib_.get()


def _four2five_ir_nchw(input_tensor, shape_4d, output):
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
    hw_per_core_count = (actual_col_size // params.device_core_num // 16) * 16
    # left element count in axis_h*axis_w
    if hw_per_core_count > 0:
        hw_left =\
            actual_col_size % (hw_per_core_count * params.device_core_num)
    else:
        hw_left = actual_col_size
    # split axis_n by core count
    n_per_core_count = axis_n // params.device_core_num
    # left axis_n
    n_left = axis_n % params.device_core_num

    c1_range = (axis_c + axis_c0 - 1) // axis_c0
    c1_loop = c1_range
    c1_tail = axis_c % axis_c0  # C1 tail
    if c1_tail != 0:
        c1_loop = c1_range - 1  # remove tail

    def _four2five_intrin(
            ib_, hw_per_core, c0_loop, c1_begin,
            c1_end, in_offset, out_offset):
        '''
        :param ib_:
            type: tvm.ir_builder
            desc: tvm.ir_builder
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
        with ib_.for_range(0, axis_n, name="n") as n_index:
            with ib_.for_range(0, c1_end - c1_begin, name="c1") as c1_index:
                buf_cube_range = (hw_per_core + vnchwconv_cube_col_size - 1) \
                                 // vnchwconv_cube_col_size
                hw_per_core_left = hw_per_core % vnchwconv_cube_col_size
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
                                    n_index * axis_c * axis_h * axis_w +
                                    ((c1_index + c1_begin) * axis_c0 + col) *
                                    actual_col_size +
                                    cube_i * vnchwconv_cube_col_size +
                                    params.block.var * hw_per_core + in_offset)

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

                            _emit_copy_gm_ubuf(
                                ib_, 'copy_ubuf_to_gm', output.dtype,
                                vnchwconv_cube_buf_max, output,
                                n_index * (c1_range * axis_c0) *
                                axis_h * axis_w +
                                (c1_index + c1_begin) *
                                axis_c0 * actual_col_size +
                                cube_i * vnchwconv_cube_buf_max +
                                params.block.var * hw_per_core * 16 +
                                out_offset, output_ub, 0)
                        with ib_.else_scope():
                            # Tail processing:
                            # Copy gm to ub in 16 copies --> Update transpose
                            # config --> Transposed data --> Copy ub to gm at
                            # one-time

                            tail_c0_col_size = \
                                hw_per_core_left \
                                    if hw_per_core_left > 0 \
                                    else vnchwconv_cube_col_size
                            if input_tensor.dtype.lower() == "float16":
                                with ib_.for_range(
                                        0, c0_loop, name="col") as col:
                                    _emit_copy_gm_ubuf(
                                        ib_, "copy_gm_to_ubuf",
                                        input_tensor.dtype, tail_c0_col_size,
                                        input_ub,
                                        col * vnchwconv_cube_col_size,
                                        input_tensor,
                                        n_index * axis_c * axis_h * axis_w +
                                        ((c1_index + c1_begin) *
                                         axis_c0 + col) *
                                        actual_col_size +
                                        cube_i * vnchwconv_cube_col_size +
                                        params.block.var * hw_per_core +
                                        in_offset)

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
                                        n_index * axis_c * axis_h * axis_w +
                                        ((c1_index + c1_begin) *
                                         axis_c0 + col) * actual_col_size +
                                        cube_i * vnchwconv_cube_col_size +
                                        params.block.var * hw_per_core +
                                        in_offset)

                                args = (ib_, input_ub, output_ub, 0, 0,
                                        tail_c0_col_size, c0_loop, reg_array)
                                _mov_data_p2p(args)

                            _emit_copy_gm_ubuf(
                                ib_, 'copy_ubuf_to_gm', output.dtype,
                                tail_c0_col_size * 16, output,
                                n_index * (c1_range * axis_c0) *
                                axis_h * axis_w +
                                (c1_index + c1_begin) *
                                axis_c0 * actual_col_size +
                                cube_i * vnchwconv_cube_buf_max +
                                params.block.var * hw_per_core * 16 +
                                out_offset, output_ub, 0)
                    else:
                        # Standard processing:
                        # Copy gm to ub in 16 copies --> Update transpose
                        # config -->
                        # Transposed data --> Copy ub to gm at one-time

                        if input_tensor.dtype.lower() == "float16":
                            with ib_.for_range(0, c0_loop, name="col") as col:
                                _emit_copy_gm_ubuf(
                                    ib_, "copy_gm_to_ubuf", input_tensor.dtype,
                                    hw_per_core, input_ub,
                                    col * vnchwconv_cube_col_size,
                                    input_tensor,
                                    n_index * axis_c * axis_h * axis_w +
                                    ((c1_index + c1_begin) * axis_c0 + col) *
                                    actual_col_size +
                                    cube_i * vnchwconv_cube_col_size +
                                    params.block.var * hw_per_core + in_offset)

                            repeat = (hw_per_core + 15) // 16
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
                            col_size = (hw_per_core + 15) // 16 * 16
                            with ib_.for_range(0, c0_loop, name="col") as col:
                                _emit_copy_gm_ubuf(
                                    ib_, "copy_gm_to_ubuf", input_tensor.dtype,
                                    hw_per_core, input_ub,
                                    col * col_size,
                                    input_tensor,
                                    n_index * axis_c * axis_h * axis_w +
                                    ((c1_index + c1_begin) * axis_c0 + col) *
                                    actual_col_size +
                                    cube_i * vnchwconv_cube_col_size +
                                    params.block.var * hw_per_core + in_offset)

                            args = (ib_, input_ub, output_ub, 0, 0,
                                    hw_per_core, c0_loop, reg_array)
                            _mov_data_p2p(args)

                        _emit_copy_gm_ubuf(
                            ib_, 'copy_ubuf_to_gm', output.dtype,
                            hw_per_core * 16, output,
                            n_index * (c1_range * axis_c0) * axis_h * axis_w +
                            (c1_index + c1_begin) * axis_c0 * actual_col_size +
                            cube_i * vnchwconv_cube_buf_max +
                            params.block.var * hw_per_core * 16 + out_offset,
                            output_ub, 0)

    def _four2five_intrin_axis_n(
            ib_, n_index, c0_loop, c1_begin,
            c1_end, in_offset, out_offset):
        '''
        :param ib_:
            type: tvm.ir_builder
            desc: tvm.ir_builder
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
            buf_cube_range = (actual_col_size + vnchwconv_cube_col_size - 1) \
                             // vnchwconv_cube_col_size
            with ib_.for_range(
                    0, buf_cube_range, name="buf_cube_index") as cube_i:
                if buf_cube_range > 1:
                    # Standard processing:
                    # Copy gm to ub in 16 copies --> Update transpose
                    # config --> Transposed data --> Copy ub to gm at
                    # one-time
                    if actual_col_size % vnchwconv_cube_col_size == 0:
                        buf_cube_range = buf_cube_range + 1

                    def _inner_vnchwconv(repeat):
                        """
                        do vnchwconv transfer
                        """

                        src_stride = 0 if repeat == 1 else 1
                        dst_stride = 0 if repeat == 1 else 16
                        ib_.emit(
                            tvm.call_extern("int32",
                                            "scatter_vnchwconv_b16",
                                            "VA2", "VA0", repeat,
                                            dst_stride, src_stride))

                    with ib_.if_scope(cube_i != buf_cube_range - 1):
                        with ib_.for_range(0, c0_loop, name="col") as col:
                            _emit_copy_gm_ubuf(
                                ib_, "copy_gm_to_ubuf", input_tensor.dtype,
                                vnchwconv_cube_col_size,
                                input_ub,
                                col * vnchwconv_cube_col_size,
                                input_tensor,
                                (params.block.var +
                                 n_index * params.device_core_num) *
                                axis_c * axis_h * axis_w +
                                ((c1_index + c1_begin) * axis_c0 + col) *
                                actual_col_size +
                                cube_i * vnchwconv_cube_col_size +
                                in_offset)

                        if input_tensor.dtype.lower() == "float16":
                            repeat = vnchwconv_cube_col_size // 16
                            _inner_vnchwconv(repeat)
                        else:
                            args = (ib_, input_ub, output_ub, 0, 0,
                                    vnchwconv_cube_col_size, c0_loop,
                                    reg_array)
                            _mov_data_p2p(args)

                        _emit_copy_gm_ubuf(
                            ib_, 'copy_ubuf_to_gm', output.dtype,
                            vnchwconv_cube_buf_max,
                            output,
                            (params.block.var +
                             n_index * params.device_core_num) *
                            (c1_range * axis_c0) * axis_h * axis_w +
                            (c1_index + c1_begin) * axis_c0 * axis_h * axis_w +
                            cube_i * vnchwconv_cube_buf_max + out_offset,
                            output_ub, 0)
                    with ib_.else_scope():
                        # Tail processing:
                        # Copy gm to ub in 16 copies --> Update transpose
                        # config --> Transposed data --> Copy ub to gm at
                        # one-time

                        tail_c0_col_size = \
                            actual_col_size % vnchwconv_cube_col_size
                        if input_tensor.dtype.lower() == "float16":
                            with ib_.for_range(
                                    0, c0_loop, name="col") as col:
                                _emit_copy_gm_ubuf(
                                    ib_, "copy_gm_to_ubuf",
                                    input_tensor.dtype,
                                    tail_c0_col_size,
                                    input_ub,
                                    col * vnchwconv_cube_col_size,
                                    input_tensor,
                                    (params.block.var +
                                     n_index * params.device_core_num) *
                                    axis_c * axis_h * axis_w +
                                    ((c1_index + c1_begin) * axis_c0 + col) *
                                    actual_col_size +
                                    cube_i * vnchwconv_cube_col_size +
                                    in_offset)

                            repeat = (tail_c0_col_size + 15) // 16
                            _inner_vnchwconv(repeat)
                        else:
                            col_size = (tail_c0_col_size + 15) // 16 * 16
                            with ib_.for_range(
                                    0, c0_loop, name="col") as col:
                                _emit_copy_gm_ubuf(
                                    ib_, "copy_gm_to_ubuf",
                                    input_tensor.dtype,
                                    tail_c0_col_size,
                                    input_ub,
                                    col * col_size,
                                    input_tensor,
                                    (params.block.var +
                                     n_index * params.device_core_num) *
                                    axis_c * axis_h * axis_w +
                                    ((c1_index + c1_begin) *
                                     axis_c0 + col) * actual_col_size +
                                    cube_i * vnchwconv_cube_col_size +
                                    in_offset)

                            args = (ib_, input_ub, output_ub, 0, 0,
                                    tail_c0_col_size, c0_loop, reg_array)
                            _mov_data_p2p(args)

                        _emit_copy_gm_ubuf(
                            ib_, 'copy_ubuf_to_gm', output.dtype,
                            tail_c0_col_size * 16,
                            output,
                            (params.block.var +
                             n_index * params.device_core_num) *
                            (c1_range * axis_c0) * axis_h * axis_w +
                            (c1_index + c1_begin) * axis_c0 * axis_h * axis_w +
                            cube_i * vnchwconv_cube_buf_max + out_offset,
                            output_ub, 0)
                else:
                    # Standard processing:
                    # Copy gm to ub in 16 copies --> Update transpose
                    # config -->
                    # Transposed data --> Copy ub to gm at one-time

                    if input_tensor.dtype.lower() == "float16":
                        with ib_.for_range(0, c0_loop, name="col") as col:
                            _emit_copy_gm_ubuf(
                                ib_, "copy_gm_to_ubuf", input_tensor.dtype,
                                actual_col_size, input_ub,
                                col * vnchwconv_cube_col_size,
                                input_tensor,
                                (params.block.var +
                                 n_index * params.device_core_num) *
                                axis_c * axis_h * axis_w +
                                ((c1_index + c1_begin) * axis_c0 + col) *
                                actual_col_size +
                                cube_i * vnchwconv_cube_col_size +
                                in_offset)

                        repeat = (actual_col_size + 15) // 16
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
                        col_size = (actual_col_size + 15) // 16 * 16
                        with ib_.for_range(0, c0_loop, name="col") as col:
                            _emit_copy_gm_ubuf(
                                ib_, "copy_gm_to_ubuf", input_tensor.dtype,
                                actual_col_size,
                                input_ub,
                                col * col_size,
                                input_tensor,
                                (params.block.var +
                                 n_index * params.device_core_num) *
                                axis_c * axis_h * axis_w +
                                ((c1_index + c1_begin) * axis_c0 + col) *
                                actual_col_size +
                                cube_i * vnchwconv_cube_col_size +
                                in_offset)

                        args = (ib_, input_ub, output_ub, 0, 0,
                                actual_col_size, c0_loop, reg_array)
                        _mov_data_p2p(args)

                    _emit_copy_gm_ubuf(
                        ib_, 'copy_ubuf_to_gm', output.dtype,
                        actual_col_size * 16,
                        output,
                        (params.block.var + n_index * params.device_core_num) *
                        (c1_range * axis_c0) * axis_h * axis_w +
                        (c1_index + c1_begin) * axis_c0 * axis_h * axis_w +
                        cube_i * vnchwconv_cube_buf_max +
                        out_offset,
                        output_ub, 0)

    if n_per_core_count < 1 < hw_per_core_count:
        if hw_per_core_count > 0 and c1_loop > 0:
            _four2five_intrin(params.ib_, hw_per_core_count, axis_c0, 0,
                              c1_loop, 0, 0)
            if hw_left > 0:
                with ib_.if_scope(params.block.var < 1):
                    _four2five_intrin(
                        params.ib_, hw_left, axis_c0, 0, c1_loop,
                        hw_per_core_count * params.device_core_num,
                        hw_per_core_count * params.device_core_num * 16)
        elif c1_loop > 0:
            # clean ubuf to avoid dirty data
            if input_tensor.dtype.lower() == "float16":
                _clean_ubuf(ib_, input_ub, 0, vnchwconv_cube_buf_max)
            else:
                _clean_ubuf(ib_, output_ub, 0, vnchwconv_cube_buf_max)
            with ib_.if_scope(params.block.var < 1):
                _four2five_intrin(params.ib_, hw_left, axis_c0, 0,
                                  c1_loop, 0, 0)

        if c1_tail != 0:  # c1 tail processing
            # clean ubuf to avoid dirty data
            clean_len = hw_per_core_count \
                if hw_per_core_count < vnchwconv_cube_col_size \
                else vnchwconv_cube_col_size
            if input_tensor.dtype.lower() == "float16":
                _clean_ubuf(ib_, input_ub, c1_tail * vnchwconv_cube_col_size,
                            (axis_c0 - c1_tail) * vnchwconv_cube_col_size)
            else:
                _clean_ubuf(ib_, output_ub, 0,
                            axis_c0 * clean_len)

            if hw_per_core_count > 0:
                _four2five_intrin(params.ib_, hw_per_core_count, c1_tail,
                                  c1_loop, c1_range, 0, 0)
                if hw_left > 0:
                    with ib_.if_scope(params.block.var < 1):
                        _four2five_intrin(
                            params.ib_, hw_left, c1_tail, c1_loop, c1_range,
                            hw_per_core_count * params.device_core_num,
                            hw_per_core_count * params.device_core_num * 16)
            else:
                with ib_.if_scope(params.block.var < 1):
                    _four2five_intrin(params.ib_, hw_left, c1_tail, c1_loop,
                                      c1_range, 0, 0)
    else:
        if n_per_core_count > 0 and c1_loop > 0:
            with ib_.for_range(0, n_per_core_count, name="n_index") as n_index:
                _four2five_intrin_axis_n(
                    params.ib_, n_index,
                    axis_c0, 0, c1_loop, 0, 0)
            if n_left > 0:
                with ib_.if_scope(params.block.var < n_left):
                    _four2five_intrin_axis_n(
                        params.ib_, n_per_core_count,
                        axis_c0, 0, c1_loop, 0, 0)
        elif c1_loop > 0:
            with ib_.if_scope(params.block.var < n_left):
                _four2five_intrin_axis_n(
                    params.ib_, 0, axis_c0, 0, c1_loop, 0, 0)

        if c1_tail != 0:  # c1 tail processing
            # clean ubuf to avoid dirty data
            clean_len = actual_col_size \
                if actual_col_size < vnchwconv_cube_col_size \
                else vnchwconv_cube_col_size
            if input_tensor.dtype.lower() == "float16":
                _clean_ubuf(ib_, input_ub, c1_tail * vnchwconv_cube_col_size,
                            (axis_c0 - c1_tail) * vnchwconv_cube_col_size)
            else:
                _clean_ubuf(ib_, output_ub, 0,
                            axis_c0 * clean_len)

            if n_per_core_count > 0:
                with ib_.for_range(0, n_per_core_count, "n_index") as n_index:
                    _four2five_intrin_axis_n(
                        params.ib_, n_index,
                        c1_tail, c1_loop, c1_range, 0, 0)
                if n_left > 0:
                    with ib_.if_scope(params.block.var < n_left):
                        _four2five_intrin_axis_n(
                            params.ib_, n_per_core_count,
                            c1_tail, c1_loop, c1_range, 0, 0)
            else:
                with ib_.if_scope(params.block.var < n_left):
                    _four2five_intrin_axis_n(
                        params.ib_, 0, c1_tail, c1_loop, c1_range, 0, 0)

    return ib_.get()
