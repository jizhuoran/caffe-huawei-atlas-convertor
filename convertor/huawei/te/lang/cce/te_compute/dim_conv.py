#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

div compute
"""

from te import tvm
from te.platform import cce_intrin as intrin
from te.platform import cce_params as param
from te.platform import cce_util as util
from te.platform import get_soc_spec

OUTPUT_NAME_SUFFIX = [0]
UB_NAME_SUFFIX = [0]


def compute_four2five(input_tensor, raw_shape_4D):
    '''
    :param input_tensor:
        tyep: Tensor
        desc: input data
    :param raw_shape_4D:
        format: (N, C, H, W)
        type: tuple or list
        desc: the raw shape of tensor
    :return:
        type: Tensor
    '''
    if not isinstance(input_tensor, tvm.tensor.Tensor):
        raise RuntimeError("The input type must be tvm.tensor")

    if not isinstance(raw_shape_4D, (list, tuple)):
        raise RuntimeError("The raw_shape_4D type must be list or tuple")

    if len(raw_shape_4D) != 4:
        raise RuntimeError("The length of raw_shape_4D must be 4")

    if input_tensor.dtype != "float16":
        raise RuntimeError("The data type of input_tensor must be float16")

    shape_n, shape_c, shape_h, shape_w = raw_shape_4D
    shape_c1 = (shape_c + 16 - 1) // 16
    OUTPUT_NAME_SUFFIX[0] += 1
    return tvm.extern([(shape_n, shape_c1, shape_h, shape_w, 16)],
                      [input_tensor],
                      lambda ins, outs: _four2five_ir(ins[0], raw_shape_4D,
                                                      outs[0]),
                      dtype=[input_tensor.dtype],
                      name="output_" + hex(OUTPUT_NAME_SUFFIX[0]))


def compute_five2four(input_tensor, raw_shape_4D):
    '''
    :param input_tensor:
        tyep: Tensor
        desc: input data
    :param raw_shape_4D:
        format: (N, C, H, W)
        type: tuple or list
        desc: the raw shape of tensor
    :return:
        type: Tensor
    '''

    if not isinstance(input_tensor, tvm.tensor.Tensor):
        raise RuntimeError("The input type must be tvm.tensor")

    if not isinstance(raw_shape_4D, (list, tuple)):
        raise RuntimeError("The raw_shape_4D type must be list or tuple")

    if len(raw_shape_4D) != 4:
        raise RuntimeError("The length of raw_shape_4D must be 4")

    if input_tensor.dtype != "float16":
        raise RuntimeError("The data type of input_tensor must be float16")

    shape_n, shape_c, shape_h, shape_w = raw_shape_4D
    OUTPUT_NAME_SUFFIX[0] += 1
    return tvm.extern([(shape_n, shape_c, shape_h, shape_w)], [input_tensor],
                      lambda ins, outs: _five2four_ir(ins[0], raw_shape_4D,
                                                      outs[0]),
                      dtype=[input_tensor.dtype],
                      name="output_" + hex(OUTPUT_NAME_SUFFIX[0]))


def _decl_memory(buffer_scope):
    '''
    :param buffer_scope:
        type: str
        desc: the scope of ubuferr
    :return:
        type: node
    '''
    func = tvm.get_global_func("tvm.info.mem.%s" % buffer_scope, True)
    if func is not None:
        return
    # pylint: disable=protected-access, unused-variable
    try:
        @tvm.register_func("tvm.info.mem.%s" % buffer_scope)
        def mem_info_ub_buffer():
            return tvm.make.node("MemoryInfo",
                                 unit_bits=32*8,
                                 max_simd_bits=32*8,
                                 max_num_bits=get_soc_spec("UB_SIZE")*8,
                                 head_address=tvm.const(0, 'int32'))
    except tvm._ffi.base.TVMError:
        raise RuntimeError("declare memory failed!")


def _allocate_ub(ib_expr, dtype, size, name):
    '''
    :param ib:
        desc: instance of ir_builder
    :param dtype:
        type: string
        desc: support data type: float16
    :param size:
        type: int
        desc: the size to allocate
    :param name:
        type: string
        desc: the name of allocated ub
    :return:
        desc: ub buffer
    '''
    # ib : instance of ir_builder
    # dtype: data type
    # size: buf size
    # name: buf name. Node suffix name must bu "local.UB"
    name = name + ".local.UB" + hex(UB_NAME_SUFFIX[0])
    scope = "local.UB" + hex(UB_NAME_SUFFIX[0])
    _decl_memory(scope)
    buf_var = ib_expr.allocate(dtype, (size,), name, scope=scope)
    return tvm.decl_buffer((size,), dtype, name, scope=scope, data=buf_var)


# pylint: disable=too-many-arguments
def _emit_copy_gm_ubuf(ib_expr, cmd, dtype, size, dst, dst_offset, src, src_offset):
    '''
    :param ib:
        desc: instance of ir_builder
    :param cmd:
        type: string
        desc: commond type, copy_gm_to_ubuf or copy_ubuf_to_gm
    :param dtype:
        type: string
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
    # emit copy_gm_to_ubuf, or emit copy_ubuf_to_gm.
    # ib : instance of ir_builder
    # cmd : string
    #     commond type, copy_gm_to_ubuf or copy_ubuf_to_gm
    n_burst = size // (2**16*32)
    tail = size % (2**16*32)
    burst_ele_num = 256 // intrin.get_bit_len(dtype)

    if tail > 0:
        dst_offset = dst_offset + n_burst*(2**16*32)
        src_offset = src_offset + n_burst*(2**16*32)
        len_burst = (tail + burst_ele_num - 1) // burst_ele_num
        sid = 0
        if cmd == "copy_gm_to_ubuf":
            sid = util.get_dma_sid("Sid_copy_gm_to_ubuf")
        ib_expr.emit(tvm.call_extern(
            dtype, cmd,
            dst.access_ptr("w", offset=dst_offset),
            src.access_ptr("rw", offset=src_offset),
            sid, 1, len_burst, 0, 0))


# pylint: disable=too-many-arguments
def _set_array_config(ib_expr, addr_array, addr_array_buf, src_array_stride,
                      dst_array_stride,
                      output_offset):
    '''
    :param ib:
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
    :return:
        None
    '''
    src0_offset = 8*0
    src1_offset = 8*1
    dst0_offset = 8*2
    dst1_offset = 8*3
    dtype_len = 2

    with ib_expr.for_range(0, 8, name="i") as i:
        ib_expr.emit(tvm.call_extern(
            "uint64", "reg_mov",
            tvm.call_extern(addr_array.dtype, "reg",
                            addr_array[src0_offset + i]),
            dtype_len*src_array_stride*i))

        ib_expr.emit(tvm.call_extern(
            "uint64", "reg_mov",
            tvm.call_extern(addr_array.dtype, "reg",
                            addr_array[src1_offset + i]),
            dtype_len*src_array_stride*(i + 8)))

        ib_expr.emit(tvm.call_extern(
            "uint64", "reg_mov",
            tvm.call_extern(addr_array.dtype, "reg",
                            addr_array[dst0_offset + i]),
            dtype_len*(output_offset + dst_array_stride*i)))

        ib_expr.emit(tvm.call_extern(
            "uint64", "reg_mov",
            tvm.call_extern(addr_array.dtype, "reg",
                            addr_array[dst1_offset + i]),
            dtype_len*(output_offset + dst_array_stride*(i + 8))))

    ib_expr.emit(tvm.call_extern(
        "int32", "set_va_reg_sb", "VA0",
        addr_array_buf.access_ptr("rw", offset=src0_offset)))

    ib_expr.emit(tvm.call_extern(
        "int32", "set_va_reg_sb", "VA1",
        addr_array_buf.access_ptr("rw", offset=src1_offset)))

    ib_expr.emit(tvm.call_extern(
        "int32", "set_va_reg_sb", "VA2",
        addr_array_buf.access_ptr("rw", offset=dst0_offset)))

    ib_expr.emit(tvm.call_extern(
        "int32", "set_va_reg_sb", "VA3",
        addr_array_buf.access_ptr("rw", offset=dst1_offset)))


def _get_vnchwconv_cube_buf_max(actual_col_size, is_four2five):
    '''
    :param actual_col_size:
        type: int
        desc: the size of actual colume (h*w)
    :param is_four2five:
        type: bool
        desc: operate type
    :return:
        type: int int
        desc: cube_size, tail_cube_size
    '''
    actual_cube_buf_size = ((actual_col_size - 1) // 16 + 1)*16*16*2
    ub_cut_upper_limit = get_soc_spec("UB_SIZE") // 2  # Byte
    if actual_cube_buf_size > ub_cut_upper_limit:
        if is_four2five:
            tail_cube_size = actual_cube_buf_size % ub_cut_upper_limit
            return (ub_cut_upper_limit // 2), (tail_cube_size // 2)  # fp16

        if actual_col_size % 16 != 0:
            # remain ubuf C0*C0*type_len(fp16) = 512 Byte
            ub_cut_upper_limit = (get_soc_spec("UB_SIZE") - 512) // 2  # Byte
            ub_cut_upper_limit = ub_cut_upper_limit // (2*16*16)*(2*16*16)
        tail_cube_size = actual_cube_buf_size % ub_cut_upper_limit
        return (ub_cut_upper_limit // 2), (tail_cube_size // 2)  # fp16

    return (actual_cube_buf_size // 2), 0  # fp16

# pylint: disable=too-many-locals, too-many-statements
def _four2five_ir(input_tensor, shape_4d, output):
    '''
    :param input_tensor:
         type: Tensor
         desc: input Tensor
    :param shape_4D:
        type: list
        desc: the raw shape fo Tensor (N, C, H, W)
    :param output:
        type: Tensor
        desc: output Tensor
    :return:
        the result statement.
    '''
    shape_n, shape_c, shape_h, shape_w = shape_4d
    shape_c0 = 16
    actual_col_size = shape_h*shape_w
    vnchwconv_cube_buf_max, tail_cube_size = _get_vnchwconv_cube_buf_max(
        actual_col_size, True)
    vnchwconv_cube_col_size = vnchwconv_cube_buf_max // shape_c0

    ib_expr = tvm.ir_builder.create()
    UB_NAME_SUFFIX[0] += 1
    input_ub = _allocate_ub(ib_expr, input_tensor.dtype, vnchwconv_cube_buf_max,
                            "input_ub")
    output_ub = _allocate_ub(ib_expr, output.dtype, vnchwconv_cube_buf_max,
                             "output_ub")
    addr_array = ib_expr.allocate("uint64", (32,), name="addr_array",
                                  scope=param.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=param.scope_reg,
                                     data=addr_array)

    src_array_stride = vnchwconv_cube_col_size
    dst_array_stride = shape_c0
    output_offset = vnchwconv_cube_buf_max
    _set_array_config(ib_expr, addr_array, addr_array_buf, src_array_stride,
                      dst_array_stride,
                      output_offset)

    c1_range = (shape_c + shape_c0 - 1) // shape_c0
    c1_loop = c1_range
    c1_tail = shape_c % shape_c0  # C1 tail
    if c1_tail != 0:
        c1_loop = c1_range - 1  # remove tail
    buf_cube_range = (actual_col_size + vnchwconv_cube_col_size - 1
                     ) // vnchwconv_cube_col_size

    def _four2five_intrin(ib_expr, c0_loop, c1_begin, c1_end):
        '''
        :param ib:
            type: tvm.ir_builder
            desc: tvm.ir_builder
        :param c0_loop:
            type: int
            desc: loop number of C0 axis
        :param c1_begin:
            type: int
            desc: start number of C1 axis
        :param c1_end:
            type: int
            desc: end number of C1 axis
        '''

        with ib_expr.for_range(0, shape_n, name="n") as idx_n:
            with ib_expr.for_range(0, c1_end - c1_begin, name="c1") as idx_c1:
                with ib_expr.for_range(0, buf_cube_range,
                                       name="buf_cube_index") as cube_i:
                    if buf_cube_range > 1:
                        # Standard processing:
                        # Copy gm to ub in 16 copies -> Update transpose config ->
                        # Transposed data --> Copy ub to gm at one-time
                        with ib_expr.if_scope(cube_i != buf_cube_range - 1):
                            repeat = vnchwconv_cube_col_size // 16
                            src_stride = 0 if repeat == 1 else 1
                            dst_stride = 0 if repeat == 1 else 16

                            with ib_expr.for_range(0, c0_loop, name="col") as col:
                                _emit_copy_gm_ubuf(
                                    ib_expr, "copy_gm_to_ubuf", input_tensor.dtype,
                                    vnchwconv_cube_col_size, input_ub,
                                    col*vnchwconv_cube_col_size, input_tensor,
                                    idx_n*shape_n*shape_n*shape_n +
                                    ((idx_c1 + c1_begin)*shape_c0 + col) *
                                    actual_col_size + cube_i *
                                    vnchwconv_cube_col_size)

                            ib_expr.emit(tvm.call_extern(
                                "int32",
                                "scatter_vnchwconv_b16",
                                "VA2",
                                "VA0",
                                repeat,
                                dst_stride,
                                src_stride))

                            _emit_copy_gm_ubuf(
                                ib_expr, 'copy_ubuf_to_gm', output.dtype,
                                vnchwconv_cube_buf_max, output,
                                idx_n*(c1_range*shape_c0)*shape_h *
                                shape_w + (idx_c1 + c1_begin)*shape_c0 *
                                actual_col_size + cube_i*vnchwconv_cube_buf_max,
                                output_ub, 0)
                        with ib_expr.else_scope():
                            # Tail processing:
                            # Copy gm to ub in 16 copies --> Update transpose config -->
                            # Transposed data --> Copy ub to gm at one-time
                            repeat = tail_cube_size // 16 // 16
                            src_stride = 0 if repeat == 1 else 1
                            dst_stride = 0 if repeat == 1 else 16

                            with ib_expr.for_range(0, c0_loop, name="col") as col:
                                _emit_copy_gm_ubuf(
                                    ib_expr, "copy_gm_to_ubuf",
                                    input_tensor.dtype, tail_cube_size // 16,
                                    input_ub, col*vnchwconv_cube_col_size,
                                    input_tensor, idx_n*shape_c*shape_h*shape_w
                                    + ((idx_c1 + c1_begin)*shape_c0 + col)*
                                    actual_col_size +
                                    cube_i*vnchwconv_cube_col_size)

                            ib_expr.emit(tvm.call_extern(
                                "int32",
                                "scatter_vnchwconv_b16",
                                "VA2",
                                "VA0",
                                repeat,
                                dst_stride,
                                src_stride))

                            _emit_copy_gm_ubuf(
                                ib_expr, 'copy_ubuf_to_gm', output.dtype,
                                (actual_col_size % vnchwconv_cube_col_size
                                )*shape_c0,
                                output, idx_n*(c1_range*shape_c0)*shape_h*
                                shape_w + (idx_c1 + c1_begin) *
                                shape_c0*actual_col_size +
                                cube_i*vnchwconv_cube_buf_max,
                                output_ub, 0)
                    else:
                        # Standard processing:
                        # Copy gm to ub in 16 copies --> Update transpose config -->
                        # Transposed data --> Copy ub to gm at one-time
                        repeat = vnchwconv_cube_col_size // 16
                        src_stride = 0 if repeat == 1 else 1
                        dst_stride = 0 if repeat == 1 else 16

                        with ib_expr.for_range(0, c0_loop, name="col") as col:
                            _emit_copy_gm_ubuf(
                                ib_expr, "copy_gm_to_ubuf", input_tensor.dtype,
                                vnchwconv_cube_col_size, input_ub,
                                col*vnchwconv_cube_col_size, input_tensor,
                                idx_n*shape_c*shape_h*shape_w +
                                ((idx_c1 + c1_begin)*shape_c0 + col) *
                                actual_col_size + cube_i *
                                vnchwconv_cube_col_size)

                        ib_expr.emit(tvm.call_extern(
                            "int32",
                            "scatter_vnchwconv_b16",
                            "VA2",
                            "VA0",
                            repeat,
                            dst_stride,
                            src_stride))

                        _emit_copy_gm_ubuf(
                            ib_expr, 'copy_ubuf_to_gm', output.dtype,
                            actual_col_size*shape_c0, output,
                            idx_n*(c1_range*shape_c0)*shape_h*shape_w +
                            (idx_c1 + c1_begin)*shape_c0*actual_col_size +
                            cube_i*vnchwconv_cube_buf_max,
                            output_ub, 0)

    _four2five_intrin(ib_expr, shape_c0, 0, c1_loop)
    if c1_tail != 0:  # c1 tail processing
        _four2five_intrin(ib_expr, c1_tail, c1_loop, c1_range)
    return ib_expr.get()

# pylint: disable=too-many-locals, too-many-statements
def _five2four_ir(input_tensor, shape_4d, output):
    '''
    :param input_tensor:
         type: Tensor
         desc: input Tensor
    :param shape_4D:
        type: list
        desc: the raw shape fo Tensor (N, C, H, W)
    :param output:
        type: Tensor
        desc: output Tensor
    :return:
        the result statement.
    '''
    shape_n, shape_c, shape_h, shape_w = shape_4d
    shape_c0 = 16
    actual_col_size = shape_h*shape_w
    vnchwconv_cube_buf_max, tail_cube_size = _get_vnchwconv_cube_buf_max(
        actual_col_size, False)
    vnchwconv_cube_col_size = vnchwconv_cube_buf_max // shape_c0

    ib_expr = tvm.ir_builder.create()
    UB_NAME_SUFFIX[0] += 1
    input_ub = _allocate_ub(ib_expr, input_tensor.dtype,
                            vnchwconv_cube_buf_max, "input_ub")
    output_ub = _allocate_ub(ib_expr, output.dtype, vnchwconv_cube_buf_max,
                             "output_ub")
    addr_array = ib_expr.allocate("uint64", (32,), name="addr_array",
                                  scope=param.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=param.scope_reg,
                                     data=addr_array)

    src_array_stride = shape_c0
    dst_array_stride = vnchwconv_cube_col_size
    output_offset = vnchwconv_cube_buf_max
    _set_array_config(ib_expr, addr_array, addr_array_buf, src_array_stride,
                      dst_array_stride,
                      output_offset)

    c1_range = (shape_c + shape_c0 - 1) // shape_c0
    c1_loop = c1_range
    c1_tail = shape_c % shape_c0  # C1 tail
    if c1_tail != 0:
        c1_loop = c1_range - 1  # remove tail
    buf_cube_range = (actual_col_size + vnchwconv_cube_col_size - 1
                     ) // vnchwconv_cube_col_size
    hw_tail_size = actual_col_size % vnchwconv_cube_col_size

    if hw_tail_size != 0 and hw_tail_size % shape_c0 != 0 and buf_cube_range > 1:
        output_head_ub = _allocate_ub(ib_expr, output.dtype, shape_c0*shape_c0,
                                      "output_head_ub")

    def _five2four_intrin(ib_expr, c0_loop, c1_begin, c1_end):
        """
        :param ib:
            type: tvm.ir_builder
            desc: tvm.ir_builder
        :param c0_loop:
            type: int
            desc: loop number of C0 axis
        :param c1_begin:
            type: int
            desc: start number of C1 axis
        :param c1_end:
            type: int
            desc: end number of C1 axis
        """
        with ib_expr.for_range(0, shape_n, name="n") as idx_n:
            with ib_expr.for_range(0, c1_end - c1_begin, name="c1") as idx_c1:
                with ib_expr.for_range(0, buf_cube_range,
                                       name="buf_cube_index") as cube_i:
                    if buf_cube_range > 1:
                        with ib_expr.if_scope(cube_i != buf_cube_range - 1):
                            # Standard processing:
                            # Copy gm to ub in one-time --> Update transpose config -->
                            # Transposed data --> Copy ub to gm in 16 copies
                            repeat = vnchwconv_cube_col_size // 16
                            src_stride = 0 if repeat == 1 else 16
                            dst_stride = 0 if repeat == 1 else 1

                            _emit_copy_gm_ubuf(
                                ib_expr, "copy_gm_to_ubuf", input_tensor.dtype,
                                vnchwconv_cube_buf_max, input_ub, 0,
                                input_tensor,
                                idx_n*c1_range*shape_c0*shape_h*shape_w
                                + (idx_c1 + c1_begin)*shape_c0 *
                                actual_col_size +
                                cube_i*vnchwconv_cube_buf_max)

                            ib_expr.emit(tvm.call_extern(
                                "int32",
                                "scatter_vnchwconv_b16",
                                "VA2",
                                "VA0",
                                repeat,
                                dst_stride,
                                src_stride))

                            with ib_expr.for_range(0, c0_loop, name="col") as col:
                                _emit_copy_gm_ubuf(
                                    ib_expr, 'copy_ubuf_to_gm', output.dtype,
                                    vnchwconv_cube_col_size, output,
                                    idx_n*shape_c*shape_h*shape_w +
                                    (idx_c1 + c1_begin)*shape_c0 *
                                    actual_col_size + col*actual_col_size +
                                    cube_i*vnchwconv_cube_col_size,
                                    output_ub, col*vnchwconv_cube_col_size)

                        with ib_expr.else_scope():
                            # Tail processing:
                            # Copy gm to ub in one-time --> Update transpose config -->
                            # Transposed data --> Copy ub to gm in 16 copies
                            repeat = tail_cube_size // 16 // 16
                            src_stride = 0 if repeat == 1 else 16
                            dst_stride = 0 if repeat == 1 else 1

                            _emit_copy_gm_ubuf(
                                ib_expr, "copy_gm_to_ubuf", input_tensor.dtype,
                                tail_cube_size, input_ub, 0, input_tensor,
                                idx_n*c1_range*shape_c0*shape_h*shape_w
                                + (idx_c1 + c1_begin)*shape_c0 *
                                actual_col_size +
                                cube_i*vnchwconv_cube_buf_max)

                            ib_expr.emit(tvm.call_extern(
                                "int32",
                                "scatter_vnchwconv_b16",
                                "VA2",
                                "VA0",
                                repeat,
                                dst_stride,
                                src_stride))

                            if hw_tail_size % shape_c0 == 0:
                                with ib_expr.for_range(0, c0_loop,
                                                       name="col") as col:
                                    _emit_copy_gm_ubuf(
                                        ib_expr, 'copy_ubuf_to_gm', output.dtype,
                                        hw_tail_size, output,
                                        idx_n*shape_c*shape_h*shape_w +
                                        (idx_c1 + c1_begin)*shape_c0 *
                                        actual_col_size + col *
                                        actual_col_size +
                                        cube_i*vnchwconv_cube_col_size,
                                        output_ub,
                                        col*vnchwconv_cube_col_size)
                            else:
                                # Exception processing: the actual colume size
                                # cann't be divided by 16
                                # 1. Copy head data (16*16) from gm for backup
                                # 2. Copy tail data from ub to gm;
                                # (because of padding data rewrite the head)
                                # 3. Copy head data (Step 1) to restore
                                with ib_expr.for_range(
                                        0, c0_loop, name="head_col_write") as \
                                        head_col_write:
                                    _emit_copy_gm_ubuf(
                                        ib_expr, "copy_gm_to_ubuf", output.dtype,
                                        shape_c0, output_head_ub,
                                        head_col_write*shape_c0, output,
                                        idx_n*shape_c*shape_h*shape_w +
                                        (idx_c1 + c1_begin)*shape_c0 *
                                        actual_col_size +
                                        head_col_write*actual_col_size)

                                with ib_expr.for_range(0, c0_loop,
                                                       name="col") as col:
                                    _emit_copy_gm_ubuf(
                                        ib_expr, 'copy_ubuf_to_gm', output.dtype,
                                        hw_tail_size, output,
                                        idx_n*shape_c*shape_h*shape_w +
                                        (idx_c1 + c1_begin)*shape_c0 *
                                        actual_col_size +
                                        col*actual_col_size +
                                        cube_i*vnchwconv_cube_col_size,
                                        output_ub,
                                        col*vnchwconv_cube_col_size)

                                with ib_expr.for_range(
                                        0, c0_loop, name="head_col_read") as \
                                        head_col_read:
                                    _emit_copy_gm_ubuf(
                                        ib_expr, 'copy_ubuf_to_gm', output.dtype,
                                        shape_c0, output,
                                        idx_n*shape_c*shape_h*shape_w +
                                        (idx_c1 + c1_begin)*shape_c0 *
                                        actual_col_size +
                                        head_col_read*actual_col_size,
                                        output_head_ub,
                                        head_col_read*shape_c0)
                    else:
                        # Standard processing:
                        # Copy gm to ub in one-time --> Update transpose config -->
                        # Transposed data --> Copy ub to gm in 16 copies
                        repeat = vnchwconv_cube_col_size // 16
                        src_stride = 0 if repeat == 1 else 16
                        dst_stride = 0 if repeat == 1 else 1

                        _emit_copy_gm_ubuf(
                            ib_expr, "copy_gm_to_ubuf", input_tensor.dtype,
                            vnchwconv_cube_buf_max, input_ub, 0, input_tensor,
                            idx_n*c1_range*shape_c0*shape_h*shape_w +
                            (idx_c1 + c1_begin)*shape_c0*actual_col_size +
                            cube_i*vnchwconv_cube_buf_max)

                        ib_expr.emit(tvm.call_extern(
                            "int32",
                            "scatter_vnchwconv_b16",
                            "VA2",
                            "VA0",
                            repeat,
                            dst_stride,
                            src_stride))

                        if hw_tail_size == 0:
                            with ib_expr.for_range(0, c0_loop, name="col") as col:
                                _emit_copy_gm_ubuf(
                                    ib_expr, 'copy_ubuf_to_gm', output.dtype,
                                    vnchwconv_cube_col_size, output,
                                    idx_n*shape_c*shape_h*shape_w +
                                    (idx_c1 + c1_begin)*shape_c0 *
                                    actual_col_size + col*actual_col_size,
                                    output_ub, col*vnchwconv_cube_col_size)
                        else:
                            with ib_expr.for_range(0, c0_loop, name="col") as col:
                                with ib_expr.new_scope():
                                    ib_expr.scope_attr(param.CCE_AXIS,
                                                       "coproc_scope", 6)
                                    _emit_copy_gm_ubuf(
                                        ib_expr, 'copy_ubuf_to_gm',
                                        output.dtype,
                                        hw_tail_size, output,
                                        idx_n*shape_c*shape_h*shape_w +
                                        (idx_c1 + c1_begin)*shape_c0 *
                                        actual_col_size + col*actual_col_size,
                                        output_ub,
                                        col*vnchwconv_cube_col_size)

    _five2four_intrin(ib_expr, shape_c0, 0, c1_loop)
    if c1_tail != 0:  # c1 tail processing
        _five2four_intrin(ib_expr, c1_tail, c1_loop, c1_range)
    return ib_expr.get()
