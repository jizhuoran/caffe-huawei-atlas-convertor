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

aipp_dynmaic
"""

# pylint: disable=import-error,too-many-lines,invalid-name,too-many-locals
# pylint: disable=too-many-branches,too-many-statements,too-many-arguments

from te import tvm
from te import platform as cce
from te.platform.cce_util import get_const
from impl import aipp_comm


def get_dync_padding_size(ib, batch_id, p_ub_buf, tmp, padding_info):
    """
    :param aipp_config:
    :return:
    """

    ib.emit(tvm.call_extern("int8",  # actual data type
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            p_ub_buf.access_ptr('r', offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE + \
                                                            aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id + \
                                                            aipp_comm.BATCH_OFFSET_PAD_SWITCH)))

    with ib.if_scope(tmp[0]):
        ib.emit(tvm.call_extern("int32",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE + \
                                                                aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id + \
                                                                aipp_comm.BATCH_OFFSET_PAD_TOP)))
        padding_info[0] = tmp[0]

        ib.emit(tvm.call_extern("int32",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE + \
                                                                aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id + \
                                                                aipp_comm.BATCH_OFFSET_PAD_BOTTOM)))
        padding_info[1] = tmp[0]

        ib.emit(tvm.call_extern("int32",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE + \
                                                                aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id + \
                                                                aipp_comm.BATCH_OFFSET_PAD_LEFT)))
        padding_info[2] = tmp[0]

        ib.emit(tvm.call_extern("int32",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE + \
                                                                aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id + \
                                                                aipp_comm.BATCH_OFFSET_PAD_RIGHT)))
        padding_info[3] = tmp[0]


def move_data_from_l1_to_gm(ib, totol_num, dtype, output_cb_buf, output_buf, gm_output_offset):
    """
    :param ib:
    :param totol_num:
    :param dtype:
    :param output_cb_buf:
    :param output_buf:
    :param gm_output_offset:
    :return:
    """

    ub_size = cce.CceProductParams().getParams("Unified_Buffer")
    ub_size -= (aipp_comm.DYNC_PARAM_SIZE + 1024 - 1) // 1024 * 1024

    if dtype == "float16":
        size = 2
    else:
        size = 1

    if totol_num*size < (ub_size // 2):
        with ib.new_scope():
            output_ub = ib.allocate(dtype, (totol_num*size,),
                                    "output_ub",
                                    scope=cce.scope_ubuf)
            output_ub_buf = tvm.decl_buffer(
                (totol_num*size,), dtype, "output_ub_buf",
                scope=cce.scope_ubuf, data=output_ub)

            lenBurst, nBurst = \
                aipp_comm.get_lenBurst_and_nBurst(
                    totol_num*size//32, 1)

            ib.emit(tvm.call_extern(
                dtype, 'copy_cbuf_to_ubuf',
                output_ub_buf.access_ptr("w",
                                         ptr_type=dtype,
                                         offset=0),
                output_cb_buf.access_ptr("rw",
                                         ptr_type=dtype,
                                         offset=0),
                0, nBurst, lenBurst, 0, 0))

            ib.emit(tvm.call_extern(
                dtype, 'copy_ubuf_to_gm',
                output_buf.access_ptr(
                    "w", ptr_type=dtype,
                    offset=gm_output_offset),
                output_ub_buf.access_ptr("rw", ptr_type=dtype,
                                         offset=0),
                0, nBurst, lenBurst, 0, 0))
    else:
        move_num = totol_num // ((ub_size // 2) // size)
        move_tail = totol_num - move_num*((ub_size // 2) // size)

        with ib.for_range(tvm.const(0, dtype="uint64"),
                          tvm.const(move_num,
                                    dtype="uint64"),
                          name="move_index",
                          dtype="uint64") as move_index:
            with ib.new_scope():
                output_ub = ib.allocate(dtype, (ub_size // 2,),
                                        "output_ub",
                                        scope=cce.scope_ubuf)
                output_ub_buf = tvm.decl_buffer(
                    (ub_size // 2,), dtype, "output_ub_buf",
                    scope=cce.scope_ubuf, data=output_ub)

                lenBurst, nBurst = \
                    aipp_comm.get_lenBurst_and_nBurst(
                        (ub_size // 2) // 32, 1)

                ib.emit(tvm.call_extern(
                    dtype, 'copy_cbuf_to_ubuf',
                    output_ub_buf.access_ptr("w", ptr_type=dtype,
                                             offset=0),
                    output_cb_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=move_index*((ub_size // 2) // size)),
                    0, nBurst, lenBurst, 0, 0))

                ib.emit(tvm.call_extern(
                    dtype, 'copy_ubuf_to_gm',
                    output_buf.access_ptr("w", ptr_type=dtype,
                                          offset=gm_output_offset +
                                          move_index*((ub_size // 2) // size)),
                    output_ub_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=0),
                    0, nBurst, lenBurst, 0, 0))
        if move_tail != 0:
            with ib.new_scope():
                output_ub = ib.allocate(dtype, (ub_size // 2,),
                                        "output_ub",
                                        scope=cce.scope_ubuf)
                output_ub_buf = tvm.decl_buffer(
                    (ub_size // 2,), dtype, "output_ub_buf",
                    scope=cce.scope_ubuf, data=output_ub)

                lenBurst, nBurst = \
                    aipp_comm.get_lenBurst_and_nBurst(
                        move_tail*size // 32, 1)

                ib.emit(tvm.call_extern(
                    dtype, 'copy_cbuf_to_ubuf',
                    output_ub_buf.access_ptr("w", ptr_type=dtype,
                                             offset=0),
                    output_cb_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=move_num*((ub_size // 2) // size)),
                    0, nBurst, lenBurst, 0, 0))

                ib.emit(tvm.call_extern(
                    dtype, 'copy_ubuf_to_gm',
                    output_buf.access_ptr("w", ptr_type=dtype,
                                          offset=gm_output_offset +
                                          move_num*((ub_size // 2) // size)),
                    output_ub_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=0),
                    0, nBurst, lenBurst, 0, 0))


def reg_move_data_from_l1_to_gm(ib, tmp_reg, totol_num, dtype, output_cb_buf, output_buf, gm_output_offset):
    """
    :param ib:
    :param tmp_reg:
    :param totol_num:
    :param dtype:
    :param output_cb_buf:
    :param output_buf:
    :param gm_output_offset:
    :return:
    """

    ub_size = cce.CceProductParams().getParams("Unified_Buffer")
    ub_size -= (aipp_comm.DYNC_PARAM_SIZE + 1024 - 1) // 1024 * 1024

    if dtype == "float16":
        size = 2
    else:
        size = 1

    tmp_reg[0] = get_const(totol_num)*tvm.const(size, dtype="uint64")
    with ib.if_scope(tmp_reg[0] < (ub_size // 2)):
        with ib.new_scope():
            output_ub = ib.allocate(dtype, (ub_size // 2,),
                                    "output_ub",
                                    scope=cce.scope_ubuf)
            output_ub_buf = tvm.decl_buffer(
                (ub_size // 2,), dtype, "output_ub_buf",
                scope=cce.scope_ubuf, data=output_ub)

            ib.emit(tvm.call_extern(
                dtype, 'copy_cbuf_to_ubuf',
                output_ub_buf.access_ptr("w",
                                         ptr_type=dtype,
                                         offset=0),
                output_cb_buf.access_ptr("rw",
                                         ptr_type=dtype,
                                         offset=0),
                0, 1, totol_num*size//32, 0, 0))

            ib.emit(tvm.call_extern(
                dtype, 'copy_ubuf_to_gm',
                output_buf.access_ptr(
                    "w", ptr_type=dtype,
                    offset=gm_output_offset),
                output_ub_buf.access_ptr("rw", ptr_type=dtype,
                                         offset=0),
                0, 1, totol_num*size//32, 0, 0))
    with ib.else_scope():
        # move_num
        tmp_reg[1] = get_const(totol_num // ((ub_size // 2) // size))
        # move_tail
        tmp_reg[2] = get_const(totol_num % ((ub_size // 2) // size))

        with ib.for_range(tvm.const(0, dtype="uint64"),
                          tmp_reg[1],
                          name="move_index",
                          dtype="uint64") as move_index:
            with ib.new_scope():
                output_ub = ib.allocate(dtype, (ub_size // 2,),
                                        "output_ub",
                                        scope=cce.scope_ubuf)
                output_ub_buf = tvm.decl_buffer(
                    (ub_size // 2,), dtype, "output_ub_buf",
                    scope=cce.scope_ubuf, data=output_ub)

                lenBurst, nBurst = \
                    aipp_comm.get_lenBurst_and_nBurst(
                        (ub_size // 2) // 32, 1)

                ib.emit(tvm.call_extern(
                    dtype, 'copy_cbuf_to_ubuf',
                    output_ub_buf.access_ptr("w", ptr_type=dtype,
                                             offset=0),
                    output_cb_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=move_index*((ub_size // 2) // size)),
                    0, nBurst, lenBurst, 0, 0))

                ib.emit(tvm.call_extern(
                    dtype, 'copy_ubuf_to_gm',
                    output_buf.access_ptr("w", ptr_type=dtype,
                                          offset=gm_output_offset +
                                          move_index*((ub_size // 2) // size)),
                    output_ub_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=0),
                    0, nBurst, lenBurst, 0, 0))
        with ib.if_scope(tmp_reg[2] != 0):
            with ib.new_scope():
                output_ub = ib.allocate(dtype, (ub_size // 2,),
                                        "output_ub",
                                        scope=cce.scope_ubuf)
                output_ub_buf = tvm.decl_buffer(
                    (ub_size // 2,), dtype, "output_ub_buf",
                    scope=cce.scope_ubuf, data=output_ub)

                ib.emit(tvm.call_extern(
                    dtype, 'copy_cbuf_to_ubuf',
                    output_ub_buf.access_ptr("w", ptr_type=dtype,
                                             offset=0),
                    output_cb_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=tmp_reg[1]*((ub_size // 2) // size)),
                    0, 1, tmp_reg[2]*size//32, 0, 0))

                ib.emit(tvm.call_extern(
                    dtype, 'copy_ubuf_to_gm',
                    output_buf.access_ptr("w", ptr_type=dtype,
                                          offset=gm_output_offset +
                                          tmp_reg[1]*((ub_size // 2) // size)),
                    output_ub_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=0),
                    0, 1, tmp_reg[2]*size//32, 0, 0))


def process_padding(ib, input_data, output_buf):
    """
    :param ib:
    :param input_data:
    :param output_buf:
    :return:
    """

    dtype = input_data[0]
    W = input_data[1]
    C0 = input_data[2]
    size = input_data[3]
    padding_size = input_data[4]
    offset = input_data[5]

    ub_size = cce.CceProductParams().getParams("Unified_Buffer")
    ub_size -= (aipp_comm.DYNC_PARAM_SIZE + 1024 - 1) // 1024*1024
    buffer_upper_limit = ub_size // 2 // 2 // C0

    with ib.if_scope(buffer_upper_limit >= padding_size*W):
        with ib.new_scope():
            padding_ub = ib.allocate(dtype, (buffer_upper_limit*C0,), "padding_ub",
                                     scope=cce.scope_ubuf)
            padding_ub_buf = tvm.decl_buffer((buffer_upper_limit*C0,), dtype,
                                             "padding_ub", scope=cce.scope_ubuf,
                                             data=padding_ub)
            if dtype == "float16":
                aipp_comm.vector_dup(ib, "float16", padding_ub_buf,
                                     padding_size*W*C0)
            else:
                with ib.new_scope():
                    tmp_padding_ub = ib.allocate("float16",
                                                 (buffer_upper_limit*C0,),
                                                 "tmp_padding_ub",
                                                 scope=cce.scope_ubuf)
                    tmp_padding_ub_buf = tvm.decl_buffer((buffer_upper_limit*C0,),
                                                         "float16",
                                                         "tmp_padding_ub_buf",
                                                         scope=cce.scope_ubuf,
                                                         data=tmp_padding_ub)

                    aipp_comm.vector_dup(ib, "float16", tmp_padding_ub_buf,
                                         padding_size*W*C0)
                    aipp_comm.conv(ib, dtype, "float16", padding_ub_buf,
                                   tmp_padding_ub_buf, padding_size*W*C0)

            ib.emit(tvm.call_extern(
                dtype, 'copy_ubuf_to_gm',
                output_buf.access_ptr("w", ptr_type=dtype, offset=offset),
                padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                0, 1, padding_size*W*C0*size//32, 0, 0))
    with ib.else_scope():
        tiling_w, w_loop = aipp_comm.get_tiling_W(W, buffer_upper_limit, 1)
        tiling_h = buffer_upper_limit // tiling_w

        h_loop = padding_size // tiling_h
        zero_const = tvm.const(0, dtype="uint64")
        h_loop_const = get_const(h_loop)
        w_loop_const = tvm.const(w_loop, dtype="uint64")

        tail_w = W % tiling_w
        tail_h = padding_size % tiling_h

        with ib.for_range(zero_const, h_loop_const, name="h1",
                          dtype="uint64") as h1:
            with ib.for_range(zero_const, w_loop_const,
                              name="w1", dtype="uint64") as w1:
                with ib.new_scope():
                    padding_ub = ib.allocate(dtype, (tiling_h*tiling_w*C0,), "padding_ub",
                                             scope=cce.scope_ubuf)
                    padding_ub_buf = tvm.decl_buffer((tiling_h*tiling_w*C0,), dtype,
                                                     "padding_ub",
                                                     scope=cce.scope_ubuf,
                                                     data=padding_ub)

                    if dtype == "float16":
                        aipp_comm.vector_dup(ib, "float16", padding_ub_buf,
                                             tiling_h*tiling_w*C0)
                    else:
                        with ib.new_scope():
                            tmp_padding_ub = ib.allocate("float16",
                                                         (tiling_h*tiling_w*C0,),
                                                         "tmp_padding_ub",
                                                         scope=cce.scope_ubuf)
                            tmp_padding_ub_buf = tvm.decl_buffer((tiling_h*tiling_w*C0,),
                                                                 "float16",
                                                                 "tmp_padding_ub_buf",
                                                                 scope=cce.scope_ubuf,
                                                                 data=tmp_padding_ub)

                            aipp_comm.vector_dup(ib, "float16", tmp_padding_ub_buf,
                                                 tiling_h*tiling_w*C0)
                            aipp_comm.conv(ib, dtype, "float16", padding_ub_buf,
                                           tmp_padding_ub_buf, tiling_h*tiling_w*C0)

                    ib.emit(tvm.call_extern(
                        dtype, 'copy_ubuf_to_gm',
                        output_buf.access_ptr("w", ptr_type=dtype,
                                              offset=offset+h1*tiling_h*W*C0+w1*tiling_w*tiling_h*C0),
                        padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                        0, 1, tiling_h*tiling_w*C0*size//32, 0, 0))
            if tail_w != 0:
                with ib.new_scope():
                    padding_ub = ib.allocate(dtype, (tiling_h*tail_w*C0,), "padding_ub",
                                             scope=cce.scope_ubuf)
                    padding_ub_buf = tvm.decl_buffer((tiling_h*tail_w*C0,), dtype,
                                                     "padding_ub",
                                                     scope=cce.scope_ubuf,
                                                     data=padding_ub)

                    if dtype == "float16":
                        aipp_comm.vector_dup(ib, "float16", padding_ub_buf,
                                             tiling_h*tail_w*C0)
                    else:
                        with ib.new_scope():
                            tmp_padding_ub = ib.allocate("float16",
                                                         (tiling_h*tail_w*C0,),
                                                         "tmp_padding_ub",
                                                         scope=cce.scope_ubuf)
                            tmp_padding_ub_buf = tvm.decl_buffer((tiling_h*tail_w*C0,),
                                                                 "float16",
                                                                 "tmp_padding_ub_buf",
                                                                 scope=cce.scope_ubuf,
                                                                 data=tmp_padding_ub)

                            aipp_comm.vector_dup(ib, "float16", tmp_padding_ub_buf,
                                                 tiling_h*tail_w*C0)
                            aipp_comm.conv(ib, dtype, "float16", padding_ub_buf,
                                           tmp_padding_ub_buf, tiling_h*tail_w*C0)

                    ib.emit(tvm.call_extern(
                        dtype, 'copy_ubuf_to_gm',
                        output_buf.access_ptr("w", ptr_type=dtype,
                                              offset=offset+h1*tiling_h*tiling_w*C0+w_loop*tiling_w*tiling_h*C0),
                        padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                        0, 1, tiling_h*tail_w*C0*size//32, 0, 0))

        with ib.if_scope(tail_h != 0):
            with ib.for_range(zero_const,
                              w_loop_const,
                              name="w1", dtype="uint64") as w1:
                with ib.new_scope():
                    padding_ub = ib.allocate(dtype, (tiling_h*tiling_w*C0,),
                                             "padding_ub", scope=cce.scope_ubuf)
                    padding_ub_buf = tvm.decl_buffer((tiling_h*tiling_w*C0,), dtype,
                                                     "padding_ub",
                                                     scope=cce.scope_ubuf,
                                                     data=padding_ub)

                    if dtype == "float16":
                        aipp_comm.vector_dup(ib, "float16", padding_ub_buf,
                                             tail_h*tiling_w*C0)
                    else:
                        with ib.new_scope():
                            tmp_padding_ub = ib.allocate("float16", (tiling_h*tiling_w*C0,),
                                                         "tmp_padding_ub",
                                                         scope=cce.scope_ubuf)
                            tmp_padding_ub_buf = tvm.decl_buffer((tiling_h*tiling_w*C0,),
                                                                 "float16",
                                                                 "tmp_padding_ub_buf",
                                                                 scope=cce.scope_ubuf,
                                                                 data=tmp_padding_ub)

                            aipp_comm.vector_dup(ib, "float16", tmp_padding_ub_buf,
                                                 tail_h*tiling_w*C0)
                            aipp_comm.conv(ib, dtype, "float16", padding_ub_buf,
                                           tmp_padding_ub_buf, tail_h*tiling_w*C0)

                    ib.emit(tvm.call_extern(
                        dtype, 'copy_ubuf_to_gm',
                        output_buf.access_ptr("w", ptr_type=dtype,
                                              offset=offset+h_loop*tiling_h*W*C0+w1*tiling_w*tail_h*C0),
                        padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                        0, 1, tail_h*tiling_w*C0*size//32, 0, 0))

            if tail_w != 0:
                with ib.new_scope():
                    padding_ub = ib.allocate(dtype, (tiling_h*tail_w*C0,), "padding_ub",
                                             scope=cce.scope_ubuf)
                    padding_ub_buf = tvm.decl_buffer((tiling_h*tail_w*C0,), dtype,
                                                     "padding_ub",
                                                     scope=cce.scope_ubuf,
                                                     data=padding_ub)

                    if dtype == "float16":
                        aipp_comm.vector_dup(ib, "float16", padding_ub_buf,
                                             tail_h*tail_w*C0)
                    else:
                        with ib.new_scope():
                            tmp_padding_ub = ib.allocate("float16",
                                                         (tiling_h*tail_w*C0,),
                                                         "tmp_padding_ub",
                                                         scope=cce.scope_ubuf)
                            tmp_padding_ub_buf = tvm.decl_buffer((tiling_h*tail_w*C0,),
                                                                 "float16",
                                                                 "tmp_padding_ub_buf",
                                                                 scope=cce.scope_ubuf,
                                                                 data=tmp_padding_ub)

                            aipp_comm.vector_dup(ib, "float16", tmp_padding_ub_buf,
                                                 tail_h*tail_w*C0)
                            aipp_comm.conv(ib, dtype, "float16", padding_ub_buf,
                                           tmp_padding_ub_buf, tail_h*tail_w*C0)

                    ib.emit(tvm.call_extern(
                        dtype, 'copy_ubuf_to_gm',
                        output_buf.access_ptr("w", ptr_type=dtype,
                                              offset=offset+h_loop*tiling_h*W*C0+w_loop*tiling_w*tail_h*C0),
                        padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                        0, 1, tail_h*tail_w*C0*size//32, 0, 0))


def get_dync_src_image_size(ib, p_ub_buf, tmp, src_image_size):
    """
    :param ib:
    :param p_ub_buf:
    :param tmp:
    :param src_image_size:
    :return:
    """

    ib.emit(tvm.call_extern("int32",  # actual data type
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            p_ub_buf.access_ptr('r', offset=aipp_comm.HEAD_OFFSET_SRCIMAGE_H)))
    with ib.if_scope(tmp[0] > 0):
        src_image_size[0] = tmp[0]

    ib.emit(tvm.call_extern("int32",  # actual data type
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            p_ub_buf.access_ptr('r', offset=aipp_comm.HEAD_OFFSET_SRCIMAGE_W)))
    with ib.if_scope(tmp[0] > 0):
        src_image_size[1] = tmp[0]


def get_dync_crop_info(ib, batch_id, p_ub_buf, tmp, load_image_info):
    """
    :param ib:
    :param batch_id:
    :param p_ub_buf:
    :param tmp:
    :param load_image_info:
    :return:
    """

    ib.emit(tvm.call_extern("int32",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            p_ub_buf.access_ptr('r', offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE + \
                                                            aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id + \
                                                            aipp_comm.BATCH_OFFSET_CROP_STARTPOS_W)))
    # load_start_pos_w
    load_image_info[1] = tmp[0]

    ib.emit(tvm.call_extern("int32",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            p_ub_buf.access_ptr('r', offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE + \
                                                            aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id + \
                                                            aipp_comm.BATCH_OFFSET_CROP_STARTPOS_H)))
    # load_start_pos_h
    load_image_info[0] = tmp[0]

    ib.emit(tvm.call_extern("int32",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            p_ub_buf.access_ptr('r', offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE + \
                                                            aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id + \
                                                            aipp_comm.BATCH_OFFSET_CROP_W)))
    # load_image_w
    load_image_info[3] = tmp[0]

    ib.emit(tvm.call_extern("int32",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            p_ub_buf.access_ptr('r', offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE + \
                                                            aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id + \
                                                            aipp_comm.BATCH_OFFSET_CROP_H)))
    # load_image_h
    load_image_info[2] = tmp[0]


def aipp_compute(input_tensor, param_tensor, input_shape, input_format, output_data):
    """
    :param input_tensor:
    :param param_tensor:
    :param input_shape:
    :param input_format:
    :param output_data:
    :return:
    """

    output_shape = output_data.get('shape')
    N, C1, H, W, C0 = output_shape

    dtype = output_data.get('dtype')
    if dtype == "float16":
        size = 2
        C0 = 16
    else:
        size = 1
        C0 = 32

    actual_col_size = H * W

    l1_image_buf_max = aipp_comm.get_l1_image_buf_max(actual_col_size, dtype, True)

    def aipp_ir(input_buf, param_buf, output_buf):
        ib = tvm.ir_builder.create()
        cur_cce_product = cce.cce_conf.CceProductParams().cce_product

        device_core_num = \
            cce.cce_conf.CceProductParams().getParams("Device_core_num")
        batch_num = N
        batch_factor = 1

        if batch_num % device_core_num == 0:
            batch_factor = batch_num // device_core_num

        block_index = tvm.thread_axis("blockIdx.x")
        ib.scope_attr(block_index, "thread_extent", batch_num // batch_factor)

        offset = batch_factor * C1 * H * W * C0

        zero_const = tvm.const(0, dtype="uint64")

        def _aipp_intrin():
            # config SPR2~SPR9
            p_ub_buf, spr, tmp = aipp_comm.set_spr_dync(ib, param_buf, dtype, cur_cce_product)

            src_image_size = ib.allocate("uint64", [2], name="src_image_size", scope=cce.scope_reg)
            src_image_size[0] = tvm.const(H, dtype="uint64")
            src_image_size[1] = tvm.const(W, dtype="uint64")
            get_dync_src_image_size(ib, p_ub_buf, tmp, src_image_size)

            actual_col_size_reg = ib.allocate("uint64", [1], name="actual_col_size_reg", scope=cce.scope_reg)
            actual_col_size_reg[0] = src_image_size[0] * src_image_size[1]

            load_image_info = ib.allocate("uint64", [4], name="load_image_info", scope=cce.scope_reg)

            crop = ib.allocate("uint64", [1], name="crop", scope=cce.scope_reg)
            crop[0] = tvm.const(0, dtype="uint64")

            padding_info = ib.allocate("uint64", [4], name="padding_info", scope=cce.scope_reg)

            padding = ib.allocate("uint64", [1], name="padding", scope=cce.scope_reg)
            padding[0] = tvm.const(0, dtype="uint64")

            input_format = ib.allocate("uint64", [1], name="input_format", scope=cce.scope_reg)
            input_format[0] = tvm.const(0, dtype="uint64")

            h_loop = ib.allocate("uint64", [1], name="h_loop", scope=cce.scope_reg)
            h_loop[0] = tvm.const(1, dtype="uint64")

            Xs = ib.allocate("uint64", [1], name="Xs", scope=cce.scope_reg)
            Xs[0] = tvm.const(0, dtype="uint64")

            load_h = ib.allocate("uint64", [1], name="load_h", scope=cce.scope_reg)
            load_h[0] = tvm.const(0, dtype="uint64")

            tail_h = ib.allocate("uint64", [1], name="tail_h", scope=cce.scope_reg)
            tail_h[0] = tvm.const(0, dtype="uint64")

            tail_h_postion = ib.allocate("uint64", [1], name="tail_h_postion", scope=cce.scope_reg)
            tail_h_postion[0] = tvm.const(0, dtype="uint64")

            load_tail_h = ib.allocate("uint64", [1], name="load_tail_h", scope=cce.scope_reg)
            load_tail_h[0] = tvm.const(0, dtype="uint64")

            tmp_reg = ib.allocate("uint64", [3], name="tail_h_tmp_reg", scope=cce.scope_reg)

            resize = ib.allocate("uint64", [1], name="resize", scope=cce.scope_reg)

            scfIncVscl = ib.allocate("uint64", [1], name="scfIncVscl", scope=cce.scope_reg)
            scfIncHscl = ib.allocate("uint64", [1], name="scfIncHscl", scope=cce.scope_reg)
            resize_input_h = ib.allocate("uint64", [1], name="resize_input_h", scope=cce.scope_reg)
            resize_input_w = ib.allocate("uint64", [1], name="resize_input_w", scope=cce.scope_reg)
            resize_output_h = ib.allocate("uint64", [1], name="resize_output_h", scope=cce.scope_reg)
            resize_output_w = ib.allocate("uint64", [1], name="resize_output_w", scope=cce.scope_reg)
            resize_input_h_stat_pos = ib.allocate("uint64", [1], name="resize_input_h_stat_pos", scope=cce.scope_reg)
            resize_input_h_end_pos = ib.allocate("uint64", [1], name="resize_input_h_end_pos", scope=cce.scope_reg)

            with ib.for_range(zero_const,
                              tvm.const(batch_factor, dtype="uint64"),
                              name="n1", dtype="uint64") as n1:
                batch_id = batch_factor * block_index + n1

                actual_col_size_reg[0] = src_image_size[0] * src_image_size[1]

                # load_start_pos_h
                load_image_info[0] = tvm.const(0, dtype="uint64")
                # load_start_pos_w
                load_image_info[1] = tvm.const(0, dtype="uint64")
                # load_image_h
                load_image_info[2] = src_image_size[0]
                # load_image_w
                load_image_info[3] = src_image_size[1]

                # top_padding_size
                padding_info[0] = tvm.const(0, dtype="uint64")
                # bottom_padding_size
                padding_info[1] = tvm.const(0, dtype="uint64")
                # left_padding_size
                padding_info[2] = tvm.const(0, dtype="uint64")
                # right_padding_size
                padding_info[3] = tvm.const(0, dtype="uint64")


                ib.emit(tvm.call_extern("int8",  # actual data type
                                        "reg_mov",
                                        tvm.call_extern("uint64", "reg", tmp[0]),
                                        p_ub_buf.access_ptr('r',
                                                            offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE + \
                                                                   aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id + \
                                                                   aipp_comm.BATCH_OFFSET_CROP_SWITCH)))
                crop[0] = tmp[0]
                #crop enable
                with ib.if_scope(tmp[0] > 0):
                    get_dync_crop_info(ib, batch_id, p_ub_buf, tmp, load_image_info)

                    actual_col_size_reg[0] = load_image_info[2] * load_image_info[3]


                aippXt = (src_image_size[1] - 1)

                ib.emit(tvm.call_extern("uint8",  # actual data type
                                        "reg_mov",
                                        tvm.call_extern("uint64", "reg", tmp[0]),
                                        p_ub_buf.access_ptr('r', offset=aipp_comm.HEAD_OFFSET_INPUT_FORMAT)))

                spr0 = 0
                uv_addr = 0
                input_format[0] = tmp[0]
                # YUV420SP_U8
                with ib.if_scope(input_format[0] == 1):
                    input_offset = \
                        batch_factor * ((3 * src_image_size[0] * src_image_size[1]) // 2)
                    spr0 = get_const(
                        input_buf.access_ptr('r',
                                             offset=block_index*input_offset +
                                             n1*((3*src_image_size[0] * src_image_size[1])//2)))
                    uv_addr = get_const(
                        input_buf.access_ptr('r',
                                             offset=block_index*input_offset +
                                             n1*((3*src_image_size[0] * src_image_size[1])//2) +
                                             src_image_size[0] * src_image_size[1]))
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_0", spr0))

                    ib.emit(tvm.call_extern("int8",  # actual data type
                                            "reg_mov",
                                            tvm.call_extern("uint64", "reg", tmp[0]),
                                            p_ub_buf.access_ptr('r', offset=aipp_comm.HEAD_OFFSET_CSC_SWITCH)))
                    if cur_cce_product in ['5.10']:
                        spr[1] = tvm.const(1 << 63, dtype="uint64") | uv_addr
                    else:
                        spr[1] = (tmp[0] & 0x1) << 63 | uv_addr
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_1", spr[1]))
                # XRGB8888_U8
                with ib.if_scope(input_format[0] == 2):
                    input_offset = \
                        batch_factor * ((4 * src_image_size[0] * src_image_size[1]))
                    spr0 = get_const(
                        input_buf.access_ptr('r',
                                             offset=block_index*input_offset +
                                             n1*((4*src_image_size[0] * src_image_size[1]))))
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_0", spr0))

                    ib.emit(tvm.call_extern("int8",  # actual data type
                                            "reg_mov",
                                            tvm.call_extern("uint64", "reg", tmp[0]),
                                            p_ub_buf.access_ptr('r', offset=aipp_comm.HEAD_OFFSET_CSC_SWITCH)))
                    spr[1] = (tmp[0] & 0x1) << 63
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_1", spr[1]))
                # RGB888_U8
                with ib.if_scope(input_format[0] == 5):
                    input_offset = \
                        batch_factor * ((3 * src_image_size[0] * src_image_size[1]))
                    spr0 = get_const(
                        input_buf.access_ptr('r',
                                             offset=block_index*input_offset +
                                             n1*((3*src_image_size[0] * src_image_size[1]))))
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_0", spr0))

                    ib.emit(tvm.call_extern("int8",  # actual data type
                                            "reg_mov",
                                            tvm.call_extern("uint64", "reg", tmp[0]),
                                            p_ub_buf.access_ptr('r', offset=aipp_comm.HEAD_OFFSET_CSC_SWITCH)))
                    spr[1] = (tmp[0] & 0x1) << 63
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_1", spr[1]))
                # ARGB8888
                with ib.if_scope(input_format[0] == 6):
                    input_offset = \
                        batch_factor * ((4 * src_image_size[0] * src_image_size[1]))
                    spr0 = get_const(
                        input_buf.access_ptr('r',
                                             offset=block_index*input_offset +
                                             n1*((4*src_image_size[0] * src_image_size[1]))))
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_0", spr0))

                    ib.emit(tvm.call_extern("int8",  # actual data type
                                            "reg_mov",
                                            tvm.call_extern("uint64", "reg", tmp[0]),
                                            p_ub_buf.access_ptr('r', offset=aipp_comm.HEAD_OFFSET_CSC_SWITCH)))
                    spr[1] = (tmp[0] & 0x1) << 63
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_1", spr[1]))
                # YUYV_U8
                with ib.if_scope(input_format[0] == 7):
                    input_offset = \
                        batch_factor * ((2 * src_image_size[0] * src_image_size[1]))
                    spr0 = get_const(
                        input_buf.access_ptr('r',
                                             offset=block_index*input_offset +
                                             n1*((2*src_image_size[0] * src_image_size[1]))))

                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_0", spr0))

                    ib.emit(tvm.call_extern("int8",  # actual data type
                                            "reg_mov",
                                            tvm.call_extern("uint64", "reg", tmp[0]),
                                            p_ub_buf.access_ptr('r', offset=aipp_comm.HEAD_OFFSET_CSC_SWITCH)))
                    if cur_cce_product in ['5.10']:
                        spr[1] = tvm.const(1 << 63, dtype="uint64")
                    else:
                        spr[1] = (tmp[0] & 0x1) << 63
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_1", spr[1]))
                # YUV422SP_U8
                with ib.if_scope(input_format[0] == 8):
                    input_offset = \
                        batch_factor * ((2 * src_image_size[0] * src_image_size[1]))
                    spr0 = get_const(
                        input_buf.access_ptr('r',
                                             offset=block_index*input_offset +
                                             n1*((2*src_image_size[0] * src_image_size[1]))))
                    uv_addr = get_const(
                        input_buf.access_ptr('r',
                                             offset=block_index*input_offset +
                                             n1*((2*src_image_size[0] * src_image_size[1])) +
                                             src_image_size[0] * src_image_size[1]))
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_0", spr0))

                    ib.emit(tvm.call_extern("int8",  # actual data type
                                            "reg_mov",
                                            tvm.call_extern("uint64", "reg", tmp[0]),
                                            p_ub_buf.access_ptr('r', offset=aipp_comm.HEAD_OFFSET_CSC_SWITCH)))
                    if cur_cce_product in ['5.10']:
                        spr[1] = tvm.const(1 << 63, dtype="uint64") | uv_addr
                    else:
                        spr[1] = (tmp[0] & 0x1) << 63 | uv_addr
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_1", spr[1]))
                # AYUV444_U8
                with ib.if_scope(input_format[0] == 9):
                    input_offset = \
                        batch_factor * ((2 * src_image_size[0] * src_image_size[1]))
                    spr0 = get_const(
                        input_buf.access_ptr('r',
                                             offset=block_index*input_offset +
                                             n1*((2*src_image_size[0] * src_image_size[1]))))

                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_0", spr0))

                    ib.emit(tvm.call_extern("int8",  # actual data type
                                            "reg_mov",
                                            tvm.call_extern("uint64", "reg", tmp[0]),
                                            p_ub_buf.access_ptr('r', offset=aipp_comm.HEAD_OFFSET_CSC_SWITCH)))
                    if cur_cce_product in ['5.10']:
                        spr[1] = tvm.const(1, dtype="uint64") << 63
                    else:
                        spr[1] = (tmp[0] & 0x1) << 63
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_1", spr[1]))
                # YUV400_U8
                with ib.if_scope(input_format[0] == 10):
                    input_offset = \
                        batch_factor * ((src_image_size[0] * src_image_size[1]))
                    spr0 = get_const(
                        input_buf.access_ptr('r',
                                             offset=block_index*input_offset +
                                             n1*((src_image_size[0] * src_image_size[1]))))
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_0", spr0))

                    spr[1] = tvm.const(0, dtype="uint64") << 63
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_1", spr[1]))

                aipp_comm.set_spr_dync_in_batch(ib, dtype, batch_id, p_ub_buf, spr, tmp)

                resize[0] = tvm.const(0, dtype="uint64")
                if cur_cce_product in ['5.10']:
                    scfIncVscl[0] = tvm.const(0, dtype="uint64")
                    scfIncHscl[0] = tvm.const(0, dtype="uint64")

                    spr[13] = tvm.const(0, dtype="uint64")
                    spr[16] = tvm.const(0, dtype="uint64")
                    ib.emit(tvm.call_extern("int8",
                                            "reg_mov",
                                            tvm.call_extern("uint64", "reg", tmp[0]),
                                            p_ub_buf.access_ptr('r',
                                                                offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE +
                                                                aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id +
                                                                aipp_comm.BATCH_OFFSET_SCF_SWITCH)))
                    resize[0] = tmp[0]
                    with ib.if_scope(resize[0] == 1):
                        ib.emit(tvm.call_extern("int32",
                                                "reg_mov",
                                                tvm.call_extern("uint64", "reg", tmp[0]),
                                                p_ub_buf.access_ptr('r',
                                                                    offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE +
                                                                    aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id +
                                                                    aipp_comm.BATCH_OFFSET_SCF_INPUT_H)))
                        resize_input_h[0] = tmp[0]
                        with ib.if_scope(resize_input_h[0] == 0):
                            with ib.if_scope(crop[0] == 1):
                                resize_input_h[0] = load_image_info[2]
                            with ib.else_scope():
                                resize_input_h[0] = src_image_size[0]

                        ib.emit(tvm.call_extern("int32",
                                                "reg_mov",
                                                tvm.call_extern("uint64", "reg", tmp[0]),
                                                p_ub_buf.access_ptr('r',
                                                                    offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE +
                                                                    aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id +
                                                                    aipp_comm.BATCH_OFFSET_SCF_OUTPUT_H)))
                        resize_output_h[0] = tmp[0]

                        with ib.if_scope(resize_input_h[0] != resize_output_h[0]):
                            spr[13] = spr[13] | tvm.const(1, dtype="uint64")

                            scfIncVscl[0] = \
                                ((resize_input_h[0] - tvm.const(1, dtype="uint64")) * tvm.const(262144, dtype="uint64") /
                                 (resize_output_h[0] - tvm.const(1, dtype="uint64"))) & \
                                0xFFFFFC
                            spr[16] = spr[16] | scfIncVscl[0]

                        ib.emit(tvm.call_extern("int32",
                                                "reg_mov",
                                                tvm.call_extern("uint64", "reg", tmp[0]),
                                                p_ub_buf.access_ptr('r',
                                                                    offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE +
                                                                    aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id +
                                                                    aipp_comm.BATCH_OFFSET_SCF_INPUT_W)))
                        resize_input_w[0] = tmp[0]
                        with ib.if_scope(resize_input_w[0] == 0):
                            with ib.if_scope(crop[0] == 1):
                                resize_input_w[0] = load_image_info[3]
                            with ib.else_scope():
                                resize_input_w[0] = src_image_size[1]

                        ib.emit(tvm.call_extern("int32",
                                                "reg_mov",
                                                tvm.call_extern("uint64", "reg", tmp[0]),
                                                p_ub_buf.access_ptr('r',
                                                                    offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE +
                                                                    aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id +
                                                                    aipp_comm.BATCH_OFFSET_SCF_OUTPUT_W)))
                        resize_output_w[0] = tmp[0]

                        with ib.if_scope(resize_input_w[0] != resize_output_w[0]):
                            spr[13] = spr[13] | tvm.const(1, dtype="uint64") << 2

                            scfIncHscl[0] = \
                                ((resize_input_w[0] - tvm.const(1, dtype="uint64")) * tvm.const(262144, dtype="uint64") /
                                 (resize_output_w[0] - tvm.const(1, dtype="uint64"))) & \
                                0xFFFFFC
                            spr[16] = spr[16] | scfIncHscl[0] << 32

                        with ib.if_scope(resize_output_w[0] > resize_input_w[0]):
                            spr[13] = spr[13] | tvm.const(1, dtype="uint64") << 7

                        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_13", spr[13]))
                        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_16", spr[16]))

                        actual_col_size_reg[0] = resize_output_h[0] * resize_output_w[0]
                    with ib.else_scope():
                        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_13", spr[13]))
                        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_16", spr[16]))

                get_dync_padding_size(ib, batch_id, p_ub_buf, tmp, padding_info)

                ib.emit(tvm.call_extern("int8",
                                        "reg_mov",
                                        tvm.call_extern("uint64", "reg", tmp[0]),
                                        p_ub_buf.access_ptr('r',
                                                            offset=aipp_comm.DYNC_PARAM_HEAD_STRUCT_SIZE + \
                                                                   aipp_comm.DYNC_PARAM_BATCH_STRUCT_SIZE*batch_id + \
                                                                   aipp_comm.BATCH_OFFSET_PAD_SWITCH)))
                padding[0] = tmp[0]
                # padding enable
                with ib.if_scope(tmp[0] > 0):
                    output_h = H - padding_info[0] - padding_info[1]
                    output_w = W
                    actual_col_size_reg[0] = get_const(output_h)*tvm.const(output_w, dtype="uint64")

                    # top_padding_size
                    with ib.if_scope(padding_info[0] > 0):
                        with ib.new_scope():
                            top_offset = block_index*offset + n1*C1*H*W*C0
                            process_padding(
                                ib,
                                (dtype, W, C0, size,
                                 padding_info[0], top_offset),
                                output_buf)
                    # bottom_padding_size
                    with ib.if_scope(padding_info[1] > 0):
                        with ib.new_scope():
                            bottom_offset = block_index*offset + \
                                            n1*C1*H*W*C0 + \
                                            (H-padding_info[1])*W*C0
                            process_padding(
                                ib,
                                (dtype, W, C0, size,
                                 padding_info[1], bottom_offset),
                                output_buf)

                with ib.if_scope(l1_image_buf_max >= actual_col_size_reg[0]):
                    with ib.new_scope():
                        output_cb_buf, output_ub_buf = \
                            aipp_comm.new_alloc(ib, dtype,
                                                l1_image_buf_max * C0)

                        if cur_cce_product in ["5.10"]:
                            spr[15] = tvm.const(0, dtype="uint64")
                            with ib.if_scope(resize[0] == 1):
                                spr[12] = (resize_output_h[0] - tvm.const(1, dtype="uint64")) | \
                                          (resize_output_w[0] - tvm.const(1, dtype="uint64")) << 16
                            with ib.else_scope():
                                spr[12] = (load_image_info[2] - tvm.const(1, dtype="uint64")) | \
                                          (load_image_info[3] - tvm.const(1, dtype="uint64")) << 16

                            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_12", spr[12]))
                            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_15", spr[15]))

                        aippXt = aippXt | (padding_info[2] & 0xff) << 32 | \
                                 (padding_info[3] & 0xff) << 45

                        aippXs = get_const(
                            (load_image_info[3] - 1) |
                            (load_image_info[2] - 1) << 16 |
                            (load_image_info[1]) << 32 |
                            (load_image_info[0]) << 48)
                        ib.emit(tvm.call_extern(dtype, "load_image_to_cbuf",
                                                output_cb_buf.access_ptr(
                                                    "rw",
                                                    ptr_type=dtype,
                                                    offset=0),
                                                aippXs,
                                                get_const(aippXt)))

                        output_offset = n1*C1*H*W*C0

                        ib.emit(tvm.call_extern(
                            dtype, 'copy_cbuf_to_ubuf',
                            output_ub_buf.access_ptr("w",
                                                     ptr_type=dtype,
                                                     offset=0),
                            output_cb_buf.access_ptr("rw",
                                                     ptr_type=dtype,
                                                     offset=0),
                            0, 1,
                            C1*(H-padding_info[0]-padding_info[1])*W*C0*size//32,
                            0, 0))

                        ib.emit(tvm.call_extern(
                            dtype, 'copy_ubuf_to_gm',
                            output_buf.access_ptr("w",
                                                  ptr_type=dtype,
                                                  offset=block_index*offset +
                                                  padding_info[0]*W*C0 +
                                                  output_offset),
                            output_ub_buf.access_ptr("rw",
                                                     ptr_type=dtype, offset=0),
                            0, 1,
                            C1*(H-padding_info[0]-padding_info[1])*W*C0*size//32,
                            0, 0))
                with ib.else_scope():
                    buffer_upper_limit = l1_image_buf_max
                    l1_size = cce.CceProductParams().getParams("L1_Buffer")

                    if 2*W > l1_image_buf_max:
                        buffer_upper_limit = l1_size // size // C0

                    tiling_h = buffer_upper_limit//W

                    # tiling_h must be even
                    if tiling_h % 2 != 0:
                        if tiling_h > 1:
                            tiling_h = tiling_h - 1

                    h_loop[0] = tvm.const(1, dtype="uint64")
                    with ib.if_scope(resize[0] == 1):
                        h_loop[0] = tvm.div(resize_output_h[0], tvm.const(tiling_h, dtype="uint64"))
                    with ib.else_scope():
                        h_loop[0] = tvm.div(load_image_info[2], tvm.const(tiling_h, dtype="uint64"))

                    h_start_pos = tvm.const(48, dtype="uint64")
                    w_start_pos = tvm.const(32, dtype="uint64")
                    load_h_pos = tvm.const(16, dtype="uint64")
                    tiling_h_const = tvm.const(tiling_h, dtype="uint64")
                    load_w = get_const(load_image_info[3] - 1)
                    load_h[0] = tvm.const(tiling_h - 1, dtype="uint64")

                    with ib.for_range(zero_const, h_loop[0], name="h1",
                                      dtype="uint64") as h1:
                        with ib.new_scope():
                            output_cb_buf, output_ub_buf = \
                                aipp_comm.new_alloc(ib, dtype,
                                                    C1*l1_image_buf_max*C0)

                            aippXt = aippXt | \
                                     (padding_info[2] & 0xff) << 32 | \
                                     (padding_info[3] & 0xff) << 45

                            output_w = W
                            output_h = tiling_h
                            output_offset = n1*C1*H*W*C0 + \
                                            C1*(h1*tiling_h)*output_w*C0

                            if cur_cce_product in ['5.10']:
                                with ib.if_scope(resize[0] == 1):
                                    spr[12] = tvm.const(tiling_h - 1, dtype="uint64") |\
                                              (resize_output_w[0] - tvm.const(1, dtype="uint64")) << 16
                                with ib.else_scope():
                                    spr[12] = tvm.const(tiling_h - 1, dtype="uint64") | \
                                              (load_image_info[3] - tvm.const(1, dtype="uint64")) << 16
                                ib.emit(tvm.call_extern(dtype, "set_aipp_spr_12", spr[12]))

                                with ib.if_scope(resize[0] == 1):
                                    with ib.if_scope(resize_input_h[0] != resize_output_h[0]):
                                        resize_input_h_stat_pos[0] = tvm.const(0, dtype="uint64")
                                        resize_input_h_end_pos[0] = tvm.const(0, dtype="uint64")

                                        resize_output_h_start_pos = h1*tiling_h
                                        resize_output_h_end_pos = \
                                            ((h1 + 1)*tiling_h - 1)

                                        resize_input_h_stat_pos[0] = \
                                            (scfIncVscl[0]*get_const(resize_output_h_start_pos)) >> 18
                                        resize_input_h_end_pos[0] = \
                                            ((scfIncVscl[0]*get_const(resize_output_h_end_pos)) +
                                             tvm.const((1 << 18) - 1, dtype="uint64")) >> 18

                                        with ib.if_scope(input_format[0] == 1):
                                            resize_input_h_stat_pos[0] = \
                                                resize_input_h_stat_pos[0] & \
                                                0xfffffffffffffffe
                                            resize_input_h_end_pos[0] += \
                                                (resize_input_h_end_pos[0] -
                                                 resize_input_h_stat_pos[0] + tvm.const(1, dtype="uint64")) & \
                                                0x1

                                        acc_vscl = \
                                            (scfIncVscl[0]*get_const(resize_output_h_start_pos)) - \
                                            (resize_input_h_stat_pos[0] << 18)
                                        spr15 = acc_vscl

                                        ib.emit(
                                            tvm.call_extern(dtype, "set_aipp_spr_15",
                                                            get_const(spr15)))

                                        load_h[0] = (resize_input_h_end_pos[0] - resize_input_h_stat_pos[0])
                                    with ib.else_scope():
                                        ib.emit(
                                            tvm.call_extern(dtype, "set_aipp_spr_15", tvm.const(0, dtype="uint64")))
                                with ib.else_scope():
                                    ib.emit(
                                        tvm.call_extern(dtype, "set_aipp_spr_15", tvm.const(0, dtype="uint64")))

                            with ib.if_scope(resize[0] == 1):
                                with ib.if_scope(resize_input_h[0] != resize_output_h[0]):
                                    Xs[0] = get_const(
                                        load_w | load_h[0] << load_h_pos |
                                        (get_const(load_image_info[1])) << w_start_pos |
                                        (load_image_info[0] + resize_input_h_stat_pos[0]) << h_start_pos)
                                with ib.else_scope():
                                    Xs[0] = get_const(
                                        load_w | load_h[0] << load_h_pos |
                                        (get_const(load_image_info[1])) << w_start_pos |
                                        (load_image_info[0] + h1*tiling_h_const) << h_start_pos)
                            with ib.else_scope():
                                Xs[0] = get_const(
                                    load_w | load_h[0] << load_h_pos |
                                    (get_const(load_image_info[1])) << w_start_pos |
                                    (load_image_info[0] + h1*tiling_h_const) << h_start_pos)

                            ib.emit(tvm.call_extern(dtype, "load_image_to_cbuf",
                                                    output_cb_buf.access_ptr(
                                                        "rw", ptr_type=dtype,
                                                        offset=0),
                                                    Xs[0],
                                                    get_const(aippXt)))

                            move_data_from_l1_to_gm(ib, C1*output_h*W*C0, dtype,
                                                    output_cb_buf, output_buf,
                                                    block_index*offset+padding_info[0]*W*C0+output_offset)

                    tail_h[0] = tvm.const(0, dtype="uint64")
                    with ib.if_scope(resize[0] == 1):
                        tail_h[0] = resize_output_h[0] % tvm.const(tiling_h, dtype="uint64")
                    with ib.else_scope():
                        tail_h[0] = load_image_info[2] % tvm.const(tiling_h, dtype="uint64")
                    with ib.if_scope(tail_h[0] != 0):
                        tail_h_postion[0] = load_image_info[0] + h_loop[0]*tvm.const(tiling_h, dtype="uint64")
                        load_tail_h[0] = (tail_h[0] - tvm.const(1, dtype="uint64"))

                        with ib.new_scope():
                            output_cb = ib.allocate(dtype,
                                                    (C1*buffer_upper_limit*C0,),
                                                    "output_cb",
                                                    scope=cce.scope_cbuf)
                            output_cb_buf = tvm.decl_buffer(
                                (C1*buffer_upper_limit*C0,), dtype,
                                "output_cb_buf", scope=cce.scope_cbuf,
                                data=output_cb)

                            output_w = W
                            aippXt = aippXt | \
                                     (padding_info[2] & 0xff) << 32 | \
                                     (padding_info[3] & 0xff) << 45
                            output_h = tail_h[0]
                            output_offset = \
                                n1*C1*H*W*C0 + C1*(h_loop[0]*tiling_h)*output_w*C0

                            if cur_cce_product in ['5.10']:
                                with ib.if_scope(resize[0] == 1):
                                    spr[12] = (tail_h[0] - tvm.const(1, dtype="uint64")) | \
                                              (resize_output_w[0] - tvm.const(1, dtype="uint64")) << 16
                                with ib.else_scope():
                                    spr[12] = (tail_h[0] - tvm.const(1, dtype="uint64")) | \
                                              (load_image_info[3] - tvm.const(1, dtype="uint64")) << 16
                                ib.emit(tvm.call_extern(dtype, "set_aipp_spr_12", spr[12]))

                                with ib.if_scope(resize[0] == 1):
                                    with ib.if_scope(resize_input_h[0] != resize_output_h[0]):
                                        resize_input_h_stat_pos[0] = tvm.const(0, dtype="uint64")
                                        resize_input_h_end_pos[0] = tvm.const(0, dtype="uint64")

                                        resize_output_h_start_pos = h_loop[0]*tvm.const(tiling_h, dtype="uint64")
                                        resize_output_h_end_pos = resize_output_h[0] - tvm.const(1, dtype="uint64")

                                        resize_input_h_stat_pos[0] = \
                                            (scfIncVscl[0]*get_const(resize_output_h_start_pos)) >> 18
                                        resize_input_h_end_pos[0] = \
                                            ((scfIncVscl[0]*get_const(resize_output_h_end_pos)) +
                                             tvm.const((1 << 18) - 1, dtype="uint64")) >> 18

                                        with ib.if_scope(input_format[0] == 1):
                                            resize_input_h_stat_pos[0] = \
                                                resize_input_h_stat_pos[0] & \
                                                0xfffffffffffffffe
                                            resize_input_h_end_pos[0] += \
                                                (resize_input_h_end_pos[0] -
                                                 resize_input_h_stat_pos[0] + tvm.const(1, dtype="uint64")) & \
                                                0x1

                                        acc_vscl = \
                                            (scfIncVscl[0]*get_const(resize_output_h_start_pos)) - \
                                            (resize_input_h_stat_pos[0] << 18)
                                        spr15 = acc_vscl

                                        ib.emit(
                                            tvm.call_extern(dtype, "set_aipp_spr_15",
                                                            get_const(spr15)))

                                        load_tail_h[0] = resize_input_h_end_pos[0] - resize_input_h_stat_pos[0]
                                        tail_h_postion[0] = load_image_info[0] + resize_input_h_stat_pos[0]
                                    with ib.else_scope():
                                        ib.emit(
                                            tvm.call_extern(dtype, "set_aipp_spr_15", tvm.const(0, dtype="uint64")))
                                with ib.else_scope():
                                    ib.emit(
                                        tvm.call_extern(dtype, "set_aipp_spr_15", tvm.const(0, dtype="uint64")))

                            aippXs = load_w | load_tail_h[0] << load_h_pos | \
                                     tail_h_postion[0] << h_start_pos | \
                                     (load_image_info[1]) << w_start_pos

                            ib.emit(tvm.call_extern(dtype, "load_image_to_cbuf",
                                                    output_cb_buf.access_ptr(
                                                        "rw", ptr_type=dtype,
                                                        offset=0),
                                                    get_const(aippXs),
                                                    get_const(aippXt)))

                            reg_move_data_from_l1_to_gm(ib, tmp_reg, C1*output_h*W*C0, dtype,
                                                        output_cb_buf, output_buf,
                                                        block_index*offset+padding_info[0]*W*C0+output_offset)

        _aipp_intrin()
        return ib.get()

    return tvm.extern([(N, C1, H, W, C0)], [input_tensor, param_tensor],
                      lambda ins, outs: aipp_ir(ins[0], ins[1], outs[0]), dtype=[dtype], name="aipp")
