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

aipp
"""

# pylint: disable=too-many-branches
# pylint: disable=ungrouped-imports

import json
from te import tvm
from te import platform as cce
from te.platform.cce_build import build_config
from topi.cce import util
from te.platform.cce_util import get_const
from impl import aipp_comm
from impl import aipp_resize_padding
from impl import aipp_dynamic

NoneType = type(None)


# pylint: disable=invalid-name,unused-argument,too-many-statements
# pylint: disable=too-many-arguments,too-many-locals,
def aipp_compute(input_tensor, input_shape, input_format,
                 output_data, aipp_config):
    """
    :param input_tensor:
    :param input_shape:
    :param input_format:
    :param output_data:
    :param aipp_config:
    :param is_dynamic:
    :return:
    """

    if input_format == "NHWC":
        N, H, W, C = input_shape
    elif input_format == "NCHW":
        N, C, H, W = input_shape
    else:
        N, C1, H, W, C0 = input_shape

    output_shape = output_data.get('shape')
    N, C1, H, W, C0 = output_shape

    src_image_size_h = H
    src_image_size_w = W
    load_start_pos_h = 0
    load_start_pos_w = 0
    load_image_h = H
    load_image_w = W

    if "crop" in aipp_config and aipp_config.get("crop") == 1:
        src_image_size_h, src_image_size_w, load_start_pos_h, load_start_pos_w, \
        load_image_h, load_image_w = aipp_comm.get_crop_info(aipp_config)

    dtype = output_data.get('dtype')
    if dtype == "float16":
        size = 2  # One pixel occupied bytes
        C0 = 16
    else:
        size = 1  # One pixel occupied bytes
        C0 = 32
    if aipp_config.get('input_format') not in ["NC1HWC0DI_S8", "NC1HWC0DI_FP16"]:
        C1 = (C + C0 - 1) // C0

    actual_col_size = H * W
    if "crop" in aipp_config and aipp_config.get("crop") == 1:
        actual_col_size = aipp_comm.get_actual_col_size(aipp_config, H, W)

    l1_image_buf_max = aipp_comm.get_l1_image_buf_max(actual_col_size, dtype, False)

    def aipp_ir(input_buf, output_buf):
        ib = tvm.ir_builder.create()
        if l1_image_buf_max < actual_col_size:
            half_l1_image_buf_max = l1_image_buf_max

        cur_cce_product = cce.cce_conf.CceProductParams().cce_product

        #1.1,1.3 is mini,5.10 is hisi-es, 2.10 is v200
        if cur_cce_product not in ["1.1", "1.3", "2.10", "2.11", "5.10"]:
            raise RuntimeError("Only support is mini,DC,MDC,hisi-es!cur_cce_product:",
                               cur_cce_product)

        device_core_num = \
            cce.cce_conf.CceProductParams().getParams("Device_core_num")
        batch_num = N
        batch_factor = 1

        if batch_num % device_core_num == 0:
            batch_factor = batch_num // device_core_num

        block_index = tvm.thread_axis("blockIdx.x")
        ib.scope_attr(block_index, "thread_extent", batch_num // batch_factor)

        if aipp_config.get('input_format') == "YUV420SP_U8":
            input_offset = \
                batch_factor*((C*src_image_size_h*src_image_size_w) // 2)
        elif aipp_config.get('input_format') in ["XRGB8888_U8", "RGB888_U8",
                                                 "ARGB8888_U8", "AYUV444_U8",
                                                 "YUV400_U8", "RAW10", "RAW12",
                                                 "RAW16", "uint16"]:
            input_offset =\
                batch_factor*((C*src_image_size_h*src_image_size_w))
        elif aipp_config.get('input_format') in ["YUYV_U8", "YUV422SP_U8"]:
            input_offset = \
                batch_factor*((2*src_image_size_h*src_image_size_w))
        elif aipp_config.get('input_format') in \
                ["NC1HWC0DI_S8", "NC1HWC0DI_FP16"]:
            input_offset = \
                batch_factor*((C1*src_image_size_h*src_image_size_w*6))

        offset = batch_factor*C1*H*W*C0

        zero_const = tvm.const(0, dtype="uint64")

        def _aipp_intrin():
            # config SPR2~SPR9

            aipp_comm.set_spr2_spr9(ib, aipp_config, dtype, cur_cce_product)

            aippXt = (src_image_size_w - 1)

            with ib.for_range(zero_const,
                              tvm.const(batch_factor, dtype="uint64"),
                              name="n1", dtype="uint64") as n1:
                with ib.for_range(zero_const,
                                  tvm.const(C1, dtype="uint64"),
                                  name="c1_index",
                                  dtype="uint64") as c1_index:
                    if aipp_config.get('input_format') == "YUV420SP_U8":
                        spr0 = get_const(
                            input_buf.access_ptr('r',
                                                 offset=block_index*input_offset +
                                                 n1*((C*src_image_size_h*src_image_size_w)//2)))
                    elif aipp_config.get('input_format') in ["XRGB8888_U8",
                                                             "RGB888_U8",
                                                             "ARGB8888_U8",
                                                             "AYUV444_U8",
                                                             "YUV400_U8",
                                                             "RAW10", "RAW12",
                                                             "RAW16", "uint16"]:
                        spr0 = get_const(
                            input_buf.access_ptr('r',
                                                 offset=block_index*input_offset +
                                                 n1*((C*src_image_size_h*src_image_size_w))))
                    elif aipp_config.get('input_format') in ["YUYV_U8",
                                                             "YUV422SP_U8"]:
                        spr0 = get_const(
                            input_buf.access_ptr('r',
                                                 offset=block_index*input_offset +
                                                 n1*((2*src_image_size_h*src_image_size_w))))
                    elif aipp_config.get('input_format') in ["NC1HWC0DI_S8",
                                                             "NC1HWC0DI_FP16"]:
                        spr0 = get_const(
                            input_buf.access_ptr('r',
                                                 offset=block_index*input_offset +
                                                 n1*((C1*src_image_size_h*src_image_size_w*6)) +
                                                 c1_index*src_image_size_h*src_image_size_w*4))

                    spr1 = tvm.const(0, dtype="uint64")
                    if ('csc_switch' in aipp_config) and \
                            (aipp_config.get('csc_switch') == 1):
                        spr1 = tvm.const(1 << 63, dtype="uint64")
                    else:
                        if cur_cce_product in ['5.10']:
                            if aipp_config.get('input_format') in \
                                    ["YUV420SP_U8", "YUYV_U8",
                                     "YUV422SP_U8", "AYUV444_U8"]:
                                spr1 = tvm.const(1 << 63, dtype="uint64")

                    if aipp_config.get('input_format') == "YUV420SP_U8":
                        spr1 = get_const(
                            input_buf.access_ptr(
                                'r',
                                offset=block_index*input_offset +
                                n1*((C*src_image_size_h*src_image_size_w)//2) +
                                src_image_size_h * src_image_size_w)) | \
                               spr1
                    elif aipp_config.get('input_format') == "YUV422SP_U8":
                        spr1 = get_const(
                            input_buf.access_ptr(
                                'r',
                                offset=block_index*input_offset +
                                n1*((2*src_image_size_h*src_image_size_w)) +
                                src_image_size_h*src_image_size_w)) | \
                               spr1
                    elif aipp_config.get('input_format') in \
                            ["NC1HWC0DI_S8", "NC1HWC0DI_FP16"]:
                        spr1 = get_const(
                            input_buf.access_ptr(
                                'r',
                                offset=block_index*input_offset +
                                n1*(C1*src_image_size_h*src_image_size_w*6) +
                                C1*src_image_size_h*src_image_size_w*4 +
                                c1_index*src_image_size_h*src_image_size_w*2)) | \
                               spr1

                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_0", spr0))
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_1", spr1))

                    if cur_cce_product in ['5.10']:
                        spr13 = 0
                        spr15 = 0
                        spr16 = 0
                        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_13",
                                                tvm.const(spr13, dtype="uint64")))
                        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_15",
                                                tvm.const(spr15, dtype="uint64")))
                        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_16",
                                                tvm.const(spr16, dtype="uint64")))
                    if l1_image_buf_max >= actual_col_size:
                        with ib.new_scope():
                            output_cb_buf, output_ub_buf = \
                                aipp_comm.new_alloc(ib, dtype,
                                                    l1_image_buf_max * C0)

                            if cur_cce_product in ['5.10']:
                                spr12 = 0
                                spr12 = \
                                    ((load_image_h - 1)) | \
                                    ((load_image_w - 1) << 16)
                                ib.emit(
                                    tvm.call_extern(dtype, "set_aipp_spr_12",
                                                    tvm.const(spr12,
                                                              dtype="uint64")))


                            aippXs = tvm.const((load_image_w - 1) |
                                               (load_image_h - 1) << 16 |
                                               (load_start_pos_w) << 32 |
                                               (load_start_pos_h) << 48,
                                               dtype="uint64")

                            ib.emit(tvm.call_extern(dtype, "load_image_to_cbuf",
                                                    output_cb_buf.access_ptr(
                                                        "rw", ptr_type=dtype,
                                                        offset=0),
                                                    aippXs,
                                                    tvm.const(aippXt,
                                                              dtype="uint64")))

                            lenBurst, nBurst =\
                                aipp_comm.get_lenBurst_and_nBurst(
                                    H*W*C0*size//32, 1)
                            output_offset = n1*C1*H*W*C0 + c1_index*H*W*C0

                            ib.emit(tvm.call_extern(
                                dtype, 'copy_cbuf_to_ubuf',
                                output_ub_buf.access_ptr("w", ptr_type=dtype,
                                                         offset=0),
                                output_cb_buf.access_ptr("rw", ptr_type=dtype,
                                                         offset=0),
                                0, nBurst, lenBurst, 0, 0))

                            ib.emit(tvm.call_extern(
                                dtype, 'copy_ubuf_to_gm',
                                output_buf.access_ptr(
                                    "w", ptr_type=dtype,
                                    offset=block_index*offset + output_offset),
                                output_ub_buf.access_ptr("rw", ptr_type=dtype,
                                                         offset=0),
                                0, nBurst, lenBurst, 0, 0))

                    else:
                        tiling_w, w_loop = \
                            aipp_comm.get_tiling_W(W, half_l1_image_buf_max, 1)
                        tiling_h = half_l1_image_buf_max // tiling_w

                        if aipp_config.get('input_format') == "YUV420SP_U8":
                            #tiling_h of YUV420SP_U8 must be even
                            if tiling_h % 2 != 0:
                                if tiling_h > 1:
                                    tiling_h = tiling_h - 1
                                if tiling_h == 1:
                                    tiling_h = 2
                                    tiling_w = tiling_w // 2
                                    w_loop = w_loop * 2

                        if aipp_config.get('input_format') in \
                                ["YUV420SP_U8", "YUV422SP_U8", "YUYV_U8"]:
                            if tiling_w % 2 != 0:
                                tiling_w = tiling_w - 1

                        tail_w = W - w_loop * tiling_w

                        h_loop = load_image_h // tiling_h

                        h_start_pos = tvm.const(48, dtype="uint64")
                        w_start_pos = tvm.const(32, dtype="uint64")
                        load_h_pos = tvm.const(16, dtype="uint64")
                        tiling_h_const = tvm.const(tiling_h, dtype="uint64")
                        tiling_w_const = tvm.const(tiling_w, dtype="uint64")
                        load_w = tvm.const(tiling_w - 1, dtype="uint64")
                        load_h = tvm.const(tiling_h - 1, dtype="uint64")
                        load_tail_w = tvm.const(tail_w - 1, dtype="uint64")

                        h_loop_const = tvm.const(h_loop, dtype="uint64")
                        w_loop_const = tvm.const(w_loop, dtype="uint64")

                        with ib.for_range(zero_const, h_loop_const,
                                          name="h1", dtype="uint64") as h1:
                            with ib.for_range(zero_const, w_loop_const,
                                              name="w1", dtype="uint64") as w1:
                                with ib.new_scope():
                                    output_cb_buf, output_ub_buf = \
                                        aipp_comm.new_alloc(
                                            ib, dtype, half_l1_image_buf_max*C0)
                                    # ib.scope_attr(output_cb_buf.data,
                                    # "double_buffer_scope", 1)
                                    # ib.scope_attr(output_ub_buf.data,
                                    # "double_buffer_scope", 1)

                                    if cur_cce_product in ['5.10']:
                                        spr12 = 0
                                        spr12 = get_const((tiling_h - 1) |
                                                          (tiling_w - 1) << 16)
                                        ib.emit(tvm.call_extern(
                                            dtype, "set_aipp_spr_12",
                                            tvm.const(spr12, dtype="uint64")))

                                    aippXs = get_const(
                                        load_w | load_h << load_h_pos |
                                        (load_start_pos_w + w1*tiling_w_const) << w_start_pos |
                                        (load_start_pos_h + h1*tiling_h_const) << h_start_pos)
                                    ib.emit(tvm.call_extern(
                                        dtype, "load_image_to_cbuf",
                                        output_cb_buf.access_ptr("rw",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        aippXs,
                                        tvm.const(aippXt, dtype="uint64")))

                                    lenBurst, nBurst =\
                                        aipp_comm.get_lenBurst_and_nBurst(
                                            tiling_h*tiling_w*C0*size//32, 1)

                                    ib.emit(tvm.call_extern(
                                        dtype, 'copy_cbuf_to_ubuf',
                                        output_ub_buf.access_ptr("w",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        output_cb_buf.access_ptr("rw",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        0, nBurst, lenBurst, 0, 0))

                                    if w_loop == 1 and W % tiling_w == 0:
                                        output_offset = n1*C1*H*W*C0 + \
                                                        c1_index*H*W*C0 + \
                                                        h1*tiling_h*tiling_w*C0
                                        ib.emit(tvm.call_extern(
                                            dtype, 'copy_ubuf_to_gm',
                                            output_buf.access_ptr(
                                                "w", ptr_type=dtype,
                                                offset=block_index*offset + \
                                                       output_offset),
                                            output_ub_buf.access_ptr(
                                                "rw", ptr_type=dtype, offset=0),
                                            0, nBurst, lenBurst, 0, 0))
                                    else:
                                        with ib.for_range(
                                                zero_const,
                                                tvm.const(tiling_h,
                                                          dtype="uint64"),
                                                name="h2", dtype="uint64") as h2:
                                            output_offset = \
                                                n1*C1*H*W*C0 + \
                                                c1_index*H*W*C0 + \
                                                (h1*tiling_h)*W*C0 + \
                                                (w1*tiling_w) * C0 + h2*W*C0
                                            lenBurst, nBurst = \
                                                aipp_comm.get_lenBurst_and_nBurst(
                                                    1*tiling_w*C0*size//32, 1)

                                            ib.emit(tvm.call_extern(
                                                dtype, 'copy_ubuf_to_gm',
                                                output_buf.access_ptr(
                                                    "w", ptr_type=dtype,
                                                    offset=block_index*offset +
                                                    output_offset),
                                                output_ub_buf.access_ptr(
                                                    "rw", ptr_type=dtype,
                                                    offset=h2*tiling_w*C0),
                                                0, nBurst, lenBurst, 0, 0))

                            #process W tail
                            if W % tiling_w != 0:
                                with ib.new_scope():
                                    output_cb_buf, output_ub_buf = \
                                        aipp_comm.new_alloc(
                                            ib, dtype, half_l1_image_buf_max*C0)
                                    # ib.scope_attr(output_cb_buf.data,
                                    # "double_buffer_scope", 1)
                                    # ib.scope_attr(output_ub_buf.data,
                                    # "double_buffer_scope", 1)

                                    if cur_cce_product in ['5.10']:
                                        spr12 = 0
                                        spr12 = \
                                            ((tiling_h - 1) | (tail_w - 1) << 16)
                                        ib.emit(
                                            tvm.call_extern(
                                                dtype, "set_aipp_spr_12",
                                                tvm.const(spr12, dtype="uint64")))

                                    aippXs = get_const(
                                        load_tail_w |
                                        load_h << load_h_pos |
                                        (load_start_pos_w +
                                         w_loop*tiling_w_const) << w_start_pos |
                                        (load_start_pos_h +
                                         h1*tiling_h_const) << h_start_pos)

                                    ib.emit(tvm.call_extern(
                                        dtype, "load_image_to_cbuf",
                                        output_cb_buf.access_ptr("rw",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        aippXs,
                                        tvm.const(aippXt, dtype="uint64")))

                                    lenBurst, nBurst = \
                                        aipp_comm.get_lenBurst_and_nBurst(
                                            tiling_h*tail_w*C0*size//32, 1)

                                    ib.emit(tvm.call_extern(
                                        dtype, 'copy_cbuf_to_ubuf',
                                        output_ub_buf.access_ptr("w",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        output_cb_buf.access_ptr("rw",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        0, nBurst, lenBurst, 0, 0))

                                    with ib.for_range(zero_const,
                                                      tiling_h_const,
                                                      name="h2",
                                                      dtype="uint64") as h2:
                                        output_offset = \
                                            n1*C1*H*W*C0 + \
                                            c1_index*H*W*C0 + \
                                            (h1*tiling_h)*W*C0 + \
                                            (w_loop*tiling_w)*C0 + h2*W*C0
                                        lenBurst, nBurst = \
                                            aipp_comm.get_lenBurst_and_nBurst(
                                                1*tail_w*C0*size//32, 1)

                                        ib.emit(tvm.call_extern(
                                            dtype, 'copy_ubuf_to_gm',
                                            output_buf.access_ptr(
                                                "w", ptr_type=dtype,
                                                offset=block_index*offset + \
                                                       output_offset),
                                            output_ub_buf.access_ptr(
                                                "rw", ptr_type=dtype,
                                                offset=h2*tail_w*C0),
                                            0, nBurst, lenBurst, 0, 0))

                        if load_image_h % tiling_h != 0:
                            tail_h = load_image_h % tiling_h

                            tail_h_postion = tvm.const(
                                load_start_pos_h + h_loop*tiling_h,
                                dtype="uint64")

                            load_tail_h = tvm.const(tail_h - 1, dtype="uint64")

                            with ib.for_range(zero_const,
                                              w_loop_const,
                                              name="w1", dtype="uint64") as w1:
                                with ib.new_scope():
                                    output_cb_buf, output_ub_buf =\
                                        aipp_comm.new_alloc(
                                            ib, dtype, half_l1_image_buf_max*C0)
                                    # ib.scope_attr(output_cb_buf.data,
                                    # "double_buffer_scope", 1)
                                    # ib.scope_attr(output_ub_buf.data,
                                    # "double_buffer_scope", 1)

                                    if cur_cce_product in ['5.10']:
                                        spr12 = 0
                                        spr12 = \
                                            ((tail_h - 1) | (tiling_w - 1) << 16)
                                        ib.emit(
                                            tvm.call_extern(
                                                dtype, "set_aipp_spr_12",
                                                tvm.const(spr12,
                                                          dtype="uint64")))

                                    aippXs = get_const(
                                        load_w | load_tail_h << load_h_pos |
                                        tail_h_postion << h_start_pos |
                                        (load_start_pos_w +
                                         w1*tiling_w_const) << w_start_pos)

                                    ib.emit(
                                        tvm.call_extern(
                                            dtype, "load_image_to_cbuf",
                                            output_cb_buf.access_ptr(
                                                "rw", ptr_type=dtype, offset=0),
                                            aippXs,
                                            tvm.const(aippXt, dtype="uint64")))

                                    lenBurst, nBurst = \
                                        aipp_comm.get_lenBurst_and_nBurst(
                                            tail_h*tiling_w*C0*size//32, 1)

                                    ib.emit(tvm.call_extern(
                                        dtype, 'copy_cbuf_to_ubuf',
                                        output_ub_buf.access_ptr("w",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        output_cb_buf.access_ptr("rw",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        0, nBurst, lenBurst, 0, 0))
                                    if w_loop == 1 and W % tiling_w == 0:
                                        output_offset = \
                                            n1*C1*H*W*C0 + \
                                            c1_index*H*W*C0 + \
                                            (tiling_h*h_loop)*tiling_w*C0
                                        ib.emit(tvm.call_extern(
                                            dtype, 'copy_ubuf_to_gm',
                                            output_buf.access_ptr(
                                                "w", ptr_type=dtype,
                                                offset=block_index*offset + output_offset),
                                            output_ub_buf.access_ptr(
                                                "rw", ptr_type=dtype, offset=0),
                                            0, nBurst, lenBurst, 0, 0))
                                    else:
                                        #if H has tail, W can not have tail
                                        with ib.for_range(0, tail_h,
                                                          name="h2",
                                                          dtype="uint64") as h2:
                                            output_offset = \
                                                n1*C1*H*W*C0 + \
                                                c1_index*H*W*C0 + \
                                                (tiling_h*h_loop)*W*C0 + \
                                                (w1*tiling_w)*C0 + h2*W*C0
                                            lenBurst, nBurst = \
                                                aipp_comm.get_lenBurst_and_nBurst(
                                                    1*tiling_w*C0*size//32, 1)

                                            ib.emit(tvm.call_extern(
                                                dtype, 'copy_ubuf_to_gm',
                                                output_buf.access_ptr(
                                                    "w", ptr_type=dtype,
                                                    offset=block_index*offset +
                                                    output_offset),
                                                output_ub_buf.access_ptr(
                                                    "rw", ptr_type=dtype,
                                                    offset=h2*tiling_w*C0),
                                                0, nBurst, lenBurst, 0, 0))

        _aipp_intrin()
        return ib.get()

    return tvm.extern([(N, C1, H, W, C0)], [input_tensor],
                      lambda ins, outs: aipp_ir(ins[0], outs[0]),
                      dtype=[dtype], name="aipp")


@util.check_input_type(dict, (dict, NoneType), dict, str, str, bool, bool)
def aipp(input_data, input_dync_param, output_data, aipp_config_json, kernel_name="aipp"):
    """Operation for aipp.
    Parameters
    ----------
    input_data: dict of input, include shape and dtype, dtype support uint8

    input_dync_param: dict of dynamic parameter,
    include shape and dtype, dtype support uint8

    aipp_config_json : json of aipp config

    kernel_name : cce kernel name, default value is "aipp"

    Returns
    -------
        None
    """

    if aipp_config_json == "{}" or aipp_config_json == "":
        raise RuntimeError("aipp_config_json is null")

    aipp_config = json.loads(aipp_config_json)

    if 'aipp_mode' not in aipp_config:
        raise RuntimeError("aipp_mode must be set!")

    aipp_mode = aipp_config.get('aipp_mode')
    if aipp_mode not in ['static', 'dynamic']:
        raise RuntimeError("the aipp mode is only support static or dynamic!")

    input_shape = input_data.get('shape')
    input_dtype = input_data.get('dtype')
    input_format = input_data.get('format')
    output_dtype = output_data.get('dtype')
    output_shape = output_data.get('shape')

    util.check_shape_rule(input_shape)
    util.check_tensor_shape_size(input_shape)

    check_list = ["float16", "uint8", "int8"]
    if  output_dtype.lower() not in check_list:
        raise RuntimeError("aipp support dtype is float16,uint8,int8!")

    input_format_list = ["NCHW", "NHWC", "NC1HWC0"]
    if input_format not in input_format_list:
        raise RuntimeError("aipp support format is NCHW, NHWC, NC1HWC0!")

    cur_cce_product = cce.cce_conf.CceProductParams().cce_product

    if aipp_mode == 'dynamic':
        input_dync_param_shape = input_dync_param.get('shape')
        input_dync_param_dtype = input_dync_param.get('dtype')
        util.check_shape_rule(input_dync_param_shape)
        util.check_tensor_shape_size(input_dync_param_shape)

        if input_dync_param_dtype != "uint8":
            raise RuntimeError("aipp input_dync_param_dtype must be uint8!")

        if cur_cce_product not in ["1.1", "1.3", "5.10"]:
            raise RuntimeError("dynamic aipp only support mini, lhisi_es")

        # Compute
        data = tvm.placeholder(input_shape, name='input', dtype=input_dtype.lower())
        param = tvm.placeholder(input_dync_param_shape, name='param', dtype=input_dync_param_dtype)
        output = aipp_dynamic.aipp_compute(data, param, input_shape, input_format, output_data)

        # Schedule
        s = tvm.create_schedule([output.op])
        with build_config:
            tvm.build(s, [data, param, output], "cce", name=kernel_name)
    else:
        aipp_comm.set_aipp_default_params(aipp_config)
        aipp_comm.check_aipp_static_config(input_shape, input_format, output_shape,
                                           aipp_config, cur_cce_product)

        if aipp_config.get("input_format") == "NC1HWC0DI_S8":
            if input_dtype != "int8" or output_dtype != "int8":
                raise RuntimeError("the input/output dtype of "
                                   "NC1HWC0DI_S8 must be int8!")
        elif aipp_config.get("input_format") == "NC1HWC0DI_FP16":
            if input_dtype != "float16" or output_dtype != "float16":
                raise RuntimeError("the input/output dtype of "
                                   "NC1HWC0DI_S8 must be float16!")
        elif aipp_config.get("input_format") in ["RAW10", "RAW12",
                                                 "RAW16", "uint16"]:
            if input_dtype != "uint16":
                raise RuntimeError("the input dtype of "
                                   "RAW10/12/16 must be uint16!")
        else:
            if input_dtype != "uint8":
                raise RuntimeError("the input dtype of "
                                   "RGB and YUV must be uint8!")

        # Compute
        data = tvm.placeholder(input_shape, name='input',
                               dtype=input_dtype.lower())
        if ("padding" in aipp_config and \
            aipp_config.get("padding") == 1) or \
           ("resize" in aipp_config and \
            aipp_config.get("resize") == 1):
            output = aipp_resize_padding.aipp_compute(data, input_shape,
                                                      input_format, output_data,
                                                      aipp_config)
        else:
            output = aipp_compute(data, input_shape, input_format,
                                  output_data, aipp_config)

        # Schedule
        s = tvm.create_schedule([output.op])
        with build_config:
            tvm.build(s, [data, output], "cce", name=kernel_name)
