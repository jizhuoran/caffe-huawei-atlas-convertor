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

Depthwise conv2D backprop input for the computation of
gradients of depthwise convolution with respect to the input.
"""
import math
from enum import Enum
from te import platform as tbe_platform
import te.lang.cce
import te.platform.cce_params as cce_params
from te import tvm
from topi.cce import util
import topi
# pylint: disable=locally-disabled, too-many-locals, too-many-statements
# pylint: disable=locally-disabled, too-many-arguments, invalid-name
# pylint: disable=too-many-lines
BLOCK_SIZE = cce_params.BLOCK_REDUCE

# shape's dim of input and output must be 4
FEATURE_MAP_DIM = 4

# shape's dim of filter must be 4
FILTER_DIM = 4

# shape's dim of strides must be 4
STRIDES_DIM = 4

# shape's dim of dilations must be 4
DILATIONS_DIM = 4

#General limitation of the size for input shape
SHAPE_SIZE_LIMIT = 1 << 30

const_dtype = "int32"

# intrinsic value
pad_mode_call = tvm.call_pure_intrin("int32", "tvm_cce_string_print",
                                     'PAD_NONE')
csize_call = tvm.call_pure_intrin("int32", "tvm_cce_string_print", 'CSIZE0')

# vector_dup only can support max repeat 255
vector_dump_max = 255


# pylint: disable=locally-disabled, too-many-locals, too-many-statements
# pylint: disable=locally-disabled, too-many-arguments, invalid-name
# pylint: disable=redefined-builtin
def check_params(shape, dtype, format):
    """
    check the parameters including shape, dtype, format

    Parameters
    ----------
    shape : shape of tensor

    dtype : data type

    format : tensor format

    Returns
    -------
    None
    """
    if format == "NCHW":
        util.check_shape_rule(shape, FEATURE_MAP_DIM, FEATURE_MAP_DIM)

    if format == "HWCK" or format == "HWCN":
        util.check_shape_rule(shape, FILTER_DIM, FILTER_DIM)
    check_list = ["float16"]
    util.check_dtype_rule(dtype, check_list)


class BlockTilingType(Enum):
    """
    The type of block tiling.
    invalid: tiling param is invalid.
    DIVISIBLE: Block tilting that can be exactly divided.
    FUSED: Uneven block tiling is split. Therefore, the block tiling is merged
            into one axis tiling.
    """
    INVALID = 0
    DIVISIBLE = 1
    FUSED = 2


# pylint: disable=locally-disabled, too-many-locals, too-many-statements
# pylint: disable=locally-disabled, too-many-arguments, invalid-name
def new_alloc(tvm_ir, dtype, shape, name, scope, double_buffer=False):
    """
    decl new buffer

    Parameters
    ----------
    tvm_ir : developer API of IR node builder make function.

    dtype : buffer date type.

    shape : buffer shape.

    name : buffer name.

    scope : buffer memory scope.

    double_buffer : whether need double buffer

    Returns
    -------
    buffer : tvm.schedule.Buffer
        Symbolic data buffer.

    """
    buf_var = tvm_ir.allocate(dtype, shape, name=name, scope=scope)
    if double_buffer:
        tvm_ir.scope_attr(buf_var.asnode(), "double_buffer_scope", 1)
    new_buffer = tvm.decl_buffer(shape,
                                 buf_var.dtype,
                                 name=name,
                                 scope=scope,
                                 data=buf_var)

    return new_buffer


# pylint: disable=locally-disabled, too-many-locals, too-many-statements
# pylint: disable=locally-disabled, too-many-arguments, invalid-name
def get_tiling(m, k, mini_dx_width):
    """
    get tiling size according to m and k
    In the case that L0A can be stored, it is preferred to ensure that
    the m value of the left matrix is larger and k >= 2*BLOCK_SIZE

    Parameters
    ----------
    m : number of left matrix rows.

    k : number of left matrix columns.

    mini_dx_width : dx width of mini kernel.

    Returns
    -------
    tile_m : tiling size of left matrix rows.

    tile_k : tiling size of left matrix columns.

    is_mul : whether split based on mini_dx_width multiples
    """
    l0a_size_bytes = tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.L0A_SIZE)
    data_size = tbe_platform.cce_intrin.get_bit_len("float16") // 8
    # tile m according to mini_dx_width multiples
    lcm_m = BLOCK_SIZE // math.gcd(mini_dx_width, BLOCK_SIZE) * mini_dx_width

    def _compute_tile_m(piece_k):
        one_block_size = piece_k * BLOCK_SIZE * data_size
        # half the size of the space because of double buffer
        space_m = l0a_size_bytes // one_block_size // 2
        if space_m >= lcm_m:
            floor_m = space_m // lcm_m * lcm_m
            is_mul = True
        else:
            floor_m = space_m // BLOCK_SIZE * BLOCK_SIZE
            is_mul = False
        return floor_m, is_mul

    if (k // BLOCK_SIZE) == 1:
        tile_k = 1
    else:
        floor_m, _ = _compute_tile_m(k // BLOCK_SIZE)
        if floor_m == 0:
            tile_k = 2
        else:
            cnt_m = (m + floor_m - 1) // floor_m
            cnt_k = min(cnt_m, k // BLOCK_SIZE)
            tile_k = max(k // BLOCK_SIZE // cnt_k, 2)
    max_m, is_mul = _compute_tile_m(tile_k)
    tile_m = max_m // BLOCK_SIZE

    return tile_m, tile_k, is_mul


# pylint: disable=locally-disabled, too-many-locals, too-many-statements
# pylint: disable=locally-disabled, too-many-arguments, invalid-name
# pylint: disable=too-many-nested-blocks
def depthwise_conv2d_backprop_input_kernel(out, src, input_shape, strides,
                                           pads, dilations):
    """
    algorithm:

    calculating depthwise conv2d backprop input in IR build method

    Parameters
    ----------
    out : input_grad

    src : out_backprop and filter

    input_shape : input tensor shape

    strides : the stride of the sliding window for height and width of the input

    pads : padding added to each dimension of the input

    dilations : the dilation factor for height and width of input

    Returns
    -------
    tvm_ir.get() : the ir_builder created

        Developer API of IR node builder make function.

    """
    tvm_ir = tvm.ir_builder.create()
    dout = src[0]
    filter_init = src[1]
    dx = out[0]

    batch, input_c1, input_height, input_width, _ = input_shape
    filter_shape = (int(i.value) for i in filter_init.shape)
    filter_height, filter_width, _, multiplier = filter_shape
    dout_shape = (int(i.value) for i in dout.shape)
    batch, output_c1, output_height, output_width, _ = dout_shape
    stride_h, stride_w = strides
    pad_top, _, pad_left, _ = pads
    dilation_h, dilation_w = dilations

    # dilation parameters
    dilated_filter_height = (filter_height - 1) * dilation_h + 1
    dilated_filter_width = (filter_width - 1) * dilation_w + 1
    full_height = input_height + dilated_filter_height - 1
    full_width = input_width + dilated_filter_width - 1
    dilated_height = (output_height - 1) * stride_h + 1
    dilated_width = (output_width - 1) * stride_w + 1
    dilated_pad_top = dilated_filter_height - pad_top - 1
    dilated_pad_left = dilated_filter_width - pad_left - 1
    dilated_pad_bottom = full_height - dilated_height - dilated_pad_top
    dilated_pad_right = full_width - dilated_width - dilated_pad_left

    # split kernel
    virtual_pad_top = dilated_pad_top - dilated_pad_top // \
                      stride_h * stride_h
    virtual_pad_left = dilated_pad_left - dilated_pad_left // \
                       stride_w * stride_w

    # compute max mini kernel size and tiling plan
    max_kernel_height = (filter_height + stride_h - 1) // stride_h
    max_kernel_width = (filter_width + stride_w - 1) // stride_w
    max_output_pad_height = output_height + dilated_pad_top // stride_h + \
                            dilated_pad_bottom // stride_h
    max_output_pad_width = output_width + dilated_pad_left // stride_w + \
                           dilated_pad_right // stride_w
    max_dx_height = max_output_pad_height - max_kernel_height + 1
    max_dx_width = max_output_pad_width - max_kernel_width + 1
    max_tile_m, max_tile_k, _ = get_tiling(max_dx_height * max_dx_width,
                                           max_kernel_height * \
                                           max_kernel_width * \
                                           BLOCK_SIZE, max_dx_width)

    def _get_mad_out_dtype():
        if te.platform.CceProductParams().cce_product == "5.10":
            mad_out_dtype = "float16"
        else:
            mad_out_dtype = "float32"
        return mad_out_dtype

    def _get_crmode_call():
        if te.platform.CceProductParams().cce_product == "5.10":
            return tvm.call_pure_intrin("int32", "tvm_cce_string_print",
                                        'CRMODE_NONE')
        return tvm.call_pure_intrin("int32", "tvm_cce_string_print",
                                    'CRMODE_F32toF16_NONE')

    def _ceil_to(value, ceil_value):
        if ceil_value <= 0:
            return value
        return ((value + ceil_value - 1) // ceil_value) * ceil_value

    def _get_block_tiling(block_axis_value):
        device_core_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        tiling = {}
        tiling["shape"] = {}

        all_block_value = 1
        for value in block_axis_value:
            all_block_value *= value
        tiling["fuse"] = all_block_value

        if all_block_value % device_core_num == 0:
            unblock_core_num = device_core_num
            cur_device_core_num = device_core_num // unblock_core_num
            block_tiling = []
            for value in block_axis_value:
                cur_axis_core_num = math.gcd(unblock_core_num, value)
                cur_device_core_num *= cur_axis_core_num
                unblock_core_num = device_core_num // cur_device_core_num
                block_tiling.append(value // cur_axis_core_num)
            tiling["shape"]["n"] = block_tiling[0]
            tiling["shape"]["c1"] = block_tiling[1]
            tiling["block_dim"] = device_core_num
            tiling["type"] = BlockTilingType.DIVISIBLE
        else:
            tiling["fuse_factor"] = _ceil_to(all_block_value, device_core_num) \
                                    // device_core_num
            tiling["block_dim"] = _ceil_to(
                all_block_value,
                tiling["fuse_factor"]) // tiling["fuse_factor"]
            tiling["type"] = BlockTilingType.FUSED
        tiling["result"] = True
        return tiling

    block_tiling = _get_block_tiling((batch, input_c1))

    def _calculation_block(n_index, c1_index):
        def _dump0(tile_left):
            uint64_all = tvm.const(2**64 - 1, dtype="uint64")
            dump_value = tvm.const(0.0, dtype="float16")
            dump_len = 8  # one repeat can process 8 blocks
            tvm_ir.emit(
                tvm.call_extern("uint64", 'set_vector_mask', uint64_all,
                                uint64_all))
            repeat = (tile_left + dump_len - 1) // dump_len
            repeat_loop = repeat // vector_dump_max
            with tvm_ir.if_scope(repeat >= vector_dump_max):
                with tvm_ir.for_range(0, repeat_loop) as i:
                    tvm_ir.emit(
                        tvm.call_extern(
                            dx.dtype, 'vector_dup',
                            dx_ub.access_ptr(
                                "rw",
                                offset=(vector_dump_max * dump_len * 16) * i),
                            dump_value, vector_dump_max, 1, 1, 8, 8))

            with tvm_ir.if_scope(repeat % vector_dump_max > 0):
                tvm_ir.emit(
                    tvm.call_extern(
                        dx.dtype, "vector_dup",
                        dx_ub.access_ptr("rw",
                                         offset=(vector_dump_max * dump_len) *
                                         repeat_loop), dump_value,
                        repeat % vector_dump_max, 1, 1, 8, 8))

            tvm_ir.emit(
                tvm.call_extern("uint64", 'set_vector_mask', uint64_all,
                                uint64_all))

        def _compute_mini_kernel_padding():
            # padding size for the mini kernel convolution
            margin_top = idx_h
            margin_bottom = ((full_height - 1) -
                             (idx_h + dilated_filter_height - 1)) % stride_h
            margin_left = idx_w
            margin_right = ((full_width - 1) -
                            (idx_w + dilated_filter_width - 1)) % stride_w
            pad_mini_top = (dilated_pad_top - \
                            margin_top) // stride_h
            pad_mini_bottom = (dilated_pad_bottom - \
                               margin_bottom) // stride_h
            pad_mini_left = (dilated_pad_left - \
                             margin_left) // stride_w
            pad_mini_right = (dilated_pad_right - \
                              margin_right) // stride_w
            return pad_mini_top, pad_mini_bottom, pad_mini_left, pad_mini_right

        def _load_out_backprop_once():
            lenBurstA = output_height * output_width
            fmap_offset = (n_index * output_c1 +
                           c1_index) * lenBurstA * BLOCK_SIZE
            tvm_ir.emit(
                tvm.call_extern(out_backprop_l1.dtype, "copy_gm_to_cbuf",
                                out_backprop_l1.access_ptr("w"),
                                dout.access_ptr("r", offset=fmap_offset), 0, 1,
                                lenBurstA, 0, 0, pad_mode_call))

        def _load_out_backprop_repeatedly():
            with tvm_ir.for_range(0, output_pad_height, name="h") as h:
                out_backprop_idx = ((-pad_mini_top + h) * output_width + \
                                    (-pad_mini_left)) * BLOCK_SIZE
                out_backprop_offset = (n_index * output_c1 +
                                       c1_index) * BLOCK_SIZE * \
                                      output_width * \
                                      output_height + \
                                      out_backprop_idx
                out_backprop_l1_offset = h * BLOCK_SIZE * \
                                         output_pad_width
                tvm_ir.emit(
                    tvm.call_extern(
                        out_backprop_l1.dtype, "copy_gm_to_cbuf",
                        out_backprop_l1.access_ptr(
                            "w", offset=out_backprop_l1_offset),
                        dout.access_ptr("r", offset=out_backprop_offset), 0, 1,
                        output_pad_width, 0, 0, pad_mode_call))

        def _load_mini_kernel(mini_kernel_height, mini_kernel_width):
            stride_offset_h = (first_pos_h + stride_h -
                               1) // stride_h * stride_h
            stride_offset_w = (first_pos_w + stride_w -
                               1) // stride_w * stride_w
            with tvm_ir.for_range(0, mini_kernel_height, name="m") as m:
                with tvm_ir.for_range(0, mini_kernel_width, name="n") as n:
                    pos_rot = (mini_kernel_width * m +
                               n) * BLOCK_SIZE * BLOCK_SIZE * multiplier
                    origin_rot_h = m * stride_h + virtual_pad_top - \
                                   idx_h + stride_offset_h
                    origin_rot_w = n * stride_w + virtual_pad_left - \
                                   idx_w + stride_offset_w
                    origin_h = dilated_filter_height - origin_rot_h - 1
                    origin_w = dilated_filter_width - origin_rot_w - 1
                    filter_offset = (c1_index *
                                     (filter_height * filter_width) +
                                     (filter_width * origin_h + origin_w)) * (
                                         BLOCK_SIZE * BLOCK_SIZE * multiplier)

                    with tvm_ir.if_scope(
                            tvm.all((origin_rot_h % dilation_h) == 0,
                                    (origin_rot_w % dilation_w) == 0)):
                        # load effective filter to l1
                        lenBurstB = BLOCK_SIZE * BLOCK_SIZE * multiplier // \
                                    BLOCK_SIZE
                        tvm_ir.emit(
                            tvm.call_extern(
                                mini_kernel_l1.dtype, "copy_gm_to_cbuf",
                                mini_kernel_l1.access_ptr("w", offset=pos_rot),
                                filter_init.access_ptr("r",
                                                       offset=filter_offset),
                                0, 1, lenBurstB, 0, 0, pad_mode_call))
                    with tvm_ir.else_scope():
                        # index map to the dilated zeros
                        dump0size = BLOCK_SIZE * BLOCK_SIZE * multiplier // \
                                    BLOCK_SIZE
                        dump_ub = new_alloc(tvm_ir,
                                            filter_init.dtype,
                                            dump0size,
                                            "dump_ub",
                                            scope=tbe_platform.scope_ubuf)
                        tvm_ir.emit(
                            tvm.call_extern(
                                mini_kernel_l1.dtype, "set_vector_mask",
                                tvm.const(0, dtype="uint64"),
                                tvm.const((2**16 - 1), dtype="uint64")))
                        tvm_ir.emit(
                            tvm.call_extern(dump_ub.dtype, "vector_dup",
                                            dump_ub.access_ptr("rw"),
                                            tvm.const(0, dtype="float16"), 1,
                                            0, 0, 0, 0))
                        tvm_ir.emit(
                            tvm.call_extern(
                                mini_kernel_l1.dtype, "set_vector_mask",
                                tvm.const((2**64 - 1), dtype="uint64"),
                                tvm.const((2**64 - 1), dtype="uint64")))
                        tvm_ir.emit(
                            tvm.call_extern(
                                mini_kernel_l1.dtype, "copy_ubuf_to_cbuf",
                                mini_kernel_l1.access_ptr("w", offset=pos_rot),
                                dump_ub.access_ptr("r"), 0, 1, dump0size, 0,
                                0))

        def _load3d_and_load2d(v, mini_kernel_height, mini_kernel_width,
                               tile_m, tile_k):
            # tile M and K according to memory size
            repeat_m = (mini_dx_height * mini_dx_width + tile_m * BLOCK_SIZE -
                        1) // (tile_m * BLOCK_SIZE)
            repeat_k = (mini_kernel_height * mini_kernel_width + tile_k -
                        1) // tile_k
            last_m = mini_dx_height * mini_dx_width - (repeat_m -
                                                       1) * tile_m * BLOCK_SIZE
            tile_piece = tvm_ir.allocate(const_dtype, (1, ),
                                         name='tile_piece',
                                         scope=tbe_platform.scope_reg)
            tile_piece_k = tvm_ir.allocate(const_dtype, (1, ),
                                           name='tile_piece_k',
                                           scope=tbe_platform.scope_reg)
            tile_left = tvm_ir.allocate(const_dtype, (1, ),
                                        name='tile_left',
                                        scope=tbe_platform.scope_reg)
            # tile M
            with tvm_ir.for_range(0, repeat_m, name="loop_m") as loop_m:
                index_m = loop_m * tile_m * BLOCK_SIZE
                loop_var = tvm.max(loop_m - repeat_m + 2, 0)
                tile_left[0] = loop_var * last_m + (
                    1 - loop_var) * tile_m * BLOCK_SIZE
                tile_piece[0] = (tile_left[0] + BLOCK_SIZE - 1) // BLOCK_SIZE

                # tile K
                with tvm_ir.for_range(0, repeat_k, name="loop_k") as loop_k:
                    out_backprop_l0a = new_alloc(tvm_ir,
                                                 dout.dtype,
                                                 img2col_buffer_size,
                                                 "out_backprop_l0a",
                                                 scope=tbe_platform.scope_ca,
                                                 double_buffer=True)
                    mini_kernel_l0b = new_alloc(tvm_ir,
                                                filter_init.dtype,
                                                filter_buffer_size,
                                                "mini_kernel_l0b",
                                                scope=tbe_platform.scope_cb,
                                                double_buffer=True)
                    index_k = loop_k * tile_k * BLOCK_SIZE
                    tile_top = tvm_ir.allocate(const_dtype, (1, ),
                                               name='tile_top',
                                               scope=tbe_platform.scope_reg)
                    with tvm_ir.if_scope(loop_k == (repeat_k - 1)):
                        tile_top[0] = mini_kernel_height * mini_kernel_width * \
                                      BLOCK_SIZE - index_k
                    with tvm_ir.else_scope():
                        tile_top[0] = tile_k * BLOCK_SIZE
                    tile_piece_k[0] = tile_top[0] // BLOCK_SIZE

                    def _img2col_repeat_mode0():
                        with tvm_ir.for_range(0, tile_piece[0], name="i") as i:
                            index_inner = i * BLOCK_SIZE
                            first_h = (index_inner //
                                       mini_dx_width) - set_pad_top
                            first_w = (index_inner %
                                       mini_dx_width) - set_pad_left
                            l0a_offset = i * BLOCK_SIZE * BLOCK_SIZE * \
                                         mini_kernel_height * mini_kernel_width
                            tvm_ir.emit(
                                tvm.call_extern(
                                    out_backprop_l0a.dtype,
                                    "img2col_cbuf_to_ca",
                                    out_backprop_l0a.access_ptr(
                                        "w", offset=l0a_offset),
                                    out_backprop_l1.access_ptr("r"), 0, 0,
                                    first_w, first_h, 0, 1, 1,
                                    mini_kernel_width, mini_kernel_height, 1,
                                    1, 0, 0,
                                    mini_kernel_width * mini_kernel_height,
                                    csize_call))

                    def _img2col_repeat_mode1():
                        pos_wk = tvm_ir.allocate(const_dtype, (1, ),
                                                 name='pos_wk',
                                                 scope=tbe_platform.scope_reg)
                        pos_hk = tvm_ir.allocate(const_dtype, (1, ),
                                                 name='pos_hk',
                                                 scope=tbe_platform.scope_reg)
                        pos_wk[0] = (index_k // BLOCK_SIZE % mini_kernel_width)
                        pos_hk[0] = (index_k // BLOCK_SIZE //
                                     mini_kernel_width)
                        with tvm_ir.for_range(0, tile_piece_k[0],
                                              name="i") as i:
                            index_inner = index_m
                            first_h = (index_inner //
                                       mini_dx_width) - set_pad_top
                            first_w = (index_inner %
                                       mini_dx_width) - set_pad_left
                            l0a_offset = i * BLOCK_SIZE * BLOCK_SIZE
                            tvm_ir.emit(
                                tvm.call_extern(
                                    out_backprop_l0a.dtype,
                                    "img2col_cbuf_to_ca",
                                    out_backprop_l0a.access_ptr(
                                        "w", offset=l0a_offset),
                                    out_backprop_l1.access_ptr("r"), pos_wk[0],
                                    pos_hk[0], first_w, first_h, 0, 1, 1,
                                    mini_kernel_width, mini_kernel_height, 1,
                                    1, tile_piece_k[0], 1, tile_piece[0],
                                    csize_call))
                            pos_wk[0] += tvm.const(1, dtype=const_dtype)
                            with tvm_ir.if_scope(
                                    pos_wk[0] == mini_kernel_width):
                                pos_wk[0] = tvm.const(0, dtype=const_dtype)
                                pos_hk[0] += tvm.const(1, dtype=const_dtype)

                    if v == 1:
                        if repeat_m == 1 and repeat_k == 1:
                            _img2col_repeat_mode0()
                        else:
                            _img2col_repeat_mode1()
                    else:
                        with tvm_ir.if_scope(
                                tvm.all(repeat_m == 1, repeat_k == 1)):
                            _img2col_repeat_mode0()
                        with tvm_ir.else_scope():
                            _img2col_repeat_mode1()

                    # load filter from l1 to l0b
                    tvm_ir.emit(
                        tvm.call_extern(
                            mini_kernel_l0b.dtype, "load_cbuf_to_cb",
                            mini_kernel_l0b.access_ptr("w"),
                            mini_kernel_l1.access_ptr("r",
                                                      offset=index_k *
                                                      BLOCK_SIZE), 0,
                            tile_piece_k[0] * multiplier, 1, 0, 0))

                    # accumulate when tile k
                    is_cover = tvm_ir.allocate(const_dtype, (1, ),
                                               name='is_cover',
                                               scope=tbe_platform.scope_reg)
                    with tvm_ir.if_scope(
                            loop_k == tvm.const(0, dtype=const_dtype)):
                        is_cover[0] = tvm.const(1, dtype=const_dtype)
                    with tvm_ir.else_scope():
                        is_cover[0] = tvm.const(0, dtype=const_dtype)

                    # GEMV mode when M=1
                    cube_m = tvm_ir.allocate(const_dtype, (1, ),
                                             name='cube_m',
                                             scope=tbe_platform.scope_reg)
                    with tvm_ir.if_scope(
                            tile_left[0] == tvm.const(1, dtype=const_dtype)):
                        cube_m[0] = BLOCK_SIZE
                    with tvm_ir.else_scope():
                        cube_m[0] = tile_left[0]

                    tvm_ir.emit(
                        tvm.call_extern(_get_mad_out_dtype(), "mad",
                                        dx_l0c.access_ptr("w"),
                                        out_backprop_l0a.access_ptr("r"),
                                        mini_kernel_l0b.access_ptr("r"),
                                        cube_m[0],
                                        tile_piece_k[0] * BLOCK_SIZE,
                                        BLOCK_SIZE, is_cover[0]))

                tvm_ir.emit(
                    tvm.call_extern(dx_ub.dtype, "copy_matrix_cc_to_ubuf",
                                    dx_ub.access_ptr("w"),
                                    dx_l0c.access_ptr("r"), 0, 1,
                                    tile_piece[0], 0, 0, _get_crmode_call()))

                if stride_h == 1 and stride_w == 1:
                    offset_gm = (n_index * input_c1 + c1_index) * (
                        input_height * input_width *
                        BLOCK_SIZE) + index_m * BLOCK_SIZE
                    tvm_ir.emit(
                        tvm.call_extern(dx.dtype, "copy_ubuf_to_gm",
                                        dx.access_ptr("w", offset=offset_gm),
                                        dx_ub.access_ptr("r"), 0, 1,
                                        tile_left[0], 0, 0))
                else:
                    loop_cnt = tvm_ir.allocate(const_dtype, (1, ),
                                               name='loop_cnt',
                                               scope=tbe_platform.scope_reg)

                    if is_mul:
                        mini_h_end = (index_m + tile_left[0] -
                                      1) // mini_dx_width
                        mini_w_end = (index_m + tile_left[0] -
                                      1) % mini_dx_width

                        def _emit_copy_ubuf_to_gm_v1(var, end):
                            offset_gm = (n_index * input_c1 + c1_index) * \
                                        (input_height * input_width * \
                                         BLOCK_SIZE) + ((idx_h + var * \
                                         stride_h) * input_width + idx_w) * \
                                        BLOCK_SIZE
                            offset_ub = (mini_dx_width +
                                         (var - index_m // mini_dx_width - 1) *
                                         mini_dx_width) * BLOCK_SIZE
                            nBurst = end + 1
                            tvm_ir.emit(
                                tvm.call_extern(
                                    dx.dtype, "copy_ubuf_to_gm",
                                    dx.access_ptr("w", offset=offset_gm),
                                    dx_ub.access_ptr("r", offset=offset_ub), 0,
                                    nBurst, 1, 0, stride_w - 1))

                        with tvm_ir.if_scope(loop_m == (repeat_m - 1)):
                            loop_cnt[0] = (tile_left[0] -
                                           1) // mini_dx_width + 1
                        with tvm_ir.else_scope():
                            loop_cnt[0] = tile_m * BLOCK_SIZE // mini_dx_width
                        with tvm_ir.for_range(0, loop_cnt[0], name="j") as j:
                            _emit_copy_ubuf_to_gm_v1(
                                j + index_m // mini_dx_width,
                                (mini_dx_width - 1))
                        _emit_copy_ubuf_to_gm_v1(mini_h_end, mini_w_end)
                    else:
                        mini_h_start = index_m // mini_dx_width
                        mini_w_start = index_m % mini_dx_width
                        mini_h_end = (index_m + tile_left[0] -
                                      1) // mini_dx_width
                        mini_w_end = (index_m + tile_left[0] -
                                      1) % mini_dx_width
                        if v == 2:
                            offset_ub_temp = tvm_ir.allocate(
                                const_dtype, (1, ),
                                name='offset_ub_temp',
                                scope=tbe_platform.scope_reg)
                            offset_gm_temp = tvm_ir.allocate(
                                const_dtype, (1, ),
                                name='offset_gm_temp',
                                scope=tbe_platform.scope_reg)

                        def _emit_copy_ubuf_to_gm_v2(var, start, end):
                            if v == 1:
                                offset_gm = (n_index * input_c1 +
                                             c1_index) * (input_height * \
                                             input_width * BLOCK_SIZE) + \
                                            ((idx_h + var * stride_h) * \
                                             input_width + (idx_w + \
                                             start * stride_w)) * BLOCK_SIZE
                                if var == mini_h_start:
                                    offset_ub = 0
                                else:
                                    offset_ub = (mini_dx_width - \
                                                 mini_w_start + (var - \
                                                 mini_h_start - 1) * \
                                                 mini_dx_width) * BLOCK_SIZE
                                nBurst = end - start + 1
                                tvm_ir.emit(
                                    tvm.call_extern(
                                        dx.dtype, "copy_ubuf_to_gm",
                                        dx.access_ptr("w", offset=offset_gm),
                                        dx_ub.access_ptr("r",
                                                         offset=offset_ub), 0,
                                        nBurst, 1, 0, stride_w - 1))
                            if v == 2:
                                offset_gm_temp[0] = (n_index * input_c1 + \
                                        c1_index) * (input_height *
                                                     input_width * \
                                        BLOCK_SIZE) + ((idx_h + var * \
                                        stride_h) * input_width + (idx_w + \
                                        start * stride_w)) * BLOCK_SIZE
                                with tvm_ir.if_scope(var == mini_h_start):
                                    offset_ub_temp[0] = tvm.const(
                                        0, dtype=const_dtype)
                                with tvm_ir.else_scope():
                                    offset_ub_temp[0] = (mini_dx_width - \
                                                         mini_w_start + (var - \
                                                         mini_h_start - 1) * \
                                                         mini_dx_width) * \
                                                         BLOCK_SIZE
                                nBurst = end - start + 1
                                tvm_ir.emit(
                                    tvm.call_extern(
                                        dx.dtype, "copy_ubuf_to_gm",
                                        dx.access_ptr(
                                            "w", offset=offset_gm_temp[0]),
                                        dx_ub.access_ptr(
                                            "r", offset=offset_ub_temp[0]), 0,
                                        nBurst, 1, 0, stride_w - 1))

                        _emit_copy_ubuf_to_gm_v2(mini_h_start, mini_w_start,
                                                 (mini_dx_width - 1))
                        loop_cnt[0] = mini_h_end - mini_h_start - 1
                        with tvm_ir.for_range(0, loop_cnt[0], name="j") as j:
                            ori_j = j + mini_h_start + 1
                            _emit_copy_ubuf_to_gm_v2(ori_j, 0,
                                                     (mini_dx_width - 1))
                        with tvm_ir.if_scope(mini_h_end - mini_h_start > 0):
                            _emit_copy_ubuf_to_gm_v2(mini_h_end, 0, mini_w_end)

        if stride_h > filter_height or stride_w > filter_width:
            total_dx_height = input_height
            total_dx_width = input_width
            total_tile_m, _, _ = get_tiling(total_dx_height * total_dx_width,
                                            BLOCK_SIZE, total_dx_width)
            dx_buffer_size = total_tile_m * BLOCK_SIZE * BLOCK_SIZE
            dx_ub = new_alloc(tvm_ir,
                              dx.dtype,
                              dx_buffer_size,
                              "dx_ub",
                              scope=tbe_platform.scope_ubuf)
            total_m = (total_dx_height * total_dx_width + total_tile_m *
                       BLOCK_SIZE - 1) // (total_tile_m * BLOCK_SIZE)
            _dump0(total_tile_m * BLOCK_SIZE)
            total_left = tvm_ir.allocate(const_dtype, (1, ),
                                         name='total_left',
                                         scope=tbe_platform.scope_reg)
            with tvm_ir.for_range(0, total_m, name="loop_m") as loop_m:
                index_m = loop_m * total_tile_m * BLOCK_SIZE
                with tvm_ir.if_scope(loop_m == (total_m - 1)):
                    total_left[0] = total_dx_height * total_dx_width - index_m
                with tvm_ir.else_scope():
                    total_left[0] = total_tile_m * BLOCK_SIZE

                total_offset = (n_index * input_c1 +
                                c1_index) * (input_height * input_width *
                                             BLOCK_SIZE) + index_m * BLOCK_SIZE
                tvm_ir.emit(
                    tvm.call_extern(dx.dtype, "copy_ubuf_to_gm",
                                    dx.access_ptr("w", offset=total_offset),
                                    dx_ub.access_ptr("r"), 0, 1, total_left[0],
                                    0, 0))

        # stride_h = 4 and stride_w = 4
        if stride_h * stride_w > 16:
            # for loop sink
            mini_kernel_height = tvm_ir.allocate(const_dtype,
                                                 (stride_h * stride_w, ),
                                                 name='mini_kernel_height',
                                                 scope=tbe_platform.scope_reg)
            mini_kernel_width = tvm_ir.allocate(const_dtype,
                                                (stride_h * stride_w, ),
                                                name='mini_kernel_width',
                                                scope=tbe_platform.scope_reg)
            tile_m = tvm_ir.allocate(const_dtype, (stride_h * stride_w, ),
                                     name='tile_m',
                                     scope=tbe_platform.scope_reg)
            tile_k = tvm_ir.allocate(const_dtype, (stride_h * stride_w, ),
                                     name='tile_k',
                                     scope=tbe_platform.scope_reg)
            for idx_h in range(stride_h):
                for idx_w in range(stride_w):
                    break_flag = False
                    for m in range(dilated_filter_height):
                        for n in range(dilated_filter_width):
                            index_h = idx_h + m
                            index_w = idx_w + n
                            # get one effective filter point
                            if ((index_h - virtual_pad_top) % stride_h) == 0 \
                                    and ((index_w - virtual_pad_left)
                                         % stride_w) == 0:
                                kernel_h = (dilated_filter_height - m -
                                            1) // stride_h + 1
                                kernel_w = (dilated_filter_width - n -
                                            1) // stride_w + 1
                                mini_kernel_height[idx_h * stride_w + idx_w] = \
                                    tvm.const(kernel_h, dtype=const_dtype)
                                mini_kernel_width[idx_h * stride_w + idx_w] = \
                                    tvm.const(kernel_w, dtype=const_dtype)
                                break_flag = True
                                break
                        if break_flag:
                            break
                    if kernel_h * kernel_w != 0:
                        pad_mini_top, pad_mini_bottom, pad_mini_left, \
                            pad_mini_right = _compute_mini_kernel_padding()
                        output_pad_height = output_height + pad_mini_top + \
                                            pad_mini_bottom
                        output_pad_width = output_width + pad_mini_left + \
                                           pad_mini_right
                        mini_dx_height = output_pad_height - \
                                         kernel_h + 1
                        mini_dx_width = output_pad_width - \
                                        kernel_w + 1
                        tile_m[idx_h * stride_w + idx_w], \
                            tile_k[idx_h * stride_w + idx_w], _ = get_tiling(
                                mini_dx_height * mini_dx_width,
                                kernel_h * kernel_w * \
                                BLOCK_SIZE, mini_dx_width)

            with tvm_ir.for_range(0, stride_h, name="idx_h") as idx_h:
                with tvm_ir.for_range(0, stride_w, name="idx_w") as idx_w:
                    kernel_area = mini_kernel_height[idx_h*stride_w+idx_w] * \
                                  mini_kernel_width[idx_h*stride_w+idx_w]

                    # dump zeros to gm when mini kernel size = 0
                    with tvm_ir.if_scope(kernel_area != 0):
                        pad_mini_top, pad_mini_bottom, \
                        pad_mini_left, \
                        pad_mini_right = _compute_mini_kernel_padding()
                        output_pad_height = output_height + pad_mini_top + \
                                            pad_mini_bottom
                        output_pad_width = output_width + pad_mini_left + \
                                           pad_mini_right
                        mini_dx_height = output_pad_height - \
                                         mini_kernel_height[
                                             idx_h*stride_w+idx_w] + 1
                        mini_dx_width = output_pad_width - \
                                        mini_kernel_width[
                                            idx_h*stride_w+idx_w] + 1
                        # set load3d config according to padding
                        reg_pad_params = tvm_ir.allocate(
                            'uint64', (6, ),
                            name='pad_params',
                            scope=tbe_platform.scope_reg)
                        set_pad_left = tvm.max(pad_mini_left, 0)
                        set_pad_right = tvm.max(pad_mini_right, 0)
                        set_pad_top = tvm.max(pad_mini_top, 0)
                        set_pad_bottom = tvm.max(pad_mini_bottom, 0)
                        set_output_width = tvm.min(output_pad_width,
                                                   output_width)
                        set_output_height = tvm.min(output_pad_height,
                                                    output_height)
                        reg_pad_params[0] = topi.cast(set_pad_left,
                                                      dtype='uint64')
                        reg_pad_params[1] = topi.cast(set_pad_right,
                                                      dtype='uint64')
                        reg_pad_params[2] = topi.cast(set_pad_top,
                                                      dtype='uint64')
                        reg_pad_params[3] = topi.cast(set_pad_bottom,
                                                      dtype='uint64')
                        reg_pad_params[4] = topi.cast(set_output_width,
                                                      dtype='uint64')
                        reg_pad_params[5] = topi.cast(set_output_height,
                                                      dtype='uint64')

                        # malloc storage space for the computation
                        filter_size = max_kernel_height * max_kernel_width * \
                                      BLOCK_SIZE * BLOCK_SIZE * multiplier
                        dout_size = output_height * output_width * \
                                    BLOCK_SIZE
                        dx_buffer_size = max_tile_m * BLOCK_SIZE * BLOCK_SIZE
                        dx_ub = new_alloc(tvm_ir,
                                          dx.dtype,
                                          dx_buffer_size,
                                          "dx_ub",
                                          scope=tbe_platform.scope_ubuf)

                        # load3d register configuration
                        # place output parameter in corresponding bit
                        fmatrixConfig = reg_pad_params[4] \
                                        | reg_pad_params[5] << 16 \
                                        | reg_pad_params[0] << 32 \
                                        | reg_pad_params[1] << 40 \
                                        | reg_pad_params[2] << 48 \
                                        | reg_pad_params[3] << 56
                        tvm_ir.emit(
                            tvm.call_extern(dout.dtype, "set_fmatrix",
                                            fmatrixConfig))

                        out_backprop_l1 = new_alloc(tvm_ir,
                                                    dout.dtype,
                                                    dout_size,
                                                    "out_backprop_l1",
                                                    scope=
                                                    tbe_platform.scope_cbuf,
                                                    double_buffer=True)

                        # move feature map from out to l1
                        with tvm_ir.if_scope(
                                tvm.all(pad_mini_left >= 0, pad_mini_top >= 0,
                                        pad_mini_right >= 0,
                                        pad_mini_bottom >= 0)):
                            _load_out_backprop_once()
                        with tvm_ir.else_scope():
                            _load_out_backprop_repeatedly()

                        # move filter from out to l1
                        mini_kernel_l1 = new_alloc(tvm_ir,
                                                   filter_init.dtype,
                                                   filter_size,
                                                   "mini_kernel_l1",
                                                   scope=
                                                   tbe_platform.scope_cbuf,
                                                   double_buffer=True)

                        first_pos_h = tvm.max(idx_h - virtual_pad_top, 0)
                        first_pos_w = tvm.max(idx_w - virtual_pad_left, 0)
                        _load_mini_kernel(
                            mini_kernel_height[idx_h * stride_w + idx_w],
                            mini_kernel_width[idx_h * stride_w + idx_w])

                        img2col_buffer_size = max_tile_m * BLOCK_SIZE * \
                                              max_tile_k * BLOCK_SIZE
                        filter_buffer_size = max_tile_k * BLOCK_SIZE * \
                                             BLOCK_SIZE * multiplier
                        is_mul = False
                        dx_l0c = new_alloc(tvm_ir,
                                           _get_mad_out_dtype(),
                                           dx_buffer_size,
                                           "dx_l0c",
                                           scope=tbe_platform.scope_cc)
                        _load3d_and_load2d(
                            2, mini_kernel_height[idx_h * stride_w + idx_w],
                            mini_kernel_width[idx_h * stride_w + idx_w],
                            tile_m[idx_h * stride_w + idx_w],
                            tile_k[idx_h * stride_w + idx_w])
        else:
            # unroll for loop
            for idx_h in range(stride_h):
                for idx_w in range(stride_w):
                    mini_kernel_height = 0
                    mini_kernel_width = 0
                    break_flag = False
                    for m in range(dilated_filter_height):
                        for n in range(dilated_filter_width):
                            index_h = idx_h + m
                            index_w = idx_w + n
                            # get one effective filter point
                            if ((index_h - virtual_pad_top) % stride_h) == 0 \
                                    and ((index_w - virtual_pad_left)
                                         % stride_w) == 0:
                                mini_kernel_height = (dilated_filter_height -
                                                      m - 1) // stride_h + 1
                                mini_kernel_width = (dilated_filter_width - n -
                                                     1) // stride_w + 1
                                break_flag = True
                                break
                        if break_flag:
                            break
                    kernel_area = mini_kernel_height * mini_kernel_width

                    if kernel_area != 0:
                        pad_mini_top, pad_mini_bottom, \
                            pad_mini_left, pad_mini_right = \
                            _compute_mini_kernel_padding()
                        output_pad_height = output_height + pad_mini_top + \
                                            pad_mini_bottom
                        output_pad_width = output_width + pad_mini_left + \
                                           pad_mini_right
                        mini_dx_height = output_pad_height - \
                                         mini_kernel_height + 1
                        mini_dx_width = output_pad_width - \
                                        mini_kernel_width + 1
                        tile_m, tile_k, is_mul = get_tiling(
                            mini_dx_height * mini_dx_width,
                            mini_kernel_height * mini_kernel_width * \
                            BLOCK_SIZE, mini_dx_width)
                        # set load3d config according to padding
                        set_pad_left = max(pad_mini_left, 0)
                        set_pad_right = max(pad_mini_right, 0)
                        set_pad_top = max(pad_mini_top, 0)
                        set_pad_bottom = max(pad_mini_bottom, 0)
                        set_output_width = min(output_pad_width, output_width)
                        set_output_height = min(output_pad_height,
                                                output_height)
                        # malloc storage space for the computation
                        filter_size = kernel_area * BLOCK_SIZE * \
                                      BLOCK_SIZE * multiplier
                        dout_size = set_output_height * set_output_width * \
                                    BLOCK_SIZE
                        dx_buffer_size = tile_m * BLOCK_SIZE * BLOCK_SIZE
                        dx_ub = new_alloc(tvm_ir,
                                          dx.dtype,
                                          dx_buffer_size,
                                          "dx_ub",
                                          scope=tbe_platform.scope_ubuf)

                        # load3d register configuration
                        # place output parameter in corresponding bit
                        fmatrixConfig = set_output_width \
                                        | set_output_height << 16 \
                                        | set_pad_left << 32 \
                                        | set_pad_right << 40 \
                                        | set_pad_top << 48 \
                                        | set_pad_bottom << 56
                        tvm_ir.emit(
                            tvm.call_extern(
                                dout.dtype, "set_fmatrix",
                                tvm.const(fmatrixConfig, dtype="uint64")))

                        out_backprop_l1 = new_alloc(tvm_ir,
                                                    dout.dtype,
                                                    dout_size,
                                                    "out_backprop_l1",
                                                    scope=
                                                    tbe_platform.scope_cbuf,
                                                    double_buffer=True)

                        # move feature map from out to l1
                        if pad_mini_left >= 0 and pad_mini_top >= 0 \
                                and pad_mini_right >= 0 \
                                and pad_mini_bottom >= 0:
                            _load_out_backprop_once()
                        else:
                            _load_out_backprop_repeatedly()

                        mini_kernel_l1 = new_alloc(tvm_ir,
                                                   filter_init.dtype,
                                                   filter_size,
                                                   "mini_kernel_l1",
                                                   scope=
                                                   tbe_platform.scope_cbuf,
                                                   double_buffer=True)

                        first_pos_h = max(idx_h - virtual_pad_top, 0)
                        first_pos_w = max(idx_w - virtual_pad_left, 0)
                        _load_mini_kernel(mini_kernel_height,
                                          mini_kernel_width)

                        img2col_buffer_size = tile_m * BLOCK_SIZE * \
                                              tile_k * BLOCK_SIZE
                        filter_buffer_size = tile_k * BLOCK_SIZE * \
                                             BLOCK_SIZE * multiplier
                        dx_l0c = new_alloc(tvm_ir,
                                           _get_mad_out_dtype(),
                                           dx_buffer_size,
                                           "dx_l0c",
                                           scope=tbe_platform.scope_cc)
                        _load3d_and_load2d(1, mini_kernel_height,
                                           mini_kernel_width, tile_m, tile_k)

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ir.scope_attr(block_index, "thread_extent", block_tiling["block_dim"])

    if block_tiling["type"] == BlockTilingType.DIVISIBLE:
        with tvm_ir.for_range(0, block_tiling["shape"]["n"],
                              name="n_i") as n_i:
            with tvm_ir.for_range(0, block_tiling["shape"]["c1"],
                                  name="loop_c1") as loop_c1:
                n_0 = block_index // (input_c1 // block_tiling["shape"]["c1"])
                n = n_0 * block_tiling["shape"]["n"] + n_i
                c1_0 = block_index - n_0 * (input_c1 //
                                            block_tiling["shape"]["c1"])
                c1 = c1_0 * block_tiling["shape"]["c1"] + loop_c1
                _calculation_block(n, c1)
    else:
        with tvm_ir.for_range(0, block_tiling["fuse_factor"],
                              name="fuse_factor") as fuse_factor:
            nc1 = block_index * block_tiling["fuse_factor"] + fuse_factor
            with tvm_ir.if_scope(nc1 < block_tiling["fuse"]):
                n = nc1 // input_c1
                c1 = nc1 % input_c1
                _calculation_block(n, c1)

    return tvm_ir.get()


# pylint: disable=locally-disabled, too-many-locals, too-many-statements
# pylint: disable=locally-disabled, too-many-arguments, invalid-name
# pylint: disable=too-many-branches, redefined-builtin
@util.check_input_type(dict, dict, dict, (list, tuple), (list, tuple),
                       (list, tuple), (list, tuple), str, str)
def depthwise_conv2d_backprop_input_d(
        filter,
        out_backprop,
        input_grad,
        input_size,
        strides,
        dilations=(1, 1, 1, 1),
        pads=(0, 0, 0, 0),
        data_format='NHWC',
        kernel_name="depthwise_conv2d_backprop_input"):
    """
    algorithm: depthwise conv2d backprop input

    computes the gradients of depthwise convolution with respect to the input

    Parameters
    ----------
    filter: dict
        4-D origin shape and dtype of filter tensor
        support [H, W, C, K], K is channel_multiplier

    out_backprop: dict
        4-D origin shape and dtype of out_backprop tensor,
        support [N, Co, Ho, Wo] or [N, Ho, Wo, Co],
        gradients w.r.t. the output of the convolution

    input_grad: dict
        4-D origin shape and dtype of input tensor,
        support [N, C, H, W] or [N, H, W, C]

    input_size: a list or tuple of four ints
        shape of input tensor, support [N, C, H, W] or [N, H, W, C]

    strides: a list or tuple of four ints
        the stride of the sliding window for height and width of the input of
        the convolution, support [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]

    dilations: an optional list or tuple of four ints
        the dilation factor for each dimension of input
        if set to k > 1, there will be k-1 skipped cells between each
        filter element on that dimension, support [1, 1, dilation_height,
        dilation_width] or [1, dilation_height, dilation_width, 1]

    pads: a list or tuple of four ints
        padding added to each dimension of the input

    data_format : str
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]

    kernel_name: str
        cce kernel name, default value is "depthwise_conv2d_backprop_input"

    Returns
    -------
    None
    """
    def _ceil(x):
        """
        Return the least multiple of 16 integer number
        which is greater than or equal to x.
        """
        return ((x + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    input_shape = input_grad.get("ori_shape")
    if input_size != input_shape:
        raise RuntimeError(
            "the output shape of depthwise_conv2d_backprop_input must be"
            "same with input_size.")
    input_dtype = input_grad.get("dtype").lower()
    filter_shape = filter.get("ori_shape")
    filter_dtype = filter.get("dtype").lower()

    output_shape = out_backprop.get("ori_shape")
    output_dtype = out_backprop.get("dtype").lower()

    input_ori_format = input_grad.get('ori_format')
    if input_ori_format != 'NCHW' and input_ori_format != 'NHWC':
        raise RuntimeError(
            "The format of input_grad in depthwise_conv2d_backprop_input only "
            "supported NCHW or NHWC.")
    filter_ori_format = filter.get('ori_format')
    if filter_ori_format not in ('HWCK', 'HWCN', 'NCHW'):
        raise RuntimeError(
            "The format of filter in depthwise_conv2d_backprop_input "
            "only supported HWCK(HWCN)/NCHW.")
    dout_ori_format = out_backprop.get('ori_format')
    if dout_ori_format != 'NCHW' and dout_ori_format != 'NHWC':
        raise RuntimeError(
            "The format of out_backprop in depthwise_conv2d_backprop_input "
            "only supported NCHW or NHWC.")

    # index of the strides dimension
    DIM_S_N, DIM_S_C, DIM_S_H, DIM_S_W = 0, 1, 2, 3
    # index of the dilations dimension
    DIM_D_N, DIM_D_C, DIM_D_H, DIM_D_W = 0, 1, 2, 3
    # index of the out_backprop dimension
    DIM_N, DIM_C, _, _ = 0, 1, 2, 3
    # index of the filter dimension
    _, _, DIM_W_C, DIM_W_K = 0, 1, 2, 3

    if input_ori_format == 'NHWC':
        DIM_S_N, DIM_S_H, DIM_S_W, DIM_S_C = 0, 1, 2, 3
        DIM_D_N, DIM_D_H, DIM_D_W, DIM_D_C = 0, 1, 2, 3
        input_shape = [
            input_shape[0], input_shape[3], input_shape[1], input_shape[2]
        ]
    if dout_ori_format == 'NHWC':
        output_shape = [
            output_shape[0], output_shape[3], output_shape[1], output_shape[2]
        ]
    if filter_ori_format == "NCHW":
        filter_shape = [
            filter_shape[2], filter_shape[3], filter_shape[1], filter_shape[0]
        ]
    if data_format != 'NCHW' and data_format != 'NHWC':
        raise RuntimeError(
            "The format of input in depthwise_conv2d_backprop input only "
            "supported NCHW and NHWC.")

    # check if the parameter is valid
    check_params(filter_shape, filter_dtype, "HWCK")
    check_params(output_shape, output_dtype, "NCHW")
    check_params(input_shape, input_dtype, "NCHW")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(output_shape, FEATURE_MAP_DIM, FEATURE_MAP_DIM)
    util.check_shape_rule(filter_shape, FILTER_DIM, FILTER_DIM)
    util.check_shape_rule(input_shape, FEATURE_MAP_DIM, FEATURE_MAP_DIM)
    util.check_shape_rule(strides, STRIDES_DIM, STRIDES_DIM)
    util.check_shape_rule(dilations, DILATIONS_DIM, DILATIONS_DIM)
    util.check_shape_size(input_shape, SHAPE_SIZE_LIMIT)
    util.check_shape_size(filter_shape, SHAPE_SIZE_LIMIT)
    util.check_shape_size(output_shape, SHAPE_SIZE_LIMIT)

    if strides[DIM_S_H] != strides[DIM_S_W]:
        raise RuntimeError(
            "current implementation only supports equal length strides in "
            "the row and column dimensions.")

    if (strides[DIM_S_N] != 1) or (strides[DIM_S_C] != 1):
        raise RuntimeError("the N-dim and C-dim of stride must be equal to 1.")

    if (dilations[DIM_D_N] != 1) or (dilations[DIM_D_C] != 1):
        raise RuntimeError(
            "the N-dim and C-dim of dilation must be equal to 1.")

    if input_shape[DIM_N] != output_shape[DIM_N]:
        raise RuntimeError(
            "feature map N-dim must be equal to out_backprop N-dim.")

    if filter_shape[DIM_W_K] != 1:
        raise RuntimeError("the K(N)-dim of filter must be equal to 1.")

    if input_shape[DIM_C] != output_shape[DIM_C]:
        raise RuntimeError(
            "feature map C-dim must be equal to out_backprop C-dim.")

    if (_ceil(input_shape[DIM_C]) //
            BLOCK_SIZE) != (_ceil(filter_shape[DIM_W_C]) // BLOCK_SIZE):
        raise RuntimeError(
            "support multiplier = 1, feature map C-dim must be equal to "
            "filter C-dim.")

    # check pad parameter
    if len(pads) != 4:
        raise RuntimeError("pads shape should be 4d.")

    # input parameters
    batch, input_channel, input_height, input_width = input_shape
    filter_height, filter_width, filter_channel, _ = filter_shape
    input_c1 = (input_channel + BLOCK_SIZE - 1) // BLOCK_SIZE
    stride_h, stride_w = strides[DIM_S_H], strides[DIM_S_W]
    dilation_h, dilation_w = dilations[DIM_D_H], dilations[DIM_D_W]
    strides = (stride_h, stride_w)
    dilations = (dilation_h, dilation_w)

    # output parameters
    batch, output_channel, output_height, output_width = output_shape
    output_c1 = (output_channel + BLOCK_SIZE - 1) // BLOCK_SIZE

    l1_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L1_SIZE)
    data_size = tbe_platform.cce_intrin.get_bit_len(output_dtype) // 8
    dilated_filter_height = (filter_height - 1) * dilation_h + 1
    dilated_filter_width = (filter_width - 1) * dilation_w + 1
    max_hw_in_l1 = (l1_size - dilated_filter_height * dilated_filter_width *
                    BLOCK_SIZE * BLOCK_SIZE * data_size) // (
                        data_size * output_width * output_height)
    dilated_output_w = output_width * stride_w - (stride_w - 1)
    max_dh_in_l1 = (l1_size - filter_height * filter_width * BLOCK_SIZE *
                    BLOCK_SIZE * data_size) // (data_size * dilated_output_w *
                                                BLOCK_SIZE) - (filter_height -
                                                               1)

    pad_top, pad_bottom, pad_left, pad_right = pads
    full_height = input_height + pad_top + pad_bottom
    full_width = input_width + pad_left + pad_right
    out_backprop_height = (full_height - dilated_filter_height) // stride_h + 1
    out_backprop_width = (full_width - dilated_filter_width) // stride_w + 1

    if output_height != out_backprop_height:
        raise RuntimeError(
            "Row number of out_backprop in depthwise_conv2d_backprop_input"
            " is wrong!")
    if output_width != out_backprop_width:
        raise RuntimeError("Column number of out_backprop in"
                           " depthwise_conv2d_backprop_input is wrong!")

    if max_hw_in_l1 >= BLOCK_SIZE and output_height != 1 and output_width != 1:
        input_shape = [batch, input_c1, input_height, input_width, BLOCK_SIZE]
        output_shape = [batch, output_c1, output_height, output_width,
                        BLOCK_SIZE]
        dout = tvm.placeholder(output_shape, dtype=output_dtype, name='dout')
        filter_init = tvm.placeholder(filter_shape,
                                      dtype=filter_dtype,
                                      name='filter')
        res = tvm.extern(
            [output_shape, filter_shape], [dout, filter_init],
            lambda ins, outs: depthwise_conv2d_backprop_input_kernel(
                outs, ins, input_shape, strides, pads, dilations),
            name="dx",
            dtype=input_dtype)
        sch = tvm.create_schedule(res.op)
    elif max_dh_in_l1 >= BLOCK_SIZE and dilation_h == 1 and dilation_w == 1:
        filter_shape = [_ceil(filter_channel) // BLOCK_SIZE,
                        filter_height * filter_width, 1, BLOCK_SIZE,
                        BLOCK_SIZE]
        filter_init = tvm.placeholder(filter_shape,
                                      dtype=filter_dtype,
                                      name='filter')

        output_shape = [batch, output_c1, 1, output_height, output_width,
                        BLOCK_SIZE]
        dout = tvm.placeholder(output_shape, dtype=output_dtype, name='dout')

        input_shape = [batch, input_c1, 1, input_height, input_width,
                       BLOCK_SIZE]
        res = te.lang.cce.depthwise_conv2d_backprop_input_d_compute(
            input_shape, filter_init, dout, [filter_height, filter_width],
            strides, pads)

        sch = te.lang.cce.te_schedule. \
            depthwise_conv2d_backprop_input_d_schedule(res)
    else:
        raise RuntimeError("L1's memory space is not enough!")

    with tbe_platform.build_config:
        tvm.build(sch, [filter_init, dout, res], "cce", name=kernel_name)
