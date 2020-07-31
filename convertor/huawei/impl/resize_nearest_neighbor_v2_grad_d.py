# pylint: disable=too-many-lines
# !/usr/bin/env python
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

resize_nearest_neighbor_v2_grad

"""
from te import tik
from topi.cce import util
from te import platform as tbe_platform
from impl.resize_nearest_neighbor_grad_d_h import resize_nearest_neighbor_grad_d_h

# C0 size
C0_SIZE = 16
# W_SLICE
W_SLICE = 256
# w slice threshold
SLICE_THRE = 160 * 1024 // (16 * 4)


# pylint: disable-msg=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable-msg=invalid-name,unused-argument
def clear_ub(tik_instance, dst_ub):
    """clear ub to zero

    Parameters
    ----------
    tik_instance: class
    dst_ub: destinatiob ub

    Returns
    -------
    None
    """
    shape = dst_ub.shape
    data_len = 1
    for i in shape:
        data_len = data_len * i
    dst_ub.reshape((data_len, ))

    total_repeat_times = data_len // 64
    tail = data_len % 64
    vector_dup_times = (total_repeat_times + 254) // 255
    with tik_instance.for_range(0, vector_dup_times) as i:
        repeat_times = calc_segment(tik_instance, total_repeat_times,
                                    i, 255)
        tik_instance.vector_dup(64, dst_ub[i * 255 * 64], 0,
                                repeat_times, 1, 8)

    if tail > 0:
        tik_instance.vector_dup(tail, dst_ub[total_repeat_times * 64],
                                0, 1, 1, 8)

    dst_ub.reshape(shape)

def _prod(values):
    """
    return: prod of values
    """
    res = 1
    for value in values:
        res *= value

    return res


def calc_slice_size(scale_w):
    """calc one line gradient

    Parameters
    ----------
    scale_w: float

    Returns
    -------
    in_w: float
    out_w: float
    """
    total_res = SLICE_THRE
    in_w = int((total_res - 1 - scale_w) // (1 + scale_w))
    out_w = int((in_w + 1) * scale_w) + 2
    return in_w, out_w


def calc_line_slice_bigscale(tik_instance, grads, y, grads_h, loc_h,
                             loc_w, n_index, start_c1, end_c1):
    """calc one line gradient
    """
    in_w = grads.shape[3]
    grads_ub = tik_instance.Tensor(
        "float32", [W_SLICE, 16], name="grads_ub", scope=tik.scope_ubuf)
    loc_reg = tik_instance.Scalar(dtype="int32")

    calc_c1_num = end_c1 - start_c1
    repeat_times = (in_w + W_SLICE - 1) // W_SLICE
    with tik_instance.for_range(0, calc_c1_num) as c1_index:

        with tik_instance.for_range(0, repeat_times) as w_index:
            cp_len = calc_segment(tik_instance, in_w, w_index, W_SLICE)
            # read one line grads
            tik_instance.tensor_mov(grads_ub,
                                    grads[n_index, start_c1 + c1_index,
                                          grads_h,
                                          w_index * W_SLICE, 0],
                                    '', 1, (cp_len * 16 * 4 + 31) // 32, 0, 0)

            with tik_instance.for_range(0, cp_len) as i:
                loc_reg.set_as(loc_w[(w_index * W_SLICE) + i])

                # move data out
                tik_instance.set_atomic_add(1)
                tik_instance.tensor_mov(y[n_index, start_c1 + c1_index,
                                          loc_h, loc_reg, 0],
                                        grads_ub[i, 0], '', 1,
                                        (1 * 16 * 4 + 31) // 32, 0, 0)
                tik_instance.set_atomic_add(0)


def calc_line_slice(tik_instance, grads, y, grads_h, loc_h,
                    loc_w, n_index, start_c1, end_c1, scale_w):
    """calc one line calc_line_slice
    """
    in_w = grads.shape[3]
    in_slice_w, out_slice_w = calc_slice_size(scale_w)

    grads_ub = tik_instance.Tensor(
        "float32", [in_slice_w, 16], name="grads_ub", scope=tik.scope_ubuf)
    y_ub = tik_instance.Tensor(
        "float32", [out_slice_w, 16], name="y_ub", scope=tik.scope_ubuf)
    loc_reg = tik_instance.Scalar(dtype="int32")
    index_reg = tik_instance.Scalar(dtype="int32")
    start_out_w = tik_instance.Scalar(dtype="int32")
    mov_out_w = tik_instance.Scalar(dtype="int32")

    calc_c1_num = end_c1 - start_c1
    repeat_times = (in_w + in_slice_w - 1) // in_slice_w
    with tik_instance.for_range(0, calc_c1_num) as c1_index:

        with tik_instance.for_range(0, repeat_times) as w_index:
            start_out_w.set_as(loc_w[w_index * in_slice_w])
            cp_len = calc_segment(tik_instance, in_w, w_index, in_slice_w)
            # read one line grads
            tik_instance.tensor_mov(grads_ub,
                                    grads[n_index, start_c1 + c1_index,
                                          grads_h,
                                          w_index * in_slice_w, 0],
                                    '', 1, (cp_len * 16 * 4 + 31) // 32, 0, 0)
            # clear out ub
            clear_ub(tik_instance, y_ub)

            with tik_instance.for_range(0, cp_len) as i:
                index_reg.set_as((w_index * in_slice_w) + i)
                index_reg.set_as(loc_w[index_reg])
                loc_reg.set_as(index_reg - start_out_w)
                tik_instance.vadd(16, y_ub[loc_reg, 0], y_ub[loc_reg, 0],
                                  grads_ub[i, 0], 1, 1, 1, 1, 0, 0, 0)

            # move data out
            mov_out_w.set_as(loc_w[(w_index * in_slice_w) + cp_len - 1])
            mov_out_w.set_as(mov_out_w - start_out_w + 1)
            tik_instance.set_atomic_add(1)
            tik_instance.tensor_mov(y[n_index, start_c1 + c1_index,
                                      loc_h, start_out_w, 0],
                                    y_ub[0, 0], '', 1,
                                    (mov_out_w * 16 * 4 + 31) // 32, 0, 0)
            tik_instance.set_atomic_add(0)


def calc_line(tik_instance, grads, y, grads_h,
              loc_h, loc_w, n_index, start_c1, end_c1):
    """calc one line gradient
    """
    in_w = grads.shape[3]
    out_w = y.shape[3]
    grads_ub = tik_instance.Tensor(
        "float32", [in_w, 16], name="grads_ub", scope=tik.scope_ubuf)
    y_ub = tik_instance.Tensor(
        "float32", [out_w, 16], name="y_ub", scope=tik.scope_ubuf)
    loc_reg = tik_instance.Scalar(dtype="int32")
    c1_reg = tik_instance.Scalar(dtype="int32")

    calc_c1_num = end_c1 - start_c1
    with tik_instance.for_range(0, calc_c1_num) as c1_index:
        # read one line grads
        c1_reg.set_as(start_c1 + c1_index)
        tik_instance.tensor_mov(grads_ub,
                                grads[n_index, c1_reg, grads_h, 0, 0],
                                '', 1, (in_w * 16 * 4 + 31) // 32, 0, 0)
        # clear out ub
        clear_ub(tik_instance, y_ub)

        with tik_instance.for_range(0, in_w) as i:
            loc_reg.set_as(loc_w[i])
            tik_instance.vadd(16, y_ub[loc_reg, 0], y_ub[loc_reg, 0],
                              grads_ub[i, 0], 1, 1, 1, 1, 0, 0, 0)

        # move data out
        tik_instance.set_atomic_add(1)
        tik_instance.tensor_mov(y[n_index, start_c1 + c1_index, loc_h, 0, 0],
                                y_ub, '', 1,
                                (out_w * 16 * 4 + 31) // 32, 0, 0)
        tik_instance.set_atomic_add(0)


def calc_location(tik_instance, grads, y, align_corners, half_pixel_centers):
    """calc output location

    Parameters
    ----------
    tik_instance: class
    grads: TVM tensor
    y: TVM tensor
    align_corners: bool
    half_pixel_centers: bool

    Returns
    -------
    loc_h: float
    loc_w: float
    """
    in_h = grads.shape[2]
    in_w = grads.shape[3]
    out_h = y.shape[2]
    out_w = y.shape[3]
    loc_h = tik_instance.Tensor(
        "int32", [in_h], name="loc_h", scope=tik.scope_ubuf)
    loc_w = tik_instance.Tensor(
        "int32", [in_w], name="loc_w", scope=tik.scope_ubuf)
    loc_h_fp = tik_instance.Tensor(
        "float32", [in_h], name="loc_h", scope=tik.scope_ubuf)
    loc_w_fp = tik_instance.Tensor(
        "float32", [in_w], name="loc_w", scope=tik.scope_ubuf)

    scale_h, scale_w = calc_scale(grads, y, align_corners)
    set_0_n(tik_instance, loc_h_fp)
    set_0_n(tik_instance, loc_w_fp)

    calc_loc_scale(tik_instance, loc_h_fp, loc_h,
                   scale_h, align_corners, half_pixel_centers, out_h)
    calc_loc_scale(tik_instance, loc_w_fp, loc_w,
                   scale_w, align_corners, half_pixel_centers, out_w)

    return loc_h, loc_w


def set_0_n(tik_instance, dst):
    """set ub as 0 to n

    Parameters
    ----------
    tik_instance: class
    dst: destinatiob ub

    Returns
    -------
    None
    """
    tmp = tik_instance.Tensor(
        "float32", [64], name="tmp", scope=tik.scope_ubuf)
    with tik_instance.for_range(0, 64) as i:
        tmp[i] = i

    data_len = dst.shape[0]
    repeat_times = data_len // 64
    tail = data_len % 64

    if repeat_times > 0:
        with tik_instance.for_range(0, repeat_times) as i:
            tik_instance.vadds(64, dst[i * 64], tmp, i * 64, 1, 1, 1, 8, 8)

    if tail > 0:
        tik_instance.vadds(tail, dst[repeat_times * 64],
                           tmp, repeat_times * 64,
                           1, 1, 1, 8, 8)


def calc_loc_scale(tik_instance, src_loc, dst_loc,
                   scale, align_corners, half_pixel_centers, bound):
    """calc location by scale

    Parameters
    ----------
    tik_instance: class
    src_loc: Tensor
    dst_loc: Tensor
    scale: float
    align_corners: bool
    half_pixel_centers: bool
    bound: int

    Returns
    -------
    None
    """

    data_len = src_loc.shape[0]
    repeat_times = data_len // 64
    tail = data_len % 64

    if repeat_times > 0:
        if not half_pixel_centers:
            tik_instance.vmuls(64, src_loc, src_loc, scale,
                               repeat_times, 1, 1, 8, 8)
        else:
            tik_instance.vadds(64, src_loc, src_loc, float(0.5),
                               repeat_times, 1, 1, 8, 8)
            tik_instance.vmuls(64, src_loc, src_loc, scale,
                               repeat_times, 1, 1, 8, 8)

    if tail > 0:
        if not half_pixel_centers:
            tik_instance.vmuls(tail, src_loc[repeat_times * 64],
                               src_loc[repeat_times * 64],
                               scale, 1, 1, 1, 8, 8)
        else:
            tik_instance.vadds(tail, src_loc[repeat_times * 64],
                               src_loc[repeat_times * 64],
                               float(0.5), 1, 1, 1, 8, 8)
            tik_instance.vmuls(tail, src_loc[repeat_times * 64],
                               src_loc[repeat_times * 64],
                               scale, 1, 1, 1, 8, 8)

    cmd = "round" if align_corners else "floor"
    if repeat_times > 0:
        tik_instance.vconv(64, cmd, dst_loc, src_loc,
                           repeat_times, 1, 1, 8, 8)

    if tail > 0:
        tik_instance.vconv(tail, cmd, dst_loc[repeat_times * 64],
                           src_loc[repeat_times * 64], 1, 1, 1, 8, 8)

    const_value_int32 = tik_instance.Tensor(
        "int32", [64], name="const_value", scope=tik.scope_ubuf)
    tik_instance.vector_dup(64, const_value_int32, bound - 1, 1, 0, 0)
    if repeat_times > 0:
        tik_instance.vmin(64, dst_loc, dst_loc, const_value_int32,
                          repeat_times, 1, 1, 0, 8, 8, 0)

    if tail > 0:
        tik_instance.vmin(tail, dst_loc[repeat_times * 64],
                          dst_loc[repeat_times * 64], const_value_int32,
                          1, 1, 1, 0, 8, 8, 0)


def calc_scale(grads, y, align_corners):
    """calc scale

    Parameters
    ----------
    grads: TVM tensor
    y: TVM tensor
    align_corners: bool

    Returns
    -------
    scale_h: float
    scale_w: float
    """

    in_h = grads.shape[2]
    in_w = grads.shape[3]
    out_h = y.shape[2]
    out_w = y.shape[3]

    scale_h = calc_resize_scale(out_h, in_h, align_corners)
    scale_w = calc_resize_scale(out_w, in_w, align_corners)

    return scale_h, scale_w


def calc_resize_scale(in_size, out_size, align_corners):
    """calc resize scale

    Parameters
    ----------
    in_size: int
    out_size: int
    align_corners: bool

    Returns
    -------
    scale: float
    """
    if align_corners and out_size > 1:
        scale = (in_size - 1) / (out_size - 1)
    else:
        scale = in_size / out_size
    return scale


def calc_segment(tik_instance, total_seg, seg_index, seg_len):
    """calc one block gradient

    Parameters
    ----------
    tik_instance: class
    total_seg: int
    seg_index: int
    seg_len: float

    Returns
    -------
    ret_seg_len:int
    """
    left_seg_len = tik_instance.Scalar(dtype="int64")
    ret_seg_len = tik_instance.Scalar(dtype="int64")
    seg_gap = tik_instance.Scalar(dtype="int64", init_value=seg_len)
    left_seg_len.set_as(total_seg - seg_index * seg_len)
    tik_instance.scalar_min(ret_seg_len, left_seg_len, seg_gap)
    return ret_seg_len


def calc_c1_segment(tik_instance, start, total_len, rois_index, c1_num):
    """calc one block gradient

    Parameters
    ----------
    tik_instance: class
    start: int
    total_len: int
    rois_index: float
    c1_num: float

    Returns
    -------
    start_c1:int
    end_c1:int
    """
    start_c1 = tik_instance.Scalar(dtype="int64")
    end_c1 = tik_instance.Scalar(dtype="int64")

    with tik_instance.if_scope(rois_index == 0):
        start_c1.set_as(start % c1_num)
        end_c1.set_as(start_c1 + total_len)
        tik_instance.scalar_min(end_c1, end_c1, c1_num)
    with tik_instance.else_scope():
        start_c1.set_as(0)
        end_c1.set_as(total_len + (start % c1_num) - (c1_num * rois_index))
        tik_instance.scalar_min(end_c1, end_c1, c1_num)

    return start_c1, end_c1


def calc_one_image(tik_instance, grads, y, loc_h, loc_w,
                   n_index, start_c1, end_c1, scale_w):
    """calc one image grad
    """
    in_w = grads.shape[3]
    out_w = y.shape[3]

    loc_h_reg = tik_instance.Scalar(dtype="int32")
    line_num = loc_h.shape[0]
    with tik_instance.for_range(0, line_num) as i:
        loc_h_reg.set_as(loc_h[i])
        if scale_w > 10:
            calc_line_slice_bigscale(tik_instance, grads, y, i, loc_h_reg,
                                     loc_w, n_index, start_c1, end_c1)
        elif (in_w + out_w) >= SLICE_THRE:
            calc_line_slice(tik_instance, grads, y, i, loc_h_reg, loc_w,
                            n_index, start_c1, end_c1, scale_w)
        else:
            calc_line(tik_instance, grads, y, i, loc_h_reg,
                      loc_w, n_index, start_c1, end_c1)


def resize_nearest_neighbor_grad_compute(tik_instance, grads,
                                         y, align_corners, half_pixel_centers,
                                         block_n, core_bias):
    """calc grad

    Parameters
    ----------
    tik_instance: class
    grads: Tensor
    y: Tensor
    align_corners: bool
    half_pixel_centers: bool
    block_n: int
    core_bias: int

    Returns
    -------

    """
    _, scale_w = calc_scale(grads, y, align_corners)
    loc_h, loc_w = calc_location(tik_instance, grads, y, align_corners, half_pixel_centers)
    c1_num = grads.shape[1]

    start_batch = tik_instance.Scalar(dtype="int32")
    start_batch.set_as(core_bias // c1_num)
    end_batch = tik_instance.Scalar(dtype="int32")
    end_batch.set_as((core_bias + block_n - 1) // c1_num)
    batch_num = tik_instance.Scalar(dtype="int32")
    batch_num.set_as(end_batch - start_batch + 1)

    with tik_instance.for_range(0, batch_num) as i:
        start_c1, end_c1 = calc_c1_segment(tik_instance, core_bias,
                                           block_n, i, c1_num)
        calc_one_image(tik_instance, grads, y, loc_h, loc_w,
                       start_batch + i, start_c1, end_c1, scale_w)


def resize_nearest_neighbor_grad_multicore(grads_shape, y_shape,
                                           align_corners, half_pixel_centers, kernel_name):
    """calc grad

    Parameters
    ----------
    grads_shape: list
    y_shape: list
    align_corners: bool
    half_pixel_centers: bool
    kernel_name: str

    Returns
    -------
    tik_instance
    """
    tik_instance = tik.Tik()
    core_counts = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
    grads_gm = tik_instance.Tensor("float32", grads_shape, name="grads",
                                   scope=tik.scope_gm)
    y_gm = tik_instance.Tensor("float32", y_shape, name="y_shape",
                               scope=tik.scope_gm, is_atomic_add=True)
    batch_size = grads_shape[0]
    c1_num = grads_shape[1]

    if (batch_size * c1_num) > core_counts:
        core_num = core_counts
    else:
        core_num = batch_size * c1_num
    core_n = (batch_size * c1_num) // core_num
    core_tail = (batch_size * c1_num) % core_num

    with tik_instance.for_range(0, core_num,
                                block_num=core_num) as core_index:
        block_n = tik_instance.Scalar(dtype="int32", init_value=core_n)
        core_bias = tik_instance.Scalar(dtype="int32")
        with tik_instance.if_scope(core_index == 0):
            block_n.set_as(block_n + core_tail)
            core_bias.set_as(0)
        with tik_instance.else_scope():
            core_bias.set_as((core_n * core_index) + core_tail)
        resize_nearest_neighbor_grad_compute(tik_instance, grads_gm, y_gm,
                                             align_corners, half_pixel_centers, block_n,
                                             core_bias)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[grads_gm], outputs=[y_gm])
    return tik_instance


def check_supported(grads, y, size, align_corners=False,
                    kernel_name="resize_nearest_neighbor_v2_grad"):
    """
    algorithm: floor
    calculating element-wise largest integer not greater than input_x,
    the type of input_data is "float32"

    Parameters
    ----------
    grads: dict
        dict with keys(shape and dtype) of input grads
    y: dict
        dict with keys(shape and dtype) of output y
    size: list
        (orig_height, orig_width)
    align_corners: bool
        whether align_corners
    kernel_name: str
        kernel_name

    Returns
    -------
    check_supported: bool
    """
    grads_format = grads.get("format")
    grads_shape = grads.get("shape")
    try:
        if grads_format == "NCHW":
            grads_w = grads_shape[3]
            grads_h = grads_shape[2]
        elif grads_format == "NHWC":
            grads_w = grads_shape[2]
            grads_h = grads_shape[1]
        elif grads_format == "NC1HWC0":
            grads_w = grads_shape[3]
            grads_h = grads_shape[2]
        else:
            return False
        if (grads_w + grads_h) >= 8 * 1024:
            return False
    except RuntimeError:
        return False

    return True


@util.check_input_type(dict, dict, (list, tuple), bool, bool, str)
def resize_nearest_neighbor_v2_grad_d(
        grads, y, size, align_corners=False, half_pixel_centers=False,
        kernel_name="resize_nearest_neighbor_v2_grad"):
    """
    calculating element-wise largest integer not greater than input_x,
    the type of input_data is "float32"

    Parameters
    ----------
    grads: dict
        dict with keys(shape and dtype) of input grads
    y: dict
        dict with keys(shape and dtype) of output y
    size: list
        (orig_height, orig_width)
    align_corners: bool
        whether align_corners
    half_pixel_centers: bool
        whether open half_pixel_centers
    kernel_name: str
        kernel_name

    Returns
    -------
    tik_instance: tik_instance
    """
    input_list = [grads, y]
    for input_data in input_list:
        input_shape = input_data.get("shape")
        input_dtype = input_data.get("dtype").lower()
        util.check_shape_rule(input_shape)
        util.check_kernel_name(kernel_name)
        check_list = {"float32"}
        util.check_dtype_rule(input_dtype, check_list)
    if half_pixel_centers:
        if align_corners:
            raise RuntimeError("If half_pixel_centers is True, "
                               "align_corners must be False.")
    if len(size) != 2:
        raise RuntimeError("the length of size should be 2")
    grads_shape = grads.get("shape")
    util.check_tensor_shape_size(grads_shape)
    if len(grads_shape) != 5:
        raise RuntimeError(
            "the length of grads_sharp must be 5,"
            " while it is: %d" % len(grads_shape))

    y_shape = y.get("shape")
    if len(y_shape) != 5:
        raise RuntimeError("the length of y_sharp must be 5,"
                           " while it is: %d" % len(y_shape))

    if (grads_shape[2] + grads_shape[3]) >= 8 * 1024:
        raise RuntimeError("the w of grads_shape and y_sharp is too big")
    grads_num = _prod(grads_shape)
    grads_limt = ((1 << 31) - 1) // 4
    weight_out = y_shape[3]
    weight_in = grads_shape[3]
    if (align_corners) and (weight_in > 1):
        weight_out -= 1
        weight_in -= 1
    scale_w = float(weight_out) / float(weight_in)
    if scale_w * 255 > 1 and y_shape[3] <= 2500 and \
            (grads_shape[3] > y_shape[3]) and grads_num <= grads_limt:
        resize_nearest_neighbor_grad_d_h(grads, y, size, align_corners, half_pixel_centers,
                                         kernel_name)
    else:
        resize_nearest_neighbor_grad_multicore(grads_shape, y_shape,
                                               align_corners, half_pixel_centers, kernel_name)
