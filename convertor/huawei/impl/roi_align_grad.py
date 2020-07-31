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

rois_align_grad
"""
# pylint: disable=too-many-lines
from te import tik
from te import platform as cce
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
from te import platform as tbe_platform


# C0 size
C0_SIZE = 16
# batch size
BATCH_SIZE = 128


# pylint: disable-msg=too-many-arguments,too-many-locals,too-many-statements
def roi_align_calc_grad_line(tik_instance, x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo,
                             x_hi, y_hi, y_lo, x_len, y_diff_ub, x_ind, x_diff,
                             image_index, start_c1, calc_c1_num):
    """calc one line gradient

    Parameters
    ----------
    tik_instance: class
    x_lo_w: float
    x_hi_w: float
    y_lo_w: float
    y_hi_w: float
    x_lo: TVM tensor
    x_hi: TVM tensor
    y_hi: int
    y_lo:int
    x_len: int
    y_diff_ub: TVM tensor
    x_ind: TVM tensor
    x_diff: TVM tensor
    image_index: int
    start_c1: int
    calc_c1_num: int

    Returns
    -------
    None
    """
    w1_vec = tik_instance.Tensor(
        "float32", (128,), name="w1", scope=tik.scope_ubuf)
    w2_vec = tik_instance.Tensor(
        "float32", (128,), name="w2", scope=tik.scope_ubuf)
    w3_vec = tik_instance.Tensor(
        "float32", (128,), name="w3", scope=tik.scope_ubuf)
    w4_vec = tik_instance.Tensor(
        "float32", (128,), name="w4", scope=tik.scope_ubuf)
    h_diff_reg = tik_instance.Scalar(dtype="int32")

    tik_instance.vmuls(64, w1_vec, x_hi_w, y_hi_w, 2, 1, 1, 8, 8)
    tik_instance.vmuls(64, w2_vec, x_lo_w, y_hi_w, 2, 1, 1, 8, 8)
    tik_instance.vmuls(64, w3_vec, x_hi_w, y_lo_w, 2, 1, 1, 8, 8)
    tik_instance.vmuls(64, w4_vec, x_lo_w, y_lo_w, 2, 1, 1, 8, 8)

    pool_w = y_diff_ub.shape[2]
    c1_num = y_diff_ub.shape[0]

    h_diff_reg.set_as(y_hi - y_lo)

    with tik_instance.for_range(0, x_len, thread_num=2) as i:
        w1_reg = tik_instance.Scalar(dtype="float32")
        w2_reg = tik_instance.Scalar(dtype="float32")
        w3_reg = tik_instance.Scalar(dtype="float32")
        w4_reg = tik_instance.Scalar(dtype="float32")
        x_lo_reg = tik_instance.Scalar(dtype="int32")
        x_hi_reg = tik_instance.Scalar(dtype="int32")
        w2_loc_reg = tik_instance.Scalar(dtype="int32")
        w3_loc_reg = tik_instance.Scalar(dtype="int32")
        w4_loc_reg = tik_instance.Scalar(dtype="int32")

        grad_index_reg = tik_instance.Scalar(dtype="int32")
        x_diff_ub = tik_instance.Tensor(
            "float32", [c1_num, 4, 16], name="x_diff_ub", scope=tik.scope_ubuf)
        tmp_result = tik_instance.Tensor(
            "float32", [3, c1_num * 16],
            name="tmp_result",
            scope=tik.scope_ubuf)

        grad_index_reg.set_as(x_ind[i])
        w1_reg.set_as(w1_vec[i])
        w2_reg.set_as(w2_vec[i])
        w3_reg.set_as(w3_vec[i])
        w4_reg.set_as(w4_vec[i])
        x_lo_reg.set_as(x_lo[i])
        x_hi_reg.set_as(x_hi[i])
        w2_loc_reg.set_as(x_hi_reg - x_lo_reg)
        w3_loc_reg.set_as(2 * h_diff_reg)
        w4_loc_reg.set_as(w2_loc_reg + w3_loc_reg)

        clear_ub(tik_instance, x_diff_ub)
        tik_instance.vmuls(16, x_diff_ub[0, 0, 0],
                           y_diff_ub[0, 0, grad_index_reg, 0], w1_reg,
                           calc_c1_num, 1, 1, 8, pool_w * 2)
        tik_instance.vmuls(16, tmp_result[0, 0],
                           y_diff_ub[0, 0, grad_index_reg, 0], w2_reg,
                           calc_c1_num, 1, 1, 2, pool_w * 2)
        tik_instance.vmuls(16, tmp_result[1, 0],
                           y_diff_ub[0, 0, grad_index_reg, 0], w3_reg,
                           calc_c1_num, 1, 1, 2, pool_w * 2)
        tik_instance.vmuls(16, tmp_result[2, 0],
                           y_diff_ub[0, 0, grad_index_reg, 0], w4_reg,
                           calc_c1_num, 1, 1, 2, pool_w * 2)

        tik_instance.vadd(16, x_diff_ub[0, w2_loc_reg, 0],
                          x_diff_ub[0, w2_loc_reg, 0], tmp_result[0, 0],
                          calc_c1_num, 1, 1, 1, 8, 8, 2)
        tik_instance.vadd(16, x_diff_ub[0, w3_loc_reg, 0],
                          x_diff_ub[0, w3_loc_reg, 0], tmp_result[1, 0],
                          calc_c1_num, 1, 1, 1, 8, 8, 2)
        tik_instance.vadd(16, x_diff_ub[0, w4_loc_reg, 0],
                          x_diff_ub[0, w4_loc_reg, 0], tmp_result[2, 0],
                          calc_c1_num, 1, 1, 1, 8, 8, 2)

        mov_data_ddr(tik_instance, x_diff, x_diff_ub, image_index, start_c1,
                     calc_c1_num, y_lo, x_lo_reg)


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
    dst_ub.reshape((data_len,))

    total_repeat_times = data_len // 64
    tail = data_len % 64
    vector_dup_times = (total_repeat_times + 254) // 255
    with tik_instance.for_range(0, vector_dup_times) as i:
        repeat_times = calc_segment(tik_instance, total_repeat_times, i, 255)
        tik_instance.vector_dup(64, dst_ub[i * 255 * 64], 0, repeat_times, 1, 8)

    if tail > 0:
        tik_instance.vector_dup(tail, dst_ub[total_repeat_times * 64, 0], 0, 1,
                                1, 8)

    dst_ub.reshape(shape)


def get_ydiff_line(tik_instance, y_diff, y_diff_ub, n_index, start_c1, c1_num,
                   line_index):
    """get one line ydiff data

    Parameters
    ----------
    tik_instance: class
    y_diff: TVM tensor
    y_diff_ub : TVM tensor
    n_index : int
    start_c1 : int
    c1_num : int
    line_index : int

    Returns
    -------
    None
    """
    h_num = y_diff.shape[2]
    w_num = y_diff.shape[3]

    c1_gap = ((h_num - 1) * w_num * 16 * 4) // 32
    if c1_gap <= 65535:
        tik_instance.tensor_mov(y_diff_ub[0],
                                y_diff[n_index, start_c1, line_index, 0, 0], '',
                                c1_num, w_num * 2, 0, c1_gap)
    else:
        with tik_instance.for_range(0, c1_num) as i:
            tik_instance.tensor_mov(
                y_diff_ub[0], y_diff[n_index, start_c1 + i, line_index, 0, 0],
                '', 1, w_num * 2, 0, 0)


def mov_data_ddr(tik_instance, x_diff, x_diff_ub,
                 image_index, start_c1, c1_num, h_index, w_index):
    """mov_data_ddr
    """
    h_num = x_diff.shape[2]
    w_num = x_diff.shape[3]

    with tik_instance.if_scope(h_index < (h_num - 1)):
        with tik_instance.if_scope(w_index < (w_num - 1)):
            mov_data_ddr_all(tik_instance, x_diff, x_diff_ub, image_index,
                             start_c1, c1_num, h_index, w_index)
        with tik_instance.else_scope():
            mov_data_ddr_onerow(tik_instance, x_diff, x_diff_ub, image_index,
                                start_c1, c1_num, h_index, w_index)
    with tik_instance.else_scope():
        with tik_instance.if_scope(w_index < (w_num - 1)):
            mov_data_ddr_oneline(tik_instance, x_diff, x_diff_ub, image_index,
                                 start_c1, c1_num, h_index, w_index)
        with tik_instance.else_scope():
            mov_data_ddr_onepoint(tik_instance, x_diff, x_diff_ub, image_index,
                                  start_c1, c1_num, h_index, w_index)


def mov_data_ddr_onepoint(tik_instance, x_diff, x_diff_ub, image_index,
                          start_c1, c1_num, h_index, w_index):
    """mov_data_ddr_onepoint
    """
    h_num = x_diff.shape[2]
    w_num = x_diff.shape[3]

    c1_gap = (((h_num * w_num) - 1) * 16 * 4) // 32
    if c1_gap < 0:
        c1_gap = 0

    w_gap = ((w_num - 2) * 16 * 4) // 32
    if w_gap < 0:
        w_gap = 0
    w_gap = 65536
    tik_instance.set_atomic_add(1)
    if c1_gap <= 65535:
        tik_instance.tensor_mov(x_diff[image_index, start_c1, h_index, w_index, 0],
                                x_diff_ub[0, 0, 0], '', c1_num, 2, c1_gap, 6)

    elif w_gap <= 65535:
        with tik_instance.for_range(0, c1_num) as i:
            tik_instance.tensor_mov(x_diff[image_index, start_c1+i, h_index, w_index, 0],
                                    x_diff_ub[i, 0, 0], '', 2, 4, w_gap, 0)
    else:
        with tik_instance.for_range(0, c1_num) as i:
            tik_instance.tensor_mov(x_diff[image_index, start_c1+i, h_index, w_index, 0],
                                    x_diff_ub[i, 0, 0], '', 1, 2, 0, 0)

    tik_instance.set_atomic_add(0)


def mov_data_ddr_oneline(tik_instance, x_diff, x_diff_ub, image_index,
                         start_c1, c1_num, h_index, w_index):
    """move xdiff to gm
    """
    h_num = x_diff.shape[2]
    w_num = x_diff.shape[3]

    c1_gap = (((h_num * w_num) - 2) * 16 * 4) // 32
    if c1_gap < 0:
        c1_gap = 0

    w_gap = ((w_num - 2) * 16 * 4) // 32
    if w_gap < 0:
        w_gap = 0
    w_gap = 65536
    tik_instance.set_atomic_add(1)
    if c1_gap <= 65535:
        tik_instance.tensor_mov(x_diff[image_index, start_c1, h_index, w_index, 0],
                                x_diff_ub[0, 0, 0], '', c1_num, 4, c1_gap, 4)

    elif w_gap <= 65535:
        with tik_instance.for_range(0, c1_num) as i:
            tik_instance.tensor_mov(x_diff[image_index, start_c1+i, h_index, w_index, 0],
                                    x_diff_ub[i, 0, 0], '', 2, 4, w_gap, 0)
    else:
        with tik_instance.for_range(0, c1_num) as i:
            tik_instance.tensor_mov(x_diff[image_index, start_c1+i, h_index, w_index, 0],
                                    x_diff_ub[i, 0, 0], '', 1, 4, 0, 0)

    tik_instance.set_atomic_add(0)


def mov_data_ddr_onerow(tik_instance, x_diff, x_diff_ub, image_index,
                        start_c1, c1_num, h_index, w_index):
    """move xdiff to gm
    """
    h_num = x_diff.shape[2]
    w_num = x_diff.shape[3]

    c1_gap = (((h_num * w_num) - 1) * 16 * 4) // 32
    if c1_gap < 0:
        c1_gap = 0

    w_gap = ((w_num - 2) * 16 * 4) // 32
    if w_gap < 0:
        w_gap = 0
    w_gap = 65536
    tik_instance.set_atomic_add(1)
    if c1_gap <= 65535:
        tik_instance.tensor_mov(x_diff[image_index, start_c1, h_index, w_index, 0],
                                x_diff_ub[0, 0, 0], '', c1_num, 2, c1_gap, 6)
        tik_instance.tensor_mov(x_diff[image_index, start_c1, h_index+1, w_index, 0],
                                x_diff_ub[0, 2, 0], '', c1_num, 2, c1_gap, 6)

    elif w_gap <= 65535:
        with tik_instance.for_range(0, c1_num) as i:
            tik_instance.tensor_mov(x_diff[image_index, start_c1+i, h_index, w_index, 0],
                                    x_diff_ub[i, 0, 0], '', 2, 4, w_gap, 0)
    else:
        with tik_instance.for_range(0, c1_num) as i:
            tik_instance.tensor_mov(x_diff[image_index, start_c1+i, h_index, w_index, 0],
                                    x_diff_ub[i, 0, 0], '', 1, 2, 0, 0)
            tik_instance.tensor_mov(x_diff[image_index, start_c1+i, h_index+1, w_index, 0],
                                    x_diff_ub[i, 2, 0], '', 1, 2, 0, 0)

    tik_instance.set_atomic_add(0)


def mov_data_ddr_all(tik_instance, x_diff,
                     x_diff_ub, image_index,
                     start_c1, c1_num, h_index,
                     w_index):
    """mov_data_ddr_all
    """
    h_num = x_diff.shape[2]
    w_num = x_diff.shape[3]

    c1_gap = (((h_num * w_num) - 2) * 16 * 4) // 32
    if c1_gap < 0:
        c1_gap = 0

    w_gap = ((w_num - 2) * 16 * 4) // 32
    if w_gap < 0:
        w_gap = 0
    w_gap = 65536
    tik_instance.set_atomic_add(1)
    if c1_gap <= 65535:
        tik_instance.tensor_mov(x_diff[image_index, start_c1,
                                       h_index, w_index, 0],
                                x_diff_ub[0, 0, 0], '', c1_num, 4, c1_gap, 4)
        tik_instance.tensor_mov(x_diff[image_index, start_c1,
                                       h_index+1, w_index, 0],
                                x_diff_ub[0, 2, 0], '', c1_num, 4, c1_gap, 4)

    elif w_gap <= 65535:
        with tik_instance.for_range(0, c1_num) as i:
            tik_instance.tensor_mov(x_diff[image_index, start_c1+i,
                                           h_index, w_index, 0],
                                    x_diff_ub[i, 0, 0], '',
                                    2, 4, w_gap, 0)
    else:
        with tik_instance.for_range(0, c1_num) as i:
            tik_instance.tensor_mov(x_diff[image_index, start_c1+i,
                                           h_index, w_index, 0],
                                    x_diff_ub[i, 0, 0], '', 1, 4, 0, 0)
            tik_instance.tensor_mov(x_diff[image_index, start_c1+i,
                                           h_index+1, w_index, 0],
                                    x_diff_ub[i, 2, 0], '', 1, 4, 0, 0)

    tik_instance.set_atomic_add(0)


def calc_max_c1_num(pool_w, c1_shape):
    """calc_max_c1_num ub resource

    Parameters
    ----------
    pool_w: int
    c1_shape: int

    Returns
    -------
    c1_num
    """
    available_res = 160 * 1024
    c1_num = available_res // ((pool_w + 14) * 16 * 4)
    if c1_num > c1_shape:
        c1_num = c1_shape

    return c1_num


def malloc_res(tik_instance, y_diff):
    """malloc ub resource

    Parameters
    ----------
    tik_instance: class
    y_diff: y_diff tensor

    Returns
    -------
    y_diff_ub
    """
    c1_shape = y_diff.shape[1]
    pool_w = y_diff.shape[3]

    c1_num = calc_max_c1_num(pool_w, c1_shape)
    y_diff_ub = tik_instance.Tensor(
        "float32", [c1_num, 1, pool_w, 16],
        name="y_diff_ub",
        scope=tik.scope_ubuf)

    return y_diff_ub


def roi_align_calc_grad_block(tik_instance, line_num, row_num, x_lo_w, x_hi_w,
                              y_lo_w, y_hi_w, x_lo, x_hi, y_lo, y_hi, y_diff,
                              n_index, x_ind, y_ind, x_diff, image_index,
                              start_c1, end_c1):
    """calc one block gradient
    """
    y_diff_ub = malloc_res(tik_instance, y_diff)
    y_lo_w_s = tik_instance.Scalar(dtype="float32")
    y_hi_w_s = tik_instance.Scalar(dtype="float32")
    y_ind_s = tik_instance.Scalar(dtype="int32")
    y_hi_s = tik_instance.Scalar(dtype="int32")
    y_lo_s = tik_instance.Scalar(dtype="int32")
    c1_range = tik_instance.Scalar(dtype="int32")
    start_c1_tmp = tik_instance.Scalar(dtype="int32")
    max_c1_num = tik_instance.Scalar(dtype="int32")
    c1_num = tik_instance.Scalar(dtype="int32", init_value=y_diff_ub.shape[0])
    max_c1_num.set_as(end_c1 - start_c1)
    with tik_instance.if_scope(c1_num > max_c1_num):
        c1_num.set_as(max_c1_num)
    with tik_instance.else_scope():
        c1_num.set_as(y_diff_ub.shape[0])

    c1_range.set_as((max_c1_num + c1_num - 1) // c1_num)

    with tik_instance.for_range(0, line_num) as i:
        y_lo_w_s.set_as(y_lo_w[i])
        y_hi_w_s.set_as(y_hi_w[i])
        y_ind_s.set_as(y_ind[i])
        y_lo_s.set_as(y_lo[i])
        y_hi_s.set_as(y_hi[i])

        start_c1_tmp.set_as(start_c1)
        with tik_instance.for_range(0, c1_range) as j:
            calc_c1_num = calc_segment(tik_instance, max_c1_num, j, c1_num)

            # move y_diff data from gm to ub
            get_ydiff_line(tik_instance, y_diff, y_diff_ub, n_index,
                           start_c1_tmp, calc_c1_num, y_ind_s)

            roi_align_calc_grad_line(tik_instance, x_lo_w, x_hi_w, y_lo_w_s,
                                     y_hi_w_s, x_lo, x_hi, y_hi_s, y_lo_s,
                                     row_num, y_diff_ub, x_ind, x_diff,
                                     image_index, start_c1_tmp, calc_c1_num)

            start_c1_tmp.set_as(start_c1_tmp + calc_c1_num)


def roi_align_calc_grid(tik_instance, h_ind, w_ind, const_value_0_127, grid_w,
                        grid_h, sample_num_w, sample_num_h, rois_start_w,
                        rois_start_h, height, width):
    """calc one block gradient

    Parameters
    ----------
    tik_instance: class
    h_ind: int
    w_ind: int
    const_value_0_127: float
    grid_w: float
    grid_h: float
    sample_num_w: float
    sample_num_h: int
    rois_start_w: int
    rois_start_h: int
    height:int
    width: int

    Returns
    -------
    x_lo_w, x_hi_w, y_lo_w, y_hi_w,
    x_lo, x_hi, y_lo, y_hi, x_ind, y_ind
    """
    x_lo_w = tik_instance.Tensor(
        "float32", [128], name="x_lo_w", scope=tik.scope_ubuf)
    x_hi_w = tik_instance.Tensor(
        "float32", [128], name="x_hi_w", scope=tik.scope_ubuf)
    y_lo_w = tik_instance.Tensor(
        "float32", [128], name="y_lo_w", scope=tik.scope_ubuf)
    y_hi_w = tik_instance.Tensor(
        "float32", [128], name="y_hi_w", scope=tik.scope_ubuf)
    x_lo = tik_instance.Tensor(
        "int32", [128], name="x_lo", scope=tik.scope_ubuf)
    x_hi = tik_instance.Tensor(
        "int32", [128], name="x_hi", scope=tik.scope_ubuf)
    y_lo = tik_instance.Tensor(
        "int32", [128], name="y_lo", scope=tik.scope_ubuf)
    y_hi = tik_instance.Tensor(
        "int32", [128], name="y_hi", scope=tik.scope_ubuf)

    raw_x = tik_instance.Tensor(
        "float32", [128], name="x", scope=tik.scope_ubuf)
    raw_y = tik_instance.Tensor(
        "float32", [128], name="y", scope=tik.scope_ubuf)
    x_vec = tik_instance.Tensor(
        "float32", [128], name="x", scope=tik.scope_ubuf)
    y_vec = tik_instance.Tensor(
        "float32", [128], name="y", scope=tik.scope_ubuf)
    x_ind = tik_instance.Tensor(
        "int32", [128], name="x_ind", scope=tik.scope_ubuf)
    y_ind = tik_instance.Tensor(
        "int32", [128], name="y_ind", scope=tik.scope_ubuf)

    tik_instance.vadds(64, x_vec, const_value_0_127, w_ind * 128, 2, 1, 1, 8, 8)
    tik_instance.vmuls(64, x_vec, x_vec, 1.0 / sample_num_w, 2, 1, 1, 8, 8)
    tik_instance.vadds(64, y_vec, const_value_0_127, h_ind * 128, 2, 1, 1, 8, 8)
    tik_instance.vmuls(64, y_vec, y_vec, 1.0 / sample_num_h, 2, 1, 1, 8, 8)
    tik_instance.vconv(64, "floor", x_ind, x_vec, 2, 1, 1, 8, 8)
    tik_instance.vconv(64, "floor", y_ind, y_vec, 2, 1, 1, 8, 8)

    grid_w_vector = tik_instance.Tensor(
        "float32", [128], name="grid_w_vector", scope=tik.scope_ubuf)
    grid_h_vector = tik_instance.Tensor(
        "float32", [128], name="grid_h_vector", scope=tik.scope_ubuf)
    tik_instance.vmuls(64, grid_w_vector, const_value_0_127, grid_w, 2, 1, 1, 8,
                       8)
    tik_instance.vmuls(64, grid_h_vector, const_value_0_127, grid_h, 2, 1, 1, 8,
                       8)

    half_grid = 0.5 * grid_w + rois_start_w
    tik_instance.vadds(64, raw_x, grid_w_vector, half_grid, 2, 1, 1, 8, 8)
    half_grid = 0.5 * grid_h + rois_start_h
    tik_instance.vadds(64, raw_y, grid_h_vector, half_grid, 2, 1, 1, 8, 8)

    const_zero = tik_instance.Tensor(
        "float32", [16], name="const_zero", scope=tik.scope_ubuf)
    tik_instance.vector_dup(16, const_zero, 0, 1, 0, 0)

    tik_instance.vmax(64, x_vec, raw_x, const_zero, 2, 1, 1, 0, 8, 8, 0)
    tik_instance.vmax(64, y_vec, raw_y, const_zero, 2, 1, 1, 0, 8, 8, 0)

    tik_instance.vconv(64, "floor", x_lo, x_vec, 2, 1, 1, 8, 8)
    tik_instance.vconv(64, "floor", y_lo, y_vec, 2, 1, 1, 8, 8)

    const_one = tik_instance.Tensor(
        "int32", [8], name="const_one", scope=tik.scope_ubuf)
    tik_instance.vector_dup(8, const_one, 1, 1, 0, 0)
    tik_instance.vadd(64, x_hi, x_lo, const_one, 2, 1, 1, 0, 8, 8, 0)
    tik_instance.vadd(64, y_hi, y_lo, const_one, 2, 1, 1, 0, 8, 8, 0)

    const_value_fp32 = tik_instance.Tensor(
        "float32", [16], name="const_value", scope=tik.scope_ubuf)
    const_value_int32 = tik_instance.Tensor(
        "int32", [16], name="const_value", scope=tik.scope_ubuf)
    tik_instance.vector_dup(16, const_value_fp32, width - 1, 1, 0, 0)
    tik_instance.vector_dup(16, const_value_int32, width - 1, 1, 0, 0)
    tik_instance.vmin(64, x_lo, x_lo, const_value_int32, 2, 1, 1, 0, 8, 8, 0)
    tik_instance.vmin(64, x_hi, x_hi, const_value_int32, 2, 1, 1, 0, 8, 8, 0)
    tik_instance.vmin(64, x_vec, x_vec, const_value_fp32, 2, 1, 1, 0, 8, 8, 0)

    tik_instance.vector_dup(16, const_value_int32, height - 1, 1, 0, 0)
    tik_instance.vector_dup(16, const_value_fp32, height - 1, 1, 0, 0)
    tik_instance.vmin(64, y_lo, y_lo, const_value_int32, 2, 1, 1, 0, 8, 8, 0)
    tik_instance.vmin(64, y_hi, y_hi, const_value_int32, 2, 1, 1, 0, 8, 8, 0)
    tik_instance.vmin(64, y_vec, y_vec, const_value_fp32, 2, 1, 1, 0, 8, 8, 0)

    tmp_fp32 = tik_instance.Tensor(
        "float32", [128], name="tmp_fp32", scope=tik.scope_ubuf)
    tik_instance.vconv(64, "", tmp_fp32, x_lo, 2, 1, 1, 8, 8)
    tik_instance.vsub(64, x_lo_w, x_vec, tmp_fp32, 2, 1, 1, 1, 8, 8, 8)
    tik_instance.vconv(64, "", tmp_fp32, y_lo, 2, 1, 1, 8, 8)
    tik_instance.vsub(64, y_lo_w, y_vec, tmp_fp32, 2, 1, 1, 1, 8, 8, 8)

    tik_instance.vector_dup(16, const_value_fp32, 1., 1, 0, 0)
    tik_instance.vsub(64, x_hi_w, const_value_fp32, x_lo_w, 2, 1, 0, 1, 8, 0, 8)
    tik_instance.vsub(64, y_hi_w, const_value_fp32, y_lo_w, 2, 1, 0, 1, 8, 0, 8)

    tik_instance.vector_dup(8, const_value_fp32, -1., 1, 0, 0)
    tik_instance.vector_dup(8, const_value_fp32[8], width, 1, 0, 0)
    cmp_mask = tik_instance.Tensor(
        "uint16", (32,), name="cmp_mask", scope=tik.scope_ubuf)

    tik_instance.vcmpv_lt(cmp_mask, raw_x, const_value_fp32, 1, 1, 0, 8, 0)
    tik_instance.vcmpv_gt(cmp_mask[16], raw_x, const_value_fp32[8], 1, 1, 0, 8,
                          0)
    tik_instance.vor(4, cmp_mask, cmp_mask, cmp_mask[16], 1, 1, 1, 1, 8, 8, 8)

    dst_cmp_mask = tik_instance.mov_tensor_to_cmpmask(cmp_mask)
    tik_instance.vsel(64, 0, x_lo_w, dst_cmp_mask, const_zero, x_lo_w, 1, 1, 0,
                      1, 8, 0, 8)
    tik_instance.vsel(64, 0, x_hi_w, dst_cmp_mask, const_zero, x_hi_w, 1, 1, 0,
                      1, 8, 0, 8)

    tik_instance.vcmpv_lt(cmp_mask, raw_x[64], const_value_fp32, 1, 1, 0, 8, 0)
    tik_instance.vcmpv_gt(cmp_mask[16], raw_x[64], const_value_fp32[8], 1, 1, 0,
                          8, 0)
    tik_instance.vor(4, cmp_mask, cmp_mask, cmp_mask[16], 1, 1, 1, 1, 8, 8, 8)

    dst_cmp_mask = tik_instance.mov_tensor_to_cmpmask(cmp_mask)
    tik_instance.vsel(64, 0, x_lo_w[64], dst_cmp_mask, const_zero, x_lo_w[64],
                      1, 1, 0, 1, 8, 0, 8)
    tik_instance.vsel(64, 0, x_hi_w[64], dst_cmp_mask, const_zero, x_hi_w[64],
                      1, 1, 0, 1, 8, 0, 8)

    tik_instance.vmuls(64, x_lo_w, x_lo_w, 1.0 / sample_num_w, 2, 1, 1, 8, 8)
    tik_instance.vmuls(64, x_hi_w, x_hi_w, 1.0 / sample_num_w, 2, 1, 1, 8, 8)

    tik_instance.vector_dup(8, const_value_fp32[8], height, 1, 0, 0)
    tik_instance.vcmpv_lt(cmp_mask, raw_y, const_value_fp32, 1, 1, 0, 8, 0)
    tik_instance.vcmpv_gt(cmp_mask[16], raw_y, const_value_fp32[8], 1, 1, 0, 8,
                          0)
    tik_instance.vor(4, cmp_mask, cmp_mask, cmp_mask[16], 1, 1, 1, 1, 8, 8, 8)

    dst_cmp_mask = tik_instance.mov_tensor_to_cmpmask(cmp_mask)
    tik_instance.vsel(64, 0, y_lo_w, dst_cmp_mask, const_zero, y_lo_w, 1, 1, 0,
                      1, 8, 0, 8)
    tik_instance.vsel(64, 0, y_hi_w, dst_cmp_mask, const_zero, y_hi_w, 1, 1, 0,
                      1, 8, 0, 8)

    tik_instance.vcmpv_lt(cmp_mask, raw_y[64], const_value_fp32, 1, 1, 0, 8, 0)
    tik_instance.vcmpv_gt(cmp_mask[16], raw_y[64], const_value_fp32[8], 1, 1, 0,
                          8, 0)
    tik_instance.vor(4, cmp_mask, cmp_mask, cmp_mask[16], 1, 1, 1, 1, 8, 8, 8)

    dst_cmp_mask = tik_instance.mov_tensor_to_cmpmask(cmp_mask)
    tik_instance.vsel(64, 0, y_lo_w[64], dst_cmp_mask, const_zero, y_lo_w[64],
                      1, 1, 0, 1, 8, 0, 8)
    tik_instance.vsel(64, 0, y_hi_w[64], dst_cmp_mask, const_zero, y_hi_w[64],
                      1, 1, 0, 1, 8, 0, 8)

    tik_instance.vmuls(64, y_lo_w, y_lo_w, 1.0 / sample_num_h, 2, 1, 1, 8, 8)
    tik_instance.vmuls(64, y_hi_w, y_hi_w, 1.0 / sample_num_h, 2, 1, 1, 8, 8)

    return x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo, y_hi, x_ind, y_ind


def roi_align_calc_scale_batch(tik_instance, rois_data_ub, scale, pool_w,
                               pool_h, sample_num):
    """calc one block gradient

    Parameters
    ----------
    tik_instance: class
    rois_data_ub: TVM tensor
    scale: float
    pool_w: float
    pool_h: float
    sample_num: float

    Returns
    -------
    list
    """
    roi_h_fp32 = tik_instance.Tensor(
        "float32", [BATCH_SIZE], name="roi_h_fp32", scope=tik.scope_ubuf)
    roi_w_fp32 = tik_instance.Tensor(
        "float32", [BATCH_SIZE], name="roi_w_fp32", scope=tik.scope_ubuf)

    rois_start_w = rois_data_ub[1, 0]
    rois_start_h = rois_data_ub[2, 0]
    rois_end_w = rois_data_ub[3, 0]
    rois_end_h = rois_data_ub[4, 0]
    tik_instance.vmuls(64, rois_start_w, rois_start_w, scale, 4, 1, 1, 8, 8)
    tik_instance.vadds(64, rois_end_w, rois_end_w, 1, 4, 1, 1, 8, 8)
    tik_instance.vmuls(64, rois_end_w, rois_end_w, scale, 4, 1, 1, 8, 8)

    tik_instance.vsub(64, roi_w_fp32, rois_end_w, rois_start_w,
                      BATCH_SIZE * 2 // 128, 1, 1, 1, 8, 8, 8)
    tik_instance.vsub(64, roi_h_fp32, rois_end_h, rois_start_h,
                      BATCH_SIZE * 2 // 128, 1, 1, 1, 8, 8, 8)

    const_zero = tik_instance.Tensor(
        "float32", [16], name="const_zero", scope=tik.scope_ubuf)
    tik_instance.vector_dup(16, const_zero, 0, 1, 0, 0)

    # compare roi_width adn roi_height to 1
    tik_instance.vmax(64, roi_w_fp32, roi_w_fp32, const_zero,
                      BATCH_SIZE * 2 // 128, 1, 1, 0, 8, 8, 0)
    tik_instance.vmax(64, roi_h_fp32, roi_h_fp32, const_zero,
                      BATCH_SIZE * 2 // 128, 1, 1, 0, 8, 8, 0)

    # Declare roi_bin_size tik_instance.Tensor
    rois_bin_w = tik_instance.Tensor(
        "float32", [BATCH_SIZE], name="roi_bin_w", scope=tik.scope_ubuf)
    rois_bin_h = tik_instance.Tensor(
        "float32", [BATCH_SIZE], name="roi_bin_h", scope=tik.scope_ubuf)
    # bin size
    tik_instance.vmuls(64, rois_bin_w[:], roi_w_fp32[:], 1.0 / pool_w,
                       BATCH_SIZE * 2 // 128, 1, 1, 8, 8)
    tik_instance.vmuls(64, rois_bin_h[:], roi_h_fp32[:], 1.0 / pool_h,
                       BATCH_SIZE * 2 // 128, 1, 1, 8, 8)

    sample_num_w = tik_instance.Tensor(
        "int32", [BATCH_SIZE], name="sample_num_w", scope=tik.scope_ubuf)
    sample_num_h = tik_instance.Tensor(
        "int32", [BATCH_SIZE], name="sample_num_h", scope=tik.scope_ubuf)

    if sample_num > 0:
        tik_instance.vector_dup(64, sample_num_w, sample_num, 2, 1, 8, 0)
        tik_instance.vector_dup(64, sample_num_h, sample_num, 2, 1, 8, 0)
    else:
        tik_instance.vconv(64, 'ceil', sample_num_w, rois_bin_w,
                           BATCH_SIZE * 2 // 128, 1, 1, 8, 8)
        tik_instance.vconv(64, 'ceil', sample_num_h, rois_bin_h,
                           BATCH_SIZE * 2 // 128, 1, 1, 8, 8)

    rois_start_w = tik_instance.Tensor(
        "float32", [BATCH_SIZE], name="roi_h_fp32", scope=tik.scope_ubuf)
    rois_start_h = tik_instance.Tensor(
        "float32", [BATCH_SIZE], name="roi_w_fp32", scope=tik.scope_ubuf)
    rois_index = tik_instance.Tensor(
        "int32", [BATCH_SIZE], name="roi_index", scope=tik.scope_ubuf)
    tik_instance.vadds(64, rois_start_w, rois_data_ub[1, 0], 0, 2, 1, 1, 8, 8)
    tik_instance.vadds(64, rois_start_h, rois_data_ub[2, 0], 0, 2, 1, 1, 8, 8)
    tik_instance.vconv(64, "floor", rois_index, rois_data_ub[0, 0], 2, 1, 1, 8,
                       8)

    return rois_bin_w, rois_bin_h, sample_num_w, sample_num_h, \
           rois_start_w, rois_start_h, rois_index


def convert_rois_data_to5n(tik_instance, rois_data_gm, rois_data_index,
                           rois_num):
    """calc one block gradient

    Parameters
    ----------
    tik_instance: class
    rois_data_gm: TVM tensor
    rois_data_index: TVM tensor
    rois_num: int

    Returns
    -------
    rois_data_ub
    """
    rois_data_ub = tik_instance.Tensor(
        "float32", (5, 128), name="rois_data_ub", scope=tik.scope_ubuf)

    if rois_data_gm.shape[1] == 5:
        rois_data_tmp = tik_instance.Tensor(
            "float32", (128, 5), name="rois_data_tmp", scope=tik.scope_ubuf)
        rois_data_s = tik_instance.Scalar(dtype="float32")
        tik_instance.tensor_mov(rois_data_tmp,
                                rois_data_gm[rois_data_index, 0],
                                '', 1, (4 * rois_num * 5 + 31) // 32, 0, 0)
        with tik_instance.for_range(0, rois_num * 5) as i:
            rois_data_s.set_as(rois_data_tmp[i // 5, i % 5])
            rois_data_ub[[i % 5, i // 5]].set_as(rois_data_s)
    else:
        rois_data_tmp = tik_instance.Tensor(
            "float32", (128, 8), name="rois_data_tmp", scope=tik.scope_ubuf)
        roi_pos = tik_instance.Tensor(
            "float16", [BATCH_SIZE, 8], name="roi_pos", scope=tik.scope_ubuf)
        roi_pos_new = tik_instance.Tensor(
            "float16", [5, BATCH_SIZE],
            name="roi_pos_new",
            scope=tik.scope_ubuf)

        tik_instance.tensor_mov(rois_data_tmp,
                                rois_data_gm[rois_data_index, 0],
                                '', 1, (4 * rois_num * 8) // 32, 0, 0)

        tik_instance.vconv(128, "", roi_pos[0, 0], rois_data_tmp[0, 0],
                           (BATCH_SIZE * 8) // 64, 1, 1, 4, 8)

        tik_instance.vextract(roi_pos_new[0, 0], roi_pos, 8, 0)
        tik_instance.vextract(roi_pos_new[1, 0], roi_pos, 8, 1)
        tik_instance.vextract(roi_pos_new[2, 0], roi_pos, 8, 2)
        tik_instance.vextract(roi_pos_new[3, 0], roi_pos, 8, 3)
        tik_instance.vextract(roi_pos_new[4, 0], roi_pos, 8, 4)

        tik_instance.vconv(128, "", rois_data_ub[0, 0], roi_pos_new[0, 0],
                           (BATCH_SIZE * 10) // 128, 1, 1, 8, 4)

    return rois_data_ub


def roi_align_grad_compute(tik_instance, y_diff, rois_data, x_diff, rois_n,
                           pooled_width, pooled_height, spatial_scale,
                           sample_num, core_bias):
    """calc one block gradient

    Parameters
    ----------
    tik_instance: class
    y_diff: TVM tensor
    rois_data: TVM tensor
    x_diff: TVM tensor
    rois_n: TVM tensor
    pooled_width: float
    pooled_height: float
    spatial_scale: float
    sample_num: float
    core_bias:int

    Returns
    -------
    None
    """
    c1_num = x_diff.shape[1]
    height = x_diff.shape[2]
    width = x_diff.shape[3]
    grid_w_s = tik_instance.Scalar(dtype="float32")
    grid_h_s = tik_instance.Scalar(dtype="float32")
    rois_start_w_s = tik_instance.Scalar(dtype="float32")
    rois_start_h_s = tik_instance.Scalar(dtype="float32")

    sample_num_w_s = tik_instance.Scalar(dtype="int32")
    sample_num_h_s = tik_instance.Scalar(dtype="int32")
    sample_num_w_fp_s = tik_instance.Scalar(dtype="float32")
    sample_num_h_fp_s = tik_instance.Scalar(dtype="float32")

    start_batch = tik_instance.Scalar(dtype="int32")
    start_batch.set_as(core_bias // c1_num)
    end_batch = tik_instance.Scalar(dtype="int32")
    end_batch.set_as((core_bias + rois_n - 1) // c1_num)
    rois_n_num = tik_instance.Scalar(dtype="int32")
    rois_n_num.set_as(end_batch - start_batch + 1)

    const_value_0_127 = tik_instance.Tensor(
        "float32", (128,), name="const_value_0_127", scope=tik.scope_ubuf)
    with tik_instance.for_range(0, 128) as i:
        const_value_0_127[i] = i

    rois_batch_num = (rois_n_num + 127) // 128

    with tik_instance.for_range(0, rois_batch_num) as i:
        # move rois data from DDR to UB
        rois_num = calc_segment(tik_instance, rois_n_num, i, 128)

        rois_data_ub = convert_rois_data_to5n(tik_instance, rois_data,
                                              start_batch + 128 * i, rois_num)
        # calc spatial_scale
        rois_bin_w, rois_bin_h, sample_num_w, \
        sample_num_h, rois_start_w, rois_start_h, rois_index \
            = roi_align_calc_scale_batch(tik_instance, rois_data_ub,
                                         spatial_scale,
                                         pooled_width, pooled_height,
                                         sample_num)

        sample_num_w_fp = tik_instance.Tensor(
            "float32", (128,), name="sample_num_w_fp", scope=tik.scope_ubuf)
        sample_num_h_fp = tik_instance.Tensor(
            "float32", (128,), name="sample_num_h_fp", scope=tik.scope_ubuf)
        tik_instance.vconv(64, "", sample_num_w_fp, sample_num_w, 2, 1, 1, 8, 8)
        tik_instance.vdiv(64, rois_bin_w, rois_bin_w, sample_num_w_fp, 2, 1, 1,
                          1, 8, 8, 8)
        tik_instance.vconv(64, "", sample_num_h_fp, sample_num_h, 2, 1, 1, 8, 8)
        tik_instance.vdiv(64, rois_bin_h, rois_bin_h, sample_num_h_fp, 2, 1, 1,
                          1, 8, 8, 8)

        with tik_instance.for_range(0, rois_num) as j:
            image_index = tik_instance.Scalar(dtype="int32")
            image_index.set_as(rois_index[j])
            grid_w_s.set_as(rois_bin_w[j])
            grid_h_s.set_as(rois_bin_h[j])
            rois_start_w_s.set_as(rois_start_w[j])
            rois_start_h_s.set_as(rois_start_h[j])
            sample_num_w_s.set_as(sample_num_w[j])
            sample_num_h_s.set_as(sample_num_h[j])
            sample_num_w_fp_s.set_as(sample_num_w_fp[j])
            sample_num_h_fp_s.set_as(sample_num_h_fp[j])
            calc_w_num = tik_instance.Scalar(dtype="int32")
            calc_h_num = tik_instance.Scalar(dtype="int32")
            calc_w_num.set_as((sample_num_w_s * pooled_width + 127) // 128)
            calc_h_num.set_as((sample_num_h_s * pooled_height + 127) // 128)

            start_c1, end_c1 = calc_c1_segment(tik_instance, core_bias, rois_n,
                                               ((i * 128) + j), c1_num)

            with tik_instance.for_range(0, calc_h_num) as k:
                start_h_s = rois_start_h_s + grid_h_s * k * 128
                line_num = calc_segment(tik_instance,
                                        sample_num_h_s * pooled_height, k, 128)
                with tik_instance.for_range(0, calc_w_num) as w_index:
                    row_num = calc_segment(tik_instance,
                                           sample_num_w_s * pooled_width,
                                           w_index, 128)

                    start_w_s = rois_start_w_s + grid_w_s * w_index * 128
                    x_lo_w, x_hi_w, y_lo_w, \
                    y_hi_w, x_lo, x_hi, y_lo, y_hi, x_ind, y_ind = \
                        roi_align_calc_grid(tik_instance, k, w_index,
                                            const_value_0_127, grid_w_s,
                                            grid_h_s, sample_num_w_fp_s,
                                            sample_num_h_fp_s,
                                            start_w_s, start_h_s,
                                            height, width)

                    roi_align_calc_grad_block(
                        tik_instance, line_num,
                        row_num, x_lo_w, x_hi_w, y_lo_w,
                        y_hi_w, x_lo, x_hi, y_lo, y_hi, y_diff,
                        start_batch + (i * 128) + j, x_ind, y_ind, x_diff,
                        image_index, start_c1, end_c1)


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


def roi_align_grad_compute_multicore(
        grad_shape, rois_shape, rois_n, pooled_width, pooled_height,
        spatial_scale, sample_num, output_grad_shape, kernel_name):
    """calc one block gradient

    Parameters
    ----------
    grad_shape: class
    rois_shape: int
    rois_n: int
    pooled_width: float
    pooled_height: float
    spatial_scale: float
    sample_num: float
    output_grad_shape: int
    kernel_name: str

    Returns
    -------
    tik_instance
    """
    tik_instance = tik.Tik()

    input_grad = tik_instance.Tensor(
        "float32", grad_shape, name="input_grad", scope=tik.scope_gm)
    rois_data = tik_instance.Tensor(
        "float32", rois_shape, name="rois_data", scope=tik.scope_gm)
    output_grad = tik_instance.Tensor(
        "float32",
        output_grad_shape,
        name="output_grad",
        scope=tik.scope_gm,
        is_atomic_add=True)

    c1_num = grad_shape[1]
    core_counts = \
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
    if (rois_n * c1_num) > core_counts:
        core_num = core_counts
    else:
        core_num = rois_n * c1_num

    core_rois_n = (rois_n * c1_num) // core_num
    core_tail = (rois_n * c1_num) % core_num

    with tik_instance.for_range(0, core_num, block_num=core_num) as core_index:
        block_rois_n = tik_instance.Scalar(
            dtype="int32", init_value=core_rois_n)
        core_bias = tik_instance.Scalar(dtype="int32")
        with tik_instance.if_scope(core_index == 0):
            block_rois_n.set_as(block_rois_n + core_tail)
            core_bias.set_as(0)
        with tik_instance.else_scope():
            core_bias.set_as((core_rois_n * core_index) + core_tail)

        roi_align_grad_compute(tik_instance, input_grad, rois_data, output_grad,
                               block_rois_n, pooled_width, pooled_height,
                               spatial_scale, sample_num, core_bias)

    tik_instance.BuildCCE(
        kernel_name=kernel_name,
        inputs=[input_grad, rois_data],
        outputs=[output_grad])

    return tik_instance


# pylint: disable=unused-argument
def roi_align_grad(y_diff,
                   rois,
                   rois_n,
                   x_diff,
                   xdiff_shape,
                   pooled_width,
                   pooled_height,
                   spatial_scale,
                   sample_num,
                   kernel_name="roi_align_grad"):
    """
    algorithm: floor
    calculating roi_align_grad,
    the type of input_data is "float32"

    Parameters
    ----------
    y_diff: dict
        dict with keys(shape and dtype) of y_diff
    rois: dict
        dict with keys(shape and dtype) of rois
    rois_n: dict
        dict with keys(shape and dtype) of rois_n
    x_diff: dict
        dict with keys(shape and dtype) of x_diff
    xdiff_shape: list
        list xdiff_shape
    pooled_width: int
        pooled_width
    pooled_height: int
        pooled_height
    spatial_scale: float
        spatial_scale
    sample_num: int
        sample_num
    kernel_name: str
        kernel name

    Returns
    -------
    tik_instance: tik_instance
    """
    input_list = [y_diff, rois]
    for input_data in input_list:
        input_shape = input_data.get("shape")
        input_dtype = input_data.get("dtype").lower()
        util.check_shape_rule(input_shape)
        util.check_kernel_name(kernel_name)
        check_list = ("float32",)

        util.check_dtype_rule(input_dtype, check_list)

    grad_shape = y_diff.get("shape")
    if len(grad_shape) != 5:
        raise RuntimeError("the length of grad_sharp must be 5, while it is: %d"
                           % len(grad_shape))

    rois_shape = rois.get("shape")
    if rois_shape[1] != 5:
        raise RuntimeError("the rois_shape must be n * 5")

    output_grad_shape = x_diff.get("shape")

    tik_instance = roi_align_grad_compute_multicore(
        grad_shape, rois_shape, rois_shape[0], pooled_width, pooled_height,
        spatial_scale, sample_num, output_grad_shape, kernel_name)

    return tik_instance

