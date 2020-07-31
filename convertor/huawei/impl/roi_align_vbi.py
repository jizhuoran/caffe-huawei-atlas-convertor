#!/usr/bin/env python
# -*- coding:utf-8 -*-
# pylint: disable=R0912, R0914, R0913, R0915, C0302
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

roi_align_vbi
"""
from te import tik
from topi.cce import util


TWO = 2
FIVE = 5

NoneType = type(None)
POOL_H = 7
POOL_W = 7
SAMPLING_RATIO = 1
NUM_SAMPLING_W = POOL_W * SAMPLING_RATIO
VBI_NUM_BLOCKS_ONEROW = (POOL_H * POOL_W + 7) // 8
VBI_NUM_ELEMENTS_ONEROW = VBI_NUM_BLOCKS_ONEROW * 8
NUM_ELEMENTS_ONEROW = VBI_NUM_ELEMENTS_ONEROW * 2
NUM_ELMENTS_ONEBIN = SAMPLING_RATIO * SAMPLING_RATIO * 4
VBI_TOTAL_ELEMENTS = VBI_NUM_ELEMENTS_ONEROW * NUM_ELMENTS_ONEBIN

STRIDE_H = POOL_H * SAMPLING_RATIO - 1
STRIDE_W = POOL_W * SAMPLING_RATIO - 1
C0SIZE = 16
BYTES = 2

BLOCK_DIM = 8

ROINUM_LIMIT = 128
FM_BUFFER_SIZE_LIMIT = 80 * 1024
MAX_NUM_GRIDW = 128

ROI_PARA_UNIT = 64
ROI_BATCH_UNIT = 64
L1_ADDR_GRID_PARA = 0


def roi_align_perf_scale(tik_instance, rois):
    '''
    calculate the pos of roi box and  wide and height of grid
    :param tik_instance:
    :param rois: the coordinates of the roi box
    :return: the pos of roi box and  wide and height of grid
    '''
    repeat = ROINUM_LIMIT // 16
    tmp_buf = tik_instance.Tensor("float16", [4, ROINUM_LIMIT], \
                                  name="tmp_buf",
                                  scope=tik.scope_ubuf)
    # x1, y1, x2, y2
    tik_instance.vreduce(128, tmp_buf[0, 0], rois, 3, 4, 1, 8, 0, 0,
                         None, "normal")
    tik_instance.vreduce(128, tmp_buf[1, 0], rois, 4, 4, 1, 8, 0, 0,
                         None, "normal")
    tik_instance.vreduce(128, tmp_buf[2, 0], rois, 5, 4, 1, 8, 0, 0,
                         None, "normal")
    tik_instance.vreduce(128, tmp_buf[3, 0], rois, 6, 4, 1, 8, 0, 0,
                         None, "normal")

    rois_fp32 = tik_instance.Tensor("float32", \
                                    [4, repeat * 16],
                                    name="rois_fp32",
                                    scope=tik.scope_ubuf)
    # 2*(xStartFp32, yStartFp32, xEndFp32, yEndFp32)
    tik_instance.vconv(64, "", rois_fp32, tmp_buf, 2 * 4, 1, 1, 8, 4)
    # height and width of each RoI
    tik_instance.vsub(64, rois_fp32[3, 0], rois_fp32[3, 0],
                      rois_fp32[1, 0], 2, 1, 1, 1, 8, 8, 8)
    tik_instance.vsub(64, rois_fp32[2, 0], rois_fp32[2, 0],
                      rois_fp32[0, 0], 2, 1, 1, 1, 8, 8, 8)
    # roi_w fp16 tmp_buf[2][0]
    # roi_h fp16 tmp_buf[3][0]
    roi_w = tmp_buf[2, 0]
    roi_h = tmp_buf[3, 0]
    tik_instance.vconv(64, "", roi_w, rois_fp32[2, 0], 2, 1, 1, 4, 8)
    tik_instance.vconv(64, "", roi_h, rois_fp32[3, 0], 2, 1, 1, 4, 8)
    # roi_height = max(roi_end_h - roi_start_h, 1.0)
    cmp_buf = tik_instance.Tensor("float16", [128, ], name="cmp_buf",
                                  scope=tik.scope_ubuf)
    one_fp16 = tik_instance.Scalar(dtype="float16", init_value=1.0)

    tik_instance.vector_dup(128, cmp_buf, one_fp16, 1, 1, 8)
    cmpmask = tik_instance.vcmp_lt(128, roi_h, cmp_buf, 1, 1)


    tik_instance.vsel(128, 0, roi_h, cmpmask, cmp_buf, roi_h, 1, 1, 1,
                      1, 8, 8, 8)
    # roi_width = max(roi_end_w - roi_start_w, 1.0)
    cmpmask = tik_instance.vcmp_lt(128, roi_w, cmp_buf, 1, 1)
    tik_instance.vsel(128, 0, roi_w, cmpmask, cmp_buf, roi_w, 1, 1, 1,
                      1, 8, 8, 8)
    grid_hw_fp32 = tik_instance.Tensor("float32", [2, 128],
                                       name="rois_fp32", \
                                       scope=tik.scope_ubuf)
    grid_h_fp32 = grid_hw_fp32[0, 0]
    grid_w_fp32 = grid_hw_fp32[1, 0]
    tmp_buf1 = tik_instance.Tensor("float32", [2, ROINUM_LIMIT],
                                   name="tmp_buf1", \
                                   scope=tik.scope_ubuf)
    tik_instance.vconv(64, "", tmp_buf1[0, 0], roi_h, 2, 1, 1, 8, 4)
    tik_instance.vconv(64, "", tmp_buf1[1, 0], roi_w, 2, 1, 1, 8, 4)

    scale_grid_hw = \
        tik_instance.Scalar(dtype="float32",
                            init_value=1.0 / (POOL_H * SAMPLING_RATIO))
    tik_instance.vmuls(64, grid_h_fp32, tmp_buf1[0, 0], scale_grid_hw,
                       2, 1, 1, 8, 8)
    tik_instance.vmuls(64, grid_w_fp32, tmp_buf1[1, 0], scale_grid_hw,
                       2, 1, 1, 8, 8)
    return rois_fp32, grid_hw_fp32


def roi_align_perf_gengrid_fp32(tik_instance, curr_roi, rois_fp32,
                                grid_hw_fp32, \
                                feature_shape, index_array_fp32):
    '''
    :param tik_instance:
    :param curr_roi: the number of roi box
    :param rois_fp32: the position of roi box
    :param grid_hw_fp32: the wide and  height  of grid in roi box
    :param feature_shape: the shape of the input featuremap
    :param index_array_fp32: a index, corresponding to center of grid
    :return: the position of 4 pixels around the center of gird ;
            and distance to them:
            x_low_int y_low_int x_high_int y_high_int
    '''
    # lx ly hx hy
    point_weights_fp32 = tik_instance.Tensor(
        "float32", [4, ROI_PARA_UNIT], name="point_weights_fp32",
        scope=tik.scope_ubuf)

    point_positions_int32 = tik_instance.Tensor(
        "int32", [4, ROI_PARA_UNIT], name="point_positions_int32",
        scope=tik.scope_ubuf)
    point_positions_fp32 = tik_instance.Tensor(
        "float32", [4, ROI_PARA_UNIT], name="pointPositionFp32",
        scope=tik.scope_ubuf)
    delta_w = tik_instance.Scalar(dtype="float32",
                                  init_value=grid_hw_fp32[
                                      1, curr_roi])
    delta_h = tik_instance.Scalar(dtype="float32",
                                  init_value=grid_hw_fp32[
                                      0, curr_roi])

    w_start = tik_instance.Scalar(dtype="float32",
                                  init_value=rois_fp32[0, curr_roi])

    h_start = tik_instance.Scalar(dtype="float32",
                                  init_value=rois_fp32[1, curr_roi])
    height = feature_shape[2]
    width = feature_shape[3]

    point_positions = tik_instance.Tensor(
        "float32", [2, ROI_PARA_UNIT], name="point_positions",
        scope=tik.scope_ubuf)
    x_pos_fp32 = point_positions[0, 0]
    y_pos_fp32 = point_positions[1, 0]


    tik_instance.vmuls(64, x_pos_fp32, index_array_fp32[0, ], delta_w,
                       1, 1, 1, 8, 8)
    tik_instance.vmuls(64, y_pos_fp32, index_array_fp32[0, ], delta_h,
                       1, 1, 1, 8, 8)
    tik_instance.vadds(64, x_pos_fp32, x_pos_fp32, w_start, 1, 1, 1,
                       8, 8)
    tik_instance.vadds(64, y_pos_fp32, y_pos_fp32, h_start, 1, 1, 1,
                       8, 8)

    # need to substract 0.5 in TensorFlow
    neg_point_five = tik_instance.Scalar(dtype="float32",
                                         init_value=-0.5)
    tik_instance.vadds(64, x_pos_fp32, x_pos_fp32, neg_point_five, 1,
                       1, 1, 8, 8)
    tik_instance.vadds(64, y_pos_fp32, y_pos_fp32, neg_point_five, 1,
                       1, 1, 8, 8)
    x_low_int = point_positions_int32[0, 0]
    y_low_int = point_positions_int32[1, 0]
    x_high_int = point_positions_int32[2, 0]
    y_high_int = point_positions_int32[3, 0]
    tik_instance.vconv(64, "floor", x_low_int, x_pos_fp32, 1, 1, 1, 8,
                       8)
    tik_instance.vconv(64, "floor", y_low_int, y_pos_fp32, 1, 1, 1, 8,
                       8)
    x_low_fp32 = point_positions_fp32[0, 0]
    y_low_fp32 = point_positions_fp32[1, 0]
    x_high_fp32 = point_positions_fp32[2, 0]
    y_high_fp32 = point_positions_fp32[3, 0]
    tik_instance.vconv(64, "", x_low_fp32, x_low_int, 1, 1, 1, 8, 8)
    tik_instance.vconv(64, "", y_low_fp32, y_low_int, 1, 1, 1, 8, 8)
    point_five = tik_instance.Scalar(dtype="float32", init_value=0.5)
    one = tik_instance.Scalar(dtype="float32", init_value=1.0)
    neg_one = tik_instance.Scalar(dtype="float32", init_value=-1.0)
    zero = tik_instance.Scalar(dtype="float32", init_value=0.0)
    tik_instance.vadds(64, x_high_fp32, x_low_fp32, point_five, 1, 1,
                       1, 8, 8)
    tik_instance.vadds(64, y_high_fp32, y_low_fp32, point_five, 1, 1,
                       1, 8, 8)
    tik_instance.vconv(64, "ceil", x_high_int, x_high_fp32, 1, 1, 1,
                       8, 8)
    tik_instance.vconv(64, "ceil", y_high_int, y_high_fp32, 1, 1, 1,
                       8, 8)
    tik_instance.vadds(64, x_high_fp32, x_low_fp32, one, 1, 1, 1, 8,
                       8)
    tik_instance.vadds(64, y_high_fp32, y_low_fp32, one, 1, 1, 1, 8,
                       8)
    # get xhigh/yhigh for all bins, in x_high_int, y_high_int
    # lx, ly, hx, hy are the weights for interpolation
    lx_fp32 = point_weights_fp32[0, 0]
    ly_fp32 = point_weights_fp32[1, 0]
    hx_fp32 = point_weights_fp32[2, 0]
    hy_fp32 = point_weights_fp32[3, 0]
    tik_instance.vsub(64, lx_fp32, x_pos_fp32, x_low_fp32, 1, 1, 1, 1,
                      8, 8, 8)
    tik_instance.vsub(64, ly_fp32, y_pos_fp32, y_low_fp32, 1, 1, 1, 1,
                      8, 8, 8)
    tik_instance.vsub(64, hx_fp32, x_high_fp32, x_pos_fp32, 1, 1, 1,
                      1, 8, 8, 8)
    tik_instance.vsub(64, hy_fp32, y_high_fp32, y_pos_fp32, 1, 1, 1,
                      1, 8, 8, 8)
    pos_cmp = tik_instance.Tensor("float32", [4, 64], name="pos_cmp",
                                  scope=tik.scope_ubuf)
    lx_pos_cmp = pos_cmp[0, 0]
    ly_pos_cmp = pos_cmp[1, 0]
    hx_pos_cmp = pos_cmp[2, 0]
    hy_pos_cmp = pos_cmp[3, 0]
    neg_pos_cmp = tik_instance.Tensor("float32", [4, 64],
                                      name="neg_pos_cmp",
                                      scope=tik.scope_ubuf)
    lx_neg_cmp = neg_pos_cmp[0, 0]
    ly_neg_cmp = neg_pos_cmp[1, 0]
    hx_neg_cmp = neg_pos_cmp[2, 0]
    hy_neg_cmp = neg_pos_cmp[3, 0]
    end_pos_cmp = tik_instance.Tensor("float32", [4, 64],
                                      name="end_pos_cmp",
                                      scope=tik.scope_ubuf)
    x_low_pos_cmp = end_pos_cmp[0, 0]
    y_low_pos_cmp = end_pos_cmp[1, 0]
    x_high_pos_cmp = end_pos_cmp[2, 0]
    h_high_pos_cmp = end_pos_cmp[3, 0]
    neg_end_pos_cmp = tik_instance.Tensor("float32", [4, 64],
                                          name="neg_end_pos_cmp", \
                                          scope=tik.scope_ubuf)
    x_low_neg_cmp = neg_end_pos_cmp[0, 0]
    y_low_neg_cmp = neg_end_pos_cmp[1, 0]
    x_high_neg_cmp = neg_end_pos_cmp[2, 0]
    y_high_neg_cmp = neg_end_pos_cmp[3, 0]
    # temporary data for comparision
    tik_instance.vadds(64, lx_pos_cmp, lx_fp32, one, 1, 1, 1, 8, 8)
    tik_instance.vadds(64, ly_pos_cmp, ly_fp32, one, 1, 1, 1, 8, 8)
    tik_instance.vadds(64, hx_pos_cmp, hx_fp32, one, 1, 1, 1, 8, 8)
    tik_instance.vadds(64, hy_pos_cmp, hy_fp32, one, 1, 1, 1, 8, 8)
    tik_instance.vadds(64, lx_neg_cmp, lx_fp32, neg_one, 1, 1, 1, 8,
                       8)
    tik_instance.vadds(64, ly_neg_cmp, ly_fp32, neg_one, 1, 1, 1, 8,
                       8)
    tik_instance.vadds(64, hx_neg_cmp, hx_fp32, neg_one, 1, 1, 1, 8,
                       8)
    tik_instance.vadds(64, hy_neg_cmp, hy_fp32, neg_one, 1, 1, 1, 8,
                       8)
    tik_instance.vadds(64, x_low_pos_cmp, x_low_fp32, one, 1, 1, 1, 8,
                       8)
    tik_instance.vadds(64, y_low_pos_cmp, y_low_fp32, one, 1, 1, 1, 8,
                       8)
    tik_instance.vadds(64, x_high_pos_cmp, x_high_fp32, one, 1, 1, 1,
                       8, 8)
    tik_instance.vadds(64, h_high_pos_cmp, y_high_fp32, one, 1, 1, 1,
                       8, 8)
    tik_instance.vadds(64, x_low_neg_cmp, x_low_fp32, neg_one, 1, 1,
                       1, 8, 8)
    tik_instance.vadds(64, y_low_neg_cmp, y_low_fp32, neg_one, 1, 1,
                       1, 8, 8)
    tik_instance.vadds(64, x_high_neg_cmp, x_high_fp32, neg_one, 1, 1,
                       1, 8, 8)
    tik_instance.vadds(64, y_high_neg_cmp, y_high_fp32, neg_one, 1, 1,
                       1, 8, 8)
    # compare lx ly with 0 and 1
    # if lx > 1:
    #   lx = lx - 1
    #   hx = hx + 1
    #   xlow = xlow + 1
    #   xhigh = xhigh + 1
    cmp_const_fp32 = tik_instance.Tensor("float32", [64, ],
                                         name="cmp_const",
                                         scope=tik.scope_ubuf)
    cmp_const = cmp_const_fp32[0, ]
    tik_instance.vector_dup(64, cmp_const, one, 1, 1, 8)  # float32
    cmpmask = tik_instance.vcmp_gt(64, lx_fp32, cmp_const[0, ], 1, 1)
    tik_instance.vsel(64, 0, lx_fp32, cmpmask, lx_neg_cmp, lx_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, hx_fp32, cmpmask, hx_pos_cmp, hx_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, x_low_fp32, cmpmask, x_low_pos_cmp,
                      x_low_fp32, 1, 1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, x_high_fp32, cmpmask, x_high_pos_cmp,
                      x_high_fp32, \
                      1, 1, 1, 1, 8, 8, 8)
    # if lx < 0:
    #   lx = lx + 1
    #   hx = hx - 1
    #   xlow = xlow - 1
    #   xhigh = xhigh - 1
    tik_instance.vector_dup(64, cmp_const, zero, 1, 1, 8)
    cmpmask = tik_instance.vcmp_lt(64, lx_fp32, cmp_const[0, ], 1, 1)
    tik_instance.vsel(64, 0, lx_fp32, cmpmask, lx_pos_cmp, lx_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, hx_fp32, cmpmask, hx_neg_cmp, hx_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, x_low_fp32, cmpmask, x_low_neg_cmp,
                      x_low_fp32, 1, 1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, x_high_fp32, cmpmask, x_high_neg_cmp,
                      x_high_fp32, \
                      1, 1, 1, 1, 8, 8, 8)
    # if ly > 1:
    #   ly = ly - 1
    #   hy = hy + 1
    #   ylow = ylow + 1
    #   yhigh = yhigh + 1
    tik_instance.vector_dup(64, cmp_const, one, 1, 1, 8)
    cmpmask = tik_instance.vcmp_gt(64, ly_fp32, cmp_const, 1, 1)
    tik_instance.vsel(64, 0, ly_fp32, cmpmask, ly_neg_cmp, ly_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, hy_fp32, cmpmask, hy_pos_cmp, hy_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, y_low_fp32, cmpmask, y_low_pos_cmp,
                      y_low_fp32, 1, 1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, y_high_fp32, cmpmask, h_high_pos_cmp,
                      y_high_fp32, \
                      1, 1, 1, 1, 8, 8, 8)
    # if ly < 0:
    #   ly = ly + 1
    #   hy = hy - 1
    #   ylow = ylow - 1
    #   yhigh = yhigh - 1
    tik_instance.vector_dup(64, cmp_const, zero, 1, 1, 8)
    cmpmask = tik_instance.vcmp_lt(64, ly_fp32, cmp_const, 1, 1)
    tik_instance.vsel(64, 0, ly_fp32, cmpmask, ly_pos_cmp, ly_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, hy_fp32, cmpmask, hy_neg_cmp, hy_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, y_low_fp32, cmpmask, y_low_neg_cmp,
                      y_low_fp32, 1, 1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, y_high_fp32, cmpmask, y_high_neg_cmp,
                      y_high_fp32, \
                      1, 1, 1, 1, 8, 8, 8)
    # update x_low_int, y_low_int, x_high_int, y_high_int
    cmp_buf_fp32 = tik_instance.Tensor("float32", [64, ],
                                       name="cmp_buf",
                                       scope=tik.scope_ubuf)
    cmp_buf = cmp_buf_fp32[0, ]
    tik_instance.vadds(64, cmp_buf, x_low_fp32, point_five, 1, 1, 1,
                       8, 8)
    tik_instance.vconv(64, "floor", x_low_int, cmp_buf, 1, 1, 1, 8, 8)
    tik_instance.vconv(64, "ceil", x_high_int, cmp_buf, 1, 1, 1, 8, 8)
    tik_instance.vadds(64, cmp_buf, y_low_fp32, point_five, 1, 1, 1,
                       8, 8)
    tik_instance.vconv(64, "floor", y_low_int, cmp_buf, 1, 1, 1, 8, 8)
    tik_instance.vconv(64, "ceil", y_high_int, cmp_buf, 1, 1, 1, 8, 8)
    # below are the conditions for TensorFlow:
    # if x_low > W-1:
    # hx = 0
    # lx = 0
    cmp_xy_fp32 = tik_instance.Tensor("float32", [64, ],
                                      name="cmp_xy",
                                      scope=tik.scope_ubuf)
    cmp_xy = cmp_xy_fp32[0, ]
    tik_instance.vector_dup(64, cmp_buf, width, 1, 1, 8)
    tik_instance.vadds(64, cmp_buf, cmp_buf, neg_one, 1, 1, 1, 8, 8)
    tik_instance.vadds(64, cmp_xy, x_low_fp32, point_five, 1, 1, 1, 8,
                       8)
    cmpmask = tik_instance.vcmp_gt(64, cmp_xy, cmp_buf, 1, 1)
    tik_instance.vector_dup(64, cmp_const, zero, 1, 1, 8)
    tik_instance.vsel(64, 0, hx_fp32, cmpmask, cmp_const, hx_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, lx_fp32, cmpmask, cmp_const, lx_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    # if y_low > H-1:
    # hy = 0
    # ly = 0
    tik_instance.vector_dup(64, cmp_buf, height, 1, 1, 8)
    tik_instance.vadds(64, cmp_buf, cmp_buf, neg_one, 1, 1, 1, 8, 8)
    tik_instance.vadds(64, cmp_xy, y_low_fp32, point_five, 1, 1, 1, 8,
                       8)
    cmpmask = tik_instance.vcmp_gt(64, cmp_xy, cmp_buf, 1, 1)
    tik_instance.vector_dup(64, cmp_const, zero, 1, 1, 8)
    tik_instance.vsel(64, 0, hy_fp32, cmpmask, cmp_const, hy_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, ly_fp32, cmpmask, cmp_const, ly_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    # if x_low < 0:
    # hx = 0
    # lx = 0
    tik_instance.vector_dup(64, cmp_buf, zero, 1, 1, 8)
    tik_instance.vector_dup(64, cmp_const, zero, 1, 1, 8)
    tik_instance.vadds(64, cmp_xy, x_low_fp32, point_five, 1, 1, 1, 8,
                       8)
    cmpmask = tik_instance.vcmp_lt(64, cmp_xy, cmp_buf, 1, 1)
    tik_instance.vsel(64, 0, hx_fp32, cmpmask, cmp_const, hx_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, lx_fp32, cmpmask, cmp_const, lx_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    # if y_low < 0:
    # hy = 0
    # ly = 0
    tik_instance.vector_dup(64, cmp_buf, zero, 1, 1, 8)
    tik_instance.vector_dup(64, cmp_const, zero, 1, 1, 8)
    tik_instance.vadds(64, cmp_xy, y_low_fp32, point_five, 1, 1, 1, 8,
                       8)
    cmpmask = tik_instance.vcmp_lt(64, cmp_xy, cmp_buf, 1, 1)
    tik_instance.vsel(64, 0, hy_fp32, cmpmask, cmp_const, hy_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    tik_instance.vsel(64, 0, ly_fp32, cmpmask, cmp_const, ly_fp32, 1,
                      1, 1, 1, 8, 8, 8)
    return point_positions_int32, point_weights_fp32


def get_delta_addresses(tik_instance, point_positions_int32, width):
    '''
    :param tik_instance:
    :param point_positions_int32:  the position of 4 pixels around;\
                                    delta_x_low deltaY_Low
                                    delta_x_high deltaY_High
    :param width: the wide of the roibox
    :return: a tmp variable  point_distance_int32

    get feature map delta address
    Get delta address for xlow/xhigh/ylow/yhigh of every grid
    (for preparing VBI addresses).
    deltaX = (w-wstart)
    deltaAddr_X = (w-wstart)*C0 (input format: C1HWC0)
    deltaAddr_Y = (h-hstart)*RoiWidth*C0 (input format: C1HWC0)
    '''

    index_height = 0
    x_start_value = tik_instance.Scalar(dtype="int32",
                                        init_value=point_positions_int32[0, 0])
    neg_xstart = tik_instance.Scalar(dtype="int32",
                                     init_value=-1 * x_start_value)
    y_start_value = \
        tik_instance.Scalar(dtype="int32",
                            init_value=point_positions_int32[1, \
                                                            index_height * \
                                                            SAMPLING_RATIO])
    neg_ystart = tik_instance.Scalar(dtype="int32",
                                     init_value=-1 * y_start_value)
    c_0 = C0SIZE * BYTES
    h_offset = tik_instance.Scalar(dtype="int32",
                                   init_value=c_0 * width)
    point_distance_int32 = tik_instance.Tensor( \
        "int32", [4, 64], name="point_distance_int32",
        scope=tik.scope_ubuf)

    tik_instance.vadds(64, point_distance_int32[0, 0],
                       point_positions_int32[0, 0], \
                       neg_xstart, 1, 1, 1, 8, 8)
    tik_instance.vadds(64, point_distance_int32[1, 0],
                       point_positions_int32[1, \
                                             index_height * \
                                             SAMPLING_RATIO],
                       neg_ystart,
                       1, 1, 1, 8, 8)
    tik_instance.vadds(64, point_distance_int32[2, 0],
                       point_positions_int32[2, 0], \
                       neg_xstart, 1, 1, 1, 8, 8)
    tik_instance.vadds(64, point_distance_int32[3, 0],
                       point_positions_int32[3, \
                                             index_height * \
                                             SAMPLING_RATIO],
                       neg_ystart,
                       1, 1, 1, 8, 8)
    tik_instance.vmuls(64, point_distance_int32[0, 0],
                       point_distance_int32[0, 0], \
                       c_0, 2, 1, 1, 16, 16)
    tik_instance.vmuls(64, point_distance_int32[1, 0],
                       point_distance_int32[1, 0], \
                       h_offset, 2, 1, 1, 16, 16)
    return point_distance_int32


def get_vbi_addr_1x1grid(tik_instance, point_distance_int32):
    '''
    :param tik_instance:
    :param point_distance_int32: the tmp variable to rearranged
            address of 4 pixels
    :return:Rearranged address(Xn)
    '''

    point_ph_addr = \
        tik_instance.Tensor("int32", [4, NUM_ELEMENTS_ONEROW],
                            name="point_ph_addr",
                            scope=tik.scope_ubuf)
    point_ph_addr_res = \
        tik_instance.Tensor("int32", [4, NUM_ELEMENTS_ONEROW],
                            name="point_ph_addr_res",
                            scope=tik.scope_ubuf)
    point_ph_addr_float = \
        tik_instance.Tensor("float32", [4, NUM_ELEMENTS_ONEROW],
                            name="point_ph_addr_float",
                            scope=tik.scope_ubuf)

    point_ph_addr_float_res = \
        tik_instance.Tensor("float32", [4, NUM_ELEMENTS_ONEROW],
                            name="point_ph_addr_float_res",
                            scope=tik.scope_ubuf)

    tik_instance.vector_dup(56, point_ph_addr[0, 0], -2, 8, 1, 7)
    tik_instance.vector_dup(56, point_ph_addr_res[0, 0], 0, 8, 1, 7)
    tik_instance.vector_dup(56, point_ph_addr_float[0, 0], 0, 8, 1, 7)
    tik_instance.vector_dup(56, point_ph_addr_float_res[0, 0], 0, 8,
                            1, 7)

    delta_x_low = point_distance_int32[0, 0]
    delta_x_high = point_distance_int32[2, 0]

    num_sampling_block = NUM_SAMPLING_W + 1
    for pool_h in range(0, POOL_H):
        delta_y_ph0_g0 = \
            tik_instance.Scalar(dtype="int32",
                                init_value=point_distance_int32[1,
                                                                pool_h * \
                                                            SAMPLING_RATIO])
        delta_y_ph0_g1 = tik_instance.Scalar(dtype="int32", \
                                             init_value=point_distance_int32[
                                                 3, pool_h * SAMPLING_RATIO])

        fm_start_addr = 0
        h_start_addr = tik_instance.Scalar(dtype="int32", \
                                           init_value=fm_start_addr + \
                                                      delta_y_ph0_g0)

        tik_instance.vadds(NUM_SAMPLING_W, point_ph_addr[
            0, pool_h * num_sampling_block], \
                           delta_x_low, h_start_addr, 1, 1, 1, 8, 8)
        tik_instance.vadds(NUM_SAMPLING_W, point_ph_addr[
            1, pool_h * num_sampling_block], \
                           delta_x_high, h_start_addr, 1, 1, 1, 8, 8)

        h_start_addr.set_as(fm_start_addr + delta_y_ph0_g1)

        tik_instance.vadds(NUM_SAMPLING_W, point_ph_addr[
            2, pool_h * num_sampling_block], \
                           delta_x_low, h_start_addr, 1, 1, 1, 8, 8)
        tik_instance.vadds(NUM_SAMPLING_W, point_ph_addr[
            3, pool_h * num_sampling_block], \
                           delta_x_high, h_start_addr, 1, 1, 1, 8, 8)

    tik_instance.vconv(56, '', point_ph_addr_float, point_ph_addr, 8,
                       1, 1, 7, 7)

    mask_reduce = tik_instance.Tensor("uint32", [8, ], \
                                      name="mask_reduce",
                                      scope=tik.scope_ubuf)

    cmp_vct_1 = tik_instance.Tensor("float32", [1, 56], \
                                    name="cmp_vct_1",
                                    scope=tik.scope_ubuf)

    tik_instance.vector_dup(56, cmp_vct_1, -1, 1, 1, 7)

    cmpmask = tik_instance.vcmp_gt(56, point_ph_addr_float, cmp_vct_1,
                                   1, 1)

    tik_instance.mov_cmpmask_to_tensor(
        mask_reduce.reinterpret_cast_to("uint64"),
        cmpmask)
    with tik_instance.for_range(0, 4) as i:
        tik_instance.vreduce(56, point_ph_addr_float_res[i, 0], \
                             point_ph_addr_float[i, 0], \
                             mask_reduce, 1, 1, 7, 0, 0, None,
                             "normal")
        tik_instance.vconv(56, 'to-zero', point_ph_addr_res[i, 0], \
                           point_ph_addr_float_res[i, 0], 1, 1, 1, 7,
                           7)

    vbi_addr = tik_instance.Tensor("int32",
                                   [VBI_NUM_ELEMENTS_ONEROW * 4, ], \
                                   name="vbi_addr",
                                   scope=tik.scope_ubuf)
    tik_instance.vector_dup(56, vbi_addr[0, ], 0, 4, 1, 7)
    horizontal_repeat = NUM_ELMENTS_ONEBIN
    num_burst = VBI_NUM_BLOCKS_ONEROW

    dst_stride = horizontal_repeat - 1
    for i in range(0, 4):
        tik_instance.data_move(vbi_addr[i * 8],
                               point_ph_addr_res[i, 0], 0, \
                               num_burst, 1, 0, dst_stride)

    tik_instance.vector_dup(7, vbi_addr[209, ], 0, 1, 1, 8)
    return vbi_addr


def get_vbi_weights_1x1grid(tik_instance, point_weights_fp32):
    '''
    :param tik_instance:
    :param point_weights_fp32: the distance to 4 pixels around lx ly hx hy
    :return:vbi_weights ,the rearranged weights(xm), hx*hy  lx*hy  hx*ly lx*ly
    '''
    vbi_tmp_weights_res = \
        tik_instance.Tensor("float32", [4, NUM_ELEMENTS_ONEROW],\
                            name="vbi_tmp_weights_res",
                            scope=tik.scope_ubuf)

    vbi_tmp_weights = tik_instance.Tensor("float32",
                                          [4, NUM_ELEMENTS_ONEROW], \
                                          name="vbi_tmp_weights",
                                          scope=tik.scope_ubuf)

    tik_instance.vector_dup(56, vbi_tmp_weights_res, 0, 8, 1, 7)
    tik_instance.vector_dup(56, vbi_tmp_weights, -1, 8, 1, 7)

    lx_fp32 = point_weights_fp32[0, 0]
    hx_fp32 = point_weights_fp32[2, 0]
    for pool_h in range(0, POOL_H):
        ly0 = tik_instance.Scalar(dtype="float32", \
                                  init_value=point_weights_fp32[
                                      1, pool_h * SAMPLING_RATIO])
        hy0 = tik_instance.Scalar(dtype="float32", \
                                  init_value=point_weights_fp32[
                                      3, pool_h * SAMPLING_RATIO])

        num_sampling_w_block = NUM_SAMPLING_W + 1

        tik_instance.vmuls(NUM_SAMPLING_W, vbi_tmp_weights[
            0, num_sampling_w_block * pool_h], \
                           hx_fp32, hy0, 1, 1, 1, 8, 8)
        tik_instance.vmuls(NUM_SAMPLING_W, vbi_tmp_weights[
            1, num_sampling_w_block * pool_h], \
                           lx_fp32, hy0, 1, 1, 1, 8, 8)
        tik_instance.vmuls(NUM_SAMPLING_W, vbi_tmp_weights[
            2, num_sampling_w_block * pool_h], \
                           hx_fp32, ly0, 1, 1, 1, 8, 8)
        tik_instance.vmuls(NUM_SAMPLING_W, vbi_tmp_weights[
            3, num_sampling_w_block * pool_h], \
                           lx_fp32, ly0, 1, 1, 1, 8, 8)

    mask_reduce_1 = tik_instance.Tensor("uint32", [8, ], \
                                        name="mask_reduce1",
                                        scope=tik.scope_ubuf)

    cmp_vct = tik_instance.Tensor("float32", [1, 56], \
                                  name="cmp_vct",
                                  scope=tik.scope_ubuf)
    tik_instance.vector_dup(56, cmp_vct, 0, 1, 1, 7)

    cmpmask = tik_instance.vcmp_gt(56, vbi_tmp_weights, cmp_vct, 1, 1)

    tik_instance.mov_cmpmask_to_tensor(
        mask_reduce_1.reinterpret_cast_to("uint64"), cmpmask)
    with tik_instance.for_range(0, 4) as i:
        tik_instance.vreduce(56, vbi_tmp_weights_res[i, 0], \
                             vbi_tmp_weights[i, 0], mask_reduce_1, 1,
                             1, 7, 0, 0, None, "normal")

    vbi_weights = tik_instance.Tensor("float16",
                                      [VBI_NUM_ELEMENTS_ONEROW * 4, ], \
                                      name="vbi_weights",
                                      scope=tik.scope_ubuf)
    v_repeat_times = (POOL_H * POOL_W + 7) // 8

    src_list1 = [vbi_tmp_weights_res[0, 0], vbi_tmp_weights_res[2, 0]]
    src_list2 = [vbi_tmp_weights_res[1, 0], vbi_tmp_weights_res[3, 0]]

    dst_list1 = [vbi_weights[0 * 16, ], vbi_weights[1 * 16, ]]
    dst_list2 = [vbi_weights[0 * 16, ], vbi_weights[1 * 16, ]]

    tik_instance.vector_dup(112, vbi_weights[0, ], 0, 2, 1, 7)
    tik_instance.scatter_vconv(16, "", dst_list1, src_list1,
                               v_repeat_times, 2, 1, None, False)
    tik_instance.scatter_vconv(16, "", dst_list2, src_list2,
                               v_repeat_times, 2, 1, None, True)
    tik_instance.vector_dup(7, vbi_weights[209, ], 0, 1, 1, 8)

    return vbi_weights


def prepare_vbi_xn(tik_instance, feature_shape,
                   point_positions_int32):
    '''
    :param tik_instance:
    :param feature_shape:the shape of the featuremap
    :param point_positions_int32: the position of 4 pixels around the
    center of gird
    :return:Rearranged address(Xn)
    '''
    # point_positions_int32 : x_low_int y_low_int x_high_int y_high_int
    fm_width = feature_shape[3]
    wstart = tik_instance.Scalar(dtype="int32",
                                 init_value=point_positions_int32[
                                     0, 0])
    wend = tik_instance.Scalar(dtype="int32",
                               init_value=point_positions_int32[
                                   2, STRIDE_W])
    with tik_instance.if_scope(wend == fm_width):
        wend.set_as(fm_width)
    with tik_instance.else_scope():
        wend.set_as(wend + 1)
    width = tik_instance.Scalar(dtype="int32",
                                init_value=wend - wstart)
    point_distance_int32 = get_delta_addresses(tik_instance, \
                                               point_positions_int32,
                                               width)
    vbi_addr = get_vbi_addr_1x1grid(tik_instance,
                                    point_distance_int32)
    return vbi_addr


def prepare_vbi_xm(tik_instance, point_weights_fp32):
    '''
    :param tik_instance:
    :param point_weights_fp32: the distance to 4 pixels around
    :return: vbi_weights ,the rearranged weights(xm),
            hx*hy  lx*hy  hx*ly lx*ly
    '''
    vbi_weights = get_vbi_weights_1x1grid(tik_instance,
                                          point_weights_fp32)
    return vbi_weights


def do_vbi_one_row_mode(tik_instance, block_id, cur_roi_num,
                        feature_shape, \
                        featuremap_gm, output_gm,
                        point_positions_int32, \
                        point_weights_fp32, cut_type_flag):
    '''
    :param tik_instance:
    :param block_id: the block used
    :param cur_roi_num: the num of current roi box
    :param feature_shape: shape of featuremap
    :param featuremap_gm: the gm of featuremap
    :param output_gm: the gm  of output
    :param point_positions_int32: positions of 4 pixels around the gird center
                                 x_low_int y_low_int x_high_int y_high_int
    :param point_weights_fp32: distance to 4 pixels
    :param cut_type_flag: the type of processing mode
    :return: None
    '''
    feature_map_c1 = feature_shape[1]
    if cut_type_flag == "c1Cut":
        feature_map_c1 = (feature_map_c1 + BLOCK_DIM - 1) // BLOCK_DIM
    feature_map_w = feature_shape[3]
    wstart = tik_instance.Scalar(dtype="int32",
                                 init_value=point_positions_int32[
                                     0, 0])
    wend = tik_instance.Scalar(dtype="int32",
                               init_value=point_positions_int32[
                                   2, STRIDE_W])
    with tik_instance.if_scope(wend == feature_map_w):
        wend.set_as(feature_map_w)
    with tik_instance.else_scope():
        wend.set_as(wend + 1)
    width = tik_instance.Scalar(dtype="int32",
                                init_value=wend - wstart)
    featuremap_ub = tik_instance.Tensor("float16", [
        FM_BUFFER_SIZE_LIMIT // BYTES, ], \
                                        name="featuremap_ub",
                                        scope=tik.scope_ubuf)
    for index_height in range(0, POOL_H):
        hstart = tik_instance.Scalar(dtype="int32",
                                     init_value=point_positions_int32[
                                         1, 0])
        num_h = SAMPLING_RATIO * 2
        if cut_type_flag == "c1Cut":
            tik_instance.data_move(featuremap_ub[0, ], \
                                   featuremap_gm[
                                       0, feature_map_c1 * block_id, \
                                       hstart, wstart, 0], 0, num_h,
                                   width, feature_map_w - width, 0)
        vbi_weights = prepare_vbi_xm(tik_instance, point_weights_fp32)
        vbi_addr = prepare_vbi_xn(tik_instance, feature_shape, \
                                  point_positions_int32)

        result_ub = tik_instance.Tensor("float16",
                                        [1, 1, POOL_H + 1, POOL_W,
                                         C0SIZE], \
                                        name="result_ub",
                                        scope=tik.scope_ubuf)
        for i in range(0, feature_map_c1):
            repeat = (POOL_W + 7) // 8
            tik_instance.vbi(128, result_ub[0, 0, index_height, 0, 0],
                             featuremap_ub[0, ], \
                             vbi_weights[0, ], vbi_addr[0, ], 1, repeat, \
                             NUM_ELMENTS_ONEBIN, 1, 8 * 32 // BYTES)
            scale = tik_instance.Scalar(dtype="float16",
                                        init_value=0.25)
            tik_instance.vmuls(128,
                               result_ub[0, 0, index_height, 0, 0], \
                               result_ub[0, 0, index_height, 0, 0], \
                               scale, repeat, 1, 1, 8, 8)
            if (i + 1) < feature_map_c1:
                if cut_type_flag == "c1Cut":
                    tik_instance.data_move(featuremap_ub[0, ], \
                                           featuremap_gm[
                                               0,
                                               feature_map_c1 * \
                                               block_id + i + 1, \
                                               hstart, wstart, 0], 0, \
                                           num_h, width,
                                           feature_map_w - width, 0)
                else:
                    tik_instance.data_move(featuremap_ub[0, ], \
                                           featuremap_gm[
                                               0, i + 1, hstart, wstart, 0], \
                                           0, num_h, width,
                                           feature_map_w - width, 0)

            tik_instance.data_move(
                output_gm[cur_roi_num, i, index_height, 0, 0], \
                result_ub[0, 0, index_height, 0, 0], \
                0, 1, POOL_W, 0, 0)


def do_vbi_full_featuremap_mode(tik_instance, block_id, cur_roi_num, \
                                feature_shape, featuremap_gm,
                                output_gm, \
                                point_positions_int32,
                                point_weights_fp32, cut_type_flag):
    '''
    :param tik_instance:
    :param block_id: the block used
    :param cur_roi_num: the num of current roi box
    :param feature_shape: shape of featuremap
    :param featuremap_gm: the gm of featuremap
    :param output_gm: the gm for output
    :param point_positions_int32: positions of 4 pixels around the gird center
                                x_low_int  y_low_int  x_high_int  y_high_int
    :param point_weights_fp32: distance to 4 pixels
    :param cut_type_flag: the type of processing mode
    :return:None
    '''
    feature_map_c1 = feature_shape[1]
    if cut_type_flag == "c1Cut":
        feature_map_c1 = (feature_map_c1 + BLOCK_DIM - 1) // BLOCK_DIM
    feature_map_h = feature_shape[2]
    feature_map_w = feature_shape[3]
    hstart = tik_instance.Scalar(dtype="int32",
                                 init_value=point_positions_int32[
                                     1, 0])
    hend = tik_instance.Scalar(dtype="int32",
                               init_value=point_positions_int32[
                                   3, STRIDE_H])
    wstart = tik_instance.Scalar(dtype="int32",
                                 init_value=point_positions_int32[
                                     0, 0])
    wend = tik_instance.Scalar(dtype="int32",
                               init_value=point_positions_int32[
                                   2, STRIDE_W])
    with tik_instance.if_scope(wend == feature_map_w):
        wend.set_as(feature_map_w)
    with tik_instance.else_scope():
        wend.set_as(wend + 1)
    with tik_instance.if_scope(hend == feature_map_h):
        hend.set_as(feature_map_h)
    with tik_instance.else_scope():
        hend.set_as(hend + 1)
    width = tik_instance.Scalar(dtype="int32",
                                init_value=wend - wstart)
    height = tik_instance.Scalar(dtype="int32",
                                 init_value=hend - hstart)
    featuremap_ub = tik_instance.Tensor("float16", [
        FM_BUFFER_SIZE_LIMIT // BYTES, ], \
                                        name="featuremap_ub",
                                        scope=tik.scope_ubuf)
    if cut_type_flag == "c1Cut":
        tik_instance.data_move(featuremap_ub[0, ], \
                               featuremap_gm[
                                   0, block_id * feature_map_c1,
                                   hstart, wstart, 0], \
                               0, height, width,
                               feature_map_w - width, 0)
    else:
        tik_instance.data_move(featuremap_ub[0, ],
                               featuremap_gm[0, 0, hstart, wstart, 0],
                               0, \
                               height, width, feature_map_w - width,
                               0)
    vbi_weights = prepare_vbi_xm(tik_instance, point_weights_fp32)
    vbi_addr = prepare_vbi_xn(tik_instance, feature_shape,
                              point_positions_int32)
    result_ub = tik_instance.Tensor("float16",
                                    [1, 1, POOL_H + 1, POOL_W,
                                     C0SIZE], \
                                    name="result_ub",
                                    scope=tik.scope_ubuf)
    for i in range(0, feature_map_c1):
        repeat = (POOL_H * POOL_W + 7) // 8
        tik_instance.vbi(128, result_ub[0, 0, 0, 0, 0],
                         featuremap_ub[0, ], vbi_weights[0, ], \
                         vbi_addr[0, ], 1, repeat, NUM_ELMENTS_ONEBIN,
                         1, 8 * 32 // BYTES)
        if (i + 1) < feature_map_c1:
            if cut_type_flag == "c1Cut":
                tik_instance.data_move(featuremap_ub[0, ], \
                                       featuremap_gm[
                                           0,
                                           feature_map_c1 * block_id + i + 1,
                                           hstart, wstart, 0], \
                                       0, height, width, \
                                       feature_map_w - width, 0)
            else:
                tik_instance.data_move(featuremap_ub[0, ], \
                                       featuremap_gm[
                                           0, i + 1, hstart, wstart, 0], \
                                       0, height, width,
                                       feature_map_w - width, 0)
        if cut_type_flag == "c1Cut":
            tik_instance.data_move(output_gm[cur_roi_num,
                                             feature_map_c1 * block_id + i,
                                             0, 0, 0],
                                   result_ub[0, 0, 0, 0, 0],
                                   0, 1, POOL_H * POOL_W, 0, 0)
        else:
            tik_instance.data_move(output_gm[cur_roi_num, i, 0, 0, 0],
                                   result_ub[0, 0, 0, 0, 0], \
                                   0, 1, POOL_H * POOL_W, 0, 0)


def process_one_roi_vbi(tik_instance, block_id,
                        cur_roi_num, feature_shape,
                        featuremap_gm, output_gm,
                        point_positions_int32,
                        point_weights_fp32, cut_type_flag):
    '''
    :param tik_instance:
    :param block_id:   the block used
    :param cur_roi_num: the num of current roi box
    :param feature_shape: shape of featuremap
    :param featuremap_gm: the gm  of featuremap
    :param output_gm: the gm  for output
    :param point_positions_int32: positions of 4 pixels
            around the center of gird /
            x_low_int  y_low_int  x_high_int  y_high_int
    :param point_weights_fp32: distance to 4 pixels
    :param cut_type_flag: the type of processing mode
    :return: None
    '''
    feature_map_h = feature_shape[2]
    feature_map_w = feature_shape[3]

    hstart = tik_instance.Scalar(dtype="int32",
                                 init_value=point_positions_int32[
                                     1, 0])
    hend = tik_instance.Scalar(dtype="int32",
                               init_value=point_positions_int32[
                                   3, STRIDE_H])
    wstart = tik_instance.Scalar(dtype="int32",
                                 init_value=point_positions_int32[
                                     0, 0])
    wend = tik_instance.Scalar(dtype="int32",
                               init_value=point_positions_int32[
                                   2, STRIDE_W])
    with tik_instance.if_scope(wend == feature_map_w):
        wend.set_as(feature_map_w)
    with tik_instance.else_scope():
        wend.set_as(wend + 1)
    with tik_instance.if_scope(hend == feature_map_h):
        hend.set_as(feature_map_h)
    with tik_instance.else_scope():
        hend.set_as(hend + 1)
    width = tik_instance.Scalar(dtype="int32",
                                init_value=wend - wstart)
    height = tik_instance.Scalar(dtype="int32",
                                 init_value=hend - hstart)
    roi_feature_map_size = \
        tik_instance.Scalar(dtype="int32",
                            init_value=width * height * C0SIZE * BYTES)
    with tik_instance.if_scope(
            FM_BUFFER_SIZE_LIMIT >= roi_feature_map_size):
        do_vbi_full_featuremap_mode(tik_instance, block_id,
                                    cur_roi_num,
                                    feature_shape, featuremap_gm,
                                    output_gm,
                                    point_positions_int32,
                                    point_weights_fp32, cut_type_flag)
    with tik_instance.else_scope():
        do_vbi_one_row_mode(tik_instance, block_id, cur_roi_num,
                            feature_shape, featuremap_gm,
                            output_gm, point_positions_int32,
                            point_weights_fp32, cut_type_flag)


def roi_align_v200_1951_compute(tik_instance, block_id, featuremap_gm,
                                rois_gm, \
                                output_gm, feature_map_dict,
                                rois_dict):
    '''
    :param tik_instance:
    :param block_id:  the block used
    :param featuremap_gm: the gm  of featuremap
    :param rois_gm: the gm for roi_boxes
    :param output_gm: the gm for output
    :param feature_map_dict: placeholder of featuremap
    :param rois_dict: the placeholder for roi_boxes
    :return: None
    '''
    feature_shape = feature_map_dict.get("shape")
    rois_shape = rois_dict.get("shape")
    rois_num = rois_shape[0]
    rois = tik_instance.Tensor("float16", [ROINUM_LIMIT, 4],
                               name="rois_ub",
                               scope=tik.scope_ubuf)
    tik_instance.vector_dup(128, rois, 0, 4, 1, 8)
    tik_instance.data_move(rois, rois_gm, 0, 1, rois_num // 4, 0, 0)
    rois_fp32, grid_hw_fp32 = roi_align_perf_scale(tik_instance, rois)
    index_array_fp32 = tik_instance.Tensor("float32",
                                           [ROINUM_LIMIT, ],
                                           name="index_array_fp32",
                                           scope=tik.scope_ubuf)
    index = tik_instance.Scalar(dtype="float32", init_value=0.0)
    for i in range(0, ROINUM_LIMIT):
        index.set_as(i + 0.5)
        index_array_fp32[i].set_as(index)

    roi_align_roi_num_cut(tik_instance, block_id, rois_fp32,
                          grid_hw_fp32, featuremap_gm,
                          output_gm, feature_shape,
                          rois_num, index_array_fp32)


def roi_align_roi_num_cut(tik_instance, block_id, rois_fp32,
                          grid_hw_fp32, featuremap_gm,
                          output_gm, feature_shape,
                          rois_num, index_array_fp32):
    '''
    :param tik_instance:
    :param block_id: the block used
    :param rois_fp32: the pos of the roi_box
    :param grid_hw_fp32: the wide and hight of grid in roi box
    :param featuremap_gm: the gm  of featuremap
    :param output_gm: the gm for output
    :param feature_shape: shape of featuremap
    :param rois_num: the number of roi_box
    :param index_array_fp32: a array corresponding to center of every  grid
    :return: None
    '''
    roi_percore = (rois_num + BLOCK_DIM - 1) // BLOCK_DIM
    with tik_instance.for_range(0, roi_percore) as roi_percore_index:
        curr_roi = block_id * roi_percore + roi_percore_index
        point_positions_int32, point_weights_fp32 = \
            roi_align_perf_gengrid_fp32(tik_instance,
                                        curr_roi,
                                        rois_fp32,
                                        grid_hw_fp32,
                                        feature_shape,
                                        index_array_fp32)
        process_one_roi_vbi(tik_instance, block_id, curr_roi, \
                            feature_shape, featuremap_gm, \
                            output_gm, point_positions_int32, \
                            point_weights_fp32, "roinumCut")


def check_roi_align_vbi_params(feature_map, rois):
    '''4*n
    :param feature_map:  placeholder of  feature_map
    :param rois: placeholder of  rois
    :return: None
    '''
    shape_featuremap = feature_map.get('shape')
    shape_rois = rois.get('shape')
    dtype_featuremap = feature_map.get('dtype').lower()
    dtype_rois = rois.get('dtype').lower()
    if dtype_featuremap != 'float16':
        raise RuntimeError("dtype of feature_map should be float16")
    if dtype_rois != 'float16':
        raise RuntimeError("dtype of rois should be float16")
    if len(shape_featuremap) != FIVE:
        raise RuntimeError("dimension of featuremap should be 5")
    if shape_featuremap != (1, 1, 32, 57, 16):
        raise RuntimeError("featuremap should be [1, 1, 32, 57, 16]")
    if len(shape_rois) != TWO:
        raise RuntimeError("dimension of rois should be TWO")
    if shape_rois[1] != 4:
        raise RuntimeError("second dimension of rois should be 4")
    if shape_rois[0] < 1:
        raise RuntimeError("the num of rois should be no less than 1")
    if shape_rois[0] % 4:
        raise RuntimeError("the num of rois should be divisible by 4")
    if shape_rois[0] > 96:
        raise RuntimeError("the num of rois should be no more than 96")




@util.check_input_type(dict, dict, str)
def roi_align_vbi(featuremap, rois_box,
                  kernel_name="roi_align"):
    '''
    roi_align API used only for 2d-h1 net in v200 aic (vbi support)
    network type: tensorflow
    dtype: float16
    pool_h: 7
    pool_w: 7
    sample_ratio: 1
    rois num range: 1-96
    block_dim: 8
    :param feature_map_dict: placeholder of  feature_map
    :param rois_dict:  placeholder of  rois
    :param kernel_name: name of kernel
    :return: the roi_align_vbi result
    '''
    check_roi_align_vbi_params(featuremap, rois_box)
    util.check_kernel_name(kernel_name)

    tik_instance = tik.Tik(tik.Dprofile(), True)
    rois_shape = rois_box.get("shape")
    dtype = featuremap.get("dtype")
    feature_shape = featuremap.get("shape")
    feature_map = tik_instance.Tensor(
        dtype, feature_shape, name="feature_map", scope=tik.scope_gm)
    rois = tik_instance.Tensor(
        dtype, rois_shape, name="rois", scope=tik.scope_gm)
    fm_c1 = feature_shape[1]
    ret = tik_instance.Tensor(
        dtype, [rois_shape[0], fm_c1, POOL_H, POOL_W, C0SIZE],
        name="ret",
        scope=tik.scope_gm)
    with tik_instance.for_range(0, BLOCK_DIM,
                                block_num=BLOCK_DIM) as block_id:
        roi_align_v200_1951_compute(tik_instance, block_id,
                                    feature_map, \
                                    rois, ret, featuremap,
                                    rois_box)
    tik_instance.BuildCCE(
        kernel_name=kernel_name, inputs=[feature_map, rois],
        outputs=[ret])
    return tik_instance
