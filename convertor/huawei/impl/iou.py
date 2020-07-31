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

iou
"""

from te.platform.fusion_manager import fusion_manager
from te import tik
from te import platform as tbe_platform
from topi.cce import util

# MAX ELIMENT NUM OF FP16 IN 1BLOCK
FP16_ELIMENTS_BLOCK = 16
# MAX ELIMENT NUM OF FP32 IN 1BLOCK
FP32_ELIMENTS_BLOCK = 8
# CONST GTBOX SLICE SEGMENT
GTBOX_SEGMENT = 4096 * 4
# CONST BBOX SLICE SEGMENT
BBOX_SEGMENT = 4096 * 4


def _apply_mem(tik_instance, dtype,
               shape, name, scope=tik.scope_ubuf):
    """apply mem fuc

    Parameters
    ----------
    tik_instance: tik_instance
        tik_instance
    dtype: str
        ub dtype
    shape: list
        ub shape
    name: str
        ub name
    scope: scope
        scope_ubuf or scope_gm
    Returns
    -------
    Tensor: Tensor
    """
    return tik_instance.Tensor(dtype, shape, name=name, scope=scope)


def _get_ceil_int(int1, int2):
    """Get Ceil Int

    Parameters
    ----------
    int1: int
        input int 1
    int2: int
        input int 2

    Returns
    -------
    ceil_int: int
    """
    _result = int1 // int2
    if int1 % int2 == 0:
        ceil_int = _result
    else:
        ceil_int = _result + 1

    return ceil_int


# pylint: disable=too-many-instance-attributes
class Iou():
    """Function: use to finish Iou main functions
    """

    # pylint: disable=too-many-statements
    def __init__(self, bboxes, gtboxes, mode):
        """
        init Iou parameters

        Parameters
        ----------
        bboxes : dict
            data of bboxes.
            source data type, support "float16"
        gtboxes : dict
            data of gtboxes.
            source data type, support "float16"
        overlap : dict
            shape and dtype of overlap
            result shape is [m, n]
        mode :  str
            ('iou','iof')
            iou : the output is inter_area / total_area
            iof : the output is inter_area / gtboxes_area

        Returns
        -------
        None
        """
        self.bboxes_shape = bboxes.get("shape")
        self.bboxes_dtype = bboxes.get("dtype").lower()
        self.gtboxes_shape = gtboxes.get("shape")
        self.gtboxes_dtype = gtboxes.get("dtype").lower()
        self.gtboxes_num = self.gtboxes_shape[0]
        self.dtype = self.bboxes_dtype
        self.mode = mode.lower()
        self.iou_shape = [self.gtboxes_shape[0], self.bboxes_shape[0]]
        self.tik_instance = tik.Tik()
        self.core_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        self.product = tbe_platform.cce_conf.api_check_support(
            "tik.vdiv", "float32")
        # input and output tensor in gm
        self.bboxes_gm = self.tik_instance.Tensor(
            self.bboxes_dtype,
            self.bboxes_shape,
            name="bboxes_gm",
            scope=tik.scope_gm)
        self.gtboxes_gm = self.tik_instance.Tensor(
            self.gtboxes_dtype,
            self.gtboxes_shape,
            name="gtboxes_gm",
            scope=tik.scope_gm)
        self.iou_gm = self.tik_instance.Tensor(
            self.bboxes_dtype,
            self.iou_shape,
            name="iou_gm",
            scope=tik.scope_gm)

        # init attr in objext
        self.point_per_core = 0
        self.core_taill_num = 0
        self.gt_ub_segment = 0
        self.bb_ub_segment = 0
        self.bb_ub_segment_point = 0
        self.gt_ub_segment_point = 0
        self.out_ub_segment = 0
        self.area_ub_size = 0
        self.gt_area_ub_size = 0
        self.area_x0_size = 0
        self.area_x0 = None
        self.area_x1 = None
        self.area_y0 = None
        self.area_y1 = None
        self.inter_area_x0 = None
        self.inter_area_x1 = None
        self.inter_area_y0 = None
        self.inter_area_y1 = None
        self.area_y1_y0 = None
        self.gtboxes_ub = None
        self.gt_boxes_area_ub = None
        self.bboxes_ub = None
        self.out_ub = None
        self.bboxes_area_ub = None
        self.inter_area_ub = None
        self.zero_ub = None
        self.gtboxes_x0 = None
        self.gtboxes_x1 = None
        self.gtboxes_y0 = None
        self.gtboxes_y1 = None
        self.index_reg = []
        block_parm_dict = {
            "float16": FP16_ELIMENTS_BLOCK,
            "float32": FP32_ELIMENTS_BLOCK,
        }
        self.min_point_per_core = block_parm_dict[self.bboxes_dtype]
        self.eliments_per_block = block_parm_dict[self.bboxes_dtype]
        self.gt_ub_segment = GTBOX_SEGMENT if self.bboxes_dtype == "float16" \
            else GTBOX_SEGMENT // 2
        self.bb_ub_segment = BBOX_SEGMENT if self.bboxes_dtype == "float16" \
            else BBOX_SEGMENT // 2
        self.max_eliments = block_parm_dict[self.bboxes_dtype] * 8
        if self.product is False:
            self.bb_ub_segment = self.bb_ub_segment // 2

    # pylint: disable=too-many-statements
    def iou_process(self):
        """do process and scedule
           main function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.point_per_core = _get_ceil_int(self.bboxes_shape[0],
                                            self.core_num)
        if self.point_per_core < self.min_point_per_core:
            self.point_per_core = self.min_point_per_core
        self.point_per_core = \
            _get_ceil_int(self.point_per_core,
                          self.min_point_per_core) * \
            self.min_point_per_core
        self.core_taill_num = self.bboxes_shape[0] % self.point_per_core
        self.core_num = _get_ceil_int(self.bboxes_shape[0],
                                      self.point_per_core)

        self.bb_ub_segment_point = self.bb_ub_segment // 4
        self.gt_ub_segment_point = self.gt_ub_segment // 4
        self.out_ub_segment = self.iou_shape[0] * self.iou_shape[1]
        self.area_ub_size = _get_ceil_int(self.bb_ub_segment_point,
                                          self.max_eliments) * \
                            self.max_eliments
        self.gt_area_ub_size = _get_ceil_int(self.gt_ub_segment_point,
                                             self.max_eliments) * \
                               self.max_eliments

        self.area_x0_size = max(self.area_ub_size, self.gt_area_ub_size)

        with self.tik_instance.for_range(
                0, self.core_num, block_num=self.core_num) as _core_id:
            # calcu gt area
            self.area_x0 = _apply_mem(self.tik_instance, self.dtype,
                                      [self.area_x0_size], "area_x0")
            self.area_x1 = _apply_mem(self.tik_instance, self.dtype,
                                      [self.area_x0_size], "area_x1")
            self.area_y0 = _apply_mem(self.tik_instance, self.dtype,
                                      [self.area_x0_size], "area_y0")
            self.area_y1 = _apply_mem(self.tik_instance, self.dtype,
                                      [self.area_x0_size], "area_y1")
            self.inter_area_x0 = _apply_mem(self.tik_instance, self.dtype,
                                            [self.area_x0_size],
                                            "inter_area_x0")
            self.inter_area_x1 = _apply_mem(self.tik_instance, self.dtype,
                                            [self.area_x0_size],
                                            "inter_area_x1")
            self.inter_area_y0 = _apply_mem(self.tik_instance, self.dtype,
                                            [self.area_x0_size],
                                            "inter_area_y0")
            self.inter_area_y1 = _apply_mem(self.tik_instance, self.dtype,
                                            [self.area_x0_size],
                                            "inter_area_y1")
            self.area_y1_y0 = _apply_mem(self.tik_instance, self.dtype,
                                         [self.area_x0_size], "area_y1_y0")
            self.gtboxes_ub = _apply_mem(self.tik_instance, self.dtype,
                                         [self.gt_ub_segment], "gtboxes_ub")
            self.gt_boxes_area_ub = _apply_mem(self.tik_instance, self.dtype,
                                               [self.gt_area_ub_size],
                                               "gt_boxes_area_ub")

            run_gt_point = self.gtboxes_shape[0]
            run_gt_point_segment = run_gt_point * 4
            # global
            nbust = _get_ceil_int(run_gt_point_segment,
                                  self.eliments_per_block)
            self.tik_instance.data_move(self.gtboxes_ub, self.gtboxes_gm, 0, 1,
                                        nbust, 0, 0)
            # [n,4] --> 4*[n,1]  by scalar
            self.data_rerange(run_gt_point, self.gtboxes_ub)
            # calcu area
            self.calcu_area(run_gt_point, self.gt_boxes_area_ub)

            bb_loop = self.point_per_core * 4 // self.bb_ub_segment
            bb_tail = self.point_per_core * 4 % self.bb_ub_segment

            min_segment = self.min_point_per_core * 4
            if (0 < bb_tail < min_segment) and bb_loop != 0:
                bb_tail_offset = bb_loop * self.bb_ub_segment + \
                                 bb_tail - min_segment
                bb_tail = min_segment
            elif bb_tail % min_segment != 0 and bb_loop != 0:
                bb_tail_offset = bb_loop * self.bb_ub_segment + \
                                 (bb_tail % min_segment) - min_segment
                bb_tail = (bb_tail // min_segment + 1) * min_segment
            else:
                bb_tail_offset = bb_loop * self.bb_ub_segment
            # one time output bb_ub_segment_point values
            self.zero_ub = _apply_mem(self.tik_instance, self.dtype,
                                      [self.eliments_per_block], "zero_ub")
            self.tik_instance.vector_dup(self.eliments_per_block,
                                         self.zero_ub, 0.0,
                                         1, 1, 8)
            thread_num = 1
            if bb_loop > 1:
                thread_num = 2
            if self.core_taill_num != 0:
                with self.tik_instance.if_scope(_core_id == (
                        self.core_num - 1)):
                    dst_gm_offset = self.point_per_core * _core_id - \
                                    self.point_per_core + self.core_taill_num
                    if self.core_num == 1:
                        dst_gm_offset = 0
                    with self.tik_instance.for_range(
                            0, bb_loop, thread_num=thread_num) as bb_loop_index:
                        gm_point_offset = \
                            (bb_loop_index * self.bb_ub_segment) // 4 + \
                            dst_gm_offset
                        self._run_segment(self.bb_ub_segment, gm_point_offset)
                    if bb_tail != 0:
                        gm_point_offset = bb_tail_offset // 4 + dst_gm_offset
                        self._run_segment(bb_tail, gm_point_offset)
                with self.tik_instance.else_scope():
                    dst_gm_offset = self.point_per_core * _core_id
                    with self.tik_instance.for_range(
                            0, bb_loop, thread_num=thread_num) as bb_loop_index:
                        gm_point_offset = \
                            (bb_loop_index * self.bb_ub_segment) // 4 + \
                            dst_gm_offset
                        self._run_segment(self.bb_ub_segment, gm_point_offset)
                    if bb_tail != 0:
                        gm_point_offset = bb_tail_offset // 4 + dst_gm_offset
                        self._run_segment(bb_tail, gm_point_offset)
            else:
                dst_gm_offset = self.point_per_core * _core_id
                with self.tik_instance.for_range(
                        0, bb_loop, thread_num=thread_num) as bb_loop_index:
                    gm_point_offset = \
                        (bb_loop_index * self.bb_ub_segment) // 4 + \
                        dst_gm_offset
                    self._run_segment(self.bb_ub_segment, gm_point_offset)
                if bb_tail != 0:
                    gm_point_offset = bb_tail_offset // 4 + dst_gm_offset
                    self._run_segment(bb_tail, gm_point_offset)

    def iou_process_cut_by_gt(self):
        """do process and scedule by gt
           main function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.bboxes_shape[0] < self.eliments_per_block:
            self.core_num = 1
        self.point_per_core = _get_ceil_int(self.gtboxes_shape[0],
                                            self.core_num)
        if self.bboxes_shape[0] < self.min_point_per_core:
            if self.point_per_core < self.min_point_per_core:
                self.point_per_core = self.min_point_per_core
            self.point_per_core = \
                _get_ceil_int(self.point_per_core,
                              self.min_point_per_core) * \
                self.min_point_per_core
        self.core_taill_num = self.gtboxes_shape[0] % self.point_per_core
        self.core_num = _get_ceil_int(self.gtboxes_shape[0],
                                      self.point_per_core)

        bb_loop = self.bboxes_shape[0] * 4 // self.bb_ub_segment
        bb_tail = self.bboxes_shape[0] * 4 % self.bb_ub_segment

        min_segment = self.min_point_per_core * 4
        if (0 < bb_tail < min_segment) and bb_loop != 0:
            bb_tail_offset = bb_loop * self.bb_ub_segment + \
                             bb_tail - min_segment
            bb_tail = min_segment
        elif bb_tail % min_segment != 0 and bb_loop != 0:
            bb_tail_offset = bb_loop * self.bb_ub_segment + \
                             (bb_tail % min_segment) - min_segment
            bb_tail = (bb_tail // min_segment + 1) * min_segment
        else:
            bb_tail_offset = bb_loop * self.bb_ub_segment

        self.area_ub_size = _get_ceil_int(self.bb_ub_segment // 4,
                                          self.max_eliments) * \
                            self.max_eliments
        self.gt_area_ub_size = _get_ceil_int(self.gt_ub_segment // 4,
                                             self.max_eliments) * \
                               self.max_eliments
        thread_num = 1
        if bb_loop > 1:
            thread_num = 2
        self.area_x0_size = max(self.area_ub_size, self.gt_area_ub_size)
        with self.tik_instance.for_range(
                0, self.core_num, block_num=self.core_num) as _core_id:
            self.zero_ub = _apply_mem(self.tik_instance, self.dtype,
                                      [self.eliments_per_block], "zero_ub")
            self.tik_instance.vector_dup(self.eliments_per_block,
                                         self.zero_ub, 0.0,
                                         1, 1, 8)

            def _run(gt_len):
                gt_loop = (gt_len) // self.gt_ub_segment
                gt_tail = (gt_len) % self.gt_ub_segment
                with self.tik_instance.for_range(0, gt_loop) as _gt_loop:
                    self._apply_all_ub()
                    dst_gm_offset = \
                        (self.point_per_core * _core_id +
                         _gt_loop * self.gt_ub_segment // 4) * \
                        self.bboxes_shape[0]
                    # global
                    nbust = _get_ceil_int(self.gt_ub_segment,
                                          self.eliments_per_block)
                    gt_gm_offset = _core_id * self.point_per_core * 4 + \
                                   _gt_loop * self.gt_ub_segment
                    self.tik_instance.data_move(self.gtboxes_ub,
                                                self.gtboxes_gm[gt_gm_offset],
                                                0, 1, nbust, 0, 0)
                    # [n,4] --> 4*[n,1]  by scalar
                    self.data_rerange(self.gt_ub_segment // 4, self.gtboxes_ub)
                    # calcu area
                    self.calcu_area(self.gt_ub_segment // 4,
                                    self.gt_boxes_area_ub)
                    self.gtboxes_num = self.gt_ub_segment // 4
                    with self.tik_instance.for_range(
                            0, bb_loop, thread_num=thread_num) as bb_loop_index:
                        gm_point_offset = \
                            (bb_loop_index * self.bb_ub_segment) // 4
                        self._run_segment(self.bb_ub_segment,
                                          gm_point_offset,
                                          dst_gm_offset)
                    if bb_tail != 0:
                        gm_point_offset = bb_tail_offset // 4
                        if (bb_tail // 4) % self.eliments_per_block == 0 or \
                                self.core_num == 1:
                            self._run_segment(bb_tail, gm_point_offset,
                                              dst_gm_offset)
                        else:
                            bb_tail_half = \
                                _get_ceil_int(bb_tail // 8,
                                              self.eliments_per_block) * \
                                self.eliments_per_block
                            self._run_segment(bb_tail_half * 4, gm_point_offset,
                                              dst_gm_offset)
                            gm_point_offset = gm_point_offset + bb_tail_half - \
                                              (bb_tail_half * 2 - bb_tail // 4)
                            self._run_segment(bb_tail_half * 4, gm_point_offset,
                                              dst_gm_offset)

                if gt_tail != 0:
                    self._apply_all_ub()
                    dst_gm_offset = \
                        (self.point_per_core * _core_id +
                         gt_loop * self.gt_ub_segment // 4) * \
                        self.bboxes_shape[0]
                    # global
                    nbust = _get_ceil_int(gt_tail,
                                          self.eliments_per_block)
                    gt_gm_offset = _core_id * self.point_per_core * 4 + \
                                   gt_loop * self.gt_ub_segment
                    self.tik_instance.data_move(self.gtboxes_ub,
                                                self.gtboxes_gm[gt_gm_offset],
                                                0, 1, nbust, 0, 0)
                    # [n,4] --> 4*[n,1]  by scalar
                    self.data_rerange(gt_tail // 4, self.gtboxes_ub)
                    # calcu area
                    self.calcu_area(gt_tail // 4, self.gt_boxes_area_ub)
                    self.gtboxes_num = gt_tail // 4
                    with self.tik_instance.for_range(
                            0, bb_loop, thread_num=thread_num) as bb_loop_index:
                        gm_point_offset = \
                            (bb_loop_index * self.bb_ub_segment) // 4
                        self._run_segment(self.bb_ub_segment,
                                          gm_point_offset,
                                          dst_gm_offset)
                    if bb_tail != 0:
                        gm_point_offset = bb_tail_offset // 4
                        if (bb_tail // 4) % self.eliments_per_block == 0 or \
                                self.core_num == 1:
                            self._run_segment(bb_tail,
                                              gm_point_offset,
                                              dst_gm_offset)
                        else:
                            bb_tail_half = \
                                _get_ceil_int(bb_tail // 8,
                                              self.eliments_per_block) * \
                                self.eliments_per_block
                            self._run_segment(bb_tail_half * 4,
                                              gm_point_offset,
                                              dst_gm_offset)
                            gm_point_offset = gm_point_offset + \
                                              bb_tail_half - \
                                              (bb_tail_half * 2 - bb_tail // 4)
                            self._run_segment(bb_tail_half * 4,
                                              gm_point_offset,
                                              dst_gm_offset)

            if self.core_taill_num == 0:
                _run(self.point_per_core * 4)
            else:
                with self.tik_instance.if_scope(_core_id == (
                        self.core_num - 1)):
                    _run(self.core_taill_num * 4)
                with self.tik_instance.else_scope():
                    _run(self.point_per_core * 4)

    def _apply_all_ub(self):
        self.area_x0 = _apply_mem(self.tik_instance, self.dtype,
                                  [self.area_x0_size], "area_x0")
        self.area_x1 = _apply_mem(self.tik_instance, self.dtype,
                                  [self.area_x0_size], "area_x1")
        self.area_y0 = _apply_mem(self.tik_instance, self.dtype,
                                  [self.area_x0_size], "area_y0")
        self.area_y1 = _apply_mem(self.tik_instance, self.dtype,
                                  [self.area_x0_size], "area_y1")
        self.inter_area_x0 = _apply_mem(self.tik_instance, self.dtype,
                                        [self.area_x0_size],
                                        "inter_area_x0")
        self.inter_area_x1 = _apply_mem(self.tik_instance, self.dtype,
                                        [self.area_x0_size],
                                        "inter_area_x1")
        self.inter_area_y0 = _apply_mem(self.tik_instance, self.dtype,
                                        [self.area_x0_size],
                                        "inter_area_y0")
        self.inter_area_y1 = _apply_mem(self.tik_instance, self.dtype,
                                        [self.area_x0_size],
                                        "inter_area_y1")
        self.area_y1_y0 = _apply_mem(self.tik_instance, self.dtype,
                                     [self.area_x0_size], "area_y1_y0")
        self.gtboxes_ub = _apply_mem(self.tik_instance, self.dtype,
                                     [self.gt_ub_segment], "gtboxes_ub")
        self.gt_boxes_area_ub = _apply_mem(self.tik_instance, self.dtype,
                                           [self.gt_area_ub_size],
                                           "gt_boxes_area_ub")

    def _run_segment(self, run_bb_point_segment, gm_offset, gm_out_offset=0):
        """
        do a segment of bbox compute

        Parameters
        ----------
        run_bb_point_segment : int
            bbox segment len
        gm_offset : int
            gm offset

        Returns
        -------
        None
        """
        run_bb_point = run_bb_point_segment // 4
        src_gm_offset = gm_offset * 4
        # copy gm to ub
        nbust = _get_ceil_int(run_bb_point_segment, self.eliments_per_block)
        self.out_ub = _apply_mem(self.tik_instance, self.dtype,
                                 [self.area_ub_size], "out_ub")
        self.bboxes_area_ub = _apply_mem(self.tik_instance, self.dtype,
                                         [self.area_ub_size],
                                         "bboxes_area_ub")
        self.inter_area_ub = _apply_mem(self.tik_instance, self.dtype,
                                        [self.area_ub_size],
                                        "inter_area_ub")
        self.bboxes_ub = _apply_mem(self.tik_instance, self.dtype,
                                    [self.bb_ub_segment], "bboxes_ub")
        self.tik_instance.data_move(
            self.bboxes_ub, self.bboxes_gm[src_gm_offset], 0, 1, nbust, 0, 0)

        # [n,4] --> 4*[n,1]  by scalar
        self.data_rerange(run_bb_point, self.bboxes_ub)
        # calcu area
        self.calcu_area(run_bb_point, self.bboxes_area_ub)

        self.gtboxes_x0 = _apply_mem(self.tik_instance, self.dtype,
                                     [self.eliments_per_block], "gtboxes_x0")
        self.gtboxes_x1 = _apply_mem(self.tik_instance, self.dtype,
                                     [self.eliments_per_block], "gtboxes_x1")
        self.gtboxes_y0 = _apply_mem(self.tik_instance, self.dtype,
                                     [self.eliments_per_block], "gtboxes_y0")
        self.gtboxes_y1 = _apply_mem(self.tik_instance, self.dtype,
                                     [self.eliments_per_block], "gtboxes_y1")
        scalar_addr = \
            [self.tik_instance.Scalar(dtype=self.dtype) for _ in range(4)]
        scalar_area = self.tik_instance.Scalar(dtype=self.dtype)
        with self.tik_instance.for_range(
                0, self.gtboxes_num) as gt_global_index:
            scalar_area.set_as(self.gt_boxes_area_ub[gt_global_index])
            for i in range(4):
                scalar_addr[i].set_as(self.gtboxes_ub[gt_global_index * 4 + i])
            self.tik_instance.vector_dup(self.eliments_per_block,
                                         self.gtboxes_x0, scalar_addr[0],
                                         1, 1, 8)
            self.tik_instance.vector_dup(self.eliments_per_block,
                                         self.gtboxes_y0, scalar_addr[1],
                                         1, 1, 8)
            self.tik_instance.vector_dup(self.eliments_per_block,
                                         self.gtboxes_x1, scalar_addr[2],
                                         1, 1, 8)
            self.tik_instance.vector_dup(self.eliments_per_block,
                                         self.gtboxes_y1, scalar_addr[3],
                                         1, 1, 8)
            # vmin vmax
            repeat_time = _get_ceil_int(run_bb_point, self.max_eliments)
            self.tik_instance.vmax(self.max_eliments,
                                   self.inter_area_x0,
                                   self.area_x0,
                                   self.gtboxes_x0,
                                   repeat_time, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vmax(self.max_eliments,
                                   self.inter_area_y0,
                                   self.area_y0,
                                   self.gtboxes_y0,
                                   repeat_time, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vmin(self.max_eliments,
                                   self.inter_area_x1,
                                   self.area_x1,
                                   self.gtboxes_x1,
                                   repeat_time, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vmin(self.max_eliments,
                                   self.inter_area_y1,
                                   self.area_y1,
                                   self.gtboxes_y1,
                                   repeat_time, 1, 1, 0, 8, 8, 0)
            self.calcu_area(run_bb_point, self.inter_area_ub, True)
            if self.mode == "iou":
                self.tik_instance.vadds(self.max_eliments, self.out_ub,
                                        self.bboxes_area_ub,
                                        scalar_area, repeat_time, 1, 1, 8, 8)
                self.tik_instance.vsub(
                    self.max_eliments, self.out_ub, self.out_ub,
                    self.inter_area_ub, repeat_time, 1, 1, 1, 8, 8, 8)
            elif self.mode == "iof":
                self.tik_instance.vector_dup(self.max_eliments, self.out_ub,
                                             scalar_area, repeat_time, 1, 8)

            if self.product is True:
                self.tik_instance.vdiv(
                    self.max_eliments, self.out_ub, self.inter_area_ub,
                    self.out_ub, repeat_time, 1, 1, 1, 8, 8, 8)
            else:
                # for mini
                rec_1 = _apply_mem(self.tik_instance, self.dtype,
                                   [self.area_x0_size],
                                   "rec_1")
                rec_2 = _apply_mem(self.tik_instance, self.dtype,
                                   [self.area_x0_size],
                                   "rec_2")
                self.tik_instance.vrec(self.max_eliments, rec_1,
                                       self.out_ub,
                                       repeat_time, 1, 1, 8, 8)
                self.tik_instance.vmul(self.max_eliments, rec_2,
                                       rec_1,
                                       self.out_ub, repeat_time, 1,
                                       1, 1, 8, 8, 8)
                self.tik_instance.vmuls(self.max_eliments, rec_2,
                                        rec_2,
                                        -1, repeat_time, 1, 1, 8, 8)
                self.tik_instance.vadds(self.max_eliments, rec_2,
                                        rec_2,
                                        2, repeat_time, 1, 1, 8, 8)
                self.tik_instance.vmul(self.max_eliments, rec_2,
                                       rec_2,
                                       rec_1, repeat_time, 1,
                                       1, 1, 8, 8, 8)
                self.tik_instance.vmul(self.max_eliments, rec_1,
                                       rec_2,
                                       self.out_ub, repeat_time, 1,
                                       1, 1, 8, 8, 8)
                self.tik_instance.vmuls(self.max_eliments, rec_1,
                                        rec_1,
                                        -1, repeat_time, 1, 1, 8, 8)
                self.tik_instance.vadds(self.max_eliments, rec_1,
                                        rec_1,
                                        2, repeat_time, 1, 1, 8, 8)
                self.tik_instance.vmul(self.max_eliments, rec_1,
                                       rec_1,
                                       rec_2, repeat_time, 1,
                                       1, 1, 8, 8, 8)

                self.tik_instance.vmul(self.max_eliments, self.out_ub,
                                       rec_1,
                                       self.inter_area_ub, repeat_time, 1,
                                       1, 1, 8, 8, 8)

            iou_gm_offset = gt_global_index * self.bboxes_shape[
                0] + gm_offset + gm_out_offset
            nbust = _get_ceil_int(run_bb_point, self.eliments_per_block)
            self.tik_instance.data_move(self.iou_gm[iou_gm_offset],
                                        self.out_ub, 0, 1, nbust, 0, 0)

    def run_tik(self, kernel_name):
        """
        run_tik start tik process, and buid cce

        Parameters
        ----------
        kernel_name : str
            bbox segment len

        Returns
        -------
        result: tik_instance
            tik_instance
        """
        if self.gtboxes_shape[0] * 4 <= GTBOX_SEGMENT:
            self.iou_process()
        else:
            self.iou_process_cut_by_gt()
        self.tik_instance.BuildCCE(
            kernel_name=kernel_name,
            inputs=[self.bboxes_gm, self.gtboxes_gm],
            outputs=[self.iou_gm])
        return self.tik_instance

    def data_rerange(self, run_point, point_ub):
        """
        run_tik start tik process, and buid cce

        Parameters
        ----------
        run_point : int
            data range len
        point_ub : TVM tensor
            UB addr

        Returns
        -------
        None
        """
        for_range = _get_ceil_int(run_point, 2)
        self.index_reg = [
            self.tik_instance.Scalar(dtype=self.dtype) for _ in range(8)
        ]
        with self.tik_instance.for_range(0, for_range) as conv_index:
            for i in range(8):
                self.index_reg[i].set_as(point_ub[conv_index * 8 + i])
            for i in range(2):
                self.area_x0[conv_index * 2 + i] \
                    .set_as(self.index_reg[i * 4 + 0])
                self.area_y0[conv_index * 2 + i] \
                    .set_as(self.index_reg[i * 4 + 1])
                self.area_x1[conv_index * 2 + i] \
                    .set_as(self.index_reg[i * 4 + 2])
                self.area_y1[conv_index * 2 + i] \
                    .set_as(self.index_reg[i * 4 + 3])

    def calcu_area(self, run_point, area_ub, inter_mode=False):
        """
        run_tik start tik process, and buid cce

        Parameters
        ----------
        run_point : int
            data range len
        area_ub : TVM tensor
            UB addr
        inter_mode: bool
            calcu mode

        Returns
        -------
        None
        """
        if inter_mode is True:
            x0_ub = self.inter_area_x0
            x1_ub = self.inter_area_x1
            y0_ub = self.inter_area_y0
            y1_ub = self.inter_area_y1
        else:
            x0_ub = self.area_x0
            x1_ub = self.area_x1
            y0_ub = self.area_y0
            y1_ub = self.area_y1
        repeat_time = _get_ceil_int(run_point, self.max_eliments)
        # cala x1 - x0

        self.tik_instance.vsub(self.max_eliments, area_ub,
                               x1_ub,
                               x0_ub, repeat_time, 1,
                               1, 1, 8, 8, 8)

        self.tik_instance.vsub(self.max_eliments, self.area_y1_y0,
                               y1_ub,
                               y0_ub, repeat_time, 1,
                               1, 1, 8, 8, 8)
        self.tik_instance.vadds(self.max_eliments, area_ub,
                                area_ub, 1, repeat_time, 1, 1, 8,
                                8)
        self.tik_instance.vadds(self.max_eliments, self.area_y1_y0,
                                self.area_y1_y0, 1, repeat_time, 1, 1, 8,
                                8)
        # vmuls 0.2 to evade fp16 overflows
        self.tik_instance.vmuls(self.max_eliments, area_ub,
                                area_ub, 0.2, repeat_time, 1, 1, 8,
                                8)
        self.tik_instance.vmuls(self.max_eliments, self.area_y1_y0,
                                self.area_y1_y0, 0.2, repeat_time, 1, 1, 8,
                                8)
        if inter_mode is True:
            self.tik_instance.vmax(self.max_eliments, area_ub,
                                   self.zero_ub, area_ub,
                                   repeat_time, 1, 0, 1, 8, 0, 8)
            self.tik_instance.vmax(self.max_eliments, self.area_y1_y0,
                                   self.zero_ub, self.area_y1_y0,
                                   repeat_time, 1, 0, 1, 8, 0, 8)
        self.tik_instance.vmul(self.max_eliments, area_ub,
                               self.area_y1_y0,
                               area_ub, repeat_time, 1,
                               1, 1, 8, 8, 8)


def _box_shape_check(input_name, shape):
    """
    box_shape_check

    Parameters
    ----------
    input_name : str
        input name
    shape : tuple
        shape of input name

    Returns
    -------
    None
    """
    shape_len = len(shape)
    if shape_len != 2:
        raise RuntimeError("the shape len should be 2")
    last_shape_dim = shape[-1]
    if last_shape_dim != 4:
        raise RuntimeError("the shape should be [n, 4]")


# pylint: disable=unused-argument
@fusion_manager.register("iou")
def iou_compute(bboxes, gtboxes, overlap, mode, kernel_name):
    """
    calculating data

    Parameters
    ----------
    bboxes : dict
        shape and dtype of bboxes, the coordinates of bbox
        shape must be [n, 4]
        [x1, y1, x2, y2]
    gtboxes : dict
        shape and dtype of gtboxes, the coordinates of bbox
        shape must be [m, 4]
        [x1, y1, x2, y2]
    overlap : dict
        shape and dtype of overlap
        result shape is [m, n]
    mode :  str
        ('iou','iof')
        iou : the output is gtbox and bbox iou
        iof :
    kernel_name : str
        kernel name, default value is "iou"

    Returns
    -------
    output tensor
    """
    iou_res = Iou(bboxes, gtboxes, mode)

    return iou_res.run_tik(kernel_name)


@util.check_input_type(dict, dict, dict, str, str)
def iou(bboxes, gtboxes, overlap, mode="iou", kernel_name="iou"):
    """
    calculating data

    Parameters
    ----------
    bboxes : dict
        shape and dtype of bboxes, the coordinates of bbox
        shape must be [n, 4]
        [x1, y1, x2, y2]
    gtboxes : dict
        shape and dtype of gtboxes, the coordinates of bbox
        shape must be [m, 4]
        [x1, y1, x2, y2]
    overlap : dict
        shape and dtype of overlap
        result shape is [m, n]
    mode :  str
        ('iou','iof')
        iou : the output is gtbox and bbox iou
        iof :
    kernel_name : str
        kernel name, default value is "iou"

    Returns
    -------
    None
    """
    bboxes_shape = bboxes.get("shape")
    gtboxes_shape = gtboxes.get("shape")

    util.check_shape_rule(bboxes_shape)
    util.check_tensor_shape_size(bboxes_shape)
    util.check_shape_rule(gtboxes_shape)
    util.check_tensor_shape_size(gtboxes_shape)
    util.check_kernel_name(kernel_name)

    _box_shape_check("bboxes", bboxes_shape)
    _box_shape_check("gtboxes", gtboxes_shape)

    bboxes_dtype = bboxes.get("dtype").lower()
    util.compare_tensor_dict_key(bboxes, gtboxes, "dtype")
    check_list = ("float16", "float32")
    util.check_dtype_rule(bboxes_dtype, check_list)

    # check whether mode is valid
    check_list = ("iou", "iof")
    if mode not in check_list:
        raise RuntimeError("Mode only support iou and iof")

    res = iou_compute(bboxes, gtboxes, overlap, mode, kernel_name)

    return res
