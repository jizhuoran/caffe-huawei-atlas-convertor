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

nz_2_nd
"""
# pylint: disable=too-many-lines,import-error
from te import tik
from topi.cce import util
from te import platform as tbe_platform
import math

# available number of cores
MAX_CORE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
# vector_repeat
MAX_REPEAT = 255
# block_size
MINI_UNIT = 32


def _get_output_shape(input_shape, paddings):
    """
    Derive the shape size of the output.
    """
    output_shape = [0 for _ in range(len(input_shape))]
    for i, _ in enumerate(zip(input_shape, paddings, output_shape)):
        output_shape[i] = paddings[i][0] + input_shape[i] + paddings[i][1]

    return output_shape


def _prod(values):
    """
    Prod the input values by multiply.
    """
    res = 1
    for value in values:
        res *= value

    return res


def _cal_core(total_core_loop_num, core_number, device_core_num):
    """
    calculate the loop number on each core

    input:
    total_core_loop_num : Virtual computing cores
    core_number : Actual cores
    device_core_num : Physical cores

    return:
    split_block_index : Watershed of different core
    list_out[i][0] : [0:i]  core_loop
    list_out[i+1][0]: [i+1,:] core_loop
    """
    list_out = []
    for block_index in range(core_number):
        core_loop = total_core_loop_num // core_number
        sum_core = ((core_loop + 1) * (total_core_loop_num % device_core_num) +
                    core_loop * (block_index - total_core_loop_num %
                                 device_core_num))
        if block_index < total_core_loop_num % device_core_num:
            core_loop = ((total_core_loop_num + core_number - 1) //
                         core_number)
            sum_core = (core_loop * block_index)

        list_in = [core_loop, sum_core]
        list_out.append(list_in)

    split_core_index = 0
    for i in range(len(list_out)-1):
        if list_out[i][0] != list_out[i+1][0]:
            return i, list_out[i][0], list_out[i+1][0]
        split_core_index += 1

    return split_core_index, list_out[0][0], list_out[0][0]


def _params_model(in_shape, ou_shape, core, ub_maxsize):
    # not consider padding of axis that is 0
    total_num = in_shape[0]
    core_num = core

    split_core_idx, \
    core_loop0, \
    core_loop1 = _cal_core(total_num, core_num, MAX_CORE)

    def _check(shape_in, shape_ou):
        model = []
        shape0 = shape_in
        shape1 = shape_ou
        for i in range(len(shape0)):
            if _prod(shape0) + _prod(shape1) <= ub_maxsize:
                model.append("ub_reorder")
            else:
                model.append("ub_move_out")
            shape0 = shape_in[i+1:]
            shape1 = shape_ou[i+1:]
        return model

    in_tiling_shape0 = in_shape.copy()
    in_tiling_shape0[0] = core_loop0
    in_tiling_shape1 = in_shape.copy()
    in_tiling_shape1[0] = core_loop1

    ou_tiling_shape0 = ou_shape.copy()
    ou_tiling_shape0[0] = core_loop0
    ou_tiling_shape1 = ou_shape.copy()
    ou_tiling_shape1[0] = core_loop1
    if core_loop1 != core_loop0:
        model0 = _check(in_tiling_shape0, ou_tiling_shape0)
        model1 = _check(in_tiling_shape1, ou_tiling_shape0)
    else:
        model0 = _check(in_tiling_shape0, ou_tiling_shape0)
        model1 = model0
    return split_core_idx, [core_loop0, core_loop1], [model0, model1]


class PadCompute(object):

    def __init__(self, in_shape, in_paddings, dtype, kernel_name):
        self.dtype = dtype
        self.in_shape = in_shape
        self.in_paddings = in_paddings
        self.kernel_name = kernel_name
        self.ou_shape = _get_output_shape(in_shape, in_paddings)
        self.core = in_shape[0]
        if self.core > MAX_CORE:
            self.core = MAX_CORE

        # make it 32B align
        if self.dtype == "float16":
            self.ub_maxsize = tbe_platform.\
                                  cce_conf.\
                                  get_soc_spec(tbe_platform.
                                               cce_conf.UB_SIZE) // 32 * 16
            self.mask = 128
            self.num_bit = 2
        else:
            self.ub_maxsize = tbe_platform.\
                                  cce_conf.\
                                  get_soc_spec(tbe_platform.
                                               cce_conf.UB_SIZE) // 32 * 8
            self.mask = 64
            self.num_bit = 4

    def set_tik_instance(self):
        """
        set tik_instance
        """
        tik_instance = tik.Tik()
        self.set_src_dst_tensor(tik_instance)

        return tik_instance

    def set_src_dst_tensor(self, tik_instance):
        """
        set input and output tensor
        """
        self.input_x_gm = tik_instance.Tensor(self.dtype,
                                              [_prod(self.in_shape), ],
                                              name="input_x_gm",
                                              scope=tik.scope_gm)
        self.output_y_gm = tik_instance.Tensor(self.dtype,
                                               [_prod(self.ou_shape), ],
                                               name="output_y_gm",
                                               scope=tik.scope_gm)

    def pad_case0(self, tik_instance, split_core_idx,
                  core_loop_list, model_list):

        with tik_instance.for_range(0, MAX_CORE,
                                    block_num=MAX_CORE) as blk_idx:
            # use as many as possible core (MAX_CORE)
            # outermost padding (top, bottom)
            # vec_mark: pad_vec_dup_outermost had worked
            # and model_list[0][x] is 'ub_reorder',it will be 'True'
            # if vec_mark = True, the followed
            # computation will not vec_dup again
            in_num_top = self.in_paddings[0][0] * _prod(self.ou_shape[1:])
            in_num_bottom = self.in_paddings[0][1] * _prod(self.ou_shape[1:])
            vec_mark = [False, False]
            if max(in_num_top, in_num_bottom) > 0:
                self.pad_vec_dup_outermost(tik_instance, in_num_top,
                                           in_num_bottom, blk_idx)
                # vec_dup doesn't care about core
                # different core must obey the same rule
                if model_list[0][0] != "ub_reorder":
                    vec_mark[0] = True
                if model_list[0][1] != "ub_reorder":
                    vec_mark[1] = True

            with tik_instance.if_scope(blk_idx <= split_core_idx):
                src_gm = 0
                dst_gm = in_num_top
                self._pad_case0_main(tik_instance, core_loop_list[0],
                                     model_list[0], blk_idx, src_gm,
                                     dst_gm, vec_mark[0])

            if core_loop_list[0] != core_loop_list[1]:
                with tik_instance.if_scope(tik.all(blk_idx > split_core_idx,
                                                   blk_idx < self.core)):
                    processed_in_shape = self.in_shape.copy()
                    processed_in_shape[0] = core_loop_list[0]
                    processed_ou_shape = self.ou_shape.copy()
                    processed_ou_shape[0] = core_loop_list[0]
                    src_gm += (split_core_idx + 1) * _prod(processed_in_shape)
                    dst_gm += (split_core_idx + 1) * _prod(processed_ou_shape)
                    blk_idx = blk_idx - split_core_idx - 1
                    self._pad_case0_main(tik_instance, core_loop_list[1],
                                         model_list[1], blk_idx, src_gm,
                                         dst_gm, vec_mark[1])

    def pad_case1(self, tik_instance):
        in_num = _prod(self.ou_shape)
        total_num = math.ceil(in_num * self.num_bit / MINI_UNIT)
        core_num = total_num
        if core_num > MAX_CORE:
            core_num = MAX_CORE

        split_core_index, \
        core_loop_before, \
        core_loop_after = _cal_core(total_num, core_num, MAX_CORE)
        ac_num_one = (MINI_UNIT // self.num_bit) * core_loop_before
        ac_num_two = (MINI_UNIT // self.num_bit) * core_loop_after

        with tik_instance.for_range(0, core_num,
                                    block_num=core_num) as blk_idx:
            if split_core_index + 1 == core_num:
                with tik_instance.if_scope(blk_idx <= core_num-1):
                    begin_index = blk_idx * ac_num_one
                    self._pad_case1_main(tik_instance, ac_num_one,
                                         begin_index, self.ubuf)

            else:
                with tik_instance.if_scope(blk_idx <= split_core_index):
                    begin_index = blk_idx * ac_num_one
                    self._pad_case1_main(tik_instance, ac_num_one,
                                         begin_index, self.ubuf)

                with tik_instance.if_scope(
                        tik.all(blk_idx > split_core_index,
                                blk_idx < core_num)):
                    begin_index = ac_num_one * (split_core_index + 1)
                    block_index = blk_idx - (split_core_index + 1)
                    begin_index += block_index * ac_num_two
                    self._pad_case1_main(tik_instance, ac_num_two,
                                         begin_index, self.ubuf)

    def _pad_case0_main(self, tik_instance, loop, model,
                        blk_idx, src_gm, dst_gm, vec_mark):

        in_shape = self.in_shape.copy()
        in_shape[0] = loop
        ou_shape = self.ou_shape.copy()
        ou_shape[0] = loop
        padding = self.in_paddings
        padding[0] = [0, 0]
        model = model
        axis = 0
        mark_reorder_first = False
        mark_out_first = vec_mark
        src_gm += blk_idx * _prod(in_shape)
        dst_gm += blk_idx * _prod(ou_shape)
        src_ub = 0
        dst_ub = 0

        self._recusive_case0(tik_instance, axis, in_shape, ou_shape, padding,
                             model, src_gm, dst_gm, src_ub, dst_ub,
                             mark_reorder_first, mark_out_first)

    def _pad_case1_main(self, tik_instance, in_num, begin_index, ubuf):

        tail = in_num // self.ub_maxsize
        tail_block = in_num % self.ub_maxsize

        def _main(serial, data_len, begin):
            begin += serial * self.ub_maxsize
            nburst = 1
            burstlen = data_len * self.num_bit // MINI_UNIT
            srcstride = 0
            dststride = 0

            tik_instance.data_move(ubuf[0],
                                   self.input_x_gm[begin],
                                   0,
                                   nburst,
                                   burstlen,
                                   srcstride,
                                   dststride)

            tik_instance.data_move(self.output_y_gm[begin],
                                   ubuf[0],
                                   0,
                                   nburst,
                                   burstlen,
                                   srcstride,
                                   dststride)
        if tail != 0:
            with tik_instance.for_range(0, tail) as serial:
                _main(serial, self.ub_maxsize, begin_index)

        if tail_block != 0:
            _main(tail, tail_block, begin_index)

    def pad_vec_dup_outermost(self, tik_instance, in_num_top,
                              in_num_bottom, blk_idx):

        top_index = 0
        bottom_index = _prod(self.ou_shape) - in_num_bottom
        in_num = max(in_num_top, in_num_bottom)
        vec_mark = False

        def _do_vec_dup(ac_num, vir_num, begin_index, block_index, mark):
            total_num = ac_num // (MINI_UNIT // self.num_bit)
            if total_num >= MAX_CORE:
                core_num = MAX_CORE
            else:
                core_num = total_num

            split_core_index, \
            core_loop_before, \
            core_loop_after = _cal_core(total_num, core_num, MAX_CORE)
            ac_num_one = (MINI_UNIT // self.num_bit) * core_loop_before
            ac_num_two = (MINI_UNIT // self.num_bit) * core_loop_after

            if not mark:
                self.set_vector_dup(tik_instance, vir_num, self.ubuf, 0)

            if split_core_index + 1 == core_num:
                with tik_instance.if_scope(block_index <= split_core_index):
                    begin_index += block_index * ac_num_one
                    self.copy_ubuf_2_gm_case01(tik_instance,
                                               ac_num_one,
                                               vir_num,
                                               self.ubuf,
                                               0, begin_index)

            else:
                with tik_instance.if_scope(block_index <= split_core_index):
                    begin_index_new = begin_index + block_index * ac_num_one
                    self.copy_ubuf_2_gm_case01(tik_instance,
                                               ac_num_one,
                                               vir_num,
                                               self.ubuf,
                                               0, begin_index_new)

                with tik_instance.if_scope(
                        tik.all(block_index > split_core_index,
                                block_index < core_num)):
                    begin_index += ac_num_one * (split_core_index + 1)
                    block_index = block_index - (split_core_index + 1)
                    begin_inde_new = begin_index + block_index * ac_num_two
                    self.copy_ubuf_2_gm_case01(tik_instance,
                                               ac_num_two,
                                               vir_num,
                                               self.ubuf,
                                               0, begin_inde_new)

        if in_num_top != 0:
            _do_vec_dup(in_num_top, in_num, top_index, blk_idx, vec_mark)
            vec_mark = True

        if in_num_bottom != 0:
            _do_vec_dup(in_num_bottom, in_num, bottom_index, blk_idx, vec_mark)

    def _recusive_case0(self, tik_instance, axis, in_shape,
                        ou_shape, padding, model, src_gm,
                        dst_gm, src_ub, dst_ub,
                        mark_reorder_first, mark_out_first):

        if axis == len(self.ou_shape):
            return tik_instance

        if model[axis] == "ub_reorder":

            if not mark_out_first and not mark_reorder_first:
                self.set_vector_dup(tik_instance,
                                    _prod(ou_shape[axis:]),
                                    self.ubuf, 0)
            if not mark_reorder_first:
                src_ub = _prod(ou_shape[axis:])
                #  in_num, dst_buf, src_ub, src_gm
                self.copy_gm_2_ubuf_case0(tik_instance,
                                          _prod(in_shape[axis:]),
                                          self.ubuf, src_ub, src_gm)

            top = padding[axis][0] * _prod(ou_shape[axis+1:])

            # -1 is last_dim
            # -2 is P * last_dim
            if axis < len(self.ou_shape) - 2:
                with tik_instance.for_range(0, in_shape[axis]) as i:
                    dst_ub += top + _prod(ou_shape[axis+1:]) * i
                    src_ub += _prod(in_shape[axis+1:]) * i
                    # in this mark_reorder_first must be True
                    self._recusive_case0(tik_instance, axis+1, in_shape,
                                         ou_shape, padding, model,
                                         src_gm, dst_gm, src_ub, dst_ub,
                                         True, mark_out_first)
            else:
                # in_num, src_dst_ubuf, src_ub, dst_ub
                total_num_ub = _prod(in_shape[axis:])
                if axis == len(self.ou_shape) - 1:
                    nburst = 1
                else:
                    nburst = in_shape[axis]
                burstlen = total_num_ub // nburst * self.num_bit // MINI_UNIT
                src_stride = 0
                dst_stride = (_prod(ou_shape[axis+1:]) -
                              _prod(in_shape[axis+1:])) * self.num_bit // MINI_UNIT
                if axis < len(self.ou_shape) - 1:
                    top += padding[axis+1][0] * _prod(ou_shape[axis+2:])
                dst_ub += top
                if nburst != 0 and burstlen != 0:
                    self.copy_ubuf_2_ubuf_case0(tik_instance, nburst, burstlen,
                                                src_stride, dst_stride, self.ubuf,
                                                src_ub, dst_ub)

            if not mark_reorder_first:
                self.copy_ubuf_2_gm_case00(tik_instance,
                                           _prod(ou_shape[axis:]),
                                           self.ubuf, 0, dst_gm)

        elif model[axis] == 'ub_move_out':
            # do_vec_dup:(top, bottom)
            # _recusive
            in_num_top = padding[axis][0] * _prod(ou_shape[axis+1:])
            in_num_bottom = padding[axis][1] * _prod(ou_shape[axis+1:])
            in_num = max(in_num_top, in_num_bottom)
            if in_num > 0:
                self.set_vector_dup(tik_instance, in_num, self.ubuf, 0)
                mark_out_first = True

                # top: actual_top_num, malloc_num, ubuf, src_ub, dst_gm
                if in_num_top > 0:
                    self.copy_ubuf_2_gm_case01(tik_instance, in_num_top,
                                               in_num, self.ubuf, 0, dst_gm)
                # bottom
                if in_num_bottom > 0:
                    dst_gm_bottom = dst_gm + self.in_shape[axis] * \
                                    _prod(self.ou_shape[axis+1:]) + in_num_top
                    self.copy_ubuf_2_gm_case01(tik_instance, in_num_bottom,
                                               in_num, self.ubuf, 0, dst_gm_bottom)
                # update
                dst_gm += in_num_top

            if axis < len(self.ou_shape) - 1:
                with tik_instance.for_range(0, in_shape[axis]) as i:
                    dst_gm += _prod(ou_shape[axis+1:]) * i
                    src_gm += _prod(in_shape[axis+1:]) * i
                    # in this mark_reorder_first must be True
                    self._recusive_case0(tik_instance, axis+1, in_shape, ou_shape,
                                         padding, model, src_gm, dst_gm,
                                         src_ub, dst_ub, mark_reorder_first,
                                         mark_out_first)

    def set_vector_dup(self, tik_instance, psm, dst, number):

        if psm > self.ub_maxsize:
            psm = self.ub_maxsize
        dup_psm = MAX_REPEAT * self.mask
        dup_repeat_merchant = psm // dup_psm
        dup_repeat_remainder = psm % dup_psm
        dst_blk_stride = 1
        dst_rep_stride = 8

        with tik_instance.for_range(0, dup_repeat_merchant) as i:
            tik_instance.vector_dup(self.mask,
                                    dst[0 + i * dup_psm],
                                    number,
                                    MAX_REPEAT,
                                    dst_blk_stride,
                                    dst_rep_stride)

        if dup_repeat_remainder != 0:
            repeats = dup_repeat_remainder // self.mask
            dup_remainder = dup_repeat_remainder % self.mask
            if repeats != 0:
                tik_instance.vector_dup(self.mask,
                                        dst[dup_repeat_merchant * dup_psm],
                                        number,
                                        repeats,
                                        dst_blk_stride,
                                        dst_rep_stride)
            if dup_remainder != 0:
                tik_instance.vector_dup(dup_remainder,
                                        dst[dup_repeat_merchant * dup_psm +
                                            repeats * self.mask],
                                        number,
                                        1,
                                        dst_blk_stride,
                                        dst_rep_stride)

    def copy_gm_2_ubuf_case0(self, tik_instance, in_num, ubuf, src_ub, src_gm):
        # ub must can be save all_data
        tik_instance.data_move(ubuf[src_ub],
                               self.input_x_gm[src_gm],
                               0,
                               1,
                               in_num * self.num_bit // 32,
                               0,
                               0
                               )

    def copy_ubuf_2_gm_case00(self, tik_instance, in_num, ubuf, src_ub, dst_gm):
        # ub must can be save all_data
        tik_instance.data_move(self.output_y_gm[dst_gm],
                               ubuf[src_ub],
                               0,
                               1,
                               in_num * self.num_bit // 32,
                               0,
                               0
                               )

    def copy_ubuf_2_gm_case01(self, tik_instance,
                              ac_num, vir_num, ubuf,
                              src_ub, dst_gm):
        if vir_num > self.ub_maxsize:
            vir_num = self.ub_maxsize

        tail = ac_num // vir_num
        tail_block = ac_num % vir_num

        def _copy_ub2gm(tik_instance, serial, data_len, dst):
            dst += serial * vir_num
            nburst = 1
            burstlen = data_len * self.num_bit // MINI_UNIT
            srcstride = 0
            dststride = 0

            tik_instance.data_move(self.output_y_gm[dst],
                                   ubuf[src_ub],
                                   0,
                                   nburst,
                                   burstlen,
                                   srcstride,
                                   dststride)

        if tail != 0:
            with tik_instance.for_range(0, tail) as serial:
                _copy_ub2gm(tik_instance, serial, vir_num, dst_gm)

        if tail_block != 0:
            _copy_ub2gm(tik_instance, tail, tail_block, dst_gm)

    def copy_ubuf_2_ubuf_case0(self, tik_instance, nburst, burstlen,
                               src_stride, dst_stride, ubuf, src_ub, dst_ub):
        # ub must can be save all_data
        tik_instance.data_move(ubuf[dst_ub],
                               ubuf[src_ub],
                               0,
                               nburst,
                               burstlen,
                               src_stride,
                               dst_stride
                               )

    def pad_compute(self):
        """
        the overall data move process
        """
        tik_instance = self.set_tik_instance()
        self.ubuf = tik_instance.Tensor(self.dtype,
                                        [self.ub_maxsize,],
                                        name="in_ubuf",
                                        scope=tik.scope_ubuf)

        if self.ou_shape != self.in_shape:
            split_core_idx, core_loop_list, model_list = \
                _params_model(self.in_shape, self.ou_shape,
                              self.core, self.ub_maxsize)

            self.pad_case0(tik_instance, split_core_idx,
                           core_loop_list, model_list)
        else:
            self.pad_case1(tik_instance)

        return tik_instance

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        tik_instance = self.pad_compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.input_x_gm],
                              outputs=[self.output_y_gm])

        return tik_instance

# pylint: disable=invalid-name,unused-argument
@util.check_input_type((list, tuple), (list, tuple), str, str)
def pad_align(shape, paddings, dtype, kernel_name):
    """
    condition:
    1.32B align
    2.len(in_shape)>=2
    3.dtype in ['float16', 'float32']
    4.in_shape[-1] + ou_shape[-1] must be saved in ub
    or not padding
    """
    in_shape = shape.copy()
    in_paddings = paddings.copy()
    result = PadCompute(in_shape, in_paddings, dtype, kernel_name)
    return result.get_tik_instance()
