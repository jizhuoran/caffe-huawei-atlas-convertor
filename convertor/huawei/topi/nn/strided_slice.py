#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=invalid-name, unused-variable
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

strided slice
"""

from __future__ import absolute_import as _abs

import copy
import math

from te import tvm


@tvm.tag_scope(tag="strided_slice")
def _shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    tmp = []
    for i in shape:
        tmp.append(i.value)
    return tmp


def _fill_list_with_ones(length):
    """
    fill a list array with ones
    """
    tmp = []
    for i in range(length):
        tmp.append(1)
    return tmp


def _init_begin_end(input_shape, begin_shape, end_shape, begin_mask, end_mask, ellipsis_mask):
    for i in range(len(input_shape)):
        # if ellipsis_mask is set, not need to make supplement for begin_shape and end_shape,
        # because if the ith bit of ellipsis_mask is set, 
        # as many unspecified dimensions as needed will be inserted between other dimensions.
        # and Only one non-zero bit is allowed in ellipsis_mask.

        if (i > len(begin_shape) - 1 and ellipsis_mask == 0):
            begin_shape.append(0)
            end_shape.append(input_shape[i])

        # If the ith bit of begin_mask is set, begin[i] is ignored,
        # and the fullest possible range in that dimension is used instead.
        # end_mask works analogously, except with the end range.

        # check begin_mask
        if ((begin_mask & 2**i) == 2**i):
            if (begin_shape[i] > 0):
                begin_shape[i] = 0
            elif (begin_shape[i] < 0):
                begin_shape[i] = -1

        # check end_mask
        if ((end_mask & 2**i) == 2**i):
            if (end_shape[i] > 0):
                end_shape[i] = input_shape[i]
            elif (end_shape[i] < 0):
                end_shape[i] = input_shape[i]*(-1) - 1
    return


def _update_begin_end(input_shape, begin_shape, end_shape, new_axis_mask):
    acc_del = 0
    for i in range(len(input_shape)):
        # if the ith bit of ellipsis_mask is set, the ith bit of new_axis_mask is ignored

        if (new_axis_mask & 2**i == 2**i):
            del (begin_shape[i - acc_del])
            del (end_shape[i - acc_del])
            acc_del += 1

    return acc_del


# to handle ellipsis_mask
def _update_begin_end_on_ellipsis(input_shape, begin_shape, end_shape, ellipsis_mask):
    diff = len(input_shape) - len(begin_shape)
    first_flag = True
    for i in range(len(input_shape)):
        if ((ellipsis_mask & 2**i) == 2**i and first_flag):
            if (end_shape[i] > 0):
                begin_shape[i] = 0
                end_shape[i] = input_shape[i]
            elif (end_shape[i] < 0):
                begin_shape[i] = -1
                end_shape[i] = input_shape[i]*(-1) - 1

            # handle the situation that new_axis_mask is set or len(begin_shape) < len(input_shape)
            if diff > 0:
                ellipsis_mask = ellipsis_mask | 2**(i + 1)
                diff = diff - 1
                first_flag = False

        elif ((ellipsis_mask & 2**i) == 2**i and (not first_flag)):
            begin_shape.insert(i, 0)
            end_shape.insert(i, input_shape[i])

            if diff > 0:
                ellipsis_mask = ellipsis_mask | 2**(i + 1)
                diff = diff - 1
    return ellipsis_mask


def compute_strided_slice_cce(input_data, begin, end, stride_shape=None, begin_mask=0, end_mask=0,
                              ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    """
    select_run_branch: 0:norm  1:new_axis 2:shrink_axis 3: new_axis and shrink_axis
    """
    select_run_branch = 0

    begin_shape = copy.deepcopy(begin)
    end_shape = copy.deepcopy(end)

    input_shape = _shape_to_list(input_data.shape)

    length = len(input_shape)

    # additional bit of ellipsis_mask and new_axis_mask should be ignored
    ellipsis_mask = ellipsis_mask & (2**length - 1)
    new_axis_mask = new_axis_mask & (2**length - 1)
    for i in range(len(input_shape)):
        # if the ith bit of ellipsis_mask is set, the ith bit of new_axis_mask is ignored
        if ((new_axis_mask & 2**i == ellipsis_mask & 2**i)):
            new_axis_mask = new_axis_mask & (~(2**i))

        # if the ith bit of ellipsis_mask is set, the ith bit of shrink_axis_mask is ignored
        if ((shrink_axis_mask & 2**i == ellipsis_mask & 2**i)):
            shrink_axis_mask = shrink_axis_mask & (~(2**i))

    # update begin_shape, end_shape
    _init_begin_end(input_shape, begin_shape, end_shape, begin_mask, end_mask, ellipsis_mask)

    if (new_axis_mask > 0 and ellipsis_mask > 0):
        acc_del = _update_begin_end(input_shape, begin_shape, end_shape, new_axis_mask)

    # adjust ellipsis_mask to handle the condition that ellipsis_mask > new_axis_mask
    # for example ellipsis_mask=2, new_axis_mask=1
    is_ellipsis_mask_great_than_new_axis_mask = False
    if (ellipsis_mask > new_axis_mask):
        for i in range(length):
            if (new_axis_mask & 2**i) == 2**i:
                ellipsis_mask = ellipsis_mask >> (i + 1)
                is_ellipsis_mask_great_than_new_axis_mask = True
                break

    # to handle ellipsis_mask
    if ellipsis_mask > 0:
        ellipsis_mask = _update_begin_end_on_ellipsis(input_shape, begin_shape, end_shape,
                                                      ellipsis_mask)

    if stride_shape == None:
        stride_shape = _fill_list_with_ones(len(begin_shape))

    # to handle only new_axis_mask
    if new_axis_mask > 0 and ellipsis_mask == 0:
        _update_begin_end(input_shape, begin_shape, end_shape, new_axis_mask)

        # adjust end_shape after new_axis_process
        for i in range(length):
            if (i > len(begin_shape) - 1):
                begin_shape.append(0)
                end_shape.append(input_shape[i])
            if (end_shape[i] > input_shape[i]):
                end_shape[i] = input_shape[i]

    # handle only shrink_axis_mask
    if shrink_axis_mask > 0 and new_axis_mask == 0:
        for i in range(length):
            if ((shrink_axis_mask & 2**i) == 2**i):
                end_shape[i] = begin_shape[i] + 1

    oshape = [int(math.ceil((end - begin)/(stride*1.0))) for begin, end, stride in
              zip(begin_shape, end_shape, stride_shape)]

    for i in range(length):
        if (begin_shape[i] <= -1 and end_shape[i] >= input_shape[i]*(-1) - 1 and begin_shape[i] >
                end_shape[i]):
            begin_shape[i] += input_shape[i]
            end_shape[i] += input_shape[i]

    # To construct oshape_new_axis accord to new_axis_mask
    oshape_new_axis = copy.deepcopy(oshape)
    for i in range(length):
        if ((new_axis_mask & 2**i) == 2**i):
            select_run_branch = 1
            if (
                    new_axis_mask > 0 and ellipsis_mask > 0 and not is_ellipsis_mask_great_than_new_axis_mask):
                oshape_new_axis.insert(i + acc_del, 1)
            else:
                oshape_new_axis.insert(i, 1)

    if (new_axis_mask > 0 and ellipsis_mask > 0 and not is_ellipsis_mask_great_than_new_axis_mask):
        new_axis_mask = new_axis_mask << 1

    length_oshape_new_axis = len(oshape_new_axis)

    if shrink_axis_mask > 0 and new_axis_mask > 0:
        oshape_shrink_axis_comp = copy.deepcopy(oshape_new_axis)
        acc_shrink = 0
        for i in range(len(oshape_new_axis)):
            if ((shrink_axis_mask & 2**i == new_axis_mask & 2**i)):
                shrink_axis_mask = shrink_axis_mask & (~(2**i))

            if ((shrink_axis_mask & 2**i) == 2**i):
                select_run_branch = 3
                del (oshape_shrink_axis_comp[i - acc_shrink])
                acc_shrink += 1

    # To construct oshape_shrink_axis accord to shrink_axis_mask
    oshape_shrink_axis = copy.deepcopy(oshape)
    if shrink_axis_mask > 0 and new_axis_mask == 0:
        acc_shrink = 0
        for i in range(length):
            if ((shrink_axis_mask & 2**i) == 2**i):
                select_run_branch = 2
                del (oshape_shrink_axis[i - acc_shrink])
                acc_shrink += 1

    length_oshape_shrink_axis = len(oshape_shrink_axis)

    # calculate index
    def map_index_norm(*index):
        for i in range(length):
            if i == 0:
                index_org = (index[i]*stride_shape[i] + begin_shape[i],)
            else:
                index_org = index_org + (index[i]*stride_shape[i] + begin_shape[i],)
        return index_org

    # add new axis
    def map_index_new_axis(*index):
        loc = 0
        for i in range(length_oshape_new_axis):
            if ((new_axis_mask & 2**i) != 2**i):
                if loc == 0:
                    index_org_new_axis = (index[i] + begin_shape[loc],)
                    loc += 1
                else:
                    index_org_new_axis = index_org_new_axis + (index[i] + begin_shape[loc],)
                    loc += 1
        return index_org_new_axis

    # shrink old axis
    def map_index_shrink_axis(*index):
        loc = 0
        for i in range(length):
            if ((shrink_axis_mask & 2**i) != 2**i):
                if i == 0:
                    index_org_shrink_axis = (index[loc] + begin_shape[i],)
                else:
                    index_org_shrink_axis = index_org_shrink_axis + (index[loc] + begin_shape[i],)
                loc += 1
            else:
                if i == 0:
                    index_org_shrink_axis = (0 + begin_shape[i],)
                else:
                    index_org_shrink_axis = index_org_shrink_axis + (0 + begin_shape[i],)
        return index_org_shrink_axis

        # shrink old axis

    def map_index_shrink_axis_and_new(*index):
        loc = 0
        for i in range(len(oshape_new_axis)):
            if ((shrink_axis_mask & 2**i) != 2**i):
                if i == 0:
                    index_org_shrink_axis = (index[loc],)
                else:
                    index_org_shrink_axis = index_org_shrink_axis + (index[loc],)
                loc += 1
            else:
                if i == 0:
                    index_org_shrink_axis = (0,)
                else:
                    index_org_shrink_axis = index_org_shrink_axis + (0,)
        return index_org_shrink_axis

    if select_run_branch == 0:
        Output = tvm.compute(oshape, lambda *i: input_data(*map_index_norm(*i)), name='Output')

    elif select_run_branch == 1:
        Output = tvm.compute(oshape_new_axis, lambda *i: input_data(*map_index_new_axis(*i)),
                             name='Output')

    elif select_run_branch == 3:
        mid = tvm.compute(oshape_new_axis, lambda *i: input_data(*map_index_new_axis(*i)),
                          name='mid')
        Output = tvm.compute(oshape_shrink_axis_comp,
                             lambda *i: mid(*map_index_shrink_axis_and_new(*i)), name='Output')

    else:
        Output = tvm.compute(oshape_shrink_axis, lambda *i: input_data(*map_index_shrink_axis(*i)),
                             name='Output')

    return Output
