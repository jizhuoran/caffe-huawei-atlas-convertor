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

unsortedsegmentsum operators
"""
from __future__ import absolute_import as _abs
from te import tvm

@tvm.tag_scope(tag="unsortedsegmentsum")
def _unsortedsegmentsum_compute(Input, segment_ids, num_segments, indices):
    """compute_func of unsorted segment mean arithmetic operator

    Parameters
    ----------
    Input : tvm.Tensor
    
    segment_ids : list
        should be the size of the first dimension
        need not be sorted and need not cover all values in the full range of valid values.

    num_segments : uint
        should equal the number of distinct segment IDs

    Returns
    -------
    res : func

    """

    unique_id = []
    for i in segment_ids:
        if i not in unique_id:
            unique_id.append(i)

    def _compute_outer_dim(i):
        new_segment_id = list(segment_ids)[:]
        if i in unique_id:
            idx = new_segment_id.index(i)
            new_segment_id[idx] = -1
            tmp = Input[(idx,) + indices[1:]]
            for j in range(segment_ids.count(i) - 1):
                new_segment_id[idx] = -1
                idx = new_segment_id.index(i)
                tmp = Input[(idx,) + indices[1:]] + tmp
        else:
            tmp = tvm.const(0, Input.dtype)
        return tmp

    res = _compute_outer_dim(0)
    for i in range(num_segments)[1:]:
        res = tvm.select(indices[0] == i, _compute_outer_dim(i), res)
    return res


def compute_unsortedsegmentsum_cce(Input, segment_ids, num_segments):
    """cce unsorted segment mean arithmetic operator

    Parameters
    ----------
    Input : tvm.Tensor
    
    segment_ids :
        should be the size of the first dimension
        need not be sorted and need not cover all values in the full range of valid values.

    num_segments : uint
        should equal the number of distinct segment IDs

    Returns
    -------
    Output : tvm.Tensor

    """
    shape = tuple(Input.shape)

    if num_segments <= max(segment_ids):
        raise RuntimeError(
            "num_segments must be larger than max value of segment_ids, while num_segments is %d and max value of segment_ids is %d" % (
            num_segments, max(segment_ids)))

    output_shape = (num_segments,) + shape[1:]

    if not (Input.dtype.lower() in ["float16", "int32"]):
        cast_input = tvm.compute(shape, lambda *indices: Input[tuple(indices)].astype("float16"),
                                 name='cast_res')
        cast_output = tvm.compute(output_shape,
                                  lambda *indices: _unsortedsegmentsum_compute(cast_input,
                                                                               segment_ids,
                                                                               num_segments,
                                                                               indices),
                                  name='unsorted_res')
        Output = tvm.compute(output_shape,
                             lambda *indices: cast_output[tuple(indices)].astype(Input.dtype),
                             name='res')
    else:
        Output = tvm.compute(output_shape,
                             lambda *indices: _unsortedsegmentsum_compute(Input, segment_ids,
                                                                          num_segments, indices),
                             name='res')

    return Output
