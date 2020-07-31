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

space to batch nd
"""

from __future__ import absolute_import as _abs
from te import tvm
import topi.nn as nn
from functools import reduce as functools_reduce


def prod(alist):
    return functools_reduce(lambda x, y: x*y, alist, 1)


def _shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    tmp = []
    for i in shape:
        tmp.append(i.value)
    return tmp


def _get_pad_before_and_after(n, m, paddings, pad_before=None, pad_after=None):
    for i in range(n):
        if i >= 1 and i < 1 + m:
            pad_before.append(paddings[i - 1][0])
            pad_after.append(paddings[i - 1][1])
        else:
            pad_before.append(0)
            pad_after.append(0)


def compute_space_to_batch_nd_cce(input_tensor, block_shape, paddings):
    """
    This operation divides "spatial" dimensions [1, ..., M] of the input_tensor into a grid of blocks of shape block_shape, and interleaves these blocks with the "batch" dimension (0) such that in the output, the spatial dimensions [1, ..., M] correspond to the position within the grid, and the batch dimension combines both the position within a spatial block and the original batch position. Prior to division into blocks, the spatial dimensions of the input_tensor are optionally zero padded according to paddings.

    Parameters
    ----------
    input_tensor: N-D with shape input_shape = [batch] + spatial_shape + remaining_shape, where spatial_shape has M dimensions.
    block_shape: 1-D with shape [M], all values must be >= 1.
    paddings: 2-D with shape [M, 2], all values must be >= 0. paddings[i] = [pad_start, pad_end] specifies the padding for input_tensor dimension i + 1, which corresponds to spatial dimension i. It is required that block_shape[i] divides input_shape[i + 1] + pad_start + pad_end.
    Returns
    -------
    ret : tvm.Tensor
    """
    assert isinstance(block_shape, list)

    pad_before = []
    pad_after = []
    n = len(input_tensor.shape)
    m = len(block_shape)
    _get_pad_before_and_after(n, m, paddings, pad_before, pad_after)

    input_shape_padded = nn.pad(input_tensor, pad_before, pad_after)

    M = len(block_shape)
    batch = input_shape_padded.shape[0]
    spatial_shape = input_shape_padded.shape[1:1 + M]
    remain_shape = input_shape_padded.shape[1 + M:]

    oshape = [batch*prod(block_shape)] + [dim // bsize for dim, bsize in
                                            zip(spatial_shape, block_shape)] + remain_shape

    def map_index(*index):
        ibatch = index[0] % batch
        ispatial = list(index[1:1 + M])
        iremain = list(index[1 + M:])

        coef = index[0] // batch
        for i in reversed(range(M)):
            ispatial[i] = coef % block_shape[i] + index[1 + i]*block_shape[i]
            coef = coef // block_shape[i]

        return [ibatch] + ispatial + iremain

    Output = tvm.compute(oshape, lambda *i: input_shape_padded(*map_index(*i)), name='Output')
    return Output
