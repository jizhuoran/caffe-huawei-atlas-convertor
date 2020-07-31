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

tile
"""

from __future__ import absolute_import as _abs
from te import tvm


@tvm.tag_scope(tag="tile")
def compute_tile_cce(a_tuple):
    """Construct an array by construct an array by repeating a_tuple[0] the number of times given by a_tuple[1].

    Parameters
    ----------
    a_tuple : tuple list of tvm.Tensor
        The arrays to compute_tile_cce
    Returns
    -------
    ret : tvm.Tensor
    """
    assert isinstance(a_tuple, (list, tuple))

    out_shape = []
    for i in range(len(a_tuple)):
        if i == 0:
            for j in range(len(a_tuple[0].shape)):
                out_shape.append(a_tuple[i].shape[j])
        else:
            for j in range(len(a_tuple[0].shape)):
                out_shape[j] = out_shape[j]*a_tuple[i].shape[j]

    def _compute(*indices):
        for i in range(len(a_tuple[0].shape)):
            if i == 0:
                index = (indices[i] % a_tuple[0].shape[i],)
            else:
                index = index + (indices[i] % a_tuple[0].shape[i],)

        return a_tuple[0](*index)

    Output = tvm.compute(out_shape, _compute)
    return Output
