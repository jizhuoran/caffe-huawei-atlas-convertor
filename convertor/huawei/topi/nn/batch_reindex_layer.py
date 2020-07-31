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

batch reindex layer
"""

from __future__ import absolute_import as _abs
from te import tvm


@tvm.tag_scope(tag="batch_reindex_layer")
def compute_batch_reindex_layer_cce(input_data, permut):
    """
    Parameters
    ----------
    input_data : tvm.Tensor
    permut: The array list
    Returns
    -------
    ret : tvm.Tensor
    """
    assert isinstance(permut, list)

    oshape = [len(permut), ] + input_data.shape[1:]

    def map_index(*index):
        idx = index[0]
        in_n = permut[0]
        for i in range(len(permut))[1:]:
            in_n = tvm.select((idx == i), permut[i], in_n)
        return [in_n, ] + list(index[1:])

    output_data = tvm.compute(oshape, lambda *i: input_data(*map_index(*i)), name='output')

    return output_data
