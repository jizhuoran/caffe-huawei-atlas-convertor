#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

util_frac_z
"""


from collections import Iterable


def is_frac_z(input_x):
    """
    judge the format is fractal Nz

    Parameters
    ----------
    input_x: dict
        shape, dtype and format of input

    Returns
    -------
    output: bool
        True if format is fractal Nz
    """
    return input_x.get("format").upper() == "FRACTAL_NZ"


def to_frac_z_axis(ori_shape, ori_axis):
    """
    judge the format is fractal Nz

    Parameters
    ----------
    ori_shape: list or tuple
        original shape of input
    ori_axis: list or tuple
        original axis of original shape to operate

    Returns
    -------
    output: list
        axis of the fractal Nz shape
    """
    if isinstance(ori_axis, Iterable):
        frac_z_axis = list(ori_axis)
    else:
        frac_z_axis = [ori_axis]
    shape_len = len(ori_shape)
    axis_count = len(frac_z_axis)
    axis_negative_1 = shape_len - 1
    axis_negative_2 = shape_len - 2
    for i in range(axis_count):
        axis_index = (frac_z_axis[i] + shape_len) % shape_len
        if axis_index == axis_negative_1:
            frac_z_axis[i] = axis_index - 1
            frac_z_axis.append(axis_index + 2)
        elif axis_index == axis_negative_2:
            frac_z_axis[i] = axis_index + 1
            frac_z_axis.append(axis_index + 2)
        else:
            frac_z_axis[i] = axis_index
    return frac_z_axis
