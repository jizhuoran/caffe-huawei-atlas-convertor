#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from te import tvm
from .util import dtype_check_decorator, check_input_tensor_shape


@dtype_check_decorator
def concat(raw_tensors, axis):
    """
    concat shapes at axis,  support int8, uint8, int16, int32 float16, float32
    Parameters
    ----------
    raw_tensors : list of tensors
    axis : concat axis
    Returns
    -------
    concat tensor :
    """
    if axis < 0:
        axis = axis + len(raw_tensors[0].shape)
    concat_para_check(raw_tensors, axis)

    def _get_input_tensors():
        shapes = []
        for in_tensor in list(raw_tensors):
            shape = [int(in_tensor.shape[i].value) for i in range(len(in_tensor.shape))]
            shapes.append(shape)

        _shapes = list(shapes)
        return _shapes

    shapes = _get_input_tensors()

    res_shape = shapes[0][:]
    for i in range(1, len(shapes)):
        res_shape[axis] += shapes[i][axis]

    sel = []
    n_tensor = len(raw_tensors)

    def compute_func(*indice):
        """
        concat compute expr
        """
        if n_tensor > 1:
            for tensor_i in range(n_tensor - 1):
                if tensor_i == 0:
                    tensor_a = raw_tensors[0]
                    tensor_b = raw_tensors[1]
                    shape_c = shapes[0][:]
                    indice2 = list(indice[:])
                    indice2[axis] = indice[axis] - tensor_a.shape[axis]
                    sel.append(
                        tvm.select(indice[axis] < shape_c[axis],
                                   tensor_a[indice], tensor_b[tuple(indice2)]))
                    shape_c[axis] += shapes[1][axis]
                else:
                    tensor_a = sel[tensor_i - 1]
                    tensor_b = raw_tensors[tensor_i + 1]
                    indice2 = list(indice[:])
                    indice2[axis] = indice[axis] - shape_c[axis]
                    sel.append(tvm.select(indice[axis] < shape_c[axis],
                                          tensor_a, tensor_b[tuple(indice2)]))
                    shape_c[axis] += shapes[tensor_i + 1][axis]
        else:
            return raw_tensors[0][indice]

        return sel[-1]

    res = tvm.compute(res_shape, compute_func, name="concat", tag="concat")

    return res


def concat_para_check(raw_tensors, axis):
    """
    concat parameter check

    Parameters
    ----------
    raw_tensors : list of tensors
    axis : concat axis

    Returns
    -------
    rasie runtime error
    """
    if not isinstance(axis, int):
        raise RuntimeError("The axis type must be int")

    # check shape
    if axis < 0 or axis >= len(raw_tensors[0].shape):
        raise RuntimeError(
            "concat axis must be in [-%d - %d), actual is %d" \
            % (len(raw_tensors[0].shape), len(raw_tensors[0].shape), axis))
    check_input_tensor_shape(raw_tensors[0])
    for i in range(1, len(raw_tensors)):
        if not isinstance(raw_tensors[i], tvm.tensor.Tensor):
            raise RuntimeError("The each element of input type must be tvm.tensor")
        check_input_tensor_shape(raw_tensors[i])
        if raw_tensors[i].dtype != raw_tensors[0].dtype:
            raise RuntimeError("dtype must be the same to each other")
        for j in range(len(raw_tensors[0].shape)):
            if (j != axis) and (raw_tensors[i].shape[j].value != raw_tensors[0].shape[j].value):
                raise RuntimeError(
                    "concat input shape len must be the same to each other except concat axis")
