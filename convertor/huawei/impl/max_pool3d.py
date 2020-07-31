#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tf max_pool3d
"""
from __future__ import absolute_import

# pylint: disable=E0401

from te import platform as cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_build import build_config
from te.platform.cce_conf import CceProductParams
from topi.cce import util

# shape limit
# int32's max value
SHAPE_SIZE_LIMIT = 2 ** 31 - 1
# c0 size
C0SIZE = 16

NoneType = type(None)


def check_window_rule(ksize, strides):
    """
    check ksize and strides of window in pooling3d
    """
    if len(ksize) != 1 and len(ksize) != 3 and len(ksize) != 5:
        raise RuntimeError("Invalid ksize params, "
                           "ksize dim must be 1 or 3 or 5.")

    if len(strides) != 1 and len(strides) != 3 and len(strides) != 5:
        raise RuntimeError("Invalid strides params, "
                           "strides dim must be 1 or 3 or 5.")

    # check support shape
    if len(ksize) == 1 and ksize[0] != 2:
        raise RuntimeError("Invalid ksize params, "
                           "ksize D H W value only support 2.")

    if len(ksize) == 3 and (ksize[0] != 2 or ksize[1] != 2 or ksize[1] != 2):
        raise RuntimeError("Invalid ksize params, "
                           "ksize D H W value only support 2.")

    ksize_check = ksize[0] != 1 or ksize[1] != 2 or \
                  ksize[2] != 2 or ksize[3] != 2 or ksize[4] != 1
    if len(ksize) == 5 and ksize_check:
        raise RuntimeError("Invalid ksize params, ksize D H W value "
                           "only support 2, and N C only support 1.")

    if len(strides) == 1 and strides[0] != 2:
        raise RuntimeError("Invalid strides params, "
                           "strides D H W value only support 2.")

    if len(strides) == 3 and (strides[0] != 2 or
                              strides[1] != 2 or strides[1] != 2):
        raise RuntimeError("Invalid strides params, "
                           "strides D H W value only support 2.")

    strides_check = strides[0] != 1 or strides[1] != 2 or \
                    strides[2] != 2 or strides[3] != 2 or strides[4] != 1
    if len(strides) == 5 and strides_check:
        raise RuntimeError("Invalid strides params, strides D H W value "
                           "only support 2, and N C only support 1.")


def check_padding(padding):
    """
    check padding in pooling3d
    """
    if padding not in ("SAME", "VALID"):
        raise RuntimeError("max_pool3d can only "
                           "support SAME or VALID padding mode.")


# pylint: disable=too-many-arguments,unused-argument,invalid-name
def max_pool3d_check_rule(input_shape, output_dtype, ksize, strides,
                          padding, data_format, kernel_name):
    """
    :param input_shape: shape of input_data
    :param output_dtype: dtype of output_data
    :param ksize: the window of max_pool3d
    :param strides: the stride of max_pool3d window
    :param padding : str, the mode of padding, support SAME or VALID
    :param data_format: NDHWC default
    :param kernel_name: cce kernel name
    :return: None
    """
    # check input and output
    util.check_shape_size(input_shape, SHAPE_SIZE_LIMIT)
    util.check_shape_rule(input_shape)
    util.check_dtype_rule(output_dtype, ["float16"])
    # check ksize and strides of window
    check_window_rule(ksize, strides)
    # check kernel name
    util.check_kernel_name(kernel_name)
    # check padding
    check_padding(padding)


# pylint: disable=too-many-locals,too-many-arguments
# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("max_pool3d")
def max_pool3d_compute(x, y, ksize, strides,
                       padding="VALID", data_format="NDHWC",
                       kernel_name="max_pool3d"):
    """
    describe compute
    return: tensor
    """
    shape = x.shape

    # copy gm to ub
    tensor_in_ub = tvm.compute(shape, lambda *i: x[i], name="tensor_in_ub")

    # vmax in W
    shape_w = (shape[0], shape[1], shape[2], shape[3] // 2, shape[4])
    tensor_w = tvm.compute(shape_w,
                           lambda n, d, h, w, c:
                           tvm.max(tensor_in_ub[n, d, h, 2 * w, c],
                                   tensor_in_ub[n, d, h, 2 * w + 1, c]),
                           name='tensor_w')

    # vmax in H
    shape_h = (shape[0], shape[1], shape[2] // 2, shape[3] // 2, shape[4])
    tensor_h = tvm.compute(shape_h,
                           lambda n, d, h, w, c:
                           tvm.max(tensor_w[n, d, 2 * h, w, c],
                                   tensor_w[n, d, 2 * h + 1, w, c]),
                           name='tensor_h')

    # vmax in D
    shape_d = (shape[0], shape[1] // 2,
               shape[2] // 2, shape[3] // 2, shape[4])
    tensor_d = tvm.compute(shape_d,
                           lambda n, d, h, w, c:
                           tvm.max(tensor_h[n, 2 * d, h, w, c],
                                   tensor_h[n, 2 * d + 1, h, w, c]),
                           name='tensor_d')

    # copy ub to gm
    res = tvm.compute(shape_d, lambda *i: tensor_d[i], name='res')

    return res


def max_pool3d_tiling(shape):
    """
    max_pool3d tiling strategy
    """
    ub_size = CceProductParams().getParams("Unified_Buffer")

    # each core process 2*d
    per_d = 2
    input_h = shape[2].value
    input_w = shape[3].value
    input_c = shape[4].value

    for i in range(input_h // 2, 0, -1):
        tensor_in_size = per_d * i * 2 * input_w * input_c

        # 2 mean need space in ub; 2 mean pingpong
        if tensor_in_size * 2 < ub_size // 2:
            return i

    raise RuntimeError("cannot find tiling in H")


def max_pool3d_schedule(res, sch):
    """
    max_pool3d schedule
    """
    tensor_d = res.op.input_tensors[0]
    tensor_h = tensor_d.op.input_tensors[0]
    tensor_w = tensor_h.op.input_tensors[0]
    tensor_in_ub = tensor_w.op.input_tensors[0]

    # set scope
    sch[tensor_in_ub].set_scope(cce.scope_ubuf)
    sch[tensor_w].set_scope(cce.scope_ubuf)
    sch[tensor_h].set_scope(cce.scope_ubuf)
    sch[tensor_d].set_scope(cce.scope_ubuf)

    # double buffer
    sch[tensor_in_ub].double_buffer()
    sch[tensor_in_ub].preload()
    sch[tensor_w].double_buffer()
    sch[tensor_h].double_buffer()
    sch[tensor_d].double_buffer()

    # bind core
    res_1o, _ = sch[res].split(res.op.axis[1], factor=1)
    thread_block = tvm.thread_axis("blockIdx.x")
    sch[res].bind(res_1o, thread_block)

    # tiling
    cut_h = max_pool3d_tiling(tensor_in_ub.shape)
    res_2o, res_2i = sch[res].split(res.op.axis[2], factor=cut_h)

    # compute at
    sch[tensor_w].compute_at(sch[res], res_2o)
    sch[tensor_h].compute_at(sch[res], res_2o)
    sch[tensor_d].compute_at(sch[res], res_2o)
    sch[tensor_in_ub].compute_at(sch[res], res_2o)

    # emit insn
    sch[tensor_in_ub].emit_insn(tensor_in_ub.op.axis[0], 'dma_copy')
    sch[tensor_w].emit_insn(tensor_w.op.axis[0], 'vector_max')
    sch[tensor_h].emit_insn(tensor_h.op.axis[0], 'vector_max')
    sch[tensor_d].emit_insn(tensor_d.op.axis[0], 'vector_max')
    sch[res].emit_insn(res_2i, 'dma_copy')


# pylint: disable=too-many-arguments,unused-argument,invalid-name
@util.check_input_type(dict, dict, (list, tuple), (list, tuple),
                       str, str,
                       str)
def max_pool3d(x, y, ksize, strides, padding,
               data_format="NDHWC", kernel_name="max_pool3d"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data,
        only support float16, shape is 5 dims, format is NDHWC

    y : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of max_pool3d,
            only support max_pool3d in D or H or W

    strides : list or tuple, the stride of max_pool3d window,
            only support max_pool3d in D or H or W

    padding : str, the mode of padding, support SAME or VALID

    data_format : str, default = "NDHWC"

    kernel_name : cce kernel name, default value is "max_pool3d"

    Returns
    -------
    None
    """
    # get shape&dtype
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    input_dtype = input_dtype.lower()
    output_dtype = y.get("dtype")
    output_dtype = output_dtype.lower()

    # check others parameter
    max_pool3d_check_rule(input_shape, output_dtype,
                          ksize, strides, padding, data_format, kernel_name)

    # compute
    # create tensor_in
    tensor_in = tvm.placeholder(input_shape,
                                name="tensor_in", dtype=input_dtype)

    res = max_pool3d_compute(tensor_in, y, ksize, strides,
                             padding, data_format, kernel_name)

    # schedule
    sch = tvm.create_schedule(res.op)
    max_pool3d_schedule(res, sch)

    with build_config:
        tvm.build(sch, [tensor_in, res], "cce", name=kernel_name)
