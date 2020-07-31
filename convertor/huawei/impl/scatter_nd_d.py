#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

scatter_nd_d
"""
from te import tik
from te import platform as tbe_platform
from topi.cce import util
from impl import scatter_nd_d_help
from impl import constant_util as constant
from impl import common_util



# pylint: disable=invalid-name, too-many-locals
@util.check_input_type(dict, dict, dict, (tuple, list), str)
def scatter_nd_d(indices, x, y, shape, kernel_name="scatter_nd_d"):
    """
    the main function of scatter_nd_d

    Parameters
    ----------
    indices: dict,shape and datatype,datatype supports int32
    x: dict,shape and datatype,datatype supports float32,float16,int32,
       int8,uint8
    y: dict,shape and datatype,datatype supports float32,float16,int32,
       int8,uint8
    shape: out put shape
    kernel_name: cce kernel name, default value is "scatter_nd_d"

    Returns
    -------
    tik_instance: tik_instance
    """
    check_param(indices, x, y, shape, kernel_name)

    if _check_1d_updates(indices, x, y):
        return _scatter_nd_d_1d(indices, x, y, kernel_name)

    indices_shape = indices.get("shape")
    indice_len = scatter_nd_d_help.get_indice_len(indices_shape)
    update_each_size = scatter_nd_d_help.get_shape_total_number(
        x.get("shape")) // indice_len
    block_dim, loop_cycle = get_blockdim_and_loop_cycle(x, shape,
                                                        update_each_size)
    output_shape = scatter_nd_d_help.get_shape_total_number(shape)
    output_spilts = output_shape // update_each_size
    last_spilt = output_spilts - output_spilts // block_dim*block_dim
    tik_instance = tik.Tik()
    input_param = (indices, x, y, shape)
    scatter = scatter_nd_d_help.ScatterNd(input_param, tik_instance)

    with tik_instance.for_range(0, block_dim,
                                block_num=block_dim) as block_id:
        process = scatter_nd_d_help.ScatterProcess(scatter.tik_instance,
                                                   scatter.updates,
                                                   scatter.indices,
                                                   scatter.shape)
        cycle_each_block = tik_instance.Scalar("int32")
        cycle_each_block.set_as(loop_cycle)
        output_offset = tik_instance.Scalar("int32")
        output_size = tik_instance.Scalar("int32")
        output_size.set_as(cycle_each_block*process.update_each_size)

        with tik_instance.if_scope(tik.all(block_dim == \
                                           constant.MAX_BLOCK_NUMBER,
                                           last_spilt != 0,
                                           block_id < last_spilt)):
            cycle_each_block.set_as(loop_cycle + 1)
            output_size.set_as(cycle_each_block*process.update_each_size)
            output_offset.set_as(block_id*output_size)
        with tik_instance.else_scope():
            output_offset.set_as(block_id*output_size + \
                                 last_spilt*process.update_each_size)

        scatter.initial_output(process, output_offset, output_size)
        scatter.update_data(process, cycle_each_block, output_offset)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=(scatter.input_indices_gm,
                                  scatter.input_updates_gm),
                          outputs=(scatter.output_y_gm),
                          enable_l2=False)
    return tik_instance


def get_blockdim_and_loop_cycle(updates, shape_out, update_each_size):
    """
    get blockdim and loop cycle

    Parameters
    ----------
    updates: dict,shape and datatype,datatype supports float32,float16,int32,
       int8,uint8
    shape_out: dict,shape and datatype,datatype supports float32,float16,int32,
       int8,uint8
    update_each_size: the elements number of each update data

    Returns
    -------
    None
    """
    blockdim = tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.CORE_NUM)
    data_size = common_util.get_data_size(updates.get("dtype").lower())
    output_shape = scatter_nd_d_help.get_shape_total_number(shape_out)
    output_spilts = output_shape // update_each_size

    # update_each_size less than 32b,use one core,
    # beacuse Less than 32B alignment to prevent multi-core coverage,
    # using single-core processing
    if update_each_size*data_size < constant.BLOCK_SIZE:
        return 1, output_spilts
    if output_spilts < blockdim:
        return output_spilts, output_spilts // output_spilts

    return blockdim, output_spilts // blockdim


def check_param(indices, updates, output_y, shape, kernel_name):
    """
    check the parameters is valid, if one is invalid,then raise error

    Parameters
    ----------
    indices: dict,shape and datatype,datatype supports int32
    updates: dict,shape and datatype,datatype supports float32,float16,int32,
             int8,uint8
    output_y: dict,shape and datatype,datatype supports float32,float16,int32,
              int8,uint8
    shape: out put shape
    kernel_name: cce kernel name, default value is "scatter_nd_d"

    Returns
    -------
    None
    """
    if len(indices.get("shape")) == 1:
        indices["shape"] = (indices.get("shape")[0], 1)
    if len(updates.get("shape")) == 1:
        updates["shape"] = (updates.get("shape")[0], 1)
    if len(output_y.get("shape")) == 1:
        output_y["shape"] = (output_y.get("shape")[0], 1)
    if len(shape) == 1:
        shape = list((shape[0], 1))
    indices_shape = indices.get("shape")
    indices_dtype = indices.get("dtype").lower()
    updates_shape = updates.get("shape")
    updates_dtype = updates.get("dtype").lower()
    y_shape = output_y.get("shape")
    y_dtype = output_y.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(indices_shape)
    util.check_tensor_shape_size(indices_shape)
    util.check_dtype_rule(indices_dtype, (constant.DATA_TYPE_INT32))

    util.check_shape_rule(updates_shape)
    util.check_tensor_shape_size(updates_shape)
    util.check_dtype_rule(updates_dtype, (
        constant.DATA_TYPE_FP16, constant.DATA_TYPE_FP32,
        constant.DATA_TYPE_INT32,
        constant.DATA_TYPE_INT8, constant.DATA_TYPE_UINT8))

    util.check_shape_rule(y_shape)
    util.check_tensor_shape_size(y_shape)
    util.check_dtype_rule(y_dtype, (
        constant.DATA_TYPE_FP16, constant.DATA_TYPE_FP32,
        constant.DATA_TYPE_INT32,
        constant.DATA_TYPE_INT8, constant.DATA_TYPE_UINT8))

    if updates_dtype != y_dtype:
        raise RuntimeError("updates's datatype must be the same as output_y's datatype")

    if not check_same_shape(y_shape, shape):
        raise RuntimeError(
            "y's shape must be the same as shape")


def check_same_shape(shape_x, shape_y):
    """
    check shape_x is the same shape as shape_y

    Parameters
    ----------
    shape_x: a tuple or list
    shape_y: a tuple or list

    Returns
    -------
    boolean: True has the same shape, False does't has the same shape
    """
    shape_x_len = len(shape_x)
    shape_y_len = len(shape_y)
    if shape_x_len != shape_y_len:
        return False
    for k in range(shape_x_len):
        if shape_x[k] != shape_y[k]:
            return False

    return True


# pylint: disable=invalid-name, too-many-return-statements
def _check_1d_updates(indices, updates, output_y):
    """
    check if updates are 1-D shape or not
    """
    # only support v100 cloud
    if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") not in ("Ascend910",):
        return False
    indices_shape = indices.get("shape")
    updates_shape = updates.get("shape")
    outputy_shape = output_y.get("shape")
    dtype = updates.get("dtype")
    if dtype != 'float32':
        return False
    if len(indices_shape) != 2 or len(updates_shape) != 2 or len(
            outputy_shape) != 2:
        return False
    if indices_shape[-1] != 1 or updates_shape[-1] != 1 or outputy_shape[
            -1] != 1:
        return False
    indices_len = indices_shape[0]
    output_y_len = outputy_shape[0]
    # indices num must less than 100W
    if indices_len > 1000000:
        return False
    # output num must less than 1000W
    if output_y_len > 10000000:
        return False
    return True


def _zero_ub(tik_instance, dst, buf_len, dtype):
    """
    zero a ub memory
    """
    dup_len = 128
    if dtype in ('float32', 'int32'):
        dup_len = 64
    if dtype in ('int8', 'uint8'):
        dup_len = 256
    repeat = buf_len // dup_len
    remain = buf_len % dup_len
    if repeat > 255:
        repeat_255 = repeat // 255
        with tik_instance.for_range(0, repeat_255) as i:
            tik_instance.vector_dup(dup_len, dst[i*255*dup_len], 0, 255, 1,
                                    8, 0)

        repeat_remain = (buf_len - repeat_255*255*dup_len) // dup_len

        if repeat_remain > 0:
            tik_instance.vector_dup(dup_len, dst[repeat_255*255*dup_len], 0,
                                    repeat_remain, 1, 8, 0)

    elif repeat > 0:
        tik_instance.vector_dup(dup_len, dst, 0, repeat, 1, 8, 0)

    if remain > 0:
        tik_instance.vector_dup(remain, dst[repeat*dup_len], 0, 1, 1, 8, 0)


def _get_core_var_elements(var):
    """
    calculate how many output elements one core will process at most
    """
    shape = var.get("shape")
    dtype = var.get("dtype")
    dtype_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
    blockdim = tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.CORE_NUM)
    ub_size_bytes = tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.UB_SIZE)
    half_ub_bytes = ub_size_bytes // 2
    max_elements = half_ub_bytes // dtype_size

    if blockdim*max_elements < shape[0]:
        return max_elements - (max_elements % 32)

    num = (shape[0] + blockdim - 1) // blockdim
    if num % 32 != 0:
        num = num + (32 - (num % 32))
    if num > shape[0]:
        return shape[0]
    return num


def _get_batch_elements(updates, core_var_elements):
    """
    calculate how many indices or updates elements one batch can process at most
    """
    dtype = updates.get("dtype")
    shape = updates.get("shape")
    ub_size_bytes = tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.UB_SIZE)
    dtype_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
    left_ub_bytes = ub_size_bytes - core_var_elements*dtype_size
    num = left_ub_bytes // (dtype_size + 4)  # +4 for indices
    if num > shape[0]:
        return shape[0]
    return num


# pylint: disable=invalid-name, too-many-statements
def _scatter_nd_d_1d(indices, x, y, kernel_name="scatter_nd_d"):
    """
    do the scatter_nd logic of 1-d update situation
    """
    tik_instance = tik.Tik()
    dtype = x.get("dtype")
    indices_shape = indices.get("shape")
    updates_shape = x.get("shape")
    output_shape = y.get("shape")
    output_type = y.get("dtype")
    updates_gm = tik_instance.Tensor(dtype, updates_shape, name="updates",
                                     scope=tik.scope_gm)
    indices_gm = tik_instance.Tensor('int32', indices_shape, name="indices",
                                     scope=tik.scope_gm)
    output_gm = tik_instance.Tensor(output_type, output_shape, name="output",
                                    scope=tik.scope_gm)

    indices_var = tik_instance.Scalar(dtype='int32')
    core_start = tik_instance.Scalar(dtype='int32')
    core_end = tik_instance.Scalar(dtype='int32')
    cur_var = tik_instance.Scalar(dtype=output_type)
    cur_update = tik_instance.Scalar(dtype=dtype)
    acc_var = tik_instance.Scalar(dtype=output_type)

    indices_block_len = 8
    updates_block_len = 8
    var_block_len = 8
    if dtype in ('float16', 'int16'):
        updates_block_len = 16
        var_block_len = 16
    elif dtype in ('uint8', 'int8'):
        updates_block_len = 32
        var_block_len = 16

    var_len = output_shape[0]
    core_elements = _get_core_var_elements(y)
    blockdim = (var_len + core_elements - 1) // core_elements
    last_core_elements = var_len - (blockdim - 1)*core_elements
    var_blocks = (core_elements + var_block_len - 1) // var_block_len
    last_var_blocks = (last_core_elements + var_block_len - 1) // var_block_len

    indices_len = indices_shape[0]
    num_elements_batch = _get_batch_elements(x, core_elements)

    num_batch = (indices_len + num_elements_batch - 1) // num_elements_batch
    num_elements_last_batch = indices_len - (num_batch - 1)*num_elements_batch

    updates_ub = tik_instance.Tensor(dtype, (num_elements_batch,),
                                     name="updates_ub", scope=tik.scope_ubuf)
    indices_ub = tik_instance.Tensor('int32', (num_elements_batch, 1),
                                     name="indices_ub", scope=tik.scope_ubuf)
    var_ub = tik_instance.Tensor(output_type, (core_elements,), name="var_ub",
                                 scope=tik.scope_ubuf)

    def check_batch(num_elements, elements_offset):
        idx_blocks = (num_elements + indices_block_len - 1) // indices_block_len
        uds_blocks = (num_elements + updates_block_len - 1) // updates_block_len
        tik_instance.data_move(indices_ub, indices_gm[elements_offset], 0, 1,
                               idx_blocks, 0, 0)
        tik_instance.data_move(updates_ub, updates_gm[elements_offset], 0, 1,
                               uds_blocks, 0, 0)
        with tik_instance.for_range(0, num_elements) as k:
            indices_var.set_as(indices_ub[k])
            with tik_instance.if_scope(
                    tik.all(indices_var >= core_start, indices_var < core_end)):
                cur_var.set_as(var_ub[indices_var - core_start])
                cur_update.set_as(updates_ub[k])
                acc_var.set_as(cur_var + cur_update)
                var_ub[indices_var - core_start] = acc_var

    with tik_instance.for_range(0, blockdim, block_num=blockdim) as block_id:
        _zero_ub(tik_instance, var_ub, core_elements, dtype)
        core_start.set_as(block_id*core_elements)
        core_end.set_as((block_id + 1)*core_elements)
        with tik_instance.for_range(0, num_batch - 1) as i:
            check_batch(num_elements_batch, num_elements_batch*i)
        check_batch(num_elements_last_batch,
                    num_elements_batch*(num_batch - 1))
        with tik_instance.if_scope(block_id < blockdim - 1):
            tik_instance.data_move(output_gm[block_id*core_elements], var_ub,
                                   0, 1, var_blocks, 0, 0)
        with tik_instance.else_scope():
            tik_instance.data_move(output_gm[block_id*core_elements], var_ub,
                                   0, 1, last_var_blocks, 0, 0)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[indices_gm, updates_gm], outputs=[output_gm])
    return tik_instance
