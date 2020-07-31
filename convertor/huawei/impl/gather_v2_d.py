#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

gather_v2_d
"""
# pylint: disable=locally-disabled,too-many-lines
from functools import reduce as functools_reduce

from topi.cce import util
from te import tvm
from te import platform as cce
from te.platform.cce_build import build_config
from te.platform.fusion_manager import fusion_manager


# available soc resources
UB_SIZE = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE)
L1_SIZE = cce.cce_conf.get_soc_spec(cce.cce_conf.L1_SIZE)
CORE_NUM = cce.cce_conf.get_soc_spec(cce.cce_conf.CORE_NUM)
INDICES_LINE = 4096


def _new_alloc(tvm_ib, dtype, shape, name, scope):
    """ decl new buffer

    Parameters
    ----------
    tvm_ib: tvm.ir_builder
        Developer API of IR node builder make function.
    dtype: string
        buffer date type.
    shape: list of int
        buffer shape.
    name: string
        buffer name.
    scope: string
        buffer memory scope.

    Returns
    -------
    new_buffer : tvm.schedule.Buffer
        Symbolic data buffer.
    """
    buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = tvm.decl_buffer(shape,
                                 buf_var.dtype,
                                 name=name,
                                 scope=scope,
                                 data=buf_var)

    return new_buffer


def _get_target_core_num(tensor_params, tensor_indices):
    """ Get the device core numbers. for example, product = cloud, then
    target_core_num = 32, and then compute the greatest common number of actual
    device core numbers and indices size

    Parameters
    ----------
    tensor_params: TVM tensor.
        The tensor from which to gather values.
    tensor_indices: TVM Tensor
        The computation graph description of indices.

    Returns
    -------
    target_core_num: int
        The device core numbers.
    row_num_each_core: int
        Number of rows per core processing.
    """
    input_shape = tensor_indices.shape[:]

    target_core_num = CORE_NUM

    row_num_each_core = input_shape[0] // target_core_num
    remaining_row = input_shape[0] % target_core_num
    row_len = int(tensor_params.shape[2])
    dtype_size = cce.cce_intrin.get_bit_len(tensor_params.dtype) // 8

    if not _is_greater_than_32b(tensor_params, tensor_indices,
                                row_num_each_core):
        target_core_num = 1
        row_num_each_core = input_shape[0]
        return target_core_num, row_num_each_core, 0

    if row_len < 32:
        if int(row_num_each_core * row_len * dtype_size) <= 32:
            target_core_num = 1
            row_num_each_core = input_shape[0]
            return target_core_num, row_num_each_core, 0

    if int(input_shape[0]) < int(target_core_num):
        target_core_num = input_shape[0]
        row_num_each_core = 1
        return target_core_num, row_num_each_core, 0

    return target_core_num, row_num_each_core, remaining_row


def _get_burst_len(row_num, dtype):
    """ Get the burst length for generic DMA move .

    Parameters
    ----------
    row_num: int
        Size of the shape at a certain axis to compute.
    dtype: str
        data type.

    Returns
    -------
    burst_len: int
        Burst length for generic DMA move.
    """
    # Convert byts to Bytes
    dtype_size = cce.cce_intrin.get_bit_len(dtype) // 8
    # get the elements number each block can put in (32 Bytes in one block)
    elements_num_per_block = 32 // dtype_size
    # The basic block size is 32B
    burst_len = (row_num + elements_num_per_block -
                 1) // elements_num_per_block

    return burst_len


def _get_ub_available_size(indices_shape, dtype, tensor_params):
    """ Get the available UB size for output data, UB total size minus indices
    size.

    Parameters
    ----------
    indices_shape: list or tuple.
        shape of Tensor.
    dtype: str
        data type.

    Returns
    -------
    ub_available_size: int
        Available UB size based on Bytes.
    """
    ub_size = UB_SIZE - 1024 # reserve 1024 Bytes for temporary UB application
    indices_size = int(functools_reduce(lambda i, j: i * j, indices_shape))
    params_size = int(
        functools_reduce(lambda i, j: i * j, tensor_params.shape[:]))
    # Convert bytes to Bytes
    dtype_size = cce.cce_intrin.get_bit_len(dtype) // 8
    params_dtype_size = cce.cce_intrin.get_bit_len(tensor_params.dtype) // 8
    if indices_size * dtype_size < ub_size // 2:
        align_indices_size = (indices_size * dtype_size // 32 + 1) * 32
        ub_available_size = ub_size - align_indices_size
    else:
        ub_available_size = ub_size // 2

    is_params_ub = False
    params_ub_size = 0
    row_size = int(tensor_params.shape[2]) * (cce.cce_intrin.get_bit_len(tensor_params.dtype) // 8)
    if params_size * params_dtype_size < ub_size // 2 \
            and (_is_align(tensor_params) or int(row_size) < 32):
        is_params_ub = True
        params_ub_size = (params_size * params_dtype_size // 32 + 1) * 32

    return ub_available_size - params_ub_size, is_params_ub


def _params_tiling(tensor_params, tensor_indices, row_num_each_core):
    """ UB tiling: calculate the Maximum number of rows that UB can store at a
    time, and based on this, get the loop numbers the input data needed to moved
    bo UB. finally calculate the amount of data that was moved in the last
    iteration.

    Parameters
    ----------
    tensor_params: TVM tensor.
        The tensor from which to gather values.
    tensor_indices: TVM tensor
        Index tensor.
    row_num_each_core: int
        Number of rows per core processing.

    Returns
    -------
    loop_num: int
        The loop count required to store all the data.
    ub_row_num_once: int
        Maximum number of rows that UB can store at a time.
    last_loop_row_num: int
        Number of rows of the last loop.
    """
    # Convert byts to Bytes
    src_dtype_size = cce.cce_intrin.get_bit_len(tensor_params.dtype) // 8
    row_size = int(tensor_params.shape[2]) * src_dtype_size
    indices_shape_each_core = [
        row_num_each_core,
    ]
    ub_available_size, is_params_ub = _get_ub_available_size(
        indices_shape_each_core, tensor_indices.dtype, tensor_params)

    ub_row_num_once = ub_available_size // row_size
    split_num = 0
    if int(ub_row_num_once) != 0:
        loop_num = row_num_each_core // int(ub_row_num_once)
        last_loop_row_num = row_num_each_core % int(ub_row_num_once)
    else:
        loop_num = row_num_each_core
        ub_row_num_once = ub_available_size // src_dtype_size
        split_num = int(tensor_params.shape[2]) // ub_row_num_once
        last_loop_row_num = int(tensor_params.shape[2]) - ub_row_num_once * split_num

    return loop_num, split_num, ub_row_num_once, last_loop_row_num, is_params_ub


def _is_greater_than_32b(tensor_params, tensor_indices, row_num_each_core):
    """Check if the last DMA num is greater than 32 bytes."""

    loop_num, _, ub_row_num_once, last_loop_row_num, _ = _params_tiling(
        tensor_params, tensor_indices, row_num_each_core)
    row_size = int(tensor_params.shape[2]) * (cce.cce_intrin.get_bit_len(tensor_params.dtype) // 8)
    burst_len_ub = _get_burst_len(ub_row_num_once, tensor_params.dtype)
    burst_len_last_loop = _get_burst_len(last_loop_row_num,
                                         tensor_params.dtype)

    if int(row_size) < 32:
        return True

    if int(loop_num) == 0 and int(burst_len_ub) > 1:
        return True

    if int(burst_len_ub) != 0 or int(burst_len_last_loop) > 1:
        return True

    return False


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
# pylint: disable=locally-disabled,too-many-statements
def _gather_v2_operation(tensor_params, tensor_indices, output, tvm_ib,
                         op_parameters, is_last):
    """ Describe the calculation process for gather, steps as bellow,
        1.Get the value in indices and assign it to reg
        2.Get the corresponding row from params according to reg and move it
        to UB
        3.Move the data to gm

    Parameters
    ----------
    tensor_params: TVM tensor.
        The tensor from which to gather values.
    tensor_indices: TVM tensor
        Index tensor.
    output: list or tuple
        Shape of the output.
    tvm_ib: tvm.ir_builder
        Developer API of IR node builder make function.
    op_parameters: dictionary
        loop_row_num: Numbers of row to put into UB per loop
        burst_row_len: Burst length for per row of Params as a generic DMA move
        burst_loop_len: Burst length for per loop as a generic DMA move
        indices_ub: Buffer for tensor_indices
        data_ub: Buffer for output
        indices_offset: Offset fot the buffer for tensor_indices
        output_offset: Offset fot the buffer for output
        gm_offset: Offset fot the buffer for gm

    Returns
    -------
    None.
    """
    if tensor_indices.dtype.lower() == "int64":
        reg = tvm_ib.allocate("int32", (1,),
                              name='reg',
                              scope=cce.scope_reg)
        row_len = tvm.const(int(tensor_params.shape[2]), dtype="int32")
    else:
        reg = tvm_ib.allocate(tensor_indices.dtype, (1,),
                              name='reg',
                              scope=cce.scope_reg)
        row_len = int(tensor_params.shape[2])

    if op_parameters.get('is_params_ub') and _is_align(tensor_params):
        tvm_ib.emit(
            tvm.call_extern(
                tensor_params.dtype, "copy_gm_to_ubuf",
                op_parameters.get('params_ub').access_ptr("w", offset=0),
                tensor_params.access_ptr('r', offset=0), 0, 1,
                op_parameters.get('burst_params'), 0, 0))

    with tvm_ib.for_range(0, op_parameters.get('loop_row_num'),
                          name='row') as row:
        tvm_ib.emit(
            tvm.call_extern(
                tensor_indices.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                op_parameters.get('indices_ub').access_ptr(
                    'r', offset=op_parameters.get('indices_offset') + row)))
        gm_offset = (op_parameters.get('gm_offset') +
                     reg[0]) * row_len + op_parameters.get('inner_offset')
        if op_parameters.get('is_params_ub') and _is_align(tensor_params):
            tvm_ib.emit(
                tvm.call_extern(
                    tensor_params.dtype, "copy_ubuf_to_ubuf",
                    op_parameters.get('data_ub').access_ptr("w", offset=row * row_len),
                    op_parameters.get('params_ub').access_ptr('r', offset=gm_offset), 0, 1,
                    op_parameters.get('burst_row_len'), 0, 0))
        else:
            tvm_ib.emit(
                tvm.call_extern(
                    tensor_params.dtype, "copy_gm_to_ubuf",
                    op_parameters.get('data_ub').access_ptr("w", offset=row * row_len),
                    tensor_params.access_ptr('r', offset=gm_offset), 0, 1,
                    op_parameters.get('burst_row_len'), 0, 0))

    num_in_32b = 32 * 8 // cce.cce_intrin.get_bit_len(tensor_params.dtype)
    is_multi_core = op_parameters.get('is_multi_core')

    if _is_align(tensor_params) or (is_last is False) or (is_multi_core is
                                                          False):
        tvm_ib.emit(
            tvm.call_extern(
                output.dtype, "copy_ubuf_to_gm",
                output.access_ptr("w", offset=op_parameters.get('output_offset')),
                op_parameters.get('data_ub').access_ptr("r"), 0, 1,
                op_parameters.get('burst_loop_len'), 0, 0))
    else:
        # copy part of 32B align
        tvm_ib.emit(
            tvm.call_extern(
                output.dtype, "copy_ubuf_to_gm",
                output.access_ptr("w", offset=op_parameters.get('output_offset')),
                op_parameters.get('data_ub').access_ptr("r"), 0, 1,
                op_parameters.get('burst_loop_len') - 1, 0, 0))

        for j in range(num_in_32b):
            tvm_ib.emit(
                tvm.call_extern(
                    output.dtype, "reg_mov",
                    op_parameters.get('ub_for_32B').access_ptr("rw", offset=j),
                    op_parameters.get('data_ub').access_ptr(
                        "r",
                        offset=op_parameters.get('output_num') - num_in_32b + j)))

        tvm_ib.emit(
            tvm.call_extern(
                output.dtype, "copy_ubuf_to_gm",
                output.access_ptr("w",
                                  offset=op_parameters.get('output_offset') + op_parameters.get(
                                      'output_num') - num_in_32b),
                op_parameters.get('ub_for_32B').access_ptr("r"), 0, 1, 1, 0,
                0))


def _is_align(tensor_params):
    """ Determine whether the size of the input shape is aligned with 32B.

    Parameters
    ----------
    tensor_params: TVM tensor.
        The tensor from which to gather values.

    Returns
    -------
    True: aligned by 32B.
    False: can't aligned by 32B.
    """
    # Convert byts to Bytes
    dtype_size = cce.cce_intrin.get_bit_len(tensor_params.dtype) // 8
    row_size = int(tensor_params.shape[2] * dtype_size)

    return bool(row_size % 32 == 0)


def _small_dataset_operation(tensor_params, tensor_indices, output, tvm_ib,
                             op_parameters):
    """ Describe the calculation process for gather, steps as bellow,
        1.Move tensor_params by copy_gm_to_ubuf
        2.Get the value in indices and assign it to reg
        3.Get the corresponding row from params according to reg and move it by
        copy_ubuf_to_ubuf
        4.Move the data to gm
    """
    if tensor_indices.dtype.lower() == "int64":
        reg = tvm_ib.allocate("int32", (1,),
                              name='reg',
                              scope=cce.scope_reg)
        row_len = tvm.const(int(tensor_params.shape[2]), dtype="int32")
    else:
        reg = tvm_ib.allocate(tensor_indices.dtype, (1,),
                              name='reg',
                              scope=cce.scope_reg)
        row_len = int(tensor_params.shape[2])

    bit_size = cce.cce_intrin.get_bit_len(tensor_params.dtype)
    total_params = int(functools_reduce(lambda i, j: i * j, tensor_params.shape[:]))
    indices_len = int(functools_reduce(lambda i, j: i * j,
                                       tensor_indices.shape[:]))
    l1_size = int(L1_SIZE)
    if total_params * bit_size // 8 < l1_size and indices_len > INDICES_LINE:
        tvm_ib.emit(tvm.call_extern(tensor_params.dtype, "copy_gm_to_cbuf",
                                    op_parameters.get('params_cbuf').access_ptr("w", offset=0),
                                    tensor_params.access_ptr('r', offset=0),
                                    0, 1, op_parameters.get('burst_params'), 0, 0, 0))

    if op_parameters.get('is_params_ub'):
        tvm_ib.emit(
            tvm.call_extern(
                tensor_params.dtype, "copy_gm_to_ubuf",
                op_parameters.get('params_ub').access_ptr("w", offset=0),
                tensor_params.access_ptr('r', offset=0), 0, 1,
                op_parameters.get('burst_params'), 0, 0))
    with tvm_ib.for_range(0, op_parameters.get('loop_row_num'),
                          name='row') as row:
        tvm_ib.emit(
            tvm.call_extern(
                tensor_indices.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                op_parameters.get('indices_ub').access_ptr(
                    'r', offset=op_parameters.get('indices_offset') + row)))
        gm_offset = (op_parameters.get('gm_offset') + reg[0]) * row_len
        l1_offset = (gm_offset // (32 // (bit_size // 8))) * (32 // (bit_size // 8))
        ub_offset = gm_offset % (32 // (bit_size // 8))

        if not op_parameters.get('is_params_ub'):
            temp_ub = _new_alloc(tvm_ib,
                                 tensor_params.dtype, [1, row_len + 32],
                                 "temp_ub",
                                 scope=cce.scope_ubuf)

            if total_params * bit_size // 8 < l1_size and \
                    indices_len > INDICES_LINE:
                tvm_ib.emit(tvm.call_extern(tensor_params.dtype, "copy_cbuf_to_ubuf",
                                            temp_ub.access_ptr("w"),
                                            op_parameters.get('params_cbuf').
                                            access_ptr('r', offset=l1_offset), 0, 1,
                                            op_parameters.get('burst_row_len') + 1, 0, 0))
            else:
                tvm_ib.emit(tvm.call_extern(tensor_params.dtype, "copy_gm_to_ubuf",
                                            temp_ub.access_ptr("w"),
                                            tensor_params.access_ptr('r', offset=gm_offset), 0, 1,
                                            op_parameters.get('burst_row_len'), 0, 0))

        with tvm_ib.for_range(0, row_len, name='row_num') as row_num:
            reg_offset = (op_parameters['gm_offset'] +
                          reg[0]) * row_len + row_num
            data_offset = row * row_len + row_num
            if op_parameters.get('is_params_ub'):
                tvm_ib.emit(
                    tvm.call_extern(
                        tensor_params.dtype, "reg_mov",
                        op_parameters.get('data_ub').access_ptr(
                            "w", offset=data_offset),
                        op_parameters.get('params_ub').access_ptr(
                            'r', offset=reg_offset)))
            elif total_params * bit_size // 8 < l1_size and \
                    indices_len > INDICES_LINE:
                tvm_ib.emit(
                    tvm.call_extern(
                        tensor_params.dtype, "reg_mov",
                        op_parameters.get('data_ub').access_ptr(
                            "w", offset=data_offset),
                        temp_ub.access_ptr('r', offset=(ub_offset + row_num))))
            else:
                tvm_ib.emit(
                    tvm.call_extern(
                        tensor_params.dtype, "reg_mov",
                        op_parameters.get('data_ub').access_ptr(
                            "w", offset=data_offset),
                        temp_ub.access_ptr('r', offset=row_num)))

    if int(op_parameters.get('loop_row_num') * row_len * bit_size // 8) % 32 != 0 \
            and op_parameters.get('is_multi_core'):
        tvm_ib.emit(
            tvm.call_extern(
                output.dtype, "copy_ubuf_to_gm",
                output.access_ptr("w",
                                  offset=op_parameters.get('output_offset')),
                op_parameters.get('data_ub').access_ptr("r", offset=0), 0, 1,
                op_parameters.get('burst_loop_len') - 1, 0, 0))
        num_in_32b = 32 * 8 // bit_size
        for j in range(num_in_32b):
            tvm_ib.emit(
                tvm.call_extern(
                    output.dtype, "reg_mov",
                    op_parameters.get('ub_for_32B').access_ptr("rw", offset=j),
                    op_parameters.get('data_ub').access_ptr(
                        "r",
                        offset=op_parameters.get('output_num') - num_in_32b + j)))
        tvm_ib.emit(
            tvm.call_extern(
                output.dtype, "copy_ubuf_to_gm",
                output.access_ptr("w", offset=op_parameters.get('output_offset') + \
                                              op_parameters.get('output_num') - num_in_32b),
                op_parameters.get('ub_for_32B').access_ptr("r"), 0, 1, 1, 0, 0))
    else:
        tvm_ib.emit(
            tvm.call_extern(
                output.dtype, "copy_ubuf_to_gm",
                output.access_ptr("w",
                                  offset=op_parameters.get('output_offset')),
                op_parameters.get('data_ub').access_ptr("r", offset=0), 0, 1,
                op_parameters.get('burst_loop_len'), 0, 0))


def _gather_v2_loop(output, tensor_params, tensor_indices, op_parameters,
                    tvm_ib, row_num_each_core, loop_offset, row_num_once,
                    block_index, pre_num):
    """Describe gather process in different segmentation scenario."""
    row_len = int(tensor_params.shape[2])
    total_row = int(
        functools_reduce(lambda i, j: i * j, tensor_indices.shape[:]))
    total_params = int(
        functools_reduce(lambda i, j: i * j, tensor_params.shape[:]))
    op_parameters['burst_row_len'] = _get_burst_len(row_len,
                                                    tensor_params.dtype)
    op_parameters['burst_loop_len'] = _get_burst_len(row_len,
                                                     tensor_params.dtype)
    op_parameters['output_num'] = row_len
    op_parameters['burst_params'] = _get_burst_len(total_params,
                                                   tensor_params.dtype)

    loop_num, split_num, ub_row_num_once, last_loop_row_num, is_params_ub = \
        _params_tiling(tensor_params, tensor_indices, row_num_once)
    row_size = int(tensor_params.shape[2]) * (cce.cce_intrin.get_bit_len(tensor_params.dtype) // 8)

    if row_size < 32:
        if is_params_ub:
            op_parameters['is_params_ub'] = is_params_ub
            op_parameters['params_ub'] = _new_alloc(tvm_ib,
                                                    tensor_params.dtype,
                                                    [1, total_params],
                                                    "params_ub",
                                                    scope=cce.scope_ubuf)
        src_shape_each_core = [row_num_once, row_len]
        op_parameters['loop_row_num'] = row_num_once
        op_parameters['data_ub'] = _new_alloc(tvm_ib,
                                              tensor_params.dtype,
                                              src_shape_each_core,
                                              "data_ub",
                                              scope=cce.scope_ubuf)

        op_parameters['gm_offset'] = pre_num * int(tensor_params.shape[1])
        op_parameters['indices_offset'] = 0
        op_parameters['output_offset'] = (pre_num * total_row +
                                          block_index * row_num_each_core +
                                          loop_offset) * row_len
        op_parameters['burst_loop_len'] = _get_burst_len(
            row_num_once * row_len, tensor_params.dtype)
        op_parameters['output_num'] = row_num_once * row_len
        _small_dataset_operation(tensor_params, tensor_indices, output, tvm_ib,
                                 op_parameters)
        return

    if is_params_ub and _is_align(tensor_params):
        op_parameters['is_params_ub'] = is_params_ub
        op_parameters['params_ub'] = _new_alloc(tvm_ib,
                                                tensor_params.dtype,
                                                [1, total_params],
                                                "params_ub",
                                                scope=cce.scope_ubuf)

    # copy into UB one-row at a time, and copy out to GM after processing
    if loop_num != row_num_once or split_num == 0:
        if not _is_align(tensor_params):
            src_shape_each_core = [1, row_len]
            op_parameters['data_ub'] = _new_alloc(tvm_ib,
                                                  tensor_params.dtype,
                                                  src_shape_each_core,
                                                  "data_ub",
                                                  scope=cce.scope_ubuf)
            with tvm_ib.for_range(0, row_num_once,
                                  name='gather_num') as gather_num:
                op_parameters['gm_offset'] = pre_num * int(tensor_params.shape[1])
                op_parameters['indices_offset'] = gather_num
                op_parameters['output_offset'] = (pre_num * total_row +
                                                  block_index * row_num_each_core +
                                                  loop_offset + gather_num) * row_len
                with tvm_ib.if_scope(gather_num != row_num_once - 1):
                    _gather_v2_operation(tensor_params,
                                         tensor_indices,
                                         output,
                                         tvm_ib,
                                         op_parameters,
                                         is_last=False)
                with tvm_ib.else_scope():
                    _gather_v2_operation(tensor_params,
                                         tensor_indices,
                                         output,
                                         tvm_ib,
                                         op_parameters,
                                         is_last=True)
        # copy into UB multi-rows at a time, and copy out to GM together after
        # processing
        else:
            # Processing previous batches(loop_num) of data
            with tvm_ib.if_scope(loop_num > 0):
                op_parameters['loop_row_num'] = ub_row_num_once
                src_shape_each_core = [ub_row_num_once, row_len]
                op_parameters['data_ub'] = _new_alloc(tvm_ib,
                                                      tensor_params.dtype,
                                                      src_shape_each_core,
                                                      "data_ub",
                                                      scope=cce.scope_ubuf)
                op_parameters['burst_loop_len'] = _get_burst_len(
                    ub_row_num_once * row_len, tensor_params.dtype)
                op_parameters['output_num'] = ub_row_num_once * row_len

                with tvm_ib.for_range(0, loop_num,
                                      name='gather_num') as gather_num:
                    op_parameters[
                        'gm_offset'] = pre_num * int(tensor_params.shape[1])
                    op_parameters[
                        'indices_offset'] = gather_num * ub_row_num_once
                    op_parameters['output_offset'] = (pre_num * total_row +
                                                      block_index * row_num_each_core +
                                                      loop_offset +
                                                      gather_num * ub_row_num_once) * row_len
                    _gather_v2_operation(tensor_params,
                                         tensor_indices,
                                         output,
                                         tvm_ib,
                                         op_parameters,
                                         is_last=False)
            # Processing the last batch of data
            with tvm_ib.if_scope(last_loop_row_num > 0):
                src_shape_each_core = [last_loop_row_num, row_len]
                op_parameters['loop_row_num'] = last_loop_row_num
                op_parameters['data_ub'] = _new_alloc(tvm_ib,
                                                      tensor_params.dtype,
                                                      src_shape_each_core,
                                                      "data_ub",
                                                      scope=cce.scope_ubuf)
                op_parameters['burst_loop_len'] = _get_burst_len(
                    last_loop_row_num * row_len, tensor_params.dtype)
                op_parameters['output_num'] = last_loop_row_num * row_len
                op_parameters['indices_offset'] = loop_num * ub_row_num_once
                op_parameters['output_offset'] = (pre_num * total_row +
                                                  block_index * row_num_each_core +
                                                  loop_offset +
                                                  loop_num * ub_row_num_once) * row_len
                op_parameters['gm_offset'] = pre_num * int(tensor_params.shape[1])
                _gather_v2_operation(tensor_params,
                                     tensor_indices,
                                     output,
                                     tvm_ib,
                                     op_parameters,
                                     is_last=False)
    else:
        src_shape_each_core = [1, ub_row_num_once]
        op_parameters['data_ub'] = _new_alloc(tvm_ib,
                                              tensor_params.dtype,
                                              src_shape_each_core,
                                              "data_ub",
                                              scope=cce.scope_ubuf)
        with tvm_ib.for_range(0, row_num_once,
                              name='gather_num') as gather_num:
            op_parameters['burst_row_len'] = _get_burst_len(
                ub_row_num_once, tensor_params.dtype)
            op_parameters['burst_loop_len'] = _get_burst_len(
                ub_row_num_once, tensor_params.dtype)
            op_parameters['output_num'] = ub_row_num_once
            with tvm_ib.for_range(0, split_num, name='inner_num') as inner_num:
                op_parameters['gm_offset'] = pre_num * int(tensor_params.shape[1])
                op_parameters['inner_offset'] = inner_num * ub_row_num_once
                op_parameters['indices_offset'] = gather_num
                op_parameters['output_offset'] = (pre_num * total_row + \
                                                  block_index * \
                                                  row_num_each_core + \
                                                  loop_offset + gather_num) * \
                                                 row_len + inner_num * \
                                                 ub_row_num_once
                if inner_num != split_num or last_loop_row_num != 0:
                    _gather_v2_operation(tensor_params,
                                         tensor_indices,
                                         output,
                                         tvm_ib,
                                         op_parameters,
                                         is_last=False)
                else:
                    _gather_v2_operation(tensor_params,
                                         tensor_indices,
                                         output,
                                         tvm_ib,
                                         op_parameters,
                                         is_last=True)
            with tvm_ib.if_scope(last_loop_row_num > 0):
                op_parameters['burst_row_len'] = _get_burst_len(
                    last_loop_row_num, tensor_params.dtype)
                op_parameters['burst_loop_len'] = _get_burst_len(
                    last_loop_row_num, tensor_params.dtype)
                op_parameters['output_num'] = last_loop_row_num
                op_parameters['gm_offset'] = pre_num * int(tensor_params.shape[1])
                op_parameters['inner_offset'] = split_num * ub_row_num_once
                op_parameters['indices_offset'] = gather_num
                op_parameters['output_offset'] = (pre_num * total_row + \
                                                  block_index * \
                                                  row_num_each_core + \
                                                  loop_offset + gather_num) * \
                                                 row_len + split_num * \
                                                 ub_row_num_once
                _gather_v2_operation(tensor_params,
                                     tensor_indices,
                                     output,
                                     tvm_ib,
                                     op_parameters,
                                     is_last=True)


def _indices_tiling(tensor_params, tensor_indices, row_num_each_core):
    """Segmentation logic for indices"""
    params_dtype_size = cce.cce_intrin.get_bit_len(tensor_params.dtype) // 8
    row_size = int(tensor_params.shape[2]) * params_dtype_size
    params_size = int(
        functools_reduce(lambda i, j: i * j, tensor_params.shape[:]))

    indices_dtype_size = cce.cce_intrin.get_bit_len(tensor_indices.dtype) // 8
    indices_size = row_num_each_core * indices_dtype_size
    ub_size = UB_SIZE - 1024

    if row_size < 32:
        if params_size * params_dtype_size < ub_size // 2:
            ub_size = ub_size // 2
        indices_size = row_num_each_core * (row_size + indices_dtype_size)
        loop_num = indices_size // ub_size
    else:
        ub_size = ub_size // 2
        loop_num = indices_size // ub_size
        row_size = 0

    if int(loop_num) != 0:
        row_num_once = ub_size // (row_size + indices_dtype_size)
        last_loop_row_num = row_num_each_core - loop_num * row_num_once
    else:
        row_num_once = row_num_each_core
        last_loop_row_num = row_num_each_core

    return loop_num, row_num_once, last_loop_row_num


def _kernel_ir(output, tensor_params, tensor_indices):
    """ IR node builder make function

    Parameters
    ----------
    output: list or tuple
        Shape of the output.
    tensor_params: TVM tensor.
        The tensor from which to gather values.
    tensor_indices: TVM tensor
        Index tensor.

    Returns
    -------
    stmt, The result statement.
    """
    tvm_ib = tvm.ir_builder.create()
    # block tiling,get the appropriate core numbers for thread extent.
    target_core_num, row_num_each_core, remaining_row = _get_target_core_num(
        tensor_params, tensor_indices)
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", target_core_num)

    # Define a dictionary to save the parameters needed by _gather_v2_operation()
    # loop_row_num: Numbers of row to put into UB per loop
    # burst_row_len: Burst length for per row of Params as a generic DMA move
    # burst_loop_len: Burst length for per loop as a generic DMA move
    # indices_ub: Buffer for tensor_indices
    # params_ub: Buffer for tensor_params
    # data_ub: Buffer for output
    # params_cbuf: Buffer for L1
    # indices_offset: Offset for the buffer for tensor_indices
    # output_offset: Offset for the buffer for output
    # gm_offset: Offset for the buffer for gm
    # inner_offset: inner offset for the buffer for gm
    # remaining_offset: remaining offset after multi core split
    # tail_num: remaining offset for alignment
    # is_multi_core: if running in multiple cores
    # is_params_ub: if using copy_ub_to_ub
    op_parameters = {
        'loop_row_num': 1,
        'burst_row_len': 1,
        'burst_loop_len': 1,
        'indices_ub': 0,
        'params_ub': 0,
        'data_ub': 0,
        'params_cbuf': 0,
        'indices_offset': 0,
        'output_offset': 0,
        'gm_offset': 0,
        'inner_offset': 0,
        'remaining_offset': 0,
        'output_num': 0,
        'is_multi_core': False,
        'is_params_ub': False
    }

    if int(target_core_num) > 1:
        op_parameters['is_multi_core'] = True
    loop_num, row_num_once, last_loop_row_num = _indices_tiling(
        tensor_params, tensor_indices, row_num_each_core)
    pre_axis_num = int(tensor_params.shape[0])
    op_parameters['ub_for_32B'] = _new_alloc(
        tvm_ib,
        tensor_params.dtype,
        32 * 8 // cce.cce_intrin.get_bit_len(tensor_params.dtype),
        "ub_for_32B",
        scope=cce.scope_ubuf)

    total_params = int(functools_reduce(lambda i, j: i * j, tensor_params.shape[:]))
    l1_size = int(L1_SIZE)
    row_size = int(tensor_params.shape[2]) * \
               (cce.cce_intrin.get_bit_len(tensor_params.dtype) // 8)
    bit_size = cce.cce_intrin.get_bit_len(tensor_params.dtype)
    indices_len = int(functools_reduce(lambda i, j: i * j,
                                       tensor_indices.shape[:]))
    if total_params * bit_size // 8 < l1_size and row_size < 32 and \
            indices_len > INDICES_LINE:
        op_parameters['params_cbuf'] = _new_alloc(tvm_ib, tensor_params.dtype,
                                                  [1, total_params],
                                                  "params_cbuf", scope=cce.scope_cbuf)

    with tvm_ib.for_range(0, pre_axis_num, name='pre_axis_num') as pre_num:
        # Processing previous batches(loop_num) of data
        with tvm_ib.if_scope(loop_num > 0):
            # Apply for UB space to put into indices
            indices_shape_each_core = [row_num_once, ]
            op_parameters['burst_row_len'] = _get_burst_len(
                row_num_once, tensor_indices.dtype)
            op_parameters['indices_ub'] = _new_alloc(tvm_ib,
                                                     tensor_indices.dtype,
                                                     indices_shape_each_core,
                                                     "indices_ub",
                                                     scope=cce.scope_ubuf)
            with tvm_ib.for_range(0, loop_num, name='num') as num:
                op_parameters['indices_offset'] = \
                    (block_index * row_num_each_core + num * row_num_once)
                # copy indices from gm to ub
                tvm_ib.emit(
                    tvm.call_extern(
                        tensor_indices.dtype, "copy_gm_to_ubuf",
                        op_parameters.get('indices_ub').access_ptr("w"),
                        tensor_indices.access_ptr(
                            'r', offset=op_parameters.get('indices_offset')),
                        0, 1, op_parameters.get('burst_row_len'), 0, 0))

                _gather_v2_loop(output, tensor_params, tensor_indices,
                                op_parameters, tvm_ib, row_num_each_core,
                                num * row_num_once, row_num_once, block_index,
                                pre_num)

        row_len = int(tensor_params.shape[2])
        align_offset = 0
        params_bit_size = cce.cce_intrin.get_bit_len(tensor_params.dtype) // 8
        with tvm_ib.if_scope(last_loop_row_num > 0):
            if row_len < 32 and params_bit_size * row_len * int(
                    last_loop_row_num) < 32 and int(target_core_num) > 1:
                last_loop_row_num += 32
                align_offset = 32
            indices_shape_each_core = [last_loop_row_num, ]
            op_parameters['burst_row_len'] = _get_burst_len(
                last_loop_row_num, tensor_indices.dtype)
            op_parameters['indices_ub'] = _new_alloc(tvm_ib,
                                                     tensor_indices.dtype,
                                                     indices_shape_each_core,
                                                     "indices_ub",
                                                     scope=cce.scope_ubuf)
            op_parameters[
                'indices_offset'] = block_index * row_num_each_core + \
                                    loop_num * row_num_once - align_offset

            # copy indices from gm to ub
            tvm_ib.emit(
                tvm.call_extern(
                    tensor_indices.dtype, "copy_gm_to_ubuf",
                    op_parameters.get('indices_ub').access_ptr("w"),
                    tensor_indices.access_ptr(
                        'r', offset=op_parameters.get('indices_offset')), 0, 1,
                    op_parameters.get('burst_row_len'), 0, 0))
            _gather_v2_loop(output, tensor_params, tensor_indices,
                            op_parameters, tvm_ib, row_num_each_core,
                            loop_num * row_num_once - align_offset,
                            last_loop_row_num, block_index, pre_num)
        align_offset = 0
        # Processing the tail data
        with tvm_ib.if_scope(remaining_row > 0):
            with tvm_ib.if_scope(block_index < 1):
                if row_len < 32 and params_bit_size * row_len * int(
                        remaining_row) < 32 and int(target_core_num) > 1:
                    remaining_row += 32
                    align_offset = 32
                indices_shape_each_core = [remaining_row, ]
                op_parameters['burst_row_len'] = _get_burst_len(
                    remaining_row, tensor_indices.dtype)
                op_parameters['indices_ub'] = _new_alloc(
                    tvm_ib,
                    tensor_indices.dtype,
                    indices_shape_each_core,
                    "indices_ub",
                    scope=cce.scope_ubuf)
                op_parameters['indices_offset'] = row_num_each_core * \
                                                  target_core_num - align_offset
                # copy indices from gm to ub
                tvm_ib.emit(
                    tvm.call_extern(
                        tensor_indices.dtype, "copy_gm_to_ubuf",
                        op_parameters.get('indices_ub').access_ptr("w"),
                        tensor_indices.access_ptr(
                            'r', offset=op_parameters.get('indices_offset')),
                        0, 1, op_parameters.get('burst_row_len'), 0, 0))
                _gather_v2_loop(
                    output, tensor_params, tensor_indices, op_parameters,
                    tvm_ib, row_num_each_core,
                    row_num_each_core * target_core_num - align_offset,
                    remaining_row, 0, pre_num)

    return tvm_ib.get()


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("gather_v2_d")
def gather_v2_compute(tensor_params,
                      tensor_indices,
                      params_shape,
                      indices_shape,
                      params_dtype,
                      indices_dtype,
                      axis=0,
                      kernel_name="cce_gather_v2",
                      need_build=False,
                      need_print=False):
    """ TVM calculation process, used for fusion operation

    Parameters
    ----------
    tensor_params: TVM tensor.
        The tensor from which to gather values.
    tensor_indices: TVM tensor
        Index tensor.
    params_shape: list or tuple.
        Shape of params.
    indices_shape: list or tuple.
        Shape of indices.
    params_dtype: str
        type of the params.Must be one of the following types: "int32", "int8",
        "uint8", "float16", "float32".
    indices_dtype: str
        type of the indices. Must be "int32"
    axis: int
        The axis in `params`. Must be 0.
    kernel_name : str
        cce kernel name, default value is "cce_gather_v2"
    need_build : bool
        if need to build CCEC kernel, default value is False
    need_print : bool
        if need to print the ir, default value is False

    Returns
    -------
    res: TVM tensor.
        The tensor created by gather compute .
    """
    out_shape = [
        tensor_params.shape[0], tensor_indices.shape[0], tensor_params.shape[2]
    ]

    res = tvm.extern([out_shape], [tensor_params, tensor_indices],
                     lambda ins, outs: _kernel_ir(outs[0], ins[0], ins[1]),
                     name="res",
                     dtype=params_dtype)

    return res


# pylint: disable=locally-disabled,invalid-name
@util.check_input_type(dict, dict, dict, int, str)
def gather_v2_d(x, indices, y, axis=0, kernel_name="gather_v2_d"):
    """ Gather slices from `params` axis `axis` according to `indices`.
    Produces an output tensor with shape `params.shape[:axis] + indices.shape +
    params.shape[axis + 1:].

    Parameters
    ----------
    x : dict
        shape and dtype of params.
    indices : dict
        shape and dtype of indices.
    y: dict
        shape and dtype of output.
    axis: int
        The axis in params.
    kernel_name: str
        cce kernel name, default value is "gather_v2_d"

    Returns
    -------
    None.
    """
    params_shape = x.get("shape")
    indices_shape = indices.get("shape")
    params_dtype = x.get("dtype").lower()
    indices_dtype = indices.get("dtype").lower()

    dim_num = len(params_shape)
    if axis < -dim_num or axis >= dim_num:
        raise RuntimeError("Axis value out of range")

    dtype_list = ("int8", "int16", "int32", "int64", "uint8", "uint16",
                  "uint32", "uint64", "float16", "float32")
    util.check_dtype_rule(params_dtype, dtype_list)
    util.check_dtype_rule(indices_dtype, ("int32", "int64"))
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(params_shape)
    util.check_shape_rule(indices_shape)
    util.check_tensor_shape_size(params_shape)
    util.check_tensor_shape_size(indices_shape)

    axis = util.axis_check(len(params_shape), axis)

    # Reshape the indices_shape to 1D and reshape the params_shape to 3D in
    # order to simplify
    # the calculation, due to only 0 is supported for axis currently.
    indices_reshape = [
        int(functools_reduce(lambda i, j: i * j, indices_shape)),
    ]
    pre_axis_shape = params_shape[:axis]
    after_axis_shape = params_shape[axis + 1:]
    if not pre_axis_shape:
        pre_axis_shape = [1, ]
    if not after_axis_shape:
        after_axis_shape = [1, ]
    params_reshape = [
        int(functools_reduce(lambda i, j: i * j, pre_axis_shape)),
        params_shape[axis],
        int(functools_reduce(lambda i, j: i * j, after_axis_shape))
    ]

    output_shape = [
        params_reshape[0],
        indices_reshape[0],
        params_reshape[2]
    ]
    util.check_tensor_shape_size(output_shape)

    tensor_params = tvm.placeholder(params_reshape,
                                    dtype=params_dtype,
                                    name="tensor_params")
    tensor_indices = tvm.placeholder(indices_reshape,
                                     dtype=indices_dtype,
                                     name="tensor_indices")

    res = gather_v2_compute(tensor_params,
                            tensor_indices,
                            params_reshape,
                            indices_reshape,
                            params_dtype,
                            indices_dtype,
                            axis,
                            kernel_name="cce_gather_v2",
                            need_build=True,
                            need_print=False)

    sch = tvm.create_schedule(res.op)

    with build_config:
        tvm.build(sch, [tensor_params, tensor_indices, res],
                  "cce",
                  name=kernel_name)
