#!/usr/bin/env python
# -*- coding:utf-8 -*-
# pylint: disable=too-many-lines
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

one_hot_d
"""
# pylint: disable=too-many-lines
from __future__ import print_function
from __future__ import absolute_import
from functools import reduce as functools_reduce

from te import tvm
from te import platform as cce
from te.platform.cce_build import build_config
from topi.cce import util
from impl.transpose_d import _tranpose_notchange_last
from impl.transpose_d import _write_code

# 8k UB buffer is a reserved space
UB_SPACE_8K = 8 * 1024
# 2 block UB buffer is for on_value and off_value
ON_OFF_VAL_SPACE = 32 * 2


def _add_workspace(indices_reshape, depth, dtype, kernel_name):
    """ add workspace accroding to indices_reshape, depth and dtype.

    Parameters
    ----------
    indices_reshape: tuple.
        Shape of indices array after reshape.
    depth: int.
        A scalar defining the depth of the one hot dimension.
    dtype: str.
        The data type of output tensor, and only support float16, float32, int32.
    kernel_name: str.
        Cce kernel name.

    Returns
    -------
    None.
    """
    workspace_dict = {
        "workspace": {
            "num":
            1,
            "size": [
                indices_reshape[0] * depth *
                cce.cce_intrin.get_bit_len(dtype) // 8
            ]
        }
    }
    _write_code(workspace_dict, "kernel_meta/" + kernel_name + ".json")


def _get_perm(shape, depth, axis):
    """ Get the output array shape and perm of it.

    Parameters
    ----------
    shape: list or tuple.
        Shape of indices array.
    depth: int.
        A scalar defining the depth of the one hot dimension.
    axis: int.
        The axis to fill, support any axis.

    Returns
    -------
    output_shape: tuple.
        Shape of output array.
    perm: list.
        permutation of the output array.
    """
    perm = list(range(len(shape)))
    perm.insert(axis, len(shape))
    output_shape = shape[:axis] + (depth, ) + shape[axis:] + (1, )
    return output_shape, perm + [
        len(shape) + 1,
    ]


# pylint: disable=too-many-arguments
def _transpose(shape, dtype, depth, axis, data, indices_input, on_value_input,
               off_value_input):
    """ transpose to right axis after doing one hot.

    Parameters
    ----------
    shape: list or tuple.
        Shape of indices array.
    dtype: str.
        The data type of output tensor, and only support float16, float32, int32.
    depth: int.
        A scalar defining the depth of the one hot dimension.
    axis: int.
        The axis to fill, support any axis.
    data: TVM tensor.
        Output tensor after one hot.
    indices_input: TVM tensor.
        Input indices tensor.

    Returns
    -------
    sch: schedule.Schedule.
        The created schedule.
    tensor_list: list
        The tensor list to be built.
    """
    output_shape, perm = _get_perm(shape, depth, axis)
    sch, output = _tranpose_notchange_last(data, output_shape, perm, dtype)
    tensor_list = [
        indices_input, on_value_input, off_value_input, output[1], data
    ]
    return sch, tensor_list


def _new_alloc(tvm_ib, dtype, shape, name, scope):
    """decl new buffer.

    Parameters
    ----------
    tvm_ib : tvm.ir_builder.
        Developer API of IR node builder make function.
    dtype : str.
        Buffer date type.
    shape : list of int.
        Buffer shape.
    name : str.
        Buffer name.
    scope : str.
        Buffer memory scope.

    Returns
    -------
    buffer : tvm.schedule.Buffer.
        Symbolic data buffer.
    """
    buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = tvm.decl_buffer(shape,
                                 buf_var.dtype,
                                 name=name,
                                 scope=scope,
                                 data=buf_var)

    return new_buffer


def _get_dup_repeat_num(output_shape, output_dtype):
    """Get the num of repeat times for vector_dup operation.

    Parameters
    ----------
    output_shape: list or tuple.
        The shape that will be dup to off_value.
    output_dtype: str.
        The data type of off_value.

    Returns
    -------
    dup_repeat_num: int.
        The repeat num for vector_dup.
    each_repeat_num: int.
        The data num in one repeat.
    """
    if output_dtype in ("float32", "int32"):
        each_repeat_num = 64
    else:
        each_repeat_num = 128

    output_size = functools_reduce(lambda i, j: i * j, output_shape)
    dup_repeat_num = ((output_size - 1) // each_repeat_num) + 1

    return int(dup_repeat_num), each_repeat_num


def _check_need_tiling(indices_shape, indice_dtype, output_shape, output_dtype,
                       dtype_dict):
    """Check whether tiling operation is needed.
    """
    ub_info = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE) - 32 - \
              UB_SPACE_8K - ON_OFF_VAL_SPACE
    total_indices_size = int(
        functools_reduce(lambda i, j: i * j, indices_shape))
    total_output_size = int(functools_reduce(lambda i, j: i * j, output_shape))

    # if output_dtype is int8 or uint8, need use float16 and then cast to real
    # dtype
    if output_dtype in ("int8", "uint8"):
        is_cast = 1
        output_dtype = "float16"
    else:
        is_cast = 0

    # check if the tiling is needed
    # calculate the total input size and cast size
    ub_space_input = total_indices_size * dtype_dict.get(indice_dtype)
    ub_space_cast = is_cast * total_output_size * dtype_dict.get('int8')
    # check if the input space and cast space is 32 bytes aligned
    if (ub_space_input % 32) != 0:
        ub_space_input = ((ub_space_input // 32) + 1) * 32
    if (ub_space_cast % 32) != 0:
        ub_space_cast = ((ub_space_cast // 32) + 1) * 32

    # compute max_mum for the input data type of float16 or float32 in UB
    num_remaining = (ub_info - ub_space_input -
                     ub_space_cast) // dtype_dict.get(output_dtype)

    if num_remaining >= total_output_size:
        return False, is_cast
    return True, is_cast


def _do_tiling(is_tiling_dict, num_index, output_shape):
    """Calculate the maximum number of rows that UB can store at one time,
    and based on this, get the loop numbers and the row numbers in the last
    iteration.
    """
    ub_row_num_once = num_index
    ub_loop = output_shape[0] // ub_row_num_once
    last_loop_row_num = output_shape[0] - ub_loop * ub_row_num_once
    if int(last_loop_row_num) != 0:
        ub_loop = ub_loop + 1
    else:
        last_loop_row_num = ub_row_num_once

    is_tiling_dict['is_tiling'] = True
    is_tiling_dict['ub_row_num_once'] = int(ub_row_num_once)
    is_tiling_dict['ub_loop'] = int(ub_loop)
    is_tiling_dict['last_loop_row_num'] = int(last_loop_row_num)

    return is_tiling_dict


def _is_need_tiling(indices_shape, indice_dtype, output_shape, output_dtype):
    """Check whether tiling operation is needed.
    Calculate the maximum number of rows that UB can store at one time,
    and based on this, get the loop numbers and the row numbers in the last
    iteration.

    Parameters
    ----------
    indices_shape: list or tuple.
        The total indices shape that one core will handle.
    indice_dtype: str.
        The data type of input tensor.
    output_shape: list or tuple.
        The shape that will be copied from UB to gm.
    output_dtype: str.
        The data type of output tensor.

    Returns
    -------
    is_tiling: bool.
        If is_tiling is True, it means tiling is needed.
    ub_row_num_once: int.
        The row num will be dealed with in each loop.
    ub_loop: int.
        The num of loop in tiling.
    last_loop_row_num: int.
        The row num will be dealed with in the last loop.
    """
    dtype_dict = {}
    # the following numbers indicate the number of bytes of the data type
    dtype_dict['float16'] = 2
    dtype_dict['float32'] = 4
    dtype_dict['int32'] = 4
    dtype_dict['int8'] = 1
    dtype_dict['uint8'] = 1

    ub_info = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE) - 32 - \
              UB_SPACE_8K - ON_OFF_VAL_SPACE
    need_tiling, is_cast = _check_need_tiling(indices_shape, indice_dtype,
                                              output_shape, output_dtype,
                                              dtype_dict)

    is_tiling_dict = {}
    if not need_tiling:
        # tiling is not needed
        is_tiling_dict['is_tiling'] = False
        is_tiling_dict['ub_row_num_once'] = None
        is_tiling_dict['ub_loop'] = None
        is_tiling_dict['last_loop_row_num'] = None
        return is_tiling_dict

    if is_cast:
        output_dtype = "float16"

    # tiling is needed, and calculate the maximum number of rows that UB can
    # store at a time
    # the space of indices, the space of output and the space of cast is less
    # than UB
    ub_row_num_once = ub_info // output_shape[1]
    # check if the remaining space can store the index array and cast array
    for i in range(ub_row_num_once):
        num_index = ub_row_num_once - i
        byte_space_output = num_index * output_shape[1] * dtype_dict.get(
            output_dtype)
        byte_space_cast = is_cast * num_index * output_shape[
            1] * dtype_dict.get('int8')
        # check if the input space and cast space is 32 bytes aligned
        if (byte_space_output % 32) != 0:
            byte_space_output = ((byte_space_output // 32) + 1) * 32
        if (byte_space_cast % 32) != 0:
            byte_space_cast = ((byte_space_cast // 32) + 1) * 32

        num_remaining = (ub_info - byte_space_output -
                         byte_space_cast) // dtype_dict.get(indice_dtype)
        if num_remaining >= num_index:
            break

    return _do_tiling(is_tiling_dict, num_index, output_shape)


def _set_off_value(tvm_ib, output, args_dict, repeats, offset=0):
    """Fill the output with off value.
    If the dtype of output is int8 or uint8, need cast.

    Parameters
    ----------
    tvm_ib : tvm.ir_builder.
        Developer API of IR node builder make function.
    output: list or tuple.
        Shape of the output.
    args_dict['depth']: int.
        A scalar defining the depth of the one hot dimension.
    args_dict['off_value']: int or float.
        A scalar defining the value to fill in output when indices[j] != i.
    args_dict['output_block_num']: int.
        The block num will be copied from UB to gm.
    args_dict['input_ub']: tvm.schedule.Buffer.
        The UB buffer for input indices.
    args_dict['output_ub']: tvm.schedule.Buffer.
        The UB buffer for output tensor.
    args_dict['cast_ub']: tvm.schedule.Buffer.
        The UB buffer for cast result.
    args_dict['ub_row_num_once']: int.
        The row num will be dealed with in one loop.
    args_dict['reg']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.
    args_dict['output_offset']: int.
        Offset of output buffer in current loop.
    args_dict['reg_for_idx']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.
    repeats: int
        The repeat times of instruction.
    offset: int
        The offset of instruction.

    Returns
    -------
    None.
    """
    if output.dtype == "uint8" or output.dtype == "int8":
        if output.dtype == "int8":
            inst = "vconv_f162s8"
        else:
            inst = "vconv_f162u8"
        tvm_ib.emit(
            tvm.call_extern(
                "float16", "vector_dup",
                args_dict.get('output_ub').access_ptr("w", offset=offset),
                (args_dict.get('reg')[1].astype("int8")).astype("float16"),
                repeats, 1, 1, 8, 8))
        tvm_ib.emit(
            tvm.call_extern(
                output.dtype, inst,
                args_dict.get('cast_ub').access_ptr("w", offset=offset),
                args_dict.get('output_ub').access_ptr("r", offset=offset),
                repeats, 1, 1, 4, 8))
    else:
        tvm_ib.emit(
            tvm.call_extern(
                output.dtype, "vector_dup",
                args_dict.get('output_ub').access_ptr("w", offset=offset),
                args_dict.get('reg')[1], repeats, 1, 1, 8, 8))


def _set_mask(length):
    """
    calculate MASK in cce

    Parameters
    ----------
    length : int
        calculate length

    Returns
    -------
    mask : tuple of int
        low and high bit of mask.
    """
    length = int(length)
    mask1 = 2**max(length - 64, 0) - 1  # high 64bits
    mask2 = 2**min(length, 64) - 1  # low 64bits
    return mask1, mask2


def _get_operation_num_one_for_loop(loop_num_origin, max_operation_num):
    """Calculate the operation num in one for loop

    Parameters
    ----------
    loop_num_origin : int.
        The original loop num.
    max_operation_num : int.
        The max operation num in one for loop.

    Returns
    -------
    The operation num in one loop,
    and new loop num.
    """
    loop_cnt = int(loop_num_origin) // max_operation_num
    left_cnt = int(loop_num_origin) % max_operation_num

    if loop_cnt:
        operation_num = max_operation_num
    else:
        operation_num = 0

    return operation_num, loop_cnt, left_cnt


def _do_operation_large_depth(tvm_ib, output, args_dict):
    """The calculation process for one_hot when depth is too large:
        1.Set output_ub buffer all to off_value.
        2.Set on_value according to input indices.
        3.Copy output_ub from UB to gm.

    Parameters
    ----------
    tvm_ib : tvm.ir_builder.
        Developer API of IR node builder make function.
    output: list or tuple.
        Shape of the output.
    args_dict['depth']: int.
        A scalar defining the depth of the one hot dimension.
    args_dict['off_value']: int or float.
        A scalar defining the value to fill in output when indices[j] != i.
    args_dict['output_block_num']: int.
        The block num will be copied from UB to gm.
    args_dict['input_ub']: tvm.schedule.Buffer.
        The UB buffer for input indices.
    args_dict['output_ub']: tvm.schedule.Buffer.
        The UB buffer for output tensor.
    args_dict['cast_ub']: tvm.schedule.Buffer.
        The UB buffer for cast result.
    args_dict['ub_row_num_once']: int.
        The row num will be dealed with in one loop.
    args_dict['reg']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.
    args_dict['output_offset']: int.
        Offset of output buffer in current loop.
    args_dict['reg_for_idx']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.

    Returns
    -------
    None.
    """
    var_dict = {}
    var_dict['dup_repeat_num'], var_dict['each_repeat_num'] = \
        _get_dup_repeat_num((1, args_dict.get('num_current_fragment')),
                            output.dtype)

    # all elements of output are set to off_value
    # for the vector_dup directive, the maximum number of iterations per
    # execution is 255.
    if var_dict.get('dup_repeat_num') > 0 and \
            var_dict.get('dup_repeat_num') <= 255:
        _set_off_value(tvm_ib, output, args_dict,
                       var_dict.get('dup_repeat_num'))
    elif var_dict.get('dup_repeat_num') > 255:
        rest_repeat_num = var_dict.get('dup_repeat_num')
        count = 0
        while rest_repeat_num > 255:
            dup_offset = count * 255 * var_dict.get('each_repeat_num')
            count = count + 1
            _set_off_value(tvm_ib, output, args_dict, 255, dup_offset)
            rest_repeat_num = rest_repeat_num - 255

        dup_offset = count * 255 * var_dict.get('each_repeat_num')

        if output.dtype == "int8" or output.dtype == "uint8":
            mask1, mask2 = \
                _set_mask(args_dict.get('num_current_fragment') % 128)
            _set_off_value(tvm_ib, output, args_dict, rest_repeat_num - 1,
                           dup_offset)
            tvm_ib.emit(
                tvm.call_extern("float16", "set_vector_mask",
                                tvm.const(mask1, dtype="uint64"),
                                tvm.const(mask2, dtype="uint64")))
            _set_off_value(tvm_ib, output, args_dict, 1,
                           dup_offset + (rest_repeat_num - 1) * 128)
            mask1, mask2 = _set_mask(128)
            tvm_ib.emit(
                tvm.call_extern("float16", "set_vector_mask",
                                tvm.const(mask1, dtype="uint64"),
                                tvm.const(mask2, dtype="uint64")))
        else:
            _set_off_value(tvm_ib, output, args_dict, rest_repeat_num,
                           dup_offset)

    if output.dtype == "int8" or output.dtype == "uint8":
        output_ub = args_dict.get('cast_ub')
    else:
        output_ub = args_dict.get('output_ub')

    # get index
    tvm_ib.emit(
        tvm.call_extern(
            args_dict.get('reg_for_idx').dtype, "reg_mov",
            tvm.call_extern(
                args_dict.get('reg_for_idx').dtype, "reg",
                args_dict.get('reg_for_idx')[0]),
            args_dict.get('input_ub').access_ptr("r")))

    index = args_dict.get('reg_for_idx')[0]
    with tvm_ib.if_scope(
            tvm.all(index >= args_dict.get('begin_idx'),
                    index <= args_dict.get('end_idx'))):
        # set on_value
        tvm_ib.emit(
            tvm.call_extern(
                args_dict.get('reg').dtype, "reg_mov",
                output_ub.access_ptr("rw",
                                     offset=index -
                                     args_dict.get('begin_idx')),
                tvm.call_extern(
                    args_dict.get('reg').dtype, "reg",
                    args_dict.get('reg')[0])))

    # copy ub to gm
    if (not args_dict.get('is_core_last_fragment')) or\
            _is_32_byte_align(args_dict.get('num_current_fragment'),
                              output.dtype) or\
            (args_dict.get('output_block_num') == 1):
        tvm_ib.emit(
            tvm.call_extern(
                output.dtype, "copy_ubuf_to_gm",
                output.access_ptr("w", offset=args_dict.get('output_offset')),
                output_ub.access_ptr("r"), 0, 1,
                args_dict.get('output_block_num'), 0, 0))
    else:
        # copy part of 32B align
        tvm_ib.emit(
            tvm.call_extern(
                output.dtype, "copy_ubuf_to_gm",
                output.access_ptr("w", offset=args_dict.get('output_offset')),
                output_ub.access_ptr("r"), 0, 1,
                args_dict.get('output_block_num') - 1, 0, 0))

        var_dict['offset_junction'] = args_dict.get('num_current_fragment') -\
                                      args_dict.get('num_in_32B')
        for j in range(args_dict.get('num_in_32B')):
            tvm_ib.emit(
                tvm.call_extern(
                    output.dtype, "reg_mov",
                    args_dict.get('ub_for_32B').access_ptr("rw", offset=j),
                    output_ub.access_ptr(
                        "r", offset=var_dict.get('offset_junction') + j)))

        tvm_ib.emit(
            tvm.call_extern(
                output.dtype, "copy_ubuf_to_gm",
                output.access_ptr("w",
                                  offset=args_dict.get('output_offset') +
                                  var_dict.get('offset_junction')),
                args_dict.get('ub_for_32B').access_ptr("r"), 0, 1, 1, 0, 0))


def _do_operation(tvm_ib,
                  output,
                  args_dict,
                  is_multicore_junction=False,
                  is_32b_align=True):
    """The calculation process for one_hot:
        1.Set output_ub buffer all to off_value.
        2.Set on_value according to input indices.
        3.Copy output_ub from UB to gm.

    Parameters
    ----------
    tvm_ib : tvm.ir_builder.
        Developer API of IR node builder make function.
    output: list or tuple.
        Shape of the output.
    args_dict['depth']: int.
        A scalar defining the depth of the one hot dimension.
    args_dict['off_value']: int or float.
        A scalar defining the value to fill in output when indices[j] != i.
    args_dict['output_block_num']: int.
        The block num will be copied from UB to gm.
    args_dict['input_ub']: tvm.schedule.Buffer.
        The UB buffer for input indices.
    args_dict['output_ub']: tvm.schedule.Buffer.
        The UB buffer for output tensor.
    args_dict['cast_ub']: tvm.schedule.Buffer.
        The UB buffer for cast result.
    args_dict['ub_row_num_once']: int.
        The row num will be dealed with in one loop.
    args_dict['reg']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.
    args_dict['output_offset']: int.
        Offset of output buffer in current loop.
    args_dict['reg_for_idx']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.

    Returns
    -------
    None.
    """
    var_dict = {}
    var_dict['dup_repeat_num'], var_dict[
        'each_repeat_num'] = _get_dup_repeat_num(
            (args_dict.get('ub_row_num_once'), args_dict.get('depth')),
            output.dtype)

    # all elements of output are set to off_value
    # for the vector_dup directive, the maximum number of iterations per execution
    # is 255.
    if var_dict.get('dup_repeat_num') > 0 and var_dict.get(
            'dup_repeat_num') <= 255:
        _set_off_value(tvm_ib, output, args_dict,
                       var_dict.get('dup_repeat_num'))
    elif var_dict.get('dup_repeat_num') > 255:
        rest_repeat_num = var_dict.get('dup_repeat_num')
        count = 0
        while rest_repeat_num > 255:
            dup_offset = count * 255 * var_dict.get('each_repeat_num')
            count = count + 1
            _set_off_value(tvm_ib, output, args_dict, 255, dup_offset)
            rest_repeat_num = rest_repeat_num - 255

        dup_offset = count * 255 * var_dict.get('each_repeat_num')

        if output.dtype == "int8" or output.dtype == "uint8":
            mask1, mask2 = _set_mask(
                args_dict.get('ub_row_num_once') * args_dict.get('depth') %
                128)
            _set_off_value(tvm_ib, output, args_dict, rest_repeat_num - 1,
                           dup_offset)
            tvm_ib.emit(
                tvm.call_extern("float16", "set_vector_mask",
                                tvm.const(mask1, dtype="uint64"),
                                tvm.const(mask2, dtype="uint64")))
            _set_off_value(tvm_ib, output, args_dict, 1,
                           dup_offset + (rest_repeat_num - 1) * 128)
            mask1, mask2 = _set_mask(128)
            tvm_ib.emit(
                tvm.call_extern("float16", "set_vector_mask",
                                tvm.const(mask1, dtype="uint64"),
                                tvm.const(mask2, dtype="uint64")))
        else:
            _set_off_value(tvm_ib, output, args_dict, rest_repeat_num,
                           dup_offset)

    if output.dtype == "int8" or output.dtype == "uint8":
        output_ub = args_dict.get('cast_ub')
    else:
        output_ub = args_dict.get('output_ub')

    # set on_value according to input indices
    var_dict['num_operation'], var_dict['new_loop_num'], \
    var_dict['left_line'] = _get_operation_num_one_for_loop(
        args_dict.get('ub_row_num_once'), 8)

    def _inner_set_on_value(operate_num, idx):
        """
        set on value
        """
        # get index from UB
        for j in range(operate_num):
            tvm_ib.emit(
                tvm.call_extern(
                    args_dict.get('reg_for_idx').dtype,
                    "reg_mov",
                    tvm.call_extern(
                        args_dict.get('reg_for_idx').dtype, "reg",
                        args_dict.get('reg_for_idx')[j]),
                    args_dict.get('input_ub').access_ptr(
                        "r",
                        offset=idx * var_dict.get('num_operation') + j),
                ))
        # set on_value
        for j in range(operate_num):
            index = args_dict.get('reg_for_idx')[j]
            with tvm_ib.new_scope():
                tvm_ib.scope_attr(cce.cce_params.CCE_AXIS, "coproc_scope", 1)
                with tvm_ib.if_scope(
                        tvm.all(index >= 0, index < args_dict.get('depth'))):
                    tvm_ib.emit(
                        tvm.call_extern(
                            args_dict.get('reg').dtype, "reg_mov",
                            output_ub.access_ptr(
                                "rw",
                                offset=((idx * var_dict.get('num_operation')
                                         + j) * args_dict.get('depth') +
                                        index)),
                            tvm.call_extern(
                                args_dict.get('reg').dtype, "reg",
                                args_dict.get('reg')[0])))

    if var_dict.get('new_loop_num'):
        with tvm_ib.for_range(0, var_dict.get('new_loop_num')) as i:
            _inner_set_on_value(var_dict.get('num_operation'), i)

    if var_dict.get('left_line'):
        _inner_set_on_value(var_dict.get('left_line'),
                            var_dict.get('new_loop_num'))

    if (is_32b_align is True) or (is_multicore_junction is False) or \
       (args_dict.get('output_block_num') == 1):
        # copy data from UB to gm
        tvm_ib.emit(
            tvm.call_extern(
                output.dtype, "copy_ubuf_to_gm",
                output.access_ptr("w",
                                  offset=(args_dict.get('output_offset') *
                                          args_dict.get('depth'))),
                output_ub.access_ptr("r"), 0, 1,
                args_dict.get('output_block_num'), 0, 0))
    else:
        # copy part of 32B align
        tvm_ib.emit(
            tvm.call_extern(
                output.dtype, "copy_ubuf_to_gm",
                output.access_ptr("w",
                                  offset=(args_dict.get('output_offset') *
                                          args_dict.get('depth'))),
                output_ub.access_ptr("r"), 0, 1,
                args_dict.get('output_block_num') - 1, 0, 0))

        var_dict['offset_junction'] = (
            args_dict.get('ub_row_num_once') *
            args_dict.get('depth')) - args_dict.get('num_in_32B')
        for j in range(args_dict.get('num_in_32B')):
            tvm_ib.emit(
                tvm.call_extern(
                    output.dtype, "reg_mov",
                    args_dict.get('ub_for_32B').access_ptr("rw", offset=j),
                    output_ub.access_ptr(
                        "r", offset=var_dict.get('offset_junction') + j)))

        tvm_ib.emit(
            tvm.call_extern(
                output.dtype, "copy_ubuf_to_gm",
                output.access_ptr("w",
                                  offset=(args_dict.get('output_offset') *
                                          args_dict.get('depth')) +
                                  var_dict.get('offset_junction')),
                args_dict.get('ub_for_32B').access_ptr("r"), 0, 1, 1, 0, 0))


def _get_copy_block_num(input_shape, input_dtype, output_shape, output_dtype):
    """Get the block num for generic DMA move.

    Parameters
    ----------
    input_shape: list or tuple.
        The indices shape that will be copied to UB.
    input_dtype: str.
        The data type of input tensor.
    output_shape: list or tuple.
        The shape that will be copied from UB to gm.
    output_dtype: str.
        The data type of output tensor.

    Returns
    -------
    input_block_num: int.
        The block num will be copied to UB.
    output_block_num: int.
        The block num will be copied from UB to gm.
    """
    input_size = int(functools_reduce(lambda i, j: i * j, input_shape))
    output_size = int(functools_reduce(lambda i, j: i * j, output_shape))

    if input_dtype == "int32":
        each_block_num = 8
    else:
        each_block_num = 32

    input_block_num = input_size // each_block_num
    if (input_size - input_block_num * each_block_num) > 0:
        input_block_num = input_block_num + 1

    if output_dtype == "float16":
        each_block_num = 16
    elif output_dtype in ("int8", "uint8"):
        each_block_num = 32
    else:
        each_block_num = 8

    output_block_num = output_size // each_block_num
    if (output_size - output_block_num * each_block_num) > 0:
        output_block_num = output_block_num + 1

    return input_block_num, output_block_num


def _is_32_byte_align(depth, output_dtype):
    """Check if it is a multiple of 32 bytes.

    Parameters
    ----------
    depth: int.
        A scalar defining the depth of the one hot dimension.
    output_dtype: str.
        The data type of output tensor.

    Returns
    -------
    ret: bool.
        If ret is true, that means depth is a multiple of 32 bytes.
    """
    ret = True
    if output_dtype == "float16":
        byte_each_num = 2
    elif output_dtype == "int8" or "uint8":
        byte_each_num = 1
    else:
        byte_each_num = 4

    byte_all = int(depth * byte_each_num)
    if (byte_all % 32) != 0:
        ret = False

    return ret


def _is_greater_than_32b(row_num_each_core, depth, indices_dtype,
                         output_dtype):
    """Check if the last DMA num is greater than 32 bytes.

    Parameters
    ----------
    row_num_each_core: int.
        The row num that the each core will deal with.
    depth: int.
        A scalar defining the depth of the one hot dimension.
    indices_dtype: str.
        The data type of indices tensor.
    output_dtype: str.
        The data type of output tensor.

    Returns
    -------
    ret: bool.
        If ret is true, that means the last DAM is greater than 32 bytes.
    """
    byte_one_num = cce.cce_intrin.get_bit_len(output_dtype) // 8
    if depth * byte_one_num >= 32:
        return True

    is_tiling_dict = _is_need_tiling((row_num_each_core, ), indices_dtype,
                                     (row_num_each_core, depth), output_dtype)
    if (is_tiling_dict.get('is_tiling') is
            False) and (row_num_each_core * depth * byte_one_num >= 32):
        return True

    # tiling is needed, and check the data num of the last DMA
    if (is_tiling_dict.get('is_tiling') is True) and (
            is_tiling_dict.get('last_loop_row_num') * depth * byte_one_num >=
            32):
        return True

    return False


def _get_core_num_depth_large(indices_input, depth, output_dtype):
    """Get the core num which will be used in device when depth is too large.

    Parameters
    ----------
    indices_input: TVM Tensor.
        The input indices tensor.
    depth: int.
        A scalar defining the depth of the one hot dimension.
    output_dtype: str.
        The data type of output tensor.

    Returns
    -------
    target_core_num: int.
        The core num will be used in cloud.
    row_num_each_core: int.
        The row num that the each core will deal with.
    """
    input_shape_0 = int(indices_input.shape[0])

    cloud_core_num = 32
    target_core_num = cloud_core_num
    for i in reversed(list(range(1, cloud_core_num + 1))):
        if int(input_shape_0 % i) == 0:
            target_core_num = i
            break

    row_num_each_core = input_shape_0 // target_core_num
    _, _, num_last_fragment = _split_large_depth_func(depth, output_dtype)
    byte_size_output_dtype = cce.cce_intrin.get_bit_len(output_dtype) // 8
    # if the last DMA is less than 32 byte, then multi-cores can not be used
    if num_last_fragment * byte_size_output_dtype < 32:
        target_core_num = 1
        row_num_each_core = input_shape_0

    return target_core_num, row_num_each_core


def _get_target_core_num(indices_input, depth, output_dtype, axis):
    """Get the core num which will be used in cloud device.

    Parameters
    ----------
    indices_input: TVM Tensor.
        The input indices tensor.
    depth: int.
        A scalar defining the depth of the one hot dimension.
    output_dtype: str.
        The data type of output tensor.

    Returns
    -------
    target_core_num: int.
        The core num will be used in cloud.
    row_num_each_core: int.
        The row num that the each core will deal with.
    """
    input_shape_0 = int(indices_input.shape[0])

    cloud_core_num = 32
    target_core_num = cloud_core_num
    for i in reversed(list(range(1, cloud_core_num + 1))):
        if int(input_shape_0 % i) == 0:
            target_core_num = i
            break

    if axis != -1:
        target_core_num = 1

    row_num_each_core = input_shape_0 // target_core_num
    # check if the last DMA is greater equal than 32B. If less than 32B, only
    # use single core
    if not _is_greater_than_32b(row_num_each_core, depth, indices_input.dtype,
                                output_dtype):
        target_core_num = 1
        row_num_each_core = input_shape_0

    return target_core_num, row_num_each_core


def _get_target_core_num_mini(indices_input):
    """Get the core num which will be used in mini device.

    Parameters
    ----------
    indices_input: TVM Tensor.
        The input indices tensor.
    depth: int.
        A scalar defining the depth of the one hot dimension.
    output_dtype: str.
        The data type of output tensor.

    Returns
    -------
    target_core_num: int.
        The core num will be used in mini.
    row_num_each_core: int.
        The row num that the each core will deal with.
    """
    input_shape_0 = int(indices_input.shape[0])
    mini_core_num = 2
    target_core_num = mini_core_num

    if 1 <= input_shape_0 <= 32:
        target_core_num = 1
        row_num_core_a = input_shape_0
        row_num_core_b = 0
        return target_core_num, row_num_core_a, row_num_core_b

    row_num_core_a = input_shape_0 // 2
    row_num_core_b = input_shape_0 - row_num_core_a

    return target_core_num, row_num_core_a, row_num_core_b


def _set_common_op_args(args_dict, end_idx, output_block_num,
                        num_current_fragment, is_core_last_fragment):
    """
    Set args for _one_core_depth_large
    """
    args_dict['end_idx'] = end_idx
    args_dict['output_block_num'] = output_block_num
    args_dict['num_current_fragment'] = num_current_fragment
    args_dict['is_core_last_fragment'] = is_core_last_fragment


def _set_common_op_args_extend(args_dict, begin_idx, id_fragment,
                               output_offset):
    """
    Set args for _one_core_depth_large
    """
    args_dict['begin_idx'] = begin_idx
    args_dict['id_fragment'] = id_fragment
    args_dict['output_offset'] = output_offset


def _one_core_depth_large(tvm_ib, indices_input, output, args_dict):
    """When depth is too large, operations for each core in a multi-core
       scenario, and the operation consists of the following steps:
        1. Copy indices array from gm to UB.
        2. Check whether tiling operation is needed.
        3. Do the function of _do_operation.

    Parameters
    ----------
    tvm_ib : tvm.ir_builder.
        Developer API of IR node builder make function.
    indices_input: TVM Tensor.
        The input indices tensor.
    output: TVM Tensor.
        The output tensor.
    args_dict['depth']: int.
        A scalar defining the depth of the one hot dimension.
    args_dict['off_value']: int or float.
        A scalar defining the value to fill in output when indices[j] != i.
    args_dict['reg']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.
    args_dict['row_num_current_core']: int.
        The row num that current core will deal with.
    args_dict['row_num_pre_core']: int.
        The row num that previous core deal with.
    args_dict['block_index']: tvm.schedule.IterVar.
        The index of current core in multi-core.
    args_dict['reg_for_idx']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.

    Returns
    -------
    None.
    """
    fragment_loop_num, num_each_fragment, num_last_fragment =\
        _split_large_depth_func(args_dict.get('depth'), output.dtype)
    # alloc UB for just one index
    input_ub = _new_alloc(tvm_ib,
                          indices_input.dtype, (1, ),
                          "input_ub",
                          scope=cce.scope_ubuf)

    op_args = {}
    op_args['input_block_num'], output_block_num =\
        _get_copy_block_num((1,), indices_input.dtype, (num_each_fragment,),
                            output.dtype)
    _, output_block_num_last = _get_copy_block_num(
        (1, ), indices_input.dtype, (num_last_fragment, ), output.dtype)

    if output.dtype == "uint8" or output.dtype == "int8":
        op_args['output_ub'] = _new_alloc(tvm_ib,
                                          "float16", (num_each_fragment, ),
                                          "output_ub",
                                          scope=cce.scope_ubuf)
        op_args['cast_ub'] = _new_alloc(tvm_ib,
                                        output.dtype, (num_each_fragment, ),
                                        "cast_ub",
                                        scope=cce.scope_ubuf)
    else:
        op_args['output_ub'] = _new_alloc(tvm_ib,
                                          output.dtype, (num_each_fragment, ),
                                          "output_ub",
                                          scope=cce.scope_ubuf)

    op_args['num_in_32B'] = 32 * 8 // cce.cce_intrin.get_bit_len(output.dtype)
    op_args['ub_for_32B'] = _new_alloc(tvm_ib,
                                       output.dtype, (op_args['num_in_32B'], ),
                                       "ub_for_32B",
                                       scope=cce.scope_ubuf)

    op_args['num_each_fragment'] = num_each_fragment
    op_args['depth'] = args_dict.get('depth')
    op_args['reg'] = args_dict.get('reg')
    op_args['reg_for_idx'] = args_dict.get('reg_for_idx')
    op_args['input_ub'] = input_ub

    with tvm_ib.for_range(0, args_dict.get('row_num_current_core')) as id_row:
        with tvm_ib.if_scope(id_row == (args_dict.get('row_num_current_core') -
                                        1)):
            # copy input in the first fragment of one depth
            indices_offset = args_dict.get('block_index') *\
                             args_dict.get('row_num_pre_core') + id_row
            tvm_ib.emit(
                tvm.call_extern(
                    input_ub.dtype, "copy_gm_to_ubuf",
                    input_ub.access_ptr("w"),
                    indices_input.access_ptr("r", offset=indices_offset), 0, 1,
                    op_args.get('input_block_num'), 0, 0))

            with tvm_ib.for_range(0, fragment_loop_num) as id_fragment:
                _set_common_op_args_extend(
                    op_args, id_fragment * num_each_fragment, id_fragment,
                    indices_offset * args_dict.get('depth') +
                    id_fragment * num_each_fragment)

                with tvm_ib.if_scope(id_fragment == (fragment_loop_num - 1)):
                    _set_common_op_args(op_args, op_args['depth'] - 1,
                                        output_block_num_last,
                                        num_last_fragment, True)
                    _do_operation_large_depth(tvm_ib, output, op_args)
                with tvm_ib.else_scope():
                    _set_common_op_args(
                        op_args, (id_fragment + 1) * num_each_fragment - 1,
                        output_block_num, num_each_fragment, False)
                    _do_operation_large_depth(tvm_ib, output, op_args)
        with tvm_ib.else_scope():
            # copy input in the first fragment of one depth
            indices_offset = args_dict.get('block_index') *\
                             args_dict.get('row_num_pre_core') + id_row
            tvm_ib.emit(
                tvm.call_extern(
                    input_ub.dtype, "copy_gm_to_ubuf",
                    input_ub.access_ptr("w"),
                    indices_input.access_ptr("r", offset=indices_offset), 0, 1,
                    op_args.get('input_block_num'), 0, 0))

            with tvm_ib.for_range(0, fragment_loop_num) as id_fragment:
                _set_common_op_args_extend(
                    op_args, id_fragment * num_each_fragment, id_fragment,
                    indices_offset * args_dict.get('depth') +
                    id_fragment * num_each_fragment)

                with tvm_ib.if_scope(id_fragment == (fragment_loop_num - 1)):
                    _set_common_op_args(op_args, op_args['depth'] - 1,
                                        output_block_num_last,
                                        num_last_fragment, False)
                    _do_operation_large_depth(tvm_ib, output, op_args)
                with tvm_ib.else_scope():
                    _set_common_op_args(
                        op_args, (id_fragment + 1) * num_each_fragment - 1,
                        output_block_num, num_each_fragment, False)
                    _do_operation_large_depth(tvm_ib, output, op_args)


def _one_core_in_multi_ir(tvm_ib, indices_input, output, args_dict):
    """Operations for each core in a multi-core scenario, and the operation
    consists
       of the following steps:
        1. Copy indices array from gm to UB.
        2. Check whether tiling operation is needed.
        3. Do the function of _do_operation.

    Parameters
    ----------
    tvm_ib : tvm.ir_builder.
        Developer API of IR node builder make function.
    indices_input: TVM Tensor.
        The input indices tensor.
    output: TVM Tensor.
        The output tensor.
    args_dict['depth']: int.
        A scalar defining the depth of the one hot dimension.
    args_dict['off_value']: int or float.
        A scalar defining the value to fill in output when indices[j] != i.
    args_dict['reg']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.
    args_dict['row_num_current_core']: int.
        The row num that current core will deal with.
    args_dict['row_num_pre_core']: int.
        The row num that previous core deal with.
    args_dict['block_index']: tvm.schedule.IterVar.
        The index of current core in multi-core.
    args_dict['reg_for_idx']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.

    Returns
    -------
    None.
    """
    output_shape_each_core = (args_dict.get('row_num_current_core'),
                              args_dict.get('depth'))
    input_ub_shape = (args_dict.get('row_num_current_core'), )
    is_tiling_dict = _is_need_tiling(input_ub_shape, indices_input.dtype,
                                     output_shape_each_core, output.dtype)

    op_args = {}
    op_args['depth'] = args_dict.get('depth')
    op_args['reg'] = args_dict.get('reg')
    op_args['reg_for_idx'] = args_dict.get('reg_for_idx')
    op_args['num_in_32B'] = 32 * 8 // cce.cce_intrin.get_bit_len(output.dtype)

    op_args['ub_for_32B'] = _new_alloc(tvm_ib,
                                       output.dtype, (op_args['num_in_32B'], ),
                                       "ub_for_32B",
                                       scope=cce.scope_ubuf)

    # check if tiling is needed for each core
    if is_tiling_dict.get('is_tiling') is False:
        # tiling is not needed
        # allocate ub space for indices
        input_ub = _new_alloc(tvm_ib,
                              indices_input.dtype,
                              input_ub_shape,
                              "input_ub",
                              scope=cce.scope_ubuf)
        input_block_num, output_block_num = _get_copy_block_num(
            input_ub_shape, indices_input.dtype, output_shape_each_core,
            output.dtype)

        tvm_ib.emit(
            tvm.call_extern(
                input_ub.dtype, "copy_gm_to_ubuf", input_ub.access_ptr("w"),
                indices_input.access_ptr(
                    "r",
                    offset=(args_dict.get('block_index') *
                            args_dict.get('row_num_pre_core'))), 0, 1,
                input_block_num, 0, 0))

        op_args['input_ub'] = input_ub
        if output.dtype == "uint8" or output.dtype == "int8":
            op_args['output_ub'] = _new_alloc(tvm_ib,
                                              "float16",
                                              output_shape_each_core,
                                              "output_ub",
                                              scope=cce.scope_ubuf)
            op_args['cast_ub'] = _new_alloc(tvm_ib,
                                            output.dtype,
                                            output_shape_each_core,
                                            "cast_ub",
                                            scope=cce.scope_ubuf)
        else:
            op_args['output_ub'] = _new_alloc(tvm_ib,
                                              output.dtype,
                                              output_shape_each_core,
                                              "output_ub",
                                              scope=cce.scope_ubuf)
        op_args['output_block_num'] = output_block_num
        op_args['ub_row_num_once'] = args_dict.get('row_num_current_core')
        op_args['output_offset'] = args_dict.get(
            'block_index') * args_dict.get('row_num_pre_core')

        _do_operation(tvm_ib,
                      output,
                      op_args,
                      is_multicore_junction=True,
                      is_32b_align=_is_32_byte_align(
                          args_dict.get('row_num_current_core') *
                          args_dict.get('depth'), output.dtype))
    else:
        # each core needs tiling
        input_ub = _new_alloc(tvm_ib,
                              indices_input.dtype,
                              (is_tiling_dict.get('ub_row_num_once'), ),
                              "input_ub",
                              scope=cce.scope_ubuf)
        output_ub_shape = (is_tiling_dict.get('ub_row_num_once'),
                           args_dict.get('depth'))
        if output.dtype == "uint8" or output.dtype == "int8":
            op_args['output_ub'] = _new_alloc(tvm_ib,
                                              "float16",
                                              output_ub_shape,
                                              "output_ub",
                                              scope=cce.scope_ubuf)
            op_args['cast_ub'] = _new_alloc(tvm_ib,
                                            output.dtype,
                                            output_ub_shape,
                                            "cast_ub",
                                            scope=cce.scope_ubuf)
        else:
            op_args['output_ub'] = _new_alloc(tvm_ib,
                                              output.dtype,
                                              output_ub_shape,
                                              "output_ub",
                                              scope=cce.scope_ubuf)

        input_block_num, output_block_num = _get_copy_block_num(
            (is_tiling_dict.get('ub_row_num_once'), ), indices_input.dtype,
            (is_tiling_dict.get('ub_row_num_once'), args_dict.get('depth')),
            output.dtype)
        input_block_num_last, output_block_num_last = _get_copy_block_num(
            (is_tiling_dict.get('last_loop_row_num'), ), indices_input.dtype,
            (is_tiling_dict.get('last_loop_row_num'), args_dict.get('depth')),
            output.dtype)

        with tvm_ib.for_range(0, is_tiling_dict.get('ub_loop')) as i:
            op_args['output_offset'] = (
                args_dict.get('block_index') *
                args_dict.get('row_num_pre_core')) + (
                    i * is_tiling_dict.get('ub_row_num_once'))

            with tvm_ib.if_scope(i == (is_tiling_dict.get('ub_loop') - 1)):
                tvm_ib.emit(
                    tvm.call_extern(
                        input_ub.dtype, "copy_gm_to_ubuf",
                        input_ub.access_ptr("w"),
                        indices_input.access_ptr(
                            "r", offset=op_args.get('output_offset')), 0, 1,
                        input_block_num_last, 0, 0))

                op_args['input_ub'] = input_ub
                op_args['output_block_num'] = output_block_num_last
                op_args['ub_row_num_once'] = is_tiling_dict.get(
                    'last_loop_row_num')
                _do_operation(tvm_ib,
                              output,
                              op_args,
                              is_multicore_junction=True,
                              is_32b_align=_is_32_byte_align(
                                  is_tiling_dict.get('last_loop_row_num') *
                                  args_dict.get('depth'), output.dtype))
            with tvm_ib.else_scope():
                tvm_ib.emit(
                    tvm.call_extern(
                        input_ub.dtype, "copy_gm_to_ubuf",
                        input_ub.access_ptr("w"),
                        indices_input.access_ptr(
                            "r", offset=op_args.get('output_offset')), 0, 1,
                        input_block_num, 0, 0))

                op_args['input_ub'] = input_ub
                op_args['output_block_num'] = output_block_num
                op_args['ub_row_num_once'] = is_tiling_dict.get(
                    'ub_row_num_once')
                _do_operation(tvm_ib, output, op_args)


# pylint: disable=locally-disabled,too-many-locals,too-many-statements
def _one_core_ir_not_neg_one(tvm_ib, indices_input, output, args_dict):
    output_size_each_core = int(functools_reduce(lambda i, j: i * j, output.shape))
    input_ub_shape = indices_input.shape

    ub_info = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE) - 32 - \
              UB_SPACE_8K - ON_OFF_VAL_SPACE

    ub_indices_space = 128 * 4
    ub_on_off_space = 32 * 2
    ub_remaining = ub_info - ub_indices_space - ub_on_off_space
    copy_ele_num_once = ub_remaining // 4
    copy_ele_num_once = (copy_ele_num_once // 8) * 8

    loop_times = output_size_each_core // copy_ele_num_once
    if output_size_each_core % copy_ele_num_once != 0:
        copy_ele_num_last = output_size_each_core - loop_times * copy_ele_num_once
        loop_times = loop_times + 1

    # allocate ub
    input_ub = _new_alloc(tvm_ib, indices_input.dtype, input_ub_shape,
                          "input_ub", scope=cce.scope_ubuf)

    offset_ub = _new_alloc(tvm_ib, indices_input.dtype, input_ub_shape,
                           "offset_ub", scope=cce.scope_ubuf)

    args_dict["output_ub"] = _new_alloc(tvm_ib, output.dtype, (copy_ele_num_once,),
                                        "output_ub", scope=cce.scope_ubuf)

    # copy indices to input_ub
    tvm_ib.emit(
        tvm.call_extern(
            input_ub.dtype, "copy_gm_to_ubuf", input_ub.access_ptr("w"),
            indices_input.access_ptr("r"), 0, 1, 16, 0, 0))

    # compute offset and store to offset_ub
    with tvm_ib.for_range(0, input_ub_shape[0]) as i:
        tvm_ib.emit(
            tvm.call_extern(
                args_dict.get('reg_for_idx').dtype,
                "reg_mov",
                tvm.call_extern(args_dict.get('reg_for_idx').dtype, "reg",
                                args_dict.get('reg_for_idx')[0]),
                input_ub.access_ptr("r", offset=i)
            ))

        args_dict.get('reg_for_idx')[1] = args_dict.get('reg_for_idx')[0] * 128 + i
        tvm_ib.emit(
            tvm.call_extern(
                offset_ub.dtype, "reg_mov",
                offset_ub.access_ptr("rw", offset=i),
                tvm.call_extern(args_dict.get('reg_for_idx').dtype, "reg",
                                args_dict.get('reg_for_idx')[1])
            ))
    var_dict = {}
    with tvm_ib.for_range(0, loop_times) as loop_idx:
        with tvm_ib.if_scope(loop_idx == loop_times - 1):
            # dup output_ub to off_value
            var_dict['dup_repeat_num'], var_dict[
                'each_repeat_num'] = _get_dup_repeat_num((copy_ele_num_last,), output.dtype)

            if var_dict.get('dup_repeat_num') > 0 and var_dict.get(
                    'dup_repeat_num') <= 255:
                _set_off_value(tvm_ib, output, args_dict,
                               var_dict.get('dup_repeat_num'))
            elif var_dict.get('dup_repeat_num') > 255:
                rest_repeat_num = var_dict.get('dup_repeat_num')
                count = 0
                while rest_repeat_num > 255:
                    dup_offset = count * 255 * var_dict.get('each_repeat_num')
                    count = count + 1
                    _set_off_value(tvm_ib, output, args_dict, 255, dup_offset)
                    rest_repeat_num = rest_repeat_num - 255

                dup_offset = count * 255 * var_dict.get('each_repeat_num')
                _set_off_value(tvm_ib, output, args_dict, rest_repeat_num, dup_offset)
            args_dict.get('reg_for_idx')[2] = loop_idx * copy_ele_num_once
            args_dict.get('reg_for_idx')[3] = (loop_idx + 1) * copy_ele_num_once
            with tvm_ib.for_range(0, 128) as offset_idx:
                tvm_ib.emit(
                    tvm.call_extern(
                        args_dict.get('reg_for_idx').dtype,
                        "reg_mov",
                        tvm.call_extern(args_dict.get('reg_for_idx').dtype, "reg",
                                        args_dict.get('reg_for_idx')[4]),
                        offset_ub.access_ptr("r", offset=offset_idx),
                    ))
                offset_val = args_dict.get('reg_for_idx')[4]
                with tvm_ib.if_scope(tvm.all(offset_val >= args_dict.get('reg_for_idx')[2],
                                             offset_val < args_dict.get('reg_for_idx')[3])):
                    # set on_value
                    tvm_ib.emit(
                        tvm.call_extern(
                            args_dict.get('reg').dtype, "reg_mov",
                            args_dict["output_ub"].access_ptr(
                                "rw", offset=(offset_val - args_dict.get('reg_for_idx')[2])),
                            tvm.call_extern(args_dict.get('reg').dtype, "reg",
                                            args_dict.get('reg')[0])))

            # copy ret to gm
            tvm_ib.emit(
                tvm.call_extern(
                    output.dtype, "copy_ubuf_to_gm",
                    output.access_ptr("w", offset=loop_idx*copy_ele_num_once),
                    args_dict["output_ub"].access_ptr("r"),
                    0, 1, copy_ele_num_last//8, 0, 0))

        with tvm_ib.else_scope():
            # dup output_ub to off_value
            var_dict['dup_repeat_num'], var_dict[
                'each_repeat_num'] = _get_dup_repeat_num((copy_ele_num_once,),
                                                         output.dtype)

            if var_dict.get('dup_repeat_num') > 0 and var_dict.get(
                    'dup_repeat_num') <= 255:
                _set_off_value(tvm_ib, output, args_dict,
                               var_dict.get('dup_repeat_num'))
            elif var_dict.get('dup_repeat_num') > 255:
                rest_repeat_num = var_dict.get('dup_repeat_num')
                count = 0
                while rest_repeat_num > 255:
                    dup_offset = count * 255 * var_dict.get('each_repeat_num')
                    count = count + 1
                    _set_off_value(tvm_ib, output, args_dict, 255, dup_offset)
                    rest_repeat_num = rest_repeat_num - 255

                dup_offset = count * 255 * var_dict.get('each_repeat_num')
                _set_off_value(tvm_ib, output, args_dict, rest_repeat_num,
                               dup_offset)

            args_dict.get('reg_for_idx')[2] = loop_idx * copy_ele_num_once
            args_dict.get('reg_for_idx')[3] = (loop_idx + 1) * copy_ele_num_once
            with tvm_ib.for_range(0, 128) as offset_idx:
                tvm_ib.emit(
                    tvm.call_extern(
                        args_dict.get('reg_for_idx').dtype,
                        "reg_mov",
                        tvm.call_extern(args_dict.get('reg_for_idx').dtype, "reg",
                                        args_dict.get('reg_for_idx')[4]),
                        offset_ub.access_ptr("r", offset=offset_idx),
                    ))
                offset_val = args_dict.get('reg_for_idx')[4]
                with tvm_ib.if_scope(tvm.all(offset_val >= args_dict.get('reg_for_idx')[2],
                                             offset_val < args_dict.get('reg_for_idx')[3])):
                    # set on_value
                    tvm_ib.emit(
                        tvm.call_extern(
                            args_dict.get('reg').dtype, "reg_mov",
                            args_dict["output_ub"].access_ptr(
                                "rw", offset=(offset_val - args_dict.get('reg_for_idx')[2])),
                            tvm.call_extern(args_dict.get('reg').dtype, "reg",
                                            args_dict.get('reg')[0])))

            # copy ret to gm
            tvm_ib.emit(
                tvm.call_extern(
                    output.dtype, "copy_ubuf_to_gm",
                    output.access_ptr("w", offset=loop_idx*copy_ele_num_once),
                    args_dict["output_ub"].access_ptr("r"),
                    0, 1, copy_ele_num_once//8, 0, 0))


def _split_large_depth_func(depth, output_dtype):
    """do the split for the depth when depth is too large.

    Parameters
    ----------
    depth: int.
        A scalar defining the depth of the one hot dimension.
    output_dtype: str.
        The data type of output tensor.

    Returns
    -------
    loop_num: int.
        The loop num in one depth.
    num_ub_store_max: int.
        The max number elements in one time.
    num_last_loop: int.
        The element number in the last loop.
    """
    one_index_space = 32
    multi_core_space = 32
    ub_remaining = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE) -\
                   one_index_space - multi_core_space - UB_SPACE_8K -\
                   ON_OFF_VAL_SPACE
    byte_num_dtype = cce.cce_intrin.get_bit_len(output_dtype) // 8
    if output_dtype in ("uint8", "int8"):
        byte_num_dtype = 3

    num_ub_store_max = ub_remaining // byte_num_dtype
    if output_dtype in ("uint8", "int8"):
        for i in range(num_ub_store_max):
            num_ub_store_max_new = num_ub_store_max - i
            ub_fp16 = ((num_ub_store_max_new * 2 // 32) + 1) * 32
            ub_cast = ((num_ub_store_max_new // 32) + 1) * 32
            if (ub_fp16 + ub_cast) <= ub_remaining:
                break

        num_ub_store_max = num_ub_store_max_new

    loop_num = depth // num_ub_store_max
    num_last_loop = depth - loop_num * num_ub_store_max
    if num_last_loop != 0:
        loop_num = loop_num + 1
    else:
        num_last_loop = num_ub_store_max

    return loop_num, num_ub_store_max, num_last_loop


def _check_is_depth_large(depth, output_dtype):
    """Check if the depth is too large to store in UB.

    Parameters
    ----------
    depth: int.
        A scalar defining the depth of the one hot dimension.
    output_dtype: str.
        The data type of output tensor.

    Returns
    -------
    True will be return when the depth is too large.
    """
    one_index_space = 32
    multi_core_space = 32
    ub_remaining = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE) -\
                   one_index_space - multi_core_space - UB_SPACE_8K -\
                   ON_OFF_VAL_SPACE

    byte_num_dtype = cce.cce_intrin.get_bit_len(output_dtype) // 8
    # when output_dtype is int8 or uint8, ub_output(fp16 2B) and ub_cast(1B) is
    # needed, hence one element of depth needs 3B
    if byte_num_dtype == 1:
        ub_fp16 = ((depth * 2 // 32) + 1) * 32
        ub_cast = ((depth // 32) + 1) * 32
        ub_need_for_one_depth = ub_fp16 + ub_cast
    else:
        ub_need_for_one_depth = depth * byte_num_dtype

    if ub_need_for_one_depth > ub_remaining:
        return True

    return False


def _multi_core_ir_depth_large(tvm_ib, indices_input, output, args_dict):
    """operations when depth is too large.

    Parameters
    ----------
    tvm_ib : tvm.ir_builder.
        Developer API of IR node builder make function.
    indices_input: TVM Tensor.
        The input indices tensor.
    output: TVM Tensor.
        The output tensor.
    args_dict['depth']: int.
        A scalar defining the depth of the one hot dimension.
    args_dict['off_value']: int or float.
        A scalar defining the value to fill in output when indices[j] != i.
    args_dict['reg']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.
    args_dict['reg_for_idx']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.

    Returns
    -------
    None.
    """
    target_core_num, row_num_each_core =\
        _get_core_num_depth_large(indices_input, args_dict.get('depth'),
                                  output.dtype)
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", target_core_num)

    op_args = {}
    op_args['depth'] = args_dict.get('depth')
    op_args['reg'] = args_dict.get('reg')
    op_args['row_num_current_core'] = row_num_each_core
    op_args['row_num_pre_core'] = row_num_each_core
    op_args['block_index'] = block_index
    op_args['reg_for_idx'] = args_dict.get('reg_for_idx')

    _one_core_depth_large(tvm_ib, indices_input, output, op_args)


def _multi_core_ir_prime_case(tvm_ib, indices_input, output, args_dict):
    """
    case for prime multiple core
    """
    target_core_num = cce.cce_conf.get_soc_spec(cce.cce_conf.CORE_NUM)
    num_indices = int(indices_input.shape[0])
    row_num_front_31_core = num_indices // target_core_num
    row_num_last_core = \
        num_indices - (target_core_num - 1) * row_num_front_31_core

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", target_core_num)

    op_args = {}
    op_args['depth'] = args_dict.get('depth')
    op_args['reg'] = args_dict.get('reg')
    op_args['block_index'] = block_index
    op_args['reg_for_idx'] = args_dict.get('reg_for_idx')
    op_args['row_num_pre_core'] = row_num_front_31_core

    with tvm_ib.if_scope(block_index.var == target_core_num - 1):
        # the operation of last core
        op_args['row_num_current_core'] = row_num_last_core
        _one_core_in_multi_ir(tvm_ib, indices_input, output, op_args)
    with tvm_ib.else_scope():
        # the operation of front 31 cores
        op_args['row_num_current_core'] = row_num_front_31_core
        _one_core_in_multi_ir(tvm_ib, indices_input, output, op_args)


def _cloud_multi_core_ir(tvm_ib, indices_input, output, args_dict, axis):
    """operations for cloud.

    Parameters
    ----------
    tvm_ib : tvm.ir_builder.
        Developer API of IR node builder make function.
    indices_input: TVM Tensor.
        The input indices tensor.
    output: TVM Tensor.
        The output tensor.
    args_dict['depth']: int.
        A scalar defining the depth of the one hot dimension.
    args_dict['off_value']: int or float.
        A scalar defining the value to fill in output when indices[j] != i.
    args_dict['reg']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.
    args_dict['reg_for_idx']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.

    Returns
    -------
    None.
    """
    target_core_num, row_num_each_core = _get_target_core_num(
        indices_input, args_dict.get('depth'), output.dtype, axis)
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", target_core_num)

    op_args = {}
    op_args['depth'] = args_dict.get('depth')
    op_args['reg'] = args_dict.get('reg')
    op_args['row_num_current_core'] = row_num_each_core
    op_args['row_num_pre_core'] = row_num_each_core
    op_args['block_index'] = block_index
    op_args['reg_for_idx'] = args_dict.get('reg_for_idx')

    _one_core_in_multi_ir(tvm_ib, indices_input, output, op_args)


def _mini_multi_core_ir(tvm_ib, indices_input, output, args_dict):
    """operations for mini.

    Parameters
    ----------
    tvm_ib : tvm.ir_builder.
        Developer API of IR node builder make function.
    indices_input: TVM Tensor.
        The indices_input indices tensor.
    output: TVM Tensor.
        The output tensor.
    args_dict['depth']: int.
        A scalar defining the depth of the one hot dimension.
    args_dict['off_value']: int or float.
        A scalar defining the value to fill in output when indices[j] != i.
    args_dict['reg']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.
    args_dict['reg_for_idx']: tvm.ir_builder.BufferVar.
        The reg buffer to save scalar.

    Returns
    -------
    None.
    """
    target_core_num, row_num_core_a, row_num_core_b = \
        _get_target_core_num_mini(indices_input)
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", target_core_num)

    op_args = {}
    op_args['depth'] = args_dict.get('depth')
    op_args['reg'] = args_dict.get('reg')
    op_args['block_index'] = block_index
    op_args['reg_for_idx'] = args_dict.get('reg_for_idx')
    op_args['row_num_pre_core'] = row_num_core_a

    if target_core_num == 1:
        op_args['row_num_current_core'] = row_num_core_a
        _one_core_in_multi_ir(tvm_ib, indices_input, output, op_args)
    else:
        with tvm_ib.if_scope(block_index.var == 0):
            op_args['row_num_current_core'] = row_num_core_a
            _one_core_in_multi_ir(tvm_ib, indices_input, output, op_args)
        with tvm_ib.else_scope():
            op_args['row_num_current_core'] = row_num_core_b
            _one_core_in_multi_ir(tvm_ib, indices_input, output, op_args)


def _check_is_prime_case(indices_input, output_dtype, depth):
    indices_shape = indices_input.shape
    indices_num = int(indices_shape[0])

    if indices_num < 33:
        return False

    for i in range(2, 33):
        if indices_num % i == 0:
            return False

    byte_size_output_dtype = cce.cce_intrin.get_bit_len(output_dtype) // 8
    if depth * byte_size_output_dtype < 32:
        return False

    return True


# pylint: disable=locally-disabled,too-many-arguments
def one_hot_ir(ins, output, depth, axis):
    """IR node builder make function.

    Parameters
    ----------
    indices_input: TVM Tensor.
        The input indices tensor.
    output: TVM Tensor.
        The output tensor.
    depth: int.
        A scalar defining the depth of the one hot dimension.
    on_value: int or float.
        a scalar defining the value to fill in output when indices[j] = i.
    off_value: int or float.
        A scalar defining the value to fill in output when indices[j] != i.

    Returns
    -------
    stmt: the result statement.
    """
    indices_input = ins[0]
    on_value_input = ins[1]
    off_value_input = ins[2]

    tvm_ib = tvm.ir_builder.create()
    device_core_num = cce.cce_conf.get_soc_spec(cce.cce_conf.CORE_NUM)
    cloud_core_num = 32
    # reg[0]:on_value   reg[1]:off_value
    reg = tvm_ib.allocate(output.dtype, (2, ),
                          name="reg_buf",
                          scope=cce.scope_reg)
    reg_for_idx = tvm_ib.allocate(indices_input.dtype, (8, ),
                                  name="reg_buf_for_idx",
                                  scope=cce.scope_reg)

    # allocate ub buffer for on_value and off_value
    on_value_ub = _new_alloc(tvm_ib,
                             output.dtype, (1, ),
                             "on_value_ub",
                             scope=cce.scope_ubuf)
    off_value_ub = _new_alloc(tvm_ib,
                              output.dtype, (1, ),
                              "off_value_ub",
                              scope=cce.scope_ubuf)

    # copy on/off value from gm to ub
    tvm_ib.emit(
        tvm.call_extern(on_value_ub.dtype, "copy_gm_to_ubuf",
                        on_value_ub.access_ptr("w"),
                        on_value_input.access_ptr("r"), 0, 1, 1, 0, 0))
    tvm_ib.emit(
        tvm.call_extern(off_value_ub.dtype, "copy_gm_to_ubuf",
                        off_value_ub.access_ptr("w"),
                        off_value_input.access_ptr("r"), 0, 1, 1, 0, 0))

    # store on/off value to reg_buffer
    tvm_ib.emit(
        tvm.call_extern(reg.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[0]),
                        on_value_ub.access_ptr("r")))
    tvm_ib.emit(
        tvm.call_extern(reg.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[1]),
                        off_value_ub.access_ptr("r")))

    op_args = {}
    op_args['depth'] = depth
    op_args['reg'] = reg
    op_args['reg_for_idx'] = reg_for_idx

    if not _check_is_depth_large(depth, output.dtype):
        if _check_is_prime_case(indices_input, output.dtype, depth):
            _multi_core_ir_prime_case(tvm_ib, indices_input, output, op_args)
        else:
            if device_core_num == cloud_core_num:
                _cloud_multi_core_ir(tvm_ib, indices_input, output, op_args, axis)
            else:
                _mini_multi_core_ir(tvm_ib, indices_input, output, op_args)
    else:
        _multi_core_ir_depth_large(tvm_ib, indices_input, output, op_args)

    return tvm_ib.get()


def _one_hot_ir_not_neg_one(ins, output, depth):
    indices_input = ins[0]
    on_value_input = ins[1]
    off_value_input = ins[2]

    tvm_ib = tvm.ir_builder.create()
    # reg[0]:on_value   reg[1]:off_value
    reg = tvm_ib.allocate(output.dtype, (2, ),
                          name="reg_buf",
                          scope=cce.scope_reg)
    reg_for_idx = tvm_ib.allocate(indices_input.dtype, (8, ),
                                  name="reg_buf_for_idx",
                                  scope=cce.scope_reg)

    # allocate ub buffer for on_value and off_value
    on_value_ub = _new_alloc(tvm_ib,
                             output.dtype, (1, ),
                             "on_value_ub",
                             scope=cce.scope_ubuf)
    off_value_ub = _new_alloc(tvm_ib,
                              output.dtype, (1, ),
                              "off_value_ub",
                              scope=cce.scope_ubuf)

    # copy on/off value from gm to ub
    tvm_ib.emit(
        tvm.call_extern(on_value_ub.dtype, "copy_gm_to_ubuf",
                        on_value_ub.access_ptr("w"),
                        on_value_input.access_ptr("r"), 0, 1, 1, 0, 0))
    tvm_ib.emit(
        tvm.call_extern(off_value_ub.dtype, "copy_gm_to_ubuf",
                        off_value_ub.access_ptr("w"),
                        off_value_input.access_ptr("r"), 0, 1, 1, 0, 0))

    # store on/off value to reg_buffer
    tvm_ib.emit(
        tvm.call_extern(reg.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[0]),
                        on_value_ub.access_ptr("r")))
    tvm_ib.emit(
        tvm.call_extern(reg.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[1]),
                        off_value_ub.access_ptr("r")))

    op_args = {}
    op_args['depth'] = depth
    op_args['reg'] = reg
    op_args['reg_for_idx'] = reg_for_idx

    _one_core_ir_not_neg_one(tvm_ib, indices_input, output, op_args)

    return tvm_ib.get()


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def one_hot_compute(indices_input,
                    shape,
                    raw_shape,
                    dtype,
                    depth,
                    on_value_input,
                    off_value_input,
                    axis=-1,
                    indices_dtype="int32"):
    """ TVM calculation process, used for fusion operation.

    Parameters
    ----------
    indices_input: TVM tensor.
        Input indices tensor.
    shape: list or tuple.
        Shape of indices array, only support 1 dimension.
    raw_shape: list or tuple
        The raw shape of indices array.
    dtype: str.
        The data type of output tensor, and only support float16, float32, int32.
    depth: int.
        A scalar defining the depth of the one hot dimension.
    on_value: int or float.
        A scalar defining the value to fill in output when indices[j] = i
        (default: 1).
    off_value: int or float.
        A scalar defining the value to fill in output when indices[j] != i
        (default: 0).
    axis: int.
        The axis to fill, support any axis.
    indices_dtype: str.
        The data type of indices tensor, and only support int32.

    Returns
    -------
    res: TVM tensor.
        The tensor created, and it's shape is [shape[0], depth].
    """
    output_dtype = dtype.lower()
    if axis == 0 and depth == 131072 and output_dtype == "float32" and shape[0] == 128:
        output_shape = (depth, shape[0])
        res = tvm.extern((output_shape, ),
                         (indices_input, on_value_input, off_value_input),
                         lambda ins, outs: _one_hot_ir_not_neg_one(ins, outs[0], depth),
                         dtype=(output_dtype, ),
                         name="res_one_hot")

        return res

    if axis == -1:
        output_shape = (shape[0], depth)
    else:
        output_shape = raw_shape + (depth, 1)

    res = tvm.extern((output_shape, ),
                     (indices_input, on_value_input, off_value_input),
                     lambda ins, outs: one_hot_ir(ins, outs[0], depth, axis),
                     dtype=(output_dtype, ),
                     name="res_one_hot")

    return res

# pylint: disable=too-many-locals,invalid-name
@util.check_input_type(dict, dict, dict, dict, int, int, str, str)
def one_hot_d(input_x,
              input_on_val,
              input_off_val,
              output_y,
              depth,
              axis=-1,
              kernel_name="one_hot_d"):
    """ The locations represented by index in indices array take value on_value,
        while all other locations take value off_value.
        Return a one-hot tensor, and it's shape is [shape[0], depth].

    Parameters
    ----------
    shape: list or tuple.
        Shape of indices array.
    dtype: str.
        The data type of output tensor, and only support float16, float32, int32.
    depth: int.
        A scalar defining the depth of the one hot dimension.
    on_value: int or float.
        A scalar defining the value to fill in output when indices[j] = i
        (default: 1).
    off_value: int or float.
        A scalar defining the value to fill in output when indices[j] != i
        (default: 0).
    axis: int.
        The axis to fill, support any axis.
    dtype: str.
        The data type of indices tensor
    kernel_name: str.
        Cce kernel name, default value is "cce_one_hot".
    need_build: bool.
        If need to build CCEC kernel, default value is False.
    need_print: bool.
        If need to print the ir, default value is False.

    Returns
    -------
    None.
    """
    input_x_shape = input_x.get("shape")
    input_on_val_shape = input_on_val.get("shape")
    util.check_shape_rule(input_x_shape)
    util.check_tensor_shape_size(input_x_shape)
    util.check_shape_rule(input_on_val_shape)
    util.check_tensor_shape_size(input_on_val_shape)
    input_off_val_shape = input_off_val.get("shape")
    util.check_shape_rule(input_off_val_shape)
    util.check_tensor_shape_size(input_off_val_shape)

    indices_dtype = input_x.get("dtype").lower()
    dtype = input_on_val.get("dtype").lower()
    dtype_off = input_off_val.get("dtype").lower()
    util.check_dtype_rule(dtype,
                          ("float16", "float32", "int32", "int8", "uint8"))
    util.check_dtype_rule(dtype_off,
                          ("float16", "float32", "int32", "int8", "uint8"))
    util.check_dtype_rule(indices_dtype, ("int32", "uint8"))
    util.check_kernel_name(kernel_name)

    shape = tuple(input_x.get("shape"))

    if axis == len(shape):
        axis = -1
    if axis != -1:
        axis = util.axis_check(len(shape), axis)

    if (not isinstance(shape[0], int)) or (shape[0] <= 0):
        raise RuntimeError(
            "The type of index must be positive int and value more than 0")

    if depth < 1:
        raise RuntimeError(
            "The depth must be greater or equal to 1, actual input depth is %d"
            % depth)

    # Reshape the indices_shape to 1D in order to simplify the calculation
    indices_reshape = (int(functools_reduce(lambda i, j: i * j, shape)), )

    indices_input = tvm.placeholder(indices_reshape,
                                    name="indices_input",
                                    dtype=indices_dtype.lower())

    on_value_input = tvm.placeholder((1, ), name="on_value_input", dtype=dtype)
    off_value_input = tvm.placeholder((1, ),
                                      name="off_value_input",
                                      dtype=dtype)

    res = one_hot_compute(indices_input, indices_reshape, shape, dtype, depth,
                          on_value_input, off_value_input, axis,
                          indices_dtype.lower())

    if axis == -1 or (axis == 0 and depth == 131072 and
                      dtype == "float32" and indices_reshape[0] == 128):
        sch = tvm.create_schedule([res.op])
        tensor_list = [indices_input, on_value_input, off_value_input, res]
    else:
        sch, tensor_list = _transpose(shape, dtype, depth, axis, res,
                                      indices_input, on_value_input,
                                      off_value_input)

    with build_config:
        tvm.build(sch, tensor_list, "cce", name=kernel_name)

    if 0 <= axis < len(shape):
        _add_workspace(indices_reshape, depth, dtype, kernel_name)
