#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

zn_2_nchw
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from functools import reduce as functools_reduce

from te import platform as cce
from te import tvm
from te.platform.cce_build import build_config
import te.platform.cce_params as cce_params
from topi.cce import util
from impl.five_2_four import _get_ir_branch
from impl.five_2_four import _get_ir_branch_fp16
from impl.five_2_four import _more_dim_ir
from impl.five_2_four import _one_dim_ir
from impl.five_2_four import _split_dim_ir
from impl.five_2_four import _more_dim_ir_fp16
from impl.five_2_four import _split_dim_ir_fp16

# available ub size
UB_SIZE_B = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE)


def _ceil_div(value, block):
    """
    integrate the input value by block

    """
    return (value + block - 1) // block


def _ceil_fill(value, block):
    """
    fill the input value by block

    """
    return _ceil_div(value, block)*block


def _new_alloc(tvm_ib, dtype, shape, name, scope):
    """
    decl new buffer for ir builder make function

    """
    buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name,
                                 scope=scope, data=buf_var)

    return new_buffer


def _write_code(wkspace_dict, fname):
    """
    write workspaces to json file

    """
    fname = os.path.realpath(fname)
    if fname.startswith(os.getcwd()):
        if os.path.exists(fname):
            with open(fname, "r") as f_var:
                load_dict = json.load(f_var)
            load_dict.update(wkspace_dict)
            with open(fname, "w") as f_var:
                json.dump(load_dict, f_var, sort_keys=True,
                          indent=4, separators=(',', ':'))


def _get_factor(ele_zero, ele_cnt, total_ele):
    """
    get split factor for _tilling_one_axis function

    Parameters
    ----------
    ele_zero: int
        the number of shape's first dimension elements
    ele_cnt: int
        the number of all elements
    total_ele: int
        the number of total elements in UB

    Returns
    -------
    split_factor: int
        the factor used when tiling the target axis
    """
    split_factor = 1
    for i in reversed(list(range(1, ele_zero))):
        if i*ele_cnt <= total_ele:
            split_factor = i
            break

    return split_factor


def _get_count_bigger_one(shape):
    """
    get the count of elements bigger than one

    """
    count = 0
    for item in shape:
        if item > 1:
            count += 1

    return count


def _tilling_axis_not_last(shape, dtype):
    """
    calculate the split parameters according to different shapes
    for last axis not changed

    Parameters
    ----------
    shape: tuple
        shape of tensor
    dtype: str
        the data type

    Returns
    -------
    split_axis: int
        the target axis that is used for tiling the tensor to find
    split_factor: int
        the factor used when tiling the target axis
    """
    ub_size_bytes = UB_SIZE_B - 1*1024
    # 8 bit = 1byte, '8' below for this reason
    dtype_bytes_size = cce.cce_intrin.get_bit_len(dtype) // 8
    # 32 means one block size(32 Bytes),
    # divide by 32 to get the numbers of data that can be stored in one block.
    flag = 32 // dtype_bytes_size
    element_new = ((shape[-1] + flag - 1) // flag)*flag
    shape_new = []
    for i in shape:
        shape_new.append(i)
    shape_new[-1] = int(element_new)
    total_ele = ub_size_bytes // dtype_bytes_size
    split_axis = 0
    split_factor = 1

    for i, item in enumerate(shape):
        ele_cnt = functools_reduce(lambda x, y: x*y, shape_new[i:])
        if ele_cnt <= total_ele:
            split_axis = i - 1
            split_factor = total_ele // ele_cnt
            break
        if i == len(shape) - 1:
            if len(shape) == 1:
                split_axis = 0
                split_factor = _get_factor(shape[0], 1, total_ele)
            else:
                split_axis = i
                split_factor = total_ele
            break

    if split_axis < 0:
        split_axis = 0
        split_factor = shape[0]

    if shape[len(shape) - 1] == 1:
        count = _get_count_bigger_one(shape)
        if count > 1:
            for i, item in enumerate(reversed(shape)):
                if item > 1:
                    flag = i
                    break
            if split_axis != len(shape) - flag - 1:
                split_axis = len(shape) - flag - 1
                split_factor = shape[split_axis]

    return split_axis, split_factor


def _tilling_axis_multi_core_fuse(shape, dtype):
    """
    calculate the split parameters according to different shapes
    for multi core scene

    Parameters
    ----------
    shape: tuple
        shape of tensor
    dtype: str
        the data type

    Returns
    -------
    split_axis: int
        the target axis that is used for tiling the tensor to find
    split_factor: int
        the factor used when tiling the target axis
    """
    ub_size_bytes = UB_SIZE_B
    dtype_bytes_size = cce.cce_intrin.get_bit_len(dtype) // 8
    total_ele = ub_size_bytes // dtype_bytes_size
    split_axis = 0
    split_factor = 1

    for i, _ in enumerate(shape):
        ele_cnt = functools_reduce(lambda x, y: x*y, shape[i:])
        if ele_cnt <= total_ele:
            split_axis = i - 1
            split_factor = total_ele // ele_cnt
            if split_axis >= 0:
                for j in reversed(list(range(1, split_factor))):
                    if shape[split_axis] % j == 0:
                        split_factor = j
                        break
            break
        if i == len(shape) - 1:
            split_axis = i
            split_factor = 1
            for k in reversed(list(range(1, total_ele))):
                if shape[split_axis] % k == 0:
                    split_factor = k

    if split_axis < 0:
        split_axis = 0
        split_factor = shape[0]

    return split_axis, split_factor


def _get_align_axis(out_shape):
    """
    get the axis_info when applying the align

    """
    flag = -1
    if out_shape[-1] != 1:
        axis = len(out_shape) - 2
    else:
        for i, item in enumerate(reversed(out_shape)):
            if item > 1:
                flag = i
                break
        if flag in (-1, 0):
            axis = 0
        else:
            axis = len(out_shape) - flag - 1

    return axis


def _do_storage_align(sch, data_ub, shape_res, dtype):
    """
    the function to do storage align

    """
    if len(shape_res) >= 2:
        do_align = True
        if shape_res[len(shape_res) - 1] == 1:
            count = 0
            for i in shape_res:
                if i > 1:
                    count += 1
            if count == 1:
                do_align = False
        if do_align:
            dtype_bytes_size = cce.cce_intrin.get_bit_len(dtype) // 8
            # 32 means one block size(32 Bytes), divide by 32 to
            # get the numbers of data that
            # can be stored in one block.
            element = 32 // dtype_bytes_size
            align_axis = _get_align_axis(shape_res)
            sch[data_ub].storage_align(data_ub.op.axis[align_axis], element, 0)

    return sch


def _get_fused_index(fused_axis_value, origin_list_copy, split_axis):
    """
    get fused index for not change last schedule

    """
    fused_index_list = []
    for target in fused_axis_value:
        for i, item in enumerate(origin_list_copy):
            if target == item:
                fused_index_list.append(i)
                origin_list_copy[i] = -1
                break

    is_fuse_outer = False
    for i in fused_index_list:
        if i == split_axis:
            is_fuse_outer = True
            fused_index_list.remove(split_axis)
            break

    return fused_index_list, origin_list_copy, is_fuse_outer


def _get_after_reorder_axis(reorder_axis_index, split_axis):
    """
    get after reorder axis for not change last schedule

    """
    after_reorder_axis_index = []
    for i in list(range(split_axis)):
        if i not in reorder_axis_index:
            after_reorder_axis_index.append(i)

    return after_reorder_axis_index


def _do_compute_at(args):
    """
    compute at for not change last schedule

    """
    reorder_axis_index, sch, res, fused_axis, split_axis, data_ub = args
    if reorder_axis_index:
        fused_index_last = reorder_axis_index[-1]
        if fused_index_last == split_axis - 1:
            sch[data_ub].compute_at(sch[res], fused_axis)
        else:
            sch[data_ub].compute_at(sch[res], res.op.axis[split_axis - 1])
    else:
        sch[data_ub].compute_at(sch[res], fused_axis)

    return sch


def _schedule_for_not_change_last(args):
    """
    the function to bind multi core

    """
    sch, data, res, data_ub, shape_res, dtype = args
    sch[data_ub].set_scope(cce.scope_ubuf)

    split_axis, split_factor = _tilling_axis_not_last(shape_res, dtype)
    axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis],
                                            factor=split_factor)
    sch[data_ub].compute_at(sch[res], axis_outer)

    sch[data_ub].emit_insn(data_ub.op.axis[split_axis], "dma_copy")
    sch = _do_storage_align(sch, data_ub, shape_res, dtype)
    sch[res].emit_insn(axis_inner, "dma_copy")
    tensor_list = [data, res]

    return sch, tensor_list


def _perm_to_flag(perm):
    """
    get the flag for permutation according to perm

    """
    flag = [i for i in perm]
    for i, item in enumerate(perm):
        flag[item] = i

    return flag


# pylint: disable = locally-disabled,too-many-arguments,too-many-locals
# pylint: disable = locally-disabled,unnecessary-lambda
def _tranpose_notchange_last(data, shape_5hd, dst_shape, perm, dtype,
                             max_dim, shape_all):
    """
    permutes the dimensions and the last axis is not transposed

    Parameters
    ----------
    data: tvm.tensor
        tensor of input data
    shape_res: list or tuple
        shape of output tensor
    perm: list or tuple
        permutation of the dimension of tensor
    dtype: str
        the data type

    Returns
    -------
    sch: tvm.schedule
        the compute schedule
    tensor_list: list
        list of tensor
    """
    def _permute(*index):
        """
        function of permute the dimensions of data

        """
        for i, item in enumerate(_perm_to_flag(perm)):
            if i == 0:
                res_axis = (index[item],)
            else:
                res_axis = res_axis + (index[item],)

        return res_axis

    # c1hwnc0 to nc1hwc0
    data_ub = tvm.compute(shape_5hd, lambda *index: data(*_permute(*index)),
                          name="data_ub")
    res_5hd = tvm.compute(shape_5hd, lambda *index: data_ub(*index),
                          name="res_5hd")

    # nc1hwc0 to nchw
    if dtype == "float32":
        branch = _get_ir_branch(shape_5hd, dtype, shape_all)
        if branch == "more_dim":
            res = tvm.extern(dst_shape, [res_5hd],
                             lambda ins, outs: _more_dim_ir(outs[0],
                                                            ins[0],
                                                            max_dim,
                                                            shape_all),
                             name="res", dtype=dtype)
        elif branch == "one_dim":
            res = tvm.extern(dst_shape, [res_5hd],
                             lambda ins, outs: _one_dim_ir(outs[0],
                                                           ins[0],
                                                           max_dim,
                                                           shape_all),
                             name="res", dtype=dtype)
        else:
            res = tvm.extern(dst_shape, [res_5hd],
                             lambda ins, outs: _split_dim_ir(outs[0],
                                                             ins[0],
                                                             max_dim,
                                                             shape_all),
                             name="res", dtype=dtype)
    else:
        branch_fp16 = _get_ir_branch_fp16(dst_shape, dtype, shape_all)
        if branch_fp16 == "more_dim_fp16":
            res = tvm.extern(dst_shape, [res_5hd],
                             lambda ins, outs: _more_dim_ir_fp16(outs[0],
                                                                 ins[0],
                                                                 max_dim,
                                                                 shape_all),
                             name="res", dtype=dtype)
        else:
            res = tvm.extern(dst_shape, [res_5hd],
                             lambda ins, outs: _split_dim_ir_fp16(
                                 outs[0], ins[0], max_dim, shape_all),
                             name="res", dtype=dtype)

    sch = tvm.create_schedule(res.op)
    args = [sch, data, res_5hd, data_ub, shape_5hd, dtype]
    sch, _ = _schedule_for_not_change_last(args)
    tensor_list = [data, res, res_5hd]

    return sch, tensor_list


def _temp_ir(dst, data):
    tvm_ib = tvm.ir_builder.create()
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size

    n_i, c_i, h_i, w_i = dst.shape

    ub_bytes = UB_SIZE_B
    ub_ele = ub_bytes // float_size
    shape_ele = n_i*c_i*h_i*w_i

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)

    with tvm_ib.if_scope(shape_ele <= ub_ele):
        burst_len = _ceil_div(shape_ele, cp_align_len)
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr("w", offset=0),
                                    data.access_ptr('r', offset=0),
                                    0, 1, burst_len, 0, 0))
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w', offset=0),
                                    data_ub.access_ptr("r", offset=0),
                                    0, 1, burst_len, 0, 0))

    with tvm_ib.if_scope(shape_ele > ub_ele):
        loop = shape_ele // ub_ele
        mod = shape_ele % ub_ele
        with tvm_ib.for_range(0, loop, name="num_p") as num_p:
            burst_len = _ceil_div(ub_ele, cp_align_len)
            tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                        data_ub.access_ptr("w", offset=0),
                                        data.access_ptr('r', offset=num_p*ub_ele),
                                        0, 1, burst_len, 0, 0))
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w', offset=num_p*ub_ele),
                                        data_ub.access_ptr("r", offset=0),
                                        0, 1, burst_len, 0, 0))
        with tvm_ib.if_scope(mod > 0):
            burst_len = _ceil_div(mod, cp_align_len)
            tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                        data_ub.access_ptr("w", offset=0),
                                        data.access_ptr('r', offset=loop*ub_ele),
                                        0, 1, burst_len, 0, 0))
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w', offset=loop*ub_ele),
                                        data_ub.access_ptr("r", offset=0),
                                        0, 1, burst_len, 0, 0))

    return tvm_ib.get()


def _tranpose_notchange_last_two(data, shape_5hd, dst_shape_full, dst_shape,
                                 perm, dtype, max_dim, shape_all):
    """
    permutes the dimensions and the last axis is not transposed

    Parameters
    ----------
    data: tvm.tensor
        tensor of input data
    shape_res: list or tuple
        shape of output tensor
    perm: list or tuple
        permutation of the dimension of tensor
    dtype: str
        the data type

    Returns
    -------
    sch: tvm.schedule
        the compute schedule
    tensor_list: list
        list of tensor
    """
    def _permute(*index):
        """
        function of permute the dimensions of data

        """
        for i, item in enumerate(_perm_to_flag(perm)):
            if i == 0:
                res_axis = (index[item],)
            else:
                res_axis = res_axis + (index[item],)

        return res_axis

    # c1hwnc0 to nc1hwc0
    data_ub = tvm.compute(shape_5hd, lambda *index: data(*_permute(*index)),
                          name="data_ub")
    res_5hd = tvm.compute(shape_5hd, lambda *index: data_ub(*index),
                          name="res_5hd")

    # nc1hwc0 to nchw
    if dtype == "float32":
        branch = _get_ir_branch(shape_5hd, dtype, shape_all)
        if branch == "more_dim":
            res = tvm.extern(dst_shape_full, [res_5hd],
                             lambda ins, outs: _more_dim_ir(outs[0],
                                                            ins[0],
                                                            max_dim,
                                                            shape_all),
                             name="res", dtype=dtype)
        elif branch == "one_dim":
            res = tvm.extern(dst_shape_full, [res_5hd],
                             lambda ins, outs: _one_dim_ir(outs[0],
                                                           ins[0],
                                                           max_dim,
                                                           shape_all),
                             name="res", dtype=dtype)
        else:
            res = tvm.extern(dst_shape_full, [res_5hd],
                             lambda ins, outs: _split_dim_ir(outs[0],
                                                             ins[0],
                                                             max_dim,
                                                             shape_all),
                             name="res", dtype=dtype)
    else:
        branch_fp16 = _get_ir_branch_fp16(dst_shape_full, dtype, shape_all)
        if branch_fp16 == "more_dim_fp16":
            res = tvm.extern(dst_shape_full, [res_5hd],
                             lambda ins, outs: _more_dim_ir_fp16(outs[0],
                                                                 ins[0],
                                                                 max_dim,
                                                                 shape_all),
                             name="res", dtype=dtype)
        else:
            res = tvm.extern(dst_shape_full, [res_5hd],
                             lambda ins, outs: _split_dim_ir_fp16(
                                 outs[0], ins[0], max_dim, shape_all),
                             name="res", dtype=dtype)

    res_end = tvm.extern(dst_shape, [res],
                         lambda ins, outs: _temp_ir(outs[0], ins[0]),
                         name="res_end", dtype=dtype)

    sch = tvm.create_schedule(res_end.op)
    args = [sch, data, res_5hd, data_ub, shape_5hd, dtype]
    sch, _ = _schedule_for_not_change_last(args)
    tensor_list = [data, res_end, res_5hd, res]

    return sch, tensor_list


def _special_ir(dst, data):
    tvm_ib = tvm.ir_builder.create()
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size

    n_i, c_i, _, _ = dst.shape
    c_0 = 16
    n_true = _ceil_fill(n_i, c_0)
    ub_max = 3968*16

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_max,
                         "data_ub", scope=cce.scope_ubuf)

    loop = c_i // c_0

    with tvm_ib.for_range(0, loop, name="n_loop") as n_loop:
        data_offset = n_loop*c_0*n_true
        burst_len = n_i * c_0 // cp_align_len
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr("w", offset=0),
                                    data.access_ptr('r', offset=data_offset),
                                    0, 1, burst_len, 0, 0))
        dst_offset = n_loop*c_0
        burst_len_data = c_0 // cp_align_len
        dst_stride = (c_i - c_0) // cp_align_len
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w', offset=dst_offset),
                                    data_ub.access_ptr("r", offset=0),
                                    0, n_i, burst_len_data, 0, dst_stride))

    return tvm_ib.get()


def _check_parameters(src, dst, src_format, dst_format, kernel_name):
    """
    check the parameters including src_shape, dst_shape,
    src_format, dst_format, dtype and kernel_name

    """
    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype")
    dtype_dst = dst.get("dtype")

    if src_format.lower() != "fractal_zn" and src_format.lower() != "fractal_z":
        raise RuntimeError("src_format must be FRACTAL_Zn !")

    if dst_format.lower() != "nchw":
        raise RuntimeError("dst_format must be NCHW !")

    util.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype, check_list)
    if dtype != dtype_dst:
        raise RuntimeError("dtype of src and dst are different !")

    util.check_shape_rule(src_shape, 4, 4)
    util.check_shape_rule(dst_shape, 4, 4)
    util.check_tensor_shape_size(src_shape)
    util.check_tensor_shape_size(dst_shape)

    if src_shape[2] != 16 or src_shape[3] != 16:
        raise RuntimeError(
            "ni and c0 must be 16 !")

    n_i, c_i, h_i, w_i = dst_shape

    c_0 = 16
    c_1 = _ceil_div(c_i, c_0)
    src_one = c_1*h_i*w_i
    src_two = _ceil_div(n_i, 16)

    if list(src_shape) != [src_one, src_two, 16, 16]:
        raise RuntimeError("src_shape is wrong !")


def _check_parameters_special(src, dst, src_format, dst_format, kernel_name):
    """
    check the parameters including src_shape, dst_shape,
    src_format, dst_format, dtype and kernel_name

    """
    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype")
    dtype_dst = dst.get("dtype")

    if src_format.lower() != "fractal_zn" and src_format.lower() != "fractal_z":
        raise RuntimeError("src_format must be FRACTAL_Zn !")

    if dst_format.lower() != "nchw":
        raise RuntimeError("dst_format must be NCHW !")

    util.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype, check_list)
    if dtype != dtype_dst:
        raise RuntimeError("dtype of src and dst are different !")

    util.check_shape_rule(src_shape, 4, 4)
    util.check_shape_rule(dst_shape, 4, 4)
    util.check_tensor_shape_size(src_shape)
    util.check_tensor_shape_size(dst_shape)

    if src_shape[2] != 16 or src_shape[3] != 16:
        raise RuntimeError(
            "ni and c0 must be 16 !")

    n_i, c_i, h_i, w_i = dst_shape

    c_0 = 16
    c_1 = _ceil_div(c_i, c_0)
    src_one = c_1*h_i*w_i
    src_two = _ceil_div(n_i, 16)

    if list(src_shape) != [src_one, src_two, 16, 16]:
        raise RuntimeError("src_shape is wrong !")


# pylint: disable=locally-disabled,too-many-statements
@util.check_input_type(dict, dict, str, str, str)
def zn_2_nchw(src, dst, src_format, dst_format, kernel_name='zn_2_nchw'):
    """
    algorithm: zn_2_nchw
    calculating: change data format from Zn to NCHW

    Parameters
    ----------
    src: dict
        contains shape and dtype information of input tensor
    dst: dict
        contains shape and dtype information of output tensor
    src_format: str
        represents the format of input tensor, only support "Zn"
    dst_format: str
        represents the format of output tensor, only support "NCHW"
    kernel_name: str
        cce kernel name, default value is "zn_2_nchw"

    Returns
    -------
    None
    """
    dst_shape = dst.get("shape")
    dtype = src.get("dtype")
    n_i, c_i, h_i, w_i = dst_shape

    if n_i % 16 == 0:
        _check_parameters(src, dst, src_format, dst_format, kernel_name)
        c_0 = 16
        if dtype == "int8":
            c_0 = 32
        c_1 = _ceil_div(c_i, c_0)

        shape_zn = [c_1, h_i, w_i, n_i, c_0]
        shape_5hd = [n_i, c_1, h_i, w_i, c_0]

        data = tvm.placeholder(shape_zn, dtype=dtype, name="data")

        max_dim = max(dst_shape)
        shape_all = functools_reduce(lambda x, y: x * y, shape_5hd[:])
        perm = [3, 0, 1, 2, 4]
        sch, tensor_list = _tranpose_notchange_last(data, shape_5hd,
                                                    dst_shape,
                                                    perm, dtype,
                                                    max_dim, shape_all)

        with build_config:
            tvm.build(sch, tensor_list, "cce", name=kernel_name)

        float_size = cce.cce_intrin.get_bit_len(dtype) // 8
        size = 1
        for item in shape_5hd:
            size *= item
        total_size = [size * float_size]
        workspace_dict = {"workspace": {"num": 1, "size": total_size}}
        _write_code(workspace_dict, "kernel_meta/" + kernel_name + ".json")
    else:
        _check_parameters(src, dst, src_format, dst_format, kernel_name)
        c_0 = 16
        if dtype == "int8":
            c_0 = 32
        c_1 = _ceil_div(c_i, c_0)

        n_ni = 16
        n_true = _ceil_fill(n_i, n_ni)
        shape_zn = [c_1, h_i, w_i, n_true, c_0]
        shape_5hd = [n_true, c_1, h_i, w_i, c_0]

        data = tvm.placeholder(shape_zn, dtype=dtype, name="data")

        max_dim = max(dst_shape)
        shape_all = functools_reduce(lambda x, y: x * y, shape_5hd[:])
        perm = [3, 0, 1, 2, 4]
        dst_shape_full = [n_true, c_i, h_i, w_i]
        sch, tensor_list = _tranpose_notchange_last_two(data, shape_5hd,
                                                        dst_shape_full,
                                                        dst_shape,
                                                        perm, dtype,
                                                        max_dim, shape_all)

        with build_config:
            tvm.build(sch, tensor_list, "cce", name=kernel_name)

        float_size = cce.cce_intrin.get_bit_len(dtype) // 8
        size = 2
        for item in shape_5hd:
            size *= item
        total_size = [size * float_size, size * float_size]
        workspace_dict = {"workspace": {"num": 2, "size": total_size}}
        _write_code(workspace_dict, "kernel_meta/" + kernel_name + ".json")
