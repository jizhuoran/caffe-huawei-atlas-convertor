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

unpack
"""

from functools import reduce as function_reduce

import te.lang.cce
from te import tvm
from te import platform as tbe_platform
from te.platform import cce_params
from te.platform import insn_cmd
from te.platform.cce_build import build_config
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
from impl.split_d import split_d
from impl.copy_only import copy_only

SHAPE_SIZE_LIMIT = 2**30


def _check_params(shape, num, axis, format, dtype, kernel_name):
    """
    check the parameters including shape, num, axis, format, dtype, kernel_name

    Parameters
    ----------
    shape: tuple
        the shape of tensor.
    num: int
        the length of the dim axis.
    axis: int.
        the axis of unapck.
    format: str.
        the data format of input.
    dtype: str
        the data type.
    kernel_name: str
        cce kernel name.

    Returns
    -------
    None
    """
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)

    # check format
    format_list = ("ND", "NHWC", "NCHW", "HWCN", "NC1HWC0")
    if format == "NC1HWC0":
        if len(shape) != 5:
            raise RuntimeError("NC1HWC0 shape length should be equal to 5")
        # index list of H W axis.
        suporrted_list = (-5, -3, -2, 0, 2, 3)
        if axis not in suporrted_list:
            raise RuntimeError("NC1HWC0 supports unpack of N H W axis")
    else:
        if format not in format_list:
            raise RuntimeError(
                            "Format supports ND,NCHW,NHWC,HWCN and NC1HWC0")

    # check axis value
    if axis < -len(shape) or axis >= len(shape):
        raise RuntimeError("axis= %d not in the union of [%d, %d) " %
                           (axis, -len(shape), len(shape)))

    # check num value
    if num is None:
        num = shape[axis]
    if num is None:
        raise RuntimeError("Cannot infer num value from shape.")
    if num != shape[axis]:
        raise RuntimeError("Num should be equal to length of dim axis %d" %
                           shape[axis])

    # 1536B means stack holding the param provided to the platform,
    # 1 param takes 8 bytes, needs Multiple output param and 1 input param
    # mini has more parameters (offset, index) than cloud
    compile_plat = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if compile_plat in ("Ascend310",):
        if num > (1536 // 3) // 8 - 1:
            raise RuntimeError(
                "Exceeds stack holding the parameters for mini")
    else:
        if num > 1536 // 8 - 1:
            raise RuntimeError(
                "Exceeds stack holding the parameters for cloud")

    check_list = ("int8", "int16", "int32", "int64", "uint8", "uint16",
                  "uint32", "uint64", "float16", "float32")
    util.check_dtype_rule(dtype, check_list)

    util.check_kernel_name(kernel_name)


def _get_public_param(dtype):
    """
    get public parameters

    Parameters
    ----------
    dtype: str
        the data type.

    Returns
    -------
    total_ele: int
        the size of the  data for UB to move at a time.
    ele_each_block: int
        the numbers of data in one block.
    device_core_num: int
        the numbers of blockdim.
    """
    ub_size_bytes = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) // 2
    # Convert bits to Bytes
    dtype_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
    # gm->ub maximum copy data at a time
    total_ele = ub_size_bytes // dtype_size

    # 32 means one block size(32 Bytes), divide by 32 to get the numbers
    # of data that can be stored in one block.
    one_block_bytes_size = cce_params.VECTOR_INST_BLOCK_WIDTH // \
                           cce_params.VECTOR_INST_BLOCK_NUM
    ele_each_block = one_block_bytes_size // dtype_size

    # get core num according to the product
    device_core_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

    return total_ele, ele_each_block, device_core_num


def check_use_special_optimize(dtype, afterdim, flag = False):
    """Function: use to optimize special scene.
    """
    is_dtype_support = dtype in ("float16", "float32")
    is_shape_support = (afterdim in (8, 16, 32, 64) or afterdim < 8)

    _, ele_each_block, _ = _get_public_param(dtype)
    if afterdim < ele_each_block:
        if dtype in ("int8", "uint8") and not (afterdim % 8):
            dtype = "uint64"
            afterdim = afterdim // 8
            flag = True
        if dtype in ("float16", "int16", "uint16") and not (afterdim % 4):
            dtype = "uint64"
            afterdim = afterdim // 4
            flag = True
        if dtype in ("float32", "int32", "uint32") and not (afterdim % 2):
            dtype = "uint64"
            afterdim = afterdim // 2
            flag = True

    return dtype, afterdim, is_dtype_support and is_shape_support and not flag


def _index_offset(shape, axis, offset, *index):
    """Compute the offset of index along one dimension.

    Parameters
    ----------
    shape: list
        the shape of tensor.
    axis: int
        the dimension along which to unpack.
    offset: int
        the offset of axis.
    index: list or tuple
        index value tuple.

    Returns
    -------
    output_index: tuple
        output index with one input index value add axis offset.
    """
    input_index = list(index)
    output_index = ()
    for i, _ in enumerate(shape):
        if i == axis:
            input_index[i] = input_index[i] + offset
        output_index += (input_index[i], )

    return output_index


def _tiling_axis(shape, dtype):
    """
    Calculate the tile parameters.

    Parameters
    ----------
    shape: list or tuple
        the shape of tensor.
    dtype: str
        the dtype of tensor.

    Returns
    -------
    split_axis: int
        the target axis that is used for tile the tensor.
    split_factor: int
        the factor used when tile the target axis.
    """
    total_ele, ele_each_block, _ = _get_public_param(dtype)

    tiling_shape = [dim for dim in shape]
    if shape[-1] % ele_each_block != 0:
        last_ele = ((shape[-1] + ele_each_block - 1) //
                    ele_each_block) * ele_each_block
        tiling_shape[-1] = int(last_ele)

    split_axis = 0
    split_factor = 1
    for index, _ in enumerate(tiling_shape):
        ele_cnt = function_reduce(lambda x, y: x * y, tiling_shape[index:])
        if ele_cnt <= total_ele:
            split_axis = index - 1
            split_factor = total_ele // ele_cnt
            break
        elif index == len(tiling_shape) - 1:
            split_axis = index
            split_factor = total_ele
            break

    if split_axis < 0:
        split_axis = 0
        split_factor = tiling_shape[0]

    return split_axis, split_factor


@fusion_manager.register("unpack")
def _unpack_compute_scalar(input_place, y, num, axis, kernel_name="unpack"):
    """
    unpack a tensor into `num` tensors along axis dimension.

    Parameters
    ----------
    input_place: TVM tensor
        the tensor of input.
    y: tuple or list
        the list of output tensor.
    num : int.
        the length of the dim axis.
    axis: int.
        the axis to unpack along.
    kernel_name : str.
        cce kernel name, default value is "unpack".

    Returns
    -------
    gm2ub_tensor: TVM tensor
        the tensors of gm2ub, tensor type is TVM tensor.
    ub2ub_tensor_list: list
        the list of ub2ub tensors, tensor type is TVM tensor.
    ub2gm_tensor_list: list
        the list of ub2gm tensors, tensor type is TVM tensor.
    virtual_node:
        the tensors of virtual output node, tensor type is TVM tensor.
    """
    input_shape = te.lang.cce.util.shape_to_list(input_place.shape)

    gm2ub_tensor = tvm.compute(input_shape,
                               lambda *index: input_place(*index),
                               name="gm2ub_tensor")

    output_shape = input_shape
    for index, _ in enumerate(output_shape):
        output_shape[index] = output_shape[index] if index != axis else 1

    offset = 0
    ub2ub_tensor_list = []
    ub2gm_tensor_list = []
    for i in range(num):
        ub2ub_tensor = tvm.compute(output_shape,
                                   lambda *index: gm2ub_tensor(*_index_offset(
                                       output_shape, axis, offset, *index)),
                                   name=''.join(['tensor', str(i)]))
        ub2ub_tensor_list.append(ub2ub_tensor)

        ub2gm_tensor = tvm.compute(output_shape,
                                   lambda *index: ub2ub_tensor(*index),
                                   name=''.join(['res', str(i)]))
        ub2gm_tensor_list.append(ub2gm_tensor)

        offset = offset + output_shape[axis]

    # create a virtual node
    def _add_compute(*index):
        virtual_tensor = ub2gm_tensor_list[0](*index)
        for ub2gm_tensor in ub2gm_tensor_list[1:]:
            virtual_tensor += ub2gm_tensor(*index)
        return virtual_tensor

    virtual_node = tvm.compute(output_shape,
                               lambda *index: _add_compute(*index),
                               name="virtual_node")

    return gm2ub_tensor, ub2ub_tensor_list, ub2gm_tensor_list, virtual_node


@fusion_manager.register("unpack")
def _unpack_compute_copy(input_place, y, num, axis, kernel_name="unpack"):
    """
    unpack a tensor into `num` tensors along axis dimension.

    Parameters
    ----------
    input_place: TVM tensor
        the tensor of input.
    y: tuple or list
        the list of output tensor.
    num : int.
        the length of the dim axis.
    axis: int.
        the axis to unpack along.
    kernel_name : str.
        cce kernel name, default value is "unpack".

    Returns
    -------
    gm2ub_tensor_list: list
        the list of gm2ub tensors, tensor type is TVM tensor.
    ub2gm_tensor_list: list
        the list of ub2gm tensors, tensor type is TVM tensor.
    virtual_node:
        the tensors of virtual output node, tensor type is TVM tensor.
    """
    input_shape = te.lang.cce.util.shape_to_list(input_place.shape)
    output_shape = input_shape
    for index, _ in enumerate(output_shape):
        output_shape[index] = output_shape[index] if index != axis else 1

    offset = 0
    gm2ub_tensor_list = []
    ub2gm_tensor_list = []
    for i in range(num):
        gm2ub_tensor = tvm.compute(output_shape,
                                   lambda *index: input_place(*_index_offset(
                                       output_shape, axis, offset, *index)),
                                   name=''.join(['tensor', str(i)]))
        gm2ub_tensor_list.append(gm2ub_tensor)

        ub2gm_tensor = tvm.compute(output_shape,
                                   lambda *index: gm2ub_tensor(*index),
                                   name=''.join(['res', str(i)]))
        ub2gm_tensor_list.append(ub2gm_tensor)

        offset = offset + output_shape[axis]

    # create a virtual node
    def _add_compute(*index):
        virtual_tensor = ub2gm_tensor_list[0](*index)
        for ub2gm_tensor in ub2gm_tensor_list[1:]:
            virtual_tensor += ub2gm_tensor(*index)
        return virtual_tensor

    virtual_node = tvm.compute(output_shape,
                               lambda *index: _add_compute(*index),
                               name="virtual_node")

    return gm2ub_tensor_list, ub2gm_tensor_list, virtual_node


def _unpack_schedule(input_place, output_shape, y, num, axis, dtype):
    """Create unpack schedule.

    Parameters
    ----------
    input_place: TVM tensor
        the tensor of input.
    output_shape: tuple or list
        the shape of output tensor.
    y: tuple or list
        the list of output tensor.
    num : int.
        the length of the dim axis.
    axis: int.
        the axis to unpack along.
    dtype: str.
        the dtype of input.

    Returns
    -------
    sch: schedule
        the created schedule.
    build_list: list
        the list of input and output tensors, tensor type is TVM tensor.
    """
    _, ele_each_block, device_core_num = _get_public_param(dtype)
    befordim, afterdim = output_shape[0], output_shape[-1]
    block_idx = tvm.thread_axis('blockIdx.x')

    # can open multi-core scene
    if befordim >= ele_each_block and afterdim < ele_each_block:
        befordim_in = ele_each_block // afterdim + 1
        befordim_out = (befordim + befordim_in - 1) // befordim_in
        while (befordim + befordim_out -
               1) // befordim_out * afterdim < ele_each_block:
            befordim_out -= 1
        if befordim_out >= device_core_num:
            befordim_out = device_core_num
        afterdim_in = afterdim

        gm2ub_tensor, ub2ub_tensor_list, ub2gm_tensor_list, virtual_node = _unpack_compute_scalar(
            input_place, y, num, axis)

        res_op = []
        build_list = [input_place]
        for ub2gm_tensor in ub2gm_tensor_list:
            res_op.append(ub2gm_tensor.op)
            build_list.append(ub2gm_tensor)

        sch = tvm.create_schedule(virtual_node.op)
        sch[gm2ub_tensor].set_scope(tbe_platform.scope_ubuf)
        for tensor in ub2ub_tensor_list:
            sch[tensor].set_scope(tbe_platform.scope_ubuf)

        befordim_outer, befordim_inner = sch[virtual_node].split(
            virtual_node.op.axis[0], nparts=befordim_out)
        afterdim_outer, afterdim_inner = sch[virtual_node].split(
            virtual_node.op.axis[2], factor=afterdim_in)

        sch[virtual_node].reorder(befordim_outer, afterdim_outer,
                                  befordim_inner, afterdim_inner)
        fused_axis = sch[virtual_node].fuse(befordim_outer, afterdim_outer)
        sch[virtual_node].bind(fused_axis, block_idx)

        new_shape = ((befordim + befordim_out - 1) // befordim_out, num,
                     afterdim_in)
        split_axis, split_factor = _tiling_axis(new_shape, dtype)
        if split_axis == 0:
            axis_outer, axis_inner = sch[virtual_node].split(
                befordim_inner, factor=split_factor)
        else:
            axis_outer, axis_inner = sch[virtual_node].split(
                afterdim_inner, factor=split_factor)

        sch[gm2ub_tensor].compute_at(sch[virtual_node], axis_outer)
        sch[gm2ub_tensor].emit_insn(gm2ub_tensor.op.axis[split_axis],
                                    insn_cmd.DMA_COPY)

        for i in range(num):
            sch[ub2gm_tensor_list[i]].compute_at(sch[virtual_node], axis_outer)
            sch[ub2ub_tensor_list[i]].compute_at(sch[virtual_node], axis_outer)

            sch[ub2ub_tensor_list[i]].emit_insn(
                ub2ub_tensor_list[i].op.axis[split_axis], insn_cmd.DATA_MOV)
            sch[ub2gm_tensor_list[i]].emit_insn(
                ub2gm_tensor_list[i].op.axis[split_axis], insn_cmd.DMA_COPY)

        sch[virtual_node].emit_insn(axis_inner, insn_cmd.PHONY_INSN)

    else:
        gm2ub_tensor_list, ub2gm_tensor_list, virtual_node = _unpack_compute_copy(
            input_place, y, num, axis)
        res_op = []
        build_list = [input_place]
        for ub2gm_tensor in ub2gm_tensor_list:
            res_op.append(ub2gm_tensor.op)
            build_list.append(ub2gm_tensor)

        sch = tvm.create_schedule(virtual_node.op)
        for tensor in gm2ub_tensor_list:
            sch[tensor].set_scope(tbe_platform.scope_ubuf)

        # can open multi-core scene
        if afterdim >= ele_each_block:
            if befordim >= device_core_num:
                befordim_out = device_core_num
                afterdim_in = afterdim
            elif befordim == 1:
                befordim_out = befordim
                afterdim_in = (afterdim + device_core_num - 1) // device_core_num
            else:
                afterdim_outer = device_core_num // befordim
                afterdim_in = (afterdim + afterdim_outer - 1) // afterdim_outer
                while afterdim % afterdim_in < ele_each_block:
                    afterdim_in += 1
                befordim_out = befordim

            befordim_outer, befordim_inner = sch[virtual_node].split(
                virtual_node.op.axis[0], nparts=befordim_out)
            afterdim_outer, afterdim_inner = sch[virtual_node].split(
                virtual_node.op.axis[2], factor=afterdim_in)

            sch[virtual_node].reorder(befordim_outer, afterdim_outer,
                                      befordim_inner, afterdim_inner)
            fused_axis = sch[virtual_node].fuse(befordim_outer, afterdim_outer)
            sch[virtual_node].bind(fused_axis, block_idx)

            new_shape = ((befordim + befordim_out - 1) // befordim_out, 1,
                         afterdim_in)
            split_axis, split_factor = _tiling_axis(new_shape, dtype)
            if split_axis == 0:
                axis_outer, axis_inner = sch[virtual_node].split(
                    befordim_inner, factor=split_factor)
            else:
                axis_outer, axis_inner = sch[virtual_node].split(
                    afterdim_inner, factor=split_factor)
        else:
            split_axis, split_factor = _tiling_axis(output_shape, dtype)
            axis_outer, axis_inner = sch[virtual_node].split(
                virtual_node.op.axis[split_axis], factor=split_factor)

        for i in range(num):
            storage_axis = split_axis - 1 if split_axis != 0 else 0
            sch[gm2ub_tensor_list[i]].storage_align(
                gm2ub_tensor_list[i].op.axis[storage_axis], ele_each_block, 0)

            sch[gm2ub_tensor_list[i]].double_buffer()
            sch[gm2ub_tensor_list[i]].compute_at(sch[virtual_node], axis_outer)
            sch[ub2gm_tensor_list[i]].compute_at(sch[virtual_node], axis_outer)

            sch[gm2ub_tensor_list[i]].emit_insn(
                gm2ub_tensor_list[i].op.axis[split_axis], insn_cmd.DMA_COPY)
            sch[ub2gm_tensor_list[i]].emit_insn(
                ub2gm_tensor_list[i].op.axis[split_axis], insn_cmd.DMA_COPY)

        sch[virtual_node].emit_insn(axis_inner, insn_cmd.PHONY_INSN)

    return sch, build_list


@util.check_input_type(dict, (tuple, list), (int, type(None)), int, str)
def unpack(x, y, num=None, axis=0, kernel_name="unpack"):
    """
    unpacks the given dimension of a rank R tensor into rank (R-1) tensors.

    Parameters
    ----------
    x : dict.
        shape, dtype and format of value to be unpacked.
    y: tuple or list
        the list of output tensor.
    num : int.
        the length of the dim axis, automatically inferred if None(default).
    axis: int.
        the axis to unpack along.
    kernel_name : str
        cce kernel name, default value is "unpack".

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    format = x.get("format")
    _check_params(shape, num, axis, format, dtype, kernel_name)

    # infer the value of num
    real_axis = axis + len(shape) if axis < 0 else axis
    num = shape[real_axis]

    # turn the input shape into three dimensions (a, b, c), so axis = 1
    beferdim = 1
    for befer_dim in shape[0:real_axis]:
        beferdim *= befer_dim
    afterdim = 1
    for after_dim in shape[real_axis + 1:]:
        afterdim *= after_dim
    reshape = (beferdim, shape[real_axis], afterdim)

    _, _, is_use_split = check_use_special_optimize(dtype, afterdim, flag=False)
    reshape_input = x.copy()
    reshape_input["shape"] = reshape
    real_axis = 1
    # only 1 output tensor, so output equals to input
    if num == 1:
        copy_only(reshape_input, reshape_input, kernel_name)
    # use split
    elif is_use_split:
        split_d(reshape_input, y, split_dim=real_axis,
                num_split=num, kernel_name=kernel_name)
    else:
        new_dtype, afterdim, _ = check_use_special_optimize(
            dtype, afterdim, flag=False)
        new_shape = (beferdim, reshape[real_axis], afterdim)

        input_place = tvm.placeholder(
            new_shape, name="input_place", dtype=new_dtype)
        sch, build_list = _unpack_schedule(
            input_place, reshape, y, num, real_axis, dtype)

        with build_config:
            tvm.build(sch, build_list, "cce", name=kernel_name)
