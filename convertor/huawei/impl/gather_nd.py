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

gather_nd
"""
from functools import reduce as functools_reduce
from te import platform as tbe_platform
from te import tvm
from te.platform import insn_cmd
from te.platform.cce_build import build_config
from te.platform.cce_build import build_config_update
from topi.cce import util
SHAPE_SIZE_LIMIT = 2**30

def _new_alloc(ir_build, dtype, shape, name, scope):
    """decl new buffer

    Parameters
    ----------
    ir_build : tvm.ir_builder
        Developer API of IR node builder make function.

    dtype : string
        buffer date type.

    shape : list of int
        buffer shape.

    name : string
        buffer name.

    scope : string
        buffer memory scope.

    Returns
    -------
    buffer : tvm.schedule.Buffer
        Symbolic data buffer.
    """
    buf_var = ir_build.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = tvm.decl_buffer(shape,
                                 buf_var.dtype,
                                 name=name,
                                 scope=scope,
                                 data=buf_var)
    return new_buffer


# pylint: disable=locally-disabled,unused-argument,too-many-locals
# pylint: disable=locally-disabled,too-many-arguments
def _indice_unit_tiling(dst, data, indices, step_indices, jump_step,
                        shape_data, shape_indices, dict_out):
    """
    tiling indices into unit
    unit = shape_indices[-1]
    """
    ir_build = tvm.ir_builder.create()

    # -1 means getting last element in the shape
    ele_unit = shape_indices[-1]

    # calculate half ub size based on indices dtype
    dtype = indices.dtype
    ub_size_bytes = tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.UB_SIZE)
    # dtype takes 8 bytes
    dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
    total_ele = ub_size_bytes // dtype_bytes_size
    # leave 100 elements space for block ub and register space
    total_ele = total_ele - 100
    # 0.5 means half ub elements
    half_ele = int(0.5 * total_ele)
    half_ub_shape = (half_ele, )
    half_data_ub = _new_alloc(ir_build,
                              data.dtype,
                              half_ub_shape,
                              "half_data_ub",
                              scope=tbe_platform.scope_ubuf)
    half_indices_ub = _new_alloc(ir_build,
                                 indices.dtype,
                                 half_ub_shape,
                                 "half_indices_ub",
                                 scope=tbe_platform.scope_ubuf)
    # allocate 32 register ub space to reg_block
    block_ub = _new_alloc(ir_build,
                          data.dtype, (32, ),
                          "block_ub",
                          scope=tbe_platform.scope_ubuf)
    _kernel_ir(dst, data, indices, step_indices, jump_step, shape_data,
               shape_indices, ele_unit, half_indices_ub, half_ele,
               half_data_ub, block_ub, ir_build)
    return ir_build.get()


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
# pylint: disable=line-too-long,too-many-statements,too-many-branches
def _kernel_ir(dst, data, indices, step_indices, jump_step, shape_data,
               shape_indices, ele_unit, half_indices_ub, half_ele,
               half_data_ub, block_ub, ir_build):
    """
    gather_nd kernel
    ---------
    calculate offset
    """
    half_param = int(0.5 * half_ele)
    half_param_ub_shape = (half_param, )
    indices_ele = int(functools_reduce(lambda i, j: i * j, shape_indices))
    # init params_total_size as 1
    params_total_size = 1
    # max dimension is 8
    if shape_indices[-1] > 8:
        raise RuntimeError("gather_nd only support 1D ~ 8D")
    # allocate 8 register space to reg
    reg = ir_build.allocate(indices.dtype, (8, ),
                            name='reg',
                            scope=tbe_platform.scope_reg)
    # check reg size which is 24
    if len(shape_data) > 24:
        raise RuntimeError("only allocating 24 registers spaces")
    # allocate 24 register space to reg_shape
    reg_shape = ir_build.allocate(indices.dtype, (24, ),
                                  name='reg_shape',
                                  scope=tbe_platform.scope_reg)
    # allocate 1 register space to reg_gm
    reg_gm = ir_build.allocate(indices.dtype, (1, ),
                               name='reg_gm',
                               scope=tbe_platform.scope_reg)
    # allocate 1 register space to reg_mul
    reg_mul = ir_build.allocate(indices.dtype, (1, ),
                                name='reg_mul',
                                scope=tbe_platform.scope_reg)

    for i, data_dim in enumerate(shape_data):
        shape_value = tvm.const(data_dim, dtype=indices.dtype)
        reg_shape[i] = shape_value
    tiling_times, tiling_last, indies_part_num = move_indice_num_per_tiling(
        shape_data, shape_indices)
    # calculate multi-core number
    params_total_size = data.shape[0].value
    post_step = params_total_size // step_indices
    burst_length = post_step
    burst_length_tiling = burst_length // half_ele

    # ub size is redefine under double buffer, avoid double buffer work with multi-core
    if burst_length_tiling.value >= 1:
        half_ele = half_ele // 8
        # calculate 32B aglined number
        half_ele = reassign_aglined_number(data, half_ele)
        burst_length_tiling = burst_length // half_ele
        block = burst_length_tiling
        block_tiling = block + 1
        block_index = tvm.thread_axis("blockIdx.x")
        ir_build.scope_attr(block_index, "thread_extent", block_tiling)
    if tiling_times > 0:
        with ir_build.for_range(0, tiling_times,
                                name='indice_row') as indice_row:
            indice_burst_len = calculate_burst_len_by_dtype(indices, indies_part_num)
            ir_build.emit(
                tvm.call_extern(
                    indices.dtype, "copy_gm_to_ubuf",
                    half_indices_ub.access_ptr("w"),
                    indices.access_ptr('r',
                                       offset=indies_part_num * indice_row), 0,
                    1, indice_burst_len, 0, 0))

            burst_length_tiling_last = burst_length % half_ele
            burst_len_last = calculate_burst_len_by_dtype(data, burst_length_tiling_last)
            indices_move_length = jump_step
            ele_num_per_indice_ub = total_ele_per_indice_ub(
                indies_part_num, jump_step, shape_data, shape_indices)
            # this deals with indices unit
            with ir_build.for_range(0,
                                    indies_part_num // jump_step,
                                    name='row1') as row1:
                # init reg_gm as 0
                reg_gm[0] = tvm.const(0, dtype=indices.dtype)
                i = row1 * jump_step
                reg_mul[0] = tvm.const(params_total_size, dtype=indices.dtype)
                with ir_build.for_range(0, indices_move_length,
                                        name='row2') as row2:
                    ir_build.emit(
                        tvm.call_extern(
                            indices.dtype, "reg_mov",
                            tvm.call_extern(reg.dtype, "reg", reg[row2]),
                            half_indices_ub.access_ptr('r', offset=i + row2)))
                    reg_mul[0] = reg_mul[0] // reg_shape[row2]
                    reg_gm[0] += reg[row2] * reg_mul[0]
                gm_offset = reg_gm[0]
                # 1 means tiling starts
                if burst_length_tiling.value >= 1:
                    burst_length_tiling_last = burst_length % half_ele
                    burst_len, burst_len_last = calculate_last_burst_len(data, half_ele, burst_length_tiling_last)

                    # calculate offset to move last 32B number to independent ub space
                    block_length = param_ele_per_block_by_dtype(data)
                    block_last = (
                        burst_length -
                        (burst_length_tiling - 1) * half_ele) % block_length
                    if int(block_last) != 0:
                        block_last = block_length - block_last
                    block_offset = (burst_len_last + burst_len -
                                    1) * block_length - block_last

                    # combine elements on last core with other elements
                    # therefore, total cores minus 1
                    with ir_build.if_scope(block_index < block - 1):
                        ir_build.emit(
                            tvm.call_extern(data.dtype, "copy_gm_to_ubuf",
                                            half_data_ub.access_ptr("w"),
                                            data.access_ptr('r',
                                                            offset=gm_offset +
                                                            block_index \
                                                                   *half_ele),
                                            0, 1, burst_len, 0, 0))
                        ir_build.emit(
                            tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr('w',
                                                           offset=indice_row*ele_num_per_indice_ub + row1*burst_length +
                                                           (block_index) \
                                                                  *half_ele),
                                            half_data_ub.access_ptr("r"), 0, 1,
                                            burst_len, 0, 0))

                    # 1 means the last core number
                    with ir_build.else_scope():
                        ir_build.emit(
                            tvm.call_extern(data.dtype, "copy_gm_to_ubuf",
                                            half_data_ub.access_ptr("w"),
                                            data.access_ptr('r',
                                                            offset=gm_offset +
                                                            (burst_length_tiling - 1) \
                                                                   * half_ele),
                                            0, 1, burst_len_last + burst_len, 0, 0))
                        with ir_build.for_range(0,
                                                block_length,
                                                name='block_row') as block_row:
                            ir_build.emit(
                                tvm.call_extern(
                                    data.dtype, "reg_mov",
                                    block_ub.access_ptr('w', offset=block_row),
                                    half_data_ub.access_ptr(
                                        'r', offset=block_offset + block_row)))

                        ir_build.emit(
                            tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr('w',
                                                           offset=indice_row*ele_num_per_indice_ub + row1*burst_length +
                                                           (burst_length_tiling - 1) \
                                                                  *half_ele),
                                            half_data_ub.access_ptr("r"), 0, 1,
                                            burst_len_last + burst_len - 1, 0, 0))
                        ir_build.emit(
                            tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr('w',
                                                           offset=row1*burst_length + indice_row*ele_num_per_indice_ub +
                                                           (burst_length_tiling - 1) \
                                                                  *half_ele + (burst_len_last + burst_len - 1)*block_length - block_last),
                                            block_ub.access_ptr("r"), 0, 1,
                                            1, 0, 0))
                else:
                    half_param, half_param_ub_shape = calculate_ele_num_by_dtype(data, burst_len_last)
                    buf_param_var = ir_build.allocate(
                        data.dtype,
                        half_param_ub_shape,
                        "half_data_ub",
                        scope=tbe_platform.scope_ubuf)
                    ir_build.scope_attr(buf_param_var.asnode(),
                                        "double_buffer_scope", 1)
                    half_data_ub = tvm.decl_buffer(
                        half_param_ub_shape,
                        buf_param_var.dtype,
                        "half_data_ub",
                        scope=tbe_platform.scope_ubuf,
                        data=buf_param_var)

                    ir_build.emit(
                        tvm.call_extern(data.dtype, "copy_gm_to_ubuf",
                                        half_data_ub.access_ptr("w"),
                                        data.access_ptr('r', offset=gm_offset),
                                        0, 1, burst_len_last, 0, 0))
                    ir_build.emit(
                        tvm.call_extern(
                            dst.dtype, "copy_ubuf_to_gm",
                            dst.access_ptr(
                                'w',
                                offset=indice_row * ele_num_per_indice_ub +
                                row1 * burst_length),
                            half_data_ub.access_ptr("r"), 0, 1, burst_len_last,
                            0, 0))
    if indies_part_num * tiling_times < indices_ele:
        if tiling_times == 0:
            ele_num_per_indice_ub = 0
        indice_burst_len = calculate_burst_len_by_dtype(indices, tiling_last)
        ir_build.emit(
            tvm.call_extern(
                indices.dtype, "copy_gm_to_ubuf",
                half_indices_ub.access_ptr("w"),
                indices.access_ptr('r', offset=indies_part_num * tiling_times),
                0, 1, indice_burst_len, 0, 0))

        burst_length_tiling_last = burst_length % half_ele
        burst_len_last = calculate_burst_len_by_dtype(data, burst_length_tiling_last)

        indices_move_length = jump_step
        # this deals with indices unit
        with ir_build.for_range(0, tiling_last // jump_step,
                                name='row1') as row1:
            # init reg_gm as 0
            reg_gm[0] = tvm.const(0, dtype=indices.dtype)
            i = row1 * jump_step
            reg_mul[0] = tvm.const(params_total_size, dtype=indices.dtype)
            with ir_build.for_range(0, indices_move_length,
                                    name='row2') as row2:
                ir_build.emit(
                    tvm.call_extern(
                        indices.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[row2]),
                        half_indices_ub.access_ptr('r', offset=i + row2)))
                reg_mul[0] = reg_mul[0] // reg_shape[row2]
                reg_gm[0] += reg[row2] * reg_mul[0]
            gm_offset = reg_gm[0]
            # 1 means tiling starts
            if burst_length_tiling.value >= 1:
                burst_length_tiling_last = burst_length % half_ele
                burst_len, burst_len_last = calculate_last_burst_len(
                    data, half_ele, burst_length_tiling_last)
                # calculate offset to move last 32B number to independent ub space
                block_length = param_ele_per_block_by_dtype(data)
                block_last = (
                    burst_length -
                    (burst_length_tiling - 1) * half_ele) % block_length
                if int(block_last) != 0:
                    block_last = block_length - block_last
                block_offset = (burst_len_last + burst_len -
                                1) * block_length - block_last

                # combine elements on last core with other elements
                # therefore, total cores minus 1
                with ir_build.if_scope(block_index < block - 1):
                    ir_build.emit(
                        tvm.call_extern(data.dtype, "copy_gm_to_ubuf",
                                        half_data_ub.access_ptr("w"),
                                        data.access_ptr('r',
                                                        offset=gm_offset +
                                                        (block_index) \
                                                               *half_ele),
                                        0, 1, burst_len, 0, 0))
                    ir_build.emit(
                        tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=tiling_times*ele_num_per_indice_ub + row1*burst_length +
                                                       (block_index) \
                                                              *half_ele),
                                        half_data_ub.access_ptr("r"), 0, 1,
                                        burst_len, 0, 0))
                # 1 means the last core number
                with ir_build.else_scope():
                    ir_build.emit(
                        tvm.call_extern(data.dtype, "copy_gm_to_ubuf",
                                        half_data_ub.access_ptr("w"),
                                        data.access_ptr('r',
                                                        offset=gm_offset +
                                                        (burst_length_tiling - 1) \
                                                               * half_ele),
                                        0, 1, burst_len_last + burst_len, 0, 0))

                    with ir_build.for_range(0, block_length,
                                            name='block_row') as block_row:
                        ir_build.emit(
                            tvm.call_extern(
                                data.dtype, "reg_mov",
                                block_ub.access_ptr('w', offset=block_row),
                                half_data_ub.access_ptr('r',
                                                        offset=block_offset +
                                                        block_row)))

                    ir_build.emit(
                        tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=tiling_times*ele_num_per_indice_ub + row1*burst_length +
                                                       (burst_length_tiling - 1) \
                                                              *half_ele),
                                        half_data_ub.access_ptr("r"), 0, 1,
                                        burst_len_last + burst_len - 1, 0, 0))

                    ir_build.emit(
                        tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=tiling_times*ele_num_per_indice_ub + row1*burst_length +
                                                       (burst_length_tiling - 1) \
                                                              *half_ele + (burst_len_last + burst_len - 1)*block_length - block_last),
                                        block_ub.access_ptr("r"), 0, 1,
                                        1, 0, 0))
            else:
                half_param, half_param_ub_shape = calculate_ele_num_by_dtype(data, burst_len_last)
                buf_param_var = ir_build.allocate(
                    data.dtype,
                    half_param_ub_shape,
                    "half_data_ub",
                    scope=tbe_platform.scope_ubuf)
                ir_build.scope_attr(buf_param_var.asnode(),
                                    "double_buffer_scope", 1)
                half_data_ub = tvm.decl_buffer(half_param_ub_shape,
                                               buf_param_var.dtype,
                                               "half_data_ub",
                                               scope=tbe_platform.scope_ubuf,
                                               data=buf_param_var)

                ir_build.emit(
                    tvm.call_extern(data.dtype, "copy_gm_to_ubuf",
                                    half_data_ub.access_ptr("w"),
                                    data.access_ptr('r', offset=gm_offset), 0,
                                    1, burst_len_last, 0, 0))

                ir_build.emit(
                    tvm.call_extern(
                        dst.dtype, "copy_ubuf_to_gm",
                        dst.access_ptr(
                            'w',
                            offset=tiling_times * ele_num_per_indice_ub +
                            row1 * burst_length), half_data_ub.access_ptr("r"),
                        0, 1, burst_len_last, 0, 0))


# pylint: disable=locally-disabled,unused-argument,too-many-locals
# pylint: disable=locally-disabled,too-many-arguments
def _scalar_performence_indice_unit_tiling(dst, data, indices, step_indices,
                                           jump_step, shape_data,
                                           shape_indices, dict_out):
    """
    tiling indices into unit
    unit = shape_indices[-1]
    """
    ir_build = tvm.ir_builder.create()
    # calculate half ub size based on indices dtype
    dtype = indices.dtype
    ub_size_bytes = tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.UB_SIZE)
    # dtype takes 8 bytes
    dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
    total_ele = ub_size_bytes // dtype_bytes_size
    # leave 100 elements space for block ub and register space
    total_ele = total_ele - 100
    # 0.5 means half ub elements
    half_ele = int(0.5 * total_ele)
    # allocate 32 register ub space to reg_block
    block_ub = _new_alloc(ir_build,
                          data.dtype, (32, ),
                          "block_ub",
                          scope=tbe_platform.scope_ubuf)
    _scalar_performence_kernel_ir(dst, data, indices, jump_step, shape_data,
                                  shape_indices, half_ele, block_ub, ir_build)
    return ir_build.get()


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
# pylint: disable=locally-disabled,too-many-statements
def _scalar_performence_kernel_ir(dst, data, indices, jump_step, shape_data,
                                  shape_indices, half_ele, block_ub, ir_build):
    """
    gather_nd kernel
    ---------
    calculate offset
    """
    platform_core_num = tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.CORE_NUM)
    indices_ele = int(functools_reduce(lambda i, j: i * j, shape_indices))
    # init params_total_size as 1
    params_total_size = 1
    # max dimension is 8
    if shape_indices[-1] > 8:
        raise RuntimeError("gather_nd only support 1D ~ 8D")
    # allocate 8 register space to reg
    reg = ir_build.allocate(indices.dtype, (8, ),
                            name='reg',
                            scope=tbe_platform.scope_reg)
    # check reg size which is 24
    if len(shape_data) > 24:
        raise RuntimeError("only allocating 24 registers spaces")
    # allocate 24 register space to reg_shape
    reg_shape = ir_build.allocate(indices.dtype, (24, ),
                                  name='reg_shape',
                                  scope=tbe_platform.scope_reg)
    # allocate 1 register space to reg_gm
    reg_gm = ir_build.allocate(indices.dtype, (4, ),
                               name='reg_gm',
                               scope=tbe_platform.scope_reg)
    # allocate 1 register space to reg_mul
    reg_mul = ir_build.allocate(indices.dtype, (4, ),
                                name='reg_mul',
                                scope=tbe_platform.scope_reg)
    for i, data_dim in enumerate(shape_data):
        shape_value = tvm.const(data_dim, dtype=indices.dtype)
        reg_shape[i] = shape_value
    # calculate multi-core number
    params_total_size = data.shape[0].value
    param_total_shape = (15000, )
    indice_total_shape = (26000, )
    out_put_shape = (20000, )
    three_indices_ub = _new_alloc(ir_build,
                                  indices.dtype,
                                  indice_total_shape,
                                  "three_indices_ub",
                                  scope=tbe_platform.scope_ubuf)
    three_data_ub = _new_alloc(ir_build,
                               data.dtype,
                               param_total_shape,
                               "three_data_ub",
                               scope=tbe_platform.scope_ubuf)
    output_ub = _new_alloc(ir_build,
                           data.dtype,
                           out_put_shape,
                           "output_ub",
                           scope=tbe_platform.scope_ubuf)
    param_len = calculate_burst_len_by_dtype(data, params_total_size)

    ir_build.emit(
        tvm.call_extern(data.dtype, "copy_gm_to_ubuf",
                        three_data_ub.access_ptr("w"), data.access_ptr('r'), 0,
                        1, param_len, 0, 0))
    indices_move_length = jump_step
    # 32B aligned
    ele_per_block = param_ele_per_block_by_dtype(data)
    ele_per_core, last_ele_per_core = get_ele_num_per_core(
        indices_ele, shape_indices[-1], platform_core_num)
    indice_ele_per_core = ele_per_core * shape_indices[-1]
    indice_last_ele_per_core = last_ele_per_core * shape_indices[-1]
    last_moved_param_ele = ele_per_core + last_ele_per_core
    core_len, core_len_last = calculate_last_burst_len(
        data, ele_per_core, last_moved_param_ele)

    indice_core_len, indice_core_len_last = indice_core_len_by_dtype(
        indices, indice_ele_per_core, indice_last_ele_per_core)
    # calculate params num list per dim
    shape_list = []
    for data_dim in shape_data:
        params_total_size = params_total_size // data_dim
        shape_list.append(params_total_size)

    # this deals with indices unit
    # get core num according to the product
    block = platform_core_num
    block_tiling = block
    block_index = tvm.thread_axis("blockIdx.x")
    ir_build.scope_attr(block_index, "thread_extent", block_tiling)
    with ir_build.if_scope(block_index < block - 1):
        ir_build.emit(
            tvm.call_extern(
                indices.dtype, "copy_gm_to_ubuf",
                three_indices_ub.access_ptr("w"),
                indices.access_ptr('r',
                                   offset=block_index * indice_ele_per_core),
                0, 1, indice_core_len, 0, 0))

        last_ele = ele_per_core % ele_per_block
        aglined_ele_num = (ele_per_core // ele_per_block) * ele_per_block
        aglined_offset = aglined_ele_num - (ele_per_block - last_ele)
        reg_tmp = ir_build.allocate(data.dtype, (4, ),
                                    name='reg_tmp',
                                    scope=tbe_platform.scope_reg)
        compile_plat = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
        if compile_plat in ("Ascend310",):
            with ir_build.for_range(0, ele_per_core, name='core_row') as core_row:
                reg_gm[0] = tvm.const(0, dtype=indices.dtype)
                reg_mul[0] = tvm.const(params_total_size, dtype=indices.dtype)
                gm_offset = 0
                for row2 in range(indices_move_length):
                    ir_build.emit(
                        tvm.call_extern(
                            indices.dtype, "reg_mov",
                            tvm.call_extern(reg.dtype, "reg", reg[row2]),
                            three_indices_ub.access_ptr(
                                'r', offset=core_row * jump_step + row2)))
                    # caculate start position
                    gm_offset += reg[row2] * shape_list[row2]
                # caculate the start position to fetch num
                ir_build.emit(
                    tvm.call_extern(
                        data.dtype, "reg_mov",
                        output_ub.access_ptr("w", offset=core_row),
                        three_data_ub.access_ptr('r', offset=gm_offset)))
        else:
            for i in range(0, ele_per_core, 4):
                for j in range(4):
                    gm_offset = 0
                    for row2 in range(indices_move_length):
                        ir_build.emit(
                            tvm.call_extern(
                                indices.dtype, "reg_mov",
                                tvm.call_extern(reg.dtype, "reg", reg[row2]),
                                three_indices_ub.access_ptr(
                                    'r', offset=(i + j) * jump_step + row2)))
                        # calculate start position
                        gm_offset += reg[row2] * shape_list[row2]
                    # calculate the start position to fetch num
                    ir_build.emit(
                        tvm.call_extern(
                            data.dtype, "reg_mov",
                            tvm.call_extern(reg_tmp.dtype, "reg", reg_tmp[j]),
                            three_data_ub.access_ptr('r', offset=gm_offset)))
                for j in range(4):
                    ir_build.emit(
                        tvm.call_extern(
                            data.dtype, "reg_mov",
                            output_ub.access_ptr('w', offset=i + j),
                            tvm.call_extern(reg_tmp.dtype, "reg", reg_tmp[j])))

        with ir_build.for_range(0, ele_per_block,
                                name='block_row') as block_row:
            ir_build.emit(
                tvm.call_extern(
                    data.dtype, "reg_mov",
                    block_ub.access_ptr('w', offset=block_row),
                    output_ub.access_ptr('r',
                                         offset=aglined_offset + block_row)))
        if core_len == 1:
            ir_build.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=(block_index) * ele_per_core),
                    output_ub.access_ptr("r"), 0, 1, 1, 0, 0))
        else:
            ir_build.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=(block_index) * ele_per_core),
                    output_ub.access_ptr("r"), 0, 1, core_len - 1, 0, 0))
            ir_build.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr(
                        'w',
                        offset=(block_index) * ele_per_core + aglined_offset),
                    block_ub.access_ptr("r"), 0, 1, 1, 0, 0))
    # 1 means the last core number
    with ir_build.else_scope():
        # combine last data with previous core
        ir_build.emit(
            tvm.call_extern(
                indices.dtype, "copy_gm_to_ubuf",
                three_indices_ub.access_ptr("w"),
                indices.access_ptr('r',
                                   offset=block_index * indice_ele_per_core),
                0, 1, indice_core_len_last, 0, 0))
        with ir_build.for_range(0,
                                ele_per_core + last_ele_per_core,
                                name='last_core_row') as last_core_row:

            reg_gm[0] = tvm.const(0, dtype=indices.dtype)
            for row2 in range(indices_move_length):
                ir_build.emit(
                    tvm.call_extern(
                        indices.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[row2]),
                        three_indices_ub.access_ptr(
                            'r', offset=last_core_row * jump_step + row2)))
                # calculate start position
                reg_gm[0] += reg[row2] * shape_list[row2]
            # calculate the start position to fetch num
            gm_offset = reg_gm[0]
            ir_build.emit(
                tvm.call_extern(
                    data.dtype, "reg_mov",
                    output_ub.access_ptr('w', offset=last_core_row),
                    three_data_ub.access_ptr('r', offset=gm_offset)))
        ir_build.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr('w', offset=(block_index) * ele_per_core),
                output_ub.access_ptr("r"), 0, 1, core_len_last, 0, 0))


# pylint: disable=locally-disabled,unused-argument,too-many-locals
# pylint: disable=too-many-arguments
def _large_param_shape_performence_indice_unit_tiling(dst, data, indices,
                                                      step_indices, jump_step,
                                                      shape_data,
                                                      shape_indices, dict_out):
    """
    tiling indices into unit
    unit = shape_indices[-1]
    """
    ir_build = tvm.ir_builder.create()
    # allocate 32 register ub space to reg_block
    block_ub = _new_alloc(ir_build,
                          data.dtype, (32, ),
                          "block_ub",
                          scope=tbe_platform.scope_ubuf)
    _large_param_shape_performence_kernel_ir(dst, data, indices, jump_step,
                                             shape_data, shape_indices,
                                             step_indices, block_ub, ir_build)
    return ir_build.get()


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
# pylint: disable=locally-disabled,too-many-statements
def _large_param_shape_performence_kernel_ir(dst, data, indices, jump_step,
                                             shape_data, shape_indices,
                                             step_indices, block_ub, ir_build):
    """
    gather_nd kernel
    ---------
    calculate offset
    """
    platform_core_num = tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.CORE_NUM)
    indices_ele = int(functools_reduce(lambda i, j: i * j, shape_indices))
    # init params_total_size as 1
    params_total_size = 1
    # max dimension is 8
    if shape_indices[-1] > 8:
        raise RuntimeError("gather_nd only support 1D ~ 8D")
    # allocate 8 register space to reg
    reg = ir_build.allocate(indices.dtype, (8, ),
                            name='reg',
                            scope=tbe_platform.scope_reg)
    # check reg size which is 24
    if len(shape_data) > 24:
        raise RuntimeError("only allocating 24 registers spaces")
    # allocate 24 register space to reg_shape
    reg_shape = ir_build.allocate(indices.dtype, (24, ),
                                  name='reg_shape',
                                  scope=tbe_platform.scope_reg)
    # allocate 1 register space to reg_gm
    reg_gm = ir_build.allocate(indices.dtype, (1, ),
                               name='reg_gm',
                               scope=tbe_platform.scope_reg)
    # allocate 1 register space to reg_mul
    reg_mul = ir_build.allocate(indices.dtype, (1, ),
                                name='reg_mul',
                                scope=tbe_platform.scope_reg)
    for i, data_dim in enumerate(shape_data):
        shape_value = tvm.const(data_dim, dtype=indices.dtype)
        reg_shape[i] = shape_value
    # calculate multi-core number
    params_total_size = data.shape[0].value
    indice_total_shape = (14000, )
    out_put_shape = (35000, )
    three_indices_ub = _new_alloc(ir_build,
                                  indices.dtype,
                                  indice_total_shape,
                                  "three_indices_ub",
                                  scope=tbe_platform.scope_ubuf)
    output_ub = _new_alloc(ir_build,
                           data.dtype,
                           out_put_shape,
                           "output_ub",
                           scope=tbe_platform.scope_ubuf)

    indices_move_length = jump_step
    # 32B aligned
    ele_per_block = param_ele_per_block_by_dtype(data)
    ele_per_core, last_ele_per_core = get_ele_num_per_core(
        indices_ele, shape_indices[-1], platform_core_num)
    indice_ele_per_core = ele_per_core * shape_indices[-1]
    indice_last_ele_per_core = last_ele_per_core * shape_indices[-1]
    post_step = params_total_size // step_indices

    param_moved_ele_per_core = ele_per_core*post_step
    last_param_moved_ele_per_core = (ele_per_core + last_ele_per_core) * post_step

    param_core_len = calculate_burst_len_by_dtype(data, post_step)
    core_len, core_len_last = calculate_last_burst_len(
        data, param_moved_ele_per_core, last_param_moved_ele_per_core)

    indice_core_len, indice_core_len_last = indice_core_len_by_dtype(
        indices, indice_ele_per_core, indice_last_ele_per_core)
    # calculate params num list per dim
    shape_list = []
    for data_dim in shape_data:
        params_total_size = params_total_size // data_dim
        shape_list.append(params_total_size)
    # this deals with indices unit
    # get core num according to the product
    block = platform_core_num
    block_tiling = block
    block_index = tvm.thread_axis("blockIdx.x")
    ir_build.scope_attr(block_index, "thread_extent", block_tiling)
    with ir_build.if_scope(block_index <= block - 2):
        ir_build.emit(
            tvm.call_extern(
                indices.dtype, "copy_gm_to_ubuf",
                three_indices_ub.access_ptr("w"),
                indices.access_ptr('r',
                                   offset=block_index * indice_ele_per_core),
                0, 1, indice_core_len, 0, 0))
        last_ele = (ele_per_core * post_step) % ele_per_block
        aglined_ele_num = (
            (ele_per_core * post_step) // ele_per_block) * ele_per_block

        aglined_offset = aglined_ele_num - (ele_per_block - last_ele)
        with ir_build.for_range(0, ele_per_core, name='core_row') as core_row:
            reg_gm[0] = tvm.const(0, dtype=indices.dtype)
            reg_mul[0] = tvm.const(params_total_size, dtype=indices.dtype)
            gm_offset = 0
            for row2 in range(indices_move_length):
                ir_build.emit(
                    tvm.call_extern(
                        indices.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[row2]),
                        three_indices_ub.access_ptr(
                            'r', offset=core_row * jump_step + row2)))
                # calculate start position
                gm_offset += reg[row2] * shape_list[row2]
            # calculate the start position to fetch num
            ir_build.emit(
                tvm.call_extern(
                    data.dtype, "copy_gm_to_ubuf",
                    output_ub.access_ptr("w", offset=core_row * post_step),
                    data.access_ptr('r', offset=gm_offset), 0, 1,
                    param_core_len, 0, 0))
        with ir_build.for_range(0, ele_per_block,
                                name='block_row') as block_row:
            ir_build.emit(
                tvm.call_extern(
                    data.dtype, "reg_mov",
                    block_ub.access_ptr('w', offset=block_row),
                    output_ub.access_ptr('r',
                                         offset=aglined_offset + block_row)))
        # move length cannot be 0
        if int(core_len - 1) != 0:
            ir_build.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w',
                                   offset=(block_index) *
                                   ele_per_core * post_step),
                    output_ub.access_ptr("r"), 0, 1, core_len - 1, 0, 0))
        ir_build.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr(
                    'w',
                    offset=(block_index) * ele_per_core * post_step +
                    aglined_offset), block_ub.access_ptr("r"), 0, 1, 1, 0, 0))
    # 1 means the last core number
    with ir_build.else_scope():
        ir_build.emit(
            tvm.call_extern(
                indices.dtype, "copy_gm_to_ubuf",
                three_indices_ub.access_ptr("w"),
                indices.access_ptr('r',
                                   offset=(block_index) * indice_ele_per_core),
                0, 1, indice_core_len_last, 0, 0))
        # combine last data with previous core
        with ir_build.for_range(0,
                                ele_per_core + last_ele_per_core,
                                name='last_core_row') as last_core_row:
            reg_gm[0] = tvm.const(0, dtype=indices.dtype)
            gm_offset = 0
            for row2 in range(indices_move_length):
                ir_build.emit(
                    tvm.call_extern(
                        indices.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[row2]),
                        three_indices_ub.access_ptr(
                            'r', offset=last_core_row * jump_step + row2)))
                gm_offset += reg[row2] * shape_list[row2]
                # calculate the start position to fetch num
            ir_build.emit(
                tvm.call_extern(data.dtype, "copy_gm_to_ubuf",
                                output_ub.access_ptr("w",
                                                     offset=post_step*last_core_row),
                                data.access_ptr('r', offset=gm_offset), 0, 1,
                                param_core_len, 0, 0))
        ir_build.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr('w',
                               offset=(block_index) *
                               ele_per_core * post_step),
                output_ub.access_ptr("r"), 0, 1, core_len_last, 0, 0))


def calculate_last_burst_len(data, ele_num, last_ele_num):
    """
    calculate burst len and related last burst len
    """
    type_dict = {
        "dtype32_ele": 8,
        "dtype64_ele": 4,
        "dtype16_ele": 16,
        "dtype8_ele": 32,
    }

    if data.dtype == "int32" or data.dtype == "float32":
        burst_len = (ele_num +
                     type_dict["dtype32_ele"] - 1) // \
                    type_dict["dtype32_ele"]
        burst_len_last = (last_ele_num +
                          type_dict["dtype32_ele"] - 1) // \
                        type_dict["dtype32_ele"]
    elif data.dtype == "float16":
        burst_len = (ele_num +
                     type_dict["dtype16_ele"] - 1) // \
                    type_dict["dtype16_ele"]
        burst_len_last = (last_ele_num +
                          type_dict["dtype16_ele"] - 1) // \
                        type_dict["dtype16_ele"]
    elif data.dtype == "int8" or data.dtype == "uint8":
        burst_len = (ele_num + type_dict["dtype8_ele"] -
                     1) // type_dict["dtype8_ele"]
        burst_len_last = (
            last_ele_num +
            type_dict["dtype8_ele"] - 1) // type_dict["dtype8_ele"]
    return burst_len, burst_len_last


def calculate_burst_len_by_dtype(data, ele_num):
    """
    calculate burst len
    """
    type_dict = {
        "dtype32_ele": 8,
        "dtype64_ele": 4,
        "dtype16_ele": 16,
        "dtype8_ele": 32,
    }

    if data.dtype == "int32" or data.dtype == "float32":
        burst_len = (ele_num +
                     type_dict["dtype32_ele"] - 1) // \
                    type_dict["dtype32_ele"]
    elif data.dtype == "float16":
        burst_len = (ele_num +
                     type_dict["dtype16_ele"] - 1) // \
                    type_dict["dtype16_ele"]
    elif data.dtype == "int8" or data.dtype == "uint8":
        burst_len = (ele_num + type_dict["dtype8_ele"] -
                     1) // type_dict["dtype8_ele"]
    elif data.dtype == "int64":
        burst_len = (ele_num + type_dict["dtype64_ele"] -
                     1) // type_dict["dtype64_ele"]
    return burst_len


def param_ele_per_block_by_dtype(data):
    """
    calculate param num per block
    """
    type_dict = {
        "dtype32_ele": 8,
        "dtype64_ele": 4,
        "dtype16_ele": 16,
        "dtype8_ele": 32,
    }

    if data.dtype == "int32" or data.dtype == "float32":
        ele_per_block = type_dict["dtype32_ele"]
    elif data.dtype == "float16":
        ele_per_block = type_dict["dtype16_ele"]
    elif data.dtype == "int8" or data.dtype == "uint8":
        ele_per_block = type_dict["dtype8_ele"]
    return ele_per_block


def calculate_ele_num_by_dtype(data, ele_num):
    """
    calculate param num per core
    """
    type_dict = {
        "dtype32_ele": 8,
        "dtype64_ele": 4,
        "dtype16_ele": 16,
        "dtype8_ele": 32,
    }

    if data.dtype == "int32" or data.dtype == "float32":
        half_param = int(type_dict["dtype32_ele"] * ele_num)
        half_param_ub_shape = (half_param, )
    elif data.dtype == "float16":
        half_param = int(type_dict["dtype16_ele"] * ele_num)
        half_param_ub_shape = (half_param, )
    elif data.dtype == "int8" or data.dtype == "uint8":
        half_param = int(type_dict["dtype8_ele"] * ele_num)
        half_param_ub_shape = (half_param, )
    return half_param, half_param_ub_shape


def reassign_aglined_number(data, ele_num):
    """
    calculate aglined number
    """
    type_dict = {
        "dtype32_ele": 8,
        "dtype64_ele": 4,
        "dtype16_ele": 16,
        "dtype8_ele": 32,
    }

    if data.dtype == "int32" or data.dtype == "float32":
        ele_num = (ele_num //
                   type_dict["dtype32_ele"]) * type_dict["dtype32_ele"]
    elif data.dtype == "float16":
        ele_num = (ele_num //
                   type_dict["dtype16_ele"]) * type_dict["dtype16_ele"]
    elif data.dtype == "int8" or data.dtype == "uint8":
        ele_num = (ele_num //
                   type_dict["dtype8_ele"]) * type_dict["dtype8_ele"]
    return ele_num


def indice_core_len_by_dtype(indices, indice_ele_per_core, indice_last_ele_per_core):
    """
    calculate indice core len
    """
    type_dict = {
        "dtype32_ele": 8,
        "dtype64_ele": 4,
        "dtype16_ele": 16,
        "dtype8_ele": 32,
    }

    if indices.dtype == "int32":
        indice_core_len = (indice_ele_per_core + type_dict["dtype32_ele"] - 1) // \
                          type_dict["dtype32_ele"]
        indice_core_len_last = (indice_ele_per_core + indice_last_ele_per_core +
                                type_dict["dtype32_ele"] - 1) // \
                               type_dict["dtype32_ele"]
    elif indices.dtype == "int64":
        indice_core_len = (indice_ele_per_core + type_dict["dtype64_ele"] - 1) // \
                          type_dict["dtype64_ele"]
        indice_core_len_last = (
            indice_ele_per_core + indice_last_ele_per_core +
            type_dict["dtype64_ele"] - 1) // type_dict["dtype64_ele"]
    return indice_core_len, indice_core_len_last


def branch_two_check(shape_data, shape_indices):
    """
    check branch two limitation
    """
    platform_core_num = tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.CORE_NUM)

    indices_ele = int(functools_reduce(lambda i, j: i * j, shape_indices))

    ele_per_core, last_ele_per_core = get_ele_num_per_core(
        indices_ele, shape_indices[-1], platform_core_num)

    param_shape_length = len(shape_data)
    param_moved_part = 1

    for i in range(shape_indices[-1], param_shape_length):
        param_moved_part *= shape_data[i]

    total_param_moved_part = ele_per_core * param_moved_part
    last_total_param_moved_part = last_ele_per_core * param_moved_part
    # 35000 is param local ub length
    if(last_total_param_moved_part > 35000 or total_param_moved_part > 35000):
        return False
    return True


def get_ele_num_per_core(indices_ele, shape_indices_last_dim,
                         platform_core_num):
    """
    calculate num per core
    """
    ele_per_core = int(
        (indices_ele / shape_indices_last_dim) // platform_core_num)

    last_ele_per_core = int(
        (indices_ele / shape_indices_last_dim) % platform_core_num)

    return ele_per_core, last_ele_per_core


def move_length_indice_total_unit(shape_data, shape_indices, indice_num):
    """
    calculate move length indice total unit
    """
    last_indice_dim = shape_indices[-1]
    shape_data_size = len(shape_data)
    diff_num = abs(shape_data_size - last_indice_dim)
    total_data_shape = 1
    part_data_shape = 1
    data_part = 1
    indice_part = indice_num // last_indice_dim
    result = 0
    for data_dim in shape_data:
        total_data_shape *= data_dim
    for i in range(0, shape_data_size - diff_num):
        part_data_shape *= shape_data[i]
    data_part = total_data_shape // part_data_shape
    result = data_part * indice_part
    return result


def move_length_indice_per_unit(shape_data, shape_indices):
    """
    calculate move length indice per unit
    """
    last_indice_dim = shape_indices[-1]
    shape_data_size = len(shape_data)
    diff_num = abs(shape_data_size - last_indice_dim)
    total_data_shape = 1
    part_data_shape = 1
    data_part = 1
    for data_dim in shape_data:
        total_data_shape *= data_dim
    for i in range(0, shape_data_size - diff_num):
        part_data_shape *= shape_data[i]
    data_part = total_data_shape // part_data_shape
    return data_part


def get_nedded_ub_number(indices, shape_indices):
    """
    calculate get nedded ub number
    """
    # calculate half ub size based on indices dtype
    dtype = indices.dtype
    ub_size_bytes = tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.UB_SIZE)
    # dtype takes 8 bytes
    dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
    total_ele = ub_size_bytes // dtype_bytes_size
    # 0.5 means half ub elements
    half_ele = int(0.5 * total_ele)
    # leave external 500 element space for regsiter
    result = half_ele - 500
    return result


def indice_offset_per_half_ub(shape_data, shape_indices, indices):
    """
    calculate indice offset per half ub
    """
    param_part_move = move_length_indice_per_unit(shape_data, shape_indices)
    indice_num = shape_indices[-1]
    ub_ele = get_nedded_ub_number(indices, shape_indices)
    indice_num_tiling_times = ub_ele // indice_num
    result = param_part_move * indice_num_tiling_times
    return result


def move_indice_num_per_tiling(shape_data, shape_indices):
    """
    calculate move indice num per tiling
    """
    indice_num = shape_indices[-1]
    indices_ele = int(functools_reduce(lambda i, j: i * j, shape_indices))
    # under int8, the ub size is mimnum, half of the ub is about 15800
    # set spare space, in case of out of memory
    # hence, set 15000 element space on ub
    indice_num_tiling_times = 15000 // indice_num
    indies_part_num = indice_num_tiling_times * indice_num
    tiling_times = indices_ele // indies_part_num
    tiling_last = indices_ele % indies_part_num
    return tiling_times, tiling_last, indies_part_num


def total_ele_per_indice_ub(indies_part_num, jump_step, shape_data,
                            shape_indices):
    """
    calculate total ele per indice ub
    """
    unit_num = indies_part_num // jump_step
    param_part_move = move_length_indice_per_unit(shape_data, shape_indices)
    result = unit_num * param_part_move
    return result


def check_supported(dict_data, dict_indices, dict_out,
                    kernel_name='gather_nd'):
    """
    support aicpu route
    """
    return True


# pylint: disable=too-many-boolean-expressions,unnecessary-lambda
@util.check_input_type(dict, dict, dict, str)
def gather_nd(dict_data, dict_indices, dict_out, kernel_name='gather_nd'):
    """
    Gather slices from params into a Tensor with shape specified by indices.
    indices is an K-dimensional integer tensor, best thought of as a
    (K-1)-dimensional tensor of indices into params, where each element defines
     a slice of params:

      output[\\(i_0, ..., i_{K-2}\\)] = params[indices[\\(i_0, ..., i_{K-2}\\)]]

    in tf.gather_nd, indices defines slices into the
    first N dimensions of params, where N = indices.shape[-1].
    ----------
    indices = [[0, 0], [1, 1]]
    params = [['a', 'b'], ['c', 'd']]
    output = ['a', 'd']

    Parameters:
    ----------
    dict_data : dict
        Shape and dtype of params

    dict_indices : dict
        Shape and dtype of indices

    dict_out : dict
        dict of out

    kernel_name : cce kernel name, default value is "gather_nd"

    Returns
    -------
    None
    """
    shape_data = dict_data.get("shape")
    data_dtype = dict_data.get("dtype")
    shape_indices = dict_indices.get("shape")
    indice_dtype = dict_indices.get("dtype")
    data_dtype = data_dtype.lower()
    indice_dtype = indice_dtype.lower()
    check_list_params = ("float16", "float32", "int32", "int8", "uint8")
    check_list_indices = ("int64", "int32")
    util.check_dtype_rule(data_dtype, check_list_params)
    util.check_dtype_rule(indice_dtype, check_list_indices)
    util.check_shape_rule(shape_data)

    if shape_indices[0] != 0:
        util.check_shape_rule(shape_indices)
        util.check_shape_size(shape_data, SHAPE_SIZE_LIMIT)
        util.check_shape_size(shape_indices, SHAPE_SIZE_LIMIT)
        # init times as 0
        jump_step = shape_indices[-1]
        # init step_indices as 1
        step_indices = 1
        param_dim = len(shape_data)
        # check whether dimension of indices is 1
        if len(shape_indices) == 1:
            if shape_indices[-1] > param_dim:
                raise RuntimeError("the 1D size of shap_indices must "
                                   "be less than params dimensions")

        if shape_indices[-1] > param_dim:
            raise RuntimeError("the size of shap_indices must "
                               "be less than params dimensions")

        for i in range(0, shape_indices[-1]):
            step_indices *= tvm.const(shape_data[i], dtype=indice_dtype)
        indice_num = int(functools_reduce(lambda i, j: i * j, shape_indices))
        indices_reshape = [
            indice_num,
        ]
        param_num = int(functools_reduce(lambda i, j: i * j, shape_data))
        data_reshape = [
            param_num,
        ]

        data_1dim = tvm.placeholder(data_reshape,
                                    dtype=data_dtype,
                                    name="input_data")
        indices_1dim = tvm.placeholder(indices_reshape,
                                       dtype=indice_dtype,
                                       name="indices_1dim")

        #32B aligned
        if indice_dtype == "int32":
            ele_per_block = 8
        elif indice_dtype == "int64":
            ele_per_block = 4
        #32B aligned
        if data_dtype == "int32" or data_dtype == "float32":
            param_ele_per_block = 8
        elif data_dtype == "float16":
            param_ele_per_block = 16
        elif data_dtype == "int8" or data_dtype == "uint8":
            param_ele_per_block = 32
        ele_per_core = int((indice_num / shape_indices[-1]) // 32)
        post_step = param_num // step_indices.value
        post_step_aligned_check = post_step % param_ele_per_block
        # check 32B in one block
        block_num_check = int(ele_per_core * post_step // ele_per_block)
        param_block_num_check = int(ele_per_core * post_step //
                                    param_ele_per_block)
        branch_two_param_ub_check = branch_two_check(shape_data, shape_indices)
        # performence improvement only apply for little shape in net
        # param shape:0~15000
        # indice_shape:0~26000
        if (shape_indices[-1] == len(shape_data) and indice_num < 26000
                and param_num < 15000 and block_num_check != 0
                and param_block_num_check != 0 and indice_dtype != "int64"):
            res = tvm.extern(
                [shape_data, shape_indices], [data_1dim, indices_1dim],
                lambda ins, outs: _scalar_performence_indice_unit_tiling(
                    outs[0], ins[0], ins[1], step_indices, jump_step,
                    shape_data, shape_indices, dict_out),
                name="res",
                dtype=data_dtype)
        elif (block_num_check != 0 and ele_per_core * post_step < 35000
              and indice_num < 14000 and post_step_aligned_check == 0
              and branch_two_param_ub_check == True):
            res = tvm.extern(
                [shape_data, shape_indices], [data_1dim, indices_1dim],
                lambda ins, outs:
                _large_param_shape_performence_indice_unit_tiling(
                    outs[0], ins[0], ins[1], step_indices, jump_step,
                    shape_data, shape_indices, dict_out),
                name="res",
                dtype=data_dtype)
        else:
            res = tvm.extern(
                [shape_data, shape_indices], [data_1dim, indices_1dim],
                lambda ins, outs: _indice_unit_tiling(outs[0], ins[0], ins[
                    1], step_indices, jump_step, shape_data, shape_indices,
                                                      dict_out),
                name="res",
                dtype=data_dtype)

        tensor_list = [data_1dim, indices_1dim, res]
        schedule = tvm.create_schedule(res.op)

        with build_config:
            tvm.build(schedule, tensor_list, "cce", name=kernel_name)
    else:
        util.check_shape_size(shape_data, SHAPE_SIZE_LIMIT)
        util.check_shape_size(shape_indices, SHAPE_SIZE_LIMIT)
        ub_size_bytes = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.UB_SIZE)
        dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(data_dtype) // 8
        total_ele = ub_size_bytes // dtype_bytes_size
        split_factor = total_ele

        ele_cnt = functools_reduce(lambda x, y: x * y, shape_data)
        data_dtype = data_dtype.lower()
        shape_data_new = (ele_cnt, )
        shape_indice = (1, )

        data = tvm.placeholder(shape_data_new,
                               dtype=data_dtype,
                               name="input_data")
        data2 = tvm.placeholder(shape_indice, dtype="int32", name='data2')
        res_ub = tvm.compute(shape_data_new,
                             lambda *i: data(*i),
                             name='res_ub')
        res = tvm.compute(shape_data_new, lambda *i: res_ub(*i), name="res")

        tensor_list = [data, data2, res]
        sch = tvm.create_schedule(res.op)
        sch[res_ub].set_scope(tbe_platform.scope_ubuf)

        thread_block = tvm.thread_axis('blockIdx.x')
        sch[res].bind(res.op.axis[0], thread_block)
        axis_outer, axis_inner = sch[res].split(res.op.axis[0],
                                                factor=split_factor)

        tile_out_axis = axis_outer
        dma_in_axis = res_ub.op.axis[0]
        dma_out_axis = axis_inner

        sch[res_ub].compute_at(sch[res], tile_out_axis)
        sch[res_ub].emit_insn(dma_in_axis, insn_cmd.DMA_COPY)
        sch[res].emit_insn(dma_out_axis, insn_cmd.DMA_COPY)
        new_config = build_config_update(build_config, "dummy_placeholder",
                                         True)
        with new_config:
            tvm.build(sch, tensor_list, "cce", name=kernel_name)
