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

dropout_do_mask
"""
import math
from functools import reduce as functools_reduce

import te.platform.cce_params as cce_params
from te import platform as tbe_platform
from te import tvm
from te.platform import cce_intrin_md
from te.platform import cce_util
from te.platform.cce_build import build_config
from topi.cce import util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

# shape limit
SHAPE_SIZE_LIMIT = 1 << 30
# elems one batch can process
ELEMS_BATCH_PROCESS_FP16 = cce_params.ELEMENTS_VECTOR_OP_FP16
VECTOR_INST_BLOCK_WIDTH = cce_params.VECTOR_INST_BLOCK_WIDTH
SIZE_SIXTEEN = 16


def _division_sixteen(shape):

    if len(shape) < 2:
        if shape[-1] == 0:
            raise RuntimeError("value of shape is illegal")
        return False

    if shape[-1] == 0 or shape[-2] == 0:
        raise RuntimeError("value of shape is illegal")

    if shape[-1] % SIZE_SIXTEEN == 0 and shape[-2] % SIZE_SIXTEEN == 0:
        return True
    else:
        return False


def op_select_format(input_tensor,
                     input_mask,
                     input_keep_prob,
                     output,
                     kernel_name="dropout_do_mask"):
    """
    _division_sixteen : judge whether the last two dimensions are divided by 16
    scalar2tensor_one : convert scalar to tensor
    """
    shape_0 = input_tensor.get("ori_shape")
    shape_1 = input_mask.get("ori_shape")
    shape_2 = input_keep_prob.get("ori_shape")

    shape_0 = util.scalar2tensor_one(shape_0)
    shape_1 = util.scalar2tensor_one(shape_1)
    shape_2 = util.scalar2tensor_one(shape_2)

    if _division_sixteen(shape_0) and not _division_sixteen(
            shape_1) and not _division_sixteen(shape_2):
        # Nz+ND+ND
        input0 = gen_param(classify="input0",
                           name="x",
                           datatype="float16,float16,float,float",
                           format="ND,FRACTAL_NZ,ND,FRACTAL_NZ")
        input1 = gen_param(classify="input1",
                           name="mask",
                           datatype="uint8,uint8,uint8,uint8",
                           format="ND,ND,ND,ND")
        input2 = gen_param(classify="input2",
                           name="keep_prob",
                           datatype="float16,float16,float,float",
                           format="ND,ND,ND,ND")
        output0 = gen_param(classify="output0",
                            name="y",
                            datatype="float16,float16,float,float",
                            format="ND,FRACTAL_NZ,ND,FRACTAL_NZ")
    else:
        # ND+ND
        input0 = gen_param(classify="input0",
                           name="x",
                           datatype="float16,float",
                           format="ND,ND")
        input1 = gen_param(classify="input1",
                           name="mask",
                           datatype="uint8,uint8",
                           format="ND,ND")
        input2 = gen_param(classify="input2",
                           name="keep_prob",
                           datatype="float16,float",
                           format="ND,ND")
        output0 = gen_param(classify="output0",
                            name="y",
                            datatype="float16,float",
                            format="ND,ND")

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _new_alloc(ir_builder, dtype, shape, name, scope):
    """
    decl new buffer

    Parameters
    ----------
    ir_builder : tvm.ir_builder
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
    buf_var = ir_builder.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = tvm.decl_buffer(shape,
                                 buf_var.dtype,
                                 name=name,
                                 scope=scope,
                                 data=buf_var)

    return new_buffer


def _get_ub_max_elements(dtype):
    """
    return how many elems can put in ub
    """
    ub_size_bytes = tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.UB_SIZE)
    # 8 bit = 1byte, lots of '8' below for this reason
    dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
    # 2.125 means tensor_data + tensor_zero + tensor_mask = 1+1+0.125=2.125
    total_ele = (ub_size_bytes // dtype_bytes_size -
                 ELEMS_BATCH_PROCESS_FP16) // 2.125
    total_ele = int(
        total_ele // ELEMS_BATCH_PROCESS_FP16)*ELEMS_BATCH_PROCESS_FP16
    total_ele = int(total_ele // (32 * 8)) * 32 * 8

    return total_ele


def _sel_data(ir_builder, src_data, alloc_mem, offset_length):
    """
    select data from src_data with alloc_mem data
    """
    if src_data.dtype == 'float16':
        # list alloc_mem has diff data adds, ref: list alloc_res
        ir_builder.emit(
            tvm.call_extern(src_data.dtype, 'vsel',
                            alloc_mem[4].access_ptr('w', offset=offset_length),
                            alloc_mem[4].access_ptr('r', offset=offset_length),
                            alloc_mem[0].access_ptr('r', offset=0), 1, 1, 1, 1,
                            0, 0, 0))
    else:
        ir_builder.emit(
            tvm.call_extern('float16', 'vsel', alloc_mem[2].access_ptr('w'),
                            alloc_mem[1].access_ptr('r'),
                            alloc_mem[0].access_ptr('r'), 1, 1, 1, 1, 0, 0, 0))
        ir_builder.emit(
            tvm.call_extern("float32", "vconv_f162f32",
                            alloc_mem[3].access_ptr("rw"),
                            alloc_mem[2].access_ptr("r"), 2, 1, 1, 8, 4))
        ir_builder.emit(
            tvm.call_extern(
                "float32", "vmul",
                alloc_mem[4].access_ptr("rw", offset=offset_length),
                alloc_mem[4].access_ptr("r", offset=offset_length),
                alloc_mem[3].access_ptr("r"), 2, 1, 1, 1, 8, 8, 8))


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-statements,unused-argument
def _do_operation(ir_builder, place_holders, plantform_paras, loops_remains, const_1, block_offset,
                  shape_each_core, num_remain_by_128, is_not_align):
    #alloc_res[0:data_zero_ub 1:data_fp16_1
    # 2:data_fp16_mask 3:data_fp32_1 4:data_tensor_ub 5:data_mask_ub]
    #offsets[0:total_gm_data_offset 1:total_gm_mask_offset
    # 2:offset_gm_data 3:offset_gm_mask
    # 4:total_ub_data_offset 5:total_ub_mask_offset]
    #repeates[0:repeate_ub_data 1:repeate_ub_mask 2:repeate_ub_vector
    # 3:repeate_d 4:repeate_m 5:repeate_v]
    # 6 = the list size

    reg = ir_builder.allocate(place_holders[0].dtype, (1, ),
                              name="reg",
                              scope=tbe_platform.scope_reg)
    [alloc_res, offsets, repeates] = [[None]*7, [0]*6, [0]*6]
    [offsets[0], offsets[1]] = [
        offsets[0] + block_offset,
        offsets[1] + block_offset // 8
    ]

    [alloc_res[0], alloc_res[1], alloc_res[2], alloc_res[3]] = [
        _new_alloc(ir_builder,
                   'float16', (ELEMS_BATCH_PROCESS_FP16, ),
                   "data_zero_ub",
                   scope=tbe_platform.scope_ubuf),
        _new_alloc(ir_builder,
                   'float16', (ELEMS_BATCH_PROCESS_FP16, ),
                   "data_fp16one_ub",
                   scope=tbe_platform.scope_ubuf),
        _new_alloc(ir_builder,
                   'float16', (ELEMS_BATCH_PROCESS_FP16, ),
                   "data_fp16_all1_mask_ub",
                   scope=tbe_platform.scope_ubuf),
        _new_alloc(ir_builder,
                   'float32', (ELEMS_BATCH_PROCESS_FP16, ),
                   "data_fp32one_ub",
                   scope=tbe_platform.scope_ubuf)
    ] if (place_holders[0].dtype == 'float32') else [
        _new_alloc(ir_builder,
                   'float16', (ELEMS_BATCH_PROCESS_FP16, ),
                   "data_zero_ub",
                   scope=tbe_platform.scope_ubuf), None, None, None
    ]

    [alloc_res[4], alloc_res[5], alloc_res[6]] = [
        _new_alloc(ir_builder,
                   place_holders[0].dtype, (plantform_paras[0], ),
                   "data_tensor_ub",
                   scope=tbe_platform.scope_ubuf),
        _new_alloc(ir_builder,
                   place_holders[1].dtype, (plantform_paras[0] // 8, ),
                   "data_mask_ub",
                   scope=tbe_platform.scope_ubuf),
        _new_alloc(ir_builder,
                   place_holders[3].dtype, (1, ),
                   "keep_prob_tensor_ub",
                   scope=tbe_platform.scope_ubuf)
    ] if (loops_remains[0] > 0) else [None, None, None]
    const_buf = _new_alloc(ir_builder,
                           const_1.dtype, (ELEMS_BATCH_PROCESS_FP16, ),
                           "const_1_ub",
                           scope=tbe_platform.scope_ubuf)
    if loops_remains[0] > 0:
        with ir_builder.for_range(0, loops_remains[0],
                                  name='index0') as index0:
            [offsets[2], offsets[3]] = [
                block_offset + plantform_paras[0]*index0,
                block_offset // 8 + plantform_paras[0] // 8 * index0
            ]
            # 16: fp16 elems can be move by once is 16,
            # lots of '16' below for this reason
            # 32: uint8 elems can be move by once is 32,
            # lots of '32' below for this reason
            # 64: fp32 elems can be process by vector instruction,
            # lots of '64' below for this reason
            [repeates[0], repeates[1], repeates[2]] = [
                plantform_paras[0] // 16, plantform_paras[0] // 8 //
                32, plantform_paras[0] // ELEMS_BATCH_PROCESS_FP16
            ] if (place_holders[0].dtype == 'float16') else [
                plantform_paras[0] // 8, plantform_paras[0] // 8 //
                32, plantform_paras[0] // 64
            ]

            ir_builder.emit(
                tvm.call_extern('float16', "vector_dup",
                                alloc_res[0].access_ptr("rw"),
                                tvm.const(0.0,
                                          dtype='float16'), 1, 1, 1, 8, 8))
            ir_builder.emit(
                tvm.call_extern(const_1.dtype, "vector_dup",
                                const_buf.access_ptr("rw"),
                                tvm.const(1.0, dtype=const_1.dtype), 1, 1, 1,
                                8, 8))

            if place_holders[0].dtype == 'float32':
                ir_builder.emit(
                    tvm.call_extern('float16', "vector_dup",
                                    alloc_res[1].access_ptr("rw"),
                                    tvm.const(1.0,
                                              dtype='float16'), 1, 1, 1, 8, 8))

            ir_builder.emit(
                tvm.call_extern(
                    place_holders[1].dtype, "copy_gm_to_ubuf",
                    alloc_res[5].access_ptr("w"),
                    place_holders[1].access_ptr("r", offset=offsets[3]), 0, 1,
                    repeates[1], 0, 0))

            ir_builder.emit(
                tvm.call_extern(
                    place_holders[0].dtype, "copy_gm_to_ubuf",
                    alloc_res[4].access_ptr("w"),
                    place_holders[0].access_ptr("r", offset=offsets[2]), 0, 1,
                    repeates[0], 0, 0))
            ir_builder.emit(
                tvm.call_extern(place_holders[3].dtype, "copy_gm_to_ubuf",
                                alloc_res[6].access_ptr("w"),
                                place_holders[3].access_ptr("r", offset=0), 0,
                                1, 1, 0, 0))
            cce_intrin_md.reset_mask_insn(ir_builder,
                                          const_1.dtype,
                                          bits=1,
                                          mask_func=None)
            ir_builder.emit(
                tvm.call_extern(place_holders[3].dtype, 'vdiv',
                                alloc_res[6].access_ptr('w'),
                                const_buf.access_ptr('r'),
                                alloc_res[6].access_ptr('r'), 1, 1, 1, 1, 8, 8,
                                8))

            cce_intrin_md.reset_mask_insn(ir_builder,
                                          const_1.dtype,
                                          bits=ELEMS_BATCH_PROCESS_FP16,
                                          mask_func=None)
            ir_builder.emit(
                tvm.call_extern(place_holders[3].dtype, "reg_mov",
                                tvm.call_extern(reg.dtype, "reg", reg[0]),
                                alloc_res[6].access_ptr("r", offset=0)))

            offset_src = 64 * 255 if place_holders[0].dtype == "float32" else 128 * 255
            repeate_vmuls = repeates[2] // 255
            repeat_left = repeates[2] % 255
            for i in range(repeate_vmuls):
                ir_builder.emit(
                    tvm.call_extern(
                        place_holders[0].dtype, 'vmuls',
                        alloc_res[4].access_ptr('w', offset=offset_src*i),
                        alloc_res[4].access_ptr('r'), reg[0], 255, 1, 1, 8, 8))
            ir_builder.emit(
                tvm.call_extern(
                    place_holders[0].dtype, 'vmuls',
                    alloc_res[4].access_ptr('w',
                                            offset=offset_src*repeate_vmuls),
                    alloc_res[4].access_ptr('r',
                                            offset=offset_src*repeate_vmuls),
                    reg[0], repeat_left, 1, 1, 8, 8))

            with ir_builder.for_range(0, loops_remains[1],
                                      name='index1') as index1:
                ir_builder.emit(
                    tvm.call_extern(
                        place_holders[1].dtype, 'set_cmpmask',
                        alloc_res[5].access_ptr('r', offset=16*index1)))
                _sel_data(ir_builder, place_holders[0], alloc_res,
                          ELEMS_BATCH_PROCESS_FP16*index1)

            ir_builder.emit(
                tvm.call_extern(
                    place_holders[2].dtype, "copy_ubuf_to_gm",
                    place_holders[2].access_ptr('w', offset=offsets[2]),
                    alloc_res[4].access_ptr("r"), 0, 1, repeates[0], 0, 0))

        [offsets[0], offsets[1]] = [
            offsets[0] + plantform_paras[0]*loops_remains[0],
            offsets[1] + plantform_paras[0] * loops_remains[0] // 8
        ]

    if loops_remains[2]:
        # 0:data_shape 1:mask_shape
        if num_remain_by_128 != 0 and is_not_align:
            remain_shapes = ((int(place_holders[0].shape[0]) -
                              plantform_paras[0] * loops_remains[0],),
                             (int(place_holders[1].shape[0]) -
                              plantform_paras[0] // 8 * loops_remains[0],))
        else:
            remain_shapes = ((shape_each_core -
                              plantform_paras[0] * loops_remains[0],),
                             (shape_each_core // 8 -
                              plantform_paras[0] // 8 * loops_remains[0],))
        [alloc_res[4], alloc_res[5], alloc_res[6]] = [
            _new_alloc(ir_builder,
                       place_holders[0].dtype,
                       remain_shapes[0],
                       "data_tensor_ub",
                       scope=tbe_platform.scope_ubuf),
            _new_alloc(ir_builder,
                       place_holders[1].dtype,
                       remain_shapes[1],
                       "data_mask_ub",
                       scope=tbe_platform.scope_ubuf),
            _new_alloc(ir_builder,
                       place_holders[3].dtype, (1, ),
                       "keep_prob_tensor_ub",
                       scope=tbe_platform.scope_ubuf)
        ]

        [repeates[3], repeates[4], repeates[5]] = [
            int(math.ceil(remain_shapes[0][0]*1.0 / 8)),
            int(math.ceil(remain_shapes[1][0]*1.0 / 32)),
            int(remain_shapes[0][0]*1.0 / 64)
        ] if (place_holders[0].dtype == 'float32') else [
            int(math.ceil(remain_shapes[0][0]*1.0 / 16)),
            int(math.ceil(remain_shapes[1][0]*1.0 / 32)),
            int(remain_shapes[0][0]*1.0 / ELEMS_BATCH_PROCESS_FP16)
        ]

        ir_builder.emit(
            tvm.call_extern('float16', "vector_dup",
                            alloc_res[0].access_ptr("rw"),
                            tvm.const(0.0, dtype='float16'), 1, 1, 1, 8, 8))
        ir_builder.emit(
            tvm.call_extern(const_1.dtype, "vector_dup",
                            const_buf.access_ptr("rw"),
                            tvm.const(1.0, dtype=const_1.dtype), 1, 1, 1, 8,
                            8))

        if place_holders[0].dtype == 'float32':
            ir_builder.emit(
                tvm.call_extern('float16', "vector_dup",
                                alloc_res[1].access_ptr("rw"),
                                tvm.const(1.0,
                                          dtype='float16'), 1, 1, 1, 8, 8))
        ir_builder.emit(
            tvm.call_extern(
                place_holders[1].dtype, "copy_gm_to_ubuf",
                alloc_res[5].access_ptr("w"),
                place_holders[1].access_ptr("r", offset=offsets[1]), 0, 1,
                repeates[4], 0, 0))

        ir_builder.emit(
            tvm.call_extern(
                place_holders[0].dtype, "copy_gm_to_ubuf",
                alloc_res[4].access_ptr("w"),
                place_holders[0].access_ptr("r", offset=offsets[0]), 0, 1,
                repeates[3], 0, 0))

        ir_builder.emit(
            tvm.call_extern(place_holders[3].dtype, "copy_gm_to_ubuf",
                            alloc_res[6].access_ptr("w"),
                            place_holders[3].access_ptr("r", offset=0), 0, 1,
                            1, 0, 0))
        cce_intrin_md.reset_mask_insn(ir_builder,
                                      const_1.dtype,
                                      bits=1,
                                      mask_func=None)

        ir_builder.emit(
            tvm.call_extern(place_holders[3].dtype, 'vdiv',
                            alloc_res[6].access_ptr('w'),
                            const_buf.access_ptr('r'),
                            alloc_res[6].access_ptr('r'), 1, 1, 1, 1, 8, 8, 8))

        cce_intrin_md.reset_mask_insn(ir_builder,
                                      const_1.dtype,
                                      bits=ELEMS_BATCH_PROCESS_FP16,
                                      mask_func=None)
        ir_builder.emit(
            tvm.call_extern(place_holders[0].dtype, "reg_mov",
                            tvm.call_extern(reg.dtype, "reg", reg[0]),
                            alloc_res[6].access_ptr("r", offset=0)))

        offset_src = 64*255 if place_holders[0].dtype == "float32" else 128*255
        repeate_vmuls = repeates[5] // 255
        repeat_left = repeates[5] % 255
        for i in range(repeate_vmuls):
            ir_builder.emit(
                tvm.call_extern(
                    place_holders[0].dtype, 'vmuls',
                    alloc_res[4].access_ptr('w', offset=offset_src*i),
                    alloc_res[4].access_ptr('r'), reg[0], 255, 1, 1, 8, 8))
        ir_builder.emit(
            tvm.call_extern(
                place_holders[0].dtype, 'vmuls',
                alloc_res[4].access_ptr('w',
                                        offset=offset_src*repeate_vmuls),
                alloc_res[4].access_ptr('r',
                                        offset=offset_src*repeate_vmuls),
                reg[0], repeat_left, 1, 1, 8, 8))

        [loops_remains[1], loops_remains[3]] = [
            remain_shapes[0][0] // ELEMS_BATCH_PROCESS_FP16,
            remain_shapes[0][0] % ELEMS_BATCH_PROCESS_FP16
        ]

        with ir_builder.for_range(0, loops_remains[1],
                                  name='index2') as index2:
            ir_builder.emit(
                tvm.call_extern(
                    place_holders[1].dtype, 'set_cmpmask',
                    alloc_res[5].access_ptr('r', offset=16*index2)))
            _sel_data(ir_builder, place_holders[0], alloc_res,
                      ELEMS_BATCH_PROCESS_FP16*index2)

        [offsets[4], offsets[5]] = [
            plantform_paras[1] * loops_remains[1] * 2, plantform_paras[2] *
            loops_remains[1] * 2
        ] if (place_holders[0].dtype == 'float32') else [
            plantform_paras[1] * loops_remains[1], plantform_paras[2] *
            loops_remains[1]
        ]

        if loops_remains[3]:
            cce_intrin_md.reset_mask_insn(ir_builder,
                                          place_holders[0].dtype,
                                          bits=loops_remains[3],
                                          mask_func=None)

            ir_builder.emit(
                tvm.call_extern(
                    place_holders[0].dtype, 'vmuls',
                    alloc_res[4].access_ptr('w', offset=offsets[4]),
                    alloc_res[4].access_ptr('r', offset=offsets[4]), reg[0], 1,
                    1, 1, 8, 8))

            ir_builder.emit(
                tvm.call_extern(
                    place_holders[1].dtype, 'set_cmpmask',
                    alloc_res[5].access_ptr('r', offset=offsets[5])))
            _sel_data(ir_builder, place_holders[0], alloc_res, offsets[4])
            cce_intrin_md.reset_mask_insn(ir_builder,
                                          place_holders[0].dtype,
                                          bits=ELEMS_BATCH_PROCESS_FP16,
                                          mask_func=None)

        ir_builder.emit(
            tvm.call_extern(
                place_holders[2].dtype, "copy_ubuf_to_gm",
                place_holders[2].access_ptr('w', offset=offsets[0]),
                alloc_res[4].access_ptr("r"), 0, 1, repeates[3], 0, 0))


def _kernel_ir(dst, src, const_1):
    """
    dropout_do_mask kernel
    """
    ir_builder = tvm.ir_builder.create()
    place_holders = [src[0], src[1], dst[0], src[2]]  # input & output params

    # 0:max_elemets
    # 1:cnt_per_vsel(VECTOR_INST_BLOCK_WIDTH=256 bytes is maximum process unit
    #   in vector process)
    # 2:mask_cnt_per_vsel
    plantform_paras = [
        _get_ub_max_elements(place_holders[0].dtype),
        VECTOR_INST_BLOCK_WIDTH //
        (cce_util.get_type_bits(place_holders[0].dtype) // 8),
        (VECTOR_INST_BLOCK_WIDTH //
         (cce_util.get_type_bits(place_holders[0].dtype) // 8)) //
        cce_util.get_type_bits(place_holders[1].dtype)
    ]

    target_core_num, mask_num_each_core, core_num_one_more, num_remain_by_128, is_not_align = \
        _get_target_core_num(src[0], src[1])
    if num_remain_by_128 != 0 and is_not_align:
        # 0:loop_for_ub 1:loop_for_128
        # 2:remain_data_ub(after tilling by ub max process elements) 3:remain_ele
        loops_remains = [
            int(place_holders[0].shape[0]) // plantform_paras[0],
            plantform_paras[0] // ELEMS_BATCH_PROCESS_FP16,
            int(place_holders[0].shape[0]) % plantform_paras[0], num_remain_by_128
        ]
        _do_operation(ir_builder, place_holders, plantform_paras, loops_remains, const_1,
                      0, 0, num_remain_by_128, is_not_align)
    else:
        block_index = tvm.thread_axis("blockIdx.x")
        ir_builder.scope_attr(block_index, "thread_extent", target_core_num)

        with ir_builder.if_scope(block_index < core_num_one_more):
            shape_each_core = (mask_num_each_core + 1) * ELEMS_BATCH_PROCESS_FP16 * 8
            block_offset = shape_each_core * block_index
            # 0:loop_for_ub 1:loop_for_128
            # 2:remain_data_ub(after tilling by ub max process elements) 3:remain_ele
            loops_remains = [
                int(shape_each_core) // plantform_paras[0],
                plantform_paras[0] // ELEMS_BATCH_PROCESS_FP16,
                int(shape_each_core) % plantform_paras[0], num_remain_by_128
            ]
            _do_operation(ir_builder, place_holders, plantform_paras, loops_remains, const_1,
                          block_offset, shape_each_core, num_remain_by_128, is_not_align)

        with ir_builder.else_scope():
            shape_each_core = mask_num_each_core * ELEMS_BATCH_PROCESS_FP16 * 8
            block_offset = ELEMS_BATCH_PROCESS_FP16 * 8 * core_num_one_more + \
                           shape_each_core * block_index
            if num_remain_by_128:
                with ir_builder.if_scope(block_index == target_core_num - 1):
                    shape_each_core += num_remain_by_128 * 8
            # 0:loop_for_ub 1:loop_for_128
            # 2:remain_data_ub(after tilling by ub max process elements) 3:remain_ele
            loops_remains = [
                int(shape_each_core) // plantform_paras[0],
                plantform_paras[0] // ELEMS_BATCH_PROCESS_FP16,
                int(shape_each_core) % plantform_paras[0], 0
            ]
            _do_operation(ir_builder, place_holders, plantform_paras, loops_remains, const_1,
                          block_offset, shape_each_core, num_remain_by_128, is_not_align)

    return ir_builder.get()


def _get_target_core_num(data_input, data_mask):
    """ Get the device core numbers. for example, product = cloud,then target_core_num = 32,
    and then compute the greatest common number of actual device core numbers
    """
    mask_shape = data_mask.shape[:]
    input_shape = data_input.shape[:]

    target_core_num = tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.CORE_NUM)

    num_div_by_128 = int(mask_shape[0]) // 128
    num_remain_by_128 = int(mask_shape[0]) % 128
    core_num_one_more = 0
    is_not_align = True
    if int(mask_shape[0]) * 8 != int(input_shape[0]) or num_div_by_128 == 0:
        target_core_num = 1
        if num_remain_by_128 != 0:
            mask_num_each_core = int(mask_shape[0])
        else:
            mask_num_each_core = num_div_by_128
        return target_core_num, mask_num_each_core, core_num_one_more, num_remain_by_128, is_not_align
    else:
        is_not_align = False
    if int(num_div_by_128) <= int(target_core_num):
        target_core_num = num_div_by_128 if num_div_by_128 != 0 else 1
        mask_num_each_core = 1
        return target_core_num, mask_num_each_core, core_num_one_more, num_remain_by_128, is_not_align

    mask_num_each_core = num_div_by_128 // target_core_num
    core_num_one_more = num_div_by_128 % target_core_num

    return target_core_num, mask_num_each_core, core_num_one_more, num_remain_by_128, is_not_align


@util.check_input_type(dict, dict, dict, dict, str)
def drop_out_do_mask(input_tensor, input_mask, input_keep_prob, output,
                     kernel_name="dropout_do_mask"):
    """
    algorithm: tf_dropout_do_mask
    scale_x = x*(1 / keep_prob)
    res = select(mask == 1, scale_x, 0)

    Parameters
    ----------
    input_tensor : dict,shape and dtype of input_tensor,only support float16 and float32
    input_mask : dict,shape and dtype of input_mask
        shape of mask,1D, dtype == uint8
        length=(size(shape_tensor)+ELEMS_BATCH_PROCESS_FP16
        -1)/ELEMS_BATCH_PROCESS_FP16*ELEMS_BATCH_PROCESS_FP16/8
        eg. shape_tensor=[2,5,8] shape_mask=[16] shape_res=[2,5,8]
        shape_tensor=[15,17,19] shape_mask=[608] shape_res=[15,17,19]
    input_keep_prob : dict,shape and dtype of input_keep_prob
        shape of keep_prob, only 1 parament and equals to (1)
        prob scale (0.0,1.0] NOTICE: type same as dytpe
    output : dict,shape and dtype of output
    kernel_name : str
        cce kernel name, default value is "dropout_do_mask"

    Returns
    -------
    None
    """
    shape_tensor = input_tensor.get("shape")
    shape_mask = input_mask.get("shape")
    shape_keep_prob = input_keep_prob.get("shape")
    dtype = input_tensor.get("dtype")
    if shape_keep_prob == 1:
        shape_keep_prob = (shape_keep_prob, )
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_tensor)
    util.check_shape_size(shape_tensor, SHAPE_SIZE_LIMIT)
    util.check_dtype_rule(dtype.lower(), ["float16", "float32"])
    if len(shape_mask) != 1:
        raise RuntimeError("The length of mask shape must be 1")
    if shape_keep_prob not in [(1, ), [1, ]]:
        raise RuntimeError("Only support shape (1, ) or [1, ]")
    # functools_reduce: product of all dimension
    # Align to ELEMS_BATCH_PROCESS_FP16
    product_mask = (functools_reduce(lambda x, y: x*y, shape_tensor[:]) +
                    ELEMS_BATCH_PROCESS_FP16 - 1) // \
                   ELEMS_BATCH_PROCESS_FP16 * ELEMS_BATCH_PROCESS_FP16 // 8
    if product_mask != shape_mask[0]:
        raise RuntimeError("The mask[0] should=%d, but now=%d" %
                           (product_mask, shape_mask[0]))
    data_tensor = tvm.placeholder(
        (functools_reduce(lambda x, y: x*y, shape_tensor), ),
        dtype=dtype,
        name="data_tensor")
    data_mask = tvm.placeholder(
        (functools_reduce(lambda x, y: x*y, shape_mask), ),
        dtype='uint8',
        name="data_mask")
    keep_prob_tensor = tvm.placeholder(shape_keep_prob,
                                       dtype=dtype,
                                       name="keep_prob_tensor")
    const_1 = tvm.const(1.0, dtype=dtype)

    res = tvm.extern([shape_tensor, shape_mask, shape_keep_prob],
                     [data_tensor, data_mask, keep_prob_tensor],
                     lambda ins, outs: _kernel_ir(outs, ins, const_1),
                     name="res",
                     dtype=dtype)

    tensor_list = [data_tensor, data_mask, keep_prob_tensor, res]
    schedule = tvm.create_schedule(res.op)

    with build_config:
        tvm.build(schedule, tensor_list, "cce", name=kernel_name)
