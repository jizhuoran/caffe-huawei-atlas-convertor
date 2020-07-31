#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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

CCE configuration constants
"""
from __future__ import absolute_import as _abs
import threading

from te import tvm

# pylint: disable=invalid-name, useless-object-inheritance, too-few-public-methods
# Switch in tik/api/tik_build.py
class OUTPUT_PATH_CLASS(object):
    """
    cce output path
    """
    output_path = "kernel_meta"


# pylint: disable=invalid-name, useless-object-inheritance, too-few-public-methods
class GM_NAME_MAP_CLASS(object):
    """
    get gm_name map
    """
    gm_name_map = {}


# save tik gm tensor whether is workspace in to a list
TIK_WORKSPACE_SIZE_LIST = threading.local()
TIK_WORKSPACE_SIZE_LIST.local_list = []

# pylint: disable=invalid-name
# save tik gm tensor whether is atomic add in to a list
TIK_ATOMIC_ADD_LIST = threading.local()
TIK_ATOMIC_ADD_LIST.local_list = []
jump_expand_flag = False


# pylint: disable=invalid-name
# def the buffer var
scope_cbuf = "local.L1"
scope_ubuf = "local.UB"
scope_ca = "local.L0A"
scope_cb = "local.L0B"
scope_cc = "local.L0C"
scope_reg = "local.REG"
scope_aicpu = "local_aicpu"
scope_gm = "global"
scope_cbuf_fusion = "local.L1_Fusion"
scope_smask = "local.SMASK"

dma_copy = "dma_copy"
dma_copy_global = "global"

# def the cce thread axis for sync
CCE_AXIS = tvm.thread_axis("cce")
CCE_MASK_AXIS = tvm.thread_axis("cce_mask")
CCE_GLOBAL_AXIS = tvm.thread_axis("cce_global")

MASK_VAR = tvm.var("MASK")
RSVD_CNT = tvm.var("RSVD_CNT")

# def the cce vector intrinsic params
VECTOR_INST_BLOCK_WIDTH = 256
VECTOR_INST_BLOCK_NUM = 8
VECTOR_INST_MAX_REPEAT_TIMES = 255

# def the gemm const
WGT_WIDTH = 16
INP_WIDTH = 16
OUT_WIDTH = 16
BLOCK_IN = 16
BLOCK_OUT = 16
BLOCK_REDUCE = 16

# def the gemm int8/uint8 reduce const
BLOCK_REDUCE_INT8 = 32
# def the gemv/gevm vector const
BLOCK_VECTOR = 1

INP_ELEM_BYTES = (BLOCK_IN*BLOCK_REDUCE*INP_WIDTH // 8)
WGT_ELEM_BYTES = (BLOCK_OUT*BLOCK_REDUCE*WGT_WIDTH // 8)
OUT_ELEM_BYTES = (BLOCK_IN*BLOCK_OUT*OUT_WIDTH // 8)
GLB_ELEM_BYTES = (16*OUT_WIDTH // 8)

# def the mad pattern
GEMM_MODE = 0
GEVM_MODE = 1
CONV_MODE = 2

C0_SIZE = 16
ELEMENTS_VECTOR_OP_FP16 = 128

DEFAULT_MUL_VALUE = 1
DEFAULT_ADD_VALUE = 0

CUBE_MKN = {"int8": {'mac': [16, 32, 16]},
            "uint8": {'mac': [16, 32, 16]},
            "int16": {'mac': [16, 16, 16]},
            "int32": {'mac': [16, 16, 16]},
            "float16": {'mac': [16, 16, 16]},
            "float32": {'mac': [16, 16, 16]}}

# def limitation of repeart times for a single copy instrin and align value
VECTOR_COPY_NBURST_LIMIT = 4096
VECTOR_SINGLE_BLOCK_WIDTH_FP16 = 16

# pylint: disable=invalid-name, useless-object-inheritance, too-few-public-methods
class conv_buffer_ex(object):
    """
    conv buffer
    """
    offsetPad = None

# represent 5 soc, currently contains in tik
ASCEND_310 = "Ascend310"
ASCEND_910 = "Ascend910"
HI3796CV300ES = "Hi3796CV300ES"
HI3796CV300CS = "Hi3796CV300CS"
ASCEND_610 = "Ascend610"
ASCEND_620 = "Ascend620"
_AIC_ENGINE = "AiCore"
_VEC_ENGINE = "VectorCore"

AIC = ASCEND_610 + _AIC_ENGINE
VEC = ASCEND_620 + _VEC_ENGINE
HI3796CV300ESAIC = HI3796CV300ES + _AIC_ENGINE
HI3796CV300CSAIC = HI3796CV300CS + _AIC_ENGINE
ASCEND_310AIC = ASCEND_310 + _AIC_ENGINE
ASCEND_910AIC = ASCEND_910 + _AIC_ENGINE

ONLY_TIK_API_MAP = {
    """
    contains api that not is instr.
    """
    "vector_dup_v1": {
        ASCEND_310AIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        ASCEND_910AIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        HI3796CV300ESAIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        AIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        VEC: ["uint16", "int16", "float16", "uint32", "int32", "float32"]
    },
    "vector_dup": {
        ASCEND_310AIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        ASCEND_910AIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        HI3796CV300ESAIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        AIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        VEC: ["uint16", "int16", "float16", "uint32", "int32", "float32"]
    },
    "vci": {
        VEC: ["uint16", "int16", "float16", "uint32", "int32", "float32"]
    },
    "scatter_vnchwconv": {
        ASCEND_310AIC: ["int8", "uint8", "uint16", "int16", "float16"],
        ASCEND_910AIC: ["int8", "uint8", "uint16", "int16", "float16"],
        HI3796CV300ESAIC: ["int8", "uint8", "uint16", "int16", "float16",
                           "uint32", "int32", "float32"],
        AIC: ["int8", "uint8", "uint16", "int16", "float16", "uint32", "int32", "float32"],
        VEC: ["int8", "uint8", "uint16", "int16", "float16", "uint32", "int32", "float32"]
    },
    "vnchwconv": {
        ASCEND_310AIC: ["int8", "uint8", "uint16", "int16", "float16"],
        ASCEND_910AIC: ["int8", "uint8", "uint16", "int16", "float16"],
        HI3796CV300ESAIC: ["int8", "uint8", "uint16", "int16",
                           "float16", "uint32", "int32", "float32"],
        AIC: ["int8", "uint8", "uint16", "int16", "float16", "uint32", "int32", "float32"],
        VEC: ["int8", "uint8", "uint16", "int16", "float16", "uint32", "int32", "float32"]
    },
    "vec_trans_scatter": {
        ASCEND_310AIC: ["int8", "uint8", "uint16", "int16", "float16"],
        ASCEND_910AIC: ["int8", "uint8", "uint16", "int16", "float16"],
        HI3796CV300ESAIC: ["int8", "uint8", "uint16", "int16", "float16",
                           "uint32", "int32", "float32"],
        AIC: ["int8", "uint8", "uint16", "int16", "float16", "uint32", "int32", "float32"],
        VEC: ["int8", "uint8", "uint16", "int16", "float16", "uint32", "int32", "float32"]
    },
    "data_move_v1": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "vnchwtrans": {
        ASCEND_310AIC: ["uint16", "int16", "float16"],
        ASCEND_910AIC: ["uint16", "int16", "float16"],
        HI3796CV300ESAIC: ["uint16", "int16", "float16"],
        AIC: ["uint16", "int16", "float16"],
        VEC: ["uint16", "int16", "float16"]
    },
    "vreduceadd": {
        ASCEND_310AIC: ["float16", "float32"],
        ASCEND_910AIC: ["float16", "float32"],
        HI3796CV300ESAIC: ["float16", "float32"],
        AIC: ["float16", "float32"],
        VEC: ["float16", "float32"]
    },
    "scalar_conv": {
        ASCEND_910AIC: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                        'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        HI3796CV300ESAIC: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                           'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        AIC: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
              'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        VEC: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
              'f322s32z', 'f162f32', 'f322f16', 'f322f16o']
    },
    "mov_rpn_cor_ir_to_scalar": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "load2dv1": {
        ASCEND_310AIC: ["int8", "uint8", "uint16", "int16", "float16"],
        ASCEND_910AIC: ["int8", "uint8", "uint16", "int16", "float16"],
        HI3796CV300ESAIC: ["uint4", "int4", "int8", "uint8", "uint16", "int16", "float16"],
        AIC: ["uint4", "int4", "int8", "uint8", "uint16", "int16", "float16"],
        VEC: ["uint4", "int4", "int8", "uint8", "uint16", "int16", "float16"]
    },
    "load2dv2": {
        HI3796CV300ESAIC: ["uint4", "int4", "int8", "uint8", "uint16", "int16", "float16"],
        AIC: ["uint4", "int4", "int8", "uint8", "uint16", "int16", "float16"],
        VEC: ["uint4", "int4", "int8", "uint8", "uint16", "int16", "float16"]
    },
    "assign": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "set_l0_set_value": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "load3dv1": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: []
    },
    "col2img": {
        ASCEND_910AIC: ["float16", "float32"],
        AIC: ["float16", "float32"]
    },
    "mmad_broadcast": {
        ASCEND_310AIC: ['f16f16', 'f32f32', 's32s32', 'f32f16'],
        ASCEND_910AIC: ['f16f16', 'f32f32', 's32s32', 'f32f16'],
        HI3796CV300ESAIC: ['f32f32', 's32s32'],
        AIC: ['f16f16', 'f32f32', 's32s32', 'f32f16'],
    },
    "tensor_padding_with_matrix": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: []
    },
    "data_move": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "data_move_quant": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "tensor_mov": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "load3dv2": {
        HI3796CV300ESAIC: ["int8", "uint8", "float16"],
        AIC: ["int8", "uint8", "float16"],
        VEC: ["int8", "uint8", "float16"]
    },
    "load_smask": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "load_image": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: []
    },
    "mov_vmrgsort4_sr_to_scalar": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "set_rpn_cor_ir": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "set_rpn_offset": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "rpn_cor": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "rpn_cor_diag": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "mov_atomic_add_to_scalar": {
        ASCEND_910AIC: []
    },
    "mov_small_channel_to_scalar": {
        ASCEND_910AIC: []
    },
    "mov_system_cache_to_scalar": {
        ASCEND_910AIC: []
    },
    "mov_fp2int_mode_to_scalar": {
        ASCEND_910AIC: []
    },
    "set_atomic_add": {
        ASCEND_910AIC: [],
        AIC: [],
        VEC: []
    },
    "set_small_channel": {
        ASCEND_910AIC: []
    },
    "set_system_cache": {
        ASCEND_910AIC: []
    },
    "set_fp2int_mode": {
        ASCEND_910AIC: []
    },
    "instr_preload": {
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "get_overflow_status": {
        ASCEND_910AIC: []
    },
    "set_overflow_status": {
        ASCEND_910AIC: []
    },
    "mov_cmpmask_to_tensor": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "mov_tensor_to_cmpmask": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        AIC: [],
        VEC: []
    },
    "scatter_vmulva": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: []
    },
    "scatter_vaddva": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: []
    },
    "vec_rsqrt_high_preci": {
        ASCEND_310AIC: ["float16", "float32"],
        ASCEND_910AIC: ["float16", "float32"]
    },
    "vec_ln_high_preci": {
        ASCEND_310AIC: ["float16"]
    },
    "vec_expm1_high_preci": {
        ASCEND_310AIC: ["float16"],
        ASCEND_910AIC: ["float16"]
    },
    "vec_rec_high_preci": {
        ASCEND_310AIC: ["float16", "float32"],
        ASCEND_910AIC: ["float16", "float32"]
    },
    "conv2d": {
        ASCEND_310AIC: ["s8s8s32", "u8s8s32", "f16f16f32"],
        ASCEND_910AIC: ["s8s8s32", "u8s8s32", "f16f16f32"]
    },
    "fixpipe": {
        ASCEND_310AIC: ["f32f16", "f32f32", "s32s32", "s32f16"],
        ASCEND_910AIC: ["f32f16", "f32f32", "s32s32", "s32f16"],
        AIC: ["f32f16", "f32f32", "s32s32", "s32f16"]
    },
    "matmul": {
        ASCEND_310AIC: ["s8s8s32", "u8s8s32", "f16f16f32"],
        ASCEND_910AIC: ["s8s8s32", "u8s8s32", "f16f16f32"],
        AIC: ["s8s8s32", "u8s8s32", "f16f16f32"]
    }
}

TRANS_TIK_API_TO_INSTR_MAP = {
    """
    use to save some api that is instr
    """
    "vcmpv_lt_v1": "vcmpv_lt",
    "vcmpv_gt_v1": "vcmpv_gt",
    "vcmpv_ge_v1": "vcmpv_ge",
    "vcmpv_eq_v1": "vcmpv_eq",
    "vcmpv_ne_v1": "vcmpv_ne",
    "vcmpv_le_v1": "vcmpv_le",
    "v_cpadd": "vcpadd",
    "scalar_sqrt": "sqrt",
    "scalar_abs": "abs",
    "scalar_countbit0": "bcnt0",
    "scalar_countbit1": "bcnt1",
    "scalar_countleading0": "clz",
    "scalar_max": "max",
    "scalar_min": "min",
    "v_add": "vadd",
    "v_sub": "vsub",
    "v_max": "vmax",
    "v_min": "vmin",
    "v_mul": "vmul",
    "v_and": "vand",
    "v_or": "vor",
    "v_relu": "vrelu",
    "v_abs": "vabs",
    "v_not": "vnot",
    "v_axpy": "vaxpy",
    "v_adds": "vadds",
    "vmuls": "vmuls",
    "v_sel": "vsel",
    "v_conv": "vconv",
    "vcmp_lt": "vcmp",
    "vcmp_gt": "vcmp",
    "vcmp_ge": "vcmp",
    "vcmp_eq": "vcmp",
    "vcmp_ne": "vcmp",
    "vcmp_le": "vcmp",
    "broadcast_ub_to_l0c": "broadcast_ub_to_cc",
    "vmrgch": "vmergech",
    "vrpsort16": "vbitsort",
    "scatter_vcmp_lt": "scatter_vcmp",
    "scatter_vcmp_gt": "scatter_vcmp",
    "scatter_vcmp_ge": "scatter_vcmp",
    "scatter_vcmp_eq": "scatter_vcmp",
    "scatter_vcmp_ne": "scatter_vcmp",
    "scatter_vcmp_le": "scatter_vcmp",
    "vreducemax": "vcmax",
    "vreducemin": "vcmin"
}
