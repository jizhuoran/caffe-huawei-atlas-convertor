"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     __init__.py
DESC:     make tik a module
CREATED:  2019-7-04 20:12:13
MODIFIED: 2019-7-24 14:04:45
"""
# pylint: disable=redefined-builtin
from te.platform.cce_params import OUTPUT_PATH_CLASS, GM_NAME_MAP_CLASS, \
    scope_aicpu, scope_ca, scope_cb, scope_cbuf, scope_cc, scope_gm, \
    scope_reg, scope_ubuf, dma_copy, dma_copy_global, CCE_AXIS, \
    VECTOR_INST_BLOCK_NUM, VECTOR_INST_BLOCK_WIDTH, WGT_ELEM_BYTES, \
    VECTOR_INST_MAX_REPEAT_TIMES, INP_ELEM_BYTES, OUT_ELEM_BYTES, BLOCK_IN, \
    BLOCK_OUT, BLOCK_REDUCE, BLOCK_REDUCE_INT8, BLOCK_VECTOR, INP_WIDTH, \
    GEMM_MODE, GEVM_MODE, GLB_ELEM_BYTES, WGT_WIDTH, C0_SIZE, CONV_MODE, \
    CUBE_MKN, ELEMENTS_VECTOR_OP_FP16, DEFAULT_ADD_VALUE, DEFAULT_MUL_VALUE, \
    conv_buffer_ex, OUT_WIDTH, scope_smask

from .api.tik_build import Tik
from .api.tik_dprofile import Dprofile
from .api import tik_conf
from .tik_lib.tik_conf_ import set_product_version, unset_product_version
from .api.tik_tensor import Tensor
from .api.tik_scalar import Scalar
from .tik_lib.tik_expr import Expr
from .tik_lib.tik_util import all, any
from .tik_lib.tik_params import scope_cbuf_out
