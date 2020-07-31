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

TVM cce runtime
"""
from __future__ import absolute_import as _abs

if __name__ == "platform":
    import sys
    import os

    # te warning: using python build-in 'platform'
    TP = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

    BAK_PATH = sys.path[:]
    for item in BAK_PATH:
        if (item == '' or os.path.realpath(item) == TP) and item in sys.path:
            sys.path.remove(item)

    sys.modules.pop('platform')
    import platform

    sys.path.insert(0, '')
    sys.path.append(TP)
else:
    from .cce_params import scope_cbuf, scope_ubuf, scope_ca, scope_cb,\
        scope_cc, scope_reg, scope_aicpu, dma_copy, dma_copy_global, \
        CUBE_MKN, scope_gm, scope_cbuf_fusion
    from .cce_params import CCE_AXIS, WGT_WIDTH, INP_WIDTH, OUT_WIDTH, BLOCK_IN, BLOCK_OUT, \
        BLOCK_REDUCE
    from .cce_params import INP_ELEM_BYTES, WGT_ELEM_BYTES, OUT_ELEM_BYTES, GLB_ELEM_BYTES,\
         conv_buffer_ex
    from .cce_params import INP_ELEM_BYTES, WGT_ELEM_BYTES, OUT_ELEM_BYTES, GLB_ELEM_BYTES, \
        C0_SIZE, ELEMENTS_VECTOR_OP_FP16
    from . import cce_intrin
    from . import cce_intrin_md
    from .cce_build import get_pass_list, build_config
    from .cce_conf import cceProduct, getValue, CceProductParams, set_status_check, \
        get_soc_spec, intrinsic_check_support, te_set_version, \
        api_check_support, SOC_VERSION, AICORE_TYPE, CORE_NUM
    from .cce_emitinsn_params import CceEmitInsnParams
    from .cce_policy import set_L1_info, get_L1_info
