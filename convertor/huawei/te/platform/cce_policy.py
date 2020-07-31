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

parser the config params
"""
from __future__ import absolute_import as _abs

from te import tvm
# pylint: disable=import-error
from tvm._ffi.function import _init_api


class L1BufferManager():
    """Manage L1 buffer
    Save and call L1 buffer function
    """
    l1_info = {
        "op_L1_space": -1,
        "op_L1_fusion_type": -1,
        "L1_fusion_enabled": False,
        "L2_fusion_enabled": False
    }


# 'pylint: disable=invalid-name
def get_L1_info(key):
    """get L1 space"""
    return L1BufferManager.l1_info[key]


# 'pylint: disable=invalid-name
def set_L1_info(key, value):
    """set L1 space"""
    if key not in L1BufferManager.l1_info:
        raise RuntimeError("key[%s] not support." % key)

    ret = True
    if key == "op_L1_space":
        if value > 0:
            func = tvm.get_global_func("cce.set_l1_buffer", True)
            if func:
                ret = func(value)
        else:
            func = tvm.get_global_func("cce.reset_l1_buffer", True)
            if func:
                ret = func()

    L1BufferManager.l1_info[key] = value
    return ret


def enableL2():
    """
    Enable L2 fusion on tvm codegen.

    Parameters
    ----------

    Returns
    -------
    succ_flag : boolean
        end of execution
    """


# pylint: disable=invalid-name
def disableL2():
    """
    Disable L2 fusion on tvm codegen.

    Parameters
    ----------

    Returns
    -------
    succ_flag : boolean
        end of execution
    """


_init_api("te.platform.cce_policy")
