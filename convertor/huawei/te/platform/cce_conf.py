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
import te.lang.cce
# pylint: disable=import-error
from tvm._ffi.function import _init_api
from .cce_params import TRANS_TIK_API_TO_INSTR_MAP
from .cce_params import ONLY_TIK_API_MAP
from .cce_policy import set_L1_info
from .insn_cmd import get_insn_cmd

DDK_SOC_VERSION_MAP = {
    "1.1": {
        "SOC_VERSION": "Ascend310",
        "CORE_TYPE": "AiCore"
    },
    "1.3": {
        "SOC_VERSION": "Ascend310",
        "CORE_TYPE": "AiCore"
    },
    "1.60": {
        "SOC_VERSION": "Ascend910",
        "CORE_TYPE": "AiCore"
    },
    "2.10": {
        "SOC_VERSION": "Ascend610",
        "CORE_TYPE": "AiCore"
    },
    "2.11": {
        "SOC_VERSION": "Ascend620",
        "CORE_TYPE": "AiCore"
    },
    "3.20": {
        "SOC_VERSION": "Ascend620", # "Ascend610"
        "CORE_TYPE": "VectorCore"
    },
    "5.10": {
        "SOC_VERSION": "Hi3796CV300ES",
        "CORE_TYPE": "AiCore"
    },
}

SOC_VERSION_MAP = {
    "Ascend310": {
        "AiCore": {"AICoreNum": 2, "DdkVersion": "1.33.xx.xxx"},
    },
    "Ascend910": {
        "AiCore": {"AICoreNum": 32, "DdkVersion": "1.60.xx.xxx"},
    },
    "Ascend610": {
        "AiCore": {"AICoreNum": 9, "DdkVersion": "2.10.xx.xxx"},
        "VectorCore": {"AICoreNum": 8, "DdkVersion": "3.20.xx.xxx"},
    },
    "Ascend620": {
        "AiCore": {"AICoreNum": 10, "DdkVersion": "2.11.xx.xxx"},
        "VectorCore": {"AICoreNum": 8, "DdkVersion": "3.20.xx.xxx"},
    },
    "Hi3796CV300ES": {
        "AiCore": {"AICoreNum": 1, "DdkVersion": "5.10.xx.xxx"},
    },
    "Hi3796CV300CS": {
        "AiCore": {"AICoreNum": 1, "DdkVersion": "5.10.xx.xxx"},
    },
    "Hi3519AV200": {
        "AiCore": {"AICoreNum": 1, "DdkVersion": None},
    },
}


# pylint: disable=too-many-branches
def get_product_version(product):
    """
    deal with raw product. return valid product

    Parameters
    ----------
    product : str
        product name\

    Returns
    -------
    product : str
        1.1,1.3,etc
    """
    if isinstance(product, str):
        product = product.lower()
        versions = product.split(".")
        if len(versions) != 4 and len(versions) != 5:
            raise RuntimeError("Not support specify the product %s" % product)
        if product.startswith("3.5."):
            product = "3.5"
        elif product.startswith("3.3."):
            product = "3.3"
        elif product.startswith("1.1."):
            product = "1.1"
        elif product.startswith("1.2."):
            product = "1.2"
        elif product.startswith("1.3."):
            product = "1.3"
        elif product.startswith("1.31."):
            product = "1.3"
        elif product.startswith("1.33."):
            product = "1.3"
        elif product.startswith("1.60."):
            product = "1.60"
        elif product.startswith("2.10."):
            product = "2.10"
        elif product.startswith("2.11."):
            product = "2.11"
        elif product.startswith("2.3."):
            product = "2.3"
        elif product.startswith("5.10."):
            product = "5.10"
        elif product.startswith("3.20."):
            product = "3.20"
        elif product.startswith("2.11."):
            product = "2.11"
        else:
            raise RuntimeError("Not support specify the product %s" % product)
    else:
        raise RuntimeError("The Supported product type error")

    return product


# pylint: disable=invalid-name
def cceProduct(product):
    """
    dynamic load the product params

    Parameters
    ----------
    product : str
        product name
    """
    product = get_product_version(product)
    cur_cce_product_params = CceProductParams()
    cur_cce_product_params.cce_product = product

    soc_version = DDK_SOC_VERSION_MAP[product]["SOC_VERSION"]
    core_type = DDK_SOC_VERSION_MAP[product]["CORE_TYPE"]
    func = tvm.get_global_func("cce.product_init")
    func(soc_version, core_type, "", "")

    return cur_cce_product_params


# pylint: disable=invalid-name
def getValue(key):
    """
    call global func to get product value

    Parameters
        ----------
        key : str
            key
    """
    if "Buffer" in key:
        func = tvm.get_global_func("cce.product_conf_buffer")
        value = func(key)
        if value == -1:
            raise RuntimeError("Unsupported buffer name: %s" % key.split("_Buffer")[0])
        return value

    if "Compiler" in key:
        func = tvm.get_global_func("cce.product_conf_compiler")
        value = func(key)
        if value == "":
            raise RuntimeError("Unsupported compiler param: %s" % key.split("Compiler_")[1])
        return value

    if "Intrinsic" in key:
        func = tvm.get_global_func("cce.product_conf_intrinsic")
        value = func(key)
        if value == "":
            raise RuntimeError("Unsupported intrinsic: %s" % key.split("Intrinsic_")[1])
        return value

    if "Sid" in key:
        func = tvm.get_global_func("cce.product_conf_sid")
        value = func(key)
        if value == "":
            raise RuntimeError("Unsupported sid param: %s" % key.split("Sid_")[1])
        return value

    if "Device" in key:
        func = tvm.get_global_func("cce.product_conf_device")
        value = func(key)
        if value == -1:
            raise RuntimeError("Unsupported device param: %s" % key.split("Device_")[1])
        return value

    return None


# pylint: disable=useless-object-inheritance, bad-classmethod-argument, invalid-name
class CceProductParams(object):
    """
    define Cce Product Params class
    """
    __instance = None

    cce_product = None

    def __init__(self):
        pass

        # singletom pattern

    def __new__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = object.__new__(self, *args, **kwargs)
        return self.__instance

    def getParams(self, key):
        """
        get Cce Product Params info
        """
        if self.cce_product is None:
            raise RuntimeError("not set product info")

        value = getValue(key)

        # if product supports os
        if key == "Compiler_aicpu_support_os":
            # string to bool
            value = (value == "true")

        if key == "is_cloud":
            value = (self.cce_product == "1.60")

        return value

def set_status_check(bool_status):
    """
    call global func to set debug mode to
    add status special register check code
    to check if the compute overflow.

    Parameters
        ----------
        bool_status : boolean
            when true, the code will print the check code
    """
    if not isinstance(bool_status, bool):
        raise RuntimeError("The input value type must be boolean")

    func = tvm.get_global_func("cce.status_check")

    func(bool_status)

_VERSION_LIST = ["1.1", "1.60", "3.3", "3.5", "1.3", "1.31", "1.33",
                 "2.10", "5.10", "3.20", "2.11"]


def _check_ddk_version(product_name):
    if not isinstance(product_name, str):
        return False
    if product_name == "":
        return False

    versions = product_name.split(".")
    if len(versions) not in (4, 5):
        return False

    # noinspection PyBroadException
    try:
        major = int(versions[0])
        minor = int(versions[1])
    except Exception:
        return False

    if major > 9 or major < 1 or minor > 99 or minor < 0:
        return False

    return True


def _check_soc_version(soc_version, core_type):
    # check Soc_Vesion
    if not isinstance(soc_version, str):
        raise RuntimeError("Soc_Vesion type should be 'str', it is [%s]"
                           % type(soc_version))
    if soc_version not in SOC_VERSION_MAP:
        raise RuntimeError("Unsupported Soc_Vesion: %s" % soc_version)

    # check Core_Type
    if not isinstance(core_type, str):
        raise RuntimeError("Core_Type type should be 'str', it is [%s]"
                           % type(core_type))
    if core_type not in SOC_VERSION_MAP[soc_version]:
        raise RuntimeError("%s Unsupported Core_Type: %s"
                           % (soc_version, core_type))


def _check_and_get_aicore_num(soc_version, core_type, aicore_num):
    # check AICore_Num
    max_aicore_num = SOC_VERSION_MAP[soc_version][core_type]["AICoreNum"]
    if aicore_num in [None, "0", 0, ""]:
        aicore_num = ""
    elif isinstance(aicore_num, int):
        if not 0 < aicore_num <= max_aicore_num:
            raise RuntimeError("Unsupported AICore_Num: %s" % aicore_num)
        aicore_num = str(aicore_num)
    elif isinstance(aicore_num, str):
        try:
            check_num = int(aicore_num)
        except Exception:
            raise RuntimeError("Unsupported AICore_Num: %s" % aicore_num)
        if not 0 < check_num <= max_aicore_num:
            raise RuntimeError("Unsupported AICore_Num: %s" % aicore_num)
    else:
        raise RuntimeError("Unsupported AICore_Num: %s" % aicore_num)

    return aicore_num


def _check_and_get_l1_fusion(l1_fusion):
    # check l1_fusion
    if l1_fusion is None:
        l1_fusion = ""
    elif l1_fusion is True:
        l1_fusion = "true"
    elif l1_fusion is False:
        l1_fusion = "false"
    elif l1_fusion in ("True", "False", "TRUE", "FALSE", "true", "false", ""):
        l1_fusion = l1_fusion.lower()
    else:
        raise RuntimeError("Unsupported l1_fusion: %s" % l1_fusion)

    return l1_fusion


def te_set_version(soc_version, core_type="AiCore",
                   aicore_num=None, l1_fusion=None,
                   l2_mode="0", l2_fusion=None):
    """set version info

    Parameters
    ----------
    soc_version : str
        "Ascend310"/"Ascend910"/"Ascend610"/"Ascend620" ...
    core_type : str
        "AiCore" or "VectorCore"
    aicore_num: int
        example: 32
    l1_fusion: bool
        example: True/False

    Returns
    -------
    errmsg : str
        error message, 'success' for OK.
    """
    l1_fusion = _check_and_get_l1_fusion(l1_fusion)
    if l1_fusion == "true":
        set_L1_info("L1_fusion_enabled", True)
    elif l1_fusion == "false":
        set_L1_info("L1_fusion_enabled", False)

    if l2_fusion == "true":
        set_L1_info("L2_fusion_enabled", True)
    else:
        set_L1_info("L2_fusion_enabled", False)

    if _check_ddk_version(soc_version):
        cceProduct(soc_version)
        te_set_l2_mode(l2_mode)
        return

    if core_type in (None, ""):
        core_type = "AiCore"

    _check_soc_version(soc_version, core_type)
    aicore_num = _check_and_get_aicore_num(soc_version, core_type,
                                           aicore_num)
    func = tvm.get_global_func("cce.product_init")
    value = func(soc_version, core_type, aicore_num, l1_fusion)

    if value != "success":
        raise RuntimeError("te_set_version() return error.")

    ddk_ver = SOC_VERSION_MAP[soc_version][core_type]["DdkVersion"]
    product = get_product_version(ddk_ver)
    cur_cce_product_params = CceProductParams()
    cur_cce_product_params.cce_product = product
    te_set_l2_mode(l2_mode)


def te_set_l2_mode(l2_mode):
    """set l2 flag

    Parameters
    ----------
    l2_flag : int

    Returns
    -------
    succ_flag : boolean
    """
    func = tvm.get_global_func("cce.set_L2_status")
    return func(int(l2_mode))



SOC_VERSION = "SOC_VERSION"
AICORE_TYPE = "AICORE_TYPE"
CORE_NUM = "CORE_NUM"
UB_SIZE = "UB_SIZE"
L2_SIZE = "L2_SIZE"
L1_SIZE = "L1_SIZE"
CUBE_SIZE = "CUBE_SIZE"
L0A_SIZE = "L0A_SIZE"
L0B_SIZE = "L0B_SIZE"
L0C_SIZE = "L0C_SIZE"
SMASK_SIZE = "SMASK_SIZE"


def get_soc_spec(key):
    """
    call global func to get soc spec

    Parameters
        ----------
        key : str
            key
    """
    support_key = (SOC_VERSION, AICORE_TYPE, CORE_NUM, UB_SIZE,
                   L2_SIZE, L1_SIZE, CUBE_SIZE,
                   L0A_SIZE, L0B_SIZE, L0C_SIZE, SMASK_SIZE)
    if key not in support_key:
        raise RuntimeError("Unsupported Key Value of get_soc_spec(): %s" % key)

    func = tvm.get_global_func("cce.get_soc_spec")
    value = func(key)
    if value == "":
        raise RuntimeError("Unsupported Key Value of get_soc_spec(): %s" % key)

    str2int_list = (CORE_NUM, UB_SIZE, L2_SIZE, L1_SIZE,
                    L0A_SIZE, L0B_SIZE, L0C_SIZE, SMASK_SIZE)
    if key in str2int_list:
        try:
            value = int(value)
        except Exception:
            raise RuntimeError("return value %s is not 'int' type" % value)
    elif key in (CUBE_SIZE,):
        value_str_list = value.split(",")
        value_int_list = []
        for i in value_str_list:
            try:
                value_int_list.append(int(i))
            except Exception:
                raise RuntimeError("return value %s is not 'int' type" % value)
        value = value_int_list

    return value


@tvm.register_func("te.cce.get_product")
def get_product():
    """
    get product c++ code.

    Parameters
    ----------

    Returns
    -------
    value: device product.
        end of execution
    """

    return CceProductParams().cce_product


def api_check_support(intrinsic, dtype=""):
    """
    check if current chip support this api.

    Parameters
    ----------
    intrinsic : str, the intrinsic need to check
    dtype: str, optional args, if not empty, will check the dtype.
    Returns
    -------
    value: bool, True if chip contains such api, else return False

    """
    if not isinstance(intrinsic, str):
        raise RuntimeError("intrinsic type should be 'str', it is [%s]"
                           % type(intrinsic))
    if not isinstance(dtype, str):
        raise RuntimeError("dtype type should be 'str', it is [%s]"
                           % type(dtype))
    if intrinsic.startswith("tik."):
        return _deal_tik_api(intrinsic, dtype)

    if intrinsic.startswith("te.lang.cce."):
        return te.lang.cce.dsl_check_support(intrinsic, dtype)

    support_insn_cmd = get_insn_cmd()
    if intrinsic in support_insn_cmd:
        func = tvm.get_global_func("cce.api_check_support")
        value = func(intrinsic)
        if value == "True":
            return True
        else:
            return False

    return False


def intrinsic_check_support(intrinsic, dtype=""):
    """
    check if current chip support this intrinsic.

    Parameters
    ----------
    intrinsic : str, the intrinsic need to check
    dtype: str, optional args, if not empty, will check the dtype.
    Returns
    -------
    value: bool, True if chip contains such api, else return False

    """
    if not isinstance(intrinsic, str):
        raise RuntimeError("intrinsic type should be 'str', it is [%s]"
                           % type(intrinsic))
    if not intrinsic.startswith("Intrinsic_"):
        raise RuntimeError("intrinsic type should start with Intrinsic_")
    if not isinstance(dtype, str):
        raise RuntimeError("dtype type should be 'str', it is [%s]"
                           % type(dtype))
    func = tvm.get_global_func("cce.intrinsic_check_support")
    value = func(intrinsic, dtype)
    if value in (-1, ""):
        raise RuntimeError("Unsupported Key Value"
                           " of get_soc_spec(): %s" % intrinsic)
    if value == "True":
        return True
    return False


def _deal_tik_api(intrinsic, dtype):
    """
    deal tik api

    Parameters
    ----------
    intrinsic : str, the intrinsic need to check
    dtype: str, optional args, if not empty, will check the dtype.
    Returns
    -------
    value: bool, True if chip contains such api, else return False

    """
    tik_api_instr = intrinsic[4:]
    soc_total_version = get_soc_spec("SOC_VERSION") + \
                        get_soc_spec("AICORE_TYPE")
    if tik_api_instr in ONLY_TIK_API_MAP:
        if soc_total_version in ONLY_TIK_API_MAP[tik_api_instr]:
            if dtype == "":
                return True
            return dtype in ONLY_TIK_API_MAP[tik_api_instr][soc_total_version]
        else:
            return False
    if tik_api_instr in TRANS_TIK_API_TO_INSTR_MAP:
        tik_api_instr = TRANS_TIK_API_TO_INSTR_MAP[tik_api_instr]
    return intrinsic_check_support("Intrinsic_" + tik_api_instr, dtype)

_init_api("te.platform.cce_conf")
