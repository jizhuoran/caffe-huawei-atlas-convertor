"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     get_soc_name.py
DESC:     get_soc_name file for tik
CREATED:  2020-01-10 19:02:50
MODIFIED: 2020-01-10 19:02:50
"""

from te.platform.cce_conf import get_soc_spec

def get_soc_name():
    """
    get current soc's name
    :return: SOC_VERSION
    """
    return get_soc_spec("SOC_VERSION")

def get_soc_core_type():
    """
    get current soc's version
    :return: AICORE_TYPE
    """
    return get_soc_spec("AICORE_TYPE")
