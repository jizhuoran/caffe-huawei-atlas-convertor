"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_conf.py
DESC:     configuration of tik
CREATED:  2019-04-18 18:53:42
MODIFIED: 2019-07-25 09:59:30
"""
from te.platform.cce_conf import te_set_version

from ..tik_lib.tik_source_info import TikSourceInfo
from ..tik_lib.tik_conf_ import SetProductFlag


def set_product_version(version):
    """set version info

    Parameters
    ----------
    version : str
        product name
        format: <major>.<middle>.<minor>.<point>
        major: 1xx or 2xx or 3xx
    """
    # need TikSourceInfo() init function to set source_info None first
    TikSourceInfo().register_source_info()
    te_set_version(version)
    SetProductFlag.is_set_product_version = True
    TikSourceInfo.clear_source_info()


def unset_product_version():
    """unset SetProductFlag to false

    """
    # need TikSourceInfo() init function to set source_info None first
    TikSourceInfo().register_source_info()
    SetProductFlag.is_set_product_version = False
    TikSourceInfo.clear_source_info()
