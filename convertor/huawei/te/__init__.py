#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use 
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

"""
from __future__ import absolute_import as _abs

import sys
import os
from .version import version as __version__

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, BASE_DIR)


def te_sysconfig():
    """
    set RLIMIT_NPROC and openblas thread number
    """
    try:
        import resource as res
        rlimit = res.getrlimit(res.RLIMIT_NPROC)
        te_nproc = 40960
        if rlimit[0] < te_nproc:
            nproc = te_nproc if te_nproc < rlimit[1] else rlimit[1]
            res.setrlimit(res.RLIMIT_NPROC, (nproc, rlimit[1]))

        # Not using openblas currently.
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    except:                     # pylint: disable=bare-except
        pass


te_sysconfig()

import tvm
from tvm import _ffi, contrib, hybrid

sys.path.pop(0)

class TEMetaPathFinder(object):
    """
    TBE MetaPathFinder
    """
    def __init__(self):
        pass

    def find_module(self, fullname, path=None):
        """
        find_module
        """
        # for pylint, reserve argument
        path = path
        if fullname == "topi.cce.cce_extended_op_build":
            print("te error: 'topi.cce.cce_extended_op_build' has been deprecated, please using "
                  "'te.platform.cce_conf.te_op_build' instead ")

        if fullname.startswith("te.tvm"):
            rname = fullname[3:]
            if rname in sys.modules:
                return TEMetaPathLoader(rname)
        return None


class TEMetaPathLoader(object):
    """
    TBE MetaPathLoader
    """
    def __init__(self, target_name):
        self.__target_module = sys.modules[target_name]

    def load_module(self, fullname):
        """
        load_module
        """
        sys.modules[fullname] = self.__target_module
        return self.__target_module


sys.meta_path.insert(0, TEMetaPathFinder())

# before, the auto_schedule belong to topi.generic module(tvm code)
# now, auto_schedule Stripped from generic module as single func
# Dynamic loading the auto_schedule to the generic module for compatibility
__import__('topi.generic.cce')

AUTO_SCH_MODULE = sys.modules["topi.generic.cce"]
FUNC = getattr(AUTO_SCH_MODULE, "auto_schedule")

GENERIC_MODULE = sys.modules["topi.generic"]
setattr(GENERIC_MODULE, "auto_schedule", FUNC)

# before, the topi.cce module belong to topi module(tvm code)
# now, topi.cce Stripped from topi module as single module
# Dynamic loading the topi.cce to the topi module for compatibility
__import__('topi.cce')
TOPI_CCE_MODULE = sys.modules["topi.cce"]
TOPI_MODULE = sys.modules["topi"]
setattr(TOPI_MODULE, "cce", TOPI_CCE_MODULE)


def msg_excepthook(etype, value, tback):
    """
    excepthook to modify some confusing exception msg
    """
    if value and hasattr(value, 'strerror') and \
       value.strerror == 'No space left on device':
        value.strerror = 'No space left on disk'

    sys.__excepthook__(etype, value, tback)


sys.excepthook = msg_excepthook
