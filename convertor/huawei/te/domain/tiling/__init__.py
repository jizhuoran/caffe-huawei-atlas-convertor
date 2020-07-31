#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

TBE tiling init
"""
#pylint: disable=import-error
import numpy as np
from te import platform as cce
from te import tvm
from te.tvm import _api_internal
from te.tvm import ir_pass
from tvm._ffi.function import _init_api
