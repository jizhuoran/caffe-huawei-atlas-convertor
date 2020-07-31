#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Log utils for ops fusion
"""

import sys
import traceback


def modify_except_msg(evalue):
    """
    modify except msg 'No space left on device' to 'No space left on disk'
    """
    if evalue and hasattr(evalue, 'strerror') and \
       evalue.strerror == 'No space left on device':
        evalue.strerror = 'No space left on disk'


def get_py_exception_str(etype, value, tback):
    """
    return python exception string, calling from C for error log printing
    """
    modify_except_msg(value)
    exc_list = traceback.format_exception(etype, value, tback)
    msg = "".join(exc_list)
    if msg.find('#Conv2DBackpropInput only support#') != -1:
        msg = "".join(value.args)
    return msg


def except_msg():
    """
    Return exception msg.
    """
    etype, evalue, tback = sys.exc_info()
    return get_py_exception_str(etype, evalue, tback)
