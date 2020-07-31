"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     codemapping.py
DESC:     To map code
CREATED:  2019-7-04 20:12:13
MODIFIED: 2019-7-17 14.03.16
"""

import os
import sys
import copy

from te.tik.tik_lib.tik_check_util import TIK_ERROR_MSG, ERROR_MSG_LEVEL
from te.tik.tik_lib.tik_params import INSPECT_RANGE, MIN_START_LINE_NO
from te.tik.tik_lib.tik_source_info import stack, current_frame


def lookup_module(filename):
    """Helper function for break/clear parsing -- may be overridden.

    lookupmodule() translates (possibly incomplete) file or module name
    into an absolute file name.
    """
    if filename is None:
        return None

    if os.path.isabs(filename) and os.path.exists(filename):
        return os.path.realpath(filename)
    _, ext = os.path.splitext(filename)
    filename_ = filename
    if ext == '':
        filename_ = filename + '.py'
    if os.path.isabs(filename_):
        return os.path.realpath(filename_)
    for dir_name in sys.path:
        if dir_name is not None:
            while os.path.islink(dir_name):
                dir_name = os.readlink(dir_name)
            fullname = os.path.join(dir_name, filename_)
            if os.path.exists(fullname):
                return fullname
    return filename_


def canonic(filename):
    """to canonic file path"""
    if filename is None:
        return ''

    if filename == ("<" + filename[1:-1] + ">"):
        return filename
    canonic_ = os.path.realpath(filename)
    canonic_ = os.path.normcase(canonic_)
    return canonic_


MIN_FRAME_INFO_LEN = 5
SOURCE_FILE_IDX = 1
LINE_NO_IDX = 2
FN_NAME_IDX = 3
LINES_IDX = 4
TARGET_IDX = 5
SYM_TABLE_IDX = 0


class CallerContext():
    """
    Function stack frame info
    """

    def __init__(self, frame_info):
        # frame_info length should greater than 5
        if len(frame_info) < MIN_FRAME_INFO_LEN:
            raise RuntimeError("frame_info length should greater than 5")
        self.source_file = canonic(lookup_module(frame_info[SOURCE_FILE_IDX]))
        if self.source_file.endswith('.pyc'):
            self.source_file = self.source_file[:-2]
        self.line_no = frame_info[LINE_NO_IDX]
        self.fn_name = frame_info[FN_NAME_IDX]
        self.lines = frame_info[LINES_IDX]
        self.target_idx = frame_info[TARGET_IDX]
        self.sym_table = copy.copy(frame_info[SYM_TABLE_IDX].f_locals)

    def __add__(self, rhs):
        return self.__str__() + rhs

    def __str__(self):
        start_lineno = self.line_no - INSPECT_RANGE // 2
        start_lineno = max(start_lineno, MIN_START_LINE_NO)
        print_lines = []
        for i, line in enumerate(self.lines):
            text = repr(i + start_lineno).rjust(3)
            if i == self.target_idx:
                text = text + ' -> '
            else:
                text = text + '    '
            text = text + line
            print_lines.append(text)
        return ''.join(print_lines).rstrip()


def get_caller_context(depth=None, **kwarg):
    """get current stack frame information"""
    if TIK_ERROR_MSG.api_source_info is not None:
        return TIK_ERROR_MSG.api_source_info
    if depth is None:
        raise RuntimeError("There are two reasons for the error:\n"
                           "If it is called by the user, please register source"
                           " info before entering decorators;\n"
                           "If it is an internal call, please specify "
                           "the stack depth;")
    additional_stack = kwarg.get('stack_depth', 0)
    depth += additional_stack
    if ERROR_MSG_LEVEL.err_msg_level == 0:
        caller = stack(depth)
    else:
        caller = current_frame(depth)
    return caller
