"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_check_util.py
DESC:     check util for api
CREATED:  2020-1-21 20:12:13
MODIFIED: 2020-2-04 14:03:16
"""

import re
import threading
import linecache
import os

from te import tvm
from .tik_params import INSPECT_RANGE, MIN_START_LINE_NO, CUR_FRAME_IDX


TIK_ERROR_MSG = threading.local()
TIK_ERROR_MSG.api_source_info = None
TIK_ERROR_MSG.location_manager = tvm.LocationManager()
TIK_CONTROL = threading.local()
TIK_CONTROL.is_user_call = True
# when api_source_info is None, node loc could be invalid,
# set "True" means raise RuntimeError, otherwise warning.
TIK_CONTROL.if_loc_invalid_raise_error = True

ERROR_MSG_LEVEL = threading.local()
ERROR_MSG_LEVEL.err_msg_level = 0


def canonic(filename):
    """to canonic file path"""
    if filename is None:
        return ''

    if filename == ("<" + filename[1:-1] + ">"):
        return filename
    canonic_ = os.path.realpath(filename)
    canonic_ = os.path.normcase(canonic_)
    return canonic_


def read_context_from_file(filename, line_no, context_len=INSPECT_RANGE):
    """read code and context according to file path

    Parameters
    ----------
    filename : file absolutely path
    line_no : line number corresponding to code
    context_len : code context len

    Returns
    -------
    code and context
    """
    start_lineno = line_no - context_len // 2
    start_lineno = max(start_lineno, MIN_START_LINE_NO)
    print_lines = []
    for i in range(context_len):
        code = linecache.getline(filename, start_lineno + i)
        if code == "":
            break
        text = repr(i + start_lineno).rjust(3)
        if start_lineno + i == line_no:
            text = text + ' -> '
        else:
            text = text + '    '
        text = text + code
        print_lines.append(text)
    return ''.join(print_lines).rstrip() + "\n"


def get_context_msg(source_file, line_no, msg):
    """get context msg list contains error, file path and code context

    Parameters
    ----------
    source_file : file absolutely path
    line_no : line number corresponding to code
    msg : error msg

    Returns
    -------
    context msg list
    """
    msg_list = ["Error: {}".format(msg),
                "File path: {}, line {}".format(
                    canonic(source_file), line_no),
                "The context code cause the exception is:",
                read_context_from_file(source_file, line_no)]
    return msg_list


def get_traceback_msg(stack_list):
    """get traceback msg list according to stack

    Parameters
    ----------
    stack_list : user call stack

    Returns
    -------
    traceback msg list
    """
    msg_list = ["Traceback:"]
    stack_list.reverse()
    for info in stack_list:
        msg_list.append("  File {}, line {}, in {}\n {}".format(
            canonic(info["filename"]), info["line_no"], info["function"],
            linecache.getline(info["filename"], info["line_no"]).rstrip("\n")))
    return msg_list


def clear_tik_error_msg():
    """clear TIK_ERROR_MSG"""
    TIK_ERROR_MSG.api_source_info = None
    TIK_CONTROL.is_user_call = True
    TIK_ERROR_MSG.location_manager = tvm.LocationManager()


def print_error_msg(msg, exception_type=RuntimeError):
    """cause exception message according to source info and err_msg_level

    Parameters
    ----------
    msg : exception msg
    exception_type : type of exception, such as ValueError

    Returns
    -------
    None
    """
    if TIK_ERROR_MSG.api_source_info is None:
        raise RuntimeError("please register source info first")

    context_msg_list = get_context_msg(
        TIK_ERROR_MSG.api_source_info[CUR_FRAME_IDX].get("filename"),
        TIK_ERROR_MSG.api_source_info[CUR_FRAME_IDX].get("line_no"), msg)
    print("\n".join(context_msg_list))
    if ERROR_MSG_LEVEL.err_msg_level == 1:
        # clear TIK_ERROR_MSG before exit
        clear_tik_error_msg()
        raise exception_type(msg)
    traceback_msg_list = get_traceback_msg(TIK_ERROR_MSG.api_source_info)
    # clear TIK_ERROR_MSG before exit
    clear_tik_error_msg()
    raise SystemExit("\n".join(traceback_msg_list))


class TikCheckUtil():
    """Provide check util function"""

    @staticmethod
    def check_type_match(var, var_type, msg="Variable's type error"):
        """
        check var's type if correct
        :param var: variable
        :param var_type: var's correct type
        :param msg: exception msg
        :return: None. If type not match, raise error.
        """
        if not isinstance(var, var_type):
            print_error_msg(msg)

    @staticmethod
    def check_in_range(var, var_range, msg="Variable out of range"):
        """
        check var if in correct range
        :param var: variable
        :param var_range: var's correct range
        :param msg: exception msg
        :return: None. If not in range, raise error.
        """
        if var not in var_range:
            print_error_msg(msg)

    @staticmethod
    def check_equality(var1, var2,
                       msg="Variables is not equal to each other"):
        """
        check if var1 == var2
        :param var1: variable1
        :param var2: variable2
        :param msg: exception msg
        :return: None. If var1 != var2, raise error.
        """
        if var1 != var2:
            print_error_msg(msg)

    @staticmethod
    def check_not_equality(var1, var2,
                           msg="Variables is equal to each other"):
        """
        check if var1 != var2
        :param var1: variable1
        :param var2: variable2
        :param msg: exception msg
        :return: None. If var1 != var2, raise error.
        """
        if var1 == var2:
            print_error_msg(msg)

    @staticmethod
    def check_ge(var1, var2, msg="Variable1 should be more than variable2"):
        """
        check if var1 < var2
        :param var1: variable1
        :param var2: variable2
        :param msg: exception msg
        :return: None, If var1 < var2, raise error
        """
        if var1 < var2:
            print_error_msg(msg)

    @staticmethod
    def check_le(var1, var2, msg="Variable1 should be less than variable2"):
        """
        check if var1 > var2
        :param var1: variable1
        :param var2: variable2
        :param msg: exception msg
        :return: None, If var1 > var2, raise error
        """
        if var1 > var2:
            print_error_msg(msg)

    @staticmethod
    def check_is(var1, var2, msg="Variables is not the other one"):
        """
        check if var1 is var2
        :param var1: variable1
        :param var2: variable2
        :param msg: exception msg
        :return: None. If var1 is not var2, raise error.
        """
        if var1 is not var2:
            print_error_msg(msg)

    @staticmethod
    def check_not_is(var1, var2=None, msg="Variables is None"):
        """
        check if var1 is not var2
        :param var1: variable1
        :param var2: variable2
        :param msg: exception msg
        :return: None. If var1 is var2, raise error.
        """
        if var1 is var2:
            print_error_msg(msg)

    @staticmethod
    def check_name_str_valid(name):
        """
        check name is str and only contains 0-9,a-z,A-Z,_
        :param name:
        :return: None. If name not match, raise error.
        """
        TikCheckUtil.check_type_match(name, str, "name must be str")
        if not re.match("^[0-9A-Za-z_]+$", name):
            print_error_msg("name should only contain 0-9, a-z, A-Z, _")

    @staticmethod
    def check_not_contains(var1, var2,
                           msg="Variables is in the other one"):
        """
        check if var1 is in var2
        :param var1: variable1
        :param var2: variable2
        :param msg: show message
        :return: None. If var1 is not in var2, raise error.
        """
        if var1 in var2:
            print_error_msg(msg)

    @staticmethod
    def raise_error(msg, exception_type=RuntimeError):
        """raise exception directly

        Parameters
        ----------
        exception_type : type of exception, such as ValueError
        msg : exception msg

        Returns
        -------
        None
        """
        print_error_msg(msg, exception_type)
