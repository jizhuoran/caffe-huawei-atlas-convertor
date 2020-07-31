"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_source_info.py
DESC:     handle source information
CREATED:  2020-1-21 20:12:13
MODIFIED: 2020-2-04 14:03:16
"""
# disabling:
# W0212: protected-access

import sys
import copy
from functools import wraps

from te import tvm

from .tik_check_util import ERROR_MSG_LEVEL, TIK_ERROR_MSG, \
    clear_tik_error_msg, TIK_CONTROL
from .tik_params import CUR_FRAME_IDX


def source_info_decorator(depth=1):
    """bind this decorator with func that need register source info

    Parameters
    ----------
    depth : stack depth

    Returns
    -------
    function
    """
    def get_source_info_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """bind this decorator with func that need register source info"""
            if TIK_CONTROL.is_user_call:
                TikSourceInfo.register_source_info(depth)
                TikSourceInfo.set_not_user_call()
                f_return = func(*args, **kwargs)
                TikSourceInfo.clear_source_info()
                TikSourceInfo.set_is_user_call()
            else:
                f_return = func(*args, **kwargs)
            return f_return
        return wrapper
    return get_source_info_decorator


def get_loc():
    """get location corresponding to node"""
    if TIK_ERROR_MSG.api_source_info is None:
        if TIK_CONTROL.if_loc_invalid_raise_error:
            raise RuntimeError("location is invalid, please check!!!\n"
                               "There are two reasons for the error:\n"
                               "If it is an open interface, please "
                               "register source at the beginning;\n"
                               "If it is an unopened interface,"
                               " please correct this mistake;")
        print("WARNING: location is invalid, please check!!!")
        return ["invalid", "invalid"]
    loc = [TIK_ERROR_MSG.api_source_info[CUR_FRAME_IDX].get("filename"),
           str(TIK_ERROR_MSG.api_source_info[CUR_FRAME_IDX].get("line_no"))]
    return loc


def get_frame_info(frame):
    """get particular information from stack frame

    Parameters
    ----------
    frame : frame object from the call stack
    """
    filename = frame.f_code.co_filename
    return {"filename": filename, "line_no": frame.f_lineno,
            "function": frame.f_code.co_name,
            "sym_table": copy.copy(frame.f_locals)}


def stack(depth=0):
    """get stack information after depth

    Parameters
    ----------
    depth : stack depth

    Return
    ----------
    list of frame info
    """
    frame = sys._getframe(depth + 1)  # pylint: disable=W0212
    frame_list = []
    while frame:
        frame_list.append(get_frame_info(frame))
        frame = frame.f_back
    return frame_list


def most_recent_traceback():
    """get system's exc info that contains type, exception, traceback"""
    trace = sys.exc_info()[2]
    trace_list = []
    while trace:
        trace_list.append({"filename": trace.tb_frame.f_code.co_filename,
                           "line_no": trace.tb_lineno,
                           "function": trace.tb_frame.f_code.co_name})
        trace = trace.tb_next
    trace_list.reverse()
    return trace_list


def current_frame(depth=0):
    """get current frame information

    Parameters
    ----------
    depth : stack depth

    Return
    ----------
    current frame information
    """
    return [get_frame_info(sys._getframe(depth + 1))]  # pylint: disable=W0212


class TikSourceInfo():
    """
    handle tik api source information
    """
    def __init__(self):
        TIK_ERROR_MSG.api_source_info = None
        TIK_CONTROL.is_user_call = True
        if not hasattr(TIK_CONTROL, "if_loc_invalid_raise_error"):
            TIK_CONTROL.if_loc_invalid_raise_error = True
        if not hasattr(TIK_ERROR_MSG, "location_manager"):
            TIK_ERROR_MSG.location_manager = tvm.LocationManager()
        if not hasattr(ERROR_MSG_LEVEL, "err_msg_level"):
            ERROR_MSG_LEVEL.err_msg_level = 0

    @staticmethod
    def register_source_info(depth=1, source_info=None):
        """register source information

        Parameters
        ----------
        depth : stack depth
        source_info : source information

        Returns
        -------
        None
        """
        if TIK_ERROR_MSG.api_source_info is not None:
            raise RuntimeError(
                "please clear source info before register, previous info: "
                "\n{}".format(TIK_ERROR_MSG.api_source_info[CUR_FRAME_IDX]))
        # set debug module info or for_range/if_scope/else_scope
        if source_info is not None:
            TIK_ERROR_MSG.api_source_info = source_info
            return
        depth += 1
        if ERROR_MSG_LEVEL.err_msg_level == 0:
            TIK_ERROR_MSG.api_source_info = stack(depth)
        else:
            TIK_ERROR_MSG.api_source_info = current_frame(depth)

    @staticmethod
    def set_not_user_call():
        """set is_user_call to false,
           indicating that the function is not called by user next
           Therefore, source info will not be recorded repeatedly."""
        TIK_CONTROL.is_user_call = False

    @staticmethod
    def set_is_user_call():
        """set is_user_call to True,
           indicating that the function is called by user next"""
        TIK_CONTROL.is_user_call = True

    @staticmethod
    def clear_source_info():
        """clear source information"""
        TIK_ERROR_MSG.api_source_info = None

    @staticmethod
    def get_source_info():
        """get source information"""
        return TIK_ERROR_MSG.api_source_info

    @staticmethod
    def set_node_loc(tvm_node, loc=None):
        """set location for node"""
        if loc is None:
            TIK_ERROR_MSG.location_manager.set_loc(tvm_node, get_loc())
        else:
            TIK_ERROR_MSG.location_manager.set_loc(tvm_node, loc)

    @staticmethod
    def get_node_loc(tvm_node):
        """get location of node"""
        return TIK_ERROR_MSG.location_manager.get_loc(tvm_node)

    @staticmethod
    def update_node_loc():
        """Update correspondence between node and location to tvm"""
        TIK_ERROR_MSG.location_manager.update_loc()

    @staticmethod
    def end_and_clear():
        """when tik end, clear TIK_ERROR_MSG"""
        clear_tik_error_msg()
