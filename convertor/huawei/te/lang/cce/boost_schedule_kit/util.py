"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

"""


def ceil_div(number_m, number_n):
    """
    Calculate the ceil result of division

    Parameters:
    ----------
    number_m : dividend, <type 'int'>

    number_n : divisor, <type 'int'>

    Returns : Result of the compute
    ----------
    """
    return (number_m + number_n - 1) // number_n


class Compare:
    """
    Compare: Compare two tensor's shape.

    Functions
    ----------
    __init__ : Initialization.

    compare : The comparison function.

    """
    EQUAL = 0
    LESS_EQ = 1
    GREATE_EQ = 2
    MISC = 256

    def __init__(self):
        pass

    @staticmethod
    def compare(lhs, rhs):
        """
        Compare two tensor's shape.
        If each number in AShape is less than Bshape, means LESS_EQ.
        If each number in AShape is greater than Bshape, means GREATE_EQ.
        If each number in AShape is equal to Bshape, means GREATE_EQ.
        Else MISC.

        Parameters:
        ----------
        lhs : left tensor's shape, list or tuple

        rhs : rigth tensor's shape, list or tuple

        Returns : Status, a number represent the comparison result.
        ----------
        """
        if lhs == rhs:
            return Compare.EQUAL
        flag_less_equal = True
        flag_great_equal = True
        for part_extent, full_extent in zip(lhs, rhs):
            if part_extent is None:
                continue
            if part_extent > full_extent:
                flag_less_equal = False
            elif part_extent < full_extent:
                flag_great_equal = False
            else:
                pass
        status = None
        if flag_less_equal and flag_great_equal:
            # means equal
            status = Compare.EQUAL
        elif flag_less_equal and not flag_great_equal:
            status = Compare.LESS_EQ
        elif flag_great_equal and not flag_less_equal:
            status = Compare.GREATE_EQ
        else:
            status = Compare.MISC

        return status


def check_common_tiling(tiling):
    """
    Check tiling.
    If not valid, Raise Error

    Parameters:
    ----------
    tiling : tiling from auto_tiling.

    Returns : None
    ----------
    """
    available_feature = {
        "AL1_shape": [3],
        "BL1_shape": [3],
        "AL0_matrix": [5],
        "BL0_matrix": [5],
        "CL0_matrix": [5],
        "CUB_matrix": [5],
        "AUB_shape": [3],
        "BUB_shape": [3],
        "block_dim": [3],
        "cout_bef_batch_flag": [1],
        "A_overhead_opt_flag": [1],
        "B_overhead_opt_flag": [1],
        "manual_pingpong_buffer": [9]
    }
    pbuffer_detail = ["AUB_pbuffer",
                      "BUB_pbuffer",
                      "AL1_pbuffer",
                      "BL1_pbuffer",
                      "AL0_pbuffer",
                      "BL0_pbuffer",
                      "CL0_pbuffer",
                      "CUB_pbuffer",
                      "UBG_pbuffer"]
    for key, limits in available_feature.items():
        feature_spec = tiling.get(key)
        if feature_spec is None:
            # special value:None
            continue
        if isinstance(feature_spec, (list, dict)):
            if not feature_spec:
                continue
            feature_size = len(feature_spec)
            if feature_size != limits[0]:
                raise RuntimeError("feature %s dismatched, "
                                   "len is expected %d while it is %s"
                                   % (key, limits[0], feature_size))
        else:  # not a list
            if limits[0] != 1:
                raise RuntimeError("feature %s dismatched" % (key))
    # check pbuffer
    pbuffer_feature = tiling.get("manual_pingpong_buffer")
    if pbuffer_feature is None:
        raise RuntimeError("need tiling feature:manual_pingpong_buffer")
    for key in pbuffer_detail:
        if pbuffer_feature.get(key) is None:
            raise RuntimeError(
                "need tiling feature:manual_pingpong_buffer.%s" % (key))


def enhance_check_tiling(check_func):
    """
    Enhance tiling check function

    Parameters:
    ----------
    check_func : Function of enhanced check rules.

    Returns : A enhanced function.
    ----------
    """
    if check_func.__code__.co_argcount != 1:
        raise RuntimeError("check function must have 1 input")

    def _check_tiling(tiling):
        check_common_tiling(tiling)
        return check_func(tiling)
    return _check_tiling
