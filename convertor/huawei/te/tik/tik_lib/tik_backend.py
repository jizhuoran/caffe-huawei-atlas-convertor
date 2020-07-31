"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_backend.py
DESC:     provide backend api for tik
CREATED:  2019-11-29 11:30:21
MODIFIED: 2019-11-29 15:30:33
"""
# disabling:
# R0913: too-many-arguments

from te.tvm import string_types
from te.tvm.schedule import convert, Buffer
from te.tvm._api_internal import _TikBufferAccessPtr
from .tik_check_util import TikCheckUtil


def tik_access_ptr(buffer, access_mask,  # pylint: disable=R0913
                   cast_coeff, ptr_type="handle",
                   content_lanes=1, offset=0):
    """

    :param buffer: tvm.schedule.Buffer, we will call Buffer.access_ptr to get
                tensor's address
    :param access_mask: 1(01) means read, 2(10) means write, 3(11) means r/w
    :param cast_coeff: cast_coeff = cast_type_bit / ori_type_bit
    :param ptr_type: the type of ptr
    :param content_lanes: the lanes of content
    :param offset: the offset of valid data. offset = tensor.offset + offset
    :return: call node
    """
    if isinstance(access_mask, string_types):
        mask = 0
        for value in access_mask:
            if value == "r":
                mask = mask | Buffer.READ
            elif value == "w":
                mask = mask | Buffer.WRITE
            else:
                TikCheckUtil.raise_error("Unknown access_mask %s" % access_mask,
                                         exception_type=ValueError)
        access_mask = mask
    offset = convert(offset)
    # for performance optimization, move offset and extent check to backend.
    # _TikBufferAccessPtr return expr node that create in tvm
    expr_node = _TikBufferAccessPtr(
        buffer, access_mask, ptr_type, content_lanes, offset, cast_coeff)
    return expr_node
