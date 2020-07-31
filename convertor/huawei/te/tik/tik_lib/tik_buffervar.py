"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_buffervar.py
DESC:     tik buffervar
CREATED:  2019-12-06 12:32:13
MODIFIED: 2020-01-08 20:01:45
"""
# disabling:
# E1101: no-member

from te.tvm import api as _api
from te.tvm import make as _make
from te.tvm import TVMType
from te.tvm.ir_builder import BufferVar
from .tik_check_util import TikCheckUtil


class TikBufferVar(BufferVar):
    """Buffer variable with content type, makes load store easily.

    Do not create it directly, create use IRBuilder.

    """
    def __setitem__(self, index, value):
        # pylint cannot recognize C++ member so disable E1101
        value = _api.convert(value)
        if value.dtype != self._content_type:
            TikCheckUtil.raise_error(
                "data type does not match content type %s vs %s" % (
                    value.dtype, self._content_type), exception_type=ValueError)
        tvm_type = TVMType(self._content_type)
        if tvm_type.lanes > 1:
            index = _make.Ramp(index * tvm_type.lanes,  # pylint: disable=E1101
                               1, tvm_type.lanes)
        stmt_node = _make.Store(self._buffer_var,  # pylint: disable=E1101
                                value, index)
        self._builder.source_info.set_node_loc(stmt_node)
        self._builder.emit(stmt_node)

        self._builder.total_ir_lines += 1
        if self._builder.is_double_buffer_for_loop:
            self._builder.double_buffer_ir_num += 1
