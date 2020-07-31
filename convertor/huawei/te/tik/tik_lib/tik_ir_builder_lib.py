"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_ir_builder.py
DESC:     Developer API of IR node builder make function for TIK.
CREATED:  2019-08-12 18:53:42
MODIFIED: 2019-08-12 21:29:18
"""
# disabling:
# R0903: too-few-public-methods
# R0913: too-many-arguments
# E1101: no-member
# R0914: too-many-locals
# W0221: arguments-differ
# R0902: too-many-instance-attributes
# disable it because this file consist of vector scatter instruction
from __future__ import absolute_import as _abs

from te.tvm.ir_builder import WithScope
from te.tik.tik_lib.tik_expr import Expr
from te.platform.cce_params import scope_cbuf, scope_ubuf, scope_ca, scope_cb, \
    scope_cc, scope_smask
from ..api.tik_tensor import Tensor
from ..api.tik_scalar import Scalar
from .tik_check_util import TikCheckUtil

_LAST_ELEMENT = -1

# block num min should greater than 0, less than 65536
_MIN_BLOCK_NUM = 1
_MAX_BLOCK_NUM = 65536

_DEFAULT_THREAD_BLOCK_NUM = 1
_MIN_NIDX = 3
_DEVICE_API = 0

# for type id value
_SERIAL_FOR_TYPE_ID = 0


class TikWithScope(WithScope):  # pylint: disable=R0903
    """
    Auxiliary scope  with
    Inherit from WithScope of ir_builder
    """
    def __init__(self, enter_value, exit_cb, source_info):

        super(TikWithScope, self).__init__(enter_value, exit_cb)
        if isinstance(enter_value, (tuple, list)):
            self._enter_value = [Expr(value) for value in enter_value]
        else:
            self._enter_value = Expr(enter_value)
        self._exit_cb = exit_cb
        self.debug_hint = None
        self.debug_limit = None
        self.source_info = source_info


class ScopeBufferManager():
    """
    Scope buffer manager.
    Scope:for_range,if_scope.else_scope.new_stmt_scope
    """
    def __init__(self):
        self.buffer_map = {
            scope_cbuf: {},
            scope_ubuf: {},
            scope_ca: {},
            scope_cb: {},
            scope_cc: {},
            scope_smask: {}
        }
        self.tensor_list = []

    def __check_tensor(self, tensor):
        TikCheckUtil.check_type_match(
            tensor, Tensor,
            "input tensor(%s) should be Tensor" % str(type(tensor)))
        TikCheckUtil.check_in_range(
            tensor.scope, self.buffer_map,
            "input tensor's scope should be scope_cbuf, scope_ubuf, scope_ca, "
            "scope_cb, scope_cc or scope_smask")

    def add_tensor(self, tensor, size):
        """
        new Tensor application  in this scope

        Parameters
        ----------
        tensor: Tensor
             Tensor application

        size : int
             Tensor used buffer size

        Returns
        -------
        return:NO return
        """
        self.__check_tensor(tensor)
        self.buffer_map[tensor.scope][tensor] = size
        self.tensor_list.append(tensor)

    def del_tensor(self, tensor):
        """
        when the scope lifecycle is in the end, tensor should be released.

        Parameters
        ----------
        tensor: Tensor
              tensor should be released in the end of the scope lifecycle

        Returns
        -------
        return:NO return
        """
        self.__check_tensor(tensor)
        if self.buffer_map[tensor.scope].get(tensor) is None:
            RuntimeError("Tensor is not in this scope")
        del self.buffer_map[tensor.scope][tensor]
        tensor.disable_tensor()

    def get_scope_buffer_used(self):
        """
        Counting tensors size in this scope

        Parameters
        ----------
        parameters:No parameters

        -------
        return: map
        buffer_count: L0A/B/C,L1,UB buffer used size in the scope.
        """
        buffer_count = {scope_cbuf: 0, scope_ubuf: 0, scope_ca: 0, scope_cb: 0,
                        scope_cc: 0, scope_smask: 0}
        for buffer_scope in self.buffer_map:
            for tensor_size in self.buffer_map[buffer_scope].values():
                buffer_count[buffer_scope] += tensor_size
        return buffer_count


class CodeBufferManager():
    """
    when tensor is applied, tensor is managed in this class
    """
    def __init__(self):
        self.scope_stack = [ScopeBufferManager()]
        self.dprofile = None
        self.total_buffer = None
        self.double_buffer_enable_ = 0
        self.double_buffer_thread_num = 2

    def new_scope(self):
        """
        when new scope is applied, the new scope should be managed
        Parameters
        ----------
        parameters:No parameters

        -------
        return:NO return
        """
        self.scope_stack.append(ScopeBufferManager())

    def del_scope(self):
        """
        when lifecycle of the scope is in the end, this scope should be deleted.
        Parameters
        ----------
        parameters:No parameters

        -------
        return:NO return
        """
        del_scope_buffer = self.scope_stack.pop()
        for del_buffer in del_scope_buffer.tensor_list:
            del_scope_buffer.del_tensor(del_buffer)

    def buffer_used(self):
        """
        count the buffers size
        Parameters
        ----------
        parameters:No parameters

        -------
        return:map
        buffer_count: L0A/B/C,L1,UB buffer used size.
        """
        buffer_count = {scope_cbuf: 0, scope_ubuf: 0, scope_ca: 0, scope_cb: 0,
                        scope_cc: 0, scope_smask: 0}
        for sbm in self.scope_stack:
            buffer_count_tmp = sbm.get_scope_buffer_used()
            for scope in buffer_count:
                if buffer_count_tmp.get(scope) is not None:
                    buffer_count[scope] += buffer_count_tmp[scope]
        return buffer_count

    def buffer_aviable(self):
        """
        count how much buffer memory is available
        Parameters
        ----------
        parameters:No parameters

        -------
        return:None or Map
        left_tesnor: L0A/B/C,L1,UB buffer available size.
        """
        if self.total_buffer is not None:
            left_tensor = {scope_cbuf: 0, scope_ubuf: 0, scope_ca: 0,
                           scope_cb: 0, scope_cc: 0, scope_smask: 0}
            buffer_count = self.buffer_used()
            for scope in self.total_buffer:
                left_tensor[scope] = self.total_buffer[scope] - \
                                     buffer_count[scope]
            return left_tensor
        return None

    def new_tensor(self, tensor):
        """
        when tensor is applied,tensor is be checked and be added in scope manage.

        Parameters
        ----------
        tensor:The tensor is applied in the scope
        -------
        return:NO return
        """
        aviable_tensor = self.buffer_aviable()
        if aviable_tensor is not None:
            if self.double_buffer_enable_ > 0:
                error_msg = "Tensor %s appiles buffer size(%dB) " \
                            "more than avaiable buffer size(%dB)." % \
                            (tensor.name, tensor.double_buffer_size(
                                self.double_buffer_thread_num),
                             aviable_tensor[tensor.scope])
                TikCheckUtil.check_le(
                    tensor.double_buffer_size(self.double_buffer_thread_num),
                    aviable_tensor[tensor.scope], error_msg)
                double_buffer_size = tensor.double_buffer_size(
                    self.double_buffer_thread_num)
                self.scope_stack[-1].add_tensor(tensor, double_buffer_size)
            else:
                error_msg = "Tensor %s appiles buffer size(%dB) " \
                            "more than avaiable buffer size(%dB)." % \
                            (tensor.name, tensor.buffer_size,
                             aviable_tensor[tensor.scope])
                TikCheckUtil.check_le(
                    tensor.buffer_size, aviable_tensor[tensor.scope], error_msg)
                size = tensor.buffer_size
                self.scope_stack[-1].add_tensor(tensor, size)

    def inject_dprofile(self, dprofile):
        """
        According the D core information, buffers parameters can be known.

        Parameters
        ----------
        parameters:Dproflie
        dprofile:the D core information
        -------
        return:buffer_map
        The D core buffers_information
        """
        self.dprofile = dprofile
        self.total_buffer = self.dprofile.buffer_size_query()

    def buffer_print(self):
        """
        print buffer information

        Parameters
        ----------
        parameters:No parameters
        -------
        return:No return
        """
        print("buffer used:", self.buffer_used())
        print("buffer available:", self.buffer_aviable())
        print("buffer total:", self.total_buffer)


class ScopeScalarManager():
    """
    managing scalars in the scope
    Scope:for_range,if_scope.else_scope.new_stmt_scope
    """
    def __init__(self):
        self.scalar_list = []

    @staticmethod
    def _check_scalar(scalar):
        """
        when scalar is used in this scope, this scope should be checked

        Parameters
        ----------
        scalar:The scalar is applied in this scope.
        -------
        return:No return
        """
        TikCheckUtil.check_type_match(
            scalar, Scalar,
            "input scalar(%s) should be Scalar" % str(type(scalar)))

    def add_scalar(self, scalar):
        """
        Adding scalar in this scope.

        Parameters
        ----------
        parameters:Scalar
        scalar:The new scalar is applied in this scope.
        -------
        return:No return
        """
        self._check_scalar(scalar)
        self.scalar_list.append(scalar)

    def del_scalar(self, scalar):
        """
        deleting scalar in this scope.

        Parameters
        ----------
        parameters:Scalar
        scalar:Scalar will be deleted in this scope.
        -------
        return:No return
        """
        self._check_scalar(scalar)
        self.scalar_list.remove(scalar)
        scalar.disable_scalar()


class CodeScalarManager():
    """
    Scalars are managed in this class.
    """
    def __init__(self):
        self.scope_stack = [ScopeScalarManager()]

    def new_scope(self):
        """
        When the new scope funtion is used, the new scope should be creaded.
        The new scope functions are for_range,
        if_scope/else_scope and new_stmt_scope

        Parameters
        ----------
        parameters:No parameters
        -------
        return:No return
        """
        self.scope_stack.append(ScopeScalarManager())

    def del_scope(self):
        """
        When this scope lifecycle is in the end,
        scalar of the scope should be deleted.

        Parameters
        ----------
        parameters:No parameters
        -------
        return:No return
        """
        del_scope_scalar = self.scope_stack.pop()
        for del_scalar in del_scope_scalar.scalar_list[:]:
            del_scope_scalar.del_scalar(del_scalar)

    def new_scalar(self, scalar):
        """
        When scalar is applied, scalar should be managed in this scope.

        Parameters
        ----------
        parameters:Scalar
        scalar : the new scalar is applied in this scope
        -------
        return:No return
        """
        self.scope_stack[-1].add_scalar(scalar)


def create_tik():
    """Create a new TIK_IRBuilder

    Returns
    -------
    builder : TikIRBuilder
        The created TIK_IRBuilder
    """
    from ..api.tik_ir_builder import TikIRBuilder
    return TikIRBuilder()
