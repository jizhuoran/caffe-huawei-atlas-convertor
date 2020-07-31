"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_tensor.py
DESC:     TODO (tik tensor explanation)
CREATED:  2019-04-18 18:53:42
MODIFIED: 2019-07-18 21:29:18
"""
# disabling:
# C0302: too-many-lines
# W0622: redefined-builtin
# W0601: global-variable-undefined
# C0103: invalid-name
# E1101: no-member
# R0913: too-many-arguments
# R0902: too-many-instance-attributes
# pylint: disable=C0302
# disable C0302 because here is object about tensor

import sys
import math
import collections

from te import tvm
from te.tvm import schedule
from te.platform.cce_params import scope_gm, scope_cbuf, scope_ubuf,\
                scope_ca, scope_cb, scope_cc, scope_smask
from ..tik_lib.tik_util import type_convert, need_check_out_of_scope
from ..tik_lib.tik_expr import Expr
from .. import debug
from ..common.util import TikUtil, reduce_mul, get_bit_len, \
    is_immediate_number, tvm_immediate_number, instance_judge
from ..tik_lib.tik_params import PYTHON_VERSION_IDX, PYTHON_VERSION3, \
    ONE_BYTE_BIT_LEN
from ..tik_lib.tik_backend import tik_access_ptr
from ..tik_lib.tik_basic_data import BasicData
from ..tik_lib.tik_check_util import TikCheckUtil
from ..tik_lib.tik_source_info import source_info_decorator, get_loc


_CONST_SELF_DTYPE_LANES = 1
_CONST_BUFFER_DTYPE_LANES = 1
_CONST_CAST_DTYPE_LANES = 1


def _tvm_value_to_int(value):
    """
    According tvm value return value.

    Parameters
    ----------
    value : tvm value

    Returns
    ----------
    return: value
    """
    # pylint: disable=W0622, W0601, C0103
    # disable them from support python3
    global long
    if sys.version_info[PYTHON_VERSION_IDX] >= PYTHON_VERSION3:
        long = int
    if isinstance(
            value, (tvm.expr.IntImm, tvm.expr.FloatImm,
                    tvm.expr.StringImm, tvm.expr.UIntImm)):
        return value.value
    if isinstance(value, (tvm.container.Array, list, tuple)):
        tmp = []
        for val in value:
            tmp.append(_tvm_value_to_int(val))
        return tmp
    if isinstance(value, (int, long)):
        return value
    return TikCheckUtil.raise_error("convert value is illegal")


def _normalize_shape(shape):
    """
    normalize input shape

    Parameters
    ----------
    shape: the input shape of new tensor

    Returns
    ----------
    return: shape_new
        new shape list
    """
    # change type of input shape to list, and each element is int
    shape_new = []
    for i in shape:
        if tvm_immediate_number([i]):
            shape_new.append(int(i.value))
        elif is_immediate_number([i]):
            if i <= 0:
                TikCheckUtil.raise_error(
                    "invalid shape input, shape: %s" % list(shape))
            shape_new.append(int(i))
        else:
            TikCheckUtil.raise_error(
                "only support static shape define now, tensor shape must be "
                "const number, while %s is not" % str(i))

    return shape_new


def _normalize_slice(slc, shape_num):
    """
    Slice tensor

    Parameters
    ----------
    slc : slice or the slice index
    shape_num: the shape index of slice

    Returns
    ----------
    return: the new slice of tensor
    """
    if not isinstance(slc, slice):
        return slice(slc, slc + 1, 1)
    start_tmp = 0
    stop_tmp = shape_num
    step_tmp = 1
    if slc.start is not None:
        start_tmp = slc.start
    if slc.stop is not None:
        stop_tmp = slc.stop
    if slc.step is not None:
        step_tmp = slc.step
    return slice(type_convert(start_tmp), type_convert(stop_tmp), type_convert(step_tmp))


def _slice_div(sls, div_num):
    """
    Slice divide

    Parameters
    ----------
    slc : slice of tensor
    div_num: the divide number of slice

    Returns
    ----------
    return: range of the slice
    """
    value = sls.step
    if tvm_immediate_number(value):
        value = int(sls.step.value)
    TikCheckUtil.check_type_match(value, (int), "value should be int")
    TikCheckUtil.check_equality(value, 1, "value should be equal to 1")
    return slice(sls.start // div_num, sls.stop // div_num, 1)


def _check_tensor_scope(scope):
    """
    when tensor is applied in the scope ,tensor's scope should be checeked

    Parameters
    ----------
    scope:the tensor's parameter scope

    Returns
    ----------
    return: No return
    """
    scope_list = [scope_gm, scope_cbuf, scope_ubuf, scope_ca, scope_cb,
                  scope_cc, scope_smask]
    TikCheckUtil.check_in_range(scope, scope_list, "scope out of Tensor Scope")


def _get_total_size(dtype, shape_new):
    """caculate total size, the unit is Bytes.

    Parameters
    ----------
    dtype : data type
    shape_new : shape

    Returns
    ----------
    total_size
    """
    offset_factor = int(get_bit_len(dtype)
                        // Tensor.ORIGIN_DTYPE_BITS)
    TikCheckUtil.check_ge(
        offset_factor, 1, "offset_factor should be more than 1")
    total_size = reduce_mul(shape_new) * offset_factor
    return total_size


def _dtype_bits_map(input_dtype):
    """
    A Map for TIK: from data type to int

    Parameters
    ----------
    input_dtype: str
    """
    output_bits = {
        'int8': 8,
        'int16': 16,
        'int32': 32,
        'uint8': 8,
        'uint16': 16,
        'uint32': 32,
        'float16': 16,
        'float32': 32,
        'int': 32,
        'float': 32,
        'uint64': 64,
        'int64': 64
    }
    if input_dtype in output_bits.keys():
        return output_bits[input_dtype]
    return TikCheckUtil.raise_error('Unexpected dtype: ' + input_dtype)


def _reduce_has_scalar(shape, local_indice):
    """
    judge if there is scalar in reinterpret_cast_to

    Parameters
    ----------
    shape : tensor's shape
    local_indice : tensor's indice

    Returns
    ----------
    True/False
    """

    def has_scalar(expr):
        """judge if there is load node or var"""
        _bin_expr = (tvm.expr.Add, tvm.expr.Sub, tvm.expr.Mul, tvm.expr.Mod,
                     tvm.expr.Min, tvm.expr.Max, tvm.expr.EQ, tvm.expr.NE,
                     tvm.expr.GE, tvm.expr.GT, tvm.expr.LE, tvm.expr.LT,
                     tvm.expr.And, tvm.expr.Or, tvm.expr.Not, tvm.expr.Div,
                     tvm.expr.FloorDiv)
        stack = collections.deque()
        stack.append(expr)
        while stack:
            _expr = stack.pop()
            if isinstance(_expr, Expr):
                _expr = _expr.get()
            if isinstance(_expr, (tvm.expr.Load,)):
                return True
            if isinstance(_expr, _bin_expr):
                stack.append(_expr.a)
                stack.append(_expr.b)
        return False
    shape_has_var = any(Expr(value).eval_value()
                        is None for value in shape)
    indice_has_var = any(Expr(value).eval_value()
                         is None for value in
                         (local_indice[-1].start, local_indice[-1].stop))
    shape_has_scalar = any(has_scalar(Expr(value))
                           is None for value in shape)
    indice_has_scalar = any(has_scalar(Expr(value))
                            is None for value in
                            (local_indice[-1].start,
                             local_indice[-1].stop))
    has_var = shape_has_var or indice_has_var
    has_sca = shape_has_scalar or indice_has_scalar

    return has_sca, not has_sca and has_var
# @cond

class TensorSlice():
    """
    hint : length of origin_shape is always same as slice_indice
    """
    def __init__(self, origin_shape, slice_indice=None):
        """
        TensorSlice init

        Parameters
        ----------
        origin_shape : tensor origin shape
        slice_indice: slice indice

        Returns
        ----------
        return: no return
        """
        self.indice = []
        if slice_indice is None:
            for i in origin_shape:
                self.indice.append(slice(0, i, 1))
        else:
            if isinstance(slice_indice, (tuple, list)):
                TikCheckUtil.check_equality(
                    len(slice_indice), len(origin_shape),
                    "length of slice_indice should be equal "
                    "to length or origin_shape")
                for (slice_value, origin_value) \
                        in zip(slice_indice, origin_shape):
                    tmp_slice = _normalize_slice(
                        slice_value, origin_value)
                    self.indice.append(tmp_slice)
            else:
                self.indice.append(_normalize_slice(
                    slice_indice, origin_shape[0]))
        self.origin_shape = origin_shape

    @staticmethod
    def __slice_length(slc):
        """
        slice length

        Parameters
        ----------
        slc : tensor slice

        Returns
        ----------
        return: the length of the slice
        """
        TikCheckUtil.check_type_match(slc, slice, "slc should be slice")
        tmp = Expr((slc.stop - slc.start) // slc.step)
        return tmp

    @staticmethod
    def __const_slice_length(slc):
        """
        tensor slice

        Parameters
        ----------
        slc : tensor slice

        Returns
        ----------
        return: tensor shape
        """
        if not isinstance(slc, slice):
            # slice length equal to 1
            return 1, 1
        slice_len = TensorSlice.__slice_length(slc)
        return slice_len, slice_len.eval_value()

    @property
    def shape(self):
        """
        return tensor shape
        Parameters
        ----------

        Returns
        ----------
        return: tensor shape
        """
        tmp_shape = []
        for i in self.indice:
            slice_len, const_slice_len = TensorSlice.__const_slice_length(i)
            if const_slice_len:
                tmp_shape.append(const_slice_len)
            else:
                tmp_shape.append(slice_len)
        return tmp_shape

    @property
    def offset(self):
        """
        tensor offset
        Parameters
        ----------

        Returns
        ----------
        return: tensor offset
        """
        TikCheckUtil.check_equality(
            len(self.indice), len(self.origin_shape),
            "length of indice should be equal to length of origin_shape")
        acc_shape = [1]
        for i in self.origin_shape[::-1]:
            acc_shape.append(acc_shape[-1]*i)
        acc_shape = acc_shape[:-1][::-1]
        idx_of_array = 0
        for i in range(len(self.origin_shape)):
            if isinstance(self.indice[i], slice):
                idx = self.indice[i].start
                if idx is None:
                    idx = 0
            else:
                idx = self.indice[i]
            idx_of_array += acc_shape[i]*idx
        return idx_of_array

    def is_single_point(self):
        """
        judging whether tensor is one point
        Parameters
        ----------

        Returns
        ----------
        return: whether tensor is one point
        """
        for i in self.indice:
            _, const_slice_len = TensorSlice.__const_slice_length(i)
            if not const_slice_len or const_slice_len != 1:
                return False
        return True

    def is_sliced(self):
        """
        judging whether tensor is sliced
        Parameters
        ----------

        Returns
        ----------
        return: whether tensor is sliced
        """
        for i in range(len(self.indice)):
            _, const_slice_len = TensorSlice.__const_slice_length(self.indice[i])
            if const_slice_len and const_slice_len != self.origin_shape[i]:
                return True
        return False

    def check_shape(self):
        """
        checking tensor shape
        Parameters
        ----------

        Returns
        ----------
        return: whether tensor shape is the data type of int
        """
        for i in self.shape:
            TikCheckUtil.check_type_match(
                i, int, "each member of shape should be int")

    def reshape(self, new_shape):
        '''
        Description:
          Sets the tensor shape.

        Args:
          - new_shape : New shape of a tensor object. The supported types
          are list(int) and tuple(int).
            - NOTICE :
              In the current version, lists or tuples consisting
              of integral immediates are supported.

        Restrictions:
          - The total size of the new shape must be the same as that
          of the old shape.
          - The new and old tensors point to the same buffer. After the value
          of the new tensor is changed, the value
          of the old tensor would also change.

        Returns:
          New tensor

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            data_A = tik_instance.Tensor("float16", (128,), name="data_A",
                                        scope=tik.scope_gm)
            data_A.reshape((64,2))
        '''
        TikCheckUtil.check_type_match(
            new_shape, (list, tuple), "new_shape should be list or tuple")
        TikCheckUtil.check_equality(
            reduce_mul(new_shape), reduce_mul(self.shape),
            "size of new_shape should be equal to size of shape")
        if not self.is_sliced():
            # change type of new_shape to list, and each element is int
            shape_new = _normalize_shape(new_shape)
            return TensorSlice(shape_new)
        if len(self.origin_shape) == len(self.indice) and \
                len(self.origin_shape) == 1:
            new_origin_shape = []
            new_indice = []
            reduce_num = 1
            tmp_origin_shape = self.origin_shape[0]
            tmp_indice = self.indice[0]
            if len(new_shape) == 1:
                return self
            for i in (new_shape[:-1])[::-1]:
                new_indice.append(slice(0, i, 1))
                new_origin_shape.append(i)
                reduce_num *= i
            tmp_origin_shape //= reduce_num
            tmp_indice = _slice_div(tmp_indice, reduce_num)
            new_indice.append(tmp_indice)
            new_origin_shape.append(tmp_origin_shape)
            return TensorSlice(new_origin_shape[::-1], new_indice[::-1])
        return TikCheckUtil.raise_error("not support for now")

    @property
    def size(self):
        """
        return trensor size
        Parameters
        ----------

        Returns
        ----------
        return:tensor size
        """
        return reduce_mul(self.shape)

    def simplify(self):
        """
        merging continuous dimensions
        Parameters
        ----------

        Returns
        ----------
        return:the new tensor
        """
        sshape = []
        sslice = []
        for i in range(len(self.origin_shape)):
            shape_length = self.origin_shape[i]
            if shape_length != 1:
                _, slice_shape = TensorSlice.__const_slice_length(
                    self.indice[i])
                if slice_shape and slice_shape == shape_length:
                    if sshape and (isinstance(sslice[-1], slice)):
                        sshape[-1] = sshape[-1]*shape_length
                        sslice[-1] = slice(sslice[-1].start*shape_length,
                                           sslice[-1].stop*shape_length,
                                           sslice[-1].step)
                    else:
                        sslice.append(slice(0, shape_length, 1))
                        sshape.append(shape_length)
                else:
                    if (sshape and (isinstance(sslice[-1], slice))
                            and (TensorSlice.__const_slice_length(
                                sslice[-1])[1] == 1)):
                        sshape[-1] = sshape[-1]*shape_length
                        sslice[-1] = slice(sslice[-1].start*shape_length +
                                           self.indice[i].start,
                                           sslice[-1].start*shape_length +
                                           self.indice[i].stop,
                                           sslice[-1].step)
                    else:
                        sshape.append(shape_length)
                        sslice.append(self.indice[i])
        for i, t_shape in enumerate(sshape):
            sslice[i] = _normalize_slice(sslice[i], t_shape)
        return TensorSlice(sshape, sslice)

    @staticmethod
    def _slice_cal(slice_, sub_slice_):
        """
        slice Calculation
        Parameters
        ----------
        slice_:tensor slice
        sub_slice_:tensor sub slice

        Returns
        ----------
        return:return tensor slice
        """
        if isinstance(sub_slice_, slice):
            ret_start = Expr(sub_slice_.start)
            ret_stop = Expr(sub_slice_.stop)
            ret_step = Expr(sub_slice_.step)
            ret_start = Expr(ret_start*slice_.step + slice_.start)

            ret_start_value = ret_start.eval_value()
            slice_stop_value = Expr(slice_.stop).eval_value()

            if ret_start_value is None or slice_stop_value is None:
                pass
            else:
                ret_start = min(ret_start_value, slice_stop_value)

            ret_stop = Expr(ret_stop*slice_.step + slice_.start)
            ret_stop_value = ret_stop.eval_value()

            if ret_stop_value is None or slice_stop_value is None:
                pass
            else:
                ret_stop = min(ret_stop_value, slice_stop_value)
            ret_step *= Expr(slice_.step)
            return slice(ret_start, ret_stop, ret_step)
        return slice_.start + sub_slice_

    def slice_expand_reduce(self,  # pylint: disable=R0914
                            src_bit_len, dst_bit_len):
        """
        address transform
        Parameters
        ----------
        src_bit_len: src dtype bit len
        dst_bit_len: dst dtype bit len

        Returns
        ----------
        return:return tensor new slice
        """
        # pylint: disable=W0601, C0103
        # disable them from support python3
        shape = self.shape[:]
        local_indice = self.indice[:]
        origin_shape = self.origin_shape[:]
        if src_bit_len > dst_bit_len:
            # python_idx
            if sys.version_info[PYTHON_VERSION_IDX] >= PYTHON_VERSION3:
                long = int
            if tvm_immediate_number(local_indice[-1].step) and isinstance(
                    local_indice[-1].step.value, (int,)) \
                    and local_indice[-1].step.value == 1:
                has_scalar, has_var = _reduce_has_scalar(shape, local_indice)
                origin_shape[-1] = origin_shape[-1]*src_bit_len // dst_bit_len
                if has_scalar:
                    shift = int(math.log2(src_bit_len))
                    shape[-1] = shape[-1] << shift
                    start = Expr(local_indice[-1].start << shift)
                else:
                    shape[-1] = shape[-1] * src_bit_len // dst_bit_len
                    start = local_indice[-1].start * src_bit_len // dst_bit_len
                start = Expr(start)
                local_indice[-1] = slice(
                    start,
                    start + shape[-1], 1)
            else:
                origin_shape.append(src_bit_len//dst_bit_len)
                local_indice.append(slice(0, src_bit_len//dst_bit_len, 1))
        elif src_bit_len == dst_bit_len:
            return self
        else:
            def _reduce(num, error_msg):
                num = Expr(num * src_bit_len // dst_bit_len).eval_value()
                TikCheckUtil.check_equality(int(num), num, error_msg)
                return int(num)

            def _imm_transform():
                shape[-1] = _reduce(shape[-1],
                                    "FAILED: Reinterpret_cast_to() shape[-1] ="
                                    "%s. Error: Last dimension in shape "
                                    "multiplies factor "
                                    "should be an integer." % shape[-1])
                origin_shape[-1] = _reduce(origin_shape[-1],
                                           "the last member of origin_shape's"
                                           "integer part should be equal to "
                                           "itself")
                TikCheckUtil.check_equality(
                    _tvm_value_to_int(local_indice[-1].step), 1)
                start = _reduce(_tvm_value_to_int(local_indice[-1].start),
                                "the offset of tensor %s is not divisible by "
                                "%s" % (local_indice[-1].start,
                                        int(1 * dst_bit_len // src_bit_len)))
                local_indice[-1] = slice(start,
                                         start + shape[-1],
                                         1)
                return origin_shape, local_indice

            def _scalar_transform():
                shift = int(math.log2(1 * dst_bit_len // src_bit_len))
                shape[-1] = shape[-1] >> shift
                origin_shape[-1] = int(
                    origin_shape[-1]*src_bit_len // dst_bit_len)
                start = Expr(local_indice[-1].start >> shift)
                local_indice[-1] = slice(start,
                                         start + shape[-1],
                                         1)
                return origin_shape, local_indice

            def _var_transform():
                shape[-1] = shape[-1]*src_bit_len // dst_bit_len
                origin_shape[-1] = int(
                    origin_shape[-1]*src_bit_len // dst_bit_len)
                start = Expr(local_indice[-1].start*src_bit_len // dst_bit_len)
                local_indice[-1] = slice(start,
                                         start + shape[-1],
                                         1)
                return origin_shape, local_indice

            has_scalar, has_var = _reduce_has_scalar(shape, local_indice)
            if has_scalar:
                origin_shape, local_indice = _scalar_transform()
            elif has_var:
                origin_shape, local_indice = _var_transform()
            else:
                origin_shape, local_indice = _imm_transform()

        return TensorSlice(origin_shape, local_indice)

    def __getitem__(self, index):
        TikCheckUtil.check_equality(
            len(index), len(self.indice),
            "the dimension of index should be equal to tensor's dimension.")
        slice_list = []
        for i in range(len(self.indice)):
            idx = self.indice[i]
            sub_idx = _normalize_slice(index[i], self.shape[i])
            new_slice = TensorSlice._slice_cal(idx, sub_idx)
            _, const_slice_len = TensorSlice.__const_slice_length(new_slice)
            if const_slice_len and const_slice_len <= 0:
                TikCheckUtil.raise_error("The {} axis of tensor which is "
                                         " {} has more than  {}"
                                         .format(str(i), str(index[i]),
                                                 str(idx.stop)))
            slice_list.append(new_slice)

        return TensorSlice(self.origin_shape, slice_list)

    def __str__(self):
        return "origin_shape: %s, indice: %s" % (str(self.origin_shape),
                                                 str(self.indice))

    def __repr__(self):
        return str(self)
    # @endcond


@debug.tensor_register
class Tensor(BasicData):
    """
    scalar expression

    content :
        self.ir_generator
        self.buffer
        self.indice
    """
    # pylint: disable=R0902
    # @cond
    VECTOR_MAC_BIT = 128*16
    COUNT = 0
    ORIGIN_DTYPE = "uint8"
    ORIGIN_DTYPE_BITS = 8
    BUFFER_REUSE_COUNT = 0

    def __init__(self, ir_generator, dtype=None,  # pylint: disable=R0913, R0914
                 shape=None, scope=None, name=None, buffer_=None, indice=None,
                 enable_buffer_reuse=False, is_workspace=False,
                 is_atomic_add=False):
        """
        tensor register initialization
        Parameters
        ----------
        ir_generator:Halide IR generator
        dtype:tensor register data type
        shape:tensor register shape
        scope:tensor register scope
        name:tensor register name
        buffer_:tensor register buffer
        indice:tensor register indice
        enable_buffer_reuse : whether specify reuse relationship between
                           tensors or not
        is_workspace: whether is workspace or not
        is_atomic_add: whether is atomic add or not
        Returns
        ----------
        return:no return
        """
        BasicData.__init__(self, "Tensor")

        if name is None:
            name = "auto_tensor_" + str(Tensor.COUNT)
        # check name is valid
        TikCheckUtil.check_name_str_valid(name)
        # check dtype is str
        TikCheckUtil.check_type_match(dtype, str, "dtype should be str")
        # _available:The scalar state variable. when scalar variable
        # "_available" is True, this scalar can be accessed.when scalar variable
        # "_available" is False, this scalar can't be accessed and assert
        # "This Scalar is not defined in this scope."
        self._available = True
        self.ir_generator = ir_generator
        self._initial_param()
        self.dtype = dtype
        self.buffer_reuse_id = None
        self.is_workspace = is_workspace
        self.is_atomic_add = is_atomic_add
        self.source_loc = None

        if (shape is not None) and (scope is not None) and (dtype is not None):
            TikCheckUtil.check_type_match(shape, (list, tuple),
                                          "shape: %s must be list or tuple"
                                          % str(shape))

            shape_new = _normalize_shape(shape)
            total_size = _get_total_size(self.dtype, shape_new)
            _check_tensor_scope(scope)

            if scope == scope_gm:
                from te.tvm.api import decl_buffer
                if ir_generator.is_tensor_in_scope():
                    TikCheckUtil.raise_error("Tensor can't be defined in local")
                self.buffer = decl_buffer(total_size, Tensor.ORIGIN_DTYPE, name)
                self.source_loc = get_loc()
            else:
                if enable_buffer_reuse:
                    self.buffer_reuse_id = Tensor.BUFFER_REUSE_COUNT
                    Tensor.BUFFER_REUSE_COUNT += 1
                self.buffer = ir_generator.apply_for_new_alloc(
                    Tensor.ORIGIN_DTYPE, total_size, scope, name,
                    self.buffer_reuse_id)
                ir_generator.code_buffer_manager.new_tensor(self)
            self.indice = TensorSlice(shape_new)

        elif (shape is None) and (scope is None):
            TikCheckUtil.check_not_is(buffer_, None,
                                      "buffer_ should not be None")
            TikCheckUtil.check_not_is(indice, None,
                                      "indice should not be None")
            self.buffer = buffer_
            self.indice = indice
        else:
            TikCheckUtil.raise_error(
                "Tensor Class must be initial by (dtype, shape, scope, name) ")

    def _initial_param(self):
        """
        parameters initialization
        Parameters
        ----------

        Returns
        ----------
        return:no return
        """
        Tensor.COUNT += 1
        self.indice = None

    @classmethod
    def access_wrap(cls, ir_generator, tvm_access, new_dtype, indice_new=None):
        """
        tensor's constructive function
        tvm_access:buffer Var
        new_dtype:tensor data type
        """
        if indice_new is None:
            indice = TensorSlice(TikUtil.to_list(
                reduce_mul(_tvm_value_to_int(tvm_access.shape)) //
                (get_bit_len(new_dtype) // Tensor.ORIGIN_DTYPE_BITS)))
            return cls(ir_generator, buffer_=tvm_access,
                       indice=indice, dtype=new_dtype)
        return cls(ir_generator, buffer_=tvm_access,
                   indice=indice_new, dtype=new_dtype)

    @property
    def scope(self):
        """
        return tensor memory scope
        Parameters
        ----------

        Returns
        ----------
        return:buffer scope
        """
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        return self.buffer.scope

    @property
    def name(self):
        """
        return buffer name
        Parameters
        ----------

        Returns
        ----------
        return:buffer name
        """
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        return self.buffer.name

    @property
    @source_info_decorator()
    def shape(self):
        """
        return tensor shape
        Parameters
        ----------

        Returns
        ----------
        return:tensor shape
        """
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        return self.indice.shape

    @property
    def size(self):
        """
        return tensor size
        Parameters
        ----------

        Returns
        ----------
        return:tensor size
        """
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
            TikCheckUtil.check_equality(self._available, True, msg)
        return self.indice.size

    @property
    def buffer_size(self):
        """
        ret unit : byte
        """
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        return reduce_mul([i.value for i in self.buffer.shape])

    def double_buffer_size(self, thread_num_value):
        """
        ret unit : byte
        """
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        return reduce_mul([i.value for i in self.buffer.shape])*thread_num_value

    def disable_tensor(self):
        """
         when this tensor lifecycle is in the end,
         this tensor condition parameter should be changed from true to false.
         Parameters
         ----------
         parameter:No parameter

         Returns
         ----------
         return:no return
        """
        self._available = False

    def is_tensor_slice(self):
        """
        judging whether tensor is sliced
        Parameters
        ----------

        Returns
        ----------
        return:judging whether tensor is sliced
        """
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        return self.indice.is_sliced()

    def is_single_point(self):
        """
        judging whether tensor is one point
        Parameters
        ----------

        Returns
        ----------
        return:judging whether tensor is one point
        """
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        return self.indice.is_single_point()

    @property
    def offset(self):
        """
        return tensor indice offset
        Parameters
        ----------

        Returns
        ----------
        return:tensor indice offset
        """
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        return self.indice.offset

    @source_info_decorator()
    def change_shape(self, new_shape):
        """
        change shape
        Parameters
        ----------
        new_shape:the new tensor register shape

        Returns
        ----------
        return:tensor register indice
        """
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        TikCheckUtil.check_type_match(
            new_shape, (list, tuple), "new_shape should be list or tuple")
        self.indice = self.indice.reshape(new_shape)
    # @endcond

    @source_info_decorator()
    def reshape(self, new_shape):
        '''
        Sets the tensor shape.
        Description:
          Sets the tensor shape.

        Args:
          new_shape : New shape of a tensor object. The supported types
          are list(int) and tuple(int).
            - NOTICE :
              In the current version, lists or tuples consisting
              of integral immediates are supported.

        Restrictions:
          - The total size of the new shape must be the same as that
          of the old shape.
          - The new and old tensors point to the same buffer. After the value
          of the new tensor is changed, the value
          of the old tensor would also change.

        Returns:
          New tensor

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            data_A = tik_instance.Tensor("float16", (128,), name="data_A",
                                        scope=tik.scope_gm)
            data_A.reshape((64,2))
        '''
        return self.reshape_(new_shape)

    # @cond
    def reshape_(self, new_shape):
        """
        tensor register reshape
        note: use this function to call tensor.reshape inside!!
        """
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        indice = self.indice.reshape(new_shape)
        tmp_t = self.access_wrap(self.ir_generator, self.buffer, self.dtype, indice)
        return tmp_t

    @source_info_decorator()
    def flatten(self):
        """
        tensor flatten shape
        Parameters
        ----------

        Returns
        ----------
        return:tensor register reshape
        """
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        return self.reshape([reduce_mul(self.shape)])

    def _get_access_ptr_args(self, offset_temp, access_mask, ptr_type,
                             content_lanes, cast_dtype, extent):
        # Offset, extent need to be casted
        # Offset: cast_dtype to buffer.dtype
        # pylint: disable=R0913, E1101
        # disable it because pylint can't find symbol from back-end
        buffer_dtype_bits = _dtype_bits_map(self.buffer.dtype)*\
                            _CONST_BUFFER_DTYPE_LANES
        offset_buffer_dtype = offset_temp*\
                              (_dtype_bits_map(self.dtype)*
                               _CONST_SELF_DTYPE_LANES // buffer_dtype_bits)
        # elem_offset, extent, e_dtype: buffer.dtype to cast_dtype
        cast_dtype_bits = _dtype_bits_map(cast_dtype)* \
                          _CONST_CAST_DTYPE_LANES
        cast_coeff = cast_dtype_bits // buffer_dtype_bits

        ret = tik_access_ptr(buffer=self.buffer,
                             access_mask=access_mask,
                             cast_coeff=cast_coeff,
                             ptr_type=ptr_type,
                             content_lanes=content_lanes,
                             offset=offset_buffer_dtype)

        elem_offset = ret.args[2] // cast_coeff
        if extent is None:
            extent = ret.args[3] // cast_coeff

        else:
            extent = extent*ONE_BYTE_BIT_LEN // cast_dtype_bits
        return [tvm.const(0, cast_dtype), ret.args[1], elem_offset,
                extent, ret.args[4]]

    def access_ptr(self, access_mask, ptr_type="handle",
                   content_lanes=1, offset=0, cast_dtype="handle",
                   extent=None):
        """Get an access pointer to the head of buffer.

        Support cast feature for TIK.
        Add a Parameter: cast_dtype

        Parameters
        ----------
        access_mask : int
            The access pattern MASK. Indicate whether the
            access will read or write to the data content.

        ptr_type : str, optional
            The data type of the result pointer. Do not specify
            unless we want to cast pointer to specific type.

        content_lanes: int, optional
            The number of lanes for the data type. This value
            is greater than one for vector types.

        offset: Expr, optional
            The offset of pointer. We can use it to offset by
            the number of elements from the address of ptr.

        cast_dtype: str, optional
            The data type of the result pointer. Do not specify
            unless we want to cast pointer to specific type.
        """
        # pylint: disable=R0913
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        if cast_dtype == "handle":
            cast_dtype = self.dtype

        string_types = (str,)  # python 3 upgraded
        if isinstance(access_mask, string_types):
            mask = 0
            for value in access_mask:
                if value == "r":
                    mask = mask | self.buffer.READ
                elif value == "w":
                    mask = mask | self.buffer.WRITE
                else:
                    TikCheckUtil.raise_error(
                        "Unknown access_mask %s" % access_mask,
                        exception_type=ValueError)
            access_mask = mask

        offset_temp = self.offset + offset
        offset_temp = schedule.convert(offset_temp)

        m_args = self._get_access_ptr_args(offset_temp, access_mask,
                                           ptr_type, content_lanes,
                                           cast_dtype, extent)
        return tvm.call_intrin(cast_dtype, 'tvm_access_ptr', *m_args)

    def ptr(self, index):
        """
        get tensor index
        Parameters
        ----------
        index:tensor register index


        Returns
        ----------
        return:tensor register
        """
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        return self[index]
    # @endcond

    @source_info_decorator()
    def reinterpret_cast_to(self, dtype):
        '''
        Casts the data type of a tensor.
        Description:
          Casts the data type of a tensor. This API reads data of the specified
           data type only. The data type of the
          data value remains unchanged.
          For example, if a tensor has 128 float16 entries, reinterpret_cast_to
           can be used to read the entries in
          float32 mode, obtaining 64 float32 entries. However,
          reinterpret_cast_to does not convert the data
          type eventually.

        Args:
          dtype : Data type of the Tensor object. Must be one of the
          following data types: uint8, int8, uint16, int16, float16, uint32
          , int32, float32, uint64, int64

        Restrictions:
          - To make it easier to describe the restrictions in calling
          reinterpret_cast_to(), we define a factor yielded
          by dividing the number of bits of the original data type by that of
          the specified data type. Assume the
          original tensor is declared to have 128 float16 data entries in the
          buffer. To read the entries in
          float32 mode, the factor should be 0.5 (16/32). The call to
          reinterpret_cast_to() must meet the
          following restrictions:
            - The factor must be greater than 0.
            - If the factor is greater than 1, it must be an integer.
            - If the factor is less than 1, pay attention to the tensor shape.
            The last dimension size (shape[-1])
            multiplied by the factor must be an integer. Assume the original
            tensor is with shape (128, 1). To read its
            128 float16 entries in float32 mode,
            shape[-1] * factor = 1 * 0.5 = 0.5, which is not an integer and
            therefore the preceding restriction is not met. In this case,
            the error message "Error: Last dimension in
            shape multiplies factor should be an integer" will be reported.
            Setting the tensor shape to 128 can avoid
            this error.

        Returns:
          The new tensor

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            data_A = tik_instance.Tensor("float16", (128,), name="data_A",
                                        scope=tik.scope_gm)
            data_A.reinterpret_cast_to("uint16")
        '''
        # check dtype is str
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        TikCheckUtil.check_type_match(dtype, str, "dtype should be str")
        indice = self.indice.slice_expand_reduce(get_bit_len(self.dtype),
                                                 get_bit_len(dtype))
        tmp_t = self.access_wrap(self.ir_generator, self.buffer, dtype, indice)
        return tmp_t

    # @cond
    def __call__(self, index):
        """
        get tensor register item
        Parameters
        ----------
        index:tensor register index


        Returns
        ----------
        return:the index tensor register
        """
        return self.__getitem__(index)
    # @endcond

    @source_info_decorator()
    def __getitem__(self, index_in):
        '''
        Obtains partial tensor data to form a new tensor.
        Description:
          Obtains partial tensor data to form a new tensor.

        Args:
          index_in : Tensor array index. The options are as follows:
            - int, long type. The multidimensional data of the original tensor
            is considered as one dimension, counted
            from 0. The new tensor becomes one-dimensional, for example,
            dataA_UB[100].
            - Slice instance: Python slice, including start, stop, and
            step. for example, dataA_UB[100:200:1].
            - Tuple type. It is represented in multidimensional mode. The
            dimensions are the same as those of the tensor.
            Each dimension can be a slice or an immediate. Dimensions are
            separated by commas (,), for example,
            dataA_UB[0,0,0].

        Restrictions:
          None

        Returns:
          The new tensor

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            data_A = tik_instance.Tensor("float16", (128,), name="data_A",
                                        scope=tik.scope_gm)
            data_b = data_A[1]
        '''
        # pylint: disable=W0601, C0103
        # disable them from support python3
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        global long
        if sys.version_info[PYTHON_VERSION_IDX] >= PYTHON_VERSION3:
            long = int
        from ..tik_lib.tik_params import INDEX_IN_START, INDEX_IN_STOP
        if isinstance(index_in, slice) and \
           instance_judge([index_in.stop, index_in.start], (int, long)) and \
           (index_in.stop == INDEX_IN_STOP) \
                and (index_in.start == INDEX_IN_START):
            index = slice(None, None, None)
        else:
            index = index_in
        if not isinstance(index, (tuple, list)):
            index = [index]
        flatten_flag = False
        if len(index) == len(self.indice.origin_shape):
            pass
        elif len(index) == 1:
            flatten_flag = True
        else:
            TikCheckUtil.raise_error("index not match shape")
        i = self.access_wrap(self.ir_generator, self.buffer, self.dtype)
        if flatten_flag:
            i = i.flatten().__getitem__(index)
        else:
            i.indice = self.indice.__getitem__(index)
        return i

    # @cond
    @source_info_decorator()
    def clean_shape(self):
        """
        clean tensor shape
        Parameters
        ----------

        Returns
        ----------
        return:the i tensor register
        """
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        i = self.access_wrap(self.ir_generator, self.buffer, self.dtype)
        i.indice = self.indice.simplify()
        return i

    def __to_reg(self):
        """
        creat temp register
        Parameters
        ----------

        Returns
        ----------
        return:the temp register
        """
        tmp_reg = self.ir_generator.Scalar_(self.dtype)
        TikCheckUtil.check_is(
            self.is_single_point(), True, "single point sliced is false")
        tmp_reg.set_as(self)
        return tmp_reg
    # @endcond

    @source_info_decorator()
    @debug.tensor_set_as_decorator
    def set_as(self, value, dst_offset=0, src_offset=None):
        '''
        Sets a tensor.
        Description:
          Sets a tensor.

        Args:
          value : Value to be assigned from:
            - A tensor of one element
            - A Scalar variable
            - An Expr consisting of a Scalar variable and an immediate,
            which cannot be of type float
          dst_offset: An internal optional parameters
          src_offset: An internal optional parameters

        Restrictions:
          None

        Returns:
          None

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            data_A = tik_instance.Tensor("float16", (128,), name="data_A",
                                        scope=tik.scope_ubuf)
            data_B = tik_instance.Tensor("float16", (128,), name="data_A",
                                        scope=tik.scope_ubuf)
            data_A[0].set_as(data_B[0])
        '''
        msg = "Tensor %s is not defined in this scope." % self.buffer.name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        if isinstance(value, Tensor) and src_offset is None and \
                (value.size > 1):
            TikCheckUtil.raise_error("This is illegal for tensor.")
        self.ir_generator.assign(self, value, dst_offset, src_offset)

    @source_info_decorator()
    def __setitem__(self, index, value):
        '''
        Changes a tensor.
        Description:
          Changes a tensor.

        Args:
          index : Tensor array index. The options are as follows:
            - int, long type. The multidimensional data of the original tensor
            is considered as one dimension, counted
            from 0. The new tensor becomes
            one-dimensional, for example, dataA_UB[100].
            - Slice instance: Python slice, including start, stop, and
            step. for example, dataA_UB[100:200:1].
            - Tuple type. It is represented in multidimensional mode. The
            dimensions are the same as those of the tensor.
            Each dimension can be a slice or an immediate. Dimensions are
            separated by commas (,), for example,
            dataA_UB[0,0,0].
          value : Specific value, which is related to the data type defined
          by the tensor.
          Currently, only the scalar, Expr, and tensor variables are supported.
           immediates are not supported.

        Restrictions:
          None

        Returns:
          None

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            data_A = tik_instance.Tensor("float16", (128,), name="data_A",
                                        scope=tik.scope_ubuf)
            scalar_B = tik_instance.Scalar(dtype="float16", name="scalar_B",
                                        init_value=2.0)
            data_A[0].set_as(scalar_B)
        '''
        self.__getitem__(index).set_as(value)

    # @cond
    @source_info_decorator()
    def __str__(self):
        """
        get tensor name
        Parameters
        ----------

        Returns
        ----------
        return:get tensor name
        """
        return self.name

    @source_info_decorator()
    def __eq__(self, other):
        """
        judging whether two tensor are equal
        Parameters
        ----------
        other:other tensor register

        Returns
        ----------
        return:juding whether two tensor are equal
        """
        reg = self.__to_reg()
        return reg == other

    @source_info_decorator()
    def __ne__(self, other):
        """
        judging whether two tensor aren't equal
        Parameters
        ----------
        other:other tensor register

        Returns
        ----------
        return:juding two tensor aren't equal
        """
        reg = self.__to_reg()
        return reg != other

    @source_info_decorator()
    def __lt__(self, other):
        """
        judging whether tensor is less than other
        Parameters
        ----------
        other:other tensor register

        Returns
        ----------
        return:judging whether tensor is less than other
        """
        reg = self.__to_reg()
        return reg < other

    @source_info_decorator()
    def __gt__(self, other):
        """
        judging whether tensor is greater than other
        Parameters
        ----------
        other:other tensor register

        Returns
        ----------
        return:judging whether tensor is greater than other
        """
        reg = self.__to_reg()
        return reg > other

    @source_info_decorator()
    def __ge__(self, other):
        """
        judging whether tensor is greater than or equal to other
        Parameters
        ----------
        other:other tensor register

        Returns
        ----------
        return:judging whether tensor is greater than or equal to other
        """
        reg = self.__to_reg()
        return reg >= other

    @source_info_decorator()
    def __le__(self, other):
        """
        judging whether tensor is less than or equal to other
        Parameters
        ----------
        other:other tensor register

        Returns
        ----------
        return:judging whether tensor is less than or equal to other
        """
        reg = self.__to_reg()
        return reg <= other

    @source_info_decorator()
    def __hash__(self):
        """
        according buffer return hash value
        Parameters
        ----------

        Returns
        ----------
        return:hash value
        """
        return hash(self.buffer)
    # @endcond


def get_addr_list(addr_list, tensor_object, mode, extent=None):
    """
    get tensor's address with mode, and put them in list
    :param addr_list: the list store address we get
    :param tensor_object: tensor instance
    :param mode: w or r
    :param extent: extent of the addr_list
    :return: None
    """
    if isinstance(tensor_object, Tensor):
        if extent is None:
            addr_list.append(
                tvm.expr.Cast("uint64", tensor_object.access_ptr(mode))*
                tvm.const(1, "uint64"))
        else:
            addr_list.append(
                tvm.expr.Cast("uint64",
                              tensor_object.access_ptr(mode, extent=extent))*
                tvm.const(1, "uint64"))
    else:
        addr_list.append(tensor_object)
