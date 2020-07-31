"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_scalar.py
DESC:     scalar
CREATED:  2019-04-18 18:53:42
MODIFIED: 2019-07-18 21:29:18
"""
# disabling:
# W0622: redefined-builtin
# R0913: too-many-argument
# E1101: no-member
# W0601: global-variable-undefined
# C0103: invalid-name
# R0903: too-few-public-methods
# R0902: too-many-instance-attributes

import sys

from te import tvm
from te.platform import cce_params
from te.platform.cce_params import scope_reg, scope_gm
from te.tvm import make, const

from ..tik_lib.tik_expr import BasicExpr, Expr
from .. import debug
from ..common.util import check_integer_in_range, check_scalar_dtype, \
    is_immediate_number, is_basic_expr, check_imme_mask_full_mode
from ..tik_lib.tik_params import MASK_LEN_CONTINOUS_MODE, MASK_LEN_FULL_MODE, \
    PIPE_MTE1, MIN_MASK, MAX_MASK, \
    MASK_LOW_SHIFT, MASK_VALUE_64, MASK_VALUE_128, MASK_VALUE_ZERO, \
    MAX_MASK_LOW_VALUE, PYTHON_VERSION_IDX, PYTHON_VERSION3, MAX_COUNTER_MASK, \
    BIT_LEN_16, MAX_MASK_64
from ..tik_lib.tik_util import need_check_out_of_scope, type_convert
from ..tik_lib.tik_buffervar import TikBufferVar
from ..tik_lib.tik_basic_data import BasicData
from ..tik_lib.tik_check_util import TikCheckUtil
from ..tik_lib.tik_source_info import get_loc, source_info_decorator


_SCALAR_MASK_VALUE_IDX = 0
_SCALAR_EXTENTS = (1,)


@debug.scalar_register
class Scalar(BasicExpr, BasicData):
    """
    hint:scalar expression
    """
    # pylint: disable = R0902
    # @cond
    COUNT = 0


    def __init__(self, ir_generator, dtype="int64",
                 name="reg_buf", init_value=None, if_global_scope=False):
        """
        scalar register initialization
        Parameters
        ----------
        ir_generator:Halide IR generator
        dtype:tensor register data type
        name:scalar_register name
        init_value:scalar_registerinit value
        if_global_scope:global_scope

        Returns
        ----------
        return:no return
        """
        # pylint: disable=R0913, E1101,
        # disable E1101 because pylint can't find back-end symbol
        # too-many-instance-attributes, so disalbe R0902

        # check dtype is str
        TikCheckUtil.check_type_match(dtype, str, "dtype should be str")
        # check name is valid
        TikCheckUtil.check_name_str_valid(name)

        BasicExpr.__init__(self)
        BasicData.__init__(self, "Scalar")

        self.instance_func = None
        Scalar.COUNT += 1
        self.ir_generator = ir_generator
        self._if_global_scope = if_global_scope
        self.class_type = Expr
        self.debug_var = tvm.var('scalar_debug_var_' + name, dtype)

        # _available:The scalar state variable. when scalar variable
        # "_available" is True, this scalar can be accessed.when scalar variable
        # "_available" is False, this scalar can't be accessed and assert
        # "This Scalar is not defined in this scope."
        self._available = True
        if if_global_scope:
            self._name = "global_" + name + str(Scalar.COUNT)
            buffer_var = tvm.var("global_" + name + str(Scalar.COUNT),
                                 dtype="handle")
            self.reg_buffer = TikBufferVar(self.ir_generator,
                                           buffer_var, dtype)

            _instance_func_loc = get_loc()

            def _instance_func(pre_stmt):
                tmp_string_imm = make.StringImm(scope_reg)
                tmp_const = const(1, dtype="uint1")
                tmp_allocate = make.Allocate(buffer_var, dtype,
                                             _SCALAR_EXTENTS,
                                             tmp_const, pre_stmt)
                ir_generator.source_info.set_node_loc(tmp_allocate,
                                                      loc=_instance_func_loc)
                tmp_attr = make.AttrStmt(
                    buffer_var, "storage_scope", tmp_string_imm,
                    tmp_allocate)
                ir_generator.source_info.set_node_loc(tmp_attr,
                                                      loc=_instance_func_loc)
                return tmp_attr

            self.instance_func = _instance_func
        else:
            self._name = name + str(Scalar.COUNT)
            self.reg_buffer = ir_generator.allocate(
                dtype, _SCALAR_EXTENTS, name=name + str(Scalar.COUNT),
                scope=scope_reg)
            ir_generator.code_scalar_manager.new_scalar(self)
            if init_value is not None:
                self.set_as(init_value)

    @property
    def name(self):
        """
        get scalar register name
        Parameters
        ----------

        Returns
        ----------
        return:scalar register name
        """
        msg = "Scalar %s is not defined in this scope." % self._name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        return self._name

    @property
    def if_global_scope(self):
        """
        judging whether is global scope
        Parameters
        ----------

        Returns
        ----------
        return:judging whether is global scope
        """
        msg = "Scalar %s is not defined in this scope." % self._name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        return self._if_global_scope

    def get(self):
        """
        get buffer
        Parameters
        ----------

        Returns
        ----------
        return:return top register buffer
        """
        msg = "Scalar %s is not defined in this scope." % self._name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        return self.reg_buffer[0]

    def merge_scalar(self, body):
        """
        merge sclar
        Parameters
        ----------

        Returns
        ----------
        return:sclar
        """
        msg = "Scalar %s is not defined in this scope." % self._name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        if self.instance_func is not None:
            return self.instance_func(body)
        return body
    # @endcond

    @source_info_decorator()
    @debug.scalar_set_as_decorator
    def set_as(self, value, src_offset=None):
        '''
        Sets the scalar value.
        Description:
          Sets the scalar value.

        Args:
          value : Value to be assigned from:
            - An immediate of type int or float
            - A Scalar variable
            - A Tensor value
            - An Expr (consisting of a Scalar variable and an immediate)
          src_offset: An internal optional parameter

        Restrictions:
          - Scalar value assignment between different data types is not
          supported, for example, between float16
          and float32.
          - Scalar value assignment between int/uint and float16/float32 is
          not supported.
          - Value assignment from an Expr of any type to a float16/float32
          scalar is not supported.
          - Value assignment from an Expr to an int/uint scalar is supported
          only when the Expr's scalar is of type int
          or uint and the Expr's immediate is of type int or float.

        Returns:
          None

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))

            index_reg = tik_instance.Scalar(dtype = "int32")
            index_reg.set_as(10)

            index_reg2 = tik_instance.Scalar(dtype = "float16")
            index_reg2.set_as(10.2)

            # A Scalar variable
            index_reg3 = tik_instance.Scalar(dtype = "float16")
            index_reg3.set_as(index_reg2)

            # A Tensor value
            data_A = tik_instance.Tensor("float16", (128,), name="data_A",
                                        scope=tik.scope_ubuf)
            index_reg3.set_as(data_A[0])

            #An Expr
            index_reg4 = tik_instance.Scalar(dtype = "int32")
            index_reg4.set_as(index_reg+20)
        '''
        # pylint: disable=W0622, W0601, C0103
        # disable it because it's to support python3
        from .tik_tensor import Tensor
        global long
        msg = "Scalar %s is not defined in this scope." % self._name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self._available, True, msg)
        if sys.version_info[PYTHON_VERSION_IDX] >= PYTHON_VERSION3:
            long = int
        if isinstance(value, Tensor):
            if src_offset is None and not value.is_single_point():
                TikCheckUtil.raise_error("value is not a single point")
            self.ir_generator.assign(self, value, src_offset=src_offset)
        else:
            if isinstance(value, BasicExpr):
                val = value.astype(self.dtype).get()
            elif isinstance(value, (float, int)):
                val = tvm.const(value, self.dtype)
            else:
                val = value
            self.reg_buffer[0] = val

    # @cond
    def disable_scalar(self):
        """
         when this scalar lifecycle is in the end,
         this scalar condition parameter should be changed from true to false.
         Parameters
         ----------
         parameter:No parameter

         Returns
         ----------
         return:no return
        """
        self._available = False

    def eval_value(self):
        """
        return expression value
        Parameters
        ----------

        Returns
        ----------
        return:no return
        """
    # @endcond


# @cond
class ScalarArray():
    """
    hint:scalar expression
    """

    # pylint: disable=R0903
    def __init__(self, ir_generator, dtype, length, name, init_value=None):
        """
        scalar array initialization
        Parameters
        ----------
        ir_generator:Halide IR generator
        dtype:scalar array data type
        length:scalar array length
        name:scalar array  name
        init_value:scalar array value

        Returns
        ----------
        return:no return
        """
        # pylint: disable=R0913
        # check dtype is str
        TikCheckUtil.check_type_match(dtype, str, "dtype should be str")
        shape = length
        if not isinstance(length, (list, tuple)):
            shape = [length]
        Scalar.COUNT += 1
        self.ir_generator = ir_generator
        self.reg_buffer = ir_generator.allocate(
            dtype, shape, name=name+str(Scalar.COUNT), scope=scope_reg,
            init_value=init_value)

    @property
    def data(self):
        """
        return scalar array data
        Parameters
        ----------

        Returns
        ----------
        return:scalar array data
        """
        return self.reg_buffer
    # @endcond


def _set_mask(length):
    """
    calculate MASK in cce

    Parameters
    ----------
    length : int
        calculate length

    Returns
    -------
    mask : tuple of int
        low and high bit of mask.
    """
    length = int(length)
    # high 64bits
    mask1 = 2**max(length - 64, 0) - 1
    # low 64bits
    mask2 = 2**min(length, 64) - 1
    return mask1, mask2


def _mask_concat_counter_mode(tik, mask):
    TikCheckUtil.check_type_match(
        mask, (int, Expr, Scalar),
        "In counter mode, mask should be int, Expr or Scalar, "
        "input type of mask: {}".format(type(mask)))
    check_scalar_dtype(mask, "scalar_mask should be a scalar of int/uint")
    if isinstance(mask, int):
        check_integer_in_range(
            mask, range(MIN_MASK, MAX_COUNTER_MASK),
            "In counter_mode, mask value should be in the range of "
            "[1, 2**32-1], input mask: %d" % mask)
        return type_convert([0, mask], "uint64")
    mask_h = tik.Scalar_(dtype="uint64", init_value=MASK_VALUE_ZERO)
    return type_convert([mask_h.get(), mask.get()], "uint64")


def mask_concat(tik, mask, mask_mode="normal", tensor_bit_len=BIT_LEN_16):
    """concat mask"""
    with tik.context.freeze():
        ret = _mask_concat_freeze(tik, mask, mask_mode=mask_mode,
                                  tensor_bit_len=tensor_bit_len)
    return ret


def _mask_concat_freeze(tik, mask, mask_mode="normal",
                        tensor_bit_len=BIT_LEN_16):
    """concat mask, frozen"""

    # mask_mode: counter
    if mask_mode == "counter":
        return _mask_concat_counter_mode(tik, mask)

    # mask_mode: normal
    if not isinstance(mask, (list, tuple)):
        mask = [mask]
    # continous mode
    if (len(mask) == MASK_LEN_CONTINOUS_MODE) and (is_basic_expr(mask)):
        for msk in mask:
            check_scalar_dtype(msk,
                               "scalar_mask should be"
                               " a scalar of int/uint")
        mask_64 = tik.Scalar_(dtype="uint64")
        mask_64.set_as(MASK_VALUE_ZERO)
        mask_low_32 = tik.Scalar_(dtype="uint64")
        mask_low_32.set_as(MAX_MASK_LOW_VALUE)

        mask_64.set_as((mask_low_32 << MASK_LOW_SHIFT) | mask_low_32)

        mask = mask[_SCALAR_MASK_VALUE_IDX]
        mask_h = tik.Scalar_(dtype="uint64")
        mask_l = tik.Scalar_(dtype="uint64")
        tik.scope_attr(cce_params.CCE_AXIS, "if_protect", PIPE_MTE1)
        with tik.if_scope_(mask < MASK_VALUE_64):
            mask_h.set_as(MASK_VALUE_ZERO)
            mask_l.set_as(((Expr(1, "uint64") <<
                            mask.astype("uint64")) - 1).astype("uint64"))
        with tik.else_scope_():
            with tik.if_scope_(mask < MASK_VALUE_128):
                mask_h.set_as(((Expr(1, "uint64") <<
                                (mask.astype("uint64") -
                                 MASK_VALUE_64)) - 1).astype("uint64"))
            with tik.else_scope_():
                mask_h.set_as(mask_64)
            mask_l.set_as(mask_64)
        return type_convert([mask_h.get(), mask_l.get()], "uint64")
    if (len(mask) == MASK_LEN_CONTINOUS_MODE) and \
            (is_immediate_number(mask)):
        # check mask
        for msk in mask:
            TikCheckUtil.check_type_match(msk, int,
                                          "mask should be int, input type is {}"
                                          "".format(type(msk)))
        # for immediate mask, value should  be in range of [1,128], b16
        if tensor_bit_len == BIT_LEN_16:
            TikCheckUtil.check_in_range(
                mask[0], range(MIN_MASK, MAX_MASK),
                "mask value should be in the range of [1, 128] for b16 "
                "tensor, input mask: {}".format(mask[0]))
        # b32, [1, 64]
        else:
            TikCheckUtil.check_in_range(
                mask[0], range(MIN_MASK, MAX_MASK_64),
                "mask value should be in the range of [1, 64] for b32 "
                "tensor, input mask: {}".format(mask[0]))
        mask_h, mask_l = _set_mask(mask[_SCALAR_MASK_VALUE_IDX])
        return type_convert([mask_h, mask_l], "uint64")
    # full mode and others
    return _mask_concat_normal_full_mode_and_other(mask, tensor_bit_len)


def _mask_concat_normal_full_mode_and_other(mask, tensor_bit_len):
    if (len(mask) == MASK_LEN_FULL_MODE) and (is_basic_expr(mask)):
        for msk in mask:
            check_scalar_dtype(msk,
                               "scalar_mask should be"
                               " a scalar of int/uint")
        return type_convert(mask, "uint64")
    if (len(mask) == MASK_LEN_FULL_MODE) and (is_immediate_number(mask)):
        # check mask
        check_imme_mask_full_mode(mask, tensor_bit_len)
        return type_convert(mask, "uint64")
    # others
    return TikCheckUtil.raise_error("not support this type of mask now")

# @cond
class InputScalar(Expr):
    """
    only use to buildcce's input
    """
    COUNT = 0

    def __init__(self, ir_generator, dtype="int64", name="reg_buf"):
        """
        init inputscalar
        :param ir_generator:
        :param dtype: dtype of var
        :param name: name of inputscalar
        """
        TikCheckUtil.check_name_str_valid(name)
        TikCheckUtil.check_type_match(dtype, str, "dtype should be str")

        super(InputScalar, self).__init__(expr_="nouse")
        InputScalar.COUNT += 1
        self.ir_generator = ir_generator
        self._name = name   # + str(InputScalar.COUNT)
        self._var = tvm.var(name=self._name, dtype=dtype)
        self.scope = scope_gm
        self.shape = (1,)
        self.debug_var = tvm.var('scalar_debug_var_' + name, dtype)
        ir_generator.scope_attr(self._var, "tik_scalar", self._var)

    @property
    def name(self):
        """
        get scalar name
        :return:
        """
        return self._name

    def get(self):
        """
        get buffer
        :return:
        """
        return self._var
# @endcond
