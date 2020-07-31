"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_expr.py
DESC:     expr
CREATED:  2019-04-18 18:53:42
MODIFIED: 2019-07-18 21:29:18
"""
# disabling:
# W0601: global-variable-undefined(for the compatibly of python2 to python3)
# W0622: redefined-builtin(for the compatibly of python2 to python3)
# C0103: invalid-name(for the compatibly of python2 to python3)
# E1101: no-member

import sympy  # pylint: disable=E0401

from te import tvm
from te.tvm import make
from te.tvm.expr import Call
from te.tvm.ir_pass import Simplify
from .tik_params import DEFAULT_VALUE_INDEX
from .tik_check_util import TikCheckUtil
from .tik_source_info import source_info_decorator

# some one should clean unused items after use
floor_divs = set()  # pylint: disable=C0103

# for simplify expr
_SYMPY_SIMPLIFY_MAP = {
    tvm.expr.Cast: lambda expr: _sympy_simplify(expr.value),
    tvm.expr.Var: lambda expr: sympy.symbols(str(expr)),
    tvm.expr.Add: lambda expr: _sympy_simplify(expr.a) +
                  _sympy_simplify(expr.b),
    tvm.expr.Sub: lambda expr: _sympy_simplify(expr.a) -
                  _sympy_simplify(expr.b),
    tvm.expr.Mul: lambda expr: _sympy_simplify(expr.a)*
                  _sympy_simplify(expr.b),
    tvm.expr.Mod: lambda expr: _sympy_simplify(expr.a) %
                  _sympy_simplify(expr.b),
    tvm.expr.FloorMod: lambda expr: _sympy_simplify(expr.a) %
                       _sympy_simplify(expr.b),
    tvm.expr.Min: lambda expr: sympy.Min(
        _sympy_simplify(expr.a), _sympy_simplify(expr.b)),
    tvm.expr.Max: lambda expr: sympy.Max(
        _sympy_simplify(expr.a), _sympy_simplify(expr.b)),
    tvm.expr.EQ: lambda expr: sympy.Eq(
        _sympy_simplify(expr.a), _sympy_simplify(expr.b)),
    tvm.expr.NE: lambda expr: sympy.Ne(
        _sympy_simplify(expr.a), _sympy_simplify(expr.b)),
    tvm.expr.LT: lambda expr: sympy.Lt(
        _sympy_simplify(expr.a), _sympy_simplify(expr.b)),
    tvm.expr.LE: lambda expr: sympy.Le(
        _sympy_simplify(expr.a), _sympy_simplify(expr.b)),
    tvm.expr.GT: lambda expr: sympy.Gt(
        _sympy_simplify(expr.a), _sympy_simplify(expr.b)),
    tvm.expr.GE: lambda expr: sympy.Ge(
        _sympy_simplify(expr.a), _sympy_simplify(expr.b)),
    tvm.expr.And: lambda expr: sympy.And(
        _sympy_simplify(expr.a), _sympy_simplify(expr.b)),
    tvm.expr.Or: lambda expr: sympy.Or(
        _sympy_simplify(expr.a), _sympy_simplify(expr.b)),
    tvm.expr.Not: lambda expr: sympy.Not(_sympy_simplify(expr.a)),
    tvm.expr.Load: lambda expr: _sympy_simplify(expr.buffer_var)
}
_SIMPLIFY_KEYS = tuple(_SYMPY_SIMPLIFY_MAP.keys())


class BasicExpr():
    """
    hint: basic expression
    """
    def __init__(self):
        self.class_type = Expr

    @property
    def dtype(self):
        """
        get data type
        Parameters
        ----------

        Returns
        ----------
        data type
        """
        return self.get().dtype

    def get(self):
        """
        get error information
        Parameters
        ----------

        Returns
        ----------
        error information
        """
        TikCheckUtil.raise_error(
            "This is a BASIC type of Scalar ad expr : %s" % self)

    @source_info_decorator()
    def __add__(self, other):
        """
        add
        Parameters
        ----------
        other:is added
        Returns
        ----------
        add result
        """
        othert = Expr(other, self.dtype)
        # here self.get() + othert.get() return tvm BinaryOpNode.Add
        return self.class_type(self.get() + othert.get())

    @source_info_decorator()
    def __radd__(self, other):
        """
        reverse add
        Parameters
        ----------
        other:add
        Returns
        ----------
        add result
        """
        othert = Expr(other, self.dtype)
        return self.__add__(othert)

    @source_info_decorator()
    def __sub__(self, other):
        """
        subtraction
        Parameters
        ----------
        other:is subtracted
        Returns
        ----------
        subtraction result
        """
        othert = Expr(other, self.dtype)
        return self.class_type(self.get() - othert.get())

    @source_info_decorator()
    def __rsub__(self, other):
        """
        reverse subtraction
        Parameters
        ----------
        other:subtract
        Returns
        ----------
        subtraction result
        """
        # disable it because pylint can't find symbol in back-end so
        othert = Expr(other, self.dtype)
        return self.class_type(make.Sub(othert.get(),  # pylint: disable=E1101
                                        self.get()))

    @source_info_decorator()
    def __mul__(self, other):
        """
        multiplication
        Parameters
        ----------
        other:is multiplied
        Returns
        ----------
        multiplication result
        """
        othert = Expr(other, self.dtype)
        return self.class_type(self.get()*othert.get())

    @source_info_decorator()
    def __rmul__(self, other):
        """
        reverse multiplication
        Parameters
        ----------
        other:multiplied
        Returns
        ----------
        multiplication result
        """
        return self.__mul__(other)

    @source_info_decorator()
    def __div__(self, other):
        """
        divide
        Parameters
        ----------
        other:is divided
        Returns
        ----------
        divide result
        """
        # disable it because pylint can't find symbol in back-end so
        othert = Expr(other, self.dtype)
        return self.class_type(make.Div(self.get(),  # pylint: disable=E1101
                                        othert.get()))

    @source_info_decorator()
    def __rdiv__(self, other):
        """
        reverse divide
        Parameters
        ----------
        other:divide
        Returns
        ----------
        divide result
        """
        # disable it because pylint can't find symbol in back-end so
        othert = Expr(other, self.dtype)
        return self.class_type(make.Div(othert.get(),  # pylint: disable=E1101
                                        self.get()))

    @source_info_decorator()
    def __truediv__(self, other):
        """
        true divide
        Parameters
        ----------
        other:divide
        Returns
        ----------
        true divide result
        """
        return self.__div__(other)

    @source_info_decorator()
    def __rtruediv__(self, other):
        """
        reverse true divide
        Parameters
        ----------
        other:is divided
        Returns
        ----------
        true divide result
        """
        return self.__rdiv__(other)

    @source_info_decorator()
    def __floordiv__(self, other):
        """
        floor divide
        Parameters
        ----------
        other:is operated
        Returns
        ----------
        floor divide result
        """
        ret = self.__div__(other)
        floor_divs.add(ret.expr_)
        return ret

    @source_info_decorator()
    def __rfloordiv__(self, other):
        """
        reverse floor divide
        Parameters
        ----------
        other:is operating
        Returns
        ----------
        floor divide result
        """
        ret = self.__rdiv__(other)
        floor_divs.add(ret.expr_)
        return ret

    @source_info_decorator()
    def __mod__(self, other):
        """
        modulus operator
        Parameters
        ----------
        other:is operated
        Returns
        ----------
        modulus operator result
        """
        # disable it because pylint can't find symbol in back-end so
        othert = Expr(other, self.dtype)
        return self.class_type(make.Mod(self.get(),  # pylint: disable=E1101
                                        othert.get()))

    @source_info_decorator()
    def __rmod__(self, other):
        """
        reverse modulus operator
        Parameters
        ----------
        other:is operating
        Returns
        ----------
        modulus operator result
        """
        # disable it because pylint can't find symbol in back-end so
        othert = Expr(other, self.dtype)
        return self.class_type(make.Mod(othert.get(),  # pylint: disable=E1101
                                        self.get()))

    @source_info_decorator()
    def __neg__(self):
        """
        negative operator
        Parameters
        ----------
        Returns
        ----------
        negative operator result
        """
        if self.dtype[:4].lower() == "uint":
            TikCheckUtil.raise_error("uint don't support negative.")
        return self.__mul__(tvm.const(-1, self.dtype))

    @source_info_decorator()
    def __lshift__(self, other):
        """
        shift operator
        Parameters
        ----------
        other:is operated
        Returns
        ----------
        shift operator result
        """
        # disable it because pylint can't find symbol in back-end so
        othert = Expr(other, self.dtype)
        tmp_node = make.Call(self.dtype,  # pylint: disable=E1101
                             "shift_left", [self.get(), othert.get()],
                             Call.PureIntrinsic, None,
                             DEFAULT_VALUE_INDEX)
        return self.class_type(tmp_node)

    @source_info_decorator()
    def __rshift__(self, other):
        """
        reverse shift operator
        Parameters
        ----------
        other:is operated
        Returns
        ----------
        shift operator result
        """
        # disable it because pylint can't find symbol in back-end so
        othert = Expr(other, self.dtype)
        tmp_node = make.Call(self.dtype,  # pylint: disable=E1101
                             "shift_right", [self.get(), othert.get()],
                             Call.PureIntrinsic, None,
                             DEFAULT_VALUE_INDEX)
        return self.class_type(tmp_node)

    @source_info_decorator()
    def __and__(self, other):
        """
        and operator
        Parameters
        ----------
        other:is Anded
        Returns
        ----------
        And result
        """
        # disable it because pylint can't find symbol in back-end so
        othert = Expr(other, self.dtype)
        tmp_node = make.Call(self.dtype,  # pylint: disable=E1101
                             "bitwise_and", [self.get(), othert.get()],
                             Call.PureIntrinsic, None,
                             DEFAULT_VALUE_INDEX)
        return self.class_type(tmp_node)

    @source_info_decorator()
    def __or__(self, other):
        """
        or operator
        Parameters
        ----------
        other:is operated
        Returns
        ----------
        or result
        """
        # disable it because pylint can't find symbol in back-end so
        othert = Expr(other, self.dtype)
        tmp_node = make.Call(self.dtype,  # pylint: disable=E1101
                             "bitwise_or", [self.get(), othert.get()],
                             Call.PureIntrinsic, None,
                             DEFAULT_VALUE_INDEX)
        return self.class_type(tmp_node)

    @source_info_decorator()
    def __xor__(self, other):
        """
        xor operator
        Parameters
        ----------
        other:is operated
        Returns
        ----------
        xor result
        """
        # disable it because pylint can't find symbol in back-end so
        othert = Expr(other, self.dtype)
        tmp_node = make.Call(self.dtype,  # pylint: disable=E1101
                             "bitwise_xor", [self.get(), othert.get()],
                             Call.PureIntrinsic, None,
                             DEFAULT_VALUE_INDEX)
        return self.class_type(tmp_node)

    @source_info_decorator()
    def __invert__(self):
        """
        invert operator
        Parameters
        ----------
        other:is operated
        Returns
        ----------
        invert result
        """
        # disable it because pylint can't find symbol in back-end so
        tmp_node = make.Call(self.dtype,  # pylint: disable=E1101
                             "bitwise_not", [self.get()],
                             Call.PureIntrinsic, None,
                             DEFAULT_VALUE_INDEX)
        return self.class_type(tmp_node)

    @source_info_decorator()
    def __lt__(self, other):
        """
        less than comparison
        Parameters
        ----------
        other:target object
        Returns
        ----------
        comparison result
        """
        # disable it because pylint can't find symbol in back-end so
        othert = Expr(other, self.dtype)
        return self.class_type(make.LT(self.get(),  # pylint: disable=E1101
                                       othert.get()))

    @source_info_decorator()
    def __le__(self, other):
        """
        less than or equal comparison
        Parameters
        ----------
        other:target object
        Returns
        ----------
        comparison result
        """
        # disable it because pylint can't find symbol in back-end so
        othert = Expr(other, self.dtype)
        return self.class_type(make.LE(self.get(),  # pylint: disable=E1101
                                       othert.get()))

    @source_info_decorator()
    def __eq__(self, other):
        """
        equal comparison
        Parameters
        ----------
        other:target object

        Returns
        ----------
        comparison result
        """
        othert = Expr(other, self.dtype)
        tmp_node = tvm.expr.EqualOp(self.get(), othert.get()).asnode()
        return self.class_type(tmp_node)

    @source_info_decorator()
    def __ne__(self, other):
        """
        Don't equal comparison
        Parameters
        ----------
        other:target object

        Returns
        ----------
        return:comparison result
        """
        othert = Expr(other, self.dtype)
        tmp_node = tvm.expr.NotEqualOp(self.get(), othert.get()).asnode()
        return self.class_type(tmp_node)

    @source_info_decorator()
    def __gt__(self, other):
        """
        greater than comparison
        Parameters
        ----------
        other:target object

        Returns
        ----------
        return:comparison result
        """
        # disable it because pylint can't find symbol in back-end so
        othert = Expr(other, self.dtype)
        return self.class_type(make.GT(self.get(),  # pylint: disable=E1101
                                       othert.get()))

    @source_info_decorator()
    def __ge__(self, other):
        """
        greater than or equal comparison
        Parameters
        ----------
        other:target object

        Returns
        ----------
        return:comparison result
        """
        # disable it because pylint can't find symbol in back-end so
        othert = Expr(other, self.dtype)
        return self.class_type(make.GE(self.get(),  # pylint: disable=E1101
                                       othert.get()))

    @source_info_decorator()
    def __nonzero__(self):
        """
        non zero comparison
        Parameters
        ----------

        Returns
        ----------
        return:comparison result
        """
        TikCheckUtil.raise_error(
            "Cannot use and / or / not operator to Expr, hint: " +
            "use tik.all / tik.any instead", exception_type=ValueError)

    @source_info_decorator()
    def __bool__(self):
        """
        bool type judement
        Parameters
        ----------

        Returns
        ----------
        return:judement result
        """
        return self.__nonzero__()

    def equal(self, other):
        """
        equal comparison
        Parameters
        ----------
        other:target object

        Returns
        ----------
        comparison result
        """
        # disable it because pylint can't find symbol in back-end so
        othert = Expr(other, self.dtype)
        return self.class_type(make.EQ(self.get(),  # pylint: disable=E1101
                                       othert.get()))

    @source_info_decorator()
    def astype(self, dtype):
        """
        set data type
        Parameters
        ----------
        dtype:data type

        Returns
        ----------
        return:result
        """
        return self.class_type(make._cast(dtype,  # pylint: disable=W0212, E1101
                                          self.get()))

    def reinterpret_cast_to(self, dtype):
        """
        transform  data type
        Parameters
        ----------
        dtype: transform data type

        Returns
        ----------
        return:result
        """
        tmp_node = tvm.call_extern(dtype, "reinterpret_cast", self.get())
        return self.class_type(tmp_node)

    def __hash__(self):
        """
        return hash value
        Parameters
        ----------

        Returns
        ----------
        return:hash value
        """
        return hash(self.get())

    def __str__(self):
        """
         return string value
         Parameters
         ----------

         Returns
         ----------
         return:string value
         """
        return str(self.get())

    def __repr__(self):
        """
         get repr value
         Parameters
         ----------

         Returns
         ----------
         return:repr() value
         """
        return str(self.get())


class Expr(BasicExpr):
    """
    hint:basic expression
    """
    def __init__(self, expr_, dtype=None):
        """
        class initilalization
        Parameters
        ----------
        expr_:expression data type
        dtype:basic expression data type

        Returns
        ----------
        return:no result
        """
        super(Expr, self).__init__()

        if isinstance(expr_, BasicExpr):
            if dtype is None:
                self.expr_ = expr_.get()
            else:
                self.expr_ = expr_.astype(dtype).get()
        elif isinstance(expr_, int):
            if dtype is None:
                self.expr_ = tvm.const(expr_, "int32")
            else:
                self.expr_ = tvm.const(expr_, dtype)
        elif isinstance(expr_, float):
            if dtype is None:
                self.expr_ = tvm.const(expr_, "float32")
            else:
                self.expr_ = tvm.const(expr_, dtype)
        else:
            self.expr_ = expr_
        self.class_type = Expr

    def get(self):
        """
        get data type
        Parameters
        ----------

        Returns
        ----------
        return:expression data type
        """
        return self.expr_

    def eval_value(self):
        """
        return expression value
        Parameters
        ----------

        Returns
        ----------
        return: expression value

        """
        if self.expr_ in floor_divs:
            tmp = _sympy_simplify(Simplify(self.get()), True)
        else:
            tmp = _sympy_simplify(Simplify(self.get()))
        if hasattr(tmp, 'is_number'):
            return _cvt_sympy_value(tmp)
        return tmp


def _sympy_simplify(expr, need_floor=False):
    """convert sympy type value
    Parameters
    ----------
    expr: expression

    Returns
    ----------
    return: sympy expr value
    """
    if not isinstance(expr, (_SIMPLIFY_KEYS,
                             tvm.expr.UIntImm, tvm.expr.IntImm,
                             tvm.expr.FloatImm, tvm.expr.Div,
                             tvm.expr.FloorDiv)):
        return None
    if isinstance(expr, tvm.expr.Div):
        ret = _sympy_simplify(expr.a) / _sympy_simplify(expr.b)
        if expr in floor_divs:
            ret = sympy.floor(ret)
        return ret
    if isinstance(expr, tvm.expr.FloorDiv):
        return sympy.floor(_sympy_simplify(expr.a) / _sympy_simplify(expr.b))
    if isinstance(expr, (tvm.expr.UIntImm,
                         tvm.expr.IntImm, tvm.expr.FloatImm)):
        if need_floor:
            # we need a flag to do floor, because before tvm.Simplify
            # the expr isinstance div, but it will be Imm after tvm.Simplify
            return sympy.floor(expr.value)
        return expr.value
    return _SYMPY_SIMPLIFY_MAP[type(expr)](expr)


def _cvt_sympy_value(sympy_expr_value):
    """convert sympy type value
    Parameters
    ----------
    sympy_expr_value: sympy expr value

    Returns
    ----------
    return: sympy type value
    """
    if getattr(sympy_expr_value, "is_Float", False):
        result = float(sympy_expr_value)
    elif getattr(sympy_expr_value, "is_Integer", False):
        result = int(sympy_expr_value)
    elif getattr(sympy_expr_value, "is_Boolean", False):
        result = int(bool(sympy_expr_value))
    elif getattr(sympy_expr_value, "is_Rational", False):
        result = float(sympy_expr_value)
    elif sympy_expr_value is sympy.nan:
        result = float('nan')
    else:
        result = None
    return result
