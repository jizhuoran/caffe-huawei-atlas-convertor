"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     context.py
DESC:     debug context
CREATED:  2019-7-04 20:12:13
MODIFIED: 2019-7-24 10:54:23
"""
# disabling:
# W0622: redefined-builtin
# W0601: global-variable-undefined
# C0103: invalid-name
# R0903: too-few-public-methods
# R0902: too-many-instance-attributes
# E1101: no-member

from __future__ import absolute_import, print_function
import sys
import numpy as np

from te.tik.common.util import DTYPE_INT_VALUE, get_check_feed_dict
from te.tik.common.common_util import is_scalar
from .tensor_buffer import TensorBuffer
from .util import safe_get_value, make_tvm_imm
from .sim import PVModel, Encoder
from .statement import TikDebug, DebugReturnException
from ..tik_lib.tik_source_info import TikSourceInfo, source_info_decorator
from ..tik_lib.tik_check_util import TikCheckUtil

class SPRProxy():
    """special_register proxy"""
    # pylint: disable=R0903
    def __init__(self, context):
        self.context = context

    def __getitem__(self, name):
        TikCheckUtil.check_not_is(
            self.context.model, None, "self.context.model is None")
        return self.context.model.read_spr(name)


class FrozenContext():
    """frozen context"""
    def __init__(self, context):
        self.context = context

    def __enter__(self):
        self.context.frozen += 1
        self.context.curr_scope().frozen += 1

    def __exit__(self, ptype, value, trace):
        self.context.frozen -= 1
        self.context.curr_scope().frozen -= 1

_SCOPE_LENGTH = 1
_LAST_SCOPE = -1

class Context():
    """
        Data binding:
            tensor ->  NumpyBuffer (buffer_mapping)
            var    ->  callback_handle (var_table)
            SPR    ->  a proxy to PVModel
    """
    # pylint: disable=R0902

    tik_debug = None
    force_interactive = False

    def __init__(self, dprofile):
        self.dprofile = dprofile
        self.scope = []
        from .statement import Block
        top_block = Block()
        top_block.traceable = False
        self.scope.append(top_block)
        self.tensor_buffer = TensorBuffer()
        self.placeholders = {}
        self.outs = []
        self.store_stmt = []
        self.var_table = {}
        self.feed_var_table = {}
        self.spr_proxy = SPRProxy(self)
        self.model = None
        self.encoder = Encoder()
        self.interactive = True
        self.scalar_mapping = {}
        self.frozen = False
        self.scalar_location = {}
        self.init_mode = 'random'
        self.init_value = None
        self.tensor_list = []

    def freeze(self):
        """Get frozen context

        Returns
        -------
        frozen context
        """
        return FrozenContext(self)

    def add_scope(self, scope):
        """add scope to stmt

        Parameters
        ----------
        scope : which ub
        """
        if self.frozen:
            return
        self.curr_scope().add_stmt(scope)
        self.scope.append(scope)

    def pop_scope(self):
        """pop scope"""
        if self.frozen:
            return
        self.scope.pop()

    def curr_scope(self):
        """return cur scope

        Returns
        -------
        the last scope
        """
        return self.scope[_LAST_SCOPE]

    def eval_(self,  # pylint: disable=R0914
              feed_dict):
        """get the result from buffer

        Parameters
        ----------
        feed_dict : symbol dict

        Returns
        -------
        the result
        """
        TikCheckUtil.check_type_match(
            feed_dict, dict, "feed_dict should be type of dict, input feed_dict"
                             " type is %s" % type(feed_dict))
        # check feed_dict with placeholder
        build_cce_input_tensor_names = " ".join(list(self.placeholders.keys()))
        build_cce_input_var_names = " ".join(list(self.feed_var_table.keys()))
        build_cce_input_names = build_cce_input_tensor_names + " " + build_cce_input_var_names
        build_list_tensor = set(
            self.placeholders.keys()).union(set(self.feed_var_table.keys()))
        feed_dict_tensor, feed_dict_var = get_check_feed_dict(
            feed_dict, self.placeholders.keys(),
            self.feed_var_table.keys(), build_list_tensor,
            build_cce_input_names, build_cce_input_tensor_names,
            build_cce_input_var_names)

        TikCheckUtil.check_equality(len(self.scope), _SCOPE_LENGTH,
                                    "self.scope length must equal to 1")

        for key, value in feed_dict_tensor.items():
            tvm_buffer = self.placeholders[key]
            np_buffer = self.tensor_buffer.get_npbuffer_by_tvmbuffer(tvm_buffer)
            TikCheckUtil.check_equality(
                np_buffer.buffer.shape, value.shape,
                "%s input shape mismatch %s vs %s" % (
                    key, np_buffer.buffer.shape, value.shape))
            TikCheckUtil.check_equality(
                np_buffer.buffer.dtype, value.dtype,
                "%s input dtype mismatch %s vs %s" % (
                    key, np_buffer.buffer.dtype, value.dtype))
            np_buffer.buffer[:] = np.ascontiguousarray(value)

        for key, value in feed_dict_var.items():
            build_cce_input_var = self.get_feed_var_by_name(key)
            if build_cce_input_var.dtype.startswith("int") or\
                    build_cce_input_var.dtype.startswith("uint"):
                TikCheckUtil.check_type_match(value, int, key + " is " +
                                              build_cce_input_var.dtype +
                                              ", but value is float!")
                TikCheckUtil.check_in_range(
                    value,
                    range(DTYPE_INT_VALUE[build_cce_input_var.dtype][0],
                          DTYPE_INT_VALUE[build_cce_input_var.dtype][1] + 1),
                    "{} is {} type, should in [{}, {}], but get {}".format(
                        key, build_cce_input_var.dtype,
                        DTYPE_INT_VALUE[build_cce_input_var.dtype][0],
                        DTYPE_INT_VALUE[build_cce_input_var.dtype][1],
                        value))
            if build_cce_input_var.dtype.startswith("float"):
                TikCheckUtil.check_type_match(
                    value, float,
                    key + " is " + build_cce_input_var.dtype +
                    ", but value is int!")
            self.update_var(build_cce_input_var, value)

        if TikDebug.tik_debug:
            TikDebug.force_interactive = True
        self.model = PVModel(self.dprofile)
        # before evaluate stmt, clear source info, such as start_debug
        TikSourceInfo.clear_source_info()
        try:
            scope = self.curr_scope()
            scope.evaluate(self)
        except DebugReturnException:
            pass
        finally:
            self.model = None

        ret = []
        for out in self.outs:
            ret.append(self.tensor_buffer.get_npbuffer_by_tvmbuffer(out).buffer)
        return ret

    @source_info_decorator()
    def eval_and_compare(self, feed_dict, golden=None):
        """Comparing result against golden

        Parameters
        ----------
        feed_dict : symbol dict

        Returns
        -------
        error message
        """
        output_data = self.eval_(feed_dict)
        print("[INFO]: Comparing result against golden")
        errors = []
        if golden is not None:
            TikCheckUtil.check_type_match(golden, list, "golden must be list")
            for i, value in enumerate(golden):
                np_buffer = output_data[i]
                diff = (np.abs(
                    np_buffer.reshape(-1).astype(np.float64) -
                    value.reshape(-1).astype(np.float64)).reshape(-1, 16))
                error = np.average(diff)
                errors.append(error)
                print('output[{}] error:{}'.format(i, error))

        return errors

    def add_tvm_store_stmt(self, stmt):
        """add tvm to store stmt

        Parameters
        ----------
        stmt: statement
        """
        self.store_stmt.append(stmt)

    def set_scalar_var_mapping(self, scalar_buf_var, shadow_var):
        """set scalar by mapping

        Parameters
        ----------
        scalar_buf_var : key
        shadow_var : value
        """
        self.scalar_mapping[scalar_buf_var.asnode()] = shadow_var

    def set_scalar_location(self, scalar, source_info):
        """set scalar source_info

        Parameters
        ----------
        scalar : key
        source_info : value
        """
        self.scalar_location[id(scalar)] = source_info

    def get_scalar_location(self, scalar):
        """get scalar source_info

        Parameters
        ----------
        scalar : key

        Returns
        -------
        source_info
        """
        return self.scalar_location[id(scalar)]

    def bind_var(self, var):
        """init the var in var_table with None

        Parameters
        ----------
        var : key
        """
        self.var_table[var] = None

    def get_feed_var_by_name(self, name):
        """
        find inputscalr in feed_var_table
        :param name: inputscalr's name
        :return:
        """
        return self.feed_var_table[name]

    def bind_feed_var(self, name, var):
        """
        use to put inputscalar with value
        :param name: inputscalar's name
        :param var: inputscalr's var
        :return:
        """
        self.feed_var_table[name] = var
        self.var_table[var] = None

    def update_var(self, var, value):
        """update the var in var_table with value

        Parameters
        ----------
        var : key
        value : value
        """
        # caused by using global long, so disable them
        # pylint: disable=W0622, W0601, C0103
        from te.tvm.expr import UIntImm, FloatImm, IntImm
        from ..tik_lib.tik_params import PYTHON_VERSION_IDX, PYTHON_VERSION3
        global long
        if sys.version_info[PYTHON_VERSION_IDX] >= PYTHON_VERSION3:
            long = int
        TikCheckUtil.check_type_match(
            value, (float, int, long, bool, UIntImm, FloatImm, IntImm),
            'update_var got invalid value {} type {}'.format(value,
                                                             type(value)))
        self.var_table[var] = value

    def evaluate_expr(self, original_expr):
        """run the expr

        Parameters
        ----------
        original_expr : the cmd string

        Returns
        -------
        the result of original_expr
        """
        # caused by some interface of te.tvm such as tvm.make.Evaluate, so disable them
        # pylint: disable=E1101
        expr = original_expr
        from te import tvm
        from ..tik_lib.tik_expr import Expr

        def process_expr_scalar(expr):
            """get expr scalar"""
            if is_scalar(expr):
                return expr.debug_var
            if isinstance(expr, Expr):
                return expr.get()
            return expr

        if isinstance(expr, (int, float, bool, complex)):
            return expr
        expr = process_expr_scalar(expr)
        expr = tvm.subsitute_scalar(expr, self.scalar_mapping)
        prev_expr = tvm.make.Evaluate(expr)
        let_ = prev_expr
        for var, var_value in self.var_table.items():
            var = process_expr_scalar(var)
            if var_value is None:
                if is_scalar(original_expr):
                    TikCheckUtil.check_not_is(
                        var, original_expr.debug_var,
                        '[Error]: found uninitialized Scalar !\nScalar define '
                        'location:\n{}'.format(
                            self.get_scalar_location(original_expr)))
                # var may not be defined now, just skip it
                continue
            if isinstance(var_value, (tvm.expr.IntImm, tvm.expr.FloatImm,
                                      tvm.expr.StringImm, tvm.expr.UIntImm)):
                var_value = safe_get_value(var_value)
            let_ = tvm.make.LetStmt(var, make_tvm_imm(var.dtype, var_value),
                                    prev_expr)
            prev_expr = let_
        result = tvm.ir_pass.Simplify(let_)
        while 'value' in dir(result):
            result = safe_get_value(result)
        return result

    def add_tensor(self, tensor):
        """add tensor to buffer

        Parameters
        ----------
        tensor : the added tensor
        """
        self.tensor_buffer.add_tensor(tensor)

    def add_proxy(self, proxy_tensor, buffer_, indice, dtype):
        """add tensor proxy

        Parameters
        ----------
        proxy_tensor : the added tensor
        buffer_: buffer
        indice: tensor indice
        dtype: tensor dtype
        """
        self.tensor_buffer.add_tensor_proxy(self, proxy_tensor, buffer_, indice, dtype)

    def get_value(self, tensor):
        """get tensor from buffer_mapping

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        buffer
        """
        try:
            return self.tensor_buffer.get_npbuffer_by_tensor(tensor)
        except KeyError:
            return None

    def record_ins(self, ins):
        """record instruction

        Parameters
        ----------
        ins : instruction
        """
        self.placeholders = {i.name: i for i in ins}

    def record_outs(self, outs):
        """record the outs

        Parameters
        ----------
        outs : the outs
        """
        self.outs = outs
