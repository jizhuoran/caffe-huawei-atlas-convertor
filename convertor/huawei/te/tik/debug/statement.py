"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     statement.py
DESC:     Create some stmt class
CREATED:  2019-04-18 18:53:42
MODIFIED: 2019-07-16 20:14:42
"""
# disabling:
# C0302: too-many-lines
# R0903: too-few-public-methods
# W0123: eval-used
# C0103: invalid-name
# pylint: disable=C0302

from __future__ import print_function
import sys
import copy
import ast

import numpy as np

from te.tik.common.util import TikUtil
from te.tik.common.common_util import is_tensor
from te.tik.tik_lib.tik_check_util import TikCheckUtil, get_traceback_msg
from .util import get_flatten_idx, get_dtype_bit_width, make_tvm_imm
from ..tik_lib.tik_params import UINT64_BIT, CUR_FRAME_IDX
from ..tik_lib.tik_source_info import TikSourceInfo, stack, \
    most_recent_traceback
from ..tik_lib.tik_expr import BasicExpr


class TikDebug():
    """
        Class for tik_debug and  force_interactive variable
    """
    # pylint: disable=R0903
    tik_debug = None
    force_interactive = False

    def __init__(self):
        pass

    @staticmethod
    def set_trace(tik_debug):
        """Setup trace function for AST evaluation"""
        TikDebug.tik_debug = tik_debug


class STMT():
    """
    Base Class for dbg_instances, used for building the debug AST
    """
    def __init__(self, source_info):
        """Initalize class STMT.

        Parameters
        ----------
        source_info:source code information
            It represents the relationship of current node with source code

        Returns
        ----------
        No returns
        """
        self.source_info = source_info
        self.parent = None
        self.break_point = None
        self.trace_event = 'statement'
        self.traceable = True
        self.frozen = False

        if TikDebug.tik_debug:
            TikDebug.tik_debug.register_debug_info(self.source_info, self)

    def evaluate(self, context):
        """evaluate function

        Parameters
        ----------
        context:information of debugger
            store all of debugger's information

        Returns
        ----------
        No returns
        """
        error_code = -1
        success_code = 0
        if TikDebug.tik_debug and self.traceable:
            TikDebug.tik_debug.trace(context, self, 'pre-%s' % self.trace_event)

        try:
            self.eval_(context)
        except (RuntimeError, ValueError, OSError) as ex:
            print("\n".join(get_traceback_msg(stack(depth=1))))
            # remove first element of list: "Traceback:"
            print("\n".join(get_traceback_msg(most_recent_traceback())[1:]))
            # in the interactive mode, enter cmdline ,
            # let user do some debug job
            print(ex)
            if context.interactive:
                TikDebug.tik_debug.running = False
                TikDebug.tik_debug.trace(context, self, 'exception')
            else:
                sys.exit(error_code)
        except SystemExit as exception:
            # success_code means user quit
            if exception.code == success_code:
                sys.exit(success_code)
            if exception.code != error_code:
                print(exception)
            # in the interactive mode, enter cmdline ,
            # let user do some debug job
            if context.interactive:
                TikDebug.tik_debug.running = False
                TikDebug.tik_debug.trace(context, self, 'exception')
            else:
                sys.exit(error_code)

    def eval_(self, context):
        """eval_ function
            implement by subclass"""

    def set_parent(self, parent):
        """ Add the parent AST node

        Parameters
        ----------
        parent:the parent AST node

        Returns
        ----------
        No returns
        """
        if self.parent:
            msg = "Parent exist: {}, {}".format(id(self.parent),
                                                type(self.parent))
            print(msg)
            TikCheckUtil.raise_error(msg)
        self.parent = parent


class Block(STMT):
    """
    Class Block inherits from STMT
    A block of statement.
    """
    def __init__(self, source_info=''):
        """Initalize class Block.

        Parameters
        ----------
        source_info:str
            Information of debugger

        Returns
        ----------
        No returns
        """
        super(Block, self).__init__(source_info)
        self.stmts = []
        self.trace_event = 'block'

    def add_stmt(self, stmt):
        """Add stmt node to current node

        Parameters
        ----------
        stmt:instance of STMT
            AST node

        Returns
        ----------
        No returns
        """
        if self.frozen:
            return

        stmt.set_parent(self)
        self.stmts.append(stmt)

    def eval_(self, context):
        """eval_ function
        evaluate all of self.function

        Parameters
        ----------
        context:information of debugger
            store all of debugger's information

        Returns
        ----------
        No returns
        """
        for stmt in self.stmts:
            if not isinstance(stmt, (ForLoop, IfScope, HighLevelAPIScope)):
                TikSourceInfo.register_source_info(source_info=stmt.source_info)
            stmt.evaluate(context)
            if not isinstance(stmt, (ForLoop, IfScope, HighLevelAPIScope)):
                TikSourceInfo.clear_source_info()


class HighLevelAPIScope(STMT):
    """
    Hide tik library implementation in debug process
    """
    def __init__(self, source_info=''):
        """Initalize class Block.

        Parameters
        ----------
        source_info:str
            Information of debugger

        Returns
        ----------
        No returns
        """
        super(HighLevelAPIScope, self).__init__(source_info)
        self.stmts = []
        self.trace_event = 'block'

    def add_stmt(self, stmt):
        """Add stmt node to current node

        Parameters
        ----------
        stmt:instance of STMT
            AST node

        Returns
        ----------
        No returns
        """
        if self.frozen:
            return

        stmt.set_parent(self)
        self.stmts.append(stmt)

    def eval_(self, context):
        """eval_ function
        evaluate all of self.function

        Parameters
        ----------
        context:information of debugger
            store all of debugger's information

        Returns
        ----------
        No returns
        """
        # note: try not add check here!
        old_running_state = TikDebug.force_interactive
        TikDebug.force_interactive = False
        for stmt in self.stmts:
            stmt.evaluate(context)
        TikDebug.force_interactive = old_running_state


def replace_obj(ctx, obj, visited):
    """Recusive replace tik obj

    Parameters
    ----------
    ctx:Class context instance

    obj:Object

    visited:list
        A list of visited

    Returns
    ----------
    np_arr:NumpyBuffer

    ctx.evaluate_expr():A Expression

    replace_container:Type container
    """
    npbuf = ctx.get_value(obj)
    if npbuf is not None:
        # this is a tensor
        np_arr = npbuf.buffer
        return np_arr

    if isinstance(obj, BasicExpr):
        return ctx.evaluate_expr(obj)

    if isinstance(obj, (tuple, list)):
        return replace_container(ctx, obj, visited)

    return None


def eval_or_cache(ctx, obj, visited):
    """Eval or cache function

    Parameters
    ----------
    ctx:Class context instance

    obj:Object

    visited:list
        A list of visited

    Returns
    ----------
    visited:list
        A list of visited

    replaced:Returns of function replace_obj
    """
    key = id(obj)
    if key in visited:
        return visited[key]

    replaced = replace_obj(ctx, obj, visited)
    visited[key] = replaced
    return replaced


def replace_container(ctx, container, visited):
    """Replace container function

    Parameters
    ----------
    ctx:Class context instance

    container:(tuple, list)
        A container object

    visited:list
        A list of visited

    Returns
    ----------
    None
    type(container)
    """
    if isinstance(container, (list, tuple)):
        temp_l = []
        for contain in container:
            temp_l.append(eval_or_cache(ctx, contain, visited))
        return type(container)(temp_l)

    return None


class VarResolver(ast.NodeTransformer):
    """
    Class VarResolver inherits from ast.NodeVisitor for visit node info
    """
    def __init__(self, symtable, ctx):
        """Initialize class VarResolver

        Parameters
        ----------
        symtable:symbol table
            Evaluates the range of each symbol in the code

        ctx:Class context instance

        Returns
        ----------
        No returns
        """
        super(VarResolver, self).__init__()
        self.symtable = symtable
        self.ctx = ctx

        def _replace_fn(value):
            """define to replace function
            Parameters
            ----------
            value:func value

            Returns
            ----------
            np_buff:numpy buffer value
            value:func value
            """
            if isinstance(value, BasicExpr):
                return self.ctx.evaluate_expr(value)
            npbuf = self.ctx.get_value(value)
            if npbuf is not None:
                # this is a tensor
                return npbuf.buffer
            return value

        self.replace_fn = _replace_fn

    def visit_Name(self, node):
        """Visit name function

        Parameters
        ----------
        node:AST node

        Returns
        ----------
        node
        """
        # pylint: disable=C0103
        name = node.id
        ret = node
        obj = self.symtable.get(name)
        if obj is None:
            return ret

        npbuf = self.ctx.get_value(obj)
        if npbuf is not None:
            # this is a tensor
            np_arr = npbuf.buffer
            self.symtable[name] = np_arr
            return ret

        if isinstance(obj, BasicExpr):
            self.symtable[name] = self.ctx.evaluate_expr(obj)
            return ret

        return ret

    def visit_Attribute(self,  # pylint: disable=C0103
                        node):
        """
        transform a.b to _attr_sub(a, 'b') -> tensor or a.b

        Parameters
        ----------
        node:AST node

        Returns
        ----------
        ast node
        """

        def _attr_stub(value, attr):
            ret = getattr(value, attr)
            return self.replace_fn(ret)

        self.symtable['_attr_stub'] = _attr_stub

        value = node.value
        attr = node.attr
        fn_name = ast.Name(id='_attr_stub', ctx=ast.Load())
        fn_name = ast.copy_location(fn_name, node)
        arg_name = ast.Str(s=attr)
        arg_name = ast.copy_location(arg_name, node)

        return ast.copy_location(ast.Call(func=fn_name,
                                          args=[self.visit(value),
                                                arg_name],
                                          keywords=[],
                                          starargs=None,
                                          kwargs=None), node)

    def visit_Subscript(self,  # pylint: disable=C0103
                        node):
        """
        transform a[b] to _subscript_sub(a, b) -> tensor or a[b]
        slice = Ellipsis | Slice(expr? lower, expr? upper, expr? step) |
            ExtSlice(slice* dims) | Index(expr value)
            slice is not a expr!

        Parameters
        ----------
        node:AST node

        Returns
        ----------
        ast node
        """

        def _subscript_index_stub(value, index):
            """subscript index stub"""
            ret = value[index]
            return self.replace_fn(ret)

        def _subscript_ellipsis_stub(value):
            """subscript ellipsis stub"""
            ret = value[...]
            return self.replace_fn(ret)

        def _subscript_slice_stub(value, slice_obj):
            """subscript slice stub"""
            ret = value[slice_obj]
            return self.replace_fn(ret)

        def _subscript_extslice_stub(value, ext_slice_list):
            """subscript extslice stub"""
            ret = value.__getitem__(ext_slice_list)
            return self.replace_fn(ret)

        def _visit_if_not_none(expr):
            """if expr not none then visit"""
            if expr is not None:
                return self.visit(expr)
            return ast.copy_location(ast.Name(id='None', ctx=ast.Load()), node)

        self.symtable['_subscript_index_stub'] = _subscript_index_stub
        self.symtable['_subscript_ellipsis_stub'] = _subscript_ellipsis_stub
        self.symtable['_subscript_slice_stub'] = _subscript_slice_stub
        self.symtable['_subscript_extslice_stub'] = _subscript_extslice_stub

        value = node.value
        slice_expr = node.slice

        if isinstance(slice_expr, ast.Index):
            fn_name = ast.Name(id='_subscript_index_stub', ctx=ast.Load())
            fn_name = ast.copy_location(fn_name, node)

            return ast.copy_location(
                ast.Call(func=fn_name,
                         args=[self.visit(value),
                               self.visit(slice_expr.value)],
                         keywords=[], starargs=None, kwargs=None), node)
        if isinstance(slice_expr, ast.Ellipsis):
            fn_name = ast.Name(id='_subscript_ellipsis_stub', ctx=ast.Load())
            fn_name = ast.copy_location(fn_name, node)

            return ast.copy_location(ast.Call(func=fn_name,
                                              args=[self.visit(value)],
                                              keywords=[],
                                              starargs=None,
                                              kwargs=None), node)
        if isinstance(slice_expr, ast.Slice):
            fn_name = ast.Name(id='_subscript_slice_stub', ctx=ast.Load())
            fn_name = ast.copy_location(fn_name, node)

            create_fn = ast.Call(
                func=ast.copy_location(ast.Name(
                    id='slice', ctx=ast.Load()), node),
                args=[_visit_if_not_none(slice_expr.lower),
                      _visit_if_not_none(slice_expr.upper),
                      _visit_if_not_none(slice_expr.step)],
                keywords=[], starargs=None, kwargs=None)

            create_fn = ast.copy_location(create_fn, node)

            return ast.copy_location(ast.Call(func=fn_name,
                                              args=[self.visit(value),
                                                    create_fn],
                                              keywords=[],
                                              starargs=None,
                                              kwargs=None), node)

        if isinstance(slice_expr, ast.ExtSlice):
            fn_name = ast.Name(id='_subscript_extslice_stub', ctx=ast.Load())
            fn_name = ast.copy_location(fn_name, node)

            slice_list = []

            for dim in slice_expr.dims:
                if isinstance(dim, ast.Ellipsis):
                    slice_list.append(ast.copy_location(
                        ast.Name(id='Ellipsis', ctx=ast.Load()), node))
                elif isinstance(dim, ast.Slice):
                    create_fn = ast.Call(
                        func=ast.copy_location(ast.Name(
                            id='slice', ctx=ast.Load()), node),
                        args=[_visit_if_not_none(dim.lower),
                              _visit_if_not_none(dim.upper),
                              _visit_if_not_none(dim.step)],
                        keywords=[], starargs=None, kwargs=None)

                    create_fn = ast.copy_location(create_fn, node)
                    slice_list.append(create_fn)
                elif isinstance(dim, ast.Index):
                    slice_list.append(self.visit(dim.value))

            list_expr = ast.Tuple(elts=slice_list, ctx=ast.Load())
            list_expr = ast.copy_location(list_expr, node)

            return ast.copy_location(ast.Call(func=fn_name,
                                              args=[self.visit(value),
                                                    list_expr],
                                              keywords=[],
                                              starargs=None,
                                              kwargs=None), node)

    def visit_Call(self, node):  # pylint: disable=C0103
        """
        transform a(b) to _call_sub(a(b)) -> tensor or a(b)

        Parameters
        ----------
        node:AST node

        Returns
        ----------
        node
        """

        def _call_stub(ret):
            return self.replace_fn(ret)

        _call_stub_name = '_call_stub'

        self.symtable[_call_stub_name] = _call_stub

        fn_name = ast.Name(id=_call_stub_name, ctx=ast.Load())
        fn_name = ast.copy_location(fn_name, node)

        return ast.copy_location(ast.Call(func=fn_name,
                                          args=[self.generic_visit(node)],
                                          keywords=[],
                                          starargs=None,
                                          kwargs=None), node)


class PrintExpr(STMT):
    """
    Class PrintExpr inherits from STMT
    Print expression.
    """
    def __init__(self, source_info, expr):
        """Initialize class PrintExpr

        Parameters
        ----------
        source_info:source code information
            It represents the relationship of current node with source code

        expr:expression
            A part of stmt.

        Returns
        ----------
        No returns
        """
        super(PrintExpr, self).__init__(source_info)
        self.expr = expr

    def eval_(self, context):
        """Eval function
        evaluate all of self.function

        Parameters
        ----------
        context:information of debugger
            store all of debugger's information

        Returns
        ----------
        No returns
        """
        symtable = copy.copy(self.source_info[CUR_FRAME_IDX].get("sym_table"))
        self.print_expr(self.expr, symtable, context)

    @staticmethod
    def print_expr(expr, sym_table, context):
        """Print expression function

        Parameters
        ----------
        expr:expression
            A part of stmt.

        sym_table:symbol table
            Evaluates the range of each symbol in the code

        context:Class Context.

        Returns
        ----------
        No returns
        """
        # pylint: disable=W0123
        # waiting for security modification
        try:
            sym_table['SPR'] = context.spr_proxy
            expr_ast = ast.parse(expr, mode='eval')
            resolver = VarResolver(sym_table, context)
            expr_ast = resolver.visit(expr_ast)
            expr_result = eval(compile(expr_ast, '<string>', 'eval'),
                               globals(), resolver.symtable)
            print(str(expr_result))
        except (SyntaxError, TypeError, NameError, ValueError,
                RuntimeError, IndexError, ZeroDivisionError,
                AttributeError, EOFError, FloatingPointError,
                IOError, ImportError, KeyError, PermissionError,
                OverflowError, SystemError) as exc:
            print('*** {}'.format(repr(exc)))


class Print(STMT):
    """
    Class Print inherits from Print
    """
    def __init__(self, tensor, reshape, source_info):
        """Initialize class Print

        Parameters
        ----------
        tensor: a type of Tensor

        reshape:reshape the buffer
        source_info:source code information
            It represents the relationship of current node with source code

        Returns
        ----------
        No returns
        """
        super(Print, self).__init__(source_info)
        self.tensor = tensor
        self.reshape = reshape

    def eval_(self, context):
        """Eval function
        evaluate all of self.function

        Parameters
        ----------
        context:information of debugger
            store all of debugger's information

        Returns
        ----------
        None
        """
        # disable print statement in interactive mode, because we can inspect
        # a tensor precisely
        if TikDebug.tik_debug:
            return

        self.dump_tensor(context, self.tensor, self.reshape, self.source_info)

    @staticmethod
    def get_tensor(context, np_buf, raw_indice, reshape=None):
        """Get Tensor

        Parameters
        ----------
        context:context:information of debugger
            store all of debugger's information

        np_buf:NumpyBuffer

        raw_indice: index of raw
        reshape:None

        Returns
        ----------
        buf:NumpyBuffer
        """
        value = np_buf
        min_slice_extent = 1

        t_indice = Print.get_t_indice(context, raw_indice)
        buf = value.buffer
        if len(t_indice) != len(value.buffer.shape):
            buf = buf.reshape(-1)
            lenght = len(buf)
            # first slice equal to t_indice start element
            first_slice = t_indice[0]
            start_ = first_slice.start
            end_ = lenght
            if first_slice.stop - first_slice.start > min_slice_extent:
                end_ = first_slice.stop
            t_indice = tuple([slice(start_, end_)])

        buf = buf.__getitem__(t_indice)
        if reshape is not None:
            print(t_indice)
            try:
                buf = buf.reshape(reshape)
            except ValueError as exc:
                print(exc)

        return buf

    @staticmethod
    def get_t_indice(context, raw_indice):
        """Get Tensor

        Parameters
        ----------
        context:context:information of debugger
            store all of debugger's information

        raw_indice: index of raw

        Returns
        ----------
        t_indice
        """
        indice = []
        for rid in raw_indice:
            if isinstance(rid, slice):
                ri_start = context.evaluate_expr(rid.start)
                ri_step = context.evaluate_expr(rid.step)
                ri_stop = context.evaluate_expr(rid.stop)
                rid = slice(ri_start, ri_stop, ri_step)
            else:
                rid = context.evaluate_expr(rid)
            indice.append(rid)

        t_indice = tuple(indice)

        return t_indice

    @staticmethod
    def dump_tensor(context, tensor, reshape=None, source_info=None):
        """ To dump the given tensor

        Parameters
        ---------
        context:context:information of debugger
            store all of debugger's information

        tensor: a type of Tensor

        reshape:None

        source_info:None

        Returns
        ----------
        No returns
        """
        np_buf = context.get_value(tensor)
        raw_indice = tensor.indice.indice
        buf = Print.get_tensor(context, np_buf, raw_indice, reshape)
        if source_info:
            print(source_info)

        print(tensor.name + '.data (id:{}):\n'.format(id(buf)) + str(buf))
        print(tensor.name + '.shape:' + str(buf.shape) + ' dtype=' +
              str(buf.dtype))


class ForLoop(STMT):
    """
    Class ForLoop inherits from STMT
    """
    def __init__(self, begin, end, bind_var, source_info):
        """Initialize class ForLoop

        Parameters
        ----------
        begin:the begin of expression

        end:the end of exppression

        bind_var:bind variable

        source_info:source code information
            It represents the relationship of current node with source code

        Returns
        ----------
        No returns
        """
        super(ForLoop, self).__init__(source_info)
        # this node is hidden to the user
        self.traceable = False
        self.begin = begin
        self.end = end
        self.i = self.begin
        self.bind_var = bind_var
        self.trace_event = 'forloop'

        self.block = Block(source_info)

    def set_visible(self, visible):
        """Set visible for a block

        Parameters
        ----------
        visible:True or False
            A flag for a block

        Returns
        ----------
        No returns
        """
        self.block.traceable = visible

    def add_stmt(self, stmt):
        """Add stmt in a block

        Parameters
        ----------
        stmt:A list of STMT instance

        Returns
        ----------
        None
        """
        if self.frozen:
            return

        # ATTENTION!!!!!!!!!!!
        # we can't capture the name of var x in `for_range as x` stmt.
        # so we have to merge for range scope with its child scope
        self.block.source_info[CUR_FRAME_IDX].get("sym_table").update(
            stmt.source_info[CUR_FRAME_IDX].get("sym_table"))
        self.block.add_stmt(stmt)

    def eval_(self, context):
        """Eval function
        evaluate all of self.function

        Parameters
        ----------
        context:information of debugger
            store all of debugger's information

        Returns
        ----------
        No returns
        """
        begin_ = context.evaluate_expr(self.begin)
        end_ = context.evaluate_expr(self.end)
        for self.i in range(begin_, end_):
            context.update_var(self.bind_var, self.i)
            self.block.evaluate(context)


class IfScope(STMT):
    """
    Class IfScope inherits from STMT
    Represent IF code
    """
    def __init__(self, cond, source_info):
        """Initialize class IfScope

        Parameters
        ----------
        cond:judgement conditions of IF code
        source_info:source code information
            It represents the relationship of current node with source code

        Returns
        ----------
        No returns
        """
        super(IfScope, self).__init__(source_info)
        self.cond = cond
        self.then_block = Block(source_info)
        self.else_block = None
        self.trace_event = 'ifscope'

        from ..tik_lib.tik_expr import Expr
        if isinstance(self.cond, Expr):
            self.cond = self.cond.get()

    @property
    def block(self):
        """return to next block

        Returns
        ----------
        then_block:next block
        """
        return self.then_block

    def eval_(self, context):
        """Eval function
        evaluate all of self.function

        Parameters
        ----------
        context:information of debugger
            store all of debugger's information

        Returns
        ----------
        No returns
        """
        cond_val = context.evaluate_expr(self.cond)
        cond_val = bool(cond_val)
        if cond_val:
            if self.then_block.stmts:
                self.then_block.evaluate(context)
        else:
            if self.else_block is not None:
                self.else_block.evaluate(context)

    def add_stmt(self, stmt):
        """Add stmt for next block

        Parameters
        ----------
        stmt:a list of STMT instance

        Returns
        ----------
        None
        """
        if self.frozen:
            return

        self.then_block.add_stmt(stmt)

    def add_else_block(self, block):
        """Add else block

        Parameters
        ----------
        block:Block instance
            other Block

        Returns
        ----------
        No returns
        """
        self.else_block = block


class ElseScope(STMT):
    """
    The eval_ function of ElseScope is a fake one.
    The ElseScope attach its block to the last IfScope \
    in its scope and evaled by IfScope
    """
    def __init__(self, source_info):
        """Initialize class ElseScope

        Parameters
        ----------
        source_info:source code information
            It represents the relationship of current node with source code

        Returns
        ----------
        No returns
        """
        super(ElseScope, self).__init__(source_info)
        self.traceable = False
        self.block = Block(source_info)
        self.trace_event = 'elsescope'

    def add_stmt(self, stmt):
        """Add stmt for a block

        Parameters
        ----------
        stmt:a list of STMT instance

        Returns
        ----------
        None
        """
        if self.frozen:
            return

        self.block.add_stmt(stmt)

    def eval_(self, context):
        """Eval function
            Not implement"""


class DebugReturnException(Exception):
    """
    Class return debug exception
    """


class Return(STMT):
    """
    Class Return inherits from STMT
    """
    def eval_(self, context):
        """
        Eval function
        """
        raise DebugReturnException()


# scalar := scalar by value (implict conversion happens)
# scalar := tensor by ref (binary reinterpretation)
class SetScalar(STMT):
    """
    Class SetScalar inherits from STMT.
    To set scalar
    """
    def __init__(self, source_info, scalar, value):
        """Initialize class SetScalar

        Parameters
        ----------
        source_info:source code information
            It represents the relationship of current node with source code

        scalar:a type of scalar

        value:the value of scalar

        Returns
        ----------
        No returns
        """
        super(SetScalar, self).__init__(source_info)
        self.scalar = scalar
        self.dtype = self.scalar.dtype
        self.value = value
        self.i = None

    def eval_(self, context):
        """Eval function
        evaluate all of self.function

        Parameters
        ----------
        context:information of debugger
            store all of debugger's information

        Returns
        ----------
        No returns
        """
        if is_tensor(self.value):
            tensor = self.value
            src_bitwidth = int(get_dtype_bit_width(tensor.dtype))
            dst_bitwidth = int(get_dtype_bit_width(self.dtype))
            tensor_buffer = context.get_value(tensor)
            flatten_np = tensor_buffer.buffer.reshape(-1)
            flatten_idx = get_flatten_idx(tensor.indice, context)
            if flatten_idx >= len(flatten_np):
                TikCheckUtil.raise_error(
                    'IndexError: index {} out of range [0, {})'.format(
                        flatten_idx, len(flatten_np)))

            if dst_bitwidth >= src_bitwidth:
                ratio = dst_bitwidth // src_bitwidth
                r_value = flatten_np[flatten_idx:flatten_idx + ratio].view(
                    self.dtype)[0]
            else:
                r_value = np.asarray([flatten_np[flatten_idx]])\
                    .view(self.dtype)[0]
            self.i = make_tvm_imm(self.dtype, r_value)
        else:
            if isinstance(self.value, BasicExpr):
                self.i = context.evaluate_expr(self.value)
            else:
                self.i = make_tvm_imm(self.dtype, self.value)

        context.update_var(self.scalar, self.i)


# tensor := scalar by ref
# tensor := tensor by ref
class SetTensor(STMT):
    """
    Class SetTensor inherits from STMT
    To set Tensor
    """
    def __init__(self, source_info, tensor, value):
        """Initialize class SetTensor

        Parameters
        ----------
        source_info:source code information
            It represents the relationship of current node with source code

        tensor:a type of tensor

        value:the value of tensor

        Returns
        ----------
        No returns
        """
        super(SetTensor, self).__init__(source_info)
        self.tensor = tensor
        self.dtype = tensor.dtype
        self.value = value

    def eval_(self, context):
        """Eval function
        evaluate all of self.function

        Parameters
        ----------
        context:information of debugger
            store all of debugger's information

        Returns
        ----------
        No returns
        """
        dst_tensor = self.tensor
        dst_tensor_buffer = context.get_value(dst_tensor)
        dst_flatten_np = dst_tensor_buffer.buffer.reshape(-1)
        dst_flatten_idx = get_flatten_idx(dst_tensor.indice, context)
        if dst_flatten_idx >= len(dst_flatten_np):
            TikCheckUtil.raise_error(
                'IndexError: index {} out of range [0, {})'.format(
                    dst_flatten_idx, len(dst_flatten_np)))

        if is_tensor(self.value):
            dst_flatten_np = self.eval_value_tensor(context, dst_flatten_np,
                                                    dst_flatten_idx)
        else:
            if isinstance(self.value, BasicExpr):
                dst_flatten_np = self.eval_value_scalar_expr(context,
                                                             dst_flatten_np,
                                                             dst_flatten_idx)
            else:
                dst_flatten_np[dst_flatten_idx] = self.value

    def eval_value_tensor(self, context, dst_flatten_np, dst_flatten_idx):
        """Eval function
        evaluate self.value when it's tensor

        Parameters
        ----------
        context:information of debugger
            store all of debugger's information

        dst_flatten_np

        dst_flatten_idx

        Returns
        ----------
        dst_flatten_np
        """
        src_bitwidth = int(get_dtype_bit_width(self.value.dtype))
        dst_bitwidth = int(get_dtype_bit_width(self.dtype))
        src_tensor = self.value
        src_tensor_buffer = context.get_value(src_tensor)
        src_flatten_np = src_tensor_buffer.buffer.reshape(-1)
        src_flatten_idx = get_flatten_idx(src_tensor.indice, context)
        if src_flatten_idx >= len(src_flatten_np):
            TikCheckUtil.raise_error(
                'IndexError: index {} out of range [0, {})'.format(
                    dst_flatten_idx, len(dst_flatten_np)))
        py_value = src_flatten_np[src_flatten_idx]
        np_value = getattr(np, self.value.dtype)(py_value)

        if dst_bitwidth == src_bitwidth:
            dst_flatten_np[dst_flatten_idx] = py_value
        elif dst_bitwidth < src_bitwidth:
            ratio = src_bitwidth // dst_bitwidth
            dst_flatten_np[dst_flatten_idx:dst_flatten_idx +
                           ratio] = np_value
        else:
            np_tmp_arr = np.asarray([dst_flatten_np[dst_flatten_idx]]) \
                .view(self.dtype)
            np_tmp_arr[0] = np_value
            dst_flatten_np[dst_flatten_idx] = np_tmp_arr.view(
                self.value.dtype)[0]

        return dst_flatten_np

    def eval_value_scalar_expr(self, context, dst_flatten_np, dst_flatten_idx):
        """Eval function
        evaluate self.value when it's scalar or expr

        Parameters
        ----------
        context:information of debugger
            store all of debugger's information

        dst_flatten_np

        dst_flatten_idx

        Returns
        ----------
        dst_flatten_np
        """
        src_bitwidth = int(get_dtype_bit_width(self.value.dtype))
        dst_bitwidth = int(get_dtype_bit_width(str(dst_flatten_np.dtype)))
        py_value = context.evaluate_expr(self.value)

        if dst_bitwidth == src_bitwidth:
            dst_flatten_np[dst_flatten_idx] = py_value
        elif dst_bitwidth < src_bitwidth:
            np_value = getattr(np, self.value.dtype)([py_value])
            ratio = src_bitwidth // dst_bitwidth
            dst_flatten_np[dst_flatten_idx:dst_flatten_idx +
                           ratio] = np_value.view(str(dst_flatten_np.dtype))
        else:
            np_value = getattr(np, self.value.dtype)(py_value)
            np_tmp_arr = np.asarray([dst_flatten_np[dst_flatten_idx]]) \
                .view(self.value.dtype)
            np_tmp_arr[0] = np_value
            dst_flatten_np[dst_flatten_idx] = np_tmp_arr.view(
                dst_flatten_np.dtype)[0]
        return dst_flatten_np


class MoveCMPMASK2Tensor(STMT):
    """
    Class MoveCMPMASK2Tensor inherits STMT
    Move cmpmask to tensor
    """
    def __init__(self, source_info, dst):
        """Initialize class MoveCMPMASK2Tensor

        Parameter
        ----------
        source_info:source code information
            It represents the relationship of current node with source code

        dst:tensor
            destination of tensor

        src_cmpmask:source cmpmask

        Returns
        ----------
        No returns
        """
        super(MoveCMPMASK2Tensor, self).__init__(source_info)
        self.dst = dst

    def eval_(self, context):
        """Eval function
        evaluate all of self.function

        Parameters
        ----------
        context:information of debugger
            store all of debugger's information

        Returns
        ----------
        No returns
        """
        mask_low = context.model.read_spr('CMPMASK0')
        mask_high = context.model.read_spr('CMPMASK1')

        tensor_buffer = context.get_value(self.dst)
        flatten_np = tensor_buffer.buffer.reshape(-1)
        flatten_idx = get_flatten_idx(self.dst.indice, context)

        mask_arr = np.array([mask_low, mask_high], dtype='uint64')

        dst_bitwidth = int(get_dtype_bit_width(str(flatten_np.dtype)))

        extent = 2*UINT64_BIT // dst_bitwidth
        if flatten_idx + extent > len(flatten_np):
            TikCheckUtil.raise_error(
                'AccessViolation: write dst tensor out of range')

        flatten_np[flatten_idx: flatten_idx + extent] = \
            mask_arr.view(str(flatten_np.dtype))


class MoveTensor2CMPMASK(STMT):
    """
    Class MoveTensor2CMPMASK inherits from STMT.
    Move tensor to cmpmask
    """
    def __init__(self, source_info, src):
        """Initialize class MoveTensor2CMPMASK

        Parameters
        ----------
        source_info:source code information
            It represents the relationship of current node with source code

        src:source tensor

        Returns
        ----------
        No returns
        """
        super(MoveTensor2CMPMASK, self).__init__(source_info)
        self.src = src

    def eval_(self, context):
        """Eval function
        evaluate all of self.function

        Parameters
        ----------
        context:information of debugger
            store all of debugger's information

        Returns
        ----------
        No returns
        """
        tensor_buffer = context.get_value(self.src)

        flatten_np = tensor_buffer.buffer.reshape(-1)
        flatten_idx = get_flatten_idx(self.src.indice, context)

        TikCheckUtil.check_type_match(
            flatten_np.dtype, np.dtype,
            "Tensor buffer in debug flow should be numpy.ndarray")

        # each CMPMASK is 8-byte long, we need to fullfill two CMPMASK
        cmp_mask_count = 2
        cmp_bit_len = 8
        extent = cmp_mask_count*cmp_bit_len // flatten_np.dtype.itemsize
        TikCheckUtil.check_le(
            flatten_idx + extent, flatten_np.size,
            "AccessViolation: read src tensor out of range")

        mask_arr = flatten_np[flatten_idx:flatten_idx +
                              extent].view(dtype=np.uint64)

        TikCheckUtil.check_equality(
            len(mask_arr), cmp_mask_count, "Bitwidth not match!")

        mask_l = mask_arr[0]
        mask_h = mask_arr[1]
        context.model.write_spr('CMPMASK0', mask_l)
        context.model.write_spr('CMPMASK1', mask_h)


class TensorDef(STMT):
    """
        reset tensor value
    """
    def __init__(self, source_info, tensor):
        """Initialize class TensorDef

        Parameters
        ----------
        source_info:source code information
            It represents the relationship of current node with source code

        tensor:source tensor

        Returns
        ----------
        No returns
        """
        super(TensorDef, self).__init__(source_info)
        self.tensor = tensor

    def eval_(self, context):
        """Eval function
        evaluate all of self.function

        Parameters
        ----------
        context:information of debugger
            store all of debugger's information

        Returns
        ----------
        No returns
        """
        scope_name = TikUtil.get_storage_scope(self.tensor.scope)
        # OUT Tensor is inited by feed_dict
        if scope_name == 'OUT':
            return

        from .npbuffer import get_uninited_buffer

        tensor_buffer = context.get_value(self.tensor)
        tensor_buffer.buffer[:] = get_uninited_buffer(
            tensor_buffer.shape, tensor_buffer.dtype,
            context.init_mode, context.init_value)
