"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_ir_builder.py
DESC:     Developer API of IR node builder make function for TIK.
CREATED:  2019-08-12 18:53:42
MODIFIED: 2019-08-12 21:29:18
"""
from __future__ import absolute_import as _abs  # pylint: disable=C0302

from te import tvm
from te.tvm import api as _api
from te.tvm import stmt as _stmt
from te.tvm import expr as _expr
from te.tvm import make as _make
from te.tvm import ir_pass as _pass
from te.tvm import container as _container
from te.tvm.expr import Call as _Call
from te.tvm.api import thread_axis
from te.tvm import call_extern as _call_extern
from te.tvm.ir_builder import WithScope, IRBuilder
from te.tik import debug
from te.tik.tik_lib.tik_expr import Expr
from te.tvm.api import convert
from te.platform.cce_params import scope_reg, CCE_AXIS
from te.tvm import string_types

from ..common.util import is_basic_expr
from ..tik_lib.tik_params import ATTR_VALUE, FOR_TYPE_ZERO, \
    DEFAULT_VALUE_INDEX, LEN_SHAPE_ONE, MAX_IR_STATEMENT_NUM, TWO_IR, ONE_IR, \
    THREE_IR
from ..tik_lib.tik_buffervar import TikBufferVar
from ..tik_lib.tik_source_info import TikSourceInfo, get_loc,\
    source_info_decorator
from ..tik_lib.tik_check_util import TikCheckUtil, TIK_CONTROL
from ..tik_lib.tik_ir_builder_lib import CodeBufferManager,\
    CodeScalarManager, TikWithScope


_LAST_ELEMENT = -1

# block num min should greater than 0, less than 65536
_MIN_BLOCK_NUM = 1
_MAX_BLOCK_NUM = 65536

_DEFAULT_THREAD_BLOCK_NUM = 1
_MIN_NIDX = 3
_DEVICE_API = 0

# for type id value
_SERIAL_FOR_TYPE_ID = 0


def _check_params_for_range(thread_num, block_num, thread_type, dtype,
                            for_type):
    """check dtype, range of params

    Parameters
    ----------
    thread_num: int, how many thread to run for loop
    block_num: int, maximum is device aicore num
    thread_type: str, whole or partial
    dtype: str, The data type of iteration variable.
    for_type: str, The special tag on the for loop.

    Returns
    -------
    None
    """
    TikCheckUtil.check_type_match(
        thread_num, int, "thread_num(%s) should be int" % str(thread_num))
    TikCheckUtil.check_in_range(
        thread_num, range(1, 3),
        "thread_num(%d) can only be 1 or 2" % thread_num)
    TikCheckUtil.check_type_match(
        block_num, int, "block_num(%s) should be int" % str(block_num))
    TikCheckUtil.check_in_range(
        block_num, range(_MIN_BLOCK_NUM, _MAX_BLOCK_NUM),
        "block_num(%d) must be in [%d, %d)" % (block_num, _MAX_BLOCK_NUM,
                                               _MAX_BLOCK_NUM))
    TikCheckUtil.check_type_match(
        thread_type, str,
        "thread_type(%s) should be str" % str(type(thread_type)))
    TikCheckUtil.check_type_match(
        dtype, str, "dtype(%s) should be str" % str(type(dtype)))
    TikCheckUtil.check_type_match(for_type, str, "for_type should be str")


def _check_mul_aic_for_range(block_num, endt, begint):
    """if block_num > 1, enable multi-AIcore,
       endt and begint must be satisfy the conditions

    Parameters
    ----------
    block_num: int, maximum is device aicore num
    endt: Expr, The end iteration scope.
    begint: Expr, The min iteration scope.

    Returns
    -------
    None
    """
    if block_num > _DEFAULT_THREAD_BLOCK_NUM:
        TikCheckUtil.check_type_match(
            endt, int, "endt(%s) should be int" % str(type(endt)))
        TikCheckUtil.check_equality(
            endt, block_num,
            "endt(%d) should be equal to block_num(%d)" % (endt, block_num))
        TikCheckUtil.check_type_match(
            begint, int, "begint(%s) should be int" % str(type(begint)))
        TikCheckUtil.check_equality(
            begint, 0, "begint(%d) should be equal to 0" % begint)


class TikIRBuilder(IRBuilder):
    """
    Auxiliary builder to build IR for testing and dev.

    Inherit from IRBuilder of ir_builder

    Examples
    --------
    .. code-block:: python

        ib = tvm.ir_builder.create()
        n = tvm.var("n")
        A = ib.allocate("float32", n, name="A")
        with ib.for_range(0, n, name="i") as i:
            with ib.if_scope((i % 2) == 0):
                A[i] = A[i] + 1
        # The result stmt.
        stmt = ib.get()
    """
    # pylint: disable=R0902
    # @cond
    def __init__(self):
        super(TikIRBuilder, self).__init__()
        self.source_info = TikSourceInfo()
        self.source_info.register_source_info()
        self.if_scope_level = 0
        self.for_scope_level = 0
        self.else_scope_level = 0
        self.stmt_scope_level = 0
        # use this to judge whether here is double buffer scope
        self.is_double_buffer_for_loop_list = []
        self.is_double_buffer_for_loop = False
        self.code_buffer_manager = CodeBufferManager()
        self.code_scalar_manager = CodeScalarManager()

        self.total_ir_lines = 0
        self.double_buffer_ir_num = 0
        self.source_info.clear_source_info()
    # @endcond

    def _get_loop_var_name(self, name):
        if self.nidx < _MIN_NIDX:
            name = chr(ord(name) + self.nidx)
        else:
            name = "%s_%d" % (name, (self.nidx - _MIN_NIDX))
        self.nidx += 1
        return name

    def _get_dthread_for_loop_var(self, name, dtype, thread_num):
        name_dthread = "%s_thread_%d" % (name, self.nidx)
        dthread_var = _api.var(name_dthread, dtype=dtype)

        name_for = "%s_for_%d" % (name, self.nidx)
        for_var = _api.var(name_for, dtype=dtype)

        loop_var = for_var * thread_num + dthread_var
        return dthread_var, for_var, loop_var

    @staticmethod
    def _get_max_if_condition(extent, thread_num):
        if isinstance(extent, int):
            if extent % thread_num != 0:
                return extent
            return None
        return extent

    @staticmethod
    def _get_attr_key(thread_type):
        """
        get attr key by thread_type
        :param thread_type: thread type
        :return: key
        """
        if thread_type == "whole":
            attr_key = "dthread_whole"
        elif thread_type == "partial":
            attr_key = "dthread_partition"
        else:
            TikCheckUtil.raise_error("thread_type(%s) only support 'partial' "
                                     "and 'whole' mode" % thread_type)
        return attr_key

    @staticmethod
    def _get_block_id(block_num):
        """
        get block id
        :param block_num: block number
        :return: block id
        """
        block_id = None
        if block_num > _DEFAULT_THREAD_BLOCK_NUM:
            block_id = thread_axis("blockIdx.x")
        return block_id

    @staticmethod
    def _get_loop_var(block_id, name, dtype):
        """
        get variable in loop
        :param block_id:
        :param name: variable name
        :param dtype: data dtype
        :return: var
        """
        if block_id is not None:
            return block_id.var
        return _api.var(name, dtype=dtype)

    @staticmethod
    def _get_begin_offset_and_extent(begint, endt):
        """
        get the value of extent and offset when they are scalar
        :param begint: scalar begin
               endt: scalar end
        :return: begin, offset, extent
        """
        # pylint: disable=E1101
        # pylint cannot recognize C++ member so disable it
        if is_basic_expr([begint]):
            begin = begint.get()
            offset = begint.get()
        else:
            begin = Expr(begint).get()
            offset = begint

        # get the end value
        end = endt
        if is_basic_expr([endt]):
            end = endt.get()

        if begin == 0:
            extent = end
        else:
            extent = _pass.Simplify(end - begin)
            begin = 0  # set begin index to 0

        if isinstance(extent, (_expr.IntImm, _expr.UIntImm)):
            extent = extent.value
        return begin, offset, extent

    @staticmethod
    def _check_thread_block_num(thread_num, block_num):
        """
        check thread number and block number
        :param thread_num: thread number
        :param block_num: block number
        :return: None
        """
        if _DEFAULT_THREAD_BLOCK_NUM not in (thread_num, block_num):
            TikCheckUtil.raise_error(
                "thread_num(%d) and block_num(%d) must be at "
                "least one equal to 1" % (thread_num, block_num))

    @staticmethod
    def _check_extent_num(extent, thread_num):
        """
        check extent number
        :param extent: extent value
        :param thread_num: thread number
        :return: None
        """
        extent_value = Expr(extent).eval_value()
        if extent_value is not None:
            TikCheckUtil.check_ge(
                extent_value, 0,
                "for_range function parameter end(%d) "
                "must be more than or equal to begin(0)" % extent_value)
            if extent_value > 0:
                TikCheckUtil.check_ge(
                    extent_value, thread_num,
                    "for_range end-begin(%d) should be more than thread_num(%d)"
                    % (extent_value, thread_num))

    @staticmethod
    def _get_for_type_id(for_type):
        # currently only support serial
        if for_type == "serial":
            return _SERIAL_FOR_TYPE_ID
        return TikCheckUtil.raise_error("Unknown for_type(%s)" % for_type,
                                        exception_type=ValueError)

    def _update_code_manager_info(self, thread_num, enter_or_exit):
        """
        update code_buffer_manager and code_scalar_manager info
        :param thread_num: the input num of thread
        :param enter_or_exit: enter for_range or exit for_range
        :return: None
        """
        if enter_or_exit == 1:
            if thread_num > 1:
                self.code_buffer_manager.double_buffer_enable_ += 1
            if thread_num > 2:  # set double buffer thread num to input value
                self.code_buffer_manager.double_buffer_thread_num = thread_num
            else:  # set double buffer thread num to 2
                self.code_buffer_manager.double_buffer_thread_num = 2
            self.code_buffer_manager.new_scope()
            self.code_scalar_manager.new_scope()
        elif enter_or_exit == 0:
            self.code_buffer_manager.del_scope()
            self.code_scalar_manager.del_scope()
            if thread_num > 1:
                self.code_buffer_manager.double_buffer_enable_ -= 1

    def _create_stmt_for_double_buffer(self, dthread_var, max_if_condition,
                                       loop_var, thread_type, thread_num):
        """
        create the body stmt for double buffer
        :param dthread_var: the inner loop var for double buffer
        :param max_if_condition: the max loop time of for
        :param loop_var: the expr of dthread_var and for_var
        :param thread_type: the thread type
        :param thread_num: input num of thread
        :return: the stmt of the double buffer
        """
        # pylint: disable=R0913, E1101
        # pylint cannot recognize C++ member so disable E1101
        TikCheckUtil.check_not_equality(dthread_var, 0, "dthread_var is 0")
        stmt = self._pop_seq()
        begin_min_value = 0
        extra_ir_num = 0
        if max_if_condition is not None:
            stmt = _make.IfThenElse(loop_var < max_if_condition, stmt, None)
            self.source_info.set_node_loc(stmt)
            # if then else include 2 ir
            extra_ir_num += TWO_IR
        attr_key = self._get_attr_key(thread_type)
        tmp_for = _make.For(dthread_var, begin_min_value,
                            thread_num, FOR_TYPE_ZERO,
                            _DEVICE_API, stmt)
        self.source_info.set_node_loc(tmp_for)
        stmt = _make.AttrStmt(CCE_AXIS, attr_key, ATTR_VALUE, tmp_for)
        self.source_info.set_node_loc(stmt)
        # for include 2 ir, attrstmt include 1ir
        extra_ir_num += THREE_IR
        return stmt, extra_ir_num

    @source_info_decorator()
    def for_range(self, begint, endt, name="i", thread_num=1,
                  thread_type="whole", block_num=1,
                  dtype="int32", for_type="serial"):
        """
        Indicates the for loop statement of the TIK.

        Description:
          Indicates the for loop statement of the TIK. The double-buffer and
          multi-core running functions can be enabled
          in the for loop.
        Args:
          begint : Start of the for loop.
          begint and endt are immediates of type int or uint, scalars
          of type int or uint, or Exprs.
          If Exprs are passed, the simplified values must be
          integers.0 <= begint <= endt <= 2147483647
          endt : End of the for loop.
          begint and endt are immediates of type int or uint, scalars
          of type int or uint, or Exprs.
          If Exprs are passed, the simplified values must be
          integers.0 <= begint <= endt <= 2147483647
            - NOTE:
            The performance deteriorates when begint and endt are scalars.
        Kwargs:
          name : Name of a variable in the for loop. The default name is i.
          thread_num : Whether to enable double buffers in the
          for loop. Value range:
            - 1: The double-buffer function is disabled.
            - 2: The double-buffer function is enabled.
          thread_type : Thread type of the for loop. This parameter is reserved
           and has no impact on system running.Value: whole
          block_num : Number of cores used in the for loop. The maximum value
          is 65535.
            - If the number of cores configured is greater than the number of
            available cores, Runtime will perform
             scheduling in batches.
            - If the number of cores configured is less than or equal to the
            number of available cores, Runtime will
             perform scheduling as required. The number of running cores may
             be less than or equal to the number of
             cores configured.
          dtype : Variable type of the for loop. This parameter is reserved
          and has no impact on system running.Value: int32
          for_type : Type of the for loop. Value: serial.
        Returns:
          TikWithScope object
        Restrictions:
          - If the multi-core function is enabled, the start value of the
          multi-core loop must be 0, and the number of
          cores must be the same as the end value of the multi-core loop.
          - In a for loop, the multi-core and double-buffer functions cannot be
           both enabled. To enable them both, use
          multiple loops.
          - If the multi-core function is enabled, the tensor used in the
          multi-core loop must also be defined in the
          multi-core loop. The tensor buffer allocation in the inner and outer
          sides of the multi-core loop both starts
          from 0. As a result, the addresses may overlap and data errors
          may occur.
          - When the double-buffer function is enabled, two buffers are
          allocated only when the tensor is defined in
          the for loop.
        Example:
            @code
            with self.tik_instance.for_range(0,1,thread_num=1):
                do_someting
            # Enable the double-buffer function. Note that two buffers are
            # allocated only when the tensor is defined in for range.
            with self.tik_instance.for_range(0,2,thread_num=2):
                Tensor definition
                do_someting
            # Enable multi-core running.
            with self.tik_instance.for_range(0,2,block_num=2):
                do_someting
            @endcode
        """
        # pylint: disable=R0913, R0914, W0221, E1101
        # disable R0914, beacuse arugments are too many, and function is close
        # disable W0221, because for_range is to extend the parent class's
        #                for_range
        return self.for_range_(begint, endt, name=name, thread_num=thread_num,
                               thread_type=thread_type, block_num=block_num,
                               dtype=dtype, for_type=for_type)

    def _gen_for_range_exit_cb_when_thread_one(self,  # pylint: disable=R0913
                                               block_num, block_id, endt,
                                               loop_var, begin, loop_times,
                                               for_type_id):
        # disable E1101, because pylint can't find symbol from back-end so
        if block_num > _DEFAULT_THREAD_BLOCK_NUM:
            TikCheckUtil.check_not_is(
                block_id, None, "block_id is None")
            # thread_extent contains 1 ir
            # one ir is thread_extent
            tmp_attr = _make.AttrStmt(block_id,  # pylint: disable=E1101
                                      "thread_extent", endt, self._pop_seq())
            self.source_info.set_node_loc(tmp_attr)
            self.emit(tmp_attr, ONE_IR)
        else:
            # for type contains 2 ir
            tmp_for = _make.For(loop_var, begin,  # pylint: disable=E1101
                                loop_times, for_type_id, _DEVICE_API,
                                self._pop_seq())
            self.source_info.set_node_loc(tmp_for)
            self.emit(tmp_for, TWO_IR)

    # @cond
    @debug.for_range_decorator
    def for_range_(self, begint, endt, name="i",  # pylint: disable=R0913, R0914
                   thread_num=1, thread_type="whole", block_num=1,
                   dtype="int32", for_type="serial"):
        """Create a for iteration scope.
        note: use this function to call for_range inside!!
        """
        # disable R0914, beacuse arugments are too many, and function is close
        # disable E1101, because pylint can't find symbol from back-end so
        _check_params_for_range(thread_num, block_num, thread_type, dtype,
                                for_type)
        self._check_thread_block_num(thread_num, block_num)
        self._seq_stack.append([])
        endt = self._init_for_range_params(endt)

        # set begin to 0, extent to (endt-begint), offset to begint
        begin, offset, extent = self._get_begin_offset_and_extent(begint, endt)
        self._check_extent_num(extent, thread_num)
        # calculate the for loop times
        loop_times = (extent + thread_num - 1) // thread_num

        if name == 'i':
            name = self._get_loop_var_name(name)

        # if block_num > 1, enable multi-AIcore,
        # endt and begint must be satisfy the conditions
        _check_mul_aic_for_range(block_num, endt, begint)
        if thread_num > _DEFAULT_THREAD_BLOCK_NUM:
            dthread_var, for_var, loop_var = self._get_dthread_for_loop_var(
                name, dtype, thread_num)
            debug_hint = [(for_var, (begin, loop_times)),
                          (dthread_var, (0, thread_num))]
            # for thread num is 2,
            # set self.in_double_buffer_for_loop append True
            self.is_double_buffer_for_loop_list.append(True)
        else:  # no double buffer and no multi-core
            block_id = self._get_block_id(block_num)
            loop_var = self._get_loop_var(block_id, name, dtype)
            debug_hint = [(loop_var, (begin, loop_times))]
            self.is_double_buffer_for_loop_list.append(False)
        # check if True in double buffer loop
        self._check_if_true_in_db_loop()
        # set the max_if_condition
        max_if_condition = self._get_max_if_condition(extent, thread_num)
        ret_loop_var = loop_var
        # if begint is not 0, the loop var need to add offset
        if Expr(begint).eval_value() != 0:
            ret_loop_var = loop_var + offset
        self._update_code_manager_info(thread_num, 1)

        def _exit_cb():
            self._update_code_manager_info(thread_num, 0)
            self.for_scope_level -= 1
            self.is_double_buffer_for_loop_list.pop(-1)
            self._check_if_true_in_db_loop()
            # check for_scope_level whether or not less than 0
            TikCheckUtil.check_ge(
                self.for_scope_level, 0, "for_scope_level(%d) should be more "
                                         "than 0" % self.for_scope_level)
            # check for_scope_level whether or not less than 0
            TikCheckUtil.check_ge(
                len(self.is_double_buffer_for_loop_list), 0,
                "in_double_buffer_for_loop length should be more than 0")
            for_type_id = self._get_for_type_id(for_type)
            if thread_num == _DEFAULT_THREAD_BLOCK_NUM:
                self._gen_for_range_exit_cb_when_thread_one(block_num, block_id,
                                                            endt, loop_var,
                                                            begin,
                                                            loop_times,
                                                            for_type_id)
            else:
                stmt, extra_ir_num = self._create_stmt_for_double_buffer(
                    dthread_var, max_if_condition, loop_var, thread_type,
                    thread_num)
                stmt = _make.For(for_var, begin,  # pylint: disable=E1101
                                 loop_times, for_type_id, _DEVICE_API, stmt)
                self.source_info.set_node_loc(stmt)
                # each for contains 2 ir
                self.emit(stmt, extra_ir_num + TWO_IR)
            if not self.is_double_buffer_for_loop_list and \
                    self.double_buffer_ir_num != 0:
                # will add double buffer body to total
                self.total_ir_lines += self.double_buffer_ir_num
                self.double_buffer_ir_num = 0
                self.is_double_buffer_for_loop = False

        with_scope = TikWithScope(ret_loop_var, _exit_cb,
                                  self.source_info.get_source_info())
        with_scope.debug_hint = debug_hint
        with_scope.debug_limit = (loop_var, extent)
        return with_scope

    def _check_if_true_in_db_loop(self):
        """check if True in double buffer loop
        """
        if True in self.is_double_buffer_for_loop_list:
            self.is_double_buffer_for_loop = True
        else:
            self.is_double_buffer_for_loop = False

    def _init_for_range_params(self, endt):
        """ initalize params in for_range
        """
        if self.for_scope_level == 0:
            # every time for for scope level is 0, set to False
            self.is_double_buffer_for_loop_list = []
            self.double_buffer_ir_num = 0
        # into for scope increase the for_scope_level
        self.for_scope_level += 1
        if isinstance(endt, float):
            # for python3
            endt = int(endt)
        return endt

    def _multi_for_range_exit_cb_(self, for_type,  # pylint: disable=R0913
                                  var_number, loop_var_list,
                                  begin, extent_list):
        """exit func when exit multi_for_range"""
        # disable E1101, because pylint can't find symbol from back-end so
        self.for_scope_level -= 1
        # check for_scope_level whether or not less than 0
        TikCheckUtil.check_ge(
            self.for_scope_level, 0, "for_scope_level(%d) should be more "
                                     "than 0" % self.for_scope_level)
        for_type_id = self._get_for_type_id(for_type)
        total_ir_num = 0
        stmt = self._pop_seq()
        for index in range(var_number - 1, -1, -1):
            stmt = _make.For(loop_var_list[index],  # pylint: disable=E1101
                             begin[index], extent_list[index], for_type_id,
                             _DEVICE_API, stmt)
            self.source_info.set_node_loc(stmt)
            # each for contains 2 ir
            total_ir_num += TWO_IR
        self.emit(stmt, total_ir_num)

    def _get_multi_for_range_params(self, var_number,  # pylint: disable=R0913
                                    name, dtype, begint, endt):
        """get serial param for multi_for_range"""
        # disable E1101, because pylint can't find symbol from back-end so
        name_list = []
        loop_var_list = []
        extent_list = []
        begin = []
        for i in range(var_number):
            name_list.append(self._get_loop_var_name(name))
            # last_name
            loop_var_list.append(_api.var(name_list[-1], dtype=dtype))
            if begint[i] == 0:
                extent_list.append(endt[i])
            else:
                extent_list.append(_pass.Simplify(endt[i] -  # pylint: disable=E1101
                                                  begint[i]))
            if is_basic_expr([begint[i]]):
                begin.append(begint[i].get())
            else:
                begin.append(begint[i])
        return loop_var_list, extent_list, begin
    # @endcond

    # @cond
    @source_info_decorator()
    def multi_for_range(self, begint, endt, name="i",  # pylint: disable=R0913
                        dtype="int32", for_type="serial"):
        """Create a for iteration scope.

        Parameters
        ----------
        begint : Expr
            The min iteration scope.

        endt : Expr
            The end iteration scope

        name : str, optional
            The name of iteration variable, if no input names,
            using typical index names i, j, k, then i_nidx

        dtype : str, optional
            The data type of iteration variable.

        for_type : str, optional
            The special tag on the for loop.

        Returns
        -------
        loop_scope : With.Scope of Var
            The for scope, when enters returns loop_var

        Examples
        --------
        .. code-block:: python

            ib = tvm.ir_builder.create()
            x = ib.pointer("float32")
            with ib.for_range(1, 10, name="i") as i:
                x[i] = x[i - 1] + 1
        """
        # pylint: disable=R0913, E1101
        # disable E1101, because pylint can't find symbol from back-end so
        self.for_scope_level += 1
        TikCheckUtil.check_type_match(
            dtype, str, "dtype(%s) should be str" % str(type(dtype)))
        TikCheckUtil.check_type_match(
            for_type, str, "for_type(%s) should be str" % str(type(for_type)))
        TikCheckUtil.check_type_match(
            begint, (list, tuple),
            "begint(%s) should be list or tuple" % str(type(begint)))
        TikCheckUtil.check_type_match(
            endt, (list, tuple),
            "endt(%s) should be list or tuple" % str(type(endt)))
        var_number = len(begint)
        TikCheckUtil.check_equality(
            var_number, len(endt),
            "length of begint(%d) should be equal to length of "
            "endt(%d)" % (var_number, len(endt)))
        loop_var_list, extent_list, begin = \
            self._get_multi_for_range_params(var_number, name, dtype,
                                             begint, endt)

        self._seq_stack.append([])

        _multi_for_range_loc = self.source_info.get_source_info()
        # multi_for_range do not have debug decorator,so must handle source info
        def _exit_cb():
            if TIK_CONTROL.is_user_call:
                self.source_info.register_source_info(
                    source_info=_multi_for_range_loc)
                self.source_info.set_not_user_call()
                self._multi_for_range_exit_cb_(for_type, var_number,
                                               loop_var_list,
                                               begin, extent_list)
                self.source_info.set_is_user_call()
                self.source_info.clear_source_info()
            else:
                self._multi_for_range_exit_cb_(for_type, var_number,
                                               loop_var_list,
                                               begin, extent_list)

        return TikWithScope(loop_var_list, _exit_cb,
                            self.source_info.get_source_info())
    # @endcond

    @source_info_decorator()
    def if_scope(self, cond):
        """
        Creates an if statement of the TIK.

        Description:
          Creates an if statement of the TIK. When the condition is met, the
          statement in the structure is executed.
        Args:
          cond : An Expr specifying the judgment condition.
            - NOTICE:
              - The float data type is not supported.
              - Expr supports greater-than sign (>), less-than sign (<),
              not-equal sign (!=), double equal sign (==)
              but does not support OR sign (||) and AND sign (&&).
        Returns:
          TikWithScope object
        Restrictions:
          None
        Example:
            @code
            with self.tik_instance.if_scope(core_index != core_num - 1):
                do_something()
            @endcode
        """
        return self.if_scope_(cond)

    # @cond
    @debug.if_scope_decorator
    def if_scope_(self, cond):
        """Create an if scope.
        note: use this function to call if_scope inside!!
        """
        self.if_scope_level += 1
        self._seq_stack.append([])
        self.code_buffer_manager.new_scope()
        self.code_scalar_manager.new_scope()

        _if_then_else_source_info = self.source_info.get_source_info()

        def _exit_cb():
            # pylint: disable=E1101
            # disable E1101, because pylint can't find symbol from back-end so
            self.if_scope_level -= 1
            self.code_buffer_manager.del_scope()
            self.code_scalar_manager.del_scope()
            # check else_scope_level whether or not less than 0
            TikCheckUtil.check_ge(
                self.if_scope_level, 0, "if_scope_level(%d) should be more "
                                        "than 0" % self.if_scope_level)
            seq = self._pop_seq()
            # change bool to int8
            bool_value = self.Scalar_("bool")
            with self.context.freeze():
                bool_value.set_as(cond)
            # if then include 2 ir
            tmp_if_then_else = _make.IfThenElse(bool_value.get(), seq, None)
            self.source_info.set_node_loc(tmp_if_then_else)
            self.emit(tmp_if_then_else, TWO_IR)

        return TikWithScope(None, _exit_cb, _if_then_else_source_info)
    # @endcond

    @source_info_decorator()
    def else_scope(self):
        """
        Creates an else statement of the TIK.

        Description:
          Creates an else statement of the TIK. If the if statement does not
          meet the conditions, the statement in the
          else_scope structure is executed.

        Args:
          None

        Returns:
          TikWithScope object
        Restrictions:
          This function must be after if_scope.
        Example:
            @code
            with self.tik_instance.if_scope(core_index != core_num - 1):
                do_something()
            with self.tik_instance.else_scope():
                do_else_something()
            @endcode
        """
        return self.else_scope_()

    # @cond
    @debug.else_scope_decorator
    def else_scope_(self):
        """Create an else scope.
        note: use this function to call else_scope inside!!
        """
        self.else_scope_level += 1
        if not self._seq_stack[_LAST_ELEMENT]:
            TikCheckUtil.raise_error("else_scope can only follow an if_scope")
        prev = self._seq_stack[_LAST_ELEMENT][_LAST_ELEMENT]
        if not isinstance(prev, _stmt.IfThenElse) or prev.else_case:
            TikCheckUtil.raise_error("else_scope can only follow an if_scope")
        self._seq_stack[_LAST_ELEMENT].pop()
        self._seq_stack.append([])
        self.code_buffer_manager.new_scope()
        self.code_scalar_manager.new_scope()

        def _exit_cb():
            # pylint: disable=E1101
            # disable E1101, because pylint can't find symbol from back-end so
            self.else_scope_level -= 1
            self.code_buffer_manager.del_scope()
            self.code_scalar_manager.del_scope()
            # check else_scope_level whether or not less than 0
            TikCheckUtil.check_ge(
                self.else_scope_level, 0, "else_scope_level(%d) should be more "
                                          "than 0" % self.else_scope_level)
            # else include 1 ir
            tmp_if_then_else = _make.IfThenElse(prev.condition, prev.then_case,
                                                self._pop_seq())
            self.source_info.set_node_loc(tmp_if_then_else)
            self.emit(tmp_if_then_else, ONE_IR)
        return TikWithScope(None, _exit_cb, self.source_info.get_source_info())
    # @endcond

    @source_info_decorator()
    def new_stmt_scope(self):
        """
        Indicates a new scope (C language).

        Description:
          Indicates a new scope (C language).
        Args:
          None
        Return:
          TikWithScope object.
        Restrictions:
          After a tensor is defined beyond new_stmt_scope, the buffer is
          automatically freed and the tensor defined in
          new_stmt_scope cannot be accessed externally.
        Example:
            @code
            with tik_instance.new_stmt_scope():
                do_something
            @endcode
        """
        self.stmt_scope_level += 1
        self._seq_stack.append([])
        self.code_buffer_manager.new_scope()
        self.code_scalar_manager.new_scope()

        _new_stmt_scope_source_info = self.source_info.get_source_info()
        def _exit_cb_():
            # pylint: disable=E1101
            # disable E1101, because pylint can't find symbol from back-end so
            self.stmt_scope_level -= 1
            # check stmt_scope_level whether or not less than 0
            TikCheckUtil.check_ge(
                self.stmt_scope_level, 0, "stmt_scope_level(%d) should be more "
                                          "than 0" % self.stmt_scope_level)
            # body scope include one ir
            tmp_attr = _make.AttrStmt(CCE_AXIS, "new_stmt_scope",
                                      ATTR_VALUE, self._pop_seq())
            self.source_info.set_node_loc(tmp_attr)
            self.emit(tmp_attr, ONE_IR)
            self.code_buffer_manager.del_scope()
            self.code_scalar_manager.del_scope()

        # new_stmt_scope do not have debug decorator, so must handle source info
        def _exit_cb():
            if TIK_CONTROL.is_user_call:
                self.source_info.register_source_info(
                    source_info=_new_stmt_scope_source_info)
                self.source_info.set_not_user_call()
                _exit_cb_()
                self.source_info.set_is_user_call()
                self.source_info.clear_source_info()
            else:
                _exit_cb_()
        return TikWithScope(None, _exit_cb, self.source_info.get_source_info())

    # @cond
    def buffer_print(self):
        """
        print buffer information

        Parameters
        ----------
        parameters:No parameters
        -------
        return:No return
        """
        self.code_buffer_manager.buffer_print()

    def allocate(self, dtype, shape, name="buf",  # pylint: disable=R0913
                 scope=None, init_value=None,
                 buffer_reuse_id=None):
        """Create a allocate statement.

        Parameters
        ----------
        dtype : str
            The content data type.

        shape : tuple of Expr
            The shape of array to be allocated.

        name : str, optional
            The name of the buffer.

        scope : str, optional
            The scope of the buffer.

        init_value: tuple pr list, optional
            The init value for the buffer

        buffer_reuse_id : tensor index for reuse/no_reuse

        Returns
        -------
        buffer : BufferVar
            The buffer var representing the buffer.
        """
        # pylint: disable=R0913, E1101, W0221
        # disable E1101, because pylint can't find symbol from back-end so
        # disable W0221, because for_range is to extend the parent class's
        #                allocate

        # check dtype is str
        TikCheckUtil.check_type_match(
            dtype, str, "dtype(%s) should be str" % str(type(dtype)))
        buffer_var = _api.var(name, dtype="handle")
        if not isinstance(shape, (list, tuple, _container.Array)):
            shape = [shape]
        if scope:
            self.scope_attr(buffer_var, "storage_scope", scope)

        _allocate_func_loc = get_loc()
        if init_value is None:

            def _allocate_func_init_none(pre_stmt):
                tmp_allocate = _make.Allocate(
                    buffer_var, dtype, shape, _api.const(1, dtype="bool"),
                    pre_stmt)
                self.source_info.set_node_loc(tmp_allocate,
                                              loc=_allocate_func_loc)
                return tmp_allocate

            self.emit(_allocate_func_init_none, ONE_IR)
        else:
            if not isinstance(init_value, (tuple, list)):
                init_value = [init_value]
            TikCheckUtil.check_equality(
                scope, scope_reg, "init_value only support for register")
            TikCheckUtil.check_equality(
                len(shape), LEN_SHAPE_ONE,
                "length of shape(%d) should be 1" % len(shape))
            TikCheckUtil.check_equality(
                len(init_value), shape[0],
                "length of init_value(%d) should be equal to size of "
                "shape(%d)" % (len(init_value), shape[0]))

            def _allocate_func_init_not_none(pre_stmt):
                tmp_const = _api.const(1, dtype="bool")
                tmp_call = _make.Call("uint64", "params", convert(init_value),
                                      _Call.Extern, None,
                                      DEFAULT_VALUE_INDEX)
                tmp_allocate = _make.Allocate(
                    buffer_var, dtype, shape, tmp_const, pre_stmt, tmp_call)
                self.source_info.set_node_loc(tmp_allocate,
                                              loc=_allocate_func_loc)
                return tmp_allocate

            self.emit(_allocate_func_init_not_none, ONE_IR)
        if buffer_reuse_id is not None:
            self.scope_attr(buffer_var, "pragma_buffer_index",
                            _call_extern("int64", "buffer_index",
                                         buffer_reuse_id))
        return TikBufferVar(self, buffer_var, dtype)

    def is_tensor_in_scope(self):
        """Check scope if has tensor
        ----------
        Returns
        True or False
        -------
        """
        if ((self.if_scope_level > 0) or
                (self.for_scope_level > 0) or
                (self.else_scope_level > 0) or
                (self.stmt_scope_level > 0)):
            return True
        return False

    @source_info_decorator()
    def scope_attr(self, node, attr_key, value):
        """Create an AttrStmt at current scope.

        Parameters
        ----------
        attr_key : str
            The key of the attribute type.

        node : Node
            The attribute node to annottate on.

        value : Expr
            Attribute value.
        --------
        """
        # disable E1101, because pylint can't find symbol from back-end so
        if isinstance(node, string_types):
            node = _make.StringImm(node)  # pylint: disable=E1101
        if isinstance(value, string_types):
            value = _make.StringImm(value)  # pylint: disable=E1101

        _attr_stmt_loc = get_loc()

        def _attr_stmt(pre_stmt):
            tmp_attr = _make.AttrStmt(node, attr_key,  # pylint: disable=E1101
                                      value, pre_stmt)
            self.source_info.set_node_loc(tmp_attr, loc=_attr_stmt_loc)
            return tmp_attr

        self.emit(_attr_stmt)

        self.total_ir_lines += ONE_IR
        if self.is_double_buffer_for_loop:
            self.double_buffer_ir_num += ONE_IR

    @source_info_decorator()
    def new_scope(self):
        self._seq_stack.append([])
        _new_scope_source_info = self.source_info.get_source_info()

        # new_scope do not have debug decorator, so must handle source info
        def _exit_cb():
            if TIK_CONTROL.is_user_call:
                self.source_info.register_source_info(
                    source_info=_new_scope_source_info)
                self.source_info.set_not_user_call()
                self.emit(self._pop_seq())
                self.source_info.set_is_user_call()
                self.source_info.clear_source_info()
            else:
                self.emit(self._pop_seq())
        return WithScope(None, _exit_cb)

    def emit(self, stmt, add_ir_num=0):  # pylint: disable=W0221
        """Emit a statement to the end of current scope.

        Parameters
        ----------
        stmt : Stmt or callable.
           The statement to be emitted or callable that build stmt given body.

        add_ir_num: int.
           Add to total ir num
        """
        # extend function of emit, keep consistency of interface, so disable it
        if isinstance(stmt, _expr.Call):
            stmt = _make.Evaluate(stmt)  # pylint: disable=E1101
            self.source_info.set_node_loc(stmt)
        if not isinstance(stmt, _stmt.Stmt) and not callable(stmt):
            TikCheckUtil.raise_error("stmt should be type of Stmt or callable")
        self._seq_stack[-1].append(stmt)

        self.total_ir_lines += add_ir_num
        if self.is_double_buffer_for_loop:
            self.double_buffer_ir_num += add_ir_num
        TikCheckUtil.check_le(
            self.total_ir_lines, MAX_IR_STATEMENT_NUM,
            "Total IR num " + str(self.total_ir_lines)
            + " is already more than " +
            str(MAX_IR_STATEMENT_NUM) + "!")

    def _pop_seq(self):
        """Pop sequence from stack"""
        # pylint: disable=E1101
        # pylint cannot recognize C++ member so disable it
        seq = self._seq_stack.pop()
        if not seq or callable(seq[-1]):
            self.total_ir_lines += ONE_IR
            if self.is_double_buffer_for_loop:
                self.double_buffer_ir_num += ONE_IR
            tmp_node = _make.Evaluate(0)
            self.source_info.set_node_loc(tmp_node)
            seq.append(tmp_node)
        stmt = seq[-1]
        for statement in reversed(seq[:-1]):
            if callable(statement):
                stmt = statement(stmt)
            else:
                assert isinstance(statement, _stmt.Stmt)
                stmt = _make.Block(statement, stmt)
                self.source_info.set_node_loc(stmt)
        return stmt
    # @endcond

    @source_info_decorator()
    @debug.tik_return_decorator
    def tik_return(self):
        """
        Returns at an instruction or layer position as required ...

        Description:
          Returns at an instruction or layer position as required, which is
          often used to set a breakpoint.
        Args:
          None
        Return:
          None
        Restrictions:
          None
        Example:
            @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            tik_instance.tik_return()
            @endcode
        """
        self.emit(tvm.call_extern(
            "uint64",
            "return",
        ), ONE_IR)
