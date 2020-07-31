"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     decorators.py
DESC:     this file contains many decorator
CREATED:  2019-7-04 20:12:13
MODIFIED: 2019-7-24 14:04:45
"""
# disabling:
# R0903: too-few-public-methods
# R0913: too-many-arguments(wrapper function, so disable them)
# R0914: too-many-locals(caused by arguments, so disable them)
# C0302: too-many-lines
# pylint: disable=C0302

import sys
from functools import wraps

from .intrinsic import Load2D, Load3DV1, DataMove, BroadcastUB, MMAD, \
    DataMoveDeQuant, WriteSPR, Col2Img, TensorMove, ScalarSingleOp, \
    ScalarConv, VMS4SR2Scalar, ScalarBinaryOp, SetL0SetValue, Set2D, \
    SetCtrlSPR, GetCtrlSPR, Load3DV2, DepthwiseConv, LoadSmask, LoadImage, \
    LoadL1ToL0BWinograd, LoadL1ToL0AWinograd, MmadBrc
from .simd import VectorScalarTemplate, Vconv, ScatterVconv, VCMPV, \
    VTranspose, VectorScalarEltwise, VecReduce, VSEL, ScatterVsel, \
    VCMP, ScatterSingleVector, ScatterVectorBinary, ScatterVmulconv, \
    ScatterVCMP, ScatterVectorScalar, Vnchwconv, ListListEltwise, \
    VectorVectorTemplate, VectorOnlyTemplate, VecCMPScalar, V4DTRANS, \
    VReduce, VPadding, VScatter, VGather, VnchwTrans, VReduceAdd, \
    VecAllReduce, VBI, VrsqrtHighPreci, VrecHighPreci, VlnHighPreci, \
    Vexpm1HighPreci
from .object_detect import VMS4, VEXTRACT, VCONCAT, RpnCor, RpnCorDiag, VIOU, \
    VRPAC, VAADD, VBS16, VMergeCH, SetRpnOffset
from .statement import SetTensor, SetScalar, ForLoop, IfScope, ElseScope, \
    Return, MoveCMPMASK2Tensor, MoveTensor2CMPMASK, TensorDef
from . statement import HighLevelAPIScope
from .tikdbg.codemapping import get_caller_context
from ..tik_lib.tik_check_util import TikCheckUtil, TIK_CONTROL
from ..common.common_util import is_tensor
from ..common.common_util import is_scalar

_FRAME_INFO_IDX = 3

_SCALAR_SET_AS_STACK_DEPTH = 3
_SCALAR_INIT_STACK_DEPTH = 4


def high_level_api_debug_decorator(func):
    """bind this decorator with build_cce

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # the first arg must be tik_instance
        tik_instance = args[0]
        ctx = tik_instance.context
        library_call = HighLevelAPIScope(get_caller_context())
        ctx.add_scope(library_call)
        return_value = func(*args, **kwargs)
        ctx.pop_scope()
        return return_value

    return wrapper


def build_cce_decorator(func):
    """bind this decorator with build_cce

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(kernel_name, tik_instance, inputs,  # pylint: disable=R0913
                outputs, global_scalar_list, workspace_tensor_list, config,
                flowtable_tmp):
        """bind this decorator with build_cce"""
        if not tik_instance.debug_disabled:
            ctx = tik_instance.context
            ctx.record_ins([i.buffer for i in inputs if hasattr(i, "buffer")])
            for i in inputs:
                if not hasattr(i, "buffer"):
                    ctx.bind_feed_var(i.name, i.get())
            ctx.record_outs([i.buffer for i in outputs
                             if hasattr(i, "buffer")])
            for i in flowtable_tmp:
                ctx.bind_feed_var(i.name, i.get())
        return func(kernel_name, tik_instance, inputs, outputs,
                    global_scalar_list, workspace_tensor_list, config,
                    flowtable_tmp)

    return wrapper


def tensor_register(cls):
    """bind this decorator with tensor

    Parameters
    ----------
    cls : the decorated class

    Returns
    -------
    class
    """
    original__init__ = cls.__init__

    @wraps(original__init__)
    def __init__(self, *args, **kwargs):
        original__init__(self, *args, **kwargs)
        if not self.ir_generator.debug_disabled:
            ctx = self.ir_generator.context
            if 'indice' not in kwargs:
                # a tensor
                tensor_init_stmt = TensorDef(
                    get_caller_context(depth=_FRAME_INFO_IDX), self)
                ctx.curr_scope().add_stmt(tensor_init_stmt)
                ctx.tensor_list.append(self)
            else:
                # a tensor proxy
                ctx.add_proxy(self, kwargs['buffer_'],
                              kwargs['indice'], kwargs['dtype'])

    cls.__init__ = __init__

    return cls


def scalar_register(cls):
    """bind this decorator with sclar

    Parameters
    ----------
    cls : the decorated class

    Returns
    -------
    class
    """
    original__init__ = cls.__init__

    @wraps(original__init__)
    def __init__(self, ir_generator, dtype="int64", # pylint: disable=R0913
                 name="reg_buf", init_value=None, if_global_scope=False):
        # too many arguments
        with ir_generator.context.freeze():
            original__init__(self, ir_generator, dtype, name,
                             init_value, if_global_scope)
        if not ir_generator.debug_disabled:
            self.ir_generator.context.set_scalar_location(
                self, get_caller_context(depth=_FRAME_INFO_IDX))
            if init_value is not None:
                scalar_set_as_fn(self, init_value, _SCALAR_INIT_STACK_DEPTH)
    cls.__init__ = __init__

    return cls


def tensor_set_as_decorator(func):
    """bind this decorator with set_as

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(self, value, dst_offset=0, src_offset=None):
        """bind this decorator with set_as"""
        min_value_size = 1
        if is_tensor(value):
            if value.size > min_value_size:
                return func(self, value, dst_offset, src_offset)
        if not self.ir_generator.debug_disabled:
            ctx = self.ir_generator.context
            stmt = SetTensor(get_caller_context(depth=2), self, value)
            stmt.traceable = False
            ctx.curr_scope().add_stmt(stmt)
        return func(self, value, dst_offset, src_offset)

    return wrapper


def scalar_set_as_fn(scalar, value, depth=_SCALAR_SET_AS_STACK_DEPTH):
    """
    set scalar's debug information
    :param scalar: a variable in scalar
    :param value: the value to be set
    :param depth: stack depth
    :return:
    """
    ctx = scalar.ir_generator.context
    stmt = SetScalar(get_caller_context(depth=depth), scalar.debug_var, value)
    stmt.traceable = False
    ctx.curr_scope().add_stmt(stmt)
    ctx.bind_var(scalar.debug_var)
    ctx.set_scalar_var_mapping(scalar.reg_buffer, scalar.debug_var)


def scalar_set_as_decorator(func):
    """bind this decorator with scalar set_as

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(scalar, value, src_offset=None):
        """bind this decorator with scalar set_as"""
        if not scalar.ir_generator.debug_disabled:
            scalar_set_as_fn(scalar, value)
        return func(scalar, value, src_offset)

    return wrapper


class WrapCtxMgr():
    """wrap context"""
    # pylint: disable=R0903
    def __init__(self, ctx, tik_instance):
        self.ctx = ctx
        self.tik = tik_instance
        self.scope_num = 1 if ctx.debug_hint is None else len(
            ctx.debug_hint) + 1

    def __enter__(self):
        return self.ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if TIK_CONTROL.is_user_call:
            self.tik.source_info.register_source_info(
                source_info=self.ctx.source_info)
            self.tik.source_info.set_not_user_call()
            if not self.tik.debug_disabled:
                for _ in range(self.scope_num):
                    self.tik.context.pop_scope()
            tmp_exit = self.ctx.__exit__(exc_type, exc_val, exc_tb)
            self.tik.source_info.set_is_user_call()
            self.tik.source_info.clear_source_info()
        else:
            if not self.tik.debug_disabled:
                for _ in range(self.scope_num):
                    self.tik.context.pop_scope()
            tmp_exit = self.ctx.__exit__(exc_type, exc_val, exc_tb)

        return tmp_exit


def for_range_decorator(func):
    """bind this decorator with for_range

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(self, begint, endt, **kwargs):
        """bind this decorator with for_range"""
        ctx_mgr = func(self, begint, endt, **kwargs)
        if not self.debug_disabled:
            ctx = self.context
            for hint in ctx_mgr.debug_hint:
                loop_var, loop_range = hint
                begint, endt = loop_range
                loop = ForLoop(begint, endt, loop_var,
                               get_caller_context(depth=2))
                loop.set_visible(False)
                ctx.add_scope(loop)
                ctx.bind_var(loop_var)
            # since a for loop may be implictly split
            # into multi loops due to muti-threading
            # only the inner most loop is visible to the user
            loop.set_visible(True)

            # insert a if stmt to limit the loop_var
            real_loop_var, real_loop_limit = ctx_mgr.debug_limit
            if_stmt = IfScope(real_loop_var < real_loop_limit,
                              get_caller_context(depth=2))
            if_stmt.traceable = False
            if_stmt.then_block.traceable = False
            ctx.add_scope(if_stmt)
        return WrapCtxMgr(ctx_mgr, self)

    return wrapper


def if_scope_decorator(func):
    """bind this decorator with if_scope

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(self, cond):
        """bind this decorator with if_scope"""
        ctx_mgr = func(self, cond)
        if not self.debug_disabled:
            self.context.add_scope(IfScope(cond, get_caller_context(depth=2)))

        return WrapCtxMgr(ctx_mgr, self)

    return wrapper


def else_scope_decorator(func):
    """bind this decorator with else_scope

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(self):
        """bind this decorator with else_scope"""
        ctx_mgr = func(self)
        if not self.debug_disabled:
            if not self.context.frozen:
                last_block_stmt = -1
                ctx = self.context
                curr_scope = ctx.curr_scope()
                curr_block = getattr(curr_scope, 'block', curr_scope)
                maybe_if_scope = curr_block.stmts[last_block_stmt]
                if not isinstance(maybe_if_scope, IfScope):
                    TikCheckUtil.raise_error(
                        'Input ElseScope not instance IfScope')
                else_scope = ElseScope(get_caller_context(depth=2))
                ctx.add_scope(else_scope)
                maybe_if_scope.add_else_block(else_scope.block)
        return WrapCtxMgr(ctx_mgr, self)

    return wrapper


def load2dv1_decorator(func):
    """bind this decorator with load2dv1_decorator

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, src, index,  # pylint: disable=R0913
                repeat_times, src_stride, sid, if_transpose=False,
                addr_mode=None):
        """bind this decorator with load2dv1_decorator"""
        if not tik_instance.debug_disabled:
            load2dv1_stmt = Load2D(get_caller_context(), dst, src, index,
                                   repeat_times, 0, src_stride, sid,
                                   if_transpose, addr_mode)
            tik_instance.context.curr_scope().add_stmt(load2dv1_stmt)
        return func(tik_instance, dst, src, index, repeat_times, src_stride,
                    sid, if_transpose, addr_mode)

    return wrapper


def load2dv2_decorator(func):
    """bind this decorator with load2dv2_decorator

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, src, index,  # pylint: disable=R0913
                repeat_times, dst_gap, src_stride, sid, if_transpose=False,
                addr_mode=None):
        """bind this decorator with load2dv1_decorator"""
        if not tik_instance.debug_disabled:
            load2dv2_stmt = Load2D(get_caller_context(), dst, src, index,
                                   repeat_times, dst_gap, src_stride, sid,
                                   if_transpose, addr_mode)
            tik_instance.context.curr_scope().add_stmt(load2dv2_stmt)
        return func(tik_instance, dst, src, index, repeat_times, dst_gap,
                    src_stride, sid, if_transpose, addr_mode)

    return wrapper


def load3dv1_decorator(func):
    """bind this decorator with load3dv1_decorator

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, src, pad,  # pylint: disable=R0913, R0914
                l1_h, l1_w, c1_index, fetch_w, fetch_h, left_top_w, left_top_h,
                stride_w, stride_h, filter_w, filter_h, dilation_filter_w,
                dilation_filter_h, jump_offset, repeat_mode, repeat_time,
                _csize=0, pad_value=0):
        """bind this decorator with load3dv1_decorator"""
        if not tik_instance.debug_disabled:
            load3dv1_stmt = Load3DV1(get_caller_context(), dst, src, pad, l1_h,
                                     l1_w, c1_index, fetch_w, fetch_h,
                                     left_top_w, left_top_h, stride_w,
                                     stride_h, filter_w, filter_h,
                                     dilation_filter_w, dilation_filter_h,
                                     jump_offset, repeat_mode, repeat_time,
                                     _csize, pad_value)

            tik_instance.context.curr_scope().add_stmt(load3dv1_stmt)
        return func(tik_instance, dst, src, pad, l1_h, l1_w, c1_index,
                    fetch_w, fetch_h, left_top_w, left_top_h, stride_w,
                    stride_h, filter_w, filter_h, dilation_filter_w,
                    dilation_filter_h, jump_offset, repeat_mode, repeat_time,
                    _csize, pad_value)

    return wrapper


def datamove_decorator(func):
    """bind this decorator with data_move

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, src, sid, nburst,  # pylint: disable=R0913
                burst, src_stride, dst_stride, *args, **argv):
        """bind this decorator with data_move"""
        if not tik_instance.debug_disabled:
            datamove_stmt = DataMove(get_caller_context(**argv), dst, src, sid,
                                     nburst, burst, src_stride, dst_stride,
                                     *args, **argv)
            tik_instance.context.curr_scope().add_stmt(datamove_stmt)
        return func(tik_instance, dst, src, sid, nburst, burst, src_stride,
                    dst_stride, *args, **argv)

    return wrapper


def vec_scalar_single_elewise_dec(func):
    """bind this decorator with vector_scalar_single_elewise

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, mask, dst,  # pylint: disable=R0913, R0914
                src, scalar, repeat_times,
                dst_blk_stride, src_blk_stride, dst_rep_stride,
                src_rep_stride, stride_unit, round_en,
                mask_mode="normal", print_name=None, mask_o=None):
        """bind this decorator with vector_scalar_single_elewise"""
        if not tik_instance.debug_disabled:
            vector_scalar_ = VectorScalarTemplate(
                get_caller_context(), name, mask, dst, src,
                scalar, repeat_times, dst_blk_stride, src_blk_stride,
                dst_rep_stride, src_rep_stride, stride_unit, round_en,
                mask_mode, print_name)
            tik_instance.context.curr_scope().add_stmt(vector_scalar_)
        return func(tik_instance, name, mask, dst, src, scalar, repeat_times,
                    dst_blk_stride, src_blk_stride, dst_rep_stride,
                    src_rep_stride, stride_unit, round_en,
                    mask_mode, print_name, mask_o)

    return wrapper


def vec_bin_elewise_func_dec(func):
    """bind this decorator with vector_binary_elewise_func

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, mask, dst,  # pylint: disable=R0913, R0914
                src0, src1, repeat_times, dst_blk_stride, src0_blk_stride,
                src1_blk_stride, dst_rep_stride, src0_rep_stride,
                src1_rep_stride, stride_unit, store_mode=None,
                print_name=None, mask_o=None):
        """bind this decorator with vector_binary_elewise_func"""
        if not tik_instance.debug_disabled:
            vector_vector_ = VectorVectorTemplate(
                get_caller_context(),
                name, mask, dst, src0, src1, repeat_times, dst_blk_stride,
                src0_blk_stride, src1_blk_stride, dst_rep_stride,
                src0_rep_stride, src1_rep_stride, stride_unit, print_name)
            tik_instance.context.curr_scope().add_stmt(vector_vector_)
        return func(tik_instance, name, mask, dst, src0, src1, repeat_times,
                    dst_blk_stride, src0_blk_stride, src1_blk_stride,
                    dst_rep_stride, src0_rep_stride, src1_rep_stride,
                    stride_unit, store_mode, print_name, mask_o)

    return wrapper


def vmulconv_decorator(func):
    """bind this decorator with vmulconv

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, mask, dst, src0, src1, # pylint: disable=R0913
                repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
                dst_rep_stride, src0_rep_stride, src1_rep_stride, stride_unit):
        """bind this decorator with vmulconv"""
        # this decorator function is not used
        # please register source info when used
        if not tik_instance.debug_disabled:
            vmulconv_ = VectorVectorTemplate(get_caller_context(), 'vmulconv',
                                             mask, dst, src0, src1,
                                             repeat_times, dst_blk_stride,
                                             src0_blk_stride, src1_blk_stride,
                                             dst_rep_stride, src0_rep_stride,
                                             src1_rep_stride, stride_unit)
            tik_instance.context.curr_scope().add_stmt(vmulconv_)

        return func(tik_instance, mask, dst, src0, src1, repeat_times, dst_blk_stride,
                    src0_blk_stride, src1_blk_stride, dst_rep_stride,
                    src0_rep_stride, src1_rep_stride, stride_unit)

    return wrapper


def vadddeqrelu_decorator(func):
    """bind this decorator with vaddeqrelu

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, mask, dst, deqscale, # pylint: disable=R0913
                src0, src1, repeat_times, dst_blk_stride, src0_blk_stride,
                src1_blk_stride, dst_rep_stride, src0_rep_stride,
                src1_rep_stride, stride_unit=0):
        """bind this decorator with vaddeqrelu"""
        if not tik_instance.debug_disabled:
            stmt = VectorVectorTemplate(get_caller_context(), 'vadddeqrelu',
                                        mask, dst, src0, src1, repeat_times,
                                        dst_blk_stride, src0_blk_stride,
                                        src1_blk_stride, dst_rep_stride,
                                        src0_rep_stride, src1_rep_stride,
                                        stride_unit, deqscale=deqscale)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, mask, dst, deqscale, src0, src1,
                    repeat_times, dst_blk_stride,
                    src0_blk_stride, src1_blk_stride, dst_rep_stride,
                    src0_rep_stride, src1_rep_stride, stride_unit)

    return wrapper


def vec_single_elewise_func_dec(func):
    """bind this decorator with vector_single_elewise_func

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, mask, dst, src,  # pylint: disable=R0913
                repeat_times, dst_blk_stride, src_blk_stride, dst_rep_stride,
                src_rep_stride, stride_unit, print_name=None, mask_o=None):
        """bind this decorator with vector_single_elewise_func"""
        if not tik_instance.debug_disabled:
            vector_only_ = VectorOnlyTemplate(
                get_caller_context(), name, mask, dst, src,
                repeat_times, dst_blk_stride, src_blk_stride, dst_rep_stride,
                src_rep_stride, stride_unit, print_name)
            tik_instance.context.curr_scope().add_stmt(vector_only_)
        return func(tik_instance, name, mask, dst, src, repeat_times,
                    dst_blk_stride, src_blk_stride, dst_rep_stride,
                    src_rep_stride, stride_unit, print_name, mask_o)

    return wrapper


def vrec_high_preci_decorator(func):
    """bind this decorator with vrec_high_preci

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, mask, dst,  # pylint: disable=R0913
                src, work_tensor, repeat_times,
                dst_rep_stride, src_rep_stride):
        """bind thid decorator with vrec_high_preci"""
        if not tik_instance.debug_disabled:
            vrec_high_preci_ = VrecHighPreci(get_caller_context(), mask,
                                             dst, src, work_tensor,
                                             repeat_times,
                                             dst_rep_stride, src_rep_stride)
            tik_instance.context.curr_scope().add_stmt(vrec_high_preci_)
        return func(tik_instance, mask, dst, src, work_tensor, repeat_times,
                    dst_rep_stride, src_rep_stride)

    return wrapper


def vrsqrt_high_preci_decorator(func):
    """bind this decorator with vrsqrt_high_preci

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, mask, dst,  # pylint: disable=R0913
                src, work_tensor, repeat_times,
                dst_rep_stride, src_rep_stride):
        """bind this decorator with vrsqrt_high_preci"""
        if not tik_instance.debug_disabled:
            vrsqrt_high_ = VrsqrtHighPreci(
                get_caller_context(), mask, dst, src, work_tensor,
                repeat_times, dst_rep_stride, src_rep_stride)
            tik_instance.context.curr_scope().add_stmt(vrsqrt_high_)
        return func(tik_instance, mask, dst, src, work_tensor, repeat_times,
                    dst_rep_stride, src_rep_stride)

    return wrapper


def vec_ln_high_preci_decorator(func):
    """bind this decorator with vec_ln_high_preci

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, mask, dst,  # pylint: disable=R0913
                src, work_tensor, repeat_times,
                dst_rep_stride, src_rep_stride):
        """bind this decorator with vec_ln_high_preci"""
        if not tik_instance.debug_disabled:
            vln_high_preci = VlnHighPreci(
                get_caller_context(), mask, dst, src, work_tensor,
                repeat_times, dst_rep_stride, src_rep_stride)
            tik_instance.context.curr_scope().add_stmt(vln_high_preci)
        return func(tik_instance, mask, dst, src, work_tensor, repeat_times,
                    dst_rep_stride, src_rep_stride)

    return wrapper


def vexpm1_high_preci_decorator(func):
    """bind this decorator with vexpm1_high_preci

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, mask,  # pylint: disable=R0913
                dst, src, work_tensor, repeat_times,
                dst_rep_stride, src_rep_stride):
        if not tik_instance.debug_disabled:
            vrec_high_preci_ = Vexpm1HighPreci(
                get_caller_context(), mask, dst, src,
                work_tensor, repeat_times,
                dst_rep_stride, src_rep_stride)
            tik_instance.context.curr_scope().add_stmt(vrec_high_preci_)
        return func(tik_instance, mask, dst, src, work_tensor, repeat_times,
                    dst_rep_stride, src_rep_stride)

    return wrapper


def vconv_decorator(func):
    """bind this decorator with vconv

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, mask, mode, dst, src, # pylint: disable=R0913
                repeat_times, dst_blk_stride, src_blk_stride, dst_rep_stride,
                src_rep_stride, deqscale=None, ldst_high_half=False,
                stride_unit=0, name="vconv", mask_o=None):
        """bind this decorator with vconv"""
        if not tik_instance.debug_disabled:
            vconv_ = Vconv(get_caller_context(), mask, mode, dst, src,
                           repeat_times, dst_blk_stride, src_blk_stride,
                           dst_rep_stride, src_rep_stride, deqscale,
                           ldst_high_half, stride_unit, name)
            tik_instance.context.curr_scope().add_stmt(vconv_)
        return func(tik_instance, mask, mode, dst, src, repeat_times,
                    dst_blk_stride, src_blk_stride, dst_rep_stride,
                    src_rep_stride, deqscale, ldst_high_half,
                    stride_unit, name, mask_o)

    return wrapper


def vec_conv_decorator(func):
    """bind this decorator with vconv

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, mask, mode, dst, src,  # pylint: disable=R0913
                repeat_times, dst_rep_stride, src_rep_stride,
                deqscale=None, ldst_high_half=False):
        """bind this decorator with vconv"""
        if not tik_instance.debug_disabled:
            blk_stride = 1
            stride_unit = 0
            vconv_ = Vconv(get_caller_context(), mask, mode, dst, src,
                           repeat_times, blk_stride, blk_stride,
                           dst_rep_stride, src_rep_stride, deqscale,
                           ldst_high_half, stride_unit, print_name="vec_conv")
            tik_instance.context.curr_scope().add_stmt(vconv_)
        return func(tik_instance, mask, mode, dst, src, repeat_times,
                    dst_rep_stride, src_rep_stride, deqscale, ldst_high_half)

    return wrapper


def scatter_vconv_decorator(func):
    """bind this decorator with scatter_vconv

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, mask, round_mode, # pylint: disable=R0913
                dst_list, src_list, repeat_times, dst_rep_stride,
                src_rep_stride, deqscale=None, ldst_high_half=False):
        """bind this decorator with scatter_vconv"""
        if not tik_instance.debug_disabled:
            scatter_vconv_ = ScatterVconv(get_caller_context(), mask,
                                          round_mode, dst_list, src_list,
                                          repeat_times, dst_rep_stride,
                                          src_rep_stride, deqscale,
                                          ldst_high_half)
            tik_instance.context.curr_scope().add_stmt(scatter_vconv_)
        return func(tik_instance, mask, round_mode, dst_list, src_list,
                    repeat_times, dst_rep_stride, src_rep_stride,
                    deqscale, ldst_high_half)

    return wrapper


def broadcast_ub_to_l0c_decorator(func):
    """bind this decorator with broadcast_ub_to_10c

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, src, *strides):
        """bind this decorator with broadcast_ub_to_10c"""
        if not tik_instance.debug_disabled:
            brc_stmt_ = BroadcastUB(get_caller_context(), dst, src, *strides)
            tik_instance.context.curr_scope().add_stmt(brc_stmt_)
        return func(tik_instance, dst, src, *strides)

    return wrapper


def mmad_decorator(func):
    """bind this decorator with mmad

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper( # pylint: disable=R0913, R0914
            tik_instance, dst_fm, src_fm, src_filter, matrix_m, matrix_k,
            matrix_n, is_bias, fm_offset=0,
            en_weight_offset=False, smask=None, en_small_channel=False,
            en_small_k=False, en_ssparse=False, en_winograd_a=False,
            en_winograd_b=False):
        """bind this decorator with mmad"""

        if not tik_instance.debug_disabled:
            mmad_stmt = MMAD(
                get_caller_context(), dst_fm, src_fm, src_filter, matrix_m,
                matrix_k, matrix_n, is_bias, fm_offset,
                en_weight_offset, smask, en_small_channel, en_small_k,
                en_ssparse, en_winograd_a, en_winograd_b)

            tik_instance.context.curr_scope().add_stmt(mmad_stmt)
        return func(tik_instance, dst_fm, src_fm, src_filter, matrix_m,
                    matrix_k, matrix_n, is_bias, fm_offset,
                    en_weight_offset, smask, en_small_channel, en_small_k,
                    en_ssparse, en_winograd_a, en_winograd_b)

    return wrapper


def dma_dquant_decorator(func):
    """bind this decorator with dma_dquant

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, src, sid, nburst, # pylint: disable=R0913
                burst, src_stride, dst_stride, quant_param, relu_flag=False):
        """bind this decorator with dma_dquant"""
        if not tik_instance.debug_disabled:
            dma_deq_stmt = DataMoveDeQuant(get_caller_context(), dst, src, sid,
                                           nburst, burst, src_stride,
                                           dst_stride, quant_param, relu_flag)
            tik_instance.context.curr_scope().add_stmt(dma_deq_stmt)
        return func(tik_instance, dst, src, sid, nburst, burst,
                    src_stride, dst_stride, quant_param, relu_flag)

    return wrapper


def vcmpv_decorator(func):
    """bind this decorator with vcmpv

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, dst, src0, # pylint: disable=R0913
                src1, repeat_times, src0_blk_stride,
                src1_blk_stride, src0_rep_stride, src1_rep_stride,
                print_name=None):
        """bind this decorator with vcmpv"""
        if not tik_instance.debug_disabled:
            vcmpv_stmt = VCMPV(get_caller_context(), name,
                               dst, src0, src1,
                               repeat_times, src0_blk_stride, src1_blk_stride,
                               src0_rep_stride, src1_rep_stride, print_name)
            tik_instance.context.curr_scope().add_stmt(vcmpv_stmt)
        return func(tik_instance, name, dst, src0, src1,
                    repeat_times, src0_blk_stride, src1_blk_stride,
                    src0_rep_stride, src1_rep_stride, print_name)

    return wrapper


def vtranspose_decorator(func):
    """bind this decorator with vtranspose

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, src):
        """bind this decorator with vtranspose"""
        if not tik_instance.debug_disabled:
            vtrans_stmt = VTranspose(get_caller_context(), dst, src)
            tik_instance.context.curr_scope().add_stmt(vtrans_stmt)
        return func(tik_instance, dst, src)

    return wrapper


def vec_trans_decorator(func):
    """bind this decorator with vec_trans

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, src, repeat_times,  # pylint: disable=R0913
                dst_repeat_stride, src_repeat_stride):
        """bind this decorator with VnchwTrans"""
        if not tik_instance.debug_disabled:
            vtrans_stmt = VnchwTrans(get_caller_context(), dst, src,
                                     repeat_times, dst_repeat_stride,
                                     src_repeat_stride)
            tik_instance.context.curr_scope().add_stmt(vtrans_stmt)

        return func(tik_instance, dst, src, repeat_times,
                    dst_repeat_stride, src_repeat_stride)
    return wrapper


def vec_scalar_elewise_func_dec(func):
    """bind this decorator with vector_scalar_elewise_func

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, mask, dst, scalar, # pylint: disable=R0913
                repeat_times, dst_blk_stride, dst_rep_stride, stride_unit=0,
                mask_mode="normal", print_name=None, mask_o=None):
        """bind this decorator with vector_scalar_elewise_func"""
        if not tik_instance.debug_disabled:
            stmt = VectorScalarEltwise(get_caller_context(),
                                       name, mask, dst, scalar, repeat_times,
                                       dst_blk_stride, dst_rep_stride,
                                       stride_unit, mask_mode)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, name, mask, dst, scalar,
                    repeat_times, dst_blk_stride,
                    dst_rep_stride, stride_unit,
                    mask_mode, print_name, mask_o)

    return wrapper


def vec_reduce_decorator(func):
    """bind this decorator with vec_reduce

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, mask, dst, src, # pylint: disable=R0913
                repeat_times, dst_rep_stride, src_blk_stride,
                src_rep_stride, stride_unit, order, maxmin_cnt_index):
        """bind this decorator with vec_reduce"""
        if not tik_instance.debug_disabled:
            ctx = tik_instance.context
            if maxmin_cnt_index is not None:
                # maxmin_cnt_index may have 1 or 3 elements
                for mci_tmp in maxmin_cnt_index:
                    ctx.bind_var(mci_tmp.debug_var)
                    ctx.set_scalar_var_mapping(mci_tmp.reg_buffer,
                                               mci_tmp.debug_var)
            vec_reduce = VecReduce(get_caller_context(), name,
                                   mask, dst, src,
                                   repeat_times, dst_rep_stride,
                                   src_blk_stride, src_rep_stride,
                                   stride_unit, order, maxmin_cnt_index)
            ctx.curr_scope().add_stmt(vec_reduce)
        return func(tik_instance, name, mask, dst, src,
                    repeat_times, dst_rep_stride, src_blk_stride,
                    src_rep_stride, stride_unit, order, maxmin_cnt_index)

    return wrapper


def vec_reduce_wo_order_decorator(func):
    """bind this decorator with vec_reduce_wo_order

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, mask, dst, src, # pylint: disable=R0913
                repeat_times, dst_rep_stride, src_blk_stride,
                src_rep_stride, stride_unit, order=None, print_name=None):
        """bind this decorator with vec_reduce_wo_order"""
        if not tik_instance.debug_disabled:
            vec_reduce = VecReduce(get_caller_context(), name,
                                   mask, dst, src, repeat_times,
                                   dst_rep_stride, src_blk_stride,
                                   src_rep_stride, stride_unit,
                                   order, maxmin_cnt_index=None,
                                   print_name=print_name)
            tik_instance.context.curr_scope().add_stmt(vec_reduce)
        return func(tik_instance, name, mask, dst, src,
                    repeat_times, dst_rep_stride,
                    src_blk_stride, src_rep_stride, stride_unit, print_name)

    return wrapper


def vec_reduce_group_decorator(func):
    """bind this decorator with vec_reduce_wo_order

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, mask, dst, src,  # pylint: disable=R0913
                repeat_times, dst_rep_stride, src_blk_stride,
                src_rep_stride, stride_unit, order=None):
        """bind this decorator with vec_reduce_wo_order"""
        # modified args' name for static changes
        if not tik_instance.debug_disabled:
            vec_reduce = VecReduce(get_caller_context(), name,
                                   mask, dst, src, repeat_times,
                                   dst_rep_stride, src_blk_stride,
                                   src_rep_stride, stride_unit,
                                   order, maxmin_cnt_index=None)
            tik_instance.context.curr_scope().add_stmt(vec_reduce)
        return func(tik_instance, name, mask, dst, src,
                    repeat_times, dst_rep_stride,
                    src_blk_stride, src_rep_stride, stride_unit)

    return wrapper


def vsel_decorator(func):
    """bind this decorator with vsel

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, mask, mode, dst,  # pylint: disable=R0913, R0914
                sel, src0, src1, repeat_times,
                dst_block_stride, src_block_stride, src1_block_stride,
                dst_rep_stride=0, src0_rep_stride=0,
                src1_rep_stride=0, name="vsel", mask_o=None):
        """bind this decorator with vsel"""
        if not tik_instance.debug_disabled:
            vsel_stmt = VSEL(get_caller_context(), mask, mode, dst, sel, src0,
                             src1, repeat_times, dst_block_stride,
                             src_block_stride, src1_block_stride,
                             dst_rep_stride, src0_rep_stride,
                             src1_rep_stride, name)
            tik_instance.context.curr_scope().add_stmt(vsel_stmt)
        return func(tik_instance, mask, mode, dst, sel, src0, src1,
                    repeat_times, dst_block_stride, src_block_stride,
                    src1_block_stride, dst_rep_stride, src0_rep_stride,
                    src1_rep_stride, name, mask_o)

    return wrapper


def vec_sel_decorator(func):
    """bind this decorator with vsel

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, mask, mode, dst, sel, # pylint: disable=R0913
                src0, src1, repeat_times, dst_rep_stride=0,
                src0_rep_stride=0, src1_rep_stride=0,
                instr_name=None, mask_o=None):
        """bind this decorator with vsel"""
        if not tik_instance.debug_disabled:
            blk_stride = 1
            if instr_name is None:
                instr_name = "vec_sel"
            vsel_stmt = VSEL(get_caller_context(), mask, mode, dst, sel, src0,
                             src1, repeat_times, blk_stride,
                             blk_stride, blk_stride,
                             dst_rep_stride, src0_rep_stride, src1_rep_stride,
                             name=instr_name)
            tik_instance.context.curr_scope().add_stmt(vsel_stmt)
        return func(tik_instance, mask, mode, dst, sel, src0, src1,
                    repeat_times, dst_rep_stride, src0_rep_stride,
                    src1_rep_stride, instr_name, mask_o)

    return wrapper


def scatter_vsel_decorator(func):
    """bind this decorator with scatter_vsel

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, mask, mode, dst_list, # pylint: disable=R0913
                sel, src0_list, src1, repeat_times,
                dst_rep_stride=0, src0_rep_stride=0, src1_sel_rep_stride=0):
        """bind this decorator with scatter_vsel"""
        if not tik_instance.debug_disabled:
            stmt = ScatterVsel(get_caller_context(), mask, mode, dst_list, sel,
                               src0_list, src1, repeat_times, dst_rep_stride,
                               src0_rep_stride, src1_sel_rep_stride)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, mask, mode, dst_list, sel, src0_list, src1,
                    repeat_times, dst_rep_stride, src0_rep_stride,
                    src1_sel_rep_stride)

    return wrapper


def vcmp_decorator(func):
    """bind this decorator with vcmp

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, mask, src0, src1, # pylint: disable=R0913
                src0_stride, src1_stride, print_name=None, mask_o=None):
        """bind this decorator with vcmp"""
        if not tik_instance.debug_disabled:
            vcmp_stmt = VCMP(get_caller_context(), name,
                             mask, src0, src1,
                             src0_stride, src1_stride)
            tik_instance.context.curr_scope().add_stmt(vcmp_stmt)
        return func(tik_instance, name, mask, src0,
                    src1, src0_stride, src1_stride, print_name, mask_o)

    return wrapper


def vms_decarator(func):
    """bind this decorator with vms

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, src_list, # pylint: disable=R0913
                element_lengths, if_exhausted_suspension,
                valid_bit, repeat_times=1, vms4_sr_scalar_array=None):
        """bind this decorator with vms"""
        if not tik_instance.debug_disabled:
            vms_stmt = VMS4(get_caller_context(), dst, src_list,
                            element_lengths, if_exhausted_suspension,
                            valid_bit, repeat_times, vms4_sr_scalar_array)
            tik_instance.context.curr_scope().add_stmt(vms_stmt)
        return func(tik_instance, dst, src_list, element_lengths,
                    if_exhausted_suspension, valid_bit, repeat_times,
                    vms4_sr_scalar_array)

    return wrapper


def vextract_decorator(func):
    """bind this decorator with vextract

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, src, repeat_times, mode_number):
        """bind this decorator with vextract"""
        if not tik_instance.debug_disabled:
            vextract_stmt = VEXTRACT(get_caller_context(), dst, src,
                                     repeat_times, mode_number)
            tik_instance.context.curr_scope().add_stmt(vextract_stmt)
        return func(tik_instance, dst, src, repeat_times, mode_number)

    return wrapper


def vconcate_decorator(func):
    """bind this decorator with

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, src, repeat_times, mode_number):
        """bind this decorator with"""
        if not tik_instance.debug_disabled:
            vconcate_stmt = VCONCAT(get_caller_context(), dst, src,
                                    repeat_times, mode_number)
            tik_instance.context.curr_scope().add_stmt(vconcate_stmt)
        return func(tik_instance, dst, src, repeat_times, mode_number)

    return wrapper


def rpn_cor_decorator(func):
    """bind this decorator with rpn_cor

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, src0, src1,  # pylint: disable=R0913
                src0_stride, src1_stride, repeat_times):
        """bind this decorator with rpn_cor"""
        if not tik_instance.debug_disabled:
            rpn_cor_stmt = RpnCor(get_caller_context(), src0, src1,
                                  src0_stride, src1_stride, repeat_times)
            tik_instance.context.curr_scope().add_stmt(rpn_cor_stmt)
        return func(tik_instance, src0, src1,
                    src0_stride, src1_stride, repeat_times)

    return wrapper


def rpn_cor_diag_decorator(func):
    """bind this decorator with rpn_cor_diag

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, src, src_register):
        """bind this decorator with rpn_cor_diag"""
        if not tik_instance.debug_disabled:
            rpn_cor_diag_stmt = RpnCorDiag(get_caller_context(), dst, src)
            tik_instance.context.curr_scope().add_stmt(rpn_cor_diag_stmt)
        return func(tik_instance, dst, src, src_register)

    return wrapper


def object_special_decorator(func):
    """bind this decorator with object_special

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    _name2clazz = {
        'viou': VIOU,
        'vrpac': VRPAC,
        'vaadd': VAADD,
        'vbitsort': VBS16,
        'vmergech': VMergeCH
    }

    @wraps(func)
    def wrapper(tik_instance, name, dst, src_list, repeat_times):
        """bind this decorator with object_special"""
        if name in _name2clazz.keys():
            if not tik_instance.debug_disabled:
                stmt = _name2clazz[name](
                    get_caller_context(),
                    dst, src_list, repeat_times)
                tik_instance.context.curr_scope().add_stmt(stmt)
        else:
            sys.stderr.write(
                "[NOTE]: '%s' is not supported in debug system yet!\n" % name)
        return func(tik_instance, name, dst, src_list, repeat_times)

    return wrapper


def set_rpn_cor_ir_decorator(func):
    """bind this decorator with set_rpn_cor_ir

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, number):
        """bind this decorator with set_rpn_cor_ir"""
        if not tik_instance.debug_disabled:
            stmt = WriteSPR(get_caller_context(), 'RPN_COR_IR', number)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, number)

    return wrapper


def set_rpn_offset_decorator(func):
    """bind this decorator with set_rpn_offset

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, number):
        """bind this decorator with set_rpn_offset"""
        if not tik_instance.debug_disabled:
            stmt = SetRpnOffset(get_caller_context(), number)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, number)

    return wrapper


def tik_return_decorator(func):
    """bind this decorator with tik_return

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance):
        """bind this decorator with tik_return"""
        if not tik_instance.debug_disabled:
            stmt = Return(get_caller_context())
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance)

    return wrapper


def scat_vec_single_elewise_dec(func):
    """bind this decorator with scatter_vector_single_elewise_func

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, mask, dst_list,  # pylint: disable=R0913
                src_list, repeat_times, dst_rep_stride, src_rep_stride):
        """bind this decorator with scatter_vector_single_elewise_func"""
        if not tik_instance.debug_disabled:
            stmt = ScatterSingleVector(get_caller_context(),
                                       name, mask, dst_list,
                                       src_list, repeat_times, dst_rep_stride,
                                       src_rep_stride)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, name, mask, dst_list, src_list,
                    repeat_times, dst_rep_stride, src_rep_stride)

    return wrapper


def scat_vec_bin_ternary_ele_dec(func):
    """bind this decorator with scatter_vector_binary_ternary_elewise_func

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, mask,  # pylint: disable=R0913
                dst_list, src0_list, src1_list, repeat_times,
                dst_rep_stride, src0_rep_stride, src1_rep_stride, *strides):
        """bind this decorator with
        scatter_vector_binary_ternary_elewise_func"""
        if not tik_instance.debug_disabled:
            stmt = ScatterVectorBinary(get_caller_context(),
                                       name, mask, dst_list,
                                       src0_list, src1_list, repeat_times,
                                       dst_rep_stride, src0_rep_stride,
                                       src1_rep_stride, *strides)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, name, mask, dst_list, src0_list, src1_list,
                    repeat_times, dst_rep_stride, src0_rep_stride,
                    src1_rep_stride, *strides)

    return wrapper


def scatter_vmulconv_decorator(func):
    """bind this decorator with scatter_vmulconv

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, mask, store_high_half,  # pylint: disable=R0913
                dst_list, src0_list, src1_list, repeat_times, dst_rep_stride,
                src0_rep_stride, src1_rep_stride):
        """bind this decorator with scatter_vmulconv"""
        if not tik_instance.debug_disabled:
            stmt = ScatterVmulconv(get_caller_context(), mask, store_high_half,
                                   dst_list, src0_list, src1_list,
                                   repeat_times, dst_rep_stride,
                                   src0_rep_stride, src1_rep_stride)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, mask, store_high_half,
                    dst_list, src0_list, src1_list,
                    repeat_times, dst_rep_stride, src0_rep_stride,
                    src1_rep_stride)

    return wrapper


def scat_vcmp_elewise_func_dec(func):
    """bind this decorator with scatter_vcmp_elewise_func

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, mask, src0_list, src1_list):
        """bind this decorator with scatter_vcmp_elewise_func"""
        if not tik_instance.debug_disabled:
            stmt = ScatterVCMP(get_caller_context(), name,
                               mask, src0_list, src1_list)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, name, mask, src0_list, src1_list)

    return wrapper


def scatter_vector_scalar_decorator(func):
    """bind this decorator with scatter_vector_scalar

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, mask,  # pylint: disable=R0913
                store_high_half, dst_list, src0_list, src1, repeat_times,
                dst_stride, src_stride, mask_mode="normal"):
        """bind this decorator with scatter_vector_scalar"""
        if not tik_instance.debug_disabled:
            stmt = ScatterVectorScalar(get_caller_context(),
                                       name, mask,
                                       store_high_half, dst_list, src0_list,
                                       src1, repeat_times, dst_stride,
                                       src_stride, mask_mode)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, name, mask,
                    store_high_half, dst_list, src0_list, src1,
                    repeat_times, dst_stride, src_stride, mask_mode)

    return wrapper


def vnchwconv_decorator(func):
    """bind this decorator with vnchwconv

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst_high_half,  # pylint: disable=R0913
                src_high_half, dst_list, src_list,
                repeat_times, dst_rep_stride, src_rep_stride, name=None):
        """bind this decorator with vnchwconv"""
        if not tik_instance.debug_disabled:
            stmt = Vnchwconv(get_caller_context(), dst_high_half,
                             src_high_half, dst_list, src_list,
                             repeat_times, dst_rep_stride, src_rep_stride, name)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, dst_high_half,
                    src_high_half, dst_list, src_list,
                    repeat_times, dst_rep_stride, src_rep_stride, name)

    return wrapper


def vec_trans_scatter_decorator(func):
    """bind this decorator with vnchwconv

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst_high_half,   # pylint: disable=R0913
                src_high_half, dst_list, src_list,
                repeat_times, dst_rep_stride, src_rep_stride):
        """bind this decorator with vec_trans_scatter"""
        if not tik_instance.debug_disabled:
            stmt_ = Vnchwconv(get_caller_context(), dst_high_half,
                              src_high_half, dst_list, src_list,
                              repeat_times, dst_rep_stride, src_rep_stride,
                              name="vec_trans_scatter")
            tik_instance.context.curr_scope().add_stmt(stmt_)
        return func(tik_instance, dst_high_half,
                    src_high_half, dst_list, src_list,
                    repeat_times, dst_rep_stride, src_rep_stride)

    return wrapper


def list_list_elewise_decorator(func):
    """bind this decorator with list_list_elewise

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, dst_list, src0_list, src1_list):
        """bind this decorator with list_list_elewise"""
        if not tik_instance.debug_disabled:
            stmt = ListListEltwise(get_caller_context(), name,
                                   dst_list, src0_list, src1_list)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, name, dst_list, src0_list, src1_list)

    return wrapper


def col2img_decorator(func):
    """bind this decorator with col2img

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, src, pad,  # pylint: disable=R0913, R0914
                l1_h, l1_w, fetch_filter_w, fetch_filter_h, left_top_w,
                left_top_h, stride_w, stride_h, filter_w, filter_h,
                dilation_filter_w, dilation_filter_h, repeat_time):
        """bind this decorator with col2img"""
        if not tik_instance.debug_disabled:
            stmt = Col2Img(get_caller_context(), dst, src, pad, l1_h, l1_w,
                           fetch_filter_w, fetch_filter_h, left_top_w,
                           left_top_h, stride_w, stride_h, filter_w, filter_h,
                           dilation_filter_w, dilation_filter_h, repeat_time)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, dst, src, pad, l1_h, l1_w, fetch_filter_w,
                    fetch_filter_h, left_top_w, left_top_h, stride_w, stride_h,
                    filter_w, filter_h, dilation_filter_w, dilation_filter_h,
                    repeat_time)

    return wrapper


def tensor_move_decorator(func):
    """bind this decorator with tensor_move

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, src,  # pylint: disable=R0913, R0914
                block_mode, nburst, burst_len,
                dst_stride, src_stride, deqscale=None, sid_store_mode=0,
                relu=False, pad_mode=None, pad_value=None, onthefly_mode=0,
                src_onthefly=None, src_onthefly_stride=0):
        """bind this decorator with tensor_move"""
        if not tik_instance.debug_disabled:
            stmt = TensorMove(
                get_caller_context(), dst, src, block_mode, nburst, burst_len,
                dst_stride, src_stride, deqscale, sid_store_mode, relu,
                pad_mode, pad_value, onthefly_mode, src_onthefly,
                src_onthefly_stride)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(
            tik_instance, dst, src, block_mode, nburst, burst_len, dst_stride,
            src_stride, deqscale, sid_store_mode, relu, pad_mode, pad_value,
            onthefly_mode, src_onthefly, src_onthefly_stride)

    return wrapper


def mov_cmpmask_to_tensor_decorator(func):
    """bind this decorator with mov_cmpmask_to_tensor

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, src_cmpmask):
        """bind this decorator with mov_cmpmask_to_tensor"""
        if not tik_instance.debug_disabled:
            stmt = MoveCMPMASK2Tensor(get_caller_context(), dst)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, dst, src_cmpmask)

    return wrapper


def mov_tensor_to_cmpmask_decorator(func):
    """bind this decorator with mov_tensor_to_cmpmask

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, src):
        """bind this decorator with mov_tensor_to_cmpmask"""
        if not tik_instance.debug_disabled:
            stmt = MoveTensor2CMPMASK(get_caller_context(depth=2), src)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, src)

    return wrapper


def scalar_single_decorator(func):
    """bind this decorator with scalar_single

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, dst, src):
        """bind this decorator with scalar_single"""
        if not tik_instance.debug_disabled:
            stmt = ScalarSingleOp(get_caller_context(),
                                  name, dst, src)
            ctx = tik_instance.context
            ctx.curr_scope().add_stmt(stmt)
            ctx.bind_var(dst.debug_var)
            ctx.set_scalar_var_mapping(dst.reg_buffer, dst.debug_var)
        return func(tik_instance, name, dst, src)

    return wrapper


def scalar_binary_decorator(func):
    """bind this decorator with scalar_single

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, name, dst, src0, src1):
        """bind this decorator with scalar binary instr"""
        if not tik_instance.debug_disabled:
            stmt = ScalarBinaryOp(get_caller_context(),
                                  name, dst, src0, src1)
            ctx = tik_instance.context
            ctx.curr_scope().add_stmt(stmt)
            ctx.bind_var(dst.debug_var)
            ctx.set_scalar_var_mapping(dst.reg_buffer, dst.debug_var)
        return func(tik_instance, name, dst, src0, src1)
    return wrapper


def scalar_conv_decorator(func):
    """bind this decorator with scalar_conv

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, round_mode, dst, src):
        """bind this decorator with scalar_conv"""
        if not tik_instance.debug_disabled:
            stmt = ScalarConv(get_caller_context(), round_mode, dst, src)
            ctx = tik_instance.context
            ctx.curr_scope().add_stmt(stmt)
            ctx.bind_var(dst.debug_var)
            ctx.set_scalar_var_mapping(dst.reg_buffer, dst.debug_var)
        return func(tik_instance, round_mode, dst, src)

    return wrapper


def vms4_to_scalar_decorator(func):
    """bind this decorator with vms4_to_scalar

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, scalar_list, vms4_sr):
        """bind this decorator with vms4_to_scalar

        Parameters
        ----------
        tik_instance: an instance of Tik

        scalar_list: a list of Scalar

        vms4_sr: tik_params's VMS4 SR

        Returns
        ----------
        func: the decorated function
        """
        TikCheckUtil.check_type_match(scalar_list, (list, tuple))
        if not tik_instance.debug_disabled:
            ctx = tik_instance.context
            stmt = VMS4SR2Scalar(get_caller_context(), scalar_list)
            ctx.curr_scope().add_stmt(stmt)
            for scalar in scalar_list:
                ctx.bind_var(scalar.debug_var)
                ctx.set_scalar_var_mapping(scalar.reg_buffer,
                                           scalar.debug_var)
        return func(tik_instance, scalar_list, vms4_sr)

    return wrapper


def set_l0_set_value_decorator(func):
    """bind this decorator with set_l0_set_value

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, value, dtype):
        """bind this decorator with set_l0_set_value"""
        if not tik_instance.debug_disabled:
            stmt = SetL0SetValue(get_caller_context(), value, dtype)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, value, dtype)
    return wrapper


def set_2d_decorator(func):
    """bind this decorator with tensor_padding_with_matrix

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """
    @wraps(func)
    def wrapper(tik_instance, dst, repeat_times, value=None):
        """bind this decorator with tensor_padding_with_matrix

        Parameters
        ----------
        tik_instance: an instance of Tik
        dst: the destination tensor
        repeat_times: Repeated iterations times

        Returns
        ----------
        func: the decorated function
        """
        if not tik_instance.debug_disabled:
            stmt = Set2D(get_caller_context(), dst, repeat_times, value)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, dst, repeat_times, value)
    return wrapper


def set_ctrl_bits(begin, end=None):
    """bind this decorator with set ctrl instructions

    Parameters
    ----------
    begin : the begin index of ctrl register
    end : the end index of ctrl register

    Returns
    -------
    function
    """
    def set_ctrl_bits_wrapper(func):
        """bind this decorator with set ctrl instructions"""
        def wrapper(tik_instance, num):
            """bind this decorator with set ctrl instructions

            Parameters
            ----------
            tik_instance: an instance of Tik
            num : ctrl special register bit value

            Returns
            -------
            function
            """
            if not tik_instance.debug_disabled:
                # CRTL is 64 bit, if it changed we must change as well
                mask = 2**64 - 1
                end_ = end
                if end_ is None:
                    end_ = begin + 1
                for i in range(begin, end_):
                    mask ^= (1 << i)
                stmt = SetCtrlSPR(get_caller_context(), num, mask, begin)
                tik_instance.context.curr_scope().add_stmt(stmt)

            with tik_instance.context.freeze():
                return func(tik_instance, num)
        return wrapper
    return set_ctrl_bits_wrapper


def get_ctrl_bits(begin, end=None):
    """bind this decorator with get ctrl instructions

    Parameters
    ----------
    begin : the begin index of ctrl register
    end : the end index of ctrl register

    Returns
    -------
    function
    """
    def get_ctrl_bits_wrapper(func):
        """bind this decorator with get ctrl instructions"""
        def wrapper(tik_instance, scalar):
            """bind this decorator with get ctrl instructions

            Parameters
            ----------
            tik_instance: an instance of Tik
            scalar : the value of ctrl register bits

            Returns
            -------
            function
            """
            if not tik_instance.debug_disabled:
                # CRTL is 64 bit, if it changed we must change as well
                mask = 0
                end_ = end
                if end_ is None:
                    end_ = begin + 1
                for i in range(begin, end_):
                    mask |= (1 << i)
                stmt = GetCtrlSPR(get_caller_context(), scalar, mask, begin)
                tik_instance.context.curr_scope().add_stmt(stmt)

            with tik_instance.context.freeze():
                return func(tik_instance, scalar)
        return wrapper
    return get_ctrl_bits_wrapper


def vcmpvs_elewise_func_decorator(func):
    """bind this decorator with vcmpvs instructions"""
    @wraps(func)
    def wrapper(tik_instance, name, dst,  # pylint: disable=R0913
                src, scalar, repeat_times, src_blk_stride, src_rep_stride):
        """vcmpvs instruction, compare by element

        Parameters
        ----------
        tik_instance: an instance of tik
        name : instruction name
        dst : destination operator
        src : source operation
        scalar : source scalar operation
        repeat_times : Repeated iterations times
        src_blk_stride : offset of src operator between different block
        src_rep_stride : offset of src operator in the same block between

        Returns
        -------
        None
        """
        if not tik_instance.debug_disabled:
            stmt = VecCMPScalar(get_caller_context(), name, dst,
                                src, scalar, repeat_times,
                                src_blk_stride, src_rep_stride)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, name, dst,
                    src, scalar, repeat_times,
                    src_blk_stride, src_rep_stride)
    return wrapper


def v4dtrans_decorator(func):
    """bind this decorator with v4dtrans instructions"""
    @wraps(func)
    def wrapper(tik_instance, chw2hwc, dst, src,  # pylint: disable=R0913
                m_len, channels):
        """ transform data between chw and hwc

        Parameters
        ----------
        tik_instance: an instance of tik
        chw2hwc : bool, True - chw->hwc; False - hwc->chw
        dst : destination operator
        src : source operation
        m_len : H*W direction dimension
        channels: size of C

        Returns
        -------
        None
        """
        if not tik_instance.debug_disabled:
            stmt = V4DTRANS(get_caller_context(), chw2hwc, dst, src, m_len,
                            channels)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, chw2hwc, dst, src, m_len, channels)

    return wrapper


def vreduce_decorator(func):
    """bind this decorator with vreduce instructions"""
    @wraps(func)
    def wrapper(tik_instance, mask, dst, src0,  # pylint: disable=R0913
                src1_pattern, repeat_times, src0_blk_stride, src0_rep_stride,
                src1_rep_stride, stride_unit=0, rsvd_scalar=None,
                mask_mode="normal"):
        """
        source vector would be reduced into shorter vector according to the
        compare masks

        Parameters
        ----------
        tik_instance: an instance of tik
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src0 : source operator
        src1_pattern : 6 fixed patterns for effective operation on element
        repeat_times : Repeated iterations times
        src0_blk_stride : offset of src operator between different block
                         in one iteration
        src0_rep_stride : offset of src0 operator in the same block
                         between adjacent iterations
        src1_rep_stride : offset of src1 operator in the same block
                         between adjacent iterations
        stride_unit : address and offset unit both affect it. default = 0
        rsvd_scalar : remaining elements count, default = none
        mask_mode: "normal" - mask normal mode
                   "counter" - mask counter mode

        Returns
        -------
        None
        """
        if not tik_instance.debug_disabled:
            if rsvd_scalar is not None:
                if not is_scalar(rsvd_scalar):
                    TikCheckUtil.raise_error(
                        "rsvd_scalar should be None or Scalar, input type of "
                        "rsvd_scalar: {}".format(type(rsvd_scalar)))
                debug_var = rsvd_scalar.debug_var
            else:
                debug_var = None

            stmt = VReduce(get_caller_context(), mask, dst, src0, src1_pattern,
                           repeat_times, src0_blk_stride, src0_rep_stride,
                           src1_rep_stride, stride_unit, debug_var, mask_mode)
            if debug_var is not None:
                tik_instance.context.bind_var(debug_var)
                tik_instance.context.set_scalar_var_mapping(
                    rsvd_scalar.reg_buffer, debug_var)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, mask, dst, src0, src1_pattern,
                    repeat_times, src0_blk_stride, src0_rep_stride,
                    src1_rep_stride, stride_unit, rsvd_scalar, mask_mode)

    return wrapper


def vpadding_decorator(func):
    """bind this decorator with vpadding instructions"""
    @wraps(func)
    def wrapper(tik_instance, mask, pad_mode,  # pylint: disable=R0913
                pad_side, dst, src, repeat_times, dst_blk_stride,
                src_blk_stride, dst_rep_stride, src_rep_stride,
                stride_unit=0, mask_mode="normal"):
        """ transform data between chw and hwc

        Parameters
        ----------
        mask:
        tik_instance: an instance of tik
        pad_mode: 0 -> nearest-padding(aaa|a)
                  1 -> symmetric_padding0(abc|cba)
                  2 -> symmetric_padding1(ab|cba)
        pad_side: 'left'/'right'.
        dst: dst operator
        src: src operator
        repeat_times: Repeated iterations times
        dst_blk_stride: offset of dst operator between different block
                         in one iteration
        src_blk_stride: offset of src operator between different block
                         in one iteration
        dst_rep_stride: offset of dst operator in the same block
                         between adjacent iterations
        src_rep_stride: offset of src operator in the same block
                         between adjacent iterations
        stride_unit: address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        if not tik_instance.debug_disabled:
            stmt = VPadding(get_caller_context(), mask, pad_mode, pad_side, dst,
                            src, repeat_times, dst_blk_stride, src_blk_stride,
                            dst_rep_stride, src_rep_stride, stride_unit,
                            mask_mode)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, mask, pad_mode, pad_side,
                    dst, src, repeat_times, dst_blk_stride, src_blk_stride,
                    dst_rep_stride, src_rep_stride, stride_unit, mask_mode)
    return wrapper


def vscatter_decorator(func):
    """bind this decorator with vscatter instructions"""
    @wraps(func)
    def wrapper(tik_instance, mask, dst, src,   # pylint: disable=R0913
                dst_offset, repeat_times,
                src_rep_stride, base_addr=0,
                stride_unit=0, mask_mode="normal"):
        """bind this decorator with vscatter_decorator"""
        if not tik_instance.debug_disabled:
            stmt = VScatter(get_caller_context(), mask, dst, src,
                            dst_offset, repeat_times,
                            src_rep_stride, base_addr, stride_unit, mask_mode)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, mask, dst, src,
                    dst_offset, repeat_times,
                    src_rep_stride, base_addr, stride_unit, mask_mode)
    return wrapper


def vgather_decorator(func):
    """bind this decorator with vgather instructions"""
    @wraps(func)
    def wrapper(tik_instance, mask, dst, src,   # pylint: disable=R0913
                src_offset, repeat_times, dst_rep_stride,
                base_addr=0, stride_unit=0, mask_mode="normal"):
        """bind this decorator with vgather_decorator"""
        if not tik_instance.debug_disabled:
            stmt = VGather(get_caller_context(), mask, dst, src,
                           src_offset, repeat_times, dst_rep_stride,
                           base_addr, stride_unit, mask_mode)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, mask, dst, src,
                    src_offset, repeat_times, dst_rep_stride,
                    base_addr, stride_unit, mask_mode)
    return wrapper


def load3dv2_decorator(func):
    """bind this decorator with load3dv2 instructions"""
    @wraps(func)
    def wrapper(tik_instance, dst, src,  # pylint: disable=R0913, R0914
                pad_list, l1_height, l1_width, channel_size, k_extension,
                m_extension, k_start_pt, m_start_pt, stride_w, stride_h,
                filter_w, filter_h, dilation_filter_w, dilation_filter_h,
                en_transpose=False, en_small_k=False, pad_value=None):
        """bind this decorator with load3dv2_decorator"""
        if not tik_instance.debug_disabled:
            load3dv2_stmt = Load3DV2(get_caller_context(), dst, src, pad_list,
                                     l1_height, l1_width, channel_size,
                                     k_extension, m_extension, k_start_pt,
                                     m_start_pt, stride_w, stride_h, filter_w,
                                     filter_h, dilation_filter_w,
                                     dilation_filter_h, en_transpose,
                                     en_small_k, pad_value)
            tik_instance.context.curr_scope().add_stmt(load3dv2_stmt)
        return func(tik_instance, dst, src, pad_list, l1_height, l1_width,
                    channel_size, k_extension, m_extension,
                    k_start_pt, m_start_pt, stride_w, stride_h,
                    filter_w, filter_h, dilation_filter_w,
                    dilation_filter_h, en_transpose, en_small_k, pad_value)
    return wrapper


def depthwise_conv_decorator(func):
    """bind this decorator with depthwise_conv instruction"""
    @wraps(func)
    def wrapper(tik_instance, dst, src0, src1,  # pylint: disable=R0913
                pad_mode, l1_h, l1_w, store_high_half=False, feature_offset=0,
                weight_offset=None, pad_value=None):
        if not tik_instance.debug_disabled:
            stmt = DepthwiseConv(get_caller_context(), dst, src0, src1, pad_mode,
                                 l1_h, l1_w, store_high_half, feature_offset,
                                 weight_offset, pad_value)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, dst, src0, src1, pad_mode, l1_h, l1_w,
                    store_high_half, feature_offset, weight_offset, pad_value)
    return wrapper


def load_smask_decorator(func):
    """bind this decorator with load_smask instruction"""
    @wraps(func)
    def wrapper(tik_instance, dst, src, load_size, sid=0):
        if not tik_instance.debug_disabled:
            stmt = LoadSmask(get_caller_context(), dst, src, load_size, sid)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, dst, src, load_size, sid)
    return wrapper


def load_image_decorator(func):
    """bind this decorator with load_image

    Parameters
    ----------
    func : the decorated function

    Returns
    -------
    function
    """

    @wraps(func)
    def wrapper(tik_instance, dst, src0, src1,  # pylint: disable=R0913, R0914
                input_format, function_switch,
                src_info, crop_info, pre_clip_info, swap_list,
                csc_info, scf_info, post_clip_info, dtc_info, flip_mode,
                channel_pad_info, area_pad_info,
                stretch_info, raw_info, sid):
        """bind this decorator with load_image"""
        if not tik_instance.debug_disabled:
            stmt = LoadImage(get_caller_context(), dst, src0, src1,
                             input_format, function_switch,
                             src_info, crop_info, pre_clip_info, swap_list,
                             csc_info, scf_info, post_clip_info, dtc_info,
                             flip_mode,
                             channel_pad_info, area_pad_info,
                             stretch_info, raw_info, sid)
            tik_instance.context.curr_scope().add_stmt(stmt)

        return func(tik_instance, dst, src0, src1, input_format,
                    function_switch,
                    src_info, crop_info, pre_clip_info, swap_list,
                    csc_info, scf_info, post_clip_info, dtc_info, flip_mode,
                    channel_pad_info, area_pad_info,
                    stretch_info, raw_info, sid)

    return wrapper


def vec_reduce_add_decorator(func):
    """bind this decorator with vec_reduce_add instructions"""
    @wraps(func)
    def wrapper(tik_instance, mask, dst, src,  # pylint: disable=R0913
                work_tensor, repeat_times, src_rep_stride):
        if not tik_instance.debug_disabled:
            stmt = VReduceAdd(get_caller_context(), mask, dst, src, work_tensor,
                              repeat_times, src_rep_stride)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, mask, dst, src, work_tensor,
                    repeat_times, src_rep_stride)
    return wrapper


def vec_all_reduce_decorator(min_max_func):
    """bind this decorator with vec_reduce_max, vec_reduce_min

    Parameters
    ----------
    min_max_func : the decorated function vec_reduce_max/vec_reduce_min

    Returns
    -------
    function
    """
    def reduce_decorator(func):
        @wraps(func)
        def wrapper(tik_instance, mask,  # pylint: disable=R0913
                    dst, src, work_tensor,
                    repeat_times, src_rep_stride, cal_index=False):
            """bind this decorator with vec_all_reduce"""
            if not tik_instance.debug_disabled:
                ctx = tik_instance.context
                vec_reduce = VecAllReduce(get_caller_context(),
                                          min_max_func, mask, dst, src,
                                          work_tensor, repeat_times,
                                          src_rep_stride, cal_index)
                ctx.curr_scope().add_stmt(vec_reduce)
            return func(tik_instance, mask, dst, src, work_tensor,
                        repeat_times, src_rep_stride, cal_index)
        return wrapper

    return reduce_decorator


def winograd_fm_transform_decorator(func):
    """bind this decorator with winograd_feature_map_transform instructions"""
    @wraps(func)
    def wrapper(tik_instance, dst, src, l1_h,  # pylint: disable=R0913, R0914
                l1_w, l1_c, pad_left, pad_right, pad_top, pad_bottom,
                m_extension, m_start_pt, k_extension, k_start_pt,
                column_indicator, dst_gap):
        """bind this decorator with winograd_fm instructions"""
        if not tik_instance.debug_disabled:
            stmt = LoadL1ToL0AWinograd(
                get_caller_context(), dst, src, l1_h, l1_w, l1_c, pad_left,
                pad_right, pad_top, pad_bottom, m_extension, m_start_pt,
                k_extension, k_start_pt, column_indicator, dst_gap)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, dst, src, l1_h, l1_w, l1_c, pad_left,
                    pad_right, pad_top, pad_bottom, m_extension, m_start_pt,
                    k_extension, k_start_pt, column_indicator, dst_gap)
    return wrapper


def winograd_weight_trans_decorator(func):
    """bind this decorator with winograd_weight_transform instructions"""
    @wraps(func)
    def wrapper(tik_instance, dst, src,  # pylint: disable=R0913
                column_indicator, repeat_dir, repeat_times, dst_blk_stride,
                dst_rep_stride, src_rep_stride, en_weight_offset=False,
                smask=None):
        if not tik_instance.debug_disabled:
            stmt = LoadL1ToL0BWinograd(
                get_caller_context(), dst, src, column_indicator, repeat_dir,
                repeat_times, dst_blk_stride, dst_rep_stride, src_rep_stride,
                en_weight_offset, smask)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, dst, src, column_indicator, repeat_dir,
                    repeat_times, dst_blk_stride, dst_rep_stride,
                    src_rep_stride, en_weight_offset, smask)
    return wrapper


def mmad_brc_decorator(func):
    """bind this decorator with mmad_broadcast instructions"""
    @wraps(func)
    def wrapper(tik_instance, dst, src, repeat_mode,  # pylint: disable=R0913
                nburst, burst_repeat, dst_gap, src_gap):
        if not tik_instance.debug_disabled:
            stmt = MmadBrc(get_caller_context(), dst, src, repeat_mode, nburst,
                           burst_repeat, dst_gap, src_gap)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, dst, src, repeat_mode, nburst, burst_repeat,
                    dst_gap, src_gap)
    return wrapper


def vbi_decorator(func):
    """bind this decorator with vbi instructions"""
    @wraps(func)
    def wrapper(tik_instance, mask, dst, src0, src1,  # pylint: disable=R0913
                src0_offset, dst_blk_stride, vertical_repeat_times,
                horizontal_repeat_times, repeat_mode, vertical_repeat_offset):
        if not tik_instance.debug_disabled:
            stmt = VBI(get_caller_context(), mask, dst, src0, src1, src0_offset,
                       dst_blk_stride, vertical_repeat_times,
                       horizontal_repeat_times, repeat_mode,
                       vertical_repeat_offset)
            tik_instance.context.curr_scope().add_stmt(stmt)
        return func(tik_instance, mask, dst, src0, src1, src0_offset,
                    dst_blk_stride, vertical_repeat_times,
                    horizontal_repeat_times, repeat_mode,
                    vertical_repeat_offset)
    return wrapper
