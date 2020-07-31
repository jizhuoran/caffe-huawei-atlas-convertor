#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Runtime function related hooks
"""
# pylint: disable=too-many-lines
from __future__ import absolute_import as _abs

import copy
from functools import reduce as functools_reduce

from te import tvm

from . import cce_params as cce
from .cce_conf import CceProductParams

UINT64_T = 'uint64'


def get_align_factor(dtype):
    """
    get_align_factor
    """
    # base on the diff data type, get the align_factor
    if dtype in ('int8', 'uint8'):
        align_factor = 32
        dtype_bytes = 1
    elif dtype in ('float16', 'int16', 'uint16'):
        align_factor = 16
        dtype_bytes = 2
    else:
        align_factor = 8
        dtype_bytes = 4
    return align_factor, dtype_bytes


def get_dma_sid(intrinsic):
    """get dma sid

    Parameters
    ----------
    intrinsic : dma intrinsic

    Returns
    -------
    dma sid : int

    """
    cce_product_params = CceProductParams()
    str_sid = cce_product_params.getParams(intrinsic)
    return int(str_sid)


def get_real_src_dst(src, dst):
    """
    get real src dst
    """
    def ptr_determinate(buffer_local, label):
        """
        determinate ptr
        """
        if _key_word_(buffer_local.scope) == "OUT":
            return buffer_local.access_ptr(label)  # buffer.data
        return buffer_local.access_ptr(label)

    return (ptr_determinate(src, "r"), ptr_determinate(dst, "w"))


def _key_word_(name):
    """
    get key word
    """
    tmp = name.split(".")
    if tmp[0].lower() == "global":
        return "OUT"
    if tmp[1].count('UB'):
        return "UB"
    return tmp[1]


def dma_dependency_scope(src, dst, ib_expr):
    """
    get dma dependency scope
    """
    # pylint: disable=useless-object-inheritance, too-few-public-methods
    class StaticDmaList(object):
        """
        static dma info key
        """
        def __init__(self):
            pass

        # pipe_line 1 is M
        dma_list = {"L0C UB": 2,  # V
                    "UB L0C": 2,
                    "L1 L0A": 4,  # LSU1
                    "L1 L0B": 4,
                    "L1 UB": 4,
                    "OUT L1": 5,  # LSU2
                    "OUT L0A": 5,
                    "OUT L0B": 5,
                    "OUT UB": 5,
                    "UB OUT": 6,  # LSU3
                    "UB L1": 6,
                    "UB UB": 2  # V
                   }

    src_key_str = _key_word_(src.scope)
    dst_key_str = _key_word_(dst.scope)
    pipe_line = StaticDmaList.dma_list[src_key_str + " " + dst_key_str]

    ib_expr.scope_attr(cce.CCE_AXIS, "coproc_scope", pipe_line)

# pylint: disable=too-many-locals, too-many-statements
def get_buf_scope(name):
    """
    get buffer scope
    """
    tmp = name.split(".")
    if len(tmp) == 1:
        return cce.dma_copy_global

    mem_dict = {"UB": cce.scope_ubuf, "L1": cce.scope_cbuf, "L0A": cce.scope_ca,
                "L0B": cce.scope_cb, "L0C": cce.scope_cbuf}
    key = tmp[-1]
    if key not in mem_dict:
        raise RuntimeError("name [%s] is error." % name)

    return mem_dict[key]


def get_dma_buffer(stmt_in, is_storage_align=False):
    """
    get buffer info form the stmt_in
    """
    buf_info = {}

    shape = []
    loop_vars = []
    buf_info["loop_var"] = None
    if_var = []
    sel_var = []

    def _get_shape(stmt_op):
        """
        get shape in stmt op
        """
        if isinstance(stmt_op, tvm.stmt.For):
            loop_vars.append(stmt_op.loop_var)
            shape.append(stmt_op.extent)
            if isinstance(stmt_op.body, tvm.stmt.For):
                _get_shape(stmt_op.body)
            if isinstance(stmt_op.body, tvm.stmt.Store) and \
                    isinstance(stmt_op.body.value, tvm.expr.Select):
                def _get_sel_vars(stmt_op):
                    """
                    get op sel vars
                    """
                    sel_var.append(stmt_op)
                    if isinstance(stmt_op.true_value, tvm.expr.Select):
                        _get_sel_vars(stmt_op.true_value)

                _get_sel_vars(stmt_op.body.value)
            if isinstance(stmt_op.body, tvm.stmt.IfThenElse):
                def _get_if_vars(stmt_op):
                    """
                    get op if vars
                    """
                    if_var.append(stmt_op)
                    if isinstance(stmt_op.then_case, tvm.stmt.IfThenElse):
                        _get_if_vars(stmt_op.then_case)

                _get_if_vars(stmt_op.body)

    # reduction shape
    while stmt_in is not None and isinstance(stmt_in, tvm.stmt.AttrStmt):
        stmt_in = stmt_in.body

    _get_shape(stmt_in)

    if not shape:
        shape.append(tvm.const(1))

    buf_info["shape"] = tuple(shape)
    buf_info["loop_var"] = loop_vars

    loads = []
    store = []

    def _post_order(stmt_op):
        """
        post order op
        """
        if isinstance(stmt_op, tvm.expr.Load):
            loads.append(stmt_op)
            return None
        if isinstance(stmt_op, tvm.stmt.Store):
            store.append(stmt_op)
            return None

        return None

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Load", "Store"])

    def _offset(strides, itv, idx):
        """
        get strides offset
        """
        idx = tvm.ir_pass.Simplify(idx)
        if itv is not None and not isinstance(idx, tvm.expr.IntImm):
            offset = tvm.ir_pass.Simplify(idx)
            for i in range(len(itv) - 1, -1, -1):
                if tvm.ir_pass.ExprUseVar(offset, itv[i]):
                    offset = tvm.ir_pass.Simplify(offset - itv[i]*strides[i])
        else:
            offset = tvm.ir_pass.Simplify(idx)
        return offset

    src_buf = []
    for i in loads:
        cur_shape = [buf_info["shape"][j].value for j in
                     range(len(buf_info["shape"]))]
        strides = [functools_reduce(lambda n, k: n*k, cur_shape[j + 1:]) \
                   for j in range(len(cur_shape) - 1)]
        strides.append(1)
        if is_storage_align:
            elem_offset = 0
        else:
            elem_offset = _offset(strides, buf_info["loop_var"],
                                  i.index)
        buf = tvm.decl_buffer(buf_info["shape"], i.dtype,
                              name=i.buffer_var.name,
                              data=i.buffer_var,
                              offset_factor=1,
                              data_alignment=16,
                              scope=get_buf_scope(i.buffer_var.name),
                              strides=strides,
                              elem_offset=elem_offset)
        src_buf.append(buf)

    dest_buf = []
    for i in store:
        cur_shape = [buf_info["shape"][j].value for j in
                     range(len(buf_info["shape"]))]
        strides = [functools_reduce(lambda n, k: n*k, cur_shape[j + 1:]) \
                   for j in range(len(cur_shape) - 1)]
        strides.append(1)
        buf = tvm.decl_buffer(buf_info["shape"], i.value.dtype,
                              name=i.buffer_var.name,
                              data=i.buffer_var,
                              offset_factor=1,
                              data_alignment=16,
                              scope=get_buf_scope(i.buffer_var.name),
                              strides=strides,
                              elem_offset=_offset(strides, buf_info["loop_var"],
                                                  i.index))
        dest_buf.append(buf)

    return src_buf, dest_buf, if_var, sel_var


def get_type_bits(data_type):
    """
    get type bits
    """
    tmp = ''
    for i in data_type[::-1]:
        if i.isdigit():
            tmp += i
        else:
            break
    return int(tmp[::-1])


def equal_const_int(expr, value):
    """Returns if expr equals value.

    Parameters
    ----------
    expr : tvm.Expr
        The input expression.

    Returns
    -------
    equal : bool
        Whether they equals.
    """
    if isinstance(expr, int):
        return expr == value
    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        expr = tvm.ir_pass.Simplify(expr)
    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        return False
    return expr.value == value

# pylint: disable=too-many-locals
def _fold_buffer_dim(buf, scope, elem_block):
    """
    get shape and strides from buffer
    """
    ndim = len(buf.shape)
    x_size = 1
    base = 0
    for i in range(1, ndim + 1):
        x_size = x_size*buf.shape[ndim - i]
        if equal_const_int(x_size - elem_block, 0):
            base = i + 1
            break

    if base == 0:
        shape = []
        strides = []
        base = 1
    else:
        shape = [elem_block]
        strides = [1]

    while base < ndim + 1:
        x_size = 1
        x_stride = buf.strides[ndim - base]
        next_base = base
        if not equal_const_int(x_stride % elem_block, 0):
            raise RuntimeError(
                "scope %s need to have block=%d, shape=%s, strides=%s" % (
                    scope, elem_block, buf.shape, buf.strides))
        for i in range(base, ndim + 1):
            k = ndim - i
            if not equal_const_int(x_size*x_stride - buf.strides[k], 0):
                break
            x_size = x_size*buf.shape[k]
            next_base = i + 1
        shape.append(tvm.ir_pass.Simplify(x_size))
        strides.append(x_stride)
        assert next_base != base
        base = next_base

    strides = list(reversed(strides))
    shape = list(reversed(shape))
    return shape, strides

# pylint: disable=too-many-locals
def get_mov_pattern(src_buf, elem_width, elem_bytes, dst_buf, allow_fold):
    """
    get mov pattern
    """
    elem_block = elem_bytes*8 // elem_width

    src_shape, src_strides = list(src_buf.shape), list(src_buf.strides)
    dst_shape, dst_strides = list(dst_buf.shape), list(dst_buf.strides)
    if allow_fold and (len(dst_shape) != 1):
        # for move intrinsic any length is ok, just extend to multiply elem_block
        src_shape, src_strides = _fold_buffer_dim(src_buf, dst_buf.scope, 1)
        dst_shape, dst_strides = _fold_buffer_dim(dst_buf, src_buf.scope, 1)

    shape = src_shape if len(src_shape) > len(dst_shape) else dst_shape
    strides = src_strides if len(src_strides) > len(
        dst_strides) else dst_strides
    src_offset = src_buf.elem_offset
    dst_offset = dst_buf.elem_offset

    def raise_error():
        """Internal function to raise error """
        raise RuntimeError(
            ("Scope[%s]: cannot detect mov pattern with elem_block=%d: "
             "shape=%s, strides=%s") % (
                 dst_buf.scope, elem_block, shape, strides))

    ndim = len(shape)
    if not equal_const_int(strides[-1] - 1, 0):
        raise_error()
    if ndim == 1:
        burst = (shape[-1] + elem_block - 1) // elem_block
        nburst = 1
        src_stride = 0
        dst_stride = 0
        return src_offset, dst_offset, nburst, burst, src_stride, dst_stride

    if not equal_const_int(strides[-2] % elem_block, 0):
        raise_error()
    if ndim == 2:
        burst = (shape[-1] + elem_block - 1) // elem_block
        nburst = shape[-2]
        src_stride = src_strides[-2] // elem_block - burst if len(
            src_strides) == ndim else 0
        dst_stride = dst_strides[-2] // elem_block - burst if len(
            dst_strides) == ndim else 0
        return src_offset, dst_offset, nburst, burst, src_stride, dst_stride

    if not equal_const_int(strides[-3] % elem_block, 0):
        raise_error()
    if not equal_const_int(shape[-1] - elem_block, 0):
        raise_error()
    if ndim == 3:
        burst = shape[-2]
        nburst = shape[-3]
        src_stride = src_strides[-3] // elem_block - burst if len(
            src_strides) == ndim else 0
        dst_stride = dst_strides[-3] // elem_block - burst if len(
            dst_strides) == ndim else 0
        return src_offset, dst_offset, nburst, burst, src_stride, dst_stride
    raise_error()


# pylint: disable=inconsistent-return-statements
def dma_copy(ib_expr, src, dst):
    """
    dma copy between diff buffer
    """
    elem_width = get_type_bits(dst.dtype)

    if dst.scope == cce.scope_cb or dst.scope == cce.scope_ca:
        # Load2D
        raise RuntimeError("Do not support copy into cb or ca")

    # other DMA
    if src.scope == cce.scope_cc or dst.scope == cce.scope_cc:
        elem_bytes = cce.BLOCK_IN*cce.BLOCK_OUT*elem_width // 8
    else:
        elem_bytes = cce.GLB_ELEM_BYTES

    allow_fold = True

    _, _, nburst, burst, src_stride, dst_stride = get_mov_pattern(
        src, elem_width, elem_bytes, dst, allow_fold=allow_fold)
    dma_dependency_scope(src, dst, ib_expr)
    src_ptr, dst_ptr = get_real_src_dst(src, dst)
    sid = 0

    pad_mode_dict = {0: 'PAD_NONE', 1: 'PAD_MODE1', 2: 'PAD_MODE2',
                     3: 'PAD_MODE3',
                     4: 'PAD_MODE4',
                     5: 'PAD_MODE5'}
    cr_mode_dict = {0: 'CRMODE_NONE', 1: 'CRMODE_F32toF16_NONE',
                    2: 'CRMODE_F32toF16_RELU',
                    3: 'CRMODE_S32toF16_NONE', 4: 'CRMODE_F16toF32_NONE',
                    5: 'CRMODE_NONE_RELU'}
    pad_mode = []
    cr_mode = []
    intrin_name = "cceMove"
    if dst.scope == cce.scope_ubuf and src.scope == cce.scope_ubuf:
        intrin_name = "copy_ubuf_to_ubuf"
    elif dst.scope == cce.scope_cbuf and src.scope == 'global':
        intrin_name = "copy_gm_to_cbuf"
        sid = get_dma_sid("Sid_copy_gm_to_cbuf")
        pad_mode.append(pad_mode_dict[0])
        pad_mode_call = tvm.call_pure_intrin("int32",
                                             "tvm_cce_string_print",
                                             *pad_mode)
        ib_expr.emit(tvm.call_extern(
            dst.dtype, intrin_name,
            dst_ptr,  # dst buffer
            src_ptr,  # src buffer
            sid,
            nburst,
            burst,
            src_stride,
            dst_stride,
            pad_mode_call
        ))
        return ib_expr.get()
    elif dst.scope == cce.scope_ubuf and src.scope == cce.scope_cc:
        intrin_name = "copy_matrix_cc_to_ubuf"
        cr_mode.append(cr_mode_dict[0])
        cr_mode_call = tvm.call_pure_intrin("int32", "tvm_cce_string_print",
                                            *cr_mode)
        ib_expr.emit(tvm.call_extern(
            dst.dtype, intrin_name,
            dst_ptr,  # dst buffer
            src_ptr,  # src buffer
            sid,
            nburst,
            burst,
            src_stride,
            dst_stride,
            cr_mode_call
        ))
        return ib_expr.get()
    elif dst.scope == 'global' and src.scope.count(cce.scope_ubuf):
        intrin_name = "copy_ubuf_to_gm"
    elif dst.scope.count(cce.scope_ubuf) and src.scope == 'global':
        intrin_name = "copy_gm_to_ubuf"
        sid = get_dma_sid("Sid_copy_gm_to_ubuf")

    ib_expr.emit(tvm.call_extern(
        dst.dtype, intrin_name,
        dst_ptr,  # dst buffer
        src_ptr,  # src buffer
        sid,
        nburst,
        burst,
        src_stride,
        dst_stride
    ))


def is_const(expr):
    """
    check expr is const or not
    """
    if isinstance(expr, (tvm.expr.IntImm, tvm.expr.FloatImm, tvm.expr.UIntImm)):
        return True

    return False


def set_mask(length):
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
    mask1 = 2**max(length - 64, 0) - 1
    mask2 = 2**min(length, 64) - 1
    return mask1, mask2


def get_inplace_ids(stmt_in):
    """
        get inplace ids from stmt_in
    """
    inplace_ids = []

    def _post_order(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.expr.Select):
            inplace_ids.append(stmt_op.condition.equal.__self__.b.value)

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Select"])

    inplace_ids = list(inplace_ids)

    if not inplace_ids:
        return None

    return inplace_ids

# pylint: disable=too-many-locals
def get_inplace_var(stmt_in):
    """
        get inplace scalar var from stmt_in
    """
    inplace_var = []

    def _post_order(stmt_op):
        if isinstance(stmt_op, tvm.expr.Select):
            if is_const(stmt_op.true_value):
                inplace_var.append(stmt_op.true_value.value)
            elif isinstance(stmt_op.true_value, (tvm.expr.Add, tvm.expr.Sub)):
                if is_const(stmt_op.true_value.equal.__self__.a):
                    inplace_var.append(stmt_op.true_value.equal.__self__.a.value)
                elif is_const(stmt_op.true_value.equal.__self__.b):
                    inplace_var.append(stmt_op.true_value.equal.__self__.b.value)

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Select"])

    inplace_var = list(inplace_var)

    if not inplace_var:
        return None

    return inplace_var[0]


# pylint: disable=unused-argument
def get_inplace_buffer(stmt_in, rhs_buffer_count, intrinsic_cmd):
    """
    descripe:get buffer info list form the ir
    stmt_in:the emit_insn ir:
            for (i1.c, 0, 100) {
                abs_2.local.UB[((i0.c*100) + i1.c)] = select((
                    cast_2.local.UB[((i0.c*100) + i1.c)] >= float16(0)),
                    cast_2.local.UB[((i0.c*100) + i1.c)],
                    (cast_2.local.UB[((i0.c*100) + i1.c)]*-1.000000h))
             }
    return :the buffer abs_2[(i0.c*100)] , cast_2.local.UB[i0.c*100]
    """
    buf_info = {}
    shape = []
    # save the below of emit_insn axis var
    buf_info["loop_var"] = []

    def _get_shape(stmt_op):
        """
        get shape from stmt_op
        """
        if isinstance(stmt_in, tvm.stmt.For):
            buf_info["loop_var"].append(stmt_op.loop_var)
            shape.append(stmt_op.extent)
            if isinstance(stmt_op.body, tvm.stmt.For):
                _get_shape(stmt_op.body)

    # ?? reduction shape
    _get_shape(stmt_in)
    if not shape:
        shape.append(1)

    buf_info["shape"] = tuple(shape)

    loads = []
    store = []

    def _post_order(stmt_op):
        """
        post order for stmt_op
        """
        if isinstance(stmt_op, tvm.expr.Load):
            loads.append(stmt_op)
            return None
        if isinstance(stmt_op, tvm.stmt.Store):
            store.append(stmt_op)
            return None
        return None

    ##get the all load and store
    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Load", "Store"])

    # caculate the offset after make the below of emit_insn axis var = 0
    def _offset(itv_list, idx):
        """
        get itv offset
        """
        idx = tvm.ir_pass.Simplify(idx)
        if itv_list != [] and isinstance(idx, tvm.expr.IntImm) is False:
            var_dict = {}
            # make the emit_insn axis var = 0
            for itv in itv_list:
                var_dict[itv] = 0
            offset = tvm.ir_pass.Substitute(idx, var_dict)
            offset = tvm.ir_pass.Simplify(offset)
        else:
            offset = tvm.ir_pass.Simplify(idx)
        return offset.value

    lhs_loads = loads[0]
    rhs_loads = loads[1]

    offset_list = []
    for i in loads:
        if i.buffer_var.name == rhs_loads.buffer_var.name:
            offset_list.append(_offset(buf_info["loop_var"], i.index))
    offset_list.sort()
    if rhs_buffer_count != len(offset_list):
        raise RuntimeError("rhs_buffer_count != len(offset_list)")

    src_buf = []
    buf = tvm.decl_buffer(buf_info["shape"], lhs_loads.dtype,
                          name=lhs_loads.buffer_var.name,
                          data=lhs_loads.buffer_var,
                          offset_factor=1,
                          data_alignment=16,
                          scope=cce.scope_ubuf,
                          elem_offset=_offset(buf_info["loop_var"], lhs_loads.index))
    src_buf.append(buf)

    for idx in range(rhs_buffer_count):
        buf = tvm.decl_buffer(buf_info["shape"], rhs_loads.dtype,
                              name=rhs_loads.buffer_var.name,
                              data=rhs_loads.buffer_var,
                              offset_factor=1,
                              data_alignment=16,
                              scope=cce.scope_ubuf,
                              elem_offset=offset_list[idx])
        src_buf.append(buf)


    dest_buf = []
    for i in store:
        buf = tvm.decl_buffer(buf_info["shape"], i.value.dtype,
                              name=i.buffer_var.name,
                              data=i.buffer_var,
                              offset_factor=1,
                              data_alignment=16,
                              scope=cce.scope_ubuf,
                              elem_offset=_offset(buf_info["loop_var"],
                                                  i.index))
        dest_buf.append(buf)

    return src_buf, dest_buf


def get_segment_outer_axis_var(stmt_in):
    """
        get segment ids from stmt_in
    """
    outer_axis_var = []

    def _post_order(stmt_op):
        if isinstance(stmt_op, tvm.expr.Select):
            if isinstance(stmt_op.condition, tvm.expr.Or):
                outer_axis_var.append(stmt_op.condition.a.equal.__self__.a)
            else:
                outer_axis_var.append(stmt_op.condition.equal.__self__.a)

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Select"])

    outer_axis_var = list(set(outer_axis_var))

    if not outer_axis_var:
        return None

    return outer_axis_var[0]


def get_arg_outer_axis_var(stmt_in):
    """
        get arg split out aixs from stmt_in
        stmt_in: for (k0.inner, 0, 29090) {
          data_fp16_cast_red_temp.v0[0] = select((
            data_fp16_cast_red_temp.v1[0] >=
            data_fp16_cast.local.UB[k0.inner]),
            data_fp16_cast_red_temp.v0[0], ((k0.outer*29090) + k0.inner))

          data_fp16_cast_red_temp.v1[0] = select((
            data_fp16_cast_red_temp.v1[0] >=
                data_fp16_cast.local.UB[k0.inner]),
            data_fp16_cast_red_temp.v1[0], data_fp16_cast.local.UB[k0.inner])
        }
        return : k0.outer or None
    """
    outer_axis_var = []

    def _post_order(stmt_op):
        """
        post order for stmt_op
        """
        if isinstance(stmt_op, tvm.expr.Select):
            if isinstance(stmt_op.condition, (tvm.expr.LE, tvm.expr.GE)):
                select_false_expr = stmt_op.false_value
            elif isinstance(stmt_op.condition, tvm.expr.LT):
                select_false_expr = stmt_op.true_value
            else:
                raise RuntimeError("Unexpected select condition")
            if isinstance(select_false_expr, tvm.expr.Add) and isinstance(
                    select_false_expr.equal.__self__.a,
                    tvm.expr.Mul):
                mul_stmt_expr = select_false_expr.equal.__self__.a
                outer_axis_var.append(mul_stmt_expr.equal.__self__.a)

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Select"])

    outer_axis_var = list(set(outer_axis_var))

    if not outer_axis_var:
        return None

    return outer_axis_var[0]


def get_init_val(stmt_in):
    """
    get init_value from stmt_in
    for (i1.c, 0, 99991) {
        broadcast_0.local.UB(0, i1.c) =-65472.000000h
    }

    return -65472.000000h

    """
    store = []

    def _post_order(stmt_op):
        """
        post order for stmt_op
        """
        if isinstance(stmt_op, tvm.stmt.Store):
            store.append(stmt_op)

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Store"])

    init_val = []
    for node in store:
        if is_const(node.value):
            init_val.append(tvm.const(tvm.ir_pass.Simplify(node.value).value,
                                      node.value.dtype))
        elif isinstance(node.value, tvm.expr.Cast) and is_const(
                node.value.value):
            init_val.append(
                tvm.const(tvm.ir_pass.Simplify(node.value.value).value,
                          node.value.dtype))
        else:
            return False, []

    if not init_val:
        return init_val.append(tvm.const(0, dtype="int16"))

    return True, init_val


def get_buffer(stmt_in, need_unique=False, buffer_shape=None,
               need_origin_adress=False, target_type=None):
    """
    descripe:get buffer info list form the ir
    stmt_in:the emit_insn ir:
            for (i1.c, 0, 100) {
                abs_2.local.UB[((i0.c*100) + i1.c)] = select((
                    cast_2.local.UB[((i0.c*100) + i1.c)] >= float16(0)),
                    cast_2.local.UB[((i0.c*100) + i1.c)],
                    (cast_2.local.UB[((i0.c*100) + i1.c)]*-1.000000h))
             }
    need_unique: if need_unique = True,
                 we will del the same buffer in buffer list
    return :the buffer abs_2[(i0.c*100)] , cast_2.local.UB[i0.c*100]
    """
    buf_info = {}
    shape = []
    # save the below of emit_insn axis var
    buf_info["loop_var"] = []

    def _get_shape(stmt_op):
        """
        get shape from stmt_op
        """
        if isinstance(stmt_in, tvm.stmt.For):
            buf_info["loop_var"].append(stmt_op.loop_var)
            shape.append(stmt_op.extent)
            if isinstance(stmt_op.body, tvm.stmt.For):
                _get_shape(stmt_op.body)

    # ?? reduction shape
    _get_shape(stmt_in)
    if not shape:
        shape.append(1)

    buf_info["shape"] = tuple(shape)

    if buffer_shape is not None:
        buf_info["shape"] = buffer_shape

    loads = []
    store = []

    def _post_order(stmt_op):
        """
        post order for stmt_op
        """
        if isinstance(stmt_op, tvm.expr.Load):
            loads.append(stmt_op)
            return None
        if isinstance(stmt_op, tvm.stmt.Store):
            store.append(stmt_op)
            return None
        return None

    ##get the all load and store
    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Load", "Store"])

    # caculate the offset after make the below of emit_insn axis var = 0
    def _offset(itv_list, idx):
        """
        get itv offset
        """
        idx = tvm.ir_pass.Simplify(idx)
        if itv_list != [] and isinstance(idx, tvm.expr.IntImm) is False:
            var_dict = {}
            # make the emit_insn axis var = 0
            for itv in itv_list:
                var_dict[itv] = 0
            offset = tvm.ir_pass.Substitute(idx, var_dict)
            offset = tvm.ir_pass.Simplify(offset)
        else:
            offset = tvm.ir_pass.Simplify(idx)
        return offset

    src_buf = []
    for i in loads:
        if need_origin_adress:
            elem_offset = 0
        else:
            elem_offset = _offset(buf_info["loop_var"], i.index)
        temp_type = i.dtype
        if target_type is not None:
            temp_type = target_type
        buf = tvm.decl_buffer(buf_info["shape"], temp_type,
                              name=i.buffer_var.name,
                              data=i.buffer_var,
                              offset_factor=1,
                              data_alignment=16,
                              scope=cce.scope_ubuf,
                              elem_offset=elem_offset)
        src_buf.append(buf)

    dest_buf = []
    for i in store:
        if need_origin_adress:
            elem_offset = 0
        else:
            elem_offset = _offset(buf_info["loop_var"], i.index)
        temp_type = i.value.dtype
        if target_type is not None:
            temp_type = target_type
        buf = tvm.decl_buffer(buf_info["shape"], temp_type,
                              name=i.buffer_var.name,
                              data=i.buffer_var,
                              offset_factor=1,
                              data_alignment=16,
                              scope=cce.scope_ubuf,
                              elem_offset=elem_offset)
        dest_buf.append(buf)

    if need_unique:
        src_buf = get_unique_buffer(src_buf)
        dest_buf = get_unique_buffer(dest_buf)

    return src_buf, dest_buf


# pylint: disable=invalid-name
def get_elewise_single_vs_extern_args(stmt_in):
    '''
    descripe:
        stmt_in:for (i1.c, 0, 3) { //the emit_insn axis
            for (i2.c, 0, 2) {
              vmuls_t.local.UB[((i1.c*2) + i2.c)] =
              (vlog_t.local.UB[((i1.c*2) + i2.c)]*0.500000h)
        }
        this func get the extern_args 0.500000h
    :param stmt_in: the ir body
    :return: the extern args of elewise single vs mul,
             this case extern_args = [0.500000h]
    '''
    args = []

    def _post_order(stmt_op):
        """
        post order for stmt_op
        """
        # op : vmuls_t.local.UB[((i1.c*2) + i2.c)] =
        #     (vlog_t.local.UB[((i1.c*2) + i2.c)]*0.500000h)
        if isinstance(stmt_op, tvm.stmt.Store):
            if isinstance(stmt_op.value, tvm.expr.BinaryOpExpr) and is_const(
                    stmt_op.value.b):
                args.append(stmt_op.value.b)

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Store"])

    if args:
        return [args[0]]

    return None


# pylint: disable=too-many-locals, too-many-statements, invalid-name
def get_binary_scalar_axpy_extern_args(stmt_in):
    """"
    descripe:
        stmt_in:
            for(xxx)
            for(xxx){
            axpy_0.local.UB[((i0.c*256) + i1.c)] = ((0.000000f*data1.local.UB[((i0.c*256) + i1.c)])
                                                    + data2.local.UB[((i0.c*256) + i1.c)])
            }
            this func get the extern_args 0.000000f
    :param stmt_in: the ir body
    :return: the extern args of binary scalar axpy extern args, this case extern_args = [0.000000h]
    """

    args = []

    # pylint: disable=inconsistent-return-statements
    def _post_order(stmt_op):
        """
        post order for stmt_op
        """
        # scalar = 0, op :  axpy_3.local.UB[((i0.c*256) + i1.c)] =
        # data2.local.UB[((i0.c*256) + i1.c)]
        # scalar > 0, op :  axpy_0.local.UB[((i0.c*1024) + i1.c)] =
        #  ((data1.local.UB[((i0.c*1024) + i1.c)]*5.000000h) +
        #  data2.local.UB[((i0.c*1024) + i1.c)])
        # or xpy_11.local.UB[((i0.c * 1024) + i1.c)] =
        #  ((5.000000h * data1.local.UB[((i0.c * 1024) + i1.c)]) +
        #  data2.local.UB[((i0.c * 1024) + i1.c)])
        # scalar < 0, op :  axpy_2.local.UB[((i0.c*128) + i1.c)] =
        #  (data2.local.UB[((i0.c*128) + i1.c)] -
        #  (data1.local.UB[((i0.c*128) + i1.c)]*5.000000h))

        if isinstance(stmt_op, tvm.stmt.Store):

            if isinstance(stmt_op.value, tvm.expr.Load):  # case: scalar = 0
                return None

            if isinstance(stmt_op.value, tvm.expr.Add):  # case: scalar > 0
                if stmt_op.value.a is not None:
                    if is_const(stmt_op.value.a.a):
                        args.append(stmt_op.value.a.a)
                    elif is_const(stmt_op.value.a.b):
                        args.append(stmt_op.value.a.b)

            if isinstance(stmt_op.value, tvm.expr.Sub):  # case: scalar < 0
                if stmt_op.value.b is not None and is_const(stmt_op.value.b.b):
                    args.append(stmt_op.value.b.b)
            return None

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Store"])

    if args:
        return [args[0]]

    return None


def get_opshape_tilesize(stmt):
    """
    descripe:
        The shape is determined by the coefficient of all the loop variable in
         the offset expr of src
        and dst.
        The tile size is determined by the extent of the loop in the stmt.
        stmt_in:
            for (j.c, 0, 12) {
                for (k.c, 0, 17) {
                    for (l.c, 0, 16) {
                        C[((((j.c*17) + k.c)*16) + l.c)] = (A[(((((j.c*17) +
                         k.c)*16) + l.c)] + B[l.c])
                    }
                }
            }
        we will get the shape as [17*16, 16, 1], and get the tile size as
         [16, 17, 12]. To be noted that,
        when get the shape (i.e., the coefficient of the loop variable
         in the offset expr), we will ignore
        the variable which is not the loop variable in the stmt
         (even the loop outside the stmt).

    :param stmt: the emit_insn ir
    :return: the length of op
    """
    length = {}
    loads = []
    stores = []
    vector_tile_size = []
    buffer_var = []

    def _post_order_for(stmt_op):
        """
        post order for stmt_op
        """
        if isinstance(stmt_op, tvm.stmt.For):
            length[stmt_op.loop_var] = stmt_op.extent.value
            vector_tile_size.append(stmt_op.extent.value)
            buffer_var.append(stmt_op.loop_var)
            return None
        if isinstance(stmt_op, tvm.expr.Load):
            loads.append(stmt_op)
            return None
        if isinstance(stmt_op, tvm.stmt.Store):
            stores.append(stmt_op)
            return None
        return None

    _ = tvm.ir_pass.IRTransform(stmt, None, _post_order_for,
                                ["For", "Load", "Store"])

    final_coef = []
    for var in buffer_var:
        coef = []

        def _get_second_size(stmt_op):
            """
            get stmt_op second size
            """
            is_add = []

            # pylint: disable=cell-var-from-loop, too-many-return-statements
            def _post_order_var(stmt_op):
                """
                post order stmt_op
                """
                if isinstance(stmt_op, tvm.expr.Var):
                    if stmt_op == var:
                        return 1
                    return 0
                if isinstance(stmt_op, tvm.expr.Mul):
                    is_add.append(False)
                    mul_res = _post_order_var(stmt_op.a)*_post_order_var(stmt_op.b)
                    is_add.pop()
                    return mul_res

                if isinstance(stmt_op, tvm.expr.Add):
                    is_add.append(True)
                    add_res = _post_order_var(stmt_op.a) + _post_order_var(stmt_op.b)
                    is_add.pop()
                    return add_res

                if isinstance(stmt_op, tvm.expr.Sub):
                    is_add.append(True)
                    sub_res = _post_order_var(stmt_op.a) - _post_order_var(stmt_op.b)
                    is_add.pop()
                    return sub_res

                if isinstance(stmt_op, tvm.expr.IntImm):
                    if not is_add or is_add[len(is_add) - 1]:
                        return 0
                    return stmt_op.value
                return None

            return _post_order_var(stmt_op)

        for sec in stores:
            coef.append(_get_second_size(sec.index))
        for sec in loads:
            coef.append(_get_second_size(sec.index))

        _is_same = False
        coef_in_store = coef[0]
        for index in range(1, len(coef)):
            ele = coef[index]
            if ele == coef_in_store:
                _is_same = True
                break

        if not _is_same:
            final_coef = coef

    sizes = []

    def _get_size(stmt_op):
        """
        get stmt_op size
        """
        size_list = []

        def _post_order_var(stmt_op):
            """
            post order stmt_op
            """
            if isinstance(stmt_op, tvm.expr.Var):
                if stmt_op in length.keys():
                    size_list.append(length[stmt_op])
                else:
                    size_list.append(1)
            if isinstance(stmt_op, tvm.expr.Mul):
                _post_order_var(stmt_op.a)
                _post_order_var(stmt_op.b)
            if isinstance(stmt_op, tvm.expr.Add):
                _post_order_var(stmt_op.a)
                _post_order_var(stmt_op.b)
            if isinstance(stmt_op, tvm.expr.Sub):
                _post_order_var(stmt_op.a)
                _post_order_var(stmt_op.b)

        _post_order_var(stmt_op)

        size = 1
        for element in size_list:
            size = size*element

        return size

    for node in stores:
        sizes.append(_get_size(node.index))

    for node in loads:
        sizes.append(_get_size(node.index))

    return sizes, vector_tile_size, final_coef


def get_op_lenth(stmt):
    """
    descripe:
        stmt_in:
        for (i1.c, 0, 3) { //the emit_insn axis
             for (i2.c, 0, 2) {
            }
        we will get the op length is 3*2 = 6
    :param stmt: the emit_insn ir
    :return: the length of op
    """
    length = [1]

    def _post_order(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.stmt.For):
            length[0] = length[0]*stmt_op.extent.value

    stmt = tvm.ir_pass.IRTransform(stmt, None, _post_order, ["For"])
    return length[0]


def get_align_oplength(stmt):
    """
    descripe:
        stmt_in:
        for (i1.c, 0, 3) { //the emit_insn axis
             for (i2.c, 0, 2) {
            }
        we will get the op length is 3*2 = 6
    :param stmt: the emit_insn ir
    :return: the length of op
    """
    length = [1]

    def _post_order(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.stmt.For):
            if isinstance(stmt_op.body, tvm.stmt.For):
                length[0] = length[0]*stmt_op.extent.value
            else:
                align_length = (stmt_op.extent.value + 15) // 16*16
                length[0] = length[0]*align_length

    stmt = tvm.ir_pass.IRTransform(stmt, None, _post_order, ["For"])
    return length[0]


def get_pad_info(stmt_in):
    """
    get pad info
    """
    pinfo = {}
    pinfo["pad"] = None
    pinfo["iv"] = None

    def _post_order(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.stmt.AttrStmt) and \
                stmt_op.attr_key == "pragma_pad":
            pinfo["pad"] = stmt_op.value
            if isinstance(stmt_op.node, tvm.schedule.IterVar):
                pinfo["iv"] = stmt_op.node
            return stmt_op.body
        return None

    stmt = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["AttrStmt"])
    return pinfo["iv"], pinfo["pad"], stmt


def get_bits_of(dtype):
    """
    calculate bits of dtype of TVM
    Parameters
    ----------
    dtype : string
        dtype of TVM

    Returns
    -------
    ret : int
        bit length of dtype.
    """
    index = 0
    for i in dtype:
        if i.isdigit():
            break
        index += 1
    return int(dtype[index:])


def get_align_of(dtype):
    """
    calculate the unified buffer data alignment
    Parameters
    ----------
    dtype : string
        dtype of TVM

    Returns
    -------
    ret : int
        the num of data alignment.
    """
    return 32*8 // get_bits_of(dtype)


def get_data_alignment(dtype):
    """
    calculate the unified buffer data alignment
    Parameters
    ----------
    dtype : string
        dtype of TVM

    Returns
    -------
    ret : int
        the num of data alignment.
    """
    return 32*8 // get_bits_of(dtype)


# pylint: disable=too-many-arguments
def concat_args(src_buffers, dst_buffers, repeat_src_offset, repeat_dst_offset,
                repeat_times,
                extern_args, args):
    """
    get concat args
    """
    res_args = []
    for i in dst_buffers:
        res_args.append(i.access_ptr("wr", offset=repeat_dst_offset))
    for i in src_buffers:
        res_args.append(i.access_ptr("r", offset=repeat_src_offset))
    if not isinstance(extern_args, type(None)):
        res_args += extern_args
    res_args.append(repeat_times)
    res_args += args
    return res_args


def dtype2ccetype(dtype):
    """
    change dtype of TVM to dtype of cce
    Parameters
    ----------
    dtype : string
        dtype of TVM

    Returns
    -------
    ret : string
        dtype of cce.
    """
    map_dict = {"float": "f",
                "int": "s",
                "uint": "u"}
    index = 0
    for i in dtype:
        if i.isdigit():
            break
        index += 1
    prefix = dtype[:index]
    bit_len = dtype[index:]
    return map_dict[prefix] + bit_len


def set_mask_argmax(length):
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
    length_l = min(length, 32)
    length_h = max(length - 32, 0)
    mask1 = ''.join(["01" for _ in range(length_h)])
    mask1 = mask1 if mask1 else "0"
    mask2 = ''.join(["01" for _ in range(length_l)])
    mask2 = mask2 if mask2 else "0"
    mask1 = int(mask1, 2)
    mask2 = int(mask2, 2)
    return mask1, mask2


def get_argnlst_outaxis(stmt_in):
    """
        get arg split out aixs from stmt_in
        stmt_in:
        for (ax3, 0, 4682) {
          data_fp16_cast_red_temp.v0[ax3] = select(
            (data_fp16_cast_red_temp.v1[ax3] >= data_fp16_cast.local.UB[ax3]),
             data_fp16_cast_red_temp.v0[ax3],
             k0)
          data_fp16_cast_red_temp.v1[ax3] = select(
            (data_fp16_cast_red_temp.v1[ax3] >= data_fp16_cast.local.UB[ax3]),
            data_fp16_cast_red_temp.v1[ax3],
            data_fp16_cast.local.UB[ax3])
        }
        return : k0 or None
    """
    var = []

    def _post_order(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.expr.Select):
            if isinstance(stmt_op.condition, (tvm.expr.GE, tvm.expr.LE)):
                var.append(stmt_op.false_value)
            elif isinstance(stmt_op.condition, tvm.expr.LT):
                var.append(stmt_op.true_value)
            else:
                raise RuntimeError("Unexpected select condition")

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Select"])

    if var and (isinstance(var[0], tvm.expr.Var) or is_const(var[0])):
        return var[0]

    return None


def get_scalar_args(stmt_in):
    """
    get scalar agrs
    """
    var = []

    def _post_order(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.expr.Select):
            if (not isinstance(stmt_op.condition.a, tvm.expr.Load) and isinstance(
                    stmt_op.condition.b, tvm.expr.Load)):
                var.append(stmt_op.condition.a)
            elif (not isinstance(stmt_op.condition.b, tvm.expr.Load) and isinstance(
                    stmt_op.condition.a, tvm.expr.Load)):
                var.append(stmt_op.condition.b)
            else:
                raise RuntimeError(
                    "either a or b must be tvm.expr.Load, other must be const")

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Select"])

    if var:
        return [var[0]]

    return None


def apply_for_new_alloc(ib_expr, dtype, shape, scope=cce.scope_ubuf, name='tmp_buf'):
    """
    alloc buffer
    """
    buf_var = ib_expr.allocate(dtype, shape, name=name, scope=scope)
    tmp_buffer = tvm.decl_buffer(shape,
                                 buf_var.dtype,
                                 name=name,
                                 scope=scope,
                                 data=buf_var)
    return tmp_buffer


def get_src_dst_type(stmt_in):
    """
    :param stmt: The emit_insn ir
    :return: the src and dst_dtype
    """
    src_type = "float16"
    dst_type = "float16"
    loads = []
    store = []

    def _post_order(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.expr.Load):
            loads.append(stmt_op)
            return None
        if isinstance(stmt_op, tvm.stmt.Store):
            store.append(stmt_op)
            return None
        return None

    ##get the all load and store
    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Load", "Store"])

    if loads:
        src_type = loads[-1].dtype

    if store:
        dst_type = store[-1].value.dtype

    return src_type, dst_type


def get_init_op(stmt_in):
    """
    reduce_0.local.UB[i0.c] = -65504.000000h
    for (k1, 0, 1024) {
        reduce_0.local.UB[i0.c] = max(reduce_0.local.UB[i0.c],
         abs_0.local.UB[(((i0.outer*51200) + (i0.c*1024)) + k1)])
        }
    )
    :param op:
    :return: reduce_0.local.UB[i0.c] = -65504.000000h
    """
    init_op = []

    def _post_order_for(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.stmt.For) and \
                isinstance(stmt_op.body, tvm.stmt.Store) and \
                is_const(stmt_op.body.value):
            init_op.append(stmt_op)

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order_for, ["For"])

    if init_op:
        return init_op[0]

    # pylint: disable=inconsistent-return-statements
    def _post_order_store(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.stmt.Store) and is_const(stmt_op.value):
            init_op.append(stmt_op)
            return None

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order_store, ["Store"])

    if init_op:
        return init_op[0]

    return None


def get_reduce_op(stmt_in):
    """
    reduce_0.local.UB[i0.c] = -65504.000000h
    for (k1, 0, 1024) {
        reduce_0.local.UB[i0.c] = max(reduce_0.local.UB[i0.c],
         abs_0.local.UB[(((i0.outer*51200) + (i0.c*1024)) + k1)])
        }
    )
    :param op:
    :return: reduce_op
    """
    reduce_op = []

    def _post_order_for(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.stmt.For):
            reduce_op.append(stmt_op)

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order_for, ["For"])

    if reduce_op:
        return reduce_op[-1]

    def _post_order_store(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.stmt.Store) and not is_const(stmt_op.value):
            reduce_op.append(stmt_op)

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order_store, ["Store"])

    if reduce_op:
        return reduce_op[-1]

    return None


def get_cmp_args(stmt_in):
    """
    :param stmt_in:
            bit mode :
            for (i0.c, 0, 16) {
              for (i1.c, 0, 512) {
                output.local.UB(i0.c, i1.c) =(uint8)0
                for (k, 0, 8) {
                  output.local.UB(i0.c, i1.c) =bitwise_and(output.local.UB(i0.c,
                   i1.c), uint8((input1.local.UB(i0.c, ((i1.c*8) + k)) >=
                   input2.local.UB(i0.c, ((i1.c*8) + k)))))
                }
              }
            }
          }
         bool mode:
        for (i0.c, 0, 4) {
                for (i1.c, 0, 4096) {
                  cmp_0.local.UB((i0.c + (i0.outer*4)), i1.c) =(input1.local.UB
                  ((i0.c + (i0.outer*4)), i1.c) >=
                   input2.local.UB((i0.c + (i0.outer*4)), i1.c))
                }
              }
            }

    :return: cmp_type:gt,lt..., mode:bool,bit
    """
    args = ['lt', 'bool', None, None]

    # pylint: disable=too-many-branches
    def _post_order(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.stmt.Store) and not is_const(
                stmt_op.value) and isinstance(stmt_op.value, tvm.expr.Call):
            args[1] = 'bit'
            if is_const(stmt_op.value.args[1].value.a):
                args[2] = stmt_op.value.args[1].value.a.value
            if is_const(stmt_op.value.args[1].value.b):
                args[3] = stmt_op.value.args[1].value.b.value
        elif isinstance(stmt_op, tvm.stmt.Store) and not is_const(stmt_op.value):
            if isinstance(stmt_op.value, (
                    tvm.expr.LT, tvm.expr.GT, tvm.expr.GE, tvm.expr.LE,
                    tvm.expr.EQ, tvm.expr.NE)):
                if is_const(stmt_op.value.a):
                    args[2] = stmt_op.value.a.value
                if is_const(stmt_op.value.b):
                    args[3] = stmt_op.value.b.value
            elif is_const(stmt_op.value.value.b):
                args[3] = stmt_op.value.value.b.value
            elif is_const(stmt_op.value.value.a):
                args[2] = stmt_op.value.value.a.value
        elif isinstance(stmt_op, tvm.expr.LT):
            args[0] = 'lt'
        elif isinstance(stmt_op, tvm.expr.GT):
            args[0] = 'gt'
        elif isinstance(stmt_op, tvm.expr.GE):
            args[0] = 'ge'
        elif isinstance(stmt_op, tvm.expr.LE):
            args[0] = 'le'
        elif isinstance(stmt_op, tvm.expr.EQ):
            args[0] = 'eq'
        elif isinstance(stmt_op, tvm.expr.NE):
            args[0] = 'ne'

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order,
                                ["Store", "LT", "GT", "GE", "LE", "EQ", "NE",
                                 "Div", "Mod"])
    return args


def del_ins_reuse_out(ins, out):
    """
    A = f(A, B, C)
    :param ins: [A,B,C]
    :param out: [A]
    :return: [B,C]
    """
    in_res = []
    for single_ins in ins:
        if not (tvm.ir_pass.Equal(single_ins.data, out.data) and \
                tvm.ir_pass.Equal(single_ins.elem_offset, out.elem_offset)):
            in_res.append(single_ins)

    return in_res


# pylint: disable=unused-argument
def get_cmp_bit_buffer(stmt_in, need_unique=False):
    """
    descripe:get buffer info list form the ir in cmp with mode is 'bit'
    stmt_in:the emit_insn ir:
            for (i0.c, 0, 16) {
              for (i1.c, 0, 512) {
                output.local.UB(i0.c, i1.c) =(uint8)0
                for (k, 0, 8) {
                  output.local.UB(i0.c, i1.c) =bitwise_and(output.local.UB
                  (i0.c, i1.c), uint8((input1.local.UB(i0.c, ((i1.c*8) + k)) >
                   input2.local.UB(i0.c, ((i1.c*8) + k)))))
                }
              }
            }

    return :the buffer output, input1, input2, shape is (16, 512*8)
    """
    buf_info = {}
    shape_src = []
    # save the below of emit_insn axis var
    buf_info["loop_var"] = []

    def _get_shape(stmt_op):
        """
        get stmt_op shape
        """
        if isinstance(stmt_in, tvm.stmt.For):
            shape_src.append(stmt_op.extent)
            if isinstance(stmt_op.body, tvm.stmt.For):
                _get_shape(stmt_op.body)

    _get_shape(stmt_in)
    if not shape_src:
        shape_src.append(1)

    shape_dst = copy.copy(shape_src)
    shape_src[-1] = shape_src[-1]*8

    buf_info["shape_src"] = tuple(shape_src)
    buf_info["shape_dst"] = tuple(shape_dst)

    loads = []
    store = []

    def _post_order(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.expr.Load):
            loads.append(stmt_op)
            return None
        if isinstance(stmt_op, tvm.stmt.Store):
            store.append(stmt_op)
            return None
        if isinstance(stmt_op, tvm.stmt.For):
            buf_info["loop_var"].append(stmt_op.loop_var)
            return None
        return None

    ##get the all load and store
    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order,
                                ["Load", "Store", "For"])

    # caculate the offset after make the below of emit_insn axis var = 0
    def _offset(itv_list, idx):
        """
        get itv offset
        """
        idx = tvm.ir_pass.Simplify(idx)
        if itv_list != [] and isinstance(idx, tvm.expr.IntImm) is False:
            var_dict = {}
            # make the emit_insn axis var = 0
            for itv in itv_list:
                var_dict[itv] = 0
            offset = tvm.ir_pass.Substitute(idx, var_dict)
            offset = tvm.ir_pass.Simplify(offset)
        else:
            offset = tvm.ir_pass.Simplify(idx)
        return offset

    src_buf = []
    for i in loads:
        buf = tvm.decl_buffer(buf_info["shape_src"], i.dtype,
                              name=i.buffer_var.name,
                              data=i.buffer_var,
                              offset_factor=1,
                              data_alignment=16,
                              scope=cce.scope_ubuf,
                              elem_offset=_offset(buf_info["loop_var"],
                                                  i.index))
        src_buf.append(buf)

    dest_buf = []
    for i in store:
        buf = tvm.decl_buffer(buf_info["shape_dst"], i.value.dtype,
                              name=i.buffer_var.name,
                              data=i.buffer_var,
                              offset_factor=1,
                              data_alignment=16,
                              scope=cce.scope_ubuf,
                              elem_offset=_offset(buf_info["loop_var"],
                                                  i.index))
        dest_buf.append(buf)

    src_buf = get_unique_buffer(src_buf)
    dest_buf = get_unique_buffer(dest_buf)

    return src_buf, dest_buf


def get_unique_buffer(buf_list):
    """
    # get unique buffer, if buffer1.data == buffer2.data and
     buffer1.elemm_offset == buffer2.element_offset
    # then buffer1 == buffer2
    :param buf_list: the src buffer list
    :return: the unique buffer in list
    """
    buf_list_unique = []

    for buffer1 in buf_list:
        is_buf_exit = False
        for buffer2 in buf_list_unique:
            if tvm.ir_pass.Equal(buffer1.data,
                                 buffer2.data) and tvm.ir_pass.Equal(
                                     buffer1.elem_offset, buffer2.elem_offset):
                is_buf_exit = True
                break

        if not is_buf_exit:
            buf_list_unique.append(buffer1)

    return buf_list_unique


def get_sel_type(stmt_in):
    """
    :param op: stmt
    :return: tensor_to_tensor tensor_to_scalar scalar_to_tensor
     scalar_to_scalar
    """
    sel_type = ["tensor_to_tensor"]
    scalar_value = []

    def _post_order(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.expr.Select):
            if not is_const(stmt_op.true_value) and not is_const(stmt_op.false_value):
                sel_type[0] = "tensor_to_tensor"
            elif not is_const(stmt_op.true_value) and is_const(stmt_op.false_value):
                sel_type[0] = "tensor_to_scalar"
                scalar_value.append(stmt_op.false_value)
            elif is_const(stmt_op.true_value) and not is_const(stmt_op.false_value):
                sel_type[0] = "scalar_to_tensor"
                scalar_value.append(stmt_op.true_value)
            else:
                sel_type[0] = "scalar_to_scalar"
                scalar_value.append(stmt_op.true_value)
                scalar_value.append(stmt_op.false_value)

    ##get the all load and store
    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Select"])

    return sel_type[0], scalar_value


def reverse_vsel_condition(ori_cond):
    """
    transport condition
    """
    if ori_cond == 'lt':
        return 'ge'
    if ori_cond == 'le':
        return 'gt'
    if ori_cond == 'gt':
        return 'le'
    if ori_cond == 'ge':
        return 'lt'
    return ori_cond


def get_cond_args(stmt_in):
    """
    :param stmt_in:
        ('op', for (i0.c, 0, 8332) {
          cond_0.local.UB[i0.c] = select((dataA.local.UB[i0.c] >= 2.000000f),
           dataA.local.UB[i0.c], 0.000000f)
        }
    :return: cmp_type:gt,lt..., threshold : 2.0, bias : 0.0
    """
    args = ['lt', '2.0', '0.0']

    def _post_order(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.expr.Select):
            if is_const(stmt_op.condition.b):
                args[1] = stmt_op.condition.b.value
            elif is_const(stmt_op.condition.a):
                args[1] = stmt_op.condition.a.value

            if is_const(stmt_op.false_value):
                args[2] = stmt_op.false_value.value
            elif is_const(stmt_op.true_value):
                args[2] = stmt_op.true_value.value
            # bug fix for mutate after simplify
            if (not isinstance(stmt_op.condition.a, tvm.expr.Load) or
                    not isinstance(stmt_op.true_value, tvm.expr.Load) or
                    stmt_op.condition.a.buffer_var.name !=
                    stmt_op.true_value.buffer_var.name):
                args[0] = reverse_vsel_condition(args[0])
        elif isinstance(stmt_op, tvm.expr.LT):
            args[0] = 'lt'
        elif isinstance(stmt_op, tvm.expr.GT):
            args[0] = 'gt'
        elif isinstance(stmt_op, tvm.expr.GE):
            args[0] = 'ge'
        elif isinstance(stmt_op, tvm.expr.LE):
            args[0] = 'le'
        elif isinstance(stmt_op, tvm.expr.EQ):
            args[0] = 'eq'
        elif isinstance(stmt_op, tvm.expr.NE):
            args[0] = 'ne'

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order,
                                ["Select", "LT", "GT", "GE", "LE", "EQ", "NE"])

    return args


def get_cmp_sel_args(stmt_in):
    """
    get cmp sel args
    """
    args = ['lt', None, None, None, None]

    def _post_order(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.expr.Select):
            if is_const(stmt_op.condition.a):
                args[1] = stmt_op.condition.a.value

            if is_const(stmt_op.condition.b):
                args[2] = stmt_op.condition.b.value

            if is_const(stmt_op.true_value):
                args[3] = stmt_op.true_value.value

            if is_const(stmt_op.false_value):
                args[4] = stmt_op.false_value.value
        elif isinstance(stmt_op, tvm.expr.LT):
            args[0] = 'lt'
        elif isinstance(stmt_op, tvm.expr.GT):
            args[0] = 'gt'
        elif isinstance(stmt_op, tvm.expr.GE):
            args[0] = 'ge'
        elif isinstance(stmt_op, tvm.expr.LE):
            args[0] = 'le'
        elif isinstance(stmt_op, tvm.expr.EQ):
            args[0] = 'eq'
        elif isinstance(stmt_op, tvm.expr.NE):
            args[0] = 'ne'

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order,
                                ["Select", "LT", "GT", "GE", "LE", "EQ", "NE"])
    return args


def get_logic_args(stmt_in):
    """
    :param stmt_in:
        for (i0.c, 0, 11110) {
          logic_0.local.UB[i0.c] = int8(bitwise_or(uint1(dataA.local.UB[i0.c]),
           uint1(dataB.local.UB[i0.c])))
        }
    :return: LogicArgs : and/or/not
    """
    args = ['and']

    def _post_order(stmt_op):
        """
        post order stmt_op
        """
        if isinstance(stmt_op, tvm.expr.Call):
            if stmt_op.name == "bitwise_or":
                args[0] = 'or'
            elif stmt_op.name == "bitwise_and":
                args[0] = 'and'
            elif stmt_op.name == "bitwise_not":
                args[0] = 'not'

    _ = tvm.ir_pass.IRTransform(stmt_in, None, _post_order, ["Call"])

    return args


def get_const(expr):
    """get const value from TVM expression

    Parameters
    ----------
    expr : tvm.expr
        tvm expression

    Returns
    -------
    value : int
      expr value

    """
    if isinstance(expr, int):
        return expr

    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        expr = tvm.ir_pass.Simplify(expr)

    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        return expr.astype(UINT64_T)

    return expr.value


def get_input_shape_from_stmt(input_shape, stmt_op):
    """get input shapes"""

    original_shape = []
    loop_var_dict = {}

    def store_ori(_var):
        if not isinstance(_var, (tvm.expr.Var, tvm.expr.IntImm)):
            store_ori(_var.a)
            store_ori(_var.b)
            return
        if isinstance(_var, tvm.expr.Var):
            original_shape.append(loop_var_dict[str(_var.name)])
            return
        if isinstance(_var, tvm.expr.IntImm):
            return
        raise RuntimeError("Backend Error: Received unexpected statement: "
                           + str(type(_var)))

    def interpret_statement(stmt):
        if isinstance(stmt, tvm.expr.Load):
            store_ori(stmt.index)
            input_shape.append(original_shape[:])
            original_shape.clear()
            return
        if isinstance(stmt, tvm.stmt.For):
            loop_var_dict[str(stmt.loop_var)] = int(stmt.extent)
            return
        raise RuntimeError("Backend Error: Received unexpected statement: "
                           + str(type(stmt)))

    tvm.ir_pass.IRTransform(stmt_op, None, interpret_statement, ["For"])
    tvm.ir_pass.IRTransform(stmt_op, None, interpret_statement, ["Load"])
