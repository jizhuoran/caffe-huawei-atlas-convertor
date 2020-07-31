#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Additional IR Pass for CCE
"""
from __future__ import absolute_import as _abs

from te import tvm

from . import cce_params as param
from . import cce_util


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


def _check_compact(buf):
    ndim = len(buf.shape)
    size = tvm.const(1, buf.shape[0].dtype)
    for i in reversed(range(ndim)):
        if not equal_const_int(size - buf.strides[i], 0):
            raise RuntimeError(
                "Cannot prove compact: shape=%s, strides=%s" % (
                    buf.shape, buf.strides))
        size = size * buf.shape[i]


def _fold_buffer_dim(buf, scope, elem_block):
    ndim = len(buf.shape)
    x_size = 1
    base = 0
    for i in range(1, ndim + 1):
        x_size = x_size * buf.shape[ndim - i]
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
            if not equal_const_int(x_size * x_stride - buf.strides[k], 0):
                break
            x_size = x_size * buf.shape[k]
            next_base = i + 1
        shape.append(tvm.ir_pass.Simplify(x_size))
        strides.append(x_stride)
        assert next_base != base
        base = next_base

    strides = list(reversed(strides))
    shape = list(reversed(shape))
    return shape, strides

def inject_dma_intrin(stmt_in, mock=False, **kwargs):
    """Pass to inject DMA copy intrinsics.

    Parameters
    ----------
    stmt_in : Stmt
        Input statement

    Returns
    -------
    stmt_out : Stmt
        Transformed statement
    """
    # pylint: disable=too-many-statements
    def _get_2d_pattern_cce(src_buf, elem_width, elem_bytes, dst_buf,
                            allow_fold):
        elem_block = elem_bytes * 8 // elem_width
        if elem_width == 16:
            dtype = "float%d" % elem_width
        elif elem_width == 8:
            dtype = "int%d" % elem_width
        else:
            raise RuntimeError(
                "Expect element width to be 16 or 8 instead of %s" %
                (elem_width))

        if src_buf.dtype != dtype:
            raise RuntimeError("Expect buffer type to be %s instead of %s" %
                               (dtype, src_buf.dtype))

        shape, strides = list(src_buf.shape), list(src_buf.strides)
        if not equal_const_int(src_buf.elem_offset % elem_block, 0):
            raise RuntimeError(
                "scope %s need to have block=%d" % (dst_buf.scope, elem_block))
        if allow_fold:
            shape, strides = _fold_buffer_dim(src_buf, dst_buf.scope,
                                              elem_block)

        def raise_error():
            """Internal function to raise error """
            raise RuntimeError(
                ("Scope[%s]: cannot detect 2d pattern with elem_block=%d:" +
                 " shape=%s, strides=%s") % (
                     dst_buf.scope, elem_block, shape, strides))

        ndim = len(shape)
        if not equal_const_int(strides[-1] - 1, 0):
            raise_error()
        if not equal_const_int(shape[-1] - elem_block, 0):
            raise_error()
        if ndim == 1:
            index = 0  # src_buf.elem_offset / elem_block
            src_stride = 1
            repeat = 1
            dst_offset = 0
            return index, src_stride, repeat, dst_offset

        if not equal_const_int(strides[-2] - elem_block, 0):
            raise_error()

        if ndim == 2:
            index = 0  # src_buf.elem_offset / elem_block
            src_stride = strides[-2] // elem_block
            repeat = shape[-2]
            dst_offset = dst_buf.elem_offset // elem_block * elem_bytes
            return index, src_stride, repeat, dst_offset
        raise_error()

    # pylint: disable=too-many-locals
    def _get_mov_pattern(src_buf, elem_width, elem_bytes, dst_buf, allow_fold):
        elem_block = elem_bytes * 8 // elem_width

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
                ("Scope[%s]: cannot detect mov pattern with elem_block=%d:" +
                 " shape=%s, strides=%s") % (
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

    def _key_word_(name):
        tmp = name.split(".")
        if tmp[0].lower() == "global":
            return "OUT"
        if tmp[1].count('UB'):
            return "UB"
        return tmp[1]

    def get_real_src_dst(src, dst):
        def ptr_determinate(buffer_local, label):
            if _key_word_(buffer_local.scope) == "OUT":
                return buffer_local.access_ptr(label)  # buffer.data
            return buffer_local.access_ptr(label)

        return (ptr_determinate(src, "r"), ptr_determinate(dst, "w"))

    def get_type_bits(data_type):
        tmp = ''
        if data_type.lower() == "bool":
            return 1
        for i in data_type[::-1]:
            if i.isdigit():
                tmp += i
            else:
                break
        return int(tmp[::-1])


    # pylint: disable=too-few-public-methods
    def dma_dependency_scope(src, dst, i_b):
        class StaticDmaList:
            """
            StaticDmaList
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

        i_b.scope_attr(param.CCE_AXIS, "coproc_scope", pipe_line)

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        pad_value = pad_value

        elem_width = get_type_bits(dst.dtype)

        if dst.scope == param.scope_cb or dst.scope == param.scope_ca:
            # Load2D
            if pad_before or pad_after:
                raise RuntimeError("Do not support copy into DRAM with pad")

            if (src.scope != param.scope_cbuf) and (src.scope != 'global'):
                raise RuntimeError("Do not support copy %s->dram" % (src.scope))

            elem_bytes = param.BLOCK_IN * param.BLOCK_OUT * elem_width // 8
            if elem_width == 8:
                elem_bytes = elem_bytes * 2

            allow_fold = True
            transpose = False
            transpose_call = tvm.call_pure_intrin("int32",
                                                  "tvm_cce_string_print",
                                                  'true' if transpose else 'false')
            _check_compact(dst)
            index, src_stride, repeat_time, _ = _get_2d_pattern_cce(
                src, elem_width, elem_bytes, dst, allow_fold=allow_fold)

            i_b = tvm.ir_builder.create()
            dma_dependency_scope(src, dst, i_b)
            src_ptr, dst_ptr = get_real_src_dst(src, dst)
            sid = 0
            intrin_name = "cceLoad2D"
            if dst.scope == param.scope_ca and src.scope == param.scope_cbuf:
                intrin_name = "load_cbuf_to_ca"
            elif dst.scope == param.scope_cb and src.scope == param.scope_cbuf:
                intrin_name = "load_cbuf_to_cb"
            i_b.emit(tvm.call_extern(
                dst.dtype, intrin_name,
                dst_ptr,  # dst buffer
                src_ptr,  # src buffer
                index,  # index
                repeat_time,  # repeat times
                src_stride,  # src_stride
                sid,
                transpose_call  # transpose
            ))
            return i_b.get()

        # other DMA
        if pad_before or pad_after:
            raise RuntimeError("Do not support copy into DRAM with pad")
        if src.scope == param.scope_cc or dst.scope == param.scope_cc:
            elem_bytes = param.BLOCK_IN * param.BLOCK_OUT * elem_width // 8
        else:
            elem_bytes = param.GLB_ELEM_BYTES

        allow_fold = True

        _, _, nburst, burst, src_stride, dst_stride = _get_mov_pattern(
            src, elem_width, elem_bytes, dst, allow_fold=allow_fold)
        i_b = tvm.ir_builder.create()
        dma_dependency_scope(src, dst, i_b)
        src_ptr, dst_ptr = get_real_src_dst(src, dst)
        sid = 0

        pad_mode_dict = {0: 'PAD_NONE', 1: 'PAD_MODE1', 2: 'PAD_MODE2',
                         3: 'PAD_MODE3',
                         4: 'PAD_MODE4',
                         5: 'PAD_MODE5'}
        cr_mode_dict = {0: 'CRMODE_NONE', 1: 'CRMODE_F32toF16_NONE',
                        2: 'CRMODE_F32toF16_RELU',
                        3: 'CRMODE_S32toF16_NONE',
                        4: 'CRMODE_F16toF32_NONE',
                        5: 'CRMODE_NONE_RELU'}
        pad_mode = []
        cr_mode = []
        intrin_name = "cceMove"
        if dst.scope == param.scope_ubuf and src.scope == param.scope_ubuf:
            intrin_name = "copy_ubuf_to_ubuf"
        elif dst.scope == param.scope_cbuf and src.scope == 'global':
            intrin_name = "copy_gm_to_cbuf"
            sid = cce_util.get_dma_sid("Sid_copy_gm_to_cbuf")
            pad_mode.append(pad_mode_dict[0])
            pad_mode_call = tvm.call_pure_intrin("int32",
                                                 "tvm_cce_string_print",
                                                 *pad_mode)
            sid = cce_util.get_dma_sid("Sid_copy_gm_to_cbuf")
            i_b.emit(tvm.call_extern(
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
            return i_b.get()
        elif dst.scope == param.scope_ubuf and src.scope == param.scope_cc:
            intrin_name = "copy_matrix_cc_to_ubuf"
            cr_mode.append(cr_mode_dict[0])
            cr_mode_call = tvm.call_pure_intrin("int32",
                                                "tvm_cce_string_print",
                                                *cr_mode)
            i_b.emit(tvm.call_extern(
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
            return i_b.get()
        elif dst.scope == 'global' and src.scope.count(param.scope_ubuf):
            intrin_name = "copy_ubuf_to_gm"
        elif dst.scope.count(param.scope_ubuf) and src.scope == 'global':
            intrin_name = "copy_gm_to_ubuf"
            sid = cce_util.get_dma_sid("Sid_copy_gm_to_ubuf")

        i_b.emit(tvm.call_extern(
            dst.dtype, intrin_name,
            dst_ptr,  # dst buffer
            src_ptr,  # src buffer
            sid,
            nburst,
            burst,
            src_stride,
            dst_stride
        ))

        return i_b.get()

    if mock:
        if kwargs["fun_name"] == "_inject_copy":
            _inject_copy(kwargs["src"], kwargs["dst"], kwargs["pad_before"],
                         kwargs["pad_after"],
                         kwargs["pad_value"])
        elif kwargs["fun_name"] == "_get_mov_pattern":
            _get_mov_pattern(kwargs["src_buf"], kwargs["elem_width"],
                             kwargs["elem_bytes"],
                             kwargs["dst_buf"],
                             kwargs["allow_fold"])
        elif kwargs["fun_name"] == "_get_2d_pattern_cce":
            _get_2d_pattern_cce(kwargs["src_buf"], kwargs["elem_width"],
                                kwargs["elem_bytes"],
                                kwargs["dst_buf"],
                                kwargs["allow_fold"])
    else:
        return tvm.ir_pass.InjectCopyIntrin(stmt_in, param.dma_copy,
                                            _inject_copy)
