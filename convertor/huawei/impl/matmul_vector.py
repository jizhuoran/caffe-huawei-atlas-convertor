#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

matmul_vector
"""
from __future__ import absolute_import
from __future__ import division

from te import tvm
from te import platform as cce
from te.platform.cce_build import build_config
from te.platform import insn_cmd

from .transpose_d import _do_storage_align
from .transpose_d import _tilling_axis_not_last
from .transpose_d import _write_code

import topi


# pylint: disable=locally-disabled,unnecessary-lambda,too-many-locals,too-many-statements,too-many-lines,too-many-branches
def _schedule_large_km_kn(shape, list_computes, src_type):
    """
    Matrix multiplication matmul_vector for KN x KN, schedule for the km_kn when the shape is large
    ----------
    """
    result = list_computes[0]
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_bais_ub = list_computes[5]
    the_result_bais_ub = list_computes[6]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]
    tensor_result_ub_cast = list_computes[9]
    tensor_temp_bias = list_computes[10]

    schedule = tvm.create_schedule([result.op])
    axis_outer = 0
    axis_inner = 1

    schedule[the_result_ub].reorder(the_result_ub.op.reduce_axis[0],
                                    the_result_ub.op.axis[axis_outer],
                                    the_result_ub.op.axis[axis_inner])
    schedule[tensor_a_ub].compute_at(schedule[the_result_mul_ub],
                                     the_result_mul_ub.op.axis[axis_outer])
    schedule[tensor_b_ub].compute_at(schedule[the_result_mul_ub],
                                     the_result_mul_ub.op.axis[axis_outer])

    if src_type == "int32":
        schedule[tensor_temp_a].compute_at(schedule[the_result_mul_ub],
                                           the_result_mul_ub.op.axis[axis_outer])
        schedule[tensor_temp_b].compute_at(schedule[the_result_mul_ub],
                                           the_result_mul_ub.op.axis[axis_outer])

    schedule[the_result_mul_ub].compute_at(schedule[the_result_ub], the_result_ub.op.reduce_axis[0])

    if tensor_bais_ub is not None:
        if src_type == "int32":
            schedule[tensor_temp_bias].compute_at(schedule[the_result_bais_ub],
                                                  the_result_bais_ub.op.axis[axis_outer])
        schedule[tensor_bais_ub].compute_at(schedule[the_result_bais_ub],
                                            the_result_bais_ub.op.axis[axis_outer])
        schedule[the_result_ub].compute_at(schedule[the_result_bais_ub],
                                           the_result_bais_ub.op.axis[axis_outer])

    n_axis_inner = _get_tiling_km_kn(shape)

    axis_one = schedule[result].split(result.op.axis[axis_outer], factor=n_axis_inner[0])
    axis_two = schedule[result].split(result.op.axis[axis_inner], factor=n_axis_inner[1])

    m_axis = shape[0]
    core_num = _get_core_num(m_axis)
    if core_num != -1:
        core_facotr = int(m_axis / core_num)
    else:
        core_facotr = int(m_axis)

    core_axis = schedule[result].split(axis_one[0],
                                       factor=core_facotr)

    schedule[result].reorder(core_axis[0], core_axis[1], axis_two[0],
                             axis_one[1], axis_two[1])

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].compute_at(schedule[result], axis_two[0])
    else:
        schedule[the_result_ub].compute_at(schedule[result], axis_two[0])

    if core_num != -1 and (n_axis_inner[1]*core_facotr) % 16 == 0:
        schedule[result].bind(core_axis[0], tvm.thread_axis('blockIdx.x'))

    if src_type == "int32":
        schedule[tensor_result_ub_cast].compute_at(schedule[result], axis_two[0])

    schedule[tensor_a_ub].set_scope(cce.scope_ubuf)
    schedule[tensor_b_ub].set_scope(cce.scope_ubuf)

    if tensor_bais_ub is not None:
        if src_type == "int32":
            schedule[tensor_temp_bias].set_scope(cce.scope_ubuf)
        schedule[tensor_bais_ub].set_scope(cce.scope_ubuf)
        schedule[the_result_bais_ub].set_scope(cce.scope_ubuf)

    schedule[the_result_mul_ub].set_scope(cce.scope_ubuf)
    schedule[the_result_ub].set_scope(cce.scope_ubuf)

    if src_type == "int32":
        schedule[tensor_temp_a].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_b].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_a].emit_insn(tensor_temp_a.op.axis[axis_inner], insn_cmd.CAST)
        schedule[tensor_temp_b].emit_insn(tensor_temp_b.op.axis[axis_inner], insn_cmd.CAST)
        schedule[tensor_result_ub_cast].emit_insn(tensor_result_ub_cast.op.axis[axis_inner],
                                                  insn_cmd.CAST_ROUND)

    schedule[tensor_a_ub].emit_insn(tensor_a_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)
    schedule[tensor_b_ub].emit_insn(tensor_b_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)

    if tensor_bais_ub is not None:
        if src_type == "int32":
            schedule[tensor_temp_bias].emit_insn(tensor_temp_bias.op.axis[axis_inner],
                                                 insn_cmd.CAST)
        schedule[tensor_bais_ub].emit_insn(tensor_bais_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)

    schedule[the_result_mul_ub].emit_insn(the_result_mul_ub.op.axis[axis_inner + 1],
                                          insn_cmd.MULVS)
    schedule[the_result_ub].emit_insn(the_result_ub.op.axis[axis_inner], insn_cmd.ADD)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].emit_insn(the_result_bais_ub.op.axis[axis_inner],
                                               insn_cmd.ADD)

    schedule[result].emit_insn(axis_two[1], insn_cmd.DMA_COPY)

    return schedule


def _get_tiling_km_kn(shape):
    """
    Matrix multiplication matmul_vector for KN x KN, get the tiling num for M, N, K
    ----------
    """
    # the float32 num take up the four bytes, there float32_size equal four
    float32_size = 4
    ub_size = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE) / float32_size / 2
    shape_n = shape[1]
    n_axis_inner = shape_n
    n_outter = 1
    min_m_axis = 1

    if _get_restriction_km_kn(min_m_axis, n_axis_inner) < ub_size:
        return min_m_axis, n_axis_inner

    while True:
        if _get_restriction_km_kn(min_m_axis, n_axis_inner) < ub_size:
            break
        n_outter = n_outter + 1

        if shape_n % n_outter != 0:
            n_axis_inner = shape_n // n_outter + 1
        else:
            n_axis_inner = shape_n // n_outter

    return min_m_axis, n_axis_inner


def _get_restriction_km_kn(m_axis_inner, n_axis_inner):
    """
    Matrix multiplication matmul_vector for KN x KN, get the space in ub
    ----------
    """
    # the ub block size is eight*float32_size, there is eight
    block_size = 8

    if n_axis_inner % block_size != 0:
        n_axis_inner = block_size*(n_axis_inner // block_size + 1)

    the_result = m_axis_inner + n_axis_inner + 2*m_axis_inner*n_axis_inner

    return the_result


# pylint: disable=locally-disabled,too-many-arguments
def _compute_for_km_kn(tensor_a_ub, tensor_b_ub, shape_a, shape_b, tensor_bais_ub, src_type):
    """
    Matrix multiplication matmul_vector for KN x KN, The compute for MK x NK
    ----------
    """
    # set output shape format is M x N.
    output_shape = (shape_a[1], shape_b[1])
    output_shape_mul = (shape_a[0], shape_a[1], shape_b[1])
    tensor_temp_a = tensor_a_ub
    tensor_temp_b = tensor_b_ub
    tensor_temp_bias = tensor_bais_ub
    if src_type == "int32":
        tensor_temp_a = tvm.compute(shape_a, lambda *i: topi.cast(tensor_a_ub(*i), "float32"),
                                    name='tensor_a_ub_cast')
        tensor_temp_b = tvm.compute(shape_b, lambda *i: topi.cast(tensor_b_ub(*i), "float32"),
                                    name='tensor_a_ub_cast')
        if tensor_bais_ub is not None:
            tensor_temp_bias = tvm.compute(output_shape,
                                           lambda *i: topi.cast(tensor_bais_ub(*i), "float32"),
                                           name='tensor_bais_ub_cast')

    the_result_mul_ub = tvm.compute(output_shape_mul,
                                    lambda k, m, n: tensor_temp_b(k, n) * tensor_temp_a(k, m),
                                    name="the_result_mul_ub")

    reduce_k_axis = tvm.reduce_axis((0, shape_a[0]), name="reduce_k_axis")
    the_result_ub = tvm.compute(output_shape,
                                lambda m, n: tvm.sum(the_result_mul_ub[reduce_k_axis, m, n],
                                                     axis=reduce_k_axis), name="the_result_ub")
    the_result_bais_ub = None
    the_result_temp = the_result_ub

    if tensor_bais_ub is not None:
        the_result_bais_ub = tvm.compute(output_shape,
                                         lambda m, n: the_result_ub(m, n) +
                                         tensor_temp_bias(m, n), name="the_result_ub")
        the_result_temp = the_result_bais_ub

    if src_type == "int32":
        tensor_result_ub_cast = tvm.compute(output_shape,
                                            lambda *i: topi.cast(the_result_temp(*i), "int32"),
                                            name='tensor_result_ub_cast')
    else:
        tensor_result_ub_cast = the_result_temp

    the_result = tvm.compute(output_shape, lambda *i: tensor_result_ub_cast(*i), name='the_result')

    return tensor_temp_a, tensor_temp_b, tensor_result_ub_cast, tensor_temp_bias, \
           the_result_mul_ub, the_result_ub, the_result_bais_ub, the_result


def _matmul_new_km_kn_cce(tensor_a, tensor_b, tensor_bais, src_type):
    """
    algorithm: Matrix multiplication matmul_vector for KM x KN situation
    ----------
    """
    shape_a = (tensor_a.shape[0].value, tensor_a.shape[1].value)
    shape_b = (tensor_b.shape[0].value, tensor_b.shape[1].value)
    output_bais = (tensor_a.shape[1].value, tensor_b.shape[1].value, tensor_a.shape[0].value)

    tensor_a_ub = tvm.compute(shape_a, lambda *i: tensor_a(*i), name='tensor_a_ub')
    tensor_b_ub = tvm.compute(shape_b, lambda *i: tensor_b(*i), name='tensor_b_ub')

    if tensor_bais is not None:
        tensor_bais_ub = tvm.compute(output_bais,
                                     lambda m, n: tensor_bais(n), name="tensor_bais_ub")
    else:
        tensor_bais_ub = None

    tensor_temp_a, tensor_temp_b, tensor_result_ub_cast, tensor_temp_bias, the_result_mul_ub,\
                        the_result_ub, the_result_bais_ub, the_result = _compute_for_km_kn(
                                                                        tensor_a_ub, tensor_b_ub,
                                                                        shape_a, shape_b,
                                                                        tensor_bais_ub, src_type)

    shape_schedule = (shape_a[1], shape_b[1], shape_a[0])
    schedule = _schedule_large_km_kn(shape_schedule, (the_result, tensor_a_ub, tensor_b_ub,
                                                      the_result_mul_ub,
                                                      the_result_ub, tensor_bais_ub,
                                                      the_result_bais_ub,
                                                      tensor_temp_a,
                                                      tensor_temp_b,
                                                      tensor_result_ub_cast,
                                                      tensor_temp_bias), src_type)
    schedule[tensor_a_ub].double_buffer()
    schedule[tensor_b_ub].double_buffer()

    return schedule, the_result


def _schedule_mini_mk_kn(list_computes, src_type):
    """
    Matrix multiplication matmul_vector for MK x KN, schedule for mini shape
    ----------
    """
    result = list_computes[0]
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_bais_ub = list_computes[5]
    the_result_bais_ub = list_computes[6]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]
    tensor_result_ub_cast = list_computes[9]
    tensor_temp_bias = list_computes[10]

    schedule = tvm.create_schedule([result.op])
    schedule[tensor_a_ub].set_scope(cce.scope_ubuf)
    schedule[tensor_b_ub].set_scope(cce.scope_ubuf)

    if src_type == "int32":
        schedule[tensor_temp_a].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_b].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].set_scope(cce.scope_ubuf)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].set_scope(cce.scope_ubuf)
        if src_type == "int32":
            schedule[tensor_temp_bias].set_scope(cce.scope_ubuf)

    schedule[the_result_mul_ub].set_scope(cce.scope_ubuf)
    schedule[the_result_ub].set_scope(cce.scope_ubuf)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].set_scope(cce.scope_ubuf)

    axis_outer = 0
    axis_inner = 1

    schedule[the_result_ub].reorder(the_result_ub.op.axis[axis_outer],
                                    the_result_ub.op.reduce_axis[0],
                                    the_result_ub.op.axis[axis_inner])

    schedule[tensor_a_ub].emit_insn(tensor_a_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)
    schedule[tensor_b_ub].emit_insn(tensor_b_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)

    if src_type == "int32":
        schedule[tensor_temp_a].emit_insn(tensor_temp_a.op.axis[axis_inner], insn_cmd.CAST)
        schedule[tensor_temp_b].emit_insn(tensor_temp_b.op.axis[axis_inner], insn_cmd.CAST)
        schedule[tensor_result_ub_cast].emit_insn(tensor_result_ub_cast.op.axis[axis_inner],
                                                  insn_cmd.CAST_ROUND)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].emit_insn(tensor_bais_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)
        if src_type == "int32":
            schedule[tensor_temp_bias].emit_insn(tensor_temp_bias.op.axis[axis_inner],
                                                 insn_cmd.CAST)

    schedule[the_result_mul_ub].emit_insn(the_result_mul_ub.op.axis[axis_inner + 1], \
                              insn_cmd.MULVS)
    schedule[the_result_ub].emit_insn(the_result_ub.op.axis[axis_inner],
                                      insn_cmd.ADD)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].emit_insn(the_result_bais_ub.op.axis[axis_inner],
                                               insn_cmd.ADD)

    schedule[result].emit_insn(result.op.axis[axis_inner], insn_cmd.DMA_COPY)

    return schedule


def _schedule_mid_mk_kn(list_computes, src_type):
    """
    Matrix multiplication matmul_vector for MK x KN, schedule for mid shape
    ----------
    """
    the_result = list_computes[0]
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_bais_ub = list_computes[5]
    the_result_bais_ub = list_computes[6]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]
    tensor_result_ub_cast = list_computes[9]
    tensor_temp_bias = list_computes[10]

    schedule = tvm.create_schedule([the_result.op])
    schedule[tensor_a_ub].set_scope(cce.scope_ubuf)
    schedule[tensor_b_ub].set_scope(cce.scope_ubuf)

    if src_type == "int32":
        schedule[tensor_temp_a].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_b].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].set_scope(cce.scope_ubuf)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].set_scope(cce.scope_ubuf)
        schedule[the_result_bais_ub].set_scope(cce.scope_ubuf)
        if src_type == "int32":
            schedule[tensor_temp_bias].set_scope(cce.scope_ubuf)

    schedule[the_result_mul_ub].set_scope(cce.scope_ubuf)
    schedule[the_result_ub].set_scope(cce.scope_ubuf)


    axis_outer = 0
    axis_inner = 1
    schedule[the_result_ub].reorder(the_result_ub.op.axis[axis_outer],
                                    the_result_ub.op.reduce_axis[0],
                                    the_result_ub.op.axis[axis_inner])
    schedule[tensor_a_ub].compute_at(schedule[the_result_ub],
                                     the_result_ub.op.axis[axis_outer])
    schedule[tensor_b_ub].compute_at(schedule[the_result_mul_ub],
                                     the_result_mul_ub.op.axis[axis_inner])

    if src_type == "int32":
        schedule[tensor_temp_a].compute_at(schedule[the_result_ub],
                                           the_result_ub.op.axis[axis_outer])
        schedule[tensor_temp_b].compute_at(schedule[the_result_mul_ub],
                                           the_result_mul_ub.op.axis[axis_inner])

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].compute_at(schedule[the_result_bais_ub],
                                            the_result_bais_ub.op.axis[axis_outer])
        if src_type == "int32":
            schedule[tensor_temp_bias].compute_at(schedule[the_result_bais_ub],
                                                  the_result_bais_ub.op.axis[axis_outer])

    schedule[the_result_mul_ub].compute_at(schedule[the_result_ub],
                                           the_result_ub.op.reduce_axis[0])

    if src_type == "int32":
        schedule[the_result_ub].compute_at(schedule[tensor_result_ub_cast],
                                           tensor_result_ub_cast.op.axis[axis_outer])
        if the_result_bais_ub is not None:
            schedule[the_result_bais_ub].compute_at(schedule[tensor_result_ub_cast],
                                                    tensor_result_ub_cast.op.axis[axis_outer])
    elif tensor_bais_ub is not None:
        schedule[the_result_ub].compute_at(schedule[the_result_bais_ub],
                                           the_result_bais_ub.op.axis[axis_outer])

    schedule[tensor_a_ub].emit_insn(tensor_a_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)
    schedule[tensor_b_ub].emit_insn(tensor_b_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)

    if src_type == "int32":
        schedule[tensor_temp_a].emit_insn(tensor_temp_a.op.axis[axis_inner], insn_cmd.CAST)
        schedule[tensor_temp_b].emit_insn(tensor_temp_b.op.axis[axis_inner], insn_cmd.CAST)
        schedule[tensor_result_ub_cast].emit_insn(tensor_result_ub_cast.op.axis[axis_inner],
                                                  insn_cmd.CAST_ROUND)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].emit_insn(tensor_bais_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)
        schedule[the_result_bais_ub].emit_insn(the_result_bais_ub.op.axis[axis_inner],
                                               insn_cmd.ADD)
        if src_type == "int32":
            schedule[tensor_temp_bias].emit_insn(tensor_temp_bias.op.axis[axis_inner],
                                                 insn_cmd.CAST)

    schedule[the_result_mul_ub].emit_insn(the_result_mul_ub.op.axis[axis_inner + 1], \
                              insn_cmd.MULVS)
    schedule[the_result_ub].emit_insn(the_result_ub.op.axis[axis_inner],
                                      insn_cmd.ADD)

    schedule[the_result].emit_insn(the_result.op.axis[axis_inner], insn_cmd.DMA_COPY)

    return schedule


def _schedule_large_mk_kn(shape, list_computes, src_type, trans_flag):
    """
    Matrix multiplication matmul_vector for MK x KN, schedule for large shape
    ----------
    """
    the_result = list_computes[0]
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_bais_ub = list_computes[5]
    the_result_bais_ub = list_computes[6]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]
    tensor_result_ub_cast = list_computes[9]
    tensor_temp_bias = list_computes[10]

    schedule = tvm.create_schedule([the_result.op])
    tiling_number = _get_tiling_mk_kn(shape)

    schedule[tensor_a_ub].set_scope(cce.scope_ubuf)
    schedule[tensor_b_ub].set_scope(cce.scope_ubuf)

    if src_type == "int32":
        schedule[tensor_temp_a].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_b].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].set_scope(cce.scope_ubuf)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].set_scope(cce.scope_ubuf)
        if src_type == "int32":
            schedule[tensor_temp_bias].set_scope(cce.scope_ubuf)

    schedule[the_result_mul_ub].set_scope(cce.scope_ubuf)
    schedule[the_result_ub].set_scope(cce.scope_ubuf)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].set_scope(cce.scope_ubuf)

    schedule[the_result_ub].reorder(the_result_ub.op.axis[0],
                                    the_result_ub.op.reduce_axis[0], the_result_ub.op.axis[1])

    axis_one = schedule[tensor_a_ub].split(tensor_a_ub.op.axis[1], factor=tiling_number[2])

    if src_type == "int32":
        axis_one_cast = schedule[tensor_temp_a].split(tensor_temp_a.op.axis[1],
                                                      factor=tiling_number[2])

    axis_two = schedule[the_result_ub].split(the_result_ub.op.reduce_axis[0],
                                             factor=tiling_number[2])

    schedule[tensor_a_ub].compute_at(schedule[the_result_ub], axis_two[0])
    schedule[tensor_b_ub].compute_at(schedule[the_result_mul_ub],
                                     the_result_mul_ub.op.axis[1])

    if src_type == "int32":
        schedule[tensor_temp_a].compute_at(schedule[the_result_ub], axis_two[0])
        schedule[tensor_temp_b].compute_at(schedule[the_result_mul_ub],
                                           the_result_mul_ub.op.axis[1])

    schedule[the_result_mul_ub].compute_at(schedule[the_result_ub], axis_two[1])

    axis_three = schedule[the_result].split(the_result.op.axis[0],
                                            factor=tiling_number[0])
    axis_four = schedule[the_result].split(the_result.op.axis[1],
                                           factor=tiling_number[1])

    m_axis = shape[0]
    core_num = _get_core_num(m_axis)
    if core_num != -1:
        core_facotr = int(m_axis / core_num)
    else:
        core_facotr = int(m_axis)

    core_axis = schedule[the_result].split(axis_three[0],
                                           factor=core_facotr)

    schedule[the_result].reorder(core_axis[0], core_axis[1], axis_four[0],
                                 axis_three[1], axis_four[1])

    if the_result_bais_ub is not None:
        schedule[tensor_bais_ub].compute_at(schedule[the_result_bais_ub],
                                            the_result_bais_ub.op.axis[0])
        schedule[the_result_ub].compute_at(schedule[the_result_bais_ub],
                                           the_result_bais_ub.op.axis[0])
        if src_type == "int32":
            schedule[tensor_temp_bias].compute_at(schedule[the_result_bais_ub],
                                                  the_result_bais_ub.op.axis[0])
        schedule[the_result_bais_ub].compute_at(schedule[the_result], axis_four[0])
    else:
        schedule[the_result_ub].compute_at(schedule[the_result], axis_four[0])

    if core_num != -1 and tiling_number[1] % 8 == 0 and trans_flag:
        schedule[the_result].bind(core_axis[0], tvm.thread_axis('blockIdx.x'))

    if src_type == "int32":
        schedule[tensor_result_ub_cast].compute_at(schedule[the_result], axis_four[0])
        schedule[tensor_temp_a].emit_insn(axis_one_cast[1], insn_cmd.CAST)
        schedule[tensor_temp_b].emit_insn(tensor_temp_b.op.axis[1], insn_cmd.CAST)
        schedule[tensor_result_ub_cast].emit_insn(tensor_result_ub_cast.op.axis[1],
                                                  insn_cmd.CAST_ROUND)

    schedule[tensor_a_ub].emit_insn(axis_one[1], insn_cmd.DMA_COPY)
    schedule[tensor_b_ub].emit_insn(tensor_b_ub.op.axis[1], insn_cmd.DMA_COPY)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].emit_insn(tensor_bais_ub.op.axis[1], insn_cmd.DMA_COPY)
        schedule[the_result_bais_ub].emit_insn(the_result_bais_ub.op.axis[1],
                                               insn_cmd.ADD)
        if src_type == "int32":
            schedule[tensor_temp_bias].emit_insn(tensor_temp_bias.op.axis[1],
                                                 insn_cmd.CAST)
    print("come here")
    schedule[the_result_mul_ub].emit_insn(the_result_mul_ub.op.axis[2], \
                              insn_cmd.MULVS)
    schedule[the_result_ub].emit_insn(the_result_ub.op.axis[1],
                                      insn_cmd.ADD)

    schedule[the_result].emit_insn(axis_four[1], insn_cmd.DMA_COPY)

    return schedule


def _get_tiling_mk_kn(shape):
    """
    Matrix multiplication matmul_vector for MK x KN, get the tiling num for M, N, K
    ----------
    """
    # the float32 num take up the four bytes, there float32_size equal four
    float32_size = 4
    ub_size = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE) / float32_size / 2
    shape_n = shape[1]
    shape_k = shape[2]
    n_axis_outer = 1
    k_axis_outer = 1
    n_axis_inner = shape_n
    k_axis_inner = shape_k
    min_m_axis = 1
    min_k_axis = 2

    if _get_restriction_mk_kn(min_m_axis, n_axis_inner,
                              min_k_axis, shape_n, shape_k) < ub_size:
        while True:
            if _get_restriction_mk_kn(min_m_axis, n_axis_inner,
                                      k_axis_inner, shape_n, shape_k) < ub_size:
                break
            k_axis_outer = k_axis_outer + 1
            if shape_k % k_axis_outer != 0:
                k_axis_inner = shape_k // k_axis_outer + 1
            else:
                k_axis_inner = shape_k // k_axis_outer
    else:
        while True:
            if _get_restriction_mk_kn(min_m_axis, n_axis_inner,
                                      min_k_axis, shape_n, shape_k) < ub_size:
                k_axis_inner = 2
                break
            n_axis_outer = n_axis_outer + 1
            if shape_n % n_axis_outer != 0:
                n_axis_inner = shape_n // n_axis_outer + 1
            else:
                n_axis_inner = shape_n // n_axis_outer

    return min_m_axis, n_axis_inner, k_axis_inner


def _get_restriction_mk_kn(m_axis_inner, n_axis_inner, k_axis_inner, shape_n, shape_k):
    """
    Matrix multiplication matmul_vector for MK x KN, get the compute space in ub,
    the space is little than us_size
    ----------
    """
    # the ub block size is eight*float32_size, there is eight
    block_size = 8
    n_axis_be_divided = False
    k_axis_be_divided = False

    if shape_n % n_axis_inner != 0:
        n_axis_be_divided = True
        n_axis_remainder = shape_n % n_axis_inner

    if shape_k % k_axis_inner != 0:
        k_axis_be_divided = True
        k_axis_remainder = shape_k % k_axis_inner

    if k_axis_inner % block_size != 0:
        cur_k_axis_inner = block_size*(k_axis_inner // block_size + 1)
    else:
        cur_k_axis_inner = k_axis_inner

    if n_axis_inner % block_size != 0:
        cur_n_axis_inner = block_size*(n_axis_inner // block_size + 1)
    else:
        cur_n_axis_inner = n_axis_inner
    the_result = m_axis_inner*cur_n_axis_inner + cur_k_axis_inner + 2*cur_n_axis_inner

    if n_axis_be_divided:
        the_result = the_result + max(3*n_axis_remainder + k_axis_inner, cur_n_axis_inner)

    if k_axis_be_divided:
        the_result = the_result + k_axis_remainder + cur_n_axis_inner

    return the_result


def _get_schedule_mk_kn(shape, list_compute, src_type, trans_flag):
    """
    Matrix multiplication matmul_vector for MK x KN, choose the schedule for different shape
    ----------
    """

    schedule = _schedule_large_mk_kn(shape, list_compute, src_type, trans_flag)

    return schedule


def _compute_for_mk_kn(tensor_a_ub, tensor_b_ub, shape_a, shape_b, tensor_bais_ub, src_type):
    """
    Matrix multiplication matmul_vector for MK x KN, The compute for MK x KN
    ----------
    """
    output_shape = (shape_a[0], shape_b[1])
    output_shape_mul = (shape_a[0], shape_a[1], shape_b[1])

    tensor_temp_bias = tensor_bais_ub
    tensor_temp_a = tensor_a_ub
    tensor_temp_b = tensor_b_ub

    if src_type == "int32":
        tensor_temp_a = tvm.compute(shape_a, lambda *i: topi.cast(tensor_a_ub(*i), "float32"),
                                    name='tensor_a_ub_cast')
        tensor_temp_b = tvm.compute(shape_b, lambda *i: topi.cast(tensor_b_ub(*i), "float32"),
                                    name='tensor_a_ub_cast')
        if tensor_bais_ub is not None:
            tensor_temp_bias = tvm.compute(output_shape,
                                           lambda *i: topi.cast(tensor_bais_ub(*i), "float32"),
                                           name='tensor_bais_ub_cast')

    the_result_mul_ub = tvm.compute(output_shape_mul, \
                             lambda m, k, n: tensor_temp_b(k, n) * tensor_temp_a(m, k), \
                             name="the_result_mul_ub")

    reduce_k_axis = tvm.reduce_axis((0, shape_a[1]), name="reduce_k_axis")
    the_result_ub = tvm.compute(output_shape, \
                         lambda m, n: tvm.sum(the_result_mul_ub[m, reduce_k_axis, n],
                                              axis=reduce_k_axis), name="the_result_ub")
    the_result_bais_ub = None
    the_result_temp = the_result_ub

    if tensor_bais_ub is not None:
        the_result_bais_ub = tvm.compute(output_shape,
                                         lambda m, n: the_result_ub(m, n) +
                                         tensor_temp_bias(m, n),
                                         name="the_result_ub")
        the_result_temp = the_result_bais_ub

    if src_type == "int32":
        tensor_result_ub_cast = tvm.compute(output_shape,
                                            lambda *i: topi.cast(the_result_temp(*i), "int32"),
                                            name='tensor_result_ub_cast')
    else:
        tensor_result_ub_cast = the_result_temp

    the_result = tvm.compute(output_shape, lambda *i: tensor_result_ub_cast(*i), name='the_result')

    return tensor_temp_a, tensor_temp_b, tensor_result_ub_cast, tensor_temp_bias, \
           the_result_mul_ub, the_result_ub, the_result_bais_ub, the_result


def _matmul_new_mk_kn_cce(tensor_a, tensor_b, tensor_bais, src_type):
    """
    algorithm: Matrix multiplication matmul_vector for MK x KN situation
    ----------
    """
    shape_a = (tensor_a.shape[0].value, tensor_a.shape[1].value)
    shape_b = (tensor_b.shape[0].value, tensor_b.shape[1].value)
    output_bais = (tensor_a.shape[0].value, tensor_b.shape[1].value, tensor_a.shape[1].value)

    tensor_a_ub = tvm.compute(shape_a, lambda *i: tensor_a(*i), name='tensor_a_ub')
    tensor_b_ub = tvm.compute(shape_b, lambda *i: tensor_b(*i), name='tensor_b_ub')

    if tensor_bais is not None:
        tensor_bais_ub = tvm.compute(output_bais,
                                     lambda m, n: tensor_bais(n), name="tensor_bais_ub")
    else:
        tensor_bais_ub = None

    tensor_temp_a, tensor_temp_b, tensor_result_ub_cast, tensor_temp_bias, the_result_mul_ub, \
    the_result_ub, the_result_bais_ub, the_result = _compute_for_mk_kn(tensor_a_ub, tensor_b_ub,
                                                                       shape_a, shape_b,
                                                                       tensor_bais_ub, src_type)

    schedule = _get_schedule_mk_kn((shape_a[0], shape_b[1], shape_a[1]), \
                                   (the_result, tensor_a_ub, tensor_b_ub,
                                    the_result_mul_ub, the_result_ub, \
                                    tensor_bais_ub, the_result_bais_ub,
                                    tensor_temp_a, tensor_temp_b,
                                    tensor_result_ub_cast,
                                    tensor_temp_bias,), src_type, True)

    schedule[tensor_a_ub].double_buffer()
    schedule[tensor_b_ub].double_buffer()

    return schedule, the_result


def _schedule_mini_mk_nk(list_computes, src_type):
    """
    Matrix multiplication matmul_vector for MK x NK, schedule for mini shape
    ----------
    """
    the_result = list_computes[0]
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_bais_ub = list_computes[5]
    the_result_bais_ub = list_computes[6]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]
    tensor_result_ub_cast = list_computes[9]
    tensor_temp_bias = list_computes[10]

    schedule = tvm.create_schedule([the_result.op])

    axis_two = 1

    schedule[the_result_mul_ub].compute_at(schedule[the_result_ub], the_result_ub.op.axis[axis_two])

    schedule[tensor_a_ub].set_scope(cce.scope_ubuf)
    schedule[tensor_b_ub].set_scope(cce.scope_ubuf)

    if src_type == "int32":
        schedule[tensor_temp_a].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_b].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].set_scope(cce.scope_ubuf)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].set_scope(cce.scope_ubuf)
        if src_type == "int32":
            schedule[tensor_temp_bias].set_scope(cce.scope_ubuf)

    schedule[the_result_mul_ub].set_scope(cce.scope_ubuf)
    schedule[the_result_ub].set_scope(cce.scope_ubuf)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].set_scope(cce.scope_ubuf)

    schedule[tensor_a_ub].emit_insn(tensor_a_ub.op.axis[axis_two], insn_cmd.DMA_COPY)
    schedule[tensor_b_ub].emit_insn(tensor_b_ub.op.axis[axis_two], insn_cmd.DMA_COPY)

    if src_type == "int32":
        schedule[tensor_temp_a].emit_insn(tensor_temp_a.op.axis[axis_two], insn_cmd.CAST)
        schedule[tensor_temp_b].emit_insn(tensor_temp_b.op.axis[axis_two], insn_cmd.CAST)
        schedule[tensor_result_ub_cast].emit_insn(tensor_result_ub_cast.op.axis[axis_two],
                                                  insn_cmd.CAST_ROUND)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].emit_insn(tensor_bais_ub.op.axis[axis_two], insn_cmd.DMA_COPY)
        if src_type == "int32":
            schedule[tensor_temp_bias].emit_insn(tensor_temp_bias.op.axis[axis_two],
                                                 insn_cmd.CAST)

    schedule[the_result_mul_ub].emit_insn(the_result_mul_ub.op.axis[axis_two + 1],
                                          insn_cmd.MUL)
    schedule[the_result_ub].emit_insn(the_result_ub.op.reduce_axis[0],
                                      insn_cmd.REDUCE_SUM)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].emit_insn(the_result_bais_ub.op.axis[axis_two],
                                               insn_cmd.ADD)
    schedule[the_result].emit_insn(the_result.op.axis[axis_two], insn_cmd.DMA_COPY)

    return schedule


def _schedule_mid_mk_nk(list_computes, src_type):
    """
    Matrix multiplication matmul_vector for MK x NK schedule for mid shape
    ----------
    """
    the_result = list_computes[0]
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_bais_ub = list_computes[5]
    the_result_bais_ub = list_computes[6]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]
    tensor_result_ub_cast = list_computes[9]
    tensor_temp_bias = list_computes[10]

    schedule = tvm.create_schedule([the_result.op])

    axis_one = 0
    axis_two = 1

    schedule[tensor_a_ub].compute_at(schedule[the_result_mul_ub],
                                     the_result_mul_ub.op.axis[axis_two])
    schedule[tensor_b_ub].compute_at(schedule[the_result_mul_ub],
                                     the_result_mul_ub.op.axis[axis_two])

    if src_type == "int32":
        schedule[tensor_temp_a].compute_at(schedule[the_result_mul_ub],
                                           the_result_mul_ub.op.axis[axis_two])
        schedule[tensor_temp_b].compute_at(schedule[the_result_mul_ub],
                                           the_result_mul_ub.op.axis[axis_two])

    schedule[the_result_mul_ub].compute_at(schedule[the_result_ub],
                                           the_result_ub.op.axis[axis_two])

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].compute_at(schedule[the_result_bais_ub],
                                            the_result_bais_ub.op.axis[axis_one])
        schedule[the_result_ub].compute_at(schedule[the_result_bais_ub],
                                           the_result_bais_ub.op.axis[axis_one])
        if src_type == "int32":
            schedule[tensor_temp_bias].compute_at(schedule[the_result_bais_ub],
                                                  the_result_bais_ub.op.axis[axis_one])
            schedule[the_result_bais_ub].compute_at(schedule[tensor_result_ub_cast],
                                                    tensor_result_ub_cast.op.axis[axis_one])
    elif src_type == "int32":
        schedule[the_result_ub].compute_at(schedule[tensor_result_ub_cast],
                                           tensor_result_ub_cast.op.axis[axis_one])

    schedule[tensor_a_ub].set_scope(cce.scope_ubuf)
    schedule[tensor_b_ub].set_scope(cce.scope_ubuf)

    if src_type == "int32":
        schedule[tensor_temp_a].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_b].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].set_scope(cce.scope_ubuf)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].set_scope(cce.scope_ubuf)
        if src_type == "int32":
            schedule[tensor_temp_bias].set_scope(cce.scope_ubuf)

    schedule[the_result_mul_ub].set_scope(cce.scope_ubuf)
    schedule[the_result_ub].set_scope(cce.scope_ubuf)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].set_scope(cce.scope_ubuf)

    schedule[tensor_a_ub].emit_insn(tensor_a_ub.op.axis[axis_two], insn_cmd.DMA_COPY)
    schedule[tensor_b_ub].emit_insn(tensor_b_ub.op.axis[axis_two], insn_cmd.DMA_COPY)

    if src_type == "int32":
        schedule[tensor_temp_a].emit_insn(tensor_temp_a.op.axis[axis_two],
                                          insn_cmd.CAST)
        schedule[tensor_temp_b].emit_insn(tensor_temp_b.op.axis[axis_two],
                                          insn_cmd.CAST)
        schedule[tensor_result_ub_cast].emit_insn(tensor_result_ub_cast.op.axis[axis_two],
                                                  insn_cmd.CAST_ROUND)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].emit_insn(tensor_bais_ub.op.axis[axis_two],
                                           insn_cmd.DMA_COPY)
        if src_type == "int32":
            schedule[tensor_temp_bias].emit_insn(tensor_temp_bias.op.axis[axis_two],
                                                 insn_cmd.CAST)

    schedule[the_result_mul_ub].emit_insn(the_result_mul_ub.op.axis[axis_two+1],
                                          insn_cmd.MUL)
    schedule[the_result_ub].emit_insn(the_result_ub.op.reduce_axis[0],
                                      insn_cmd.REDUCE_SUM)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].emit_insn(the_result_bais_ub.op.axis[axis_two],
                                               insn_cmd.ADD)
    schedule[the_result].emit_insn(the_result.op.axis[axis_two], insn_cmd.DMA_COPY)

    return schedule


def _single_tiling(shape, schedule, list_computes, src_type):
    """
    Matrix multiplication matmul_vector for MK x NK handle some axis
    ----------
    """
    tiling_number = _get_tiling_mk_nk(shape)
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]

    axis_one = schedule[tensor_a_ub].split(tensor_a_ub.op.axis[1],
                                           factor=tiling_number[2])
    axis_two = schedule[tensor_b_ub].split(tensor_b_ub.op.axis[1],
                                           factor=tiling_number[2])

    if src_type == "int32":
        axis_one_cast = schedule[tensor_temp_a].split(tensor_temp_a.op.axis[1],
                                                      factor=tiling_number[2])
        axis_two_cast = schedule[tensor_temp_b].split(tensor_temp_b.op.axis[1],
                                                      factor=tiling_number[2])

    axis_three = schedule[the_result_mul_ub].split(the_result_mul_ub.op.axis[2],
                                                   factor=tiling_number[2])
    axis_four = schedule[the_result_ub].split(the_result_ub.op.reduce_axis[0],
                                              factor=tiling_number[2])

    schedule[tensor_a_ub].compute_at(schedule[the_result_mul_ub], axis_three[0])
    schedule[tensor_b_ub].compute_at(schedule[the_result_mul_ub], axis_three[0])

    if src_type == "int32":
        schedule[tensor_temp_a].compute_at(schedule[the_result_mul_ub],
                                           axis_three[0])
        schedule[tensor_temp_b].compute_at(schedule[the_result_mul_ub],
                                           axis_three[0])

    schedule[the_result_mul_ub].compute_at(schedule[the_result_ub], axis_four[0])
    schedule[tensor_a_ub].emit_insn(axis_one[1], insn_cmd.DMA_COPY)
    schedule[tensor_b_ub].emit_insn(axis_two[1], insn_cmd.DMA_COPY)

    if src_type == "int32":
        schedule[tensor_temp_a].emit_insn(axis_one_cast[1], insn_cmd.CAST)
        schedule[tensor_temp_b].emit_insn(axis_two_cast[1], insn_cmd.CAST)

    schedule[the_result_mul_ub].emit_insn(axis_three[1], insn_cmd.MUL)
    schedule[the_result_ub].emit_insn(axis_four[1], insn_cmd.REDUCE_SUM)


def _schedule_large_mk_nk(shape, list_computes, src_type):
    """
    Matrix multiplication matmul_vector for MK x NK schedule for large shape
    ----------
    """
    the_result = list_computes[0]
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_bais_ub = list_computes[5]
    the_result_bais_ub = list_computes[6]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]
    tensor_result_ub_cast = list_computes[9]
    tensor_temp_bias = list_computes[10]

    schedule = tvm.create_schedule([the_result.op])

    tiling_number = _get_tiling_mk_nk(shape)
    _single_tiling(shape, schedule, list_computes, src_type)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].compute_at(schedule[the_result_bais_ub],
                                            the_result_bais_ub.op.axis[0])
        schedule[the_result_ub].compute_at(schedule[the_result_bais_ub],
                                           the_result_bais_ub.op.axis[0])
        if src_type == "int32":
            schedule[tensor_temp_bias].compute_at(schedule[the_result_bais_ub],
                                                  the_result_bais_ub.op.axis[0])

    axis_one = schedule[the_result].split(the_result.op.axis[0], factor=tiling_number[0])
    axis_two = schedule[the_result].split(the_result.op.axis[1], factor=tiling_number[1])

    m_axis = shape[0]
    core_num = _get_core_num(m_axis)

    if core_num != -1:
        core_factor = int(m_axis / core_num)
    else:
        core_factor = int(m_axis)

    core_axis = schedule[the_result].split(axis_one[0], factor=core_factor)

    schedule[the_result].reorder(core_axis[0], core_axis[1], axis_two[0], axis_one[1], axis_two[1])

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].compute_at(schedule[the_result], axis_two[0])
    else:
        schedule[the_result_ub].compute_at(schedule[the_result], axis_two[0])

    align_size = tiling_number[1] + 8
    schedule[the_result_ub].storage_align(the_result_ub.op.axis[0], align_size, 0)

    if src_type == "int32":
        schedule[tensor_result_ub_cast].compute_at(schedule[the_result], axis_two[0])

    if core_num != -1 and (core_factor*tiling_number[1]) % 8 == 0 and tiling_number[1] % 8 == 0:
        schedule[the_result].bind(core_axis[0], tvm.thread_axis('blockIdx.x'))

    schedule[tensor_a_ub].set_scope(cce.scope_ubuf)
    schedule[tensor_b_ub].set_scope(cce.scope_ubuf)

    if src_type == "int32":
        schedule[tensor_temp_a].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_b].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].emit_insn(tensor_result_ub_cast.op.axis[1],
                                                  insn_cmd.CAST_ROUND)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].set_scope(cce.scope_ubuf)
        if src_type == "int32":
            schedule[tensor_temp_bias].set_scope(cce.scope_ubuf)

    schedule[the_result_mul_ub].set_scope(cce.scope_ubuf)
    schedule[the_result_ub].set_scope(cce.scope_ubuf)


    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].set_scope(cce.scope_ubuf)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].emit_insn(tensor_bais_ub.op.axis[1], insn_cmd.DMA_COPY)
        if src_type == "int32":
            schedule[tensor_temp_bias].emit_insn(tensor_temp_bias.op.axis[1],
                                                 insn_cmd.CAST)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].emit_insn(the_result_bais_ub.op.axis[1],
                                               insn_cmd.ADD)

    schedule[the_result].emit_insn(axis_two[1], insn_cmd.DMA_COPY)

    return schedule


def _get_core_num(m_axis):
    """

    :param m_axis:
    :return:
    """
    res = -1
    if m_axis > 32 or m_axis == 32:
        for i in range(32, -1, -1):
            if (m_axis % i) == 0:
                res = i
                break

    return res


def _get_tiling_mk_nk(shape):
    """
    Matrix multiplication matmul_vector for MK x NK get the tiling num for M, N, K
    ----------
    """
    # the float32 num take up the four bytes, there float32_size equal four
    float32_size = 4
    ub_size = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE) / float32_size / 2
    shape_n = shape[1]
    shape_k = shape[2]
    n_axis_outer = 1
    k_axis_outer = 1
    n_axis_inner = shape_n
    k_axis_inner = shape_k

    min_m_axis = 1
    if shape_n % 8 == 0:
        min_n_axis = 8
    else:
        min_n_axis = 2

    if _get_restraint_mk_nk(min_m_axis, n_axis_inner, k_axis_inner) < ub_size:
        return min_m_axis, n_axis_inner, k_axis_inner

    if _get_restraint_mk_nk(min_m_axis, min_n_axis, k_axis_inner) < ub_size:
        while True:
            if _get_restraint_mk_nk(min_m_axis, n_axis_inner, k_axis_inner) < ub_size:
                m_axis_inner = 1
                break
            n_axis_outer = n_axis_outer + 1
            if shape_n % n_axis_outer != 0:
                n_axis_inner = shape_n // n_axis_outer + 1
            else:
                n_axis_inner = shape_n // n_axis_outer
    else:
        while True:
            if _get_restraint_mk_nk(min_m_axis, min_n_axis, k_axis_inner) < ub_size:
                m_axis_inner = 1
                n_axis_inner = min_n_axis
                break
            k_axis_outer = k_axis_outer + 1
            if shape_k % k_axis_outer != 0:
                k_axis_inner = shape_k // k_axis_outer + 1
            else:
                k_axis_inner = shape_k // k_axis_outer

    if shape_n == 21128:
        n_axis_inner = 7040

    return m_axis_inner, n_axis_inner, k_axis_inner


def _get_restraint_mk_nk(m_axis_inner, n_axis_inner, k_axis_inner):
    """
    Matrix multiplication matmul_vector for MK x NK get the space in ub
    ----------
    """
    # the ub block size is eight*float32_size, there is eight
    block_size = 8

    if k_axis_inner % block_size != 0:
        k_axis_inner = block_size*(k_axis_inner // block_size + 1)

    if n_axis_inner % block_size != 0:
        n_axis_inner = block_size*(n_axis_inner // block_size + 1)

    the_result = m_axis_inner*n_axis_inner + 3*k_axis_inner + 3*n_axis_inner

    return the_result


def _get_schedule_mk_nk(shape, list_computes, src_type):
    """
    Matrix multiplication matmul_vector for MK x NK choose the schedule for different shape
    ----------
    """

    schedule = _schedule_large_mk_nk(shape, list_computes, src_type)

    return schedule


def _compute_for_mk_nk(tensor_a_ub, tensor_b_ub, shape_a, shape_b, tensor_bais_ub, src_type):
    """
    The compute for Matrix multiplication MK x NK Situation
    ----------
    """
    output_shape = (shape_a[0], shape_b[0])
    output_shape_mul = (shape_a[0], shape_b[0], shape_a[1])
    tensor_temp_a = tensor_a_ub
    tensor_temp_b = tensor_b_ub
    tensor_temp_bias = tensor_bais_ub

    if src_type == "int32":
        tensor_temp_a = tvm.compute(shape_a, lambda *i: topi.cast(tensor_a_ub(*i), "float32"),
                                    name='tensor_a_ub_cast')
        tensor_temp_b = tvm.compute(shape_b, lambda *i: topi.cast(tensor_b_ub(*i), "float32"),
                                    name='tensor_a_ub_cast')
        if tensor_bais_ub is not None:
            tensor_temp_bias = tvm.compute(output_shape,
                                           lambda *i: topi.cast(tensor_bais_ub(*i), "float32"),
                                           name='tensor_bais_ub_cast')

    the_result_mul_ub = tvm.compute(output_shape_mul, lambda m, n, k:
                                    tensor_temp_a(m, k)*tensor_temp_b(n, k),
                                    name="the_result_mul_ub")
    reduce_k_axis = tvm.reduce_axis((0, shape_a[1]), name="reduce_k_axis")
    the_result_ub = tvm.compute(output_shape, \
                                lambda m, n: tvm.sum(the_result_mul_ub[m, n, reduce_k_axis],
                                                     axis=reduce_k_axis), name="the_result_ub")
    the_result_bais_ub = None
    the_result_temp = the_result_ub

    if tensor_bais_ub is not None:
        the_result_bais_ub = tvm.compute(output_shape,
                                         lambda m, n: the_result_ub(m, n) +
                                         tensor_temp_bias(m, n), name="the_result_ub")
        the_result_temp = the_result_bais_ub

    if src_type == "int32":
        tensor_result_ub_cast = tvm.compute(output_shape, lambda *i: topi.cast(the_result_temp(*i),
                                                                               "int32"),
                                            name='tensor_result_ub_cast')
    else:
        tensor_result_ub_cast = the_result_temp

    the_result = tvm.compute(output_shape, lambda *i: tensor_result_ub_cast(*i), name='the_result')

    return tensor_temp_a, tensor_temp_b, tensor_result_ub_cast,\
           tensor_temp_bias, the_result_mul_ub, the_result_ub, the_result_bais_ub, the_result


def _matmul_new_mk_nk_cce(tensor_a, tensor_b, tensor_bais, src_type):
    """
    algorithm: Matrix multiplication matmul_vector for MK x NK Situation
    ----------
    """
    shape_a = (tensor_a.shape[0].value, tensor_a.shape[1].value)
    shape_b = (tensor_b.shape[0].value, tensor_b.shape[1].value)
    output_bais = (shape_a[0], shape_b[0])

    tensor_a_ub = tvm.compute(shape_a, lambda *i: tensor_a(*i), name='tensor_a_ub')
    tensor_b_ub = tvm.compute(shape_b, lambda *i: tensor_b(*i), name='tensor_b_ub')

    if tensor_bais is not None:
        tensor_bais_ub = tvm.compute(output_bais, \
                                     lambda m, n: tensor_bais(n), name="tensor_bais_ub")
    else:
        tensor_bais_ub = None

    tensor_temp_a, tensor_temp_b, tensor_result_ub_cast, tensor_temp_bias, the_result_mul_ub, \
    the_result_ub, the_result_bais_ub, the_result = _compute_for_mk_nk(tensor_a_ub, tensor_b_ub,
                                                                       tensor_a.shape,
                                                                       tensor_b.shape,
                                                                       tensor_bais_ub, src_type)

    schedule_shape = (tensor_a.shape[0].value, tensor_b.shape[0].value, tensor_a.shape[1].value)
    schedule = _get_schedule_mk_nk(schedule_shape, \
                              (the_result, tensor_a_ub, tensor_b_ub, the_result_mul_ub,
                               the_result_ub, \
                               tensor_bais_ub, the_result_bais_ub,
                               tensor_temp_a, tensor_temp_b,
                               tensor_result_ub_cast,
                               tensor_temp_bias), src_type)

    schedule[tensor_a_ub].double_buffer()
    schedule[tensor_b_ub].double_buffer()

    return schedule, the_result


def _tranpose_schedule(schedule, data, data_ub, res, shape_res, dtype):
    """
    Schedule permutes the dimensions and the last axis is not transposed

    Parameters
    ----------
    """
    sch = schedule
    sch[data_ub].set_scope(cce.scope_ubuf)

    split_axis, split_factor = _tilling_axis_not_last(shape_res, dtype)
    axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis], factor=split_factor)
    sch[data_ub].compute_at(sch[res], axis_outer)
    sch[data_ub].emit_insn(data_ub.op.axis[split_axis], insn_cmd.DMA_COPY)
    sch = _do_storage_align(sch, data_ub, shape_res, dtype)
    sch[res].emit_insn(axis_inner, insn_cmd.DMA_COPY)
    tensor_list = [data, res]

    return sch, tensor_list


def _tranpose_notchange_last(data, shape_res, perm, dtype):
    """
    permutes the dimensions and the last axis is not transposed

    Parameters
    ----------
    """

    def _perm_to_flag(perm):
        """
        get the flag for permutation according to perm

        """
        flag = [i for i in perm]
        for i, item in enumerate(perm):
            flag[item] = i

        return flag

    def _permute(*index):
        """
        function of permute the dimensions of data

        """
        for i, item in enumerate(_perm_to_flag(perm)):
            if i == 0:
                res_axis = (index[item],)
            else:
                res_axis = res_axis + (index[item],)

        return res_axis
    ub_name = ["data_ub_1", "data_ub_2"]
    res_name = ["res_1", "res_2"]
    if dtype == "1":
        data_ub = tvm.compute(shape_res, lambda *index: data(*_permute(*index)), name=ub_name[0])
        res = tvm.compute(shape_res, lambda *index: data_ub(*index), name=res_name[0])
    else:
        data_ub = tvm.compute(shape_res, lambda *index: data(*_permute(*index)), name=ub_name[1])
        res = tvm.compute(shape_res, lambda *index: data_ub(*index), name=res_name[1])
    return res, data_ub


def _matmul_new_km_nk_cce(tensor_a_pre, tensor_b_pre, tensor_bais, src_type):
    """
    algorithm: Matrix multiplication matmul_vector for KM x NK Situation
    ----------
    """
    # transpose A
    tensor_a, data_ub_a = _tranpose_notchange_last(tensor_a_pre, (tensor_a_pre.shape[1].value,
                                                                  tensor_a_pre.shape[0].value, 1),
                                                   [1, 0, 2], "1")
    # transpose B
    tensor_b, data_ub_b = _tranpose_notchange_last(tensor_b_pre, (tensor_b_pre.shape[1].value,
                                                                  tensor_b_pre.shape[0].value, 1),
                                                   [1, 0, 2], src_type)
    # create schedule
    shape_a = [tensor_a.shape[0].value, tensor_a.shape[1].value]
    shape_b = [tensor_b.shape[0].value, tensor_b.shape[1].value]

    output_bais = (tensor_a.shape[0].value, tensor_b.shape[1].value, tensor_a.shape[1].value)

    tensor_a_ub = tvm.compute(shape_a, lambda m, j: tensor_a(m, j, 0), name='tensor_a_ub')
    tensor_b_ub = tvm.compute(shape_b, lambda m, j: tensor_b(m, j, 0), name='tensor_b_ub')

    if tensor_bais is not None:
        tensor_bais_ub = tvm.compute(output_bais,
                                     lambda m, n: tensor_bais(n), name="tensor_bais_ub")
    else:
        tensor_bais_ub = None

    tensor_temp_a, tensor_temp_b, tensor_result_ub_cast, tensor_temp_bias, the_result_mul_ub, \
    the_result_ub, the_result_bais_ub, the_result = _compute_for_mk_kn(tensor_a_ub, tensor_b_ub,
                                                                       shape_a, shape_b,
                                                                       tensor_bais_ub, src_type)
    schedule = _get_schedule_mk_kn((shape_a[0], shape_b[1], shape_a[1]), \
                                   (the_result, tensor_a_ub, tensor_b_ub,
                                    the_result_mul_ub, the_result_ub, \
                                    tensor_bais_ub, the_result_bais_ub,
                                    tensor_temp_a, tensor_temp_b,
                                    tensor_result_ub_cast,
                                    tensor_temp_bias,), src_type, False)

    shape_a.append(1)
    shape_b.append(1)

    _tranpose_schedule(schedule, tensor_a_pre, data_ub_a, tensor_a, shape_a, src_type)
    _tranpose_schedule(schedule, tensor_b_pre, data_ub_b, tensor_b, shape_b, src_type)

    return schedule, the_result, tensor_a, tensor_b


# pylint: disable=locally-disabled,too-many-arguments
def matmul_vector_cce(shape_a, shape_b, src_type, trans_a, trans_b,
                      shape_bias, kernel_name="matmul_vector"):
    """
    algorithm: matmul_vector
    calculating  matrix multiplication with bias, use vector mode ,C = A*B + bias

    Parameters
    ----------
    shape_a : list or tuple
        shape of tensor_a
    shape_b : list or tuple
        shape of tensor_b
    src_type : str
        the data type, assume src_dtype equals dst_dtype,
        only support float32
    trans_a : bool
        if the tensor A need transport, the value == True
    trans_b : bool
        if the tensor B need transport, the value == True
    shape_bias : list or tuple
        the shape of tensor_bias
    kernel_name : str
        cce kernel name, default value == "matmul_vector"

    Returns
    -------
    None
    """
    tensor_bias = None
    trans_flag = False

    if trans_b and trans_a:
        trans_flag = True

    if trans_flag:
        shape_a = list(shape_a)
        shape_b = list(shape_b)

        shape_a.append(1)
        shape_b.append(1)

        tensor_a = tvm.placeholder(shape_a, name='tensor_a', dtype=src_type)
        tensor_b = tvm.placeholder(shape_b, name='tensor_b', dtype=src_type)
    else:
        tensor_a = tvm.placeholder(shape_a, name='tensor_a', dtype=src_type)
        tensor_b = tvm.placeholder(shape_b, name='tensor_b', dtype=src_type)

    size_bias = len(shape_bias)
    size_bias_limit = 0

    if size_bias > size_bias_limit:
        tensor_bias = tvm.placeholder(shape_bias, name='tensor_bias', dtype=src_type)

    tensor_a_gm = None
    tensor_b_gm = None

    if trans_a:
        if trans_b:
            trans_flag = True
            schedule, the_result, tensor_a_gm, \
            tensor_b_gm = _matmul_new_km_nk_cce(tensor_a, tensor_b,
                                                tensor_bias, src_type)
        else:
            schedule, the_result = _matmul_new_km_kn_cce(tensor_a, tensor_b, tensor_bias, src_type)
    elif trans_b:
        schedule, the_result = _matmul_new_mk_nk_cce(tensor_a, tensor_b, tensor_bias, src_type)
    else:
        schedule, the_result = _matmul_new_mk_kn_cce(tensor_a, tensor_b, tensor_bias, src_type)

    if tensor_bias is not None:
        schedule[tensor_bias].double_buffer()

    if tensor_bias is not None:
        build_list = [tensor_a, tensor_b, tensor_bias, the_result]
    else:
        build_list = [tensor_a, tensor_b, the_result]

    if trans_flag:
        build_list = [tensor_a, tensor_b, the_result, tensor_a_gm, tensor_b_gm]

    with build_config:
        tvm.lower(schedule, build_list, simple_mode=True)
    with build_config:
        tvm.build(schedule, build_list, "cce", name=kernel_name)
    if trans_flag:
        wk_size_a = shape_a[0] * shape_a[1] * 4
        wk_size_b = shape_b[0] * shape_b[1] * 4
        workspace_dict = {"workspace": {"num": 2, "size": [wk_size_a, wk_size_b]}}
        _write_code(workspace_dict, "kernel_meta/" + kernel_name + ".json")
